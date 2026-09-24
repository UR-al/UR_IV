"""옛 썸네일 폴더(image_cache/thumbs) 은퇴 — core.legacy_thumb_cache.

실제 사용자 캐시 폴더는 건드리지 않는다: 모든 테스트가 임시 폴더만 쓴다.
"""
from __future__ import annotations

import hashlib
import os
import shutil
import tempfile
import threading
import unittest
from pathlib import Path
from unittest import mock

from PIL import Image

from core.cache_cleanup import (
    read_thumb_signature,
    shard_path,
    source_signature,
    thumb_is_stale,
    thumb_signature_comment,
)
from core.legacy_thumb_cache import (
    LegacyThumbRetirement,
    adopt_legacy_thumb,
    folders_overlap,
    is_current_format_thumb,
    legacy_thumb_candidates,
    retire_legacy_thumb_dir,
    start_legacy_thumb_retirement,
)
from core.thumb_cache import thumb_key, thumb_path


def _digest(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


class _TempCache(unittest.TestCase):
    def setUp(self):
        self.root = Path(tempfile.mkdtemp(prefix="legacy_thumbs_"))
        self.legacy = self.root / "image_cache" / "thumbs"
        self.new = self.root / "image_cache" / "thumbs_v2"
        self.legacy.mkdir(parents=True)
        self.source = self.root / "out" / "a.png"
        self.source.parent.mkdir()
        Image.new("RGB", (64, 48), (200, 10, 10)).save(self.source)

    def tearDown(self):
        shutil.rmtree(self.root, ignore_errors=True)

    def _signed(self, path: Path, source: Path | None = None) -> Path:
        """core.thumb_cache 가 쓰는 모양 — JPEG 주석에 원본 서명."""
        path.parent.mkdir(parents=True, exist_ok=True)
        signature = source_signature(str(source or self.source))
        Image.new("RGB", (8, 6), (1, 2, 3)).save(path, "JPEG", comment=thumb_signature_comment(signature))
        return path

    def _unsigned(self, path: Path) -> Path:
        """레거시 PyQt 갤러리(_create_thumbnail·gallery_worker)가 쓰던 모양 — 주석 없음."""
        path.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (8, 6), (9, 9, 9)).save(path, "JPEG")
        return path


class RetireLegacyThumbDirTests(_TempCache):
    def test_moves_current_thumbs_deletes_the_rest_and_removes_the_folder(self):
        key = thumb_key(str(self.source), 256)
        sharded = self._signed(Path(shard_path(str(self.legacy), key)))
        flat_key = thumb_key(str(self.source), 384)
        flat = self._signed(self.legacy / f"{flat_key}.jpg")   # 1f8198f13 이전 평면 저장분
        legacy_key = _digest(self.source.resolve().as_posix())
        legacy_sharded = self._unsigned(Path(shard_path(str(self.legacy), legacy_key)))
        legacy_flat = self._unsigned(self.legacy / f"{_digest('other')}.jpg")
        temp = self.legacy / "ab" / f"{_digest('x')}.jpg.123.456.tmp"
        temp.parent.mkdir(parents=True, exist_ok=True)
        temp.write_bytes(b"partial")
        junk = self.legacy / "notes.txt"
        junk.write_text("x", encoding="utf-8")

        result = retire_legacy_thumb_dir(str(self.legacy), str(self.new))

        self.assertEqual(result, LegacyThumbRetirement(moved=2, removed=4, failed=0, done=True))
        self.assertFalse(self.legacy.exists())
        moved_256 = Path(shard_path(str(self.new), key))
        moved_384 = Path(shard_path(str(self.new), flat_key))
        self.assertTrue(moved_256.is_file())
        self.assertTrue(moved_384.is_file())
        # 옮긴 썸네일은 지금 원본 기준으로 여전히 유효하다(다시 만들지 않는다)
        self.assertFalse(thumb_is_stale(str(self.source), str(moved_256)))
        self.assertEqual(read_thumb_signature(str(moved_256)), source_signature(str(self.source)))
        for gone in (sharded, flat, legacy_sharded, legacy_flat, temp, junk):
            self.assertFalse(gone.exists(), gone)
        # 레거시 키는 새 폴더로 오지 않는다
        self.assertFalse(Path(shard_path(str(self.new), legacy_key)).exists())

    def test_moved_thumb_of_a_changed_source_is_rebuilt_on_read(self):
        key = thumb_key(str(self.source), 256)
        self._signed(Path(shard_path(str(self.legacy), key)))
        Image.new("RGB", (80, 40), (0, 0, 255)).save(self.source)   # 원본이 그 뒤에 바뀌었다
        retire_legacy_thumb_dir(str(self.legacy), str(self.new))
        self.assertTrue(thumb_is_stale(str(self.source), shard_path(str(self.new), key)))

    def test_existing_new_thumb_wins(self):
        key = thumb_key(str(self.source), 256)
        self._signed(Path(shard_path(str(self.legacy), key)))
        target = Path(shard_path(str(self.new), key))
        target.parent.mkdir(parents=True)
        target.write_bytes(b"newer")
        result = retire_legacy_thumb_dir(str(self.legacy), str(self.new))
        self.assertEqual((result.moved, result.removed, result.done), (0, 1, True))
        self.assertEqual(target.read_bytes(), b"newer")

    def test_missing_legacy_folder_is_already_done(self):
        shutil.rmtree(self.legacy)
        self.assertEqual(retire_legacy_thumb_dir(str(self.legacy), str(self.new)),
                         LegacyThumbRetirement(done=True))
        self.assertFalse(self.new.exists())

    def test_refuses_overlapping_folders(self):
        inner = self.legacy / "v2"
        keep = self._unsigned(self.legacy / f"{_digest('k')}.jpg")
        for new_dir in (self.legacy, inner, self.legacy.parent):
            with self.subTest(new_dir=str(new_dir)):
                result = retire_legacy_thumb_dir(str(self.legacy), str(new_dir))
                self.assertFalse(result.done)
                self.assertTrue(keep.exists())
        self.assertFalse(retire_legacy_thumb_dir(str(self.legacy), "").done)
        self.assertTrue(keep.exists())

    def test_locked_files_are_left_for_the_next_run(self):
        legacy_file = self._unsigned(self.legacy / "cd" / f"cd{_digest('z')[2:]}.jpg")
        real_remove = os.remove

        def locked(path, *args, **kwargs):
            if os.path.normcase(str(path)) == os.path.normcase(str(legacy_file)):
                raise PermissionError(32, "locked")
            return real_remove(path, *args, **kwargs)

        with mock.patch("core.legacy_thumb_cache.os.remove", side_effect=locked):
            first = retire_legacy_thumb_dir(str(self.legacy), str(self.new))
        self.assertEqual((first.failed, first.done), (1, False))
        self.assertTrue(legacy_file.exists())
        second = retire_legacy_thumb_dir(str(self.legacy), str(self.new))
        self.assertEqual((second.removed, second.failed, second.done), (1, 0, True))
        self.assertFalse(self.legacy.exists())

    def test_never_follows_a_junction_out_of_the_cache(self):
        try:
            import _winapi
            create = _winapi.CreateJunction
        except (ImportError, AttributeError):
            self.skipTest("정션을 만들 수 없는 환경")
        outside = self.root / "precious"
        outside.mkdir()
        precious = outside / f"{_digest('p')}.jpg"
        self._unsigned(precious)
        link = self.legacy / "ef"
        try:
            create(str(outside), str(link))
        except OSError as exc:
            self.skipTest(f"정션 생성 실패: {exc}")
        try:
            result = retire_legacy_thumb_dir(str(self.legacy), str(self.new))
            self.assertTrue(precious.exists(), "정션 너머의 파일을 지웠다")
            self.assertFalse(result.done)
        finally:
            try:
                os.rmdir(link)   # 정션만 지운다(대상은 그대로)
            except OSError:
                pass
        self.assertTrue(precious.exists())

    def test_the_legacy_folder_itself_being_a_link_is_refused(self):
        try:
            import _winapi
            create = _winapi.CreateJunction
        except (ImportError, AttributeError):
            self.skipTest("정션을 만들 수 없는 환경")
        target = self.root / "real_thumbs"
        target.mkdir()
        keep = self._unsigned(target / f"{_digest('q')}.jpg")
        shutil.rmtree(self.legacy)
        try:
            create(str(target), str(self.legacy))
        except OSError as exc:
            self.skipTest(f"정션 생성 실패: {exc}")
        try:
            self.assertFalse(retire_legacy_thumb_dir(str(self.legacy), str(self.new)).done)
            self.assertTrue(keep.exists())
        finally:
            try:
                os.rmdir(self.legacy)
            except OSError:
                pass


class AdoptLegacyThumbTests(_TempCache):
    """조회가 새 폴더에서 빗나가면 그 키를 옛 폴더에서 먼저 옮겨 온다 — 배경 정리(시작 30초 뒤)를
    기다리면 첫 실행에 히스토리 전체를 다시 렌더하고, 원본을 지운 항목은 썸네일을 잃었다."""

    def _dest(self, width: int = 256) -> Path:
        return Path(thumb_path(str(self.new), str(self.source), width))

    def _legacy_for(self, dest: Path) -> Path:
        return Path(shard_path(str(self.legacy), dest.stem))

    def _adopt(self, dest: Path) -> bool:
        return adopt_legacy_thumb(str(self.legacy), str(dest), str(self.source))

    def test_signed_thumb_of_the_current_source_is_moved_as_is(self):
        dest = self._dest()
        legacy = self._signed(self._legacy_for(dest))
        payload = legacy.read_bytes()
        self.assertTrue(self._adopt(dest))
        self.assertFalse(legacy.exists())
        self.assertEqual(dest.read_bytes(), payload)
        self.assertFalse(thumb_is_stale(str(self.source), str(dest)))

    def test_flat_legacy_thumb_is_found_too(self):
        dest = self._dest(384)
        flat = self._signed(self.legacy / f"{dest.stem}.jpg")   # 1f8198f13 이전 평면 저장분
        self.assertEqual(legacy_thumb_candidates(str(self.legacy), dest.stem),
                         (shard_path(str(self.legacy), dest.stem), str(flat)))
        self.assertTrue(self._adopt(dest))
        self.assertFalse(flat.exists())
        self.assertTrue(dest.is_file())

    def test_thumb_of_a_deleted_source_is_adopted(self):
        dest = self._dest()
        self._signed(self._legacy_for(dest))
        self.source.unlink()
        self.assertTrue(self._adopt(dest))
        # 원본이 없으면 있는 썸네일을 그대로 쓴다 — 지운 히스토리 항목의 카드가 깨지지 않는다
        self.assertFalse(thumb_is_stale(str(self.source), str(dest)))

    def test_thumb_of_a_changed_source_is_left_for_the_sweep(self):
        dest = self._dest()
        legacy = self._signed(self._legacy_for(dest))
        Image.new("RGB", (80, 40), (0, 0, 255)).save(self.source)   # 원본이 그 뒤에 바뀌었다
        stat = os.stat(self.source)
        os.utime(self.source, ns=(stat.st_atime_ns, stat.st_mtime_ns + 5_000_000_000))
        self.assertFalse(self._adopt(dest))
        self.assertTrue(legacy.exists())   # 어차피 다시 만든다 — 옛 것은 배경 정리가 치운다
        self.assertFalse(dest.exists())

    def test_unsigned_legacy_file_with_the_same_name_is_not_adopted(self):
        dest = self._dest()
        legacy = self._unsigned(self._legacy_for(dest))
        self.assertFalse(self._adopt(dest))
        self.assertTrue(legacy.exists())
        self.assertFalse(dest.exists())

    def test_existing_new_thumb_is_never_replaced(self):
        dest = self._dest()
        legacy = self._signed(self._legacy_for(dest))
        dest.parent.mkdir(parents=True)
        dest.write_bytes(b"newer")
        self.assertFalse(self._adopt(dest))
        self.assertEqual(dest.read_bytes(), b"newer")
        self.assertTrue(legacy.exists())

    def test_nothing_to_do_without_a_legacy_folder(self):
        dest = self._dest()
        shutil.rmtree(self.legacy)
        self.assertFalse(self._adopt(dest))
        self.assertFalse(adopt_legacy_thumb(None, str(dest), str(self.source)))
        self.assertFalse(adopt_legacy_thumb("", str(dest), str(self.source)))
        self.assertFalse(self.new.exists())

    def test_non_cache_names_are_ignored(self):
        self._signed(self.legacy / "sh" / "short.jpg")
        self.assertFalse(adopt_legacy_thumb(str(self.legacy), str(self.new / "sh" / "short.jpg"), str(self.source)))

    def test_overlapping_folders_are_refused(self):
        key = thumb_key(str(self.source), 256)
        flat = self._signed(self.legacy / f"{key}.jpg")
        same_folder_dest = Path(shard_path(str(self.legacy), key))   # 새 캐시 = 옛 폴더(잘못된 설정)
        self.assertFalse(self._adopt(same_folder_dest))
        self.assertTrue(flat.exists())

    def test_locked_legacy_file_falls_back_to_a_miss(self):
        dest = self._dest()
        legacy = self._signed(self._legacy_for(dest))
        with mock.patch("core.legacy_thumb_cache.os.replace", side_effect=PermissionError(32, "locked")):
            self.assertFalse(self._adopt(dest))
        self.assertTrue(legacy.exists())
        self.assertFalse(dest.exists())

    def test_never_adopts_through_a_junction(self):
        try:
            import _winapi
            create = _winapi.CreateJunction
        except (ImportError, AttributeError):
            self.skipTest("정션을 만들 수 없는 환경")
        dest = self._dest()
        outside = self.root / "precious"
        precious = self._signed(outside / f"{dest.stem}.jpg")
        link = self.legacy / dest.stem[:2]
        try:
            create(str(outside), str(link))
        except OSError as exc:
            self.skipTest(f"정션 생성 실패: {exc}")
        try:
            self.assertFalse(self._adopt(dest))
            self.assertTrue(precious.exists(), "정션 너머의 파일을 캐시로 끌어왔다")
            self.assertFalse(dest.exists())
        finally:
            try:
                os.rmdir(link)   # 정션만 지운다(대상은 그대로)
            except OSError:
                pass

    def test_thumbnail_entrances_pass_the_legacy_folder(self):
        # 히스토리(generateThumbnails 프리페처)·카드(aithumb: 핸들러) 둘 다 옛 폴더를 넘긴다
        root = Path(__file__).resolve().parents[1]
        bridge = (root / "ui" / "vue_bridge.py").read_text(encoding="utf-8")
        setup = (root / "ui" / "generator_ui_setup.py").read_text(encoding="utf-8")
        self.assertIn("legacy_dir=LEGACY_THUMB_DIR", bridge)
        self.assertIn("legacy_dir=LEGACY_THUMB_DIR", setup)


class HelperTests(_TempCache):
    def test_current_format_needs_a_sha1_name_and_a_signature(self):
        key = thumb_key(str(self.source), 256)
        self.assertTrue(is_current_format_thumb(str(self._signed(self.legacy / f"{key}.jpg"))))
        self.assertTrue(is_current_format_thumb(str(self._signed(self.legacy / f"{key[:-1]}A.JPG"))))
        self.assertFalse(is_current_format_thumb(str(self._unsigned(self.legacy / f"{_digest('u')}.jpg"))))
        self.assertFalse(is_current_format_thumb(str(self._signed(self.legacy / "short.jpg"))))
        self.assertFalse(is_current_format_thumb(str(self.legacy / f"{_digest('missing')}.jpg")))

    def test_folders_overlap(self):
        self.assertTrue(folders_overlap(str(self.legacy), str(self.legacy)))
        self.assertTrue(folders_overlap(str(self.legacy), str(self.legacy / "x")))
        self.assertTrue(folders_overlap(str(self.legacy / "x"), str(self.legacy)))
        self.assertFalse(folders_overlap(str(self.legacy), str(self.new)))

    def test_config_uses_a_new_folder_that_does_not_overlap_the_legacy_one(self):
        import config
        self.assertNotEqual(os.path.normcase(config.THUMB_DIR), os.path.normcase(config.LEGACY_THUMB_DIR))
        self.assertEqual(os.path.basename(config.LEGACY_THUMB_DIR), "thumbs")
        self.assertFalse(folders_overlap(config.THUMB_DIR, config.LEGACY_THUMB_DIR))


class StartLegacyThumbRetirementTests(_TempCache):
    def test_runs_once_in_a_daemon_thread(self):
        self._unsigned(self.legacy / f"{_digest('a')}.jpg")
        finished = threading.Event()
        results = []

        def _done(result):
            results.append(result)
            finished.set()

        thread = start_legacy_thumb_retirement(str(self.legacy), str(self.new),
                                               delay_seconds=0, on_done=_done)
        self.assertIsNotNone(thread)
        self.assertTrue(thread.daemon)
        self.assertTrue(finished.wait(20))
        thread.join(5)
        self.assertTrue(results[0].done)
        self.assertFalse(self.legacy.exists())
        # 같은 프로세스에서 같은 폴더로는 다시 띄우지 않는다(다시 생겨도)
        self.legacy.mkdir()
        self.assertIsNone(start_legacy_thumb_retirement(str(self.legacy), str(self.new), delay_seconds=0))

    def test_no_thread_without_a_legacy_folder(self):
        shutil.rmtree(self.legacy)
        self.assertIsNone(start_legacy_thumb_retirement(str(self.legacy), str(self.new), delay_seconds=0))

    def test_main_window_schedules_the_retirement_with_config_paths(self):
        source = (Path(__file__).resolve().parents[1] / "ui" / "generator_main.py").read_text(encoding="utf-8")
        self.assertIn("start_legacy_thumb_retirement(LEGACY_THUMB_DIR, THUMB_DIR)", source)


if __name__ == "__main__":
    unittest.main()
