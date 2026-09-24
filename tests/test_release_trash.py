"""core.release_trash — 이전 release 는 이름만 바꾸고 백그라운드에서 지운다."""
from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path

from core.release_trash import (
    TRASH_PREFIX,
    is_trash,
    move_to_trash,
    purge_trash,
    purge_trash_async,
    trash_entries,
)


def _release(root: Path, name: str) -> Path:
    folder = root / name
    (folder / "venv" / "Lib").mkdir(parents=True)
    (folder / "venv" / "Lib" / "big.bin").write_bytes(b"x" * 1024)
    return folder


def make_dir_links(target: Path, link: Path) -> list[tuple[str, Path]]:
    """``link`` → ``target`` 디렉터리 링크를 만들 수 있는 방식마다 만들어 본다.

    반환: [(종류, 링크 경로)] — Windows 정션(관리자 권한 불필요)과 심볼릭 링크(개발자 모드가
    아니면 OSError 로 빠진다). 각 링크 이름은 ``link`` 에 종류 접미사를 붙인다.
    """
    made = []
    if os.name == "nt":
        import _winapi

        junction = link.with_name(f"{link.name}-junction")
        try:
            _winapi.CreateJunction(str(target), str(junction))
            made.append(("junction", junction))
        except OSError:
            pass
    symlink = link.with_name(f"{link.name}-symlink")
    try:
        os.symlink(target, symlink, target_is_directory=True)
        made.append(("symlink", symlink))
    except (OSError, NotImplementedError):
        pass
    return made


def remove_dir_link(link: Path) -> None:
    """링크 자체만 지운다(대상은 따라가지 않는다)."""
    if not os.path.lexists(link):
        return
    try:
        os.unlink(link)
    except OSError:
        os.rmdir(link)


class ReleaseTrashTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name) / "releases"
        self.root.mkdir()

    def test_move_is_a_rename_inside_the_same_releases_root(self):
        old = _release(self.root, "20260101-000000-aaaa-111111")
        moved = move_to_trash(old, self.root)
        self.assertIsNotNone(moved)
        self.assertFalse(old.exists())
        self.assertTrue(is_trash(moved))
        self.assertEqual(moved.parent, self.root.resolve())
        self.assertTrue((moved / "venv" / "Lib" / "big.bin").is_file())

    def test_purge_only_removes_trash_and_never_a_candidate_release(self):
        candidate = _release(self.root, "20260102-000000-bbbb-222222")   # 설치 중 후보
        moved = move_to_trash(_release(self.root, "20260101-000000-aaaa-111111"), self.root)
        self.assertEqual(trash_entries(self.root), [moved])
        self.assertEqual(purge_trash([self.root]), 1)
        self.assertFalse(moved.exists())
        self.assertTrue(candidate.is_dir())

    def test_outside_or_nested_paths_are_never_moved(self):
        outside = Path(self.temp.name) / "user-models"
        outside.mkdir()
        nested = _release(self.root, "keep") / "venv"
        self.assertIsNone(move_to_trash(outside, self.root))
        self.assertIsNone(move_to_trash(nested, self.root))
        self.assertIsNone(move_to_trash(self.root, self.root))
        self.assertTrue(outside.is_dir() and nested.is_dir())

    def test_already_trashed_folder_is_not_renamed_again(self):
        moved = move_to_trash(_release(self.root, "old"), self.root)
        self.assertIsNone(move_to_trash(moved, self.root))
        self.assertTrue(moved.name.startswith(TRASH_PREFIX))

    def test_async_purge_runs_off_the_caller_and_is_skipped_when_nothing_to_do(self):
        self.assertIsNone(purge_trash_async([self.root, Path(self.temp.name) / "missing"]))
        moved = move_to_trash(_release(self.root, "old"), self.root)
        thread = purge_trash_async([self.root])
        self.assertIsNotNone(thread)
        self.assertTrue(thread.daemon)
        thread.join(10)
        self.assertFalse(moved.exists())

    def _links(self, target: Path, name: str) -> list[tuple[str, Path]]:
        links = make_dir_links(target, self.root / name)
        for _kind, link in links:
            self.addCleanup(remove_dir_link, link)   # LIFO — temp 정리보다 먼저 링크만 지운다
        if not links:
            self.skipTest("디렉터리 링크를 만들 수 없는 환경")
        return links

    def test_trash_link_to_active_release_is_never_followed(self):
        """예전: ``.trash-*`` 링크를 resolve 한 대상(활성 release)을 rmtree 했다."""
        active = _release(self.root, "20260101-000000-aaaa-111111")
        big = active / "venv" / "Lib" / "big.bin"
        links = self._links(active, ".trash-old-deadbeef")
        self.assertEqual(trash_entries(self.root), [])
        self.assertEqual(purge_trash([self.root]), 0)
        self.assertIsNone(purge_trash_async([self.root]))
        self.assertTrue(big.is_file(), "활성 release 는 그대로")
        for kind, link in links:
            with self.subTest(kind=kind):
                self.assertTrue(os.path.lexists(link), "링크도 따라가지 않고 남겨 둔다")
                self.assertIsNone(move_to_trash(link, self.root), "링크를 넘겨도 대상을 옮기지 않는다")
        self.assertTrue(big.is_file())
        self.assertEqual(sorted(p.name for p in self.root.iterdir() if not p.name.startswith(".")),
                         [active.name])

    def test_trash_link_to_real_trash_only_purges_the_real_folder_once(self):
        real = move_to_trash(_release(self.root, "old"), self.root)
        self._links(real, ".trash-alias")
        self.assertEqual(trash_entries(self.root), [real])
        self.assertEqual(purge_trash([self.root]), 1)
        self.assertFalse(real.exists())


if __name__ == "__main__":
    unittest.main()
