"""캡션 저장 폴더(outDir) 규칙 — core.caption_out_dir.

수동 저장은 폴더를 만들지 않고 영어 오류('not a directory: [path]')로 실패하는데, 일괄 처리는
같은 outDir 로 폴더 트리를 새로 만들었다. 이제 세 경로(불러오기·수동 저장·일괄)가 같은
규칙을 쓴다: 이미 있는 폴더만, 없으면 다시 고르라는 한국어 오류.
"""
from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path

from core.caption_out_dir import (
    APPROVED_PREFS_KEY,
    MANIFEST_TARGET_MESSAGE,
    MAX_APPROVED,
    MISSING_MESSAGE,
    NOT_A_FOLDER_MESSAGE,
    NOT_APPROVED_MESSAGE,
    OUT_DIR_PREFS_KEY,
    PROTECTED_TARGET_MESSAGE,
    REFUSED_MESSAGE,
    CaptionOutDirError,
    filter_client_caption_prefs,
    is_approved,
    is_manifest_caption_target,
    is_protected_caption_target,
    normalize_approved,
    remember_approved,
    resolve_caption_out_dir,
)


class ResolveCaptionOutDirTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.root = Path(self._tmp.name)

    def test_blank_means_next_to_the_image(self):
        for raw in ("", "   ", None):
            self.assertEqual(resolve_caption_out_dir(raw), "")

    def test_existing_folder_is_resolved(self):
        folder = self.root / "captions"
        folder.mkdir()
        self.assertEqual(resolve_caption_out_dir(f"  {folder}  "), str(folder.resolve()))
        self.assertEqual(resolve_caption_out_dir(folder.as_uri()), str(folder.resolve()))

    def test_missing_folder_is_never_created(self):
        missing = self.root / "deleted" / "captions"
        with self.assertRaises(CaptionOutDirError) as ctx:
            resolve_caption_out_dir(str(missing))
        self.assertEqual(str(ctx.exception), MISSING_MESSAGE)
        self.assertFalse((self.root / "deleted").exists())

    def test_file_in_place_of_the_folder(self):
        not_folder = self.root / "captions.txt"
        not_folder.write_text("x", encoding="utf-8")
        with self.assertRaises(CaptionOutDirError) as ctx:
            resolve_caption_out_dir(str(not_folder))
        self.assertEqual(str(ctx.exception), NOT_A_FOLDER_MESSAGE)

    def test_system_folders_and_drive_roots_are_refused(self):
        anchor = Path(self._tmp.name).resolve().anchor or "/"
        raws = [anchor]
        if os.name == "nt":
            raws.append(os.environ.get("SystemRoot") or r"C:\Windows")
            raws.append("\\\\?\\" + (os.environ.get("SystemRoot") or r"C:\Windows") + r"\Temp")
        for raw in raws:
            with self.subTest(raw=raw):
                with self.assertRaises(CaptionOutDirError) as ctx:
                    resolve_caption_out_dir(raw)
                self.assertEqual(str(ctx.exception), REFUSED_MESSAGE)

    def test_messages_are_user_facing_and_path_free(self):
        for message in (MISSING_MESSAGE, NOT_A_FOLDER_MESSAGE, REFUSED_MESSAGE,
                        NOT_APPROVED_MESSAGE, PROTECTED_TARGET_MESSAGE, MANIFEST_TARGET_MESSAGE):
            self.assertNotIn(":\\", message)
            self.assertNotIn("\u2014", message)   # cp949 콘솔·로그에서 깨지는 문자
        self.assertTrue(issubclass(CaptionOutDirError, ValueError))


class ApprovedCaptionOutDirTests(unittest.TestCase):
    """outDir 는 클라이언트 값 — 호스트 대화상자가 승인한 폴더만 받는다(Codex R3 #0)."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.root = Path(self._tmp.name)
        self.picked = self.root / "Picked"
        self.picked.mkdir()
        self.other = self.root / "other"
        self.other.mkdir()

    def test_approved_folder_matches_through_case_slashes_and_file_scheme(self):
        approved = [str(self.picked.resolve())]
        spellings = [str(self.picked), self.picked.as_posix(), self.picked.as_uri(),
                     str(self.picked) + os.sep, str(self.root / "x" / ".." / "Picked")]
        if os.name == "nt":
            spellings.append(str(self.picked).upper())
        for raw in spellings:
            with self.subTest(raw=raw):
                resolved = resolve_caption_out_dir(raw, approved=approved)
                self.assertEqual(os.path.normcase(resolved),
                                 os.path.normcase(str(self.picked.resolve())))

    def test_existing_but_unapproved_folder_is_refused(self):
        for approved in ([], [str(self.picked)]):
            with self.subTest(approved=approved):
                with self.assertRaises(CaptionOutDirError) as ctx:
                    resolve_caption_out_dir(str(self.other), approved=approved)
                self.assertEqual(str(ctx.exception), NOT_APPROVED_MESSAGE)

    def test_blank_is_always_allowed_and_an_approved_folder_is_still_checked(self):
        self.assertEqual(resolve_caption_out_dir("", approved=[]), "")
        missing = self.root / "gone"
        with self.assertRaises(CaptionOutDirError) as ctx:
            resolve_caption_out_dir(str(missing), approved=[str(missing)])
        self.assertEqual(str(ctx.exception), MISSING_MESSAGE)
        not_folder = self.root / "file.txt"
        not_folder.write_text("x", encoding="utf-8")
        with self.assertRaises(CaptionOutDirError) as ctx:
            resolve_caption_out_dir(str(not_folder), approved=[str(not_folder)])
        self.assertEqual(str(ctx.exception), NOT_A_FOLDER_MESSAGE)

    def test_unapproved_path_gets_one_answer_whatever_is_there(self):
        """Codex R3 재검토 — 승인 검사가 존재·종류 검사보다 먼저다(경로 존재 오라클 차단).

        예전엔 승인 안 된 outDir 에도 없으면 MISSING, 파일이면 NOT_A_FOLDER, 있는 폴더면 NOT_APPROVED 로
        답이 갈려, 웹 클라이언트가 loadCaption/saveCaption 으로 호스트 경로의 존재·종류를 알아냈다.
        """
        not_folder = self.root / "secret.txt"
        not_folder.write_text("x", encoding="utf-8")
        raws = [self.other, self.root / "no" / "such", not_folder, self.other.as_uri()]
        raws.append(Path(self._tmp.name).resolve().anchor or "/")   # 드라이브 루트
        if os.name == "nt":
            raws.append(os.environ.get("SystemRoot") or r"C:\Windows")
        for approved in ([], [str(self.picked)]):
            for raw in raws:
                with self.subTest(approved=approved, raw=str(raw)):
                    with self.assertRaises(CaptionOutDirError) as ctx:
                        resolve_caption_out_dir(str(raw), approved=approved)
                    self.assertEqual(str(ctx.exception), NOT_APPROVED_MESSAGE)
        self.assertFalse((self.root / "no").exists())

    def test_remember_is_most_recent_first_deduplicated_and_capped(self):
        approved = remember_approved([], str(self.other))
        approved = remember_approved(approved, str(self.picked))
        approved = remember_approved(approved, self.other.as_posix())   # 같은 폴더 다른 표기
        self.assertEqual(approved, [self.other.as_posix(), str(self.picked)])
        many = [str(self.root / f"d{i}") for i in range(MAX_APPROVED + 5)]
        capped = remember_approved(many, str(self.picked))
        self.assertEqual(len(capped), MAX_APPROVED)
        self.assertEqual(capped[0], str(self.picked))

    def test_normalize_drops_garbage(self):
        self.assertEqual(normalize_approved("C:/x"), [])
        self.assertEqual(normalize_approved(None), [])
        self.assertEqual(normalize_approved([1, None, "", "  ", str(self.picked), str(self.picked)]),
                         [str(self.picked)])
        self.assertFalse(is_approved("", [str(self.picked)]))

    def test_client_prefs_cannot_mint_approvals_or_plant_an_unapproved_folder(self):
        approved = [str(self.picked)]
        payload = {
            APPROVED_PREFS_KEY: [str(self.other)],
            OUT_DIR_PREFS_KEY: str(self.other),
            "captionEngine": "caformer",
        }
        dropped = filter_client_caption_prefs(payload, approved)
        self.assertEqual(sorted(dropped), sorted([APPROVED_PREFS_KEY, OUT_DIR_PREFS_KEY]))
        self.assertEqual(payload, {"captionEngine": "caformer"})

        for value in ("", self.picked.as_posix()):
            with self.subTest(value=value):
                payload = {OUT_DIR_PREFS_KEY: value}
                self.assertEqual(filter_client_caption_prefs(payload, approved), [])
                self.assertEqual(payload, {OUT_DIR_PREFS_KEY: value})
        payload = {OUT_DIR_PREFS_KEY: ["not", "a", "string"]}
        self.assertEqual(filter_client_caption_prefs(payload, approved), [OUT_DIR_PREFS_KEY])
        self.assertEqual(filter_client_caption_prefs(None, approved), [])


class ProtectedCaptionTargetTests(unittest.TestCase):
    """웹 모드의 두 번째 벽 — 앱 설치 폴더 안 .txt(생성 이미지 폴더 제외)는 대상이 아니다."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.app = Path(self._tmp.name) / "Image viewer"
        self.output = self.app / "generated_images"
        (self.output / "a").mkdir(parents=True)
        (self.app / "config").mkdir()
        self.outside = Path(self._tmp.name) / "dataset"
        self.outside.mkdir()

    def protected(self, target) -> bool:
        return is_protected_caption_target(
            target, app_root=self.app, allowed_roots=(str(self.output),))

    def test_app_files_are_protected(self):
        for target in (self.app / "requirements.txt", self.app / "config" / "default_excludes.txt",
                       self.app / "venv" / "Lib" / "x.txt", self.app / "missing" / "deep.txt"):
            with self.subTest(target=target.name):
                self.assertTrue(self.protected(target))
        self.assertTrue(self.protected(str(self.app / "config" / ".." / "requirements.txt")))
        if os.name == "nt":
            self.assertTrue(self.protected(str(self.app / "REQUIREMENTS.TXT").upper()))

    def test_output_folder_and_outside_folders_are_allowed(self):
        self.assertFalse(self.protected(self.output / "a" / "b.txt"))
        self.assertFalse(self.protected(self.output.as_posix() + "/c.txt"))
        self.assertFalse(self.protected(self.outside / "img.txt"))
        # 이름만 앱 폴더로 시작하는 형제 폴더는 앱 폴더 안이 아니다
        sibling = Path(self._tmp.name) / "Image viewer2"
        self.assertFalse(self.protected(sibling / "requirements.txt"))

    def test_unresolvable_target_fails_closed(self):
        self.assertTrue(is_protected_caption_target("", app_root=self.app))


class ManifestCaptionTargetTests(unittest.TestCase):
    """웹 모드의 세 번째 벽(Codex R3 재검토 #0) — 설치·빌드 목록 이름의 .txt 는 어디서든 대상이 아니다."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.root = Path(self._tmp.name)

    def test_dependency_and_build_manifests_are_refused(self):
        names = ("requirements.txt", "requirements_versions.txt", "Requirements-Dev.txt",
                 "dev-requirements.txt", "constraints.txt", "pip_constraints.txt", "CMakeLists.txt",
                 "REQUIREMENTS.TXT")
        for name in names:
            with self.subTest(name=name):
                self.assertTrue(is_manifest_caption_target(self.root / "SomeApp" / name))
        self.assertTrue(is_manifest_caption_target((self.root / "requirements.txt").as_uri()))
        self.assertTrue(is_manifest_caption_target(str(self.root / "x" / ".." / "requirements.txt")))

    def test_ordinary_caption_names_are_allowed(self):
        for name in ("00001-1234.txt", "1girl solo.txt", "cmakelists_notes.txt", "요구사항.txt", "img.txt"):
            with self.subTest(name=name):
                self.assertFalse(is_manifest_caption_target(self.root / name))

    def test_unresolvable_target_fails_closed(self):
        self.assertTrue(is_manifest_caption_target(""))
        self.assertTrue(is_manifest_caption_target(None))

    @unittest.skipUnless(os.name == "nt", "8.3 짧은 이름은 윈도 전용")
    def test_short_8dot3_alias_of_a_manifest_is_refused(self):
        import ctypes

        target = self.root / "requirements.txt"
        target.write_text("numpy\n", encoding="utf-8")
        buffer = ctypes.create_unicode_buffer(1024)
        length = ctypes.windll.kernel32.GetShortPathNameW(str(target), buffer, len(buffer))
        short = buffer.value if length else ""
        if not short or Path(short).name.lower() == target.name.lower():
            self.skipTest("이 볼륨은 8.3 짧은 이름을 만들지 않는다")
        self.assertTrue(is_manifest_caption_target(short), short)


if __name__ == "__main__":
    unittest.main()
