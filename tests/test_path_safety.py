"""core.path_safety 시스템 폴더 차단 회귀 테스트.

예전 _is_forbidden 은 resolve 된 문자열의 startswith 비교라서
``\\?\C:\Windows``·관리 공유(``\\localhost\C$``) 표기로 우회되고,
``C:\WindowsImages`` 같은 정상 폴더는 오탐으로 막혔다.
"""
from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path

from core.path_safety import (
    UnsafePathError,
    _normalize_windows_for_check,
    is_forbidden_path,
    missing_input_path,
    safe_input_path,
    safe_output_dir,
    strip_file_url,
)

BS = "\\"
# 테스트는 환경변수에 기대지 않는다 — 시스템 드라이브가 C: 인 평범한 PC.
_ENV_C = {
    "SystemRoot": r"C:\Windows",
    "windir": r"C:\Windows",
    "ProgramFiles": r"C:\Program Files",
    "ProgramFiles(x86)": r"C:\Program Files (x86)",
    "ProgramData": r"C:\ProgramData",
    "SystemDrive": "C:",
}


def _win(path: str, environ=None) -> bool:
    return is_forbidden_path(path, windows=True, environ=_ENV_C if environ is None else environ)


class WindowsForbiddenRootTests(unittest.TestCase):
    def test_plain_system_paths_are_blocked(self):
        for path in (
            r"C:\Windows",
            r"C:\Windows\win.ini",
            r"c:\windows\system32\drivers\etc\hosts",
            r"C:\Program Files\app\x.png",
            r"C:\Program Files (x86)\app\x.png",
            r"C:\ProgramData\x\y.png",
        ):
            with self.subTest(path=path):
                self.assertTrue(_win(path))

    def test_extended_length_and_device_prefixes_do_not_bypass(self):
        for path in (
            BS * 2 + r"?\C:\Windows\win.ini",
            BS * 2 + r"?\c:\WINDOWS\Temp\new",
            "//?/C:/Windows/win.ini",
            BS * 2 + r".\C:\Windows\win.ini",
            BS * 2 + r"?\UNC\localhost\C$\Windows\win.ini",
            BS * 2 + r".\UNC\127.0.0.1\c$\Program Files\x.png",
        ):
            with self.subTest(path=path):
                self.assertTrue(_win(path))

    def test_admin_shares_map_back_to_drive_roots(self):
        for path in (
            BS * 2 + r"localhost\C$\Windows\win.ini",
            BS * 2 + r"127.0.0.1\c$\Windows\System32\x.png",
            BS * 2 + r"[::1]\C$\ProgramData\x.png",
            BS * 2 + r"MY-PC\C$\Program Files\x.png",
            "//localhost/C$/Windows/win.ini",
            BS * 2 + r"localhost\C$\Users\..\Windows\win.ini",
        ):
            with self.subTest(path=path):
                self.assertTrue(_win(path))

    def test_system_shares_and_unknown_device_namespaces_fail_closed(self):
        for path in (
            BS * 2 + r"localhost\ADMIN$\win.ini",
            BS * 2 + r"localhost\admin$",
            BS * 2 + r"localhost\PRINT$\x64\x.dll",
            BS * 2 + r"localhost\IPC$",
            BS * 2 + r"?\GLOBALROOT\Device\HarddiskVolume3\Windows\win.ini",
            BS * 2 + r"?\Volume{3a62c904-f12f-414d-8239-36919bbc8cba}\Windows\win.ini",
            BS * 2 + r".\PhysicalDrive0",
            BS * 2 + r".\pipe\x",
        ):
            with self.subTest(path=path):
                self.assertTrue(_win(path))

    def test_user_paths_nas_and_prefix_lookalikes_stay_allowed(self):
        for path in (
            r"C:\Users\me\Pictures\a.png",
            r"D:\images\a.png",
            r"C:\WindowsImages\a.png",
            r"C:\Windows_backup\a.png",
            r"C:\Program Files Custom\a.png",
            r"C:\ProgramDataset\a.png",
            BS * 2 + r"nas\share\gallery\a.png",
            BS * 2 + r"?\UNC\nas\share\gallery\a.png",
            BS * 2 + r"pc2\D$\images\a.png",
            BS * 2 + r"nas\photos$\a.png",
            BS * 2 + r"?\D:\images\a.png",
            r"C:\a.png",
        ):
            with self.subTest(path=path):
                self.assertFalse(_win(path))

    def test_roots_follow_environment_not_hardcoded_c_drive(self):
        env = {"SystemRoot": r"D:\WINNT", "ProgramFiles": r"D:\Apps", "SystemDrive": "D:"}
        self.assertTrue(_win(r"D:\WINNT\system32\x.png", env))
        self.assertTrue(_win(r"D:\Apps\tool\x.png", env))
        self.assertTrue(_win(r"D:\Program Files\x.png", env))
        self.assertTrue(_win(BS * 2 + r"localhost\D$\WINNT\x.png", env))
        # C: 바닥값은 환경변수가 달라도 남는다.
        self.assertTrue(_win(r"C:\Windows\x.png", env))
        self.assertFalse(_win(r"D:\images\x.png", env))

    def test_bogus_environment_values_never_block_whole_drives(self):
        env = {"SystemRoot": "C:\\", "ProgramFiles": "relative\\dir", "ProgramData": "", "SystemDrive": "nonsense"}
        self.assertFalse(_win(r"C:\Users\me\a.png", env))
        self.assertFalse(_win(r"E:\a.png", env))

    def test_normalizer_maps_forms_to_one_spelling(self):
        self.assertEqual(_normalize_windows_for_check(BS * 2 + r"?\C:\Windows\x"), r"C:\Windows\x")
        self.assertEqual(
            _normalize_windows_for_check(BS * 2 + r"?\UNC\srv\share\a.png"), BS * 2 + r"srv\share\a.png"
        )
        self.assertEqual(_normalize_windows_for_check(BS * 2 + r"host\c$\Windows"), r"C:\Windows")
        self.assertEqual(_normalize_windows_for_check(BS * 2 + r"host\c$"), "C:" + BS)
        self.assertIsNone(_normalize_windows_for_check(BS * 2 + r"?\GLOBALROOT\Device\X"))


class PosixForbiddenRootTests(unittest.TestCase):
    def test_posix_roots_compare_by_component(self):
        self.assertTrue(is_forbidden_path("/etc/passwd", windows=False))
        self.assertTrue(is_forbidden_path("/usr", windows=False))
        self.assertFalse(is_forbidden_path("/etcetera/a.png", windows=False))
        self.assertFalse(is_forbidden_path("/home/me/a.png", windows=False))


@unittest.skipUnless(os.name == "nt", "Windows 경로 표기 전용")
class WindowsResolveIntegrationTests(unittest.TestCase):
    """실제 resolve() 를 거친 뒤에도 우회가 막히는지 — 공개 API 기준."""

    def _system_root(self) -> str:
        return os.environ.get("SystemRoot") or r"C:\Windows"

    def test_output_dir_rejects_extended_prefix_without_creating(self):
        target = BS * 2 + "?" + BS + self._system_root() + r"\Temp\aistudio_path_safety_probe"
        with self.assertRaises(UnsafePathError) as ctx:
            safe_output_dir(target, create=True)
        self.assertIn("forbidden", str(ctx.exception))
        self.assertFalse(os.path.exists(self._system_root() + r"\Temp\aistudio_path_safety_probe"))

    def test_output_dir_rejects_admin_share_and_url_encoded_forms(self):
        drive = self._system_root()[0]
        for raw in (
            BS * 2 + f"localhost{BS}{drive}${self._system_root()[2:]}{BS}Temp",
            "file:///%5C%5C%3F%5C" + self._system_root().replace(BS, "%5C") + "%5CTemp",
        ):
            with self.subTest(raw=raw):
                with self.assertRaises(UnsafePathError) as ctx:
                    safe_output_dir(raw, create=False)
                self.assertIn("forbidden", str(ctx.exception))

    def test_input_path_blocks_extended_prefix_to_system_image(self):
        # 존재 여부와 무관하게 먼저 루트 검사에서 걸려야 한다.
        raw = BS * 2 + "?" + BS + self._system_root() + r"\Web\Wallpaper\Windows\img0.jpg"
        self.assertIsNone(safe_input_path(raw))

    def test_regular_image_under_temp_is_still_allowed(self):
        with tempfile.TemporaryDirectory() as tmp:
            image = Path(tmp) / "WindowsImages" / "a.png"
            image.parent.mkdir()
            image.write_bytes(b"\x89PNG\r\n\x1a\n")
            self.assertEqual(safe_input_path(str(image)), str(image.resolve()))
            self.assertEqual(safe_input_path(BS * 2 + "?" + BS + str(image)), BS * 2 + "?" + BS + str(image.resolve()))


class MissingInputPathTests(unittest.TestCase):
    """'허용된 경로인데 파일이 없음' 만 참 — 거부 사유(시스템 폴더·확장자)는 거짓."""

    def test_only_an_allowed_but_absent_file_counts_as_missing(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            present = root / "a.png"
            present.write_bytes(b"png")
            (root / "folder.png").mkdir()
            self.assertTrue(missing_input_path(str(root / "gone.png")))
            self.assertTrue(missing_input_path((root / "gone.png").as_uri()))
            self.assertFalse(missing_input_path(str(present)))            # 있으면 '없음' 이 아니다
            self.assertFalse(missing_input_path(str(root / "folder.png")))  # 폴더가 그 자리에 있음
            self.assertFalse(missing_input_path(str(root / "gone.txt")))    # 막힌 확장자
            self.assertTrue(missing_input_path(str(root / "gone.txt"), allowed_exts=None))
            self.assertTrue(missing_input_path(str(root / "gone.mp4"), allowed_exts=frozenset({".mp4"})))
            self.assertFalse(missing_input_path(""))

    @unittest.skipUnless(os.name == "nt", "Windows 시스템 폴더 규칙")
    def test_system_folders_are_refused_not_missing(self):
        system_root = os.environ.get("SystemRoot") or r"C:\Windows"
        self.assertFalse(missing_input_path(system_root + r"\no_such_dir\x.png"))
        self.assertFalse(missing_input_path(BS * 2 + "?" + BS + system_root + r"\no_such_dir\x.png"))


class LiteralPercentPathTests(unittest.TestCase):
    """원시 경로의 ``%`` 는 이름의 일부다 — 퍼센트 디코딩은 file URL 에만.

    예전에는 모든 입력을 unquote 해서 ``a%20b.png`` 가 ``a b.png`` 로 바뀌었다: 그 파일의 썸네일·메타데이터·
    즐겨찾기·편집 저장·삭제가 실패했고, ``a b.png`` 형제가 있으면 그 파일을 대상으로 삼았다(휴지통 포함).
    """

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self.pct = self.root / "a%20b.png"
        self.sibling = self.root / "a b.png"
        self.pct.write_bytes(b"\x89PNG\r\n\x1a\npct")
        self.sibling.write_bytes(b"\x89PNG\r\n\x1a\nsibling")

    def tearDown(self):
        self._tmp.cleanup()

    def test_strip_file_url_decodes_only_file_urls(self):
        self.assertEqual(strip_file_url("C:/dl/a%20b.png"), "C:/dl/a%20b.png")
        self.assertEqual(strip_file_url("C:/x/100%.png"), "C:/x/100%.png")
        self.assertEqual(strip_file_url("file:///C:/dl/a%20b.png"), "C:/dl/a b.png")
        self.assertEqual(strip_file_url("FILE:///C:/dl/a%2520b.png"), "C:/dl/a%20b.png")
        self.assertEqual(strip_file_url("file://server/share/a%20b.png"), "server/share/a b.png")
        self.assertEqual(strip_file_url("file:///C:/x/100%.png"), "C:/x/100%.png")  # 깨진 시퀀스는 그대로
        self.assertEqual(strip_file_url(""), "")

    def test_raw_path_with_percent_is_the_percent_file_not_its_sibling(self):
        self.assertEqual(safe_input_path(str(self.pct)), str(self.pct.resolve()))
        self.assertEqual(safe_input_path(self.pct.as_posix()), str(self.pct.resolve()))

    def test_file_url_is_decoded_exactly_once(self):
        self.assertEqual(safe_input_path(self.pct.as_uri()), str(self.pct.resolve()))  # as_uri: % → %25
        self.assertEqual(safe_input_path("file:///" + self.pct.as_posix()), str(self.sibling.resolve()))
        self.assertEqual(safe_input_path(self.sibling.as_uri()), str(self.sibling.resolve()))

    def test_missing_checks_use_the_literal_name_too(self):
        gone = self.root / "gone%20x.png"
        (self.root / "gone x.png").write_bytes(b"png")      # 디코드된 이름은 있다
        self.assertTrue(missing_input_path(str(gone)))      # 예전: 형제가 있어 '없음' 아님 → 목록 정리 실패
        self.pct.unlink()
        self.assertTrue(missing_input_path(str(self.pct)))  # 예전: 'a b.png' 가 있어 False
        self.assertFalse(missing_input_path(str(self.sibling)))

    def test_generator_main_clean_path_uses_the_same_rule(self):
        try:
            from ui.generator_main import _clean_path
        except ModuleNotFoundError as exc:  # PyQt6 없는 환경
            self.skipTest(str(exc))
        self.assertEqual(_clean_path(str(self.pct)), str(self.pct).replace("/", os.sep))
        self.assertEqual(safe_input_path(_clean_path(str(self.pct))), str(self.pct.resolve()))
        self.assertEqual(_clean_path(self.pct.as_uri()), str(self.pct).replace("/", os.sep))
        self.assertEqual(_clean_path("file:///C:/a%20b/c.png"), "C:/a b/c.png".replace("/", os.sep))
        self.assertEqual(_clean_path("C:/a%20b/c.png"), "C:/a%20b/c.png".replace("/", os.sep))
        self.assertEqual(_clean_path(""), "")


if __name__ == "__main__":
    unittest.main()
