"""개행·인코딩 정책 회귀 가드 — 배치 파일 CRLF·코드 페이지·종료코드, Vue scoped id 의 개행 무관성.

- cmd.exe 는 LF 전용 줄이 섞인 .bat 에서 `call :label`/`goto` 의 라벨 탐색이 바이트 오프셋에 따라
  빗나간다. 앱 업데이터가 new_run_main_ui.bat 을 재시작에 쓰므로 런처가 깨지면 venv 가 옆으로
  밀리고 새로 만들어진다. .gitattributes 가 체크아웃을 CRLF 로 고정하고, 여기서 바이트를 지킨다.
- 배치 파일은 BOM 없는 UTF-8 이다(BOM 이 붙으면 첫 줄 `@echo off` 가 명령으로 안 읽힌다). 한글이
  든 런처는 새 콘솔의 기본 코드 페이지(한국어 Windows 는 cp949)로는 echo 와 UTF-8 크래시 로그
  (`type logs\\last_crash.log`)가 깨지므로, 첫 비ASCII 바이트 **앞에서** 65001 로 바꾸고 원래 코드
  페이지로 되돌린다. (venv 의 activate.bat 은 호출 전 코드 페이지로 되돌리므로 그보다 앞서야 한다.)
  단 Ctrl+C 로 끊길 수 있는 명령(업데이트 대기 powershell · 앱 · pause)은 호출한 콘솔의 코드
  페이지에서 돈다 — "일괄 작업을 끝내시겠습니까 (Y/N)?" 에 Y 면 :finish 의 복원을 건너뛰어, 기존
  cmd·터미널에서 띄운 경우 그 콘솔이 65001 로 남는다(웹 모드는 Ctrl+C 가 평소 종료 방법이다).
- 런처는 앱 종료코드가 0 이 아니면 전부 크래시로 본다 — 네이티브 크래시(0xC0000005 등)는 음수라
  `if errorlevel 1` 로는 못 잡아 크래시 로그 없이 창이 닫혔다.
- plugin-vue 의 production scoped id 기본값은 hash(경로 + 소스)라, 체크아웃 개행이 다르면
  git 상 변경이 없어도 data-v id 와 dist 청크가 흔들린다. vite.config.js 는 경로만 쓴다.
"""
from __future__ import annotations

import re
import subprocess
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _tracked_batch_files() -> list[Path]:
    """추적 중인 .bat/.cmd. git 을 못 쓰면 루트·frontend/dev 를 직접 훑는다."""
    try:
        out = subprocess.run(
            ["git", "ls-files", "-z", "--", "*.bat", "*.cmd"],
            cwd=ROOT, capture_output=True, check=True, timeout=30,
        ).stdout
        names = [n for n in out.decode("utf-8").split("\0") if n]
        if names:
            return [ROOT / n for n in names]
    except (OSError, subprocess.SubprocessError, UnicodeDecodeError):
        pass
    found = []
    for folder in (ROOT, ROOT / "frontend" / "dev"):
        found += sorted(folder.glob("*.bat")) + sorted(folder.glob("*.cmd"))
    return found


def _attribute_lines() -> list[str]:
    text = (ROOT / ".gitattributes").read_text(encoding="utf-8")
    return [line.strip() for line in text.splitlines() if line.strip() and not line.lstrip().startswith("#")]


class BatchFileLineEndingTests(unittest.TestCase):
    def test_gitattributes_forces_crlf_for_batch_files(self) -> None:
        rules = {tuple(line.split()) for line in _attribute_lines()}
        self.assertIn(("*.bat", "text", "eol=crlf"), rules)
        self.assertIn(("*.cmd", "text", "eol=crlf"), rules)

    def test_gitattributes_has_no_repo_wide_renormalizing_rule(self) -> None:
        """`* text=auto` 같은 광범위 규칙은 더러운 트리 전체를 renormalize 대상으로 만든다."""
        for line in _attribute_lines():
            pattern = line.split()[0]
            with self.subTest(rule=line):
                self.assertNotIn(pattern, {"*", "**", "*.*"})

    def test_tracked_batch_files_are_pure_crlf(self) -> None:
        files = _tracked_batch_files()
        self.assertTrue(any(p.name == "new_run_main_ui.bat" for p in files), files)
        for path in files:
            data = path.read_bytes()
            with self.subTest(file=path.relative_to(ROOT).as_posix()):
                self.assertEqual(data.count(b"\n"), data.count(b"\r\n"), "LF 전용 줄이 섞였습니다")
                self.assertIsNone(re.search(rb"\r(?!\n)", data), "고립된 CR 이 있습니다")


# `chcp 65001` · `"%SystemRoot%\System32\chcp.com" 65001` 모두 — 코드 페이지를 UTF-8 로 바꾸는 줄.
_CHCP_UTF8 = re.compile(rb'(?im)^[ \t]*@?(?:"[^"\r\n]*\\chcp\.com"|chcp(?:\.com)?)[ \t]+65001\b')
# 저장해 둔 코드 페이지로 되돌리는 줄(`chcp %PREV%`).
_CHCP_RESTORE = re.compile(rb'(?i)\bchcp(?:\.com)?"?[ \t]+%(\w+)%')
# 파일 내용을 콘솔에 쏟는 `type` 명령 줄(`if exist X type X` 포함).
_TYPE_COMMAND = re.compile(rb'(?im)^[ \t]*(?:if exist \S+ )?type[ \t]')
_LAUNCHERS = {"new_run_main_ui.bat": "new_main_ui.py", "run_gui.bat": "new_main_ui.py",
              "run_WEB_gui.bat": "web_main_ui.py"}
# Ctrl+C 로 끊길 수 있는(사람을 기다리거나 오래 도는) 명령 — 앱 실행 줄은 _LAUNCHERS 로 따로 본다.
_INTERRUPTIBLE = re.compile(r"(?i)^[ \t]*(?:if defined \w+ )?(?:pause|powershell(?:\.exe)?)\b")
_EXIT_LINE = re.compile(r"(?i)\bexit /b\b")


def _code_page_walk(data: bytes, entry: str | None = None) -> list[tuple[int, str]]:
    """배치 파일을 위에서 아래로 읽으며 콘솔 코드 페이지 상태를 따라간다 → 규칙 위반 [(줄 번호, 사유)].

    - 비ASCII 줄·크래시 로그 `type` 은 65001 에서(cp949 콘솔에선 깨진다)
    - Ctrl+C 로 끊길 수 있는 명령(pause·powershell·앱 실행 줄)은 호출한 콘솔의 코드 페이지에서
    - `exit /b` 는 호출한 코드 페이지로 되돌린 뒤에
    goto 는 따라가지 않는다 — 런처의 goto 는 전부 :finish(복원 → pause → exit)로 가므로 줄 순서가 곧
    실행 순서다. 복원과 다음 65001 사이엔 ASCII 줄만 둔다(첫 규칙이 그것을 지킨다).
    """
    problems = []
    state = "caller"
    for number, raw in enumerate(data.split(b"\r\n"), 1):
        line = raw.decode("utf-8")
        if _CHCP_UTF8.search(raw):
            state = "utf8"
        elif _CHCP_RESTORE.search(raw):
            state = "caller"
        is_rem = line.lstrip().upper().startswith("REM")
        if state != "utf8" and (any(ord(ch) > 0x7F for ch in line) or _TYPE_COMMAND.search(raw)):
            problems.append((number, "비ASCII 줄·크래시 로그 type 이 65001 밖에 있다"))
        launches_app = bool(entry) and entry in line and not is_rem
        if state != "caller" and (_INTERRUPTIBLE.search(line) or launches_app):
            problems.append((number, "Ctrl+C 로 끊길 수 있는 명령이 65001 에서 돈다"))
        if state != "caller" and _EXIT_LINE.search(line) and not is_rem:
            problems.append((number, "65001 인 채로 끝난다"))
    return problems


class BatchFileEncodingTests(unittest.TestCase):
    def test_batch_files_are_utf8_without_bom(self) -> None:
        for path in _tracked_batch_files():
            data = path.read_bytes()
            with self.subTest(file=path.relative_to(ROOT).as_posix()):
                self.assertFalse(data.startswith(b"\xef\xbb\xbf"), "BOM 이 있으면 첫 줄이 깨집니다")
                data.decode("utf-8")  # UnicodeDecodeError = 다른 인코딩으로 저장됨

    def test_non_ascii_batch_files_switch_to_utf8_first_and_restore(self) -> None:
        checked = 0
        for path in _tracked_batch_files():
            data = path.read_bytes()
            first_non_ascii = next((i for i, byte in enumerate(data) if byte > 0x7F), None)
            if first_non_ascii is None:
                continue
            checked += 1
            with self.subTest(file=path.relative_to(ROOT).as_posix()):
                switch = _CHCP_UTF8.search(data)
                self.assertIsNotNone(switch, "한글이 있는데 chcp 65001 이 없습니다 — cp949 콘솔에서 깨집니다")
                self.assertLess(switch.start(), first_non_ascii,
                                "chcp 65001 보다 앞에 비ASCII 바이트가 있습니다")
                restore = _CHCP_RESTORE.search(data)
                self.assertIsNotNone(restore, "원래 코드 페이지로 되돌리는 줄이 없습니다")
                saved = restore.group(1)
                self.assertRegex(data, rb"(?i)for /f .*chcp.*" + saved,
                                 "되돌릴 코드 페이지를 chcp 출력에서 저장하지 않습니다")
        self.assertGreater(checked, 0, "한글이 든 런처를 하나도 찾지 못했습니다(탐지가 깨졌습니다)")

    def test_launchers_show_the_utf8_crash_log_under_utf8(self) -> None:
        """logs\\last_crash.log 는 UTF-8(한글 메시지·경로) — `type` 은 콘솔 코드 페이지로 그리므로
        cp949 콘솔에선 깨진다. 모든 `type` 줄 앞에서 65001 로 바꾸고, 원래 코드 페이지로 되돌린다."""
        for launcher in _LAUNCHERS:
            data = (ROOT / launcher).read_bytes()
            with self.subTest(launcher=launcher):
                types = [m.start() for m in _TYPE_COMMAND.finditer(data)]
                self.assertTrue(types, "크래시 로그를 보여 주는 type 줄이 없습니다")
                switch = _CHCP_UTF8.search(data)
                self.assertIsNotNone(switch)
                self.assertLess(switch.start(), min(types))
                self.assertIsNotNone(_CHCP_RESTORE.search(data), "원래 코드 페이지로 되돌리는 줄이 없습니다")

    def test_interruptible_commands_run_under_the_callers_code_page(self) -> None:
        """Ctrl+C → "일괄 작업을 끝내시겠습니까 (Y/N)?" Y 는 :finish 를 건너뛴다. 그 순간 콘솔이 65001 이면
        기존 cmd·터미널에서 띄운 경우 그 콘솔이 65001 로 남는다 — 끊길 수 있는 명령 전에 되돌린다."""
        files = {path.relative_to(ROOT).as_posix(): path for path in _tracked_batch_files()}
        for launcher in _LAUNCHERS:
            self.assertIn(launcher, files)
        for rel, path in files.items():
            with self.subTest(file=rel):
                self.assertEqual(_code_page_walk(path.read_bytes(), _LAUNCHERS.get(rel)), [])

    def test_code_page_walk_catches_the_reviewed_shapes(self) -> None:
        good = (b'@echo off\r\n'
                b'"%SystemRoot%\\System32\\chcp.com" 65001 >nul\r\n'
                b'REM \xed\x95\x9c\xea\xb8\x80\r\n'
                b'if defined OLD "%SystemRoot%\\System32\\chcp.com" %OLD% >nul\r\n'
                b'python app.py\r\n'
                b'set "RC=%ERRORLEVEL%"\r\n'
                b'"%SystemRoot%\\System32\\chcp.com" 65001 >nul\r\n'
                b'if not "%RC%"=="0" (\r\n'
                b'    echo \xed\x95\x9c\xea\xb8\x80\r\n'
                b'    type logs\\last_crash.log\r\n'
                b'    set "WAIT=1"\r\n'
                b')\r\n'
                b':finish\r\n'
                b'if defined OLD "%SystemRoot%\\System32\\chcp.com" %OLD% >nul\r\n'
                b'if defined WAIT pause\r\n'
                b'endlocal & exit /b %RC%\r\n')
        self.assertEqual(_code_page_walk(good, "app.py"), [])
        # 리뷰가 짚은 모양: 맨 위에서 65001 로 바꾸고 앱·pause 를 그대로 돌린 뒤 :finish 에서만 되돌린다.
        reviewed = (b'"%SystemRoot%\\System32\\chcp.com" 65001 >nul\r\n'
                    b'powershell.exe -Command "exit 0"\r\n'
                    b'python app.py\r\n'
                    b'pause\r\n'
                    b':finish\r\n'
                    b'if defined OLD "%SystemRoot%\\System32\\chcp.com" %OLD% >nul\r\n'
                    b'exit /b 0\r\n')
        self.assertEqual([line for line, _why in _code_page_walk(reviewed, "app.py")], [2, 3, 4])
        # 되돌린 뒤 다시 65001 로 바꾸지 않고 한글을 찍거나, 65001 인 채로 끝낸다.
        drift = (b'"%SystemRoot%\\System32\\chcp.com" 65001 >nul\r\n'
                 b'if defined OLD "%SystemRoot%\\System32\\chcp.com" %OLD% >nul\r\n'
                 b'echo \xed\x95\x9c\xea\xb8\x80\r\n'
                 b'"%SystemRoot%\\System32\\chcp.com" 65001 >nul\r\n'
                 b'exit /b 1\r\n')
        self.assertEqual([line for line, _why in _code_page_walk(drift)], [3, 5])

    def test_policy_patterns_match_the_intended_lines(self) -> None:
        self.assertIsNotNone(_CHCP_UTF8.search(b'"%SystemRoot%\\System32\\chcp.com" 65001 >nul\r\n'))
        self.assertIsNotNone(_CHCP_UTF8.search(b"chcp 65001 >nul\r\n"))
        self.assertIsNone(_CHCP_UTF8.search(b"REM chcp 65001 is needed\r\n"))
        self.assertEqual(_CHCP_RESTORE.search(b'"%SystemRoot%\\System32\\chcp.com" %OLD_CP% >nul').group(1),
                         b"OLD_CP")


class LauncherExitCodeTests(unittest.TestCase):
    def test_any_nonzero_app_exit_shows_the_crash_log(self) -> None:
        """앱 종료코드를 곧바로 저장하고 `if not "%X%"=="0"` 로 본다(음수 크래시 코드 포함)."""
        for launcher, entry in _LAUNCHERS.items():
            lines = (ROOT / launcher).read_text(encoding="utf-8").splitlines()
            with self.subTest(launcher=launcher):
                index = next(i for i, line in enumerate(lines) if entry in line and not line.lstrip().upper().startswith("REM"))
                saved = re.fullmatch(r'\s*set "(\w+)=%ERRORLEVEL%"\s*', lines[index + 1])
                self.assertIsNotNone(saved, f"{entry} 다음 줄에서 종료코드를 저장하지 않습니다: {lines[index + 1]!r}")
                rest = "\n".join(lines[index + 2:])
                self.assertIn(f'if not "%{saved.group(1)}%"=="0" (', rest)
                crash_block = rest.split(f'if not "%{saved.group(1)}%"=="0" (', 1)[1]
                self.assertRegex(crash_block, r"(?im)^\s*(?:if exist \S+ )?type\b",
                                 "비정상 종료 분기에서 크래시 로그를 보여 주지 않습니다")
                self.assertRegex(rest, r"exit /b %" + saved.group(1) + r"%",
                                 "종료코드를 호출자(업데이터·셸)에 돌려주지 않습니다")


class VueScopedIdTests(unittest.TestCase):
    def test_vite_uses_path_only_component_ids(self) -> None:
        text = (ROOT / "frontend" / "vite.config.js").read_text(encoding="utf-8")
        code = "\n".join(line.split("//", 1)[0] for line in text.splitlines())
        self.assertRegex(code, r"componentIdGenerator\s*:\s*['\"]filepath['\"]")


if __name__ == "__main__":
    unittest.main()
