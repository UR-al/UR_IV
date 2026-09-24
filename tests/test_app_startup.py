from __future__ import annotations

import unittest
from pathlib import Path
from unittest.mock import patch

from core.app_startup import prepare_application


ROOT = Path(__file__).resolve().parents[1]


class AppStartupTests(unittest.TestCase):
    def test_prepare_stops_before_data_when_requirements_fail(self) -> None:
        with (
            patch("core.check_requirements.main", return_value=7),
            patch("core.fetch_data.ensure_data") as fetch,
        ):
            self.assertEqual(7, prepare_application())
        fetch.assert_not_called()

    def test_prepare_checks_requirements_then_data(self) -> None:
        order: list[str] = []
        with (
            patch("core.check_requirements.main", side_effect=lambda: order.append("deps") or 0),
            patch("core.fetch_data.ensure_data", side_effect=lambda: order.append("data") or 0),
        ):
            self.assertEqual(0, prepare_application())
        self.assertEqual(["deps", "data"], order)

    def test_launchers_delegate_preparation_after_instance_registration(self) -> None:
        for entrypoint in ("new_main_ui.py", "web_main_ui.py"):
            source = (ROOT / entrypoint).read_text(encoding="utf-8")
            with self.subTest(entrypoint=entrypoint):
                self.assertLess(
                    source.index("register_app_instance"),
                    source.index("prepare_application"),
                )
                self.assertLess(
                    source.index("prepare_application"),
                    source.index("from config import"),
                )

        for launcher in ("new_run_main_ui.bat", "run_gui.bat", "run_WEB_gui.bat"):
            source = (ROOT / launcher).read_text(encoding="utf-8")
            data = (ROOT / launcher).read_bytes()
            with self.subTest(launcher=launcher):
                self.assertNotIn("core\\check_requirements.py", source)
                self.assertNotIn("core\\fetch_data.py", source)
                # 순수 CRLF — LF 전용 줄이 섞이면 cmd.exe 의 라벨 탐색(call :venv_is_usable 등)이
                # 바이트 오프셋에 따라 라벨을 못 찾는다(.gitattributes *.bat eol=crlf 가 체크아웃을 맞춘다).
                self.assertEqual(data.count(b"\n"), data.count(b"\r\n"),
                                 f"{launcher} 에 LF 전용 줄이 섞였습니다 — 파일 전체를 CRLF 로 저장하세요")

    def test_agent_guides_credit_the_dependency_install_to_app_startup(self) -> None:
        """런처는 아무것도 설치하지 않는다(위 테스트) — 누락 패키지는 앱 시작 시 core/check_requirements 가
        설치한다. 안내 문서 셋(AGENTS.md·README.md·Copilot)이 한목소리로 그렇게 말해야 한다(감사 #168 후속:
        Copilot 안내만 '런처가 설치한다'로 남아 있었다)."""
        import re

        launcher_installs = re.compile(r"new_run_main_ui\.bat[^\n]*install", re.I)
        for rel in ("AGENTS.md", "README.md", ".github/copilot-instructions.md"):
            text = (ROOT / rel).read_text(encoding="utf-8")
            with self.subTest(guide=rel):
                self.assertRegex(text, r"core[\\/]check_requirements\.py")
                self.assertIsNone(launcher_installs.search(text), "런처가 패키지를 설치한다고 적혀 있다")


ENTRYPOINTS = ("new_main_ui.py", "web_main_ui.py")
LAUNCHERS = ("new_run_main_ui.bat", "run_gui.bat", "run_WEB_gui.bat")


def _gate_source(entrypoint: str) -> str:
    """엔트리포인트의 stdlib 전용 업데이트 게이트 함수 본문."""
    import ast

    source = (ROOT / entrypoint).read_text(encoding="utf-8")
    for node in ast.parse(source).body:
        if isinstance(node, ast.FunctionDef) and node.name == "_acquire_detached_update_gate":
            return ast.get_source_segment(source, node) or ""
    raise AssertionError(f"{entrypoint}: _acquire_detached_update_gate 가 없습니다")


class UpdateMutexNameContractTests(unittest.TestCase):
    """업데이트 뮤텍스 이름은 의도적으로 6곳에 복사돼 있다(게이트 전에는 프로젝트 코드를
    import 하지 않는다). 하나라도 어긋나면 업데이트 중 실행 방지가 조용히 깨지므로 고정한다."""

    def test_entrypoint_gates_use_the_updater_mutex_name(self) -> None:
        import ast

        from core.app_update_lock import MUTEX_NAME

        for entrypoint in ENTRYPOINTS:
            with self.subTest(entrypoint=entrypoint):
                gate = ast.parse(_gate_source(entrypoint))
                names = [
                    node.value for node in ast.walk(gate)
                    if isinstance(node, ast.Constant) and isinstance(node.value, str)
                    and node.value.startswith("Local\\")
                ]
                self.assertEqual([MUTEX_NAME], names)

    def test_both_entrypoint_gates_are_identical(self) -> None:
        import ast

        def body(entrypoint: str) -> str:
            function = ast.parse(_gate_source(entrypoint)).body[0]
            statements = list(function.body)
            if (statements and isinstance(statements[0], ast.Expr)
                    and isinstance(statements[0].value, ast.Constant)
                    and isinstance(statements[0].value.value, str)):
                statements = statements[1:]  # docstring 은 비교하지 않는다
            return ast.dump(ast.Module(body=statements, type_ignores=[]))

        self.assertEqual(body(ENTRYPOINTS[0]), body(ENTRYPOINTS[1]))

    def test_batch_launchers_wait_on_the_updater_mutex_name(self) -> None:
        import re

        from core.app_update_lock import MUTEX_NAME

        for launcher in LAUNCHERS:
            source = (ROOT / launcher).read_text(encoding="utf-8")
            with self.subTest(launcher=launcher):
                names = re.findall(r"\$n='([^']+)'", source)
                self.assertEqual([MUTEX_NAME], names)

    def test_entrypoints_share_a_real_app_user_model_id(self) -> None:
        from core.app_instance import APP_USER_MODEL_ID

        self.assertNotIn("mycompany", APP_USER_MODEL_ID)
        for entrypoint in ENTRYPOINTS:
            source = (ROOT / entrypoint).read_text(encoding="utf-8")
            with self.subTest(entrypoint=entrypoint):
                self.assertIn("SetCurrentProcessExplicitAppUserModelID(APP_USER_MODEL_ID)", source)
                self.assertNotIn("mycompany", source)


class InstallerProcessProbeTests(unittest.TestCase):
    def test_installer_reuses_the_single_process_exists_implementation(self) -> None:
        import inspect

        from core import app_instance, app_update_installer

        self.assertIs(app_update_installer.process_exists, app_instance.process_exists)
        default = inspect.signature(app_update_installer.wait_for_exit).parameters["exists"].default
        self.assertIs(default, app_instance.process_exists)


if __name__ == "__main__":
    unittest.main()
