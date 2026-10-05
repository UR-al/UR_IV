"""회귀 방지: 창 표시 전 import 경로가 무거운 패키지(pandas·cv2·jsonschema)와 노드 팩을 끌어오지 않는다.

new_main_ui.py / web_main_ui.py 는 창을 띄우기 전에 ``ui.generator_main`` 을 import 하고,
GeneratorMainUI._setup_ui 가 Web·Backend 네이티브 탭을 import·생성한다. 이 경로의 모듈 최상단 import 는
그대로 기동 임계 경로가 된다 — 쓰는 메서드 안(또는 utils.lazy_import)으로 미룬다.

또 패키지 ``__init__`` 이 서브모듈을 재수출하지 않는지 본다. 재수출은 서브모듈 하나만 import
해도 형제 모듈 전부(예: workers → search_worker → pandas)를 끌어왔다.

새 프로세스에서 확인한다 — 같은 프로세스의 다른 테스트가 이미 pandas 등을 불러왔을 수 있다.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# ui/generator_ui_setup.py 의 _setup_ui 가 창 표시 전에 import 하는 탭 모듈들.
# (숨은 I2I·Inpaint·Upscale 탭은 은퇴했다 — tests/test_legacy_i2i_inpaint_upscale_retirement.py)
_SETUP_UI_TABS = (
    "tabs.browser_tab", "tabs.backend_ui_tab",
)
_HEAVY = ("pandas", "pyarrow", "cv2", "jsonschema", "comfy_custom_nodes.ai_studio_forge_parity")


def _loaded_after(code: str) -> dict:
    env = dict(os.environ)
    env.setdefault("AISTUDIO_LOG_FILE", os.path.join(tempfile.gettempdir(), "aistudio_tests", "app.log"))
    env["QT_QPA_PLATFORM"] = env.get("QT_QPA_PLATFORM", "offscreen")
    script = (
        "import json, sys\n"
        f"{code}\n"
        f"print('@@' + json.dumps({{m: (m in sys.modules) for m in {list(_HEAVY) + ['workers.search_worker', 'ui.generator_main']!r}}}))\n"
    )
    proc = subprocess.run(
        [sys.executable, "-c", script], cwd=str(ROOT), env=env,
        capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=180,
    )
    if proc.returncode != 0:
        raise AssertionError(f"import 실패 (rc={proc.returncode}):\n{proc.stderr[-4000:]}")
    for line in proc.stdout.splitlines():
        if line.startswith("@@"):
            return json.loads(line[2:])
    raise AssertionError(f"결과 줄이 없음:\n{proc.stdout[-2000:]}")


class StartupImportTests(unittest.TestCase):
    def test_window_path_does_not_import_heavy_packages(self):
        tabs = "\n".join(f"import {name}" for name in _SETUP_UI_TABS)
        loaded = _loaded_after(f"import ui.generator_main\nimport ui.widget_proxies\n{tabs}")
        self.assertTrue(loaded["ui.generator_main"])
        heavy = [name for name in _HEAVY if loaded[name]]
        self.assertEqual(
            heavy, [],
            "창 표시 전 import 경로가 무거운 패키지/노드 팩을 불러왔다 — 사용하는 메서드 안에서 "
            "import 하거나 utils.lazy_import.lazy_module 을 쓸 것",
        )

    def test_restoring_a_saved_custom_expression_does_not_import_the_pack(self):
        """창 생성자의 load_settings 가 Extra Schedulers 저장값을 칸에 쓰면 custom 식 미리 검사(계획 C7)가 돈다. 저장된 식이
        기본값이 아니면 매 기동마다 그 검사가 팩 패키지(``__init__`` 이 노드 모듈 전부·asyncio·설치 훅)를 불러왔다(GC 검토
        c7). 검사는 계산 파일 하나만 불러오고, 오류 줄은 그대로 나간다."""
        tabs = "\n".join(f"import {name}" for name in _SETUP_UI_TABS)
        loaded = _loaded_after(
            f"import ui.generator_main\nimport ui.widget_proxies\n{tabs}\n"
            "from ui import extra_schedulers_ui as xui\n"
            "class _Bridge:\n"
            "    def __init__(self): self.props = []\n"
            "    def _register_proxy(self, *args): pass\n"
            "    def pushWidgetValue(self, *args): pass\n"
            "    def pushWidgetProperty(self, *args): self.props.append(args)\n"
            "class _Host: pass\n"
            "host, bridge = _Host(), _Bridge()\n"
            "host.extra_schedulers_widgets = xui.init_extra_schedulers_proxies(bridge)\n"
            "xui.apply_saved_settings(host, {'extra_schedulers_settings': {'custom_mode': 'expression', "
            "'custom_expression': 'm + (M - m) * (1 - q) ** 3'}}, only_present=False)\n"
            "assert any(\"unknown name 'q'\" in str(prop) for prop in bridge.props), bridge.props\n"
            "xui.apply_saved_settings(host, {'extra_schedulers_settings': {'custom_mode': 'sigmas', "
            "'custom_sigmas': '[1.0, 0.5, 0.0]'}}, only_present=False)\n")
        self.assertFalse(loaded["comfy_custom_nodes.ai_studio_forge_parity"],
                         "Extra Schedulers 미리 검사가 기동 경로에서 노드 팩 패키지를 불러왔다")

    def test_package_inits_do_not_reexport_submodules(self):
        loaded = _loaded_after("import workers.generation_worker\nimport ui.vue_bridge\nimport widgets.queue_manager\nimport core.file_naming")
        self.assertFalse(loaded["workers.search_worker"], "workers/__init__ 이 search_worker 를 재수출한다")
        self.assertFalse(loaded["ui.generator_main"], "ui/__init__ 이 메인 창을 재수출한다")
        self.assertFalse(loaded["cv2"], "widgets/__init__ 이 cv2 를 쓰는 위젯을 재수출한다")
        self.assertFalse(loaded["pandas"])

    def test_package_init_files_hold_no_imports(self):
        for package in ("core", "workers", "tabs", "widgets", "ui"):
            source = (ROOT / package / "__init__.py").read_text(encoding="utf-8")
            code_lines = [
                line for line in source.splitlines()
                if line.strip().startswith(("import ", "from ")) or line.strip().startswith("__all__")
            ]
            self.assertEqual(code_lines, [], f"{package}/__init__.py 가 다시 재수출을 한다: {code_lines}")

    def test_repository_root_is_not_a_package(self):
        # 초기 업로드 잔재(여러 __init__ 을 이어 붙인 파일)는 import 하면 즉시 실패했다.
        self.assertFalse((ROOT / "__init__.py").exists())


if __name__ == "__main__":
    unittest.main()
