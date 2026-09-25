"""회귀 방지: 숨은 PyQt I2I·Inpaint·Upscale 탭과 그 전용 위젯을 다시 만들지 않는다.

세 탭은 기동 때 만들어지자마자 setParent(None) 으로 떼어져 한 번도 보이지 않았다.
  - Img2ImgTab: Vue ``generate_i2i`` 페이로드를 숨은 입력칸에 넣었다가 다시 읽어 워커를 돌렸다
    (이전 요청의 입력 이미지가 남아 이미지 없는 요청도 그 이미지로 생성됐다). 이제
    core/i2i_payload(순수) + ui/i2i_actions(Qt 접착층)가 Vue 페이로드만 본다.
  - InpaintTab: 생성은 이미 core/inpaint_payload + ui/inpaint_actions 로 옮겨졌고 설정 왕복만 남아 있었다.
  - UpscaleTab: 만들어지기만 하고 쓰이지 않았다(Vue 업스케일은 ui/upscale_actions).
prompt_settings.json 의 i2i_settings·inpaint_settings 는 숨은 위젯 왕복이었다(Vue 화면은 읽은 적이
없다) — 이제 읽지도 쓰지도 않는다(core/prompt_settings_extras.RETIRED_KEYS). 탭만 쓰던
widgets/common_widgets(NoScrollComboBox)·widgets/sliders(NumericSlider)도 함께 은퇴했다.
EventGen·XYZ·PNG Info·Gallery 은퇴(tests/test_legacy_gallery_tabs_retirement.py)와 같은 방식이다.
"""
from __future__ import annotations

import ast
import json
import tempfile
import unittest
import warnings
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]

RETIRED_FILES = (
    "tabs/i2i_tab.py",
    "tabs/inpaint_tab.py",
    "tabs/upscale_tab.py",
    "widgets/common_widgets.py",
    "widgets/sliders.py",
)
RETIRED_MODULES = tuple(path[:-3].replace("/", ".") for path in RETIRED_FILES)

# 코드(주석·독스트링 제외)에 다시 나오면 안 되는 이름 — 탭 속성·클래스·탭 전용 메서드·위젯
RETIRED_NAMES = (
    "i2i_tab", "inpaint_tab", "upscale_tab", "Img2ImgTab", "InpaintTab", "UpscaleTab", "MaskCanvas",
    "generate_from_payload", "start_from_payload", "set_image_and_mask",
    "_get_i2i_settings", "_apply_i2i_settings", "_get_inpaint_settings", "_apply_inpaint_settings",
    "NoScrollComboBox", "NumericSlider",
)

# Vue 화면이 쓰는 살아 있는 짝 — 은퇴와 함께 지우면 안 된다.
LIVE_MODULES = (
    "core/i2i_payload.py", "ui/i2i_actions.py",
    "core/inpaint_payload.py", "ui/inpaint_actions.py",
    "core/upscale_settings.py", "ui/upscale_actions.py", "workers/upscale_worker.py",
    "core/image_payload.py", "core/request_numbers.py", "workers/generation_worker.py",
    "tabs/browser_tab.py", "tabs/backend_ui_tab.py",
)

_SOURCE_DIRS = ("core", "ui", "tabs", "widgets", "workers", "utils", "backends", "tools", "tests")


def _python_sources():
    files = sorted(ROOT.glob("*.py"))
    for name in _SOURCE_DIRS:
        base = ROOT / name
        if base.is_dir():
            files.extend(p for p in sorted(base.rglob("*.py")) if "__pycache__" not in p.parts)
    return [p for p in files if p.resolve() != Path(__file__).resolve()]


def _parse(path: Path) -> ast.AST:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")   # 다른 파일의 잘못된 이스케이프 경고는 이 테스트 몫이 아니다
        return ast.parse(path.read_text(encoding="utf-8"), filename=str(path))


def _code_names(tree: ast.AST) -> set[str]:
    """코드가 실제로 쓰는 이름 — 주석·독스트링은 빼고 hasattr(self, 'x') 같은 이름 문자열은 넣는다."""
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Name):
            names.add(node.id)
        elif isinstance(node, ast.Attribute):
            names.add(node.attr)
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names.add(node.name)
        elif isinstance(node, ast.alias):
            names.update(node.name.split("."))
            if node.asname:
                names.add(node.asname)
        elif isinstance(node, ast.ImportFrom) and node.module:
            names.update(node.module.split("."))
        elif isinstance(node, ast.Constant) and isinstance(node.value, str) and node.value.isidentifier():
            names.add(node.value)
    return names


def _retired_import(node) -> str | None:
    names = []
    if isinstance(node, ast.Import):
        names = [alias.name for alias in node.names]
    elif isinstance(node, ast.ImportFrom) and node.module:
        names = [node.module] + [f"{node.module}.{alias.name}" for alias in node.names]
    for name in names:
        for retired in RETIRED_MODULES:
            if name == retired or name.startswith(retired + "."):
                return name
    return None


class RetiredFilesTests(unittest.TestCase):
    def test_retired_files_are_gone(self):
        present = [rel for rel in RETIRED_FILES if (ROOT / rel).exists()]
        self.assertEqual(present, [])
        # 패키지 표시 파일은 남긴다 — 없으면 namespace package 가 된다
        self.assertTrue((ROOT / "tabs" / "__init__.py").is_file())
        self.assertTrue((ROOT / "widgets" / "__init__.py").is_file())

    def test_live_counterparts_remain(self):
        missing = [rel for rel in LIVE_MODULES if not (ROOT / rel).is_file()]
        self.assertEqual(missing, [])

    def test_no_source_imports_or_references_the_retired_tabs(self):
        offenders = []
        for path in _python_sources():
            rel = path.relative_to(ROOT).as_posix()
            tree = _parse(path)
            for node in ast.walk(tree):
                hit = _retired_import(node)
                if hit:
                    offenders.append(f"{rel}:{node.lineno} import {hit}")
            # 테스트는 은퇴를 설명·검증하느라 옛 이름을 문자열로 쓸 수 있다 — 앱 코드만 이름을 본다
            if not rel.startswith("tests/"):
                for name in sorted(set(RETIRED_NAMES) & _code_names(tree)):
                    offenders.append(f"{rel}: {name}")
        self.assertEqual(offenders, [])

    def test_ui_setup_builds_only_the_native_web_and_backend_tabs(self):
        tree = _parse(ROOT / "ui" / "generator_ui_setup.py")
        tab_imports = sorted({node.module for node in ast.walk(tree)
                              if isinstance(node, ast.ImportFrom) and (node.module or "").startswith("tabs.")})
        self.assertEqual(tab_imports, ["tabs.backend_ui_tab", "tabs.browser_tab"])
        self.assertNotIn("setParent", _code_names(tree), "숨은 탭을 만들고 떼어 두는 패턴이 되살아났다")


class DispatcherTests(unittest.TestCase):
    def _block(self, action: str) -> str:
        source = (ROOT / "ui" / "generator_main.py").read_text(encoding="utf-8")
        start = source.index(f"elif action == '{action}':")
        return source[start:source.index("elif action", start + 10)]

    def test_generation_actions_use_the_vue_payload_glue(self):
        for action, call in (("generate_i2i", "start_vue_i2i(self, payload)"),
                             ("generate_inpaint", "start_vue_inpaint(self, payload)"),
                             ("start_upscale", "start_vue_upscale(self, payload)")):
            with self.subTest(action=action):
                block = self._block(action)
                self.assertIn(call, block)
                self.assertNotIn("_tab", block)


class RetiredSettingsKeysTests(unittest.TestCase):
    KEYS = ("i2i_settings", "inpaint_settings")

    def test_keys_are_retired_with_a_reason(self):
        from core.prompt_settings_extras import RETIRED_KEYS
        for key in self.KEYS:
            with self.subTest(key=key):
                self.assertTrue(RETIRED_KEYS.get(key), "은퇴한 키는 이유와 함께 RETIRED_KEYS 에 남긴다")
        # 저장·복원 코드에 문자열로도 남지 않는다(주석의 설명은 괜찮다)
        tree = _parse(ROOT / "ui" / "generator_settings.py")
        constants = {node.value for node in ast.walk(tree)
                     if isinstance(node, ast.Constant) and isinstance(node.value, str)}
        self.assertEqual(sorted(set(self.KEYS) & constants), [])

    def test_old_settings_file_loads_and_save_drops_the_keys(self):
        """옛 prompt_settings.json(두 키 포함)이 오류 없이 읽히고, 다음 저장엔 두 키가 없다."""
        from ui.generator_settings import SettingsMixin

        class _Host(SettingsMixin):
            def __getattr__(self, name):   # 이 테스트가 보지 않는 위젯은 MagicMock
                if name.startswith("__"):
                    raise AttributeError(name)
                value = mock.MagicMock(name=name)
                setattr(self, name, value)
                return value

        saved = {
            "i2i_settings": {"prompt": "old", "denoising": "0.3", "resize_mode": 2, "width": "640"},
            "inpaint_settings": {"prompt": "old", "fill_mode": 0, "brush_size": 12, "full_res": False},
            "wildcard_enabled": True,
        }
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "prompt_settings.json"
            path.write_text(json.dumps(saved), encoding="utf-8")
            host = _Host()
            with mock.patch("ui.generator_settings.PROMPT_SETTINGS_FILE", str(path)), \
                    mock.patch("ui.generator_settings.migrate_legacy_gallery_folder", return_value=""), \
                    mock.patch("ui.generation_settings_apply.apply_generation_settings", return_value=[]), \
                    mock.patch("utils.theme_manager.get_theme_manager"), \
                    mock.patch("ui.status_line.notify_user") as notify_user:
                SettingsMixin.load_settings(host)
        notify_user.assert_not_called()   # 불러오기 실패 알림이 없다
        self.assertEqual(host.loaded_settings, saved)
        for name in ("i2i_tab", "inpaint_tab"):
            self.assertNotIn(name, vars(host), f"불러오기가 은퇴한 {name} 에 손대지 않는다")
        written = SettingsMixin._build_settings_dict(host)
        self.assertEqual(sorted(set(self.KEYS) & set(written)), [])


class ReadmeTests(unittest.TestCase):
    def test_readme_no_longer_describes_the_hidden_tabs(self):
        readme = (ROOT / "README.md").read_text(encoding="utf-8")
        for name in ("Img2ImgTab", "InpaintTab", "UpscaleTab", "숨겨진 I2I"):
            self.assertNotIn(name, readme)
        line = next(l for l in readme.splitlines() if l.startswith("├── tabs/"))
        self.assertIn("Web·Backend 네이티브 화면", line)


if __name__ == "__main__":
    unittest.main()
