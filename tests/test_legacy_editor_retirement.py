"""회귀 방지: 숨은 PyQt 에디터(MosaicEditor)와 숨은 BatchTab 은 은퇴했다(감사 #60).

MosaicEditor 는 기동 때 InteractiveLabel·서브패널 약 5,500줄을 만들고 곧바로 setParent(None) 해 두었다.
라이브 코드가 쓰던 것은 YOLO 경로 헬퍼(→ core/yolo_models.py), 숨은 라벨 갱신, 죽은 설정 왕복
(editor_defaults·bg_removal_model)뿐이었고 Vue 에디터 연산은 ui/vue_bridge.py 와 core/editor_* 가 한다.
BatchTab 은 Vue 배치가 workers/batch_image_worker.py 로 옮겨간 뒤 tabs/editor/batch_worker 만 붙잡고
있던 사문이었다. utils/shortcut_manager 의 13개 단축키도 이 에디터만 읽었고(바꾸는 UI 는 숨은 설정 탭)
prompt_settings 'shortcuts' 왕복만 남아 함께 지웠다. 다시 import·생성하지 않도록 AST 로 지킨다.
"""
from __future__ import annotations

import ast
import unittest
import warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# 파이썬 소스가 사는 곳 — 큰 런타임 폴더(image_cache·generated_images·venv 등)는 훑지 않는다.
_SOURCE_DIRS = ("core", "ui", "tabs", "widgets", "workers", "utils", "backends", "tools", "tests")

_RETIRED_MODULES = ("tabs.editor", "tabs.editor_tab", "widgets.interactive_label", "tabs.batch_tab",
                    "utils.shortcut_manager")
_RETIRED_FILES = ("tabs/editor", "tabs/editor_tab.py", "widgets/interactive_label.py", "tabs/batch_tab.py",
                  "utils/shortcut_manager.py")
_RETIRED_NAMES = ("MosaicEditor", "InteractiveLabel", "BatchTab", "mosaic_editor", "batch_tab",
                  "ShortcutManager", "get_shortcut_manager")

# Vue 에디터가 쓰는 살아 있는 순수 로직 — 은퇴와 함께 지우면 안 된다.
_LIVE_EDITOR_CORE = (
    "core/editor_ops.py", "core/editor_preview.py", "core/editor_save.py", "core/editor_watermark.py",
    "core/editor_autosave.py", "core/yolo_models.py", "core/bg_removal.py", "core/curves.py",
    "core/batch_image_ops.py", "workers/batch_image_worker.py",
)


def _python_sources():
    files = sorted(ROOT.glob("*.py"))
    for name in _SOURCE_DIRS:
        base = ROOT / name
        if base.is_dir():
            files.extend(p for p in sorted(base.rglob("*.py")) if "__pycache__" not in p.parts)
    return [p for p in files if p.resolve() != Path(__file__).resolve()]


def _retired_import(node) -> str | None:
    names = []
    if isinstance(node, ast.Import):
        names = [alias.name for alias in node.names]
    elif isinstance(node, ast.ImportFrom) and node.module:
        names = [node.module] + [f"{node.module}.{alias.name}" for alias in node.names]
    for name in names:
        for retired in _RETIRED_MODULES:
            if name == retired or name.startswith(retired + "."):
                return name
    return None


class LegacyEditorRetirementTests(unittest.TestCase):
    def test_retired_files_are_gone(self):
        for rel in _RETIRED_FILES:
            self.assertFalse((ROOT / rel).exists(), f"은퇴한 레거시 에디터 파일이 되살아났다: {rel}")

    def test_live_editor_core_modules_remain(self):
        for rel in _LIVE_EDITOR_CORE:
            self.assertTrue((ROOT / rel).is_file(), f"Vue 에디터/배치가 쓰는 모듈이 없다: {rel}")

    def test_no_source_imports_or_references_the_retired_editor(self):
        offenders = []
        for path in _python_sources():
            rel = path.relative_to(ROOT).as_posix()
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")   # 다른 파일의 잘못된 이스케이프 경고는 이 테스트 몫이 아니다
                tree = ast.parse(path.read_text(encoding="utf-8"), filename=rel)
            # 테스트는 '지워졌는지' 확인하느라 옛 이름을 문자열로 쓸 수 있다 — import 만 막는다.
            names_too = not rel.startswith("tests/")
            for node in ast.walk(tree):
                hit = _retired_import(node)
                if hit:
                    offenders.append(f"{rel}:{node.lineno} import {hit}")
                elif not names_too:
                    continue
                elif isinstance(node, ast.Attribute) and node.attr in _RETIRED_NAMES:
                    offenders.append(f"{rel}:{node.lineno} .{node.attr}")
                elif isinstance(node, ast.Name) and node.id in _RETIRED_NAMES:
                    offenders.append(f"{rel}:{node.lineno} {node.id}")
                elif (isinstance(node, ast.Constant) and isinstance(node.value, str)
                      and node.value in _RETIRED_NAMES):
                    # hasattr(self, 'mosaic_editor') / getattr(..., 'batch_tab') 같은 이름 참조
                    offenders.append(f"{rel}:{node.lineno} '{node.value}'")
        self.assertEqual(offenders, [], "은퇴한 PyQt 에디터/BatchTab 을 다시 참조한다")

    def test_dead_editor_settings_are_not_round_tripped(self):
        settings = (ROOT / "ui" / "generator_settings.py").read_text(encoding="utf-8")
        tree = ast.parse(settings)
        keys = {node.value for node in ast.walk(tree)
                if isinstance(node, ast.Constant) and isinstance(node.value, str)}
        # 숨은 에디터 전용 키 — 저장·복원 모두 없어야 한다(옛 파일의 키는 그냥 무시된다).
        from core.prompt_settings_extras import PromptSettingsExtras, RETIRED_KEYS
        for key in ("bg_removal_model", "editor_defaults", "shortcuts"):
            with self.subTest(key=key):
                self.assertNotIn(key, keys)
                self.assertIn(key, RETIRED_KEYS, "은퇴한 키는 이유와 함께 RETIRED_KEYS 에 남긴다")
                self.assertNotIn(key, PromptSettingsExtras().to_settings())


if __name__ == "__main__":
    unittest.main()
