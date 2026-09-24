"""T2I 생성 family 콤보(Standard/Krea2) — Vue 라벨과 Python ComboBoxProxy 항목 정합성.

예전 버그: Vue 옵션은 'Standard'/'Krea2', Python 항목은 'STANDARD'/'KREA2'. ComboBoxProxy 가
대소문자를 구분해 Vue 의 'Krea2' 를 버렸고, 화면은 Krea2(steps 8·CFG 1)인데 숨은 Standard
체크포인트로 생성됐다. 두 목록이 글자 그대로 같은지와 실제 프록시 경로를 고정한다.
"""
from __future__ import annotations

import os
import re
import unittest

from core.generation_family import GENERATION_FAMILY_ITEMS, is_krea2_family
from ui.generator_generation import GenerationMixin
from ui.widget_proxies import ComboBoxProxy

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FRONT_UTIL = os.path.join(ROOT, 'frontend', 'src', 'utils', 'generationFamily.ts')
PROMPT_PANEL = os.path.join(ROOT, 'frontend', 'src', 'components', 'PromptPanel.vue')
UI_SETUP = os.path.join(ROOT, 'ui', 'generator_ui_setup.py')


def _read(path: str) -> str:
    with open(path, encoding='utf-8') as f:
        return f.read()


class _FakeBridge:
    def __init__(self):
        self.pushed = []

    def _register_proxy(self, widget_id, proxy):
        pass

    def pushWidgetProperty(self, widget_id, prop, value):
        self.pushed.append((widget_id, prop, value))

    def pushWidgetValue(self, widget_id, value):
        self.pushed.append((widget_id, 'value', value))


class _Host(GenerationMixin):
    def __init__(self, combo):
        self.generation_family_combo = combo


class GenerationFamilyContractTests(unittest.TestCase):
    def test_vue_labels_equal_python_items_exactly(self):
        match = re.search(r"GENERATION_FAMILY_ITEMS\s*=\s*\[([^\]]*)\]", _read(FRONT_UTIL))
        self.assertIsNotNone(match, 'generationFamily.ts 의 GENERATION_FAMILY_ITEMS 를 찾지 못했습니다')
        vue_labels = tuple(re.findall(r"['\"]([^'\"]+)['\"]", match.group(1)))
        # ComboBoxProxy 는 대소문자까지 정확히 비교하므로 upper 집합이 아니라 글자 그대로 같아야 한다.
        self.assertEqual(vue_labels, GENERATION_FAMILY_ITEMS)

    def test_prompt_panel_and_ui_setup_use_the_shared_lists(self):
        panel = _read(PROMPT_PANEL)
        self.assertIn('GENERATION_FAMILY_ITEMS', panel)
        self.assertIn(':options="generationFamilyItems"', panel)
        # PromptPanel 이 자기만의 라벨 목록을 다시 하드코딩하면 드리프트가 생긴다.
        self.assertNotRegex(panel, r"generationFamilyItems\s*=\s*\[\s*['\"]")
        setup = _read(UI_SETUP)
        self.assertIn('generation_family_combo.addItems(list(GENERATION_FAMILY_ITEMS))', setup)
        self.assertNotIn('"KREA2"]', setup)

    def test_mount_restore_never_overwrites_live_krea2_steps(self):
        # Python 이 새로고침을 넘어 Krea2 를 유지하면 steps/CFG 는 사용자가 맞춘 현재값이다.
        # 마운트 복원은 8/1 을 직접 쓰지 않는다(기본값은 Standard→Krea2 전환 watch 만 적용).
        panel = _read(PROMPT_PANEL)
        restore = panel.split('const restore = () => {', 1)
        self.assertEqual(len(restore), 2, 'PromptPanel 의 family 복원 블록을 찾지 못했습니다')
        body = restore[1].split('\n    restore()', 1)[0]
        self.assertNotIn("steps_input = '8'", body)
        self.assertNotIn("cfg_input = '1'", body)
        self.assertNotIn('apply-krea2', panel)
        # 저장된 선택이 없는 새 브라우저는 Python 콤보를 그대로 둔다
        self.assertIn('storedFamilyLabel(window.localStorage.getItem(generationFamilyStorageKey))', panel)
        self.assertRegex(body, r"if \(family === null\) return")

    def test_krea2_selected_in_vue_reaches_python_generation_check(self):
        combo = ComboBoxProxy(_FakeBridge(), 'generation_family_combo')
        combo.addItems(list(GENERATION_FAMILY_ITEMS))
        host = _Host(combo)
        self.assertEqual(combo.currentText(), 'Standard')
        self.assertFalse(host._is_krea2_generation())

        combo._on_vue_changed('Krea2')   # Vue CustomSelect 가 보내는 라벨 그대로
        self.assertEqual(combo.currentText(), 'Krea2')
        self.assertTrue(host._is_krea2_generation())

        combo._on_vue_changed('Standard')
        self.assertFalse(host._is_krea2_generation())

    def test_is_krea2_family_ignores_case_for_stored_values(self):
        self.assertTrue(is_krea2_family('KREA2'))
        self.assertTrue(is_krea2_family(' krea2 '))
        self.assertTrue(is_krea2_family('Krea2'))
        self.assertFalse(is_krea2_family('Standard'))
        self.assertFalse(is_krea2_family(None))


if __name__ == '__main__':
    unittest.main()
