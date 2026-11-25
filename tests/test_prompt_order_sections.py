"""프롬프트 섹션 순서 ↔ 입력 칸: 키가 Python·TS·패널에서 같고, 선행/후행 이름이 맞는지."""
import re
import unittest
from pathlib import Path

from core import prompt_order

ROOT = Path(__file__).resolve().parent.parent


class PromptSectionContractTests(unittest.TestCase):
    def test_the_frontend_lists_the_same_sections_in_the_same_default_order(self):
        source = (ROOT / "frontend/src/utils/promptSections.ts").read_text(encoding="utf-8")
        listed = re.search(r"DEFAULT_PROMPT_SECTIONS = \[(.*?)\]", source, re.S)
        self.assertEqual(re.findall(r"'(\w+)'", listed.group(1)), prompt_order.DEFAULT_ORDER)

    def test_the_panel_has_a_slot_per_section(self):
        panel = (ROOT / "frontend/src/components/PromptPanel.vue").read_text(encoding="utf-8")
        for key in prompt_order.DEFAULT_ORDER:
            with self.subTest(key=key):
                self.assertEqual(panel.count(f"<template #{key}>"), 1)

    def test_the_two_boxes_are_called_fixed(self):
        self.assertEqual(prompt_order.SECTION_LABELS["prefix"], "선행 고정 (Prefix)")
        self.assertEqual(prompt_order.SECTION_LABELS["suffix"], "후행 고정 (Suffix)")

    def test_the_labels_reach_the_panel_in_the_saved_order(self):
        labels = prompt_order.order_with_labels()
        self.assertEqual([item["key"] for item in labels], prompt_order.load_order())
        self.assertTrue(all(item["label"] for item in labels))


if __name__ == "__main__":
    unittest.main()
