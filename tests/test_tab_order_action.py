"""set_tab_order 액션 — 설정 화면에서 탭 순서를 바꾸면 하단 상태줄에 한국어 확인 한 줄이 뜬다.

감사 #136: show_status 가 더미 라벨에만 쓰던 시절엔 아무 효과가 없던 분기였다. 지금은
ui.status_line.publish_status 로 Vue StatusStrip 에 보이므로 문구가 한국어 UI 와 맞아야 한다
(예전 'Tab order updated: N tabs'). 저장은 앞서 오는 save_ui_prefs(tabOrder)가 한다.
"""
import unittest
from types import SimpleNamespace

from ui.generator_main import GeneratorMainUI
from ui.model_download_actions import ModelDownloadActionsMixin


class _Harness(ModelDownloadActionsMixin):
    _handle_vue_action = GeneratorMainUI._handle_vue_action

    def __init__(self):
        self.statuses = []
        self.vue_bridge = SimpleNamespace()

    def show_status(self, message):
        self.statuses.append(message)

    def _handle_chat_action(self, _action, _payload):
        return False

    def _handle_creator_action(self, _action, _payload):
        return False


class SetTabOrderActionTests(unittest.TestCase):
    def test_reorder_shows_korean_status_with_tab_count(self):
        host = _Harness()
        host._handle_vue_action("set_tab_order", {"order": ["T2I", "I2I", "Settings"]})
        self.assertEqual(host.statuses, ["탭 순서 저장됨 (3개 탭)"])

    def test_status_text_is_not_english(self):
        host = _Harness()
        host._handle_vue_action("set_tab_order", {"order": ["T2I"]})
        self.assertTrue(host.statuses)
        self.assertNotIn("Tab order", host.statuses[0])
        self.assertIn("탭 순서", host.statuses[0])

    def test_empty_or_malformed_order_shows_nothing(self):
        for payload in ({}, {"order": []}, {"order": None}, {"order": "T2I,I2I"}, {"order": 3}):
            host = _Harness()
            host._handle_vue_action("set_tab_order", payload)
            self.assertEqual(host.statuses, [], payload)


if __name__ == "__main__":
    unittest.main()
