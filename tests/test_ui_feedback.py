"""Offline regressions for visible UI feedback; no GPU jobs or real clipboard writes."""
import json
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from ui.generator_main import GeneratorMainUI
from ui.vue_bridge import VueBridge


class UIFeedbackTests(unittest.TestCase):
    # 런타임 설치/확장·Forge 모델 폴더 선택기는 Studio native.pick_directory 로만 연다 —
    # 현재 경로 전달·취소 계약은 tests/test_studio_application.py
    # test_directory_picker_uses_exact_purpose_selector_and_current_contract 가 검증한다.

    def test_vram_poll_does_not_overlap_slow_reads_and_recovers_after_error(self):
        started = []
        def thread_factory(*, target, daemon):
            return SimpleNamespace(start=lambda: started.append(target))
        host = SimpleNamespace(vue_bridge=SimpleNamespace(vramUpdated=Mock()))
        with patch('threading.Thread', side_effect=thread_factory):
            GeneratorMainUI._update_vram_status(host)
            GeneratorMainUI._update_vram_status(host)
            self.assertEqual(len(started), 1, '1-second polls must not pile up slow workers')
            with patch('core.gpu_stats.read_vram', side_effect=RuntimeError('offline')):
                started.pop()()
            GeneratorMainUI._update_vram_status(host)
            self.assertEqual(len(started), 1)
            with patch('core.gpu_stats.read_vram', return_value={'vram_used': 2**30, 'vram_total': 8 * 2**30, 'source': 'fake'}):
                started.pop()()
        data = json.loads(host.vue_bridge.vramUpdated.emit.call_args.args[0])
        self.assertEqual(data['used'], 1)
        self.assertEqual(data['total'], 8)

    def test_desktop_text_clipboard_reports_success_only_after_write(self):
        host = SimpleNamespace(_backend_runtime_is_web_mode=lambda: False)
        clipboard = Mock()
        clipboard.text.return_value = 'prompt\nnegative'
        with patch('PyQt6.QtWidgets.QApplication.clipboard', return_value=clipboard):
            self.assertTrue(VueBridge.copyTextToClipboard(host, 'prompt\nnegative'))
        clipboard.setText.assert_called_once_with('prompt\nnegative')

    def test_remote_web_cannot_touch_host_clipboard(self):
        host = SimpleNamespace(_backend_runtime_is_web_mode=lambda: True)
        with patch('PyQt6.QtWidgets.QApplication.clipboard') as clipboard:
            self.assertFalse(VueBridge.copyTextToClipboard(host, 'remote text'))
        clipboard.assert_not_called()

    def test_failed_clipboard_write_is_not_reported_as_success(self):
        host = SimpleNamespace(_backend_runtime_is_web_mode=lambda: False)
        clipboard = Mock()
        clipboard.text.return_value = 'unchanged'
        with patch('PyQt6.QtWidgets.QApplication.clipboard', return_value=clipboard):
            self.assertFalse(VueBridge.copyTextToClipboard(host, 'new text'))


if __name__ == '__main__':
    unittest.main()
