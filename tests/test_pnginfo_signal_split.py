"""PNG Info '열기'와 인페인트 전송의 시그널 분리 (#11).

예전엔 open_png_info_file 이 inpaintImageLoaded 를 emit 해서, keep-alive 로 살아 있는
InpaintView 가 PNG Info 에서 파일을 열 때마다 원본·마스크·undo 를 날렸고, 반대로 인페인트로
보낼 때마다 PngInfoView 화면이 바뀌었다.
"""
from __future__ import annotations

import re
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / 'frontend' / 'src'


def _read(*parts: str) -> str:
    return ROOT.joinpath(*parts).read_text(encoding='utf-8')


def _action_block(source: str, action: str) -> str:
    start = source.index(f"elif action == '{action}':")
    return source[start:source.index('elif action', start + 10)]


class SignalSplitTests(unittest.TestCase):
    def test_python_emitters(self):
        main = _read('ui', 'generator_main.py')
        png_block = _action_block(main, 'open_png_info_file')
        self.assertIn('pngInfoImageLoaded.emit', png_block)
        self.assertNotIn('inpaintImageLoaded', png_block)
        # 인페인트 전송은 그대로 inpaintImageLoaded
        self.assertRegex(main, r"send_to_inpaint'[\s\S]{0,400}inpaintImageLoaded\.emit")

    def test_bridge_declares_both_signals(self):
        bridge = _read('ui', 'vue_bridge.py')
        self.assertRegex(bridge, r'(?m)^\s*pngInfoImageLoaded = pyqtSignal\(str\)')
        self.assertRegex(bridge, r'(?m)^\s*inpaintImageLoaded = pyqtSignal\(str\)')

    def test_each_view_listens_only_to_its_own_signal(self):
        png = (SRC / 'views' / 'PngInfoView.vue').read_text(encoding='utf-8')
        inpaint = (SRC / 'views' / 'InpaintView.vue').read_text(encoding='utf-8')
        listened = re.compile(r"onBackendEvent\(\s*'(pngInfoImageLoaded|inpaintImageLoaded)'")
        self.assertEqual(listened.findall(png), ['pngInfoImageLoaded'])
        self.assertEqual(listened.findall(inpaint), ['inpaintImageLoaded'])

    def test_web_mode_whitelists_both(self):
        import web_main_ui
        self.assertIn('pngInfoImageLoaded', web_main_ui._WEB_SIGNALS)
        self.assertIn('inpaintImageLoaded', web_main_ui._WEB_SIGNALS)
        self.assertIn('batchJobState', web_main_ui._WEB_SIGNALS)

    def test_dev_harnesses_follow_the_split(self):
        creator = (ROOT / 'frontend' / 'dev' / 'creator-integration.js').read_text(encoding='utf-8')
        audit = (ROOT / 'frontend' / 'dev' / 'theme-audit.js').read_text(encoding='utf-8')
        self.assertIn("'pngInfoImageLoaded'", creator)
        self.assertIn("backend.pngInfoImageLoaded.emit", creator)
        self.assertIn("emit('pngInfoImageLoaded', sampleImage)", audit)


if __name__ == '__main__':
    unittest.main()
