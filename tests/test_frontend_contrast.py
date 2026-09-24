"""Frontend contrast and tooltip accessibility regression tests."""

from __future__ import annotations

import re
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
STYLE = (ROOT / "frontend" / "src" / "style.css").read_text(encoding="utf-8")
# 색 토큰은 `core/theme_presets.py` 로 이사했고 style.css 는 치수/타입만 남았다.
# 여기서 보는 건 그 생성물(기본 프리셋의 폴백) — 스크립트가 죽어도 실제로 화면에
# 깔리는 값이라, 이 검사가 지키려던 "기본 화면의 대비"의 뜻은 그대로다.
# 프리셋 3종 전체의 대비는 `tests/test_theme_contract.py` 가 본다.
TOKENS = (ROOT / "frontend" / "src" / "styles" / "theme-fallback.css").read_text(encoding="utf-8")
APP = (ROOT / "frontend" / "src" / "App.vue").read_text(encoding="utf-8")
ANIMA_PANEL = (ROOT / "frontend" / "src" / "components" / "AnimaGuidancePanel.vue").read_text(encoding="utf-8")


def _css_hex_variable(name: str) -> str:
    pattern = rf"{re.escape(name)}\s*:\s*(#[0-9a-fA-F]{{6}})"
    match = re.search(pattern, TOKENS) or re.search(pattern, STYLE)
    assert match, f"missing CSS variable: {name}"
    return match.group(1)


def _relative_luminance(hex_color: str) -> float:
    channels = [int(hex_color[i : i + 2], 16) / 255 for i in (1, 3, 5)]
    linear = [channel / 12.92 if channel <= 0.04045 else ((channel + 0.055) / 1.055) ** 2.4 for channel in channels]
    return 0.2126 * linear[0] + 0.7152 * linear[1] + 0.0722 * linear[2]


def _contrast(foreground: str, background: str) -> float:
    high, low = sorted((_relative_luminance(foreground), _relative_luminance(background)), reverse=True)
    return (high + 0.05) / (low + 0.05)


class FrontendContrastTests(unittest.TestCase):
    def test_muted_text_remains_readable_on_cards(self) -> None:
        ratio = _contrast(_css_hex_variable("--text-muted"), _css_hex_variable("--bg-card"))
        self.assertGreaterEqual(ratio, 4.5, f"muted text contrast is only {ratio:.2f}:1")

    def test_disabled_buttons_keep_a_visible_opacity_floor(self) -> None:
        match = re.search(r"button:disabled\s*\{[^}]*opacity\s*:\s*([0-9.]+)\s*!important", STYLE, re.DOTALL)
        self.assertIsNotNone(match, "global disabled-button visibility floor is missing")
        self.assertGreaterEqual(float(match.group(1)), 0.6)

    def test_app_installs_theme_aware_tooltips(self) -> None:
        self.assertIn("import AppTooltip from './components/AppTooltip.vue'", APP)
        self.assertIn("<AppTooltip />", APP)
        tooltip = ROOT / "frontend" / "src" / "components" / "AppTooltip.vue"
        self.assertTrue(tooltip.is_file())
        source = tooltip.read_text(encoding="utf-8")
        self.assertIn("removeAttribute('title')", source)
        self.assertIn("role=\"tooltip\"", source)

    def test_anima_dropdown_uses_compact_panel_font_size(self) -> None:
        self.assertRegex(
            ANIMA_PANEL,
            r":deep\(\.csel-display\)\s*\{[^}]*font-size\s*:\s*11px",
        )
        # 타입 스케일의 하한은 `--fs-label`(11px)이다. 예전에는 10px 리터럴이었는데
        # 탭 전체 규격 정리에서 11px 로 올렸다 — 이 검사의 뜻(Anima 패널은 조밀한
        # 라벨 크기를 쓴다)은 그대로고, 값만 토큰으로 옮겼다.
        self.assertRegex(
            ANIMA_PANEL,
            r"\.ext-note\s*\{[^}]*font-size\s*:\s*var\(--fs-label\)",
        )

    def test_anima_panel_exposes_forge_import_action(self) -> None:
        self.assertIn("requestAction('import_anima_from_forge')", ANIMA_PANEL)

    def test_anima_panel_exposes_current_forge_smc_controls(self) -> None:
        self.assertIn("b('guid_smc_master_enabled')", ANIMA_PANEL)
        self.assertIn('v-model="w._guid_smc_preset"', ANIMA_PANEL)
        self.assertIn("'Cosmos / Wan'", ANIMA_PANEL)
        self.assertIn("w._guid_smc_preset === 'Custom'", ANIMA_PANEL)

    def test_anima_panel_exposes_current_forge_rdc_controls(self) -> None:
        # 실제 컨트롤 바인딩을 본다 — 예전엔 패널의 기본값 사본(DEFAULTS)에 있는 키 문자열만으로도
        # 통과했다. 기본값은 이제 Python 스펙이 단일 출처다(reset_anima_guidance).
        self.assertIn("b('guid_rdc_enabled')", ANIMA_PANEL)
        for widget_id in ('_guid_rdc_tau', '_guid_rdc_alpha_ll', '_guid_rdc_alpha_hh'):
            self.assertIn(f'v-model="w.{widget_id}"', ANIMA_PANEL)


if __name__ == "__main__":
    unittest.main()
