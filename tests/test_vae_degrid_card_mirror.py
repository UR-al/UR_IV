"""VAE DeGrid 카드(frontend/src/utils/vaeDegrid.ts · components/params/VaeDegridCard.vue)는 core/vae_degrid 의 거울이다.

위젯 키·앱 기본값·모드 키와 확장 UI 라벨·짧은 이름·강도/타일 범위·ComfyUI 속성 이름·Forge 옵션 키와 ComfyUI 고정값을
파이썬 표와 대조한다(하나라도 다르면 카드가 파이썬 프록시와 다른 값을 쓰거나 다른 값을 보인다). 카드 요약 문구는
파이썬 ``describe`` 와 같은 문장이어야 한다 — 같은 사례를 양쪽이 같은 문자열로 고정한다. App.vue 배치와 업스케일러
선택 칸(Hires.fix·배치 업스케일) DeGrid 경고의 배선도 본다(새 브리지 액션 없음).
"""
from __future__ import annotations

import re
import unittest
from pathlib import Path

from core import forge_override_settings as fos
from core import vae_degrid as vdg
from ui import vae_degrid_ui

REPO = Path(__file__).resolve().parent.parent
SRC = REPO / "frontend" / "src"
TS = SRC / "utils" / "vaeDegrid.ts"
TS_TEST = SRC / "utils" / "vaeDegrid.test.ts"
CARD = SRC / "components" / "params" / "VaeDegridCard.vue"
HIRES = SRC / "components" / "params" / "HiresFixCard.vue"
BATCH = SRC / "views" / "BatchView.vue"
UPSCALER_WARNING = SRC / "composables" / "useUpscalerDegridWarning.ts"
APP = SRC / "App.vue"


def _block(text: str, head: str) -> str:
    match = re.search(re.escape(head) + r"(.*?)\n[}\]]", text, re.S)
    if match is None:
        raise AssertionError(f"{head!r} 블록을 찾지 못했다")
    return match.group(1)


def _range(text: str, name: str) -> dict:
    match = re.search(rf"export const {name} = \{{ min: ([\d.]+), max: ([\d.]+), step: ([\d.]+) \}} as const", text)
    if match is None:
        raise AssertionError(f"{name} 를 찾지 못했다")
    return {key: float(value) for key, value in zip(("min", "max", "step"), match.groups())}


class VaeDegridCardMirrorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.ts = TS.read_text(encoding="utf-8")
        cls.card = CARD.read_text(encoding="utf-8")

    def test_widget_ids_and_defaults(self):
        ids = dict(re.findall(r"^\s+(\w+): '(_degrid_\w+)',$", _block(self.ts, "export const WIDGET_IDS"), re.M))
        self.assertEqual(ids, {key: vdg.widget_id(key) for key in vdg.WIDGET_KEYS})
        defaults = dict(re.findall(r"(\w+): '([^']*)'", _block(self.ts, "export const DEFAULTS: Readonly<DegridValues> = {")))
        self.assertEqual(defaults, vdg.widget_values(vdg.APP_DEFAULTS))

    def test_modes_labels_and_short_names(self):
        options = tuple(re.findall(r"\{ key: '([^']+)', label: '([^']+)' \}",
                                   _block(self.ts, "export const MODE_OPTIONS: readonly DegridOption[] = [")))
        self.assertEqual(options, tuple(zip(vdg.MODES, vdg.MODE_CHOICES)))
        short = dict(re.findall(r"(\w+): '([^']*)'", _block(self.ts, "export const SHORT_MODE")))
        self.assertEqual(short, dict(vdg.SHORT_MODE))

    def test_ranges(self):
        self.assertEqual(_range(self.ts, "STRENGTH_RANGE"),
                         {"min": vdg.STRENGTH_MIN, "max": vdg.STRENGTH_MAX, "step": vdg.STRENGTH_STEP})
        self.assertEqual(_range(self.ts, "TILE_RANGE"), {"min": 0.0, "max": vdg.MAX_TILE, "step": vdg.TILE_STEP})
        self.assertIn(f"const MIN_TILE = {vdg.MIN_TILE}\n", self.ts)

    def test_names_shared_with_the_python_glue(self):
        self.assertIn(f"export const COMFY_MODELS_PROPERTY = '{vae_degrid_ui.COMFY_MODELS_PROPERTY}'", self.ts)
        self.assertIn(f"export const NONE_NAME = '{vdg.NONE_NAME}'", self.ts)
        keys = re.search(r"export const OPTION_KEYS = \[([^\]]*)\] as const", self.ts).group(1)
        self.assertEqual(tuple(re.findall(r"'([^']+)'", keys)),
                         (vdg.OPT_DEVICE, vdg.OPT_GPU_PRECISION, vdg.OPT_KEEP_LOADED))
        self.assertTrue(set(re.findall(r"'([^']+)'", keys)) <= set(fos.OPTION_KEYS))
        comfy = dict(re.findall(r"(sam3_degrid_\w+): ('[^']*'|true|false)",
                                _block(self.ts, "export const COMFY_OPTIONS")))
        by_key = {vdg.OPT_DEVICE: "device", vdg.OPT_GPU_PRECISION: "precision", vdg.OPT_KEEP_LOADED: "keep_loaded"}
        expected = {}
        for key, name in by_key.items():
            value = vdg.COMFY_OPTIONS[name]
            expected[key] = ("true" if value else "false") if isinstance(value, bool) else f"'{value}'"
        self.assertEqual(comfy, expected)

    def test_summary_wording_equals_python_describe(self):
        """TS summary 와 파이썬 describe 는 같은 문장 — 같은 사례를 vaeDegrid.test.ts 가 같은 문자열로 고정한다."""
        cases = (
            ({"enabled": "true"}, "Full 1 · 512"),
            ({"enabled": "true", "mode": "dark", "strength": "0.8", "tile": "0", "model": "qwenVAEDegridNafnet_v11"},
             "Dark 0.8 · 타일 없음 · qwenVAEDegridNafnet_v11"),
            ({"enabled": "true", "mode": "bright", "strength": "9", "tile": "100"}, "Bright 1.5 · 128"),
        )
        ts_test = TS_TEST.read_text(encoding="utf-8")
        for raw, want in cases:
            with self.subTest(raw=raw):
                self.assertEqual(vdg.describe(vdg.parse_settings(raw)), want)
                literal = want.replace("qwenVAEDegridNafnet_v11", "${V11}")
                self.assertTrue(f"'{want}'" in ts_test or f"`{literal}`" in ts_test, want)

    def test_card_binds_every_widget_and_reads_only_existing_bridge_names(self):
        self.assertIn('<script setup lang="ts">', self.card)
        for key in vdg.WIDGET_KEYS:
            with self.subTest(key=key):
                self.assertRegex(self.card, rf"WIDGET_IDS\.{key}\b|WIDGET_IDS\[key\]")
        self.assertNotIn("requestAction", self.card)                    # 새 브리지 액션 없음
        self.assertNotRegex(self.card, r":disabled=")                    # 칸은 막지 않는다
        self.assertIn("getProperty(WIDGET_IDS.model, COMFY_MODELS_PROPERTY, null)", self.card)
        self.assertIn("backend?.getUiPrefs?.(", self.card)               # 펼칠 때 P10 값을 다시 읽는다
        self.assertIn("useSamExtraCapabilities", self.card)
        self.assertIn('class="ext-sub" @toggle="onOptionsToggle"', self.card)

    def test_app_mounts_the_card_after_sam3_mask(self):
        app = APP.read_text(encoding="utf-8")
        self.assertIn("import VaeDegridCard from './components/params/VaeDegridCard.vue'", app)
        sam3 = app.index("<Sam3MaskCard />")
        degrid = app.index("<VaeDegridCard />")
        self.assertGreater(degrid, sam3)
        between = app[sam3 + len("<Sam3MaskCard />"):degrid]
        self.assertNotRegex(between, r"<[A-Z]\w+ ", "DeGrid 카드는 SAM3 마스크 카드 바로 뒤")
        self.assertEqual(app.count("<VaeDegridCard />"), 1)

    def test_every_upscaler_selector_warns_with_the_shared_rule_on_both_model_lists(self):
        """업스케일러 목록에는 DeGrid NAFNet 도 올라온다(Forge models/ESRGAN·ComfyUI upscale_models). 업스케일러를 고르는
        칸은 모두 같은 composable 로 두 백엔드 목록과 비교한다 — 새 칸이 생기면 이 검사가 잡는다."""
        shared = UPSCALER_WARNING.read_text(encoding="utf-8")
        self.assertIn("upscalerDegridWarning(", shared)
        self.assertIn("capabilities.value", shared)                                    # Forge 스냅샷 목록
        self.assertIn("getProperty(WIDGET_IDS.model, COMFY_MODELS_PROPERTY, null)", shared)   # ComfyUI 목록
        users = {HIRES: "() => storeWidgets.upscaler_combo", BATCH: "() => upscaler.value"}
        for path, getter in users.items():
            with self.subTest(view=path.name):
                text = path.read_text(encoding="utf-8")
                self.assertIn(f"useUpscalerDegridWarning({getter})", text)
                self.assertNotIn("upscalerDegridWarning(", text.replace("useUpscalerDegridWarning(", ""),
                                 "경고 규칙은 composable 한 곳에서만 부른다")
        selectors = re.compile(r'<CustomSelect[^>]*v-model="[^"]*upscaler[^"]*"', re.I)
        for path in sorted(SRC.rglob("*.vue")):
            text = path.read_text(encoding="utf-8")
            if selectors.search(text):
                with self.subTest(selector=path.name):
                    self.assertIn(path, users, "업스케일러 선택 칸은 useUpscalerDegridWarning 으로 DeGrid 경고를 보인다")
                    self.assertRegex(text, r'v-if="(degridWarning|upscalerDegridWarning)"')


if __name__ == "__main__":
    unittest.main()
