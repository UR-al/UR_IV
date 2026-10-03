"""Anima SPEED 카드(frontend/src/utils/animaSpeed.ts · components/params/AnimaSpeedCard.vue)는 core/anima_speed 의 거울이다.

위젯 키·앱 기본값(문자열과 값)·선택지·프리셋 A·δ/σ 나눔값 범위(= 확장 슬라이더 — 픽스처와도 직접 대조)·검사 문구를 파이썬
표와 대조한다(하나라도 다르면 카드가 파이썬 프록시와 다른 값을 쓰거나, Forge 가 거절할 설정을 괜찮다고 보인다). 카드 값
정규화와 설정 검사는 두 구현이 같은 골든 파일(utils/animaSpeed.cases.json — 파이썬이 만든 기대값)을 읽는다 — 여기서는
파이썬 쪽, vitest 가 TS 쪽.
"""
from __future__ import annotations

import dataclasses
import json
import re
import unittest
from pathlib import Path

from core import anima_speed as asp
from core.sam_extra_capabilities import FEATURE_FLAGS

REPO = Path(__file__).resolve().parent.parent
SRC = REPO / "frontend" / "src"
TS = SRC / "utils" / "animaSpeed.ts"
CASES = SRC / "utils" / "animaSpeed.cases.json"
FIXTURE = REPO / "tests" / "fixtures" / "sam_extra_script_info.json"
CARD = SRC / "components" / "params" / "AnimaSpeedCard.vue"
APP = SRC / "App.vue"


def _block(text: str, head: str) -> str:
    match = re.search(re.escape(head) + r"(.*?)\n[}\]]", text, re.S)
    if match is None:
        raise AssertionError(f"{head!r} 블록을 찾지 못했다")
    return match.group(1)


def _range(text: str, name: str) -> dict:
    match = re.search(rf"export const {name} = \{{ min: (-?[\d.]+), max: (-?[\d.]+), step: (-?[\d.]+) \}} as const",
                      text)
    if match is None:
        raise AssertionError(f"{name} 를 찾지 못했다")
    return {key: float(value) for key, value in zip(("min", "max", "step"), match.groups())}


def _list(text: str, name: str) -> tuple:
    match = re.search(rf"export const {name}: readonly \w+\[\] = \[([^\]]*)\]", text)
    if match is None:
        raise AssertionError(f"{name} 를 찾지 못했다")
    return tuple(re.findall(r"'([^']+)'", match.group(1)))


def _literal(token: str):
    if token.startswith("'"):
        return token[1:-1]
    if token in ("true", "false"):
        return token == "true"
    if token == "null":
        return None
    return float(token)


class AnimaSpeedCardMirrorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.ts = TS.read_text(encoding="utf-8")
        cls.cases = json.loads(CASES.read_text(encoding="utf-8"))

    def test_widget_ids_and_defaults(self):
        ids = dict(re.findall(r"^\s+(\w+): '(_aspeed_\w+)',$", _block(self.ts, "export const WIDGET_IDS"), re.M))
        self.assertEqual(ids, {key: asp.widget_id(key) for key in asp.WIDGET_KEYS})
        self.assertEqual(list(ids), list(asp.WIDGET_KEYS))
        defaults = dict(re.findall(r"(\w+): '([^']*)'",
                                   _block(self.ts, "export const DEFAULTS: Readonly<SpeedValues> = {")))
        self.assertEqual(defaults, asp.widget_values(asp.APP_DEFAULTS))
        settings = dict(re.findall(r"(\w+): ('[^']*'|-?[\d.]+|true|false)",
                                   _block(self.ts, "export const DEFAULT_SETTINGS: Readonly<SpeedSettings> = {")))
        self.assertEqual({key: _literal(value) for key, value in settings.items()},
                         dataclasses.asdict(asp.APP_DEFAULTS))

    def test_choices_and_preset_a(self):
        self.assertEqual(_list(self.ts, "MODES"), asp.MODES)
        self.assertEqual(_list(self.ts, "THRESHOLDS"), asp.THRESHOLDS)
        self.assertEqual(_list(self.ts, "TRANSFORMS"), asp.TRANSFORMS)
        self.assertEqual(_list(self.ts, "PRESET_NAMES"), asp.PRESET_NAMES)
        rows = dict(re.findall(r"'?([\w-]+)'?: (-?[\d.]+|null)",
                               _block(self.ts, "export const PRESET_A: Readonly<Record<string, number | null>> = {")))
        self.assertEqual({name: _literal(value) for name, value in rows.items()},
                         {name: (preset or {}).get("A") for name, preset in asp.PRESETS.items()})
        for name in ("MODE_OPTIONS", "THRESHOLD_OPTIONS", "TRANSFORM_OPTIONS"):
            keys = tuple(re.findall(r"\{ key: '([\w-]+)', label: '[^']+' \}", _block(self.ts, f"export const {name}")))
            self.assertEqual(keys, {"MODE_OPTIONS": asp.MODES, "THRESHOLD_OPTIONS": asp.THRESHOLDS,
                                    "TRANSFORM_OPTIONS": asp.TRANSFORMS}[name])

    def test_ranges_follow_python_and_the_live_sliders(self):
        delta = {"min": asp.DELTA_MIN, "max": asp.DELTA_MAX, "step": asp.DELTA_STEP}
        divisor = {"min": asp.DIVISOR_MIN, "max": asp.DIVISOR_MAX, "step": asp.DIVISOR_STEP}
        self.assertEqual(_range(self.ts, "DELTA_RANGE"), delta)
        self.assertEqual(_range(self.ts, "DIVISOR_RANGE"), divisor)
        scripts = json.loads(FIXTURE.read_text(encoding="utf-8"))["scripts"]
        entries = [s for s in scripts if s["name"].lower() == asp.SCRIPT_NAME.lower()]
        self.assertEqual({s["is_img2img"] for s in entries}, {False, True})
        for entry in entries:
            for index, expected in ((5, delta), (6, divisor)):
                arg = entry["args"][index]
                with self.subTest(img2img=entry["is_img2img"], index=index):
                    self.assertEqual({"min": arg["minimum"], "max": arg["maximum"], "step": arg["step"]}, expected)

    def test_validation_messages_and_codes(self):
        block = _block(self.ts, "export const VALIDATION_MESSAGES: Readonly<Record<ErrorCode, string>> = {")
        messages = dict(re.findall(r"^\s+(\w+): '([^']+)',$", block, re.M))
        self.assertEqual(messages, dict(asp.VALIDATION_MESSAGES))
        union = re.search(r"export type ErrorCode =(.*?)\n\n", self.ts, re.S).group(1)
        self.assertEqual(set(re.findall(r"'(\w+)'", union)), set(asp.VALIDATION_MESSAGES))
        self.assertIn(f"export const INVALID_NOTE = '{asp.INVALID_NOTE}'", self.ts)

    def test_golden_cases_are_what_python_gives(self):
        for item in self.cases["settings"]:
            with self.subTest(case=item["name"]):
                settings = asp.parse_settings(item["widgets"])
                self.assertEqual(dataclasses.asdict(settings), item["settings"])
                self.assertEqual(list(asp.validate(settings)), item["errors"])
        errors = {code for item in self.cases["settings"] for code in item["errors"]}
        self.assertEqual(errors, set(asp.VALIDATION_MESSAGES) - {asp.ERR_CHOICE, asp.ERR_SPECTRUM_BETA},
                         "카드 값으로 날 수 있는 오류 코드를 모두 다룬다(선택지·β 는 정규화가 막는다)")
        self.assertGreaterEqual(len(self.cases["settings"]), 60)

    def test_old_forge_note_reads_the_feature_flag(self):
        self.assertIn("speed", FEATURE_FLAGS)
        self.assertIn("caps.features?.speed !== true", self.ts)

    def test_placement_after_the_anima38_card(self):
        """계획 R8: 파라미터 열에서 Anima38Card 바로 다음, AnimaGuidancePanel 앞(카드가 스토어를 직접 읽는다 —
        App.vue destructure 없음, 새 브리지 액션 없음)."""
        app = APP.read_text(encoding="utf-8")
        a38 = app.index("<Anima38Card />")
        card = app.index("<AnimaSpeedCard />")
        guidance = app.index("<AnimaGuidancePanel")
        self.assertLess(a38, card)
        self.assertLess(card, guidance)
        self.assertEqual(app[a38:card].count("<"), app[a38:card].count("<!--") + 1, "Anima38Card 바로 다음이어야 한다")
        self.assertEqual(app.count("<AnimaSpeedCard"), 1)
        self.assertIn("import AnimaSpeedCard from './components/params/AnimaSpeedCard.vue'", app)
        card_src = CARD.read_text(encoding="utf-8")
        for name in ("WIDGET_IDS", "DEFAULTS", "DELTA_RANGE", "DIVISOR_RANGE", "readSettings", "validationMessages",
                     "visibleFields", "cardNote", "INVALID_NOTE"):
            self.assertIn(name, card_src)
        self.assertNotIn("requestAction", card_src, "새 브리지 액션 없음 — 값은 위젯 스토어로만")
        self.assertIn('<script setup lang="ts">', card_src)


if __name__ == "__main__":
    unittest.main()
