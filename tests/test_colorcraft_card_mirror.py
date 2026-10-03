"""Colorcraft 카드(frontend/src/utils/colorcraft.ts · colorcraftSchema.generated.ts)는 core/colorcraft 의 거울이다.

* 표: ``colorcraftSchema.generated.ts`` = ``core.colorcraft.schema_ts()`` 바이트 그대로(파이썬이 만든다 — 손으로 고치면
  실패). 카드의 위젯 id·기본 상태 = 파이썬 ``widget_id``·``DEFAULT_STATE``.
* 골든(utils/colorcraft.cases.json — 파이썬이 만든 기대값): 설정(켬·마스크·상태 문자열)마다 ●/○ 표시·요약 글자
  (``overview``)·패스별 체인 태그(``build_chain``)·infotext 문자열(``to_infotext`` — 카드의 '현재 값 복사')을 여기서
  다시 계산해 파일과 같아야 하고, vitest 가 TS 쪽을 같은 파일로
  본다. 스케줄 미리보기 기대값은 설치된 확장의 ``make_schedule``(업스트림 MIT 함수 그대로)이 만든 값이다.
* 상태 문자열 골든(tests/fixtures/colorcraft_state_golden.json)은 tests/test_colorcraft_core.py 가 파이썬, vitest 가 TS.
"""
from __future__ import annotations

import importlib
import json
import re
import sys
import unittest
from pathlib import Path

from core import colorcraft as cc
from tests._sam_extra_ext import EXT_ROOT, requires_extension

REPO = Path(__file__).resolve().parent.parent
SRC = REPO / "frontend" / "src"
TS = SRC / "utils" / "colorcraft.ts"
SCHEMA_TS = REPO / cc.SCHEMA_TS_PATH
CASES = SRC / "utils" / "colorcraft.cases.json"
CORPUS_CASES = 60          # 골든에 넣는 말뭉치 설정 수(손으로 고른 것 포함, tests/_colorcraft_corpus.py 앞에서부터)


# ── 골든 입력·기대값(A-CC7 생성기와 이 시험이 같은 함수를 쓴다) ─────────────────────────
def _config(enabled=True, masking=False, **paths) -> cc.Config:
    config = cc.config_from_mapping({key.replace("__", "."): value for key, value in paths.items()})
    config.enabled, config.masking = enabled, masking
    return config


def _hand_picked() -> list:
    design = _config(masking=True, I__kind="Luma", I__mask="M1", I__exposure=0.3, I__tone_compression=-0.2,
                     III__active=True, III__kind="Punch", III__pass="Hires", III__clarity=0.5, III__sharpness=0.25,
                     M1__mask_mode="lows", M1__mask_hardness=2.0, M1__blur=8.0)
    return [
        ("off defaults", cc.default_config()),
        ("on defaults (nothing changes)", _config()),
        ("one exposure", _config(I__exposure=0.3)),
        ("design 4.1", design),
        ("design 4.1 off", _config(enabled=False, masking=True, I__kind="Luma", I__exposure=0.3)),
        ("edited but inactive", _config(I__exposure=0.3, II__contrast=1.0, IV__pass="Hires")),
        ("hires only", _config(I__pass="Hires", I__exposure=0.3)),
        ("both passes", _config(I__pass="Both", I__clarity=0.5)),
        ("incomplete combo", _config(masking=True, I__mask="C1", I__exposure=0.3, C1__mask_a="M1")),
        ("chained combos", _config(masking=True, I__mask="C2", I__exposure=0.3, C1__mask_a="M1", C1__mask_b="M2",
                                   C2__mask_a="C1", C2__mask_b="M3", C2__operation="subtract")),
        ("masking on, nothing masked", _config(masking=True, I__exposure=0.3, M4__blur=2.0)),
        ("basic ignores its mask", _config(masking=True, I__kind="Basic", I__mask="M1", I__contrast=0.5)),
        ("shift with a mask needs the basis", _config(masking=True, I__kind="Shift", I__mask="M2",
                                                       I__color_shift_amount=1.0)),
        ("shift without a mask", _config(I__kind="Shift", I__color_shift_amount=1.0)),
        ("gates off make it a no-op", _config(I__more_colors=False, I__color_shift=False, I__lab_a=1.0,
                                              I__color_shift_amount=1.0)),
        ("zero strength", _config(I__strength=0.0, I__exposure=0.3)),
        ("zero strength with an offset", _config(I__strength=0.0, I__advanced=True, I__start_off=0.2,
                                                 I__exposure=0.3)),
        ("offset without advanced", _config(I__strength=0.0, I__start_off=0.2, I__exposure=0.3)),
        ("all ten", _config(**{f"{tag}__active": True for tag in cc.MODIFIER_TAGS},
                            **{f"{tag}__kind": "Punch" for tag in cc.MODIFIER_TAGS},
                            **{f"{tag}__clarity": 0.1 * (i + 1) for i, tag in enumerate(cc.MODIFIER_TAGS)})),
        ("I off, V on", _config(I__active=False, V__active=True, V__kind="Chroma Plus", V__lab_b=-0.5)),
    ]


def config_inputs() -> list:
    """``[(이름, 켬, 마스크, 상태 문자열), …]`` — 손으로 고른 것 + 말뭉치 앞 ``CORPUS_CASES`` 개."""
    from tests._colorcraft_corpus import corpus

    out = [(name, bool(c.enabled), bool(c.masking), cc.dump_state(c)) for name, c in _hand_picked()]
    for i, config in enumerate(corpus(count=CORPUS_CASES)[:CORPUS_CASES]):
        out.append((f"corpus {i}", bool(config.enabled), bool(config.masking), cc.dump_state(config)))
    return out


def _chain(config: cc.Config, is_hires: bool) -> dict:
    chain = cc.build_chain(config, is_hires)
    return {"tags": list(chain.tags), "unresolved": list(chain.unresolved_masks), "needsBasis": chain.needs_basis}


def case_expectations(enabled: bool, masking: bool, state: str) -> dict:
    """카드가 그 위젯 값으로 보여야 하는 것 — 파이썬 ``overview``·``build_chain``."""
    config = cc.config_from_state(enabled, masking, state)
    view = cc.overview(config)
    return {
        "overview": {"modifierMarks": [list(item) for item in view.modifier_marks],
                     "maskMarks": [list(item) for item in view.mask_marks], "text": view.text},
        "base": _chain(config, False),
        "hires": _chain(config, True),
        "infotext": cc.to_infotext(config),
    }


def _ext_schedule():
    root = str(EXT_ROOT)
    if root not in sys.path:
        sys.path.append(root)
    return importlib.import_module("sam3ext.colorcraft.schedule")


def schedule_expectation(modifier: dict, steps: int):
    """수정자 스케줄 칸 → 확장 ``make_schedule``(업스트림 그대로) 값. 시작·끝·치우침이 0~1 밖이거나 스텝이 없으면
    None(카드가 미리보기를 그리지 않는 범위 — 확장은 그 값도 쓴다)."""
    schedule = cc.schedule_of(modifier)
    if steps < 1 or any(not 0 <= schedule[name] <= 1 for name in ("start", "end", "bias")):
        return None
    values = _ext_schedule().make_schedule(steps, schedule["start"], schedule["end"], schedule["bias"],
                                           schedule["strength"], schedule["exponent"], schedule["start_off"],
                                           schedule["end_off"], schedule["smooth"])
    return [float(v) for v in values]


class SchemaMirrorTests(unittest.TestCase):
    def test_generated_schema_is_what_python_writes(self):
        self.assertEqual(SCHEMA_TS.read_text(encoding="utf-8"), cc.schema_ts(),
                         "colorcraftSchema.generated.ts 가 낡았다 — "
                         'venv\\Scripts\\python.exe -c "from core import colorcraft; colorcraft.write_schema_ts()"')
        self.assertNotIn(b"\r\n", SCHEMA_TS.read_bytes())

    def test_widget_ids_and_default_state(self):
        ts = TS.read_text(encoding="utf-8")
        ids = dict(re.findall(r"(\w+): '(_cc_\w+)'", ts))
        self.assertEqual(ids, {key: cc.widget_id(key) for key in cc.WIDGET_KEYS})
        self.assertIn(f"export const DEFAULT_STATE = '{cc.DEFAULT_STATE}'", ts)
        self.assertIn(f"export const INFOTEXT_VERSION = '{cc.VERSION}'", ts)


class PlacementTests(unittest.TestCase):
    def test_card_sits_after_the_guidance_panel(self):
        """계획 R8: 파라미터 열에서 AnimaGuidancePanel 바로 다음, LoraStackCard 앞(카드가 스토어를 직접 읽는다 —
        App.vue destructure 없음)."""
        app = (SRC / "App.vue").read_text(encoding="utf-8")
        guidance = app.index("<AnimaGuidancePanel")
        card = app.index("<ColorcraftCard />")
        lora = app.index("<LoraStackCard")
        self.assertLess(guidance, card)
        self.assertLess(card, lora)
        between = app[app.index("/>", guidance) + 2:card]
        self.assertEqual(between.count("<"), between.count("<!--"), "AnimaGuidancePanel 바로 다음이어야 한다")
        self.assertEqual(app.count("<ColorcraftCard"), 1)
        self.assertIn("import ColorcraftCard from './components/params/ColorcraftCard.vue'", app)
        card_src = (SRC / "components" / "params" / "ColorcraftCard.vue").read_text(encoding="utf-8")
        self.assertIn('<script setup lang="ts">', card_src)
        self.assertIn("useWidgetStore()", card_src)
        self.assertNotIn("requestAction", card_src.split("<script")[0])


class GoldenCaseTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cases = json.loads(CASES.read_text(encoding="utf-8"))

    def test_config_cases_are_what_python_gives(self):
        inputs = config_inputs()
        self.assertEqual([(c["name"], c["enabled"], c["masking"], c["state"]) for c in self.cases["configs"]],
                         [tuple(item) for item in inputs])
        for case in self.cases["configs"]:
            with self.subTest(case=case["name"]):
                expected = case_expectations(case["enabled"], case["masking"], case["state"])
                self.assertEqual({key: case[key] for key in expected}, expected)

    def test_config_cases_cover_the_rules(self):
        configs = self.cases["configs"]
        marks = {label.split(" ")[-1] for c in configs for label, _tag in c["overview"]["modifierMarks"]
                 + c["overview"]["maskMarks"]}
        self.assertTrue({cc.MARK_ON, cc.MARK_EDITED} <= marks)
        self.assertTrue(any(cc.INCOMPLETE in c["overview"]["text"] for c in configs))
        self.assertTrue(any(c["base"]["unresolved"] for c in configs))
        self.assertTrue(any(c["hires"]["tags"] and not c["base"]["tags"] for c in configs))
        self.assertTrue(any(c["enabled"] and not c["base"]["tags"] and not c["hires"]["tags"] for c in configs))
        self.assertTrue(any(c["base"]["tags"] and not c["base"]["needsBasis"] for c in configs))
        self.assertGreaterEqual(len(configs), 80)

    def test_schedule_case_inputs(self):
        schedules = self.cases["schedules"]
        self.assertGreaterEqual(len(schedules), 50)
        self.assertTrue(any(s["values"] is None for s in schedules))
        for item in schedules:
            with self.subTest(case=item["name"]):
                self.assertEqual(set(item["modifier"]), {"strength", "start", "end", "advanced", "bias", "exponent",
                                                         "start_off", "end_off", "smooth"})
                self.assertEqual(item["values"] is None or len(item["values"]) == item["steps"], True)


@requires_extension
class ScheduleGoldenTests(unittest.TestCase):
    """미리보기 기대값 = 설치된 확장 ``make_schedule``(업스트림 MIT 함수, numpy) — 그 함수가 바뀌면 여기가 실패한다."""

    def test_schedule_cases_are_what_the_extension_gives(self):
        for item in json.loads(CASES.read_text(encoding="utf-8"))["schedules"]:
            with self.subTest(case=item["name"]):
                self.assertEqual(item["values"], schedule_expectation(item["modifier"], item["steps"]))


if __name__ == "__main__":
    unittest.main()
