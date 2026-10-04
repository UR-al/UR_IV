"""Colorcraft(sam-extra) — core/colorcraft 의 앱 쪽 규칙(확장 없이 도는 시험).

* ``JsNumberTests`` — 상태 문자열의 숫자 글자 = JavaScript ``JSON.stringify``(표는 node 로 뽑아 둔 값).
* ``StateGoldenTests`` — tests/fixtures/colorcraft_state_golden.json: 입력 → 정규 상태 문자열 → 경로. Vue
  (utils/colorcraft.ts, 작업 A-CC7)도 같은 파일을 왕복한다.
* ``StateRuleTests`` — 상태 읽기/쓰기 규칙(기본값 생략·경로 순서·-0·모르는 경로·버전·읽을 수 없는 값).
* ``TableTests`` — 표의 내부 일관성(경로 575개·조합 선택지·종류 표). 확장과의 일치는 tests/test_colorcraft_spec_parity.py.
* ``InfotextTests`` — 보내는 문자열(설계 §4.1 의 예시 그대로)·자기 형식 읽기·블록 모양.
* ``ChainTests``·``OverviewTests`` — 체인 항목·패스·no-op·마스크 규칙, 요약 줄과 ●/○ 표시의 예.
* ``SchemaTsTests`` — 생성 TS 의 본문 = ``schema()``(끝 쉼표만 다르다), 쓰기 도우미.
* ``SettingsTests``·``PlanTests`` — 위젯 id·저장 dict·상태 정규화, 보낼지 판정(꺼짐·Krea2·ComfyUI 대기·I2I 토글·돌 항목 없음).
"""
from __future__ import annotations

import json
import math
import os
import unittest

from core import colorcraft as cc

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GOLDEN = os.path.join(ROOT, "tests", "fixtures", "colorcraft_state_golden.json")


def _golden() -> dict:
    with open(GOLDEN, encoding="utf-8") as fh:
        return json.load(fh)


class JsNumberTests(unittest.TestCase):
    # node -e 'JSON.stringify(x)' 로 뽑은 글자(2026-10-04, node 의 Number::toString).
    TABLE = (
        (0.3, "0.3"), (1.0, "1"), (-1.0, "-1"), (-0.0, "0"), (0.0, "0"), (1e-7, "1e-7"), (1.5e-6, "0.0000015"),
        (1e-6, "0.000001"), (123.456, "123.456"), (1e16, "10000000000000000"), (1e21, "1e+21"),
        (1e20, "100000000000000000000"), (1.2345678901234567e20, "123456789012345670000"), (5e-324, "5e-324"),
        (1.7976931348623157e308, "1.7976931348623157e+308"), (0.1 + 0.2, "0.30000000000000004"), (2.5, "2.5"),
        (64.0, "64"), (-4.0, "-4"), (1e22, "1e+22"), (9.999999999999999e20, "999999999999999900000"),
        (0.000012, "0.000012"), (1.25e-7, "1.25e-7"), (-3.5e+25, "-3.5e+25"), (7, "7"),
    )

    def test_table(self):
        for value, text in self.TABLE:
            with self.subTest(value=value):
                self.assertEqual(cc.js_number(value), text)

    def test_round_trip_is_exact(self):
        for value, _ in self.TABLE:
            self.assertEqual(float(cc.js_number(value)), float(value))

    def test_non_finite_is_refused(self):
        for value in (math.inf, -math.inf, math.nan):
            with self.assertRaises(ValueError):
                cc.js_number(value)


class StateGoldenTests(unittest.TestCase):
    def test_every_case(self):
        doc = _golden()
        self.assertEqual(doc["version"], cc.STATE_VERSION)
        self.assertGreaterEqual(len(doc["cases"]), 20)
        for case in doc["cases"]:
            with self.subTest(case=case["name"]):
                self.assertEqual(cc.canonical_state(case["input"]), case["state"])
                self.assertEqual(cc.parse_state(case["state"]), case["paths"])
                # 정규 문자열은 고정점이다.
                self.assertEqual(cc.canonical_state(case["state"]), case["state"])
                # 경로 dict 로 읽어도 같은 상태.
                self.assertEqual(cc.dump_state(cc.config_from_mapping(case["paths"])), case["state"])

    def test_golden_covers_the_rules(self):
        names = {case["name"] for case in _golden()["cases"]}
        for needed in ("empty", "integer_float", "negative_zero", "small_large", "choice_case", "combo_refs",
                       "active_flags", "unknown_and_rev", "out_of_range_kept", "extension_dump"):
            self.assertIn(needed, names)


class StateRuleTests(unittest.TestCase):
    def test_defaults_give_empty_state(self):
        self.assertEqual(cc.dump_state(cc.default_config()), "")
        self.assertEqual(cc.parse_state(""), {})
        self.assertEqual(cc.parse_state(None), {})
        self.assertEqual(cc.parse_state("   "), {})

    def test_enabled_and_masking_are_not_in_the_state(self):
        config = cc.default_config()
        config.enabled = True
        config.masking = True
        self.assertEqual(cc.dump_state(config), "")
        back = cc.config_from_state("true", "1", "")
        self.assertTrue(back.enabled)
        self.assertTrue(back.masking)
        self.assertEqual(cc.config_values(back)[2:], cc.config_values(cc.default_config())[2:])

    def test_blank_config_writes_modifier_one_off(self):
        self.assertEqual(cc.dump_state(cc.blank_config()), '{"v":1,"I.active":false}')

    def test_paths_follow_arg_names_order(self):
        config = cc.default_config()
        config.combos[4]["operation"] = "or"
        config.leaves[0]["blur"] = 2.0
        config.modifiers[9]["exposure"] = 1.0
        config.modifiers[0]["exposure"] = 0.5
        self.assertEqual(cc.dump_state(config),
                         '{"v":1,"I.exposure":0.5,"X.exposure":1,"M1.blur":2,"C5.operation":"or"}')

    def test_negative_zero_is_zero(self):
        config = cc.default_config()
        config.modifiers[0]["exposure"] = -0.0          # = 기본값 0
        config.modifiers[0]["bias"] = -0.0              # 기본값 0.5 와 다름 → 0 으로 쓴다
        self.assertEqual(cc.dump_state(config), '{"v":1,"I.bias":0}')
        self.assertEqual(repr(cc.config_from_state(False, False, cc.dump_state(config)).modifiers[0]["bias"]), "0.0")

    def test_state_paths_hold_plain_zero(self):
        config = cc.default_config()
        config.modifiers[0]["bias"] = -0.0
        self.assertEqual(repr(cc.state_paths(config)["I.bias"]), "0.0")

    def test_mapping_starts_from_the_panel_defaults_and_switches_on(self):
        config = cc.config_from_mapping({})
        self.assertTrue(config.enabled)                  # dict 를 보냄 = Colorcraft 를 원함
        self.assertFalse(config.masking)
        self.assertEqual(cc.config_values(config)[2:], cc.config_values(cc.default_config())[2:])
        self.assertFalse(cc.config_from_mapping({"enabled": "off", "masking": None}).enabled)
        self.assertTrue(cc.config_from_mapping({"enabled": None}).enabled)
        self.assertTrue(cc.config_from_mapping({"masking": "yes"}).masking)

    def test_dump_coerces_what_it_writes(self):
        config = cc.default_config()
        config.modifiers[1]["kind"] = "punch"
        config.modifiers[1]["exposure"] = "0.75"
        config.modifiers[1]["smooth"] = "off"
        config.combos[0]["mask_a"] = "C3"                # C1 은 조합을 고르지 못한다 → 기본값
        del config.leaves[0]["blur"]                     # 빠진 칸 = 기본값
        self.assertEqual(cc.dump_state(config), '{"v":1,"II.kind":"Punch","II.smooth":false,"II.exposure":0.75}')

    def test_version_must_be_one(self):
        for bad in ('{"v":2}', '{"v":true}', '{"v":"1"}', '{"v":1.5}'):
            with self.subTest(state=bad), self.assertRaises(cc.StateError):
                cc.parse_state(bad)
        self.assertEqual(cc.parse_state('{"v":1.0,"I.exposure":1}'), {"I.exposure": 1})
        self.assertEqual(cc.parse_state('{"I.exposure":1}'), {"I.exposure": 1})       # v 없음 = 1

    def test_unreadable_state(self):
        for bad in ("{", "{]", "{[1]", " {1:2} ", "[1]", '"x"', "v1;mods=I", 5, ["I.exposure"], True):
            with self.subTest(state=bad), self.assertRaises(cc.StateError):
                cc.parse_state(bad)

    def test_dict_state_is_read_like_the_string(self):
        self.assertEqual(cc.canonical_state({"v": 1, "I.exposure": 0.3, "rev": "x"}), '{"v":1,"I.exposure":0.3}')

    def test_huge_integer_falls_back_to_default(self):
        # 확장 spec.coerce 는 float(10**400) 의 OverflowError 를 그대로 올린다 — 앱은 기본값(모듈 설명).
        self.assertEqual(cc.canonical_state('{"v":1,"I.exposure":' + "9" * 400 + "}"), "")
        self.assertEqual(cc.canonical_state('{"v":1,"I.exposure":1e400}'), "")


class TableTests(unittest.TestCase):
    def test_state_paths(self):
        self.assertEqual(len(cc.STATE_PATHS), 575)
        self.assertEqual(len(set(cc.STATE_PATHS)), 575)
        self.assertEqual(cc.STATE_PATHS[0], "I.active")
        self.assertEqual(cc.STATE_PATHS[-1], "C5.normalize")
        self.assertEqual(cc.arg_names()[:3], ("enabled", "masking", "I.active"))
        self.assertEqual(len(cc.config_values(cc.default_config())), 577)

    def test_field_counts_and_unique_names(self):
        self.assertEqual((len(cc.MODIFIER_FIELDS), len(cc.LEAF_FIELDS), len(cc.COMBO_FIELDS)), (44, 10, 7))
        for fields in (cc.MODIFIER_FIELDS, cc.LEAF_FIELDS, cc.COMBO_FIELDS):
            names = [f.name for f in fields]
            self.assertEqual(len(names), len(set(names)))

    def test_field_defaults_are_their_own_kind_and_in_choices(self):
        for f in cc.MODIFIER_FIELDS + cc.LEAF_FIELDS + cc.COMBO_FIELDS + cc.GLOBAL_FIELDS:
            with self.subTest(field=f.name):
                if f.kind == cc.FLOAT:
                    self.assertIsInstance(f.default, float)
                    self.assertLessEqual(f.minimum, f.default)
                    self.assertLessEqual(f.default, f.maximum)
                    self.assertIn(f.step, (0.01, 0.1))
                elif f.kind == cc.BOOL:
                    self.assertIsInstance(f.default, bool)
                else:
                    self.assertIn(f.default, f.choices)
                self.assertTrue(f.label and f.title)

    def test_combo_ref_choices(self):
        self.assertEqual(cc.combo_ref_choices(0), ("none",) + cc.MASK_TAGS)
        self.assertEqual(cc.combo_ref_choices(2), ("none",) + cc.MASK_TAGS + ("C1", "C2"))
        self.assertEqual(cc.node_choices("C2", cc.COMBO_BY_NAME["mask_a"]), cc.combo_ref_choices(1))
        self.assertEqual(cc.node_choices("C2", cc.COMBO_BY_NAME["operation"]), cc.COMBO_OP_OPTIONS)

    def test_kind_tables_name_known_fields(self):
        self.assertEqual(set(cc.KIND_PARAMS), set(cc.KINDS))
        self.assertEqual(set(cc.KIND_GROUPS), set(cc.KINDS))
        for kind, pairs in cc.KIND_PARAMS.items():
            for _, panel_name in pairs:
                self.assertIn(panel_name, cc.MODIFIER_BY_NAME, kind)
            for group in cc.KIND_GROUPS[kind]:
                self.assertIn(group, cc.GROUP_ORDER)
        for kind, gates in cc.KIND_GATES.items():
            for gate in gates:
                self.assertEqual(cc.MODIFIER_BY_NAME[gate].kind, cc.BOOL)
        self.assertFalse(cc.KIND_TAKES_MASK["basic"])
        self.assertTrue(all(cc.KIND_TAKES_MASK[k] for k in cc.KINDS if k != "basic"))

    def test_choice_labels_cover_their_choices(self):
        for name, labels in cc.CHOICE_LABELS.items():
            f = cc.MODIFIER_BY_NAME.get(name) or cc.LEAF_BY_NAME.get(name) or cc.COMBO_BY_NAME.get(name)
            self.assertIsNotNone(f, name)
            if name != "mask":
                self.assertEqual(tuple(labels), f.choices, name)
        for group in cc.GROUP_ORDER + ("head", "schedule", "shaping"):
            self.assertIn(group, cc.GROUP_LABELS)

    def test_default_and_blank_config(self):
        default = cc.default_config()
        self.assertEqual([m["active"] for m in default.modifiers], [True] + [False] * 9)
        self.assertFalse(default.enabled or default.masking)
        self.assertEqual([m["active"] for m in cc.blank_config().modifiers], [False] * 10)
        copy = default.copy()
        copy.modifiers[0]["exposure"] = 1.0
        self.assertEqual(default.modifiers[0]["exposure"], 0.0)

    def test_title_constant_and_option_single_source(self):
        from core import forge_override_settings as fos
        self.assertEqual(cc.SCRIPT_NAME, "Colorcraft (sam-extra)")
        self.assertEqual(cc.OPT_PRE_DD, fos.OPT_COLORCRAFT_PRE_DD)
        self.assertEqual(cc.PRE_DD_INFOTEXT_KEY, fos.INFOTEXT_COLORCRAFT_PRE_DD)


class InfotextTests(unittest.TestCase):
    """확장 없이 보는 문자열 규칙 — 확장과의 일치는 InfotextParityTests."""

    def _design_config(self) -> cc.Config:
        config = cc.blank_config()
        config.enabled = config.masking = True
        config.modifiers[0].update({"active": True, "kind": "Luma", "mask": "M1", "exposure": 0.3,
                                    "tone_compression": -0.2})
        config.modifiers[2].update({"active": True, "kind": "Punch", "pass": "Hires", "clarity": 0.5,
                                    "sharpness": 0.25})
        config.leaves[0].update({"mask_mode": "lows", "mask_hardness": 2.0, "blur": 8.0})
        return config

    def test_design_example_string(self):
        # 설계 §4.1 의 문자열 = 확장 to_infotext 가 같은 설정에 만든 값(10-03 CPU 실측).
        self.assertEqual(
            cc.to_infotext(self._design_config()),
            "v1;mods=I,III;masking=1;masks=M1;combos=;I.kind=Luma;I.mask=M1;I.exposure=0.3;I.tone_compression=-0.2;"
            "III.kind=Punch;III.pass=Hires;III.clarity=0.5;III.sharpness=0.25;M1.mask_mode=lows;M1.mask_hardness=2;"
            "M1.blur=8")

    def test_only_fields_of_the_type_are_written(self):
        config = cc.blank_config()
        config.modifiers[1].update({"active": True, "kind": "Basic", "mask": "M2", "exposure": 1.0, "contrast": 0.5})
        self.assertEqual(cc.to_infotext(config), "v1;mods=II;II.kind=Basic;II.contrast=0.5")

    def test_ten_significant_digits(self):
        config = cc.blank_config()
        config.modifiers[0].update({"active": True, "exposure": 0.1 + 0.2, "contrast": 123.456789123,
                                    "clarity": 1e-7})
        self.assertEqual(cc.to_infotext(config), "v1;mods=I;I.exposure=0.3;I.contrast=123.4567891;I.clarity=1e-07")

    def test_parse_back(self):
        config = self._design_config()
        back = cc.parse_own(cc.to_infotext(config))
        self.assertTrue(back.enabled)
        self.assertEqual(cc.config_values(back), cc.config_values(config))

    def test_string_starts_blank_dict_starts_from_panel(self):
        self.assertEqual([m["active"] for m in cc.parse_compact("v1;mods=").modifiers], [False] * 10)
        self.assertEqual([m["active"] for m in cc.parse_compact({}).modifiers], [True] + [False] * 9)
        self.assertIsNone(cc.parse_compact("MOD(I:none)"))
        self.assertIsNone(cc.parse_compact(None))

    def test_block(self):
        config = self._design_config()
        block = cc.as_block(config)
        self.assertEqual(block, {"args": [cc.to_infotext(config)]})
        self.assertEqual(cc.config_values(cc.parse_script_block(block)), cc.config_values(config))
        dict_block = {"args": [{"I.exposure": 0.3}, False]}
        self.assertEqual(cc.parse_script_block(dict_block).modifiers[0]["exposure"], 0.3)
        self.assertIsNone(cc.parse_script_block(None))
        self.assertIsNone(cc.parse_script_block(["v1;mods=I"]))
        for bad in ({"args": []}, {"args": [True, False, ""]}, {"args": "v1;mods=I"}, {}):
            with self.subTest(block=bad), self.assertRaises(cc.BlockError):
                cc.parse_script_block(bad)


def _on(**modifier_i) -> cc.Config:
    config = cc.default_config()
    config.enabled = True
    config.modifiers[0].update(modifier_i)
    return config


class ChainTests(unittest.TestCase):
    """확장 없이 보는 체인 규칙 — 확장과의 일치는 ChainParityTests."""

    def test_defaults_run_nothing(self):
        self.assertEqual(cc.build_chain(_on(), False).entries, [])
        self.assertEqual(cc.build_chain(cc.default_config(), False).entries, [])

    def test_one_entry_exactly(self):
        chain = cc.build_chain(_on(kind="Luma", exposure=0.3, advanced=False, bias=0.2), False)
        self.assertEqual(chain.tags, ["I"])
        self.assertEqual(chain.entries, [{
            "kind": "luma", "params": {"exposure": 0.3, "tone_compression": 0.0},
            "schedule": {"strength": 1.0, "start": 0.5, "end": 0.75, "bias": 0.5, "exponent": 0.0, "start_off": 0.0,
                         "end_off": 0.0, "smooth": True},                 # 고급 스케줄 끔 → 치우침 되돌림
            "mask": None}])
        self.assertTrue(chain.needs_basis)
        self.assertEqual(cc.build_chain(_on(kind="Luma", exposure=0.3), True).entries, [])     # Base 만

    def test_passes(self):
        both = _on(exposure=0.3, **{"pass": "Both"})
        self.assertEqual((cc.build_chain(both, False).tags, cc.build_chain(both, True).tags), (["I"], ["I"]))
        hires = _on(exposure=0.3, **{"pass": "Hires"})
        self.assertEqual((cc.build_chain(hires, False).tags, cc.build_chain(hires, True).tags), ([], ["I"]))

    def test_noop_and_zero_schedule(self):
        self.assertEqual(cc.build_chain(_on(strength=0.0, exposure=1.0), False).entries, [])
        self.assertEqual(len(cc.build_chain(_on(strength=0.0, exposure=1.0, advanced=True, end_off=0.2), False).entries), 1)
        self.assertEqual(cc.build_chain(_on(more_colors=False, lab_a=1.0), False).entries, [])   # 스위치가 끈 값만
        self.assertEqual(len(cc.build_chain(_on(lab_a=1.0), False).entries), 1)
        self.assertEqual(cc.build_chain(_on(kind="Punch", exposure=1.0), False).entries, [])     # 다른 종류의 칸

    def test_masks(self):
        config = _on(exposure=0.3, mask="C2")
        config.masking = True
        config.combos[0].update({"mask_a": "M1", "mask_b": "M2"})
        config.combos[1].update({"mask_a": "C1", "mask_b": "none"})
        chain = cc.build_chain(config, False)
        self.assertEqual(chain.unresolved_masks, ["I->C2"])
        self.assertIsNone(chain.entries[0]["mask"])
        config.combos[1]["mask_b"] = "M3"
        spec = cc.build_chain(config, False).entries[0]["mask"]
        self.assertEqual(spec["operation"], "and")
        self.assertEqual(spec["a"]["operation"], "and")
        self.assertEqual(spec["b"]["mask_axis"], "exposure")
        basic = _on(kind="Basic", contrast=1.0, mask="M1")
        basic.masking = True
        self.assertIsNone(cc.build_chain(basic, False).entries[0]["mask"])             # Basic 은 마스크 입력이 없다
        shift = _on(kind="Shift", color_shift_amount=1.0)
        self.assertFalse(cc.build_chain(shift, False).needs_basis)
        shift.modifiers[0]["mask"] = "M1"
        shift.masking = True
        self.assertTrue(cc.build_chain(shift, False).needs_basis)                       # 마스크를 쓰는 Shift

    def test_leaf_shaping_wraps(self):
        config = _on(exposure=0.3, mask="M1")
        config.masking = True
        config.leaves[0]["blur"] = 4.0
        mask = cc.build_chain(config, False).entries[0]["mask"]
        self.assertEqual(set(mask), {"blur", "spread", "contrast", "normalize", "a"})


class OverviewTests(unittest.TestCase):
    def test_text_and_marks(self):
        config = cc.default_config()
        config.masking = True
        config.modifiers[0].update({"mask": "C1", "exposure": 0.2})
        config.modifiers[2].update({"active": True, "kind": "Punch", "pass": "Hires"})
        config.modifiers[4]["exposure"] = 1.0                      # 꺼졌지만 바꿈
        config.combos[0].update({"mask_a": "M1"})                  # B 없음 → 미완성
        config.leaves[3]["blur"] = 1.0
        view = cc.overview(config)
        self.assertEqual(view.text, "꺼짐 — 켠 수정자: I Advanced ➜ C1(미완성), III Punch (Hires) · 쓰는 마스크: 없음")
        self.assertEqual(view.modifier_marks[:5], (("I ●", "I"), ("II", "II"), ("III ●", "III"), ("IV", "IV"),
                                                   ("V ○", "V")))
        self.assertEqual(dict((tag, label) for label, tag in view.mask_marks)["M4"], "M4 ○")
        self.assertEqual(dict((tag, label) for label, tag in view.mask_marks)["C1"], "C1 ○")
        config.enabled = True
        config.combos[0]["mask_b"] = "M2"
        self.assertEqual(cc.describe(config), "켠 수정자: I Advanced ➜ C1, III Punch (Hires) · 쓰는 마스크: M1, M2, C1")
        self.assertEqual(cc.describe(cc.blank_config()), "꺼짐 — 켠 수정자 없음")


class SchemaTsTests(unittest.TestCase):
    def test_body_is_the_schema(self):
        import re

        text = cc.schema_ts()
        self.assertTrue(text.startswith("// Generated from core/colorcraft.py by schema_ts()"))
        self.assertTrue(text.endswith("}\n"))
        self.assertNotIn("\r", text)
        self.assertEqual(text, cc.schema_ts())
        body = text.split("export const COLORCRAFT_SCHEMA: ColorcraftSchema = ", 1)[1]
        as_json = re.sub(r",(\s*[}\]])", r"\1", body)               # TS 의 끝 쉼표를 빼면 JSON 이다
        self.assertEqual(json.loads(as_json), json.loads(json.dumps(cc.schema())))

    def test_schema_tables(self):
        schema = cc.schema()
        self.assertEqual([f["name"] for f in schema["modifierFields"]], [f.name for f in cc.MODIFIER_FIELDS])
        self.assertEqual(schema["activeDefaults"], [True] + [False] * 9)
        self.assertEqual(schema["comboRefChoices"][0], ["none"] + list(cc.MASK_TAGS))
        self.assertEqual(schema["choiceLabels"]["kind"]["Chroma Plus"], "Chroma Plus (색 축 6개)")
        self.assertEqual(schema["marks"], {"on": "●", "edited": "○", "incomplete": "(미완성)"})
        for fields in ("globalFields", "modifierFields", "leafFields", "comboFields"):
            for f in schema[fields]:
                self.assertEqual(set(f), {"name", "kind", "default", "min", "max", "step", "choices", "group",
                                          "label", "title"})

    def test_write_schema_ts(self):
        import tempfile

        with tempfile.TemporaryDirectory() as folder:
            path = os.path.join(folder, "schema.ts")
            self.assertEqual(cc.write_schema_ts(path), path)
            with open(path, "rb") as fh:
                data = fh.read()
        self.assertEqual(data, cc.schema_ts().encode("utf-8"))
        self.assertTrue(cc.SCHEMA_TS_PATH.endswith("colorcraftSchema.generated.ts"))


class SettingsTests(unittest.TestCase):
    def test_widget_ids(self):
        self.assertEqual([cc.widget_id(k) for k in cc.WIDGET_KEYS], ["_cc_enabled", "_cc_masking", "_cc_img2img",
                                                                     "_cc_state"])
        with self.assertRaises(KeyError):
            cc.widget_id("debug")
        self.assertEqual(cc.SETTINGS_KEY, "colorcraft_settings")

    def test_defaults(self):
        self.assertEqual(cc.APP_DEFAULTS, cc.ColorcraftSettings(False, False, False, ""))
        self.assertEqual(cc.widget_values(cc.APP_DEFAULTS),
                         {"enabled": "false", "masking": "false", "img2img": "false", "state": ""})
        self.assertEqual(cc.config_values(cc.config_of(cc.APP_DEFAULTS)), cc.config_values(cc.default_config()))

    def test_parse_settings(self):
        base = cc.ColorcraftSettings(True, True, True, '{"v":1,"I.exposure":1}')
        self.assertEqual(cc.parse_settings(None, base=base), base)
        self.assertEqual(cc.parse_settings({}, base=base), base)
        self.assertEqual(cc.parse_settings({"enabled": "", "masking": None}, base=base), base)
        got = cc.parse_settings({"enabled": "false", "img2img": "0", "state": '{"I.exposure":0.50,"v":1,"rev":"x"}'},
                                base=base)
        self.assertEqual(got, cc.ColorcraftSettings(False, True, False, '{"v":1,"I.exposure":0.5}'))
        self.assertEqual(cc.parse_settings({"state": ""}, base=base).state, "")
        with self.assertLogs("core.colorcraft", "WARNING"):
            self.assertEqual(cc.parse_settings({"state": "{broken"}, base=base).state, base.state)
        self.assertEqual(cc.parse_settings({"enabled": "maybe"}, base=base).enabled, True)    # 모르는 낱말 = base

    def test_widget_round_trip(self):
        settings = cc.ColorcraftSettings(True, False, True, '{"v":1,"III.active":true,"III.contrast":0.5}')
        self.assertEqual(cc.parse_settings(cc.widget_values(settings)), settings)

    def test_settings_of_config(self):
        config = _on(exposure=0.25)
        config.masking = True
        settings = cc.settings_of(config, img2img=True)
        self.assertEqual(settings, cc.ColorcraftSettings(True, True, True, '{"v":1,"I.exposure":0.25}'))
        self.assertEqual(cc.config_values(cc.config_of(settings)), cc.config_values(config))

    def test_unreadable_state_in_settings_object(self):
        with self.assertLogs("core.colorcraft", "WARNING"):
            config = cc.config_of(cc.ColorcraftSettings(True, False, False, "{broken"))
        self.assertEqual(cc.config_values(config)[2:], cc.config_values(cc.default_config())[2:])


class PlanTests(unittest.TestCase):
    def _settings(self, **modifier_i) -> cc.ColorcraftSettings:
        return cc.settings_of(_on(**modifier_i))

    def test_reasons(self):
        on = self._settings(exposure=0.3)
        off = cc.ColorcraftSettings(False, False, False, on.state)
        self.assertEqual(cc.plan(off, target="t2i", backend="webui").reason, cc.REASON_OFF)
        self.assertEqual(cc.plan(on, target="t2i", backend="krea2").reason, cc.REASON_KREA2)
        # 계획 C6 — ComfyUI 도 Forge 와 같은 규칙·같은 블록(컴파일러가 앱 노드 팩 1.7.0 의 노드로 — C5)
        for settings in (on, off, cc.ColorcraftSettings(True), cc.settings_of(_on(exposure=0.3), img2img=True)):
            for target in ("t2i", "i2i", "aux"):
                with self.subTest(target=target):
                    self.assertEqual(cc.plan(settings, target=target, backend="comfyui"),
                                     cc.plan(settings, target=target, backend="webui"))
        self.assertEqual(cc.plan(on, target="t2i", backend="comfyui").reason, cc.REASON_SEND)
        self.assertEqual(cc.plan(on, target="i2i", backend="webui").reason, cc.REASON_I2I_OFF)
        self.assertEqual(cc.plan(cc.ColorcraftSettings(True), target="t2i", backend="webui").reason,
                         cc.REASON_NO_ENTRIES)
        for reason_plan in (cc.plan(off, target="t2i", backend="webui"),
                            cc.plan(cc.ColorcraftSettings(True), target="t2i", backend="webui")):
            self.assertIsNone(reason_plan.block)

    def test_send(self):
        on = self._settings(exposure=0.3)
        for target in ("t2i", "aux"):
            result = cc.plan(on, target=target, backend="webui")
            self.assertEqual(result.reason, cc.REASON_SEND)
            self.assertEqual(result.block, {"args": ["v1;mods=I;I.exposure=0.3"]})
            self.assertEqual(result.provenance, "user")
            self.assertEqual((result.base_tags, result.hires_tags), (("I",), ()))
            self.assertIsNone(result.notice)
        i2i = cc.ColorcraftSettings(True, False, True, on.state)
        self.assertEqual(cc.plan(i2i, target="i2i", backend="webui").block, {"args": ["v1;mods=I;I.exposure=0.3"]})

    def test_hires_only(self):
        hires = self._settings(exposure=0.3, **{"pass": "Hires"})
        result = cc.plan(hires, target="t2i", backend="webui")
        self.assertEqual((result.reason, result.base_tags, result.hires_tags), (cc.REASON_SEND, (), ("I",)))
        i2i = cc.ColorcraftSettings(True, False, True, hires.state)
        self.assertEqual(cc.plan(i2i, target="i2i", backend="webui").reason, cc.REASON_NO_ENTRIES)   # i2i 는 Base 만

    def test_comfy_constants(self):
        self.assertEqual(cc.COMFY_NODE_CLASS, "ForgeNeoColorcraft")
        self.assertEqual(cc.COMFY_CONTRACT, "v1")
        self.assertTrue(cc.COMFY_PRE_DD_SIGMA)
        self.assertEqual(cc.COMFY_MIN_PACK_VERSION, "1.7.0")


if __name__ == "__main__":
    unittest.main()
