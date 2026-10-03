"""Colorcraft(sam-extra 'Colorcraft (sam-extra)') — core/colorcraft = 설치된 확장(시험 대조 기준).

설치된 확장(``tests/_sam_extra_ext.EXT_ROOT``)이 기준이다 — 같은 입력을 양쪽에 넣어 같은 결과를 요구한다. 확장의
``sam3ext/colorcraft/spec.py`` 는 ``engine`` → ``basis`` 를 거쳐 torch 를 올리므로 import 비교는 @requires_torch 다
(CPU 만 — CUDA_VISIBLE_DEVICES=-1 로 돌린다).

* ``FixtureTests`` — 녹화된 script-info(tests/fixtures/sam_extra_script_info.json, 67칸)의 슬롯 0–1 / 6–49 / 50–59 /
  60–66 = 앱 ``GLOBAL_FIELDS`` / ``MODIFIER_FIELDS`` / ``LEAF_FIELDS`` / ``COMBO_FIELDS``(영어 이름·기본값·범위·선택지).
* ``ExtensionConstantTests`` — 확장 소스의 리터럴 상수(AST, import 없음) = 앱 상수.
* ``TableParityTests`` — 확장 모듈의 표(칸·종류·묶음·스위치·조합 선택지·경로) = 앱 표.
* ``CoerceParityTests`` — ``coerce`` = ``spec.coerce``(값 표 × 모든 칸). 너무 큰 정수만 다르다(앱은 기본값).
* ``StateParityTests`` — 상태 문자열: 골든 입력을 확장 ``panel_state.config_from_state`` 와 앱이 같은 설정으로 읽고,
  확장 ``dump_state``(rev 포함)를 앱이, 앱 상태를 확장이 같은 설정으로 읽는다.
* ``InfotextParityTests`` — 고정 시드 말뭉치(tests/_colorcraft_corpus.py, 600개 이상): ``to_infotext``·``active_masks`` =
  확장, ``parse_own``(손으로 고른 문자열 포함)·``is_compact_arg``·dict 형식(``config_from_mapping``) = 확장, 보낸 블록을
  확장 훅의 67칸 판독(``panel_state.config_from_script_args`` — 나머지 66칸은 ``ui()`` 기본값)이 같은 설정으로 읽는다
  (compact == 패널).
* ``PasteParityTests`` — 붙여 넣기 775가지(자기 형식, 업스트림 ``MOD(...)`` 문자열과 그 콜백 dict, aoleg 포크 ``v1;name=value``,
  두 열쇠, 엉뚱한 값): ``from_infotext`` = ``spec.decode``, 모든 경로의 ``spec.paste_value``, 확장이 붙여 넣기 때 쓰는 상태
  (``panel_state.pasted``) = 앱 상태, 맨 값의 형식 구분(``value_format``·``from_value``) = 확장 구분 규칙. 업스트림·포크 표
  (값 순서·포크 칸·선택지·미리보기 색) = 확장.
* ``ChainParityTests`` — ``build_chain``(두 패스)·``mask_specs`` = 확장, ``needs_basis`` = ``engine.needs_basis``, 보낸 블록과
  확장 패널 67칸(``args_from_config``)이 같은 체인(compact == 패널 — 카드가 만들 수 있는 값으로), 카드 값은 문자열을 거쳐도
  같은 체인, ``overview``(●/○ 표시·요약 글자)·``used_masks`` = ``panel_state``, 앱 ``schema()`` 에 확장 ``schema()`` 가 같은
  값으로 다 들어 있다.

확장이 없으면 확장 비교만 skip, AISTUDIO_REQUIRE_FORGE_EXT=1 이면 실패. torch 는 지연 import(tests/_optional_deps).
"""
from __future__ import annotations

import ast
import importlib
import json
import math
import os
import sys
import unittest

from core import colorcraft as cc
from tests._optional_deps import requires_torch
from tests._sam_extra_ext import EXT_ROOT, requires_extension

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIXTURE = os.path.join(ROOT, "tests", "fixtures", "sam_extra_script_info.json")
GOLDEN = os.path.join(ROOT, "tests", "fixtures", "colorcraft_state_golden.json")


def _ext(module: str):
    """설치된 확장 모듈(``sam3ext.<module>``). colorcraft.spec·panel_state 는 torch 를 올린다."""
    root = str(EXT_ROOT)
    if root not in sys.path:
        sys.path.append(root)
    return importlib.import_module(f"sam3ext.{module}")


def _ext_constant(relative: str, name: str):
    """확장 파일의 모듈 상수(리터럴, AST — import 없음)."""
    tree = ast.parse((EXT_ROOT / relative).read_text(encoding="utf-8"))
    for item in tree.body:
        targets = item.targets if isinstance(item, ast.Assign) else (
            [item.target] if isinstance(item, ast.AnnAssign) else [])
        if any(isinstance(target, ast.Name) and target.id == name for target in targets):
            return ast.literal_eval(item.value)
    raise AssertionError(f"{relative} 에 모듈 상수 {name} 이 없다")


def _key(value):
    """float 는 repr 로 — -0.0 과 0.0, 마지막 비트까지 다르게 본다. bool 과 1.0 도 다르게."""
    if isinstance(value, float):
        return ("float", repr(value))
    return (type(value).__name__, value)


def _keys(values) -> list:
    return [_key(v) for v in values]


def _no_negative_zero(values) -> list:
    return [0.0 if isinstance(v, float) and v == 0 else v for v in values]


def _ext_values(ext_config) -> list:
    """확장 ``config_to_args`` 에서 디버그 두 칸(끝)을 뺀 값 = 앱 ``config_values`` 와 같은 순서."""
    return _ext("colorcraft.spec").config_to_args(ext_config)[:-2]


def _golden_inputs() -> list:
    with open(GOLDEN, encoding="utf-8") as fh:
        return [(case["name"], case["input"]) for case in json.load(fh)["cases"]]


# ── 픽스처(녹화된 Forge script-info) ────────────────────────────────────────────
class FixtureTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with open(FIXTURE, encoding="utf-8") as fh:
            scripts = json.load(fh)["scripts"]
        cls.entries = [s for s in scripts if s.get("name") == cc.SCRIPT_NAME.lower()]

    def test_txt2img_and_img2img_entries(self):
        self.assertEqual(sorted(bool(e.get("is_img2img")) for e in self.entries), [False, True])
        for entry in self.entries:
            self.assertEqual(len(entry["args"]), 67)

    def _check(self, slot, f, default):
        self.assertEqual(slot["label"], f.title)
        self.assertEqual(_key(slot["value"]), _key(default))
        if f.kind == cc.FLOAT:
            self.assertEqual((slot["minimum"], slot["maximum"], slot["step"]), (f.minimum, f.maximum, f.step))
        else:
            self.assertEqual((slot["minimum"], slot["maximum"], slot["step"]), (None, None, None))
        self.assertEqual(slot["choices"], list(f.choices) if f.kind == cc.CHOICE else None)

    def test_editor_slots_are_the_app_tables(self):
        for entry in self.entries:
            args = entry["args"]
            for i, f in enumerate(cc.GLOBAL_FIELDS):
                with self.subTest(img2img=entry.get("is_img2img"), slot=i):
                    self._check(args[i], f, f.default)
            for i, f in enumerate(cc.MODIFIER_FIELDS):
                with self.subTest(img2img=entry.get("is_img2img"), slot=6 + i, field=f.name):
                    self._check(args[6 + i], f, cc.modifier_default(0, f.name))   # 편집기는 수정자 I 로 시작
            for i, f in enumerate(cc.LEAF_FIELDS):
                with self.subTest(img2img=entry.get("is_img2img"), slot=50 + i, field=f.name):
                    self._check(args[50 + i], f, f.default)
            for i, f in enumerate(cc.COMBO_FIELDS):
                with self.subTest(img2img=entry.get("is_img2img"), slot=60 + i, field=f.name):
                    self._check(args[60 + i], f, f.default)

    def test_head_slots(self):
        for entry in self.entries:
            args = entry["args"]
            self.assertEqual(args[2]["value"], cc.DEFAULT_STATE)                # 숨은 상태 칸: "" = 전부 기본값
            self.assertEqual(args[5]["value"], "I|M1|0")                        # 편집기 ref(수정자 I·M1)


# ── 확장 소스 상수(AST) ─────────────────────────────────────────────────────────
@requires_extension
class ExtensionConstantTests(unittest.TestCase):
    def test_title_and_infotext_keys(self):
        self.assertEqual(_ext_constant("scripts/colorcraft.py", "TITLE"), cc.SCRIPT_NAME)
        spec = "sam3ext/colorcraft/spec.py"
        self.assertEqual(_ext_constant(spec, "INFOTEXT_KEY"), cc.INFOTEXT_KEY)
        self.assertEqual(_ext_constant(spec, "STATUS_KEY"), cc.STATUS_KEY)
        self.assertEqual(_ext_constant(spec, "FOREIGN_KEY"), cc.FOREIGN_KEY)
        self.assertEqual(_ext_constant(spec, "VERSION"), cc.VERSION)
        hook = "sam3ext/colorcraft/hook.py"
        self.assertEqual(_ext_constant(hook, "INFOTEXT_PRE_DD"), cc.PRE_DD_INFOTEXT_KEY)
        self.assertEqual(_ext_constant(hook, "OPT_PRE_DD"), cc.OPT_PRE_DD)

    def test_choice_lists(self):
        spec = "sam3ext/colorcraft/spec.py"
        self.assertEqual(tuple(_ext_constant("sam3ext/colorcraft/engine.py", "KINDS")), cc.KINDS)
        self.assertEqual(tuple(_ext_constant("sam3ext/colorcraft/masking.py", "MASK_AXIS_OPTIONS")),
                         cc.MASK_AXIS_OPTIONS)
        self.assertEqual(tuple(_ext_constant(spec, "MASK_MODE_OPTIONS")), cc.MASK_MODE_OPTIONS)
        self.assertEqual(tuple(_ext_constant(spec, "COMBO_OP_OPTIONS")), cc.COMBO_OP_OPTIONS)
        self.assertEqual(tuple(_ext_constant(spec, "COLOR_SHIFT_MODE_OPTIONS")), cc.COLOR_SHIFT_MODE_OPTIONS)
        self.assertEqual(tuple(_ext_constant(spec, "CHROMA_PLANE_OPTIONS")), cc.CHROMA_PLANE_OPTIONS)
        self.assertEqual(_ext_constant(spec, "KIND_LABELS"), dict(cc.KIND_LABELS))
        self.assertEqual(_ext_constant(spec, "MASK_NONE"), cc.MASK_NONE)
        self.assertEqual(set(_ext_constant("sam3ext/colorcraft/engine.py", "ADVANCED_KINDS")), cc.ADVANCED_KINDS)

    def test_counts_and_groups(self):
        spec = "sam3ext/colorcraft/spec.py"
        self.assertEqual((_ext_constant(spec, "MODIFIER_COUNT"), _ext_constant(spec, "MASK_COUNT"),
                          _ext_constant(spec, "COMBO_COUNT")), (cc.MODIFIER_COUNT, cc.MASK_COUNT, cc.COMBO_COUNT))
        self.assertEqual(tuple(_ext_constant(spec, "ROMAN")), cc.MODIFIER_TAGS)
        self.assertEqual({k: tuple(v) for k, v in _ext_constant(spec, "KIND_GROUPS").items()}, dict(cc.KIND_GROUPS))
        self.assertEqual({k: tuple(v) for k, v in _ext_constant(spec, "KIND_GATES").items()}, dict(cc.KIND_GATES))
        self.assertEqual(tuple(_ext_constant(spec, "_SCHEDULE_FIELDS")), cc.SCHEDULE_NAMES)
        self.assertEqual(tuple(_ext_constant(spec, "_SHAPING_FIELDS")), cc.SHAPING_NAMES)
        self.assertEqual(tuple(_ext_constant(spec, "_CHROMA_PLUS_KEYS")), cc.CHROMA_PLUS_NAMES)
        self.assertEqual(tuple(_ext_constant("sam3ext/colorcraft/panel_state.py", "GROUP_ORDER")), cc.GROUP_ORDER)
        self.assertEqual(_ext_constant("sam3ext/colorcraft/panel_state.py", "STATE_VERSION"), cc.STATE_VERSION)


# ── 확장 모듈 표(import) ────────────────────────────────────────────────────────
@requires_extension
@requires_torch
class TableParityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.spec = _ext("colorcraft.spec")
        cls.panel = _ext("colorcraft.panel_state")

    def _same_fields(self, ext_fields, app_fields):
        self.assertEqual(len(ext_fields), len(app_fields))
        for ext_f, app_f in zip(ext_fields, app_fields):
            with self.subTest(field=app_f.name):
                self.assertEqual(ext_f.name, app_f.name)
                self.assertEqual(ext_f.kind, app_f.kind)
                self.assertEqual(_key(ext_f.default), _key(app_f.default))
                self.assertEqual(ext_f.label, app_f.title)
                self.assertEqual((ext_f.minimum, ext_f.maximum, ext_f.step),
                                 (app_f.minimum, app_f.maximum, app_f.step))
                self.assertEqual(tuple(ext_f.choices), app_f.choices)
                self.assertEqual(ext_f.group, app_f.group)

    def test_field_tables(self):
        self._same_fields(self.spec.MODIFIER_FIELDS, cc.MODIFIER_FIELDS)
        self._same_fields(self.spec.LEAF_FIELDS, cc.LEAF_FIELDS)
        self._same_fields(self.spec.COMBO_FIELDS, cc.COMBO_FIELDS)
        self._same_fields(self.spec.GLOBAL_FIELDS, cc.GLOBAL_FIELDS)

    def test_kind_tables(self):
        spec = self.spec
        self.assertEqual(tuple(spec.KINDS if hasattr(spec, "KINDS") else _ext("colorcraft.engine").KINDS), cc.KINDS)
        self.assertEqual(tuple(spec.KIND_CHOICES), cc.KIND_CHOICES)
        self.assertEqual(dict(spec.KIND_IDS), dict(cc.KIND_IDS))
        self.assertEqual({k: tuple(v) for k, v in spec._KIND_PARAMS.items()}, dict(cc.KIND_PARAMS))
        self.assertEqual(dict(spec.KIND_TAKES_MASK), dict(cc.KIND_TAKES_MASK))
        self.assertEqual(tuple(spec.PASS_CHOICES), cc.PASS_CHOICES)
        self.assertEqual(tuple(spec.MASK_CHOICES), cc.MASK_CHOICES)
        self.assertEqual((tuple(spec.MASK_TAGS), tuple(spec.COMBO_TAGS)), (cc.MASK_TAGS, cc.COMBO_TAGS))
        self.assertEqual(set(_ext("colorcraft.engine").ADVANCED_KINDS), cc.ADVANCED_KINDS)

    def test_combo_ref_choices_and_defaults(self):
        for i in range(cc.COMBO_COUNT + 1):
            self.assertEqual(tuple(self.spec.combo_ref_choices(i)), cc.combo_ref_choices(i))
        for i in range(cc.MODIFIER_COUNT):
            for f in cc.MODIFIER_FIELDS:
                self.assertEqual(_key(self.spec.modifier_default(i, f.name)), _key(cc.modifier_default(i, f.name)))

    def test_paths(self):
        self.assertEqual(tuple(self.spec.arg_names()[:-2]), cc.arg_names())
        self.assertEqual(tuple(self.spec.arg_names()[-2:]), ("debug", "debug_step"))
        self.assertEqual(tuple(self.panel.STATE_PATHS), cc.STATE_PATHS)
        self.assertEqual(tuple(self.panel.NODE_TAGS), cc.NODE_TAGS)

    def test_default_and_blank_configs(self):
        self.assertEqual(_keys(_ext_values(self.spec.default_config())), _keys(cc.config_values(cc.default_config())))
        self.assertEqual(_keys(_ext_values(self.spec.blank_config())), _keys(cc.config_values(cc.blank_config())))


# ── 값 정규화 ─────────────────────────────────────────────────────────────────
_ANY_VALUES = (None, "", "  ", 0, 1, 2, -1, 0.0, -0.0, 1.0, 0.5, -3.75, 1e300, True, False, math.nan, math.inf,
               -math.inf, "0", "1", "0.3", " 2.5 ", "-0", "1e3", "1_000", "nan", "inf", "-inf", "abc", "true", "True",
               " YES ", "on", "Off", "no", "false", "FALSE", "maybe", [], [1], {}, {"a": 1}, (1,), "none", "None",
               b"1")
_CHOICE_VALUES = ("Advanced", "advanced", " ADVANCED ", "Basic", "chroma plus", "Chroma Plus", "chroma_plus",
                  "CHROMA_PLUS", "Chroma-Plus", "punch ", "Base", "base", "HIRES", "Both", "both ", "none", "None",
                  "NONE", "M1", "m1", " M10 ", "M11", "C1", "c5", "C6", "hue", "HUE", "lab-a+b", "protect range",
                  "Protect Range", "protect_range", "and", "OR", "xor", "default", "Legacy", "lab", "LAB",
                  "temp_tint", "Temp_Tint", "temp tint", "exposure", "sharpness", "clarity ")
_HUGE_INT = 10 ** 400


@requires_extension
@requires_torch
class CoerceParityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.spec = _ext("colorcraft.spec")

    def _pairs(self):
        for app_f, ext_f in zip(cc.MODIFIER_FIELDS, self.spec.MODIFIER_FIELDS):
            yield app_f, ext_f, None
        for app_f, ext_f in zip(cc.LEAF_FIELDS, self.spec.LEAF_FIELDS):
            yield app_f, ext_f, None
        for index in range(cc.COMBO_COUNT):
            for app_f, ext_f in zip(cc.COMBO_FIELDS, self.spec.COMBO_FIELDS):
                choices = cc.combo_ref_choices(index) if app_f.name in ("mask_a", "mask_b") else None
                yield app_f, ext_f, choices
        for app_f, ext_f in zip(cc.GLOBAL_FIELDS, self.spec.GLOBAL_FIELDS):
            yield app_f, ext_f, None

    def test_every_field_and_value(self):
        count = 0
        for app_f, ext_f, choices in self._pairs():
            for value in _ANY_VALUES + _CHOICE_VALUES:
                ext_choices = list(choices) if choices is not None else None
                expected = self.spec.coerce(ext_f, value, ext_choices)
                with self.subTest(field=app_f.name, value=value, choices=choices):
                    self.assertEqual(_key(cc.coerce(app_f, value, choices)), _key(expected))
                count += 1
        self.assertGreater(count, 6000)

    def test_huge_integer_is_the_one_difference(self):
        f, ext_f = cc.MODIFIER_BY_NAME["exposure"], self.spec.MODIFIER_BY_NAME["exposure"]
        with self.assertRaises(OverflowError):
            self.spec.coerce(ext_f, _HUGE_INT)
        self.assertEqual(_key(cc.coerce(f, _HUGE_INT)), _key(0.0))
        self.assertEqual(_key(cc.coerce(f, -_HUGE_INT)), _key(0.0))
        # 참/거짓·선택지 칸은 float() 를 거치지 않아 같다.
        for name in ("advanced", "kind"):
            self.assertEqual(_key(cc.coerce(cc.MODIFIER_BY_NAME[name], _HUGE_INT)),
                             _key(self.spec.coerce(self.spec.MODIFIER_BY_NAME[name], _HUGE_INT)))


# ── 상태 문자열 ───────────────────────────────────────────────────────────────
@requires_extension
@requires_torch
class StateParityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.spec = _ext("colorcraft.spec")
        cls.panel = _ext("colorcraft.panel_state")

    def test_golden_inputs_read_the_same(self):
        for name, raw in _golden_inputs():
            for enabled, masking in ((True, False), (False, True)):
                with self.subTest(case=name, enabled=enabled, masking=masking):
                    ext_config = self.panel.config_from_state(enabled, masking, raw)
                    app_config = cc.config_from_state(enabled, masking, raw)
                    self.assertEqual(_keys(cc.config_values(app_config)), _keys(_ext_values(ext_config)))

    def test_extension_reads_the_app_state(self):
        for name, raw in _golden_inputs():
            state = cc.canonical_state(raw)
            with self.subTest(case=name):
                ext_config = self.panel.config_from_state(True, True, state)
                app_config = cc.config_from_state(True, True, raw)
                # 앱 상태는 -0 을 0 으로 쓴다 — 그 밖에는 비트까지 같은 설정이다.
                self.assertEqual(_keys(_no_negative_zero(cc.config_values(app_config))), _keys(_ext_values(ext_config)))
                # rev 없음 = 바깥 상태(편집기 값을 얹지 않는다). "" 는 확장의 첫 rev "0".
                self.assertEqual(self.panel.parse_state(state)[1], None if state else "0")

    def test_app_reads_the_extension_dump(self):
        ext_config = self.spec.default_config()
        ext_config.modifiers[0]["exposure"] = 0.3
        ext_config.modifiers[2].update({"active": True, "kind": "Punch", "pass": "Hires", "clarity": 0.5})
        ext_config.leaves[0].update({"mask_mode": "lows", "blur": 8.0})
        ext_config.combos[1].update({"mask_a": "C1", "mask_b": "M3", "operation": "xor"})
        dumped = self.panel.dump_state(ext_config, "a1b2c3")
        app_config = cc.config_from_state(False, False, dumped)
        self.assertEqual(_keys(cc.config_values(app_config)), _keys(_ext_values(ext_config)))
        # 같은 설정을 앱이 쓰면 rev 만 빠진다.
        expected = json.loads(dumped)
        del expected["rev"]
        self.assertEqual(json.loads(cc.dump_state(app_config)), expected)


# ── infotext 문자열·한 값 형식 (A-CC2) ──────────────────────────────────────────
def _messy_mappings(configs) -> list:
    """dict 형식 입력: 설정 전체 경로, 일부 경로, 모르는 열쇠, None, 엉뚱한 값·열쇠."""
    import random

    rng = random.Random(77)
    out = [{}, {"enabled": False}, {"enabled": None, "masking": "1"}, {"I.active": False},
           {"I.exposure": 0.3}, {"enabled": "maybe", "I.kind": "punch", "I.clarity": "0.5", "M1.blur": None},
           {1: 2, None: 3, ("I", "exposure"): 1, "I.": 1, ".exposure": 1, "XI.exposure": 1, "I.debug": True,
            "debug": True, "debug_step": 9, "rev": "x", "v": 2},
           {"C1.mask_a": "C1", "C2.mask_a": "c1", "C5.mask_b": "C5", "I.mask": "C9", "II.pass": "BOTH"}]
    for config in configs[:120]:
        paths = dict(zip(cc.arg_names(), cc.config_values(config)))
        out.append(paths)
        keys = rng.sample(list(paths), k=rng.randint(1, 40))
        partial = {k: paths[k] for k in keys}
        for k in rng.sample(keys, k=min(len(keys), 5)):
            partial[k] = rng.choice(_ANY_VALUES + _CHOICE_VALUES)
        out.append(partial)
    return out


@requires_extension
@requires_torch
class InfotextParityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from tests._colorcraft_corpus import corpus, to_ext
        cls.spec = _ext("colorcraft.spec")
        cls.panel = _ext("colorcraft.panel_state")
        cls.configs = corpus()
        cls.to_ext = staticmethod(lambda c: to_ext(cls.spec, c))

    def test_corpus_size_and_reach(self):
        self.assertGreaterEqual(len(self.configs), 500)
        kinds = {(cc.kind_of(m), m["pass"]) for c in self.configs for m in c.modifiers if m["active"]}
        self.assertEqual(len(kinds), 21)
        self.assertTrue(any(c.masking and cc.active_masks(c)[1] for c in self.configs))

    def test_to_infotext(self):
        for i, config in enumerate(self.configs):
            with self.subTest(i=i):
                self.assertEqual(cc.to_infotext(config), self.spec.to_infotext(self.to_ext(config)))

    def test_active_masks(self):
        for i, config in enumerate(self.configs):
            with self.subTest(i=i):
                leaves, combos = self.spec.active_masks(self.to_ext(config))
                self.assertEqual(cc.active_masks(config), (list(leaves), list(combos)))

    def test_parse_own(self):
        from tests._colorcraft_corpus import OWN_TEXTS
        texts = list(OWN_TEXTS) + [cc.to_infotext(c) for c in self.configs]
        for text in texts:
            with self.subTest(text=text[:120]):
                self.assertEqual(_keys(cc.config_values(cc.parse_own(text))),
                                 _keys(_ext_values(self.spec.parse_own(text))))

    def test_is_compact_arg_and_parse_compact(self):
        from tests._colorcraft_corpus import OWN_TEXTS
        values = list(OWN_TEXTS) + [" v1;mods=I", "\tv1 ;mods =", "V1;mods=I", "v1;Mods=I", "v1,mods=I", "v1;mod=I",
                                    "v1;mods", "v1 ; mods_x=I", "v1;mods =", " \n v1;mods=I \n",
                                    "MOD(I:none)", {}, {"I.exposure": 1}, [], ["v1;mods=I"], None, 1, b"v1;mods=I"]
        for value in values:
            with self.subTest(value=value):
                self.assertEqual(cc.is_compact_arg(value), self.spec.is_compact_arg(value))
                app = cc.parse_compact(value)
                if not self.spec.is_compact_arg(value):
                    self.assertIsNone(app)
                    continue
                self.assertEqual(_keys(cc.config_values(app)),
                                 _keys(_ext_values(self.spec.config_from_args([value]))))

    def test_mapping_form(self):
        for i, mapping in enumerate(_messy_mappings(self.configs)):
            with self.subTest(i=i):
                self.assertEqual(_keys(cc.config_values(cc.parse_compact(mapping))),
                                 _keys(_ext_values(self.spec.config_from_mapping(mapping))))

    def test_compact_block_is_what_the_hook_reads(self):
        """Forge 가 ``ui()`` 기본값으로 채운 나머지 66칸과 함께 받아도 확장 훅은 첫 인자만 읽는다."""
        defaults = self.panel.default_args()
        self.assertEqual(len(defaults), 67)
        for i, config in enumerate(self.configs):
            block = cc.as_block(config)
            with self.subTest(i=i):
                hook = self.panel.config_from_script_args(block["args"] + defaults[1:])
                self.assertEqual(_keys(_ext_values(hook)), _keys(cc.config_values(cc.parse_script_block(block))))
                self.assertTrue(self.panel.args_enabled(block["args"] + defaults[1:]))

    def test_round_trip_is_stable(self):
        """한 번 읽은 문자열은 고정점이다. 말뭉치에는 표가 막는 값(스스로·뒤 조합, 모르는 종류)을 직접 넣은 설정도 있어
        읽을 때 기본값이 되므로 첫 문자열과 다를 수 있다 — 그런 설정만 다르고, 그런 설정이 실제로 있는지도 본다."""
        changed = 0
        for i, config in enumerate(self.configs):
            text = cc.to_infotext(config)
            normal = cc.to_infotext(cc.parse_own(text))
            legal = _is_coerced(config)
            changed += normal != text
            with self.subTest(i=i):
                self.assertEqual(cc.to_infotext(cc.parse_own(normal)), normal)
                if legal:
                    self.assertEqual(normal, text)
        self.assertGreater(changed, 0)


# ── 붙여 넣기 (A-CC3) ──────────────────────────────────────────────────────────
def _paste_params(spec) -> list:
    """붙여 넣기 입력: 자기 형식·업스트림 문자열과 dict·포크 문자열·두 열쇠·엉뚱한 값."""
    import random

    from tests._colorcraft_corpus import FORK_TEXTS, OWN_TEXTS, UPSTREAM_TEXTS, corpus, fork_text, upstream_text

    rng = random.Random(4242)
    params = [{}, {"Steps": "20"}, {cc.INFOTEXT_KEY: None}, {cc.FOREIGN_KEY: None}, {cc.FOREIGN_KEY: 5},
              {cc.FOREIGN_KEY: ""}, {cc.FOREIGN_KEY: "garbage"}, {cc.FOREIGN_KEY: "  v1"}, {cc.INFOTEXT_KEY: "garbage"},
              {cc.INFOTEXT_KEY: 12}, {cc.INFOTEXT_KEY: ["v1;mods=I"]}]
    params += [{cc.INFOTEXT_KEY: text} for text in OWN_TEXTS]
    params += [{cc.INFOTEXT_KEY: cc.to_infotext(c)} for c in corpus()[:150]]
    uppers = list(UPSTREAM_TEXTS) + [upstream_text(rng) for _ in range(250)]
    params += [{cc.FOREIGN_KEY: text} for text in uppers]
    params += [{cc.FOREIGN_KEY: spec.upstream_dict(text)} for text in uppers[:60]]       # 업스트림 콜백이 만든 dict
    params += [{cc.INFOTEXT_KEY: spec.upstream_dict(text)} for text in uppers[:10]]      # 자기 열쇠에 dict(확장도 업스트림으로)
    params += [{cc.FOREIGN_KEY: d} for d in ({}, {"modifiers": [1]}, {"modifiers": {"I": None}}, {"masks": {"X": {}}},
                                              {"modifiers": {"I": {"exposure": "0.5", "hires": "1", "active": False}}},
                                              {"combos": {"C2": {"mask_a": "C1", "mask_b": "M1"}}}, {"modifiers": None})]
    forks = list(FORK_TEXTS) + [fork_text(rng) for _ in range(250)]
    params += [{cc.FOREIGN_KEY: text} for text in forks]
    params += [{cc.INFOTEXT_KEY: OWN_TEXTS[2], cc.FOREIGN_KEY: uppers[1]}, {cc.INFOTEXT_KEY: "", cc.FOREIGN_KEY: forks[1]}]
    return params


def _paste_vector(spec, params) -> list:
    """확장 ``paste_value`` 를 ``arg_names()`` 경로마다 — 붙여 넣기가 패널 칸마다 돌려주는 값."""
    return [spec.paste_value(params, path) for path in spec.arg_names()]


def _app_paste_vector(config) -> list:
    """앱 ``from_infotext`` 결과로 같은 벡터를 만든다(켬 = 설정이 있고 켜짐, 디버그 = 붙여 넣지 않음)."""
    if config is None:
        return [False] + [None] * (len(cc.arg_names()) + 1)
    return [bool(config.enabled), bool(config.masking)] + cc.config_values(config)[2:] + [None, None]


@requires_extension
@requires_torch
class PasteParityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.spec = _ext("colorcraft.spec")
        cls.panel = _ext("colorcraft.panel_state")
        cls.params = _paste_params(cls.spec)

    def test_tables(self):
        self.assertEqual(tuple(self.spec.UPSTREAM_MODIFIER_VALUE_FIELDS), cc.UPSTREAM_MODIFIER_VALUE_FIELDS)
        self.assertEqual(tuple(self.spec.UPSTREAM_LEAF_FIELDS), cc.UPSTREAM_LEAF_FIELDS)
        self.assertEqual(tuple(self.spec.UPSTREAM_COMBO_FIELDS), cc.UPSTREAM_COMBO_FIELDS)
        self.assertEqual(dict(self.spec.FORK_PARAMS), dict(cc.FORK_PARAMS))
        self.assertEqual({k: tuple(v) for k, v in self.spec._FORK_CHOICES.items()}, dict(cc.FORK_CHOICES))
        self.assertEqual(dict(self.spec.FORK_PREVIEW_COLORS), dict(cc.FORK_PREVIEW_COLORS))

    def test_from_infotext_is_decode(self):
        self.assertGreater(len(self.params), 700)
        seen = {"own": 0, "upstream": 0, "fork": 0, "none": 0}
        for i, params in enumerate(self.params):
            expected = self.spec.decode(params)
            app = cc.from_infotext(params)
            with self.subTest(i=i, params=str(params)[:160]):
                if expected is None:
                    self.assertIsNone(app)
                    seen["none"] += 1
                    continue
                self.assertIsNotNone(app)
                self.assertEqual(_keys(cc.config_values(app)), _keys(_ext_values(expected)))
                raw = params.get(cc.INFOTEXT_KEY, params.get(cc.FOREIGN_KEY))
                seen["own" if cc.INFOTEXT_KEY in params and not isinstance(raw, dict)
                     else "fork" if cc.is_fork_text(raw) else "upstream"] += 1
        self.assertTrue(min(seen["own"], seen["upstream"], seen["fork"]) >= 20 and seen["none"] >= 5, seen)

    def test_paste_value_for_every_path(self):
        for i, params in enumerate(self.params):
            with self.subTest(i=i, params=str(params)[:160]):
                self.assertEqual(_keys(_app_paste_vector(cc.from_infotext(params))),
                                 _keys(_paste_vector(self.spec, params)))

    def test_pasted_state_is_the_extension_state(self):
        """확장이 붙여 넣기 때 만드는 상태(rev 포함)와 앱 상태가 같은 값을 적는다(-0 을 0 으로 보는 것만 다르다)."""
        for i, params in enumerate(self.params):
            hit = self.panel.pasted(params)
            app = cc.from_infotext(params)
            with self.subTest(i=i, params=str(params)[:160]):
                if hit is None:
                    self.assertIsNone(app)
                    continue
                ext_paths = json.loads(hit[1])
                del ext_paths["v"], ext_paths["rev"]
                for path in list(ext_paths):
                    tag, _, name = path.partition(".")
                    f = next(f for f in cc.node_fields(tag) if f.name == name)
                    if f.kind == cc.FLOAT and ext_paths[path] == cc.node_default(tag, f):
                        del ext_paths[path]          # 확장은 -0.0 을 기본값 0.0 과 다르게 적는다(repr 비교)
                self.assertEqual(cc.parse_state(cc.dump_state(app)), ext_paths)

    def test_value_format_and_from_value(self):
        from tests._colorcraft_corpus import FORK_TEXTS, OWN_TEXTS, UPSTREAM_TEXTS
        values = list(OWN_TEXTS) + list(UPSTREAM_TEXTS) + list(FORK_TEXTS) + [None, 3, {}, "  v1;mods=I", "x MOD(", "V1"]
        for value in values:
            with self.subTest(value=value):
                spec = self.spec
                if isinstance(value, str) and spec.is_compact_arg(value):
                    expected_format = "own"
                elif spec.is_upstream_text(value):
                    expected_format = "upstream"
                elif spec.is_fork_text(value):
                    expected_format = "fork"
                else:
                    expected_format = None
                self.assertEqual(cc.value_format(value), expected_format)
                key = cc.INFOTEXT_KEY if expected_format == "own" else cc.FOREIGN_KEY
                expected = spec.decode({key: value}) if expected_format else None
                app = cc.from_value(value)
                if expected is None:
                    self.assertIsNone(app)
                else:
                    self.assertEqual(_keys(cc.config_values(app)), _keys(_ext_values(expected)))


# ── 체인·한눈에 보기·표 (A-CC4) ────────────────────────────────────────────────
def _deep(value):
    """dict/list 를 비교용 모양으로 — float 는 repr(비트까지), dict 열쇠 순서도 본다."""
    if isinstance(value, dict):
        return ("dict", tuple((k, _deep(v)) for k, v in value.items()))
    if isinstance(value, (list, tuple)):
        return ("list", tuple(_deep(v) for v in value))
    return _key(value)


def _is_coerced(config) -> bool:
    """모든 칸이 ``coerce`` 를 거친 값인가(말뭉치에는 표가 막는 값을 직접 넣은 설정도 있다)."""
    return all(node[f.name] == cc.coerce(f, node[f.name], cc.node_choices(tag, f))
               for tag in cc.NODE_TAGS for node in (cc.node_values(config, tag),) for f in cc.node_fields(tag))


def _card_values(config):
    """카드가 만들 수 있는 값으로 바꾼 사본 — 숫자는 유효숫자 10자리(``to_infotext`` 의 정밀도), ``-0.0`` 은 ``0.0``
    (앱 상태 문자열이 -0 을 0 으로 쓴다). 문자열은 ``-0.0`` 을 기본값 0 과 같이 보고 쓰지 않으므로 패널 67칸과 0 의 부호만
    다를 수 있다 — 계산에서 0 의 부호는 정확히 0 인 표본의 부호 말고는 바꾸지 못한다."""
    out = config.copy()
    for tag in cc.NODE_TAGS:
        values = cc.node_values(out, tag)
        for f in cc.node_fields(tag):
            if f.kind == cc.FLOAT and isinstance(values[f.name], float):
                values[f.name] = float(format(values[f.name], ".10g")) + 0.0
    return out


def _chain_key(chain) -> tuple:
    return (_deep(chain.entries), tuple(chain.tags), tuple(chain.unresolved_masks), chain.needs_basis)


def _overview_text(html_text: str) -> str:
    import html
    import re

    inner = re.fullmatch(r'<div class="samextra-cc-summary">(.*)</div>', html_text, re.S)
    assert inner is not None, html_text
    return html.unescape(inner.group(1))


@requires_extension
@requires_torch
class ChainParityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from tests._colorcraft_corpus import corpus, to_ext
        cls.spec = _ext("colorcraft.spec")
        cls.panel = _ext("colorcraft.panel_state")
        cls.engine = _ext("colorcraft.engine")
        cls.configs = corpus()
        cls.to_ext = staticmethod(lambda c: to_ext(cls.spec, c))

    def test_build_chain(self):
        nonempty = {False: 0, True: 0}
        for i, config in enumerate(self.configs):
            ext_config = self.to_ext(config)
            for hires in (False, True):
                app = cc.build_chain(config, hires)
                with self.subTest(i=i, hires=hires):
                    self.assertEqual(_chain_key(app), _chain_key(self.spec.build_chain(ext_config, hires)))
                    self.assertEqual(app.needs_basis, self.engine.needs_basis(app.entries))
                nonempty[hires] += bool(app.entries)
        self.assertGreater(min(nonempty.values()), 150, nonempty)

    def test_mask_specs(self):
        for i, config in enumerate(self.configs):
            with self.subTest(i=i):
                self.assertEqual(_deep(cc.mask_specs(config)), _deep(self.spec.build_mask_specs(self.to_ext(config))))

    def test_compact_equals_panel(self):
        """보낸 문자열(나머지 66칸은 ``ui()`` 기본값)과 확장 패널의 67칸이 같은 설정에 같은 체인을 만든다 — 설계 §4.1 의
        CPU 실측을 말뭉치 전체로. 문자열은 숫자를 유효숫자 10자리로 쓰고 -0 을 쓰지 않는다(``_card_values``) — 그래서
        카드가 만들 수 있는 값으로 바꾼 설정끼리 비교하고, 바꾼 설정이 실제로 있는지도 센다."""
        defaults = self.panel.default_args()
        compared = only_digits = 0
        for i, original in enumerate(self.configs):
            if not original.enabled:
                continue
            config = _card_values(original)
            if cc.to_infotext(config) != cc.to_infotext(original):
                only_digits += 1
            ext_config = self.to_ext(config)
            compact = self.panel.config_from_script_args(cc.as_block(config)["args"] + defaults[1:])
            panel = self.panel.config_from_script_args(self.panel.args_from_config(ext_config, rev="r1"))
            for hires in (False, True):
                with self.subTest(i=i, hires=hires):
                    expected = _chain_key(self.spec.build_chain(panel, hires))
                    self.assertEqual(_chain_key(self.spec.build_chain(compact, hires)), expected)
                    self.assertEqual(_chain_key(cc.build_chain(cc.parse_script_block(cc.as_block(config)), hires)),
                                     expected)
            compared += 1
        self.assertGreater(compared, 450)
        self.assertGreater(only_digits, 50)                    # 말뭉치에 17자리·-0 값이 실제로 있다

    def test_card_values_round_trip_exactly(self):
        """카드가 만들 수 있는 값은 문자열을 거쳐도 같은 체인이다(표가 막는 값을 넣은 설정은 빼고)."""
        checked = 0
        for i, original in enumerate(self.configs):
            config = _card_values(original)
            if not _is_coerced(config):
                continue
            checked += 1
            with self.subTest(i=i):
                back = cc.parse_own(cc.to_infotext(config))
                for hires in (False, True):
                    if config.enabled:
                        self.assertEqual(_chain_key(cc.build_chain(back, hires)), _chain_key(cc.build_chain(config, hires)))
        self.assertGreater(checked, 400)

    def test_overview(self):
        for i, config in enumerate(self.configs):
            mod_choices, mask_choices, html_text = self.panel.overview(self.to_ext(config))
            app = cc.overview(config)
            with self.subTest(i=i):
                self.assertEqual(app.modifier_marks, tuple(tuple(c) for c in mod_choices))
                self.assertEqual(app.mask_marks, tuple(tuple(c) for c in mask_choices))
                self.assertEqual(app.text, _overview_text(html_text))
                self.assertEqual(cc.describe(config), app.text)
                used, unresolved = self.panel._used_masks(self.to_ext(config))
                self.assertEqual(cc.used_masks(config), (used, unresolved))

    def test_schema_holds_the_extension_schema(self):
        ext = self.panel.schema()
        app = cc.schema()
        for key, value in ext.items():
            with self.subTest(key=key):
                if key == "initialRev":
                    continue                                     # 앱 상태에는 rev 가 없다
                if key in ("modifierFields", "leafFields", "comboFields"):
                    self.assertEqual([{k: f[k] for k in ("name", "default")} for f in app[key]], value)
                    continue
                self.assertEqual(app[key], value)


if __name__ == "__main__":
    unittest.main()
