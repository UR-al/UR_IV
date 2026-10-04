"""Anima SPEED(sam-extra ``scripts/anima_speed.py``) — core/anima_speed 순수 로직.

part 1 (torch 없음):
* ``FixtureTests`` — 녹화된 script-info(tests/fixtures/sam_extra_script_info.json, t2i·i2i)의 14 인자 모양·선택지·범위·
  기본값이 앱 상수와 같다.
* ``ExtensionConstantTests`` — 설치된 확장 소스의 상수(AST 로 읽음 — 아무것도 import 하지 않는다)와 앱 상수가 같다:
  이름·별칭·infotext 키·옵션·선택지·프리셋·기본값·``SpeedSettings`` 필드와 기본값·붙여 넣기 표.
* ``CardSettingsTests`` · ``BlockTests`` · ``CoerceTests`` · ``PasteTests`` · ``ValidateTests`` · ``StaticSkipTests`` ·
  ``PlanTests`` · ``DescribeTests`` · ``ComfyConstantTests`` · ``ModuleTests`` — 앱 규칙.

part 2 — 설치된 확장을 시험 대조 기준으로 같은 입력을 넣는다(확장 코드는 앱에 옮기지 않는다). **sam-extra 0.33.0 에
고정**(``a/0`` → 'division by zero' 설정 오류, 아주 큰 정수 → 기본값, UniPC bh2 거절 — 0.32.1 에서는 그 행들이 일부러
실패한다). 0.33.0 이 설치되기 전에는 AISTUDIO_FORGE_EXTENSION_DIR 로 그 작업 트리를 가리킨다:
* ``CoerceParityTests`` (torch — sam3ext.speed 가 import 한다): 139행 표(위치·dict·별칭·텍스트·숫자·불리언·선택지·시드,
  ``a/0``·아주 큰 정수 행 포함 — 어느 행도 예외로 죽지 않는다) — ``coerce`` == ``forge_host.coerce_settings``: 모든
  필드(타입·float 비트)·``error`` 문구·``summary()``. ``parse_summary`` 도.
* ``BlockParityTests`` (torch): 앱이 보내는 JSON 블록을 확장이 읽은 값 == 카드에서 계산한 값.
* ``ValidateParityTests`` (torch): ``validate`` 오류 유무 == 확장의 T2I 판정(coerce 오류·``build_plan`` 의
  ``PlanError(kind="invalid")``) — latent 크기 셋 × 스케줄 셋에서. 실행 중에만 아는 것은 따로 적어 둔다.
* ``PasteParityTests`` (torch 없음): 확장 ``_paste``·``_paste_bool``·``_paste_int``·``parse_summary`` 를 AST 로 떼어 Forge
  붙여 넣기 루프로 돌린 값 == ``paste_values``.
확장이 없으면 skip, AISTUDIO_REQUIRE_FORGE_EXT=1 이면 실패(tests/_sam_extra_ext).
"""
from __future__ import annotations

import ast
import copy
import dataclasses
import importlib
import json
import math
import os
import sys
import unittest
from types import SimpleNamespace

from core import anima_speed as sp
from tests._optional_deps import load_torch, requires_torch
from tests._sam_extra_ext import EXT_ROOT, requires_extension

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIXTURE = os.path.join(ROOT, "tests", "fixtures", "sam_extra_script_info.json")
FORGE_HOST = "sam3ext/speed/forge_host.py"
SCHEDULE = "sam3ext/speed/schedule.py"
SCRIPT = "scripts/anima_speed.py"


def _fixture_entries():
    with open(FIXTURE, encoding="utf-8") as fh:
        scripts = json.load(fh)["scripts"]
    return [s for s in scripts if s.get("name") == sp.SCRIPT_NAME.lower()]


def _ext_tree(relative: str) -> ast.Module:
    return ast.parse((EXT_ROOT / relative).read_text(encoding="utf-8"))


def _ext_assign(relative: str, name: str) -> ast.expr:
    for item in _ext_tree(relative).body:
        if isinstance(item, ast.Assign) and any(isinstance(t, ast.Name) and t.id == name for t in item.targets):
            return item.value
        if isinstance(item, ast.AnnAssign) and isinstance(item.target, ast.Name) and item.target.id == name:
            return item.value
    raise AssertionError(f"{relative} has no module-level {name}")


def _ext_literal(relative: str, name: str):
    """확장 모듈의 최상위 리터럴(AST — import 하지 않는다)."""
    return ast.literal_eval(_ext_assign(relative, name))


def _schedule_namespace() -> SimpleNamespace:
    names = ("CUSTOM_DEFAULT_A", "CUSTOM_DEFAULT_BETA", "MODES", "THRESHOLDS", "TRANSFORMS", "PRESETS",
             "PRESET_NAMES", "REFERENCE_LATENT")
    return SimpleNamespace(**{name: _ext_literal(SCHEDULE, name) for name in names})


def _ext_expr(node: ast.expr):
    """``schedule.X`` 를 참조하는 확장 식 — schedule 리터럴만 든 이름공간에서 계산한다."""
    code = compile(ast.Expression(body=node), "<ext-expr>", "eval")
    return eval(code, {"__builtins__": {}}, {"schedule": _schedule_namespace()})  # noqa: S307 — 로컬 확장 소스


def _ext_class(relative: str, name: str) -> ast.ClassDef:
    for item in _ext_tree(relative).body:
        if isinstance(item, ast.ClassDef) and item.name == name:
            return item
    raise AssertionError(f"{relative} has no class {name}")


def card(**changes) -> sp.SpeedSettings:
    return dataclasses.replace(sp.APP_DEFAULTS, **changes)


# ─────────────────────────────────────────────────────────────────────────────
# part 1 — torch 없음
# ─────────────────────────────────────────────────────────────────────────────
class FixtureTests(unittest.TestCase):
    """녹화된 script-info(확장 ui())의 인자 모양·선택지·범위·기본값이 앱 상수와 같다."""

    def test_both_tabs_have_fourteen_args_in_arg_name_order_with_the_extension_defaults(self):
        entries = _fixture_entries()
        self.assertEqual(sorted(e["is_img2img"] for e in entries), [False, True])
        for entry in entries:
            with self.subTest(img2img=entry["is_img2img"]):
                args = entry["args"]
                self.assertEqual(len(args), len(sp.ARG_NAMES))
                values = dict(zip(sp.ARG_NAMES, (arg["value"] for arg in args)))
                self.assertEqual(values, dict(sp.DEFAULTS))
                by_name = dict(zip(sp.ARG_NAMES, args))
                self.assertEqual(tuple(by_name["mode"]["choices"]), sp.MODES)
                self.assertEqual(tuple(by_name["threshold"]["choices"]), sp.THRESHOLDS)
                self.assertEqual(tuple(by_name["preset"]["choices"]), sp.PRESET_NAMES)
                self.assertEqual(tuple(by_name["transform"]["choices"]), sp.TRANSFORMS)
                delta, divisor = by_name["delta"], by_name["sigma_divisor"]
                self.assertEqual((delta["minimum"], delta["maximum"], delta["step"]),
                                 (sp.DELTA_MIN, sp.DELTA_MAX, sp.DELTA_STEP))
                self.assertEqual((divisor["minimum"], divisor["maximum"], divisor["step"]),
                                 (sp.DIVISOR_MIN, sp.DIVISOR_MAX, sp.DIVISOR_STEP))
                for name in ("scales", "manual_sigmas", "spectrum_A", "spectrum_beta", "seed"):
                    self.assertIsNone(by_name[name]["choices"], name)

    def test_app_defaults_send_exactly_the_fixture_defaults(self):
        arg = sp.APP_DEFAULTS.as_arg()
        values = dict(zip(sp.ARG_NAMES, (a["value"] for a in _fixture_entries()[0]["args"])))
        self.assertEqual(arg, values)
        for name in sp.ARG_NAMES:
            self.assertIs(type(arg[name]), type(values[name]), name)


@requires_extension
class ExtensionConstantTests(unittest.TestCase):
    """설치된 확장 소스(AST)의 상수 == 앱 상수. 확장이 값을 바꾸면 여기서 먼저 깨진다."""

    def test_title_keys_and_options(self):
        self.assertEqual(_ext_literal(SCRIPT, "TITLE"), sp.SCRIPT_NAME)
        self.assertEqual(_ext_literal(SCRIPT, "XYZ_PREFIX"), sp.XYZ_PREFIX)
        self.assertEqual(_ext_literal(FORGE_HOST, "KEY"), sp.INFOTEXT_KEY)
        self.assertEqual(_ext_literal(FORGE_HOST, "STATUS_KEY"), sp.STATUS_KEY)
        self.assertEqual(_ext_literal(FORGE_HOST, "INFOTEXT_IMG2IMG_RESCALE"), sp.INFOTEXT_IMG2IMG_RESCALE)
        self.assertEqual(_ext_literal(FORGE_HOST, "OPT_IMG2IMG_RESCALE"), sp.OPT_IMG2IMG_RESCALE)
        self.assertEqual(_ext_literal(FORGE_HOST, "OPT_LOG"), sp.OPT_LOG)

    def test_argument_names_aliases_and_unsupported_samplers(self):
        self.assertEqual(_ext_literal(FORGE_HOST, "ARG_NAMES"), sp.ARG_NAMES)
        self.assertEqual(_ext_literal(FORGE_HOST, "ARG_ALIASES"), dict(sp.ARG_ALIASES))
        self.assertEqual(_ext_literal(FORGE_HOST, "UNSUPPORTED_SAMPLERS"), dict(sp.UNSUPPORTED_SAMPLERS))

    def test_schedule_choices_presets_and_custom_spectrum(self):
        for name in ("MODES", "THRESHOLDS", "TRANSFORMS", "PRESET_NAMES", "CUSTOM_DEFAULT_A", "CUSTOM_DEFAULT_BETA",
                     "REFERENCE_LATENT"):
            self.assertEqual(_ext_literal(SCHEDULE, name), getattr(sp, name), name)
        self.assertEqual(_ext_literal(SCHEDULE, "PRESETS"), sp.PRESETS)
        self.assertEqual(list(sp.PRESETS), list(_ext_literal(SCHEDULE, "PRESETS")))
        self.assertEqual(set(sp.PRESETS), set(sp.PRESET_NAMES))

    def test_defaults_dict(self):
        self.assertEqual(_ext_expr(_ext_assign(FORGE_HOST, "DEFAULTS")), dict(sp.DEFAULTS))
        self.assertEqual(list(_ext_expr(_ext_assign(FORGE_HOST, "DEFAULTS"))), list(sp.ARG_NAMES))

    def test_speed_settings_fields_and_defaults(self):
        fields = []
        for item in _ext_class(FORGE_HOST, "SpeedSettings").body:
            if isinstance(item, ast.AnnAssign) and isinstance(item.target, ast.Name):
                fields.append((item.target.id, _ext_expr(item.value)))
        app = [(f.name, f.default) for f in dataclasses.fields(sp.ScriptSettings)]
        self.assertEqual(fields[-1], ("xyz", ()))     # XYZ 덮어쓰기 기록 — 앱 거울에는 없다
        self.assertEqual(fields[:-1], app)

    def test_paste_table_matches_the_script_infotext_fields(self):
        ui = next(item for item in _ext_class(SCRIPT, "AnimaSpeed").body
                  if isinstance(item, ast.FunctionDef) and item.name == "ui")
        returned = next(node.value for node in ast.walk(ui) if isinstance(node, ast.Return))
        order = [element.id for element in returned.elts]
        self.assertEqual(len(order), len(sp.ARG_NAMES))
        component_field = dict(zip(order, sp.ARG_NAMES))
        fields = next(node.value for node in ast.walk(ui) if isinstance(node, ast.Assign)
                      and any(isinstance(t, ast.Attribute) and t.attr == "infotext_fields" for t in node.targets))
        rows = []
        for element in fields.elts:
            component, reader = element.elts
            field = component_field[component.id]
            if isinstance(reader, ast.Lambda):
                self.assertEqual(ast.unparse(reader), "lambda params: KEY in params")
                self.assertEqual(field, "enabled")
                continue
            self.assertEqual(reader.func.id, "_paste")
            key, convert = reader.args
            rows.append((field, key.value, convert.id))
        self.assertEqual(rows, [(field, key, convert.__name__) for field, key, convert in sp.PASTE_FIELDS])


class CardSettingsTests(unittest.TestCase):
    def test_app_defaults_are_the_extension_defaults_plus_i2i_off(self):
        self.assertIs(sp.APP_DEFAULTS, sp.EXTENSION_DEFAULTS)
        self.assertEqual(sp.APP_DEFAULTS.as_arg(), dict(sp.DEFAULTS))
        self.assertFalse(sp.APP_DEFAULTS.apply_img2img)

    def test_widget_ids(self):
        self.assertEqual(sp.widget_id("enabled"), "_aspeed_enabled")
        self.assertEqual(sp.widget_id("spectrum_A"), "_aspeed_spectrum_A")
        self.assertEqual(len(sp.WIDGET_KEYS), 15)
        self.assertEqual(set(sp.WIDGET_KEYS), set(sp.ARG_NAMES) | {"apply_img2img"})
        with self.assertRaises(KeyError):
            sp.widget_id("speed_enabled")
        self.assertEqual(sp.SETTINGS_KEY, "anima_speed_settings")

    def test_widget_values_round_trip(self):
        cases = [
            sp.APP_DEFAULTS,
            card(enabled=True, mode="respace", threshold="manual", preset="custom", scales="0.25, 0.5, 1",
                 delta=0.0001, sigma_divisor=2.0, manual_sigmas="0.8;0.6", adaptive=False, transform="fft",
                 spectrum_A=1e-12, spectrum_beta=-0.5, seed=123456789012, hires=True, apply_img2img=True),
            card(delta=0.1 + 0.2, sigma_divisor=1 / 3, spectrum_A=8664.998524, scales="", manual_sigmas="x"),
        ]
        for settings in cases:
            with self.subTest(settings=settings):
                values = sp.widget_values(settings)
                self.assertEqual(list(values), list(sp.WIDGET_KEYS))
                self.assertTrue(all(isinstance(v, str) for v in values.values()))
                self.assertEqual(sp.parse_settings(values), settings)

    def test_widget_number_text(self):
        values = sp.widget_values(card(delta=0.01, sigma_divisor=1.0, spectrum_A=203.615097, seed=-1))
        self.assertEqual((values["delta"], values["sigma_divisor"], values["spectrum_A"], values["seed"]),
                         ("0.01", "1", "203.615097", "-1"))

    def test_parse_settings_keeps_base_for_unreadable_fields(self):
        base = card(enabled=True, delta=0.05, preset="flux", seed=7)
        raw = {"enabled": "", "mode": "sideways", "threshold": None, "preset": "nope", "delta": "nan",
               "sigma_divisor": "inf", "spectrum_A": "x", "spectrum_beta": True, "seed": "", "transform": "DWT",
               "hires": "on", "adaptive": "off"}
        got = sp.parse_settings(raw, base=base)
        self.assertEqual(got, dataclasses.replace(base, transform="dwt", hires=True, adaptive=False))
        self.assertIs(sp.parse_settings(None, base=base), base)
        self.assertIs(sp.parse_settings("enabled", base=base), base)

    def test_parse_settings_normalises_like_the_extension(self):
        got = sp.parse_settings({"mode": " Respace ", "threshold": "MANUAL", "preset": "Krea-2", "seed": "-7.5",
                                 "scales": "  0.5,1.0  ", "manual_sigmas": " 0.7 ", "delta": " 0.02 "})
        self.assertEqual((got.mode, got.threshold, got.preset, got.seed, got.scales, got.manual_sigmas, got.delta),
                         ("respace", "manual", "krea-2", -1, "0.5,1.0", "0.7", 0.02))
        self.assertEqual(sp.parse_settings({"seed": "12.9"}).seed, 12)
        self.assertEqual(sp.parse_settings({"seed": 3}).seed, 3)

    def test_text_fields_keep_what_was_typed(self):
        got = sp.parse_settings({"scales": "1/2,1", "manual_sigmas": ""})
        self.assertEqual((got.scales, got.manual_sigmas), ("1/2,1", ""))


class BlockTests(unittest.TestCase):
    def test_block_is_one_dict_with_the_fourteen_arg_names(self):
        settings = card(enabled=True, scales="0.5,1", seed=42, hires=True)
        block = sp.as_block(settings)
        self.assertEqual(list(block), ["args"])
        self.assertEqual(len(block["args"]), 1)
        arg = block["args"][0]
        self.assertEqual(list(arg), list(sp.ARG_NAMES))
        self.assertIs(arg["enabled"], True)
        self.assertIs(arg["adaptive"], True)
        self.assertIs(arg["hires"], True)
        self.assertEqual(arg["scales"], "0.5,1")          # 친 그대로 — Forge 가 같은 규칙으로 다시 읽는다
        self.assertEqual(arg["seed"], 42)
        self.assertEqual(json.loads(json.dumps(block)), block)

    def test_design_example_block(self):
        self.assertEqual(sp.as_block(card(enabled=True)), {"args": [{
            "enabled": True, "mode": "transition", "threshold": "neo_shift", "preset": "anima",
            "scales": "0.5,1.0", "delta": 0.01, "sigma_divisor": 1.03, "manual_sigmas": "0.7",
            "adaptive": True, "transform": "dct", "spectrum_A": 203.615097, "spectrum_beta": 1.915461,
            "seed": -1, "hires": False,
        }]})

    def test_parse_script_block_reads_dict_positional_and_alias_forms_alike(self):
        settings = card(enabled=True, mode="respace", threshold="manual", preset="custom", scales="0.25,0.5,1",
                        manual_sigmas="0.6", spectrum_A=12.5, spectrum_beta=2.5, seed=9, hires=True)
        arg = settings.as_arg()
        positional = [arg[name] for name in sp.ARG_NAMES]
        aliased = {({v: k for k, v in sp.ARG_ALIASES.items()}).get(name, name): value for name, value in arg.items()}
        want = sp.script_settings(settings)
        self.assertEqual(sp.parse_script_block(sp.as_block(settings)), want)
        self.assertEqual(sp.parse_script_block({"args": positional}), want)
        self.assertEqual(sp.parse_script_block({"args": [aliased]}), want)
        self.assertEqual(sp.parse_script_block({"args": tuple(positional[:3])}),
                         dataclasses.replace(sp.ScriptSettings(), enabled=True, mode="respace", threshold="manual"))

    def test_parse_script_block_edges(self):
        self.assertIsNone(sp.parse_script_block(None))
        self.assertIsNone(sp.parse_script_block(["args"]))
        self.assertEqual(sp.parse_script_block({}), sp.ScriptSettings())
        self.assertEqual(sp.parse_script_block({"args": "x"}), sp.ScriptSettings())

    def test_parse_script_block_reads_a_zero_denominator_like_any_bad_value(self):
        """sam-extra 0.33.0: ``a/0`` 은 그 칸의 설정 오류(해상도 비율은 늘, 수동 σ 는 threshold 가 manual 일 때만) — 0.32.1 은
        ZeroDivisionError 로 스크립트가 죽었다. 어느 블록도 예외를 올리지 않는다."""
        cases = (
            ({"args": [{"enabled": True, "scales": "1/0,1"}]}, True, "scales: division by zero: '1/0'", {}),
            ({"args": [{"manual": "0/0"}]}, False, None, {}),
            ({"args": [{"threshold": "manual", "manual": "0.9, 0.7/0"}]}, False,
             "manual sigmas: division by zero: '0.7/0'", {"threshold": "manual"}),
            ({"args": [True, "transition", "neo_shift", "anima", "0.5,1", 0.01, 1.03, "1/0"]}, True, None, {}),
            ({"args": [True, "respace", "neo_shift", "anima", "0.5,1/-0"]}, True, "scales: division by zero: '1/-0'",
             {"mode": "respace"}),
        )
        for block, enabled, error, fields in cases:
            with self.subTest(block=block):
                got = sp.parse_script_block(block)
                self.assertEqual(got, sp.coerce(block["args"]))
                self.assertIs(got.enabled, enabled)
                self.assertEqual(got.error, error)
                self.assertEqual(dataclasses.replace(got, enabled=False, error=None),
                                 dataclasses.replace(sp.ScriptSettings(), **fields))


class CoerceTests(unittest.TestCase):
    """앱 쪽 성질(확장 대조는 part 2 ``CoerceParityTests``)."""

    def test_defaults(self):
        self.assertEqual(sp.coerce([]), sp.ScriptSettings())
        self.assertEqual(sp.coerce(None), sp.ScriptSettings())
        self.assertEqual(sp.coerce([{}]), sp.ScriptSettings())
        self.assertIsNone(sp.coerce([]).error)

    def test_integers_too_large_for_a_float_read_as_the_defaults(self):
        """sam-extra 0.33.0 (_as_float 가 OverflowError 도 잡는다) — 0.32.1 은 예외가 그대로 나갔다."""
        huge = 10 ** 400
        got = sp.coerce([{"delta": huge, "sigma_divisor": -huge, "seed": huge, "spectrum_A": huge,
                          "spectrum_beta": -huge, "preset": "custom"}])
        self.assertEqual(got, sp.ScriptSettings(preset="custom"))
        self.assertEqual(sp.coerce([True, "transition", "neo_shift", "anima", "0.5,1.0", huge, huge]),
                         sp.ScriptSettings(enabled=True))

    def test_script_settings_of_a_card_is_the_card(self):
        settings = card(enabled=True, mode="respace", threshold="delta_optimal", preset="custom", scales="1/4;0.5;1",
                        delta=0.2, sigma_divisor=1.5, manual_sigmas="0.9,0.5", adaptive=False, transform="dwt",
                        spectrum_A=5.0, spectrum_beta=1.0, seed=3, hires=True)
        got = sp.script_settings(settings)
        self.assertEqual(got, sp.ScriptSettings(
            enabled=True, mode="respace", threshold="delta_optimal", preset="custom", scales=(0.25, 0.5, 1.0),
            delta=0.2, sigma_divisor=1.5, manual_sigmas=(0.9, 0.5), adaptive=False, transform="dwt", spectrum_A=5.0,
            spectrum_beta=1.0, seed=3, hires=True, error=None))

    def test_error_text(self):
        got = sp.coerce([{"mode": "x", "scales": "0.5", "threshold": "manual", "manual_sigmas": "0.5,0.7"}])
        self.assertEqual(got.error, "mode must be one of transition, respace (got 'x'); "
                                    "last scale must equal 1.0 (full resolution); got 0.5; "
                                    "manual sigmas must be strictly decreasing; got [0.5, 0.7]")
        self.assertEqual((got.mode, got.scales, got.manual_sigmas), ("transition", (0.5, 1.0), (0.7,)))

    def test_summary_format(self):
        self.assertEqual(sp.ScriptSettings().summary(),
                         "mode=transition; threshold=neo_shift; preset=anima; scales=0.5,1; delta=0.01; "
                         "divisor=1.03; manual=0.7; adaptive=True; transform=dct; seed=-1; hires=False")
        custom = sp.coerce([{"preset": "custom", "A": 1e-12, "beta": 2.0, "seed": 5, "hires": True}])
        self.assertEqual(custom.summary(),
                         "mode=transition; threshold=neo_shift; preset=custom; scales=0.5,1; delta=0.01; "
                         "divisor=1.03; manual=0.7; adaptive=True; transform=dct; A=1e-12; beta=2; seed=5; hires=True")

    def test_summary_of_a_card(self):
        self.assertIsNone(sp.summary(sp.APP_DEFAULTS))
        # a/0 도 다른 설정 오류처럼 요약은 남는다(해상도 비율은 기본값 — sam-extra 0.33.0, 0.32.1 은 스크립트 오류로 없음)
        self.assertEqual(sp.summary(card(enabled=True, scales="1/0")), sp.ScriptSettings().summary())
        self.assertEqual(sp.summary(card(enabled=True)), sp.ScriptSettings().summary())

    def test_parse_summary_round_trip(self):
        text = sp.coerce([{"preset": "custom", "scales": "0.25,0.5,1", "manual": "0.8,0.6"}]).summary()
        parsed = sp.parse_summary(text)
        self.assertEqual(parsed["scales"], "0.25,0.5,1")
        self.assertEqual(parsed["manual"], "0.8,0.6")
        self.assertEqual(parsed["A"], "203.615097")
        self.assertEqual(sp.coerce([parsed]), sp.coerce([{"preset": "custom", "scales": "0.25,0.5,1",
                                                           "manual": "0.8,0.6"}]))
        self.assertEqual(sp.parse_summary(None), {})
        self.assertEqual(sp.parse_summary(" a = 1 ; b=x=y; c ;"), {"a": "1", "b": "x=y"})


class PasteTests(unittest.TestCase):
    TEXT = ("mode=respace; threshold=manual; preset=custom; scales=0.25,0.5,1; delta=0.02; divisor=1.1; "
            "manual=0.8,0.6; adaptive=False; transform=fft; A=12; beta=2; seed=77; hires=True")

    def test_paste_values(self):
        got = sp.paste_values({"Anima SPEED": self.TEXT})
        self.assertEqual(got, {"enabled": True, "mode": "respace", "threshold": "manual", "preset": "custom",
                               "scales": "0.25,0.5,1", "delta": 0.02, "sigma_divisor": 1.1,
                               "manual_sigmas": "0.8,0.6", "adaptive": False, "transform": "fft",
                               "spectrum_A": 12.0, "spectrum_beta": 2.0, "seed": 77, "hires": True})

    def test_quoted_value_is_unquoted(self):
        quoted = json.dumps(self.TEXT)
        self.assertEqual(sp.paste_values({"Anima SPEED": quoted}), sp.paste_values({"Anima SPEED": self.TEXT}))

    def test_missing_key_disables_and_changes_nothing_else(self):
        got = sp.paste_values({"Anima SPEED status": "base: applied"})
        self.assertEqual(got, {"enabled": False, **{field: None for field, _k, _c in sp.PASTE_FIELDS}})
        self.assertEqual(sp.paste_values(None)["enabled"], False)
        base = card(enabled=True, delta=0.3, apply_img2img=True)
        self.assertEqual(sp.from_infotext({}, base=base), dataclasses.replace(base, enabled=False))

    def test_unconvertible_values_become_none(self):
        got = sp.paste_values({"Anima SPEED": "delta=x; seed=inf; A=; adaptive=maybe; mode="})
        self.assertIsNone(got["delta"])
        self.assertIsNone(got["seed"])           # OverflowError — Forge 의 붙여 넣기 루프가 None
        self.assertIsNone(got["spectrum_A"])
        self.assertIs(got["adaptive"], False)    # _paste_bool: 모르는 말은 False
        self.assertEqual(got["mode"], "")        # str 변환은 늘 된다(카드는 모르는 선택지라 무시)

    def test_from_infotext(self):
        got = sp.from_infotext({"Anima SPEED": json.dumps(self.TEXT)}, base=card(apply_img2img=True))
        self.assertEqual(got, sp.SpeedSettings(
            enabled=True, mode="respace", threshold="manual", preset="custom", scales="0.25,0.5,1", delta=0.02,
            sigma_divisor=1.1, manual_sigmas="0.8,0.6", adaptive=False, transform="fft", spectrum_A=12.0,
            spectrum_beta=2.0, seed=77, hires=True, apply_img2img=True))

    def test_from_infotext_keeps_base_for_values_the_card_cannot_take(self):
        base = card(mode="respace", delta=0.3, seed=5)
        got = sp.from_infotext({"Anima SPEED": "mode=sideways; delta=nan; seed=-9; transform=DWT"}, base=base)
        self.assertEqual(got, dataclasses.replace(base, enabled=True, seed=-1, transform="dwt"))

    def test_summary_pastes_back_to_the_same_script_settings(self):
        for settings in (card(enabled=True), card(enabled=True, preset="custom", spectrum_A=0.5, spectrum_beta=3.25,
                                                  scales="0.25,0.5,1", seed=11, hires=True, mode="respace")):
            with self.subTest(settings=settings):
                text = sp.summary(settings)
                pasted = sp.from_infotext({"Anima SPEED": text})
                self.assertEqual(sp.script_settings(pasted), sp.script_settings(settings))


class ValidateTests(unittest.TestCase):
    CASES = (
        (card(), ()),
        (card(scales="0.5"), (sp.ERR_SCALES,)),
        (card(scales="1,0.5"), (sp.ERR_SCALES,)),
        (card(scales=""), (sp.ERR_SCALES,)),
        (card(scales="nan,1"), (sp.ERR_SCALES,)),
        (card(scales="1/0,1"), (sp.ERR_SCALES,)),                # 0.33.0: 0으로 나눔도 그 칸의 오류
        (card(scales="x,1", manual_sigmas="1/0"), (sp.ERR_SCALES,)),
        (card(manual_sigmas="0/0"), ()),                         # threshold ≠ manual 이면 수동 σ 칸은 무시(0.33.0)
        (card(threshold="manual", manual_sigmas="0.7/0"), (sp.ERR_MANUAL,)),
        (card(manual_sigmas="x"), ()),                           # threshold ≠ manual 이면 오류 아님
        (card(threshold="manual", manual_sigmas="x"), (sp.ERR_MANUAL,)),
        (card(threshold="manual", manual_sigmas="0.5,0.7"), (sp.ERR_MANUAL,)),
        (card(threshold="manual", manual_sigmas="0.5,0.7", mode="respace", scales="0.25,0.5,1"), ()),
        (card(threshold="manual", manual_sigmas="0.7,0.5"), (sp.ERR_MANUAL_COUNT,)),
        (card(threshold="manual", manual_sigmas="0.7", scales="0.25,0.5,1"), (sp.ERR_MANUAL_COUNT,)),
        (card(threshold="manual", manual_sigmas="0.7", scales="0.25,0.5,1", mode="respace"), ()),
        (card(delta=0.0), (sp.ERR_DELTA,)),
        (card(delta=1.0), (sp.ERR_DELTA,)),
        (card(delta=1.0, threshold="manual"), ()),
        (card(sigma_divisor=0.0), (sp.ERR_DIVISOR,)),
        (card(sigma_divisor=-1.0, threshold="delta_optimal"), (sp.ERR_DIVISOR,)),   # build_plan 이 검사
        (card(sigma_divisor=-1.0, threshold="manual"), ()),
        (card(preset="custom", spectrum_A=0.0), (sp.ERR_SPECTRUM_A,)),
        (card(preset="custom", spectrum_A=-3.0), (sp.ERR_SPECTRUM_A,)),
        (card(preset="custom", spectrum_A=1e-12), ()),
        (card(preset="custom", spectrum_A=-3.0, threshold="manual"), ()),
        (card(preset="flux", spectrum_A=-3.0), ()),             # 프리셋은 제 A 를 쓴다
        (card(preset="custom", spectrum_beta=-2.0), ()),        # β 는 유한하면 된다
        (card(scales="1.0", preset="custom", spectrum_A=-1.0), ()),    # 확장은 그 전에 '건너뜀'
        (card(transform="dwt"), ()),
        (card(transform="dwt", scales="0.25,0.5,1"), ()),
        (card(transform="dwt", scales="0.3,1"), (sp.ERR_DWT_RATIO,)),
        (card(transform="dwt", scales="0.5001,1"), (sp.ERR_DWT_RATIO,)),        # 1e-6 허용치(확장과 같다)
        (card(transform="dwt", scales="0.2,0.5,1"), (sp.ERR_DWT_RATIO,)),
        (card(mode="sideways"), (sp.ERR_CHOICE,)),
        (card(mode="x", preset="y", scales="2,1"), (sp.ERR_CHOICE, sp.ERR_SCALES)),
        (card(delta=0.0, sigma_divisor=0.0), (sp.ERR_DELTA, sp.ERR_DIVISOR)),
    )

    def test_cases(self):
        for settings, codes in self.CASES:
            with self.subTest(settings=settings):
                self.assertEqual(sp.validate(settings), codes)

    def test_every_code_has_a_message(self):
        codes = {value for name, value in vars(sp).items() if name.startswith("ERR_")}
        self.assertEqual(codes, set(sp.VALIDATION_MESSAGES))
        self.assertEqual(sp.validation_messages(card(preset="custom", spectrum_A=0.0)),
                         ("Spectrum A 는 0보다 커야 합니다",))
        self.assertEqual(sp.validation_messages(card()), ())

    def test_coerce_codes_follow_the_error_text(self):
        """coerce 의 오류 코드는 ``error`` 문구와 같은 자리에서 나온다 — 코드가 있으면 문구도 있고, 없으면 없다."""
        for settings, codes in self.CASES:
            with self.subTest(settings=settings):
                script, coerce_codes = sp._coerce(sp.as_block(settings)["args"])
                self.assertEqual(bool(coerce_codes), script.error is not None)
                if coerce_codes:
                    self.assertEqual(sp.validate(settings), tuple(dict.fromkeys(coerce_codes)))
        script, codes = sp._coerce([{"mode": "a", "threshold": "b", "preset": "c", "transform": "d", "scales": "",
                                     "delta": 0, "sigma_divisor": 0}])
        self.assertEqual(codes, (sp.ERR_CHOICE,) * 4 + (sp.ERR_SCALES, sp.ERR_DELTA, sp.ERR_DIVISOR))
        self.assertEqual(script.error.count(" must be one of ") + script.error.count("unknown spectrum preset"), 4)


class StaticSkipTests(unittest.TestCase):
    def test_reasons(self):
        on = card(enabled=True)
        self.assertIsNone(sp.static_skip_reason(on, {}))
        self.assertIsNone(sp.static_skip_reason(on, None))
        self.assertEqual(sp.static_skip_reason(on, {"mask": "iVBORw0KGgo="}), sp.SKIP_INPAINT)
        self.assertIsNone(sp.static_skip_reason(on, {"mask": "  "}))
        self.assertEqual(sp.static_skip_reason(on, {"use_image_alpha_as_mask": True}), sp.SKIP_INPAINT)
        cn = {"alwayson_scripts": {"ControlNet": {"args": [{"enabled": False}, {"enabled": True, "model": "x"}]}}}
        self.assertEqual(sp.static_skip_reason(on, cn), sp.SKIP_CONTROLNET)
        off_cn = {"alwayson_scripts": {"controlnet": {"args": [{"enabled": False}]}}}
        self.assertIsNone(sp.static_skip_reason(on, off_cn))
        self.assertEqual(sp.static_skip_reason(on, {"spectrum_enabled": True}, backend="comfyui"),
                         sp.SKIP_COMFY_SPECTRUM)
        self.assertIsNone(sp.static_skip_reason(on, {"spectrum_enabled": True}, backend="webui"))
        self.assertEqual(sp.static_skip_reason(card(scales="1.0"), {}), sp.SKIP_NO_COARSE_STAGE)
        self.assertIsNone(sp.static_skip_reason(card(scales="1/0"), {}))   # 설정 오류는 validate 몫
        self.assertEqual(sp.static_skip_reason(card(scales="1.0"), {"mask": "x"}), sp.SKIP_INPAINT)

    def test_every_reason_has_a_message(self):
        codes = {value for name, value in vars(sp).items() if name.startswith("SKIP_") and name != "SKIP_MESSAGES"}
        self.assertEqual(codes, set(sp.SKIP_MESSAGES))


class PlanTests(unittest.TestCase):
    def test_off_krea2_comfy_pending_and_i2i(self):
        on = card(enabled=True)
        self.assertEqual(sp.plan(sp.APP_DEFAULTS, target="t2i", backend="webui").reason, sp.REASON_OFF)
        self.assertEqual(sp.plan(on, target="t2i", backend="krea2").reason, sp.REASON_KREA2)
        pending = sp.plan(on, target="t2i", backend="comfyui")
        self.assertEqual((pending.block, pending.reason), (None, sp.REASON_COMFY_PENDING))
        self.assertEqual(sp.plan(on, target="i2i", backend="webui").reason, sp.REASON_I2I_OFF)
        sent = sp.plan(dataclasses.replace(on, apply_img2img=True), target="i2i", backend="webui")
        self.assertEqual(sent.reason, sp.REASON_SEND)

    def test_send(self):
        on = card(enabled=True, hires=True)
        for target in ("t2i", "aux"):
            with self.subTest(target=target):
                got = sp.plan(on, target=target, backend="webui")
                self.assertEqual(got, sp.SpeedPlan(sp.as_block(on), None, sp.REASON_SEND, sp.PROVENANCE_USER, ()))

    def test_invalid_settings_are_still_sent_with_the_codes(self):
        bad = card(enabled=True, preset="custom", spectrum_A=-1.0)
        with self.assertLogs("core.anima_speed", level="INFO"):
            got = sp.plan(bad, target="t2i", backend="webui")
        self.assertEqual((got.block, got.reason, got.errors), (sp.as_block(bad), sp.REASON_SEND, (sp.ERR_SPECTRUM_A,)))


class DescribeTests(unittest.TestCase):
    def test_lines(self):
        self.assertEqual(sp.describe(card(enabled=True)), "transition · neo_shift ÷1.03 δ 0.01 적응형 · anima · 0.5,1.0 · dct")
        self.assertEqual(sp.describe(card(threshold="delta_optimal", adaptive=False, preset="custom", spectrum_A=12.0,
                                          spectrum_beta=2.5, transform="fft", seed=4, hires=True, apply_img2img=True)),
                         "transition · delta_optimal δ 0.01 · custom A 12 β 2.5 · 0.5,1.0 · fft · 시드 4 · Hires · I2I")
        self.assertEqual(sp.describe(card(mode="respace", threshold="manual", manual_sigmas="0.7")),
                         "respace · manual σ 0.7 · 0.5,1.0 · dct")
        self.assertEqual(sp.describe(card(delta=0.0)), "⚠ 설정 오류 · transition · neo_shift ÷1.03 δ 0 적응형 · anima · "
                                                       "0.5,1.0 · dct")


class ComfyConstantTests(unittest.TestCase):
    def test_node_inputs_cover_every_script_argument(self):
        self.assertEqual(sp.COMFY_NODE_CLASS, "ForgeNeoAnimaSpeed")
        self.assertEqual(sp.COMFY_INPUTS[0], "model")
        # hires 는 노드 입력이 아니다 — 컴파일러가 hires 가 켜졌을 때만 Hires 패스에 노드를 넣고 pass_label 로 알린다
        self.assertEqual(sp.COMFY_INPUTS[1:14], tuple(name for name in sp.ARG_NAMES if name != "hires"))
        self.assertEqual(sp.COMFY_INPUTS[14:], ("pass_label", "image_seed", "img2img_rescale"))
        self.assertEqual(sp.COMFY_PASS_LABELS, ("base", "img2img", "hires"))
        self.assertEqual(sp.COMFY_MIN_PACK_VERSION, "1.7.0")

    def test_comfy_img2img_rescale_is_the_option_default(self):
        from core import forge_override_settings as fos
        spec = next(s for s in fos.SPECS if s.key == sp.OPT_IMG2IMG_RESCALE)
        self.assertIs(sp.COMFY_IMG2IMG_RESCALE, spec.default)


class ModuleTests(unittest.TestCase):
    def test_imports_only_the_standard_library_and_pure_app_modules(self):
        with open(os.path.join(ROOT, "core", "anima_speed.py"), encoding="utf-8") as fh:
            tree = ast.parse(fh.read())
        modules = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                modules.update(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                modules.add(node.module)
        self.assertEqual(modules, {"__future__", "logging", "math", "re", "dataclasses", "types", "typing",
                                   "core.forge_override_settings", "core.infotext"})

    def test_title_is_the_single_source(self):
        from core import sam_extra_capabilities as caps
        self.assertEqual(caps.TITLE_SPEED, sp.SCRIPT_NAME.lower())
        self.assertEqual(sp.INFOTEXT_KEY, sp.SCRIPT_NAME)
        self.assertEqual(sp.INFOTEXT_KEYS, ("Anima SPEED", "Anima SPEED status", "Anima SPEED img2img rescale"))

    def test_numbers_and_preset_params(self):
        self.assertEqual(sp.parse_scales("1/4; 0.5 ,, 1"), (0.25, 0.5, 1.0))
        self.assertEqual(sp.parse_scales([0.5, "1"]), (0.5, 1.0))
        self.assertEqual(sp.parse_sigmas("0.9,0.9", strictly_decreasing=False), (0.9, 0.9))
        self.assertEqual(sp.preset_params("custom", 3.0, 4.0), (3.0, 4.0, 128, None))
        self.assertEqual(sp.preset_params("flux2"), (520.495848, 1.915461, 64, None))
        self.assertEqual(sp.preset_params("anima"), (8664.998524, 2.422687, 128, 3.0))
        self.assertTrue(math.isfinite(sp.CUSTOM_DEFAULT_A))


# ─────────────────────────────────────────────────────────────────────────────
# part 2 — 설치된 확장과 대조(torch: sam3ext.speed 가 import 한다). 붙여 넣기 표는 AST 라 torch 없음.
# ─────────────────────────────────────────────────────────────────────────────
def _ext_speed(name: str):
    root = str(EXT_ROOT)
    if root not in sys.path:
        sys.path.insert(0, root)
    return importlib.import_module(f"sam3ext.speed.{name}")


def _canon(value):
    """비교용 모양 — float 는 비트(hex), bool/int/str/None 은 타입까지, 튜플·리스트는 원소마다."""
    if isinstance(value, float):
        return ("float", value.hex())
    if isinstance(value, (list, tuple)):
        return (type(value).__name__, tuple(_canon(item) for item in value))
    return (type(value).__name__, value)


def _canon_settings(settings) -> tuple:
    names = [f.name for f in dataclasses.fields(sp.ScriptSettings)]
    return tuple((name, _canon(getattr(settings, name))) for name in names)


def _outcome(fn, *args):
    try:
        result = fn(*args)
    except Exception as exc:  # 확장이 예외로 죽는 입력도 계약이다(타입·문구)
        return ("raise", type(exc).__name__, str(exc))
    return ("ok", _canon_settings(result), result.summary())


_BIG = float(2 ** 70)
# coerce_settings 표 — 위치 인자(짧은 꼬리 포함)·dict·별칭·텍스트 규칙·숫자·불리언·선택지·시드.
COERCE_ROWS = (
    [], None, (), [True], [True, "respace"], [True, "transition", "manual"],
    [True, "transition", "neo_shift", "anima", "0.5,1.0", 0.01, 1.03, "0.7", True, "dct", 203.615097, 1.915461, -1,
     False],
    [True, "respace", "delta_optimal", "flux", "0.25,0.5,1", 0.2, 1.5, "0.9,0.4", False, "fft", 1.0, 2.0, 5, True,
     "extra trailing value"],
    [{}], [{"enabled": "yes"}], [{"enabled": "False"}], [{"enabled": "0"}], [{"enabled": 0}], [{"enabled": None}],
    [{"enabled": "maybe"}], [{"enabled": 2}], [{"enabled": 1.5}], [{"enabled": " ENABLED "}], [{"enabled": "off"}],
    [{"enabled": "enable"}], [{"enabled": " Disable "}], [{"hires": "Enable"}], [{"adaptive": "disabled"}],
    [{"divisor": 1.2, "manual": "0.6", "A": 5, "beta": 1}],
    [{"sigma_divisor": 1.1, "divisor": 1.5}], [{"divisor": 1.5, "sigma_divisor": 1.1}],
    [{"foo": 1, 1: "x", "preset": "flux"}],
    [{"scales": "1/2,1"}], [{"scales": "0.5;1.0"}], [{"scales": "0.9999999"}], [{"scales": "0.5,0.9999999"}],
    [{"scales": "1.0"}], [{"scales": "1"}], [{"scales": ""}], [{"scales": "  "}], [{"scales": None}],
    [{"scales": "0.5,1.5"}], [{"scales": "0,1"}], [{"scales": "-0.5,1"}], [{"scales": "0.5,0.5,1"}],
    [{"scales": "1,0.5"}], [{"scales": "x,1"}], [{"scales": "nan,1"}], [{"scales": "inf,1"}],
    [{"scales": "1e400,1"}], [{"scales": "1/2/3,1"}], [{"scales": " 0.25, 0.5 ,1 "}], [{"scales": [0.5, 1.0]}],
    [{"scales": (0.5, "1")}], [{"scales": [True, 1]}], [{"scales": "1/0,1"}], [{"scales": "0/0,1"}],
    [{"scales": "0.5,1/0"}], [{"scales": "1/0,x"}], [{"scales": ["1/-0", "1"]}],
    [{"scales": "1e-308/1e308,1"}], [{"scales": 0.5}],
    [{"threshold": "manual", "manual": "0.7,0.5", "scales": "0.25,0.5,1"}],
    [{"threshold": "manual", "manual": "0.5,0.7"}],
    [{"threshold": "manual", "manual": "0.5,0.7", "mode": "respace"}],
    [{"threshold": "manual", "manual": "0"}], [{"threshold": "manual", "manual": "1"}],
    [{"threshold": "manual", "manual": "1.0"}], [{"threshold": "manual", "manual": "-0.2"}],
    [{"threshold": "manual", "manual": ""}], [{"threshold": "manual", "manual": "x"}],
    [{"threshold": "manual", "manual": [0.7]}], [{"threshold": "manual", "manual": None}],
    [{"threshold": "manual", "manual": "0.7,0.7", "mode": "respace"}],
    [{"manual": "x"}], [{"manual": "0.5,0.7"}], [{"manual": "1/0"}], [{"threshold": "manual", "manual": "2/0"}],
    [{"threshold": "manual", "manual": "0.9,nan/0", "mode": "respace"}], [{"threshold": "delta_optimal", "manual": "0/0"}],
    [{"delta": 10 ** 400}], [{"sigma_divisor": -10 ** 400, "seed": 10 ** 400, "A": 10 ** 400, "beta": -10 ** 400,
                              "preset": "custom"}],
    [True, "transition", "neo_shift", "anima", "0.5,1.0", 10 ** 400, 10 ** 400, "0.7", True, "dct", 1.0, 1.0,
     -10 ** 400],
    [{"delta": 0}], [{"delta": 1}], [{"delta": -0.1}], [{"delta": 0.5}], [{"delta": "x"}], [{"delta": "nan"}],
    [{"delta": None}], [{"delta": True}], [{"delta": "1e-9"}], [{"delta": 1, "threshold": "manual"}],
    [{"delta": 1, "threshold": "delta_optimal"}],
    [{"sigma_divisor": 0}], [{"sigma_divisor": -1, "threshold": "delta_optimal"}], [{"sigma_divisor": "inf"}],
    [{"sigma_divisor": -1, "threshold": "manual"}], [{"divisor": "  2.5 "}],
    [{"mode": "RESPACE"}], [{"mode": " Respace "}], [{"mode": "foo"}], [{"mode": None}], [{"mode": 0}],
    [{"threshold": "Delta_Optimal"}], [{"threshold": "bad"}], [{"transform": "DWT"}], [{"transform": "wavelet"}],
    [{"preset": "custom"}], [{"preset": "Custom"}], [{"preset": ""}], [{"preset": None}], [{"preset": 0}],
    [{"preset": " flux "}], [{"preset": "flux2"}], [{"preset": "z-image"}],
    [{"seed": -5}], [{"seed": 3.7}], [{"seed": ""}], [{"seed": "12"}], [{"seed": "1e3"}], [{"seed": True}],
    [{"seed": "-0.5"}], [{"seed": _BIG}], [{"seed": "inf"}], [{"seed": 0}], [{"seed": 2 ** 63}],
    [{"preset": "custom", "A": "nan", "beta": "inf"}], [{"preset": "custom", "A": "-1"}],
    [{"preset": "custom", "A": 0}], [{"preset": "custom", "A": "1e-12", "beta": -3}], [{"A": True, "beta": None}],
    [{"adaptive": "False"}], [{"adaptive": "0"}], [{"adaptive": 0}], [{"adaptive": None}], [{"adaptive": "maybe"}],
    [{"hires": "True"}], [{"hires": "1"}], [{"hires": None}], [{"hires": "disable"}],
    ["True", {"mode": "respace"}],
    [{"mode": "x", "threshold": "y", "preset": "z", "transform": "w", "scales": "", "delta": 0,
      "sigma_divisor": 0}],
    [False, "x", "manual", "nope", "2,1", "nan", 0, "0.5,0.7", "?", "dwt", "a", "b", "c", "d"],
)


@requires_torch
@requires_extension
class CoerceParityTests(unittest.TestCase):
    """``sp.coerce`` == 설치된 ``forge_host.coerce_settings`` — 모든 필드(타입·float 비트)·``error`` 문구·요약,
    예외면 같은 타입·문구."""

    @classmethod
    def setUpClass(cls):
        load_torch()
        cls.fh = _ext_speed("forge_host")

    def test_table(self):
        self.assertGreaterEqual(len(COERCE_ROWS), 60)
        raised = zero = 0
        for row in COERCE_ROWS:
            with self.subTest(row=row):
                want = _outcome(self.fh.coerce_settings, copy.deepcopy(row))
                got = _outcome(sp.coerce, copy.deepcopy(row))
                self.assertEqual(got, want)
                raised += want[0] == "raise"
                if want[0] == "ok":
                    zero += "division by zero" in (self.fh.coerce_settings(copy.deepcopy(row)).error or "")
        self.assertEqual(raised, 0)          # sam-extra 0.33.0: a/0·아주 큰 정수도 예외 없이(오류 목록·기본값)
        self.assertGreaterEqual(zero, 5)     # a/0 행이 정말 'division by zero' 설정 오류가 된다

    def test_extension_settings_carry_no_xyz_here(self):
        self.assertEqual(self.fh.coerce_settings([True]).xyz, ())

    def test_parse_summary(self):
        texts = [self.fh.coerce_settings(copy.deepcopy(row)).summary() for row in COERCE_ROWS
                 if _outcome(self.fh.coerce_settings, copy.deepcopy(row))[0] == "ok"]
        texts += ["", None, 5, "a=1;;b = 2 ; c", "x=y=z; =v; k=", " mode = respace ;threshold=manual"]
        for text in texts:
            with self.subTest(text=text):
                self.assertEqual(sp.parse_summary(text), self.fh.parse_summary(text))


@requires_torch
@requires_extension
class BlockParityTests(unittest.TestCase):
    """앱이 보내는 JSON 블록을 확장이 읽은 값 == ``sp.script_settings``(카드에서 바로 계산) — 카드가 보여 준 그
    설정이 Forge 에 간다. 위젯 문자열 → 카드 → 블록 경로도 같다."""

    CARDS = (
        card(), card(enabled=True), card(enabled=True, hires=True, seed=42),
        card(enabled=True, mode="respace", threshold="manual", manual_sigmas="0.6,0.8", scales="0.25,0.5,1"),
        card(enabled=True, threshold="delta_optimal", sigma_divisor=-1.0, adaptive=False),
        card(enabled=True, preset="custom", spectrum_A=-1.0, spectrum_beta=0.5, transform="dwt"),
        card(enabled=True, preset="custom", spectrum_A=1e-12, spectrum_beta=7.25, transform="fft"),
        card(enabled=True, scales="1/2;1", delta=0.0001, sigma_divisor=2.0),
        card(enabled=True, scales="", manual_sigmas="", delta=1.0),
        card(enabled=True, seed=2 ** 40, delta=0.1 + 0.2, sigma_divisor=1 / 3),
        card(enabled=True, apply_img2img=True, preset="wan21"),
    )

    @classmethod
    def setUpClass(cls):
        load_torch()
        cls.fh = _ext_speed("forge_host")

    def test_json_block_reads_back_as_the_card(self):
        for settings in self.CARDS:
            with self.subTest(settings=settings):
                sent = json.loads(json.dumps(sp.as_block(settings)))
                self.assertEqual(_outcome(self.fh.coerce_settings, sent["args"]),
                                 _outcome(sp.script_settings, settings))
                widgets = sp.widget_values(settings)
                self.assertEqual(sp.as_block(sp.parse_settings(widgets)), sp.as_block(settings))

    def test_valid_cards_reach_forge_unchanged(self):
        checked = 0
        for settings in self.CARDS:
            script = self.fh.coerce_settings(sp.as_block(settings)["args"])
            if script.error:
                continue
            checked += 1
            with self.subTest(settings=settings):
                for name in ("enabled", "mode", "threshold", "preset", "delta", "sigma_divisor", "adaptive",
                             "transform", "spectrum_A", "spectrum_beta", "seed", "hires"):
                    self.assertEqual(_canon(getattr(script, name)), _canon(getattr(settings, name)), name)
        self.assertGreaterEqual(checked, 7)


@requires_torch
@requires_extension
class ValidateParityTests(unittest.TestCase):
    """``validate`` == 확장이 T2I(σ 1 시작)에서 내리는 판정: coerce_settings 의 오류(``a/0`` 포함 — 0.33.0) →
    ``invalid settings``, build_plan 의 ``PlanError(kind="invalid")`` → ``invalid settings``. 건너뜀
    (``kind="skip"``)과 정상 계획은 오류 없음. 여러 latent 크기·스텝·shift 에서 같아야 한다(설정만 보는 검사)."""

    EXTRA = (
        card(preset="custom", spectrum_A=-1e-300), card(preset="custom", spectrum_A=5.0, spectrum_beta=-0.5),
        card(threshold="delta_optimal", sigma_divisor=0.0), card(threshold="neo_shift", sigma_divisor=1e-9),
        card(transform="dwt", scales="0.125,0.25,0.5,1"), card(transform="dwt", scales="0.5000001,1"),
        card(mode="respace", threshold="manual", manual_sigmas="0.9,0.4,0.2", scales="0.25,0.5,1"),
        card(mode="respace", threshold="manual", manual_sigmas="0.9,0.4", scales="0.25,0.5,1"),
        card(threshold="manual", manual_sigmas="0.3", scales="0.5,1"), card(threshold="manual", manual_sigmas="0.05"),
        card(delta=0.4999), card(preset="krea-2", adaptive=False, delta=0.3), card(scales="0.75,1"),
    )
    # 확장이 실행 중에야 아는 'invalid'(스펙트럼 계산 실패·격자) — validate 는 미리 모른다(Forge status 가 말한다)
    RUNTIME_ONLY = (card(preset="custom", spectrum_beta=1e6),)
    GRIDS = ((128, 128), (104, 152), (64, 64))
    SCHEDULES = ((28, 3.0), (8, 1.0), (50, 5.0))

    @classmethod
    def setUpClass(cls):
        cls.torch = load_torch()
        cls.fh = _ext_speed("forge_host")
        cls.schedule = _ext_speed("schedule")

    def _sigmas(self, steps: int, shift: float):
        values = [shift * t / (1.0 + (shift - 1.0) * t) for t in (1.0 - i / steps for i in range(steps))]
        return self.torch.tensor(values + [0.0], dtype=self.torch.float32)

    def _forge_verdicts(self, settings) -> set:
        script = self.fh.coerce_settings(sp.as_block(settings)["args"])     # 0.33.0: 예외 없음
        if script.error:
            return {"invalid"}
        verdicts = set()
        for h, w in self.GRIDS:
            for steps, shift in self.SCHEDULES:
                x = self.torch.zeros(1, 16, h, w)
                try:
                    self.fh._plan_for(script, x, self._sigmas(steps, shift))
                except self.schedule.PlanError as exc:
                    verdicts.add("invalid" if exc.kind == "invalid" else "ok")
                else:
                    verdicts.add("ok")
        return verdicts

    def test_validate_matches_the_extension_on_t2i(self):
        rows = [settings for settings, _codes in ValidateTests.CASES] + list(self.EXTRA)
        seen = set()
        for settings in rows:
            with self.subTest(settings=settings):
                verdicts = self._forge_verdicts(settings)
                self.assertEqual(len(verdicts), 1, f"extension verdict depends on the grid/schedule: {verdicts}")
                codes = sp.validate(settings)
                want = verdicts.pop()
                seen.add(want)
                if want == "invalid":
                    self.assertTrue(codes, codes)
                else:
                    self.assertEqual(codes, ())
        self.assertEqual(seen, {"invalid", "ok"})

    def test_runtime_only_invalid_settings_are_documented(self):
        for settings in self.RUNTIME_ONLY:
            with self.subTest(settings=settings):
                self.assertEqual(self._forge_verdicts(settings), {"invalid"})
                self.assertEqual(sp.validate(settings), ())

    def test_spectrum_a_reason_follows_the_extension(self):
        script = self.fh.coerce_settings(sp.as_block(card(preset="custom", spectrum_A=-1.0))["args"])
        with self.assertRaises(self.schedule.PlanError) as caught:
            self.fh._plan_for(script, self.torch.zeros(1, 16, 128, 128), self._sigmas(28, 3.0))
        self.assertEqual((caught.exception.kind, caught.exception.reason),
                         ("invalid", "spectrum A must be a positive number; got -1.0"))


def _ext_function(relative: str, name: str, namespace: dict):
    """확장 소스의 함수 정의 하나를 떼어 ``namespace`` 에서 실행한다(Forge·gradio import 없음)."""
    node = next(item for item in _ext_tree(relative).body if isinstance(item, ast.FunctionDef) and item.name == name)
    exec(compile(ast.Module(body=[node], type_ignores=[]), f"{relative}:{name}", "exec"), namespace)  # noqa: S102
    return namespace[name]


@requires_extension
class PasteParityTests(unittest.TestCase):
    """``sp.paste_values`` == 확장 infotext_fields 를 Forge 붙여 넣기 루프로 돌린 값(``None`` = 칸 그대로).

    ``_paste``·``_paste_bool``·``_paste_int``(scripts/anima_speed.py)와 ``parse_summary``(forge_host.py)는 AST 로 떼어
    실행한다 — gradio·Forge 없이. 표의 모양(칸 ↔ 요약 키 ↔ 변환)은 ``ExtensionConstantTests`` 가 지킨다."""

    @classmethod
    def setUpClass(cls):
        parse_summary = _ext_function(FORGE_HOST, "parse_summary", {})
        script_ns = {"KEY": _ext_literal(FORGE_HOST, "KEY"), "fh": SimpleNamespace(parse_summary=parse_summary)}
        paste = _ext_function(SCRIPT, "_paste", script_ns)
        converters = {"str": str, "float": float,
                      "_paste_bool": _ext_function(SCRIPT, "_paste_bool", script_ns),
                      "_paste_int": _ext_function(SCRIPT, "_paste_int", script_ns)}
        key = script_ns["KEY"]
        cls.readers = [("enabled", lambda params: key in params)]
        cls.readers += [(field, paste(name, converters[convert.__name__])) for field, name, convert in sp.PASTE_FIELDS]

    def _forge(self, params) -> dict:
        out = {}
        for field, reader in self.readers:
            try:
                value = reader(params)
            except Exception:   # modules/infotext_utils.py connect_paste: errors.report → None
                value = None
            out[field] = value
        return out

    def test_table(self):
        texts = [
            "mode=respace; threshold=manual; preset=custom; scales=0.25,0.5,1; delta=0.02; divisor=1.1; "
            "manual=0.8,0.6; adaptive=False; transform=fft; A=12; beta=2; seed=77; hires=True",
            sp.ScriptSettings().summary(), "", "garbage", "seed=inf", "seed=1e3", "seed=-2.7", "seed=nan", "seed=",
            "delta=", "delta=nan", "A=nan; beta=-inf", "adaptive=YES; hires= on ", "adaptive=1; hires=0",
            "mode=Respace; threshold=; preset=Flux; transform=DWT", "divisor=1e400; manual=1/0; scales=x",
            "unknown=1; mode=transition", " mode = respace ;; =x; seed = 12 ",
        ]
        params_list = [{"Anima SPEED": text} for text in texts]
        params_list += [{}, {"Anima SPEED status": "base: applied"}, {"Anima SPEED": None}, {"Anima SPEED": 5},
                        {"Anima SPEED": "seed=4", "Steps": "28"}]
        for params in params_list:
            with self.subTest(params=params):
                got, want = sp.paste_values(params), self._forge(params)
                self.assertEqual({k: _canon(v) for k, v in got.items()}, {k: _canon(v) for k, v in want.items()})


class StatusParseTests(unittest.TestCase):
    """'Anima SPEED status' → 패스마다(forge_host._write_status 의 역, 계획 A-SP3 결과 알림이 쓴다)."""

    def test_passes_kinds_and_details(self):
        text = ("base: applied (transition) - 5/28 steps coarse, 52x76 -> 104x152, step 5 sigma 0.7208->0.7208 | "
                "hires: not applied - Apply to Hires pass is off")
        base, hires = sp.parse_status(text)
        self.assertEqual((base.label, base.kind), (sp.PASS_BASE, sp.STATUS_APPLIED))
        self.assertEqual(base.detail, "5/28 steps coarse, 52x76 -> 104x152, step 5 sigma 0.7208->0.7208")
        self.assertEqual((hires.label, hires.kind, hires.detail),
                         (sp.PASS_HIRES, sp.STATUS_NOT_APPLIED, "Apply to Hires pass is off"))
        cases = {
            "img2img: invalid settings - delta must be in (0, 1); got 0.0":
                (sp.PASS_IMG2IMG, sp.STATUS_INVALID, "delta must be in (0, 1); got 0.0"),
            "base: skipped - no coarse steps: the pass starts at sigma 0.6000, at or below the first transition sigma 0.7":
                (sp.PASS_BASE, sp.STATUS_SKIPPED,
                 "no coarse steps: the pass starts at sigma 0.6000, at or below the first transition sigma 0.7"),
            "base: pending - sampler wrapped, waiting for sampling":
                (sp.PASS_BASE, sp.STATUS_PENDING, "sampler wrapped, waiting for sampling"),
            "hires: interrupted": (sp.PASS_HIRES, sp.STATUS_INTERRUPTED, ""),
            "base: applied (respace) - 3/20 steps coarse": (sp.PASS_BASE, sp.STATUS_APPLIED, "3/20 steps coarse"),
            "base: something new": (sp.PASS_BASE, "", "something new"),
        }
        for text, expected in cases.items():
            with self.subTest(text=text):
                [status] = sp.parse_status(text)
                self.assertEqual((status.label, status.kind, status.detail), expected)
                self.assertEqual(status.text, text.split(": ", 1)[1])

    def test_exception_text_with_separators_stays_in_its_pass(self):
        text = ('"base: fell back to plain sampling after RuntimeError: shape [2, 16] | a: b, c | '
                'hires: skipped - ControlNet is on (its hint is tied to the full latent grid)"')    # JSON 따옴표째
        base, hires = sp.parse_status(text)
        self.assertEqual((base.kind, base.detail), (sp.STATUS_FALLBACK, "RuntimeError: shape [2, 16] | a: b, c"))
        self.assertEqual((hires.label, hires.kind), (sp.PASS_HIRES, sp.STATUS_SKIPPED))

    def test_empty_or_unreadable(self):
        for value in (None, "", "   ", "no label here", "upscale: applied (x) - y", 12):
            with self.subTest(value=value):
                self.assertEqual(sp.parse_status(value), ())
        self.assertEqual(set(sp.STATUS_KINDS), {sp.STATUS_APPLIED, sp.STATUS_INVALID, sp.STATUS_SKIPPED,
                                                sp.STATUS_NOT_APPLIED, sp.STATUS_PENDING, sp.STATUS_FALLBACK,
                                                sp.STATUS_INTERRUPTED})


if __name__ == "__main__":
    unittest.main()
