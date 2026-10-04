"""Extra Samplers(sam-extra 'Extra Samplers', v0.33.0) — core/extra_samplers 순수 로직.

설치된 확장(``tests/_sam_extra_ext.EXT_ROOT``)이 기준이다 — 같은 입력을 양쪽에 넣어 같은 결과를 요구한다. v0.33.0 이
설치되기 전에는 그 작업 트리를 ``AISTUDIO_FORGE_EXTENSION_DIR`` 로 가리킨다.

* ``ExtensionConstantTests`` — 확장 소스 상수(AST, import 없음) = 앱 상수: params(범위·기본값·infotext 키·ER SDE 표기),
  registry(표기 15개), substep_guard(상태 키·이유), 스크립트(제목·XYZ), Restart·UD10·IPNDM/DEIS 상수.
* ``SamplerTableTests`` — registry.SPECS(생성자 호출 — AST 로 풀어 읽는다)의 순서·별칭·kind·cfg_pp = 앱 표,
  params 의 η/구간 표기 묶음, 스크립트 ui() 컴포넌트(라벨·기본값·범위·step) = ``ARG_SPEC``.
* ``CoercionParityTests``·``PasteParityTests``·``InfotextParityTests`` — params.py 를 파일 하나로 올려(패키지
  ``__init__`` 은 registry → torch 를 올린다) coerce_*·settings_from_args(+ process 의 예외 → 기본값)·paste_*·
  parse/format_noise_window·record_infotext 와 비교. ``StatusParityTests``(@requires_torch) — substep_guard 가 쓴 상태 줄.
* ``BlockTests``·``PlanTests``·``NameTests``·``PassTests``·``DescribeTests`` — 확장 없이: 보내는 블록(늘 인자 다섯, X10),
  위젯 왕복, 싣는 규칙(X5)·옛 확장 경고(X10), 이름 접기, 값을 읽는 패스.

확장이 없으면 확장 비교만 skip, AISTUDIO_REQUIRE_FORGE_EXT=1 이면 실패. torch 는 지연 import(tests/_optional_deps).
"""
from __future__ import annotations

import ast
import importlib.util
import math
import os
import sys
import unittest
from dataclasses import replace
from types import SimpleNamespace

from core import extra_samplers as xsa
from core import sam3_args
from core.extra_schedulers import PASS_HIRES, PASS_MAIN, PASS_SAM3
from core.sam_extra_capabilities import SamExtraCapabilities, _freeze
from tests._optional_deps import requires_torch
from tests._sam_extra_ext import EXT_ROOT, requires_extension
from tests.test_alwayson_propagation import caps

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PARAMS = "sam3ext/extra_samplers/params.py"
REGISTRY = "sam3ext/extra_samplers/registry.py"
GUARD = "sam3ext/extra_samplers/substep_guard.py"
SCRIPT = "scripts/anima_extra_samplers.py"
_LOADED: dict = {}


def _load_ext_file(relative: str):
    """확장 파일 하나를 단독 모듈로 올린다(패키지 __init__ 을 거치지 않는다 — registry·torch 를 올리지 않게)."""
    if relative not in _LOADED:
        name = "_xsa_oracle_" + relative.replace("/", "_").replace(".", "_")
        spec = importlib.util.spec_from_file_location(name, EXT_ROOT / relative)
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module   # dataclass 가 모듈을 찾는다
        spec.loader.exec_module(module)
        _LOADED[relative] = module
    return _LOADED[relative]


def _tree(relative: str) -> ast.Module:
    return ast.parse((EXT_ROOT / relative).read_text(encoding="utf-8"))


def _ext_constant(relative: str, name: str):
    """확장 파일의 모듈 상수(리터럴, AST — import 없음)."""
    for item in _tree(relative).body:
        targets = item.targets if isinstance(item, ast.Assign) else (
            [item.target] if isinstance(item, ast.AnnAssign) else [])
        if any(isinstance(target, ast.Name) and target.id == name for target in targets):
            return ast.literal_eval(item.value)
    raise AssertionError(f"{relative} 에 모듈 상수 {name} 이 없다")


def _registry_value(node: ast.expr):
    """registry.py 의 식 하나 — 리터럴, 같은 파일의 상수 이름(LABEL_*), params 상수(sampler_params.LABEL_*)."""
    if isinstance(node, ast.Name):
        for item in _tree(REGISTRY).body:
            if isinstance(item, ast.Assign) and any(getattr(t, "id", None) == node.id for t in item.targets):
                return _registry_value(item.value)
        raise AssertionError(f"registry.py 에 {node.id} 이 없다")
    if isinstance(node, ast.Attribute) and getattr(node.value, "id", None) == "sampler_params":
        return _ext_constant(PARAMS, node.attr)
    return ast.literal_eval(node)


def _params_label_tuple(name: str) -> tuple:
    """params.py 의 표기 묶음(LABEL_* 이름의 튜플)."""
    for item in _tree(PARAMS).body:
        if isinstance(item, ast.Assign) and any(getattr(t, "id", None) == name for t in item.targets):
            return tuple(_ext_constant(PARAMS, element.id) for element in item.value.elts)
    raise AssertionError(f"params.py 에 {name} 이 없다")


def _specs() -> list:
    """registry.SPECS 의 SamplerSpec 호출 → (label, aliases, kind, cfg_pp) 목록(순서 그대로)."""
    out = []
    for node in ast.walk(_tree(REGISTRY)):
        if isinstance(node, ast.Call) and getattr(node.func, "id", None) == "SamplerSpec":
            keywords = {kw.arg: kw.value for kw in node.keywords}
            out.append((
                _registry_value(node.args[0]),
                _registry_value(node.args[2]),
                _registry_value(keywords["kind"]) if "kind" in keywords else "",
                _registry_value(keywords["cfg_pp"]) if "cfg_pp" in keywords else False,
            ))
    return out


def _key(value):
    """float 는 repr 로 — -0.0 과 0.0, 마지막 비트까지 다르게 본다(bool 과 int 도 다르게)."""
    if isinstance(value, float):
        return ("float", repr(value))
    if isinstance(value, (list, tuple)):
        return (type(value).__name__, tuple(_key(item) for item in value))
    return (type(value).__name__, value)


def _fields(settings) -> tuple:
    return tuple(_key(getattr(settings, name)) for name in xsa.ARG_NAMES)


def _outcome(function):
    try:
        return "ok", function()
    except Exception as exc:
        return "error", type(exc).__name__


_MISSING = object()
_NUMBER_VALUES = (
    None, True, False, 0, 1, 2, 3, 4, -1, 0.0, -0.0, 0.5, 1.4999, 1.5, 2.5, 3.5, 0.01, 0.2, 0.8, 1.0, 1.01, -0.2,
    9.99, 10, 10.0, 10.5, -10.5, 1e9, -1e9, 1e308, -1e308, 1e-320, float("nan"), float("inf"), float("-inf"),
    "nan", "inf", "-inf", "1e9", " 2 ", "2.5", "3.25", "-0", "0.3", "abc", "", [], {}, (1,), "true", "1", "0",
)
_WINDOW_VALUES = (None, True, False, 0, 1, 2, -1, 0.0, 1.0, 1.5, "true", "True", " TRUE ", "on", "On", "yes",
                  "1", " 1 ", "1.0", "0", "false", "off", "no", "", "x", [], [1], {}, float("nan"))
_ARGS_CORPUS = (
    None, [], (), [3], [2], [2, 0.5], [3, 1.0], [1, 2.5, True, 0.1, 0.9], [3, 1.0, False, 0.2, 0.8],
    [2, 0.37, "on", "0.25", " 0.75 "], [5, -1, 1, 2, -2], [3, 1.0, True, 0.8, 0.2, "extra", 9], ["2", "0.5"],
    [float("nan"), float("inf"), None, None, None], [True, True, True, True, True], [None] * 5, "21", "x",
    {"max_stage": 2}, (2, 0.5, 1), [10 ** 400], [2, 10 ** 400], [2, 0.5, True, 10 ** 400], 5, 2.5, True,
    [2, 0.5, "yes", float("nan"), float("-inf")], [[2], 0.5],
)


class _P:
    """Forge 요청 대역 — record_infotext·record_status 가 읽고 쓰는 칸만."""

    def __init__(self, params=None):
        self.extra_generation_params = dict(params or {})


# ── 확장 상수 ──────────────────────────────────────────────────────────────────
@requires_extension
class ExtensionConstantTests(unittest.TestCase):
    """확장 소스 상수(AST) = 앱 상수(레지스트리 SEMANTIC_PINS 도 같은 값을 지키지만 여기서는 모듈 전체를 본다)."""

    def test_params_constants(self):
        for name in ("ARG_NAMES", "DEFAULT_MAX_STAGE", "MAX_STAGE_MIN", "MAX_STAGE_MAX", "DEFAULT_ETA", "ETA_MIN",
                     "ETA_MAX", "ETA_STEP", "DEFAULT_NOISE_WINDOW", "DEFAULT_NOISE_START", "DEFAULT_NOISE_END",
                     "NOISE_PERCENT_MIN", "NOISE_PERCENT_MAX", "NOISE_PERCENT_STEP", "KEY_MAX_STAGE", "KEY_ETA",
                     "KEY_NOISE_WINDOW", "NOISE_WINDOW_OFF", "WINDOW_SKIPPED_STATUS", "LABEL_ER_SDE_REVERSE_TIME",
                     "LABEL_ER_SDE_ODE", "LABEL_ER_SDE_TUNABLE"):
            with self.subTest(name=name):
                self.assertEqual(_key(getattr(xsa, name)), _key(_ext_constant(PARAMS, name)))
        self.assertEqual(xsa.KEY_FORGE_ETA, _ext_constant(PARAMS, "_FORGE_ETA_KEY"))
        self.assertEqual(xsa.ETA_LABELS, _params_label_tuple("_ETA_LABELS"))
        self.assertEqual(xsa.ER_SDE_LABELS, _params_label_tuple("_ER_SDE_LABELS"))
        self.assertEqual(xsa.WINDOW_LABELS, xsa.ETA_LABELS)

    def test_registry_labels(self):
        names = [n for n in dir(xsa) if n.startswith("LABEL_") and n != "LABEL_OF_COMFY"]
        self.assertEqual(len(names), 18)
        for name in names:
            with self.subTest(name=name):
                source = PARAMS if name.startswith("LABEL_ER_SDE_") else REGISTRY
                self.assertEqual(getattr(xsa, name), _ext_constant(source, name))

    def test_status_script_and_sampler_constants(self):
        self.assertEqual(xsa.STATUS_KEY, _ext_constant(GUARD, "STATUS_KEY"))
        self.assertEqual(xsa.DY_SKIPPED_PREFIX, _ext_constant(GUARD, "_STATUS_PREFIX"))
        self.assertEqual(xsa.DY_REASONS, tuple(_ext_constant(GUARD, f"REASON_{n}")
                                                 for n in ("SPECTRUM", "WAN", "PID", "EXTRA_CONCAT")))
        self.assertEqual(xsa.SCRIPT_NAME, _ext_constant(SCRIPT, "TITLE"))
        self.assertEqual(xsa.XYZ_PREFIX, _ext_constant(SCRIPT, "XYZ_PREFIX"))
        for relative, names in (
                ("sam3ext/extra_samplers/restart_flow.py", ("RESTART_S_MIN", "RESTART_S_MAX", "RESTART_MIN_STEPS",
                                                            "RESTART_TIMES", "RESTART_TIMES_LONG", "RESTART_LONG_FROM")),
                ("sam3ext/extra_samplers/cfgpp_ud10_ab.py", ("UD10_HISTORY_WEIGHT", "UD10_ZERO_WEIGHT",
                                                             "UD10_UNCOND_HISTORY_WEIGHT")),
                ("sam3ext/extra_samplers/ipndm_deis.py", ("IPNDM_MAX_ORDER", "DEIS_MAX_ORDER", "DEIS_MODE"))):
            for name in names:
                with self.subTest(name=name):
                    self.assertEqual(_key(getattr(xsa, name)), _key(_ext_constant(relative, name)))

    def test_xyz_axes_are_the_extension_axes(self):
        from core.sam_extra_scan import ExtensionSource

        axes = {label for label in ExtensionSource(EXT_ROOT).axis_labels() if label.startswith(xsa.XYZ_PREFIX)}
        self.assertEqual(axes, set(xsa.XYZ_AXES))


@requires_extension
class SamplerTableTests(unittest.TestCase):
    """registry.SPECS = 앱 표(순서·별칭·묶음), 스크립트 ui() = ARG_SPEC."""

    @classmethod
    def setUpClass(cls):
        cls.specs = _specs()

    def test_order_and_aliases(self):
        self.assertEqual(len(self.specs), 18)
        self.assertEqual(tuple(label for label, *_ in self.specs), xsa.LABELS)
        self.assertEqual({label: tuple(aliases) for label, aliases, *_ in self.specs}, dict(xsa.ALIASES))
        self.assertEqual(len(set(xsa.COMFY_NAMES.values())), 18)
        self.assertEqual(set(xsa.PACK_SAMPLERS) | set(xsa.CORE_SAMPLERS), set(xsa.COMFY_NAMES.values()))
        self.assertEqual((len(xsa.PACK_SAMPLERS), len(xsa.CORE_SAMPLERS)), (12, 6))
        self.assertFalse(set(xsa.PACK_SAMPLERS) & set(xsa.CORE_SAMPLERS))
        self.assertEqual({xsa.LABEL_OF_COMFY[name] for name in xsa.COMFY_NAMES.values()}, set(xsa.LABELS))

    def test_groups_follow_kind_and_cfg_pp(self):
        kinds = {label: kind for label, _aliases, kind, _cfg in self.specs}
        self.assertEqual(xsa.ER_SDE_LABELS, tuple(label for label, kind in kinds.items() if kind == "er_sde"))
        self.assertEqual(set(xsa.DY_KIND), {label for label, kind in kinds.items() if kind == "dy"})
        self.assertEqual({label for label, *_rest, cfg in self.specs if cfg}, set(xsa.CFG_PP_LABELS))
        self.assertEqual({label for label, kind in xsa.DY_KIND.items() if kind == "smea"},
                         {label for label in xsa.DY_KIND if "SMEA" in label})
        self.assertTrue(set(kinds.values()) <= {"er_sde", "dy", ""})

    def test_script_ui_is_the_arg_spec(self):
        from core.sam_extra_scan import ExtensionSource

        info = ExtensionSource(EXT_ROOT).scripts()[xsa.SCRIPT_NAME]
        self.assertEqual(info.ui_return, xsa.ARG_NAMES)
        self.assertEqual(tuple(key[len(xsa.SPEC_PREFIX):] for key, *_ in xsa.ARG_SPEC), xsa.ARG_NAMES)
        steps = {"max_stage": xsa.MAX_STAGE_STEP, "eta": xsa.ETA_STEP, "noise_start": xsa.NOISE_PERCENT_STEP,
                 "noise_end": xsa.NOISE_PERCENT_STEP}
        for (key, kind, default, extent), component in zip(xsa.ARG_SPEC, info.ui_components):
            name = key[len(xsa.SPEC_PREFIX):]
            with self.subTest(name=name):
                self.assertEqual(component["label"], f"ER SDE {name.replace('_', ' ')}")
                self.assertEqual(_key(component["value"]), _key(default))
                self.assertEqual(kind, {bool: "bool", int: "int", float: "float"}[type(default)])
                if extent is None:
                    self.assertNotIn("minimum", component)
                else:
                    self.assertEqual((_key(component["minimum"]), _key(component["maximum"]), component["step"]),
                                     (_key(extent[0]), _key(extent[1]), steps[name]))


# ── 정규화·붙여 넣기·기록 ─────────────────────────────────────────────────────────
@requires_extension
class CoercionParityTests(unittest.TestCase):
    """coerce_* = 확장 params.coerce_*(값마다, float 는 비트까지), ``coerce_args`` = 스크립트 process."""

    @classmethod
    def setUpClass(cls):
        cls.ext = _load_ext_file(PARAMS)

    def test_single_value_coercers(self):
        for value in _NUMBER_VALUES:
            with self.subTest(value=value):
                self.assertEqual(_key(xsa.coerce_max_stage(value)), _key(self.ext.coerce_max_stage(value)))
                self.assertEqual(_key(xsa.coerce_max_stage(value, 2)), _key(self.ext.coerce_max_stage(value, 2)))
                self.assertEqual(_key(xsa.coerce_eta(value)), _key(self.ext.coerce_eta(value)))
                self.assertEqual(_key(xsa.coerce_eta(value, 0.5)), _key(self.ext.coerce_eta(value, 0.5)))
                for default in (xsa.DEFAULT_NOISE_START, xsa.DEFAULT_NOISE_END):
                    self.assertEqual(_key(xsa.coerce_noise_percent(value, default)),
                                     _key(self.ext.coerce_noise_percent(value, default)))
                self.assertEqual(_key(xsa.coerce_noise_start(value)),
                                 _key(self.ext.coerce_noise_percent(value, self.ext.DEFAULT_NOISE_START)))
                self.assertEqual(_key(xsa.coerce_noise_end(value)),
                                 _key(self.ext.coerce_noise_percent(value, self.ext.DEFAULT_NOISE_END)))
        for value in (*_WINDOW_VALUES, *_NUMBER_VALUES):
            with self.subTest(window=value):
                self.assertEqual(_key(xsa.coerce_noise_window(value)), _key(self.ext.coerce_noise_window(value)))

    def test_huge_int_is_the_one_deliberate_difference(self):
        """확장 coerce_* 는 float 로 못 바꾸는 큰 int 에서 OverflowError 를 낸다 — 앱은 기본값(저장값 하나로 기여자가
        실패하지 않게). 블록 해석(coerce_args)은 확장 process 처럼 전부 기본값이라 같다(test_coerce_args_is_process)."""
        huge = 10 ** 400
        for ext_call, app_value in ((lambda: self.ext.coerce_max_stage(huge), xsa.coerce_max_stage(huge)),
                                    (lambda: self.ext.coerce_eta(huge), xsa.coerce_eta(huge)),
                                    (lambda: self.ext.coerce_noise_percent(huge, 0.2), xsa.coerce_noise_start(huge))):
            with self.assertRaises(OverflowError):
                ext_call()
            self.assertIn(app_value, (xsa.DEFAULT_MAX_STAGE, xsa.DEFAULT_ETA, xsa.DEFAULT_NOISE_START))

    def _process(self, args):
        """스크립트 process: settings_from_args, 예외면 ErSdeSettings()(XYZ 없음)."""
        try:
            return self.ext.settings_from_args(args)
        except Exception:
            return self.ext.ErSdeSettings()

    def test_coerce_args_is_process(self):
        for args in _ARGS_CORPUS:
            with self.subTest(args=repr(args)[:60]):
                self.assertEqual(_fields(xsa.coerce_args(args)), _fields(self._process(args)))
                self.assertEqual(_fields(xsa.parse_script_block({"args": args})), _fields(self._process(args)))

    def test_parse_settings_with_default_base_is_the_extension_coercion(self):
        corpus = [{}, {"unknown": 1}]
        for name in xsa.ARG_NAMES:
            for value in (*_NUMBER_VALUES, *_WINDOW_VALUES):
                corpus.append({name: value})
        corpus.append({"max_stage": "2", "eta": "0.5", "noise_window": "1", "noise_start": "0.1", "noise_end": "0.9"})
        for raw in corpus:
            with self.subTest(raw=repr(raw)[:60]):
                want = self.ext.ErSdeSettings(
                    max_stage=self.ext.coerce_max_stage(raw.get("max_stage", 3)),
                    eta=self.ext.coerce_eta(raw.get("eta", 1.0)),
                    noise_window=self.ext.coerce_noise_window(raw.get("noise_window", False)),
                    noise_start=self.ext.coerce_noise_percent(raw.get("noise_start", 0.2), 0.2),
                    noise_end=self.ext.coerce_noise_percent(raw.get("noise_end", 0.8), 0.8),
                )
                self.assertEqual(_fields(xsa.parse_settings(raw)), _fields(want))

    def test_defaults_and_settings_fields(self):
        self.assertEqual(_fields(xsa.EXTENSION_DEFAULTS), _fields(self.ext.ErSdeSettings()))
        self.assertEqual(tuple(self.ext.ErSdeSettings().as_dict()), xsa.ARG_NAMES)
        self.assertIs(xsa.APP_DEFAULTS, xsa.EXTENSION_DEFAULTS)


_WINDOW_TEXTS = (
    "0.2-0.8", "off", "OFF", " Off ", "0.1-0.9", " 0.1 - 0.9 ", "0-1", "1-0", "0.5-0.5", "0.8-0.2", ".5-.75",
    "1.-0.5", "1e-06-0.5", "1E-1-2e-1", "0.2 -0.8", "0.2- 0.8", "0.2--0.8", "-0.2-0.8", "0.2-1.5", "1.5-0.2",
    "0.2", "0.2-", "-", "", "a-b", "0.2-0.8-0.9", "0,2-0,8", "nan-0.5", "inf-0.5", "0x1-0.5", "２-0.5", None, 0.5,
    "0.2\t-\n0.8", "0.20000000001-0.8",
)
_PASTE_CASES = (
    {},
    {"ER SDE max stage": 2},
    {"ER SDE max stage": "1"},
    {"ER SDE max stage": "x"},
    {"ER SDE max stage": 7.6},
    {"ER SDE eta": 0.5},
    {"ER SDE eta": "2.25"},
    {"ER SDE eta": "nan"},
    {"ER SDE eta": None},
    {"Sampler": "ER SDE (Reverse-time)", "Eta": 0.7},
    {"Sampler": "ER SDE (Reverse-time) Karras", "Eta": "0.3"},
    {"Hires sampler": "ER SDE (Reverse-time)", "Eta": 12},
    {"Sampler": "ER SDE (Tunable)", "Eta": 0.7},
    {"Sampler": "ER SDE (Reverse-time)", "Eta": 0.7, "ER SDE eta": 2.0},
    {"Sampler": "er_sde_reverse_time", "Eta": 0.7},
    {"Sampler": " ER SDE (Reverse-time) ", "Eta": 0.7},
    {"Sampler": "ER SDE (Reverse-timex)", "Eta": 0.7},
    {"Sampler": "ER SDE (Reverse-time)Karras", "Eta": 0.7},
    {"Sampler": "ER SDE (Reverse-time)"},
    *({"ER SDE noise window": text} for text in _WINDOW_TEXTS),
    {"ER SDE noise window": "0.1-0.9", "ER SDE max stage": 1, "ER SDE eta": 0.4, "Sampler": "ER SDE (Tunable)"},
)


@requires_extension
class PasteParityTests(unittest.TestCase):
    """``from_infotext`` = paste_* 다섯, parse/format_noise_window = params 의 같은 함수."""

    @classmethod
    def setUpClass(cls):
        cls.ext = _load_ext_file(PARAMS)

    def test_window_texts(self):
        for text in _WINDOW_TEXTS:
            with self.subTest(text=text):
                self.assertEqual(_key(xsa.parse_noise_window(text)), _key(self.ext.parse_noise_window(text)))
        for start, end in ((0.2, 0.8), (0.0, 1.0), (1 / 3, 2 / 3), (0.1 + 0.2, 0.9), (1e-6, 0.5), (1e-7, 1.0),
                           (0.5, 0.25), ("0.4", 0.6), (0.123456789, 0.987654321)):
            with self.subTest(start=start, end=end):
                text = xsa.format_noise_window(start, end)
                self.assertEqual(text, self.ext.format_noise_window(start, end))
                self.assertEqual(_key(xsa.parse_noise_window(text)), _key(self.ext.parse_noise_window(text)))
                self.assertIsNotNone(xsa.parse_noise_window(text))

    def test_paste_table(self):
        for params in _PASTE_CASES:
            with self.subTest(params=params):
                want = (self.ext.paste_max_stage(params), self.ext.paste_eta(params),
                        self.ext.paste_noise_window(params), self.ext.paste_noise_start(params),
                        self.ext.paste_noise_end(params))
                self.assertEqual(_fields(xsa.from_infotext(params)), tuple(_key(value) for value in want))
        self.assertEqual(xsa.from_infotext(None), xsa.APP_DEFAULTS)

    def test_generate_from_metadata_restore(self):
        """계획 A-R1(X3b) — ``restore_from_infotext``: 메인·Hires 가 ER SDE 셋이면 차수는 paste_max_stage, Reverse-time·
        Tunable 이 있으면 η·잡음 구간도 paste_*(접은 표기는 확장 표기로 바꾼 붙여 넣기 — 'Eta' 대체가 앱 콤보가 고른
        샘플러를 본다), 다른 칸은 base 그대로(키가 있어도), ER SDE 가 없으면 None."""
        cases = _PASTE_CASES + (
            {"Sampler": "ER SDE (ODE)", "ER SDE max stage": 1, "ER SDE eta": 0.4, "ER SDE noise window": "0.1-0.9"},
            {"Sampler": "ER SDE (ODE)", "Hires sampler": "ER SDE (Tunable)", "ER SDE eta": "2.5"},
            {"Sampler": "Euler a", "ER SDE max stage": 1, "ER SDE eta": 0.4, "Eta": 0.3},
            {"Sampler": "ER SDE", "ER SDE max stage": 2, "Eta": 0.3},
            {"Sampler": "er_sde_tunable", "Hires sampler": "Use same sampler", "ER SDE noise window": "0.3-0.6"},
            {"Sampler": "Euler", "Hires sampler": "er_sde_reverse_time", "Eta": "0.45"},
        )
        bases = (xsa.APP_DEFAULTS, xsa.ExtraSamplersSettings(max_stage=1, eta=4.5, noise_window=True, noise_start=0.3,
                                                             noise_end=0.7))
        keys = (xsa.KEY_SAMPLER, xsa.KEY_HIRES_SAMPLER)
        seen = {"none": 0, "stage_only": 0, "all": 0}
        for base in bases:
            for params in cases:
                with self.subTest(base=base, params=params):
                    used = {xsa.resolve_sampler(params.get(key)) for key in keys} & set(xsa.ER_SDE_LABELS)
                    restored = xsa.restore_from_infotext(params, base=base)
                    if not used:
                        seen["none"] += 1
                        self.assertIsNone(restored)
                        continue
                    labelled = dict(params)
                    for key in keys:
                        label = xsa.resolve_sampler(params.get(key))
                        if label is not None:
                            labelled[key] = label
                    pasted = {"max_stage": self.ext.paste_max_stage(labelled), "eta": self.ext.paste_eta(labelled),
                              "noise_window": self.ext.paste_noise_window(labelled),
                              "noise_start": self.ext.paste_noise_start(labelled),
                              "noise_end": self.ext.paste_noise_end(labelled)}
                    eta_used = not used.isdisjoint(xsa.ETA_LABELS)
                    seen["all" if eta_used else "stage_only"] += 1
                    fields = ("max_stage",) + (("eta", "noise_window", "noise_start", "noise_end") if eta_used else ())
                    want = tuple(_key(pasted[name] if name in fields else getattr(base, name)) for name in xsa.ARG_NAMES)
                    self.assertEqual(_fields(restored), want)
        self.assertGreater(seen["none"], 10)
        self.assertGreater(seen["stage_only"], 1)
        self.assertGreater(seen["all"], 10)


_RECORD_SETTINGS = (
    xsa.ExtraSamplersSettings(),
    xsa.ExtraSamplersSettings(max_stage=2),
    xsa.ExtraSamplersSettings(eta=0.5),
    xsa.ExtraSamplersSettings(eta=0.1 + 0.2),
    xsa.ExtraSamplersSettings(eta=0.0, noise_window=True),
    xsa.ExtraSamplersSettings(max_stage=1, eta=2.5, noise_window=True, noise_start=0.1, noise_end=0.9),
    xsa.ExtraSamplersSettings(noise_window=True, noise_start=0.8, noise_end=0.2),
    xsa.ExtraSamplersSettings(noise_window=True, noise_start=1 / 3, noise_end=2 / 3),
)
_RECORD_SAMPLERS = (
    (), ("Euler",), (xsa.LABEL_ER_SDE_REVERSE_TIME,), (xsa.LABEL_ER_SDE_ODE,), (xsa.LABEL_ER_SDE_TUNABLE,),
    (xsa.LABEL_ER_SDE_ODE, xsa.LABEL_ER_SDE_TUNABLE), ("Euler a", xsa.LABEL_ER_SDE_REVERSE_TIME),
    (xsa.LABEL_DPMPP_4M_SDE,), (xsa.LABEL_ER_SDE_REVERSE_TIME, xsa.LABEL_ER_SDE_REVERSE_TIME),
)


@requires_extension
class InfotextParityTests(unittest.TestCase):
    """``infotext_items`` = 패스마다 record_infotext 를 부른 결과(이 기능의 키) — ComfyUI 메타데이터용."""

    @classmethod
    def setUpClass(cls):
        cls.ext = _load_ext_file(PARAMS)

    def test_record_table(self):
        compared = 0
        for settings in _RECORD_SETTINGS:
            ext_settings = self.ext.ErSdeSettings(**{name: getattr(settings, name) for name in xsa.ARG_NAMES})
            for samplers in _RECORD_SAMPLERS:
                for forge_eta in (None, 1.0, 0.5, "x", float("nan")):
                    for before in ({}, {"Eta": 0.7}):
                        for skipped in (False, True):
                            p = _P(before)
                            for label in samplers:
                                self.ext.record_infotext(p, label, ext_settings, forge_eta=forge_eta,
                                                         window_skipped=skipped)
                            want = {key: value for key, value in p.extra_generation_params.items() if key not in before}
                            got = xsa.infotext_items(settings, samplers, forge_eta=forge_eta, window_skipped=skipped,
                                                     params=before)
                            self.assertEqual({k: _key(v) for k, v in got.items()},
                                             {k: _key(v) for k, v in want.items()},
                                             (settings, samplers, forge_eta, before, skipped))
                            compared += 1
        self.assertGreater(compared, 1000)

    def test_comfy_names_record_like_labels_and_paste_back(self):
        settings = xsa.ExtraSamplersSettings(max_stage=2, eta=0.25, noise_window=True, noise_start=0.15,
                                             noise_end=0.85)
        by_name = xsa.infotext_items(settings, ["er_sde_tunable"])
        self.assertEqual(by_name, xsa.infotext_items(settings, [xsa.LABEL_ER_SDE_TUNABLE]))
        self.assertEqual(xsa.from_infotext({"Sampler": xsa.LABEL_ER_SDE_TUNABLE, **by_name}), settings)


@requires_extension
@requires_torch
class StatusParityTests(unittest.TestCase):
    """``parse_status`` 가 substep_guard.record_status·record_status_part 가 쓴 값을 읽는다(부분은 "; " 로 이어진다)."""

    @classmethod
    def setUpClass(cls):
        cls.guard = _load_ext_file(GUARD)

    def test_status_values(self):
        reasons = xsa.DY_REASONS
        cases = (((), False), ((reasons[0],), False), (reasons[:2], False), (reasons, False), ((), True),
                 ((reasons[1],), True), (reasons, True))
        for dy, window in cases:
            for window_first in (False, True):
                with self.subTest(dy=dy, window=window, window_first=window_first):
                    p = _P()
                    if window and window_first:
                        self.guard.record_status_part(p, "window", xsa.WINDOW_SKIPPED_STATUS)
                    for reason in dy:   # one pass after another, like Forge's base and hires passes
                        self.guard.record_status(p, [reason])
                    if window and not window_first:
                        self.guard.record_status_part(p, "window", xsa.WINDOW_SKIPPED_STATUS)
                    value = p.extra_generation_params.get(xsa.STATUS_KEY)
                    self.assertEqual(value is None, not dy and not window)
                    self.assertEqual(xsa.parse_status(value), xsa.SamplerStatus(tuple(dy), window, ()))

    def test_unknown_parts_are_kept(self):
        p = _P()
        self.guard.record_status(p, [xsa.DY_REASON_PID])
        self.guard.record_status_part(p, "future", "something new (x)")
        self.assertEqual(xsa.parse_status(p.extra_generation_params[xsa.STATUS_KEY]),
                         xsa.SamplerStatus((xsa.DY_REASON_PID,), False, ("something new (x)",)))


# ── 블록·계획·이름·패스 (확장 없이) ─────────────────────────────────────────────────
class BlockTests(unittest.TestCase):
    def test_default_block_is_five_positional_args(self):
        block = xsa.as_block(xsa.APP_DEFAULTS)
        self.assertEqual(block, {"args": [3, 1.0, False, 0.2, 0.8]})
        self.assertEqual([type(value) for value in block["args"]], [int, float, bool, float, float])
        user = xsa.as_block(xsa.ExtraSamplersSettings(2, 0.5, True, 0.1, 0.9))
        self.assertEqual(user, {"args": [2, 0.5, True, 0.1, 0.9]})

    def test_block_round_trips_and_old_blocks_read_with_window_defaults(self):
        for settings in _RECORD_SETTINGS:
            self.assertEqual(xsa.parse_script_block(xsa.as_block(settings)), settings)
        self.assertEqual(xsa.parse_script_block({"args": [2, 0.5]}), xsa.ExtraSamplersSettings(2, 0.5))
        self.assertEqual(xsa.parse_script_block({"args": []}), xsa.APP_DEFAULTS)
        self.assertEqual(xsa.parse_script_block({}), xsa.APP_DEFAULTS)
        self.assertIsNone(xsa.parse_script_block([2, 0.5]))
        self.assertEqual(xsa.parse_script_block({"args": [2, 0.5, True, 0.3, 0.6, "x"]}),
                         xsa.ExtraSamplersSettings(2, 0.5, True, 0.3, 0.6))

    def test_widget_values_round_trip(self):
        for settings in (*_RECORD_SETTINGS, xsa.ExtraSamplersSettings(1, 1 / 3, True, 0.0, 1.0)):
            with self.subTest(settings=settings):
                values = xsa.widget_values(settings)
                self.assertTrue(all(isinstance(value, str) for value in values.values()))
                self.assertEqual(xsa.parse_settings(values), settings)
        self.assertEqual(xsa.widget_values(xsa.APP_DEFAULTS),
                         {"max_stage": "3", "eta": "1", "noise_window": "0", "noise_start": "0.2", "noise_end": "0.8"})

    def test_parse_settings_keeps_base_for_missing_and_unreadable(self):
        base = xsa.ExtraSamplersSettings(1, 2.5, True, 0.3, 0.7)
        self.assertEqual(xsa.parse_settings({}, base=base), base)
        self.assertEqual(xsa.parse_settings(None, base=base), base)
        unreadable = {"max_stage": "x", "eta": "nan", "noise_window": None, "noise_start": True, "noise_end": ""}
        self.assertEqual(xsa.parse_settings(unreadable, base=base), base)
        self.assertEqual(xsa.parse_settings({"noise_window": "0", "eta": "12"}, base=base),
                         replace(base, noise_window=False, eta=10.0))

    def test_widget_ids_and_settings_key(self):
        self.assertEqual(tuple(xsa.widget_id(key) for key in xsa.WIDGET_KEYS),
                         ("_xsamp_max_stage", "_xsamp_eta", "_xsamp_noise_window", "_xsamp_noise_start",
                          "_xsamp_noise_end"))
        with self.assertRaises(KeyError):
            xsa.widget_id("cfg")
        self.assertEqual(xsa.SETTINGS_KEY, "extra_samplers_settings")
        self.assertEqual(tuple(key for key, *_ in xsa.ARG_SPEC),
                         tuple("xsamp_" + key for key in xsa.ARG_NAMES))
        self.assertEqual(xsa.COMFY_INPUTS, ("model", *xsa.ARG_NAMES))

    def test_empty_window(self):
        self.assertTrue(xsa.ExtraSamplersSettings(noise_window=True, noise_start=0.5, noise_end=0.5).window_empty)
        self.assertFalse(xsa.ExtraSamplersSettings(noise_window=False, noise_start=0.9, noise_end=0.1).window_empty)
        self.assertFalse(xsa.ExtraSamplersSettings(noise_window=True).window_empty)


def _caps_with_argc(argc, *, img2img=True):
    detail = {"present": True, "img2img": img2img, "live_argc": argc, "spec_argc": None,
              "trailing_unmapped": [], "extra_live_args": 0}
    return SamExtraCapabilities(status="ok", installed=True, scripts=_freeze({xsa.SCRIPT_NAME.lower(): detail}))


class PlanTests(unittest.TestCase):
    """싣는 규칙(계획 X5): 확인되면 늘, 확인 못 하면 ER SDE 패스가 있을 때만, ComfyUI 는 C6 전이라 없음. X10 경고."""

    CONFIRMED = caps(present=(xsa.SCRIPT_NAME,))
    ABSENT = caps(present=(), img2img=())
    I2I_ONLY = caps(present=(), img2img=(xsa.SCRIPT_NAME,))
    USER = xsa.ExtraSamplersSettings(eta=0.5)

    def test_krea2_and_comfy(self):
        for settings in (xsa.APP_DEFAULTS, self.USER):
            self.assertEqual(xsa.plan(settings, target="t2i", backend="krea2", capabilities=self.CONFIRMED,
                                      passes=("main",)).reason, xsa.REASON_KREA2)
            result = xsa.plan(settings, target="t2i", backend="comfyui", passes=("main",))
            self.assertEqual((result.block, result.reason), (None, xsa.REASON_COMFY_PENDING))

    def test_confirmed_always_sends_with_value_provenance(self):
        for target in ("t2i", "i2i", "aux"):
            with self.subTest(target=target):
                default = xsa.plan(xsa.APP_DEFAULTS, target=target, backend="webui", capabilities=self.CONFIRMED)
                self.assertEqual((default.reason, default.provenance), (xsa.REASON_SEND, xsa.PROVENANCE_APP_DEFAULT))
                self.assertEqual(default.block, {"args": [3, 1.0, False, 0.2, 0.8]})
                user = xsa.plan(self.USER, target=target, backend="webui", capabilities=self.CONFIRMED)
                self.assertEqual((user.reason, user.provenance), (xsa.REASON_SEND, xsa.PROVENANCE_USER))
                self.assertEqual(user.block, xsa.as_block(self.USER))
                self.assertIsNone(user.notice)

    def test_unconfirmed_sends_only_when_a_pass_reads_the_values(self):
        for label, capabilities in (("unknown", None), ("absent", self.ABSENT)):
            with self.subTest(label):
                idle = xsa.plan(self.USER, target="t2i", backend="webui", capabilities=capabilities)
                self.assertEqual((idle.block, idle.reason), (None, xsa.REASON_NOT_NEEDED))
                used = xsa.plan(self.USER, target="t2i", backend="webui", capabilities=capabilities,
                                passes=(xsa.PassUse(PASS_HIRES, xsa.LABEL_ER_SDE_ODE),))
                self.assertEqual((used.reason, used.provenance), (xsa.REASON_SEND_UNCONFIRMED, xsa.PROVENANCE_USER))
                self.assertEqual(used.block, xsa.as_block(self.USER))

    def test_img2img_reads_the_img2img_list_and_aux_follows_t2i(self):
        self.assertEqual(xsa.plan(xsa.APP_DEFAULTS, target="i2i", backend="webui", capabilities=self.I2I_ONLY).reason,
                         xsa.REASON_SEND)
        self.assertEqual(xsa.plan(xsa.APP_DEFAULTS, target="t2i", backend="webui", capabilities=self.I2I_ONLY).reason,
                         xsa.REASON_NOT_NEEDED)
        self.assertEqual(xsa.plan(xsa.APP_DEFAULTS, target="aux", backend="webui", capabilities=self.I2I_ONLY).reason,
                         xsa.REASON_NOT_NEEDED)

    def test_script_confirmed_matches_the_propagation_gate(self):
        from core import alwayson_propagation as ap

        for capabilities in (None, self.CONFIRMED, self.ABSENT, self.I2I_ONLY, caps(present=()),
                             caps(present=(xsa.SCRIPT_NAME,), status="error")):
            for img2img in (False, True):
                with self.subTest(capabilities=capabilities, img2img=img2img):
                    self.assertEqual(xsa.script_confirmed(capabilities, img2img=img2img),
                                     ap.script_available(capabilities, xsa.SCRIPT_NAME, img2img=img2img))

    def test_window_support_by_live_argument_count(self):
        on = xsa.ExtraSamplersSettings(noise_window=True)
        for argc, supported in ((2, False), (4, False), (5, True), (6, True)):
            capabilities = _caps_with_argc(argc)
            with self.subTest(argc=argc):
                self.assertEqual(xsa.live_argc(capabilities), argc)
                self.assertEqual(xsa.window_supported(capabilities), supported)
                self.assertEqual(xsa.window_unsupported(on, capabilities), not supported)
                self.assertFalse(xsa.window_unsupported(xsa.APP_DEFAULTS, capabilities))
                # X10: the block keeps five arguments whatever the installed script takes
                self.assertEqual(len(xsa.plan(on, target="t2i", backend="webui", capabilities=capabilities).block["args"]),
                                 5)
        for capabilities in (None, self.ABSENT, caps(present=(xsa.SCRIPT_NAME,), status="error"),
                             _caps_with_argc(None), _caps_with_argc(True)):
            with self.subTest(capabilities=capabilities):
                self.assertIsNone(xsa.window_supported(capabilities))
                self.assertFalse(xsa.window_unsupported(on, capabilities))
        self.assertIn(xsa.WINDOW_MIN_VERSION, xsa.WINDOW_UNSUPPORTED_TEXT)


class NameTests(unittest.TestCase):
    def test_resolve_folds_labels_comfy_names_and_reforge_aliases(self):
        for label, aliases in xsa.ALIASES.items():
            for text in (label, label.upper(), f"  {label.lower()} ", label.replace(" ", "  "), *aliases,
                         *(alias.upper() for alias in aliases)):
                with self.subTest(text=text):
                    self.assertEqual(xsa.resolve_sampler(text), label)
            self.assertEqual(xsa.comfy_name(label), aliases[0])
        for text in ("ER SDE", "er_sde", "Euler", "Euler a", "UniPC", "Restart", "DPM++ 3M SDE", "", None,
                     xsa.USE_SAME_SAMPLER, "ER SDE (Reverse-time) Karras", "euler dy cfg pp"):
            with self.subTest(text=text):
                self.assertIsNone(xsa.resolve_sampler(text))
                self.assertIsNone(xsa.sampler_kind(text))
                self.assertIsNone(xsa.comfy_name(text))

    def test_kinds_and_predicates(self):
        expected = {
            xsa.LABEL_ER_SDE_REVERSE_TIME: "er_sde", xsa.LABEL_ER_SDE_ODE: "er_sde", xsa.LABEL_ER_SDE_TUNABLE: "er_sde",
            xsa.LABEL_EULER_DY_CFG_PP: "dy", xsa.LABEL_EULER_SMEA_DY_CFG_PP: "smea", xsa.LABEL_EULER_DY: "dy",
            xsa.LABEL_EULER_SMEA_DY: "smea", xsa.LABEL_CFGPP_UD10_AB: "cfg_pp", xsa.LABEL_RESTART_FLOW: "restart",
        }
        for label in xsa.LABELS:
            with self.subTest(label=label):
                self.assertEqual(xsa.sampler_kind(label), expected.get(label, ""))
                self.assertEqual(xsa.sampler_kind(xsa.COMFY_NAMES[label]), expected.get(label, ""))
                self.assertEqual(xsa.uses_values(label), label in xsa.ER_SDE_LABELS)
                self.assertEqual(xsa.uses_eta(label), label in (xsa.LABEL_ER_SDE_REVERSE_TIME, xsa.LABEL_ER_SDE_TUNABLE))
                self.assertEqual(xsa.uses_window(label), xsa.uses_eta(label))
                self.assertEqual(xsa.is_cfg_pp(label), label in (xsa.LABEL_EULER_DY_CFG_PP,
                                                                 xsa.LABEL_EULER_SMEA_DY_CFG_PP,
                                                                 xsa.LABEL_CFGPP_UD10_AB))
        self.assertEqual(set(xsa.FLOW_ODE_LABELS), {label for label in xsa.LABELS if "flow ODE" in label})

    def test_comfy_tables(self):
        self.assertEqual(set(xsa.COMFY_DISCARD_PENULTIMATE) - set(xsa.PACK_SAMPLERS), set())
        self.assertEqual(dict(xsa.COMFY_AUTOMATIC_SCHEDULER), {"dpmpp_4m_sde": "exponential"})
        self.assertEqual(xsa.COMFY_REPORT_INPUT, "owned_names")
        self.assertIn("cfgpp_ud10_ab", xsa.CORE_SAMPLERS)


def _slot(**values):
    return {"ad_tab_enable": True, "ad_model": "face_yolov8n.pt", **values}


class PassTests(unittest.TestCase):
    """어느 패스가 값을 읽나 — 위젯(카드·기여자)과 요청 페이로드(컴파일러) 두 판이 같은 패스를 고른다."""

    RT, ODE, TUN = xsa.LABEL_ER_SDE_REVERSE_TIME, xsa.LABEL_ER_SDE_ODE, xsa.LABEL_ER_SDE_TUNABLE

    def _pair(self, *, main="Euler", hires=None, hires_on=False, ad=(), sam3=None):
        """같은 화면을 위젯 값과 페이로드 두 모양으로."""
        widgets = {"sampler_combo": main, "hires_options_group": hires_on,
                   "hires_sampler_combo": hires or "Use same sampler", "adetailer_group": bool(ad)}
        payload = {"sampler_name": main, "enable_hr": hires_on}
        if hires_on and hires and hires != "Use same sampler":
            payload["hr_sampler_name"] = hires
        slots = []
        for index, (use, sampler) in enumerate(ad, start=1):
            widgets.update({f"ad_slot{index}_group": True, f"_ad_s{index}_use_sampler": use,
                            f"_ad_s{index}_sampler": sampler})
            slots.append(_slot(ad_use_sampler=use, ad_sampler=sampler))
        scripts = {}
        if slots:
            scripts["ADetailer"] = {"args": [True, False, *slots]}
        if sam3 is not None:
            use, sampler, mode = sam3
            widgets.update({"sam3_group": True, "_sam3_mode": mode, "_sam3_use_sampler": use,
                            "_sam3_sampler": sampler})
            scripts[sam3_args.SCRIPT_SAM3] = {"args": [{"sam3_enable": True, "sam3_mode": mode,
                                                        "sam3_use_sampler": use,
                                                        "sam3_sampler": sampler if use else "Use same sampler"}]}
        if scripts:
            payload["alwayson_scripts"] = scripts
        return widgets, payload

    def test_widgets_and_payload_pick_the_same_passes(self):
        cases = (
            ({"main": self.RT}, ((PASS_MAIN, self.RT),)),
            ({"main": "Euler"}, ()),
            ({"main": "Euler", "hires_on": True, "hires": self.ODE}, ((PASS_HIRES, self.ODE),)),
            ({"main": self.TUN, "hires_on": True}, ((PASS_MAIN, self.TUN), (PASS_HIRES, self.TUN))),
            ({"main": self.TUN, "hires_on": False, "hires": "Euler"}, ((PASS_MAIN, self.TUN),)),
            ({"main": "Euler", "ad": ((True, self.RT), (False, "DPM++ 2M"))}, (("adetailer_1", self.RT),)),
            ({"main": self.ODE, "ad": ((False, "Euler"), (True, "Use same sampler"))},
             ((PASS_MAIN, self.ODE), ("adetailer_1", self.ODE), ("adetailer_2", self.ODE))),
            ({"main": "Euler", "sam3": (True, self.TUN, "Inpaint")}, ((PASS_SAM3, self.TUN),)),
            ({"main": self.RT, "sam3": (False, "Euler", "Inpaint")}, ((PASS_MAIN, self.RT), (PASS_SAM3, self.RT))),
            ({"main": self.RT, "sam3": (True, self.TUN, sam3_args.MODE_MASK_ONLY)}, ((PASS_MAIN, self.RT),)),
            ({"main": "er_sde_ode"}, ((PASS_MAIN, self.ODE),)),
        )
        for kwargs, expected in cases:
            with self.subTest(kwargs=kwargs):
                widgets, payload = self._pair(**kwargs)
                want = tuple(xsa.PassUse(*item) for item in expected)
                self.assertEqual(xsa.passes_from_widgets(widgets), want)
                self.assertEqual(xsa.value_sampler_passes(payload), want)

    def test_img2img_main_and_hires_do_not_read(self):
        widgets, _payload = self._pair(main=self.RT, hires_on=True, hires=self.ODE, ad=((True, self.TUN),))
        self.assertEqual(xsa.passes_from_widgets(widgets, img2img=True), (xsa.PassUse("adetailer_1", self.TUN),))
        widgets, _payload = self._pair(main=self.RT, ad=((False, "Euler"),))
        self.assertEqual(xsa.passes_from_widgets(widgets, img2img=True), ())

    def test_payload_details_the_compiler_reads(self):
        self.assertEqual(xsa.value_sampler_passes(None), ())
        off = {"sampler_name": "Euler", "alwayson_scripts": {"ADetailer": {"args": [False, False, _slot(
            ad_use_sampler=True, ad_sampler=self.RT)]}}}
        self.assertEqual(xsa.value_sampler_passes(off), ())
        no_model = {"sampler_name": "Euler", "alwayson_scripts": {"adetailer": {"args": [True, False, _slot(
            ad_model="None", ad_use_sampler=True, ad_sampler=self.RT)]}}}
        self.assertEqual(xsa.value_sampler_passes(no_model), ())
        self.assertEqual(xsa.value_sampler_passes({"sampler_name": "Lap"}, resolve=lambda t: self.TUN if t else None),
                         (xsa.PassUse(PASS_MAIN, self.TUN),))

    def test_pass_widget_ids_exist_in_the_qt_setup(self):
        with open(os.path.join(ROOT, "ui", "generator_ui_setup.py"), encoding="utf-8") as handle:
            source = handle.read()
        for widget in xsa.pass_widget_ids():
            with self.subTest(widget=widget):
                if widget.startswith("_ad_s"):
                    prefix, key = widget[:len("_ad_s1")], widget[len("_ad_s1_"):]
                    self.assertIn(f"f'{{prefix}}_{key}'", source)
                    self.assertIn(f"_ad_slot('{prefix}')", source)
                else:
                    self.assertIn(f"'{widget}'", source)


class DescribeTests(unittest.TestCase):
    def test_lines(self):
        self.assertEqual(xsa.describe(xsa.APP_DEFAULTS), "ER SDE 차수 3 · η 1")
        settings = xsa.ExtraSamplersSettings(2, 0.5, True, 0.2, 0.8)
        self.assertEqual(xsa.describe(settings), "ER SDE 차수 2 · η 0.5 · 잡음 0.2–0.8")
        self.assertEqual(xsa.describe(settings, ["er_sde_ode"]), "ER SDE 차수 2")
        self.assertEqual(xsa.describe(settings, [xsa.LABEL_ER_SDE_TUNABLE, "Euler"]),
                         "ER SDE 차수 2 · η 0.5 · 잡음 0.2–0.8")
        self.assertEqual(xsa.describe(settings, ["Euler"]), "")
        self.assertEqual(xsa.describe(replace(settings, noise_start=0.8, noise_end=0.8)),
                         "ER SDE 차수 2 · η 0.5 · 잡음 0.8–0.8 (빈 구간)")


class ImportTests(unittest.TestCase):
    def test_module_imports_no_torch_or_qt(self):
        with open(os.path.join(ROOT, "core", "extra_samplers.py"), encoding="utf-8") as handle:
            tree = ast.parse(handle.read())
        imported = set()
        for item in tree.body:
            if isinstance(item, ast.Import):
                imported.update(alias.name.split(".")[0] for alias in item.names)
            elif isinstance(item, ast.ImportFrom):
                imported.add((item.module or "").split(".")[0])
        self.assertLessEqual(imported, {"__future__", "math", "re", "dataclasses", "types", "typing", "core"})


class RestoreFromInfotextTests(unittest.TestCase):
    """계획 A-R1(X3b) — '이 설정으로 생성'은 이미지의 메인·Hires 샘플러가 ER SDE 셋일 때만 그 샘플러가 읽는 칸을 샘플러와
    함께 되살린다(차수는 셋 다, η·잡음 구간은 Reverse-time·Tunable). 확장 없이(확장 비교는
    ``PasteParityTests.test_generate_from_metadata_restore``)."""

    USER = xsa.ExtraSamplersSettings(max_stage=1, eta=4.5, noise_window=True, noise_start=0.3, noise_end=0.7)
    ALL_KEYS = {"ER SDE max stage": "2", "ER SDE eta": "0.5", "ER SDE noise window": "0.1-0.9", "Eta": "0.7"}
    RT, ODE, TUN = xsa.LABEL_ER_SDE_REVERSE_TIME, xsa.LABEL_ER_SDE_ODE, xsa.LABEL_ER_SDE_TUNABLE

    def test_reverse_time_and_tunable_restore_all_five(self):
        for sampler in (self.RT, self.TUN, "er_sde_reverse_time", "ER SDE (TUNABLE)"):
            with self.subTest(sampler=sampler):
                restored = xsa.restore_from_infotext({"Sampler": sampler, **self.ALL_KEYS}, base=self.USER)
                self.assertEqual(restored, xsa.ExtraSamplersSettings(max_stage=2, eta=0.5, noise_window=True,
                                                                     noise_start=0.1, noise_end=0.9))
        # 키가 없다 — 확장 붙여 넣기처럼 기본값(그 샘플러가 읽는 칸 전부)
        self.assertEqual(xsa.restore_from_infotext({"Sampler": self.RT}, base=self.USER), xsa.APP_DEFAULTS)
        self.assertEqual(xsa.restore_from_infotext({"Sampler": self.TUN, "ER SDE noise window": "off"}, base=self.USER),
                         xsa.APP_DEFAULTS)

    def test_forge_eta_fallback_is_reverse_time_only(self):
        """aoleg/Neo_ExtraSchedulers 이미지: 'ER SDE eta' 없이 Forge 'Eta' — Reverse-time 일 때만 η(설계 §4.3)."""
        self.assertEqual(xsa.restore_from_infotext({"Sampler": self.RT, "Eta": "0.7"}, base=self.USER),
                         xsa.ExtraSamplersSettings(eta=0.7))
        self.assertEqual(xsa.restore_from_infotext({"Sampler": self.RT, "Eta": "0.7", "ER SDE eta": "2"}, base=self.USER),
                         xsa.ExtraSamplersSettings(eta=2.0))
        self.assertEqual(xsa.restore_from_infotext({"Sampler": self.TUN, "Eta": "0.7"}, base=self.USER),
                         xsa.ExtraSamplersSettings(eta=1.0))
        self.assertEqual(xsa.restore_from_infotext({"Sampler": "Euler a", "Hires sampler": self.RT, "Eta": "0.4"},
                                                   base=self.USER), xsa.ExtraSamplersSettings(eta=0.4))
        # ComfyUI 이름도 앱 콤보처럼 같은 샘플러 — 'Eta' 대체도 그 샘플러를 본다(확장 paste 는 정확한 표기만 안다)
        self.assertEqual(xsa.from_infotext({"Sampler": "er_sde_reverse_time", "Eta": "0.4"}).eta, 1.0)
        self.assertEqual(xsa.restore_from_infotext({"Sampler": "er_sde_reverse_time", "Eta": "0.4"}, base=self.USER),
                         xsa.ExtraSamplersSettings(eta=0.4))

    def test_ode_restores_only_the_stage(self):
        restored = xsa.restore_from_infotext({"Sampler": self.ODE, **self.ALL_KEYS}, base=self.USER)
        self.assertEqual(restored, replace(self.USER, max_stage=2))
        self.assertEqual(xsa.restore_from_infotext({"Sampler": self.ODE}, base=self.USER), replace(self.USER, max_stage=3))
        # ODE 메인 + Tunable Hires — η·구간도
        restored = xsa.restore_from_infotext({"Sampler": self.ODE, "Hires sampler": self.TUN, **self.ALL_KEYS},
                                             base=self.USER)
        self.assertEqual(restored, xsa.ExtraSamplersSettings(max_stage=2, eta=0.5, noise_window=True, noise_start=0.1,
                                                             noise_end=0.9))
        self.assertEqual(xsa.infotext_er_sde_samplers({"Sampler": self.ODE, "Hires sampler": "er_sde_tunable"}),
                         (self.ODE, self.TUN))
        self.assertEqual(xsa.infotext_er_sde_samplers({"Sampler": self.RT, "Hires sampler": "er_sde_reverse_time"}),
                         (self.RT,))

    def test_values_present_but_not_used_leave_the_fields_alone(self):
        for params in ({"Sampler": "Euler a", **self.ALL_KEYS},
                       {"Sampler": "Euler a", "Hires sampler": xsa.USE_SAME_SAMPLER, **self.ALL_KEYS},
                       {"Sampler": "ER SDE", **self.ALL_KEYS},                         # Forge 자체 ER SDE(= Comfy er_sde)
                       {"Sampler": "DPM++ 2M SDE", **self.ALL_KEYS},
                       {"Sampler": xsa.LABEL_DPMPP_4M_SDE, "Hires sampler": xsa.LABEL_EULER_DY, **self.ALL_KEYS},
                       {"Sampler": "ER SDE (Reverse-time) Karras", **self.ALL_KEYS},   # 옛 결합 표기 — 콤보도 못 고른다
                       {"Schedule type": "Laplace", **self.ALL_KEYS}, {"Sampler": "", **self.ALL_KEYS},
                       {}, self.ALL_KEYS):
            with self.subTest(params=params):
                self.assertIsNone(xsa.restore_from_infotext(params, base=self.USER))
                self.assertEqual(xsa.infotext_er_sde_samplers(params), ())
        self.assertIsNone(xsa.restore_from_infotext(None, base=self.USER))
        params = {"Sampler": "er_sde_reverse_time", "Eta": "0.4"}
        xsa.restore_from_infotext(params, base=self.USER)
        self.assertEqual(params, {"Sampler": "er_sde_reverse_time", "Eta": "0.4"}, "넘긴 dict 를 고치지 않는다")

    def test_restore_fields_are_the_widget_keys(self):
        self.assertEqual(xsa.RESTORE_STAGE_FIELDS + xsa.RESTORE_ETA_FIELDS, xsa.ARG_NAMES)
        self.assertEqual(xsa.ETA_LABELS, (self.RT, self.TUN))


if __name__ == "__main__":
    unittest.main()
