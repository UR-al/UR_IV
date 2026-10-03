"""Extra Schedulers(sam-extra 'Extra Schedulers (sam-extra)') — core/extra_schedulers 순수 로직.

설치된 확장(``tests/_sam_extra_ext.EXT_ROOT``)이 기준이다 — 같은 입력을 양쪽에 넣어 같은 결과를 요구한다.

* ``FixtureTests`` — 녹화된 script-info(tests/fixtures/sam_extra_script_info.json)의 인자 모양·범위·기본값 = 앱 상수.
* ``ExtensionConstantTests`` — 확장 소스의 상수(AST 로 읽음, import 없음) = 앱 상수.
* ``CoercionParityTests`` — ``coerce`` = 확장 ``settings.coerce``(torch 없음 — settings.py 는 가벼운 모듈),
  ``parse_settings`` 의 base 규칙. ``CoerceArgsParityTests`` — ``coerce_args`` = ``ui_extra_schedulers.coerce_args``
  (그 모듈은 registry → schedulers → torch 를 올린다 — @requires_torch).
* ``BlockTests``·``PlanTests``·``DescribeTests`` — 확장 없이: 보내는 블록(계획 GA-XS 의 정확한 JSON), 위젯 왕복, 싣는 규칙(X5).
* ``SchedulerTableParityTests``·``PasteParityTests``·``InfotextItemsParityTests`` (@requires_torch) — 이름·라벨·별칭 =
  registry.SCHEDULER_SPECS, ``from_infotext`` = paste_*(Forge schedulers_map 은 registry 가 넣을 열쇠로 흉내),
  ``infotext_items`` = infotext_items. ``PassTests`` — 값을 읽는 패스(위젯 판·페이로드 판이 같은 패스를 고른다).

확장이 없으면 확장 비교만 skip, AISTUDIO_REQUIRE_FORGE_EXT=1 이면 실패. torch 는 지연 import(tests/_optional_deps).
"""
from __future__ import annotations

import ast
import importlib
import json
import math
import os
import sys
import types
import unittest
from dataclasses import replace
from unittest import mock

from core import extra_schedulers as xs
from tests._optional_deps import requires_torch
from tests._sam_extra_ext import EXT_ROOT, requires_extension
from tests.test_alwayson_propagation import caps

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIXTURE = os.path.join(ROOT, "tests", "fixtures", "sam_extra_script_info.json")
_MISSING = object()


def _ext(module: str):
    """설치된 확장 모듈(``sam3ext.<module>``). settings 는 가볍고 ui_extra_schedulers·registry 는 torch 를 올린다."""
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


def _fixture_entries():
    with open(FIXTURE, encoding="utf-8") as fh:
        scripts = json.load(fh)["scripts"]
    return [s for s in scripts if s.get("name") == xs.SCRIPT_NAME.lower()]


def _key(value):
    """float 는 repr 로 — -0.0 과 0.0, 마지막 비트까지 다르게 본다."""
    if isinstance(value, float):
        return ("float", repr(value))
    return (type(value).__name__, value)


def _fields(settings) -> tuple:
    return tuple(_key(getattr(settings, name)) for name in xs.ARG_NAMES)


# ── 비교 입력 ─────────────────────────────────────────────────────────────────
_MODE_VALUES = (_MISSING, None, "", "expression", "Expression", "EXPRESSION", "sigmas", "Sigmas", "Sigma list",
                "sigma list", " sigma_list ", "SIGMA", "list", "List", "sigma lists", "other", 0, 1, True, False,
                ["sigmas"], 1.5)
_TEXT_VALUES = (_MISSING, None, "", "   ", "  M * x  ", "m\n+ x", 0, 1.5, True, ["a"], "x" * 3000)
_NUMBER_VALUES = (_MISSING, None, True, False, 0, 1, -1, 0.0, -0.0, 0.1, -0.1, 0.5, 2.0, -2.5, 9.99, 10, 10.0,
                  10.5, -10, -10.5, 11, 1e9, -1e9, float("nan"), float("inf"), float("-inf"), "nan", "inf", "-inf",
                  "1e9", " 2 ", "3.25", "-0", "abc", "", [], {}, (1,), 1e-320, -1e-320, 1e308, "1e400")


def _raw_corpus() -> list:
    corpus: list = [None, {}]
    for value in _MODE_VALUES:
        corpus.append({} if value is _MISSING else {"custom_mode": value})
    for key in ("custom_expression", "custom_sigmas"):
        for value in _TEXT_VALUES:
            corpus.append({} if value is _MISSING else {key: value})
    for key in ("laplace_mu", "laplace_beta"):
        for value in _NUMBER_VALUES:
            corpus.append({} if value is _MISSING else {key: value})
    corpus.append({"custom_mode": "Sigma list", "custom_expression": " m ", "custom_sigmas": "[1, 0]",
                   "laplace_mu": "0.75", "laplace_beta": 3, "unknown": 1})
    return corpus


class FixtureTests(unittest.TestCase):
    """녹화된 script-info(확장 ui())의 다섯 칸 = 앱 상수."""

    def test_both_tabs_have_the_five_args_with_the_app_ranges_and_defaults(self):
        entries = _fixture_entries()
        self.assertEqual(sorted(e["is_img2img"] for e in entries), [False, True])
        for entry in entries:
            with self.subTest(img2img=entry["is_img2img"]):
                mode, expression, sigmas, mu, beta = entry["args"]
                self.assertEqual(len(entry["args"]), len(xs.ARG_NAMES))
                self.assertEqual(tuple(mode["choices"]), tuple(xs.MODE_LABELS[m] for m in xs.MODES))
                self.assertEqual(xs.coerce_mode(mode["value"]), xs.EXTENSION_DEFAULTS.custom_mode)
                self.assertEqual(expression["value"], xs.DEFAULT_EXPRESSION)
                self.assertEqual(sigmas["value"], xs.DEFAULT_SIGMAS)
                self.assertEqual((mu["minimum"], mu["maximum"], mu["step"], mu["value"]),
                                 (xs.LAPLACE_MU_MIN, xs.LAPLACE_MU_MAX, xs.LAPLACE_MU_STEP, xs.LAPLACE_MU_DEFAULT))
                self.assertEqual((beta["minimum"], beta["maximum"], beta["step"], beta["value"]),
                                 (xs.LAPLACE_BETA_MIN, xs.LAPLACE_BETA_MAX, xs.LAPLACE_BETA_STEP,
                                  xs.LAPLACE_BETA_DEFAULT))

    def test_every_live_mode_label_coerces_to_its_key_in_extension_order(self):
        labels = _fixture_entries()[0]["args"][0]["choices"]
        self.assertEqual(tuple(xs.coerce_mode(label) for label in labels), xs.MODES)

    def test_app_defaults_equal_the_extension_defaults(self):
        self.assertIs(xs.APP_DEFAULTS, xs.EXTENSION_DEFAULTS)
        self.assertEqual(xs.APP_DEFAULTS, xs.ExtraSchedulerSettings(
            "expression", "M * (m / M) ** x", "[1.0, 0.6, 0.25, 0.1, 0.0]", 0.0, 0.5))


@requires_extension
class ExtensionConstantTests(unittest.TestCase):
    """확장 소스 상수(AST) = 앱 상수. 레지스트리 핀(xsched_*)도 같은 값을 지키지만 여기서는 core 모듈 전체를 본다."""

    UI = "sam3ext/ui_extra_schedulers.py"
    SETTINGS = "sam3ext/extra_schedulers/settings.py"
    REGISTRY = "sam3ext/extra_schedulers/registry.py"

    def test_title_arg_names_and_infotext_keys(self):
        self.assertEqual(_ext_constant(self.UI, "TITLE"), xs.SCRIPT_NAME)
        self.assertEqual(_ext_constant(self.UI, "ARG_NAMES"), xs.ARG_NAMES)
        self.assertEqual(_ext_constant(self.UI, "USE_SAME_SCHEDULER"), xs.USE_SAME_SCHEDULER)
        for name in ("KEY_CUSTOM_EXPRESSION", "KEY_CUSTOM_SIGMAS", "KEY_LAPLACE_MU", "KEY_LAPLACE_BETA"):
            with self.subTest(key=name):
                self.assertEqual(_ext_constant(self.UI, name), getattr(xs, name))

    def test_settings_defaults_and_ranges(self):
        for name in ("MODE_EXPRESSION", "MODE_SIGMAS", "DEFAULT_EXPRESSION", "DEFAULT_SIGMAS"):
            with self.subTest(name=name):
                self.assertEqual(_ext_constant(self.SETTINGS, name), getattr(xs, name))
        settings = _ext("extra_schedulers.settings")
        self.assertEqual(settings.MODES, xs.MODES)
        for name in ("LAPLACE_MU_MIN", "LAPLACE_MU_MAX", "LAPLACE_MU_DEFAULT", "LAPLACE_BETA_MIN",
                     "LAPLACE_BETA_MAX", "LAPLACE_BETA_DEFAULT"):
            with self.subTest(name=name):
                self.assertEqual(_key(getattr(settings, name)), _key(getattr(xs, name)))
        self.assertEqual(_fields(settings.DEFAULTS), _fields(xs.EXTENSION_DEFAULTS))

    def test_scheduler_labels_and_readme_aliases(self):
        self.assertEqual(_ext_constant(self.REGISTRY, "README_LABELS"), dict(xs.README_LABELS))
        self.assertEqual((_ext_constant(self.REGISTRY, "LABEL_LAPLACE"), _ext_constant(self.REGISTRY, "LABEL_CUSTOM")),
                         (xs.LABEL_LAPLACE, xs.LABEL_CUSTOM))
        source = (EXT_ROOT / "sam3ext/extra_schedulers/schedulers.py").read_text(encoding="utf-8")
        self.assertGreaterEqual(source.count(f'f"{xs.ERROR_PREFIX} '), 8, "스케줄러 오류 문구의 머리(ERROR_PREFIX)")


@requires_extension
class CoercionParityTests(unittest.TestCase):
    """``coerce`` = 확장 settings.coerce — 칸마다 같은 값(float 는 비트까지)."""

    @classmethod
    def setUpClass(cls):
        cls.settings = _ext("extra_schedulers.settings")

    def _ext_outcome(self, raw):
        try:
            return "ok", _fields(self.settings.coerce(raw))
        except Exception as exc:   # 확장이 내는 예외도 계약이다(앱이 일부러 다르게 하는 것은 아래 테스트)
            return "error", type(exc).__name__

    def test_corpus(self):
        corpus = _raw_corpus()
        self.assertGreater(len(corpus), 100)
        for raw in corpus:
            with self.subTest(raw=repr(raw)[:80]):
                self.assertEqual(("ok", _fields(xs.coerce(raw))), self._ext_outcome(raw))

    def test_huge_int_is_the_one_deliberate_difference(self):
        """float 로 못 바꾸는 큰 int: 확장은 OverflowError(그 요청 실패), 앱은 기본값(블록이 통째로 빠지지 않게)."""
        for key in ("laplace_mu", "laplace_beta"):
            with self.subTest(key=key):
                raw = {key: 10 ** 400}
                self.assertEqual(self._ext_outcome(raw), ("error", "OverflowError"))
                self.assertEqual(getattr(xs.coerce(raw), key), getattr(xs.EXTENSION_DEFAULTS, key))

    def test_individual_coercers_match(self):
        for value in _NUMBER_VALUES:
            if value is _MISSING:
                continue
            with self.subTest(value=repr(value)):
                self.assertEqual(_key(xs.coerce_laplace_mu(value)), _key(self.settings.coerce_laplace_mu(value)))
                self.assertEqual(_key(xs.coerce_laplace_beta(value)), _key(self.settings.coerce_laplace_beta(value)))
        for value in _MODE_VALUES:
            if value is not _MISSING:
                self.assertEqual(xs.coerce_mode(value), self.settings.coerce_mode(value))
        for value in _TEXT_VALUES:
            if value is not _MISSING:
                self.assertEqual(xs.coerce_text(value, "d"), self.settings.coerce_text(value, "d"))

    def test_parse_settings_with_default_base_is_coerce(self):
        for raw in _raw_corpus():
            with self.subTest(raw=repr(raw)[:80]):
                self.assertEqual(_fields(xs.parse_settings(raw)), _fields(self.settings.coerce(raw)))


@requires_extension
@requires_torch
class CoerceArgsParityTests(unittest.TestCase):
    """``coerce_args`` = 확장 ui_extra_schedulers.coerce_args — 위치 인자(짧은 것 포함)·dict·쓰레기."""

    @classmethod
    def setUpClass(cls):
        cls.ui = _ext("ui_extra_schedulers")

    def test_positional_dict_and_junk(self):
        cases = [None, [], (), ["sigmas"], ["Sigma list", " m + x "], ("expression", "M", "[1, 0]", 2.5),
                 ["sigmas", "M", " [1.0, 0.5, 0.0] ", "nan", "11"], ["list", None, None, True, -0.0],
                 [1, 2, 3, 4, 5, 6], [{"laplace_mu": 3}], [{"custom_mode": "Sigma list", "laplace_beta": "inf"}],
                 [{}, "ignored"], [{"custom_expression": None, "laplace_mu": "-2.25"}]]
        cases += [[raw] for raw in _raw_corpus() if isinstance(raw, dict)]
        for args in cases:
            with self.subTest(args=repr(args)[:80]):
                self.assertEqual(_fields(xs.coerce_args(args)), _fields(self.ui.coerce_args(args)))
                block = {"args": list(args) if args is not None else None}
                self.assertEqual(_fields(xs.parse_script_block(block)), _fields(self.ui.coerce_args(args)))

    def test_infotext_number_format(self):
        for value in (0.0, -0.0, 0.5, 1 / 3, -2.5, 10.0, 1e-7):
            self.assertEqual(xs.format_number(value), self.ui.format_number(value))
        for text in ("  M  *  x ", "[1.0,\n 0.0]", "", None, "a\tb"):
            self.assertEqual(xs.normalize_text(text), self.ui.normalize_text(text))


class BlockTests(unittest.TestCase):
    def test_default_block_is_the_design_json_exactly(self):
        """계획 GA-XS 의 기준 — design_schedulers.md §3.1 의 블록(키 순서·타입까지)."""
        expected = ('{"args": [{"custom_mode": "expression", "custom_expression": "M * (m / M) ** x", '
                    '"custom_sigmas": "[1.0, 0.6, 0.25, 0.1, 0.0]", "laplace_mu": 0.0, "laplace_beta": 0.5}]}')
        self.assertEqual(json.dumps(xs.as_block(xs.APP_DEFAULTS)), expected)
        self.assertEqual(tuple(xs.as_block(xs.APP_DEFAULTS)["args"][0]), xs.ARG_NAMES)

    def test_block_round_trips_through_the_extension_rules(self):
        for settings in (xs.APP_DEFAULTS,
                         xs.ExtraSchedulerSettings("sigmas", "m + (M - m) * (1 - x) ** 3", "[1, 0.5, 0]", -2.5, 3.0),
                         xs.ExtraSchedulerSettings("expression", "", "", 10.0, 0.0)):
            with self.subTest(settings=settings):
                block = xs.as_block(settings)
                self.assertIsInstance(block["args"][0]["laplace_mu"], float)
                self.assertEqual(xs.parse_script_block(block), settings)
                self.assertEqual(xs.parse_script_block(json.loads(json.dumps(block))), settings)

    def test_positional_and_partial_blocks(self):
        self.assertEqual(xs.parse_script_block({"args": ["Sigma list", "x", "[1, 0]"]}),
                         xs.ExtraSchedulerSettings("sigmas", "x", "[1, 0]", 0.0, 0.5))
        self.assertEqual(xs.parse_script_block({"args": []}), xs.EXTENSION_DEFAULTS)
        self.assertEqual(xs.parse_script_block({}), xs.EXTENSION_DEFAULTS)
        self.assertIsNone(xs.parse_script_block(None))
        self.assertIsNone(xs.parse_script_block(["expression"]))

    def test_widget_values_round_trip(self):
        for settings in (xs.APP_DEFAULTS, xs.ExtraSchedulerSettings("sigmas", " [1, 0] ", "[0.9, 0.0]", 1 / 3, 9.9),
                         xs.ExtraSchedulerSettings("expression", "M", "", -10.0, 10.0)):
            with self.subTest(settings=settings):
                values = xs.widget_values(settings)
                self.assertEqual(set(values), set(xs.WIDGET_KEYS))
                self.assertTrue(all(isinstance(v, str) for v in values.values()))
                parsed = xs.parse_settings(values)
                self.assertEqual(parsed, replace(settings, custom_expression=settings.custom_expression.strip(),
                                                 custom_sigmas=settings.custom_sigmas.strip()))
        self.assertEqual(xs.widget_values(xs.APP_DEFAULTS)["laplace_mu"], "0")
        self.assertEqual(xs.widget_values(xs.APP_DEFAULTS)["laplace_beta"], "0.5")
        self.assertEqual(xs.widget_number(2.0), "2")
        self.assertEqual(xs.widget_number(-0.0), "0")

    def test_parse_settings_keeps_base_for_missing_and_unreadable(self):
        base = xs.ExtraSchedulerSettings("sigmas", "x", "[1, 0]", 2.0, 1.5)
        self.assertEqual(xs.parse_settings({}, base=base), base)
        self.assertEqual(xs.parse_settings(None, base=base), base)
        self.assertEqual(xs.parse_settings({"laplace_mu": "abc", "laplace_beta": "", "custom_mode": None}, base=base),
                         base)
        self.assertEqual(xs.parse_settings({"laplace_mu": "12", "custom_expression": "  M  "}, base=base),
                         replace(base, laplace_mu=10.0, custom_expression="M"))
        self.assertEqual(xs.parse_settings({"custom_sigmas": ""}, base=base).custom_sigmas, "",
                         "빈 목록은 값이다 — 확장처럼 그대로 보내고 생성할 때 확장이 거절한다")

    def test_widget_ids(self):
        self.assertEqual([xs.widget_id(key) for key in xs.WIDGET_KEYS],
                         ["_xsched_custom_mode", "_xsched_custom_expression", "_xsched_custom_sigmas",
                          "_xsched_laplace_mu", "_xsched_laplace_beta"])
        with self.assertRaises(KeyError):
            xs.widget_id("enabled")


class PlanTests(unittest.TestCase):
    """싣는 규칙(계획 X5): 확인되면 늘, 확인 못 하면 laplace·custom 패스가 있을 때만, ComfyUI 는 C6 전이라 없음."""

    CONFIRMED = caps(present=(xs.SCRIPT_NAME,))
    ABSENT = caps(present=(), img2img=())
    I2I_ONLY = caps(present=(), img2img=(xs.SCRIPT_NAME,))
    USER = xs.ExtraSchedulerSettings(laplace_mu=0.5)

    def test_krea2_and_comfy(self):
        for settings in (xs.APP_DEFAULTS, self.USER):
            self.assertEqual(xs.plan(settings, target="t2i", backend="krea2", capabilities=self.CONFIRMED,
                                     passes=("main",)).reason, xs.REASON_KREA2)
            result = xs.plan(settings, target="t2i", backend="comfyui", passes=("main",))
            self.assertIsNone(result.block)
            self.assertEqual(result.reason, xs.REASON_COMFY_PENDING)

    def test_confirmed_always_sends_with_value_provenance(self):
        for target in ("t2i", "i2i", "aux"):
            with self.subTest(target=target):
                default = xs.plan(xs.APP_DEFAULTS, target=target, backend="webui", capabilities=self.CONFIRMED)
                self.assertEqual((default.reason, default.provenance), (xs.REASON_SEND, xs.PROVENANCE_APP_DEFAULT))
                self.assertEqual(default.block, xs.as_block(xs.APP_DEFAULTS))
                user = xs.plan(self.USER, target=target, backend="webui", capabilities=self.CONFIRMED)
                self.assertEqual((user.reason, user.provenance), (xs.REASON_SEND, xs.PROVENANCE_USER))
                self.assertEqual(user.block, xs.as_block(self.USER))
                self.assertIsNone(user.notice)

    def test_unconfirmed_sends_only_when_a_pass_reads_the_values(self):
        for label, capabilities in (("unknown", None), ("absent", self.ABSENT)):
            with self.subTest(label):
                idle = xs.plan(self.USER, target="t2i", backend="webui", capabilities=capabilities)
                self.assertEqual((idle.block, idle.reason), (None, xs.REASON_NOT_NEEDED))
                used = xs.plan(self.USER, target="t2i", backend="webui", capabilities=capabilities,
                               passes=("hires",))
                self.assertEqual((used.reason, used.provenance), (xs.REASON_SEND_UNCONFIRMED, xs.PROVENANCE_USER))
                self.assertEqual(used.block, xs.as_block(self.USER))

    def test_img2img_reads_the_img2img_list_and_aux_follows_t2i(self):
        self.assertEqual(xs.plan(xs.APP_DEFAULTS, target="i2i", backend="webui", capabilities=self.I2I_ONLY).reason,
                         xs.REASON_SEND)
        self.assertEqual(xs.plan(xs.APP_DEFAULTS, target="t2i", backend="webui", capabilities=self.I2I_ONLY).reason,
                         xs.REASON_NOT_NEEDED)
        self.assertEqual(xs.plan(xs.APP_DEFAULTS, target="aux", backend="webui", capabilities=self.I2I_ONLY).reason,
                         xs.REASON_NOT_NEEDED)

    def test_script_confirmed_matches_the_propagation_gate(self):
        from core import alwayson_propagation as ap
        for capabilities in (None, self.CONFIRMED, self.ABSENT, self.I2I_ONLY, caps(present=()),
                             caps(present=(xs.SCRIPT_NAME,), status="error")):
            for img2img in (False, True):
                with self.subTest(capabilities=capabilities, img2img=img2img):
                    self.assertEqual(xs.script_confirmed(capabilities, img2img=img2img),
                                     ap.script_available(capabilities, xs.SCRIPT_NAME, img2img=img2img))


class DescribeTests(unittest.TestCase):
    def test_lines(self):
        self.assertEqual(xs.describe(xs.APP_DEFAULTS),
                         "custom 식 'M * (m / M) ** x' · Laplace μ 0 · β 0.5")
        sigmas = xs.ExtraSchedulerSettings("sigmas", custom_sigmas="[1.0,\n 0.0]", laplace_mu=-1.5)
        self.assertEqual(xs.describe(sigmas, ["Laplace"]), "Laplace μ -1.5 · β 0.5")
        self.assertEqual(xs.describe(sigmas, ["custom"]), "custom 목록 [1.0, 0.0]")
        self.assertEqual(xs.describe(sigmas, ["karras"]), "")


# ── A-XS2: 스케줄러 이름·패스·infotext ─────────────────────────────────────────────
def _ext_schedulers_map():
    """설치된 확장 registry 가 Forge schedulers_map 에 넣을 열쇠(이름·라벨·README 별칭 → 스케줄러) — Forge 없이 흉내."""
    registry = _ext("extra_schedulers.registry")
    mapping = {}
    for spec in registry.SCHEDULER_SPECS:
        scheduler = types.SimpleNamespace(name=spec.name, label=spec.label, function=spec.function,
                                          aliases=list(spec.aliases))
        for key in (spec.name, spec.label, *spec.aliases):
            mapping.setdefault(key, scheduler)
    return mapping


@requires_extension
@requires_torch
class SchedulerTableParityTests(unittest.TestCase):
    """이름·라벨·README 별칭 = 확장 registry.SCHEDULER_SPECS(생성자 호출이라 AST 핀 대신 import)."""

    @classmethod
    def setUpClass(cls):
        cls.registry = _ext("extra_schedulers.registry")

    def test_names_labels_and_aliases(self):
        specs = self.registry.SCHEDULER_SPECS
        self.assertEqual(tuple(spec.name for spec in specs), xs.SCHEDULER_NAMES)
        self.assertEqual({spec.name: spec.label for spec in specs}, dict(xs.SCHEDULER_LABELS))
        self.assertEqual(self.registry.SCHEDULER_LABELS, tuple(xs.SCHEDULER_LABELS[n] for n in xs.SCHEDULER_NAMES))
        for spec in specs:
            for text in (spec.name, spec.label, *spec.aliases, spec.label.upper(), f"  {spec.label.lower()} "):
                with self.subTest(text=text):
                    self.assertEqual(xs.resolve_scheduler(text), spec.name)
        for text in ("Karras", "Automatic", "simple", "", None, xs.USE_SAME_SCHEDULER, "laplace scheduler"):
            self.assertIsNone(xs.resolve_scheduler(text))
        self.assertEqual(set(xs.VALUE_SCHEDULERS), {s.name for s in specs if s.label in (
            self.registry.LABEL_LAPLACE, self.registry.LABEL_CUSTOM)})
        self.assertEqual(set(xs.FLOW_UNSUITABLE) | set(xs.VALUE_SCHEDULERS), set(xs.SCHEDULER_NAMES))


_INFOTEXT_CASES = (
    {},
    {"Schedule type": "Karras"},
    {"Custom scheduler expression": "m + (M - m) * (1 - x) ** 3", "Schedule type": "custom"},
    {"Custom scheduler sigmas": "[1.0, 0.6, 0.25, 0.1, 0.0]", "Schedule type": "custom"},
    {"Custom scheduler sigmas": " [0.9, 0.1] ", "Custom scheduler expression": " M ", "Schedule type": "custom"},
    {"Custom scheduler expression": 2, "Schedule type": "custom"},
    {"Schedule type": "Laplace", "Laplace mu": 0.5, "Laplace beta": 1.25},
    {"Schedule type": "Laplace"},
    {"Schedule type": "Laplace", "Laplace mu": "12"},
    {"Schedule type": "Laplace", "Laplace beta": "nan"},
    {"Schedule type": "laplace"},
    {"Schedule type": "LAPLACE"},
    {"Schedule type": "Karras", "Hires schedule type": "Laplace"},
    {"Schedule type": "Laplace", "Hires schedule type": "Use same scheduler"},
    {"Schedule type": "Karras", "Laplace mu": -3},
    {"Schedule type": "Karras Dynamic", "Hires schedule type": "custom", "Custom scheduler expression": "x"},
    {"Schedule type": "cosine-exponential blend"},
    {"Schedule type": "", "Laplace beta": 7},
    {"Schedule type": None, "Hires schedule type": "laplace", "Custom scheduler sigmas": None},
)


@requires_extension
@requires_torch
class PasteParityTests(unittest.TestCase):
    """``from_infotext`` = 확장 paste_*(Gradio 가 None 이 아닌 값만 칸에 넣고, 생성할 때 coerce 한다)."""

    @classmethod
    def setUpClass(cls):
        cls.ui = _ext("ui_extra_schedulers")
        cls.settings = _ext("extra_schedulers.settings")
        cls.schedulers_map = _ext_schedulers_map()

    def _ext_paste(self, params, base):
        fields = {name: getattr(base, name) for name in xs.ARG_NAMES}
        with mock.patch.object(self.ui, "lookup_scheduler", side_effect=lambda name: self.schedulers_map.get(name)):
            pasted = {"custom_mode": self.ui.paste_mode(params), "custom_expression": self.ui.paste_expression(params),
                      "custom_sigmas": self.ui.paste_sigmas(params), "laplace_mu": self.ui.paste_laplace_mu(params),
                      "laplace_beta": self.ui.paste_laplace_beta(params)}
        fields.update({key: value for key, value in pasted.items() if value is not None})
        return self.settings.coerce(fields)

    def test_table(self):
        bases = (xs.APP_DEFAULTS, xs.ExtraSchedulerSettings("sigmas", "M * 0.5", "[2, 1, 0]", -1.0, 4.0))
        for base in bases:
            for params in _INFOTEXT_CASES:
                with self.subTest(base=base, params=params):
                    self.assertEqual(_fields(xs.from_infotext(params, base=base)),
                                     _fields(self._ext_paste(params, base)))

    def test_labels_in_params(self):
        with mock.patch.object(self.ui, "lookup_scheduler", side_effect=lambda name: self.schedulers_map.get(name)):
            for params in _INFOTEXT_CASES:
                with self.subTest(params=params):
                    self.assertEqual(set(xs.infotext_labels(params)), self.ui._labels_in_params(params))


@requires_extension
@requires_torch
class InfotextItemsParityTests(unittest.TestCase):
    """``infotext_items`` = 확장 infotext_items(쓴 스케줄러의 라벨 집합) — ComfyUI 메타데이터가 Forge 와 같게 적는다."""

    @classmethod
    def setUpClass(cls):
        cls.ui = _ext("ui_extra_schedulers")
        cls.settings = _ext("extra_schedulers.settings")

    def test_items(self):
        variants = (xs.APP_DEFAULTS, xs.ExtraSchedulerSettings("sigmas", "M", " [1.0,\n 0.5,  0.0] ", -0.0, 1 / 3),
                    xs.ExtraSchedulerSettings("expression", "  m +   (M - m) * x ", "[1, 0]", 10.0, 0.0))
        for settings in variants:
            current = self.settings.ExtraSchedulerSettings(**{n: getattr(settings, n) for n in xs.ARG_NAMES})
            for labels in ((), ("Laplace",), ("custom",), ("Laplace", "custom"), ("Karras",), ("Cosine", "Laplace")):
                with self.subTest(settings=settings, labels=labels):
                    self.assertEqual(xs.infotext_items(settings, labels), self.ui.infotext_items(current, labels))
                    names = [xs.resolve_scheduler(label) or label for label in labels]
                    self.assertEqual(xs.infotext_items(settings, names), self.ui.infotext_items(current, labels))

    def test_items_paste_back(self):
        settings = xs.ExtraSchedulerSettings("sigmas", "M", "[1.0, 0.5, 0.0]", 1 / 3, 2.5)
        params = {"Schedule type": "Laplace", "Hires schedule type": "custom",
                  **xs.infotext_items(settings, ["laplace", "custom"])}
        self.assertEqual(xs.from_infotext(params, base=xs.APP_DEFAULTS),
                         replace(settings, custom_expression=xs.DEFAULT_EXPRESSION))


def _slot(**values):
    return {"ad_tab_enable": True, "ad_model": "face_yolov8n.pt", **values}


class PassTests(unittest.TestCase):
    """어느 패스가 값을 읽나 — 위젯(카드·기여자)과 요청 페이로드(컴파일러) 두 판이 같은 패스를 고른다."""

    PAIRS = (
        ("main simple", {"scheduler_combo": "simple"}, {"scheduler": "simple"}, ()),
        ("main laplace", {"scheduler_combo": "laplace"}, {"scheduler": "Laplace"}, (("main", "laplace"),)),
        ("hires off", {"scheduler_combo": "simple", "hires_options_group": False, "hires_scheduler_combo": "custom"},
         {"scheduler": "simple", "hr_scheduler": "custom"}, ()),
        ("hires custom", {"scheduler_combo": "simple", "hires_options_group": "true", "hires_scheduler_combo": "custom"},
         {"scheduler": "simple", "enable_hr": True, "hr_scheduler": "custom"}, (("hires", "custom"),)),
        ("hires same", {"scheduler_combo": "Laplace", "hires_options_group": True,
                        "hires_scheduler_combo": "Use same scheduler"},
         {"scheduler": "laplace", "enable_hr": True}, (("main", "laplace"), ("hires", "laplace"))),
        ("adetailer own scheduler", {"scheduler_combo": "karras", "adetailer_group": True, "ad_slot1_group": True,
                                     "_ad_s1_use_sampler": True, "_ad_s1_scheduler": "laplace",
                                     "ad_slot2_group": True, "_ad_s2_use_sampler": False, "_ad_s2_scheduler": "custom"},
         {"scheduler": "karras", "alwayson_scripts": {"ADetailer": {"args": [True, False, _slot(
             ad_use_sampler=True, ad_scheduler="laplace"), _slot(ad_use_sampler=False, ad_scheduler="custom")]}}},
         (("adetailer_1", "laplace"),)),
        ("adetailer inherits main", {"scheduler_combo": "custom", "adetailer_group": True, "ad_slot1_group": True,
                                     "_ad_s1_use_sampler": True, "_ad_s1_scheduler": "Use same scheduler"},
         {"scheduler": "custom", "alwayson_scripts": {"ADetailer": {"args": [True, False, _slot(
             ad_use_sampler=True, ad_scheduler="Use same scheduler")]}}},
         (("main", "custom"), ("adetailer_1", "custom"))),
        ("adetailer group off", {"scheduler_combo": "karras", "adetailer_group": False, "ad_slot1_group": True,
                                 "_ad_s1_use_sampler": True, "_ad_s1_scheduler": "laplace"},
         {"scheduler": "karras"}, ()),
        ("sam3 own", {"scheduler_combo": "karras", "sam3_group": True, "_sam3_mode": "Inpaint",
                      "_sam3_use_scheduler": True, "_sam3_scheduler": "Laplace"},
         {"scheduler": "karras", "alwayson_scripts": {"SAM3 Mask": {"args": [{
             "sam3_enable": True, "sam3_mode": "Inpaint", "sam3_scheduler": "Laplace"}]}}},
         (("sam3", "laplace"),)),
        ("sam3 mask only", {"scheduler_combo": "laplace", "sam3_group": True, "_sam3_mode": "Mask only",
                            "_sam3_use_scheduler": True, "_sam3_scheduler": "custom"},
         {"scheduler": "laplace", "alwayson_scripts": {"SAM3 Mask": {"args": [{
             "sam3_mode": "mask only", "sam3_scheduler": "custom"}]}}},
         (("main", "laplace"),)),
        ("sam3 switch off", {"scheduler_combo": "karras", "sam3_group": True, "_sam3_mode": "Inpaint",
                             "_sam3_use_scheduler": False, "_sam3_scheduler": "custom"},
         {"scheduler": "karras", "alwayson_scripts": {"SAM3 Mask": {"args": [{
             "sam3_mode": "Inpaint", "sam3_use_scheduler": False, "sam3_scheduler": "Use same scheduler"}]}}}, ()),
    )

    def test_widgets_and_payload_pick_the_same_passes(self):
        for label, widgets, payload, expected in self.PAIRS:
            with self.subTest(label):
                expected = tuple(xs.PassUse(*item) for item in expected)
                self.assertEqual(xs.passes_from_widgets(widgets), expected)
                self.assertEqual(xs.value_scheduler_passes(payload), expected)

    def test_payload_rules_the_compiler_reads(self):
        skipped = {"scheduler": "karras", "alwayson_scripts": {"ADetailer": {"args": [True, False,
            _slot(ad_tab_enable=False, ad_use_sampler=True, ad_scheduler="laplace"),
            _slot(ad_model="None", ad_use_sampler=True, ad_scheduler="laplace"),
            _slot(ad_use_sampler=True, ad_scheduler="custom")]}}}
        self.assertEqual(xs.value_scheduler_passes(skipped), (xs.PassUse("adetailer_1", "custom"),))
        off = {"scheduler": "karras", "alwayson_scripts": {"ADetailer": {"args": [False, False, _slot(
            ad_use_sampler=True, ad_scheduler="laplace")]}, "SAM3 Mask": {"args": [{
                "sam3_enable": False, "sam3_scheduler": "laplace"}]}}}
        self.assertEqual(xs.value_scheduler_passes(off), ())
        self.assertEqual(xs.value_scheduler_passes({"scheduler": "LAPLACE", "alwayson_scripts": {"sam3 mask": {
            "args": [{"sam3_scheduler": "custom"}]}}}),
            (xs.PassUse("main", "laplace"), xs.PassUse("sam3", "custom")))
        self.assertEqual(xs.value_scheduler_passes(None), ())
        self.assertEqual(xs.value_scheduler_passes({}), ())
        # ComfyUI 컴파일러는 자기 이름 변환을 넘긴다(계획 C2a) — 그 결과로 판정한다
        self.assertEqual(xs.value_scheduler_passes({"scheduler": "Lap"}, resolve=lambda t: "laplace" if t else None),
                         (xs.PassUse("main", "laplace"),))

    def test_pass_widget_ids_and_labels(self):
        ids = xs.pass_widget_ids()
        self.assertEqual(len(ids), len(set(ids)))
        self.assertEqual(ids[0], "scheduler_combo")
        self.assertIn("_sam3_use_scheduler", ids)
        self.assertEqual([xs.pass_label(p) for p in ("main", "hires", "adetailer_2", "sam3")],
                         ["메인", "Hires", "ADetailer 2", "SAM3"])

    def test_pass_widget_ids_exist_in_the_qt_setup(self):
        """id 는 ui/generator_ui_setup 의 프록시 id 와 같아야 한다(오타면 패스가 늘 꺼진 것으로 보인다)."""
        with open(os.path.join(ROOT, "ui", "generator_ui_setup.py"), encoding="utf-8") as fh:
            source = fh.read()
        for widget in xs.pass_widget_ids():
            with self.subTest(widget=widget):
                if widget.startswith("_ad_s"):
                    prefix, key = widget[:len("_ad_s1")], widget[len("_ad_s1_"):]
                    self.assertIn(f"f'{{prefix}}_{key}'", source)
                    self.assertIn(f"_ad_slot('{prefix}')", source)
                else:
                    self.assertIn(f"'{widget}'", source)

if __name__ == "__main__":
    unittest.main()
