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
* ``ReactDynSFTests`` — v0.33.0 의 값 없는 스케줄러: 이름·라벨이 풀리고 블록·패스·infotext 에 아무것도 더하지 않는다.
  ``FlowCosmosRho7Tests`` — v0.33.0 Flow Cosmos rho7: 이름·라벨이 풀리고(붙여 넣은 라벨이 Forge 콤보에서 그 이름을 고른다)
  D19 부터 값 셋(ρ·σ̃ max·σ̃ min)을 읽는 값 스케줄러다(패스·infotext·붙여 넣기). ``FlowCosmosDynamicTests`` — v0.33.0 D20
  Flow Cosmos Dynamic: 같은 값 셋을 같이 읽는다(따로 값 없음), ρ > 2 규칙(``flow_cosmos_dynamic_rho_ok``).
* ``FlowCosmosKnobTests`` — 확장 없이: 값 셋의 범위·기본값·자르기·늘 키 8개·옛 블록(인자 다섯)·``flow_cosmos_range_ok``.
  ``ExtensionControlsTests``(@requires_torch) — 확장 ``build_controls``(가짜 gradio 로 부른다)의 컴포넌트 8개 = 앱 상수
  (픽스처가 v0.33.0 으로 다시 녹화되기 전에도 확장 소스로 본다). ``FlowCosmosOracleTests``(@requires_torch) — 앱이 보내는
  블록을 확장 coerce_args → set_active → schedulers.flow_cosmos_rho7 에 그대로 넣는다: 기본값·옛 블록·블록 없음은 v0.33.0
  D18 목록(= GPU A/B custom 식)과 비트까지 같고, 다른 값은 그 수를 넣은 A/B 식과 같다; eps/v 는 블록의 ρ 로 Forge Karras.
  ``FlowCosmosDynamicOracleTests``(@requires_torch) — 같은 길로 schedulers.flow_cosmos_dynamic: 기본값·옛 블록 = 확장
  기본값의 목록, 다른 값 = 확장이 그 수로 만든 목록(같은 블록의 rho7 과 지수가 ρ 인 스텝에서 같다), eps/v = 확장 Karras
  Dynamic, ρ 규칙 = 확장 _check_karras_dynamic_rho.

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
from tests._optional_deps import load_torch, requires_torch
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
                  "1e9", " 2 ", "3.25", "-0", "abc", "", [], {}, (1,), 1e-320, -1e-320, 1e308, "1e400",
                  # v0.33.0: Flow Cosmos 범위 경계(ρ 1~15 · σ̃ max 1~1000 · σ̃ min 0.0001~1)와 float 로 못 바꾸는 큰 int(K1)
                  0.0001, 0.00009, 0.002, 7, 15, 15.5, 80, 1000, 1000.5, 10 ** 400)


def _raw_corpus() -> list:
    corpus: list = [None, {}]
    for value in _MODE_VALUES:
        corpus.append({} if value is _MISSING else {"custom_mode": value})
    for key in ("custom_expression", "custom_sigmas"):
        for value in _TEXT_VALUES:
            corpus.append({} if value is _MISSING else {key: value})
    for key in ("laplace_mu", "laplace_beta", "flow_cosmos_rho", "flow_cosmos_sigma_max", "flow_cosmos_sigma_min"):
        for value in _NUMBER_VALUES:
            corpus.append({} if value is _MISSING else {key: value})
    corpus.append({"custom_mode": "Sigma list", "custom_expression": " m ", "custom_sigmas": "[1, 0]",
                   "laplace_mu": "0.75", "laplace_beta": 3, "unknown": 1})
    corpus.append({"custom_mode": "expression", "laplace_mu": 1, "flow_cosmos_rho": "9.3",
                   "flow_cosmos_sigma_max": 240, "flow_cosmos_sigma_min": "0.006", "flow_cosmos": 3})
    corpus.append({"flow_cosmos_sigma_max": 1, "flow_cosmos_sigma_min": 1})    # 슬라이더 둘이 만나는 곳
    return corpus



def _fixture():
    with open(FIXTURE, encoding="utf-8") as fh:
        return json.load(fh)


class FixtureTests(unittest.TestCase):
    """녹화된 script-info(확장 ui())의 칸 = 앱 상수."""

    def test_both_tabs_have_the_custom_and_laplace_args_first_with_the_app_ranges_and_defaults(self):
        """앞 다섯 칸 — v0.31 부터 같은 자리다(확장은 인자를 뒤에만 덧붙인다)."""
        entries = _fixture_entries()
        self.assertEqual(sorted(e["is_img2img"] for e in entries), [False, True])
        for entry in entries:
            with self.subTest(img2img=entry["is_img2img"]):
                mode, expression, sigmas, mu, beta = entry["args"][:5]
                self.assertEqual(tuple(mode["choices"]), tuple(xs.MODE_LABELS[m] for m in xs.MODES))
                self.assertEqual(xs.coerce_mode(mode["value"]), xs.EXTENSION_DEFAULTS.custom_mode)
                self.assertEqual(expression["value"], xs.DEFAULT_EXPRESSION)
                self.assertEqual(sigmas["value"], xs.DEFAULT_SIGMAS)
                self.assertEqual((mu["minimum"], mu["maximum"], mu["step"], mu["value"]),
                                 (xs.LAPLACE_MU_MIN, xs.LAPLACE_MU_MAX, xs.LAPLACE_MU_STEP, xs.LAPLACE_MU_DEFAULT))
                self.assertEqual((beta["minimum"], beta["maximum"], beta["step"], beta["value"]),
                                 (xs.LAPLACE_BETA_MIN, xs.LAPLACE_BETA_MAX, xs.LAPLACE_BETA_STEP,
                                  xs.LAPLACE_BETA_DEFAULT))

    def test_v033_fixture_has_the_three_flow_cosmos_sliders_last(self):
        """v0.33.0 D19: 인자 8개 — 뒤 셋은 Flow Cosmos ρ·σ̃ max·σ̃ min 슬라이더(범위·step·기본값 = 앱 상수).

        픽스처는 계획 S3 2단계에서 v0.33.0 Forge(테스트 7862)로 다시 녹화했다(tools/refresh_sam_extra_fixture.py). 그 전의
        0.32.1 픽스처로는 실패했다 — ``ExtensionControlsTests`` 는 픽스처와 상관없이 확장 소스(build_controls)로 같은 값을 본다."""
        version = _fixture().get("ext_version")
        for entry in _fixture_entries():
            with self.subTest(img2img=entry["is_img2img"], fixture_ext_version=version):
                self.assertEqual(len(entry["args"]), len(xs.ARG_NAMES),
                                 f"픽스처가 sam-extra {version} 녹화(인자 {len(entry['args'])}개) — 계획 S3 2단계에서 v0.33.0 "
                                 "Forge 로 다시 녹화한다")
                rho, sigma_max, sigma_min = entry["args"][5:]
                for item, expected in ((rho, (xs.FLOW_COSMOS_RHO_MIN, xs.FLOW_COSMOS_RHO_MAX, xs.FLOW_COSMOS_RHO_STEP,
                                              xs.FLOW_COSMOS_RHO_DEFAULT)),
                                       (sigma_max, (xs.FLOW_COSMOS_SIGMA_MAX_MIN, xs.FLOW_COSMOS_SIGMA_MAX_MAX,
                                                    xs.FLOW_COSMOS_SIGMA_MAX_STEP, xs.FLOW_COSMOS_SIGMA_MAX_DEFAULT)),
                                       (sigma_min, (xs.FLOW_COSMOS_SIGMA_MIN_MIN, xs.FLOW_COSMOS_SIGMA_MIN_MAX,
                                                    xs.FLOW_COSMOS_SIGMA_MIN_STEP, xs.FLOW_COSMOS_SIGMA_MIN_DEFAULT))):
                    self.assertEqual((item["minimum"], item["maximum"], item["step"], item["value"]), expected)

    def test_every_live_mode_label_coerces_to_its_key_in_extension_order(self):
        labels = _fixture_entries()[0]["args"][0]["choices"]
        self.assertEqual(tuple(xs.coerce_mode(label) for label in labels), xs.MODES)

    def test_app_defaults_equal_the_extension_defaults(self):
        self.assertIs(xs.APP_DEFAULTS, xs.EXTENSION_DEFAULTS)
        self.assertEqual(xs.APP_DEFAULTS, xs.ExtraSchedulerSettings(
            "expression", "M * (m / M) ** x", "[1.0, 0.6, 0.25, 0.1, 0.0]", 0.0, 0.5, 7.0, 80.0, 0.002))


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
        for name in ("KEY_CUSTOM_EXPRESSION", "KEY_CUSTOM_SIGMAS", "KEY_LAPLACE_MU", "KEY_LAPLACE_BETA",
                     "KEY_FLOW_COSMOS_RHO", "KEY_FLOW_COSMOS_SIGMA_MAX", "KEY_FLOW_COSMOS_SIGMA_MIN"):
            with self.subTest(key=name):
                self.assertEqual(_ext_constant(self.UI, name), getattr(xs, name))
        self.assertEqual(xs.ARG_NAMES[5:], ("flow_cosmos_rho", "flow_cosmos_sigma_max", "flow_cosmos_sigma_min"))

    def test_settings_defaults_and_ranges(self):
        for name in ("MODE_EXPRESSION", "MODE_SIGMAS", "DEFAULT_EXPRESSION", "DEFAULT_SIGMAS"):
            with self.subTest(name=name):
                self.assertEqual(_ext_constant(self.SETTINGS, name), getattr(xs, name))
        settings = _ext("extra_schedulers.settings")
        self.assertEqual(settings.MODES, xs.MODES)
        for name in ("LAPLACE_MU_MIN", "LAPLACE_MU_MAX", "LAPLACE_MU_DEFAULT", "LAPLACE_BETA_MIN",
                     "LAPLACE_BETA_MAX", "LAPLACE_BETA_DEFAULT",
                     # v0.33.0 D19 Flow Cosmos (최소·최대·기본·슬라이더 step)
                     "FLOW_COSMOS_RHO_MIN", "FLOW_COSMOS_RHO_MAX", "FLOW_COSMOS_RHO_DEFAULT", "FLOW_COSMOS_RHO_STEP",
                     "FLOW_COSMOS_SIGMA_MAX_MIN", "FLOW_COSMOS_SIGMA_MAX_MAX", "FLOW_COSMOS_SIGMA_MAX_DEFAULT",
                     "FLOW_COSMOS_SIGMA_MAX_STEP", "FLOW_COSMOS_SIGMA_MIN_MIN", "FLOW_COSMOS_SIGMA_MIN_MAX",
                     "FLOW_COSMOS_SIGMA_MIN_DEFAULT", "FLOW_COSMOS_SIGMA_MIN_STEP"):
            with self.subTest(name=name):
                self.assertEqual(_key(getattr(settings, name)), _key(getattr(xs, name)))
        self.assertEqual(_fields(settings.DEFAULTS), _fields(xs.EXTENSION_DEFAULTS))
        # 기본값 = schedulers.py 리터럴(D18 의 고정 목록 — 레지스트리 핀 xsched_cosmos_* 가 그 리터럴을 잡는다)
        for default, literal in (("FLOW_COSMOS_RHO_DEFAULT", "FLOW_COSMOS_RHO"),
                                 ("FLOW_COSMOS_SIGMA_MAX_DEFAULT", "FLOW_COSMOS_SIGMA_MAX"),
                                 ("FLOW_COSMOS_SIGMA_MIN_DEFAULT", "FLOW_COSMOS_SIGMA_MIN")):
            with self.subTest(default=default):
                self.assertEqual(_key(getattr(settings, default)),
                                 _key(_ext_constant("sam3ext/extra_schedulers/schedulers.py", literal)))
                self.assertEqual(_key(getattr(xs, default)), _key(getattr(xs, literal)))

    def test_scheduler_labels_and_readme_aliases(self):
        self.assertEqual(_ext_constant(self.REGISTRY, "README_LABELS"), dict(xs.README_LABELS))
        self.assertEqual((_ext_constant(self.REGISTRY, "LABEL_LAPLACE"), _ext_constant(self.REGISTRY, "LABEL_CUSTOM")),
                         (xs.LABEL_LAPLACE, xs.LABEL_CUSTOM))
        # v0.33.0: React Cosinusoidal DynSF (no README alias, fixed factor)
        self.assertEqual(_ext_constant(self.REGISTRY, "LABEL_REACT_DYNSF"), xs.LABEL_REACT_DYNSF)
        self.assertEqual(repr(_ext_constant("sam3ext/extra_schedulers/schedulers.py", "REACT_DYNSF_FACTOR")),
                         repr(xs.REACT_DYNSF_FACTOR))
        self.assertNotIn(xs.NAME_REACT_DYNSF, xs.README_LABELS)
        # v0.33.0: Flow Cosmos rho7 (no README alias; Cosmos-Predict2's EDM range and Karras's rho as module literals)
        self.assertEqual(_ext_constant(self.REGISTRY, "LABEL_FLOW_COSMOS_RHO7"), xs.LABEL_FLOW_COSMOS_RHO7)
        for name in ("FLOW_COSMOS_SIGMA_MAX", "FLOW_COSMOS_SIGMA_MIN", "FLOW_COSMOS_RHO"):
            with self.subTest(name=name):
                self.assertEqual(_key(_ext_constant("sam3ext/extra_schedulers/schedulers.py", name)),
                                 _key(getattr(xs, name)))
        self.assertEqual((xs.FLOW_COSMOS_SIGMA_MAX, xs.FLOW_COSMOS_SIGMA_MIN, xs.FLOW_COSMOS_RHO), (80.0, 0.002, 7.0))
        self.assertNotIn(xs.NAME_FLOW_COSMOS_RHO7, xs.README_LABELS)
        # v0.33.0 D20: Flow Cosmos Dynamic (no README alias, no values of its own — Flow Cosmos rho7's three; ρ above 2)
        self.assertEqual(_ext_constant(self.REGISTRY, "LABEL_FLOW_COSMOS_DYNAMIC"), xs.LABEL_FLOW_COSMOS_DYNAMIC)
        self.assertEqual(_ext_constant("sam3ext/extra_schedulers/schedulers.py", "FLOW_COSMOS_DYNAMIC"),
                         xs.LABEL_FLOW_COSMOS_DYNAMIC)
        self.assertNotIn(xs.NAME_FLOW_COSMOS_DYNAMIC, xs.README_LABELS)
        source = (EXT_ROOT / "sam3ext/extra_schedulers/schedulers.py").read_text(encoding="utf-8")
        self.assertGreaterEqual(source.count(f'f"{xs.ERROR_PREFIX} '), 8, "스케줄러 오류 문구의 머리(ERROR_PREFIX)")


class _Control:
    """가짜 gradio 컴포넌트 — 만든 인자를 그대로 담는다(``with`` 블록도 된다)."""

    def __init__(self, kind, *args, **kwargs):
        self.kind, self.args, self.kwargs = kind, args, kwargs

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def _fake_gradio() -> types.ModuleType:
    module = types.ModuleType("gradio")
    for kind in ("Accordion", "Markdown", "Radio", "Textbox", "Row", "Slider"):
        setattr(module, kind, lambda *args, _kind=kind, **kwargs: _Control(_kind, *args, **kwargs))
    return module


@requires_extension
@requires_torch
class ExtensionControlsTests(unittest.TestCase):
    """확장 Gradio 컨트롤(ui_extra_schedulers.build_controls — script-info 가 녹화하는 바로 그것) = 앱 상수. gradio 대신
    인자를 담는 가짜 모듈로 부른다(확장 코드는 읽기만). 픽스처가 v0.33.0 으로 다시 녹화되기 전에도 슬라이더 다섯을 본다."""

    @classmethod
    def setUpClass(cls):
        cls.ui = _ext("ui_extra_schedulers")
        with mock.patch.dict(sys.modules, {"gradio": _fake_gradio()}):
            cls.controls = tuple(cls.ui.build_controls(lambda name: name))

    def test_eight_controls_in_argument_order(self):
        self.assertEqual([control.kwargs["elem_id"] for control in self.controls], list(xs.ARG_NAMES))
        self.assertEqual(tuple(self.ui.ARG_NAMES), xs.ARG_NAMES)
        self.assertEqual(tuple(self.ui.INFOTEXT_KEYS), xs.INFOTEXT_KEYS)
        mode, expression, sigmas = self.controls[:3]
        self.assertEqual((mode.kind, tuple(mode.kwargs["choices"])),
                         ("Radio", tuple(xs.MODE_LABELS[m] for m in xs.MODES)))
        self.assertEqual(xs.coerce_mode(mode.kwargs["value"]), xs.EXTENSION_DEFAULTS.custom_mode)
        self.assertEqual((expression.kind, expression.kwargs["value"]), ("Textbox", xs.DEFAULT_EXPRESSION))
        self.assertEqual((sigmas.kind, sigmas.kwargs["value"]), ("Textbox", xs.DEFAULT_SIGMAS))

    def test_sliders_are_the_app_ranges_steps_and_defaults(self):
        expected = {
            "laplace_mu": (xs.LAPLACE_MU_MIN, xs.LAPLACE_MU_MAX, xs.LAPLACE_MU_STEP, xs.LAPLACE_MU_DEFAULT),
            "laplace_beta": (xs.LAPLACE_BETA_MIN, xs.LAPLACE_BETA_MAX, xs.LAPLACE_BETA_STEP, xs.LAPLACE_BETA_DEFAULT),
            "flow_cosmos_rho": (xs.FLOW_COSMOS_RHO_MIN, xs.FLOW_COSMOS_RHO_MAX, xs.FLOW_COSMOS_RHO_STEP,
                                xs.FLOW_COSMOS_RHO_DEFAULT),
            "flow_cosmos_sigma_max": (xs.FLOW_COSMOS_SIGMA_MAX_MIN, xs.FLOW_COSMOS_SIGMA_MAX_MAX,
                                      xs.FLOW_COSMOS_SIGMA_MAX_STEP, xs.FLOW_COSMOS_SIGMA_MAX_DEFAULT),
            "flow_cosmos_sigma_min": (xs.FLOW_COSMOS_SIGMA_MIN_MIN, xs.FLOW_COSMOS_SIGMA_MIN_MAX,
                                      xs.FLOW_COSMOS_SIGMA_MIN_STEP, xs.FLOW_COSMOS_SIGMA_MIN_DEFAULT),
        }
        sliders = {control.kwargs["elem_id"]: control for control in self.controls if control.kind == "Slider"}
        self.assertEqual(list(sliders), list(expected))
        for name, values in expected.items():
            with self.subTest(name=name):
                kwargs = sliders[name].kwargs
                self.assertEqual(tuple(_key(kwargs[field]) for field in ("minimum", "maximum", "step", "value")),
                                 tuple(_key(value) for value in values))
                self.assertEqual(_key(getattr(xs.EXTENSION_DEFAULTS, name)), _key(values[3]))


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

    def test_huge_int_reads_as_the_default_like_the_extension(self):
        """float 로 못 바꾸는 큰 int(JSON 이 만들 수 있다): v0.33.0 확장도 그 칸만 기본값(D19 검토 K1 — 0.32.x 는
        OverflowError 로 process() 가 요청 전체를 기본값으로 돌렸다). 이제 차이가 없다."""
        for key in ("laplace_mu", "laplace_beta", "flow_cosmos_rho", "flow_cosmos_sigma_max", "flow_cosmos_sigma_min"):
            for value in (10 ** 400, -(10 ** 400)):
                with self.subTest(key=key, sign=value > 0):
                    raw = {key: value, "laplace_beta": 2.0} if key != "laplace_beta" else {key: value, "laplace_mu": 2.0}
                    self.assertEqual(self._ext_outcome(raw), ("ok", _fields(xs.coerce(raw))))
                    self.assertEqual(getattr(xs.coerce(raw), key), getattr(xs.EXTENSION_DEFAULTS, key))

    def test_individual_coercers_match(self):
        for value in _NUMBER_VALUES:
            if value is _MISSING:
                continue
            with self.subTest(value=repr(value)):
                for name in ("coerce_laplace_mu", "coerce_laplace_beta", "coerce_flow_cosmos_rho",
                             "coerce_flow_cosmos_sigma_max", "coerce_flow_cosmos_sigma_min"):
                    self.assertEqual(_key(getattr(xs, name)(value)), _key(getattr(self.settings, name)(value)), name)
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
                 [{}, "ignored"], [{"custom_expression": None, "laplace_mu": "-2.25"}],
                 # v0.33.0: 인자 8개(뒤 셋 Flow Cosmos), 6·7개, 옛 다섯, 9개(남는 것은 버린다), dict 의 Flow Cosmos 키
                 ["expression", "M", "[1, 0]", 0.5, 1.5, 5.5, 240, "0.006"], ["sigmas", "M", "[1, 0]", 0, 0.5, "20"],
                 ["expression", "M", "[1, 0]", 0, 0.5, 7, "nan", -1], ["expression", "M", "[1, 0]", 0, 0.5, 7, 80, 0.002, 9],
                 [{"flow_cosmos_rho": "9.3", "flow_cosmos_sigma_max": 1, "flow_cosmos_sigma_min": 1}],
                 [{"flow_cosmos_rho": True, "flow_cosmos_sigma_max": "inf", "flow_cosmos_sigma_min": 10 ** 400}]]
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
                    '"custom_sigmas": "[1.0, 0.6, 0.25, 0.1, 0.0]", "laplace_mu": 0.0, "laplace_beta": 0.5, '
                    '"flow_cosmos_rho": 7.0, "flow_cosmos_sigma_max": 80.0, "flow_cosmos_sigma_min": 0.002}]}')
        self.assertEqual(json.dumps(xs.as_block(xs.APP_DEFAULTS)), expected)
        self.assertEqual(tuple(xs.as_block(xs.APP_DEFAULTS)["args"][0]), xs.ARG_NAMES)

    def test_block_round_trips_through_the_extension_rules(self):
        for settings in (xs.APP_DEFAULTS,
                         xs.ExtraSchedulerSettings("sigmas", "m + (M - m) * (1 - x) ** 3", "[1, 0.5, 0]", -2.5, 3.0),
                         xs.ExtraSchedulerSettings("expression", "", "", 10.0, 0.0),
                         xs.ExtraSchedulerSettings("expression", "M", "[1, 0]", 0.0, 0.5, 1 / 3 + 5, 1000.0, 0.0001),
                         xs.ExtraSchedulerSettings(flow_cosmos_rho=15.0, flow_cosmos_sigma_max=1.0,
                                                   flow_cosmos_sigma_min=1.0)):
            with self.subTest(settings=settings):
                block = xs.as_block(settings)
                self.assertEqual(len(block["args"]), 1)
                for key in ("laplace_mu", "laplace_beta", "flow_cosmos_rho", "flow_cosmos_sigma_max",
                            "flow_cosmos_sigma_min"):
                    self.assertIsInstance(block["args"][0][key], float)
                self.assertEqual(xs.parse_script_block(block), settings)
                self.assertEqual(xs.parse_script_block(json.loads(json.dumps(block))), settings)

    def test_positional_and_partial_blocks(self):
        self.assertEqual(xs.parse_script_block({"args": ["Sigma list", "x", "[1, 0]"]}),
                         xs.ExtraSchedulerSettings("sigmas", "x", "[1, 0]", 0.0, 0.5))
        self.assertEqual(xs.parse_script_block({"args": ["Sigma list", "x", "[1, 0]", 1, 2, 3, 4, 0.5]}),
                         xs.ExtraSchedulerSettings("sigmas", "x", "[1, 0]", 1.0, 2.0, 3.0, 4.0, 0.5))
        self.assertEqual(xs.parse_script_block({"args": []}), xs.EXTENSION_DEFAULTS)
        self.assertEqual(xs.parse_script_block({}), xs.EXTENSION_DEFAULTS)
        self.assertIsNone(xs.parse_script_block(None))
        self.assertIsNone(xs.parse_script_block(["expression"]))

    def test_widget_values_round_trip(self):
        for settings in (xs.APP_DEFAULTS, xs.ExtraSchedulerSettings("sigmas", " [1, 0] ", "[0.9, 0.0]", 1 / 3, 9.9),
                         xs.ExtraSchedulerSettings("expression", "M", "", -10.0, 10.0),
                         xs.ExtraSchedulerSettings("expression", "M", "", 0.0, 0.5, 9.3, 1 / 3 * 1000, 0.00015)):
            with self.subTest(settings=settings):
                values = xs.widget_values(settings)
                self.assertEqual(set(values), set(xs.WIDGET_KEYS))
                self.assertTrue(all(isinstance(v, str) for v in values.values()))
                parsed = xs.parse_settings(values)
                self.assertEqual(parsed, replace(settings, custom_expression=settings.custom_expression.strip(),
                                                 custom_sigmas=settings.custom_sigmas.strip()))
        self.assertEqual(xs.widget_values(xs.APP_DEFAULTS)["laplace_mu"], "0")
        self.assertEqual(xs.widget_values(xs.APP_DEFAULTS)["laplace_beta"], "0.5")
        self.assertEqual({key: xs.widget_values(xs.APP_DEFAULTS)[key] for key in xs.ARG_NAMES[5:]},
                         {"flow_cosmos_rho": "7", "flow_cosmos_sigma_max": "80", "flow_cosmos_sigma_min": "0.002"})
        self.assertEqual(xs.widget_number(0.0001), "0.0001")
        self.assertEqual(xs.widget_number(2.0), "2")
        self.assertEqual(xs.widget_number(-0.0), "0")

    def test_parse_settings_keeps_base_for_missing_and_unreadable(self):
        base = xs.ExtraSchedulerSettings("sigmas", "x", "[1, 0]", 2.0, 1.5, 5.0, 240.0, 0.006)
        self.assertEqual(xs.parse_settings({}, base=base), base)
        self.assertEqual(xs.parse_settings(None, base=base), base)
        self.assertEqual(xs.parse_settings({"laplace_mu": "abc", "laplace_beta": "", "custom_mode": None,
                                            "flow_cosmos_rho": "nan", "flow_cosmos_sigma_max": None,
                                            "flow_cosmos_sigma_min": True}, base=base),
                         base)
        self.assertEqual(xs.parse_settings({"flow_cosmos_rho": "0.5", "flow_cosmos_sigma_max": "2e3",
                                            "flow_cosmos_sigma_min": 0}, base=base),
                         replace(base, flow_cosmos_rho=1.0, flow_cosmos_sigma_max=1000.0, flow_cosmos_sigma_min=0.0001))
        self.assertEqual(xs.parse_settings({"laplace_mu": "12", "custom_expression": "  M  "}, base=base),
                         replace(base, laplace_mu=10.0, custom_expression="M"))
        self.assertEqual(xs.parse_settings({"custom_sigmas": ""}, base=base).custom_sigmas, "",
                         "빈 목록은 값이다 — 확장처럼 그대로 보내고 생성할 때 확장이 거절한다")

    def test_widget_ids(self):
        self.assertEqual([xs.widget_id(key) for key in xs.WIDGET_KEYS],
                         ["_xsched_custom_mode", "_xsched_custom_expression", "_xsched_custom_sigmas",
                          "_xsched_laplace_mu", "_xsched_laplace_beta", "_xsched_flow_cosmos_rho",
                          "_xsched_flow_cosmos_sigma_max", "_xsched_flow_cosmos_sigma_min"])
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

    def test_flow_cosmos_values_are_user_values_and_its_passes_count(self):
        user = xs.ExtraSchedulerSettings(flow_cosmos_sigma_max=240.0, flow_cosmos_sigma_min=0.006)
        confirmed = xs.plan(user, target="t2i", backend="webui", capabilities=self.CONFIRMED)
        self.assertEqual((confirmed.reason, confirmed.provenance), (xs.REASON_SEND, xs.PROVENANCE_USER))
        self.assertEqual(confirmed.block["args"][0]["flow_cosmos_sigma_max"], 240.0)
        passes = xs.passes_from_widgets({"scheduler_combo": xs.NAME_FLOW_COSMOS_RHO7})
        self.assertEqual(passes, (xs.PassUse("main", xs.NAME_FLOW_COSMOS_RHO7),))
        used = xs.plan(user, target="t2i", backend="webui", capabilities=None, passes=passes)
        self.assertEqual((used.reason, used.provenance, used.block), (xs.REASON_SEND_UNCONFIRMED, xs.PROVENANCE_USER,
                                                                      xs.as_block(user)))

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
                         "custom 식 'M * (m / M) ** x' · Laplace μ 0 · β 0.5 · Flow Cosmos ρ 7 · σ̃ 0.002~80")
        cosmos = xs.ExtraSchedulerSettings(flow_cosmos_rho=5.5, flow_cosmos_sigma_max=240.0, flow_cosmos_sigma_min=0.006)
        self.assertEqual(xs.describe(cosmos, ["Flow Cosmos rho7"]), "Flow Cosmos ρ 5.5 · σ̃ 0.006~240")
        self.assertEqual(xs.describe(cosmos, ["flow_cosmos_rho7", "laplace"]),
                         "Laplace μ 0 · β 0.5 · Flow Cosmos ρ 5.5 · σ̃ 0.006~240")
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
        # 값 스케줄러 = 블록의 값을 읽는 넷(v0.33.0 D19 부터 Flow Cosmos rho7, D20 의 Dynamic 도 — ρ·σ̃ max·σ̃ min)
        self.assertEqual(set(xs.VALUE_SCHEDULERS), {s.name for s in specs if s.label in (
            self.registry.LABEL_LAPLACE, self.registry.LABEL_CUSTOM, *self.registry.FLOW_COSMOS_LABELS)})
        self.assertEqual(xs.VALUE_SCHEDULERS, (xs.NAME_LAPLACE, xs.NAME_CUSTOM, xs.NAME_FLOW_COSMOS_RHO7,
                                               xs.NAME_FLOW_COSMOS_DYNAMIC))
        self.assertEqual(xs.FLOW_COSMOS_LABELS, tuple(self.registry.FLOW_COSMOS_LABELS))
        self.assertEqual(xs.LABEL_FLOW_COSMOS_DYNAMIC, self.registry.LABEL_FLOW_COSMOS_DYNAMIC)
        # every scheduler is flow-unsuitable (card hint), a value scheduler or React DynSF (its own hint, no values)
        self.assertEqual(set(xs.FLOW_UNSUITABLE) | set(xs.VALUE_SCHEDULERS) | {xs.NAME_REACT_DYNSF},
                         set(xs.SCHEDULER_NAMES))
        self.assertEqual(len(xs.SCHEDULER_NAMES), 9)
        self.assertEqual(xs.SCHEDULER_NAMES[-3:], (xs.NAME_REACT_DYNSF, xs.NAME_FLOW_COSMOS_RHO7,
                                                   xs.NAME_FLOW_COSMOS_DYNAMIC))
        self.assertNotIn(xs.NAME_REACT_DYNSF, xs.FLOW_UNSUITABLE + xs.VALUE_SCHEDULERS)
        self.assertNotIn(xs.NAME_FLOW_COSMOS_RHO7, xs.FLOW_UNSUITABLE)
        self.assertNotIn(xs.NAME_FLOW_COSMOS_DYNAMIC, xs.FLOW_UNSUITABLE)

    def test_every_label_and_alias_picks_its_name_in_a_forge_combo(self):
        """붙여 넣기·프리셋 경로(core.preset_choice.match_scheduler — F3): infotext 'Schedule type' 의 라벨·README 별칭이
        이름 값 Forge 콤보에서 그 이름을 고른다. tests/test_preset_choice 의 AST 스캔은 라벨이 상수 이름인 SchedulerSpec
        (React DynSF·Flow Cosmos rho7·Dynamic)을 건너뛰므로 여기서 확장 표 전체로 본다."""
        from core.preset_choice import match_scheduler
        specs = self.registry.SCHEDULER_SPECS
        combo = ["automatic", "uniform", "karras", "exponential", "polyexponential", "sgm_uniform", "kl_optimal",
                 "align_your_steps", "simple", "normal", "ddim", "beta", "turbo", "linear_quadratic",
                 *(spec.name for spec in specs)]
        for spec in specs:
            for text in (spec.name, spec.label, *spec.aliases):
                with self.subTest(text=text):
                    index = match_scheduler(text, combo)
                    self.assertEqual(combo[index] if index >= 0 else None, spec.name)


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
    # v0.33.0 React Cosinusoidal DynSF — no values of its own; other keys still paste
    {"Schedule type": "React Cosinusoidal DynSF"},
    {"Schedule type": "react_cosinusoidal_dynsf", "Laplace mu": 1.5},
    {"Schedule type": "React Cosinusoidal DynSF", "Hires schedule type": "Laplace"},
    {"Schedule type": "REACT COSINUSOIDAL DYNSF", "Custom scheduler expression": "x"},
    # v0.33.0 Flow Cosmos rho7 — no values of its own (fixed rho: a pasted 'Schedule rho' is not ours); other keys still paste
    {"Schedule type": "Flow Cosmos rho7"},
    {"Schedule type": "flow_cosmos_rho7", "Laplace beta": 2},
    {"Schedule type": "Flow Cosmos rho7", "Hires schedule type": "Laplace"},
    {"Schedule type": "Laplace", "Hires schedule type": "Flow Cosmos rho7", "Laplace mu": "-1.5"},
    {"Schedule type": "FLOW COSMOS RHO7", "Custom scheduler sigmas": "[1, 0]"},
    {"Schedule type": "Flow Cosmos rho7", "Schedule rho": "3", "Hires schedule type": "Use same scheduler"},
    # v0.33.0 D19 Flow Cosmos rho7 values — keys like Laplace's (key -> coerced, label without keys -> defaults)
    {"Schedule type": "Flow Cosmos rho7", "Flow Cosmos rho": 5.5, "Flow Cosmos sigma max": "240.0",
     "Flow Cosmos sigma min": "0.006"},
    {"Schedule type": "Flow Cosmos rho7", "Flow Cosmos rho": "0.5", "Flow Cosmos sigma max": "1e4",
     "Flow Cosmos sigma min": "0"},
    {"Schedule type": "flow_cosmos_rho7", "Flow Cosmos sigma min": "nan"},
    {"Schedule type": "Flow Cosmos rho7", "Flow Cosmos rho": True, "Flow Cosmos sigma max": None},
    {"Schedule type": "Karras", "Flow Cosmos rho": "3"},
    {"Schedule type": "Karras", "Hires schedule type": "Flow Cosmos rho7", "Flow Cosmos sigma max": 1000},
    {"Schedule type": "Laplace", "Hires schedule type": "Flow Cosmos rho7", "Laplace mu": "-1.5",
     "Flow Cosmos rho": "9.3", "Flow Cosmos sigma min": "1"},
    {"Schedule type": "FLOW COSMOS RHO7", "Flow Cosmos rho": "2", "Schedule rho": "4"},
    {"Schedule type": "Flow Cosmos rho7", "Flow Cosmos rho": "1e400", "Flow Cosmos sigma max": "-5",
     "Flow Cosmos sigma min": " 0.25 "},
    # v0.33.0 D20 Flow Cosmos Dynamic — the same three keys (registry.FLOW_COSMOS_LABELS): label without keys -> defaults
    {"Schedule type": "Flow Cosmos Dynamic"},
    {"Schedule type": "flow_cosmos_dynamic", "Flow Cosmos rho": "5.5", "Flow Cosmos sigma max": 240,
     "Flow Cosmos sigma min": "0.006"},
    {"Schedule type": "FLOW COSMOS DYNAMIC", "Flow Cosmos rho": "1.5"},
    {"Schedule type": "Karras", "Hires schedule type": "Flow Cosmos Dynamic"},
    {"Schedule type": "Flow Cosmos Dynamic", "Hires schedule type": "Flow Cosmos rho7", "Flow Cosmos sigma min": "nan"},
    {"Schedule type": "Laplace", "Hires schedule type": "flow_cosmos_dynamic", "Laplace beta": "3",
     "Schedule rho": "4"},
    {"Schedule type": "Karras Dynamic", "Flow Cosmos rho": "9"},
    {"Schedule type": "Flow Cosmos Dynamic", "Hires schedule type": "Use same scheduler", "Flow Cosmos rho": True},
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
                      "laplace_beta": self.ui.paste_laplace_beta(params),
                      "flow_cosmos_rho": self.ui.paste_flow_cosmos_rho(params),
                      "flow_cosmos_sigma_max": self.ui.paste_flow_cosmos_sigma_max(params),
                      "flow_cosmos_sigma_min": self.ui.paste_flow_cosmos_sigma_min(params)}
        fields.update({key: value for key, value in pasted.items() if value is not None})
        return self.settings.coerce(fields)

    def test_table(self):
        bases = (xs.APP_DEFAULTS, xs.ExtraSchedulerSettings("sigmas", "M * 0.5", "[2, 1, 0]", -1.0, 4.0, 3.5, 500.0,
                                                            0.01))
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

    def test_generate_from_metadata_restore(self):
        """계획 A-R1(X3b) — ``restore_from_infotext``: 메인·Hires 가 값 스케줄러면 그 스케줄러의 칸만 확장 paste_* 의 값
        (접은 표기는 확장 라벨로 바꾼 붙여 넣기 — 앱 콤보가 고른 스케줄러와 같게), 다른 칸은 base 그대로(키가 있어도),
        값 스케줄러가 없으면 None."""
        bases = (xs.APP_DEFAULTS, xs.ExtraSchedulerSettings("sigmas", "M * 0.5", "[2, 1, 0]", -1.0, 4.0, 3.5, 500.0,
                                                            0.01))
        keys = (xs.KEY_SCHEDULE_TYPE, xs.KEY_HIRES_SCHEDULE_TYPE)
        seen = {"none": 0, "partial": 0}
        for base in bases:
            for params in _INFOTEXT_CASES:
                with self.subTest(base=base, params=params):
                    names = {xs.resolve_scheduler(params.get(key)) for key in keys}
                    used = names & set(xs.VALUE_SCHEDULERS)
                    restored = xs.restore_from_infotext(params, base=base)
                    if not used:
                        seen["none"] += 1
                        self.assertIsNone(restored)
                        continue
                    labelled = dict(params)
                    for key in keys:
                        name = xs.resolve_scheduler(params.get(key))
                        if name is not None:
                            labelled[key] = xs.SCHEDULER_LABELS[name]
                    pasted = self._ext_paste(labelled, base)
                    fields = {field for name in used for field in xs.RESTORE_FIELDS[name]}
                    seen["partial"] += len(fields) < len(xs.ARG_NAMES)
                    want = tuple(_key(getattr(pasted if name in fields else base, name)) for name in xs.ARG_NAMES)
                    self.assertEqual(_fields(restored), want)
        self.assertGreater(seen["none"], 10)
        self.assertGreater(seen["partial"], 20)


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
                    xs.ExtraSchedulerSettings("expression", "  m +   (M - m) * x ", "[1, 0]", 10.0, 0.0),
                    xs.ExtraSchedulerSettings("expression", "M", "[1, 0]", 0.0, 0.5, 1 / 3, 240.0, 0.006),
                    xs.ExtraSchedulerSettings("expression", "M", "[1, 0]", 0.0, 0.5, 15.0, 1000.0, 0.0001))
        for settings in variants:
            current = self.settings.ExtraSchedulerSettings(**{n: getattr(settings, n) for n in xs.ARG_NAMES})
            for labels in ((), ("Laplace",), ("custom",), ("Laplace", "custom"), ("Karras",), ("Cosine", "Laplace"),
                           ("React Cosinusoidal DynSF",), ("React Cosinusoidal DynSF", "custom"),
                           ("Flow Cosmos rho7",), ("Flow Cosmos rho7", "Laplace"),
                           ("Flow Cosmos rho7", "React Cosinusoidal DynSF"), ("Flow Cosmos rho7", "custom", "Laplace"),
                           ("Flow Cosmos Dynamic",), ("Flow Cosmos Dynamic", "Flow Cosmos rho7"),
                           ("Flow Cosmos Dynamic", "Laplace", "custom"), ("Karras Dynamic", "Flow Cosmos Dynamic")):
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
        cosmos = xs.ExtraSchedulerSettings(flow_cosmos_rho=1 / 3 + 5, flow_cosmos_sigma_max=240.0,
                                           flow_cosmos_sigma_min=0.006)
        params = {"Schedule type": "Flow Cosmos rho7", **xs.infotext_items(cosmos, ["Flow Cosmos rho7"])}
        self.assertEqual(xs.from_infotext(params, base=xs.ExtraSchedulerSettings(laplace_mu=2.0)),
                         replace(cosmos, laplace_mu=2.0))
        params = {"Schedule type": "Flow Cosmos Dynamic", **xs.infotext_items(cosmos, ["flow_cosmos_dynamic"])}
        self.assertEqual(xs.from_infotext(params, base=xs.APP_DEFAULTS), cosmos)


class ReactDynSFTests(unittest.TestCase):
    """v0.33.0 React Cosinusoidal DynSF — 이름·표기가 풀리고, 값이 없어 블록·패스·infotext 에 아무것도 더하지 않는다."""

    def test_resolves_from_name_label_and_folded_spellings(self):
        for text in ("react_cosinusoidal_dynsf", "React Cosinusoidal DynSF", "REACT COSINUSOIDAL DYNSF",
                     "  react   cosinusoidal dynsf ", "React_Cosinusoidal_DynSF"):
            with self.subTest(text=text):
                self.assertEqual(xs.resolve_scheduler(text), xs.NAME_REACT_DYNSF)
        self.assertEqual(xs.SCHEDULER_LABELS[xs.NAME_REACT_DYNSF], xs.LABEL_REACT_DYNSF)
        for text in ("React Cosinusoidal", "react-cosinusoidal-dynsf", "React DynSF"):
            self.assertIsNone(xs.resolve_scheduler(text), text)

    def test_reads_no_values(self):
        user = xs.ExtraSchedulerSettings("sigmas", "M", "[1, 0]", 2.0, 3.0)
        self.assertEqual(xs.describe(user, [xs.LABEL_REACT_DYNSF]), "")
        self.assertEqual(xs.infotext_items(user, [xs.LABEL_REACT_DYNSF, xs.NAME_REACT_DYNSF]), {})
        self.assertEqual(xs.value_scheduler_passes({"scheduler": xs.LABEL_REACT_DYNSF, "enable_hr": True,
                                                    "hr_scheduler": xs.NAME_REACT_DYNSF}), ())
        self.assertEqual(xs.passes_from_widgets({"scheduler_combo": xs.NAME_REACT_DYNSF}), ())
        self.assertEqual(xs.infotext_labels({"Schedule type": xs.NAME_REACT_DYNSF}), frozenset({xs.LABEL_REACT_DYNSF}))
        self.assertEqual(xs.from_infotext({"Schedule type": xs.LABEL_REACT_DYNSF}, base=user), user)


class FlowCosmosRho7Tests(unittest.TestCase):
    """v0.33.0 Flow Cosmos rho7 — 이름·표기가 풀리고, D19 부터 값 셋(ρ·σ̃ max·σ̃ min)을 읽는 값 스케줄러다(패스·infotext·
    붙여 넣기 — Laplace 와 같은 규칙). 붙여 넣은 'Schedule type: Flow Cosmos rho7' 은 Forge 콤보(이름 값)에서
    flow_cosmos_rho7 을 고른다."""

    def test_name_and_label(self):
        self.assertEqual((xs.NAME_FLOW_COSMOS_RHO7, xs.LABEL_FLOW_COSMOS_RHO7), ("flow_cosmos_rho7", "Flow Cosmos rho7"))
        self.assertEqual(xs.SCHEDULER_LABELS[xs.NAME_FLOW_COSMOS_RHO7], xs.LABEL_FLOW_COSMOS_RHO7)
        self.assertEqual(xs.SCHEDULER_NAMES[-2:], (xs.NAME_FLOW_COSMOS_RHO7, xs.NAME_FLOW_COSMOS_DYNAMIC))

    def test_resolves_from_name_label_and_folded_spellings(self):
        for text in ("flow_cosmos_rho7", "Flow Cosmos rho7", "FLOW COSMOS RHO7", "  flow   cosmos rho7 ",
                     "Flow_Cosmos_Rho7", "FLOW_COSMOS_RHO7"):
            with self.subTest(text=text):
                self.assertEqual(xs.resolve_scheduler(text), xs.NAME_FLOW_COSMOS_RHO7)
        for text in ("Flow Cosmos", "flow cosmos rho 7", "flow-cosmos-rho7", "Cosmos rho7", "Flow Cosmos \u03c17",
                     "flow_cosmos_rho", "Cosmos \u03c17", "karras"):
            self.assertIsNone(xs.resolve_scheduler(text), text)

    def test_reads_its_three_values(self):
        user = xs.ExtraSchedulerSettings("sigmas", "M", "[1, 0]", 2.0, 3.0, 5.5, 240.0, 0.006)
        cosmos_items = {xs.KEY_FLOW_COSMOS_RHO: "5.5", xs.KEY_FLOW_COSMOS_SIGMA_MAX: "240.0",
                        xs.KEY_FLOW_COSMOS_SIGMA_MIN: "0.006"}
        self.assertEqual(xs.describe(user, [xs.LABEL_FLOW_COSMOS_RHO7]), "Flow Cosmos ρ 5.5 · σ̃ 0.006~240")
        self.assertEqual(xs.infotext_items(user, [xs.LABEL_FLOW_COSMOS_RHO7, xs.NAME_FLOW_COSMOS_RHO7]), cosmos_items)
        self.assertEqual(xs.infotext_items(user, [xs.LABEL_FLOW_COSMOS_RHO7, "Laplace"]),
                         {xs.KEY_LAPLACE_MU: "2.0", xs.KEY_LAPLACE_BETA: "3.0", **cosmos_items})
        self.assertEqual(list(xs.infotext_items(xs.APP_DEFAULTS, [xs.NAME_FLOW_COSMOS_RHO7]).values()),
                         ["7.0", "80.0", "0.002"])
        payload = {"scheduler": xs.LABEL_FLOW_COSMOS_RHO7, "enable_hr": True, "hr_scheduler": xs.NAME_FLOW_COSMOS_RHO7,
                   "alwayson_scripts": {
                       "ADetailer": {"args": [True, False, _slot(ad_use_sampler=True,
                                                                 ad_scheduler=xs.NAME_FLOW_COSMOS_RHO7)]},
                       "SAM3 Mask": {"args": [{"sam3_enable": True, "sam3_mode": "Inpaint",
                                               "sam3_scheduler": xs.LABEL_FLOW_COSMOS_RHO7}]}}}
        name = xs.NAME_FLOW_COSMOS_RHO7
        self.assertEqual(xs.value_scheduler_passes(payload), (xs.PassUse("main", name), xs.PassUse("hires", name),
                                                              xs.PassUse("adetailer_1", name), xs.PassUse("sam3", name)))
        self.assertEqual(xs.passes_from_widgets({"scheduler_combo": xs.NAME_FLOW_COSMOS_RHO7, "hires_options_group": True,
                                                 "hires_scheduler_combo": xs.LABEL_FLOW_COSMOS_RHO7}),
                         (xs.PassUse("main", name), xs.PassUse("hires", name)))
        self.assertEqual(xs.passes_from_widgets({"scheduler_combo": xs.NAME_FLOW_COSMOS_RHO7}, img2img=True), ())
        self.assertEqual(xs.infotext_labels({"Schedule type": xs.NAME_FLOW_COSMOS_RHO7}),
                         frozenset({xs.LABEL_FLOW_COSMOS_RHO7}))
        # 키 없이 Flow Cosmos rho7 → 기본값(D19 전 이미지·다른 도구). 'Schedule rho' 는 이 확장 값이 아니다
        defaults = replace(user, flow_cosmos_rho=7.0, flow_cosmos_sigma_max=80.0, flow_cosmos_sigma_min=0.002)
        self.assertEqual(xs.from_infotext({"Schedule type": xs.LABEL_FLOW_COSMOS_RHO7, "Schedule rho": "3"}, base=user),
                         defaults)
        self.assertEqual(xs.from_infotext({"Schedule type": "Karras", "Schedule rho": "3"}, base=user), user)
        self.assertEqual(xs.from_infotext({"Schedule type": xs.LABEL_FLOW_COSMOS_RHO7, "Flow Cosmos rho": "3.5",
                                           "Flow Cosmos sigma max": "240", "Flow Cosmos sigma min": "0.006"},
                                          base=xs.APP_DEFAULTS),
                         xs.ExtraSchedulerSettings(flow_cosmos_rho=3.5, flow_cosmos_sigma_max=240.0,
                                                   flow_cosmos_sigma_min=0.006))

    def test_pasted_label_selects_the_name_in_the_forge_combo(self):
        """F3 의 붙여 넣기(core.preset_choice.match_scheduler): 라벨·소문자 표기·이름 → 이름 값 콤보의 flow_cosmos_rho7.
        목록에 없으면(옛 sam-extra) 고르지 않는다 — Karras 등으로 새지 않는다."""
        from core.preset_choice import match_scheduler
        combo = ["automatic", "karras", "simple", *xs.SCHEDULER_NAMES]
        for text in (xs.LABEL_FLOW_COSMOS_RHO7, "flow cosmos rho7", xs.NAME_FLOW_COSMOS_RHO7):
            with self.subTest(text=text):
                self.assertEqual(combo[match_scheduler(text, combo)], xs.NAME_FLOW_COSMOS_RHO7)
        self.assertEqual(match_scheduler(xs.LABEL_FLOW_COSMOS_RHO7, ["automatic", "karras", "simple"]), -1)


class FlowCosmosDynamicTests(unittest.TestCase):
    """v0.33.0 D20 Flow Cosmos Dynamic — 이름·표기가 풀리고, Flow Cosmos rho7 의 값 셋(ρ·σ̃ max·σ̃ min)을 같이 읽는 값
    스케줄러다(패스·infotext·붙여 넣기 — 확장 registry.FLOW_COSMOS_LABELS). 따로 값이 없고 README 별칭·힌트도 없다. 확장 없이."""

    def test_name_label_and_shared_values(self):
        self.assertEqual((xs.NAME_FLOW_COSMOS_DYNAMIC, xs.LABEL_FLOW_COSMOS_DYNAMIC),
                         ("flow_cosmos_dynamic", "Flow Cosmos Dynamic"))
        self.assertEqual(xs.SCHEDULER_LABELS[xs.NAME_FLOW_COSMOS_DYNAMIC], xs.LABEL_FLOW_COSMOS_DYNAMIC)
        self.assertEqual(xs.FLOW_COSMOS_SCHEDULERS, (xs.NAME_FLOW_COSMOS_RHO7, xs.NAME_FLOW_COSMOS_DYNAMIC))
        self.assertEqual(xs.FLOW_COSMOS_LABELS, (xs.LABEL_FLOW_COSMOS_RHO7, xs.LABEL_FLOW_COSMOS_DYNAMIC))
        self.assertEqual(xs.VALUE_SCHEDULERS[-2:], xs.FLOW_COSMOS_SCHEDULERS)
        self.assertNotIn(xs.NAME_FLOW_COSMOS_DYNAMIC, xs.README_LABELS)
        self.assertNotIn(xs.NAME_FLOW_COSMOS_DYNAMIC, xs.FLOW_UNSUITABLE)
        self.assertEqual(xs.ARG_NAMES[-3:], ("flow_cosmos_rho", "flow_cosmos_sigma_max", "flow_cosmos_sigma_min"),
                         "새 인자 없음 — Flow Cosmos rho7 의 셋")

    def test_resolves_from_name_label_and_folded_spellings(self):
        for text in ("flow_cosmos_dynamic", "Flow Cosmos Dynamic", "FLOW COSMOS DYNAMIC", "  flow   cosmos dynamic ",
                     "Flow_Cosmos_Dynamic"):
            with self.subTest(text=text):
                self.assertEqual(xs.resolve_scheduler(text), xs.NAME_FLOW_COSMOS_DYNAMIC)
        self.assertEqual(xs.resolve_scheduler("Karras Dynamic"), "karras_dynamic")
        for text in ("Flow Cosmos", "Cosmos Dynamic", "flow-cosmos-dynamic", "Flow Cosmos Dyn", "flow_cosmos"):
            self.assertIsNone(xs.resolve_scheduler(text), text)

    def test_reads_the_flow_cosmos_values(self):
        user = xs.ExtraSchedulerSettings("sigmas", "M", "[1, 0]", 2.0, 3.0, 5.5, 240.0, 0.006)
        cosmos_items = {xs.KEY_FLOW_COSMOS_RHO: "5.5", xs.KEY_FLOW_COSMOS_SIGMA_MAX: "240.0",
                        xs.KEY_FLOW_COSMOS_SIGMA_MIN: "0.006"}
        name, label = xs.NAME_FLOW_COSMOS_DYNAMIC, xs.LABEL_FLOW_COSMOS_DYNAMIC
        self.assertEqual(xs.describe(user, [label]), "Flow Cosmos ρ 5.5 · σ̃ 0.006~240")
        self.assertEqual(xs.describe(user, [label, xs.NAME_FLOW_COSMOS_RHO7]), "Flow Cosmos ρ 5.5 · σ̃ 0.006~240")
        self.assertEqual(xs.infotext_items(user, [label]), cosmos_items)
        self.assertEqual(xs.infotext_items(user, [name, xs.LABEL_FLOW_COSMOS_RHO7]), cosmos_items)
        self.assertEqual(xs.infotext_items(user, [name, "custom"]),
                         {xs.KEY_CUSTOM_SIGMAS: "[1, 0]", **cosmos_items})
        payload = {"scheduler": label, "enable_hr": True, "hr_scheduler": xs.NAME_FLOW_COSMOS_RHO7,
                   "alwayson_scripts": {
                       "ADetailer": {"args": [True, False, _slot(ad_use_sampler=True, ad_scheduler=name)]},
                       "SAM3 Mask": {"args": [{"sam3_enable": True, "sam3_mode": "Inpaint", "sam3_scheduler": label}]}}}
        self.assertEqual(xs.value_scheduler_passes(payload),
                         (xs.PassUse("main", name), xs.PassUse("hires", xs.NAME_FLOW_COSMOS_RHO7),
                          xs.PassUse("adetailer_1", name), xs.PassUse("sam3", name)))
        self.assertEqual(xs.passes_from_widgets({"scheduler_combo": "karras", "hires_options_group": "true",
                                                 "hires_scheduler_combo": label}), (xs.PassUse("hires", name),))
        self.assertEqual(xs.passes_from_widgets({"scheduler_combo": name}, img2img=True), ())
        self.assertEqual(xs.infotext_labels({"Schedule type": name}), frozenset({label}))
        # 키 없이 Dynamic → 기본값(rho7 과 같은 규칙), 키가 있으면 그 값. 'Schedule rho' 는 이 확장 값이 아니다
        defaults = replace(user, flow_cosmos_rho=7.0, flow_cosmos_sigma_max=80.0, flow_cosmos_sigma_min=0.002)
        self.assertEqual(xs.from_infotext({"Schedule type": label, "Schedule rho": "3"}, base=user), defaults)
        self.assertEqual(xs.from_infotext({"Schedule type": "Karras", "Hires schedule type": label}, base=user), defaults)
        self.assertEqual(xs.from_infotext({"Schedule type": label, **cosmos_items}, base=xs.APP_DEFAULTS),
                         xs.ExtraSchedulerSettings(flow_cosmos_rho=5.5, flow_cosmos_sigma_max=240.0,
                                                   flow_cosmos_sigma_min=0.006))
        plan = xs.plan(user, target=xs.TARGET_T2I, backend=xs.BACKEND_WEBUI, capabilities=None,
                       passes=xs.passes_from_widgets({"scheduler_combo": name}))
        self.assertEqual((plan.reason, plan.block), (xs.REASON_SEND_UNCONFIRMED, xs.as_block(user)))

    def test_rho_rule(self):
        """확장 _check_karras_dynamic_rho: ρ > 2(NaN 은 거짓). 정규화된 값(1~15)에서는 1~2 가 거짓."""
        for rho, expected in ((7.0, True), (15.0, True), (4.0, True), (2.1, True), (2.0000000001, True), (2.0, False),
                              (1.5, False), (1.0, False), (0.0, False), (-3.0, False), (math.nan, False),
                              (math.inf, True), (-math.inf, False), ("x", False), (None, False)):
            with self.subTest(rho=rho):
                self.assertIs(xs.flow_cosmos_dynamic_rho_ok(xs.ExtraSchedulerSettings(flow_cosmos_rho=rho)), expected)
        self.assertEqual(xs.FLOW_COSMOS_DYNAMIC_RHO_ABOVE, 2.0)
        self.assertTrue(xs.flow_cosmos_dynamic_rho_ok(xs.APP_DEFAULTS))
        self.assertFalse(xs.flow_cosmos_dynamic_rho_ok(xs.coerce({"flow_cosmos_rho": 0})))   # 자르면 1
        self.assertTrue(xs.flow_cosmos_dynamic_rho_ok(xs.coerce({"flow_cosmos_rho": "nan"})))   # 기본값 7

    def test_pasted_label_selects_the_name_in_the_forge_combo(self):
        from core.preset_choice import match_scheduler
        combo = ["automatic", "karras", "simple", *xs.SCHEDULER_NAMES]
        for text in (xs.LABEL_FLOW_COSMOS_DYNAMIC, "flow cosmos dynamic", xs.NAME_FLOW_COSMOS_DYNAMIC):
            with self.subTest(text=text):
                self.assertEqual(combo[match_scheduler(text, combo)], xs.NAME_FLOW_COSMOS_DYNAMIC)
        self.assertEqual(match_scheduler(xs.LABEL_FLOW_COSMOS_DYNAMIC, ["automatic", "karras", "karras_dynamic"]), -1)


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


class FlowCosmosKnobTests(unittest.TestCase):
    """v0.33.0 D19 — Flow Cosmos ρ·σ̃ max·σ̃ min: Laplace μ/β 처럼 범위 밖은 자르고 유한한 숫자가 아니면 기본값, 블록은 늘
    키 8개(옛 sam-extra 는 뒤 셋을 버린다 — X10 같은 규칙), 옛 블록(인자 다섯)·옛 저장값은 기본값. 확장 없이."""

    def test_defaults_ranges_and_steps(self):
        self.assertEqual((xs.FLOW_COSMOS_RHO_MIN, xs.FLOW_COSMOS_RHO_MAX, xs.FLOW_COSMOS_RHO_DEFAULT,
                          xs.FLOW_COSMOS_RHO_STEP), (1.0, 15.0, 7.0, 0.1))
        self.assertEqual((xs.FLOW_COSMOS_SIGMA_MAX_MIN, xs.FLOW_COSMOS_SIGMA_MAX_MAX, xs.FLOW_COSMOS_SIGMA_MAX_DEFAULT,
                          xs.FLOW_COSMOS_SIGMA_MAX_STEP), (1.0, 1000.0, 80.0, 0.1))
        self.assertEqual((xs.FLOW_COSMOS_SIGMA_MIN_MIN, xs.FLOW_COSMOS_SIGMA_MIN_MAX, xs.FLOW_COSMOS_SIGMA_MIN_DEFAULT,
                          xs.FLOW_COSMOS_SIGMA_MIN_STEP), (0.0001, 1.0, 0.002, 0.0001))
        self.assertEqual((xs.FLOW_COSMOS_RHO_DEFAULT, xs.FLOW_COSMOS_SIGMA_MAX_DEFAULT, xs.FLOW_COSMOS_SIGMA_MIN_DEFAULT),
                         (xs.FLOW_COSMOS_RHO, xs.FLOW_COSMOS_SIGMA_MAX, xs.FLOW_COSMOS_SIGMA_MIN))
        defaults = xs.EXTENSION_DEFAULTS
        self.assertEqual((defaults.flow_cosmos_rho, defaults.flow_cosmos_sigma_max, defaults.flow_cosmos_sigma_min),
                         (7.0, 80.0, 0.002))
        self.assertEqual(xs.ARG_NAMES[5:], ("flow_cosmos_rho", "flow_cosmos_sigma_max", "flow_cosmos_sigma_min"))
        self.assertEqual(xs.INFOTEXT_KEYS[4:], ("Flow Cosmos rho", "Flow Cosmos sigma max", "Flow Cosmos sigma min"))

    def test_coercion_clamps_out_of_range_and_defaults_what_is_not_a_finite_number(self):
        rows = (("flow_cosmos_rho", "0.5", 1.0), ("flow_cosmos_rho", 20, 15.0), ("flow_cosmos_rho", "9.3", 9.3),
                ("flow_cosmos_rho", "nan", 7.0), ("flow_cosmos_rho", True, 7.0), ("flow_cosmos_rho", 10 ** 400, 7.0),
                ("flow_cosmos_sigma_max", 0.5, 1.0), ("flow_cosmos_sigma_max", "5000", 1000.0),
                ("flow_cosmos_sigma_max", "inf", 80.0), ("flow_cosmos_sigma_max", 240, 240.0),
                ("flow_cosmos_sigma_min", 0, 0.0001), ("flow_cosmos_sigma_min", "-1", 0.0001),
                ("flow_cosmos_sigma_min", 2, 1.0), ("flow_cosmos_sigma_min", "0.006", 0.006),
                ("flow_cosmos_sigma_min", None, 0.002), ("flow_cosmos_sigma_min", "abc", 0.002))
        for key, value, expected in rows:
            with self.subTest(key=key, value=repr(value)[:20]):
                coerced = xs.coerce({key: value})
                self.assertEqual(_key(getattr(coerced, key)), _key(expected))
                self.assertEqual(_key(xs.as_block(coerced)["args"][0][key]), _key(expected))

    def test_the_block_always_has_the_eight_keys(self):
        for settings in (xs.APP_DEFAULTS, xs.ExtraSchedulerSettings(laplace_mu=1.0),
                         xs.ExtraSchedulerSettings(flow_cosmos_rho=5.0)):
            with self.subTest(settings=settings):
                block = xs.as_block(settings)
                self.assertEqual(tuple(block["args"][0]), xs.ARG_NAMES)
                self.assertEqual(len(block["args"]), 1, "dict 한 개 — Forge API 는 인자 수와 무관하게 넘긴다")

    def test_old_blocks_and_saved_settings_read_the_defaults(self):
        """v0.31–0.32 의 블록(키·위치 인자 다섯)과 옛 설정 파일(키 다섯)은 Flow Cosmos 기본값(ρ 7 · σ̃ 0.002~80)."""
        user = xs.ExtraSchedulerSettings("sigmas", "M", "[1, 0]", -2.0, 3.0, 5.0, 240.0, 0.006)
        five = {key: value for key, value in user.as_arg().items() if key in xs.ARG_NAMES[:5]}
        expected = replace(user, flow_cosmos_rho=7.0, flow_cosmos_sigma_max=80.0, flow_cosmos_sigma_min=0.002)
        self.assertEqual(xs.parse_script_block({"args": [five]}), expected)
        self.assertEqual(xs.parse_script_block({"args": list(five.values())}), expected)
        self.assertEqual(xs.parse_script_block({"args": list(user.as_arg().values())}), user)
        self.assertEqual(xs.parse_script_block({"args": [*five.values(), "5"]}), replace(expected, flow_cosmos_rho=5.0))
        self.assertEqual(xs.parse_script_block({"args": [user.as_arg()]}), user)
        saved = {key: value for key, value in xs.widget_values(user).items() if key in xs.ARG_NAMES[:5]}
        self.assertEqual(xs.parse_settings(saved), expected)
        self.assertEqual(xs.parse_settings(saved, base=user), user)   # 저장값에 없는 칸은 base(프리셋 only_present)

    def test_range_ok(self):
        self.assertTrue(xs.flow_cosmos_range_ok(xs.APP_DEFAULTS))
        nan, inf = math.nan, math.inf
        for rho, top, bottom, expected in (
                (7.0, 80.0, 0.002, True), (1.0, 1.0, 0.0001, True), (15.0, 1000.0, 1.0, True), (1e-3, 80.0, 0.002, True),
                (7.0, 1.0, 1.0, False),            # 슬라이더로 닿는 유일한 경우 — 두 범위가 만나는 1
                (7.0, 0.5, 0.6, False), (0.0, 80.0, 0.002, False), (-1.0, 80.0, 0.002, False),
                (nan, 80.0, 0.002, False), (inf, 80.0, 0.002, False), (7.0, inf, 0.002, False),
                (7.0, 80.0, 0.0, False), (7.0, 80.0, nan, False), (7.0, 80.0, -0.002, False)):
            with self.subTest(rho=rho, sigma_max=top, sigma_min=bottom):
                settings = xs.ExtraSchedulerSettings(flow_cosmos_rho=rho, flow_cosmos_sigma_max=top,
                                                     flow_cosmos_sigma_min=bottom)
                self.assertIs(xs.flow_cosmos_range_ok(settings), expected)
        self.assertFalse(xs.flow_cosmos_range_ok(xs.coerce({"flow_cosmos_sigma_max": 0.2, "flow_cosmos_sigma_min": 7})))
        self.assertTrue(xs.flow_cosmos_range_ok(xs.coerce({"flow_cosmos_rho": 0, "flow_cosmos_sigma_min": 0})))
        self.assertFalse(xs.flow_cosmos_range_ok(xs.ExtraSchedulerSettings(flow_cosmos_rho="x")))   # 읽을 수 없는 값


# GPU A/B 의 custom 식(Anima, PLAN_amend_v033 X13·DESIGN D18) — 기본값의 Flow Cosmos rho7 목록과 비트까지 같아야 한다
COSMOS_AB_EXPRESSION = "(80**(1/7)+x*(0.002**(1/7)-80**(1/7)))**7/(1+(80**(1/7)+x*(0.002**(1/7)-80**(1/7)))**7)"


def _cosmos_expression(rho: float, top: float, bottom: float) -> str:
    """위 A/B 식에 7·80·0.002 대신 다른 수(repr — 정확한 float)."""
    r, hi, lo = repr(float(rho)), repr(float(top)), repr(float(bottom))
    ramp = f"({hi}**(1/{r})+x*({lo}**(1/{r})-{hi}**(1/{r})))**{r}"
    return f"{ramp}/(1+{ramp})"


def _model(prediction_type: str):
    """Forge 가 need_inner_model 스케줄러에 넘기는 것(model_wrap) 중 확장이 읽는 곳만 — predictor.prediction_type."""
    return types.SimpleNamespace(predictor=types.SimpleNamespace(prediction_type=prediction_type))


@requires_extension
@requires_torch
class FlowCosmosOracleTests(unittest.TestCase):
    """앱이 보내는 블록 → 확장 ui_extra_schedulers.coerce_args → settings.set_active → schedulers.flow_cosmos_rho7
    (확장 process() 와 같은 길, CPU). 기본값·옛 블록·블록 없음은 v0.33.0 D18 의 고정 목록(= GPU A/B custom 식 — 같은 길로
    확장 custom 스케줄러가 계산)과 비트까지 같아야 하고, 다른 값은 그 수를 넣은 A/B 식과 같아야 한다."""

    FLOW_RANGE = (0.001, 1.0)                 # flow 모델이 넘기는 σ 범위(flow 갈래는 쓰지 않는다)
    SDXL_RANGE = (0.0291675, 14.614642)
    STEPS = (1, 2, 3, 28, 50, 150)

    @classmethod
    def setUpClass(cls):
        cls.ui = _ext("ui_extra_schedulers")
        cls.settings = _ext("extra_schedulers.settings")
        cls.schedulers = _ext("extra_schedulers.schedulers")

    def setUp(self):
        self.addCleanup(self.settings.reset)
        patcher = mock.patch.object(self.schedulers, "_flow_cosmos_not_flow_logged", True)   # 콘솔 한 줄 — 조용히
        patcher.start()
        self.addCleanup(patcher.stop)

    def _run(self, args, n, *, scheduler="flow_cosmos_rho7", model="const", sigma_range=None):
        self.settings.set_active(self.ui.coerce_args(args))
        low, high = sigma_range or self.FLOW_RANGE
        if scheduler == "custom":
            return self.schedulers.custom(n, low, high)
        return self.schedulers.flow_cosmos_rho7(n, low, high, inner_model=_model(model))

    def test_defaults_old_blocks_and_no_block_give_the_d18_list_bit_for_bit(self):
        torch = load_torch()
        ab = xs.as_block(xs.ExtraSchedulerSettings(custom_expression=COSMOS_AB_EXPRESSION))["args"]
        default = xs.as_block(xs.APP_DEFAULTS)["args"]
        five = {key: value for key, value in default[0].items() if key in xs.ARG_NAMES[:5]}
        same = {"app default block": default, "old 5-key dict": [five], "old 5 positional": list(five.values()),
                "no args": [], "numbers as text": [{"flow_cosmos_rho": "7", "flow_cosmos_sigma_max": "80",
                                                     "flow_cosmos_sigma_min": "0.002"}]}
        for n in self.STEPS:
            expected = self._run(ab, n, scheduler="custom")
            self.assertEqual((expected.dtype, tuple(expected.shape), float(expected[-1])), (torch.float32, (n + 1,), 0.0))
            for label, args in same.items():
                with self.subTest(block=label, steps=n):
                    got = self._run(args, n)
                    self.assertTrue(torch.equal(got, expected))
                    self.assertEqual(got.numpy().tobytes(), expected.numpy().tobytes())
            self.settings.reset()                          # 블록이 없는 요청 = ui() 기본값(확장 DEFAULTS)
            no_block = self.schedulers.flow_cosmos_rho7(n, *self.FLOW_RANGE, inner_model=_model("const"))
            self.assertTrue(torch.equal(no_block, expected))
        self.assertAlmostEqual(float(self._run(default, 28)[0]), 80.0 / 81.0, places=6)

    def test_other_values_are_the_ab_expression_with_those_numbers(self):
        torch = load_torch()
        reference = {n: self._run(xs.as_block(xs.APP_DEFAULTS)["args"], n) for n in self.STEPS}
        for rho, top, bottom in ((5.0, 240.0, 0.006), (15.0, 1000.0, 0.0001), (1.0, 1.0, 0.0001),
                                 (9.3, 14.614642, 0.0291675), (2.5, 1000.0, 1.0)):
            settings = xs.coerce({"flow_cosmos_rho": rho, "flow_cosmos_sigma_max": top, "flow_cosmos_sigma_min": bottom})
            self.assertEqual((settings.flow_cosmos_rho, settings.flow_cosmos_sigma_max, settings.flow_cosmos_sigma_min),
                             (rho, top, bottom), "슬라이더 안의 값")
            ab = xs.as_block(replace(settings, custom_expression=_cosmos_expression(rho, top, bottom)))["args"]
            for n in self.STEPS:
                with self.subTest(rho=rho, sigma_max=top, sigma_min=bottom, steps=n):
                    got = self._run(xs.as_block(settings)["args"], n)
                    self.assertTrue(torch.equal(got, self._run(ab, n, scheduler="custom")))
                    self.assertFalse(torch.equal(got, reference[n]))

    def test_eps_and_v_use_forges_karras_with_the_blocks_rho_and_not_the_range(self):
        """eps/v: Forge 의 Karras(실행 때 k_diffusion.sampling 에서 찾는다 — 여기서는 기록하는 대역)에 블록의 ρ, Forge 가
        넘긴 σ 범위. σ̃ 둘이 만나는 값(flow 에서는 오류)도 eps/v 는 그대로 돈다."""
        torch = load_torch()
        calls = []

        def get_sigmas_karras(n, sigma_min, sigma_max, rho=7.0, device="cpu"):
            calls.append((n, sigma_min, sigma_max, rho))
            ramp = torch.linspace(0, 1, n)
            top, bottom = sigma_max ** (1 / rho), sigma_min ** (1 / rho)
            return torch.cat([(top + ramp * (bottom - top)) ** rho, torch.zeros(1)])

        sampling = types.ModuleType("k_diffusion.sampling")
        sampling.get_sigmas_karras = get_sigmas_karras
        package = types.ModuleType("k_diffusion")
        package.__path__ = []
        package.sampling = sampling
        settings = xs.coerce({"flow_cosmos_rho": "3.3", "flow_cosmos_sigma_max": 1, "flow_cosmos_sigma_min": 1})
        self.assertFalse(xs.flow_cosmos_range_ok(settings))
        block = xs.as_block(settings)["args"]
        with mock.patch.dict(sys.modules, {"k_diffusion": package, "k_diffusion.sampling": sampling}):
            for kind in ("epsilon", "v_prediction"):
                with self.subTest(kind=kind):
                    calls.clear()
                    got = self._run(block, 28, model=kind, sigma_range=self.SDXL_RANGE)
                    self.assertEqual(calls, [(28, *self.SDXL_RANGE, 3.3)])
                    self.assertTrue(torch.equal(got, get_sigmas_karras(28, *self.SDXL_RANGE, rho=3.3)))
            calls.clear()
            self._run(xs.as_block(xs.APP_DEFAULTS)["args"], 28, model="epsilon", sigma_range=self.SDXL_RANGE)
            self.assertEqual(calls, [(28, *self.SDXL_RANGE, 7.0)])
        with self.assertRaises(self.schedulers.ExtraSchedulerError) as caught:      # flow: σ̃ min = σ̃ max
            self._run(block, 28)
        self.assertTrue(str(caught.exception).startswith(xs.ERROR_PREFIX), str(caught.exception))

    def test_range_ok_is_the_extensions_flow_checks(self):
        sch = self.schedulers

        def extension_ok(settings):
            try:
                sch._check_flow_cosmos_rho(settings.flow_cosmos_rho)
                sch._check_flow_cosmos_range(settings.flow_cosmos_sigma_min, settings.flow_cosmos_sigma_max)
            except sch.ExtraSchedulerError:
                return False
            return True

        values = (-1.0, 0.0, 1e-300, 0.0001, 0.002, 1.0, 7.0, 80.0, 1000.0, math.inf, -math.inf, math.nan)
        for rho in (7.0, 1.0, 0.0, -2.0, 1e-3, math.nan, math.inf):
            for top in values:
                for bottom in values:
                    settings = xs.ExtraSchedulerSettings(flow_cosmos_rho=rho, flow_cosmos_sigma_max=top,
                                                         flow_cosmos_sigma_min=bottom)
                    with self.subTest(rho=rho, sigma_max=top, sigma_min=bottom):
                        self.assertIs(xs.flow_cosmos_range_ok(settings), extension_ok(settings))

    def test_three_times_the_sigma_range_is_animas_shift_3(self):
        """안내 문구의 수식 사실(PLAN_amend_v033 §5): σ̃ 범위 ×3 = flow 시간 shift 3(t' = 3t/(1+2t)) — float64, 반올림까지."""
        times = self.schedulers._flow_cosmos_times
        for n in (2, 3, 28, 151):
            with self.subTest(steps=n):
                base, scaled = times(n, 7.0, 0.002, 80.0), times(n, 7.0, 0.006, 240.0)
                for t, shifted in zip(base, scaled):
                    self.assertTrue(math.isclose(3.0 * t / (1.0 + 2.0 * t), shifted, rel_tol=1e-13, abs_tol=0.0))


@requires_extension
@requires_torch
class FlowCosmosDynamicOracleTests(unittest.TestCase):
    """v0.33.0 D20 — 앱 블록 → 확장 ui_extra_schedulers.coerce_args → settings.set_active → schedulers.flow_cosmos_dynamic
    (확장 process() 와 같은 길, CPU). 기본값·옛 블록·블록 없음은 확장 기본값의 목록과 비트까지 같고, 다른 값은 확장이 그
    수로 직접 만든 목록과 같으며, 같은 블록의 Flow Cosmos rho7 과 지수가 ρ 인 스텝(i = 0, n − 1, 4 | n 이면 n/4·3n/4)에서
    float32 로 같다 — 세 칸이 두 스케줄러에 같이 간다. eps/v 는 블록의 ρ 로 확장 Karras Dynamic 그대로(σ̃ 는 쓰지 않는다)."""

    FLOW_RANGE = (0.001, 1.0)
    SDXL_RANGE = (0.0291675, 14.614642)
    STEPS = (1, 2, 3, 28, 50, 150)
    VALUES = ((5.0, 240.0, 0.006), (15.0, 1000.0, 0.0001), (9.3, 14.614642, 0.0291675), (7.0, 80.0, 0.002),
              (5.5, 1.0, 0.0001))

    @classmethod
    def setUpClass(cls):
        cls.ui = _ext("ui_extra_schedulers")
        cls.settings = _ext("extra_schedulers.settings")
        cls.schedulers = _ext("extra_schedulers.schedulers")

    def setUp(self):
        self.addCleanup(self.settings.reset)
        for flag in ("_flow_cosmos_not_flow_logged", "_flow_cosmos_dynamic_not_flow_logged"):   # 콘솔 한 줄 — 조용히
            patcher = mock.patch.object(self.schedulers, flag, True)
            patcher.start()
            self.addCleanup(patcher.stop)

    def _run(self, args, n, *, scheduler="flow_cosmos_dynamic", model="const", sigma_range=None):
        self.settings.set_active(self.ui.coerce_args(args))
        low, high = sigma_range or self.FLOW_RANGE
        return getattr(self.schedulers, scheduler)(n, low, high, inner_model=_model(model))

    def test_defaults_old_blocks_and_no_block_are_the_extension_default_list(self):
        torch = load_torch()
        default = xs.as_block(xs.APP_DEFAULTS)["args"]
        five = {key: value for key, value in default[0].items() if key in xs.ARG_NAMES[:5]}
        same = {"app default block": default, "old 5-key dict": [five], "old 5 positional": list(five.values()),
                "no args": [], "numbers as text": [{"flow_cosmos_rho": "7", "flow_cosmos_sigma_max": "80",
                                                     "flow_cosmos_sigma_min": "0.002"}]}
        for n in self.STEPS:
            self.settings.reset()                          # 블록이 없는 요청 = ui() 기본값(확장 DEFAULTS)
            expected = self.schedulers.flow_cosmos_dynamic(n, *self.FLOW_RANGE, inner_model=_model("const"))
            self.assertEqual((expected.dtype, tuple(expected.shape), float(expected[-1])), (torch.float32, (n + 1,), 0.0))
            for label, args in same.items():
                with self.subTest(block=label, steps=n):
                    got = self._run(args, n)
                    self.assertEqual(got.numpy().tobytes(), expected.numpy().tobytes())

    def test_block_values_reach_it_and_rho7_shares_them(self):
        torch = load_torch()
        reference = {n: self._run(xs.as_block(xs.APP_DEFAULTS)["args"], n) for n in self.STEPS}
        for rho, top, bottom in self.VALUES:
            settings = xs.coerce({"flow_cosmos_rho": rho, "flow_cosmos_sigma_max": top, "flow_cosmos_sigma_min": bottom})
            self.assertEqual((settings.flow_cosmos_rho, settings.flow_cosmos_sigma_max, settings.flow_cosmos_sigma_min),
                             (rho, top, bottom), "슬라이더 안의 값")
            self.assertTrue(xs.flow_cosmos_range_ok(settings) and xs.flow_cosmos_dynamic_rho_ok(settings))
            block = xs.as_block(settings)["args"]
            direct = self.settings.ExtraSchedulerSettings(flow_cosmos_rho=rho, flow_cosmos_sigma_max=top,
                                                          flow_cosmos_sigma_min=bottom)
            for n in self.STEPS:
                with self.subTest(rho=rho, sigma_max=top, sigma_min=bottom, steps=n):
                    got = self._run(block, n)
                    self.settings.set_active(direct)
                    expected = self.schedulers.flow_cosmos_dynamic(n, *self.FLOW_RANGE, inner_model=_model("const"))
                    self.assertEqual(got.numpy().tobytes(), expected.numpy().tobytes())
                    rho7 = self._run(block, n, scheduler="flow_cosmos_rho7")
                    same = {0, n - 1} | ({n // 4, 3 * n // 4} if n % 4 == 0 else set())
                    for index in sorted(same):
                        self.assertEqual(float(got[index]), float(rho7[index]), f"step {index}")
                    if n >= 3:
                        self.assertFalse(torch.equal(got, rho7), "지수가 스텝마다 다르다")
                    if (rho, top, bottom) != (7.0, 80.0, 0.002):
                        self.assertFalse(torch.equal(got, reference[n]) and n > 1)

    def test_eps_and_v_are_karras_dynamic_with_the_blocks_rho(self):
        torch = load_torch()
        settings = xs.coerce({"flow_cosmos_rho": "5.5", "flow_cosmos_sigma_max": 1, "flow_cosmos_sigma_min": 1})
        self.assertFalse(xs.flow_cosmos_range_ok(settings))       # flow 에서는 오류 — eps/v 는 σ̃ 를 쓰지 않는다
        block = xs.as_block(settings)["args"]
        for kind in ("epsilon", "v_prediction"):
            for n in (2, 3, 28, 50):
                with self.subTest(kind=kind, steps=n):
                    got = self._run(block, n, model=kind, sigma_range=self.SDXL_RANGE)
                    expected = self.schedulers.karras_dynamic(n, *self.SDXL_RANGE, rho=5.5)
                    self.assertTrue(torch.equal(got, expected))
        with self.assertRaises(self.schedulers.ExtraSchedulerError) as caught:
            self._run(block, 28)
        self.assertTrue(str(caught.exception).startswith(xs.ERROR_PREFIX), str(caught.exception))

    def test_rho_rule_is_the_extensions(self):
        sch = self.schedulers

        def extension_ok(rho):
            try:
                sch._check_karras_dynamic_rho(rho, sch.FLOW_COSMOS_DYNAMIC, sch.FLOW_COSMOS_VALUES_SOURCE)
            except sch.KarrasDynamicError:
                return False
            return True

        for rho in (-1.0, 0.0, 1.0, 1.5, 1.9, 2.0, 2.0000000001, 2.05, 3.0, 4.0, 7.0, 15.0, math.nan, math.inf, -math.inf):
            with self.subTest(rho=rho):
                self.assertIs(xs.flow_cosmos_dynamic_rho_ok(xs.ExtraSchedulerSettings(flow_cosmos_rho=rho)),
                              extension_ok(rho))
        for rho in ("1.5", 2, "2.0"):          # 슬라이더 안의 거절 — 두 갈래 모두 목록 전에 멈춘다
            block = xs.as_block(xs.coerce({"flow_cosmos_rho": rho}))["args"]
            self.assertFalse(xs.flow_cosmos_dynamic_rho_ok(xs.coerce({"flow_cosmos_rho": rho})))
            for model, sigma_range in (("const", self.FLOW_RANGE), ("epsilon", self.SDXL_RANGE)):
                with self.subTest(rho=rho, model=model):
                    with self.assertRaises(sch.KarrasDynamicError) as caught:
                        self._run(block, 28, model=model, sigma_range=sigma_range)
                    self.assertTrue(str(caught.exception).startswith(xs.ERROR_PREFIX), str(caught.exception))


class RestoreFromInfotextTests(unittest.TestCase):
    """계획 A-R1(X3b) — '이 설정으로 생성'은 이미지의 메인·Hires 스케줄러가 값을 읽을 때만 **그 스케줄러의** 칸을 스케줄러와
    함께 되살린다. 확장 없이(확장 비교는 ``PasteParityTests.test_generate_from_metadata_restore``)."""

    USER = xs.ExtraSchedulerSettings("sigmas", "  M * 0.5 ", "[2, 1, 0]", -1.0, 4.0, 3.5, 500.0, 0.01)
    ALL_KEYS = {"Custom scheduler expression": "m + (M - m) * (1 - x) ** 3", "Custom scheduler sigmas": "[1.0, 0.0]",
                "Laplace mu": "0.25", "Laplace beta": "1.5", "Flow Cosmos rho": "5.5",
                "Flow Cosmos sigma max": "240.0", "Flow Cosmos sigma min": "0.006"}

    def test_laplace_restores_mu_and_beta_only(self):
        restored = xs.restore_from_infotext({"Schedule type": "Laplace", **self.ALL_KEYS}, base=self.USER)
        self.assertEqual(restored, replace(self.USER, laplace_mu=0.25, laplace_beta=1.5))
        # 키 없이 Laplace — 확장처럼 그 스케줄러의 기본값(다른 도구·옛 이미지)
        self.assertEqual(xs.restore_from_infotext({"Schedule type": "Laplace"}, base=self.USER),
                         replace(self.USER, laplace_mu=xs.LAPLACE_MU_DEFAULT, laplace_beta=xs.LAPLACE_BETA_DEFAULT))
        # 범위 밖은 확장처럼 자른다
        self.assertEqual(xs.restore_from_infotext({"Schedule type": "Laplace", "Laplace mu": "12"}, base=self.USER),
                         replace(self.USER, laplace_mu=10.0, laplace_beta=xs.LAPLACE_BETA_DEFAULT))

    def test_custom_restores_mode_and_text(self):
        base = xs.ExtraSchedulerSettings(laplace_mu=2.0, flow_cosmos_rho=9.0)
        restored = xs.restore_from_infotext({"Schedule type": "custom", **self.ALL_KEYS}, base=base)
        self.assertEqual(restored, replace(base, custom_mode=xs.MODE_SIGMAS,
                                           custom_expression="m + (M - m) * (1 - x) ** 3", custom_sigmas="[1.0, 0.0]"))
        restored = xs.restore_from_infotext({"Schedule type": "custom", "Custom scheduler expression": "  M * x  ",
                                             "Laplace mu": "3"}, base=self.USER)
        self.assertEqual(restored, replace(self.USER, custom_mode=xs.MODE_EXPRESSION, custom_expression="M * x"))
        restored = xs.restore_from_infotext({"Schedule type": "custom", "Custom scheduler sigmas": "[0.9, 0.1]"},
                                            base=xs.APP_DEFAULTS)
        self.assertEqual(restored, replace(xs.APP_DEFAULTS, custom_mode=xs.MODE_SIGMAS, custom_sigmas="[0.9, 0.1]"))
        # custom 인데 키가 없다 — 확장 붙여 넣기처럼 칸 그대로(무엇을 썼는지 모른다)
        self.assertEqual(xs.restore_from_infotext({"Schedule type": "custom"}, base=self.USER), self.USER)

    def test_flow_cosmos_values_for_either_label(self):
        cosmos = {"flow_cosmos_rho": 5.5, "flow_cosmos_sigma_max": 240.0, "flow_cosmos_sigma_min": 0.006}
        defaults = {"flow_cosmos_rho": xs.FLOW_COSMOS_RHO_DEFAULT, "flow_cosmos_sigma_max": xs.FLOW_COSMOS_SIGMA_MAX_DEFAULT,
                    "flow_cosmos_sigma_min": xs.FLOW_COSMOS_SIGMA_MIN_DEFAULT}
        for label in (xs.LABEL_FLOW_COSMOS_RHO7, xs.LABEL_FLOW_COSMOS_DYNAMIC, xs.NAME_FLOW_COSMOS_RHO7,
                      xs.NAME_FLOW_COSMOS_DYNAMIC, "FLOW COSMOS DYNAMIC"):
            with self.subTest(label=label):
                restored = xs.restore_from_infotext({"Schedule type": label, **self.ALL_KEYS}, base=self.USER)
                self.assertEqual(restored, replace(self.USER, **cosmos))
                self.assertEqual(xs.restore_from_infotext({"Schedule type": label, "Schedule rho": "3"}, base=self.USER),
                                 replace(self.USER, **defaults))

    def test_hires_pass_counts_and_both_passes_combine(self):
        restored = xs.restore_from_infotext({"Schedule type": "Karras", "Hires schedule type": "Laplace",
                                             **self.ALL_KEYS}, base=self.USER)
        self.assertEqual(restored, replace(self.USER, laplace_mu=0.25, laplace_beta=1.5))
        restored = xs.restore_from_infotext({"Schedule type": "Laplace", "Hires schedule type": "Flow Cosmos Dynamic",
                                             **self.ALL_KEYS}, base=self.USER)
        self.assertEqual(restored, replace(self.USER, laplace_mu=0.25, laplace_beta=1.5, flow_cosmos_rho=5.5,
                                           flow_cosmos_sigma_max=240.0, flow_cosmos_sigma_min=0.006))
        restored = xs.restore_from_infotext({"Schedule type": "Laplace", "Hires schedule type": xs.USE_SAME_SCHEDULER,
                                             **self.ALL_KEYS}, base=self.USER)
        self.assertEqual(restored, replace(self.USER, laplace_mu=0.25, laplace_beta=1.5))
        self.assertEqual(xs.infotext_value_schedulers({"Schedule type": "laplace", "Hires schedule type": "Laplace"}),
                         (xs.NAME_LAPLACE,))
        self.assertEqual(xs.infotext_value_schedulers({"Schedule type": "custom",
                                                       "Hires schedule type": "Flow Cosmos rho7"}),
                         (xs.NAME_CUSTOM, xs.NAME_FLOW_COSMOS_RHO7))

    def test_values_present_but_not_used_leave_the_fields_alone(self):
        for params in ({"Schedule type": "Karras", **self.ALL_KEYS},
                       {"Schedule type": "Karras", "Hires schedule type": xs.USE_SAME_SCHEDULER, **self.ALL_KEYS},
                       {"Schedule type": "Karras Dynamic", **self.ALL_KEYS},
                       {"Schedule type": xs.LABEL_REACT_DYNSF, **self.ALL_KEYS},
                       {"Schedule type": "CosineExponential blend", "Hires schedule type": "Cosine", **self.ALL_KEYS},
                       {"Sampler": "Euler a", **self.ALL_KEYS}, {"Schedule type": "", **self.ALL_KEYS},
                       {"Schedule type": None, "Hires schedule type": None, **self.ALL_KEYS},
                       {}, self.ALL_KEYS):
            with self.subTest(params=params):
                self.assertIsNone(xs.restore_from_infotext(params, base=self.USER))
                self.assertEqual(xs.infotext_value_schedulers(params), ())
        self.assertIsNone(xs.restore_from_infotext(None, base=self.USER))
        self.assertIsNone(xs.restore_from_infotext("Laplace", base=self.USER))

    def test_folded_labels_restore_like_the_combo_choice(self):
        """앱 콤보는 'LAPLACE'·README 별칭도 그 스케줄러로 고른다 — 값도 같이(키 없으면 그 스케줄러의 기본값). 확장
        paste 는 정확한 표기만 알아 ``from_infotext`` 는 그대로 두지만, '이 설정으로 생성'은 고른 스케줄러를 따른다."""
        self.assertEqual(xs.from_infotext({"Schedule type": "LAPLACE"}, base=self.USER), self.USER)
        self.assertEqual(xs.restore_from_infotext({"Schedule type": "LAPLACE"}, base=self.USER),
                         replace(self.USER, laplace_mu=xs.LAPLACE_MU_DEFAULT, laplace_beta=xs.LAPLACE_BETA_DEFAULT))
        self.assertEqual(xs.restore_from_infotext({"Schedule type": " flow   cosmos rho7 "}, base=self.USER),
                         replace(self.USER, flow_cosmos_rho=xs.FLOW_COSMOS_RHO_DEFAULT,
                                 flow_cosmos_sigma_max=xs.FLOW_COSMOS_SIGMA_MAX_DEFAULT,
                                 flow_cosmos_sigma_min=xs.FLOW_COSMOS_SIGMA_MIN_DEFAULT))
        params = {"Schedule type": "LAPLACE", "Laplace mu": "1"}
        xs.restore_from_infotext(params, base=self.USER)
        self.assertEqual(params, {"Schedule type": "LAPLACE", "Laplace mu": "1"}, "넘긴 dict 를 고치지 않는다")

    def test_restore_fields_cover_every_value_scheduler(self):
        self.assertEqual(set(xs.RESTORE_FIELDS), set(xs.VALUE_SCHEDULERS))
        covered = {field for fields in xs.RESTORE_FIELDS.values() for field in fields}
        self.assertEqual(covered, set(xs.ARG_NAMES))
        self.assertEqual(xs.RESTORE_FIELDS[xs.NAME_FLOW_COSMOS_RHO7], xs.RESTORE_FIELDS[xs.NAME_FLOW_COSMOS_DYNAMIC])


if __name__ == "__main__":
    unittest.main()
