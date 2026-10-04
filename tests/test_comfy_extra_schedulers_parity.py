"""Pack 1.7.0 Extra Schedulers = sam-extra v0.33.0 Extra Schedulers (CPU, no GPU, no ComfyUI).

The pack re-implements sam-extra's Extra Schedulers for ComfyUI
(``comfy_custom_nodes/ai_studio_forge_parity/extra_schedulers_math.py`` — pure Python — and
``extra_schedulers.py`` — torch/ComfyUI glue). sam-extra is GPL-3.0 and is never imported by the
pack; here the **installed** extension (``tests/_sam_extra_ext.EXT_ROOT``) is the oracle: both run on
the same inputs and must give the same values, the same error classes and the same messages (the
messages reach the user).

* ``SettingsCoercionTests`` / ``ExpressionCorpusTests`` — pure, no torch: the block's coercion and the
  custom scheduler's safe expression (accept/reject, exception class, message, ``source``, and the
  evaluated values over step grids).
* ``SigmaListCorpusTests`` / ``ClosedFormTests`` — pure (numpy for Forge's interpolation): the sigma list
  with Forge's ``sd_schedulers._loglinear_interp`` cut out of the Forge install by AST on both sides,
  and the float64 closed forms against sam-extra's helpers cut out of ``schedulers.py``.
* ``SchedulerParityTests`` / ``HandlerTests`` — torch (CPU): every float32 schedule bit for bit
  (n x sigma range, the mu/beta grid incl. the spread and the flat-error paths, the rho grid, custom
  expressions and lists, React Cosinusoidal DynSF up to its float32 flat-step limit, errors alike), and the ComfyUI handlers ``fn(model_sampling, steps)`` reading
  the carried values. Laplace's host function is ComfyUI's ``get_sigmas_laplace`` cut out of
  ``AISTUDIO_COMFY_TEST_ROOT`` when set (else sam-extra's unchanged copy of it; the two are compared
  by AST when both exist).
* Flow Cosmos rho7 (``FlowCosmosRho7ParityTests``, torch): on flow models the fixed flow list == sam-extra's,
  == the GPU A/B's custom expression and == the upstream function's numbers (an inline test reference,
  never shipped), up to its float32 flat-step limit; on eps/v models the host's Karras with ρ 7 == sam-extra
  running Forge's own ``get_sigmas_karras`` (Forge's k-diffusion executed from the Forge checkout); the
  pack's ComfyUI host function is ``get_sigmas_karras`` cut out of ``AISTUDIO_COMFY_TEST_ROOT`` when set,
  else Forge's (the two are compared by AST when both exist); errors, the one-per-process log line and the
  flow rule (the pack's ``extra_samplers_dy.is_const``). Its three values (sam-extra v0.33.0 D19: ρ, σ̃ max,
  σ̃ min, carried like Laplace's μ/β): with the defaults the list is the fixed one (also through an old
  five-value block); over a ρ x σ̃-range x n grid the flow list == sam-extra's and == the A/B expression with
  those numbers, eps/v == Forge's Karras with that ρ (σ̃ unused there); values past the node's limits
  (non-finite or non-positive ρ, a σ̃ range that is not 0 < min < max, an overflow, a last sigma that is 0,
  subnormal or NaN, a flat step of a narrow σ̃ range) stop alike with the same message; the log line names the
  first eps/v run's ρ.
* Flow Cosmos Dynamic (``FlowCosmosDynamicParityTests``, torch; sam-extra v0.33.0 D20 — its own composition of Karras
  Dynamic and Flow Cosmos rho7, with Flow Cosmos rho7's three values): on flow models Karras Dynamic's per-step-ρ ramp
  over σ̃ in flow time == sam-extra's bit for bit over a ρ x σ̃-range x n grid (the model's own range unused), meeting
  Flow Cosmos rho7's list where cos(2πi/n) = 0; on eps/v models Karras Dynamic itself with the node's ρ == sam-extra ==
  the extension's and the pack's ``karras_dynamic`` with that ρ (also with a tiny model sigma_min: no float32-tiny
  rule there); sam-extra's checks in its order with the same classes and texts (ρ above 2, σ̃ range, a σ̃ that is not
  finite, float32-tiny, flat step, a rising step), the float32 flat-step limit (1,277,194 steps with the defaults),
  and its own one-per-process log line. The float64 ramp and flow times == sam-extra's helpers cut out by AST.
* ``PackBehaviourTests`` — no extension: registration in a fake ``comfy.samplers`` (never replaces a
  foreign handler, names published once, ``owned_names`` read live), the carrier (class kept, the
  original ``model_sampling`` untouched, survives ``clone``), the node (inputs = the block, coercion,
  no text check), the Hires override carry and the pack's flow shift keeping the values.

The real-ComfyUI tier is ``tests/test_comfy_extra_schedulers_runtime.py``.

Skipped without the extension, failing under ``AISTUDIO_REQUIRE_FORGE_EXT=1``. torch is imported
lazily (``tests/_optional_deps``). Until sam-extra v0.33.0 is installed, point the oracle at its
work tree: ``AISTUDIO_FORGE_EXTENSION_DIR=<tree>`` and ``SAM3_FORGE_ROOT=<Forge>`` (Forge's own files are
then not two levels above the extension; ``tests/_sam_extra_ext.forge_root``).
"""
from __future__ import annotations

import ast
import importlib
import importlib.util
import math
import os
import sys
import types
import unittest
from pathlib import Path
from typing import Any, Callable, NamedTuple
from unittest import mock

if os.environ.get("AISTUDIO_COMFY_TEST_ROOT") and importlib.util.find_spec("comfy") is not None:
    # run by ComfyUI's own (portable) python, which has ComfyUI on its path: importing the pack
    # loads ComfyUI modules, so force --cpu before its model management starts
    from comfy.cli_args import args as _comfy_args

    _comfy_args.cpu = True

from comfy_custom_nodes.ai_studio_forge_parity import extra_schedulers_math as xmath  # noqa: E402
from tests._forge_k_diffusion import forge_k_sampling, installed_k_sampling  # noqa: E402
from tests._optional_deps import load_torch, requires_torch  # noqa: E402
from tests._sam_extra_ext import EXT_ROOT, FORGE_ROOT, requires_extension  # noqa: E402

torch = None  # filled by bind_torch() in the @requires_torch classes

PACK_DIR = Path(__file__).resolve().parents[1] / "comfy_custom_nodes" / "ai_studio_forge_parity"
_MISSING = object()


def _ext(name: str):
    """``sam3ext.extra_schedulers.<name>`` of the installed extension (expression/settings/sigma_list
    import nothing heavy; schedulers/registry need torch)."""
    root = str(EXT_ROOT)
    if root not in sys.path:
        sys.path.append(root)
    return importlib.import_module(f"sam3ext.extra_schedulers.{name}")


def _ext_ui():
    root = str(EXT_ROOT)
    if root not in sys.path:
        sys.path.append(root)
    return importlib.import_module("sam3ext.ui_extra_schedulers")


def _ext_constant(relative: str, name: str) -> Any:
    """A module-level literal of the installed extension, read by AST (nothing is imported)."""
    tree = ast.parse((EXT_ROOT / relative).read_text(encoding="utf-8"))
    for item in tree.body:
        targets = item.targets if isinstance(item, ast.Assign) else (
            [item.target] if isinstance(item, ast.AnnAssign) else [])
        if any(isinstance(target, ast.Name) and target.id == name for target in targets):
            return ast.literal_eval(item.value)
    raise AssertionError(f"{relative} has no module-level {name}")


def _outcome(function: Callable[[], Any]) -> tuple[str, Any]:
    """('ok', value) or ('error', (class name, message)) — what a caller of ``function`` sees."""
    try:
        return "ok", function()
    except Exception as exc:  # the class name and the text are both part of the contract
        return "error", (type(exc).__name__, str(exc))


def _float_key(value: Any) -> Any:
    """Floats by ``repr`` so -0.0 and 0.0 (and every last bit) count as different."""
    if isinstance(value, float):
        return ("float", repr(value))
    if isinstance(value, (list, tuple)):
        return type(value).__name__, tuple(_float_key(item) for item in value)
    return type(value).__name__, value


# ── settings ────────────────────────────────────────────────────────────────────────────────
_MODE_VALUES = (
    _MISSING, None, "", "expression", "Expression", "EXPRESSION", "sigmas", "Sigmas", "Sigma list", "sigma list",
    " sigma_list ", "SIGMA", "list", "List", "sigma lists", "other", 0, 1, True, False, ["sigmas"], 1.5,
)
_TEXT_VALUES = (_MISSING, None, "", "   ", "  M * x  ", "m\n+ x", 0, 1.5, True, ["a"], "x" * 30000)
_NUMBER_VALUES = (
    _MISSING, None, True, False, 0, 1, -1, 0.0, -0.0, 0.1, -0.1, 0.5, 2.0, -2.5, 9.99, 10, 10.0, 10.5, -10, -10.5,
    11, 1e9, -1e9, float("nan"), float("inf"), float("-inf"), "nan", "inf", "-inf", "1e9", " 2 ", "3.25", "-0",
    "abc", "", [], {}, (1,), 1e-320, -1e-320,
    # around Flow Cosmos rho7's ranges (ρ 1 … 15, σ̃ max 1 … 1000, σ̃ min 0.0001 … 1)
    1e-5, 0.0001, 0.00015, 0.002, 0.9999, 7, 7.25, 14.99, 15, 15.5, 80, 240.0, 999.9, 1000, 1000.5, "80", "0.002",
)


def _raw_corpus() -> list[Any]:
    corpus: list[Any] = [None, {}, (), [("laplace_mu", 2.0)]]
    for value in _MODE_VALUES:
        corpus.append({} if value is _MISSING else {"custom_mode": value})
    for key in ("custom_expression", "custom_sigmas"):
        for value in _TEXT_VALUES:
            corpus.append({} if value is _MISSING else {key: value})
    for key in ("laplace_mu", "laplace_beta", "flow_cosmos_rho", "flow_cosmos_sigma_max", "flow_cosmos_sigma_min"):
        for value in _NUMBER_VALUES:
            corpus.append({} if value is _MISSING else {key: value})
    corpus.append({
        "custom_mode": "Sigma list", "custom_expression": " m ", "custom_sigmas": "[1, 0]",
        "laplace_mu": "0.75", "laplace_beta": 3, "unknown": 1,
    })
    corpus.append({
        "custom_mode": "expression", "custom_expression": "M - s", "custom_sigmas": "1 0", "laplace_mu": 1,
        "laplace_beta": "2", "flow_cosmos_rho": "4.5", "flow_cosmos_sigma_max": 240, "flow_cosmos_sigma_min": 0.006,
    })
    return corpus


@requires_extension
class SettingsCoercionTests(unittest.TestCase):
    """``coerce``/``coerce_args`` == sam-extra ``settings.coerce`` / ``ui_extra_schedulers.coerce_args``."""

    @classmethod
    def setUpClass(cls):
        cls.ext = _ext("settings")

    @staticmethod
    def _fields(settings) -> tuple:
        return tuple(_float_key(getattr(settings, name)) for name in xmath.ARG_NAMES)

    def _same(self, pack_call, ext_call, label):
        pack_kind, pack_value = _outcome(pack_call)
        ext_kind, ext_value = _outcome(ext_call)
        self.assertEqual(pack_kind, ext_kind, label)
        if pack_kind == "ok":
            self.assertEqual(self._fields(pack_value), self._fields(ext_value), label)
        else:
            self.assertEqual(pack_value, ext_value, label)

    def test_constants_match_the_extension(self):
        ext = self.ext
        self.assertEqual(xmath.ARG_NAMES, _ext_constant("sam3ext/ui_extra_schedulers.py", "ARG_NAMES"))
        self.assertEqual((xmath.MODE_EXPRESSION, xmath.MODE_SIGMAS, xmath.MODES),
                         (ext.MODE_EXPRESSION, ext.MODE_SIGMAS, ext.MODES))
        self.assertEqual((xmath.DEFAULT_EXPRESSION, xmath.DEFAULT_SIGMAS), (ext.DEFAULT_EXPRESSION, ext.DEFAULT_SIGMAS))
        self.assertEqual(
            (xmath.LAPLACE_MU_MIN, xmath.LAPLACE_MU_MAX, xmath.LAPLACE_MU_DEFAULT,
             xmath.LAPLACE_BETA_MIN, xmath.LAPLACE_BETA_MAX, xmath.LAPLACE_BETA_DEFAULT),
            (ext.LAPLACE_MU_MIN, ext.LAPLACE_MU_MAX, ext.LAPLACE_MU_DEFAULT,
             ext.LAPLACE_BETA_MIN, ext.LAPLACE_BETA_MAX, ext.LAPLACE_BETA_DEFAULT))
        # the dataclass fields are the block's argument names, in order, with the same defaults
        ext_fields = tuple(ext.ExtraSchedulerSettings.__dataclass_fields__)
        self.assertEqual(xmath.Settings._fields, ext_fields)
        self.assertEqual(xmath.ARG_NAMES, ext_fields)
        self.assertEqual(self._fields(xmath.DEFAULTS), self._fields(ext.DEFAULTS))
        # v0.33.0 appended the three Flow Cosmos values: the first five keep their places (old blocks still read)
        self.assertEqual(xmath.ARG_NAMES[:5], ("custom_mode", "custom_expression", "custom_sigmas", "laplace_mu",
                                               "laplace_beta"))
        self.assertEqual(xmath.ARG_NAMES[5:], ("flow_cosmos_rho", "flow_cosmos_sigma_max", "flow_cosmos_sigma_min"))
        names = [f"FLOW_COSMOS_{value}_{part}" for value in ("RHO", "SIGMA_MAX", "SIGMA_MIN")
                 for part in ("MIN", "MAX", "DEFAULT", "STEP")]
        for name in names:
            with self.subTest(name=name):
                self.assertEqual(repr(getattr(xmath, name)), repr(getattr(ext, name)))
        self.assertEqual((xmath.FLOW_COSMOS_RHO_MIN, xmath.FLOW_COSMOS_RHO_MAX, xmath.FLOW_COSMOS_RHO_DEFAULT,
                          xmath.FLOW_COSMOS_RHO_STEP), (1.0, 15.0, 7.0, 0.1))
        self.assertEqual((xmath.FLOW_COSMOS_SIGMA_MAX_MIN, xmath.FLOW_COSMOS_SIGMA_MAX_MAX,
                          xmath.FLOW_COSMOS_SIGMA_MAX_DEFAULT, xmath.FLOW_COSMOS_SIGMA_MAX_STEP), (1.0, 1000.0, 80.0, 0.1))
        self.assertEqual((xmath.FLOW_COSMOS_SIGMA_MIN_MIN, xmath.FLOW_COSMOS_SIGMA_MIN_MAX,
                          xmath.FLOW_COSMOS_SIGMA_MIN_DEFAULT, xmath.FLOW_COSMOS_SIGMA_MIN_STEP),
                         (0.0001, 1.0, 0.002, 0.0001))
        # the defaults are the closed form's literals (the fixed list of before) — on both sides
        schedulers = "sam3ext/extra_schedulers/schedulers.py"
        for default, literal in (("FLOW_COSMOS_RHO_DEFAULT", "FLOW_COSMOS_RHO"),
                                 ("FLOW_COSMOS_SIGMA_MAX_DEFAULT", "FLOW_COSMOS_SIGMA_MAX"),
                                 ("FLOW_COSMOS_SIGMA_MIN_DEFAULT", "FLOW_COSMOS_SIGMA_MIN")):
            with self.subTest(default=default):
                self.assertEqual(repr(getattr(xmath, default)), repr(getattr(xmath, literal)))
                self.assertEqual(repr(getattr(ext, default)), repr(_ext_constant(schedulers, literal)))

    def test_coerce_corpus(self):
        corpus = _raw_corpus()
        self.assertGreater(len(corpus), 100)
        for raw in corpus:
            with self.subTest(raw=repr(raw)[:80]):
                self._same(lambda: xmath.coerce(raw), lambda: self.ext.coerce(raw), repr(raw)[:80])

    def test_an_int_too_large_for_a_float_reads_as_the_default_like_the_extension(self):
        """sam-extra v0.33.0: an integer too large for a float (a JSON integer past ~1.8e308, where ``float()`` raises
        ``OverflowError``) is not a finite number and becomes that value's default, like ``1e400``; the block's other
        values are kept (before v0.33.0 the error left the coercion on both sides)."""
        defaults = {"laplace_mu": xmath.LAPLACE_MU_DEFAULT, "laplace_beta": xmath.LAPLACE_BETA_DEFAULT,
                    "flow_cosmos_rho": xmath.FLOW_COSMOS_RHO_DEFAULT,
                    "flow_cosmos_sigma_max": xmath.FLOW_COSMOS_SIGMA_MAX_DEFAULT,
                    "flow_cosmos_sigma_min": xmath.FLOW_COSMOS_SIGMA_MIN_DEFAULT}
        for key, default in defaults.items():
            other = "laplace_mu" if key == "laplace_beta" else "laplace_beta"
            for huge in (10 ** 400, -(10 ** 400)):
                raw = {key: huge, "custom_expression": "M - s", other: 2.0}
                with self.subTest(key=key, positive=huge > 0):
                    self._same(lambda: xmath.coerce(raw), lambda: self.ext.coerce(raw), key)
                    settings = xmath.coerce(raw)
                    self.assertEqual(repr(getattr(settings, key)), repr(default))
                    self.assertEqual((settings.custom_expression, getattr(settings, other)), ("M - s", 2.0))
                    self.assertEqual(settings, xmath.coerce({**raw, key: 1e400}))
                    self.assertEqual(xmath.coerce_args([raw]), settings)
        self.assertEqual(repr(xmath.coerce_float(10 ** 400, 0.5, 0.0, 1.0)), repr(0.5))
        self.assertEqual(repr(xmath.coerce_float(10 ** 400, 0.5, 0.0, 1.0)),
                         repr(self.ext.coerce_float(10 ** 400, 0.5, 0.0, 1.0)))

    def test_single_value_coercers(self):
        ext = self.ext
        for value in (*_MODE_VALUES[1:],):
            self.assertEqual(xmath.coerce_mode(value), ext.coerce_mode(value), repr(value))
        for value in _NUMBER_VALUES[1:]:
            with self.subTest(value=value):
                self.assertEqual(_float_key(xmath.coerce_laplace_mu(value)), _float_key(ext.coerce_laplace_mu(value)))
                self.assertEqual(_float_key(xmath.coerce_laplace_beta(value)),
                                 _float_key(ext.coerce_laplace_beta(value)))
                for name in ("coerce_flow_cosmos_rho", "coerce_flow_cosmos_sigma_max", "coerce_flow_cosmos_sigma_min"):
                    self.assertEqual(_float_key(getattr(xmath, name)(value)), _float_key(getattr(ext, name)(value)),
                                     name)
                for bounds in ((0.0, -1.0, 1.0), (0.5, 0.0, 10.0), (-3.0, -5.0, -1.0)):
                    self.assertEqual(_float_key(xmath.coerce_float(value, *bounds)),
                                     _float_key(ext.coerce_float(value, *bounds)))
        for value in _TEXT_VALUES[1:]:
            self.assertEqual(xmath.coerce_text(value, "dflt"), ext.coerce_text(value, "dflt"))

    @requires_torch
    def test_coerce_args_matches_the_scripts_argument_rule(self):
        """``ui_extra_schedulers.coerce_args`` (its module imports the registry, hence torch)."""
        ext_ui = _ext_ui()
        cases: list[Any] = [None, (), [], ["sigmas"], ["Sigma list", " m ", "[1, 0]", "0.5", 2],
                            ["expression", None, None, None, None, "extra"], [{"laplace_mu": 1.5}],
                            [{"custom_mode": "list"}, "ignored"], [{}], ({"laplace_beta": True},),
                            ["sigmas", "x", "y", float("nan"), -0.0], [None, None, None, -11, 11],
                            # v0.33.0's eight (the three Flow Cosmos values appended), clamped, defaulted, short, long
                            ["Sigma list", " m ", "[1, 0]", "0.5", 2, 4.5, 240, 0.006],
                            ["expression", None, None, None, None, 16, 0.5, 2],
                            ["expression", None, None, None, None, "x", float("inf"), None],
                            ["expression", None, None, None, None, 9.3, 300.0],
                            ["expression", None, None, None, None, 9.3, 300.0, 0.01, "extra"],
                            [None, None, None, None, None, 10 ** 400, -(10 ** 400), 10 ** 400],
                            [{"flow_cosmos_rho": 3, "flow_cosmos_sigma_min": "0.01"}],
                            [{"flow_cosmos_sigma_max": 1.0, "flow_cosmos_sigma_min": 1.0}]]
        for args in cases:
            with self.subTest(args=args):
                self._same(lambda: xmath.coerce_args(args), lambda: ext_ui.coerce_args(args), repr(args))
        # an old five-value block (v0.31–v0.32 requests, presets, history): Flow Cosmos runs on its defaults
        old = xmath.coerce_args(["sigmas", "x", "y", 1.0, 2.0])
        self.assertEqual(tuple(old[5:]), (xmath.FLOW_COSMOS_RHO_DEFAULT, xmath.FLOW_COSMOS_SIGMA_MAX_DEFAULT,
                                          xmath.FLOW_COSMOS_SIGMA_MIN_DEFAULT))
        self.assertEqual(tuple(old[:5]), ("sigmas", "x", "y", 1.0, 2.0))


# ── the custom scheduler's expression ─────────────────────────────────────────────────────────
def _balanced_min_chain(count: int) -> str:
    """``count`` eight-argument ``min`` calls joined by ``+``: many nodes, shallow, short text."""
    return "+".join(["min(x,x,x,x,x,x,x,x)"] * count)


def _expression_corpus() -> list[Any]:
    valid = [
        xmath.DEFAULT_EXPRESSION, "m + (M - m) * (1 - x) ** 3", "M * (1 - x) + m * x", "M", "m", "n", "s", "x",
        "n - s", "s / (n - 1)", "phi", "pi * x", "e ** -x", "phi ** 2", "--x", "-+-x", "+x", "- - - M", "+-+-+M",
        "9**2**2", "2 ** 64", "2 ** -64", "2.0 ** 64.0", "(-2) ** 2", "(-2) ** 3", "(-8) ** (1/3)", "(-1)**0.5",
        "0 ** -1", "0 ** 0", "x ** 64.5", "9**9**9", "2 ** 65", "M ** 64", "1e-3", "1E3", ".5", "5.", "0x10",
        "0b101", "0o17", "1_000", "  M\t*\n(m/M)**x  ", "M*(m/M)**x", "1/0", "M / 0", "1 / (x - x)",
        "exp(1000)", "1e308 * 10", "1e308 + 1e308", "-1e308 - 1e308", "1e308 / 1e-10", "1e-308 / 1e308",
        "1e-200 * 1e-200", "log(0)", "sqrt(-1)", "log(1, 1)", "log(2, 0)", "log(2, -1)", "asin(2)", "acos(M)",
        "floor(1e300)", "ceil(-0.5)", "floor(-0.5)", "abs(-0.0)", "abs(m - M)", "sqrt(M)", "exp(-x)", "log(M)",
        "log(M, 2)", "log(M, 10)", "log2(M)", "log10(M)", "sin(x)", "cos(x)", "tan(x)", "tan(pi / 2)",
        "asin(x)", "acos(x)", "atan(x)", "atan2(m, M)", "atan2(0, -1)", "sinh(x)", "cosh(x)", "tanh(x)",
        "sinh(1000)", "cosh(-1000)", "floor(M)", "ceil(m)", "min(m, M)", "max(m, M, x, 1)",
        "min(1,2,3,4,5,6,7,8)", "max(0.0, -0.0)", "min(0.0, -0.0)", "min(-0.0, 0.0)", "max(x, s, n, m, M)",
        "M * (m / M) ** (x ** 0.5)", "m + (M - m) * cos(pi / 2 * x)", "exp(log(M) + (log(m) - log(M)) * x)",
        "m + (M - m) * (1 - x) ** (phi ** 2)", "max(m, M - s)", "M - s * (M - m) / (n - 1)", "x - x",
        "-(M)", "M * -1", "ｘ", "((((((M))))))",
        "-" * 31 + "x", "-" * 32 + "x", "+" * 40 + "x", _balanced_min_chain(20), _balanced_min_chain(21),
        "x" + "+x" * 99, "-x" + "+x" * 99,
    ]
    invalid = [
        "", "   ", "\t\n", None, 1, 1.5, b"x", ["x"],
        "x" * 501, "x" + "+x" * 250, "1" + "0" * 400, "1e400", "-1e400", "True", "False", "None", "'a'", "b'a'",
        "...", "1j", "2 + 3j", "x if 1 else 2", "a.b", "M.real", "().__class__", "__import__('os')",
        "open('x')", "eval('1')", "f(x=1)", "min(x=1, y=2)", "min(m, M, key=abs)", "min(*x)", "max(**x)",
        "max(*[1, 2])", "lambda: 1", "[x for x in y]", "(x for x in y)", "{x for x in y}", "{x: 1 for x in y}",
        "x < 1", "x == 1", "x and 1", "x or 1", "not x", "~1", "x % 2", "x // 2", "x @ 2", "x << 1",
        "x >> 1", "x | 1", "x & 1", "x ^ 1", "f'{x}'", "(y := 1)", "*x", "[1, 2]", "(1, 2)", "{1: 2}", "{1}",
        "x[0]", "x[1:2]", "y", "X", "Phi", "PI", "sin", "math.sin(x)", "sin()", "atan2(1)", "log(1, 2, 3)",
        "min(1)", "max(1,2,3,4,5,6,7,8,9)", "sin(1)(2)", "(sin)(1)", "abs(1, 2)", "x +", "x)", "(x", "x x",
        "1 2", "x\x00", "\x00", "(" * 201 + "x" + ")" * 201, "π", "١", "x = 1", "import os", "await x",
        "yield x", "x if x else x", "-" * 33 + "x", "+" * 33 + "x", "min(" * 20 + "x" + ",x)" * 20,
        "x" + "+x" * 100, _balanced_min_chain(24), "--" * 20 + "x", "sin(x, )", "min(x,)",
        "sqrt(x" + ")" * 2, "'" * 3, "\\", "#x", "x # comment", "x;y", "x\ny",
    ]
    return valid + invalid


_STEP_GRIDS = (
    (1, 0.0291675, 14.614642), (2, 0.0291675, 14.614642), (5, 0.002, 1.0), (10, 0.5, 0.5), (28, 0.0, 1.0),
    (3, 1e-30, 1e30), (4, 0.03, 4500.0),
)


@requires_extension
class ExpressionCorpusTests(unittest.TestCase):
    """``compile_expression``/``Expression.evaluate``/``step_variables`` == sam-extra ``expression``."""

    @classmethod
    def setUpClass(cls):
        cls.ext = _ext("expression")

    def test_bounds_names_and_functions_match_the_extension(self):
        ext = self.ext
        self.assertEqual(
            (xmath.MAX_LENGTH, xmath.MAX_NODES, xmath.MAX_DEPTH, xmath.MAX_EXPONENT, xmath.MAX_CALL_ARGS),
            (ext.MAX_LENGTH, ext.MAX_NODES, ext.MAX_DEPTH, ext.MAX_EXPONENT, ext.MAX_CALL_ARGS))
        self.assertEqual(repr(xmath.GOLDEN_RATIO), repr(ext.GOLDEN_RATIO))
        self.assertEqual(xmath.VARIABLES, ext.VARIABLES)
        self.assertEqual(list(xmath.CONSTANTS.items()), list(ext.CONSTANTS.items()))
        self.assertEqual(list(xmath.FUNCTIONS), list(ext.FUNCTIONS))   # the order the messages list them in
        self.assertEqual({name: spec[1:] for name, spec in xmath.FUNCTIONS.items()},
                         {name: spec[1:] for name, spec in ext.FUNCTIONS.items()})
        self.assertEqual(xmath.ExpressionError.__name__, ext.ExpressionError.__name__)
        self.assertTrue(issubclass(xmath.ExpressionError, ValueError))

    def test_functions_give_the_same_values(self):
        samples = (-3.5, -1.0, -0.5, -0.0, 0.0, 1e-9, 0.25, 0.5, 1.0, 2.0, 7.25, 700.0, 1e300)
        for name, (function, fewest, most) in xmath.FUNCTIONS.items():
            ext_function = self.ext.FUNCTIONS[name][0]
            for count in sorted({fewest, most, min(most, 3)}):
                for index in range(len(samples)):
                    args = [samples[(index + k * 5) % len(samples)] for k in range(count)]
                    with self.subTest(name=name, args=args):
                        self.assertEqual(_float_key(_outcome(lambda: function(*args))),
                                         _float_key(_outcome(lambda: ext_function(*args))))

    def test_corpus_accepts_and_rejects_alike_with_the_same_messages(self):
        corpus = _expression_corpus()
        self.assertGreaterEqual(len(corpus), 150)
        accepted = rejected = 0
        for text in corpus:
            with self.subTest(text=repr(text)[:60]):
                pack_kind, pack = _outcome(lambda: xmath.compile_expression(text))
                ext_kind, ext = _outcome(lambda: self.ext.compile_expression(text))
                self.assertEqual(pack_kind, ext_kind)
                if pack_kind == "error":
                    rejected += 1
                    self.assertEqual(pack, ext)
                    continue
                accepted += 1
                self.assertEqual(pack.source, ext.source)
                self._same_values(pack, ext)
        # the corpus really exercises both sides
        self.assertGreater(accepted, 100)
        self.assertGreater(rejected, 80)

    def _same_values(self, pack, ext):
        for steps, sigma_min, sigma_max in _STEP_GRIDS:
            for step in range(steps):
                variables = xmath.step_variables(step, steps, sigma_min, sigma_max)
                self.assertEqual(_float_key(variables),
                                 _float_key(self.ext.step_variables(step, steps, sigma_min, sigma_max)))
                got = _outcome(lambda: pack.evaluate(**variables))
                want = _outcome(lambda: ext.evaluate(**variables))
                self.assertEqual(_float_key(got), _float_key(want), (steps, step, sigma_min, sigma_max))

    def test_documented_limits(self):
        """Each bound fires where sam-extra's does — and the corpus above holds the cases on both sides."""
        self.assertIn("more than 200 parts", str(_outcome(lambda: xmath.compile_expression(
            _balanced_min_chain(21)))[1]))   # 21 calls: 209 nodes in 440 characters
        xmath.compile_expression(_balanced_min_chain(20))                  # 199 nodes
        # a left-leaning chain of 100 terms is 100 levels deep: the depth bound fires first
        self.assertIn("nested too deeply", str(_outcome(lambda: xmath.compile_expression("x" + "+x" * 99))[1]))
        self.assertIn("nested too deeply", str(_outcome(lambda: xmath.compile_expression("-" * 32 + "x"))[1]))
        self.assertIn("longer than 500", str(_outcome(lambda: xmath.compile_expression("x" * 501))[1]))
        error = _outcome(lambda: xmath.compile_expression("9**9**9").evaluate(
            **xmath.step_variables(0, 1, 0.1, 1.0)))
        self.assertEqual(error, ("error", ("ExpressionError", f"exponent {9.0 ** 9:g} is outside ±64")))

    def test_evaluate_needs_every_variable_and_finite_values(self):
        pack, ext = xmath.compile_expression("m + x"), self.ext.compile_expression("m + x")
        cases = (
            {"m": 1, "M": 2, "n": 3, "s": 0},
            {"m": float("nan"), "M": 2, "n": 3, "s": 0, "x": 0},
            {"m": 1, "M": float("inf"), "n": 3, "s": 0, "x": 0},
            {"m": "0.5", "M": 2, "n": 3, "s": 0, "x": "0.25", "extra": object()},
            {"m": 1, "M": 2, "n": 3, "s": 0, "x": "nope"},
            {"m": 1, "M": 2, "n": 3, "s": 0, "x": None},
            {},
        )
        for variables in cases:
            with self.subTest(variables=variables):
                self.assertEqual(_float_key(_outcome(lambda: pack.evaluate(**variables))),
                                 _float_key(_outcome(lambda: ext.evaluate(**variables))))

    def test_normalize_source_matches(self):
        for text in ("  a \t b\n\nc ", "", "x", "　x y", None, 5):
            self.assertEqual(_outcome(lambda: xmath.normalize_source(text)),
                             _outcome(lambda: self.ext.normalize_source(text)))


# ── the custom scheduler's sigma list, the closed forms (PB-XS2) ──────────────────────────────
# FORGE_ROOT (tests._sam_extra_ext.forge_root): $SAM3_FORGE_ROOT, else two levels above the extension.
COMFY_ROOT = os.environ.get("AISTUDIO_COMFY_TEST_ROOT")


def _cut_function(path: Path, name: str, **namespace: Any) -> Callable[..., Any]:
    """One top-level function cut out of a host file and run alone (the file's imports are not)."""
    node = next((item for item in ast.parse(path.read_text(encoding="utf-8")).body
                 if isinstance(item, ast.FunctionDef) and item.name == name), None)
    if node is None:
        raise AssertionError(f"{path} has no top-level function {name}")
    scope = dict(namespace)
    exec(compile(ast.Module(body=[node], type_ignores=[]), f"{path.name}:{name}", "exec"), scope)
    return scope[name]


def _function_body_dump(path: Path, name: str) -> str:
    """The function's body without its docstring, as an AST dump (names and comments do not count)."""
    node = next(item for item in ast.parse(path.read_text(encoding="utf-8")).body
                if isinstance(item, ast.FunctionDef) and item.name == name)
    body = list(node.body)
    if body and isinstance(body[0], ast.Expr) and isinstance(getattr(body[0], "value", None), ast.Constant) \
            and isinstance(body[0].value.value, str):
        body = body[1:]
    return ast.dump(ast.Module(body=body, type_ignores=[]), include_attributes=False) + ast.dump(node.args)


def forge_loglinear_interp() -> Callable[..., Any]:
    """Forge's ``modules.sd_schedulers._loglinear_interp`` (what sam-extra calls at run time), cut by AST."""
    import numpy as np

    path = FORGE_ROOT / "modules" / "sd_schedulers.py"
    if not path.is_file():
        raise unittest.SkipTest(f"Forge's {path} is missing")
    return _cut_function(path, "_loglinear_interp", np=np)


def _sigma_list_corpus() -> list[Any]:
    many = ", ".join(f"{1.0 - i / 1000:.4f}" for i in range(1000))
    too_many = many + ", 0.0001"
    return [
        xmath.DEFAULT_SIGMAS, "[1.0, 0.6, 0.25, 0.1, 0.0]", "(1, 0.5, 0)", "1 0", "1.0, 0.0", "[1.0, 0.0]",
        "[14.6, 7, 3.1, 1.2, 0.03]", "14.6 7 3 0", "14.6, 6, 2, 0.5, 0.03", "[10, 3, 1, 0.1, 0]", "0.5, 0.5",
        "[1.0, 0.5, 0.0]", "[14.6, 3, 0.03]", "1 0 0", "1, 0, 0.5", "0, 1", "2, 1, 0.0", "1.0 0.5 0.0 0.0",
        "[1, 2", "(1, 2]", "[1, 2)", "[]", "()", "[1]", "1", "", "   ", "1,,2", " , 1 , 2 , ", "1\t2\n3",
        "+1, 2", ".5 .25", "5. 2.", "1e-3 1E-5", "1e", "e1", "--1 2", "1 -2", "-0.1, 1", "-0.0, 1", "1, -0.0",
        "nan, 1", "inf 1", "1e400 1", "0x1 2", "1_0 2", "١ ٢", "1 ２", "1;2", "1/2 3", "[[1, 2]]",
        "( 3 , 2 , 1 )", "[3 2 1 0]", "1e-45 1e-46", "1e38 1e39", "3.4028235e38 1", "1.0000000001 0.9999999999",
        many, too_many, "1 " * 10001, None, 5, ["1", "2"], b"1 2",
    ]


_LIST_STEPS = (0, 1, 2, 3, 5, 10, 28, 1000)
_LIST_RANGES = ((0.0291675, 14.614642), (0.002, 1.0), (0.5, 0.5))


@requires_extension
class SigmaListCorpusTests(unittest.TestCase):
    """``parse_sigma_list``/``sigmas_from_list`` == sam-extra ``sigma_list`` (Forge's interpolation for both)."""

    @classmethod
    def setUpClass(cls):
        cls.ext = _ext("sigma_list")
        cls.interp = staticmethod(forge_loglinear_interp())

    def test_bounds_match(self):
        self.assertEqual((xmath.MAX_TEXT_LENGTH, xmath.MAX_VALUES), (self.ext.MAX_TEXT_LENGTH, self.ext.MAX_VALUES))
        self.assertEqual(xmath.SigmaListError.__name__, self.ext.SigmaListError.__name__)

    def test_parse_corpus(self):
        corpus = _sigma_list_corpus()
        self.assertGreater(len(corpus), 60)
        for text in corpus:
            with self.subTest(text=repr(text)[:60]):
                self.assertEqual(_float_key(_outcome(lambda: xmath.parse_sigma_list(text))),
                                 _float_key(_outcome(lambda: self.ext.parse_sigma_list(text))))
                parsed = _outcome(lambda: xmath.parse_sigma_list(text))
                if parsed[0] == "ok":
                    self.assertEqual(xmath.is_normalized(parsed[1]), self.ext.is_normalized(parsed[1]))

    def test_sigmas_from_list_corpus(self):
        compared = 0
        for text in _sigma_list_corpus():
            for steps in _LIST_STEPS:
                for sigma_min, sigma_max in _LIST_RANGES:
                    got = _outcome(lambda: xmath.sigmas_from_list(text, steps, sigma_min, sigma_max,
                                                                  interp=self.interp))
                    want = _outcome(lambda: self.ext.sigmas_from_list(text, steps, sigma_min, sigma_max,
                                                                      interp=self.interp))
                    self.assertEqual(_float_key(got), _float_key(want), (repr(text)[:60], steps, sigma_min))
                    compared += 1
        self.assertGreater(compared, 1000)

    def test_an_unusable_interpolation_is_refused_alike(self):
        def short(values, count):
            return [1.0] * (count - 1)

        def not_finite(values, count):
            return [float("nan")] * count

        def zero_tail(values, count):
            return [1.0] * (count - 1) + [0.0]

        def numpy_like(values, count):
            import numpy as np

            return np.linspace(values[0], values[-1], count)

        for interp in (short, not_finite, zero_tail, numpy_like):
            for text, steps in (("[14.6, 3, 0.03]", 7), ("[1.0, 0.5, 0.0]", 9), ("2 1", 2)):
                with self.subTest(interp=interp.__name__, text=text):
                    self.assertEqual(
                        _float_key(_outcome(lambda: xmath.sigmas_from_list(text, steps, 0.03, 14.6, interp=interp))),
                        _float_key(_outcome(lambda: self.ext.sigmas_from_list(text, steps, 0.03, 14.6,
                                                                              interp=interp))))

    def test_the_pack_needs_the_host_interpolation(self):
        """The pack never reaches for Forge: the interpolation is always the caller's."""
        with self.assertRaises(TypeError):
            xmath.sigmas_from_list("[14.6, 3, 0.03]", 7, 0.03, 14.6)   # noqa: missing keyword on purpose

    def test_comfy_loglinear_interp_is_forge_s_function(self):
        """ComfyUI's ``loglinear_interp`` (what the pack calls at run time) has Forge's body."""
        if not COMFY_ROOT:
            self.skipTest("Set AISTUDIO_COMFY_TEST_ROOT to compare ComfyUI's loglinear_interp with Forge's")
        comfy = Path(COMFY_ROOT) / "comfy_extras" / "nodes_align_your_steps.py"
        forge = FORGE_ROOT / "modules" / "sd_schedulers.py"
        self.assertEqual(_function_body_dump(comfy, "loglinear_interp"), _function_body_dump(forge, "_loglinear_interp"))


@requires_extension
class ClosedFormTests(unittest.TestCase):
    """The float64 closed forms and the expression schedule == the same helpers of sam-extra
    ``schedulers.py`` (cut out by AST and run without torch). The float32 schedules the samplers
    get are compared in ``SchedulerParityTests``."""

    SCHEDULERS = "sam3ext/extra_schedulers/schedulers.py"

    @classmethod
    def setUpClass(cls):
        path = EXT_ROOT / cls.SCHEDULERS
        cls.epsilon = _ext_constant(cls.SCHEDULERS, "LAPLACE_EPSILON")
        cls.ext_expr = _ext("expression")
        names = ("_positions", "_cosine_value", "_exponential_value", "_laplace_position", "sigmas_from_expression")
        cls.ext = {name: _cut_function(path, name, math=math, LAPLACE_EPSILON=cls.epsilon, expr=cls.ext_expr)
                   for name in names}

    def test_constants_match(self):
        self.assertEqual(xmath.LAPLACE_EPSILON, self.epsilon)
        self.assertEqual(xmath.KARRAS_DYNAMIC_RHO, _ext_constant(self.SCHEDULERS, "KARRAS_DYNAMIC_RHO"))
        self.assertEqual(xmath.RHO_SOURCE, _ext_constant(self.SCHEDULERS, "RHO_SOURCE"))
        self.assertEqual(repr(xmath.REACT_DYNSF_FACTOR), repr(_ext_constant(self.SCHEDULERS, "REACT_DYNSF_FACTOR")))
        for name in ("FLOW_COSMOS_SIGMA_MAX", "FLOW_COSMOS_SIGMA_MIN", "FLOW_COSMOS_RHO", "FLOW_COSMOS_NOT_FLOW_LOG",
                     "FLOW_COSMOS_VALUES_SOURCE", "FLOW_COSMOS_DYNAMIC_NOT_FLOW_LOG"):
            with self.subTest(name=name):
                self.assertEqual(repr(getattr(xmath, name)), repr(_ext_constant(self.SCHEDULERS, name)))
        self.assertEqual((xmath.FLOW_COSMOS_SIGMA_MAX, xmath.FLOW_COSMOS_SIGMA_MIN, xmath.FLOW_COSMOS_RHO),
                         (80.0, 0.002, 7.0))
        # D19 made the log line a template; with the default ρ it is D18's line
        self.assertEqual(xmath.FLOW_COSMOS_NOT_FLOW_LOG.format(rho=xmath.FLOW_COSMOS_RHO),
                         "[Extra Schedulers] Flow Cosmos rho7: not a flow model - using Karras rho 7 on the model's "
                         "own sigma range")
        # D20: Flow Cosmos Dynamic's own line, the same template shape
        self.assertEqual(xmath.FLOW_COSMOS_DYNAMIC_NOT_FLOW_LOG.format(rho=xmath.FLOW_COSMOS_RHO),
                         "[Extra Schedulers] Flow Cosmos Dynamic: not a flow model - using Karras Dynamic rho 7 on the "
                         "model's own sigma range")
        self.assertEqual(_ext_constant(self.SCHEDULERS, "FLOW_COSMOS_DYNAMIC"), "Flow Cosmos Dynamic")

    def test_flow_cosmos_rho7_times(self):
        """The float64 flow times == sam-extra's ``_flow_cosmos_times`` (cut out by AST) and == the GPU A/B's
        custom expression through the safe evaluator, on both sides — the same values before the float32 cast."""
        path = EXT_ROOT / self.SCHEDULERS
        constants = {name: _ext_constant(self.SCHEDULERS, name)
                     for name in ("FLOW_COSMOS_SIGMA_MAX", "FLOW_COSMOS_SIGMA_MIN", "FLOW_COSMOS_RHO")}
        ext_times = _cut_function(path, "_flow_cosmos_times", _positions=self.ext["_positions"], **constants)
        for n in sorted({-1, 0, 1000, 1001, *range(1, 301)}):
            with self.subTest(n=n):
                ours = xmath.flow_cosmos_rho7_times(n)
                _assert_same(self, _float_key(ours), _float_key(ext_times(n)), n)
                self.assertEqual(len(ours), max(n, 0))
        for n in sorted({*COSMOS_STEPS, *range(1, 201)}):
            for sigma_min, sigma_max in ((0.0029940119, 1.0), (0.0291675, 14.614642), (0.002, 80.0)):
                with self.subTest(n=n, range=(sigma_min, sigma_max)):
                    by_expression = xmath.sigmas_from_expression(COSMOS_AB_EXPRESSION, n, sigma_min, sigma_max)
                    _assert_same(self, _float_key(by_expression), _float_key(xmath.flow_cosmos_rho7_times(n)), n)
                    _assert_same(self, _float_key(by_expression), _float_key(self.ext["sigmas_from_expression"](
                        COSMOS_AB_EXPRESSION, n, sigma_min, sigma_max)), n)
        top = xmath.flow_cosmos_rho7_times(1)   # one step: p = 0, s = 80 -> 80/81 (the others' n == 1 rule)
        self.assertEqual(len(top), 1)
        self.assertAlmostEqual(top[0], 80.0 / 81.0, delta=1e-15)
        self.assertAlmostEqual(xmath.flow_cosmos_rho7_times(2)[-1], 0.002 / 1.002, delta=1e-15)
        # the defaults spelt out are the same list
        for n in (1, 2, 28, 151):
            self.assertEqual(_float_key(xmath.flow_cosmos_rho7_times(n, 7.0, 0.002, 80.0)),
                             _float_key(xmath.flow_cosmos_rho7_times(n)))

    def test_flow_cosmos_rho7_times_with_the_values(self):
        """D19: the float64 flow times for a ρ x σ̃-range x n grid (sliders and past them) == sam-extra's
        ``_flow_cosmos_times`` cut out by AST — values, or the ``OverflowError`` of the float maths alike — and, within
        the sliders, == the A/B expression with those numbers through both safe evaluators (the same operations)."""
        path = EXT_ROOT / self.SCHEDULERS
        constants = {name: _ext_constant(self.SCHEDULERS, name)
                     for name in ("FLOW_COSMOS_SIGMA_MAX", "FLOW_COSMOS_SIGMA_MIN", "FLOW_COSMOS_RHO")}
        ext_times = _cut_function(path, "_flow_cosmos_times", _positions=self.ext["_positions"], **constants)
        outcomes = set()
        for rho in (*COSMOS_RHOS, 0.05, 0.001, 30.0, 1e-300):
            for top, bottom in (*COSMOS_VE_RANGES, (80.0, 1e-30), (80.0, 1e-300), (80.0, 5e-324)):
                for n in COSMOS_KNOB_STEPS:
                    with self.subTest(rho=rho, sigma_max=top, sigma_min=bottom, n=n):
                        ours = _outcome(lambda: xmath.flow_cosmos_rho7_times(n, rho, bottom, top))
                        theirs = _outcome(lambda: ext_times(n, rho, bottom, top))
                        _assert_same(self, _float_key(ours), _float_key(theirs), (rho, top, bottom, n))
                        outcomes.add(ours[0] if ours[0] == "ok" else ours[1][0])
        self.assertEqual(outcomes, {"ok", "OverflowError"})
        for rho in COSMOS_RHOS:
            for top, bottom in COSMOS_VE_RANGES:
                expression = cosmos_expression(rho, top, bottom)
                for n in COSMOS_KNOB_STEPS:
                    with self.subTest(expression=expression, n=n):
                        times = xmath.flow_cosmos_rho7_times(n, rho, bottom, top)
                        _assert_same(self, _float_key(xmath.sigmas_from_expression(expression, n, 0.03, 1.0)),
                                     _float_key(times), (rho, top, bottom, n))
                        _assert_same(self, _float_key(self.ext["sigmas_from_expression"](expression, n, 0.03, 1.0)),
                                     _float_key(times), (rho, top, bottom, n))
        for n in COSMOS_KNOB_STEPS:   # with the defaults it is the A/B expression itself
            self.assertEqual(_float_key(xmath.sigmas_from_expression(cosmos_expression(7.0, 80.0, 0.002), n, 0.03, 1.0)),
                             _float_key(xmath.sigmas_from_expression(COSMOS_AB_EXPRESSION, n, 0.03, 1.0)))

    def test_scaling_the_range_is_the_flow_time_shift(self):
        """The maths fact the docs state (not a quality claim): t' = k·t/(1 + (k − 1)·t) ⇔ σ̃' = k·σ̃, and scaling both
        σ̃ ends by k scales every ramp value by k — Anima's shift 3 on the default list is σ̃ 0.006 … 240 (float64, to
        rounding)."""
        for k in (3.0, 2.0, 0.5):
            for rho in (7.0, 4.0):
                for n in (2, 3, 28, 151):
                    base = xmath.flow_cosmos_rho7_times(n, rho, 0.002, 80.0)
                    scaled = xmath.flow_cosmos_rho7_times(n, rho, 0.002 * k, 80.0 * k)
                    for t, want in zip(base, scaled):
                        self.assertLessEqual(abs(k * t / (1.0 + (k - 1.0) * t) - want), 1e-13 * want, (k, rho, n))

    def _ext_dynamic(self) -> tuple[Callable[..., Any], Callable[..., Any]]:
        """sam-extra's ``_karras_dynamic_values`` (Karras Dynamic's ramp, shared with Flow Cosmos Dynamic) and
        ``_flow_cosmos_dynamic_times`` cut out of ``schedulers.py`` by AST and run without torch."""
        path = EXT_ROOT / self.SCHEDULERS
        constants = {name: _ext_constant(self.SCHEDULERS, name)
                     for name in ("FLOW_COSMOS_SIGMA_MAX", "FLOW_COSMOS_SIGMA_MIN", "FLOW_COSMOS_RHO")}
        ramp = _cut_function(path, "_karras_dynamic_values", math=math, _positions=self.ext["_positions"])
        times = _cut_function(path, "_flow_cosmos_dynamic_times", _karras_dynamic_values=ramp, **constants)
        return ramp, times

    @staticmethod
    def _ramp_kind(outcome: tuple[str, Any]) -> str:
        if outcome[0] == "error":
            return outcome[1][0]
        return "bad step" if outcome[1][1] is not None else "ok"

    def test_karras_dynamic_ramp(self):
        """The float64 ramp with the per-step exponent ρᵢ = ρ + 2cos(2πi/n) == sam-extra's ``_karras_dynamic_values``:
        the values and the first step that is not finite, also for inputs its callers refuse first (ρ ≤ 2, inf, a
        reversed or tiny range, an overflow) — the raw helper behaves alike everywhere."""
        ramp, _times = self._ext_dynamic()
        kinds = set()
        for rho in (2.0000001, 2.5, 3.6, 4.0, 7.0, 15.0, 30.0, math.inf, 1e300, 2.0, 0.5, -1.0):
            for low, high in ((0.0291675, 14.614642), (0.002, 80.0), (0.0001, 1000.0), (0.9999, 1.0), (1.0, 1.0),
                              (2.0, 1.0), (1e-300, 80.0), (5e-324, 80.0), (0.002, 1e308), (0.002, math.inf),
                              (1e308, 1e-308)):
                for n in (-1, 0, 1, 2, 3, 4, 5, 6, 10, 28, 151):
                    with self.subTest(rho=rho, range=(low, high), n=n):
                        ours = _outcome(lambda: xmath.karras_dynamic_ramp(n, low, high, rho))
                        _assert_same(self, _float_key(ours), _float_key(_outcome(lambda: ramp(n, low, high, rho))),
                                     (rho, low, high, n))
                        kinds.add(self._ramp_kind(ours))
        self.assertTrue({"ok", "bad step", "ZeroDivisionError"} <= kinds, kinds)
        # the pack's Karras Dynamic values are that ramp after its checks (rho 7, a model range)
        for n in (1, 2, 28, 151):
            self.assertEqual(_float_key(xmath.karras_dynamic_values(n, 0.0291675, 14.614642)),
                             _float_key(ramp(n, 0.0291675, 14.614642, 7.0)[0]))

    def test_karras_dynamic_texts_name_the_scheduler_and_the_rho_source(self):
        """Karras Dynamic's checks keep their texts; Flow Cosmos Dynamic on an eps/v model gets the same checks under
        its own name with the node's values as the rho's origin (sam-extra's ``_karras_dynamic(name=, rho_source=)``)."""
        source = xmath.FLOW_COSMOS_VALUES_SOURCE
        cases = ((5, 0.03, 14.6, 2.0), (5, 0.0, 14.6, 7.0), (5, 0.03, math.inf, 7.0), (5, 0.03, 14.6, math.nan))
        for n, low, high, rho in cases:
            with self.subTest(case=(n, low, high, rho)):
                plain = _outcome(lambda: xmath.karras_dynamic_values(n, low, high, rho))
                named = _outcome(lambda: xmath.karras_dynamic_values(n, low, high, rho, name="Flow Cosmos Dynamic",
                                                                     rho_source=source))
                self.assertEqual(plain[0], "error")
                self.assertEqual(named[1][0], plain[1][0])
                self.assertEqual(named[1][1], plain[1][1].replace("Karras Dynamic", "Flow Cosmos Dynamic")
                                 .replace(xmath.RHO_SOURCE, source))
        self.assertEqual(_outcome(lambda: xmath.karras_dynamic_values(5, 0.03, 14.6, 2.0))[1],
                         ("KarrasDynamicError", "Extra Schedulers: Karras Dynamic needs rho above 2 (its per-step "
                                                f"exponent goes down to rho - 2), got 2. Set rho to about 4 or more: "
                                                f"{xmath.RHO_SOURCE}."))
        self.assertEqual(_outcome(lambda: xmath.require_karras_dynamic_rho(1.5, "Flow Cosmos Dynamic", source))[1][1],
                         "Extra Schedulers: Flow Cosmos Dynamic needs rho above 2 (its per-step exponent goes down to "
                         f"rho - 2), got 1.5. Set rho to about 4 or more: {source}.")
        self.assertEqual(_outcome(lambda: xmath.require_flow_cosmos_range(1.0, 1.0, "Flow Cosmos Dynamic"))[1][1],
                         "Extra Schedulers: Flow Cosmos Dynamic needs 0 < sigma~ min < sigma~ max, both finite, got "
                         f"sigma~ min 1 and sigma~ max 1 ({source})")

    def test_flow_cosmos_dynamic_times(self):
        """Flow Cosmos Dynamic's float64 flow times == sam-extra's ``_flow_cosmos_dynamic_times`` (cut out by AST with
        its ``_karras_dynamic_values``): the times and the first step whose σ̃ is not finite, over ρ x σ̃ range x n
        inside the node's ranges and past them; they are s/(1 + s) of the ramp on the σ̃ range; the defaults spelt out
        are the same list."""
        _ramp, ext_times = self._ext_dynamic()
        kinds = set()
        for rho in (*COSMOS_RHOS, 2.0000001, 3.6, 3.9, math.inf, 1e300):
            for top, bottom in (*COSMOS_VE_RANGES, (80.0, 1e-30), (80.0, 1e-300), (80.0, 5e-324), (1e308, 0.002)):
                for n in sorted({*COSMOS_KNOB_STEPS, 4, 8, 100}):
                    with self.subTest(rho=rho, ve=(top, bottom), n=n):
                        ours = _outcome(lambda: xmath.flow_cosmos_dynamic_times(n, rho, bottom, top))
                        theirs = _outcome(lambda: ext_times(n, rho, bottom, top))
                        _assert_same(self, _float_key(ours), _float_key(theirs), (rho, top, bottom, n))
                        kinds.add(self._ramp_kind(ours))
                        if ours[0] == "ok":
                            levels, bad = xmath.karras_dynamic_ramp(n, bottom, top, rho)
                            self.assertEqual(_float_key(ours[1]),
                                             _float_key(([level / (1.0 + level) for level in levels], bad)))
        self.assertTrue({"ok", "bad step"} <= kinds, kinds)
        for n in (0, 1, 2, 28, 151):
            defaults = xmath.flow_cosmos_dynamic_times(n)
            self.assertEqual(_float_key(defaults), _float_key(xmath.flow_cosmos_dynamic_times(n, 7.0, 0.002, 80.0)))
            self.assertEqual(_float_key(defaults), _float_key(ext_times(n)))
        times, bad = xmath.flow_cosmos_dynamic_times(2)
        self.assertIsNone(bad)
        self.assertEqual(times, [80.0 / 81.0, 0.002 / 1.002])   # σ̃ max at p = 0 (ρ₀ = ρ + 2), exactly σ̃ min at p = 1

    def test_flow_cosmos_dynamic_meets_flow_cosmos_rho7_where_the_exponent_is_rho(self):
        """D20's maths, checked on the pack alone (float64 and the float32 the samplers get): ρᵢ = ρ only where
        cos(2πi/n) = 0, so with the same ρ and σ̃ range the list equals Flow Cosmos rho7's in float32 at the first and
        the last step and, when 4 divides n, at n/4 and 3n/4; elsewhere it is lower in the first and last quarter
        (ρᵢ > ρ) and higher in the middle half (a weighted power mean grows with its exponent 1/ρᵢ)."""
        import struct

        def f32(value: float) -> bytes:
            return struct.pack("<f", value)

        for rho in (4.0, 7.0, 15.0):
            for top, bottom in ((80.0, 0.002), (240.0, 0.006), (1000.0, 0.0001), (1.5, 0.75)):
                for n in range(1, 161):
                    with self.subTest(rho=rho, ve=(top, bottom), n=n):
                        dynamic, bad = xmath.flow_cosmos_dynamic_times(n, rho, bottom, top)
                        self.assertIsNone(bad)
                        fixed = xmath.flow_cosmos_rho7_times(n, rho, bottom, top)
                        meets = {0, n - 1} | ({n // 4, 3 * n // 4} if n % 4 == 0 else set())
                        for i in range(n):
                            if i in meets:
                                self.assertEqual(f32(dynamic[i]), f32(fixed[i]), i)
                            elif math.cos(2.0 * math.pi * i / n) > 0.0:
                                self.assertLess(dynamic[i], fixed[i], i)
                            else:
                                self.assertGreater(dynamic[i], fixed[i], i)

    def test_positions(self):
        for n in (-3, 0, 1, 2, 3, 7, 28, 150, 1001):
            self.assertEqual(_float_key(xmath.positions(n)), _float_key(self.ext["_positions"](n)), n)

    def test_cosine_and_exponential_values(self):
        ranges = ((0.0291675, 14.614642), (0.0029, 1.0), (0.002, 1.0), (1e-4, 80.0), (0.5, 0.5), (2.0, 1.0))
        for n in (1, 2, 5, 28, 150):
            for sigma_min, sigma_max in ranges:
                for p in xmath.positions(n):
                    with self.subTest(n=n, p=p, range=(sigma_min, sigma_max)):
                        self.assertEqual(repr(xmath.cosine_value(p, sigma_min, sigma_max)),
                                         repr(self.ext["_cosine_value"](p, sigma_min, sigma_max)))
                        self.assertEqual(repr(xmath.exponential_value(p, sigma_min, sigma_max)),
                                         repr(self.ext["_exponential_value"](p, sigma_min, sigma_max)))

    def test_react_cosinusoidal_dynsf_values(self):
        """The float64 values (and the positive-range error) == sam-extra's ``react_cosinusoidal_dynsf``
        before it builds its float32 tensor: the function is cut out by AST with its tensor builder
        replaced by a plain list and its flat-step check by nothing (``SchedulerParityTests`` compares
        the float32 schedules with both)."""
        path = EXT_ROOT / self.SCHEDULERS
        check_positive = _cut_function(path, "_check_positive", ExtraSchedulerError=xmath.ExtraSchedulerError)
        ext_react = _cut_function(
            path, "react_cosinusoidal_dynsf", math=math, _positions=self.ext["_positions"],
            _check_positive=check_positive,
            REACT_DYNSF_FACTOR=_ext_constant(self.SCHEDULERS, "REACT_DYNSF_FACTOR"),
            _with_final_zero=lambda values, device: [*map(float, values), 0.0],
            _check_no_flat_step=lambda *args: None,
        )
        ranges = ((0.0291675, 14.614642), (0.0029, 1.0), (0.002, 1.0), (1e-4, 80.0), (0.5, 0.5), (2.0, 1.0),
                  (0.0, 1.0), (1.0, 0.0), (-1.0, 1.0), (0.0, 0.0))
        outcomes = set()
        for n in (-1, 0, 1, 2, 3, 5, 28, 50, 150, 447, 448, 1000):
            for sigma_min, sigma_max in ranges:
                got = _outcome(lambda: [*xmath.react_cosinusoidal_dynsf_values(n, sigma_min, sigma_max), 0.0])
                want = _outcome(lambda: ext_react(n, sigma_min, sigma_max))
                self.assertEqual(_float_key(got), _float_key(want), (n, sigma_min, sigma_max))
                outcomes.add(got[0] if got[0] == "ok" else got[1][0])
        self.assertEqual(outcomes, {"ok", "ExtraSchedulerError"})

    def test_laplace_position(self):
        sigmas = (0.0, -1.0, 1e-30, 0.0029, 0.0291675, 0.5, 1.0, 2.0, 14.614642, 1e30)
        for mu in (-10.0, -2.0, -0.5, 0.0, 0.5, 2.0, 10.0):
            for beta in (0.1, 0.5, 1.0, 3.0, 10.0):
                for sigma in sigmas:
                    self.assertEqual(
                        _float_key(_outcome(lambda: xmath.laplace_position(sigma, mu, beta))),
                        _float_key(_outcome(lambda: self.ext["_laplace_position"](sigma, mu, beta))),
                        (sigma, mu, beta))

    def test_laplace_curve_ends(self):
        for mu in (-10.0, 0.0, 0.5, 10.0):
            for beta in (0.0, 0.5, 10.0):
                top = math.exp(mu - beta * math.log(self.epsilon))
                bottom = math.exp(mu + beta * math.log(self.epsilon))
                self.assertEqual(_float_key(xmath.laplace_curve_ends(mu, beta)), _float_key((top, bottom)))

    def test_sigmas_from_expression(self):
        texts = (xmath.DEFAULT_EXPRESSION, "m + (M - m) * (1 - x) ** 3", "M - s", "m * (1 - x)", "x", "-1",
                 "M * (m / M) ** x - m", "1e-300 * x + 1e-320", "log(0)", "", "y", "M / (n - 1 - s)")
        for text in texts:
            for steps in (0, 1, 2, 5, 28):
                for sigma_min, sigma_max in ((0.0291675, 14.614642), (0.002, 1.0), (1.0, 1.0)):
                    with self.subTest(text=text, steps=steps):
                        self.assertEqual(
                            _float_key(_outcome(lambda: xmath.sigmas_from_expression(text, steps, sigma_min,
                                                                                     sigma_max))),
                            _float_key(_outcome(lambda: self.ext["sigmas_from_expression"](text, steps, sigma_min,
                                                                                           sigma_max))))

    def test_error_classes(self):
        self.assertTrue(issubclass(xmath.CustomSchedulerError, xmath.ExtraSchedulerError))
        self.assertTrue(issubclass(xmath.KarrasDynamicError, xmath.ExtraSchedulerError))
        self.assertTrue(issubclass(xmath.ExtraSchedulerError, ValueError))
        tree = ast.parse((EXT_ROOT / self.SCHEDULERS).read_text(encoding="utf-8"))
        bases = {item.name: [getattr(base, "id", None) for base in item.bases]
                 for item in tree.body if isinstance(item, ast.ClassDef)}
        self.assertEqual(bases, {
            "ExtraSchedulerError": ["ValueError"],
            "CustomSchedulerError": ["ExtraSchedulerError"],
            "KarrasDynamicError": ["ExtraSchedulerError"],
        })


# ── float32 schedules and handlers (PB-XS3) ───────────────────────────────────────────────────
_SCHEDULE_STEPS = (1, 2, 3, 4, 5, 8, 20, 28, 50, 150)
_MU_GRID = (-2.0, -0.5, 0.0, 0.5, 2.0)
_BETA_GRID = (0.0, 0.1, 0.5, 1.0, 3.0, 10.0)
_RHO_GRID = (2.0, 3.5, 4.0, 7.0, 12.0)
_CUSTOM_EXPRESSIONS = (
    xmath.DEFAULT_EXPRESSION, "m + (M - m) * (1 - x) ** 3", "M - s", "m * (1 - x)", "x", "max(m, M - s)",
    "M * (1 - x) + m * x", "m + (M - m) * cos(pi / 2 * x)", "1e39", "1e-39 + 0 * x", "1e-40", "M * 1e-30 ** x",
    "m + 1e-9 * (1 - x)", "log(0)", "", "nope",
)
_CUSTOM_LISTS = (
    xmath.DEFAULT_SIGMAS, "[14.6, 7, 3.1, 1.2, 0.03]", "14.6 7 3 0", "[1.0, 0.5, 0.0]", "0.5, 0.5", "[1, 2",
    "2, 1, 0.5, 0.25", "1e39 1", "1e-40 1e-41", "1 0 0", "[1.0, 0.99999999, 0.0]",
)

# Flow Cosmos rho7 (sam-extra v0.33.0). The custom expression of the user's GPU A/B on Anima: on a flow model the
# scheduler must give its numbers bit for bit.
COSMOS_AB_EXPRESSION = "(80**(1/7)+x*(0.002**(1/7)-80**(1/7)))**7/(1+(80**(1/7)+x*(0.002**(1/7)-80**(1/7)))**7)"
COSMOS_STEPS = (1, 2, 3, 28, 50, 150)
# sam-extra's documented float32 limit: the first two equal neighbouring flow times (steps 467/468, t ≈ 0.987626).
COSMOS_FIRST_FLAT_STEPS = 1_119_764
# Flow Cosmos rho7's values (D19): a grid of ρ within the node's 1 … 15 and (σ̃ max, σ̃ min) ranges — Cosmos's, ×3
# (Anima's shift 3), the slider corners, SDXL's model range, a narrow one, the narrowest the sliders allow.
COSMOS_RHOS = (1.0, 2.5, 5.0, 7.0, 9.3, 15.0)
COSMOS_VE_RANGES = ((80.0, 0.002), (240.0, 0.006), (1.0, 0.0001), (1000.0, 1.0), (1000.0, 0.0001),
                    (14.614642, 0.0291675), (1.5, 0.75), (1.0, 0.9999))
COSMOS_KNOB_STEPS = (0, 1, 2, 3, 10, 28, 151)


def cosmos_expression(rho: float, sigma_max: float, sigma_min: float) -> str:
    """The A/B custom expression with ρ, σ̃ max and σ̃ min in place of 7, 80 and 0.002 (the same operations)."""
    root_max, root_min = f"{sigma_max!r}**(1/{rho!r})", f"{sigma_min!r}**(1/{rho!r})"
    ramp = f"({root_max}+x*({root_min}-{root_max}))**{rho!r}"
    return f"{ramp}/(1+{ramp})"


def upstream_flow_cosmos_rho(steps: int, order: float = 7.0, sigma_max: float = 80.0,
                             sigma_min: float = 0.002) -> list[float]:
    """TEST REFERENCE ONLY — never shipped in the pack. The numbers of KeithZ117/Comfyui-anima-sampler @ effba3c5
    (MIT), ``src/animasampler/cosmos_schedules.py`` ``build_flow_cosmos_rho_sigmas``, with its helpers
    ``rho_space_descending(start, end, count, order)`` (the two ends' ``order``-th roots, a constant step from the
    first to the second over ``count - 1`` intervals, each point to the power ``order``) and ``_cosmos_rflow_time``
    (``t = sigma / (sigma + 1)``, 0 for ``sigma <= 0``, sigma clamped to ``sigma_max``) written out here in that
    operation order (steps >= 2), then the final 0. The pack's code is written from the formula instead."""
    start_root, end_root = sigma_max ** (1 / order), sigma_min ** (1 / order)
    step = (end_root - start_root) / (steps - 1)
    times = []
    for i in range(steps):
        sigma = (start_root + step * i) ** order
        if sigma <= 0.0:
            times.append(0.0)
            continue
        sigma = min(float(sigma), float(sigma_max))
        times.append(sigma / (sigma + 1.0))
    return times + [0.0]


def _first_difference(got: Any, want: Any, path: tuple = ()) -> tuple[tuple, Any, Any]:
    """Where two nested tuples/lists first differ: (index path, got's item, want's item)."""
    if type(got) is not type(want) or not isinstance(got, (tuple, list)):
        return path, got, want
    if len(got) != len(want):
        return (*path, "len"), len(got), len(want)
    for index, (left, right) in enumerate(zip(got, want)):
        if left != right:
            return _first_difference(left, right, (*path, index))
    return path, got, want


def _assert_same(test: unittest.TestCase, got: Any, want: Any, label: Any) -> None:
    """``got == want`` exactly, as ``assertEqual`` checks it, with a short message on failure: unittest would diff
    the full reprs of two long schedules (minutes for each failing comparison)."""
    if got == want:
        return
    where, left, right = _first_difference(got, want)
    test.fail(f"{label}: first difference at {where}: {repr(left)[:300]} != {repr(right)[:300]}")


def _forge_model(kind: str | None) -> Any:
    """What Forge hands a ``need_inner_model`` scheduler, reduced to what sam-extra reads (``model_wrap.predictor.
    prediction_type``); ``None`` = no model."""
    if kind is None:
        return None
    return types.SimpleNamespace(predictor=types.SimpleNamespace(prediction_type=kind))


def _karras_host_function() -> tuple[Callable[..., Any], str]:
    """What the pack's Flow Cosmos rho7 calls on an eps/v model inside ComfyUI: ``comfy.k_diffusion.sampling.
    get_sigmas_karras`` (with its ``append_zero``) cut out of ``AISTUDIO_COMFY_TEST_ROOT``'s ComfyUI; without one,
    Forge's own (the same function body — compared by AST when both exist)."""
    if COMFY_ROOT:
        path = Path(COMFY_ROOT) / "comfy" / "k_diffusion" / "sampling.py"
        append_zero = _cut_function(path, "append_zero", torch=torch)
        return _cut_function(path, "get_sigmas_karras", torch=torch, append_zero=append_zero), "ComfyUI"
    return forge_k_sampling().get_sigmas_karras, "Forge"


def _flow_sigma_range(shift: float) -> tuple[float, float]:
    """A flow model's ``(sigma_min, sigma_max)`` as ComfyUI's ``ModelSamplingDiscreteFlow`` holds them
    (and Forge's ``PredictionDiscreteFlow``): ``time_snr_shift`` over ``arange(1..1000)/1000``, float32."""
    t = ((torch.arange(1, 1001, 1) / 1000) * 1000) / 1000
    sigmas = t if shift == 1.0 else shift * t / (1 + (shift - 1) * t)
    return float(sigmas[0]), float(sigmas[-1])


def _sigma_ranges() -> list[tuple[float, float]]:
    import numpy as np

    sdxl = (float(np.float32(0.0291675)), float(np.float32(14.614642)))
    return [_flow_sigma_range(1.0), _flow_sigma_range(3.0), _flow_sigma_range(5.0), sdxl, (0.002, 80.0)]


def _tensor_key(value: Any) -> Any:
    """A tensor by dtype, shape, device and its exact bits; anything else as ``_float_key``."""
    if hasattr(value, "dtype") and hasattr(value, "view"):
        bits = value.contiguous().view(torch.int32) if value.dtype == torch.float32 else value
        return ("tensor", str(value.dtype), tuple(value.shape), str(value.device), tuple(bits.tolist()))
    if isinstance(value, tuple):
        return tuple(_tensor_key(item) for item in value)
    return _float_key(value)


def _laplace_host_function() -> tuple[Callable[..., Any], str]:
    """What the pack's Laplace calls in ComfyUI: ``comfy.k_diffusion.sampling.get_sigmas_laplace`` cut
    out of ``AISTUDIO_COMFY_TEST_ROOT``'s ComfyUI; without one, sam-extra's unchanged copy of it."""
    if COMFY_ROOT:
        path = Path(COMFY_ROOT) / "comfy" / "k_diffusion" / "sampling.py"
        return _cut_function(path, "get_sigmas_laplace", torch=torch), "ComfyUI"
    return _ext("schedulers").get_sigmas_laplace, "sam-extra copy"


@requires_torch
@requires_extension
class SchedulerParityTests(unittest.TestCase):
    """Every pack schedule == the extension's function, bit for bit (float32, CPU), errors alike."""

    @classmethod
    def setUpClass(cls):
        from tests._optional_deps import bind_torch

        bind_torch(globals())
        from comfy_custom_nodes.ai_studio_forge_parity import extra_schedulers

        cls.pack = extra_schedulers
        cls.ext = _ext("schedulers")
        cls.ext_settings = _ext("settings")
        cls.ext_sigma_list = _ext("sigma_list")
        cls.interp = staticmethod(forge_loglinear_interp())
        plain, cls.plain_source = _laplace_host_function()
        cls.plain = staticmethod(plain)
        cls.ranges = _sigma_ranges()

    def tearDown(self):
        self.ext_settings.reset()

    def _ext_with(self, settings: xmath.Settings, call: Callable[[], Any]) -> Any:
        """Run an extension scheduler with ``settings`` active and Forge's interpolation in reach."""
        from unittest import mock

        self.ext_settings.set_active(self.ext_settings.ExtraSchedulerSettings(**settings._asdict()))
        with mock.patch.object(self.ext_sigma_list, "forge_loglinear_interp", return_value=self.interp):
            return call()

    def _compare(self, got_call, want_call, label) -> str:
        got, want = _outcome(got_call), _outcome(want_call)
        self.assertEqual(_tensor_key(got), _tensor_key(want), label)
        return got[0]

    def test_constants_match(self):
        self.assertEqual(self.pack.FLOAT32_MAX, float(torch.finfo(torch.float32).max))
        self.assertEqual(self.pack.FLOAT32_TINY, float(torch.finfo(torch.float32).tiny))
        self.assertEqual(xmath.ZERO_LENGTH_STEP, _ext_constant(ClosedFormTests.SCHEDULERS, "ZERO_LENGTH_STEP"))
        relative = "sam3ext/extra_schedulers/registry.py"
        tree = ast.parse((EXT_ROOT / relative).read_text(encoding="utf-8"))
        specs = [node for node in ast.walk(tree) if isinstance(node, ast.Call)
                 and getattr(node.func, "id", None) == "SchedulerSpec"]

        def literal(node: ast.expr) -> Any:
            # v0.33.0 passes React's label as the module literal LABEL_REACT_DYNSF
            if isinstance(node, ast.Name):
                return _ext_constant(relative, node.id)
            return ast.literal_eval(node)

        names_labels = [(literal(spec.args[0]), literal(spec.args[1])) for spec in specs]
        self.assertEqual(names_labels, [(name, self.pack.LABELS[name]) for name in self.pack.SCHEDULER_NAMES])
        self.assertEqual(self.pack.LABELS, {name: label for name, label in names_labels})
        self.assertEqual(len(self.pack.SCHEDULER_NAMES), 9)
        self.assertEqual(self.pack.LABELS["react_cosinusoidal_dynsf"], _ext_constant(relative, "LABEL_REACT_DYNSF"))
        self.assertEqual(self.pack.LABELS["flow_cosmos_rho7"], _ext_constant(relative, "LABEL_FLOW_COSMOS_RHO7"))
        self.assertEqual(self.pack.LABELS["flow_cosmos_dynamic"], _ext_constant(relative, "LABEL_FLOW_COSMOS_DYNAMIC"))
        # appended after React DynSF: Flow Cosmos rho7, then Flow Cosmos Dynamic right after it
        self.assertEqual(self.pack.SCHEDULER_NAMES[-3:], ("react_cosinusoidal_dynsf", "flow_cosmos_rho7",
                                                          "flow_cosmos_dynamic"))
        # sam-extra's two Flow Cosmos schedulers need Forge's model (flow or eps/v) and have no default_rho: their rho
        # is the accordion's, so neither Forge's Settings rho nor a pasted "Schedule rho" reaches them — nothing to
        # mirror on ComfyUI (the node's flow_cosmos_rho is that accordion value).
        for name in ("flow_cosmos_rho7", "flow_cosmos_dynamic"):
            with self.subTest(name=name):
                cosmos = next(spec for spec in specs if literal(spec.args[0]) == name)
                keywords = {keyword.arg: ast.literal_eval(keyword.value) for keyword in cosmos.keywords}
                self.assertEqual(keywords, {"need_inner_model": True})

    def test_cosine_cosine_exponential_and_phi(self):
        ranges = [*self.ranges, (0.0, 1.0), (1.0, 1.0), (2.0, 1.0), (-1.0, 1.0)]
        pairs = (("cosine", self.ext.cosine), ("cosine_exponential", self.ext.cosine_exponential_blend),
                 ("phi", self.ext.phi))
        outcomes = set()
        for name, ext_function in pairs:
            pack_function = getattr(self.pack, name)
            for n in (0, *_SCHEDULE_STEPS):
                for sigma_min, sigma_max in ranges:
                    outcomes.add(self._compare(lambda: pack_function(n, sigma_min, sigma_max),
                                               lambda: ext_function(n, sigma_min, sigma_max),
                                               (name, n, sigma_min, sigma_max)))
                    # through the shared entry point too
                    self._compare(lambda: self.pack.schedule(name, n, sigma_min, sigma_max),
                                  lambda: ext_function(n, sigma_min, sigma_max), (name, "schedule", n))
        self.assertEqual(outcomes, {"ok", "error"})   # the grid reaches the guards too

    def test_react_cosinusoidal_dynsf(self):
        """Bitwise on the Anima/SDXL ranges (n 1, 2, 3, 28, 50, 150 and the rest of the grid), through
        the function and ``schedule``; the float32 flat-step limit (n 448 on Anima's range, 434 on SDXL's
        float32 range — 489 on its decimal one, as sam-extra's docstring counts — and 1000) and the
        positive-range check raise alike."""
        pack_function, ext_function = self.pack.react_cosinusoidal_dynsf, self.ext.react_cosinusoidal_dynsf
        anima, sdxl = _flow_sigma_range(3.0), self.ranges[3]
        outcomes = set()
        for n in (0, *_SCHEDULE_STEPS):
            for sigma_min, sigma_max in [*self.ranges, (0.0, 1.0), (1.0, 1.0), (2.0, 1.0), (-1.0, 1.0)]:
                outcomes.add(self._compare(lambda: pack_function(n, sigma_min, sigma_max),
                                           lambda: ext_function(n, sigma_min, sigma_max),
                                           ("react", n, sigma_min, sigma_max)))
                self._compare(lambda: self.pack.schedule("react_cosinusoidal_dynsf", n, sigma_min, sigma_max),
                              lambda: ext_function(n, sigma_min, sigma_max), ("react", "schedule", n))
        self.assertEqual(outcomes, {"ok", "error"})
        limits = {}
        for sigma_min, sigma_max in (anima, sdxl):
            for n in (433, 434, 447, 448, 488, 489, 490, 508, 509, 1000):
                limits[(sigma_min, n)] = self._compare(lambda: pack_function(n, sigma_min, sigma_max),
                                                       lambda: ext_function(n, sigma_min, sigma_max),
                                                       ("react limit", n, sigma_min, sigma_max))
        self.assertEqual((limits[(anima[0], 447)], limits[(anima[0], 448)]), ("ok", "error"))
        self.assertEqual((limits[(sdxl[0], 433)], limits[(sdxl[0], 434)]), ("ok", "error"))
        self.assertEqual((limits[(sdxl[0], 1000)], limits[(anima[0], 1000)]), ("error", "error"))
        error = _outcome(lambda: pack_function(1000, *sdxl))[1]
        self.assertEqual(error[0], "ExtraSchedulerError")
        self.assertTrue(error[1].startswith("Extra Schedulers: React Cosinusoidal DynSF gives sigma "), error[1])
        # the shape sam-extra documents for Anima (shift 3, 28 steps)
        values = [float(value) for value in pack_function(28, *anima)[:-1]]
        self.assertEqual(sum(value > 0.5 for value in values), 17)
        self.assertEqual(sum(value < 0.05 for value in values), 4)
        self.assertEqual(round(sorted(values)[len(values) // 2], 3), 0.720)   # the upper median, as documented
        self.assertAlmostEqual(values[-1], 3.749e-6, delta=0.001e-6)

    def test_karras_dynamic_rho_grid(self):
        ranges = [*self.ranges, (1.0, 1.0), (2.0, 1.0), (0.0, 1.0)]
        outcomes = set()
        for rho in (*_RHO_GRID, 1.0, -3.0, 3.6, 3.85):
            for n in _SCHEDULE_STEPS:
                for sigma_min, sigma_max in ranges:
                    outcomes.add(self._compare(
                        lambda: self.pack.karras_dynamic(n, sigma_min, sigma_max, rho=rho),
                        lambda: self.ext.karras_dynamic(n, sigma_min, sigma_max, rho=rho),
                        (rho, n, sigma_min, sigma_max)))
        for n in _SCHEDULE_STEPS:   # the default rho 7, also through the handler entry point
            for sigma_min, sigma_max in self.ranges:
                self._compare(lambda: self.pack.schedule("karras_dynamic", n, sigma_min, sigma_max),
                              lambda: self.ext.karras_dynamic(n, sigma_min, sigma_max), ("default rho", n))
        self.assertEqual(outcomes, {"ok", "error"})

    def test_laplace_mu_beta_grid(self):
        outcomes = []
        ranges = [*self.ranges, (0.5, 0.5)]
        for mu in _MU_GRID:
            for beta in _BETA_GRID:
                settings = xmath.coerce({"laplace_mu": mu, "laplace_beta": beta})
                for n in _SCHEDULE_STEPS:
                    for sigma_min, sigma_max in ranges:
                        outcomes.append(self._compare(
                            lambda: self.pack.laplace(n, sigma_min, sigma_max, settings, plain=self.plain),
                            lambda: self._ext_with(settings, lambda: self.ext.laplace(n, sigma_min, sigma_max)),
                            (mu, beta, n, sigma_min, sigma_max, self.plain_source)))
        self.assertIn("ok", outcomes)
        self.assertIn("error", outcomes)

    def test_laplace_covers_the_plain_the_spread_and_the_flat_paths(self):
        """Flow range, mu 0: the clamp repeats sigma_max -> spread; beta 0 -> one sigma -> error; SDXL
        range -> ComfyUI's schedule unchanged."""
        flow = _flow_sigma_range(3.0)
        sdxl = self.ranges[3]
        spread_case = self.pack.laplace(28, *flow, xmath.DEFAULTS, plain=self.plain)
        plain_flow = torch.cat([self.plain(28, *flow, mu=0.0, beta=0.5), torch.zeros(1)])
        self.assertIsNotNone(self.pack.first_flat_step(plain_flow))
        self.assertIsNone(self.pack.first_flat_step(spread_case))
        plain_sdxl = self.plain(28, *sdxl, mu=0.0, beta=0.5)
        self.assertTrue(torch.equal(self.pack.laplace(28, *sdxl, xmath.DEFAULTS, plain=self.plain)[:-1], plain_sdxl))
        flat = _outcome(lambda: self.pack.laplace(28, *flow, xmath.coerce({"laplace_beta": 0}), plain=self.plain))
        self.assertEqual((flat[0], flat[1][0]), ("error", "ExtraSchedulerError"))

    def test_custom_expression_and_sigma_list(self):
        outcomes = []
        cases = [xmath.coerce({"custom_expression": text}) for text in _CUSTOM_EXPRESSIONS]
        cases += [xmath.coerce({"custom_mode": "sigmas", "custom_sigmas": text}) for text in _CUSTOM_LISTS]
        for settings in cases:
            for n in (0, *_SCHEDULE_STEPS):
                for sigma_min, sigma_max in [*self.ranges, (0.5, 0.5)]:
                    outcomes.append(self._compare(
                        lambda: self.pack.custom(n, sigma_min, sigma_max, settings, interp=self.interp),
                        lambda: self._ext_with(settings, lambda: self.ext.custom(n, sigma_min, sigma_max)),
                        (settings, n, sigma_min, sigma_max)))
        self.assertIn("ok", outcomes)
        self.assertIn("error", outcomes)
        errors = set()
        for settings in cases:
            outcome = _outcome(lambda: self.pack.custom(8, 0.03, 14.6, settings, interp=self.interp))
            if outcome[0] == "error":
                errors.add(outcome[1][0])
        self.assertEqual(errors, {"CustomSchedulerError"})

    def test_laplace_host_function_is_sam_extra_s_copy(self):
        """sam-extra copied ComfyUI's ``get_sigmas_laplace`` unchanged; the pack calls ComfyUI's own."""
        if not COMFY_ROOT:
            self.skipTest("Set AISTUDIO_COMFY_TEST_ROOT to compare ComfyUI's get_sigmas_laplace with sam-extra's copy")
        comfy = Path(COMFY_ROOT) / "comfy" / "k_diffusion" / "sampling.py"
        self.assertEqual(_function_body_dump(comfy, "get_sigmas_laplace"),
                         _function_body_dump(EXT_ROOT / ClosedFormTests.SCHEDULERS, "get_sigmas_laplace"))


@requires_torch
@requires_extension
class FlowCosmosRho7ParityTests(unittest.TestCase):
    """``flow_cosmos_rho7`` == sam-extra v0.33.0's, bit for bit (float32, CPU), on flow and eps/v models, errors
    alike. sam-extra's eps/v branch runs Forge's ``k_diffusion.sampling.get_sigmas_karras`` (Forge's file,
    executed); the pack's runs the host function (``_karras_host_function``)."""

    OTHER_KINDS = ("epsilon", "v_prediction")

    @classmethod
    def setUpClass(cls):
        from tests._optional_deps import bind_torch

        bind_torch(globals())
        from comfy_custom_nodes.ai_studio_forge_parity import extra_schedulers

        cls.pack = extra_schedulers
        cls.ext = _ext("schedulers")
        cls.ext_settings = _ext("settings")
        cls.forge_k = forge_k_sampling()
        karras, cls.karras_source = _karras_host_function()
        cls.karras = staticmethod(karras)
        cls.ranges = _sigma_ranges()
        cls.anima = _flow_sigma_range(3.0)

    def setUp(self):
        # one log line per process on both sides: kept quiet here, except in the test of that line
        for module, name in ((self.pack, "_flow_cosmos_not_flow_logged"), (self.ext, "_flow_cosmos_not_flow_logged")):
            patcher = mock.patch.object(module, name, True)
            patcher.start()
            self.addCleanup(patcher.stop)

    def tearDown(self):
        self.ext_settings.reset()

    @staticmethod
    def _sampling(kind: str) -> Any:
        """A model_sampling the pack's flow rule reads (``extra_samplers_dy.is_const``: ``prediction_type``)."""
        return types.SimpleNamespace(prediction_type=kind)

    def _pack(self, n, sigma_min, sigma_max, kind, settings: xmath.Settings | None = None):
        return self.pack.flow_cosmos_rho7(n, sigma_min, sigma_max, self._sampling(kind), settings, karras=self.karras)

    def _ext(self, n, sigma_min, sigma_max, kind, settings: xmath.Settings | None = None):
        """sam-extra with ``settings`` active (its scheduler reads ``settings.active()`` when it runs); the values go
        in as they are — not coerced — so values past the sliders reach both sides alike."""
        values = (settings if settings is not None else xmath.DEFAULTS)._asdict()
        self.ext_settings.set_active(self.ext_settings.ExtraSchedulerSettings(**values))
        with installed_k_sampling(self.forge_k):
            return self.ext.flow_cosmos_rho7(n=n, sigma_min=sigma_min, sigma_max=sigma_max,
                                             inner_model=_forge_model(kind), device="cpu")

    def _compare(self, n, sigma_min, sigma_max, kind, settings: xmath.Settings | None = None) -> str:
        got = _outcome(lambda: self._pack(n, sigma_min, sigma_max, kind, settings))
        want = _outcome(lambda: self._ext(n, sigma_min, sigma_max, kind, settings))
        label = (kind, n, sigma_min, sigma_max, settings, self.karras_source)
        _assert_same(self, _tensor_key(got), _tensor_key(want), label)
        if got[0] == "ok":
            self.assertTrue(torch.equal(got[1], want[1]), label)
            self.assertEqual(got[1].dtype, torch.float32, label)
        return got[0] if got[0] == "ok" else got[1][0]

    def test_flow_models_get_the_extension_s_flow_list(self):
        """Every sigma range (the list does not depend on it — Anima's, shift 1/5, SDXL's, Cosmos' EDM range, a
        reversed one) and the n grid incl. 0 and 1; non-positive ranges raise alike; through ``schedule`` too."""
        ranges = [*self.ranges, (2.0, 1.0), (0.0, 1.0), (-1.0, 1.0), (0.003, 0.0), (0.0, 0.0)]
        outcomes = set()
        reference = {}
        for n in sorted({0, *_SCHEDULE_STEPS, *COSMOS_STEPS, 300, 1000}):
            for sigma_min, sigma_max in ranges:
                outcomes.add(self._compare(n, sigma_min, sigma_max, "const"))
                if sigma_min > 0.0 and sigma_max > 0.0:
                    got = self.pack.schedule("flow_cosmos_rho7", n, sigma_min, sigma_max,
                                             model_sampling=self._sampling("const"))
                    self.assertTrue(torch.equal(got, reference.setdefault(n, got)), (n, sigma_min, sigma_max))
        self.assertEqual(outcomes, {"ok", "ExtraSchedulerError"})
        error = _outcome(lambda: self._pack(28, 0.0, 1.0, "const"))[1]
        self.assertEqual(error, ("ExtraSchedulerError", "Extra Schedulers: Flow Cosmos rho7 needs sigma_min and "
                                                        "sigma_max above 0 (got 0 and 1)"))
        top = float(torch.tensor(80.0 / 81.0, dtype=torch.float32))
        self.assertEqual(reference[1].tolist(), [top, 0.0])          # n == 1: p = 0 like the other schedulers
        self.assertEqual(reference[0].tolist(), [0.0])
        self.assertEqual(float(reference[28][0]), top)
        values = reference[28][:-1]
        self.assertEqual((int((values > 0.5).sum()), int((values < 0.05).sum())), (17, 5))   # as sam-extra documents

    def test_flow_list_is_the_ab_expression_and_the_upstream_numbers(self):
        """The GPU A/B's custom expression through the pack's ``custom`` scheduler (another code path) gives the
        same float32 list on every flow range incl. Anima's; the upstream builder's numbers (inline reference) too."""
        settings = xmath.coerce({"custom_expression": COSMOS_AB_EXPRESSION})
        for sigma_min, sigma_max in (self.anima, _flow_sigma_range(1.0), _flow_sigma_range(5.0), (0.05, 0.9)):
            for n in sorted({*COSMOS_STEPS, *range(1, 201)}):
                with self.subTest(range=(sigma_min, sigma_max), n=n):
                    flow = self._pack(n, sigma_min, sigma_max, "const")
                    self.assertTrue(torch.equal(self.pack.custom(n, sigma_min, sigma_max, settings), flow))
        # float32 equal for every n here (sam-extra checked up to 30,000); in float64 the upstream builder's
        # i·((r_min − r_max)/(n − 1)) and the expression's x·(r_min − r_max) may differ in the last bit (D18)
        for n in sorted({*COSMOS_STEPS[1:], *range(2, 1201)}):
            with self.subTest(upstream=n):
                expected = torch.tensor(upstream_flow_cosmos_rho(n), dtype=torch.float32)
                self.assertTrue(torch.equal(self._pack(n, *self.anima, "const"), expected))

    def test_eps_and_v_models_get_karras_rho_7_on_their_own_range(self):
        """Not a flow model: the host's Karras with ρ 7 on the range passed (the MODEL's) == sam-extra running
        Forge's Karras == Forge's/the host's Karras called directly; flat and non-positive ranges raise alike."""
        sdxl = self.ranges[3]
        ranges = [sdxl, (0.1, 20.0), (0.0291675, 1.0), (0.002, 80.0), self.anima, (2.0, 1.0),
                  (2.0, 2.0), (0.0, 1.0), (0.003, 0.0)]
        outcomes = set()
        for kind in self.OTHER_KINDS:
            for n in (0, *_SCHEDULE_STEPS, 300):
                for sigma_min, sigma_max in ranges:
                    outcome = self._compare(n, sigma_min, sigma_max, kind)
                    outcomes.add(outcome)
                    if outcome == "ok":
                        got = self._pack(n, sigma_min, sigma_max, kind)
                        self.assertTrue(torch.equal(got, self.karras(n, sigma_min, sigma_max)))
                        self.assertTrue(torch.equal(got, self.forge_k.get_sigmas_karras(n, sigma_min, sigma_max,
                                                                                        rho=7.0)))
        self.assertEqual(outcomes, {"ok", "ExtraSchedulerError"})
        flat = _outcome(lambda: self._pack(10, 2.0, 2.0, "epsilon"))[1]
        self.assertEqual(flat[0], "ExtraSchedulerError")
        self.assertTrue(flat[1].startswith("Extra Schedulers: Flow Cosmos rho7 gives sigma 2 at steps 0 and 1 "),
                        flat[1])
        calls = []

        def recording(*args, **kwargs):
            calls.append((args, kwargs))
            return self.karras(*args, **kwargs)

        self.pack.flow_cosmos_rho7(5, *sdxl, self._sampling("epsilon"), karras=recording)
        self.assertEqual(calls, [((5, sdxl[0], sdxl[1]), {"rho": 7.0, "device": "cpu"})])

    def test_float32_limit_on_flow_models(self):
        """sam-extra's documented limit: from 1,119,764 steps two neighbouring float32 flow times are equal —
        refused alike, with the same message; one step fewer is the same list on both sides."""
        below = COSMOS_FIRST_FLAT_STEPS - 1
        got, want = self._pack(below, *self.anima, "const"), self._ext(below, *self.anima, "const")
        self.assertTrue(torch.equal(got, want))
        self.assertTrue(bool((got[1:] < got[:-1]).all()))
        got = _outcome(lambda: self._pack(COSMOS_FIRST_FLAT_STEPS, *self.anima, "const"))
        want = _outcome(lambda: self._ext(COSMOS_FIRST_FLAT_STEPS, *self.anima, "const"))
        _assert_same(self, _tensor_key(got), _tensor_key(want), COSMOS_FIRST_FLAT_STEPS)
        self.assertEqual(got[0], "error")
        self.assertEqual(got[1][0], "ExtraSchedulerError")
        self.assertTrue(got[1][1].startswith("Extra Schedulers: Flow Cosmos rho7 gives sigma 0.987626 at steps 467 "
                                             "and 468 "), got[1][1])

    def test_without_a_model_sampling_it_stops(self):
        """Neither side guesses: no model -> ExtraSchedulerError (each host names what it needs)."""
        got = _outcome(lambda: self.pack.schedule("flow_cosmos_rho7", 10, 0.03, 1.0))
        self.assertEqual(got, ("error", ("ExtraSchedulerError", "Extra Schedulers: Flow Cosmos rho7 needs the MODEL's "
                                         "model_sampling to tell a flow model from an eps/v one; none was given")))
        want = _outcome(lambda: self._ext(10, 0.03, 1.0, None))
        self.assertEqual(want[1][0], "ExtraSchedulerError")
        self.assertIn("Flow Cosmos rho7 needs Forge's model", want[1][1])
        # the positive-range check comes first on both sides
        self.assertEqual(_outcome(lambda: self.pack.schedule("flow_cosmos_rho7", 10, 0.0, 1.0))[1],
                         _outcome(lambda: self._ext(10, 0.0, 1.0, None))[1])

    def test_without_the_host_karras_an_eps_model_stops_and_a_flow_model_does_not_need_it(self):
        flow_list = self._pack(10, 0.03, 1.0, "const")
        missing = types.ModuleType("comfy.k_diffusion.sampling")
        broken = types.ModuleType("comfy.k_diffusion.sampling")
        broken.get_sigmas_karras = None
        for case, module in (("no module", None), ("no function", missing), ("not callable", broken)):
            with self.subTest(case=case), mock.patch.dict(sys.modules, {"comfy.k_diffusion.sampling": module}), \
                    mock.patch.object(self.pack, "_flow_cosmos_not_flow_logged", False):
                with self.assertRaises(xmath.ExtraSchedulerError) as caught:
                    self.pack.flow_cosmos_rho7(10, 0.03, 14.6, self._sampling("epsilon"))
                self.assertIn("comfy.k_diffusion.sampling.get_sigmas_karras", str(caught.exception))
                self.assertFalse(self.pack._flow_cosmos_not_flow_logged, "no 'using Karras' line without Karras")
                self.assertTrue(torch.equal(self.pack.flow_cosmos_rho7(10, 0.03, 1.0, self._sampling("const")),
                                            flow_list))

    def test_not_a_flow_model_is_logged_once_per_process(self):
        self.assertEqual(xmath.FLOW_COSMOS_NOT_FLOW_LOG, self.ext.FLOW_COSMOS_NOT_FLOW_LOG)   # the template
        with mock.patch.object(self.pack, "_flow_cosmos_not_flow_logged", False):
            with self.assertNoLogs("ai_studio_forge_parity", level="INFO"):
                self._pack(10, 0.03, 1.0, "const")   # a flow model logs nothing
            with self.assertLogs("ai_studio_forge_parity", level="INFO") as logs:
                self._pack(10, 0.03, 14.6, "epsilon")
                self._pack(20, 0.03, 14.6, "v_prediction")
                self._pack(10, 0.03, 14.6, "epsilon")
        self.assertEqual([record.getMessage() for record in logs.records],
                         [xmath.FLOW_COSMOS_NOT_FLOW_LOG.format(rho=7.0)])
        self.assertEqual(logs.records[0].getMessage(), self.ext.FLOW_COSMOS_NOT_FLOW_LOG.format(rho=7.0))
        self.assertEqual(logs.records[0].levelname, "INFO")

    # -- the three values (sam-extra v0.33.0 D19) ------------------------------------------------------------------

    @staticmethod
    def _values(rho: float = 7.0, sigma_max: float = 80.0, sigma_min: float = 0.002, **others: Any) -> xmath.Settings:
        """Settings with these Flow Cosmos values as they are (not coerced — the node clamps, the scheduler checks)."""
        return xmath.Settings(**{**xmath.DEFAULTS._asdict(), **others, "flow_cosmos_rho": rho,
                                 "flow_cosmos_sigma_max": sigma_max, "flow_cosmos_sigma_min": sigma_min})

    def test_the_defaults_are_the_list_of_before(self):
        """No values, the defaults in every form (coerced, spelt out, an old five-value block, Laplace/custom values
        changed) == the fixed D18 list == the A/B expression == the upstream numbers; eps/v == Karras ρ 7."""
        forms = (None, xmath.DEFAULTS, self._values(), xmath.coerce({}),
                 xmath.coerce_args(["sigmas", "1 0", "M - s", 1.5, 2.0]),
                 xmath.coerce({"flow_cosmos_rho": "7", "flow_cosmos_sigma_max": 80, "flow_cosmos_sigma_min": "0.002"}),
                 xmath.coerce({"laplace_mu": -3.0, "laplace_beta": 9.0, "custom_expression": "M"}))
        ab = xmath.coerce({"custom_expression": COSMOS_AB_EXPRESSION})
        for n in sorted({0, *COSMOS_STEPS, *range(1, 121)}):
            with self.subTest(n=n):
                fixed = self._pack(n, *self.anima, "const")
                if n:
                    self.assertTrue(torch.equal(fixed, self.pack.custom(n, *self.anima, ab)))
                if n > 1:
                    self.assertTrue(torch.equal(fixed, torch.tensor(upstream_flow_cosmos_rho(n), dtype=torch.float32)))
                for settings in forms:
                    self.assertTrue(torch.equal(self._pack(n, *self.anima, "const", settings), fixed), settings)
                    self.assertTrue(torch.equal(self._pack(n, *self.ranges[3], "epsilon", settings),
                                                self.karras(n, *self.ranges[3], rho=7.0)), settings)
                self._compare(n, *self.anima, "const", xmath.coerce_args(["expression", None, None, 0.0, 0.5]))

    def test_the_values_on_flow_models_over_a_grid(self):
        """ρ x σ̃ range x n on a flow model == sam-extra bit for bit (the model's own range is still not used), and ==
        the A/B expression with those numbers through the pack's ``custom`` scheduler (another code path)."""
        outcomes = set()
        for rho in COSMOS_RHOS:
            for top, bottom in COSMOS_VE_RANGES:
                settings = self._values(rho, top, bottom)
                expression = xmath.coerce({"custom_expression": cosmos_expression(rho, top, bottom)})
                for n in COSMOS_KNOB_STEPS:
                    outcomes.add(self._compare(n, *self.anima, "const", settings))
                    got = self._pack(n, *self.anima, "const", settings)
                    for sigma_min, sigma_max in (self.ranges[0], self.ranges[3], (0.002, 80.0)):
                        self.assertTrue(torch.equal(self._pack(n, sigma_min, sigma_max, "const", settings), got))
                    if n:
                        self.assertTrue(torch.equal(self.pack.custom(n, *self.anima, expression), got),
                                        (rho, top, bottom, n))
                    if n > 1:
                        self.assertTrue(bool((got[1:] < got[:-1]).all()), (rho, top, bottom, n))
        self.assertEqual(outcomes, {"ok"})
        moved = self._pack(28, *self.anima, "const", self._values(5.0, 240.0, 0.006))
        self.assertFalse(torch.equal(moved, self._pack(28, *self.anima, "const")))   # the values are really used
        self.assertAlmostEqual(float(moved[0]), 240.0 / 241.0, delta=1e-7)   # σ̃ max/(1 + σ̃ max) first

    def test_the_values_on_eps_and_v_models_are_karras_with_that_rho(self):
        """Not a flow model: the host's Karras with the run's ρ on the range passed == sam-extra running Forge's
        Karras; σ̃ max / σ̃ min are not used there (even a range a flow model refuses)."""
        sdxl = self.ranges[3]
        for kind in self.OTHER_KINDS:
            for rho in (1.0, 2.5, 3.3, 7.0, 15.0):
                for sigma_min, sigma_max in (sdxl, (0.1, 20.0), (0.002, 80.0)):
                    for top, bottom in ((80.0, 0.002), (240.0, 0.006), (1.0, 1.0), (0.5, 2.0)):
                        settings = self._values(rho, top, bottom)
                        for n in (0, 1, 2, 10, 28, 151):
                            with self.subTest(kind=kind, rho=rho, range=(sigma_min, sigma_max), ve=(top, bottom), n=n):
                                self.assertEqual(self._compare(n, sigma_min, sigma_max, kind, settings), "ok")
                                got = self._pack(n, sigma_min, sigma_max, kind, settings)
                                self.assertTrue(torch.equal(got, self.karras(n, sigma_min, sigma_max, rho=rho)))
                                self.assertTrue(torch.equal(got, self.forge_k.get_sigmas_karras(
                                    n, sigma_min, sigma_max, rho=rho)))
        calls = []

        def recording(*args, **kwargs):
            calls.append((args, kwargs))
            return self.karras(*args, **kwargs)

        self.pack.flow_cosmos_rho7(5, *sdxl, self._sampling("epsilon"), self._values(4.5, 1.0, 1.0), karras=recording)
        self.assertEqual(calls, [((5, sdxl[0], sdxl[1]), {"rho": 4.5, "device": "cpu"})])

    def test_values_past_the_node_limits_stop_alike(self):
        """Values the node's clamps keep out (only reachable without them) and the one pair they let through
        (σ̃ min = σ̃ max = 1): the same error class and message as sam-extra, in its order of checks — ρ finite and
        above 0 (both branches, before the Karras lookup), 0 < σ̃ min < σ̃ max both finite (flow only), an overflow of
        either branch's maths, a sigma before the final 0 that is not a normal float32 (0, subnormal, NaN, inf)."""
        nan, inf = math.nan, math.inf
        cases = [self._values(rho=rho) for rho in (0.0, -0.0, -1.0, nan, inf, -inf, 0.001, 0.02, 0.05, 1e-300,
                                                   16.0, 100.0, 1000.0)]
        cases += [self._values(sigma_max=top, sigma_min=bottom)
                  for top, bottom in ((1.0, 1.0), (80.0, 80.0), (0.5, 1.0), (80.0, 0.0), (80.0, -0.002), (80.0, nan),
                                      (nan, 0.002), (inf, 0.002), (inf, inf), (80.0, -inf), (0.0, 0.002), (80.0, 5e-324),
                                      (80.0, 1e-300), (80.0, 1e-30), (1e308, 0.002), (2000.0, 1e-5))]
        cases += [self._values(rho=1.0, sigma_min=1e-300), self._values(rho=0.05, sigma_max=1.0, sigma_min=0.9999)]
        ranges = (self.ranges[3], self.anima, (1e-7, 14.6), (1e-40, 14.6))
        messages: dict[str, set] = {}
        for settings in cases:
            for kind in ("const", *self.OTHER_KINDS):
                for sigma_min, sigma_max in ranges:
                    for n in (0, 1, 10, 28):
                        outcome = self._compare(n, sigma_min, sigma_max, kind, settings)
                        if outcome != "ok":
                            error = _outcome(lambda: self._pack(n, sigma_min, sigma_max, kind, settings))[1]
                            messages.setdefault(error[0], set()).add(error[1].split(",")[0].split(" got ")[0])
        self.assertEqual(set(messages), {"ExtraSchedulerError"})
        texts = " | ".join(sorted(messages["ExtraSchedulerError"]))
        for fragment in ("Flow Cosmos rho7 needs a finite rho above 0", "Flow Cosmos rho7 needs 0 < sigma~ min < sigma~ max",
                         "overflows", "in Forge's Karras", "Flow Cosmos rho7 gives sigma "):
            self.assertIn(fragment, texts)
        rho_error = _outcome(lambda: self._pack(10, *self.ranges[3], "epsilon", self._values(rho=0.0)))[1]
        self.assertEqual(rho_error, ("ExtraSchedulerError", "Extra Schedulers: Flow Cosmos rho7 needs a finite rho "
                                     "above 0, got 0 (Extra Schedulers accordion → Flow Cosmos rho / sigma max / sigma "
                                     "min)"))
        range_error = _outcome(lambda: self._pack(10, *self.anima, "const", self._values(sigma_max=1.0, sigma_min=1.0)))
        self.assertEqual(range_error[1], ("ExtraSchedulerError", "Extra Schedulers: Flow Cosmos rho7 needs 0 < sigma~ "
                                          "min < sigma~ max, both finite, got sigma~ min 1 and sigma~ max 1 (Extra "
                                          "Schedulers accordion → Flow Cosmos rho / sigma max / sigma min)"))
        # the pair the clamps let through is refused on a flow model only, also without steps to fall
        for n in (0, 1):
            self.assertEqual(_outcome(lambda: self._pack(n, *self.anima, "const",
                                                         xmath.coerce({"flow_cosmos_sigma_max": 1.0,
                                                                       "flow_cosmos_sigma_min": 1.0})))[0], "error")
        tiny = _outcome(lambda: self._pack(28, *self.ranges[3], "const", self._values(rho=0.05)))[1]
        self.assertIn("the smallest usable value is 1.175e-38", tiny[1])
        # no Karras lookup (and no log line) before the ρ check
        with mock.patch.object(self.pack, "host_get_sigmas_karras", side_effect=AssertionError("looked up")), \
                mock.patch.object(self.pack, "_flow_cosmos_not_flow_logged", False):
            with self.assertRaises(xmath.ExtraSchedulerError):
                self.pack.flow_cosmos_rho7(10, *self.ranges[3], self._sampling("epsilon"), self._values(rho=nan))
            self.assertFalse(self.pack._flow_cosmos_not_flow_logged)

    def test_the_node_limits_keep_the_slider_corners_usable(self):
        """The corners of the node's ranges (σ̃ min below σ̃ max), both branches, 1000 steps: a usable, strictly
        falling list == sam-extra (a σ̃ range as narrow as 0.9999 … 1 has its flat-step limit at 841 steps — below)."""
        corners = ((xmath.FLOW_COSMOS_SIGMA_MAX_MAX, xmath.FLOW_COSMOS_SIGMA_MIN_MIN),
                   (xmath.FLOW_COSMOS_SIGMA_MAX_MIN, xmath.FLOW_COSMOS_SIGMA_MIN_MIN),
                   (xmath.FLOW_COSMOS_SIGMA_MAX_MAX, xmath.FLOW_COSMOS_SIGMA_MIN_MAX))
        for rho in (xmath.FLOW_COSMOS_RHO_MIN, xmath.FLOW_COSMOS_RHO_MAX):
            for top, bottom in corners:
                for kind in ("const", "epsilon"):
                    with self.subTest(rho=rho, ve=(top, bottom), kind=kind):
                        settings = xmath.coerce({"flow_cosmos_rho": rho, "flow_cosmos_sigma_max": top,
                                                 "flow_cosmos_sigma_min": bottom})
                        self.assertEqual(self._compare(1000, *self.ranges[3], kind, settings), "ok")
                        sigmas = self._pack(1000, *self.ranges[3], kind, settings)
                        self.assertGreater(float(sigmas[-2]), 1e-5)
                        self.assertTrue(bool((sigmas[1:] < sigmas[:-1]).all()))

    def test_a_narrow_sigma_range_reaches_the_flat_step_limit_alike(self):
        """σ̃ 0.9999 … 1: 840 steps fall strictly, 841 have a flat step (sam-extra's README figure, for ρ at both ends
        of the node's range and the default); on a flow model the message names ρ and σ̃, not Settings."""
        for rho in (xmath.FLOW_COSMOS_RHO_MIN, 7.0, xmath.FLOW_COSMOS_RHO_MAX):
            with self.subTest(rho=rho):
                settings = self._values(rho, 1.0, 0.9999)
                self.assertEqual(self._compare(840, *self.anima, "const", settings), "ok")
                sigmas = self._pack(840, *self.anima, "const", settings)
                self.assertTrue(bool((sigmas[1:] < sigmas[:-1]).all()))
                self.assertEqual(self._compare(841, *self.anima, "const", settings), "ExtraSchedulerError")
        message = _outcome(lambda: self._pack(1000, *self.anima, "const", self._values(7.0, 1.0, 0.9999)))[1][1]
        self.assertTrue(message.startswith("Extra Schedulers: Flow Cosmos rho7 gives sigma 0.5 at steps 3 and 4 (rho "
                                           "7, sigma~ 0.9999 … 1 — Extra Schedulers accordion → Flow Cosmos rho / sigma "
                                           "max / sigma min; the model's sigma range is not used on a flow model); "),
                        message)
        self.assertNotIn("Settings → Sampler Parameters", message)

    def test_the_log_line_names_the_rho_of_the_first_eps_v_run(self):
        with mock.patch.object(self.pack, "_flow_cosmos_not_flow_logged", False):
            with self.assertLogs("ai_studio_forge_parity", level="INFO") as logs:
                self._pack(10, 0.03, 14.6, "const", self._values(5.5))        # a flow model logs nothing
                self._pack(10, 0.03, 14.6, "epsilon", self._values(5.5))
                self._pack(10, 0.03, 14.6, "v_prediction")
        self.assertEqual([record.getMessage() for record in logs.records],
                         ["[Extra Schedulers] Flow Cosmos rho7: not a flow model - using Karras rho 5.5 on the model's "
                          "own sigma range"])
        self.assertEqual(logs.records[0].getMessage(), self.ext.FLOW_COSMOS_NOT_FLOW_LOG.format(rho=5.5))

    def test_karras_host_function_is_forge_s(self):
        """ComfyUI's ``get_sigmas_karras`` (what the pack calls) and Forge's (what sam-extra calls) are the same
        function body, and so is ``append_zero``."""
        if not COMFY_ROOT:
            self.skipTest("Set AISTUDIO_COMFY_TEST_ROOT to compare ComfyUI's get_sigmas_karras with Forge's")
        comfy = Path(COMFY_ROOT) / "comfy" / "k_diffusion" / "sampling.py"
        forge = FORGE_ROOT / "modules_forge" / "packages" / "k_diffusion" / "sampling.py"
        for name in ("get_sigmas_karras", "append_zero"):
            with self.subTest(name=name):
                self.assertEqual(_function_body_dump(comfy, name), _function_body_dump(forge, name))


# Flow Cosmos Dynamic (sam-extra v0.33.0 D20): its documented float32 limit with the default values — the first two
# equal neighbouring flow times (steps 476/477, t ≈ 0.987626); 1,277,193 steps still fall strictly.
DYNAMIC_FIRST_FLAT_STEPS = 1_277_194
DYNAMIC_RHOS = (2.5, 3.0, 3.6, 3.85, 3.9, 4.0, 5.0, 7.0, 9.3, 15.0)
DYNAMIC_STEPS = (0, 1, 2, 3, 4, 5, 8, 10, 20, 25, 28, 50, 151, 300)


@requires_torch
@requires_extension
class FlowCosmosDynamicParityTests(unittest.TestCase):
    """``flow_cosmos_dynamic`` == sam-extra v0.33.0's, bit for bit (float32, CPU), on flow and eps/v models, errors
    alike (class and text). sam-extra reads the generation's values from ``settings.active()``; the pack gets them as
    ``settings`` (the carrier on ComfyUI). No host function: both branches are the pack's own maths."""

    OTHER_KINDS = ("epsilon", "v_prediction")

    @classmethod
    def setUpClass(cls):
        from tests._optional_deps import bind_torch

        bind_torch(globals())
        from comfy_custom_nodes.ai_studio_forge_parity import extra_schedulers

        cls.pack = extra_schedulers
        cls.ext = _ext("schedulers")
        cls.ext_settings = _ext("settings")
        cls.ranges = _sigma_ranges()
        cls.anima = _flow_sigma_range(3.0)

    def setUp(self):
        # each Flow Cosmos scheduler logs one line per process on both sides: kept quiet here, except in the log test
        for module in (self.pack, self.ext):
            for name in ("_flow_cosmos_not_flow_logged", "_flow_cosmos_dynamic_not_flow_logged"):
                patcher = mock.patch.object(module, name, True)
                patcher.start()
                self.addCleanup(patcher.stop)

    def tearDown(self):
        self.ext_settings.reset()

    @staticmethod
    def _sampling(kind: str | None) -> Any:
        """A model_sampling the pack's flow rule reads (``extra_samplers_dy.is_const``: ``prediction_type``)."""
        return None if kind is None else types.SimpleNamespace(prediction_type=kind)

    @staticmethod
    def _values(rho: float = 7.0, sigma_max: float = 80.0, sigma_min: float = 0.002, **others: Any) -> xmath.Settings:
        """Settings with these Flow Cosmos values as they are (not coerced — the node clamps, the scheduler checks)."""
        return xmath.Settings(**{**xmath.DEFAULTS._asdict(), **others, "flow_cosmos_rho": rho,
                                 "flow_cosmos_sigma_max": sigma_max, "flow_cosmos_sigma_min": sigma_min})

    def _pack(self, n, sigma_min, sigma_max, kind, settings: xmath.Settings | None = None):
        return self.pack.flow_cosmos_dynamic(n, sigma_min, sigma_max, self._sampling(kind), settings)

    def _ext(self, n, sigma_min, sigma_max, kind, settings: xmath.Settings | None = None):
        """sam-extra with ``settings`` active, the values as they are (not coerced)."""
        values = (settings if settings is not None else xmath.DEFAULTS)._asdict()
        self.ext_settings.set_active(self.ext_settings.ExtraSchedulerSettings(**values))
        return self.ext.flow_cosmos_dynamic(n=n, sigma_min=sigma_min, sigma_max=sigma_max,
                                            inner_model=_forge_model(kind), device="cpu")

    def _compare(self, n, sigma_min, sigma_max, kind, settings: xmath.Settings | None = None) -> str:
        got = _outcome(lambda: self._pack(n, sigma_min, sigma_max, kind, settings))
        want = _outcome(lambda: self._ext(n, sigma_min, sigma_max, kind, settings))
        label = (kind, n, sigma_min, sigma_max, settings)
        _assert_same(self, _tensor_key(got), _tensor_key(want), label)
        if got[0] == "ok":
            self.assertTrue(torch.equal(got[1], want[1]), label)
            self.assertEqual(got[1].dtype, torch.float32, label)
        return got[0] if got[0] == "ok" else got[1][0]

    @staticmethod
    def _as_karras_dynamic_text(error: tuple[str, str]) -> tuple[str, str]:
        """Karras Dynamic's error as Flow Cosmos Dynamic says it on an eps/v model: its own name, the node's values
        as the rho's origin (sam-extra ``_karras_dynamic(name=, rho_source=)``)."""
        name, text = error
        return name, text.replace("Karras Dynamic", "Flow Cosmos Dynamic").replace(xmath.RHO_SOURCE,
                                                                                    xmath.FLOW_COSMOS_VALUES_SOURCE)

    def test_flow_models_over_the_value_grid(self):
        """ρ x σ̃ range x n on a flow model == sam-extra (values, and errors: ρ ≤ 2 or a rising list
        ``KarrasDynamicError``, a flat step ``ExtraSchedulerError``), whatever the model's own sigma range; through
        ``schedule`` too."""
        outcomes = set()
        for rho in DYNAMIC_RHOS:
            for top, bottom in COSMOS_VE_RANGES:
                settings = self._values(rho, top, bottom)
                for n in DYNAMIC_STEPS:
                    outcome = self._compare(n, *self.anima, "const", settings)
                    outcomes.add(outcome)
                    if outcome != "ok":
                        continue
                    got = self._pack(n, *self.anima, "const", settings)
                    for sigma_min, sigma_max in (self.ranges[0], self.ranges[3], (0.002, 80.0), (2.0, 1.0)):
                        self.assertTrue(torch.equal(self._pack(n, sigma_min, sigma_max, "const", settings), got))
                    self.assertTrue(torch.equal(self.pack.schedule("flow_cosmos_dynamic", n, *self.anima, settings,
                                                                   model_sampling=self._sampling("const")), got))
                    self.assertTrue(bool((got[1:] < got[:-1]).all()), (rho, top, bottom, n))
        self.assertEqual(outcomes, {"ok", "KarrasDynamicError", "ExtraSchedulerError"})
        moved = self._pack(28, *self.anima, "const", self._values(5.0, 240.0, 0.006))
        self.assertFalse(torch.equal(moved, self._pack(28, *self.anima, "const")))   # the values are really used
        self.assertAlmostEqual(float(moved[0]), 240.0 / 241.0, delta=1e-7)   # σ̃ max/(1 + σ̃ max) first

    def test_the_defaults_and_the_documented_shape(self):
        """No values and the defaults in every form give one list (== sam-extra) == the float64 flow times as float32;
        sam-extra's figures for Anima's 28 steps: 18 above 0.5, 5 below 0.05, median (lower) 0.792 — Flow Cosmos
        rho7: 17, 5, 0.680; strictly falling for every n up to 600."""
        forms = (None, xmath.DEFAULTS, self._values(), xmath.coerce({}),
                 xmath.coerce_args(["sigmas", "1 0", "M - s", 1.5, 2.0]),
                 xmath.coerce({"flow_cosmos_rho": "7", "flow_cosmos_sigma_max": 80, "flow_cosmos_sigma_min": "0.002"}),
                 xmath.coerce({"laplace_mu": -3.0, "laplace_beta": 9.0, "custom_expression": "M"}))
        for n in sorted({*DYNAMIC_STEPS, *range(0, 61)}):
            with self.subTest(n=n):
                reference = self._pack(n, *self.anima, "const")
                self.assertEqual(self._compare(n, *self.anima, "const"), "ok")
                for settings in forms:
                    self.assertTrue(torch.equal(self._pack(n, *self.anima, "const", settings), reference), settings)
                times, bad = xmath.flow_cosmos_dynamic_times(n)
                self.assertIsNone(bad)
                self.assertTrue(torch.equal(reference, torch.tensor([*times, 0.0], dtype=torch.float32)))
        dynamic = [float(value) for value in self._pack(28, *self.anima, "const")[:-1]]
        fixed = [float(value) for value in self.pack.flow_cosmos_rho7(28, *self.anima, self._sampling("const"))[:-1]]
        for values, figures in ((dynamic, (18, 5, 0.792)), (fixed, (17, 5, 0.680))):
            ordered = sorted(values)
            self.assertEqual((sum(value > 0.5 for value in values), sum(value < 0.05 for value in values),
                              round(ordered[len(ordered) // 2 - 1], 3)), figures)
        for n in range(1, 601):
            sigmas = self._pack(n, *self.anima, "const")
            self.assertTrue(bool((sigmas[1:] < sigmas[:-1]).all()), n)

    def test_eps_and_v_models_are_karras_dynamic_with_the_node_rho(self):
        """Not a flow model: Karras Dynamic itself with the node's ρ on the range passed (the MODEL's) == sam-extra ==
        the extension's and the pack's ``karras_dynamic`` with that ρ, bit for bit; σ̃ is not used there (even a range a
        flow model refuses); errors are Karras Dynamic's, under this scheduler's name and naming the node's values."""
        sdxl = self.ranges[3]
        ranges = [sdxl, (0.1, 20.0), (0.0291675, 1.0), (0.002, 80.0), self.anima, (2.0, 1.0), (2.0, 2.0), (0.0, 1.0),
                  (0.003, 0.0), (1e308, 1e-308)]
        outcomes = set()
        for kind in self.OTHER_KINDS:
            for rho in (2.0, 2.5, 3.3, 3.85, 4.0, 5.5, 7.0, 15.0):
                for top, bottom in ((80.0, 0.002), (1.0, 1.0), (0.5, 2.0)):
                    settings = self._values(rho, top, bottom)
                    for n in (0, 1, 2, 5, 10, 28, 151):
                        for sigma_min, sigma_max in ranges:
                            outcomes.add(self._compare(n, sigma_min, sigma_max, kind, settings))
                            got = _outcome(lambda: self._pack(n, sigma_min, sigma_max, kind, settings))
                            plain = _outcome(lambda: self.ext.karras_dynamic(n, sigma_min, sigma_max, rho=rho))
                            label = (kind, rho, top, bottom, n, sigma_min, sigma_max)
                            self.assertEqual(got[0], plain[0], label)
                            if got[0] == "ok":
                                self.assertTrue(torch.equal(got[1], plain[1]), label)
                                self.assertTrue(torch.equal(got[1], self.pack.karras_dynamic(n, sigma_min, sigma_max,
                                                                                             rho=rho)), label)
                            else:
                                self.assertEqual(got[1], self._as_karras_dynamic_text(plain[1]), label)
        self.assertEqual(outcomes, {"ok", "KarrasDynamicError", "ExtraSchedulerError"})
        rho_error = _outcome(lambda: self._pack(10, *sdxl, "epsilon", self._values(rho=2.0)))[1]
        self.assertEqual(rho_error, ("KarrasDynamicError", "Extra Schedulers: Flow Cosmos Dynamic needs rho above 2 (its "
                                     "per-step exponent goes down to rho - 2), got 2. Set rho to about 4 or more: Extra "
                                     "Schedulers accordion → Flow Cosmos rho / sigma max / sigma min."))
        # ρ 7 on the MODEL's range: the pack's own karras_dynamic handler (ρ 7) gives the same list
        for n in (1, 5, 28, 151):
            self.assertTrue(torch.equal(self._pack(n, *sdxl, "epsilon"), self.pack.schedule("karras_dynamic", n, *sdxl)))

    def test_a_tiny_model_sigma_min_on_eps_v_runs_like_karras_dynamic(self):
        """D20's review fix: on an eps/v model there is no float32-tiny rule — a sigma_min that makes Karras Dynamic
        end on 0 or a subnormal runs (or stops) exactly as Karras Dynamic does; a flow model keeps the rule."""
        for kind in self.OTHER_KINDS:
            for rho in (4.0, 7.0, 15.0):
                for sigma_min in (1e-46, 1e-40, 1e-38, 1e-37, 1e-30, 1e-5):
                    for n in (2, 10, 28):
                        with self.subTest(kind=kind, rho=rho, sigma_min=sigma_min, n=n):
                            settings = self._values(rho)
                            self.assertEqual(self._compare(n, sigma_min, 14.6, kind, settings), "ok")
                            self.assertTrue(torch.equal(self._pack(n, sigma_min, 14.6, kind, settings),
                                                        self.pack.karras_dynamic(n, sigma_min, 14.6, rho=rho)))
        ends_on_zero = self._pack(10, 1e-46, 14.6, "epsilon")
        self.assertEqual(float(ends_on_zero[-2]), 0.0)
        tiny = _outcome(lambda: self._pack(28, *self.anima, "const", self._values(sigma_min=1e-300)))[1]
        self.assertEqual(tiny[0], "ExtraSchedulerError")
        self.assertTrue(tiny[1].startswith("Extra Schedulers: Flow Cosmos Dynamic gives sigma 0 at step 27 (sigma~ "
                                           "1e-300 … 80; rho 7 — "), tiny[1])
        self.assertIn("the smallest usable value is 1.175e-38", tiny[1])

    def test_values_past_the_node_limits_stop_alike(self):
        """Values the node's clamps keep out and the pair they let through (σ̃ min = σ̃ max = 1), on flow and eps/v
        models and odd model ranges: the same error class and text as sam-extra, in its order of checks — the range
        above 0, ρ above 2 (both branches), then on a flow model the σ̃ range, the float32-tiny rule, the flat step and
        the rising step; on eps/v Karras Dynamic's own checks."""
        nan, inf = math.nan, math.inf
        cases = [self._values(rho=rho) for rho in (0.0, -0.0, -1.0, 1.0, 2.0, 2.0000001, 2.5, 3.0, nan, inf, -inf,
                                                   1e-300, 16.0, 100.0, 1e300)]
        cases += [self._values(sigma_max=top, sigma_min=bottom)
                  for top, bottom in ((1.0, 1.0), (80.0, 80.0), (0.5, 1.0), (80.0, 0.0), (80.0, -0.002), (80.0, nan),
                                      (nan, 0.002), (inf, 0.002), (inf, inf), (80.0, -inf), (0.0, 0.002), (80.0, 5e-324),
                                      (80.0, 1e-300), (80.0, 1e-30), (1e308, 0.002), (2000.0, 1e-5), (1.0, 0.9999))]
        cases += [self._values(rho=1.0, sigma_min=1e-300), self._values(rho=3.0, sigma_max=1.0, sigma_min=0.9999)]
        ranges = (self.ranges[3], self.anima, (1e-7, 14.6), (1e-40, 14.6), (2.0, 1.0), (0.0, 1.0), (1e308, 1e-308))
        messages: dict[str, set] = {}
        for settings in cases:
            for kind in ("const", *self.OTHER_KINDS):
                for sigma_min, sigma_max in ranges:
                    for n in (0, 1, 10, 28, 1000):
                        outcome = self._compare(n, sigma_min, sigma_max, kind, settings)
                        if outcome != "ok":
                            error = _outcome(lambda: self._pack(n, sigma_min, sigma_max, kind, settings))[1]
                            messages.setdefault(error[0], set()).add(error[1].split(",")[0].split(" got ")[0])
        self.assertEqual(set(messages), {"ExtraSchedulerError", "KarrasDynamicError"})
        texts = " | ".join(sorted(messages["ExtraSchedulerError"] | messages["KarrasDynamicError"]))
        for fragment in ("Flow Cosmos Dynamic needs sigma_min and sigma_max above 0",
                         "Flow Cosmos Dynamic needs rho above 2", "Flow Cosmos Dynamic needs 0 < sigma~ min < sigma~ max",
                         "Flow Cosmos Dynamic gives sigma ", "Flow Cosmos Dynamic with rho ",
                         "Flow Cosmos Dynamic cannot fall from sigma_max", "Flow Cosmos Dynamic gave no finite sigma at"):
            self.assertIn(fragment, texts)
        range_error = _outcome(lambda: self._pack(10, *self.anima, "const", self._values(sigma_max=1.0, sigma_min=1.0)))
        self.assertEqual(range_error[1], ("ExtraSchedulerError", "Extra Schedulers: Flow Cosmos Dynamic needs 0 < sigma~ "
                                          "min < sigma~ max, both finite, got sigma~ min 1 and sigma~ max 1 (Extra "
                                          "Schedulers accordion → Flow Cosmos rho / sigma max / sigma min)"))
        # the pair the clamps let through is refused on a flow model only, also without steps to fall
        for n in (0, 1):
            self.assertEqual(_outcome(lambda: self._pack(n, *self.anima, "const",
                                                         xmath.coerce({"flow_cosmos_sigma_max": 1.0,
                                                                       "flow_cosmos_sigma_min": 1.0})))[0], "error")
        # ρ is checked before σ̃ (a ρ of 2 with σ̃ 1/1 names ρ), and before the log line (no line for a refused run)
        both = _outcome(lambda: self._pack(10, *self.anima, "const", self._values(2.0, 1.0, 1.0)))[1]
        self.assertEqual(both[0], "KarrasDynamicError")
        with mock.patch.object(self.pack, "_flow_cosmos_dynamic_not_flow_logged", False):
            with self.assertRaises(xmath.KarrasDynamicError):
                self._pack(10, *self.ranges[3], "epsilon", self._values(rho=nan))
            self.assertFalse(self.pack._flow_cosmos_dynamic_not_flow_logged)

    def test_rising_lists_stop_alike(self):
        """D20: on σ̃ 0.002 … 80 ρ 3.8 rises for some n up to 300 (first at 25 steps), ρ 3.9 for none — the same steps
        on both sides, ``KarrasDynamicError`` naming the node's values; on a narrow σ̃ range the flat step comes first
        (``ExtraSchedulerError`` naming ρ and σ̃, not Settings)."""
        first = {}
        for rho in (3.8, 3.9):
            for n in range(1, 301):
                outcome = self._compare(n, *self.anima, "const", self._values(rho))
                if outcome != "ok":
                    first.setdefault(rho, (n, outcome))
        self.assertEqual(first, {3.8: (25, "KarrasDynamicError")})
        rising = _outcome(lambda: self._pack(25, *self.anima, "const", self._values(3.8)))[1][1]
        self.assertTrue(rising.startswith("Extra Schedulers: Flow Cosmos Dynamic with rho 3.8 does not fall over 25 "
                                          "steps — sigma "), rising)
        self.assertTrue(rising.endswith("Raise rho to about 4 or more: Extra Schedulers accordion → Flow Cosmos rho / "
                                        "sigma max / sigma min."), rising)
        for rho in (xmath.FLOW_COSMOS_RHO_MAX, 7.0, 4.0):
            with self.subTest(rho=rho):
                settings = self._values(rho, 1.0, 0.9999)
                self.assertEqual(self._compare(840, *self.anima, "const", settings), "ok")
                self.assertEqual(self._compare(841, *self.anima, "const", settings), "ExtraSchedulerError")
        flat = _outcome(lambda: self._pack(1000, *self.anima, "const", self._values(7.0, 1.0, 0.9999)))[1][1]
        self.assertTrue(flat.startswith("Extra Schedulers: Flow Cosmos Dynamic gives sigma 0.5 at steps 3 and 4 (rho 7, "
                                        "sigma~ 0.9999 … 1 — Extra Schedulers accordion → Flow Cosmos rho / sigma max / "
                                        "sigma min; the model's sigma range is not used on a flow model); "), flat)
        self.assertNotIn("Settings → Sampler Parameters", flat)

    def test_float32_limit_with_the_defaults(self):
        """sam-extra's documented limit: from 1,277,194 steps two neighbouring float32 flow times are equal — refused
        alike with the same text; one step fewer is the same strictly falling list on both sides."""
        below = DYNAMIC_FIRST_FLAT_STEPS - 1
        got, want = self._pack(below, *self.anima, "const"), self._ext(below, *self.anima, "const")
        self.assertTrue(torch.equal(got, want))
        self.assertTrue(bool((got[1:] < got[:-1]).all()))
        got = _outcome(lambda: self._pack(DYNAMIC_FIRST_FLAT_STEPS, *self.anima, "const"))
        want = _outcome(lambda: self._ext(DYNAMIC_FIRST_FLAT_STEPS, *self.anima, "const"))
        _assert_same(self, _tensor_key(got), _tensor_key(want), DYNAMIC_FIRST_FLAT_STEPS)
        self.assertEqual(got[1][0], "ExtraSchedulerError")
        self.assertTrue(got[1][1].startswith("Extra Schedulers: Flow Cosmos Dynamic gives sigma 0.987626 at steps 476 "
                                             "and 477 (rho 7, sigma~ 0.002 … 80 — "), got[1][1])

    def test_without_a_model_sampling_it_stops(self):
        """Neither side guesses: no model -> ExtraSchedulerError (each host names what it needs); the positive-range
        check comes first on both sides."""
        got = _outcome(lambda: self.pack.schedule("flow_cosmos_dynamic", 10, 0.03, 1.0))
        self.assertEqual(got, ("error", ("ExtraSchedulerError", "Extra Schedulers: Flow Cosmos Dynamic needs the "
                                         "MODEL's model_sampling to tell a flow model from an eps/v one; none was "
                                         "given")))
        want = _outcome(lambda: self._ext(10, 0.03, 1.0, None))
        self.assertEqual(want[1][0], "ExtraSchedulerError")
        self.assertIn("Flow Cosmos Dynamic needs Forge's model", want[1][1])
        self.assertEqual(_outcome(lambda: self.pack.schedule("flow_cosmos_dynamic", 10, 0.0, 1.0))[1],
                         _outcome(lambda: self._ext(10, 0.0, 1.0, None))[1])
        # no model and ρ 2: the model comes first (sam-extra's order)
        self.assertIn("model_sampling", _outcome(lambda: self._pack(10, 0.03, 1.0, None, self._values(2.0)))[1][1])

    def test_not_a_flow_model_is_logged_once_per_process_with_its_own_line(self):
        """sam-extra's line (its own flag, apart from Flow Cosmos rho7's), with the ρ of the first eps/v run; a flow
        model logs nothing."""
        import contextlib
        import io

        self.assertEqual(xmath.FLOW_COSMOS_DYNAMIC_NOT_FLOW_LOG, self.ext.FLOW_COSMOS_DYNAMIC_NOT_FLOW_LOG)
        karras = _karras_host_function()[0]
        with mock.patch.object(self.pack, "_flow_cosmos_dynamic_not_flow_logged", False), \
                mock.patch.object(self.pack, "_flow_cosmos_not_flow_logged", False):
            with self.assertNoLogs("ai_studio_forge_parity", level="INFO"):
                self._pack(10, 0.03, 1.0, "const", self._values(5.5))   # a flow model logs nothing
            with self.assertLogs("ai_studio_forge_parity", level="INFO") as logs:
                self.pack.flow_cosmos_rho7(10, 0.03, 14.6, self._sampling("epsilon"), self._values(4.5), karras=karras)
                self._pack(10, 0.03, 14.6, "epsilon", self._values(5.5))
                self._pack(20, 0.03, 14.6, "v_prediction")
                self.pack.flow_cosmos_rho7(10, 0.03, 14.6, self._sampling("epsilon"), karras=karras)
                self._pack(10, 0.03, 14.6, "epsilon")
        self.assertEqual([record.getMessage() for record in logs.records],
                         [xmath.FLOW_COSMOS_NOT_FLOW_LOG.format(rho=4.5),
                          xmath.FLOW_COSMOS_DYNAMIC_NOT_FLOW_LOG.format(rho=5.5)])
        self.assertEqual(logs.records[1].getMessage(),
                         "[Extra Schedulers] Flow Cosmos Dynamic: not a flow model - using Karras Dynamic rho 5.5 on the "
                         "model's own sigma range")
        self.assertEqual({record.levelname for record in logs.records}, {"INFO"})
        # sam-extra prints the same line (to stderr) once, with the same ρ
        stderr = io.StringIO()
        with mock.patch.object(self.ext, "_flow_cosmos_dynamic_not_flow_logged", False), \
                contextlib.redirect_stderr(stderr):
            self._ext(10, 0.03, 14.6, "epsilon", self._values(5.5))
            self._ext(10, 0.03, 14.6, "epsilon")
        self.assertEqual(stderr.getvalue().splitlines(), [xmath.FLOW_COSMOS_DYNAMIC_NOT_FLOW_LOG.format(rho=5.5)])


class _FakeSampling:
    """A model_sampling stand-in: float32 0-d ``sigma_min``/``sigma_max`` like ComfyUI's properties."""

    def __init__(self, sigma_min: float, sigma_max: float, **attributes: Any):
        self.sigmas = torch.tensor([sigma_min, sigma_max], dtype=torch.float32)
        self.__dict__.update(attributes)

    @property
    def sigma_min(self):
        return self.sigmas[0]

    @property
    def sigma_max(self):
        return self.sigmas[-1]


@requires_torch
@requires_extension
class HandlerTests(unittest.TestCase):
    """``handler_function(name)(model_sampling, steps)`` == the extension with Forge's float range."""

    @classmethod
    def setUpClass(cls):
        from tests._optional_deps import bind_torch

        bind_torch(globals())
        from comfy_custom_nodes.ai_studio_forge_parity import extra_schedulers

        cls.pack = extra_schedulers
        cls.ext = _ext("schedulers")
        cls.ext_settings = _ext("settings")
        cls.ext_sigma_list = _ext("sigma_list")
        cls.interp = staticmethod(forge_loglinear_interp())
        cls.plain = staticmethod(_laplace_host_function()[0])
        cls.karras = staticmethod(_karras_host_function()[0])
        cls.forge_k = forge_k_sampling()

    def setUp(self):
        for module in (self.pack, self.ext):   # the Flow Cosmos schedulers' one log line per process: quiet here
            for flag in ("_flow_cosmos_not_flow_logged", "_flow_cosmos_dynamic_not_flow_logged"):
                patcher = mock.patch.object(module, flag, True)
                patcher.start()
                self.addCleanup(patcher.stop)

    def tearDown(self):
        self.ext_settings.reset()

    def _ext_schedule(self, name: str, steps: int, sampling, settings: xmath.Settings):
        from unittest import mock

        functions = {"cosine": self.ext.cosine, "cosine_exponential": self.ext.cosine_exponential_blend,
                     "phi": self.ext.phi, "laplace": self.ext.laplace, "karras_dynamic": self.ext.karras_dynamic,
                     "custom": self.ext.custom, "react_cosinusoidal_dynsf": self.ext.react_cosinusoidal_dynsf,
                     "flow_cosmos_rho7": self.ext.flow_cosmos_rho7, "flow_cosmos_dynamic": self.ext.flow_cosmos_dynamic}
        self.assertEqual(set(functions), set(self.pack.SCHEDULER_NAMES))
        self.ext_settings.set_active(self.ext_settings.ExtraSchedulerSettings(**settings._asdict()))
        # Forge: model_wrap.sigmas[0] / [-1] .item(); a need_inner_model scheduler also gets model_wrap
        extra = {}
        if name in ("flow_cosmos_rho7", "flow_cosmos_dynamic"):
            extra["inner_model"] = _forge_model(getattr(sampling, "prediction_type", None))
        with mock.patch.object(self.ext_sigma_list, "forge_loglinear_interp", return_value=self.interp), \
                installed_k_sampling(self.forge_k):
            return functions[name](n=steps, sigma_min=sampling.sigmas[0].item(), sigma_max=sampling.sigmas[-1].item(),
                                   device="cpu", **extra)

    def _pack_schedule(self, name: str, steps: int, sampling):
        from unittest import mock

        with mock.patch.object(self.pack, "host_get_sigmas_laplace", return_value=self.plain), \
                mock.patch.object(self.pack, "host_loglinear_interp", side_effect=self.interp), \
                mock.patch.object(self.pack, "host_get_sigmas_karras", return_value=self.karras):
            return self.pack.handler_function(name)(sampling, steps)

    def test_handlers_read_the_carried_settings(self):
        carried = (
            None,
            xmath.coerce({"laplace_mu": 0.5, "laplace_beta": 1.0}),
            xmath.coerce({"custom_expression": "m + (M - m) * (1 - x) ** 3"}),
            xmath.coerce({"custom_mode": "sigmas", "custom_sigmas": "[14.6, 7, 3.1, 1.2, 0.03]"}),
            {"laplace_mu": "2", "laplace_beta": 3, "custom_mode": "Sigma list", "custom_sigmas": "1 0.5 0"},
            # Flow Cosmos values (D19; Flow Cosmos Dynamic reads the same, D20): Anima's shift 3 as σ̃, a high ρ on a
            # short range, the one pair the clamps let through that both refuse (σ̃ min = σ̃ max = 1 — an error alike),
            # a ρ Flow Cosmos Dynamic refuses (2 — Flow Cosmos rho7 runs) and one where it rises (3.0)
            xmath.coerce({"flow_cosmos_rho": 4.5, "flow_cosmos_sigma_max": 240.0, "flow_cosmos_sigma_min": 0.006}),
            {"flow_cosmos_rho": "12", "flow_cosmos_sigma_max": 999, "flow_cosmos_sigma_min": 0.5, "laplace_beta": 2},
            {"flow_cosmos_sigma_max": 1.0, "flow_cosmos_sigma_min": 1.0},
            {"flow_cosmos_rho": 2.0},
            {"flow_cosmos_rho": 3.0},
        )
        for value in carried:
            for shift in (1.0, 3.0):
                sampling = _FakeSampling(*_flow_sigma_range(shift), prediction_type="const")   # a flow model
                if value is not None:
                    setattr(sampling, self.pack.CARRIER_ATTR, value)
                settings = self.pack.as_settings(value)
                for name in self.pack.SCHEDULER_NAMES:
                    for steps in (1, 5, 28):
                        with self.subTest(value=value, shift=shift, name=name, steps=steps):
                            self.assertEqual(_tensor_key(_outcome(lambda: self._pack_schedule(name, steps, sampling))),
                                             _tensor_key(_outcome(lambda: self._ext_schedule(name, steps, sampling,
                                                                                             settings))))

    def test_handlers_on_an_eps_or_v_model(self):
        """An SDXL-range model_sampling that is not a flow model: every handler == the extension (Flow Cosmos rho7
        takes the Karras branch on both sides — the host's Karras here, Forge's in sam-extra; Flow Cosmos Dynamic
        is Karras Dynamic with the carried ρ on both)."""
        sdxl = (float(torch.tensor(0.0291675)), float(torch.tensor(14.614642)))
        for kind in ("epsilon", "v_prediction"):
            for value in (None, xmath.coerce({"laplace_mu": 0.5, "laplace_beta": 1.0}),
                          xmath.coerce({"flow_cosmos_rho": 3.3, "flow_cosmos_sigma_max": 1.0,
                                        "flow_cosmos_sigma_min": 1.0}),   # σ̃ unused on eps/v: no error here
                          xmath.coerce({"flow_cosmos_rho": 5.5, "flow_cosmos_sigma_max": 240.0,
                                        "flow_cosmos_sigma_min": 0.006})):
                sampling = _FakeSampling(*sdxl, prediction_type=kind)
                if value is not None:
                    setattr(sampling, self.pack.CARRIER_ATTR, value)
                settings = self.pack.as_settings(value)
                for name in self.pack.SCHEDULER_NAMES:
                    for steps in (1, 5, 28):
                        with self.subTest(kind=kind, value=value, name=name, steps=steps):
                            got = _outcome(lambda: self._pack_schedule(name, steps, sampling))
                            self.assertEqual(_tensor_key(got),
                                             _tensor_key(_outcome(lambda: self._ext_schedule(name, steps, sampling,
                                                                                             settings))))
                            if name == "flow_cosmos_rho7":
                                self.assertTrue(torch.equal(got[1], self.karras(steps, *sdxl,
                                                                                rho=settings.flow_cosmos_rho)))
                            if name == "flow_cosmos_dynamic":   # Karras Dynamic with the carried ρ, or its error
                                plain = _outcome(lambda: self.ext.karras_dynamic(steps, *sdxl,
                                                                                 rho=settings.flow_cosmos_rho))
                                self.assertEqual(got[0], plain[0])
                                if got[0] == "ok":
                                    self.assertTrue(torch.equal(got[1], plain[1]))

    def test_carrier_is_read_from_the_object_itself(self):
        class Sampling(_FakeSampling):
            pass

        setattr(Sampling, self.pack.CARRIER_ATTR, xmath.coerce({"laplace_mu": 2.0}))   # a class attribute
        sampling = Sampling(*_flow_sigma_range(3.0))
        self.assertEqual(self.pack.carried_settings(sampling), xmath.DEFAULTS)
        setattr(sampling, self.pack.CARRIER_ATTR, xmath.coerce({"laplace_mu": 1.5}))
        self.assertEqual(self.pack.carried_settings(sampling).laplace_mu, 1.5)
        self.assertEqual(self.pack.as_settings({"laplace_beta": 99}).laplace_beta, xmath.LAPLACE_BETA_MAX)
        self.assertEqual(self.pack.as_settings("junk"), xmath.DEFAULTS)

    def test_unknown_names_are_refused(self):
        with self.assertRaises(KeyError):
            self.pack.handler_function("simple")
        with self.assertRaises(KeyError):
            self.pack.schedule("karras", 5, 0.03, 14.6)

    def test_debug_log_lists_the_schedule(self):
        from unittest import mock

        sampling = _FakeSampling(*_flow_sigma_range(3.0))
        with mock.patch.dict(os.environ, {self.pack.DEBUG_ENV: "1"}), \
                self.assertLogs("ai_studio_forge_parity", level="INFO") as logs:
            sigmas = self._pack_schedule("cosine", 4, sampling)
        self.assertEqual(len(logs.records), 1)
        self.assertIn(repr([float(value) for value in sigmas]), logs.output[0])


# ── pack behaviour: registration, carrier, node, Hires carry (PB-XS4; no extension needed) ────────
class _SchedulerHandler(NamedTuple):
    """``comfy.samplers.SchedulerHandler``."""

    handler: Callable[..., Any]
    use_ms: bool = True


def _stock_handler(model_sampling, steps):
    return "stock"


def _fake_comfy_samplers(*, shared_lists: bool = True, handlers: dict | None = None,
                         with_registry: bool = True) -> types.ModuleType:
    """A ``comfy.samplers`` with the registry shape of ComfyUI 0.36.0 (samplers.py:1358-1400)."""
    module = types.ModuleType("comfy.samplers")
    if with_registry:
        module.SchedulerHandler = _SchedulerHandler
        module.SCHEDULER_HANDLERS = dict(handlers if handlers is not None else {
            "simple": _SchedulerHandler(_stock_handler), "karras": _SchedulerHandler(_stock_handler, use_ms=False),
        })
        module.SCHEDULER_NAMES = list(module.SCHEDULER_HANDLERS)
        module.KSampler = type("KSampler", (), {
            "SCHEDULERS": module.SCHEDULER_NAMES if shared_lists else list(module.SCHEDULER_NAMES),
        })
    return module


def _with_comfy(module: types.ModuleType | None):
    comfy = types.ModuleType("comfy")
    entries: dict[str, Any] = {"comfy": comfy}
    if module is not None:
        comfy.samplers = module
        entries["comfy.samplers"] = module
    else:
        entries["comfy.samplers"] = None   # import fails like outside ComfyUI
    return mock.patch.dict(sys.modules, entries)


class _Sampling:
    """A model_sampling stand-in without torch."""

    def __init__(self, sigma_min: float = 0.002, sigma_max: float = 1.0):
        self.sigma_min, self.sigma_max = sigma_min, sigma_max


class _Patcher:
    """ModelPatcher's clone / object patch / get_model_object / model_options (comfy/model_patcher.py)."""

    def __init__(self, sampling: Any = None):
        self.base_sampling = sampling if sampling is not None else _Sampling()
        self.object_patches: dict[str, Any] = {}
        self.model_options: dict[str, Any] = {"transformer_options": {}}

    def clone(self):
        clone = _Patcher(self.base_sampling)
        clone.object_patches = dict(self.object_patches)
        clone.model_options = {key: (dict(value) if isinstance(value, dict) else value)
                               for key, value in self.model_options.items()}
        return clone

    def add_object_patch(self, name, obj):
        self.object_patches[name] = obj

    def get_model_object(self, name):
        if name != "model_sampling":
            raise AttributeError(name)
        return self.object_patches.get(name, self.base_sampling)


class PackBehaviourTests(unittest.TestCase):
    """Host rules of the ComfyUI glue (fake ComfyUI tables and ModelPatchers; no torch, no extension)."""

    @classmethod
    def setUpClass(cls):
        from comfy_custom_nodes.ai_studio_forge_parity import compat, extra_schedulers, hires_carry

        cls.compat, cls.pack, cls.hires_carry = compat, extra_schedulers, hires_carry

    def test_value_schedulers(self):
        """The handlers of these read the carried values; Flow Cosmos rho7 joined Laplace and custom in sam-extra
        v0.33.0 (D19), Flow Cosmos Dynamic reads the same three values (D20)."""
        self.assertEqual(self.pack.VALUE_SCHEDULERS, ("laplace", "custom", "flow_cosmos_rho7", "flow_cosmos_dynamic"))
        self.assertLessEqual(set(self.pack.VALUE_SCHEDULERS), set(self.pack.SCHEDULER_NAMES))

    def test_registration_adds_the_nine_names_once_and_tags_them(self):
        fake = _fake_comfy_samplers()
        with _with_comfy(fake):
            first = self.pack.install_extra_schedulers()
            second = self.pack.install_extra_schedulers()
            owned = self.pack.owned_names()
            choices = self.pack.ForgeNeoExtraSchedulers.INPUT_TYPES()["optional"][self.compat.OWNED_NAMES_INPUT][0]
        names = self.pack.SCHEDULER_NAMES
        self.assertEqual(names, ("cosine", "cosine_exponential", "phi", "laplace", "karras_dynamic", "custom",
                                 "react_cosinusoidal_dynsf", "flow_cosmos_rho7", "flow_cosmos_dynamic"))
        self.assertEqual(choices, list(names))
        self.assertEqual((first.owned, first.inserted, first.foreign, first.published), (names, names, (), True))
        self.assertEqual((second.owned, second.inserted), (names, ()))
        self.assertEqual(owned, names)
        self.assertEqual(fake.SCHEDULER_NAMES, ["simple", "karras", *names])
        self.assertIs(fake.KSampler.SCHEDULERS, fake.SCHEDULER_NAMES)
        for name in names:
            entry = fake.SCHEDULER_HANDLERS[name]
            self.assertIsInstance(entry, _SchedulerHandler)
            self.assertTrue(entry.use_ms)
            self.assertEqual(self.compat.entry_owner(entry), self.pack.OWNER)
            self.assertEqual(entry.handler.__name__, f"extra_scheduler_{name}")

    def test_distinct_choice_lists_both_get_the_names(self):
        fake = _fake_comfy_samplers(shared_lists=False)
        with _with_comfy(fake):
            self.pack.install_extra_schedulers()
        count = len(self.pack.SCHEDULER_NAMES)
        self.assertEqual(fake.SCHEDULER_NAMES[-count:], list(self.pack.SCHEDULER_NAMES))
        self.assertEqual(fake.KSampler.SCHEDULERS[-count:], list(self.pack.SCHEDULER_NAMES))

    def test_a_foreign_scheduler_is_never_replaced_or_published(self):
        foreign = _SchedulerHandler(lambda model_sampling, steps: "theirs")
        handlers = {"simple": _SchedulerHandler(_stock_handler), "laplace": foreign}
        fake = _fake_comfy_samplers(handlers=handlers)
        fake.SCHEDULER_NAMES.remove("laplace")   # a provider that did not list it: the pack must not either
        with _with_comfy(fake), mock.patch.object(self.compat, "_FOREIGN_LOGGED", set()),                 self.assertLogs("ai_studio_forge_parity", level="WARNING") as logs:
            report = self.pack.install_extra_schedulers()
            owned = self.pack.owned_names()
            choices = self.pack.ForgeNeoExtraSchedulers.INPUT_TYPES()["optional"][self.compat.OWNED_NAMES_INPUT][0]
        self.assertIs(fake.SCHEDULER_HANDLERS["laplace"], foreign)
        self.assertEqual(report.foreign, ("laplace",))
        self.assertNotIn("laplace", report.owned)
        self.assertNotIn("laplace", fake.SCHEDULER_NAMES)
        self.assertNotIn("laplace", owned)
        self.assertEqual(choices, [name for name in self.pack.SCHEDULER_NAMES if name != "laplace"])
        self.assertTrue(any("'laplace'" in line for line in logs.output))

    def test_owned_names_are_read_live(self):
        fake = _fake_comfy_samplers()
        with _with_comfy(fake):
            self.pack.install_extra_schedulers()
            fake.SCHEDULER_HANDLERS["custom"] = _SchedulerHandler(_stock_handler)   # replaced afterwards
            fake.SCHEDULER_NAMES.remove("phi")                                       # no longer offered
            self.assertEqual(self.pack.owned_names(), ("cosine", "cosine_exponential", "laplace", "karras_dynamic",
                                                       "react_cosinusoidal_dynsf", "flow_cosmos_rho7",
                                                       "flow_cosmos_dynamic"))

    def test_outside_comfyui_or_without_a_registry(self):
        # The '(none)' choice is checked inside each context: in the real-ComfyUI tier the pack's __init__ has
        # already registered the nine names in ComfyUI's own tables (pack 1.7.0), so the ambient answer is theirs.
        empty = self.compat.NameRegistration()
        node = self.pack.ForgeNeoExtraSchedulers
        with _with_comfy(None):
            self.assertEqual(self.pack.install_extra_schedulers(), empty)
            self.assertEqual(self.pack.owned_names(), ())
            self.assertEqual(node.INPUT_TYPES()["optional"][self.compat.OWNED_NAMES_INPUT][0],
                             [self.compat.NO_OWNED_NAMES])
            with self.assertRaises(RuntimeError):
                self.pack.install_extra_schedulers(required=True)
        with _with_comfy(_fake_comfy_samplers(with_registry=False)):
            self.assertEqual(self.pack.install_extra_schedulers(), empty)
            self.assertEqual(self.pack.owned_names(), ())
            self.assertEqual(node.INPUT_TYPES()["optional"][self.compat.OWNED_NAMES_INPUT][0],
                             [self.compat.NO_OWNED_NAMES])
            with self.assertRaises(RuntimeError):
                self.pack.install_extra_schedulers(required=True)

    def test_a_host_that_breaks_registration_is_logged_not_raised(self):
        class Refusing(dict):
            def __setitem__(self, key, value):
                raise RuntimeError("read-only scheduler table")

        fake = _fake_comfy_samplers(handlers={"simple": _SchedulerHandler(_stock_handler)})
        fake.SCHEDULER_HANDLERS = Refusing(fake.SCHEDULER_HANDLERS)
        with _with_comfy(fake):
            with self.assertLogs("ai_studio_forge_parity", level="ERROR"):
                self.assertEqual(self.pack.install_extra_schedulers(), self.compat.NameRegistration())
            with self.assertRaises(RuntimeError):
                self.pack.install_extra_schedulers(required=True)
            self.assertEqual(self.pack.owned_names(), ())
        self.assertEqual(fake.SCHEDULER_NAMES, ["simple"])

    def test_node_inputs_follow_the_block(self):
        spec = self.pack.ForgeNeoExtraSchedulers.INPUT_TYPES()
        required = spec["required"]
        self.assertEqual(tuple(required), ("model", *xmath.ARG_NAMES))
        self.assertEqual(required["custom_mode"][0], list(xmath.MODES))
        self.assertEqual(required["custom_mode"][1]["default"], xmath.MODE_EXPRESSION)
        self.assertEqual(required["custom_expression"][1]["default"], xmath.DEFAULT_EXPRESSION)
        self.assertEqual(required["custom_sigmas"][1]["default"], xmath.DEFAULT_SIGMAS)
        self.assertEqual({key: required["laplace_mu"][1][key] for key in ("default", "min", "max")},
                         {"default": xmath.LAPLACE_MU_DEFAULT, "min": xmath.LAPLACE_MU_MIN, "max": xmath.LAPLACE_MU_MAX})
        self.assertEqual({key: required["laplace_beta"][1][key] for key in ("default", "min", "max")},
                         {"default": xmath.LAPLACE_BETA_DEFAULT, "min": xmath.LAPLACE_BETA_MIN,
                          "max": xmath.LAPLACE_BETA_MAX})
        # Flow Cosmos rho7's three values: sam-extra's slider ranges, defaults and steps (D19)
        for name, prefix in (("flow_cosmos_rho", "FLOW_COSMOS_RHO"), ("flow_cosmos_sigma_max", "FLOW_COSMOS_SIGMA_MAX"),
                             ("flow_cosmos_sigma_min", "FLOW_COSMOS_SIGMA_MIN")):
            with self.subTest(input=name):
                kind, options = required[name]
                self.assertEqual(kind, "FLOAT")
                self.assertEqual({key: options[key] for key in ("default", "min", "max", "step")},
                                 {key: getattr(xmath, f"{prefix}_{key.upper()}")
                                  for key in ("default", "min", "max", "step")})
                self.assertIn("flow_cosmos_rho7", options["tooltip"])
                self.assertIn("flow_cosmos_dynamic", options["tooltip"])   # D20: the same three values
        self.assertEqual({key: required["flow_cosmos_rho"][1][key] for key in ("default", "min", "max", "step")},
                         {"default": 7.0, "min": 1.0, "max": 15.0, "step": 0.1})
        self.assertEqual({key: required["flow_cosmos_sigma_max"][1][key] for key in ("default", "min", "max", "step")},
                         {"default": 80.0, "min": 1.0, "max": 1000.0, "step": 0.1})
        self.assertEqual({key: required["flow_cosmos_sigma_min"][1][key] for key in ("default", "min", "max", "step")},
                         {"default": 0.002, "min": 0.0001, "max": 1.0, "step": 0.0001})
        self.assertIn("shift", required["flow_cosmos_sigma_max"][1]["tooltip"])
        self.assertIn("rho above 2", required["flow_cosmos_rho"][1]["tooltip"])
        description = self.pack.ForgeNeoExtraSchedulers.DESCRIPTION
        for name in self.pack.SCHEDULER_NAMES:
            self.assertIn(name, description)
        self.assertIn("nine schedulers", description)
        self.assertEqual(list(spec["optional"]), [self.compat.OWNED_NAMES_INPUT])
        self.assertEqual(self.pack.NODE_CLASS_MAPPINGS, {"ForgeNeoExtraSchedulers": self.pack.ForgeNeoExtraSchedulers})
        self.assertEqual(self.pack.ForgeNeoExtraSchedulers.RETURN_TYPES, ("MODEL",))

    def test_node_coerces_and_carries_without_checking_the_text(self):
        base = _Patcher()
        (patched,) = self.pack.ForgeNeoExtraSchedulers().patch(
            base, "Sigma list", "  not ( an expression  ", " [1, 0] ", 99.0, -1.0,
            **{self.compat.OWNED_NAMES_INPUT: "laplace"},
        )
        settings = self.pack.model_settings(patched)
        self.assertEqual(settings, xmath.Settings("sigmas", "not ( an expression", "[1, 0]", 10.0, 0.0))
        self.assertEqual(patched.model_options[self.pack.MODEL_OPTIONS_KEY], settings._asdict())
        self.assertIsNone(self.pack.model_settings(base))
        self.assertNotIn(self.pack.MODEL_OPTIONS_KEY, base.model_options)
        # Flow Cosmos rho7's values: clamped to the node's ranges, a non-number -> its default; the pair the clamps
        # let through (σ̃ min = σ̃ max = 1) is carried as it is and refused only when a flow schedule is built
        (patched,) = self.pack.ForgeNeoExtraSchedulers().patch(base, "expression", "M", "1 0", 0.0, 0.5, 0.5, 5000.0,
                                                               0.0)
        self.assertEqual(tuple(self.pack.model_settings(patched))[5:], (1.0, 1000.0, 0.0001))
        (patched,) = self.pack.ForgeNeoExtraSchedulers().patch(base, flow_cosmos_rho=float("nan"),
                                                               flow_cosmos_sigma_max="x", flow_cosmos_sigma_min=1.0)
        self.assertEqual(tuple(self.pack.model_settings(patched))[5:], (7.0, 80.0, 1.0))
        (patched,) = self.pack.ForgeNeoExtraSchedulers().patch(base, flow_cosmos_sigma_max=1.0, flow_cosmos_sigma_min=1.0)
        self.assertEqual(tuple(self.pack.model_settings(patched))[5:], (7.0, 1.0, 1.0))
        self.assertEqual(patched.model_options[self.pack.MODEL_OPTIONS_KEY]["flow_cosmos_sigma_max"], 1.0)
        # the five inputs of before only: Flow Cosmos keeps its defaults
        (patched,) = self.pack.ForgeNeoExtraSchedulers().patch(base, "expression", "M", "1 0", 0.0, 0.5)
        self.assertEqual(self.pack.model_settings(patched)[5:], (7.0, 80.0, 0.002))

    def test_carry_keeps_the_class_and_leaves_the_original_alone(self):
        original = _Sampling(0.03, 14.6)
        base = _Patcher(original)
        settings = xmath.coerce({"laplace_mu": 0.5})
        carried = self.pack.carry(base, settings)
        duplicate = carried.get_model_object("model_sampling")
        self.assertIsNot(duplicate, original)
        self.assertIs(type(duplicate), _Sampling)
        self.assertEqual((duplicate.sigma_min, duplicate.sigma_max), (0.03, 14.6))
        self.assertIs(getattr(duplicate, self.pack.CARRIER_ATTR), settings)
        self.assertNotIn(self.pack.CARRIER_ATTR, vars(original))
        self.assertIs(base.get_model_object("model_sampling"), original)
        # a later clone keeps the carrier (ModelPatcher.clone copies object patches)
        self.assertEqual(self.pack.model_settings(carried.clone()), settings)
        # carrying again replaces the values on a new copy, the first copy keeps its own
        again = self.pack.carry(carried, {"laplace_mu": -1})
        self.assertEqual(self.pack.model_settings(again).laplace_mu, -1.0)
        self.assertIs(getattr(duplicate, self.pack.CARRIER_ATTR), settings)
        with self.assertRaises(RuntimeError):
            self.pack.carry(object(), settings)   # not a MODEL

    def test_handlers_without_a_carrier_use_the_defaults(self):
        seen = []

        def fake_schedule(name, steps, sigma_min, sigma_max, settings=None, device="cpu", *, model_sampling=None):
            seen.append((name, steps, sigma_min, sigma_max, settings, model_sampling))
            return [sigma_max, 0.0]

        plain = _Sampling(0.25, 2.0)
        with mock.patch.object(self.pack, "schedule", side_effect=fake_schedule):
            self.pack.handler_function("laplace")(plain, 7)
            carried = _Sampling(0.25, 2.0)
            setattr(carried, self.pack.CARRIER_ATTR, {"laplace_beta": 3})
            self.pack.handler_function("laplace")(carried, 7)
            self.pack.handler_function("phi")(carried, 3)   # values do not reach the value-free ones
            self.pack.handler_function("flow_cosmos_rho7")(carried, 4)   # its values, and the model_sampling itself
            self.pack.handler_function("flow_cosmos_rho7")(plain, 5)
            cosmos = _Sampling(0.25, 2.0)
            setattr(cosmos, self.pack.CARRIER_ATTR, xmath.coerce({"flow_cosmos_rho": 3.5, "flow_cosmos_sigma_max": 9}))
            self.pack.handler_function("flow_cosmos_rho7")(cosmos, 6)
            self.pack.handler_function("laplace")(cosmos, 6)   # one carrier holds every value scheduler's values
            self.pack.handler_function("flow_cosmos_dynamic")(cosmos, 7)   # D20: the same three values
            self.pack.handler_function("flow_cosmos_dynamic")(plain, 8)
        self.assertEqual(seen, [
            ("laplace", 7, 0.25, 2.0, xmath.DEFAULTS, plain),
            ("laplace", 7, 0.25, 2.0, xmath.coerce({"laplace_beta": 3}), carried),
            ("phi", 3, 0.25, 2.0, xmath.DEFAULTS, carried),
            ("flow_cosmos_rho7", 4, 0.25, 2.0, xmath.coerce({"laplace_beta": 3}), carried),
            ("flow_cosmos_rho7", 5, 0.25, 2.0, xmath.DEFAULTS, plain),
            ("flow_cosmos_rho7", 6, 0.25, 2.0, xmath.coerce({"flow_cosmos_rho": 3.5, "flow_cosmos_sigma_max": 9}),
             cosmos),
            ("laplace", 6, 0.25, 2.0, xmath.coerce({"flow_cosmos_rho": 3.5, "flow_cosmos_sigma_max": 9}), cosmos),
            ("flow_cosmos_dynamic", 7, 0.25, 2.0, xmath.coerce({"flow_cosmos_rho": 3.5, "flow_cosmos_sigma_max": 9}),
             cosmos),
            ("flow_cosmos_dynamic", 8, 0.25, 2.0, xmath.DEFAULTS, plain),
        ])
        self.assertIs(seen[3][5], carried)

    def test_flow_cosmos_rho7_flow_rule_is_the_pack_s_const_rule(self):
        """``is_flow_sampling`` is ``extra_samplers_dy.is_const`` (the Dy/Restart entries' rule for sam-extra's
        ``prediction_type == "const"``): ComfyUI's ``CONST`` class (incl. a subclass or a private copy of one),
        a Forge-style ``prediction_type``; None without a model_sampling."""
        from comfy_custom_nodes.ai_studio_forge_parity import extra_samplers_dy

        model_sampling = types.ModuleType("comfy.model_sampling")
        const = type("CONST", (), {})
        model_sampling.CONST = const
        flow_class = type("ModelSamplingAdvanced", (_Sampling, const), {})
        eps_class = type("ModelSamplingDiscrete", (_Sampling,), {})
        cases = [
            (flow_class(), True), (eps_class(), False), (self.pack._private_sampling_copy(flow_class()), True),
            (types.SimpleNamespace(prediction_type="const"), True),
            (types.SimpleNamespace(prediction_type="epsilon"), False),
            (types.SimpleNamespace(prediction_type="v_prediction"), False),
        ]
        comfy = types.ModuleType("comfy")
        comfy.model_sampling = model_sampling
        with mock.patch.dict(sys.modules, {"comfy": comfy, "comfy.model_sampling": model_sampling}):
            for sampling, flow in cases:
                with self.subTest(sampling=sampling):
                    self.assertIs(self.pack.is_flow_sampling(sampling), flow)
                    self.assertEqual(self.pack.is_flow_sampling(sampling), extra_samplers_dy.is_const(sampling))
        self.assertIsNone(self.pack.is_flow_sampling(None))
        with mock.patch.object(extra_samplers_dy, "is_const", return_value=True) as rule:
            self.assertIs(self.pack.is_flow_sampling(plain := _Sampling()), True)
        rule.assert_called_once_with(plain)
        # Flow Cosmos Dynamic decides by the same rule (D20): the rule's answer picks its branch
        for flow in (True, False):
            with self.subTest(flow=flow), mock.patch.object(self.pack, "is_flow_sampling", return_value=flow) as rule, \
                    mock.patch.object(self.pack, "_karras_dynamic", return_value="eps/v") as eps_branch, \
                    mock.patch.object(self.pack, "_flow_cosmos_dynamic_not_flow_logged", True), \
                    mock.patch.object(self.pack, "with_final_zero", return_value="flow list"), \
                    mock.patch.object(self.pack, "_refuse_unusable_flow_cosmos"), \
                    mock.patch.object(self.pack, "_refuse_flat_step"), \
                    mock.patch.object(self.pack, "_refuse_karras_dynamic_rise"):
                sampling = _Sampling(0.03, 14.6)
                result = self.pack.flow_cosmos_dynamic(5, 0.03, 14.6, sampling)
                rule.assert_called_once_with(sampling)
                self.assertEqual(result, "flow list" if flow else "eps/v")
                self.assertEqual(eps_branch.called, not flow)

    def test_hires_registration_and_override_carry(self):
        self.assertIn(self.pack.OWNER, self.hires_carry.registered())
        self.assertIn(self.pack.CARRIER_ATTR, self.hires_carry.sampling_attrs())
        settings = xmath.coerce({"custom_mode": "sigmas", "laplace_mu": 1.0, "flow_cosmos_rho": 4.5,
                                 "flow_cosmos_sigma_max": 240.0, "flow_cosmos_sigma_min": 0.006})
        source = self.pack.carry(_Patcher(), settings)
        override = _Patcher(_Sampling(0.1, 3.0))
        result = self.hires_carry.apply(source, override, "override-vae")
        self.assertEqual(self.pack.model_settings(result), settings)
        self.assertEqual(tuple(self.pack.model_settings(result))[5:], (4.5, 240.0, 0.006))   # Flow Cosmos values too
        self.assertEqual(result.get_model_object("model_sampling").sigma_max, 3.0)   # the override's own range
        self.assertIsNone(self.pack.model_settings(override))
        # nothing carried -> the override stays as it is
        self.assertIsNone(self.pack._carry_to_override(_Patcher(), override, None))

    def test_the_pack_flow_shift_keeps_the_values(self):
        from comfy_custom_nodes.ai_studio_forge_parity import generation

        settings = xmath.coerce({"laplace_mu": 2.0, "flow_cosmos_rho": 12.0, "flow_cosmos_sigma_max": 999.0,
                                 "flow_cosmos_sigma_min": 0.5})
        carried = self.pack.carry(_Patcher(), settings)

        def provider(name, *, method, feature, args=(), kwargs=None):
            self.assertEqual(name, "ModelSamplingSD3")
            model, shift, multiplier = args
            patched = model.clone()
            patched.add_object_patch("model_sampling", _Sampling(0.003, 1.0))
            return (patched,)

        with mock.patch.object(generation, "invoke_provider", side_effect=provider):
            shifted = generation._patch_flow_shift(carried, 3.0)
        self.assertEqual(self.pack.model_settings(shifted), settings)
        self.assertEqual(tuple(self.pack.model_settings(shifted))[5:], (12.0, 999.0, 0.5))
        self.assertEqual(shifted.get_model_object("model_sampling").sigma_min, 0.003)

    @requires_torch
    def test_a_device_move_of_the_copy_never_touches_the_original(self):
        torch_module = load_torch()

        class Sampling(torch_module.nn.Module):
            def __init__(self):
                super().__init__()
                self.register_buffer("sigmas", torch_module.linspace(0.002, 1.0, 5))

        original = Sampling()
        carried = self.pack.carry(_Patcher(original), xmath.DEFAULTS)
        duplicate = carried.get_model_object("model_sampling")
        self.assertIs(type(duplicate), Sampling)
        self.assertIs(duplicate.sigmas, original.sigmas)          # shared until something moves it
        self.assertIsNot(duplicate._buffers, original._buffers)
        duplicate.to(torch_module.float64)                        # Module._apply rewrites the copy's own dict
        self.assertEqual(duplicate.sigmas.dtype, torch_module.float64)
        self.assertEqual(original.sigmas.dtype, torch_module.float32)
        self.assertEqual(getattr(duplicate, self.pack.CARRIER_ATTR), xmath.DEFAULTS)


class PureModuleImportTests(unittest.TestCase):
    """The pure module stays importable by the app (no torch/ComfyUI at its top)."""

    def test_top_level_imports_are_standard_library_only(self):
        tree = ast.parse((PACK_DIR / "extra_schedulers_math.py").read_text(encoding="utf-8"))
        imported = set()
        for item in tree.body:
            if isinstance(item, ast.Import):
                imported.update(alias.name.split(".")[0] for alias in item.names)
            elif isinstance(item, ast.ImportFrom):
                imported.add((item.module or "").split(".")[0])
        self.assertLessEqual(imported, {"__future__", "ast", "math", "re", "typing"})


if __name__ == "__main__":
    unittest.main()
