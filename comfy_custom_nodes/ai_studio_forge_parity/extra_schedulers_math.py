"""Extra Schedulers for ComfyUI — the pure part (pack 1.7.0): settings, the safe expression.

The nine schedulers sam-extra adds to Forge's "Schedule type" list (``cosine``,
``cosine_exponential``, ``phi``, ``laplace``, ``karras_dynamic``, ``custom``,
``react_cosinusoidal_dynsf``, ``flow_cosmos_rho7``, ``flow_cosmos_dynamic``) are registered in ComfyUI's
KSampler scheduler table by ``extra_schedulers.py``. This module holds everything that needs neither torch nor ComfyUI,
so the app's own Python can import it too:

* **settings** — the eight values of sam-extra's Extra Schedulers block (``ARG_NAMES``: the custom
  scheduler's mode/expression/sigma list, Laplace's μ/β and the ρ / σ̃ max / σ̃ min Flow Cosmos rho7 and Flow
  Cosmos Dynamic share), their
  defaults, ranges and coercion (``coerce`` / ``coerce_args``): the rules of sam-extra
  ``sam3ext/extra_schedulers/settings.py`` and ``ui_extra_schedulers.coerce_args`` (a block of the
  five arguments sam-extra had before v0.33.0, or fewer, gives the rest their defaults);
* **the custom scheduler's expression** — ``compile_expression`` checks untrusted text against a
  whitelist and turns it into a small evaluator (``Expression.evaluate``); nothing in the text is
  ever compiled or run as Python. The accepted language, the bounds, the evaluation order and
  every error text are sam-extra's ``sam3ext/extra_schedulers/expression.py`` (the texts reach the
  user: ComfyUI shows them as the sampler node's error, Forge as its HTTP 500 detail);
* **the closed forms** in float64 (the glue stores them as float32), incl. React Cosinusoidal DynSF
  written from its published one-line formula (``react_cosinusoidal_dynsf_values``), Flow Cosmos
  rho7's flow times written from Karras et al.'s ρ ramp and the rectified-flow time map
  (``flow_cosmos_rho7_times``) and Flow Cosmos Dynamic's — Karras Dynamic's per-step-ρ ramp over the same
  σ̃ range, mapped the same way (``flow_cosmos_dynamic_times``).

Behaviour mirrored from sam-extra v0.33.0, and much of the code is sam-extra's (``expression.py``,
``sigma_list.py``, ``settings.py``, ``schedulers.py``), licensed to this pack under MIT by its author
(THIRD_PARTY_NOTICES.md). sam-extra is not imported (it is not installed next to ComfyUI);
``tests/test_comfy_extra_schedulers_parity.py`` runs both on the same inputs and requires the same results,
error classes and messages.

The expression language (for the user, in short): numbers; the variables ``m`` (sigma_min), ``M``
(sigma_max), ``n`` (steps), ``s`` (this step, from 0) and ``x`` (``s / (n - 1)``, 0 for one step);
the constants ``phi``, ``pi``, ``e``; ``+ - * / **`` and unary ``-``/``+``; calls of the functions in
``FUNCTIONS`` by bare name with positional arguments. At most ``MAX_LENGTH`` characters,
``MAX_NODES`` syntax nodes, nesting ``MAX_DEPTH``; ``**`` exponents within ``±MAX_EXPONENT``; every
intermediate value must be finite.
"""

from __future__ import annotations

import ast
import math
import re
from typing import Any, Callable, Iterable, Mapping, NamedTuple, Sequence


# ── settings (sam-extra settings.py / ui_extra_schedulers.ARG_NAMES) ──────────────────────────

# The block's argument order (positional form) and dict keys; the node's inputs use the same names.
# sam-extra v0.33.0 appended the three Flow Cosmos values (5 -> 8); missing trailing values take the defaults.
ARG_NAMES = ("custom_mode", "custom_expression", "custom_sigmas", "laplace_mu", "laplace_beta",
             "flow_cosmos_rho", "flow_cosmos_sigma_max", "flow_cosmos_sigma_min")

MODE_EXPRESSION = "expression"
MODE_SIGMAS = "sigmas"
MODES = (MODE_EXPRESSION, MODE_SIGMAS)
# What the mode field also accepts for "sigmas" (the Forge UI label "Sigma list" and spellings of it).
_SIGMAS_MODE_WORDS = frozenset({"sigmas", "sigma list", "sigma_list", "list", "sigma"})

# M at the first step, m at the last: the Exponential schedule's curve.
DEFAULT_EXPRESSION = "M * (m / M) ** x"
# Starts at 1.0 and ends at 0.0, so it is scaled between sigma_max and sigma_min.
DEFAULT_SIGMAS = "[1.0, 0.6, 0.25, 0.1, 0.0]"

# ComfyUI's LaplaceScheduler ranges and defaults (sam-extra uses the same).
LAPLACE_MU_MIN, LAPLACE_MU_MAX, LAPLACE_MU_DEFAULT = -10.0, 10.0, 0.0
LAPLACE_BETA_MIN, LAPLACE_BETA_MAX, LAPLACE_BETA_DEFAULT = 0.0, 10.0, 0.5

# Flow Cosmos rho7's values (sam-extra v0.33.0): ρ and the VE-equivalent noise range σ̃ max / σ̃ min its ρ ramp is
# laid on before the flow map t = σ̃/(1 + σ̃) — (min, max, default, step) as sam-extra's sliders; the node's FLOAT
# inputs use the same. The defaults are Cosmos-Predict2's ρ 7 on 0.002 … 80, the same numbers as the closed form's
# ``FLOW_COSMOS_RHO`` / ``FLOW_COSMOS_SIGMA_MAX`` / ``FLOW_COSMOS_SIGMA_MIN`` below (tested), so the defaults give the
# fixed list of before. Only σ̃ min = σ̃ max = 1 (where the two ranges meet) passes the clamps and is still refused.
# Flow Cosmos Dynamic (sam-extra v0.33.0 D20) reads the same three values; it has none of its own.
FLOW_COSMOS_RHO_MIN, FLOW_COSMOS_RHO_MAX, FLOW_COSMOS_RHO_DEFAULT, FLOW_COSMOS_RHO_STEP = 1.0, 15.0, 7.0, 0.1
FLOW_COSMOS_SIGMA_MAX_MIN, FLOW_COSMOS_SIGMA_MAX_MAX, FLOW_COSMOS_SIGMA_MAX_DEFAULT, FLOW_COSMOS_SIGMA_MAX_STEP = (
    1.0, 1000.0, 80.0, 0.1)
FLOW_COSMOS_SIGMA_MIN_MIN, FLOW_COSMOS_SIGMA_MIN_MAX, FLOW_COSMOS_SIGMA_MIN_DEFAULT, FLOW_COSMOS_SIGMA_MIN_STEP = (
    0.0001, 1.0, 0.002, 0.0001)


class Settings(NamedTuple):
    """One generation's Extra Schedulers values (``laplace``, ``custom``, ``flow_cosmos_rho7`` and
    ``flow_cosmos_dynamic`` read them)."""

    custom_mode: str = MODE_EXPRESSION
    custom_expression: str = DEFAULT_EXPRESSION
    custom_sigmas: str = DEFAULT_SIGMAS
    laplace_mu: float = LAPLACE_MU_DEFAULT
    laplace_beta: float = LAPLACE_BETA_DEFAULT
    flow_cosmos_rho: float = FLOW_COSMOS_RHO_DEFAULT
    flow_cosmos_sigma_max: float = FLOW_COSMOS_SIGMA_MAX_DEFAULT
    flow_cosmos_sigma_min: float = FLOW_COSMOS_SIGMA_MIN_DEFAULT


DEFAULTS = Settings()


def coerce_float(value: Any, default: float, low: float, high: float) -> float:
    """``value`` clamped to ``[low, high]``; ``default`` for a bool or anything not a finite number.

    An ``int`` too large for a float (``float()`` raises ``OverflowError``, e.g. an API request's JSON
    integer past ~1.8e308) counts as not finite and becomes the default, like ``1e400`` — sam-extra
    v0.33.0's rule (before, the error left its coercion and the whole block fell back to the defaults).
    """

    if isinstance(value, bool):
        return default
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        return default
    if math.isnan(number) or math.isinf(number):
        return default
    # max() then min() as sam-extra clamps: a -0.0 against the bound 0.0 becomes that 0.0, which
    # shows in the error texts ("beta 0", not "beta -0").
    return min(high, max(low, number))


def coerce_laplace_mu(value: Any) -> float:
    return coerce_float(value, LAPLACE_MU_DEFAULT, LAPLACE_MU_MIN, LAPLACE_MU_MAX)


def coerce_laplace_beta(value: Any) -> float:
    return coerce_float(value, LAPLACE_BETA_DEFAULT, LAPLACE_BETA_MIN, LAPLACE_BETA_MAX)


def coerce_flow_cosmos_rho(value: Any) -> float:
    return coerce_float(value, FLOW_COSMOS_RHO_DEFAULT, FLOW_COSMOS_RHO_MIN, FLOW_COSMOS_RHO_MAX)


def coerce_flow_cosmos_sigma_max(value: Any) -> float:
    return coerce_float(value, FLOW_COSMOS_SIGMA_MAX_DEFAULT, FLOW_COSMOS_SIGMA_MAX_MIN, FLOW_COSMOS_SIGMA_MAX_MAX)


def coerce_flow_cosmos_sigma_min(value: Any) -> float:
    return coerce_float(value, FLOW_COSMOS_SIGMA_MIN_DEFAULT, FLOW_COSMOS_SIGMA_MIN_MIN, FLOW_COSMOS_SIGMA_MIN_MAX)


def coerce_mode(value: Any) -> str:
    """``sigmas`` for the stored name or a label of the sigma-list mode, else ``expression``."""

    word = str(value or "").strip().lower()
    return MODE_SIGMAS if word in _SIGMAS_MODE_WORDS else MODE_EXPRESSION


def coerce_text(value: Any, default: str) -> str:
    """The text stripped (never shortened: a text that is too long fails when the schedule is
    built, instead of silently becoming another schedule); ``default`` only for ``None``."""

    return default if value is None else str(value).strip()


def coerce(raw: Mapping[str, Any] | None) -> Settings:
    """Settings from a dict of possibly missing or malformed values."""

    values = dict(raw or {})
    return Settings(
        custom_mode=coerce_mode(values.get("custom_mode", MODE_EXPRESSION)),
        custom_expression=coerce_text(values.get("custom_expression"), DEFAULT_EXPRESSION),
        custom_sigmas=coerce_text(values.get("custom_sigmas"), DEFAULT_SIGMAS),
        laplace_mu=coerce_laplace_mu(values.get("laplace_mu", LAPLACE_MU_DEFAULT)),
        laplace_beta=coerce_laplace_beta(values.get("laplace_beta", LAPLACE_BETA_DEFAULT)),
        flow_cosmos_rho=coerce_flow_cosmos_rho(values.get("flow_cosmos_rho", FLOW_COSMOS_RHO_DEFAULT)),
        flow_cosmos_sigma_max=coerce_flow_cosmos_sigma_max(
            values.get("flow_cosmos_sigma_max", FLOW_COSMOS_SIGMA_MAX_DEFAULT)),
        flow_cosmos_sigma_min=coerce_flow_cosmos_sigma_min(
            values.get("flow_cosmos_sigma_min", FLOW_COSMOS_SIGMA_MIN_DEFAULT)),
    )


def coerce_args(args: Iterable[Any] | None) -> Settings:
    """The block's arguments — positional in ``ARG_NAMES`` order, or one dict — as settings. A block of
    the five arguments of sam-extra v0.31–v0.32 (or fewer) gives the missing values their defaults."""

    items = list(args or ())
    if items and isinstance(items[0], dict):
        return coerce(items[0])
    return coerce(dict(zip(ARG_NAMES, items)))


# ── the custom scheduler's expression (sam-extra expression.py) ───────────────────────────────

MAX_LENGTH = 500
MAX_NODES = 200
MAX_DEPTH = 32
MAX_EXPONENT = 64.0
MAX_CALL_ARGS = 8

GOLDEN_RATIO = (1.0 + math.sqrt(5.0)) / 2.0

VARIABLES = ("m", "M", "n", "s", "x")
CONSTANTS = {"phi": GOLDEN_RATIO, "pi": math.pi, "e": math.e}


class ExpressionError(ValueError):
    """The text is not an allowed expression, or evaluating it gave no finite number."""


def _logarithm(*args: float) -> float:
    """``log(v)`` or ``log(v, base)`` — ``math.log`` with the same arguments."""

    return math.log(*args)


def _smallest(*values: float) -> float:
    return min(values)


def _largest(*values: float) -> float:
    return max(values)


# name -> (function, fewest arguments, most arguments). The order is the order the error texts list.
FUNCTIONS: dict[str, tuple[Callable[..., float], int, int]] = {
    "abs": (math.fabs, 1, 1),
    "sqrt": (math.sqrt, 1, 1),
    "exp": (math.exp, 1, 1),
    "log": (_logarithm, 1, 2),
    "log2": (math.log2, 1, 1),
    "log10": (math.log10, 1, 1),
    "sin": (math.sin, 1, 1),
    "cos": (math.cos, 1, 1),
    "tan": (math.tan, 1, 1),
    "asin": (math.asin, 1, 1),
    "acos": (math.acos, 1, 1),
    "atan": (math.atan, 1, 1),
    "atan2": (math.atan2, 2, 2),
    "sinh": (math.sinh, 1, 1),
    "cosh": (math.cosh, 1, 1),
    "tanh": (math.tanh, 1, 1),
    "floor": (math.floor, 1, 1),
    "ceil": (math.ceil, 1, 1),
    "min": (_smallest, 2, MAX_CALL_ARGS),
    "max": (_largest, 2, MAX_CALL_ARGS),
}

# An evaluator step: the variables' values in, one float out.
_Step = Callable[[Mapping[str, float]], float]


def _checked(value: float, what: str) -> float:
    if math.isnan(value) or math.isinf(value):
        raise ExpressionError(f"{what} gave a value that is not finite")
    return value


def _constant(value: float) -> _Step:
    return lambda env: value


def _variable(name: str) -> _Step:
    return lambda env: env[name]


def _negated(operand: _Step) -> _Step:
    return lambda env: -operand(env)


def _sum(left: _Step, right: _Step) -> _Step:
    return lambda env: _checked(left(env) + right(env), "an addition")


def _difference(left: _Step, right: _Step) -> _Step:
    return lambda env: _checked(left(env) - right(env), "a subtraction")


def _product(left: _Step, right: _Step) -> _Step:
    return lambda env: _checked(left(env) * right(env), "a multiplication")


def _quotient(left: _Step, right: _Step) -> _Step:
    def step(env):
        numerator = left(env)
        denominator = right(env)
        if denominator == 0.0:
            raise ExpressionError("division by zero")
        return _checked(numerator / denominator, "a division")

    return step


def _power(left: _Step, right: _Step) -> _Step:
    def step(env):
        base = left(env)
        exponent = right(env)
        if abs(exponent) > MAX_EXPONENT:
            raise ExpressionError(f"exponent {exponent:g} is outside ±{MAX_EXPONENT:g}")
        # math.pow: an error (not a complex number) for a negative base with a fractional
        # exponent, OverflowError (not a hang) for a huge result.
        return _checked(math.pow(base, exponent), "a power")

    return step


def _call(name: str, arguments: tuple[_Step, ...]) -> _Step:
    function = FUNCTIONS[name][0]
    what = f"{name}()"

    def step(env):
        values = [argument(env) for argument in arguments]
        return _checked(float(function(*values)), what)

    return step


# ast operator class -> (how the operator is written, the step builder)
_OPERATORS: dict[type, Callable[[_Step, _Step], _Step]] = {
    ast.Add: _sum,
    ast.Sub: _difference,
    ast.Mult: _product,
    ast.Div: _quotient,
    ast.Pow: _power,
}


class _Builder:
    """Walks the parsed tree once, depth first and left to right: every syntax node is counted
    and its depth checked before its kind, and anything off the whitelist raises before any value
    is computed. Returns the evaluator for the tree."""

    def __init__(self) -> None:
        self.count = 0

    def build(self, node: ast.AST, depth: int) -> _Step:
        self.count += 1
        if self.count > MAX_NODES:
            raise ExpressionError(f"the expression is too long (more than {MAX_NODES} parts)")
        if depth > MAX_DEPTH:
            raise ExpressionError(f"the expression is nested too deeply (more than {MAX_DEPTH} levels)")
        visit = _VISITORS.get(type(node))
        if visit is None:
            raise ExpressionError(f"{type(node).__name__} is not allowed in a scheduler expression")
        return visit(self, node, depth + 1)

    def literal(self, node: ast.Constant, below: int) -> _Step:
        value = node.value
        if type(value) is not int and type(value) is not float:   # bool, str, None, complex, ...
            raise ExpressionError(f"only numbers are allowed as literals, not {type(value).__name__}")
        try:
            number = float(value)
        except OverflowError:
            raise ExpressionError("a number in the expression is too large") from None
        if math.isnan(number) or math.isinf(number):
            raise ExpressionError("numbers in the expression must be finite")
        return _constant(number)

    def name(self, node: ast.Name, below: int) -> _Step:
        if node.id in VARIABLES:
            return _variable(node.id)
        if node.id in CONSTANTS:
            return _constant(CONSTANTS[node.id])
        allowed = ", ".join((*VARIABLES, *CONSTANTS))
        raise ExpressionError(
            f"unknown name {node.id!r}; allowed names are {allowed} and the functions {', '.join(FUNCTIONS)}"
        )

    def unary(self, node: ast.UnaryOp, below: int) -> _Step:
        kind = type(node.op)
        if kind is ast.USub:
            return _negated(self.build(node.operand, below))
        if kind is ast.UAdd:
            return self.build(node.operand, below)
        raise ExpressionError(f"the operator {kind.__name__} is not allowed")

    def binary(self, node: ast.BinOp, below: int) -> _Step:
        combine = _OPERATORS.get(type(node.op))
        if combine is None:
            raise ExpressionError(f"the operator {type(node.op).__name__} is not allowed (use + - * / **)")
        left = self.build(node.left, below)
        return combine(left, self.build(node.right, below))

    def call(self, node: ast.Call, below: int) -> _Step:
        target = node.func
        name = target.id if type(target) is ast.Name else None
        if name not in FUNCTIONS:
            raise ExpressionError("only these functions can be called by name: " + ", ".join(FUNCTIONS))
        if node.keywords:
            raise ExpressionError(f"{name}() takes no keyword arguments")
        if any(type(argument) is ast.Starred for argument in node.args):
            raise ExpressionError(f"{name}() does not accept *arguments")
        fewest, most = FUNCTIONS[name][1:]
        given = len(node.args)
        if given < fewest or given > most:
            expected = f"{fewest}" if fewest == most else f"{fewest} to {most}"
            raise ExpressionError(f"{name}() takes {expected} arguments, got {given}")
        return _call(name, tuple(self.build(argument, below) for argument in node.args))


_VISITORS: dict[type, Callable[[_Builder, Any, int], _Step]] = {
    ast.Constant: _Builder.literal,
    ast.Name: _Builder.name,
    ast.UnaryOp: _Builder.unary,
    ast.BinOp: _Builder.binary,
    ast.Call: _Builder.call,
}


class Expression:
    """A checked expression; ``source`` is the whitespace-normalised text that was parsed."""

    __slots__ = ("source", "_step")

    def __init__(self, source: str, step: _Step):
        self.source = source
        self._step = step

    def __repr__(self) -> str:
        return f"Expression({self.source!r})"

    def evaluate(self, **variables: float) -> float:
        """The value for these variables (``m``, ``M``, ``n``, ``s``, ``x``; others are ignored)."""

        env: dict[str, float] = {}
        for name in VARIABLES:
            if name not in variables:
                raise ExpressionError(f"missing value for {name!r}")
            value = float(variables[name])
            if math.isnan(value) or math.isinf(value):
                raise ExpressionError(f"{name} is not finite")
            env[name] = value
        try:
            return self._step(env)
        except ExpressionError:
            raise
        except ZeroDivisionError:
            raise ExpressionError("division by zero") from None
        except OverflowError:
            raise ExpressionError("a value in the expression overflowed") from None
        except (ValueError, TypeError) as exc:
            raise ExpressionError(f"math error in the expression: {exc}") from None


def normalize_source(text: Any) -> str:
    """Tabs and newlines become spaces, runs of spaces one, the ends stripped (a pasted
    multi-line expression still parses)."""

    if not isinstance(text, str):
        raise ExpressionError("the expression must be text")
    return " ".join(text.split())


def compile_expression(text: Any) -> Expression:
    """Check ``text`` and return its evaluator; ``ExpressionError`` if it is not allowed."""

    source = normalize_source(text)
    if not source:
        raise ExpressionError("the expression is empty")
    if len(source) > MAX_LENGTH:
        raise ExpressionError(f"the expression is longer than {MAX_LENGTH} characters")
    try:
        parsed = ast.parse(source, mode="eval")
    except SyntaxError as exc:
        raise ExpressionError(f"cannot parse the expression: {exc.msg}") from None
    except (ValueError, MemoryError, RecursionError) as exc:
        raise ExpressionError(f"cannot parse the expression: {type(exc).__name__}") from None
    return Expression(source, _Builder().build(parsed.body, 1))


def step_variables(step: int, steps: int, sigma_min: float, sigma_max: float) -> dict[str, float]:
    """The variables of step ``step`` (from 0) of ``steps``."""

    position = step / (steps - 1) if steps > 1 else 0.0
    return {"m": float(sigma_min), "M": float(sigma_max), "n": float(steps), "s": float(step), "x": position}


# ── errors the schedulers raise (sam-extra schedulers.py class names; ComfyUI shows the text) ──

class ExtraSchedulerError(ValueError):
    """A scheduler cannot make a usable schedule; the generation stops (nothing falls back)."""


class CustomSchedulerError(ExtraSchedulerError):
    """The custom scheduler's expression or sigma list cannot be used."""


class KarrasDynamicError(ExtraSchedulerError):
    """Karras Dynamic cannot make a falling schedule with this rho."""


class SigmaListError(ValueError):
    """The sigma list text cannot be used."""


# ── the custom scheduler's sigma list (sam-extra sigma_list.py) ──────────────────────────────

MAX_TEXT_LENGTH = 20000
MAX_VALUES = 1000

# A plain decimal literal with ASCII digits only (float() would also take other scripts' digits,
# "nan", "inf" and underscores — none of them is a sigma list item).
_PLAIN_NUMBER = re.compile(r"[+-]?(?:[0-9]+(?:\.[0-9]*)?|\.[0-9]+)(?:[eE][+-]?[0-9]+)?")
_ITEM_SEPARATOR = re.compile(r"[\s,]+")
_BRACKETS = {"[": "]", "(": ")"}


def parse_sigma_list(text: Any) -> list[float]:
    """The numbers of a sigma list: separated by commas and/or whitespace, optionally inside one
    pair of ``[]``/``()``. At least two, each finite and not negative."""

    if not isinstance(text, str):
        raise SigmaListError("the sigma list must be text")
    body = text.strip()
    if len(body) > MAX_TEXT_LENGTH:
        raise SigmaListError(f"the sigma list is longer than {MAX_TEXT_LENGTH} characters")
    opening = body[:1]
    if opening in _BRACKETS:
        closing = _BRACKETS[opening]
        if not body.endswith(closing):
            raise SigmaListError(f"the sigma list opens with {opening!r} but does not end with {closing!r}")
        body = body[1:-1]
    items = [item for item in _ITEM_SEPARATOR.split(body.strip()) if item]
    if len(items) < 2:
        raise SigmaListError("a sigma list needs at least two numbers")
    if len(items) > MAX_VALUES:
        raise SigmaListError(f"a sigma list can have at most {MAX_VALUES} numbers")
    numbers = []
    for item in items:
        shown = repr(item[:40])
        if _PLAIN_NUMBER.fullmatch(item) is None:
            raise SigmaListError(f"{shown} is not a plain number")
        try:
            number = float(item)
        except (ValueError, OverflowError):
            raise SigmaListError(f"{shown} is not a usable number") from None
        if math.isnan(number) or math.isinf(number):
            raise SigmaListError(f"{shown} is not finite")
        if number < 0.0:
            raise SigmaListError(f"{shown} is negative; sigmas cannot be below 0")
        numbers.append(number)
    return numbers


def is_normalized(values: Sequence[float]) -> bool:
    """A list from exactly 1.0 to exactly 0.0 is scaled between sigma_max and sigma_min."""

    return len(values) >= 2 and values[0] == 1.0 and values[-1] == 0.0


# interp(values, count) -> count values: the host's log-linear interpolation
Interpolation = Callable[[list, int], Iterable[float]]


def sigmas_from_list(text: Any, steps: int, sigma_min: float, sigma_max: float, *,
                     interp: Interpolation) -> list[float]:
    """``steps`` sigmas (without the final 0) for the sigma list ``text``.

    A normalised list is scaled (1.0 -> sigma_max, 0.0 -> sigma_min); any other list is taken as
    sigmas, a trailing 0.0 read as the final zero and dropped. What remains must be above 0 and is
    log-linearly interpolated to ``steps`` values by ``interp(values, steps)`` — the host's own
    function (ComfyUI ``nodes_align_your_steps.loglinear_interp`` = Forge
    ``sd_schedulers._loglinear_interp``) — when the counts differ. One step takes the first value.
    """

    values = parse_sigma_list(text)
    if is_normalized(values):
        low, high = float(sigma_min), float(sigma_max)
        values = [low + value * (high - low) for value in values]
    elif values[-1] == 0.0:
        values = values[:-1]
    for number, value in enumerate(values, start=1):
        if not value > 0.0:
            raise SigmaListError(
                f"sigma #{number} is {value:g}; sigmas before the end must be above 0 (a list that starts "
                "with 1.0 and ends with 0.0 is scaled between sigma_max and sigma_min instead)"
            )
    if steps <= 0:
        return []
    if steps == 1:
        # the interpolation would put one position at the list's last (smallest) value
        return [float(values[0])]
    if len(values) == steps:
        return [float(value) for value in values]
    result = [float(value) for value in interp(values, steps)]
    if len(result) != steps or any(math.isnan(v) or math.isinf(v) or not v > 0.0 for v in result):
        raise SigmaListError("the interpolated sigma list is not usable")
    return result


# ── closed forms (sam-extra schedulers.py; Python float64, the glue stores them as float32) ───

KARRAS_DYNAMIC_RHO = 7.0
# React Cosinusoidal DynSF's exponent factor f (fixed, like sam-extra; reForge's default for its setting).
REACT_DYNSF_FACTOR = 2.15
# Flow Cosmos rho7: Cosmos-Predict2's EDM noise range (ComfyUI's supported_models gives Cosmos the same one) and
# Karras's rho — the defaults of the node's Flow Cosmos values (FLOW_COSMOS_*_DEFAULT above, the same numbers; module
# literals the app pins by AST). A model that is not a flow model gets the host's Karras with the run's rho on its own
# sigma range.
FLOW_COSMOS_SIGMA_MAX = 80.0
FLOW_COSMOS_SIGMA_MIN = 0.002
FLOW_COSMOS_RHO = 7.0
# Logged once per process the first time Flow Cosmos rho7 runs on a model that is not a flow model (sam-extra's line),
# formatted with the rho of that run (with the default 7: "... using Karras rho 7 on the model's own sigma range").
FLOW_COSMOS_NOT_FLOW_LOG = ("[Extra Schedulers] Flow Cosmos rho7: not a flow model - using Karras rho {rho:g} on the "
                            "model's own sigma range")
# The same for Flow Cosmos Dynamic (sam-extra's line; its own once-per-process flag): on a model that is not a flow
# model it is Karras Dynamic with the run's rho.
FLOW_COSMOS_DYNAMIC_NOT_FLOW_LOG = ("[Extra Schedulers] Flow Cosmos Dynamic: not a flow model - using Karras Dynamic "
                                    "rho {rho:g} on the model's own sigma range")
# Where Flow Cosmos rho7's values come from, named in its error texts (sam-extra's words: on ComfyUI they are the
# ForgeNeoExtraSchedulers node's flow_cosmos_rho / flow_cosmos_sigma_max / flow_cosmos_sigma_min inputs).
FLOW_COSMOS_VALUES_SOURCE = "Extra Schedulers accordion → Flow Cosmos rho / sigma max / sigma min"
# Where Forge's rho comes from (named in the Karras Dynamic error texts).
RHO_SOURCE = ("Settings → Sampler Parameters → rho (shared with Karras; listed with --adv-samplers), "
              "or a 'Schedule rho' override from pasted infotext")
# get_sigmas_laplace's "avoid log(0)" epsilon; laplace_position inverts that same curve.
LAPLACE_EPSILON = 1e-5
# Why a schedule may not keep sigma where it was (ends the "flat step" error texts).
ZERO_LENGTH_STEP = ("a step that keeps sigma where it was is wasted with Euler and makes multistep samplers such as "
                    "Res Multistep divide by zero (NaN, a black image)")


def positions(n: int) -> list[float]:
    """``p = i / (n - 1)`` for step ``i`` of ``n`` (``[0.0]`` for one step, nothing for none)."""

    if n <= 1:
        return [0.0] * max(n, 0)
    last = n - 1
    return [i / last for i in range(n)]


def require_positive_range(name: str, sigma_min: float, sigma_max: float) -> None:
    if sigma_min > 0.0 and sigma_max > 0.0:
        return
    raise ExtraSchedulerError(
        f"Extra Schedulers: {name} needs sigma_min and sigma_max above 0 (got {sigma_min:g} and {sigma_max:g})"
    )


def cosine_value(p: float, sigma_min: float, sigma_max: float) -> float:
    """``σmin + ½(σmax − σmin)(1 − cos(π(1 − √p)))``."""

    return sigma_min + 0.5 * (sigma_max - sigma_min) * (1.0 - math.cos(math.pi * (1.0 - math.sqrt(p))))


def exponential_value(p: float, sigma_min: float, sigma_max: float) -> float:
    """k-diffusion ``get_sigmas_exponential`` at position ``p``: ``exp(lerp(log σmax, log σmin, p))``."""

    top = math.log(sigma_max)
    return math.exp(top + (math.log(sigma_min) - top) * p)


def cosine_values(n: int, sigma_min: float, sigma_max: float) -> list[float]:
    return [cosine_value(p, sigma_min, sigma_max) for p in positions(n)]


def cosine_exponential_values(n: int, sigma_min: float, sigma_max: float) -> list[float]:
    """``(1 − p)·Cosine + p·Exponential`` (both ends must be above 0 for the exponential)."""

    require_positive_range("CosineExponential blend", sigma_min, sigma_max)
    return [
        (1.0 - p) * cosine_value(p, sigma_min, sigma_max) + p * exponential_value(p, sigma_min, sigma_max)
        for p in positions(n)
    ]


def phi_values(n: int, sigma_min: float, sigma_max: float) -> list[float]:
    """``σmin + (σmax − σmin)(1 − p)^(φ²)`` (the idea of Extraltodeus's Golden Scheduler)."""

    exponent = GOLDEN_RATIO ** 2
    return [sigma_min + (sigma_max - sigma_min) * (1.0 - p) ** exponent for p in positions(n)]


def karras_dynamic_ramp(n: int, sigma_min: float, sigma_max: float, rho: float) -> tuple[list[float], int | None]:
    """The Karras ramp from ``sigma_max`` to ``sigma_min`` with a per-step exponent ``ρᵢ = ρ + 2cos(2πi/n)``.

    Written ``σmax((1 − p) + p(σmin/σmax)^(1/ρᵢ))^ρᵢ`` (no overflow for a small ρᵢ) and exactly σmin at
    ``p = 1``. Returns the values before the first step that is not finite (an overflow counts as one) and that
    step — None when every step is finite. No checks here: ``karras_dynamic_values`` (a model's σ range) and the
    Flow Cosmos Dynamic glue (the VE-equivalent σ̃ range, ``flow_cosmos_dynamic_times``) check the inputs first.
    """

    ratio = sigma_min / sigma_max
    values: list[float] = []
    for i, p in enumerate(positions(n)):
        if p >= 1.0:
            # computed, ratio ** (1/ρᵢ) can underflow to 0 for a tiny ρᵢ and give sigma 0
            values.append(sigma_min)
            continue
        exponent = rho + 2.0 * math.cos(2.0 * math.pi * i / n)
        try:
            value = sigma_max * ((1.0 - p) + p * ratio ** (1.0 / exponent)) ** exponent
        except OverflowError:
            value = math.inf
        if math.isnan(value) or math.isinf(value):
            return values, i
        values.append(value)
    return values, None


def require_karras_dynamic_rho(rho: float, name: str = "Karras Dynamic", rho_source: str = RHO_SOURCE) -> None:
    """``rho`` must be above 2 (every per-step exponent, ρ − 2 … ρ + 2, positive); ``name`` and ``rho_source``
    (where the rho comes from) go into the text."""

    if rho > 2.0:
        return
    raise KarrasDynamicError(
        f"Extra Schedulers: {name} needs rho above 2 (its per-step exponent goes down to rho - 2), "
        f"got {rho:g}. Set rho to about 4 or more: {rho_source}."
    )


def karras_dynamic_values(n: int, sigma_min: float, sigma_max: float, rho: float = KARRAS_DYNAMIC_RHO, *,
                          name: str = "Karras Dynamic", rho_source: str = RHO_SOURCE) -> list[float]:
    """Karras Dynamic's float64 values on a sigma range (``karras_dynamic_ramp``) after its checks: both ends
    above 0, ``rho`` above 2, every step finite. Whether the float32 schedule falls is checked by the caller.
    ``name`` / ``rho_source`` name the scheduler and the rho's origin in the texts (Flow Cosmos Dynamic on a model
    that is not a flow model is Karras Dynamic under its own name, with the node's rho)."""

    require_positive_range(name, sigma_min, sigma_max)
    require_karras_dynamic_rho(rho, name, rho_source)
    values, bad = karras_dynamic_ramp(n, sigma_min, sigma_max, rho)
    if bad is not None:
        raise KarrasDynamicError(
            f"Extra Schedulers: {name} gave no finite sigma at step {bad} (rho {rho:g}, "
            f"sigma_min {sigma_min:g}, sigma_max {sigma_max:g})"
        )
    return values


def react_cosinusoidal_dynsf_values(n: int, sigma_min: float, sigma_max: float) -> list[float]:
    """React Cosinusoidal DynSF: ``σmax·((σmin + (σmax − σmin)·cos(πp/2)) / σmax)^(f·p)``, f =
    ``REACT_DYNSF_FACTOR`` — σmax at p = 0, σmax·(σmin/σmax)^f at p = 1 (both ends must be above 0).

    Written from the formula yoinked-h published (reForge ships it; no reForge code was read or
    used). A σ-range formula: the model's shift is ignored and many steps stay at high σ."""

    require_positive_range("React Cosinusoidal DynSF", sigma_min, sigma_max)
    values: list[float] = []
    for p in positions(n):
        blend = sigma_min + (sigma_max - sigma_min) * math.cos(math.pi * p / 2.0)
        values.append(sigma_max * (blend / sigma_max) ** (REACT_DYNSF_FACTOR * p))
    return values


def flow_cosmos_rho7_times(n: int, rho: float = FLOW_COSMOS_RHO, ve_sigma_min: float = FLOW_COSMOS_SIGMA_MIN,
                           ve_sigma_max: float = FLOW_COSMOS_SIGMA_MAX) -> list[float]:
    """Flow Cosmos rho7 on a flow model: the flow times ``t = s / (1 + s)`` of Karras et al.'s ρ ramp
    (arXiv:2206.00364, eq. 5) over the VE-equivalent noise levels σ̃,
    ``s = (σ̃max^(1/ρ) + p·(σ̃min^(1/ρ) − σ̃max^(1/ρ)))^ρ`` — σ̃max/(1 + σ̃max) at p = 0, σ̃min/(1 + σ̃min) at
    p = 1, whatever the model's own sigma range. The defaults are Cosmos-Predict2's ρ 7 on 0.002 … 80
    (``FLOW_COSMOS_*``): 80/81 at p = 0, about 0.002 at p = 1.

    The float64 operations are those of the custom expression
    ``(80**(1/7)+x*(0.002**(1/7)-80**(1/7)))**7/(1+(80**(1/7)+x*(0.002**(1/7)-80**(1/7)))**7)`` with ρ, σ̃max
    and σ̃min in place of 7, 80 and 0.002, so the two give the same values. Scaling the σ̃ range by k is the flow
    time shift k (t' = k·t/(1 + (k − 1)·t) ⇔ σ̃' = k·σ̃): Anima's shift 3 on the default list is σ̃ 0.006 … 240 —
    a fact of the maths, not a quality claim. A ρ or range past the node's limits can raise ``OverflowError``
    here (the caller turns it into ``ExtraSchedulerError``). The name and the idea are
    KeithZ117/Comfyui-anima-sampler's (commit effba3c5, MIT); this is written from the formula — no code of
    that repository was copied."""

    root_max = ve_sigma_max ** (1.0 / rho)
    root_min = ve_sigma_min ** (1.0 / rho)
    times: list[float] = []
    for p in positions(n):
        level = (root_max + p * (root_min - root_max)) ** rho
        times.append(level / (1.0 + level))
    return times


def require_flow_cosmos_rho(rho: float) -> None:
    """Flow Cosmos rho7's ρ must be a finite number above 0 (both branches; the node's input keeps it in 1 … 15)."""

    if math.isfinite(rho) and rho > 0.0:
        return
    raise ExtraSchedulerError(
        f"Extra Schedulers: Flow Cosmos rho7 needs a finite rho above 0, got {rho:g} ({FLOW_COSMOS_VALUES_SOURCE})"
    )


def require_flow_cosmos_range(ve_sigma_min: float, ve_sigma_max: float, name: str = "Flow Cosmos rho7") -> None:
    """On a flow model 0 < σ̃ min < σ̃ max, both finite (the node's inputs only meet at 1 = 1); ``name`` is the
    scheduler the text names (Flow Cosmos rho7 or Flow Cosmos Dynamic, which share the values)."""

    if math.isfinite(ve_sigma_min) and math.isfinite(ve_sigma_max) and 0.0 < ve_sigma_min < ve_sigma_max:
        return
    raise ExtraSchedulerError(
        f"Extra Schedulers: {name} needs 0 < sigma~ min < sigma~ max, both finite, got sigma~ min "
        f"{ve_sigma_min:g} and sigma~ max {ve_sigma_max:g} ({FLOW_COSMOS_VALUES_SOURCE})"
    )


def flow_cosmos_dynamic_times(n: int, rho: float = FLOW_COSMOS_RHO, ve_sigma_min: float = FLOW_COSMOS_SIGMA_MIN,
                              ve_sigma_max: float = FLOW_COSMOS_SIGMA_MAX) -> tuple[list[float], int | None]:
    """Flow Cosmos Dynamic on a flow model: Karras Dynamic's ramp (``karras_dynamic_ramp``, per-step exponent
    ``ρᵢ = ρ + 2cos(2πi/n)``) laid on the VE-equivalent range σ̃ min … σ̃ max instead of a model's sigma range, each
    value s mapped to flow time ``t = s / (1 + s)`` — σ̃max/(1 + σ̃max) at p = 0, σ̃min/(1 + σ̃min) at p = 1. The
    defaults are Flow Cosmos rho7's (ρ 7 on 0.002 … 80). Returns the times before the first step whose s is not
    finite and that step (None when every one is); the caller checks ρ above 2 and the σ̃ range first.

    The composition is sam-extra v0.33.0's own (its "Flow Cosmos Dynamic"; Karras Dynamic's origin is
    unverified); this follows sam-extra's code (its author's MIT grant), not another project's. Since ρᵢ = ρ
    only where cos(2πi/n) = 0, the list meets Flow Cosmos rho7's (same ρ and range) at the first and the last
    step and, when 4 divides n, at steps n/4 and 3n/4; elsewhere its σ̃ is lower in the first and last quarter
    and higher in the middle half."""

    levels, bad = karras_dynamic_ramp(n, ve_sigma_min, ve_sigma_max, rho)
    return [level / (1.0 + level) for level in levels], bad


def laplace_position(sigma: float, mu: float, beta: float) -> float:
    """Where on ``get_sigmas_laplace``'s x axis (0 … 1) the curve passes ``sigma``, clamped (beta > 0).

    The curve's log is ``mu − beta·log(2x + ε)`` below x = 0.5 and ``mu + beta·log(2 − 2x + ε)``
    above it; both exponents below are at most 0, so nothing overflows.
    """

    if not sigma > 0.0:
        return 1.0
    level = math.log(sigma)
    if level >= mu:
        x = (math.exp((mu - level) / beta) - LAPLACE_EPSILON) / 2.0
    else:
        x = 1.0 - (math.exp((level - mu) / beta) - LAPLACE_EPSILON) / 2.0
    return min(max(x, 0.0), 1.0)


def laplace_curve_ends(mu: float, beta: float) -> tuple[float, float]:
    """The highest and lowest sigma ``get_sigmas_laplace``'s curve reaches (named in its error)."""

    return math.exp(mu - beta * math.log(LAPLACE_EPSILON)), math.exp(mu + beta * math.log(LAPLACE_EPSILON))


def sigmas_from_expression(text: Any, steps: int, sigma_min: float, sigma_max: float) -> list[float]:
    """``steps`` sigmas (without the final 0): the expression at each step, each above 0."""

    compiled = compile_expression(text)
    count = int(steps)
    values = []
    for step in range(count):
        value = compiled.evaluate(**step_variables(step, count, sigma_min, sigma_max))
        if not value > 0.0:
            raise ExpressionError(
                f"step {step} gives sigma {value:g}; every step's sigma must be above 0 — the final 0 is "
                "added after the last step (end at m, e.g. m + (M - m) * (1 - x) ** 3)"
            )
        values.append(value)
    return values
