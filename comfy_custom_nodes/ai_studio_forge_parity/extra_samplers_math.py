"""Extra Samplers for ComfyUI (pack 1.7.0): the run's values and the three ER SDE samplers.

sam-extra's "Extra Samplers" accordion holds five values — ``max_stage``, ``eta``, ``noise_window``,
``noise_start``, ``noise_end`` — that only its three ER SDE entries read. The app sends them to
ComfyUI as a MODEL patch (``ForgeNeoExtraSamplerSettings`` → ``model_options[MODEL_OPTIONS_KEY]``),
which ``CFGGuider`` hands to every sampler function in ``extra_args["model_options"]``; absent → the
defaults, exactly as a Forge request without the block runs the script's defaults.

**The ER SDE trio** runs **ComfyUI's own** ``comfy.k_diffusion.sampling.sample_er_sde`` — the solver
Forge Neo copied from ComfyUI v0.3.75, term for term the same body — with one of three noise scale
functions ``h(λ)`` of the ER-SDE paper (Cui et al., "Elucidating the Solution Space of Extended
Reverse-Time SDE for Diffusion Models", arXiv:2309.06169), written here from their formulas:

* ``er_sde_reverse_time`` — ``h(λ) = λ^(η+1)`` (the classical reverse-time SDE at η = 1);
* ``er_sde_ode`` — ``h(λ) = λ`` and no noise (``s_noise = 0``): deterministic;
* ``er_sde_tunable`` — ``h(λ) = λ·(e^(λ^0.3) + 10)^η``: the paper's tuned scaler with η as exponent
  of its noise factor — at η = 1 exactly ComfyUI's default ``er_sde`` (``t ** 1.0`` is exact).

η = 0 is the ODE for the two SDE kinds (ODE scaler, ``s_noise = 0``, nothing drawn) — the rule of
ComfyUI's ``SamplerER_SDE`` node, so ``er_sde_tunable`` equals that node's "ER-SDE" solver type and
``er_sde_reverse_time`` its "Reverse-time SDE" on the same inputs.

**The noise window** (Reverse-time and Tunable, η > 0): with ``noise_window`` on, only the steps whose
target σ lies in ``[percent_to_sigma(end), percent_to_sigma(start)]`` (sampling percentages of the
model's own schedule) inject noise; every other step uses the ODE scaler and draws nothing. Every
step of the solver is a consistent step of a marginal-preserving reverse SDE picked by the ``h`` used
inside that step, so choosing ``h`` per step keeps the marginals (paper §3). It is done without a
copy of the solver loop: one call of the host's ``sample_er_sde`` with three per-call wrappers that
share the step index — the callback (which the solver calls right after the step's model evaluation
and before its first use of the scaler) records the index, the scaler picks the SDE or the ODE ``h``
for that step, and the noise sampler draws only for an in-window step (zeros, without drawing,
elsewhere). ``start ≥ end`` is an empty window (no step gets noise). The idea of a σ window for
ER-SDE's noise is pamparamm/ComfyUI-ppm's (AGPL-3.0); nothing of it was opened or copied.

Behaviour mirrored from sam-extra v0.33.0 ``sam3ext/extra_samplers`` (``params``, ``er_sde``,
``er_sde_window``: same coercion, the same scalers, the same window rule); the code is this pack's
own — sam-extra (GPL-3.0) is not imported or copied. ``tests/test_comfy_extra_samplers_parity.py``
runs both on Forge's k-diffusion module and compares the latents bit for bit.

Each sampler takes ``kd`` (the k-diffusion module; default ``comfy.k_diffusion.sampling``, resolved
at call time) so the parity tests can hand it Forge's module. No torch or ComfyUI import at module
level (the app imports the pack in its own tests).
"""

from __future__ import annotations

import importlib
import inspect
import logging
import math
from typing import Any, Callable, Mapping, NamedTuple


LOGGER = logging.getLogger("ai_studio_forge_parity")

# ``model_options`` key of the run's values (written by ForgeNeoExtraSamplerSettings).
MODEL_OPTIONS_KEY = "ai_studio_extra_samplers"

# The Extra Samplers script's positional arguments, in order (sam-extra ``params.ARG_NAMES``).
ARG_NAMES = ("max_stage", "eta", "noise_window", "noise_start", "noise_end")

# Ranges and defaults (sam-extra ``params``; the stage and η are ComfyUI SamplerER_SDE's inputs).
DEFAULT_MAX_STAGE = 3
MAX_STAGE_MIN = 1
MAX_STAGE_MAX = 3
DEFAULT_ETA = 1.0
ETA_MIN = 0.0
ETA_MAX = 10.0
ETA_STEP = 0.01
DEFAULT_NOISE_WINDOW = False
DEFAULT_NOISE_START = 0.2
DEFAULT_NOISE_END = 0.8
NOISE_PERCENT_MIN = 0.0
NOISE_PERCENT_MAX = 1.0
NOISE_PERCENT_STEP = 0.01

_TRUE_TEXTS = ("true", "on", "yes", "1")

# ComfyUI SamplerER_SDE ``solver_type`` values of the three scalers.
ER_SDE = "ER-SDE"
REVERSE_TIME = "Reverse-time SDE"
ODE = "ODE"
SOLVER_TYPES = (ER_SDE, REVERSE_TIME, ODE)

# The pack's ER SDE sampler names (= sam-extra's first aliases) and their solver types.
ER_SDE_SAMPLERS = {
    "er_sde_reverse_time": REVERSE_TIME,
    "er_sde_ode": ODE,
    "er_sde_tunable": ER_SDE,
}
# The names η and the noise window apply to.
WINDOW_SAMPLERS = ("er_sde_reverse_time", "er_sde_tunable")


class Settings(NamedTuple):
    max_stage: int = DEFAULT_MAX_STAGE
    eta: float = DEFAULT_ETA
    noise_window: bool = DEFAULT_NOISE_WINDOW
    noise_start: float = DEFAULT_NOISE_START
    noise_end: float = DEFAULT_NOISE_END

    def as_dict(self) -> dict[str, Any]:
        return dict(self._asdict())


DEFAULTS = Settings()


# ---------------------------------------------------------------------------
# Coercion (sam-extra ``params.coerce_*``: the same accept/clamp/default rules)
# ---------------------------------------------------------------------------


def _finite(value: Any) -> float | None:
    # OverflowError (an int too large for a float) propagates, as in sam-extra.
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def coerce_max_stage(value: Any, default: int = DEFAULT_MAX_STAGE) -> int:
    """An integer stage 1..3 (rounded, clamped); not a finite number → ``default``."""
    number = _finite(value)
    if number is None:
        return default
    return int(min(MAX_STAGE_MAX, max(MAX_STAGE_MIN, int(round(number)))))


def coerce_eta(value: Any, default: float = DEFAULT_ETA) -> float:
    """η clamped to 0..10; not a finite number → ``default``."""
    number = _finite(value)
    if number is None:
        return default
    return float(min(ETA_MAX, max(ETA_MIN, number)))


def coerce_noise_window(value: Any) -> bool:
    """The checkbox: True for ``True``, the number 1 and the texts true/on/yes/1 (any case)."""
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return value == 1
    if isinstance(value, str):
        return value.strip().lower() in _TRUE_TEXTS
    return False


def coerce_noise_percent(value: Any, default: float) -> float:
    """A sampling percentage clamped to 0..1; a bool or anything not finite → ``default``."""
    if isinstance(value, bool):
        return default
    number = _finite(value)
    if number is None:
        return default
    return float(min(NOISE_PERCENT_MAX, max(NOISE_PERCENT_MIN, number)))


def coerce(values: Mapping[str, Any] | Settings | None) -> Settings:
    """``Settings`` from a mapping with any of ``ARG_NAMES`` (missing → defaults)."""
    if isinstance(values, Settings):
        values = values.as_dict()
    if not isinstance(values, Mapping):
        return DEFAULTS
    return Settings(
        max_stage=coerce_max_stage(values.get("max_stage", DEFAULT_MAX_STAGE)),
        eta=coerce_eta(values.get("eta", DEFAULT_ETA)),
        noise_window=coerce_noise_window(values.get("noise_window", DEFAULT_NOISE_WINDOW)),
        noise_start=coerce_noise_percent(values.get("noise_start", DEFAULT_NOISE_START), DEFAULT_NOISE_START),
        noise_end=coerce_noise_percent(values.get("noise_end", DEFAULT_NOISE_END), DEFAULT_NOISE_END),
    )


def coerce_args(args: Any) -> Settings:
    """``Settings`` from the script's positional arguments (missing → defaults, extra ignored)."""
    values = list(args or ())
    return coerce({name: values[index] for index, name in enumerate(ARG_NAMES) if index < len(values)})


def settings_from_model_options(model_options: Any) -> Settings:
    """The run's values from ``model_options[MODEL_OPTIONS_KEY]`` (absent → the defaults)."""
    if not isinstance(model_options, Mapping):
        return DEFAULTS
    return coerce(model_options.get(MODEL_OPTIONS_KEY))


def settings_from_extra_args(extra_args: Any) -> Settings:
    """The run's values as a sampler function sees them (``extra_args["model_options"]``)."""
    if not isinstance(extra_args, Mapping):
        return DEFAULTS
    return settings_from_model_options(extra_args.get("model_options"))


def settings_window(settings: Settings) -> tuple[float, float] | None:
    """``(start, end)`` when the run asks for the window and η > 0 (η = 0 is the ODE), else None."""
    if not settings.noise_window or settings.eta <= 0:
        return None
    return (settings.noise_start, settings.noise_end)


# ---------------------------------------------------------------------------
# Host plumbing
# ---------------------------------------------------------------------------


def k_diffusion(kd: Any = None) -> Any:
    """``kd`` or ComfyUI's ``comfy.k_diffusion.sampling`` (resolved at call time)."""
    return kd if kd is not None else importlib.import_module("comfy.k_diffusion.sampling")


def model_sampling(model: Any) -> Any:
    """The model sampling object ``sample_er_sde`` reads.

    ComfyUI: ``model.inner_model.model_patcher.get_model_object("model_sampling")``; a model without
    a patcher is read the way Forge's copy reads it, ``model.inner_model.predictor`` (the parity
    tests run these samplers on Forge's k-diffusion module)."""
    inner = model.inner_model
    patcher = getattr(inner, "model_patcher", None)
    if patcher is not None:
        return patcher.get_model_object("model_sampling")
    return inner.predictor


def default_noise_sampler(kd: Any, x: Any, extra_args: Mapping[str, Any] | None) -> Callable[..., Any]:
    """The noise sampler the host's ``sample_er_sde`` would build itself when given none.

    ComfyUI seeds its own generator from ``extra_args["seed"]``; Forge's copy takes only ``x``."""
    build = kd.default_noise_sampler
    try:
        takes_seed = "seed" in inspect.signature(build).parameters
    except (TypeError, ValueError):
        takes_seed = True
    if takes_seed:
        return build(x, seed=(extra_args or {}).get("seed", None))
    return build(x)


def _no_grad():
    return importlib.import_module("torch").no_grad()


# ---------------------------------------------------------------------------
# Noise scale functions h(λ) (arXiv:2309.06169) and the SamplerER_SDE rule
# ---------------------------------------------------------------------------


def tunable_scaler(eta: float) -> Callable[[Any], Any]:
    """``h(λ) = λ·(e^(λ^0.3) + 10)^η``."""

    def scaler(x):
        return x * ((x ** 0.3).exp() + 10.0) ** eta

    return scaler


def reverse_time_scaler(eta: float) -> Callable[[Any], Any]:
    """``h(λ) = λ^(η+1)``."""

    def scaler(x):
        return x ** (eta + 1)

    return scaler


def ode_scaler(x: Any) -> Any:
    """``h(λ) = λ``: the noise term of the step vanishes."""
    return x


_SDE_SCALERS = {ER_SDE: tunable_scaler, REVERSE_TIME: reverse_time_scaler}


def er_sde_kwargs(solver_type: str, max_stage: int, eta: float, s_noise: float) -> dict[str, Any]:
    """``s_noise``, ``noise_scaler`` and ``max_stage`` for ``sample_er_sde``.

    ``ODE`` or η = 0 → the ODE scaler with ``s_noise = 0`` (nothing is drawn)."""
    if solver_type not in SOLVER_TYPES:
        raise ValueError(f"unknown ER SDE solver type: {solver_type!r}")
    if solver_type == ODE or eta == 0:
        return {"s_noise": 0.0, "noise_scaler": ode_scaler, "max_stage": max_stage}
    return {"s_noise": s_noise, "noise_scaler": _SDE_SCALERS[solver_type](eta), "max_stage": max_stage}


# ---------------------------------------------------------------------------
# The noise window
# ---------------------------------------------------------------------------


def window_bounds(sampling: Any, start: float, end: float) -> tuple[float, float]:
    """``(sigma_hi, sigma_lo)`` of the window ``start..end`` on ``sampling`` (``percent_to_sigma``)."""
    percent_to_sigma = getattr(sampling, "percent_to_sigma", None)
    if not callable(percent_to_sigma):
        raise ValueError(
            "the ER SDE noise window needs the model's percent_to_sigma, which this model sampling lacks"
        )
    return float(percent_to_sigma(float(start))), float(percent_to_sigma(float(end)))


def in_window_steps(sigmas: Any, start: float, end: float, sigma_hi: float, sigma_lo: float) -> list[bool]:
    """Per step ``i`` (σ_i → σ_{i+1}): does the σ it lands on lie in the window? ``start ≥ end`` → none."""
    count = len(sigmas) - 1
    if float(start) >= float(end):
        return [False] * count
    return [sigma_lo <= float(sigmas[index + 1]) <= sigma_hi for index in range(count)]


class StepWindow:
    """The three per-call wrappers of one windowed ``sample_er_sde`` call (module docstring)."""

    def __init__(self, inside: list[bool], sde_scaler: Callable[[Any], Any], noise_sampler: Callable[..., Any],
                 like: Any, callback: Callable[..., Any] | None = None):
        self.inside = list(inside)
        self.sde_scaler = sde_scaler
        self.real_noise_sampler = noise_sampler
        self.like = like
        self.real_callback = callback
        self.step: int | None = None
        self.draws = 0

    def _step_inside(self) -> bool:
        if self.step is None:
            # The solver must announce a step before it scales; never guess another step's choice.
            raise RuntimeError("ER SDE noise window: the noise scaler ran before the step's callback")
        return self.inside[self.step]

    def callback(self, info: dict[str, Any]) -> None:
        self.step = int(info["i"])
        if self.real_callback is not None:
            self.real_callback(info)

    def scaler(self, x: Any) -> Any:
        return self.sde_scaler(x) if self._step_inside() else ode_scaler(x)

    def noise_sampler(self, sigma: Any, sigma_next: Any) -> Any:
        if self._step_inside():
            self.draws += 1
            return self.real_noise_sampler(sigma, sigma_next)
        return importlib.import_module("torch").zeros_like(self.like)


def run_er_sde(model, x, sigmas, *, extra_args=None, callback=None, disable=None, noise_sampler=None,
               solver: Mapping[str, Any], window: tuple[float, float] | None = None, kd: Any = None):
    """The host's ``sample_er_sde`` with ``solver`` (``er_sde_kwargs``); windowed when ``window`` is
    given and the solver is an SDE (the ODE has no noise to restrict)."""
    kd = k_diffusion(kd)
    if window is None or solver["noise_scaler"] is ode_scaler:
        return kd.sample_er_sde(
            model, x, sigmas, extra_args=extra_args, callback=callback, disable=disable,
            noise_sampler=noise_sampler, **solver,
        )
    start, end = float(window[0]), float(window[1])
    sigma_hi, sigma_lo = window_bounds(model_sampling(model), start, end)
    inside = in_window_steps(sigmas, start, end, sigma_hi, sigma_lo)
    real_noise = default_noise_sampler(kd, x, extra_args) if noise_sampler is None else noise_sampler
    wrappers = StepWindow(inside, solver["noise_scaler"], real_noise, x, callback=callback)
    return kd.sample_er_sde(
        model, x, sigmas, extra_args=extra_args, callback=wrappers.callback, disable=disable,
        s_noise=solver["s_noise"], noise_sampler=wrappers.noise_sampler, noise_scaler=wrappers.scaler,
        max_stage=solver["max_stage"],
    )


# ---------------------------------------------------------------------------
# The three sampler functions (ComfyUI KSAMPLER signature)
# ---------------------------------------------------------------------------

# ``er_sde_window`` default: take the window from the run's settings (an explicit None = no window).
FROM_SETTINGS = object()
_window_logged: set[str] = set()


def _resolved(extra_args: Any, max_stage: Any, er_sde_eta: Any) -> tuple[Settings, int, float]:
    settings = settings_from_extra_args(extra_args)
    stage = settings.max_stage if max_stage is None else coerce_max_stage(max_stage)
    eta = settings.eta if er_sde_eta is None else coerce_eta(er_sde_eta)
    return settings, stage, eta


def _settings_window(name: str, model: Any, settings: Settings, eta: float) -> tuple[float, float] | None:
    """The window the run's settings ask for, or None — also when the model sampling cannot place it
    (no ``percent_to_sigma``: logged once, the sampler then runs with noise on every step, as sam-extra
    drops the window there)."""
    window = settings_window(settings._replace(eta=eta))
    if window is None:
        return None
    if not callable(getattr(model_sampling(model), "percent_to_sigma", None)):
        if name not in _window_logged:
            _window_logged.add(name)
            LOGGER.warning(
                "AI Studio pack: %s: the ER SDE noise window is skipped - this model sampling has no "
                "percent_to_sigma; it runs with noise on every step.", name,
            )
        return None
    return window


def _sde_sampler(name: str, solver_type: str):
    def sampler(model, x, sigmas, extra_args=None, callback=None, disable=None, s_noise=1.0,
                noise_sampler=None, max_stage=None, er_sde_eta=None, er_sde_window=FROM_SETTINGS, kd=None):
        settings, stage, eta = _resolved(extra_args, max_stage, er_sde_eta)
        if er_sde_window is FROM_SETTINGS:
            window = _settings_window(name, model, settings, eta)
        else:
            window = er_sde_window
        with _no_grad():
            return run_er_sde(
                model, x, sigmas, extra_args=extra_args, callback=callback, disable=disable,
                noise_sampler=noise_sampler, solver=er_sde_kwargs(solver_type, stage, eta, s_noise),
                window=window, kd=kd,
            )

    sampler.__name__ = sampler.__qualname__ = "sample_" + name
    return sampler


sample_er_sde_reverse_time = _sde_sampler("er_sde_reverse_time", REVERSE_TIME)
sample_er_sde_reverse_time.__doc__ = (
    "ER-SDE with ``h(λ) = λ^(η+1)``; η = 0 is the ODE. ``max_stage``/``er_sde_eta`` None and "
    "``er_sde_window`` unset → the run's settings (``model_options[MODEL_OPTIONS_KEY]``); "
    "``er_sde_window`` ``(start, end)`` or None (no window). ``er_sde_eta`` is not named ``eta`` so "
    "no host passes an ancestral η into it."
)
sample_er_sde_tunable = _sde_sampler("er_sde_tunable", ER_SDE)
sample_er_sde_tunable.__doc__ = (
    "ER-SDE with ``h(λ) = λ·(e^(λ^0.3) + 10)^η`` — ComfyUI's ``er_sde`` at η = 1 without a window; "
    "arguments as ``sample_er_sde_reverse_time``."
)


def sample_er_sde_ode(model, x, sigmas, extra_args=None, callback=None, disable=None, noise_sampler=None,
                      max_stage=None, kd=None):
    """ER-SDE with ``h(λ) = λ`` and ``s_noise = 0`` — deterministic, never draws noise."""
    _settings, stage, _eta = _resolved(extra_args, max_stage, None)
    with _no_grad():
        return run_er_sde(
            model, x, sigmas, extra_args=extra_args, callback=callback, disable=disable,
            noise_sampler=noise_sampler, solver=er_sde_kwargs(ODE, stage, 0.0, 0.0), window=None, kd=kd,
        )
