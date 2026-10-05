"""ComfyUI glue for the pack's Anima SPEED (pack 1.7.0): settings, runtime guard and sampling host.

``speed_runner.run_speed`` is host-neutral; this module is what ComfyUI needs around it:

* **Settings** (:func:`coerce_settings`): the node's explicit inputs (or a dict/positional list
  under sam-extra's script argument names and infotext aliases) become a :class:`SpeedSettings`
  with the same rules, defaults and ``error`` texts as sam-extra's ``Anima SPEED`` script, so a
  ComfyUI pass refuses exactly what a Forge pass refuses. :func:`plan_for` turns settings, a
  latent and a sigma list into the run's :class:`speed_schedule.Plan`.
* **Runtime guard** (:func:`comfy_runtime_guard`): why SPEED must not run on one sampling call,
  checked when the sampler starts, worded like sam-extra's status lines. ComfyUI equivalents of
  the extension's checks: no 1-D sigma list; a latent that is not 4-D/5-D; a ``model_sampling``
  that is not ``CONST`` (flow matching); sigma_max > 1; a ``denoise_mask`` (inpainting); processed
  conds carrying ``ref_latents`` (Anima reference, Flux Kontext, Qwen edit …) or ``c_concat``
  (image-to-video / fill conditioning) or a ``control`` entry (ControlNet); the pack's
  ControlNet-LLLite marker; ComfyUI Spectrum. ComfyUI-only: area/mask conds and several latents
  sampled together (``latent_shapes``) are tied to the full grid as well.
* **Sampling host** (:class:`ComfySpeedHost`, a ``speed_runner.SpeedHost``): the spectral seeds
  (the fixed SPEED seed, else the generation's per-image seeds ``image_seed + i`` like Forge's
  ``p.seeds``); sampler noise that stays **one stream across the segments** — a sampler that
  takes a ``noise_sampler`` gets one run-wide generator made the way ComfyUI's
  ``default_noise_sampler`` makes its own (seeded from ``extra_args["seed"]``, ``+1`` on the
  CPU), each segment drawing its own latent shape from it; Brownian-tree samplers get, per coarse
  stage, a tree seeded per image at ``seed + (stage + 1) * 1_000_003`` (sam-extra's Forge rule)
  and, for the full-size tail, the tree a plain ComfyUI run builds (``extra_args["seed"]``, the
  pass's sigma range); the patched schedule is published as
  ``transformer_options["sample_sigmas"]`` (ComfyUI's own key, which the pack's schedule
  consumers read) tagged with ``guidance_common.mark_republished`` and put back afterwards.
* **Sampler wrapper and node**: ``ForgeNeoAnimaSpeed`` (MODEL -> MODEL, explicit inputs) registers a
  ``SAMPLER_SAMPLE`` wrapper (:func:`speed_sampler_sample_wrapper`) moved to the front of the
  MODEL's wrappers, behind a CNS wrapper already there (the guidance suite's; the pack's sampler
  nodes re-assert their own CNS outermost), so the runtime order is CNS -> SPEED -> Detail
  Daemon -> sampler on every sampler node, a stock ``KSamplerAdvanced`` included: Detail Daemon
  builds its schedule from the whole pass and
  SPEED segments inside it (``guidance_dd`` follows the re-published list). Per run the wrapper
  swaps the KSAMPLER's ``sampler_function`` for :func:`speed_sampler_function` (UniPC / DPM fast /
  adaptive and non-KSAMPLER samplers are skipped), which guards, plans and runs SPEED, falls back
  to plain sampling when SPEED itself fails and lets an interrupt through, with sam-extra's
  status wording on the console (``[AnimaSPEED] <pass>: …``). A Hires checkpoint override keeps
  the patch (``hires_carry``).

The tree range: ComfyUI's ``sample_sigmas`` is the pass's own list, so an img2img/hires pass
starts below 1 and an aligned hand-off sigma (``sigma * kappa``) can lie above the list's maximum
(Forge's ``sampling_sigmas`` is the whole schedule and starts near 1). The trees span up to the
largest aligned sigma then, so no query falls outside a tree; with every aligned sigma inside the
pass range (txt2img, sigma0 = 1) the tail tree is exactly a plain run's.

The behaviour is sam-extra's (``forge_sam3_extension`` ``sam3ext/speed/forge_host.py``,
v0.33.0) where ComfyUI allows it, and so is much of the code (sam-extra's changes to the MIT upstream,
licensed to this pack under MIT by its author (THIRD_PARTY_NOTICES.md)) on the MIT upstream
aoleg/ComfyUI-SPEED@a8873591a27f2c1e086a2caf546f9b6aeec62b81 (``scripts/speed_forge.py``: the
shape-safe coarse noise, the inpaint/reference guards, wrapping the sampler function) —
Copyright (c) 2026 A. Izzuddin Al Faruq (the Forge Neo script is Oleg Afonin's work); notices in
THIRD_PARTY_NOTICES.md, texts in LICENSES/. ``tests/test_comfy_speed_parity.py`` checks the
settings and the wording against the installed extension; ``tests/test_comfy_speed_host.py``
checks the ComfyUI rules. torch and ComfyUI are imported when a run starts, never at import.
"""

from __future__ import annotations

import logging
import math
import traceback
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Callable

from . import hires_carry, speed_schedule
from .anima_lllite import LLLITE_MODEL_OPTION
from .compat import clone_model, make_first, require_torch, sampler_sample_wrapper_type
from .guidance_cns import CNS_WRAPPER_KEY
from .guidance_common import (
    DY_SUBSTEPS_OFF_KEY,
    SAMPLER_BASE_ATTR,
    declares_keyword,
    mark_republished,
    sampler_base,
)
from .speed_runner import SpeedHost, SpeedRun, run_speed
from .speed_schedule import Plan, PlanError

__all__ = [
    "ARG_ALIASES",
    "ARG_NAMES",
    "BROWNIAN_SAMPLERS",
    "BROWNIAN_STAGE_SEED_STRIDE",
    "CARRY_KEY",
    "DEFAULTS",
    "LLLITE_MARKER",
    "LOG_PREFIX",
    "NODE_CLASS_MAPPINGS",
    "NODE_DISPLAY_NAME_MAPPINGS",
    "PASS_LABELS",
    "SPEED_WRAPPER_KEY",
    "UNSUPPORTED_SAMPLERS",
    "ComfySpeedHost",
    "ForgeNeoAnimaSpeed",
    "SpeedPatch",
    "SpeedSettings",
    "apply_speed",
    "coerce_settings",
    "comfy_runtime_guard",
    "log_line",
    "model_speed_patch",
    "plan_for",
    "speed_sampler_function",
    "speed_sampler_sample_wrapper",
]

LOGGER = logging.getLogger("ai_studio_forge_parity")

# sam-extra's console prefix. The pack writes no infotext: the app writes "Anima SPEED: <summary>" into
# ComfyUI PNGs itself (core/comfy_metadata, key core/anima_speed.INFOTEXT_KEY).
LOG_PREFIX = "[AnimaSPEED]"
# ModelPatcher.add_wrapper_with_key key of the node's SAMPLER_SAMPLE wrapper (one per MODEL).
SPEED_WRAPPER_KEY = "ai_studio_forge_parity.speed"
# model_options key the pack's ControlNet-LLLite patch sets (anima_lllite.apply_lllite): its hint
# is tied to the full latent grid, like sam-extra's ``ACTIVE_LLLITE_DIT`` check.
LLLITE_MARKER = LLLITE_MODEL_OPTION
# hires_carry key: a Hires checkpoint override keeps the pass's SPEED wrapper.
CARRY_KEY = "speed"
PASS_LABELS = ("base", "img2img", "hires")
# Coarse Brownian trees are seeded away from the full-grid tree (sam-extra's Forge rule).
BROWNIAN_STAGE_SEED_STRIDE = 1_000_003

# Script argument names (sam-extra's positional order) and the named aliases (infotext keys).
ARG_NAMES = (
    "enabled", "mode", "threshold", "preset", "scales", "delta", "sigma_divisor",
    "manual_sigmas", "adaptive", "transform", "spectrum_A", "spectrum_beta", "seed", "hires",
)
DEFAULTS: dict[str, Any] = {
    "enabled": False,
    "mode": "transition",
    "threshold": "neo_shift",
    "preset": "anima",
    "scales": "0.5,1.0",
    "delta": 0.01,
    "sigma_divisor": 1.03,
    "manual_sigmas": "0.7",
    "adaptive": True,
    "transform": "dct",
    "spectrum_A": speed_schedule.CUSTOM_DEFAULT_A,
    "spectrum_beta": speed_schedule.CUSTOM_DEFAULT_BETA,
    "seed": -1,
    "hires": False,
}
ARG_ALIASES = {"divisor": "sigma_divisor", "manual": "manual_sigmas", "A": "spectrum_A", "beta": "spectrum_beta"}

# Sampler functions SPEED cannot segment, by the ``__name__`` of the real function behind the
# run's sampler (``guidance_common.sampler_base`` sees through CNS). The first five are
# sam-extra's, in its order (Forge names; v0.33.0 added ``sample_unipc_bh2`` for its UniPC bh2
# entry, which is also the function behind ComfyUI's ``uni_pc_bh2``); ComfyUI also wraps DPM
# fast/adaptive in ``comfy.samplers.ksampler``'s ``dpm_fast_function``/``dpm_adaptive_function``.
UNSUPPORTED_SAMPLERS = {
    "restart_sampler": "Restart re-plans its own schedule",
    "sample_unipc": "UniPC is not segment-safe",
    "sample_unipc_bh2": "UniPC is not segment-safe",
    "sample_dpm_fast": "DPM fast has no sigma schedule",
    "sample_dpm_adaptive": "DPM adaptive has no sigma schedule",
    "dpm_fast_function": "DPM fast has no sigma schedule",
    "dpm_adaptive_function": "DPM adaptive has no sigma schedule",
}
# Samplers whose own default noise is a ``BrownianTreeNoiseSampler`` -> its ``cpu`` flag
# (ComfyUI 0.36.0 ``comfy/k_diffusion/sampling.py``; ``tests/test_comfy_speed_host.py`` pins the
# table to that file). ``sample_dpmpp_4m_sde`` is the pack's Extra Samplers port, which follows
# ComfyUI's 3M SDE convention (``cpu=True``).
BROWNIAN_SAMPLERS = {
    "sample_dpmpp_sde": True,
    "sample_dpmpp_2m_sde": True,
    "sample_dpmpp_2m_sde_heun": True,
    "sample_dpmpp_3m_sde": True,
    "sample_dpmpp_sde_gpu": False,
    "sample_dpmpp_2m_sde_gpu": False,
    "sample_dpmpp_2m_sde_heun_gpu": False,
    "sample_dpmpp_3m_sde_gpu": False,
    "sample_dpmpp_4m_sde": True,
}


def log_line(message: str) -> None:
    """One ``[AnimaSPEED]`` console line (sam-extra's prefix); never raises."""
    try:
        LOGGER.info("%s %s", LOG_PREFIX, message)
    except Exception:  # pragma: no cover - a broken handler must not fail the sampling run
        pass


# ------------------------------------------------------------------------------------------------
# Settings
# ------------------------------------------------------------------------------------------------

_TRUE_WORDS = frozenset({"true", "1", "yes", "on", "enable", "enabled"})
_FALSE_WORDS = frozenset({"false", "0", "no", "off", "disable", "disabled", ""})


def _flag(value: Any, default: bool) -> bool:
    """A boolean input: real booleans and numbers as they are, known words, else ``default``."""
    if isinstance(value, bool):
        return value
    if value is None:
        return default
    if isinstance(value, (int, float)):
        return bool(value)
    word = str(value).strip().lower()
    if word in _TRUE_WORDS:
        return True
    if word in _FALSE_WORDS:
        return False
    return default


def _number(value: Any, default: float) -> float:
    """A float input; anything ``float()`` refuses (also an integer too large for a float, as in
    sam-extra 0.33.0) or a non-finite value gives ``default``."""
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        return default
    if not math.isfinite(number):
        return default
    return number


def _pick(value: Any, choices: tuple[str, ...], default: str, name: str, errors: list[str]) -> str:
    word = str(default if value is None else value).strip().lower()
    if word in choices:
        return word
    errors.append(f"{name} must be one of {', '.join(choices)} (got {value!r})")
    return default


def _fmt(value: float) -> str:
    return format(float(value), ".10g")


@dataclass(frozen=True)
class SpeedSettings:
    enabled: bool = False
    mode: str = "transition"
    threshold: str = "neo_shift"
    preset: str = "anima"
    scales: tuple = (0.5, 1.0)
    delta: float = 0.01
    sigma_divisor: float = 1.03
    manual_sigmas: tuple = (0.7,)
    adaptive: bool = True
    transform: str = "dct"
    spectrum_A: float = speed_schedule.CUSTOM_DEFAULT_A
    spectrum_beta: float = speed_schedule.CUSTOM_DEFAULT_BETA
    seed: int = -1
    hires: bool = False
    error: str | None = None

    def summary(self) -> str:
        """sam-extra's ``Anima SPEED`` value: ``key=value`` pairs joined by ``; ``."""
        pairs = [
            ("mode", self.mode),
            ("threshold", self.threshold),
            ("preset", self.preset),
            ("scales", ",".join(_fmt(value) for value in self.scales)),
            ("delta", _fmt(self.delta)),
            ("divisor", _fmt(self.sigma_divisor)),
            ("manual", ",".join(_fmt(value) for value in self.manual_sigmas)),
            ("adaptive", str(self.adaptive)),
            ("transform", self.transform),
        ]
        if self.preset == "custom":
            pairs.extend((("A", _fmt(self.spectrum_A)), ("beta", _fmt(self.spectrum_beta))))
        pairs.extend((("seed", str(self.seed)), ("hires", str(self.hires))))
        return "; ".join(f"{key}={value}" for key, value in pairs)


def _argument_reader(values: Any) -> Callable[[str], Any]:
    """``name -> value`` over a mapping (names or aliases) or a positional list (missing = default).

    A list whose first item is a mapping is read as that mapping (sam-extra's one-dict API form).
    """
    if not isinstance(values, Mapping):
        items = list(values or ())
        if items and isinstance(items[0], Mapping):
            values = items[0]
        else:
            def positional(name: str) -> Any:
                index = ARG_NAMES.index(name)
                return items[index] if index < len(items) else DEFAULTS[name]

            return positional
    named = {ARG_ALIASES.get(str(key), str(key)): value for key, value in values.items()}
    return lambda name: named.get(name, DEFAULTS[name])


def coerce_settings(values: Any = None) -> SpeedSettings:
    """Node inputs / script arguments -> :class:`SpeedSettings` (``error`` set when invalid).

    The checks and their messages are sam-extra's (v0.33.0), in its order: mode, threshold, preset,
    transform, scales, manual sigmas (only an error under the manual threshold), then delta and
    the neo_shift divisor (not checked under the manual threshold). A zero denominator in the
    scales/manual text (``1/0``) is a ``division by zero`` settings error; unparseable numbers
    (also integers too large for a float) fall back to the defaults; a negative seed is -1 (the
    image seeds are used). Nothing raises: every bad value is an ``error`` or a default.
    """
    arg = _argument_reader(values)
    errors: list[str] = []
    enabled = _flag(arg("enabled"), False)
    mode = _pick(arg("mode"), speed_schedule.MODES, "transition", "mode", errors)
    threshold = _pick(arg("threshold"), speed_schedule.THRESHOLDS, "neo_shift", "threshold", errors)
    preset = str(arg("preset") or DEFAULTS["preset"]).strip()
    if preset not in speed_schedule.PRESETS:
        errors.append(f"unknown spectrum preset {preset!r}")
        preset = DEFAULTS["preset"]
    transform = _pick(arg("transform"), speed_schedule.TRANSFORMS, "dct", "transform", errors)
    delta = _number(arg("delta"), DEFAULTS["delta"])
    divisor = _number(arg("sigma_divisor"), DEFAULTS["sigma_divisor"])
    try:
        scales = speed_schedule.parse_scales(arg("scales"))
    except ValueError as exc:
        errors.append(str(exc))
        scales = (0.5, 1.0)
    try:
        manual = speed_schedule.parse_sigmas(arg("manual_sigmas"), strictly_decreasing=mode == "transition")
    except ValueError as exc:
        manual = (0.7,)
        if threshold == "manual":
            errors.append(str(exc))
    if threshold != "manual":
        if not 0.0 < delta < 1.0:
            errors.append(f"delta must be in (0, 1); got {delta}")
        if threshold == "neo_shift" and not divisor > 0.0:
            errors.append(f"sigma divisor must be positive; got {divisor}")
    seed_value = _number(arg("seed"), -1.0)
    return SpeedSettings(
        enabled=enabled, mode=mode, threshold=threshold, preset=preset, scales=tuple(scales),
        delta=delta, sigma_divisor=divisor, manual_sigmas=tuple(manual),
        adaptive=_flag(arg("adaptive"), True), transform=transform,
        spectrum_A=_number(arg("spectrum_A"), DEFAULTS["spectrum_A"]),
        spectrum_beta=_number(arg("spectrum_beta"), DEFAULTS["spectrum_beta"]),
        seed=int(seed_value) if seed_value >= 0 else -1,
        hires=_flag(arg("hires"), False),
        error="; ".join(errors) or None,
    )


def plan_for(settings: SpeedSettings, x, sigmas) -> Plan:
    """The pass plan for ``settings`` on latent ``x`` (its last two axes are the grid)."""
    A, beta, ref_latent, ref_shift = speed_schedule.preset_params(
        settings.preset, settings.spectrum_A, settings.spectrum_beta,
    )
    return speed_schedule.build_plan(
        mode=settings.mode, transform=settings.transform, sigmas=sigmas,
        full_grid=(int(x.shape[-2]), int(x.shape[-1])), scales=settings.scales,
        threshold=settings.threshold, delta=settings.delta, A=A, beta=beta, ref_latent=ref_latent,
        ref_shift=ref_shift, adaptive=settings.adaptive, sigma_divisor=settings.sigma_divisor,
        manual_sigmas=settings.manual_sigmas,
    )


# ------------------------------------------------------------------------------------------------
# Runtime guard
# ------------------------------------------------------------------------------------------------

# Reasons, worded like sam-extra's status lines (tests/test_comfy_speed_parity.py compares them).
_FULL_GRID = "tied to the full latent grid"
NO_SCHEDULE_REASON = "the sampler gives no sigma schedule"
MASK_REASON = f"inpainting / mask (masks and inpaint conditioning are {_FULL_GRID})"
ANIMA_REFERENCE_REASON = (
    "Anima reference latents are present (Anima Edit / img2img referencing) - they are "
    "concatenated onto the latent at full size; sample without the reference to use SPEED"
)
REFERENCE_REASON = (
    "reference latents are present (Flux Kontext / Flux.2 Klein / Qwen-Image-Edit / Krea 2 edit) - "
    "the DiT appends them to the image tokens at full size, so on the coarse grid image and "
    "reference positions would not line up"
)
CONCAT_REASON = (
    "concat conditioning (c_concat: Wan 2.2 image-to-video, fill / inpaint models) is on - the DiT "
    "concatenates it with the latent on the channel axis at full size"
)
CONTROLNET_REASON = f"ControlNet is on (its hint is {_FULL_GRID})"
LLLITE_REASON = f"ControlNet-LLLite is on (its hint is {_FULL_GRID})"
SPECTRUM_REASON = (
    "ComfyUI Spectrum (DiTSpectrumPatch) is on - SPEED steps aside (its stop-caching point comes "
    "from the step count and it can forecast a coarse-sized tensor across the size change)"
)
AREA_REASON = f"area / mask conditioning is on (it is {_FULL_GRID})"
LATENT_SHAPES_REASON = f"several latents are sampled together (latent_shapes) - they are {_FULL_GRID}"
SPECTRUM_FLAG = "Spectrum"   # the DY_SUBSTEPS_OFF_KEY value ForgeNeoKSamplerCNS sets for Spectrum

# ComfyUI's prediction classes (comfy.model_sampling), most specific first.
_PREDICTIONS = ("CONST", "COSMOS_RFLOW", "EDM", "V_PREDICTION_DDPM", "V_PREDICTION", "IMG_TO_IMG", "X0", "EPS")


def prediction_kind(model_sampling: Any) -> str | None:
    """``"const"`` for a flow-matching ``model_sampling`` (``CONST`` anywhere in its classes —
    a private copy of the same class counts too), else the lower-cased prediction class name."""
    if model_sampling is None:
        return None
    names = [cls.__name__ for cls in type(model_sampling).__mro__]
    for known in _PREDICTIONS:
        if known in names:
            return known.lower()
    return type(model_sampling).__name__.lower()


def _model_sampling(guider: Any) -> Any:
    patcher = getattr(guider, "model_patcher", None)
    getter = getattr(patcher, "get_model_object", None)
    if callable(getter):
        try:
            return getter("model_sampling")
        except Exception:
            pass
    return getattr(getattr(guider, "inner_model", None), "model_sampling", None)


def _base_model(guider: Any) -> Any:
    inner = getattr(guider, "inner_model", None)
    if inner is None:
        inner = getattr(getattr(guider, "model_patcher", None), "model", None)
    return inner


def _is_anima(guider: Any) -> bool:
    model = _base_model(guider)
    return model is not None and any("Anima" in cls.__name__ for cls in type(model).__mro__)


def _processed_conds(guider: Any) -> list[dict]:
    """Every cond dict of the run (``guider.conds`` after ``process_conds``)."""
    conds = getattr(guider, "conds", None)
    found: list[dict] = []
    if isinstance(conds, Mapping):
        for value in conds.values():
            if isinstance(value, (list, tuple)):
                found.extend(item for item in value if isinstance(item, Mapping))
    return found


def _model_conds(cond: Mapping) -> Mapping:
    value = cond.get("model_conds")
    return value if isinstance(value, Mapping) else {}


def _options(extra_args: Any) -> tuple[Mapping, Mapping]:
    options = extra_args.get("model_options") if isinstance(extra_args, Mapping) else None
    options = options if isinstance(options, Mapping) else {}
    transformer = options.get("transformer_options")
    return options, transformer if isinstance(transformer, Mapping) else {}


def _spectrum_function(options: Mapping) -> bool:
    """A model-function wrapper from a Spectrum provider (its own name or module says so)."""
    function = options.get("model_function_wrapper")
    if function is None:
        return False
    name = getattr(function, "__name__", None) or type(function).__name__
    module = getattr(function, "__module__", None) or type(function).__module__
    return "spectrum" in f"{module}.{name}".lower()


def comfy_runtime_guard(guider: Any, x: Any, sigmas: Any, extra_args: Any) -> str | None:
    """Why SPEED must not run on this sampling call, or None. Checked when the sampler starts.

    ``guider`` is the run's ``CFGGuider`` (the SAMPLER_SAMPLE wrapper's first argument): its
    ``model_patcher``/``inner_model`` give the model sampling and the model class, its ``conds``
    the processed conditioning.
    """
    torch = require_torch()
    if not torch.is_tensor(sigmas) or sigmas.ndim != 1 or len(sigmas) < 2:
        return NO_SCHEDULE_REASON
    if not torch.is_tensor(x) or x.ndim not in (4, 5):
        return f"unsupported latent shape {tuple(getattr(x, 'shape', ()))}"
    kind = prediction_kind(_model_sampling(guider))
    if kind != "const":
        return (
            f"not a flow-matching model (prediction_type={kind}) - SDXL/SD1.x eps/v models are not "
            "supported (kappa and the sigma* presets assume flow sigmas in (0, 1])"
        )
    if float(sigmas.max()) > 1.0 + 1e-4:
        return f"sigma_max {float(sigmas.max()):.3f} > 1 is not a flow schedule"
    if isinstance(extra_args, Mapping) and extra_args.get("denoise_mask") is not None:
        return MASK_REASON
    conds = _processed_conds(guider)
    if any("ref_latents" in _model_conds(cond) for cond in conds):
        return ANIMA_REFERENCE_REASON if _is_anima(guider) else REFERENCE_REASON
    if any("c_concat" in _model_conds(cond) for cond in conds):
        return CONCAT_REASON
    if any(cond.get("control") is not None for cond in conds):
        return CONTROLNET_REASON
    options, transformer = _options(extra_args)
    if options.get(LLLITE_MARKER):
        return LLLITE_REASON
    flag = transformer.get(DY_SUBSTEPS_OFF_KEY)
    if flag == SPECTRUM_FLAG or _spectrum_function(options):
        return SPECTRUM_REASON
    if flag:
        return f"{flag} is on - SPEED steps aside (that sampler already changes how this pass samples)"
    if any(cond.get("area") is not None or cond.get("mask") is not None for cond in conds):
        return AREA_REASON
    shapes = getattr(getattr(guider, "inner_model", None), "latent_shapes", None)
    if isinstance(shapes, (list, tuple)) and len(shapes) > 1:
        return LATENT_SHAPES_REASON
    return None


def unsupported_sampler_reason(sampler: Any) -> str | None:
    """Why the run's sampler cannot be segmented (checked before the run is wrapped), or None."""
    function = getattr(sampler, "sampler_function", None)
    if not callable(function):
        return f"{type(sampler).__name__} is not a k-diffusion sampler"
    return UNSUPPORTED_SAMPLERS.get(getattr(sampler_base(function), "__name__", ""))


# ------------------------------------------------------------------------------------------------
# Sampling host
# ------------------------------------------------------------------------------------------------


class _NoiseStream:
    """One generator for a whole SPEED run, made like ComfyUI's ``default_noise_sampler``.

    ``default_noise_sampler(x, seed)`` seeds a ``torch.Generator`` on ``x``'s device with ``seed``
    (``seed + 1`` on the CPU) and returns a noise sampler that draws ``x``'s shape. Here every
    segment gets a noise sampler for its own latent from the same generator, so the draws go on
    across the size changes instead of restarting at every segment (Forge's per-image stream).
    """

    def __init__(self, x: Any, seed: Any):
        torch = require_torch()
        self.generator = None
        if seed is not None:
            value = int(seed)
            if x.device == torch.device("cpu"):
                value += 1
            self.generator = torch.Generator(device=x.device)
            self.generator.manual_seed(value)

    def sampler_for(self, x: Any) -> Callable[[Any, Any], Any]:
        torch = require_torch()
        size, dtype, layout, device = x.size(), x.dtype, x.layout, x.device
        generator = self.generator

        def noise_sampler(sigma, sigma_next):
            return torch.randn(size, dtype=dtype, layout=layout, device=device, generator=generator)

        noise_sampler._ai_studio_speed_stream = self
        return noise_sampler


def _is_brownian(noise_sampler: Any) -> bool:
    return noise_sampler is not None and type(noise_sampler).__name__ == "BrownianTreeNoiseSampler"


def _kd_module():
    import comfy.k_diffusion.sampling as kd_sampling  # lazy: only inside ComfyUI

    return kd_sampling


class ComfySpeedHost(SpeedHost):
    """``SpeedHost`` for one ComfyUI sampling call (the sampler function of one KSAMPLER run).

    ``sampler_function`` is the function SPEED segments (the run's ``sampler_function``, CNS or
    not); ``sigmas``/``extra_args`` are the ones the sampler was called with; ``settings_seed`` is
    the node's SPEED seed (-1 = the image seeds) and ``image_seed`` the generation seed (image ``i``
    of the batch has ``image_seed + i``). ``plan`` (optional) lets the Brownian trees cover the
    aligned hand-off sigmas. ``kd_module`` replaces ``comfy.k_diffusion.sampling`` in tests.
    """

    def __init__(self, *, sampler_function: Callable, sigmas: Any, extra_args: Any = None,
                 settings_seed: int = -1, image_seed: int = 0, plan: Plan | None = None,
                 kd_module: Any = None, log: Callable[[str], None] | None = None):
        torch = require_torch()
        self.extra_args = extra_args if isinstance(extra_args, dict) else {}
        self.settings_seed = int(settings_seed)
        self.image_seed = int(image_seed)
        self._kd = kd_module
        self._log = log or (lambda message: None)
        base = sampler_base(sampler_function)
        self.sampler_name = str(getattr(base, "__name__", "") or "")
        self.brownian_cpu = BROWNIAN_SAMPLERS.get(self.sampler_name)
        self.takes_noise_sampler = _declares_noise_sampler(sampler_function)
        self._stream: _NoiseStream | None = None
        self._options = None
        self._original = None
        self._offset = 0
        self._locate(sigmas)
        whole = self._original if self._original is not None else sigmas
        self._tree_range = _tree_range(torch, whole, plan)

    # -- where the pass's schedule is published ---------------------------------------------------
    def _locate(self, sigmas) -> None:
        """Find the run's ``sample_sigmas`` list: ``sigmas`` itself or a list ending with it."""
        torch = require_torch()
        _, transformer = _options(self.extra_args)
        whole = transformer.get("sample_sigmas")
        if not (torch.is_tensor(whole) and torch.is_tensor(sigmas)) or whole.ndim != 1 or len(whole) < len(sigmas):
            return
        offset = len(whole) - len(sigmas)
        if whole is not sigmas:
            try:
                same = torch.allclose(whole[offset:].detach().float().cpu(), sigmas.detach().float().cpu(),
                                      rtol=0.0, atol=1e-7)
            except Exception:
                same = False
            if not same:
                return
        self._options, self._original, self._offset = transformer, whole, offset

    def publish_sigmas(self, sigmas) -> None:
        if self._options is None:
            return
        published = self._original.clone()
        published[self._offset:] = sigmas.to(device=published.device, dtype=published.dtype)
        self._options["sample_sigmas"] = mark_republished(published, self._original)

    def restore_sigmas(self) -> None:
        if self._options is not None:
            self._options["sample_sigmas"] = self._original

    # -- seeds -------------------------------------------------------------------------------------
    def batch_image_seeds(self, batch: int) -> list[int]:
        return [self.image_seed + index for index in range(int(batch))]

    def seeds(self, batch: int) -> list[int]:
        """Seeds of the spectral (expansion / init) noise: the fixed SPEED seed, else the image seeds."""
        if self.settings_seed >= 0:
            return [self.settings_seed] * int(batch)
        return self.batch_image_seeds(batch)

    # -- sampler noise -----------------------------------------------------------------------------
    def _stream_for(self, x):
        if self._stream is None:
            self._stream = _NoiseStream(x, self.extra_args.get("seed"))
        return self._stream.sampler_for(x)

    def _brownian(self, x, seed, cpu: bool, template: Any = None):
        if self._tree_range is None:
            return None
        cls = type(template) if _is_brownian(template) else None
        if cls is None:
            cls = getattr(self._kd if self._kd is not None else _kd_module(), "BrownianTreeNoiseSampler")
        low, high = self._tree_range
        return cls(x, low, high, seed=seed, cpu=cpu)

    def sampler_kwargs(self, kwargs: Mapping, x) -> dict:
        """The sampler keywords of the run: a missing ``noise_sampler`` becomes the run's own.

        Only a sampler that names ``noise_sampler`` gets one (never through ``**kwargs``); a
        noise sampler the caller passed is kept and stands for the full-size tail.
        """
        out = dict(kwargs)
        if out.get("noise_sampler") is not None:
            return out
        out.pop("noise_sampler", None)
        if not self.takes_noise_sampler:
            return out
        if self.brownian_cpu is not None:
            tail = self._brownian(x, self.extra_args.get("seed"), self.brownian_cpu)
        else:
            tail = self._stream_for(x)
        if tail is not None:
            out["noise_sampler"] = tail
        return out

    def coarse_noise_sampler(self, template, x, stage, sigmas):
        if self.brownian_cpu is not None or _is_brownian(template):
            cpu = self.brownian_cpu
            if cpu is None:
                cpu = bool(getattr(getattr(template, "tree", None), "cpu_tree", True))
            seeds = [seed + (stage + 1) * BROWNIAN_STAGE_SEED_STRIDE
                     for seed in self.batch_image_seeds(int(x.shape[0]))]
            return self._brownian(x, seeds, cpu, template)
        return self._stream_for(x)

    def log(self, message: str) -> None:
        self._log(message)


def _declares_noise_sampler(function: Callable) -> bool:
    return declares_keyword(function, "noise_sampler")


def _tree_range(torch, whole, plan: Plan | None):
    """``(sigma_min, sigma_max)`` tensors a plain ComfyUI SDE sampler would use for ``whole``,
    with the top raised to the largest aligned hand-off sigma when one lies above it."""
    if not torch.is_tensor(whole) or whole.ndim != 1:
        return None
    positive = whole[whole > 0]
    if positive.numel() == 0:
        return None
    low, high = positive.min(), whole.max()
    if plan is not None and plan.transitions:
        top = float(high)
        for transition in plan.transitions:
            aligned = float(transition.aligned)
            if math.isfinite(aligned):
                # the value as a float32 working list (respace) or the list's own dtype stores it
                top = max(top, float(torch.tensor(aligned, dtype=torch.float32)), float(whole.new_tensor(aligned)))
        if top > float(high):
            high = whole.new_tensor(top)
    return low, high


# ------------------------------------------------------------------------------------------------
# The sampler wrapper (one sampling run)
# ------------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class SpeedPatch:
    """What ``ForgeNeoAnimaSpeed`` put on a MODEL: the settings and the pass it is for."""

    settings: SpeedSettings
    label: str = "base"                 # base / img2img / hires (sam-extra's pass labels)
    image_seed: int = 0                 # the generation seed; image i of a batch has image_seed + i
    img2img_rescale: bool = True        # sam-extra's sam3_speed_img2img_rescale option (default on)


class _RunStatus:
    """The last status of a wrapper (sam-extra's ``Anima SPEED status`` wording; console only)."""

    def __init__(self, label: str):
        self.label = label
        self.text: str | None = None

    def set(self, text: str) -> None:
        self.text = text

    def log(self, message: str) -> None:
        log_line(f"{self.label}: {message}")


def _is_interrupt(exc: BaseException) -> bool:
    # comfy.model_management.InterruptProcessingException (a BaseException): ComfyUI drops the run.
    return type(exc).__name__ == "InterruptProcessingException"


def _plan_line(settings: SpeedSettings, plan: Plan) -> str:
    """sam-extra's plan console line: each hand-off with the sigmas the run uses there."""
    hand_offs = ", ".join(
        f"step {t.step}: {t.scale_from:g}->{t.scale_to:g} {t.grid_from[0]}x{t.grid_from[1]}"
        f"->{t.grid_to[0]}x{t.grid_to[1]} (sigma {t.sigma:.4f}->{t.aligned:.4f})"
        for t in plan.transitions
    )
    return f"{settings.mode} plan - {hand_offs}; {plan.coarse_steps}/{plan.n_steps} steps coarse"


def speed_sampler_function(base_fn: Callable, guider: Any, patch: "SpeedPatch", *,
                           status: _RunStatus | None = None, kd_module: Any = None) -> Callable:
    """A KSAMPLER ``sampler_function`` that runs ``base_fn`` with SPEED for one sampling run.

    The first call is SPEED's (a later call through the same function passes through): the
    runtime guard, the plan, then ``run_speed`` with a :class:`ComfySpeedHost`. A guard reason or
    a :class:`PlanError` samples plainly with a ``skipped``/``invalid settings`` status; any other
    exception from SPEED logs the traceback and samples again without it (``fell back to plain
    sampling after …``); an interrupt goes through untouched. ``base_fn`` stays reachable through
    ``guidance_common.SAMPLER_BASE_ATTR`` for wrappers that read the real sampler's signature.
    """
    settings = patch.settings
    label = patch.label
    status = status or _RunStatus(label)
    used = {"done": False}

    def speed_sample(model, x, sigmas, extra_args=None, callback=None, disable=None, **kwargs):
        def plain():
            return base_fn(model, x, sigmas, extra_args=extra_args, callback=callback, disable=disable, **kwargs)

        if used["done"]:
            return plain()
        used["done"] = True
        run_guider = guider if guider is not None else getattr(model, "inner_model", None)
        reason = comfy_runtime_guard(run_guider, x, sigmas, extra_args)
        if reason is not None:
            status.set(f"skipped - {reason}")
            status.log(f"skipped - {reason}")
            return plain()
        try:
            plan = plan_for(settings, x, sigmas)
        except PlanError as exc:
            prefix = "invalid settings" if exc.kind == "invalid" else "skipped"
            status.set(f"{prefix} - {exc.reason}")
            status.log(f"{prefix} - {exc.reason}")
            return plain()
        host = ComfySpeedHost(
            sampler_function=base_fn, sigmas=sigmas, extra_args=extra_args, settings_seed=settings.seed,
            image_seed=patch.image_seed, plan=plan, kd_module=kd_module, log=status.log,
        )
        run = SpeedRun(plan=plan, seeds=host.seeds(int(x.shape[0])), init_latent=label in ("hires", "img2img"),
                       img2img_rescale=bool(patch.img2img_rescale))
        for note in plan.notes:
            status.log(note)
        status.log(_plan_line(settings, plan))
        try:
            out, report = run_speed(
                base_fn, model, x, sigmas, run=run, host=host, extra_args=extra_args, callback=callback,
                disable=disable, sampler_kwargs=host.sampler_kwargs(kwargs, x),
            )
        except BaseException as exc:
            if _is_interrupt(exc):
                status.set("interrupted")
                raise
            if not isinstance(exc, Exception):
                raise
            status.log(f"SPEED failed, sampling again without it:\n{traceback.format_exc()}")
            status.set(f"fell back to plain sampling after {type(exc).__name__}: {exc}")
            return plain()
        summary = report.summary(plan)
        status.set(f"applied ({settings.mode}) - {summary}")
        status.log(f"done - {summary}")
        return out

    setattr(speed_sample, SAMPLER_BASE_ATTR, base_fn)
    return speed_sample


def speed_sampler_sample_wrapper(patch: "SpeedPatch", *, kd_module: Any = None):
    """``WrappersMP.SAMPLER_SAMPLE`` wrapper: SPEED on the KSAMPLER of every sampling run.

    ``executor.class_obj`` is the run's sampler and the first argument its ``CFGGuider``. A sampler
    SPEED cannot segment (not a KSAMPLER, UniPC, DPM fast/adaptive) samples unchanged with a
    ``skipped`` status; otherwise its ``sampler_function`` is swapped for this run only, as CNS
    does. CNS, re-asserted outermost by the sampler nodes, runs first, so SPEED segments the CNS
    function and the coloured noise follows the segments.
    """
    status = _RunStatus(patch.label)

    def speed_sampler_sample(executor, *args, **kwargs):
        sampler = getattr(executor, "class_obj", None)
        reason = unsupported_sampler_reason(sampler)
        if reason is not None:
            status.set(f"skipped - {reason}")
            status.log(f"skipped - {reason}")
            return executor(*args, **kwargs)
        original = sampler.sampler_function
        guider = args[0] if args else kwargs.get("model_wrap")
        sampler.sampler_function = speed_sampler_function(original, guider, patch, status=status, kd_module=kd_module)
        try:
            return executor(*args, **kwargs)
        finally:
            sampler.sampler_function = original

    speed_sampler_sample._ai_studio_speed = patch
    speed_sampler_sample._ai_studio_speed_status = status
    return speed_sampler_sample


# ------------------------------------------------------------------------------------------------
# MODEL patch, Hires carry and the node
# ------------------------------------------------------------------------------------------------


def _speed_wrappers(model: Any) -> list:
    wrappers = getattr(model, "wrappers", None)
    if not isinstance(wrappers, dict):
        return []
    registered = wrappers.get(sampler_sample_wrapper_type(), {})
    found = registered.get(SPEED_WRAPPER_KEY) if isinstance(registered, dict) else None
    return list(found or [])


def model_speed_patch(model: Any) -> "SpeedPatch | None":
    """The :class:`SpeedPatch` ``apply_speed`` put on ``model`` (else None)."""
    for wrapper in reversed(_speed_wrappers(model)):
        patch = getattr(wrapper, "_ai_studio_speed", None)
        if isinstance(patch, SpeedPatch):
            return patch
    return None


def _install(patched: Any, wrappers: list) -> Any:
    add = getattr(patched, "add_wrapper_with_key", None)
    if not callable(add):
        raise RuntimeError("Anima SPEED needs a ComfyUI MODEL that supports sampler wrappers; update ComfyUI.")
    wrapper_type = sampler_sample_wrapper_type()
    remove = getattr(patched, "remove_wrappers_with_key", None)
    if callable(remove):
        remove(wrapper_type, SPEED_WRAPPER_KEY)
    for wrapper in wrappers:
        add(wrapper_type, SPEED_WRAPPER_KEY, wrapper)
    # Ahead of Detail Daemon: DD builds its schedule from the whole pass and SPEED segments inside
    # it. A CNS wrapper already on the MODEL (the guidance suite's, which a stock KSampler /
    # KSamplerAdvanced does not re-assert) stays ahead of SPEED: CNS -> SPEED -> DD on every
    # sampler; ForgeNeoKSamplerCNS / ForgeNeoHiresFix put their own CNS in front anyway.
    make_first(patched, wrapper_type, SPEED_WRAPPER_KEY)
    make_first(patched, wrapper_type, CNS_WRAPPER_KEY)
    return patched


def apply_speed(model: Any, patch: "SpeedPatch", *, kd_module: Any = None) -> Any:
    """Clone ``model`` with SPEED on its sampling runs (replaces an earlier SPEED patch)."""
    return _install(clone_model(model, "Anima SPEED"), [speed_sampler_sample_wrapper(patch, kd_module=kd_module)])


def _carry_to_override(source_model: Any, target_model: Any, target_vae: Any) -> Any:
    """``hires_carry``: a Hires checkpoint override samples with the pass's SPEED patch too."""
    wrappers = _speed_wrappers(source_model)
    if not wrappers:
        return None
    return _install(clone_model(target_model, "Anima SPEED"), wrappers)


def _float_input(default: float, low: float, high: float, step: float, tooltip: str):
    return ("FLOAT", {"default": default, "min": low, "max": high, "step": step, "round": False, "tooltip": tooltip})


class ForgeNeoAnimaSpeed:
    """sam-extra's "Anima SPEED" for ComfyUI: a MODEL patch whose sampling runs go coarse-to-fine.

    Every input is explicit (the app compiler sends them all; ComfyUI validates each). ``seed`` is
    SPEED's spectral seed (-1 = the image seeds); ``image_seed`` is the generation seed and
    ``pass_label`` the pass the node sits on (an img2img/hires start is flow-rescaled when
    ``img2img_rescale`` is on, sam-extra's default).
    """

    CATEGORY = "AI Studio/Forge Neo parity/Generation"
    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "patch"
    DESCRIPTION = (
        "Anima SPEED (sam-extra v0.33.0, Spectral Progressive Diffusion): the early, noise-dominated "
        "steps run on a DCT-truncated coarse latent, then the latent is expanded and the pass "
        "finishes at full size."
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "enabled": ("BOOLEAN", {"default": False}),
                "mode": (list(speed_schedule.MODES), {"default": DEFAULTS["mode"]}),
                "threshold": (list(speed_schedule.THRESHOLDS), {"default": DEFAULTS["threshold"]}),
                "preset": (list(speed_schedule.PRESET_NAMES), {"default": DEFAULTS["preset"]}),
                "scales": ("STRING", {"default": DEFAULTS["scales"],
                                      "tooltip": "Resolution ladder ending at 1.0, e.g. 0.5,1.0"}),
                "delta": _float_input(DEFAULTS["delta"], 0.0, 1.0, 0.0001,
                                      "Noise tolerance delta in (0, 1) (not used by the manual threshold)"),
                "sigma_divisor": _float_input(DEFAULTS["sigma_divisor"], 0.0, 100.0, 0.01,
                                              "neo_shift: the transition sigma is divided by this"),
                "manual_sigmas": ("STRING", {"default": DEFAULTS["manual_sigmas"],
                                             "tooltip": "Manual hand-off sigmas in (0, 1), one per coarse stage"}),
                "adaptive": ("BOOLEAN", {"default": DEFAULTS["adaptive"]}),
                "transform": (list(speed_schedule.TRANSFORMS), {"default": DEFAULTS["transform"]}),
                "spectrum_A": _float_input(DEFAULTS["spectrum_A"], -1e12, 1e12, 0.000001,
                                           "Power spectrum A of the custom preset (must be > 0)"),
                "spectrum_beta": _float_input(DEFAULTS["spectrum_beta"], -1e12, 1e12, 0.000001,
                                              "Power spectrum beta of the custom preset"),
                "seed": ("INT", {"default": DEFAULTS["seed"], "min": -1, "max": 0xFFFFFFFFFFFFFFFF,
                                 "tooltip": "Spectral noise seed (-1 = the image seeds)"}),
                "pass_label": (list(PASS_LABELS), {"default": "base"}),
                "image_seed": ("INT", {"default": 0, "min": 0, "max": 0xFFFFFFFFFFFFFFFF,
                                       "tooltip": "The generation seed (image i of a batch uses image_seed + i)"}),
                "img2img_rescale": ("BOOLEAN", {"default": True}),
            },
        }

    def patch(self, model, enabled=False, mode=DEFAULTS["mode"], threshold=DEFAULTS["threshold"],
              preset=DEFAULTS["preset"], scales=DEFAULTS["scales"], delta=DEFAULTS["delta"],
              sigma_divisor=DEFAULTS["sigma_divisor"], manual_sigmas=DEFAULTS["manual_sigmas"],
              adaptive=DEFAULTS["adaptive"], transform=DEFAULTS["transform"], spectrum_A=DEFAULTS["spectrum_A"],
              spectrum_beta=DEFAULTS["spectrum_beta"], seed=DEFAULTS["seed"], pass_label="base", image_seed=0,
              img2img_rescale=True):
        label = str(pass_label)
        if label not in PASS_LABELS:
            raise ValueError(f"Anima SPEED pass_label must be one of {', '.join(PASS_LABELS)}, not {pass_label!r}.")
        values = {
            "enabled": enabled, "mode": mode, "threshold": threshold, "preset": preset, "scales": scales,
            "delta": delta, "sigma_divisor": sigma_divisor, "manual_sigmas": manual_sigmas, "adaptive": adaptive,
            "transform": transform, "spectrum_A": spectrum_A, "spectrum_beta": spectrum_beta, "seed": seed,
        }
        if not _flag(enabled, False):
            return (model,)
        settings = coerce_settings(values)   # never raises: "1/0" is a settings error (sam-extra 0.33.0)
        if settings.error:
            log_line(f"{label}: invalid settings, not applied - {settings.error}")
            return (model,)
        patch = SpeedPatch(settings=settings, label=label, image_seed=int(image_seed),
                           img2img_rescale=bool(img2img_rescale))
        return (apply_speed(model, patch),)


NODE_CLASS_MAPPINGS = {"ForgeNeoAnimaSpeed": ForgeNeoAnimaSpeed}
NODE_DISPLAY_NAME_MAPPINGS = {"ForgeNeoAnimaSpeed": "Forge Neo Anima SPEED (sam-extra)"}

# A Hires checkpoint/UNET override keeps the pass's SPEED patch (hires_carry, ForgeNeoHiresFix).
hires_carry.register(_carry_to_override, key=CARRY_KEY)
