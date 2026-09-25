"""CNS (Colored Noise Sampling) — per-step coloured ancestral noise.

Behaviour of ``namemechan/comfyui-cns_sampler_patch@42278b13`` (GPL-3.0),
reimplemented here without its code (this pack is not GPL).  The numbers are
checked against the original by ``tests/test_comfy_cns_origin.py`` with
``tests/fixtures/cns_origin_golden.json`` (generated from the original).

What happens, as in the original:

* The sampler's ``noise_sampler(sigma, sigma_next)`` — the noise an ancestral
  or SDE sampler adds each step — is replaced.  Each draw takes the base noise
  (the ``noise_sampler`` the sampler was given, else ``torch.randn_like(x_t)``)
  and recolours it from the live latent ``x_t``: Haar subband energy share
  ``g = e / sum(e) / gamma_scale`` (clamped 0..1), deficit ``d = 1 - g``,
  weight ``beta = d ** gamma_power`` normalised to RMS 1, per-band multiply,
  std matched to the base noise, then ``lerp(noise, coloured, strength)`` when
  ``strength < 1`` — with no renormalisation after the lerp.
* ``x_t`` is the ``x`` of the sampler's per-step callback (the sampler's
  starting ``x`` until the first callback).
* The initial noise, the model and CFG are untouched.  ODE samplers never call
  ``noise_sampler``, so they are unchanged.  ``strength`` 0 returns the base
  noise as it is, but still draws it through CNS (so without a given
  ``noise_sampler`` it is ``randn_like``, not the sampler's seeded default).
* Path A: the sampler function is a ``comfy.k_diffusion.sampling.sample_*``
  function — the noise sampler and callback are passed to it directly.
  Path B: it is a wrapper (SPEED and the like) that calls ``sample_*`` by name —
  every ``sample_*`` that takes a ``noise_sampler`` is swapped for a colouring
  version while the wrapper runs.  As in the original, a swapped ``sample_*``
  that another ``sample_*`` calls by keyword (``*_gpu`` delegation, a second
  CNS patch stacked on the first) colours the noise again.

Deliberate differences from the original — only where the original raises:

* Path B: a ``sample_*`` that another ``sample_*`` calls with positional
  arguments after ``sigmas`` while the noise is already coloured (the
  flow-model ``*_RF`` dispatch) passes straight through, so the noise is
  coloured once.  The original raised ``TypeError`` there.
* Path B also accepts positional arguments after ``x`` on the first call
  (``dpm_fast`` / ``dpm_adaptive`` call their ``sample_*`` that way) and
  colours it; the original raised ``TypeError``.

Entry points: ``ForgeNeoCNSSamplerPatch`` (SAMPLER -> SAMPLER, the original's
node inputs) and ``apply_cns`` (a MODEL clone with an outermost
``SAMPLER_SAMPLE`` wrapper, so KSampler, Hires, Detail Daemon and user graphs
colour every step).  A sampling run whose sampler is not a KSAMPLER (no
``sampler_function``) raises, as the original node does.
"""

from __future__ import annotations

import functools
import inspect
import logging
import threading
from typing import Any, Callable

from .compat import clone_model, require_torch
from .guidance_common import CATEGORY, _haar_dwt, _haar_idwt, _pad_even


LOGGER = logging.getLogger("ai_studio_forge_parity")

# Inputs of the original node (default, range, step, round) —
# origin: namemechan/comfyui-cns_sampler_patch@42278b13:cns_sampler_patch.py:396-437
CNS_INPUTS: dict[str, dict[str, float]] = {
    "strength": {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.05, "round": 0.01},
    "gamma_power": {"default": 0.5, "min": 0.1, "max": 2.0, "step": 0.05, "round": 0.01},
    "gamma_scale": {"default": 2.0, "min": 0.1, "max": 25.0, "step": 0.1, "round": 0.01},
}
CNS_DEFAULTS: dict[str, float] = {name: spec["default"] for name, spec in CNS_INPUTS.items()}

# One CNS wrapper per MODEL: applying CNS again replaces the settings.
CNS_WRAPPER_KEY = "ai_studio_forge_parity.cns"
_SAMPLER_SAMPLE_FALLBACK = "sampler_sample"   # comfy.patcher_extension.WrappersMP.SAMPLER_SAMPLE
_SETTINGS_ATTR = "forge_neo_cns"
_ACTIVE = threading.local()   # >0 while a colouring noise sampler is installed


def cns_float_input(name: str, **tooltip: str) -> tuple[str, dict[str, Any]]:
    """Comfy ``FLOAT`` input spec with the original node's default and range."""

    return ("FLOAT", {**CNS_INPUTS[name], **tooltip})


def color_noise_wavelet(
    noise: Any,
    x_t: Any,
    strength: float = 1.0,
    gamma_power: float = 0.5,
    gamma_scale: float = 2.0,
):
    """Recolour one noise draw from the live latent (see the module docstring)."""

    if strength == 0.0:
        return noise
    torch = require_torch()
    out_dtype = noise.dtype
    white = noise.float()
    live, (height, width) = _pad_even(x_t.to(device=white.device).float())
    reduce_dims = tuple(range(1, live.ndim))
    energy = tuple(
        band.float().square().mean(dim=reduce_dims, keepdim=True).clamp_min(1e-8)
        for band in _haar_dwt(live)
    )
    total = sum(energy)
    weights = tuple(
        (1.0 - (share / total / gamma_scale).clamp(0.0, 1.0))
        .clamp_min(1e-8)
        .pow(gamma_power)
        for share in energy
    )
    rms = (sum(item.square() for item in weights) / 4.0).sqrt().clamp_min(1e-8)
    weights = tuple(item / rms for item in weights)
    padded, _ = _pad_even(white)
    colored = _haar_idwt(
        *(band * weight for band, weight in zip(_haar_dwt(padded), weights))
    )[..., :height, :width]
    colored = colored * (white.std().clamp_min(1e-8) / colored.std().clamp_min(1e-8))
    if strength < 1.0:
        colored = torch.lerp(white, colored, strength)
    return colored.to(dtype=out_dtype)


def make_cns_noise_sampler(
    x_initial: Any,
    strength: float,
    gamma_power: float,
    gamma_scale: float,
    base_noise_sampler: Callable[..., Any] | None = None,
):
    """Return ``(noise_sampler, state)``; keep ``state["x"]`` on the live x_t."""

    torch = require_torch()
    state = {"x": x_initial.detach().clone()}

    def cns_noise_sampler(sigma, sigma_next):
        live = state["x"]
        if base_noise_sampler is None:
            noise = torch.randn_like(live)
        else:
            try:
                noise = base_noise_sampler(sigma, sigma_next)
            except Exception:
                noise = torch.randn_like(live)
        try:
            return color_noise_wavelet(noise, live, strength, gamma_power, gamma_scale)
        except Exception as exc:
            LOGGER.warning("CNS: colouring skipped at this step - %s", exc)
            return noise

    setattr(cns_noise_sampler, _SETTINGS_ATTR, _settings(strength, gamma_power, gamma_scale))
    return cns_noise_sampler, state


def _settings(strength: float, gamma_power: float, gamma_scale: float) -> dict[str, float]:
    return {
        "strength": float(strength),
        "gamma_power": float(gamma_power),
        "gamma_scale": float(gamma_scale),
    }


def _tracking_callback(state: dict[str, Any], user_callback: Callable[..., Any] | None):
    def callback(info):
        current = info.get("x")
        if current is not None:
            state["x"] = current.detach()
        if user_callback is not None:
            user_callback(info)

    return callback


class _Colouring:
    """Marks that a CNS noise sampler is installed for the current call."""

    def __enter__(self):
        _ACTIVE.depth = getattr(_ACTIVE, "depth", 0) + 1

    def __exit__(self, *exc_info):
        _ACTIVE.depth -= 1
        return False


def _colouring_active() -> bool:
    return getattr(_ACTIVE, "depth", 0) > 0


def _kd_module():
    import comfy.k_diffusion.sampling as kd_sampling  # lazy: only inside ComfyUI

    return kd_sampling


def _takes_noise_sampler(function: Callable[..., Any]) -> bool:
    try:
        parameters = inspect.signature(inspect.unwrap(function)).parameters.values()
    except (ValueError, TypeError, StopIteration):
        return True
    return any(
        item.kind is inspect.Parameter.VAR_KEYWORD or item.name == "noise_sampler"
        for item in parameters
    )


def _kd_samplers(kd_module: Any) -> dict[str, tuple[Callable[..., Any], bool]]:
    """Every callable ``sample_*`` of the k-diffusion module -> (function, takes noise_sampler)."""

    table = {}
    for name in dir(kd_module):
        if not name.startswith("sample_"):
            continue
        function = getattr(kd_module, name, None)
        if function is not None and callable(function):
            table[name] = (function, _takes_noise_sampler(function))
    return table


def _unwrapped(function: Callable[..., Any]) -> Any:
    try:
        return inspect.unwrap(function)
    except Exception:
        return function


def _positional_names(function: Callable[..., Any]) -> list[str] | None:
    try:
        parameters = inspect.signature(inspect.unwrap(function)).parameters.values()
    except (ValueError, TypeError, StopIteration):
        return None
    kinds = (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
    return [item.name for item in parameters if item.kind in kinds]


def _install(kwargs: dict[str, Any], x: Any, settings: dict[str, float], base: Any) -> None:
    noise_sampler, state = make_cns_noise_sampler(
        x, settings["strength"], settings["gamma_power"], settings["gamma_scale"], base,
    )
    kwargs["noise_sampler"] = noise_sampler
    kwargs["callback"] = _tracking_callback(state, kwargs.pop("callback", None))


def _colouring_kd_function(kd_function: Callable[..., Any], settings: dict[str, float]):
    """Path B stand-in for one ``sample_*`` function (keeps ``__wrapped__``)."""

    names = _positional_names(kd_function)

    @functools.wraps(kd_function)
    def colouring(*args, **kwargs):
        if len(args) > 3:
            if _colouring_active():
                # A sample_* calling another positionally (the flow-model *_RF
                # dispatch) while its noise is already coloured: the original
                # raised TypeError here; pass it through, coloured once.
                return kd_function(*args, **kwargs)
            # dpm_fast/dpm_adaptive-style positional call: name the extras.
            if names is None or len(args) > len(names):
                return kd_function(*args, **kwargs)
            named = dict(zip(names[3:], args[3:]))
            if set(named) & set(kwargs):
                return kd_function(*args, **kwargs)
            kwargs = {**named, **kwargs}
            args = args[:3]
        x = args[1] if len(args) > 1 else kwargs.get("x")
        if x is None:
            return kd_function(*args, **kwargs)
        base = kwargs.pop("noise_sampler", None)
        _install(kwargs, x, settings, base)
        with _Colouring():
            return kd_function(*args, **kwargs)

    return colouring


def cns_sampler_function(
    original: Callable[..., Any],
    strength: float = CNS_DEFAULTS["strength"],
    gamma_power: float = CNS_DEFAULTS["gamma_power"],
    gamma_scale: float = CNS_DEFAULTS["gamma_scale"],
    *,
    kd_module: Any = None,
):
    """Wrap a KSAMPLER ``sampler_function`` so every noise draw is coloured.

    ``kd_module`` defaults to ``comfy.k_diffusion.sampling`` (tests pass a fake).
    The returned function deliberately has no ``__wrapped__``: a second CNS
    patch sees it as a wrapper (Path B), as with the original.
    """

    settings = _settings(strength, gamma_power, gamma_scale)

    def cns_sampler(model, x, sigmas, **kwargs):
        module = _kd_module() if kd_module is None else kd_module
        table = _kd_samplers(module)
        target = _unwrapped(original)
        direct = any(
            _unwrapped(function) is target for function, _ in table.values()
        )
        base = kwargs.pop("noise_sampler", None)
        if direct:
            # Path A: ``original`` is a sample_* function.
            if not _takes_noise_sampler(original):
                if base is not None:
                    kwargs["noise_sampler"] = base
                return original(model, x, sigmas, **kwargs)
            _install(kwargs, x, settings, base)
            with _Colouring():
                return original(model, x, sigmas, **kwargs)
        # Path B: a wrapper that calls sample_* by name.
        swapped = {
            name: (function, _colouring_kd_function(function, settings))
            for name, (function, takes_noise) in table.items()
            if takes_noise
        }
        for name, (_, stand_in) in swapped.items():
            setattr(module, name, stand_in)
        if base is not None:
            kwargs["noise_sampler"] = base
        try:
            return original(model, x, sigmas, **kwargs)
        finally:
            for name, (function, _) in swapped.items():
                setattr(module, name, function)

    setattr(cns_sampler, _SETTINGS_ATTR, dict(settings))
    return cns_sampler


def _sampler_sample_wrapper_type() -> str:
    try:
        import comfy.patcher_extension as patcher_extension  # lazy: only inside ComfyUI
    except Exception:
        return _SAMPLER_SAMPLE_FALLBACK
    wrappers = getattr(patcher_extension, "WrappersMP", None)
    return str(getattr(wrappers, "SAMPLER_SAMPLE", _SAMPLER_SAMPLE_FALLBACK))


def cns_sampler_sample_wrapper(
    strength: float = CNS_DEFAULTS["strength"],
    gamma_power: float = CNS_DEFAULTS["gamma_power"],
    gamma_scale: float = CNS_DEFAULTS["gamma_scale"],
    *,
    kd_module: Any = None,
):
    """``WrappersMP.SAMPLER_SAMPLE`` wrapper: CNS on the sampler of every run."""

    settings = _settings(strength, gamma_power, gamma_scale)

    def wrapper(executor, *args, **kwargs):
        sampler = getattr(executor, "class_obj", None)
        original = getattr(sampler, "sampler_function", None)
        if not callable(original):
            # The original node reads ``sampler.sampler_function`` and fails on
            # anything but a KSAMPLER (origin :448); never drop CNS silently.
            raise RuntimeError(
                "CNS wraps KSAMPLER samplers only, like the original node; "
                f"this sampling run uses {type(sampler).__name__}."
            )
        sampler.sampler_function = cns_sampler_function(
            original, **settings, kd_module=kd_module,
        )
        try:
            return executor(*args, **kwargs)
        finally:
            sampler.sampler_function = original

    setattr(wrapper, _SETTINGS_ATTR, dict(settings))
    return wrapper


def apply_cns(
    model: Any,
    strength: float = CNS_DEFAULTS["strength"],
    gamma_power: float = CNS_DEFAULTS["gamma_power"],
    gamma_scale: float = CNS_DEFAULTS["gamma_scale"],
):
    """Clone ``model`` with CNS on every sampler run (replaces an earlier CNS).

    The CNS wrapper is made the outermost ``SAMPLER_SAMPLE`` wrapper (ComfyUI
    runs them in registration order), so it patches the run's real KSAMPLER
    before any other wrapper swaps the sampler.  With Detail Daemon's wrapper
    (``guidance_dd``) that gives the original's working graph
    ``KSamplerSelect -> CNSSamplerPatch -> DetailDaemonSampler``: DD calls
    ``dds_wrapped_sampler.sampler_function`` at run time, which is then the CNS
    function around the real ``sample_*`` (Path A).  Were CNS inside DD, it
    would wrap DD's own function (Path B) and the ``sample_*`` DD already holds
    would never see the coloured noise.
    """

    patched = clone_model(model, "CNS")
    add = getattr(patched, "add_wrapper_with_key", None)
    if not callable(add):
        raise RuntimeError(
            "CNS needs a ComfyUI MODEL that supports sampler wrappers; update ComfyUI."
        )
    wrapper_type = _sampler_sample_wrapper_type()
    remove = getattr(patched, "remove_wrappers_with_key", None)
    if callable(remove):
        remove(wrapper_type, CNS_WRAPPER_KEY)
    add(wrapper_type, CNS_WRAPPER_KEY, cns_sampler_sample_wrapper(strength, gamma_power, gamma_scale))
    _make_outermost(patched, wrapper_type, CNS_WRAPPER_KEY)
    return patched


def _make_outermost(patched: Any, wrapper_type: str, key: str) -> None:
    """Move ``key`` to the front of ``patched.wrappers[wrapper_type]``.

    ``comfy.patcher_extension.get_all_wrappers`` lists the wrappers in the
    dict's order and ``WrapperExecutor`` calls the first one first (outermost).
    ``ModelPatcher.clone`` gives the clone its own per-type dicts, so the model
    that was cloned keeps its order.
    """

    wrappers = getattr(patched, "wrappers", None)
    registered = wrappers.get(wrapper_type) if isinstance(wrappers, dict) else None
    if not isinstance(registered, dict) or key not in registered:
        return
    wrappers[wrapper_type] = {
        key: registered[key],
        **{name: items for name, items in registered.items() if name != key},
    }


def model_cns_settings(model: Any) -> dict[str, float] | None:
    """Settings of the CNS wrapper ``apply_cns`` put on ``model`` (else None)."""

    wrappers = getattr(model, "wrappers", None)
    if not isinstance(wrappers, dict):
        return None
    registered = wrappers.get(_sampler_sample_wrapper_type(), {}).get(CNS_WRAPPER_KEY) or []
    for wrapper in reversed(list(registered)):
        settings = getattr(wrapper, _SETTINGS_ATTR, None)
        if isinstance(settings, dict):
            return dict(settings)
    return None


class ForgeNeoCNSSamplerPatch:
    """SAMPLER -> SAMPLER with coloured ancestral noise (the original node's inputs)."""

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "sampler": ("SAMPLER",),
            "strength": cns_float_input(
                "strength", tooltip="0 = white noise, 1 = fully coloured noise.",
            ),
            "gamma_power": cns_float_input(
                "gamma_power", tooltip="Exponent on the per-band deficit; higher routes noise harder.",
            ),
            "gamma_scale": cns_float_input(
                "gamma_scale",
                tooltip="Divides the band energy share; higher reacts later. Anima + cfg_pp: 3.0.",
            ),
        }}

    RETURN_TYPES = ("SAMPLER",)
    RETURN_NAMES = ("sampler",)
    FUNCTION = "patch"
    CATEGORY = CATEGORY
    DESCRIPTION = (
        "Colored Noise Sampling: recolours the noise an ancestral/SDE sampler adds each step "
        "from the live latent's Haar band energies. ODE samplers are unchanged."
    )

    def patch(self, sampler, strength, gamma_power, gamma_scale):
        import comfy.samplers  # lazy: only reachable inside ComfyUI

        return (comfy.samplers.KSAMPLER(
            cns_sampler_function(sampler.sampler_function, strength, gamma_power, gamma_scale),
            extra_options=sampler.extra_options,
            inpaint_options=sampler.inpaint_options,
        ),)
