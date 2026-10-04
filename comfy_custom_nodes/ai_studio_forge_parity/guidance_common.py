"""Helpers shared by this pack's guidance nodes (``guidance*.py``).

Sampling progress from a sigma, ``settings_json`` parsing, block-index parsing,
Anima/Cosmos block lookup and the Haar wavelet that CNS, DCW and CWM use.
Imports are deliberately lazy so the app can validate contracts without
loading ComfyUI or torch.

Since pack 1.7.0 it also holds the names sam-extra's features use to find each
other (sub-step marker, step-offset keyword, re-published sigma lists) and the
small helpers that read them, so every pack port of a sam-extra feature writes
and reads them one way. The values are sam-extra's; the code is the pack's own.
"""

from __future__ import annotations

import inspect
from typing import Any, Callable

from .compat import json_object, require_torch


CATEGORY = "AI Studio/Forge Neo parity/Guidance"
# Detail Daemon (guidance_dd) scales the sigma the model gets; the DAVE gate (guidance_dave) looks
# the sigma up in the schedule, where a scaled one is never found (the originals chained run DAVE
# on every step). Detail Daemon notes the sampler's own sigma under this transformer_options key
# for every model call it wraps; DAVE's pre-DD option (default on) looks that one up instead.
PRE_DD_SIGMAS_KEY = "ai_studio_pre_dd_sigmas"
# S²-Guidance (guidance_s2): the blocks one SLG weak evaluation skips, from the suite's draw to the block
# wrappers (guidance_dave._patch_anima_blocks). Absent = fixed SLG (every wrapped SLG block is skipped).
S2_DROP_KEY = "forge_neo_slg_drop"

# --- sam-extra coordination names (pack 1.7.0) ------------------------------------------------
# The pack's ports of sam-extra features read what another port wrote under the extension's own
# names. tests/test_comfy_pack_foundation.py pins each value to the installed extension's source.
#
# transformer_options key on the extra model evaluations of Euler (SMEA) Dy CFG++ (the
# half-resolution Dy evaluation, the x1.25 SMEA one), value "dy" or "smea" — sam-extra
# sam3ext/extra_samplers/common.py ``SUBSTEP_MARKER``. State kept per sampler step (momentum,
# history, trajectories, e_prev) ignores those evaluations: ``is_extra_sampler_substep``.
SUBSTEP_MARKER = "sam_extra_substep"
# Keyword a sampler function names in its signature to learn where a segment starts in its pass:
# SPEED calls the sampler once per segment, Dy places its sub-steps by the pass's step index —
# sam-extra sam3ext/speed/runner.py ``STEP_OFFSET_KWARG``. Only passed to a function that names it
# (``declares_keyword``), never through ``**kwargs``.
STEP_OFFSET_KWARG = "sam_extra_step_offset"
# Attribute on a sigma list published in place of the run's own list mid-run (SPEED's patched
# copy), pointing at the run's list — sam-extra sam3ext/guidance/sigmas.py ``REPUBLISHED_FROM``.
REPUBLISHED_FROM = "_sam_extra_republished_from"
# Attribute a sampler-function wrapper without ``__wrapped__`` sets to the function it wraps
# (guidance_cns.cns_sampler_function has no ``__wrapped__`` on purpose: a second CNS patch must see
# it as a wrapper). ``sampler_base`` follows it to the real ``sample_*``.
SAMPLER_BASE_ATTR = "_forge_neo_base"
# transformer_options key (the pack's own): Euler (SMEA) Dy CFG++ runs without its sub-steps on
# this MODEL's sampling runs. ForgeNeoKSamplerCNS sets it on its own clone when ComfyUI Spectrum
# (DiTSpectrumPatch) or the legacy SPEED provider (SpectrumSPDKSampler, ``speed_enabled``) samples:
# both forecast whole steps (sam-extra skips the sub-steps under Spectrum Integrated). The value
# is the reason, for the console line. Never set for the pack's ForgeNeoAnimaSpeed: sam-extra keeps
# Dy sub-steps under SPEED and passes ``STEP_OFFSET_KWARG`` instead.
DY_SUBSTEPS_OFF_KEY = "ai_studio_dy_substeps_off"


def parse_indices(spec: str, count: int, *, default: str = "") -> set[int]:
    """Parse ``8-18,22`` and clamp it to the available block count.

    A reversed range is swapped, so ``20-18`` is 18..20 (origin:
    iljung1106/comfyui-anima-safe-pag@905b0107:__init__.py:49-51).
    """

    text = str(spec or default).replace(" ", "")
    result: set[int] = set()
    for part in text.split(","):
        if not part:
            continue
        if "-" in part:
            left, _, right = part.partition("-")
            try:
                start, end = int(left), int(right)
            except ValueError:
                continue
            if end < start:
                start, end = end, start
            result.update(range(start, end + 1))
        else:
            try:
                result.add(int(part))
            except ValueError:
                continue
    return {index for index in result if 0 <= index < max(0, int(count))}


def _scalar_sigma(sigma: Any) -> float:
    try:
        return float(sigma.flatten()[0].item())
    except Exception:
        try:
            return float(sigma)
        except Exception:
            return 1.0


def _sampling_percent(sigma: Any) -> float:
    # Anima is a flow model whose public sampler sigma falls from ~1 to 0.
    return min(1.0, max(0.0, 1.0 - _scalar_sigma(sigma)))


def _sampling_percent_for_model(model: Any, sigma: Any) -> float:
    """Invert Comfy's model-specific percent_to_sigma when it is available."""

    inner = getattr(model, "model", None)
    sampling = getattr(inner, "model_sampling", None)
    convert = getattr(sampling, "percent_to_sigma", None)
    if not callable(convert):
        return _sampling_percent(sigma)
    target = _scalar_sigma(sigma)
    try:
        low, high = 0.0, 1.0
        for _ in range(24):
            middle = (low + high) * 0.5
            candidate = float(convert(middle))
            if candidate > target:
                low = middle
            else:
                high = middle
        return min(1.0, max(0.0, (low + high) * 0.5))
    except Exception:
        return _sampling_percent(sigma)


# settings_json parsing is shared by every node in this pack (compat.json_object).
_json_settings = json_object


def _as_bool(value: Any, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    return str(value).strip().casefold() in {"1", "true", "yes", "on", "enabled"}


def _setting(settings: dict[str, Any], key: str, default: Any) -> Any:
    value = settings.get(key, default)
    return default if value is None else value


def _pad_even(value: Any):
    torch = require_torch()
    height, width = int(value.shape[-2]), int(value.shape[-1])
    pad_h, pad_w = height % 2, width % 2
    if not (pad_h or pad_w):
        return value, (height, width)
    padding = (0, pad_w, 0, pad_h) + (0, 0) * max(0, value.ndim - 4)
    mode = "reflect" if height > 1 and width > 1 else "replicate"
    try:
        padded = torch.nn.functional.pad(value, padding, mode=mode)
    except RuntimeError:
        padded = torch.nn.functional.pad(value, padding, mode="constant", value=0)
    return padded, (height, width)


def _haar_dwt(value: Any):
    low_h = (value[..., 0::2, :] + value[..., 1::2, :]) * 0.5
    high_h = (value[..., 0::2, :] - value[..., 1::2, :]) * 0.5
    return (
        (low_h[..., 0::2] + low_h[..., 1::2]) * 0.5,
        (low_h[..., 0::2] - low_h[..., 1::2]) * 0.5,
        (high_h[..., 0::2] + high_h[..., 1::2]) * 0.5,
        (high_h[..., 0::2] - high_h[..., 1::2]) * 0.5,
    )


def _haar_idwt(ll: Any, lh: Any, hl: Any, hh: Any):
    torch = require_torch()
    *leading, height, width = ll.shape
    low_h = torch.empty(
        *leading, height, width * 2, device=ll.device, dtype=ll.dtype
    )
    high_h = torch.empty_like(low_h)
    low_h[..., 0::2], low_h[..., 1::2] = ll + lh, ll - lh
    high_h[..., 0::2], high_h[..., 1::2] = hl + hh, hl - hh
    output = torch.empty(
        *leading, height * 2, width * 2, device=ll.device, dtype=ll.dtype
    )
    output[..., 0::2, :] = low_h + high_h
    output[..., 1::2, :] = low_h - high_h
    return output


def _sigma_norm(sigma: Any, like: Any):
    torch = require_torch()
    if torch.is_tensor(sigma):
        value = sigma.float() / (sigma.float() + 1.0)
        if value.ndim == 1:
            value = value.view(-1, *([1] * (like.ndim - 1)))
        return value.to(device=like.device, dtype=like.dtype)
    value = float(sigma)
    return torch.as_tensor(
        value / (value + 1.0), device=like.device, dtype=like.dtype
    )


def _model_blocks(model: Any) -> list[Any]:
    current = getattr(model, "model", None)
    diffusion = getattr(current, "diffusion_model", None)
    blocks = getattr(diffusion, "blocks", None)
    if blocks is None:
        return []
    return list(blocks)


def _transformer_options(args: tuple[Any, ...], kwargs: dict[str, Any]) -> dict[str, Any]:
    options = kwargs.get("transformer_options")
    if isinstance(options, dict):
        return options
    # Cosmos/Predict2 Block.forward has transformer_options as positional #7.
    if len(args) > 6 and isinstance(args[6], dict):
        return args[6]
    return {}


# --- sam-extra coordination helpers (pack 1.7.0) ----------------------------------------------


def set_transformer_option(model: Any, key: str, value: Any) -> Any:
    """Set ``model_options["transformer_options"][key]`` on ``model`` and return it.

    ``model`` must be a clone the caller owns; the option dicts are copied first, so the
    MODEL it was cloned from (and Comfy's cached outputs) never see the key.
    """

    options = dict(getattr(model, "model_options", None) or {})
    transformer = dict(options.get("transformer_options") or {})
    transformer[key] = value
    options["transformer_options"] = transformer
    model.model_options = options
    return model


def is_extra_sampler_substep(args: Any) -> bool:
    """Is this evaluation one of Euler (SMEA) Dy CFG++'s extra sub-steps (``SUBSTEP_MARKER``)?

    ``args`` is a post-CFG/pre-CFG dict (options under ``args["model_options"]``) or a
    model-function-wrapper dict (options under ``args["c"]``). The model options' non-empty
    ``transformer_options`` are read first, else the wrapper's — sam-extra's lookup order. A
    sub-step's x0 is not the step's: step-keyed state is neither updated nor reset by it, while
    stateless stages (TSR) still apply.
    """

    if not isinstance(args, dict):
        return False
    options = _first_options(args.get("model_options"), args.get("c"))
    return bool(options.get(SUBSTEP_MARKER))


def _first_options(*holders: Any) -> dict[str, Any]:
    for holder in holders:
        if isinstance(holder, dict):
            options = holder.get("transformer_options")
            if options:
                return options if isinstance(options, dict) else {}
    return {}


def mark_republished(sigmas: Any, source: Any) -> Any:
    """Return ``sigmas`` tagged as published in place of the run's list ``source``.

    The tag always names the run's first list: re-publishing a published copy points back at
    the list the copy stood for. A list that takes no attributes stays untagged.
    """

    origin = republished_from(source)
    if origin is None:
        origin = source
    try:
        setattr(sigmas, REPUBLISHED_FROM, origin)
    except (AttributeError, TypeError):
        pass
    return sigmas


def republished_from(sigmas: Any) -> Any:
    """The run's list ``sigmas`` was published in place of (``mark_republished``), else None."""

    return getattr(sigmas, REPUBLISHED_FROM, None)


def sampler_base(function: Callable[..., Any]) -> Callable[..., Any]:
    """The sampler function a chain of wrappers stands for.

    Follows ``SAMPLER_BASE_ATTR`` (wrappers without ``__wrapped__``, CNS) and ``__wrapped__``
    (``functools.wraps``) until neither is set; a cycle stops at the last new function and the
    walk gives up after ``_SAMPLER_BASE_MAX_HOPS`` links (an object inventing attributes).
    """

    seen = {id(function)}
    current = function
    for _ in range(_SAMPLER_BASE_MAX_HOPS):
        following = getattr(current, SAMPLER_BASE_ATTR, None)
        if following is None:
            following = getattr(current, "__wrapped__", None)
        if following is None or not callable(following) or id(following) in seen:
            return current
        seen.add(id(following))
        current = following
    return current


_SAMPLER_BASE_MAX_HOPS = 32


def declares_keyword(function: Callable[..., Any], name: str) -> bool:
    """Does the sampler behind ``function`` (``sampler_base``) name the keyword ``name`` itself?

    A bare ``**kwargs`` does not count — sam-extra passes ``STEP_OFFSET_KWARG`` only to a
    sampler that names it. An unreadable signature counts as not declared.
    """

    try:
        parameter = inspect.signature(sampler_base(function)).parameters.get(name)
    except (TypeError, ValueError):
        return False
    return parameter is not None and parameter.kind in (
        inspect.Parameter.POSITIONAL_OR_KEYWORD,
        inspect.Parameter.KEYWORD_ONLY,
    )
