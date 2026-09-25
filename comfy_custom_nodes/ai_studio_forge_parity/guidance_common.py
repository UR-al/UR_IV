"""Helpers shared by this pack's guidance nodes (``guidance*.py``).

Sampling progress from a sigma, ``settings_json`` parsing, block-index parsing,
Anima/Cosmos block lookup and the Haar wavelet that CNS, DCW and CWM use.
Imports are deliberately lazy so the app can validate contracts without
loading ComfyUI or torch.
"""

from __future__ import annotations

from typing import Any

from .compat import json_object, require_torch


CATEGORY = "AI Studio/Forge Neo parity/Guidance"


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
