"""One image through VAE DeGrid, given a model callable (no ComfyUI, no files).

``degrid_one`` is the per-image pipeline of ``ForgeNeoAnimaVAEDeGrid``:

1. optional Forge quantisation of the input (``uint8(255 x) / 255``),
2. residual by tiles (reflect-padded to the model's multiple, fp32 input;
   fp16 autocast only when asked and on CUDA, with a per-tile fp32 re-run
   when a tile overflows), out of memory halves the tile down to 128,
3. residual guard (image-like output or blow-up -> ``DegridSkip``),
4. ``clamp(x + s * f(delta), 0, 1)``, optionally rounded to 8-bit levels.

The node (``degrid_nodes``) owns model loading, devices and ComfyUI memory
management; this module only needs torch.
"""

from __future__ import annotations

import contextlib
from dataclasses import dataclass
from typing import Any, Callable, Optional

from . import degrid_math as dm
from .compat import require_torch

PRECISION_FP32 = "fp32"
PRECISION_FP16 = "fp16"
PRECISION_LABEL_FP32 = "fp32"
PRECISION_LABEL_AUTOCAST = "fp16-autocast"
PRECISION_LABEL_NONE = "-"


@dataclass(frozen=True)
class DegridResult:
    image: Any            # [1, 3, H, W] float32
    tile_used: int
    precision: str        # "fp32" | "fp16-autocast"
    fp32_retiles: int = 0
    check: Optional[dm.ResidualCheck] = None


def pad_multiple_of(model: Any) -> int:
    """The multiple the model pads to internally (spandrel ``padder_size``), at least 1."""
    try:
        multiple = int(getattr(model, "padder_size", dm.PAD_MULTIPLE))
    except (TypeError, ValueError):
        multiple = dm.PAD_MULTIPLE
    return max(1, multiple)


def use_autocast(device: Any, precision: Any) -> bool:
    """fp16 autocast only for an explicit ``fp16`` on a CUDA device; everything else runs fp32."""
    torch = require_torch()
    return torch.device(device).type == "cuda" and str(precision or "").strip().lower() == PRECISION_FP16


def residual_function(
    model: Callable,
    device: Any,
    autocast: bool,
    *,
    autocast_context: Optional[Callable[[], Any]] = None,
    before_piece: Optional[Callable[[], None]] = None,
):
    """``(fn, retile_counter)``: fn maps a tile to its float32 residual on ``device``.

    The raw model forward is the residual (never a clamping upscaler wrapper,
    which would drop the negative half). With ``autocast`` a tile whose
    output is not finite is computed again in fp32 and counted.
    """
    torch = require_torch()
    retiles = [0]
    multiple = pad_multiple_of(model)
    target = torch.device(device)

    def make_context():
        if autocast_context is not None:
            return autocast_context()
        return torch.autocast(device_type=target.type, dtype=torch.float16)

    def residual(piece):
        if before_piece is not None:
            before_piece()
        piece = piece.to(device=target, dtype=torch.float32)
        if autocast:
            with make_context():
                out = dm.call_padded(model, piece, multiple)
            if bool(torch.isfinite(out).all()):
                return out.float()
            retiles[0] += 1
        return dm.call_padded(model, piece, multiple).float()

    return residual, retiles


def degrid_one(
    image,
    model: Callable,
    *,
    model_name: str,
    mode: str,
    strength: float,
    tile: int,
    device: Any = "cpu",
    precision: str = PRECISION_FP32,
    forge_quantize: bool = True,
    is_oom: Callable[[BaseException], bool] = dm.message_says_oom,
    on_retry: Optional[Callable[[int, BaseException], None]] = None,
    on_piece: Optional[Callable[[], None]] = None,
    before_piece: Optional[Callable[[], None]] = None,
    autocast_context: Optional[Callable[[], Any]] = None,
) -> DegridResult:
    """DeGrid ``image`` ([1, 3, H, W], values in [0, 1]); raises ``DegridSkip`` for guarded skips.

    ``mode``/``strength``/``tile`` must already be normalised (callers handle
    strength 0 as a no-op before loading any model).
    """
    torch = require_torch()
    x = image.detach().to(device="cpu", dtype=torch.float32)
    if forge_quantize:
        x = dm.forge_quantize_in(x)
    autocast = use_autocast(device, precision)
    fn, retiles = residual_function(
        model, device, autocast, autocast_context=autocast_context, before_piece=before_piece,
    )
    guard_context = torch.inference_mode() if hasattr(torch, "inference_mode") else contextlib.nullcontext()
    with guard_context:
        delta, used = dm.tiled_residual_with_oom_retry(
            x, fn, tile=tile, overlap=dm.TILE_OVERLAP, out_device="cpu",
            is_oom=is_oom, on_retry=on_retry, on_piece=on_piece,
        )
        check = dm.guard(x, delta, model_name)
        result = dm.finalize(dm.apply_residual(x, delta, mode, strength))
        if forge_quantize:
            result = dm.round_out(result)
    return DegridResult(
        image=result,
        tile_used=int(used),
        precision=PRECISION_LABEL_AUTOCAST if autocast else PRECISION_LABEL_FP32,
        fp32_retiles=retiles[0],
        check=check,
    )


def report(
    *,
    status: str,
    model: str,
    mode: str,
    strength: float,
    tile: int,
    precision: str = PRECISION_LABEL_NONE,
    error: str = "",
) -> dict:
    """One batch item's report (``status`` ok / skipped / off; ``mode`` is a label)."""
    label = dm.MODE_LABELS.get(dm.normalize_mode(mode) or dm.DEFAULT_MODE, dm.MODE_LABELS[dm.DEFAULT_MODE])
    return {
        "status": status,
        "model": str(model),
        "mode": label,
        "strength": float(strength),
        "tile": int(tile),
        "precision": str(precision or PRECISION_LABEL_NONE),
        "error": " ".join(str(error or "").split()),
    }


__all__ = [
    "DegridResult", "PRECISION_FP16", "PRECISION_FP32", "PRECISION_LABEL_AUTOCAST",
    "PRECISION_LABEL_FP32", "PRECISION_LABEL_NONE", "degrid_one", "pad_multiple_of", "report",
    "residual_function", "use_autocast",
]
