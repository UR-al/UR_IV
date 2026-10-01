"""VAE DeGrid (NAFNet residual) arithmetic for ``ForgeNeoAnimaVAEDeGrid``.

Local implementation of the behaviour of the sam-extra Forge extension's
"Anima VAE DeGrid (NAFNet)" script (forge_sam3_extension @ 395854b,
``sam3ext/vae_degrid.py``). That extension is GPL-3.0; none of its code is
copied here. What is shared are behavioural facts: the mode meanings, value
ranges, tile/overlap sizes, padding multiple, the residual-guard thresholds
(plain numbers) and the error *category* words the app's notices key on.
``tests/test_comfy_degrid_origin.py`` checks this module against numbers
produced by running the extension on CPU (``tests/fixtures/degrid_origin_golden.json``).

The model (DraconicDragon/NAFNet-VAE-DeGrid, Apache-2.0) returns a *residual*:
its raw forward output (spandrel NAFNet, including the network's own
``+ input`` term) is added to the image instead of replacing it. Modes:

    Full                   image + s * delta
    Dark Pixels Mainly     image + s * max(delta, 0)
    Bright Pixels Mainly   image + s * min(delta, 0)

followed by a single clamp to [0, 1] of the final image.

Tiling is a scale-1 weighted-overlap blend (tile 512, overlap 32, linear edge
ramps) whose tile positions and weights equal ComfyUI ``comfy.utils.tiled_scale``
and the extension's port of it, so all three produce the same residual.

torch is imported lazily (``compat.require_torch``): the desktop app imports
this pack in its test process, where only the constants and plain-Python
helpers may load.
"""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass
from typing import Any, Callable, Optional

from .compat import require_torch


# ── modes ────────────────────────────────────────────────────────────────
MODE_FULL = "full"
MODE_DARK = "dark"
MODE_BRIGHT = "bright"
MODES = (MODE_FULL, MODE_DARK, MODE_BRIGHT)
DEFAULT_MODE = MODE_FULL

# Node-pack labels (ComfyUI-NAFNet-Residual names); also the infotext values.
MODE_LABELS = {
    MODE_FULL: "Full",
    MODE_DARK: "Dark Pixels Mainly",
    MODE_BRIGHT: "Bright Pixels Mainly",
}
MODE_CHOICES = tuple(MODE_LABELS[key] for key in MODES)

_MODE_WORDS = {
    "full": MODE_FULL,
    "dark": MODE_DARK,
    "dark pixels": MODE_DARK,
    "dark_pixels": MODE_DARK,
    "dark pixels mainly": MODE_DARK,
    "bright": MODE_BRIGHT,
    "bright pixels": MODE_BRIGHT,
    "bright_pixels": MODE_BRIGHT,
    "bright pixels mainly": MODE_BRIGHT,
}

# ── ranges ───────────────────────────────────────────────────────────────
DEFAULT_STRENGTH = 1.0
STRENGTH_MIN = 0.0
STRENGTH_MAX = 1.5

DEFAULT_TILE = 512
TILE_OVERLAP = 32
MIN_TILE = 128
MAX_TILE = 4096

# Fallback for the multiple a NAFNet pads to internally (spandrel
# ``padder_size`` = 2 ** encoder stages; NAFNet-small has four stages).
PAD_MULTIPLE = 16

# ── residual guard (numbers in [0, 1] units) ─────────────────────────────
# A restoration NAFNet (denoise/deblur) outputs an *image* that follows the
# input; a DeGrid model outputs a small residual that does not. Below the
# minimum the output is not examined at all.
IMAGE_LIKE_MIN_ABS_MEAN = 2 / 255
IMAGE_LIKE_CORRELATION = 0.9          # follows the input this closely -> image
IMAGE_LIKE_ABS_MEAN = 25 / 255        # large output ...
IMAGE_LIKE_LARGE_CORRELATION = 0.5    # ... that follows moderately (or a flat input) -> image
IMAGE_LIKE_DC_MEAN = 25 / 255         # brightness rule: |mean output| above this,
IMAGE_LIKE_DC_SIGN = 0.8              # one-signed (|mean| >= 0.8 * mean|output|),
IMAGE_LIKE_DC_RATIO = 0.5             # channel means projected on the input's above this
RESIDUAL_BLOWUP_ABS_MEAN = 100 / 255  # a residual this large ruins the image -> skip it

# ── model files ──────────────────────────────────────────────────────────
# spandrel's NAFNet detection keys (spandrel is MIT): a file is listed only
# when its state dict has every one of them.
NAFNET_KEYS = (
    "intro.weight",
    "ending.weight",
    "ups.0.0.weight",
    "downs.0.weight",
    "middle_blks.0.beta",
    "middle_blks.0.gamma",
    "middle_blks.0.conv1.weight",
    "middle_blks.0.conv2.weight",
    "middle_blks.0.conv3.weight",
    "middle_blks.0.sca.1.weight",
    "middle_blks.0.conv4.weight",
    "middle_blks.0.conv5.weight",
    "middle_blks.0.norm1.weight",
    "middle_blks.0.norm2.weight",
    "encoders.0.0.beta",
    "encoders.0.0.gamma",
    "decoders.0.0.beta",
    "decoders.0.0.gamma",
)
MODEL_EXTENSIONS = (".safetensors", ".pth", ".pt")

# Error categories (the leading words of a report's ``error``). The app maps
# them to its hints; the rest of each message is this pack's own wording.
ERROR_MODEL_NOT_FOUND = "model not found"
ERROR_NOT_RESIDUAL = "not a DeGrid residual model"
ERROR_BLEW_UP = "output blew up"


class DegridSkip(ValueError):
    """An expected skip: the image is kept unchanged and only the message is reported."""


class NotResidualModel(DegridSkip):
    """The model's output follows the input like an image (not a DeGrid model)."""


class ResidualBlewUp(DegridSkip):
    """The residual is far too large for this image (untrained pattern)."""


# ── plain-Python coercion ────────────────────────────────────────────────
def normalize_mode(value: Any) -> Optional[str]:
    """Mode key from a key, node label or UI label (text before ``(`` counts). Unknown -> None."""
    if value is None:
        return None
    text = str(value).split("(", 1)[0].strip().lower()
    if not text:
        return None
    return _MODE_WORDS.get(text)


def coerce_strength(value: Any, default: float = DEFAULT_STRENGTH) -> float:
    """Strength clamped to [0, 1.5]; unreadable or NaN -> ``default``."""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return float(default)
    if math.isnan(number):
        return float(default)
    return min(STRENGTH_MAX, max(STRENGTH_MIN, number))


def coerce_tile(value: Any, default: int = DEFAULT_TILE) -> int:
    """Tile size: 0 (one piece) or 128..4096 (1..127 become 128); unreadable -> ``default``."""
    try:
        number = int(float(value))
    except (TypeError, ValueError, OverflowError):
        return int(default)
    if number <= 0:
        return 0
    return min(MAX_TILE, max(MIN_TILE, number))


def tile_positions(size: int, tile: int, overlap: int = TILE_OVERLAP) -> list[int]:
    """Start offsets along one axis: every ``tile - overlap`` below ``size - overlap``."""
    size, tile, overlap = int(size), int(tile), int(overlap)
    if size <= tile:
        return [0]
    last = size - overlap
    return [min(last, max(0, start)) for start in range(0, last, tile - overlap)]


# ── residual arithmetic ──────────────────────────────────────────────────
def select_residual(delta, mode: str):
    """The part of ``delta`` a mode uses: all, the positive part, or the negative part."""
    if mode == MODE_FULL:
        return delta
    if mode == MODE_DARK:
        return delta.clamp_min(0)
    if mode == MODE_BRIGHT:
        return delta.clamp_max(0)
    raise ValueError(f"Unknown DeGrid mode: {mode!r}")


def apply_residual(image, delta, mode: str, strength: float = DEFAULT_STRENGTH):
    """``image + strength * select(delta)`` without clamping (``finalize`` clamps once)."""
    part = select_residual(delta, mode)
    if strength != 1.0:
        part = part * float(strength)
    return image + part


def finalize(image):
    """Clamp the finished image to [0, 1]."""
    return require_torch().clamp(image, min=0.0, max=1.0)


def forge_quantize_in(image):
    """Forge hands the extension an 8-bit image: ``uint8(255 * x)`` (truncation) / 255.

    Comfy's decoded IMAGE is float; flooring it the same way makes the node see
    exactly the pixels the Forge script sees.
    """
    torch = require_torch()
    x = torch.clamp(image.to(dtype=torch.float32), min=0.0, max=1.0)
    return torch.floor(x * 255.0) / 255.0


def round_out(image):
    """Round to the nearest 8-bit level (half to even) as ``level / 255`` in float32.

    Every level survives the truncating ``uint8(255 * x)`` of ComfyUI's save
    nodes, so the saved file holds the same bytes as Forge's rounded PIL image.
    """
    torch = require_torch()
    x = image.to(dtype=torch.float32)
    return torch.clamp(torch.round(x * 255.0), min=0.0, max=255.0) / 255.0


# ── padding / tiling ─────────────────────────────────────────────────────
def pad_to_multiple(x, multiple: int):
    """Pad the bottom/right of ``[B, C, H, W]`` up to ``multiple`` by reflection.

    Inputs too small to reflect (padding >= the axis length) replicate their
    edge instead. Already aligned inputs are returned as they are.
    """
    torch = require_torch()
    multiple = int(multiple)
    if multiple <= 1:
        return x
    height, width = int(x.shape[-2]), int(x.shape[-1])
    extra_h = -height % multiple
    extra_w = -width % multiple
    if extra_h == 0 and extra_w == 0:
        return x
    reflectable = extra_h < height and extra_w < width
    return torch.nn.functional.pad(
        x, (0, extra_w, 0, extra_h), mode="reflect" if reflectable else "replicate",
    )


def call_padded(fn: Callable, piece, multiple: int):
    """Run ``fn`` on ``piece`` padded to ``multiple`` and crop back to its size.

    spandrel's NAFNet zero-pads unaligned inputs itself, which shows up as a
    strong residual along the right/bottom edge; reflection avoids that.
    Aligned pieces (all tiles of the standard Anima sizes) run unpadded.
    """
    padded = pad_to_multiple(piece, multiple)
    if padded is piece:
        return fn(piece)
    height, width = int(piece.shape[-2]), int(piece.shape[-1])
    return fn(padded)[..., :height, :width]


def _edge_weights(length: int, overlap: int) -> list[float]:
    """Per-index factors of one axis: the ``overlap`` cells at each end ramp (t + 1) / overlap."""
    factors = [1.0] * length
    if 0 < overlap < length:
        for step in range(overlap):
            ramp = (step + 1) / overlap
            factors[step] *= ramp
            factors[length - 1 - step] *= ramp
    return factors


def blend_mask(height: int, width: int, overlap: int, *, device="cpu"):
    """``[1, 1, H, W]`` float32 blend weights of one tile."""
    torch = require_torch()
    mask = torch.ones((1, 1, height, width), dtype=torch.float32, device=device)
    for dim, length in ((2, height), (3, width)):
        for index, factor in enumerate(_edge_weights(length, overlap)):
            if factor != 1.0:
                mask.narrow(dim, index, 1).mul_(factor)
    return mask


def tiled_residual(
    x,
    fn: Callable,
    *,
    tile: int = DEFAULT_TILE,
    overlap: int = TILE_OVERLAP,
    out_device="cpu",
    on_piece: Optional[Callable[[], None]] = None,
):
    """Residual of ``x`` (``[B, 3, H, W]``) from ``fn`` run tile by tile.

    Overlapping tiles are blended with ``blend_mask`` weights. ``tile <= 0``
    or an image that fits in one tile runs in one call. Returns float32 on
    ``out_device``.
    """
    torch = require_torch()
    batch, channels, height, width = (int(v) for v in x.shape)
    result = torch.empty((batch, channels, height, width), dtype=torch.float32, device=out_device)
    for index in range(batch):
        sample = x[index:index + 1]
        if tile <= 0 or (height <= tile and width <= tile):
            result[index:index + 1] = fn(sample).to(device=out_device, dtype=torch.float32)
            if on_piece is not None:
                on_piece()
            continue
        total = torch.zeros((1, channels, height, width), dtype=torch.float32, device=out_device)
        weight = torch.zeros((1, 1, height, width), dtype=torch.float32, device=out_device)
        rows = tile_positions(height, tile, overlap)
        cols = tile_positions(width, tile, overlap)
        for top, left in itertools.product(rows, cols):
            tile_h = min(tile, height - top)
            tile_w = min(tile, width - left)
            piece = fn(sample[:, :, top:top + tile_h, left:left + tile_w])
            piece = piece.to(device=out_device, dtype=torch.float32)
            mask = blend_mask(tile_h, tile_w, overlap, device=out_device)
            total[:, :, top:top + tile_h, left:left + tile_w].add_(piece * mask)
            weight[:, :, top:top + tile_h, left:left + tile_w].add_(mask)
            if on_piece is not None:
                on_piece()
        result[index:index + 1] = total / weight
    return result


def message_says_oom(exc: BaseException) -> bool:
    """Out-of-memory by type (torch OOM error) or by message."""
    try:
        torch = require_torch()
        oom_type = getattr(torch, "OutOfMemoryError", None) or getattr(
            getattr(torch, "cuda", None), "OutOfMemoryError", None,
        )
    except RuntimeError:
        oom_type = None
    if oom_type is not None and isinstance(exc, oom_type):
        return True
    return "out of memory" in str(exc).lower()


def tiled_residual_with_oom_retry(
    x,
    fn: Callable,
    *,
    tile: int = DEFAULT_TILE,
    overlap: int = TILE_OVERLAP,
    out_device="cpu",
    is_oom: Callable[[BaseException], bool] = message_says_oom,
    on_retry: Optional[Callable[[int, BaseException], None]] = None,
    on_piece: Optional[Callable[[], None]] = None,
):
    """``(residual, tile actually used)``; out of memory halves the tile down to 128.

    Tile 0 (one piece) that runs out of memory continues at half the long
    side. Other errors, and running out below 128, are raised.
    """
    height, width = int(x.shape[-2]), int(x.shape[-1])
    current = int(tile)
    while True:
        try:
            residual = tiled_residual(
                x, fn, tile=current, overlap=overlap, out_device=out_device, on_piece=on_piece,
            )
            return residual, current
        except Exception as exc:
            if not is_oom(exc):
                raise
            current = (current if current > 0 else max(height, width)) // 2
            if current < MIN_TILE:
                raise
            if on_retry is not None:
                on_retry(current, exc)


# ── residual guard ───────────────────────────────────────────────────────
@dataclass(frozen=True)
class ResidualCheck:
    """What the guard measured on a residual (``[0, 1]`` units)."""

    mean_abs: float
    correlation: Optional[float]
    input_flat: bool
    looks_like_image: bool
    signed_mean: float = 0.0
    dc_ratio: Optional[float] = None
    follows_input_mean: bool = False
    blew_up: bool = False


def _per_channel_means(t):
    """float64 per-channel means of ``(..., C, H, W)`` (one overall mean below 3 dims)."""
    torch = require_torch()
    data = t.detach().to(dtype=torch.float32)
    if data.ndim < 3:
        return data.mean().reshape(1).to(dtype=torch.float64)
    channel_dim = data.ndim - 3
    others = tuple(dim for dim in range(data.ndim) if dim != channel_dim)
    return data.mean(dim=others).to(dtype=torch.float64)


def brightness_ratio(image, residual) -> Optional[float]:
    """Projection of the residual's channel means on the input's (image-like ~1, residual ~0).

    None when the input is black (nothing to project on) or shapes differ.
    """
    image_means = _per_channel_means(image)
    residual_means = _per_channel_means(residual).to(device=image_means.device)
    if residual_means.shape != image_means.shape:
        return None
    norm = float((image_means * image_means).sum())
    if not norm > 0.0:
        return None
    return float((residual_means * image_means).sum()) / norm


def check_residual(image, residual) -> ResidualCheck:
    """Decide whether a model output is a residual (keep) or an image / blow-up (skip)."""
    torch = require_torch()
    mean_abs = float(residual.abs().mean()) if residual.numel() else 0.0
    if not mean_abs > IMAGE_LIKE_MIN_ABS_MEAN:
        return ResidualCheck(mean_abs, None, False, False)
    blew_up = mean_abs > RESIDUAL_BLOWUP_ABS_MEAN
    signed_mean = float(residual.detach().to(dtype=torch.float32).mean())
    ratio = brightness_ratio(image, residual)
    follows_mean = bool(
        ratio is not None
        and abs(signed_mean) > IMAGE_LIKE_DC_MEAN
        and abs(signed_mean) >= IMAGE_LIKE_DC_SIGN * mean_abs
        and ratio > IMAGE_LIKE_DC_RATIO
    )

    def verdict(correlation, flat, image_like):
        return ResidualCheck(
            mean_abs, correlation, flat, bool(image_like or follows_mean),
            signed_mean, ratio, follows_mean, blew_up,
        )

    inp = image.detach().reshape(-1).to(dtype=torch.float32)
    if bool(inp.amax() == inp.amin()):
        # A one-colour input has no pattern to follow; a large output there
        # is that colour coming back, i.e. an image.
        return verdict(None, True, mean_abs > IMAGE_LIKE_ABS_MEAN)
    out = residual.detach().reshape(-1).to(device=inp.device, dtype=torch.float32)
    inp = inp - inp.mean()
    out = out - out.mean()
    scale = float(inp.norm()) * float(out.norm())
    if not scale > 0.0:
        return verdict(None, False, False)
    correlation = float(torch.dot(inp, out)) / scale
    image_like = correlation > IMAGE_LIKE_CORRELATION or (
        mean_abs > IMAGE_LIKE_ABS_MEAN and correlation > IMAGE_LIKE_LARGE_CORRELATION
    )
    return verdict(correlation, False, image_like)


def not_residual_message(model_name: str, check: ResidualCheck) -> str:
    if check.correlation is not None:
        follow = f"correlation {check.correlation:+.2f}"
    elif check.input_flat:
        follow = "single-colour input"
    else:
        follow = "correlation n/a"
    if check.follows_input_mean and check.dc_ratio is not None:
        follow += f", brightness x{check.dc_ratio:.2f} of the input"
    return (
        f"{ERROR_NOT_RESIDUAL}: {model_name} returns an image, not a residual "
        f"(mean |output| {check.mean_abs * 255:.1f}/255, {follow}) - pick a VAE DeGrid NAFNet"
    )


def blew_up_message(model_name: str, check: ResidualCheck) -> str:
    return (
        f"{ERROR_BLEW_UP}: mean |residual| {check.mean_abs * 255:.1f}/255 is above "
        f"{RESIDUAL_BLOWUP_ABS_MEAN * 255:.0f}/255 with {model_name} - image kept as it was"
    )


def guard(image, residual, model_name: str) -> ResidualCheck:
    """``check_residual`` that raises the matching ``DegridSkip`` subclass."""
    check = check_residual(image, residual)
    if check.looks_like_image:
        raise NotResidualModel(not_residual_message(model_name, check))
    if check.blew_up:
        raise ResidualBlewUp(blew_up_message(model_name, check))
    return check


def failure_text(exc: BaseException) -> str:
    """One-line report text: expected skips as they are, anything else with its type."""
    text = str(exc) if isinstance(exc, DegridSkip) else f"{type(exc).__name__}: {exc}"
    return " ".join(text.split())


__all__ = [
    "DEFAULT_MODE", "DEFAULT_STRENGTH", "DEFAULT_TILE", "DegridSkip", "ERROR_BLEW_UP",
    "ERROR_MODEL_NOT_FOUND", "ERROR_NOT_RESIDUAL", "IMAGE_LIKE_ABS_MEAN", "IMAGE_LIKE_CORRELATION",
    "IMAGE_LIKE_DC_MEAN", "IMAGE_LIKE_DC_RATIO", "IMAGE_LIKE_DC_SIGN", "IMAGE_LIKE_LARGE_CORRELATION",
    "IMAGE_LIKE_MIN_ABS_MEAN", "MAX_TILE", "MIN_TILE", "MODEL_EXTENSIONS", "MODES", "MODE_BRIGHT",
    "MODE_CHOICES", "MODE_DARK", "MODE_FULL", "MODE_LABELS", "NAFNET_KEYS", "NotResidualModel",
    "PAD_MULTIPLE", "RESIDUAL_BLOWUP_ABS_MEAN", "ResidualBlewUp", "ResidualCheck", "STRENGTH_MAX",
    "STRENGTH_MIN", "TILE_OVERLAP", "apply_residual", "blend_mask", "brightness_ratio",
    "call_padded", "check_residual", "coerce_strength", "coerce_tile", "failure_text", "finalize",
    "forge_quantize_in", "guard", "message_says_oom", "normalize_mode", "pad_to_multiple",
    "round_out", "select_residual", "tile_positions", "tiled_residual",
    "tiled_residual_with_oom_retry",
]
