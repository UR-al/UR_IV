"""Deterministic inputs for the VAE DeGrid origin goldens (tests/test_comfy_degrid_origin.py).

The same recipes build the inputs in the golden generator (run with the Forge
venv against the extension's own code, CPU only) and in the tests, so the
fixture stores only numbers and checksums — no extension code, no weights.

Every value here is exact in float32 (8-bit levels, dyadic fractions, 24-bit
LCG fractions) and the fake models use only element-wise ops, flips and
max/min reductions, so results do not depend on the torch build or CPU
kernels. torch is imported lazily (``torch`` is passed in or imported inside
the functions) — importing this module never loads it.
"""
from __future__ import annotations

import hashlib
import math

_MASK64 = (1 << 64) - 1


def _states(count: int, seed: int):
    state = (seed * 0x9E3779B97F4A7C15 + 0x632BE59BD9B4E019) & _MASK64
    for _ in range(count):
        state = (state * 6364136223846793005 + 1442695040888963407) & _MASK64
        yield state


def lcg_unit(count: int, seed: int) -> list[float]:
    """``count`` values in [-1, 1), 24-bit fractions (exact in float32)."""
    return [(state >> 40) / 8388608.0 - 1.0 for state in _states(count, seed)]


def lcg_bytes(count: int, seed: int) -> list[int]:
    """``count`` integers in 0..255."""
    return [state >> 56 for state in _states(count, seed)]


# ── images ───────────────────────────────────────────────────────────────
def grain_levels(height: int, width: int, seed: int) -> list[int]:
    """HWC 8-bit levels, uniform noise."""
    return lcg_bytes(height * width * 3, seed)


def smooth_levels(height: int, width: int, seed: int = 7) -> list[int]:
    """HWC 8-bit levels of a soft picture (gradients, two discs, faint noise) — integer maths only."""
    noise = lcg_bytes(height * width * 3, seed)
    levels = []
    cx1, cy1, r1 = width // 3, height // 2, min(height, width) // 4
    cx2, cy2, r2 = (2 * width) // 3, height // 3, min(height, width) // 6
    i = 0
    for y in range(height):
        for x in range(width):
            base = (
                40 + (150 * x) // max(1, width - 1),
                60 + (120 * y) // max(1, height - 1),
                90 + (80 * (x + y)) // max(1, width + height - 2),
            )
            inside1 = (x - cx1) ** 2 + (y - cy1) ** 2 <= r1 * r1
            inside2 = (x - cx2) ** 2 + (y - cy2) ** 2 <= r2 * r2
            for c in range(3):
                value = base[c]
                if inside1:
                    value = (value + (200, 170, 150)[c]) // 2
                if inside2:
                    value = (value + (30, 40, 70)[c]) // 2
                value += noise[i] % 5 - 2
                i += 1
                levels.append(max(0, min(255, value)))
    return levels


def gridded_levels(height: int, width: int, period: int = 8, lift: int = 6) -> list[int]:
    """``smooth_levels`` with a faint ``period``-pixel lattice (a stand-in for the VAE grid)."""
    base = smooth_levels(height, width)
    out = []
    i = 0
    for y in range(height):
        for x in range(width):
            on_grid = x % period == 0 or y % period == 0
            for _c in range(3):
                out.append(min(255, base[i] + lift) if on_grid else base[i])
                i += 1
    return out


def levels_to_nchw(levels: list[int], height: int, width: int, torch):
    """8-bit HWC levels -> ``[1, 3, H, W]`` float32 ``level / 255`` (Forge's PIL -> tensor)."""
    data = torch.tensor(levels, dtype=torch.float32).reshape(height, width, 3) / 255.0
    return data.permute(2, 0, 1).unsqueeze(0).contiguous()


def levels_to_image(levels: list[int], height: int, width: int, torch):
    """8-bit HWC levels -> ComfyUI IMAGE ``[1, H, W, 3]`` float32 ``level / 255``."""
    return (torch.tensor(levels, dtype=torch.float32).reshape(1, height, width, 3) / 255.0).contiguous()


def dyadic_nchw(height: int, width: int, seed: int, torch):
    """``[1, 3, H, W]`` values ``byte / 256`` (sums stay exact)."""
    data = torch.tensor(lcg_bytes(3 * height * width, seed), dtype=torch.float32) / 256.0
    return data.reshape(1, 3, height, width).contiguous()


def unit_tensor(shape, seed: int, torch, *, scale: float = 1.0, offset: float = 0.0):
    data = torch.tensor(lcg_unit(math.prod(shape), seed), dtype=torch.float32).reshape(shape)
    return (data * scale + offset).contiguous()


# ── fake models (nn.Module so runtimes can move them) ────────────────────
FAKE_KINDS = ("residual", "image_like", "blow_up")


def fake_forward(kind: str, piece, torch):
    """Deterministic per-tile maps. ``residual`` depends on the whole tile (flip, max-min)."""
    if kind == "residual":
        spread = piece.amax(dim=(-2, -1), keepdim=True) - piece.amin(dim=(-2, -1), keepdim=True)
        return (piece.flip(-1) - piece) * 0.25 + spread * 0.0625
    if kind == "image_like":
        return piece * 0.875 + 0.03125
    if kind == "blow_up":
        return (0.5 - piece) * 8.0
    raise ValueError(kind)


def fake_model(kind: str, torch, *, padder_size: int = 16, oom_above: int = 0):
    """An ``nn.Module`` running ``fake_forward``; ``oom_above`` > 0 raises an OOM-like error for larger pieces."""

    class FakeDegrid(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.anchor = torch.nn.Parameter(torch.zeros(1), requires_grad=False)
            self.padder_size = padder_size

        def forward(self, x):
            if oom_above and int(x.shape[-2]) * int(x.shape[-1]) > oom_above:
                raise RuntimeError("CUDA out of memory (fake, for the tile retry test)")
            return fake_forward(kind, x, torch)

    return FakeDegrid().eval()


# ── tiny NAFNet state dicts (spandrel key layout, LCG weights) ───────────
TINY_NAFNET = {"width": 8, "middle_blk_num": 1, "enc_blk_nums": [1, 1, 1, 1], "dec_blk_nums": [1, 1, 1, 1]}
TINY_ENDING_SCALE = {"image_like": 0.05, "blow_up": 40.0}


def tiny_nafnet_state(kind: str, torch, nafnet_class):
    """A tiny NAFNet (``nafnet_class`` = spandrel's) whose tensors are LCG-filled, as an ordered dict."""
    model = nafnet_class(img_channel=3, **TINY_NAFNET)
    state = {}
    for index, (name, tensor) in enumerate(model.state_dict().items()):
        count = tensor.numel()
        values = torch.tensor(lcg_unit(count, 1000 + index), dtype=torch.float32).reshape(tensor.shape)
        scale = 0.25
        if name.startswith("ending."):
            scale *= TINY_ENDING_SCALE[kind]
        state[name] = (values * scale).contiguous()
    return state


# ── checksums ────────────────────────────────────────────────────────────
def sha_float(tensor, torch) -> str:
    return hashlib.sha256(tensor.detach().to("cpu", torch.float32).contiguous().numpy().tobytes()).hexdigest()


def sha_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def comfy_save_bytes(image_nhwc, torch) -> bytes:
    """What ComfyUI's savers write: ``clip(255 * x, 0, 255).astype(uint8)`` (truncation)."""
    import numpy as np

    array = image_nhwc.detach().to("cpu", torch.float32).numpy()
    return np.clip(array * 255.0, 0, 255).astype(np.uint8).tobytes()


def nchw_to_hwc_bytes(image_nchw, torch) -> bytes:
    """8-bit bytes (HWC) of an already level-exact ``[1, 3, H, W]`` tensor via truncation."""
    return comfy_save_bytes(image_nchw.permute(0, 2, 3, 1), torch)


def strided_sample(tensor, stride: int = 9) -> list[float]:
    """Every ``stride``-th value of the flattened tensor (diagnostics / tolerance fallback)."""
    flat = tensor.detach().reshape(-1).tolist()
    return [float(v) for v in flat[::stride]]
