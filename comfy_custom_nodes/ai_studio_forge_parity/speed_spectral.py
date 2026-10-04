"""SPEED spectral primitives for the pack's ``ForgeNeoAnimaSpeed`` (pack 1.7.0).

SPEED (Spectral Progressive Diffusion, Xiao et al., arXiv 2605.18736) samples the early, noise-
dominated steps on a DCT-truncated coarse grid and, at a transition sigma, embeds the coarse
spectrum in the full grid, fills the new coefficients with sigma-scaled noise and rescales the
state by kappa (paper Eq. 5/6).

* ``dct2``/``idct2``: the orthonormal 2-D DCT-II over the last two axes and its inverse (what
  ``scipy.fft.dctn(x, type=2, norm="ortho")`` computes), as two matrix products with a cached
  basis — so any leading axes (4-D images, 5-D video/Anima latents) work alike.
* ``dct_downscale``: the paper's T_Phi — keep the low ``h x w`` block of the spectrum, invert on
  the small grid. ``amplitude_resize`` resizes an x0 estimate keeping its magnitude.
* ``expand_dct`` / ``expand_fft`` / ``expand_haar``: embed the coarse spectrum (or the Haar LL band)
  in the larger grid and fill the new coefficients with ``sigma * noise``. They return the expansion
  *before* the kappa rescale; the caller multiplies (the official order) and casts back.

The behaviour is sam-extra's (``forge_sam3_extension`` ``sam3ext/speed/spectral.py``): torch only
(no scipy/numpy/PyWavelets at run time), the maths in float64 on the latent's device (float32 on
MPS, which has no float64), the noise passed in by the caller (drawn per image seed), the FFT fill
``sigma * (n_r + i n_i)`` without aoleg's ``/ sqrt(2)`` (the official ca7801c9 fix) and the one-level
Haar synthesis written out with pywt's ``waverec2(..., "haar", mode="periodization")`` layout. The
code is the pack's own, written from the MIT upstreams (full notices in THIRD_PARTY_NOTICES.md,
texts in LICENSES/):

* howardhx/speed@ca7801c9bdffe681742e9592345bcf4885959be5 ``utils.py`` — ``kappa``,
  ``align_timestep``, the DCT/DWT/FFT expansions (``_dct_expand_np``, ``_dwt_expand_np``,
  ``_fft_expand_np``) and ``comfyui/speed_sampler.py`` ``_initial_dct_downscale`` — Copyright (c)
  2026 Howard Xiao;
* aoleg/ComfyUI-SPEED@a8873591a27f2c1e086a2caf546f9b6aeec62b81 ``spectral_utils.py`` /
  ``speed_core.py`` (the same expansions, 5-D latents) — Copyright (c) 2026 A. Izzuddin Al Faruq;
* sorryhyun/ComfyUI-Spectrum-KSampler@b46a364aec3b161b889c9cc26cd976a49eb537ae
  ``_vendor/networks/spd_core.py`` — the cached DCT matrix, ``_snap``, the respace alignment
  arithmetic of ``spectral_expand`` — Copyright (c) 2026 sorryhyun.

``tests/test_comfy_speed_parity.py`` checks every function bit for bit against the installed
extension on the CPU (float32/float64/bfloat16 inputs, 4-D and 5-D). No torch at module level (the
app imports the pack without it).
"""

from __future__ import annotations

import math
from collections import OrderedDict

from .compat import require_torch

__all__ = [
    "align_timestep",
    "amplitude_resize",
    "clear_basis_cache",
    "dct2",
    "dct_basis",
    "dct_downscale",
    "expand_dct",
    "expand_fft",
    "expand_haar",
    "idct2",
    "kappa",
    "respace_alignment",
    "snap",
    "work_dtype",
]

# Bases per (size, device, dtype): a pass touches a few grid sizes; the bound keeps a long session
# with many resolutions from holding stale device matrices.
_BASES: "OrderedDict[tuple, object]" = OrderedDict()
_BASES_LIMIT = 16


# ------------------------------------------------------------------------------------------------
# Scalars (official Eq. 5/6; sorryhyun's respace arithmetic and snap)
# ------------------------------------------------------------------------------------------------


def kappa(t: float, r: float) -> float:
    """State rescaling after an expansion by the grid ratio ``r`` at flow time ``t`` (Eq. 5)."""
    return r / (1.0 + (r - 1.0) * t)


def align_timestep(t: float, r: float) -> float:
    """Flow time the next segment starts from after the expansion (Eq. 6: ``t * kappa``)."""
    return t * kappa(t, r)


def respace_alignment(sigma: float, r: float) -> tuple[float, float]:
    """``(aligned sigma, kappa)`` in sorryhyun's arithmetic (``spectral_expand``).

    The same quantities as :func:`align_timestep` / :func:`kappa`, rounded the way the respace
    plan of the extension rounds them (``r * sigma`` first), so respace schedules match bit for bit.
    """
    denominator = 1.0 + (r - 1.0) * sigma
    return (r * sigma) / denominator, r / denominator


def snap(value: float, multiple: int) -> int:
    """Nearest positive multiple of ``multiple`` (sorryhyun ``_snap``; Python round-half-even)."""
    return max(multiple, int(round(value / multiple)) * multiple)


# ------------------------------------------------------------------------------------------------
# The DCT
# ------------------------------------------------------------------------------------------------


def work_dtype(device):
    """The dtype the spectral maths runs in: float64, or float32 on MPS (no float64 kernels)."""
    torch = require_torch()
    where = torch.device(device) if device is not None else torch.device("cpu")
    return torch.float32 if where.type == "mps" else torch.float64


def dct_basis(n: int, device=None, dtype=None):
    """The orthonormal DCT-II matrix ``D`` of size ``n`` (``D @ v`` transforms a column), cached.

    ``D[k, i] = sqrt(2 / n) * cos(pi * (2 i + 1) * k / (2 n))`` with row 0 divided by ``sqrt(2)``,
    always evaluated in float64 on the CPU and then moved, so every device and dtype gets the same
    rounding. The returned tensor is shared: callers must not modify it.
    """
    torch = require_torch()
    if dtype is None:
        dtype = torch.float64
    n = int(n)
    if n <= 0:
        raise ValueError(f"DCT size must be positive, got {n}")
    where = torch.device(device) if device is not None else torch.device("cpu")
    key = (n, str(where), dtype)
    basis = _BASES.get(key)
    if basis is not None:
        _BASES.move_to_end(key)
        return basis
    rows = torch.arange(n, dtype=torch.float64).unsqueeze(1)      # frequency k
    cols = torch.arange(n, dtype=torch.float64).unsqueeze(0)      # sample i
    basis = torch.cos(math.pi * (2.0 * cols + 1.0) * rows / (2.0 * n)) * math.sqrt(2.0 / n)
    basis[0] = basis[0] / math.sqrt(2.0)
    basis = basis.to(device=where, dtype=dtype)
    _BASES[key] = basis
    while len(_BASES) > _BASES_LIMIT:
        _BASES.popitem(last=False)
    return basis


def clear_basis_cache() -> None:
    _BASES.clear()


def dct2(x):
    """2-D orthonormal DCT-II over the last two axes (any leading axes), in ``x``'s dtype."""
    rows = dct_basis(x.shape[-2], x.device, x.dtype)
    cols = dct_basis(x.shape[-1], x.device, x.dtype)
    return rows @ x @ cols.transpose(0, 1)


def idct2(c):
    """Inverse of :func:`dct2` (the basis is orthonormal: its transpose inverts it)."""
    rows = dct_basis(c.shape[-2], c.device, c.dtype)
    cols = dct_basis(c.shape[-1], c.device, c.dtype)
    return rows.transpose(0, 1) @ c @ cols


def _in_work_dtype(x):
    return x.to(dtype=work_dtype(x.device))


def dct_downscale(x, size):
    """T_Phi: the low ``size`` block of the spectrum, inverted on the small grid (work dtype)."""
    h, w = int(size[0]), int(size[1])
    big_h, big_w = int(x.shape[-2]), int(x.shape[-1])
    if h > big_h or w > big_w or h <= 0 or w <= 0:
        raise ValueError(f"DCT downscale: cannot go from {(big_h, big_w)} to {(h, w)}")
    work = _in_work_dtype(x)
    if (h, w) == (big_h, big_w):
        return work
    return idct2(dct2(work)[..., :h, :w])


def amplitude_resize(x, size):
    """Resize an x0 estimate through the DCT keeping its amplitude (work dtype).

    Upscaling zero-pads the spectrum (band-limited interpolation), downscaling truncates it; the
    ``sqrt(hw / HW)`` factor undoes the orthonormal transform's change of scale. sam-extra uses it
    when a run is interrupted on the coarse grid so the host still gets a full-size latent.
    """
    torch = require_torch()
    h, w = int(size[0]), int(size[1])
    src_h, src_w = int(x.shape[-2]), int(x.shape[-1])
    work = _in_work_dtype(x)
    if (h, w) == (src_h, src_w):
        return work
    spectrum = dct2(work)
    resized = torch.zeros(*work.shape[:-2], h, w, dtype=work.dtype, device=work.device)
    keep_h, keep_w = min(h, src_h), min(w, src_w)
    resized[..., :keep_h, :keep_w] = spectrum[..., :keep_h, :keep_w]
    return idct2(resized) * math.sqrt((h * w) / (src_h * src_w))


# ------------------------------------------------------------------------------------------------
# The three expansions (no kappa yet; the caller rescales and casts)
# ------------------------------------------------------------------------------------------------


def _target(x, size, name: str) -> tuple[int, int, int, int]:
    big_h, big_w = int(size[0]), int(size[1])
    h, w = int(x.shape[-2]), int(x.shape[-1])
    if big_h < h or big_w < w:
        raise ValueError(
            f"{name} expand: cannot expand to target {(big_h, big_w)} smaller than source ({h}, {w})."
        )
    return h, w, big_h, big_w


def expand_dct(x, size, sigma: float, noise):
    """DCT expansion (official ``_dct_expand_np``, sorryhyun ``spectral_expand``).

    ``noise`` has the full target shape ``(..., H, W)``. Upstream draws the whole grid and writes
    the source spectrum over its low block; every other coefficient is ``sigma * noise``.
    """
    h, w, big_h, big_w = _target(x, size, "DCT")
    work = _in_work_dtype(x)
    if tuple(noise.shape[-2:]) != (big_h, big_w) or noise.shape[:-2] != work.shape[:-2]:
        raise ValueError(
            f"DCT expand: noise {tuple(noise.shape)} does not fit {tuple(work.shape[:-2])}+{(big_h, big_w)}"
        )
    spectrum = noise.to(device=work.device, dtype=work.dtype) * float(sigma)
    spectrum[..., :h, :w] = dct2(work)
    return idct2(spectrum)


def expand_fft(x, size, sigma: float, noise_real, noise_imag):
    """FFT expansion (official ``_fft_expand_np`` after ca7801c9).

    The centred spectrum ``fftshift(fft2(x, ortho))`` replaces the centre of a full grid of
    ``sigma * (n_r + i n_i)`` (fftshifted like upstream); the real part of the inverse is returned.
    """
    torch = require_torch()
    h, w, big_h, big_w = _target(x, size, "FFT")
    work = _in_work_dtype(x)
    top, left = (big_h - h) // 2, (big_w - w) // 2
    axes = (-2, -1)
    real = noise_real.to(device=work.device, dtype=work.dtype)
    imag = noise_imag.to(device=work.device, dtype=work.dtype)
    centred = torch.fft.fftshift(torch.fft.fft2(work, dim=axes, norm="ortho"), dim=axes)
    spectrum = torch.fft.fftshift(torch.complex(real, imag) * float(sigma), dim=axes)
    spectrum[..., top:top + h, left:left + w] = centred
    return torch.fft.ifft2(torch.fft.ifftshift(spectrum, dim=axes), dim=axes, norm="ortho").real


def expand_haar(x, sigma: float, lh, hl, hh):
    """One-level Haar synthesis: ``x`` is the LL band, ``sigma``-scaled noise the LH/HL/HH bands.

    The output grid is exactly twice the source grid; each 2x2 output block is
    ``(a +- h +- v +- d) / 2`` in pywt's periodization layout (official ``_dwt_expand_np``).
    """
    torch = require_torch()
    work = _in_work_dtype(x)
    scale = float(sigma)
    a = work
    h = lh.to(device=work.device, dtype=work.dtype) * scale
    v = hl.to(device=work.device, dtype=work.dtype) * scale
    d = hh.to(device=work.device, dtype=work.dtype) * scale
    rows, cols = int(work.shape[-2]), int(work.shape[-1])
    out = torch.empty(*work.shape[:-2], 2 * rows, 2 * cols, dtype=work.dtype, device=work.device)
    out[..., 0::2, 0::2] = (a + h + v + d) * 0.5
    out[..., 0::2, 1::2] = (a + h - v - d) * 0.5
    out[..., 1::2, 0::2] = (a - h + v - d) * 0.5
    out[..., 1::2, 1::2] = (a - h - v + d) * 0.5
    return out
