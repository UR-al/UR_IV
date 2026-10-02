"""Anima detail stages after the perturbation term: HiFlow (hires) -> Momentum Guidance -> HiGS -> TSR.

This mirrors sam-extra v0.30.0 (the user's ``forge_sam3_extension``: ``scripts/anima_safe_pag.py``
``_apply_detail_stages``/``_hiflow_*`` and ``sam3ext/guidance/{tsr,history,hiflow,trajectory,sigmas}.py``):
the same equations, defaults, clamps and host rules, written again for ComfyUI's post-CFG hook.
``tests/test_comfy_detail_parity.py`` runs both on the same tensors on CPU.

- **TSR** -- Temporal Score Rescaling (Xu et al., arXiv 2510.01184; ComfyUI ``nodes_eps.py``
  ``TemporalScoreRescaling`` is the behavioural reference): ``snr = exp(2*hls(sigma))``,
  ``r = (snr*v + 1) / (snr*v/k + 1)`` with ``v = tsr_sigma**2``, ``x0' = lerp(x/alpha, x0, r)``,
  ``alpha = sigma*exp(hls)``. ``k == 1`` is an exact no-op. Uses the sigma the model saw.
- **Momentum Guidance** (arXiv 2602.20360): ``D + alpha*sigma*(v - m)`` with ``v = (D - x)/sigma``
  and the EMA ``m <- (1-beta)*v + beta*m`` of the unmodified velocity; optional per-sample norm
  matching (paper section 8.2). The window is the noise level (flow sigma, sigma/(1+sigma) else).
- **HiGS** (arXiv 2509.22300): ``D + w(t)*iDCT(H*DCT(dD(eta)))`` with ``dD = D - g``, the
  zero-initialised EMA ``g`` of the guided prediction, the parallel part scaled by ``eta`` and a
  sigmoid high-pass ``H`` (sharpness 50) on the orthonormal 2-D DCT.
- **HiFlow** (Bu et al., arXiv 2504.06232; Bujiazi/HiFlow @ 31cc2b1, Apache-2.0 -- see
  THIRD_PARTY_NOTICES.md): on the hires pass, direction alignment ``X + a*LPF(R - X)`` with the
  order-4 Butterworth low-pass on the centred FFT and acceleration alignment
  ``O_k = D_k - b*[(D_k - R_k) - (sigma_k/sigma_{k-1})*(O_{k-1} - R_{k-1})]``, both weighted by
  ``(N-k)/N``. ``R`` is the base pass's final x0 at the same sigma (linear in sigma between records,
  bicubic-upsampled in latent space).

Host rules shared with the extension:

- MG/HiGS/HiFlow and adaptive SMC read the sampler's own sigma -- the one Detail Daemon noted
  before scaling the model call (``guidance_common.PRE_DD_SIGMAS_KEY``); TSR reads the model's.
- MG/HiGS update their history once per sampler step: an evaluation off the schedule (a
  second-order midpoint) neither applies nor updates, a repeat of the last sigma applies without
  updating, a rising sigma starts a new run. An Adaptive Guidance cond-only step resets the history.
- HiFlow aligns only on the run tagged ``hires`` (``PASS_KEY``, set by ``ForgeNeoHiresFix``) and
  records on every other run -- ``ForgeNeoKSamplerCNS`` tags ``base``, a custom workflow's own sampler
  carries no tag and is the base pass too (S² reads an untagged run the same way). A detailer run after
  the hires pass records into a trajectory nothing reads; the next generation's base run clears it.
  The app compiler turns HiFlow on only for a txt2img Hires.fix generation (the extension's rule).

Host differences: a hires pass that loads another checkpoint gets a fresh MODEL without this suite
(``ForgeNeoHiresFix``), so HiFlow cannot align there; sam-extra keeps aligning when the latent
shape still matches.
"""

from __future__ import annotations

import bisect
import logging
import math
from dataclasses import dataclass, field
from typing import Any

from .compat import clone_model, require_torch
from .guidance_common import PRE_DD_SIGMAS_KEY, _as_bool, _setting

LOGGER = logging.getLogger("ai_studio_forge_parity")

# The pass a sampling run belongs to, set on the MODEL's transformer_options by the sampler nodes —
# only on a MODEL whose model_options carry PASS_AWARE_KEY (the suite sets it when S² or HiFlow is on).
PASS_KEY = "forge_neo_pass"
PASS_AWARE_KEY = "forge_neo_pass_aware"
PASS_BASE = "base"
PASS_HIRES = "hires"

# torch.isclose rule the extension uses to find a sigma on the schedule (sam3ext/guidance/dave_gate.py).
ISCLOSE_RTOL = 1e-4
ISCLOSE_ATOL = 1e-6
_SIGMA_RTOL = 1e-4          # history/HiFlow "same sigma" (Heun's corrector reused as the predictor)
_EPS = 1e-12
HIGS_SHARPNESS = 50.0       # HiGS Appendix F / Algorithm 3
BUTTERWORTH_ORDER = 4       # Bujiazi/HiFlow utils.py butterworth_low_pass_filter n=4

ROLE_NEW = "new"            # apply, then update the history
ROLE_REPEAT = "repeat"      # apply with the current history, do not update it
ROLE_SKIP = "skip"          # leave the prediction and the history alone

# Defaults (sam-extra v0.30.0; the app sends its own values, these only fill missing keys).
TSR_DEFAULTS = {"k": 0.95, "sigma": 1.0}
MG_DEFAULTS = {"alpha": 0.5, "beta": 0.6, "normalize": False, "min": 0.30, "max": 0.95}
HIGS_DEFAULTS = {"weight": 1.75, "eta": 0.0, "alpha": 0.75, "cutoff": 0.05, "t_min": 0.40, "t_max": 1.00}
HIFLOW_DEFAULTS = {"alpha": 1.0, "beta": 0.5, "cutoff": 0.2}


# ---------------------------------------------------------------------------
# Sigma helpers
# ---------------------------------------------------------------------------

def first_float(value: Any) -> float | None:
    """First element of a tensor sigma (or the number itself) as a Python float, else None."""
    if value is None:
        return None
    try:
        if hasattr(value, "flatten") and hasattr(value, "shape"):
            if value.numel() == 0:
                return None
            return float(value.flatten()[0].item())
        return float(value)
    except (TypeError, ValueError, RuntimeError):
        return None


def transformer_options(args: dict[str, Any]) -> dict[str, Any]:
    options = args.get("model_options") if isinstance(args, dict) else None
    transformer = (options or {}).get("transformer_options") if isinstance(options, dict) else None
    return transformer if isinstance(transformer, dict) else {}


def sampler_sigma(args: dict[str, Any]) -> float | None:
    """The sampler's own sigma for this call: Detail Daemon's note, else the model's sigma."""
    noted = transformer_options(args).get(PRE_DD_SIGMAS_KEY)
    value = first_float(noted)
    if value is not None:
        return value
    return first_float(args.get("sigma") if isinstance(args, dict) else None)


def sampling_schedule(args: dict[str, Any]):
    """The run's ``sample_sigmas`` (Comfy publishes them in transformer_options), or None."""
    return transformer_options(args).get("sample_sigmas")


def pass_tag(args: dict[str, Any]) -> str | None:
    tag = transformer_options(args).get(PASS_KEY)
    return str(tag) if tag else None


def _schedule_values(schedule) -> list[float] | None:
    if schedule is None:
        return None
    try:
        values = schedule.flatten().tolist() if hasattr(schedule, "flatten") else list(schedule)
        return [float(value) for value in values]
    except (TypeError, ValueError, RuntimeError):
        return None


def schedule_index(schedule, sigma: float | None) -> int | None:
    """Index of the first schedule entry ``isclose`` to ``sigma``, else None (also without a schedule)."""
    values = schedule if isinstance(schedule, list) else _schedule_values(schedule)
    if values is None or sigma is None:
        return None
    for index, value in enumerate(values):
        if abs(value - sigma) <= ISCLOSE_ATOL + ISCLOSE_RTOL * abs(sigma):
            return index
    return None


def is_flow(args: dict[str, Any]) -> bool | None:
    """True for a rectified-flow (CONST) model, False for eps/v, None when unknown."""
    sampling = getattr(args.get("model") if isinstance(args, dict) else None, "model_sampling", None)
    if sampling is None:
        return None
    names = {cls.__name__ for cls in type(sampling).__mro__}
    if "CONST" in names:
        return True
    if names & {"EPS", "V_PREDICTION", "EDM", "V_PREDICTION_EDM"}:
        return False
    return None


def half_log_snr(sigma: float, flow: bool) -> float:
    """``log(alpha/sigma)``: flow ``log((1-s)/s)``, eps/v k-diffusion ``-log s``."""
    sigma = float(sigma)
    if flow:
        return math.log((1.0 - sigma) / sigma)
    return -math.log(sigma)


def _finite_clamp(value: Any, low: float, high: float, default: float) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        number = float(default)
    if not math.isfinite(number):
        number = float(default)
    return min(high, max(low, number))


# ---------------------------------------------------------------------------
# TSR
# ---------------------------------------------------------------------------

def tsr_factors(sigma: float, k: float, tsr_sigma: float, flow: bool) -> tuple[float, float]:
    """``(r, alpha)`` for one sigma -- ``(1, 1)`` where TSR leaves the row alone."""
    sigma = float(sigma)
    if k == 1.0 or not math.isfinite(sigma) or sigma <= 0.0:
        return 1.0, 1.0
    if flow and sigma >= 1.0:
        return 1.0, 1.0
    hls = half_log_snr(sigma, flow)
    snr = math.exp(2.0 * hls)
    variance = float(tsr_sigma) ** 2
    if snr == 0.0:
        return 1.0, 1.0
    if math.isinf(snr):
        return float(k), 1.0
    return (snr * variance + 1.0) / (snr * variance / float(k) + 1.0), sigma * math.exp(hls)


def apply_tsr(denoised: Any, x: Any, sigma: Any, *, k: float, tsr_sigma: float, flow: bool) -> Any:
    """TSR on a ``[B, ...]`` x0 prediction; ``sigma`` is a scalar or one value per row."""
    torch = require_torch()
    if float(k) == 1.0:
        return denoised
    if denoised.shape != x.shape:
        raise ValueError(f"TSR needs matching tensors, got {tuple(denoised.shape)} and {tuple(x.shape)}")
    batch = denoised.shape[0]
    values = sigma.detach().float().flatten().cpu().tolist() if torch.is_tensor(sigma) else [float(sigma)]
    if len(values) == 1:
        values = values * batch
    if len(values) != batch:
        raise ValueError(f"TSR got {len(values)} sigmas for a batch of {batch}")
    factors = [tsr_factors(value, k, tsr_sigma, flow) for value in values]
    if all(r == 1.0 for r, _alpha in factors):
        return denoised
    view = (batch,) + (1,) * (denoised.ndim - 1)
    r = torch.tensor([f[0] for f in factors], dtype=torch.float32, device=denoised.device).view(view)
    alpha = torch.tensor([f[1] for f in factors], dtype=torch.float32, device=denoised.device).view(view)
    clean = denoised.float()
    scaled_input = x.to(device=clean.device, dtype=torch.float32) / alpha
    return torch.lerp(scaled_input, clean, r).to(denoised.dtype)


# ---------------------------------------------------------------------------
# Momentum Guidance and HiGS (history)
# ---------------------------------------------------------------------------

@dataclass
class HistoryState:
    """One run's MG velocity EMA and HiGS prediction EMA (each None until seeded)."""

    mg_m: Any = None
    higs_g: Any = None
    last_sigma: float | None = None
    counters: dict = field(default_factory=lambda: {"mg": 0, "higs": 0})

    def reset(self) -> None:
        self.mg_m = None
        self.higs_g = None
        self.last_sigma = None


def noise_level(sigma: float, flow: bool) -> float:
    """t in [0, 1] with 1 = noise: sigma on a flow model, sigma/(1+sigma) on an eps/v model."""
    sigma = float(sigma)
    return sigma if flow else sigma / (1.0 + sigma)


def step_role(state: HistoryState, sigma: float | None, on_schedule: bool | None) -> str:
    """How this evaluation treats the history; may reset ``state`` (a rising sigma = a new run)."""
    if sigma is None or not math.isfinite(float(sigma)) or float(sigma) <= 0.0:
        return ROLE_SKIP
    if on_schedule is False:
        return ROLE_SKIP
    sigma = float(sigma)
    last = state.last_sigma
    if last is not None:
        if abs(sigma - last) <= _SIGMA_RTOL * max(abs(last), 1e-12):
            return ROLE_REPEAT
        if sigma > last:
            state.reset()
    state.last_sigma = sigma
    return ROLE_NEW


def _per_sample_norm(tensor: Any) -> Any:
    torch = require_torch()
    dims = tuple(range(1, tensor.ndim))
    return torch.linalg.vector_norm(tensor, dim=dims, keepdim=True)


def apply_mg(denoised: Any, x: Any, sigma: float, state: HistoryState, *, alpha: float, beta: float,
             normalize: bool, active: bool, role: str) -> Any:
    """``D + alpha*sigma*(v - m)``; the first evaluation seeds ``m = v`` and returns ``denoised``."""
    torch = require_torch()
    if role == ROLE_SKIP:
        return denoised
    sigma = float(sigma)
    clean = denoised.float()
    velocity = (clean - x.to(device=clean.device, dtype=torch.float32)) / sigma
    m = state.mg_m
    if not torch.is_tensor(m) or m.shape != velocity.shape:
        if role == ROLE_NEW:
            state.mg_m = velocity.detach()
        return denoised
    m = m.to(device=velocity.device, dtype=velocity.dtype)
    out = denoised
    if active and float(alpha) != 0.0:
        used = m
        if normalize:
            used = m * (_per_sample_norm(velocity) / (_per_sample_norm(m) + 1e-8))
        out = (clean + float(alpha) * sigma * (velocity - used)).to(denoised.dtype)
        state.counters["mg"] += 1
    if role == ROLE_NEW:
        state.mg_m = ((1.0 - float(beta)) * velocity + float(beta) * m).detach()
    return out


_DCT_CACHE: dict = {}


def _dct_matrix(n: int, device: Any, dtype: Any) -> Any:
    """Orthonormal DCT-II matrix ``C`` (``C @ x`` transforms a column vector)."""
    torch = require_torch()
    key = (int(n), str(device), dtype)
    cached = _DCT_CACHE.get(key)
    if cached is not None:
        return cached
    k = torch.arange(n, dtype=torch.float64).unsqueeze(1)
    i = torch.arange(n, dtype=torch.float64).unsqueeze(0)
    matrix = torch.cos(math.pi * (2.0 * i + 1.0) * k / (2.0 * n)) * math.sqrt(2.0 / n)
    matrix[0, :] = matrix[0, :] / math.sqrt(2.0)
    matrix = matrix.to(device=device, dtype=dtype)
    if len(_DCT_CACHE) > 16:
        _DCT_CACHE.clear()
    _DCT_CACHE[key] = matrix
    return matrix


def dct_highpass(tensor: Any, cutoff: float, sharpness: float = HIGS_SHARPNESS) -> Any:
    """``iDCT(H*DCT(t))`` over the last two axes, ``H = sigmoid(sharpness*(R - cutoff))``."""
    torch = require_torch()
    height, width = int(tensor.shape[-2]), int(tensor.shape[-1])
    work = tensor.float()
    c_h = _dct_matrix(height, work.device, work.dtype)
    c_w = _dct_matrix(width, work.device, work.dtype)
    spectrum = c_h @ work @ c_w.transpose(0, 1)
    u = torch.arange(height, device=work.device, dtype=work.dtype) / height
    v = torch.arange(width, device=work.device, dtype=work.dtype) / width
    radius = torch.sqrt(u.unsqueeze(1) ** 2 + v.unsqueeze(0) ** 2)
    mask = torch.sigmoid(float(sharpness) * (radius - float(cutoff)))
    return c_h.transpose(0, 1) @ (spectrum * mask) @ c_w


def higs_weight(t: float, weight: float, t_min: float, t_max: float) -> float:
    """``w(t) = w*sqrt((t - t_min)/(t_max - t_min))`` for ``t_min < t <= t_max``, else 0."""
    if not (t_min < t <= t_max) or t_max <= t_min:
        return 0.0
    return float(weight) * math.sqrt((t - t_min) / (t_max - t_min))


def apply_higs(denoised: Any, sigma: float, state: HistoryState, *, weight: float, eta: float, alpha: float,
               cutoff: float, t_min: float, t_max: float, flow: bool, role: str) -> Any:
    """HiGS on one evaluation; seeds/updates ``state.higs_g`` on ``ROLE_NEW``."""
    torch = require_torch()
    if role == ROLE_SKIP:
        return denoised
    clean = denoised.float()
    g = state.higs_g
    if not torch.is_tensor(g) or g.shape != clean.shape:
        if role == ROLE_NEW:
            state.higs_g = (float(alpha) * clean).detach()
        return denoised
    g = g.to(device=clean.device, dtype=clean.dtype)
    out = denoised
    w = higs_weight(noise_level(sigma, flow), weight, t_min, t_max)
    if w != 0.0:
        diff = (clean - g).double()
        reference = clean.double()
        dims = tuple(range(1, diff.ndim))
        dot = (diff * reference).sum(dim=dims, keepdim=True)
        norm = (reference * reference).sum(dim=dims, keepdim=True).clamp_min(_EPS)
        parallel = dot / norm * reference
        diff = (diff - parallel + float(eta) * parallel).float()
        out = (clean + w * dct_highpass(diff, cutoff)).to(denoised.dtype)
        state.counters["higs"] += 1
    if role == ROLE_NEW:
        state.higs_g = (float(alpha) * clean + (1.0 - float(alpha)) * g).detach()
    return out


# ---------------------------------------------------------------------------
# HiFlow
# ---------------------------------------------------------------------------

class Trajectory:
    """Sigma-keyed x0 records of one pass (CPU, float16). Re-recording a sigma replaces it."""

    def __init__(self, storage_dtype: Any = None):
        self._sigmas: list[float] = []   # ascending
        self._records: list[Any] = []
        self.shape: tuple | None = None
        self.storage_dtype = storage_dtype

    def __len__(self) -> int:
        return len(self._sigmas)

    def clear(self) -> None:
        self._sigmas.clear()
        self._records.clear()
        self.shape = None

    @property
    def sigmas(self) -> tuple[float, ...]:
        return tuple(self._sigmas)

    def record(self, sigma: float, x0: Any) -> None:
        torch = require_torch()
        sigma = float(sigma)
        shape = tuple(x0.shape)
        if self.shape is not None and shape != self.shape:
            self.clear()
        self.shape = shape
        dtype = self.storage_dtype if self.storage_dtype is not None else torch.float16
        stored = x0.detach().to(device="cpu", dtype=dtype).clone()
        index = bisect.bisect_left(self._sigmas, sigma)
        if index < len(self._sigmas) and abs(self._sigmas[index] - sigma) <= 1e-9 * max(1.0, abs(sigma)):
            self._records[index] = stored
            return
        self._sigmas.insert(index, sigma)
        self._records.insert(index, stored)

    def at(self, sigma: float, *, device: Any = None) -> Any:
        """x0 at ``sigma``: linear in sigma between neighbours, clamped at the ends. None if empty."""
        torch = require_torch()
        if not self._sigmas:
            return None
        sigma = float(sigma)
        sigmas = self._sigmas
        if sigma <= sigmas[0]:
            out = self._records[0].to(dtype=torch.float32)
        elif sigma >= sigmas[-1]:
            out = self._records[-1].to(dtype=torch.float32)
        else:
            hi = bisect.bisect_left(sigmas, sigma)
            lo = hi - 1
            s0, s1 = sigmas[lo], sigmas[hi]
            weight = 0.0 if s1 == s0 else (sigma - s0) / (s1 - s0)
            out = torch.lerp(self._records[lo].to(dtype=torch.float32),
                             self._records[hi].to(dtype=torch.float32), weight)
        return out.to(device=device) if device is not None else out


@dataclass
class HiFlowState:
    """The previous hires step's ``(sigma, O, R)`` for acceleration alignment."""

    prev_sigma: float | None = None
    prev_out: Any = None
    prev_ref: Any = None
    counters: dict = field(default_factory=lambda: {"direction": 0, "acceleration": 0})

    def reset(self) -> None:
        self.prev_sigma = None
        self.prev_out = None
        self.prev_ref = None


_MASK_CACHE: dict = {}


def butterworth_mask(height: int, width: int, cutoff: float, *, order: int = BUTTERWORTH_ORDER,
                     device: Any = None, dtype: Any = None) -> Any:
    """Centred Butterworth low-pass mask ``1/(1 + (d^2/D^2)^n)``."""
    torch = require_torch()
    dtype = torch.float32 if dtype is None else dtype
    key = (int(height), int(width), float(cutoff), int(order), str(device), dtype)
    cached = _MASK_CACHE.get(key)
    if cached is not None:
        return cached
    h = torch.arange(height, dtype=torch.float64)
    w = torch.arange(width, dtype=torch.float64)
    d_square = (2.0 * h / height - 1.0).unsqueeze(1) ** 2 + (2.0 * w / width - 1.0).unsqueeze(0) ** 2
    cutoff = max(float(cutoff), 1e-6)
    mask = (1.0 / (1.0 + (d_square / cutoff ** 2) ** int(order))).to(device=device, dtype=dtype)
    if len(_MASK_CACHE) > 16:
        _MASK_CACHE.clear()
    _MASK_CACHE[key] = mask
    return mask


def butterworth_lowpass(tensor: Any, cutoff: float, *, order: int = BUTTERWORTH_ORDER) -> Any:
    """Low band of ``tensor`` over its last two axes (fp32 FFT)."""
    torch = require_torch()
    work = tensor.float()
    mask = butterworth_mask(work.shape[-2], work.shape[-1], cutoff, order=order, device=work.device)
    spectrum = torch.fft.fftshift(torch.fft.fft2(work, dim=(-2, -1)), dim=(-2, -1))
    return torch.fft.ifft2(torch.fft.ifftshift(spectrum * mask, dim=(-2, -1)), dim=(-2, -1)).real


def resize_latent(tensor: Any, size: tuple[int, int]) -> Any:
    """Bicubic latent upsampling (``align_corners=False``) of a 4-D or 5-D latent to ``size`` (H, W)."""
    torch = require_torch()
    height, width = int(size[0]), int(size[1])
    work = tensor.float()
    if tuple(work.shape[-2:]) == (height, width):
        return work
    interpolate = torch.nn.functional.interpolate
    if work.ndim == 4:
        return interpolate(work, size=(height, width), mode="bicubic", align_corners=False)
    if work.ndim == 5:
        b, c, frames, h, w = work.shape
        out = interpolate(work.reshape(b, c * frames, h, w), size=(height, width), mode="bicubic",
                          align_corners=False)
        return out.reshape(b, c, frames, height, width)
    raise ValueError(f"HiFlow expects a 4-D or 5-D latent, got {tuple(work.shape)}")


def step_weight(position: float | None, steps: int | None) -> float:
    """``(N - k)/N`` for (fractional) hires step ``k`` of ``N`` -- 1.0 at the first step and when unknown."""
    if position is None or not steps or steps <= 0:
        return 1.0
    return max(0.0, min(1.0, (float(steps) - float(position)) / float(steps)))


def apply_hiflow(denoised: Any, sigma: float, reference: Any, state: HiFlowState, *, alpha: float, beta: float,
                 cutoff: float, weight: float, step_start: bool) -> Any:
    """Direction alignment, then acceleration alignment on step starts (a midpoint gets direction only)."""
    torch = require_torch()
    clean = denoised.float()
    ref = reference.to(device=clean.device, dtype=torch.float32)
    a = float(alpha) * float(weight)
    out = clean
    if a != 0.0:
        out = clean + a * butterworth_lowpass(ref - clean, cutoff)
        state.counters["direction"] += 1
    if not step_start:
        return out.to(denoised.dtype)
    b = float(beta) * float(weight)
    if b != 0.0 and torch.is_tensor(state.prev_out) and state.prev_out.shape == out.shape and state.prev_sigma:
        prev_out = state.prev_out.to(device=out.device, dtype=torch.float32)
        prev_ref = state.prev_ref.to(device=out.device, dtype=torch.float32)
        ratio = float(sigma) / float(state.prev_sigma)
        out = out - b * ((out - ref) - ratio * (prev_out - prev_ref))
        state.counters["acceleration"] += 1
    state.prev_sigma = float(sigma)
    state.prev_out = out.detach()
    state.prev_ref = ref.detach()
    return out.to(denoised.dtype)


def hiflow_position(walked: list[float] | None, sigma: float) -> tuple[float | None, int | None, bool | None]:
    """``(fractional step k, N, on_schedule)`` of ``sigma`` on the hires pass's walked sigmas."""
    if not walked or len(walked) < 2:
        return None, None, None
    steps = len(walked) - 1
    index = schedule_index(walked, sigma)
    if index is not None:
        return float(index), steps, True
    for k in range(steps):
        hi, lo = walked[k], walked[k + 1]
        if hi >= sigma >= lo and hi != lo:
            return k + (hi - sigma) / (hi - lo), steps, False
    return None, steps, False


# ---------------------------------------------------------------------------
# Settings and the suite's post-CFG functions
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class DetailSettings:
    tsr: bool = False
    tsr_k: float = TSR_DEFAULTS["k"]
    tsr_sigma: float = TSR_DEFAULTS["sigma"]
    mg: bool = False
    mg_alpha: float = MG_DEFAULTS["alpha"]
    mg_beta: float = MG_DEFAULTS["beta"]
    mg_normalize: bool = MG_DEFAULTS["normalize"]
    mg_min: float = MG_DEFAULTS["min"]
    mg_max: float = MG_DEFAULTS["max"]
    higs: bool = False
    higs_weight: float = HIGS_DEFAULTS["weight"]
    higs_eta: float = HIGS_DEFAULTS["eta"]
    higs_alpha: float = HIGS_DEFAULTS["alpha"]
    higs_cutoff: float = HIGS_DEFAULTS["cutoff"]
    higs_t_min: float = HIGS_DEFAULTS["t_min"]
    higs_t_max: float = HIGS_DEFAULTS["t_max"]
    hiflow: bool = False
    hiflow_alpha: float = HIFLOW_DEFAULTS["alpha"]
    hiflow_beta: float = HIFLOW_DEFAULTS["beta"]
    hiflow_cutoff: float = HIFLOW_DEFAULTS["cutoff"]

    @property
    def any(self) -> bool:
        return self.tsr or self.mg or self.higs or self.hiflow


def detail_settings(settings: dict[str, Any]) -> DetailSettings:
    """The suite's ``guid_tsr_*``/``guid_mg_*``/``guid_higs_*``/``guid_hiflow_*`` keys, read and clamped like the
    extension (TSR is off at k 1, HiGS at weight 0; reversed windows are swapped)."""

    def num(key: str, low: float, high: float, default: float) -> float:
        return _finite_clamp(_setting(settings, key, default), low, high, default)

    tsr_k = num("guid_tsr_k", 0.01, 100.0, TSR_DEFAULTS["k"])
    tsr_sigma = num("guid_tsr_sigma", 0.01, 100.0, TSR_DEFAULTS["sigma"])
    mg_min = num("guid_mg_min", 0.0, 1.0, MG_DEFAULTS["min"])
    mg_max = num("guid_mg_max", 0.0, 1.0, MG_DEFAULTS["max"])
    higs_weight_value = num("guid_higs_weight", 0.0, 3.0, HIGS_DEFAULTS["weight"])
    t_min = num("guid_higs_t_min", 0.0, 1.0, HIGS_DEFAULTS["t_min"])
    t_max = num("guid_higs_t_max", 0.0, 1.0, HIGS_DEFAULTS["t_max"])
    return DetailSettings(
        tsr=_as_bool(_setting(settings, "guid_tsr_enabled", False)) and tsr_k != 1.0,
        tsr_k=tsr_k, tsr_sigma=tsr_sigma,
        mg=_as_bool(_setting(settings, "guid_mg_enabled", False)),
        mg_alpha=num("guid_mg_alpha", 0.0, 3.0, MG_DEFAULTS["alpha"]),
        mg_beta=num("guid_mg_beta", 0.0, 0.99, MG_DEFAULTS["beta"]),
        mg_normalize=_as_bool(_setting(settings, "guid_mg_normalize", False)),
        mg_min=min(mg_min, mg_max), mg_max=max(mg_min, mg_max),
        higs=_as_bool(_setting(settings, "guid_higs_enabled", False)) and higs_weight_value > 0.0,
        higs_weight=higs_weight_value,
        higs_eta=num("guid_higs_eta", 0.0, 1.0, HIGS_DEFAULTS["eta"]),
        higs_alpha=num("guid_higs_alpha", 0.01, 0.99, HIGS_DEFAULTS["alpha"]),
        higs_cutoff=num("guid_higs_cutoff", 0.0, 0.5, HIGS_DEFAULTS["cutoff"]),
        higs_t_min=min(t_min, t_max), higs_t_max=max(t_min, t_max),
        hiflow=_as_bool(_setting(settings, "guid_hiflow_enabled", False)),
        hiflow_alpha=num("guid_hiflow_alpha", 0.0, 2.0, HIFLOW_DEFAULTS["alpha"]),
        hiflow_beta=num("guid_hiflow_beta", 0.0, 1.0, HIFLOW_DEFAULTS["beta"]),
        hiflow_cutoff=num("guid_hiflow_cutoff", 0.05, 1.0, HIFLOW_DEFAULTS["cutoff"]),
    )


class DetailRun:
    """State the suite's detail functions share across the calls of one MODEL (history, HiFlow)."""

    def __init__(self, settings: DetailSettings):
        self.settings = settings
        self.history = HistoryState()
        self.trajectory = Trajectory()
        self.hiflow = HiFlowState()
        self.hiflow_last_sigma: float | None = None   # the hires run's last evaluation (new-run detection)
        self.record_last_sigma: float | None = None   # the base run's last evaluation
        self.tsr_steps = 0
        self.warned: set[str] = set()

    def warn_once(self, stage: str, message: str) -> None:
        if stage not in self.warned:
            self.warned.add(stage)
            LOGGER.warning(message)


def _new_run(last: float | None, sigma: float) -> bool:
    return last is None or sigma > last + 1e-6


def _hiflow_apply(run: DetailRun, args: dict[str, Any], result: Any, sigma: float) -> Any:
    settings = run.settings
    if _new_run(run.hiflow_last_sigma, sigma):
        run.hiflow.reset()
    run.hiflow_last_sigma = sigma
    lr = run.trajectory.at(sigma, device=result.device)
    if lr is None:
        return result
    if lr.ndim != result.ndim or tuple(lr.shape[:2]) != tuple(result.shape[:2]):
        run.warn_once(
            "hiflow_shape",
            "Anima HiFlow skipped: the base-pass trajectory "
            f"{tuple(lr.shape)} does not match the hires latent {tuple(result.shape)}.",
        )
        return result
    reference = resize_latent(lr, tuple(result.shape[-2:]))
    walked = _schedule_values(sampling_schedule(args))
    position, steps, on_schedule = hiflow_position(walked, sigma)
    state = run.hiflow
    step_start = on_schedule is not False and not (
        state.prev_sigma is not None
        and abs(float(state.prev_sigma) - sigma) <= _SIGMA_RTOL * max(abs(sigma), 1e-12)
    )
    return apply_hiflow(
        result, sigma, reference, state,
        alpha=settings.hiflow_alpha, beta=settings.hiflow_beta, cutoff=settings.hiflow_cutoff,
        weight=step_weight(position, steps), step_start=step_start,
    )


def apply_detail_stages(run: DetailRun, args: dict[str, Any], result: Any, *, adg_skipped: bool = False) -> Any:
    """HiFlow (hires run) -> Momentum Guidance -> HiGS -> TSR on the guided x0 (the extension's order)."""
    torch = require_torch()
    settings = run.settings
    x = args.get("input")
    if not torch.is_tensor(x) or tuple(x.shape) != tuple(result.shape):
        run.warn_once("input", "Anima detail stages skipped: post-CFG input does not match the prediction.")
        return result
    sigma = sampler_sigma(args)
    if sigma is None:
        run.warn_once("sigma", "Anima detail stages skipped: the post-CFG call carries no sigma.")
        return result
    flow = is_flow(args)

    if settings.hiflow and pass_tag(args) == PASS_HIRES and sigma > 0.0:
        try:
            result = _hiflow_apply(run, args, result, sigma)
        except Exception as exc:   # noqa: BLE001 - one stage must not lose the earlier guidance
            run.warn_once("hiflow", f"Anima HiFlow fallback (earlier guidance kept): {type(exc).__name__}: {exc}")

    if settings.mg or settings.higs:
        history = run.history
        if adg_skipped:
            history.reset()
        else:
            try:
                schedule = sampling_schedule(args)
                on_schedule = None if schedule is None else schedule_index(schedule, sigma) is not None
                role = step_role(history, sigma, on_schedule)
                if settings.mg:
                    level = noise_level(sigma, bool(flow)) if sigma else 0.0
                    result = apply_mg(
                        result, x, sigma, history,
                        alpha=settings.mg_alpha, beta=settings.mg_beta, normalize=settings.mg_normalize,
                        active=settings.mg_min <= level <= settings.mg_max, role=role,
                    )
                if settings.higs:
                    result = apply_higs(
                        result, sigma, history,
                        weight=settings.higs_weight, eta=settings.higs_eta, alpha=settings.higs_alpha,
                        cutoff=settings.higs_cutoff, t_min=settings.higs_t_min, t_max=settings.higs_t_max,
                        flow=bool(flow), role=role,
                    )
            except Exception as exc:   # noqa: BLE001
                history.reset()
                run.warn_once("history", f"Anima MG/HiGS fallback (earlier guidance kept): {type(exc).__name__}: {exc}")

    if settings.tsr:
        if flow is None:
            run.warn_once("tsr", "Anima TSR skipped: the model's parameterisation (flow or eps/v) is unknown.")
        else:
            try:
                rescaled = apply_tsr(result, x, args.get("sigma"), k=settings.tsr_k,
                                     tsr_sigma=settings.tsr_sigma, flow=bool(flow))
                if rescaled is not result:
                    run.tsr_steps += 1
                result = rescaled
            except Exception as exc:   # noqa: BLE001
                run.warn_once("tsr", f"Anima TSR fallback (earlier guidance kept): {type(exc).__name__}: {exc}")
    return result


def record_hiflow(run: DetailRun, args: dict[str, Any], result: Any) -> None:
    """Base run of a Hires.fix generation: keep this evaluation's final x0 at the sampler's sigma (last wins).

    Every run but the hires one is a base run here -- tagged ``base`` or untagged (a custom workflow's sampler)."""
    torch = require_torch()
    if pass_tag(args) == PASS_HIRES:
        return
    sigma = sampler_sigma(args)
    if sigma is None or sigma <= 0.0 or not torch.is_tensor(result):
        return
    if _new_run(run.record_last_sigma, sigma):
        run.trajectory.clear()   # a new base run (the next generation) starts a new trajectory
    run.record_last_sigma = sigma
    run.trajectory.record(sigma, result)


def append_post_cfg(patched: Any, function: Any) -> None:
    """Append ``function`` to the clone's ``sampler_post_cfg_function`` list — what ComfyUI's
    ``set_model_sampler_post_cfg_function`` does, written on model_options like ``guidance_dcw.patch_dcw``."""
    options = dict(getattr(patched, "model_options", {}) or {})
    options["sampler_post_cfg_function"] = [*(options.get("sampler_post_cfg_function") or []), function]
    patched.model_options = options


def patch_detail_stages(model: Any, run: DetailRun, *, adg_skipped_of=None) -> Any:
    """Append the detail post-CFG function (after the PAG/SEG/SLG functions, before DCW/RDC)."""
    if not run.settings.any:
        return model
    patched = clone_model(model, "Anima detail stages")

    def detail_stages(args: dict[str, Any]):
        result = args["denoised"]
        skipped = bool(adg_skipped_of(args)) if callable(adg_skipped_of) else False
        try:
            return apply_detail_stages(run, args, result, adg_skipped=skipped)
        except Exception as exc:   # noqa: BLE001
            run.warn_once("detail", f"Anima detail stages fallback: {type(exc).__name__}: {exc}")
            return result

    append_post_cfg(patched, detail_stages)
    return patched


def patch_hiflow_recorder(model: Any, run: DetailRun) -> Any:
    """Append the HiFlow recorder after every other post-CFG function (the final guided x0, after DCW)."""
    if not run.settings.hiflow:
        return model
    patched = clone_model(model, "Anima HiFlow recorder")

    def hiflow_record(args: dict[str, Any]):
        result = args["denoised"]
        try:
            record_hiflow(run, args, result)
        except Exception as exc:   # noqa: BLE001
            run.warn_once("hiflow_record", f"Anima HiFlow recording failed: {type(exc).__name__}: {exc}")
        return result

    append_post_cfg(patched, hiflow_record)
    return patched


def mark_pass_aware(model: Any) -> Any:
    """A MODEL clone the sampler nodes tag with their pass (only S² and HiFlow need it)."""
    patched = clone_model(model, "Anima pass-aware guidance")
    options = dict(getattr(patched, "model_options", {}) or {})
    options[PASS_AWARE_KEY] = True
    patched.model_options = options
    return patched


def with_pass_tag(model: Any, tag: str) -> Any:
    """A MODEL clone whose sampling runs carry ``PASS_KEY = tag`` in transformer_options.

    Only a MODEL the suite marked pass-aware (``mark_pass_aware`` — S² or HiFlow on) is cloned; any
    other MODEL is returned as it is, so the sampler nodes keep sampling the incoming MODEL."""
    if not (getattr(model, "model_options", {}) or {}).get(PASS_AWARE_KEY):
        return model
    patched = clone_model(model, f"Anima pass tag ({tag})")
    options = dict(getattr(patched, "model_options", {}) or {})
    transformer = dict(options.get("transformer_options", {}) or {})
    transformer[PASS_KEY] = tag
    options["transformer_options"] = transformer
    patched.model_options = options
    return patched
