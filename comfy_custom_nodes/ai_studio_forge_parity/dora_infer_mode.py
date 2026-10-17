"""DoRA inference mode for ComfyUI (pack 1.9.0): the sam-extra Forge extension's "DoRA 추론 방식" per MODEL.

ComfyUI merges a LoRA/LoHa/LoKr/GLoRA patch that carries a ``dora_scale`` with ``weight_decompose``
(``comfy/weight_adapter/base.py``), the same function Forge ships. It differs from the LyCORIS training merge
(``lycoris/modules/lokr.py`` ``get_merged_weight`` -> ``apply_weight_decompose``) twice: on the output axis
(``dora_scale`` [out, 1, ...], LyCORIS' default) it divides by the norm of the ORIGINAL weight ``||W0||`` where
LyCORIS divides by ``||W0 + dW||`` (the input axis, [1, in, ...], uses ``||W0 + dW||`` in both), and it runs in
the dtype ComfyUI merges in: fp16 on most GPUs for a full load, the model's bf16/fp16 for a LowVramPatch (dynamic
VRAM, lowvram), with that dtype's eps. LyCORIS computes in fp32 with the fp32 eps.

Modes (sam-extra ``sam3ext/dora_infer_mode.py`` ``MODES``, same keys and meaning):

- ``lycoris`` ("LyCORIS fp32 (as trained)"): LyCORIS' ``apply_weight_decompose`` operation for operation in fp32:
  ``W = W0 + a*dW``, ``r = m / (vector_norm(W) + eps32)`` over every dim but the DoRA axis (per row on the output
  axis, per column on the input axis), ``r = 1*(r - 1) + 1`` (its multiplier step at 1, not a no-op in fp32 when
  ``r`` is outside [0.5, 2]), ``W * r``. At strength 1 this is the LyCORIS training merge bit for bit, computed from
  the ``W0`` ComfyUI hands the patch: on its GPU full-load path that is a copy in ``lora_compute_dtype`` (fp16 on most
  GPUs), lossless for fp16 and fp8 weights and practically for bf16 ones (only values below fp16's normal range lose
  bits), while an fp32-stored weight keeps fp16's rounding of ``W0`` (stock ComfyUI has the same limit).
- ``forge_fp32`` ("Forge/Comfy fp32"): ComfyUI's formula (``||W0||`` on the output axis) in fp32 with the fp32 eps,
  sam-extra's operations.
- ``forge`` ("Forge/Comfy (stock)"): ComfyUI's own merge, unchanged.
- ``no_magnitude`` ("DoRA off (no magnitude)"): the ``dora_scale`` is ignored and the patch merges the way ComfyUI
  merges the same file without it (``W0 + s*a*dW`` in the merge dtype, ComfyUI's own non-DoRA code).

Strength (the LoRA loader's strength, ``patch[0]``) blends the two fp32 modes like Forge and sam-extra:
``W0 + s*(W_dora - W0)``; LyCORIS' own multiplier (all of ``dW``, only the magnitude interpolated) is not followed.
``patch[2]`` (strength_model) is applied by ComfyUI before the adapter, as before.

How (no ComfyUI function or module is patched; only MODELs that pass through the node change): the node clones the
MODEL and gives the clone new patch lists in which every LoRA/LoHa/LoKr/GLoRA adapter with a ``dora_scale`` is an
instance of a subclass of its own ComfyUI class (``mode_class``) that carries the mode as a CLASS attribute.
ComfyUI's LowVramPatch rebuilds adapters as ``type(v)(v.loaded_keys, weights)`` (``comfy/lora.py``
``prefetch_prepared_value``), so an instance attribute would be lost there; the class is kept. The clone gets a new
``patches_uuid``: ComfyUI only re-merges (and dynamic VRAM only drops its cached patched weights) when that changes.
The adapter objects, their ``weights`` tuples and tensors are shared with the parent MODEL and the CLIP and are never
written.

The subclass' ``calculate_weight`` (``merge_dora_patch``) gets ``a*dW`` in fp32 from the adapter's own ComfyUI code
(``fp32_delta``: its non-DoRA path at strength 1 with fp32 intermediates, stopped by the ``function`` it is handed
before it adds anything), whatever dtype ComfyUI merges in, and ``decompose`` writes the merge into that delta in
fp32. Besides the weight, at most two fp32 copies of the layer exist at a time (``dW`` and ComfyUI's intermediate for
it, then ``W0`` and the merge), as in ComfyUI's stock full-load merge; its LowVramPatch merge (bf16/fp16) needs half
that. The result is returned in fp32 only on ComfyUI's full-load path (``patch_weight_to_device``: not a view, fp32
intermediate, a non-fp32 weight), which rounds it once to the stored dtype; offset patches (a ``narrow`` view whose
return value ComfyUI drops) and LowVramPatch (the result goes straight into the layer) get it written in place in the
weight's dtype. A failure merges that patch the stock way; an OOM empties the cache first, and so does a ``dW`` that
ComfyUI's adapter never produced (the adapters log and swallow their own errors, OOM included).

Console: the node logs one line when it runs (ComfyUI then caches its output). Its clone also carries an
``OUTER_SAMPLE`` wrapper (``WRAPPER_KEY``) that starts a fresh account for each sampling run of that MODEL (sam-extra
resets its counters per generation, ``take_counters``) and logs at its end the DoRA patches merged in that run and the
merges that fell back to the stock way with the first cause. A fallback is also logged when it happens, once per kind
per run.

origin: sam-extra (installed v0.34.0, HEAD 659d354) ``sam3ext/dora_infer_mode.py`` by the same author, under the
author's MIT grant (THIRD_PARTY_NOTICES.md): ``decompose`` (lines 153-186) — its Forge/Comfy branch, the strength
blend and the in-place/return rule are sam-extra's operations; the LyCORIS branch takes LyCORIS' own operations
instead of sam-extra's reshape norms, and the delta arrives alpha-scaled and is merged in place — and
``_can_return_fp32`` (189-202) as it is; ``_is_oom``/``_soft_empty_cache`` (205-232) read ComfyUI's
``comfy.model_management`` instead of Forge's ``backend.memory_management``; the mode keys, their meaning and the stock
fallback follow ``MODES`` (48-52) and ``_make_wrapper`` (243-278). sam-extra swaps ``weight_decompose`` in Forge's
adapter modules for one global mode; here the mode travels with the patches of one MODEL. No torch or ComfyUI import
at module level.
"""

from __future__ import annotations

import importlib
import sys
import uuid
from collections import Counter
from typing import Any, Callable, Mapping, NamedTuple

from .compat import LOGGER, clone_model

FEATURE = "DoRA inference mode"
LOG_PREFIX = "[DoRA mode]"

MODE_FORGE = "forge"
MODE_FORGE_FP32 = "forge_fp32"
MODE_LYCORIS = "lycoris"
MODE_NO_MAGNITUDE = "no_magnitude"
MODES = (MODE_FORGE, MODE_FORGE_FP32, MODE_LYCORIS, MODE_NO_MAGNITUDE)  # sam3ext/dora_infer_mode.py MODES

# The node's COMBO, in sam-extra's radio order (scripts/dora_infer_mode.py MODE_CHOICES) and English like the pack;
# sam-extra's ``normalize_mode`` reads each label as the same key (tests/test_comfy_dora_infer_mode.py).
MODE_LABELS = {
    MODE_LYCORIS: "LyCORIS fp32 (as trained)",
    MODE_FORGE_FP32: "Forge/Comfy fp32",
    MODE_FORGE: "Forge/Comfy (stock)",
    MODE_NO_MAGNITUDE: "DoRA off (no magnitude)",
}
MODE_CHOICES = tuple(MODE_LABELS.values())
DEFAULT_MODE = MODE_LYCORIS  # sam-extra's radio default
MODE_BY_LABEL = {label: mode for mode, label in MODE_LABELS.items()}
_MODE_SUMMARY = {
    MODE_LYCORIS: "fp32, LyCORIS formula (||W0+dW|| on both axes)",
    MODE_FORGE_FP32: "fp32, Forge/Comfy formula (||W0|| on the output axis)",
    MODE_FORGE: "ComfyUI's own merge",
    MODE_NO_MAGNITUDE: "magnitude off (W0 + s*a*dW, ComfyUI's non-DoRA merge)",
}

# Class attributes of the mode subclasses (``mode_class``).
MODE_ATTR = "_ai_studio_dora_mode"
STOCK_ATTR = "_ai_studio_dora_stock_class"
SPEC_ATTR = "_ai_studio_dora_spec"

# ModelPatcher.add_wrapper_with_key key of the node's OUTER_SAMPLE wrapper (the per-run account and report), and
# comfy.patcher_extension.WrappersMP.OUTER_SAMPLE's value outside ComfyUI.
WRAPPER_KEY = "ai_studio_forge_parity.dora_infer_mode"
_OUTER_SAMPLE_FALLBACK = "outer_sample"


class AdapterSpec(NamedTuple):
    """Where a ComfyUI adapter class keeps its ``dora_scale`` in ``weights`` (and a LoRA its ``reshape_weight``),
    and whether its ``dW`` reads the values of ``W0`` (GLoRA)."""

    dora_index: int
    reshape_index: int | None = None
    reads_weight: bool = False


# ComfyUI 0.36 ``comfy/weight_adapter``: lora.py ``(up, down, alpha, mid, dora_scale, reshape)``, loha.py
# ``(w1a, w1b, alpha, w2a, w2b, t1, t2, dora_scale)``, lokr.py ``(w1, w2, alpha, w1_a, w1_b, w2_a, w2_b, t2,
# dora_scale)``, glora.py ``(a1, a2, b1, b2, alpha, dora_scale)`` — the four adapters sam-extra wraps
# (``TARGET_ADAPTERS``). Only these exact classes are re-classed.
SUPPORTED_ADAPTERS = {
    "LoRAAdapter": AdapterSpec(4, 5),
    "LoHaAdapter": AdapterSpec(7),
    "LoKrAdapter": AdapterSpec(8),
    "GLoRAAdapter": AdapterSpec(5, reads_weight=True),
}
# Adapters that may carry a dora_scale but stay stock, as in sam-extra: OFT/BOFT put the strength into the diff and
# pass their own constraint as alpha (oft.py, boft.py ``weights = (blocks, rescale, alpha, dora_scale)``).
STOCK_ONLY_ADAPTERS = {"OFTAdapter": AdapterSpec(3), "BOFTAdapter": AdapterSpec(3)}

_CLASSES: dict[tuple[type, str], type] = {}
_STATS: Counter = Counter()          # "merges", "fallbacks" since import (read by the tests through ``merge_stats``)


def mode_key(value: Any) -> str:
    """The mode key of a COMBO label (or of a key). Unknown values raise ValueError."""
    text = str(value if value is not None else "").strip()
    if text in MODE_BY_LABEL:
        return MODE_BY_LABEL[text]
    if text in MODES:
        return text
    raise ValueError(f"{FEATURE}: unknown mode {value!r} (choices: {', '.join(MODE_CHOICES)})")


def merge_stats() -> dict[str, int]:
    return dict(_STATS)


# ── the account of one sampling run ────────────────────────────────────────────────────────────────────

class RunAccount:
    """What the mode merges did since the account was last reset: when a sampling run of a patched MODEL starts
    (``dora_outer_sample_wrapper``) or the node builds a clone. Per mode: the DoRA patches merged (a model key and
    the patch's ``loaded_keys``, which ComfyUI's LowVramPatch rebuild keeps), the merges (LowVramPatch merges on
    every forward), the fallbacks and the first cause; and the fallback kinds already logged."""

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.patches: dict[str, set] = {}
        self.merges: Counter = Counter()
        self.fallbacks: Counter = Counter()
        self.first_cause: dict[str, str] = {}
        self.logged: set[tuple[str, str]] = set()

    def merged(self, mode: str, key: Any, adapter: Any) -> None:
        _STATS["merges"] += 1
        self.merges[mode] += 1
        self.patches.setdefault(mode, set()).add((str(key), id(getattr(adapter, "loaded_keys", adapter))))

    def fell_back(self, mode: str, key: Any, kind: str, detail: str) -> None:
        _STATS["fallbacks"] += 1
        self.fallbacks[mode] += 1
        self.first_cause.setdefault(mode, f"{kind}: {detail} ({key})")
        if (mode, kind) in self.logged:
            return
        self.logged.add((mode, kind))
        LOGGER.error("%s %s: %s: %s (%s) - merged the stock way; more of this kind are only counted until the "
                     "sampling run ends", LOG_PREFIX, MODE_LABELS.get(mode, mode), kind, detail, key)

    def report(self) -> list[tuple[int, str]]:
        """``(logging level, line)`` per mode that merged or fell back since the reset."""
        import logging

        lines = []
        for mode in MODES:
            label = MODE_LABELS[mode]
            merges, patches = self.merges.get(mode, 0), len(self.patches.get(mode, ()))
            if merges:
                every = f" ({merges} merges: LowVramPatch merges on every forward)" if merges > patches else ""
                lines.append((logging.INFO, f"{LOG_PREFIX} {label}: {patches} DoRA patches merged this run{every}"))
            fallbacks = self.fallbacks.get(mode, 0)
            if fallbacks:
                lines.append((logging.WARNING, f"{LOG_PREFIX} {label}: {fallbacks} merges fell back to the stock way "
                                               f"this run - first cause: {self.first_cause.get(mode)}"))
        return lines


RUN = RunAccount()


def dora_outer_sample_wrapper() -> Callable:
    """``WrappersMP.OUTER_SAMPLE`` wrapper (around ComfyUI's model load and the sampling): a fresh :data:`RUN`
    account, reported in ``finally``."""

    def dora_mode_outer_sample(executor, *args, **kwargs):
        RUN.reset()
        try:
            return executor(*args, **kwargs)
        finally:
            for level, line in RUN.report():
                LOGGER.log(level, "%s", line)

    return dora_mode_outer_sample


# ── fp32 DoRA (sam-extra's code, the author's MIT grant; LyCORIS' operations in the LyCORIS branch) ──────────

def decompose(dora_scale, weight, delta, strength, function, *, lycoris: bool, inplace: bool = True, w0=None):
    """DoRA in fp32 of ``W0`` (``weight``) and ``a*dW`` (``delta``, fp32, owned by this call).

    ``function`` (the patch's) is applied to ``delta``; when it hands the same tensor back, ``W0`` is added into it
    (``W0 + a*dW`` is the same sum either way round), so besides the weight only ``W0`` in fp32 (``w0``, made here
    when not given) and the merge exist. ``lycoris``: LyCORIS ``apply_weight_decompose`` (``lycoris/modules``
    ``lokr.py``, ``locon.py``, ``loha.py``) operation for operation — ``vector_norm`` over ``_dora_norm_dims`` of the
    merged weight plus the fp32 eps, ``dora_scale / norm``, ``1 * (scale - 1) + 1``, the product. Otherwise
    sam-extra's Forge/Comfy formula: ``||W0||`` per row on the output axis, ``||W0 + dW||`` per column on the input
    axis (ComfyUI's reshape/transpose norms), plus the fp32 eps. The axis is ComfyUI's: output when
    ``dora_scale.shape[0] == weight.shape[0]``. Then sam-extra's strength blend ``W0 + s*(W - W0)``.

    With ``inplace`` the result is written into ``weight`` and ``weight`` is returned: an offset patch hands over a
    ``narrow`` view and drops the return value. ``weight`` changes only through the final ``copy_``, so a failure on
    the way leaves it untouched. Without ``inplace`` the fp32 result tensor is returned.
    """
    import torch

    f32 = torch.float32
    if w0 is None:
        w0 = weight.to(f32)  # read only: the same tensor when weight is already fp32
    diff = function(delta)
    if diff is delta and diff.dtype == f32:
        merged = diff.add_(w0)
    else:
        merged = w0 + diff.to(f32)
    del diff, delta

    scale = dora_scale.to(device=merged.device, dtype=f32)
    output_axis = scale.shape[0] == merged.shape[0]  # ComfyUI's rule (base.py weight_decompose)
    if lycoris:
        dims = tuple(range(1, merged.dim())) if output_axis else (0, *range(2, merged.dim()))
        ratio = scale / (torch.linalg.vector_norm(merged, dim=dims, keepdim=True) + torch.finfo(f32).eps)
        merged.mul_(1 * (ratio - 1) + 1)  # LyCORIS: multiplier * (scale - 1) + 1, multiplier 1
    else:
        if output_axis:  # the Forge/Comfy rule: the original weight's norm
            norm = w0.reshape(w0.shape[0], -1).norm(dim=1, keepdim=True)
            norm = norm.reshape(w0.shape[0], *[1] * (w0.dim() - 1))
        else:  # input axis: the merged weight in Forge/Comfy and LyCORIS alike
            norm = merged.transpose(0, 1).reshape(merged.shape[1], -1).norm(dim=1, keepdim=True)
            norm = norm.reshape(merged.shape[1], *[1] * (merged.dim() - 1)).transpose(0, 1)
        norm = norm + torch.finfo(f32).eps
        merged.mul_(scale / norm)
    if strength != 1.0:
        merged.sub_(w0).mul_(float(strength)).add_(w0)
    if not inplace:
        return merged
    weight.copy_(merged)
    return weight


def _can_return_fp32(weight, intermediate_dtype) -> bool:
    """ComfyUI's full-load path, the only caller that takes an fp32 result and rounds it once itself.

    ``patch_weight_to_device`` calls ``calculate_weight`` without ``intermediate_dtype`` (= fp32) on a fresh copy in
    ``lora_compute_dtype`` and rounds the result to the stored dtype. LowVramPatch passes ``weight.dtype`` and an
    offset patch a ``narrow`` view (``_base`` set): both are written in place.
    """
    import torch

    return (
        getattr(weight, "_base", None) is None
        and intermediate_dtype == torch.float32
        and weight.dtype != torch.float32
    )


# ── the patch's calculate_weight ───────────────────────────────────────────────────────────────────────

def _identity(tensor):
    return tensor


def without_dora(adapter):
    """The adapter's stock ComfyUI class on the same tensors with ``dora_scale`` left out (a new object; the
    ``weights`` tuple of ``adapter`` is not touched)."""
    cls = type(adapter)
    stock = getattr(cls, STOCK_ATTR)
    index = getattr(cls, SPEC_ATTR).dora_index
    weights = adapter.weights
    plain = tuple(None if position == index else value for position, value in enumerate(weights))
    return stock(adapter.loaded_keys, list(plain) if isinstance(weights, list) else plain)


class MissingDelta(RuntimeError):
    """ComfyUI's adapter returned without handing its delta to ``function``: it logged and swallowed its own error
    (out of memory, more often than not, while building the full ``dW``)."""


class _DeltaCaptured(BaseException):
    """Raised from inside ComfyUI's adapter by :func:`_capture_delta` with ``a*dW``. A BaseException, so the
    adapter's ``except Exception`` (log and swallow) lets it through and the adapter never adds the delta."""

    def __init__(self, tensor) -> None:
        super().__init__()
        self.tensor = tensor


def _capture_delta(tensor):
    raise _DeltaCaptured(tensor)


def fp32_delta(plain, weight, key, original_weight=None, *, w0=None):
    """``a*dW`` of a DoRA-free adapter in fp32, a fresh tensor the caller owns.

    ComfyUI's non-DoRA line is ``weight += function(((strength * alpha) * dW).type(weight.dtype))``: called with
    strength 1, fp32 intermediates and an fp32 ``weight``, the tensor handed to ``function`` is ``a*dW`` in fp32.
    ``function`` is :func:`_capture_delta`, which carries it out of the adapter before the ``+=``, so the weight it is
    handed is never written: ``w0`` (``W0`` in fp32) for an adapter whose ``dW`` reads ``W0`` (GLoRA), otherwise a
    zero-stride fp32 stand-in with ``weight``'s shape and device (all the other adapters read), so no fp32 copy of
    ``W0`` exists while ComfyUI builds ``dW``. An adapter that fails logs and swallows its own error and returns
    without a delta: :class:`MissingDelta`.
    """
    import torch

    f32 = torch.float32
    target = w0 if w0 is not None else torch.empty((), dtype=f32, device=weight.device).expand(weight.shape)
    delta = None
    try:
        plain.calculate_weight(target, key, 1.0, 1.0, None, _capture_delta, f32, original_weight)
    except _DeltaCaptured as captured:
        delta = captured.tensor
    del target
    if delta is None:
        raise MissingDelta(f"ComfyUI's {type(plain).__name__} built no weight delta (its error is logged above)")
    if tuple(delta.shape) != tuple(weight.shape) or delta.dtype != f32:
        raise RuntimeError(f"weight delta {tuple(delta.shape)} {delta.dtype} does not fit {tuple(weight.shape)}")
    return delta


def _is_oom(exc: BaseException) -> bool:
    management = sys.modules.get("comfy.model_management")
    checker = getattr(management, "is_oom", None)
    if callable(checker):
        try:
            return bool(checker(exc))
        except Exception:
            pass
    torch = sys.modules.get("torch")
    oom = getattr(torch, "OutOfMemoryError", None) if torch is not None else None
    if oom is not None and isinstance(exc, oom):
        return True
    return "out of memory" in str(exc).lower()


def _soft_empty_cache() -> None:
    management = sys.modules.get("comfy.model_management")
    empty = getattr(management, "soft_empty_cache", None)
    if callable(empty):
        try:
            empty()
        except Exception:
            pass


def merge_dora_patch(adapter, weight, key, strength, strength_model, offset, function,
                     intermediate_dtype=None, original_weight=None):
    """``calculate_weight`` of a mode subclass (same arguments as ComfyUI's adapters)."""
    import torch

    cls = type(adapter)
    mode = getattr(cls, MODE_ATTR)
    stock = getattr(cls, STOCK_ATTR)
    spec = getattr(cls, SPEC_ATTR)
    if intermediate_dtype is None:
        intermediate_dtype = torch.float32
    function = function if function is not None else _identity
    args = (weight, key, strength, strength_model, offset, function, intermediate_dtype, original_weight)
    dora_scale = adapter.weights[spec.dora_index]
    if dora_scale is None or mode == MODE_FORGE:
        return stock.calculate_weight(adapter, *args)
    plain = without_dora(adapter)
    if mode == MODE_NO_MAGNITUDE:
        RUN.merged(mode, key, adapter)
        return plain.calculate_weight(*args)
    failure: tuple[str, str] | None = None
    out = w0 = None
    try:
        if spec.reads_weight:
            w0 = weight.to(torch.float32)
        out = decompose(dora_scale, weight, fp32_delta(plain, weight, key, original_weight, w0=w0), strength,
                        function, lycoris=(mode == MODE_LYCORIS),
                        inplace=not _can_return_fp32(weight, intermediate_dtype), w0=w0)
    except MissingDelta as exc:
        failure = ("possible OOM", str(exc))
    except Exception as exc:
        failure = ("OOM" if _is_oom(exc) else type(exc).__name__, f"{type(exc).__name__}: {exc}")
    w0 = None
    if failure is None:
        RUN.merged(mode, key, adapter)
        return out
    # Outside the except clause the frames holding the fp32 temporaries are gone; free the cache before the stock
    # merge (smaller temporaries) as sam-extra does — also when the adapter swallowed its error (often an OOM).
    kind, detail = failure
    if kind in ("OOM", "possible OOM"):
        _soft_empty_cache()
    RUN.fell_back(mode, key, kind, detail)
    return stock.calculate_weight(adapter, *args)


class _DoraModeMixin:
    """Mixed in front of a ComfyUI adapter class by ``mode_class``; the mode is a class attribute."""

    def calculate_weight(self, weight, key, strength, strength_model, offset, function,
                         intermediate_dtype=None, original_weight=None):
        return merge_dora_patch(self, weight, key, strength, strength_model, offset, function,
                                intermediate_dtype, original_weight)


def mode_class(stock: type, spec: AdapterSpec, mode: str) -> type:
    """The subclass of ComfyUI adapter class ``stock`` that merges its DoRA in ``mode`` (one class per pair)."""
    if mode not in MODES or mode == MODE_FORGE:
        raise ValueError(f"{FEATURE}: no subclass for mode {mode!r}")
    cached = _CLASSES.get((stock, mode))
    if cached is None:
        cached = type(f"{stock.__name__}DoRA_{mode}", (_DoraModeMixin, stock), {
            MODE_ATTR: mode, STOCK_ATTR: stock, SPEC_ATTR: spec, "__module__": __name__,
        })
        _CLASSES[(stock, mode)] = cached
    return cached


# ── the patches of a MODEL ─────────────────────────────────────────────────────────────────────────────

def adapter_specs(module: Any = None) -> dict[type, tuple[str, AdapterSpec, bool]]:
    """ComfyUI's adapter classes -> (name, spec, supported) from ``comfy.weight_adapter`` (or ``module``)."""
    if module is None:
        module = importlib.import_module("comfy.weight_adapter")
    specs = {}
    for table, supported in ((SUPPORTED_ADAPTERS, True), (STOCK_ONLY_ADAPTERS, False)):
        for name, spec in table.items():
            cls = getattr(module, name, None)
            if isinstance(cls, type):
                specs[cls] = (name, spec, supported)
    return specs


class Rewrite(NamedTuple):
    changes: dict          # model key -> new patch list (only the keys that change)
    dora: Counter          # DoRA patches that follow the mode, by adapter name
    stock: Counter         # DoRA patches left stock, by reason
    changed: int           # patch tuples replaced


def _weights(adapter) -> Any:
    weights = getattr(adapter, "weights", None)
    return weights if isinstance(weights, (tuple, list)) else None


def _convert(value, mode: str, specs, dora: Counter, stock: Counter):
    """The patch value for ``mode`` (``value`` itself when nothing changes)."""
    if isinstance(value, list) and value and isinstance(value[0], tuple) and len(value[0]) == 2:
        # a model-merge entry: [(weight, convert_func), patch tuples...]
        inner = [_convert_patch(patch, mode, specs, dora, stock) for patch in value[1:]]
        if all(new is old for new, old in zip(inner, value[1:])):
            return value
        return [value[0], *inner]
    cls = type(value)
    stock_class = cls.__dict__.get(STOCK_ATTR)
    if stock_class is not None:  # one of ours (the node ran before on this MODEL)
        spec = cls.__dict__[SPEC_ATTR]
        dora[stock_class.__name__] += 1
        if mode == cls.__dict__[MODE_ATTR]:
            return value
        target = stock_class if mode == MODE_FORGE else mode_class(stock_class, spec, mode)
        return target(value.loaded_keys, value.weights)
    known = specs.get(cls)
    weights = _weights(value)
    if known is None or weights is None:
        return value
    name, spec, supported = known
    if len(weights) <= spec.dora_index or weights[spec.dora_index] is None:
        return value
    if not supported:
        stock[f"{name} (stock in sam-extra too)"] += 1
        return value
    if spec.reshape_index is not None and len(weights) > spec.reshape_index and weights[spec.reshape_index] is not None:
        stock[f"{name} with reshape_weight"] += 1
        return value
    dora[name] += 1
    if mode == MODE_FORGE:
        return value
    return mode_class(cls, spec, mode)(value.loaded_keys, weights)


def _convert_patch(patch, mode, specs, dora, stock):
    if not isinstance(patch, tuple) or len(patch) < 2:
        return patch
    value = _convert(patch[1], mode, specs, dora, stock)
    if value is patch[1]:
        return patch
    return (patch[0], value, *patch[2:])


def rewrite_patches(patches: Mapping[str, list], mode: str, specs) -> Rewrite:
    """New patch lists for ``mode``; the given lists, tuples and adapters are not changed."""
    if mode not in MODES:
        raise ValueError(f"{FEATURE}: unknown mode {mode!r}")
    dora: Counter = Counter()
    stock: Counter = Counter()
    changes = {}
    changed = 0
    for key, items in patches.items():
        if not isinstance(items, list):
            continue
        converted = [_convert_patch(patch, mode, specs, dora, stock) for patch in items]
        replaced = sum(1 for new, old in zip(converted, items) if new is not old)
        if replaced:
            changes[key] = converted
            changed += replaced
    return Rewrite(changes, dora, stock, changed)


def _counts(counter: Counter) -> str:
    return ", ".join(f"{name.replace('Adapter', '')} {count}" for name, count in sorted(counter.items()))


def _outer_sample_type() -> str:
    try:
        return str(importlib.import_module("comfy.patcher_extension").WrappersMP.OUTER_SAMPLE)
    except Exception:
        return _OUTER_SAMPLE_FALLBACK


def _set_run_wrapper(patched: Any, mode: str) -> bool:
    """Put the per-run account wrapper on ``patched`` (replacing an earlier one of this node); none in stock mode.
    False when the MODEL has no wrapper support (the account then resets only when the node builds a clone)."""
    add = getattr(patched, "add_wrapper_with_key", None)
    remove = getattr(patched, "remove_wrappers_with_key", None)
    if not callable(add) or not callable(remove):
        return False
    wrapper_type = _outer_sample_type()
    remove(wrapper_type, WRAPPER_KEY)
    if mode != MODE_FORGE:
        add(wrapper_type, WRAPPER_KEY, dora_outer_sample_wrapper())
    return True


def apply_dora_mode(model: Any, mode: str, *, specs=None) -> Any:
    """``model`` with every DoRA patch already on it merged in ``mode`` (a clone with a new ``patches_uuid``), or
    ``model`` itself when nothing changes. Patches added to the result later (LoRA loaders after the node) stay
    stock."""
    if mode not in MODES:
        raise ValueError(f"{FEATURE}: unknown mode {mode!r}")
    patches = getattr(model, "patches", None)
    if not isinstance(patches, dict):
        raise RuntimeError(f"{FEATURE} requires a ComfyUI MODEL input.")
    plan = rewrite_patches(patches, mode, adapter_specs() if specs is None else specs)
    total = sum(plan.dora.values())
    left = f"; left stock: {_counts(plan.stock)}" if plan.stock else ""
    if not total:
        LOGGER.warning("%s no DoRA patch to change on this MODEL%s - returned unchanged. Put the node after the "
                       "LoRA loaders.", LOG_PREFIX, left)
        return model
    if not plan.changed:
        LOGGER.info("%s %s: %d DoRA patches already merge this way (%s)%s - returned unchanged.",
                    LOG_PREFIX, MODE_LABELS[mode], total, _counts(plan.dora), left)
        return model
    patched = clone_model(model, FEATURE)
    for key, items in plan.changes.items():
        patched.patches[key] = items
    patched.patches_uuid = uuid.uuid4()
    _set_run_wrapper(patched, mode)
    RUN.reset()
    LOGGER.info("%s %s: %d DoRA patches merge with %s (%s)%s. LoRAs added after this node are not affected.",
                LOG_PREFIX, MODE_LABELS[mode], total, _MODE_SUMMARY[mode], _counts(plan.dora), left)
    return patched


__all__ = [
    "AdapterSpec",
    "DEFAULT_MODE",
    "FEATURE",
    "LOG_PREFIX",
    "MODES",
    "MODE_ATTR",
    "MODE_BY_LABEL",
    "MODE_CHOICES",
    "MODE_FORGE",
    "MODE_FORGE_FP32",
    "MODE_LABELS",
    "MODE_LYCORIS",
    "MODE_NO_MAGNITUDE",
    "MissingDelta",
    "RUN",
    "Rewrite",
    "RunAccount",
    "STOCK_ATTR",
    "STOCK_ONLY_ADAPTERS",
    "SUPPORTED_ADAPTERS",
    "WRAPPER_KEY",
    "adapter_specs",
    "apply_dora_mode",
    "decompose",
    "dora_outer_sample_wrapper",
    "fp32_delta",
    "merge_dora_patch",
    "merge_stats",
    "mode_class",
    "mode_key",
    "rewrite_patches",
    "without_dora",
]
