"""The DoRA merge formulas the pack 1.9.0 DoRA inference mode is checked against, written out (nothing imported).

* ``lycoris_merge``: LyCORIS 3.2 ``lycoris/modules/lokr.py`` ``get_merged_weight(multiplier=1)`` ->
  ``apply_weight_decompose(W0 + dW, 1)`` operation for operation: the weight in the fp32 of ``dora_scale``,
  ``vector_norm`` over ``_dora_norm_dims`` ((1..n) on the output axis, (0, 2..n) on the input axis, keepdim) plus
  ``_dora_eps`` (the fp32 eps), ``scale = dora_scale / norm``, ``scale = multiplier * (scale - 1) + 1`` (not a no-op
  in fp32 when ``scale`` is outside [0.5, 2]), ``weight * scale``; LoCon/LoHa (``locon.py``, ``loha.py``) are the
  same. lycoris is installed in neither the app venv nor ComfyUI's python, so the formula is written here.
* ``comfy_formula_fp32``: ComfyUI's/Forge's ``weight_decompose`` formula (the output axis divides by the norm of the
  ORIGINAL weight, no multiplier step) evaluated in fp32 with the fp32 eps. ComfyUI's own operations reduce the norm
  through reshape/transpose, so on conv weights this is equal to the pack's Forge/Comfy fp32 mode up to fp32
  rounding, not bit for bit.

Both blend a strength like Forge and sam-extra, ``W0 + s*(W_dora - W0)`` (LyCORIS' own multiplier is not the pack's
semantics). ``torch`` is passed in: test helpers never import it at module level (tests/_optional_deps).
"""
from __future__ import annotations


def norm_dims(ndim: int, wd_on_output: bool) -> tuple:
    return tuple(range(1, ndim)) if wd_on_output else (0,) + tuple(range(2, ndim))


def _decompose(torch, w0, delta, dora_scale, wd_on_output, strength, lycoris):
    f32 = torch.float32
    base = w0.to(f32)
    weight = base + delta.to(f32)
    source = weight if lycoris or not wd_on_output else base
    norm = torch.linalg.vector_norm(source, dim=norm_dims(weight.dim(), wd_on_output), keepdim=True)
    norm = norm + torch.tensor(torch.finfo(f32).eps)
    scale = dora_scale.to(f32) / norm
    if lycoris:
        scale = 1 * (scale - 1) + 1
    merged = weight * scale
    return merged if strength == 1.0 else base + strength * (merged - base)


def lycoris_merge(torch, w0, delta, dora_scale, wd_on_output: bool, strength: float = 1.0):
    """LyCORIS' training merge of ``W0`` and the alpha-scaled ``delta`` (fp32), blended by ``strength``."""
    return _decompose(torch, w0, delta, dora_scale, wd_on_output, strength, lycoris=True)


def comfy_formula_fp32(torch, w0, delta, dora_scale, wd_on_output: bool, strength: float = 1.0):
    """ComfyUI's formula in fp32: ``||W0||`` on the output axis, ``||W0 + dW||`` on the input axis."""
    return _decompose(torch, w0, delta, dora_scale, wd_on_output, strength, lycoris=False)


def trained_magnitude(torch, w0, wd_on_output: bool, generator, spread: float = 0.3):
    """A ``dora_scale`` like a trained one: ``||W0||`` along the DoRA axis (LyCORIS' initialisation), moved by up to
    +-``spread``/2."""
    norm = torch.linalg.vector_norm(w0.to(torch.float32), dim=norm_dims(w0.dim(), wd_on_output), keepdim=True)
    return norm * (1.0 + spread * (torch.rand(norm.shape, generator=generator) - 0.5))


def far_magnitude(torch, merged, wd_on_output: bool, generator):
    """A ``dora_scale`` whose ratio to ``||W0 + dW||`` leaves [0.5, 2] on some rows (columns): 3.7 on every third,
    0.23 on every fifth from the second, +-15 % elsewhere — where LyCORIS' ``1 * (scale - 1) + 1`` rounds."""
    norm = torch.linalg.vector_norm(merged.to(torch.float32), dim=norm_dims(merged.dim(), wd_on_output),
                                    keepdim=True)
    factor = 1.0 + 0.3 * (torch.rand(norm.shape, generator=generator) - 0.5)
    flat = factor.view(-1)
    flat[0::3] = 3.7
    flat[1::5] = 0.23
    return norm * factor
