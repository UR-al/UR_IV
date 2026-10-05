"""Colorcraft's modifier chain — what one model evaluation runs on the CFG-combined prediction.

Pack 1.7.0, part of the ComfyUI port of sam-extra v0.32.1's Colorcraft (``ForgeNeoColorcraft``). Written from
the MIT upstream node with sam-extra's three substitutions (sam-extra's code, licensed to this pack under MIT by
its author (THIRD_PARTY_NOTICES.md); the installed extension is the test oracle,
``tests/test_comfy_colorcraft_parity.py``):

origin: muerrilla/ComfyUI-Colorcraft@d28ac6a4e997d0f8a2f1a60b7361b561c4a15bbf:nodes.py

* ``sigma_to_value`` — upstream's function, unchanged.
* ``build_entries`` — the per-entry schedule loop of ``ColorcraftSampler.wrap.wrapped_sampler_function``,
  unchanged except that the key tuple is the module constant ``_SCHEDULE_KEYS``.
* ``needs_basis`` — that function's ``any_advanced`` test (with its ``ADVANCED_KINDS``).
* ``apply_chain`` — ``post_cfg_function``: the 5-D squeeze, the modifier loop statement for statement and
  the unsqueeze, with the three substitutions the sam-extra Forge hook makes (so both hosts run the same
  numbers; the tests compare the loop's syntax tree with upstream's after exactly these):

  1. ``get_color_latent(colour, latent_format, device, dtype)`` → ``get_color_latent(colour, device,
     dtype)``: the anchors are VAE-encoded once before sampling (``colorcraft_host``), not inside it;
  2. ``VAE_DOWNSCALE_FACTOR`` (upstream's constant 8) → ``vae_downscale_factor``, the family's own
     (16 for Flux2);
  3. ``s = sigma_to_value(cur_sigma, sigmas, schedule)`` → ``s = values[i]``: the host works the values
     out first (``schedule_values``) so an evaluation with every value at 0 can pass through untouched.

  Upstream's debug capture before the loop is not part of the port.
* ``needed_colors`` (pack): every anchor colour the loop's branches can ask for, so the host can encode
  them before sampling; ``schedule_values`` (pack): ``sigma_to_value`` per entry.

The vendored ``lib_colorcraft`` modules import torch/numpy at module level, so they are imported inside the
functions that use them: this module loads without torch.

MIT License — Copyright (c) 2026 Sahand Ahmadian Tehrani (Muerrilla). Full text:
LICENSES/ComfyUI-Colorcraft-MIT.txt.
"""

from __future__ import annotations

from .colorcraft_basis import resolve_dev

ORIGIN = "muerrilla/ComfyUI-Colorcraft@d28ac6a4e997d0f8a2f1a60b7361b561c4a15bbf:nodes.py"

# The ``kind`` each upstream node class writes into its chain entry (ColorcraftAdvanced, ColorcraftBasic,
# ColorcraftLuma, ColorcraftChroma, ColorcraftChromaPlus, ColorcraftPunch, ColorcraftShift).
KINDS = ("advanced", "basic", "luma", "chroma", "chroma_plus", "punch", "shift")

# Kinds whose controls depend on the resolved basis vectors (upstream wrapped_sampler_function).
ADVANCED_KINDS = {"advanced", "luma", "chroma", "chroma_plus", "punch"}

# The anchor contrast pivots on: build_color_latent of the plain mid-grey image.
NEUTRAL_COLOR = (0.0, 0.0, 0.0, 0.0)

_SCHEDULE_KEYS = ("start", "end", "bias", "exponent", "start_off", "end_off", "smooth")


def _lib():
    """The vendored upstream library modules (torch/numpy are imported here, on first use)."""
    from .vendor.comfyui_colorcraft.lib_colorcraft import color, masking, schedule, vectors

    return color, masking, schedule, vectors


# --- upstream nodes.py, unchanged --------------------------------------------------------------------

def sigma_to_value(sigma, sigmas, schedule):
    """Maps the sigma a model eval actually ran at to a value from the
    discrete per-step schedule array. Adapted from Jonseed's ComfyUI
    port of Detail Daemon: https://github.com/Jonseed/ComfyUI-Detail-Daemon"""
    real_sigmas = sigmas[:-1]
    n = len(schedule)
    if n < 2 or len(real_sigmas) < 2 or sigma <= 0:
        return float(schedule[0]) if n else 0.0

    deltas = (real_sigmas - sigma).abs()
    idx = int(deltas.argmin())

    if (
        (idx == 0 and sigma >= real_sigmas[0])
        or (idx == n - 1 and sigma <= real_sigmas[-1])
        or deltas[idx] == 0
    ):
        return float(schedule[idx])

    idx_lo, idx_hi = (idx, idx - 1) if sigma > real_sigmas[idx] else (idx + 1, idx)
    sig_lo, sig_hi = real_sigmas[idx_lo], real_sigmas[idx_hi]
    if sig_hi == sig_lo:
        return float(schedule[idx_lo])
    ratio = float(max(0.0, min(1.0, (sigma - sig_lo) / (sig_hi - sig_lo))))
    return float(schedule[idx_lo] + ratio * (schedule[idx_hi] - schedule[idx_lo]))


# --- upstream ColorcraftSampler.wrap ------------------------------------------------------------------

def build_entries(modifiers, num_steps):
    """``(schedule, kind, params, mask)`` per chain entry, each schedule ``num_steps`` long.

    The chain entries are the dicts upstream's modifier nodes build: ``{"kind", "params", "schedule",
    "mask"}``. Basic/Advanced keep their schedule widgets in ``params``; the others carry a separate
    ``schedule`` dict."""
    make_schedule = _lib()[2].make_schedule
    built = []
    for entry in modifiers:
        p = entry["params"]
        # Basic/Advanced own their schedule widgets inline (in `p`);
        # Luma/Chroma/Punch get theirs from a required
        # COLORCRAFT_SCHEDULE input instead, stashed on the entry.
        sched = entry.get("schedule") or p
        schedule_kwargs = {k: sched[k] for k in _SCHEDULE_KEYS}
        schedule = make_schedule(num_steps, amount=sched["strength"], **schedule_kwargs)
        built.append((schedule, entry["kind"], p, entry.get("mask")))
    return built


def needs_basis(built):
    """Does the chain need the basis vectors? Upstream's ``any_advanced``: a vector kind, or a Shift with
    a mask (Shift's own colour shift works on any model, its masking does not). Takes built entries
    ``(schedule, kind, params, mask)`` or chain-entry dicts."""
    return any(
        kind in ADVANCED_KINDS or (kind == "shift" and mask_params)
        for kind, mask_params in (_kind_and_mask(item) for item in built)
    )


def _kind_and_mask(item):
    if isinstance(item, dict):
        return item["kind"], item.get("mask")
    return item[1], item[3]


# --- pack ----------------------------------------------------------------------------------------------

def needed_colors(modifiers):
    """The ``(red, green, blue, brightness)`` anchors ``apply_chain`` can ask ``get_color_latent`` for, in
    first-use order without repeats — read from the chain entries so they can be encoded before sampling.

    The branches of the loop below: contrast (any entry whose params carry a non-zero ``contrast``:
    Advanced, Basic, Punch) pivots on ``NEUTRAL_COLOR``; colour shift with a non-zero amount on Basic and
    Shift, and on Advanced when its ``color_shift`` gate is on, uses the entry's colour."""
    colors = []
    for entry in modifiers:
        kind, p = entry["kind"], entry["params"]
        if p.get("contrast", 0) != 0:
            colors.append(NEUTRAL_COLOR)
        shifts = kind in ("basic", "shift") or (kind == "advanced" and p.get("color_shift"))
        if shifts and p.get("color_shift_amount", 0) != 0:
            colors.append((p["red"], p["green"], p["blue"], p["brightness"]))
    return list(dict.fromkeys(colors))


def schedule_values(built, cur_sigma, sigmas):
    """``sigma_to_value`` of every entry: the ``values`` ``apply_chain`` runs one evaluation with."""
    return [sigma_to_value(cur_sigma, sigmas, schedule) for schedule, _kind, _p, _mask in built]


def apply_chain(built, x0, values, family, cur_basis, get_color_latent, vae_downscale_factor):
    """Run the chain on ``x0`` (the CFG-combined ``denoised``) and return the graded prediction.

    ``built`` from ``build_entries``; ``values[i]`` entry ``i``'s schedule value for this evaluation;
    ``family`` the basis family (None: no basis); ``cur_basis`` that family's vectors on ``x0``'s device
    and dtype (None: the vector controls are off, as upstream); ``get_color_latent(colour, device, dtype)``
    the pre-encoded model-space anchor; ``vae_downscale_factor`` the family's image/latent pixel ratio.
    Out of place: with every value at 0 the result is ``x0`` itself (a 5-D view of it when ``x0`` is 5-D)."""
    color, masking, _schedule, vectors = _lib()
    apply_contrast, apply_color_shift = color.apply_contrast, color.apply_color_shift
    apply_mask_gate = masking.apply_mask_gate
    apply_vector_offset, apply_tone_compression = vectors.apply_vector_offset, vectors.apply_tone_compression
    apply_vibrance, apply_chroma_contrast = vectors.apply_vibrance, vectors.apply_chroma_contrast
    chroma_axes = vectors.chroma_axes

    is_5d = x0.dim() == 5
    if is_5d:
        x0 = x0.squeeze(2)

    out = x0
    for i, (schedule, kind, p, mask_params) in enumerate(built):
        s = values[i]
        if s == 0:
            continue

        pre = out
        if kind != "advanced" and "contrast" in p and p["contrast"] != 0:
            neutral_anchor = get_color_latent(
                (0.0, 0.0, 0.0, 0.0), out.device, out.dtype,
            )
            out = apply_contrast(out, s * p["contrast"], neutral_anchor)

        if kind == "basic":
            if p["color_shift_amount"] != 0:
                color_anchor = get_color_latent(
                    (p["red"], p["green"], p["blue"], p["brightness"]),
                    out.device, out.dtype,
                )
                out = apply_color_shift(out, s * p["color_shift_amount"], p["mode"], color_anchor)

        elif kind == "advanced":
            dev = resolve_dev(p, family) if cur_basis is not None else None
            if cur_basis is not None:
                axis1, axis2 = chroma_axes(cur_basis, dev["chroma_plane"])
                out = apply_vector_offset(out, cur_basis["exposure"], s * p["exposure"])
                out = apply_tone_compression(out, cur_basis["exposure"], s * p["tone_compression"])

            if p["contrast"] != 0:
                neutral_anchor = get_color_latent(
                    (0.0, 0.0, 0.0, 0.0), out.device, out.dtype,
                )
                out = apply_contrast(out, s * p["contrast"], neutral_anchor)

            if cur_basis is not None:
                out = apply_vibrance(out, axis1, axis2, s * p["vibrance"] * 2.0, k=dev["vibrance_k"], recenter=dev["recenter"], r_max=dev["max_chroma"])
                out = apply_vibrance(out, axis1, axis2, s * p["saturation"], k=0.0, r_max=0.0, recenter=dev["recenter"])
                out = apply_vector_offset(out, cur_basis["temperature"], s * p["temperature"])
                out = apply_vector_offset(out, cur_basis["tint"], s * p["tint"])
                if p["more_colors"]:
                    out = apply_vector_offset(out, cur_basis["temp+tint"], s * p["temp_plus_tint"])
                    out = apply_vector_offset(out, cur_basis["temp-tint"], s * p["temp_minus_tint"])
                    out = apply_vector_offset(out, cur_basis["lab-a"], s * p["lab_a"])
                    out = apply_vector_offset(out, cur_basis["lab-b"], s * p["lab_b"])
                    out = apply_vector_offset(out, cur_basis["lab-a+b"], s * p["lab_a_plus_b"])
                    out = apply_vector_offset(out, cur_basis["lab-a-b"], s * p["lab_a_minus_b"])
                out = apply_chroma_contrast(
                    out, axis1, axis2, s * p["chroma_contrast"],
                    r_max=dev["max_chroma"], chroma_center=p["chroma_center"], recenter=dev["recenter"],
                )

            if p["color_shift"] and p["color_shift_amount"] != 0:
                color_anchor = get_color_latent(
                    (p["red"], p["green"], p["blue"], p["brightness"]),
                    out.device, out.dtype,
                )
                out = apply_color_shift(out, s * p["color_shift_amount"], p["mode"], color_anchor)

            if cur_basis is not None:
                out = apply_vector_offset(out, cur_basis["clarity"], s * p["clarity"])
                out = apply_vector_offset(out, cur_basis["sharpness"], s * p["sharpness"])

            # Reuses the same `dev` already resolved above for its own
            # edits to gate an external mask -- one resolution, shared.
            if mask_params and cur_basis is not None:
                out = apply_mask_gate(pre, out, mask_params, dev, cur_basis, vae_downscale_factor)

        elif kind == "luma":
            dev = resolve_dev(p, family) if cur_basis is not None else None
            if cur_basis is not None:
                out = apply_vector_offset(out, cur_basis["exposure"], s * p["exposure"])
                out = apply_tone_compression(out, cur_basis["exposure"], s * p["tone_compression"])
            if mask_params and cur_basis is not None:
                out = apply_mask_gate(pre, out, mask_params, dev, cur_basis, vae_downscale_factor)

        elif kind == "chroma":
            dev = resolve_dev(p, family) if cur_basis is not None else None
            if cur_basis is not None:
                axis1, axis2 = chroma_axes(cur_basis, dev["chroma_plane"])
                out = apply_vibrance(out, axis1, axis2, s * p["vibrance"] * 2.0, k=dev["vibrance_k"], recenter=dev["recenter"], r_max=dev["max_chroma"])
                out = apply_vibrance(out, axis1, axis2, s * p["saturation"], k=0.0, r_max=0.0, recenter=dev["recenter"])
                out = apply_chroma_contrast(
                    out, axis1, axis2, s * p["chroma_contrast"],
                    r_max=dev["max_chroma"], chroma_center=p["chroma_center"], recenter=dev["recenter"],
                )
                out = apply_vector_offset(out, cur_basis["temperature"], s * p["temperature"])
                out = apply_vector_offset(out, cur_basis["tint"], s * p["tint"])
            if mask_params and cur_basis is not None:
                out = apply_mask_gate(pre, out, mask_params, dev, cur_basis, vae_downscale_factor)

        elif kind == "chroma_plus":
            dev = resolve_dev(p, family) if cur_basis is not None else None
            if cur_basis is not None:
                out = apply_vector_offset(out, cur_basis["temp+tint"], s * p["temp_plus_tint"])
                out = apply_vector_offset(out, cur_basis["temp-tint"], s * p["temp_minus_tint"])
                out = apply_vector_offset(out, cur_basis["lab-a"], s * p["lab_a"])
                out = apply_vector_offset(out, cur_basis["lab-b"], s * p["lab_b"])
                out = apply_vector_offset(out, cur_basis["lab-a+b"], s * p["lab_a_plus_b"])
                out = apply_vector_offset(out, cur_basis["lab-a-b"], s * p["lab_a_minus_b"])
            if mask_params and cur_basis is not None:
                out = apply_mask_gate(pre, out, mask_params, dev, cur_basis, vae_downscale_factor)

        elif kind == "punch":
            dev = resolve_dev(p, family) if cur_basis is not None else None
            if cur_basis is not None:
                out = apply_vector_offset(out, cur_basis["clarity"], s * p["clarity"])
                out = apply_vector_offset(out, cur_basis["sharpness"], s * p["sharpness"])
            if mask_params and cur_basis is not None:
                out = apply_mask_gate(pre, out, mask_params, dev, cur_basis, vae_downscale_factor)

        elif kind == "shift":
            if p["color_shift_amount"] != 0:
                color_anchor = get_color_latent(
                    (p["red"], p["green"], p["blue"], p["brightness"]),
                    out.device, out.dtype,
                )
                out = apply_color_shift(out, s * p["color_shift_amount"], p["mode"], color_anchor)
            if mask_params and cur_basis is not None:
                dev = resolve_dev(p, family)
                out = apply_mask_gate(pre, out, mask_params, dev, cur_basis, vae_downscale_factor)

    if is_5d:
        out = out.unsqueeze(2)
    return out
