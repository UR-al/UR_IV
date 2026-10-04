"""Colorcraft basis vectors: which family a model's latent format uses, its calibration, the vector files.

Pack 1.7.0, part of the ComfyUI port of sam-extra v0.32.1's Colorcraft (``ForgeNeoColorcraft``). Written from
two MIT sources (no sam-extra code — the extension is GPL-3.0 and serves only as the test oracle,
``tests/test_comfy_colorcraft_parity.py``):

* origin: muerrilla/ComfyUI-Colorcraft@d28ac6a4e997d0f8a2f1a60b7361b561c4a15bbf:lib_colorcraft/basis.py —
  ``load_basis`` and ``resolve_dev`` unchanged, ``VAE_DOWNSCALE_FACTOR`` and the krea2/zimage rows of
  ``MODEL_DEV_DEFAULTS`` unchanged. Upstream's file is not vendored: it has no Flux2 row and its
  ``to_model_space`` reads ``vae.latent_dim``.
* origin: aoleg/ComfyUI-Colorcraft@f00066c63c9d8f96abc119cada3b51b36688fcac:lib_colorcraft/core.py —
  ``family_for_latent_format`` and ``to_model_space(latent_format, anchor, dims)`` unchanged,
  ``BASIS_FAMILIES``, ``VAE_DOWNSCALE_FACTORS``, ``LATENT_FORMAT_TO_FAMILY`` and the ``flux2`` row of
  ``MODEL_DEV_DEFAULTS``, to which upstream's later ``detail_scale`` (4.0) is added: upstream's
  ``resolve_dev`` reads that key for every family and the fork's row predates it.

Pack-only: ``load_file`` wraps ``safetensors.torch.load_file`` so this module imports neither torch nor
safetensors when the pack loads; ``vectors_dir``/``basis_for`` find the vendored vector files
(``vendor/comfyui_colorcraft/vectors`` and, for Flux2, ``vendor/comfyui_colorcraft_fork/vectors``);
``vae_downscale_factor`` reads the per-family table with upstream's constant as the fallback.

MIT License — Copyright (c) 2026 Sahand Ahmadian Tehrani (Muerrilla); the aoleg fork (Oleg Afonin) is
under the same license (its LICENSE carries the original notice "Copyright (c) 2026 Sahand
Ahmadiantehrani (Muerrilla)"). Full texts: LICENSES/ComfyUI-Colorcraft-MIT.txt and
LICENSES/ComfyUI-Colorcraft-aoleg-MIT.txt.
"""

from __future__ import annotations

import os

UPSTREAM_ORIGIN = "muerrilla/ComfyUI-Colorcraft@d28ac6a4e997d0f8a2f1a60b7361b561c4a15bbf"
FORK_ORIGIN = "aoleg/ComfyUI-Colorcraft@f00066c63c9d8f96abc119cada3b51b36688fcac"

_VENDOR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "vendor")
UPSTREAM_VECTORS_DIR = os.path.join(_VENDOR, "comfyui_colorcraft", "vectors")
FORK_VECTORS_DIR = os.path.join(_VENDOR, "comfyui_colorcraft_fork", "vectors")

# fork core.py
BASIS_FAMILIES = ["krea2", "zimage", "flux2"]

# upstream basis.py: image pixels per latent pixel when a family has no entry below
VAE_DOWNSCALE_FACTOR = 8

# fork core.py: Flux2's VAE downscales 16x (the mask-blur radius is given in image pixels)
VAE_DOWNSCALE_FACTORS = {
    "krea2": 8,
    "zimage": 8,
    "flux2": 16,
}

# fork core.py (upstream basis.py has the first two rows): latent_format class name -> basis family.
# Krea 2 / Qwen-Image / Wan 2.1 / Anima report "Wan21", Flux / Z-Image / Lumina 2 "Flux", Flux 2 "Flux2".
LATENT_FORMAT_TO_FAMILY = {
    "Wan21": "krea2",
    "Flux": "zimage",
    "Flux2": "flux2",
}

# Per-family calibration of the chroma-plane math. krea2/zimage: upstream basis.py; flux2: the fork's
# measured row plus upstream's detail_scale.
MODEL_DEV_DEFAULTS = {
    "krea2":  {"vibrance_k": 1.0, "max_chroma": 2.5, "recenter": 0.5, "chroma_plane": "temp_tint", "exposure_scale": 3.5, "color_scale": 3.0, "detail_scale": 4.0, "hue_bias": -0.1},
    "zimage": {"vibrance_k": 1.5, "max_chroma": 5.0, "recenter": 0.5, "chroma_plane": "temp_tint", "exposure_scale": 7.5, "color_scale": 6.0, "detail_scale": 4.0, "hue_bias": -0.4},
    "flux2":  {"vibrance_k": 1.83, "max_chroma": 4.6, "recenter": 0.5, "chroma_plane": "temp_tint", "exposure_scale": 4.0, "color_scale": 3.9, "detail_scale": 4.0, "hue_bias": +0.12},
}

# Which vendored directory holds each family's colorcraft-<family>.safetensors.
_FAMILY_VECTORS_DIR = {
    "krea2": UPSTREAM_VECTORS_DIR,
    "zimage": UPSTREAM_VECTORS_DIR,
    "flux2": FORK_VECTORS_DIR,
}


def load_file(path):
    """``safetensors.torch.load_file``, imported on first use (the pack loads without torch)."""
    from safetensors.torch import load_file as _load_file

    return _load_file(path)


# --- upstream lib_colorcraft/basis.py, unchanged ---------------------------------------------------

def load_basis(family, vectors_dir):
    """Loads colorcraft-<family>.safetensors from vectors_dir. Returns
    dict[name -> 1D tensor] or None if missing. vectors_dir must be
    supplied by the caller, computed relative to its own entry-point
    file -- resolving it from this module's own __file__ would point at
    lib_colorcraft/ instead of the repo root."""
    path = os.path.join(vectors_dir, f"colorcraft-{family}.safetensors")
    if not os.path.isfile(path):
        return None
    return load_file(path)


def resolve_dev(params, family):
    """Resolves recenter/max_chroma/chroma_plane against
    MODEL_DEV_DEFAULTS, honoring per-field *_override flags in params
    when present (only Advanced exposes these)."""
    dev = MODEL_DEV_DEFAULTS[family]
    return {
        "vibrance_k": dev["vibrance_k"],
        "exposure_scale": dev["exposure_scale"],
        "color_scale": dev["color_scale"],
        "detail_scale": dev["detail_scale"],
        "hue_bias": dev["hue_bias"],
        "recenter": params["recenter"] if params.get("recenter_override") else dev["recenter"],
        "max_chroma": params["max_chroma"] if params.get("max_chroma_override") else dev["max_chroma"],
        "chroma_plane": params["chroma_plane"] if params.get("chroma_plane_override") else dev["chroma_plane"],
    }


# --- fork lib_colorcraft/core.py, unchanged --------------------------------------------------------

def family_for_latent_format(latent_format):
    """`latent_format` may be a class or an instance; ComfyUI hands over an
    instance and so does Forge Neo (`model_list.py:66` instantiates it)."""
    if latent_format is None:
        return None
    name = latent_format.__name__ if isinstance(latent_format, type) else type(latent_format).__name__
    return LATENT_FORMAT_TO_FAMILY.get(name)


def to_model_space(latent_format, anchor, dims=2):
    """Lifts a per-channel VAE-space anchor into the model space the sampler
    actually works in. `dims` is the latent's spatial rank — 2 for Flux-family
    VAEs, 3 for the Wan-family ones (which carry a temporal axis)."""
    if latent_format is None or not hasattr(latent_format, "process_in"):
        return anchor
    shaped = anchor.view((1, anchor.shape[0]) + (1,) * dims)
    return latent_format.process_in(shaped)[0].reshape(-1)


# --- pack ------------------------------------------------------------------------------------------

def vae_downscale_factor(family):
    """Image pixels per latent pixel for ``family`` (upstream's 8 for a family with no entry)."""
    return VAE_DOWNSCALE_FACTORS.get(family, VAE_DOWNSCALE_FACTOR)


def vectors_dir(family):
    """The vendored directory that holds ``family``'s vector file (the upstream one for unknown names)."""
    return _FAMILY_VECTORS_DIR.get(family, UPSTREAM_VECTORS_DIR)


def basis_for(family):
    """``family``'s basis vectors (``dict[name -> 1-D float32 tensor]``) or None when the file is missing."""
    if family is None:
        return None
    return load_basis(family, vectors_dir(family))
