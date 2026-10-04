# Pinned upstream manifest

## Source

- Repository: `https://github.com/aoleg/ComfyUI-Colorcraft` (Oleg Afonin's fork of
  muerrilla/ComfyUI-Colorcraft; adds the Flux2 family and a Forge Neo script)
- Pinned commit: `f00066c63c9d8f96abc119cada3b51b36688fcac`
- Source-code license: MIT; the fork's `LICENSE` carries the original notice
  "Copyright (c) 2026 Sahand Ahmadiantehrani (Muerrilla)" (full text:
  `LICENSES/ComfyUI-Colorcraft-aoleg-MIT.txt`)

The file below is the fork's file at that revision, byte for byte (`.gitattributes`
here marks it binary). SHA-256 is computed over the copied bytes.

| Vendored path | Upstream path | SHA-256 |
| --- | --- | --- |
| `vectors/colorcraft-flux2.safetensors` | `vectors/colorcraft-flux2.safetensors` | `303e8a77d8fcba89162cbcc7b90a2e1094af5ff923b76be81b72a27c411bfacc` |

`tests/test_comfy_colorcraft_parity.py` pins the same hash and, when the sam-extra
Forge extension is installed, checks it against the extension's `data/` copy of the
same fork file (test oracle only).

## Local integration boundary

`UPSTREAM.md` and `.gitattributes` are the only files in this directory that AI Studio
wrote. No fork code is copied. `colorcraft_basis.py` one level up loads this file for
the `flux2` family (128 channels, 16x VAE downscale) and writes again, from the fork's
`lib_colorcraft/core.py`, the parts the original repository lacks: the
`to_model_space(latent_format, anchor, dims)` signature (latent rank passed in),
`family_for_latent_format`, the `Flux2` row of the latent-format table, the
per-family VAE downscale table and the `flux2` calibration row (plus the original's
`detail_scale` 4.0, which the fork's row predates).

Not copied: `lib_colorcraft/` (`core.py`, `params.py`, `spec.py`), the fork's nodes and
Forge Neo script, `LICENSE` (its text is `LICENSES/ComfyUI-Colorcraft-aoleg-MIT.txt`),
`README.md` and other repository metadata.
