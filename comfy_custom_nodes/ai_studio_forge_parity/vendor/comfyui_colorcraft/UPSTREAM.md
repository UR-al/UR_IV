# Pinned upstream manifest

## Source

- Repository: `https://github.com/muerrilla/ComfyUI-Colorcraft`
- Pinned commit: `d28ac6a4e997d0f8a2f1a60b7361b561c4a15bbf` (2026-09-01)
- Source-code license: MIT, Copyright (c) 2026 Sahand Ahmadian Tehrani (Muerrilla)
  (full text: `LICENSES/ComfyUI-Colorcraft-MIT.txt`)

The files below are upstream's files at that revision, unchanged: the Python files with
the LF line endings upstream stores (`vendor/.gitattributes` keeps `*.py` LF on
Windows), the vector files byte for byte (`.gitattributes` here marks them binary).
They carry no added header. SHA-256 is computed over the copied bytes.

| Vendored path | Upstream path | SHA-256 |
| --- | --- | --- |
| `lib_colorcraft/__init__.py` | `lib_colorcraft/__init__.py` (empty) | `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855` |
| `lib_colorcraft/vectors.py` | `lib_colorcraft/vectors.py` | `9f8a1dce3866a2c57cd912d89606d38a6e972b6d8a39009b18e0fd42f098666d` |
| `lib_colorcraft/masking.py` | `lib_colorcraft/masking.py` | `ce584f9b0a6db5ab8e42560afbc8c1a092cb268553045bc345b36f318b9f2df8` |
| `lib_colorcraft/color.py` | `lib_colorcraft/color.py` | `b7df301fcadc685ad0a6a78ee3a2a4c657e6e33fcf9639ef7a776b238307f37f` |
| `lib_colorcraft/schedule.py` | `lib_colorcraft/schedule.py` | `bc1b8b98079e861d50b7a1f4ad045fbd216523e9ee0ca3fb4b624f88c64c2534` |
| `vectors/colorcraft-krea2.safetensors` | `vectors/colorcraft-krea2.safetensors` | `0612ace7de9857927fc843f008a64633a097ab260b2c9ec3dc76b64c86b46cf2` |
| `vectors/colorcraft-zimage.safetensors` | `vectors/colorcraft-zimage.safetensors` | `824313efd6046492927087b3745cdc7e83ef8cd5dd5858fe42ec86c9b693c3f4` |

`tests/test_comfy_colorcraft_parity.py` pins the same hashes, so an edit to a vendored
file fails the app test suite. When the sam-extra Forge extension is installed, the
same test also checks them against the extension's own pins of upstream d28ac6a (the
`ORIGIN_SHA256` table of its `tests/test_colorcraft_origin.py`, which hashes the
verbatim upstream text of its test copies) and against the extension's `data/` vector
files, which are byte copies of the same upstream files. The extension is only a test
oracle; no extension file is part of this pack.

## Local integration boundary

`UPSTREAM.md`, `.gitattributes` and the top-level `__init__.py` are the only files in
this directory that AI Studio wrote. AI Studio does not edit the copied files. The
Colorcraft integration is one level up (pack 1.7.0):

- `colorcraft_basis.py` loads the vector files from `vectors/` here and from
  `../comfyui_colorcraft_fork/vectors/` (Flux2), with `safetensors.torch` imported
  inside the function.
- `colorcraft_engine.py` imports `lib_colorcraft.vectors`, `masking`, `color` and
  `schedule` when it runs (they import `torch`/`numpy` at module level, so the pack's
  module import stays torch-free) and calls their functions unchanged. Its modifier
  loop is written again from upstream `nodes.py` (`ColorcraftSampler.wrap` →
  `post_cfg_function`); the tests compare its syntax tree with upstream's loop.
- `colorcraft_host.py` / `colorcraft_nodes.py` attach that loop to ComfyUI's post-CFG
  hook through the pack's own `ForgeNeoColorcraft` node.

Upstream's `NODE_CLASS_MAPPINGS` (the `Colorcraft*` nodes of `nodes.py`) is never
imported or registered by this pack, so an independently installed upstream
ComfyUI-Colorcraft does not collide.

The following upstream repository files were intentionally not copied:

- `nodes.py` — the ComfyUI nodes (the pack has its own node; the modifier loop is
  re-expressed in `colorcraft_engine.py`, see above).
- `lib_colorcraft/basis.py` — it has no Flux2 row and its `to_model_space` reads
  `vae.latent_dim`; `colorcraft_basis.py` follows the fork's signature instead
  (`../comfyui_colorcraft_fork/UPSTREAM.md`).
- `lib_colorcraft/debug.py` — the debug renderer (not exposed by the app).
- `scripts/colorcraft.py`, `javascript/` — upstream's Forge script and slider helper.
- `LICENSE` (its text is `LICENSES/ComfyUI-Colorcraft-MIT.txt`), `README.md` and other
  repository metadata.

## Runtime prerequisites

`torch`, `numpy` and `safetensors` (all part of ComfyUI). No model weight is
included; the vector files are 1.4 KB tables of 16-channel directions.
