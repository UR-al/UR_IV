# Pinned upstream manifest

## Source

- Repository: `https://github.com/iljung1106/comfyui-anima-safe-pag.git`
- Pinned commit: `905b0107d1f924fc6acbcac3b6a879b566ff671c`
- Commit subject: `Ignore local Comfy CLI files`
- Upstream package version at that commit: `0.1.0` (`pyproject.toml`)
- Source-code license: MIT (author `iljung1106`)

The file in the table below was copied byte-for-byte from the Git blob at that
revision (LF line endings, as stored upstream). It has not been reformatted,
renamed internally, or patched. SHA-256 is computed over the copied bytes.

| Vendored path | Upstream path | SHA-256 |
| --- | --- | --- |
| `__init__.py` | `__init__.py` | `e894ce01f7fcebe957eb3e682f959fa7b7826579af4534e001539aacd246bf37` |

`tests/test_forge_parity_pag_origin.py` pins the same hash, so an edit to the
vendored file fails the app test suite.

## Local integration boundary

`UPSTREAM.md` is the only locally authored file in this directory. AI Studio
does not edit the copied node. `guidance_pag.py` one level above imports it
lazily (it imports `torch` and `comfy.samplers` at module import) and calls
`AnimaSafePAG().patch(...)` for both `ForgeNeoAnimaSafePAG` and the PAG part of
`ForgeNeoAnimaGuidanceSuite`. The upstream `NODE_CLASS_MAPPINGS` is never
merged into the node pack's mappings, so no `AnimaSafePAG` node is registered
by this pack and an independently installed upstream copy does not collide.
`vendor/.gitattributes` (`*.py text eol=lf`) keeps the LF checkout on Windows.

The following upstream repository files were intentionally not copied because
they are repository metadata, documentation, images, or package metadata rather
than imported runtime code:

- `.comfyignore`
- `.gitignore`
- `README.md`
- `README.ko.md`
- `assets/*.png`
- `pyproject.toml`
- `LICENSE` (its text is `LICENSES/comfyui-anima-safe-pag-MIT.txt`)

## Runtime prerequisites

ComfyUI must provide `comfy.samplers.calc_cond_batch`, the MODEL patcher's
`set_model_sampler_calc_cond_batch_function` and
`set_model_sampler_post_cfg_function`, `model_sampling.percent_to_sigma`, and an
Anima/Cosmos/Predict2 diffusion model whose `blocks[i].self_attn` exposes
`compute_attention` and `n_heads`. No model weight is included.
