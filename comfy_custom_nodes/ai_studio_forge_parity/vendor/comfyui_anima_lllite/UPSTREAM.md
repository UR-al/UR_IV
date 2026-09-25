# Pinned upstream manifest

## Source

- Repository: `https://github.com/kohya-ss/ComfyUI-Anima-LLLite.git`
- Pinned commit: `b7495bd8eb876e334509976896702484ed19cdbb`
- Commit subject: `Merge pull request #9 from kohya-ss/fix-node-id-collision-with-core`
- Source-code license: Apache-2.0 (full text: `LICENSES/ComfyUI-Anima-LLLite-Apache-2.0.txt`;
  upstream ships no `NOTICE` file)

The three files below were copied byte-for-byte from the Git blobs at that
revision (LF line endings, as stored upstream). They have not been
reformatted, renamed internally, or patched. SHA-256 is computed over the
copied bytes, which equal the upstream blobs.

| Vendored path | Upstream path | SHA-256 |
| --- | --- | --- |
| `__init__.py` | `__init__.py` | `4c0f96b0f0006df5a559d09d44b8d521a32d7992383223310b5ba2a3f88e3e42` |
| `nodes.py` | `nodes.py` | `5adffdab1a71d7da463e269d6656b8f9ef0d8720340dd02569793dbaf48383cb` |
| `control_net_lllite_anima.py` | `control_net_lllite_anima.py` | `91611fb108d25f6e402d7552326080bd7d70000cd05733239defb92e3bf4d308` |

`tests/test_comfy_anima_lllite.py` pins the same hashes, so an edit to a
vendored file fails the app test suite.

## Local integration boundary

`UPSTREAM.md` is the only locally authored file in this directory. AI Studio
does not edit the copied node. `anima_lllite.py` one level above imports
`nodes.py` lazily (it imports `torch` and ComfyUI's `folder_paths` at module
import) and calls `AnimaLLLiteApply_sdscripts().apply(...)` unchanged for:

- `ForgeNeoAnimaTileRepair` (the Anima Tile & Repair pipeline of civitai
  2708551 / kohya sd-scripts `anima_minimal_inference_control_net_lllite.py`);
- the SAM3 Detailer/Refine ControlNet slot, when the selected ControlNet file
  is an Anima ControlNet-LLLite (safetensors keys `lllite_conditioning1.*`).
  Such a file is applied as this MODEL patch instead of being handed to
  ComfyUI's `ControlNetLoader`, which cannot read it.

The upstream `NODE_CLASS_MAPPINGS` is never merged into the node pack's
mappings, so no `AnimaLLLiteApply_sdscripts` node is registered by this pack
and an independently installed upstream copy does not collide.
`vendor/.gitattributes` (`*.py text eol=lf`) keeps the LF checkout on Windows.

The following upstream repository files were intentionally not copied because
they are repository metadata or documentation rather than imported runtime
code:

- `.gitignore`
- `README.md`
- `LICENSE` (its text is `LICENSES/ComfyUI-Anima-LLLite-Apache-2.0.txt`)

## Runtime prerequisites

ComfyUI must provide `folder_paths` with a registered `controlnet` model
folder, the MODEL patcher's `clone`, `model_options`,
`set_model_unet_function_wrapper` and `get_model_object("model_sampling")`
(with `percent_to_sigma`), and an Anima diffusion model whose blocks expose
the `Attention` modules (`is_selfattn`, `q_proj`/`k_proj`/`v_proj`) the weight
metadata targets. `safetensors` reads the weights. No model weight is included.
