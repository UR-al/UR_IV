# Third-party notices

AI Studio Pro includes an unmodified, pinned copy of the Python runtime from
[`GumGum10/comfyui-anima-3-8B`](https://github.com/GumGum10/comfyui-anima-3-8B)
at commit `381c13af328b958febf86c155d2f4b007cd0f55b`. The copied runtime is under
`vendor/comfyui_anima_3_8b/` and is licensed under the MIT License. The full
license text is included at `LICENSES/comfyui-anima-3-8B-MIT.txt`.

The following two files within that runtime are unmodified copies from
[`Qwen/Qwen3.5-4B`](https://huggingface.co/Qwen/Qwen3.5-4B), revision
`851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`:

- `vendor/comfyui_anima_3_8b/qwen35_tokenizer/tokenizer.json`
- `vendor/comfyui_anima_3_8b/qwen35_tokenizer/tokenizer_config.json`

The Qwen upstream repository identifies these files under Apache-2.0. The full
license text is included at `LICENSES/Qwen3.5-Apache-2.0.txt`.

## Anima block-lineage metadata

The adjacent-generation LoRA mapping in `anima_lora.py` is locally authored
from published model-layout metadata; it does not copy Forge's loader code.
Its `_make_expansion_mapping` is the same code as the sam-extra extension's
(same author, licensed to this pack under MIT; see "Code shared with the
sam-extra Forge extension").
The 28-to-40 insertion positions come from
[`Gazingstars123/Anima-2.9B`](https://huggingface.co/Gazingstars123/Anima-2.9B/blob/9f9cb502dbae7a616c3cc5a530633427fe735665/expand_manifest.json).
The 40-to-52 positions are read from the `insertion_positions` and
`inserted_to_source` metadata published in the user-supplied
`Anima-3.8B-v1.1.safetensors` checkpoint. Only the integer layout facts are
encoded here; the expansion, contraction, composition, collision checks, and
LoRA key handling are local implementation code. No manifest or model weight
from either model repository is redistributed in this node pack.

AI Studio's `anima38_nodes.py` wrapper, its Forge Neo node names, and the
surrounding `vendor/__init__.py` namespace file are local integration code.
`anima38_nodes.py` includes a small integration copy of the 52-block detection
algorithm from upstream `patches.py`; it keeps the upstream compatibility
marker while avoiding eager ML-library imports. It is identified as local glue,
not as a byte-identical upstream file, and the upstream MIT notice is retained.
`anima38_cache.py` is a local integration shim that ties registered semantic
conditioning runs to Comfy cache ownership; it patches runtime entrypoints in
memory without modifying the pinned upstream source files.
No model checkpoint, text-encoder weight, VAE weight, LoRA, GGUF, or other model
binary is redistributed with this node pack. Those files must be supplied
separately by the user and remain subject to their own licenses.

See `vendor/comfyui_anima_3_8b/UPSTREAM.md` for the complete copied-file list,
SHA-256 values, omitted upstream files, dependencies, and local modification
boundary. These notices do not replace or modify either upstream license.

## Anima Safe PAG

AI Studio Pro includes an unmodified, pinned copy of the node module from
[`iljung1106/comfyui-anima-safe-pag`](https://github.com/iljung1106/comfyui-anima-safe-pag)
at commit `905b0107d1f924fc6acbcac3b6a879b566ff671c`. The copied file is
`vendor/comfyui_anima_safe_pag/__init__.py` and is licensed under the MIT
License (author iljung1106). The full license text is included at
`LICENSES/comfyui-anima-safe-pag-MIT.txt`.

`guidance_pag.py` calls the copied `AnimaSafePAG().patch(...)` for PAG; it is
local integration code and does not modify the copied file. See
`vendor/comfyui_anima_safe_pag/UPSTREAM.md` for the SHA-256 value, omitted
upstream files, and the local integration boundary. This notice does not
replace or modify the upstream license.

## Skimmed CFG

AI Studio Pro includes an unmodified, pinned copy of `skimmed_CFG.py` and
`__init__.py` from
[`Extraltodeus/Skimmed_CFG`](https://github.com/Extraltodeus/Skimmed_CFG)
at commit `d83005832ac42783adfd6f4ae96f6ef6406d1a74`. The copied files are in
`vendor/skimmed_cfg/` and are licensed under the Apache License, Version 2.0.
Only their line endings differ from upstream (CRLF there, LF here). The full
license text is included at `LICENSES/Skimmed_CFG-Apache-2.0.txt`; the upstream
repository ships no NOTICE file.

`guidance_skim.py` (`ForgeNeoSkimmedCFG`) calls the copied
`CFG_Skimming_Single_Scale_Pre_CFG.execute(...)` unchanged; it is local
integration code and does not modify the copied files. See
`vendor/skimmed_cfg/UPSTREAM.md` for the SHA-256 values, omitted upstream
files, and the local integration boundary. This notice does not replace or
modify the upstream license.

## Detail Daemon

`guidance_dd.py` contains unmodified copies of three functions and the
`DetailDaemonSamplerNode.INPUT_TYPES` body from `detail_daemon_node.py` of
[`Jonseed/ComfyUI-Detail-Daemon`](https://github.com/Jonseed/ComfyUI-Detail-Daemon)
at commit `3394e44afea04ed0188fb37b21f0d9952469766b`:
`make_detail_daemon_schedule` (lines 24-67), `get_dd_schedule` (226-262) and
`detail_daemon_sampler` (265-310). Only their line endings differ from upstream
(CRLF there, LF here). They are licensed under the MIT License (Copyright (c)
2024 Jonseed); the notice is kept above the copied code and the full license
text is included at `LICENSES/ComfyUI-Detail-Daemon-MIT.txt`.

`make_detail_daemon_schedule` is Jonseed's port of `make_schedule` from
`scripts/detail_daemon.py` (lines 309-338) of
[`muerrilla/sd-webui-detail-daemon`](https://github.com/muerrilla/sd-webui-detail-daemon)
at commit `19479998340831d7804fca8efd3f262b54b6373f`, where the Detail Daemon
schedule originates. That code is licensed under the MIT License (Copyright (c)
2024 Sahand Ahmadian); the notice is kept above the copied code and the full
license text is included at `LICENSES/sd-webui-detail-daemon-MIT.txt`.

The node class `ForgeNeoAnimaDetailDaemon`, its settings parsing and its
sampler-wrapper registration are local integration code. This notice does not
replace or modify either upstream license.

## Anima ControlNet-LLLite

AI Studio Pro includes unmodified, pinned copies of `__init__.py`, `nodes.py`
and `control_net_lllite_anima.py` from
[`kohya-ss/ComfyUI-Anima-LLLite`](https://github.com/kohya-ss/ComfyUI-Anima-LLLite)
at commit `b7495bd8eb876e334509976896702484ed19cdbb`. The copied files are in
`vendor/comfyui_anima_lllite/` and are licensed under the Apache License,
Version 2.0 (`control_net_lllite_anima.py` is kohya-ss's ComfyUI port of
`networks/control_net_lllite_anima.py` from
[`kohya-ss/sd-scripts`](https://github.com/kohya-ss/sd-scripts), also
Apache-2.0). The full license text is included at
`LICENSES/ComfyUI-Anima-LLLite-Apache-2.0.txt`; the upstream repository ships
no NOTICE file.

`anima_lllite.py` (`ForgeNeoAnimaTileRepair` and the SAM3 detailer's LLLite
slot) calls the copied `AnimaLLLiteApply_sdscripts().apply(...)` unchanged; it
is local integration code and does not modify the copied files. The Tile &
Repair defaults it uses are the argument defaults of kohya sd-scripts
`anima_minimal_inference.py` at commit
`690ea7f96c23182352ec63def76d431c6120bd2f`. Its `tile_repair_sigmas`
repeats the σ formula of `get_timesteps_sigmas` in
`library/hunyuan_image_utils.py` (lines 276-292) at that commit, and
`tile_repair_noise` and `sdscripts_control_image` follow that script's
initial-noise draw and control-image loading. sd-scripts is Copyright 2022
kohya-ss and licensed under the Apache License, Version 2.0, the same license
text as `LICENSES/ComfyUI-Anima-LLLite-Apache-2.0.txt`. No other sd-scripts
code is copied. `read_safetensors_header` and `tile_repair_size` are the
same code as the sam-extra extension's (same author, licensed to this pack under
MIT; see "Code shared with the sam-extra Forge extension"). See
`vendor/comfyui_anima_lllite/UPSTREAM.md` for the SHA-256 values, omitted
upstream files, and the local integration boundary. No LLLite or other model
weight is included. This notice does not replace or modify the upstream
license.

## Anima VAE DeGrid

`degrid_math.py`, `degrid_files.py`, `degrid_runner.py` and `degrid_nodes.py`
(`ForgeNeoAnimaVAEDeGrid`) reproduce the "Anima VAE DeGrid (NAFNet)" script of the
sam-extra Forge extension (`forge_sam3_extension`, GPL-3.0-only) on ComfyUI.
Some of their functions are sam-extra's own code — the safetensors header and
state-dict readers, the out-of-memory test and tile retry, the residual guard and
the residual run. The author licenses that code to this pack under MIT (see
"Code shared with the sam-extra Forge extension"). Besides that code, the mode
meanings, value ranges, tile/overlap sizes, the padding multiple, the
residual-guard thresholds and the leading error-category words follow the
extension. `tests/fixtures/degrid_origin_golden.json` holds only numbers and
checksums produced by running that extension on CPU.

The NAFNet detection key list is the one spandrel (MIT) uses to recognise the
architecture. spandrel itself (model loading), `comfy.utils.load_torch_file` and
ComfyUI's model management are called from the ComfyUI runtime at execution
time; none of their code is included. The tile blend has the same positions and
weights as ComfyUI's `tiled_scale` and is written here independently.

No model weight is included. The DraconicDragon NAFNet-VAE-DeGrid weights
(Apache-2.0) and any other NAFNet file must be supplied by the user and remain
subject to their own licenses.

## Detail guidance (pack 1.6.0)

`guidance_detail.py`, `guidance_s2.py`, `guidance_optimal_scale.py` and the adaptive SMC step in
`guidance_dcw.py` follow the sam-extra Forge extension v0.30.0 (by the author of AI Studio Pro).
Much of their code is sam-extra's own code, licensed to this pack under MIT by the author. "Code shared
with the sam-extra Forge extension" names the functions. Their sources:

| Stage | Reference | License | Notes |
| --- | --- | --- | --- |
| HiFlow | [Bujiazi/HiFlow](https://github.com/Bujiazi/HiFlow) `flux_pipeline_hiflow.py` and `utils.py` at commit `31cc2b1c515195d8bfee002d3da58ac7c7773fef`; defaults from `run_hiflow.py` at commit `da351a8ef036384a42485744ba47bc9c1d882b94` | Apache-2.0 | The direction/acceleration alignment and the order-4 Butterworth mask follow the official code, rewritten as a per-evaluation x0 replacement for ComfyUI's post-CFG hook. That rewrite is sam-extra's (`sam3ext/guidance/hiflow.py`, MIT for this pack by its author). The changes are listed in `guidance_detail.py`'s module docstring (Apache-2.0 section 4(b)). The upstream repository ships no NOTICE file. Full license text: `LICENSES/HiFlow-Apache-2.0.txt`. |
| Adaptive SMC | [sorryhyun/anima_lora](https://github.com/sorryhyun/anima_lora) `library/inference/corrections/smc_cfg.py`; CFG-Ctrl, [arXiv:2603.03281](https://arxiv.org/abs/2603.03281) | MIT (Copyright (c) 2026 Seunghyun Ji) | Formula and defaults (alpha 0.2, lambda 5) only; no code is copied. |
| TSR | Temporal Score Rescaling, [arXiv:2510.01184](https://arxiv.org/abs/2510.01184); ComfyUI's `TemporalScoreRescaling` node (GPL-3.0), which sam-extra's `tsr.py` ports | — | sam-extra's per-row code (MIT for this pack by its author). Only the node's formula is shared with ComfyUI (see "Formulas shared with GPL-3.0 hosts"); the node's own code is not included. |
| S²-Guidance | [arXiv:2508.12880](https://arxiv.org/abs/2508.12880) Eq. 4, Algorithm 1 | no official code | Paper reimplementation. |
| Momentum Guidance | [arXiv:2602.20360](https://arxiv.org/abs/2602.20360) Eq. 12–13 | no official code | Paper reimplementation. |
| HiGS | [arXiv:2509.22300](https://arxiv.org/abs/2509.22300) Eq. 6, 8, Algorithms 2–3 | paper code not used | Paper reimplementation. |
| Optimal Scale | CFG-Zero*, [arXiv:2503.18886](https://arxiv.org/abs/2503.18886) | — | The optimized-scale equation only (no zero-init). |

This notice does not replace or modify the upstream licenses.

## Forge inpaint crop rule (`mask_ops.py`)

The SAM3 node's "only masked" inpaint widens the padded mask crop to the processing aspect ratio the way Forge
does (`inpaint_full_res`), so the ComfyUI crop lands on the same pixels. `mask_ops.py`'s `expand_crop_region`
follows the rule of Forge's `modules/masking.py` `expand_crop_region` (A1111 stable-diffusion-webui code,
AGPL-3.0): the short side grows around its centre by the same integer amount, the crop slides back inside the
image at an edge, and an image too small for the ratio cuts it. Only that rule is shared. No Forge code is
included: the function is this pack's own code (one helper, `_grow_span`, for either side), and
`tests/test_comfy_sam3_nodes.py` checks its results against Forge's function over a grid of crops.
`tests/test_comfy_node_pack.py` also checks that no pack function is a near-copy of the code in Forge's `modules/`,
`modules_forge/` (without its third-party `packages/`) or `backend/`.

## Code shared with the sam-extra Forge extension (MIT, by its author)

The sam-extra Forge extension ([UR-al/sam-extra](https://github.com/UR-al/sam-extra), installed as
`forge_sam3_extension`) is distributed under GPL-3.0-only. Several pack modules contain code that sam-extra itself
wrote: functions copied from it, or written the same way statement for statement. **These are sam-extra
(GPL-3.0-only) parts written by the same author (UR-al), licensed to this pack under the MIT License by that
author.** The full text is in `LICENSES/sam-extra-MIT.txt`. The GPL-3.0-only terms of sam-extra do not apply to
these parts in this pack.

The grant covers only code that sam-extra itself wrote. In sam-extra's git history the listed functions come from
the author's own commits (accounts UR_al, UR-al and UR-MJ), and sam-extra's file headers mark every third-party
part. Third-party code inside those sam-extra files keeps its own license, and this pack keeps its notice: HiFlow
(Apache-2.0), Koishi-Star's Euler (SMEA) Dy (Apache-2.0), Clybius' DPM++ 4M SDE (BSD-3-Clause), the SPEED and
Colorcraft upstreams (MIT) and Detail Daemon (MIT); see the sections below. The pack takes no third-party GPL code
from sam-extra. In a few places sam-extra evaluates a published formula the way a GPL-3.0 host (ComfyUI, Forge)
writes it. There the pack shares only that formula, in the same floating-point operations, because its tests check
bit-equal results. These formulas are listed below.

The pack never imports the extension (`tests/test_comfy_pack_foundation.py` checks the imports of every pack
file). The installed extension is also the test oracle: the pack's tests compare it with the pack, bit for bit on
CPU.

The table lists the shared code outside the 1.7.0 ports; the 1.7.0 ports are in the next section. Both tables name
the functions that a token-similarity check of the pack against the installed sam-extra v0.33.0 finds at 0.80 or
above (docstrings and comments ignored; `tests/test_comfy_node_pack.py` keeps the lists complete). The same grant
covers other code of these modules that follows sam-extra's code more loosely. The same code may have been written
for the pack first. Either way, the author wrote both copies.

| Pack module | Pack function(s) | sam-extra source (installed v0.33.0) |
| --- | --- | --- |
| `anima_lora.py` | `_make_expansion_mapping` | `sam3ext/anima_lora_blocks.py` `_make_expansion_mapping` |
| `anima_lllite.py` | `read_safetensors_header`, `tile_repair_size` | `sam3ext/sam3_cn_lllite.py` `read_safetensors_header`; `sam3ext/anima_core.py` `tile_repair_size` |
| `mask_ops.py` | `split_prompt_groups` | `sam3ext/core.py` `_split_prompt_groups` |
| `degrid_files.py`, `degrid_math.py`, `degrid_runner.py` | `safetensors_header`, `_unwrap_state`; `message_says_oom`, `tiled_residual_with_oom_retry`, `check_residual`; `residual` | `sam3ext/vae_degrid_models.py` `read_safetensors_header`, `_unwrap`; `sam3ext/vae_degrid.py` `is_oom_error`, `tiled_residual_with_oom_retry`, `check_residual`; `sam3ext/vae_degrid_runtime.py` `residual` |
| `guidance_common.py` | `_pad_even`, `_haar_dwt`, `_haar_idwt`, `_sigma_norm` | `sam3ext/guidance/haar.py` `pad_even`, `haar_dwt2d`, `haar_idwt2d`, `sigma_norm` |
| `guidance_dcw.py` | `_smc_adaptive_error` | `sam3ext/guidance/cwm_smc.py` `apply_smc_adaptive` |
| `guidance_detail.py` | `first_float`; `tsr_factors`, `apply_tsr`; `step_role`, `apply_mg`, `_dct_matrix`, `dct_highpass`, `apply_higs`; `Trajectory.record`, `Trajectory.at`; `butterworth_mask`, `butterworth_lowpass`, `resize_latent`, `apply_hiflow` | `sam3ext/guidance/sigmas.py` `first_float`; `tsr.py` `rescale_factors`, `apply_tsr`; `history.py` (same names); `trajectory.py` `Trajectory`; `hiflow.py` (same names; sam-extra's modified port of HiFlow, Apache-2.0, see "Detail guidance") |
| `guidance_optimal_scale.py` | `optimal_scale_residual`, `make_callback` (its `optimal_scale`) | `scripts/anima_cfg_optimal_scale.py` `_optimal_scale_residual`, `callback` |

Formulas shared with GPL-3.0 hosts. Each item is the formula only: the code around it is sam-extra's or the pack's,
and the bit-equal tests fix its floating-point operations.

- **Laplace schedule** (`extra_schedulers._laplace_over_range`): the curve
  `exp(μ − β·sign(0.5 − x)·log(1 − 2|0.5 − x| + 1e-5))` clamped to [σmin, σmax] (Hang et al., arXiv:2407.03297), as
  ComfyUI's `get_sigmas_laplace` writes it, evaluated at other positions. The plain schedule is ComfyUI's own
  function, called at run time.
- **TSR** (`guidance_detail.tsr_factors`, `apply_tsr`): `r = (snr·v + 1)/(snr·v/k + 1)` (Xu et al.,
  arXiv:2510.01184, eq. 6), `α = σ·e^λ` and `x0' = lerp(x/α, x0, r)`, as ComfyUI's `TemporalScoreRescaling` node
  writes them; sam-extra's `tsr.py` ports that node. The pack's code is sam-extra's per-row version, not the node's
  tensor code.
- **DPM++ 4M SDE** (`extra_samplers_dpmpp_4m._sample`): the half-log-SNR form of the DPM-Solver++ SDE step. λ comes
  from `sigma_to_half_log_snr` after `offset_first_sigma_for_snr`, `α_t = σ_t·e^{λ_t}`, the step is
  `x ← (σ_t/σ_s)·e^{−hη}·x + α_t·(1 − e^{−h(1+η)})·x0` and the history terms are scaled by `α_t`. ComfyUI's
  `sample_dpmpp_3m_sde` writes it this way on top of k-diffusion's MIT 3M SDE, and sam-extra applies it to Clybius'
  BSD-3-Clause 4M SDE. The helpers are ComfyUI's own, called at run time.
- **Euler (SMEA) Dy CFG++** (`extra_samplers_dy.cfg_pp_target`): the CFG++ Euler step at η = 0,
  `x' = α_t·x0 + σ_t·(x − α_s·x0_uncond)/σ_s` through `to_d` and `get_ancestral_step` (Chung et al.,
  arXiv:2406.08070), as ComfyUI's and Forge's `sample_euler_ancestral_cfg_pp` write it.
- **ER SDE** (`extra_samplers_math.py`): the three noise scale functions h(λ) of ComfyUI's `SamplerER_SDE` node (Cui
  et al., arXiv:2309.06169). The solver is ComfyUI's own `sample_er_sde`, called at run time.
- **IPNDM_V weights** (`extra_samplers_ipndm.weights`): the variable-step weights of zju-pi/diff-sampler's
  `ipndm_v_sampler` (Apache-2.0, notice in the file header; Zhang & Chen, arXiv:2204.13902), as ComfyUI's
  `sample_ipndm_v` writes them (code that ComfyUI marks as zju-pi's, under Apache 2), with the fourth-order weight's
  last factor `h_{n−1}/h_{n−2}` replaced by `h_{n−2}/h_{n−3}` — sam-extra 0.33.1's fix. ComfyUI's own function keeps
  the typo and is not called; the loop and the history are the pack's own code.

## sam-extra ports (pack 1.7.0): sam-extra code under the author's MIT grant, and the test oracle

Pack 1.7.0 brings four features of the sam-extra Forge extension to ComfyUI: Extra Schedulers, Extra Samplers and
Anima SPEED (sam-extra v0.33.0), and Colorcraft (sam-extra v0.32.1). sam-extra is GPL-3.0-only and is by the
author of AI Studio Pro. Much of the pack's code for these features is sam-extra's own code: **sam-extra
(GPL-3.0-only) parts written by the same author (UR-al), licensed to this pack under the MIT License by that
author** (`LICENSES/sam-extra-MIT.txt`, previous section). The table names the reused functions. The upstreams
inside sam-extra keep their own licenses and notices. The formulas shared with GPL-3.0 hosts are listed in the
previous section. No pack module imports the extension (`tests/test_comfy_pack_foundation.py` checks the imports
of every pack file).

The installed extension is also the **test oracle**: the pack's tests run it next to the pack on CPU and compare
the results bit for bit.

| Pack module(s) | sam-extra code in the pack (MIT grant) | Third-party code inside it (own license, notice kept) | sam-extra source, also the test oracle | Tests |
| --- | --- | --- | --- | --- |
| `extra_schedulers.py`, `extra_schedulers_math.py` | `flow_cosmos_rho7`, `_refuse_karras_dynamic_rise` (sam-extra's `_check_karras_dynamic_falls`), `_laplace_over_range` (`_laplace_in_range`), `custom`; `coerce`, `evaluate`, `compile_expression`, `parse_sigma_list`, `sigmas_from_list`, `karras_dynamic_ramp` (`_karras_dynamic_values`), `flow_cosmos_rho7_times` (`_flow_cosmos_times`), `laplace_position` (`_laplace_position`), `sigmas_from_expression`; the rules, ranges and error texts of the nine schedules, the safe-expression language, the sigma-list format and the settings coercion | None. The Laplace curve is a formula shared with ComfyUI (previous section); the published formulas and credits are under "Extra Schedulers" | `sam3ext/extra_schedulers/` (`schedulers.py`, `expression.py`, `sigma_list.py`, `settings.py`, `registry.py`) | `test_comfy_extra_schedulers_parity.py`, `test_comfy_extra_schedulers_runtime.py` |
| `extra_samplers_math.py` | The noise window (a per-step scaler and a gated noise sampler around ComfyUI's own `sample_er_sde`) and the coercion of the five values follow sam-extra's code | The three ER SDE noise scalers: formulas shared with ComfyUI's `SamplerER_SDE` (previous section) | `sam3ext/extra_samplers/er_sde.py`, `er_sde_window.py`, `params.py` | `test_comfy_extra_samplers_parity.py`, `test_comfy_extra_samplers_runtime.py` |
| `extra_samplers_flow.py` | None: three calls of ComfyUI's own `sample_dpmpp_2m_sde` / `sample_dpmpp_3m_sde` at η = 0 | — | `sam3ext/extra_samplers/dpmpp_flow.py` | same |
| `extra_samplers_restart.py` | `restart_kernel`, `_heun_step`; the restart window and step rules (Xu et al., arXiv:2306.14878, Algorithm 1) | The Heun step body is k-diffusion's `sample_heun` (MIT), as ComfyUI ships it | `sam3ext/extra_samplers/restart_flow.py` | same |
| `extra_samplers_dy.py`, `extra_samplers_dy_host.py` | sam-extra's changes to the Dy port: `cfg_pp_target`, `churn`, `evaluate`, `dy_substep`, `smea_substep`, `euler_dy`, the entry functions (`_entry`'s `sampler`), `spatial_interpolate`, `SubstepInputs`, the uncond capture; the sub-step placement and the reasons the sub-steps step aside | Koishi-Star's Euler-Smea-Dyn-Sampler (Apache-2.0, below; notice in the file header). The CFG++ step is a formula shared with ComfyUI and Forge (previous section) | `sam3ext/extra_samplers/euler_dy.py`, `substep_guard.py` | same |
| `extra_samplers_ipndm.py` | `clamp_order` (sam-extra's `_order`, the 1–4 order rule). The rest is not sam-extra's own code: sam-extra's IPNDM_V is zju-pi's code as ComfyUI adapts it. The pack's loop and history (`sample_ipndm_v_fixed`, `step_sizes`, `combine`) are its own and follow sam-extra 0.33.1's rules (order 1 without history, the callback's current latent) | zju-pi/diff-sampler's IPNDM_V weights (Apache-2.0, below; notice in the file header). The weights are a formula shared with ComfyUI (previous section) | `sam3ext/extra_samplers/ipndm_deis.py` (`sample_ipndm_v`, sam-extra ≥ 0.33.1) | `test_comfy_extra_samplers_parity.py`, `test_comfy_extra_samplers_runtime.py` |
| `extra_samplers_register.py`, `extra_samplers_nodes.py` | None. The registration and node code are this pack's own; the twelve sam-extra names, their order and the penultimate-σ discard set are sam-extra's (the thirteenth name, `ipndm_v_fixed`, is the pack's) | — | `sam3ext/extra_samplers/registry.py`, `params.py` | `test_comfy_extra_samplers_pack.py`, `test_comfy_extra_samplers_runtime.py` |
| `extra_samplers_dpmpp_4m.py` | `_sample`: sam-extra's port (momentum 0, half-log-SNR flow handling, the noise rule) | Clybius' DPM++ 4M SDE (BSD-3-Clause, below; notice in the file header). The flow form of the step is a formula shared with ComfyUI (previous section) | `sam3ext/extra_samplers/dpmpp_4m_sde.py` | `test_comfy_extra_samplers_parity.py` |
| `guidance.py`, `guidance_dcw.py`, `guidance_detail.py` (Dy sub-steps) | sam-extra's sub-step rule: the guidance suite keeps its step state (APG momentum, ADG, SMC `e_prev`, RDC, HiFlow, Momentum Guidance, HiGS) through Euler (SMEA) Dy sub-steps | — | `sam3ext/guidance/` | `test_comfy_detail_parity.py` |
| `speed_schedule.py`, `speed_spectral.py`, `speed_runner.py`, `speed_nodes.py`, the Detail Daemon σ lookup in `guidance_dd.py` | `validate_scales`, `parse_sigmas`, `_transition_hand_offs`; `dct_basis`, `dct_downscale`, `amplitude_resize`, `_target` (`_check_target`), `expand_dct`, `expand_fft`, `expand_haar`; `_expand`, `_init_coarse`, `_apply_transition`, `run_speed` with its `segment`; `plan_for` (`_plan_for`), `_flag` (Colorcraft `spec.py` `_truthy`). Also sam-extra's changes to the upstreams: refusals incl. custom A ≤ 0, seeds and noise streams, the img2img rescale, guards, status lines, and Detail Daemon following the re-published σ list | The three MIT upstreams under "Anima SPEED" | `sam3ext/speed/` (`schedule.py`, `spectral.py`, `runner.py`, `forge_host.py`) | `test_comfy_speed_parity.py`, `test_comfy_speed_host.py`, `test_comfy_speed_runtime.py` |
| `colorcraft_basis.py`, `colorcraft_engine.py`, `colorcraft_host.py`, `colorcraft_nodes.py` | sam-extra's three substitutions in `apply_chain`; `encode_anchors`, `_anchor_cache_key` (`_anchor_key`), `AnchorCache.put` (`_remember_anchor`), `RunReport.render`, `grid_downscale`, the anchor look-up `get_color_latent`; sam-extra's host rules: anchors, σ look-up, pre-Detail-Daemon σ, fp16/bf16, sub-step pass-through | The MIT upstream and fork under "Colorcraft": `resolve_dev`, `sigma_to_value`, `build_entries` and the modifier loop are theirs | `sam3ext/colorcraft/` (`engine.py`, `hook.py`, `basis.py`, `spec.py`, `schedule.py`) | `test_comfy_colorcraft_parity.py`, `test_comfy_colorcraft_runtime.py` |
| `guidance_common.py`, `compat.py`, `hires_carry.py` | None in their 1.7.0 additions. Only the values of sam-extra's coordination names (`sam_extra_substep`, `sam_extra_step_offset`, the re-published σ tag) are sam-extra's; the code that writes and reads them is this pack's own. The older Haar helpers of `guidance_common.py` (`_pad_even`, `_haar_dwt`, `_haar_idwt`, `_sigma_norm`) are sam-extra code under the grant (previous section) | — | `sam3ext/extra_samplers/common.py`, `sam3ext/speed/runner.py`, `sam3ext/guidance/sigmas.py` | `test_comfy_pack_foundation.py` |

The sections below give each feature's upstreams and licenses. This notice does not replace or modify the
upstream licenses.

## Extra Schedulers (pack 1.7.0)

`extra_schedulers.py`, `extra_schedulers_math.py`: no third-party source file is included and no third-party code is copied:

- The scheduler rules, value ranges, the safe-expression language, the sigma-list format and the error
  texts are those of the sam-extra Forge extension's "Extra Schedulers". Much of the code is sam-extra's own
  code, licensed to this pack under MIT by its author (see "sam-extra ports (pack 1.7.0)"). The pack does
  not import the extension. The pack's tests run the installed extension next to the pack to compare results.
- Laplace: the plain schedule is ComfyUI's own `comfy.k_diffusion.sampling.get_sigmas_laplace`, called at
  run time inside ComfyUI (not copied). Where its clamp would repeat a sigma, the pack evaluates the same
  published curve — Hang et al., "Improved Noise Schedule for Diffusion Training", arXiv:2407.03297 — at
  other positions, as sam-extra does. It uses the curve expression of ComfyUI's function, in the same
  floating-point operations, so the result matches sam-extra bit for bit. This is a formula shared with
  ComfyUI (see "Formulas shared with GPL-3.0 hosts").
- The sigma list's log-linear interpolation is ComfyUI's own
  `comfy_extras.nodes_align_your_steps.loglinear_interp`, called at run time (not copied).
- React Cosinusoidal DynSF is written here from its published one-line formula (yoinked-h; added to
  Panchovix/stable-diffusion-webui-reForge, AGPL-3.0, by commit 25a5fb38f). No reForge code was read,
  copied or used — the label is reForge's, so its infotexts paste.
- Flow Cosmos rho7: the name `flow_cosmos_rho7` and the idea (Cosmos-Predict2's ρ schedule in rectified-flow
  time for Anima) are KeithZ117/Comfyui-anima-sampler's (https://github.com/KeithZ117/Comfyui-anima-sampler,
  commit `effba3c5`, MIT) — credit only: none of its code is included or copied (the app's parity test checks
  the pack against the numbers of its `build_flow_cosmos_rho_sigmas`, outside the pack). The pack writes the
  schedule from the formula: Karras et al., "Elucidating the Design Space of Diffusion-Based Generative
  Models", arXiv:2206.00364 (the ρ ramp, eq. 5), over a VE-equivalent noise range σ̃ — by default
  Cosmos-Predict2's EDM noise range 0.002–80 with ρ 7; the node's `flow_cosmos_rho` / `flow_cosmos_sigma_max` /
  `flow_cosmos_sigma_min` values and their ranges, checks and texts follow sam-extra v0.33.0 (sam-extra's code,
  under the author's MIT grant) — mapped to flow time t = s/(1 + s). On a model that is not a flow model it calls ComfyUI's own
  `comfy.k_diffusion.sampling.get_sigmas_karras` with that ρ at run time (not copied).
- Flow Cosmos Dynamic: sam-extra v0.33.0's own composition of its Karras Dynamic and Flow Cosmos rho7 (not taken
  from another project): Karras Dynamic's per-step exponent ρ + 2cos(2πi/n) on Flow Cosmos rho7's σ̃ range, the
  same flow-time map, the same three values, checks and texts (sam-extra's code, under the author's MIT grant). On a model that is not a flow model it is the pack's own Karras Dynamic with that ρ.
- Karras Dynamic uses the Karras/EDM ramp (Karras et al., arXiv:2206.00364) with a per-step exponent;
  CosineExponential blend uses k-diffusion's exponential schedule formula; Phi credits the idea of
  Extraltodeus's "Golden Scheduler" (github.com/Extraltodeus/sigmas_tools_and_the_golden_scheduler). These
  are public formulas; no code of those projects, nor of aoleg/Neo_ExtraSchedulers or
  DenOfEquity/webUI_ExtraSchedulers, was read or used.

## Extra Samplers (pack 1.7.0)

The `extra_samplers_*.py` modules give ComfyUI the samplers the sam-extra Forge extension v0.33.0 (by the author of
AI Studio Pro, GPL-3.0-only) adds to Forge's sampler list. Much of their code is sam-extra's own code, licensed to
this pack under MIT by its author (see "sam-extra ports (pack 1.7.0)"). The pack never imports the extension, and
the installed extension is the test oracle (`tests/test_comfy_extra_samplers_parity.py` runs it next to the pack on
CPU). Three modules are ports of permissive upstreams (two in sam-extra's version), carrying their notices in the
file header:

| File | Upstream | Commit | License |
| --- | --- | --- | --- |
| `extra_samplers_dy.py` (Euler Dy / Euler SMEA Dy and their CFG++ entries) | [Koishi-Star/Euler-Smea-Dyn-Sampler](https://github.com/Koishi-Star/Euler-Smea-Dyn-Sampler) `smea_sampling.py` (`sample_euler_dy`, `sample_euler_smea_dy`, `dy_sampling_step`, `smea_sampling_step`, `_Rescaler`) | `d98a504c8419be5274068ad91ca6bbf2e13635a8` | Apache-2.0, Copyright 2024 KBlueLeaf — `LICENSES/Euler-Smea-Dyn-Sampler-Apache-2.0.txt`. Modified (Apache-2.0 §4(b)): every change is listed in the module docstring |
| `extra_samplers_dpmpp_4m.py` (DPM++ 4M SDE) | [Clybius/ComfyUI-Extra-Samplers](https://github.com/Clybius/ComfyUI-Extra-Samplers) `extra_samplers.py` `sample_clyb_4m_sde_momentumized` | `52eac1b7c847d2727e0ca93ca26d9ffd77029daa` | BSD-3-Clause, Copyright (c) 2024, Clybius — `LICENSES/ComfyUI-Extra-Samplers-BSD-3-Clause.txt` |
| `extra_samplers_ipndm.py` (`ipndm_v_fixed` — IPNDM_V, `weights`) | [zju-pi/diff-sampler](https://github.com/zju-pi/diff-sampler) `diff-solvers-main/solvers.py` `ipndm_v_sampler`, as ComfyUI's `sample_ipndm_v` adapts it (comfyanonymous/ComfyUI `387f98aa`, marked there as zju-pi's code under Apache 2) | `68d5ce427f261962b89ce3b0ee8f6b29f0577328` | Apache-2.0 — `LICENSES/diff-sampler-Apache-2.0.txt` (the standard Apache License 2.0 text; that repository's LICENSE file was not read here, so no copyright line of it is reproduced — the repository and the commit are named, as ComfyUI does). Modified (Apache-2.0 §4(b)): the fourth-order weight's last factor (sam-extra 0.33.1's fix) and the other changes listed in the module docstring |

Other modules (no other third-party code copied):

- `extra_samplers_math.py` — the five values' coercion and the three ER SDE samplers. The solver is ComfyUI's own
  `comfy.k_diffusion.sampling.sample_er_sde`, called at run time inside ComfyUI (not copied); the three noise scale
  functions are the published formulas (Cui et al., "Elucidating the Solution Space of Extended
  Reverse-Time SDE for Diffusion Models", arXiv:2309.06169; the η-scaled ER-SDE scaler is the one ComfyUI's
  `SamplerER_SDE` node uses since ComfyUI 40e46c71), as that node writes them (a formula shared with ComfyUI). The idea of restricting ER-SDE's noise to a window of the
  schedule is pamparamm/ComfyUI-ppm's `er_sde_scheduled` (AGPL-3.0); its code was not opened or copied — the pack
  wraps ComfyUI's solver with a per-step noise scaler and a gated noise sampler instead.
- `extra_samplers_flow.py` — DPM++ 2M / 2M Heun / 3M (flow ODE): three calls of ComfyUI's own
  `sample_dpmpp_2m_sde` (`solver_type` midpoint / heun) and `sample_dpmpp_3m_sde` at η = 0, at run time inside
  ComfyUI (not copied). DPM-Solver++: Lu et al., arXiv:2211.01095; the Heun (φ₂) correction is k-diffusion's
  (crowsonkb/k-diffusion, MIT), already part of ComfyUI's function.
- `extra_samplers_restart.py` — Restart (flow), written from the paper: Y. Xu et al., "Restart Sampling for
  Improving Generative Processes", NeurIPS 2023, arXiv:2306.14878 (Algorithm 1), with the restart window and step
  rules sam-extra uses. The paper's reference repository (Newbeeer/diffusion_restart_sampling) has no license and was
  not used; Forge's built-in Restart (A1111 code) was not used either. The Heun steps follow ComfyUI's own
  `sample_heun` loop body (k-diffusion, MIT; `to_d` called at run time). The code is sam-extra's own
  (`restart_flow.py`: `restart_kernel`, `_heun_step`), licensed to this pack under MIT by its author; the installed
  extension is also the test oracle (bit-equal on CPU, `tests/test_comfy_extra_samplers_parity.py`).
- `extra_samplers_dy_host.py` (when the Dy sub-steps step aside) follows sam-extra's sub-step guard (MIT grant);
  `extra_samplers_register.py` (registration in ComfyUI's sampler tables, never replacing another provider's name,
  and the penultimate-σ discard Impact's FaceDetailer gets through `scheduler_func_opt`) and
  `extra_samplers_nodes.py` (`ForgeNeoExtraSamplerSettings`) are this pack's own code.
- The 4M port's flow-model handling and the Dy port's CFG++ update use ComfyUI's own k-diffusion helpers
  (`sigma_to_half_log_snr`, `offset_first_sigma_for_snr`, `to_d`, `get_ancestral_step`,
  `set_model_options_post_cfg_function`, `BrownianTreeNoiseSampler`, `default_noise_sampler`), called at run time.

This notice does not replace or modify the upstream licenses.

## Anima SPEED (pack 1.7.0)

`speed_schedule.py`, `speed_spectral.py`, `speed_runner.py` and `speed_nodes.py` reproduce the "Anima SPEED"
sampler wrapper of the sam-extra Forge extension v0.33.0 (by the author of AI Studio Pro, GPL-3.0-only) on
ComfyUI. They follow sam-extra's code: the three MIT upstreams below, which keep their notices, plus sam-extra's
own changes to them. Those changes are sam-extra code, licensed to this pack under MIT by its author (see
"sam-extra ports (pack 1.7.0)"). They are checked against the installed extension on CPU
(`tests/test_comfy_speed_parity.py`; the pack never imports the extension). The module docstrings list what
each module derives from which upstream.

| Upstream | Commit | Used for | License |
| --- | --- | --- | --- |
| [howardhx/speed](https://github.com/howardhx/speed) (official SPEED, arXiv 2605.18736) | `ca7801c9bdffe681742e9592345bcf4885959be5` | `utils.py`: power spectrum, activation time (Eq. 9), transition times (Eq. 10), scale validation, `kappa`/`align_timestep` (Eq. 5/6), the DCT/DWT/FFT noise expansions; `comfyui/speed_sampler.py`: the initial DCT downscale, the segmented run (segment callback, expansion seed `seed + (k + 1) * 10000`, only the hand-off sigma patched) | MIT, Copyright (c) 2026 Howard Xiao — `LICENSES/speed-MIT.txt` |
| [aoleg/ComfyUI-SPEED](https://github.com/aoleg/ComfyUI-SPEED) (fork of ruwwww/ComfyUI-SPEED) | `a8873591a27f2c1e086a2caf546f9b6aeec62b81` | `speed_core.py`: preset table, reference latent, coarse-share backstop, adaptive delta, `sigma_divisor` (neo_shift), transition lookup; `spectral_utils.py`: `equivalent_delta`, `reference_coarse_fraction`; `sample_speed_core`: sampler kwargs to every segment, the full-grid noise sampler only to the full-size segment, 5-D latents; `scripts/speed_forge.py`: shape-safe coarse noise (`_ShapeSafeRandn`), the inpaint / reference-latent guards, wrapping the sampler function for one pass (`speed_nodes.py`) | MIT, Copyright (c) 2026 A. Izzuddin Al Faruq (the presets, adaptive delta and neo_shift are Oleg Afonin's work in that fork) — `LICENSES/ComfyUI-SPEED-MIT.txt` |
| [sorryhyun/ComfyUI-Spectrum-KSampler](https://github.com/sorryhyun/ComfyUI-Spectrum-KSampler) | `b46a364aec3b161b889c9cc26cd976a49eb537ae` | `spd.py`: one sigma per hand-off, the respace loop and its sigma arithmetic, the expansion generator `seed + 10_000`; `_vendor/networks/spd_core.py`: the cached DCT matrix, `_snap`, the respace alignment | MIT, Copyright (c) 2026 sorryhyun — `LICENSES/ComfyUI-Spectrum-KSampler-MIT.txt` |

The aoleg fork's README names the official repository as BSD 3-Clause, but the `LICENSE` file of
the pinned official commit is MIT, which is the license followed here.

The pack's legacy `speed_*` inputs on `ForgeNeoKSamplerCNS` (the external sorryhyun
`SpectrumSPDKSampler` provider) are unrelated to this port and unchanged.

This notice does not replace or modify the upstream licenses.

## Colorcraft (pack 1.7.0)

The Colorcraft port reproduces the "Colorcraft" sampling-time colour grading of the sam-extra Forge
extension v0.32.1 (by the author of AI Studio Pro, GPL-3.0-only) on ComfyUI. The vendored files are the MIT
upstream's own. The pack's modules follow the MIT upstream and fork below, and sam-extra's code on them. That
code covers the three substitutions in the modifier loop and the host rules: Base/Hires passes, sigma look-up,
pre-Detail-Daemon sigma, anchors encoded before sampling, fp16/bf16 handling and sub-step pass-through. It is
licensed to this pack under MIT by its author (see "sam-extra ports (pack 1.7.0)") and checked against the
installed extension on CPU (`tests/test_comfy_colorcraft_parity.py`; the pack never imports the extension).

| Upstream | Commit | Used for | License |
| --- | --- | --- | --- |
| [muerrilla/ComfyUI-Colorcraft](https://github.com/muerrilla/ComfyUI-Colorcraft) | `d28ac6a4e997d0f8a2f1a60b7361b561c4a15bbf` | Vendored unchanged in `vendor/comfyui_colorcraft/` (see its `UPSTREAM.md`): `lib_colorcraft/__init__.py`, `vectors.py`, `masking.py`, `color.py`, `schedule.py` and `vectors/colorcraft-krea2.safetensors`, `colorcraft-zimage.safetensors`. Written again from it: the modifier loop of `nodes.py` (`ColorcraftSampler.wrap` → `post_cfg_function`, the per-entry schedule of `wrapped_sampler_function`, `sigma_to_value`) in `colorcraft_engine.py`; `lib_colorcraft/basis.py`'s `load_basis`, `resolve_dev` and the krea2/zimage calibration rows in `colorcraft_basis.py` | MIT, Copyright (c) 2026 Sahand Ahmadian Tehrani (Muerrilla) — `LICENSES/ComfyUI-Colorcraft-MIT.txt` |
| [aoleg/ComfyUI-Colorcraft](https://github.com/aoleg/ComfyUI-Colorcraft) (fork, Oleg Afonin) | `f00066c63c9d8f96abc119cada3b51b36688fcac` | Vendored unchanged in `vendor/comfyui_colorcraft_fork/`: `vectors/colorcraft-flux2.safetensors`. Written again from its `lib_colorcraft/core.py` in `colorcraft_basis.py`: `to_model_space(latent_format, anchor, dims)`, `family_for_latent_format`, the `Flux2` latent-format row, the per-family VAE downscale table and the `flux2` calibration row | MIT (the fork's `LICENSE` keeps the original notice, Copyright (c) 2026 Sahand Ahmadiantehrani (Muerrilla)) — `LICENSES/ComfyUI-Colorcraft-aoleg-MIT.txt` |

The pack's own modules `colorcraft_basis.py`, `colorcraft_engine.py`, `colorcraft_host.py` and
`colorcraft_nodes.py` name in their docstrings which upstream function each part follows. The two
`LICENSES/` texts are the upstream and fork `LICENSE` files at the pinned commits, unchanged.

This notice does not replace or modify the upstream licenses.
