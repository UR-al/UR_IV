# AI Studio Forge Neo parity nodes

This node pack is bundled with AI Studio Pro and installed into the selected
ComfyUI `custom_nodes` directory. Part of it is the execution layer used by
the app's default ComfyUI workflow compiler (no workflow JSON is required for
normal T2I, I2I, inpaint, upscale, ADetailer, or SAM3 jobs); the rest are
public nodes for user-built ComfyUI workflows. See "Nodes by consumer".

## Nodes by consumer

### Built by the app compiler

`core/comfy_workflow_compiler.py` (and the Creator H3 path in
`core/creator_workflows.py`) emits these nodes. The compiler follows Forge's
`save_images`: only payloads that send `save_images=True` (the main
T2I/I2I/inpaint generation) end in Comfy's core `SaveImage`, leaving a copy in
ComfyUI/output. Every other compiled graph (post-processing, ADetailer/SAM3/
Refine, upscale, SAM3 mask-only, chat, hand repair) ends in core
`PreviewImage`, so its result is returned through ComfyUI's temp folder and
read back the same way. Creator workflows keep their own `SaveImage` nodes.

- Loaders/prompts: `ForgeNeoAnimaQwen35Loader`, `ForgeNeoAnimaQwen35Prompt`,
  `ForgeNeoAnima38V2Loader`, `ForgeNeoAnima38V2Prompt`,
  `ForgeNeoAnimaLoraLoader`
- Model patches/guidance: `ForgeNeoModelSamplingShift`, `ForgeNeoNegPip`,
  `ForgeNeoSkimmedCFG`, `ForgeNeoAnimaGuidanceSuite`,
  `ForgeNeoAnimaDetailDaemon`
- Sampling: `ForgeNeoLatentInput`, `ForgeNeoKSamplerCNS`, `ForgeNeoHiresFix`
- Detailers: `ForgeNeoADetailer`, `ForgeNeoSAM3Mask`, `ForgeNeoSAM3Detailer`,
  `ForgeNeoSAM3Refine`
- Creator H3 conditioning cache: `ForgeNeoH3ConditioningCachePrepare`,
  `ForgeNeoH3ConditioningCacheLoad`

### Custom ComfyUI workflows only

The app compiler never emits these. They stay registered because saved user
workflows and the embedded ComfyUI editor reference them, and the workflow
inspector recognises them.

- `ForgeNeoAnimaLoraLoaderModelOnly`, `ForgeNeoLoraBlockWeight`
- `ForgeNeoAnimaDAVE`, `ForgeNeoAnimaModGuidance`, `ForgeNeoAnimaSafePAG`,
  `ForgeNeoDCWCWMSMC`
- `ForgeNeoCNSSamplerPatch` (SAMPLER -> SAMPLER, the inputs of
  namemechan/comfyui-cns_sampler_patch; use with `SamplerCustomAdvanced`)
- `ForgeNeoCharacterReference`, `ForgeNeoReferencePrompt`,
  `ForgeNeoReferenceOutput`, `ForgeNeoMaskSelector`
- `ForgeNeoAnimaPiD`, `ForgeNeoAnimaVAE2x`, `ForgeNeoSAM3TileRepair`
- `ForgeNeoAnimaTileRepair` (Anima Tile & Repair ControlNet-LLLite, pack
  1.4.0+)
- `ForgeNeoSaveImage`, `AIStudioRelight`

When the app generates from an ANIMA custom workflow it keeps
`ForgeNeoLoraBlockWeight` and `ForgeNeoAnimaLoraLoaderModelOnly` in the active
model chain as they are (they already handle the 28/40/52-block layouts) and
remaps only a core `LoraLoader` to `ForgeNeoAnimaLoraLoader`. Other LoRA nodes
(including core `LoraLoaderModelOnly`) are rejected before queueing.

## 1.4.1 changes

- DAVE + Detail Daemon: Detail Daemon hands the model a scaled sigma, which is on no schedule, so
  the original DAVE gate (off schedule = step 0 = on) ran DAVE on every step and the image fell
  apart — the two original nodes chained do the same. Detail Daemon now notes the sampler's own
  sigma (`transformer_options["ai_studio_pre_dd_sigmas"]`) and DAVE judges its steps by it. Toggle:
  `ForgeNeoAnimaDAVE.pre_dd_sigma` / suite setting `guid_dave_pre_dd` (default on; off = the
  originals' behaviour). Runs without Detail Daemon are unchanged.

## 1.3.0 changes

- SAM3 Detailer/Refine "only masked" now follows Forge `inpaint_full_res`:
  the padded mask crop is widened to the processing aspect ratio
  (`expand_crop_region`) and sampled at the processing size
  (`target_width`/`target_height`, or the custom size), then scaled back into
  the crop. `0` keeps the old crop-size sampling for saved workflows.
- IMAGE/MASK normalisation only rescales integer tensors; float bicubic
  overshoot (values slightly above 1.0) is clamped instead of divided by 255.
- A loaded SAM3 bundle is kept in CPU RAM between runs (`unload_after`) and
  moved back to the device instead of being re-read from disk. The
  `ForgeNeoSAM3Mask` input `cache_model=False` loads per run and frees the
  kept copy; the app sets it from its "keep SAM3 in RAM" setting (like Forge's
  `sam3_unload_keep_in_ram`), and `unload_after=False` always keeps the
  bundle on its device. ComfyUI's "unload all models" (`/free`, which the
  app's unload-after-generation sends, OOM recovery, `--disable-smart-memory`)
  also releases the kept bundle. A failed move back to the device drops the
  half-moved bundle instead of caching it.
- Standalone SAM3/Refine sample "only masked" at the input image size, like
  Forge's standalone img2img (the app pins `target_width`/`target_height`).
- App compiler (same release): graphs honour `save_images` — `SaveImage` only
  for `save_images=True` (main generation), core `PreviewImage` (temp output)
  for everything else, so post-processing, chat and hand-repair results no
  longer accumulate in ComfyUI/output. Input uploads are named by content
  hash (`input_<sha256[:32]>`, `krea2_source_…`, `krea2_reference_…`) with
  `overwrite=false`, so re-processing the same image reuses one input file.
- The H3 conditioning cache keeps the diffusion model loaded on a cache hit,
  persists model digests across restarts, and serves its HTTP routes off the
  event loop.

## 1.1.2 compatibility fixes

- Inpaint masks use the source image's crop/contain geometry.
- PAG blends the selected self-attention output toward each token's value
  after normalization, and runs only during the weak prediction.
- Semantic v2 cached conditionings own their runtime records; changing the
  positive prompt no longer expires a cached negative after 64 registrations.
  Discarded conditionings release their records without disabling Comfy's cache.
- The app compiler resolves full model/LoRA/TE/VAE paths before filename
  fallbacks, rejects ambiguous names, and honors learned-upscaler factors.
- Standalone detail operations retain model/semantic settings. Custom workflows
  retain separate negative encoders, and bundled samplers/encoders are recognised
  by the workflow picker.

The pinned files under `vendor/` are unchanged. The cache-lifetime adapter is
local integration code in `anima38_cache.py`. Restart a running Comfy backend
after the updated pack is installed.

## Node groups

- Generation: native checkpoint/split-model loading support, ANIMA-aware LoRA
  loading and block weight, latent input, CNS sampling, hires fix, Anima VAE
  2x, character reference, ADetailer, and metadata-aware image saving. Forge's
  `ER SDE` label maps to Comfy's native `er_sde`; `Beta57 (RES4LYF)` is
  registered as the exact beta schedule with alpha 0.5 and beta 0.7 rather
  than being approximated by Comfy's stock 0.6/0.6 `beta` schedule. Flow-shift
  patching preserves the loaded model's timestep multiplier (1.0 for Anima)
  instead of resetting it to SD3's 1000-unit scale.
- Guidance: NegPiP, DAVE, modulation guidance, Skim CFG, PAG/SEG/SLG,
  APG/CWM/SMC/DCW/RDC, adaptive guidance, and Detail Daemon compatibility.
  `ForgeNeoAnimaDetailDaemon` (pack 1.4.0+) behaves exactly like the
  original "Detail Daemon Sampler" node of
  [`Jonseed/ComfyUI-Detail-Daemon`](https://github.com/Jonseed/ComfyUI-Detail-Daemon)
  at `3394e44`, whose schedule, sigma lookup and sampler wrapper it copies
  unchanged (MIT, `LICENSES/ComfyUI-Detail-Daemon-MIT.txt`; the schedule
  function is Jonseed's port of muerrilla's `make_schedule`, MIT,
  `LICENSES/sd-webui-detail-daemon-MIT.txt`). As a MODEL patch
  it wraps every sampling run of the patched model (a `SAMPLER_SAMPLE`
  wrapper) in that node's sampler: each model call's sigma is looked up in the
  sampler's sigma list, interpolated between neighbours, and scaled by
  `max(1e-06, 1 - schedule * 0.1 * cfg)`. `settings_json` carries the node's
  values under `dd_amount`, `dd_start`, `dd_end`, `dd_bias`, `dd_exponent`,
  `dd_start_offset`, `dd_end_offset`, `dd_fade` and `dd_smooth` (missing keys
  use the node's defaults, values outside its ranges are refused); presets and
  the older `dd_amount_scale`/`dd_schedule`/`dd_multiplier`/`dd_cfg_couple`
  keys are ignored. `cfg_scale_override` is the node's own input (0 = the
  sampler's CFG). The app compiler gives the patched model to the passes
  Forge runs it on with muerrilla's script and with the sam-extra extension:
  the chosen main pass (base, or hires with Hires Pass), ADetailer when that
  pass was the last main pass, and SAM3 detailer passes without Hires Pass
  (with the SAM3 pass's own CFG and sampler); standalone post-processing gets
  none. Packs before 1.4.0 read the same settings 10 times
  stronger; their node lacks `cfg_scale_override`, so the app refuses them
  before queueing.
- SAM3: text/manual mask composition, mask-only output, sequential inpaint,
  independent-result Refine, Comfy masked-region repair, ControlNet hand-off,
  face restore, overlays, and artifacts.
- Anima 3.8B: Qwen3.5 4B loading, progressive-cross v1 conditioning,
  bundled Semantic Connector v2 loading and timestep-aware conditioning. The
  28/40/52-block LoRA adapter is shared by normal, model-only Character
  Reference, and block-weight paths. ANIMA block weighting uses one value per
  active DiT block (28, 40, or 52), with an optional leading base value for
  non-block model and text-encoder keys; other architectures retain Inspire
  Pack's native vector behavior.

## Optional provider nodes

The pack resolves installed ComfyUI providers at execution time. A feature that
needs a provider fails with an explicit node/dependency message instead of being
silently ignored. Depending on the enabled options, providers include Easy SAM3,
Impact Pack/Subpack, ControlNet auxiliary preprocessors, Spectrum, Inspire Pack,
and CLIPNegPip.

SAM3 model weights and all other model/LoRA files are deliberately excluded.
They continue to come from the model directories configured in AI Studio.

The SAM3 behavior is ported from the user's Forge extension
`forge_sam3_extension` (0.21.x lineage). Some Forge-only hook mechanics are
translated to ComfyUI model/conditioning patches; the node report JSON records
fallbacks or semantic adaptations made at runtime.

Forge's vendored Anima/PiD/LLLite Tile Repair stack is not presented as the
same feature: `SAM3 Region Repair (Comfy)` handles the portable masked-region
subset, while Forge-only Tile Repair settings fail with an explicit dependency
message instead of silently running a different pipeline.

## Anima ControlNet-LLLite (pack 1.4.0+)

Anima ControlNet-LLLite weights (kohya sd-scripts v2 format, safetensors keys
`lllite_conditioning1.*`, e.g. Tile & Repair from civitai 2708551) are applied
by the unmodified node of
[`kohya-ss/ComfyUI-Anima-LLLite`](https://github.com/kohya-ss/ComfyUI-Anima-LLLite)
at `b7495bd` (`vendor/comfyui_anima_lllite/`, Apache-2.0,
`LICENSES/ComfyUI-Anima-LLLite-Apache-2.0.txt`). The weights are read from
ComfyUI's `controlnet` folder, as that node does.

- `ForgeNeoAnimaTileRepair` runs the Tile & Repair pipeline of kohya
  sd-scripts `anima_minimal_inference_control_net_lllite.py` (which the Forge
  extension's Tile-Repair panel runs too): the output keeps the source aspect
  ratio (short side = `short_side`, each side rounded down to a multiple of 32,
  at least 256), the control image is resized to that size with PIL bicubic
  like the script, sampling starts from pure noise (denoise 1.0), and the
  result is VAE-decoded. Sampling follows the script's `generate_body`: the
  initial latent is its bf16 draw from a CPU `torch.Generator` seeded with
  `seed` (not Comfy's fp32 `prepare_noise`), the σ list is its shifted
  linspace (`get_timesteps_sigmas`: `steps + 1` points of `linspace(1, 0)`
  under `flow_shift`, not a Comfy scheduler, whose 1000-entry table drifts
  from it when 1000 is not a multiple of `steps`), and each step is Comfy
  `euler` with CFG `cfg`. The defaults are the script's: 50 steps, flow shift
  5.0, CFG 3.5 and an empty negative. `strength`, `start_percent`,
  `end_percent` and `preserve_wrapper` are the original node's inputs with its
  ranges. Only 3-channel (RGB) LLLite files are listed; 4-channel inpaint
  LLLites need a mask.
- `ForgeNeoSAM3Detailer`/`ForgeNeoSAM3Refine`: when the ControlNet model is an
  Anima LLLite (read from its header), it is applied per pass as that node's
  MODEL patch instead of going to `ControlNetLoader`, which cannot read it.
  The pass image is the control image and the pass mask is passed only to
  4-channel (inpaint) weights. Tile & Repair LLLites always get preprocessor
  `None`, other Anima LLLites never an `inpaint_*` one (the report records
  the override). Under `None`, `threshold_a`/`threshold_b` left from an
  earlier preprocessor are ignored like Forge's `None` preprocessor does
  (reported as `thresholds_ignored`). The kohya node uses only the first
  frame of its control image and mask, so an image batch is patched and
  sampled one image at a time (`lllite_per_image`), each with its own
  control image and `batch_index`, like Forge running SAM3 per image.
  `control_mode` does not apply to an LLLite (Forge's built-in LLLite
  patcher has none either).

## Anima 3.8B provenance

The Anima 3.8B Comfy runtime is pinned from
`GumGum10/comfyui-anima-3-8B` commit
`381c13af328b958febf86c155d2f4b007cd0f55b`. Model, text-encoder, adapter,
LoRA, and VAE weights are not bundled; they are resolved from the model paths
configured in AI Studio. The exact copied-file boundary and upstream hashes
are recorded in `vendor/comfyui_anima_3_8b/UPSTREAM.md`.

The runtime code is MIT-licensed and the unmodified Qwen3.5 tokenizer assets
are Apache-2.0. Full license texts and attribution are included under
`LICENSES/` and `THIRD_PARTY_NOTICES.md`.

The 28-to-40 LoRA block lineage is derived from Anima-2.9B's published
`expand_manifest.json`; the 40-to-52 lineage is derived from the Anima 3.8B
checkpoint metadata. The node pack stores only those integer layout facts and
locally derives expansion, contraction, and composed mappings in every
direction. See `THIRD_PARTY_NOTICES.md` for the pinned metadata provenance.
