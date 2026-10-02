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
  `ForgeNeoAnimaDetailDaemon`, `ForgeNeoAnimaOptimalScale` (pack 1.6.0+; before
  the suite so its post-CFG function runs first)
- Sampling: `ForgeNeoLatentInput`, `ForgeNeoKSamplerCNS`, `ForgeNeoHiresFix`
- Detailers: `ForgeNeoADetailer`, `ForgeNeoSAM3Mask`, `ForgeNeoSAM3Detailer`,
  `ForgeNeoSAM3Refine`
- Creator H3 conditioning cache: `ForgeNeoH3ConditioningCachePrepare`,
  `ForgeNeoH3ConditioningCacheLoad`
- Final image post-process: `ForgeNeoAnimaVAEDeGrid` (pack 1.5.0+; after the
  last image extension, before the save/preview node, main generations only —
  see "Anima VAE DeGrid")

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

## 1.6.0 changes

The detail guidance of the sam-extra Forge extension v0.30.0, with its equations, defaults, clamps
and order (`tests/test_comfy_detail_parity.py` runs the pack and the installed extension on the same
tensors on CPU):

- `ForgeNeoAnimaGuidanceSuite` reads the new suite keys:
  - **S²-Guidance** (`guid_slg_mode` "Stochastic (S²)", `guid_s2_*`, `guid_s2_seed`): SLG's weak row
    skips a fresh random draw of blocks at every evaluation, `random.Random("s2:<seed>:<pass>:<draw>")`
    as in the extension, with omega as SLG's scale and the S² window as a step fraction
    (`guidance_s2.py`).
  - **Adaptive SMC** (`guid_smc_mode` "Adaptive sign", `guid_smc_adaptive_alpha`/`_lambda`): the SMC
    step of the CFG stage becomes the adaptive sign controller (`guidance_dcw._smc_adaptive_error`).
  - **TSR, Momentum Guidance, HiGS, HiFlow** (`guid_tsr_*`, `guid_mg_*`, `guid_higs_*`,
    `guid_hiflow_*`): one post-CFG function after PAG/SEG/SLG and before DCW, HiFlow -> MG -> HiGS
    -> TSR (`guidance_detail.py`). HiFlow records the base pass's final x0 (a function after DCW) and
    aligns the Hires.fix pass with it.
- `ForgeNeoKSamplerCNS`/`ForgeNeoHiresFix` tag their runs `base`/`hires` (transformer_options
  `forge_neo_pass`) on a MODEL the suite marked for S² or HiFlow; any other MODEL is sampled as is.
  An untagged run is a base run (a custom workflow's own sampler before the app's `ForgeNeoHiresFix`),
  so HiFlow also works there; the app turns HiFlow on for txt2img Hires.fix generations only.
- New `ForgeNeoAnimaOptimalScale`: the extension's experimental "Anima Optimal Scale" (CFG-Zero*'s
  optimized scale without zero-init) with its skip rules.
- The suite's `settings_json` hides the new keys from the input-contract check, so the app refuses a
  graph that turns one of them on when ComfyUI's `/object_info` has no `ForgeNeoAnimaOptimalScale`
  (a 1.5.0 pack would ignore them silently).
- Host differences: HiFlow cannot align a Hires.fix pass that loads another checkpoint (that MODEL has
  no suite); a second-order sampler's corrector evaluation is placed in the S² window by its sigma.

## 1.5.0 changes

- New `ForgeNeoAnimaVAEDeGrid`: the sam-extra Forge extension's "Anima VAE DeGrid
  (NAFNet)" step as a ComfyUI node (see "Anima VAE DeGrid"). It is a local
  implementation checked against numbers produced by the extension itself
  (`tests/test_comfy_degrid_origin.py`); no extension code is included.
- The pack registers a `degrid` model folder (`ComfyUI/models/degrid`, merged into
  an existing `degrid` entry from `extra_model_paths.yaml` or another pack).
- Packs before 1.5.0 do not have the node; the app then leaves DeGrid out of the
  graph with a notice instead of failing the generation.

## 1.4.2 changes

- SAM3 on OpenCV 5.0: the IMAGE from `VAEDecode` is a `movedim(1, -1)` view whose strides survive
  the mask blend, and OpenCV 5.0 refuses such a non-C-contiguous array as a drawing target
  (`cv2.rectangle`: "Layout of the output array img is incompatible with cv::Mat"), so every
  in-generation `ForgeNeoSAM3Mask` failed while drawing its preview boxes. The preview, the convex
  hull (`fillPoly` into a `zeros_like` copy of a transposed mask had the same problem), the
  edge-aware outline, dilation and the relight blur now hand OpenCV C-contiguous arrays. Results
  are unchanged for contiguous inputs.
- DAVE / SLG block wrappers: ComfyUI leaves a finished run's object patches applied, so reading
  `blocks[i].forward` wrapped the previous run's wrapper and results depended on the run history
  (a previous DAVE with a different tau or pre-DD setting kept running inside the new one). The
  wrapped `forward` now comes from `ModelPatcher.get_model_object`, i.e. the original block
  method, or an upstream DAVE/SLG node's patch in the same graph, which is chained like every
  ComfyUI object patch.

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
- Guidance: NegPiP, DAVE, modulation guidance, Skim CFG, PAG/SEG/SLG (and S²),
  APG/CWM/SMC (unit-L2 or adaptive)/DCW/RDC, adaptive guidance, the detail stages
  (TSR, Momentum Guidance, HiGS, HiFlow), Optimal Scale and Detail Daemon compatibility.
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

## Anima VAE DeGrid (pack 1.5.0+)

`ForgeNeoAnimaVAEDeGrid` removes the Qwen/Wan VAE grid pattern with a
DraconicDragon NAFNet-VAE-DeGrid model, the way the sam-extra Forge script
"Anima VAE DeGrid (NAFNet)" does it (extension commit `395854b`):

- The model's raw forward output is a residual that is added to the image:
  `Full` = `x + s·δ`, `Dark Pixels Mainly` = `x + s·max(δ, 0)`,
  `Bright Pixels Mainly` = `x + s·min(δ, 0)`, then one clamp to [0, 1].
  Strength 0-1.5 (0 = recorded but nothing runs, like Forge).
- Tiles of `tile` pixels (default 512; 1-127 become 128; 0 = one piece) with a
  32-pixel linear-ramp overlap — the same positions and weights as ComfyUI's
  `tiled_scale` and the extension's port of it. Each tile is reflect-padded to the
  model's multiple (16) and cropped back. Out of memory halves the tile down to
  128 (tile 0 starts at half the long side); the report gives the tile used.
- `forge_quantize` (default on) floors the incoming IMAGE to 8-bit levels like
  Forge's `uint8(255·x)` before the model and rounds the result to 8-bit levels
  like the extension's saved PNG, so ComfyUI's savers write the same bytes as
  Forge (checked bit for bit on CPU with the v1.1 model).
- Residual guard: an output that follows the input like an image (a denoise or
  deblur NAFNet) is refused with `not a DeGrid residual model: …`; a residual
  whose mean exceeds 100/255 is skipped with `output blew up: …`; a missing file
  gives `model not found: <name>`. Skips and any other error keep that image as
  it was and are reported — never fatal. A user cancel still cancels.
- Models: NAFNet files (all spandrel NAFNet detection keys; safetensors headers
  and zip `data.pkl` key names are read without `torch.load`; legacy pickles are
  skipped) directly in `upscale_models` (Forge `models/ESRGAN`) and `degrid`.
  `auto` = the highest `modelspec.version`, then folder, then name. A file name in
  both folders gets its folder as prefix. Reports name models like Forge's
  infotext (stem; `ESRGAN/<stem>` or `DeGrid/<stem>` on a clash).
- `device` auto/cpu, `precision` fp32/fp16 (fp16 = autocast on CUDA with fp32
  weights; a tile that overflows is redone in fp32; CPU always fp32),
  `keep_loaded` (default off: unloaded after the batch). The app sends the
  extension defaults auto/fp32/off; the Forge settings `sam3_degrid_device`,
  `sam3_degrid_gpu_precision` and `sam3_degrid_keep_loaded` do not apply to
  ComfyUI. On a GPU the model goes through ComfyUI's `load_models_gpu`; ComfyUI's
  unload-all (`/free`) also drops the cached model.
- Outputs: the IMAGE and `report_json`; the same per-image reports are in the
  node's UI output `ai_studio_degrid`: `status` (ok/skipped/off), `model`,
  `mode` (label), `strength`, `tile` (used), `precision` (`fp32`,
  `fp16-autocast`, `-` when nothing ran), `error`.

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
