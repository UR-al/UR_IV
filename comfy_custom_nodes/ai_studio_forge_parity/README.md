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
- sam-extra features (pack 1.7.0+; each only when its feature is on, in this MODEL
  order): `ForgeNeoExtraSchedulers` (right after the flow shift, only when the
  request carries the Extra Schedulers values and a pass uses `laplace`, `custom`,
  `flow_cosmos_rho7` or `flow_cosmos_dynamic`; without the node those four use the
  defaults), `ForgeNeoExtraSamplerSettings` (only when an ER SDE
  value differs from the defaults, on the MODEL every pass uses, before the
  guidance stack), then per main pass after Detail Daemon `ForgeNeoAnimaSpeed`
  and `ForgeNeoColorcraft` (last). Detailer passes get neither of the last two,
  as in Forge
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

## 1.8.0 changes

SAM3 "search inside faces only" (face crop) of the sam-extra Forge extension (v0.34.0, with the crop growth of
the v0.34.1 fix) in `ForgeNeoSAM3Mask`. The code is sam-extra's own: sam-extra (GPL-3.0-only) parts written by the
same author (UR-al), licensed to this pack under the MIT License by that author (`LICENSES/sam-extra-MIT.txt`,
`THIRD_PARTY_NOTICES.md` "SAM3 face crop (pack 1.8.0)"). The pack never imports the extension; the installed
extension is the test oracle on CPU (`tests/test_comfy_sam3_face_crop.py`).

- New optional inputs `face_crop` (default off), `face_crop_model` (default `face_yolov8n.pt`) and
  `face_crop_confidence` (default 0.3, 0–1) — the extension's three SAM3 Mask settings. Off (or left out) is the
  node's run without them, unchanged.
- On: an ultralytics YOLO box model finds the faces (`sam3_face_detect.py`: the extension's call
  `YOLO(file)(image, conf=…, device=…)`, `"cpu"` when SAM3 runs on the CPU, otherwise ultralytics' current CUDA
  device; nothing is downloaded). Each face box is widened (20 % sides and bottom, 40 % top), made square and pushed
  inside the image, and SAM3 runs only in those crops, every Detect token and then the Exclude tokens in the crop
  widened by Outline + Dilation. Pieces are pasted back at image coordinates and each gets an owner face (pieces cut
  at a crop edge are grouped with their continuation in a neighbouring crop). A token whose own pieces reach an
  inner crop edge (hair) runs again on a crop grown toward that side, at most two more times; a grown crop that
  finds nothing keeps the earlier result. Individual gives one mask per group and face (left to right), Combined
  their union; Convex Hull, Outline, Dilation, the hand-drawn mask and Exclude follow in the extension's order
  (`sam3_face_crop.py`, `sam3_face_crop_run.py`). Without a face in any image SAM3 is not loaded and the mask is
  empty, like a run that found nothing. SAM3's object masks come from Easy SAM3's `obj_masks` output, each cut to
  its box like the extension's `_clean_split_masks`.
- The model is an absolute path (the app sends its Forge `models/adetailer` file, like the SAM3 checkpoint) or a
  file name found in ComfyUI's `ultralytics_bbox`/`ultralytics` folders (Impact Subpack), `models/ultralytics/bbox`,
  `models/ultralytics` or `models/adetailer`.
- ultralytics is not a ComfyUI dependency. The informational `face_crop_detector` COMBO publishes whether this
  ComfyUI can import it (`ultralytics` or `(ultralytics not installed)`; nothing sets it). The app reads it from
  `/object_info`: its SAM3 card refuses to switch face crop on without it, and the compiler refuses a face-crop graph
  for a pack before 1.8.0 or a ComfyUI without ultralytics, with the reason. The app sends the three inputs only
  for a request with face crop on, so other graphs are byte-identical to 1.7.0's.
- The report JSON gains `face_crop` (the extension's artifact meta: model, confidence, faces with box, score,
  crop and the grown crops) and `face_crop_status` (its console line), a list per image for a batch; the overlay
  draws the face crops (orange, 2 px with the face score) and grown crops (1 px) before the SAM3 boxes. The
  console shows the extension's `[-] SAM3: face crop …` lines.
- An IMAGE batch runs per image with that image's faces; Individual pass k holds every image's k-th mask (empty
  where an image has fewer).

## 1.7.0 changes

Four features of the sam-extra Forge extension on ComfyUI: Extra Schedulers, Extra Samplers and
Anima SPEED (sam-extra v0.33.0), Colorcraft (sam-extra v0.32.1). Much of the code is sam-extra's
own: sam-extra (GPL-3.0-only) parts written by the same author (UR-al), licensed to this pack
under the MIT License by that author (`LICENSES/sam-extra-MIT.txt`). The rest is the pack's own
or comes from permissive upstreams; `THIRD_PARTY_NOTICES.md` names the reused functions and the
upstreams. The pack never imports the extension. The installed extension is the test oracle the
pack is compared with, bit for bit on CPU.
The app installs the pack before its next ComfyUI generation and restarts a ComfyUI it started
itself; restart a ComfyUI you started yourself.

- **Extra Schedulers** (sam-extra v0.33.0 "Extra Schedulers (sam-extra)"): the nine schedulers sam-extra adds
  to Forge's "Schedule type" list are registered in ComfyUI's KSampler scheduler table under their Forge
  names — `cosine`, `cosine_exponential`, `phi`, `laplace`, `karras_dynamic`, `custom`,
  `react_cosinusoidal_dynsf`, `flow_cosmos_rho7`, `flow_cosmos_dynamic` — so the app's base,
  Hires, ADetailer (Impact FaceDetailer), SAM3 and custom-workflow samplers all build the schedule Forge
  builds for the same steps and sigma range (`extra_schedulers.py`, pure part `extra_schedulers_math.py`).
  A name another node pack (or a future ComfyUI) already provides is never replaced: it is skipped, logged
  once and left out of the pack's `owned_names`, and the app refuses a graph that needs it.
- New `ForgeNeoExtraSchedulers` (MODEL -> MODEL; inputs `custom_mode`, `custom_expression`,
  `custom_sigmas`, `laplace_mu`, `laplace_beta`, `flow_cosmos_rho`, `flow_cosmos_sigma_max`,
  `flow_cosmos_sigma_min` = the extension's block of eight values (v0.33.0 appended the last three; an
  older five-value block gives them their defaults), plus the informational `owned_names` COMBO): the
  values `laplace`, `custom`, `flow_cosmos_rho7` and `flow_cosmos_dynamic` use (the last two share the
  three Flow Cosmos values), carried on a private copy of the MODEL's
  `model_sampling` (a scheduler handler receives nothing else). Every later sampler of that MODEL — the
  pack's nodes, stock `KSampler`/`KSamplerAdvanced`, `BasicScheduler`, Impact's module-level
  `calculate_sigmas` — sees them; the pack's flow shift keeps them (every Hires pass with Shift > 0) and a
  Hires checkpoint/UNET override gets them too. A ModelSampling* node placed *after* it drops them: the
  app inserts the node after its own flow shift.
- Bit-for-bit with the installed extension on CPU (`tests/test_comfy_extra_schedulers_parity.py`: settings
  coercion, a 219-text expression corpus with the same messages, sigma lists with Forge's interpolation,
  every float32 schedule over n x sigma ranges x mu/beta x rho grids, Flow Cosmos rho7 and Flow Cosmos
  Dynamic on flow and eps/v models over a ρ x σ̃ grid incl. values past the node's ranges (same errors and
  texts); real-ComfyUI tier
  `tests/test_comfy_extra_schedulers_runtime.py`). A schedule that cannot be used stops the run with the
  extension's English message (ComfyUI shows it as the sampler node's error), like Forge.
- **Extra Samplers** (sam-extra v0.33.0 "Extra Samplers"): twelve of the 18 samplers sam-extra adds to Forge's
  sampler list are registered in ComfyUI's KSampler sampler list under sam-extra's own names (its first aliases) —
  `er_sde_reverse_time`, `er_sde_ode`, `er_sde_tunable`, `dpmpp_4m_sde`, `euler_dy_cfg_pp`, `euler_smea_dy_cfg_pp`,
  `euler_dy`, `euler_smea_dy`, `dpmpp_2m_flow_ode`, `dpmpp_2m_heun_flow_ode`, `dpmpp_3m_flow_ode`, `restart_flow`
  — and a thirteenth, `ipndm_v_fixed`, is sam-extra 0.33.1's IPNDM_V (next item), so the app's base, Hires,
  ADetailer (Impact FaceDetailer), SAM3 and custom-workflow samplers find them by name
  (`extra_samplers_register.py`). The other five are ComfyUI's own (`dpmpp_2m_sde_heun`, `uni_pc_bh2`,
  `cfgpp_ud10_ab` from ComfyUI v0.35.0, `ipndm`, `deis`): the app only maps the Forge label. A name another
  node pack already provides is never replaced (e.g. Koishi-Star's own node registers `euler_dy`/`euler_smea_dy` with
  its always-churn rule): it is skipped, logged once and left out of the pack's `owned_names`, and the app refuses a
  graph that needs it. `dpmpp_4m_sde` and `dpmpp_3m_flow_ode` join `KSampler.DISCARD_PENULTIMATE_SIGMA_SAMPLERS`
  (Forge's option of those entries). Impact's FaceDetailer (ADetailer, the SAM3 restore-face pass) builds its sigmas
  with its own discard list, so for those two the pack hands it `scheduler_func_opt`
  (`extra_samplers_register.impact_scheduler_func`): the same sigmas as KSampler's (one step more, the penultimate
  dropped), as Forge discards on the ADetailer pass too.
- **IPNDM_V fix** (`ipndm_v_fixed`, `extra_samplers_ipndm.py`): ComfyUI's `ipndm_v` and sam-extra's IPNDM_V up to
  0.33.0 carry an upstream (zju-pi) typo in the fourth-order weight — `h_{n-1}/h_{n-2}` where the third divided
  difference needs `h_{n-2}/h_{n-3}` — so the weights do not sum to 1 where the step ratio changes: about -174 at
  step 15 (0-based) of Anima's Linear Quadratic 28-step list, and the image turns to green noise. sam-extra 0.33.1
  fixes that one factor — a consistency fix, not a cure for Linear Quadratic: with it the image is no longer green
  noise but still burnt and smeared on Forge and on ComfyUI (2026-10-05 GPU check). That list's step size jumps
  ×3.71, ×2.46, ×1.59 after 14 tiny steps, and any step-size-aware fourth-order extrapolation weights the old
  derivatives up to about 155 there (the exact variable-step Adams–Bashforth weights reach 117) — a sampler ×
  scheduler combination. Use IPNDM_V with Simple (Normal is coherent but grainy), or IPNDM, which is fine on Linear
  Quadratic. The pack never replaces ComfyUI's core `ipndm_v`, so it registers the fixed sampler as
  `ipndm_v_fixed` and the app sends Forge's "IPNDM_V" there (the label and the core name fold to the same key, so
  the app's passes never run the core one). Two other upstream inaccuracies are kept on purpose (`temp1`'s
  parenthesisation, the sign of `temp2`'s second term — the weights still sum to 1). The result equals sam-extra
  0.33.1's IPNDM_V bit for bit on every list. Where the step ratio is constant (equal steps, geometric lists) its
  weights are ComfyUI's `ipndm_v` ones in exact arithmetic, and the latents ComfyUI's bit for bit only when the step
  sizes are exact in binary (power-of-two equal steps, halving or doubling lists); float32/float64 linspace or
  geometric schedules differ at rounding level.
- New `ForgeNeoExtraSamplerSettings` (MODEL -> MODEL; inputs `max_stage`, `eta`, `noise_window`, `noise_start`,
  `noise_end` = the extension's block, plus the informational `owned_names` COMBO): the ER SDE values, carried in
  the MODEL's `model_options`, so every later sampler of that MODEL uses them; a Hires checkpoint/UNET override keeps
  them. Without the node the samplers run the extension's defaults (3, 1.0, off, 0.2, 0.8).
- Same latents as the installed extension on CPU for the same inputs (`tests/test_comfy_extra_samplers_parity.py`:
  ER SDE trio incl. the noise window, DPM++ 4M SDE, Euler (SMEA) Dy (CFG++), the DPM++ flow ODE trio, Restart (flow)
  — bit for bit on Forge's k-diffusion module; registration and node: `tests/test_comfy_extra_samplers_pack.py`;
  on ComfyUI's real k-diffusion and sampler tables on CPU, incl. `er_sde_reverse_time`/`er_sde_tunable` == ComfyUI's
  own `SamplerER_SDE` sampler and ComfyUI's `cfgpp_ud10_ab`, `ipndm`, `deis`, `dpmpp_2m_sde_heun` ==
  sam-extra's entries (`uni_pc_bh2`: the same algorithm, not claimed bit-equal), `ipndm_v_fixed` == sam-extra
  0.33.1's IPNDM_V (Linear Quadratic and random lists) and == ComfyUI's `ipndm_v` on constant-ratio lists whose
  steps are exact in binary (within rounding on ComfyUI's float `exponential` list):
  `tests/test_comfy_extra_samplers_runtime.py`). Images still differ from Forge's for the same seed: the two hosts
  draw their noise differently (ComfyUI's seeded generator / Brownian tree vs Forge's per-image streams).
- **Anima SPEED** (sam-extra v0.33.0 "Anima SPEED", Spectral Progressive Diffusion, arXiv 2605.18736):
  the early, noise-dominated steps of a pass run on a DCT-truncated coarse latent grid; at a
  transition sigma the coarse spectrum is embedded in the full grid, the new frequencies are filled
  with sigma-scaled noise and the state is rescaled by kappa. Computation modules (no torch at
  import; the MIT upstreams howardhx/speed, aoleg/ComfyUI-SPEED and
  sorryhyun/ComfyUI-Spectrum-KSampler with sam-extra's code on them, under the author's MIT
  grant — see THIRD_PARTY_NOTICES.md):
  - `speed_schedule.py` — presets (anima, flux, flux2, krea-2, krea-2-raw, z-image, wan21, custom),
    transition sigmas (delta tolerance, adaptive delta, neo_shift divisor, manual), the two plans
    (`transition`: official/aoleg, only the hand-off sigma is patched; `respace`: sorryhyun, the rest
    of the schedule is re-spaced) and the refusals (`PlanError` skip/invalid, incl. custom A <= 0 and
    non-finite A/beta as settings errors, as in sam-extra 0.32.1; a zero denominator such as `1/0` in
    the scales / manual sigmas text is a settings error too, as in sam-extra 0.33.0).
  - `speed_spectral.py` — torch-only spectral maths in float64 on the latent's device (float32 on
    MPS): cached orthonormal DCT-II basis, DCT downscale (T_Phi), amplitude-keeping resize, the DCT /
    FFT (official ca7801c9 normalization, no `/sqrt(2)`) / one-level Haar noise expansions,
    kappa / aligned time (Eq. 5/6), sorryhyun's respace arithmetic and even snap.
  - `speed_runner.py` — the segmented run around any k-diffusion `sample_*`: DCT-truncated start
    (img2img/hires starts flow-rescaled with per-seed top-up noise, sam-extra's default), coarse
    segments, expansion per image seed (numpy `default_rng(seed + (k + 1) * 10000)` for
    `transition`, `torch.Generator(seed + 10000)` for `respace` — a batch image equals the same seed
    alone), the patched schedule published after each hand-off and restored afterwards, coarse noise
    samplers from the host, the step offset for samplers that name `sam_extra_step_offset` (read
    through CNS's `_forge_neo_base`), and the status summary.
  Every plan, refusal message, transform, run output, published schedule and status equals the
  installed extension's (`tests/test_comfy_speed_parity.py`, CPU, exact).
  - `speed_nodes.py` — node **`ForgeNeoAnimaSpeed`** (MODEL -> MODEL; explicit inputs `enabled`, `mode`,
    `threshold`, `preset`, `scales`, `delta`, `sigma_divisor`, `manual_sigmas`, `adaptive`, `transform`,
    `spectrum_A`, `spectrum_beta`, `seed` (-1 = image seeds), `pass_label` (base / img2img / hires),
    `image_seed` (the generation seed), `img2img_rescale`). Settings are read with sam-extra's rules and
    messages; invalid settings leave the MODEL unchanged with an `[AnimaSPEED] … invalid settings` console
    line. The node registers a SAMPLER_SAMPLE wrapper placed ahead of Detail Daemon and behind a CNS wrapper
    already on the MODEL (the guidance suite's, which a stock KSampler/KSamplerAdvanced never re-asserts; the
    pack's sampler nodes re-assert their own): runtime order CNS -> SPEED -> Detail Daemon -> sampler on every
    sampler node.
  - Per sampling run (KSAMPLER only; UniPC / DPM fast / adaptive are skipped) the wrapper checks, like
    sam-extra: a 1-D sigma list, a 4-D/5-D latent, a flow-matching (`CONST`) model, sigma_max <= 1, no
    inpainting mask, no reference latents / `c_concat` / ControlNet / Anima ControlNet-LLLite, no ComfyUI
    Spectrum, and (ComfyUI only) no area/mask conds or several packed latents. Otherwise it plans and runs
    SPEED; a SPEED failure samples the pass again without it; an interrupt goes through. Status lines
    (`[AnimaSPEED] base: …`) use sam-extra's wording; nothing is written into the image.
  - Sampler noise stays one stream across the size changes: a sampler that takes a `noise_sampler` gets
    one generator for the run, made like ComfyUI's default one (`extra_args` seed, CPU `+1`); Brownian
    SDE samplers get a per-image tree per coarse stage (`seed + (stage + 1) * 1_000_003`) and a plain-run
    tree for the full-size tail (its range raised to an aligned hand-off sigma above the pass's range).
  - The patched schedule is published as `transformer_options["sample_sigmas"]` (tagged as re-published)
    while the run lasts. Detail Daemon follows it: inside SPEED it looks each sigma up in the published
    copy (sam-extra's Detail Daemon reads Forge's current `sampling_sigmas`), so the aligned hand-off sigma
    gets its own step's schedule value; the copied Detail Daemon node code is unchanged.
  - A Hires checkpoint/UNET override keeps the pass's SPEED patch (`hires_carry`).
  - `anima_lllite.apply_lllite` marks its MODEL (`model_options["forge_neo_lllite"]`) so SPEED steps aside.
  - The legacy `speed_*` inputs of `ForgeNeoKSamplerCNS` (the external sorryhyun SpectrumSPDKSampler
    provider) are unrelated and unchanged.
- **Colorcraft** (sam-extra v0.32.1 "Colorcraft", muerrilla/ComfyUI-Colorcraft's sampling-time colour grading):
  node **`ForgeNeoColorcraft`** (MODEL -> MODEL; inputs `model`, `vae`, `settings_json`, `contract` [v1],
  `pre_dd_sigma`, default on). Exposure, tone compression, contrast, clarity/sharpness, temperature/tint,
  vibrance/saturation, chroma contrast, Chroma Plus, colour shift — as up to ten scheduled modifiers, optionally
  masked by latent-space masks and their combinations — applied to the CFG-combined prediction at every step.
  - `vendor/comfyui_colorcraft/` — the MIT upstream's `lib_colorcraft/` vector, mask, colour and schedule
    functions and the krea2/zimage basis vectors, unchanged (`UPSTREAM.md`); `vendor/comfyui_colorcraft_fork/` —
    the Flux2 basis vectors of the aoleg fork. The vector files are `.safetensors`, now part of the pack
    fingerprint: a vectors-only change also asks for a ComfyUI restart.
  - `colorcraft_basis.py` (families: `Wan21` -> krea2 — Anima, Krea 2, Qwen-Image, Wan 2.1; `Flux` -> zimage;
    `Flux2` -> flux2, 128 channels, 16x VAE; anything else — SD 1.5/SDXL — has no basis: only Contrast and
    Color Shift apply) and `colorcraft_engine.py` (upstream's modifier loop, statement for statement, with the
    sam-extra hook's three substitutions: anchors encoded before sampling, the family's VAE factor, schedule
    values worked out first). Every result equals the installed extension's engine bit for bit on CPU
    (`tests/test_comfy_colorcraft_parity.py`).
  - `colorcraft_host.py` — sam-extra's host rules in ComfyUI terms: the colour anchors are VAE-encoded with the
    pass's VAE when the node runs (flat 512x512 image, model-space lift with the encode's rank, float32; cached
    across prompts per VAE and weight patches, 32 entries); the schedules are built on each sampling run's own
    sigma list (the denoise tail on img2img/Hires) and looked up by the evaluation's sigma — with Detail Daemon,
    by the sampler's own sigma it noted (`pre_dd_sigma`); Anima SPEED's re-published list is followed; Euler
    (SMEA) Dy CFG++ sub-steps pass through; fp16/bf16 graded in fp32; a non-finite result or an error passes
    the prediction through (one console line). The post-CFG function goes last, after Optimal Scale, the
    guidance suite, Skimmed CFG and Detail Daemon (Forge's order). One console line per run, the extension's
    status line character for character: `[AI Studio Colorcraft] base: krea2 (Wan21); mods I+II; 30 evals, 28 edited`.
    When the lookup used Detail Daemon's noted sigma, a second line carries what Forge records as its own
    infotext key: `[AI Studio Colorcraft] base pass: SAM Extra Colorcraft pre-DD sigma: True (x16)`.
  - A Hires checkpoint/UNET override keeps the hires pass's Colorcraft (`hires_carry`), prepared again for the
    override model's latent format and with its VAE. A Hires VAE-only override prepares it again with that VAE
    on the pass's own model (`hires_carry.apply_vae_change`), as Forge, which reloads the model with the new VAE
    for the Hires pass, encodes the anchors with it. Detailer passes (ADetailer/SAM3) are never graded.
- Shared ground for the four sam-extra ports (`guidance_common.py`, `compat.py`, `hires_carry.py`):
  - sam-extra's coordination names, one definition each: `SUBSTEP_MARKER` (`sam_extra_substep`, Euler
    (SMEA) Dy CFG++'s extra evaluations, read by `is_extra_sampler_substep`), `STEP_OFFSET_KWARG`
    (`sam_extra_step_offset`, passed only to a sampler that names it — `declares_keyword`),
    `REPUBLISHED_FROM` (`mark_republished`/`republished_from`, SPEED's re-published sigma list). Values
    pinned to the installed extension (`tests/test_comfy_pack_foundation.py`); the code is the pack's own.
  - `compat.register_names` registers sampler/scheduler names in ComfyUI's tables without ever replacing
    another provider's entry (logged once, not published); nodes report what the pack owns through the
    optional `owned_names` COMBO (`(none)` when nothing), which the app reads from `/object_info`.
  - CNS's sampler function carries `_forge_neo_base` (the function it wraps), so SPEED can see whether
    the real `sample_*` declares the step-offset keyword. `compat.make_first` orders SAMPLER_SAMPLE
    wrappers (CNS stays outermost: it re-asserts itself at the pack's sampler nodes, and the SPEED node keeps a
    CNS already on the MODEL ahead of its own wrapper).
- `ForgeNeoHiresFix`: after a checkpoint/UNET override (and its flow shift) the pack features re-apply
  their MODEL patches to the override model (`hires_carry.apply`, in registration order; a failing
  carry fails the pass). Detail Daemon and the guidance suite still do not reach an override model.
  With only the VAE overridden, the features registered with `uses_vae` (Colorcraft) prepare their patch
  again with that VAE (`hires_carry.apply_vae_change`, Shift 0 or not).
- The pack's flow shift (`ForgeNeoModelSamplingShift`, `ForgeNeoHiresFix` with Shift > 0, Anima Tile &
  Repair) keeps the values features carry on `model_sampling` (registered attribute names) — before
  1.7.0 nothing carried any; with Extra Schedulers it keeps μ/β/custom and the Flow Cosmos values on every
  Hires pass.
- `ForgeNeoKSamplerCNS` marks its clone `transformer_options["ai_studio_dy_substeps_off"]` when ComfyUI
  Spectrum or the legacy SPEED provider (`speed_enabled`) samples: Euler (SMEA) Dy CFG++ then runs
  without sub-steps (sam-extra's Spectrum rule). The pack's own SPEED node never sets it.

## 1.6.0 changes

The detail guidance of the sam-extra Forge extension v0.30.0, with its equations, defaults, clamps
and order (`tests/test_comfy_detail_parity.py` runs the pack and the installed extension on the same
tensors on CPU). Much of the code is sam-extra's, licensed to this pack under MIT by the author
(`THIRD_PARTY_NOTICES.md`):

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
  (`tests/test_comfy_degrid_origin.py`). Some of its functions are sam-extra's code, licensed to
  this pack under MIT by the author (see `THIRD_PARTY_NOTICES.md`).
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
  face restore, overlays, artifacts, and search inside faces only (face crop,
  pack 1.8.0+; needs ultralytics in ComfyUI's Python).
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

## Extra Schedulers (pack 1.7.0+)

| Scheduler | Schedule (p = i/(n-1); n sigmas + final 0, float32) |
|---|---|
| `cosine` | σmin + ½(σmax − σmin)(1 − cos(π(1 − √p))) |
| `cosine_exponential` | (1 − p)·Cosine + p·Exponential (k-diffusion `get_sigmas_exponential`) |
| `phi` | σmin + (σmax − σmin)(1 − p)^(φ²) — the idea of Extraltodeus's Golden Scheduler |
| `laplace` | ComfyUI's `get_sigmas_laplace` with `laplace_mu`/`laplace_beta` (Hang et al., arXiv:2407.03297); where its clamp repeats a sigma (a flow model's σmax 1 with mu 0), the same curve over the part inside [σmin, σmax] |
| `karras_dynamic` | the Karras/EDM ramp (arXiv:2206.00364) with a per-step ρᵢ = 7 + 2cos(2πi/n) |
| `react_cosinusoidal_dynsf` | σmax·((σmin + (σmax − σmin)·cos(πp/2)) / σmax)^(2.15·p) — React Cosinusoidal DynSF, yoinked-h's published formula (reForge's label); a σ-range formula that keeps many steps at high σ (Anima shift 3, 28 steps: 17 above σ 0.5); no values |
| `flow_cosmos_rho7` | Flow Cosmos rho7 (sam-extra's label; the name stays whatever ρ is). **Flow model** (ComfyUI `CONST`, e.g. Anima): the Karras ρ ramp over a VE-equivalent noise range σ̃, s = (σ̃max^(1/ρ) + p(σ̃min^(1/ρ) − σ̃max^(1/ρ)))^ρ, mapped to flow time t = s/(1 + s) — from σ̃max/(1 + σ̃max) to σ̃min/(1 + σ̃min), whatever the MODEL's sigma range. The defaults `flow_cosmos_rho` 7, `flow_cosmos_sigma_max` 80, `flow_cosmos_sigma_min` 0.002 are Cosmos-Predict2's schedule, a list from 80/81 ≈ 0.988 to ≈ 0.002 — the same numbers as the custom expression `(80**(1/7)+x*(0.002**(1/7)-80**(1/7)))**7/(1+(80**(1/7)+x*(0.002**(1/7)-80**(1/7)))**7)` (with other values: that expression with them in place of 7, 80 and 0.002). **Any other model** (SD/SDXL eps/v): ComfyUI's own `get_sigmas_karras` with `flow_cosmos_rho` on the MODEL's sigma range (with ρ 7 its `karras` schedule); σ̃ is not used there; one console line per process. Name and idea: KeithZ117/Comfyui-anima-sampler (credit only) |
| `flow_cosmos_dynamic` | Flow Cosmos Dynamic (sam-extra's label; sam-extra's own composition of Karras Dynamic and Flow Cosmos rho7, not from another project). **Flow model**: Karras Dynamic's ramp with its per-step exponent ρᵢ = ρ + 2cos(2πi/n), written σ̃max((1 − p) + p(σ̃min/σ̃max)^(1/ρᵢ))^ρᵢ (exactly σ̃min at p = 1), over Flow Cosmos rho7's σ̃ range and mapped to flow time t = s/(1 + s); the same three values (`flow_cosmos_rho`, `flow_cosmos_sigma_max`, `flow_cosmos_sigma_min`; defaults ρ 7 on σ̃ 0.002–80), no values of its own; the MODEL's sigma range is not used. It meets Flow Cosmos rho7's list (same values) where cos(2πi/n) = 0 — the first and last step and, when 4 divides n, n/4 and 3n/4 — and lies below it in the first and last quarter, above it in the middle half. **Any other model**: Karras Dynamic itself (bit for bit `karras_dynamic`) with `flow_cosmos_rho` on the MODEL's sigma range; σ̃ is not used there; one console line per process |
| `custom` | `custom_expression` (variables m, M, n, s, x; constants phi, pi, e; + − * / ** and 20 math functions; checked against a whitelist, never run as Python) or `custom_sigmas` (a list; 1.0 … 0.0 is scaled between σmax and σmin, log-linear interpolation = ComfyUI's `loglinear_interp`) |

- A schedule with the same sigma twice before the final 0 is refused (a zero-length step makes multistep
  samplers divide by zero); Karras Dynamic must fall at every step. React Cosinusoidal DynSF reaches that
  limit near σmax from about 448 steps on an Anima range (434 on SDXL's); Flow Cosmos rho7 on a flow model
  only from 1,119,764 steps with the default values, Flow Cosmos Dynamic from 1,277,194. Flow Cosmos rho7 also
  stops on a sigma before the final 0 that is not a usable float32 (0, a subnormal, NaN — only values past the
  node's ranges, or an eps/v MODEL with a tiny sigma_min, get there) and on an overflow of its maths, with the
  extension's texts; Flow Cosmos Dynamic does so on a flow model only (on an eps/v MODEL it is Karras Dynamic,
  which has no such rule).
- ComfyUI has no Forge "rho"/"sigma min/max" settings: Karras Dynamic uses ρ 7 (the two Flow Cosmos schedulers
  never read Forge's rho: they have their own), and every scheduler but the Flow Cosmos ones on a flow model the
  MODEL's own sigma range.
- Flow Cosmos rho7's values (sam-extra v0.33.0): `flow_cosmos_rho` 1–15 (step 0.1, default 7),
  `flow_cosmos_sigma_max` 1–1000 (step 0.1, default 80), `flow_cosmos_sigma_min` 0.0001–1 (step 0.0001,
  default 0.002) — sam-extra's slider ranges; σ̃ is the VE-equivalent noise level. Without
  `ForgeNeoExtraSchedulers` (or with an older five-value block) the defaults apply, which give exactly the
  list of before. Like Laplace's μ/β, a value that is not a finite number becomes the default and one outside
  its range is clamped (the node coerces; ComfyUI itself refuses a widget value outside min/max). ρ must be
  finite and above 0, and on a flow model 0 < σ̃ min < σ̃ max, both finite — only σ̃ min = σ̃ max = 1 passes the
  ranges and still stops the run with the extension's message; an eps/v model does not use σ̃ and runs.
  Scaling both σ̃ ends by k is the flow time shift k (t' = k·t/(1 + (k − 1)·t) ⇔ σ̃' = k·σ̃), so Anima's shift 3
  on the default list corresponds to σ̃ 0.006–240 — a fact of the maths, not a quality claim. A narrow σ̃ range
  reaches the flat-step limit much sooner (σ̃ 0.9999–1: from 841 steps).
- Flow Cosmos Dynamic reads the same three values. Like Karras Dynamic it needs ρ above 2 (the node allows
  1–15: ρ ≤ 2 stops the run with the extension's text, on every model) and a list that falls at every step:
  below ρ ≈ 4 it rises mid-run for some step counts (σ̃ 0.002–80, also ×k: every ρ up to 3.8 for some n ≤ 300,
  from 3.9 none), which stops the run like Karras Dynamic; on a flow model a flat step is refused first (its
  text names ρ and σ̃). With the defaults Anima's 28 steps put 18 above 0.5 and 5 below 0.05 (Flow Cosmos
  rho7: 17 and 5). Its quality on Anima is unverified (no A/B). The console line on a model that is not a flow
  model is sam-extra's, separate from Flow Cosmos rho7's and with the ρ of that first run:
  `[Extra Schedulers] Flow Cosmos Dynamic: not a flow model - using Karras Dynamic rho 7 on the model's own sigma range`.
- Flow Cosmos rho7 is a Karras ramp seen in flow time, so many steps stay near t = 1 (Anima, 28 steps: 17
  above 0.5, 5 below 0.05). Its quality on Anima is unverified beyond one 2-seed A/B on Forge with the
  defaults; that A/B is an observation, not a claim, and other values are unverified. The console line on a
  model that is not a flow model is sam-extra's, with the ρ of that first run:
  `[Extra Schedulers] Flow Cosmos rho7: not a flow model - using Karras rho 7 on the model's own sigma range`.
  ComfyUI's Cosmos-Predict2 sampling (`COSMOS_RFLOW`, EDM σ 0.002–80) is not `CONST` and takes that branch:
  the same EDM ramp in that model's own σ units, which it maps to t = σ/(σ + 1) itself (the flow list up to
  float32 rounding).
- `AISTUDIO_XSCHED_DEBUG=1` in ComfyUI's environment logs every schedule a handler builds.

## Extra Samplers (pack 1.7.0+)

| ComfyUI name | Forge label (sam-extra) | What runs on ComfyUI |
|---|---|---|
| `er_sde_reverse_time` | ER SDE (Reverse-time) | ComfyUI's `sample_er_sde` with h(λ) = λ^(η+1) (η = 0 → the ODE); optional noise window |
| `er_sde_ode` | ER SDE (ODE) | ComfyUI's `sample_er_sde` with h(λ) = λ, no noise |
| `er_sde_tunable` | ER SDE (Tunable) | ComfyUI's `sample_er_sde` with h(λ) = λ(e^(λ^0.3)+10)^η — ComfyUI's `er_sde` at η 1 without a window; optional noise window |
| `dpmpp_4m_sde` | DPM++ 4M SDE | Clybius' 4M SDE (BSD-3-Clause), momentum 0, half-log-SNR flow handling; Brownian-tree noise; penultimate sigma discarded |
| `euler_dy_cfg_pp` / `euler_smea_dy_cfg_pp` | Euler Dy CFG++ / Euler SMEA Dy CFG++ | Koishi-Star's Dy/SMEA sub-steps (Apache-2.0) with the CFG++ update; CFG 1–2 recommended |
| `euler_dy` / `euler_smea_dy` | Euler Dy / Euler SMEA Dy | the same sub-steps with plain Euler steps; churn only with `s_churn` > 0 (`min` rule) |
| `dpmpp_2m_flow_ode` / `dpmpp_2m_heun_flow_ode` / `dpmpp_3m_flow_ode` | DPM++ 2M (flow ODE) / DPM++ 2M Heun (flow ODE) / DPM++ 3M (flow ODE) | ComfyUI's own `dpmpp_2m_sde` (midpoint / Heun correction) and `dpmpp_3m_sde` at η 0: DPM-Solver++ for the model's probability-flow ODE in its half-log-SNR λ; no noise is ever drawn (no `eta`, no noise sampler); `dpmpp_3m_flow_ode` discards the penultimate sigma like Forge's "DPM++ 3M SDE" (ComfyUI's own `dpmpp_3m_sde` does not) |
| `ipndm_v_fixed` | IPNDM_V | zju-pi's variable-step iPNDM (Apache-2.0) with the fourth-order weight fixed as in sam-extra 0.33.1 (ComfyUI's core `ipndm_v` keeps the upstream typo); one model call per step, orders 1–4 (default 4). Not usable with Linear Quadratic (green noise before the fix, a burnt image after it) — use Simple, or `ipndm` |
| `restart_flow` | Restart (flow) | Restart sampling (Xu et al., arXiv:2306.14878) on the scheduler's own grid: Heun steps; from 20 steps, after the step that reaches s = σ/α ≈ 0.1 it jumps back to s ≈ 2 (σ ≈ 0.09 → 0.67 on a flow model) with the model's exact forward transition and re-walks those steps, once (twice from 36 steps) — about 1.4× Heun's model calls at 28 steps; fewer than 20 steps = plain Heun |

- ER SDE values (`ForgeNeoExtraSamplerSettings`): `max_stage` 1–3 (solver order, all three ER SDE samplers), `eta`
  0–10 (Reverse-time and Tunable), `noise_window` with `noise_start`/`noise_end` 0–1 (sampling progress, default
  0.2–0.8): only the steps whose target σ lies in `[percent_to_sigma(end), percent_to_sigma(start)]` add SDE noise,
  the others are ODE steps that draw nothing; `start ≥ end` = no noise at all. ComfyUI's own `er_sde` ignores them.
- Euler (SMEA) Dy: the Dy step evaluates the (odd row, odd column) pixels — a half-resolution latent — at steps 2
  and 3; SMEA adds a ×1.25 evaluation at step 0 (more VRAM) and the half-resolution one at step 1. The evaluations
  carry `transformer_options["sam_extra_substep"]` so the pack's guidance keeps its step state; an inpainting mask,
  latent and noise follow each evaluation's size. The sub-steps are skipped (plain (CFG++) Euler steps, logged once)
  under ComfyUI Spectrum / the legacy SPEED provider of `ForgeNeoKSamplerCNS`, a Spectrum model-function wrapper,
  concat conditioning (inpainting / fill models, Wan I2V) and area/mask conditioning.
- ComfyUI has no Forge "Sampler parameters": η (ancestral), sigma noise and churn run at Forge's defaults (1, 1, 0).

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
