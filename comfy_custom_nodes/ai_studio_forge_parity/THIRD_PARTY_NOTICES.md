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
code is copied. See
`vendor/comfyui_anima_lllite/UPSTREAM.md` for the SHA-256 values, omitted
upstream files, and the local integration boundary. No LLLite or other model
weight is included. This notice does not replace or modify the upstream
license.
