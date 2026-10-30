# -*- coding: utf-8 -*-
"""Writes the Anima studio example workflow (ComfyUI UI workflow JSON v0.4).

Output: ``comfy_custom_nodes/ai_studio_forge_parity/example_workflows/AI Studio Anima.json``.
``tests/test_studio_workflow_json.py`` checks that this script reproduces the saved file.

Layout: the three studio UI nodes on top (panel, result view, XYZ view); below them the engine room, one group per
row from top to bottom (① models, ①b 3.8B, ② LoRA, ③ prompts, ④ extensions, ⑤ latent/sampling, ⑥ Hires, ⑦ upscale,
⑧ SAM3, ⑨ save, ⑩ gallery post-processing), nodes left to right, every link left->right in a row or from an upper row
to a lower one (R6). Engine nodes are saved folded (the panel's engine-room button unfolds them); groups are sized for
the unfolded layout. Each engine node carries ``properties.aistudio_role``; the roles come from
``web/ai_studio/studio_roles.js`` (R20, single source).

The extension nodes sit where the app compiler puts them (``core/comfy_workflow_compiler.py``) and are saved bypassed:
a bypassed node passes its first input of the same type through, so ``ext_*`` off is the plain graph. Every pass has
its own MODEL: ER SDE -> Optimal Scale -> guidance suite -> Skim CFG is shared, then base = Detail Daemon -> SPEED,
Hires = its own pair, SAM3 = its own Detail Daemon (never SPEED). The post row (R19) takes the same pass models.

How the frontend (1.57.4) restores a node from this file, which the node entries must follow:
- ``inputs[i]`` overwrites the node's i-th input slot; the slots are the socket inputs, then one slot per widget
  (both in INPUT_TYPES order, required before optional; ``socketless`` widgets get no slot).
- ``widgets_values`` fill the widgets by position; an INT widget named ``seed``/``noise_seed`` is followed by its
  ``control_after_generate`` combo. Values are written only where every widget value is known without the user's
  ComfyUI (no install-dependent choice lists); other nodes keep their defaults. The panel writes the controlled
  inputs at queue time anyway.

Run: ``venv\\Scripts\\python.exe tools\\build_anima_studio_workflow.py [--out PATH]``
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from comfy_custom_nodes.ai_studio_forge_parity import NODE_CLASS_MAPPINGS  # noqa: E402
from comfy_custom_nodes.ai_studio_forge_parity.studio_state import effective_prompt, normalize_state  # noqa: E402

PACK = ROOT / "comfy_custom_nodes" / "ai_studio_forge_parity"
OUT = PACK / "example_workflows" / "AI Studio Anima.json"
ROLES_JS = PACK / "web" / "ai_studio" / "studio_roles.js"

ROLE_PROPERTY = "aistudio_role"
MODE_ALWAYS, MODE_BYPASS = 0, 4
WIDGET_TYPES = frozenset({"INT", "FLOAT", "STRING", "BOOLEAN"})
SEED_WIDGETS = frozenset({"seed", "noise_seed"})
SEED_CONTROL = "fixed"  # the panel owns the seed; the engine widget should not change by itself
DORA_FORGE_MODE = "LyCORIS fp32 (as trained)"  # "Forge처럼" DoRA, studio_logic.js

# ---------------------------------------------------------------------------------------------- canvas layout

UI_NODES = (  # (class, pos, size, saved widget value or None for the panel state)
    ("AIStudioPanel", (0, 0), (790, 960), None),
    ("AIStudioResultView", (806, 0), (680, 960), "{}"),
    ("AIStudioXYZView", (1502, 0), (400, 960), "{}"),
)
ENGINE_TOP = 1040
GROUP_HEADER = 80  # group title band + the node's own title bar above its pos
GROUP_PAD = 20
ROW_GAP = 40
NODE_GAP = 40
SLOT_HEIGHT = 20
WIDGET_HEIGHT = 24
MULTILINE_HEIGHT = 80
NODE_WIDTH = 340
NODE_WIDTHS = {
    "AIStudioSwitch": 220, "VAEDecode": 220, "VAEEncode": 220, "ForgeNeoModelSamplingShift": 260,
    "AIStudioPromptEncode": 400, "ForgeNeoAnima38V2Prompt": 400, "AIStudioLoraStack": 400,
    "ForgeNeoHiresFix": 400, "ForgeNeoSAM3Mask": 400, "ForgeNeoSAM3Detailer": 400, "AIStudioSaveForge": 400,
}
LOAD_IMAGE_HEIGHT = 340  # room for the preview

ROWS = (  # (group title, [(role, class, node title)])
    ("① 모델", [
        ("unet", "UNETLoader", "UNet (2.9B·base)"),
        ("te", "CLIPLoader", "TE"),
        ("vae", "VAELoader", "VAE"),
    ]),
    ("①b 3.8B", [
        ("unet38", "ForgeNeoAnima38V2Loader", "UNet 3.8B"),
        ("qwen35", "ForgeNeoAnimaQwen35Loader", "Qwen3.5 TE (3.8B)"),
        ("family_switch_model", "AIStudioSwitch", "계열: 모델"),
    ]),
    ("② 시프트·LoRA", [
        ("shift", "ForgeNeoModelSamplingShift", "Shift"),
        ("sched", "ForgeNeoExtraSchedulers", "스케줄러 값"),
        ("lora_stack", "AIStudioLoraStack", "LoRA"),
        ("ext_lora_block", "ForgeNeoLoraBlockWeight", "LoRA 블록"),
        ("negpip", "ForgeNeoNegPip", "NegPiP"),
        ("dora", "ForgeNeoDoraInferenceMode", "DoRA"),
        ("ext_er_sde", "ForgeNeoExtraSamplerSettings", "ER SDE"),
    ]),
    ("③ 프롬프트", [
        ("pos_encode", "AIStudioPromptEncode", "프롬프트"),
        ("neg_encode", "AIStudioPromptEncode", "네거티브"),
        ("pos_encode38", "ForgeNeoAnima38V2Prompt", "프롬프트 (3.8B)"),
        ("neg_encode38", "ForgeNeoAnima38V2Prompt", "네거티브 (3.8B)"),
        ("family_switch_pos", "AIStudioSwitch", "계열: 프롬프트"),
        ("family_switch_neg", "AIStudioSwitch", "계열: 네거티브"),
    ]),
    ("④ 확장", [
        ("ext_optimal_scale", "ForgeNeoAnimaOptimalScale", "Optimal Scale"),
        ("ext_guidance", "ForgeNeoAnimaGuidanceSuite", "가이던스"),
        ("ext_skim", "ForgeNeoSkimmedCFG", "Skim CFG"),
        ("ext_dd", "ForgeNeoAnimaDetailDaemon", "Detail Daemon"),
        ("ext_speed", "ForgeNeoAnimaSpeed", "SPEED"),
        ("ext_dd_hires", "ForgeNeoAnimaDetailDaemon", "Detail Daemon (Hires)"),
        ("ext_speed_hires", "ForgeNeoAnimaSpeed", "SPEED (Hires)"),
        ("ext_dd_sam3", "ForgeNeoAnimaDetailDaemon", "Detail Daemon (SAM3)"),
    ]),
    ("⑤ 잠재·샘플링", [
        ("i2i_image", "LoadImage", "i2i 이미지"),
        ("latent", "ForgeNeoLatentInput", "잠재"),
        ("sampler", "ForgeNeoKSamplerCNS", "샘플러"),
    ]),
    ("⑥ Hires", [
        ("hires", "ForgeNeoHiresFix", "Hires"),
        ("ext_vae_2x", "ForgeNeoAnimaVAE2x", "VAE 2x"),
        ("decode", "VAEDecode", "VAE 디코드"),
        ("ext_pid", "ForgeNeoAnimaPiD", "PiD"),
    ]),
    ("⑦ 업스케일", [
        ("upscale", "AIStudioUpscaleImage", "업스케일"),
    ]),
    ("⑧ SAM3", [
        ("sam3_mask", "ForgeNeoSAM3Mask", "SAM3 마스크"),
        ("sam3_detailer", "ForgeNeoSAM3Detailer", "SAM3 디테일"),
        ("ext_sam3_refine", "ForgeNeoSAM3Refine", "SAM3 Refine"),
        ("ext_sam3_tile", "ForgeNeoSAM3TileRepair", "SAM3 타일"),
    ]),
    ("⑨ 저장", [
        ("ext_degrid", "ForgeNeoAnimaVAEDeGrid", "VAE DeGrid"),
        ("save", "AIStudioSaveForge", "저장"),
    ]),
    ("⑩ 후처리", [  # R19: studio_logic.js POST_ROUTES re-link these slot-0 IMAGE links
        ("post_image", "LoadImage", "후처리 이미지"),
        ("post_upscale", "AIStudioUpscaleImage", "후처리 업스케일"),
        ("post_encode", "VAEEncode", "후처리 인코드"),
        ("post_hires", "ForgeNeoHiresFix", "후처리 Hires"),
        ("post_decode", "VAEDecode", "후처리 디코드"),
        ("post_sam3_mask", "ForgeNeoSAM3Mask", "후처리 SAM3 마스크"),
        ("post_sam3_detailer", "ForgeNeoSAM3Detailer", "후처리 SAM3 디테일"),
        ("post_save", "AIStudioSaveForge", "후처리 저장"),
    ]),
)


def _detailer_links(role: str, image: str, mask: str) -> list[tuple]:
    return [
        (image, 0, role, "image"), (mask, 0, role, "mask"),
        ("ext_dd_sam3", 0, role, "model"), ("negpip", 1, role, "clip"), ("vae", 0, role, "vae"),
        ("family_switch_pos", 0, role, "positive"), ("family_switch_neg", 0, role, "negative"),
    ]


def _hires_links(role: str, samples: str) -> list[tuple]:
    return [
        ("ext_speed_hires", 0, role, "model"), ("family_switch_pos", 0, role, "positive"),
        ("family_switch_neg", 0, role, "negative"), (samples, 0, role, "samples"),
        ("vae", 0, role, "base_vae"), ("negpip", 1, role, "base_clip"),
    ]


LINKS = (  # (source role, output slot, target role, input name); ids follow this order
    ("unet", 0, "family_switch_model", "a"),
    ("unet38", 0, "family_switch_model", "b"),
    ("family_switch_model", 0, "shift", "model"),
    ("shift", 0, "sched", "model"),
    ("sched", 0, "lora_stack", "model"),
    ("te", 0, "lora_stack", "clip"),  # R6: one native TE for both families
    ("lora_stack", 0, "ext_lora_block", "model"),
    ("lora_stack", 1, "ext_lora_block", "clip"),
    ("ext_lora_block", 0, "negpip", "model"),
    ("ext_lora_block", 1, "negpip", "clip"),
    ("negpip", 0, "dora", "model"),
    ("dora", 0, "ext_er_sde", "model"),
    ("negpip", 1, "pos_encode", "clip"),
    ("lora_stack", 2, "pos_encode", "text"),
    ("negpip", 1, "neg_encode", "clip"),
    ("lora_stack", 3, "neg_encode", "text"),
    # 3.8B prompts: the pack's own node as it is (no Forge prompt-editing syntax)
    ("dora", 0, "pos_encode38", "model"),
    ("negpip", 1, "pos_encode38", "native_clip"),
    ("qwen35", 0, "pos_encode38", "qwen35_clip"),
    ("lora_stack", 2, "pos_encode38", "prompt"),
    ("dora", 0, "neg_encode38", "model"),
    ("negpip", 1, "neg_encode38", "native_clip"),
    ("qwen35", 0, "neg_encode38", "qwen35_clip"),
    ("lora_stack", 3, "neg_encode38", "prompt"),
    ("pos_encode", 0, "family_switch_pos", "a"),
    ("pos_encode38", 0, "family_switch_pos", "b"),
    ("neg_encode", 0, "family_switch_neg", "a"),
    ("neg_encode38", 0, "family_switch_neg", "b"),
    # ④ extensions: the stack every pass shares, then each pass's own Detail Daemon / SPEED
    ("ext_er_sde", 0, "ext_optimal_scale", "model"),
    ("ext_optimal_scale", 0, "ext_guidance", "model"),
    ("negpip", 1, "ext_guidance", "clip"),
    ("family_switch_pos", 0, "ext_guidance", "positive"),
    ("family_switch_neg", 0, "ext_guidance", "negative"),
    ("ext_guidance", 0, "ext_skim", "model"),
    ("ext_skim", 0, "ext_dd", "model"),
    ("ext_dd", 0, "ext_speed", "model"),
    ("ext_skim", 0, "ext_dd_hires", "model"),
    ("ext_dd_hires", 0, "ext_speed_hires", "model"),
    ("ext_skim", 0, "ext_dd_sam3", "model"),
    ("vae", 0, "latent", "vae"),
    ("i2i_image", 0, "latent", "img2img_image"),
    ("i2i_image", 0, "latent", "inpaint_image"),
    ("i2i_image", 1, "latent", "inpaint_mask"),  # the mask editor's alpha
    ("ext_speed", 0, "sampler", "model"),
    ("family_switch_pos", 0, "sampler", "positive"),
    ("family_switch_neg", 0, "sampler", "negative"),
    ("latent", 0, "sampler", "latent_image"),
    *_hires_links("hires", "sampler"),
    ("hires", 1, "ext_vae_2x", "vae"),
    ("hires", 0, "decode", "samples"),
    ("ext_vae_2x", 0, "decode", "vae"),
    ("decode", 0, "ext_pid", "image"),
    ("hires", 0, "ext_pid", "latent"),
    ("ext_pid", 0, "upscale", "image"),
    ("upscale", 0, "sam3_mask", "image"),
    *_detailer_links("sam3_detailer", "upscale", "sam3_mask"),
    *_detailer_links("ext_sam3_refine", "sam3_detailer", "sam3_mask"),
    *_detailer_links("ext_sam3_tile", "ext_sam3_refine", "sam3_mask"),
    ("ext_sam3_tile", 0, "ext_degrid", "image"),
    ("ext_degrid", 0, "save", "images"),
    # ⑩ post-processing of a gallery image; model and conditioning come from the rows above
    ("post_image", 0, "post_upscale", "image"),
    ("post_upscale", 0, "post_encode", "pixels"),
    ("vae", 0, "post_encode", "vae"),
    *_hires_links("post_hires", "post_encode"),
    ("post_hires", 0, "post_decode", "samples"),
    ("post_hires", 1, "post_decode", "vae"),
    ("post_decode", 0, "post_sam3_mask", "image"),
    *_detailer_links("post_sam3_detailer", "post_decode", "post_sam3_mask"),
    ("post_sam3_detailer", 0, "post_save", "images"),
)

# ---------------------------------------------------------------------------------------------- node specs

COMBO = []  # a choice list from the user's ComfyUI (only the slot matters here)
# Nodes whose INPUT_TYPES cannot be read outside ComfyUI: ComfyUI 0.43.1 nodes.py and the pack's 3.8B nodes
# (vendor/comfyui_anima_3_8b v2.py / loader.py). (spec, outputs as (name, type)).
LOCAL_SPECS = {
    "UNETLoader": ({"required": {"unet_name": (COMBO,), "weight_dtype": (COMBO,)}}, (("MODEL", "MODEL"),)),
    "CLIPLoader": ({"required": {"clip_name": (COMBO,), "type": (COMBO,)}, "optional": {"device": (COMBO,)}},
                   (("CLIP", "CLIP"),)),
    "VAELoader": ({"required": {"vae_name": (COMBO,)}}, (("VAE", "VAE"),)),
    "LoadImage": ({"required": {"image": (COMBO, {"image_upload": True})}}, (("IMAGE", "IMAGE"), ("MASK", "MASK"))),
    "VAEDecode": ({"required": {"samples": ("LATENT",), "vae": ("VAE",)}}, (("IMAGE", "IMAGE"),)),
    "VAEEncode": ({"required": {"pixels": ("IMAGE",), "vae": ("VAE",)}}, (("LATENT", "LATENT"),)),
    "ForgeNeoAnima38V2Loader": ({"required": {"model_name": (COMBO,)}}, None),
    "ForgeNeoAnimaQwen35Loader": ({"required": {"qwen35_model": (COMBO,)}}, None),
    "ForgeNeoAnima38V2Prompt": ({"required": {
        "model": ("MODEL",), "native_clip": ("CLIP",), "qwen35_clip": ("CLIP",),
        "prompt": ("STRING", {"multiline": True, "dynamicPrompts": True}),
    }}, None),
}
# COMBO inputs whose choices depend on the user's ComfyUI (files, registered samplers/schedulers, detected
# features): a node gets widgets_values only when every one of these has a value from role_values().
INSTALL_COMBOS = {
    "ForgeNeoExtraSchedulers": {"owned_names"},
    "ForgeNeoKSamplerCNS": {"sampler_name", "scheduler"},
    "ForgeNeoHiresFix": {"upscale_method", "sampler_name", "scheduler", "base_sampler_name", "base_scheduler",
                         "checkpoint_name", "text_encoder_name", "vae_name"},
    "AIStudioUpscaleImage": {"model_name"},
    "ForgeNeoSAM3Mask": {"face_crop_detector"},
    "ForgeNeoSAM3Detailer": {"sampler_name", "scheduler"},
    "ForgeNeoSAM3Refine": {"sampler_name", "scheduler"},
    "ForgeNeoLoraBlockWeight": {"lora_name"},
    "ForgeNeoExtraSamplerSettings": {"owned_names"},
    "ForgeNeoAnimaVAE2x": {"vae_name"},
    "ForgeNeoAnimaPiD": {"ckpt_name"},
    "ForgeNeoAnimaVAEDeGrid": {"model_name"},
}


def _node_spec(class_type: str):
    """(entries [(name, kind, options, choices)], outputs [(name, type)], readable): entries in the frontend's input
    order; ``choices`` is the COMBO list (None for other kinds)."""
    local = LOCAL_SPECS.get(class_type)
    cls = NODE_CLASS_MAPPINGS.get(class_type)
    spec = local[0] if local else cls.INPUT_TYPES()
    entries = []
    for section in ("required", "optional"):
        for name, value in (spec.get(section) or {}).items():
            choices = value[0] if isinstance(value[0], list) else None
            entries.append((name, "COMBO" if choices is not None else value[0], value[1] if len(value) > 1 else {},
                            choices))
    if local and local[1] is not None:
        outputs = list(local[1])
    else:
        names = getattr(cls, "RETURN_NAMES", None) or cls.RETURN_TYPES
        outputs = list(zip(names, cls.RETURN_TYPES))
    return entries, outputs, local is None


def _is_widget(kind: str, options: dict) -> bool:
    return (kind in WIDGET_TYPES or kind == "COMBO") and not options.get("forceInput")


def _slots(entries) -> list[tuple[str, str, bool]]:
    """Input slots in frontend order: sockets, then widget inputs. (name, type, is_widget)"""
    sockets = [(name, kind, False) for name, kind, options, _ in entries if not _is_widget(kind, options)]
    widgets = [(name, kind, True) for name, kind, options, _ in entries
               if _is_widget(kind, options) and not options.get("socketless")]
    return sockets + widgets


def _widget_values(class_type: str, entries, readable: bool, values: dict) -> list | None:
    """The node's widgets_values, or None when any value depends on the user's ComfyUI."""
    if not readable:
        return None
    install = INSTALL_COMBOS.get(class_type, set())
    out = []
    for name, kind, options, choices in entries:
        if not _is_widget(kind, options):
            continue
        if any(key.endswith("upload") for key in options):
            return None  # the frontend adds an upload widget of its own
        if name in values:
            value = values[name]
        elif name in install:
            return None
        elif "default" in options:
            value = options["default"]
        elif choices:
            value = choices[0]  # a fixed list: the frontend picks its first choice
        else:
            return None  # no value the frontend would pick for sure
        out.append(value)
        if kind == "INT" and (name in SEED_WIDGETS or options.get("control_after_generate")):
            out.append(SEED_CONTROL)
    return out


# ---------------------------------------------------------------------------------------------- default values


def role_values(state: dict) -> dict[str, dict]:
    """role -> {widget: value} for the default panel state (as studio_logic.js mainInputs writes them)."""
    sampler, scheduler = state["sampler"], state["scheduler"]
    forge = state["forge_like"]
    noise = forge["master"] and forge["noise"]
    hires = {
        "enabled": state["hires"]["on"],
        "scale_by": state["hires"]["scale"],
        "upscale_method": state["hires"]["upscaler"],
        "steps": state["hires"]["steps"],
        "denoise": state["hires"]["denoise"],
        "shift": state["shift"],
        "sampler_name": state["hires"]["sampler"] or sampler,
        "scheduler": scheduler,
        "cfg": state["cfg"],
        "base_steps": state["steps"],
        "base_sampler_name": sampler,
        "base_scheduler": scheduler,
        # the fixed first choices of these file lists (generation.py ForgeNeoHiresFix)
        "checkpoint_name": "Use same checkpoint",
        "text_encoder_name": "Use same choices",
        "vae_name": "Use same choices",
        "forge_noise": noise,
    }
    detailer = {
        "enabled": state["sam3"]["on"],
        "denoise": state["sam3"]["denoise"],
        "steps": state["sam3"]["steps"] or state["steps"],
        "cfg": state["sam3"]["cfg"] or state["cfg"],
        "sampler_name": sampler,
        "scheduler": scheduler,
    }
    encode = {"steps": state["steps"], "forge_syntax": forge["master"] and forge["prompt_syntax"]}
    save = {"studio_state": json.dumps(state)}
    select = {"select": 0}  # the default state's unet is not a 3.8B file
    return {
        "family_switch_model": select,
        "family_switch_pos": select,
        "family_switch_neg": select,
        "shift": {"shift": state["shift"]},
        "lora_stack": {"positive": effective_prompt(state), "negative": state["negative"]},
        "negpip": {"enabled": state["negpip"]},
        "dora": {"enabled": True, "mode": DORA_FORGE_MODE if forge["master"] and forge["dora"] else state["dora_mode"]},
        "pos_encode": encode,
        "neg_encode": encode,
        "latent": {"mode": "txt2img", "width": state["width"], "height": state["height"],
                   "batch_size": state["batch"], "mask_blur": state["i2i"]["mask_blur"]},
        "sampler": {"steps": state["steps"], "cfg": state["cfg"], "sampler_name": sampler, "scheduler": scheduler,
                    "denoise": 1.0, "forge_noise": noise},
        "hires": hires,
        "post_hires": hires,
        "sam3_detailer": detailer,
        "post_sam3_detailer": detailer,
        "save": save,
        "post_save": save,
        "ext_speed": {"pass_label": "base"},
        "ext_speed_hires": {"pass_label": "hires"},
    }


def default_bypassed(state: dict) -> set[str]:
    """studio_logic.js stageModes(DEFAULT_STATE) roles that are bypassed (R7): saved that way as a safe default."""
    bypassed = {"unet38", "qwen35", "pos_encode38", "neg_encode38"}  # the default unet is not a 3.8B file
    if state["tab"] == "t2i":
        bypassed.add("i2i_image")
    if not state["sam3"]["on"]:
        bypassed |= {"sam3_mask", "sam3_detailer"}
    bypassed |= {role for _, row in ROWS for role, _, _ in row if role.startswith(("post_", "ext_"))}
    return bypassed


# ---------------------------------------------------------------------------------------------- building


def read_roles(path: Path = ROLES_JS) -> tuple[str, ...]:
    """``ROLES`` of studio_roles.js, in order."""
    source = path.read_text(encoding="utf-8")
    match = re.search(r"export const ROLES = Object\.freeze\(\[(.*?)\]\);", source, re.S)
    if not match:
        raise ValueError(f"{path}: no `export const ROLES = Object.freeze([...]);`")
    body = re.sub(r"//[^\n]*", "", match.group(1))
    return tuple(re.findall(r'"([a-z0-9_]+)"', body))


def _node_height(entries, outputs, class_type: str) -> int:
    if class_type == "LoadImage":
        return LOAD_IMAGE_HEIGHT
    sockets = sum(1 for _, kind, options, _ in entries if not _is_widget(kind, options))
    height = 2 * 10 + SLOT_HEIGHT * max(sockets, len(outputs))
    for name, kind, options, _ in entries:
        if not _is_widget(kind, options):
            continue
        height += MULTILINE_HEIGHT if options.get("multiline") else WIDGET_HEIGHT
        if kind == "INT" and name in SEED_WIDGETS:
            height += WIDGET_HEIGHT
    return height


def build_workflow() -> dict:
    roles = read_roles()
    placed = [role for _, row in ROWS for role, _, _ in row]
    if sorted(placed) != sorted(roles) or len(set(placed)) != len(placed):
        raise ValueError(f"ROWS roles differ from studio_roles.js ROLES: {sorted(set(placed) ^ set(roles))}")
    state = normalize_state({})
    values = role_values(state)
    bypassed = default_bypassed(state)

    nodes = []
    for class_type, pos, size, saved in UI_NODES:
        nodes.append({
            "id": len(nodes) + 1, "type": class_type, "pos": list(pos), "size": list(size), "flags": {},
            "order": len(nodes), "mode": MODE_ALWAYS, "inputs": [], "outputs": [], "properties": {},
            "widgets_values": [json.dumps(state) if saved is None else saved],
        })

    by_role, slots_of, groups = {}, {}, []
    top = ENGINE_TOP
    for title, row in ROWS:
        x = GROUP_PAD
        row_height = 0
        for role, class_type, node_title in row:
            entries, outputs, readable = _node_spec(class_type)
            width = NODE_WIDTHS.get(class_type, NODE_WIDTH)
            height = _node_height(entries, outputs, class_type)
            slots = _slots(entries)
            node = {
                "id": len(nodes) + 1, "type": class_type, "pos": [x, top + GROUP_HEADER], "size": [width, height],
                "flags": {"collapsed": True}, "order": len(nodes),
                "mode": MODE_BYPASS if role in bypassed else MODE_ALWAYS,
                "inputs": [{"name": name, "type": kind, **({"widget": {"name": name}} if widget else {}), "link": None}
                           for name, kind, widget in slots],
                "outputs": [{"name": name, "type": kind, "links": None} for name, kind in outputs],
                "title": node_title,
                "properties": {ROLE_PROPERTY: role},
            }
            widgets = _widget_values(class_type, entries, readable, values.get(role, {}))
            if widgets is not None:
                node["widgets_values"] = widgets
            nodes.append(node)
            by_role[role] = node
            slots_of[role] = [name for name, _, _ in slots]
            x += width + NODE_GAP
            row_height = max(row_height, height)
        group_height = GROUP_HEADER + row_height + GROUP_PAD
        groups.append({"id": len(groups) + 1, "title": title, "bounding": [0, top, x - NODE_GAP + GROUP_PAD, group_height],
                       "font_size": 24, "flags": {}})
        top += group_height + ROW_GAP

    links = []
    for source_role, slot, target_role, input_name in LINKS:
        source, target = by_role[source_role], by_role[target_role]
        index = slots_of[target_role].index(input_name)
        output, entry = source["outputs"][slot], target["inputs"][index]
        if entry["link"] is not None:
            raise ValueError(f"{target_role}.{input_name} is linked twice")
        kind = entry["type"] if entry["type"] != "*" else output["type"]
        if kind == "*" or "*" not in (output["type"], entry["type"]) and output["type"] != entry["type"]:
            raise ValueError(f"{source_role}[{slot}] {output['type']} -> {target_role}.{input_name} {entry['type']}")
        link_id = len(links) + 1
        links.append([link_id, source["id"], slot, target["id"], index, kind])
        output["links"] = [*(output["links"] or []), link_id]
        entry["link"] = link_id

    return {
        "last_node_id": len(nodes),
        "last_link_id": len(links),
        "nodes": nodes,
        "links": links,
        "groups": groups,
        "config": {},
        # 0.9: the three C4 columns (790 + 680 + 400 px, 960 px tall) fit ComfyUI at 1920x1080 (at 1 they are cut off)
        "extra": {"ds": {"scale": 0.9, "offset": [40, 60]}},
        "version": 0.4,
    }


def write_workflow(path: Path = OUT) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(build_workflow(), ensure_ascii=False, indent=2) + "\n"
    with open(path, "w", encoding="utf-8", newline="\n") as handle:
        handle.write(text)
    return path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, default=OUT, help="output path (default: the pack's example workflow)")
    args = parser.parse_args(argv)
    path = write_workflow(args.out).resolve()
    workflow = json.loads(path.read_text(encoding="utf-8"))
    print(f"wrote {path} ({len(workflow['nodes'])} nodes, {len(workflow['links'])} links, "
          f"{len(workflow['groups'])} groups)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
