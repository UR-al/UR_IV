# -*- coding: utf-8 -*-
"""Writes the Anima studio's extension table from the app's specs and the pack nodes' inputs.

Output (same object, both under ``comfy_custom_nodes/ai_studio_forge_parity/web/ai_studio/``):
- ``ext_schema.json``: what Python reads (``studio_state.py``).
- ``ext_schema.js``: ``export const EXT_SCHEMA = ...``, what the JS imports (``studio_ext.js``). The browser and
  node:test share that import and cannot both import a ``.json`` file; ComfyUI auto-loads every ``.js`` under ``web/``,
  so it only exports.
``tests/test_studio_ext_schema.py`` checks that this script reproduces both files and that they match the app's specs.

The pack cannot import ``core``, which is why the table is generated. Each entry:
``key`` (state ``ext.<key>``), ``label``, ``group``, ``role`` (the node's ``aistudio_role``; the pass copies of Detail
Daemon and SPEED are ``studio_ext.js``'s), ``class``, ``form`` (``json`` = the fields go into ``settings_json``,
``widgets`` = each field with an ``input`` writes that node input), ``on_key`` (the app's master-switch key, the state's
``on``), ``fixed`` (node inputs the panel does not show), ``active_keys`` (guidance: any of them on = the node runs) and
``families`` (the model families it runs on) and ``fields`` (``key``, ``label``, ``kind``, ``default``, ``min``/``max``,
``choices`` (empty = the node's file list, read from ``/object_info``), ``input``, ``section``).

Colorcraft is not in the table: its node takes the chain ``core/colorcraft.build_chain`` compiles (mask tags, combos,
per-pass filtering, ~1500 lines of app code), which the pack does not have.

Run: ``venv\\Scripts\\python.exe tools\\build_studio_ext_schema.py``
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

WEB = ROOT / "comfy_custom_nodes" / "ai_studio_forge_parity" / "web" / "ai_studio"
if not (WEB / "ext_schema.json").exists():  # the pack's studio_state reads the table on import; a first run has none
    (WEB / "ext_schema.json").write_text('{"extensions": []}\n', encoding="utf-8")

from comfy_custom_nodes.ai_studio_forge_parity import NODE_CLASS_MAPPINGS  # noqa: E402
from core import anima_guidance as ag  # noqa: E402
from core import anima_speed, extra_samplers, vae_degrid  # noqa: E402

MAX_SAFE_INT = 2 ** 53 - 1  # JS holds these numbers as doubles; a larger bound would round past the node's own max

# What runs outside Anima: the sampling helpers and SAM3 work on any model, the VAE ones on Krea2 because it shares
# Anima's VAE and latents. The rest (guidance suite, SPEED, LoRA blocks, ...) reach into Anima's blocks or are tuned to it.
ANY_MODEL = ("anima", "moss", "krea2")
QWEN_VAE = ("anima", "krea2")
EXT_FAMILIES = {
    "skim": ANY_MODEL, "detail_daemon": ANY_MODEL, "er_sde": ANY_MODEL, "sam3_refine": ANY_MODEL, "sam3_tile": ANY_MODEL,
    "vae_2x": QWEN_VAE, "degrid": QWEN_VAE, "pid": QWEN_VAE,
}

# (section, [(key, label)]): every non-fixed PERTURBATION_SPEC key exactly once, in display order
GUIDANCE_SECTIONS = (
    ("PAG·SEG", (
        ("guid_enabled", "켜기"), ("guid_attn_method", "방식"), ("guid_scale", "강도"),
        ("guid_block_indices", "블록"), ("guid_head_indices", "헤드"), ("guid_official_strength", "공식 강도"),
        ("guid_legacy_attn", "Legacy"), ("guid_legacy_strength", "Legacy 강도"), ("guid_seg_sigma", "SEG σ"),
    )),
    ("스케줄", (
        ("guid_start_percent", "시작"), ("guid_end_percent", "끝"), ("guid_rescale", "Rescale"),
        ("guid_rescale_mode", "Rescale 방식"), ("guid_auto_decay", "자동 감쇠"),
    )),
    ("SLG·S²", (
        ("guid_slg_on", "SLG"), ("guid_slg_scale", "강도"), ("guid_slg_blocks", "블록"), ("guid_slg_mode", "방식"),
        ("guid_s2_scale", "S² 강도"), ("guid_s2_ratio", "S² 비율"), ("guid_s2_blocks", "S² 블록"),
        ("guid_s2_start", "S² 시작"), ("guid_s2_end", "S² 끝"),
    )),
    ("CFG 기준", (("guid_cfg_mode", "방식"), ("guid_experimental_stack", "실험 스택"))),
    ("APG", (
        ("guid_apg_enabled", "APG"), ("guid_apg_eta", "η"), ("guid_apg_norm", "Norm"),
        ("guid_apg_momentum", "Momentum"), ("guid_apg_autooff", "자동 끔"),
    )),
    ("ADG", (("guid_adg_enabled", "ADG"), ("guid_adg_start", "시작"), ("guid_adg_interval", "간격"))),
    ("SMC", (
        ("guid_smc_master_enabled", "SMC"), ("guid_smc_enabled", "SMC 보조"), ("guid_smc_preset", "프리셋"),
        ("guid_smc_lambda", "λ"), ("guid_smc_k", "k"), ("guid_smc_mode", "방식"),
        ("guid_smc_adaptive_alpha", "적응 α"), ("guid_smc_adaptive_lambda", "적응 λ"),
    )),
    ("CWM", (("guid_cwm_enabled", "CWM"), ("guid_cwm_alpha_low", "α 저주파"), ("guid_cwm_alpha_high", "α 고주파"))),
    ("DCW·RDC", (
        ("guid_dcw_enabled", "DCW"), ("guid_dcw_lambda_low", "λ 저주파"), ("guid_dcw_lambda_high", "λ 고주파"),
        ("guid_rdc_enabled", "RDC"), ("guid_rdc_tau", "τ"), ("guid_rdc_alpha_ll", "α LL"), ("guid_rdc_alpha_hh", "α HH"),
    )),
    ("DAVE", (
        ("guid_dave_enabled", "DAVE"), ("guid_dave_strength", "강도"), ("guid_dave_tau", "τ"),
        ("guid_dave_blocks", "블록"),
    )),
    ("CNS", (
        ("guid_cns_enabled", "CNS"), ("guid_cns_strength", "강도"), ("guid_cns_gamma_power", "γ 지수"),
        ("guid_cns_gamma_scale", "γ 배율"),
    )),
    ("Mod", (
        ("guid_mod_enabled", "Mod"), ("guid_mod_clip_model", "CLIP 모델"), ("guid_mod_weight", "강도"),
        ("guid_mod_start_layer", "시작 층"), ("guid_mod_end_layer", "끝 층"), ("guid_mod_base_source", "기준"),
        ("guid_mod_base_prompt", "기준 프롬프트"), ("guid_mod_positive_prompt", "프롬프트"),
        ("guid_mod_negative_source", "네거티브 기준"), ("guid_mod_negative_prompt", "네거티브"),
        ("guid_mod_adapter_mode", "어댑터"), ("guid_mod_adapter_path", "어댑터 경로"),
    )),
    ("TSR", (("guid_tsr_enabled", "TSR"), ("guid_tsr_k", "k"), ("guid_tsr_sigma", "σ"))),
    ("MG", (
        ("guid_mg_enabled", "MG"), ("guid_mg_alpha", "α"), ("guid_mg_beta", "β"), ("guid_mg_normalize", "정규화"),
        ("guid_mg_min", "최소"), ("guid_mg_max", "최대"),
    )),
    ("HiGS", (
        ("guid_higs_enabled", "HiGS"), ("guid_higs_weight", "강도"), ("guid_higs_eta", "η"), ("guid_higs_alpha", "α"),
        ("guid_higs_cutoff", "Cutoff"), ("guid_higs_t_min", "t 최소"), ("guid_higs_t_max", "t 최대"),
    )),
    ("HiFlow", (
        ("guid_hiflow_enabled", "HiFlow"), ("guid_hiflow_alpha", "α"), ("guid_hiflow_beta", "β"),
        ("guid_hiflow_cutoff", "Cutoff"),
    )),
)


def _spec_field(entry: tuple, label: str, *, input_name: str | None = None, section: str | None = None) -> dict:
    """One app spec tuple ``(key, kind, default, extra)`` as a table field."""
    key, kind, default, extra = entry
    field: dict = {"key": key, "label": label, "kind": kind, "default": default}
    if kind in ("int", "float"):
        field["min"], field["max"] = extra
    if kind == "choice":
        field["choices"] = list(extra)
    if input_name:
        field["input"] = input_name
    if section:
        field["section"] = section
    return field


def _spec_fields(spec, labels: dict, inputs: dict | None = None) -> list[dict]:
    by_key = {entry[0]: entry for entry in spec}
    return [_spec_field(by_key[key], label, input_name=(inputs or {}).get(key)) for key, label in labels.items()]


def _guidance_fields() -> list[dict]:
    by_key = {entry[0]: entry for entry in ag.PERTURBATION_SPEC if entry[1] != ag.FIXED}
    fields = [_spec_field(by_key[key], label, section=section)
              for section, items in GUIDANCE_SECTIONS for key, label in items]
    missing = set(by_key) ^ {field["key"] for field in fields}
    if missing or len(fields) != len(by_key):
        raise ValueError(f"GUIDANCE_SECTIONS differ from PERTURBATION_SPEC: {sorted(missing)}")
    return fields


def _inputs(class_type: str) -> dict:
    spec = NODE_CLASS_MAPPINGS[class_type].INPUT_TYPES()
    return {**spec.get("required", {}), **spec.get("optional", {})}


def _node_field(class_type: str, name: str, label: str, *, key: str | None = None, as_input: bool = True,
                default=None) -> dict:
    """A table field from a node input: kind, range, default and fixed choices are the node's own."""
    spec = _inputs(class_type)[name]
    kind, options = spec[0], spec[1] if len(spec) > 1 else {}
    field: dict = {"key": key or name, "label": label}
    if isinstance(kind, list):
        field.update(kind="choice", default=options.get("default", kind[0]), choices=list(kind))
    elif kind == "BOOLEAN":
        field.update(kind="bool", default=bool(options.get("default", False)))
    elif kind == "INT":
        field.update(kind="int", default=int(options.get("default", 0)), min=max(-MAX_SAFE_INT, options["min"]),
                     max=min(MAX_SAFE_INT, options["max"]))
    elif kind == "FLOAT":
        field.update(kind="float", default=float(options.get("default", 0.0)), min=float(options["min"]),
                     max=float(options["max"]))
    elif kind == "STRING":
        field.update(kind="text", default=str(options.get("default", "")))
    else:
        raise ValueError(f"{class_type}.{name}: {kind} is not a panel field")
    if default is not None:
        field["default"] = default
    if as_input:
        field["input"] = name
    return field


def _file_field(class_type: str, name: str, label: str, *, key: str | None = None, default: str = "") -> dict:
    """A COMBO of the user's files: no fixed choices (the panel reads ``/object_info``), nothing picked by default."""
    return {"key": key or name, "label": label, "kind": "choice", "default": default, "choices": [], "input": name}


def _node_fields(class_type: str, labels: dict) -> list[dict]:
    return [_node_field(class_type, name, label) for name, label in labels.items()]


def _flag(key: str, label: str) -> dict:
    """A switch of the panel only: it picks the passes, the node has no such input."""
    return {"key": key, "label": label, "kind": "bool", "default": False}


def _speed_fields() -> list[dict]:
    class_type = anima_speed.COMFY_NODE_CLASS
    return [
        *_node_fields(class_type, {
            "mode": "방식", "threshold": "전환 σ", "preset": "프리셋", "scales": "해상도 비율", "delta": "δ",
            "sigma_divisor": "σ 나눔값", "manual_sigmas": "수동 σ", "adaptive": "적응형", "transform": "변환",
            "spectrum_A": "A", "spectrum_beta": "β", "seed": "시드",
        }),
        _flag("hires", "Hires에도"),
        _flag("apply_img2img", "i2i에도"),
    ]


def _degrid_fields() -> list[dict]:
    class_type = vae_degrid.COMFY_NODE_CLASS
    return [
        _file_field(class_type, "model_name", "모델", key="model", default=vae_degrid.COMFY_AUTO),
        _node_field(class_type, "mode", "방식"),
        _node_field(class_type, "strength", "강도"),
        _node_field(class_type, "tile", "타일"),
        _flag("apply_img2img", "i2i에도"),
    ]


def _tile_fields() -> list[dict]:
    """``settings_json`` keys of the tile repair node, ranged as the detailer inputs they end up in."""
    class_type = "ForgeNeoSAM3TileRepair"
    defaults = json.loads(_inputs(class_type)["settings_json"][1]["default"])
    detailer = "ForgeNeoSAM3Detailer"
    fields = []
    for key, source, label in (
            ("mask_mode", "mask_mode", "마스크"), ("tile_width", "custom_width", "타일 폭"),
            ("tile_height", "custom_height", "타일 높이"), ("tile_padding", "mask_padding", "패딩"),
            ("denoise", "denoise", "디노이즈"), ("noise_multiplier", "noise_multiplier", "노이즈 배율"),
            ("fill_mode", "fill_mode", "채우기"), ("grow_mask_by", "grow_mask_by", "마스크 확장"),
            ("inpaint_prompt", "inpaint_prompt", "프롬프트"), ("negative_prompt", "negative_prompt", "네거티브")):
        fields.append(_node_field(detailer, source, label, key=key, as_input=False, default=defaults[key]))
    return fields


def build_schema() -> dict:
    guidance_keys = (*ag._ACTIVATION_KEYS[ag.SCRIPT_PERTURBATION], "guid_rdc_enabled")  # the compiler's active_keys
    extensions = [
        {"key": "guidance", "label": "가이던스", "group": "가이던스", "role": "ext_guidance",
         "class": "ForgeNeoAnimaGuidanceSuite", "form": "json", "fixed": {"enabled": True},
         "active_keys": list(guidance_keys), "fields": _guidance_fields()},
        {"key": "optimal_scale", "label": "Optimal Scale", "group": "가이던스", "role": "ext_optimal_scale",
         "class": "ForgeNeoAnimaOptimalScale", "form": "widgets", "on_key": "ocfg_enabled", "fixed": {"enabled": True},
         "fields": _spec_fields(ag.OPTIMAL_SCALE_SPEC, {"ocfg_blend": "Blend", "ocfg_start": "시작", "ocfg_end": "끝"},
                                {"ocfg_blend": "blend", "ocfg_start": "start_percent", "ocfg_end": "end_percent"})},
        {"key": "skim", "label": "Skim CFG", "group": "가이던스", "role": "ext_skim",
         "class": "ForgeNeoSkimmedCFG", "form": "widgets", "on_key": "skim_enabled", "fixed": {"enabled": True},
         "fields": _spec_fields(ag.SKIMMED_SPEC, {
             "skim_skimming_cfg": "Skim CFG", "skim_full_skim_negative": "네거티브 전부",
             "skim_disable_flipping_filter": "반전 필터 끔", "skim_start_percent": "시작", "skim_end_percent": "끝",
             "skim_flip_at": "반전 지점",
         }, {
             "skim_skimming_cfg": "skimming_cfg", "skim_full_skim_negative": "full_skim_negative",
             "skim_disable_flipping_filter": "disable_flipping_filter", "skim_start_percent": "start_percent",
             "skim_end_percent": "end_percent", "skim_flip_at": "flip_percent",
         })},
        {"key": "detail_daemon", "label": "Detail Daemon", "group": "디테일", "role": "ext_dd",
         "class": "ForgeNeoAnimaDetailDaemon", "form": "json", "on_key": "dd_enabled", "fixed": {"enabled": True},
         "fields": _spec_fields(ag.DETAIL_DAEMON_SPEC, {
             "dd_amount": "강도", "dd_start": "시작", "dd_end": "끝", "dd_bias": "Bias", "dd_exponent": "지수",
             "dd_start_offset": "시작 오프셋", "dd_end_offset": "끝 오프셋", "dd_fade": "Fade", "dd_smooth": "Smooth",
             "dd_hires": "Hires 패스만",
         })},
        {"key": "lora_block", "label": "LoRA 블록", "group": "디테일", "role": "ext_lora_block",
         "class": "ForgeNeoLoraBlockWeight", "form": "widgets", "fixed": {"enabled": True},
         "fields": [_file_field("ForgeNeoLoraBlockWeight", "lora_name", "LoRA"),
                    *_node_fields("ForgeNeoLoraBlockWeight", {
                        "strength_model": "모델 강도", "strength_clip": "CLIP 강도", "inverse": "반전", "A": "A", "B": "B",
                        "block_vector": "블록 값",
                    })]},
        {"key": "speed", "label": "SPEED", "group": "샘플러·VAE", "role": "ext_speed",
         "class": anima_speed.COMFY_NODE_CLASS, "form": "widgets", "on_key": "enabled",
         "fixed": {"enabled": True, "img2img_rescale": anima_speed.COMFY_IMG2IMG_RESCALE}, "fields": _speed_fields()},
        {"key": "er_sde", "label": "ER SDE", "group": "샘플러·VAE", "role": "ext_er_sde",
         "class": extra_samplers.COMFY_SETTINGS_NODE, "form": "widgets", "fixed": {},
         "fields": _spec_fields(extra_samplers.ARG_SPEC, {
             "xsamp_max_stage": "차수", "xsamp_eta": "η", "xsamp_noise_window": "잡음 구간", "xsamp_noise_start": "구간 시작",
             "xsamp_noise_end": "구간 끝",
         }, {f"{extra_samplers.SPEC_PREFIX}{name}": name for name in extra_samplers.ARG_NAMES})},
        {"key": "vae_2x", "label": "VAE 2x", "group": "샘플러·VAE", "role": "ext_vae_2x",
         "class": "ForgeNeoAnimaVAE2x", "form": "widgets", "fixed": {"enabled": True},
         "fields": [_file_field("ForgeNeoAnimaVAE2x", "vae_name", "VAE"),
                    *_node_fields("ForgeNeoAnimaVAE2x", {
                        "output_mode": "출력", "blur_sigma": "Blur σ", "latent_renorm": "Latent 재정규화",
                    })]},
        {"key": "degrid", "label": "VAE DeGrid", "group": "샘플러·VAE", "role": "ext_degrid",
         "class": vae_degrid.COMFY_NODE_CLASS, "form": "widgets", "on_key": "enabled",
         "fixed": {"enabled": True, **vae_degrid.COMFY_OPTIONS, "forge_quantize": vae_degrid.COMFY_FORGE_QUANTIZE},
         "fields": _degrid_fields()},
        # The node refuses anything but these fixed values (ComfyUI's PiD API exposes no such override).
        {"key": "pid", "label": "PiD", "group": "샘플러·VAE", "role": "ext_pid",
         "class": "ForgeNeoAnimaPiD", "form": "widgets",
         "fixed": {"enabled": True, "dtype": "bf16", "tile_latent": 64, "tile_overlap": 16, "compile": False,
                   "use_calib": False, "attention": "auto"},
         "fields": [_file_field("ForgeNeoAnimaPiD", "ckpt_name", "체크포인트"),
                    *_node_fields("ForgeNeoAnimaPiD", {"steps": "스텝", "sigma": "σ", "seed": "시드"})]},
        {"key": "sam3_refine", "label": "SAM3 Refine", "group": "SAM3", "role": "ext_sam3_refine",
         "class": "ForgeNeoSAM3Refine", "form": "widgets", "fixed": {"enabled": True},
         "fields": _node_fields("ForgeNeoSAM3Refine", {
             "denoise": "디노이즈", "mask_mode": "마스크", "fill_mode": "채우기", "only_masked": "마스크 영역만",
             "mask_padding": "패딩", "grow_mask_by": "마스크 확장", "noise_multiplier": "노이즈 배율",
             "use_custom_size": "크기 지정", "custom_width": "폭", "custom_height": "높이",
             "inpaint_prompt": "프롬프트", "negative_prompt": "네거티브",
         })},
        {"key": "sam3_tile", "label": "SAM3 타일", "group": "SAM3", "role": "ext_sam3_tile",
         "class": "ForgeNeoSAM3TileRepair", "form": "json", "fixed": {"enabled": True}, "fields": _tile_fields()},
    ]
    return {"extensions": [{**item, "families": list(EXT_FAMILIES.get(item["key"], ("anima",)))} for item in extensions]}


def write_schema(folder: Path = WEB) -> tuple[Path, Path]:
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    text = json.dumps(build_schema(), ensure_ascii=False, indent=1)
    json_path, js_path = folder / "ext_schema.json", folder / "ext_schema.js"
    with open(json_path, "w", encoding="utf-8", newline="\n") as handle:
        handle.write(text + "\n")
    with open(js_path, "w", encoding="utf-8", newline="\n") as handle:
        handle.write("// Generated by tools/build_studio_ext_schema.py from the app's specs: do not edit.\n"
                     "// The same object as ext_schema.json (no side effects: ComfyUI loads every .js under web/).\n"
                     f"export const EXT_SCHEMA = {text};\n")
    return json_path, js_path


def main() -> int:
    json_path, js_path = write_schema()
    schema = json.loads(json_path.read_text(encoding="utf-8"))
    fields = sum(len(item["fields"]) for item in schema["extensions"])
    print(f"wrote {json_path.resolve()} and {js_path.resolve()} ({len(schema['extensions'])} extensions, {fields} fields)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
