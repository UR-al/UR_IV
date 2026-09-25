"""Compile Forge-style generation payloads into ComfyUI API graphs.

The desktop UI intentionally owns one generation payload contract.  Forge can
consume that contract directly, while ComfyUI needs an explicit graph.  This
module is the narrow translation boundary: it is pure (no HTTP/Qt), never
mutates caller data, and rejects enabled features that the connected ComfyUI
cannot execute instead of silently dropping them.
"""
from __future__ import annotations

import copy
import json
import os
import re
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Iterable, Mapping, Optional, Sequence

from core import anima38
from core.comfy_node_classes import (
    CHECKPOINT_LOADER_NODES,
    CUSTOM_SAMPLER_NODES,
    IMAGE_SAVE_NODES,
    MODEL_LOADER_INPUTS,
    SAMPLER_NODES,
    SEMANTIC_ENCODER_NODES,
    TEXT_ENCODER_NODES,
)
from core.comfy_seed import concrete_seed
from core.lenient_numbers import finite_float as _float, lenient_int as _int


class WorkflowCompileError(RuntimeError):
    """A Forge request cannot be represented by the target ComfyUI runtime."""


@dataclass(frozen=True)
class LoraSpec:
    name: str
    strength_model: float
    strength_clip: float


@dataclass(frozen=True)
class _Anima38Plan:
    """Resolved loader/conditioning decisions for one immutable payload copy."""

    loader_kind: str = "standard"
    conditioning_kind: str = "native"
    model_name: str = ""
    native_modules: tuple[str, ...] = ()
    native_clip_modules: tuple[str, ...] = ()
    vae_modules: tuple[str, ...] = ()
    unknown_modules: tuple[str, ...] = ()
    qwen35_model: str = ""
    adapter_name: str = ""
    settings: anima38.Anima38Settings = anima38.DEFAULT_SETTINGS
    is_anima: bool = False

    @property
    def semantic(self) -> bool:
        return self.conditioning_kind in {"v1", "v2"}

    @property
    def v2(self) -> bool:
        return self.loader_kind == "v2"


_LORA_RE = re.compile(
    r"<lora\s*:\s*([^:>]+?)\s*(?::\s*([^:>]+?))?\s*(?::\s*([^>]+?))?\s*>",
    re.IGNORECASE,
)
# Node vocabularies are shared with the workflow picker (core/comfy_node_classes).
_SAMPLERS = SAMPLER_NODES
_SAVE_NODES = IMAGE_SAVE_NODES
_SEMANTIC_ENCODERS = SEMANTIC_ENCODER_NODES
_TEXT_ENCODERS = TEXT_ENCODER_NODES
# Loaders that read an uploaded input file; their choice list can be older than
# the upload that happened just before compilation (see ``validate``).
_UPLOAD_INPUT_LOADERS = frozenset({"LoadImage", "LoadImageMask"})
# Bundled LoRA nodes that already handle ANIMA block layouts themselves and
# can stay in an Anima custom workflow without a class remap.
_ANIMA_SAFE_LORA_NODES = frozenset({
    "ForgeNeoAnimaLoraLoader", "ForgeNeoAnimaLoraLoaderModelOnly",
    "ForgeNeoLoraBlockWeight",
})
# Core LoRA loader remapped to the bundled ANIMA-compatible equivalent with the
# same inputs.  Core LoraLoaderModelOnly stays an explicit error (the bundled
# ForgeNeoAnimaLoraLoaderModelOnly is the supported model-only path).
_ANIMA_LORA_REMAP = {
    "LoraLoader": "ForgeNeoAnimaLoraLoader",
}
# LoRA nodes with a CLIP input/output (index 1) chained like the model.
_CLIP_LORA_NODES = frozenset({
    "LoraLoader", "ForgeNeoAnimaLoraLoader", "ForgeNeoLoraBlockWeight",
})

# Forge/A1111 sampler labels → ComfyUI KSampler names.  The bundled node pack
# is copied into ComfyUI and cannot import core, so
# comfy_custom_nodes/ai_studio_forge_parity/generation.py keeps the same
# tables for user workflows; tests/test_comfy_sampler_aliases.py pins both.
_FORGE_SAMPLER_ALIASES = {
    "euler": "euler", "euler a": "euler_ancestral", "euler ancestral": "euler_ancestral",
    "lms": "lms", "heun": "heun", "dpm2": "dpm_2", "dpm2 a": "dpm_2_ancestral",
    "dpm++ 2s a": "dpmpp_2s_ancestral", "dpm++ 2m": "dpmpp_2m",
    "dpm++ sde": "dpmpp_sde", "dpm++ 2m sde": "dpmpp_2m_sde",
    "dpm++ 2m sde heun": "dpmpp_2m_sde_heun", "dpm++ 3m sde": "dpmpp_3m_sde",
    "dpm fast": "dpm_fast", "dpm adaptive": "dpm_adaptive",
    "unipc": "uni_pc", "uni pc": "uni_pc", "uni_pc": "uni_pc",
    "lcm": "lcm", "ddim": "ddim", "er sde": "er_sde", "er-sde": "er_sde",
}
# Old combined labels ("DPM++ 2M Karras") carry the scheduler as a suffix.
_FORGE_SAMPLER_SCHEDULER_SUFFIXES = (
    (" karras", "karras"), (" exponential", "exponential"),
    (" sgm uniform", "sgm_uniform"), (" beta", "beta"),
)
_FORGE_SCHEDULER_ALIASES = {
    "karras": "karras", "exponential": "exponential",
    "sgm uniform": "sgm_uniform", "sgm_uniform": "sgm_uniform",
    "simple": "simple", "normal": "normal",
    "ddim uniform": "ddim_uniform", "ddim_uniform": "ddim_uniform",
    "beta": "beta",
    "beta57": "beta57", "beta 57": "beta57",
    "beta57 (res4lyf)": "beta57", "beta 57 (res4lyf)": "beta57",
}
_FORGE_SAME_SCHEDULER = frozenset({"use same scheduler", "same", "automatic", "auto", ""})
# ComfyUI comfy/samplers.py ``KSampler.DISCARD_PENULTIMATE_SIGMA_SAMPLERS``:
# these samplers build their sigma array for steps+1 and drop the next-to-last
# sigma, so one step index is a different noise level than for any other
# sampler.  Passes that split one schedule must stay on the same side.
_PENULTIMATE_SIGMA_DISCARD_SAMPLERS = frozenset({
    "dpm_2", "dpm_2_ancestral", "uni_pc", "uni_pc_bh2",
})
# Fallbacks for the CNS sampler inputs = the original CNSSamplerPatch defaults
# (strength 1.0, gamma_power 0.5, gamma_scale 2.0; the pack's
# ``guidance_cns.CNS_INPUTS``) — origin:
# namemechan/comfyui-cns_sampler_patch@42278b13:cns_sampler_patch.py:396-437
_CNS_DEFAULTS: Mapping[str, float] = MappingProxyType({
    "cns_strength": 1.0, "cns_gamma_power": 0.5, "cns_gamma_scale": 2.0,
})


def _cns_inputs(options: Mapping[str, Any]) -> dict[str, Any]:
    """ForgeNeoKSamplerCNS/ForgeNeoHiresFix CNS inputs from sampler options."""
    return {
        "cns_enabled": _bool(options.get("cns_enabled")),
        **{key: _float(options.get(key), default) for key, default in _CNS_DEFAULTS.items()},
    }


# A class only the per-step CNS pack (1.4.0+) publishes: the original's SAMPLER
# node.  Pack 1.3.0 coloured only the initial noise, only on ForgeNeoKSamplerCNS/
# ForgeNeoHiresFix (plan §5.2 #1), so a KSamplerAdvanced graph got no CNS at all —
# yet its suite/sampler input contracts are identical to 1.4.0's, so validate()'s
# contract check cannot see a stale pack.  This marker can.
_PER_STEP_CNS_MARKER = "ForgeNeoCNSSamplerPatch"
_CNS_SAMPLER_CARRIERS = frozenset({"ForgeNeoKSamplerCNS", "ForgeNeoHiresFix"})
# PAG legacy (legacy strength) and head indices reach the original PAG node from pack
# 1.4.0 on.  1.3.0's ForgeNeoAnimaGuidanceSuite/ForgeNeoAnimaSafePAG take the same
# inputs but raise on both inside patch() (1.3.0 guidance.py:1078-1085, :862-865), so
# the contract check passes a stale pack and the queued run fails.  1.4.0 adds no
# PAG-only class; the per-step CNS node ships from that same 1.4.0 on.
_PAG_ORIGIN_MARKER = _PER_STEP_CNS_MARKER


def _suite_settings(inputs: Mapping[str, Any]) -> Mapping[str, Any]:
    """A ForgeNeoAnimaGuidanceSuite node's settings_json ({} when unreadable)."""
    raw = inputs.get("settings_json")
    try:
        settings = json.loads(raw) if isinstance(raw, str) else {}
    except ValueError:
        return {}
    return settings if isinstance(settings, Mapping) else {}


def _graph_enables_cns(workflow: Mapping[str, Any]) -> bool:
    """Whether a compiled graph asks the pack for CNS (suite or sampler inputs)."""
    for node in workflow.values():
        if not isinstance(node, Mapping):
            continue
        inputs = node.get("inputs")
        if not isinstance(inputs, Mapping):
            continue
        class_type = node.get("class_type")
        if class_type == "ForgeNeoAnimaGuidanceSuite":
            if (
                _bool(inputs.get("enabled"), True)
                and _bool(_suite_settings(inputs).get("guid_cns_enabled"))
            ):
                return True
        elif class_type in _CNS_SAMPLER_CARRIERS:
            value = inputs.get("cns_enabled")
            if not _is_link(value) and _bool(value):
                return True
    return False


def _graph_uses_pag_legacy_or_heads(workflow: Mapping[str, Any]) -> bool:
    """Whether a graph asks for PAG legacy/head indices — what pack 1.3.0 raises on."""
    for node in workflow.values():
        if not isinstance(node, Mapping):
            continue
        inputs = node.get("inputs")
        if not isinstance(inputs, Mapping):
            continue
        class_type = node.get("class_type")
        if class_type == "ForgeNeoAnimaGuidanceSuite":
            settings = _suite_settings(inputs)
            method = settings.get("guid_attn_method")
            method = str("PAG" if method is None else method).strip().casefold()
            if (
                _bool(inputs.get("enabled"), True) and _bool(settings.get("guid_enabled"))
                and method == "pag"
                and (
                    _bool(settings.get("guid_legacy_attn"))
                    or str(settings.get("guid_head_indices") or "").strip()
                )
            ):
                return True
        elif class_type == "ForgeNeoAnimaSafePAG":
            # 1.3.0's node returns the model before its head check when off or at zero.
            enabled, heads = inputs.get("enabled"), inputs.get("head_indices")
            if (
                not _is_link(enabled) and _bool(enabled)
                and not _is_link(heads) and str(heads or "").strip()
                and all(
                    _is_link(inputs.get(key)) or _float(inputs.get(key), 1.0) != 0.0
                    for key in ("scale", "perturbation_strength")
                )
            ):
                return True
    return False


def unsupported_sampler_message(class_type: str) -> Optional[str]:
    """Why a custom workflow sampler cannot receive the app's payload, if so.

    Only payload mapping decides this.  CNS does not: it is the pack's
    SAMPLER_SAMPLE wrapper on the MODEL, which colours any KSAMPLER run.
    Open item (not parity): plan §5.3 A / §8.3 APP-COMP ask to allow
    SamplerCustom(Advanced), the original CNS node's host; that needs a
    BasicScheduler/CFGGuider/RandomNoise payload mapping and is deferred until
    its scope is decided.
    """
    if class_type in CUSTOM_SAMPLER_NODES:
        return (
            f"{class_type} custom workflow는 Forge payload의 steps/CFG/denoise를 "
            "안전하게 자동 매핑할 수 없습니다. KSampler/KSamplerAdvanced를 쓰거나 "
            "완성된 그래프를 run_workflow로 실행하세요."
        )
    return None


def _bool(value: Any, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    return str(value).strip().casefold() not in {"", "0", "false", "no", "off", "none"}


def _clean_prompt_after_loras(text: str) -> str:
    text = _LORA_RE.sub("", text)
    text = re.sub(r"(?:\s*,\s*){2,}", ", ", text)
    return text.strip(" \t\r\n,")


def parse_lora_tags(*prompts: str) -> tuple[list[LoraSpec], list[str]]:
    """Return ordered LoRA requests and prompts with loader syntax removed."""
    loras: list[LoraSpec] = []
    cleaned: list[str] = []
    for prompt in prompts:
        text = str(prompt or "")
        for match in _LORA_RE.finditer(text):
            name = match.group(1).strip()
            if not name:
                continue
            model_strength = _float(match.group(2), 1.0)
            clip_strength = _float(match.group(3), model_strength)
            loras.append(LoraSpec(name, model_strength, clip_strength))
        cleaned.append(_clean_prompt_after_loras(text))
    return loras, cleaned


def _is_link(value: Any) -> bool:
    return isinstance(value, list) and len(value) >= 2


def _filename(value: Any) -> str:
    return os.path.basename(str(value or "").replace("\\", "/")).strip()


class _Graph:
    def __init__(self, initial: Optional[Mapping[str, Any]] = None):
        self.nodes: dict[str, dict] = copy.deepcopy(dict(initial or {}))
        numeric = [int(str(key)) for key in self.nodes if str(key).isdigit()]
        self._next = max(numeric, default=0) + 1

    def add(self, class_type: str, inputs: Mapping[str, Any], title: str = "") -> str:
        while str(self._next) in self.nodes:
            self._next += 1
        node_id = str(self._next)
        self._next += 1
        node = {"class_type": class_type, "inputs": copy.deepcopy(dict(inputs))}
        if title:
            node["_meta"] = {"title": title}
        self.nodes[node_id] = node
        return node_id


class ComfyWorkflowCompiler:
    """Compile a canonical app payload for one concrete ComfyUI capability set."""

    def __init__(
        self,
        object_info: Optional[Mapping[str, Any]] = None,
        *,
        sam3_keep_in_ram: bool = True,
        object_info_cached: bool = False,
    ):
        self.object_info = None if object_info is None else dict(object_info)
        # True when ``object_info`` is a backend snapshot rather than a fetch
        # made for this compile; the backend then retries a failed compile
        # once with a fresh document (backends/comfyui_backend._compile_graph).
        self.object_info_cached = bool(object_info_cached)
        # App setting (core/comfy_sam3_cache_policy.py), the Comfy counterpart
        # of Forge's ``sam3_unload_keep_in_ram``: after "Unload after", keep the
        # SAM3 bundle in ComfyUI's CPU RAM for the next job, or free it.
        self.sam3_keep_in_ram = bool(sam3_keep_in_ram)

    # ---- public ---------------------------------------------------------

    def compile(
        self,
        mode: str,
        model_name: str,
        payload: Mapping[str, Any],
        *,
        workflow: Optional[Mapping[str, Any]] = None,
        workflow_controls: Optional[Mapping[str, Any]] = None,
        uploaded_image: str = "",
        uploaded_mask: str = "",
    ) -> dict:
        normalized = {
            "t2i": "txt2img", "txt2img": "txt2img",
            "i2i": "img2img", "img2img": "img2img",
            "inpaint": "inpaint",
        }.get(str(mode or "").strip().casefold())
        if normalized is None:
            raise WorkflowCompileError(f"지원하지 않는 ComfyUI 생성 모드입니다: {mode!r}")
        if not isinstance(payload, Mapping):
            raise WorkflowCompileError("생성 payload는 객체여야 합니다.")

        local_payload = copy.deepcopy(dict(payload))
        from core.spectrum_settings import validate_spectrum_payload
        try:
            validate_spectrum_payload(local_payload, self.object_info or {})
        except ValueError as exc:
            raise WorkflowCompileError(str(exc)) from exc
        detail_passes = local_payload.get("_comfy_detail_passes", [])
        if not isinstance(detail_passes, list) or len(detail_passes) > 3 or any(
            not isinstance(target, str) or not target.strip() or len(target) > 128
            for target in detail_passes
        ):
            raise WorkflowCompileError("추가 SAM3 보정 패스는 최대 3개의 비어 있지 않은 대상 문자열이어야 합니다.")
        if detail_passes:
            sam_state = self._sam3_state(local_payload)
            if sam_state is None or str(sam_state.get("sam3_mode") or "Inpaint") != "Inpaint":
                raise WorkflowCompileError("추가 보정 패스를 사용하려면 SAM3 Inpaint를 켜세요.")
        if normalized != "txt2img":
            local_payload.setdefault("denoising_strength", 0.75)
        plan_model, rewrite_model = model_name, model_name
        if workflow is not None:
            # A workflow whose model loader the app cannot rewrite (GGUF and
            # other custom loaders; the picker shows it as "workflow fixes the
            # model") owns its model: the UI's disabled combo value is ignored
            # and the workflow's own model identifies the family.  An empty UI
            # selection means the workflow's own loader model, for rewritable
            # loaders too (identity and model written).
            plan_model, rewrite_model = self._custom_workflow_model(
                workflow, model_name, workflow_controls=workflow_controls,
            )
        loras, anima_plan = self._prepare_payload(plan_model, local_payload)

        if workflow is None:
            graph = self._compile_default(
                normalized, model_name, local_payload, loras, anima_plan,
                uploaded_image=uploaded_image, uploaded_mask=uploaded_mask,
            )
        else:
            graph = self._compile_custom(
                normalized, rewrite_model, local_payload, loras, workflow, anima_plan,
                uploaded_image=uploaded_image, uploaded_mask=uploaded_mask,
            )
        if workflow_controls is not None:
            from core.comfy_workflow_controls import apply_controls, WorkflowControlError
            if workflow is None:
                raise WorkflowCompileError("상세 설정을 적용할 사용자 워크플로가 없습니다.")
            try:
                graph = apply_controls(graph, workflow, self.object_info, workflow_controls)
            except WorkflowControlError as exc:
                raise WorkflowCompileError(str(exc)) from exc
        self.validate(graph, uploaded_inputs=(uploaded_image, uploaded_mask))
        return graph

    def _prepare_payload(
        self, model_name: str, payload: dict,
    ) -> tuple[list[LoraSpec], _Anima38Plan]:
        """Shared payload normalisation for generation and post-processing.

        Makes the seed concrete (one value for every pass), strips LoRA tags
        into loader specs and resolves the Anima plan.  ``payload`` is the
        compiler's private deep copy and is updated in place.
        """
        payload["seed"] = concrete_seed(payload.get("seed", -1))
        loras, prompts = parse_lora_tags(
            str(payload.get("prompt", "") or ""),
            str(payload.get("negative_prompt", "") or ""),
        )
        payload["prompt"], payload["negative_prompt"] = prompts
        anima_plan = self._resolve_anima38_plan(model_name, payload)
        if anima_plan.native_modules != tuple(self._module_names(payload)):
            payload["forge_additional_modules"] = list(anima_plan.native_modules)
        return loras, anima_plan

    def _custom_workflow_model(
        self, workflow: Mapping[str, Any], model_name: str,
        *, workflow_controls: Optional[Mapping[str, Any]] = None,
    ) -> tuple[str, str]:
        """``(model identity, model to write)`` for a custom workflow.

        When the sampler's model chain ends in a loader the app can rewrite
        (``MODEL_LOADER_INPUTS``), the selected ``model_name`` is both.  A
        chain ending in any other loader is locked to that loader: nothing is
        written, and the loader's own model file (when it has one) identifies
        the model family for LoRA/Anima decisions instead of the UI value.
        A locked loader's model is not app-managed, so a saved workflow
        control may replace it; the identity is then read from the workflow
        *with* those controls — the model the queued graph really loads.

        An empty selection (a ComfyUI whose models are all GGUF has an empty
        model combo, and ComfyUI allows that) means "the workflow's own
        model": a rewritable loader's own file is then both the identity and
        the model written (rewriting it to itself only normalises the choice,
        and the Anima v2 path needs the loader), and a locked loader goes
        through the same identity lookup as above.  Returning ``''`` here
        used to skip the family check, so an Anima workflow kept core
        ``LoraLoader`` (no block remap) and a v2 bundle lost its connector.
        """
        sampler_id = self._find_sampler(workflow)
        inputs = self._node_inputs(workflow.get(sampler_id))
        loader_id = self._trace_model_loader(workflow, inputs.get("model"))
        if loader_id:
            if model_name:
                return model_name, model_name
            loader = workflow.get(loader_id)
            own = self._node_inputs(loader).get(
                MODEL_LOADER_INPUTS[str(loader.get("class_type") or "")]
            )
            own_model = own.strip() if isinstance(own, str) else ""
            return own_model, own_model
        if workflow_controls is not None:
            workflow = self._workflow_with_controls(workflow, workflow_controls)
        from backends.comfyui_workflow_inspector import inspect_workflow

        try:
            inspection = inspect_workflow(dict(workflow))
        except Exception:  # pragma: no cover - defensive; inspector is pure
            return "", ""
        node_inputs = self._node_inputs(workflow.get(str(inspection.model_node_id or "")))
        for key in (inspection.model_param, "unet_name", "model_name", "ckpt_name", "gguf_name"):
            value = node_inputs.get(key) if key else None
            if isinstance(value, str) and value.strip():
                return value.strip(), ""
        return "", ""

    def _workflow_with_controls(
        self, workflow: Mapping[str, Any], workflow_controls: Mapping[str, Any],
    ) -> dict:
        """``workflow`` with its saved scalar controls applied (a new graph).

        The same validation ``compile`` runs after compilation, applied to the
        untouched workflow, so both see identical errors.
        """
        from core.comfy_workflow_controls import apply_controls, WorkflowControlError

        try:
            return apply_controls(workflow, workflow, self.object_info, workflow_controls)
        except WorkflowControlError as exc:
            raise WorkflowCompileError(str(exc)) from exc

    @staticmethod
    def _node_inputs(node: Any) -> Mapping[str, Any]:
        """A node's ``inputs`` for reading; ``{}`` for malformed nodes."""
        inputs = node.get("inputs") if isinstance(node, Mapping) else None
        return inputs if isinstance(inputs, Mapping) else {}

    @staticmethod
    def _sampler_inputs(node: dict) -> dict:
        """The sampler's writable ``inputs`` dict, or a clear compile error."""
        inputs = node.setdefault("inputs", {})
        if not isinstance(inputs, dict):
            raise WorkflowCompileError(
                f"sampler 노드({node.get('class_type')})의 inputs가 객체가 아닙니다."
            )
        return inputs

    def compile_upscale(
        self, uploaded_image: str, settings: Mapping[str, Any],
        *, source_width: Optional[int] = None, source_height: Optional[int] = None,
    ) -> dict:
        if not uploaded_image:
            raise WorkflowCompileError("업스케일 입력 이미지가 업로드되지 않았습니다.")
        graph = _Graph()
        load = graph.add("LoadImage", {"image": uploaded_image}, "Upscale source")
        method = str(settings.get("upscaler_name") or "Lanczos").strip()
        mode = str(settings.get("scale_mode") or "factor").strip().casefold()
        builtin = {
            "lanczos": "lanczos", "nearest": "nearest-exact", "nearest-exact": "nearest-exact",
            "bilinear": "bilinear", "bicubic": "bicubic", "area": "area",
        }.get(method.casefold())
        if builtin is not None:
            if mode == "size":
                image = graph.add("ImageScale", {
                    "image": [load, 0], "upscale_method": builtin,
                    "width": max(1, _int(settings.get("target_width"), 1024)),
                    "height": max(1, _int(settings.get("target_height"), 1024)),
                    "crop": "disabled",
                }, "Exact-size upscale")
            else:
                image = graph.add("ImageScaleBy", {
                    "image": [load, 0], "upscale_method": builtin,
                    "scale_by": max(0.01, _float(settings.get("scale_factor"), 2.0)),
                }, "Factor upscale")
        else:
            model_name = self._resolve_choice("UpscaleModelLoader", "model_name", method)
            loader = graph.add("UpscaleModelLoader", {"model_name": model_name}, "Upscale model")
            image = graph.add("ImageUpscaleWithModel", {
                "upscale_model": [loader, 0], "image": [load, 0],
            }, "Model upscale")
            if mode == "size":
                image = graph.add("ImageScale", {
                    "image": [image, 0], "upscale_method": "lanczos",
                    "width": max(1, _int(settings.get("target_width"), 1024)),
                    "height": max(1, _int(settings.get("target_height"), 1024)),
                    "crop": "disabled",
                }, "Final exact size")
            else:
                width, height = _int(source_width, 0), _int(source_height, 0)
                if width <= 0 or height <= 0:
                    raise WorkflowCompileError(
                        "모델 업스케일 배율을 적용하려면 입력 이미지의 원본 너비와 높이가 필요합니다."
                    )
                factor = max(0.01, _float(settings.get("scale_factor"), 2.0))
                image = graph.add("ImageScale", {
                    "image": [image, 0], "upscale_method": "lanczos",
                    "width": max(1, round(width * factor)),
                    "height": max(1, round(height * factor)),
                    "crop": "disabled",
                }, "Requested factor from original size")
        self._add_output_image(graph, [image, 0], settings, "AIStudio/upscale", "Save")
        self.validate(graph.nodes, uploaded_inputs=(uploaded_image,))
        return graph.nodes

    def compile_postprocess(
        self,
        model_name: str,
        payload: Mapping[str, Any],
        *,
        uploaded_image: str,
        sam3_detailer_class: str = "ForgeNeoSAM3Detailer",
    ) -> dict:
        """Compile ADetailer/SAM3 directly on an image without a no-op base sample."""
        if not uploaded_image:
            raise WorkflowCompileError("후처리 입력 이미지가 업로드되지 않았습니다.")
        if sam3_detailer_class not in {
            "ForgeNeoSAM3Detailer", "ForgeNeoSAM3Refine",
        }:
            raise WorkflowCompileError(
                f"지원하지 않는 SAM3 후처리 노드입니다: {sam3_detailer_class}"
            )
        local_payload = copy.deepcopy(dict(payload))
        if not self._has_image_scripts(local_payload):
            raise WorkflowCompileError("ADetailer 또는 SAM3 후처리 설정이 없습니다.")
        loras, anima_plan = self._prepare_payload(model_name, local_payload)
        graph = _Graph()
        (model, clip, vae, positive, negative,
         _sampler_options) = self._add_default_model_stack(
            graph, model_name, local_payload, loras, anima_plan,
        )
        source = graph.add("LoadImage", {"image": uploaded_image}, "Postprocess source")
        image = self._add_image_extensions(
            graph, [source, 0], model, clip, vae, positive, negative, local_payload,
            sam3_detailer_class=sam3_detailer_class,
        )
        self._add_output_image(
            graph, image, local_payload, "AIStudio/postprocess", "Save postprocessed image",
        )
        self.validate(graph.nodes, uploaded_inputs=(uploaded_image,))
        return graph.nodes

    def compile_sam3_mask_only(
        self, payload: Mapping[str, Any], *, uploaded_image: str,
    ) -> dict:
        """Compile SAM3 segmentation/artifact output without loading diffusion models."""
        if not uploaded_image:
            raise WorkflowCompileError("SAM3 입력 이미지가 업로드되지 않았습니다.")
        state = self._sam3_state(payload)
        if state is None:
            raise WorkflowCompileError("활성화된 SAM3 Mask 설정이 없습니다.")
        local_payload = copy.deepcopy(dict(payload))
        local_scripts = copy.deepcopy(dict(local_payload.get("alwayson_scripts") or {}))
        for name in list(local_scripts):
            if str(name).casefold() == "sam3 mask":
                local_scripts[name]["args"][0]["sam3_mode"] = "Mask only"
            else:
                del local_scripts[name]
        local_payload["alwayson_scripts"] = local_scripts
        graph = _Graph()
        source = graph.add("LoadImage", {"image": uploaded_image}, "SAM3 source")
        image = self._add_image_extensions(
            graph, [source, 0], [], [], [], [], [], local_payload,
        )
        self._add_output_image(
            graph, image, local_payload, "AIStudio/sam3-mask", "Save SAM3 mask",
        )
        self.validate(graph.nodes, uploaded_inputs=(uploaded_image,))
        return graph.nodes

    def validate(
        self, workflow: Mapping[str, Any], *, uploaded_inputs: Iterable[str] = (),
    ) -> None:
        """Validate every executable class when a /object_info document exists.

        ``uploaded_inputs`` are the input file names this job uploaded just
        before compiling.  The capability document may be a short-lived
        snapshot taken before the upload, so ``LoadImage``/``LoadImageMask``
        accept exactly those names even when their choice list lacks them;
        every other value is still checked against the published choices.
        """
        if self.object_info is None:
            return
        uploads = {str(name) for name in uploaded_inputs if str(name or "").strip()}
        used = {
            str(node.get("class_type") or "")
            for node in workflow.values()
            if isinstance(node, Mapping) and node.get("class_type")
        }
        missing = sorted(name for name in used if name not in self.object_info)
        if missing:
            forge_nodes = [name for name in missing if name.startswith("ForgeNeo")]
            hint = (
                " ComfyUI를 재시작하고 AI Studio Forge Parity custom node pack 연결을 확인하세요."
                if forge_nodes else " 대상 ComfyUI의 custom nodes/버전을 확인하세요."
            )
            raise WorkflowCompileError(
                "ComfyUI에 필요한 노드가 없습니다: " + ", ".join(missing) + "." + hint
            )
        if _PER_STEP_CNS_MARKER not in self.object_info and _graph_enables_cns(workflow):
            # 옛 팩(1.3.0)은 노드 계약이 같아 아래 검사를 통과하지만 CNS 를 초기 노이즈에만 걸고
            # KSamplerAdvanced 그래프에는 아예 걸지 않는다 — 조용히 다른 그림을 내지 않게 큐 전에 막는다.
            raise WorkflowCompileError(
                "CNS에는 스텝마다 노이즈를 색칠하는 AI Studio Forge Parity 노드 팩(1.4.0 이상)이 "
                f"필요한데 ComfyUI에 {_PER_STEP_CNS_MARKER}가 없습니다(옛 팩은 CNS를 초기 노이즈에만 "
                "적용합니다). 번들 노드 팩을 갱신하고 ComfyUI를 재시작하세요."
            )
        if _PAG_ORIGIN_MARKER not in self.object_info and _graph_uses_pag_legacy_or_heads(workflow):
            # 옛 팩(1.3.0)은 노드 계약이 같아 아래 검사를 통과하지만 PAG legacy·헤드 지정을 실행 중에
            # 예외로 거부한다 — 큐에 넣고 실패하지 않게 여기서 막는다.
            raise WorkflowCompileError(
                "PAG legacy·헤드 지정에는 원본 PAG 노드를 쓰는 AI Studio Forge Parity 노드 팩"
                f"(1.4.0 이상)이 필요한데 ComfyUI에 {_PAG_ORIGIN_MARKER}가 없습니다(옛 팩은 둘 다 "
                "실행 중에 거부합니다). 번들 노드 팩을 갱신하고 ComfyUI를 재시작하세요."
            )

        # A stale copy of the bundled pack can expose the class name while
        # still having an older input contract.  Catch that before /prompt so
        # enabled Forge features never disappear behind Comfy's node errors.
        for node_id, node in workflow.items():
            if not isinstance(node, Mapping):
                continue
            class_type = str(node.get("class_type") or "")
            schema = self.object_info.get(class_type, {})
            input_schema = schema.get("input", {}) if isinstance(schema, Mapping) else {}
            required = input_schema.get("required", {}) if isinstance(input_schema, Mapping) else {}
            optional = input_schema.get("optional", {}) if isinstance(input_schema, Mapping) else {}
            if not isinstance(required, Mapping) or not isinstance(optional, Mapping):
                continue
            known = set(required) | set(optional)
            if class_type.startswith("ForgeNeo") and not known:
                # Some lightweight capability probes publish class names only.
                continue
            supplied = node.get("inputs", {})
            supplied = supplied if isinstance(supplied, Mapping) else {}
            unexpected = sorted(set(supplied) - known) if class_type.startswith("ForgeNeo") else []
            absent = sorted(set(required) - set(supplied)) if class_type.startswith("ForgeNeo") else []
            invalid_choices: list[str] = []
            for name, value in supplied.items():
                if _is_link(value):
                    continue
                spec = required.get(name, optional.get(name))
                choices = (
                    list(spec[0])
                    if isinstance(spec, (list, tuple)) and spec
                    and isinstance(spec[0], (list, tuple))
                    else []
                )
                if (
                    class_type in _UPLOAD_INPUT_LOADERS and name == "image"
                    and value in uploads
                ):
                    continue
                if choices and value not in choices:
                    invalid_choices.append(f"{name}={value!r}")
            if unexpected or absent or invalid_choices:
                details = []
                if absent:
                    details.append("누락=" + ",".join(absent))
                if unexpected:
                    details.append("미지원=" + ",".join(unexpected))
                if invalid_choices:
                    details.append("허용되지 않은 선택=" + ",".join(invalid_choices))
                raise WorkflowCompileError(
                    f"ComfyUI {class_type} 노드 계약이 앱과 다릅니다 "
                    f"(node {node_id}; {'; '.join(details)}). "
                    "번들 노드 팩을 갱신하고 ComfyUI를 재시작하세요."
                )

    # ---- default graph -------------------------------------------------

    def _compile_default(
        self,
        mode: str,
        model_name: str,
        payload: dict,
        loras: Sequence[LoraSpec],
        anima_plan: _Anima38Plan,
        *,
        uploaded_image: str,
        uploaded_mask: str,
    ) -> dict:
        graph = _Graph()
        (model, clip, vae, positive_ref, negative_ref,
         sampler_options) = self._add_default_model_stack(
            graph, model_name, payload, loras, anima_plan,
        )
        latent = self._add_latent(
            graph, mode, vae, payload,
            uploaded_image=uploaded_image, uploaded_mask=uploaded_mask,
        )
        # Detail Daemon 은 샘플링 패스마다 _add_detail_daemon 이 고른다(base 또는 hires 한 곳) — model 자체에는
        # 없다. 디테일러가 받는 모델은 _add_image_extensions(last_pass_model=) 이 정한다.
        pass_model = self._add_detail_daemon(graph, model, payload)
        sampler = self._add_sampler(
            graph, pass_model,
            positive_ref, negative_ref, latent, payload, sampler_options,
            mode=mode,
        )
        samples, decode_vae = [sampler, 0], vae
        if _bool(payload.get("enable_hr")):
            pass_model = self._add_detail_daemon(graph, model, payload, hires_pass=True)
            samples, decode_vae = self._add_hires(
                graph, pass_model,
                clip, vae, positive_ref, negative_ref, samples, payload,
                sampler_options=sampler_options, anima_plan=anima_plan,
            )
        decode = graph.add("VAEDecode", {"samples": samples, "vae": decode_vae}, "Decode")
        image = [decode, 0]
        image = self._add_image_extensions(
            graph, image, model, clip, vae, positive_ref, negative_ref, payload,
            last_pass_model=pass_model,
        )
        self._add_output_image(
            graph, image, payload, "AIStudio/generated", "Save generated image",
        )
        return graph.nodes

    def _add_default_model_stack(
        self,
        graph: _Graph,
        model_name: str,
        payload: Mapping[str, Any],
        loras: Sequence[LoraSpec],
        anima_plan: _Anima38Plan,
    ) -> tuple[list, list, list, list, list, dict[str, Any]]:
        """Loaders → model patches → conditioning → guidance (app graphs).

        Shared by generation and standalone post-processing so both build the
        same model/conditioning stack in the same node order.
        """
        model, clip, vae = self._add_loaders(graph, model_name, payload, anima_plan)
        model, clip = self._add_model_patches(graph, model, clip, payload, loras, anima_plan)
        positive, negative = self._add_conditioning(
            graph, model, clip, payload, anima_plan,
        )
        model, sampler_options = self._add_anima_guidance(
            graph, model, clip, positive, negative, payload,
        )
        return model, clip, vae, positive, negative, sampler_options

    def _add_model_patches(
        self,
        graph: _Graph,
        model: list,
        clip: list,
        payload: Mapping[str, Any],
        loras: Sequence[LoraSpec],
        anima_plan: Optional[_Anima38Plan],
    ) -> tuple[list, list]:
        """Flow shift → LoRAs → NegPiP, in Forge's application order."""
        shift = _float(payload.get("distilled_cfg_scale"), 0.0)
        if shift > 0:
            shift_node = graph.add("ForgeNeoModelSamplingShift", {
                "model": model, "shift": shift,
            }, "Forge flow shift (preserve timestep scale)")
            model = [shift_node, 0]
        model, clip = self._add_loras(graph, model, clip, loras, anima_plan)
        return self._add_negpip(graph, model, clip, payload)

    @staticmethod
    def _add_output_image(
        graph: _Graph, image: list, payload: Mapping[str, Any],
        default_prefix: str, title: str,
    ) -> str:
        """Honor Forge's ``save_images``: only an explicit True keeps a copy.

        The main generation sends ``save_images=True`` (a ComfyUI/output copy,
        like Forge's outputs folder).  Post-processing, chat, hand repair and
        other callers send False or omit it (Forge's API default), so their
        result is returned through ComfyUI's temp ``PreviewImage`` instead of
        accumulating in ComfyUI/output.  Both are read back the same way.
        """
        if _bool(payload.get("save_images"), False):
            return graph.add("SaveImage", {
                "images": image,
                "filename_prefix": str(payload.get("filename_prefix") or default_prefix),
            }, title)
        return graph.add("PreviewImage", {"images": image}, title)

    def _add_loaders(
        self,
        graph: _Graph,
        model_name: str,
        payload: Mapping[str, Any],
        anima_plan: Optional[_Anima38Plan] = None,
    ):
        modules = (
            list(anima_plan.native_modules)
            if anima_plan is not None
            else self._module_names(payload)
        )
        if anima_plan is not None and anima_plan.v2:
            selected_model = self._resolve_choice(
                "ForgeNeoAnima38V2Loader", "model_name",
                anima_plan.model_name or model_name,
            )
            model_node = graph.add(
                "ForgeNeoAnima38V2Loader",
                {"model_name": selected_model},
                "Anima 3.8B Semantic Connector v2 bundle",
            )
            clip_modules, vae_modules, unknown = self._classify_modules(modules)
            if unknown:
                raise WorkflowCompileError(
                    "Anima 3.8B v2 additional modules 매핑 실패: "
                    + ", ".join(unknown)
                )
            if not clip_modules:
                explicit = payload.get("text_encoder_name") or payload.get("clip_name")
                if explicit:
                    clip_modules = [str(explicit)]
            if not vae_modules and payload.get("vae_name"):
                vae_modules = [str(payload["vae_name"])]
            if len(clip_modules) != 1:
                raise WorkflowCompileError(
                    "Anima 3.8B v2에는 native Qwen 0.6B text encoder 한 개가 "
                    f"필요합니다. 현재 매핑: {len(clip_modules)}개"
                )
            if len(vae_modules) != 1:
                raise WorkflowCompileError(
                    "Anima 3.8B v2에는 VAE 한 개가 필요합니다. "
                    f"현재 매핑: {len(vae_modules)}개"
                )
            clip = self._add_clip_loader(graph, clip_modules, payload)
            vae_name = self._resolve_choice(
                "VAELoader", "vae_name", vae_modules[0],
            )
            vae_node = graph.add("VAELoader", {"vae_name": vae_name}, "VAE")
            return [model_node, 0], clip, [vae_node, 0]

        checkpoint_choices = self._choices("CheckpointLoaderSimple", "ckpt_name")
        unet_choices = self._choices("UNETLoader", "unet_name")
        checkpoint = self._match_choice(model_name, checkpoint_choices)
        unet = self._match_choice(model_name, unet_choices)
        use_split = unet is not None and checkpoint is None
        if self.object_info is None and modules:
            # Offline compilation cannot disambiguate a checkpoint from a bare
            # diffusion model.  Forge additional modules conventionally means
            # the latter; a live backend always resolves from /object_info.
            use_split = True

        if not use_split:
            selected = self._resolve_choice("CheckpointLoaderSimple", "ckpt_name", model_name)
            node = graph.add("CheckpointLoaderSimple", {"ckpt_name": selected}, "Checkpoint")
            model, clip, vae = [node, 0], [node, 1], [node, 2]
            if not modules:
                return model, clip, vae
            clip_modules, vae_modules, unknown = self._classify_modules(modules)
            if unknown:
                raise WorkflowCompileError(
                    "forge_additional_modules를 ComfyUI 로더에 매핑할 수 없습니다: "
                    + ", ".join(unknown)
                )
            if clip_modules:
                clip = self._add_clip_loader(graph, clip_modules, payload)
            if len(vae_modules) > 1:
                raise WorkflowCompileError("forge_additional_modules의 VAE는 한 개만 지정할 수 있습니다.")
            if vae_modules:
                vae_name = self._resolve_choice("VAELoader", "vae_name", vae_modules[0])
                vae_node = graph.add("VAELoader", {"vae_name": vae_name}, "VAE override")
                vae = [vae_node, 0]
            return model, clip, vae

        selected_unet = self._resolve_choice("UNETLoader", "unet_name", model_name)
        unet_node = graph.add("UNETLoader", {
            "unet_name": selected_unet,
            "weight_dtype": str(payload.get("weight_dtype") or "default"),
        }, "Diffusion model")
        clip_modules, vae_modules, unknown = self._classify_modules(modules)
        if unknown:
            raise WorkflowCompileError(
                "forge_additional_modules를 ComfyUI 로더에 매핑할 수 없습니다: "
                + ", ".join(unknown)
            )
        if not clip_modules:
            explicit = payload.get("text_encoder_name") or payload.get("clip_name")
            if explicit:
                clip_modules = [str(explicit)]
        if not vae_modules and payload.get("vae_name"):
            vae_modules = [str(payload["vae_name"])]
        if not clip_modules:
            raise WorkflowCompileError(
                "분리 UNET에는 text encoder가 필요합니다. 설정의 TE 또는 "
                "forge_additional_modules를 지정하세요."
            )
        if len(vae_modules) != 1:
            raise WorkflowCompileError(
                "분리 UNET에는 VAE 한 개가 필요합니다. "
                f"현재 매핑된 VAE: {len(vae_modules)}개"
            )
        clip = self._add_clip_loader(graph, clip_modules, payload)
        vae_name = self._resolve_choice("VAELoader", "vae_name", vae_modules[0])
        vae_node = graph.add("VAELoader", {"vae_name": vae_name}, "VAE")
        return [unet_node, 0], clip, [vae_node, 0]

    def _add_clip_loader(self, graph: _Graph, names: Sequence[str], payload: Mapping[str, Any]):
        resolved = [self._resolve_choice("CLIPLoader", "clip_name", item) for item in names]
        if len(resolved) == 1:
            node = graph.add("CLIPLoader", {
                "clip_name": resolved[0],
                "type": str(payload.get("comfy_clip_type") or payload.get("clip_type") or "stable_diffusion"),
                "device": str(payload.get("clip_device") or "default"),
            }, "Text encoder")
        elif len(resolved) == 2:
            node = graph.add("DualCLIPLoader", {
                "clip_name1": resolved[0], "clip_name2": resolved[1],
                "type": str(payload.get("comfy_dual_clip_type") or "sd3"),
                "device": str(payload.get("clip_device") or "default"),
            }, "Dual text encoder")
        elif len(resolved) == 3:
            node = graph.add("TripleCLIPLoader", {
                "clip_name1": resolved[0], "clip_name2": resolved[1], "clip_name3": resolved[2],
            }, "Triple text encoder")
        else:
            raise WorkflowCompileError(
                f"ComfyUI 기본 그래프는 text encoder 1~3개를 지원합니다: {len(resolved)}개 요청됨"
            )
        return [node, 0]

    @staticmethod
    def _looks_like_qwen35(value: Any) -> bool:
        name = _filename(value).casefold()
        return any(marker in name for marker in (
            "qwen35_4b", "qwen3.5-4b", "qwen3_5_4b",
        ))

    @staticmethod
    def _looks_like_anima_adapter(value: Any) -> bool:
        name = _filename(value).casefold()
        return "anima" in name and any(
            marker in name for marker in ("adapter", "connector")
        )

    @staticmethod
    def _looks_like_anima_model(value: Any) -> bool:
        return "anima" in _filename(value).casefold()

    @staticmethod
    def _looks_like_anima38_v2_bundle(value: Any) -> bool:
        """Recognise bundle release names without treating every Anima UNET as v2."""

        name = _filename(value).casefold()
        family = "anima" in name and any(
            marker in name for marker in ("3.8b", "3-8b", "3_8b")
        )
        release = any(
            marker in name for marker in (
                "-v2", "_v2", ".v2", "-v1.1", "_v1.1", ".v1.1",
            )
        )
        return family and release

    @staticmethod
    def _preferred_qwen35_choice(choices: Sequence[str]) -> str:
        for preferred in (
            "qwen35_4b.safetensors",
            "qwen3.5-4b.safetensors",
            "qwen3_5_4b.safetensors",
        ):
            for choice in choices:
                if _filename(choice).casefold() == preferred:
                    return str(choice)
        return str(choices[0]) if choices else ""

    def _resolve_anima38_plan(
        self, model_name: str, payload: Mapping[str, Any],
    ) -> _Anima38Plan:
        """Resolve ANIMA resources once, before any graph nodes are emitted.

        The dedicated v2 loader publishes only metadata-verified bundle
        choices.  Qwen3.5 and legacy connector files can also appear in
        ``CLIPLoader``'s broad text-encoder list, so they are classified first
        and never leak into Dual/TripleCLIPLoader.
        """

        block = self._script(payload, anima38.SCRIPT_NAME)
        settings = anima38.parse_script_block(block)
        modules = self._module_names(payload)

        v2_choices = self._choices("ForgeNeoAnima38V2Loader", "model_name")
        v2_model = (
            self._match_choice(model_name, v2_choices)
            if v2_choices
            else None
        )
        if (
            v2_model is None
            and v2_choices is not None
            and self._looks_like_anima38_v2_bundle(model_name)
        ):
            raise WorkflowCompileError(
                "Anima 3.8B v2 bundle은 metadata 검증 전용 loader로만 열 수 "
                "있지만 ForgeNeoAnima38V2Loader.model_name 선택 목록에서 찾지 "
                f"못했습니다: {model_name}"
            )
        qwen_choices = self._choices(
            "ForgeNeoAnimaQwen35Loader", "qwen35_model",
        )
        adapter_choices = self._choices(
            "ForgeNeoAnimaQwen35Prompt", "adapter_name",
        )

        qwen_modules: list[str] = []
        adapter_modules: list[str] = []
        ordinary_modules: list[str] = []
        for item in modules:
            qwen_match = (
                self._match_choice(item, qwen_choices) if qwen_choices else None
            )
            adapter_match = (
                self._match_choice(item, adapter_choices)
                if adapter_choices else None
            )
            if qwen_match is not None or self._looks_like_qwen35(item):
                qwen_modules.append(qwen_match or item)
            elif adapter_match is not None or self._looks_like_anima_adapter(item):
                adapter_modules.append(adapter_match or item)
            else:
                ordinary_modules.append(item)

        def one_resource(values: Sequence[str], label: str) -> str:
            unique: list[str] = []
            seen: set[str] = set()
            for value in values:
                key = str(value).strip().replace("\\", "/").casefold()
                if key not in seen:
                    unique.append(str(value))
                    seen.add(key)
            if len(unique) > 1:
                raise WorkflowCompileError(
                    f"Anima 3.8B {label} 리소스가 여러 개라 선택할 수 없습니다: "
                    + ", ".join(unique)
                )
            return unique[0] if unique else ""

        detected_qwen = one_resource(qwen_modules, "Qwen3.5")
        detected_adapter = one_resource(adapter_modules, "adapter")
        legacy_candidate = bool(detected_qwen and detected_adapter)

        loader_kind = "v2" if v2_model is not None else "standard"
        if settings.bypass:
            conditioning_kind = "native"
        elif v2_model is not None:
            conditioning_kind = "v2"
        elif settings.enabled or legacy_candidate:
            conditioning_kind = "v1"
        else:
            conditioning_kind = "native"

        # Qwen3.5 and connector adapters are never native CLIP inputs.  Strip
        # them even when the script is disabled so a broad CLIPLoader choice
        # list cannot accidentally turn an incomplete semantic setup into a
        # Dual/TripleCLIPLoader graph.
        native_modules = ordinary_modules
        native_clips, vaes, unknown = self._classify_modules(native_modules)

        qwen35_model = ""
        adapter_name = ""
        if conditioning_kind in {"v1", "v2"}:
            requested_qwen = detected_qwen
            if qwen_choices:
                requested_qwen = requested_qwen or self._preferred_qwen35_choice(
                    qwen_choices,
                )
                qwen35_model = self._resolve_choice(
                    "ForgeNeoAnimaQwen35Loader", "qwen35_model", requested_qwen,
                )
            elif qwen_choices is None:
                qwen35_model = requested_qwen or "qwen35_4b.safetensors"
            else:
                raise WorkflowCompileError(
                    "Anima 3.8B Semantic Connector에 필요한 Qwen3.5 4B "
                    "text encoder를 ComfyUI에서 찾을 수 없습니다."
                )

        if conditioning_kind == "v1":
            requested_adapter = (
                settings.adapter if settings.enabled else detected_adapter
            ) or settings.adapter
            if adapter_choices:
                adapter_name = self._resolve_choice(
                    "ForgeNeoAnimaQwen35Prompt", "adapter_name", requested_adapter,
                )
            elif adapter_choices is None:
                adapter_name = requested_adapter
            else:
                raise WorkflowCompileError(
                    "Anima 3.8B legacy Semantic Connector adapter를 "
                    "ComfyUI에서 찾을 수 없습니다."
                )

        anima_markers = [model_name, *native_modules, *qwen_modules, *adapter_modules]
        is_anima = bool(v2_model is not None or conditioning_kind == "v1") or any(
            self._looks_like_anima_model(item) for item in anima_markers
        )
        return _Anima38Plan(
            loader_kind=loader_kind,
            conditioning_kind=conditioning_kind,
            model_name=str(v2_model or model_name),
            native_modules=tuple(native_modules),
            native_clip_modules=tuple(native_clips),
            vae_modules=tuple(vaes),
            unknown_modules=tuple(unknown),
            qwen35_model=qwen35_model,
            adapter_name=adapter_name,
            settings=settings,
            is_anima=is_anima,
        )

    def _add_latent(
        self, graph: _Graph, mode: str, vae: list, payload: Mapping[str, Any],
        *, uploaded_image: str, uploaded_mask: str,
    ) -> list:
        if mode != "txt2img" and not uploaded_image:
            raise WorkflowCompileError(f"{mode} 입력 이미지가 업로드되지 않았습니다.")
        image_node = graph.add("LoadImage", {"image": uploaded_image}, "Input image") if uploaded_image else None
        mask_node = graph.add("LoadImage", {"image": uploaded_mask}, "Input mask") if uploaded_mask else None
        if mode == "inpaint" and not (uploaded_mask or payload.get("use_image_alpha_as_mask")):
            raise WorkflowCompileError("inpaint에는 마스크 이미지 또는 알파 마스크가 필요합니다.")
        batch = max(1, _int(payload.get("batch_size"), 1)) * max(
            1, _int(payload.get("n_iter", payload.get("batch_count", 1)), 1)
        )
        inputs: dict[str, Any] = {
            "vae": vae,
            "mode": mode,
            "width": max(64, _int(payload.get("width"), 512)),
            "height": max(64, _int(payload.get("height"), 512)),
            "batch_size": batch,
            "fit": self._fit_mode(payload.get("resize_mode")),
            "mask_invert": _bool(payload.get("inpainting_mask_invert", payload.get("mask_invert"))),
            "mask_blur": max(0, _int(payload.get("mask_blur"), 4)),
            "grow_mask_by": max(0, _int(payload.get("mask_dilation", payload.get("grow_mask_by", 6)), 6)),
            "reference_enabled": False,
            "reference_layout": "reference_left",
            "reference_fraction": 0.5,
            "reference_gap": 0,
            "reference_background": 0.0,
            "reference_fit": "contain",
        }
        if mode == "img2img":
            inputs["img2img_image"] = [image_node, 0]
        elif mode == "inpaint":
            inputs["inpaint_image"] = [image_node, 0]
            if mask_node:
                inputs["inpaint_mask_image"] = [mask_node, 0]
            else:
                inputs["inpaint_mask"] = [image_node, 1]
        node = graph.add("ForgeNeoLatentInput", inputs, f"{mode} latent")
        return [node, 0]

    def _add_sampler(
        self, graph: _Graph, model: list, positive: list, negative: list, latent: list,
        payload: Mapping[str, Any], options: Mapping[str, Any], *, mode: str,
    ) -> str:
        # compile() already made payload["seed"] concrete; this only guards
        # direct callers so no node ever receives -1.
        seed = concrete_seed(payload.get("seed"))
        sampler_name, scheduler = self._runtime_sampler_values(
            payload.get("sampler_name") or "euler",
            payload.get("scheduler") or "normal",
        )
        inputs = {
            "model": model, "positive": positive, "negative": negative,
            "latent_image": latent, "seed": seed,
            "steps": max(1, _int(payload.get("steps"), 20)),
            "cfg": _float(payload.get("cfg_scale"), 7.0),
            "sampler_name": sampler_name,
            "scheduler": scheduler,
            "denoise": (
                1.0 if mode == "txt2img" else
                max(0.0, min(1.0, _float(payload.get("denoising_strength"), 0.75)))
            ),
            **_cns_inputs(options),
            "spectrum_enabled": _bool(payload.get("spectrum_enabled")),
            "spectrum_window_size": _float(payload.get("spectrum_window_size"), 2.0),
            "spectrum_flex_window": _float(payload.get("spectrum_flex_window"), 0.25),
            "spectrum_warmup_steps": _int(payload.get("spectrum_warmup_steps"), 6),
            "spectrum_tail_actual_steps": _int(payload.get("spectrum_tail_actual_steps"), 3),
            "spectrum_blend_w": _float(payload.get("spectrum_blend_w"), 0.3),
            "spectrum_cheby_degree": _int(payload.get("spectrum_cheby_degree"), 3),
            "spectrum_ridge_lambda": _float(payload.get("spectrum_ridge_lambda"), 0.1),
            "spectrum_history_size": _int(payload.get("spectrum_history_size"), 100),
            "spectrum_one_sampler_only": _bool(payload.get("spectrum_one_sampler_only")),
            "spectrum_verbose": _bool(payload.get("spectrum_verbose")),
            "speed_enabled": _bool(payload.get("speed_enabled")),
            "speed_split_mode": str(payload.get("speed_split_mode") or "single"),
            "speed_spd_scale": _float(payload.get("speed_spd_scale"), 0.5),
            "speed_spd_sigma": _float(payload.get("speed_spd_sigma"), 0.7),
            "speed_adaptive_smc_alpha": _float(payload.get("speed_adaptive_smc_alpha"), 0.0),
        }
        return graph.add("ForgeNeoKSamplerCNS", inputs, "Forge-compatible sampler")

    # ---- graph stages --------------------------------------------------

    def _add_loras(
        self,
        graph: _Graph,
        model: list,
        clip: list,
        loras: Sequence[LoraSpec],
        anima_plan: Optional[_Anima38Plan] = None,
    ):
        loader_type = (
            "ForgeNeoAnimaLoraLoader"
            if anima_plan is not None and anima_plan.is_anima
            else "LoraLoader"
        )
        for spec in loras:
            name = self._resolve_choice(loader_type, "lora_name", spec.name)
            node = graph.add(loader_type, {
                "model": model, "clip": clip, "lora_name": name,
                "strength_model": spec.strength_model, "strength_clip": spec.strength_clip,
            }, f"LoRA: {name}")
            model, clip = [node, 0], [node, 1]
        return model, clip

    def _add_conditioning(
        self,
        graph: _Graph,
        model: list,
        clip: list,
        payload: Mapping[str, Any],
        anima_plan: _Anima38Plan,
    ) -> tuple[list, list]:
        positive_text = str(payload.get("prompt", "") or "")
        negative_text = str(payload.get("negative_prompt", "") or "")
        if not anima_plan.semantic:
            positive = graph.add(
                "CLIPTextEncode", {"clip": clip, "text": positive_text}, "Positive",
            )
            negative = graph.add(
                "CLIPTextEncode", {"clip": clip, "text": negative_text}, "Negative",
            )
            return [positive, 0], [negative, 0]

        qwen_node = graph.add(
            "ForgeNeoAnimaQwen35Loader",
            {"qwen35_model": anima_plan.qwen35_model},
            "Anima Qwen3.5 semantic encoder",
        )
        qwen_clip = [qwen_node, 0]
        if anima_plan.conditioning_kind == "v2":
            prompt_type = "ForgeNeoAnima38V2Prompt"

            def prompt_inputs(text: str, _strength: float) -> dict[str, Any]:
                return {
                    "model": model,
                    "native_clip": clip,
                    "qwen35_clip": qwen_clip,
                    "prompt": text,
                }
        else:
            prompt_type = "ForgeNeoAnimaQwen35Prompt"

            def prompt_inputs(text: str, strength: float) -> dict[str, Any]:
                return {
                    "model": model,
                    "native_clip": clip,
                    "qwen35_clip": qwen_clip,
                    "adapter_name": anima_plan.adapter_name,
                    "prompt": text,
                    "adapter_strength": strength,
                }

        positive = graph.add(
            prompt_type,
            prompt_inputs(positive_text, anima_plan.settings.strength),
            "Anima semantic positive",
        )
        if anima_plan.settings.negative:
            negative = graph.add(
                prompt_type,
                prompt_inputs(negative_text, anima_plan.settings.negative_strength),
                "Anima semantic negative",
            )
            negative_ref = [negative, 0]
        else:
            negative = graph.add(
                "CLIPTextEncode", {"clip": clip, "text": negative_text},
                "Anima native negative",
            )
            negative_ref = [negative, 0]
        return [positive, 0], negative_ref

    def _add_negpip(self, graph: _Graph, model: list, clip: list, payload: Mapping[str, Any]):
        block = self._script(payload, "NegPiP")
        if block is None:
            return model, clip
        args = block.get("args", []) if isinstance(block, Mapping) else []
        enabled = _bool(args[0] if args else True, True)
        if not enabled:
            return model, clip
        node = graph.add("ForgeNeoNegPip", {
            "model": model, "clip": clip, "enabled": True,
        }, "NegPiP")
        return [node, 0], [node, 1]

    def _add_anima_guidance(
        self, graph: _Graph, model: list, clip: list, positive: list, negative: list,
        payload: Mapping[str, Any],
    ) -> tuple[list, dict[str, Any]]:
        from core import anima_guidance

        sampler_options: dict[str, Any] = {}
        perturb = self._script(payload, anima_guidance.SCRIPT_PERTURBATION)
        if perturb is not None:
            settings = self._script_settings(perturb, anima_guidance.PERTURBATION_SPEC)
            active_keys = (
                "guid_enabled", "guid_slg_on", "guid_apg_enabled", "guid_adg_enabled",
                "guid_smc_enabled", "guid_smc_master_enabled", "guid_cwm_enabled",
                "guid_dcw_enabled", "guid_rdc_enabled", "guid_dave_enabled",
                "guid_cns_enabled", "guid_mod_enabled", "guid_experimental_stack",
            )
            if any(_bool(settings.get(key)) for key in active_keys):
                self._validate_anima_guidance_settings(settings)
                suite_settings = dict(settings)
                suite_settings["cfg_scale"] = _float(payload.get("cfg_scale"), 7.0)
                node = graph.add("ForgeNeoAnimaGuidanceSuite", {
                    "model": model, "clip": clip, "positive": positive, "negative": negative,
                    "enabled": True,
                    "settings_json": json.dumps(
                        suite_settings, ensure_ascii=False, sort_keys=True, separators=(",", ":")
                    ),
                }, "Anima guidance suite")
                model = [node, 0]
            sampler_options.update(_cns_inputs({
                key: settings.get(f"guid_{key}")
                for key in ("cns_enabled", *_CNS_DEFAULTS)
            }))

        skim = self._script(payload, anima_guidance.SCRIPT_SKIMMED_CFG)
        if skim is not None:
            settings = self._script_settings(skim, anima_guidance.SKIMMED_SPEC)
            if _bool(settings.get("skim_enabled")):
                node = graph.add("ForgeNeoSkimmedCFG", {
                    "model": model, "enabled": True,
                    "skimming_cfg": _float(settings.get("skim_skimming_cfg"), 7.0),
                    "full_skim_negative": _bool(settings.get("skim_full_skim_negative")),
                    "disable_flipping_filter": _bool(settings.get("skim_disable_flipping_filter")),
                    "start_percent": _float(settings.get("skim_start_percent"), 0.0),
                    "end_percent": _float(settings.get("skim_end_percent"), 1.0),
                    "flip_percent": _float(settings.get("skim_flip_at"), 0.0),
                }, "Anima skimmed CFG")
                model = [node, 0]
        # Detail Daemon 은 여기서 걸지 않는다 — 패스마다 _add_detail_daemon 이 고른다(base 또는 hires 한 곳만).
        return model, sampler_options

    # settings_json 에 넣는 Detail Daemon 키 = dd_enabled + 팩 노드가 원본 노드 입력으로 읽는 키
    # (comfy_custom_nodes/ai_studio_forge_parity/guidance_dd.SETTING_KEYS — tests/test_comfy_detail_daemon.py 가 맞춘다).
    # preset·multiplier·cfg_couple(원본에 없는 숨은 자리)과 dd_hires(패스 선택, 아래)는 보내지 않는다.
    _DD_NODE_KEYS = (
        "dd_enabled", "dd_amount", "dd_start", "dd_end", "dd_bias", "dd_exponent",
        "dd_start_offset", "dd_end_offset", "dd_fade", "dd_smooth",
    )

    def _add_detail_daemon(
        self, graph: _Graph, model: list, payload: Mapping[str, Any], *, hires_pass: bool = False,
        cfg_scale: Any = None, sampler_name: Any = None, title: str = "",
    ) -> list:
        """한 샘플링 패스가 쓸 모델 — Detail Daemon 은 그 패스가 맞을 때만 건다.

        Hires Pass(dd_hires, 인자 13)는 muerrilla 와 같다: 끄면(기본) base 패스만, 켜면 hires 패스만
        (origin: muerrilla/sd-webui-detail-daemon@19479998:scripts/detail_daemon.py:104, :276).
        값은 원본 노드(Jonseed/ComfyUI-Detail-Daemon) 단위 그대로 넘기고 ×0.1×cfg 는 노드가 한다.
        cfg 는 base cfg_scale 이다(muerrilla 는 hr_cfg 가 아니라 p.cfg_scale: detail_daemon.py:259, :304) —
        노드의 cfg_scale_override 로 준다.

        디테일러(``_add_image_extensions`` 의 ``last_pass_model``):
        - ADetailer 는 마지막 본 패스(hires 를 돌렸으면 hires, 아니면 base)의 모델을 그대로 받는다. muerrilla 의
          on_cfg_denoiser 콜백은 process() 에서 걸려 postprocess() 에서야 풀리는데(detail_daemon.py:256-258,
          :269-272) Forge 는 postprocess_image(ADetailer) 를 그보다 먼저 부르고(modules/processing.py:1068 <
          :1180), ADetailer 의 i2i 는 고른 스크립트만 돌려 DD 를 다시 판정하지 않는다(aadetailer script_filter,
          기본 ad_script_names 에 DD 없음). 그래서 dd_hires == (hires 를 돌렸는가) 일 때 base cfg 로 걸린다
          (detail_daemon.py:276). 확장도 같다(_DD['on'] 이 마지막 본 패스 값으로 남는다).
        - SAM3 디테일러 패스는 이 함수를 ``hires_pass=False`` 와 그 패스의 cfg·샘플러로 부른다. 확장의 SAM3
          인페인트(sam3ext/inpaint_core.py build_i2i → script_filter)는 SAM3 만 빼고 alwayson 스크립트를 다시 돌려
          anima_detail_daemon.process_before_every_sampling 이 p2(img2img, is_hr_pass False·p2.cfg_scale·
          p2.sampler_name)로 다시 판정하기 때문이다 — dd_hires 가 꺼져 있으면 hires 여부와 상관없이 걸린다.
        - 단독 후처리(compile_postprocess)와 mask-only 는 DD 가 없다: Forge 의 단독 요청(webui_backend
          _build_postprocess_payload)은 ADetailer/SAM3 인자만 보내 DD 가 UI 기본값(끔)으로 돈다.
        스케줄은 원본 노드처럼 그 디테일러 샘플러의 σ 목록으로 새로 만든다(확장 _node_lookup 과 같음). muerrilla 는
        ADetailer 패스에서도 본 패스에서 만든 스케줄을 디테일러의 호출 카운터로 읽는다 — 호스트 차이(F·C 공통).
        """
        from core import anima_guidance

        daemon = self._script(payload, anima_guidance.SCRIPT_DETAIL_DAEMON)
        if daemon is None:
            return model
        settings = self._script_settings(daemon, anima_guidance.DETAIL_DAEMON_SPEC)
        if not _bool(settings.get("dd_enabled")) or _bool(settings.get("dd_hires")) != hires_pass:
            return model
        sampler = payload.get("sampler_name") if sampler_name is None else sampler_name
        if self._comfy_sampler(sampler).casefold() in {"dpm_adaptive", "heunpp2"}:
            # muerrilla: "Selected sampler (DPM adaptive/HeunPP2) is not supported" — 생성 전체에서 끄고, 판정은
            # base 샘플러(p.sampler_name)로 한다(detail_daemon.py:197-199). 확장도 같다(SAM3 패스는 p2 의 샘플러).
            # (원본 Comfy 노드는 이 제외가 없다 — 손으로 만든 워크플로의 팩 노드는 그대로 건다.)
            return model
        cfg_scale = _float(payload.get("cfg_scale") if cfg_scale is None else cfg_scale, 7.0)
        if cfg_scale <= 0.0:
            # muerrilla 의 σ *= 1 − s·0.1·cfg 는 cfg 0 에서 정확히 아무것도 안 한다. 노드의 override 0 은
            # '샘플러의 CFG' 라는 뜻이라(hires 는 hr_cfg) 노드를 넣지 않는다.
            return model
        daemon_settings = {key: settings[key] for key in self._DD_NODE_KEYS if key in settings}
        node = graph.add("ForgeNeoAnimaDetailDaemon", {
            "model": model, "enabled": True,
            "settings_json": json.dumps(
                daemon_settings, ensure_ascii=False, sort_keys=True, separators=(",", ":")
            ),
            # 원본 노드 입력. 옛 팩(1.3.0)의 /object_info 에는 없어 validate() 가 큐 전에 '노드 계약' 오류로 막는다.
            "cfg_scale_override": cfg_scale,
        }, title or ("Anima detail daemon (hires pass)" if hires_pass else "Anima detail daemon"))
        return [node, 0]

    @staticmethod
    def _detail_daemon_scripts(payload: Mapping[str, Any]) -> dict[str, Any]:
        """payload 의 Detail Daemon 스크립트 블록(없으면 빈 dict) — 순차 SAM3 패스가 DD 를 다시 판정하게 넘긴다."""
        from core import anima_guidance

        scripts = payload.get("alwayson_scripts", {})
        if not isinstance(scripts, Mapping):
            return {}
        folded = anima_guidance.SCRIPT_DETAIL_DAEMON.casefold()
        return {name: block for name, block in scripts.items() if str(name).casefold() == folded}

    def _add_hires(
        self, graph: _Graph, model: list, clip: list, vae: list,
        positive: list, negative: list, samples: list, payload: Mapping[str, Any],
        *, sampler_options: Optional[Mapping[str, Any]] = None,
        anima_plan: Optional[_Anima38Plan] = None,
    ) -> tuple[list, list]:
        if anima_plan is not None and anima_plan.semantic:
            alternate_checkpoint = str(
                payload.get("hr_checkpoint_name") or ""
            ).strip()
            alternate_modules = payload.get("hr_additional_modules", [])
            if isinstance(alternate_modules, str):
                alternate_modules = [
                    item.strip() for item in alternate_modules.split(",")
                    if item.strip()
                ]
            module_override = bool(
                isinstance(alternate_modules, Sequence)
                and not isinstance(alternate_modules, (str, bytes, bytearray))
                and any(
                    str(item).strip()
                    and str(item).strip().casefold() != "use same choices"
                    for item in alternate_modules
                )
            )
            prompt_override = bool(
                str(payload.get("hr_prompt") or "").strip()
                or str(payload.get("hr_negative_prompt") or "").strip()
            )
            if alternate_checkpoint or module_override or prompt_override:
                raise WorkflowCompileError(
                    "Anima Semantic Connector에서 Hires의 다른 checkpoint/TE/prompt "
                    "override는 아직 안전하게 표현할 수 없습니다. Hires는 같은 "
                    "model과 conditioning을 사용하세요."
                )
        sampler_options = sampler_options or {}
        upscaler = str(payload.get("hr_upscaler") or "latent:bislerp")
        upscaler_map = {
            "none": "latent:bislerp", "latent": "latent:bislerp",
            "latent (nearest)": "latent:nearest-exact",
            "latent (nearest-exact)": "latent:nearest-exact",
            "latent (bilinear)": "latent:bilinear", "latent (bicubic)": "latent:bicubic",
            "latent (bislerp)": "latent:bislerp",
        }
        method = upscaler_map.get(upscaler.casefold(), upscaler)
        method = self._resolve_when_enumerated(
            "ForgeNeoHiresFix", "upscale_method", method,
        )
        hires_sampler, hires_scheduler = self._runtime_sampler_values(
            payload.get("hr_sampler_name") or payload.get("sampler_name") or "euler",
            payload.get("hr_scheduler") or payload.get("scheduler") or "normal",
        )
        base_sampler, base_scheduler = self._runtime_sampler_values(
            payload.get("sampler_name") or "euler",
            payload.get("scheduler") or "normal",
        )
        inputs = {
            "model": model, "positive": positive, "negative": negative, "samples": samples,
            "seed": _int(payload.get("seed"), -1), "enabled": True,
            "scale_by": max(1.0, _float(payload.get("hr_scale"), 2.0)),
            "upscale_method": method,
            "steps": max(0, _int(payload.get("hr_second_pass_steps"), 0)),
            "cfg": _float(payload.get("hr_cfg"), _float(payload.get("cfg_scale"), 7.0)),
            "sampler_name": hires_sampler,
            "scheduler": hires_scheduler,
            "denoise": max(0.0, min(1.0, _float(payload.get("denoising_strength"), 0.5))),
            "seed_delta": _int(payload.get("hr_seed_delta"), 0),
            **_cns_inputs(sampler_options),
            "base_vae": vae, "base_clip": clip,
            "base_steps": max(1, _int(payload.get("steps"), 20)),
            "base_sampler_name": base_sampler,
            "base_scheduler": base_scheduler,
            "resize_width": max(0, _int(payload.get("hr_resize_x"), 0)),
            "resize_height": max(0, _int(payload.get("hr_resize_y"), 0)),
            "shift": max(0.0, _float(payload.get("distilled_cfg_scale"), 0.0)),
            "checkpoint_name": self._resolve_when_enumerated(
                "ForgeNeoHiresFix", "checkpoint_name",
                payload.get("hr_checkpoint_name") or "Use same checkpoint",
            ),
            "checkpoint_weight_dtype": str(payload.get("hr_weight_dtype") or "default"),
            "text_encoder_name": self._hr_module(payload, "clip"),
            "vae_name": self._hr_module(payload, "vae"),
            "positive_text": str(payload.get("hr_prompt") or ""),
            "negative_text": str(payload.get("hr_negative_prompt") or ""),
            "base_positive_text": str(payload.get("prompt") or ""),
            "base_negative_text": str(payload.get("negative_prompt") or ""),
            "clip_type": str(
                payload.get("comfy_clip_type") or payload.get("clip_type")
                or "stable_diffusion"
            ),
        }
        node = graph.add("ForgeNeoHiresFix", inputs, "Forge-compatible Hires.fix")
        return [node, 0], [node, 1]

    def _add_image_extensions(
        self, graph: _Graph, image: list, model: list, clip: list, vae: list,
        positive: list, negative: list, payload: Mapping[str, Any],
        *, sam3_detailer_class: str = "ForgeNeoSAM3Detailer",
        last_pass_model: Optional[list] = None,
    ) -> list:
        """ADetailer → SAM3 → 순차 SAM3 패스.

        ``last_pass_model``: 생성 안에서만 준다 — 마지막 본 샘플링 패스(base, 또는 돌렸으면 hires)가 쓴 모델.
        ADetailer 는 그것을 받고, SAM3 디테일러는 그 패스의 cfg·샘플러로 Detail Daemon 을 다시 판정한다
        (규칙과 근거는 ``_add_detail_daemon``). None(단독 후처리, mask-only)이면 디테일러는 ``model`` 을 받는다.
        """
        adetailer_model = model if last_pass_model is None else last_pass_model
        slots = self._adetailer_slots(payload)
        for index, slot in enumerate(slots, start=1):
            self._validate_adetailer_slot(slot, index)
            normalized_slot = dict(slot)
            requested_ad_sampler = (
                str(slot.get("ad_sampler") or "euler")
                if _bool(slot.get("ad_use_sampler"))
                else str(payload.get("sampler_name") or "euler")
            )
            requested_ad_scheduler = (
                str(slot.get("ad_scheduler") or "normal")
                if _bool(slot.get("ad_use_sampler"))
                else str(payload.get("scheduler") or "normal")
            )
            ad_sampler, ad_scheduler = self._runtime_sampler_values(
                requested_ad_sampler, requested_ad_scheduler,
            )
            # Forge ADetailer has no seed of its own: every slot uses the
            # image's seed (compile() made payload["seed"] concrete).
            ad_seed = _int(slot.get("ad_seed"), -1)
            normalized_slot.update({
                # Impact detector choice ("bbox/…" or "segm/…"), resolved
                # before /prompt so a missing model fails before sampling.
                "model_name": self._resolve_adetailer_model(slot.get("ad_model"), index),
                "seed": ad_seed if ad_seed >= 0 else concrete_seed(payload.get("seed")),
                "steps": (
                    _int(slot.get("ad_steps"), 28)
                    if _bool(slot.get("ad_use_steps"))
                    else _int(payload.get("steps"), 28)
                ),
                "cfg": (
                    _float(slot.get("ad_cfg_scale"), 7.0)
                    if _bool(slot.get("ad_use_cfg_scale"))
                    else _float(payload.get("cfg_scale"), 7.0)
                ),
                "sampler_name": ad_sampler,
                "scheduler": ad_scheduler,
                "denoise": _float(slot.get("ad_denoising_strength"), 0.4),
                "guide_size": (
                    max(
                        _int(slot.get("ad_inpaint_width"), 512),
                        _int(slot.get("ad_inpaint_height"), 512),
                    )
                    if _bool(slot.get("ad_use_inpaint_width_height")) else 512
                ),
                "bbox_threshold": _float(slot.get("ad_confidence"), 0.3),
                "bbox_dilation": _int(slot.get("ad_dilate_erode"), 4),
                "prompt": str(slot.get("ad_prompt") or ""),
            })
            slot_negative = negative
            if str(slot.get("ad_negative_prompt") or "").strip():
                negative_node = graph.add("CLIPTextEncode", {
                    "clip": clip, "text": str(slot.get("ad_negative_prompt")),
                }, f"ADetailer slot {index} negative")
                slot_negative = [negative_node, 0]
            node = graph.add("ForgeNeoADetailer", {
                "image": image, "model": adetailer_model, "clip": clip, "vae": vae,
                "positive": positive, "negative": slot_negative, "enabled": True,
                "settings_json": json.dumps(
                    normalized_slot, ensure_ascii=False, sort_keys=True, separators=(",", ":")
                ),
            }, f"ADetailer slot {index}")
            image = [node, 0]

        sam_state = self._sam3_state(payload)
        if sam_state is not None:
            state = dict(sam_state)
            # Forge SAM3: its own seed only with "use seed" on (-1 = random);
            # otherwise the image's seed.  One value for mask metadata and
            # the detailer so the saved graph replays exactly.
            sam_seed = concrete_seed(
                _int(state.get("sam3_seed"), -1)
                if _bool(state.get("sam3_use_seed"))
                else payload.get("seed")
            )
            mask_node = graph.add("ForgeNeoSAM3Mask", {
                "image": image,
                "prompt": str(state.get("sam3_prompt") or "face"),
                "exclude_prompt": str(state.get("sam3_exclude_prompt") or ""),
                "mask_mode": str(state.get("sam3_mask_mode") or "Individual"),
                "mask_source": "generated",
                "threshold": _float(state.get("sam3_threshold"), 0.4),
                "detection_limit": max(-1, _int(state.get("sam3_detection_limit"), -1)),
                "convex_hull": _bool(state.get("sam3_mask_hull")),
                "mask_dilation": max(0, _int(state.get("sam3_mask_dilation"), 0)),
                "mask_outline_px": max(0, _int(state.get("sam3_mask_outline_px"), 0)),
                "mask_blur": max(0, _int(state.get("sam3_mask_blur"), 4)),
                "invert": False,
                "checkpoint": str(state.get("sam3_checkpoint") or "sam3.pt"),
                "device": str(state.get("sam3_device") or "cuda"),
                "precision": str(
                    state.get("sam3_precision")
                    or ("fp16" if str(state.get("sam3_device") or "cuda").casefold() == "cuda" else "fp32")
                ),
                "unload_after": _bool(state.get("sam3_unload_after"), True),
                "save_artifacts": _bool(state.get("sam3_save_artifacts"), True),
                # Empty delegates to the node's Comfy output/sam3 directory.
                # sam3_args intentionally has no artifact path field.
                "artifact_directory": str(state.get("sam3_artifact_directory") or ""),
                "seed": sam_seed,
                "enabled": True,
                "cache_model": self._sam3_cache_model(state),
            }, "SAM3 mask")
            if str(state.get("sam3_mode") or "Inpaint").casefold() == "mask only":
                if _bool(state.get("sam3_preview_overlay")):
                    image = [mask_node, 3]
                else:
                    mask_image = graph.add("MaskToImage", {"mask": [mask_node, 0]}, "SAM3 mask image")
                    image = [mask_image, 0]
            else:
                sampler = str(state.get("sam3_sampler") or payload.get("sampler_name") or "euler")
                if sampler.casefold() == "use same sampler":
                    sampler = str(payload.get("sampler_name") or "euler")
                scheduler = str(state.get("sam3_scheduler") or payload.get("scheduler") or "normal")
                if scheduler.casefold() == "use same scheduler":
                    scheduler = str(payload.get("scheduler") or "normal")
                sampler, scheduler = self._runtime_sampler_values(sampler, scheduler)
                seed = sam_seed
                sam3_cfg = _float(state.get("sam3_cfg_scale"), _float(payload.get("cfg_scale"), 7.0)) if _bool(state.get("sam3_use_cfg_scale")) else _float(payload.get("cfg_scale"), 7.0)
                # Detail Daemon: 생성 안의 SAM3 패스는 이 패스의 cfg·샘플러로 다시 판정한다(_add_detail_daemon).
                sam3_model = model if last_pass_model is None else self._add_detail_daemon(
                    graph, model, payload, cfg_scale=sam3_cfg, sampler_name=sampler,
                    title="Anima detail daemon (SAM3 detailer)",
                )
                detail = graph.add(sam3_detailer_class, {
                    "image": image, "mask": [mask_node, 0], "model": sam3_model, "clip": clip, "vae": vae,
                    "positive": positive, "negative": negative,
                    "inpaint_prompt": str(state.get("sam3_inpaint_prompt") or ""),
                    "negative_prompt": str(state.get("sam3_negative_prompt") or ""),
                    "mask_mode": str(state.get("sam3_mask_mode") or "Individual"),
                    "seed": seed,
                    "steps": _int(state.get("sam3_steps"), _int(payload.get("steps"), 28)) if _bool(state.get("sam3_use_steps")) else _int(payload.get("steps"), 28),
                    "cfg": sam3_cfg,
                    "sampler_name": sampler, "scheduler": scheduler,
                    "denoise": _float(state.get("sam3_denoising_strength"), 0.4),
                    "noise_multiplier": _float(state.get("sam3_noise_multiplier"), 1.0) if _bool(state.get("sam3_use_noise_multiplier")) else 1.0,
                    "fill_mode": str(state.get("sam3_inpainting_fill") or "original"),
                    "only_masked": _bool(state.get("sam3_inpaint_only_masked"), True),
                    "mask_padding": _int(state.get("sam3_inpaint_only_masked_padding"), 32),
                    "use_custom_size": _bool(state.get("sam3_use_inpaint_width_height")),
                    "custom_width": _int(state.get("sam3_inpaint_width"), 512),
                    "custom_height": _int(state.get("sam3_inpaint_height"), 512),
                    **dict(zip(
                        ("target_width", "target_height"),
                        self._sam3_processing_size(payload),
                    )),
                    "grow_mask_by": max(0, _int(state.get("sam3_grow_mask_by"), 6)),
                    "controlnet_enable": _bool(state.get("sam3_cn_enable")),
                    "controlnet_model_name": str(state.get("sam3_cn_model") or "None"),
                    "controlnet_module": str(state.get("sam3_cn_module") or "inpaint_only"),
                    "controlnet_override_external": _bool(state.get("sam3_cn_override_external")),
                    "controlnet_strength": _float(state.get("sam3_cn_weight"), 1.0),
                    "controlnet_start": _float(state.get("sam3_cn_guidance_start"), 0.0),
                    "controlnet_end": _float(state.get("sam3_cn_guidance_end"), 1.0),
                    "controlnet_processor_resolution": max(0, _int(state.get("sam3_cn_processor_res"), 512)),
                    "controlnet_settings_json": json.dumps({
                        "pixel_perfect": _bool(state.get("sam3_cn_pixel_perfect"), True),
                        "control_mode": str(state.get("sam3_cn_control_mode") or "Balanced"),
                        "resize_mode": str(state.get("sam3_cn_resize_mode") or "Crop and Resize"),
                        "threshold_a": _float(state.get("sam3_cn_threshold_a"), -1.0),
                        "threshold_b": _float(state.get("sam3_cn_threshold_b"), -1.0),
                        "override_external": _bool(state.get("sam3_cn_override_external")),
                    }, ensure_ascii=False, sort_keys=True, separators=(",", ":")),
                    "restore_face": _bool(state.get("sam3_restore_face")),
                    "restore_face_settings_json": json.dumps({
                        "detector_model": str(state.get("sam3_face_detector_model") or "bbox/face_yolov8m.pt"),
                        "guide_size": _int(state.get("sam3_face_guide_size"), 512),
                        "max_size": _int(state.get("sam3_face_max_size"), 1024),
                        "bbox_threshold": _float(state.get("sam3_face_bbox_threshold"), 0.5),
                        "bbox_dilation": _int(state.get("sam3_face_bbox_dilation"), 10),
                        "bbox_crop_factor": _float(state.get("sam3_face_bbox_crop_factor"), 3.0),
                        "denoise": _float(state.get("sam3_face_denoise"), 0.4),
                        "feather": _int(state.get("sam3_face_feather"), 5),
                        "cycle": _int(state.get("sam3_face_cycle"), 1),
                    }, ensure_ascii=False, sort_keys=True, separators=(",", ":")),
                    "enabled": True,
                }, "SAM3 detailer")
                image = [detail, 0]
        # A sequential detail preset re-detects on the previous pass's result,
        # not on the original image. Reuse the exact supported SAM3 state while
        # avoiding a second ADetailer/Hires or recursive pass schedule.
        for target in payload.get("_comfy_detail_passes", []):
            next_payload = copy.deepcopy(dict(payload))
            next_payload.pop("_comfy_detail_passes", None)
            next_state = dict(self._sam3_state(payload) or {})
            next_state["sam3_prompt"] = target
            next_payload["alwayson_scripts"] = {
                "SAM3 Mask": {"args": [next_state]},
                # Detail Daemon 은 SAM3 패스마다 다시 판정한다(_add_detail_daemon) — 블록을 같이 넘긴다.
                **self._detail_daemon_scripts(next_payload),
            }
            image = self._add_image_extensions(
                graph, image, model, clip, vae, positive, negative, next_payload,
                sam3_detailer_class=sam3_detailer_class, last_pass_model=last_pass_model,
            )
        return image

    # ---- custom workflow ----------------------------------------------

    def _compile_custom(
        self, mode: str, model_name: str, payload: dict, loras: Sequence[LoraSpec],
        workflow: Mapping[str, Any], anima_plan: _Anima38Plan,
        *, uploaded_image: str, uploaded_mask: str,
    ) -> dict:
        graph = _Graph(workflow)
        sampler_id = self._find_sampler(graph.nodes)
        sampler = graph.nodes[sampler_id]
        inputs = self._sampler_inputs(sampler)
        self._map_sampler_inputs(inputs, sampler.get("class_type", ""), payload, mode=mode)
        self._apply_custom_latent_size(graph.nodes, inputs, payload, mode=mode)

        loader_id: Optional[str] = None
        loader_type = ""
        if model_name:
            # compile() passes a model only for a rewritable loader chain.
            loader_id = self._trace_model_loader(graph.nodes, inputs.get("model"))
            if not loader_id:
                raise WorkflowCompileError(
                    "custom workflow sampler의 upstream 모델 로더("
                    + "/".join(MODEL_LOADER_INPUTS) + ")를 찾지 못했습니다."
                )
            loader = graph.nodes[loader_id]
            loader_type = str(loader.get("class_type") or "")
            if not anima_plan.v2:
                if loader_type == "ForgeNeoAnima38V2Loader":
                    loader_type = "UNETLoader"
                    loader["class_type"] = loader_type
                    loader["inputs"] = {
                        "weight_dtype": str(payload.get("weight_dtype") or "default"),
                    }
                loader_input = MODEL_LOADER_INPUTS[loader_type]
                loader.setdefault("inputs", {})[loader_input] = self._resolve_choice(
                    loader_type, loader_input, model_name,
                )

        pos_ids = self._trace_classes(
            graph.nodes, inputs.get("positive"), _TEXT_ENCODERS,
        )
        neg_ids = self._trace_classes(
            graph.nodes, inputs.get("negative"), _TEXT_ENCODERS,
        )
        if len(pos_ids) != 1 or len(neg_ids) != 1:
            raise WorkflowCompileError(
                "custom workflow의 positive/negative CLIPTextEncode 연결은 각각 하나여야 합니다."
            )
        pos_id, neg_id = pos_ids[0], neg_ids[0]
        self._shares_text_encoder(pos_id, neg_id, payload.get("negative_prompt"))
        if anima_plan.semantic and any(
            graph.nodes[node_id].get("class_type") not in {"CLIPTextEncode"} | _SEMANTIC_ENCODERS
            for node_id in (pos_id, neg_id)
        ):
            raise WorkflowCompileError(
                "Anima Semantic Connector custom workflow는 positive/negative에 "
                "각각 일반 CLIPTextEncode 한 개가 필요합니다."
            )
        model = inputs.get("model")
        positive, negative = inputs.get("positive"), inputs.get("negative")
        if not (_is_link(model) and _is_link(positive) and _is_link(negative)):
            raise WorkflowCompileError("custom workflow sampler의 model/positive/negative 연결이 유효하지 않습니다.")
        pos_clip = graph.nodes[pos_id].get("inputs", {}).get(self._encode_clip_input(graph.nodes[pos_id]))
        neg_clip = graph.nodes[neg_id].get("inputs", {}).get(self._encode_clip_input(graph.nodes[neg_id]))
        if not (_is_link(pos_clip) and _is_link(neg_clip)):
            raise WorkflowCompileError("custom workflow CLIP 연결을 찾지 못했습니다.")
        if pos_clip != neg_clip and (
            loras or self._script(payload, "NegPiP") or anima_plan.semantic
        ):
            raise WorkflowCompileError("서로 다른 positive/negative CLIP을 쓰는 custom workflow에는 LoRA/NegPiP를 자동 삽입할 수 없습니다.")
        clip = pos_clip
        active_ids = self._upstream_node_ids(
            graph.nodes,
            (inputs.get("model"), inputs.get("positive"), inputs.get("negative")),
        )
        active_ids.add(sampler_id)
        active_lora_ids, active_clip_uses_lora = self._rewrite_custom_anima_loras(
            graph, inputs.get("model"), active_ids, pos_clip, neg_clip, anima_plan,
        )

        decode_id = self._find_decode_after(graph.nodes, sampler_id)
        vae = self._find_vae_link_for_branch(
            graph.nodes, inputs.get("latent_image"), decode_id,
        ) or self._infer_vae_from_model(graph.nodes, model)
        needs_vae = (
            mode != "txt2img"
            or _bool(payload.get("enable_hr"))
            or self._has_image_scripts(payload)
        )
        if needs_vae and not _is_link(vae):
            raise WorkflowCompileError("custom workflow의 VAE 연결을 찾지 못했습니다.")
        override_vae = self._override_custom_modules(
            graph, payload, pos_id, neg_id, decode_id, inputs.get("latent_image"),
        )
        # Overrides can replace encoder/VAE links.
        clip = graph.nodes[pos_id].get("inputs", {}).get(self._encode_clip_input(graph.nodes[pos_id]), clip)
        if override_vae is not None and active_clip_uses_lora:
            clip = self._rebase_custom_lora_clips(graph, active_lora_ids, clip)
            for node_id in (pos_id, neg_id):
                node = graph.nodes[node_id]
                node.setdefault("inputs", {})[self._encode_clip_input(node)] = clip
        vae = override_vae or vae or []

        if anima_plan.v2:
            if not loader_id:
                raise WorkflowCompileError(
                    "Anima 3.8B v2 custom workflow model loader를 찾지 못했습니다."
                )
            if loader_type in CHECKPOINT_LOADER_NODES:
                stale_consumers = self._direct_link_consumers(
                    graph.nodes, loader_id, {1, 2},
                )
                if stale_consumers:
                    detail = ", ".join(
                        f"{node_id}.{name}" for node_id, name in stale_consumers
                    )
                    raise WorkflowCompileError(
                        "Anima 3.8B v2 loader는 MODEL만 반환하지만 custom workflow가 "
                        "기존 checkpoint의 CLIP/VAE 출력을 계속 사용합니다: " + detail
                    )
            shared_model = [
                item for item in self._direct_link_consumers(
                    graph.nodes, loader_id, {0},
                )
                if item[0] not in active_ids
            ]
            if shared_model:
                detail = ", ".join(
                    f"{node_id}.{name}" for node_id, name in shared_model
                )
                raise WorkflowCompileError(
                    "Anima 3.8B v2 loader를 선택하지 않은 custom workflow 분기가 "
                    "공유하고 있어 안전하게 교체할 수 없습니다: " + detail
                )
            graph.nodes[loader_id]["class_type"] = "ForgeNeoAnima38V2Loader"
            graph.nodes[loader_id]["inputs"] = {
                "model_name": self._resolve_choice(
                    "ForgeNeoAnima38V2Loader", "model_name",
                    anima_plan.model_name or model_name,
                ),
            }

        model, clip = self._add_model_patches(graph, model, clip, payload, loras, anima_plan)
        self._rewrite_custom_conditioning(
            graph, pos_id, neg_id, model, clip, payload, anima_plan,
            negative_clip=(
                neg_clip if pos_clip != neg_clip and override_vae is None else clip
            ),
        )
        model, sampler_options = self._add_anima_guidance(
            graph, model, clip, positive, negative, payload,
        )
        # Detail Daemon 은 이 sampler(base) 또는 hires 한 곳에만 — model 자체에는 없다. 디테일러가 받는 모델은
        # _add_image_extensions(last_pass_model=) 이 정한다.
        inputs["model"] = pass_model = self._add_detail_daemon(graph, model, payload)

        if mode != "txt2img":
            if not uploaded_image:
                raise WorkflowCompileError(f"{mode} 입력 이미지가 업로드되지 않았습니다.")
            latent = self._add_latent(
                graph, mode, vae, payload,
                uploaded_image=uploaded_image, uploaded_mask=uploaded_mask,
            )
            inputs["latent_image"] = latent
        # CNS rides on the MODEL, not on the sampler node: the suite's MODEL carries the
        # pack's SAMPLER_SAMPLE wrapper (guidance_cns.apply_cns), which colours the step
        # noise of whatever KSAMPLER runs — the original is a SAMPLER patch used with
        # SamplerCustomAdvanced (origin: namemechan/comfyui-cns_sampler_patch@42278b13:
        # cns_sampler_patch.py:441-609).  So CNS refuses no sampler class (KSamplerAdvanced
        # here; SamplerCustom* is refused earlier by unsupported_sampler_message for its
        # steps/CFG/denoise, not for CNS — plan §5.3 A wants it allowed; that payload
        # mapping is a deferred open item, not parity).  A pack older than the per-step
        # CNS wrapper would drop CNS here silently, so validate() refuses CNS without
        # _PER_STEP_CNS_MARKER.  A plain KSampler still becomes
        # ForgeNeoKSamplerCNS with the same CNS values as inputs (apply_cns replaces the
        # MODEL's CNS wrapper, never stacks one).  Spectrum/SPEED exist only on that node.
        spectrum_or_speed = (
            _bool(payload.get("spectrum_enabled")) or _bool(payload.get("speed_enabled"))
        )
        if sampler_options.get("cns_enabled") or spectrum_or_speed:
            class_type = sampler.get("class_type")
            if class_type == "KSampler":
                sampler["class_type"] = class_type = "ForgeNeoKSamplerCNS"
            elif spectrum_or_speed and class_type != "ForgeNeoKSamplerCNS":
                raise WorkflowCompileError(
                    "Spectrum/SPEED 자동 삽입은 KSampler custom workflow에서만 지원됩니다."
                )
        else:
            class_type = None
        if class_type == "ForgeNeoKSamplerCNS":
            for key, default in {
                "cns_enabled": False, **_CNS_DEFAULTS,
                "spectrum_enabled": False, "spectrum_window_size": 2.0, "spectrum_flex_window": 0.25,
                "spectrum_warmup_steps": 6, "spectrum_tail_actual_steps": 3, "spectrum_blend_w": 0.3,
                "spectrum_cheby_degree": 3, "spectrum_ridge_lambda": 0.1, "spectrum_history_size": 100,
                "spectrum_one_sampler_only": False, "spectrum_verbose": False,
                "speed_enabled": False, "speed_split_mode": "single", "speed_spd_scale": 0.5,
                "speed_spd_sigma": 0.7, "speed_adaptive_smc_alpha": 0.0,
            }.items():
                inputs[key] = sampler_options.get(key, payload.get(key, default))

        if not decode_id:
            if _bool(payload.get("enable_hr")) or self._has_image_scripts(payload):
                raise WorkflowCompileError("Hires/ADetailer/SAM3 삽입에 필요한 VAEDecode를 custom workflow에서 찾지 못했습니다.")
            return graph.nodes
        decode = graph.nodes[decode_id]
        samples = decode.get("inputs", {}).get("samples")
        if not _is_link(samples):
            raise WorkflowCompileError("custom workflow VAEDecode의 samples 연결이 유효하지 않습니다.")
        if _bool(payload.get("enable_hr")):
            pass_model = self._add_detail_daemon(graph, model, payload, hires_pass=True)
            samples, decode_vae = self._add_hires(
                graph, pass_model,
                clip, vae, positive, negative, samples, payload,
                sampler_options=sampler_options, anima_plan=anima_plan,
            )
            decode.setdefault("inputs", {})["samples"] = samples
            decode["inputs"]["vae"] = decode_vae
        if self._has_image_scripts(payload):
            outputs = self._find_outputs_after(graph.nodes, decode_id)
            if not outputs:
                raise WorkflowCompileError(
                    "ADetailer/SAM3 삽입 대상인 custom workflow 출력 노드를 찾지 못했습니다."
                )
            source_links = {tuple(link[:2]) for _node_id, _key, link in outputs}
            if len(source_links) != 1:
                raise WorkflowCompileError(
                    "custom workflow의 선택 분기에 서로 다른 이미지 출력이 여러 개입니다. "
                    "ADetailer/SAM3 자동 삽입 대상을 하나로 줄여주세요."
                )
            image = list(next(iter(source_links)))
            post = self._add_image_extensions(
                graph, image, model, clip, vae, positive, negative, payload,
                last_pass_model=pass_model,
            )
            if post != image:
                for output_id, key, _source in outputs:
                    graph.nodes[output_id].setdefault("inputs", {})[key] = post
        return graph.nodes

    # ---- external (Generation API profile) workflows --------------------

    def map_external_workflow(
        self,
        workflow: Mapping[str, Any],
        mode: str,
        model_name: str,
        payload: Mapping[str, Any],
        *,
        warnings: Optional[list] = None,
        missing_loras: Optional[list] = None,
    ) -> dict:
        """Apply a Forge payload to a caller-owned API workflow, without app nodes.

        Generation API profiles may point at a remote or stock ComfyUI that
        does not have the bundled node pack, so unlike ``compile(workflow=…)``
        this inserts no ForgeNeo nodes (no latent/Hires/ADetailer/SAM3/NegPiP
        rewiring), and it accepts multi-pass graphs: the payload goes to the
        *main* sampler only (``_find_external_sampler`` — the first pass that
        leads to the image output), and every other sampler keeps the
        workflow's own values, as the legacy mapper did.  On that sampler it
        shares the compiler's rules: Forge sampler names; txt2img denoise 1.0
        and the KSamplerAdvanced step window (steps, scheduler and window kept
        as authored when the pass hands leftover noise to a later sampler —
        they are one sigma schedule with that pass, and so is the sampler's
        sigma layout: a payload sampler whose layout would move the noise
        boundary with the continuation pass's own sampler is reported in
        ``warnings`` and the authored one kept); latent size/batch; the
        model on a rewritable loader (a custom loader keeps its model).
        ``<lora:…>`` tags leave the prompt and become ``LoraLoader`` nodes
        for every consumer of that model/CLIP; like Forge, a LoRA the target
        does not have is skipped and reported in ``warnings`` (its name also
        in ``missing_loras`` — the only warning a fresher schema can change)
        instead of failing the job.  The input image of img2img is already
        written into ``LoadImage`` by the caller.  Returns a new graph;
        ``workflow``/``payload`` are untouched.
        """
        normalized = {
            "t2i": "txt2img", "txt2img": "txt2img", "i2i": "img2img", "img2img": "img2img",
        }.get(str(mode or "").strip().casefold())
        if normalized is None:
            raise WorkflowCompileError(f"지원하지 않는 ComfyUI 생성 모드입니다: {mode!r}")
        local_payload = copy.deepcopy(dict(payload))
        if _bool(local_payload.get("enable_hr")):
            raise WorkflowCompileError(
                "대상 프로필의 사용자 ComfyUI 워크플로에는 Hires.fix를 자동으로 넣을 수 "
                "없습니다. 워크플로 안에 업스케일 단계를 두거나 enable_hr를 끄세요."
            )
        if normalized != "txt2img":
            local_payload.setdefault("denoising_strength", 0.75)
        local_payload["seed"] = concrete_seed(local_payload.get("seed", -1))
        loras, prompts = parse_lora_tags(
            str(local_payload.get("prompt", "") or ""),
            str(local_payload.get("negative_prompt", "") or ""),
        )
        positive_text, negative_text = prompts

        graph = _Graph(workflow)
        sampler_id = self._find_external_sampler(graph.nodes)
        sampler = graph.nodes[sampler_id]
        inputs = self._sampler_inputs(sampler)
        keep_step_window = self._hands_off_leftover_noise(graph.nodes, sampler_id)
        self._map_sampler_inputs(
            inputs, str(sampler.get("class_type") or ""), local_payload, mode=normalized,
            keep_step_window=keep_step_window,
            continuation_samplers=(
                self._continuation_sampler_names(graph.nodes, sampler_id)
                if keep_step_window else ()
            ),
            warnings=warnings,
        )
        self._apply_custom_latent_size(graph.nodes, inputs, local_payload, mode=normalized)

        if model_name:
            loader_id = self._trace_model_loader(graph.nodes, inputs.get("model"))
            if loader_id:
                loader = graph.nodes[loader_id]
                loader_type = str(loader.get("class_type") or "")
                loader_input = MODEL_LOADER_INPUTS[loader_type]
                loader.setdefault("inputs", {})[loader_input] = self._resolve_choice(
                    loader_type, loader_input, model_name,
                )
            # else: a custom loader owns the model (same rule as compile()).

        pos_ids = self._trace_classes(graph.nodes, inputs.get("positive"), _TEXT_ENCODERS)
        neg_ids = self._trace_classes(graph.nodes, inputs.get("negative"), _TEXT_ENCODERS)
        if len(pos_ids) != 1 or len(neg_ids) != 1:
            raise WorkflowCompileError(
                "workflow의 positive/negative 텍스트 인코더 연결은 각각 하나여야 합니다."
            )
        pos_id, neg_id = pos_ids[0], neg_ids[0]
        shared_encoder = self._shares_text_encoder(pos_id, neg_id, negative_text)
        pos_node, neg_node = graph.nodes[pos_id], graph.nodes[neg_id]
        available = self._available_external_loras(loras, warnings, missing_loras)
        if available:
            model = inputs.get("model")
            clip = self._node_inputs(pos_node).get(self._encode_clip_input(pos_node))
            neg_clip = self._node_inputs(neg_node).get(self._encode_clip_input(neg_node))
            if not (_is_link(model) and _is_link(clip)) or clip != neg_clip:
                raise WorkflowCompileError(
                    "<lora:…> 태그를 넣으려면 sampler의 model과 positive/negative "
                    "인코더가 같은 CLIP에 연결되어 있어야 합니다."
                )
            before = set(graph.nodes)
            lora_model, lora_clip = self._add_loras(graph, model, clip, available, None)
            lora_ids = set(graph.nodes) - before
            # Forge applies prompt LoRAs to the whole job: every pass (and
            # encoder) that used this model/CLIP now uses the patched one.
            self._redirect_link(graph.nodes, model, lora_model, skip=lora_ids)
            self._redirect_link(graph.nodes, clip, lora_clip, skip=lora_ids)
        targets = [(pos_node, positive_text)]
        if not shared_encoder:
            targets.append((neg_node, negative_text))
        for node, text in targets:
            if node.get("class_type") in _SEMANTIC_ENCODERS:
                node.setdefault("inputs", {})["prompt"] = text
            else:
                self._set_encode_text(node, text)
        return graph.nodes

    def _available_external_loras(
        self, loras: Sequence[LoraSpec], warnings: Optional[list],
        missing: Optional[list] = None,
    ) -> list[LoraSpec]:
        """Prompt LoRAs the target has, by its exact name; the rest are reported.

        Forge's rule for an unknown ``<lora:…>``: the tag leaves the prompt, a
        warning is logged and the image is still generated.  An ambiguous name
        stays an error (``_match_choice``) — picking one would be a guess.
        Skipped names also go to ``missing`` (a cached schema can predate them).
        """
        choices = self._choices("LoraLoader", "lora_name")
        available: list[LoraSpec] = []
        for spec in loras:
            name = self._match_choice(spec.name, choices)
            if name is None:
                if warnings is not None:
                    warnings.append(f"대상 ComfyUI에 없는 LoRA라 건너뛰었습니다: {spec.name}")
                if missing is not None:
                    missing.append(spec.name)
                continue
            available.append(LoraSpec(name, spec.strength_model, spec.strength_clip))
        return available

    @staticmethod
    def _redirect_link(
        nodes: Mapping[str, Any], old: Sequence[Any], new: Sequence[Any],
        *, skip: Iterable[str] = (),
    ) -> None:
        """Point every input that reads link ``old`` at ``new`` (``skip`` excluded)."""
        skipped = {str(node_id) for node_id in skip}
        source = (str(old[0]), old[1])
        for node_id, node in nodes.items():
            if str(node_id) in skipped or not isinstance(node, Mapping):
                continue
            node_inputs = node.get("inputs")
            if not isinstance(node_inputs, dict):
                continue
            for key, value in node_inputs.items():
                if _is_link(value) and (str(value[0]), value[1]) == source:
                    node_inputs[key] = list(new)

    @staticmethod
    def _shares_text_encoder(pos_id: str, neg_id: str, negative_text: Any) -> bool:
        """True when the negative conditioning is derived from the positive encoder.

        ``ConditioningZeroOut`` (or any other conditioning node) fed by the
        positive encoder makes both sampler inputs trace to one text encoder.
        Writing the negative text there would replace the positive prompt, so
        only the positive prompt is written, and a negative prompt that has no
        encoder of its own is an explicit error instead of a silent swap.
        """
        if str(pos_id) != str(neg_id):
            return False
        if str(negative_text or "").strip():
            raise WorkflowCompileError(
                "이 워크플로의 negative 조건은 positive 텍스트 인코더에서 만들어집니다"
                "(예: ConditioningZeroOut). 네거티브 프롬프트를 넣을 인코더가 없으니 "
                "네거티브 프롬프트를 비우거나, negative 전용 CLIPTextEncode를 연결하세요."
            )
        return True

    @classmethod
    def _find_external_sampler(cls, workflow: Mapping[str, Any]) -> str:
        """The sampler that receives an external payload in a multi-pass graph.

        A pass whose latent comes from another sampler (Hires refine, SDXL
        refiner, upscale-and-resample) is a follow-up pass and keeps its
        authored values; among the first passes, one whose result reaches an
        image output wins, and ties keep workflow order (the legacy mapper
        took the first sampler).  ``SamplerCustom*`` is rejected later only
        when it is this sampler, and only because the payload's steps/CFG/
        denoise cannot be mapped onto it (``unsupported_sampler_message``) —
        CNS is a MODEL-level sampler wrapper and limits no sampler class.
        """
        samplers = [
            str(node_id) for node_id, node in workflow.items()
            if isinstance(node, Mapping) and node.get("class_type") in _SAMPLERS
        ]
        if not samplers:
            raise WorkflowCompileError("workflow에서 sampler 노드를 찾지 못했습니다.")
        if len(samplers) == 1:
            return samplers[0]
        first_passes = [
            sampler_id for sampler_id in samplers
            if not any(
                other != sampler_id and cls._link_depends_on(
                    workflow,
                    cls._node_inputs(workflow.get(sampler_id)).get("latent_image"),
                    other,
                )
                for other in samplers
            )
        ] or samplers
        output_links = [
            value
            for node in workflow.values()
            if isinstance(node, Mapping) and node.get("class_type") in _SAVE_NODES
            for value in cls._node_inputs(node).values()
            if _is_link(value)
        ]
        reaching = [
            sampler_id for sampler_id in first_passes
            if any(cls._link_depends_on(workflow, link, sampler_id) for link in output_links)
        ]
        return (reaching or first_passes)[0]

    @classmethod
    def _hands_off_leftover_noise(cls, workflow: Mapping[str, Any], sampler_id: str) -> bool:
        """True for a KSamplerAdvanced pass that returns leftover noise to a later sampler.

        That is a split schedule (e.g. SDXL base + refiner): its ``steps``,
        ``scheduler``, step window and the sampler's sigma layout
        (``_PENULTIMATE_SIGMA_DISCARD_SAMPLERS``) belong to the workflow, or
        the next pass would start from the wrong noise level.
        """
        node = workflow.get(sampler_id)
        if not isinstance(node, Mapping) or node.get("class_type") != "KSamplerAdvanced":
            return False
        leftover = str(cls._node_inputs(node).get("return_with_leftover_noise") or "")
        if leftover.strip().casefold() != "enable":
            return False
        return any(
            str(other_id) != str(sampler_id)
            and isinstance(other, Mapping)
            and other.get("class_type") in _SAMPLERS
            and cls._link_depends_on(
                workflow, cls._node_inputs(other).get("latent_image"), sampler_id,
            )
            for other_id, other in workflow.items()
        )

    @classmethod
    def _continuation_sampler_names(
        cls, workflow: Mapping[str, Any], sampler_id: str,
    ) -> list[Any]:
        """``sampler_name`` of every pass that takes ``sampler_id``'s latent directly.

        A continuation pass is a sampler whose ``latent_image`` traces back to
        ``sampler_id`` with no other sampler in between (an SDXL refiner, not
        a later Hires pass fed by the refiner).  Its own sampler decides the
        sigma layout of the step it starts at.  The value is kept as authored:
        a link (a sampler picked by another node) or ``None`` (a sampler class
        without ``sampler_name``) means that pass's layout is unknown.
        """
        names: list[Any] = []
        for other_id, other in workflow.items():
            if (
                str(other_id) == str(sampler_id)
                or not isinstance(other, Mapping)
                or other.get("class_type") not in _SAMPLERS
            ):
                continue
            other_inputs = cls._node_inputs(other)
            upstream = cls._trace_classes(workflow, other_inputs.get("latent_image"), _SAMPLERS)
            if str(sampler_id) in upstream:
                names.append(other_inputs.get("sampler_name"))
        return names

    # ---- small helpers -------------------------------------------------

    def _apply_custom_latent_size(
        self, nodes: dict, sampler_inputs: Mapping[str, Any],
        payload: Mapping[str, Any], *, mode: str,
    ) -> None:
        """Write size/batch (and ForgeNeoLatentInput mode) on the sampler's latent source."""
        batch = max(1, _int(payload.get("batch_size"), 1)) * max(
            1, _int(payload.get("n_iter", payload.get("batch_count", 1)), 1)
        )
        latent_ids = self._trace_classes(
            nodes, sampler_inputs.get("latent_image"),
            {"EmptyLatentImage", "ForgeNeoLatentInput"},
        )
        if len(latent_ids) > 1:
            raise WorkflowCompileError(
                "custom workflow의 sampler latent 분기에 EmptyLatentImage가 여러 개입니다."
            )
        if not latent_ids:
            return
        latent_id = latent_ids[0]
        latent_inputs = nodes[latent_id].setdefault("inputs", {})
        latent_inputs["width"] = max(64, _int(payload.get("width"), 512))
        latent_inputs["height"] = max(64, _int(payload.get("height"), 512))
        latent_inputs["batch_size"] = batch
        if nodes[latent_id].get("class_type") == "ForgeNeoLatentInput":
            latent_inputs["mode"] = mode

    def _map_sampler_inputs(
        self, inputs: dict, class_type: str, payload: Mapping[str, Any], *, mode: str,
        keep_step_window: bool = False,
        continuation_samplers: Sequence[Any] = (),
        warnings: Optional[list] = None,
    ) -> None:
        """Write the payload's sampling settings onto one sampler node.

        ``keep_step_window`` (external multi-pass graphs only): a
        ``KSamplerAdvanced`` that splits one schedule with a later pass keeps
        its authored ``steps``/``scheduler``/start/end/leftover-noise inputs —
        the sigma schedule is shared with the later pass, so writing only this
        pass's scheduler (even the ``normal`` default of a payload without one)
        would hand the next pass a latent at the wrong noise level.  Seed and
        cfg still come from the payload, and so does the sampler algorithm
        when its sigma layout keeps the boundary with the later pass, whose
        ``sampler_name`` values are ``continuation_samplers``
        (``_map_split_pass_sampler``).
        """
        blocked = unsupported_sampler_message(class_type)
        if blocked:
            raise WorkflowCompileError(blocked)
        seed = concrete_seed(payload.get("seed"))
        inputs["noise_seed" if class_type == "KSamplerAdvanced" else "seed"] = seed
        inputs["cfg"] = _float(payload.get("cfg_scale"), 7.0)
        # Resolved even for a split pass so an unsupported sampler/scheduler still fails.
        sampler, scheduler = self._runtime_sampler_values(
            payload.get("sampler_name") or "euler",
            payload.get("scheduler") or "normal",
        )
        if keep_step_window and class_type == "KSamplerAdvanced":
            self._map_split_pass_sampler(
                inputs, payload, sampler, continuation_samplers, warnings,
            )
            return
        inputs["sampler_name"] = sampler
        inputs["scheduler"] = scheduler
        steps = max(1, _int(payload.get("steps"), 20))
        inputs["steps"] = steps
        denoise = (
            1.0 if mode == "txt2img" else
            max(0.0, min(1.0, _float(payload.get("denoising_strength"), 0.75)))
        )
        if class_type == "KSamplerAdvanced":
            inputs["add_noise"] = "enable"
            inputs["start_at_step"] = max(0, steps - int(steps * denoise))
            inputs["end_at_step"] = steps
            inputs["return_with_leftover_noise"] = "disable"
        else:
            inputs["denoise"] = denoise

    @staticmethod
    def _map_split_pass_sampler(
        inputs: dict, payload: Mapping[str, Any], sampler: str,
        continuation_samplers: Sequence[Any], warnings: Optional[list],
    ) -> None:
        """The payload's sampler on a split-schedule pass, only if the boundary holds.

        ComfyUI builds each pass's sigmas from its own sampler: the
        ``_PENULTIMATE_SIGMA_DISCARD_SAMPLERS`` (DPM2/UniPC) use steps+1 and
        drop the next-to-last sigma, so ``end_at_step`` here and the
        continuation pass's ``start_at_step`` name one noise level only when
        both passes are on the same side of that line.  The boundary is with
        the continuation pass, so its own ``sampler_name``
        (``continuation_samplers``, see ``_continuation_sampler_names``)
        decides: a payload sampler of that layout is written.  So is one of
        the authored sampler's layout — the boundary then stays exactly as
        authored, which is all that can be checked when the continuation's
        sampler is wired from another node; when the authored passes already
        disagree, that is reported.  Any other sampler keeps the authored one
        (a link included) and is reported.  A payload without a sampler keeps
        the authored one, like the scheduler.
        """
        authored = inputs.get("sampler_name")
        if authored is None or authored == "":
            inputs["sampler_name"] = sampler  # nothing authored to keep
            return
        if not str(payload.get("sampler_name") or "").strip():
            return
        discards = _PENULTIMATE_SIGMA_DISCARD_SAMPLERS
        continuations = list(continuation_samplers)
        continuation_known = bool(continuations) and all(
            isinstance(name, str) and name for name in continuations
        )
        next_names = ", ".join(dict.fromkeys(continuations)) if continuation_known else ""
        # 다음 패스와 같은 시그마 배열 → 경계가 맞는다.
        aligned = continuation_known and all(
            (name in discards) == (sampler in discards) for name in continuations
        )
        # 작성된 첫 패스와 같은 배열 → 경계가 작성된 그대로다(맞든 이미 어긋났든).
        as_authored = isinstance(authored, str) and (
            (authored in discards) == (sampler in discards)
        )
        if aligned or as_authored:
            inputs["sampler_name"] = sampler
            if not aligned and continuation_known and warnings is not None:
                warnings.append(
                    f"분할 샘플링 워크플로의 두 패스가 이미 시그마 배열이 달라({authored} → "
                    f"{next_names}) 노이즈 경계가 어긋나 있습니다 — {sampler}도 첫 패스와 같은 "
                    f"배열입니다. 경계를 맞추려면 다음 패스({next_names})와 시그마 배열이 같은 "
                    "샘플러를 고르세요."
                )
            return
        if warnings is None:
            return
        first = (
            f"첫 패스 샘플러({authored})" if isinstance(authored, str)
            else "첫 패스 샘플러(다른 노드에서 연결됨)"
        )
        if continuation_known:
            warnings.append(
                f"분할 샘플링 워크플로라 {first}를 유지했습니다 — {sampler}는 다음 패스 "
                f"샘플러({next_names})와 시그마 배열이 달라 노이즈 경계가 어긋납니다."
            )
        else:
            warnings.append(
                f"분할 샘플링 워크플로라 {first}를 유지했습니다 — 다음 패스 샘플러를 알 수 "
                f"없어(다른 노드에서 연결됨) {sampler}가 같은 시그마 배열인지 확인할 수 없습니다."
            )

    @staticmethod
    def _encode_clip_input(node: Mapping[str, Any]) -> str:
        return "native_clip" if node.get("class_type") in _SEMANTIC_ENCODERS else "clip"

    @staticmethod
    def _set_encode_text(node: dict, text: str) -> None:
        inputs = node.setdefault("inputs", {})
        if node.get("class_type") == "CLIPTextEncodeSDXL":
            inputs["text_g"] = text
            inputs["text_l"] = text
        else:
            inputs["text"] = text

    def _rewrite_custom_conditioning(
        self,
        graph: _Graph,
        pos_id: str,
        neg_id: str,
        model: list,
        clip: list,
        payload: Mapping[str, Any],
        anima_plan: _Anima38Plan,
        *, negative_clip: Optional[list] = None,
    ) -> None:
        positive_text = str(payload.get("prompt") or "")
        negative_text = str(payload.get("negative_prompt") or "")
        # One encoder feeding both (ConditioningZeroOut negative): write the
        # positive prompt only — the negative would otherwise replace it.
        shared_encoder = self._shares_text_encoder(pos_id, neg_id, negative_text)
        if not anima_plan.semantic:
            targets = [(pos_id, positive_text, clip)]
            if not shared_encoder:
                targets.append(
                    (neg_id, negative_text, negative_clip if negative_clip is not None else clip)
                )
            for node_id, text, encoder in targets:
                if graph.nodes[node_id].get("class_type") in _SEMANTIC_ENCODERS:
                    graph.nodes[node_id]["class_type"] = "CLIPTextEncode"
                    graph.nodes[node_id]["inputs"] = {}
                self._set_encode_text(graph.nodes[node_id], text)
                graph.nodes[node_id].setdefault("inputs", {})["clip"] = encoder
            return

        # Imported guidance nodes can already depend on these encoders.
        # Semantic encoding needs the upstream diffusion model/connector,
        # not the downstream sampling hooks, which would create a cycle.
        conditioning_model = model
        visited: set[str] = set()
        while any(
            self._link_depends_on(graph.nodes, conditioning_model, node_id)
            for node_id in (pos_id, neg_id)
        ):
            source_id = str(conditioning_model[0])
            if source_id in visited:
                raise WorkflowCompileError("custom workflow semantic model 연결이 순환합니다.")
            visited.add(source_id)
            conditioning_model = graph.nodes[source_id].get("inputs", {}).get("model")
            if not _is_link(conditioning_model):
                raise WorkflowCompileError("custom workflow semantic encoding의 upstream model을 찾지 못했습니다.")

        qwen_node = graph.add(
            "ForgeNeoAnimaQwen35Loader",
            {"qwen35_model": anima_plan.qwen35_model},
            "Anima Qwen3.5 semantic encoder",
        )
        qwen_clip = [qwen_node, 0]

        def semantic_inputs(text: str, strength: float) -> dict[str, Any]:
            common = {
                "model": conditioning_model,
                "native_clip": clip,
                "qwen35_clip": qwen_clip,
                "prompt": text,
            }
            if anima_plan.conditioning_kind == "v1":
                common["adapter_name"] = anima_plan.adapter_name
                common["adapter_strength"] = strength
            return common

        prompt_type = (
            "ForgeNeoAnima38V2Prompt"
            if anima_plan.conditioning_kind == "v2"
            else "ForgeNeoAnimaQwen35Prompt"
        )
        graph.nodes[pos_id]["class_type"] = prompt_type
        graph.nodes[pos_id]["inputs"] = semantic_inputs(
            positive_text, anima_plan.settings.strength,
        )
        if shared_encoder:
            return
        if anima_plan.settings.negative:
            graph.nodes[neg_id]["class_type"] = prompt_type
            graph.nodes[neg_id]["inputs"] = semantic_inputs(
                negative_text, anima_plan.settings.negative_strength,
            )
        else:
            graph.nodes[neg_id]["class_type"] = "CLIPTextEncode"
            graph.nodes[neg_id]["inputs"] = {
                "clip": clip,
                "text": negative_text,
            }

    @staticmethod
    def _direct_link_consumers(
        workflow: Mapping[str, Any], source_id: str, output_indexes: set[int],
    ) -> list[tuple[str, str]]:
        consumers: list[tuple[str, str]] = []
        for node_id, node in workflow.items():
            if not isinstance(node, Mapping):
                continue
            inputs = node.get("inputs", {})
            if not isinstance(inputs, Mapping):
                continue
            for name, value in inputs.items():
                if (
                    _is_link(value)
                    and str(value[0]) == str(source_id)
                    and _int(value[1], -1) in output_indexes
                ):
                    consumers.append((str(node_id), str(name)))
        return consumers

    @staticmethod
    def _upstream_node_ids(
        workflow: Mapping[str, Any], roots: Iterable[Any],
    ) -> set[str]:
        active: set[str] = set()

        def visit(value: Any) -> None:
            if not _is_link(value):
                return
            node_id = str(value[0])
            if node_id in active:
                return
            node = workflow.get(node_id)
            if not isinstance(node, Mapping):
                return
            active.add(node_id)
            inputs = node.get("inputs", {})
            if isinstance(inputs, Mapping):
                for upstream in inputs.values():
                    visit(upstream)

        for root in roots:
            visit(root)
        return active

    def _rewrite_custom_anima_loras(
        self,
        graph: _Graph,
        model_link: Any,
        active_ids: set[str],
        pos_clip: Any,
        neg_clip: Any,
        anima_plan: _Anima38Plan,
    ) -> tuple[list[str], bool]:
        """Upgrade only active native LoRAs; reject unsupported/shared seams.

        Core ``LoraLoader`` becomes the bundled ANIMA loader with the same
        inputs.  Bundled nodes that already handle ANIMA
        block layouts (``_ANIMA_SAFE_LORA_NODES``, including block weighting)
        stay as they are.  Returns the CLIP-carrying LoRA chain (model order)
        and whether both encoders read its CLIP output.
        """

        if not anima_plan.is_anima:
            return [], False
        lora_ids: list[str] = []      # CLIP-carrying LoRA chain, model order
        remap_ids: list[str] = []     # core loaders switched to ANIMA classes
        visited: set[str] = set()
        link = model_link
        for _depth in range(31):
            if not _is_link(link):
                break
            node_id = str(link[0])
            if node_id in visited:
                break
            visited.add(node_id)
            node = graph.nodes.get(node_id)
            if not isinstance(node, Mapping):
                break
            class_type = str(node.get("class_type") or "")
            if class_type in MODEL_LOADER_INPUTS:
                break
            if class_type in _ANIMA_LORA_REMAP:
                remap_ids.append(node_id)
            elif class_type not in _ANIMA_SAFE_LORA_NODES and "lora" in class_type.casefold():
                raise WorkflowCompileError(
                    "Anima custom workflow의 활성 model 분기에 호환 remap을 "
                    f"적용할 수 없는 LoRA node가 있습니다: {node_id} ({class_type}). "
                    "LoraLoader 또는 번들 ForgeNeo LoRA 노드(ForgeNeoAnimaLoraLoader·"
                    "ForgeNeoAnimaLoraLoaderModelOnly·ForgeNeoLoraBlockWeight)를 쓰세요."
                )
            if class_type in _CLIP_LORA_NODES:
                lora_ids.append(node_id)
            inputs = node.get("inputs", {})
            link = inputs.get("model") if isinstance(inputs, Mapping) else None

        # A class switch changes every consumer, and a later CLIP rebase
        # rewrites the chain's clip inputs: neither may leak into a branch the
        # sampler does not use.
        for node_id in dict.fromkeys([*remap_ids, *lora_ids]):
            outputs = {0, 1} if graph.nodes[node_id].get("class_type") in _CLIP_LORA_NODES else {0}
            external = [
                item for item in self._direct_link_consumers(
                    graph.nodes, node_id, outputs,
                )
                if item[0] not in active_ids
            ]
            if external:
                detail = ", ".join(
                    f"{consumer}.{name}" for consumer, name in external
                )
                raise WorkflowCompileError(
                    "Anima LoRA node를 선택하지 않은 custom workflow 분기가 "
                    "공유하고 있어 안전하게 교체할 수 없습니다: " + detail
                )
        for node_id in remap_ids:
            node = graph.nodes[node_id]
            node["class_type"] = _ANIMA_LORA_REMAP[str(node.get("class_type"))]

        clip_uses_lora = bool(
            lora_ids
            and pos_clip == [lora_ids[0], 1]
            and neg_clip == [lora_ids[0], 1]
        )
        return lora_ids, clip_uses_lora

    @staticmethod
    def _rebase_custom_lora_clips(
        graph: _Graph, lora_ids: Sequence[str], base_clip: list,
    ) -> list:
        clip = list(base_clip)
        for node_id in reversed(lora_ids):
            graph.nodes[node_id].setdefault("inputs", {})["clip"] = clip
            clip = [node_id, 1]
        return clip

    def _override_custom_modules(
        self,
        graph: _Graph,
        payload: Mapping[str, Any],
        pos_id: str,
        neg_id: str,
        decode_id: Optional[str],
        latent_link: Any,
    ) -> Optional[list]:
        modules = self._module_names(payload)
        if not modules:
            return None
        clips, vaes, unknown = self._classify_modules(modules)
        if unknown or not clips or len(vaes) != 1:
            details = unknown or [f"TE={len(clips)}, VAE={len(vaes)}"]
            raise WorkflowCompileError("custom workflow additional modules 매핑 실패: " + ", ".join(details))
        clip_ref = self._add_clip_loader(graph, clips, payload)
        vae_name = self._resolve_choice("VAELoader", "vae_name", vaes[0])
        vae_node = graph.add("VAELoader", {"vae_name": vae_name}, "VAE override")
        for node_id in (pos_id, neg_id):
            node = graph.nodes[node_id]
            node.setdefault("inputs", {})[self._encode_clip_input(node)] = clip_ref
        vae_ref = [vae_node, 0]
        if decode_id:
            graph.nodes[decode_id].setdefault("inputs", {})["vae"] = vae_ref
        latent_vae_id = self._trace_class(
            graph.nodes,
            latent_link,
            {"VAEEncode", "VAEEncodeForInpaint", "ForgeNeoLatentInput"},
        )
        if latent_vae_id:
            graph.nodes[latent_vae_id].setdefault("inputs", {})["vae"] = vae_ref
        return vae_ref

    def _module_names(self, payload: Mapping[str, Any]) -> list[str]:
        raw = payload.get("forge_additional_modules", [])
        if isinstance(raw, str):
            raw = [item.strip() for item in raw.split(",")]
        if not isinstance(raw, Iterable) or isinstance(raw, (bytes, bytearray, Mapping)):
            return []
        return [
            str(item).strip() for item in raw
            if item and str(item).strip() and str(item).strip().casefold() != "use same choices"
        ]

    def _classify_modules(self, modules: Sequence[str]) -> tuple[list[str], list[str], list[str]]:
        clip_choices = self._choices("CLIPLoader", "clip_name")
        vae_choices = self._choices("VAELoader", "vae_name")
        clips: list[str] = []
        vaes: list[str] = []
        unknown: list[str] = []
        for item in modules:
            clip = self._match_choice(item, clip_choices)
            vae = self._match_choice(item, vae_choices)
            if vae is not None and (clip is None or "vae" in item.casefold()):
                vaes.append(vae)
            elif clip is not None:
                clips.append(clip)
            elif self.object_info is None:
                (vaes if "vae" in item.casefold() else clips).append(item)
            else:
                unknown.append(item)
        return clips, vaes, unknown

    def _hr_module(self, payload: Mapping[str, Any], kind: str) -> str:
        raw = payload.get("hr_additional_modules", [])
        if isinstance(raw, str):
            raw = [item.strip() for item in raw.split(",")]
        items = [str(item).strip() for item in raw if item and str(item).strip()] if isinstance(raw, Sequence) else []
        if not items or any(item.casefold() == "use same choices" for item in items):
            return "Use same choices"
        clips, vaes, unknown = self._classify_modules(items)
        if unknown:
            raise WorkflowCompileError("Hires additional modules 매핑 실패: " + ", ".join(unknown))
        if kind == "vae":
            if len(vaes) > 1:
                raise WorkflowCompileError("Hires VAE는 한 개만 지정할 수 있습니다.")
            return vaes[0] if vaes else "Use same choices"
        if len(clips) > 1:
            raise WorkflowCompileError("ForgeNeoHiresFix text_encoder_name은 한 개만 지원합니다.")
        return clips[0] if clips else "Use same choices"

    def _choices(self, class_type: str, input_name: str) -> Optional[list[str]]:
        if self.object_info is None:
            return None
        node = self.object_info.get(class_type)
        if not isinstance(node, Mapping):
            return []
        input_doc = node.get("input", {})
        for section in ("required", "optional"):
            spec = input_doc.get(section, {}).get(input_name) if isinstance(input_doc, Mapping) else None
            if isinstance(spec, (list, tuple)) and spec and isinstance(spec[0], (list, tuple)):
                return [str(item) for item in spec[0]]
        return []

    @staticmethod
    def _match_choice(requested: Any, choices: Optional[Sequence[str]]) -> Optional[str]:
        if choices is None:
            return str(requested or "").strip() or None
        value = str(requested or "").strip()
        if not value:
            return None
        folded = value.replace("\\", "/").casefold()
        basename = _filename(value).casefold()
        stem = os.path.splitext(basename)[0]
        for choice in choices:
            if str(choice).replace("\\", "/").casefold() == folded:
                return str(choice)
        # A bare filename/stem is a convenience, never permission to choose
        # whichever copy from a shared model directory happens to be first.
        for match in (
            lambda name: "/" in folded and os.path.splitext(name.replace("\\", "/").casefold())[0] == folded,
            lambda name: _filename(name).casefold() == basename,
            lambda name: bool(stem) and os.path.splitext(_filename(name).casefold())[0] == stem,
        ):
            candidates: dict[str, str] = {}
            for choice in choices:
                if match(str(choice)):
                    candidates.setdefault(str(choice).replace("\\", "/").casefold(), str(choice))
            if len(candidates) > 1:
                raise WorkflowCompileError(
                    f"ComfyUI 리소스가 여러 개 일치합니다: {value} → "
                    + ", ".join(candidates.values())
                    + ". 폴더를 포함한 정확한 경로를 지정하세요."
                )
            if candidates:
                return next(iter(candidates.values()))
        return None

    def _resolve_choice(self, class_type: str, input_name: str, requested: Any) -> str:
        value = str(requested or "").strip()
        if not value:
            raise WorkflowCompileError(f"{class_type}.{input_name} 값이 비어 있습니다.")
        choices = self._choices(class_type, input_name)
        matched = self._match_choice(value, choices)
        if matched is None:
            raise WorkflowCompileError(
                f"ComfyUI에서 {class_type}.{input_name} 리소스를 찾을 수 없습니다: {value}"
            )
        return matched

    def _resolve_when_enumerated(
        self, class_type: str, input_name: str, requested: Any,
    ) -> str:
        """Normalize a combo value when the live node publishes its choices.

        Tests and offline callers may provide a class-only capability document;
        in that case there is nothing to resolve and normal class validation is
        still useful.  A live /object_info response always carries the choices.
        """
        value = str(requested or "").strip()
        choices = self._choices(class_type, input_name)
        if not choices:
            return value
        matched = self._match_choice(value, choices)
        if matched is None:
            raise WorkflowCompileError(
                f"ComfyUI에서 {class_type}.{input_name} 값을 지원하지 않습니다: {value}"
            )
        return matched

    def _resolve_adetailer_model(self, requested: Any, index: int) -> str:
        """Map a Forge ADetailer model name to an Impact detector choice.

        Impact Subpack's ``UltralyticsDetectorProvider`` publishes
        ``bbox/<file>`` and ``segm/<file>``.  A Forge name such as
        ``person_yolov8n-seg.pt`` must become the ``segm/`` choice when it
        exists (Forge inpaints the silhouette, not the box); a model ComfyUI
        does not have (e.g. ``mediapipe_face_full``) is rejected here, before
        the base image is sampled, instead of inside the ADetailer node.
        """
        value = str(requested or "").strip().replace("\\", "/")
        if not value or value.casefold() in {"none", "disabled"}:
            raise WorkflowCompileError(f"ADetailer 슬롯 {index}에 검출 모델이 없습니다.")
        if self.object_info is None:
            return value  # offline compile: the node keeps its bbox/ default
        missing = [
            name for name in ("UltralyticsDetectorProvider", "FaceDetailer")
            if name not in self.object_info
        ]
        if missing:
            raise WorkflowCompileError(
                "ComfyUI ADetailer에는 Impact Pack(FaceDetailer)과 Impact Subpack"
                "(UltralyticsDetectorProvider)이 필요합니다. 없는 노드: "
                + ", ".join(missing)
            )
        choices = self._choices("UltralyticsDetectorProvider", "model_name")
        if not choices:
            return value  # class-only capability probe publishes no files
        by_file: dict[str, list[str]] = {}
        for choice in choices:
            kind, separator, rest = str(choice).replace("\\", "/").partition("/")
            if separator and kind.casefold() in {"bbox", "segm"} and rest:
                by_file.setdefault(rest, []).append(str(choice))
        prefix, separator, _rest = value.partition("/")
        if separator and prefix.casefold() in {"bbox", "segm"}:
            # An explicit bbox/ or segm/ is a deliberate detector kind: match
            # the whole path only (no basename fallback across kinds).
            matched = next((
                str(choice) for choice in choices
                if str(choice).replace("\\", "/").casefold() == value.casefold()
            ), None)
            if matched is None:
                raise WorkflowCompileError(
                    f"ADetailer 슬롯 {index}의 검출 모델을 ComfyUI에서 찾을 수 없습니다: {value}"
                )
            return matched
        matched_file = self._match_choice(value, list(by_file))
        if matched_file is None:
            raise WorkflowCompileError(
                f"ADetailer 슬롯 {index}의 검출 모델을 ComfyUI에서 찾을 수 없습니다: {value}. "
                "ComfyUI ADetailer는 models/ultralytics/bbox·segm의 YOLO 모델만 지원합니다"
                "(mediapipe 모델은 Forge 전용)."
            )
        candidates = by_file[matched_file]
        segm = [choice for choice in candidates if choice.casefold().startswith("segm/")]
        return (segm or candidates)[0]

    @staticmethod
    def _script(payload: Mapping[str, Any], wanted: str) -> Optional[Mapping[str, Any]]:
        scripts = payload.get("alwayson_scripts", {})
        if not isinstance(scripts, Mapping):
            return None
        folded = wanted.casefold()
        for name, block in scripts.items():
            if str(name).casefold() == folded and isinstance(block, Mapping):
                return block
        return None

    @staticmethod
    def _script_settings(block: Mapping[str, Any], spec: Sequence[tuple]) -> dict[str, Any]:
        args = block.get("args", []) if isinstance(block, Mapping) else []
        if isinstance(args, Mapping):
            return dict(args)
        values = list(args) if isinstance(args, (list, tuple)) else []
        return {
            item[0]: values[index] if index < len(values) else item[2]
            for index, item in enumerate(spec)
        }

    @staticmethod
    def _fit_mode(resize_mode: Any) -> str:
        raw = str(resize_mode if resize_mode is not None else "crop").strip().casefold()
        return {
            "0": "stretch", "just resize": "stretch", "stretch": "stretch",
            "1": "crop", "crop and resize": "crop", "crop": "crop",
            "2": "contain", "resize and fill": "contain", "contain": "contain",
        }.get(raw, "crop")

    @staticmethod
    def _split_sampler_label(value: Any) -> tuple[str, Optional[str]]:
        """Folded Forge sampler label and the scheduler its suffix implies."""
        folded = re.sub(r"\s+", " ", str(value or "euler").strip()).casefold()
        for suffix, scheduler in _FORGE_SAMPLER_SCHEDULER_SUFFIXES:
            if folded.endswith(suffix):
                return folded[: -len(suffix)].strip(), scheduler
        return folded, None

    @staticmethod
    def _comfy_sampler(value: Any) -> str:
        """Forge/A1111 sampler label → ComfyUI ``KSampler.sampler_name``.

        The bundled ADetailer node keeps an identical table
        (``generation._AD_SAMPLER_ALIASES``); a golden test pins both.
        """
        raw = str(value or "euler").strip()
        folded, _scheduler = ComfyWorkflowCompiler._split_sampler_label(raw)
        if folded in _FORGE_SAMPLER_ALIASES:
            return _FORGE_SAMPLER_ALIASES[folded]
        # Unknown multi-word labels follow Comfy's snake_case naming
        # ("Res Multistep" → "res_multistep"); native names pass through.
        return folded.replace(" ", "_") if " " in folded else raw

    @staticmethod
    def _comfy_scheduler(value: Any, *, sampler_text: str = "") -> str:
        _folded, implied = ComfyWorkflowCompiler._split_sampler_label(sampler_text)
        if implied:
            return implied
        raw = str(value or "normal").strip()
        folded = re.sub(r"\s+", " ", raw).casefold()
        if folded in _FORGE_SAME_SCHEDULER:
            return "normal"
        if folded in _FORGE_SCHEDULER_ALIASES:
            return _FORGE_SCHEDULER_ALIASES[folded]
        return folded.replace(" ", "_") if " " in folded else raw

    def _runtime_sampler_values(self, sampler: Any, scheduler: Any) -> tuple[str, str]:
        original_sampler = str(sampler or "euler")
        sampler_value = self._comfy_sampler(original_sampler)
        scheduler_value = self._comfy_scheduler(scheduler, sampler_text=original_sampler)
        for input_name, value in (
            ("sampler_name", sampler_value), ("scheduler", scheduler_value),
        ):
            choices = self._choices("KSampler", input_name)
            if choices:
                matched = self._match_choice(value, choices)
                if matched is None:
                    raise WorkflowCompileError(
                        f"ComfyUI KSampler.{input_name}에서 지원하지 않는 값입니다: {value}"
                    )
                if input_name == "sampler_name":
                    sampler_value = matched
                else:
                    scheduler_value = matched
        return sampler_value, scheduler_value

    @staticmethod
    def _find_sampler(workflow: Mapping[str, Any]) -> str:
        candidates = [
            str(node_id) for node_id, node in workflow.items()
            if isinstance(node, Mapping) and node.get("class_type") in _SAMPLERS
        ]
        if not candidates:
            raise WorkflowCompileError("custom workflow에서 sampler 노드를 찾지 못했습니다.")
        if len(candidates) != 1:
            raise WorkflowCompileError(
                "custom workflow에 sampler 노드가 여러 개이므로 자동 삽입 대상이 불명확합니다: "
                + ", ".join(candidates)
            )
        return candidates[0]

    @staticmethod
    def _trace_classes(
        workflow: Mapping[str, Any], link: Any, wanted: set[str],
    ) -> list[str]:
        matches: list[str] = []
        visited: set[str] = set()

        def visit(value: Any, depth: int) -> None:
            if depth > 30 or not _is_link(value):
                return
            node_id = str(value[0])
            if node_id in visited:
                return
            node = workflow.get(node_id)
            if not isinstance(node, Mapping):
                return
            visited.add(node_id)
            if node.get("class_type") in wanted:
                matches.append(node_id)
                return
            for upstream in ComfyWorkflowCompiler._node_inputs(node).values():
                visit(upstream, depth + 1)

        visit(link, 0)
        return matches

    @staticmethod
    def _trace_class(workflow: Mapping[str, Any], link: Any, wanted: set[str], depth: int = 0) -> Optional[str]:
        if depth:
            return None
        matches = ComfyWorkflowCompiler._trace_classes(workflow, link, wanted)
        return matches[0] if matches else None

    @staticmethod
    def _trace_model_loader(workflow: Mapping[str, Any], link: Any) -> Optional[str]:
        visited: set[str] = set()
        for _depth in range(31):
            if not _is_link(link):
                return None
            node_id = str(link[0])
            if node_id in visited:
                return None
            visited.add(node_id)
            node = workflow.get(node_id)
            if not isinstance(node, Mapping):
                return None
            if node.get("class_type") in MODEL_LOADER_INPUTS:
                return node_id
            inputs = node.get("inputs", {})
            link = inputs.get("model") if isinstance(inputs, Mapping) else None
        return None

    @staticmethod
    def _find_vae_link_for_branch(
        workflow: Mapping[str, Any], latent_link: Any, decode_id: Optional[str],
    ) -> Optional[list]:
        if decode_id:
            decode = workflow.get(str(decode_id))
            if isinstance(decode, Mapping):
                value = decode.get("inputs", {}).get("vae")
                if _is_link(value):
                    return list(value)
        latent_nodes = ComfyWorkflowCompiler._trace_classes(
            workflow,
            latent_link,
            {"VAEEncode", "VAEEncodeForInpaint", "ForgeNeoLatentInput"},
        )
        vae_links = []
        for node_id in latent_nodes:
            value = workflow[node_id].get("inputs", {}).get("vae")
            if _is_link(value) and tuple(value[:2]) not in {
                tuple(item[:2]) for item in vae_links
            }:
                vae_links.append(list(value))
        if len(vae_links) > 1:
            raise WorkflowCompileError(
                "custom workflow의 선택 sampler latent 분기에 VAE 연결이 여러 개입니다."
            )
        return vae_links[0] if vae_links else None

    @staticmethod
    def _infer_vae_from_model(
        workflow: Mapping[str, Any], model_link: Any, depth: int = 0,
    ) -> Optional[list]:
        if depth > 30 or not _is_link(model_link):
            return None
        node_id = str(model_link[0])
        node = workflow.get(node_id)
        if not isinstance(node, Mapping):
            return None
        if node.get("class_type") in CHECKPOINT_LOADER_NODES:
            return [node_id, 2]
        if node.get("class_type") == "VAELoader":
            return [node_id, 0]
        upstream = node.get("inputs", {}).get("model")
        return ComfyWorkflowCompiler._infer_vae_from_model(workflow, upstream, depth + 1)

    @staticmethod
    def _find_decode_after(workflow: Mapping[str, Any], sampler_id: str) -> Optional[str]:
        decoders = [
            str(node_id)
            for node_id, node in workflow.items()
            if isinstance(node, Mapping)
            and node.get("class_type") == "VAEDecode"
            and ComfyWorkflowCompiler._link_depends_on(
                workflow, node.get("inputs", {}).get("samples"), sampler_id,
            )
        ]
        if len(decoders) > 1:
            raise WorkflowCompileError(
                "custom workflow의 sampler 출력에 연결된 VAEDecode가 여러 개입니다: "
                + ", ".join(decoders)
            )
        return decoders[0] if decoders else None

    @staticmethod
    def _link_depends_on(
        workflow: Mapping[str, Any], link: Any, ancestor_id: str,
        visited: Optional[set[str]] = None,
    ) -> bool:
        if not _is_link(link):
            return False
        node_id = str(link[0])
        if node_id == str(ancestor_id):
            return True
        node = workflow.get(node_id)
        if not isinstance(node, Mapping):
            return False
        seen = set() if visited is None else visited
        if node_id in seen:
            return False
        seen.add(node_id)
        return any(
            ComfyWorkflowCompiler._link_depends_on(
                workflow, upstream, ancestor_id, seen,
            )
            for upstream in ComfyWorkflowCompiler._node_inputs(node).values()
        )

    @staticmethod
    def _find_outputs_after(
        workflow: Mapping[str, Any], ancestor_id: str,
    ) -> list[tuple[str, str, list]]:
        outputs: list[tuple[str, str, list]] = []
        for node_id, node in workflow.items():
            if not isinstance(node, Mapping) or node.get("class_type") not in _SAVE_NODES:
                continue
            inputs = node.get("inputs", {})
            if not isinstance(inputs, Mapping):
                continue
            key = "images" if "images" in inputs else "image" if "image" in inputs else ""
            link = inputs.get(key) if key else None
            if _is_link(link) and ComfyWorkflowCompiler._link_depends_on(
                workflow, link, ancestor_id,
            ):
                outputs.append((str(node_id), key, list(link)))
        return outputs

    @staticmethod
    def _has_image_scripts(payload: Mapping[str, Any]) -> bool:
        return bool(
            ComfyWorkflowCompiler._adetailer_slots(payload)
            or ComfyWorkflowCompiler._sam3_state(payload) is not None
        )

    @staticmethod
    def _adetailer_slots(payload: Mapping[str, Any]) -> list[Mapping[str, Any]]:
        block = ComfyWorkflowCompiler._script(payload, "ADetailer")
        args = block.get("args", []) if isinstance(block, Mapping) else []
        if not isinstance(args, (list, tuple)) or not args or not _bool(args[0], True):
            return []
        # ADetailer skips a tab that is disabled or whose model is "None"
        # (ADetailerArgs.need_skip); such a slot is not an error.
        return [
            item for item in list(args)[2:]
            if isinstance(item, Mapping) and _bool(item.get("ad_tab_enable"), True)
            and str(item.get("ad_model") or "None").strip().casefold() != "none"
        ]

    @staticmethod
    def _sam3_state(payload: Mapping[str, Any]) -> Optional[Mapping[str, Any]]:
        block = ComfyWorkflowCompiler._script(payload, "SAM3 Mask")
        args = block.get("args", []) if isinstance(block, Mapping) else []
        if not isinstance(args, (list, tuple)) or not args or not isinstance(args[0], Mapping):
            return None
        state = args[0]
        enabled = state.get("sam3_enable", state.get("enabled", True))
        return state if _bool(enabled, True) else None

    def _sam3_cache_model(self, state: Mapping[str, Any]) -> bool:
        """``ForgeNeoSAM3Mask.cache_model`` with Forge's bundle-keeping rules.

        Forge keeps its SAM3 ``_BUNDLE`` on the device when "Unload after" is
        off, and after an unload keeps a CPU RAM copy only while its
        ``sam3_unload_keep_in_ram`` setting is on.  The app setting plays that
        role here; ComfyUI's unload-all-models (/free) frees either copy.
        """
        return (not _bool(state.get("sam3_unload_after"), True)) or self.sam3_keep_in_ram

    @staticmethod
    def _sam3_processing_size(payload: Mapping[str, Any]) -> tuple[int, int]:
        """Forge ``p.width/p.height`` for the SAM3 "only masked" pass.

        In-flight SAM3 runs inside the generation whose width/height is the
        base resolution before Hires.fix.  Standalone post-processing (SAM3,
        Refine) is a Forge img2img whose width/height is the input image size,
        so its payload size is the image size; the backend also pins it as
        ``_sam3_processing_width/height`` so a preserved generation payload
        cannot leak its size.  Explicit ``_sam3_processing_*`` keys win over
        the payload size.  ``(0, 0)`` asks the node to sample at crop size.
        """
        for width_key, height_key in (
            ("_sam3_processing_width", "_sam3_processing_height"),
            ("width", "height"),
        ):
            width = _int(payload.get(width_key), 0)
            height = _int(payload.get(height_key), 0)
            if width > 0 and height > 0:
                # The node accepts 64..8192 (or 0x0); keep the aspect intent.
                return (
                    min(8192, max(64, width)),
                    min(8192, max(64, height)),
                )
        return 0, 0

    @staticmethod
    def _validate_adetailer_slot(slot: Mapping[str, Any], index: int) -> None:
        unsupported: list[str] = []
        for key, label in (
            ("ad_hires_fix_only", "Hires-only"),
            ("ad_use_autotag", "autotag"),
            ("ad_copy_main_lora_triggers", "LoRA trigger copy"),
            ("ad_copy_main_lora_triggers_only", "LoRA trigger-only copy"),
            ("ad_use_checkpoint", "separate checkpoint"),
            ("ad_use_vae", "separate VAE"),
            ("ad_use_noise_multiplier", "noise multiplier"),
            ("ad_use_clip_skip", "CLIP skip"),
            ("ad_restore_face", "restore face"),
        ):
            if _bool(slot.get(key)):
                unsupported.append(label)
        if str(slot.get("ad_model_classes") or "").strip():
            unsupported.append("model class filter")
        if str(slot.get("ad_mask_filter_method") or "Area") != "Area":
            unsupported.append("mask filter")
        if _int(slot.get("ad_mask_k"), 0) != 0:
            unsupported.append("mask K")
        if _float(slot.get("ad_mask_min_ratio"), 0.0) != 0.0:
            unsupported.append("minimum mask ratio")
        if _float(slot.get("ad_mask_max_ratio"), 1.0) != 1.0:
            unsupported.append("maximum mask ratio")
        if _int(slot.get("ad_x_offset"), 0) != 0 or _int(slot.get("ad_y_offset"), 0) != 0:
            unsupported.append("mask offset")
        if str(slot.get("ad_mask_merge_invert") or "None") != "None":
            unsupported.append("mask merge/invert")
        if _float(slot.get("ad_inpaint_scale"), 1.0) != 1.0:
            unsupported.append("inpaint scale")
        control_model = str(slot.get("ad_controlnet_model") or "None").strip().casefold()
        control_module = str(slot.get("ad_controlnet_module") or "None").strip().casefold()
        if control_model not in {"", "none"} or control_module not in {"", "none"}:
            unsupported.append("ControlNet")
        if unsupported:
            raise WorkflowCompileError(
                f"ADetailer 슬롯 {index}의 설정은 현재 ComfyUI 노드로 표현할 수 없습니다: "
                + ", ".join(unsupported)
            )

    @staticmethod
    def _validate_anima_guidance_settings(settings: Mapping[str, Any]) -> None:
        if _bool(settings.get("guid_enabled")):
            method = str(settings.get("guid_attn_method") or "PAG").strip().casefold()
            if method not in {"pag", "seg", "none", "off", ""}:
                raise WorkflowCompileError(f"지원하지 않는 Anima attention 방식입니다: {method}")
            # PAG 는 팩이 원본 노드(comfyui-anima-safe-pag)를 그대로 불러 legacy(=legacy
            # strength 로 같은 공식)와 헤드 지정을 받는다. SEG 는 원본이 없고 팩 구현이
            # 둘 다 못 하므로 여기서 막는다.
            if method == "seg" and _bool(settings.get("guid_legacy_attn")):
                raise WorkflowCompileError(
                    "Anima legacy SEG는 ComfyUI pre-projection hook으로 표현할 수 없습니다."
                )
            if method == "seg" and str(settings.get("guid_head_indices") or "").strip():
                raise WorkflowCompileError(
                    "Anima head-selective SEG는 ComfyUI에서 지원되지 않습니다."
                )
        cfg_mode = str(
            settings.get("guid_cfg_mode") or "Preserve incoming"
        ).strip().casefold()
        if cfg_mode not in {
            "preserve incoming", "preserve", "apg", "cwm", "smc",
            "smc + cwm", "smc+cwm",
        }:
            raise WorkflowCompileError(f"지원하지 않는 Anima CFG 방식입니다: {cfg_mode}")
        if _bool(settings.get("guid_mod_enabled")):
            adapter_mode = str(
                settings.get("guid_mod_adapter_mode") or "Auto-download official"
            ).strip().casefold()
            if adapter_mode == "local file" and not str(
                settings.get("guid_mod_adapter_path") or ""
            ).strip():
                raise WorkflowCompileError(
                    "Anima modulation Local file 모드에는 adapter path가 필요합니다."
                )
            if adapter_mode not in {"local file", "auto-download official"}:
                raise WorkflowCompileError(
                    f"지원하지 않는 Anima modulation adapter 방식입니다: {adapter_mode}"
                )


__all__ = [
    "ComfyWorkflowCompiler", "LoraSpec", "WorkflowCompileError", "parse_lora_tags",
]
