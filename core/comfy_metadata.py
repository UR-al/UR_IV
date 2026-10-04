"""Read-only, bounded Comfy graph interpretation. Never executes graph nodes.

Only understood connections become reusable prompt text. Ambiguous branches
remain separate candidates; the original JSON belongs to image_metadata.
"""
from __future__ import annotations

import json
from typing import Any

from core import anima_speed
from core import colorcraft
from core import extra_samplers
from core import extra_schedulers
from core import vae_degrid
from core.comfy_node_classes import SAMPLER_NODES


MAX_NODES = 4096
MAX_DEPTH = 96
_SAMPLERS = SAMPLER_NODES  # 선택 화면·컴파일러와 같은 샘플러 분류
_OUTPUTS = {"SaveImage", "PreviewImage", "SaveAnimatedWEBP", "SaveVideo", "SaveAnimatedPNG"}
_WIDGETS = {
    "CLIPTextEncode": ("text",),
    "CLIPTextEncodeSDXL": ("width", "height", "crop_w", "crop_h", "target_width", "target_height", "text_g", "text_l"),
    "CLIPTextEncodeSDXLRefiner": ("ascore", "width", "height", "text"),
    "CLIPTextEncodeFlux": ("clip_l", "t5xxl", "guidance"),
    "KSampler": ("seed", "_control", "steps", "cfg", "sampler_name", "scheduler", "denoise"),
    "KSamplerAdvanced": ("add_noise", "noise_seed", "_control", "steps", "cfg", "sampler_name", "scheduler", "start_at_step", "end_at_step", "return_with_leftover_noise"),
    "CheckpointLoaderSimple": ("ckpt_name",), "UNETLoader": ("unet_name", "weight_dtype"),
    "VAELoader": ("vae_name",), "EmptyLatentImage": ("width", "height", "batch_size"),
    "EmptySD3LatentImage": ("width", "height", "batch_size"),
    "EmptyMiniMaxH3LatentAV": ("width", "height", "length"),
    "CFGGuider": ("cfg",), "RandomNoise": ("noise_seed", "_control"),
    "KSamplerSelect": ("sampler_name",), "BasicScheduler": ("scheduler", "steps", "denoise"),
    "FluxGuidance": ("guidance",), "PrimitiveString": ("value",),
    "PrimitiveStringMultiline": ("value",), "PrimitiveNode": ("value",),
    # 앱 노드 팩 VAE DeGrid — image 는 연결, 나머지는 위젯(입력 계약 순서 core/vae_degrid.COMFY_INPUTS)
    vae_degrid.COMFY_NODE_CLASS: tuple(name for name in vae_degrid.COMFY_INPUTS if name != "image"),
    # 앱 노드 팩 Extra Schedulers — model 은 연결, 값 여덟은 위젯, 마지막은 정보용 선택 입력 owned_names(읽지 않는다 — '_')
    extra_schedulers.COMFY_NODE_CLASS: (*(name for name in extra_schedulers.COMFY_INPUTS if name != "model"),
                                        "_" + extra_schedulers.COMFY_REPORT_INPUT),
    # 앱 노드 팩 Extra Samplers 값 — model 은 연결, 값 다섯은 위젯, 마지막은 정보용 선택 입력 owned_names(읽지 않는다)
    extra_samplers.COMFY_SETTINGS_NODE: (*(name for name in extra_samplers.COMFY_INPUTS if name != "model"),
                                         "_" + extra_samplers.COMFY_REPORT_INPUT),
    # 앱 노드 팩 Anima SPEED — model 은 연결, 나머지는 위젯. seed 뒤에는 ComfyUI 프런트가 붙이는 control_after_generate
    # 위젯이 있다(KSampler 의 seed 와 같은 이름 규칙 — '_control', 읽지 않는다)
    anima_speed.COMFY_NODE_CLASS: tuple(
        item for name in anima_speed.COMFY_INPUTS if name != "model"
        for item in ((name, "_control") if name == "seed" else (name,))
    ),
}
# 앱 노드 팩 Colorcraft — model·vae 는 연결, 나머지 셋은 위젯(core/colorcraft.COMFY_INPUTS 순서)
_WIDGETS[colorcraft.COMFY_NODE_CLASS] = tuple(name for name in colorcraft.COMFY_INPUTS if name not in ("model", "vae"))
# Anima SPEED 노드의 글자 입력(선택지·텍스트) — Primitive 연결도 따라가 읽는다
_SPEED_TEXT_INPUTS = frozenset({"mode", "threshold", "preset", "scales", "manual_sigmas", "transform"})
# 모델 시그마(model_sampling)를 새로 거는 노드 — 그 위의 Extra Schedulers 값은 그 아래 샘플러에 닿지 않는다
# (core/comfy_workflow_compiler._reaches_extra_schedulers 와 같은 규칙)
_MODEL_SAMPLING_PREFIX = "ModelSampling"
_FLOW_SHIFT_NODE = "ForgeNeoModelSamplingShift"


def _workflow_graph(workflow):
    """Convert only known positional widget layouts; named links are authoritative."""
    if not isinstance(workflow, dict) or not isinstance(workflow.get("nodes"), list):
        return {}
    nodes = workflow["nodes"]
    if len(nodes) > MAX_NODES:
        return {}
    links = {}
    for link in workflow.get("links", []) if isinstance(workflow.get("links", []), list) else []:
        if isinstance(link, list) and len(link) >= 6:
            links[str(link[0])] = link
    graph = {}
    for node in nodes:
        if not isinstance(node, dict) or "id" not in node or not isinstance(node.get("type"), str):
            continue
        kind = node["type"]
        inputs = {}
        # Muted/bypassed graphs need runtime semantics not encoded by widgets.
        if node.get("mode", 0) != 0:
            kind = "UnresolvedWorkflowMode:" + kind
        widgets = node.get("widgets_values", [])
        names = _WIDGETS.get(kind, ())
        if isinstance(widgets, dict):
            inputs.update({name: widgets[name] for name in names if name in widgets and not name.startswith("_")})
        elif isinstance(widgets, list) and len(widgets) == len(names):
            inputs.update({name: value for name, value in zip(names, widgets) if not name.startswith("_")})
        for index, field in enumerate(node.get("inputs", []) if isinstance(node.get("inputs", []), list) else []):
            if not isinstance(field, dict) or not isinstance(field.get("name"), str):
                continue
            link = links.get(str(field.get("link")))
            if link and str(link[3]) == str(node["id"]) and link[4] == index:
                inputs[field["name"]] = [str(link[1]), link[2]]
        graph[str(node["id"])] = {"class_type": kind, "inputs": inputs}
    return graph


def _link(value):
    if (isinstance(value, (list, tuple)) and len(value) == 2
            and isinstance(value[0], (str, int)) and isinstance(value[1], int)
            and not isinstance(value[1], bool) and value[1] >= 0):
        return str(value[0]), value[1]
    return None


class _Reader:
    def __init__(self, graph):
        self.graph = graph
        self.warnings = []
        self.visits = 0

    def warn(self, text):
        if text not in self.warnings and len(self.warnings) < 50:
            self.warnings.append(text)

    def ancestors(self, ids):
        seen, pending = set(), list(ids)
        while pending and len(seen) <= MAX_NODES:
            key = pending.pop()
            if key in seen or key not in self.graph:
                continue
            seen.add(key)
            pending.extend(link[0] for value in self.graph[key]["inputs"].values() if (link := _link(value)))
        return seen

    def conditioning(self, value, seen=()):
        self.visits += 1
        if self.visits > MAX_NODES * 4:
            self.warn("프롬프트 분기가 너무 많아 일부 자동 해석을 중단했습니다.")
            return [], False
        link = _link(value)
        if not link or link[0] not in self.graph:
            self.warn("프롬프트 연결이 없거나 잘못되어 텍스트를 추정하지 않았습니다.")
            return [], False
        key, output = link
        if key in seen or len(seen) >= MAX_DEPTH:
            self.warn("순환하거나 너무 깊은 프롬프트 연결을 건너뛰었습니다.")
            return [], False
        node = self.graph[key]
        kind, inputs = node["class_type"], node["inputs"]
        if kind == "ConditioningZeroOut" and output == 0:
            return [], True
        fields = {"CLIPTextEncode": [("text", "CLIP")],
                  "CLIPTextEncodeSDXL": [("text_g", "CLIP-G"), ("text_l", "CLIP-L")],
                  "CLIPTextEncodeSDXLRefiner": [("text", "Refiner")],
                  "ForgeNeoAnima38V2Prompt": [("prompt", "Anima semantic")],
                  "ForgeNeoAnimaQwen35Prompt": [("prompt", "Anima semantic")],
                  "MiniMaxH3ImageToVideo": [("prompt", "H3")],
                  "MiniMaxH3ReferenceToVideo": [("prompt", "H3")],
                  "TextEncodeQwenImageEdit": [("prompt", "Qwen Edit")],
                  "TextEncodeQwenImageEditPlus": [("prompt", "Qwen Edit")],
                  "CLIPTextEncodeFlux": [("clip_l", "CLIP-L"), ("t5xxl", "T5")]} .get(kind)
        if fields and (output == 0 or kind in {"ForgeNeoAnima38V2Prompt", "ForgeNeoAnimaQwen35Prompt"} and output == 1):
            parts = []
            for field, label in fields:
                text = self.text(inputs.get(field), (*seen, key))
                if text is None:
                    self.warn(f"{kind}.{field}의 동적 텍스트 연결을 해석하지 못했습니다.")
                    return parts, False
                parts.append({"node_id": key, "label": label, "text": text})
            known = len({part["text"] for part in parts}) <= 1
            if not known:
                self.warn(f"{kind}의 인코더별 프롬프트가 달라 하나의 텍스트로 합치지 않았습니다.")
            return parts, known
        if kind in {"ConditioningSetArea", "ConditioningSetAreaPercentage", "ConditioningSetMask",
                    "ConditioningSetTimestepRange", "FluxGuidance", "ControlNetApply"} and output == 0:
            return self.conditioning(inputs.get("conditioning"), (*seen, key))
        if kind == "ControlNetApplyAdvanced" and output in (0, 1):
            return self.conditioning(inputs.get("positive" if output == 0 else "negative"), (*seen, key))
        if kind in {"ConditioningCombine", "ConditioningConcat", "ConditioningAverage"} and output == 0:
            fields = {"ConditioningCombine": ("conditioning_1", "conditioning_2"),
                      "ConditioningConcat": ("conditioning_to", "conditioning_from"),
                      "ConditioningAverage": ("conditioning_to", "conditioning_from")}[kind]
            parts, known = [], True
            for field in fields:
                branch, resolved = self.conditioning(inputs.get(field), (*seen, key))
                parts.extend(branch)
                known = known and resolved
            known = known and len({part["text"] for part in parts}) <= 1
            if not known:
                self.warn(f"{kind}의 여러 프롬프트 분기를 별도로 표시합니다.")
            return parts, known
        if kind == "Reroute" and len(inputs) == 1 and output == 0:
            return self.conditioning(next(iter(inputs.values())), (*seen, key))
        self.warn(f"{node['class_type']} 노드의 프롬프트는 자동 해석하지 않았습니다.")
        return [], False

    def text(self, value, seen=()):
        if isinstance(value, str):
            return value
        link = _link(value)
        if not link or link[0] not in self.graph or link[0] in seen or len(seen) >= MAX_DEPTH:
            return None
        key, output = link
        node = self.graph[key]
        if output != 0:
            return None
        field = {"PrimitiveString": "value", "PrimitiveStringMultiline": "value",
                 "StringConstant": "string", "PrimitiveNode": "value"}.get(node["class_type"])
        if field:
            return self.text(node["inputs"].get(field), (*seen, key))
        if node["class_type"] == "Reroute" and len(node["inputs"]) == 1:
            return self.text(next(iter(node["inputs"].values())), (*seen, key))
        return None

    def parameters(self, node):
        inputs = node["inputs"]
        names = {"seed": "Seed", "noise_seed": "Seed", "steps": "Steps", "cfg": "CFG scale",
                 "sampler_name": "Sampler", "scheduler": "Schedule type", "denoise": "Denoising strength"}
        result = {label: inputs[key] for key, label in names.items()
                  if isinstance(inputs.get(key), (str, int, float)) and not isinstance(inputs.get(key), bool)}
        model_link = _link(inputs.get("model"))
        if model_link:
            models = set()
            for key in self.ancestors([model_link[0]]):
                model = self.graph[key]
                field = {"CheckpointLoaderSimple": "ckpt_name", "CheckpointLoader": "ckpt_name",
                         "UNETLoader": "unet_name", "ForgeNeoAnima38V2Loader": "model_name"}.get(model["class_type"])
                if field and isinstance(model["inputs"].get(field), str):
                    models.add(model["inputs"][field])
            if len(models) == 1:
                result["Model"] = next(iter(models))
        latent = _link(inputs.get("latent_image"))
        if latent and latent[0] in self.graph:
            values = self.graph[latent[0]]["inputs"]
            width, height = values.get("width"), values.get("height")
            if isinstance(width, int) and isinstance(height, int) and width > 0 and height > 0:
                result["Size"] = f"{width}x{height}"
        return result

    def sampler_inputs(self, node):
        inputs = dict(node["inputs"])
        if node["class_type"] != "SamplerCustomAdvanced":
            return inputs
        link = _link(inputs.get("guider"))
        guider = self.graph.get(link[0], {}) if link and link[1] == 0 else {}
        if guider.get("class_type") in {"CFGGuider", "BasicGuider"}:
            inputs.update(guider["inputs"])
            if guider["class_type"] == "BasicGuider":
                inputs["positive"] = inputs.get("conditioning")
                inputs["_no_negative"] = True
        else:
            self.warn("지원하지 않는 Guider의 positive/negative 역할은 추정하지 않았습니다.")
        for field, kind, fields in (("noise", "RandomNoise", ("noise_seed",)),
                                    ("sampler", "KSamplerSelect", ("sampler_name",)),
                                    ("sigmas", "BasicScheduler", ("steps", "scheduler", "denoise"))):
            link = _link(inputs.get(field))
            dependency = self.graph.get(link[0], {}) if link and link[1] == 0 else {}
            if dependency.get("class_type") == kind:
                inputs.update({name: dependency["inputs"][name] for name in fields if name in dependency["inputs"]})
        return inputs


def _degrid_parameters(reader: _Reader, active: set) -> dict:
    """그래프에 켜진 VAE DeGrid 노드(앱 노드 팩)가 출력에 닿으면 Forge infotext 와 같은 키로 그 **요청** 값.

    Forge 는 실제로 쓴 값을 남기고 실패하면 'Anima DeGrid error' 만 남기지만, ComfyUI PNG 에는 큐에 넣은 그래프만
    있다 — 노드가 건너뛴 이미지도 요청 값이 보인다(레지스트리 gap). 모델 'auto' 는 그대로 둔다(붙여 넣으면 자동).
    노드가 여럿이고 값이 다르면 고르지 않는다."""
    found = []
    for key in sorted(active):
        node = reader.graph.get(key) or {}
        if node.get("class_type") != vae_degrid.COMFY_NODE_CLASS:
            continue
        inputs = node["inputs"]
        if inputs.get("enabled") is not True or not _link(inputs.get("image")):
            continue
        model = str(inputs.get("model_name") or "").strip()
        params = {
            vae_degrid.KEY_MODEL: (vae_degrid.COMFY_AUTO if vae_degrid.normalize_model(model) == vae_degrid.AUTO
                                   else vae_degrid.comfy_stem(model)),
            vae_degrid.KEY_MODE: vae_degrid.MODE_LABELS[vae_degrid.coerce_mode(inputs.get("mode"))],
            vae_degrid.KEY_STRENGTH: vae_degrid.format_strength(vae_degrid.coerce_strength(inputs.get("strength"))),
            vae_degrid.KEY_TILE: vae_degrid.coerce_tile(inputs.get("tile")),
        }
        if params not in found:
            found.append(params)
    if len(found) > 1:
        reader.warn("VAE DeGrid 노드가 여럿이고 설정이 달라 DeGrid 값을 고르지 않았습니다.")
        return {}
    return found[0] if found else {}


def _extra_schedulers_carrier(reader: _Reader, link) -> str | None:
    """``model`` 사슬을 거슬러 처음 만나는 Extra Schedulers 노드 id — 그 전에 model_sampling 을 새로 거는 노드가 있거나
    없으면 None."""
    seen = set()
    while (found := _link(link)) and found[0] in reader.graph and found[0] not in seen and len(seen) < MAX_DEPTH:
        key = found[0]
        seen.add(key)
        node = reader.graph[key]
        kind = node["class_type"]
        if kind == extra_schedulers.COMFY_NODE_CLASS:
            return key
        if kind.startswith(_MODEL_SAMPLING_PREFIX) or kind == _FLOW_SHIFT_NODE:
            return None
        link = node["inputs"].get("model")
    return None


def _extra_schedulers_parameters(reader: _Reader, active: set) -> dict:
    """메인·Hires 패스가 값 스케줄러(laplace·custom·Flow Cosmos rho7·Flow Cosmos Dynamic)로 샘플링했고 그 model 사슬에
    앱 노드 팩 Extra Schedulers 노드가 닿으면 Forge infotext 와 같은 키(core/extra_schedulers.infotext_items — 확장
    infotext_items 의 거울): custom 은 쓴 모드의 글자, Laplace 는 μ·β, Flow Cosmos 둘 중 하나면 ρ·σ̃ max·σ̃ min 셋 다.

    Forge 처럼 메인·Hires 의 스케줄러만 본다(ADetailer·SAM3 패스는 그 이미지의 infotext 에 적지 않는다). 노드 값은
    확장 규칙으로 정규화한 요청 값이다(노드도 같은 규칙으로 정규화한다). 값이 연결로 들어오거나 노드가 여럿이고 값이
    다르면 고르지 않는다."""
    used: dict = {}
    for key in sorted(active):
        node = reader.graph.get(key) or {}
        kind = node.get("class_type")
        if kind in _SAMPLERS:
            inputs = reader.sampler_inputs(node)
            texts = [inputs.get("scheduler")]
        elif kind == "ForgeNeoHiresFix":
            inputs = node["inputs"]
            texts = [inputs.get("scheduler"), inputs.get("base_scheduler")]
        else:
            continue
        names = {extra_schedulers.resolve_scheduler(text) for text in texts if isinstance(text, str)}
        names &= set(extra_schedulers.VALUE_SCHEDULERS)
        carrier = _extra_schedulers_carrier(reader, inputs.get("model")) if names else None
        if carrier is not None:
            used.setdefault(carrier, set()).update(names)
    found = []
    for carrier, names in sorted(used.items()):
        inputs = reader.graph[carrier]["inputs"]
        raw = {}
        for name in extra_schedulers.ARG_NAMES:
            value = inputs.get(name)
            if name in ("custom_mode", "custom_expression", "custom_sigmas"):
                value = reader.text(value)
            elif isinstance(value, bool) or not isinstance(value, (int, float)):
                value = None
            if value is None:
                reader.warn("Extra Schedulers 값이 연결로 들어와 그 값을 기록하지 않았습니다.")
                return {}
            raw[name] = value
        items = extra_schedulers.infotext_items(extra_schedulers.coerce(raw), names)
        if items not in found:
            found.append(items)
    if len(found) > 1:
        reader.warn("Extra Schedulers 노드가 여럿이고 값이 달라 Extra Schedulers 값을 고르지 않았습니다.")
        return {}
    return found[0] if found else {}


def _extra_samplers_carrier(reader: _Reader, link) -> str | None:
    """``model`` 사슬을 거슬러 처음 만나는 Extra Samplers 값 노드 id(없으면 None). 값은 ``model_options`` 키라 사슬의 다른
    노드가 지우지 않는다 — 가장 가까운 노드가 그 위의 노드를 덮어쓴다(팩 ``ForgeNeoExtraSamplerSettings``)."""
    seen = set()
    while (found := _link(link)) and found[0] in reader.graph and found[0] not in seen and len(seen) < MAX_DEPTH:
        key = found[0]
        seen.add(key)
        node = reader.graph[key]
        if node["class_type"] == extra_samplers.COMFY_SETTINGS_NODE:
            return key
        link = node["inputs"].get("model")
    return None


def _extra_samplers_parameters(reader: _Reader, active: set) -> dict:
    """메인·Hires 패스의 샘플러와 그 model 사슬의 앱 노드 팩 Extra Samplers 값 노드 → Forge infotext 와 같은 키
    (core/extra_samplers.infotext_items — 확장 record_infotext 의 거울): ``ER SDE max stage``(ER SDE 셋, 3 이 아닐 때),
    ``ER SDE eta``(Reverse-time·Tunable, 1 이 아닐 때), ``ER SDE noise window``(구간 켬 + Reverse-time·Tunable + η > 0,
    'a-b'). ComfyUI 모델은 늘 percent_to_sigma 가 있어 구간을 버리지 않는다.

    Forge 처럼 메인·Hires 의 샘플러만 본다(ADetailer·SAM3 패스는 그 이미지의 infotext 에 적지 않는다). 노드가 없으면
    값은 기본값이라 적을 키가 없다. 값이 연결로 들어오거나 노드가 여럿이고 결과가 다르면 고르지 않는다."""
    used: dict = {}
    for key in sorted(active):
        node = reader.graph.get(key) or {}
        kind = node.get("class_type")
        if kind in _SAMPLERS:
            inputs = reader.sampler_inputs(node)
            name = inputs.get("sampler_name")
        elif kind == "ForgeNeoHiresFix":
            inputs = node["inputs"]
            name = inputs.get("sampler_name") or inputs.get("base_sampler_name")
        else:
            continue
        if not isinstance(name, str) or not extra_samplers.uses_values(name):
            continue
        carrier = _extra_samplers_carrier(reader, inputs.get("model"))
        if carrier is not None:
            used.setdefault(carrier, []).append(name)
    found = []
    for carrier, names in sorted(used.items()):
        inputs = reader.graph[carrier]["inputs"]
        values = []
        for name in extra_samplers.ARG_NAMES:
            value = inputs.get(name)
            if not isinstance(value, (int, float)):   # bool 은 int — 구간 켬
                reader.warn("Extra Samplers 값이 연결로 들어와 그 값을 기록하지 않았습니다.")
                return {}
            values.append(value)
        items = extra_samplers.infotext_items(extra_samplers.coerce_args(values), names)
        if items not in found:
            found.append(items)
    if len(found) > 1:
        reader.warn("Extra Samplers 값 노드가 여럿이고 값이 달라 ER SDE 값을 고르지 않았습니다.")
        return {}
    return found[0] if found else {}


def _speed_parameters(reader: _Reader, active: set) -> dict:
    """켜진 앱 노드 팩 Anima SPEED 노드가 출력에 닿으면 Forge infotext 와 같은 ``Anima SPEED`` 요약(core/anima_speed —
    확장 ``SpeedSettings.summary`` 의 거울; 붙여 넣기가 되읽는다). 값은 노드 입력(노드가 확장 규칙으로 다시 읽는 값)이고
    ``hires=`` 는 Hires 패스 노드(``pass_label`` hires)가 있었는지다 — Forge 는 블록의 hires 칸을 Hires 를 끈 생성에도
    적지만 ComfyUI 그래프에는 그때 Hires 노드가 없다(레지스트리 gap). 실행 결과인 ``Anima SPEED status``·``… img2img
    rescale`` 은 그래프로 알 수 없어 적지 않는다(D8). 값이 연결로 들어오거나 노드끼리 값이 다르면 고르지 않는다."""
    found: list = []
    hires = False
    for key in sorted(active):
        node = reader.graph.get(key) or {}
        if node.get("class_type") != anima_speed.COMFY_NODE_CLASS:
            continue
        inputs = node["inputs"]
        enabled = inputs.get("enabled")
        if _link(enabled):
            reader.warn("Anima SPEED 값이 연결로 들어와 그 값을 기록하지 않았습니다.")
            return {}
        if enabled is not True:
            continue
        values: dict = {}
        for name in anima_speed.COMFY_INPUTS:
            if name in ("model", "enabled", "pass_label", "image_seed", "img2img_rescale"):
                continue
            value = inputs.get(name)
            if name in _SPEED_TEXT_INPUTS:
                value = reader.text(value)
            elif not isinstance(value, (int, float)):      # bool 은 int — adaptive
                value = None
            if value is None:
                reader.warn("Anima SPEED 값이 연결로 들어와 그 값을 기록하지 않았습니다.")
                return {}
            values[name] = value
        hires = hires or inputs.get("pass_label") == anima_speed.PASS_HIRES
        if values not in found:
            found.append(values)
    if len(found) > 1:
        reader.warn("Anima SPEED 노드가 여럿이고 값이 달라 Anima SPEED 값을 고르지 않았습니다.")
        return {}
    if not found:
        return {}
    script = anima_speed.coerce([{"enabled": True, **found[0], "hires": hires}])
    return {anima_speed.INFOTEXT_KEY: script.summary()}


def _colorcraft_parameters(reader: _Reader, active: set) -> dict:
    """출력에 닿는 앱 노드 팩 Colorcraft 노드(돌 항목이 있는 패스만 생긴다)의 ``settings_json`` 에 적힌 설정 문자열 →
    Forge infotext 와 같은 ``SAM Extra Colorcraft``(core/colorcraft.to_infotext — 붙여 넣기·가져오기가 되읽는다). base·hires
    노드는 한 설정이라 같은 문자열이다. 다르거나 설정을 읽지 못하면 고르지 않는다. 실행 결과인 status·pre-DD sigma 키는
    그래프로 알 수 없어 적지 않는다(레지스트리 gap)."""
    found = []
    for key in sorted(active):
        node = reader.graph.get(key) or {}
        if node.get("class_type") != colorcraft.COMFY_NODE_CLASS:
            continue
        text = reader.text(node["inputs"].get("settings_json"))
        try:
            settings = json.loads(text) if text is not None else None
        except ValueError:
            settings = None
        infotext = settings.get("infotext") if isinstance(settings, dict) else None
        if not isinstance(infotext, str) or not infotext.strip():
            reader.warn("Colorcraft 노드 설정을 읽지 못해 Colorcraft 값을 기록하지 않았습니다.")
            return {}
        if not settings.get("entries"):
            continue
        if infotext not in found:
            found.append(infotext)
    if len(found) > 1:
        reader.warn("Colorcraft 노드가 여럿이고 설정이 달라 Colorcraft 값을 고르지 않았습니다.")
        return {}
    return {colorcraft.INFOTEXT_KEY: found[0]} if found else {}


def parse_comfy_metadata(prompt_graph: Any = None, workflow: Any = None) -> dict:
    result = {"prompt": "", "negative_prompt": "", "parameters": {}, "warnings": [], "candidates": [], "complete": False}
    graph = {str(key): node for key, node in (prompt_graph.items() if isinstance(prompt_graph, dict) and len(prompt_graph) <= MAX_NODES else [])
             if isinstance(node, dict) and isinstance(node.get("class_type"), str)
             and isinstance(node.get("inputs"), dict)}
    if not graph:
        graph = _workflow_graph(workflow)
    if not graph:
        result["warnings"] = ["해석 가능한 ComfyUI 그래프가 없거나 크기 제한을 초과했습니다. 원본 워크플로를 확인하세요."]
        return result
    reader = _Reader(graph)
    outputs = [key for key, node in graph.items() if node["class_type"] in _OUTPUTS]
    active = reader.ancestors(outputs) if outputs else set(graph)
    for key, node in graph.items():
        if key not in active or node["class_type"] not in _SAMPLERS:
            continue
        if len(result["candidates"]) >= 128:
            reader.warn("샘플러 후보가 너무 많아 자동 적용을 중단했습니다.")
            result["warnings"] = reader.warnings
            return result
        inputs = reader.sampler_inputs(node)
        positive, positive_known = reader.conditioning(inputs.get("positive"))
        negative, negative_known = ([], True) if inputs.get("_no_negative") else reader.conditioning(inputs.get("negative"))
        result["candidates"].append({"node_id": key, "sampler": node["class_type"],
            "prompt": positive[0]["text"] if positive_known and positive else "",
            "negative": negative[0]["text"] if negative_known and negative else "",
            "positive_parts": positive, "negative_parts": negative,
            "positive_known": positive_known, "negative_known": negative_known,
            "parameters": reader.parameters({"inputs": inputs})})
    candidates = result["candidates"]
    if candidates:
        for field, normalized, known in (("prompt", "prompt", "positive_known"), ("negative", "negative_prompt", "negative_known")):
            values = {candidate[field] for candidate in candidates}
            if len(values) == 1 and all(candidate[known] for candidate in candidates):
                result[normalized] = next(iter(values))
            elif len(candidates) > 1:
                reader.warn("샘플러별 프롬프트가 다르거나 미해석 분기가 있어 하나를 임의로 선택하지 않았습니다.")
        result["parameters"] = {key: value for key, value in candidates[0]["parameters"].items()
                                if all(candidate["parameters"].get(key) == value for candidate in candidates)}
    else:
        reader.warn("지원하는 샘플러 연결을 찾지 못했습니다. 텍스트 노드 순서로 프롬프트를 추정하지 않았습니다.")
    result["parameters"].update(_degrid_parameters(reader, active))
    result["parameters"].update(_extra_schedulers_parameters(reader, active))
    result["parameters"].update(_extra_samplers_parameters(reader, active))
    result["parameters"].update(_speed_parameters(reader, active))
    result["parameters"].update(_colorcraft_parameters(reader, active))
    result["warnings"] = reader.warnings
    result["complete"] = True
    return result
