"""Read-only, bounded Comfy graph interpretation. Never executes graph nodes.

Only understood connections become reusable prompt text. Ambiguous branches
remain separate candidates; the original JSON belongs to image_metadata.
"""
from __future__ import annotations

from typing import Any

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
}


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
    result["warnings"] = reader.warnings
    result["complete"] = True
    return result
