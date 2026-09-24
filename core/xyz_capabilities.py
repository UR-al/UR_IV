"""Backend-advertised XYZ controls intersected with executable app parameters."""
from __future__ import annotations

import copy
import hashlib
import itertools
import json
import math
from concurrent.futures import ThreadPoolExecutor

# 샘플러 클래스 목록은 선택 화면·컴파일러와 한 곳에서 공유한다.
from core.comfy_node_classes import SAMPLER_NODES


LABELS = {"steps": "Steps", "cfg_scale": "CFG Scale", "seed": "Seed", "width": "Width",
          "height": "Height", "sampler_name": "Sampler", "scheduler": "Scheduler", "model": "Model",
          "prompt_sr": "Prompt S/R", "negative_sr": "Negative S/R", "distilled_cfg_scale": "Shift",
          "denoising_strength": "Hires denoising"}
# These are application execution/safety bounds, not invented server choices.
BOUNDS = {"steps": (1, 150, "integer"), "cfg_scale": (1, 30, "number"),
          "seed": (-1, 2**32 - 1, "integer"), "width": (64, 4096, "integer"),
          "height": (64, 4096, "integer"), "distilled_cfg_scale": (0, 100, "number"),
          "denoising_strength": (0, 1, "number")}
MAX_JOBS = 256
KREA2_NOTE = "Krea2 전용 생성은 일반 XYZ 샘플링 축을 사용하지 않습니다."


def _krea2_result():
    return {"axes": [], "unsupported": [], "notes": [KREA2_NOTE]}


def backend_identity(kind, backend):
    # A transport-safe identifier; never broadcast private URLs/credentials.
    text = f"{kind}\n{backend.api_url.rstrip('/')}"
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:24]


def _axis(key, spec=None, *, choices=None, source=""):
    result = {"id": key, "label": LABELS[key], "source": source}
    if choices is not None:
        values = list(dict.fromkeys(x for x in choices if isinstance(x, str) and x))
        return {**result, "type": "choice", "choices": values} if values else None
    if key in {"prompt_sr", "negative_sr"}:
        return {**result, "type": "replace"}
    if key not in BOUNDS:
        return None
    low, high, kind = BOUNDS[key]
    spec = spec if isinstance(spec, dict) else {}
    for candidate in (spec.get("minimum", spec.get("min")),):
        if isinstance(candidate, (int, float)):
            low = max(low, candidate)
    for candidate in (spec.get("maximum", spec.get("max")),):
        if isinstance(candidate, (int, float)):
            high = min(high, candidate)
    return {**result, "type": kind, "min": low, "max": high,
            "step": 8 if key in {"width", "height"} else (1 if kind == "integer" else 0.1)} if low <= high else None


def _fields(schema, node):
    info = schema.get(node, {}).get("input", {})
    return {**info.get("required", {}), **info.get("optional", {})} if isinstance(info, dict) else {}


def comfy_capabilities(schema, *, workflow=None, hires=False, family="standard"):
    if not isinstance(schema, dict):
        raise ValueError("ComfyUI 기능 응답이 올바르지 않습니다")
    if family == "krea2":
        return _krea2_result()
    used = set(schema)
    custom_sampler = None
    if workflow:
        candidates = [(str(key), node) for key, node in workflow.items()
                      if node.get("class_type") in SAMPLER_NODES]
        if len(candidates) == 1:
            node_id, node = candidates[0]
            custom_sampler = node["class_type"]
            visited = set()
            def visit(key):
                if key in visited or key not in workflow:
                    return
                visited.add(key)
                for value in workflow[key].get("inputs", {}).values():
                    if isinstance(value, list) and len(value) == 2 and isinstance(value[1], int):
                        visit(str(value[0]))
            visit(node_id)
            used = {workflow[key]["class_type"] for key in visited}
        else:
            used = set()
    sampler = next((name for name in ("KSampler", "KSamplerAdvanced", "ForgeNeoKSamplerCNS")
                    if name in used and name in schema and (not workflow or name == custom_sampler)), None)
    if workflow and not sampler:
        return {"axes": [], "unsupported": [], "notes": ["선택한 사용자 워크플로의 샘플러를 앱의 XYZ 설정에 안전하게 연결할 수 없습니다. KSampler/KSamplerAdvanced 워크플로가 필요합니다."]}
    axes = []
    def add(key, definition, source):
        if not isinstance(definition, (list, tuple)) or not definition:
            return
        choice = definition[0] if isinstance(definition[0], list) else None
        axis = _axis(key, definition[1] if len(definition) > 1 else {}, choices=choice, source=source)
        if axis:
            axes.append(axis)
    if sampler:
        fields = _fields(schema, sampler)
        for key, field in (("seed", "noise_seed" if sampler == "KSamplerAdvanced" else "seed"),
                           ("steps", "steps"), ("cfg_scale", "cfg"),
                           ("sampler_name", "sampler_name"), ("scheduler", "scheduler")):
            if field in fields:
                add(key, fields[field], f"{sampler}.{field}")
        if hires and "denoise" in fields:
            add("denoising_strength", fields["denoise"], f"{sampler}.denoise")
    for kind in ("ForgeNeoLatentInput", "EmptyLatentImage"):
        if kind in used:
            fields = _fields(schema, kind)
            for key in ("width", "height"):
                if key in fields and not any(axis["id"] == key for axis in axes):
                    add(key, fields[key], f"{kind}.{key}")
    text_node = next((kind for kind, field in (("CLIPTextEncode", "text"), ("CLIPTextEncodeSDXL", "text_g"),
        ("ForgeNeoAnimaQwen35Prompt", "prompt"), ("ForgeNeoAnima38V2Prompt", "prompt"))
        if kind in used and field in _fields(schema, kind)), None)
    if text_node:
        axes.extend(_axis(key, source=f"{text_node} text") for key in ("prompt_sr", "negative_sr"))
    choices = []
    for kind, field in (("CheckpointLoaderSimple", "ckpt_name"), ("UNETLoader", "unet_name"),
                        ("ForgeNeoAnima38V2Loader", "model_name")):
        if kind in used:
            definition = _fields(schema, kind).get(field, [])
            if definition and isinstance(definition[0], list):
                choices.extend(definition[0])
    if choices:
        axes.append(_axis("model", choices=choices, source="ComfyUI model loader choices"))
    if "ForgeNeoModelSamplingShift" in used and "shift" in _fields(schema, "ForgeNeoModelSamplingShift"):
        add("distilled_cfg_scale", _fields(schema, "ForgeNeoModelSamplingShift")["shift"], "ForgeNeoModelSamplingShift.shift")
    return {"axes": axes, "unsupported": [], "notes": ["현재 앱의 T2I 실행 경로에 적용 가능한 서버 필드만 표시합니다."]}


def forge_capabilities(openapi, *, scripts=(), samplers=(), schedulers=(), models=(), hires=False, family="standard"):
    if not isinstance(openapi, dict):
        raise ValueError("Forge API schema 응답이 올바르지 않습니다")
    if family == "krea2":
        return _krea2_result()
    try:
        body = openapi["paths"]["/sdapi/v1/txt2img"]["post"]["requestBody"]["content"]["application/json"]["schema"]
        if "$ref" in body:
            body = openapi["components"]["schemas"][body["$ref"].rsplit("/", 1)[-1]]
        fields = body["properties"]
    except (KeyError, TypeError):
        raise ValueError("Forge txt2img API schema를 확인할 수 없습니다") from None
    axes = []
    for key in BOUNDS:
        if key in fields and (key != "denoising_strength" or hires):
            axis = _axis(key, fields[key], source=f"txt2img.{key}")
            if axis:
                axes.append(axis)
    for key, field in (("prompt_sr", "prompt"), ("negative_sr", "negative_prompt")):
        if field in fields:
            axes.append(_axis(key, source=f"txt2img.{field}"))
    for key, values in (("sampler_name", [x.get("name") for x in samplers if isinstance(x, dict)]),
                        ("scheduler", [x.get("label") or x.get("name") for x in schedulers if isinstance(x, dict)])):
        if key in fields:
            axis = _axis(key, choices=values, source=f"sdapi/v1/{'samplers' if key == 'sampler_name' else 'schedulers'}")
            if axis:
                axes.append(axis)
    model_axis = _axis("model", choices=[x.get("title") for x in models if isinstance(x, dict)], source="sdapi/v1/sd-models")
    if model_axis:
        axes.append(model_axis)
    advertised = []
    for script in scripts if isinstance(scripts, list) else ():
        if (isinstance(script, dict) and not script.get("is_img2img")
                and str(script.get("name", "")).replace(" ", "").casefold() in {"x/y/zplot", "xyzplot"}):
            for arg in script.get("args", []):
                if str(arg.get("label", "")).casefold() == "x type":
                    advertised = [x for x in arg.get("choices", []) if isinstance(x, str)]
    mapped = {axis["label"] for axis in axes} | {"Nothing", "Schedule type", "Checkpoint name", "Size", "Distilled CFG Scale"}
    return {"axes": axes, "unsupported": [label for label in advertised if label not in mapped],
            "notes": ["Forge XYZ 스크립트의 임의 확장 축은 실행하지 않습니다. API와 앱이 함께 지원하는 축만 사용합니다."]}


def fetch_capabilities(backend, kind, *, hires=False, family="standard"):
    if kind not in {"comfyui", "webui"}:
        raise ValueError("지원하지 않는 XYZ 백엔드입니다")
    if family == "krea2":
        # Krea2는 어차피 빈 축이므로 서버 스키마를 받으러 네트워크를 타지 않는다.
        result = _krea2_result()
    elif kind == "comfyui":
        workflow = backend._load_configured_workflow("txt2img")
        result = comfy_capabilities(backend.get_object_info(), workflow=workflow, hires=hires, family=family)
    elif kind == "webui":
        import requests
        endpoints = {"openapi": "/openapi.json", "scripts": "/sdapi/v1/script-info",
                     "samplers": "/sdapi/v1/samplers", "schedulers": "/sdapi/v1/schedulers",
                     "models": "/sdapi/v1/sd-models"}
        def read(pair):
            name, path = pair
            response = requests.get(backend.api_url.rstrip("/") + path, timeout=15)
            if response.status_code == 404 and name != "openapi":
                return name, []
            response.raise_for_status()
            data = response.json()
            if not isinstance(data, dict if name == "openapi" else list):
                raise ValueError("백엔드 기능 응답 형식이 올바르지 않습니다")
            return name, data
        with ThreadPoolExecutor(max_workers=5) as executor:
            data = dict(executor.map(read, endpoints.items()))
        result = forge_capabilities(data.pop("openapi"), **data, hires=hires, family=family)
    result.update(backend=kind, backendId=backend_identity(kind, backend), maxJobs=MAX_JOBS)
    result["capabilityId"] = hashlib.sha256(json.dumps(result, sort_keys=True, ensure_ascii=False).encode()).hexdigest()[:24]
    return result


def build_jobs(base_payload, model, axes, capability):
    """Validate a bounded Cartesian product and preserve the full generation snapshot."""
    if not isinstance(axes, list) or not 1 <= len(axes) <= 3:
        raise ValueError("XYZ 축을 1–3개 선택하세요")
    definitions = {axis["id"]: axis for axis in capability.get("axes", [])}
    normalized = []
    seen = set()
    total = 1
    for axis in axes:
        key = axis.get("id") if isinstance(axis, dict) else None
        if key not in definitions or key in seen:
            raise ValueError("현재 백엔드가 지원하지 않거나 중복된 XYZ 축입니다")
        seen.add(key)
        definition = definitions[key]
        values = axis.get("values")
        if not isinstance(values, list) or not values:
            raise ValueError("각 XYZ 축에 값을 입력하세요")
        total *= len(values)
        if total > MAX_JOBS:
            raise ValueError(f"XYZ 조합은 최대 {MAX_JOBS}개입니다")
        checked = []
        for value in values:
            kind = definition["type"]
            if kind in {"integer", "number"}:
                try:
                    number = float(value)
                except (ValueError, TypeError, OverflowError):
                    raise ValueError(f"{definition['label']}: 숫자를 입력하세요") from None
                if (isinstance(value, bool) or not math.isfinite(number) or
                        number < definition["min"] or number > definition["max"] or
                        (kind == "integer" and not number.is_integer()) or
                        (key in {"width", "height"} and number % 8)):
                    raise ValueError(f"{definition['label']}: 허용 범위/간격을 확인하세요")
                checked.append(int(number) if kind == "integer" else number)
            elif kind == "choice":
                if value not in definition["choices"]:
                    raise ValueError(f"{definition['label']}: 서버 목록에 없는 값입니다")
                checked.append(value)
            else:
                search = axis.get("search")
                if not isinstance(search, str) or not search or len(search) > 8000:
                    raise ValueError("S/R 검색어를 입력하세요")
                if not isinstance(value, str) or len(value) > 8000:
                    raise ValueError("S/R 대체값이 올바르지 않습니다")
                checked.append(value)
        normalized.append({"id": key, "label": definition["label"], "search": axis.get("search"), "values": checked})
    jobs = []
    for values in itertools.product(*(axis["values"] for axis in normalized)):
        payload = copy.deepcopy(base_payload)
        selected_model = model
        metadata = {}
        for axis, value in zip(normalized, values):
            key = axis["id"]
            if key in {"prompt_sr", "negative_sr"}:
                field = "prompt" if key == "prompt_sr" else "negative_prompt"
                payload[field] = str(payload.get(field, "")).replace(axis["search"], value)
            elif key == "model":
                selected_model = value
            else:
                payload[key] = value
            metadata[axis["label"]] = value
        payload["_xyz_info"] = {"axes": metadata, "label": ", ".join(f"{k}={v}" for k, v in metadata.items())}
        payload["_xyz_model"] = selected_model
        jobs.append(payload)
    return jobs
