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
          "denoising_strength": "Hires denoising", "text_encoder": "Text Encoder"}

# ── Text Encoder 축 (sam-extra 0.34.0 의 Forge XYZ '[Modules] Text Encoder' 축과 같은 뜻) ─────────────────────────────
# 칸마다 앱의 TE 선택(TE 칩 — forge_additional_modules 의 TE 칸)을 그 파일로 바꾼다. VAE·그 밖의 모듈은 그대로다. 앱 XYZ 는 칸마다
# 따로 생성하는 대기열이라 Forge·ComfyUI 모두 보통 생성과 같은 길로 간다: Forge 는 모듈이 바뀐 칸에서 /sdapi/v1/options 로
# 모듈을 바꿔 모델을 다시 불러오고(드롭다운에서 고를 때와 같음 — backends/webui_backend._switch_model_if_needed), ComfyUI 는
# 컴파일러가 그 파일을 CLIP 로더에 넣는다. 그래서 Forge XYZ 의 기본 'VAE' 축이 형식이 다른 VAE 를 덮어써 검은 칸이 나던 문제
# (sam-extra 0.34.0 이 고침)는 앱 XYZ 에 없다 — 앱은 Forge XYZ 스크립트를 부르지 않는다.
TEXT_ENCODER = "text_encoder"
TEXT_ENCODER_CURRENT = "Use current"   # 칸에서 TE 를 바꾸지 않는다(지금 TE 칩 그대로) — 확장의 'Use Forge current'
# 확장이 Forge XYZ 에 더한 축 라벨 — 앱 Text Encoder 축이 같은 일을 하므로 '실행하지 않는 확장 축' 목록에 넣지 않는다
FORGE_TEXT_ENCODER_AXIS = "[Modules] Text Encoder"
TEXT_ENCODER_NOTE = ("Text Encoder 축은 칸마다 앱의 TE 선택을 그 파일로 바꿉니다(VAE 는 그대로) — 바뀌는 칸마다 모델을 다시 "
                     "불러오므로 X 축(가장 바깥 반복)에 두면 가장 적게 다시 불러옵니다.")
# Text Encoder 축을 쓴 칸의 비공개 표시 — Forge 는 칸의 모듈 목록을 옵션(config.json 에도 저장)으로 바꾸므로, WebUI 백엔드가
# 처음 바꾸기 전 목록을 기억했다가 모듈 키 없이 오는 다음 요청(TE 칩이 빈 보통 생성 등) 때 되돌린다
# (backends/webui_backend._switch_model_if_needed). 확장의 '[Modules] Text Encoder' 축이 칸의 로딩 파라미터만 바꾸고 칸이 끝나면
# 되돌려 config.json·드롭다운을 건드리지 않는 것과 같은 결과다. 요청에 싣지 않는다(WebUI 백엔드가 떼고, ComfyUI 대기열은
# ui/xyz_actions 가 뗀다 — 그래프는 칸마다 새로 만들어 남는 상태가 없다).
TRANSIENT_MODULES_KEY = "_forge_modules_transient"
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


def text_encoder_axis(text_encoders, *, source):
    """TE 칩 선택지 → Text Encoder 축(값 '지금 TE 그대로' + 파일들). 선택지가 없으면 None(축 없음)."""
    files = [name for name in dict.fromkeys(str(item or "").strip() for item in text_encoders or ())
             if name and name != TEXT_ENCODER_CURRENT]
    if not files:
        return None
    return _axis(TEXT_ENCODER, choices=[TEXT_ENCODER_CURRENT, *files], source=source)


def replace_text_encoder(modules, current, value):
    """forge_additional_modules 에서 지금 TE(``current`` — TE 칩 값, 하나 이하) 칸을 ``value`` 로 바꾼 새 목록.

    TE 가 없으면 ``value`` 를 더한다(TE 칩에서 그 파일을 고른 것과 같다). VAE·다른 모듈은 자리 그대로. TE 가 둘 이상이면
    (Flux CLIP-L + T5 처럼) 어느 칸을 바꿀지 앱이 모르므로 ValueError — 확장 축은 파일 헤더로 자리를 보지만 앱은 파일을
    읽지 않는다.
    """
    current = [str(item).strip() for item in current or () if str(item or "").strip()]
    if len(current) > 1:
        raise ValueError(f"Text Encoder 축: 지금 TE 가 {len(current)}개({', '.join(current)})라 칸마다 어느 것을 바꿀지 "
                         "정할 수 없습니다 — TE 를 하나만 고르거나 이 축을 빼세요")
    out, replaced = [], False
    for item in modules or ():
        if not isinstance(item, str):
            continue
        if item.strip() in current:
            if not replaced:
                out.append(value)
                replaced = True
            continue
        out.append(item)
    if not replaced:
        out.append(value)
    return out


def _fields(schema, node):
    info = schema.get(node, {}).get("input", {})
    return {**info.get("required", {}), **info.get("optional", {})} if isinstance(info, dict) else {}


def comfy_capabilities(schema, *, workflow=None, hires=False, family="standard", text_encoders=()):
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
    notes = ["현재 앱의 T2I 실행 경로에 적용 가능한 서버 필드만 표시합니다."]
    # Text Encoder: 앱이 그래프를 만들 때만(컴파일러가 forge_additional_modules 를 CLIP 로더에 넣는다) — 사용자 워크플로는
    # 그 그래프의 로더를 그대로 쓰므로 앱 TE 선택이 닿지 않는다
    te_axis = text_encoder_axis(text_encoders, source="forge_additional_modules → CLIPLoader") if not workflow else None
    if te_axis:
        axes.append(te_axis)
        notes.append(TEXT_ENCODER_NOTE)
    return {"axes": axes, "unsupported": [], "notes": notes}


def forge_capabilities(openapi, *, scripts=(), samplers=(), schedulers=(), models=(), hires=False, family="standard",
                       text_encoders=()):
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
    te_axis = text_encoder_axis(text_encoders, source="forge_additional_modules")
    if te_axis:
        axes.append(te_axis)
    advertised = []
    for script in scripts if isinstance(scripts, list) else ():
        if (isinstance(script, dict) and not script.get("is_img2img")
                and str(script.get("name", "")).replace(" ", "").casefold() in {"x/y/zplot", "xyzplot"}):
            for arg in script.get("args", []):
                if str(arg.get("label", "")).casefold() == "x type":
                    advertised = [x for x in arg.get("choices", []) if isinstance(x, str)]
    mapped = {axis["label"] for axis in axes} | {"Nothing", "Schedule type", "Checkpoint name", "Size", "Distilled CFG Scale"}
    if te_axis:
        mapped.add(FORGE_TEXT_ENCODER_AXIS)
    notes = ["Forge XYZ 스크립트의 임의 확장 축은 실행하지 않습니다. API와 앱이 함께 지원하는 축만 사용합니다."]
    if te_axis:
        notes.append(TEXT_ENCODER_NOTE)
    return {"axes": axes, "unsupported": [label for label in advertised if label not in mapped], "notes": notes}


def fetch_capabilities(backend, kind, *, hires=False, family="standard", text_encoders=()):
    if kind not in {"comfyui", "webui"}:
        raise ValueError("지원하지 않는 XYZ 백엔드입니다")
    if family == "krea2":
        # Krea2는 어차피 빈 축이므로 서버 스키마를 받으러 네트워크를 타지 않는다.
        result = _krea2_result()
    elif kind == "comfyui":
        workflow = backend._load_configured_workflow("txt2img")
        result = comfy_capabilities(backend.get_object_info(), workflow=workflow, hires=hires, family=family,
                                    text_encoders=text_encoders)
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
        result = forge_capabilities(data.pop("openapi"), **data, hires=hires, family=family, text_encoders=text_encoders)
    result.update(backend=kind, backendId=backend_identity(kind, backend), maxJobs=MAX_JOBS)
    result["capabilityId"] = hashlib.sha256(json.dumps(result, sort_keys=True, ensure_ascii=False).encode()).hexdigest()[:24]
    return result


def build_jobs(base_payload, model, axes, capability, *, text_encoders=()):
    """Validate a bounded Cartesian product and preserve the full generation snapshot.

    ``text_encoders``: 지금 앱 TE 선택(TE 칩 값 — base_payload 의 forge_additional_modules 중 TE 칸). Text Encoder 축이 그
    칸을 칸마다 바꾼다(replace_text_encoder).
    """
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
        if key == TEXT_ENCODER and any(value != TEXT_ENCODER_CURRENT for value in checked):
            replace_text_encoder((), text_encoders, "")   # TE 가 둘 이상이면 첫 칸을 만들기 전에 거절한다
        normalized.append({"id": key, "label": definition["label"], "search": axis.get("search"), "values": checked})
    jobs = []
    text_encoder_axis_used = any(axis["id"] == TEXT_ENCODER for axis in normalized)
    for values in itertools.product(*(axis["values"] for axis in normalized)):
        payload = copy.deepcopy(base_payload)
        if text_encoder_axis_used:
            payload[TRANSIENT_MODULES_KEY] = True
        selected_model = model
        metadata = {}
        for axis, value in zip(normalized, values):
            key = axis["id"]
            if key in {"prompt_sr", "negative_sr"}:
                field = "prompt" if key == "prompt_sr" else "negative_prompt"
                payload[field] = str(payload.get(field, "")).replace(axis["search"], value)
            elif key == "model":
                selected_model = value
            elif key == TEXT_ENCODER:
                modules = [item for item in payload.get("forge_additional_modules") or () if isinstance(item, str)]
                if value != TEXT_ENCODER_CURRENT:
                    modules = replace_text_encoder(modules, text_encoders, value)
                # 칸마다 목록을 늘 싣는다 — 'Use current' 칸이 키 없이 가면 Forge 에는 앞 칸이 옵션으로 바꿔 둔 TE 가 남는다.
                # 앱에서 VAE·TE 를 고르지 않았으면 빈 목록(체크포인트 내장 VAE·TE — ComfyUI 와 같은 뜻)이다.
                payload["forge_additional_modules"] = modules
                # 베이스에 모듈이 없어 Hires 가 'Built-in'(빈 목록)이었으면 1패스와 같은 목록을 쓰게 한다(보통 생성과
                # 같은 규칙 — ui/generator_generation: 모듈이 있으면 'Use same choices')
                if modules and payload.get("enable_hr") and payload.get("hr_additional_modules") == []:
                    payload["hr_additional_modules"] = ["Use same choices"]
            else:
                payload[key] = value
            metadata[axis["label"]] = value
        payload["_xyz_info"] = {"axes": metadata, "label": ", ".join(f"{k}={v}" for k, v in metadata.items())}
        payload["_xyz_model"] = selected_model
        jobs.append(payload)
    return jobs
