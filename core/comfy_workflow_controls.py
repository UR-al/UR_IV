"""Schema-bound scalar controls for imported Comfy API graphs.

No Comfy nodes, links, files or installed packages are changed. Saved overrides
are scoped to the selected endpoint and file, then bound to graph and schema
fingerprints. A changed graph/schema requires explicit reinspection, never a
best-effort application to another node.
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
import os
from pathlib import Path
import re
import threading
from typing import Any, Mapping

from utils.atomic_json import atomic_write_json

_LOCK = threading.RLock()
_PATH = Path(__file__).resolve().parent.parent / "config" / "comfy_workflow_controls.json"
# These inputs are authored by the app compiler. Exposing a second editor would
# either silently overwrite controls or break prompt/model/seed reproducibility.
_MANAGED = {
    "KSampler": {"seed", "steps", "cfg", "sampler_name", "scheduler", "denoise"},
    "KSamplerAdvanced": {"noise_seed", "steps", "cfg", "sampler_name", "scheduler", "add_noise", "start_at_step", "end_at_step", "return_with_leftover_noise"},
    "ForgeNeoKSamplerCNS": {"seed", "steps", "cfg", "sampler_name", "scheduler", "denoise"},
    "EmptyLatentImage": {"width", "height", "batch_size"},
    "ForgeNeoLatentInput": {"width", "height", "batch_size", "mode"},
    "CheckpointLoaderSimple": {"ckpt_name"},
    # Every loader the compiler writes the selected model into
    # (core/comfy_node_classes.MODEL_LOADER_INPUTS) is app-managed.
    "CheckpointLoader": {"ckpt_name"},
    "UNETLoader": {"unet_name"},
    "DiffusionModelLoaderKJ": {"model_name"},
    "ForgeNeoAnima38V2Loader": {"model_name"},
    "CLIPTextEncode": {"text"},
    "VAELoader": {"vae_name"},
    "CLIPLoader": {"clip_name", "type"},
    "DualCLIPLoader": {"clip_name1", "clip_name2", "type"},
    "TripleCLIPLoader": {"clip_name1", "clip_name2", "clip_name3"},
}


class WorkflowControlError(ValueError):
    pass


def _digest(value: Any) -> str:
    try:
        data = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise WorkflowControlError("워크플로 또는 설정에 JSON으로 표현할 수 없는 값이 있습니다") from exc
    return hashlib.sha256(data.encode("utf-8")).hexdigest()


def workflow_fingerprint(graph: Mapping) -> str:
    if not isinstance(graph, Mapping) or not graph:
        raise WorkflowControlError("API 형식 워크플로를 먼저 선택하세요")
    nodes = {}
    for key, node in graph.items():
        if not isinstance(node, dict) or not isinstance(node.get("inputs"), dict) or not node.get("class_type"):
            raise WorkflowControlError("워크플로 노드 형식이 올바르지 않습니다")
        nodes[str(key)] = {"class_type": node["class_type"], "inputs": node["inputs"]}
    return _digest(nodes)


def _scalar(value: Any) -> bool:
    return isinstance(value, (str, bool, int, float)) and not (
        isinstance(value, float) and not math.isfinite(value))


def describe_controls(graph: Mapping, object_info: Mapping) -> dict:
    fingerprint = workflow_fingerprint(graph)
    if not isinstance(object_info, Mapping) or not object_info:
        raise WorkflowControlError("ComfyUI에 연결하여 입력 스키마를 읽어야 합니다")
    nodes = []
    contracts = []
    for node_id, node in graph.items():
        class_type = str(node["class_type"])
        info = object_info.get(class_type)
        if not isinstance(info, Mapping):
            raise WorkflowControlError(f"ComfyUI에 필요한 노드가 없습니다: {class_type} ({node_id})")
        inputs = info.get("input", {})
        definitions = {**inputs.get("required", {}), **inputs.get("optional", {})}
        fields = []
        for name, value in node["inputs"].items():
            if not _scalar(value):
                continue  # Links/containers are never exposed as editable JSON.
            spec = definitions.get(name)
            if not isinstance(spec, (list, tuple)) or not spec:
                continue
            kind, options = spec[0], spec[1] if len(spec) > 1 and isinstance(spec[1], dict) else {}
            if options.get("forceInput"):
                continue
            if isinstance(kind, list) and kind and all(_scalar(item) for item in kind):
                field = {"type": "enum", "choices": kind}
            elif kind in ("INT", "FLOAT", "BOOLEAN", "STRING"):
                field = {"type": kind.lower()}
            else:
                continue
            app_sampler_option = class_type == "ForgeNeoKSamplerCNS" and name.startswith(("spectrum_", "speed_", "cns_"))
            field.update({"nodeId": str(node_id), "classType": class_type, "name": name,
                          "managed": name in _MANAGED.get(class_type, set()) or app_sampler_option})
            for bound in ("min", "max", "step"):
                if field["type"] in ("int", "float") and isinstance(options.get(bound), (int, float)) and math.isfinite(options[bound]):
                    field[bound] = str(options[bound]) if field["type"] == "int" else options[bound]
            if field["type"] == "string":
                field["multiline"] = bool(options.get("multiline"))
            contracts.append(copy.deepcopy(field))
            # Keep uint64 seeds exact across QWebChannel's JavaScript boundary.
            field["value"] = str(value) if field["type"] == "int" else value
            fields.append(field)
        if fields:
            nodes.append({"id": str(node_id), "classType": class_type,
                          "title": str(node.get("_meta", {}).get("title") or info.get("display_name") or class_type),
                          "fields": fields})
    return {"workflowFingerprint": fingerprint, "schemaFingerprint": _digest(contracts), "nodes": nodes}


def validate_value(field: Mapping, value: Any) -> Any:
    name = f"{field['nodeId']}.{field['name']}"
    kind = field["type"]
    if kind == "boolean":
        if not isinstance(value, bool):
            raise WorkflowControlError(f"{name}: 체크 상태는 true/false여야 합니다")
        return value
    if kind == "string":
        if not isinstance(value, str) or len(value) > 262144:
            raise WorkflowControlError(f"{name}: 텍스트 형식 또는 길이를 확인하세요")
        return value
    if kind == "enum":
        if not any(type(value) is type(choice) and value == choice for choice in field["choices"]):
            raise WorkflowControlError(f"{name}: 현재 서버의 선택지에 없는 값입니다")
        return value
    if isinstance(value, bool):
        raise WorkflowControlError(f"{name}: 숫자 값이 필요합니다")
    if kind == "int":
        if isinstance(value, str) and re.fullmatch(r"[+-]?\d+", value.strip()):
            value = int(value)
        if not isinstance(value, int):
            raise WorkflowControlError(f"{name}: 정수 값이 필요합니다")
    elif kind == "float":
        try:
            value = float(value)
        except (TypeError, ValueError) as exc:
            raise WorkflowControlError(f"{name}: 숫자 값이 필요합니다") from exc
        if not math.isfinite(value):
            raise WorkflowControlError(f"{name}: 유한한 숫자가 필요합니다")
    else:
        raise WorkflowControlError(f"{name}: 지원하지 않는 입력 형식입니다")
    cast = int if kind == "int" else float
    if "min" in field and value < cast(field["min"]) or "max" in field and value > cast(field["max"]):
        raise WorkflowControlError(f"{name}: 허용 범위 {field.get('min', '−∞')} ~ {field.get('max', '∞')}를 확인하세요")
    return value


def validate_controls(graph: Mapping, object_info: Mapping, binding: Mapping) -> dict:
    schema = describe_controls(graph, object_info)
    if not isinstance(binding, Mapping) or binding.get("workflowFingerprint") != schema["workflowFingerprint"]:
        raise WorkflowControlError("워크플로가 변경되었습니다. 상세 설정을 새로 읽고 다시 저장하거나 저장된 설정을 해제하세요")
    if binding.get("schemaFingerprint") != schema["schemaFingerprint"]:
        raise WorkflowControlError("노드 입력 스키마가 변경되었습니다. 상세 설정을 새로 읽고 다시 저장하세요")
    overrides = binding.get("overrides", [])
    if not isinstance(overrides, list) or len(overrides) > 4096:
        raise WorkflowControlError("상세 설정 목록 형식이 올바르지 않습니다")
    allowed = {(field["nodeId"], field["name"]): field for node in schema["nodes"] for field in node["fields"]}
    cleaned, seen = [], set()
    for override in overrides:
        if not isinstance(override, dict):
            raise WorkflowControlError("상세 설정은 입력별 객체여야 합니다")
        key = (str(override.get("nodeId", "")), override.get("name"))
        field = allowed.get(key)
        if field is None or field["managed"] or override.get("classType") != field["classType"]:
            raise WorkflowControlError(f"수정할 수 없는 워크플로 입력입니다: {key[0]}.{key[1]}")
        if key in seen:
            raise WorkflowControlError(f"중복된 워크플로 입력입니다: {key[0]}.{key[1]}")
        seen.add(key)
        cleaned.append({"nodeId": key[0], "name": key[1], "classType": field["classType"],
                        "value": validate_value(field, override.get("value"))})
    return {"workflowFingerprint": schema["workflowFingerprint"], "schemaFingerprint": schema["schemaFingerprint"],
            "overrides": cleaned}


def apply_controls(compiled: Mapping, original: Mapping, object_info: Mapping, binding: Mapping) -> dict:
    binding = validate_controls(original, object_info, binding)
    graph = copy.deepcopy(dict(compiled))
    for override in binding["overrides"]:
        node = graph.get(override["nodeId"])
        name = override["name"]
        if not node or node.get("class_type") != override["classType"] or name not in node.get("inputs", {}) or not _scalar(node["inputs"][name]):
            raise WorkflowControlError(f"컴파일로 입력 구조가 변경되었습니다: {override['nodeId']}.{name}; 상세 설정을 해제하세요")
        # Do not overwrite a field which an app compiler hook has taken ownership
        # of since inspection. Unchanged custom scalar inputs remain safe.
        if node["inputs"][name] != original[override["nodeId"]]["inputs"][name]:
            raise WorkflowControlError(f"앱 설정과 상세 설정이 충돌합니다: {override['nodeId']}.{name}")
        node["inputs"][name] = override["value"]
    return graph


def _key(endpoint: str, workflow_path: str) -> str:
    return _digest([endpoint.rstrip("/"), os.path.normcase(os.path.abspath(workflow_path))])


def _read_store(path: Path) -> dict:
    if not path.exists():
        return {"version": 1, "bindings": {}}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise WorkflowControlError("워크플로 상세 설정 파일을 읽지 못했습니다. 원본 파일을 보존하고 확인하세요") from exc
    if not isinstance(data, dict) or data.get("version") != 1 or not isinstance(data.get("bindings"), dict):
        raise WorkflowControlError("워크플로 상세 설정 저장 형식이 올바르지 않습니다")
    return data


def load_workflow_controls(endpoint: str, workflow_path: str, graph: Mapping | None, *, store_path: Path | None = None) -> dict | None:
    if not workflow_path or graph is None:
        return None
    with _LOCK:
        binding = _read_store(store_path or _PATH)["bindings"].get(_key(endpoint, workflow_path))
    if binding is not None and (not isinstance(binding, dict) or binding.get("workflowFingerprint") != workflow_fingerprint(graph)):
        raise WorkflowControlError("저장된 상세 설정과 워크플로가 다릅니다. 상세 설정을 다시 저장하거나 해제하세요")
    return copy.deepcopy(binding)


def snapshot_comfy_payload(backend, payload: Mapping, mode: str = "txt2img") -> dict:
    """Freeze local workflow choices without querying, installing or running Comfy.

    Only identities and explicit scalar overrides enter the queued payload. The
    original JSON remains on disk and is revalidated when the job executes.
    A frozen absence of overrides is meaningful: later settings must not apply.
    """
    mode = "img2img" if mode == "inpaint" else mode
    if mode not in ("txt2img", "img2img"):
        raise WorkflowControlError("스냅샷 생성 모드가 올바르지 않습니다")
    path = backend._configured_workflow_path(mode)
    graph = backend._load_configured_workflow(mode)
    result = copy.deepcopy(dict(payload))
    result["_comfy_workflow_snapshot"] = {
        "version": 1, "mode": mode, "endpointId": _digest(backend.api_url.rstrip("/")),
        "pathId": _key(backend.api_url, path),
        "workflowFingerprint": workflow_fingerprint(graph) if graph is not None else None,
        "controls": load_workflow_controls(backend.api_url, path, graph),
    }
    return result


def generation_workflow_controls(endpoint: str, path: str, graph: Mapping | None, payload: Mapping, mode: str) -> dict | None:
    """Resolve frozen queue controls, or current controls for direct API callers."""
    if "_comfy_workflow_snapshot" not in payload:
        return load_workflow_controls(endpoint, path, graph)
    frozen = payload["_comfy_workflow_snapshot"]
    expected = {
        "version": 1, "mode": "img2img" if mode == "inpaint" else mode,
        "endpointId": _digest(endpoint.rstrip("/")), "pathId": _key(endpoint, path),
        "workflowFingerprint": workflow_fingerprint(graph) if graph is not None else None,
    }
    if not isinstance(frozen, Mapping) or any(frozen.get(key) != value for key, value in expected.items()):
        raise WorkflowControlError("대기열을 만든 ComfyUI 서버·워크플로와 현재 설정이 다릅니다. 원래 설정으로 돌아가거나 작업을 다시 등록하세요")
    if "controls" not in frozen or frozen["controls"] is not None and not isinstance(frozen["controls"], Mapping):
        raise WorkflowControlError("대기열의 워크플로 상세 설정 스냅샷이 올바르지 않습니다")
    return copy.deepcopy(frozen["controls"])


def save_workflow_controls(endpoint: str, workflow_path: str, graph: Mapping, object_info: Mapping, binding: Mapping, *, store_path: Path | None = None) -> dict:
    if not workflow_path:
        raise WorkflowControlError("저장할 워크플로 파일을 먼저 선택하세요")
    clean = validate_controls(graph, object_info, binding)
    with _LOCK:
        path = store_path or _PATH
        data = _read_store(path)
        data["bindings"][_key(endpoint, workflow_path)] = clean
        atomic_write_json(str(path), data)
    return clean


def clear_workflow_controls(endpoint: str, workflow_path: str, *, store_path: Path | None = None) -> None:
    with _LOCK:
        path = store_path or _PATH
        data = _read_store(path)
        data["bindings"].pop(_key(endpoint, workflow_path), None)
        atomic_write_json(str(path), data)


def controls_for_wire(binding: Mapping | None, schema: Mapping) -> dict | None:
    """Encode INT override values as decimal text for JavaScript precision."""
    if binding is None:
        return None
    copy_binding = copy.deepcopy(dict(binding))
    kinds = {(field["nodeId"], field["name"]): field["type"] for node in schema["nodes"] for field in node["fields"]}
    for item in copy_binding["overrides"]:
        if kinds.get((item["nodeId"], item["name"])) == "int":
            item["value"] = str(item["value"])
    return copy_binding


def feature_preflight(compiler, model_name: str, payload: Mapping, *, workflow=None, workflow_controls=None) -> dict:
    """Dry compilation: no /prompt, model loads, downloads or generation calls."""
    sam = compiler._sam3_state(payload)
    requested = [
        ("hires", "Hires.fix 업스케일", bool(payload.get("enable_hr")), "ForgeNeoHiresFix"),
        ("adetailer", "ADetailer", bool(compiler._adetailer_slots(payload)), "ForgeNeoADetailer"),
        ("sam3", "SAM3 영역 보정", sam is not None, "ForgeNeoSAM3Mask"),
        ("negpip", "NegPiP", bool(compiler._script(payload, "NegPiP")), "ForgeNeoNegPip"),
        ("spectrum", "Spectrum 실험 가속", bool(payload.get("spectrum_enabled")), "ForgeNeoKSamplerCNS"),
    ]
    if sam is not None:
        requested.append(("sam3_targets", f"SAM3 대상: {sam.get('sam3_prompt', 'face')}", True, "ForgeNeoSAM3Mask"))
    for index, target in enumerate(payload.get("_comfy_detail_passes", []), start=2):
        requested.append((f"sam3_pass_{index}", f"SAM3 {index}차 보정: {target}", True, "ForgeNeoSAM3Mask"))
    rows = [{"id": key, "label": label, "requested": enabled, "state": "pending" if enabled else "off"}
            for key, label, enabled, _ in requested]
    try:
        graph = compiler.compile("txt2img", model_name, payload, workflow=workflow, workflow_controls=workflow_controls)
        classes = {node.get("class_type") for node in graph.values()}
        sam_masks = [node for node in graph.values() if node.get("class_type") == "ForgeNeoSAM3Mask"]
        if sam is not None and len(sam_masks) < 1 + len(payload.get("_comfy_detail_passes", [])):
            raise WorkflowControlError("요청한 SAM3 보정 패스가 최종 그래프에 모두 포함되지 않았습니다")
        for row, (_, _, enabled, class_type) in zip(rows, requested):
            if enabled:
                row["state"] = "ready" if class_type in classes else "missing"
        if any(row["state"] == "missing" for row in rows):
            raise WorkflowControlError("선택한 기능이 최종 워크플로에 포함되지 않았습니다")
        return {"ok": True, "features": rows, "nodeCount": len(graph),
                "note": "T2I 그래프와 현재 노드 스키마 검증 완료. 모델 파일 로딩·VRAM·실제 화질은 생성 시 확인됩니다."}
    except Exception as exc:
        for row in rows:
            if row["state"] == "pending":
                row["state"] = "blocked"
        return {"ok": False, "features": rows, "error": str(exc),
                "note": "생성은 실행하지 않았습니다. 오류를 해결한 뒤 다시 확인하세요."}
