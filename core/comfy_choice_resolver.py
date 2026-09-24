# core/comfy_choice_resolver.py
"""creator_workflows.build() 그래프를 ComfyUI 서버에 맞추는 준비 단계 (순수, Qt 비의존).

예전엔 같은 로직이 Krea2 T2I/I2I 실행(core/krea2_generation)과 Creator Studio
(ui/creator_actions)에 한 줄씩 복제돼 있었고, Creator 쪽만 H3 캐시 stage/descriptor 를
다뤘다. 이 모듈이 그 상위집합 한 벌이다.

- ``check_required_nodes``: build() 가 선언한 필수 노드가 /object_info 에 모두 있는지.
- ``resolve_choices``: 이식 가능한 모델 경로('Krea2/x.safetensors')를 서버의 정확한
  combo 값('Krea2\\x.safetensors')으로 바꾼다. 경로 구분자·대소문자 무시 정확 일치 →
  유일한 같은 stem(패키징 변형, .pth/.safetensors) → 없으면 RuntimeError.
  H3 캐시 노드의 JSON descriptor 안에 든 조건부/모델 그래프까지 같은 규칙으로 맞춘다
  (캐시 키가 서버 모델 이름으로 계산되게).
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

_CACHE_DESCRIPTOR_NODES = frozenset({
    "ForgeNeoH3ConditioningCachePrepare",
    "ForgeNeoH3ConditioningCacheLoad",
})


def check_required_nodes(built: Mapping[str, Any], available: set[str] | frozenset[str]) -> None:
    missing = sorted(set(built.get("required_node_types", ())) - set(available))
    if missing:
        raise RuntimeError("ComfyUI 필수 노드가 없습니다: " + ", ".join(missing))


def _graph_nodes(graph: Any):
    if isinstance(graph, Mapping):
        return [node for node in graph.values() if isinstance(node, dict)]
    return []


def _input_definitions(object_info: Mapping[str, Any], class_type: str) -> dict[str, Any]:
    schema = object_info.get(class_type, {})
    input_schema = schema.get("input", {}) if isinstance(schema, Mapping) else {}
    definitions: dict[str, Any] = {}
    for section in ("required", "optional"):
        values = input_schema.get(section, {}) if isinstance(input_schema, Mapping) else {}
        if isinstance(values, Mapping):
            definitions.update(values)
    return definitions


def _match_choice(value: str, choices) -> str | None:
    normalized = value.replace("\\", "/").casefold()
    match = next(
        (
            choice for choice in choices
            if isinstance(choice, str) and choice.replace("\\", "/").casefold() == normalized
        ),
        None,
    )
    if match is not None:
        return match
    requested_stem = Path(normalized).stem
    stem_matches = [
        choice for choice in choices
        if isinstance(choice, str) and Path(choice.replace("\\", "/").casefold()).stem == requested_stem
    ]
    return stem_matches[0] if len(stem_matches) == 1 else None


def _resolve_node(node: dict, object_info: Mapping[str, Any]) -> None:
    class_type = str(node.get("class_type", ""))
    inputs = node.get("inputs", {})
    if not isinstance(inputs, dict):
        return
    definitions = _input_definitions(object_info, class_type)
    for name, value in list(inputs.items()):
        if class_type == "LoadImage" and name == "image":
            # 방금 올린 파일은 이 /object_info 스냅샷보다 새로울 수 있다.
            continue
        definition = definitions.get(name)
        if not isinstance(value, str) or not isinstance(definition, (list, tuple)) or not definition:
            continue
        choices = definition[0]
        if not isinstance(choices, (list, tuple)) or not choices:
            continue
        match = _match_choice(value, choices)
        if match is None:
            raise RuntimeError(f"ComfyUI 리소스 선택지에 {class_type}.{name}={value!r} 항목이 없습니다")
        inputs[name] = match


def resolve_choices(built: Mapping[str, Any], object_info: Mapping[str, Any]) -> None:
    """build() 결과(workflow + stages + H3 캐시 descriptor)의 combo 값을 제자리에서 맞춘다."""
    graphs: list[Any] = [built.get("workflow", {})]
    graphs.extend(stage.get("workflow", {}) for stage in built.get("stages", []) or []
                  if isinstance(stage, Mapping))
    descriptors: list[tuple[dict, dict]] = []
    for graph in list(graphs):
        for node in _graph_nodes(graph):
            if node.get("class_type") not in _CACHE_DESCRIPTOR_NODES:
                continue
            inputs = node.get("inputs")
            if not isinstance(inputs, dict) or not isinstance(inputs.get("descriptor"), str):
                continue
            descriptor = json.loads(inputs["descriptor"])
            if not isinstance(descriptor, dict):
                continue
            descriptors.append((inputs, descriptor))
            graphs.append(descriptor.get("conditioning", {}))
            models = descriptor.get("models", [])
            if isinstance(models, list):
                graphs.append(dict(enumerate(models)))
    for graph in graphs:
        for node in _graph_nodes(graph):
            _resolve_node(node, object_info)
    for inputs, descriptor in descriptors:
        inputs["descriptor"] = json.dumps(descriptor, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
