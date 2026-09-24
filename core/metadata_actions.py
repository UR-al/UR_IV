# core/metadata_actions.py
"""이미지 메타데이터 → UI 액션(T2I 전송·즉시 생성·큐 추가) 공용 규칙. Qt 비의존.

Gallery 'T2I에서 사용', PNG Info 'T2I 전송/즉시 생성', History '프롬프트 당겨오기/다음 큐'가
각자 raw 를 split 하던 것을 없애고 core.image_metadata 의 파싱 결과(prompt / negative /
parameters dict / can_apply)만 쓰게 한다. 그래서 네거티브가 없는 A1111 이미지도 네 액션이
같은 (prompt, negative, size) 를 낸다.
"""
from __future__ import annotations

import json
from typing import Any, Callable, Optional

from core.image_metadata import metadata_ui_fields, parse_infotext

COMFY_AMBIGUOUS_MESSAGE = "ComfyUI 프롬프트를 확정할 수 없습니다. 메타데이터에서 내용을 확인하세요."
READ_FAILED_MESSAGE = "이미지 메타데이터를 읽지 못했습니다."
NO_PROMPT_MESSAGE = "이 이미지에 프롬프트 정보가 없습니다"
NO_GENERATION_INFO_MESSAGE = "이 이미지에 생성 정보가 없습니다"


def infotext_ui_fields(raw: str) -> dict:
    """infotext 문자열만 있는 레거시 페이로드 → getImageExif 와 같은 필드."""
    return metadata_ui_fields(parse_infotext(raw))


def resolve_action_metadata(payload: Any, read_image_metadata: Callable[[str], str],
                            clean_path: Callable[[str], str] = lambda p: p,
                            is_file: Callable[[str], bool] = lambda _p: False) -> Optional[dict]:
    """액션 페이로드에서 메타데이터 dict 를 고른다.

    1. ``payload['metadata']`` 또는 페이로드 자체가 getImageExif 결과(source 키가 있음)면 그대로
       — Gallery 확대 뷰에서 편집한 프롬프트가 여기로 온다.
    2. 경로만 있으면 ``read_image_metadata(path)``(getImageExif JSON)로 읽는다 — Favorites 등.
    3. infotext 문자열만 있으면(구 프런트) core 파서로 읽는다.
    못 고르면 None.
    """
    if not isinstance(payload, dict):
        return None
    metadata = payload.get("metadata")
    if not isinstance(metadata, dict) and "source" in payload and (
            "prompt" in payload or "parameters" in payload or "raw" in payload):
        metadata = payload
    if isinstance(metadata, dict) and metadata.get("source"):
        return metadata
    path = clean_path(str(payload.get("path") or ""))
    if path and is_file(path):
        try:
            info = json.loads(read_image_metadata(path))
        except (TypeError, ValueError):
            return {"error": READ_FAILED_MESSAGE}
        return info if isinstance(info, dict) else {"error": READ_FAILED_MESSAGE}
    raw = payload.get("raw") or payload.get("exif")
    if not raw and isinstance(payload.get("parameters"), str):
        raw = payload.get("parameters")
    if isinstance(raw, str) and raw.strip():
        return infotext_ui_fields(raw)
    return None


def apply_block_reason(info: Optional[dict], *, comfy_message: str = COMFY_AMBIGUOUS_MESSAGE) -> str:
    """이 메타데이터를 UI 에 적용하면 안 되는 이유(사용자 메시지). 적용해도 되면 ''."""
    if not isinstance(info, dict) or info.get("error"):
        return READ_FAILED_MESSAGE
    if info.get("source") == "comfyui" and info.get("can_apply") is not True:
        return comfy_message
    if info.get("can_apply") is False:
        return NO_PROMPT_MESSAGE
    return ""


def prompt_transfer(info: Optional[dict], *,
                    comfy_message: str = COMFY_AMBIGUOUS_MESSAGE) -> tuple[str, str, str]:
    """'T2I 로 프롬프트 보내기' 계열(Gallery·PNG Info·History 당겨오기) 공용 판정.

    (prompt, negative, 거부 사유). 적용할 수 없거나 프롬프트·네거티브가 둘 다 비었으면
    사유만 돌려준다 — 파라미터만 있는 이미지(can_apply=True, prompt='')를 보내면 사용자의
    T2I 프롬프트 칸이 알림 없이 비워진다.
    """
    reason = apply_block_reason(info, comfy_message=comfy_message)
    if reason:
        return "", "", reason
    prompt = str(info.get("prompt") or "")
    negative = str(info.get("negative") or "")
    if not (prompt or negative):
        return "", "", NO_PROMPT_MESSAGE
    return prompt, negative, ""


def _lower_keys(params: Any) -> dict:
    if not isinstance(params, dict):
        return {}
    return {str(key).lower().replace(" ", "_"): value for key, value in params.items()}


def _to_int(value: Any, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError, OverflowError):
        return default


def _to_float(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError, OverflowError):
        return default


def build_generation_item(info: dict, defaults: dict, *, preserve_seed: bool = False) -> Optional[dict]:
    """메타데이터 → 큐/즉시 생성 아이템.

    프롬프트·샘플러·스케줄러·steps·cfg·해상도는 메타데이터 값(없으면 ``defaults``), seed 는
    ``preserve_seed`` 면 메타데이터 값, 아니면 -1(비슷한 변형). 적용할 수 없거나 프롬프트가
    없으면 None. 파라미터는 core 가 파싱한 dict 만 쓴다 — raw 를 다시 split 하지 않는다.
    """
    if not isinstance(info, dict) or apply_block_reason(info):
        return None
    prompt = str(info.get("prompt") or "")
    negative = str(info.get("negative") or "")
    if not prompt:
        return None
    parameters = info.get("parameters")
    if not isinstance(parameters, dict) and info.get("source") != "comfyui" and isinstance(info.get("raw"), str):
        parameters = parse_infotext(info["raw"]).parameters
    params = _lower_keys(parameters)
    item = {
        "prompt": prompt,
        "negative_prompt": negative,
        "sampler_name": str(params.get("sampler") or defaults.get("sampler_name") or ""),
        "scheduler": str(params.get("schedule_type") or defaults.get("scheduler") or ""),
        "steps": _to_int(params.get("steps"), _to_int(defaults.get("steps"), 20)),
        "cfg_scale": _to_float(params.get("cfg_scale"), _to_float(defaults.get("cfg_scale"), 7.0)),
        "seed": _to_int(params.get("seed"), -1) if preserve_seed else -1,
        "width": _to_int(defaults.get("width"), 1024),
        "height": _to_int(defaults.get("height"), 1024),
    }
    size = str(params.get("size") or "").lower().split("x")
    if len(size) == 2:
        item["width"] = _to_int(size[0].strip(), item["width"])
        item["height"] = _to_int(size[1].strip(), item["height"])
    return item
