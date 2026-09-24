"""Vue 업스케일(``start_upscale``) 페이로드 → ``BatchUpscaleWorker`` 설정 (순수 로직).

예전 경로는 숨은 레거시 UpscaleTab 을 거쳤다. 업스케일러는 한 번도 채워지지 않은 숨은
콤보에서 ``findText`` 로 찾았고(항상 실패 → ``'(로드 필요)'``), 배율은 존재한 적 없는
``spin_scale`` 을 ``hasattr`` 로 확인해 늘 버려졌다(→ 항상 2배). 그래서 Forge 는
``upscaler_1='(로드 필요)'`` 로, ComfyUI 는 '허용되지 않은 선택'으로 모든 파일이 실패했는데
워커가 파일별 예외를 삼키고 완료 신호를 보내 '업스케일 완료' 토스트가 떴다.

여기서는 Vue 가 보낸 값만으로 설정을 만든다.
"""
from __future__ import annotations

import math
from typing import Iterable, Mapping

DEFAULT_UPSCALER = 'Lanczos'
DEFAULT_SCALE = 2.0
MIN_SCALE = 1.0
MAX_SCALE = 8.0

# 백엔드가 이해하지 못하는 자리표시 이름 — 이 값이 오면 기본 업스케일러로 바꾼다.
_PLACEHOLDER_UPSCALERS = frozenset({'', 'none', '(로드 필요)'})

# ComfyUI 컴파일러(core/comfy_workflow_compiler.compile_upscale)가 모델 없이 처리하는 방식.
# Forge 의 /sdapi/v1/upscalers 에도 Lanczos/Nearest 가 있어 기본값 'Lanczos' 는 양쪽에서 통한다.
COMFY_BUILTIN_UPSCALERS = ('Lanczos', 'Nearest', 'Bilinear', 'Bicubic', 'Area')


def upscaler_name(value) -> str:
    name = str(value or '').strip()
    return DEFAULT_UPSCALER if name.casefold() in _PLACEHOLDER_UPSCALERS else name


def scale_factor(value) -> float:
    try:
        factor = float(value)
    except (TypeError, ValueError):
        return DEFAULT_SCALE
    if not math.isfinite(factor) or factor <= 0:
        return DEFAULT_SCALE
    return max(MIN_SCALE, min(MAX_SCALE, factor))


def input_files(payload: Mapping) -> list[str]:
    """중복·빈 값·문자열 아닌 항목을 뺀 입력 경로 목록(순서 유지)."""
    seen: set[str] = set()
    files: list[str] = []
    raw = payload.get('files') if isinstance(payload, Mapping) else None
    for item in raw or []:
        if not isinstance(item, str):
            continue
        path = item.strip()
        if not path or path in seen:
            continue
        seen.add(path)
        files.append(path)
    return files


def build_upscale_settings(payload: Mapping, output_folder: str) -> dict:
    """``BatchUpscaleWorker`` 설정 — 업스케일만(ADetailer/SAM3 는 각자 탭에서 한다)."""
    data = payload if isinstance(payload, Mapping) else {}
    return {
        'mode': 'upscale_only',
        'upscaler_name': upscaler_name(data.get('upscaler')),
        'scale_mode': 'factor',
        'scale_factor': scale_factor(data.get('scale', DEFAULT_SCALE)),
        'target_width': 1024,
        'target_height': 1024,
        'ad_enabled': False,
        'sam3_enabled': False,
        'output_folder': output_folder,
    }


def comfy_upscale_model_names(object_info) -> list[str]:
    """``/object_info/UpscaleModelLoader`` 응답 → 업스케일 모델 파일 이름.

    예전 형식 ``[[names...]]`` 과 새 COMBO 형식 ``["COMBO", {"options": [...]}]`` 둘 다 읽는다.
    """
    if not isinstance(object_info, Mapping):
        return []
    node = object_info.get('UpscaleModelLoader')
    required = (node or {}).get('input', {}).get('required', {}) if isinstance(node, Mapping) else {}
    spec = required.get('model_name') if isinstance(required, Mapping) else None
    if not isinstance(spec, (list, tuple)) or not spec:
        return []
    head = spec[0]
    if isinstance(head, (list, tuple)):
        return [str(name) for name in head if isinstance(name, str)]
    if head == 'COMBO' and len(spec) > 1 and isinstance(spec[1], Mapping):
        options = spec[1].get('options')
        if isinstance(options, (list, tuple)):
            return [str(name) for name in options if isinstance(name, str)]
    return []


def upscaler_choices(backend_kind: str, names: Iterable) -> list[str]:
    """Vue 업스케일러 드롭다운 목록.

    Forge: ``/sdapi/v1/upscalers`` 이름에서 'None'(아무것도 안 함)을 뺀다.
    ComfyUI: 모델 없이 되는 내장 방식 + ``UpscaleModelLoader.model_name`` 선택지.
    """
    out: list[str] = []
    seen: set[str] = set()

    def add(name) -> None:
        text = str(name or '').strip()
        key = text.casefold()
        if not text or key in _PLACEHOLDER_UPSCALERS or key in seen:
            return
        seen.add(key)
        out.append(text)

    if str(backend_kind or '').strip().casefold() == 'comfyui':
        for name in COMFY_BUILTIN_UPSCALERS:
            add(name)
    for name in names or []:
        add(name)
    return out
