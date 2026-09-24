# ui/queue_item_dispatch.py
"""대기열 항목 → 생성 시작 준비 — 대기열 매니저(GeneratorMainUI._on_generation_requested)와
자동화 '큐 우선' 경로(ActionsMixin._continue_automation)가 **같이** 쓴다.

동결 항목은 만들 때 굳힌 payload 를 그대로 보낸다:
- 시드 탐색(``_seed_explore``) — subseed · LoRA · hires · ADetailer 까지 담은 완성 payload
- XYZ(``_xyz_backend_id``)
- ComfyUI 워크플로 스냅숏(``_comfy_workflow_snapshot``)
- ComfyUI 워크플로 컨트롤 동결(``_comfy_queued_controls``) — 항목을 UI 에 올린 뒤 동결 컨트롤로 만든다

예전 자동화 '큐 우선' 경로는 모든 항목을 _apply_payload_to_ui + start_generation() 으로 UI 에서 다시
만들어, 멈춘 대기열에 남은 시드 탐색 항목이 subseed 를 잃고 같은 이미지를 냈고 seed 칸이 기준
시드로 고정된 채 남았다(이후 자동화 장이 전부 그 시드). 두 소비 경로가 이 한 곳을 거치게 해 다시
갈라지지 않게 한다.
"""
from __future__ import annotations

import copy
from typing import Any, Optional, Tuple

# (보낼 payload, model_override 또는 None=UI 모델, backend_override)
Prepared = Tuple[dict, Optional[str], Any]

_QUEUE_KEYS = ('id', 'group_id', 'group_index', 'group_total', 'is_last_of_group')


def touches_ui(item: Any) -> bool:
    """생성 준비가 UI 위젯에 항목 값을 올리는가 — 일반 항목과 ``_comfy_queued_controls``.

    자동화는 이런 항목 뒤에 자동화 프롬프트를 되돌려야 한다(_queue_dirtied_prompt).
    """
    if not isinstance(item, dict):
        return True
    # prepare_frozen_generation 의 분기 순서와 같다 — 이 셋은 UI 를 건드리지 않는다
    return not (item.get('_seed_explore') or item.get('_xyz_backend_id')
                or '_comfy_workflow_snapshot' in item)


def prepare_frozen_generation(owner: Any, item: Any) -> Optional[Prepared]:
    """동결 항목이면 보낼 준비를 끝낸 (payload, model, backend), 일반 항목이면 None.

    보낼 수 없으면(만든 백엔드가 아님 · 스냅숏 손상 등) 예외 — 호출자가 사유를 보이고 멈춘다.
    ``_comfy_queued_controls`` 만 UI 에 항목을 올린다(동결 대상이 워크플로 컨트롤뿐이라 나머지는
    항목 값을 UI 로 받아 만든다 — 대기열 매니저 경로와 같은 동작).
    """
    if not isinstance(item, dict):
        return None
    if item.get('_seed_explore'):
        # 시드 탐색 — 클릭 때 동결한 완성 payload(subseed 포함)를 그대로 보낸다.
        from ui.seed_explore_actions import seed_explore_queue_generation
        return seed_explore_queue_generation(item)
    if item.get('_xyz_backend_id'):
        return owner._xyz_prepare_queue_generation(item)
    if '_comfy_workflow_snapshot' in item:
        from backends import BackendType, get_backend, get_backend_type
        if get_backend_type() != BackendType.COMFYUI:
            raise ValueError('이 대기열 항목을 만든 ComfyUI 백엔드를 다시 선택하세요.')
        payload = copy.deepcopy(item)
        if '_comfy_model_snapshot' not in payload:
            raise ValueError('대기열의 모델 스냅샷이 없습니다. 작업을 다시 등록하세요.')
        model = str(payload.pop('_comfy_model_snapshot', '') or '')
        for key in _QUEUE_KEYS:
            payload.pop(key, None)
        return payload, model, get_backend()
    if '_comfy_queued_controls' in item:
        from backends import BackendType, get_backend, get_backend_type
        if get_backend_type() != BackendType.COMFYUI or owner._is_krea2_generation():
            raise ValueError('이 대기열 항목을 만든 ComfyUI 이미지 생성 경로를 다시 선택하세요.')
        frozen = item['_comfy_queued_controls']
        if not isinstance(frozen, dict):
            raise ValueError('대기열의 워크플로 설정 스냅샷이 올바르지 않습니다.')
        owner._apply_payload_to_ui(item)
        payload, error = owner._build_generation_payload(comfy_workflow_snapshot=frozen)
        if payload is None:
            raise ValueError(error or '대기열 생성 설정을 확인하세요.')
        return payload, None, get_backend()
    return None


def start_prepared_generation(owner: Any, prepared: Prepared) -> bool:
    """prepare_frozen_generation 결과로 생성을 시작한다. 시작했으면 True."""
    payload, model, backend = prepared
    kwargs = {'payload_override': payload, 'backend_override': backend}
    if model is not None:
        kwargs['model_override'] = model
    return bool(owner.start_generation(**kwargs))


__all__ = [
    'Prepared',
    'prepare_frozen_generation',
    'start_prepared_generation',
    'touches_ui',
]
