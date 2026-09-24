# core/seed_explore.py
"""'시드 탐색'(3×3) 작업 만들기 — 순수 로직(Qt 무관).

예전 구현은 subseed/subseed_strength 만 담은 **축약** 아이템 9개를 큐에 넣었고, 큐 소비 경로가
UI 위젯에서 payload 를 다시 만들며 두 키를 버렸다 → 같은 이미지 9장 + seed 칸이 기준 시드로
고정됐다. 이제는 클릭 순간의 **완성 payload**(LoRA·hires·ADetailer·VAE/TE 포함, 와일드카드·훅
해석 완료)를 한 번 만들어 변형마다 deepcopy 하고, 큐 소비 경로(_on_generation_requested)가
그 payload 를 그대로 보낸다(XYZ 와 같은 '동결 스냅숏' 방식).

- Forge/A1111(WebUI): seed 고정 + subseed 변형(variation seed). 0번은 원본(strength 0),
  나머지 8장은 서로 다른 strength(0.05 ~ 0.40)라 원본 복제가 섞이지 않는다.
- ComfyUI(Krea2 포함): subseed 개념이 없다 → 이웃 시드(seed, seed+1, …)로 9장.
"""
from __future__ import annotations

import copy
import random
from typing import Any, Optional

# 앱 생성 경로가 받는 seed 상한 — start_generation 이 payload 를 보내기 전에 검증한다
# (core.payload_validator). 이보다 큰 시드로 만든 작업은 9장 모두 생성 직전에 거부된다.
from core.payload_validator import SEED_MAX

SEED_EXPLORE_COUNT = 9
MARKER = '_seed_explore'
_QUEUE_KEYS = ('id', 'group_id', 'group_index', 'group_total', 'is_last_of_group')


class SeedExploreError(ValueError):
    """큐 항목을 그대로 보낼 수 없을 때(백엔드가 바뀜 등) — 대기열을 멈추고 사유를 보인다."""


def parse_base_seed(value: Any, rng: Optional[random.Random] = None) -> int:
    """이미지 뷰어가 보낸 시드 → 기준 시드. 비었거나 숫자가 아니거나 음수(-1)면 새 랜덤 시드.

    생성 경로가 받는 범위(0~SEED_MAX)를 넘는 시드는 SeedExploreError 다. 예전엔 조용히 새 랜덤
    시드로 바꿔 이미지와 무관한 시드를 탐색했고, 알림의 시드 숫자 말고는 알 길이 없었다.
    (그 시드를 그대로 쓰면 9장 모두 생성 직전 검증에서 거부된다 — 범위는 검증기와 한 값이다)
    """
    try:
        seed = int(str(value).strip())
    except (TypeError, ValueError):
        seed = -1
    if seed < 0:
        return (rng or random).randint(0, SEED_MAX)
    if seed > SEED_MAX:
        raise SeedExploreError(
            f'시드 {seed} 은(는) 생성할 수 있는 범위(0~{SEED_MAX})를 넘어 탐색할 수 없습니다.')
    return seed


def variation_strengths(count: int = SEED_EXPLORE_COUNT) -> list[float]:
    """0번은 원본(0.0), 나머지는 0.05 간격의 서로 다른 강도."""
    return [round(0.05 * i, 2) for i in range(count)]


def build_seed_explore_jobs(
    base_payload: dict,
    *,
    base_seed: int,
    model: str,
    backend_id: str,
    supports_subseed: bool,
    count: int = SEED_EXPLORE_COUNT,
) -> list[dict]:
    """완성 payload 1개 → 큐 항목 count 개(deepcopy). 각 항목에는 MARKER 가 붙는다."""
    strengths = variation_strengths(count)
    jobs: list[dict] = []
    for index in range(count):
        job = copy.deepcopy(base_payload)
        if supports_subseed:
            job['seed'] = base_seed
            job['subseed'] = (base_seed + (index + 1) * 1000) % (SEED_MAX + 1)
            job['subseed_strength'] = strengths[index]
        else:
            job['seed'] = (base_seed + index) % (SEED_MAX + 1)
            job.pop('subseed', None)
            job.pop('subseed_strength', None)
        job[MARKER] = {
            'index': index,
            'total': count,
            'base_seed': base_seed,
            'model': str(model or ''),
            'backend_id': str(backend_id or ''),
        }
        jobs.append(job)
    return jobs


def prepare_seed_explore_generation(item: dict, current_backend_id: str) -> tuple[dict, str]:
    """큐 항목 → (보낼 payload, 모델). 큐 관리 키와 마커는 떼고, 만든 백엔드가 아니면 거부한다."""
    info = item.get(MARKER)
    if not isinstance(info, dict):
        raise SeedExploreError('시드 탐색 항목이 아닙니다.')
    if str(info.get('backend_id') or '') != str(current_backend_id or ''):
        raise SeedExploreError(
            '시드 탐색을 만든 백엔드와 현재 백엔드가 다릅니다. 원래 백엔드를 선택하거나 다시 실행하세요.')
    payload = copy.deepcopy(item)
    payload.pop(MARKER, None)
    for key in _QUEUE_KEYS:
        payload.pop(key, None)
    model = str(info.get('model') or '')
    # ComfyUI 스냅숏은 모델을 따로 들고 있다 — 있으면 그것이 기준(큐 스냅숏 경로와 같다)
    comfy_model = payload.pop('_comfy_model_snapshot', None)
    if comfy_model:
        model = str(comfy_model)
    return payload, model


__all__ = [
    'MARKER',
    'SEED_EXPLORE_COUNT',
    'SeedExploreError',
    'build_seed_explore_jobs',
    'parse_base_seed',
    'prepare_seed_explore_generation',
    'variation_strengths',
]
