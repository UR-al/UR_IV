# ui/seed_explore_actions.py
"""'시드 탐색' 액션 — 완성 payload 를 한 번 동결해 3×3 변형을 대기열에 넣고, 대기열이 그 payload 를
그대로 보낸다(core.seed_explore). generator_main 의 액션 핸들러와 대기열 소비 경로
(ui.queue_item_dispatch — 대기열 매니저 · 자동화 '큐 우선' 둘 다)가 부른다.

UI 위젯에서 다시 만들지 않으므로 subseed 가 살아 있고, seed 칸도 건드리지 않는다(예전엔 기준
시드로 고정된 채 남아 이후 일반 생성이 같은 시드를 썼다).
"""
from __future__ import annotations

from typing import Any


def _notify(owner: Any, kind: str, message: str) -> None:
    bridge = getattr(owner, 'vue_bridge', None)
    if bridge is not None:
        try:
            bridge.showNotification.emit(kind, message)
        except Exception:
            pass
    show = getattr(owner, 'show_status', None)
    if callable(show):
        show(message)


def _generation_in_progress(owner: Any) -> bool:
    worker = getattr(owner, 'gen_worker', None)
    try:
        active = worker is not None and worker.isRunning()
    except RuntimeError:
        active = False   # 지워진 Qt 래퍼는 돌고 있지 않다
    return bool(active and getattr(worker, '_result_emitted', False) is not True)


def start_seed_explore(owner: Any, payload: Any) -> bool:
    """explore_seed 액션. 성공하면 True(대기열에 넣고 시작/합류), 아니면 알림 후 False."""
    from core.seed_explore import build_seed_explore_jobs, parse_base_seed
    try:
        # 기준 시드 — 비었거나 -1 이면 새 랜덤. 생성 경로가 받는 범위를 넘으면 엉뚱한 랜덤 시드로
        # 바꾸지 않고 사유를 보인다(SeedExploreError).
        seed = parse_base_seed(payload.get('seed') if isinstance(payload, dict) else None)
        manager = getattr(owner, 'queue_manager', None)
        panel = getattr(owner, 'queue_panel', None)
        if manager is None or panel is None:
            raise ValueError('대기열을 쓸 수 없습니다.')
        running = bool(getattr(manager, 'is_running', False))
        # 자동화 중이면 대기열 매니저를 시작하지 않는다 — 자동화 '큐 우선' 경로가 다음 장 전에 이 항목들을
        # 먼저 낸다(대기열 매니저와 같은 준비 규칙 ui.queue_item_dispatch 라 동결 payload 그대로 나간다).
        automating = bool(getattr(owner, 'is_automating', False))
        if not running and not automating and _generation_in_progress(owner):
            # 대기열 시작은 메인 생성 워커를 바꾼다 — 수동 생성을 몰래 끊지 않는다
            raise ValueError('이미지 생성 중입니다. 현재 생성이 끝난 뒤 시드 탐색을 시작하세요.')

        base, error = owner._build_generation_payload(snapshot=True)
        if error or base is None:
            raise ValueError(error or '생성 설정을 확인하세요.')
        # 와일드카드·프롬프트 훅을 지금 한 번만 푼다 — 9장이 같은 프롬프트여야 시드 비교가 된다
        from core.chat_generation import prepare_prompt_payload
        base = prepare_prompt_payload(base)

        from backends import BackendType, get_backend, get_backend_type
        from core.xyz_capabilities import backend_identity
        backend = get_backend()
        kind = get_backend_type()
        is_krea2 = bool(getattr(owner, '_is_krea2_generation', lambda: False)())
        supports_subseed = kind == BackendType.WEBUI and not is_krea2
        jobs = build_seed_explore_jobs(
            base,
            base_seed=seed,
            model=owner.model_combo.currentText(),
            backend_id=backend_identity(kind.value, backend),
            supports_subseed=supports_subseed,
        )
        for job in jobs:
            panel.add_single_item(job)
        if running:
            manager.total_count += len(jobs)   # 도는 대기열에 합류 — 카운터를 리셋하지 않는다
        elif not automating:
            manager.start()
        how = '변형 시드(subseed) 강도 0~0.40' if supports_subseed else '이웃 시드(+0~+8)'
        when = ' · 자동화가 다음 장 전에 먼저 생성' if automating and not running else ''
        _notify(owner, 'info', f'시드 탐색: 시드 {seed} 기준 {len(jobs)}장 — {how}{when}')
        return True
    except Exception as exc:
        _notify(owner, 'warning', f'시드 탐색을 시작하지 못했습니다: {exc}')
        return False


def seed_explore_queue_generation(item: dict):
    """대기열 항목 → (payload, model, backend). 만든 백엔드가 아니면 SeedExploreError."""
    from backends import get_backend, get_backend_type
    from core.seed_explore import prepare_seed_explore_generation
    from core.xyz_capabilities import backend_identity
    backend = get_backend()
    payload, model = prepare_seed_explore_generation(
        item, backend_identity(get_backend_type().value, backend))
    return payload, model, backend


__all__ = ['seed_explore_queue_generation', 'start_seed_explore']
