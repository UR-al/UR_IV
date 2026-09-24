# core/cond_rules_store.py
"""전역 조건식(config/cond_rules.json) 저장 — save_cond_rules 액션의 순수 부분 (감사 #108).

Vue(composables/condRules.js)는 두 가지로 저장한다.
- 자동저장: 편집 800ms 뒤·부팅 복원 동기화. 조용히 쓴다(토스트 없음).
- 수동 저장: 조건부 모달의 '즉시 저장' 버튼. ``_manual: true`` 를 실어 보낸다 → 토스트.

``_manual`` 은 제어 플래그라 파일에 남기면 getInitialConfig 로 프론트에 되돌아간다 — 쓰기 전에 뺀다.
``updatedAt``(ms)은 데이터다. 부팅 때 localStorage 사본과 파일 중 더 최신을 고르는 근거라 그대로 쓴다
(frontend/src/utils/condRulesSource.ts pickCondRulesSource — 캐시가 **엄격히** 더 최신이면 캐시가 이긴다).

save_cond_rules 말고 파일을 통째로 바꾸는 경로는 그 규칙에 맞춰 시각을 정해야 한다.
- 설정 백업 가져오기(core/settings_backup.import_settings_archive): :func:`stamp_cond_rules_file` 로
  **가져온 시각**을 찍는다. 백업 안의 updatedAt 은 내보낸 때(또는 없음=0)라, 그대로 두면 백업 뒤에
  편집한 브라우저 캐시가 재시작 부팅에서 이겨 가져온 규칙을 조용히 되덮었다.
- 레거시 이관(generator_main._migrate_legacy_cond_rules): 시각을 찍지 않는다(0). 옛
  prompt_settings.cond_rules_json 은 더 이상 쓰이지 않아, 시각이 있는 캐시는 늘 그보다 새 편집이다.
"""
from __future__ import annotations

import json
import time
from typing import Any, Optional

COND_RULES_NAME = "cond_rules.json"
_CONTROL_KEYS = ("_manual",)


def cond_rules_path() -> str:
    from core.storage_paths import config_file

    return str(config_file(COND_RULES_NAME))


def split_control_flags(payload: Any) -> tuple[dict, bool]:
    """(파일에 쓸 dict, 수동 저장 여부). 입력은 건드리지 않는다."""
    data = dict(payload) if isinstance(payload, dict) else {}
    manual = bool(data.get("_manual", False))
    for key in _CONTROL_KEYS:
        data.pop(key, None)
    return data, manual


def save_cond_rules_payload(payload: Any, path: Optional[str] = None) -> bool:
    """조건식을 원자적으로 쓰고 수동 저장이었는지 돌려준다(토스트 판단은 호출부)."""
    data, manual = split_control_flags(payload)
    from utils.atomic_json import atomic_write_json

    atomic_write_json(path or cond_rules_path(), data)
    return manual


def now_ms() -> int:
    return int(time.time() * 1000)


def stamped_cond_rules(data: Any, stamp_ms: int) -> Optional[dict]:
    """``updatedAt`` 을 ``stamp_ms`` 로 바꾼 사본(제어 플래그 제외). 조건식 dict 가 아니면 None."""
    if not isinstance(data, dict):
        return None
    stamped, _manual = split_control_flags(data)
    stamped["updatedAt"] = int(stamp_ms)
    return stamped


def stamp_cond_rules_file(path: str, stamp_ms: Optional[int] = None) -> bool:
    """파일의 ``updatedAt`` 을 지금(또는 ``stamp_ms``)으로 다시 쓴다 — 바꿨으면 True.

    설정 백업 가져오기가 게시 전에 부른다. 읽을 수 없거나 조건식 dict 가 아닌 파일은 손대지 않는다
    (가져오기는 파일을 그대로 두고, 프론트 getInitialConfig 가 늘 하던 대로 처리한다).
    """
    try:
        with open(path, "r", encoding="utf-8-sig") as handle:
            data = json.load(handle)
    except (OSError, ValueError):
        return False
    stamped = stamped_cond_rules(data, now_ms() if stamp_ms is None else stamp_ms)
    if stamped is None:
        return False
    from utils.atomic_json import atomic_write_json

    atomic_write_json(path, stamped)
    return True
