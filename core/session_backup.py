# core/session_backup.py
"""크래시 복구용 세션 백업(cache/session/session_backup.json) — 읽기·쓰기·정상 종료 표시.

Vue 가 편집 2.5초 뒤·30초마다 {tab, prompt, negative} 를 저장한다. 예전엔 복구 제안이
'현재 메인 프롬프트가 비어 있을 때'만 떴는데, 부팅의 load_settings 가 prompt_settings.json 으로
프롬프트를 늘 채우므로 사실상 뜨지 않았고, 곧이어 주기 저장이 백업을 낡은 프롬프트로 덮어
크래시 직전 편집이 영구히 사라졌다(감사 #160).

이제 정상 종료(_save_shutdown_state)가 백업에 ``clean: true`` 를 표시한다. 다음 부팅에서
``clean`` 이 아니고 백업 프롬프트가 현재 값과 다르면 크래시로 남은 작업이다
(frontend/src/utils/sessionBackup.ts shouldOfferSessionRestore). mtime 비교는 쓰지 않는다 —
부팅 중 연결 경로의 save_settings 가 prompt_settings.json 의 mtime 을 갱신한다.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Optional

SESSION_BACKUP_NAME = "session/session_backup.json"
_LEGACY_PATH = "config/session_backup.json"


def session_backup_path() -> Path:
    from core.storage_paths import cache_file

    return cache_file(SESSION_BACKUP_NAME, legacy_paths=_LEGACY_PATH)


def read_session_backup(path: Optional[Path] = None) -> dict:
    """백업 dict — 없거나 깨졌으면 ``{}``."""
    target = Path(path) if path is not None else session_backup_path()
    try:
        if not target.exists():
            return {}
        data = json.loads(target.read_text(encoding="utf-8") or "{}")
    except (OSError, ValueError):
        return {}
    return data if isinstance(data, dict) else {}


def write_session_backup(payload: Mapping[str, Any], path: Optional[Path] = None) -> dict:
    """편집 중 백업 — 늘 ``clean: false`` 로 쓴다(정상 종료만 clean 을 세운다)."""
    if not isinstance(payload, Mapping):
        raise ValueError("세션 payload는 JSON 객체여야 합니다")
    data = dict(payload)
    data["clean"] = False
    from utils.atomic_json import atomic_write_json

    target = Path(path) if path is not None else session_backup_path()
    atomic_write_json(str(target), data, indent=None)
    return data


def mark_session_clean(path: Optional[Path] = None) -> bool:
    """정상 종료 표시 — 다음 부팅이 이 백업을 복구 대상으로 보지 않는다. 백업이 없으면 no-op."""
    target = Path(path) if path is not None else session_backup_path()
    data = read_session_backup(target)
    if not data or data.get("clean") is True:
        return False
    data["clean"] = True
    from utils.atomic_json import atomic_write_json

    atomic_write_json(str(target), data, indent=None)
    return True
