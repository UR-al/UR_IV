"""Generation API 가 쓰기로 한 로컬 포트 — 관리형 엔진 포트 선택이 피해 가야 할 포트 (Qt·HTTP 없음).

앱 시작 순서는 ``_run_startup_sequence``(백엔드 자동 시작 → 워커가 곧바로 ``_choose_port``)가
먼저고, Generation API 서버 bind(``start_if_enabled``)는 그 뒤다. 그래서 bind 시험
(core/port_probe.py)만으로는 아직 비어 있는 API 포트를 Forge 가 가져가고, 이어서 API 가 같은
포트를 배타 bind 하면 Forge 는 모델을 다 올린 뒤에야 bind 에 실패한다. 포트 선택은 시작
순서와 무관하게 **켜 둔** Generation API 설정의 포트를 건너뛴다.

꺼 둔 설정은 포트를 쓰지 않으므로 예약하지 않는다(나중에 켜면 API 쪽 bind 가 분명한 오류로
실패한다). 설정 파일이 없거나 깨졌거나 포트가 잘못됐으면 GenerationApiManager 가 안전한
기본값(꺼짐)으로 뜨므로 역시 예약하지 않는다.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Optional

# 관리형 Forge(17860~17909)·ComfyUI(18188~18237) 자동 포트 범위 밖에 둔다.
# core.generation_api 가 이 값을 가져다 쓴다(단일 출처).
DEFAULT_PORT = 17990
MIN_PORT = 1024
MAX_PORT = 65535


def default_config_path() -> Path:
    """GenerationApiManager 의 기본 설정 파일 (user_data/generation_api.json)."""
    return Path(__file__).resolve().parent.parent / "user_data" / "generation_api.json"


def enabled_listen_port(raw: Any) -> Optional[int]:
    """설정 dict → 켜져 있으면 들을 포트, 아니면 None (순수 함수)."""
    if not isinstance(raw, Mapping) or not bool(raw.get("enabled", False)):
        return None
    value = raw.get("port", DEFAULT_PORT)
    if isinstance(value, bool):
        return None
    try:
        port = int(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return port if MIN_PORT <= port <= MAX_PORT else None


def reserved_ports(config_path: str | Path | None = None) -> frozenset[int]:
    """관리형 엔진이 고르지 말아야 할 포트들 — 켜 둔 Generation API 의 포트."""
    path = Path(config_path) if config_path is not None else default_config_path()
    try:
        with open(path, "r", encoding="utf-8") as handle:
            raw = json.load(handle)
    except (OSError, ValueError):
        return frozenset()
    port = enabled_listen_port(raw)
    return frozenset() if port is None else frozenset({port})


__all__ = [
    "DEFAULT_PORT",
    "default_config_path",
    "enabled_listen_port",
    "reserved_ports",
]
