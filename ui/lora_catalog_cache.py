"""LoRA 매니저용 병합 카탈로그 캐시 — 비동기 requestLoras·동기 getLoras·백그라운드 프리워밍이 공유한다.

캐시가 비면 백엔드 ``get_loras()`` HTTP(timeout 10초)와 디스크 카탈로그 병합
(ModelInventory.merge_loras)이 필요하다. 예전엔 LoRA 매니저가 동기 슬롯 getLoras 로 이 둘을
GUI 스레드에서 해 창이 수 초 멈췄다. 지금은
- 매니저가 ``VueBridge.requestLoras`` → 워커 스레드의 :func:`load_catalog_json` → ``lorasReady`` 로 받고,
- 연결 성공 경계(generator_webui.on_webui_info_loaded)가 캐시를 비운 직후 :func:`prewarm_async` 로
  미리 채워, 첫 매니저 열기는 대개 캐시 적중으로 끝난다.
``VueBridge.getLoras`` 는 슬롯이 아닌 파이썬 전용 동기 진입점으로, :func:`load_catalog_json` 을
그대로 부르는 얇은 래퍼다 — 캐시 알고리즘 사본을 따로 두면 운영 경로(requestLoras)와 테스트가
보는 경로가 조용히 갈라진다.

규칙
- raw 캐시(이 모듈의 ``_raw_loras``, 읽기는 :func:`raw_loras`)에는 백엔드의 raw 응답만 둔다.
  예전엔 레거시 PyQt 다이얼로그의 클래스 속성(``LoraManagerDialog._lora_cache``)이었다 — 다이얼로그는
  호출자가 없어 지웠고(audit #177), 캐시 상태는 이 모듈 한 곳에 둔다.
- 병합 결과는 ``bridge._merged_lora_cache`` 에 ``{activeEngine, rawSignature, json, raw}``
  으로 둔다. ``raw`` 는 병합에 쓴 raw 목록 객체 자체다. 캐시 적중은 엔진·서명이 같고
  **지금 raw 캐시가 바로 그 객체일 때만**이다 — 무효화가 넣은 새 ``[]`` 는 내용(서명)이
  같아도 다른 객체라, 늦게 끝난 프리워밍이 무효화 전 상태로 만든 결과를 내보내지 않는다.
  ``bridge`` 가 None 이면(브리지 없는 파이썬 호출) 병합 캐시는 두지 않고 raw 캐시만 갱신한다.
- 목록을 받은 쪽(프리워밍·새로고침)은 받기 전과 raw 캐시 객체가 같을 때만 자기 목록을 넣고 병합한다.
  받는 사이 무효화·다른 새로고침으로 raw 캐시 객체가 바뀌었으면 아무것도 쓰지 않는다(바꾼 쪽이 책임진다).
- raw 캐시의 조건부 교체와 무효화(:func:`invalidate`)는 ``_LOCK`` 아래에서 한다.
  확인과 대입 사이에 무효화가 끼어 무효화 전 목록이 되살아나지 않게 하기 위해서다.
"""
from __future__ import annotations

import hashlib
import json
import threading
from typing import Any, Optional

# raw·병합 캐시를 바꾸는 쪽(프리워밍 워커, GUI 스레드 무효화·새로고침)이 함께 잡는다.
# 백엔드 HTTP·디스크 병합처럼 느린 일은 이 락 밖에서 한다.
_LOCK = threading.Lock()

# 백엔드 get_loras() 의 raw 응답. 비었으면(무효화 직후·첫 사용) 다음 요청이 백엔드에서 받는다.
# 객체 동일성(is)이 캐시 규칙의 일부라 바꿀 때는 늘 새 리스트를 대입한다(제자리 수정 금지).
_raw_loras: list[dict] = []


def raw_loras() -> list[dict]:
    """지금 raw 캐시 객체(복사본 아님 — 동일성 비교에 쓴다). 바꾸지 말 것."""
    return _raw_loras


def active_engine_for(backend_type: Any) -> str:
    value = str(getattr(backend_type, "value", backend_type) or "").casefold()
    return "comfyui" if value == "comfyui" else "forge"


def raw_signature(items: Any) -> str:
    try:
        payload = json.dumps(
            items or [], ensure_ascii=False, sort_keys=True, default=str,
            separators=(",", ":"),
        )
    except Exception:
        payload = repr(items)
    return hashlib.sha256(payload.encode("utf-8", errors="replace")).hexdigest()


def cached_json(cache: Any, active_engine: str, signature: str, *, raw: Any) -> Optional[str]:
    """``raw``(지금의 raw 캐시 객체)로 만든 병합 결과가 캐시에 있으면 그 JSON."""
    if (
        isinstance(cache, dict)
        and cache.get("raw") is raw
        and cache.get("activeEngine") == active_engine
        and cache.get("rawSignature") == signature
        and isinstance(cache.get("json"), str)
    ):
        return cache["json"]
    return None


def cache_entry(active_engine: str, signature: str, encoded: str, *, raw: Any) -> dict[str, Any]:
    return {"activeEngine": active_engine, "rawSignature": signature, "json": encoded, "raw": raw}


def invalidate(bridge: Any) -> None:
    """raw·병합 캐시를 함께 비운다 — 다음 getLoras/프리워밍이 백엔드에서 다시 받는다."""
    global _raw_loras
    with _LOCK:
        _raw_loras = []
        if bridge is not None:
            bridge._merged_lora_cache = None


def _store_merged(bridge: Any, entry: dict[str, Any]) -> None:
    """병합 캐시를 브리지에 둔다(``_LOCK`` 안에서 부른다). 브리지가 없으면(None) 둘 곳이 없다."""
    if bridge is not None:
        bridge._merged_lora_cache = entry


def merged_json(loras: Any, active_engine: str) -> str:
    from core.model_inventory import get_model_inventory

    merged = get_model_inventory(active_engine=active_engine).merge_loras(loras or [])
    return json.dumps(merged, ensure_ascii=False)


def _fetch_and_cache(bridge: Any, *, only_if_cold: bool = False) -> Optional[str]:
    """백엔드에서 raw 목록을 받아 raw·병합 캐시를 채우고 병합 JSON 을 돌려준다(호출 스레드에서).

    받는 사이·병합하는 사이에 raw 캐시 객체가 바뀌었으면(무효화·다른 새로고침) 아무것도 쓰지 않고
    None. 백엔드가 없거나 ``only_if_cold`` 인데 이미 따뜻해도 None.
    """
    global _raw_loras
    from backends import get_backend, get_backend_type

    before = _raw_loras
    engine = active_engine_for(get_backend_type())
    if only_if_cold and before and cached_json(
        getattr(bridge, "_merged_lora_cache", None), engine, raw_signature(before), raw=before,
    ) is not None:
        return None
    backend = get_backend()
    if backend is None:
        return None
    # 빈 응답도 유효한 최신 상태다(삭제된 LoRA 가 계속 보이지 않게 이전 캐시를 그대로 두지 않는다).
    loras = list(backend.get_loras() or [])
    with _LOCK:
        if _raw_loras is not before:
            # 받는 사이 무효화(새 [])나 강제 새로고침이 raw 캐시를 바꿨다. 무효화 뒤의 빈
            # 목록을 병합해 캐시하면 다음 열기가 백엔드를 다시 읽지 않고 디스크 전용
            # 카탈로그를 내보낸다 — 바꾼 쪽(다음 요청·프리워밍)에 맡기고 버린다.
            return None
        _raw_loras = loras
    signature = raw_signature(loras)
    encoded = merged_json(loras, engine)
    with _LOCK:
        if _raw_loras is not loras:
            return None   # 병합하는 사이 무효화/강제 새로고침 — 이 결과는 버린다
        _store_merged(bridge, cache_entry(engine, signature, encoded, raw=loras))
    return encoded


def warm_once(bridge: Any, *, only_if_cold: bool = False) -> bool:
    """raw 목록을 받아 병합 결과까지 캐시에 채운다(호출 스레드에서). 채웠으면 True."""
    return _fetch_and_cache(bridge, only_if_cold=only_if_cold) is not None


def load_catalog_json(bridge: Any, mode: str = "", *, attempts: int = 2) -> str:
    """LoRA 매니저가 받을 병합 카탈로그 JSON — 워커 스레드용(VueBridge.requestLoras)이자
    파이썬 동기 진입점 ``VueBridge.getLoras`` 의 본문. 캐시 갱신은 경합에 안전하게 한다:
    - ``mode != 'force'`` 이고 지금 raw 캐시로 만든 병합 결과가 있으면 그대로(HTTP·디스크 스캔 없음).
    - raw 는 있는데 병합 결과만 없으면 병합만 다시 한다(백엔드 HTTP 없음).
    - raw 가 비었거나 force 면 백엔드에서 받아 병합하고 캐시에 넣는다(:func:`warm_once` 와 같은 규칙).
    받는 사이 백엔드 전환(무효화)·다른 새로고침이 캐시를 바꿨으면 그 결과는 캐시하지 않고 한 번 더
    시도한다 — 옛 백엔드 목록이 새 백엔드 캐시를 덮지 않게. 끝까지 겨루면 이번에 받은 목록을
    캐시 없이 병합해 돌려준다(사용자에게는 결과가 가야 한다).
    ``bridge`` 가 None 이면 병합 캐시 없이 raw 캐시만 같은 규칙으로 갱신한다.
    """
    from backends import get_backend, get_backend_type

    force = mode == "force"
    if force:
        # '목록 다시 스캔' — Forge 는 LoRA 폴더를 스스로 다시 읽지 않는다. 새로 넣은 파일이 보이게
        # 먼저 재스캔을 시킨다(동기 getLoras 와 같은 규칙. ComfyUI 는 object_info 가 매번 새로 읽는다).
        refresh = getattr(get_backend(), "refresh_loras", None)
        if callable(refresh):
            refresh()
    for _ in range(max(1, int(attempts))):
        engine = active_engine_for(get_backend_type())
        raw = _raw_loras
        if not force:
            hit = cached_json(getattr(bridge, "_merged_lora_cache", None), engine,
                              raw_signature(raw), raw=raw)
            if hit is not None:
                return hit
            if raw:
                signature = raw_signature(raw)
                encoded = merged_json(raw, engine)
                with _LOCK:
                    if _raw_loras is raw:
                        _store_merged(bridge, cache_entry(engine, signature, encoded, raw=raw))
                        return encoded
                continue   # 병합하는 사이 캐시가 바뀌었다 — 새 상태로 다시
        encoded = _fetch_and_cache(bridge)
        if encoded is not None:
            if bridge is None:
                return encoded   # 병합 캐시를 둘 곳이 없다 — 방금 raw 캐시에 넣은 목록의 병합 결과
            with _LOCK:
                current = _raw_loras
                entry = getattr(bridge, "_merged_lora_cache", None)
                if isinstance(entry, dict) and entry.get("raw") is current and isinstance(entry.get("json"), str):
                    return entry["json"]
    # 경합이 계속됐다 — 이번 요청의 결과만 캐시 없이 만든다.
    backend = get_backend()
    loras = list(backend.get_loras() or []) if backend is not None else []
    return merged_json(loras, active_engine_for(get_backend_type()))


def prewarm_async(bridge: Any, *, only_if_cold: bool = False) -> threading.Thread:
    """연결 성공 직후처럼 GUI 스레드에서 부른다. 실패는 로그만 남긴다."""

    def _work() -> None:
        try:
            if warm_once(bridge, only_if_cold=only_if_cold):
                print("[LoRA] 목록·카탈로그 프리워밍 완료", flush=True)
        except Exception as exc:
            print(f"[LoRA] 프리워밍 건너뜀: {exc}", flush=True)

    thread = threading.Thread(target=_work, daemon=True, name="lora-prewarm")
    thread.start()
    return thread


__all__ = [
    "active_engine_for",
    "cache_entry",
    "cached_json",
    "invalidate",
    "load_catalog_json",
    "merged_json",
    "prewarm_async",
    "raw_loras",
    "raw_signature",
    "warm_once",
]
