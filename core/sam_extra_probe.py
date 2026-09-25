# core/sam_extra_probe.py
"""sam-extra 기능 스냅샷 수집 — GET + WebUI 주소별 TTL 캐시. Qt 의존 없음.

**워커 전용**: ``get_capabilities`` 는 네트워크를 기다리므로 GUI 스레드에서 부르지 않는다.
GUI 스레드는 ``peek_capabilities`` (캐시만 봄)나 메인 UI 의 ``sam_extra_capabilities`` 속성을 읽는다.

보내는 요청은 모두 GET 이다. POST·옵션 변경·``/sam3-lora/spawn``(프로세스를 띄움)은 쓰지 않는다.
응답 본문이 필요 없는 존재 확인(메모·TIPO·레퍼런스 라우트)은 ``stream=True`` 로 헤더만 읽고 끊는다
— 메모 목록이 커도 받지 않는다. Gradio ``/config`` (수 MB)는 sam-extra 스크립트가 보일 때만 받는다.

**예외 — ``/sdapi/v1/extensions`` 는 GET 이지만 부작용이 있다**: Forge classic 은 이 요청마다
``extensions.list_extensions()`` 로 확장 레지스트리를 비우고 다시 스캔한다(요구 사항 확인과 콘솔 출력,
확장마다 git 정보 읽기와 cache.json 쓰기). 그동안 레지스트리를 읽는 Forge 코드는 빈 목록을 볼 수 있다.
그래서 이 응답만 ``ExtensionsCache`` 에 주소별로 오래(``EXTENSIONS_TTL_S``) 두고, 연결·강제 새로고침
(``refresh=True``)으로도 다시 묻지 않는다. 사용자의 수동 새로고침만 ``invalidate_extensions`` 로 버린다
(호출하는 쪽이 간격을 제한한다). 이 응답은 버전 힌트와 '확장 꺼짐' 경고에만 쓰여 늦어도 ``may_use`` 는
바뀌지 않는다.
"""
from __future__ import annotations

import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Callable, Dict, Mapping, Optional

from core.sam_extra_capabilities import (
    EP_CN_MODELS, EP_CN_MODULES, EP_CONTRACT, EP_EXTENSIONS, EP_GRADIO_CONFIG, EP_LORA_CONFIG,
    EP_MEMOS, EP_REFERENCE, EP_SCRIPTS, EP_SCRIPT_INFO, EP_TILE_REPAIR, EP_TIPO, NOTEBOOK_HEADERS,
    STATUS_ERROR, STATUS_OK, HttpResult, SamExtraCapabilities, build_capabilities,
    has_sam_extra_scripts, unknown_capabilities,
)

DEFAULT_TTL_S = 300.0      # 정상 스냅샷 재사용 시간 — 연결·수동 새로고침은 refresh=True 로 무시한다
FAILED_TTL_S = 20.0        # 실패 스냅샷은 짧게 — 곧 뜰 Forge 를 오래 '없음'으로 두지 않는다
EXTENSIONS_TTL_S = 1800.0  # /sdapi/v1/extensions (부작용 있는 GET) 재사용 시간 — refresh=True 도 무시하지 않는다
REQUEST_TIMEOUT_S = 10.0
CONFIG_TIMEOUT_S = 30.0
MAX_BODY_BYTES = 48 * 1024 * 1024   # Gradio /config 가 약 4 MB (확장 많은 설치는 더 크다)

# (경로, 헤더, 본문 필요?) — 1단계는 병렬로 보낸다. /sdapi/v1/extensions 는 ExtensionsCache 를 거쳐 따로 보낸다.
_PHASE1 = (
    (EP_SCRIPTS, None, True),
    (EP_SCRIPT_INFO, None, True),
    (EP_CN_MODELS, None, True),
    (EP_CN_MODULES, None, True),
    # LoRA Manager 라우트도 메모처럼 같은 출처 헤더가 없으면 403 이다. 헤더를 보지 않는 옛 확장에 보내도
    # 무해하다(모르는 요청 헤더는 무시된다).
    (EP_LORA_CONFIG, NOTEBOOK_HEADERS, True),
    (EP_MEMOS, NOTEBOOK_HEADERS, False),
    (EP_TIPO, None, False),
    (EP_REFERENCE, None, False),
    (EP_TILE_REPAIR, None, False),   # POST 전용 → GET 405 = 있음(확장은 아무 일도 안 한다)
    (EP_CONTRACT, None, True),
)

HttpGet = Callable[..., HttpResult]  # (url, *, headers, timeout, want_body) -> HttpResult


class _TooLarge(Exception):
    pass


def _read_bounded(response) -> bytes:
    headers = getattr(response, "headers", None) or {}
    try:
        declared = int(headers.get("Content-Length") or 0)
    except (TypeError, ValueError):
        declared = 0
    if declared > MAX_BODY_BYTES:
        raise _TooLarge()
    body = bytearray()
    for chunk in response.iter_content(chunk_size=1024 * 1024):
        if chunk:
            body.extend(chunk)
            if len(body) > MAX_BODY_BYTES:
                raise _TooLarge()
    return bytes(body)


def requests_get(url: str, *, headers: Optional[Mapping[str, str]] = None,
                 timeout: float = REQUEST_TIMEOUT_S, want_body: bool = True) -> HttpResult:
    """기본 HTTP GET. 예외 문자열은 URL 을 담을 수 있어서 **예외 이름만** 남긴다."""
    import requests
    from core.backend_probe import request_timeout

    merged = {"accept": "application/json", **(dict(headers) if headers else {})}
    try:
        response = requests.get(url, headers=merged, timeout=request_timeout(url, timeout), stream=True)
    except requests.exceptions.RequestException as exc:
        return HttpResult(None, None, type(exc).__name__)
    try:
        if not want_body or response.status_code != 200:
            return HttpResult(response.status_code)
        try:
            return HttpResult(200, json.loads(_read_bounded(response).decode("utf-8")))
        except _TooLarge:
            return HttpResult(200, None, "response too large")
        except ValueError:  # JSONDecodeError · UnicodeDecodeError
            return HttpResult(200, None, "invalid JSON")
        except requests.exceptions.RequestException as exc:
            return HttpResult(None, None, type(exc).__name__)
    finally:
        response.close()


def cache_key(base_url: str) -> str:
    return str(base_url or "").strip().rstrip("/")


@dataclass(frozen=True)
class _ExtensionsEntry:
    fetched_at: float
    result: HttpResult


class ExtensionsCache:
    """``/sdapi/v1/extensions`` 응답을 WebUI 주소별로 오래 둔다 — 부작용 있는 GET 을 연결마다 보내지 않는다.

    연결 실패(status None)는 담지 않는다(다음 확인이 다시 묻는다). 그 밖의 응답은 TTL 동안 재사용하고
    ``invalidate`` (사용자의 수동 새로고침)만 먼저 버린다. 백엔드 변경 무효화(``invalidate_capabilities``)는
    이 캐시를 건드리지 않는다 — 앱은 연결할 때마다 URL 변경을 알리므로, 따라 버리면 매번 다시 묻는다.
    같은 주소의 동시 요청은 한 번으로 합치고, 무효화 전에 시작한 요청의 결과는 담지 않는다.
    """

    def __init__(self, *, ttl: float = EXTENSIONS_TTL_S, clock: Callable[[], float] = time.monotonic):
        self._ttl = float(ttl)
        self._clock = clock
        self._lock = threading.Lock()
        self._entries: Dict[str, _ExtensionsEntry] = {}
        self._key_locks: Dict[str, threading.Lock] = {}
        self._generations: Dict[str, int] = {}

    def _cached(self, key: str) -> Optional[HttpResult]:
        entry = self._entries.get(key)
        if entry is not None and self._clock() - entry.fetched_at < self._ttl:
            return entry.result
        return None

    def get(self, base_url: str, fetch: Callable[[], HttpResult]) -> HttpResult:
        """캐시가 신선하면 그 응답, 아니면 ``fetch()`` (**블로킹** — 워커에서만)."""
        key = cache_key(base_url)
        with self._lock:
            cached = self._cached(key)
            if cached is not None:
                return cached
            key_lock = self._key_locks.setdefault(key, threading.Lock())
        with key_lock:
            with self._lock:
                cached = self._cached(key)   # 기다리는 사이 다른 요청이 받아 왔으면 그것을 쓴다
                if cached is not None:
                    return cached
                generation = self._generations.get(key, 0)
            result = fetch()
            with self._lock:
                if result.status is not None and self._generations.get(key, 0) == generation:
                    self._entries[key] = _ExtensionsEntry(self._clock(), result)
            return result

    def invalidate(self, base_url: Optional[str] = None) -> None:
        """다음 확인이 다시 묻게 한다 (base_url 없으면 전부)."""
        with self._lock:
            keys = set(self._entries) | set(self._key_locks) if base_url is None else {cache_key(base_url)}
            for key in keys:
                self._entries.pop(key, None)
                self._generations[key] = self._generations.get(key, 0) + 1


def fetch_responses(base_url: str, *, http_get: Optional[HttpGet] = None,
                    extensions_cache: Optional[ExtensionsCache] = None) -> Dict[str, HttpResult]:
    """base_url 의 GET 응답 묶음. 실패도 HttpResult 로 담는다(예외를 올리지 않음).

    ``extensions_cache`` 가 있으면 ``/sdapi/v1/extensions`` (부작용 있는 GET)는 그 캐시를 거친다.
    없으면 매번 묻는다 — 테스트·일회성 도구용. 앱은 ``get_capabilities`` 로 공용 캐시를 쓴다.
    """
    get = http_get or requests_get
    base = str(base_url or "").strip().rstrip("/")

    def one(spec):
        path, headers, want_body = spec
        try:
            return path, get(base + path, headers=headers, timeout=REQUEST_TIMEOUT_S, want_body=want_body)
        except Exception as exc:  # 주입된 get 이 예외를 올려도 스냅샷은 만든다
            return path, HttpResult(None, None, type(exc).__name__)

    def extensions():
        fetch_one = lambda: one((EP_EXTENSIONS, None, True))[1]  # noqa: E731
        return EP_EXTENSIONS, (extensions_cache.get(base, fetch_one) if extensions_cache else fetch_one())

    with ThreadPoolExecutor(max_workers=6, thread_name_prefix="sam-extra-probe") as pool:
        futures = [pool.submit(one, spec) for spec in _PHASE1] + [pool.submit(extensions)]
        responses = dict(future.result() for future in futures)
    scripts, info = responses[EP_SCRIPTS], responses[EP_SCRIPT_INFO]
    if has_sam_extra_scripts(scripts.body if scripts.ok else None, info.body if info.ok else None):
        try:
            responses[EP_GRADIO_CONFIG] = get(base + EP_GRADIO_CONFIG, headers=None,
                                              timeout=CONFIG_TIMEOUT_S, want_body=True)
        except Exception as exc:
            responses[EP_GRADIO_CONFIG] = HttpResult(None, None, type(exc).__name__)
    return responses


def fetch_capabilities(base_url: str, *, http_get: Optional[HttpGet] = None,
                       extensions_cache: Optional[ExtensionsCache] = None) -> SamExtraCapabilities:
    """네트워크 수집 + 해석. 해석 중 예외도 'error' 스냅샷으로 바꾼다(워커가 죽지 않게)."""
    try:
        return build_capabilities(fetch_responses(base_url, http_get=http_get,
                                                  extensions_cache=extensions_cache))
    except Exception as exc:
        return unknown_capabilities(STATUS_ERROR, error=type(exc).__name__)


@dataclass(frozen=True)
class _Entry:
    sequence: int          # 이 주소에서 몇 번째로 시작한 수집인가 (시계 해상도와 무관하게 순서를 안다)
    finished_at: float
    capabilities: SamExtraCapabilities


class CapabilityCache:
    """WebUI 주소별 스냅샷 캐시. 같은 주소의 동시 요청은 한 번의 수집으로 합친다."""

    def __init__(self, fetch: Callable[[str], SamExtraCapabilities] = fetch_capabilities, *,
                 ttl: float = DEFAULT_TTL_S, failed_ttl: float = FAILED_TTL_S,
                 clock: Callable[[], float] = time.monotonic):
        self._fetch = fetch
        self._ttl = float(ttl)
        self._failed_ttl = float(failed_ttl)
        self._clock = clock
        self._lock = threading.Lock()
        self._entries: Dict[str, _Entry] = {}
        self._key_locks: Dict[str, threading.Lock] = {}
        self._generations: Dict[str, int] = {}
        self._started: Dict[str, int] = {}

    def _fresh(self, entry: Optional[_Entry], now: float) -> bool:
        if entry is None:
            return False
        ttl = self._ttl if entry.capabilities.status == STATUS_OK else self._failed_ttl
        return now - entry.finished_at < ttl

    def peek(self, base_url: str) -> Optional[SamExtraCapabilities]:
        """네트워크 없이 마지막 스냅샷(오래됐어도). GUI 스레드에서 불러도 된다."""
        with self._lock:
            entry = self._entries.get(cache_key(base_url))
        return entry.capabilities if entry else None

    def get(self, base_url: str, *, refresh: bool = False) -> SamExtraCapabilities:
        """스냅샷. 신선하면 캐시, 아니면 수집한다(**블로킹** — 워커에서만)."""
        key = cache_key(base_url)
        with self._lock:
            entry = self._entries.get(key)
            if not refresh and self._fresh(entry, self._clock()):
                return entry.capabilities
            ticket = self._started.get(key, 0)   # 이 요청 시점까지 시작된 수집 수
            key_lock = self._key_locks.setdefault(key, threading.Lock())
        with key_lock:
            with self._lock:
                entry = self._entries.get(key)
                # 기다리는 사이 '이 요청 뒤에 시작한' 수집이 끝났으면 그것을 쓴다(refresh 도 합친다).
                if entry is not None and entry.sequence > ticket:
                    return entry.capabilities
                if not refresh and self._fresh(entry, self._clock()):
                    return entry.capabilities
                generation = self._generations.get(key, 0)
                sequence = self._started[key] = self._started.get(key, 0) + 1
            capabilities = self._fetch(base_url)
            with self._lock:
                if self._generations.get(key, 0) == generation:
                    self._entries[key] = _Entry(sequence, self._clock(), capabilities)
            return capabilities

    def invalidate(self, base_url: Optional[str] = None) -> None:
        """캐시를 버린다. 진행 중이던 수집 결과도 저장하지 않는다(base_url 없으면 전부)."""
        with self._lock:
            keys = list(self._entries) + list(self._key_locks) if base_url is None else [cache_key(base_url)]
            for key in set(keys):
                self._entries.pop(key, None)
                self._generations[key] = self._generations.get(key, 0) + 1


_EXTENSIONS_CACHE = ExtensionsCache()


def _fetch_with_shared_extensions(base_url: str) -> SamExtraCapabilities:
    return fetch_capabilities(base_url, extensions_cache=_EXTENSIONS_CACHE)


_DEFAULT_CACHE = CapabilityCache(_fetch_with_shared_extensions)


def get_capabilities(base_url: str, *, refresh: bool = False) -> SamExtraCapabilities:
    """프로세스 공용 캐시로 스냅샷을 얻는다 (워커 전용).

    ``refresh=True`` 는 스냅샷 캐시만 무시한다 — ``/sdapi/v1/extensions`` 는 공용 ``ExtensionsCache``
    에서 온다. 그것까지 다시 물으려면 먼저 ``invalidate_extensions`` 를 부른다(수동 새로고침만).
    """
    return _DEFAULT_CACHE.get(base_url, refresh=refresh)


def peek_capabilities(base_url: str) -> Optional[SamExtraCapabilities]:
    return _DEFAULT_CACHE.peek(base_url)


def invalidate_capabilities(base_url: Optional[str] = None) -> None:
    """스냅샷 캐시를 버린다(백엔드 변경). 확장 목록 캐시는 그대로 둔다 — ``ExtensionsCache`` 참고."""
    _DEFAULT_CACHE.invalidate(base_url)


def invalidate_extensions(base_url: Optional[str] = None) -> None:
    """다음 확인이 ``/sdapi/v1/extensions`` 를 다시 묻게 한다 — 사용자의 수동 새로고침에서만 부른다."""
    _EXTENSIONS_CACHE.invalidate(base_url)


__all__ = [
    "CapabilityCache", "DEFAULT_TTL_S", "EXTENSIONS_TTL_S", "ExtensionsCache", "FAILED_TTL_S",
    "cache_key", "fetch_capabilities", "fetch_responses", "get_capabilities",
    "invalidate_capabilities", "invalidate_extensions", "peek_capabilities",
    "requests_get", "STATUS_OK",
]
