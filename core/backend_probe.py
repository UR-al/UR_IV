"""백엔드 health probe — 시작 게이트·비상 선택 창·관리형 health 루프가 함께 쓴다.

Qt 도 core.backend_runtime 도 import 하지 않는다(테스트가 core.backend_runtime 을 가짜 모듈로
바꿔 끼워도 이 모듈은 그대로 쓸 수 있게). ``requests`` 는 호출할 때만 불러온다.

**왜 루프백을 따로 다루나**: Windows 는 닫힌 127.0.0.1 포트로의 connect 를 SYN 재전송 때문에
약 2초 뒤에야 거부한다. ``localhost`` 는 ::1 → 127.0.0.1 을 차례로 시도해 약 4초가 걸린다.
두 백엔드가 모두 꺼져 있으면 시작 게이트가 그동안 '확인 중'에 머물렀다. 루프백에 붙는
connect 는 살아 있는 리스너면 즉시 끝나므로 connect 타임아웃만 짧게(0.3초) 주고, 응답(read)
타임아웃은 그대로 둔다. 사용자가 입력한 LAN·터널·원격 주소는 RTT 가 크므로 예전처럼
전체 타임아웃을 그대로 쓴다.

**HTTPS 는 호스트를 바꾸지 않는다**: TLS 는 URL 의 호스트로 SNI·인증서 이름을 확인하므로
``https://localhost`` 를 127.0.0.1/::1 로 바꾸면 ``localhost`` 전용 인증서가 IP 불일치로
실패해, 멀쩡한 백엔드가 '응답 없음'으로 보였다. HTTPS 루프백은 원래 URL 그대로 두드리고
짧은 connect 타임아웃만 쓴다 — urllib3 는 TLS 핸드셰이크도 connect 타임아웃 안에서 하므로
(connectionpool ``_validate_conn``) 0.3초가 핸드셰이크까지 덮는다(루프백이면 충분하다).
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from typing import Callable, Mapping, Optional
from urllib.parse import urlsplit, urlunsplit

#: 엔진별 health 경로 — core.backend_runtime.ENGINE_DEFINITIONS 도 이 값을 쓴다.
HEALTH_PATHS = {
    "forge": "/sdapi/v1/samplers",
    "webui": "/sdapi/v1/samplers",
    "comfyui": "/system_stats",
}
DEFAULT_TIMEOUT = 2.0
LOOPBACK_CONNECT_TIMEOUT = 0.3
_LOOPBACK_HOSTS = frozenset({"127.0.0.1", "::1", "localhost"})

Getter = Callable[..., object]


def _with_host(parts, host: str) -> str:
    port = f":{parts.port}" if parts.port else ""
    userinfo = ""
    if parts.username:
        userinfo = parts.username + (f":{parts.password}" if parts.password else "") + "@"
    shown = f"[{host}]" if ":" in host else host
    return urlunsplit((parts.scheme, f"{userinfo}{shown}{port}", parts.path, parts.query, parts.fragment))


def probe_targets(url: str) -> tuple[list[str], bool]:
    """(실제로 두드릴 URL 목록, 루프백 여부).

    ``http://localhost`` 는 IPv4·IPv6 루프백을 둘 다 후보로 둔다 — 순차 fallback(약 4초) 대신
    병렬로 확인하고, ::1 에만 리슨하는 서버도 놓치지 않는다. ``https://localhost`` 는 TLS 가
    URL 호스트로 인증서를 확인하므로 바꾸지 않는다(루프백 여부만 True).
    """
    text = str(url or "").strip()
    if not text:
        return [], False
    try:
        parts = urlsplit(text)
        host = (parts.hostname or "").lower()
        parts.port  # noqa: B018 — 잘못된 포트면 ValueError
    except ValueError:
        return [text], False
    if host not in _LOOPBACK_HOSTS:
        return [text], False
    if host == "localhost" and parts.scheme.lower() == "http":
        return [_with_host(parts, "127.0.0.1"), _with_host(parts, "::1")], True
    return [text], True


def request_timeout(url: str, timeout: float):
    """requests 용 timeout — 루프백이면 (connect 0.3초, read ``timeout``), 아니면 ``timeout``.

    꺼진 로컬 백엔드는 Windows 에서 connect 거부까지 약 2초(localhost 는 약 4초)가 걸린다.
    살아 있는 루프백 리스너는 connect 가 즉시 끝나므로 connect 만 짧게 끊어도 안전하다.
    """
    _targets, loopback = probe_targets(url)
    if loopback:
        return (min(LOOPBACK_CONNECT_TIMEOUT, float(timeout)), float(timeout))
    return float(timeout)


def _get_ok(get: Getter, url: str, timeout) -> bool:
    try:
        response = get(url, timeout=timeout)
        return getattr(response, "status_code", None) == 200
    except Exception:
        return False


def probe_url(url: str, path: str, *, timeout: float = DEFAULT_TIMEOUT,
              get: Optional[Getter] = None) -> bool:
    """``url + path`` 가 HTTP 200 을 돌려주면 True. 예외는 모두 False."""
    targets, _loopback = probe_targets(url)
    if not targets:
        return False
    if get is None:
        import requests

        get = requests.get
    effective = request_timeout(url, timeout)
    endpoints = [f"{target.rstrip('/')}{path}" for target in targets]
    if len(endpoints) == 1:
        return _get_ok(get, endpoints[0], effective)
    with ThreadPoolExecutor(max_workers=len(endpoints)) as pool:
        return any(pool.map(lambda endpoint: _get_ok(get, endpoint, effective), endpoints))


def probe_health(url: str, engine: str, *, timeout: float = DEFAULT_TIMEOUT,
                 get: Optional[Getter] = None) -> bool:
    return probe_url(url, HEALTH_PATHS[engine], timeout=timeout, get=get)


def probe_backends(urls: Mapping[str, str], *, timeout: float = DEFAULT_TIMEOUT,
                   get: Optional[Getter] = None) -> dict[str, bool]:
    """``{engine: url}`` 를 병렬로 확인한다 — 두 백엔드가 꺼져 있어도 한 번의 대기로 끝난다."""
    items = [(engine, url) for engine, url in urls.items()]
    if not items:
        return {}
    with ThreadPoolExecutor(max_workers=len(items)) as pool:
        results = pool.map(
            lambda item: probe_health(item[1], item[0], timeout=timeout, get=get), items
        )
        return {engine: bool(ok) for (engine, _url), ok in zip(items, results)}


__all__ = [
    "DEFAULT_TIMEOUT",
    "HEALTH_PATHS",
    "LOOPBACK_CONNECT_TIMEOUT",
    "probe_backends",
    "probe_health",
    "probe_targets",
    "probe_url",
    "request_timeout",
]
