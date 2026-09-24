"""로컬 TCP 포트가 비어 있는지 판정 — 관리형 백엔드 포트 선택용 (Qt 없음).

Windows 의 ``SO_REUSEADDR`` 는 POSIX 와 뜻이 다르다. 이미 ``SO_REUSEADDR`` 로 리슨 중인
소켓(예: Generation API 의 ``ThreadingHTTPServer.allow_reuse_address``)과 포트를 '공유'
시키므로, 그 옵션으로 bind 를 시험하면 쓰이는 포트를 빈 포트로 오판한다(실측). 그러면
Forge 가 모델을 다 올린 뒤에야 bind 에 실패하거나 health probe 가 엉뚱한 서버로 간다.

그래서 Windows 에서는 ``SO_EXCLUSIVEADDRUSE`` 로 시험한다 — 같은 포트를 쥔 어떤 소켓과도
겹치면 실패한다. 또 특정 주소(127.0.0.1) bind 는 와일드카드(0.0.0.0) 리스너와 겹쳐도
성공하므로 두 주소를 모두 시험해 둘 다 비어야 True 다.

POSIX 는 기존대로 ``SO_REUSEADDR`` 로 시험한다 — 거기서는 리스너와 겹치지 않고 TIME_WAIT
잔여 연결만 허용하므로, 방금 끈 엔진의 포트를 불필요하게 건너뛰지 않는다.
"""
from __future__ import annotations

import os
import socket

_EXCLUSIVE = getattr(socket, "SO_EXCLUSIVEADDRUSE", None)


def _family(host: str) -> int:
    return socket.AF_INET6 if ":" in str(host) else socket.AF_INET


def _wildcard(family: int) -> str:
    return "::" if family == socket.AF_INET6 else "0.0.0.0"


def _try_bind(family: int, host: str, port: int, *, exclusive: bool) -> bool:
    try:
        with socket.socket(family, socket.SOCK_STREAM) as sock:
            if exclusive and _EXCLUSIVE is not None:
                sock.setsockopt(socket.SOL_SOCKET, _EXCLUSIVE, 1)
            else:
                sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            sock.bind((host, int(port)))
        return True
    except OSError:
        return False


def port_available(host: str, port: int, *, windows: bool | None = None) -> bool:
    """``host:port`` 에 새 리스너를 열 수 있으면 True.

    ``windows`` 는 테스트 주입용 — 기본은 ``os.name == 'nt'``.
    """
    family = _family(host)
    if windows is None:
        windows = os.name == "nt"
    if windows and _EXCLUSIVE is not None:
        addresses = dict.fromkeys((str(host), _wildcard(family)))
        return all(_try_bind(family, address, port, exclusive=True) for address in addresses)
    return _try_bind(family, str(host), port, exclusive=False)


def exclusive_bind_supported() -> bool:
    """이 플랫폼이 ``SO_EXCLUSIVEADDRUSE`` 를 제공하는지(Windows)."""
    return os.name == "nt" and _EXCLUSIVE is not None


def apply_exclusive_bind(sock: socket.socket) -> None:
    """Windows 서버 소켓이 다른 ``SO_REUSEADDR`` 소켓과 포트를 공유하지 못하게 한다."""
    if exclusive_bind_supported():
        sock.setsockopt(socket.SOL_SOCKET, _EXCLUSIVE, 1)


__all__ = ["apply_exclusive_bind", "exclusive_bind_supported", "port_available"]
