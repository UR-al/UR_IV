"""앱 시작 시 managed runtime 자동기동 — Qt 없는 판정 + Studio ``runtime.execute`` 요청.

예전엔 이 경로만 레거시 ``VueBridge.runBackendRuntimeOperation`` 을 탔다. 그 실행기는
``backendRuntimeEvent`` 만 내보내고 Studio journal 에는 아무것도 남기지 않아서, 자동기동
도중 Settings 를 처음 열면 busy=true 스냅샷만 받고 완료 이벤트를 영영 못 받아 두 엔진의
런타임 버튼이 앱 재시작 전까지 비활성으로 남았다. 이제 Settings 와 같은 Studio 작업으로
시작한다 — 진행/완료는 journal(Settings)과 DesktopNativeHost → backendRuntimeEvent
(generator_main._on_backend_runtime_event)로 함께 흐른다.
"""

from __future__ import annotations

import uuid
from collections.abc import Mapping
from typing import Any

AUTOSTART_ENGINES = ('forge', 'comfyui')
DEFAULT_REJECT_MESSAGE = 'managed backend 시작 요청이 거부되었습니다'


def pick_autostart_engine(snapshot: Any) -> str:
    """auto-start 가 켜져 있고 설치된 엔진 — 활성 엔진을 우선한다. 없으면 ''.

    activeEngine 은 BackendRuntimeManager 가 이미 정규화한다('forge_neo' → 'forge').
    설치 여부를 먼저 보므로 이 판정은 다운로드·설치를 유발하지 않는다.
    """
    if not isinstance(snapshot, Mapping):
        return ''
    runtimes = snapshot.get('engines')
    runtimes = runtimes if isinstance(runtimes, Mapping) else {}
    active = str(snapshot.get('activeEngine') or '')
    candidates = [
        engine for engine in AUTOSTART_ENGINES
        if isinstance(runtimes.get(engine), Mapping)
        and bool(runtimes[engine].get('autoStart', False))
    ]
    kind = active if active in candidates else (candidates[0] if candidates else '')
    runtime = runtimes.get(kind)
    if kind in AUTOSTART_ENGINES and isinstance(runtime, Mapping) and bool(runtime.get('installed', False)):
        return kind
    return ''


def autostart_request(engine: str, request_id: str | None = None) -> dict[str, Any]:
    """Studio ``runtime.execute`` 요청 envelope (startup=True — 설치를 유발하지 않는 시작)."""
    from core.studio_application import PROTOCOL_VERSION

    return {
        'version': PROTOCOL_VERSION,
        'requestId': request_id or f'startup-autostart-{uuid.uuid4().hex}',
        'operation': 'runtime.execute',
        'input': {'engine': engine, 'action': 'start', 'payload': {'startup': True}},
    }


def request_runtime_autostart(application: Any, context: Any, engine: str) -> tuple[bool, str]:
    """Studio 로 자동기동을 요청한다. (accepted, 거부 사유)."""
    if application is None or context is None:
        return False, 'Studio application이 준비되지 않았습니다'
    reply = application.invoke(context, autostart_request(engine))
    if isinstance(reply, Mapping) and reply.get('status') == 'accepted':
        return True, ''
    error = reply.get('error') if isinstance(reply, Mapping) else None
    if isinstance(error, Mapping):
        message = str(error.get('message') or error.get('code') or '')
    else:
        message = str(error or '')
    return False, message or DEFAULT_REJECT_MESSAGE
