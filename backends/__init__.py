# backends/__init__.py
"""백엔드 팩토리 및 전역 관리"""
from enum import Enum
from typing import Optional

# BackendInfo·GenerationResult 는 재수출하지 않는다 — 사용처는 backends.base 에서 직접 가져온다.
from backends.base import AbstractBackend


class BackendType(Enum):
    WEBUI = "webui"
    COMFYUI = "comfyui"


_current_backend: Optional[AbstractBackend] = None
_current_type: BackendType = BackendType.WEBUI


def get_backend() -> AbstractBackend:
    """현재 활성 백엔드 반환 (없으면 WebUI 기본 생성)"""
    global _current_backend
    if _current_backend is None:
        from backends.webui_backend import WebUIBackend
        import config
        _current_backend = WebUIBackend(config.WEBUI_API_URL)
    return _current_backend


def set_backend(backend_type: BackendType, api_url: str):
    """백엔드 전환"""
    global _current_backend, _current_type
    prev_type = _current_type
    _current_type = backend_type
    api_url = api_url.strip()

    if backend_type == BackendType.WEBUI:
        from backends.webui_backend import WebUIBackend
        _current_backend = WebUIBackend(api_url)
        import config
        config.WEBUI_API_URL = api_url
    elif backend_type == BackendType.COMFYUI:
        from backends.comfyui_backend import ComfyUIBackend
        _current_backend = ComfyUIBackend(api_url)
        import config
        config.COMFYUI_API_URL = api_url

    # AppContext 이벤트 발행 — ModeAwareMixin 등 구독자에게 알림.
    # 실패해도 백엔드 전환 자체는 영향 없도록 격리.
    try:
        from core.app_context import get_context, Events
        ctx = get_context()
        # 타입 변경 시 BACKEND_CHANGED, URL은 항상 별도 알림 (URL만 바뀌는 경우 포함)
        if prev_type != backend_type:
            ctx.publish(Events.BACKEND_CHANGED, backend_type)
        ctx.publish(Events.BACKEND_URL_CHANGED, api_url)
    except Exception:
        # AppContext 임포트 실패는 무시 — 백엔드 전환은 정상 동작 유지
        pass


_BACKEND_CLASS_NAMES = {
    BackendType.WEBUI: "WebUIBackend",
    BackendType.COMFYUI: "ComfyUIBackend",
}


def _normalized_url(api_url) -> str:
    return str(api_url or "").strip().rstrip("/")


def is_active_backend(backend_type: BackendType, api_url: str) -> bool:
    """이미 만든 백엔드가 같은 타입·같은 URL인가 — 설정 복원이 어댑터를 다시 만들지 않게.

    ``get_backend()`` 는 미설정이면 WebUI 를 즉석에서 만들어 버리므로 쓰지 않는다: 아직 아무것도
    만들지 않았으면 False 여야 첫 복원이 ``set_backend`` 로 인스턴스와 이벤트를 만든다.
    명시적 재연결(연결 버튼·managed runtime)은 이 검사를 거치지 않고 ``set_backend`` 를 불러
    ComfyUI node-pack preflight 를 다시 돌린다.
    """
    current = _current_backend
    if current is None or _current_type != backend_type:
        return False
    expected = _BACKEND_CLASS_NAMES.get(backend_type)
    if not any(cls.__name__ == expected for cls in type(current).__mro__):
        return False
    return _normalized_url(getattr(current, "api_url", "")) == _normalized_url(api_url)


def get_backend_type() -> BackendType:
    """현재 백엔드 타입 반환 — 바꾸는 길은 set_backend 하나뿐이다(타입과 인스턴스를 함께 바꾼다)."""
    return _current_type
