# ui/vae_degrid_ui.py
"""VAE DeGrid 카드 ↔ 위젯 프록시 ↔ 샘플링 블록 — 얇은 Qt 접착층.

판정·정규화는 core/vae_degrid 가 한다. 여기서는

- ``init_degrid_proxies(bridge)``  ``_degrid_<key>`` LineEditProxy 6개(값은 모두 문자열 — DoRA·가이던스 프록시와 같은
  패턴), 초기값은 앱 기본값(``APP_DEFAULTS`` = 확장 기본값: 꺼짐·자동·Full·1·512)
- ``contribute(host, ctx)``      ui/sampling_blocks.CONTRIBUTORS 의 한 줄(DoRA 뒤). 메인 체인(t2i·i2i)과 보조 패스
  봉투(aux)가 같은 기여자를 거친다 — aux 는 t2i 와 같은 블록이고 알림이 없다(봉투 = 메인 체인, T15). 보조 패스에
  실제로 보내지 않는 것은 core/alwayson_propagation 의 DeGrid 행(``passes=()``)이 정한다. 출처는 늘 사용자 값.
- ``get_settings``/``apply_saved_settings``  prompt_settings.json·생성 프리셋(``vae_degrid_settings``). 옛 설정 파일에
  키가 없으면 앱 기본값(꺼짐)으로 시작한다 — 동작이 바뀌지 않으므로 안내 토스트는 없다.
- ``push_comfy_models(host, choices)``  ComfyUI 노드의 모델 선택지(Forge 식 이름)를 카드로(``_degrid_model`` 위젯 속성
  ``comfyModels``). None = Forge·모름·옛 팩 — 카드는 Forge 기능 스냅샷 ``choices.degrid_models`` 를 쓴다.
- ``refresh_comfy_models(host)``  카드 ↻ 의 ComfyUI 쪽 — 데몬 스레드에서 /object_info 를 새로 받아 같은 속성으로 다시
  보낸다(새 브리지 이름 없음: Vue 는 기존 ``sam_extra_capabilities_get {refresh: true}`` 를 보내고,
  ui/sam_extra_capabilities_actions 가 ComfyUI 백엔드면 이것을 부른다). 받은 문서는 컴파일 스냅샷도 새로 한다.

``refresh_comfy_models`` 의 워커 말고는 모두 GUI 스레드에서 부른다(위젯을 읽는다).
"""
from __future__ import annotations

import logging
import threading
from typing import Any, Iterable, Mapping, Optional

from core import vae_degrid as vdg

logger = logging.getLogger(__name__)

_WIDGETS_ATTR = "degrid_widgets"
# 목록을 보낸 차례 — 연결·끊김(push_comfy_models)이 올리면 그 전에 시작한 ↻ 워커의 결과는 버린다(옛 백엔드 목록 방지)
_push_lock = threading.Lock()
_push_serial = 0
COMFY_MODELS_PROPERTY = "comfyModels"   # _degrid_model 위젯 속성 — frontend/src/utils/vaeDegrid.ts COMFY_MODELS_PROPERTY


def init_degrid_proxies(bridge) -> dict:
    """``_degrid_<key>`` 프록시 6개 — 앱 기본값을 문자열로 심어 둔다(Vue 가 값을 보내기 전에도 같은 블록이 나가게)."""
    from ui.widget_proxies import LineEditProxy

    values = vdg.widget_values(vdg.APP_DEFAULTS)
    widgets = {}
    for key in vdg.WIDGET_KEYS:
        proxy = LineEditProxy(bridge, vdg.widget_id(key))
        proxy.setText(values[key])
        widgets[key] = proxy
    return widgets


def _widget_text(proxy: Any) -> Any:
    text = getattr(proxy, "text", None)
    try:
        return text() if callable(text) else None
    except Exception:
        return None


def _raw(host) -> Optional[dict]:
    widgets = getattr(host, _WIDGETS_ATTR, None)
    if not isinstance(widgets, Mapping):
        return None
    return {key: _widget_text(widgets.get(key)) for key in vdg.WIDGET_KEYS}


def get_settings(host) -> dict:
    """저장용 위젯 문자열(키 6개). 위젯이 없는 호스트는 빈 dict."""
    raw = _raw(host)
    return {} if raw is None else vdg.widget_values(vdg.parse_settings(raw))


def current_settings(host) -> Optional[vdg.DegridSettings]:
    """카드의 지금 값. 위젯이 없는 호스트(테스트 더블·옛 호스트)는 None — 블록을 만들지 않는다."""
    raw = _raw(host)
    return None if raw is None else vdg.parse_settings(raw)


def set_settings(host, raw: Optional[Mapping]) -> None:
    """저장값(또는 None = 앱 기본값)을 위젯에 쓴다. 모르는 칸은 앱 기본값."""
    widgets = getattr(host, _WIDGETS_ATTR, None)
    if not isinstance(widgets, Mapping):
        return
    values = vdg.widget_values(vdg.parse_settings(raw if isinstance(raw, Mapping) else None))
    for key, value in values.items():
        proxy = widgets.get(key)
        if proxy is not None and hasattr(proxy, "setText"):
            proxy.setText(value)


def apply_saved_settings(host, settings: Mapping, *, only_present: bool) -> None:
    """``apply_generation_settings`` 의 VAE DeGrid 칸 — 키가 있으면 복원.

    키가 없을 때: 프리셋(``only_present=True``)은 그대로 두고, load_settings(옛 설정 파일)는 앱 기본값(꺼짐)으로
    채운다. 앱 기본값이 꺼짐이라 동작이 바뀌지 않으므로 안내하지 않는다."""
    if not hasattr(host, _WIDGETS_ATTR):
        return
    raw = settings.get(vdg.SETTINGS_KEY) if isinstance(settings, Mapping) else None
    if isinstance(raw, Mapping):
        set_settings(host, raw)
    elif not only_present:
        set_settings(host, None)


def contribute(host, ctx) -> Any:
    """ui/sampling_blocks 기여자 — VAE DeGrid 블록 하나(또는 없음). 알림은 없다(결과 infotext 가 정답).

    ctx.target: 't2i'(T2I 계열 전부 — 대기열·XYZ·시드 탐색·채팅·만화 컷) · 'i2i'(I2I·인페인트·채팅 이미지 편집 — 카드의
    'I2I·인페인트에도 적용' 토글) · 'aux'(보조 패스 봉투 — t2i 와 같은 블록, 보내지는 않는다: passes=())."""
    from ui.sampling_blocks import Contribution

    settings = current_settings(host)
    if settings is None:
        return Contribution()
    result = vdg.plan(settings, target=ctx.target, backend=ctx.backend, capabilities=ctx.capabilities)
    contribution = Contribution()
    if result.block is not None:
        contribution.add(vdg.SCRIPT_NAME, result.block, provenance=result.provenance)
        # 보낼지는 뒤의 게이트(core/alwayson_propagation)가 정한다 — 여기서는 준비만 기록한다
        logger.debug("VAE DeGrid 블록 준비(%s): %s", ctx.target, vdg.describe(settings))
    return contribution


def _clean_choices(choices: Optional[Iterable[Any]]) -> Optional[list]:
    return None if choices is None else [
        str(item).strip() for item in choices
        if str(item or "").strip() and str(item).strip() != vdg.NONE_NAME]


def _push(host, value: Optional[list]) -> None:
    bridge = getattr(host, "vue_bridge", None)
    push = getattr(bridge, "pushWidgetProperty", None)
    if not callable(push):
        return
    try:
        push(vdg.widget_id("model"), COMFY_MODELS_PROPERTY, value)
    except RuntimeError:
        pass   # 종료 중 QObject 가 이미 사라졌다


def push_comfy_models(host, choices: Optional[Iterable[Any]]) -> None:
    """ComfyUI 노드의 모델 선택지(``core/vae_degrid.comfy_model_choices`` — Forge 식 이름)를 카드로(``_degrid_model``
    위젯 속성 ``comfyModels``). None = Forge·모름·옛 팩(노드 없음) — 카드는 Forge 기능 스냅샷 규칙을 쓴다. 백엔드를
    바꾸거나 연결이 끊기면 None 으로 지운다(옛 목록을 남기지 않는다). 진행 중인 ↻ 워커의 결과는 버린다."""
    global _push_serial
    value = _clean_choices(choices)
    with _push_lock:
        _push_serial += 1
        _push(host, value)


def refresh_comfy_models(host, backend: Any = None, *, start: bool = True) -> Optional[threading.Thread]:
    """카드 ↻ (ComfyUI) — 데몬 스레드에서 지금 백엔드의 /object_info 를 새로 받아 ``comfyModels`` 를 다시 보낸다.

    GUI 스레드는 네트워크를 기다리지 않는다. 그사이 연결·끊김이 목록을 바꿨거나(``push_comfy_models``) 활성 백엔드가
    바뀌었으면 결과를 버린다. 받지 못하면(연결 오류) 지난 목록을 그대로 둔다 — 연결 실패는 연결 훅이 따로 지운다.
    ``get_object_info`` 는 받은 문서로 컴파일 스냅샷도 새로 하므로 다음 생성도 같은 목록으로 컴파일한다."""
    from backends import get_backend

    target = backend if backend is not None else get_backend()
    fetch = getattr(target, "get_object_info", None)
    if not callable(fetch):
        return None
    with _push_lock:
        serial = _push_serial

    def work():
        try:
            document = fetch()
        except Exception as exc:
            logger.info("[DeGrid] ComfyUI 모델 목록 다시 받기 실패(지난 목록 유지): %s", exc)
            return
        choices = vdg.comfy_model_choices(document if isinstance(document, Mapping) else None)
        with _push_lock:
            if serial != _push_serial or get_backend() is not target:
                return   # 그사이 연결·끊김·백엔드 전환 — 옛 백엔드 목록을 보내지 않는다
            _push(host, _clean_choices(choices))
        logger.info("[DeGrid] ComfyUI 모델 목록 다시 받음: %s",
                    "노드 없음(팩 업데이트 필요)" if choices is None else f"{len(choices)}개")

    thread = threading.Thread(target=work, daemon=True, name="degrid-comfy-models")
    if start:
        thread.start()
    return thread


__all__ = [
    "COMFY_MODELS_PROPERTY", "apply_saved_settings", "contribute", "current_settings", "get_settings",
    "init_degrid_proxies", "push_comfy_models", "refresh_comfy_models", "set_settings",
]
