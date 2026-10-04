# ui/extra_samplers_ui.py
"""Extra Samplers ER SDE 값 칸(파라미터 카드의 Sampler/Scheduler 줄 아래 — 계획 R8, Vue 패널은 A-XSa5) ↔ 위젯 프록시 ↔
샘플링 블록 — 얇은 Qt 접착층.

판정·정규화는 core/extra_samplers 가 한다. 여기서는

- ``init_extra_samplers_proxies(bridge)``  ``_xsamp_<인자>`` LineEditProxy 5개(``core/extra_samplers.WIDGET_KEYS`` = 확장
  ARG_NAMES — 값은 모두 문자열, DeGrid·Extra Schedulers 프록시와 같은 패턴: 차수 '3', η '1', 잡음 구간 체크 '0'/'1', 시작 '0.2',
  끝 '0.8'). 초기값은 앱 기본값(``APP_DEFAULTS`` = 확장 기본값 — 블록이 없던 지금까지의 Forge 결과와 같다)
- ``contribute(host, ctx)``      ui/sampling_blocks.CONTRIBUTORS 의 마지막 줄(Extra Schedulers 뒤 — 계획 R7). 지금 화면에서
  ER SDE (Reverse-time)·(ODE)·(Tunable) 을 쓰는 패스(``pass_values`` → ``core/extra_samplers.passes_from_widgets``)를 ``plan``
  에 넘긴다 — 스냅샷이 스크립트를 확인했으면 늘, 확인 못 했으면 그런 패스가 있을 때만 싣는다(계획 X5). 출처는 ``plan`` 이
  값으로 정한다(기본값이면 앱 기본값). 보낼지는 뒤의 게이트(core/alwayson_propagation)가 정한다. aux(보조 패스 봉투)는 t2i 와
  같은 블록이고 모든 보조 img2img 로 간다(블록 없는 요청은 확장 ui() 기본값으로 돈다). ComfyUI 는 팩 1.7.0 단계(계획 C6)
  전이라 ``plan`` 이 'comfy_pending' 으로 싣지 않는다(R16 — 그래프가 바이트 단위로 그대로). 블록은 늘 위치 인자 다섯이다
  (계획 X10 — 인자 2개인 0.31·0.32 는 뒤 셋을 버린다: 카드와 결과 알림이 말한다)
- ``get_settings``/``set_settings``/``apply_saved_settings``  prompt_settings.json·생성 프리셋(``extra_samplers_settings``,
  키 = ARG_NAMES 다섯). 옛 설정 파일에 키가 없거나 저장값에 칸이 빠졌으면(v0.33.0 전 계획의 두 칸 — 차수·η) 그 칸은 앱
  기본값 — 확장 기본값과 같아 동작이 바뀌지 않으므로 안내하지 않는다.
- ``apply_infotext(host, params)``  붙여 넣은 infotext(PNG Info·'이 설정으로 생성' — 계획 A-R1 이 연결)의 값을 칸에
  되살린다(``core/extra_samplers.from_infotext`` — 확장 paste_* 다섯처럼 늘 다섯 칸을 정한다: 키가 없으면 기본값, ER SDE
  (Reverse-time) 이미지의 Forge 'Eta' 는 η 로 읽는다).

모두 GUI 스레드에서 부른다(위젯을 읽는다).
"""
from __future__ import annotations

import logging
from typing import Any, Mapping, Optional

from core import extra_samplers as xsa

logger = logging.getLogger(__name__)

_WIDGETS_ATTR = "extra_samplers_widgets"


def _pass_sources() -> dict:
    """패스 판정 위젯 id → host 에서 읽는 곳(속성 이름, 위젯 dict 키|None, 'check'|'text') — ui/generator_ui_setup 의
    프록시(ADetailer 슬롯 위젯 dict 는 ``s<번호>_widgets``, SAM3 는 ``sam3_widgets``)."""
    sources = {
        xsa.MAIN_SAMPLER_WIDGET: ("sampler_combo", None, "text"),
        xsa.HIRES_GROUP_WIDGET: ("hires_options_group", None, "check"),
        xsa.HIRES_SAMPLER_WIDGET: ("hires_sampler_combo", None, "text"),
        xsa.ADETAILER_GROUP_WIDGET: ("adetailer_group", None, "check"),
        xsa.SAM3_GROUP_WIDGET: ("sam3_group", None, "check"),
        xsa.SAM3_MODE_WIDGET: ("sam3_widgets", "mode", "text"),
        xsa.SAM3_USE_SAMPLER_WIDGET: ("sam3_widgets", "use_sampler_check", "check"),
        xsa.SAM3_SAMPLER_WIDGET: ("sam3_widgets", "sampler", "text"),
    }
    for index, enabled, use_sampler, sampler in xsa.ADETAILER_SLOT_WIDGETS:
        sources[enabled] = (enabled, None, "check")
        sources[use_sampler] = (f"s{index}_widgets", "use_sampler_check", "check")
        sources[sampler] = (f"s{index}_widgets", "sampler_combo", "text")
    return sources


_PASS_SOURCES = _pass_sources()


def init_extra_samplers_proxies(bridge) -> dict:
    """``_xsamp_<인자>`` 프록시 5개 — 앱 기본값을 문자열로 심어 둔다(Vue 가 값을 보내기 전에도 같은 블록이 나가게)."""
    from ui.widget_proxies import LineEditProxy

    values = xsa.widget_values(xsa.APP_DEFAULTS)
    widgets = {}
    for key in xsa.WIDGET_KEYS:
        proxy = LineEditProxy(bridge, xsa.widget_id(key))
        proxy.setText(values[key])
        widgets[key] = proxy
    return widgets


def _call(widget: Any, name: str) -> Any:
    method = getattr(widget, name, None)
    try:
        return method() if callable(method) else None
    except Exception:
        return None


def _raw(host) -> Optional[dict]:
    widgets = getattr(host, _WIDGETS_ATTR, None)
    if not isinstance(widgets, Mapping):
        return None
    return {key: _call(widgets.get(key), "text") for key in xsa.WIDGET_KEYS}


def get_settings(host) -> dict:
    """저장용 위젯 문자열(키 5개). 위젯이 없는 호스트는 빈 dict."""
    raw = _raw(host)
    return {} if raw is None else xsa.widget_values(xsa.parse_settings(raw))


def current_settings(host) -> Optional[xsa.ExtraSamplersSettings]:
    """칸의 지금 값. 위젯이 없는 호스트(테스트 더블·옛 호스트)는 None — 블록을 만들지 않는다."""
    raw = _raw(host)
    return None if raw is None else xsa.parse_settings(raw)


def set_settings(host, raw: Optional[Mapping]) -> None:
    """저장값(또는 None = 앱 기본값)을 위젯에 쓴다. 없는·읽을 수 없는 칸은 앱 기본값."""
    widgets = getattr(host, _WIDGETS_ATTR, None)
    if not isinstance(widgets, Mapping):
        return
    values = xsa.widget_values(xsa.parse_settings(raw if isinstance(raw, Mapping) else None))
    for key, value in values.items():
        proxy = widgets.get(key)
        if proxy is not None and hasattr(proxy, "setText"):
            proxy.setText(value)


def apply_saved_settings(host, settings: Mapping, *, only_present: bool) -> None:
    """``apply_generation_settings`` 의 Extra Samplers 칸 — 키가 있으면 복원(빠진 칸은 앱 기본값).

    키가 없을 때: 프리셋(``only_present=True``)은 그대로 두고, load_settings(옛 설정 파일)는 앱 기본값으로 채운다. 앱
    기본값이 확장 기본값과 같아(블록이 없던 Forge 요청과 같은 값) 동작이 바뀌지 않으므로 안내하지 않는다."""
    if not hasattr(host, _WIDGETS_ATTR):
        return
    raw = settings.get(xsa.SETTINGS_KEY) if isinstance(settings, Mapping) else None
    if isinstance(raw, Mapping):
        set_settings(host, raw)
    elif not only_present:
        set_settings(host, None)


def apply_infotext(host, params: Any) -> Optional[xsa.ExtraSamplersSettings]:
    """붙여 넣은 infotext → 칸(``core/extra_samplers.from_infotext`` — 확장 paste_* 다섯의 규칙: 늘 다섯 칸을 정한다,
    키가 없으면 기본값). 되살린 설정을 돌려준다. 위젯이 없는 호스트는 None."""
    current = current_settings(host)
    if current is None:
        return None
    restored = xsa.from_infotext(params)
    if restored != current:
        set_settings(host, xsa.widget_values(restored))
        logger.info("Extra Samplers 값 되살림: %s", xsa.describe(restored))
    return restored


def pass_values(host) -> dict:
    """패스 판정에 쓰는 화면 값(위젯 id → 콤보 글자·체크 bool, 없으면 None) — ``passes_from_widgets`` 의 입력."""
    values: dict = {}
    for widget_id in xsa.pass_widget_ids():
        attr, key, kind = _PASS_SOURCES[widget_id]
        widget = getattr(host, attr, None)
        if key is not None:
            widget = widget.get(key) if isinstance(widget, Mapping) else None
        if widget is None:
            values[widget_id] = None
        elif kind == "check":
            checked = _call(widget, "isChecked")
            values[widget_id] = None if checked is None else bool(checked)
        else:
            text = _call(widget, "currentText")
            values[widget_id] = text if text is not None else _call(widget, "text")
    return values


def passes_in_use(host, *, img2img: bool = False) -> tuple:
    """지금 화면에서 ER SDE 셋(``core/extra_samplers.ER_SDE_LABELS`` — 이 블록의 값을 읽는 샘플러)을 쓰는 패스."""
    return xsa.passes_from_widgets(pass_values(host), img2img=img2img)


def contribute(host, ctx) -> Any:
    """ui/sampling_blocks 기여자 — Extra Samplers 블록 하나(또는 없음). 알림은 없다(빠지면 게이트가, 연결된 sam-extra 가
    잡음 구간을 버렸거나 Dy 보조 스텝을 건너뛰었으면 결과 알림이 말한다).

    ctx.target: 't2i'(T2I 계열 전부) · 'i2i'(I2I·인페인트·채팅 편집 — 그 요청은 샘플러를 싣지 않아 메인 패스는 값을 읽지
    않는다) · 'aux'(보조 패스 봉투 — t2i 와 같은 블록)."""
    from ui.sampling_blocks import Contribution

    settings = current_settings(host)
    if settings is None:
        return Contribution()
    passes = passes_in_use(host, img2img=ctx.target == xsa.TARGET_I2I) if ctx.backend == xsa.BACKEND_WEBUI else ()
    result = xsa.plan(settings, target=ctx.target, backend=ctx.backend, capabilities=ctx.capabilities, passes=passes)
    contribution = Contribution()
    if result.block is not None:
        contribution.add(xsa.SCRIPT_NAME, result.block, provenance=result.provenance)
        # 보낼지는 뒤의 게이트(core/alwayson_propagation)가 정한다 — 여기서는 준비만 기록한다
        logger.debug("Extra Samplers 블록 준비(%s, %s): %s", ctx.target, result.reason,
                     xsa.describe(settings, [use.sampler for use in passes] if passes else None))
        if xsa.window_unsupported(settings, ctx.capabilities):
            logger.info("Extra Samplers: %s — 연결된 sam-extra 의 인자 %s개(잡음 구간 칸은 Forge 가 버린다)",
                        xsa.WINDOW_UNSUPPORTED_TEXT, xsa.live_argc(ctx.capabilities))
    elif result.reason == xsa.REASON_COMFY_PENDING and not xsa.is_default(settings):
        logger.debug("Extra Samplers: ComfyUI 는 노드 팩 1.7.0 단계 전이라 ER SDE 값을 보내지 않는다(%s)",
                     xsa.describe(settings))
    return contribution


__all__ = [
    "apply_infotext", "apply_saved_settings", "contribute", "current_settings", "get_settings",
    "init_extra_samplers_proxies", "pass_values", "passes_in_use", "set_settings",
]
