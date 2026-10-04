# ui/extra_schedulers_ui.py
"""Extra Schedulers 값 칸(파라미터 카드의 Sampler/Scheduler 줄 아래) ↔ 위젯 프록시 ↔ 샘플링 블록 — 얇은 Qt 접착층.

판정·정규화는 core/extra_schedulers 가 한다. 여기서는

- ``init_extra_schedulers_proxies(bridge)``  ``_xsched_<인자>`` LineEditProxy 8개(``core/extra_schedulers.WIDGET_KEYS`` =
  확장 ARG_NAMES — 값은 모두 문자열, DeGrid·DoRA 프록시와 같은 패턴), 초기값은 앱 기본값(``APP_DEFAULTS`` = 확장 기본값:
  식 모드·'M * (m / M) ** x'·'[1.0, 0.6, 0.25, 0.1, 0.0]'·μ 0·β 0.5·Flow Cosmos ρ 7·σ̃ max 80·σ̃ min 0.002)
- ``contribute(host, ctx)``      ui/sampling_blocks.CONTRIBUTORS 의 한 줄(DeGrid 뒤 — 계획 R7). 지금 화면에서 laplace·
  custom·flow_cosmos_rho7·flow_cosmos_dynamic 을 쓰는 패스(``pass_values`` → ``core/extra_schedulers.passes_from_widgets``)를
  ``plan`` 에 넘긴다 — 스냅샷이
  스크립트를 확인했으면 늘, 확인 못 했으면 그런 패스가 있을 때만 싣는다(계획 X5). 출처는 ``plan`` 이 값으로 정한다
  (기본값이면 앱 기본값). 보낼지는 뒤의 게이트(core/alwayson_propagation)가 정한다. aux(보조 패스 봉투)는 t2i 와 같은
  블록이고 모든 보조 img2img 로 간다(확장 전역 상태가 요청마다 덮인다).
- ``get_settings``/``set_settings``/``apply_saved_settings``  prompt_settings.json·생성 프리셋(``extra_schedulers_settings``).
  옛 설정 파일에 키가 없으면 앱 기본값 — 확장 기본값과 같아 동작이 바뀌지 않으므로 안내하지 않는다. v0.33.0 전에 저장한
  값(키 다섯)의 Flow Cosmos 셋도 앱 기본값이다(확장이 키 없는 옛 요청을 읽는 것과 같다).
- ``apply_infotext(host, params)``  붙여 넣은 infotext 의 값을 칸에 되살린다(``core/extra_schedulers.from_infotext`` —
  확장 붙여 넣기 규칙: 이미지가 쓰지 않은 스케줄러의 μ/β·Flow Cosmos 칸은 키가 없으면 그대로)
- ``restore_from_metadata(host, params)``  '이 설정으로 생성'(ui/image_metadata_actions — 계획 A-R1·X3b): 이미지의 메인·
  Hires 스케줄러가 값 스케줄러일 때만 그 스케줄러의 칸을 함께 되살린다(``core/extra_schedulers.restore_from_infotext``).

- 식·목록 미리 검사(계획 C7): custom 식·시그마 목록 칸이 바뀔 때마다(Vue 입력·setText 모두 ``textChanged``)
  ``core/extra_schedulers.check_text`` 로 보고 ``pushWidgetProperty('_xsched_custom_expression' | '_xsched_custom_sigmas',
  'error', 문구|None)`` — 카드가 그 칸 아래 오류 줄로 보인다(새 브리지 이름 없음). 생성 전에는 ``contribute`` 가 custom 을
  쓰는 패스가 있고 그 방식의 글자를 쓸 수 없으면 경고 알림(``CODE_XSCHED_INVALID_TEXT``)을 단다 — 요청은 그대로 보낸다.

모두 GUI 스레드에서 부른다(위젯을 읽는다).
"""
from __future__ import annotations

import logging
from typing import Any, Mapping, Optional

from core import extra_schedulers as xs

logger = logging.getLogger(__name__)

_WIDGETS_ATTR = "extra_schedulers_widgets"

# 패스 판정 위젯 id → host 에서 읽는 곳(속성 이름, 위젯 dict 키|None, 'check'|'text') — ui/generator_ui_setup 의 프록시
_PASS_SOURCES = {
    xs.MAIN_SCHEDULER_WIDGET: ("scheduler_combo", None, "text"),
    xs.HIRES_GROUP_WIDGET: ("hires_options_group", None, "check"),
    xs.HIRES_SCHEDULER_WIDGET: ("hires_scheduler_combo", None, "text"),
    xs.ADETAILER_GROUP_WIDGET: ("adetailer_group", None, "check"),
    "ad_slot1_group": ("ad_slot1_group", None, "check"),
    "_ad_s1_use_sampler": ("s1_widgets", "use_sampler_check", "check"),
    "_ad_s1_scheduler": ("s1_widgets", "scheduler_combo", "text"),
    "ad_slot2_group": ("ad_slot2_group", None, "check"),
    "_ad_s2_use_sampler": ("s2_widgets", "use_sampler_check", "check"),
    "_ad_s2_scheduler": ("s2_widgets", "scheduler_combo", "text"),
    xs.SAM3_GROUP_WIDGET: ("sam3_group", None, "check"),
    xs.SAM3_MODE_WIDGET: ("sam3_widgets", "mode", "text"),
    xs.SAM3_USE_SCHEDULER_WIDGET: ("sam3_widgets", "use_scheduler_check", "check"),
    xs.SAM3_SCHEDULER_WIDGET: ("sam3_widgets", "scheduler", "text"),
}


def init_extra_schedulers_proxies(bridge) -> dict:
    """``_xsched_<인자>`` 프록시 8개 — 앱 기본값을 문자열로 심어 둔다(Vue 가 값을 보내기 전에도 같은 블록이 나가게).
    custom 식·시그마 목록 칸은 바뀔 때마다 미리 검사해 오류 줄을 보낸다(``watch_text_errors`` — 계획 C7)."""
    from ui.widget_proxies import LineEditProxy

    values = xs.widget_values(xs.APP_DEFAULTS)
    widgets = {}
    for key in xs.WIDGET_KEYS:
        proxy = LineEditProxy(bridge, xs.widget_id(key))
        proxy.setText(values[key])
        widgets[key] = proxy
    watch_text_errors(bridge, widgets)
    return widgets


# 미리 검사하는 칸 → 그 칸을 읽는 custom 방식
_CHECKED_TEXTS = (("custom_expression", xs.MODE_EXPRESSION), ("custom_sigmas", xs.MODE_SIGMAS))


def push_text_error(bridge, key: str, message: Optional[str]) -> None:
    """칸 ``key`` 의 오류 줄(위젯 속성 ``core/extra_schedulers.ERROR_PROPERTY``) — None 이면 지운다. 종료 중이면 조용히."""
    push = getattr(bridge, "pushWidgetProperty", None)
    if not callable(push):
        return
    try:
        push(xs.widget_id(key), xs.ERROR_PROPERTY, message)
    except RuntimeError:
        pass   # 종료 중 QObject 가 이미 사라졌다


def watch_text_errors(bridge, widgets: Mapping) -> None:
    """custom 식·시그마 목록 칸의 ``textChanged`` → ``check_text`` → 오류 줄. 같은 문구는 다시 보내지 않는다(타자마다
    부르지만 바뀔 때만 보낸다). 지금 값(앱 기본값 — 쓸 수 있다)은 오류가 없어 처음에는 보내지 않는다."""
    shown: dict = {}
    for key, mode in _CHECKED_TEXTS:
        signal = getattr(widgets.get(key), "textChanged", None)
        connect = getattr(signal, "connect", None)
        if not callable(connect):
            continue
        shown[key] = None

        def _check(text, key=key, mode=mode):
            message = xs.check_text(mode, text)
            if message != shown.get(key):
                shown[key] = message
                push_text_error(bridge, key, message)

        connect(_check)


def _call(widget: Any, name: str) -> Any:
    method = getattr(widget, name, None)
    try:
        return method() if callable(method) else None
    except Exception:
        return None


def _widget_text(proxy: Any) -> Any:
    return _call(proxy, "text")


def _raw(host) -> Optional[dict]:
    widgets = getattr(host, _WIDGETS_ATTR, None)
    if not isinstance(widgets, Mapping):
        return None
    return {key: _widget_text(widgets.get(key)) for key in xs.WIDGET_KEYS}


def get_settings(host) -> dict:
    """저장용 위젯 문자열(키 8개). 위젯이 없는 호스트는 빈 dict."""
    raw = _raw(host)
    return {} if raw is None else xs.widget_values(xs.parse_settings(raw))


def current_settings(host) -> Optional[xs.ExtraSchedulerSettings]:
    """칸의 지금 값. 위젯이 없는 호스트(테스트 더블·옛 호스트)는 None — 블록을 만들지 않는다."""
    raw = _raw(host)
    return None if raw is None else xs.parse_settings(raw)


def set_settings(host, raw: Optional[Mapping]) -> None:
    """저장값(또는 None = 앱 기본값)을 위젯에 쓴다. 모르는·읽을 수 없는 칸은 앱 기본값."""
    widgets = getattr(host, _WIDGETS_ATTR, None)
    if not isinstance(widgets, Mapping):
        return
    values = xs.widget_values(xs.parse_settings(raw if isinstance(raw, Mapping) else None))
    for key, value in values.items():
        proxy = widgets.get(key)
        if proxy is not None and hasattr(proxy, "setText"):
            proxy.setText(value)


def apply_saved_settings(host, settings: Mapping, *, only_present: bool) -> None:
    """``apply_generation_settings`` 의 Extra Schedulers 칸 — 키가 있으면 복원.

    키가 없을 때: 프리셋(``only_present=True``)은 그대로 두고, load_settings(옛 설정 파일)는 앱 기본값으로 채운다. 앱
    기본값이 확장 기본값과 같아(블록이 없던 Forge 요청과 같은 값) 동작이 바뀌지 않으므로 안내하지 않는다."""
    if not hasattr(host, _WIDGETS_ATTR):
        return
    raw = settings.get(xs.SETTINGS_KEY) if isinstance(settings, Mapping) else None
    if isinstance(raw, Mapping):
        set_settings(host, raw)
    elif not only_present:
        set_settings(host, None)


def apply_infotext(host, params: Any) -> Optional[xs.ExtraSchedulerSettings]:
    """붙여 넣은 infotext → 칸(``core/extra_schedulers.from_infotext`` — 확장 paste_* 규칙: 이미지가 쓰지 않은 스케줄러의
    칸은 지금 값 그대로). 되살린 설정을 돌려준다. 위젯이 없는 호스트는 None."""
    current = current_settings(host)
    if current is None:
        return None
    restored = xs.from_infotext(params, base=current)
    if restored != current:
        set_settings(host, xs.widget_values(restored))
        logger.info("Extra Schedulers 값 되살림: %s", xs.describe(restored))
    return restored


def restore_from_metadata(host, params: Any) -> Optional[xs.ExtraSchedulerSettings]:
    """'이 설정으로 생성'(계획 A-R1·X3b) — 이미지의 메인·Hires 스케줄러가 값 스케줄러일 때만 그 스케줄러의 칸을
    되살린다(``core/extra_schedulers.restore_from_infotext`` — 스케줄러와 그 값을 함께). 값이 바뀐 칸만 쓰고 나머지 칸의
    글자는 건드리지 않는다. 아무 패스도 값 스케줄러가 아니거나 위젯이 없는 호스트면 None(칸 그대로)."""
    current = current_settings(host)
    if current is None:
        return None
    restored = xs.restore_from_infotext(params, base=current)
    if restored is None:
        return None
    if restored != current:
        widgets = getattr(host, _WIDGETS_ATTR)
        values = xs.widget_values(restored)
        for key in xs.WIDGET_KEYS:
            proxy = widgets.get(key)
            if getattr(restored, key) != getattr(current, key) and proxy is not None and hasattr(proxy, "setText"):
                proxy.setText(values[key])
        logger.info("Extra Schedulers 값 되살림(이 설정으로 생성 — %s): %s",
                    ", ".join(xs.infotext_value_schedulers(params)), xs.describe(restored))
    return restored


def pass_values(host) -> dict:
    """패스 판정에 쓰는 화면 값(위젯 id → 콤보 글자·체크 bool, 없으면 None) — ``passes_from_widgets`` 의 입력."""
    values: dict = {}
    for widget_id in xs.pass_widget_ids():
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
    """지금 화면에서 값 스케줄러(``core/extra_schedulers.VALUE_SCHEDULERS``)를 쓰는 패스(``PassUse`` 튜플)."""
    return xs.passes_from_widgets(pass_values(host), img2img=img2img)


def _invalid_text_notice(host, settings: xs.ExtraSchedulerSettings, *, img2img: bool) -> Any:
    """생성 전 경고(계획 C7, 설계 XS §3.2): 어느 패스가 custom 을 쓰는데 그 방식의 글자를 쓸 수 없으면 — 요청은 그대로
    보낸다(Forge·ComfyUI 가 같은 영어 문구로 멈춘다). custom 을 쓰지 않거나 쓸 수 있으면 None."""
    if not any(use.scheduler == xs.NAME_CUSTOM for use in passes_in_use(host, img2img=img2img)):
        return None
    message = xs.check_settings(settings)
    if message is None:
        return None
    from core import sam_extra_notices as sn

    logger.warning("Extra Schedulers: custom %s 을 쓸 수 없다 — %s",
                   "시그마 목록" if settings.custom_mode == xs.MODE_SIGMAS else "식", message)
    return sn.xsched_invalid_text_notice(message, sigmas=settings.custom_mode == xs.MODE_SIGMAS)


def contribute(host, ctx) -> Any:
    """ui/sampling_blocks 기여자 — Extra Schedulers 블록 하나(또는 없음). 알림은 custom 식·목록을 쓸 수 없을 때의 생성 전
    경고 하나뿐이다(t2i·i2i — ``_invalid_text_notice``; 빠지면 게이트가, 다른 확장의 스케줄러가 돌았으면 결과 알림이 말한다).

    ctx.target: 't2i'(T2I 계열 전부) · 'i2i'(I2I·인페인트·채팅 편집 — 그 요청은 스케줄러를 싣지 않아 메인 패스는 값을 읽지
    않는다) · 'aux'(보조 패스 봉투 — t2i 와 같은 블록)."""
    from ui.sampling_blocks import Contribution

    settings = current_settings(host)
    if settings is None:
        return Contribution()
    passes = passes_in_use(host, img2img=ctx.target == xs.TARGET_I2I) if ctx.backend == xs.BACKEND_WEBUI else ()
    result = xs.plan(settings, target=ctx.target, backend=ctx.backend, capabilities=ctx.capabilities, passes=passes)
    contribution = Contribution()
    if result.block is not None:
        contribution.add(xs.SCRIPT_NAME, result.block, provenance=result.provenance)
        if ctx.target != xs.TARGET_AUX:
            notice = _invalid_text_notice(host, settings, img2img=ctx.target == xs.TARGET_I2I)
            if notice is not None:
                contribution.notices.append(notice)
        # 보낼지는 뒤의 게이트(core/alwayson_propagation)가 정한다 — 여기서는 준비만 기록한다
        logger.debug("Extra Schedulers 블록 준비(%s, %s): %s", ctx.target, result.reason,
                     xs.describe(settings, [use.scheduler for use in passes] if passes else None))
    return contribution


__all__ = [
    "apply_infotext", "apply_saved_settings", "contribute", "current_settings", "get_settings",
    "init_extra_schedulers_proxies", "pass_values", "passes_in_use", "push_text_error", "restore_from_metadata",
    "set_settings", "watch_text_errors",
]
