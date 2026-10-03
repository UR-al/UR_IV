# ui/anima_speed_ui.py
"""Anima SPEED 카드 ↔ 위젯 프록시 ↔ 샘플링 블록 — 얇은 Qt 접착층.

판정·정규화는 core/anima_speed 가 한다. 여기서는

- ``init_speed_proxies(bridge)``  ``_aspeed_<key>`` LineEditProxy 15개(값은 모두 문자열 — DeGrid·Extra Schedulers 프록시와
  같은 패턴), 초기값은 앱 기본값(``APP_DEFAULTS`` = 확장 기본값: 꺼짐·transition·neo_shift·anima·0.5,1.0·δ 0.01·1.03·0.7·
  적응형·dct·A·β·시드 -1·Hires 끔 + 앱 전용 'I2I에도 적용' 끔)
- ``contribute(host, ctx)``      ui/sampling_blocks.CONTRIBUTORS 의 한 줄(Anima38 뒤 — 계획 R7). 메인 체인(t2i·i2i)과 보조
  패스 봉투(aux)가 같은 기여자를 거친다 — aux 는 t2i 와 같은 블록이고 알림이 없다(봉투 = 메인 체인, T15). 보조 패스에
  실제로 보내지 않는 것은 core/alwayson_propagation 의 SPEED 행(``passes=()`` — 메인 요청 전용, 계획 D4)이 정한다. 출처는
  늘 사용자 값(앱 기본값이 꺼짐). 설정만 보고 아는 건너뜀(해상도 비율 1.0 하나 — ``static_skip_reason``)은 생성 전 정보
  알림 하나. 인페인트·ControlNet·ComfyUI Spectrum 은 요청 페이로드를 봐야 알아서 여기서는 말하지 않는다 — Forge 가
  'Anima SPEED status' 에 이유를 남기고 결과 알림이 말한다(ComfyUI 는 컴파일러, 계획 C4a).
- ``get_settings``/``set_settings``/``apply_saved_settings``  prompt_settings.json·생성 프리셋(``anima_speed_settings``).
  옛 설정 파일에 키가 없으면 앱 기본값(꺼짐) — 동작이 바뀌지 않으므로 안내하지 않는다.
- ``apply_infotext(host, params)``  붙여 넣은 infotext(가져오기 — 계획 X3c ``sam_extra_import`` 가 연결)의 값을 카드에
  되살린다(``core/anima_speed.from_infotext`` — 'Anima SPEED' 키가 있을 때만 켠다, 'I2I에도 적용' 은 그대로).

모두 GUI 스레드에서 부른다(위젯을 읽는다).
"""
from __future__ import annotations

import logging
from typing import Any, Mapping, Optional

from core import anima_speed as asp

logger = logging.getLogger(__name__)

_WIDGETS_ATTR = "anima_speed_widgets"


def init_speed_proxies(bridge) -> dict:
    """``_aspeed_<key>`` 프록시 15개 — 앱 기본값을 문자열로 심어 둔다(Vue 가 값을 보내기 전에도 같은 판정이 나오게)."""
    from ui.widget_proxies import LineEditProxy

    values = asp.widget_values(asp.APP_DEFAULTS)
    widgets = {}
    for key in asp.WIDGET_KEYS:
        proxy = LineEditProxy(bridge, asp.widget_id(key))
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
    return {key: _widget_text(widgets.get(key)) for key in asp.WIDGET_KEYS}


def get_settings(host) -> dict:
    """저장용 위젯 문자열(키 15개). 위젯이 없는 호스트는 빈 dict."""
    raw = _raw(host)
    return {} if raw is None else asp.widget_values(asp.parse_settings(raw))


def current_settings(host) -> Optional[asp.SpeedSettings]:
    """카드의 지금 값. 위젯이 없는 호스트(테스트 더블·옛 호스트)는 None — 블록을 만들지 않는다."""
    raw = _raw(host)
    return None if raw is None else asp.parse_settings(raw)


def set_settings(host, raw: Optional[Mapping]) -> None:
    """저장값(또는 None = 앱 기본값)을 위젯에 쓴다. 모르는·읽을 수 없는 칸은 앱 기본값(텍스트 칸은 친 그대로)."""
    widgets = getattr(host, _WIDGETS_ATTR, None)
    if not isinstance(widgets, Mapping):
        return
    values = asp.widget_values(asp.parse_settings(raw if isinstance(raw, Mapping) else None))
    for key, value in values.items():
        proxy = widgets.get(key)
        if proxy is not None and hasattr(proxy, "setText"):
            proxy.setText(value)


def apply_saved_settings(host, settings: Mapping, *, only_present: bool) -> None:
    """``apply_generation_settings`` 의 Anima SPEED 칸 — 키가 있으면 복원.

    키가 없을 때: 프리셋(``only_present=True``)은 그대로 두고, load_settings(옛 설정 파일)는 앱 기본값(꺼짐)으로
    채운다. 앱 기본값이 꺼짐이라 동작이 바뀌지 않으므로 안내하지 않는다."""
    if not hasattr(host, _WIDGETS_ATTR):
        return
    raw = settings.get(asp.SETTINGS_KEY) if isinstance(settings, Mapping) else None
    if isinstance(raw, Mapping):
        set_settings(host, raw)
    elif not only_present:
        set_settings(host, None)


def apply_infotext(host, params: Any) -> Optional[asp.SpeedSettings]:
    """붙여 넣은 infotext → 카드(``core/anima_speed.from_infotext`` — 확장 붙여 넣기 규칙: 'Anima SPEED' 키가 있으면 켜고
    그 요약의 값, 없으면 끈다. 'I2I에도 적용' 은 지금 값 그대로). 되살린 설정을 돌려준다. 위젯이 없는 호스트는 None."""
    current = current_settings(host)
    if current is None:
        return None
    restored = asp.from_infotext(params, base=current)
    if restored != current:
        set_settings(host, asp.widget_values(restored))
        logger.info("Anima SPEED 값 되살림: %s", asp.describe(restored) if restored.enabled else "꺼짐")
    return restored


def contribute(host, ctx) -> Any:
    """ui/sampling_blocks 기여자 — Anima SPEED 블록 하나(또는 없음).

    ctx.target: 't2i'(T2I 계열 전부 — 대기열·XYZ·시드 탐색·채팅·만화 컷) · 'i2i'(I2I·인페인트·채팅 이미지 편집 — 카드의
    'I2I에도 적용' 토글) · 'aux'(보조 패스 봉투 — t2i 와 같은 블록, 보내지는 않는다: passes=()). 잘못된 설정도 싣는다 —
    Forge 가 'invalid settings' 로 기록하고 평소처럼 샘플링한다(``core/anima_speed.plan`` 이 로그를 남긴다)."""
    from core.sam_extra_notices import speed_static_skip_notice
    from ui.sampling_blocks import Contribution

    settings = current_settings(host)
    if settings is None:
        return Contribution()
    result = asp.plan(settings, target=ctx.target, backend=ctx.backend, capabilities=ctx.capabilities)
    contribution = Contribution()
    if result.block is None:
        return contribution
    contribution.add(asp.SCRIPT_NAME, result.block, provenance=result.provenance)
    # 보낼지는 뒤의 게이트(core/alwayson_propagation)가 정한다 — 여기서는 준비만 기록한다
    logger.debug("Anima SPEED 블록 준비(%s): %s", ctx.target, asp.describe(settings))
    if ctx.target != asp.TARGET_AUX and not result.errors:
        skip = asp.static_skip_reason(settings, None, backend=ctx.backend)
        if skip is not None:
            contribution.notices.append(speed_static_skip_notice(skip))
    return contribution


__all__ = [
    "apply_infotext", "apply_saved_settings", "contribute", "current_settings", "get_settings",
    "init_speed_proxies", "set_settings",
]
