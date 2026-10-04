# ui/colorcraft_ui.py
"""Colorcraft 카드 ↔ 위젯 프록시 ↔ 샘플링 블록 — 얇은 Qt 접착층.

판정·정규화·상태 문자열은 core/colorcraft 가 한다. 여기서는

- ``init_colorcraft_proxies(bridge)``  ``_cc_<key>`` LineEditProxy 4개(``enabled``·``masking``·``img2img`` 는 'true'/'false',
  ``state`` 는 정규 상태 문자열 — 기본값과 다른 경로만 담은 JSON, 전부 기본값이면 ""). 초기값은 앱 기본값(``APP_DEFAULTS``
  = 확장 기본값: 꺼짐·마스크 끔·수정자 I 켜짐(Advanced·Base)·나머지 기본값 + 앱 전용 'I2I·인페인트에도 적용' 끔)
- ``contribute(host, ctx)``      ui/sampling_blocks.CONTRIBUTORS 의 한 줄(가이던스 뒤·Anima38 앞 — 계획 R7, Forge 에서
  Colorcraft 가 Anima Optimal Scale 바로 아래 있는 자리). 메인 체인(t2i·i2i)과 보조 패스 봉투(aux)가 같은 기여자를
  거친다 — aux 는 t2i 와 같은 블록이다(봉투 = 메인 체인, T15). 보조 패스에 실제로 보내지 않는 것은
  core/alwayson_propagation 의 Colorcraft 행(``passes=()`` — 메인 요청 전용)이 정한다. 출처는 늘 사용자 값(앱 기본값이 꺼짐).
  블록은 첫 인자 하나 = 확장 infotext 문자열(``core/colorcraft.as_block``). 돌 항목이 없으면(켠 수정자가 없거나 모두 0)
  싣지 않는다 — 카드가 '바뀌는 값 없음'을 말한다(생성 전 알림 없음).
- ``get_settings``/``set_settings``/``apply_saved_settings``  prompt_settings.json·생성 프리셋(``colorcraft_settings``).
  옛 설정 파일에 키가 없으면 앱 기본값(꺼짐) — 동작이 바뀌지 않으므로 안내하지 않는다.
- ``apply_config(host, config)``  가져오기(액션 ``sam_extra_import`` — ui/sam_extra_import)가 읽은 설정을 카드에 쓴다.

모두 GUI 스레드에서 부른다(위젯을 읽는다).
"""
from __future__ import annotations

import logging
from typing import Any, Mapping, Optional

from core import colorcraft as cc

logger = logging.getLogger(__name__)

_WIDGETS_ATTR = "colorcraft_widgets"


def init_colorcraft_proxies(bridge) -> dict:
    """``_cc_<key>`` 프록시 4개 — 앱 기본값을 문자열로 심어 둔다(Vue 가 값을 보내기 전에도 같은 판정이 나오게)."""
    from ui.widget_proxies import LineEditProxy

    values = cc.widget_values(cc.APP_DEFAULTS)
    widgets = {}
    for key in cc.WIDGET_KEYS:
        proxy = LineEditProxy(bridge, cc.widget_id(key))
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
    return {key: _widget_text(widgets.get(key)) for key in cc.WIDGET_KEYS}


def get_settings(host) -> dict:
    """저장용 위젯 문자열(키 4개, 상태는 정규 문자열). 위젯이 없는 호스트는 빈 dict."""
    raw = _raw(host)
    return {} if raw is None else cc.widget_values(cc.parse_settings(raw))


def current_settings(host) -> Optional[cc.ColorcraftSettings]:
    """카드의 지금 값. 위젯이 없는 호스트(테스트 더블·옛 호스트)는 None — 블록을 만들지 않는다."""
    raw = _raw(host)
    return None if raw is None else cc.parse_settings(raw)


def set_settings(host, raw: Optional[Mapping]) -> None:
    """저장값(또는 None = 앱 기본값)을 위젯에 쓴다. 빠진 칸·읽을 수 없는 상태는 앱 기본값(상태는 정규 문자열로)."""
    widgets = getattr(host, _WIDGETS_ATTR, None)
    if not isinstance(widgets, Mapping):
        return
    values = cc.widget_values(cc.parse_settings(raw if isinstance(raw, Mapping) else None))
    for key, value in values.items():
        proxy = widgets.get(key)
        if proxy is not None and hasattr(proxy, "setText"):
            proxy.setText(value)


def apply_saved_settings(host, settings: Mapping, *, only_present: bool) -> None:
    """``apply_generation_settings`` 의 Colorcraft 칸 — 키가 있으면 복원.

    키가 없을 때: 프리셋(``only_present=True``)은 그대로 두고, load_settings(옛 설정 파일)는 앱 기본값(꺼짐)으로
    채운다. 앱 기본값이 꺼짐이라 동작이 바뀌지 않으므로 안내하지 않는다."""
    if not hasattr(host, _WIDGETS_ATTR):
        return
    raw = settings.get(cc.SETTINGS_KEY) if isinstance(settings, Mapping) else None
    if isinstance(raw, Mapping):
        set_settings(host, raw)
    elif not only_present:
        set_settings(host, None)


def apply_config(host, config: cc.Config) -> bool:
    """가져온 설정(``core/colorcraft.from_infotext``·``from_value`` — 확장 붙여 넣기 규칙: 켬)을 카드에 쓴다. 마스크 켬과
    상태는 그 설정, 앱 전용 'I2I·인페인트에도 적용' 은 지금 값 그대로. 위젯이 없는 호스트는 False."""
    current = current_settings(host)
    if current is None:
        return False
    restored = cc.settings_of(config, img2img=current.img2img)
    set_settings(host, cc.widget_values(restored))
    logger.info("Colorcraft 값 가져옴: %s", cc.describe(config))
    return True


def contribute(host, ctx) -> Any:
    """ui/sampling_blocks 기여자 — Colorcraft 블록 하나(또는 없음).

    ctx.target: 't2i'(T2I 계열 전부 — 대기열·XYZ·시드 탐색·채팅·만화 컷) · 'i2i'(I2I·인페인트·채팅 이미지 편집 — 카드의
    'I2I·인페인트에도 적용' 토글, Base 패스 수정자만 돈다) · 'aux'(보조 패스 봉투 — t2i 와 같은 블록, 보내지는 않는다:
    passes=()). ComfyUI 도 같은 블록(컴파일러가 앱 노드 팩 1.7.0 의 노드로 — 계획 C6), Krea2 는 싣지 않는다."""
    from ui.sampling_blocks import Contribution

    settings = current_settings(host)
    if settings is None:
        return Contribution()
    result = cc.plan(settings, target=ctx.target, backend=ctx.backend, capabilities=ctx.capabilities)
    contribution = Contribution()
    if result.block is None:
        if result.reason not in (cc.REASON_OFF, cc.REASON_I2I_OFF):
            logger.debug("Colorcraft 블록 없음(%s): %s", ctx.target, result.reason)
        return contribution
    contribution.add(cc.SCRIPT_NAME, result.block, provenance=result.provenance)
    # 보낼지는 뒤의 게이트(core/alwayson_propagation)가 정한다 — 여기서는 준비만 기록한다
    logger.debug("Colorcraft 블록 준비(%s): base %s · hires %s", ctx.target, "·".join(result.base_tags) or "-",
                 "·".join(result.hires_tags) or "-")
    return contribution


__all__ = [
    "apply_config", "apply_saved_settings", "contribute", "current_settings", "get_settings",
    "init_colorcraft_proxies", "set_settings",
]
