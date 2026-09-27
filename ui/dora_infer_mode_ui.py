# ui/dora_infer_mode_ui.py
"""DoRA 추론 방식 카드(frontend/src/components/params/DoraModeCard.vue) ↔ 위젯 프록시 ↔ 샘플링 블록 — 얇은 Qt 접착층.

판정·정규화는 core/dora_infer_mode 가 한다. 여기서는

- ``init_dora_proxies(bridge)``  ``_dora_<key>`` LineEditProxy 6개(값은 모두 문자열 — 가이던스 프록시와 같은 패턴),
  초기값은 앱 기본값(``APP_DEFAULTS`` — 사용자 Forge ui-config 의 txt2img 값)
- ``contribute(host, ctx)``      ui/sampling_blocks.CONTRIBUTORS 의 한 줄. 메인 체인(t2i·i2i)과 보조 패스 봉투(aux)가
  같은 기여자를 거친다. 블록 출처는 카드 값이 앱 기본값 그대로면 앱 기본값(모를 때·없을 때 조용히 — critic A2·A6),
  아니면 사용자 값. 스크립트 없음·확인 전 게이트는 core/alwayson_propagation 의 DoRA 행(모르면 늘 SKIP)이 한다.
- ``get_settings``/``apply_saved_settings``  prompt_settings.json·생성 프리셋(``dora_infer_settings``)
- 옛 설정 파일에 키가 없으면 앱 기본값으로 시작하고, 그 안내 토스트는 Vue 가 준비되고 메인 창이 보인 뒤에 띄운다
  (critic A5 — load_settings 는 __init__ 에서 Vue 보다 먼저 돈다. ui/boot_notices). 문구는 그때의 백엔드로 정한다
  (ComfyUI 면 띄우지 않는다 — ``app_default_boot_notice``).

모두 GUI 스레드에서 부른다(위젯을 읽는다).
"""
from __future__ import annotations

import logging
from typing import Any, Mapping, Optional

from core import dora_infer_mode as dim

logger = logging.getLogger(__name__)

_WIDGETS_ATTR = "dora_widgets"


def init_dora_proxies(bridge) -> dict:
    """``_dora_<key>`` 프록시 6개 — 앱 기본값을 문자열로 심어 둔다(Vue 가 값을 보내기 전에도 같은 블록이 나가게)."""
    from ui.widget_proxies import LineEditProxy

    values = dim.widget_values(dim.APP_DEFAULTS)
    widgets = {}
    for key in dim.WIDGET_KEYS:
        proxy = LineEditProxy(bridge, dim.widget_id(key))
        proxy.setText(values[key])
        widgets[key] = proxy
    return widgets


def _widget_text(proxy: Any) -> Any:
    text = getattr(proxy, "text", None)
    try:
        return text() if callable(text) else None
    except Exception:
        return None


def get_settings(host) -> dict:
    """저장용 위젯 문자열(키 6개). 위젯이 없는 호스트는 빈 dict."""
    widgets = getattr(host, _WIDGETS_ATTR, None)
    if not isinstance(widgets, Mapping):
        return {}
    raw = {key: _widget_text(widgets.get(key)) for key in dim.WIDGET_KEYS}
    return dim.widget_values(dim.parse_settings(raw))


def current_settings(host) -> Optional[dim.DoraSettings]:
    """카드의 지금 값. 위젯이 없는 호스트(테스트 더블·옛 호스트)는 None — 블록을 만들지 않는다."""
    widgets = getattr(host, _WIDGETS_ATTR, None)
    if not isinstance(widgets, Mapping):
        return None
    return dim.parse_settings({key: _widget_text(widgets.get(key)) for key in dim.WIDGET_KEYS})


def set_settings(host, raw: Optional[Mapping]) -> None:
    """저장값(또는 None = 앱 기본값)을 위젯에 쓴다. 모르는 칸은 앱 기본값."""
    widgets = getattr(host, _WIDGETS_ATTR, None)
    if not isinstance(widgets, Mapping):
        return
    values = dim.widget_values(dim.parse_settings(raw if isinstance(raw, Mapping) else None))
    for key, value in values.items():
        proxy = widgets.get(key)
        if proxy is not None and hasattr(proxy, "setText"):
            proxy.setText(value)


def apply_saved_settings(host, settings: Mapping, *, only_present: bool) -> None:
    """``apply_generation_settings`` 의 DoRA 칸 — 키가 있으면 복원.

    키가 없을 때: 프리셋(``only_present=True``)은 그대로 두고, load_settings(옛 설정 파일)는 앱 기본값으로 채운 뒤
    안내 토스트 한 번을 Vue 준비 뒤로 미룬다(다음 저장부터 키가 생겨 다시 뜨지 않는다).
    """
    if not hasattr(host, _WIDGETS_ATTR):
        return
    raw = settings.get(dim.SETTINGS_KEY) if isinstance(settings, Mapping) else None
    if isinstance(raw, Mapping):
        set_settings(host, raw)
        return
    if only_present:
        return
    set_settings(host, None)
    try:
        from core.sam_extra_notices import CODE_DORA_APP_DEFAULT
        from ui.boot_notices import DeferredNotice, defer_boot_notice
        defer_boot_notice(host, DeferredNotice(CODE_DORA_APP_DEFAULT, app_default_boot_notice))
    except Exception:
        logger.debug("DoRA 첫 로드 안내 예약 실패(무시)", exc_info=True)


def app_default_boot_notice() -> Any:
    """첫 로드 안내 — 띄울 때(Vue 준비·창 표시 뒤)의 백엔드로 정한다. load_settings 때는 백엔드가 아직 안 정해졌다.

    ComfyUI 는 DoRA 블록을 만들지 않아(``plan`` 의 'comfy' — 순정으로만 합친다) 바뀌는 것이 없으므로 띄우지 않는다
    (카드가 'ComfyUI 는 순정으로만 합칩니다' 를 보여 준다). 시작 게이트에서 백엔드를 아직 고르지 않았을 때도 틀리지
    않게 문구는 'Forge 로 생성할 때' 로 한정한다.
    """
    from backends import BackendType, get_backend_type
    from core.sam_extra_notices import dora_app_default_notice

    if get_backend_type() == BackendType.COMFYUI:
        logger.info("[DoRA] 첫 로드 안내 생략 — ComfyUI 는 DoRA 추론 방식 없이 순정으로 합친다")
        return None
    return dora_app_default_notice(dim.describe(dim.APP_DEFAULTS))


def _has_loras(host) -> bool:
    """켜진 LoRA 스택 항목이 있거나 메인 프롬프트에 <lora:>/<lyco:> 태그가 있나 (Comfy 순정 안내용)."""
    for entry in getattr(host, "_vue_lora_entries", None) or ():
        # core/lora_stack.normalize_lora_entries 형식 — enabled False 만 꺼짐, 이름 없는 항목은 텍스트에서 빠진다
        if isinstance(entry, Mapping) and entry.get("enabled", True) is not False and str(entry.get("name") or ""):
            return True
    display = getattr(host, "total_prompt_display", None)
    try:
        text = display.toPlainText() if display is not None and hasattr(display, "toPlainText") else ""
    except Exception:
        text = ""
    return dim.has_lora_tags(text)


def contribute(host, ctx) -> Any:
    """ui/sampling_blocks 기여자 — DoRA 블록 하나(또는 없음)와 안내 알림.

    ctx.target: 't2i'(T2I 계열 전부) · 'i2i'(I2I·인페인트·채팅 이미지 편집 — 카드의 'I2I·인페인트에도 적용' 토글) ·
    'aux'(Refine·단독/배치 SAM3·단독 ADetailer·손 재구성 — 토글과 무관하게 T2I 값).
    """
    from ui.sampling_blocks import Contribution

    settings = current_settings(host)
    if settings is None:
        return Contribution()
    result = dim.plan(settings, target=ctx.target, backend=ctx.backend, capabilities=ctx.capabilities,
                      has_loras=_has_loras(host) if ctx.backend == dim.BACKEND_COMFY else False)
    contribution = Contribution()
    if result.notice is not None:
        contribution.notices.append(result.notice)
    if result.block is not None:
        contribution.add(dim.SCRIPT_NAME, result.block, provenance=result.provenance)
        # 보낼지는 뒤의 게이트(core/alwayson_propagation)가 정한다 — 여기서는 준비만 기록한다
        logger.debug("DoRA 추론 방식 블록 준비(%s): %s", ctx.target, dim.describe(settings))
    return contribution


__all__ = ["app_default_boot_notice", "apply_saved_settings", "contribute", "current_settings", "get_settings",
           "init_dora_proxies", "set_settings"]
