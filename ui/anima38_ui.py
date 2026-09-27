# ui/anima38_ui.py
"""Anima 3.8B 카드(frontend/src/components/params/Anima38Card.vue) ↔ 위젯 프록시 ↔ 샘플링 블록 — 얇은 Qt 접착층.

판정·정규화는 core/anima38(블록·출처)과 core/anima_model_kind(모델 종류)가 한다. 여기서는

- ``init_anima38_proxies(bridge)``  ``_a38_<key>`` LineEditProxy 7개(확장 인자 6 + 'I2I·인페인트에도 적용'), 초기값은
  앱 기본값(``APP_T2I_DEFAULTS`` — 부정 커넥터 켬, v1 끔)
- ``contribute(host, ctx)``      ui/sampling_blocks.CONTRIBUTORS 의 한 줄(가이던스 뒤·DoRA 앞). 모델 종류는 T2I 콤보
  값을 **캐시만으로** 판정한다(GUI 스레드에서 디스크를 읽지 않는다 — 캐시가 없으면 이름). 보조 패스(aux)는 클릭한
  순간의 T2I 모델로 만든 블록이라, 실제로 쓰는 모델이 다르면 백엔드가 뺀다(core/alwayson_propagation.drop_model_bound,
  critic A7). 스크립트 없음·확인 전 게이트는 core/alwayson_propagation 의 Anima38 행(앱 기본값은 모르면 SKIP — A2)
- ``prewarm_model_kinds(host, titles, paths)``  연결 때(GUI 슬롯 ``on_webui_info_loaded``) 데몬 스레드에서 체크포인트
  헤더를 읽어 종류를 기억하고 Vue 에 ``model_combo.animaKinds`` 로 보낸다(기존 위젯 속성 채널 — 새 브리지 이름 없음).
  일련번호로 옛 결과를 버린다(ui/sam_extra_capabilities_actions 의 serial 패턴). 캐시 쓰기·읽기는 락(critic B16)
- ``push_comfy_adapters(host, choices)``  연결 때 ComfyUI 의 v1 어댑터 선택지(object_info)를 ``_a38_adapter`` 위젯
  속성 ``comfyAdapters`` 로 보낸다 — Forge·연결 실패는 None(카드는 Forge 기능 스냅샷 선택지를 쓴다, P9 리뷰 2)
- ``get_settings``/``apply_saved_settings``  prompt_settings.json·생성 프리셋(``anima38_settings``). 옛 설정 파일에 키가
  없으면 앱 기본값으로 시작하고 안내 토스트는 Vue 준비·창 표시 뒤(critic A5, ui/boot_notices)
"""
from __future__ import annotations

import logging
import threading
from typing import Any, Iterable, Mapping, Optional

from core import anima38 as a38
from core import anima_model_kind as amk

logger = logging.getLogger(__name__)

_WIDGETS_ATTR = "anima38_widgets"
KIND_PROPERTY = "animaKinds"          # model_combo 위젯 속성 — frontend/src/utils/anima38Card.ts KIND_PROPERTY
MODEL_WIDGET = "model_combo"
COMFY_ADAPTER_PROPERTY = "comfyAdapters"   # _a38_adapter 위젯 속성 — anima38Card.ts COMFY_ADAPTER_PROPERTY

_serial_lock = threading.Lock()
_serial = 0


def init_anima38_proxies(bridge) -> dict:
    """``_a38_<key>`` 프록시 7개 — 앱 기본값을 문자열로 심어 둔다(Vue 가 값을 보내기 전에도 같은 블록이 나가게)."""
    from ui.widget_proxies import LineEditProxy

    values = a38.widget_values(a38.APP_T2I_DEFAULTS, False)
    widgets = {}
    for key in a38.WIDGET_KEYS:
        proxy = LineEditProxy(bridge, a38.widget_id(key))
        proxy.setText(values[key])
        widgets[key] = proxy
    return widgets


def _widget_text(proxy: Any) -> Any:
    text = getattr(proxy, "text", None)
    try:
        return text() if callable(text) else None
    except Exception:
        return None


def current_values(host) -> Optional[tuple]:
    """(설정, I2I·인페인트에도 적용). 위젯이 없는 호스트(테스트 더블·옛 호스트)는 None — 블록을 만들지 않는다."""
    widgets = getattr(host, _WIDGETS_ATTR, None)
    if not isinstance(widgets, Mapping):
        return None
    return a38.settings_from_widgets({key: _widget_text(widgets.get(key)) for key in a38.WIDGET_KEYS})


def get_settings(host) -> dict:
    """저장용 위젯 문자열(키 7개). 위젯이 없는 호스트는 빈 dict."""
    values = current_values(host)
    if values is None:
        return {}
    return a38.widget_values(*values)


def set_settings(host, raw: Optional[Mapping]) -> None:
    """저장값(또는 None = 앱 기본값)을 위젯에 쓴다. 모르는 칸은 앱 기본값."""
    widgets = getattr(host, _WIDGETS_ATTR, None)
    if not isinstance(widgets, Mapping):
        return
    values = a38.widget_values(*a38.settings_from_widgets(raw if isinstance(raw, Mapping) else None))
    for key, value in values.items():
        proxy = widgets.get(key)
        if proxy is not None and hasattr(proxy, "setText"):
            proxy.setText(value)


def apply_saved_settings(host, settings: Mapping, *, only_present: bool) -> None:
    """``apply_generation_settings`` 의 Anima38 칸 — 키가 있으면 복원.

    키가 없을 때: 프리셋(``only_present=True``)은 그대로 두고, load_settings(옛 설정 파일)는 앱 기본값으로 채운 뒤
    안내 토스트 한 번을 Vue 준비 뒤로 미룬다(다음 저장부터 키가 생겨 다시 뜨지 않는다).
    """
    if not hasattr(host, _WIDGETS_ATTR):
        return
    raw = settings.get(a38.SETTINGS_KEY) if isinstance(settings, Mapping) else None
    if isinstance(raw, Mapping):
        set_settings(host, raw)
        return
    if only_present:
        return
    set_settings(host, None)
    try:
        from core.sam_extra_notices import CODE_ANIMA38_APP_DEFAULT, anima38_app_default_notice
        from ui.boot_notices import DeferredNotice, defer_boot_notice
        defer_boot_notice(host, DeferredNotice(CODE_ANIMA38_APP_DEFAULT, anima38_app_default_notice))
    except Exception:
        logger.debug("Anima38 첫 로드 안내 예약 실패(무시)", exc_info=True)


def model_kind(title: Any) -> str:
    """T2I 콤보 값의 종류 — 캐시만(GUI 스레드). 연결 때 헤더를 읽기 전이면 이름으로만(비 Anima 는 unknown)."""
    return amk.classify(title, cached_only=True)


def _modules(host) -> list:
    """T2I 패널의 추가 모듈(VAE + TE, _build_generation_payload 의 forge_additional_modules 와 같은 칸)."""
    out = []
    combo = getattr(host, "vae_main_combo", None)
    try:
        vae = combo.currentText() if combo is not None and hasattr(combo, "currentText") else ""
    except Exception:
        vae = ""
    if str(vae or "").strip():
        out.append(str(vae).strip())
    te = getattr(host, "te_main_input", None)
    try:
        text = te.text() if te is not None and hasattr(te, "text") else ""
    except Exception:
        text = ""
    out.extend(item.strip() for item in str(text or "").split(",") if item.strip())
    return out


def contribute(host, ctx) -> Any:
    """ui/sampling_blocks 기여자 — Anima38 블록 하나(또는 없음)와 안내 알림.

    ctx.target: 't2i'(T2I 계열 전부) · 'i2i'(I2I·인페인트·채팅 이미지 편집 — 'I2I·인페인트에도 적용' 토글) ·
    'aux'(Refine·단독/배치 SAM3·단독 ADetailer·손 재구성 — 토글과 무관하게 T2I 값).
    """
    from ui.sampling_blocks import Contribution

    values = current_values(host)
    if values is None:
        return Contribution()
    settings, apply_img2img = values
    kind = model_kind(ctx.model)
    modules = _modules(host) if ctx.backend == a38.BACKEND_COMFY and ctx.target != a38.TARGET_AUX else ()
    result = a38.plan(settings, target=ctx.target, backend=ctx.backend, kind=kind,
                      apply_img2img=apply_img2img, modules=modules)
    contribution = Contribution()
    if result.notice is not None:
        contribution.notices.append(result.notice)
    if result.block is not None:
        contribution.add(a38.SCRIPT_NAME, result.block, provenance=result.provenance)
        # 보낼지는 뒤의 게이트(core/alwayson_propagation)가 정한다 — 여기서는 준비만 기록한다
        logger.debug("Anima38 블록 준비(%s, %s): %s", ctx.target, kind, a38.describe(settings))
    return contribution


# ── 모델 종류 미리 읽기 (데몬 스레드) ──────────────────────────────────────────────
def _next_serial() -> int:
    global _serial
    with _serial_lock:
        _serial += 1
        return _serial


def _push(host, kinds: Mapping) -> None:
    bridge = getattr(host, "vue_bridge", None)
    push = getattr(bridge, "pushWidgetProperty", None)
    if not callable(push):
        return
    try:
        push(MODEL_WIDGET, KIND_PROPERTY, dict(kinds))
    except RuntimeError:
        pass   # 종료 중 QObject 가 이미 사라졌다


def _publish(host, serial: int, kinds: Mapping, header_kinds: Mapping) -> bool:
    """더 새 요청이 없을 때만 기억하고 Vue 에 보낸다(판정과 보내기를 한 락 안에서 — 옛 결과가 새 결과를 덮지 않게)."""
    with _serial_lock:
        if serial != _serial:
            return False
        amk.remember_kinds(header_kinds, replace=True)
        _push(host, kinds)
        return True


def read_model_kinds(titles: Iterable[Any], paths: Mapping[str, str]) -> tuple[dict, dict]:
    """(Vue 로 보낼 {제목: 종류}, 헤더로 확인한 {제목: 종류}). 디스크를 읽는다 — 데몬 스레드에서만."""
    kinds, header = {}, {}
    for title in titles or ():
        name = str(title or "").strip()
        if not name or name in kinds:
            continue
        path = (paths or {}).get(name)
        found = amk.header_kind(path) if path else None
        if found:
            header[name] = found
        kinds[name] = found or amk.name_kind(name)
    return kinds, header


def prewarm_model_kinds(host, titles: Iterable[Any], paths: Optional[Mapping[str, str]] = None,
                        *, start: bool = True) -> threading.Thread:
    """모델 목록의 종류를 데몬 스레드에서 읽는다(헤더만 — 수십~수백 KB). GUI 스레드를 막지 않는다."""
    serial = _next_serial()
    names = [str(title) for title in (titles or ()) if str(title or "").strip()]
    mapping = dict(paths or {})

    def work():
        try:
            kinds, header = read_model_kinds(names, mapping)
        except Exception:
            logger.debug("Anima 모델 종류 읽기 실패(무시)", exc_info=True)
            return
        if _publish(host, serial, kinds, header):
            counts = {kind: sum(1 for k in kinds.values() if k == kind) for kind in amk.KINDS}
            logger.info("[Anima38] 모델 종류 확인: %s",
                        ", ".join(f"{kind}={count}" for kind, count in counts.items() if count) or "-")

    thread = threading.Thread(target=work, daemon=True, name="anima-model-kinds")
    if start:
        thread.start()
    return thread


def clear_model_kinds(host) -> None:
    """연결 실패·백엔드 전환 — 진행 중인 읽기를 버리고 기억·Vue 속성을 비운다."""
    serial = _next_serial()
    with _serial_lock:
        if serial == _serial:
            amk.forget_kinds()
            _push(host, {})


def push_comfy_adapters(host, choices: Optional[Iterable[Any]]) -> None:
    """ComfyUI 의 v1 어댑터 선택지를 카드로(``_a38_adapter`` 위젯 속성). None = Forge·모름 — 카드는 Forge 기능 스냅샷
    ``choices.anima38_adapters`` 규칙을 쓴다. 백엔드를 바꾸거나 연결이 끊기면 None 으로 지운다(옛 목록을 남기지 않는다)."""
    value = None if choices is None else [str(item).strip() for item in choices if str(item or "").strip()]
    bridge = getattr(host, "vue_bridge", None)
    push = getattr(bridge, "pushWidgetProperty", None)
    if not callable(push):
        return
    try:
        push(a38.widget_id("adapter"), COMFY_ADAPTER_PROPERTY, value)
    except RuntimeError:
        pass   # 종료 중 QObject 가 이미 사라졌다


def checkpoint_paths(inventory, models: Iterable[Any]) -> dict:
    """통합 인벤토리 → {백엔드 모델 이름: 로컬 경로}(백엔드 목록과 맞은 항목만). 원격 Forge 는 비어 있다(이름으로만)."""
    paths: dict = {}
    if inventory is None:
        return paths
    for entry in inventory.entries("checkpoints", backend_items=list(models or ())):
        if not isinstance(entry, Mapping) or not entry.get("backendAvailable"):
            continue
        name, path = str(entry.get("runtimeName") or ""), str(entry.get("path") or "")
        if name and path:
            paths.setdefault(name, path)
    return paths


__all__ = [
    "COMFY_ADAPTER_PROPERTY", "KIND_PROPERTY", "apply_saved_settings", "checkpoint_paths", "clear_model_kinds", "contribute", "current_values",
    "get_settings", "init_anima38_proxies", "model_kind", "prewarm_model_kinds", "push_comfy_adapters",
    "read_model_kinds", "set_settings",
]
