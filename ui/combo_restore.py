# ui/combo_restore.py
"""백엔드 연결 때 다시 채우는 콤보의 선택 보존 (감사 #29).

on_webui_info_loaded 는 모델·샘플러·VAE 등 콤보를 비우고 새 목록으로 채운다. 예전엔 그 뒤
``load_settings()`` 전체를 다시 돌렸는데, 그러면
- 저장하지 않은 프롬프트 편집이 prompt_settings.json 값으로 덮였고(설치 경로 변경·PRIMARY 전환
  뒤의 재연결 포함),
- ``_restore_backend_settings`` 가 같은 URL 인데도 어댑터를 새로 만들어 BACKEND_URL_CHANGED 로
  XYZ 축 설정이 지워졌고,
- 레거시 cleaning_options 가 Vue 정리 토글을 덮었다.

이제는 비우기 직전의 선택을 기억(:func:`snapshot_backend_combos`)했다가 새 목록에서 같은 항목을
다시 고른다(:func:`restore_backend_combos`). 기억한 값이 비었거나 새 목록에 없을 때만 디스크 값을
본다. 둘 다 없으면 ComboBoxProxy.addItems 가 고른 항목(첫 항목 — Vue 에도 알림)을 그대로 둔다.

'같은 항목' 은 load_settings(ui/generation_settings_apply)와 같은 matcher 로 가린다 — 체크포인트·
VAE 는 해시·경로 무시, 샘플러·스케줄러는 Forge 표기 ↔ ComfyUI 이름 별칭. 정확히 같은 문자열만
보면 Forge 에서 고른 'DPM++ 2M'/'Karras' 가 ComfyUI 로 연결할 때 'euler'/'normal' 로 조용히 바뀌었다.

연결 오류(on_webui_info_error)가 콤보를 비운 뒤 다시 연결하는 경우, 스냅숏은
ComboBoxProxy.preservedText 로 비우기 직전의 **실제 선택**을 읽는다. currentText() 는 비운 콤보에서
부팅 때 setText 로 미뤄 둔 옛 값(fallback)을 돌려줘, 사용자가 나중에 고른 VAE 대신 부팅 값으로
되돌아갔다.
"""
from __future__ import annotations

import json
import os
from typing import Any, Callable, Optional

from core.combo_selection import Matcher, pick_combo_index
from ui.generation_settings_apply import (
    match_checkpoint_index,
    match_plain_index,
    match_sampler_index,
    match_scheduler_index,
)


def _attr(name: str) -> Callable[[Any], Any]:
    return lambda host: getattr(host, name, None)


def _slot(slots_attr: str, key: str) -> Callable[[Any], Any]:
    def resolve(host):
        widgets = getattr(host, slots_attr, None)
        return widgets.get(key) if isinstance(widgets, dict) else None
    return resolve


def _saved(*path: str) -> Callable[[dict], str]:
    def read(settings: dict) -> str:
        value: Any = settings
        for part in path:
            if not isinstance(value, dict):
                return ""
            value = value.get(part)
        return str(value or "")
    return read


#: (키, 콤보 찾기, prompt_settings.json 에서 읽기, 같은 항목 찾기)
#: matcher 는 load_settings 가 같은 콤보에 쓰는 것과 같다(ui/generation_settings_apply.apply_generation_settings,
#: ui/generator_settings._set_slot_settings/_set_sam3_settings) — 연결 경로와 불러오기 경로가 갈라지지 않게.
BACKEND_COMBOS: tuple[tuple[str, Callable[[Any], Any], Callable[[dict], str], Matcher], ...] = (
    ("model_combo", _attr("model_combo"), _saved("model"), match_checkpoint_index),
    ("sampler_combo", _attr("sampler_combo"), _saved("sampler"), match_sampler_index),
    ("scheduler_combo", _attr("scheduler_combo"), _saved("scheduler"), match_scheduler_index),
    ("upscaler_combo", _attr("upscaler_combo"), _saved("hires_upscaler"), match_plain_index),
    ("hires_checkpoint_combo", _attr("hires_checkpoint_combo"), _saved("hires_checkpoint"), match_checkpoint_index),
    ("hires_sampler_combo", _attr("hires_sampler_combo"), _saved("hires_sampler"), match_sampler_index),
    ("hires_scheduler_combo", _attr("hires_scheduler_combo"), _saved("hires_scheduler"), match_scheduler_index),
    ("vae_main_combo", _attr("vae_main_combo"), _saved("vae_main"), match_checkpoint_index),
    ("s1.checkpoint_combo", _slot("s1_widgets", "checkpoint_combo"), _saved("adetailer_slot1", "checkpoint"), match_checkpoint_index),
    ("s1.vae_combo", _slot("s1_widgets", "vae_combo"), _saved("adetailer_slot1", "vae"), match_checkpoint_index),
    ("s1.sampler_combo", _slot("s1_widgets", "sampler_combo"), _saved("adetailer_slot1", "sampler"), match_sampler_index),
    ("s1.scheduler_combo", _slot("s1_widgets", "scheduler_combo"), _saved("adetailer_slot1", "scheduler"), match_scheduler_index),
    ("s2.checkpoint_combo", _slot("s2_widgets", "checkpoint_combo"), _saved("adetailer_slot2", "checkpoint"), match_checkpoint_index),
    ("s2.vae_combo", _slot("s2_widgets", "vae_combo"), _saved("adetailer_slot2", "vae"), match_checkpoint_index),
    ("s2.sampler_combo", _slot("s2_widgets", "sampler_combo"), _saved("adetailer_slot2", "sampler"), match_sampler_index),
    ("s2.scheduler_combo", _slot("s2_widgets", "scheduler_combo"), _saved("adetailer_slot2", "scheduler"), match_scheduler_index),
    ("sam3.checkpoint", _slot("sam3_widgets", "checkpoint"), _saved("sam3_settings", "checkpoint"), match_checkpoint_index),
)


def _combo_items(combo) -> list[str]:
    items = getattr(combo, "_items", None)
    if isinstance(items, list):
        return list(items)
    try:
        return [combo.itemText(index) for index in range(combo.count())]
    except Exception:
        return []


def _selection_text(combo) -> str:
    """되살릴 선택 — ComboBoxProxy 는 preservedText(지금 항목 → 아직 못 고른 설정값 → 연결 오류로
    비우기 직전 항목), 그 밖의 콤보는 currentText."""
    reader = getattr(combo, "preservedText", None)
    if not callable(reader):
        reader = combo.currentText
    return str(reader() or "")


def snapshot_backend_combos(host) -> dict[str, str]:
    """비우기 직전의 선택. 목록 도착 전 설정값(부팅 load_settings 의 setText)도 포함한다."""
    snapshot: dict[str, str] = {}
    for key, resolve, _read, _match in BACKEND_COMBOS:
        combo = resolve(host)
        if combo is None or not hasattr(combo, "currentText"):
            continue
        try:
            snapshot[key] = _selection_text(combo)
        except Exception:
            snapshot[key] = ""
    return snapshot


def load_saved_prompt_settings(path: Optional[str] = None) -> dict:
    """prompt_settings.json — 없거나 깨졌으면 ``{}``."""
    if path is None:
        from config import PROMPT_SETTINGS_FILE
        path = PROMPT_SETTINGS_FILE
    try:
        if not os.path.exists(path):
            return {}
        with open(path, "r", encoding="utf-8") as handle:
            data = json.load(handle)
    except (OSError, ValueError):
        return {}
    return data if isinstance(data, dict) else {}


def restore_backend_combos(
    host,
    preserved: dict[str, str],
    load_saved: Callable[[], dict] = load_saved_prompt_settings,
) -> list[str]:
    """새 목록에서 보존한 선택(없으면 디스크 값)을 다시 고른다. 고른 콤보 키 목록을 돌려준다.

    디스크는 필요할 때 한 번만 읽는다. 선택 변경은 사용자 편집이 아니므로 호스트의
    ``is_programmatic_change`` 를 세운 채로 바꾼다(load_settings 와 같은 규칙).
    """
    saved: Optional[dict] = None
    restored: list[str] = []
    previous_flag = getattr(host, "is_programmatic_change", False)
    host.is_programmatic_change = True
    try:
        for key, resolve, read_saved, match in BACKEND_COMBOS:
            combo = resolve(host)
            if combo is None or not hasattr(combo, "setCurrentIndex"):
                continue
            items = _combo_items(combo)
            if not items:
                continue
            index = pick_combo_index(items, [preserved.get(key, "")], match=match)
            if index < 0:
                if saved is None:
                    saved = load_saved() or {}
                index = pick_combo_index(items, [read_saved(saved)], match=match)
            if index < 0:
                continue
            combo.setCurrentIndex(index)
            restored.append(key)
    finally:
        host.is_programmatic_change = previous_flag
    return restored
