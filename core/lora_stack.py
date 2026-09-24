"""LoRA 스택 단위 정규화 · 생성용 텍스트 변환 (순수 함수, Qt 비의존, 테스트 용이).

단위 계약 — 예전엔 두 단위가 같은 필드에 섞여 LoRA 가 100배(<lora:x:95.00>)로 생성됐다:

- ``config/ui_prefs.json`` 의 loraStack, Vue loraStack(localStorage) — weight = **정수 퍼센트** (80 → 0.80배)
- 브리지 ``set_lora_stack`` / ``loraStackLoaded``, Python ``host._vue_lora_entries``,
  워크플로 프로파일(version 2+) — weight = **배율** (0.8)
- 단위 표기가 없는 옛 프로파일(version 1)은 목록 전체 기준으로 판정한다(:func:`detect_weight_unit`).

생성 payload 의 LoRA 텍스트는 ``host._vue_lora_entries``(배율) **하나에서만** 파생한다
(:func:`append_lora_stack_to_prompt`). 예전엔 생성 직전에만 갱신되는 ``_vue_lora_text``
미러를 따로 읽어, LoRA 를 모두 끄면 옛 LoRA 가 계속 붙었다.

텍스트 포맷은 Vue 와 같다: ``<lora:name:0.80>`` (소수 2자리), 여러 개는 ``", "`` 로 잇는다.
"""
from __future__ import annotations

import math
import re
from typing import Any, Iterable

UNIT_PERCENT = "percent"
UNIT_MULTIPLIER = "multiplier"
UNIT_AUTO = "auto"
_UNITS = (UNIT_PERCENT, UNIT_MULTIPLIER, UNIT_AUTO)

# 배율 슬라이더 상한은 3 — |w| > 3 이 하나라도 있으면 퍼센트로 저장된 목록이다.
# (항목별 판정은 5% 이하 LoRA·음수 LoRA 를 잘못 가르므로 목록 전체 단위로 본다.)
LEGACY_PERCENT_THRESHOLD = 3.0

_PROMPT_LORA_NAME = re.compile(r"<lora:([^:>]+)", re.IGNORECASE)


def _finite(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def detect_weight_unit(entries: Iterable | None) -> str:
    """단위 표기가 없는 목록의 단위: |weight| > 3 인 항목이 하나라도 있으면 퍼센트."""
    for entry in entries or []:
        if not isinstance(entry, dict):
            continue
        weight = _finite(entry.get("weight"))
        if weight is not None and abs(weight) > LEGACY_PERCENT_THRESHOLD:
            return UNIT_PERCENT
    return UNIT_MULTIPLIER


def normalize_lora_entries(entries: Iterable | None, *, unit: str) -> list[dict]:
    """엔트리 목록을 **배율 단위**로 정규화한 새 목록.

    - dict 가 아닌 항목은 버린다(이름 없는 항목은 남기되 텍스트에서만 빠진다).
    - weight: 퍼센트면 /100, 숫자가 아니거나 없으면 1.0배(100%).
    - enabled: ``False`` 만 꺼짐(키 없으면 켜짐) — 기존 계약 그대로.
    - triggerWords: 리스트가 아니면 [].
    - 그 밖의 키는 보존한다.
    """
    if unit not in _UNITS:
        raise ValueError(f"unknown LoRA weight unit: {unit!r}")
    items = [dict(entry) for entry in (entries or []) if isinstance(entry, dict)]
    effective = detect_weight_unit(items) if unit == UNIT_AUTO else unit
    scale = 100.0 if effective == UNIT_PERCENT else 1.0
    normalized: list[dict] = []
    for item in items:
        raw = _finite(item.get("weight"))
        weight = 1.0 if raw is None else raw / scale
        trigger_words = item.get("triggerWords")
        item.update({
            "name": str(item.get("name", "") or ""),
            "weight": round(weight, 4),
            "enabled": item.get("enabled", True) is not False,
            "triggerWords": list(trigger_words) if isinstance(trigger_words, list) else [],
        })
        normalized.append(item)
    return normalized


def to_percent_entries(entries: Iterable | None) -> list[dict]:
    """배율 엔트리 → ui_prefs/Vue 저장 형식(정수 퍼센트)."""
    result = []
    for entry in normalize_lora_entries(entries, unit=UNIT_MULTIPLIER):
        entry["weight"] = int(round(entry["weight"] * 100))
        result.append(entry)
    return result


def _format(entries: list[dict], exclude: set[str] | None = None) -> str:
    parts = []
    for entry in entries:
        if entry.get("enabled") is False:
            continue
        name = entry["name"].strip()
        if not name or (exclude and name.lower() in exclude):
            continue
        parts.append(f"<lora:{name}:{entry['weight']:.2f}>")
    return ", ".join(parts)


def build_lora_text(entries: Iterable | None, *, unit: str = UNIT_PERCENT) -> str:
    """enabled LoRA 엔트리들을 ``<lora:name:weight>, ...`` 문자열로.

    ``unit`` 기본값은 ui_prefs 저장 형식(퍼센트). 런타임 ``_vue_lora_entries`` 는
    ``unit=UNIT_MULTIPLIER`` 로 부른다. 이름 없음/enabled=False 는 제외, 빈 입력 → ''.
    """
    return _format(normalize_lora_entries(entries, unit=unit))


def prompt_lora_names(prompt: str | None) -> set[str]:
    """프롬프트에 이미 들어 있는 ``<lora:NAME...>`` 이름들(소문자)."""
    return {name.strip().lower() for name in _PROMPT_LORA_NAME.findall(prompt or "")}


def append_lora_stack_to_prompt(prompt: str | None, entries: Iterable | None,
                                *, unit: str = UNIT_MULTIPLIER) -> str:
    """활성 LoRA 스택을 프롬프트 뒤에 붙인다.

    프롬프트에 같은 이름의 ``<lora:...>`` 가 이미 있으면(큐 항목의 EXIF LoRA 등) 스택의 그
    LoRA 는 빼서 이중 적용을 막는다. 스택이 비었거나 전부 꺼져 있으면 프롬프트 그대로.
    """
    base = prompt or ""
    text = _format(normalize_lora_entries(entries, unit=unit), prompt_lora_names(base))
    if not text:
        return base
    return f"{base}, {text}" if base else text
