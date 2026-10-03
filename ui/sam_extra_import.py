# ui/sam_extra_import.py
"""sam-extra 가져오기 — 붙여 넣은 infotext 의 카드 기능 값(Colorcraft·Anima SPEED)을 카드에 되살린다(계획 X3c).

브리지 액션 하나 ``sam_extra_import {text, only?}`` 가 이 모듈로 온다(Colorcraft 카드의 가져오기 칸, PNG Info·갤러리의
'sam-extra 값 가져오기' 단추). 다음 기능도 액션을 새로 만들지 않고 ``FEATURES`` 에 한 줄을 더한다.

- ``text``: 이미지 infotext 전체, ``Key: value`` 한 줄(들), 또는 맨 값 — Colorcraft 는 확장의 형식 구분 그대로
  (``v1;mods=`` 자기 형식 · ``MOD(`` 업스트림 · 다른 ``v1;`` aoleg 포크, core/colorcraft.from_value), SPEED 는 요약
  (``mode=…; threshold=…``)
- ``only``: ``"colorcraft"`` / ``"speed"`` — 그 기능만(없으면 글에 있는 것 전부)

읽기는 ``core.image_metadata.parse_infotext``(프롬프트 줄이 없는 ``Key: value`` 한 줄은 그 줄을 매개변수 줄로 다시
읽는다) → ``core.colorcraft.from_infotext``·``core.anima_speed`` 붙여 넣기 규칙. 글에 그 기능의 키가 있을 때만 그 카드를
바꾼다(없는 기능은 그대로 — 가져오기는 '이미지에 있는 것'만 옮긴다). 결과는 토스트 하나: 무엇을 가져왔는지, 없으면 경고.
디버그 값은 가져오지 않는다(확장도 붙여 넣지 않는다). 앱 전용 'I2I 에도 적용' 토글은 그대로 둔다.

판정(``parse_import``)은 순수하고, 위젯에 쓰기·토스트(``import_infotext``)만 GUI 스레드에서 한다.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Any, Mapping, Optional

from core import anima_speed as asp
from core import colorcraft as cc

logger = logging.getLogger(__name__)

FEATURE_COLORCRAFT, FEATURE_SPEED = "colorcraft", "speed"
FEATURES = (FEATURE_COLORCRAFT, FEATURE_SPEED)
FEATURE_LABELS = {FEATURE_COLORCRAFT: "Colorcraft", FEATURE_SPEED: "Anima SPEED"}
# 글에서 이 기능을 알아보는 infotext 키(core 상수 — 확장 계약)
FEATURE_KEYS = {FEATURE_COLORCRAFT: (cc.INFOTEXT_KEY, cc.FOREIGN_KEY), FEATURE_SPEED: (asp.INFOTEXT_KEY,)}
_SPEED_SUMMARY_KEYS = frozenset({"mode", "threshold", "preset", "scales"})
MAX_TEXT = 65536            # 붙여 넣기 칸 한도(infotext 한 장은 몇 KB)

NOTHING_MESSAGE = ("가져올 sam-extra 값이 없습니다 — 이미지 정보(infotext) 전체나 'SAM Extra Colorcraft: …'·'Colorcraft: …'·"
                   "'Anima SPEED: …' 줄, 또는 그 값을 붙여 넣으세요.")


@dataclass(frozen=True)
class ImportPlan:
    colorcraft: Optional[cc.Config] = None      # 되살릴 Colorcraft 설정(켬 — 확장 붙여 넣기 규칙)
    speed: Optional[dict] = None                # 'Anima SPEED' 키를 가진 매개변수(core/anima_speed.from_infotext 가 읽는다)
    asked: tuple = FEATURES                     # 찾은 기능(only 가 있으면 그 하나)

    @property
    def found(self) -> tuple:
        return tuple(name for name, value in ((FEATURE_COLORCRAFT, self.colorcraft), (FEATURE_SPEED, self.speed))
                     if value is not None)


def _has_any(params: Mapping, feature_names) -> bool:
    return any(key in params for name in feature_names for key in FEATURE_KEYS[name])


def _key_lines(raw: str) -> dict:
    """``Key: value`` 줄마다 첫 ':' 에서 나눈다(값의 쉼표·세미콜론은 그대로, JSON 따옴표 값은 푼다)."""
    out: dict = {}
    for line in raw.splitlines():
        key, sep, value = line.partition(":")
        key, value = key.strip(), value.strip()
        if not sep or not key:
            continue
        if len(value) >= 2 and value[0] == value[-1] == '"':
            try:
                value = json.loads(value)
            except ValueError:
                pass
        out[key] = value
    return out


def parse_params(text: Any, features=FEATURES) -> dict:
    """글 → infotext 매개변수. 이미지 infotext 전체(``parse_infotext``)에 그 기능의 키가 없으면 ``Key: value`` 줄로 읽는다
    (프롬프트 줄이 없는 한 줄은 infotext 파서가 프롬프트로 본다)."""
    from core.image_metadata import parse_infotext

    raw = str(text or "").strip()
    if not raw:
        return {}
    params = dict(parse_infotext(raw).parameters or {})
    if _has_any(params, features):
        return params
    lines = _key_lines(raw)
    return lines if _has_any(lines, features) else params


def parse_import(text: Any, only: Optional[str] = None) -> ImportPlan:
    """붙여 넣은 글에서 가져올 값(순수). ``only`` 가 모르는 이름이면 ``ValueError``."""
    if only not in (None, "", *FEATURES):
        raise ValueError(f"모르는 sam-extra 기능: {only!r}")
    asked = (only,) if only else FEATURES
    raw = str(text or "").strip()[:MAX_TEXT]
    params = parse_params(raw, asked)
    config = speed = None
    if FEATURE_COLORCRAFT in asked:
        config = cc.from_infotext(params) if _has_any(params, (FEATURE_COLORCRAFT,)) else None
        if config is None and not _has_any(params, FEATURES):
            config = cc.from_value(raw)                    # 맨 값(자기·업스트림·포크 형식)
    if FEATURE_SPEED in asked:
        if asp.INFOTEXT_KEY in params:
            speed = params
        elif config is None and not _has_any(params, FEATURES) and _SPEED_SUMMARY_KEYS & set(asp.parse_summary(raw)):
            speed = {asp.INFOTEXT_KEY: raw}                # 맨 요약 값
    return ImportPlan(config, speed, asked)


def _notify(host, level: str, message: str) -> None:
    bridge = getattr(host, "vue_bridge", None)
    signal = getattr(bridge, "showNotification", None)
    if signal is not None:
        signal.emit(level, message)


def import_infotext(host, payload: Any) -> ImportPlan:
    """``sam_extra_import`` 액션 — 글을 읽어 카드 위젯에 쓰고 토스트 하나로 알린다. 읽은 계획을 돌려준다."""
    from ui.anima_speed_ui import apply_infotext as apply_speed
    from ui.colorcraft_ui import apply_config as apply_colorcraft

    payload = payload if isinstance(payload, Mapping) else {}
    only = payload.get("only") or None
    try:
        plan = parse_import(payload.get("text"), only)
    except ValueError as exc:
        _notify(host, "warning", str(exc))
        return ImportPlan(asked=())
    done = []
    if plan.colorcraft is not None and apply_colorcraft(host, plan.colorcraft):
        done.append(f"Colorcraft({cc.describe(plan.colorcraft)})")
    if plan.speed is not None:
        restored = apply_speed(host, plan.speed)
        if restored is not None:
            done.append(f"Anima SPEED({asp.describe(restored) if restored.enabled else '꺼짐'})")
    if done:
        _notify(host, "success", "sam-extra 값을 가져왔습니다 — " + ", ".join(done))
    elif only:
        _notify(host, "warning", f"가져올 {FEATURE_LABELS[only]} 값이 없습니다 — " + NOTHING_MESSAGE.split(" — ", 1)[1])
    else:
        _notify(host, "warning", NOTHING_MESSAGE)
    logger.info("sam-extra 가져오기(%s): %s", only or "전부", ", ".join(plan.found) or "없음")
    return plan


__all__ = [
    "FEATURES", "FEATURE_COLORCRAFT", "FEATURE_KEYS", "FEATURE_LABELS", "FEATURE_SPEED", "ImportPlan", "MAX_TEXT",
    "NOTHING_MESSAGE", "import_infotext", "parse_import", "parse_params",
]
