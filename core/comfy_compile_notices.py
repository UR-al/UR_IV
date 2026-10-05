# core/comfy_compile_notices.py
"""ComfyUI 컴파일 경고 → sam-extra 결과 알림(Anima SPEED·Extra Samplers) — 순수 로직(Qt·네트워크 없음).

ComfyUI 컴파일러는 sam-extra 기능을 그래프에서 빼거나 그 기능이 다르게 돌 때 ``compile(warnings=…)`` 목록에
``{"code", "reason", "feature", "cause"}`` 를 남긴다(core/comfy_workflow_compiler ``_feature_warning``·
``_extra_samplers_warnings``). Forge 에서는 같은 일이 결과 infotext(status)를 거쳐 알림이 되므로, ComfyUI 백엔드가 그
경고를 **Forge 결과와 같은 알림**(같은 code·feature·detail — 생성 전 알림과 겹치면 ``NoticeThrottle`` 이 한 번만 띄운다)으로
바꿔 generation info 에 싣는다(backends/comfyui_backend ``_with_result_notices``):

- SPEED 건너뜀(``speed_skipped`` — 인페인트·마스크·ControlNet·ComfyUI Spectrum·해상도 비율 하나) →
  ``sam_extra_notices.speed_static_skip_notice``(정보, detail = ``core/anima_speed`` 의 SKIP_* 코드 — Forge: status
  'skipped - …')
- SPEED 설정 오류(``speed_invalid``) → CODE_SPEED_INVALID 경고(Forge: 'invalid settings - …')
- Euler (SMEA) Dy 보조 스텝 건너뜀(``extra_samplers_substeps_skipped`` — ComfyUI Spectrum) →
  CODE_EXTRA_SAMPLERS_SUBSTEPS_SKIPPED 정보(detail 'Spectrum' — Forge: 'Extra Samplers status: dy sub-steps skipped
  (Spectrum)')

Forge 도 알리지 않는 것은 알리지 않는다(로그만): CFG++ 샘플러의 CFG > 2(``extra_samplers_cfg_pp`` — Forge 는 콘솔에만
남기고, 메인·Hires 패스의 CFG 는 Sampler 줄 힌트가 생성 전에 보인다 — frontend/src/utils/extraSamplers.ts
``cfgWarnings``), Colorcraft 돌 항목 없음(``colorcraft_no_entries`` — Forge 도 붙일 것이 없으면 말하지 않는다:
``sam_extra_notices._colorcraft_result_notices``). VAE DeGrid 경고는 core/comfy_degrid_report 가 맡는다.
"""
from __future__ import annotations

from typing import Any, Iterable, Mapping, Optional

from core import anima_speed, extra_samplers
from core import sam_extra_notices as notices

FEATURE_SPEED = "speed"
FEATURE_EXTRA_SAMPLERS = "extra_samplers"

# 컴파일러의 SPEED 건너뜀 경고 reason 은 ``anima_speed.SKIP_MESSAGES[코드]`` 그대로다(_speed_plan) → 코드로 되돌린다
# (생성 전 기여자 알림과 같은 detail — 같은 생성의 같은 이유는 한 번만 보인다).
_SPEED_SKIP_CODES = {text: code for code, text in anima_speed.SKIP_MESSAGES.items()}

COMFY_DY_SPECTRUM_TEXT = "ComfyUI 의 Spectrum 실험 기능이 켜져 있습니다(Forge 의 Spectrum Integrated 와 같은 규칙)"


def _text(value: Any) -> str:
    return " ".join(str(value or "").split())


def speed_invalid_notice(reason: Any) -> notices.Notice:
    """ComfyUI: SPEED 설정을 sam-extra 가 거절하는 값이라 컴파일러가 노드를 뺐다(경고 — Forge 'invalid settings')."""
    text = _text(reason)
    return notices.Notice(
        notices.CODE_SPEED_INVALID, notices.LEVEL_WARNING,
        f"{text} — 이번 생성은 SPEED 없이 샘플링했습니다. 파라미터 카드의 SPEED 칸을 고치세요.",
        feature=FEATURE_SPEED, detail=text)


def dy_substeps_skipped_notice() -> notices.Notice:
    """ComfyUI: Spectrum 이 켜진 샘플러라 Euler (SMEA) Dy (CFG++) 가 보조 스텝 없이 돌았다(정보)."""
    return notices.Notice(
        notices.CODE_EXTRA_SAMPLERS_SUBSTEPS_SKIPPED, notices.LEVEL_INFO,
        f"Euler (SMEA) Dy 의 보조 스텝을 건너뛰고 보조 스텝 없는 Euler 스텝으로 생성했습니다 — {COMFY_DY_SPECTRUM_TEXT}.",
        feature=FEATURE_EXTRA_SAMPLERS, detail=extra_samplers.DY_REASON_SPECTRUM)


def compile_warning_notices(warnings: Optional[Iterable[Any]]) -> list[notices.Notice]:
    """컴파일 경고 목록 → 알림(위 규칙, 경고 순서, 같은 알림은 한 번). 다른 기능·모르는 모양은 뺀다."""
    out: list[notices.Notice] = []
    for item in warnings or ():
        if not isinstance(item, Mapping):
            continue
        feature, code, reason = item.get("feature"), item.get("code"), _text(item.get("reason"))
        if feature == FEATURE_SPEED and code == notices.CODE_SPEED_SKIPPED:
            out.append(notices.speed_static_skip_notice(_SPEED_SKIP_CODES.get(reason, reason)))
        elif feature == FEATURE_SPEED and code == notices.CODE_SPEED_INVALID:
            out.append(speed_invalid_notice(reason))
        elif feature == FEATURE_EXTRA_SAMPLERS and code == notices.CODE_EXTRA_SAMPLERS_SUBSTEPS_SKIPPED:
            out.append(dy_substeps_skipped_notice())
    seen: set = set()
    unique = []
    for notice in out:
        if notice.key not in seen:
            seen.add(notice.key)
            unique.append(notice)
    return unique


__all__ = ["COMFY_DY_SPECTRUM_TEXT", "FEATURE_EXTRA_SAMPLERS", "FEATURE_SPEED", "compile_warning_notices",
           "dy_substeps_skipped_notice", "speed_invalid_notice"]
