"""Opt-in Spectrum settings. No provider installation or global model eviction.

Spectrum and SPEED on ComfyUI (plan D6/R8, C4a):

- the **legacy** SPEED inputs (``speed_enabled`` … — the external sorryhyun ``SpectrumSPDKSampler`` provider behind
  ``ForgeNeoKSamplerCNS``; the app never sends them) cannot be combined with Spectrum: ``validate_spectrum_payload``
  refuses that pair, unchanged;
- sam-extra's **Anima SPEED** (alwayson block ``Anima SPEED``, pack 1.7.0 ``ForgeNeoAnimaSpeed``) is not refused: like
  Forge's Spectrum Integrated, SPEED steps aside when Spectrum is on — the compiler leaves the SPEED node out and warns
  (``core/anima_speed.SKIP_COMFY_SPECTRUM``). ``skips_anima_speed`` is that test, with the truth rule the compiler uses
  for the sampler's own ``spectrum_enabled`` input, so "Spectrum runs" and "SPEED steps aside" cannot disagree.
"""
from __future__ import annotations

import math
from collections.abc import Mapping

LEGACY_SPEED_KEY = "speed_enabled"     # ForgeNeoKSamplerCNS legacy provider input (not sam-extra Anima SPEED)
# core/comfy_workflow_compiler._bool — the values it reads as off (anything else non-empty is on)
_OFF_TEXTS = frozenset({"", "0", "false", "no", "off", "none"})


FIELDS = {
    "window_size": (2.0, 1.0, 4.0, False),
    "flex_window": (0.25, 0.0, 1.0, False),
    "warmup_steps": (6, 1, 150, True),
    "tail_actual_steps": (3, 1, 150, True),
    "blend_w": (0.3, 0.0, 1.0, False),
    "cheby_degree": (3, 1, 10, True),
    "ridge_lambda": (0.1, 0.001, 10.0, False),
    "history_size": (100, 5, 256, True),
}


def spectrum_payload_from_prefs(prefs: Mapping) -> dict:
    settings = prefs.get("comfySpectrum", {})
    if not isinstance(settings, Mapping) or settings.get("enabled") is not True:
        return {}
    payload = {"spectrum_enabled": True, "spectrum_one_sampler_only": True}
    for name, (default, low, high, integer) in FIELDS.items():
        value = settings.get(name, default)
        if isinstance(value, bool):
            raise ValueError(f"Spectrum {name}: 숫자를 입력하세요.")
        try:
            number = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Spectrum {name}: 숫자를 입력하세요.") from exc
        if not math.isfinite(number) or not low <= number <= high or (integer and number != int(number)):
            raise ValueError(f"Spectrum {name}: {low}~{high} 범위를 확인하세요.")
        payload[f"spectrum_{name}"] = int(number) if integer else number
    return payload


def skips_anima_speed(payload: Mapping) -> bool:
    """Does this ComfyUI request run Spectrum on its sampler (``spectrum_enabled`` read like the compiler reads it for
    ``ForgeNeoKSamplerCNS``) — then sam-extra Anima SPEED is left out with a warning (plan D6)."""
    value = payload.get("spectrum_enabled") if isinstance(payload, Mapping) else None
    if value is None:
        return False
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    return str(value).strip().casefold() not in _OFF_TEXTS


def validate_spectrum_payload(payload: Mapping, object_info: Mapping) -> None:
    if payload.get("spectrum_enabled") not in (True, "true", 1):
        return
    settings = {name: payload.get(f"spectrum_{name}", spec[0]) for name, spec in FIELDS.items()}
    spectrum_payload_from_prefs({"comfySpectrum": {"enabled": True, **settings}})
    if "DiTSpectrumPatch" not in object_info:
        raise ValueError("Spectrum 실험 기능을 켰지만 ComfyUI에 DiTSpectrumPatch가 없습니다. 호환 조합 안내를 확인하거나 Spectrum을 끄세요.")
    steps = int(payload.get("steps", 20))
    if int(settings["warmup_steps"]) + int(settings["tail_actual_steps"]) >= steps:
        raise ValueError("Spectrum 준비 단계 + 마지막 실제 계산 단계는 전체 Steps보다 작아야 합니다.")
    if payload.get(LEGACY_SPEED_KEY) in (True, "true", 1):
        # legacy provider keys only — sam-extra Anima SPEED is skipped with a warning instead (skips_anima_speed)
        raise ValueError("Spectrum 실험 기능과 옛 SPEED 입력(speed_enabled — SpectrumSPDKSampler)은 함께 켤 수 없습니다. "
                         "독립적으로 A/B 비교하세요.")
