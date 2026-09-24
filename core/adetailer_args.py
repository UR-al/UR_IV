# core/adetailer_args.py
"""ADetailer REST 슬롯(dict) 단일 소스 — Qt 의존 없음.

예전엔 42키짜리 슬롯 리터럴이 t2i(ui/generator_generation._build_adetailer_slot)와
배치·백엔드 후처리(workers/upscale_worker._build_adetailer_slot) 두 곳에 키 순서까지 복제돼
있었고, backends/webui_backend.adetailer() 는 PyQt 워커 모듈을 lazy import 하는
계층 역전이 있었다. 폴백 기본값도 'face_yolov8s.pt'/0.25 와 'face_yolov8n.pt'/0.4 로
갈라지기 시작했다. 이 모듈이 상수 부분과 기본값을 한 벌로 갖고, 호출처는 바뀌는 값만
override 로 넘긴다. (SAM3 는 core/sam3_args 가 같은 역할을 한다.)

키 순서는 ADetailer 공식 API 스펙 순서 그대로다 — dict(DEFAULT_SLOT) 에 update 하면
기존 페이로드와 JSON 직렬화 결과까지 같다(tests/test_adetailer_args.py 골든 테스트).
"""
from __future__ import annotations

from types import MappingProxyType
from typing import Any, Mapping

DEFAULT_MODEL = 'face_yolov8n.pt'
DEFAULT_CONFIDENCE = 0.3
DEFAULT_DENOISE = 0.4
DEFAULT_SAMPLER = 'DPM++ 2M Karras'
DEFAULT_SCHEDULER = 'Use same scheduler'

# 공식 REST API 스펙 순서. 값은 '사용자가 따로 정하지 않았을 때' 보낼 기본값.
DEFAULT_SLOT: Mapping[str, Any] = MappingProxyType({
    "ad_model": DEFAULT_MODEL,
    "ad_model_classes": "",
    "ad_tab_enable": True,
    "ad_prompt": "",
    "ad_negative_prompt": "",
    "ad_confidence": DEFAULT_CONFIDENCE,
    "ad_mask_filter_method": "Area",
    "ad_mask_k": 0,
    "ad_mask_min_ratio": 0.0,
    "ad_mask_max_ratio": 1.0,
    "ad_dilate_erode": 4,
    "ad_x_offset": 0,
    "ad_y_offset": 0,
    "ad_mask_merge_invert": "None",
    "ad_mask_blur": 4,
    "ad_denoising_strength": DEFAULT_DENOISE,
    "ad_inpaint_only_masked": True,
    "ad_inpaint_only_masked_padding": 32,
    "ad_use_inpaint_width_height": False,
    "ad_inpaint_width": 512,
    "ad_inpaint_height": 512,
    "ad_use_steps": False,
    "ad_steps": 28,
    "ad_use_cfg_scale": False,
    "ad_cfg_scale": 7.0,
    "ad_use_checkpoint": False,
    "ad_checkpoint": None,
    "ad_use_vae": False,
    "ad_vae": None,
    "ad_use_sampler": False,
    "ad_sampler": DEFAULT_SAMPLER,
    "ad_scheduler": DEFAULT_SCHEDULER,
    "ad_use_noise_multiplier": False,
    "ad_noise_multiplier": 1.0,
    "ad_use_clip_skip": False,
    "ad_clip_skip": 1,
    "ad_restore_face": False,
    "ad_controlnet_model": "None",
    "ad_controlnet_module": "None",
    "ad_controlnet_weight": 1.0,
    "ad_controlnet_guidance_start": 0.0,
    "ad_controlnet_guidance_end": 1.0,
})


def build_slot(**overrides: Any) -> dict:
    """DEFAULT_SLOT 사본에 overrides(ad_* 키)를 덮어쓴 새 슬롯.

    모르는 키는 오타로 보고 KeyError — 조용히 확장이 무시하는 필드를 보내지 않게.
    """
    unknown = sorted(set(overrides) - set(DEFAULT_SLOT))
    if unknown:
        raise KeyError(f"unknown ADetailer slot keys: {', '.join(unknown)}")
    slot = dict(DEFAULT_SLOT)
    slot.update(overrides)
    return slot


def build_simple_slot(model: str = DEFAULT_MODEL, confidence: float = DEFAULT_CONFIDENCE,
                      denoise: float = DEFAULT_DENOISE, prompt: str = '') -> dict:
    """배치/후처리용 간단 슬롯 — 모델·신뢰도·denoise·프롬프트만 정한다."""
    return build_slot(
        ad_model=model,
        ad_confidence=confidence,
        ad_denoising_strength=denoise,
        ad_prompt=prompt,
    )


def slot_from_settings(settings: Mapping[str, Any]) -> dict:
    """배치 settings(ad_model/ad_confidence/ad_denoise/ad_prompt/ad_negative) → 슬롯.

    WebUIBackend.adetailer() 가 adetailer_args 를 받지 못했을 때의 폴백. 빈 값은 기본값.
    """
    settings = settings if isinstance(settings, Mapping) else {}

    def _pick(key: str, default):
        value = settings.get(key)
        return default if value is None or value == '' else value

    slot = build_simple_slot(
        model=_pick('ad_model', DEFAULT_MODEL),
        confidence=_pick('ad_confidence', DEFAULT_CONFIDENCE),
        denoise=_pick('ad_denoise', DEFAULT_DENOISE),
        prompt=settings.get('ad_prompt') or '',
    )
    if settings.get('ad_negative'):
        slot['ad_negative_prompt'] = settings['ad_negative']
    return slot
