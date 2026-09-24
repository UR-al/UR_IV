"""Vue InpaintView 페이로드 → img2img 인페인트 요청 (Qt 비의존 순수 로직).

예전에는 Vue 페이로드를 숨은 레거시 ``InpaintTab`` 위젯에 넣었다가 다시 읽었다. 그래서
  1) 화면의 '마스크 영역 초기값'·'인페인트 범위'는 아무도 읽지 않아 숨은 위젯의
     fill=원본·원본 해상도=켜짐이 늘 쓰였고(화면 기본값은 정반대인 채우기·전체 이미지),
  2) steps/cfg/seed/negative 는 Vue 가 보내지 않아 숨은 값(설정 복원·PNG Info 전송이
     바꾼 값)이 보이지 않게 적용됐고,
  3) data URL 로 받은 이미지도 ``current_image_path`` 를 비우지 않아 **이전 이미지의 크기**
     (없으면 1024×1024)가 width/height 로 나갔다 — ComfyUI 는 stretch 라 비율이 찌그러졌다.
여기서는 페이로드만 보고 요청 전체를 만든다. 크기는 실제로 보낼 이미지 바이트의 헤더에서
구한다(경로면 파일, data URL 이면 디코드한 바이트).

Vue 옵션 ↔ A1111/Forge 필드:
  mask_content 0..3 → ``inpainting_fill`` (채우기 / 원본 유지 / latent noise / latent nothing)
  inpaint_area  0/1 → ``inpaint_full_res`` (전체 이미지=False / 마스크 영역만=True)
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Mapping, Optional

from core.image_payload import (
    ImagePayloadError,
    decode_base64_image,
    encode_image_bytes,
    encode_image_file,
    image_size_from_bytes,
    strip_data_url,
)

# 화면 기본값 — 숨은 탭이 실제로 쓰던 값(원본 유지 · 마스크 영역만)과 같게 둔다.
# 이 연결이 생기면서 결과가 조용히 바뀌지 않도록.
DEFAULT_DENOISING = 0.75
DEFAULT_MASK_CONTENT = 1
DEFAULT_INPAINT_AREA = 1
DEFAULT_MASK_BLUR = 4
DEFAULT_PADDING = 32
DEFAULT_STEPS = 20
DEFAULT_CFG = 7.0
DEFAULT_SEED = -1

MASK_CONTENT_COUNT = 4          # fill / original / latent noise / latent nothing
MAX_STEPS = 150
MAX_CFG = 30.0
MAX_MASK_BLUR = 64
MAX_PADDING = 256
MAX_SEED = 2 ** 32 - 1


@dataclass(frozen=True)
class InpaintRequest:
    payload: dict          # 백엔드 img2img payload (alwayson_scripts 는 아직 비어 있다)
    source_path: str       # 경로로 받았으면 검증된 절대 경로, data URL 이면 ''
    width: int
    height: int


def _finite_float(value, default: float) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if math.isfinite(out) else default


def _clamped_int(value, default: int, lo: int, hi: int) -> int:
    try:
        out = int(float(str(value).strip()))
    except (TypeError, ValueError, OverflowError):
        return default
    return max(lo, min(hi, out))


def _clamped_float(value, default: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, _finite_float(value, default)))


def parse_seed(value) -> int:
    """'-1'·'' ·잘못된 값 → -1, 그 밖은 0..2^32-1 로 자른다."""
    seed = _clamped_int(value, DEFAULT_SEED, -1, MAX_SEED)
    return seed


def _default_resolver(raw: str) -> Optional[str]:
    from core.path_safety import safe_input_path
    return safe_input_path(raw)


def build_inpaint_request(
    vue_payload: Mapping,
    *,
    main_prompt: str = "",
    main_negative: str = "",
    resolve_path: Callable[[str], Optional[str]] = _default_resolver,
) -> InpaintRequest:
    """InpaintView 의 ``generate_inpaint`` 페이로드 → 백엔드 요청.

    실패는 사용자에게 보여 줄 문구를 담은 ``ImagePayloadError`` 로 올린다.
    프롬프트·네거티브가 비었으면 T2I 의 값(main_prompt/main_negative)을 쓴다 — 예전과 같다.
    """
    data = vue_payload if isinstance(vue_payload, Mapping) else {}

    image_field = strip_data_url(data.get("image"))
    raw_path = str(data.get("image_path") or "").strip()
    if image_field:
        # 파일 선택·드롭·조명 편집 결과 — 경로가 없다. 이전 경로의 크기를 쓰면 안 된다.
        encoded = encode_image_bytes(decode_base64_image(image_field, label="Inpaint"), label="Inpaint")
        source_path = ""
    elif raw_path:
        resolved = resolve_path(raw_path)
        if not resolved:
            raise ImagePayloadError("Inpaint: 입력 이미지 파일을 찾을 수 없습니다")
        encoded = encode_image_file(resolved, label="Inpaint")
        source_path = resolved
    else:
        raise ImagePayloadError("Inpaint: 입력 이미지가 없습니다")

    mask = strip_data_url(data.get("mask"))
    if not mask:
        raise ImagePayloadError("Inpaint: 마스크를 그려주세요")
    # 마스크가 실제 이미지인지 먼저 확인 — 깨진 마스크로 백엔드 왕복을 하지 않는다.
    image_size_from_bytes(decode_base64_image(mask, label="Inpaint 마스크"), label="Inpaint 마스크")

    prompt = str(data.get("prompt") or "").strip() or str(main_prompt or "")
    negative = str(data.get("negative_prompt") or "").strip() or str(main_negative or "")

    mask_content = _clamped_int(data.get("mask_content", DEFAULT_MASK_CONTENT),
                                DEFAULT_MASK_CONTENT, 0, MASK_CONTENT_COUNT - 1)
    inpaint_area = _clamped_int(data.get("inpaint_area", DEFAULT_INPAINT_AREA),
                                DEFAULT_INPAINT_AREA, 0, 1)

    payload = {
        "init_images": [encoded.b64],
        "mask": mask,
        "prompt": prompt,
        "negative_prompt": negative,
        "denoising_strength": _clamped_float(data.get("denoising"), DEFAULT_DENOISING, 0.0, 1.0),
        "inpainting_fill": mask_content,
        "inpaint_full_res": inpaint_area == 1,
        "inpaint_full_res_padding": _clamped_int(data.get("padding"), DEFAULT_PADDING, 0, MAX_PADDING),
        "mask_blur": _clamped_int(data.get("mask_blur"), DEFAULT_MASK_BLUR, 0, MAX_MASK_BLUR),
        "inpainting_mask_invert": 0,
        "resize_mode": 0,
        "steps": _clamped_int(data.get("steps"), DEFAULT_STEPS, 1, MAX_STEPS),
        "cfg_scale": _clamped_float(data.get("cfg"), DEFAULT_CFG, 0.0, MAX_CFG),
        "seed": parse_seed(data.get("seed", DEFAULT_SEED)),
        "width": encoded.width,
        "height": encoded.height,
        "send_images": True,
        "save_images": True,
        "alwayson_scripts": {},
    }
    return InpaintRequest(payload=payload, source_path=source_path,
                          width=encoded.width, height=encoded.height)
