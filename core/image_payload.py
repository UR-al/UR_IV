"""생성 요청용 입력 이미지 → base64 (Qt 비의존 순수 로직).

I2I/Inpaint 가 경로로 받은 이미지(갤러리·히스토리에서 보낸 것)를 백엔드로 보낼 때,
예전에는 레거시 탭의 ``_load_image`` 가 Qt 메인 스레드에서 QPixmap 디코드 → 스무스
스케일 → PIL 재디코드 → PNG 재인코딩 → base64 를 전부 했다. 2048×1144 기준 약 250ms가
클릭마다 GUI 를 멈췄는데, 원본 바이트를 그대로 base64 로 바꾸면 약 10ms 이고 백엔드가
받는 픽셀도 같다.

규칙:

* **원본 바이트 그대로** — PNG/JPEG/WebP 단일 프레임이고, 알파가 없고(RGB/L),
  EXIF 방향이 없거나 1일 때. 백엔드(Forge/ComfyUI)가 디코드하는 픽셀이 재인코딩본과 같다.
* **RGB PNG 재인코딩** — 그 밖(알파·팔레트 투명·CMYK·16비트·BMP/GIF/TIFF·방향 태그).
  예전 동작과 같이 RGB 로 바꾸되, 방향 태그는 적용해 세운다: 브라우저(QtWebEngine)는
  EXIF 방향을 적용해 보여 주고, 인페인트 마스크도 그 방향의 캔버스에 그려진다.
  알파는 흰 바탕에 합성한다(Forge 가 RGBA init 이미지에 하는 처리와 같다).

반환 크기는 백엔드가 받을 이미지(방향 적용 후)의 크기다 — 인페인트 payload 의
width/height 로 그대로 쓴다.
"""
from __future__ import annotations

import base64
import binascii
import io
import math
from dataclasses import dataclass
from typing import Callable, Optional, Union

from PIL import Image, ImageOps

# 원본 바이트를 그대로 보내도 되는 형식·모드
_PASSTHROUGH_FORMATS = frozenset({"PNG", "JPEG", "WEBP"})
_PASSTHROUGH_MODES = frozenset({"RGB", "L"})
_EXIF_ORIENTATION = 0x0112
# 입력 한도 — 생성 입력으로 이보다 큰 파일/픽셀은 받지 않는다(디컴프레션 폭탄 방지).
MAX_INPUT_BYTES = 128 * 1024 * 1024
MAX_INPUT_PIXELS = 64 * 1024 * 1024


class ImagePayloadError(ValueError):
    """사용자에게 그대로 보여 줄 수 있는 입력 이미지 오류."""


@dataclass(frozen=True)
class EncodedImage:
    b64: str
    width: int
    height: int
    reencoded: bool   # True 면 RGB PNG 로 다시 인코딩했다


def _orientation(image: Image.Image) -> int:
    try:
        return int(image.getexif().get(_EXIF_ORIENTATION, 1) or 1)
    except Exception:
        return 1


def _can_pass_through(image: Image.Image) -> bool:
    if image.format not in _PASSTHROUGH_FORMATS:
        return False
    if getattr(image, "n_frames", 1) != 1:
        return False
    if image.mode not in _PASSTHROUGH_MODES:
        return False
    if "transparency" in image.info:
        return False
    return _orientation(image) == 1


def _to_rgb(image: Image.Image) -> Image.Image:
    """알파가 있으면 흰 바탕에 합성한다 — Forge 가 RGBA init 이미지를 받으면 하는 일
    (img2img_background_color 기본 흰색)과 같다. 그냥 convert('RGB') 하면 투명한 곳에
    남은 임의의 RGB(대개 검정)가 드러난다."""
    has_alpha = image.mode in ("RGBA", "LA", "PA", "RGBa", "La") or (
        image.mode == "P" and "transparency" in image.info)
    if not has_alpha:
        return image.convert("RGB")
    rgba = image.convert("RGBA")
    background = Image.new("RGB", rgba.size, (255, 255, 255))
    background.paste(rgba, mask=rgba.getchannel("A"))
    return background


def _reencode_rgb_png(image: Image.Image) -> tuple[bytes, int, int]:
    upright = ImageOps.exif_transpose(image)
    rgb = _to_rgb(upright)
    out = io.BytesIO()
    rgb.save(out, format="PNG")
    return out.getvalue(), rgb.width, rgb.height


def encode_image_bytes(data: bytes, *, label: str = "입력 이미지") -> EncodedImage:
    """이미지 바이트 → 백엔드용 base64 + 백엔드가 받을 크기."""
    if not isinstance(data, (bytes, bytearray)) or not data:
        raise ImagePayloadError(f"{label}: 비어 있는 이미지입니다.")
    if len(data) > MAX_INPUT_BYTES:
        raise ImagePayloadError(f"{label}: 이미지 파일이 너무 큽니다.")
    try:
        with Image.open(io.BytesIO(bytes(data))) as image:
            width, height = image.size
            if width < 1 or height < 1 or width * height > MAX_INPUT_PIXELS:
                raise ImagePayloadError(f"{label}: 지원하지 않는 이미지 크기입니다 ({width}×{height}).")
            if _can_pass_through(image):
                return EncodedImage(base64.b64encode(bytes(data)).decode("ascii"), width, height, False)
            image.load()
            encoded, width, height = _reencode_rgb_png(image)
    except ImagePayloadError:
        raise
    except (OSError, SyntaxError, ValueError, Image.DecompressionBombError) as exc:
        raise ImagePayloadError(f"{label}: 이미지를 읽을 수 없습니다.") from exc
    return EncodedImage(base64.b64encode(encoded).decode("ascii"), width, height, True)


def encode_image_file(path: str, *, label: str = "입력 이미지") -> EncodedImage:
    """이미지 파일 → 백엔드용 base64 + 크기. 한글 경로도 PIL/open 으로 읽는다."""
    try:
        with open(path, "rb") as handle:
            data = handle.read(MAX_INPUT_BYTES + 1)
    except OSError as exc:
        raise ImagePayloadError(f"{label}: 파일을 열 수 없습니다.") from exc
    return encode_image_bytes(data, label=label)


def strip_data_url(value: object) -> str:
    """``data:image/...;base64,XXXX`` 또는 순수 base64 → base64 본문. 공백은 제거한다."""
    if not isinstance(value, str):
        return ""
    text = value.strip()
    if text.startswith("data:"):
        text = text.split(",", 1)[1] if "," in text else ""
    return "".join(text.split())


def decode_base64_image(value: object, *, label: str = "입력 이미지") -> bytes:
    """data URL/base64 → 이미지 바이트. 깨진 base64 는 ImagePayloadError."""
    body = strip_data_url(value)
    if not body:
        raise ImagePayloadError(f"{label}: 이미지 데이터가 없습니다.")
    try:
        data = base64.b64decode(body, validate=True)
    except (ValueError, binascii.Error) as exc:
        raise ImagePayloadError(f"{label}: 올바르지 않은 base64 이미지입니다.") from exc
    if not data:
        raise ImagePayloadError(f"{label}: 이미지 데이터가 없습니다.")
    return data


def image_size_from_bytes(data: bytes, *, label: str = "입력 이미지") -> tuple[int, int]:
    """헤더만 읽어 (width, height) — 픽셀은 디코드하지 않는다."""
    try:
        with Image.open(io.BytesIO(data)) as image:
            width, height = image.size
    except (OSError, SyntaxError, ValueError, Image.DecompressionBombError) as exc:
        raise ImagePayloadError(f"{label}: 이미지를 읽을 수 없습니다.") from exc
    if width < 1 or height < 1:
        raise ImagePayloadError(f"{label}: 지원하지 않는 이미지 크기입니다.")
    return width, height


def image_size_from_base64(value: object, *, label: str = "입력 이미지") -> tuple[int, int]:
    """data URL/base64 → (width, height). 헤더만 읽는다."""
    return image_size_from_bytes(decode_base64_image(value, label=label), label=label)


# 요청 크기로 받아 줄 수 있는 한 변의 최대값 (UI 입력 한도와 같은 자릿수)
MAX_TARGET_DIMENSION = 16384


def _target_dimension(value: object) -> Optional[int]:
    """요청의 width/height 값 → 양의 정수, 비었거나(null·''·NaN) 잘못됐으면 None."""
    if value is None or isinstance(value, bool):
        return None
    try:
        number = float(str(value).strip())
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number):
        return None
    number = int(number)
    return number if 1 <= number <= MAX_TARGET_DIMENSION else None


def resolve_target_size(
    width: object,
    height: object,
    image_size: Union[tuple[int, int], Callable[[], tuple[int, int]], None],
) -> tuple[Optional[int], Optional[int]]:
    """요청 크기 — 비었거나 잘못된 변은 입력 이미지 크기로 채운다.

    I2IView 는 ``parseInt(width)`` 를 보내서 칸을 비우면 JSON 에 null 이 온다. 예전 경로 전송
    (``_load_image``)은 이미지 크기를 먼저 넣고 payload 값으로 덮었으므로 빈 칸은 이미지
    크기가 됐는데, 원본 바이트 전송으로 바뀐 뒤엔 숨은 입력칸의 옛 값(1024·이전 이미지 크기)이
    나갔다. ``image_size`` 는 (w, h) 또는 필요할 때만 부르는 callable(헤더 읽기 — 못 읽으면
    ImagePayloadError). 돌려주는 None 은 '정할 수 없음'이다(호출자가 기존 값을 둔다).
    """
    w = _target_dimension(width)
    h = _target_dimension(height)
    if w is not None and h is not None:
        return w, h
    try:
        size = image_size() if callable(image_size) else image_size
    except ImagePayloadError:
        size = None
    if size:
        image_w, image_h = size
        if w is None:
            w = _target_dimension(image_w)
        if h is None:
            h = _target_dimension(image_h)
    return w, h
