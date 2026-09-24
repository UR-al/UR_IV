# core/result_image.py
"""생성 결과 바이트의 해상도 — 헤더만 읽는다(Qt 비의존).

T2I·I2I·인페인트 완료 경로가 Vue 뷰어의 해상도 표시(send_image 의 width/height)에 쓴다.
예전 T2I 완료 처리는 이 두 숫자를 얻으려고 GUI 스레드에서 QPixmap 으로 PNG 전체를 디코드했고
(2048×1144 기준 30~55ms), I2I·인페인트는 숨은 탭의 result_label 에 같은 디코드·축소를 했다.
PNG/JPEG/WebP 헤더만 읽으면 0.1ms 안쪽이다.

생성 정보(gen_info)의 width/height 는 **요청값**이라 hires fix·업스케일을 거친 결과와 다르다 —
헤더를 못 읽을 때의 폴백으로만 쓴다.
"""
from __future__ import annotations

from typing import Mapping, Optional


def _positive_int(value: object) -> int:
    if isinstance(value, bool):
        return 0
    try:
        number = int(float(value))  # type: ignore[arg-type]
    except (TypeError, ValueError, OverflowError):
        return 0
    return number if number > 0 else 0


def result_image_size(data: object, info: Optional[Mapping] = None) -> tuple[int, int]:
    """결과 이미지 바이트 → (width, height). 헤더만 읽고 픽셀은 디코드하지 않는다.

    헤더를 읽지 못하면 ``info`` 의 width/height(요청값)로, 그것도 없으면 (0, 0).
    """
    if isinstance(data, (bytes, bytearray, memoryview)) and len(data):
        from core.image_payload import ImagePayloadError, image_size_from_bytes
        try:
            return image_size_from_bytes(bytes(data), label="생성 결과")
        except ImagePayloadError:
            pass
    source = info if isinstance(info, Mapping) else {}
    width = _positive_int(source.get("width"))
    height = _positive_int(source.get("height"))
    if not (width and height):
        return 0, 0
    return width, height
