"""자석 올가미용 Canny 엣지맵 (Qt 비의존).

``VueBridge.getEdgeMap`` 이 쓰는 본체. EditorView·InpaintView 의 자석 올가미가 이 PNG 를
받아 흑백 배열로 풀고, 커서 근처 엣지 픽셀에 점을 붙인다.

예전에는 ``cv2.imread`` 로 읽어서 한글·일본어 경로의 이미지면 None → 빈 문자열이 돌아가,
자석 모드가 아무 표시 없이 '자유' 올가미처럼 굴었다. 읽기는 ``core.cv_io`` 로 한다.
"""
from __future__ import annotations

import base64

# 프론트(EditorView/InpaintView)가 보내는 기본 임계값
DEFAULT_CANNY_LOW = 50
DEFAULT_CANNY_HIGH = 150


def compute_edge_map(path: str, canny_low: int = DEFAULT_CANNY_LOW, canny_high: int = DEFAULT_CANNY_HIGH):
    """``path`` 이미지의 Canny 엣지(uint8, 0/255, 원본과 같은 크기). 읽기 실패면 None."""
    import cv2
    from core.cv_io import imread_unicode
    img = imread_unicode(path, cv2.IMREAD_COLOR)
    if img is None:
        return None
    low = max(0, int(canny_low))
    high = max(low, int(canny_high))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    return cv2.Canny(blurred, low, high)


def edge_map_data_url(path: str, canny_low: int = DEFAULT_CANNY_LOW, canny_high: int = DEFAULT_CANNY_HIGH) -> str:
    """엣지맵을 흑백 PNG data URL 로. 읽기·인코딩 실패면 빈 문자열(슬롯 계약)."""
    import cv2
    edges = compute_edge_map(path, canny_low, canny_high)
    if edges is None:
        return ''
    ok, buf = cv2.imencode('.png', edges)
    if not ok:
        return ''
    return 'data:image/png;base64,' + base64.b64encode(buf.tobytes()).decode('ascii')
