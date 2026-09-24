# core/curves.py
"""톤 커브 — 제어점을 256칸 LUT 로 바꾸고 이미지에 적용한다. Qt 의존 없음.

PyQt 판(`tabs/editor/curves_widget.py`, 은퇴해 삭제됨)은 계산이 위젯 안에 들어 있어 Qt 없이는
테스트할 수 없었다. Vue 로 옮기면서 계산만 떼어낸다.

곡선 정의는 **정렬된 제어점 사이의 선형보간** 하나뿐이다. 프론트는 곡선을 그리려고
같은 계산을 JS 로 한 번 더 하는데(`frontend/src/utils/curves.ts`), 정의가 이만큼
단순해야 그린 것과 적용된 것이 어긋나지 않는다.

제어점은 0~1 정규화 좌표다 — x 는 입력 레벨, y 는 출력 레벨.
"""
import numpy as np

CHANNELS = ('rgb', 'r', 'g', 'b')
IDENTITY_POINTS = [(0.0, 0.0), (1.0, 1.0)]

_LUT_SIZE = 256


def normalize_points(points) -> list:
    """제어점을 (x, y) 튜플 리스트로 다듬는다 — 0~1 로 자르고 x 순 정렬.

    프론트에서 오는 값이므로 형태를 믿지 않는다. 쓸 수 없으면 항등 커브로 돌린다.
    """
    if not points:
        return list(IDENTITY_POINTS)
    cleaned = []
    for point in points:
        try:
            x, y = float(point[0]), float(point[1])
        except (TypeError, ValueError, IndexError):
            continue
        if x != x or y != y:      # NaN
            continue
        cleaned.append((min(1.0, max(0.0, x)), min(1.0, max(0.0, y))))
    if len(cleaned) < 2:
        return list(IDENTITY_POINTS)
    cleaned.sort(key=lambda p: p[0])
    return cleaned


def build_lut(points) -> np.ndarray:
    """제어점 → 256칸 uint8 LUT."""
    pts = normalize_points(points)
    xs = np.array([p[0] for p in pts], dtype=np.float64)
    ys = np.array([p[1] for p in pts], dtype=np.float64)
    ramp = np.linspace(0.0, 1.0, _LUT_SIZE)
    out = np.interp(ramp, xs, ys)
    # 반올림이어야 한다. PyQt 판은 그냥 잘라서(`astype`) 항등 커브조차 LUT 가
    # 대각선이 아니었다 — i/255*255 가 부동소수점상 i 바로 아래로 떨어지는 칸이 생겨
    # 아무것도 안 건드렸는데 밝기가 1 레벨씩 내려갔다.
    return np.clip(np.rint(out * 255.0), 0, 255).astype(np.uint8)


def channel_luts(curves: dict) -> tuple:
    """(lut_r, lut_g, lut_b) — 채널 커브를 먼저 태우고 그 위에 RGB 마스터를 얹는다.

    순서가 뒤바뀌면 결과가 달라진다. PyQt 판과 같은 순서를 유지한다.
    """
    curves = curves or {}
    master = build_lut(curves.get('rgb'))
    return tuple(master[build_lut(curves.get(ch))] for ch in ('r', 'g', 'b'))


def is_identity(curves) -> bool:
    """아무것도 바꾸지 않는 커브인지. 프리뷰를 건너뛸지 판단하는 데 쓴다.

    제어점을 끝점 두 개까지 되돌린 경우뿐 아니라, 점을 여럿 찍었지만 결과가
    대각선인 경우도 잡아야 해서 LUT 로 비교한다.
    """
    if not curves:
        return True
    ramp = np.arange(_LUT_SIZE, dtype=np.uint8)
    for channel in CHANNELS:
        if not np.array_equal(build_lut(curves.get(channel)), ramp):
            return False
    return True


def apply_curves(bgr: np.ndarray, curves: dict) -> np.ndarray:
    """BGR 3채널 이미지에 커브를 적용한다. 알파는 호출부에서 보존한다."""
    lut_r, lut_g, lut_b = channel_luts(curves)
    # cv2.LUT 는 채널 수가 맞으면 한 번에 처리한다 — split/merge 보다 빠르다.
    lut = np.dstack([lut_b, lut_g, lut_r])
    import cv2
    return cv2.LUT(bgr, lut)
