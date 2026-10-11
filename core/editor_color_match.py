"""참조 색 맞추기 — Lab 평균·표준편차 옮기기 (Qt 비의존 순수 로직).

``VueBridge._editor_process_impl`` 의 ``color_match`` 본체. Reinhard et al. 2001 'Color Transfer
between Images' 의 공개된 식만으로 새로 짰다: 채널마다 ``(x - μ_대상) · σ_참조/σ_대상 + μ_참조``.
원 논문의 lαβ 대신 OpenCV 실수 CIE Lab 을 쓴다.

두 가지 방식
* ``reference`` — 참조 이미지(전체 해상도)의 통계로 맞춘다. 마스크가 오면 칠한 곳만 바꾼다.
* ``ring`` — 인페인트·손 보정 뒤 경계 맞추기. 칠한 영역 **안쪽 테두리 띠**를 바로 **바깥 띠**(같은 폭)의
  통계에 맞춘 뒤 그 보정을 칠한 영역 전체에 준다. 두 띠가 같은 이음매를 사이에 두고 있어 같은 재질이다 —
  칠한 영역 전체 통계를 쓰면 다시 그린 손이 배경색 쪽으로 끌려간다.

규약
* 바깥 픽셀·알파는 **바이트 그대로**다(마스크 안만 대입한다).
* 통계에서 투명한 픽셀(알파 절반 미만)은 뺀다. 남는 불투명 픽셀이 너무 적으면 알파를 무시한다.
* σ 비율은 [0.2, 5] 로 자른다 — 평평한 영역에서 잡음이 폭발하지 않게.
* 결과는 ``원본 + 강도 × (Lab→BGR(옮긴 Lab) − Lab→BGR(원래 Lab))`` 이다. 왕복 변환 오차를 서로
  지워서, 통계가 같으면 픽셀이 정확히 그대로다.
* uint8·uint16·실수 이미지를 모두 받고 dtype 을 지킨다.
"""
from __future__ import annotations

import os
import threading
from collections import OrderedDict
from typing import Any, Mapping, Optional

import numpy as np

RATIO_MIN = 0.2
RATIO_MAX = 5.0
STD_FLOOR = 1e-3
MIN_PIXELS = 16
RING_PX_DEFAULT = 16
RING_PX_MIN = 2
RING_PX_MAX = 256

MSG_NO_REFERENCE = '참조 이미지를 먼저 불러오세요'
MSG_UNREADABLE_REFERENCE = '참조 이미지를 읽을 수 없습니다'
MSG_RING_NEEDS_MASK = '마스크 주변 모드는 먼저 영역을 칠하세요'
MSG_TOO_SMALL = '칠한 영역이 너무 작습니다'
MSG_NO_OUTER = '마스크 바깥 테두리에 비교할 픽셀이 없습니다'
MSG_BAD_IMAGE = '색 맞추기를 할 수 없는 이미지 형식입니다'


class ColorMatchError(ValueError):
    """사용자에게 그대로 보여 줄 수 있는 색 맞추기 실패."""


# ── 값 범위 ────────────────────────────────────────────────────────────────

def to_unit_float(img: np.ndarray) -> tuple[np.ndarray, tuple[np.dtype, float]]:
    """``(0~1 실수 float32, codec)``. codec 은 :func:`from_unit_float` 이 원래 dtype·범위로 되돌릴 때 쓴다.

    정수형은 그 형의 최댓값으로 나눈다. 실수형은 0~1 이면 그대로, 아니면 0~255 로 본다.
    """
    dtype = img.dtype
    if np.issubdtype(dtype, np.integer):
        scale = float(np.iinfo(dtype).max)
        data = img.astype(np.float32)
    else:
        data = np.nan_to_num(img.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
        scale = 1.0 if (not data.size or float(data.max()) <= 1.0) else 255.0
    if scale != 1.0:
        data /= scale
    np.clip(data, 0.0, 1.0, out=data)
    return data, (dtype, scale)


def from_unit_float(data: np.ndarray, codec: tuple[np.dtype, float]) -> np.ndarray:
    dtype, scale = codec
    values = np.clip(data, 0.0, 1.0) * scale
    if np.issubdtype(dtype, np.integer):
        return np.rint(values).astype(dtype)
    return values.astype(dtype)


def _split(img: np.ndarray):
    if img.ndim != 3 or img.shape[2] not in (3, 4):
        raise ColorMatchError(MSG_BAD_IMAGE)
    if img.shape[2] == 4:
        return img[:, :, :3], img[:, :, 3]
    return img, None


def _opaque(alpha: Optional[np.ndarray]) -> Optional[np.ndarray]:
    if alpha is None:
        return None
    unit, _ = to_unit_float(alpha)
    return unit >= 0.5


def _region_from_mask(mask: Optional[np.ndarray], shape: tuple[int, int]) -> Optional[np.ndarray]:
    """마스크(0~255) → 불리언 영역. 크기가 다르면 최근접으로 맞춘다(보간하면 가장자리가 번진다)."""
    if mask is None:
        return None
    import cv2
    m = mask
    if m.ndim == 3:
        m = m[:, :, 0]
    if m.shape[:2] != shape:
        m = cv2.resize(m, (shape[1], shape[0]), interpolation=cv2.INTER_NEAREST)
    return m > 127


# ── 통계 ───────────────────────────────────────────────────────────────────

def _lab(bgr_unit: np.ndarray) -> np.ndarray:
    import cv2
    return cv2.cvtColor(bgr_unit, cv2.COLOR_BGR2Lab)


def lab_stats(lab: np.ndarray, region: Optional[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    """``region``(불리언, None = 전체) 안의 Lab 채널별 ``(평균[3], 표준편차[3])``.

    픽셀이 ``MIN_PIXELS`` 보다 적으면 ColorMatchError — 몇 픽셀의 통계로 영역 전체를 바꾸면 안 된다.
    """
    import cv2
    if region is None:
        if lab.shape[0] * lab.shape[1] < MIN_PIXELS:
            raise ColorMatchError(MSG_TOO_SMALL)
        mean, std = cv2.meanStdDev(lab)
    else:
        if int(np.count_nonzero(region)) < MIN_PIXELS:
            raise ColorMatchError(MSG_TOO_SMALL)
        mean, std = cv2.meanStdDev(lab, mask=region.astype(np.uint8))
    return mean.reshape(3).astype(np.float64), std.reshape(3).astype(np.float64)


def _stats_region(region: Optional[np.ndarray], opaque: Optional[np.ndarray]) -> Optional[np.ndarray]:
    """통계 영역 = 영역 ∩ 불투명. 불투명 픽셀이 너무 적으면 알파를 무시한다."""
    if opaque is None:
        return region
    both = opaque if region is None else (region & opaque)
    return both if int(np.count_nonzero(both)) >= MIN_PIXELS else region


def transfer_lab(lab: np.ndarray, mu_t, sd_t, mu_r, sd_r) -> np.ndarray:
    """Reinhard 옮기기: ``(lab - μt) · clip(σr/σt, 0.2, 5) + μr`` (채널별)."""
    mu_t = np.asarray(mu_t, np.float64).reshape(3)
    mu_r = np.asarray(mu_r, np.float64).reshape(3)
    sd_t = np.maximum(np.asarray(sd_t, np.float64).reshape(3), STD_FLOOR)
    ratio = np.clip(np.asarray(sd_r, np.float64).reshape(3) / sd_t, RATIO_MIN, RATIO_MAX)
    out = (lab - mu_t.astype(np.float32)) * ratio.astype(np.float32)
    out += mu_r.astype(np.float32)
    return out.astype(np.float32, copy=False)


def _bbox(region: np.ndarray) -> Optional[tuple[int, int, int, int]]:
    rows = np.flatnonzero(region.any(axis=1))
    if rows.size == 0:
        return None
    cols = np.flatnonzero(region.any(axis=0))
    return int(rows[0]), int(rows[-1]) + 1, int(cols[0]), int(cols[-1]) + 1


def _apply_transfer(img: np.ndarray, region: Optional[np.ndarray], target: tuple, reference: tuple,
                    strength: float) -> np.ndarray:
    """``region`` 안(없으면 전체)에 옮기기를 적용한 사본. 바깥·알파는 그대로다."""
    import cv2
    out = img.copy()
    bgr, _alpha = _split(out)
    h, w = bgr.shape[:2]
    if region is None:
        y0, y1, x0, x1 = 0, h, 0, w
    else:
        box = _bbox(region)
        if box is None:
            return out
        y0, y1, x0, x1 = box
    crop = bgr[y0:y1, x0:x1]
    unit, codec = to_unit_float(crop)
    lab = _lab(unit)
    moved = cv2.cvtColor(transfer_lab(lab, target[0], target[1], reference[0], reference[1]), cv2.COLOR_Lab2BGR)
    moved -= cv2.cvtColor(lab, cv2.COLOR_Lab2BGR)   # 왕복 오차를 지운 순수 변화량
    del lab
    moved *= float(strength)
    moved += unit
    result = from_unit_float(moved, codec)
    view = bgr[y0:y1, x0:x1]
    if region is None:
        view[...] = result
    else:
        sub = region[y0:y1, x0:x1]
        view[sub] = result[sub]
    return out


def _stats_on(bgr: np.ndarray, regions: list[Optional[np.ndarray]]) -> list[tuple[np.ndarray, np.ndarray]]:
    """영역마다 Lab 통계. Lab 변환은 영역들을 모두 덮는 경계 상자에서 한 번만 한다(4K 전체 변환을 피한다)."""
    if any(r is None for r in regions):
        unit, _ = to_unit_float(bgr)
        lab = _lab(unit)
        return [lab_stats(lab, r) for r in regions]
    union = regions[0].copy()
    for r in regions[1:]:
        union |= r
    box = _bbox(union)
    if box is None:
        raise ColorMatchError(MSG_TOO_SMALL)
    y0, y1, x0, x1 = box
    unit, _ = to_unit_float(bgr[y0:y1, x0:x1])
    lab = _lab(unit)
    return [lab_stats(lab, r[y0:y1, x0:x1]) for r in regions]


def _strength(value: Any) -> float:
    try:
        s = float(value)
    except (TypeError, ValueError):
        return 1.0
    if not np.isfinite(s):
        return 1.0
    return max(0.0, min(1.0, s))


def color_match(img: np.ndarray, ref_mean, ref_std, *, strength: float = 1.0,
                mask: Optional[np.ndarray] = None) -> np.ndarray:
    """참조 통계 ``(ref_mean, ref_std)`` 로 맞춘다. ``mask`` 가 있으면 칠한 곳(>127)만 바꾼다."""
    s = _strength(strength)
    bgr, alpha = _split(img)
    if s <= 0.0:
        return img.copy()
    region = _region_from_mask(mask, bgr.shape[:2])
    if region is not None and int(np.count_nonzero(region)) < MIN_PIXELS:
        raise ColorMatchError(MSG_TOO_SMALL)
    target = _stats_on(bgr, [_stats_region(region, _opaque(alpha))])[0]
    return _apply_transfer(img, region, target, (ref_mean, ref_std), s)


def ring_bands(mask_bool: np.ndarray, band_px: int) -> tuple[np.ndarray, np.ndarray]:
    """``(안쪽 띠, 바깥 띠)`` — 마스크 경계에서 ``band_px`` 안쪽·바깥쪽 픽셀.

    양쪽 거리 변환(DIST_L2) 한 번씩이라 띠 폭과 무관하게 비용이 같다. 마스크가 이미지 가장자리에
    닿으면 그쪽 바깥 띠는 없다(한쪽 링) — 그래도 된다.
    """
    import cv2
    band = max(1, int(band_px))
    inside = mask_bool.astype(np.uint8)
    dist_in = cv2.distanceTransform(inside, cv2.DIST_L2, 5)
    dist_out = cv2.distanceTransform((1 - inside).astype(np.uint8), cv2.DIST_L2, 5)
    inner = mask_bool & (dist_in <= band)
    outer = (~mask_bool) & (dist_out <= band)
    return inner, outer


def color_match_ring(img: np.ndarray, mask: np.ndarray, *, band_px: int = RING_PX_DEFAULT,
                     strength: float = 1.0) -> np.ndarray:
    """칠한 영역의 안쪽 띠를 바깥 띠에 맞추고, 그 보정을 칠한 영역 전체에 준다."""
    s = _strength(strength)
    bgr, alpha = _split(img)
    region = _region_from_mask(mask, bgr.shape[:2])
    if region is None or not region.any():
        raise ColorMatchError(MSG_RING_NEEDS_MASK)
    if s <= 0.0:
        return img.copy()
    if int(np.count_nonzero(region)) < MIN_PIXELS:
        raise ColorMatchError(MSG_TOO_SMALL)
    inner, outer = ring_bands(region, band_px)
    opaque = _opaque(alpha)
    outer_stats = outer if opaque is None else (outer & opaque)
    if int(np.count_nonzero(outer_stats)) < MIN_PIXELS:
        raise ColorMatchError(MSG_NO_OUTER)
    target, reference = _stats_on(bgr, [_stats_region(inner, opaque), outer_stats])
    return _apply_transfer(img, region, target, reference, s)


# ── 참조 이미지 통계 캐시 ───────────────────────────────────────────────────

def reference_stats(img: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """참조 이미지(원본 해상도)의 Lab 통계. 투명한 픽셀은 뺀다."""
    from core.editor_preview import normalize_channels
    img = normalize_channels(img)
    bgr, alpha = _split(img)
    region = _stats_region(None, _opaque(alpha))
    if region is None:
        import cv2
        unit, _ = to_unit_float(bgr)
        mean, std = cv2.meanStdDev(_lab(unit))   # 아주 작은 참조 이미지도 받는다
        return mean.reshape(3).astype(np.float64), std.reshape(3).astype(np.float64)
    return _stats_on(bgr, [region])[0]


class ReferenceStatsCache:
    """참조 이미지 통계 몇 장. 키 = ``(경로, mtime_ns, 크기)`` — 파일이 바뀌면 다시 읽는다.

    프리뷰(슬라이더 틱마다)와 확정이 **같은 통계**를 쓴다(원본 해상도에서 한 번 계산).
    워커 스레드마다 불리므로 Lock 으로 보호한다.
    """

    def __init__(self, max_entries: int = 4):
        self._max = max(1, int(max_entries))
        self._lock = threading.Lock()
        self._items: OrderedDict[tuple, tuple[np.ndarray, np.ndarray]] = OrderedDict()
        self.decode_count = 0   # 테스트·진단용: 실제 디코드 횟수

    def get(self, path: str) -> tuple[np.ndarray, np.ndarray]:
        try:
            st = os.stat(path)
        except OSError as exc:
            raise ColorMatchError(MSG_UNREADABLE_REFERENCE) from exc
        key = (os.path.normcase(os.path.abspath(path)), st.st_mtime_ns, st.st_size)
        with self._lock:
            hit = self._items.get(key)
            if hit is not None:
                self._items.move_to_end(key)
                return hit[0].copy(), hit[1].copy()
        from core.editor_preview import imread_unchanged
        try:
            img = imread_unchanged(path)
        except Exception as exc:   # 잠긴 파일·깨진 헤더 — 사용자에겐 '읽을 수 없음' 하나로
            raise ColorMatchError(MSG_UNREADABLE_REFERENCE) from exc
        if img is None or img.size == 0:
            raise ColorMatchError(MSG_UNREADABLE_REFERENCE)
        stats = reference_stats(img)
        with self._lock:
            self.decode_count += 1
            self._items[key] = stats
            self._items.move_to_end(key)
            while len(self._items) > self._max:
                self._items.popitem(last=False)
        return stats[0].copy(), stats[1].copy()

    def clear(self) -> None:
        with self._lock:
            self._items.clear()


REFERENCE_STATS = ReferenceStatsCache()


def _num(value: Any, default: float, lo: float, hi: float) -> float:
    try:
        v = float(value)
    except (TypeError, ValueError):
        return default
    if not np.isfinite(v):
        return default
    return max(lo, min(hi, v))


def run_color_match(img: np.ndarray, params: Mapping[str, Any], mask: Optional[np.ndarray], *,
                    pixel_scale: float = 1.0, cache: Optional[ReferenceStatsCache] = None) -> np.ndarray:
    """브리지 페이로드로 실행한다.

    params: mode('reference'|'ring'), strength(1~100), reference_path, ring_px(원본 px).
    프리뷰는 ``pixel_scale``(축소본 px / 원본 px)만큼 띠 폭을 줄인다(최소 2px).
    """
    mode = str(params.get('mode') or 'reference').strip().lower()
    strength = _num(params.get('strength'), 100.0, 0.0, 100.0) / 100.0
    if mode == 'ring':
        if mask is None or not np.any(mask > 127):
            raise ColorMatchError(MSG_RING_NEEDS_MASK)
        ring_px = _num(params.get('ring_px'), RING_PX_DEFAULT, RING_PX_MIN, RING_PX_MAX)
        scale = _num(pixel_scale, 1.0, 1e-4, 1e4)
        band = max(RING_PX_MIN, int(round(ring_px * scale)))
        return color_match_ring(img, mask, band_px=band, strength=strength)
    if mode == 'reference':
        from core.path_safety import safe_input_path
        path = safe_input_path(str(params.get('reference_path') or ''))
        if not path:
            raise ColorMatchError(MSG_NO_REFERENCE)
        ref_mean, ref_std = (cache or REFERENCE_STATS).get(path)
        return color_match(img, ref_mean, ref_std, strength=strength, mask=mask)
    raise ColorMatchError(f'알 수 없는 색 맞추기 방식입니다: {mode}')
