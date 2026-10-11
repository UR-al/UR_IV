"""픽셀아트 필터 — 블록 평균으로 줄이고 최근접으로 키운 뒤, 원하면 색 수를 줄인다 (Qt 비의존).

``core.editor_ops.FILTERS['pixelart']`` 의 본체. 공개된 기법만으로 새로 짰다:
블록 평균(정수 배율 INTER_AREA = 상자 평균) + 최근접 확대 + PIL 중앙값 분할(median cut) 팔레트.

* 블록은 **좌상단 기준**이다. 오른쪽·아래 가장자리는 복제 패딩으로 블록을 채운 뒤 잘라 내므로
  가장자리 블록이 일부만 보여도 블록 하나는 늘 한 색이다.
* 알파는 바꾸지 않는다(``FILTERS`` 규약 — 투명도는 호출자가 다시 붙인다). 대신 블록 색은
  **알파 가중 평균**이라 투명한 곳에 숨은 RGB 가 테두리 블록으로 번지지 않는다.
* 프리뷰는 긴 변 1024 로 줄인 소스에서 돈다 — ``pixel_scale``(축소본 px / 원본 px)을 블록 크기에
  곱해야 확정 결과와 같은 굵기로 보인다. 작은 블록은 프리뷰에서 1 이 되어 팔레트만 보일 수 있다.
* 팔레트는 결정적이다(난수 없음, 디더 끔): ≤256² 로 성기게 뽑은 표본에서 median cut 팔레트를 만들고
  전체를 그 팔레트로 매핑한다. 16비트는 상위 8비트로 팔레트를 고르고 ×257 로 되돌린다.
"""
from __future__ import annotations

from typing import Optional

import numpy as np

PIXEL_SIZE_DEFAULT = 8
PIXEL_SIZE_MIN = 2
PIXEL_SIZE_MAX = 64
COLORS_DEFAULT = 16
COLORS_MAX = 64          # 2 미만이면 팔레트를 끈다(UI '끔' = 0)
PALETTE_SAMPLE_EDGE = 256


def _value_scale(dtype: np.dtype, data: np.ndarray) -> float:
    """정수형은 최댓값, 실수형은 0~1 이면 1, 아니면 255 — 8비트로 내리고 올릴 때 쓰는 배율."""
    if np.issubdtype(dtype, np.integer):
        return float(np.iinfo(dtype).max)
    peak = float(np.nanmax(data)) if data.size else 0.0
    return 1.0 if peak <= 1.0 else 255.0


def _cast_like(values: np.ndarray, dtype: np.dtype) -> np.ndarray:
    if np.issubdtype(dtype, np.integer):
        info = np.iinfo(dtype)
        return np.clip(np.rint(values), info.min, info.max).astype(dtype)
    return values.astype(dtype)


def _block_means(bgr: np.ndarray, s: int, alpha: Optional[np.ndarray]) -> np.ndarray:
    """``s``×``s`` 블록 평균(실수). 알파가 있으면 알파 가중 평균, 블록이 전부 투명하면 단순 평균."""
    import cv2
    h, w = bgr.shape[:2]
    ph, pw = -(-h // s) * s, -(-w // s) * s
    work = bgr.astype(np.float32)
    if ph != h or pw != w:
        work = cv2.copyMakeBorder(work, 0, ph - h, 0, pw - w, cv2.BORDER_REPLICATE)
    size = (pw // s, ph // s)
    plain = cv2.resize(work, size, interpolation=cv2.INTER_AREA)
    if alpha is None:
        return plain
    a = alpha.astype(np.float32) / float(_value_scale(alpha.dtype, alpha))
    if ph != h or pw != w:
        a = cv2.copyMakeBorder(a, 0, ph - h, 0, pw - w, cv2.BORDER_REPLICATE)
    num = cv2.resize(work * a[:, :, None], size, interpolation=cv2.INTER_AREA)
    den = cv2.resize(a, size, interpolation=cv2.INTER_AREA)
    covered = den > 1e-6
    out = plain.copy()
    out[covered] = num[covered] / den[covered][:, None]
    return out


def _to_rgb8(small: np.ndarray, scale: float) -> np.ndarray:
    """실수 BGR(값 범위 0~scale) → 8비트 RGB."""
    if scale == 255.0:
        bgr8 = np.clip(np.rint(small), 0, 255).astype(np.uint8)
    else:
        bgr8 = np.clip(np.rint(small * (255.0 / scale)), 0, 255).astype(np.uint8)
    return np.ascontiguousarray(bgr8[:, :, ::-1])


def _quantize_median_cut(small: np.ndarray, k: int, scale: float) -> np.ndarray:
    """``small``(실수 BGR, 0~scale)의 색을 median cut 팔레트 ``k`` 색으로 줄인다. 같은 범위로 돌려준다."""
    from PIL import Image
    rgb8 = _to_rgb8(small, scale)
    h, w = rgb8.shape[:2]
    sy = max(1, -(-h // PALETTE_SAMPLE_EDGE))
    sx = max(1, -(-w // PALETTE_SAMPLE_EDGE))
    sample = Image.fromarray(np.ascontiguousarray(rgb8[::sy, ::sx]), 'RGB')
    palette = sample.quantize(colors=int(k), method=Image.Quantize.MEDIANCUT, dither=Image.Dither.NONE)
    mapped = Image.fromarray(rgb8, 'RGB').quantize(palette=palette, dither=Image.Dither.NONE)
    out8 = np.asarray(mapped.convert('RGB'))[:, :, ::-1].astype(np.float32)
    if scale == 255.0:
        return out8
    if scale == 65535.0:
        return out8 * 257.0
    return out8 * (scale / 255.0)


def pixel_art(bgr: np.ndarray, *, pixel_size: int = PIXEL_SIZE_DEFAULT, colors: int = COLORS_DEFAULT,
              alpha: Optional[np.ndarray] = None, pixel_scale: float = 1.0) -> np.ndarray:
    """블록 ``pixel_size`` px 의 픽셀아트. 같은 모양·dtype 의 BGR 을 돌려준다.

    colors: 2 이상이면 그 색 수의 팔레트로 줄인다(0·1 = 팔레트 끔).
    alpha: 같은 크기의 알파(있으면 블록 색을 알파 가중 평균으로). 반환값에 알파는 없다.
    pixel_scale: 프리뷰 축소 배율 — 블록 크기에 곱한다.
    """
    if bgr.ndim != 3 or bgr.shape[2] != 3:
        raise ValueError('픽셀아트는 BGR 이미지만 받습니다')
    h, w = bgr.shape[:2]
    dtype = bgr.dtype
    scale = _value_scale(dtype, bgr)
    try:
        ratio = float(pixel_scale)
    except (TypeError, ValueError):
        ratio = 1.0
    if not np.isfinite(ratio) or ratio <= 0:
        ratio = 1.0
    s = max(1, int(round(int(pixel_size) * ratio)))
    if alpha is not None and alpha.shape[:2] != (h, w):
        alpha = None
    small = _block_means(bgr, s, alpha) if s > 1 else bgr.astype(np.float32)
    if int(colors) >= 2:
        small = _quantize_median_cut(small, int(colors), scale)
    if s > 1:
        small = np.repeat(np.repeat(small, s, axis=0), s, axis=1)[:h, :w]
    return _cast_like(small, dtype)
