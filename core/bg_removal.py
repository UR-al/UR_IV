# core/bg_removal.py
"""에디터 '배경 제거'(rembg) — 세션 재사용과 품질 프리셋. Qt 를 쓰지 않는다.

왜 따로 두나
- 예전 Vue 경로(ui/vue_bridge.py 의 remove_bg)는 ``rembg.remove(pil, **kw)`` 를 session 없이
  불렀다. 그러면 rembg(bg.py ``if session is None: session = new_session("u2net")``)가 호출마다
  ONNX 세션(u2net 약 170MB)을 새로 만든다 — 클릭마다 모델 로드를 처음부터 다시 했다.
  세션을 한 번 만들어 넘기던 곳은 숨은 PyQt 에디터의 RemovalWorker 뿐이었고, 그 에디터는
  은퇴했다(tests/test_legacy_editor_retirement.py).
- 세션은 ``core.model_cache.REMBG_CACHE``(유휴 캐시)가 쥔다. 연속 작업은 캐시 히트이고, 손을 떼면
  유휴 시간 뒤 반납되며, VRAM 게이지 수동 언로드(``model_cache.clear_all``)에도 반납된다.
  추론하는 동안은 lease 로 빌려서, 그사이 반납 요청이 와도 작업이 끝난 뒤에 내린다.

rembg·onnxruntime 은 이 모듈을 import 할 때가 아니라 첫 배경 제거 때 불러온다(pymatting 의 numba
JIT 때문에 콜드 import 가 수십 초 걸린다). 테스트는 ``remove_fn``·``session_loader``·``cache`` 를
주입해 모델 없이 돈다.
"""
from __future__ import annotations

import logging
from typing import Callable, Optional

import numpy as np

logger = logging.getLogger(__name__)

# Vue 에디터는 모델을 고르지 않는다 — rembg 기본값과 같은 u2net.
DEFAULT_MODEL = 'u2net'

QUALITY_FAST = 'fast'
QUALITY_BALANCED = 'balanced'
QUALITY_BEST = 'quality'
QUALITIES = (QUALITY_FAST, QUALITY_BALANCED, QUALITY_BEST)
DEFAULT_QUALITY = QUALITY_BALANCED

# 알파 매팅 프리셋('fast' 는 매팅 없음). rembg(bg.py alpha_matting_cutout)는 uint8 마스크(0~255)로
# trimap 을 만든다: ``mask > foreground`` 가 확실한 전경, ``mask < background`` 가 확실한 배경이고,
# 둘 다 ``erode_size`` 정사각형으로 침식한 뒤 나머지(경계 띠)를 pymatting 이 푼다.
# - 'quality' 는 'balanced' 보다 엄격한 전경·배경 문턱과 큰 침식으로 경계 띠를 넓혀 더 많이 매팅하고,
#   결과 알파를 refine_alpha 로 한 번 더 다듬는다.
# - 전경 문턱은 254 이하여야 한다. 옛 vue_bridge 인라인 값 270 은 uint8 최댓값을 넘어 전경이 늘 비었고,
#   pymatting 이 ValueError('Trimap did not contain foreground values')를 내면 rembg 가 알파 매팅 없는
#   naive cutout 으로 조용히 떨어졌다 — '품질'이 실제로는 '균형'보다 거친 결과였다(2026-09-24 실측).
#   tests/test_bg_removal.py 가 rembg 의 trimap 규칙으로 모든 프리셋에 전경·배경이 남는지 지킨다.
_MATTING = {
    QUALITY_BALANCED: {
        'alpha_matting': True,
        'alpha_matting_foreground_threshold': 240,
        'alpha_matting_background_threshold': 10,
        'alpha_matting_erode_size': 10,
    },
    QUALITY_BEST: {
        'alpha_matting': True,
        'alpha_matting_foreground_threshold': 250,
        'alpha_matting_background_threshold': 20,
        'alpha_matting_erode_size': 15,
    },
}


def normalize_quality(quality) -> str:
    """프론트 값 → 프리셋 이름. 비었으면 기본('balanced'), 모르는 값은 매팅 없는 'fast'.

    옛 코드(``params.get('quality', 'balanced')`` 후 ``in ('balanced', 'quality')`` 검사)와 같은 규칙이다.
    """
    if quality is None or (isinstance(quality, str) and not quality.strip()):
        return DEFAULT_QUALITY
    q = str(quality).strip().lower()
    return q if q in QUALITIES else QUALITY_FAST


def matting_kwargs(quality) -> dict:
    """``rembg.remove`` 에 넘길 알파 매팅 인자(사본). 'fast' 는 빈 dict."""
    return dict(_MATTING.get(normalize_quality(quality), {}))


def load_session(model_name: str):
    """rembg ONNX 세션을 새로 만든다. 모델 파일이 없으면 rembg 가 처음 한 번 내려받는다."""
    from rembg import new_session
    return new_session(model_name)


def to_rgb(img: np.ndarray) -> np.ndarray:
    """에디터 이미지(BGR·BGRA·흑백, 8/16비트·실수) → rembg 입력용 RGB uint8.

    BGRA 의 알파는 버린다 — rembg 가 새 마스크로 알파를 다시 만든다(옛 코드와 같다).
    16비트 PNG 는 PIL 이 3채널 uint16 을 받지 못하므로 8비트로 줄인다.
    """
    import cv2
    from core.editor_preview import to_display_uint8

    data = to_display_uint8(np.asarray(img))
    if data.ndim == 2:
        return cv2.cvtColor(data, cv2.COLOR_GRAY2RGB)
    channels = data.shape[2]
    if channels == 1:
        return cv2.cvtColor(data[:, :, 0], cv2.COLOR_GRAY2RGB)
    if channels == 2:  # 흑백 + 알파
        return cv2.cvtColor(data[:, :, 0], cv2.COLOR_GRAY2RGB)
    if channels == 4:
        return cv2.cvtColor(data, cv2.COLOR_BGRA2RGB)
    return cv2.cvtColor(data, cv2.COLOR_BGR2RGB)


def _default_refine(bgra: np.ndarray) -> np.ndarray:
    from core.edge_refiner import refine_alpha
    return refine_alpha(bgra)


def remove_background(img: np.ndarray, quality=DEFAULT_QUALITY, *,
                      model_name: str = DEFAULT_MODEL,
                      cache=None,
                      session_loader: Optional[Callable[[str], object]] = None,
                      remove_fn: Optional[Callable[..., object]] = None,
                      refine_fn: Optional[Callable[[np.ndarray], np.ndarray]] = None) -> np.ndarray:
    """배경을 지운 BGRA(uint8) 이미지를 돌려준다.

    세션은 ``cache``(기본 ``core.model_cache.REMBG_CACHE``)에서 ``model_name`` 키로 빌린다 —
    호출마다 new_session 을 만들지 않는다. 'quality' 는 결과 알파를 ``refine_alpha`` 로 한 번 더
    다듬고, 그 단계가 실패하면 다듬지 않은 결과를 쓴다(옛 동작). 그 밖의 실패는 호출자에게 올린다.
    """
    import cv2
    from PIL import Image

    preset = normalize_quality(quality)
    if remove_fn is None:
        from rembg import remove as remove_fn
    if cache is None:
        from core.model_cache import REMBG_CACHE as cache
    loader = session_loader or load_session

    pil_img = Image.fromarray(to_rgb(img))
    with cache.lease(model_name, loader) as session:
        result = remove_fn(pil_img, session=session, **matting_kwargs(preset))
        session = None  # lease 가 끝나기 전에 참조를 놓는다(model_cache 계약)
    if hasattr(result, 'convert'):
        result = result.convert('RGBA')
    rgba = np.asarray(result)
    if rgba.ndim != 3 or rgba.shape[2] != 4:
        raise ValueError(f'rembg 결과가 RGBA 가 아닙니다: shape={getattr(rgba, "shape", None)}')
    bgra = cv2.cvtColor(rgba, cv2.COLOR_RGBA2BGRA)

    if preset == QUALITY_BEST:
        refine = refine_fn or _default_refine
        try:
            bgra = refine(bgra)
        except Exception as e:  # opencv-contrib 부재 등 — 다듬기만 건너뛴다
            logger.warning("[Editor] Edge refine skipped: %s", e)
    return bgra
