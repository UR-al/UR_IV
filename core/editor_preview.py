"""에디터 이미지 입출력과 실시간 프리뷰 소스 캐시 (Qt 비의존).

프리뷰는 슬라이더를 멈출 때마다(120ms 디바운스) 한 번씩 나간다. 예전에는 매번
(1) 원본 전체를 새로 디코드하고(15~115ms) (2) 그다음에야 긴 변 1024px 로 줄였고
(3) 결과를 PNG base64(1.3~2.8MB)로 브리지에 실었다. 여기서는

* 축소된 프리뷰 소스를 ``(경로, mtime_ns, 크기)`` 키로 **한 장만** 캐시한다 —
  같은 이미지에서 슬라이더를 움직이는 동안 디코드·축소는 처음 한 번뿐이다.
  연산이 배열을 제자리에서 고치므로(ROI 대입, 복원 마스크 대입) 호출자에게는
  반드시 **사본**을 준다. 확정 작업은 이 캐시를 쓰지 않는다.
* 3채널 결과는 JPEG(q90, 1~3ms · 0.06~0.28MB)로, 알파가 있는 결과만 PNG 로 보낸다.
  PNG 압축 레벨을 낮추는 건 실측상 오히려 두 배 느려서 기본값을 둔다.

경로는 한글이 섞일 수 있어 ``cv2.imread``/``imwrite`` 대신
``np.fromfile + cv2.imdecode`` / ``cv2.imencode + tofile`` 을 쓴다.
"""
from __future__ import annotations

import base64
import os
import threading
from typing import Optional

import numpy as np

PREVIEW_MAX_EDGE = 1024
PREVIEW_JPEG_QUALITY = 90


def imread_unchanged(path: str) -> Optional[np.ndarray]:
    """에디터의 디코드 지점 하나 — 화면(브라우저)과 같은 방향·투명도의 배열. 실패하면 None.

    프리뷰 캐시·확정 편집·복원 원본이 모두 이 함수로 읽는다(한글 경로 안전, ``core.cv_io``).

    * ``IMREAD_UNCHANGED`` — 알파·16비트를 그대로 둔다.
    * **EXIF 방향을 적용한다.** IMREAD_UNCHANGED 는 방향 태그를 무시해, 폰·카메라 사진
      (Orientation 2~8)을 열면 브라우저 캔버스(세워 보임)와 배열이 어긋났다 — 세운 좌표로 그린
      선택·마스크가 눕힌 배열에 늘어나 적용되고(검은띠가 엉뚱한 자리에), 결과가 옆으로 누운 채
      저장됐다. ``editor_save.load_image_array``·``edge_map`` 과 같은 방향이다. 편집 임시본(PNG)은
      EXIF 가 없어 두 번 돌지 않는다.
      **TIFF 는 빼고** — OpenCV TIFF 디코더(libtiff)가 IMREAD_UNCHANGED 에서도 방향을 이미 적용해 준다.
      또 돌리면 3·2·4… 는 저장된 그대로, 6·8 은 180° 뒤집힌 채 누웠다(Codex S5 #1, ``cv_io.is_tiff_bytes``).
    * **흑백 PNG 의 tRNS 투명색** — cv2 는 흑백+tRNS 를 1채널로 읽어 투명한 곳이 불투명해진다.
      키 색을 알파 0 으로 바꾼 BGRA 로 돌려준다(RGB·팔레트+tRNS 는 cv2 가 이미 BGRA 로 준다).
    """
    import cv2
    from core.cv_io import apply_exif_orientation, decode_bytes, exif_orientation, is_tiff_bytes, read_file_bytes
    data = read_file_bytes(path)
    img = decode_bytes(data, cv2.IMREAD_UNCHANGED)
    if img is None:
        return None
    img = gray_key_to_alpha(img, data)
    return apply_exif_orientation(img, 1 if is_tiff_bytes(data) else exif_orientation(data))


def gray_key_to_alpha(img: np.ndarray, data) -> np.ndarray:
    """흑백 PNG(색 유형 0)에 tRNS 키가 있으면 키 색 = 알파 0 인 BGRA. 아니면 그대로.

    cv2 는 1·2·4비트 흑백을 8비트로 늘려(×255/(2^깊이−1)) 주므로 키도 같은 배율로 맞춘다.
    """
    import struct
    import cv2
    from core.png_chunks import find_chunk
    if img is None or img.ndim != 2 or data is None:
        return img
    ihdr = find_chunk(data, b"IHDR", before_idat=True)
    if not ihdr or len(ihdr) < 13 or ihdr[9] != 0:
        return img
    trns = find_chunk(data, b"tRNS", before_idat=True)
    if not trns or len(trns) < 2:
        return img
    depth = ihdr[8]
    key = struct.unpack(">H", trns[:2])[0]
    if depth < 8:
        key = (key & ((1 << depth) - 1)) * (255 // ((1 << depth) - 1))
    elif depth == 8:
        key &= 0xFF
    opaque = np.iinfo(img.dtype).max if np.issubdtype(img.dtype, np.integer) else 1
    alpha = np.where(img == key, 0, opaque).astype(img.dtype)
    bgr = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    return np.ascontiguousarray(np.dstack([bgr, alpha]))


def imwrite(path: str, img: np.ndarray) -> bool:
    """한글 경로 안전 ``cv2.imwrite``. 확장자로 포맷을 정한다(``core.cv_io`` 별칭)."""
    from core.cv_io import imwrite_unicode
    return imwrite_unicode(path, img)


def normalize_channels(img: np.ndarray) -> np.ndarray:
    """흑백 → BGR. BGRA 는 그대로 두고 각 연산이 알파를 보존한다."""
    import cv2
    if img.ndim == 2:
        return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    if img.ndim == 3 and img.shape[2] == 2:
        return cv2.cvtColor(img[:, :, 0], cv2.COLOR_GRAY2BGR)
    return img


def downscale_for_preview(img: np.ndarray, max_edge: int = PREVIEW_MAX_EDGE) -> np.ndarray:
    """긴 변이 ``max_edge`` 를 넘으면 INTER_AREA 로 줄인다(프론트 마스크 축소와 같은 식)."""
    import cv2
    h, w = img.shape[:2]
    long_edge = max(h, w)
    if long_edge <= max_edge:
        return img
    ratio = max_edge / float(long_edge)
    return cv2.resize(img, (max(1, int(w * ratio)), max(1, int(h * ratio))),
                      interpolation=cv2.INTER_AREA)


class PreviewSourceCache:
    """축소된 프리뷰 소스 한 장. 스레드마다 호출되므로 Lock 으로 보호한다."""

    def __init__(self, max_edge: int = PREVIEW_MAX_EDGE):
        self._max_edge = max_edge
        self._lock = threading.Lock()
        self._key: Optional[tuple] = None
        self._img: Optional[np.ndarray] = None
        self._scale = 1.0       # 축소본 픽셀 / 원본 픽셀
        self.decode_count = 0   # 테스트·진단용: 실제 디코드 횟수

    def get(self, path: str) -> Optional[np.ndarray]:
        """``path`` 의 축소본 **사본**. 파일이 바뀌면(mtime/크기) 다시 디코드한다."""
        return self.get_scaled(path)[0]

    def get_scaled(self, path: str) -> tuple[Optional[np.ndarray], float]:
        """``(축소본 사본, 축소 배율)``. 배율은 축소본 픽셀 / 원본 픽셀(축소 안 했으면 1.0).

        워터마크 글자 크기처럼 '원본 픽셀 단위'로 받은 값을 프리뷰에서 같은 비율로
        보이게 하려면 이 배율을 곱해야 한다(안 곱하면 4K 원본의 프리뷰에서 글자가 4배로 보인다).
        """
        try:
            st = os.stat(path)
        except OSError:
            return None, 1.0
        key = (os.path.normcase(os.path.abspath(path)), st.st_mtime_ns, st.st_size)
        with self._lock:
            if self._key == key and self._img is not None:
                return self._img.copy(), self._scale
        img = imread_unchanged(path)
        if img is None:
            return None, 1.0
        src_w = img.shape[1]
        img = downscale_for_preview(normalize_channels(img), self._max_edge)
        scale = (img.shape[1] / float(src_w)) if src_w else 1.0
        with self._lock:
            self.decode_count += 1
            self._key = key
            self._img = img
            self._scale = scale
        return img.copy(), scale

    def clear(self) -> None:
        with self._lock:
            self._key = None
            self._img = None
            self._scale = 1.0


def to_display_uint8(img: np.ndarray) -> np.ndarray:
    """프리뷰 인코딩용 8비트 변환 — 브라우저가 16비트/실수 이미지를 보여 주는 것과 같게.

    ``imread_unchanged`` 는 16비트 PNG 를 uint16 그대로 둔다. 그대로 JPEG 로 넣으면
    OpenCV 가 CV_8U 로 '잘라서'(saturate) 떨어뜨려 프리뷰가 온통 흰색이 됐다.
    uint16 은 257 로 나눠 비율대로 줄이고, 실수는 0~1 범위면 255 배, 아니면 0~255 로 자른다.
    """
    if img.dtype == np.uint8:
        return img
    if img.dtype == np.uint16:
        return ((img.astype(np.uint32) + 128) // 257).astype(np.uint8)
    if np.issubdtype(img.dtype, np.floating):
        data = np.nan_to_num(img.astype(np.float32), nan=0.0, posinf=255.0, neginf=0.0)
        peak = float(data.max()) if data.size else 0.0
        if peak <= 1.0:
            data = data * 255.0
        return np.clip(np.rint(data), 0, 255).astype(np.uint8)
    if np.issubdtype(img.dtype, np.integer):
        info = np.iinfo(img.dtype)
        if info.max > 255 and int(img.max(initial=0)) > 255:
            # 8비트보다 넓은 정수(부호 있는 16비트 등) — 0 아래는 자르고 최댓값 기준으로 줄인다
            scale = 255.0 / float(info.max)
            return np.clip(np.rint(img.astype(np.float64) * scale), 0, 255).astype(np.uint8)
        return np.clip(img, 0, 255).astype(np.uint8)
    return img.astype(np.uint8)


def encode_preview(img: np.ndarray) -> Optional[str]:
    """프리뷰 결과 → data URL. 알파가 없으면 JPEG, 있으면 PNG. 8비트가 아니면 먼저 줄인다."""
    import cv2
    img = to_display_uint8(img)
    has_alpha = img.ndim == 3 and img.shape[2] == 4
    if has_alpha:
        ok, buf = cv2.imencode('.png', img)
        mime = 'image/png'
    else:
        ok, buf = cv2.imencode('.jpg', img, [int(cv2.IMWRITE_JPEG_QUALITY), PREVIEW_JPEG_QUALITY])
        mime = 'image/jpeg'
    if not ok:
        return None
    return f'data:{mime};base64,' + base64.b64encode(buf.tobytes()).decode('ascii')
