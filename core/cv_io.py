"""경로 기반 OpenCV 이미지 입출력 — 한글·일본어 경로 안전판 (Qt 비의존).

Windows(ACP 949)의 ``cv2.imread``/``cv2.imwrite`` 는 경로를 시스템 코드페이지로 넘겨서
비ASCII 경로면 조용히 실패한다(imread → None, imwrite → False). 그래서 한글 폴더의
이미지를 에디터에서 열면 첫 편집부터 '이미지를 읽을 수 없습니다'가 떴고, 자석 올가미는
아무 표시 없이 동작하지 않았다.

여기서는 파일 입출력을 numpy(파이썬 유니코드 경로)로 하고 OpenCV 에는 메모리 버퍼만 준다.
  * ``imread_unicode``  = ``np.fromfile`` + ``cv2.imdecode``
  * ``imwrite_unicode`` = ``cv2.imencode`` + ``ndarray.tofile``
실패 계약은 원래 함수와 같다 — 읽기는 None, 쓰기는 False. 호출부의 기존 오류 분기가
그대로 살아 있도록 예외를 밖으로 내지 않는다.

``core/__init__.py`` 가 패키지 import 때 다른 모듈을 끌어오므로 cv2/numpy 는 함수 안에서
늦게 import 한다.
"""
from __future__ import annotations

import os
from contextlib import contextmanager
from typing import Any, Iterator, Optional, Sequence

# imdecode 가 IMREAD_* 플래그 없이 불리면 IMREAD_COLOR(1) 이다 — cv2.imread 기본값과 같게.
_IMREAD_COLOR = 1


def imread_unicode(path: Any, flags: Optional[int] = None):
    """``cv2.imread(path, flags)`` 의 유니코드 경로 안전판. 실패하면 None.

    ``flags`` 를 생략하면 ``cv2.IMREAD_COLOR`` (``cv2.imread`` 기본값). 알파가 필요한
    곳은 ``cv2.IMREAD_UNCHANGED`` 를 넘긴다. 없는 파일·빈 파일·디코드 불가 모두 None.
    """
    return decode_bytes(read_file_bytes(path), flags)


def read_file_bytes(path: Any):
    """파일 전체를 uint8 배열로(유니코드 경로 안전). 없거나 비었거나 못 읽으면 None."""
    import numpy as np
    if path is None:
        return None
    try:
        data = np.fromfile(os.fspath(path), dtype=np.uint8)
    except (OSError, ValueError, TypeError):
        return None
    return data if data.size else None


def decode_bytes(data, flags: Optional[int] = None):
    """``cv2.imdecode`` — ``data`` 가 None 이거나 디코드할 수 없으면 None."""
    import cv2
    if data is None:
        return None
    try:
        return cv2.imdecode(data, _IMREAD_COLOR if flags is None else int(flags))
    except cv2.error:
        return None


# ─────────────────────────────────────────────
# EXIF 방향 (opt-in) — imread_unicode 의 기본 계약은 그대로 둔다
# ─────────────────────────────────────────────
# cv2 IMREAD_UNCHANGED 는 EXIF 방향을 무시한다(IMREAD_COLOR 는 적용). 브라우저·PIL
# exif_transpose 와 같은 방향의 픽셀이 필요한 곳(에디터)은 아래 함수를 쓴다.
# 단 **TIFF 는 예외**다 — OpenCV(libtiff)와 Pillow(TiffImageFile.load)의 TIFF 디코더는 방향 태그를
# 디코드하면서 이미 적용한다(IMREAD_UNCHANGED 여도). 거기에 또 돌리면 두 번 돈다(Codex S5 #1) —
# ``is_tiff_bytes`` 로 가려 방향 1 로 친다.

_EXIF_ORIENTATION_TAG = 0x0112
# 리틀·빅 엔디언 TIFF 와 BigTIFF 머리
_TIFF_MAGICS = (b"II*\x00", b"MM\x00*", b"II+\x00", b"MM\x00+")


def is_tiff_bytes(data: Any) -> bool:
    """이미지 바이트(bytes·uint8 배열)가 TIFF 인지 — 디코더가 EXIF 방향을 이미 적용하는 포맷."""
    try:
        return bytes(data[:4]) in _TIFF_MAGICS
    except Exception:
        return False


def exif_orientation(data: Any) -> int:
    """이미지 바이트(bytes·uint8 배열)의 EXIF 방향 1~8. 없거나 못 읽으면 1. 픽셀은 디코드하지 않는다.

    판정 규칙은 ``image_exif_orientation`` — 브라우저 캔버스·OpenCV(IMREAD_COLOR)처럼 진짜 EXIF 만 본다.
    """
    import io
    try:
        from PIL import Image
        stream = io.BytesIO(data)
        with Image.open(stream) as image:
            return image_exif_orientation(image, stream)
    except Exception:
        return 1


def image_exif_orientation(image: Any, stream: Any = None) -> int:
    """열린 PIL 이미지의 EXIF 방향 1~8. 없거나 못 읽으면 1. 픽셀은 디코드하지 않는다.

    **진짜 EXIF 만** 본다 — JPEG APP1·WebP EXIF(``info['exif']``), PNG eXIf 청크, TIFF IFD0.
    Pillow ``getexif()``(와 그걸 쓰는 ``ImageOps.exif_transpose``)는 EXIF 에 방향이 없으면 XMP 의
    ``tiff:Orientation`` 까지 읽는데 Chromium·OpenCV 는 XMP 방향을 무시한다. 그걸 따르면 XMP 에만
    방향이 적힌 사진의 배열이 캔버스·edge_map(IMREAD_COLOR)과 어긋나 선택·마스크가 돌아간 배열에
    늘어나 적용됐다. 또 PNG 의 ``getexif()`` 는 eXIf 를 찾으려고 픽셀을 디코드한다 — 헤더에 eXIf 가
    없으면 청크 머리만 훑어 IDAT 뒤 eXIf 를 찾는다(core.png_chunks, ``stream`` 이 없으면 ``image.fp``).
    """
    try:
        from PIL import Image
        raw = image.info.get("exif")
        if not raw and image.format == "PNG":
            source = stream if stream is not None else getattr(image, "fp", None)
            if source is not None:
                from core.png_chunks import find_chunk
                raw = find_chunk(source, b"eXIf")   # 픽셀은 읽지 않는다 — 같은 스트림을 훑는다
        if raw:
            exif = Image.Exif()
            exif.load(bytes(raw))
            value = exif.get(_EXIF_ORIENTATION_TAG, 1)
        elif image.format != "PNG" and hasattr(image, "tag_v2"):
            value = image.tag_v2.get(_EXIF_ORIENTATION_TAG, 1)   # TIFF — 방향이 IFD0 태그다
        else:
            return 1
        value = int(value or 1)
    except Exception:
        return 1
    return value if 1 <= value <= 8 else 1


def apply_exif_orientation(img, orientation: int):
    """EXIF 방향대로 세운 배열(PIL ``exif_transpose`` 와 같은 결과). 1·모르는 값은 그대로."""
    import cv2
    if img is None:
        return None
    if orientation == 2:
        return cv2.flip(img, 1)
    if orientation == 3:
        return cv2.rotate(img, cv2.ROTATE_180)
    if orientation == 4:
        return cv2.flip(img, 0)
    if orientation == 5:
        return cv2.transpose(img)
    if orientation == 6:
        return cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    if orientation == 7:
        return cv2.rotate(cv2.transpose(img), cv2.ROTATE_180)
    if orientation == 8:
        return cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
    return img


# ─────────────────────────────────────────────
# Pillow 로 픽셀을 읽을 때의 TIFF 방향 함정 (Codex S5 #1-a·#1-b)
# ─────────────────────────────────────────────
# Pillow TiffImageFile 은 load() 끝(load_end)에서 ``ImageOps.exif_transpose`` 로 방향을 적용한다.
# 에디터 디코드(``core.editor_preview.imread_unchanged`` — OpenCV/libtiff)도 IFD0 방향을 적용하므로
# 둘이 같아야 하는데, 두 군데서 어긋났다:
#   a) 방향 5~8 이면 Pillow 가 열 때 ``size`` 를 세운 크기로 바꿔 두는데, **경로로 연** 무압축 한
#      스트립 TIFF(L·P·RGBA·RGBX·CMYK·I;16 — Pillow 의 mmap 모드)는 ``ImageFile.load`` 의 mmap 지름길이
#      원래 크기가 아니라 그 바뀐 크기로 버퍼를 매핑해 픽셀이 뒤섞인 뒤 돌아갔다(20x40 L → (20,40) 쓰레기).
#      Pillow 는 경로로 열었을 때만 mmap 을 쓴다 — 파일 객체로 열면 일반 디코더를 탄다.
#   b) exif_transpose 의 ``getexif()`` 는 IFD0 에 방향이 없으면 XMP ``tiff:Orientation`` 을 채운다 —
#      XMP 에만 방향이 적힌 TIFF 가 Pillow 로만 돌았다(에디터·브라우저 규칙은 '진짜 EXIF 만').


def suppress_xmp_orientation(image: Any) -> None:
    """열린 PIL 이미지의 ``load()`` 가 XMP 에만 적힌 방향으로 돌지 않게 한다 — **load() 전에** 부른다.

    TIFF 만 해당한다(다른 포맷의 load 는 방향을 적용하지 않는다). ``getexif()`` 는 읽은 Exif 를 이미지에
    캐시하고 load_end 도 그 객체를 보므로(다시 읽는 건 seek 때뿐), 거기서 XMP 가 채운 방향만 지운다.
    IFD0 의 진짜 방향 태그는 그대로 둔다 — 그건 OpenCV 도 적용한다. 실패해도 조용히 넘어간다.
    """
    try:
        if getattr(image, "format", None) != "TIFF" or not hasattr(image, "tag_v2"):
            return
        if _EXIF_ORIENTATION_TAG in image.tag_v2:
            return
        exif = image.getexif()
        if _EXIF_ORIENTATION_TAG in exif:
            del exif[_EXIF_ORIENTATION_TAG]
    except Exception:
        pass


@contextmanager
def open_pil_image(path: Any) -> Iterator[Any]:
    """픽셀을 읽을 PIL 이미지를 연다(``with``) — load() 결과가 에디터 디코드와 같은 방향이 된다.

    파일 객체로 열어 mmap 을 끄고(위 a), ``suppress_xmp_orientation`` 을 부른다(위 b). 한글 경로 안전
    (파이썬 ``open``). 알아볼 수 없는 파일은 경로로 열 때와 같은 문구의 ``UnidentifiedImageError``.
    """
    from PIL import Image, UnidentifiedImageError
    target = os.fspath(path)
    with open(target, "rb") as handle:
        try:
            image = Image.open(handle)
        except UnidentifiedImageError as exc:
            # 파일 객체로 열면 문구가 '<_io.BufferedReader name=…>' 가 된다 — 경로로 열 때처럼
            raise UnidentifiedImageError(f"cannot identify image file {target!r}") from exc
        with image:
            suppress_xmp_orientation(image)
            yield image


def imwrite_unicode(path: Any, img, params: Optional[Sequence[int]] = None) -> bool:
    """``cv2.imwrite(path, img, params)`` 의 유니코드 경로 안전판. 성공하면 True.

    포맷은 확장자로 정하고, 확장자가 없으면 PNG 로 쓴다. 인코딩 실패·쓰기 실패는 False.
    """
    import cv2
    if path is None or img is None:
        return False
    try:
        target = os.fspath(path)
        ext = os.path.splitext(target)[1] or '.png'
        if params:
            ok, buf = cv2.imencode(ext, img, [int(v) for v in params])
        else:
            ok, buf = cv2.imencode(ext, img)
        if not ok:
            return False
        buf.tofile(target)
        return True
    except (OSError, ValueError, TypeError, cv2.error):
        return False
