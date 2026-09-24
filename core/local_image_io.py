"""로컬 업로드 이미지 I/O 공용 헬퍼 — 손 재구성(core.hand_reconstruction)과 조명 편집
(ui.relight_actions)이 함께 쓴다.

두 모듈은 data URL 디코드·정지 래스터 검증·PNG 메타데이터 보존·O_EXCL 내보내기를 거의
1:1로 복제했고, 검증이 이미 네 군데에서 갈라져 있었다.
  1) 1 MB 넘는 텍스트 메타데이터: relight는 조용히 버림, hand는 오류
  2) MIME과 실제 형식 불일치(PNG를 image/jpeg로 위장): relight만 통과
  3) 깨진 이미지: relight만 PIL 원문 오류를 그대로 노출
  4) ICC/EXIF 한도와 메타데이터 크기 계산 방식이 다름
여기서는 더 엄격한 쪽(손 재구성)을 기준으로 하나로 맞춘다. 출처 보존을 위해 메타데이터
초과는 '조용히 버리기'가 아니라 실패로 통일한다. ICC는 relight 규칙이 옳다 — RGB 계열
원본의 프로파일만 RGB/RGBA PNG에 붙인다(회색·CMYK 프로파일을 붙이면 잘못된 PNG).

한도 상수는 각 호출 모듈에 그대로 두고 **인자로** 받는다. 테스트가
`core.hand_reconstruction.MAX_*`, `ui.relight_actions.MAX_*`를 patch하므로 호출자는 매번
모듈 전역을 읽어 넘겨야 한다(기본 인자로 굳히면 patch가 무력해진다).

export_exclusive_png 외에는 파일시스템·네트워크·GPU를 쓰지 않는다.
"""
from __future__ import annotations

import base64
import binascii
from contextlib import contextmanager
from datetime import datetime
import io
from pathlib import Path
import re
from typing import Any, Callable, Iterator

from PIL import Image, PngImagePlugin

STILL_FORMATS = {"png": "PNG", "jpeg": "JPEG", "webp": "WEBP"}
_DATA_URL = re.compile(r"data:image/(png|jpeg|webp);base64,([A-Za-z0-9+/=]+)")
# RGB/RGBA PNG에 붙여도 되는 ICC — 원본 색공간이 RGB 계열일 때만 (팔레트는 RGB 항목).
_RGB_ICC_MODES = frozenset({"RGB", "RGBA", "RGBX", "RGBa", "P", "PA"})
_PNG_KEYWORD_MAX_BYTES = 79
_RASTER_ERRORS = (OSError, SyntaxError, Image.DecompressionBombError)


def bytes_label(limit: int) -> str:
    """사용자 문구용 크기 표기 — 64 MiB → '64 MB', 테스트가 줄인 한도 → '16 bytes'."""
    mib = 1024 * 1024
    return f"{limit / mib:g} MB" if limit >= mib else f"{limit:,} bytes"


def pixels_label(limit: int) -> str:
    """사용자 문구용 픽셀 한도 — 16,777,216 → '16 MP', 테스트가 줄인 한도 → '100 px'."""
    mp = 1024 * 1024
    return f"{limit / mp:g} MP" if limit >= mp else f"{limit:,} px"


def decode_image_data_url(value: Any, label: str, *, max_bytes: int) -> bytes:
    """업로드한 PNG/JPEG/WebP data URL → 원본 바이트.

    경로·외부 URL·SVG는 거부하고, 선언한 MIME을 파일 확장자가 아니라 실제 래스터 형식과
    대조한다(PNG를 image/jpeg로 위장한 입력 거부).
    """
    size = bytes_label(max_bytes)
    if not isinstance(value, str) or len(value) > max_bytes * 4 // 3 + 128:
        raise ValueError(f"{label}: {size} 이하의 PNG/JPEG/WebP data URL이 필요합니다.")
    match = _DATA_URL.fullmatch(value)
    if not match:
        raise ValueError(f"{label}: 업로드한 PNG/JPEG/WebP만 허용합니다. 경로나 외부 URL은 사용할 수 없습니다.")
    try:
        data = base64.b64decode(match[2], validate=True)
    except (ValueError, binascii.Error) as exc:
        raise ValueError(f"{label}: 올바르지 않은 base64 이미지입니다.") from exc
    if not data or len(data) > max_bytes:
        raise ValueError(f"{label}: 이미지 파일은 {size} 이하여야 합니다.")
    try:
        with Image.open(io.BytesIO(data)) as opened:
            actual = opened.format
    except _RASTER_ERRORS as exc:
        raise ValueError(f"{label}: 이미지를 읽을 수 없습니다.") from exc
    if actual != STILL_FORMATS[match[1]]:
        raise ValueError(f"{label}: 이미지 MIME과 실제 파일 형식이 다릅니다.")
    return data


@contextmanager
def open_still_raster(data: Any, label: str, *, max_bytes: int, max_pixels: int) -> Iterator[Image.Image]:
    """단일 프레임 PNG/JPEG/WebP만 연다. 본문에서 난 PIL 오류도 한국어 ValueError로 바꾼다.

    크기 한도는 픽셀을 디코드하기 전에(헤더만으로) 검사한다.
    """
    if not isinstance(data, bytes) or not data or len(data) > max_bytes:
        raise ValueError(f"{label}: {bytes_label(max_bytes)} 이하의 이미지 바이트가 필요합니다.")
    try:
        with Image.open(io.BytesIO(data)) as opened:
            if opened.format not in STILL_FORMATS.values() or getattr(opened, "n_frames", 1) != 1:
                raise ValueError(f"{label}: 단일 정지 PNG/JPEG/WebP 이미지만 허용합니다.")
            width, height = opened.size
            if min(width, height) < 2 or width * height > max_pixels:
                raise ValueError(f"{label}: 최소 2×2, 최대 {pixels_label(max_pixels)} 이미지를 사용하세요.")
            yield opened
    except _RASTER_ERRORS as exc:
        raise ValueError(f"{label}: 손상되었거나 지원되지 않는 이미지입니다.") from exc


def _is_png_keyword(key: str) -> bool:
    """PNG tEXt/iTXt 키워드로 쓸 수 있는가 (1–79 byte Latin-1). 아니면 PNG에 적을 수 없다."""
    try:
        encoded = key.encode("latin-1")
    except UnicodeEncodeError:
        return False
    return 0 < len(encoded) <= _PNG_KEYWORD_MAX_BYTES


def capture_png_metadata(image: Image.Image, *, limit: int, source_mode: str) -> dict[str, Any]:
    """PNG로 다시 쓸 원본 메타데이터(텍스트·ICC·EXIF). 한도를 넘으면 버리지 않고 실패한다.

    image: load()와 exif_transpose를 마친 이미지 — PNG의 IDAT 뒤 텍스트 청크까지 읽혀 있고,
      EXIF 방향 태그가 정리된 상태여야 한다(방향을 두 번 적용하지 않게).
    source_mode: 디코드 직후 원본 모드. ICC가 RGB 계열 프로파일인지 판단한다.
    텍스트 크기는 키+값 UTF-8 바이트 합으로 센다. PNG 키워드로 쓸 수 없는 키는 원본 PNG에도
    합법적으로 존재할 수 없으므로 제외한다.
    """
    text = {key: value for key, value in image.info.items()
            if isinstance(key, str) and isinstance(value, str) and _is_png_keyword(key)}
    if sum(len(key.encode("utf-8")) + len(value.encode("utf-8")) for key, value in text.items()) > limit:
        raise ValueError(f"원본의 텍스트 메타데이터가 {bytes_label(limit)}를 넘습니다.")
    icc = image.info.get("icc_profile") if source_mode in _RGB_ICC_MODES else None
    exif = image.getexif().tobytes() if image.getexif() else None
    for value in (icc, exif):
        if value is not None and (not isinstance(value, bytes) or len(value) > limit):
            raise ValueError("원본의 ICC/EXIF 메타데이터가 지원 범위를 넘습니다.")
    return {"text": text, "icc": icc or None, "exif": exif}


def png_save_options(metadata: dict[str, Any] | None) -> dict[str, Any]:
    """capture_png_metadata 결과 → Image.save(format='PNG', **options).

    icc_profile은 **항상 명시**한다(없으면 None). Pillow의 PNG 저장은 옵션에 없으면
    image.info['icc_profile']로 폴백하므로, 원본에서 복사한 이미지(convert/copy는 info를
    유지)가 걸러 낸 회색·CMYK 프로파일을 몰래 다시 써 버린다.
    """
    options: dict[str, Any] = {"icc_profile": (metadata or {}).get("icc") or None}
    if not metadata:
        return options
    info = PngImagePlugin.PngInfo()
    for key, value in metadata.get("text", {}).items():
        info.add_text(key, value)
    options["pnginfo"] = info
    if metadata.get("exif"):
        options["exif"] = metadata["exif"]
    return options


def encode_png(image: Image.Image, metadata: dict[str, Any] | None = None) -> bytes:
    """승인된 메타데이터만 쓴다 — 이미지 자신의 info(원본 ICC 등)는 새지 않는다."""
    stream = io.BytesIO()
    image.save(stream, format="PNG", **png_save_options(metadata))
    return stream.getvalue()


def png_data_url(data: bytes) -> str:
    return "data:image/png;base64," + base64.b64encode(data).decode("ascii")


def export_exclusive_png(data: bytes, output_root: Any, *, subdir: str, prefix: str,
                         now: Callable[[], datetime], token: Callable[[], str],
                         outside_message: str, exhausted_message: str, attempts: int = 5) -> str:
    """앱 출력 폴더의 subdir 아래에 **새 파일로만** 저장한다(O_EXCL). 기존 파일은 절대 덮어쓰지 않는다.

    now/token은 호출 모듈이 넘긴다 — 테스트가 그 모듈의 datetime/secrets를 patch한다.
    쓰기 도중 실패하면 방금 만든 미완성 파일만 지운다.
    """
    root = Path(output_root).resolve()
    destination = root / subdir
    destination.mkdir(parents=True, exist_ok=True)
    destination = destination.resolve()
    if not destination.is_relative_to(root):
        raise ValueError(outside_message)
    for _ in range(max(1, int(attempts))):
        path = destination / f"{prefix}_{now():%Y%m%d_%H%M%S}_{token()}.png"
        try:
            with path.open("xb") as stream:
                try:
                    stream.write(data)
                except Exception:
                    stream.close()
                    path.unlink(missing_ok=True)  # 이번에 새로 만든 미완성 파일만
                    raise
            return str(path)
        except FileExistsError:
            continue
    raise ValueError(exhausted_message)
