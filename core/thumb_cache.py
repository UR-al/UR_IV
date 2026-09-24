# core/thumb_cache.py
"""갤러리·히스토리·웹 모드가 함께 쓰는 썸네일 디스크 캐시. Qt 비의존.

예전에는 구현이 두 벌이었다 — Vue ``generateThumbnails``(sha1 키, 품질 80, 알파를 검게)와
웹 ``/thumbnail``(sha256 키, 품질 86, 알파를 배경색으로). 이제 둘 다, 그리고 Qt 모드 갤러리
카드용 ``aithumb:`` 스킴 핸들러도 이 모듈 하나를 쓴다.

- 키: ``sha1(normpath(path)@width)`` — 폴더는 config.THUMB_DIR(image_cache/thumbs_v2). 옛
  image_cache/thumbs 의 이 형식 썸네일은 core.legacy_thumb_cache 가 새 폴더로 옮긴다(재생성 없음).
  배경 정리(시작 30초 뒤)를 기다리지 않게, ``legacy_dir`` 를 주면 새 폴더에서 빗나간 키를 렌더 전에
  옛 자리에서 먼저 옮겨 온다(원본을 지운 항목도 썸네일을 잃지 않는다)
- 배치: sha1 앞 2자리 샤딩(core.cache_cleanup.shard_path)
- 갱신: 썸네일 JPEG 주석에 적은 원본 서명(mtime_ns, 크기)이 지금 원본과 다르면 다시 만든다
  (core.cache_cleanup.thumb_is_stale). 렌더 중 원본이 바뀌면 결과를 버리고 다시 만든다.
- 쓰기: 같은 폴더 임시 파일 → os.replace (동시 요청·중단에도 깨진 JPEG 가 남지 않음)
"""
from __future__ import annotations

import hashlib
import os
import threading
from typing import Optional
from urllib.parse import quote, unquote

from core.cache_cleanup import (
    ensure_shard_dir,
    shard_path,
    source_signature,
    thumb_is_stale,
    thumb_signature_comment,
)
from core.legacy_thumb_cache import adopt_legacy_thumb

THUMB_QUALITY = 85
# Qt 모드 카드 썸네일 URL 스킴 — 프런트 utils/media.js thumbnailUrl 과 같은 모양이어야 한다:
#   aithumb:thumb?path=<encodeURIComponent(path)>&width=<버킷 폭>
THUMB_SCHEME = "aithumb"
THUMB_URL_HOST_PATH = "thumb"
# 썸네일로 줄일 수 있는 정지/애니메이션 이미지(갤러리 목록 _GALLERY_MEDIA_EXTS 의 이미지 부분)
THUMB_SOURCE_EXTS: frozenset[str] = frozenset({
    ".png", ".jpg", ".jpeg", ".webp", ".gif", ".apng", ".bmp", ".tif", ".tiff", ".avif",
})
# 알파가 있는 원본은 앱 기본 배경(--bg-primary)과 같은 어두운 면에 합성한다.
THUMB_BACKGROUND = (13, 13, 13)

# 갤러리 카드 폭 버킷(px). 슬라이더(100~380px) × DPR(최대 2) = 760px 까지 선명해야 한다.
# 폭마다 캐시 파일이 생기므로 요청 폭을 가장 가까운 '같거나 큰' 버킷으로 올린다.
GALLERY_THUMB_BUCKETS: tuple[int, ...] = (192, 256, 384, 512, 768)
MIN_THUMB_WIDTH = 64
MAX_THUMB_WIDTH = 1024
DEFAULT_THUMB_WIDTH = 384


def clamp_thumb_width(width: object, default: int = DEFAULT_THUMB_WIDTH) -> int:
    try:
        value = int(round(float(width)))  # type: ignore[arg-type]
    except (TypeError, ValueError, OverflowError):
        value = default
    return max(MIN_THUMB_WIDTH, min(MAX_THUMB_WIDTH, value))


def bucket_thumb_width(width: object, buckets: tuple[int, ...] = GALLERY_THUMB_BUCKETS) -> int:
    """요청 폭 이상인 가장 작은 버킷(없으면 가장 큰 버킷). 잘못된 값은 기본 폭 기준."""
    value = clamp_thumb_width(width)
    for bucket in buckets:
        if bucket >= value:
            return bucket
    return buckets[-1]


def build_thumb_url(path: str, width: object) -> str:
    """media.js thumbnailUrl(Qt 모드)과 같은 URL. 테스트·파이썬 쪽 생성용."""
    return f"{THUMB_SCHEME}:{THUMB_URL_HOST_PATH}?path={quote(str(path), safe='')}&width={bucket_thumb_width(width)}"


def parse_thumb_url(url: str) -> Optional[tuple[str, int]]:
    """``aithumb:thumb?path=…&width=…`` → (원본 경로, 버킷 폭). 모양이 다르면 None.

    ``+`` 를 공백으로 바꾸지 않는다(form 디코딩 아님) — 프런트 encodeURIComponent 는 공백을
    %20, '+' 를 %2B 로 보낸다.
    """
    text = str(url or "")
    prefix = f"{THUMB_SCHEME}:"
    if not text.lower().startswith(prefix):
        return None
    rest = text[len(prefix):].lstrip("/")
    head, sep, query = rest.partition("?")
    if not sep or head.rstrip("/").lower() != THUMB_URL_HOST_PATH:
        return None
    fields: dict[str, str] = {}
    for part in query.split("&"):
        key, _eq, value = part.partition("=")
        if key and key not in fields:
            fields[key] = unquote(value)
    path = fields.get("path", "")
    if not path:
        return None
    return path, bucket_thumb_width(fields.get("width", DEFAULT_THUMB_WIDTH))


def thumb_key(path: str, width: int) -> str:
    return hashlib.sha1(f"{os.path.normpath(path)}@{int(width)}".encode("utf-8")).hexdigest()


def thumb_path(cache_dir: str, path: str, width: int) -> str:
    return shard_path(cache_dir, thumb_key(path, width), ".jpg")


class SourceChangedError(OSError):
    """썸네일을 만드는 사이 원본이 바뀌었다 — 만든 것을 버리고 다시 만든다."""


# 렌더 중 원본이 바뀌면 다시 해 보는 횟수. 계속 바뀌는 중이면 옛 썸네일을 내주지 않고 None.
RENDER_ATTEMPTS = 3


def render_thumbnail(source: str, dest: str, width: int, *, quality: int = THUMB_QUALITY) -> None:
    """``source`` 를 긴 변 ``width`` 이하 JPEG 로 ``dest`` 에 원자적으로 쓴다(실패는 예외).

    썸네일 주석(COM)에 **읽기 전** 원본 서명(mtime_ns, 크기)을 적고, 쓰기 직전에 원본 서명을 다시
    잰다 — 그 사이 원본이 바뀌었으면(렌더 중 덮어쓰기) 결과를 버리고 ``SourceChangedError``.
    그래서 옛 그림이 새 원본의 썸네일로 남지 않는다(판정: core.cache_cleanup.thumb_is_stale).
    """
    from PIL import Image, ImageOps

    before = source_signature(source)
    if before is None:
        raise FileNotFoundError(source)
    tmp = f"{dest}.{os.getpid()}.{threading.get_ident()}.tmp"
    try:
        with Image.open(source) as image:
            try:
                # JPEG 는 축소 디코드로 풀해상도 디코드를 피한다(다른 포맷은 무시됨)
                image.draft("RGB", (width, width))
            except Exception:
                pass
            image = ImageOps.exif_transpose(image)
            image.thumbnail((width, width), Image.Resampling.LANCZOS)
            has_alpha = image.mode in ("RGBA", "LA", "PA") or (image.mode == "P" and "transparency" in image.info)
            if has_alpha:
                rgba = image.convert("RGBA")
                canvas = Image.new("RGB", rgba.size, THUMB_BACKGROUND)
                canvas.paste(rgba, mask=rgba.getchannel("A"))
                image = canvas
            else:
                image = image.convert("RGB")
            image.save(tmp, "JPEG", quality=quality, optimize=True, comment=thumb_signature_comment(before))
        if source_signature(source) != before:
            raise SourceChangedError(source)
        os.replace(tmp, dest)
    finally:
        try:
            if os.path.exists(tmp):
                os.remove(tmp)
        except OSError:
            pass


def get_or_make_thumb(
    source: str,
    width: int,
    cache_dir: str,
    *,
    quality: int = THUMB_QUALITY,
    legacy_dir: Optional[str] = None,
) -> Optional[str]:
    """캐시된 썸네일 경로. 없거나 만든 뒤 원본이 바뀌었으면 만든다. 만들 수 없으면 None.

    ``legacy_dir`` (옛 캐시 폴더, config.LEGACY_THUMB_DIR)를 주면 ``cache_dir`` 에 없는 키를 먼저
    거기서 옮겨 온다(core.legacy_thumb_cache.adopt_legacy_thumb) — 그대로 쓸 수 있는 썸네일이면
    렌더하지 않고, 원본이 지워졌어도 그 썸네일을 돌려준다.
    원본을 읽을 수 없는데 옛 썸네일이 있으면(지워진 파일 등) 그것을 돌려준다.
    렌더 중 원본이 바뀌면 다시 만들고, ``RENDER_ATTEMPTS`` 번 모두 바뀌는 중이었으면 None
    (호출자는 원본으로 폴백한다 — 옛 그림을 새것처럼 내주지 않는다).
    다른 이유로 렌더가 실패해도(다른 스레드가 캐시 JPEG 를 열고 있어 os.replace 가
    PermissionError, 비원자 쓰기 도중이라 원본이 잘려 있음 등) 원본이 있는 한 서명이 맞는
    썸네일만 돌려준다 — 옛 서명의 파일을 내주면 프리페처가 옛 ``v`` 로 알려 히스토리가 그대로 굳는다.
    """
    width = clamp_thumb_width(width)
    dest = thumb_path(cache_dir, source, width)
    if legacy_dir:
        # 새 폴더에 없을 때만 옛 자리를 연다(adopt 가 dest 를 먼저 확인). '원본 없음 → None' 분기보다 먼저.
        adopt_legacy_thumb(legacy_dir, dest, source)
    if thumb_is_stale(source, dest):
        if not os.path.isfile(source):
            return dest if os.path.isfile(dest) else None
        ensure_shard_dir(dest)
        for _attempt in range(RENDER_ATTEMPTS):
            try:
                render_thumbnail(source, dest, width, quality=quality)
                break
            except SourceChangedError:
                continue
            except Exception:
                # 그사이 다른 스레드가 지금 원본으로 만들었으면 그것은 쓴다(서명 일치).
                # 원본이 사라졌으면 있는 썸네일(thumb_is_stale 가 False) — 위 분기와 같은 규칙.
                return None if thumb_is_stale(source, dest) else dest
        else:
            return None
    return dest if os.path.isfile(dest) else None
