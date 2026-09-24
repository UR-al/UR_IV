# core/clipboard_paste.py
"""에디터 클립보드 붙여넣기 — base64 이미지를 임시 폴더에 안전하게 저장한다.

예전 editorPasteImage 는 ``mime_type.split('/')[-1]`` 을 그대로 확장자로 붙였다.
그래서 ``image/..\\..\\x.bat`` 같은 mime 을 주면 임시 폴더 밖 임의 경로에 호출자가
정한 바이트를 쓸 수 있었다(웹 모드 _WEB_METHODS 공개 슬롯).

여기서는
  - 확장자를 **명시 맵**에서만 고른다(mimetypes.guess_extension 은 이 venv 에서
    image/webp·image/jpg 를 모른다),
  - 바이트가 실제 그 계열 이미지인지 시그니처로 확인하고,
  - 파일은 ``tempfile.mkstemp`` 로 만들어 같은 초에 붙여넣어도 덮어쓰지 않으며,
  - 만든 경로가 임시 폴더 안인지 한 번 더 검증한다.
  - 붙여넣을 때마다 오래된 ``clipboard_*`` 파일을 치운다(예전엔 %TEMP% 에 끝없이 쌓였다).
    같은 폴더의 자동 저장 복구본(``_autosave_session.png``)은 건드리지 않는다.

데스크톱은 Qt 시스템 클립보드를 직접 읽는 슬롯(``editorPasteFromSystemClipboard``)이
``store_clipboard_bytes`` 로, 웹 모드는 브라우저가 보낸 base64 가 ``save_clipboard_image``
로 같은 저장 규칙을 탄다. Qt 를 모르는 순수 모듈이다.
"""
from __future__ import annotations

import base64
import binascii
import os
import tempfile
import time
from pathlib import Path
from typing import Iterable

#: 프론트 EditorView 화이트리스트(_CLIPBOARD_ALLOWED_TYPES)와 같은 5종.
PASTE_EXTENSIONS: dict[str, str] = {
    "image/png": "png",
    "image/jpeg": "jpg",
    "image/jpg": "jpg",
    "image/bmp": "bmp",
    "image/webp": "webp",
}
#: 붙여넣기 한 장의 상한 — 8K RGBA PNG 도 넉넉히 들어간다.
MAX_PASTE_BYTES = 64 * 1024 * 1024
PASTE_DIR_NAME = "AIStudioPro_editor"


class ClipboardPasteError(ValueError):
    """허용되지 않은 붙여넣기 요청."""


def paste_extension_for_mime(mime_type: object) -> str | None:
    """허용된 mime 이면 확장자(점 없이), 아니면 None."""
    key = str(mime_type or "").strip().lower()
    return PASTE_EXTENSIONS.get(key)


def sniff_image_extension(data: bytes) -> str | None:
    """바이트 시그니처로 본 실제 이미지 계열(png/jpg/bmp/webp). 모르면 None."""
    if data.startswith(b"\x89PNG\r\n\x1a\n"):
        return "png"
    if data.startswith(b"\xff\xd8\xff"):
        return "jpg"
    if data.startswith(b"BM") and len(data) >= 26:
        return "bmp"
    if len(data) >= 12 and data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return "webp"
    return None


def decode_paste_payload(b64_data: object) -> bytes:
    """엄격한 base64 디코드. data: URL 접두사는 떼어 준다."""
    text = str(b64_data or "").strip()
    if text.startswith("data:"):
        _, _, text = text.partition(",")
    if not text:
        raise ClipboardPasteError("붙여넣을 이미지 데이터가 비어 있습니다")
    # base64 4바이트 ≈ 원본 3바이트 — 디코드 전에 상한을 먼저 본다.
    if len(text) > (MAX_PASTE_BYTES // 3 + 1) * 4:
        raise ClipboardPasteError("붙여넣을 이미지가 너무 큽니다")
    try:
        raw = base64.b64decode(text, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise ClipboardPasteError("이미지 데이터(base64)가 올바르지 않습니다") from exc
    if not raw:
        raise ClipboardPasteError("붙여넣을 이미지 데이터가 비어 있습니다")
    if len(raw) > MAX_PASTE_BYTES:
        raise ClipboardPasteError("붙여넣을 이미지가 너무 큽니다")
    return raw


def default_paste_dir() -> Path:
    return Path(tempfile.gettempdir()) / PASTE_DIR_NAME


#: 붙여넣기 임시 파일 정리 — 이보다 오래됐고, 최신 KEEP 장에 들지 않는 것만 지운다.
#: (한 세션이 하루를 넘겨도 방금 붙여넣은 몇 장은 undo 바닥으로 살아 있어야 한다)
CLIPBOARD_MAX_AGE_HOURS = 24.0
CLIPBOARD_KEEP_NEWEST = 10
_PASTE_PREFIX = "clipboard_"
_PASTE_SUFFIXES = frozenset({".png", ".jpg", ".bmp", ".webp"})
#: 탐색기에서 '복사'한 이미지 파일을 붙여넣을 때 받는 확장자
CLIPBOARD_FILE_EXTENSIONS = frozenset({".png", ".jpg", ".jpeg", ".bmp", ".webp"})


def prune_clipboard_images(
    directory: str | os.PathLike | None = None,
    *,
    max_age_hours: float | None = None,
    keep_newest: int | None = None,
    now: float | None = None,
    exclude: Iterable[str | os.PathLike] = (),
) -> int:
    """오래된 붙여넣기 임시 파일(``clipboard_*.png|jpg|bmp|webp``)을 지우고 지운 개수를 돌려준다.

    같은 폴더의 자동 저장 복구본(``_autosave_session.png``)이나 다른 파일은 절대 건드리지
    않는다(``prune_editor_temp`` 를 쓰지 않는 이유). 지우기 실패는 조용히 넘긴다.
    기준값을 생략하면 호출 시점의 모듈 상수(CLIPBOARD_MAX_AGE_HOURS/KEEP_NEWEST)를 쓴다.
    """
    if max_age_hours is None:
        max_age_hours = CLIPBOARD_MAX_AGE_HOURS
    if keep_newest is None:
        keep_newest = CLIPBOARD_KEEP_NEWEST
    base = Path(directory) if directory is not None else default_paste_dir()
    if not base.is_dir():
        return 0
    cutoff = (time.time() if now is None else now) - max(0.0, float(max_age_hours)) * 3600.0
    skip = {os.path.normcase(os.path.abspath(os.fspath(p))) for p in exclude}
    entries = []
    for entry in base.iterdir():
        name = entry.name
        if not name.startswith(_PASTE_PREFIX) or entry.suffix.lower() not in _PASTE_SUFFIXES:
            continue
        try:
            if not entry.is_file():
                continue
            mtime = entry.stat().st_mtime
        except OSError:
            continue
        entries.append((mtime, entry))
    entries.sort(key=lambda item: item[0], reverse=True)
    removed = 0
    for index, (mtime, entry) in enumerate(entries):
        if index < max(0, int(keep_newest)) or mtime >= cutoff:
            continue
        if os.path.normcase(os.path.abspath(entry)) in skip:
            continue
        try:
            entry.unlink()
            removed += 1
        except OSError:
            pass
    return removed


def pick_clipboard_image_file(paths: Iterable[object]) -> str | None:
    """클립보드의 파일 목록(탐색기 '복사') 중 열 수 있는 첫 이미지 파일. 없으면 None."""
    for raw in paths or ():
        try:
            text = os.fspath(raw) if isinstance(raw, (str, os.PathLike)) else ""
        except TypeError:
            continue
        if not text:
            continue
        candidate = Path(text)
        if candidate.suffix.lower() not in CLIPBOARD_FILE_EXTENSIONS:
            continue
        try:
            if candidate.is_file():
                return str(candidate.resolve())
        except OSError:
            continue
    return None


def save_clipboard_image(b64_data: object, mime_type: object, *, directory: str | os.PathLike | None = None) -> str:
    """검증된 클립보드 이미지를 임시 폴더에 새 파일로 쓰고 절대 경로를 돌려준다.

    실패하면 ClipboardPasteError(사용자에게 그대로 보여 줄 한국어 메시지).
    """
    if paste_extension_for_mime(mime_type) is None:
        raise ClipboardPasteError(
            f"지원하지 않는 이미지 형식입니다: {str(mime_type or '')[:40]!r} (PNG/JPG/BMP/WEBP만 가능)"
        )
    raw = decode_paste_payload(b64_data)
    return store_clipboard_bytes(raw, directory=directory)


def store_clipboard_bytes(raw: bytes, *, directory: str | os.PathLike | None = None) -> str:
    """이미지 바이트를 붙여넣기 임시 폴더에 새 파일로 쓰고 절대 경로를 돌려준다.

    바이트 시그니처로 확장자를 정하고(PNG/JPG/BMP/WEBP 가 아니면 거부), 크기 상한을 지키며,
    다 쓴 뒤 오래된 붙여넣기 파일을 치운다. 실패하면 ClipboardPasteError.
    """
    if not raw:
        raise ClipboardPasteError("붙여넣을 이미지 데이터가 비어 있습니다")
    if len(raw) > MAX_PASTE_BYTES:
        raise ClipboardPasteError("붙여넣을 이미지가 너무 큽니다")
    ext = sniff_image_extension(raw)
    if ext is None:
        raise ClipboardPasteError("클립보드 데이터가 PNG/JPG/BMP/WEBP 이미지가 아닙니다")
    # 확장자는 mime 이 아니라 실제 바이트 기준 — 파일 이름이 내용을 속이지 않게.
    base = Path(directory) if directory is not None else default_paste_dir()
    base.mkdir(parents=True, exist_ok=True)
    root = base.resolve(strict=True)
    fd, name = tempfile.mkstemp(prefix="clipboard_", suffix=f".{ext}", dir=str(root))
    target = Path(name)
    try:
        resolved = target.resolve(strict=True)
        if resolved.parent != root:
            raise ClipboardPasteError("임시 폴더 밖으로 저장할 수 없습니다")
        with os.fdopen(fd, "wb") as fh:
            fd = -1
            fh.write(raw)
    except BaseException:
        if fd != -1:
            os.close(fd)
        try:
            target.unlink()
        except OSError:
            pass
        raise
    # 정리는 붙여넣기를 절대 실패시키지 않는다 — 방금 쓴 파일은 최신이라 남는다.
    try:
        prune_clipboard_images(root, exclude=(resolved,))
    except OSError:
        pass
    return str(resolved)
