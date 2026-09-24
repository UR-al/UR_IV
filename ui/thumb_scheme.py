# ui/thumb_scheme.py
"""Qt 모드 갤러리·즐겨찾기 카드 썸네일 — ``aithumb:`` 커스텀 URL 스킴.

QWebEngineView 는 카드(<img>)에 넣은 file:/// 원본을 카드 크기로 줄이지 않고 원본
해상도로 디코드한다(1852장·3.5GB 폴더에서 첫 화면만 약 48장·72MB). 프런트
``thumbnailUrl()`` 이 Qt 모드에서 ``aithumb:thumb?path=…&width=…`` 를 돌려주면 이 핸들러가
core.thumb_cache 로 만든 JPEG 를 응답한다.

- 요청 순서는 <img loading="lazy"> 가 정한다. 생성은 작은 QThreadPool 에서 하고(GUI 비차단),
  응답(job.reply)만 GUI 스레드에서 한다.
- GUI 스레드(requestStarted)는 요청자 확인·URL 파싱·확장자 문자열 검사만 한다. 파일 시스템을
  건드리는 경로 검증(resolve·is_file·시스템 폴더 차단)은 워커에서 한다 — NAS·매핑 드라이브
  갤러리에서 카드마다 네트워크 왕복이 GUI 스레드를 막지 않게.
- 화면에서 사라져 취소된 요청(job 파괴)은 생성 전에 건너뛴다.
- 실패하면 404 — 프런트가 원본 URL 로 한 번 폴백한다(utils/thumbFallback.ts).
- 스킴 등록(``register_thumb_scheme``)은 QApplication 생성 **전**에 해야 한다. 등록되지 않은
  실행 경로(웹 모드 등)에서는 핸들러를 달지 않는다.
"""
from __future__ import annotations

import os
import threading
from typing import Optional

from PyQt6.QtCore import QBuffer, QByteArray, QIODevice, QThreadPool, pyqtSignal
from PyQt6.QtWebEngineCore import (
    QWebEngineUrlRequestJob,
    QWebEngineUrlScheme,
    QWebEngineUrlSchemeHandler,
)

from core.path_safety import safe_input_path
from core.thumb_cache import THUMB_SCHEME, THUMB_SOURCE_EXTS, get_or_make_thumb, parse_thumb_url

_SCHEME_BYTES = THUMB_SCHEME.encode("ascii")
# 이 스킴을 요청할 수 있는 페이지 — 앱 번들(file://)과 개발 서버(loopback)만. 같은 프로필의
# 외부 웹 페이지가 로컬 이미지를 끌어가지 못하게 한다.
_ALLOWED_INITIATOR_SCHEMES = frozenset({"", "file", "qrc", THUMB_SCHEME})
_LOOPBACK_HOSTS = frozenset({"localhost", "127.0.0.1", "::1", "[::1]"})


def register_thumb_scheme() -> bool:
    """QApplication 생성 전에 한 번. 이미 등록돼 있으면 그대로 True."""
    try:
        if bytes(QWebEngineUrlScheme.schemeByName(_SCHEME_BYTES).name()) == _SCHEME_BYTES:
            return True
        scheme = QWebEngineUrlScheme(_SCHEME_BYTES)
        scheme.setSyntax(QWebEngineUrlScheme.Syntax.Path)
        scheme.setFlags(QWebEngineUrlScheme.Flag.SecureScheme)
        QWebEngineUrlScheme.registerScheme(scheme)
        return True
    except Exception:
        return False


def thumb_scheme_registered() -> bool:
    try:
        return bytes(QWebEngineUrlScheme.schemeByName(_SCHEME_BYTES).name()) == _SCHEME_BYTES
    except Exception:
        return False


def initiator_allowed(scheme: str, host: str) -> bool:
    scheme = (scheme or "").lower()
    if scheme in _ALLOWED_INITIATOR_SCHEMES:
        return True
    return scheme in ("http", "https") and (host or "").lower() in _LOOPBACK_HOSTS


def thumb_source_ext_allowed(raw_path: str) -> bool:
    """GUI 스레드용 사전 검사 — 파일 시스템을 건드리지 않고 확장자 문자열만 본다.

    최종 판정(resolve 후 확장자·is_file·시스템 폴더)은 워커의 safe_input_path 가 한다.
    Windows 가 끝의 공백·점을 떼고 여는 이름('a.png.')은 통과시켜 최종 판정에 맡긴다.
    """
    text = str(raw_path or "").rstrip(" .")
    return os.path.splitext(text)[1].lower() in THUMB_SOURCE_EXTS


def _default_thread_count() -> int:
    return max(2, min(4, (os.cpu_count() or 4) // 4))


class ThumbSchemeHandler(QWebEngineUrlSchemeHandler):
    """``aithumb:`` 요청 → 캐시된(없으면 만든) JPEG 썸네일."""

    _finished = pyqtSignal(int, object)  # job id, JPEG bytes | None

    def __init__(
        self,
        cache_dir: str,
        parent=None,
        *,
        max_threads: Optional[int] = None,
        legacy_dir: Optional[str] = None,
    ):
        super().__init__(parent)
        self._cache_dir = cache_dir
        # 옛 캐시 폴더 — 새 폴더에 없는 카드 썸네일을 렌더 전에 옮겨 온다(core.legacy_thumb_cache)
        self._legacy_dir = legacy_dir or None
        self._pool = QThreadPool(self)
        self._pool.setMaxThreadCount(max_threads or _default_thread_count())
        self._jobs: dict[int, tuple[QWebEngineUrlRequestJob, threading.Event]] = {}
        self._next_id = 0
        self._finished.connect(self._reply)

    def requestStarted(self, job: QWebEngineUrlRequestJob) -> None:  # noqa: N802 (Qt API)
        try:
            initiator = job.initiator()
            if not initiator_allowed(initiator.scheme(), initiator.host()):
                job.fail(QWebEngineUrlRequestJob.Error.RequestDenied)
                return
            parsed = parse_thumb_url(bytes(job.requestUrl().toEncoded()).decode("ascii", "replace"))
            # 여기서는 문자열 검사만 — resolve·is_file 은 네트워크 드라이브에서 느리므로 워커에서.
            if not parsed or not thumb_source_ext_allowed(parsed[0]):
                job.fail(QWebEngineUrlRequestJob.Error.UrlNotFound)
                return
        except RuntimeError:
            return
        raw_path, width = parsed
        self._next_id += 1
        job_id = self._next_id
        cancelled = threading.Event()
        self._jobs[job_id] = (job, cancelled)
        job.destroyed.connect(lambda *_args, _id=job_id, _flag=cancelled: self._forget(_id, _flag))
        self._pool.start(lambda: self._render(job_id, raw_path, width, cancelled))

    def _forget(self, job_id: int, cancelled: threading.Event) -> None:
        cancelled.set()
        self._jobs.pop(job_id, None)

    def _render(self, job_id: int, raw_path: str, width: int, cancelled: threading.Event) -> None:
        """워커 스레드 — 경로 검증과 생성. 화면에서 사라진 요청은 만들지 않는다.

        검증에 실패하면(없는 파일·시스템 폴더·막힌 확장자) data=None 으로 알려 404 가 된다.
        """
        data = None
        if not cancelled.is_set():
            try:
                source = safe_input_path(raw_path, allowed_exts=THUMB_SOURCE_EXTS)
                thumb = (get_or_make_thumb(source, width, self._cache_dir, legacy_dir=self._legacy_dir)
                         if source else None)
                if thumb and not cancelled.is_set():
                    with open(thumb, "rb") as handle:
                        data = handle.read()
            except Exception:
                data = None
        try:
            self._finished.emit(job_id, data)
        except RuntimeError:
            pass

    def _reply(self, job_id: int, data) -> None:
        entry = self._jobs.pop(job_id, None)
        if entry is None:
            return
        job, _cancelled = entry
        try:
            if not data:
                job.fail(QWebEngineUrlRequestJob.Error.UrlNotFound)
                return
            buffer = QBuffer(job)  # job 이 파괴될 때 함께 정리된다
            buffer.setData(QByteArray(data))
            buffer.open(QIODevice.OpenModeFlag.ReadOnly)
            job.reply(b"image/jpeg", buffer)
        except RuntimeError:
            # 응답 직전에 페이지가 요청을 취소함 — job 이 이미 파괴됐다.
            pass


def install_thumb_scheme_handler(
    profile, cache_dir: str, *, legacy_dir: Optional[str] = None,
) -> Optional[ThumbSchemeHandler]:
    """프로필에 핸들러를 단다. 스킴이 등록되지 않은 실행 경로면 None(프런트는 원본 폴백).

    ``legacy_dir`` (옛 캐시 폴더)를 주면 새 폴더에 없는 키를 렌더 전에 거기서 옮겨 온다.
    """
    if profile is None or not thumb_scheme_registered():
        return None
    try:
        os.makedirs(cache_dir, exist_ok=True)
    except OSError:
        return None
    handler = ThumbSchemeHandler(cache_dir, profile, legacy_dir=legacy_dir)
    profile.installUrlSchemeHandler(_SCHEME_BYTES, handler)
    return handler
