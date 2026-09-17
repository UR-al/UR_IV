# web_main_ui.py
"""
AI Studio Pro — 웹(브라우저) 진입점.

run_gui.bat(new_main_ui.py)는 PyQt 창 안의 QWebEngineView로 Vue를 띄운다.
이 파일(run_WEB_gui.bat)은 **같은 백엔드/같은 vue_bridge**를 재사용하되,
A1111처럼 localhost 포트를 열어 일반 브라우저(폰/타 PC 포함)로 접속하게 한다.

핵심: QWebChannel은 transport 교체가 가능하다. 창 모드는 '동일 프로세스 직통선',
웹 모드는 'WebSocket'을 transport로 쓸 뿐, vue_bridge의 슬롯/시그널/액션 디스패치는
그대로 재사용된다 (재작성 없음). 전부 localhost/LAN이라 인터넷은 필요 없다.

구성:
  - GeneratorMainUI 를 (창을 띄우지 않고) 생성 → 모든 백엔드 로직/프록시가 살아있음
  - QWebSocketServer + WebSocketChannelTransport 로 vue_bridge 를 'backend'로 publish
  - frontend_dist 를 HTTP 정적 서빙(index.html 에 WS 포트 주입)
  - 기본 브라우저 자동 오픈
"""
import sys
import os
import json
import hashlib
import hmac
import secrets
import shutil
import threading
import functools
import mimetypes
from email.utils import formatdate, parsedate_to_datetime
from http.cookies import SimpleCookie
from urllib.parse import urlparse, parse_qs, urlencode, urlunparse
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer


def _acquire_detached_update_gate(timeout_ms: int = 120_000):
    """Keep direct web-mode starts away from a checkout being replaced."""

    if os.name != "nt":
        return None
    import ctypes

    kernel32 = ctypes.windll.kernel32
    kernel32.CreateMutexW.restype = ctypes.c_void_p
    kernel32.CreateMutexW.argtypes = (ctypes.c_void_p, ctypes.c_bool, ctypes.c_wchar_p)
    kernel32.WaitForSingleObject.restype = ctypes.c_ulong
    kernel32.WaitForSingleObject.argtypes = (ctypes.c_void_p, ctypes.c_ulong)
    kernel32.ReleaseMutex.argtypes = (ctypes.c_void_p,)
    kernel32.CloseHandle.argtypes = (ctypes.c_void_p,)
    handle = kernel32.CreateMutexW(None, False, r"Local\AIStudioPro.UR_IV.Update")
    if not handle:
        raise RuntimeError("앱 업데이트 시작 잠금을 확인하지 못했습니다.")
    result = int(kernel32.WaitForSingleObject(handle, timeout_ms))
    if result not in {0x00000000, 0x00000080}:
        kernel32.CloseHandle(handle)
        raise RuntimeError("앱 업데이트가 아직 진행 중입니다. 잠시 후 다시 실행하세요.")
    return kernel32, handle


_startup_update_gate = _acquire_detached_update_gate() if __name__ == "__main__" else None
try:
    if __name__ == "__main__":
        from core.app_instance import register_app_instance

        register_app_instance(update_guarded=_startup_update_gate is not None)
finally:
    if _startup_update_gate is not None:
        _startup_update_gate[0].ReleaseMutex(_startup_update_gate[1])
        _startup_update_gate[0].CloseHandle(_startup_update_gate[1])

if __name__ == "__main__":
    from core.app_startup import prepare_application

    _prepare_result = prepare_application()
    if _prepare_result:
        from core.app_instance import unregister_app_instance

        unregister_app_instance()
        raise SystemExit(_prepare_result)

os.environ.setdefault(
    "QTWEBENGINE_CHROMIUM_FLAGS",
    "--disable-logging --log-level=3 --disable-features=WebRtcHideLocalIpsWithMdns",
)
os.environ.setdefault("QT_LOGGING_RULES", "qt.webenginecontext.debug=false")

from config import *  # noqa: F401,F403  (OUTPUT_DIR 등)
from ui.generator_main import GeneratorMainUI

from PyQt6.QtWidgets import QApplication
from PyQt6.QtCore import Qt, QObject, pyqtSignal, pyqtSlot, QEventLoop, QTimer, QJsonDocument
from PyQt6.QtWebChannel import QWebChannel, QWebChannelAbstractTransport
from PyQt6.QtWebSockets import QWebSocketServer, QWebSocketProtocol
from PyQt6.QtNetwork import QHostAddress

from core.studio_application import CallContext
from ui.studio_qwebchannel import StudioQWebChannelAdapter

# ── 포트 설정 (환경변수로 덮어쓰기 가능) ──
HTTP_PORT = int(os.environ.get("AISTUDIO_HTTP_PORT", "7800"))
WS_PORT = int(os.environ.get("AISTUDIO_WS_PORT", "7801"))
# 안전한 기본값은 이 PC에서만 접근 가능한 loopback. LAN 공유는 사용자가 명시적으로
# AISTUDIO_BIND=0.0.0.0 을 지정한 경우에만 허용한다.
BIND_HOST = os.environ.get("AISTUDIO_BIND", "127.0.0.1").strip() or "127.0.0.1"

# 매 실행마다 새로 생성되는 bearer 세션. 디스크/설정 파일에는 절대 저장하지 않는다.
SESSION_TOKEN = secrets.token_urlsafe(32)
SESSION_COOKIE = "aistudio_session"

_ROOT = os.path.dirname(os.path.abspath(__file__))
_DIST = os.path.join(_ROOT, "frontend_dist")
_THUMB_DIR = os.path.join(_ROOT, "image_cache", "web_thumbs")


def _token_matches(candidate: str | None) -> bool:
    """타이밍 차이를 줄여 세션 토큰을 비교한다."""
    return bool(candidate) and hmac.compare_digest(str(candidate), SESSION_TOKEN)


def _cookie_token(raw_cookie: str) -> str:
    try:
        cookie = SimpleCookie()
        cookie.load(raw_cookie or "")
        morsel = cookie.get(SESSION_COOKIE)
        return morsel.value if morsel else ""
    except Exception:
        return ""


def _origin_matches(origin_text: str, request_host: str) -> bool:
    """브라우저 Origin이 현재 HTTP 앱과 같은 호스트/포트인지 검사한다."""
    try:
        origin = urlparse(origin_text)
        expected_host = (request_host or "").lower().rstrip(".")
        origin_host = (origin.hostname or "").lower().rstrip(".")
        origin_port = origin.port or (443 if origin.scheme == "https" else 80)
        return (
            origin.scheme == "http"
            and origin_port == HTTP_PORT
            and bool(expected_host)
            and origin_host == expected_host
        )
    except (TypeError, ValueError):
        return False


# 웹 브라우저가 실제로 사용하는 메서드/시그널만 네트워크 QWebChannel에 공개한다.
# 새 프론트 기능이 슬롯을 추가하면 이 목록도 함께 갱신해야 한다.
_WEB_METHODS = frozenset({
    "onWidgetChanged", "onAction", "getWidgetValue", "getAllWidgetValues",
    "editorProcess", "getLastGalleryFolder", "getGalleryImages",
    "requestGalleryImages", "getFavorites", "generateThumbnails",
    "searchDanbooru", "loadLastSearchResults", "loadFullResults",
    "getActiveSearchDataset", "getUiPrefs",
    "getAiAssistInstructions", "saveAiAssistInstructions",
    "getInstructionPresets", "saveInstructionPreset", "deleteInstructionPreset",
    # Backend runtime, generation API, model-path 설정은 redaction과 native
    # capability 검사를 한곳에서 강제하는 ``studio`` 객체로만 공개한다.
    "getUpscalers", "requestUpscalers", "saveImageExif", "renameFile",
    "getEdgeMap", "ollamaEnhance", "convertPromptToNl", "editorPasteImage",
    "editorAutoSave", "editorCheckAutoSave", "editorClearAutoSave", "getFileInfo",
    "ollamaListModels", "requestOllamaModels", "getRandomResolutions",
    "getInitialConfig", "getGenStats", "getWildcardTree", "getPresetList",
    "getPresetData", "searchCharacters", "getCharacterFeatures",
    "getCharacterCopyright", "fetchCharacterTagsOnline", "separateTags", "pairColors",
    "refineToSpecificTags", "getLoras", "saveSession", "getSession",
    "getClothingRegions", "saveCharacterPreset", "deleteCharacterPreset",
    "getCharGlobalPrefs", "saveCharGlobalPrefs", "applyCharacterPreset",
    "getDeckCharacters", "submitABTest", "getCharFeatureOverride",
    "setCharFeatureOverride", "saveWildcard", "deleteWildcard", "renameWildcard",
    "getExcludeMatches", "deepCleanPrompt", "getCharacterInsight", "classifyTags",
    "exportCompareGif", "getTabDefaults", "getADetailerModels",
    "requestADetailerModels", "getYoloModelLabel", "refreshYoloModels",
    "getTagSuggestions", "generateXYZCombinations", "captionImage",
    "startCaptionBatch", "requestCaptionRuntime", "getCaptionJobStatus",
    "loadCaption", "saveCaption", "getImageExif",
    # sam-extra 임베드 LoRA Manager 주소 조회 (워크플로 4)
    "requestLoraManagerUrl",
})

_WEB_SIGNALS = frozenset({
    "imageGenerated", "generationStarted", "generationError", "generationProgress", "generationPreview",
    "editorImageLoaded", "editorResult", "captionFilesSelected", "captionProgress",
    "captionDone", "captionOutDirSelected", "captionModelDirSelected", "captionRuntimeReady",
    "i2iImageLoaded", "galleryFolderLoaded",
    "inpaintImageLoaded", "searchStatus", "searchResultLineage", "loraInserted", "loraStackLoaded",
    "yoloModelUpdated", "condRulesLoaded", "batchFilesSelected", "ollamaResult",
    "genNlResult", "globalWeightsLoaded", "uiPrefsLoaded", "compareImageLoaded",
    "galleryImagesReady", "thumbnailReady", "upscalersReady", "ollamaModelsReady",
    "chatToken", "chatDone", "chatThreads", "chatGenerationEvent", "chatModelInfo",
    "chatModelsReady", "aiAssistInstructionsChanged", "instructionPresetsChanged",
    "xyzCapabilitiesReceived", "xyzPlotEvent",
    "adetailerModelsReady", "queueUpdated", "queueItemAdded", "queueCompleted",
    "showNotification", "adetailerResult", "adetailerProgress", "sam3Result",
    "sam3Progress", "eventSearchProgress", "eventSearchResults", "eventImportResults",
    # automationStatus 는 prompt·paused 필드가 늘어도 그대로 통과한다(JSON 통째 전달).
    # 새 액션(automation_override_next / pause_automation / resume_automation)은
    # onAction 이 이미 _WEB_METHODS 에 있어 별도 등록이 필요 없다 — 액션은 이름별로
    # 막지 않는다. 여기 목록은 '시그널'만 막는다.
    "automationStatus", "automationSettingsLoaded", "instantWildcardsList",
    "promptOrderLoaded", "workflowProfilesList", "widgetValueChanged",
    "widgetPropertyChanged", "batchUpdate", "tabChanged", "vramUpdated",
    "creatorStateChanged", "creatorProgress", "creatorResult", "creatorCacheEvent",
    "creatorMediaSelected", "comicStoryboardReady", "comicDocumentChanged",
    # SAM3 Refine (sam-extra 워크플로 2) + 임베드 LoRA Manager (워크플로 4)
    "refineResult", "loraManagerUrlReady",
    "editorWatermarkImageLoaded",
    # 태그 검색 결과 — 누락되면 웹 모드에서 결과가 영영 도착하지 않는다.
    "searchResultsReady",
    # 백엔드 선택 게이트 4종. 액션(probe_backend/select_backend/pick_comfy_workflow)은
    # 이미 onAction 으로 열려 있으니 여기 빠지면 '요청은 가는데 답이 안 오는' 반쪽이
    # 된다 — 게이트가 backendSelected 로만 닫히므로 웹 모드에서 영영 안 닫힌다.
    "backendSelectionRequired", "backendProbeResult", "backendSelected",
    "comfyWorkflowPicked",
    # 하단 계기 스트립의 백엔드 칸. 빠지면 웹 모드에서만 스트립이 영영 비어 있고
    # 데스크톱은 멀쩡해 눈치채기 어렵다(게이트 4종과 같은 함정).
    "backendStatus",
})


class WebBridgeFacade(QObject):
    """인증된 웹 클라이언트에만 노출하는 최소 QWebChannel 표면."""

    event = pyqtSignal(str, str)  # (signal_name, JSON args)

    def __init__(self, bridge, parent=None):
        super().__init__(parent)
        self._bridge = bridge
        self._signal_links = []
        for name in sorted(_WEB_SIGNALS):
            signal = getattr(bridge, name, None)
            if signal is None or not hasattr(signal, "connect"):
                continue
            handler = functools.partial(self._forward_event, name)
            signal.connect(handler)
            self._signal_links.append((signal, handler))

    def _forward_event(self, name, *args):
        try:
            self.event.emit(name, json.dumps(list(args), ensure_ascii=False))
        except Exception as e:
            print(f"[web] event 직렬화 실패({name}): {e}")

    @pyqtSlot(result=str)
    def getCapabilities(self) -> str:
        return json.dumps({
            "methods": sorted(_WEB_METHODS),
            "signals": sorted(name for name in _WEB_SIGNALS if hasattr(self._bridge, name)),
        })

    @pyqtSlot(str, str, result=str)
    def invoke(self, method: str, args_json: str) -> str:
        if method not in _WEB_METHODS:
            return json.dumps({"ok": False, "error": "method not allowed"})
        target = getattr(self._bridge, method, None)
        if not callable(target):
            return json.dumps({"ok": False, "error": "method unavailable"})
        try:
            args = json.loads(args_json) if args_json else []
            if not isinstance(args, list):
                raise ValueError("arguments must be a JSON array")
            value = target(*args)
            return json.dumps({"ok": True, "value": value}, ensure_ascii=False)
        except Exception as e:
            print(f"[web] bridge 호출 실패({method}): {e}")
            return json.dumps({"ok": False, "error": str(e)}, ensure_ascii=False)


# ──────────────────────────────────────────────────────────────────────────
# QWebChannel ↔ WebSocket transport (Qt 공식 chat 예제의 PyQt 포팅)
# ──────────────────────────────────────────────────────────────────────────
class WebSocketTransport(QWebChannelAbstractTransport):
    """단일 QWebSocket 연결을 QWebChannel transport로 감싼다.

    Qt 공식 'qwebchannel/standalone' 예제의 PyQt 포팅. 메시지를 dict가 아니라
    QJsonDocument/QJsonObject로 주고받아야 핸드셰이크가 완료된다 (dict 직접 emit은
    버전에 따라 QJsonObject 변환이 어긋나 채널 초기화가 멈출 수 있음).
    """

    def __init__(self, socket):
        super().__init__(socket)
        self._socket = socket
        self._socket.textMessageReceived.connect(self._on_text)

    @pyqtSlot(str)
    def _on_text(self, message):
        doc = QJsonDocument.fromJson(message.encode("utf-8"))
        if doc.isNull() or not doc.isObject():
            return
        # QWebChannel 로 실제 QJsonObject 를 전달
        self.messageReceived.emit(doc.object(), self)

    def sendMessage(self, message):
        # message: QJsonObject(dict 로 도착) → 컴팩트 JSON 텍스트로 송신
        try:
            doc = QJsonDocument(message)
            text = bytes(doc.toJson(QJsonDocument.JsonFormat.Compact)).decode("utf-8")
            self._socket.sendTextMessage(text)
        except Exception as e:
            print(f"[web] sendMessage 실패: {e}")


class WebChannelServer(QObject):
    """WebSocket 서버를 띄우고 새 연결마다 channel에 transport를 연결한다."""

    def __init__(self, channel, host, port, on_first_client=None):
        super().__init__()
        self._channel = channel
        self._on_first_client = on_first_client
        self._had_client = False
        self._connections = []  # [(socket, transport)]
        self.server = QWebSocketServer(
            "AIStudioPro", QWebSocketServer.SslMode.NonSecureMode
        )
        addr = QHostAddress.SpecialAddress.Any if host in ("0.0.0.0", "") else QHostAddress(host)
        if not self.server.listen(addr, port):
            raise RuntimeError(
                f"WebSocket 서버가 포트 {port} 에서 listen 실패: {self.server.errorString()}"
            )
        self.server.newConnection.connect(self._on_new_connection)
        print(f"[web] WebChannel(WebSocket) listening on ws://{host}:{port}")

    def _on_new_connection(self):
        socket = self.server.nextPendingConnection()
        if not self._is_authorized(socket):
            peer = socket.peerAddress().toString()
            print(f"[web] WebSocket 인증 거부: {peer}")
            socket.disconnected.connect(socket.deleteLater)
            socket.close(
                QWebSocketProtocol.CloseCode.CloseCodePolicyViolated,
                "Unauthorized",
            )
            return

        transport = WebSocketTransport(socket)
        self._connections.append((socket, transport))
        self._channel.connectTo(transport)
        socket.disconnected.connect(
            lambda s=socket, t=transport: self._on_disconnect(s, t)
        )
        print("[web] 브라우저 연결됨")
        if not self._had_client:
            self._had_client = True
            if self._on_first_client:
                self._on_first_client()

    def _is_authorized(self, socket) -> bool:
        """세션 쿠키와 동일 출처 Origin을 모두 만족해야 연결한다."""
        try:
            request = socket.request()
            raw_cookie = bytes(request.rawHeader(b"Cookie")).decode("latin-1")
            if not _token_matches(_cookie_token(raw_cookie)):
                return False

            request_host = (socket.requestUrl().host() or "").lower().rstrip(".")
            return _origin_matches(socket.origin(), request_host)
        except Exception as e:
            print(f"[web] WebSocket 인증 검사 실패: {e}")
            return False

    def _on_disconnect(self, socket, transport):
        try:
            self._channel.disconnectFrom(transport)
        except Exception:
            pass
        try:
            self._connections.remove((socket, transport))
        except ValueError:
            pass
        transport.deleteLater()
        socket.deleteLater()
        print("[web] 브라우저 연결 종료")

    def close(self):
        """앱 종료 시 채널 등록과 소켓을 순서대로 정리한다."""
        self.server.close()
        for socket, transport in list(self._connections):
            try:
                self._channel.disconnectFrom(transport)
            except Exception:
                pass
            try:
                socket.close(QWebSocketProtocol.CloseCode.CloseCodeGoingAway, "Server shutdown")
            except Exception:
                pass
            transport.deleteLater()
            socket.deleteLater()
        self._connections.clear()


# ──────────────────────────────────────────────────────────────────────────
# 정적 HTTP 서버 (frontend_dist) — index.html 에 WS 포트 주입
# ──────────────────────────────────────────────────────────────────────────
class _DistHandler(SimpleHTTPRequestHandler):
    """인증된 브라우저에만 SPA와 로컬 이미지를 제공한다."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=_DIST, **kwargs)

    def log_message(self, *args):  # 콘솔 스팸 억제
        pass

    def end_headers(self):
        # 토큰 URL이나 로컬 파일 경로가 외부 페이지의 Referer로 새지 않게 한다.
        self.send_header("Referrer-Policy", "no-referrer")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("X-Frame-Options", "DENY")
        self.send_header("Cross-Origin-Opener-Policy", "same-origin")
        self.send_header(
            "Content-Security-Policy",
            "default-src 'self'; script-src 'self'; style-src 'self' 'unsafe-inline'; "
            "img-src 'self' data: blob:; media-src 'self' blob:; font-src 'self' data:; "
            "connect-src 'self' ws:; object-src 'none'; frame-ancestors 'none'; "
            "base-uri 'none'; form-action 'self'",
        )
        super().end_headers()

    def _require_auth(self) -> bool:
        """최초 token URL은 쿠키를 발급하고 토큰 없는 URL로 즉시 정리한다."""
        parsed = urlparse(self.path)
        query = parse_qs(parsed.query, keep_blank_values=True)
        query_token = (query.get("token") or [""])[0]
        if _token_matches(query_token):
            query.pop("token", None)
            clean = urlunparse(("", "", parsed.path or "/", "", urlencode(query, doseq=True), ""))
            self.send_response(303)
            self.send_header("Location", clean or "/")
            self.send_header(
                "Set-Cookie",
                f"{SESSION_COOKIE}={SESSION_TOKEN}; Path=/; HttpOnly; SameSite=Strict",
            )
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            return False

        if _token_matches(_cookie_token(self.headers.get("Cookie", ""))):
            return True

        body = b"AI Studio Pro web session is not authorized."
        self.send_response(401)
        self.send_header("Content-Type", "text/plain; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        if self.command != "HEAD":
            self.wfile.write(body)
        return False

    def _serve_index(self, head_only=False):
        index_path = os.path.join(_DIST, "index.html")
        try:
            with open(index_path, "r", encoding="utf-8") as f:
                html = f.read()
        except OSError:
            self.send_error(404, "frontend_dist/index.html 없음 — 'npm run build' 먼저")
            return
        # 런타임 설정을 외부 스크립트로 주입해 CSP의 unsafe-inline script를 피한다.
        inject = '<script src="/runtime-config.js"></script>'
        if "</head>" in html:
            html = html.replace("</head>", inject + "</head>", 1)
        else:
            html = inject + html
        body = html.encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        if not head_only:
            self.wfile.write(body)

    def _serve_runtime_config(self, head_only=False):
        body = f"window.__AISTUDIO_WS_PORT__={WS_PORT};\n".encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/javascript; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        if not head_only:
            self.wfile.write(body)

    def _safe_image_from_query(self):
        try:
            from core.path_safety import safe_input_path
        except Exception as e:
            print(f"[web] path_safety 로드 실패: {e}")
            return None
        query = parse_qs(urlparse(self.path).query)
        raw = (query.get("path") or [""])[0]
        return safe_input_path(raw)

    def _not_modified(self, stat, etag) -> bool:
        if self.headers.get("If-None-Match") == etag:
            return True
        since = self.headers.get("If-Modified-Since")
        if not since:
            return False
        try:
            dt = parsedate_to_datetime(since)
            return stat.st_mtime <= dt.timestamp() + 1
        except (TypeError, ValueError, OverflowError):
            return False

    def _stream_file(self, path, *, content_type=None, head_only=False, cache_seconds=60):
        try:
            stat = os.stat(path)
            stream = open(path, "rb")
        except OSError:
            self.send_error(404, "read error")
            return

        with stream:
            etag = f'"{stat.st_mtime_ns:x}-{stat.st_size:x}"'
            if self._not_modified(stat, etag):
                self.send_response(304)
                self.send_header("ETag", etag)
                self.send_header("Last-Modified", formatdate(stat.st_mtime, usegmt=True))
                self.send_header("Cache-Control", f"private, max-age={cache_seconds}, must-revalidate")
                self.end_headers()
                return

            self.send_response(200)
            self.send_header("Content-Type", content_type or mimetypes.guess_type(path)[0] or "application/octet-stream")
            self.send_header("Content-Length", str(stat.st_size))
            self.send_header("ETag", etag)
            self.send_header("Last-Modified", formatdate(stat.st_mtime, usegmt=True))
            self.send_header("Cache-Control", f"private, max-age={cache_seconds}, must-revalidate")
            self.end_headers()
            if not head_only:
                try:
                    shutil.copyfileobj(stream, self.wfile, length=1024 * 1024)
                except (BrokenPipeError, ConnectionResetError):
                    pass

    def _serve_local_file(self, head_only=False):
        """`/file?path=<로컬경로>` — 로컬 이미지 파일을 HTTP 로 서빙.

        원격 브라우저는 file:/// 를 못 읽으므로 프론트 mediaUrl() 이 이 경로로 요청한다.
        인증 쿠키와 safe_input_path를 모두 통과한 이미지만 스트리밍한다.
        """
        safe = self._safe_image_from_query()
        if not safe or not os.path.isfile(safe):
            self.send_error(404, "file not found or not allowed")
            return
        self._stream_file(safe, head_only=head_only, cache_seconds=60)

    def _serve_thumbnail(self, head_only=False):
        """갤러리 카드용 축소 이미지를 디스크 캐시에 생성해 전송량을 줄인다."""
        safe = self._safe_image_from_query()
        if not safe:
            self.send_error(404, "file not found or not allowed")
            return
        query = parse_qs(urlparse(self.path).query)
        try:
            width = max(64, min(1024, int((query.get("width") or ["384"])[0])))
        except (TypeError, ValueError):
            width = 384

        os.makedirs(_THUMB_DIR, exist_ok=True)
        key = hashlib.sha256(f"{os.path.normcase(safe)}@{width}".encode("utf-8")).hexdigest()
        thumb = os.path.join(_THUMB_DIR, f"{key}.jpg")
        try:
            source_stat = os.stat(safe)
            stale = not os.path.isfile(thumb) or os.stat(thumb).st_mtime_ns < source_stat.st_mtime_ns
            if stale:
                from PIL import Image, ImageOps
                tmp = f"{thumb}.{threading.get_ident()}.tmp"
                try:
                    with Image.open(safe) as image:
                        image = ImageOps.exif_transpose(image)
                        image.thumbnail((width, width), Image.Resampling.LANCZOS)
                        if image.mode in ("RGBA", "LA"):
                            canvas = Image.new("RGB", image.size, (13, 13, 13))
                            canvas.paste(image, mask=image.getchannel("A"))
                            image = canvas
                        else:
                            image = image.convert("RGB")
                        image.save(tmp, "JPEG", quality=86, optimize=True)
                    os.replace(tmp, thumb)
                finally:
                    try:
                        if os.path.exists(tmp):
                            os.remove(tmp)
                    except OSError:
                        pass
        except Exception as e:
            print(f"[web] 썸네일 생성 실패: {e}")
            self.send_error(500, "thumbnail error")
            return
        self._stream_file(thumb, content_type="image/jpeg", head_only=head_only, cache_seconds=3600)

    def do_GET(self):
        if not self._require_auth():
            return
        path = self.path.split("?", 1)[0]
        if path in ("/", "/index.html"):
            self._serve_index()
            return
        if path == "/runtime-config.js":
            self._serve_runtime_config()
            return
        if path == "/file":
            self._serve_local_file()
            return
        if path == "/thumbnail":
            self._serve_thumbnail()
            return
        super().do_GET()

    def do_HEAD(self):
        if not self._require_auth():
            return
        path = self.path.split("?", 1)[0]
        if path in ("/", "/index.html"):
            self._serve_index(head_only=True)
            return
        if path == "/runtime-config.js":
            self._serve_runtime_config(head_only=True)
            return
        if path == "/file":
            self._serve_local_file(head_only=True)
            return
        if path == "/thumbnail":
            self._serve_thumbnail(head_only=True)
            return
        super().do_HEAD()


def _start_http_server():
    httpd = ThreadingHTTPServer((BIND_HOST, HTTP_PORT), _DistHandler)
    httpd.daemon_threads = True
    t = threading.Thread(target=httpd.serve_forever, daemon=True)
    t.start()
    print(f"[web] HTTP 정적 서버: http://{BIND_HOST}:{HTTP_PORT}")
    return httpd


# ──────────────────────────────────────────────────────────────────────────
# 웹 모드 시작 시퀀스 — 임베드 Vue 로드/대기는 건너뛰고 백엔드만 준비
# ──────────────────────────────────────────────────────────────────────────
def _web_startup(window, app, web_server):
    """창 모드의 _run_startup_sequence 에서 '임베드 뷰 로드/대기'만 제외한 버전.

    순서: 백엔드 선택(호스트에서 1회) → 서버 기동 → 브라우저 접속 대기 →
          접속되면 백엔드 적용(데이터 push). push 시점에 클라이언트가 붙어 있어야
          sticky 가 아닌 이벤트(모델 목록 등)도 유실 없이 도달한다.
    """
    # 1. 백엔드 선택 (호스트 화면에 1회 — 창 모드와 동일 로직)
    if hasattr(window, "_startup_backend_check"):
        window._startup_backend_check()

    # 2. 첫 브라우저 접속 대기 (최대 60초) — 접속 전이면 push 가 유실되므로
    loop = QEventLoop()
    if web_server._had_client:
        pass
    else:
        web_server._on_first_client = loop.quit
        QTimer.singleShot(60000, loop.quit)  # 안전 타임아웃
        print("[web] 브라우저 접속 대기 중…")
        loop.exec()

    # 3. 백엔드 연결 + 모델/샘플러/LoRA → Vue(브라우저)로 push
    try:
        if hasattr(window, "_apply_backend_startup_result"):
            window._apply_backend_startup_result()
    except Exception as e:
        print(f"[web] 백엔드 적용 실패(무시): {e}")
    try:
        if hasattr(window, "_restore_search_deck"):
            QTimer.singleShot(600, window._restore_search_deck)
    except Exception:
        pass


def main():
    if sys.platform == "win32":
        import ctypes
        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(
            "mycompany.myproduct.subproduct.version"
        )

    QApplication.setHighDpiScaleFactorRoundingPolicy(
        Qt.HighDpiScaleFactorRoundingPolicy.PassThrough
    )
    app = QApplication(sys.argv)
    app.setApplicationName("AI Studio Pro (Web)")
    app.setOrganizationName("AI Studio")
    app.setStyleSheet("")

    os.makedirs(OUTPUT_DIR, exist_ok=True)  # noqa: F405

    if not os.path.exists(os.path.join(_DIST, "index.html")):
        print("[web] frontend_dist 가 없습니다. 먼저 'cd frontend && npm run build' 하세요.")
        sys.exit(1)

    # 백엔드 윈도우 생성 (창은 띄우지 않음 — 로직/프록시/vue_bridge 만 사용)
    window = GeneratorMainUI()
    # 웹 모드 표시 — 저장된 창 기하 복원(showMaximized 등)이 호스트 창을 띄우지 않게.
    window.web_mode = True

    # WebChannel + WebSocket 서버 — 네트워크에는 허용 목록 façade만 공개
    channel = QWebChannel()
    web_bridge = WebBridgeFacade(window.vue_bridge, channel)
    web_studio_transport = StudioQWebChannelAdapter(
        window.studio_application,
        CallContext(
            principal_id="web-ui",
            transport="qwebchannel-websocket",
            capabilities=frozenset(),
        ),
        channel,
    )
    channel.registerObject("backend", web_bridge)
    channel.registerObject("studio", web_studio_transport)
    # QWebChannel owns the QObject too, but explicit Python references make the
    # lifetime contract clear and prevent wrapper collection during long jobs.
    window.web_studio_transport = web_studio_transport
    web_server = WebChannelServer(channel, BIND_HOST, WS_PORT)

    # HTTP 정적 서버 + 브라우저 오픈
    http_server = _start_http_server()
    # 정적 UI 포트를 먼저 점유한다. generation API 설정이 같은 포트를
    # 가리키더라도 웹 UI가 죽지 않고 API 시작만 안전하게 거부된다.
    generation_api_manager = None
    try:
        from core.generation_api import get_generation_api_manager

        generation_api_manager = get_generation_api_manager()
        generation_api_manager.start_if_enabled()
    except Exception as exc:
        print(f"[Generation API] startup skipped: {exc}")
    browser_host = "127.0.0.1" if BIND_HOST in ("0.0.0.0", "::") else BIND_HOST
    url = f"http://{browser_host}:{HTTP_PORT}/?token={SESSION_TOKEN}"
    print("=" * 60)
    print(f"[web] 인증된 브라우저 URL:  {url}")
    if BIND_HOST in ("0.0.0.0", "::"):
        print(
            f"[web] LAN 기기 URL:  http://<이 PC의 IP>:{HTTP_PORT}/?token={SESSION_TOKEN}"
        )
    else:
        print("[web] LAN 공유는 AISTUDIO_BIND=0.0.0.0 을 명시해야 활성화됩니다.")
    print("=" * 60)
    try:
        import webbrowser
        webbrowser.open(url)
    except Exception:
        pass

    # 백엔드 준비 (브라우저 접속 후 데이터 push)
    QTimer.singleShot(0, functools.partial(_web_startup, window, app, web_server))

    def _shutdown_servers():
        from core.app_instance import unregister_app_instance

        unregister_app_instance()
        if generation_api_manager is not None:
            generation_api_manager.shutdown()
        web_server.close()
        http_server.shutdown()
        http_server.server_close()

    app.aboutToQuit.connect(_shutdown_servers)

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
