"""웹 모드 인증, 파일 전송, 제한 브릿지 계약 회귀 테스트."""

from __future__ import annotations

import http.client
import json
import os
import pathlib
import re
import tempfile
import threading
import unittest
from urllib.parse import quote
from unittest import mock

from PIL import Image

try:
    import web_main_ui
    from ui.vue_bridge import VueBridge
    _WEB_IMPORT_ERROR = ""
except ModuleNotFoundError as exc:
    if not (exc.name or "").startswith("PyQt6"):
        raise
    web_main_ui = None
    VueBridge = None
    _WEB_IMPORT_ERROR = str(exc)


@unittest.skipIf(web_main_ui is None, f"PyQt6 WebEngine unavailable: {_WEB_IMPORT_ERROR}")
class TestWebModeSecurity(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp_dir = tempfile.TemporaryDirectory()
        cls.thumb_patch = mock.patch.object(web_main_ui, "_THUMB_DIR", cls.temp_dir.name)
        cls.thumb_patch.start()
        cls.image_path = os.path.join(cls.temp_dir.name, "source.png")
        Image.new("RGB", (320, 160), (20, 40, 80)).save(cls.image_path)

        cls.server = web_main_ui.ThreadingHTTPServer(
            ("127.0.0.1", 0), web_main_ui._DistHandler
        )
        cls.server.daemon_threads = True
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()
        cls.port = cls.server.server_address[1]

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()
        cls.thread.join(timeout=2)
        cls.thumb_patch.stop()
        cls.temp_dir.cleanup()

    def _request(self, path, *, headers=None, method="GET"):
        conn = http.client.HTTPConnection("127.0.0.1", self.port, timeout=5)
        conn.request(method, path, headers=headers or {})
        response = conn.getresponse()
        body = response.read()
        result = response.status, dict(response.getheaders()), body
        conn.close()
        return result

    def _authenticated_cookie(self):
        status, headers, _body = self._request(
            f"/?token={quote(web_main_ui.SESSION_TOKEN)}"
        )
        self.assertEqual(status, 303)
        self.assertNotIn("token=", headers["Location"])
        return headers["Set-Cookie"].split(";", 1)[0]

    def test_http_requires_session_and_cleans_token_url(self):
        status, _headers, _body = self._request("/")
        self.assertEqual(status, 401)

        cookie = self._authenticated_cookie()
        status, headers, body = self._request(
            "/runtime-config.js", headers={"Cookie": cookie}
        )
        self.assertEqual(status, 200)
        self.assertIn(b"__AISTUDIO_WS_PORT__", body)
        self.assertEqual(headers["Referrer-Policy"], "no-referrer")

    def test_file_streaming_cache_and_thumbnail(self):
        cookie = self._authenticated_cookie()
        encoded = quote(self.image_path)
        status, headers, body = self._request(
            f"/file?path={encoded}", headers={"Cookie": cookie}
        )
        self.assertEqual(status, 200)
        self.assertEqual(len(body), os.path.getsize(self.image_path))
        self.assertIn("ETag", headers)
        self.assertIn("Last-Modified", headers)

        status, _headers, body = self._request(
            f"/file?path={encoded}",
            headers={"Cookie": cookie, "If-None-Match": headers["ETag"]},
        )
        self.assertEqual(status, 304)
        self.assertEqual(body, b"")

        status, headers, body = self._request(
            f"/thumbnail?path={encoded}&width=96", headers={"Cookie": cookie}
        )
        self.assertEqual(status, 200)
        self.assertEqual(headers["Content-Type"], "image/jpeg")
        self.assertGreater(len(body), 0)

    def test_file_overwritten_within_a_second_is_not_revalidated_as_unchanged(self):
        """ETag 가 다르면 If-Modified-Since 가 '안 바뀜'이어도 새 본문을 준다(RFC 9110 §13.1.3).

        예전에는 If-None-Match 가 어긋나도 If-Modified-Since(1초 정밀도 + 1초 여유)로 넘어가, 같은 초 안에
        덮어쓴 파일(에디터 재저장 등)에 304 를 돌려줘 브라우저가 옛 그림을 계속 썼다.
        """
        cookie = self._authenticated_cookie()
        path = os.path.join(self.temp_dir.name, "overwritten.png")
        Image.new("RGB", (8, 8), (1, 2, 3)).save(path)
        encoded = quote(path)
        status, first, old_body = self._request(f"/file?path={encoded}", headers={"Cookie": cookie})
        self.assertEqual(status, 200)

        mtime = os.stat(path).st_mtime
        Image.new("RGB", (16, 16), (200, 100, 50)).save(path)   # 다른 크기로 제자리 덮어쓰기
        os.utime(path, (mtime, mtime))                           # 같은 초 안에 쓴 것처럼 — Last-Modified 가 같다
        new_size = os.path.getsize(path)
        self.assertNotEqual(new_size, len(old_body))

        revalidate = {"Cookie": cookie, "If-None-Match": first["ETag"], "If-Modified-Since": first["Last-Modified"]}
        status, headers, body = self._request(f"/file?path={encoded}", headers=revalidate)
        self.assertEqual(status, 200, "ETag 가 바뀐 파일에 304 — 브라우저가 옛 그림을 쓴다")
        self.assertEqual(len(body), new_size)
        self.assertNotEqual(headers["ETag"], first["ETag"])

        # 맞는 ETag 는 여전히 304 — If-Modified-Since 가 없어도, 목록·약한 비교로 와도
        for inm in (headers["ETag"], f'"other", {headers["ETag"]}', f"W/{headers['ETag']}"):
            status, _headers, body = self._request(
                f"/file?path={encoded}", headers={"Cookie": cookie, "If-None-Match": inm}
            )
            self.assertEqual(status, 304, inm)
            self.assertEqual(body, b"")
        # If-None-Match 없이 If-Modified-Since 만 오면 예전처럼 날짜로 판단한다
        status, _headers, _body = self._request(
            f"/file?path={encoded}", headers={"Cookie": cookie, "If-Modified-Since": headers["Last-Modified"]}
        )
        self.assertEqual(status, 304)

    def test_origin_and_cookie_validation(self):
        self.assertTrue(web_main_ui._token_matches(web_main_ui.SESSION_TOKEN))
        self.assertFalse(web_main_ui._token_matches("wrong"))
        self.assertEqual(
            web_main_ui._cookie_token(
                f"other=x; {web_main_ui.SESSION_COOKIE}={web_main_ui.SESSION_TOKEN}"
            ),
            web_main_ui.SESSION_TOKEN,
        )
        self.assertTrue(
            web_main_ui._origin_matches(
                f"http://127.0.0.1:{web_main_ui.HTTP_PORT}", "127.0.0.1"
            )
        )
        self.assertFalse(web_main_ui._origin_matches("https://evil.example", "127.0.0.1"))

    def test_websocket_requires_cookie_and_matching_origin(self):
        from PyQt6.QtCore import QCoreApplication, QEventLoop, QTimer, QUrl
        from PyQt6.QtNetwork import QNetworkRequest
        from PyQt6.QtWebChannel import QWebChannel
        from PyQt6.QtWebSockets import QWebSocket

        app = QCoreApplication.instance() or QCoreApplication([])
        bridge = VueBridge()
        channel = QWebChannel()
        facade = web_main_ui.WebBridgeFacade(bridge, channel)
        channel.registerObject("backend", facade)
        server = web_main_ui.WebChannelServer(channel, "127.0.0.1", 0)
        port = server.server.serverPort()

        def wait_for(signal, action, timeout=2000):
            loop = QEventLoop()
            fired = []
            signal.connect(lambda: (fired.append(True), loop.quit()))
            QTimer.singleShot(timeout, loop.quit)
            action()
            loop.exec()
            return bool(fired)

        origin = f"http://127.0.0.1:{web_main_ui.HTTP_PORT}"
        good = QWebSocket(origin)
        request = QNetworkRequest(QUrl(f"ws://127.0.0.1:{port}"))
        request.setRawHeader(
            b"Cookie",
            f"{web_main_ui.SESSION_COOKIE}={web_main_ui.SESSION_TOKEN}".encode("ascii"),
        )
        self.assertTrue(wait_for(good.connected, lambda: good.open(request)))
        app.processEvents()
        self.assertEqual(len(server._connections), 1)
        self.assertTrue(wait_for(good.disconnected, good.close))
        app.processEvents()
        self.assertEqual(len(server._connections), 0)

        bad = QWebSocket("https://evil.example")
        bad_request = QNetworkRequest(QUrl(f"ws://127.0.0.1:{port}"))
        bad_request.setRawHeader(
            b"Cookie",
            f"{web_main_ui.SESSION_COOKIE}={web_main_ui.SESSION_TOKEN}".encode("ascii"),
        )
        self.assertTrue(wait_for(bad.disconnected, lambda: bad.open(bad_request)))
        app.processEvents()
        self.assertEqual(len(server._connections), 0)
        server.close()

    def test_facade_blocks_unlisted_methods_and_config_is_not_broadcast(self):
        bridge = VueBridge()
        facade = web_main_ui.WebBridgeFacade(bridge)
        capabilities = json.loads(facade.getCapabilities())
        self.assertIn("getInitialConfig", capabilities["methods"])
        self.assertNotIn("requestInitialConfig", capabilities["methods"])
        # 테스트만 부르던 호환 별칭은 없앴다(P13c) — 프론트는 getInitialConfig 만 당겨 간다.
        self.assertFalse(hasattr(VueBridge, "requestInitialConfig"))

        legacy_settings_methods = {
            "getBackendRuntimeState",
            "runBackendRuntimeOperation",
            "getGenerationApiState",
            "runGenerationApiOperation",
            "selectBackendExtensionDirectory",
            "selectBackendInstallDirectory",
            "getForgeModelPaths",
            "selectForgeModelDirectory",
            "saveForgeModelPaths",
            "resetForgeModelPaths",
            "refreshForgeModelPaths",
        }
        self.assertTrue(
            legacy_settings_methods.isdisjoint(capabilities["methods"]),
            "웹은 redacted studio settings Interface만 사용해야 합니다.",
        )
        for method in legacy_settings_methods:
            # 레거시 설정 슬롯은 도달 불가라 제거했다(audit #176) — 되살아나지 않게 고정한다.
            self.assertFalse(hasattr(VueBridge, method), method)
            reply = json.loads(facade.invoke(method, "[]"))
            self.assertFalse(reply["ok"], method)

        allowed = json.loads(facade.invoke("getAllWidgetValues", "[]"))
        blocked = json.loads(facade.invoke("set_action_handler", "[]"))
        self.assertTrue(allowed["ok"])
        self.assertFalse(blocked["ok"])

        broadcasts = []
        bridge.uiPrefsLoaded.connect(broadcasts.append)
        payload = json.loads(bridge.getInitialConfig())
        self.assertIn("uiPrefs", payload)
        self.assertEqual(broadcasts, [])

    def test_frontend_slot_usage_is_covered_by_facade(self):
        root = pathlib.Path(__file__).resolve().parents[1]
        source = (root / "ui" / "vue_bridge.py").read_text(encoding="utf-8")
        frontend = "\n".join(
            path.read_text(encoding="utf-8")
            for path in (root / "frontend" / "src").rglob("*")
            if path.suffix in {".vue", ".js", ".ts"}
        )
        slots = set(re.findall(
            r"(?:^[ \t]*@pyqtSlot[^\n]*\n)+^[ \t]*def\s+(\w+)",
            source,
            re.MULTILINE,
        ))
        used = {
            name for name in slots
            if re.search(r"\." + re.escape(name) + r"\b", frontend)
            or re.search(r"[\"']" + re.escape(name) + r"[\"']", frontend)
        }
        desktop_only = {
            "copyTextToClipboard",
            # Reads the host PC clipboard via Qt; remote web clients paste through editorPasteImage.
            "editorPasteFromSystemClipboard",
            "loadImageBase64",  # Local original bytes for the opt-in hand repair panel.
        }
        missing = used - web_main_ui._WEB_METHODS - desktop_only
        self.assertEqual(missing, set())
        # 반대 방향: 프론트가 부르지 않는 슬롯이 화이트리스트에 남으면 웹 공격면만 넓힌다
        # (예전 captionImage·pairColors·generateXYZCombinations — 호출자가 사라진 뒤에도 방치됐다).
        self.assertEqual(web_main_ui._WEB_METHODS - used, set())
        self.assertTrue(desktop_only <= used, "데스크톱 전용 예외도 프론트가 실제로 부르는 슬롯이어야 한다.")
        self.assertTrue(desktop_only.isdisjoint(web_main_ui._WEB_METHODS))
        self.assertTrue({
            "backendRuntimeEvent",
        }.isdisjoint(web_main_ui._WEB_SIGNALS))
        self.assertTrue(all(callable(getattr(VueBridge, name, None)) for name in web_main_ui._WEB_METHODS))

    def test_frontend_events_are_covered_by_web_signals(self):
        """공용 프론트 이벤트는 웹에 공개하고, 네이티브 전용 이벤트는 차단한다.

        누락되면 웹 모드에서 그 이벤트만 조용히 사라진다(예전 searchResultsReady).
        로컬 설치 경로를 포함하는 모델 다운로드는 명시적인 데스크톱 전용 예외다.
        """
        root = pathlib.Path(__file__).resolve().parents[1]
        source = (root / "ui" / "vue_bridge.py").read_text(encoding="utf-8")
        frontend = "\n".join(
            path.read_text(encoding="utf-8")
            for path in (root / "frontend" / "src").rglob("*")
            if path.suffix in {".vue", ".js", ".ts"}
        )
        signals = set(re.findall(r"^[ \t]*(\w+)\s*=\s*pyqtSignal\(", source, re.MULTILINE))
        listened = set(re.findall(r"onBackendEvent\(\s*[\"'](\w+)[\"']", frontend))
        native_only_events = {"modelDownloadEvent", "comfyCompatibilityResult", "comfyWorkflowEvent", "relightEvent", "handReconstructionEvent",
                              "tileRepairResult"}
        self.assertTrue(native_only_events <= signals,
                        "네이티브 전용 예외도 실제 VueBridge 시그널이어야 합니다.")
        self.assertTrue(native_only_events.isdisjoint(web_main_ui._WEB_SIGNALS),
                        "로컬 설치·설정·파일 처리 이벤트는 웹 클라이언트에 공개하면 안 됩니다.")
        missing = (listened & signals) - web_main_ui._WEB_SIGNALS - native_only_events
        self.assertEqual(missing, set())
        # 반대 방향: 화이트리스트에 실제 시그널이 아닌 이름(오타/사문)이 남지 않도록.
        self.assertEqual(web_main_ui._WEB_SIGNALS - signals, set())

    def test_web_onaction_never_reaches_backend_gate_or_host_url_handlers(self):
        """onAction 은 웹에 통째로 열려 있다 — 호스트 권한 액션은 핸들러 전에 막혀야 한다.

        백엔드 게이트는 웹 모드에서 열리지 않는데(generator_webui._can_use_backend_gate)
        select_backend 는 백엔드 URL·워크플로 경로를 save_settings 로 영구 변경하고,
        open_url 은 Windows 에서 os.startfile 로 이어진다(core.web_action_policy).
        """
        bridge = VueBridge()
        bridge._backend_runtime_is_web_mode = lambda: True
        bridge._web_clients_are_remote = lambda: False   # loopback 웹 모드도 마찬가지
        reached = []
        bridge.set_action_handler(lambda action, payload: reached.append(action))
        for action in ("select_backend", "probe_backend", "pick_comfy_workflow", "open_url", "show_api_manager"):
            bridge.onAction(action, json.dumps({"url": "http://127.0.0.1:9", "type": "webui"}))
        self.assertEqual(reached, [])
        # 평범한 액션은 그대로 핸들러에 간다.
        bridge.onAction("show_toast", json.dumps({"type": "info", "msg": "ok"}))
        self.assertEqual(reached, ["show_toast"])


def _calls_outside_nested_scopes(func):
    """함수 본문의 Call 노드 — 안쪽 def·lambda(워커로 넘기는 몫)는 빼고."""
    import ast

    calls = []
    stack = list(func.body)
    while stack:
        node = stack.pop()
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda, ast.ClassDef)):
            continue
        if isinstance(node, ast.Call):
            calls.append(node)
        stack.extend(ast.iter_child_nodes(node))
    return calls


#: 이름만 봐도 HTTP·모델 추론인 호출 — 모듈 이름(requests …)으로는 안 보이는 간접 경로.
#: captionImage 가 self._run_caption_inference → engine.caption_result(Ollama HTTP·CAFormer ONNX)로
#: GUI 스레드를 수백 초 막았는데 예전 가드는 직접 requests 호출만 봐서 놓쳤다(Codex R3 재검토 #3).
_BLOCKING_SELF_METHODS = frozenset({"_run_caption_inference", "_caption_runtime_snapshot"})
_BLOCKING_ATTRS = frozenset({"caption_result"})
_BLOCKING_CONSTRUCTORS = frozenset({"OllamaClient"})


def _blocking_call_name(call):
    """GUI 스레드에서 돌면 안 되는 네트워크·조회·추론 호출이면 그 이름, 아니면 None."""
    import ast

    func = call.func
    if isinstance(func, ast.Attribute):
        if isinstance(func.value, ast.Name) and func.value.id in {"requests", "urllib", "http"}:
            return f"{func.value.id}.{func.attr}"
        if func.attr == "urlopen":
            return "urlopen"
        if isinstance(func.value, ast.Name) and func.value.id == "self" \
                and (re.fullmatch(r"_load_\w+_json", func.attr) or func.attr in _BLOCKING_SELF_METHODS):
            return f"self.{func.attr}"
        if func.attr in _BLOCKING_ATTRS:
            return f".{func.attr}"
    if isinstance(func, ast.Name) and (func.id == "urlopen" or func.id in _BLOCKING_CONSTRUCTORS):
        return func.id
    return None


def _blocking_calls_reachable(methods, name):
    """슬롯 ``name`` 이 호출 스레드에서 (같은 클래스의 self.helper 를 따라가며) 부르는 막는 호출.

    → ``["slot → helper : label", …]``. 안쪽 def·lambda(워커·_run_async_lookup 로 넘기는 몫)는 뺀다.
    """
    import ast

    found = []
    seen = {name}
    queue = [(name, (name,))]
    while queue:
        current, chain = queue.pop()
        for call in _calls_outside_nested_scopes(methods[current]):
            label = _blocking_call_name(call)
            if label:
                found.append(f"{' → '.join(chain)} : {label}")
            func = call.func
            if isinstance(func, ast.Attribute) and isinstance(func.value, ast.Name) \
                    and func.value.id == "self" and func.attr in methods and func.attr not in seen:
                seen.add(func.attr)
                queue.append((func.attr, chain + (func.attr,)))
    return sorted(found)


@unittest.skipIf(web_main_ui is None, f"PyQt6 WebEngine unavailable: {_WEB_IMPORT_ERROR}")
class WebSlotsNeverBlockTheGuiThreadTests(unittest.TestCase):
    """Codex R3 #3 — 웹에 공개한 슬롯이 GUI 스레드에서 HTTP 를 돌리면 그동안 창과 모든 WebSocket
    클라이언트가 멈춘다. 동기 getUpscalers·ollamaListModels·getADetailerModels(timeout 5초, ollama 는
    클라이언트가 준 주소)는 프론트가 쓰지도 않는 폴백이었는데 인증된 클라이언트는 직접 부를 수 있었다.
    동기 captionImage(재검토 #3)도 같은 부류 — 클라이언트가 준 Ollama url 로 최대 900초 HTTP 또는
    CAFormer ONNX 추론을 GUI 스레드에서 돌렸다. 캡션은 startCaptionBatch(워커 스레드)만 남긴다.
    동기 getGalleryImages(P13c)는 GUI 스레드 폴더 스캔 — 갤러리·히스토리는 requestGalleryImages 만 쓴다."""

    REMOVED = ("getUpscalers", "ollamaListModels", "getADetailerModels", "captionImage",
               "getGalleryImages")
    #: 프론트 호출자 없이 웹 화이트리스트에만 남아 있던 사문 슬롯(P13c) — 되살리지 않는다.
    DEAD_SLOTS = ("requestAction", "requestInitialConfig", "getCharacterCopyright", "pairColors",
                  "refineToSpecificTags", "getClothingRegions", "generateXYZCombinations")

    def test_dead_slots_are_gone_from_bridge_and_facade(self):
        facade = web_main_ui.WebBridgeFacade(VueBridge())
        capabilities = json.loads(facade.getCapabilities())["methods"]
        for name in self.DEAD_SLOTS:
            with self.subTest(name=name):
                self.assertNotIn(name, web_main_ui._WEB_METHODS)
                self.assertNotIn(name, capabilities)
                self.assertFalse(hasattr(VueBridge, name))
                self.assertFalse(json.loads(facade.invoke(name, "[]"))["ok"])
        # 살아 있는 짝 — 액션은 onAction, 초기 설정은 getInitialConfig, 작품명은 getCharacterFeatures
        for name in ("onAction", "getInitialConfig", "getCharacterFeatures", "requestGalleryImages"):
            self.assertIn(name, capabilities)

    def test_removed_sync_lookups_are_unreachable(self):
        facade = web_main_ui.WebBridgeFacade(VueBridge())
        capabilities = json.loads(facade.getCapabilities())["methods"]
        for name in self.REMOVED:
            with self.subTest(name=name):
                self.assertNotIn(name, web_main_ui._WEB_METHODS)
                self.assertNotIn(name, capabilities)
                self.assertFalse(hasattr(VueBridge, name))
                reply = json.loads(facade.invoke(name, json.dumps(["http://127.0.0.1:9"])))
                self.assertFalse(reply["ok"])
        for name in ("requestUpscalers", "requestOllamaModels", "requestADetailerModels",
                     "startCaptionBatch", "requestCaptionRuntime"):
            self.assertIn(name, capabilities)

    @staticmethod
    def _bridge_methods():
        import ast

        root = pathlib.Path(__file__).resolve().parents[1]
        tree = ast.parse((root / "ui" / "vue_bridge.py").read_text(encoding="utf-8"))
        bridge_cls = next(node for node in tree.body
                          if isinstance(node, ast.ClassDef) and node.name == "VueBridge")
        return {node.name: node for node in bridge_cls.body
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))}

    def test_no_web_slot_does_network_io_on_the_calling_thread(self):
        methods = self._bridge_methods()
        offenders = {}
        for name in sorted(web_main_ui._WEB_METHODS & set(methods)):
            blocking = _blocking_calls_reachable(methods, name)
            if blocking:
                offenders[name] = blocking
        self.assertEqual(
            offenders, {},
            "웹 슬롯은 네트워크·조회·추론을 _run_async_lookup 이나 워커 스레드로 넘겨야 한다"
            "(request* → *Ready, 캡션은 startCaptionBatch)",
        )

    def test_guard_follows_self_helpers_to_indirect_blocking_calls(self):
        """self.helper 를 거친 HTTP·추론도 잡는다 — 예전 가드는 직접 호출만 봐서 captionImage 를 놓쳤다."""
        import ast

        source = (
            "class VueBridge:\n"
            "    def captionImage(self, payload):\n"
            "        p = self._prepare(payload)\n"
            "        return self._run_caption_inference(self._create_engine(p), 'x', p)\n"
            "    def _prepare(self, payload):\n"
            "        return self._helper()\n"
            "    def _helper(self):\n"
            "        return OllamaClient('http://x', 'm').list_models()\n"
            "    def _create_engine(self, p):\n"
            "        return object()\n"
            "    def _run_caption_inference(self, engine, path, p):\n"
            "        return engine.caption_result(path)\n"
            "    def startCaptionBatch(self, payload):\n"
            "        def _run():\n"
            "            self._run_caption_inference(None, 'x', {})\n"
            "        threading.Thread(target=_run).start()\n"
            "        self._run_async_lookup('k', lambda: self._helper(), None)\n"
            "    def _run_async_lookup(self, key, fn, signal):\n"
            "        threading.Thread(target=lambda: signal.emit(fn())).start()\n"
        )
        cls = ast.parse(source).body[0]
        methods = {node.name: node for node in cls.body if isinstance(node, ast.FunctionDef)}
        self.assertEqual(
            _blocking_calls_reachable(methods, "captionImage"),
            sorted([
                "captionImage : self._run_caption_inference",
                "captionImage → _prepare → _helper : OllamaClient",
                "captionImage → _run_caption_inference : .caption_result",
            ]),
        )
        self.assertEqual(_blocking_calls_reachable(methods, "startCaptionBatch"), [])

    def test_real_caption_inference_helper_is_recognised_as_blocking(self):
        """가드의 이름 목록이 실제 코드와 어긋나지 않게 — 목록의 헬퍼가 진짜 추론을 부른다."""
        methods = self._bridge_methods()
        self.assertIn("_run_caption_inference", methods)
        self.assertIn(
            "_run_caption_inference : .caption_result",
            _blocking_calls_reachable(methods, "_run_caption_inference"),
        )
        self.assertTrue(any(label.endswith(": OllamaClient")
                            for label in _blocking_calls_reachable(methods, "_caption_runtime_snapshot")))

    def test_guard_recognises_the_patterns_it_forbids(self):
        import ast

        source = (
            "def slot(self):\n"
            "    requests.get('http://x', timeout=5)\n"
            "    return self._load_upscalers_json()\n"
            "def async_slot(self):\n"
            "    def _work():\n"
            "        requests.get('http://x')\n"
            "    self._run_async_lookup('k', lambda: self._load_upscalers_json(), None)\n"
        )
        blocking, deferred = ast.parse(source).body
        self.assertEqual(
            sorted(_blocking_call_name(c) for c in _calls_outside_nested_scopes(blocking)
                   if _blocking_call_name(c)),
            ["requests.get", "self._load_upscalers_json"],
        )
        self.assertEqual([c for c in _calls_outside_nested_scopes(deferred) if _blocking_call_name(c)], [])


if __name__ == "__main__":
    unittest.main()
