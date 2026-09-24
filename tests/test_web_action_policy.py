"""웹 모드 onAction 권한 정책 + 외부 URL 열기 회귀 테스트.

예전 onAction 의 웹 거부 목록은 show_api_manager 하나뿐이었다. 그래서 세션 토큰을 가진
웹 클라이언트가
  - open_url 로 호스트에서 webbrowser.open(= Windows os.startfile)을 부를 수 있었고,
  - 웹에선 열리지도 않는 백엔드 게이트 액션으로 포트를 두드리거나 백엔드 설정을 바꿨고,
  - 원격(LAN) 웹 모드에서 호스트 데스크톱에 파일 대화상자를 띄웠다.
"""
from __future__ import annotations

import ast
import json
import pathlib
import re
import unittest
from types import SimpleNamespace
from unittest import mock

from core.url_safety import UNSAFE_URL_MESSAGE, is_safe_external_url, open_external_url
from core.web_action_policy import (
    BRANCH_HANDLED_DIALOG_ACTIONS,
    DESKTOP_DIALOG_ACTIONS,
    HOST_DIALOG_MESSAGE,
    WEB_BLOCKED_ACTIONS,
    host_dialogs_blocked,
    is_loopback_bind_host,
    web_action_denial,
)

ROOT = pathlib.Path(__file__).resolve().parents[1]
UI = ROOT / "ui"


class UrlSafetyTests(unittest.TestCase):
    def test_http_and_https_with_host_are_allowed(self):
        for url in (
            "https://github.com/UR-al/UR_IV/releases",
            "http://example.com/release-notes?x=1#y",
            "HTTPS://Example.com:8443/path",
            "  https://example.com  ",
        ):
            with self.subTest(url=url):
                self.assertTrue(is_safe_external_url(url))

    def test_values_the_host_shell_could_execute_are_rejected(self):
        for url in (
            "file:///C:/Windows/System32/calc.exe",
            "C:\\Windows\\System32\\calc.exe",
            "C:/Windows/System32/calc.exe",
            "\\\\server\\share\\run.exe",
            "//server/share/run.exe",
            "search-ms:query=x",
            "shell:startup",
            "ms-settings:privacy",
            "javascript:alert(1)",
            "data:text/html,<script>1</script>",
            "http:evil.example",
            "https:///missing-host",
            "https://exa mple.com",
            "https://example.com/\\..\\x",
            "https://example.com/\x00",
            "https://example.com:99999/",
            "mailto:someone@example.com",
            "",
            "https://" + "a" * 9000 + ".com",
            None,
            42,
        ):
            with self.subTest(url=url):
                self.assertFalse(is_safe_external_url(url))

    def test_mailto_only_when_the_caller_allows_it(self):
        self.assertTrue(is_safe_external_url("mailto:maintainer@example.com", allow_mailto=True))
        self.assertFalse(is_safe_external_url("mailto:", allow_mailto=True))
        self.assertFalse(is_safe_external_url("mailto:no-at-sign", allow_mailto=True))


class OpenExternalUrlTests(unittest.TestCase):
    def test_desktop_opens_only_safe_urls(self):
        opener = mock.Mock()
        self.assertEqual(open_external_url(" https://example.com/r ", web_mode=False, opener=opener), (True, ""))
        opener.assert_called_once_with("https://example.com/r")

        opener.reset_mock()
        opened, message = open_external_url("file:///C:/Windows/System32/calc.exe", web_mode=False, opener=opener)
        self.assertFalse(opened)
        self.assertEqual(message, UNSAFE_URL_MESSAGE)
        opener.assert_not_called()

    def test_web_mode_never_opens_on_the_host(self):
        opener = mock.Mock()
        opened, message = open_external_url("https://example.com", web_mode=True, opener=opener)
        self.assertFalse(opened)
        self.assertIn("웹 모드", message)
        opener.assert_not_called()

    def test_empty_values_are_ignored_silently(self):
        opener = mock.Mock()
        for value in ("", "   ", None):
            self.assertEqual(open_external_url(value, web_mode=False, opener=opener), (False, ""))
        opener.assert_not_called()

    def test_open_url_branch_uses_the_checked_opener(self):
        tree = ast.parse((UI / "generator_main.py").read_text(encoding="utf-8"))
        branch = _action_branch(tree, "open_url")
        self.assertIsNotNone(branch)
        called = _called_names(branch)
        self.assertIn("open_external_url", called)
        self.assertNotIn("startfile", called)
        self.assertFalse(
            any(isinstance(n, ast.Attribute) and n.attr == "open"
                and isinstance(n.value, ast.Name) and n.value.id == "webbrowser" for n in ast.walk(branch)),
            "open_url 분기가 webbrowser.open 을 직접 부르면 검사를 건너뛴다",
        )


class PolicyTests(unittest.TestCase):
    def test_desktop_mode_never_denies(self):
        for action in (*WEB_BLOCKED_ACTIONS, *DESKTOP_DIALOG_ACTIONS, "generate"):
            self.assertIsNone(web_action_denial(action, web_mode=False, remote=True))

    def test_host_authority_actions_are_denied_in_every_web_mode(self):
        for remote in (False, True):
            for action in ("show_api_manager", "probe_backend", "select_backend", "pick_comfy_workflow", "open_url",
                           " SELECT_BACKEND ", "copy_to_clipboard"):
                with self.subTest(action=action, remote=remote):
                    self.assertTrue(web_action_denial(action, web_mode=True, remote=remote))

    def test_dialog_actions_are_denied_only_for_remote_web_clients(self):
        for action in DESKTOP_DIALOG_ACTIONS - BRANCH_HANDLED_DIALOG_ACTIONS:
            with self.subTest(action=action):
                self.assertIsNone(web_action_denial(action, web_mode=True, remote=False))
                self.assertEqual(web_action_denial(action, web_mode=True, remote=True), HOST_DIALOG_MESSAGE)

    def test_branch_handled_dialog_actions_reach_their_handler(self):
        # 에디터 저장은 editorSaveResult 로 결과를 돌려줘야 프론트 '저장 중' 이 풀린다.
        for action in BRANCH_HANDLED_DIALOG_ACTIONS:
            self.assertIsNone(web_action_denial(action, web_mode=True, remote=True))

    def test_ordinary_actions_stay_open_on_the_web(self):
        for action in ("generate", "delete_image", "add_favorite", "show_toast", "chat_send", "editor_save"):
            self.assertIsNone(web_action_denial(action, web_mode=True, remote=True))

    def test_loopback_bind_detection(self):
        for host in ("127.0.0.1", "127.5.6.7", "localhost", "LOCALHOST", "::1", "[::1]"):
            self.assertTrue(is_loopback_bind_host(host), host)
        for host in ("0.0.0.0", "::", "", None, "192.168.0.10", "10.0.0.2", "my-pc", "fe80::1"):
            self.assertFalse(is_loopback_bind_host(host), host)
        self.assertTrue(host_dialogs_blocked(web_mode=True, remote=True))
        self.assertFalse(host_dialogs_blocked(web_mode=True, remote=False))
        self.assertFalse(host_dialogs_blocked(web_mode=False, remote=True))


class FrontendMirrorTests(unittest.TestCase):
    def test_frontend_dialog_list_matches_python_policy(self):
        source = (ROOT / "frontend" / "src" / "utils" / "hostDialogs.ts").read_text(encoding="utf-8")
        block = re.search(r"DESKTOP_DIALOG_ACTIONS:\s*readonly string\[\]\s*=\s*\[(.*?)\]", source, re.S)
        self.assertIsNotNone(block)
        names = set(re.findall(r"'([a-z_]+)'", block.group(1)))
        self.assertEqual(names, set(DESKTOP_DIALOG_ACTIONS))
        message = re.search(r"HOST_DIALOG_MESSAGE\s*=\s*'([^']+)'", source)
        self.assertIsNotNone(message)
        self.assertEqual(message.group(1), HOST_DIALOG_MESSAGE)

    def test_runtime_flag_name_matches_web_server(self):
        ts = (ROOT / "frontend" / "src" / "utils" / "hostDialogs.ts").read_text(encoding="utf-8")
        web = (ROOT / "web_main_ui.py").read_text(encoding="utf-8")
        self.assertIn("__AISTUDIO_HOST_DIALOGS__", ts)
        self.assertIn("window.__AISTUDIO_HOST_DIALOGS__=", web)


# ── 대화상자 도달 분석 ────────────────────────────────────────────────────────
_SKIP = {"vue_bridge.py", "native_dialogs.py", "studio_qwebchannel.py"}
_DIALOG_CLASSES = {"QFileDialog", "QInputDialog", "QColorDialog", "QFontDialog"}
_DISPATCHER = re.compile(r"^_handle_\w*action$")


def _called_names(node) -> set[str]:
    names = set()
    for n in ast.walk(node):
        if isinstance(n, ast.Call):
            f = n.func
            names.add(f.attr if isinstance(f, ast.Attribute) else getattr(f, "id", ""))
    return names


def _action_branch(tree, action):
    for n in ast.walk(tree):
        if isinstance(n, ast.If) and isinstance(n.test, ast.Compare) and isinstance(n.test.left, ast.Name) \
                and n.test.left.id == "action":
            comp = n.test.comparators[0]
            names = [comp.value] if isinstance(comp, ast.Constant) else [
                e.value for e in getattr(comp, "elts", []) if isinstance(e, ast.Constant)]
            if action in names:
                return ast.Module(body=n.body, type_ignores=[])
    return None


def _load_ui_functions():
    funcs: dict[str, list] = {}
    for path in sorted(UI.glob("*.py")):
        if path.name in _SKIP:
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"))
        aliases = set(_DIALOG_CLASSES)
        for n in ast.walk(tree):
            if isinstance(n, ast.ImportFrom):
                for a in n.names:
                    if a.name in _DIALOG_CLASSES:
                        aliases.add(a.asname or a.name)
        for n in tree.body:
            if isinstance(n, ast.ClassDef):
                for fn in n.body:
                    if isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        funcs.setdefault(fn.name, []).append((fn, aliases))
            elif isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)):
                funcs.setdefault(n.name, []).append((n, aliases))
    return funcs


def _opens_host_window(node, aliases) -> bool:
    for n in ast.walk(node):
        if not isinstance(n, ast.Call):
            continue
        f = n.func
        if isinstance(f, ast.Attribute) and isinstance(f.value, ast.Name) and f.value.id in aliases:
            return True
        if isinstance(f, ast.Name) and (f.id in {"select_directory", "QMenu"} or f.id.endswith("Dialog")):
            return True
    return False


def _refs(node) -> set[str]:
    out = set()
    for n in ast.walk(node):
        if isinstance(n, ast.Attribute) and isinstance(n.value, ast.Name) and n.value.id == "self":
            out.add(n.attr)
        elif isinstance(n, ast.Call) and isinstance(n.func, ast.Name):
            out.add(n.func.id)
    return out


def _dialog_reaching_actions() -> set[str]:
    funcs = _load_ui_functions()
    flagged = {name for name, defs in funcs.items() if any(_opens_host_window(fn, al) for fn, al in defs)}
    changed = True
    while changed:
        changed = False
        for name, defs in funcs.items():
            if name in flagged or _DISPATCHER.match(name):
                continue
            if any(_refs(fn) & flagged for fn, _ in defs):
                flagged.add(name)
                changed = True
    hits: set[str] = set()
    main_fn, main_aliases = funcs["_handle_vue_action"][0]
    for n in ast.walk(main_fn):
        if isinstance(n, ast.If) and isinstance(n.test, ast.Compare) and isinstance(n.test.left, ast.Name) \
                and n.test.left.id == "action":
            comp = n.test.comparators[0]
            names = [comp.value] if isinstance(comp, ast.Constant) else [
                e.value for e in getattr(comp, "elts", []) if isinstance(e, ast.Constant)]
            body = ast.Module(body=n.body, type_ignores=[])
            if _opens_host_window(body, main_aliases) or (_refs(body) & (flagged - {"_handle_vue_action"})):
                hits.update(names)
    # 믹스인의 {'action': self._method} 디스패치 표(creator/chat)
    for defs in funcs.values():
        for fn, _ in defs:
            for n in ast.walk(fn):
                if isinstance(n, ast.Dict):
                    for k, v in zip(n.keys, n.values):
                        if isinstance(k, ast.Constant) and isinstance(k.value, str) \
                                and isinstance(v, ast.Attribute) and isinstance(v.value, ast.Name) \
                                and v.value.id == "self" and v.attr in flagged:
                            hits.add(k.value)
    return hits


class DialogReachabilityContractTests(unittest.TestCase):
    """호스트에 네이티브 창을 띄우는 액션은 전부 정책 목록에 있어야 한다(새 액션 누락 방지)."""

    @classmethod
    def setUpClass(cls):
        cls.reaching = _dialog_reaching_actions()

    def test_every_dialog_action_is_covered_by_the_policy(self):
        covered = set(DESKTOP_DIALOG_ACTIONS) | set(WEB_BLOCKED_ACTIONS) | set(BRANCH_HANDLED_DIALOG_ACTIONS)
        self.assertEqual(self.reaching - covered, set(),
                         "호스트 대화상자를 여는 새 액션을 core.web_action_policy 에 넣으세요")

    def test_policy_lists_no_stale_dialog_actions(self):
        self.assertEqual(set(DESKTOP_DIALOG_ACTIONS) - self.reaching, set(),
                         "대화상자를 더 이상 열지 않는 액션이 목록에 남아 있습니다")
        self.assertLessEqual(set(BRANCH_HANDLED_DIALOG_ACTIONS), self.reaching)

    def test_branch_handled_save_refuses_host_dialog_itself(self):
        tree = ast.parse((UI / "generator_main.py").read_text(encoding="utf-8"))
        for action in BRANCH_HANDLED_DIALOG_ACTIONS:
            branch = _action_branch(tree, action)
            self.assertIsNotNone(branch, action)
            source = ast.unparse(branch)
            self.assertIn("host_dialogs_blocked", source)
            self.assertIn("ask_path", source)


# ── VueBridge / web_main_ui / 대기열 완료 통합 ─────────────────────────────────
try:
    from PyQt6.QtCore import QObject
    from ui.vue_bridge import VueBridge
    _QT_ERROR = ""
except ModuleNotFoundError as exc:  # pragma: no cover
    if not (exc.name or "").startswith("PyQt6"):
        raise
    QObject = object
    VueBridge = None
    _QT_ERROR = str(exc)


@unittest.skipIf(VueBridge is None, f"PyQt6 unavailable: {_QT_ERROR}")
class VueBridgeOnActionTests(unittest.TestCase):
    def _bridge(self, **host_attrs):
        host = QObject()
        for key, value in host_attrs.items():
            setattr(host, key, value)
        bridge = VueBridge(host)
        calls, notes = [], []
        bridge.set_action_handler(lambda action, payload: calls.append(action))
        bridge.showNotification.connect(lambda level, message: notes.append((level, message)))
        self.addCleanup(host.deleteLater)
        return bridge, calls, notes

    def test_remote_web_blocks_gate_url_and_dialog_actions_before_the_handler(self):
        bridge, calls, notes = self._bridge(web_mode=True, web_remote=True)
        for action in ("select_backend", "probe_backend", "pick_comfy_workflow", "open_url",
                       "open_batch_files", "gallery_open_folder", "chat_export"):
            bridge.onAction(action, json.dumps({"url": "file:///C:/Windows/System32/calc.exe"}))
        self.assertEqual(calls, [])
        self.assertEqual(len(notes), 7)
        self.assertTrue(all(level == "warning" for level, _ in notes))

    def test_loopback_web_keeps_dialogs_but_still_blocks_host_authority(self):
        bridge, calls, notes = self._bridge(web_mode=True, web_remote=False)
        bridge.onAction("open_batch_files", "{}")
        bridge.onAction("select_backend", json.dumps({"type": "webui", "url": "http://127.0.0.1:1"}))
        self.assertEqual(calls, ["open_batch_files"])
        self.assertEqual(len(notes), 1)

    def test_missing_remote_flag_fails_closed_in_web_mode(self):
        bridge, calls, _notes = self._bridge(web_mode=True)
        bridge.onAction("editor_open_file", "{}")
        self.assertEqual(calls, [])

    def test_branch_handled_save_and_desktop_actions_reach_the_handler(self):
        bridge, calls, _ = self._bridge(web_mode=True, web_remote=True)
        bridge.onAction("editor_save", "{}")
        bridge.onAction("editor_save_as", "{}")
        self.assertEqual(calls, ["editor_save", "editor_save_as"])
        desktop, desktop_calls, notes = self._bridge()
        for action in ("open_url", "select_backend", "open_batch_files"):
            desktop.onAction(action, "{}")
        self.assertEqual(desktop_calls, ["open_url", "select_backend", "open_batch_files"])
        self.assertEqual(notes, [])

    def test_image_copy_never_reaches_the_host_clipboard_from_the_web(self):
        """Codex R3 #2 — 원격 복사 버튼이 호스트 PC 사용자의 클립보드를 덮어쓰고 '복사됨'으로 답했다."""
        from core.web_action_policy import COPY_IMAGE_MESSAGE

        payload = json.dumps({"path": "C:/x/a.png"})
        for remote in (True, False):
            with self.subTest(remote=remote):
                bridge, calls, notes = self._bridge(web_mode=True, web_remote=remote)
                bridge.onAction("copy_to_clipboard", payload)
                self.assertEqual(calls, [])
                self.assertEqual(len(notes), 1)
                self.assertEqual(notes[0][0], "warning")
                self.assertIn("브라우저", notes[0][1])
                self.assertEqual(COPY_IMAGE_MESSAGE, web_action_denial(
                    "copy_to_clipboard", web_mode=True, remote=remote))
        desktop, desktop_calls, desktop_notes = self._bridge()
        desktop.onAction("copy_to_clipboard", payload)
        self.assertEqual(desktop_calls, ["copy_to_clipboard"])
        self.assertEqual(desktop_notes, [])


class CopyToClipboardBranchTests(unittest.TestCase):
    """분기도 웹 모드를 스스로 거른다(open_url 처럼 정책 한 겹에만 기대지 않는다)."""

    def _run(self, web_mode, path):
        from ui.generator_main import GeneratorMainUI
        notes = []
        host = SimpleNamespace(
            web_mode=web_mode,
            vue_bridge=SimpleNamespace(showNotification=SimpleNamespace(
                emit=lambda level, message: notes.append((level, message)))),
            show_status=lambda *_a: None,
            _handle_model_download_action=lambda *_a: False,
            _handle_chat_action=lambda *_a: False,
            _handle_creator_action=lambda *_a: False,
        )
        with mock.patch("ui.generator_main.QApplication") as app, \
                mock.patch("PyQt6.QtGui.QPixmap") as pixmap:
            pixmap.return_value.isNull.return_value = False
            GeneratorMainUI._handle_vue_action(host, "copy_to_clipboard", {"path": path})
        return app, pixmap, notes

    def test_web_mode_branch_never_touches_the_host_clipboard(self):
        from core.web_action_policy import COPY_IMAGE_MESSAGE

        app, pixmap, notes = self._run(True, "file:///C:/x/a.png")
        app.clipboard.assert_not_called()
        pixmap.assert_not_called()
        self.assertEqual(notes, [("warning", COPY_IMAGE_MESSAGE)])

    def test_desktop_branch_still_copies(self):
        app, pixmap, notes = self._run(False, "file:///C:/x/a.png")
        app.clipboard.return_value.setPixmap.assert_called_once_with(pixmap.return_value)
        self.assertEqual([level for level, _ in notes], ["success"])


try:
    import web_main_ui
    _WEB_ERROR = ""
except ModuleNotFoundError as exc:  # pragma: no cover
    if not (exc.name or "").startswith("PyQt6"):
        raise
    web_main_ui = None
    _WEB_ERROR = str(exc)


@unittest.skipIf(web_main_ui is None, f"PyQt6 WebEngine unavailable: {_WEB_ERROR}")
class WebServerFlagTests(unittest.TestCase):
    def test_host_dialog_flag_follows_bind_address(self):
        with mock.patch.object(web_main_ui, "BIND_HOST", "127.0.0.1"):
            self.assertTrue(web_main_ui._host_dialogs_available())
        with mock.patch.object(web_main_ui, "BIND_HOST", "0.0.0.0"):
            self.assertFalse(web_main_ui._host_dialogs_available())
        with mock.patch.object(web_main_ui, "BIND_HOST", "192.168.0.5"):
            self.assertFalse(web_main_ui._host_dialogs_available())

    def test_runtime_config_carries_the_flag(self):
        handler = web_main_ui._DistHandler.__new__(web_main_ui._DistHandler)
        sent = {}
        handler.send_response = lambda code: sent.setdefault("code", code)
        handler.send_header = lambda *a: None
        handler.end_headers = lambda: None

        class _Sink:
            def __init__(self):
                self.data = b""

            def write(self, b):
                self.data += b

        for bind, expected in (("127.0.0.1", b"__AISTUDIO_HOST_DIALOGS__=true"),
                               ("0.0.0.0", b"__AISTUDIO_HOST_DIALOGS__=false")):
            handler.wfile = _Sink()
            with mock.patch.object(web_main_ui, "BIND_HOST", bind):
                handler._serve_runtime_config()
            self.assertIn(expected, handler.wfile.data)
            self.assertIn(b"__AISTUDIO_WS_PORT__", handler.wfile.data)


class QueueCompletedModalTests(unittest.TestCase):
    def _run(self, web_mode):
        from ui.generator_main import GeneratorMainUI
        emitted = []
        signal = SimpleNamespace(emit=lambda *a: emitted.append(a))
        fake = SimpleNamespace(
            web_mode=web_mode,
            vue_bridge=SimpleNamespace(queueCompleted=signal, showNotification=signal),
            queue_manager=SimpleNamespace(last_stop_natural=True),
        )
        with mock.patch("ui.generator_main.QMessageBox") as box:
            GeneratorMainUI._on_queue_completed(fake, 3)
        return box, emitted

    def test_web_mode_never_leaves_a_host_modal(self):
        box, emitted = self._run(web_mode=True)
        box.information.assert_not_called()
        self.assertIn(("success", "3장 생성 완료"), emitted)

    def test_desktop_keeps_the_completion_notice(self):
        box, _ = self._run(web_mode=False)
        box.information.assert_called_once()


if __name__ == "__main__":
    unittest.main()
