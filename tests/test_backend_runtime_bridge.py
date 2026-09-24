"""앱 시작 managed runtime 자동기동 — Studio runtime.execute 경로 회귀 테스트.

예전엔 자동기동만 레거시 VueBridge.runBackendRuntimeOperation 을 타서 backendRuntimeEvent 만
나왔고, Studio journal 만 구독하는 Settings 는 완료 이벤트를 못 받아 런타임 버튼이 앱 재시작
전까지 비활성으로 남았다(audit #46). 이제 자동기동도 Settings 와 같은 Studio 작업이고,
레거시 실행기는 제거했다(audit #176).
"""
import json
import sys
import tempfile
import threading
import time
import types
import unittest
from unittest.mock import patch

from PyQt6.QtCore import QCoreApplication, QObject, pyqtSignal

from core.runtime_autostart import autostart_request, pick_autostart_engine, request_runtime_autostart
from core.studio_application import CallContext, StudioApplication
from ui.generator_webui import WebUIMixin
from ui.studio_qwebchannel import DesktopNativeHost
from ui.vue_bridge import VueBridge

NATIVE = CallContext("desktop-ui", "qwebchannel", frozenset({"native"}))
WEB = CallContext("web-ui", "qwebchannel-websocket", frozenset())


class _Manager:
    def __init__(self, *, forge_auto=True, comfy_auto=False, comfy_installed=False, active="forge"):
        self.execute_calls = []
        self._forge_auto = forge_auto
        self._comfy_auto = comfy_auto
        self._comfy_installed = comfy_installed
        self._active = active
        self._busy = False

    def snapshot(self):
        return {
            "ok": True,
            "activeEngine": self._active,
            "primaryModelEngine": "forge",
            "engines": {
                "forge": {"engine": "forge", "installed": True, "running": False,
                          "autoStart": self._forge_auto, "busy": self._busy,
                          "apiUrl": "http://127.0.0.1:7860"},
                "comfyui": {"engine": "comfyui", "installed": self._comfy_installed,
                            "running": False, "autoStart": self._comfy_auto, "busy": False,
                            "apiUrl": "http://127.0.0.1:8188"},
            },
        }

    def execute(self, engine, action, payload=None, on_progress=None):
        self.execute_calls.append((engine, action, dict(payload or {})))
        self._busy = True
        if on_progress:
            on_progress({"phase": "health", "message": "ready"})
        self._busy = False
        return {
            "ok": True,
            "engine": engine,
            "action": action,
            "message": "ready",
            "apiUrl": "http://127.0.0.1:7860",
            "owned": True,
            "activate": action == "use" or bool((payload or {}).get("startup", False)),
        }


def _runtime_module(manager):
    module = types.ModuleType("core.backend_runtime")
    module.get_backend_runtime_manager = lambda: manager
    return module


class _Host(WebUIMixin):
    web_mode = False

    def __init__(self, application, context=NATIVE):
        self.studio_application = application
        self.studio_native_context = context


class _FakeStudio:
    def __init__(self, reply):
        self.reply = reply
        self.calls = []

    def invoke(self, context, request):
        self.calls.append((context, request))
        return dict(self.reply, requestId=request["requestId"])


def _wait(predicate, timeout=2.0):
    app = QCoreApplication.instance() or QCoreApplication([])
    deadline = time.monotonic() + timeout
    while not predicate() and time.monotonic() < deadline:
        app.processEvents()
        time.sleep(0.01)
    app.processEvents()
    return predicate()


class PickAutostartEngineTests(unittest.TestCase):
    def test_prefers_active_engine_and_requires_install(self):
        self.assertEqual(pick_autostart_engine(_Manager().snapshot()), "forge")
        both = _Manager(comfy_auto=True, comfy_installed=True, active="comfyui").snapshot()
        self.assertEqual(pick_autostart_engine(both), "comfyui")
        self.assertEqual(pick_autostart_engine(_Manager(forge_auto=False).snapshot()), "")
        not_installed = _Manager(forge_auto=False, comfy_auto=True, comfy_installed=False).snapshot()
        self.assertEqual(pick_autostart_engine(not_installed), "")
        self.assertEqual(pick_autostart_engine(None), "")

    def test_request_envelope_is_a_startup_runtime_execute(self):
        request = autostart_request("forge", "rid")
        self.assertEqual(request["operation"], "runtime.execute")
        self.assertEqual(request["input"], {"engine": "forge", "action": "start", "payload": {"startup": True}})
        self.assertEqual(request["version"], 1)


class ManagedAutostartTests(unittest.TestCase):
    def test_autostart_uses_the_unique_toggle_even_when_another_engine_was_active(self):
        manager = _Manager(forge_auto=False, comfy_auto=True, comfy_installed=True, active="forge")
        studio = _FakeStudio({"version": 1, "status": "accepted", "seq": 1, "data": {}})
        host = _Host(studio)
        with patch.dict(sys.modules, {"core.backend_runtime": _runtime_module(manager)}):
            self.assertTrue(host._try_managed_backend_autostart())

        context, request = studio.calls[0]
        self.assertIs(context, NATIVE)
        self.assertEqual(request["input"]["engine"], "comfyui")
        self.assertEqual(request["input"]["action"], "start")
        self.assertTrue(request["input"]["payload"]["startup"])
        self.assertEqual(host._backend_startup_result, "managed_pending")
        self.assertTrue(host._managed_runtime_startup_inflight)

    def test_rejected_request_marks_managed_failed_with_reason(self):
        manager = _Manager()
        studio = _FakeStudio({"version": 1, "status": "error", "seq": 1,
                              "error": {"code": "FORBIDDEN", "message": "권한 없음"}})
        host = _Host(studio)
        with patch.dict(sys.modules, {"core.backend_runtime": _runtime_module(manager)}):
            self.assertTrue(host._try_managed_backend_autostart())
        self.assertEqual(host._backend_startup_result, "managed_failed")
        self.assertFalse(host._managed_runtime_startup_inflight)
        self.assertEqual(host._managed_runtime_startup_error, "권한 없음")

    def test_web_host_and_ineligible_runtime_do_not_start(self):
        studio = _FakeStudio({"status": "accepted"})
        web = _Host(studio)
        web.web_mode = True
        self.assertFalse(web._try_managed_backend_autostart())
        host = _Host(studio)
        with patch.dict(sys.modules, {"core.backend_runtime": _runtime_module(_Manager(forge_auto=False))}):
            self.assertFalse(host._try_managed_backend_autostart())
        self.assertEqual(studio.calls, [])

    def test_missing_studio_application_fails_without_crashing(self):
        accepted, error = request_runtime_autostart(None, NATIVE, "forge")
        self.assertFalse(accepted)
        self.assertTrue(error)

    def test_autostart_reaches_settings_journal_and_desktop_host_through_real_studio(self):
        """자동기동 진행/완료가 Settings(Studio journal)와 generator_main(backendRuntimeEvent) 둘 다에 간다."""

        class _Bridge(QObject):
            backendRuntimeEvent = pyqtSignal(str)

            def __init__(self):
                super().__init__()
                self.events = []
                self.backendRuntimeEvent.connect(lambda raw: self.events.append(json.loads(raw)))

        QCoreApplication.instance() or QCoreApplication([])
        bridge = _Bridge()
        window = QObject()
        native_host = DesktopNativeHost(window, bridge)
        manager = _Manager()
        application = StudioApplication(host=native_host, runtime_manager=manager)
        journal = []
        stop = application.subscribe(NATIVE, journal.append)
        self.addCleanup(stop)

        host = _Host(application)
        with patch.dict(sys.modules, {"core.backend_runtime": _runtime_module(manager)}):
            self.assertTrue(host._try_managed_backend_autostart())
            self.assertTrue(_wait(lambda: any(e.get("type") == "completed" for e in bridge.events)))

        runtime_types = [e["type"] for e in journal if e["topic"] == "runtime.operation"]
        self.assertEqual(runtime_types[0], "accepted")
        self.assertIn("started", runtime_types)
        self.assertIn("progress", runtime_types)
        self.assertEqual(runtime_types[-1], "completed", "Settings 가 완료를 받아 busy 를 풀 수 있어야 한다")
        completed_snapshot = [e for e in journal if e["type"] == "completed"][-1]["data"]["snapshot"]
        self.assertFalse(completed_snapshot["engines"]["forge"]["busy"])

        host_types = [e["type"] for e in bridge.events]
        self.assertIn("started", host_types)
        completed = [e for e in bridge.events if e["type"] == "completed"][-1]
        self.assertTrue(completed["startup"])
        self.assertTrue(completed["activate"], "시작 자동기동은 연결 전환(activate)을 일으킨다")
        self.assertEqual(completed["engine"], "forge")
        self.assertEqual(manager.execute_calls, [("forge", "start", {"startup": True})])

    def test_web_context_cannot_execute_runtime(self):
        manager = _Manager()
        reply = StudioApplication(runtime_manager=manager).invoke(WEB, autostart_request("forge", "r"))
        self.assertEqual(reply["status"], "error")
        self.assertEqual(reply["error"]["code"], "FORBIDDEN")
        self.assertEqual(manager.execute_calls, [])

    def test_actual_core_configure_operation_through_studio(self):
        from core import backend_runtime

        with tempfile.TemporaryDirectory() as temp:
            manager = backend_runtime.BackendRuntimeManager(
                config_path=f"{temp}/runtime.json",
                runtime_root=f"{temp}/managed",
            )
            application = StudioApplication(runtime_manager=manager)
            done = threading.Event()
            events = []

            def sink(event):
                if event["topic"] == "runtime.operation":
                    events.append(event)
                    if event["type"] in {"completed", "error"}:
                        done.set()

            stop = application.subscribe(NATIVE, sink)
            self.addCleanup(stop)
            reply = application.invoke(NATIVE, {
                "version": 1, "requestId": "cfg", "operation": "runtime.execute",
                "input": {"engine": "forge", "action": "set_auto_start", "payload": {"autoStart": True}},
            })
            self.assertEqual(reply["status"], "accepted")
            self.assertTrue(done.wait(2.0))
            self.assertEqual(events[-1]["type"], "completed")
            state = application.invoke(NATIVE, {"version": 1, "requestId": "snap",
                                                "operation": "runtime.snapshot", "input": {}})
            self.assertTrue(state["data"]["engines"]["forge"]["autoStart"])

    def test_legacy_runtime_slots_are_gone(self):
        for name in ("runBackendRuntimeOperation", "_run_backend_runtime_operation",
                     "_emit_backend_runtime_event", "getBackendRuntimeState",
                     "_backend_runtime_public_snapshot", "selectBackendExtensionDirectory",
                     "selectBackendInstallDirectory"):
            self.assertFalse(hasattr(VueBridge, name), name)
        # Python 내부 시그널과 웹 모드 판정은 남는다(DesktopNativeHost·generator_main 이 쓴다).
        self.assertTrue(hasattr(VueBridge, "backendRuntimeEvent"))
        self.assertTrue(hasattr(VueBridge, "_backend_runtime_is_web_mode"))
        self.assertTrue(hasattr(VueBridge, "_refresh_forge_module_widgets"))


if __name__ == "__main__":
    unittest.main()
