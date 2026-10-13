import json
import threading
import time
import unittest
from types import SimpleNamespace
from unittest import mock

from PyQt6.QtCore import QCoreApplication, QObject

from backends import BackendType
from ui.xyz_actions import XYZActionsMixin


SCHEMA = {"KSampler": {"input": {"required": {"steps": ["INT", {"min": 1, "max": 40}]}}}}


class Signal:
    def __init__(self):
        self.values = []
        self.ready = threading.Event()
    def emit(self, text):
        self.values.append(json.loads(text))
        self.ready.set()

    def wait_for(self, count, timeout=3):
        """값이 ``count`` 개가 될 때까지(작업자 스레드가 답하는 클릭) 기다린다."""
        deadline = time.monotonic() + timeout
        while len(self.values) < count and time.monotonic() < deadline:
            time.sleep(0.01)
        return len(self.values) >= count


class XYZActionTests(unittest.TestCase):
    def test_slow_previous_backend_response_never_replaces_new_axes(self):
        actions = XYZActionsMixin()
        actions.vue_bridge = SimpleNamespace(xyzCapabilitiesReceived=Signal(), xyzPlotEvent=Signal())
        entered, release = threading.Event(), threading.Event()
        workers = []
        def slow_schema():
            workers.append(threading.current_thread())
            entered.set()
            release.wait(3)
            return SCHEMA
        old = SimpleNamespace(api_url="http://old.invalid", get_object_info=slow_schema, _load_configured_workflow=lambda _: None)
        new = SimpleNamespace(api_url="http://new.invalid", get_object_info=lambda: {}, _load_configured_workflow=lambda _: None)
        with mock.patch("backends.get_backend", return_value=old) as selected, mock.patch("backends.get_backend_type", return_value=BackendType.COMFYUI):
            actions._handle_xyz_action("get_xyz_capabilities", {"requestId": "old-query"})
            self.assertTrue(entered.wait(3))
            selected.return_value = new
            actions._handle_xyz_action("get_xyz_capabilities", {"requestId": "new-query"})
            self.assertTrue(actions.vue_bridge.xyzCapabilitiesReceived.ready.wait(3))
            release.set()
            workers[0].join(3)
            self.assertFalse(workers[0].is_alive())
            self.assertEqual(actions.vue_bridge.xyzCapabilitiesReceived.values[-1]["requestId"], "new-query")
            self.assertEqual(actions.vue_bridge.xyzCapabilitiesReceived.values[-1]["axes"], [])

    def test_api_error_clears_capabilities_and_rejects_enqueue(self):
        actions = XYZActionsMixin()
        actions.vue_bridge = SimpleNamespace(xyzCapabilitiesReceived=Signal(), xyzPlotEvent=Signal())
        backend = SimpleNamespace(api_url="http://private.invalid", get_object_info=mock.Mock(side_effect=RuntimeError("http://private.invalid failed")),
                                  _load_configured_workflow=lambda _: None)
        with mock.patch("backends.get_backend", return_value=backend), mock.patch("backends.get_backend_type", return_value=BackendType.COMFYUI):
            actions._handle_xyz_action("get_xyz_capabilities", {"requestId": "failure"})
            self.assertTrue(actions.vue_bridge.xyzCapabilitiesReceived.ready.wait(3))
            result = actions.vue_bridge.xyzCapabilitiesReceived.values[-1]
            self.assertFalse(result["ok"])
            self.assertEqual(result["axes"], [])
            self.assertNotIn("private.invalid", result["error"])
            actions._handle_xyz_action("start_xyz_plot", {"requestId": "bad", "capabilityId": "stale"})
            self.assertFalse(actions.vue_bridge.xyzPlotEvent.values[-1]["ok"])

    def test_backend_schema_response_can_queue_and_preserve_actual_axis_values(self):
        actions = XYZActionsMixin()
        actions.vue_bridge = SimpleNamespace(xyzCapabilitiesReceived=Signal(), xyzPlotEvent=Signal())
        actions.model_combo = SimpleNamespace(currentText=lambda: "active-model")
        actions._build_generation_payload = lambda **_: ({"prompt": "synthetic", "steps": 10, "alwayson_scripts": {"test": {"args": [True]}}}, None)
        queued = []
        actions.queue_panel = SimpleNamespace(add_single_item=lambda payload: queued.append(payload))
        actions.queue_manager = SimpleNamespace(is_running=True, total_count=0)
        backend = SimpleNamespace(api_url="http://synthetic.invalid", get_object_info=lambda: SCHEMA,
                                  _load_configured_workflow=lambda _: None)
        with mock.patch("backends.get_backend", return_value=backend), mock.patch("backends.get_backend_type", return_value=BackendType.COMFYUI):
            actions._handle_xyz_action("get_xyz_capabilities", {"requestId": "read-one"})
            self.assertTrue(actions.vue_bridge.xyzCapabilitiesReceived.ready.wait(3))
            result = actions.vue_bridge.xyzCapabilitiesReceived.values[-1]
            self.assertTrue(result["ok"], result)
            actions._handle_xyz_action("start_xyz_plot", {"requestId": "plot-one", "capabilityId": result["capabilityId"],
                "axes": [{"id": "steps", "values": [20, 30]}]})
            self.assertEqual([item["steps"] for item in queued], [20, 30])
            payload, model, captured = actions._xyz_prepare_queue_generation(queued[1])
            self.assertEqual(payload["steps"], 30)
            self.assertEqual(model, "active-model")
            self.assertIs(captured, backend)
            self.assertTrue(payload["alwayson_scripts"]["test"]["args"][0])

    def test_the_forge_module_restore_marker_goes_to_webui_only(self):
        """XYZ Text Encoder 축 칸의 표시(core/xyz_capabilities.TRANSIENT_MODULES_KEY)와 칩이 빈 칸의 TE 선택(FORGE_TE_CELL_KEY)은
        Forge 옵션용 — ComfyUI 칸에서는 뗀다."""
        from core.xyz_capabilities import FORGE_TE_CELL_KEY, TRANSIENT_MODULES_KEY, backend_identity

        backend = SimpleNamespace(api_url="http://synthetic.invalid")
        for kind, kept in ((BackendType.WEBUI, True), (BackendType.COMFYUI, False)):
            with self.subTest(kind=kind.value), mock.patch("backends.get_backend", return_value=backend), \
                    mock.patch("backends.get_backend_type", return_value=kind):
                item = {"prompt": "p", "_xyz_model": "m", TRANSIENT_MODULES_KEY: True,
                        FORGE_TE_CELL_KEY: {"value": "te.safetensors", "encoders": ["te.safetensors"]},
                        "_xyz_backend_id": backend_identity(kind.value, backend)}
                payload, _model, _backend = XYZActionsMixin()._xyz_prepare_queue_generation(item)
                self.assertEqual(TRANSIENT_MODULES_KEY in payload, kept)
                self.assertEqual(FORGE_TE_CELL_KEY in payload, kept)
                self.assertIn(TRANSIENT_MODULES_KEY, item, "the queued snapshot is not changed")
                self.assertIn(FORGE_TE_CELL_KEY, item, "the queued snapshot is not changed")

    def _forge_te_actions(self, tes):
        actions = XYZActionsMixin()
        actions.vue_bridge = SimpleNamespace(
            xyzCapabilitiesReceived=Signal(), xyzPlotEvent=Signal(),
            last_widget_property=lambda wid, prop: list(tes) if (wid, prop) == ("te_main_input", "items") else None)
        actions.te_main_input = SimpleNamespace(text=lambda: "")      # 앱 TE 칩 비었음
        actions.model_combo = SimpleNamespace(currentText=lambda: "anima.safetensors")
        actions._build_generation_payload = lambda **_: ({"prompt": "p"}, None)   # 앱 VAE 칩도 비었음 — 모듈 키 없음
        actions.queued = []
        actions.queue_panel = SimpleNamespace(add_single_item=actions.queued.append)
        actions.queue_manager = SimpleNamespace(is_running=True, total_count=0)
        return actions

    def test_forge_without_app_modules_checks_forges_list_before_the_first_cell(self):
        """칩이 빈 Forge: 축 목록을 받는 작업자 스레드가 보통 생성이 쓸 Forge 목록도 읽어 두고(forge_module_base), 시작 클릭이
        그 목록에 TE 가 둘 이상이면 첫 칸 전에 거절한다. 하나면 칸은 목록 없이 TE 선택만 싣는다."""
        from core.xyz_capabilities import FORGE_TE_CELL_KEY, TEXT_ENCODER_CURRENT, text_encoder_axis

        tes = ["qwen_3_06b_base.safetensors", "qwen_alt.safetensors", "clip_l.safetensors"]
        capability = {"axes": [text_encoder_axis(tes, source="forge_additional_modules")], "unsupported": [],
                      "notes": [], "backend": "webui", "backendId": "b", "maxJobs": 256, "capabilityId": "cap-te"}
        axes = [{"id": "text_encoder", "values": [TEXT_ENCODER_CURRENT, "qwen_alt.safetensors"]}]
        for forge_list, error in ((["qwen_image_vae.safetensors", "qwen_3_06b_base.safetensors"], None),
                                  (["qwen_3_06b_base.safetensors", "clip_l.safetensors"], "TE 가 2개")):
            with self.subTest(forge=forge_list):
                actions = self._forge_te_actions(tes)
                backend = SimpleNamespace(api_url="http://forge.invalid", forge_module_base=mock.Mock(return_value=forge_list))
                with mock.patch("backends.get_backend", return_value=backend), \
                        mock.patch("backends.get_backend_type", return_value=BackendType.WEBUI), \
                        mock.patch("ui.xyz_actions.fetch_capabilities", return_value=capability):
                    actions._handle_xyz_action("get_xyz_capabilities", {"requestId": "query"})
                    self.assertTrue(actions.vue_bridge.xyzCapabilitiesReceived.ready.wait(3))
                    self.assertEqual(actions._xyz_capabilities["forge_modules"], forge_list)
                    actions._handle_xyz_action("start_xyz_plot", {"requestId": "plot", "capabilityId": "cap-te", "axes": axes})
                    # 거절은 Forge 의 지금 목록을 다시 읽은 뒤(작업자 스레드) 답한다
                    self.assertTrue(actions.vue_bridge.xyzPlotEvent.wait_for(1))
                event = actions.vue_bridge.xyzPlotEvent.values[-1]
                if error:
                    self.assertFalse(event["ok"])
                    self.assertIn(error, event["error"])
                    self.assertEqual(actions.queued, [], "no cell is queued")
                else:
                    self.assertTrue(event["ok"], event)
                    self.assertEqual([job[FORGE_TE_CELL_KEY]["value"] for job in actions.queued],
                                     [None, "qwen_alt.safetensors"])
                    self.assertTrue(all("forge_additional_modules" not in job for job in actions.queued))

    @staticmethod
    def _forge_te_capability(tes):
        from core.xyz_capabilities import text_encoder_axis

        return {"axes": [text_encoder_axis(tes, source="forge_additional_modules")], "unsupported": [], "notes": [],
                "backend": "webui", "backendId": "b", "maxJobs": 256, "capabilityId": "cap-te"}

    def test_a_refusal_from_the_list_read_with_the_axes_rereads_forge_before_answering(self):
        """축 목록을 받을 때 읽어 둔 Forge 목록(TE 둘)이 시작을 거절하면 Forge 의 지금 목록을 다시 읽고 한 번 더 본다 —
        사용자가 거절 메시지대로 Forge 에서 TE 를 하나만 고르면 '축 새로고침' 없이 다음 클릭이 칸을 넣는다."""
        from core.xyz_capabilities import FORGE_TE_CELL_KEY, TEXT_ENCODER_CURRENT

        tes = ["clip_l.safetensors", "t5xxl.safetensors", "clip_g.safetensors"]
        axes = [{"id": "text_encoder", "values": [TEXT_ENCODER_CURRENT, "clip_g.safetensors"]}]
        forge = ["ae.safetensors", "clip_l.safetensors", "t5xxl.safetensors"]
        reader = mock.Mock(side_effect=lambda: list(forge))
        backend = SimpleNamespace(api_url="http://forge.invalid", forge_module_base=reader)
        actions = self._forge_te_actions(tes)
        events = actions.vue_bridge.xyzPlotEvent
        with mock.patch("backends.get_backend", return_value=backend), \
                mock.patch("backends.get_backend_type", return_value=BackendType.WEBUI), \
                mock.patch("ui.xyz_actions.fetch_capabilities", return_value=self._forge_te_capability(tes)):
            actions._handle_xyz_action("get_xyz_capabilities", {"requestId": "query"})
            self.assertTrue(actions.vue_bridge.xyzCapabilitiesReceived.ready.wait(3))
            self.assertEqual(reader.call_count, 1)
            # Forge 에 아직 TE 가 둘 — 다시 읽어도 둘이라 거절한다
            actions._handle_xyz_action("start_xyz_plot", {"requestId": "p1", "capabilityId": "cap-te", "axes": axes})
            self.assertTrue(events.wait_for(1))
            self.assertEqual(reader.call_count, 2, "the refusal re-reads Forge's list")
            self.assertFalse(events.values[-1]["ok"])
            self.assertEqual(events.values[-1]["requestId"], "p1")
            self.assertIn("TE 가 2개", events.values[-1]["error"])
            self.assertNotIn("다시 읽지 못해", events.values[-1]["error"])
            self.assertEqual(actions.queued, [])
            # 사용자가 Forge 에서 TE 를 하나만 골랐다 — 축 목록은 그대로(새로고침 안 함)
            forge[:] = ["ae.safetensors", "clip_l.safetensors"]
            actions._handle_xyz_action("start_xyz_plot", {"requestId": "p2", "capabilityId": "cap-te", "axes": axes})
            self.assertTrue(events.wait_for(2))
            self.assertEqual(len(events.values), 2, "one answer per click")
            self.assertTrue(events.values[-1]["ok"], events.values[-1])
            self.assertEqual((events.values[-1]["requestId"], events.values[-1]["count"]), ("p2", 2))
            self.assertEqual([job[FORGE_TE_CELL_KEY]["value"] for job in actions.queued], [None, "clip_g.safetensors"])
            self.assertEqual(actions._xyz_capabilities["forge_modules"], forge, "the cache keeps the fresh list")
            # 새 목록이 통과하면 다음 클릭은 다시 읽지 않는다(GUI 스레드에서 바로 답한다)
            actions._handle_xyz_action("start_xyz_plot", {"requestId": "p3", "capabilityId": "cap-te", "axes": axes})
            self.assertEqual(len(events.values), 3)
            self.assertTrue(events.values[-1]["ok"], events.values[-1])
            self.assertEqual(reader.call_count, 3)

    def test_a_failed_reread_refuses_with_the_list_read_with_the_axes_and_says_so(self):
        from core.xyz_capabilities import TEXT_ENCODER_CURRENT

        tes = ["clip_l.safetensors", "t5xxl.safetensors", "clip_g.safetensors"]
        axes = [{"id": "text_encoder", "values": [TEXT_ENCODER_CURRENT, "clip_g.safetensors"]}]
        stale = ["ae.safetensors", "clip_l.safetensors", "t5xxl.safetensors"]
        reader = mock.Mock(side_effect=[list(stale), RuntimeError("offline")])
        backend = SimpleNamespace(api_url="http://forge.invalid", forge_module_base=reader)
        actions = self._forge_te_actions(tes)
        events = actions.vue_bridge.xyzPlotEvent
        with mock.patch("backends.get_backend", return_value=backend), \
                mock.patch("backends.get_backend_type", return_value=BackendType.WEBUI), \
                mock.patch("ui.xyz_actions.fetch_capabilities", return_value=self._forge_te_capability(tes)):
            actions._handle_xyz_action("get_xyz_capabilities", {"requestId": "query"})
            self.assertTrue(actions.vue_bridge.xyzCapabilitiesReceived.ready.wait(3))
            actions._handle_xyz_action("start_xyz_plot", {"requestId": "p1", "capabilityId": "cap-te", "axes": axes})
            self.assertTrue(events.wait_for(1))
        self.assertEqual(reader.call_count, 2)
        self.assertFalse(events.values[-1]["ok"])
        self.assertIn("TE 가 2개", events.values[-1]["error"])
        self.assertIn("다시 읽지 못해", events.values[-1]["error"])
        self.assertEqual(actions.queued, [])
        self.assertEqual(actions._xyz_capabilities["forge_modules"], stale)

    def test_a_reread_does_not_replace_axes_refreshed_meanwhile(self):
        """다시 읽는 사이 축 목록을 새로 받았으면(캐시가 바뀜) 그 캐시를 덮어쓰지 않는다 — 그쪽 목록이 같거나 더 새것이다."""
        from core.xyz_capabilities import TEXT_ENCODER_CURRENT

        tes = ["clip_l.safetensors", "t5xxl.safetensors", "clip_g.safetensors"]
        capability = self._forge_te_capability(tes)
        axes = [{"id": "text_encoder", "values": [TEXT_ENCODER_CURRENT, "clip_g.safetensors"]}]
        actions = self._forge_te_actions(tes)
        actions._xyz_ensure()
        backend = SimpleNamespace(api_url="http://forge.invalid")
        stale = {"data": capability, "backend": backend, "context": actions._xyz_context(),
                 "forge_modules": ["ae.safetensors", "clip_l.safetensors", "t5xxl.safetensors"]}
        refreshed = {**stale, "forge_modules": ["ae.safetensors", "clip_l.safetensors"]}
        def reread():
            actions._xyz_capabilities = refreshed      # 축 새로고침이 먼저 끝났다
            return ["ae.safetensors", "clip_l.safetensors", "t5xxl.safetensors"]
        backend.forge_module_base = reread
        actions._xyz_capabilities = stale
        events = actions.vue_bridge.xyzPlotEvent
        with mock.patch("backends.get_backend", return_value=backend), \
                mock.patch("backends.get_backend_type", return_value=BackendType.WEBUI):
            actions._handle_xyz_action("start_xyz_plot", {"requestId": "p1", "capabilityId": "cap-te", "axes": axes})
            self.assertTrue(events.wait_for(1))
        self.assertIs(actions._xyz_capabilities, refreshed)
        self.assertTrue(events.values[-1]["ok"], events.values[-1])
        self.assertEqual(len(actions.queued), 2)

    def test_the_reread_answer_runs_on_the_window_thread(self):
        """창(QObject) 호스트는 다시 읽은 뒤의 시작을 GUI 스레드로 넘긴다(대기열·위젯은 GUI 스레드에서만)."""
        app = QCoreApplication.instance() or QCoreApplication([])

        class QtHost(QObject, XYZActionsMixin):
            pass

        host = QtHost()
        host._xyz_ensure()
        self.assertIsNotNone(host._xyz_gui_relay)
        ran = []
        worker = threading.Thread(target=lambda: host._xyz_on_gui(lambda: ran.append(threading.current_thread())))
        worker.start()
        worker.join()
        self.assertEqual(ran, [], "queued for the window thread")
        deadline = time.monotonic() + 3
        while not ran and time.monotonic() < deadline:
            app.processEvents()
        self.assertEqual(ran, [threading.main_thread()])
        plain = XYZActionsMixin()
        plain._xyz_ensure()
        self.assertIsNone(plain._xyz_gui_relay)

    def test_forges_list_is_read_only_for_a_forge_text_encoder_axis(self):
        from core.xyz_capabilities import text_encoder_axis

        te_axis = {"axes": [text_encoder_axis(["te.safetensors"], source="t")]}
        reader = SimpleNamespace(forge_module_base=lambda: ["vae.safetensors", "te.safetensors"])
        self.assertEqual(XYZActionsMixin._xyz_forge_modules(reader, "webui", te_axis), ["vae.safetensors", "te.safetensors"])
        self.assertIsNone(XYZActionsMixin._xyz_forge_modules(reader, "comfyui", te_axis))
        self.assertIsNone(XYZActionsMixin._xyz_forge_modules(reader, "webui", {"axes": []}))
        self.assertIsNone(XYZActionsMixin._xyz_forge_modules(SimpleNamespace(), "webui", te_axis))
        failing = SimpleNamespace(forge_module_base=mock.Mock(side_effect=RuntimeError("offline")))
        self.assertIsNone(XYZActionsMixin._xyz_forge_modules(failing, "webui", te_axis), "the cell checks again")

    def test_text_encoder_choices_and_selection_come_from_the_te_chips(self):
        actions = XYZActionsMixin()
        actions.vue_bridge = SimpleNamespace(last_widget_property=lambda wid, prop: (
            ["te_a.safetensors", "", "te_b.safetensors"] if (wid, prop) == ("te_main_input", "items") else None))
        actions.te_main_input = SimpleNamespace(text=lambda: " te_a.safetensors , ")
        self.assertEqual(actions._xyz_text_encoder_choices(), ("te_a.safetensors", "te_b.safetensors"))
        self.assertEqual(actions._xyz_text_encoder_selection(), ["te_a.safetensors"])
        self.assertEqual(actions._xyz_context()["text_encoders"], ("te_a.safetensors", "te_b.safetensors"))
        bare = XYZActionsMixin()      # 위젯이 없는 호스트(테스트 대역·초기화 전) — 축 없음
        self.assertEqual((bare._xyz_text_encoder_choices(), bare._xyz_text_encoder_selection()), ((), []))


if __name__ == "__main__":
    unittest.main()
