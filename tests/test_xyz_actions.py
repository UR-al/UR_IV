import json
import threading
import unittest
from types import SimpleNamespace
from unittest import mock

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
