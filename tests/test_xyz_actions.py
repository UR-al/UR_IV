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
