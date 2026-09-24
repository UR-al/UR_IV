"""Public action flow with mocked Comfy, real compiler/store, no GPU or server."""
import json
import copy
from pathlib import Path
import tempfile
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from core import comfy_workflow_controls
from tests.test_comfy_workflow_controls import fixture, binding_for
from tests.test_model_download_actions import Signal
from ui.comfy_workflow_actions import ComfyWorkflowActionsMixin


class PresetSessionTests(unittest.TestCase):
    def setUp(self):
        from PyQt6.QtCore import QObject
        from ui.vue_bridge import VueBridge
        from ui.widget_proxies import CheckBoxProxy, LineEditProxy
        from ui.comfy_workflow_actions import PRESET_FIELDS

        class Owner(QObject, ComfyWorkflowActionsMixin):
            pass

        self.owner = Owner()
        self.owner.vue_bridge = VueBridge(self.owner)
        for key in PRESET_FIELDS:
            cls = CheckBoxProxy if key.endswith(('group', 'masked')) else LineEditProxy
            cls(self.owner.vue_bridge, key)
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name) / 'preset.json'
        patcher = patch('ui.comfy_workflow_actions._PRESET_PATH', self.path)
        patcher.start()
        self.addCleanup(patcher.stop)
        self.endpoint = 'http://fixture-comfy'

    def apply(self):
        from ui.comfy_workflow_actions import apply_quality_preset
        apply_quality_preset(self.owner, 'detail', 'face_then_eyes', endpoint=self.endpoint)

    def payload(self, host=None, endpoint=None):
        from core import sam3_args
        from ui.comfy_workflow_actions import quality_preset_payload
        payload = {}
        sam3_args.apply_to_payload(payload, {'sam3_prompt': 'face', 'sam3_mode': 'Inpaint'})
        return quality_preset_payload(payload, host=host or self.owner, endpoint=endpoint or self.endpoint)

    def test_user_round_trip_changes_disarm_until_explicit_reapply(self):
        for widget, changed, original in (
            ('_sam3_detect_prompt', 'hands', 'face'),
            ('_sam3_mode', 'Mask Only', 'Inpaint'),
            ('sam3_group', False, True),
        ):
            with self.subTest(widget=widget):
                self.apply()
                backup = json.loads(self.path.read_text())['backup']
                self.assertEqual(self.payload(), {'_comfy_detail_passes': ['eyes']})
                bridge = self.owner.vue_bridge
                bridge.onWidgetChanged(widget, changed)
                bridge.onWidgetChanged(widget, original)
                self.assertEqual(self.payload(), {})
                state = json.loads(self.path.read_text())
                self.assertFalse(state['armed'])
                self.assertEqual(state['backup'], backup)
                self.apply()
                self.assertEqual(self.payload(), {'_comfy_detail_passes': ['eyes']})

    def test_new_window_and_other_endpoint_cannot_inherit_extra_pass(self):
        self.apply()
        self.assertEqual(self.payload(host=SimpleNamespace()), {})
        self.assertEqual(self.payload(endpoint='http://other-comfy'), {})
        self.assertEqual(self.payload(endpoint=self.endpoint + '/'), {'_comfy_detail_passes': ['eyes']})
        self.owner.vue_bridge.onWidgetChanged('sam3_group', True)
        self.owner.vue_bridge.onWidgetChanged('_sam3_detect_prompt', 'face')
        self.assertEqual(self.payload(), {'_comfy_detail_passes': ['eyes']})


class LegacyQueueControlTests(unittest.TestCase):
    def test_search_exif_queue_freezes_scalar_overrides_but_keeps_other_legacy_options(self):
        from unittest.mock import Mock, MagicMock
        from backends import BackendType
        from tests.test_chat_generation_snapshot import SnapshotHost, ReadOnlyWidget
        from ui.generator_main import GeneratorMainUI

        class MutableWidget(ReadOnlyWidget):
            def setText(self, value): self.value = str(value)
            setPlainText = setText

        host = SnapshotHost()
        for key, widget in vars(host).copy().items():
            if isinstance(widget, ReadOnlyWidget):
                setattr(host, key, MutableWidget(widget.value))
        host.random_res_check.value = False
        host.hires_options_group.value = False
        host.prompt_settings_extras.wildcard_enabled = False
        host._on_generation_requested = lambda item: GeneratorMainUI._on_generation_requested(host, item)
        host._apply_payload_to_ui = lambda item: GeneratorMainUI._apply_payload_to_ui(host, item)
        for name in ('_on_queue_completed', '_sync_queue_to_vue', '_sync_queue_item_added', '_abort_generation'):
            setattr(host, name, Mock())
        host.start_generation = Mock(return_value=True)
        panel = MagicMock()
        panel.queue_items = []
        panel.add_single_item.side_effect = lambda item: panel.queue_items.append(copy.deepcopy(item))
        graph, info = fixture()
        backend = SimpleNamespace(api_url='http://fixture-comfy',
            _configured_workflow_path=lambda mode: 'fixture.json',
            _load_configured_workflow=lambda mode: graph)
        with tempfile.TemporaryDirectory() as tmp, \
             patch('core.comfy_workflow_controls._PATH', Path(tmp) / 'controls.json'), \
             patch('backends.get_backend', return_value=backend), \
             patch('backends.get_backend_type', return_value=BackendType.COMFYUI), \
             patch('core.config_migration.load_ui_prefs', return_value={}), \
             patch('ui.generator_main.QueuePanel', return_value=panel), \
             patch('ui.generator_main.QueueManager'):
            schema = comfy_workflow_controls.describe_controls(graph, info)
            binding = binding_for(schema, value=.25)
            comfy_workflow_controls.save_workflow_controls(backend.api_url, 'fixture.json', graph, info, binding)
            GeneratorMainUI._setup_queue(host)
            original = {'prompt': 'queued search/exif prompt', 'steps': 32}
            panel.add_single_item(original)
            item = panel.queue_items[0]
            self.assertIn('_comfy_queued_controls', item)
            self.assertNotIn('_comfy_queued_controls', original)
            comfy_workflow_controls.save_workflow_controls(backend.api_url, 'fixture.json', graph, info, binding_for(schema, value=.9))
            host.cfg_input.value = '6.5'  # Non-snapshotted legacy option still follows UI.
            host._on_generation_requested(item)
            host._abort_generation.assert_not_called()
            payload = host.start_generation.call_args.kwargs['payload_override']
            resolved = comfy_workflow_controls.generation_workflow_controls(backend.api_url, 'fixture.json', graph, payload, 'txt2img')
            self.assertEqual(resolved, binding)
            self.assertEqual(payload['steps'], 32)
            self.assertEqual(payload['cfg_scale'], 6.5)
            self.assertTrue(payload['prompt'].startswith(original['prompt']))


class ComfyWorkflowActionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.graph, self.info = fixture()
        self.backend = SimpleNamespace(
            api_url="http://fixture-comfy", _configured_workflow_path=lambda mode: "fixture.json",
            _load_configured_workflow=lambda mode: self.graph, get_object_info=lambda: self.info)
        self.signal = Signal()
        self.host = ComfyWorkflowActionsMixin()
        self.host.vue_bridge = SimpleNamespace(comfyWorkflowEvent=self.signal)
        self.host.model_combo = SimpleNamespace(currentText=lambda: "checkpoint.safetensors")
        thread_id = threading.get_ident()
        def snapshot(**kwargs):
            self.assertEqual(threading.get_ident(), thread_id, "WidgetProxy read in background thread")
            self.assertEqual(kwargs, {"snapshot": True})
            return {"prompt": "checked"}, None
        self.host._build_generation_payload = snapshot
        for patcher in (
            patch("backends.get_backend", return_value=self.backend),
            patch("backends.get_backend_type", return_value=SimpleNamespace(value="comfyui")),
            patch.object(comfy_workflow_controls, "_PATH", Path(self.temp.name) / "controls.json"),
        ):
            patcher.start()
            self.addCleanup(patcher.stop)
        self.addCleanup(self.host._shutdown_comfy_workflow_controls)

    def send(self, action, **payload):
        request = f"request-{len(self.signal.events)}"
        self.host._handle_comfy_workflow_action(action, {"requestId": request, **payload})
        return self.signal.wait_for(lambda event: event["requestId"] == request and not event.get("busy"))

    def test_inspect_save_preflight_clear_end_to_end(self):
        event = self.send("comfy_controls_inspect")
        self.assertTrue(event["ok"], event)
        self.assertIsNone(event["binding"])
        binding = binding_for(event["schema"])
        event = self.send("comfy_controls_save", binding=binding)
        self.assertTrue(event["ok"], event)
        self.assertTrue(event["saved"])
        event = self.send("comfy_controls_inspect")
        self.assertEqual(event["binding"], binding)
        event = self.send("comfy_feature_preflight")
        self.assertTrue(event["preflight"]["ok"], event)
        self.assertTrue(self.send("comfy_controls_clear")["cleared"])
        self.assertIsNone(self.send("comfy_controls_inspect")["binding"])

    def test_drift_is_shown_and_cannot_silently_apply(self):
        schema = self.send("comfy_controls_inspect")["schema"]
        self.assertTrue(self.send("comfy_controls_save", binding=binding_for(schema))["ok"])
        self.graph["extra"]["inputs"]["count"] = 123
        event = self.send("comfy_controls_inspect")
        self.assertTrue(event["ok"])
        self.assertTrue(event["warning"])
        self.assertIsNone(event["binding"])
        self.assertFalse(self.send("comfy_feature_preflight")["ok"])

    def test_web_mode_is_refused_and_unrelated_action_not_consumed(self):
        self.assertFalse(self.host._handle_comfy_workflow_action("unrelated", {}))
        self.host.web_mode = True
        event = self.send("comfy_controls_inspect")
        self.assertFalse(event["ok"])
        self.assertIn("로컬", event["error"])

    def test_concurrent_request_is_bounded_and_shutdown_suppresses_late_result(self):
        entered, release = threading.Event(), threading.Event()
        def slow_info():
            entered.set()
            release.wait(2)
            return self.info
        self.backend.get_object_info = slow_info
        self.host._handle_comfy_workflow_action("comfy_controls_inspect", {"requestId": "first"})
        self.assertTrue(entered.wait(2))
        self.host._handle_comfy_workflow_action("comfy_controls_inspect", {"requestId": "second"})
        self.assertTrue(self.signal.wait_for(lambda event: event["requestId"] == "second")["busy"])
        self.assertTrue(self.host._comfy_controls_busy)
        self.host._handle_comfy_workflow_action("comfy_controls_inspect", {"requestId": "first"})
        self.assertTrue(self.host._comfy_controls_busy, "duplicate IDs cannot unlock an active request")
        self.host._shutdown_comfy_workflow_controls()
        before = len(self.signal.events)
        release.set()
        # Join our bounded worker before cleanup restores mocked functions.
        for thread in threading.enumerate():
            if thread.name == "comfy-workflow-controls":
                thread.join(3)
        self.assertEqual(len(self.signal.events), before)


if __name__ == "__main__":
    unittest.main()
