"""Schema contracts, graph binding, real compilation and preset recovery."""
import copy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from core.comfy_workflow_controls import (
    WorkflowControlError, apply_controls, clear_workflow_controls,
    controls_for_wire, describe_controls, feature_preflight, load_workflow_controls,
    save_workflow_controls, validate_controls, validate_value, workflow_fingerprint,
)
from core.comfy_workflow_compiler import ComfyWorkflowCompiler, WorkflowCompileError
from core import sam3_args
from tests.test_comfy_workflow_compiler import _capabilities, _custom_workflow
from ui.comfy_workflow_actions import apply_quality_preset, PRESET_FIELDS, quality_preset_payload


def fixture():
    graph, info = _custom_workflow(), _capabilities()
    graph["extra"] = {"class_type": "CustomImageAdjust", "inputs": {
        "image": ["6", 0], "amount": 0.5, "count": 3, "enabled": True,
        "strategy": "safe", "notes": "custom", "secret": "hidden", "unknown": "unknown",
    }}
    graph["7"]["inputs"]["images"] = ["extra", 0]
    info["CustomImageAdjust"] = {"input": {"required": {
        "image": ["IMAGE"], "amount": ["FLOAT", {"min": 0, "max": 1, "step": .1}],
        "count": ["INT", {"min": 0, "max": 18446744073709551615}],
        "enabled": ["BOOLEAN"], "strategy": [["safe", "strong"]],
        "notes": ["STRING", {"multiline": True}], "secret": ["STRING", {"forceInput": True}],
    }}}
    return graph, info


def binding_for(schema, name="amount", value=.75):
    return {"workflowFingerprint": schema["workflowFingerprint"], "schemaFingerprint": schema["schemaFingerprint"],
            "overrides": [{"nodeId": "extra", "classType": "CustomImageAdjust", "name": name, "value": value}]}


class WorkflowControlsTests(unittest.TestCase):
    def setUp(self):
        self.graph, self.info = fixture()
        self.schema = describe_controls(self.graph, self.info)

    def test_schema_excludes_links_and_forced_fields_and_marks_app_ownership(self):
        fields = {f["name"]: f for n in self.schema["nodes"] if n["id"] == "extra" for f in n["fields"]}
        self.assertEqual(set(fields), {"amount", "count", "enabled", "strategy", "notes"})
        self.assertEqual(fields["count"]["max"], "18446744073709551615")
        managed = next(f for n in self.schema["nodes"] for f in n["fields"] if f["name"] == "ckpt_name")
        self.assertTrue(managed["managed"])

    def test_graph_binding_ignores_titles_but_not_input_or_class_changes(self):
        graph = copy.deepcopy(self.graph)
        graph["extra"]["_meta"] = {"title": "Better title"}
        self.assertEqual(workflow_fingerprint(graph), workflow_fingerprint(self.graph))
        graph["extra"]["inputs"]["amount"] = .6
        with self.assertRaisesRegex(WorkflowControlError, "워크플로가 변경"):
            validate_controls(graph, self.info, binding_for(self.schema))

    def test_schema_drift_is_explicit(self):
        self.info["CustomImageAdjust"]["input"]["required"]["amount"][1]["max"] = .6
        with self.assertRaisesRegex(WorkflowControlError, "스키마가 변경"):
            validate_controls(self.graph, self.info, binding_for(self.schema))

    def test_invalid_scalar_values_links_missing_and_duplicate_inputs_rejected(self):
        for name, value in (("amount", "nan"), ("amount", -1), ("amount", True),
                            ("count", 1.5), ("count", "1.0"), ("count", "18446744073709551616"),
                            ("enabled", "true"), ("strategy", "missing"), ("notes", {}), ("image", ["5", 0])):
            with self.subTest(name=name, value=value), self.assertRaises(WorkflowControlError):
                validate_controls(self.graph, self.info, binding_for(self.schema, name, value))
        binding = binding_for(self.schema)
        binding["overrides"] *= 2
        with self.assertRaisesRegex(WorkflowControlError, "중복"):
            validate_controls(self.graph, self.info, binding)

    def test_managed_inputs_cannot_be_overridden(self):
        binding = binding_for(self.schema)
        binding["overrides"] = [{"nodeId": "1", "classType": "CheckpointLoaderSimple", "name": "ckpt_name", "value": "checkpoint.safetensors"}]
        with self.assertRaises(WorkflowControlError):
            validate_controls(self.graph, self.info, binding)

    def test_app_sampler_spectrum_speed_and_cns_controls_cannot_bypass_settings_safety(self):
        self.graph["s"] = {"class_type": "ForgeNeoKSamplerCNS", "inputs": {
            "spectrum_enabled": False, "speed_enabled": False, "spectrum_warmup_steps": 6,
            "cns_enabled": False,
        }}
        self.info["ForgeNeoKSamplerCNS"]["input"]["required"].update({
            "spectrum_enabled": ["BOOLEAN"], "speed_enabled": ["BOOLEAN"],
            "spectrum_warmup_steps": ["INT", {"min": 1, "max": 150}], "cns_enabled": ["BOOLEAN"],
        })
        schema = describe_controls(self.graph, self.info)
        fields = next(node["fields"] for node in schema["nodes"] if node["id"] == "s")
        for field in fields:
            self.assertTrue(field["managed"])
            binding = {"workflowFingerprint": schema["workflowFingerprint"], "schemaFingerprint": schema["schemaFingerprint"],
                       "overrides": [{"nodeId": "s", "name": field["name"], "classType": "ForgeNeoKSamplerCNS", "value": True}]}
            with self.assertRaises(WorkflowControlError):
                validate_controls(self.graph, self.info, binding)
        # A similarly named input in an explicit third-party custom node is
        # still user-owned. Never reserve generic names across all nodes.
        self.graph["extra"]["inputs"]["spectrum_enabled"] = False
        self.info["CustomImageAdjust"]["input"]["required"]["spectrum_enabled"] = ["BOOLEAN"]
        schema = describe_controls(self.graph, self.info)
        self.assertTrue(validate_controls(self.graph, self.info, binding_for(schema, "spectrum_enabled", True))["overrides"][0]["value"])

    def test_uint64_is_exact_in_python_and_js_wire(self):
        binding = validate_controls(self.graph, self.info, binding_for(self.schema, "count", "18446744073709551615"))
        self.assertEqual(binding["overrides"][0]["value"], 18446744073709551615)
        self.assertEqual(controls_for_wire(binding, self.schema)["overrides"][0]["value"], "18446744073709551615")

    def test_compile_applies_custom_input_after_standard_ui_mapping_without_mutation(self):
        before = copy.deepcopy(self.graph)
        result = ComfyWorkflowCompiler(self.info).compile("txt2img", "checkpoint.safetensors", {
            "prompt": "new prompt", "negative_prompt": "new neg", "width": 768, "height": 768,
        }, workflow=self.graph, workflow_controls=binding_for(self.schema))
        self.assertEqual(result["extra"]["inputs"]["amount"], .75)
        self.assertEqual(result["2"]["inputs"]["text"], "new prompt")
        self.assertEqual(result["4"]["inputs"]["width"], 768)
        self.assertEqual(self.graph, before)

    def test_changed_compiled_input_or_class_cannot_be_overwritten(self):
        for compiled in (copy.deepcopy(self.graph), copy.deepcopy(self.graph)):
            compiled["extra"]["inputs"]["amount"] = .1
            with self.assertRaisesRegex(WorkflowControlError, "충돌"):
                apply_controls(compiled, self.graph, self.info, binding_for(self.schema))
        compiled = copy.deepcopy(self.graph)
        compiled["extra"]["inputs"]["amount"] = ["5", 0]
        with self.assertRaisesRegex(WorkflowControlError, "구조가 변경"):
            apply_controls(compiled, self.graph, self.info, binding_for(self.schema))

    def test_copies_of_a_controlled_node_get_the_value_while_they_still_hold_the_original(self):
        """컴파일러가 복제한 노드(예: SAM3 LoRA 분기가 다시 만든 조건 노드)도 같은 워크플로 노드라 같은 값을 받는다.
        컴파일러가 값을 바꾼 복제본·다른 종류의 노드는 건드리지 않고, 원본 워크플로와 넘긴 그래프는 바꾸지 않는다."""
        compiled = copy.deepcopy(self.graph)
        compiled["copy"] = copy.deepcopy(self.graph["extra"])
        compiled["owned"] = copy.deepcopy(self.graph["extra"])
        compiled["owned"]["inputs"]["amount"] = .2
        compiled["other"] = {"class_type": "OtherAdjust", "inputs": {"amount": .5}}
        before, original = copy.deepcopy(compiled), copy.deepcopy(self.graph)
        result = apply_controls(compiled, self.graph, self.info, binding_for(self.schema),
                                copies={"extra": ["copy", "owned", "other", "absent"]})
        self.assertEqual([result[node_id]["inputs"]["amount"] for node_id in ("extra", "copy", "owned", "other")],
                         [.75, .75, .2, .5])
        self.assertEqual((compiled, self.graph), (before, original))
        self.assertEqual(apply_controls(compiled, self.graph, self.info, binding_for(self.schema))["copy"],
                         before["copy"])

    def test_persistence_endpoint_path_and_drift_scoped_and_clear_preserves_json(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = Path(tmp) / "settings.json"
            path = str(Path(tmp) / "workflow.json")
            binding = binding_for(self.schema)
            save_workflow_controls("http://a", path, self.graph, self.info, binding, store_path=store)
            self.assertEqual(load_workflow_controls("http://a/", path, self.graph, store_path=store), binding)
            self.assertIsNone(load_workflow_controls("http://b", path, self.graph, store_path=store))
            changed = copy.deepcopy(self.graph)
            changed["extra"]["inputs"]["count"] = 4
            with self.assertRaises(WorkflowControlError):
                load_workflow_controls("http://a", path, changed, store_path=store)
            clear_workflow_controls("http://a", path, store_path=store)
            self.assertIsNone(load_workflow_controls("http://a", path, changed, store_path=store))
            self.assertEqual(self.graph["extra"]["inputs"]["amount"], .5)

    def test_corrupt_store_is_not_silently_discarded(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = Path(tmp) / "bad.json"
            store.write_text("{bad", encoding="utf-8")
            with self.assertRaises(WorkflowControlError):
                load_workflow_controls("a", "workflow", self.graph, store_path=store)
            self.assertEqual(store.read_text(encoding="utf-8"), "{bad")

    def test_preflight_reports_missing_features_without_running_backend(self):
        info = _capabilities(include_forge=False)
        result = feature_preflight(ComfyWorkflowCompiler(info), "checkpoint.safetensors", {"enable_hr": True})
        self.assertFalse(result["ok"])
        self.assertEqual(next(row for row in result["features"] if row["id"] == "hires")["state"], "blocked")
        result = feature_preflight(ComfyWorkflowCompiler(_capabilities()), "checkpoint.safetensors", {"enable_hr": True})
        self.assertTrue(result["ok"], result)
        self.assertEqual(next(row for row in result["features"] if row["id"] == "hires")["state"], "ready")

    def test_face_then_eye_chains_second_pass_from_first_result(self):
        payload = {"enable_hr": True, "_comfy_detail_passes": ["eyes"]}
        sam3_args.apply_to_payload(payload, {"sam3_prompt": "face", "sam3_mode": "Inpaint"})
        graph = ComfyWorkflowCompiler(_capabilities()).compile("txt2img", "checkpoint.safetensors", payload)
        masks = [(key, node) for key, node in graph.items() if node["class_type"] == "ForgeNeoSAM3Mask"]
        self.assertEqual([node["inputs"]["prompt"] for _, node in masks], ["face", "eyes"])
        first_mask_id = masks[0][0]
        first_detail = next(key for key, node in graph.items() if node["class_type"] == "ForgeNeoSAM3Detailer" and node["inputs"]["mask"] == [first_mask_id, 0])
        self.assertEqual(masks[1][1]["inputs"]["image"], [first_detail, 0])
        self.assertEqual(sum(node["class_type"] == "ForgeNeoHiresFix" for node in graph.values()), 1)
        result = feature_preflight(ComfyWorkflowCompiler(_capabilities()), "checkpoint.safetensors", payload)
        self.assertTrue(result["ok"], result)
        self.assertTrue(any(row["id"] == "sam3_pass_2" and row["state"] == "ready" for row in result["features"]))

    def test_extra_pass_requires_active_sam3_and_valid_targets(self):
        for passes in (["eyes"], [""], "eyes", ["eyes"] * 4):
            with self.subTest(passes=passes), self.assertRaises(WorkflowCompileError):
                ComfyWorkflowCompiler(_capabilities()).compile("txt2img", "checkpoint.safetensors", {"_comfy_detail_passes": passes})

    def test_queued_workflow_controls_are_frozen_before_later_settings_edits(self):
        from backends.comfyui_backend import ComfyUIBackend
        from backends.base import GenerationResult
        from core.comfy_workflow_controls import snapshot_comfy_payload
        from unittest.mock import Mock
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "workflow.json"
            path.write_text(json.dumps(self.graph), encoding="utf-8")
            store = Path(tmp) / "controls.json"
            backend = ComfyUIBackend("http://fixture-comfy", workflow_path=str(path))
            backend._node_pack_preflight_done = True
            backend.get_object_info = Mock(return_value=self.info)
            backend._queue_and_wait = Mock(return_value=GenerationResult(success=True, image_data=b"result"))
            with patch("core.comfy_workflow_controls._PATH", store):
                save_workflow_controls(backend.api_url, str(path), self.graph, self.info, binding_for(self.schema, value=.25))
                queued = snapshot_comfy_payload(backend, {"prompt": "queued"}, "txt2img")
                saved_snapshot = copy.deepcopy(queued)
                save_workflow_controls(backend.api_url, str(path), self.graph, self.info, binding_for(self.schema, value=.9))
                result = backend.txt2img("checkpoint.safetensors", queued)
                self.assertTrue(result.success, result.error)
                self.assertEqual(backend._queue_and_wait.call_args.args[0]["extra"]["inputs"]["amount"], .25)
                self.assertEqual(queued, saved_snapshot)

    def test_queued_absence_of_controls_does_not_pick_up_new_settings(self):
        from core.comfy_workflow_controls import snapshot_comfy_payload, generation_workflow_controls
        with tempfile.TemporaryDirectory() as tmp, patch('core.comfy_workflow_controls._PATH', Path(tmp) / 'controls.json'):
            backend = SimpleNamespace(api_url='http://fixture-comfy',
                _configured_workflow_path=lambda mode: 'fixture.json',
                _load_configured_workflow=lambda mode: self.graph)
            payload = snapshot_comfy_payload(backend, {})
            save_workflow_controls(backend.api_url, 'fixture.json', self.graph, self.info, binding_for(self.schema))
            self.assertIsNone(generation_workflow_controls(backend.api_url, 'fixture.json', self.graph, payload, 'txt2img'))

    def test_queued_controls_revalidate_graph_schema_path_and_server_at_execution(self):
        from backends.comfyui_backend import ComfyUIBackend
        from core.comfy_workflow_controls import snapshot_comfy_payload
        from unittest.mock import Mock
        for changed in ('graph', 'schema', 'server', 'path', 'mode'):
            with self.subTest(changed=changed), tempfile.TemporaryDirectory() as tmp:
                path = Path(tmp) / 'workflow.json'
                path.write_text(json.dumps(self.graph), encoding='utf-8')
                info = copy.deepcopy(self.info)
                backend = ComfyUIBackend('http://fixture-comfy', workflow_path=str(path))
                backend._node_pack_preflight_done = True
                backend.get_object_info = Mock(return_value=info)
                backend._queue_and_wait = Mock()
                with patch('core.comfy_workflow_controls._PATH', Path(tmp) / 'controls.json'):
                    save_workflow_controls(backend.api_url, str(path), self.graph, info, binding_for(self.schema))
                    payload = snapshot_comfy_payload(backend, {'prompt': 'queued'})
                    if changed == 'graph':
                        graph = copy.deepcopy(self.graph)
                        graph['extra']['inputs']['count'] = 4
                        path.write_text(json.dumps(graph), encoding='utf-8')
                    elif changed == 'schema':
                        info['CustomImageAdjust']['input']['required']['amount'][1]['max'] = .5
                    elif changed == 'server':
                        backend.api_url = 'http://other-comfy'
                    elif changed == 'path':
                        backend._configured_workflow_path = lambda mode: str(Path(tmp) / 'different.json')
                    else:
                        payload['_comfy_workflow_snapshot']['mode'] = 'img2img'
                    result = backend.txt2img('checkpoint.safetensors', payload)
                    self.assertFalse(result.success, changed)
                    self.assertTrue(result.error)
                    backend._queue_and_wait.assert_not_called()


class BoolProxy:
    def __init__(self, value): self.value = value
    def isChecked(self): return self.value
    def setChecked(self, value): self.value = value


class TextProxy:
    def __init__(self, value): self.value = value
    def text(self): return self.value
    def setText(self, value): self.value = value


class PresetTests(unittest.TestCase):
    def test_fast_detail_restore_keeps_original_snapshot_and_not_unrelated_fields(self):
        proxies = {key: (BoolProxy(True) if key.endswith('group') or key.endswith('masked') else TextProxy("original")) for key in PRESET_FIELDS}
        proxies["model_combo"] = TextProxy("keep-model")
        host = SimpleNamespace(vue_bridge=SimpleNamespace(_proxies=proxies))
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "preset.json"
            apply_quality_preset(host, "fast", state_path=path)
            backup = json.loads(path.read_text())["backup"]
            self.assertFalse(proxies["hires_options_group"].value)
            apply_quality_preset(host, "detail", "face_then_eyes", endpoint='http://comfy', state_path=path)
            self.assertEqual(json.loads(path.read_text())["backup"], backup)
            self.assertEqual(proxies["_sam3_detect_prompt"].value, "face")
            self.assertEqual(proxies["model_combo"].value, "keep-model")
            payload = {}
            sam3_args.apply_to_payload(payload, {"sam3_prompt": "face", "sam3_mode": "Inpaint"})
            self.assertEqual(quality_preset_payload(payload, host=host, endpoint='http://comfy', state_path=path), {"_comfy_detail_passes": ["eyes"]})
            changed = {}
            sam3_args.apply_to_payload(changed, {"sam3_prompt": "hands", "sam3_mode": "Inpaint"})
            self.assertEqual(quality_preset_payload(changed, host=host, endpoint='http://comfy', state_path=path), {})
            self.assertEqual(quality_preset_payload(payload, host=host, endpoint='http://comfy', state_path=path), {})
            apply_quality_preset(host, "restore", state_path=path)
            for key in PRESET_FIELDS:
                self.assertEqual(proxies[key].value, backup[key])
            self.assertIsNone(json.loads(path.read_text())["backup"])
            self.assertEqual(quality_preset_payload(payload, host=host, endpoint='http://comfy', state_path=path), {})

    def test_invalid_preset_does_not_write_or_mutate(self):
        proxies = {key: TextProxy("original") for key in PRESET_FIELDS}
        host = SimpleNamespace(vue_bridge=SimpleNamespace(_proxies=proxies))
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "preset.json"
            with self.assertRaises(WorkflowControlError):
                apply_quality_preset(host, "danger", state_path=path)
            self.assertFalse(path.exists())
            self.assertTrue(all(proxy.value == "original" for proxy in proxies.values()))


if __name__ == "__main__":
    unittest.main()
