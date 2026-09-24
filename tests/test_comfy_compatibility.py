"""Compatibility is schema evidence, not a generation success certification."""
from __future__ import annotations

import copy
import json
import tempfile
import threading
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from core.comfy_compatibility import (
    RECIPES, REFERENCES, check_recipes,
    compare_references, inspect_compatibility, save_baseline,
)
from ui.comfy_compatibility_actions import ComfyCompatibilityActionsMixin


def schema_fixture():
    schema = {}
    for recipe in RECIPES:
        for name, fields in recipe["nodes"].items():
            schema.setdefault(name, {"input": {"required": {}}, "output": []})
            schema[name]["input"]["required"].update({key: [["choice"]] if kind == "CHOICE" else [kind] for key, kind in fields.items()})
    for node, field in [("UNETLoader", "unet_name"), ("CLIPLoader", "clip_name"), ("VAELoader", "vae_name")]:
        schema[node] = {"input": {"required": {field: [["example.safetensors"]]}}}
    return schema


class CompatibilityTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name) / "baseline.json"
        self.schema = schema_fixture()

    def report(self, endpoint="http://127.0.0.1:8188", runtime=None):
        return inspect_compatibility(endpoint, runtime_snapshot=runtime or {}, baseline_path=self.path,
            fetch=lambda url, path, **kw: self.schema if path == "/object_info" else {"system": {"comfyui_version": "0.21.1"}})

    def test_offline_never_claims_compatible(self):
        for recipe in check_recipes(None):
            self.assertEqual(recipe["status"], "unknown")

    def test_matching_input_types_are_available_not_generation_verified(self):
        results = {item["id"]: item for item in check_recipes(self.schema)}
        self.assertEqual(results["forge-parity"]["status"], "available")
        self.assertEqual(results["spectrum"]["status"], "available")
        self.assertEqual(results["sam3"]["status"], "unknown")  # explicit path cannot prove weights exist

    def test_missing_nodes_and_type_drift_are_identified(self):
        self.schema.pop("DiTSpectrumPatch")
        self.schema["ForgeNeoKSamplerCNS"]["input"]["required"]["steps"] = ["STRING"]
        results = {item["id"]: item for item in check_recipes(self.schema)}
        self.assertEqual(results["spectrum"]["status"], "missing")
        self.assertEqual(results["forge-parity"]["status"], "missing")
        sampler = next(x for x in results["forge-parity"]["checks"] if x["label"] == "ForgeNeoKSamplerCNS")
        self.assertIn("steps: INT", sampler["detail"])

    def test_empty_model_choices_do_not_pass(self):
        self.schema["UNETLoader"]["input"]["required"]["unet_name"] = [[]]
        self.assertEqual(check_recipes(self.schema)[0]["status"], "missing")

    def test_local_revision_must_not_be_assigned_to_remote_server(self):
        runtime = {"engines": {"comfyui": {"running": True, "apiUrl": "http://127.0.0.1:8188", "version": "wrong-version",
            "extensions": [{"repoUrl": REFERENCES[1]["repoUrl"], "commit": REFERENCES[1]["commit"]}]}}}
        report = self.report(endpoint="https://comfy.example.test", runtime=runtime)
        self.assertFalse(report["localRevisionKnown"])
        self.assertEqual(report["references"][1]["status"], "unknown")
        report = self.report(runtime=runtime)
        self.assertTrue(report["localRevisionKnown"])
        self.assertEqual(report["references"][1]["status"], "same")

    def test_stopped_runtime_revision_is_not_attributed_to_other_process(self):
        report = self.report(runtime={"engines": {"comfyui": {"running": False, "apiUrl": "http://127.0.0.1:8188"}}})
        self.assertFalse(report["localRevisionKnown"])

    def test_upstream_reference_difference_is_not_called_incompatible(self):
        refs = compare_references({}, "99.1.0")
        self.assertEqual(refs[0]["status"], "different")
        self.assertTrue(all(item["repoUrl"].startswith("https://github.com/") for item in REFERENCES))

    def test_forge_parity_recipe_checks_nodes_the_default_compiler_builds(self):
        # 기본 컴파일러의 출력 노드는 save_images 를 따른다 — 메인 생성(True)은 코어
        # SaveImage, 그 밖의 경로는 코어 PreviewImage. 레시피는 둘 다 확인해야 한다.
        from core.comfy_workflow_compiler import ComfyWorkflowCompiler

        recipe = next(item for item in RECIPES if item["id"] == "forge-parity")
        self.assertNotIn("ForgeNeoSaveImage", recipe["nodes"])
        for save_images, expected in ((True, "SaveImage"), (False, "PreviewImage")):
            with self.subTest(save_images=save_images):
                graph = ComfyWorkflowCompiler().compile(
                    "txt2img", "checkpoint.safetensors",
                    {"prompt": "portrait", "save_images": save_images},
                )
                classes = {node["class_type"] for node in graph.values()}
                self.assertIn(expected, classes)
                self.assertIn(expected, recipe["nodes"])

    def test_explicit_baseline_round_trip_and_schema_drift(self):
        report = self.report()
        self.assertFalse(self.path.exists())
        baseline = save_baseline(report, path=self.path)
        self.assertTrue(baseline["exists"])
        self.assertFalse(self.report()["baseline"]["drift"])
        self.schema["ForgeNeoKSamplerCNS"]["input"]["required"]["steps"] = ["INT", {"max": 50}]
        report = self.report()
        self.assertIn("ForgeNeoKSamplerCNS", [item["field"] for item in report["baseline"]["drift"]])
        saved = self.path.read_text()
        self.assertNotIn("http://", saved)
        self.assertNotIn("example.safetensors", saved)

    def test_cannot_save_offline_or_corrupt_baseline_overwrite(self):
        with self.assertRaises(ValueError):
            save_baseline({"connected": False}, path=self.path)
        # Test fixture only. Production edits always use atomic JSON publication.
        self.path.write_text("broken baseline", encoding="utf-8")
        report = self.report()
        self.assertTrue(any("손상" in item for item in report["warnings"]))
        with self.assertRaises(ValueError):
            save_baseline(report, path=self.path)
        self.assertEqual(self.path.read_text(), "broken baseline")

    def test_offline_fetch_errors_return_unknown_recipes_and_no_raw_url(self):
        def fail(*args, **kwargs):
            raise ValueError("secret credential url here")
        report = inspect_compatibility("https://user:secret@example.test", runtime_snapshot={}, fetch=fail, baseline_path=self.path)
        self.assertFalse(report["connected"])
        self.assertNotIn("secret", json.dumps(report))


class Host(ComfyCompatibilityActionsMixin):
    def __init__(self, *, web=False):
        self.web_mode = web
        self.vue_bridge = SimpleNamespace(comfyCompatibilityResult=mock.Mock())
        self.url = "http://127.0.0.1:8188"
        self._compatibility_inspector = mock.Mock(return_value={"ok": True, "connected": True})
        self._compatibility_baseline_saver = mock.Mock(return_value={"exists": True})

    def _compatibility_url(self):
        return self.url


class CompatibilityActionTests(unittest.TestCase):
    def test_web_client_gets_no_local_information_or_network_request(self):
        host = Host(web=True)
        self.assertTrue(host._handle_comfy_compatibility_action("comfy_compatibility_refresh", {}))
        host._compatibility_inspector.assert_not_called()
        host.vue_bridge.comfyCompatibilityResult.emit.assert_not_called()

    def test_async_action_ignores_arbitrary_payload_urls_and_baseline_requires_explicit_action(self):
        host = Host()
        host._handle_comfy_compatibility_action("comfy_compatibility_refresh", {"requestId": "one", "apiUrl": "http://attacker"})
        host._compatibility_worker.join(2)
        host._compatibility_inspector.assert_called_once_with(host.url)
        host._compatibility_baseline_saver.assert_not_called()
        result = json.loads(host.vue_bridge.comfyCompatibilityResult.emit.call_args.args[0])
        self.assertEqual(result["requestId"], "one")
        host._handle_comfy_compatibility_action("comfy_compatibility_save_baseline", {"requestId": "two"})
        host._compatibility_worker.join(2)
        host._compatibility_baseline_saver.assert_called_once()

    def test_duplicate_requests_are_bounded_and_stale_endpoint_cannot_save(self):
        host = Host()
        started, release = threading.Event(), threading.Event()
        def inspect(url):
            started.set()
            release.wait(2)
            return {"ok": True, "connected": True}
        host._compatibility_inspector = mock.Mock(side_effect=inspect)
        host._handle_comfy_compatibility_action("comfy_compatibility_save_baseline", {"requestId": "first"})
        self.assertTrue(started.wait(1))
        host._handle_comfy_compatibility_action("comfy_compatibility_refresh", {"requestId": "second"})
        host.url = "http://127.0.0.1:9999"
        release.set()
        host._compatibility_worker.join(2)
        host._compatibility_inspector.assert_called_once()
        host._compatibility_baseline_saver.assert_not_called()
        messages = [json.loads(call.args[0]) for call in host.vue_bridge.comfyCompatibilityResult.emit.call_args_list]
        self.assertTrue(all(item["ok"] is False for item in messages))

    def test_shutdown_prevents_baseline_save_and_emission(self):
        host = Host()
        host._shutdown_comfy_compatibility()
        host._handle_comfy_compatibility_action("comfy_compatibility_save_baseline", {})
        host._compatibility_inspector.assert_not_called()
        host._compatibility_baseline_saver.assert_not_called()
        host.vue_bridge.comfyCompatibilityResult.emit.assert_not_called()


if __name__ == "__main__":
    unittest.main()
