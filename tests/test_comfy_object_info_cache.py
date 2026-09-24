"""ComfyUI /object_info is not re-downloaded for every compile pass (감사 #184).

No real ComfyUI is contacted: get_object_info is stubbed.
"""
from __future__ import annotations

import base64
import io
import unittest
from unittest import mock

from PIL import Image

from backends.base import GenerationResult
from backends.comfyui_backend import ComfyUIBackend
from core import sam3_args
from core.comfy_object_info_cache import ObjectInfoCache, ObjectInfoCacheRegistry
from core.comfy_workflow_compiler import ComfyWorkflowCompiler, WorkflowCompileError
from tests.test_comfy_workflow_compiler import _capabilities, _choice


def _png() -> str:
    buffer = io.BytesIO()
    Image.new("RGB", (32, 32)).save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("ascii")


class _Clock:
    def __init__(self):
        self.now = 100.0

    def __call__(self):
        return self.now


class ObjectInfoCacheTests(unittest.TestCase):
    def test_hit_within_ttl_miss_after_ttl_url_change_or_invalidate(self):
        clock = _Clock()
        cache = ObjectInfoCache(ttl=60, clock=clock)
        fetch = mock.Mock(side_effect=lambda: {"n": fetch.call_count})
        first, cached = cache.get("http://a/", fetch)
        self.assertEqual((first, cached), ({"n": 1}, False))
        self.assertEqual(cache.get("http://a", fetch), ({"n": 1}, True))  # trailing slash ignored
        clock.now += 61
        self.assertEqual(cache.get("http://a", fetch), ({"n": 2}, False))
        self.assertEqual(cache.get("http://b", fetch), ({"n": 3}, False))
        cache.invalidate()
        self.assertEqual(cache.get("http://b", fetch), ({"n": 4}, False))
        self.assertEqual(cache.get("http://b", fetch, fresh=True), ({"n": 5}, False))

    def test_failed_or_non_dict_fetch_is_not_cached(self):
        cache = ObjectInfoCache()
        with self.assertRaises(RuntimeError):
            cache.get("http://a", mock.Mock(side_effect=RuntimeError("down")))
        self.assertIsNone(cache.peek("http://a"))
        self.assertEqual(cache.get("http://a", lambda: None), (None, False))
        self.assertIsNone(cache.peek("http://a"))

    def test_expired_snapshot_is_released(self):
        clock = _Clock()
        cache = ObjectInfoCache(ttl=60, clock=clock)
        cache.store("http://a", {"big": True})
        self.assertFalse(cache.expired())
        clock.now += 61
        self.assertIsNone(cache.peek("http://a"))
        self.assertIsNone(cache._data)  # the multi-MB document is not kept alive
        self.assertTrue(cache.expired())


class ObjectInfoCacheRegistryTests(unittest.TestCase):
    """Per-endpoint snapshots shared by the per-job Generation API backends."""

    def test_one_cache_per_url_shared_across_lookups(self):
        registry = ObjectInfoCacheRegistry()
        first = registry.for_url("http://a/")
        self.assertIs(registry.for_url("http://a"), first)
        self.assertIsNot(registry.for_url("http://b"), first)

    def test_least_recently_used_and_expired_endpoints_are_dropped(self):
        clock = _Clock()
        registry = ObjectInfoCacheRegistry(max_urls=2, ttl=60, clock=clock)
        for url in ("http://a", "http://b"):
            registry.for_url(url).store(url, {"url": url})
        registry.for_url("http://a")  # a is now the most recent
        registry.for_url("http://c").store("http://c", {"url": "c"})
        self.assertEqual(len(registry), 2)
        self.assertEqual(registry.for_url("http://a").peek("http://a"), {"url": "http://a"})
        self.assertIsNone(registry.for_url("http://b").peek("http://b"))  # evicted: a new, empty cache
        clock.now += 61
        registry.for_url("http://d")  # expired snapshots are dropped on use
        self.assertEqual(len(registry), 1)

    def test_invalidate_one_or_all(self):
        registry = ObjectInfoCacheRegistry()
        for url in ("http://a", "http://b"):
            registry.for_url(url).store(url, {})
        registry.invalidate("http://a/")
        self.assertIsNone(registry.for_url("http://a").peek("http://a"))
        self.assertEqual(registry.for_url("http://b").peek("http://b"), {})
        registry.invalidate()
        self.assertIsNone(registry.for_url("http://b").peek("http://b"))


class BackendSnapshotTests(unittest.TestCase):
    def _backend(self, info=None):
        backend = ComfyUIBackend("http://127.0.0.1:1", workflow_path="", img2img_workflow_path="")
        backend._preflight_bundled_node_pack = mock.Mock()
        self.fetches = []

        def fake_fetch(_self=backend):
            data = info if info is not None else _capabilities()
            self.fetches.append(1)
            _self._object_info_cache.store(_self.api_url, data)
            return data

        backend.get_object_info = fake_fetch
        backend._queue_and_wait = mock.Mock(return_value=GenerationResult(success=True, image_data=b"img"))
        backend._upload_image = mock.Mock(return_value="input_new.png")
        return backend

    def test_generation_and_its_postprocess_passes_share_one_schema_fetch(self):
        backend = self._backend()
        self.assertTrue(backend.txt2img("checkpoint.safetensors", {"prompt": "a"}).success)
        backend.adetailer(_png(), {"ad_model": "face_yolov8n.pt"})
        backend.sam3(_png(), {"sam3_prompt": "face", "sam3_mode": "Inpaint"})
        backend.upscale(_png(), {"upscaler_name": "Lanczos"})
        self.assertEqual(len(self.fetches), 1)

    def test_live_fetch_refreshes_and_node_pack_restart_invalidates(self):
        backend = self._backend()
        backend.get_info()  # connect: a live fetch that seeds the snapshot
        backend.txt2img("checkpoint.safetensors", {"prompt": "a"})
        self.assertEqual(len(self.fetches), 1)
        backend._object_info_cache.invalidate()  # what a node-pack install/restart does
        backend.txt2img("checkpoint.safetensors", {"prompt": "a"})
        self.assertEqual(len(self.fetches), 2)

    def test_uploaded_input_newer_than_the_snapshot_still_validates(self):
        backend = self._backend()
        backend.txt2img("checkpoint.safetensors", {"prompt": "a"})  # snapshot without the upload
        capabilities = _capabilities()
        capabilities["LoadImage"]["input"]["required"]["image"] = _choice("old.png")
        backend._object_info_cache.store(backend.api_url, capabilities)
        result = backend.img2img("checkpoint.safetensors", {"init_images": [_png()]})
        self.assertTrue(result.success, result.error)
        self.assertEqual(len(self.fetches), 1)

    def test_compile_error_on_a_cached_schema_is_retried_once_with_a_fresh_one(self):
        stale = _capabilities()
        fresh = _capabilities()
        fresh["LoraLoader"]["input"]["required"]["lora_name"] = _choice("styles/new.safetensors")
        backend = self._backend()
        backend._object_info_cache.store(backend.api_url, stale)  # snapshot predates the LoRA
        backend.get_object_info = mock.Mock(side_effect=lambda: (
            backend._object_info_cache.store(backend.api_url, fresh) or fresh
        ))
        result = backend.txt2img("checkpoint.safetensors", {"prompt": "a <lora:new:0.5>"})
        self.assertTrue(result.success, result.error)
        backend.get_object_info.assert_called_once_with()

    def test_errors_on_a_fresh_schema_are_not_retried(self):
        backend = self._backend()
        result = backend.txt2img("checkpoint.safetensors", {"prompt": "a <lora:missing:1>"})
        self.assertFalse(result.success)
        self.assertIn("missing", result.error)
        self.assertEqual(len(self.fetches), 1)


class ValidateUploadTests(unittest.TestCase):
    def test_only_the_uploaded_names_bypass_load_image_choices(self):
        capabilities = _capabilities()
        capabilities["LoadImage"]["input"]["required"]["image"] = _choice("old.png")
        compiler = ComfyWorkflowCompiler(capabilities)
        graph = {"1": {"class_type": "LoadImage", "inputs": {"image": "input_new.png"}}}
        compiler.validate(graph, uploaded_inputs=("input_new.png", ""))
        with self.assertRaisesRegex(WorkflowCompileError, "허용되지 않은 선택"):
            compiler.validate(graph)
        with self.assertRaisesRegex(WorkflowCompileError, "허용되지 않은 선택"):
            compiler.validate({"1": {"class_type": "LoadImage", "inputs": {"image": "other.png"}}},
                              uploaded_inputs=("input_new.png",))

    def test_sam3_mask_only_compiles_against_a_snapshot_without_the_upload(self):
        capabilities = _capabilities()
        capabilities["LoadImage"]["input"]["required"]["image"] = _choice("old.png")
        graph = ComfyWorkflowCompiler(capabilities).compile_sam3_mask_only(
            {"alwayson_scripts": sam3_args.build_alwayson({"sam3_mode": "Mask only"})},
            uploaded_image="input_new.png",
        )
        self.assertIn("input_new.png", [node["inputs"].get("image") for node in graph.values()])


if __name__ == "__main__":
    unittest.main()
