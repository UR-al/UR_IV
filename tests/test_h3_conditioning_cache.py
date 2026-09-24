"""Public cache and two-stage H3 workflow contracts, without model loading."""
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from core.creator_workflows import build
from tests._optional_deps import load_torch, requires_torch


def _torch(_test):
    """CPU 텐서 캐시 검사용 지연 import — @requires_torch 로 표시한 테스트만 부른다.

    (torch 미설치면 SkipTest. --quick 은 표시된 테스트를 걸러 torch 를 올리지 않는다.)
    """
    return load_torch()


class H3CacheWorkflowTests(unittest.TestCase):
    def test_cached_i2v_encodes_before_sampling_without_loading_te_in_sample(self):
        built = build("h3_i2v", {"prompt": "synthetic", "input_image": "start.png",
                                     "conditioning_cache": True})
        self.assertEqual([stage["name"] for stage in built["stages"]], ["encode", "sample"])
        encode, sample = [stage["workflow"] for stage in built["stages"]]
        self.assertNotIn("UNETLoader", {n["class_type"] for n in encode.values()})
        self.assertTrue({"CLIPLoader", "LoadImage"}.isdisjoint(
            n["class_type"] for n in sample.values()))
        self.assertEqual(sample["6"]["class_type"], "ForgeNeoH3ConditioningCacheLoad")
        self.assertEqual(sample["11"]["inputs"]["latent_image"], ["17", 0])
        self.assertEqual(encode["90"]["class_type"], "ForgeNeoH3ConditioningCachePrepare")
        json.dumps(built)

    @requires_torch
    def test_cache_public_store_round_trip_uses_cpu_tensors(self):
        from comfy_custom_nodes.ai_studio_forge_parity.h3_cache_nodes import ConditioningCache
        torch = _torch(self)
        with tempfile.TemporaryDirectory() as tmp:
            store = ConditioningCache(Path(tmp) / "h3", max_bytes=1024 ** 2, max_entries=2)
            value = [[torch.tensor([[1.0, 2.0]], requires_grad=True), {"ref": torch.tensor([3])}]]
            store.put("a" * 64, value)
            loaded = store.get("a" * 64)
            self.assertEqual(loaded[0][0].tolist(), [[1.0, 2.0]])
            self.assertEqual(str(loaded[0][0].device), "cpu")
            self.assertFalse(loaded[0][0].requires_grad)
            self.assertEqual(store.stats()["entries"], 1)

    @requires_torch
    def test_comfy_prepare_hit_does_not_request_lazy_encoder_inputs(self):
        from comfy_custom_nodes.ai_studio_forge_parity.h3_cache_nodes import (
            ForgeNeoH3ConditioningCachePrepare, ForgeNeoH3ConditioningCacheLoad,
        )
        torch = _torch(self)
        built = build("h3_t2v", {"prompt": "synthetic", "conditioning_cache": True})
        descriptor = built["stages"][0]["workflow"]["90"]["inputs"]["descriptor"]
        with tempfile.TemporaryDirectory() as tmp:
            model = Path(tmp) / "model.safetensors"
            model.write_bytes(b"synthetic model content")
            folder_paths = SimpleNamespace(get_full_path=lambda *_: str(model),
                get_output_directory=lambda: str(Path(tmp) / "output"),
                get_input_directory=lambda: str(Path(tmp) / "input"), base_path=tmp)
            calls = []

            class FakeBaseModel:  # comfy.model_base.BaseModel 계약만 흉내 낸다
                pass

            diffusion = SimpleNamespace(model=SimpleNamespace(model=FakeBaseModel()))
            text_encoder = SimpleNamespace(model=SimpleNamespace(model=object()))
            manager = SimpleNamespace(processing_interrupted=lambda: False,
                unload_all_models=lambda: calls.append("unload"), soft_empty_cache=lambda: calls.append("empty"),
                current_loaded_models=[diffusion, text_encoder],
                get_all_torch_devices=lambda: ["cuda:0"],
                free_memory=lambda memory, device, keep_loaded=(): calls.append(
                    ("free", device, tuple(keep_loaded))))
            with mock.patch.dict("sys.modules", {
                "folder_paths": folder_paths, "comfy.model_management": manager,
                "comfy.model_base": SimpleNamespace(BaseModel=FakeBaseModel),
            }):
                first = ForgeNeoH3ConditioningCachePrepare()
                self.assertEqual(first.check_lazy_status(descriptor), ["conditioning"])
                saved = first.prepare(descriptor, conditioning=[[torch.tensor([[4.0]]), {}]])
                self.assertTrue(saved["ui"]["h3_conditioning_cache"][0].get("models_unloaded"))
                self.assertFalse(saved["ui"]["h3_conditioning_cache"][0].get("diffusion_model_kept"))
                # 미스: 방금 TE 로 인코딩했으니 전부 내린다.
                self.assertEqual(calls, ["unload", "empty"])
                calls.clear()
                second = ForgeNeoH3ConditioningCachePrepare()
                self.assertEqual(second.check_lazy_status(descriptor), [])
                receipt = second.prepare(descriptor)["ui"]["h3_conditioning_cache"][0]
                self.assertTrue(receipt["hit"])
                self.assertTrue(receipt["models_unloaded"])
                # 히트(시드 리롤): 인코더 배리어는 돌되 상주한 UNET 은 남긴다.
                self.assertTrue(receipt["diffusion_model_kept"])
                self.assertEqual(calls, [("free", "cuda:0", (diffusion,)), "empty"])
                loaded = ForgeNeoH3ConditioningCacheLoad().load(descriptor)[0]
                self.assertEqual(loaded[0][0].tolist(), [[4.0]])
                # 배리어 실패는 히트에서도 삼키지 않는다(선별 해제 경로 포함).
                manager.free_memory = mock.Mock(side_effect=RuntimeError("synthetic unload failure"))
                with self.assertRaisesRegex(RuntimeError, "synthetic unload failure"):
                    ForgeNeoH3ConditioningCachePrepare().prepare(descriptor)
                manager.unload_all_models = mock.Mock(side_effect=RuntimeError("synthetic unload failure"))
                manager.current_loaded_models = None  # 선별 불가 → 전체 해제 폴백
                with self.assertRaisesRegex(RuntimeError, "synthetic unload failure"):
                    ForgeNeoH3ConditioningCachePrepare().prepare(descriptor)

    def test_hit_without_model_manager_introspection_falls_back_to_full_unload(self):
        from comfy_custom_nodes.ai_studio_forge_parity import h3_cache_nodes
        calls = []
        manager = SimpleNamespace(processing_interrupted=lambda: False,
            unload_all_models=lambda: calls.append("unload"),
            soft_empty_cache=lambda: calls.append("empty"))
        with mock.patch.dict("sys.modules", {"comfy.model_management": manager}):
            kept = h3_cache_nodes._unload_encoder_models(keep_diffusion=True)
        self.assertFalse(kept)
        self.assertEqual(calls, ["unload", "empty"])

    def test_model_digest_survives_a_restart_through_the_sidecar(self):
        from comfy_custom_nodes.ai_studio_forge_parity import h3_cache_nodes
        with tempfile.TemporaryDirectory() as tmp:
            model = Path(tmp) / "unet.safetensors"
            model.write_bytes(b"large synthetic model")
            sidecar = Path(tmp) / "cache" / h3_cache_nodes.DIGEST_SIDECAR
            self.addCleanup(h3_cache_nodes._FINGERPRINTS.clear)
            h3_cache_nodes._FINGERPRINTS.clear()
            first = h3_cache_nodes.content_identity(model, memoize=True, sidecar=sidecar)
            self.assertTrue(sidecar.is_file())
            # '재시작': 메모리 memo 가 비어도 사이드카가 같은 stamp 의 digest 를 준다.
            h3_cache_nodes._FINGERPRINTS.clear()
            with mock.patch.object(h3_cache_nodes, "_digest_file",
                                   side_effect=AssertionError("must not re-hash")):
                again = h3_cache_nodes.content_identity(model, memoize=True, sidecar=sidecar)
            self.assertEqual(again, first)
            # 파일이 바뀌면 stamp 가 달라져 다시 해시한다.
            h3_cache_nodes._FINGERPRINTS.clear()
            model.write_bytes(b"replaced synthetic model!")
            changed = h3_cache_nodes.content_identity(model, memoize=True, sidecar=sidecar)
            self.assertNotEqual(changed["sha256"], first["sha256"])
            # 깨진 사이드카는 무시하고(오류 없이) 다시 해시한다.
            sidecar.write_text("{not json", encoding="utf-8")
            h3_cache_nodes._FINGERPRINTS.clear()
            self.assertEqual(
                h3_cache_nodes.content_identity(model, memoize=True, sidecar=sidecar), changed,
            )
            # 미디어(memoize=False)는 사이드카에 남기지 않는다.
            media = Path(tmp) / "input.png"
            media.write_bytes(b"media")
            h3_cache_nodes.content_identity(media, sidecar=sidecar)
            entries = json.loads(sidecar.read_text(encoding="utf-8"))["entries"]
            self.assertEqual(set(entries), {str(model.resolve())})

    @requires_torch
    def test_orphaned_temporary_files_are_reclaimed_but_live_data_is_not(self):
        from comfy_custom_nodes.ai_studio_forge_parity.h3_cache_nodes import (
            ConditioningCache, DIGEST_SIDECAR,
        )
        import os
        import time
        torch = _torch(self)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "h3"
            root.mkdir()
            old_tmp = root / (("b" * 64) + "." + ("1" * 32) + ".tmp")
            old_manifest = root / (("b" * 64) + "." + ("2" * 32) + ".json.tmp")
            recent_tmp = root / (("c" * 64) + "." + ("3" * 32) + ".tmp")
            old_sidecar_part = root / (DIGEST_SIDECAR + "." + ("4" * 32) + ".part")
            foreign = root / "notes.tmp"
            sidecar = root / DIGEST_SIDECAR
            for path in (old_tmp, old_manifest, recent_tmp, old_sidecar_part, foreign, sidecar):
                path.write_bytes(b"x" * 8)
            stale = time.time() - 3600
            for path in (old_tmp, old_manifest, old_sidecar_part):
                os.utime(path, (stale, stale))
            store = ConditioningCache(root)
            store.put("a" * 64, [[torch.tensor([[1.0]]), {}]])
            # put 시작 정리는 오래된 고아만 지운다(다른 프로세스가 쓰는 중일 수 있는 최근 것은 둔다).
            self.assertFalse(old_tmp.exists())
            self.assertFalse(old_manifest.exists())
            self.assertFalse(old_sidecar_part.exists())
            self.assertTrue(recent_tmp.exists())
            result = store.clear()
            self.assertEqual(result["removedEntries"], 1)
            self.assertFalse(recent_tmp.exists())
            self.assertTrue(foreign.exists())
            self.assertTrue(sidecar.exists())

    @requires_torch
    def test_reduced_limit_is_enforced_even_on_a_cache_hit(self):
        from comfy_custom_nodes.ai_studio_forge_parity.h3_cache_nodes import ConditioningCache
        torch = _torch(self)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "h3"
            store = ConditioningCache(root, max_entries=3)
            for key in "abc":
                store.put(key * 64, [[torch.tensor([[1.0]]), {}]])
            limited = ConditioningCache(root, max_entries=1)
            self.assertIsNotNone(limited.get("a" * 64))
            self.assertEqual(limited.stats()["entries"], 1)

    @requires_torch
    def test_second_lazy_check_rejects_inputs_changed_while_encoding(self):
        from comfy_custom_nodes.ai_studio_forge_parity.h3_cache_nodes import ForgeNeoH3ConditioningCachePrepare
        torch = _torch(self)
        descriptor = build("h3_i2v", {"prompt": "synthetic", "input_image": "start.png", "conditioning_cache": True})["stages"][0]["workflow"]["90"]["inputs"]["descriptor"]
        with tempfile.TemporaryDirectory() as tmp:
            model = Path(tmp) / "model.safetensors"
            model.write_bytes(b"model-v1")
            media = Path(tmp) / "start.png"
            media.write_bytes(b"image-v1")
            folders = SimpleNamespace(get_full_path=lambda *_: str(model),
                get_output_directory=lambda: str(Path(tmp) / "output"),
                get_input_directory=lambda: tmp, base_path=tmp)
            with mock.patch.dict("sys.modules", {"folder_paths": folders}):
                node = ForgeNeoH3ConditioningCachePrepare()
                self.assertEqual(node.check_lazy_status(descriptor), ["conditioning"])
                media.write_bytes(b"image-v2-new-content")
                with self.assertRaisesRegex(RuntimeError, "변경"):
                    node.check_lazy_status(descriptor, conditioning=[[torch.tensor([[1.0]]), {}]])

    @requires_torch
    def test_in_place_model_replacement_requires_server_restart(self):
        from comfy_custom_nodes.ai_studio_forge_parity.h3_cache_nodes import ForgeNeoH3ConditioningCachePrepare
        _torch(self)
        descriptor = build("h3_t2v", {"prompt": "synthetic", "conditioning_cache": True})["stages"][0]["workflow"]["90"]["inputs"]["descriptor"]
        with tempfile.TemporaryDirectory() as tmp:
            model = Path(tmp) / "model.safetensors"
            model.write_bytes(b"model-v1")
            folders = SimpleNamespace(get_full_path=lambda *_: str(model),
                get_output_directory=lambda: str(Path(tmp) / "output"),
                get_input_directory=lambda: tmp, base_path=tmp)
            with mock.patch.dict("sys.modules", {"folder_paths": folders}):
                self.assertEqual(ForgeNeoH3ConditioningCachePrepare().check_lazy_status(descriptor), ["conditioning"])
                model.write_bytes(b"model-v2-new-content")
                with self.assertRaisesRegex(RuntimeError, "재시작"):
                    ForgeNeoH3ConditioningCachePrepare().check_lazy_status(descriptor)

    @requires_torch
    def test_corrupt_entry_is_a_miss_and_foreign_files_survive_clear(self):
        from comfy_custom_nodes.ai_studio_forge_parity.h3_cache_nodes import ConditioningCache
        torch = _torch(self)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "h3"
            store = ConditioningCache(root)
            store.put("a" * 64, [[torch.tensor([[1.0]]), {}]])
            (root / (("a" * 64) + ".pt")).write_bytes(b"corrupt")
            self.assertIsNone(store.get("a" * 64))
            foreign = root / "do-not-delete.txt"
            foreign.write_text("foreign", encoding="utf-8")
            store.clear()
            self.assertEqual(foreign.read_text(encoding="utf-8"), "foreign")
            with self.assertRaises(ValueError):
                store.get("../escape")

    @requires_torch
    def test_cancel_during_save_never_publishes_partial_entry(self):
        from comfy_custom_nodes.ai_studio_forge_parity.h3_cache_nodes import ConditioningCache, ConditioningCacheCancelled
        torch = _torch(self)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "h3"
            store = ConditioningCache(root)
            calls = []
            def cancelled():
                calls.append(None)
                return len(calls) >= 2
            with self.assertRaises(ConditioningCacheCancelled):
                store.put("a" * 64, [[torch.ones(32, 32), {}]], cancelled=cancelled)
            self.assertIsNone(store.get("a" * 64))
            self.assertEqual(list(root.iterdir()), [])

    def test_content_and_model_changes_invalidate_while_seed_and_upload_name_do_not(self):
        from comfy_custom_nodes.ai_studio_forge_parity.h3_cache_nodes import conditioning_identity
        def descriptor(**params):
            values = {"prompt": "synthetic", "input_image": "one.png", "conditioning_cache": True, **params}
            return build("h3_i2v", values)["stages"][0]["workflow"]["90"]["inputs"]["descriptor"]
        with tempfile.TemporaryDirectory() as tmp:
            model, media = Path(tmp) / "model", Path(tmp) / "image"
            model.write_bytes(b"model-v1")
            media.write_bytes(b"same-image")
            def key(**params):
                return conditioning_identity(descriptor(**params), lambda *_: model, lambda *_: media)
            first = key(seed=1)
            self.assertEqual(first, key(seed=99, input_image="another-name.png", quality="quality"))
            media.write_bytes(b"changed-image")
            self.assertNotEqual(first, key())
            media.write_bytes(b"same-image")
            model.write_bytes(b"model-v2-has-different-content")
            self.assertNotEqual(first, key())

    @requires_torch
    def test_byte_budget_rejects_a_single_oversized_conditioning(self):
        from comfy_custom_nodes.ai_studio_forge_parity.h3_cache_nodes import ConditioningCache
        torch = _torch(self)
        with tempfile.TemporaryDirectory() as tmp:
            store = ConditioningCache(Path(tmp) / "h3", max_bytes=1024 ** 2)
            with self.assertRaisesRegex(ValueError, "한도"):
                store.put("a" * 64, [[torch.zeros(1024, 512), {}]])
            self.assertEqual(store.stats()["bytes"], 0)

    def test_cache_prepare_receipt_must_confirm_ready_and_identity(self):
        from core.h3_conditioning_cache import prepare_receipt
        for info in ({"node_outputs": {"90": {"text": ["done"]}}},
                     {"node_outputs": []}, {"node_outputs": None},
                     {"node_outputs": {"90": {"h3_conditioning_cache": None}}},
                     {"node_outputs": {"90": {"h3_conditioning_cache": [{"ready": True, "key": 4}]}}}):
            with self.subTest(info=info), self.assertRaises(RuntimeError):
                prepare_receipt(info)
        receipt = prepare_receipt({"node_outputs": {"90": {"h3_conditioning_cache": [
            {"ready": True, "key": "a" * 64, "hit": True, "models_unloaded": True}
        ]}}})
        self.assertTrue(receipt["hit"])

    def test_all_h3_modes_split_without_sampler_or_encoder_cross_links(self):
        for mode, extra in [("h3_t2v", {}), ("h3_i2v", {"input_image": "start.png"}),
                            ("h3_v2v", {"input_video": "motion.mp4", "include_reference_audio": True})]:
            with self.subTest(mode=mode):
                result = build(mode, {"prompt": "synthetic", "conditioning_cache": True, **extra})
                for stage in result["stages"]:
                    graph = stage["workflow"]
                    for node in graph.values():
                        for value in node["inputs"].values():
                            if isinstance(value, list) and len(value) == 2:
                                self.assertIn(value[0], graph)


if __name__ == "__main__":
    unittest.main()
