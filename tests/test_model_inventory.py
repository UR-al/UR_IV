from __future__ import annotations

import json
import io
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from core import model_inventory
from core.model_inventory import ModelInventory


class ModelInventoryTests(unittest.TestCase):
    def _snapshot(
        self,
        forge_root: Path,
        comfy_root: Path,
        *,
        primary: str = "forge",
        active: str = "comfyui",
    ) -> dict:
        def paths(root: Path) -> dict[str, list[str]]:
            result = {}
            for category in ("checkpoints", "diffusion_models", "loras", "vae", "text_encoders"):
                target = root / category
                target.mkdir(parents=True, exist_ok=True)
                result[category] = [str(target)]
            return result

        return {
            "primaryModelEngine": primary,
            "activeEngine": active,
            "engines": {
                "forge": {"name": "Forge Neo", "modelPaths": paths(forge_root)},
                "comfyui": {"name": "ComfyUI", "modelPaths": paths(comfy_root)},
            },
        }

    def test_primary_all_and_secondary_content_unique(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            snapshot = self._snapshot(root / "forge", root / "comfy")
            forge_loras = Path(snapshot["engines"]["forge"]["modelPaths"]["loras"][0])
            comfy_loras = Path(snapshot["engines"]["comfyui"]["modelPaths"]["loras"][0])

            (forge_loras / "shared.safetensors").write_bytes(b"same model content")
            (forge_loras / "primary_only.safetensors").write_bytes(b"primary")
            (comfy_loras / "shared.safetensors").write_bytes(b"same model content")
            (comfy_loras / "comfy_only.safetensors").write_bytes(b"secondary")

            entries = ModelInventory(snapshot).entries("loras")

            self.assertEqual(
                [(item["source"], item["name"]) for item in entries],
                [
                    ("forge", "primary_only"),
                    ("forge", "shared"),
                    ("comfyui", "comfy_only"),
                ],
            )
            self.assertTrue(all(item["primary"] for item in entries[:2]))
            self.assertEqual(entries[2]["group"], "secondary_unique")
            for item in entries:
                for field in (
                    "id", "path", "runtimeName", "sourceName", "primary",
                    "backendAvailable", "nameConflict",
                ):
                    self.assertIn(field, item)

    def test_same_name_different_content_is_retained_and_marked_conflict(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            snapshot = self._snapshot(root / "forge", root / "comfy")
            forge_loras = Path(snapshot["engines"]["forge"]["modelPaths"]["loras"][0])
            comfy_loras = Path(snapshot["engines"]["comfyui"]["modelPaths"]["loras"][0])
            (forge_loras / "Style.safetensors").write_bytes(b"AAAA")
            (comfy_loras / "style.safetensors").write_bytes(b"BBBB")

            entries = ModelInventory(snapshot).entries("lora")

            self.assertEqual(len(entries), 2)
            self.assertEqual({item["source"] for item in entries}, {"forge", "comfyui"})
            self.assertTrue(all(item["nameConflict"] for item in entries))

    def test_different_names_never_trigger_sampled_fingerprint(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            snapshot = self._snapshot(root / "forge", root / "comfy")
            forge_loras = Path(snapshot["engines"]["forge"]["modelPaths"]["loras"][0])
            comfy_loras = Path(snapshot["engines"]["comfyui"]["modelPaths"]["loras"][0])
            (forge_loras / "alpha.safetensors").write_bytes(b"identical bytes")
            (comfy_loras / "beta.safetensors").write_bytes(b"identical bytes")

            with patch("core.model_inventory._sampled_fingerprint") as fingerprint:
                entries = ModelInventory(snapshot).entries("loras")

            self.assertEqual(len(entries), 2)
            fingerprint.assert_not_called()

    def test_sampled_fingerprint_reads_only_first_middle_and_last_chunks(self):
        size = model_inventory._SAMPLE_CHUNK_BYTES * 8
        reads: list[int] = []

        class TrackingStream(io.BytesIO):
            def read(self, amount=-1):
                reads.append(amount)
                return super().read(amount)

        stream = TrackingStream(b"x" * size)
        with patch.object(Path, "open", return_value=stream):
            fingerprint = model_inventory._sampled_fingerprint(Path("unused"), size)

        self.assertIsNotNone(fingerprint)
        self.assertEqual(reads, [model_inventory._SAMPLE_CHUNK_BYTES] * 3)

    def test_same_resolved_file_is_merged_before_fingerprinting(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            shared = root / "shared-loras"
            shared.mkdir()
            (shared / "one.safetensors").write_bytes(b"model")
            empty_paths = {
                category: []
                for category in ("checkpoints", "diffusion_models", "loras", "vae", "text_encoders")
            }
            forge_paths = dict(empty_paths)
            forge_paths["loras"] = [str(shared)]
            comfy_paths = dict(empty_paths)
            comfy_paths["loras"] = [str(shared)]
            snapshot = {
                "primaryModelEngine": "forge",
                "activeEngine": "comfyui",
                "engines": {
                    "forge": {"name": "Forge Neo", "modelPaths": forge_paths},
                    "comfyui": {"name": "ComfyUI", "modelPaths": comfy_paths},
                },
            }

            with patch("core.model_inventory._sampled_fingerprint") as fingerprint:
                entries = ModelInventory(snapshot).entries("loras")

            self.assertEqual(len(entries), 1)
            self.assertEqual(entries[0]["source"], "forge")
            fingerprint.assert_not_called()

    def test_option_groups_keep_backend_raw_values_undecorated(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            snapshot = self._snapshot(root / "forge", root / "comfy")
            forge_models = Path(snapshot["engines"]["forge"]["modelPaths"]["checkpoints"][0])
            comfy_models = Path(snapshot["engines"]["comfyui"]["modelPaths"]["checkpoints"][0])
            (forge_models / "shared.safetensors").write_bytes(b"forge")
            (comfy_models / "comfy-only.safetensors").write_bytes(b"comfy")
            inventory = ModelInventory(snapshot)

            raw = ["shared.safetensors [a1b2c3d4]", "comfy-only.safetensors"]
            groups = inventory.option_groups("checkpoints", raw)

            self.assertEqual(groups[0], {
                "label": "Forge Neo",
                "source": "forge",
                "primary": True,
                "options": ["shared.safetensors [a1b2c3d4]"],
            })
            self.assertEqual(groups[1], {
                "label": "ComfyUI",
                "source": "comfyui",
                "primary": False,
                "options": ["comfy-only.safetensors"],
            })
            self.assertEqual([option for group in groups for option in group["options"]], raw)

    def test_content_duplicate_api_aliases_are_emitted_once_as_main(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            snapshot = self._snapshot(root / "forge", root / "comfy")
            forge_loras = Path(snapshot["engines"]["forge"]["modelPaths"]["loras"][0])
            comfy_loras = Path(snapshot["engines"]["comfyui"]["modelPaths"]["loras"][0])
            (forge_loras / "main").mkdir()
            (comfy_loras / "other").mkdir()
            (forge_loras / "main" / "shared.safetensors").write_bytes(b"same content")
            (comfy_loras / "other" / "shared.safetensors").write_bytes(b"same content")
            inventory = ModelInventory(snapshot)
            raw_names = ["main/shared.safetensors", "other/shared.safetensors"]

            merged = inventory.merge_loras([{"name": name} for name in raw_names])
            groups = inventory.option_groups("loras", raw_names)

            self.assertEqual(len(merged), 1)
            self.assertEqual(merged[0]["runtimeName"], raw_names[0])
            self.assertTrue(merged[0]["primary"])
            self.assertEqual(groups, [{
                "label": "Forge Neo",
                "source": "forge",
                "primary": True,
                "options": [raw_names[0]],
            }])

    def test_same_name_distinct_content_api_paths_keep_both_conflicts(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            snapshot = self._snapshot(root / "forge", root / "comfy")
            forge_loras = Path(snapshot["engines"]["forge"]["modelPaths"]["loras"][0])
            comfy_loras = Path(snapshot["engines"]["comfyui"]["modelPaths"]["loras"][0])
            (forge_loras / "main").mkdir()
            (comfy_loras / "other").mkdir()
            forge_file = forge_loras / "main" / "style.safetensors"
            comfy_file = comfy_loras / "other" / "style.safetensors"
            forge_file.write_bytes(b"AAAA")
            comfy_file.write_bytes(b"BBBB")
            inventory = ModelInventory(snapshot)

            merged = inventory.merge_loras([
                {"name": "main/style.safetensors", "path": str(forge_file)},
                {"name": "other/style.safetensors", "path": str(comfy_file)},
            ])

            self.assertEqual(len(merged), 2)
            self.assertEqual(len({item["id"] for item in merged}), 2)
            self.assertTrue(all(item["backendAvailable"] for item in merged))
            self.assertTrue(all(item["nameConflict"] for item in merged))

    def test_api_matching_uses_linear_indexes_without_pairwise_samefile(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            snapshot = self._snapshot(root / "forge", root / "comfy", active="forge")
            forge_loras = Path(snapshot["engines"]["forge"]["modelPaths"]["loras"][0])
            count = 100
            api_items = []
            for index in range(count):
                path = forge_loras / f"item-{index}.safetensors"
                path.write_bytes(f"model-{index}".encode("ascii"))
                api_items.append({"name": f"item-{index}", "path": str(path)})
            inventory = ModelInventory(snapshot)

            with (
                patch("os.path.samefile") as samefile,
                patch(
                    "core.model_inventory._value_variants",
                    wraps=model_inventory._value_variants,
                ) as variants,
            ):
                merged = inventory.merge_loras(api_items)

            self.assertEqual(len(merged), count)
            samefile.assert_not_called()
            self.assertLessEqual(variants.call_count, count * 10)

    def test_lora_merge_keeps_api_metadata_and_disk_source_state(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            snapshot = self._snapshot(root / "forge", root / "comfy")
            forge_loras = Path(snapshot["engines"]["forge"]["modelPaths"]["loras"][0])
            comfy_loras = Path(snapshot["engines"]["comfyui"]["modelPaths"]["loras"][0])
            (forge_loras / "nested").mkdir()
            (forge_loras / "nested" / "shared.safetensors").write_bytes(b"shared")
            (forge_loras / "offline.safetensors").write_bytes(b"offline")
            (comfy_loras / "comfy-only.safetensors").write_bytes(b"comfy")
            inventory = ModelInventory(snapshot)

            merged = inventory.merge_loras([
                {
                    "name": "nested/shared.safetensors",
                    "alias": "Shared Style",
                    "trigger_words": ["shared trigger"],
                },
                {"name": "comfy-only.safetensors", "trigger_words": []},
            ])
            by_runtime = {item["runtimeName"]: item for item in merged}

            shared = by_runtime["nested/shared.safetensors"]
            self.assertEqual(shared["label"], "Shared Style")
            self.assertEqual(shared["triggerWords"], ["shared trigger"])
            self.assertEqual(shared["source"], "forge")
            self.assertTrue(shared["primary"])
            self.assertTrue(shared["backendAvailable"])

            comfy = by_runtime["comfy-only.safetensors"]
            self.assertEqual(comfy["source"], "comfyui")
            self.assertFalse(comfy["primary"])
            self.assertTrue(comfy["backendAvailable"])

            offline = by_runtime["offline"]
            self.assertFalse(offline["backendAvailable"])

    def test_vae_and_text_encoder_catalogs_use_runtime_model_paths(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            snapshot = self._snapshot(root / "forge", root / "comfy")
            forge_vae = Path(snapshot["engines"]["forge"]["modelPaths"]["vae"][0])
            comfy_te = Path(snapshot["engines"]["comfyui"]["modelPaths"]["text_encoders"][0])
            (forge_vae / "main.vae.safetensors").write_bytes(b"vae")
            (comfy_te / "clip.gguf").write_bytes(b"te")
            inventory = ModelInventory(snapshot)

            self.assertEqual(
                [item["runtimeName"] for item in inventory.entries("vae")],
                ["main.vae.safetensors"],
            )
            text_encoders = inventory.entries("te")
            self.assertEqual([item["runtimeName"] for item in text_encoders], ["clip.gguf"])
            self.assertEqual(text_encoders[0]["source"], "comfyui")

    def test_vue_lora_force_refresh_keeps_raw_backend_cache(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            snapshot = self._snapshot(root / "forge", root / "comfy")
            comfy_loras = Path(snapshot["engines"]["comfyui"]["modelPaths"]["loras"][0])
            (comfy_loras / "fresh.safetensors").write_bytes(b"fresh")
            fresh = [{
                "name": "fresh.safetensors",
                "alias": "Fresh LoRA",
                "trigger_words": ["fresh trigger"],
            }]

            class RuntimeManager:
                def snapshot(self):
                    return snapshot

            class Backend:
                def get_loras(self):
                    return fresh

            from ui.vue_bridge import VueBridge
            from ui import lora_catalog_cache

            previous_cache = lora_catalog_cache.raw_loras()
            lora_catalog_cache._raw_loras = [{"name": "stale"}]
            try:
                with (
                    patch("backends.get_backend", return_value=Backend()),
                    patch("backends.get_backend_type", return_value=SimpleNamespace(value="comfyui")),
                    patch("core.backend_runtime.get_backend_runtime_manager", return_value=RuntimeManager()),
                ):
                    result = json.loads(VueBridge.getLoras(None, "force"))
            finally:
                cached_after = lora_catalog_cache.raw_loras()
                lora_catalog_cache._raw_loras = previous_cache

            self.assertEqual(cached_after, fresh)
            self.assertNotIn("source", cached_after[0])
            self.assertEqual(result[0]["source"], "comfyui")
            self.assertEqual(result[0]["runtimeName"], "fresh.safetensors")
            self.assertEqual(result[0]["triggerWords"], ["fresh trigger"])
            self.assertTrue(result[0]["backendAvailable"])

    def test_vue_lora_force_refresh_accepts_empty_backend_result(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            snapshot = self._snapshot(root / "forge", root / "comfy")

            class RuntimeManager:
                def snapshot(self):
                    return snapshot

            class Backend:
                def get_loras(self):
                    return []

            from ui.vue_bridge import VueBridge
            from ui import lora_catalog_cache

            previous_cache = lora_catalog_cache.raw_loras()
            lora_catalog_cache._raw_loras = [{"name": "removed-lora"}]
            try:
                with (
                    patch("backends.get_backend", return_value=Backend()),
                    patch("backends.get_backend_type", return_value=SimpleNamespace(value="forge")),
                    patch("core.backend_runtime.get_backend_runtime_manager", return_value=RuntimeManager()),
                ):
                    result = json.loads(VueBridge.getLoras(None, "force"))
            finally:
                cached_after = lora_catalog_cache.raw_loras()
                lora_catalog_cache._raw_loras = previous_cache

            self.assertEqual(cached_after, [])
            self.assertEqual(result, [])

    def test_catalog_and_matcher_touch_each_file_once(self):
        """파일당 resolve·stat 을 한 번으로 — 예전엔 LoRA 하나에 resolve 5.6회, stat 9.4회."""
        import os

        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            snapshot = self._snapshot(root / "forge", root / "comfy")
            forge_loras = Path(snapshot["engines"]["forge"]["modelPaths"]["loras"][0])
            comfy_loras = Path(snapshot["engines"]["comfyui"]["modelPaths"]["loras"][0])
            count = 12
            for index in range(count):
                (forge_loras / f"style_{index}.safetensors").write_bytes(b"f" * (index + 1))
                (comfy_loras / f"comfy_{index}.safetensors").write_bytes(b"c" * (index + 1))

            real_realpath = os.path.realpath
            real_stat = os.stat
            resolved, stats = [], []

            def counting_realpath(path, *args, **kwargs):
                # Path.resolve 도 내부에서 os.path.realpath 를 부르므로 둘 다 잡힌다.
                resolved.append(str(path))
                return real_realpath(path, *args, **kwargs)

            def counting_stat(path, *args, **kwargs):
                stats.append(str(path))
                return real_stat(path, *args, **kwargs)

            inventory = ModelInventory(snapshot)
            api = [{"name": f"style_{index}", "alias": f"style_{index}"} for index in range(count)]
            with patch("core.model_inventory.os.path.realpath", side_effect=counting_realpath), \
                    patch("core.model_inventory.os.stat", side_effect=counting_stat):
                merged = inventory.merge_loras(api)

            files = 2 * count
            self.assertEqual(len(merged), files)
            model_resolves = [p for p in resolved if p.endswith(".safetensors")]
            model_stats = [p for p in stats if str(p).endswith(".safetensors")]
            self.assertLessEqual(len(model_resolves), files, model_resolves)
            self.assertLessEqual(len(model_stats), files, model_stats)
            self.assertTrue(all(item["backendAvailable"] for item in merged[:count]))

    def test_matcher_reuses_catalog_physical_keys_for_absolute_api_paths(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            snapshot = self._snapshot(root / "forge", root / "comfy", active="forge")
            forge_loras = Path(snapshot["engines"]["forge"]["modelPaths"]["loras"][0])
            target = forge_loras / "sub" / "Deep.safetensors"
            target.parent.mkdir()
            target.write_bytes(b"deep")
            inventory = ModelInventory(snapshot)
            merged = inventory.merge_loras([{"name": "renamed", "path": str(target)}])
            self.assertEqual(1, len(merged))
            self.assertEqual("renamed", merged[0]["runtimeName"])
            self.assertEqual(str(target.resolve()), merged[0]["path"])
            self.assertNotIn("physical", json.dumps(merged))


class LoraCatalogPrewarmTests(unittest.TestCase):
    def setUp(self):
        from ui import lora_catalog_cache

        self.cache = lora_catalog_cache
        self.previous = lora_catalog_cache.raw_loras()
        self.addCleanup(setattr, lora_catalog_cache, "_raw_loras", self.previous)

    def _patches(self, backend, merged=None):
        merged = merged if merged is not None else [{"name": "x"}]
        return (
            patch("backends.get_backend", return_value=backend),
            patch("backends.get_backend_type", return_value=SimpleNamespace(value="forge")),
            patch("ui.lora_catalog_cache.merged_json", side_effect=lambda loras, engine: json.dumps(
                {"engine": engine, "count": len(loras)})),
        )

    def test_prewarmed_cache_is_served_by_get_loras_without_http(self):
        from ui.lora_catalog_cache import warm_once
        from ui.vue_bridge import VueBridge

        class Backend:
            calls = 0

            def get_loras(self):
                Backend.calls += 1
                return [{"name": "a"}, {"name": "b"}]

        bridge = SimpleNamespace(_merged_lora_cache=None)
        self.cache._raw_loras = []
        backend = Backend()
        p1, p2, p3 = self._patches(backend)
        with p1, p2, p3:
            self.assertTrue(warm_once(bridge))
            self.assertEqual(1, Backend.calls)
            served = json.loads(VueBridge.getLoras(bridge, ""))
            self.assertEqual({"engine": "forge", "count": 2}, served)
            self.assertEqual(1, Backend.calls, "프리워밍된 캐시는 GUI 스레드 HTTP 없이 적중한다")
            self.assertFalse(warm_once(bridge, only_if_cold=True), "이미 따뜻하면 다시 받지 않는다")
            self.assertEqual(1, Backend.calls)

    def test_prewarm_does_not_clobber_a_newer_force_refresh(self):
        from ui.lora_catalog_cache import warm_once
        from ui.vue_bridge import VueBridge

        cache = self.cache
        bridge = SimpleNamespace(_merged_lora_cache=None)

        class Backend:
            calls = 0

            def get_loras(self):
                Backend.calls += 1
                if Backend.calls == 1:
                    # 프리워밍이 HTTP 를 기다리는 사이 사용자가 강제 새로고침했다(GUI 스레드)
                    VueBridge.getLoras(bridge, "force")
                    return [{"name": "older-1"}, {"name": "older-2"}]
                return [{"name": "newer"}]

        cache._raw_loras = []
        p1, p2, p3 = self._patches(Backend())
        with p1, p2, p3:
            self.assertFalse(warm_once(bridge), "늦게 끝난 프리워밍은 아무것도 쓰지 않는다")
            self.assertEqual([{"name": "newer"}], cache.raw_loras())
            cached = bridge._merged_lora_cache
            self.assertIs(cache.raw_loras(), cached["raw"])
            self.assertEqual({"engine": "forge", "count": 1}, json.loads(cached["json"]))
            served = json.loads(VueBridge.getLoras(bridge, ""))
        self.assertEqual({"engine": "forge", "count": 1}, served)
        self.assertEqual(2, Backend.calls, "강제 새로고침 결과가 그대로 적중한다")

    def test_invalidation_during_prewarm_fetch_is_not_cached(self):
        """받는 사이 모델 경로 저장(_refresh_forge_module_widgets)이 캐시를 비웠으면,
        무효화 뒤의 빈 목록을 병합해 캐시하지 않는다 — 다음 getLoras 가 백엔드를 다시 읽는다."""
        from ui.lora_catalog_cache import invalidate, warm_once
        from ui.vue_bridge import VueBridge

        bridge = SimpleNamespace(_merged_lora_cache=None)

        class Backend:
            calls = 0

            def get_loras(self):
                Backend.calls += 1
                if Backend.calls == 1:
                    invalidate(bridge)
                return [{"name": "api-lora", "path": "api.safetensors"}]

        self.cache._raw_loras = [{"name": "stale"}]
        p1, p2, p3 = self._patches(Backend())
        with p1, p2, p3:
            self.assertFalse(warm_once(bridge))
            self.assertIsNone(bridge._merged_lora_cache, "빈 목록으로 만든 병합 결과를 남기지 않는다")
            self.assertEqual([], self.cache.raw_loras())
            served = json.loads(VueBridge.getLoras(bridge, ""))
        self.assertEqual(2, Backend.calls, "무효화 뒤 첫 getLoras 는 백엔드를 다시 읽는다")
        self.assertEqual({"engine": "forge", "count": 1}, served)
        self.assertEqual([{"name": "api-lora", "path": "api.safetensors"}], self.cache.raw_loras())

    def test_invalidation_during_prewarm_merge_discards_result(self):
        from ui.lora_catalog_cache import invalidate, warm_once

        bridge = SimpleNamespace(_merged_lora_cache=None)

        class Backend:
            def get_loras(self):
                return [{"name": "a"}]

        def merge_then_invalidate(loras, engine):
            invalidate(bridge)   # 디스크 병합 중에 무효화가 끼었다
            return json.dumps({"count": len(loras)})

        self.cache._raw_loras = []
        with (
            patch("backends.get_backend", return_value=Backend()),
            patch("backends.get_backend_type", return_value=SimpleNamespace(value="forge")),
            patch("ui.lora_catalog_cache.merged_json", side_effect=merge_then_invalidate),
        ):
            self.assertFalse(warm_once(bridge))
        self.assertIsNone(bridge._merged_lora_cache)
        self.assertEqual([], self.cache.raw_loras())

    def test_cache_hit_requires_the_same_raw_list_object(self):
        """무효화가 넣은 새 [] 는 내용이 같아도 다른 객체다 — 예전 [] 로 만든 병합 결과가 적중하면
        백엔드 LoRA 가 모두 '사용 불가'로 보이는 디스크 전용 카탈로그가 나간다."""
        from ui.lora_catalog_cache import cache_entry, cached_json, raw_signature

        fetched_empty = []
        entry = cache_entry("forge", raw_signature(fetched_empty), "[]", raw=fetched_empty)
        self.assertEqual("[]", cached_json(entry, "forge", raw_signature([]), raw=fetched_empty))
        self.assertIsNone(cached_json(entry, "forge", raw_signature([]), raw=[]))
        self.assertIsNone(cached_json(entry, "comfyui", raw_signature([]), raw=fetched_empty))

    def test_invalidate_clears_both_caches(self):
        from ui.lora_catalog_cache import invalidate

        bridge = SimpleNamespace(_merged_lora_cache={"json": "[]"})
        old = [{"name": "a"}]
        self.cache._raw_loras = old
        invalidate(bridge)
        self.assertEqual([], self.cache.raw_loras())
        self.assertIsNot(old, self.cache.raw_loras())
        self.assertIsNone(bridge._merged_lora_cache)
        invalidate(None)   # 브리지가 아직 없어도 raw 캐시는 비운다
        self.assertEqual([], self.cache.raw_loras())


if __name__ == "__main__":
    unittest.main()
