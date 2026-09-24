"""ComfyUI SAM3 모델 보관 설정(ui_prefs.comfySam3KeepInRam) → 컴파일러 cache_model (감사 #167)."""
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from backends.comfyui_backend import ComfyUIBackend
from core import comfy_sam3_cache_policy as policy
from core import sam3_args


class KeepInRamPrefTests(unittest.TestCase):
    def test_default_is_on_and_only_a_boolean_turns_it_off(self):
        self.assertTrue(policy.comfy_sam3_keep_in_ram_enabled(None))
        self.assertTrue(policy.comfy_sam3_keep_in_ram_enabled({}))
        self.assertTrue(policy.comfy_sam3_keep_in_ram_enabled({policy.PREF_KEY: "false"}))
        self.assertTrue(policy.comfy_sam3_keep_in_ram_enabled({policy.PREF_KEY: 0}))
        self.assertTrue(policy.comfy_sam3_keep_in_ram_enabled({policy.PREF_KEY: True}))
        self.assertFalse(policy.comfy_sam3_keep_in_ram_enabled({policy.PREF_KEY: False}))

    def test_file_cache_reads_rereads_on_change_and_falls_back_to_default(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "ui_prefs.json"
            cache = policy._KeepInRamPrefCache(lambda: path)
            self.assertTrue(cache.get())  # 파일 없음 → 기본(켜짐)

            path.write_text(json.dumps({policy.PREF_KEY: False}), encoding="utf-8")
            self.assertFalse(cache.get())
            with mock.patch("builtins.open", side_effect=AssertionError("cached")):
                self.assertFalse(cache.get())  # stamp 가 같으면 다시 읽지 않는다

            path.write_text(json.dumps({policy.PREF_KEY: True, "other": 1}), encoding="utf-8")
            stat = path.stat()
            os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1_000_000))
            self.assertTrue(cache.get())

            path.write_text("{broken", encoding="utf-8")
            self.assertTrue(cache.get())  # 못 읽으면 기본값

            path.write_text(json.dumps([policy.PREF_KEY]), encoding="utf-8")
            self.assertTrue(cache.get())  # 객체가 아니면 기본값


class BackendCompilerWiringTests(unittest.TestCase):
    def _backend(self):
        backend = ComfyUIBackend("http://127.0.0.1:1", workflow_path="")
        backend._preflight_bundled_node_pack = mock.Mock()
        backend.get_object_info = mock.Mock(return_value=None)
        return backend

    def test_workflow_compiler_carries_the_setting_into_sam3_cache_model(self):
        scripts = sam3_args.build_alwayson({"sam3_mode": "Mask only", "sam3_unload_after": True})
        for keep_in_ram in (True, False):
            with self.subTest(keep_in_ram=keep_in_ram), mock.patch(
                "core.comfy_sam3_cache_policy.comfy_sam3_keep_in_ram_from_prefs_file",
                return_value=keep_in_ram,
            ):
                compiler = self._backend()._workflow_compiler()
                self.assertIs(compiler.sam3_keep_in_ram, keep_in_ram)
                graph = compiler.compile_sam3_mask_only(
                    {"alwayson_scripts": scripts}, uploaded_image="source.png",
                )
                mask = next(node for node in graph.values()
                            if node["class_type"] == "ForgeNeoSAM3Mask")
                self.assertIs(mask["inputs"]["cache_model"], keep_in_ram)


if __name__ == "__main__":
    unittest.main()
