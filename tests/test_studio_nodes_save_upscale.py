"""AIStudioSaveForge writes Forge-style ``parameters`` plus ComfyUI chunks; AIStudioUpscaleImage."""
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np
from PIL import Image

from comfy_custom_nodes.ai_studio_forge_parity import studio_nodes


class _FolderPaths:
    """Just enough of ComfyUI's folder_paths: numbered ``<prefix>_00001_.png`` files under an output dir."""

    def __init__(self, root, upscalers=()):
        self.root = Path(root)
        self.upscalers = list(upscalers)

    def get_output_directory(self):
        return str(self.root)

    def get_filename_list(self, kind):
        return list(self.upscalers) if kind == "upscale_models" else []

    def get_save_image_path(self, prefix, output_dir, width=0, height=0):
        subfolder, name = prefix.rsplit("/", 1) if "/" in prefix else ("", prefix)
        folder = Path(output_dir) / subfolder
        folder.mkdir(parents=True, exist_ok=True)
        counter = len(list(folder.glob(f"{name}_*_.png"))) + 1
        return str(folder), name, counter, subfolder, prefix


def _state_json(**changes):
    state = {"prompt": "1girl, solo", "negative": "bad hands", "unet": "anima.safetensors", "steps": 20}
    state.update(changes)
    return json.dumps(state)


def _images(count=1, height=8, width=8):
    return np.zeros((count, height, width, 3), dtype=np.float32)


class SaveForgeTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.root = Path(self._tmp.name)

    def _save(self, images, **kwargs):
        node = studio_nodes.AIStudioSaveForge()
        args = {"studio_state": _state_json(), "seed": 42, "filename_prefix": "AIStudio/anima",
                "prompt": {"1": {"class_type": "KSampler"}}, "extra_pnginfo": {"workflow": {"nodes": []}}}
        args.update(kwargs)
        with mock.patch.dict(sys.modules, {"folder_paths": _FolderPaths(self.root)}):
            return node.save(images, **args)

    def test_saves_png_with_parameters_prompt_and_workflow_chunks(self):
        result = self._save(_images(1, 16, 8))
        entry = result["ui"]["images"][0]
        self.assertEqual(entry, {"filename": "anima_00001_.png", "subfolder": "AIStudio", "type": "output"})
        with Image.open(self.root / "AIStudio" / "anima_00001_.png") as image:
            self.assertEqual(image.size, (8, 16))
            info = dict(image.info)
        self.assertTrue(info["parameters"].startswith("1girl, solo\nNegative prompt: bad hands\nSteps: 20, "))
        self.assertEqual(json.loads(info["prompt"]), {"1": {"class_type": "KSampler"}})
        self.assertEqual(json.loads(info["workflow"]), {"nodes": []})

    def test_batch_gets_numbered_files(self):
        result = self._save(_images(2))
        self.assertEqual([x["filename"] for x in result["ui"]["images"]], ["anima_00001_.png", "anima_00002_.png"])

    def _seeds(self, count, stem="anima"):
        seeds = []
        for index in range(count):
            with Image.open(self.root / "AIStudio" / f"{stem}_{index + 1:05}_.png") as image:
                line = image.info["parameters"].splitlines()[-1]
            seeds.append(int(line.split("Seed: ", 1)[1].split(",", 1)[0]))
        return seeds

    def test_batch_with_forge_like_noise_writes_each_images_own_seed(self):
        # Forge-like noise (the default) drew image i from seed + i; Forge writes each image's own seed.
        self._save(_images(3))
        self.assertEqual(self._seeds(3), [42, 43, 44])

    def test_batch_without_forge_like_noise_keeps_the_batch_seed(self):
        cases = {"noise_off": {"master": True, "noise": False}, "master_off": {"master": False, "noise": True}}
        for stem, forge_like in cases.items():
            with self.subTest(forge_like=forge_like):
                self._save(_images(2), studio_state=_state_json(forge_like=forge_like), filename_prefix=f"AIStudio/{stem}")
                self.assertEqual(self._seeds(2, stem), [42, 42])

    def test_app_reader_reads_our_parameters(self):
        from core.image_metadata import read_metadata_for_ui

        self._save(_images(), studio_state=_state_json(hires={"on": True}))
        meta = read_metadata_for_ui(self.root / "AIStudio" / "anima_00001_.png")
        self.assertEqual(meta["source"], "webui")
        self.assertEqual(meta["prompt"], "1girl, solo")
        self.assertEqual(meta["negative"], "bad hands")
        self.assertEqual(meta["parameters"]["Seed"], 42)
        self.assertEqual(meta["parameters"]["Steps"], 20)
        self.assertEqual(meta["parameters"]["Hires upscale"], 1.5)
        # The ComfyUI graph rides along for the app's "open the workflow" paths.
        self.assertIn("KSampler", meta["raw_prompt"])
        self.assertIn("nodes", meta["raw_workflow"])

    def test_bad_state_json_is_reported_not_silently_defaulted(self):
        with self.assertRaisesRegex(ValueError, "studio_state"):
            self._save(_images(), studio_state="{not json")

    def test_hidden_inputs_are_optional(self):
        result = self._save(_images(), prompt=None, extra_pnginfo=None)
        with Image.open(self.root / "AIStudio" / "anima_00001_.png") as image:
            self.assertIn("parameters", image.info)
            self.assertNotIn("prompt", image.info)
            self.assertNotIn("workflow", image.info)
        self.assertEqual(len(result["ui"]["images"]), 1)

    def test_node_shape(self):
        cls = studio_nodes.AIStudioSaveForge
        self.assertTrue(cls.OUTPUT_NODE)
        spec = cls.INPUT_TYPES()
        self.assertEqual(list(spec["required"]), ["images", "studio_state", "seed", "filename_prefix"])
        self.assertEqual(spec["required"]["filename_prefix"][1]["default"], "AIStudio/anima")
        self.assertEqual(spec["hidden"], {"prompt": "PROMPT", "extra_pnginfo": "EXTRA_PNGINFO"})
        self.assertEqual(cls.RETURN_TYPES, ())


class _FakeImage:
    """Image-tensor stand-in with the two members the upscale node touches."""

    def __init__(self, width, height):
        self.shape = (1, height, width, 3)

    def movedim(self, *_):
        return self


class UpscaleImageTests(unittest.TestCase):
    def test_upscale_disabled_passthrough(self):
        image = object()
        node = studio_nodes.AIStudioUpscaleImage()
        with mock.patch.object(studio_nodes, "invoke_provider", side_effect=AssertionError("must not run")):
            self.assertIs(node.upscale(image, False, "x.pth", 2.0)[0], image)

    def test_enabled_runs_model_then_resizes_to_original_times_scale(self):
        image = _FakeImage(100, 50)
        calls = []

        def provider(name, **kwargs):
            calls.append((name, kwargs["args"]))
            return ("model",) if name == "UpscaleModelLoader" else (_FakeImage(400, 200),)

        def common_upscale(samples, width, height, method, crop):
            calls.append(("resize", width, height, method, crop))
            return samples

        comfy_utils = SimpleNamespace(common_upscale=common_upscale)
        with mock.patch.object(studio_nodes, "invoke_provider", side_effect=provider), \
                mock.patch.dict(sys.modules, {"comfy": SimpleNamespace(utils=comfy_utils), "comfy.utils": comfy_utils}):
            out = studio_nodes.AIStudioUpscaleImage().upscale(image, True, "4x.pth", 2.0)
        self.assertEqual(calls[0], ("UpscaleModelLoader", ("4x.pth",)))
        self.assertEqual(calls[1][0], "ImageUpscaleWithModel")
        self.assertEqual(calls[2], ("resize", 200, 100, "lanczos", "disabled"))
        self.assertEqual(len(out), 1)

    def test_enabled_without_a_model_is_an_error(self):
        with self.assertRaisesRegex(ValueError, "upscale model"):
            studio_nodes.AIStudioUpscaleImage().upscale(object(), True, "", 2.0)

    def test_node_shape_and_combo_lists_upscale_models(self):
        cls = studio_nodes.AIStudioUpscaleImage
        with mock.patch.dict(sys.modules, {"folder_paths": _FolderPaths(".", ["4x.pth", "2x.pth"])}):
            spec = cls.INPUT_TYPES()["required"]
        self.assertEqual(list(spec), ["image", "enabled", "model_name", "scale_by"])
        self.assertEqual(spec["model_name"][0], ["4x.pth", "2x.pth"])
        self.assertEqual(spec["scale_by"][1], {"default": 2.0, "min": 1.0, "max": 8.0, "step": 0.05})
        self.assertEqual(spec["enabled"][1]["default"], False)
        self.assertEqual(cls.RETURN_TYPES, ("IMAGE",))

    def test_both_nodes_are_registered(self):
        for name in ("AIStudioSaveForge", "AIStudioUpscaleImage"):
            self.assertIn(name, studio_nodes.NODE_CLASS_MAPPINGS)
            self.assertIn(name, studio_nodes.NODE_DISPLAY_NAME_MAPPINGS)


if __name__ == "__main__":
    unittest.main()
