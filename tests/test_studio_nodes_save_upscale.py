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

from comfy_custom_nodes.ai_studio_forge_parity import studio_nodes, studio_upscale


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
        # Forge-like noise drew image i from seed + i; Forge writes each image's own seed.
        self._save(_images(3), studio_state=_state_json(forge_like={"master": True}))
        self.assertEqual(self._seeds(3), [42, 43, 44])

    def test_an_inpaint_batch_keeps_the_batch_seed(self):
        # The inpaint detailer draws a batch from one seed, Forge-like noise or not.
        inpaint = {"tab": "i2i", "i2i_mode": "inpaint"}
        self._save(_images(2), studio_state=_state_json(forge_like={"master": True}, **inpaint))
        self.assertEqual(self._seeds(2), [42, 42])

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
            return ("model",)

        def common_upscale(samples, width, height, method, crop):
            calls.append(("resize", width, height, method, crop))
            return samples

        def model_pass(model, source, tile, overlap):
            calls.append(("model", model, tile, overlap))
            return _FakeImage(400, 200)

        comfy_utils = SimpleNamespace(common_upscale=common_upscale)
        with mock.patch.object(studio_upscale, "invoke_provider", side_effect=provider), \
                mock.patch.object(studio_upscale, "model_pass", side_effect=model_pass), \
                mock.patch.dict(sys.modules, {"comfy": SimpleNamespace(utils=comfy_utils), "comfy.utils": comfy_utils}):
            out = studio_nodes.AIStudioUpscaleImage().upscale(image, True, "4x.pth", 2.0)
        self.assertEqual(calls[0], ("UpscaleModelLoader", ("4x.pth",)))
        self.assertEqual(calls[1], ("model", "model", 512, 32))
        self.assertEqual(calls[2], ("resize", 200, 96, "lanczos", "disabled"))
        self.assertEqual(len(out), 1)

    def test_a_second_upscaler_is_blended_from_the_same_source(self):
        image = _FakeImage(100, 50)
        calls = []

        def upscale(source, name, scale, tile, overlap):
            calls.append((source, name, scale, tile, overlap))
            return 10.0 if name == "4x.pth" else 20.0

        with mock.patch.object(studio_upscale, "upscale", side_effect=upscale):
            out = studio_nodes.AIStudioUpscaleImage().upscale(
                image, True, "4x.pth", 2.0, model_name_2="2x.pth", upscaler_2_visibility=0.25, tile_size=256, tile_overlap=16)
        self.assertEqual(calls, [(image, "4x.pth", 2.0, 256, 16), (image, "2x.pth", 2.0, 256, 16)])
        self.assertEqual(out, (12.5,))

    def test_a_second_upscaler_at_zero_visibility_never_runs(self):
        with mock.patch.object(studio_upscale, "upscale", return_value=1.0) as upscale:
            studio_nodes.AIStudioUpscaleImage().upscale(_FakeImage(100, 50), True, "4x.pth", 2.0, model_name_2="2x.pth")
        self.assertEqual(upscale.call_count, 1)

    def test_scale_to_pastes_into_the_box(self):
        with mock.patch.object(studio_upscale, "upscale", return_value="big") as upscale, \
                mock.patch.object(studio_upscale, "paste_centered", return_value="boxed") as paste:
            out = studio_nodes.AIStudioUpscaleImage().upscale(
                _FakeImage(100, 50), True, "Lanczos", 2.0, resize_mode="Scale to", width=300, height=300)
        self.assertEqual(upscale.call_args.args[2], 6.0)
        paste.assert_called_once_with("big", 300, 300)
        self.assertEqual(out, ("boxed",))

    def test_a_missing_opencv_stops_colour_correction_before_any_upscale(self):
        with mock.patch.object(studio_upscale, "require_opencv", side_effect=RuntimeError("opencv-python")), \
                mock.patch.object(studio_upscale, "upscale", side_effect=AssertionError("must not run")), \
                self.assertRaisesRegex(RuntimeError, "opencv-python"):
            studio_nodes.AIStudioUpscaleImage().upscale(_FakeImage(100, 50), True, "Lanczos", 2.0, color_correction=True)

    def test_colour_correction_uses_the_picture_before_the_upscale(self):
        image = _FakeImage(100, 50)
        with mock.patch.object(studio_upscale, "upscale", return_value="big"), \
                mock.patch.object(studio_upscale, "match_colors", return_value="matched") as match:
            out = studio_nodes.AIStudioUpscaleImage().upscale(image, True, "Lanczos", 2.0, color_correction=True)
        match.assert_called_once_with("big", image)
        self.assertEqual(out, ("matched",))

    def test_enabled_without_a_model_is_an_error(self):
        with self.assertRaisesRegex(ValueError, "upscale model"):
            studio_nodes.AIStudioUpscaleImage().upscale(object(), True, "", 2.0)

    def test_node_shape_and_combo_lists_upscale_models(self):
        cls = studio_nodes.AIStudioUpscaleImage
        with mock.patch.dict(sys.modules, {"folder_paths": _FolderPaths(".", ["4x.pth", "2x.pth"])}):
            spec = cls.INPUT_TYPES()["required"]
        self.assertEqual(list(spec), ["image", "enabled", "model_name", "scale_by"])
        self.assertEqual(spec["model_name"][0], ["4x.pth", "2x.pth", "Lanczos", "Nearest"])
        self.assertEqual(spec["scale_by"][1], {"default": 2.0, "min": 1.0, "max": 8.0, "step": 0.05})
        self.assertEqual(spec["enabled"][1]["default"], False)
        self.assertEqual(cls.RETURN_TYPES, ("IMAGE",))

    def test_both_nodes_are_registered(self):
        for name in ("AIStudioSaveForge", "AIStudioUpscaleImage"):
            self.assertIn(name, studio_nodes.NODE_CLASS_MAPPINGS)
            self.assertIn(name, studio_nodes.NODE_DISPLAY_NAME_MAPPINGS)
