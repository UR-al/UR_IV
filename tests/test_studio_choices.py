"""Anima studio choice lists: SAM3 files, unet families, folder_paths / comfy.samplers kinds (no ComfyUI needed)."""
import json
import struct
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from comfy_custom_nodes.ai_studio_forge_parity import dora_infer_mode, generation, studio_choices


def _touch(directory: Path, *names: str) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    for name in names:
        (directory / name).write_bytes(b"x")


def _fake_folder_paths(base: Path, lists: dict, sam3_paths=None):
    def get_filename_list(kind):
        if kind not in lists:
            raise KeyError(kind)
        return list(lists[kind])

    def get_folder_paths(kind):
        if kind != "sam3" or sam3_paths is None:
            raise KeyError(kind)
        return list(sam3_paths)

    return SimpleNamespace(base_path=str(base), get_filename_list=get_filename_list, get_folder_paths=get_folder_paths)


def _safetensors(path: Path, *keys: str) -> None:
    """A file with only the header a reader of the key names opens."""
    header = json.dumps({key: {"dtype": "F32", "shape": [1], "data_offsets": [0, 4]} for key in keys}).encode("utf-8")
    path.write_bytes(struct.pack("<Q", len(header)) + header + b"\0\0\0\0")


class Sam3ChoicesTests(unittest.TestCase):
    def test_sam3_choices_filter_prefix(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _touch(root, "sam3.pt", "sam3.1_multiplex_fp16.safetensors", "other.json", "anima-lllite.safetensors")
            items = studio_choices.sam3_items([str(root)])
            self.assertEqual([item["label"] for item in items], ["sam3.1_multiplex_fp16.safetensors", "sam3.pt"])
            self.assertEqual(items[1]["value"], str(root / "sam3.pt"))
            self.assertTrue(all(Path(item["value"]).is_absolute() for item in items))

    def test_sam3_list_leaves_out_the_transformers_layout(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _safetensors(root / "sam3.safetensors", "detector_model.vision_encoder.a", "tracker_model.b")
            _safetensors(root / "sam3.1_multiplex_fp16.safetensors", "detector.backbone.a", "tracker.model.b")
            _touch(root, "sam3.pt")
            labels = [item["label"] for item in studio_choices.sam3_items([str(root)])]
            self.assertEqual(labels, ["sam3.1_multiplex_fp16.safetensors", "sam3.pt"])

    def test_missing_directory_and_first_folder_wins(self):
        with tempfile.TemporaryDirectory() as directory:
            first, second = Path(directory, "a"), Path(directory, "b")
            _touch(first, "sam3.pt")
            _touch(second, "sam3.pt", "sam3.safetensors")
            items = studio_choices.sam3_items([str(Path(directory, "none")), str(first), str(second)])
            self.assertEqual([(i["label"], Path(i["value"]).parent.name) for i in items],
                             [("sam3.pt", "a"), ("sam3.safetensors", "b")])

    def test_directories_include_comfy_models_and_forge_base_from_extra_paths(self):
        with tempfile.TemporaryDirectory() as directory:
            comfy, forge = Path(directory, "comfy"), Path(directory, "forge")
            _touch(comfy / "models" / "sam3", "sam3.pt")
            _touch(forge / "models" / "sam3", "sam3.1_multiplex_fp16.safetensors", "kohya-inpainting-v1.json")
            (comfy / "extra_model_paths.yaml").write_text(
                f"forge_neo:\n    base_path: {forge.as_posix()}/\n    loras: models/Lora\n", encoding="utf-8")
            paths = studio_choices.sam3_directories(_fake_folder_paths(comfy, {}))
            self.assertEqual([Path(p) for p in paths], [comfy / "models" / "sam3", forge / "models" / "sam3"])
            labels = [i["label"] for i in studio_choices.sam3_items(paths)]
            self.assertEqual(labels, ["sam3.pt", "sam3.1_multiplex_fp16.safetensors"])

    def test_extra_paths_sam3_key_overrides_folder(self):
        with tempfile.TemporaryDirectory() as directory:
            comfy, forge = Path(directory, "comfy"), Path(directory, "forge")
            (comfy).mkdir()
            (comfy / "extra_model_paths.yaml").write_text(
                f"forge_neo:\n    base_path: {forge.as_posix()}\n    sam3: |\n        weights/sam3\n", encoding="utf-8")
            paths = studio_choices.sam3_directories(_fake_folder_paths(comfy, {}))
            self.assertIn(forge / "weights" / "sam3", [Path(p) for p in paths])

    def test_broken_extra_paths_file_is_ignored(self):
        with tempfile.TemporaryDirectory() as directory:
            comfy = Path(directory)
            (comfy / "extra_model_paths.yaml").write_text("a: [unclosed\n", encoding="utf-8")
            paths = studio_choices.sam3_directories(_fake_folder_paths(comfy, {}))
            self.assertEqual([Path(p) for p in paths], [comfy / "models" / "sam3"])


class UnetFamilyTests(unittest.TestCase):
    def test_unet_families(self):
        names = ["anima-2.9b.safetensors", "anima-base.safetensors", "Anima_3.8B_v2.safetensors",
                 "x/anima-3_8b.safetensors", "anima38-ep38.safetensors", "bundle-without-size.safetensors"]
        families = studio_choices.unet_families(names, frozenset({"bundle-without-size.safetensors"}))
        self.assertEqual(families, {
            "anima-2.9b.safetensors": "2.9b", "anima-base.safetensors": "base",
            "Anima_3.8B_v2.safetensors": "3.8b", "x/anima-3_8b.safetensors": "3.8b",
            "anima38-ep38.safetensors": "2.9b", "bundle-without-size.safetensors": "3.8b"})

    def test_moss_and_krea_come_from_the_name(self):
        for name, family in (("moss_image_sft_bf16.safetensors", "moss"), ("MOSS-Image-Base.safetensors", "moss"),
                             ("krea2_turbo_bf16.safetensors", "krea2"), ("krea2Anime_v15.safetensors", "krea2"),
                             ("homofidelisKrea2NSFW_v10.safetensors", "krea2"), ("krea/base_3.8b.safetensors", "krea2")):
            with self.subTest(name=name):
                self.assertEqual(studio_choices.unet_family(name), family)

    def test_bundle_names_are_empty_outside_comfyui(self):
        self.assertEqual(studio_choices.anima38_bundle_names(), frozenset())

    def test_bundle_names_come_from_the_loader_node(self):
        loader = SimpleNamespace(INPUT_TYPES=lambda: {"required": {"model_name": (["a.safetensors"],)}})
        with mock.patch.dict("sys.modules", {}), \
                mock.patch("comfy_custom_nodes.ai_studio_forge_parity.anima38_nodes.ForgeNeoAnima38V2Loader", loader):
            self.assertEqual(studio_choices.anima38_bundle_names(), frozenset({"a.safetensors"}))

    def test_bundle_beats_a_plain_name(self):
        self.assertEqual(studio_choices.unet_family("anima-base.safetensors", frozenset({"anima-base.safetensors"})),
                         "3.8b")


class ChoicesKindTests(unittest.TestCase):
    def _run(self, kind, lists=None, samplers=None):
        lists = lists or {}
        with tempfile.TemporaryDirectory() as directory:
            folder_paths = _fake_folder_paths(Path(directory), lists)
            ksampler = SimpleNamespace(SAMPLERS=["euler", "exp_heun_2_x0_sde"],
                                       SCHEDULERS=["simple", "flow_cosmos_dynamic"])
            modules = {"folder_paths": folder_paths, "comfy": SimpleNamespace(),
                       "comfy.samplers": SimpleNamespace(KSampler=samplers or ksampler)}
            with mock.patch.dict("sys.modules", modules):
                return studio_choices.choices(kind)

    def test_samplers_and_schedulers_come_from_ksampler(self):
        self.assertEqual(self._run("samplers"), {"items": ["euler", "exp_heun_2_x0_sde"]})
        self.assertEqual(self._run("schedulers"), {"items": ["simple", "flow_cosmos_dynamic"]})

    def test_file_kinds(self):
        lists = {"vae": ["a.safetensors"], "text_encoders": ["te.safetensors"], "loras": ["x/l.safetensors"]}
        self.assertEqual(self._run("vaes", lists), {"items": ["a.safetensors"]})
        self.assertEqual(self._run("tes", lists), {"items": ["te.safetensors"]})
        self.assertEqual(self._run("loras", lists), {"items": ["x/l.safetensors"]})

    def test_unets_merge_unet_folder_and_report_families(self):
        lists = {"diffusion_models": ["a-base.safetensors", "b.safetensors"], "unet": ["b.safetensors", "c-3.8b.safetensors"]}
        result = self._run("unets", lists)
        self.assertEqual(result["items"], ["a-base.safetensors", "b.safetensors", "c-3.8b.safetensors"])
        self.assertEqual(result["families"], {"a-base.safetensors": "base", "b.safetensors": "2.9b",
                                               "c-3.8b.safetensors": "3.8b"})

    def test_upscalers_are_the_hires_fix_list(self):
        lists = {"upscale_models": ["4x-UltraSharp.pth"]}
        with tempfile.TemporaryDirectory() as directory:
            folder_paths = _fake_folder_paths(Path(directory), lists)
            with mock.patch.dict("sys.modules", {"folder_paths": folder_paths}):
                items = studio_choices.choices("upscalers")["items"]
                self.assertEqual(items, generation.upscale_method_choices())
        self.assertEqual(items[:2], ["latent:bislerp", "latent:bicubic"])
        self.assertIn("4x-UltraSharp.pth", items)

    def test_an_empty_upscale_folder_offers_only_what_needs_no_file(self):
        with tempfile.TemporaryDirectory() as directory:
            folder_paths = _fake_folder_paths(Path(directory), {"upscale_models": []})
            with mock.patch.dict("sys.modules", {"folder_paths": folder_paths}):
                items = studio_choices.choices("upscalers")["items"]
        self.assertNotIn("None", items)
        self.assertIn("Lanczos", items)

    def test_dora_modes_are_the_node_choices(self):
        self.assertEqual(self._run("dora_modes"), {"items": list(dora_infer_mode.MODE_CHOICES)})

    def test_sam3_items_are_objects(self):
        items = self._run("sam3")["items"]
        self.assertEqual(items, [])

    def test_controlnets_are_the_controlnet_files(self):
        self.assertEqual(self._run("controlnets", {"controlnet": ["lllite.safetensors"]}), {"items": ["lllite.safetensors"]})

    def test_face_models_read_forge_adetailer_and_keep_the_path(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            forge = root / "forge"
            (forge / "models" / "adetailer").mkdir(parents=True)
            (forge / "models" / "adetailer" / "face_yolov8n.pt").write_bytes(b"x")
            (forge / "models" / "adetailer" / "yolov8s-world.pt").write_bytes(b"x")
            (root / studio_choices.EXTRA_PATHS_FILE).write_text(f"forge:\n  base_path: {forge.as_posix()}\n", encoding="utf-8")
            folder_paths = _fake_folder_paths(root, {})
            folder_paths.base_path = str(root)
            with mock.patch.dict("sys.modules", {"folder_paths": folder_paths}):
                items = studio_choices.choices("face_models")["items"]
        self.assertEqual([item["label"] for item in items], ["face_yolov8n.pt"])
        self.assertTrue(items[0]["value"].endswith("face_yolov8n.pt"))
        self.assertTrue(Path(items[0]["value"]).is_absolute())

    def test_unknown_kind_raises(self):
        with self.assertRaises(ValueError):
            studio_choices.choices("nope")

    def test_without_comfyui_lists_are_empty(self):
        with mock.patch.dict("sys.modules", {"folder_paths": None, "comfy.samplers": None}):
            for kind in ("samplers", "schedulers", "vaes", "tes", "loras"):
                self.assertEqual(studio_choices.choices(kind), {"items": []})


if __name__ == "__main__":
    unittest.main()
