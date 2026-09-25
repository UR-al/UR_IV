"""Anima ControlNet-LLLite in the Comfy pack = the original kohya node (TR-C, parity_plan §7.3 C).

origin: kohya-ss/ComfyUI-Anima-LLLite@b7495bd8eb876e334509976896702484ed19cdbb
  nodes.py:49-63 (_target_cond_hw), :65-81 (_prepare_cond_image), :122-140 (INPUT_TYPES; :130-133
  strength/start/end/preserve_wrapper ranges), :154-177 (metadata: cond_in_channels default 3, a 4-channel
  weight needs a MASK, a 3-channel one ignores it), :203-278 (percent → σ window, model_function_wrapper)
  control_net_lllite_anima.py:481-484 (saved key prefixes), :487-520 (saved → internal layout)
origin: kohya-ss/sd-scripts@690ea7f96c23182352ec63def76d431c6120bd2f
  anima_minimal_inference.py:82-98 (guidance 3.5, negative "", 1024², 50 steps, flow shift 5.0),
  :226-227 (sizes divisible by 32), :546-552 (pure-noise start)
  anima_minimal_inference_control_net_lllite.py:65-74 (_load_control_image: RGB, PIL BICUBIC, /127.5 − 1),
  :400-406 (4-channel weights need a mask)
origin: forge_sam3_extension@861ac02 sam3ext/anima_core.py:405-423 (tile_repair_size),
  sam3ext/sam3_cn_lllite.py (forced_cn_module, lllite_tile_repair_from_header)

The pack copies the kohya node unchanged into vendor/comfyui_anima_lllite/ (hash-pinned below) and calls its
``apply``. The end-to-end tests run that vendored node on a tiny CPU "DiT" with real LLLite weights written in
the saved v2 format, so the comparison target is the original class itself, not a re-implementation.
"""
from __future__ import annotations

import ast
import contextlib
import hashlib
import importlib
import json
import struct
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

from comfy_custom_nodes.ai_studio_forge_parity import anima_lllite, sam3_nodes
from tests._optional_deps import bind_torch, requires_torch

# torch 는 표시된 클래스의 setUpClass 에서 bind_torch(globals()) 로 채운다.
torch = None

ROOT = Path(__file__).resolve().parents[1]
PACK_ROOT = ROOT / "comfy_custom_nodes" / "ai_studio_forge_parity"
VENDOR_ROOT = PACK_ROOT / "vendor" / "comfyui_anima_lllite"
UPSTREAM_COMMIT = "b7495bd8eb876e334509976896702484ed19cdbb"
VENDOR_PACKAGE = f"{anima_lllite.__package__}.vendor.comfyui_anima_lllite"
VENDOR_MODULES = (
    VENDOR_PACKAGE,
    f"{VENDOR_PACKAGE}.nodes",
    f"{VENDOR_PACKAGE}.control_net_lllite_anima",
)
# 원본 blob(LF) 그대로의 SHA-256 — vendor/comfyui_anima_lllite/UPSTREAM.md 표와 같다.
EXPECTED_VENDOR_SHA256 = {
    "__init__.py": "4c0f96b0f0006df5a559d09d44b8d521a32d7992383223310b5ba2a3f88e3e42",
    "nodes.py": "5adffdab1a71d7da463e269d6656b8f9ef0d8720340dd02569793dbaf48383cb",
    "control_net_lllite_anima.py": "91611fb108d25f6e402d7552326080bd7d70000cd05733239defb92e3bf4d308",
}
LICENSE_SHA256 = "c71d239df91726fc519c6eb72d318ec65820627232b2f796219e87dcf35d0ab4"
CARD_PROMPT = (
    "repair the low-quality anime image, reduce blur and compression artifacts, "
    "preserve the original composition"
)


def _sha256_lf(path: Path) -> str:
    return hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def _write_header(path: Path, tensors: dict, metadata: dict | None = None) -> Path:
    """A header-only .safetensors (the reader never touches tensor bytes)."""
    header = {
        name: {"dtype": "F32", "shape": list(shape), "data_offsets": [0, 0]}
        for name, shape in tensors.items()
    }
    if metadata is not None:
        header["__metadata__"] = metadata
    blob = json.dumps(header).encode("utf-8")
    path.write_bytes(struct.pack("<Q", len(blob)) + blob)
    return path


# Saved v2 key layout (control_net_lllite_anima.py:481-484, README 'Weight format').
_ANIMA_V2_KEYS = {
    "lllite_conditioning1.conv1.weight": (16, 3, 4, 4),
    "lllite_dit_blocks_0_self_attn_q_proj.down.weight": (64, 2048),
    "lllite_dit_blocks_0_self_attn_q_proj.depth_embed": (32,),
}
_SDXL_LLLITE_KEYS = {
    "lllite_unet_input_blocks_4_1_transformer_blocks_0_attn1_to_q.conditioning1.0.weight": (16, 3, 4, 4),
}
_CONTROLNET_KEYS = {"control_model.input_blocks.0.0.weight": (320, 4, 3, 3)}
_LEGACY_LLLITE_KEYS = {"lllite_modules.0.down.weight": (64, 2048)}


class _FolderPaths(types.ModuleType):
    """ComfyUI ``folder_paths`` stand-in with one ``controlnet`` folder (get_full_path semantics)."""

    def __init__(self, root: Path):
        super().__init__("folder_paths")
        self.folder_names_and_paths = {"controlnet": ([str(root)], {".safetensors", ".pth"})}

    def get_full_path(self, folder, filename):
        for directory in self.folder_names_and_paths[folder][0]:
            candidate = Path(directory) / filename
            if candidate.is_file():
                return str(candidate)
        return None

    def get_filename_list(self, folder):
        names = []
        for directory in self.folder_names_and_paths[folder][0]:
            base = Path(directory)
            names.extend(
                path.relative_to(base).as_posix() for path in sorted(base.rglob("*")) if path.is_file()
            )
        return names


class _TempControlnetFolder:
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.folder = Path(self._tmp.name) / "controlnet"
        self.folder.mkdir()
        self.folder_paths = _FolderPaths(self.folder)
        patcher = mock.patch.dict(sys.modules, {"folder_paths": self.folder_paths})
        patcher.start()
        self.addCleanup(patcher.stop)


# ─────────────────────────────────────────────────────────────────────────────────────
# torch-free: vendoring, original input ranges, header detection, loader branch
# ─────────────────────────────────────────────────────────────────────────────────────

class TestAnimaLLLiteVendoring(unittest.TestCase):
    def test_vendored_files_match_pinned_upstream(self):
        copied = {
            path.relative_to(VENDOR_ROOT).as_posix()
            for path in VENDOR_ROOT.rglob("*.py")
            if "__pycache__" not in path.parts
        }
        self.assertEqual(copied, set(EXPECTED_VENDOR_SHA256))
        self.assertEqual(
            {name: _sha256_lf(VENDOR_ROOT / name) for name in EXPECTED_VENDOR_SHA256},
            EXPECTED_VENDOR_SHA256,
        )
        manifest = (VENDOR_ROOT / "UPSTREAM.md").read_text(encoding="utf-8")
        self.assertIn(UPSTREAM_COMMIT, manifest)
        for digest in EXPECTED_VENDOR_SHA256.values():
            self.assertIn(digest, manifest)

    def test_apache_license_and_notice_ship_with_the_pack(self):
        self.assertEqual(
            _sha256_lf(PACK_ROOT / "LICENSES" / "ComfyUI-Anima-LLLite-Apache-2.0.txt"), LICENSE_SHA256,
        )
        notice = (PACK_ROOT / "THIRD_PARTY_NOTICES.md").read_text(encoding="utf-8")
        self.assertIn(UPSTREAM_COMMIT, notice)
        self.assertIn("LICENSES/ComfyUI-Anima-LLLite-Apache-2.0.txt", notice)

    def test_docs_match_the_sd_scripts_sampling_and_credit_it(self):
        # tile_repair_sigmas repeats sd-scripts' get_timesteps_sigmas (Apache-2.0): the notice credits
        # it, and the README no longer claims a Comfy ``simple`` schedule for Tile & Repair.
        notice = (PACK_ROOT / "THIRD_PARTY_NOTICES.md").read_text(encoding="utf-8")
        self.assertNotIn("no sd-scripts code is copied", notice)
        self.assertIn("get_timesteps_sigmas", notice)
        self.assertIn("library/hunyuan_image_utils.py", notice)
        readme = (PACK_ROOT / "README.md").read_text(encoding="utf-8")
        self.assertNotIn("`euler`/`simple`", readme)
        self.assertIn("get_timesteps_sigmas", readme)

    def test_the_pack_never_registers_the_upstream_node(self):
        from comfy_custom_nodes.ai_studio_forge_parity import NODE_CLASS_MAPPINGS

        self.assertIn("ForgeNeoAnimaTileRepair", NODE_CLASS_MAPPINGS)
        self.assertNotIn("AnimaLLLiteApply_sdscripts", NODE_CLASS_MAPPINGS)
        # The pack imports with ComfyUI absent: the vendored node (torch, folder_paths) stays unloaded.
        self.assertNotIn(f"{VENDOR_PACKAGE}.nodes", sys.modules)


def _original_apply_inputs() -> dict:
    """``AnimaLLLiteApply_sdscripts.INPUT_TYPES()["required"]`` literals read from the vendored file
    (nodes.py:124-140) without importing it — it imports torch and folder_paths at module level."""
    tree = ast.parse((VENDOR_ROOT / "nodes.py").read_text(encoding="utf-8"))
    node_class = next(
        item for item in tree.body
        if isinstance(item, ast.ClassDef) and item.name == "AnimaLLLiteApply_sdscripts"
    )
    method = next(
        item for item in node_class.body
        if isinstance(item, ast.FunctionDef) and item.name == "INPUT_TYPES"
    )
    returned = next(item for item in ast.walk(method) if isinstance(item, ast.Return)).value
    required = next(
        value for key, value in zip(returned.keys, returned.values) if ast.literal_eval(key) == "required"
    )
    inputs = {}
    for key, value in zip(required.keys, required.values):
        with contextlib.suppress(ValueError):   # lllite_name: folder_paths call, not a literal
            inputs[ast.literal_eval(key)] = ast.literal_eval(value)
    return inputs


class TestTileRepairNodeInputs(unittest.TestCase):
    def setUp(self):
        with mock.patch.dict(sys.modules, {"folder_paths": _FolderPaths(Path(tempfile.gettempdir()) / "none")}):
            self.inputs = anima_lllite.ForgeNeoAnimaTileRepair.INPUT_TYPES()["required"]

    def test_lllite_inputs_equal_the_original_node(self):
        # origin: nodes.py:130-133 — strength 1.0 [-10, 10] .01, start 0 / end 1 [0, 1] .001, preserve_wrapper True.
        original = _original_apply_inputs()
        names = ("strength", "start_percent", "end_percent", "preserve_wrapper")
        self.assertEqual({name: original[name] for name in names}, {
            "strength": ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.01}),
            "start_percent": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.001}),
            "end_percent": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.001}),
            "preserve_wrapper": ("BOOLEAN", {"default": True}),
        })
        for name in names:
            with self.subTest(name=name):
                self.assertEqual(self.inputs[name], original[name])
                self.assertEqual(anima_lllite.LLLITE_APPLY_INPUTS[name], original[name])

    def test_pipeline_defaults_are_the_sd_scripts_defaults(self):
        # origin: anima_minimal_inference.py:82-98 — guidance 3.5, negative "", 1024x1024, 50 steps, shift 5.0.
        defaults = {name: spec[1]["default"] for name, spec in self.inputs.items() if len(spec) > 1}
        self.assertEqual(defaults["steps"], 50)
        self.assertEqual(defaults["cfg"], 3.5)
        self.assertEqual(defaults["flow_shift"], 5.0)
        self.assertEqual(defaults["negative"], "")
        self.assertEqual(defaults["short_side"], 1024)
        # civitai 2708551 suggested prompt = the extension panel default.
        self.assertEqual(defaults["positive"], CARD_PROMPT)
        short = self.inputs["short_side"][1]
        self.assertEqual((short["min"], short["max"], short["step"]), (256, 4096, 32))
        # sd-scripts Euler flow step from pure noise → Comfy ``euler`` on the script's own σ list
        # (TestTileRepairSamplingOrigin), never Comfy's ``simple`` table lookup.
        self.assertEqual(
            (anima_lllite.TILE_REPAIR_SAMPLER, anima_lllite.TILE_REPAIR_DENOISE), ("euler", 1.0),
        )
        self.assertFalse(hasattr(anima_lllite, "TILE_REPAIR_SCHEDULER"))
        self.assertEqual(tuple(self.inputs)[:5], ("model", "clip", "vae", "image", "lllite_name"))
        self.assertEqual(anima_lllite.ForgeNeoAnimaTileRepair.RETURN_TYPES, ("IMAGE",))

    def test_size_keeps_the_source_aspect_on_multiples_of_32(self):
        # Extension anima_core.tile_repair_size (= sd-scripts check_inputs /32, 256 floor).
        cases = {
            (1024, 1024, 1024): (1024, 1024),
            (1920, 1080, 1024): (1792, 1024),   # int(1920 * 1024 / 1080) = 1820 → 1792
            (1080, 1920, 1024): (1024, 1792),
            (1000, 1500, 1024): (1024, 1536),
            (1032, 1032, 1024): (1024, 1024),
            (96, 64, 256): (384, 256),
            (100, 100, 300): (288, 288),
            (4000, 100, 256): (10240, 256),
        }
        for (width, height, edge), expected in cases.items():
            with self.subTest(source=(width, height), short_side=edge):
                self.assertEqual(anima_lllite.tile_repair_size(width, height, edge), expected)
        with self.assertRaises(ValueError):
            anima_lllite.tile_repair_size(0, 10, 1024)


class TestLLLiteHeaderDetection(_TempControlnetFolder, unittest.TestCase):
    def _info(self, name, tensors, metadata=None):
        return anima_lllite.lllite_info(_write_header(self.folder / name, tensors, metadata))

    def test_anima_lllite_is_recognised_by_its_conditioning_trunk_keys(self):
        info = self._info("lineart.safetensors", _ANIMA_V2_KEYS, {"lllite.cond_in_channels": "3"})
        self.assertEqual(info, anima_lllite.LLLiteInfo(channels=3, tile_repair=False))
        info = self._info("inpaint.safetensors", _ANIMA_V2_KEYS, {"lllite.cond_in_channels": "4"})
        self.assertEqual(info, anima_lllite.LLLiteInfo(channels=4, tile_repair=False))

    def test_missing_channel_metadata_defaults_to_three_like_the_original(self):
        # origin: nodes.py:168 int(meta.get("lllite.cond_in_channels", 3))
        self.assertEqual(self._info("noname.safetensors", _ANIMA_V2_KEYS).channels, 3)
        self.assertEqual(self._info("empty_meta.safetensors", _ANIMA_V2_KEYS, {}).channels, 3)
        # A value the original's int() rejects: recognised, channels unknown (the node raises when run).
        self.assertIsNone(
            self._info("broken.safetensors", _ANIMA_V2_KEYS, {"lllite.cond_in_channels": "x"}).channels
        )

    def test_other_files_are_not_anima_lllite(self):
        for name, keys in (
            ("sdxl_controllllite.safetensors", _SDXL_LLLITE_KEYS),
            ("controlnet.safetensors", _CONTROLNET_KEYS),
            ("legacy_lllite.safetensors", _LEGACY_LLLITE_KEYS),
        ):
            with self.subTest(name=name):
                self.assertIsNone(self._info(name, keys, {"lllite.cond_in_channels": "3"}))
        (self.folder / "short.safetensors").write_bytes(b"\x01\x02")
        (self.folder / "huge.safetensors").write_bytes(struct.pack("<Q", 1 << 40) + b"{}")
        (self.folder / "garbage.safetensors").write_bytes(struct.pack("<Q", 4) + b"\xff\xfe{]")
        _write_header(self.folder / "tile.pth", _ANIMA_V2_KEYS)
        for name in ("short.safetensors", "huge.safetensors", "garbage.safetensors", "tile.pth", "absent.safetensors"):
            with self.subTest(name=name):
                self.assertIsNone(anima_lllite.lllite_info(self.folder / name))

    def test_tile_repair_is_read_from_the_title_then_the_name(self):
        # Extension sam3_cn_lllite.lllite_tile_repair_from_header: v1.0 'anima_tiled_lllite_v1',
        # v2.0 'anima_tile_multitask_v1'; without a title the file name decides; 3-channel only.
        cases = {
            ("a.safetensors", "anima_tiled_lllite_v1", "3"): True,
            ("b.safetensors", "anima_tile_multitask_v1", "3"): True,
            ("animaTileRepair_v20.safetensors", "anima_lineart_v1", "3"): False,
            ("animaTileRepair_v10.safetensors", None, "3"): True,
            ("anima_lllite_lineart.safetensors", None, "3"): False,
            ("anima_tile_inpaint.safetensors", "anima_tile_inpaint", "4"): False,
        }
        for (name, title, channels), expected in cases.items():
            metadata = {"lllite.cond_in_channels": channels}
            if title is not None:
                metadata["modelspec.title"] = title
            with self.subTest(name=name, title=title):
                self.assertIs(self._info(name, _ANIMA_V2_KEYS, metadata).tile_repair, expected)

    def test_tile_repair_list_holds_three_channel_lllites_and_defaults_to_the_newest(self):
        _write_header(self.folder / "animaTileRepair_v10.safetensors", _ANIMA_V2_KEYS)
        _write_header(self.folder / "animaTileRepair_v20.safetensors", _ANIMA_V2_KEYS)
        _write_header(self.folder / "anima_lineart.safetensors", _ANIMA_V2_KEYS, {"lllite.cond_in_channels": "3"})
        _write_header(self.folder / "anima-lllite-inpainting-v2.safetensors", _ANIMA_V2_KEYS,
                      {"lllite.cond_in_channels": "4"})
        _write_header(self.folder / "control_v11f1e_sd15_tile.safetensors", _CONTROLNET_KEYS)
        _write_header(self.folder / "animaTileRepair_v30.pth", _ANIMA_V2_KEYS)
        choices, default = anima_lllite.tile_repair_lllite_choices()
        self.assertEqual(choices, [
            "anima_lineart.safetensors",
            "animaTileRepair_v10.safetensors",
            "animaTileRepair_v20.safetensors",
        ])
        self.assertEqual(default, "animaTileRepair_v20.safetensors")
        spec = anima_lllite.ForgeNeoAnimaTileRepair.INPUT_TYPES()["required"]["lllite_name"]
        self.assertEqual(spec, (choices, {"default": "animaTileRepair_v20.safetensors"}))

    def test_empty_folder_lists_none(self):
        _write_header(self.folder / "controlnet.safetensors", _CONTROLNET_KEYS)
        self.assertEqual(anima_lllite.tile_repair_lllite_choices(), (["None"], "None"))

    def test_forced_module_matches_the_extension_rules(self):
        tile = anima_lllite.AnimaLLLiteControl(Path("t"), "tile", 3, True)
        lineart = anima_lllite.AnimaLLLiteControl(Path("l"), "lineart", 3, False)
        inpaint = anima_lllite.AnimaLLLiteControl(Path("i"), "inpaint", 4, False)
        cases = [
            (tile, "inpaint_only", "None", True),
            (tile, "tile_resample", "None", True),
            (tile, "", "None", True),               # empty = the detailer default inpaint_only
            (tile, "None", "None", False),
            (tile, "none", "none", False),
            (lineart, "inpaint_only", "None", True),
            (lineart, "inpaint_only+lama", "None", True),
            (lineart, "lineart_anime", "lineart_anime", False),
            (inpaint, "inpaint_global_harmonious", "None", True),
            (inpaint, "canny", "canny", False),
        ]
        for control, module, expected, changed in cases:
            with self.subTest(control=control.name, module=module):
                used, reason = anima_lllite.forced_control_module(module, control)
                self.assertEqual(used, expected)
                self.assertIs(reason is not None, changed)


class TestSam3LoaderBranch(_TempControlnetFolder, unittest.TestCase):
    """sam3_nodes._load_controlnet: an Anima LLLite never reaches ControlNetLoader."""

    class _RefusingLoader:
        @classmethod
        def load_controlnet(cls, control_net_name):
            raise AssertionError(f"ControlNetLoader must not load {control_net_name}")

    def test_named_lllite_becomes_a_model_patch_handle(self):
        _write_header(self.folder / "animaTileRepair_v20.safetensors", _ANIMA_V2_KEYS,
                      {"lllite.cond_in_channels": "3", "modelspec.title": "anima_tile_multitask_v1"})
        with mock.patch.object(sam3_nodes, "_node_mappings",
                               return_value={"ControlNetLoader": self._RefusingLoader}):
            control, source = sam3_nodes._load_controlnet(None, " animaTileRepair_v20.safetensors ")
        self.assertTrue(anima_lllite.is_lllite_control(control))
        self.assertEqual(source, "animaTileRepair_v20.safetensors")
        self.assertEqual((control.channels, control.tile_repair), (3, True))
        self.assertEqual(control.path, self.folder / "animaTileRepair_v20.safetensors")

    def test_absolute_path_lllite_is_recognised_too(self):
        outside = Path(self._tmp.name) / "elsewhere"
        outside.mkdir()
        path = _write_header(outside / "anima-lllite-inpainting-v2.safetensors", _ANIMA_V2_KEYS,
                             {"lllite.cond_in_channels": "4"})
        with mock.patch.object(sam3_nodes, "_node_mappings",
                               return_value={"ControlNetLoader": self._RefusingLoader}):
            control, source = sam3_nodes._load_controlnet(None, str(path))
        self.assertEqual((source, control.name, control.channels), (path.name, path.name, 4))

    def test_ordinary_controlnet_still_uses_controlnet_loader(self):
        _write_header(self.folder / "cn.safetensors", _CONTROLNET_KEYS)
        loaded = []

        class Loader:
            @classmethod
            def load_controlnet(cls, control_net_name):
                loaded.append(control_net_name)
                return ("CONTROL_NET",)

        with mock.patch.object(sam3_nodes, "_node_mappings", return_value={"ControlNetLoader": Loader}):
            self.assertEqual(sam3_nodes._load_controlnet(None, "cn.safetensors"), ("CONTROL_NET", "cn.safetensors"))
        self.assertEqual(loaded, ["cn.safetensors"])

    def test_connected_control_net_input_wins(self):
        handle = anima_lllite.AnimaLLLiteControl(Path("x"), "x", 3, True)
        self.assertEqual(sam3_nodes._load_controlnet(handle, "ignored"), (handle, "connected CONTROL_NET"))

    def test_patch_model_validates_like_a_controlnet(self):
        control = anima_lllite.AnimaLLLiteControl(self.folder / "x.safetensors", "x.safetensors", 3, False)
        with self.assertRaisesRegex(ValueError, "control_mode"):
            control.patch_model(object(), None, None, 1.0, 0.0, 1.0, "Other")
        with self.assertRaisesRegex(ValueError, "strength"):
            control.patch_model(object(), None, None, 10.5, 0.0, 1.0)
        with self.assertRaisesRegex(ValueError, "start < end"):
            control.patch_model(object(), None, None, 1.0, 0.5, 0.5)


# ─────────────────────────────────────────────────────────────────────────────────────
# torch (CPU): the detailer pass, and the vendored node end to end
# ─────────────────────────────────────────────────────────────────────────────────────

@requires_torch
class TestSam3DetailerLLLitePass(_TempControlnetFolder, unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        bind_torch(globals())

    def _detail(self, model_name, module, *, channels_meta, strength=0.8):
        _write_header(self.folder / model_name, _ANIMA_V2_KEYS, channels_meta)
        source = torch.rand((1, 32, 48, 3), generator=torch.Generator().manual_seed(3))
        mask = torch.zeros((1, 32, 48))
        mask[:, 8:24, 16:40] = 1.0
        applied, sampled_models, sampled_conds = [], [], []
        patched = object()
        model, positive, negative = object(), [["pos", {}]], [["neg", {}]]

        def fake_apply(model_in, lllite_name, image, strength_in, start, end, *, preserve_wrapper, mask):
            applied.append({
                "model": model_in, "name": lllite_name, "image": image.clone(),
                "strength": strength_in, "window": (start, end),
                "preserve_wrapper": preserve_wrapper, "mask": None if mask is None else mask.clone(),
            })
            return patched

        def fake_sample(model_in, seed, steps, cfg, sampler, scheduler, pos, neg, latent, *rest):
            sampled_models.append(model_in)
            sampled_conds.append((pos, neg))
            return latent

        def fake_encode(vae, pixels, mask, grow_mask_by, **kwargs):
            return {"samples": torch.zeros((1, 4, 1, 1)), "shape": tuple(pixels.shape), "mask": mask.clone()}

        with (
            mock.patch.object(anima_lllite, "apply_lllite", side_effect=fake_apply),
            mock.patch.object(sam3_nodes, "_apply_controlnet",
                              side_effect=AssertionError("an LLLite is not a CONTROL_NET")),
            mock.patch.object(sam3_nodes, "_node_mappings", return_value={}),
            mock.patch.object(sam3_nodes, "_vae_encode_for_inpaint", side_effect=fake_encode),
            mock.patch.object(sam3_nodes, "_sample_latent", side_effect=fake_sample),
            mock.patch.object(sam3_nodes, "_vae_decode",
                              side_effect=lambda vae, latent: torch.ones(latent["shape"])),
        ):
            _, _, report_json = sam3_nodes.ForgeNeoSAM3Detailer().detail(
                source, mask, model, object(), object(), positive, negative,
                only_masked=False, grow_mask_by=0,
                controlnet_enable=True, controlnet_model_name=model_name,
                controlnet_module=module, controlnet_strength=strength,
                controlnet_settings_json=json.dumps(sam3_nodes._CONTROLNET_EXTRA_DEFAULTS),
            )
        return source, mask, model, patched, (positive, negative), applied, sampled_models, sampled_conds, \
            json.loads(report_json)

    def test_tile_repair_lllite_patches_the_model_with_the_unprocessed_image(self):
        (source, _mask, model, patched, conds, applied, sampled_models, sampled_conds,
         report) = self._detail("animaTileRepair_v20.safetensors", "inpaint_only",
                                channels_meta={"lllite.cond_in_channels": "3"})
        self.assertEqual(len(applied), 1)
        call = applied[0]
        self.assertIs(call["model"], model)
        self.assertEqual(call["name"], "animaTileRepair_v20.safetensors")
        self.assertEqual(call["strength"], 0.8)
        self.assertEqual(call["window"], (0.0, 1.0))
        self.assertIs(call["preserve_wrapper"], True)
        # 3-channel weights: no mask (the original would warn and drop it).
        self.assertIsNone(call["mask"])
        # Preprocessor forced to None: the control image is the pass image, not an inpaint hint
        # with the masked region set to -1.
        self.assertEqual(tuple(call["image"].shape), (1, 32, 48, 3))
        self.assertTrue(torch.allclose(call["image"], source, atol=1e-6))
        self.assertGreaterEqual(float(call["image"].min()), 0.0)
        # The sampler runs the patched MODEL with the conditioning untouched.
        self.assertEqual(sampled_models, [patched])
        self.assertIs(sampled_conds[0][0], conds[0])
        self.assertIs(sampled_conds[0][1], conds[1])
        control = report["passes"][0]["controlnet"]
        self.assertEqual(control["module"], "None")
        self.assertEqual(control["preprocessor"], "none")
        self.assertEqual(control["translation"], "anima_lllite_model_patch")
        self.assertEqual(control["lllite_cond_in_channels"], 3)
        self.assertIs(control["lllite_mask"], False)
        self.assertIn("inpaint_only", control["module_override"])
        self.assertTrue(report["controlnet_enabled"])

    def test_inpaint_lllite_gets_the_pass_mask(self):
        (_source, mask, _model, patched, _conds, applied, sampled_models, _sampled_conds,
         report) = self._detail("anima-lllite-inpainting-v2.safetensors", "inpaint_only",
                                channels_meta={"lllite.cond_in_channels": "4"})
        call = applied[0]
        self.assertIsNotNone(call["mask"])
        self.assertTrue(torch.equal(call["mask"], mask))
        self.assertEqual(sampled_models, [patched])
        control = report["passes"][0]["controlnet"]
        self.assertEqual((control["module"], control["lllite_mask"]), ("None", True))

    def _detail_batch(self, model_name, module, *, channels_meta, batch=2, settings=None,
                      masks=None):
        """A batch-``batch`` whole-image detailer pass with one fake LLLite patch per apply call."""
        _write_header(self.folder / model_name, _ANIMA_V2_KEYS, channels_meta)
        source = torch.rand((batch, 32, 48, 3), generator=torch.Generator().manual_seed(5))
        if masks is None:
            masks = torch.zeros((batch, 32, 48))
            masks[:, 8:24, 16:40] = 1.0
        applied, patches, sampled, decoded = [], [], [], []
        encoded = {}

        def fake_apply(model_in, lllite_name, image, strength_in, start, end, *, preserve_wrapper, mask):
            applied.append({"image": image.clone(), "mask": None if mask is None else mask.clone()})
            patches.append(object())
            return patches[-1]

        def fake_encode(vae, pixels, mask, grow_mask_by, **kwargs):
            encoded["samples"] = torch.arange(1.0, pixels.shape[0] + 1.0).view(-1, 1, 1, 1).expand(-1, 4, 2, 3).clone()
            encoded["noise_mask"] = mask.unsqueeze(1).clone()
            return {"samples": encoded["samples"], "noise_mask": encoded["noise_mask"]}

        def fake_sample(model_in, seed, steps, cfg, sampler, scheduler, pos, neg, latent, *rest):
            sampled.append({"model": model_in, "seed": seed, "latent": dict(latent)})
            # The decoded value tells which patch sampled this item.
            value = 0.25 * (patches.index(model_in) + 1)
            return {**latent, "samples": torch.full_like(latent["samples"], value)}

        def fake_decode(vae, latent):
            decoded.append(latent["samples"].clone())
            return latent["samples"][:, 0, 0, 0].view(-1, 1, 1, 1) * torch.ones((1, 32, 48, 3))

        with (
            mock.patch.object(anima_lllite, "apply_lllite", side_effect=fake_apply),
            mock.patch.object(sam3_nodes, "_apply_controlnet",
                              side_effect=AssertionError("an LLLite is not a CONTROL_NET")),
            mock.patch.object(sam3_nodes, "_node_mappings", return_value={}),
            mock.patch.object(sam3_nodes, "_vae_encode_for_inpaint", side_effect=fake_encode),
            mock.patch.object(sam3_nodes, "_sample_latent", side_effect=fake_sample),
            mock.patch.object(sam3_nodes, "_vae_decode", side_effect=fake_decode),
        ):
            image_out, _, report_json = sam3_nodes.ForgeNeoSAM3Detailer().detail(
                source, masks, object(), object(), object(), [["pos", {}]], [["neg", {}]],
                only_masked=False, grow_mask_by=0, seed=77,
                controlnet_enable=True, controlnet_model_name=model_name,
                controlnet_module=module, controlnet_strength=1.0,
                controlnet_settings_json=json.dumps(settings or sam3_nodes._CONTROLNET_EXTRA_DEFAULTS),
            )
        return {
            "source": source, "masks": masks, "image": image_out, "applied": applied,
            "patches": patches, "sampled": sampled, "decoded": decoded, "encoded": encoded,
            "report": json.loads(report_json),
        }

    def test_batch_pass_controls_each_image_by_its_own_image(self):
        # origin: kohya-ss/ComfyUI-Anima-LLLite@b7495bd8:nodes.py:75 `img = img[:1]  # use first frame only`,
        #   :99 `m = m[:1]`; control_net_lllite_anima.py:266-270 repeats that one embedding over the batch.
        # origin: forge_sam3_extension@3522928:scripts/!sam3.py:536 — Forge runs SAM3 once per image
        #   (postprocess_image), so every image is repaired toward itself.
        run = self._detail_batch("animaTileRepair_v20.safetensors", "inpaint_only",
                                 channels_meta={"lllite.cond_in_channels": "3"})
        source, applied, sampled = run["source"], run["applied"], run["sampled"]
        # One node call per image, each with only that image as the control image.
        self.assertEqual(len(applied), 2)
        for index, call in enumerate(applied):
            self.assertEqual(tuple(call["image"].shape), (1, 32, 48, 3))
            self.assertTrue(torch.allclose(call["image"], source[index:index + 1], atol=1e-6))
            self.assertIsNone(call["mask"])
        # Item i is sampled under its own patch, with its own latent slice, and batch_index [i] so
        # Comfy's prepare_noise gives it the batched run's initial noise for that item.
        self.assertEqual([entry["model"] for entry in sampled], run["patches"])
        self.assertEqual([entry["seed"] for entry in sampled], [77, 77])
        for index, entry in enumerate(sampled):
            latent = entry["latent"]
            self.assertEqual(latent["batch_index"], [index])
            self.assertTrue(torch.equal(latent["samples"], run["encoded"]["samples"][index:index + 1]))
            self.assertTrue(torch.equal(latent["noise_mask"], run["encoded"]["noise_mask"][index:index + 1]))
        # The items are decoded as one batch, in order, and composited into their own image.
        self.assertEqual(len(run["decoded"]), 1)
        self.assertEqual(run["decoded"][0][:, 0, 0, 0].tolist(), [0.25, 0.5])
        masked = run["masks"] > 0.5
        for index, value in enumerate((0.25, 0.5)):
            self.assertTrue(torch.allclose(run["image"][index][masked[index]], torch.full((1,), value)))
            self.assertTrue(torch.equal(run["image"][index][~masked[index]], source[index][~masked[index]]))
        control = run["report"]["passes"][0]["controlnet"]
        self.assertEqual(control["lllite_per_image"], 2)
        self.assertEqual((control["module"], control["lllite_mask"]), ("None", False))

    def test_batch_inpaint_lllite_gets_each_image_its_own_mask(self):
        # origin: nodes.py:99 `m = m[:1]` — a 4-channel weight would see image 0's mask for every image.
        masks = torch.zeros((2, 32, 48))
        masks[0, 4:12, 4:20] = 1.0
        masks[1, 16:30, 20:44] = 1.0
        run = self._detail_batch("anima-lllite-inpainting-v2.safetensors", "inpaint_only",
                                 channels_meta={"lllite.cond_in_channels": "4"}, masks=masks)
        self.assertEqual(len(run["applied"]), 2)
        for index, call in enumerate(run["applied"]):
            self.assertTrue(torch.equal(call["mask"], masks[index:index + 1]))
        control = run["report"]["passes"][0]["controlnet"]
        self.assertEqual((control["lllite_per_image"], control["lllite_mask"]), (2, True))

    def test_thresholds_left_from_another_preprocessor_are_ignored_under_none(self):
        # origin: sd-webui-forge-classic@e33f40e4:modules_forge/supported_preprocessor.py:68-69 — the None
        #   preprocessor returns input_image and ignores slider_1/slider_2 (threshold_a/b).
        base = dict(sam3_nodes._CONTROLNET_EXTRA_DEFAULTS)
        cases = (
            # The pack forces Tile & Repair to None (canny 100/200 left in the settings).
            ("canny", {"threshold_a": 100.0, "threshold_b": 200.0}),
            # tile_resample's down-sampling rate 1.0 left in threshold_a.
            ("tile_resample", {"threshold_a": 1.0}),
            # The app already sent 'None' (core/sam3_cn_names.lllite_module_override) with old thresholds.
            ("None", {"threshold_a": 100.0, "threshold_b": 200.0}),
        )
        for module, thresholds in cases:
            with self.subTest(module=module):
                run = self._detail_batch("animaTileRepair_v20.safetensors", module,
                                         channels_meta={"lllite.cond_in_channels": "3"}, batch=1,
                                         settings={**base, **thresholds})
                call = run["applied"][0]
                self.assertTrue(torch.allclose(call["image"], run["source"], atol=1e-6))
                control = run["report"]["passes"][0]["controlnet"]
                self.assertEqual((control["module"], control["preprocessor"]), ("None", "none"))
                self.assertEqual((control["threshold_a"], control["threshold_b"]), (-1.0, -1.0))
                self.assertEqual(control["thresholds_ignored"], thresholds)
                self.assertNotIn("lllite_per_image", control)
                self.assertEqual(run["sampled"][0]["latent"].get("batch_index"), None)

    def test_latent_item_slices_batched_tensors_only(self):
        samples = torch.arange(24.0).view(3, 8, 1, 1)
        shared_mask = torch.ones((1, 1, 4, 4))
        item = anima_lllite.latent_item(
            {"samples": samples, "noise_mask": shared_mask, "batch_index": [4, 5, 6], "other": 7}, 1,
        )
        self.assertTrue(torch.equal(item["samples"], samples[1:2]))
        self.assertIs(item["noise_mask"], shared_mask)
        self.assertEqual((item["batch_index"], item["other"]), ([5], 7))


def _tiny_dit(dim=16, blocks=2):
    """Just enough of an Anima DiT for the original module discovery (class name 'Attention' with
    ``is_selfattn`` and q/k/v/output projections; control_net_lllite_anima.py _create_modules)."""
    nn = torch.nn

    class Attention(nn.Module):
        def __init__(self, is_selfattn):
            super().__init__()
            self.is_selfattn = is_selfattn
            self.q_proj = nn.Linear(dim, dim)
            self.k_proj = nn.Linear(dim, dim)
            self.v_proj = nn.Linear(dim, dim)
            self.output_proj = nn.Linear(dim, dim)

    class Block(nn.Module):
        def __init__(self):
            super().__init__()
            self.self_attn = Attention(True)
            self.cross_attn = Attention(False)

    class TinyDiT(nn.Module):
        patch_spatial = 2

        def __init__(self):
            super().__init__()
            self.blocks = nn.ModuleList([Block() for _ in range(blocks)])

    torch.manual_seed(0)
    return TinyDiT().eval().requires_grad_(False)


_LLLITE_META = {
    "lllite.version": "2",
    "lllite.cond_emb_dim": "8",
    "lllite.mlp_dim": "8",
    "lllite.cond_dim": "8",
    "lllite.cond_resblocks": "1",
    "lllite.target_atomics": "self_attn_q_pre",
}


def _time_snr_shift(shift, t):
    # σ = s·t/(1+(s−1)·t) (parity_plan §8.2 '공통', Comfy model_sampling.py time_snr_shift).
    return shift * t / (1 + (shift - 1) * t)


def _sdscripts_get_timesteps_sigmas(sampling_steps, shift, device):
    """kohya-ss/sd-scripts@690ea7f9 library/hunyuan_image_utils.py:276-292, verbatim (Apache-2.0).

    anima_minimal_inference.py:565 samples on these σ; :595 ``step`` = latents − (σᵢ − σᵢ₊₁)·v.
    """
    sigmas = torch.linspace(1, 0, sampling_steps + 1)
    sigmas = (shift * sigmas) / (1 + (shift - 1) * sigmas)
    sigmas = sigmas.to(torch.float32)
    timesteps = (sigmas[:-1] * 1000).to(dtype=torch.float32, device=device)
    return timesteps, sigmas


def _sdscripts_initial_latent(shape, seed):
    """sd-scripts anima_minimal_inference.py:528-529 + :552 — a CPU Generator seeded with ``seed``
    and diffusers ``randn_tensor(shape, generator=seed_g, dtype=torch.bfloat16)``, whose CPU-generator
    branch is ``torch.randn(shape, generator=…, device="cpu", dtype=bfloat16)`` (checked against
    diffusers 0.37.1 in the Forge venv: equal)."""
    seed_g = torch.Generator(device="cpu")
    seed_g.manual_seed(seed)
    return torch.randn(shape, generator=seed_g, device="cpu", dtype=torch.bfloat16)


def _comfy_simple_sigmas(shift, steps):
    """ComfyUI@387f98aa comfy/samplers.py:645-652 simple_scheduler on ModelSamplingDiscreteFlow
    (model_sampling.py:308-312, Anima multiplier 1.0) — the table lookup the pack used to sample."""
    table = _time_snr_shift(shift, torch.arange(1, 1001, 1) / 1000)
    stride = len(table) / steps
    return torch.FloatTensor([float(table[-(1 + int(x * stride))]) for x in range(steps)] + [0.0])


@requires_torch
class TestTileRepairSamplingOrigin(unittest.TestCase):
    """Tile & Repair samples on sd-scripts' σ list from sd-scripts' bf16 CPU-generator noise."""

    @classmethod
    def setUpClass(cls):
        bind_torch(globals())

    def test_sigmas_are_the_sd_scripts_shifted_linspace(self):
        for steps, shift in ((50, 5.0), (30, 5.0), (75, 5.0), (28, 3.0), (1, 5.0), (1500, 5.0)):
            with self.subTest(steps=steps, shift=shift):
                got = anima_lllite.tile_repair_sigmas(steps, shift)
                _, expected = _sdscripts_get_timesteps_sigmas(steps, shift, torch.device("cpu"))
                self.assertEqual(got.dtype, torch.float32)
                self.assertTrue(torch.equal(got, expected))
        # Golden σ printed by the original function (Forge venv anima_vendor, and Comfy's python).
        golden = {(75, 5.0): 0.06329114735126495, (30, 5.0): 0.14705882966518402,
                  (50, 5.0): 0.09259258210659027, (1500, 5.0): 0.003324467921629548}
        for (steps, shift), last_positive in golden.items():
            with self.subTest(golden=(steps, shift)):
                self.assertEqual(float(anima_lllite.tile_repair_sigmas(steps, shift)[-2]), last_positive)

    def test_sigmas_differ_from_comfy_simple_where_1000_is_not_a_multiple(self):
        # The reviewer's CPU case: shift 5, 75 steps → original 0.06329115, Comfy simple 0.06628788.
        simple = _comfy_simple_sigmas(5.0, 75)
        self.assertAlmostEqual(float(simple[-2]), 0.06628788, places=7)
        self.assertNotAlmostEqual(float(anima_lllite.tile_repair_sigmas(75, 5.0)[-2]), float(simple[-2]), places=4)
        # Above 1000 steps Comfy's table repeats σ (zero-length Euler steps); the original never does.
        simple_many = _comfy_simple_sigmas(5.0, 1500)
        self.assertEqual(int((simple_many[1:-1] == simple_many[:-2]).sum()), 500)
        many = anima_lllite.tile_repair_sigmas(1500, 5.0)
        self.assertTrue(bool((many[1:] < many[:-1]).all()))

    def test_initial_noise_is_the_sd_scripts_bf16_cpu_draw(self):
        latent = torch.zeros((1, 16, 1, 32, 48))   # EmptyLatentImage after fix_empty_latent_channels
        for seed in (0, 123, 0xFFFFFFFFFFFFFFFF):
            with self.subTest(seed=seed):
                got = anima_lllite.tile_repair_noise(latent, seed)
                self.assertEqual(got.dtype, torch.bfloat16)
                self.assertEqual(got.device.type, "cpu")
                self.assertTrue(torch.equal(got, _sdscripts_initial_latent((1, 16, 1, 32, 48), seed)))
        # Comfy's prepare_noise draws fp32 from torch.manual_seed(seed) (comfy/sample.py:9-11): not the
        # original's latent — on torch 2.13 it is the bf16 draw before rounding, on torch 2.11 it is
        # a different stream altogether (max |Δ| ≈ 6 at seed 123).
        fp32 = torch.randn((1, 16, 1, 32, 48), generator=torch.manual_seed(123), dtype=torch.float32)
        self.assertFalse(torch.equal(anima_lllite.tile_repair_noise(latent, 123).float(), fp32))
        # The original draws from its own Generator: the global RNG is left alone.
        torch.manual_seed(7)
        before = torch.random.get_rng_state()
        anima_lllite.tile_repair_noise(latent, 123)
        self.assertTrue(torch.equal(torch.random.get_rng_state(), before))


class _FlowSampling:
    """Comfy ModelSamplingDiscreteFlow.percent_to_sigma: p ≤ 0 → 1.0, p ≥ 1 → 0.0, else σ(1 − p)."""

    def __init__(self, shift, multiplier):
        self.shift, self.multiplier = float(shift), float(multiplier)

    def percent_to_sigma(self, percent):
        if percent <= 0.0:
            return 1.0
        if percent >= 1.0:
            return 0.0
        return _time_snr_shift(self.shift, 1.0 - percent)


class _Patcher:
    """ComfyUI ModelPatcher surface the original node and the flow-shift patch use."""

    def __init__(self, dit, sampling):
        self.model = types.SimpleNamespace(diffusion_model=dit)
        self.model_options = {"transformer_options": {}}
        self.object_patches = {"model_sampling": sampling}

    def clone(self):
        copy = _Patcher(self.model.diffusion_model, None)
        copy.model = self.model
        copy.model_options = dict(self.model_options)
        copy.object_patches = dict(self.object_patches)
        return copy

    def get_model_object(self, name):
        return self.object_patches[name]

    def add_object_patch(self, name, value):
        self.object_patches[name] = value

    def set_model_unet_function_wrapper(self, wrapper):
        self.model_options["model_function_wrapper"] = wrapper


@requires_torch
class TestVendoredNodeEndToEnd(_TempControlnetFolder, unittest.TestCase):
    """The pinned kohya node on CPU with real saved-format LLLite weights."""

    @classmethod
    def setUpClass(cls):
        bind_torch(globals())

    def setUp(self):
        super().setUp()
        for name in VENDOR_MODULES:   # the node binds folder_paths at import: import it under this stub
            sys.modules.pop(name, None)
        self.addCleanup(lambda: [sys.modules.pop(name, None) for name in VENDOR_MODULES])
        self.original = importlib.import_module(f"{VENDOR_PACKAGE}.control_net_lllite_anima")
        self.dit = _tiny_dit()

    def _save_weights(self, name, channels, title=None):
        """LLLite weights in the saved v2 layout (the inverse of _from_saved_state_dict)."""
        from safetensors.torch import save_file

        net = self.original.ControlNetLLLiteDiT(
            self.dit, cond_emb_dim=8, mlp_dim=8, target_layers="self_attn_q_pre", cond_dim=8,
            cond_resblocks=1, cond_in_channels=channels,
        )
        generator = torch.Generator().manual_seed(11 + channels)
        with torch.no_grad():
            for parameter in net.parameters():
                parameter.copy_(torch.randn(parameter.shape, generator=generator) * 0.3)
        saved = {}
        for key, value in net.state_dict().items():
            if key.startswith("conditioning1."):
                saved["lllite_conditioning1." + key[len("conditioning1."):]] = value
            elif key.startswith("lllite_modules."):
                index, rest = key[len("lllite_modules."):].split(".", 1)
                saved[f"{net.lllite_modules[int(index)].lllite_name}.{rest}"] = value
            elif key == "depth_embeds":
                for index, module in enumerate(net.lllite_modules):
                    saved[f"{module.lllite_name}.depth_embed"] = value[index]
        metadata = {**_LLLITE_META, "lllite.cond_in_channels": str(channels)}
        if title:
            metadata["modelspec.title"] = title
        path = self.folder / name
        save_file({key: value.contiguous() for key, value in saved.items()}, str(path), metadata=metadata)
        return path

    def _reference_q(self, path, cond_image, tokens, strength):
        """The original class used directly: load, set cond, apply, run block 0's q_proj."""
        net = self.original.ControlNetLLLiteDiT(
            self.dit, cond_emb_dim=8, mlp_dim=8, target_layers="self_attn_q_pre", cond_dim=8,
            cond_resblocks=1, cond_in_channels=cond_image.shape[1],
        )
        self.original.load_lllite_weights(net, str(path), strict=False)
        net.eval().requires_grad_(False)
        net.set_multiplier(strength)
        net.set_cond_image(cond_image)
        net.apply_to()
        try:
            with torch.no_grad():
                return self.dit.blocks[0].self_attn.q_proj(tokens)
        finally:
            net.restore()
            net.clear_cond_image()

    @staticmethod
    def _tokens(latent_h, latent_w, dim=16, seed=5):
        count = (latent_h // 2) * (latent_w // 2)
        return torch.randn((1, count, dim), generator=torch.Generator().manual_seed(seed))

    def _run_wrapper(self, patched, latent_h, latent_w, sigma, tokens):
        seen = []

        def apply_model(input_x, timestep, **c):
            with torch.no_grad():
                seen.append(self.dit.blocks[0].self_attn.q_proj(tokens))
            return input_x

        wrapper = patched.model_options["model_function_wrapper"]
        latent = torch.zeros((1, 16, 1, latent_h, latent_w))
        wrapper(apply_model, {"input": latent, "timestep": torch.tensor([sigma]), "c": {}})
        return seen[0]

    def test_detailer_handle_applies_the_original_patch(self):
        path = self._save_weights("animaTileRepair_v20.safetensors", 3, "anima_tile_multitask_v1")
        control = anima_lllite.lllite_control_for("animaTileRepair_v20.safetensors")
        base = _Patcher(self.dit, _FlowSampling(3.0, 1.0))
        image = torch.rand((1, 64, 96, 3), generator=torch.Generator().manual_seed(2))
        mask = torch.ones((1, 64, 96))
        nodes_logger = f"{VENDOR_PACKAGE}.nodes"
        with self.assertNoLogs(nodes_logger, level="WARNING"):   # 3-channel: the mask is not handed over
            patched, report = control.patch_model(base, image, mask, 0.7, 0.0, 1.0)
        self.assertNotIn("model_function_wrapper", base.model_options)   # clone only
        self.assertEqual(report["lllite_mask"], False)
        tokens = self._tokens(8, 12)
        got = self._run_wrapper(patched, 8, 12, 0.5, tokens)
        cond = image.permute(0, 3, 1, 2).contiguous() * 2.0 - 1.0   # nodes.py:65-81 at latent×8 = 64x96
        expected = self._reference_q(path, cond, tokens, 0.7)
        self.assertTrue(torch.equal(got, expected))
        with torch.no_grad():
            plain = self.dit.blocks[0].self_attn.q_proj(tokens)
        self.assertFalse(torch.allclose(got, plain))           # the LLLite really changed the input

    def test_batch_is_patched_per_image_with_the_original_node(self):
        # origin: kohya-ss/ComfyUI-Anima-LLLite@b7495bd8:nodes.py:75 `img = img[:1]` — one node call
        #   with a batch conditions on image 0 only; sample_per_image gives image i its own patch.
        path = self._save_weights("animaTileRepair_v20.safetensors", 3, "anima_tile_multitask_v1")
        control = anima_lllite.lllite_control_for("animaTileRepair_v20.safetensors")
        base = _Patcher(self.dit, _FlowSampling(3.0, 1.0))
        images = torch.rand((2, 64, 96, 3), generator=torch.Generator().manual_seed(8))
        tokens = self._tokens(8, 12)
        conds = [
            images[index:index + 1].permute(0, 3, 1, 2).contiguous() * 2.0 - 1.0 for index in range(2)
        ]
        expected = [self._reference_q(path, cond, tokens, 0.7) for cond in conds]
        self.assertFalse(torch.allclose(expected[0], expected[1]))
        # The original quirk the detailer must not hit: a batch-2 control image → image 0 for all.
        batched, _ = control.patch_model(base, images, None, 0.7, 0.0, 1.0)
        self.assertTrue(torch.equal(self._run_wrapper(batched, 8, 12, 0.5, tokens), expected[0]))

        latent = {"samples": torch.zeros((2, 16, 1, 8, 12)), "noise_mask": torch.ones((2, 1, 64, 96))}
        seen = []

        def sample(patched, item_latent):
            seen.append((item_latent["batch_index"], self._run_wrapper(patched, 8, 12, 0.5, tokens)))
            return {**item_latent, "samples": item_latent["samples"] + len(seen)}

        output = control.sample_per_image(base, images, None, 0.7, 0.0, 1.0, "Balanced", latent, sample)
        self.assertEqual([index for index, _ in seen], [[0], [1]])
        for index, (_, got) in enumerate(seen):
            self.assertTrue(torch.equal(got, expected[index]))
        self.assertEqual(output["samples"][:, 0, 0, 0, 0].tolist(), [1.0, 2.0])
        self.assertNotIn("batch_index", output)
        self.assertNotIn("model_function_wrapper", base.model_options)

    def test_inpaint_weights_receive_the_mask_channel(self):
        path = self._save_weights("anima-lllite-inpainting-v2.safetensors", 4)
        control = anima_lllite.lllite_control_for(str(path))   # absolute path: registry swap
        self.assertEqual(control.channels, 4)
        base = _Patcher(self.dit, _FlowSampling(3.0, 1.0))
        image = torch.rand((1, 64, 96, 3), generator=torch.Generator().manual_seed(4))
        mask = torch.zeros((1, 64, 96))
        mask[:, 16:48, 24:72] = 1.0
        before = self.folder_paths.folder_names_and_paths["controlnet"]
        patched, report = control.patch_model(base, image, mask, 1.0, 0.0, 1.0)
        self.assertIs(self.folder_paths.folder_names_and_paths["controlnet"], before)
        self.assertEqual(report["lllite_mask"], True)
        tokens = self._tokens(8, 12)
        got = self._run_wrapper(patched, 8, 12, 0.5, tokens)
        rgb = image.permute(0, 3, 1, 2) * 2.0 - 1.0
        cond = self.original_nodes()._build_inpaint_cond_image(rgb, mask.unsqueeze(1), False)
        self.assertTrue(torch.equal(got, self._reference_q(path, cond, tokens, 1.0)))

    def original_nodes(self):
        return importlib.import_module(f"{VENDOR_PACKAGE}.nodes")

    # ── ForgeNeoAnimaTileRepair ──────────────────────────────────────────────────────
    def _comfy_nodes(self, log):
        test = self

        class CLIPTextEncode:
            def encode(self, clip, text):
                log.append(("encode", text))
                return ([[f"cond:{text}", {}]],)

        class EmptyLatentImage:
            def generate(self, width, height, batch_size=1):
                log.append(("empty_latent", width, height, batch_size))
                return ({"samples": torch.zeros((batch_size, 4, height // 8, width // 8)),
                         "downscale_ratio_spacial": 8},)

        class ModelSamplingSD3:
            def patch(self, model, shift, multiplier=1000, sampling="flow"):
                log.append(("flow_shift", shift, multiplier))
                patched = model.clone()
                patched.add_object_patch("model_sampling", _FlowSampling(shift, multiplier))
                return (patched,)

        class VAEDecode:
            def decode(self, vae, samples):
                log.append(("decode", tuple(samples["samples"].shape)))
                _, _, _, h, w = samples["samples"].shape
                return (torch.full((1, h * 8, w * 8, 3), 0.5),)

        module = types.ModuleType("nodes")
        module.NODE_CLASS_MAPPINGS = {
            "CLIPTextEncode": CLIPTextEncode, "EmptyLatentImage": EmptyLatentImage,
            "ModelSamplingSD3": ModelSamplingSD3, "VAEDecode": VAEDecode,
        }
        # No common_ksampler: its prepare_noise (fp32) and ``simple`` σ table are not sd-scripts'.
        return module

    def _comfy_sampling(self, log):
        """comfy.sample / comfy.samplers / comfy.utils / latent_preview as SamplerCustom uses them."""
        test = self
        comfy = types.ModuleType("comfy")
        sample = types.ModuleType("comfy.sample")
        samplers = types.ModuleType("comfy.samplers")
        utils = types.ModuleType("comfy.utils")
        preview = types.ModuleType("latent_preview")

        def fix_empty_latent_channels(model, latent_image, spacial=None, temporal=None):
            # Wan21 latent format (Anima): 16 channels, 3 latent dimensions (comfy/sample.py:45-59).
            log.append(("fix_channels", tuple(latent_image.shape), spacial, temporal))
            b, _, h, w = latent_image.shape
            return torch.zeros((b, 16, 1, h, w))

        def sampler_object(name):
            return f"sampler:{name}"

        def sample_custom(model, noise, cfg, sampler, sigmas, positive, negative, latent_image,
                          noise_mask=None, callback=None, disable_pbar=False, seed=None):
            log.append(("sample_custom", sampler, cfg, positive[0][0], negative[0][0],
                        tuple(latent_image.shape), noise_mask, callback, disable_pbar, seed))
            test.sampled = {"noise": noise, "sigmas": sigmas}
            h, w = latent_image.shape[-2:]
            tokens = test._tokens(h, w)
            test.window = {
                sigma: test._run_wrapper(model, h, w, sigma, tokens) for sigma in (0.99, 0.5)
            }
            test.window_tokens = tokens
            return torch.zeros_like(latent_image)

        sample.fix_empty_latent_channels = fix_empty_latent_channels
        sample.sample_custom = sample_custom
        samplers.sampler_object = sampler_object
        utils.PROGRESS_BAR_ENABLED = False
        preview.prepare_callback = lambda model, steps: log.append(("preview", steps)) or "callback"
        comfy.sample, comfy.samplers, comfy.utils = sample, samplers, utils
        return {
            "comfy": comfy, "comfy.sample": sample, "comfy.samplers": samplers,
            "comfy.utils": utils, "latent_preview": preview,
        }

    def test_tile_repair_graph_follows_sd_scripts(self):
        path = self._save_weights("animaTileRepair_v20.safetensors", 3, "anima_tile_multitask_v1")
        base = _Patcher(self.dit, _FlowSampling(3.0, 1.0))
        source = torch.rand((1, 64, 96, 3), generator=torch.Generator().manual_seed(9))
        log = []
        with mock.patch.dict(sys.modules, {"nodes": self._comfy_nodes(log), **self._comfy_sampling(log)}):
            (image,) = anima_lllite.ForgeNeoAnimaTileRepair().repair(
                base, "CLIP", "VAE", source, "animaTileRepair_v20.safetensors",
                positive=CARD_PROMPT, negative="", seed=123, steps=50, cfg=3.5, flow_shift=5.0,
                short_side=256, strength=0.9, start_percent=0.2, end_percent=1.0,
            )
        width, height = anima_lllite.tile_repair_size(96, 64, 256)
        self.assertEqual((width, height), (384, 256))
        self.assertEqual(log, [
            ("encode", CARD_PROMPT),
            ("encode", ""),
            ("flow_shift", 5.0, 1.0),                          # Anima's multiplier 1.0 is kept
            ("empty_latent", 384, 256, 1),                     # source aspect, multiple of 32
            ("fix_channels", (1, 4, 32, 48), 8, None),
            ("preview", 50),
            ("sample_custom", "sampler:euler", 3.5, f"cond:{CARD_PROMPT}", "cond:",
             (1, 16, 1, 32, 48), None, "callback", True, 123),  # pure noise start
            ("decode", (1, 16, 1, 32, 48)),
        ])
        # sd-scripts generate_body: its own bf16 CPU-generator latent and its shifted-linspace σ.
        noise = self.sampled["noise"]
        self.assertEqual(noise.dtype, torch.bfloat16)
        self.assertTrue(torch.equal(noise, _sdscripts_initial_latent((1, 16, 1, 32, 48), 123)))
        _, sigmas = _sdscripts_get_timesteps_sigmas(50, 5.0, torch.device("cpu"))
        self.assertTrue(torch.equal(self.sampled["sigmas"], sigmas))
        self.assertEqual(tuple(image.shape), (1, 256, 384, 3))
        self.assertNotIn("model_function_wrapper", base.model_options)
        # The σ window is read from the shift-5 schedule that is sampled: start 0.2 → σ 0.952.
        tokens = self.window_tokens
        with torch.no_grad():
            plain = self.dit.blocks[0].self_attn.q_proj(tokens)
        self.assertGreater(0.99, _time_snr_shift(5.0, 0.8))
        self.assertTrue(torch.equal(self.window[0.99], plain))  # before the window: original Linear
        control = anima_lllite.sdscripts_control_image(source, width, height)
        expected = self._reference_q(path, control.permute(0, 3, 1, 2).contiguous() * 2.0 - 1.0, tokens, 0.9)
        self.assertTrue(torch.equal(self.window[0.5], expected))

    def test_tile_repair_uses_the_users_steps_shift_cfg_and_seed(self):
        # The graph case above passes the defaults (50 steps, shift 5.0, cfg 3.5): a node that
        # ignored the user's values would still pass it. Non-default values must reach the shift
        # patch, the σ list, the preview callback, the sampler's CFG and the noise.
        self._save_weights("animaTileRepair_v20.safetensors", 3, "anima_tile_multitask_v1")
        base = _Patcher(self.dit, _FlowSampling(3.0, 1.0))
        source = torch.rand((1, 64, 96, 3), generator=torch.Generator().manual_seed(9))
        steps, flow_shift, cfg, seed = 30, 3.0, 4.5, 7
        self.assertNotEqual(
            (steps, flow_shift, cfg),
            (anima_lllite.TILE_REPAIR_STEPS, anima_lllite.TILE_REPAIR_FLOW_SHIFT,
             anima_lllite.TILE_REPAIR_CFG),
        )
        log = []
        with mock.patch.dict(sys.modules, {"nodes": self._comfy_nodes(log), **self._comfy_sampling(log)}):
            anima_lllite.ForgeNeoAnimaTileRepair().repair(
                base, "CLIP", "VAE", source, "animaTileRepair_v20.safetensors",
                positive=CARD_PROMPT, negative="", seed=seed, steps=steps, cfg=cfg,
                flow_shift=flow_shift, short_side=256,
            )
        self.assertIn(("flow_shift", flow_shift, 1.0), log)
        self.assertIn(("preview", steps), log)
        sampled = [entry for entry in log if entry[0] == "sample_custom"]
        self.assertEqual(len(sampled), 1)
        self.assertEqual(sampled[0][2], cfg)
        self.assertEqual(sampled[0][-1], seed)
        _, sigmas = _sdscripts_get_timesteps_sigmas(steps, flow_shift, torch.device("cpu"))
        self.assertTrue(torch.equal(self.sampled["sigmas"], sigmas))
        self.assertTrue(torch.equal(self.sampled["noise"], _sdscripts_initial_latent((1, 16, 1, 32, 48), seed)))

    def test_control_image_is_the_sd_scripts_pil_bicubic_resize(self):
        # origin: anima_minimal_inference_control_net_lllite.py:65-74
        import numpy as np
        from PIL import Image

        pixels = np.random.default_rng(1).integers(0, 256, size=(37, 53, 3), dtype=np.uint8)
        comfy_image = torch.from_numpy(pixels.astype(np.float32) / 255.0)[None]   # LoadImage
        ours = anima_lllite.sdscripts_control_image(comfy_image, 96, 64) * 2.0 - 1.0
        resized = Image.fromarray(pixels).convert("RGB").resize((96, 64), Image.BICUBIC)
        script = torch.from_numpy(np.asarray(resized).astype(np.float32) / 127.5 - 1.0)[None]
        self.assertEqual(tuple(ours.shape), (1, 64, 96, 3))
        self.assertLessEqual(float((ours - script).abs().max()), 1e-6)

    def test_tile_repair_refuses_what_the_original_cannot_run(self):
        self._save_weights("anima-lllite-inpainting-v2.safetensors", 4)
        _write_header(self.folder / "cn.safetensors", _CONTROLNET_KEYS)
        base = _Patcher(self.dit, _FlowSampling(3.0, 1.0))
        source = torch.rand((1, 64, 96, 3))
        node = anima_lllite.ForgeNeoAnimaTileRepair()
        for name, message in (
            ("None", "Pick an Anima"),
            ("cn.safetensors", "not an Anima ControlNet-LLLite"),
            ("anima-lllite-inpainting-v2.safetensors", "cond_in_channels=4"),
        ):
            with self.subTest(name=name), self.assertRaisesRegex(ValueError, message):
                node.repair(base, "CLIP", "VAE", source, name)
        self._save_weights("animaTileRepair_v20.safetensors", 3)
        with self.assertRaisesRegex(ValueError, "strength"):
            node.repair(base, "CLIP", "VAE", source, "animaTileRepair_v20.safetensors", strength=11.0)


if __name__ == "__main__":
    unittest.main()
