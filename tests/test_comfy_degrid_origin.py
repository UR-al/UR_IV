r"""ComfyUI pack ``ForgeNeoAnimaVAEDeGrid`` = the sam-extra "Anima VAE DeGrid (NAFNet)" script.

origin: forge_sam3_extension@395854b — sam3ext/vae_degrid.py, vae_degrid_runtime.py,
vae_degrid_models.py, ui_vae_degrid.py (GPL-3.0). The pack holds none of that code
(``degrid_math``/``degrid_files``/``degrid_runner``/``degrid_nodes`` are local
implementations). ``tests/fixtures/degrid_origin_golden.json`` stores only numbers and
checksums produced by running the extension's own functions on CPU (Forge venv,
CUDA_VISIBLE_DEVICES=-1) on the inputs of ``tests/_degrid_recipes.py``; the generator is
named in the fixture's ``generator`` field.

Tiers
- no torch (runs in --quick): node inputs/outputs contract, coercion tables, tile
  positions, model-file inspection (safetensors header / zip data.pkl without torch.load),
  folder registration, model naming/ordering/resolution, the unload hook, and that the
  pack's DeGrid modules import no torch/comfy/spandrel at module level.
- ``requires_torch`` (app venv, CPU): residual arithmetic × modes × strengths, Forge 8-bit
  quantisation in/out, reflect padding, tiling, the residual guard, and the node's whole
  per-image pipeline with fake models (8-bit output, tile actually used after OOM halving,
  report = Forge infotext, skip categories). Every input is float32-exact and the fake
  models use element-wise ops and max/min only, so checksums match on any torch build.
- real ComfyUI (skip unless ``AISTUDIO_COMFY_TEST_ROOT``; run with the portable
  ``python_embeded`` and ``CUDA_VISIBLE_DEVICES=-1`` — ComfyUI is forced to ``--cpu``
  before its model management loads): the pack tiler equals ``comfy.utils.tiled_scale``
  bit for bit, tiny spandrel NAFNets built from LCG weights hit the same skip categories
  with the same raw residual, and the real ``qwenVAEDegridNafnet_v11`` (when present with
  the recorded SHA-256; ``AISTUDIO_DEGRID_MODEL`` overrides the path) gives the same saved
  8-bit bytes as the extension. No GPU, no ComfyUI server. The portable python's ``._pth``
  puts ComfyUI (with its own ``tests`` package) on sys.path, so put the repo first::

    $env:CUDA_VISIBLE_DEVICES='-1'; $env:AISTUDIO_COMFY_TEST_ROOT='C:\ComfyUI_windows_portable\ComfyUI'
    C:\ComfyUI_windows_portable\python_embeded\python.exe -c "import sys, unittest; sys.path.insert(0, r'<repo>'); unittest.main(module=None, argv=['degrid', 'tests.test_comfy_degrid_origin', '-v'])"
"""
from __future__ import annotations

import ast
import contextlib
import json
import logging
import os
import pickle
import re
import struct
import sys
import tempfile
import unittest
import zipfile
from collections import OrderedDict
from pathlib import Path
from types import SimpleNamespace

_COMFY_ROOT = os.environ.get("AISTUDIO_COMFY_TEST_ROOT")
if _COMFY_ROOT:  # real-ComfyUI tier only: force --cpu before anything loads model management
    if _COMFY_ROOT not in sys.path:
        sys.path.append(_COMFY_ROOT)
    from comfy.cli_args import args as _comfy_args  # noqa: E402

    _comfy_args.cpu = True

from comfy_custom_nodes.ai_studio_forge_parity import (  # noqa: E402
    degrid_files,
    degrid_math as dm,
    degrid_nodes,
    degrid_runner,
)
from tests import _degrid_recipes as R  # noqa: E402
from tests._optional_deps import bind_torch, requires_torch  # noqa: E402

torch = None  # filled by bind_torch() in the marked classes

ROOT = Path(__file__).resolve().parent.parent
PACK = ROOT / "comfy_custom_nodes" / "ai_studio_forge_parity"
FIXTURE = ROOT / "tests" / "fixtures" / "degrid_origin_golden.json"
GOLDEN = json.loads(FIXTURE.read_text(encoding="utf-8"))
DEFAULT_REAL_MODEL = Path(r"C:\sd-webui-forge-classic\models\ESRGAN\qwenVAEDegridNafnet_v11.safetensors")

# Frozen cross-package contract (plan v2 §3.1 COMFY_INPUTS) — the app compiler builds these.
COMFY_INPUTS = (
    "image", "enabled", "model_name", "mode", "strength", "tile",
    "device", "precision", "keep_loaded", "forge_quantize",
)
KEY_MODEL = "Anima DeGrid model"
KEY_MODE = "Anima DeGrid mode"
KEY_STRENGTH = "Anima DeGrid strength"
KEY_TILE = "Anima DeGrid tile"
KEY_PRECISION = "Anima DeGrid precision"


def _error_facts(text: str) -> dict:
    """Category (words before ':' or ' (') and the quoted mean, like the golden stores them."""
    if not text:
        return {"error_category": "", "error_mean_255": None}
    match = re.search(r"mean \|(?:output|residual)\| ([0-9.]+)/255", text)
    return {
        "error_category": re.split(r":| \(", text, maxsplit=1)[0].strip(),
        "error_mean_255": float(match.group(1)) if match else None,
    }


def _write_safetensors_header(path: Path, keys, metadata=None) -> None:
    """A header-only safetensors-shaped file (1-element f32 tensors, zero data)."""
    header = {}
    offset = 0
    for key in keys:
        header[key] = {"dtype": "F32", "shape": [1], "data_offsets": [offset, offset + 4]}
        offset += 4
    if metadata:
        header["__metadata__"] = metadata
    raw = json.dumps(header).encode("utf-8")
    path.write_bytes(struct.pack("<Q", len(raw)) + raw + b"\0" * offset)


class _SideEffect:
    """Pickled into a zip checkpoint: unpickling it with a normal Unpickler would call ``_touch``."""

    calls = []

    def __reduce__(self):
        return (_touch, ("unpickled",))


def _touch(tag):
    _SideEffect.calls.append(tag)
    return tag


def _write_zip_checkpoint(path: Path, state) -> None:
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("archive/data.pkl", pickle.dumps(state, protocol=2))
        archive.writestr("archive/version", "3\n")


class _FakeFolderPaths:
    """The bits of ComfyUI ``folder_paths`` the node uses, over real temp folders."""

    def __init__(self, roots: dict):
        self.roots = {category: [Path(p) for p in paths] for category, paths in roots.items()}

    def get_filename_list(self, category):
        names = []
        for root in self.roots.get(category, ()):
            for path in sorted(root.rglob("*")):
                if path.is_file():
                    rel = path.relative_to(root).as_posix()
                    if rel not in names:
                        names.append(rel)
        return names

    def get_full_path(self, category, filename):
        for root in self.roots.get(category, ()):
            candidate = root / filename
            if candidate.is_file():
                return str(candidate)
        return None


# ─────────────────────────────────────────────────────────────────────────
# no torch
# ─────────────────────────────────────────────────────────────────────────
class TestDegridNodeContract(unittest.TestCase):
    def test_inputs_follow_the_frozen_compiler_contract(self):
        spec = degrid_nodes.ForgeNeoAnimaVAEDeGrid.INPUT_TYPES()
        required = spec["required"]
        self.assertEqual(tuple(required), COMFY_INPUTS)
        self.assertEqual(degrid_nodes.INPUT_NAMES, COMFY_INPUTS)
        self.assertEqual(set(spec), {"required"})
        self.assertEqual(required["image"], ("IMAGE",))
        self.assertEqual(required["enabled"], ("BOOLEAN", {"default": True}))
        self.assertEqual(required["model_name"][0][0], "auto")  # outside ComfyUI: only auto
        self.assertEqual(
            required["mode"],
            (["Full", "Dark Pixels Mainly", "Bright Pixels Mainly"], {"default": "Full"}),
        )
        self.assertEqual(
            required["strength"],
            ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.5, "step": 0.05}),
        )
        self.assertEqual(required["tile"], ("INT", {"default": 512, "min": 0, "max": 4096, "step": 1}))
        self.assertEqual(required["device"], (["auto", "cpu"], {"default": "auto"}))
        self.assertEqual(required["precision"], (["fp32", "fp16"], {"default": "fp32"}))
        self.assertEqual(required["keep_loaded"], ("BOOLEAN", {"default": False}))
        self.assertEqual(required["forge_quantize"], ("BOOLEAN", {"default": True}))

    def test_outputs_mapping_and_ui_key(self):
        node = degrid_nodes.ForgeNeoAnimaVAEDeGrid
        self.assertEqual(node.RETURN_TYPES, ("IMAGE", "STRING"))
        self.assertEqual(node.RETURN_NAMES, ("image", "report_json"))
        self.assertTrue(callable(getattr(node(), node.FUNCTION)))
        self.assertEqual(degrid_nodes.NODE_CLASS_MAPPINGS, {"ForgeNeoAnimaVAEDeGrid": node})
        self.assertEqual(degrid_nodes.UI_KEY, "ai_studio_degrid")
        from comfy_custom_nodes.ai_studio_forge_parity import (
            NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS,
        )
        self.assertIs(NODE_CLASS_MAPPINGS["ForgeNeoAnimaVAEDeGrid"], node)
        self.assertEqual(
            NODE_DISPLAY_NAME_MAPPINGS["ForgeNeoAnimaVAEDeGrid"], "Forge Neo Anima VAE DeGrid (NAFNet)",
        )

    def test_modules_import_no_heavy_runtime_at_module_level(self):
        banned = {"torch", "comfy", "spandrel", "folder_paths", "nodes", "safetensors", "numpy"}
        for name in ("degrid_math.py", "degrid_files.py", "degrid_runner.py", "degrid_nodes.py"):
            tree = ast.parse((PACK / name).read_text(encoding="utf-8"))
            for node in tree.body:
                roots = []
                if isinstance(node, ast.Import):
                    roots = [alias.name.split(".")[0] for alias in node.names]
                elif isinstance(node, ast.ImportFrom) and node.level == 0:
                    roots = [(node.module or "").split(".")[0]]
                with self.subTest(module=name, line=getattr(node, "lineno", 0)):
                    self.assertFalse(banned.intersection(roots), roots)

    def test_mode_labels_and_ranges_match_the_extension(self):
        self.assertEqual(dm.MODES, ("full", "dark", "bright"))
        self.assertEqual(
            dm.MODE_LABELS, {"full": "Full", "dark": "Dark Pixels Mainly", "bright": "Bright Pixels Mainly"},
        )
        self.assertEqual((dm.STRENGTH_MIN, dm.STRENGTH_MAX, dm.DEFAULT_STRENGTH), (0.0, 1.5, 1.0))
        self.assertEqual((dm.MIN_TILE, dm.MAX_TILE, dm.DEFAULT_TILE, dm.TILE_OVERLAP), (128, 4096, 512, 32))
        self.assertEqual(dm.PAD_MULTIPLE, 16)
        self.assertEqual(len(dm.NAFNET_KEYS), 18)
        self.assertEqual(
            (dm.ERROR_MODEL_NOT_FOUND, dm.ERROR_NOT_RESIDUAL, dm.ERROR_BLEW_UP),
            ("model not found", "not a DeGrid residual model", "output blew up"),
        )


class TestDegridCoercionGolden(unittest.TestCase):
    def test_strength_tile_and_mode_tables(self):
        table = GOLDEN["coerce"]
        inputs = {
            "strength": [-1, 0, 0.3, 1.5, 2, float("nan"), "x", None, "0.75", True, "1e-3"],
            "tile": [0, -3, 1, 127, 128, 300, 4096, 5000, "x", "256.9", None, "64", 1e12],
            "mode": ["full", "Full", "FULL", "dark", "Dark Pixels Mainly", "dark_pixels", "dark pixels",
                     "Dark Pixels Mainly (어두운 점 위주)", "bright", "Bright Pixels Mainly (밝은 점 위주)",
                     "bright_pixels", "", None, "unknown", "  Bright  ", "Full (전체)", "bright pixels"],
        }
        functions = {"strength": dm.coerce_strength, "tile": dm.coerce_tile, "mode": dm.normalize_mode}
        for kind, values in inputs.items():
            self.assertEqual(len(values), len(table[kind]))
            for value, (shown, expected) in zip(values, table[kind]):
                with self.subTest(kind=kind, value=shown):
                    self.assertEqual(functions[kind](value), expected)
        self.assertEqual(dm.normalize_mode("nonsense") or dm.DEFAULT_MODE, table["args_mode_fallback"])

    def test_tile_positions_equal_the_origin_tile_starts(self):
        for size, tile, overlap, starts in GOLDEN["tile_starts"]:
            with self.subTest(size=size, tile=tile):
                self.assertEqual(dm.tile_positions(size, tile, overlap), starts)


class TestDegridModelFiles(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)

    def tearDown(self):
        self._tmp.cleanup()

    def test_safetensors_header_decides_and_reads_the_version(self):
        good = self.tmp / "degrid.safetensors"
        _write_safetensors_header(good, dm.NAFNET_KEYS, {"modelspec.version": "1.1"})
        partial = self.tmp / "esrgan.safetensors"
        _write_safetensors_header(partial, dm.NAFNET_KEYS[:-1] + ("conv_first.weight",))
        prefixed = self.tmp / "prefixed.safetensors"
        _write_safetensors_header(prefixed, ["module." + key for key in dm.NAFNET_KEYS])
        garbage = self.tmp / "garbage.safetensors"
        garbage.write_bytes(b"\xff" * 64)
        self.assertEqual(degrid_files.classify(good), (True, (1, 1)))
        self.assertEqual(degrid_files.classify(partial), (False, ()))
        self.assertEqual(degrid_files.classify(prefixed), (True, ()))
        self.assertEqual(degrid_files.classify(garbage), (False, ()))
        self.assertEqual(degrid_files.version_tuple("v1.2.3.4.5"), (1, 2, 3, 4))

    def test_zip_checkpoint_keys_are_read_without_running_pickled_code(self):
        _SideEffect.calls.clear()
        state = OrderedDict((key, _SideEffect()) for key in dm.NAFNET_KEYS)
        wrapped = self.tmp / "wrapped.pth"
        _write_zip_checkpoint(wrapped, {"params": OrderedDict(("net_g." + k, v) for k, v in state.items())})
        other = self.tmp / "other.pt"
        _write_zip_checkpoint(other, OrderedDict((k, _SideEffect()) for k in ("conv.weight", "conv.bias")))
        legacy = self.tmp / "legacy.pth"
        legacy.write_bytes(pickle.dumps(state, protocol=2))  # not a zip: never opened
        self.assertEqual(degrid_files.classify(wrapped), (True, ()))
        self.assertEqual(degrid_files.classify(other), (False, ()))
        self.assertEqual(degrid_files.classify(legacy), (False, ()))
        self.assertEqual(_SideEffect.calls, [])
        # a normal unpickler would have run it — the file really carries the call
        with zipfile.ZipFile(other) as archive:
            pickle.loads(archive.read("archive/data.pkl"))
        self.assertEqual(_SideEffect.calls, ["unpickled", "unpickled"])

    def _model_dirs(self):
        upscale = self.tmp / "upscale_models"
        degrid = self.tmp / "degrid"
        (upscale / "sub").mkdir(parents=True)
        degrid.mkdir()
        _write_safetensors_header(upscale / "qwenVAEDegridNafnet_v11.safetensors", dm.NAFNET_KEYS,
                                  {"modelspec.version": "1.1"})
        _write_safetensors_header(upscale / "NAFNet-QwenVAE-DeGrid.safetensors", dm.NAFNET_KEYS)
        _write_safetensors_header(upscale / "4x-UltraSharp.safetensors", ("conv_first.weight",))
        _write_safetensors_header(upscale / "sub" / "nested.safetensors", dm.NAFNET_KEYS,
                                  {"modelspec.version": "9"})
        (upscale / "notes.txt").write_text("x", encoding="utf-8")
        _write_safetensors_header(degrid / "anzhc.safetensors", dm.NAFNET_KEYS, {"modelspec.version": "1.0"})
        _write_safetensors_header(degrid / "NAFNet-QwenVAE-DeGrid.safetensors", dm.NAFNET_KEYS)
        _write_zip_checkpoint(degrid / "anzhc.pth", OrderedDict((k, None) for k in dm.NAFNET_KEYS))
        return _FakeFolderPaths({"upscale_models": [upscale], "degrid": [degrid]})

    def test_discovery_orders_names_and_resolves_like_forge(self):
        files = degrid_nodes.discover_models(self._model_dirs())
        self.assertEqual(
            [(f.category, f.filename, f.version) for f in files],
            [
                ("upscale_models", "qwenVAEDegridNafnet_v11.safetensors", (1, 1)),
                ("degrid", "anzhc.safetensors", (1, 0)),
                ("upscale_models", "NAFNet-QwenVAE-DeGrid.safetensors", ()),
                ("degrid", "anzhc.pth", ()),
                ("degrid", "NAFNet-QwenVAE-DeGrid.safetensors", ()),
            ],
        )
        self.assertEqual(
            [f.choice for f in files],
            [
                "qwenVAEDegridNafnet_v11.safetensors",
                "anzhc.safetensors",
                "upscale_models/NAFNet-QwenVAE-DeGrid.safetensors",
                "anzhc.pth",
                "degrid/NAFNet-QwenVAE-DeGrid.safetensors",
            ],
        )
        self.assertEqual(
            [f.forge_name for f in files],
            ["qwenVAEDegridNafnet_v11", "DeGrid/anzhc", "ESRGAN/NAFNet-QwenVAE-DeGrid",
             "DeGrid/anzhc", "DeGrid/NAFNet-QwenVAE-DeGrid"],
        )
        resolve = degrid_nodes.resolve_model
        self.assertIs(resolve("auto", files), files[0])
        self.assertIs(resolve("", files), files[0])
        self.assertIs(resolve("None", files), files[0])
        self.assertIs(resolve("anzhc.pth", files), files[3])
        self.assertIs(resolve("QWENVAEDEGRIDNAFNET_V11", files), files[0])
        self.assertIs(resolve("ESRGAN/NAFNet-QwenVAE-DeGrid", files), files[2])
        self.assertIs(resolve("degrid/NAFNet-QwenVAE-DeGrid.safetensors", files), files[4])
        self.assertIsNone(resolve("missing", files))
        self.assertIsNone(resolve("auto", []))
        self.assertEqual(
            degrid_nodes.model_choices(self._fake_again()), ["auto", *[f.choice for f in files]],
        )

    def _fake_again(self):
        return _FakeFolderPaths({"upscale_models": [self.tmp / "upscale_models"], "degrid": [self.tmp / "degrid"]})

    def test_folder_registration_creates_or_merges_without_replacing(self):
        created = SimpleNamespace(folder_names_and_paths={}, models_dir="M:/models",
                                  supported_pt_extensions={".safetensors", ".pth"},
                                  filename_list_cache={"degrid": "stale", "loras": "keep"})
        self.assertTrue(degrid_nodes.register_degrid_folder(created))
        self.assertEqual(
            created.folder_names_and_paths["degrid"],
            ([os.path.join("M:/models", "degrid")], {".safetensors", ".pth"}),
        )
        self.assertEqual(created.filename_list_cache, {"loras": "keep"})
        self.assertFalse(degrid_nodes.register_degrid_folder(created))  # idempotent

        existing_exts = {".safetensors"}
        merged = SimpleNamespace(folder_names_and_paths={"degrid": (["D:/yaml/degrid"], existing_exts)},
                                 models_dir="M:/models", supported_pt_extensions={".pt"})
        self.assertTrue(degrid_nodes.register_degrid_folder(merged))
        paths, exts = merged.folder_names_and_paths["degrid"]
        self.assertEqual(paths, ["D:/yaml/degrid", os.path.join("M:/models", "degrid")])
        self.assertIs(exts, existing_exts)
        self.assertFalse(degrid_nodes.register_degrid_folder(SimpleNamespace()))  # not ComfyUI

    def test_unload_hook_chains_and_drops_the_cache(self):
        calls = []
        management = SimpleNamespace(unload_all_models=lambda: calls.append("comfy"))
        self.assertTrue(degrid_nodes.install_unload_release_hook(management))
        self.assertFalse(degrid_nodes.install_unload_release_hook(management))  # idempotent
        degrid_nodes._CACHE.update(key=("x", 1, 2), model=None, patcher=None)
        management.unload_all_models()
        self.assertEqual(calls, ["comfy"])
        self.assertEqual(degrid_nodes._CACHE, {"key": None, "model": None, "patcher": None})


# ─────────────────────────────────────────────────────────────────────────
# torch (CPU)
# ─────────────────────────────────────────────────────────────────────────
@requires_torch
class TestDegridMathGolden(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        bind_torch(globals())

    def test_residual_arithmetic_modes_strengths_and_final_clamp(self):
        spec = GOLDEN["arithmetic"]
        h, w, seed = spec["image"]
        x = R.levels_to_nchw(R.grain_levels(h, w, seed), h, w, torch)
        deltas = {name: R.unit_tensor((1, 3, h, w), s, torch, scale=scale)
                  for name, (s, scale) in spec["delta_seed"].items()}
        self.assertEqual(len(spec["cases"]), 24)
        for case in spec["cases"]:
            with self.subTest(delta=case["delta"], mode=case["mode"], strength=case["strength"]):
                raw = dm.apply_residual(x, deltas[case["delta"]], case["mode"], case["strength"])
                final = dm.finalize(raw)
                self.assertEqual(R.sha_float(raw, torch), case["raw_sha"])
                self.assertEqual(R.sha_float(final, torch), case["final_sha"])
                self.assertEqual(R.sha_bytes(R.nchw_to_hwc_bytes(dm.round_out(final), torch)), case["u8_sha"])

    def test_forge_quantisation_in_and_rounding_out(self):
        spec = GOLDEN["quantize"]
        h, w = GOLDEN["arithmetic"]["image"][:2]
        seed, scale, offset = spec["input"]
        q_in = R.unit_tensor((1, 3, h, w), seed, torch, scale=scale, offset=offset)
        self.assertEqual(R.sha_float(dm.forge_quantize_in(q_in), torch), spec["forge_input_sha"])
        seed, scale, offset = spec["output"]
        q_out = R.unit_tensor((1, 3, h, w), seed, torch, scale=scale, offset=offset)
        self.assertEqual(R.sha_bytes(R.nchw_to_hwc_bytes(dm.round_out(q_out), torch)), spec["rounded_u8_sha"])

    def test_every_8bit_level_survives_comfy_truncating_savers(self):
        levels = torch.arange(256, dtype=torch.float32).reshape(1, 1, 16, 16)
        rounded = dm.round_out(levels / 255.0)
        saved = R.comfy_save_bytes(rounded.reshape(1, 16, 16, 1), torch)
        self.assertEqual(list(saved), list(range(256)))
        self.assertTrue(torch.equal(dm.forge_quantize_in(rounded), rounded))

    def test_reflect_padding_and_padded_calls(self):
        for case in GOLDEN["padding"]:
            h, w = case["shape"]
            with self.subTest(shape=case["shape"], multiple=case["multiple"]):
                t = R.dyadic_nchw(h, w, case["seed"], torch)
                padded = dm.pad_to_multiple(t, case["multiple"])
                self.assertEqual(list(padded.shape[-2:]), case["padded_shape"])
                self.assertEqual(R.sha_float(padded, torch), case["padded_sha"])
                called = dm.call_padded(lambda p: R.fake_forward("residual", p, torch), t, case["multiple"])
                self.assertEqual(R.sha_float(called, torch), case["called_sha"])
        aligned = R.dyadic_nchw(32, 48, 5, torch)
        self.assertIs(dm.pad_to_multiple(aligned, 16), aligned)

    def test_tiled_residual_matches_the_origin_blend(self):
        for case in GOLDEN["tiling"]:
            h, w = case["shape"]
            with self.subTest(shape=case["shape"], tile=case["tile"]):
                t = R.dyadic_nchw(h, w, case["seed"], torch)
                pieces = []
                res = dm.tiled_residual(
                    t, lambda p: dm.call_padded(lambda q: R.fake_forward("residual", q, torch), p, 16),
                    tile=case["tile"], overlap=dm.TILE_OVERLAP, on_piece=lambda: pieces.append(1),
                )
                if R.sha_float(res, torch) != case["sha"]:  # tolerance fallback (other torch build)
                    self.assertEqual(len(R.strided_sample(res, 997)), len(case["sample"]))
                    for got, want in zip(R.strided_sample(res, 997), case["sample"]):
                        self.assertAlmostEqual(got, want, delta=1e-6)
                    self.assertAlmostEqual(float(res.double().sum()), case["sum"], delta=1e-3)
                tiles = 1 if case["tile"] <= 0 or (h <= case["tile"] and w <= case["tile"]) else (
                    len(dm.tile_positions(h, case["tile"])) * len(dm.tile_positions(w, case["tile"])))
                self.assertEqual(len(pieces), tiles)

    def test_residual_guard_decisions(self):
        cases = _guard_cases()
        self.assertEqual([c["name"] for c in GOLDEN["guard"]], list(cases))
        for expected in GOLDEN["guard"]:
            image, raw = cases[expected["name"]]
            check = dm.check_residual(image, raw)
            with self.subTest(case=expected["name"]):
                for flag in ("input_flat", "looks_like_image", "follows_input_mean", "blew_up"):
                    self.assertEqual(getattr(check, flag), expected[flag], flag)
                for number in ("mean_abs", "correlation", "signed_mean", "dc_ratio"):
                    got, want = getattr(check, number), expected[number]
                    if want is None:
                        self.assertIsNone(got, number)
                    else:
                        self.assertAlmostEqual(got, want, delta=2e-6, msg=number)
        covered = {(c["looks_like_image"], c["blew_up"], c["input_flat"], c["follows_input_mean"])
                   for c in GOLDEN["guard"]}
        self.assertGreaterEqual(len(covered), 6)  # every branch family is exercised


def _guard_cases():
    """The guard inputs of the golden generator (same recipes, same order)."""
    g = R.levels_to_nchw(R.grain_levels(40, 40, 51), 40, 40, torch)
    checker = torch.tensor([[0.2 if (i + j) % 2 else 0.6 for j in range(40)] for i in range(40)],
                           dtype=torch.float32).expand(1, 3, 40, 40).contiguous()
    fine = torch.full((1, 3, 40, 40), 200 / 255) + R.unit_tensor((1, 3, 40, 40), 52, torch, scale=4 / 255)
    flat = torch.full((1, 3, 16, 16), 0.5)
    noise = R.unit_tensor((1, 3, 40, 40), 53, torch)
    return {
        "tiny": (g, noise * 0.003),
        "anti_checker": (checker, -(checker - checker.mean()) * 0.5),
        "follows": (g, g * 0.95 + noise * 0.01),
        "large_moderate": (g, (g - 0.5) * 0.35 + noise * 0.25),
        "large_weak": (g, (g - 0.5) * 0.05 + noise * 0.2),
        "flat_large": (flat, torch.full_like(flat, 0.3)),
        "flat_small": (flat, torch.full_like(flat, 0.05)),
        "dc_follow": (fine, torch.full_like(fine, 0.7) + noise * 0.02),
        "dc_negative": (fine, torch.full_like(fine, -0.3) + noise * 0.02),
        "blow_up": (checker, -(checker - checker.mean()) * 5.0),
        "black_input": (torch.zeros(1, 3, 8, 8), torch.full((1, 3, 8, 8), 0.2)),
        "const_output": (g, torch.full_like(g, 0.05)),
    }


class _Captured:
    output: list


@contextlib.contextmanager
def _captured_warnings():
    """Collect the pack logger's warnings (none is fine, unlike ``assertLogs``)."""
    logger = logging.getLogger("ai_studio_forge_parity")
    captured = _Captured()
    captured.output = []

    class _Collect(logging.Handler):
        def emit(self, record):
            captured.output.append(record.getMessage())

    handler = _Collect(logging.WARNING)
    previous = logger.propagate
    logger.addHandler(handler)
    logger.propagate = False
    try:
        yield captured
    finally:
        logger.removeHandler(handler)
        logger.propagate = previous


def _pipeline_levels(case):
    if case["image"] == "smooth":
        return R.smooth_levels(case["h"], case["w"])
    return R.grain_levels(case["h"], case["w"], 61)


@requires_torch
class TestDegridNodePipelineGolden(unittest.TestCase):
    """``run_degrid`` (the node body) with fake models vs the extension runtime on the same inputs."""

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        bind_torch(globals())

    def setUp(self):
        degrid_nodes.release_model(drop=True)
        self._tmp = tempfile.TemporaryDirectory()
        self.stand_in = Path(self._tmp.name) / "fake.safetensors"
        self.stand_in.write_bytes(b"not read - the loader is replaced")

    def tearDown(self):
        degrid_nodes.release_model(drop=True)
        self._tmp.cleanup()

    def _files(self, kind):
        return [degrid_nodes.ModelFile("degrid", "fake.safetensors", str(self.stand_in), (),
                                       "fake.safetensors", "fake_" + kind)]

    def _run(self, kind, image, *, oom_above=0, **kwargs):
        degrid_nodes.release_model(drop=True)
        options = dict(enabled=True, model_name="auto", mode="full", strength=1.0, tile=512,
                       device="auto", precision="fp32", keep_loaded=False, forge_quantize=True)
        options.update(kwargs)
        return degrid_nodes.run_degrid(
            image, files=self._files(kind),
            loader=lambda path: R.fake_model(kind, torch, oom_above=oom_above), **options,
        )

    def test_node_matches_the_extension_runtime_image_by_image(self):
        for case in GOLDEN["pipeline_fake"]:
            label = f"{case['kind']} {case['image']} {case['mode']} s={case['strength']} tile={case['tile']}"
            with self.subTest(case=label):
                levels = _pipeline_levels(case)
                image = R.levels_to_image(levels, case["h"], case["w"], torch)
                with _captured_warnings() as logs:
                    reports, out = self._run(case["kind"], image, oom_above=case.get("oom_above", 0),
                                             mode=case["mode"], strength=case["strength"], tile=case["tile"])
                warned = case.get("error_category") or case.get("oom_above")
                self.assertEqual(bool(logs.output), bool(warned), logs.output)
                self.assertEqual(R.sha_bytes(R.comfy_save_bytes(out, torch)), case["u8_sha"])
                (report,) = reports
                if case.get("error_category"):
                    self.assertEqual(report["status"], "skipped")
                    self.assertEqual(_error_facts(report["error"]), {
                        "error_category": case["error_category"], "error_mean_255": case["error_mean_255"],
                    })
                    self.assertTrue(torch.equal(out, image))  # skipped = untouched
                    continue
                self.assertEqual(report["status"], "ok")
                self.assertEqual(report["error"], "")
                self.assertEqual(report["tile"], case["tile_used"])
                info = case["infotext"]
                self.assertEqual(report["model"], info[KEY_MODEL])
                self.assertEqual(report["mode"], info[KEY_MODE])
                self.assertEqual(f"{round(report['strength'], 3):g}", info[KEY_STRENGTH])
                self.assertEqual(str(report["tile"]), info[KEY_TILE])
                self.assertEqual(report["precision"] != "-", KEY_PRECISION in info)
                if KEY_PRECISION in info:
                    self.assertEqual(report["precision"], info[KEY_PRECISION])

    def test_forge_quantisation_makes_off_level_input_match(self):
        case = GOLDEN["pipeline_fake"][0]
        levels = _pipeline_levels(case)
        exact = R.levels_to_image(levels, case["h"], case["w"], torch)
        wobble = R.unit_tensor(tuple(exact.shape), 71, torch, scale=0.45 / 255, offset=0.5 / 255)
        _reports, out = self._run("residual", (exact + wobble).clamp(0, 1), mode=case["mode"],
                                  strength=case["strength"], tile=case["tile"])
        self.assertEqual(R.sha_bytes(R.comfy_save_bytes(out, torch)), case["u8_sha"])
        _reports, raw = self._run("residual", (exact + wobble).clamp(0, 1), mode=case["mode"],
                                  strength=case["strength"], tile=case["tile"], forge_quantize=False)
        self.assertNotEqual(R.sha_bytes(R.comfy_save_bytes(raw, torch)), case["u8_sha"])

    def test_off_missing_model_and_zero_strength_leave_the_batch_untouched(self):
        image = R.levels_to_image(R.grain_levels(24, 32, 3), 24, 32, torch).repeat(2, 1, 1, 1)
        reports, out = self._run("residual", image, enabled=False, model_name="")
        self.assertIs(out, image)
        self.assertEqual([r["status"] for r in reports], ["off", "off"])
        self.assertEqual(reports[0]["model"], "auto")

        reports, out = degrid_nodes.run_degrid(
            image, enabled=True, model_name="qwen_v2", mode="dark", strength=0.5, tile=64, files=[],
        )
        self.assertIs(out, image)
        self.assertEqual(reports, [{
            "status": "skipped", "model": "qwen_v2", "mode": "Dark Pixels Mainly", "strength": 0.5,
            "tile": 128, "precision": "-", "error": "model not found: qwen_v2",
        }] * 2)
        reports, _out = degrid_nodes.run_degrid(image, enabled=True, model_name="auto", mode="full",
                                                strength=1, tile=512, files=[])
        self.assertEqual(reports[0]["error"], "model not found: auto")

        loads = []
        reports, out = degrid_nodes.run_degrid(
            image, enabled=True, model_name="fake.safetensors", mode="bright", strength=0.0, tile=0,
            files=self._files("residual"), loader=lambda path: loads.append(path),
        )
        self.assertIs(out, image)
        self.assertEqual(loads, [])  # strength 0: recorded like Forge, nothing loaded
        self.assertEqual(reports[0], {
            "status": "ok", "model": "fake_residual", "mode": "Bright Pixels Mainly", "strength": 0.0,
            "tile": 0, "precision": "-", "error": "",
        })

    def test_batch_items_are_judged_one_by_one_and_alpha_is_kept(self):
        levels = R.smooth_levels(40, 48)
        item = R.levels_to_image(levels, 40, 48, torch)
        white = torch.ones_like(item)
        alpha = torch.linspace(0, 1, 40 * 48).reshape(1, 40, 48, 1)
        batch = torch.cat([torch.cat([item, alpha], -1), torch.cat([white, alpha], -1)], 0)

        def picky(path):
            model = R.fake_model("residual", torch)
            forward = model.forward

            def maybe_fail(x):
                if float(x.min()) >= 1.0:
                    raise ValueError("white tile refused (test)")
                return forward(x)

            model.forward = maybe_fail
            return model

        with self.assertLogs("ai_studio_forge_parity", "WARNING") as logs:
            reports, out = degrid_nodes.run_degrid(
                batch, enabled=True, model_name="auto", mode="full", strength=1.0, tile=512,
                files=self._files("residual"), loader=picky,
            )
        self.assertEqual(len(logs.output), 1)
        self.assertIn("white tile refused", logs.output[0])
        self.assertEqual([r["status"] for r in reports], ["ok", "skipped"])
        self.assertEqual(reports[1]["error"], "ValueError: white tile refused (test)")
        self.assertEqual(tuple(out.shape), (2, 40, 48, 4))
        self.assertTrue(torch.equal(out[..., 3:], batch[..., 3:]))
        self.assertTrue(torch.equal(out[1], batch[1]))
        single, alone = degrid_nodes.run_degrid(
            item, enabled=True, model_name="auto", mode="full", strength=1.0, tile=512,
            files=self._files("residual"), loader=lambda p: R.fake_model("residual", torch),
        )
        self.assertTrue(torch.equal(out[0, ..., :3], alone[0]))

    def test_a_model_that_cannot_be_placed_skips_the_batch(self):
        image = R.levels_to_image(R.grain_levels(16, 16, 9), 16, 16, torch).repeat(2, 1, 1, 1)
        original = degrid_nodes.acquire

        def refuse(*args, **kwargs):
            raise RuntimeError("no room for the model (test)")

        degrid_nodes.acquire = refuse
        self.addCleanup(setattr, degrid_nodes, "acquire", original)
        with self.assertLogs("ai_studio_forge_parity", "WARNING"):
            reports, out = self._run("residual", image)
        self.assertIs(out, image)
        self.assertEqual([r["error"] for r in reports], ["RuntimeError: no room for the model (test)"] * 2)
        self.assertEqual({r["status"] for r in reports}, {"skipped"})

    def test_a_user_cancel_is_not_swallowed(self):
        class InterruptProcessingException(Exception):
            pass

        def cancelling(path):
            model = R.fake_model("residual", torch)

            def stop(x):
                raise InterruptProcessingException()

            model.forward = stop
            return model

        image = R.levels_to_image(R.grain_levels(16, 16, 9), 16, 16, torch)
        with self.assertRaises(InterruptProcessingException):
            degrid_nodes.run_degrid(image, enabled=True, model_name="auto", mode="full", strength=1.0,
                                    tile=512, files=self._files("residual"), loader=cancelling)

    def test_node_apply_returns_ui_reports_and_json(self):
        node = degrid_nodes.ForgeNeoAnimaVAEDeGrid()
        image = R.levels_to_image(R.grain_levels(16, 16, 9), 16, 16, torch)
        out = node.apply(image, False, "auto", "Full", 1.0, 512, "auto", "fp32", False, True)
        self.assertEqual(set(out), {"ui", "result"})
        reports = out["ui"]["ai_studio_degrid"]
        self.assertIs(out["result"][0], image)
        self.assertEqual(json.loads(out["result"][1]), reports)
        self.assertEqual(reports[0]["status"], "off")

    def test_fp16_overflow_reruns_the_tile_in_fp32(self):
        calls = []

        class Flaky(torch.nn.Module):
            padder_size = 16

            def forward(self, x):
                calls.append(bool(getattr(Flaky, "in_autocast", False)))
                if getattr(Flaky, "in_autocast", False):
                    return torch.full_like(x, float("inf"))
                return x * 0.0 + 0.001

        class FakeAutocast:
            def __enter__(self):
                Flaky.in_autocast = True

            def __exit__(self, *exc):
                Flaky.in_autocast = False

        fn, retiles = degrid_runner.residual_function(Flaky(), "cpu", True, autocast_context=FakeAutocast)
        out = fn(torch.zeros(1, 3, 16, 16))
        self.assertEqual(calls, [True, False])
        self.assertEqual(retiles, [1])
        self.assertTrue(bool(torch.isfinite(out).all()))
        self.assertFalse(degrid_runner.use_autocast("cpu", "fp16"))  # CPU is always fp32
        self.assertFalse(degrid_runner.use_autocast("cuda", "fp32"))
        self.assertTrue(degrid_runner.use_autocast("cuda", "FP16"))

    def test_oom_halving_stops_below_the_minimum_tile(self):
        image = R.levels_to_image(R.smooth_levels(300, 300), 300, 300, torch)
        with self.assertLogs("ai_studio_forge_parity", "WARNING") as logs:
            reports, out = self._run("residual", image, oom_above=100 * 100, tile=256)
        self.assertIn("retrying with tile 128", logs.output[0])
        self.assertEqual(reports[0]["status"], "skipped")
        self.assertTrue(reports[0]["error"].startswith("RuntimeError: CUDA out of memory"))
        self.assertTrue(torch.equal(out, image))


@requires_torch
class TestDegridComfyMemory(unittest.TestCase):
    """GPU placement goes through ComfyUI's model management — exercised with fakes only.

    Nothing here allocates on a GPU: the model is a stand-in whose ``to`` only records
    the device, and ``comfy.model_management``/``comfy.model_patcher`` are fakes.
    """

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        bind_torch(globals())

    def setUp(self):
        degrid_nodes.release_model(drop=True)
        events = self.events = []

        class Param:
            def __init__(self):
                self.device = torch.device("cpu")

        class StandIn:
            def __init__(self):
                self.param = Param()

            def parameters(self):
                return iter([self.param])

            def to(self, device):
                events.append(("to", str(torch.device(device))))
                self.param.device = torch.device(device)
                return self

        class Patcher:
            def __init__(self, model, load_device, offload_device):
                self.model, self.load_device, self.offload_device = model, load_device, offload_device

        class Loaded:
            def __init__(self, patcher):
                self.model = patcher

            def model_unload(self):
                events.append(("unload",))
                self.model.model.to(self.model.offload_device)

        management = SimpleNamespace(current_loaded_models=[])

        def load_models_gpu(patchers, memory_required=0, force_full_load=False):
            events.append(("load", memory_required, force_full_load))
            for patcher in patchers:
                if not any(entry.model is patcher for entry in management.current_loaded_models):
                    management.current_loaded_models.append(Loaded(patcher))
                patcher.model.to(patcher.load_device)

        management.load_models_gpu = load_models_gpu
        management.soft_empty_cache = lambda: events.append(("empty",))
        self.model = StandIn()
        self.management = management
        fake_patcher_module = SimpleNamespace(CoreModelPatcher=Patcher, ModelPatcher=Patcher)
        originals = (degrid_nodes._management, sys.modules.get("comfy"), sys.modules.get("comfy.model_patcher"))
        degrid_nodes._management = lambda: management
        sys.modules["comfy"] = SimpleNamespace(__path__=[])
        sys.modules["comfy.model_patcher"] = fake_patcher_module

        def restore():
            degrid_nodes._management = originals[0]
            for name, value in (("comfy", originals[1]), ("comfy.model_patcher", originals[2])):
                if value is None:
                    sys.modules.pop(name, None)
                else:
                    sys.modules[name] = value
            degrid_nodes.release_model(drop=True)

        self.addCleanup(restore)
        degrid_nodes._CACHE.update(key=("stand-in", 0, 0), model=self.model, patcher=None)

    def test_gpu_placement_uses_load_models_gpu_and_release_unloads_it(self):
        gpu = torch.device("cuda", 0)  # a device name only — never allocated here
        degrid_nodes.acquire(self.model, gpu, 1536.0 * 512 * 512)
        self.assertEqual(self.events, [("load", 1536.0 * 512 * 512, True), ("to", "cuda:0")])
        patcher = degrid_nodes._CACHE["patcher"]
        self.assertEqual(patcher.load_device, gpu)
        self.assertEqual(patcher.offload_device, torch.device("cpu"))
        degrid_nodes.acquire(self.model, gpu, 1.0)  # same patcher, loaded again
        self.assertIs(degrid_nodes._CACHE["patcher"], patcher)
        self.events.clear()
        self.assertTrue(degrid_nodes.release_model())
        self.assertEqual(self.events, [("unload",), ("to", "cpu"), ("empty",)])
        self.assertEqual(self.management.current_loaded_models, [])
        self.assertIs(degrid_nodes._CACHE["model"], self.model)  # the CPU copy stays cached

    def test_cpu_run_after_a_kept_gpu_model_brings_it_back(self):
        degrid_nodes.acquire(self.model, torch.device("cuda", 0), 1.0)
        self.events.clear()
        degrid_nodes.acquire(self.model, torch.device("cpu"), 1.0)
        self.assertEqual(self.events, [("unload",), ("to", "cpu"), ("empty",)])
        self.assertEqual(self.model.param.device, torch.device("cpu"))


# ─────────────────────────────────────────────────────────────────────────
# real ComfyUI runtime (CPU)
# ─────────────────────────────────────────────────────────────────────────
def _real_model_path() -> Path:
    return Path(os.environ.get("AISTUDIO_DEGRID_MODEL") or DEFAULT_REAL_MODEL)


def _file_sha256(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 22), b""):
            digest.update(chunk)
    return digest.hexdigest()


@requires_torch
@unittest.skipUnless(_COMFY_ROOT, "Set AISTUDIO_COMFY_TEST_ROOT to run DeGrid on the real ComfyUI runtime (CPU)")
class TestDegridOnRealComfy(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        bind_torch(globals())
        import comfy.model_management as management
        import comfy.utils
        import folder_paths

        if management.get_torch_device().type != "cpu":
            raise unittest.SkipTest("ComfyUI did not start on CPU - refusing to touch the GPU")
        cls.management = management
        cls.comfy_utils = comfy.utils
        # CPU conv reductions depend on the thread count — use the generator's.
        cls._threads = torch.get_num_threads()
        torch.set_num_threads(GOLDEN["threads"])
        cls.same_build = (torch.__version__, torch.backends.cpu.get_cpu_capability()) == (
            GOLDEN["torch"], GOLDEN["cpu_capability"])
        cls.folder_paths = folder_paths
        cls._tmp = tempfile.TemporaryDirectory()
        cls.models = Path(cls._tmp.name)
        cls._saved = {key: folder_paths.folder_names_and_paths.get(key)
                      for key in ("upscale_models", "degrid")}

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls._threads)
        for key, value in cls._saved.items():
            if value is None:
                cls.folder_paths.folder_names_and_paths.pop(key, None)
            else:
                cls.folder_paths.folder_names_and_paths[key] = value
        degrid_nodes.release_model(drop=True)
        cls._tmp.cleanup()
        super().tearDownClass()

    def setUp(self):
        degrid_nodes.release_model(drop=True)

    def _use_folders(self, upscale, degrid):
        registry = self.folder_paths.folder_names_and_paths
        registry["upscale_models"] = ([str(p) for p in upscale], {".safetensors", ".pth", ".pt"})
        registry["degrid"] = ([str(p) for p in degrid], {".safetensors", ".pth", ".pt"})
        cache = getattr(self.folder_paths, "filename_list_cache", None)
        if isinstance(cache, dict):
            cache.clear()

    def test_pack_tiler_equals_comfy_tiled_scale(self):
        for case in GOLDEN["tiling"]:
            if case["tile"] <= 0:
                continue
            h, w = case["shape"]
            with self.subTest(shape=case["shape"], tile=case["tile"]):
                t = R.dyadic_nchw(h, w, case["seed"], torch)
                fn = lambda p: dm.call_padded(lambda q: R.fake_forward("residual", q, torch), p, 16)  # noqa: E731
                ours = dm.tiled_residual(t, fn, tile=case["tile"])
                comfy = self.comfy_utils.tiled_scale(
                    t, fn, tile_x=case["tile"], tile_y=case["tile"], overlap=dm.TILE_OVERLAP,
                    upscale_amount=1, out_channels=3, output_device="cpu",
                )
                self.assertTrue(torch.equal(ours, comfy))
                self.assertEqual(R.sha_float(ours, torch), case["sha"])

    def test_tiny_nafnets_hit_the_origin_skip_categories(self):
        from safetensors.torch import save_file
        from spandrel.architectures.NAFNet import NAFNet

        folder = self.models / "tiny"
        folder.mkdir(exist_ok=True)
        spec = GOLDEN["tiny_nafnet"]
        h, w = spec["image"]
        image = R.levels_to_image(R.smooth_levels(h, w), h, w, torch)
        for case in spec["cases"]:
            with self.subTest(kind=case["kind"]):
                path = folder / f"tiny_{case['kind']}.safetensors"
                save_file(R.tiny_nafnet_state(case["kind"], torch, NAFNet), str(path),
                          metadata={"modelspec.version": "0.1"})
                self.assertEqual(degrid_files.classify(path), (case["nafnet_file"], tuple(case["version"])))
                model = degrid_nodes.load_nafnet(str(path))
                x = image.movedim(-1, 1).contiguous()  # the node's layout (and Forge's)
                with torch.inference_mode():
                    raw = dm.tiled_residual(x, lambda p: dm.call_padded(model, p, 16), tile=512)
                if self.same_build:
                    self.assertEqual(R.sha_float(raw, torch), case["raw_sha"])
                else:  # another torch build / CPU: reductions may differ in the last bits
                    self.assertAlmostEqual(dm.check_residual(x, raw).mean_abs, case["mean_abs"], delta=1e-5)
                self._use_folders([], [folder])
                with _captured_warnings() as logs:
                    result = degrid_nodes.ForgeNeoAnimaVAEDeGrid().apply(
                        image, True, path.name, "Full", 1.0, 512, "auto", "fp32", False, True,
                    )
                self.assertEqual(len(logs.output), 1)
                (report,) = json.loads(result["result"][1])
                self.assertTrue(torch.equal(result["result"][0], image))
                self.assertEqual(report["status"], "skipped")
                self.assertEqual(_error_facts(report["error"]), {
                    "error_category": case["error_category"], "error_mean_255": case["error_mean_255"],
                })

    def test_real_degrid_model_saves_the_same_bytes_as_forge(self):
        spec = GOLDEN["real_model"]
        path = _real_model_path()
        if not spec.get("present") or not path.is_file():
            self.skipTest(f"{path} is not here")
        if path.stat().st_size != spec["size"] or _file_sha256(path) != spec["sha256"]:
            self.skipTest(f"{path.name} is not the file the golden was made with")
        self._use_folders([path.parent], [self.models / "empty"])
        choices = degrid_nodes.ForgeNeoAnimaVAEDeGrid.INPUT_TYPES()["required"]["model_name"][0]
        self.assertIn(path.name, choices)
        h, w = spec["image"]
        image = R.levels_to_image(R.gridded_levels(h, w), h, w, torch)
        node = degrid_nodes.ForgeNeoAnimaVAEDeGrid()
        for case in spec["cases"]:
            with self.subTest(mode=case["mode"], strength=case["strength"], tile=case["tile"]):
                result = node.apply(image, True, path.name, dm.MODE_LABELS[case["mode"]], case["strength"],
                                    case["tile"], "cpu", "fp32", False, True)
                out = result["result"][0]
                (report,) = result["ui"]["ai_studio_degrid"]
                self.assertEqual(report["status"], "ok", report["error"])
                self.assertEqual(R.sha_bytes(R.comfy_save_bytes(out, torch)), case["u8_sha"])
                info = case["infotext"]
                self.assertEqual(report["model"], info[KEY_MODEL])
                self.assertEqual(report["mode"], info[KEY_MODE])
                self.assertEqual(f"{round(report['strength'], 3):g}", info[KEY_STRENGTH])
                self.assertEqual(str(report["tile"]), info[KEY_TILE])
                self.assertEqual(report["precision"], info[KEY_PRECISION])

    def test_release_hook_on_the_real_model_management(self):
        degrid_nodes.install_unload_release_hook(self.management)
        self.assertTrue(getattr(self.management.unload_all_models, "_ai_studio_degrid_release_hook", False))
        degrid_nodes._CACHE.update(key=("k", 0, 0), model=None, patcher=None)
        self.management.unload_all_models()
        self.assertIsNone(degrid_nodes._CACHE["key"])


if __name__ == "__main__":
    unittest.main()
