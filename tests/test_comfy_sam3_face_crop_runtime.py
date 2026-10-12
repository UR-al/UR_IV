r"""Pack 1.8.0 SAM3 face crop on the real ComfyUI runtime (CPU) == sam-extra's face crop.

The pack node runs through the user's real Easy SAM3 ``Sam3ImageSegmentation`` node (its own sorting, stacking and
``obj_masks``/``boxes``/``scores`` outputs, read by ``sam3_nodes._detect_token_instances``) with a fake SAM3
*processor* (colour components, no weights) and a fake ultralytics; the installed sam-extra's ``run_sam3_on_pil``
with the same fake processor and detector is the oracle (``tests/test_comfy_sam3_face_crop.py`` has the scenes and
the comparison). Also: this ComfyUI's ``face_crop_detector`` status and the node's inputs from the real import.

Skipped unless ``AISTUDIO_COMFY_TEST_ROOT`` is set. Run it with the portable python, the repo first on
``sys.path``, ``CUDA_VISIBLE_DEVICES=-1`` (ComfyUI is forced to ``--cpu`` before its model management loads; no
server, no GPU, no SAM3 weights)::

    $env:CUDA_VISIBLE_DEVICES='-1'; $env:AISTUDIO_COMFY_TEST_ROOT='C:\ComfyUI_windows_portable\ComfyUI'
    C:\ComfyUI_windows_portable\python_embeded\python.exe -X utf8 -c "import sys, unittest; sys.path.insert(0, r'<repo>'); unittest.main(module=None, argv=['x', 'tests.test_comfy_sam3_face_crop_runtime', '-v'])"

Point ``AISTUDIO_FORGE_EXTENSION_DIR`` at a sam-extra v0.34.1 tree to compare the crop growth too (with v0.34.0
installed the pack runs with ``GROW_ROUNDS`` 0 against it).
"""
from __future__ import annotations

import importlib.util
import os
import sys
import unittest
from pathlib import Path
from unittest import mock

_COMFY_ROOT = os.environ.get("AISTUDIO_COMFY_TEST_ROOT")
if _COMFY_ROOT:  # force --cpu before anything loads ComfyUI's model management
    if _COMFY_ROOT not in sys.path:
        sys.path.append(_COMFY_ROOT)
    from comfy.cli_args import args as _comfy_args  # noqa: E402

    _comfy_args.cpu = True

import numpy as np  # noqa: E402

from tests._optional_deps import bind_torch, requires_torch  # noqa: E402
from tests._sam_extra_ext import requires_extension  # noqa: E402

torch = None  # bind_torch()
_EASY_MODULE = "aistudio_test_easy_sam3"


def _easy_sam3():
    """The user's ComfyUI-Easy-SAM3 package (its folder name has a dash), loaded once as a package."""
    if _EASY_MODULE in sys.modules:
        return sys.modules[_EASY_MODULE]
    folder = Path(_COMFY_ROOT) / "custom_nodes" / "comfyui-easy-sam3"
    if not folder.is_dir():
        raise unittest.SkipTest(f"Easy SAM3 is not installed in {folder}")
    spec = importlib.util.spec_from_file_location(_EASY_MODULE, folder / "__init__.py",
                                                  submodule_search_locations=[str(folder)])
    module = importlib.util.module_from_spec(spec)
    sys.modules[_EASY_MODULE] = module
    spec.loader.exec_module(module)
    return module


class _TorchProcessor:
    """The fake SAM3 processor behind the real Easy SAM3 node: colour components as torch tensors."""

    def __init__(self):
        from tests import test_comfy_sam3_face_crop as cases

        self.cases = cases
        self.calls = []

    def set_confidence_threshold(self, value):
        self.threshold = value

    def set_image(self, image):
        return {"rgb": np.asarray(image.convert("RGB")).copy()}

    def set_text_prompt(self, prompt, state):
        rgb = state["rgb"]
        h, w = rgb.shape[:2]
        self.calls.append((self.cases._digest(rgb), prompt))
        masks, boxes = self.cases._components(rgb, prompt)
        state["masks"] = (torch.from_numpy(np.stack(masks))[:, None] if masks
                          else torch.zeros((0, 1, h, w), dtype=torch.bool))
        state["boxes"] = torch.tensor(np.asarray(boxes, dtype=np.float32).reshape(-1, 4))
        state["scores"] = torch.full((len(masks),), 0.9, dtype=torch.float32)
        return state


class _Model:
    def to(self, *args, **kwargs):
        return self


@unittest.skipUnless(_COMFY_ROOT, "AISTUDIO_COMFY_TEST_ROOT not set — real-ComfyUI tier skipped")
@requires_torch
class RealEasySam3Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        bind_torch(globals())
        from tests import test_comfy_sam3_face_crop as cases

        cases.torch = torch
        cls.cases = cases
        cls.segmentation = _easy_sam3().Sam3ImageSegmentation

    def _bundle(self, processor):
        return {"model": _Model(), "processor": processor, "device": torch.device("cpu"), "dtype": torch.float32,
                "segmentor": "image"}

    def test_instances_from_the_real_node_outputs(self):
        from comfy_custom_nodes.ai_studio_forge_parity import sam3_nodes

        rgb = np.full((60, 80, 3), 128, dtype=np.uint8)
        rgb[5:15, 5:20] = self.cases.COLORS["eye"]
        rgb[30:40, 50:70] = self.cases.COLORS["eye"]
        image = torch.from_numpy(rgb[None]).float().div(255.0)
        bundle = self._bundle(_TorchProcessor())
        pieces, boxes, scores = sam3_nodes._detect_token_instances(
            self.segmentation, bundle, image, "eye", 0.4, -1)
        self.assertEqual(len(pieces), 2)
        self.assertEqual(sorted(boxes), [[5.0, 5.0, 20.0, 15.0], [50.0, 30.0, 70.0, 40.0]])
        self.assertEqual(scores, [float(np.float32(0.9))] * 2)
        self.assertEqual(sum(int(p.sum()) for p in pieces), 10 * 15 + 10 * 20)
        one = sam3_nodes._detect_token_instances(self.segmentation, bundle, image, "eye", 0.4, 1)
        self.assertEqual((len(one[0]), len(one[1]), len(one[2])), (1, 1, 1), "detection limit, a single object")
        self.assertEqual(sam3_nodes._detect_token_instances(self.segmentation, bundle, image, "hair", 0.4, -1),
                         ([], [], []), "the no-detection sentinel")

    @requires_extension
    def test_scenes_match_sam_extra_through_the_real_node(self):
        from comfy_custom_nodes.ai_studio_forge_parity import sam3_face_crop as pfc

        growth = self.cases._oracle_has_growth()
        parity = self.cases.OracleParityTests("test_random_scenes")
        for index, rgb, faces, scores, opts, user_mask in self.cases._scenes(24, seed=5):
            with self.subTest(scene=index):
                oracle, kwargs = parity._run_both(index, rgb, faces, scores, opts, user_mask)
                processor = _TorchProcessor()
                segmenter = self.segmentation
                bundle = self._bundle(processor)
                with mock.patch.object(pfc, "GROW_ROUNDS", pfc.GROW_ROUNDS if growth else 0):
                    pack = self._run(rgb, opts, faces, scores, kwargs, segmenter, bundle)
                pack.calls = processor.calls
                parity._assert_same(pack, oracle, f"scene {index}")

    def _run(self, rgb, opts, faces, scores, kwargs, segmenter, bundle):
        import json
        import tempfile
        import types

        from comfy_custom_nodes.ai_studio_forge_parity import sam3_nodes

        yolo = self.cases.FakeUltralytics([(faces, scores)])
        lines = []
        image = torch.from_numpy(rgb[None]).float().div(255.0)
        user_mask = kwargs["user_mask"]
        with tempfile.TemporaryDirectory() as folder:
            model = self.cases._model_file(Path(folder))
            with mock.patch.dict(sys.modules, {"ultralytics": yolo.module}), \
                    mock.patch.object(sam3_nodes, "_resolve_easy_node", return_value=segmenter), \
                    mock.patch.object(sam3_nodes, "_log_face_crop", side_effect=lines.append):
                out = sam3_nodes.ForgeNeoSAM3Mask().segment(
                    image, prompt=opts["prompt"], exclude_prompt=kwargs["exclude"], mask_mode="Individual",
                    mask_source="intersection" if user_mask is not None else "generated", threshold=0.4,
                    detection_limit=-1, convex_hull=kwargs["hull"], mask_dilation=kwargs["dilation"],
                    mask_outline_px=kwargs["outline"], mask_blur=0, invert=False, save_artifacts=False,
                    unload_after=False, sam3_model=bundle,
                    manual_mask=None if user_mask is None else torch.from_numpy(user_mask.astype(np.float32))[None],
                    face_crop=True, face_crop_model=str(model), face_crop_confidence=0.3,
                )
        selected, combined, individuals, overlay, boxes, out_scores, report = out
        return types.SimpleNamespace(combined=combined, individuals=individuals, boxes=boxes, scores=out_scores,
                                     report=json.loads(report), lines=lines)

    def test_this_comfyui_publishes_its_detector_status(self):
        from comfy_custom_nodes.ai_studio_forge_parity import sam3_face_detect, sam3_nodes

        optional = sam3_nodes.ForgeNeoSAM3Mask.INPUT_TYPES()["optional"]
        status = optional[sam3_face_detect.DETECTOR_INPUT][0]
        found = importlib.util.find_spec("ultralytics") is not None
        self.assertEqual(status, [sam3_face_detect.DETECTOR_READY if found else sam3_face_detect.DETECTOR_MISSING])
        print(f"\n[face crop] this ComfyUI's face_crop_detector: {status[0]}", file=sys.stderr)


if __name__ == "__main__":
    unittest.main()
