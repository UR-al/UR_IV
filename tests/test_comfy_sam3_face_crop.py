"""Pack 1.8.0 SAM3 face crop == sam-extra's face crop (CPU, fake SAM3 and face detector, no GPU, no ComfyUI).

``ForgeNeoSAM3Mask`` with ``face_crop`` runs sam-extra's face-crop branch of ``run_sam3_on_pil``
(``comfy_custom_nodes/ai_studio_forge_parity/sam3_face_crop*.py`` — sam-extra's own code under its author's MIT grant).
The installed extension is the oracle (``tests/_sam_extra_ext.EXT_ROOT``; the pack never imports it):

* ``GeometryParityTests`` — pure, no torch: the copied geometry (crop boxes, plans, ownership, the v0.34.1 growth
  functions when the oracle has them) gives the oracle's values on random inputs.
* ``OracleParityTests`` — torch: random synthetic scenes (colour components as SAM3 objects) through the oracle's
  ``run_sam3_on_pil`` and the pack node: the same Individual masks, Combined mask, boxes, scores, face plan and growth
  records (status line and meta), console lines and SAM3 calls (prompt and crop pixels, in order). The oracle is
  v0.34.0 until sam-extra 0.34.1 is installed: then the pack runs with ``GROW_ROUNDS`` 0 against it (v0.34.0 has no
  growth; with 0 rounds the v0.34.1 code is v0.34.0's) and, with growth on, every scene that did not grow must still
  match. Point the oracle at a v0.34.1 tree (``AISTUDIO_FORGE_EXTENSION_DIR``) to compare the growth itself.
* ``PackGrowthTests`` / ``PackNodeTests`` / ``DetectorTests`` — the pack alone: sam-extra's growth cases, the node's
  inputs and report, face crop off unchanged, no SAM3 load without faces, batches, the overlay frames, the ultralytics
  call (model path, confidence, device rule), model lookup and the informational ``face_crop_detector`` COMBO.

torch is imported lazily (tests/_optional_deps).
"""
from __future__ import annotations

import contextlib
import hashlib
import importlib
import io
import json
import random
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from comfy_custom_nodes.ai_studio_forge_parity import sam3_face_crop as pfc
from comfy_custom_nodes.ai_studio_forge_parity import sam3_face_detect as pfd
from tests._optional_deps import bind_torch, requires_torch
from tests._sam_extra_ext import EXT_ROOT, requires_extension

torch = None  # bind_torch() in the torch classes

COLORS = {"eye": (255, 0, 0), "mouth": (0, 255, 0), "skin": (250, 220, 200), "hair": (0, 0, 255)}
BACKGROUND = (128, 128, 128)
SIZE = (256, 192)   # (width, height)
PROMPTS = ["eye", "hair", "eye, mouth / skin", "hair / eye", "skin, hair"]


def _ext(name: str):
    root = str(EXT_ROOT)
    if root not in sys.path:
        sys.path.insert(0, root)
    return importlib.import_module(f"sam3ext.{name}")


def _oracle_has_growth() -> bool:
    return EXT_ROOT is not None and hasattr(_ext("face_crop"), "growth_requests")


def _components(rgb: np.ndarray, prompt: str):
    """The fake SAM3: one 8-connected component of the prompt's colour = one object (mask, box)."""
    import cv2

    h, w = rgb.shape[:2]
    color = COLORS.get(prompt)
    hit = np.all(rgb == np.array(color, dtype=np.uint8), axis=-1) if color else np.zeros((h, w), bool)
    count, labels = cv2.connectedComponents(hit.astype(np.uint8), connectivity=8)
    masks, boxes = [], []
    for label in range(1, count):
        mask = labels == label
        ys, xs = np.nonzero(mask)
        masks.append(mask)
        boxes.append([xs.min(), ys.min(), xs.max() + 1, ys.max() + 1])
    return masks, boxes


def _digest(rgb: np.ndarray) -> str:
    return hashlib.md5(np.ascontiguousarray(rgb).tobytes()).hexdigest() + f"{rgb.shape[:2]}"


class OracleProcessor:
    """Fake Sam3Processor for the oracle: records (crop pixels, prompt) per text prompt."""

    def __init__(self, keep=None):
        self.calls: list[tuple[str, str]] = []
        self.keep = keep or (lambda rgb, prompt: True)

    def set_confidence_threshold(self, value):
        self.threshold = value

    def set_image(self, image):
        return {"backbone_out": {"rgb": np.asarray(image.convert("RGB")).copy()}}

    def set_text_prompt(self, prompt, state):
        rgb = state["backbone_out"]["rgb"]
        h, w = rgb.shape[:2]
        self.calls.append((_digest(rgb), prompt))
        masks, boxes = _components(rgb, prompt) if self.keep(rgb, prompt) else ([], [])
        state["masks"] = np.stack(masks)[:, None] if masks else np.zeros((0, 1, h, w), dtype=bool)
        state["boxes"] = np.asarray(boxes, dtype=np.float32).reshape(-1, 4)
        state["scores"] = np.full((len(masks),), 0.9, dtype=np.float32)
        return state


class PackSegmenter:
    """Fake Easy SAM3 ``Sam3ImageSegmentation.execute`` with its output shapes: (combined, images, obj_masks, boxes,
    scores), score-sorted objects, the detection limit, and the no-detection sentinel (empty mask, box 0, score 0)."""

    def __init__(self, keep=None):
        self.calls: list[tuple[str, str]] = []
        self.keep = keep or (lambda rgb, prompt: True)

    def execute(self, sam3_model, images, prompt, threshold=0.3, keep_model_loaded=False, add_background="none",
                detection_limit=-1, coordinates_positive=None, coordinates_negative=None, bboxes=None, mask=None):
        assert images.shape[0] == 1
        rgb = (images[0, ..., :3] * 255.0).round().clamp(0, 255).to(torch.uint8).numpy()
        h, w = rgb.shape[:2]
        self.calls.append((_digest(rgb), prompt))
        masks, boxes = _components(rgb, prompt) if self.keep(rgb, prompt) else ([], [])
        if masks:
            raw = torch.from_numpy(np.stack(masks))[:, None]
            box_t = torch.tensor(np.asarray(boxes, dtype=np.float32))
            scores = torch.full((len(masks),), 0.9, dtype=torch.float32)
            order = torch.argsort(scores, descending=True, stable=True)
            raw, box_t, scores = raw[order], box_t[order], scores[order]
            if detection_limit > -1:
                raw, box_t, scores = raw[:detection_limit], box_t[:detection_limit], scores[:detection_limit]
            combined = (raw.squeeze(1).float().sum(dim=0) > 0).float()
        else:
            raw = torch.zeros(1, h, w)
            box_t = torch.zeros(1, 4)
            scores = torch.zeros(1)
            combined = torch.zeros(h, w)
        return (
            combined.unsqueeze(0), images, torch.stack([raw], dim=0),
            torch.stack([box_t], dim=0).squeeze(0).tolist(), torch.stack([scores], dim=0).squeeze().tolist(),
        )


class FakeUltralytics:
    """``sys.modules['ultralytics']`` stand-in: ``YOLO(path)(image, conf=…, device=…, verbose=…)`` -> boxes."""

    def __init__(self, per_image):
        self.per_image = list(per_image)   # [(boxes, scores)] per call
        self.loads: list[str] = []
        self.calls: list[tuple] = []
        outer = self

        class _Boxes:
            def __init__(self, boxes, scores):
                self.xyxy = np.asarray(boxes, dtype=np.float32).reshape(-1, 4)
                self.conf = np.asarray(scores, dtype=np.float32)

        class YOLO:
            def __init__(self, path):
                outer.loads.append(path)

            def __call__(self, image, conf=0.25, device=None, verbose=True):
                outer.calls.append((image.size, conf, device, verbose))
                boxes, scores = outer.per_image[min(len(outer.calls), len(outer.per_image)) - 1]
                return [types.SimpleNamespace(boxes=_Boxes(boxes, scores))]

        self.module = types.ModuleType("ultralytics")
        self.module.YOLO = YOLO


class _FakeModel:
    def to(self, target):
        return self


def _draw_face(rgb, box, *, eyes=True, mouth=True):
    x0, y0, x1, y1 = box
    rgb[y0:y1, x0:x1] = COLORS["skin"]
    if eyes:
        rgb[y0 + 10:y0 + 14, x0 + 8:x0 + 12] = COLORS["eye"]
        rgb[y0 + 10:y0 + 14, x1 - 12:x1 - 8] = COLORS["eye"]
    if mouth:
        rgb[y1 - 12:y1 - 9, x0 + 16:x1 - 16] = COLORS["mouth"]


def _color(rgb, name):
    return np.all(rgb == np.array(COLORS[name], dtype=np.uint8), axis=-1)


def _scenes(count: int, seed: int = 20261012):
    """sam-extra's equivalence scenes (v0.34.1 check): faces, random colour blocks, random options."""
    rng = np.random.default_rng(seed)
    for index in range(count):
        width, height = int(rng.integers(120, 320)), int(rng.integers(120, 320))
        rgb = np.full((height, width, 3), BACKGROUND, dtype=np.uint8)
        faces = []
        for _ in range(int(rng.integers(1, 4))):
            side = int(rng.integers(16, 70))
            x0, y0 = int(rng.integers(-10, width - 10)), int(rng.integers(-10, height - 10))
            faces.append([x0, y0, x0 + side, y0 + side])
            cx0, cy0 = max(0, x0), max(0, y0)
            rgb[cy0:max(cy0, y0 + side), cx0:max(cx0, x0 + side)] = COLORS["skin"]
        for _ in range(int(rng.integers(2, 10))):
            name = ["eye", "mouth", "hair", "skin"][int(rng.integers(0, 4))]
            bw, bh = int(rng.integers(2, 40)), int(rng.integers(2, 40))
            x0, y0 = int(rng.integers(0, width)), int(rng.integers(0, height))
            rgb[y0:y0 + bh, x0:x0 + bw] = COLORS[name]
        opts = {
            "prompt": PROMPTS[int(rng.integers(0, len(PROMPTS)))],
            "exclude_prompt": ["", "mouth", "eye"][int(rng.integers(0, 3))],
            "mask_dilation": int(rng.choice([0, 0, 2])),
            "mask_hull": bool(rng.integers(0, 2)),
            "mask_outline_px": int(rng.choice([0, 0, 2])),
        }
        user_mask = None
        if index % 7 == 3:
            user_mask = np.zeros((height, width), dtype=bool)
            ux, uy = int(rng.integers(0, width // 2)), int(rng.integers(0, height // 2))
            user_mask[uy:uy + height // 2, ux:ux + width // 2] = True
        scores = [float(np.float32(s)) for s in rng.uniform(0.3, 0.95, len(faces))]
        yield index, rgb, faces, scores, opts, user_mask


# --------------------------------------------------------------------------------------------------------------------


@requires_extension
class GeometryParityTests(unittest.TestCase):
    """The pack's copy of ``sam3ext/face_crop.py`` gives the oracle's values (pure functions, no torch)."""

    @classmethod
    def setUpClass(cls):
        cls.ext = _ext("face_crop")

    def _crop(self, crop):
        return pfc.CropBox(*crop.as_tuple())

    def test_constants_are_the_extensions(self):
        for name in ("EXPAND_SIDE", "EXPAND_TOP", "EXPAND_BOTTOM", "MIN_CROP_SIDE", "SAME_FACE_IOU"):
            with self.subTest(name=name):
                self.assertEqual(getattr(pfc, name), getattr(self.ext, name))
        if not _oracle_has_growth():
            self.skipTest("installed sam-extra has no crop growth (v0.34.0) — growth constants not compared")
        for name in ("GROW_FACTOR", "GROW_ROUNDS", "TOUCH_BAND", "TOUCH_MIN_PX", "TOUCH_MIN_FRACTION", "SIDES"):
            with self.subTest(name=name):
                self.assertEqual(getattr(pfc, name), getattr(self.ext, name))

    def test_crop_boxes_and_plans_match_on_random_boxes(self):
        rng = random.Random(7)
        for _ in range(400):
            size = (rng.randint(8, 400), rng.randint(8, 400))
            boxes = [[rng.uniform(-50, size[0] + 50), rng.uniform(-50, size[1] + 50),
                      rng.uniform(-50, size[0] + 50), rng.uniform(-50, size[1] + 50)] for _ in range(rng.randint(0, 5))]
            if rng.random() < 0.1:
                boxes.append([float("nan"), 0, 1, 1])
            scores = [rng.uniform(0, 1) for _ in boxes]
            for box in boxes:
                got, want = pfc.face_crop_box(box, size), self.ext.face_crop_box(box, size)
                self.assertEqual(None if got is None else got.as_tuple(), None if want is None else want.as_tuple())
                if want is not None:
                    margin = rng.randint(-2, 30)
                    self.assertEqual(pfc.grow_crop(got, margin, size).as_tuple(),
                                     self.ext.grow_crop(want, margin, size).as_tuple())
            got = pfc.plan_face_crops(boxes, scores, size, model="m.pt", confidence=0.3)
            want = self.ext.plan_face_crops(boxes, scores, size, model="m.pt", confidence=0.3)
            self.assertEqual(got.as_meta(), want.as_meta())
            self.assertEqual(got.status(), want.status())

    def test_owners_match_on_random_pieces(self):
        rng = np.random.default_rng(11)
        for _ in range(150):
            w, h = int(rng.integers(40, 120)), int(rng.integers(40, 120))
            faces = []
            for _ in range(int(rng.integers(1, 4))):
                side = int(rng.integers(8, 40))
                x0, y0 = int(rng.integers(0, w - 4)), int(rng.integers(0, h - 4))
                faces.append([x0, y0, x0 + side, y0 + side])
            plan = self.ext.plan_face_crops(faces, [0.5] * len(faces), (w, h), model="m", confidence=0.3)
            if not plan.crops:
                continue
            pieces, sources = [], []
            for face, crop in enumerate(plan.crops):
                for _ in range(int(rng.integers(0, 4))):
                    piece = np.zeros((h, w), dtype=bool)
                    x0 = int(rng.integers(crop.x0, crop.x1))
                    y0 = int(rng.integers(crop.y0, crop.y1))
                    piece[y0:min(crop.y1, y0 + int(rng.integers(1, 30))), x0:min(crop.x1, x0 + int(rng.integers(1, 30)))] = True
                    pieces.append(piece)
                    sources.append(face)
            crops = [self._crop(c) for c in plan.crops]
            self.assertEqual(pfc.assign_owners(pieces, sources, plan.boxes, crops),
                             self.ext.assign_owners(pieces, sources, plan.boxes, plan.crops))
            if _oracle_has_growth():
                got = pfc.growth_requests(pieces, sources, plan.boxes, crops, (w, h))
                want = self.ext.growth_requests(pieces, sources, plan.boxes, plan.crops, (w, h))
                self.assertEqual({k: (s, c.as_tuple()) for k, (s, c) in got.items()},
                                 {k: (s, c.as_tuple()) for k, (s, c) in want.items()})

    def test_growth_geometry_matches(self):
        if not _oracle_has_growth():
            self.skipTest("installed sam-extra has no crop growth (v0.34.0) — point AISTUDIO_FORGE_EXTENSION_DIR at a "
                          "v0.34.1 tree to compare touching_sides/grow_toward/enclosing_crop")
        rng = random.Random(3)
        for _ in range(500):
            size = (rng.randint(10, 300), rng.randint(10, 300))
            crop = self.ext.square_crop((rng.uniform(0, size[0]), rng.uniform(0, size[1]),
                                         rng.uniform(0, size[0]), rng.uniform(0, size[1])), size, min_side=1)
            if crop is None:
                continue
            sides = tuple(s for s in pfc.SIDES if rng.random() < 0.4)
            self.assertEqual(pfc.grow_toward(self._crop(crop), sides, size).as_tuple(),
                             self.ext.grow_toward(crop, sides, size).as_tuple())
            mask = np.zeros((size[1], size[0]), dtype=bool)
            for _ in range(rng.randint(0, 4)):
                x0, y0 = rng.randrange(size[0]), rng.randrange(size[1])
                mask[y0:y0 + rng.randint(1, 40), x0:x0 + rng.randint(1, 40)] = True
            self.assertEqual(pfc.touching_sides(mask, self._crop(crop)), self.ext.touching_sides(mask, crop))
            other = self.ext.square_crop((rng.uniform(0, size[0]), rng.uniform(0, size[1]),
                                          rng.uniform(0, size[0]), rng.uniform(0, size[1])), size, min_side=1)
            if other is not None:
                self.assertEqual(pfc.enclosing_crop([self._crop(crop), self._crop(other)], size).as_tuple(),
                                 self.ext.enclosing_crop([crop, other], size).as_tuple())


# --------------------------------------------------------------------------------------------------------------------


def _model_file(directory: Path, name: str = "face_yolov8n.pt") -> Path:
    path = directory / name
    path.write_bytes(b"fake")
    return path


def _pack_run(test, rgb, prompt, faces, scores, *, exclude="", dilation=0, hull=False, outline=0, user_mask=None,
              segmenter=None, mask_mode="Individual", batch_rgbs=None, face_crop=True):
    from comfy_custom_nodes.ai_studio_forge_parity import sam3_nodes

    segmenter = segmenter or PackSegmenter()
    images = batch_rgbs if batch_rgbs is not None else [rgb]
    per_image = faces if batch_rgbs is not None else [(faces, scores)]
    yolo = FakeUltralytics(per_image)
    lines: list[str] = []
    image = torch.from_numpy(np.stack(images)).float().div(255.0)
    with tempfile.TemporaryDirectory() as folder:
        model = _model_file(Path(folder))
        with mock.patch.dict(sys.modules, {"ultralytics": yolo.module}), \
                mock.patch.object(sam3_nodes, "_resolve_easy_node", return_value=segmenter), \
                mock.patch.object(sam3_nodes, "_log_face_crop", side_effect=lines.append):
            out = sam3_nodes.ForgeNeoSAM3Mask().segment(
                image, prompt=prompt, exclude_prompt=exclude, mask_mode=mask_mode,
                mask_source="intersection" if user_mask is not None else "generated",
                threshold=0.4, detection_limit=-1, convex_hull=hull, mask_dilation=dilation,
                mask_outline_px=outline, mask_blur=0, invert=False, save_artifacts=False, unload_after=False,
                sam3_model={"model": _FakeModel(), "device": "cpu", "segmentor": "image"},
                manual_mask=None if user_mask is None else torch.from_numpy(user_mask.astype(np.float32))[None],
                face_crop=face_crop, face_crop_model=str(model), face_crop_confidence=0.3,
            )
    selected, combined, individuals, overlay, boxes, out_scores, report = out
    return types.SimpleNamespace(
        selected=selected, combined=combined, individuals=individuals, overlay=overlay, boxes=boxes,
        scores=out_scores, report=json.loads(report), lines=lines, calls=segmenter.calls, yolo=yolo,
    )


def _oracle_run(rgb, prompt, faces, scores, *, exclude="", dilation=0, hull=False, outline=0, user_mask=None,
                processor=None):
    from PIL import Image

    core = _ext("core")
    fd = _ext("face_detect")
    processor = processor or OracleProcessor()
    stderr = io.StringIO()

    def detect(image, model, confidence, device):
        # what ultralytics hands both sides: float32 boxes and confidences
        return fd.FaceDetection(boxes=[[float(np.float32(v)) for v in f] for f in faces],
                                scores=[float(np.float32(s)) for s in scores],
                                model_path=f"C:/models/adetailer/{model}")

    with mock.patch.object(core, "resolve_checkpoint_path", return_value=None), \
            mock.patch.object(core, "_get_model_bundle", return_value=(None, processor)), \
            mock.patch.object(core, "detect_faces", side_effect=detect), \
            contextlib.redirect_stderr(stderr):
        result = core.run_sam3_on_pil(
            Image.fromarray(rgb), prompt, 0.4, "", "cpu", mask_dilation=dilation, mask_hull=hull,
            mask_outline_px=outline, exclude_prompt=exclude, user_mask=user_mask,
            face_crop=True, face_crop_model="face_yolov8n.pt", face_crop_confidence=0.3,
        )
    lines = [line[len("[-] SAM3: "):] for line in stderr.getvalue().splitlines()
             if line.startswith("[-] SAM3: face crop")]
    return types.SimpleNamespace(result=result, lines=lines, calls=processor.calls)


@requires_torch
@requires_extension
class OracleParityTests(unittest.TestCase):
    """Random scenes: the pack node == sam-extra ``run_sam3_on_pil`` with face crop on."""

    SCENES = 64

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        bind_torch(globals())
        cls.growth = _oracle_has_growth()

    def _assert_same(self, pack, oracle, label):
        result = oracle.result
        want_masks = [np.asarray(m) > 127 for m in result.masks]
        got = pack.individuals.numpy() > 0.5
        if want_masks:
            self.assertEqual(got.shape[0], len(want_masks), label)
            for index, want in enumerate(want_masks):
                np.testing.assert_array_equal(got[index], want, f"{label} mask {index}")
        else:
            self.assertFalse(got.any(), label)
        np.testing.assert_array_equal(pack.combined[0].numpy() > 0.5, np.asarray(result.mask) > 127, label)
        self.assertEqual(pack.boxes, result.boxes, label)
        self.assertEqual(pack.scores, result.scores, label)
        self.assertEqual(pack.report["face_crop"], result.face_crop.as_meta(), label)
        self.assertEqual(pack.report["face_crop_status"], result.face_crop.status(), label)
        self.assertEqual(pack.lines, oracle.lines, label)
        self.assertEqual(pack.calls, oracle.calls, label)

    def _run_both(self, index, rgb, faces, scores, opts, user_mask):
        kwargs = dict(exclude=opts["exclude_prompt"], dilation=opts["mask_dilation"], hull=opts["mask_hull"],
                      outline=opts["mask_outline_px"], user_mask=user_mask)
        oracle = _oracle_run(rgb, opts["prompt"], faces, scores, **kwargs)
        return oracle, kwargs

    def test_random_scenes(self):
        grown_scenes = 0
        for index, rgb, faces, scores, opts, user_mask in _scenes(self.SCENES):
            label = f"scene {index} {opts} manual={user_mask is not None}"
            with self.subTest(scene=index):
                oracle, kwargs = self._run_both(index, rgb, faces, scores, opts, user_mask)
                if self.growth:
                    pack = _pack_run(self, rgb, opts["prompt"], faces, scores, **kwargs)
                    self._assert_same(pack, oracle, label)
                    grown_scenes += bool(pack.report["face_crop"]["faces"] and any(
                        "grown" in face for face in pack.report["face_crop"]["faces"]))
                    continue
                with mock.patch.object(pfc, "GROW_ROUNDS", 0):
                    pack = _pack_run(self, rgb, opts["prompt"], faces, scores, **kwargs)
                self._assert_same(pack, oracle, label + " (GROW_ROUNDS 0 == v0.34.0)")
                grown = _pack_run(self, rgb, opts["prompt"], faces, scores, **kwargs)
                if any("grown" in face for face in grown.report["face_crop"]["faces"]):
                    grown_scenes += 1
                    if not opts["exclude_prompt"] and user_mask is None:
                        # growth replaces a cut result by the grown one: without Exclude the union only grows here
                        self.assertTrue(np.all(grown.combined.numpy() >= pack.combined.numpy()), label)
                else:
                    self._assert_same(grown, oracle, label + " (no growth == v0.34.0)")
        self.assertGreater(grown_scenes, 5, "the scenes must exercise the crop growth")

    def test_hair_growth_scene(self):
        rgb = np.full((SIZE[1], SIZE[0], 3), BACKGROUND, dtype=np.uint8)
        face = (100, 40, 140, 80)
        _draw_face(rgb, face)
        rgb[30:115, 140:150] = COLORS["hair"]
        oracle = _oracle_run(rgb, "hair, eye", [face], [0.8])
        if not self.growth:
            with mock.patch.object(pfc, "GROW_ROUNDS", 0):
                pack = _pack_run(self, rgb, "hair, eye", [face], [0.8])
            self._assert_same(pack, oracle, "pony (no growth)")
            self.skipTest("installed sam-extra has no crop growth (v0.34.0): compared with GROW_ROUNDS 0 only")
        pack = _pack_run(self, rgb, "hair, eye", [face], [0.8])
        self._assert_same(pack, oracle, "pony")


# --------------------------------------------------------------------------------------------------------------------


PONY_FACE = (100, 40, 140, 80)       # crop (88, 24, 152, 88), side 64


def _pony_image(bottom):
    rgb = np.full((SIZE[1], SIZE[0], 3), BACKGROUND, dtype=np.uint8)
    _draw_face(rgb, PONY_FACE)
    rgb[30:bottom, 140:150] = COLORS["hair"]
    return rgb


@requires_torch
class PackGrowthTests(unittest.TestCase):
    """sam-extra v0.34.1's growth cases on the pack alone (its tests/test_sam3_face_crop.py FaceCropGrowthTests)."""

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        bind_torch(globals())

    def test_hair_cut_at_the_crop_edge_is_found_again_in_a_grown_crop(self):
        rgb = _pony_image(bottom=115)
        crop = pfc.face_crop_box(PONY_FACE, SIZE)
        self.assertEqual(crop, pfc.CropBox(88, 24, 152, 88))
        grown = pfc.CropBox(72, 24, 168, 120)
        run = _pack_run(self, rgb, "hair, eye", [PONY_FACE], [0.8])
        self.assertEqual([prompt for _, prompt in run.calls], ["hair", "eye", "hair"], "only the cut token again")
        self.assertEqual([d[-8:] for d, _ in run.calls], ["(64, 64)", "(64, 64)", "(96, 96)"])
        np.testing.assert_array_equal(run.combined[0].numpy() > 0.5, _color(rgb, "hair") | _color(rgb, "eye"))
        face = run.report["face_crop"]["faces"][0]
        self.assertEqual(face["crop"], list(crop.as_tuple()))
        self.assertEqual(face["grown"], [{"prompt": "hair", "crop": list(grown.as_tuple()), "rounds": 1,
                                          "sides": ["bottom"]}])
        self.assertIn("face crop — face 1 'hair' reached the crop edge (bottom); SAM3 again on 96x96@72,24 "
                      "(round 1/2)", run.lines)
        self.assertIn("grown where the mask reached the crop edge: face 1 'hair' → 96x96@72,24 (1 round)",
                      run.report["face_crop_status"])
        self.assertIn([140.0, 30.0, 150.0, 115.0], run.boxes)
        overlay = np.round(run.overlay[0].numpy() * 255.0).astype(int)
        np.testing.assert_array_equal(overlay[grown.y1 - 1, grown.x0 + 5], [255, 190, 40], "grown crop frame")
        np.testing.assert_array_equal(overlay[crop.y0, crop.x0 + 30], [255, 190, 40], "face crop frame")

    def test_growth_rounds_are_bounded_and_patchable(self):
        rgb = _pony_image(bottom=SIZE[1])
        run = _pack_run(self, rgb, "hair", [PONY_FACE], [0.8])
        self.assertEqual(len(run.calls), 3, f"cap {pfc.GROW_ROUNDS}")
        last = pfc.CropBox(48, 24, 192, 168)
        self.assertEqual(run.report["face_crop"]["faces"][0]["grown"][0]["crop"], list(last.as_tuple()))
        hair = _color(rgb, "hair")
        within = np.zeros_like(hair)
        within[last.y0:last.y1, last.x0:last.x1] = True
        np.testing.assert_array_equal(run.combined[0].numpy() > 0.5, hair & within)
        with mock.patch.object(pfc, "GROW_ROUNDS", 3):
            more = _pack_run(self, rgb, "hair", [PONY_FACE], [0.8])
        self.assertEqual(len(more.calls), 4)
        np.testing.assert_array_equal(more.combined[0].numpy() > 0.5, hair)

    def test_neighbours_cut_pieces_do_not_grow_a_face(self):
        face_a, face_b = (40, 70, 80, 110), (110, 70, 150, 110)
        rgb = np.full((SIZE[1], SIZE[0], 3), BACKGROUND, dtype=np.uint8)
        _draw_face(rgb, face_a)
        _draw_face(rgb, face_b)
        rgb[60:70, 40:80] = COLORS["hair"]
        rgb[100:170, 60:70] = COLORS["hair"]
        rgb[60:70, 110:150] = COLORS["hair"]
        rgb[100:170, 120:130] = COLORS["hair"]
        hair = _color(rgb, "hair")
        own_a = hair.copy()
        own_a[:, 95:] = False
        with mock.patch.object(pfc, "GROW_ROUNDS", 3):
            run = _pack_run(self, rgb, "hair", [face_b, face_a], [0.8, 0.8])
        grown = [(i, f["grown"][0]["crop"], f["grown"][0]["rounds"]) for i, f in enumerate(run.report["face_crop"]["faces"])]
        self.assertEqual(grown, [(0, [0, 48, 144, 192], 2), (1, [58, 48, 202, 192], 2)])
        self.assertEqual(len(run.calls), 6)
        individuals = run.individuals.numpy() > 0.5
        self.assertEqual(individuals.shape[0], 2)
        np.testing.assert_array_equal(individuals[0], own_a)
        np.testing.assert_array_equal(individuals[1], hair & ~own_a)

    def test_a_grown_crop_that_finds_nothing_keeps_the_face_crop_result(self):
        rgb = _pony_image(bottom=115)
        segmenter = PackSegmenter(keep=lambda crop, prompt: crop.shape[0] <= 64)
        run = _pack_run(self, rgb, "hair", [PONY_FACE], [0.8], segmenter=segmenter)
        self.assertEqual(len(run.calls), 2, "no second growth after an empty grown run")
        self.assertNotIn("grown", run.report["face_crop"]["faces"][0])
        self.assertTrue(any("nothing found in the grown crop 96x96@72,24" in line for line in run.lines))
        crop = pfc.face_crop_box(PONY_FACE, SIZE)
        within = np.zeros(rgb.shape[:2], dtype=bool)
        within[crop.y0:crop.y1, crop.x0:crop.x1] = True
        np.testing.assert_array_equal(run.combined[0].numpy() > 0.5, _color(rgb, "hair") & within)


@requires_torch
class PackNodeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        bind_torch(globals())

    def setUp(self):
        from comfy_custom_nodes.ai_studio_forge_parity import sam3_nodes

        self.nodes = sam3_nodes
        self.rgb = np.full((SIZE[1], SIZE[0], 3), BACKGROUND, dtype=np.uint8)
        _draw_face(self.rgb, (40, 70, 80, 110))
        _draw_face(self.rgb, (170, 70, 210, 110))
        self.rgb[5:12, 120:127] = COLORS["eye"]   # outside the faces: never found with face crop

    def test_inputs_are_optional_with_the_extension_defaults_and_the_detector_combo(self):
        optional = self.nodes.ForgeNeoSAM3Mask.INPUT_TYPES()["optional"]
        self.assertEqual(optional["face_crop"][1]["default"], False)
        self.assertEqual(optional["face_crop_model"][1]["default"], "face_yolov8n.pt")
        self.assertEqual(optional["face_crop_confidence"][1]["default"], 0.3)
        self.assertEqual((optional["face_crop_confidence"][1]["min"], optional["face_crop_confidence"][1]["max"]),
                         (0.0, 1.0))
        required = self.nodes.ForgeNeoSAM3Mask.INPUT_TYPES()["required"]
        self.assertFalse({"face_crop", "face_crop_model", "face_crop_confidence"} & set(required))
        for found, status in ((True, pfd.DETECTOR_READY), (False, pfd.DETECTOR_MISSING)):
            with self.subTest(found=found), mock.patch.object(
                    pfd.importlib.util, "find_spec", return_value=object() if found else None):
                spec = self.nodes.ForgeNeoSAM3Mask.INPUT_TYPES()["optional"][pfd.DETECTOR_INPUT]
                self.assertEqual(spec[0], [status])
                self.assertEqual(spec[1]["default"], status)

    def test_face_crop_off_is_the_run_without_it(self):
        def run(**extra):
            segmenter = PackSegmenter()
            image = torch.from_numpy(self.rgb[None]).float().div(255.0)
            with mock.patch.object(self.nodes, "_resolve_easy_node", return_value=segmenter), \
                    mock.patch.object(self.nodes, "_plan_face_crops", side_effect=AssertionError("detector")):
                out = self.nodes.ForgeNeoSAM3Mask().segment(
                    image, prompt="eye / mouth", exclude_prompt="skin", mask_blur=2, save_artifacts=False,
                    sam3_model={"model": _FakeModel(), "device": "cpu", "segmentor": "image"}, **extra)
            return out, segmenter.calls

        (base, calls), (off, off_calls) = run(), run(face_crop=False, face_crop_model="x.pt")
        self.assertEqual(calls, off_calls)
        for a, b in zip(base[:4], off[:4]):
            self.assertTrue(torch.equal(a, b))
        self.assertEqual(base[4:], off[4:])
        self.assertNotIn("face_crop", json.loads(off[6]))

    def test_face_crop_finds_only_inside_the_faces_in_individual_order(self):
        run = _pack_run(self, self.rgb, "eye / mouth", [[170, 70, 210, 110], [40, 70, 80, 110]], [0.7, 0.9])
        crop_a = pfc.face_crop_box((40, 70, 80, 110), SIZE)
        crop_b = pfc.face_crop_box((170, 70, 210, 110), SIZE)
        eye, mouth = _color(self.rgb, "eye"), _color(self.rgb, "mouth")

        def inside(crop):
            out = np.zeros(eye.shape, dtype=bool)
            out[crop.y0:crop.y1, crop.x0:crop.x1] = True
            return out

        want = [eye & inside(crop_a), eye & inside(crop_b), mouth & inside(crop_a), mouth & inside(crop_b)]
        got = run.individuals.numpy() > 0.5
        self.assertEqual(got.shape[0], 4)
        for index, mask in enumerate(want):
            np.testing.assert_array_equal(got[index], mask)
        self.assertFalse((run.combined[0].numpy() > 0.5)[5:12, 120:127].any())
        self.assertEqual(run.report["face_crop"]["model"], "face_yolov8n.pt")
        self.assertEqual([f["crop"] for f in run.report["face_crop"]["faces"]],
                         [list(crop_a.as_tuple()), list(crop_b.as_tuple())])
        self.assertTrue(run.lines[0].startswith("face crop — 2 face(s) by face_yolov8n.pt (confidence 0.30)"))
        self.assertEqual(len(run.yolo.loads), 1)
        self.assertTrue(Path(run.yolo.loads[0]).is_absolute())
        self.assertEqual(run.yolo.calls, [(SIZE, 0.3, "cpu", False)])

    def test_no_face_never_loads_sam3_and_masks_nothing(self):
        image = torch.from_numpy(self.rgb[None]).float().div(255.0)
        yolo = FakeUltralytics([([], [])])
        with tempfile.TemporaryDirectory() as folder:
            model = _model_file(Path(folder))
            with mock.patch.dict(sys.modules, {"ultralytics": yolo.module}), \
                    mock.patch.object(self.nodes, "_acquire_sam3", side_effect=AssertionError("SAM3 loaded")), \
                    mock.patch.object(self.nodes, "_resolve_easy_node", side_effect=AssertionError("provider")):
                out = self.nodes.ForgeNeoSAM3Mask().segment(
                    image, prompt="eye", mask_blur=0, save_artifacts=False, device="cpu", precision="fp32",
                    face_crop=True, face_crop_model=str(model))
        self.assertFalse(bool((out[1] > 0).any()))
        report = json.loads(out[6])
        self.assertEqual(report["status"], "no_detection")
        self.assertEqual(report["face_crop"]["faces"], [])
        self.assertEqual(report["face_crop_status"],
                         "face crop — no faces found by face_yolov8n.pt (confidence 0.30); nothing to mask.")
        self.assertFalse(report["model_loaded_by_node"])
        self.assertEqual(yolo.calls, [(SIZE, 0.3, "cpu", False)], "device rule: SAM3 on cpu -> 'cpu'")

    def test_batch_runs_each_image_with_its_faces(self):
        second = np.full_like(self.rgb, BACKGROUND)
        _draw_face(second, (100, 40, 140, 80))
        faces = [([[170, 70, 210, 110], [40, 70, 80, 110]], [0.7, 0.9]), ([[100, 40, 140, 80]], [0.8])]
        run = _pack_run(self, None, "eye", faces, None, batch_rgbs=[self.rgb, second])
        self.assertEqual(run.individuals.shape[0], 4, "two passes x two images")
        singles = [_pack_run(self, rgb, "eye", f, s) for rgb, (f, s) in zip([self.rgb, second], faces)]
        passes = run.individuals.reshape(2, 2, SIZE[1], SIZE[0]).numpy() > 0.5
        first = singles[0].individuals.numpy() > 0.5
        np.testing.assert_array_equal(passes[:, 0], first)
        np.testing.assert_array_equal(passes[0, 1], singles[1].individuals.numpy()[0] > 0.5)
        self.assertFalse(passes[1, 1].any(), "the second image has one face: its second pass is empty")
        self.assertEqual(run.boxes, singles[0].boxes + singles[1].boxes)
        self.assertEqual([len(meta["faces"]) for meta in run.report["face_crop"]], [2, 1])
        self.assertEqual(len(run.yolo.loads), 1, "one detector load for the batch")

    def test_combined_mode_and_exclude_inside_the_crops(self):
        run = _pack_run(self, self.rgb, "skin", [[40, 70, 80, 110]], [0.9], exclude="eye, mouth",
                        mask_mode="Combined")
        crop = pfc.face_crop_box((40, 70, 80, 110), SIZE)
        within = np.zeros(self.rgb.shape[:2], dtype=bool)
        within[crop.y0:crop.y1, crop.x0:crop.x1] = True
        want = _color(self.rgb, "skin") & within
        np.testing.assert_array_equal(run.selected[0].numpy() > 0.5, want)
        self.assertEqual([prompt for _, prompt in run.calls], ["skin", "eye", "mouth"])


class DetectorTests(unittest.TestCase):
    """``sam3_face_detect`` without torch: the ultralytics call, model lookup and messages (fake ultralytics)."""

    def test_constants_match_the_app_and_the_extension_defaults(self):
        from core import sam3_args
        from core import sam3_face_crop as app

        self.assertEqual(pfd.DEFAULT_MODEL, sam3_args.FACE_CROP_DEFAULT_MODEL)
        self.assertEqual(pfd.DEFAULT_CONFIDENCE, sam3_args.FACE_CROP_DEFAULT_CONFIDENCE)
        self.assertEqual(pfd.MODEL_SUFFIX, app.MODEL_SUFFIX)
        self.assertEqual(pfd.DETECTOR_INPUT, app.COMFY_DETECTOR_INPUT)
        self.assertEqual(pfd.DETECTOR_READY, app.COMFY_DETECTOR_READY)
        self.assertEqual(app.COMFY_INPUTS, ("face_crop", "face_crop_model", "face_crop_confidence"))

    def test_device_rule_is_the_extensions(self):
        self.assertEqual(pfd.ultralytics_device("cpu"), "cpu")
        self.assertEqual(pfd.ultralytics_device(" CPU "), "cpu")
        for device in ("cuda", "cuda:1", "mps", "", None):
            self.assertEqual(pfd.ultralytics_device(device), "")

    def test_absolute_path_name_lookup_and_world_models(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / "ultralytics" / "bbox").mkdir(parents=True)
            (root / "adetailer" / "sub").mkdir(parents=True)
            bbox = _model_file(root / "ultralytics" / "bbox", "face_yolov8m.pt")
            nested = _model_file(root / "adetailer" / "sub", "Face_Yolov8n.PT")
            world = _model_file(root / "adetailer", "yolov8x-worldv2.pt")
            stub = types.SimpleNamespace(models_dir=str(root),
                                         get_folder_paths=mock.Mock(side_effect=KeyError("not registered")))
            with mock.patch.dict(sys.modules, {"folder_paths": stub}):
                self.assertEqual(pfd.model_dirs(),
                                 [root / "ultralytics" / "bbox", root / "ultralytics", root / "adetailer"])
                self.assertEqual(pfd.resolve_face_model("face_yolov8m.pt"), bbox)
                self.assertEqual(pfd.resolve_face_model("bbox/face_yolov8m.pt"), bbox)
                self.assertEqual(pfd.resolve_face_model("face_yolov8n.pt"), nested)
                self.assertEqual(pfd.resolve_face_model(""), nested, "empty = the default model")
                self.assertIsNone(pfd.resolve_face_model("yolov8x-worldv2.pt"))
                self.assertIsNone(pfd.resolve_face_model(str(world)))
                self.assertEqual(pfd.resolve_face_model(str(bbox)), bbox)
                message = pfd.missing_model_message("missing.pt")
                self.assertIn("'missing.pt' not found", message)
                self.assertIn(str(root / "adetailer"), message)
                with self.assertRaisesRegex(pfd.FaceDetectorUnavailable, "never downloaded"):
                    pfd.detect_faces(object(), "missing.pt", 0.3, "cpu")

    def test_registered_folders_come_first(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            impact = root / "impact"
            impact.mkdir()
            stub = types.SimpleNamespace(models_dir=str(root / "models"),
                                         get_folder_paths=lambda key: [str(impact)] if key == "ultralytics_bbox" else [])
            with mock.patch.dict(sys.modules, {"folder_paths": stub}):
                self.assertEqual(pfd.model_dirs(), [impact])

    def test_ultralytics_call_and_missing_package(self):
        with tempfile.TemporaryDirectory() as folder:
            model = _model_file(Path(folder))
            yolo = FakeUltralytics([([[1, 2, 30, 40], [5, 6, 7, 8]], [0.9, 0.4])])
            image = types.SimpleNamespace(size=(64, 48))
            with mock.patch.dict(sys.modules, {"ultralytics": yolo.module}):
                found = pfd.detect_faces(image, str(model), 0.45, "cuda")
            self.assertEqual(yolo.loads, [str(model)])
            self.assertEqual(yolo.calls, [((64, 48), 0.45, "", False)])
            self.assertEqual(found.boxes, [[1.0, 2.0, 30.0, 40.0], [5.0, 6.0, 7.0, 8.0]])
            self.assertEqual(found.scores, [float(np.float32(0.9)), float(np.float32(0.4))])
            self.assertEqual(found.model_path, str(model))
            with mock.patch.dict(sys.modules, {"ultralytics": None}):
                with self.assertRaisesRegex(pfd.FaceDetectorUnavailable, "ultralytics' package is not installed"):
                    pfd.detect_faces(image, str(model), 0.3, "cpu")

    def test_detector_status_does_not_import_ultralytics(self):
        with mock.patch.object(pfd.importlib.util, "find_spec", return_value=None) as find:
            self.assertEqual(pfd.detector_status(), pfd.DETECTOR_MISSING)
        find.assert_called_once_with("ultralytics")
        with mock.patch.object(pfd.importlib.util, "find_spec", return_value=object()):
            self.assertEqual(pfd.detector_status(), pfd.DETECTOR_READY)
            self.assertEqual(pfd.detector_input()[0], [pfd.DETECTOR_READY])


if __name__ == "__main__":
    unittest.main()
