"""SAM3 "search inside faces only" (face crop) for ``ForgeNeoSAM3Mask``: the pure geometry (pack 1.8.0).

This is sam-extra's ``sam3ext/face_crop.py`` (v0.34.0, with the crop growth of v0.34.1 as fixed before its release,
sam-extra commit b815f68) on ComfyUI. The code is sam-extra's own, licensed to this pack under MIT by its author
(``THIRD_PARTY_NOTICES.md`` "Code shared with the sam-extra Forge extension"); only the docstrings are rewritten.
NumPy (and Pillow for a mask of another size) only: no torch, no ultralytics, no ComfyUI. The flow is
``sam3_face_crop_run.run_face_crop`` (sam-extra's face-crop branch of ``run_sam3_on_pil``):

1. A detector box (``sam3_face_detect``) is widened (20 % left/right/bottom, 40 % top: forehead and hair), made a
   square around its centre and pushed inside the image (``face_crop_box``); only an axis longer than the image is cut.
2. SAM3 runs on each crop as it is. The crop is never resized (the SAM3 processor scales it to its 1008 working size
   like a whole image and returns masks and boxes at crop size), so crop -> image is a translation (``box_to_image``).
3. ``paste_mask`` puts a crop mask back at the same pixels of an image-sized bool mask.
4. Overlapping crops see a neighbour's eyes or hair: every detected piece gets an owner face (``assign_owner``: the
   nearest face centre in face sizes). A piece cut at an inner crop edge (not an image border) is grouped with the
   pieces of other crops that continue it and the group is judged by its joint centroid (``assign_owners``). No piece
   is dropped, so the Combined mask is the union.
5. Two detector boxes on one face (ultralytics NMS keeps both below IoU 0.7) would split a face into two passes:
   ``dedupe_faces`` keeps the higher-scoring one of a ``same_face`` pair.
6. A part that leaves the face (hair) is cut with a straight edge at an inner crop edge. Per detected token, when the
   pieces that face owns reach an inner edge of its crop far enough (``touching_sides``) the crop grows toward those
   sides (``grow_toward``: square, pushed inside) and only that token runs again (``growth_requests``, at most
   ``GROW_ROUNDS`` times per face and token). Of a grown crop's result only the pieces that overlap the earlier result
   are used (``continuing_pieces``: a piece that came into view only in the grown crop, cut off from the face's mask,
   is dropped). Without contact the SAM3 calls and the masks are those without growth.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Collection, Sequence

import numpy as np

# Face box widening (fractions of the box width/height). face_yolov8 boxes end near the eyebrows and the chin, so the
# top grows more for the forehead, hair and ears.
EXPAND_SIDE = 0.20
EXPAND_TOP = 0.40
EXPAND_BOTTOM = 0.20
# Smallest crop side in pixels (the image size when the image is smaller): a distant face still gets some context.
MIN_CROP_SIDE = 32
# Two detector boxes are one face at this IoU (or when one box's centre lies inside the other).
SAME_FACE_IOU = 0.3
# Crop growth (step 6): one side GROW_FACTOR times longer per round, at most GROW_ROUNDS more SAM3 runs per face x token.
GROW_FACTOR = 1.5
GROW_ROUNDS = 2
# "Touching": the mask in the TOUCH_BAND px band inside an inner crop edge, projected along that edge, covers at least
# max(TOUCH_MIN_PX, edge length x TOUCH_MIN_FRACTION) positions. Cut hair runs along the edge; eyes and mouths stay
# inside the crop (chosen from sam-extra v0.34.0's measured masks).
TOUCH_BAND = 2
TOUCH_MIN_PX = 4
TOUCH_MIN_FRACTION = 0.02
# Side names, in the order they are returned.
SIDES = ("left", "top", "right", "bottom")

Box = Sequence[float]   # (x0, y0, x1, y1) in image pixels; x1 and y1 are exclusive


@dataclass(frozen=True)
class CropBox:
    """An integer crop ``[x0, x1) x [y0, y1)`` inside the image (a PIL ``Image.crop`` box)."""

    x0: int
    y0: int
    x1: int
    y1: int

    @property
    def width(self) -> int:
        return self.x1 - self.x0

    @property
    def height(self) -> int:
        return self.y1 - self.y0

    def as_tuple(self) -> tuple[int, int, int, int]:
        return (self.x0, self.y0, self.x1, self.y1)

    def contains(self, x: float, y: float) -> bool:
        return self.x0 <= x < self.x1 and self.y0 <= y < self.y1


def _finite_box(box: Box) -> tuple[float, float, float, float] | None:
    try:
        x0, y0, x1, y1 = (float(v) for v in box)
    except (TypeError, ValueError):
        return None
    if not all(math.isfinite(v) for v in (x0, y0, x1, y1)):
        return None
    return x0, y0, x1, y1


def clip_box(box: Box, image_size: tuple[int, int]) -> tuple[float, float, float, float] | None:
    """The box cut to the image (``image_size`` = (width, height), PIL order); None when empty or not numeric."""
    finite = _finite_box(box)
    if finite is None:
        return None
    width, height = image_size
    x0, y0, x1, y1 = finite
    x0, x1 = max(0.0, min(x0, x1)), min(float(width), max(x0, x1))
    y0, y1 = max(0.0, min(y0, y1)), min(float(height), max(y0, y1))
    if x1 - x0 <= 0 or y1 - y0 <= 0:
        return None
    return x0, y0, x1, y1


def expand_box(
    box: Box,
    *,
    side: float = EXPAND_SIDE,
    top: float = EXPAND_TOP,
    bottom: float = EXPAND_BOTTOM,
) -> tuple[float, float, float, float]:
    """Widen a face box: ``side`` of the width left and right, ``top``/``bottom`` of the height above/below."""
    x0, y0, x1, y1 = (float(v) for v in box)
    w, h = x1 - x0, y1 - y0
    return (x0 - w * side, y0 - h * top, x1 + w * side, y1 + h * bottom)


def _fit_axis(start: int, length: int, limit: int) -> tuple[int, int]:
    """Push a span of ``length`` inside ``[0, limit)`` keeping its length; the whole axis when it is longer."""
    if length >= limit:
        return 0, limit
    start = min(max(start, 0), limit - length)
    return start, start + length


def square_crop(
    box: Box,
    image_size: tuple[int, int],
    *,
    min_side: int = MIN_CROP_SIDE,
) -> CropBox | None:
    """The integer square around ``box`` (same centre), pushed inside the image.

    Side = ceil(max(width, height, ``min_side``)). A crop past the border slides back inside (keeping the square and
    its content); only an axis longer than the image is cut to the image.
    """
    finite = _finite_box(box)
    if finite is None:
        return None
    width, height = (int(v) for v in image_size)
    if width <= 0 or height <= 0:
        return None
    x0, y0, x1, y1 = finite
    side = int(math.ceil(max(x1 - x0, y1 - y0, float(min_side), 1.0) - 1e-9))
    cx, cy = (x0 + x1) / 2.0, (y0 + y1) / 2.0
    left = int(math.floor(cx - side / 2.0 + 1e-9))
    top = int(math.floor(cy - side / 2.0 + 1e-9))
    left, right = _fit_axis(left, side, width)
    top, bottom = _fit_axis(top, side, height)
    return CropBox(left, top, right, bottom)


def face_crop_box(box: Box, image_size: tuple[int, int]) -> CropBox | None:
    """One detector box -> the crop SAM3 sees: cut to the image, widen, square, push inside. None when unusable."""
    clipped = clip_box(box, image_size)
    if clipped is None:
        return None
    return square_crop(expand_box(clipped), image_size)


def grow_crop(crop: CropBox, margin: int, image_size: tuple[int, int]) -> CropBox:
    """The crop widened by ``margin`` px on every side (square, same centre, pushed inside): where Exclude is searched.

    Outline and Dilation spread a mask up to ``margin`` px (chessboard distance) past the crop; this area always covers
    the crop +- ``margin`` cut to the image (pushed at a border, it covers that much more on the other side). The crop
    itself when ``margin`` <= 0.
    """
    margin = int(margin)
    if margin <= 0:
        return crop
    grown = square_crop(
        (crop.x0 - margin, crop.y0 - margin, crop.x1 + margin, crop.y1 + margin), image_size, min_side=1
    )
    return grown if grown is not None else crop


def box_to_image(box: Box, crop: CropBox) -> list[float]:
    """A box in crop coordinates (SAM3 output) -> image coordinates: a translation (the crop is never resized)."""
    x0, y0, x1, y1 = (float(v) for v in box)
    return [x0 + crop.x0, y0 + crop.y0, x1 + crop.x0, y1 + crop.y0]


def box_to_crop(box: Box, crop: CropBox) -> list[float]:
    """A box in image coordinates -> crop coordinates (the inverse of ``box_to_image``)."""
    x0, y0, x1, y1 = (float(v) for v in box)
    return [x0 - crop.x0, y0 - crop.y0, x1 - crop.x0, y1 - crop.y0]


def paste_mask(crop_mask: np.ndarray, crop: CropBox, image_shape: tuple[int, int]) -> np.ndarray:
    """A crop-sized mask at the crop's place in an image-sized bool mask (``image_shape`` = (height, width)).

    False outside the crop. A mask of another size (another processor) is fitted to the crop with nearest neighbour.
    """
    height, width = (int(v) for v in image_shape)
    if not (0 <= crop.x0 < crop.x1 <= width and 0 <= crop.y0 < crop.y1 <= height):
        raise ValueError(f"face crop {crop.as_tuple()} is outside the image {width}x{height}")
    piece = np.asarray(crop_mask).astype(bool)
    if piece.ndim != 2:
        raise ValueError(f"face crop mask must be 2-D, got shape {piece.shape}")
    if piece.shape != (crop.height, crop.width):
        from PIL import Image

        resized = Image.fromarray(piece.astype(np.uint8) * 255, mode="L").resize(
            (crop.width, crop.height), Image.NEAREST
        )
        piece = np.asarray(resized) > 127
    full = np.zeros((height, width), dtype=bool)
    full[crop.y0:crop.y1, crop.x0:crop.x1] = piece
    return full


def mask_centroid(mask: np.ndarray) -> tuple[float, float] | None:
    """The centroid (x, y) of the mask pixels, at pixel centres (+0.5). None for an empty mask."""
    ys, xs = np.nonzero(np.asarray(mask))
    if ys.size == 0:
        return None
    return float(xs.mean()) + 0.5, float(ys.mean()) + 0.5


def assign_owner(
    point: tuple[float, float],
    face_boxes: Sequence[Box],
    crops: Sequence[CropBox],
    source: int | Collection[int],
) -> int:
    """The owner face of a piece found in crop ``source`` (centroid ``point``).

    Candidates: the faces whose crop holds the centroid, plus ``source`` (the piece lies in that crop). The winner has
    the smallest distance to its face box centre in face sizes (the box's longer side); ties go to ``source``, then
    to the lower number. ``source`` is several crop numbers for a group of pieces across crops (``assign_owners``).
    """
    sources = {int(source)} if isinstance(source, (int, np.integer)) else {int(s) for s in source}
    px, py = point
    candidates = sources | {i for i, crop in enumerate(crops) if crop.contains(px, py)}

    def score(index: int) -> tuple[float, bool, int]:
        x0, y0, x1, y1 = (float(v) for v in face_boxes[index])
        size = max(x1 - x0, y1 - y0, 1.0)
        dist = math.hypot(px - (x0 + x1) / 2.0, py - (y0 + y1) / 2.0) / size
        return (dist, index not in sources, index)

    return min(sorted(candidates), key=score)


def _spread(line: np.ndarray) -> np.ndarray:
    """Widen by one along the edge (8-connected: a diagonal neighbour also touches)."""
    out = line.copy()
    out[1:] |= line[:-1]
    out[:-1] |= line[1:]
    return out


def edge_contact(piece: np.ndarray, crop: CropBox) -> tuple[tuple[int, int, int, int], np.ndarray] | None:
    """Where a piece cut at an inner crop edge would continue: the one-pixel line just outside the crop.

    ``piece`` is an image-coordinate bool mask (``paste_mask``). Inner edges are the crop sides that are not image
    borders. Returns ((y0, y1, x0, x1), the bool mask of that range), or None when no inner edge is touched.
    """
    height, width = piece.shape
    lines: list[tuple[int, int, int, int, np.ndarray]] = []   # (y0, y1, x0, x1, the line outside)
    if crop.x0 > 0:
        line = piece[crop.y0:crop.y1, crop.x0]
        if line.any():
            lines.append((crop.y0, crop.y1, crop.x0 - 1, crop.x0, _spread(line)[:, None]))
    if crop.x1 < width:
        line = piece[crop.y0:crop.y1, crop.x1 - 1]
        if line.any():
            lines.append((crop.y0, crop.y1, crop.x1, crop.x1 + 1, _spread(line)[:, None]))
    if crop.y0 > 0:
        line = piece[crop.y0, crop.x0:crop.x1]
        if line.any():
            lines.append((crop.y0 - 1, crop.y0, crop.x0, crop.x1, _spread(line)[None, :]))
    if crop.y1 < height:
        line = piece[crop.y1 - 1, crop.x0:crop.x1]
        if line.any():
            lines.append((crop.y1, crop.y1 + 1, crop.x0, crop.x1, _spread(line)[None, :]))
    if not lines:
        return None
    y0 = min(item[0] for item in lines)
    y1 = max(item[1] for item in lines)
    x0 = min(item[2] for item in lines)
    x1 = max(item[3] for item in lines)
    contact = np.zeros((y1 - y0, x1 - x0), dtype=bool)
    for ly0, ly1, lx0, lx1, values in lines:
        contact[ly0 - y0:ly1 - y0, lx0 - x0:lx1 - x0] |= values
    return (y0, y1, x0, x1), contact


def assign_owners(
    pieces: Sequence[np.ndarray],
    sources: Sequence[int],
    face_boxes: Sequence[Box],
    crops: Sequence[CropBox],
) -> list[int]:
    """Owner faces of the pieces one token found (image coordinates; ``sources[i]`` = the crop of piece i).

    A piece cut at an inner crop edge joins the pieces of other crops that cross the line just past it
    (``edge_contact``): the rest of an object cut by one crop is seen by the neighbouring crop. A group is judged by the
    centroid of its union, a single piece by its own (``assign_owner``). Pieces of one crop are never joined (SAM3
    reported them separately).
    """
    count = len(pieces)
    parent = list(range(count))

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    for i in range(count):
        contact = edge_contact(pieces[i], crops[sources[i]])
        if contact is None:
            continue
        (y0, y1, x0, x1), where = contact
        for j in range(count):
            if sources[j] == sources[i] or find(i) == find(j):
                continue
            if np.any(pieces[j][y0:y1, x0:x1] & where):
                parent[find(j)] = find(i)

    owners = [0] * count
    clusters: dict[int, list[int]] = {}
    for index in range(count):
        clusters.setdefault(find(index), []).append(index)
    for members in clusters.values():
        if len(members) == 1:
            joined = pieces[members[0]]
        else:
            joined = np.any(np.stack([pieces[i] for i in members], axis=0), axis=0)
        point = mask_centroid(joined)
        if point is None:
            continue
        owner = assign_owner(point, face_boxes, crops, [sources[i] for i in members])
        for index in members:
            owners[index] = owner
    return owners


def touching_sides(
    mask: np.ndarray,
    crop: CropBox,
    *,
    band: int = TOUCH_BAND,
    min_px: int = TOUCH_MIN_PX,
    min_fraction: float = TOUCH_MIN_FRACTION,
) -> tuple[str, ...]:
    """The inner crop edges an image-coordinate bool mask reaches far enough (``SIDES`` order; empty when none).

    Inner edges are the crop sides that are not image borders. Per side, the mask in the ``band`` px band inside the
    edge is projected along it; the side counts when the covered length is at least
    ``max(min_px, ceil(side length x min_fraction))``, so a few stray pixels never grow a crop. A whole-image crop
    never touches.
    """
    piece = np.asarray(mask, dtype=bool)
    height, width = piece.shape
    b = max(1, int(band))
    strips = []   # (side, the band projected along the side, side length)
    if crop.x0 > 0:
        strips.append(("left", piece[crop.y0:crop.y1, crop.x0:min(crop.x1, crop.x0 + b)].any(axis=1), crop.height))
    if crop.y0 > 0:
        strips.append(("top", piece[crop.y0:min(crop.y1, crop.y0 + b), crop.x0:crop.x1].any(axis=0), crop.width))
    if crop.x1 < width:
        strips.append(("right", piece[crop.y0:crop.y1, max(crop.x0, crop.x1 - b):crop.x1].any(axis=1), crop.height))
    if crop.y1 < height:
        strips.append(("bottom", piece[max(crop.y0, crop.y1 - b):crop.y1, crop.x0:crop.x1].any(axis=0), crop.width))
    touched = set()
    for side, line, length in strips:
        need = max(int(min_px), int(math.ceil(length * float(min_fraction) - 1e-9)))
        if int(np.count_nonzero(line)) >= max(1, need):
            touched.add(side)
    return tuple(side for side in SIDES if side in touched)


def _grow_axis(start: int, end: int, size: int, low: bool, high: bool, limit: int) -> tuple[int, int]:
    """One axis ``[start, end)`` to length ``size``: toward the touched end only (``low`` / ``high``), half each way
    when both or neither touched, then pushed inside (``_fit_axis``). Always holds the original span."""
    extra = max(0, size - (end - start))
    if low and not high:
        begin = start - extra
    elif high and not low:
        begin = start
    else:
        begin = start - extra // 2
    return _fit_axis(begin, size, limit)


def grow_toward(
    crop: CropBox,
    sides: Collection[str],
    image_size: tuple[int, int],
    *,
    factor: float = GROW_FACTOR,
) -> CropBox:
    """The square crop grown toward the touched ``sides``: side = ceil(current side x ``factor``), at least 1 px more.

    Per axis it grows toward the touched end only (both ends or none touched: around the centre) and is pushed inside
    like ``square_crop``; an axis longer than the image is the image. The grown crop always holds the original one;
    a whole-image crop comes back unchanged.
    """
    width, height = (int(v) for v in image_size)
    side = max(crop.width, crop.height)
    size = max(side + 1, int(math.ceil(side * float(factor) - 1e-9)))
    touched = set(sides)
    x0, x1 = _grow_axis(crop.x0, crop.x1, size, "left" in touched, "right" in touched, width)
    y0, y1 = _grow_axis(crop.y0, crop.y1, size, "top" in touched, "bottom" in touched, height)
    return CropBox(x0, y0, x1, y1)


def enclosing_crop(crops: Sequence[CropBox], image_size: tuple[int, int]) -> CropBox:
    """The square holding all ``crops`` (same centre, pushed inside): where a grown face's Exclude is searched."""
    if len(crops) == 1:
        return crops[0]
    box = (min(c.x0 for c in crops), min(c.y0 for c in crops), max(c.x1 for c in crops), max(c.y1 for c in crops))
    grown = square_crop(box, image_size, min_side=1)
    return grown if grown is not None else crops[0]


def growth_requests(
    pieces: Sequence[np.ndarray],
    sources: Sequence[int],
    face_boxes: Sequence[Box],
    crops: Sequence[CropBox],
    image_size: tuple[int, int],
    *,
    faces: Collection[int] | None = None,
) -> dict[int, tuple[tuple[str, ...], CropBox]]:
    """One token's pieces (shaped as for ``assign_owners``) -> (touched sides, grown crop) per face to grow.

    Face k grows when the union of the pieces found in its crop (``crops[k]``, the crop this token uses for that face
    now) and owned by k reaches an inner edge of that crop (``touching_sides``). A neighbour-owned piece (the
    neighbour's hair cut at this crop's edge) never grows it: the neighbour's crop sees the rest. ``faces`` limits
    the faces looked at. A crop that cannot grow (the whole image) is left out.
    """
    owners = assign_owners(pieces, sources, face_boxes, crops)
    requests: dict[int, tuple[tuple[str, ...], CropBox]] = {}
    for face in range(len(crops)):
        if faces is not None and face not in faces:
            continue
        own = [piece for piece, source, owner in zip(pieces, sources, owners) if source == face and owner == face]
        if not own:
            continue
        mask = own[0] if len(own) == 1 else np.any(np.stack(own, axis=0), axis=0)
        sides = touching_sides(mask, crops[face])
        if not sides:
            continue
        grown = grow_toward(crops[face], sides, image_size)
        if grown != crops[face]:
            requests[face] = (sides, grown)
    return requests


def continuing_pieces(pieces: Sequence[np.ndarray], earlier: Sequence[np.ndarray]) -> list[bool]:
    """Per piece found in a grown crop (image coordinates): does it share at least one pixel with the earlier result
    (``earlier``: the pieces this face and token found in its previous crop)? Only those pieces are used.

    A grown crop is there to see more of what the previous crop saw (hair cut at its edge). A piece that does not touch
    the earlier result came into view only in the grown crop and is cut off from the face's mask (the hair of a person
    whose face was not detected, say): it is dropped, so it neither widens the mask nor grows the crop toward it.
    Without ``earlier`` every piece is False.
    """
    if not earlier:
        return [False] * len(pieces)
    seen = np.asarray(earlier[0], dtype=bool)
    if len(earlier) > 1:
        seen = np.any(np.stack([np.asarray(piece, dtype=bool) for piece in earlier], axis=0), axis=0)
    return [bool(np.any(np.logical_and(np.asarray(piece, dtype=bool), seen))) for piece in pieces]


def order_faces(boxes: Sequence[Box]) -> list[int]:
    """Face numbers left -> right (then top -> bottom), so the Individual pass order does not follow detector scores."""
    def key(index: int) -> tuple[float, float, int]:
        x0, y0, x1, y1 = (float(v) for v in boxes[index])
        return ((x0 + x1) / 2.0, (y0 + y1) / 2.0, index)

    return sorted(range(len(boxes)), key=key)


@dataclass(frozen=True)
class CropGrowth:
    """One face x token whose crop grew: its mask reached the inner edges ``sides``, the crop grew ``rounds`` times and
    the token's result comes from the last crop (``crop``). ``face`` is the ``FaceCropInfo`` face number (from 0)."""

    face: int
    prompt: str
    crop: CropBox
    rounds: int
    sides: tuple[str, ...]


def _crop_text(crop: CropBox) -> str:
    return f"{crop.width}x{crop.height}@{crop.x0},{crop.y0}"


@dataclass
class FaceCropInfo:
    """One image's face-crop plan and outcome (the report's ``face_crop`` and its console line).

    ``boxes``, ``scores`` and ``crops`` share one order (left -> right) without boxes outside the image or empty ones.
    ``growth`` lists the (face, token) crops that grew; empty without growth, and then the line and the meta are those
    without growth.
    """

    model: str
    confidence: float
    boxes: list[list[float]]
    scores: list[float]
    crops: list[CropBox]
    growth: list[CropGrowth] = field(default_factory=list)

    @property
    def count(self) -> int:
        return len(self.crops)

    def status(self) -> str:
        """The console line (the caller adds the '[-] SAM3: ' prefix)."""
        where = f"{self.model} (confidence {float(self.confidence):.2f})"
        if not self.crops:
            return f"face crop — no faces found by {where}; nothing to mask."
        sizes = ", ".join(_crop_text(c) for c in self.crops)
        text = f"face crop — {self.count} face(s) by {where}; SAM3 runs on each crop: {sizes}"
        if self.growth:
            grown = ", ".join(
                f"face {g.face + 1} {g.prompt!r} → {_crop_text(g.crop)} "
                f"({g.rounds} round{'s' if g.rounds != 1 else ''})"
                for g in self.growth
            )
            text += f"; grown where the mask reached the crop edge: {grown}"
        return text

    def as_meta(self) -> dict:
        faces = [
            {"box": [float(v) for v in box], "score": float(score), "crop": list(crop.as_tuple())}
            for box, score, crop in zip(self.boxes, self.scores, self.crops)
        ]
        for g in self.growth:
            if 0 <= g.face < len(faces):
                faces[g.face].setdefault("grown", []).append({
                    "prompt": g.prompt,
                    "crop": list(g.crop.as_tuple()),
                    "rounds": int(g.rounds),
                    "sides": list(g.sides),
                })
        return {
            "model": self.model,
            "confidence": float(self.confidence),
            "faces": faces,
        }


def box_iou(a: Box, b: Box) -> float:
    """IoU of two (x0, y0, x1, y1) boxes (overlap / union); 0 without area."""
    ax0, ay0, ax1, ay1 = (float(v) for v in a)
    bx0, by0, bx1, by1 = (float(v) for v in b)
    inter = max(0.0, min(ax1, bx1) - max(ax0, bx0)) * max(0.0, min(ay1, by1) - max(ay0, by0))
    union = (ax1 - ax0) * (ay1 - ay0) + (bx1 - bx0) * (by1 - by0) - inter
    return inter / union if union > 0 else 0.0


def _centre_inside(a: Box, b: Box) -> bool:
    ax0, ay0, ax1, ay1 = (float(v) for v in a)
    bx0, by0, bx1, by1 = (float(v) for v in b)
    cx, cy = (ax0 + ax1) / 2.0, (ay0 + ay1) / 2.0
    return bx0 <= cx < bx1 and by0 <= cy < by1


def same_face(a: Box, b: Box) -> bool:
    """Whether two detector boxes are one face: IoU >= ``SAME_FACE_IOU`` or one box's centre inside the other."""
    return box_iou(a, b) >= SAME_FACE_IOU or _centre_inside(a, b) or _centre_inside(b, a)


def dedupe_faces(boxes: Sequence[Box], scores: Sequence[float]) -> list[int]:
    """One box per face, the higher-scoring one: the kept box numbers in their original order.

    Boxes are visited by score (higher first; equal or non-numeric scores by number) and a box ``same_face`` with a
    kept one is dropped.
    """
    def rank(index: int) -> tuple[bool, float, int]:
        score = float(scores[index]) if index < len(scores) else float("nan")
        finite = math.isfinite(score)
        return (not finite, -score if finite else 0.0, index)

    kept: list[int] = []
    for index in sorted(range(len(boxes)), key=rank):
        if not any(same_face(boxes[index], boxes[other]) for other in kept):
            kept.append(index)
    return sorted(kept)


def plan_face_crops(
    boxes: Sequence[Box],
    scores: Sequence[float],
    image_size: tuple[int, int],
    *,
    model: str,
    confidence: float,
) -> FaceCropInfo:
    """Detector boxes -> the usable faces as a left -> right crop plan (``image_size`` = (width, height)).

    Of boxes on one face (``same_face``) only the higher-scoring one is kept; keeping both would split that face into
    two passes.
    """
    kept_boxes: list[list[float]] = []
    kept_scores: list[float] = []
    for index, box in enumerate(boxes):
        clipped = clip_box(box, image_size)
        if clipped is None:
            continue
        kept_boxes.append(list(clipped))
        kept_scores.append(float(scores[index]) if index < len(scores) else float("nan"))
    unique = dedupe_faces(kept_boxes, kept_scores)
    kept_boxes = [kept_boxes[i] for i in unique]
    kept_scores = [kept_scores[i] for i in unique]
    order = order_faces(kept_boxes)
    ordered_boxes = [kept_boxes[i] for i in order]
    ordered_scores = [kept_scores[i] for i in order]
    crops = [square_crop(expand_box(box), image_size) for box in ordered_boxes]
    keep = [i for i, crop in enumerate(crops) if crop is not None]
    return FaceCropInfo(
        model=str(model),
        confidence=float(confidence),
        boxes=[ordered_boxes[i] for i in keep],
        scores=[ordered_scores[i] for i in keep],
        crops=[crops[i] for i in keep],
    )
