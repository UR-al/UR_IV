"""SAM3 face crop for ``ForgeNeoSAM3Mask``: sam-extra's face-crop branch of ``run_sam3_on_pil`` (pack 1.8.0).

``run_face_crop`` is the face-crop part of sam-extra's ``sam3ext/core.py`` ``run_sam3_on_pil`` (v0.34.0 with the crop
growth of the v0.34.1 fix, sam-extra commit 959e5d6), statement for statement: SAM3 runs per face crop (every Detect
token of every group, then the Exclude tokens in the crop widened by Outline + Dilation), the pieces are pasted back
at image coordinates, a token whose owned pieces reach an inner crop edge runs again on a grown crop (at most
``sam3_face_crop.GROW_ROUNDS`` rounds; a grown crop that finds nothing keeps the earlier result), a grown face's
Exclude is also searched around its grown crops, and every group gives one mask per owner face (left -> right) after
Convex Hull -> Outline -> Dilation. Boxes and scores come face -> group -> token. The SAM3 calls are made in sam-extra's
order. ``_clean_split_masks`` and its helpers are sam-extra's (``core.py``): every SAM3 object mask is cut to its best
box plus a margin. All of it is sam-extra's own code under its author's MIT grant (``THIRD_PARTY_NOTICES.md``).

``segment_batch`` runs it per image of an IMAGE batch with the node's torch mask stages: the hand-drawn mask
(``select_mask_groups``) and then the Exclude subtraction per image, as sam-extra orders them. A batch group k holds
every image's k-th mask (zeros where an image has fewer), like the node's other Individual batches.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Sequence

import numpy as np

from . import sam3_face_crop as geometry
from .sam3_face_crop import (
    SIDES,
    CropBox,
    CropGrowth,
    FaceCropInfo,
    assign_owners,
    box_to_image,
    enclosing_crop,
    grow_crop,
    growth_requests,
    paste_mask,
)

# (pieces in crop coordinates, boxes in crop coordinates, scores) of one SAM3 text prompt on one crop.
Segment = Callable[[str, CropBox], "tuple[list[np.ndarray], list[list[float]], list[float]]"]


# ---- sam-extra core.py: SAM3 object masks -> cleaned pieces --------------------------------------------------------

def _split_masks(masks: np.ndarray, height: int, width: int) -> list[np.ndarray]:
    if masks.size == 0:
        return []
    masks = masks.astype(bool)
    if masks.ndim == 2:
        return [masks]
    flat = masks.reshape((-1, height, width))
    return [mask for mask in flat if np.any(mask)]


def _mask_bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    ys, xs = np.where(mask)
    if ys.size == 0 or xs.size == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


def _intersection_area(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> int:
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])
    if x2 <= x1 or y2 <= y1:
        return 0
    return int((x2 - x1) * (y2 - y1))


def _restrict_mask_to_box(mask: np.ndarray, box: np.ndarray, height: int, width: int) -> np.ndarray:
    x1, y1, x2, y2 = [int(round(v)) for v in box.tolist()]
    pad = max(8, int(round(max(x2 - x1, y2 - y1) * 0.08)))
    x1 = max(0, x1 - pad)
    y1 = max(0, y1 - pad)
    x2 = min(width, x2 + pad)
    y2 = min(height, y2 + pad)

    clipped = np.zeros_like(mask, dtype=bool)
    clipped[y1:y2, x1:x2] = mask[y1:y2, x1:x2]
    return clipped


def _clean_split_masks(masks: np.ndarray, boxes: np.ndarray, height: int, width: int) -> list[np.ndarray]:
    split_masks = _split_masks(masks, height, width)
    if not split_masks:
        return []
    if boxes.size == 0:
        return split_masks

    box_rects = [tuple(int(round(v)) for v in box.tolist()) for box in boxes]
    cleaned_masks: list[np.ndarray] = []
    for mask in split_masks:
        mask_box = _mask_bbox(mask)
        if mask_box is None:
            continue

        best_index = max(
            range(len(box_rects)),
            key=lambda idx: _intersection_area(mask_box, box_rects[idx]),
        )
        cleaned = _restrict_mask_to_box(mask, boxes[best_index], height, width)
        if np.any(cleaned):
            cleaned_masks.append(cleaned)
    return cleaned_masks


def clean_pieces(masks: Any, boxes: Sequence[Sequence[float]], height: int, width: int) -> list[np.ndarray]:
    """SAM3's object masks of one ``height`` x ``width`` region (any ``(..., h, w)`` array or tensor) and its boxes ->
    the non-empty pieces sam-extra keeps (``_clean_split_masks``)."""
    if hasattr(masks, "detach"):
        masks = masks.detach().cpu().numpy()
    array = np.asarray(masks)
    if array.ndim >= 2 and tuple(array.shape[-2:]) != (int(height), int(width)):
        height, width = int(array.shape[-2]), int(array.shape[-1])   # paste_mask fits it to the crop
    box_array = np.asarray([[float(v) for v in box] for box in boxes], dtype=np.float64).reshape(-1, 4)
    return _clean_split_masks(array, box_array, int(height), int(width))


# ---- sam-extra core.py run_sam3_on_pil, face-crop branch ---------------------------------------------------------

@dataclass
class FaceCropOutcome:
    """One image: the finished group masks (group -> owner face, non-empty), boxes/scores (face -> group -> token,
    image coordinates), the Exclude pieces (image coordinates) and the plan with its growth records."""

    group_masks: list[np.ndarray] = field(default_factory=list)
    boxes: list[list[float]] = field(default_factory=list)
    scores: list[float] = field(default_factory=list)
    exclude_pieces: list[np.ndarray] = field(default_factory=list)
    info: FaceCropInfo | None = None


def _crop_text(crop: CropBox) -> str:
    return f"{crop.width}x{crop.height}@{crop.x0},{crop.y0}"


def run_face_crop(
    info: FaceCropInfo,
    groups: Sequence[Sequence[str]],
    exclude_tokens: Sequence[str],
    segment: Segment,
    finish: Callable[[np.ndarray], np.ndarray],
    *,
    image_size: tuple[int, int],
    margin: int,
    log: Callable[[str], None] = lambda line: None,
) -> FaceCropOutcome:
    """sam-extra's face-crop branch for one image (``image_size`` = (width, height)).

    ``segment(token, crop)`` runs SAM3 on that crop of the image and returns its cleaned pieces, boxes (both in crop
    coordinates) and scores. ``finish(mask)`` is Convex Hull -> Outline -> Dilation on an image-sized bool mask.
    ``margin`` = Outline px + Dilation px. ``log`` gets sam-extra's console lines without the '[-] SAM3: ' prefix.
    Without crops nothing runs (sam-extra returns before loading SAM3).
    """
    w, h = int(image_size[0]), int(image_size[1])
    outcome = FaceCropOutcome(info=info)
    face_boxes = info.boxes
    crops = info.crops
    n_crops = len(crops)
    if not n_crops:
        return outcome

    runs: list[list[list[tuple[list[np.ndarray], list[list[float]], list[float]]]]] = [
        [[([], [], []) for _ in crops] for _ in tokens] for tokens in groups
    ]
    token_crops = [[list(crops) for _ in tokens] for tokens in groups]
    exclude_pieces: list[np.ndarray] = []

    def _run_token(sub_prompt: str, crop: CropBox) -> tuple[list[np.ndarray], list[list[float]], list[float]]:
        split, crop_boxes, crop_scores = segment(sub_prompt, crop)
        pieces = [paste_mask(piece, crop, (h, w)) for piece in split if np.any(piece)]
        boxes = [box_to_image([float(v) for v in box], crop) for box in crop_boxes]
        return pieces, boxes, [float(score) for score in crop_scores]

    def _token_pieces(group_index: int, token_index: int) -> tuple[list[np.ndarray], list[int]]:
        pieces: list[np.ndarray] = []
        sources: list[int] = []
        for face, (face_pieces, _, _) in enumerate(runs[group_index][token_index]):
            pieces.extend(face_pieces)
            sources.extend([face] * len(face_pieces))
        return pieces, sources

    def _search_exclude(area: CropBox) -> None:
        for sub_prompt in exclude_tokens:
            split, _, _ = segment(sub_prompt, area)
            exclude_pieces.extend(paste_mask(piece, area, (h, w)) for piece in split)

    growth = max(0, int(margin))
    exclude_area = list(crops)
    if exclude_tokens and growth > 0:
        exclude_area = [grow_crop(crop, growth, (w, h)) for crop in crops]
    for region, crop in enumerate(crops):
        for group_index, group_tokens in enumerate(groups):
            for token_index, sub_prompt in enumerate(group_tokens):
                runs[group_index][token_index][region] = _run_token(sub_prompt, crop)
        _search_exclude(exclude_area[region])

    rounds_cap = int(geometry.GROW_ROUNDS)
    grown_rounds = [[[0] * n_crops for _ in tokens] for tokens in groups]
    grown_sides: dict[tuple[int, int, int], set[str]] = {}
    settled: set[tuple[int, int, int]] = set()
    for round_index in range(1, rounds_cap + 1):
        jobs: dict[CropBox, list[tuple[int, int, int, tuple[str, ...]]]] = {}
        for group_index, group_tokens in enumerate(groups):
            for token_index in range(len(group_tokens)):
                pieces, sources = _token_pieces(group_index, token_index)
                if not pieces:
                    continue
                open_faces = {f for f in range(n_crops) if (group_index, token_index, f) not in settled}
                requests = growth_requests(
                    pieces, sources, face_boxes, token_crops[group_index][token_index], (w, h), faces=open_faces
                )
                for face, (sides, grown) in requests.items():
                    jobs.setdefault(grown, []).append((group_index, token_index, face, sides))
        if not jobs:
            break
        for grown, items in jobs.items():
            for group_index, token_index, face, sides in items:
                sub_prompt = groups[group_index][token_index]
                where = _crop_text(grown)
                result = _run_token(sub_prompt, grown)
                key = (group_index, token_index, face)
                if not result[0]:
                    settled.add(key)
                    log(f"face crop — face {face + 1} {sub_prompt!r}: nothing found in the grown crop {where}; "
                        "keeping the face crop result")
                    continue
                runs[group_index][token_index][face] = result
                token_crops[group_index][token_index][face] = grown
                grown_rounds[group_index][token_index][face] = round_index
                grown_sides.setdefault(key, set()).update(sides)
                log(f"face crop — face {face + 1} {sub_prompt!r} reached the crop edge ({', '.join(sides)}); "
                    f"SAM3 again on {where} (round {round_index}/{rounds_cap})")
    for (group_index, token_index, face), sides in grown_sides.items():
        info.growth.append(
            CropGrowth(
                face=face,
                prompt=groups[group_index][token_index],
                crop=token_crops[group_index][token_index][face],
                rounds=grown_rounds[group_index][token_index][face],
                sides=tuple(side for side in SIDES if side in sides),
            )
        )
    info.growth.sort(key=lambda g: (g.face, g.prompt, g.crop.as_tuple()))
    if exclude_tokens and grown_sides:
        for face in range(n_crops):
            used: list[CropBox] = []
            for group_crops in token_crops:
                for face_crops in group_crops:
                    if face_crops[face] != crops[face] and face_crops[face] not in used:
                        used.append(face_crops[face])
            if not used:
                continue
            area = grow_crop(enclosing_crop(used, (w, h)), growth, (w, h))
            if area == exclude_area[face]:
                continue
            _search_exclude(area)
    outcome.exclude_pieces = exclude_pieces

    for face in range(n_crops):
        for group_runs in runs:
            for face_runs in group_runs:
                outcome.boxes.extend(face_runs[face][1])
                outcome.scores.extend(face_runs[face][2])
    for group_index, group_tokens in enumerate(groups):
        per_face: list[list[np.ndarray]] = [[] for _ in crops]
        for token_index in range(len(group_tokens)):
            pieces, sources = _token_pieces(group_index, token_index)
            owners = assign_owners(pieces, sources, face_boxes, token_crops[group_index][token_index])
            for piece, owner in zip(pieces, owners):
                per_face[owner].append(piece)
        for owned in per_face:
            if owned:
                finished = np.asarray(finish(np.any(np.stack(owned, axis=0), axis=0)), dtype=bool)
                if np.any(finished):
                    outcome.group_masks.append(finished)
    return outcome


# ---- the node's IMAGE batch ----------------------------------------------------------------------------------------

@dataclass
class FaceCropBatch:
    """``segment_batch`` result: the selected group masks as ``[B,H,W]`` batches, boxes/scores of every image (image
    order) and each image's outcome (``outcomes[b].info`` is its plan)."""

    groups: list[Any] = field(default_factory=list)
    boxes: list[list[float]] = field(default_factory=list)
    scores: list[float] = field(default_factory=list)
    outcomes: list[FaceCropOutcome] = field(default_factory=list)


def segment_batch(
    image: Any,
    plans: Sequence[FaceCropInfo],
    *,
    groups: Sequence[Sequence[str]],
    exclude_tokens: Sequence[str],
    segment_image: Callable[[int, str, CropBox], "tuple[list[np.ndarray], list[list[float]], list[float]]"],
    refine: Callable[[np.ndarray, int], np.ndarray],
    manual_mask: Any,
    source: str,
    margin: int,
    log: Callable[[str], None] = lambda line: None,
) -> FaceCropBatch:
    """``run_face_crop`` for every image of ``image`` ([B,H,W,C]) with its plan, then per image the hand-drawn mask
    (``mask_source``) and the Exclude subtraction (sam-extra: manual intersection first, then Exclude)."""
    from .mask_ops import _torch, ensure_mask, select_mask_groups, subtract_exclusion

    torch = _torch()
    batch, height, width = int(image.shape[0]), int(image.shape[1]), int(image.shape[2])
    manual = None if manual_mask is None else ensure_mask(manual_mask, height=height, width=width, batch=batch)
    result = FaceCropBatch()
    per_image: list[list[Any]] = []
    for index in range(batch):
        outcome = run_face_crop(
            plans[index], groups, exclude_tokens,
            lambda token, crop, index=index: segment_image(index, token, crop),
            lambda mask, index=index: refine(mask, index),
            image_size=(width, height), margin=margin, log=log,
        )
        reference = image[index:index + 1]
        generated = [
            torch.from_numpy(mask.astype(np.float32)).unsqueeze(0).to(device=image.device)
            for mask in outcome.group_masks
        ]
        selected = select_mask_groups(
            generated, None if manual is None else manual[index:index + 1], source, reference=reference,
        )
        if exclude_tokens and selected and outcome.exclude_pieces:
            protected = np.any(np.stack(outcome.exclude_pieces, axis=0), axis=0).astype(np.float32)
            selected = subtract_exclusion(
                selected, torch.from_numpy(protected).unsqueeze(0).to(device=image.device),
            )
        per_image.append(selected)
        result.boxes.extend(outcome.boxes)
        result.scores.extend(outcome.scores)
        result.outcomes.append(outcome)
    count = max((len(items) for items in per_image), default=0)
    for slot in range(count):
        rows = [
            items[slot][0] if slot < len(items)
            else torch.zeros((height, width), dtype=torch.float32, device=image.device)
            for items in per_image
        ]
        result.groups.append(torch.stack(rows, dim=0))
    return result
