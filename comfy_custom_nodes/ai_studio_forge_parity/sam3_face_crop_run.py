"""SAM3 face crop for ``ForgeNeoSAM3Mask``: sam-extra's face-crop branch of ``run_sam3_on_pil`` (pack 1.8.0).

``run_face_crop`` is the face-crop part of sam-extra's ``sam3ext/core.py`` ``run_sam3_on_pil`` (v0.34.0 with the crop
growth of v0.34.1 as fixed before its release, sam-extra commit b815f68), statement for statement: SAM3 runs per face
crop (every Detect token of every group, then the Exclude tokens in the crop widened by Outline + Dilation), the
pieces are pasted back at image coordinates, a token whose owned pieces reach an inner crop edge runs again on a grown
crop (at most ``sam3_face_crop.GROW_ROUNDS`` times per face and token; only the pieces that overlap the earlier result
are kept, with their boxes and scores; a grown crop that keeps nothing keeps the earlier result), a grown face's
Exclude is also searched around its grown crops (on its last grown crop right away when Outline + Dilation is 0, and
every crop/area and token is asked once), and every group gives one mask per owner face (left -> right) after Convex
Hull -> Outline -> Dilation. Boxes and scores come face -> group -> token. The SAM3 calls are made in sam-extra's
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
    continuing_pieces,
    enclosing_crop,
    grow_crop,
    growth_requests,
    paste_mask,
    touching_sides,
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


def _continuing_result(
    found: tuple[list[np.ndarray], list[list[float]], list[float], list[int]],
    earlier: list[np.ndarray],
) -> tuple[tuple[list[np.ndarray], list[list[float]], list[float]], int]:
    """sam-extra ``core._continuing_result``: of what a grown crop found (pieces, boxes, scores and each piece's box
    number from ``_run_token``) only the pieces that overlap the earlier result (``earlier``: the pieces this face and
    token found in its previous crop) with their boxes and scores -> ((pieces, boxes, scores), dropped piece count).

    A box (and the score with its number) that only dropped pieces point at is dropped too; a box a kept piece also
    points at stays. When every piece overlaps, the result is the one given.
    """
    pieces, boxes, scores, links = found
    keep = continuing_pieces(pieces, earlier)
    if all(keep):
        return (list(pieces), list(boxes), list(scores)), 0
    kept_links = {link for link, flag in zip(links, keep) if flag}
    dropped = {link for link, flag in zip(links, keep) if not flag} - kept_links
    return (
        [piece for piece, flag in zip(pieces, keep) if flag],
        [box for index, box in enumerate(boxes) if index not in dropped],
        [score for index, score in enumerate(scores) if index not in dropped],
    ), keep.count(False)


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

    def _run_token(
        sub_prompt: str, crop: CropBox
    ) -> tuple[list[np.ndarray], list[list[float]], list[float], list[int]]:
        """One token on one crop: pieces and boxes (image coordinates), scores, and per piece its box number (-1 when
        there is none): the box whose rounded rectangle overlaps the piece's bbox most, as ``_clean_split_masks`` picks
        it. The numbers are only used to drop a dropped piece's box and score (``_continuing_result``)."""
        split, crop_boxes, crop_scores = segment(sub_prompt, crop)
        found = [piece for piece in split if np.any(piece)]
        pieces = [paste_mask(piece, crop, (h, w)) for piece in found]
        boxes = [box_to_image([float(v) for v in box], crop) for box in crop_boxes]
        links = [-1] * len(found)
        if crop_boxes:
            rects = [tuple(int(round(float(v))) for v in box) for box in crop_boxes]
            for index, piece in enumerate(found):
                piece_box = _mask_bbox(piece)
                if piece_box is not None:
                    links[index] = max(range(len(rects)), key=lambda i: _intersection_area(piece_box, rects[i]))
        return pieces, boxes, [float(score) for score in crop_scores], links

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
                runs[group_index][token_index][region] = _run_token(sub_prompt, crop)[:3]
        _search_exclude(exclude_area[region])

    # Crop growth: a (token, face) whose owned pieces are cut at an inner edge of its crop runs that token again on a
    # crop grown toward the touched sides (owners are decided again every pass). Only the pieces of a grown run that
    # overlap the earlier result are used (``continuing_pieces``; the others and their boxes/scores are dropped). The
    # cap is GROW_ROUNDS growths per face x token, so a face that first reaches an edge after its neighbour grew still
    # grows its share. Without contact SAM3 is not called again.
    rounds_cap = int(geometry.GROW_ROUNDS)
    grown_count = [[[0] * n_crops for _ in tokens] for tokens in groups]   # growths per (group, token, face)
    grown_sides: dict[tuple[int, int, int], set[str]] = {}
    settled: set[tuple[int, int, int]] = set()   # a grown crop kept nothing: the earlier result stays, no more growth
    # What a grown crop found, and Exclude pieces, per (crop, token): two faces grown to the same crop, or the same
    # Exclude area, ask SAM3 once.
    grown_found: dict[tuple[CropBox, str], tuple[list[np.ndarray], list[list[float]], list[float], list[int]]] = {}
    exclude_found: dict[tuple[CropBox, str], list[np.ndarray]] = {}

    def _exclude_in(sub_prompt: str, area: CropBox) -> list[np.ndarray]:
        key = (area, sub_prompt)
        if key not in exclude_found:
            split, _, _ = segment(sub_prompt, area)
            exclude_found[key] = [paste_mask(piece, area, (h, w)) for piece in split]
        return exclude_found[key]

    while True:   # every pass grows (count + 1) or settles at least one face x token, so it always ends
        jobs: dict[CropBox, list[tuple[int, int, int, tuple[str, ...]]]] = {}
        for group_index, group_tokens in enumerate(groups):
            for token_index in range(len(group_tokens)):
                pieces, sources = _token_pieces(group_index, token_index)
                if not pieces:
                    continue
                counts = grown_count[group_index][token_index]
                open_faces = {
                    f for f in range(n_crops)
                    if (group_index, token_index, f) not in settled and counts[f] < rounds_cap
                }
                if not open_faces:
                    continue
                requests = growth_requests(
                    pieces, sources, face_boxes, token_crops[group_index][token_index], (w, h), faces=open_faces
                )
                for face, (sides, grown) in requests.items():
                    jobs.setdefault(grown, []).append((group_index, token_index, face, sides))
        if not jobs:
            break
        for grown, items in jobs.items():
            where = _crop_text(grown)
            last = False   # this crop is some face x token's last crop (it grows no further)
            for group_index, token_index, face, sides in items:
                sub_prompt = groups[group_index][token_index]
                key = (group_index, token_index, face)
                if (grown, sub_prompt) not in grown_found:
                    grown_found[(grown, sub_prompt)] = _run_token(sub_prompt, grown)
                found = grown_found[(grown, sub_prompt)]
                result, left_out = _continuing_result(found, runs[group_index][token_index][face][0])
                if not result[0]:
                    settled.add(key)
                    before = token_crops[group_index][token_index][face]
                    what = f"nothing found in the grown crop {where}"
                    if found[0]:
                        what = f"no piece of the grown crop {where} overlaps the earlier result"
                    log(f"face crop — face {face + 1} {sub_prompt!r}: {what}; keeping the result from "
                        f"{_crop_text(before)}")
                    continue
                runs[group_index][token_index][face] = result
                token_crops[group_index][token_index][face] = grown
                grown_count[group_index][token_index][face] += 1
                count = grown_count[group_index][token_index][face]
                grown_sides.setdefault(key, set()).update(sides)
                if not last:
                    reached = np.any(np.stack(result[0], axis=0), axis=0)
                    last = count >= rounds_cap or not touching_sides(reached, grown)
                note = f"; left out {left_out} piece(s) not overlapping the earlier result" if left_out else ""
                log(f"face crop — face {face + 1} {sub_prompt!r} reached the crop edge ({', '.join(sides)}); "
                    f"SAM3 again on {where} (round {count}/{rounds_cap}){note}")
            # With Outline + Dilation 0 a grown face's Exclude area is its last crop: search it now, while that crop is
            # the one SAM3 just saw (sam-extra keeps its backbone; the Exclude step below then reuses this result).
            if last and exclude_tokens and growth == 0:
                for sub_prompt in exclude_tokens:
                    _exclude_in(sub_prompt, grown)
    for (group_index, token_index, face), sides in grown_sides.items():
        info.growth.append(
            CropGrowth(
                face=face,
                prompt=groups[group_index][token_index],
                crop=token_crops[group_index][token_index][face],
                rounds=grown_count[group_index][token_index][face],
                sides=tuple(side for side in SIDES if side in sides),
            )
        )
    info.growth.sort(key=lambda g: (g.face, g.prompt, g.crop.as_tuple()))
    # A grown face's Exclude is also searched where its mask can reach (around every crop it used + Outline +
    # Dilation); two faces with the same area search it once.
    if exclude_tokens and grown_sides:
        searched: set[CropBox] = set()
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
            if area in searched:
                continue
            searched.add(area)
            for sub_prompt in exclude_tokens:
                exclude_pieces.extend(_exclude_in(sub_prompt, area))
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
