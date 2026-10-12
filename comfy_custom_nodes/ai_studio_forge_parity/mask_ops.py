"""Shape-safe mask operations shared by the Forge-parity ComfyUI nodes.

This module deliberately has no ComfyUI imports.  Torch, OpenCV and Pillow are
loaded only by the operations that need them, which keeps node discovery and
the pure contract tests usable outside a ComfyUI installation.
"""

from __future__ import annotations

import json
import re
import time
from pathlib import Path
from typing import Any, Iterable, Sequence


_TOKEN_SPLIT = re.compile(r"[,;|\n]")
_SAFE_NAME = re.compile(r"[^A-Za-z0-9._-]+")


def split_prompt_groups(prompt: str | None) -> list[list[str]]:
    """Return slash-separated passes containing OR-separated prompt tokens."""

    groups: list[list[str]] = []
    for raw_group in re.split(r"/", prompt or ""):
        tokens = [part.strip() for part in _TOKEN_SPLIT.split(raw_group)]
        tokens = [part for part in tokens if part]
        if tokens:
            groups.append(tokens)
    return groups


# Lazy torch import shared by the whole pack (compat.require_torch).
from .compat import require_torch as _torch  # noqa: E402


# A float IMAGE/MASK above this is on a 0..255 scale, not a [0,1] tensor with
# resampling overshoot: Comfy's bicubic ImageScale/ImageScaleBy (and this pack's
# own bicubic resize before its clamp) peak around 1.2 on hard edges.
_FLOAT_BYTE_SCALE_THRESHOLD = 2.0


def _to_unit_float(value: Any):
    """Convert to float32 in ``[0,1]`` using the dtype, not a max>1 heuristic.

    Integer tensors are byte-scaled (``/255``); bool masks are already 0/1.
    Float tensors are ``[0,1]`` by Comfy contract, so interpolation overshoot
    is clamped rather than divided by 255 (which turned a bicubic-resized
    white-background image almost black).  Only a float tensor far outside the
    overshoot range is treated as 0..255 data.
    """

    torch = _torch()
    if value.dtype == torch.bool:
        return value.to(dtype=torch.float32)
    if not value.is_floating_point() and not value.is_complex():
        return value.to(dtype=torch.float32) / 255.0
    value = value.to(dtype=torch.float32)
    if value.numel() and float(value.detach().max()) > _FLOAT_BYTE_SCALE_THRESHOLD:
        value = value / 255.0
    return value.clamp(0.0, 1.0)


def ensure_image(image: Any):
    """Normalize an IMAGE-like value to float ``[B,H,W,C]`` in ``[0,1]``."""

    torch = _torch()
    value = image if torch.is_tensor(image) else torch.as_tensor(image)
    if value.ndim == 3:
        value = value.unsqueeze(0)
    if value.ndim != 4:
        raise ValueError(f"IMAGE must be [B,H,W,C] or [H,W,C], got {tuple(value.shape)}")
    if value.shape[-1] not in (1, 3, 4):
        if value.shape[1] in (1, 3, 4):
            value = value.permute(0, 2, 3, 1)
        else:
            raise ValueError(f"IMAGE channel dimension is ambiguous: {tuple(value.shape)}")
    return _to_unit_float(value).clamp(0.0, 1.0)


def ensure_mask(mask: Any, *, height: int | None = None, width: int | None = None,
                batch: int | None = None):
    """Normalize a MASK/image-like value to float ``[B,H,W]`` in ``[0,1]``.

    A singleton mask is broadcast to ``batch``.  Any other batch mismatch is
    rejected instead of silently pairing masks with the wrong images.
    """

    torch = _torch()
    import torch.nn.functional as functional

    value = mask if torch.is_tensor(mask) else torch.as_tensor(mask)
    # Scale by dtype before any channel mean: integer/bool tensors cannot be
    # averaged, and the byte-vs-unit decision must not depend on max().
    value = _to_unit_float(value)
    if value.ndim == 2:
        value = value.unsqueeze(0)
    elif value.ndim == 3:
        # Comfy's canonical MASK is [B,H,W].  Interpret [H,W,C] only when
        # the caller supplies dimensions that make that intent unambiguous.
        if (
            value.shape[-1] in (1, 3, 4)
            and height is not None
            and width is not None
            and value.shape[0] == height
            and value.shape[1] == width
        ):
            value = value[..., 3] if value.shape[-1] == 4 else value[..., :3].mean(dim=-1)
            value = value.unsqueeze(0)
    elif value.ndim == 4:
        if value.shape[-1] in (1, 3, 4):
            value = value[..., 3] if value.shape[-1] == 4 else value[..., :3].mean(dim=-1)
        elif value.shape[1] in (1, 3, 4):
            value = value[:, 3] if value.shape[1] == 4 else value[:, :3].mean(dim=1)
        else:
            raise ValueError(f"MASK channel dimension is ambiguous: {tuple(value.shape)}")
    if value.ndim != 3:
        raise ValueError(f"MASK must resolve to [B,H,W], got {tuple(value.shape)}")
    value = value.clamp(0.0, 1.0)
    if height is not None and width is not None and tuple(value.shape[-2:]) != (height, width):
        value = functional.interpolate(
            value.unsqueeze(1), size=(height, width), mode="nearest"
        ).squeeze(1)
    if batch is not None and value.shape[0] != batch:
        if value.shape[0] == 1:
            value = value.expand(batch, -1, -1).clone()
        else:
            raise ValueError(f"MASK batch {value.shape[0]} does not match IMAGE batch {batch}")
    return value


def empty_mask_like(image: Any):
    torch = _torch()
    image = ensure_image(image)
    return torch.zeros(
        (image.shape[0], image.shape[1], image.shape[2]),
        dtype=torch.float32,
        device=image.device,
    )


def union_masks(masks: Iterable[Any], *, reference: Any | None = None):
    """Union equally-shaped masks, or return an empty mask from ``reference``."""

    torch = _torch()
    normalized = [ensure_mask(mask) for mask in masks]
    if not normalized:
        if reference is None:
            raise ValueError("reference is required when unioning an empty mask collection")
        return empty_mask_like(reference)
    shape = tuple(normalized[0].shape)
    if any(tuple(mask.shape) != shape for mask in normalized[1:]):
        raise ValueError("All masks in a union must have exactly the same [B,H,W] shape")
    return torch.stack(normalized, dim=0).amax(dim=0).clamp(0.0, 1.0)


def _opencv():
    try:
        import cv2
        import numpy as np
    except ImportError as exc:
        raise RuntimeError(
            "Convex-hull, dilation and outline processing require opencv-python."
        ) from exc
    return cv2, np


def _map_binary_cv(mask: Any, operation):
    import numpy as np

    torch = _torch()
    normalized = ensure_mask(mask)
    device = normalized.device
    arrays = []
    for item in normalized.detach().cpu().numpy():
        # A transposed/permuted MASK keeps its strides through ensure_mask and
        # .numpy(); OpenCV 5 rejects such a layout as an output array (see
        # make_overlay), so every cv2 step gets a C-contiguous copy.
        arrays.append(operation(np.ascontiguousarray(item > 0.5)))
    return torch.from_numpy(np.stack(arrays, axis=0)).to(
        device=device, dtype=torch.float32
    )


def convex_hull(mask: Any):
    """Fill a convex hull per connected component, matching the Forge extension."""

    cv2, np = _opencv()

    def apply(item):
        if not item.any():
            return item.astype(np.float32)
        source = np.ascontiguousarray(item, dtype=np.uint8) * 255
        count, labels = cv2.connectedComponents(source)
        # fillPoly draws into ``result``: np.zeros_like would copy a
        # non-C-contiguous layout that OpenCV 5 refuses as an output array.
        result = np.zeros(source.shape, dtype=np.uint8)
        for label in range(1, count):
            component = (labels == label).astype(np.uint8) * 255
            contours, _ = cv2.findContours(
                component, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            if contours:
                hull = cv2.convexHull(np.vstack(contours))
                cv2.fillPoly(result, [hull], 255)
        return (result > 0).astype(np.float32)

    return _map_binary_cv(mask, apply)


def dilate(mask: Any, pixels: int):
    if int(pixels) <= 0:
        return ensure_mask(mask)
    cv2, np = _opencv()
    diameter = 2 * int(pixels) + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (diameter, diameter))
    return _map_binary_cv(
        mask, lambda item: (cv2.dilate(item.astype(np.uint8), kernel) > 0).astype(np.float32)
    )


def edge_aware_outline(mask: Any, image: Any, pixels: int,
                       canny_low: int = 100, canny_high: int = 200):
    """Expand until strong image edges, then stop; one image is used per batch."""

    if int(pixels) <= 0:
        return ensure_mask(mask)
    cv2, np = _opencv()
    image_value = ensure_image(image)
    mask_value = ensure_mask(
        mask, height=image_value.shape[1], width=image_value.shape[2],
        batch=image_value.shape[0],
    )
    results = []
    kernel = np.ones((3, 3), dtype=np.uint8)
    for batch_index, item in enumerate(mask_value.detach().cpu().numpy()):
        # VAEDecode's IMAGE is a movedim view (channels-first strides); hand
        # OpenCV C-contiguous arrays like every other cv2 call in this module.
        rgb = np.ascontiguousarray(
            (image_value[batch_index, ..., :3].detach().cpu().numpy() * 255.0).round().astype(np.uint8)
        )
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        edges = cv2.dilate(cv2.Canny(gray, canny_low, canny_high), kernel) > 0
        current = np.ascontiguousarray(item > 0.5)
        for _ in range(int(pixels)):
            expanded = cv2.dilate(current.astype(np.uint8), kernel) > 0
            additions = expanded & ~current & ~edges
            if not additions.any():
                break
            current |= additions
        results.append(current.astype(np.float32))
    return _torch().from_numpy(np.stack(results, axis=0)).to(mask_value.device)


def gaussian_blur(mask: Any, pixels: int):
    """Feather a mask without changing its shape or batch correspondence."""

    torch = _torch()
    import torch.nn.functional as functional

    value = ensure_mask(mask)
    radius = int(pixels)
    if radius <= 0:
        return value
    sigma = max(radius / 3.0, 0.5)
    coords = torch.arange(-radius, radius + 1, device=value.device, dtype=value.dtype)
    kernel_1d = torch.exp(-(coords * coords) / (2.0 * sigma * sigma))
    kernel_1d = kernel_1d / kernel_1d.sum()
    horizontal = kernel_1d.reshape(1, 1, 1, -1)
    vertical = kernel_1d.reshape(1, 1, -1, 1)
    work = value.unsqueeze(1)
    work = functional.pad(work, (radius, radius, 0, 0), mode="replicate")
    work = functional.conv2d(work, horizontal)
    work = functional.pad(work, (0, 0, radius, radius), mode="replicate")
    return functional.conv2d(work, vertical).squeeze(1).clamp(0.0, 1.0)


def refine_generated_mask(mask: Any, image: Any, *, use_convex_hull: bool,
                          outline_pixels: int, dilation_pixels: int):
    """Forge order: convex hull -> edge-aware outline -> plain dilation."""

    result = ensure_mask(mask)
    if use_convex_hull:
        result = convex_hull(result)
    if int(outline_pixels) > 0:
        result = edge_aware_outline(result, image, int(outline_pixels))
    if int(dilation_pixels) > 0:
        result = dilate(result, int(dilation_pixels))
    return result


def subtract_exclusion(groups: Sequence[Any], exclusion: Any | None):
    if exclusion is None:
        return [ensure_mask(mask) for mask in groups]
    protected = ensure_mask(exclusion)
    result = []
    for group in groups:
        current = ensure_mask(group)
        if tuple(current.shape) != tuple(protected.shape):
            raise ValueError("Exclusion mask must match every generated group mask")
        remaining = current * (1.0 - (protected > 0.5).to(current.dtype))
        if bool((remaining > 0.0).any()):
            result.append(remaining)
    return result


def select_mask_groups(generated: Sequence[Any], manual: Any | None,
                       source: str, *, reference: Any):
    """Select generated/manual masks, including Forge's safe intersection fallback."""

    mode = str(source or "generated").strip().lower()
    groups = [ensure_mask(mask) for mask in generated]
    if mode == "generated":
        return groups
    if manual is None:
        raise ValueError(f"mask_source={source!r} requires manual_mask")
    image = ensure_image(reference)
    user = ensure_mask(
        manual, height=image.shape[1], width=image.shape[2], batch=image.shape[0]
    )
    if mode == "manual":
        return [user]
    combined = union_masks(groups, reference=image)
    if mode == "union":
        return [torch_max(combined, user)]
    if mode == "intersection":
        intersection = combined * (user > 0.5).to(combined.dtype)
        # Preserve the hand-drawn ROI for each image where SAM3 missed it.
        chosen = intersection.clone()
        for index in range(chosen.shape[0]):
            if not bool((chosen[index] > 0.0).any()):
                chosen[index] = user[index]
        return [chosen]
    raise ValueError(
        f"Unknown mask_source {source!r}; expected generated, manual, intersection, or union"
    )


def torch_max(left: Any, right: Any):
    torch = _torch()
    left_value, right_value = ensure_mask(left), ensure_mask(right)
    if tuple(left_value.shape) != tuple(right_value.shape):
        raise ValueError("Masks must have identical shapes")
    return torch.maximum(left_value, right_value)


def finish_masks(groups: Sequence[Any], image: Any, *, blur_pixels: int, invert: bool):
    """Apply the final feather/invert stage and return selected mask batches."""

    image_value = ensure_image(image)
    processed = []
    for mask in groups:
        current = gaussian_blur(mask, int(blur_pixels))
        processed.append(current.clamp(0.0, 1.0))
    combined = union_masks(processed, reference=image_value)
    individual = (
        _torch().cat(processed, dim=0)
        if processed
        else combined.clone()
    )
    # Forge checks whether detection produced a usable mask *before* applying
    # the inpaint-mask inversion flag.  Inverting an empty detection here would
    # otherwise turn "nothing found" into a full-frame mask and unexpectedly
    # redraw the entire image.  Forge decides per image, and an empty pass does
    # not exist there (it drops empty group masks), so every batch row is
    # judged on its own: an image of an IMAGE batch that found nothing, or a
    # zero row that only pads an image with fewer Individual masks, stays empty.
    if invert:
        combined = _invert_rows_with_pixels(combined)
        individual = _invert_rows_with_pixels(individual)
    return combined, individual


def _invert_rows_with_pixels(masks: Any):
    """``1 - mask`` for every ``[B,H,W]`` row that has pixels; empty rows stay empty."""
    torch = _torch()
    has_pixels = (masks > 0.0).flatten(1).any(dim=1).view(-1, 1, 1)
    return torch.where(has_pixels, 1.0 - masks, masks)


# sam-extra's face-crop frame colour (RGB): orange, apart from the blue SAM3 boxes.
FACE_CROP_OVERLAY_COLOR = (255, 190, 40)


def make_overlay(image: Any, mask: Any, boxes: Sequence[Sequence[float]] | None = None,
                 scores: Sequence[float] | None = None,
                 regions: Sequence[tuple[Sequence[int], int, str | None]] | None = None):
    """Return Forge-style cyan mask preview with boxes/scores for one image.

    ``regions`` are sam-extra's face-crop frames ``((x0, y0, x1, y1), thickness, label or None)``, drawn before the
    boxes in ``FACE_CROP_OVERLAY_COLOR`` (one image only, like the boxes). Without them the preview is unchanged.
    """

    torch = _torch()
    image_value = ensure_image(image)[..., :3]
    mask_value = ensure_mask(
        mask, height=image_value.shape[1], width=image_value.shape[2],
        batch=image_value.shape[0],
    ).unsqueeze(-1)
    color = torch.tensor([30.0, 210.0, 255.0], device=image_value.device) / 255.0
    strength = mask_value * 0.65
    overlay = (image_value * (1.0 - strength) + color * strength).clamp(0.0, 1.0)
    # Easy SAM3 flattens object metadata for the common B=1 workflow. For a
    # multi-image batch the association is not retained by that provider, so
    # drawing the same boxes on every image would be incorrect; keep the
    # accurate cyan masks and expose boxes separately in that case.
    frames = list(regions or ())
    if ((boxes is not None and len(boxes) > 0) or frames) and overlay.shape[0] == 1:
        cv2, np = _opencv()
        # The IMAGE from VAEDecode is a movedim(1, -1) view, and the blend
        # above keeps its channels-first strides, so this array is not
        # C-contiguous. cv2.rectangle/putText draw in place, and OpenCV 5.0
        # rejects that layout ("Layout of the output array img is
        # incompatible with cv::Mat"), which failed every in-generation SAM3.
        array = np.ascontiguousarray(
            (overlay[0].detach().cpu().numpy() * 255.0).round().astype(np.uint8)
        )
        for (x0, y0, x1, y1), thickness, label in frames:
            cv2.rectangle(array, (int(x0), int(y0)), (int(x1) - 1, int(y1) - 1), FACE_CROP_OVERLAY_COLOR,
                          int(thickness))
            if label:
                cv2.putText(
                    array, str(label), (int(x0) + 4, min(int(y1) - 6, int(y0) + 18)), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, FACE_CROP_OVERLAY_COLOR, 1, cv2.LINE_AA,
                )
        score_values = list(scores or [])
        for index, box in enumerate(boxes or ()):
            if len(box) != 4:
                continue
            x1, y1, x2, y2 = [int(round(float(value))) for value in box]
            cv2.rectangle(array, (x1, y1), (x2, y2), (30, 120, 255), 2)
            label = f"{float(score_values[index]):.2f}" if index < len(score_values) else "n/a"
            cv2.putText(
                array, label, (x1, max(y1 - 10, 18)), cv2.FONT_HERSHEY_SIMPLEX,
                0.6, (30, 120, 255), 2, cv2.LINE_AA,
            )
        overlay = torch.from_numpy(array).to(device=image_value.device, dtype=image_value.dtype)
        overlay = overlay.unsqueeze(0) / 255.0
    return overlay


def _safe_slug(value: str | None) -> str:
    cleaned = _SAFE_NAME.sub("_", (value or "").strip()).strip("_")
    return (cleaned or "mask")[:32]


def save_mask_artifacts(directory: str | Path, *, combined: Any, individuals: Any,
                        overlay: Any, prompt: str, seed: int | None,
                        metadata: dict[str, Any] | None = None) -> dict[str, Any]:
    """Persist unique combined/individual masks, overlay and JSON metadata."""

    try:
        from PIL import Image
        import numpy as np
    except ImportError as exc:  # pragma: no cover - declared app dependencies
        raise RuntimeError("Saving SAM3 artifacts requires Pillow and NumPy.") from exc

    output = Path(directory).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    stem = f"sam3_{seed}" if seed is not None and int(seed) >= 0 else "sam3"
    slug = _safe_slug(prompt)
    nonce = time.time_ns()
    prefix = output / f"{stem}_{slug}_{nonce}"

    def as_u8(value):
        tensor = value.detach().cpu().clamp(0.0, 1.0)
        return (tensor.numpy() * 255.0).round().astype(np.uint8)

    combined_value = ensure_mask(combined)
    individual_value = ensure_mask(individuals)
    overlay_value = ensure_image(overlay)
    mask_paths = []
    for index, item in enumerate(individual_value):
        path = Path(f"{prefix}_mask_{index + 1:02d}.png")
        Image.fromarray(as_u8(item), mode="L").save(path)
        mask_paths.append(str(path))
    combined_path = Path(f"{prefix}_mask.png")
    overlay_path = Path(f"{prefix}_overlay.png")
    meta_path = Path(f"{prefix}_prompt.json")
    Image.fromarray(as_u8(combined_value[0]), mode="L").save(combined_path)
    Image.fromarray(as_u8(overlay_value[0, ..., :3]), mode="RGB").save(overlay_path)
    payload = dict(metadata or {})
    payload.update({
        "prompt": prompt,
        "seed": seed,
        "combined_mask": str(combined_path),
        "individual_masks": mask_paths,
        "overlay": str(overlay_path),
    })
    meta_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    payload["metadata"] = str(meta_path)
    return payload


def mask_bounds(mask: Any, padding: int = 0) -> tuple[int, int, int, int] | None:
    """Return inclusive-exclusive ``(x1,y1,x2,y2)`` bounds for a mask batch."""

    torch = _torch()
    value = ensure_mask(mask)
    points = torch.nonzero(value.amax(dim=0) > 0.001, as_tuple=False)
    if points.numel() == 0:
        return None
    height, width = value.shape[-2:]
    y1 = max(0, int(points[:, 0].min()) - int(padding))
    y2 = min(height, int(points[:, 0].max()) + 1 + int(padding))
    x1 = max(0, int(points[:, 1].min()) - int(padding))
    x2 = min(width, int(points[:, 1].max()) + 1 + int(padding))
    return x1, y1, x2, y2


def _grow_span(start: int, stop: int, grow: int, limit: int) -> tuple[int, int]:
    """``[start, stop)`` grown by ``grow`` pixels around its centre (an odd
    pixel goes to the far end), slid back inside ``[0, limit)`` -- first off
    the far edge, then off zero -- and cut at ``limit`` if still too long."""

    start, stop = start - grow // 2, stop + grow - grow // 2
    overshoot = max(0, stop - limit)
    start, stop = start - overshoot, stop - overshoot
    if start < 0:
        start, stop = 0, stop - start
    return start, min(stop, limit)


def expand_crop_region(crop_region: Sequence[int], processing_width: int,
                       processing_height: int, image_width: int,
                       image_height: int) -> tuple[int, int, int, int]:
    """Widen the padded mask crop to the processing aspect ratio (inpaint
    "only masked"), e.g. a 128x32 crop processed at 512x512 becomes 128x128.

    The short side grows around its centre; at an image edge the crop slides
    back inside, and an image too small for the ratio cuts it.  This is the
    crop rule of Forge's ``modules/masking.py`` ``expand_crop_region`` (A1111
    webui, AGPL-3.0) with the same integer results, so the Comfy crop lands on
    the same pixels.  The code is this pack's own; no Forge code is included
    (THIRD_PARTY_NOTICES.md "Forge inpaint crop rule";
    ``tests/test_comfy_sam3_nodes.py`` compares it with Forge's function).
    """

    x1, y1, x2, y2 = (int(value) for value in crop_region)
    crop_width, crop_height = x2 - x1, y2 - y1
    if crop_width <= 0 or crop_height <= 0:
        raise ValueError(f"crop region must be non-empty, got {(x1, y1, x2, y2)}")
    target_width, target_height = int(processing_width), int(processing_height)
    if target_width <= 0 or target_height <= 0:
        raise ValueError("processing size must be positive")
    aspect = target_width / target_height
    if crop_width / crop_height > aspect:      # wider than the target: taller
        y1, y2 = _grow_span(y1, y2, int(crop_width / aspect - crop_height), int(image_height))
    else:                                      # narrower (or equal): wider
        x1, x2 = _grow_span(x1, x2, int(crop_height * aspect - crop_width), int(image_width))
    return x1, y1, x2, y2


def fit_sample_size(crop_width: int, crop_height: int, processing_width: int,
                    processing_height: int) -> tuple[int, int]:
    """Aspect-preserving sample size for a crop processed at a target size.

    Forge resizes the (already aspect-matched) crop to the processing size
    with "Resize and Fill" and later scales the result back with "Crop and
    Resize", i.e. the crop content is effectively sampled at
    ``min(W/cw, H/ch)``.  Returning that size directly keeps the same scale
    and never stretches the crop when the image edge prevented a full
    aspect-ratio expansion.

    Only for the only-masked crop path: a whole-picture pass is "Just
    Resize"d to exactly the processing size instead.
    """

    crop_width, crop_height = int(crop_width), int(crop_height)
    processing_width, processing_height = int(processing_width), int(processing_height)
    if min(crop_width, crop_height, processing_width, processing_height) <= 0:
        raise ValueError("crop and processing sizes must be positive")
    scale = min(processing_width / crop_width, processing_height / crop_height)
    return (
        max(1, min(processing_width, int(round(crop_width * scale)))),
        max(1, min(processing_height, int(round(crop_height * scale)))),
    )
