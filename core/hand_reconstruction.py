"""Pixel-preserving preparation/compositing for experimental hand rebuilding.

This is deliberately not a finger counter or an anatomical correctness judge.
It removes the old pixels under a user-drawn mask before handing a context crop
to the selected image backend. No files, network, models, or GPU are accessed.

Upload decoding, still-raster checks and PNG metadata preservation are shared
with the relight editor through core.local_image_io (one validation policy).
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from typing import Any

from PIL import Image, ImageChops, ImageFilter, ImageOps

from core.local_image_io import (capture_png_metadata, decode_image_data_url, encode_png,
                                 open_still_raster)


# Limits stay module globals and are read at call time (tests patch them).
MAX_PIXELS = 16_777_216
MAX_FILE_BYTES = 64 * 1024 * 1024
MAX_REQUEST_CHARS = 128 * 1024 * 1024
MAX_METADATA_BYTES = 1024 * 1024


@dataclass(frozen=True)
class PreparedHandRepair:
    """Validated repair input.

    There is deliberately no re-encoded full-resolution copy of the source: the
    panel already holds the exact image it sent, so the completion event carries
    only the working-size prepared crop and the composited candidates.
    """
    source: Image.Image
    source_metadata: dict[str, Any]
    edit_mask: Image.Image
    bbox: tuple[int, int, int, int]
    working_size: tuple[int, int]
    content_box: tuple[int, int, int, int]
    settings: dict[str, Any]
    source_sha256: str
    init_png: bytes
    mask_png: bytes
    prepared_png: bytes


def _settings_from(raw: Any) -> dict[str, Any]:
    if not isinstance(raw, dict) or raw.get("enabled") is not True:
        raise ValueError("실험 손 재구성을 먼저 켜 주세요. 기본 상태에서는 실행하지 않습니다.")
    result: dict[str, Any] = {"enabled": True}
    value = raw.get("strength", 0.9)
    if (isinstance(value, bool) or not isinstance(value, (int, float))
            or isinstance(value, float) and not math.isfinite(value)):
        raise ValueError("재구성 강도는 유한한 숫자여야 합니다.")
    if not 0.65 <= value <= 1.0:
        raise ValueError("재구성 강도는 0.65~1.0 범위여야 합니다.")
    result["strength"] = float(value)
    for key, default, low, high in (("candidates", 2, 1, 4), ("padding", 64, 0, 256),
                                    ("feather", 4, 0, 16), ("resolution", 768, 512, 1024)):
        value = raw.get(key, default)
        if isinstance(value, bool) or not isinstance(value, int) or not low <= value <= high:
            raise ValueError(f"{key}: {low}~{high} 범위의 정수여야 합니다.")
        if key == "resolution" and value not in (512, 768, 1024):
            raise ValueError("재구성 해상도는 512, 768, 1024 중에서 선택하세요.")
        result[key] = value
    return result


def _decode_data_url(value: Any, label: str) -> bytes:
    # Verifies the claimed MIME against the actual raster format, not its suffix.
    return decode_image_data_url(value, label, max_bytes=MAX_FILE_BYTES)


def _read_raster(data: bytes, label: str, *, metadata: bool = False) -> tuple[Image.Image, dict[str, Any]]:
    with open_still_raster(data, label, max_bytes=MAX_FILE_BYTES, max_pixels=MAX_PIXELS) as opened:
        opened.load()
        oriented = ImageOps.exif_transpose(opened)
        mode = "RGBA" if "A" in oriented.getbands() or "transparency" in oriented.info else "RGB"
        source = oriented.convert(mode)
        # Preserve PNG text (including original generation parameters), ICC and
        # EXIF; fail rather than silently dropping oversized data.
        details = (capture_png_metadata(oriented, limit=MAX_METADATA_BYTES, source_mode=opened.mode)
                   if metadata else {})
        return source, details


def _png(image: Image.Image, metadata: dict[str, Any] | None = None) -> bytes:
    return encode_png(image, metadata)


def prepare_hand_repair(request: Any) -> PreparedHandRepair:
    """Prepare a reset crop; white/opaque mask pixels are the editable area.

    Source orientation is normalized exactly once before comparing dimensions to
    the canvas mask. The original upload's SHA-256 is retained as provenance.
    All masked RGB pixels are replaced before resampling, then replaced again at
    working resolution so the malformed shape cannot leak through interpolation.
    """
    if not isinstance(request, dict):
        raise ValueError("손 재구성 요청은 객체여야 합니다.")
    settings = _settings_from(request.get("settings"))
    values = [request.get(key) for key in ("image", "mask")]
    if any(not isinstance(value, str) for value in values) or sum(map(len, values)) > MAX_REQUEST_CHARS:
        raise ValueError("원본과 마스크의 총 전송 크기는 128 MB 이하여야 합니다.")
    source_bytes = _decode_data_url(values[0], "원본")
    mask_bytes = _decode_data_url(values[1], "마스크")
    source, metadata = _read_raster(source_bytes, "원본", metadata=True)
    raw_mask, _ = _read_raster(mask_bytes, "마스크")
    if raw_mask.size != source.size:
        raise ValueError("마스크와 원본의 표시 해상도가 같아야 합니다. 자동 리사이즈하지 않습니다.")
    luminance = raw_mask.convert("L")
    if "A" in raw_mask.getbands():
        luminance = ImageChops.multiply(luminance, raw_mask.getchannel("A"))
    edit_mask = luminance.point(lambda value: 255 if value >= 128 else 0)
    bounds = edit_mask.getbbox()
    if bounds is None:
        raise ValueError("재구성할 손 영역을 흰색 마스크로 먼저 칠해 주세요.")
    if edit_mask.getextrema()[0] == 255:
        raise ValueError("전체 이미지 마스크는 사용할 수 없습니다. 손 주변의 원본 문맥을 남겨 주세요.")
    width, height = source.size
    padding = settings["padding"]
    bbox = (max(0, bounds[0] - padding), max(0, bounds[1] - padding),
            min(width, bounds[2] + padding), min(height, bounds[3] + padding))
    crop_mask = edit_mask.crop(bbox)
    if crop_mask.getextrema()[0] == 255:
        raise ValueError("재구성 영역에 원본 문맥이 없습니다. 주변 여백(padding)을 늘려 주세요.")
    crop = source.convert("RGB").crop(bbox)
    crop.paste((127, 127, 127), mask=crop_mask)
    crop_width, crop_height = crop.size
    resolution = settings["resolution"]
    scale = resolution / max(crop_width, crop_height)
    content_size = (max(1, round(crop_width * scale)), max(1, round(crop_height * scale)))
    working_size = tuple(max(64, math.ceil(size / 64) * 64) for size in content_size)
    offset = tuple((outer - inner) // 2 for outer, inner in zip(working_size, content_size))
    content_box = (*offset, offset[0] + content_size[0], offset[1] + content_size[1])
    resized = crop.resize(content_size, Image.Resampling.LANCZOS)
    # BOX keeps tiny masked regions when downsampling; final composition is still
    # bounded by the exact original binary mask, never this resized approximation.
    resized_mask = crop_mask.resize(content_size, Image.Resampling.BOX if scale < 1 else Image.Resampling.NEAREST)
    resized_mask = resized_mask.point(lambda value: 255 if value else 0)
    if resized_mask.getbbox() is None:
        raise ValueError("마스크가 재구성 해상도에서 너무 작습니다. 손 영역을 더 넓게 선택하세요.")
    resized.paste((127, 127, 127), mask=resized_mask)
    init = Image.new("RGB", working_size, (127, 127, 127))
    init.paste(resized, offset)
    working_mask = Image.new("L", working_size, 0)
    working_mask.paste(resized_mask, offset)
    init_png = _png(init)
    return PreparedHandRepair(
        source=source, source_metadata=metadata, edit_mask=edit_mask, bbox=bbox,
        working_size=working_size, content_box=content_box, settings=settings,
        source_sha256=hashlib.sha256(source_bytes).hexdigest(), init_png=init_png,
        mask_png=_png(working_mask), prepared_png=init_png,
    )


def _inward_feather_mask(mask: Image.Image, feather: int, box: tuple[int, int, int, int]) -> Image.Image:
    """Make a true inward-only ramp, in original-image pixel coordinates.

    A clipped Gaussian still starts near 50% opacity at a straight mask edge.
    Instead, successive one-pixel erosions measure an 8-neighbour interior
    distance. Boundary pixels stay at zero; opacity increases linearly to 255
    over ``feather`` pixels. This also protects holes and disconnected regions.
    The halo avoids creating artificial edges when the context crop is tight;
    computation is restricted to that crop rather than the full source image.
    """
    if not feather:
        return mask.crop(box)
    halo = (max(0, box[0] - feather), max(0, box[1] - feather),
            min(mask.width, box[2] + feather), min(mask.height, box[3] + feather))
    eroded = mask.crop(halo)
    ramp = Image.new("L", eroded.size, 0)
    erosion = ImageFilter.MinFilter(3)
    for depth in range(1, feather + 1):
        eroded = eroded.filter(erosion)
        ramp.paste(round(255 * depth / feather), mask=eroded)
    return ramp.crop((box[0] - halo[0], box[1] - halo[1], box[2] - halo[0], box[3] - halo[1]))


def compose_hand_candidate(prepared: PreparedHandRepair, generated_png: bytes, *, provenance: dict) -> bytes:
    """Composite one size-checked candidate without changing outside-mask pixels.

    Softening ramps from zero at the boundary *inside* the original mask. Source
    alpha is retained at every pixel, including the edited area. Letterboxing is
    removed before the candidate is mapped back into the original context crop.
    """
    if not isinstance(prepared, PreparedHandRepair):
        raise ValueError("먼저 검증된 손 재구성 입력을 준비하세요.")
    generated, _ = _read_raster(generated_png, "생성 결과")
    if generated.size != prepared.working_size:
        raise ValueError(f"생성 결과 해상도가 요청한 {prepared.working_size[0]}×{prepared.working_size[1]}와 다릅니다.")
    if not isinstance(provenance, dict):
        raise ValueError("생성 이력은 객체여야 합니다.")
    try:
        run = json.dumps(provenance, ensure_ascii=False, allow_nan=False)
    except (ValueError, TypeError, RecursionError) as exc:
        raise ValueError("생성 이력은 유효한 JSON이어야 합니다.") from exc
    if len(run.encode("utf-8")) > MAX_METADATA_BYTES:
        raise ValueError("생성 이력이 1 MB를 넘습니다.")
    box = prepared.bbox
    original = prepared.source.crop(box)
    candidate = generated.convert("RGB").crop(prepared.content_box).resize(original.size, Image.Resampling.LANCZOS)
    alpha = _inward_feather_mask(prepared.edit_mask, prepared.settings["feather"], box)
    repaired = Image.composite(candidate, original.convert("RGB"), alpha)
    if "A" in prepared.source.getbands():
        repaired.putalpha(original.getchannel("A"))
    result = prepared.source.copy()
    result.paste(repaired, box)
    record = {
        "version": 1, "experimental": True, "method": "masked-neutral-reconstruction",
        "source_sha256": prepared.source_sha256, "source_size": list(prepared.source.size),
        "mask_sha256": hashlib.sha256(prepared.edit_mask.tobytes()).hexdigest(),
        "bbox": list(box), "working_size": list(prepared.working_size),
        "content_box": list(prepared.content_box), "settings": prepared.settings,
        "outside_mask": "unchanged", "original_alpha_preserved": True,
        "feather_mode": "inward-distance-ramp",
        "run": json.loads(run),
    }
    details = {**prepared.source_metadata, "text": dict(prepared.source_metadata.get("text", {}))}
    details["text"]["ai_studio_hand_reconstruction"] = json.dumps(record, ensure_ascii=False, allow_nan=False)
    return _png(result, details)
