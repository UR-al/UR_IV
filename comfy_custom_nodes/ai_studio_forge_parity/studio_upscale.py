"""Upscaling the way Forge does it, for ``ForgeNeoHiresFix`` and ``AIStudioUpscaleImage``.

The rules come from Forge's ``modules/upscaler.py``, ``modules/processing.py`` and ``scripts/postprocessing_upscale.py``: the
model runs again until the picture reaches the target (4 passes at most), Lanczos makes up the rest, and the target is a
multiple of 8 (Hires: of the Resolution Step). Core's ``ImageUpscaleWithModel`` hard-codes its tiles, so ``model_pass`` is
that node with the tile settings as arguments.

ComfyUI, torch, cv2 and skimage imports stay inside functions: the arithmetic runs in the app's test process.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

from .compat import filename_choices, invoke_provider

MODEL_PASSES = 4  # Forge UPSCALE_ITERATIONS
SIZE_GRID = 8  # Forge Upscaler.upscale rounds its target to this
DEFAULT_TILE = 512  # core ImageUpscaleWithModel's numbers
DEFAULT_OVERLAP = 32
MIN_TILE = 128  # core gives up below this when memory runs out
SCALE_TO = "Scale to"
NO_SECOND_MODEL = "None"
PIXEL_RESIZERS = {"Lanczos": "lanczos", "Nearest": "nearest-exact"}  # names of the upscaler lists that need no model file
LATENT_ANTIALIASED = {"bilinear antialiased": "bilinear", "bicubic antialiased": "bicubic"}
LATENT_METHODS = ("nearest", *LATENT_ANTIALIASED)  # beyond the five latent methods the Hires node always had


def snap(value: float, step: int) -> int:
    """Forge ``sRound``: the nearest multiple of ``step``, halves up."""
    return math.floor(value / step + 0.5) * step


def hires_size(width: int, height: int, scale: float, resize_width: int, resize_height: int, step: int,
               grid: int = SIZE_GRID) -> tuple[int, int]:
    """Forge's Hires target: ``scale`` when no size is given; one side alone keeps the aspect. ``step`` is the Resolution
    Step, ``grid`` the latent's cell in pixels (the node can only make multiples of it)."""
    if not resize_width and not resize_height:
        target = (width * scale, height * scale)
    elif not resize_height:
        target = (resize_width, resize_width * (height / width))
    elif not resize_width:
        target = (resize_height * (width / height), resize_height)
    else:
        target = (resize_width, resize_height)

    def side(value: float) -> int:
        return max(grid, max(step, snap(value, step)) // grid * grid)

    return side(target[0]), side(target[1])


@dataclass(frozen=True)
class ExtrasPlan:
    """What Extras does to one picture: the scale it upscales by, and the box it then pastes into (None = no box)."""

    scale: float
    crop_to: tuple[int, int] | None


def limit_side(width: float, height: float, limit: int) -> tuple[int, int]:
    """Max Side Length: the longer side becomes ``limit``, the other shrinks with it and rounds down."""
    if max(width, height) <= limit:
        return int(width), int(height)
    return (int(limit * width // height), limit) if height > width else (limit, int(limit * height // width))


def extras_plan(width: int, height: int, mode: str, scale: float, max_side: int, to_width: int, to_height: int,
                crop: bool) -> ExtrasPlan:
    """Forge's Extras rule for a ``width`` x ``height`` picture."""
    if mode == SCALE_TO:
        return ExtrasPlan(max(to_width / width, to_height / height), (to_width, to_height) if crop else None)
    if max_side and max(width, height) * scale > max_side:
        limited_w, limited_h = limit_side(width * scale, height * scale, max_side)
        return ExtrasPlan(max(limited_w / width, limited_h / height), None)
    return ExtrasPlan(scale, None)


def model_files() -> list[str]:
    """The upscale model files; ``filename_choices`` answers ``["None"]`` for an empty folder."""
    return [name for name in filename_choices("upscale_models") if name != NO_SECOND_MODEL]


def load_model(name: str):
    return invoke_provider("UpscaleModelLoader", method="load_model", feature="Studio upscale", args=(name,))[0]


def _model_tiles(upscale_model, image, tile: int, overlap: int):
    """The tiled run of core's node; memory running out halves the tiles, as there."""
    import comfy.model_management as model_management
    import comfy.utils

    pixels = image.movedim(-1, -3).to(upscale_model.patcher.load_device)
    height, width = pixels.shape[-2:]
    tile_size = tile if tile > 0 else max(height, width)  # Forge 0 = no tiling: one tile the size of the picture
    while True:
        overlap_size = min(overlap, tile_size // 2)
        try:
            steps = comfy.utils.get_tiled_scale_steps(width, height, tile_x=tile_size, tile_y=tile_size, overlap=overlap_size)
            return comfy.utils.tiled_scale(
                pixels, lambda a: upscale_model(a.float()), tile_x=tile_size, tile_y=tile_size, overlap=overlap_size,
                upscale_amount=upscale_model.scale, pbar=comfy.utils.ProgressBar(pixels.shape[0] * steps),
                output_device=model_management.intermediate_device(),
            )
        except Exception as error:
            model_management.raise_non_oom(error)
            tile_size //= 2
            if tile_size < MIN_TILE:
                raise


def model_pass(upscale_model, image, tile: int, overlap: int):
    """Core ``ImageUpscaleWithModel`` with ``tile`` and ``overlap``: the same memory request, retry, clamp and alpha."""
    import comfy.model_management as model_management
    import comfy.utils
    import torch

    if not (hasattr(upscale_model, "patcher") and hasattr(model_management, "raise_non_oom")):  # an older ComfyUI: its fixed tiles
        return invoke_provider("ImageUpscaleWithModel", method="upscale", feature="Studio upscale", args=(upscale_model, image))[0]
    alpha = image[..., 3:4] if image.shape[-1] == 4 else None
    colour = image[..., :3]
    memory = (512 * 512 * 3) * colour.element_size() * max(upscale_model.scale, 1.0) * 384.0  # core's estimate
    memory += colour.nelement() * colour.element_size()
    model_management.load_models_gpu([upscale_model.patcher], memory_required=memory, force_full_load=True)
    upscaled = _model_tiles(upscale_model, colour, int(tile), int(overlap))
    upscaled = torch.clamp(upscaled.movedim(-3, -1), min=0, max=1.0).to(model_management.intermediate_dtype())
    if alpha is None:
        return upscaled
    grown = comfy.utils.common_upscale(alpha.movedim(-1, -3).to(upscaled), upscaled.shape[2], upscaled.shape[1], "bilinear", "disabled")
    return torch.cat((upscaled, grown.movedim(-3, -1)), dim=-1)


def resize(image, width: int, height: int, method: str):
    import comfy.utils

    return comfy.utils.common_upscale(image.movedim(-1, 1), width, height, method, "disabled").movedim(1, -1)


def resize_latent(samples, width: int, height: int, method: str):
    """``comfy.utils.common_upscale`` plus Forge's antialiased latent modes (a 5-D latent is folded the same way)."""
    import comfy.utils
    import torch

    mode = LATENT_ANTIALIASED.get(method)
    if mode is None:
        return comfy.utils.common_upscale(samples, width, height, method, "disabled")
    shape = tuple(samples.shape)
    flat = samples if len(shape) == 4 else samples.movedim(2, 1).reshape(-1, shape[1], shape[-2], shape[-1])
    resized = torch.nn.functional.interpolate(flat, size=(height, width), mode=mode, antialias=True)
    if len(shape) == 4:
        return resized
    return resized.reshape(shape[0], shape[2], shape[1], height, width).movedim(1, 2)


def _size(image) -> tuple[int, int]:
    return int(image.shape[2]), int(image.shape[1])


def _grid_target(image, scale: float) -> tuple[int, int]:
    return tuple(round(side * scale / SIZE_GRID) * SIZE_GRID for side in _size(image))


def _covers(size: tuple[int, int], target: tuple[int, int]) -> bool:
    return size[0] >= target[0] and size[1] >= target[1]


def upscale(image, name: str, scale: float, tile: int, overlap: int):
    """Forge's rule: the model until the picture covers the target or stops growing, then Lanczos to the exact size."""
    target = _grid_target(image, scale)
    if name in PIXEL_RESIZERS:
        return resize(image, *target, PIXEL_RESIZERS[name])
    model = load_model(name)
    passes = 0
    grown = image
    while passes < MODEL_PASSES:
        previous = _size(grown)
        grown = model_pass(model, grown, tile, overlap)
        passes += 1
        if _size(grown) == previous or _covers(_size(grown), target):
            break
    return grown if _size(grown) == target else resize(grown, *target, "lanczos")


def upscale_to(image, name: str, width: int, height: int, tile: int, overlap: int):
    """Forge's Hires rule: the upscaler only when the picture must grow, then Lanczos to the exact size."""
    source_width, source_height = _size(image)
    scale = max(width / source_width, height / source_height)
    if scale > 1.0:
        image = upscale(image, name, scale, tile, overlap)
    return image if _size(image) == (width, height) else resize(image, width, height, "lanczos")


def paste_centered(image, width: int, height: int):
    """Forge pastes the result in the middle of a black ``width`` x ``height`` canvas: a centre crop, black where it falls short."""
    import torch

    columns, rows = _size(image)
    left, top = width // 2 - columns // 2, height // 2 - rows // 2
    from_x, from_y = max(0, -left), max(0, -top)
    to_x, to_y = max(0, left), max(0, top)
    copy_w, copy_h = min(columns - from_x, width - to_x), min(rows - from_y, height - to_y)
    canvas = torch.zeros((image.shape[0], height, width, image.shape[3]), dtype=image.dtype, device=image.device)
    canvas[:, to_y:to_y + copy_h, to_x:to_x + copy_w, :] = image[:, from_y:from_y + copy_h, from_x:from_x + copy_w, :]
    return canvas


def blend(first, second, visibility: float):
    """PIL ``Image.blend``: ``first * (1 - visibility) + second * visibility``."""
    return first + (second - first) * visibility


def require_opencv() -> None:
    """Colour correction works in OpenCV's Lab; ComfyUI does not ship it, and a late failure would lose the upscale."""
    try:
        import cv2  # noqa: F401
    except ImportError as error:
        raise RuntimeError("색 보정에는 opencv-python 이 필요함: ComfyUI 의 Python 에 설치하거나 색 보정을 끄세요") from error


def match_channel(channel, reference):
    """One uint8 channel with the histogram of ``reference``: each value moves to the reference value of the same
    quantile (what ``skimage.exposure.match_histograms`` does; ComfyUI does not ship scikit-image)."""
    import numpy as np

    present = np.flatnonzero(np.bincount(reference.ravel(), minlength=256))
    counts = np.bincount(reference.ravel(), minlength=256)[present]
    quantiles = np.cumsum(np.bincount(channel.ravel(), minlength=256)) / channel.size
    moved = np.interp(quantiles, np.cumsum(counts) / reference.size, present)
    return moved[channel].astype(np.uint8)


def match_colors_uint8(image, source):
    """Forge's Color Correction for one uint8 RGB picture: histograms matched channel by channel in Lab."""
    import cv2
    import numpy as np

    lab, reference = cv2.cvtColor(image, cv2.COLOR_RGB2LAB), cv2.cvtColor(source, cv2.COLOR_RGB2LAB)
    matched = np.stack([match_channel(lab[..., index], reference[..., index]) for index in range(3)], axis=2)
    return cv2.cvtColor(matched, cv2.COLOR_Lab2RGB)


def match_colors(image, source):
    """``match_colors_uint8`` for each picture of an IMAGE batch; ``source`` is the batch before the upscale."""
    import numpy as np
    import torch

    def to_uint8(picture):
        return np.clip(255.0 * picture[..., :3].cpu().numpy(), 0, 255).astype(np.uint8)

    matched = [match_colors_uint8(to_uint8(picture), to_uint8(source[index % len(source)])) for index, picture in enumerate(image)]
    colour = torch.from_numpy(np.stack(matched).astype(np.float32) / 255.0).to(image)
    return torch.cat((colour, image[..., 3:]), dim=-1) if image.shape[-1] > 3 else colour
