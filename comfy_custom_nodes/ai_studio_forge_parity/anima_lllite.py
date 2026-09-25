"""Anima ControlNet-LLLite for the AI Studio ComfyUI pack (Tile & Repair, SAM3 slot).

Two consumers share one application path, the pinned kohya node
``vendor/comfyui_anima_lllite`` (Apache-2.0, see ``UPSTREAM.md``):

* ``ForgeNeoAnimaTileRepair`` runs the Anima Tile & Repair pipeline of civitai
  2708551 the way kohya sd-scripts ``anima_minimal_inference_control_net_lllite.py``
  does: source-aspect latent from pure noise (denoise 1.0) drawn like the script
  (CPU Generator, bf16), flow shift 5.0, Euler (Comfy ``euler``) 50 steps on the
  script's own shifted-linspace σ, CFG 3.5, negative ``""``, the
  LLLite on at every step, then a VAE decode. The Forge extension runs the same
  script (``sam3ext/anima_core.run_tile_repair``), so the defaults and the size
  rule below are the extension's too.
* The SAM3 Detailer/Refine ControlNet slot (``sam3_nodes._load_controlnet``)
  recognises an Anima LLLite file by its safetensors header and applies it as
  this MODEL patch instead of handing it to Comfy's ``ControlNetLoader`` (which
  rejects it: 'controlnet file is invalid'). The mask reaches the node only for
  4-channel (inpaint) weights, exactly the case the original node requires it.

Everything at import time is stdlib only: the header reader needs no torch, and
the vendored node (torch, ``folder_paths``) is imported when a node runs.
"""
from __future__ import annotations

import contextlib
import importlib
import json
import logging
import math
import os
import re
import struct
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .compat import (
    folder_paths_module,
    invoke_provider,
    require_torch,
)


CATEGORY = "AI Studio/Forge Neo parity/Anima"
LOGGER = logging.getLogger("ai_studio_forge_parity")

# ── the original's constants (kohya-ss/ComfyUI-Anima-LLLite@b7495bd8) ────────────────
# Saved v2 weight format: the shared conditioning trunk is stored under this prefix
# (control_net_lllite_anima.py:484 ``_SAVED_COND_PREFIX``). SDXL ControlNet-LLLite
# (``lllite_unet_*``), ordinary ControlNets and the legacy ``lllite_modules.*`` layout
# never carry it.
LLLITE_COND_PREFIX = "lllite_conditioning1."
# nodes.py:168 — ``int(meta.get("lllite.cond_in_channels", 3))``; 4 = inpaint (:172-177).
META_COND_IN_CHANNELS = "lllite.cond_in_channels"
DEFAULT_COND_IN_CHANNELS = 3
INPAINT_COND_IN_CHANNELS = 4
# nodes.py:128, :148 — the node reads weights from Comfy's ``controlnet`` folder.
CONTROLNET_FOLDER = "controlnet"
# Tile & Repair header title: v1.0 'anima_tiled_lllite_v1', v2.0 'anima_tile_multitask_v1'
# (same rule as the extension's sam3ext/sam3_cn_lllite.py ``lllite_tile_repair_from_header``).
META_TITLE = "modelspec.title"
TILE_REPAIR_MARK = "tile"
NONE = "None"

# nodes.py:130-133 — the original node's inputs, repeated here unchanged.
LLLITE_APPLY_INPUTS = {
    "strength": ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.01}),
    "start_percent": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.001}),
    "end_percent": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.001}),
    "preserve_wrapper": ("BOOLEAN", {"default": True}),
}

# ── Tile & Repair pipeline (kohya-ss/sd-scripts@690ea7f9:anima_minimal_inference.py) ──
TILE_REPAIR_CFG = 3.5            # :82-84 --guidance_scale
TILE_REPAIR_NEGATIVE = ""        # :86 --negative_prompt
TILE_REPAIR_SHORT_SIDE = 1024    # :87 --image_size 1024 1024 (short side of the output)
TILE_REPAIR_STEPS = 50           # :88 --infer_steps
TILE_REPAIR_FLOW_SHIFT = 5.0     # :93-98 --flow_shift
TILE_REPAIR_SIZE_MULTIPLE = 32   # :226-227 check_inputs
TILE_REPAIR_DENOISE = 1.0        # :546-552 pure noise start (no img2img)
# The script's Euler flow step (hunyuan_image_utils.step: x − (σᵢ − σᵢ₊₁)·v) = Comfy ``euler``
# on a flow model. Its σ list is the script's own (``tile_repair_sigmas``), not a Comfy
# scheduler: ``simple`` indexes a 1000-entry table and drifts from the shifted linspace
# whenever 1000 is not a multiple of the step count (and repeats σ above 1000 steps).
TILE_REPAIR_SAMPLER = "euler"
# The extension's floor (forge_sam3_extension@861ac02:sam3ext/anima_core.py TILE_REPAIR_MIN_SIDE).
TILE_REPAIR_MIN_SIDE = 256
TILE_REPAIR_MAX_SHORT_SIDE = 4096
# civitai 2708551 model card's suggested repair prompt = the extension panel default.
TILE_REPAIR_PROMPT = (
    "repair the low-quality anime image, reduce blur and compression artifacts, "
    "preserve the original composition"
)

# safetensors caps its JSON header at 100 MB; anything larger is not a real file.
_MAX_HEADER_BYTES = 100 * 1024 * 1024
_VENDOR_NODES = "vendor.comfyui_anima_lllite.nodes"
_CONTROL_MODES = frozenset({
    "Balanced", "My prompt is more important", "ControlNet is more important",
})


# ─────────────────────────────────────────────────────────────────────────────────────
# Header inspection (no torch, no tensor read)
# ─────────────────────────────────────────────────────────────────────────────────────

def read_safetensors_header(path: Any) -> dict | None:
    """The JSON header of a ``.safetensors`` file, or ``None`` when unreadable.

    Reads the 8-byte little-endian length and the JSON after it — never a weight.
    """
    if not path:
        return None
    try:
        file = Path(str(path))
        if file.suffix.casefold() != ".safetensors" or not file.is_file():
            return None
        with file.open("rb") as handle:
            raw_length = handle.read(8)
            if len(raw_length) != 8:
                return None
            (length,) = struct.unpack("<Q", raw_length)
            if length <= 0 or length > _MAX_HEADER_BYTES:
                return None
            blob = handle.read(length)
        if len(blob) != length:
            return None
        header = json.loads(blob.decode("utf-8"))
    except (OSError, ValueError, UnicodeDecodeError):
        return None
    return header if isinstance(header, dict) else None


def _flat(value: Any) -> str:
    text = "" if value is None else str(value).strip().casefold()
    return re.sub(r"[^0-9a-z]", "", text)


@dataclass(frozen=True)
class LLLiteInfo:
    """What the original node reads from an Anima LLLite header before it loads it.

    ``channels`` is ``None`` when ``lllite.cond_in_channels`` is not an integer — the
    original node's ``int(...)`` then raises when it runs.
    """

    channels: int | None
    tile_repair: bool


def lllite_info_from_header(header: Any, name: Any = "") -> LLLiteInfo | None:
    """Header → Anima ControlNet-LLLite info, or ``None`` for any other file."""
    if not isinstance(header, dict):
        return None
    if not any(
        isinstance(key, str) and key.startswith(LLLITE_COND_PREFIX)
        for key in header
        if key != "__metadata__"
    ):
        return None
    meta = header.get("__metadata__")
    if not isinstance(meta, dict):
        meta = {}
    try:
        channels: int | None = int(meta.get(META_COND_IN_CHANNELS, DEFAULT_COND_IN_CHANNELS))
    except (TypeError, ValueError):
        channels = None
    tile_repair = False
    if channels == DEFAULT_COND_IN_CHANNELS:
        title = meta.get(META_TITLE)
        if title is not None and str(title).strip():
            tile_repair = TILE_REPAIR_MARK in _flat(title)
        else:
            tile_repair = TILE_REPAIR_MARK in _flat(Path(str(name or "")).name)
    return LLLiteInfo(channels=channels, tile_repair=tile_repair)


_HEADER_CACHE: dict[tuple[str, int, int], LLLiteInfo | None] = {}
_HEADER_CACHE_LOCK = threading.Lock()
_HEADER_CACHE_LIMIT = 512


def lllite_info(path: Any) -> LLLiteInfo | None:
    """``lllite_info_from_header`` for a file, cached by (path, size, mtime)."""
    try:
        file = Path(str(path))
        stat = file.stat()
    except (OSError, TypeError, ValueError):
        return None
    key = (str(file), int(stat.st_size), int(stat.st_mtime_ns))
    with _HEADER_CACHE_LOCK:
        if key in _HEADER_CACHE:
            return _HEADER_CACHE[key]
    info = lllite_info_from_header(read_safetensors_header(file), file.name)
    with _HEADER_CACHE_LOCK:
        if len(_HEADER_CACHE) >= _HEADER_CACHE_LIMIT:
            _HEADER_CACHE.clear()
        _HEADER_CACHE[key] = info
    return info


# ─────────────────────────────────────────────────────────────────────────────────────
# Controlnet-folder resolution and the one call into the original node
# ─────────────────────────────────────────────────────────────────────────────────────

def _is_none_choice(value: Any) -> bool:
    return str(value or "").strip().casefold() in {"", "none", "null"}


def resolve_controlnet_file(requested: Any) -> Path | None:
    """An absolute path, or a name in Comfy's ``controlnet`` folder → the file (or None)."""
    if _is_none_choice(requested):
        return None
    text = str(requested).strip()
    path = Path(text).expanduser()
    if path.is_absolute():
        return path if path.is_file() else None
    try:
        folder_paths = folder_paths_module()
        full = folder_paths.get_full_path(CONTROLNET_FOLDER, text)
    except Exception:
        return None
    return Path(full) if full else None


def _same_file(left: Any, right: Path) -> bool:
    if not left:
        return False
    try:
        return os.path.samefile(str(left), str(right))
    except OSError:
        return False


@contextlib.contextmanager
def _controlnet_folder_name(path: Path, requested: str):
    """The name under which the original node's ``folder_paths.get_full_path`` finds ``path``.

    A name from the ``controlnet`` folder is used as it is. An absolute path (the app
    sends resolved model paths) puts its parent first in that folder list for the
    duration of the call, the way ``sam3_nodes._load_controlnet`` does for
    ``ControlNetLoader``.
    """
    folder_paths = folder_paths_module()
    if not Path(requested).expanduser().is_absolute():
        if _same_file(folder_paths.get_full_path(CONTROLNET_FOLDER, requested), path):
            yield requested
            return
    previous = folder_paths.folder_names_and_paths.get(CONTROLNET_FOLDER)
    if previous is None:
        raise RuntimeError("ComfyUI did not register its ControlNet model folder.")
    paths, extensions = previous
    folder_paths.folder_names_and_paths[CONTROLNET_FOLDER] = (
        [str(path.parent), *[str(item) for item in paths]], extensions,
    )
    try:
        yield path.name
    finally:
        folder_paths.folder_names_and_paths[CONTROLNET_FOLDER] = previous


def _vendored_nodes():
    return importlib.import_module(f"{__package__}.{_VENDOR_NODES}")


def apply_lllite(model: Any, lllite_name: str, image: Any, strength: float,
                 start_percent: float, end_percent: float, *,
                 preserve_wrapper: bool = True, mask: Any = None) -> Any:
    """``AnimaLLLiteApply_sdscripts().apply(...)`` of the pinned original → patched MODEL."""
    node = _vendored_nodes().AnimaLLLiteApply_sdscripts()
    output = node.apply(
        model, lllite_name, image, float(strength), float(start_percent),
        float(end_percent), preserve_wrapper=bool(preserve_wrapper), mask=mask,
    )
    return output[0]


# ─────────────────────────────────────────────────────────────────────────────────────
# SAM3 Detailer/Refine ControlNet slot
# ─────────────────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class AnimaLLLiteControl:
    """An Anima LLLite picked in the SAM3 ControlNet slot — applied as a MODEL patch."""

    path: Path
    name: str
    channels: int | None
    tile_repair: bool

    def _checked_report(self, strength: float, start: float, end: float,
                        control_mode: str) -> dict:
        """Validate a pass's settings like a ControlNet's → the report fields.

        Forge's built-in LLLite patcher has no control mode; the setting is
        validated like a ControlNet's and reported as not applicable.
        """
        mode = str(control_mode)
        if mode not in _CONTROL_MODES:
            raise ValueError(f"Unknown ControlNet control_mode {mode!r}")
        requested_strength = float(strength)
        if not 0.0 <= requested_strength <= 10.0:
            raise ValueError("ControlNet strength must be between zero and 10")
        if not 0.0 <= float(start) < float(end) <= 1.0:
            raise ValueError("ControlNet guidance must satisfy 0 <= start < end <= 1")
        return {
            "mode": mode,
            "translation": "anima_lllite_model_patch",
            "requested_strength": requested_strength,
            "effective_strength": requested_strength,
            "lllite_cond_in_channels": self.channels,
            "lllite_tile_repair": self.tile_repair,
        }

    def patch_model(self, model: Any, image: Any, mask: Any, strength: float,
                    start: float, end: float, control_mode: str = "Balanced"):
        """(patched MODEL, report) for one detailer pass of ONE image.

        ``image`` is the pass's control image (the region at its sample size) and
        ``mask`` its pass mask. The original node needs the mask for 4-channel
        weights and ignores (warns about) it otherwise, so it is passed only then.
        The original node keeps only the first frame of both (nodes.py:75, :99), so a
        batch goes through ``sample_per_image``.
        """
        report = self._checked_report(strength, start, end, control_mode)
        mask_value = mask if self.channels == INPAINT_COND_IN_CHANNELS else None
        with _controlnet_folder_name(self.path, self.name) as lllite_name:
            patched = apply_lllite(
                model, lllite_name, image, report["requested_strength"], float(start),
                float(end), preserve_wrapper=True, mask=mask_value,
            )
        report["lllite_mask"] = mask_value is not None
        return patched, report

    def per_image_report(self, image: Any, mask: Any, strength: float, start: float,
                         end: float, control_mode: str = "Balanced") -> dict:
        """The report of a ``sample_per_image`` pass, validated before anything is sampled."""
        report = self._checked_report(strength, start, end, control_mode)
        report["lllite_mask"] = mask is not None and self.channels == INPAINT_COND_IN_CHANNELS
        report["lllite_per_image"] = int(image.shape[0])
        return report

    def sample_per_image(self, model: Any, image: Any, mask: Any, strength: float,
                         start: float, end: float, control_mode: str, latent: dict,
                         sample: Any) -> dict:
        """Sample a batch pass one image at a time, each under its own LLLite patch.

        kohya's node conditions on the first frame only (ComfyUI-Anima-LLLite@b7495bd8
        nodes.py:75 ``img = img[:1]``, :99 ``m = m[:1]``) and the LLLite module repeats
        that one embedding over the whole runtime batch (control_net_lllite_anima.py:
        266-270), so one patch for a B-image pass would control every image by image 0.
        Forge runs SAM3 once per image (``postprocess_image``) and Comfy's ControlNet
        path gives each image its own hint, so here item ``i`` gets
        ``patch_model(model, image[i:i+1], mask[i:i+1])`` and ``sample(patched,
        latent_item(latent, i))``; the decoded batch is the items concatenated. Only one
        LLLite is alive at a time.
        """
        torch = require_torch()
        outputs = []
        for index in range(int(image.shape[0])):
            item_mask = None if mask is None else mask[index:index + 1]
            patched, _report = self.patch_model(
                model, image[index:index + 1], item_mask, strength, start, end, control_mode,
            )
            outputs.append(sample(patched, latent_item(latent, index))["samples"])
            del patched
        output = dict(latent)
        output["samples"] = torch.cat(outputs, dim=0)
        return output


def is_lllite_control(value: Any) -> bool:
    return isinstance(value, AnimaLLLiteControl)


def lllite_control_for(requested: Any) -> AnimaLLLiteControl | None:
    """The SAM3 ControlNet slot's model name → an LLLite handle, or None (not an LLLite)."""
    path = resolve_controlnet_file(requested)
    if path is None:
        return None
    info = lllite_info(path)
    if info is None:
        return None
    text = str(requested).strip()
    name = path.name if Path(text).expanduser().is_absolute() else text
    return AnimaLLLiteControl(
        path=path, name=name, channels=info.channels, tile_repair=info.tile_repair,
    )


def forced_control_module(module: Any, control: AnimaLLLiteControl) -> tuple[str, str | None]:
    """(preprocessor to use, reason when it was changed) for an Anima LLLite.

    Mirrors the extension (forge_sam3_extension sam3ext/sam3_cn_lllite.py
    ``forced_cn_module``) and the app (core/sam3_cn_names.py): the original feeds the
    LLLite the user's control image as it is.
      * Tile & Repair takes the unprocessed image to repair → always ``None``.
      * Any other Anima LLLite → only ``inpaint_*`` becomes ``None`` (it blanks the
        masked region into the control image; 3-channel weights ignore the mask and
        4-channel ones take it as their own channel).
    An empty module is the detailer's default ``inpaint_only``.
    """
    current = str(module or "inpaint_only").strip()
    folded = current.casefold()
    if folded in {"none", "raw"}:
        return current, None
    if control.tile_repair:
        return NONE, (
            f"Anima Tile & Repair ControlNet-LLLite '{control.name}' takes the unprocessed "
            f"image to repair (like kohya sd-scripts / ComfyUI-Anima-LLLite); preprocessor "
            f"'{current}' was replaced by 'None'."
        )
    if not folded.startswith("inpaint"):
        return current, None
    if control.channels == INPAINT_COND_IN_CHANNELS:
        return NONE, (
            f"4-channel Anima ControlNet-LLLite '{control.name}' takes the mask as its own "
            f"channel; preprocessor '{current}' was replaced by 'None'."
        )
    return NONE, (
        f"{control.channels}-channel Anima ControlNet-LLLite '{control.name}' ignores the "
        f"mask (like kohya ComfyUI-Anima-LLLite); preprocessor '{current}' would blank the "
        f"masked region of the control image and was replaced by 'None'."
    )


_THRESHOLD_KEYS = ("threshold_a", "threshold_b")


def thresholds_for_module(module: Any, settings: dict) -> tuple[dict, dict | None]:
    """(settings for the hint, the thresholds dropped | None) once an LLLite's module is ``None``.

    Forge's ``None`` preprocessor ignores both sliders (modules_forge/supported_preprocessor.py
    ``Preprocessor.__call__`` returns ``input_image``), and the original LLLite node has no
    thresholds at all. The module becomes ``None`` either here (``forced_control_module``)
    or in the app (core/sam3_cn_names.py ``lllite_module_override``), both leaving the
    user's ``threshold_a``/``threshold_b`` of the earlier preprocessor (canny 100/200,
    tile_resample 1.0) in place; the detailer's strict ``none`` check would then refuse a
    state Forge runs. Those thresholds become -1 (unset) for this pass and are reported.
    """
    if str(module or "").strip().casefold() not in {"none", "raw"}:
        return settings, None
    dropped = {
        key: float(settings[key])
        for key in _THRESHOLD_KEYS
        if key in settings and float(settings[key]) >= 0
    }
    if not dropped:
        return settings, None
    return {**settings, **{key: -1.0 for key in _THRESHOLD_KEYS}}, dropped


def latent_item(latent: dict, index: int) -> dict:
    """Batch item ``index`` of a detailer latent as a batch-1 latent.

    Tensors batched like ``samples`` (``samples``, ``noise_mask``) are sliced; a
    batch-1 ``noise_mask`` stays shared. ``batch_index`` becomes ``[index]`` (or the
    latent's own index for that item), so Comfy's ``prepare_noise`` draws this item's
    initial noise from the same seed stream position as in the batched run.
    """
    torch = require_torch()
    batch = int(latent["samples"].shape[0])
    item = {}
    for key, value in latent.items():
        if torch.is_tensor(value) and value.ndim > 0 and int(value.shape[0]) == batch:
            item[key] = value[index:index + 1]
        else:
            item[key] = value
    indices = latent.get("batch_index")
    item["batch_index"] = [int(indices[index]) if indices is not None else int(index)]
    return item


# ─────────────────────────────────────────────────────────────────────────────────────
# Tile & Repair
# ─────────────────────────────────────────────────────────────────────────────────────

def tile_repair_size(src_width: int, src_height: int, short_side: int) -> tuple[int, int]:
    """``(width, height)`` of a Tile & Repair run: the short side becomes ``short_side``,
    the long side follows the source aspect ratio (``int(long * edge / short)``), then
    each side is rounded down to a multiple of 32 (sd-scripts ``check_inputs``) with a
    256 floor. Same rule as the extension's ``anima_core.tile_repair_size``."""
    width, height = int(src_width), int(src_height)
    if width <= 0 or height <= 0:
        raise ValueError(f"source size must be positive, got {width}x{height}")
    edge = int(short_side)
    if width < height:
        new_width, new_height = edge, int(height * (edge / width))
    else:
        new_height, new_width = edge, int(width * (edge / height))

    def snap(value: int) -> int:
        return max(
            TILE_REPAIR_MIN_SIDE,
            (value // TILE_REPAIR_SIZE_MULTIPLE) * TILE_REPAIR_SIZE_MULTIPLE,
        )

    return snap(new_width), snap(new_height)


_NAME_VERSION_RE = re.compile(r"v(\d+(?:[._]\d+)*)")


def _name_version(name: str) -> tuple[int, ...]:
    stem = re.sub(r"\.safetensors$", "", name.lower())
    found = _NAME_VERSION_RE.findall(stem)
    if not found:
        return ()
    return tuple(int(part) for part in re.split(r"[._]", found[-1]))


def default_tile_repair_choice(choices: list[str]) -> str:
    """The newest Tile & Repair file (``animaTileRepair_v20`` over ``_v10``), else the
    first listed LLLite, else ``None`` — the extension's ``default_lllite_choice``."""
    real = [choice for choice in choices if choice and choice != NONE]
    tiles = [choice for choice in real if "tilerepair" in _flat(Path(choice).name)]
    if tiles:
        return max(tiles, key=lambda choice: (_name_version(Path(choice).name), choice.lower()))
    return real[0] if real else NONE


def tile_repair_lllite_choices() -> tuple[list[str], str]:
    """(3-channel Anima LLLite files in Comfy's ``controlnet`` folder, default).

    Read from each file's header, so 4-channel inpaint LLLites (which need a mask) and
    ordinary ControlNets never show up whatever their name — like the extension's
    Tile-Repair panel list.
    """
    try:
        folder_paths = folder_paths_module()
        names = list(folder_paths.get_filename_list(CONTROLNET_FOLDER))
    except Exception:
        return [NONE], NONE
    choices = []
    for name in names:
        if not str(name).casefold().endswith(".safetensors"):
            continue
        try:
            full = folder_paths.get_full_path(CONTROLNET_FOLDER, name)
        except Exception:
            continue
        info = lllite_info(full) if full else None
        if info is not None and info.channels == DEFAULT_COND_IN_CHANNELS:
            choices.append(name)
    if not choices:
        return [NONE], NONE
    return choices, default_tile_repair_choice(choices)


def sdscripts_control_image(image: Any, width: int, height: int):
    """The control image sd-scripts builds (``_load_control_image``, lllite script :65-74):
    RGB, resized to ``(width, height)`` with PIL BICUBIC when the size differs.

    Returned as a Comfy IMAGE in ``[0, 1]``. At a multiple-of-32 size it already equals
    the original node's ``latent × 8`` target, so the node skips its own torch-bicubic
    resize and only applies ``* 2 - 1`` — the script's ``/ 127.5 - 1``.
    """
    torch = require_torch()
    import numpy as np
    from PIL import Image

    frames = []
    for item in image:
        array = (item.detach().float().cpu().clamp(0.0, 1.0).numpy() * 255.0).round()
        picture = Image.fromarray(array.astype(np.uint8))
        if picture.size != (int(width), int(height)):
            picture = picture.resize((int(width), int(height)), Image.BICUBIC)
        frames.append(torch.from_numpy(np.asarray(picture).astype(np.float32) / 255.0))
    return torch.stack(frames, dim=0)


def _rgb_image(image: Any):
    """Comfy IMAGE → float ``[B,H,W,3]`` like PIL ``convert("RGB")`` (grey repeated, alpha dropped)."""
    from .mask_ops import ensure_image

    value = ensure_image(image)
    if value.shape[-1] == 1:
        return value.repeat(1, 1, 1, 3)
    return value[..., :3]


def _encode(clip: Any, text: str):
    return invoke_provider(
        "CLIPTextEncode", method="encode", feature="Anima Tile & Repair prompt",
        kwargs={"clip": clip, "text": str(text)},
    )[0]


def _flow_shift(model: Any, shift: float):
    # Keeps the model's timestep multiplier (1.0 for Anima) — ModelSamplingAuraFlow(shift).
    from .generation import _patch_flow_shift

    return _patch_flow_shift(model, float(shift))


def _empty_latent(width: int, height: int):
    return invoke_provider(
        "EmptyLatentImage", method="generate", feature="Anima Tile & Repair latent",
        kwargs={"width": int(width), "height": int(height), "batch_size": 1},
    )[0]


def tile_repair_sigmas(steps: int, shift: float):
    """The script's σ list: kohya-ss/sd-scripts@690ea7f9 library/hunyuan_image_utils.py:276-292
    ``get_timesteps_sigmas`` (called at anima_minimal_inference.py:565) — ``steps + 1`` points
    of ``linspace(1, 0)`` shifted by ``σ = s·t / (1 + (s − 1)·t)``, float32."""
    torch = require_torch()
    sigmas = torch.linspace(1, 0, int(steps) + 1)
    shift = float(shift)
    sigmas = (shift * sigmas) / (1 + (shift - 1) * sigmas)
    return sigmas.to(torch.float32)


def tile_repair_noise(latent_image: Any, seed: int):
    """The script's initial latent (anima_minimal_inference.py:528-529, :552): a CPU
    ``torch.Generator`` seeded with ``seed`` and diffusers ``randn_tensor(..., dtype=bfloat16)``,
    whose CPU-generator branch is ``torch.randn(shape, generator, device="cpu", dtype=bf16)``.

    Comfy's ``prepare_noise`` draws float32 from ``torch.manual_seed(seed)`` instead: on torch
    2.13 that is this draw before bf16 rounding, on torch 2.11 a different stream altogether.
    The Anima latent here is ``[1, 16, 1, H/8, W/8]`` like the script's ``shape``, so the same
    elements land in the same places. The global RNG is not touched (own Generator).
    """
    torch = require_torch()
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    return torch.randn(
        tuple(latent_image.shape), generator=generator, device="cpu", dtype=torch.bfloat16,
    )


def _sample(model, seed, steps, cfg, flow_shift, positive, negative, latent):
    """sd-scripts ``generate_body`` sampling: its noise and its σ list, Comfy's Euler and CFG.

    ``common_ksampler`` is not used — its noise (fp32 ``prepare_noise``) and its schedulers are
    not the script's. This is the SamplerCustom path (``comfy.sample.sample_custom``) with the
    script's own noise and σ; the latent is fixed to the model's channels like common_ksampler.
    Denoise is 1.0 (``TILE_REPAIR_DENOISE``): σ₀ = 1, so the start is the noise itself.
    """
    require_torch()
    import comfy.sample
    import comfy.samplers
    import comfy.utils
    import latent_preview

    latent_image = comfy.sample.fix_empty_latent_channels(
        model, latent["samples"], latent.get("downscale_ratio_spacial"),
        latent.get("downscale_ratio_temporal"),
    )
    noise = tile_repair_noise(latent_image, seed)
    sigmas = tile_repair_sigmas(steps, flow_shift)
    callback = latent_preview.prepare_callback(model, int(steps))
    samples = comfy.sample.sample_custom(
        model, noise, float(cfg), comfy.samplers.sampler_object(TILE_REPAIR_SAMPLER), sigmas,
        positive, negative, latent_image, noise_mask=latent.get("noise_mask"), callback=callback,
        disable_pbar=not comfy.utils.PROGRESS_BAR_ENABLED, seed=int(seed),
    )
    output = dict(latent)
    output.pop("downscale_ratio_spacial", None)
    output.pop("downscale_ratio_temporal", None)
    output["samples"] = samples
    return output


def _decode(vae: Any, latent: Any):
    return invoke_provider(
        "VAEDecode", method="decode", feature="Anima Tile & Repair decode",
        kwargs={"vae": vae, "samples": latent},
    )[0]


class ForgeNeoAnimaTileRepair:
    """Anima Tile & Repair (civitai 2708551) with the original ControlNet-LLLite node.

    Pipeline per input image: prompt/negative encode → flow shift → LLLite (the pinned
    kohya ``AnimaLLLiteApply_sdscripts``) on the sd-scripts control image → empty
    source-aspect latent → Comfy ``euler`` on sd-scripts' σ list from sd-scripts' bf16
    CPU-generator noise (``_sample``), denoise 1.0 → VAE decode. The shift is patched
    before the LLLite so ``start_percent``/``end_percent`` map to sigmas of the schedule
    that is actually sampled (Comfy's own percent semantics); at the default 0/1 window
    the LLLite is on for every step, as in sd-scripts.
    """

    @classmethod
    def INPUT_TYPES(cls):
        choices, default = tile_repair_lllite_choices()
        required = {
            "model": ("MODEL",),
            "clip": ("CLIP",),
            "vae": ("VAE",),
            "image": ("IMAGE",),
            "lllite_name": (choices, {"default": default}),
            "positive": ("STRING", {"default": TILE_REPAIR_PROMPT, "multiline": True}),
            "negative": ("STRING", {"default": TILE_REPAIR_NEGATIVE, "multiline": True}),
            "seed": ("INT", {"default": 0, "min": 0, "max": 0xFFFFFFFFFFFFFFFF}),
            # Comfy KSampler ranges; sd-scripts defaults.
            "steps": ("INT", {"default": TILE_REPAIR_STEPS, "min": 1, "max": 10000}),
            "cfg": ("FLOAT", {
                "default": TILE_REPAIR_CFG, "min": 0.0, "max": 100.0, "step": 0.1, "round": 0.01,
            }),
            # Comfy ModelSamplingAuraFlow range; sd-scripts --flow_shift default.
            "flow_shift": ("FLOAT", {
                "default": TILE_REPAIR_FLOW_SHIFT, "min": 0.0, "max": 100.0, "step": 0.01,
            }),
            # The extension panel's short-side slider.
            "short_side": ("INT", {
                "default": TILE_REPAIR_SHORT_SIDE, "min": TILE_REPAIR_MIN_SIDE,
                "max": TILE_REPAIR_MAX_SHORT_SIDE, "step": TILE_REPAIR_SIZE_MULTIPLE,
            }),
        }
        required.update({name: (kind, dict(spec)) for name, (kind, spec) in LLLITE_APPLY_INPUTS.items()})
        return {"required": required}

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "repair"
    CATEGORY = CATEGORY

    def repair(self, model, clip, vae, image, lllite_name, positive=TILE_REPAIR_PROMPT,
               negative=TILE_REPAIR_NEGATIVE, seed=0, steps=TILE_REPAIR_STEPS,
               cfg=TILE_REPAIR_CFG, flow_shift=TILE_REPAIR_FLOW_SHIFT,
               short_side=TILE_REPAIR_SHORT_SIDE, strength=1.0, start_percent=0.0,
               end_percent=1.0, preserve_wrapper=True):
        torch = require_torch()
        if _is_none_choice(lllite_name):
            raise ValueError(
                "Pick an Anima Tile & Repair ControlNet-LLLite (e.g. animaTileRepair_v20.safetensors)."
            )
        control = lllite_control_for(lllite_name)
        if control is None:
            raise ValueError(
                f"{lllite_name!r} is not an Anima ControlNet-LLLite file (no "
                f"'{LLLITE_COND_PREFIX}*' tensors) in ComfyUI's controlnet folder."
            )
        if control.channels != DEFAULT_COND_IN_CHANNELS:
            raise ValueError(
                f"{lllite_name!r} has cond_in_channels={control.channels}; Tile & Repair takes "
                "3-channel (RGB) weights. 4-channel LLLites are inpaint models that need a mask."
            )
        if int(steps) < 1:
            raise ValueError("Anima Tile & Repair steps must be at least one.")
        if not math.isfinite(float(cfg)) or float(cfg) < 0.0:
            raise ValueError("Anima Tile & Repair cfg cannot be negative.")
        edge = int(short_side)
        if not TILE_REPAIR_MIN_SIDE <= edge <= TILE_REPAIR_MAX_SHORT_SIDE:
            raise ValueError(
                f"Anima Tile & Repair short_side must be between {TILE_REPAIR_MIN_SIDE} "
                f"and {TILE_REPAIR_MAX_SHORT_SIDE}."
            )
        for name, value in (
            ("strength", strength), ("start_percent", start_percent), ("end_percent", end_percent),
        ):
            spec = LLLITE_APPLY_INPUTS[name][1]
            if not spec["min"] <= float(value) <= spec["max"]:
                raise ValueError(
                    f"Anima Tile & Repair {name} must be between {spec['min']} and {spec['max']}."
                )

        source = _rgb_image(image)
        width, height = tile_repair_size(int(source.shape[2]), int(source.shape[1]), edge)
        positive_cond = _encode(clip, positive)
        negative_cond = _encode(clip, negative)
        shifted = _flow_shift(model, flow_shift)
        outputs = []
        for index in range(int(source.shape[0])):
            control_image = sdscripts_control_image(source[index:index + 1], width, height)
            with _controlnet_folder_name(control.path, control.name) as name:
                patched = apply_lllite(
                    shifted, name, control_image, float(strength), float(start_percent),
                    float(end_percent), preserve_wrapper=bool(preserve_wrapper),
                )
            sampled = _sample(
                patched, seed, steps, cfg, flow_shift, positive_cond, negative_cond,
                _empty_latent(width, height),
            )
            decoded = _decode(vae, sampled)
            outputs.append(decoded[..., :3])
        return (torch.cat(outputs, dim=0).clamp(0.0, 1.0),)


NODE_CLASS_MAPPINGS = {
    "ForgeNeoAnimaTileRepair": ForgeNeoAnimaTileRepair,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ForgeNeoAnimaTileRepair": "Forge Neo Anima Tile & Repair (LLLite)",
}


__all__ = [
    "AnimaLLLiteControl",
    "ForgeNeoAnimaTileRepair",
    "LLLiteInfo",
    "NODE_CLASS_MAPPINGS",
    "NODE_DISPLAY_NAME_MAPPINGS",
    "apply_lllite",
    "forced_control_module",
    "is_lllite_control",
    "latent_item",
    "lllite_control_for",
    "lllite_info",
    "lllite_info_from_header",
    "read_safetensors_header",
    "thresholds_for_module",
    "tile_repair_lllite_choices",
    "tile_repair_noise",
    "tile_repair_sigmas",
    "tile_repair_size",
]
