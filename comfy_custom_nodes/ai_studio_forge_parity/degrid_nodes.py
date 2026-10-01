"""``ForgeNeoAnimaVAEDeGrid`` — the sam-extra "Anima VAE DeGrid (NAFNet)" step for ComfyUI.

Forge runs the extension once per finished image, after every other
post-process (ADetailer, SAM3) and right before saving. The app's compiler
places this node at the same point: after the last image extension, before
the save/preview node, on main generations only.

Models: NAFNet files (``degrid_files``) from ComfyUI's ``upscale_models``
(Forge ``models/ESRGAN`` — where the model's download page says to put it)
and ``degrid`` (registered by this pack as ``ComfyUI/models/degrid``, merged
into an existing entry). Like Forge, only files directly in those folders are
listed. Choices are the file names; a name present in both folders gets its
folder as prefix. ``auto`` is the first file by ``modelspec.version``
(highest first), then folder, then name. Reports name models the way Forge's
infotext does (stem; ``ESRGAN/<stem>`` / ``DeGrid/<stem>`` on a stem clash).

Never fatal: a missing model, a file that is not a 1x RGB NAFNet, an
image-like or blown-up output, or any other error keeps that image as it
was and says why in the report (``ui.ai_studio_degrid`` and the
``report_json`` output), like Forge keeps the image and writes
``Anima DeGrid error``. A user cancel still cancels.

Memory: one model is cached (CPU, fp32). On a GPU it is loaded through
ComfyUI's model management (``load_models_gpu``; it can evict other models)
and unloaded after the batch unless ``keep_loaded``. ComfyUI's
unload-all (``/free``) also drops the cache (``install_unload_release_hook``).
"""

from __future__ import annotations

import functools
import json
import logging
import os
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from . import degrid_files, degrid_math as dm, degrid_runner as runner
from .compat import folder_paths_module, require_torch


NODE_CLASS = "ForgeNeoAnimaVAEDeGrid"
DISPLAY_NAME = "Forge Neo Anima VAE DeGrid (NAFNet)"
CATEGORY = "AI Studio/Forge Neo parity/Image"
UI_KEY = "ai_studio_degrid"
AUTO = "auto"

INPUT_NAMES = (
    "image", "enabled", "model_name", "mode", "strength", "tile",
    "device", "precision", "keep_loaded", "forge_quantize",
)
DEVICE_AUTO = "auto"
DEVICE_CPU = "cpu"
DEVICE_CHOICES = (DEVICE_AUTO, DEVICE_CPU)
PRECISION_CHOICES = (runner.PRECISION_FP32, runner.PRECISION_FP16)

UPSCALE_CATEGORY = "upscale_models"
DEGRID_CATEGORY = "degrid"
MODEL_CATEGORIES = (UPSCALE_CATEGORY, DEGRID_CATEGORY)
# Forge folder names, for report names that match Forge's infotext.
FORGE_FOLDERS = {UPSCALE_CATEGORY: "ESRGAN", DEGRID_CATEGORY: "DeGrid"}

STATUS_OK = "ok"
STATUS_SKIPPED = "skipped"
STATUS_OFF = "off"

# load_models_gpu estimate: bytes per pixel of the largest piece (measured
# peak about 1.15 KB/px at fp32, plus cuDNN workspace headroom).
BYTES_PER_PIXEL = 1536

_LOG = logging.getLogger("ai_studio_forge_parity")
_UNLOAD_HOOK_MARK = "_ai_studio_degrid_release_hook"


# ── model folders ────────────────────────────────────────────────────────
def register_degrid_folder(folder_paths: Any = None) -> bool:
    """Add ``<models>/degrid`` to ComfyUI's ``degrid`` category (created or merged).

    An existing entry (extra_model_paths.yaml, another pack) keeps its paths
    and extensions. No-op outside ComfyUI. Returns whether anything changed.
    """
    if folder_paths is None:
        try:
            folder_paths = folder_paths_module()
        except RuntimeError:
            return False
    registry = getattr(folder_paths, "folder_names_and_paths", None)
    models_dir = getattr(folder_paths, "models_dir", None)
    if not isinstance(registry, dict) or not models_dir:
        return False
    default = os.path.join(str(models_dir), DEGRID_CATEGORY)
    entry = registry.get(DEGRID_CATEGORY)
    if entry is None:
        extensions = set(getattr(folder_paths, "supported_pt_extensions", ()) or dm.MODEL_EXTENSIONS)
        registry[DEGRID_CATEGORY] = ([default], extensions)
        changed = True
    else:
        paths, extensions = entry
        known = {os.path.normcase(os.path.abspath(str(path))) for path in paths}
        changed = os.path.normcase(os.path.abspath(default)) not in known
        if changed:
            registry[DEGRID_CATEGORY] = ([*paths, default], extensions)
    if changed:
        cache = getattr(folder_paths, "filename_list_cache", None)
        if isinstance(cache, dict):
            cache.pop(DEGRID_CATEGORY, None)
    return changed


@dataclass(frozen=True)
class ModelFile:
    category: str
    filename: str     # relative to the category root (top level only)
    path: str
    version: tuple
    choice: str = ""      # combo value
    forge_name: str = ""  # name Forge's infotext uses for the same file


def _category_files(folder_paths: Any, category: str) -> list[tuple[str, str]]:
    try:
        names = list(folder_paths.get_filename_list(category))
    except Exception:
        return []
    files = []
    for filename in sorted(names, key=str.lower):
        if "/" in filename or "\\" in filename:
            continue  # Forge lists top-level files only
        if Path(filename).suffix.lower() not in dm.MODEL_EXTENSIONS:
            continue
        try:
            path = folder_paths.get_full_path(category, filename)
        except Exception:
            path = None
        if path and os.path.isfile(path):
            files.append((filename, str(path)))
    return files


def discover_models(folder_paths: Any = None) -> list[ModelFile]:
    """NAFNet files in both categories, highest declared version first."""
    if folder_paths is None:
        try:
            folder_paths = folder_paths_module()
        except RuntimeError:
            return []
    found: list[ModelFile] = []
    for category in MODEL_CATEGORIES:
        for filename, path in _category_files(folder_paths, category):
            is_nafnet, version = degrid_files.classify(path)
            if is_nafnet:
                found.append(ModelFile(category, filename, path, version))
    found.sort(key=lambda item: item.version, reverse=True)  # stable: folder, then name
    return name_models(found)


def name_models(files: list[ModelFile]) -> list[ModelFile]:
    """Fill ``choice`` (file name, folder-prefixed when both folders have it) and ``forge_name``."""
    by_filename: dict[str, set] = {}
    by_stem: dict[str, int] = {}
    for item in files:
        by_filename.setdefault(item.filename.lower(), set()).add(item.category)
        stem = Path(item.filename).stem.lower()
        by_stem[stem] = by_stem.get(stem, 0) + 1
    named = []
    for item in files:
        clash = len(by_filename[item.filename.lower()]) > 1
        choice = f"{item.category}/{item.filename}" if clash else item.filename
        stem = Path(item.filename).stem
        forge = stem if by_stem[stem.lower()] == 1 else f"{FORGE_FOLDERS[item.category]}/{stem}"
        named.append(ModelFile(item.category, item.filename, item.path, item.version, choice, forge))
    return named


def model_choices(folder_paths: Any = None) -> list[str]:
    return [AUTO, *(item.choice for item in discover_models(folder_paths))]


def resolve_model(name: Any, files: list[ModelFile]) -> Optional[ModelFile]:
    """``auto``/empty/``None`` -> first file; else choice, Forge name, file name or stem."""
    files = list(files or ())
    text = str(name or "").strip()
    if not files:
        return None
    if text.lower() in ("", AUTO, "none"):
        return files[0]
    for item in files:
        if text in (item.choice, item.forge_name):
            return item
    lowered = text.lower()
    for item in files:
        if lowered in (item.choice.lower(), item.forge_name.lower()):
            return item
    for item in files:
        if lowered in (item.filename.lower(), Path(item.filename).stem.lower()):
            return item
    return None


# ── model cache / memory ─────────────────────────────────────────────────
_CACHE_LOCK = threading.RLock()
_CACHE: dict[str, Any] = {"key": None, "model": None, "patcher": None}


def _file_key(path: str) -> tuple:
    stat = os.stat(path)
    return (os.path.abspath(path), int(stat.st_size), int(stat.st_mtime_ns))


def load_nafnet(path: str):
    """A 1x RGB NAFNet ``nn.Module`` (CPU, fp32, eval) through spandrel."""
    import importlib

    torch = require_torch()
    comfy_utils = importlib.import_module("comfy.utils")
    spandrel = importlib.import_module("spandrel")
    name = Path(path).name
    with torch.inference_mode(False), torch.no_grad():
        state = comfy_utils.load_torch_file(str(path), safe_load=True)
        descriptor = spandrel.ModelLoader(device="cpu").load_from_state_dict(state)
    arch = str(getattr(getattr(descriptor, "architecture", None), "id", "") or "")
    if arch.lower() != "nafnet":
        raise ValueError(f"{name} is not a NAFNet model (spandrel: {arch or type(descriptor).__name__})")
    if (getattr(descriptor, "scale", 1), getattr(descriptor, "input_channels", 3),
            getattr(descriptor, "output_channels", 3)) != (1, 3, 3):
        raise ValueError(f"{name}: expected a 1x RGB NAFNet (scale {getattr(descriptor, 'scale', '?')})")
    model = descriptor.model
    model.eval()
    model.requires_grad_(False)
    return model


def cached_model(path: str, loader=load_nafnet):
    """The model of ``path``; a different file replaces (and releases) the cached one."""
    key = _file_key(path)
    with _CACHE_LOCK:
        if _CACHE["model"] is not None and _CACHE["key"] == key:
            return _CACHE["model"]
        release_model(drop=True)
        model = loader(path)
        _CACHE.update(key=key, model=model, patcher=None)
        return model


def _management():
    import importlib

    try:
        return importlib.import_module("comfy.model_management")
    except Exception:
        return None


def _module_off(model: Any, device: Any) -> bool:
    torch = require_torch()
    target = torch.device(device)
    for param in model.parameters():
        if param.device.type != target.type:
            return True
        if target.index is not None and param.device.index != target.index:
            return True
    return False


def choose_device(requested: Any):
    torch = require_torch()
    if str(requested or "").strip().lower() == DEVICE_CPU:
        return torch.device("cpu")
    management = _management()
    getter = getattr(management, "get_torch_device", None)
    if callable(getter):
        try:
            return torch.device(getter())
        except Exception:
            pass
    return torch.device("cpu")


def acquire(model: Any, device: Any, memory_required: float) -> None:
    """Put the cached model on ``device`` (through ComfyUI's model management on a GPU)."""
    torch = require_torch()
    device = torch.device(device)
    with _CACHE_LOCK:
        if device.type == "cpu":
            if _module_off(model, device):
                release_model()
            return
        management = _management()
        if management is None:
            with torch.inference_mode(False):
                model.to(device)
            return
        import importlib

        patcher_module = importlib.import_module("comfy.model_patcher")
        patcher_class = getattr(patcher_module, "CoreModelPatcher", None) or patcher_module.ModelPatcher
        patcher = _CACHE.get("patcher")
        if patcher is None or getattr(patcher, "model", None) is not model \
                or torch.device(getattr(patcher, "load_device", "cpu")) != device:
            if patcher is not None:
                release_model()
            patcher = patcher_class(model, load_device=device, offload_device=torch.device("cpu"))
            _CACHE["patcher"] = patcher
        management.load_models_gpu([patcher], memory_required=memory_required, force_full_load=True)
        if _module_off(model, device):
            with torch.inference_mode(False):
                model.to(device)


def release_model(*, drop: bool = False) -> bool:
    """Take the model off the GPU (and out of ComfyUI's loaded list); ``drop`` also forgets it."""
    with _CACHE_LOCK:
        released = False
        patcher = _CACHE.get("patcher")
        management = _management() if patcher is not None else None
        if management is not None:
            loaded = getattr(management, "current_loaded_models", None)
            for entry in list(loaded or ()):
                if getattr(entry, "model", None) is patcher:
                    try:
                        entry.model_unload()
                    finally:
                        loaded.remove(entry)
                    released = True
                    break
        model = _CACHE.get("model")
        if model is not None and _module_off(model, "cpu"):
            torch = require_torch()
            with torch.inference_mode(False):
                model.to(torch.device("cpu"))
            released = True
        if released:
            management = management or _management()
            empty = getattr(management, "soft_empty_cache", None)
            if callable(empty):
                try:
                    empty()
                except Exception:
                    pass
        if drop:
            _CACHE.update(key=None, model=None, patcher=None)
        return released


def install_unload_release_hook(management: Any = None) -> bool:
    """ComfyUI's ``unload_all_models`` (``/free``, OOM recovery) also drops the DeGrid cache."""
    if management is None:
        management = _management()
        if management is None:
            return False
    original = getattr(management, "unload_all_models", None)
    if not callable(original) or getattr(original, _UNLOAD_HOOK_MARK, False):
        return False

    @functools.wraps(original)
    def unload_all_models(*args, **kwargs):
        try:
            return original(*args, **kwargs)
        finally:
            try:
                release_model(drop=True)
            except Exception:
                _LOG.warning("AI Studio VAE DeGrid cache release failed", exc_info=True)

    setattr(unload_all_models, _UNLOAD_HOOK_MARK, True)
    management.unload_all_models = unload_all_models
    return True


def _is_cancel(exc: BaseException) -> bool:
    management = _management()
    interrupt = getattr(management, "InterruptProcessingException", None)
    if isinstance(interrupt, type) and isinstance(exc, interrupt):
        return True
    return type(exc).__name__ == "InterruptProcessingException"


def _oom_check():
    management = _management()
    checker = getattr(management, "is_oom", None)

    def is_oom(exc: BaseException) -> bool:
        if callable(checker):
            try:
                if checker(exc):
                    return True
            except Exception:
                pass
        return dm.message_says_oom(exc)

    return is_oom


def _progress(total: int):
    import importlib

    try:
        bar = importlib.import_module("comfy.utils").ProgressBar(max(1, int(total)))
    except Exception:
        return None
    return lambda: bar.update(1)


def _interrupt_check():
    management = _management()
    check = getattr(management, "throw_exception_if_processing_interrupted", None)
    return check if callable(check) else None


def _tile_count(height: int, width: int, tile: int) -> int:
    if tile <= 0 or (height <= tile and width <= tile):
        return 1
    return len(dm.tile_positions(height, tile)) * len(dm.tile_positions(width, tile))


# ── the node ─────────────────────────────────────────────────────────────
class ForgeNeoAnimaVAEDeGrid:
    """Removes the Qwen/Wan VAE grid pattern with a NAFNet residual model (Forge sam-extra parity)."""

    CATEGORY = CATEGORY
    FUNCTION = "apply"
    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("image", "report_json")
    DESCRIPTION = (
        "Anima VAE DeGrid (NAFNet): adds the model's residual to each image like the Forge "
        "sam-extra script (Full / Dark / Bright, strength, tile 512 with overlap 32). Skips "
        "(missing model, not a DeGrid model, blown-up residual, errors) keep the image and "
        "are reported, never fatal."
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "enabled": ("BOOLEAN", {"default": True}),
                "model_name": (model_choices(),),
                "mode": (list(dm.MODE_CHOICES), {"default": dm.MODE_LABELS[dm.DEFAULT_MODE]}),
                "strength": ("FLOAT", {
                    "default": dm.DEFAULT_STRENGTH, "min": dm.STRENGTH_MIN,
                    "max": dm.STRENGTH_MAX, "step": 0.05,
                }),
                "tile": ("INT", {"default": dm.DEFAULT_TILE, "min": 0, "max": dm.MAX_TILE, "step": 1}),
                "device": (list(DEVICE_CHOICES), {"default": DEVICE_AUTO}),
                "precision": (list(PRECISION_CHOICES), {"default": runner.PRECISION_FP32}),
                "keep_loaded": ("BOOLEAN", {"default": False}),
                "forge_quantize": ("BOOLEAN", {"default": True}),
            },
        }

    def apply(self, image, enabled, model_name, mode, strength, tile,
              device=DEVICE_AUTO, precision=runner.PRECISION_FP32,
              keep_loaded=False, forge_quantize=True):
        reports, output = run_degrid(
            image, enabled=enabled, model_name=model_name, mode=mode, strength=strength,
            tile=tile, device=device, precision=precision, keep_loaded=keep_loaded,
            forge_quantize=forge_quantize,
        )
        return {"ui": {UI_KEY: reports}, "result": (output, json.dumps(reports, ensure_ascii=False))}


def run_degrid(image, *, enabled, model_name, mode, strength, tile, device=DEVICE_AUTO,
               precision=runner.PRECISION_FP32, keep_loaded=False, forge_quantize=True,
               files: Optional[list[ModelFile]] = None, loader=None):
    """``(reports, images)`` for an IMAGE batch ``[B, H, W, C]``; skipped items come back unchanged."""
    torch = require_torch()
    batch = int(image.shape[0])
    mode_key = dm.normalize_mode(mode) or dm.DEFAULT_MODE
    strength = dm.coerce_strength(strength)
    tile = dm.coerce_tile(tile)
    requested = str(model_name or "").strip() or AUTO

    def same_for_all(status, name, error="", precision_label=runner.PRECISION_LABEL_NONE):
        item = runner.report(status=status, model=name, mode=mode_key, strength=strength,
                             tile=tile, precision=precision_label, error=error)
        return [dict(item) for _ in range(batch)], image

    if not enabled:
        return same_for_all(STATUS_OFF, requested)
    files = discover_models() if files is None else files
    entry = resolve_model(requested, files)
    if entry is None:
        return same_for_all(STATUS_SKIPPED, requested, f"{dm.ERROR_MODEL_NOT_FOUND}: {requested}")
    if strength == 0.0:
        return same_for_all(STATUS_OK, entry.forge_name)  # Forge records it, runs nothing
    try:
        model = cached_model(entry.path, loader or load_nafnet)
    except Exception as exc:
        if _is_cancel(exc):
            raise
        _LOG.warning("AI Studio VAE DeGrid: %s", dm.failure_text(exc))
        return same_for_all(STATUS_SKIPPED, entry.forge_name, dm.failure_text(exc))

    target = choose_device(device)
    height, width = int(image.shape[1]), int(image.shape[2])
    reports: list[dict] = []
    outputs = []
    is_oom = _oom_check()
    management = _management()
    step = _progress(batch * _tile_count(height, width, tile))
    interrupted = _interrupt_check()

    def on_retry(new_tile: int, exc: BaseException) -> None:
        _LOG.warning("AI Studio VAE DeGrid: out of memory, retrying with tile %s", new_tile)
        empty = getattr(management, "soft_empty_cache", None)
        if callable(empty):
            empty()

    def skipped(item, exc: BaseException) -> None:
        if _is_cancel(exc):
            raise exc
        _LOG.warning("AI Studio VAE DeGrid: %s", dm.failure_text(exc))
        outputs.append(item.to(dtype=torch.float32))
        reports.append(runner.report(
            status=STATUS_SKIPPED, model=entry.forge_name, mode=mode_key,
            strength=strength, tile=tile, error=dm.failure_text(exc),
        ))

    try:
        piece = min(height, tile) * min(width, tile) if tile > 0 else height * width
        try:
            acquire(model, target, float(BYTES_PER_PIXEL * piece))
        except Exception as exc:  # e.g. not enough memory to load it at all
            if _is_cancel(exc):
                raise
            _LOG.warning("AI Studio VAE DeGrid: %s", dm.failure_text(exc))
            return same_for_all(STATUS_SKIPPED, entry.forge_name, dm.failure_text(exc))
        for index in range(batch):
            item = image[index:index + 1]
            rgb = item[..., :3].movedim(-1, 1).contiguous()
            try:
                done = runner.degrid_one(
                    rgb, model, model_name=entry.forge_name, mode=mode_key, strength=strength,
                    tile=tile, device=target, precision=precision, forge_quantize=bool(forge_quantize),
                    is_oom=is_oom, on_retry=on_retry, on_piece=step, before_piece=interrupted,
                )
            except Exception as exc:
                skipped(item, exc)
                continue
            out = done.image.movedim(1, -1).to(device=item.device, dtype=torch.float32)
            if item.shape[-1] > 3:  # keep extra channels (alpha) as they were
                out = torch.cat([out, item[..., 3:].to(dtype=torch.float32)], dim=-1)
            outputs.append(out)
            reports.append(runner.report(
                status=STATUS_OK, model=entry.forge_name, mode=mode_key, strength=strength,
                tile=done.tile_used, precision=done.precision,
            ))
    finally:
        if not keep_loaded or target.type == "cpu":
            try:
                release_model()
            except Exception:
                _LOG.warning("AI Studio VAE DeGrid: release failed", exc_info=True)
    return reports, torch.cat(outputs, dim=0)


NODE_CLASS_MAPPINGS = {NODE_CLASS: ForgeNeoAnimaVAEDeGrid}
NODE_DISPLAY_NAME_MAPPINGS = {NODE_CLASS: DISPLAY_NAME}

__all__ = [
    "AUTO", "DEVICE_CHOICES", "FORGE_FOLDERS", "ForgeNeoAnimaVAEDeGrid", "INPUT_NAMES",
    "MODEL_CATEGORIES", "ModelFile", "NODE_CLASS", "NODE_CLASS_MAPPINGS",
    "NODE_DISPLAY_NAME_MAPPINGS", "PRECISION_CHOICES", "UI_KEY", "acquire", "cached_model",
    "choose_device", "discover_models", "install_unload_release_hook", "load_nafnet",
    "model_choices", "name_models", "register_degrid_folder", "release_model", "resolve_model",
    "run_degrid",
]
