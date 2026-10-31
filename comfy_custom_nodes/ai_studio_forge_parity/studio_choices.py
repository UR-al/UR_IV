"""Choice lists for the Anima studio's pickers (samplers, models, LoRAs, SAM3 checkpoints, ...).

No ComfyUI import at module level: ``folder_paths`` / ``comfy.samplers`` are read when a list is asked for, and a
missing host just gives an empty list.  Every kind returns plain strings except ``sam3``, which returns
``{"value": absolute path, "label": file name}`` objects (the Forge ``[SAM3] Checkpoint`` axis and the SAM3 node take the
path; the picker shows the file name).
"""
from __future__ import annotations

import importlib
import os
import re
from pathlib import Path

from .dora_infer_mode import MODE_CHOICES

KINDS = ("samplers", "schedulers", "unets", "vaes", "tes", "loras", "upscalers", "sam3", "dora_modes")
SAM3_PREFIX = "sam3"
SAM3_SUFFIXES = (".pt", ".safetensors")  # what sam-extra's find_checkpoint_options accepts
EXTRA_PATHS_FILE = "extra_model_paths.yaml"
_NAME_38 = re.compile(r"3[._-]?8b", re.IGNORECASE)


def _folder_paths():
    try:
        return importlib.import_module("folder_paths")
    except Exception:
        return None


def _file_list(folder_paths, kind: str) -> list[str]:
    if folder_paths is None:
        return []
    try:
        return list(folder_paths.get_filename_list(kind))
    except Exception:
        return []


def _samplers(name: str) -> list[str]:
    try:
        return list(getattr(importlib.import_module("comfy.samplers").KSampler, name))
    except Exception:
        return []


# ------------------------------------------------------------------------------------------------- unet families

def anima38_bundle_names() -> frozenset:
    """The files the 3.8B v2 loader offers (checkpoints whose header says they are a bundle); empty outside ComfyUI."""
    try:
        from .anima38_nodes import ForgeNeoAnima38V2Loader

        return frozenset(ForgeNeoAnima38V2Loader.INPUT_TYPES()["required"]["model_name"][0])
    except Exception:
        return frozenset()


def unet_family(name: str, bundle_names=frozenset()) -> str:
    """``"3.8b"`` for a v2 bundle; ``"moss"`` / ``"krea2"`` if the name says MOSS / KREA; ``"3.8b"`` for a name like
    ``3.8b`` / ``3_8B``; ``"base"`` if the name says base; else ``"2.9b"``."""
    if name in bundle_names:
        return "3.8b"
    lowered = name.lower()
    if "moss" in lowered:
        return "moss"
    if "krea" in lowered:
        return "krea2"
    if _NAME_38.search(name):
        return "3.8b"
    return "base" if "base" in lowered else "2.9b"


def unet_names(folder_paths=None) -> list[str]:
    folder_paths = folder_paths if folder_paths is not None else _folder_paths()
    names = _file_list(folder_paths, "diffusion_models") + _file_list(folder_paths, "unet")
    return list(dict.fromkeys(names))


def unet_families(names=None, bundle_names=None) -> dict:
    names = unet_names() if names is None else list(names)
    bundle = anima38_bundle_names() if bundle_names is None else bundle_names
    return {name: unet_family(name, bundle) for name in names}


def lora_names(folder_paths=None) -> list[str]:
    """ComfyUI's LoRA file list.  Unlike the picker lists this raises when the list cannot be read, so the planner
    does not report every ``<lora:...>`` tag as a missing file."""
    folder_paths = folder_paths if folder_paths is not None else _folder_paths()
    if folder_paths is None:
        raise RuntimeError("ComfyUI folder_paths is not available")
    return list(folder_paths.get_filename_list("loras"))


# ------------------------------------------------------------------------------------------------- SAM3 files

def _forge_sam3_directories(base_path: str) -> list[str]:
    """``<base_path>/models/sam3`` of every ``extra_model_paths.yaml`` entry (a ``sam3:`` key overrides the folder)."""
    config = Path(base_path) / EXTRA_PATHS_FILE
    try:
        import yaml

        data = yaml.safe_load(config.read_text(encoding="utf-8"))
    except Exception:
        return []
    found = []
    for entry in (data.values() if isinstance(data, dict) else ()):
        if not isinstance(entry, dict) or not isinstance(entry.get("base_path"), str):
            continue
        root = Path(entry["base_path"]).expanduser()
        if not root.is_absolute():
            root = config.parent / root
        sam3 = entry.get("sam3")
        lines = [line.strip() for line in sam3.splitlines()] if isinstance(sam3, str) else []
        for line in [line for line in lines if line] or ["models/sam3"]:
            found.append(str(root / line))
    return found


def sam3_directories(folder_paths=None) -> list[str]:
    """ComfyUI's own ``models/sam3`` (and any registered ``sam3`` folder), then the Forge folders of the extra paths file."""
    folder_paths = folder_paths if folder_paths is not None else _folder_paths()
    if folder_paths is None:
        return []
    found = []
    try:
        found += [str(path) for path in folder_paths.get_folder_paths("sam3")]
    except Exception:
        pass
    base_path = getattr(folder_paths, "base_path", None)
    if base_path:
        found.append(os.path.join(str(base_path), "models", "sam3"))
        found += _forge_sam3_directories(str(base_path))
    unique = {}
    for path in found:
        unique.setdefault(os.path.normcase(os.path.abspath(path)), os.path.abspath(path))
    return list(unique.values())


def sam3_items(directories) -> list[dict]:
    """SAM3 checkpoints (file name starting with ``sam3``, ``.pt`` / ``.safetensors``) as ``{value: path, label: name}``.

    The first folder wins when two folders hold the same file name."""
    items, seen = [], set()
    for directory in directories:
        try:
            names = sorted(os.listdir(directory), key=str.lower)
        except OSError:
            continue
        for name in names:
            lowered = name.lower()
            path = os.path.join(directory, name)
            if (not lowered.startswith(SAM3_PREFIX) or not lowered.endswith(SAM3_SUFFIXES)
                    or lowered in seen or not os.path.isfile(path)):
                continue
            seen.add(lowered)
            items.append({"value": path, "label": name})
    return items


# ------------------------------------------------------------------------------------------------- the lists

def upscaler_items() -> list[str]:
    from .generation import upscale_method_choices  # the list ForgeNeoHiresFix offers

    return upscale_method_choices()


def choices(kind: str) -> dict:
    """``{"items": [...]}`` for one kind (``unets`` adds ``"families"``); ValueError for an unknown kind."""
    if kind not in KINDS:
        raise ValueError(f"unknown kind: {kind}")
    folder_paths = _folder_paths()
    if kind == "samplers":
        return {"items": _samplers("SAMPLERS")}
    if kind == "schedulers":
        return {"items": _samplers("SCHEDULERS")}
    if kind == "unets":
        names = unet_names(folder_paths)
        return {"items": names, "families": unet_families(names)}
    if kind == "vaes":
        return {"items": _file_list(folder_paths, "vae")}
    if kind == "tes":
        return {"items": _file_list(folder_paths, "text_encoders")}
    if kind == "loras":
        return {"items": _file_list(folder_paths, "loras")}
    if kind == "upscalers":
        return {"items": upscaler_items()}
    if kind == "sam3":
        return {"items": sam3_items(sam3_directories(folder_paths))}
    return {"items": list(MODE_CHOICES)}
