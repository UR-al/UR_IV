"""Choice lists for the Anima studio's pickers (samplers, models, LoRAs, SAM3 checkpoints, ...).

No ComfyUI import at module level: ``folder_paths`` / ``comfy.samplers`` are read when a list is asked for, and a
missing host just gives an empty list.  Every kind returns plain strings except ``sam3`` and ``face_models``, which return
``{"value": absolute path, "label": file name}`` objects (the Forge ``[SAM3] Checkpoint`` axis and the SAM3 node take the
path; the picker shows the file name).
"""
from __future__ import annotations

import importlib
import os
import re
from pathlib import Path
from typing import Optional

from .degrid_files import safetensors_header
from .dora_infer_mode import MODE_CHOICES
from .sam3_face_detect import model_dirs as face_model_dirs, scan_models as scan_face_models
from .sam3_files import is_transformers_layout
from .studio_state import ATTENTION_MODES
from .studio_wildcards import FILES as WILDCARD_FILES

KINDS = ("samplers", "schedulers", "unets", "vaes", "tes", "loras", "upscalers", "sam3", "dora_modes", "controlnets", "face_models",
         "attentions", "wildcards")
SAM3_PREFIX = "sam3"
SAM3_SUFFIXES = (".pt", ".safetensors")  # what sam-extra's find_checkpoint_options accepts
EXTRA_PATHS_FILE = "extra_model_paths.yaml"
_NAME_38 = re.compile(r"3[._-]?8b", re.IGNORECASE)
_bundle_scan: dict = {}  # the last scan: "stamps" (see _model_stamps), "bundles"
_lora_epochs: dict = {}  # file path -> ((size, time), epoch)


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

def _read_bundle_names() -> frozenset:
    from .anima38_nodes import ForgeNeoAnima38V2Loader

    return frozenset(ForgeNeoAnima38V2Loader.INPUT_TYPES()["required"]["model_name"][0])


def anima38_bundle_names() -> frozenset:
    """The files the 3.8B v2 loader offers (checkpoints whose header says they are a bundle); empty outside ComfyUI."""
    try:
        return _read_bundle_names()
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


def _model_stamps(names: list) -> tuple:
    """What a bundle scan depends on: the model files, each with its size and time."""
    folder_paths = _folder_paths()
    stamps = []
    for name in names:
        try:
            stat = os.stat(folder_paths.get_full_path("diffusion_models", name))
            stamps.append((name, stat.st_size, stat.st_mtime_ns))
        except (AttributeError, OSError, TypeError):
            stamps.append((name, None, None))
    return tuple(stamps)


def _bundle_names_of(names: list) -> frozenset:
    """The loader opens every model file, which can take seconds, and the X/Y/Z panel plans on each edit: it runs again only
    when a file changed. A scan that failed is not kept."""
    stamps = _model_stamps(names)
    if _bundle_scan.get("stamps") != stamps:
        try:
            _bundle_scan.update(stamps=stamps, bundles=_read_bundle_names())
        except Exception:
            _bundle_scan.clear()
            return frozenset()
    return _bundle_scan["bundles"]


def unet_families(names=None, bundle_names=None) -> dict:
    names = unet_names() if names is None else list(names)
    bundle = _bundle_names_of(names) if bundle_names is None else bundle_names
    return {name: unet_family(name, bundle) for name in names}


def lora_names(folder_paths=None) -> list[str]:
    """ComfyUI's LoRA file list.  Unlike the picker lists this raises when the list cannot be read, so the planner
    does not report every ``<lora:...>`` tag as a missing file."""
    folder_paths = folder_paths if folder_paths is not None else _folder_paths()
    if folder_paths is None:
        raise RuntimeError("ComfyUI folder_paths is not available")
    return list(folder_paths.get_filename_list("loras"))


def _ss_epoch(header: dict) -> Optional[int]:
    metadata = header.get("__metadata__")
    text = str(metadata.get("ss_epoch")) if isinstance(metadata, dict) else ""
    return int(text) if text.isdecimal() and len(text) < 10 and int(text) > 0 else None


def lora_epoch(name: str, folder_paths=None) -> Optional[int]:
    """The epoch a training tool wrote into a LoRA file (``ss_epoch``), None if it wrote none or the file cannot be read.
    Read once per version of the file: a header can be megabytes of JSON and the X/Y/Z panel plans on each edit. A read that
    failed is not kept."""
    folder_paths = folder_paths if folder_paths is not None else _folder_paths()
    try:
        path = folder_paths.get_full_path("loras", name)
        stat = os.stat(path)
    except (AttributeError, OSError, TypeError):
        return None
    stamp = (stat.st_size, stat.st_mtime_ns)
    known = _lora_epochs.get(path)
    if known is not None and known[0] == stamp:
        return known[1]
    try:
        header = safetensors_header(path)
    except RecursionError:
        header = None
    if header is None:
        return None
    _lora_epochs[path] = (stamp, _ss_epoch(header))
    return _lora_epochs[path][1]


# ------------------------------------------------------------------------------------------------- SAM3 files

def _forge_directories(base_path: str, key: str) -> list[str]:
    """``<base_path>/models/<key>`` of every ``extra_model_paths.yaml`` entry (an entry's own ``key:`` overrides the folder)."""
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
        folder = entry.get(key)
        lines = [line.strip() for line in folder.splitlines()] if isinstance(folder, str) else []
        for line in [line for line in lines if line] or [f"models/{key}"]:
            found.append(str(root / line))
    return found


def _unique_paths(paths) -> list[str]:
    unique = {}
    for path in paths:
        unique.setdefault(os.path.normcase(os.path.abspath(path)), os.path.abspath(path))
    return list(unique.values())


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
        found += _forge_directories(str(base_path), "sam3")
    return _unique_paths(found)


def sam3_items(directories) -> list[dict]:
    """SAM3 checkpoints (file name starting with ``sam3``, ``.pt`` / ``.safetensors``) as ``{value: path, label: name}``.

    The first folder wins when two folders hold the same file name. A file Easy SAM3 cannot read is left out."""
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
                    or lowered in seen or not os.path.isfile(path) or is_transformers_layout(path)):
                continue
            seen.add(lowered)
            items.append({"value": path, "label": name})
    return items


def face_model_directories(folder_paths=None) -> list[str]:
    """The folders ``sam3_face_detect`` reads a bare name from, then Forge's ``models/adetailer`` (it never looks there)."""
    folder_paths = folder_paths if folder_paths is not None else _folder_paths()
    base_path = getattr(folder_paths, "base_path", None)
    forge = _forge_directories(str(base_path), "adetailer") if base_path else []
    return _unique_paths([*(str(path) for path in face_model_dirs()), *forge])


def face_model_items(directories) -> list[dict]:
    """The ultralytics box models of ``directories`` as ``{value: path, label: name}``; the first folder wins a name."""
    items, seen = [], set()
    for directory in directories:
        for path in scan_face_models(Path(directory)):
            if path.name.lower() not in seen:
                seen.add(path.name.lower())
                items.append({"value": str(path), "label": path.name})
    return items


# ------------------------------------------------------------------------------------------------- the lists

def upscaler_items() -> list[str]:
    from .generation import upscale_method_choices  # the list ForgeNeoHiresFix offers

    return upscale_method_choices()


def attention_choices() -> dict:
    """The attention modes this ComfyUI has; ``launch`` names what it was started with."""
    try:
        attention = importlib.import_module("comfy.ldm.modules.attention")
    except Exception:
        return {"items": list(ATTENTION_MODES[:2]), "launch": ""}
    known = attention.REGISTERED_ATTENTION_FUNCTIONS
    started = next((name for name, function in known.items() if function is attention.optimized_attention), "")
    return {"items": [mode for mode in ATTENTION_MODES if mode == "launch" or mode in known], "launch": started}


def choices(kind: str) -> dict:
    """``{"items": [...]}`` for one kind (``unets`` adds ``"families"``, ``attentions`` adds ``"launch"``); ValueError for
    an unknown kind."""
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
    if kind == "controlnets":
        return {"items": _file_list(folder_paths, "controlnet")}
    if kind == "face_models":
        return {"items": face_model_items(face_model_directories(folder_paths))}
    if kind == "attentions":
        return attention_choices()
    if kind == "wildcards":
        return {"items": WILDCARD_FILES.names()}
    return {"items": list(MODE_CHOICES)}
