"""Which model files are VAE DeGrid candidates — read without loading any tensor.

A file qualifies when its state dict has every spandrel NAFNet detection key
(``degrid_math.NAFNET_KEYS``), like the Forge extension's list. Upscale-model
folders hold ordinary upscalers too, so each file is inspected:

- ``.safetensors``: the JSON header only (8-byte length + JSON).
- ``.pth``/``.pt`` in torch's zip format: only ``data.pkl`` is unpickled, by a
  restricted unpickler that rebuilds dicts and key names and turns every
  tensor/storage/other global into an inert placeholder. ``torch.load`` is
  never called and no code from the file runs.
- Legacy (non-zip) torch pickles are skipped: DeGrid is distributed as
  safetensors or zip ``.pth``.

Order key: ``modelspec.version`` from the safetensors metadata (``"1.1"`` ->
``(1, 1)``; missing -> ``()``, sorted after declared versions).

Verdicts are cached by (path, size, mtime). No torch import.
"""

from __future__ import annotations

import collections
import io
import json
import os
import pickle
import re
import struct
import threading
import zipfile
from pathlib import Path
from typing import Any, Iterable, Optional

from .degrid_math import MODEL_EXTENSIONS, NAFNET_KEYS

_HEADER_LIMIT = 100 << 20      # larger JSON headers are not safetensors we read
_PICKLE_LIMIT = 64 << 20       # data.pkl of a ~100 MB model is well under 1 MB
_ZIP_SIGNATURE = b"PK\x03\x04"
_WRAPPERS = ("params_ema", "params-ema", "params", "state_dict", "model_state_dict", "model", "net")
_COMMON_PREFIXES = ("module.", "netG.", "net_g.")


def safetensors_header(path) -> Optional[dict]:
    try:
        with open(path, "rb") as handle:
            raw_length = handle.read(8)
            if len(raw_length) != 8:
                return None
            (length,) = struct.unpack("<Q", raw_length)
            if not 0 < length <= _HEADER_LIMIT:
                return None
            header = json.loads(handle.read(length).decode("utf-8"))
    except (OSError, ValueError, UnicodeDecodeError):
        return None
    return header if isinstance(header, dict) else None


def version_tuple(text: Any) -> tuple:
    """``"1.1"`` -> ``(1, 1)``; up to four numbers; none -> ``()``."""
    return tuple(int(part) for part in re.findall(r"\d+", str(text or ""))[:4])


def modelspec_version(path) -> tuple:
    if Path(path).suffix.lower() != ".safetensors":
        return ()
    metadata = (safetensors_header(path) or {}).get("__metadata__")
    if not isinstance(metadata, dict):
        return ()
    return version_tuple(metadata.get("modelspec.version"))


class _Inert:
    """Stands in for tensors, storages and every non-dict global of ``data.pkl``."""

    def __new__(cls, *args, **kwargs):
        return object.__new__(cls)

    def __init__(self, *args, **kwargs):
        pass

    def __setstate__(self, state):
        pass


class _KeyNamesUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if (module, name) == ("collections", "OrderedDict"):
            return collections.OrderedDict
        return _Inert

    def persistent_load(self, pid):
        return _Inert()


def is_zip_checkpoint(path) -> bool:
    try:
        with open(path, "rb") as handle:
            if handle.read(4) != _ZIP_SIGNATURE:
                return False
        return zipfile.is_zipfile(path)
    except OSError:
        return False


def zip_state_skeleton(path) -> Any:
    """The unpickled ``data.pkl`` of a zip checkpoint with placeholders for tensors, or None."""
    if not is_zip_checkpoint(path):
        return None
    try:
        with zipfile.ZipFile(path) as archive:
            member = None
            for info in archive.infolist():
                parts = info.filename.replace("\\", "/").split("/")
                if parts[-1] == "data.pkl" and len(parts) <= 2:
                    member = info
                    break
            if member is None or member.file_size > _PICKLE_LIMIT:
                return None
            payload = archive.read(member)
        return _KeyNamesUnpickler(io.BytesIO(payload), encoding="utf-8", errors="replace").load()
    except Exception:
        return None


def _unwrap_state(state: Any) -> Optional[dict]:
    if not isinstance(state, dict):
        return None
    for wrapper in _WRAPPERS:
        inner = state.get(wrapper)
        if isinstance(inner, dict):
            state = inner
            break
    if len(state) == 1:
        (only,) = state.values()
        if isinstance(only, dict):
            state = only
    return state


def state_keys(path) -> Optional[list[str]]:
    suffix = Path(path).suffix.lower()
    if suffix == ".safetensors":
        header = safetensors_header(path)
        return None if header is None else [key for key in header if key != "__metadata__"]
    if suffix in (".pth", ".pt"):
        state = _unwrap_state(zip_state_skeleton(path))
        return None if state is None else [str(key) for key in state]
    return None


def without_common_prefix(keys: Iterable[str]) -> list[str]:
    names = list(keys)
    for prefix in _COMMON_PREFIXES:
        if names and all(name.startswith(prefix) for name in names):
            names = [name[len(prefix):] for name in names]
    return names


def has_nafnet_keys(keys: Optional[Iterable[str]]) -> bool:
    present = set(without_common_prefix(keys or ()))
    return all(key in present for key in NAFNET_KEYS)


_VERDICTS: dict[tuple, tuple[bool, tuple]] = {}
_VERDICT_LOCK = threading.Lock()


def classify(path) -> tuple[bool, tuple]:
    """``(is a NAFNet state dict, declared version)`` — cached by path, size and mtime."""
    try:
        stat = os.stat(path)
    except OSError:
        return False, ()
    key = (os.path.abspath(str(path)), int(stat.st_size), int(stat.st_mtime_ns))
    with _VERDICT_LOCK:
        cached = _VERDICTS.get(key)
    if cached is not None:
        return cached
    if Path(path).suffix.lower() not in MODEL_EXTENSIONS:
        verdict: tuple[bool, tuple] = (False, ())
    else:
        is_nafnet = has_nafnet_keys(state_keys(path))
        verdict = (is_nafnet, modelspec_version(path) if is_nafnet else ())
    with _VERDICT_LOCK:
        _VERDICTS[key] = verdict
    return verdict


__all__ = [
    "classify", "has_nafnet_keys", "is_zip_checkpoint", "modelspec_version", "safetensors_header",
    "state_keys", "version_tuple", "without_common_prefix", "zip_state_skeleton",
]
