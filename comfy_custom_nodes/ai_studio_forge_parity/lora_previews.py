"""Automatic LoRA previews: the newest output image per LoRA stem, found from the PNG ``parameters`` infotext.

No Qt and no ComfyUI imports. Only PNG header chunks are read (``Image.open(...).info``, never ``load()``), and only
files that are new or whose mtime changed since the last scan. Nothing is written except the cache file we were given.
``update`` may run in a worker thread while ``latest_for`` is called from another: shared state sits behind a lock.
"""
from __future__ import annotations

import json
import logging
import os
import re
import threading
import time
from pathlib import Path
from typing import Callable, Optional

from PIL import Image

from .lora_atomic import atomic_path

log = logging.getLogger("ai_studio.lora_previews")

_CACHE_VERSION = 1
_LORA_TAG = re.compile(r"<lora:([^:<>]+):", re.IGNORECASE)
_MODEL_EXT = re.compile(r"\.(safetensors|ckpt|pt|pth)\Z", re.IGNORECASE)
_SKIP_DIRS = frozenset(("xyz",))  # XYZ grids (output/AIStudio/xyz, up to ~110 MP): never a LoRA's cover


def lora_names_in_infotext(text: str) -> list:
    """Stems of the ``<lora:NAME:...>`` tags in an infotext (folder prefix dropped), without duplicates, in order."""
    out: list = []
    for raw in _LORA_TAG.findall(text or ""):
        stem = _MODEL_EXT.sub("", re.split(r"[\\/]", raw.strip())[-1]).strip()
        if stem and stem not in out:
            out.append(stem)
    return out


def _read_parameters(path: str) -> str:
    """The PNG ``parameters`` text, or "" when absent; header chunks only, pixels are never decoded."""
    with Image.open(path) as img:
        value = img.info.get("parameters")
    if isinstance(value, bytes):
        value = value.decode("utf-8", "replace")
    return value if isinstance(value, str) else ""


def _png_files(root: str):
    """``(path, mtime)`` of every ``.png`` below ``root`` (recursive; symlinked dirs and ``xyz`` folders skipped)."""
    stack = [root]
    while stack:
        try:
            with os.scandir(stack.pop()) as it:
                for entry in it:
                    try:
                        if entry.is_dir(follow_symlinks=False):
                            if entry.name.casefold() not in _SKIP_DIRS:
                                stack.append(entry.path)
                        elif entry.name.lower().endswith(".png") and entry.is_file():
                            yield entry.path, entry.stat().st_mtime
                    except OSError:
                        continue
        except OSError:
            continue


class OutputScan:
    """Incremental scan of the output folder; cache ``{path: [mtime, [stems]]}`` lives in ``cache_path``."""

    def __init__(self, cache_path, clock: Callable[[], float] = time.monotonic):
        self.cache_path = Path(cache_path)
        self._clock = clock
        self._lock = threading.Lock()
        self._files: dict = self._load()
        self._latest: dict = self._rebuild(self._files)
        self._pending = 0

    def _load(self) -> dict:
        try:
            data = json.loads(self.cache_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return {}
        files = data.get("files") if isinstance(data, dict) and data.get("version") == _CACHE_VERSION else None
        if not isinstance(files, dict):
            return {}
        return {p: v for p, v in files.items() if isinstance(v, list) and len(v) == 2
                and isinstance(v[0], (int, float)) and isinstance(v[1], list)}

    def _save(self, files: dict) -> None:
        try:
            self.cache_path.parent.mkdir(parents=True, exist_ok=True)
            with atomic_path(self.cache_path) as tmp:
                tmp.write_text(json.dumps({"version": _CACHE_VERSION, "files": files}, ensure_ascii=False),
                               encoding="utf-8")
        except OSError as exc:
            log.warning("LoRA preview cache not saved: %s", exc)

    @staticmethod
    def _rebuild(files: dict) -> dict:
        latest: dict = {}
        for path, (mtime, stems) in files.items():
            for stem in stems:
                if stem not in latest or mtime > latest[stem]["mtime"]:
                    latest[stem] = {"path": path, "mtime": mtime}
        return latest

    def update(self, output_dir, limit_seconds: float = 20.0) -> dict:
        """Read new or changed PNGs (newest first) until ``limit_seconds`` is used up; the rest stays pending."""
        found = dict(_png_files(str(output_dir)))
        with self._lock:
            known = dict(self._files)
        files = {p: known[p] for p in found if p in known and known[p][0] == found[p]}
        todo = sorted((p for p in found if p not in files), key=lambda p: found[p], reverse=True)
        start = self._clock()
        done = 0
        for path in todo:
            if done and self._clock() - start >= limit_seconds:
                break
            try:
                stems = lora_names_in_infotext(_read_parameters(path))
            except Exception:  # broken / non-PNG content: remember it as "no LoRAs" so it is not re-read
                stems = []
            files[path] = [found[path], stems]
            done += 1
        latest = self._rebuild(files)
        with self._lock:
            changed = files != self._files
            self._files, self._latest, self._pending = files, latest, len(todo) - done
            result = {"latest": dict(latest), "pending": self._pending}
        if changed:
            self._save(files)
        return result

    def latest_for(self, stems: list) -> Optional[dict]:
        """The most recent image among the given stems (a run's epochs), or None."""
        with self._lock:
            hits = [self._latest[s] for s in stems if s in self._latest]
        if not hits:
            return None
        return dict(max(hits, key=lambda h: h["mtime"]))
