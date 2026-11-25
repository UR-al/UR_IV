"""HTTP routes of the LoRA picker (index, detail, thumb, prefs, preview) under ``/aistudio/studio``.

Each ``handle_*`` is a pure ``(status, payload)`` function; aiohttp wrappers live at the bottom.
Only names present in the current LoRA list are accepted, so a request can never leave the LoRA folders.
"""
from __future__ import annotations

import asyncio
import hashlib
import logging
import os
import threading
import time
from pathlib import Path

from . import lora_samples, lora_store, studio_routes
from .compat import folder_paths_module
from .degrid_files import safetensors_header
from .lora_index import LoraIndex, header_summary, load_board
from .lora_previews import OutputScan

log = logging.getLogger("ai_studio.lora_routes")

INDEX_CACHE = "lora_index.json"
SCAN_CACHE = "lora_output_scan.json"
THUMBS_DIR = "thumbs"
MANIFEST_NAME = "_manifest.jsonl"
CACHE_HEADERS = {"Cache-Control": "max-age=600"}

_SCANS: dict = {}  # cache path -> OutputScan (one object, so its lock guards the cache file)
_SCAN_LOCK = threading.Lock()
_SCAN_THREAD: threading.Thread | None = None

INDEX_TTL_SECONDS = 10
MAX_THUMB_FILES = 2000
_THUMB_PRUNE_INTERVAL = 64
_THUMB_LOCK = threading.Lock()
_THUMB_WRITES: dict[str, int] = {}
_clock = time.monotonic
_INDEX_LOCK = threading.Lock()  # held while rebuilding: concurrent callers wait, then share the result
_INDEX: dict = {}  # last build: key, at, items, by_name, by_run_key, board


def reset_index_cache() -> None:
    with _INDEX_LOCK:
        _INDEX.clear()


def _error(message: str, status: int = 400):
    return status, {"error": message}


# ---------------------------------------------------------------------------------------------- background scan

def _scan_for(store: Path) -> OutputScan:
    key = str(store / SCAN_CACHE)
    with _SCAN_LOCK:
        if key not in _SCANS:
            _SCANS[key] = OutputScan(key)
        return _SCANS[key]


def start_output_scan(store: Path, output_dir) -> bool:
    """Start one background ``OutputScan.update``; False if one is still running (never two at once)."""
    global _SCAN_THREAD
    scan = _scan_for(store)

    def work():
        try:
            scan.update(output_dir)
        except Exception:
            log.exception("LoRA output scan failed")

    with _SCAN_LOCK:
        if _SCAN_THREAD is not None and _SCAN_THREAD.is_alive():
            return False
        _SCAN_THREAD = threading.Thread(target=work, name="ai-studio-lora-scan", daemon=True)
        _SCAN_THREAD.start()
        return True


# ---------------------------------------------------------------------------------------------- shared context

def _drop_list_cache(folder_paths) -> None:
    """Forget ComfyUI's cached LoRA list (kept while the folder mtimes look unchanged, which exFAT does not bump)."""
    cache = getattr(folder_paths, "filename_list_cache", None)
    if isinstance(cache, dict):
        cache.pop("loras", None)


def _lora_names(folder_paths, fresh: bool):
    """``(names, error)``: ComfyUI's LoRA list, listed again for ``fresh`` and after an OSError (a run folder deleted,
    the USB drive unplugged: ComfyUI's cache check stats every folder it saw); still failing -> ``([], message)``."""
    if fresh:
        _drop_list_cache(folder_paths)
    try:
        return list(folder_paths.get_filename_list("loras")), None
    except OSError as exc:
        log.warning("LoRA list failed (%s); listing again without ComfyUI's cache", exc)
    _drop_list_cache(folder_paths)
    try:
        return list(folder_paths.get_filename_list("loras")), None
    except OSError as exc:
        log.warning("LoRA list cannot be read: %s", exc)
        return [], f"LoRA 폴더를 읽지 못했어요: {exc}"


class _Context:
    """One request's view: user store, settings, the current LoRA index and the board."""

    def __init__(self, fresh: bool = False, reuse: bool = False):
        """``reuse`` accepts an index built less than ``INDEX_TTL_SECONDS`` ago (thumb/detail/preview);
        the index route rebuilds every time (``fresh`` also drops the cache file)."""
        self.folder_paths = folder_paths_module()
        self.store = lora_store.store_dir(Path(self.folder_paths.get_user_directory()))
        self.settings = lora_store.load_settings(self.store)
        names, self.list_error = _lora_names(self.folder_paths, fresh)
        key = (str(self.store), self.settings["board_root"], tuple(names))
        with _INDEX_LOCK:
            now = _clock()
            if not (reuse and _INDEX.get("key") == key and now - _INDEX["at"] < INDEX_TTL_SECONDS):
                cache = self.store / INDEX_CACHE
                if fresh:
                    try:
                        cache.unlink()
                    except OSError:
                        pass
                board = load_board(self.settings["board_root"])
                # an unreadable list builds nothing: the header cache file stays for when the drive is back
                items = [] if self.list_error else LoraIndex(cache).build(names, self.full_path, board)
                by_run_key: dict[str, list] = {}
                for item in items:
                    by_run_key.setdefault(item["run_key"], []).append(item)
                _INDEX.clear()
                _INDEX.update(key=key, at=_clock(), items=items, board=board,
                              by_name={item["name"]: item for item in items},
                              by_run_key=by_run_key)
            self.items = _INDEX["items"]
            self.board = _INDEX["board"]
            self.by_name = _INDEX["by_name"]
            self.by_run_key = _INDEX["by_run_key"]

    def full_path(self, name: str):
        return self.folder_paths.get_full_path("loras", name)

    def run_items(self, item: dict) -> list:
        return list(self.by_run_key.get(item["run_key"], ()))

    def manifest(self, path: Path) -> Path:
        roots = []
        getter = getattr(self.folder_paths, "get_folder_paths", None)
        if getter is not None:
            try:
                roots = [Path(r) for r in getter("loras")]
            except Exception:
                roots = []
        for root in roots:
            try:
                if Path(path).is_relative_to(root):
                    return root / MANIFEST_NAME
            except (OSError, ValueError):
                continue
        return Path(path).parent / MANIFEST_NAME

    def dirs_for(self, item: dict) -> list:
        path = self.full_path(item["name"])
        if not path:
            return []
        anima = self.settings["anima_root"]
        return lora_samples.run_dirs(Path(path), item["run"], Path(anima) if anima else None,
                                     self.manifest(Path(path)))

    def epoch_item(self, item: dict, epoch: int):
        for other in self.run_items(item):
            if lora_samples.file_epoch(other) == epoch:
                return other
        return None

    def last_item(self, item: dict) -> dict:
        known = [i for i in self.run_items(item) if lora_samples.file_epoch(i) is not None]
        return max(known, key=lora_samples.file_epoch) if known else item


def _find(query, ctx: _Context):
    name = query.get("name") if isinstance(query, dict) else None
    return ctx.by_name.get(name) if isinstance(name, str) else None


def _int(value, default=None):
    if value in (None, ""):
        return default
    try:
        return int(value)
    except (TypeError, ValueError):
        raise ValueError("a whole number is required")


# ---------------------------------------------------------------------------------------------- sample lookup

def _sample_source(ctx: _Context, item: dict, epoch: int, index: int):
    """``(path, crop)`` of the ``index``-th sample of an epoch: C: sample file, else a board strip tile."""
    owner = ctx.epoch_item(item, epoch) or item
    samples = lora_samples.epoch_samples(ctx.dirs_for(owner), item["run"], epoch)
    if 0 <= index < len(samples):
        return samples[index], None
    board_root = ctx.settings["board_root"]
    strip = lora_samples.board_strip(Path(board_root) if board_root else None, item["run"], epoch)
    if strip is not None and index >= 0:
        try:
            tiles = lora_samples.strip_tiles(strip)
        except Exception:
            return None
        if index < len(tiles):
            return strip, tiles[index]
    return None


def _cover_source(ctx: _Context, item: dict):
    """``(kind, path, crop)`` of the run's cover: custom -> last epoch sample 0 -> newest output image (not huge)."""
    custom = lora_store.custom_preview(ctx.store, item["run_key"])
    if custom is not None:
        return "custom", custom, None
    last = ctx.last_item(item)
    epoch = lora_samples.file_epoch(last)
    if epoch is not None:
        found = _sample_source(ctx, item, epoch, 0)
        if found is not None:
            return "sample", found[0], found[1]
    hit = _scan_for(ctx.store).latest_for([i["stem"] for i in ctx.run_items(item)])
    if hit and Path(hit["path"]).is_file() and not _too_large(hit["path"]):
        return "output", Path(hit["path"]), None
    return None, None, None


def _too_large(path) -> bool:
    """Above ``lora_samples.MAX_COVER_PIXELS`` (header only); an unreadable file is left to the thumbnail (404)."""
    try:
        return lora_samples.pixel_count(path) > lora_samples.MAX_COVER_PIXELS
    except Exception:
        return False


def _prune_thumbs(thumb_dir: Path) -> None:
    """Keep at most ``MAX_THUMB_FILES`` cached webps in ``thumb_dir``, removing oldest mtime first."""
    limit = max(0, int(MAX_THUMB_FILES))
    try:
        entries = []
        with os.scandir(thumb_dir) as it:
            for entry in it:
                if not entry.name.endswith(".webp"):
                    continue
                try:
                    if entry.is_file():
                        entries.append((entry.stat().st_mtime_ns, entry.path))
                except OSError:
                    continue
    except OSError:
        return
    if len(entries) <= limit:
        return
    entries.sort(key=lambda pair: pair[0])
    for _mtime, path in entries[:len(entries) - limit]:
        try:
            os.unlink(path)
        except OSError:
            continue


def _maybe_prune_thumbs(thumb_dir: Path) -> None:
    step = 1 if MAX_THUMB_FILES < _THUMB_PRUNE_INTERVAL else _THUMB_PRUNE_INTERVAL
    key = str(thumb_dir)
    with _THUMB_LOCK:
        count = _THUMB_WRITES.get(key, 0) + 1
        if count < step:
            _THUMB_WRITES[key] = count
            return
        _THUMB_WRITES[key] = 0
    _prune_thumbs(thumb_dir)


def _thumb_path(ctx: _Context, src: Path, crop, max_pixels=None):
    """Cached webp named by source path + mtime + size + crop; above ``max_pixels``: ValueError, nothing decoded."""
    size = ctx.settings["thumb_size"]
    stat = Path(src).stat()
    key = f"{src}|{stat.st_mtime_ns}|{stat.st_size}|{size}|{crop}"
    thumb_dir = ctx.store / THUMBS_DIR
    dst = thumb_dir / (hashlib.sha1(key.encode("utf-8")).hexdigest() + ".webp")
    existed = dst.is_file()
    out = lora_samples.make_thumb(Path(src), dst, size, crop, max_pixels)
    if not existed and out.is_file():
        _maybe_prune_thumbs(thumb_dir)
    return out


# ---------------------------------------------------------------------------------------------- handlers

def handle_index(query):
    query = query if isinstance(query, dict) else {}
    ctx = _Context(fresh=query.get("fresh") in ("1", "true"))
    try:
        start_output_scan(ctx.store, ctx.folder_paths.get_output_directory())
    except Exception:
        log.exception("LoRA output scan not started")
    runs = {item["run"] for item in ctx.items}
    marks, board_runs = ctx.board.get("marks", {}), ctx.board.get("runs", {})
    stars, labels = {}, {}
    for run in runs:
        mark = marks.get(run)
        starred = mark.get("stars") if isinstance(mark, dict) else None
        if isinstance(starred, list) and starred:
            stars[run] = [e for e in starred if isinstance(e, int) and not isinstance(e, bool)]
        board_run = board_runs.get(run)
        prompts = board_run.get("prompts") if isinstance(board_run, dict) else None
        if isinstance(prompts, list) and prompts:
            labels[run] = [p.get("label", "") for p in prompts if isinstance(p, dict)]
    versions = lora_store.custom_preview_versions(ctx.store)
    body = {"items": ctx.items, "prefs": lora_store.load_prefs(ctx.store),
            "board_stars": stars, "prompt_labels": labels,
            "cover_versions": {k: v for k, v in versions.items() if k in ctx.by_run_key}}
    if ctx.list_error:
        body["error"] = ctx.list_error
    return 200, body


def handle_detail(query):
    ctx = _Context(reuse=True)
    item = _find(query, ctx)
    if item is None:
        return _error("unknown LoRA", 404)
    path = ctx.full_path(item["name"])
    summary = header_summary(safetensors_header(path)) if path else header_summary(None)
    mark = ctx.board.get("marks", {}).get(item["run"])
    starred = set(mark.get("stars") or []) if isinstance(mark, dict) and isinstance(mark.get("stars"), list) else set()
    board_run = ctx.board.get("runs", {}).get(item["run"])
    prompts = board_run.get("prompts") if isinstance(board_run, dict) else None
    epochs, seen = [], set()
    for other in sorted(ctx.run_items(item), key=lambda i: (lora_samples.file_epoch(i) is None,
                                                            lora_samples.file_epoch(i) or 0)):
        epoch = lora_samples.file_epoch(other)
        if epoch is None or epoch in seen:
            continue
        seen.add(epoch)
        count = len(lora_samples.epoch_samples(ctx.dirs_for(other), item["run"], epoch))
        board_root = ctx.settings["board_root"]
        if not count and lora_samples.board_strip(Path(board_root) if board_root else None, item["run"], epoch):
            count = lora_samples.MAX_SAMPLES
        epochs.append({"epoch": epoch, "name": other["name"], "samples": count, "starred": epoch in starred})
    cover, _path, _crop = _cover_source(ctx, item)
    return 200, {"item": item, "summary": summary, "trigger": item.get("trigger", ""), "epochs": epochs,
                 "prompt_labels": [p.get("label", "") for p in prompts if isinstance(p, dict)]
                 if isinstance(prompts, list) else [],
                 "cover": cover}


def thumb_response(query):
    """``(200, Path)`` of a cached webp, or ``(status, {"error"})``."""
    if not isinstance(query, dict):
        return _error("bad request")
    ctx = _Context(reuse=True)
    item = _find(query, ctx)
    if item is None:
        return _error("unknown LoRA", 404)
    kind = query.get("kind", "cover")
    if kind not in ("cover", "sample"):
        return _error("kind must be cover or sample")
    max_pixels = None
    try:
        if kind == "cover":
            max_pixels = lora_samples.MAX_COVER_PIXELS
            found, src, crop = _cover_source(ctx, item)
            if src is None:
                return _error("no cover image", 404)
            if found == "custom":
                return 200, src
        else:
            index = _int(query.get("i"), 0)
            epoch = _int(query.get("epoch"), lora_samples.file_epoch(ctx.last_item(item)))
            if epoch is None:
                return _error("no epoch", 404)
            hit = _sample_source(ctx, item, epoch, index)
            if hit is None:
                return _error("no sample image", 404)
            src, crop = hit
    except ValueError as exc:
        return _error(str(exc))
    try:
        return 200, _thumb_path(ctx, src, crop, max_pixels)
    except Exception as exc:  # corrupt or unreadable image
        log.warning("LoRA thumbnail failed for %s: %s", src, exc)
        return _error("image cannot be read", 404)


def handle_prefs_get(_argument=None):
    return 200, lora_store.load_prefs(lora_store.store_dir(Path(folder_paths_module().get_user_directory())))


def handle_prefs_set(body):
    if not isinstance(body, dict):
        return _error("body must be an object")
    store = lora_store.store_dir(Path(folder_paths_module().get_user_directory()))
    return 200, lora_store.save_prefs(store, body)


def handle_preview(body):
    if not isinstance(body, dict):
        return _error("body must be an object")
    run_key = body.get("run_key")
    if not isinstance(run_key, str) or not run_key:
        return _error("run_key is required")
    ctx = _Context(reuse=True)
    if run_key not in ctx.by_run_key:
        return _error("unknown run", 404)
    if body.get("clear") is True:
        lora_store.clear_custom_preview(ctx.store, run_key)
        return 200, {"ok": True, "cleared": True}
    ref = body.get("image")
    if not isinstance(ref, dict) or ref.get("type") not in ("output", "input"):
        return _error("image must be {filename, subfolder, type} in output or input")
    try:
        path = studio_routes.resolve_image(ref)
    except studio_routes._BadRequest as exc:
        return _error(str(exc))
    try:
        lora_store.set_custom_preview(ctx.store, run_key, path, ctx.settings["thumb_size"])
    except Exception as exc:  # unreadable / corrupt image
        log.warning("LoRA preview not set from %s: %s", path, exc)
        return _error("image cannot be read")
    return 200, {"ok": True, "cover": "custom"}


# ---------------------------------------------------------------------------------------------- aiohttp wrappers

def _thumb_route():
    async def route(request):
        from aiohttp import web

        def work():
            status, payload = thumb_response(dict(request.rel_url.query))
            if isinstance(payload, Path):
                return status, payload.read_bytes()
            return status, payload

        try:
            status, payload = await asyncio.to_thread(work)
        except Exception as exc:
            studio_routes.LOGGER.exception("LoRA thumb route failed")
            return web.json_response({"error": f"{type(exc).__name__}: {exc}"}, status=500)
        if isinstance(payload, bytes):
            return web.Response(body=payload, content_type="image/webp", headers=CACHE_HEADERS)
        return web.json_response(payload, status=status)

    return route


def register_lora_handlers() -> None:
    route = studio_routes._route
    studio_routes.add_handler("GET", "/loras/index", route(handle_index, read="query"))
    studio_routes.add_handler("GET", "/loras/detail", route(handle_detail, read="query"))
    studio_routes.add_handler("GET", "/loras/thumb", _thumb_route())
    studio_routes.add_handler("GET", "/loras/prefs", route(handle_prefs_get, read="none"))
    studio_routes.add_handler("POST", "/loras/prefs", route(handle_prefs_set))
    studio_routes.add_handler("POST", "/loras/preview", route(handle_preview))
