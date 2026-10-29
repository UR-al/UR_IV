"""HTTP routes of the Anima studio web extension.

No ComfyUI or aiohttp import at module level: the desktop app imports this pack
without a Comfy host.  Handlers are added with ``add_handler`` and mounted under
``ROUTE_PREFIX`` once ComfyUI's ``PromptServer`` exists, so every handler is added while this
module imports (``register_studio_handlers`` at the bottom), never lazily after startup.

The routes (all JSON; a failure is ``{"error": ...}`` with status 400):

- ``GET  /choices?kind=...``  picker lists (``studio_choices``)
- ``POST /xyz/plan``          ``{state, axes, options}`` -> the planned X/Y/Z cells (``xyz_plan``)
- ``POST /xyz/grid``          generated cell images -> grid PNG(s) in ``output/AIStudio/xyz`` (``xyz_grid_draw``)
- ``POST /open_output``       opens the output folder in the file manager (Windows only)
- ``POST /to_input``          copies an output / temp / input image into the input folder (answer: name, width, height)

Each ``handle_*`` is a pure function ``-> (status, dict)``; the aiohttp wrappers are thin.  An image is only ever
named by ``{filename, subfolder, type}`` and must resolve inside ComfyUI's output, temp or input folder.
"""
from __future__ import annotations

import asyncio
import dataclasses
import logging
import os
import random
import re
import shutil
import sys
from pathlib import Path

from . import studio_choices, xyz_plan
from .compat import folder_paths_module

LOGGER = logging.getLogger("ai_studio_forge_parity")

ROUTE_PREFIX = "/aistudio/studio"

# (METHOD, path under ROUTE_PREFIX) -> handler, in registration order.
_HANDLERS: dict[tuple[str, str], object] = {}


def _server_instance():
    server = sys.modules.get("server")
    return getattr(getattr(server, "PromptServer", None), "instance", None)


def _mount(instance) -> None:
    mounted = getattr(instance, "_aistudio_studio_routes", None)
    if mounted is None:
        mounted = set()
        instance._aistudio_studio_routes = mounted
    for key, handler in _HANDLERS.items():
        if key in mounted:
            continue
        method, path = key
        instance.routes.route(method, ROUTE_PREFIX + path)(handler)
        mounted.add(key)


def add_handler(method: str, path: str, handler) -> None:
    """Register ``handler`` for ``METHOD ROUTE_PREFIX+path`` (mounted at once if the server is up)."""
    if not path.startswith("/"):
        raise ValueError("route path must start with '/'")
    _HANDLERS[(method.upper(), path)] = handler
    instance = _server_instance()
    if instance is not None:
        _mount(instance)


def register_routes() -> bool:
    """Mount every known handler once; False (and no effect) outside ComfyUI."""
    instance = _server_instance()
    if instance is None:
        return False
    _mount(instance)
    return True


# ---------------------------------------------------------------------------------------------- shared helpers

IMAGE_TYPES = ("output", "temp", "input")
IMAGE_SUFFIXES = (".png", ".jpg", ".jpeg", ".webp", ".bmp", ".gif")
GRID_SUBFOLDER = "AIStudio/xyz"
GRID_STEM = "xyz_grid"
MAX_AXIS_COUNT = 256
MAX_GRID_CELLS = 2048
MAX_MARGIN = 1000


class _BadRequest(Exception):
    pass


def _error(message: str, status: int = 400):
    return status, {"error": message}


def _directory(kind: str) -> Path:
    try:
        path = folder_paths_module().get_directory_by_type(kind)
    except RuntimeError as exc:
        raise _BadRequest(str(exc)) from exc
    if not path:
        raise _BadRequest(f"unknown image type: {kind}")
    return Path(path)


def _relative_parts(text, what: str, *, allow_empty: bool) -> list[str]:
    """``text`` as clean path parts: no absolute path, drive, ``..`` or NUL."""
    if not isinstance(text, str):
        raise _BadRequest(f"{what} must be text")
    if "\0" in text:
        raise _BadRequest(f"bad {what}")
    text = text.replace("\\", "/")
    if text.startswith("/") or ":" in text:
        raise _BadRequest(f"{what} must be a relative path")
    parts = [part for part in text.split("/") if part not in ("", ".")]
    if ".." in parts:
        raise _BadRequest(f"{what} must stay inside the folder")
    if not parts and not allow_empty:
        raise _BadRequest(f"{what} is empty")
    return parts


def resolve_image(ref) -> Path:
    """The file a ``{filename, subfolder, type}`` reference names, only inside ComfyUI's output / temp / input folder."""
    if not isinstance(ref, dict):
        raise _BadRequest("image must be {filename, subfolder, type}")
    kind = ref.get("type")
    if kind not in IMAGE_TYPES:
        raise _BadRequest("image type must be output, temp or input")
    name_parts = _relative_parts(ref.get("filename"), "filename", allow_empty=False)
    if len(name_parts) != 1:
        raise _BadRequest("filename must not contain a folder")
    folder_parts = _relative_parts(ref.get("subfolder") or "", "subfolder", allow_empty=True)
    base = _directory(kind).resolve()
    path = base.joinpath(*folder_parts, name_parts[0]).resolve()  # resolves symlinks too
    if not path.is_relative_to(base):
        raise _BadRequest("image must stay inside the ComfyUI folder")
    if path.suffix.lower() not in IMAGE_SUFFIXES:
        raise _BadRequest("not an image file")
    if not path.is_file():
        raise _BadRequest("image not found")
    return path


def _int_field(source: dict, key: str, default: int, low: int, high: int, what: str) -> int:
    value = source.get(key, default)
    if isinstance(value, bool) or not isinstance(value, int) or not low <= value <= high:
        raise _BadRequest(f"{what} must be a whole number from {low} to {high}")
    return value


def _dict_field(body: dict, key: str) -> dict:
    value = body.get(key)
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise _BadRequest(f"{key} must be an object")
    return value


# ---------------------------------------------------------------------------------------------- choices / plan

def handle_choices(query) -> tuple[int, dict]:
    kind = query.get("kind") if isinstance(query, dict) else None
    try:
        return 200, studio_choices.choices(kind or "")
    except ValueError as exc:
        return _error(str(exc))


def handle_plan(body, *, families=None, loras=None) -> tuple[int, dict]:
    """Plan an X/Y/Z grid.  ``families`` (file -> "2.9b" | "base" | "3.8b") defaults to the installed UNET files,
    ``loras`` to ComfyUI's LoRA list (read only when a cell has a ``<lora:...>`` tag; a tag no file matches is an error).

    A plan the planner rejects (bad values, mixed families, ...) is still a 200 answer: ``cells`` is empty and
    ``errors`` lists the reasons.  400 is for a body that is not ``{state, axes, options}`` at all."""
    try:
        if not isinstance(body, dict):
            raise _BadRequest("body must be an object")
        state, options = _dict_field(body, "state"), _dict_field(body, "options")
        axes = body.get("axes", [])
        if not isinstance(axes, list):
            raise _BadRequest("axes must be a list")
    except _BadRequest as exc:
        return _error(str(exc))
    if families is None:
        families = studio_choices.unet_families()
    if loras is None:
        loras = studio_choices.lora_names
    plan = xyz_plan.plan_xyz(state, axes, options, rng=random.Random(), families=families, loras=loras)
    return 200, dataclasses.asdict(plan)


# ---------------------------------------------------------------------------------------------- grid

def _grid_labels(value) -> dict:
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise _BadRequest("labels must be an object")
    labels = {}
    for letter in ("x", "y", "z"):
        texts = value.get(letter) or []
        if not isinstance(texts, list):
            raise _BadRequest(f"labels.{letter} must be a list")
        labels[letter] = [str(text) for text in texts]
    return labels


def _load_cells(cells, nx: int, ny: int, nz: int) -> dict:
    from PIL import Image

    if not isinstance(cells, list):
        raise _BadRequest("cells must be a list")
    loaded = {}
    for cell in cells:
        if not isinstance(cell, dict):
            raise _BadRequest("each cell must be an object")
        key = (_int_field(cell, "ix", -1, 0, nx - 1, "cell ix"),
               _int_field(cell, "iy", -1, 0, ny - 1, "cell iy"),
               _int_field(cell, "iz", -1, 0, nz - 1, "cell iz"))
        if cell.get("image") is None:
            loaded[key] = None
            continue
        path = resolve_image(cell["image"])
        try:
            with Image.open(path) as image:
                image.load()
                loaded[key] = image.copy()
        except (OSError, ValueError) as exc:
            raise _BadRequest(f"cannot read image {path.name}: {exc}") from exc
    return loaded


def _write_png(path: Path, image, infotext: str) -> None:
    """Write ``image`` to a NEW file ``path`` (FileExistsError if taken) with ``infotext`` as the ``parameters`` chunk."""
    from PIL import PngImagePlugin

    info = PngImagePlugin.PngInfo()
    info.add_text("parameters", infotext)
    with open(path, "xb") as handle:
        image.save(handle, format="PNG", pnginfo=info, compress_level=4)


def _save_grid(directory: Path, image, infotext: str) -> tuple[str, int]:
    """Save as the next free ``xyz_grid_NNNNN.png``; returns the file name and NNNNN."""
    pattern = re.compile(re.escape(GRID_STEM) + r"_(\d+)")
    while True:
        used = [int(match.group(1)) for match in map(pattern.match, os.listdir(directory)) if match]
        number = max(used, default=0) + 1
        name = f"{GRID_STEM}_{number:05}.png"
        try:
            _write_png(directory / name, image, infotext)
        except FileExistsError:
            continue
        return name, number


def handle_grid(body) -> tuple[int, dict]:
    """Draw the grid from generated cell images and save it (and, if asked, the Z sub-grids) with the infotext."""
    from .xyz_grid_draw import draw_xyz

    try:
        if not isinstance(body, dict):
            raise _BadRequest("body must be an object")
        nx = _int_field(body, "nx", 0, 1, MAX_AXIS_COUNT, "nx")
        ny = _int_field(body, "ny", 0, 1, MAX_AXIS_COUNT, "ny")
        nz = _int_field(body, "nz", 0, 1, MAX_AXIS_COUNT, "nz")
        if nx * ny * nz > MAX_GRID_CELLS:
            raise _BadRequest(f"a grid can have at most {MAX_GRID_CELLS} cells")
        options = _dict_field(body, "options")
        margin = _int_field(options, "margin", 0, 0, MAX_MARGIN, "margin")
        row_count = _int_field(options, "row_count", 0, 0, MAX_AXIS_COUNT, "row_count")
        infotext = body.get("infotext", "")
        if not isinstance(infotext, str):
            raise _BadRequest("infotext must be text")
        labels = _grid_labels(body.get("labels"))
        cells = _load_cells(body.get("cells"), nx, ny, nz)
        try:
            grid, subgrids = draw_xyz(cells, nx, ny, nz, labels, draw_legend=bool(options.get("draw_legend", True)),
                                      margin=margin, row_count=row_count)
        except ValueError as exc:
            raise _BadRequest(str(exc)) from exc
        directory = _directory("output").joinpath(*GRID_SUBFOLDER.split("/"))
        directory.mkdir(parents=True, exist_ok=True)
        name, number = _save_grid(directory, grid, infotext)
        saved = []
        if options.get("include_sub_grids") and nz > 1:
            for index, sub in enumerate(subgrids, start=1):
                sub_name = f"{GRID_STEM}_{number:05}_z{index}.png"
                try:
                    _write_png(directory / sub_name, sub, infotext)
                except FileExistsError:
                    continue
                saved.append({"filename": sub_name, "subfolder": GRID_SUBFOLDER, "type": "output"})
    except _BadRequest as exc:
        return _error(str(exc))
    except OSError as exc:
        return _error(f"cannot save the grid: {exc}", 500)
    return 200, {"grid": {"filename": name, "subfolder": GRID_SUBFOLDER, "type": "output"}, "subgrids": saved}


# ---------------------------------------------------------------------------------------------- folders and files

def _is_windows() -> bool:
    return sys.platform == "win32"


def handle_open_output(_body=None) -> tuple[int, dict]:
    if not _is_windows():
        return _error("opening the folder is only available on Windows")
    try:
        path = str(_directory("output"))
        os.startfile(path)  # type: ignore[attr-defined]  # Windows only (guarded above)
    except (_BadRequest, OSError) as exc:
        return _error(f"cannot open the output folder: {exc}")
    return 200, {"path": path}


def _image_size(path: Path) -> dict:
    """``{"width", "height"}`` of an image file (read from its header), or ``{}`` when it cannot be read."""
    from PIL import Image

    try:
        with Image.open(path) as image:
            width, height = image.size
    except (OSError, ValueError):
        return {}
    return {"width": int(width), "height": int(height)}


def handle_to_input(body) -> tuple[int, dict]:
    """Copy an output / temp image into the input folder; a taken name gets ``_1``, ``_2`` ... (never overwrites).

    The answer carries the image's ``width`` / ``height`` too: like Forge's send-to buttons, the panel then runs
    i2i / inpaint at the image's size, not the t2i size."""
    try:
        source = resolve_image(body)
        target_dir = _directory("input")
        target_dir.mkdir(parents=True, exist_ok=True)
        for attempt in range(10000):
            name = source.name if attempt == 0 else f"{source.stem}_{attempt}{source.suffix}"
            try:
                with open(source, "rb") as reader, open(target_dir / name, "xb") as writer:
                    shutil.copyfileobj(reader, writer)
            except FileExistsError:
                continue
            return 200, {"name": name, **_image_size(target_dir / name)}
        raise _BadRequest("too many copies of this image")
    except _BadRequest as exc:
        return _error(str(exc))
    except OSError as exc:
        return _error(f"cannot copy the image: {exc}", 500)


# ---------------------------------------------------------------------------------------------- aiohttp wrappers

def _route(handle, *, read: str = "json"):
    """A thin aiohttp handler: ``read`` is ``"json"`` (request body), ``"query"`` (query string) or ``"none"``."""

    async def route(request):
        from aiohttp import web

        if read == "json":
            try:
                argument = await request.json()
            except Exception:
                return web.json_response({"error": "body is not valid JSON"}, status=400)
        elif read == "query":
            argument = dict(request.rel_url.query)
        else:
            argument = None
        try:
            status, payload = await asyncio.to_thread(handle, argument)
        except Exception as exc:  # keep the server answering; the log has the traceback
            LOGGER.exception("studio route failed")
            return web.json_response({"error": f"{type(exc).__name__}: {exc}"}, status=500)
        return web.json_response(payload, status=status)

    return route


def register_studio_handlers() -> None:
    add_handler("GET", "/choices", _route(handle_choices, read="query"))
    add_handler("POST", "/xyz/plan", _route(handle_plan))
    add_handler("POST", "/xyz/grid", _route(handle_grid))
    add_handler("POST", "/open_output", _route(handle_open_output, read="none"))
    add_handler("POST", "/to_input", _route(handle_to_input))


register_studio_handlers()
