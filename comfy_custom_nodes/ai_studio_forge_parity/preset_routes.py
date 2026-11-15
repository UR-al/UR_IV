"""HTTP routes of the studio's prompt-box presets (``studio_presets``), under ``/aistudio/studio``.

- ``GET  /presets``  ``{doc, limits, broken}``: the stored presets, the length / count limits, and whether a file is
  there that could not be parsed (``doc`` is then empty; the first change keeps that file as ``.bak``)
- ``POST /presets``  one change ``{op, ...}`` (see ``studio_presets.apply``) -> ``{doc, limits}`` (+ ``id`` for ``add``)

Each ``handle_*`` is a pure ``(status, payload)`` function.  ``register_preset_handlers`` is called by
``studio_routes.register_studio_handlers`` with that module's ``add_handler`` and ``_route``, so this module imports
nothing from it.
"""
from __future__ import annotations

import logging
from pathlib import Path

from . import lora_store, studio_presets
from .compat import folder_paths_module

log = logging.getLogger("ai_studio.preset_routes")


def _store() -> Path:
    return lora_store.store_dir(Path(folder_paths_module().get_user_directory()))


def handle_presets_get(_argument=None):
    try:
        doc, broken = studio_presets.load_state(_store())
    except OSError as exc:  # the file is there but cannot be opened: not an empty store
        log.warning("prompt presets not read: %s", exc)
        return 500, {"error": f"프리셋 파일을 열지 못했어요: {exc}"}
    return 200, {"doc": doc, "limits": studio_presets.limits(), "broken": broken}


def handle_presets_post(body):
    if not isinstance(body, dict):
        return 400, {"error": "body must be an object"}
    try:
        result = studio_presets.apply(_store(), body)
    except studio_presets.PresetError as exc:
        return exc.status, {"error": str(exc)}
    except OSError as exc:
        log.warning("prompt presets not saved: %s", exc)
        return 500, {"error": f"프리셋 파일을 읽거나 쓰지 못했어요: {exc}"}
    return 200, {**result, "limits": studio_presets.limits()}


def register_preset_handlers(add_handler, route) -> None:
    add_handler("GET", "/presets", route(handle_presets_get, read="none"))
    add_handler("POST", "/presets", route(handle_presets_post))
