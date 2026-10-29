"""HTTP route registry for the Anima studio web extension.

No ComfyUI or aiohttp import at module level: the desktop app imports this pack
without a Comfy host.  Later tasks add handlers with ``add_handler``; they are
mounted under ``ROUTE_PREFIX`` once ComfyUI's ``PromptServer`` exists.
"""
from __future__ import annotations

import sys

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
