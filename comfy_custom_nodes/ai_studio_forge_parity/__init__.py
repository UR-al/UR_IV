"""AI Studio Pro's bundled Forge Neo compatibility nodes for ComfyUI."""
from __future__ import annotations

from .anima38_nodes import (
    NODE_CLASS_MAPPINGS as _ANIMA38_NODES,
    NODE_DISPLAY_NAME_MAPPINGS as _ANIMA38_DISPLAY_NAMES,
)
from .anima_lllite import (
    NODE_CLASS_MAPPINGS as _ANIMA_LLLITE_NODES,
    NODE_DISPLAY_NAME_MAPPINGS as _ANIMA_LLLITE_DISPLAY_NAMES,
)
from .anima_lora_nodes import NODE_CLASS_MAPPINGS as _ANIMA_LORA_NODES
from .generation import NODE_CLASS_MAPPINGS as _GENERATION_NODES
from .guidance import NODE_CLASS_MAPPINGS as _GUIDANCE_NODES
from .h3_cache_nodes import NODE_CLASS_MAPPINGS as _H3_CACHE_NODES
from .relight import NODE_CLASS_MAPPINGS as _RELIGHT_NODES
from .sam3_nodes import (
    NODE_CLASS_MAPPINGS as _SAM3_NODES,
    NODE_DISPLAY_NAME_MAPPINGS as _SAM3_DISPLAY_NAMES,
    install_unload_release_hook as _install_sam3_unload_release_hook,
)


__version__ = "1.4.1"

# ComfyUI's unload-all-models (/free, OOM recovery, --disable-smart-memory)
# also releases the SAM3 bundle this pack keeps between runs. No-op outside
# ComfyUI (the app imports the relight module from this package).
_install_sam3_unload_release_hook()


def _merge_node_maps(*maps):
    merged = {}
    for mapping in maps:
        duplicates = set(merged).intersection(mapping)
        if duplicates:
            names = ", ".join(sorted(duplicates))
            raise RuntimeError(f"Duplicate AI Studio ComfyUI node IDs: {names}")
        merged.update(mapping)
    return merged


NODE_CLASS_MAPPINGS = _merge_node_maps(
    _ANIMA38_NODES,
    _ANIMA_LORA_NODES,
    _GUIDANCE_NODES,
    _GENERATION_NODES,
    _SAM3_NODES,
    _H3_CACHE_NODES,
    _RELIGHT_NODES,
    _ANIMA_LLLITE_NODES,
)

NODE_DISPLAY_NAME_MAPPINGS = {
    node_id: node_id.replace("ForgeNeo", "Forge Neo ")
    for node_id in NODE_CLASS_MAPPINGS
}
NODE_DISPLAY_NAME_MAPPINGS.update(_SAM3_DISPLAY_NAMES)
NODE_DISPLAY_NAME_MAPPINGS.update(_ANIMA38_DISPLAY_NAMES)
NODE_DISPLAY_NAME_MAPPINGS.update(_ANIMA_LLLITE_DISPLAY_NAMES)


__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
