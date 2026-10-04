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
from .degrid_nodes import (
    NODE_CLASS_MAPPINGS as _DEGRID_NODES,
    NODE_DISPLAY_NAME_MAPPINGS as _DEGRID_DISPLAY_NAMES,
    install_unload_release_hook as _install_degrid_unload_release_hook,
    register_degrid_folder as _register_degrid_folder,
)
from .generation import NODE_CLASS_MAPPINGS as _GENERATION_NODES
from .guidance import NODE_CLASS_MAPPINGS as _GUIDANCE_NODES
from .h3_cache_nodes import NODE_CLASS_MAPPINGS as _H3_CACHE_NODES
from .relight import NODE_CLASS_MAPPINGS as _RELIGHT_NODES
from .sam3_nodes import (
    NODE_CLASS_MAPPINGS as _SAM3_NODES,
    NODE_DISPLAY_NAME_MAPPINGS as _SAM3_DISPLAY_NAMES,
    install_unload_release_hook as _install_sam3_unload_release_hook,
)
# The four sam-extra ports (pack 1.7.0). Keep this order: importing each module registers its
# Hires checkpoint/UNET override carry in ``hires_carry`` (applied in registration order), so an
# override MODEL gets Extra Schedulers -> Extra Samplers -> Anima SPEED -> Colorcraft, with
# Colorcraft's post-CFG function added last (PLAN R2/R9/R13).
from .extra_schedulers import (
    NODE_CLASS_MAPPINGS as _EXTRA_SCHEDULER_NODES,
    NODE_DISPLAY_NAME_MAPPINGS as _EXTRA_SCHEDULER_DISPLAY_NAMES,
    install_extra_schedulers as _install_extra_schedulers,
)
from .extra_samplers_nodes import (
    NODE_CLASS_MAPPINGS as _EXTRA_SAMPLER_NODES,
    NODE_DISPLAY_NAME_MAPPINGS as _EXTRA_SAMPLER_DISPLAY_NAMES,
)
from .extra_samplers_register import install_extra_samplers as _install_extra_samplers
from .speed_nodes import (
    NODE_CLASS_MAPPINGS as _SPEED_NODES,
    NODE_DISPLAY_NAME_MAPPINGS as _SPEED_DISPLAY_NAMES,
)
from .colorcraft_nodes import (
    NODE_CLASS_MAPPINGS as _COLORCRAFT_NODES,
    NODE_DISPLAY_NAME_MAPPINGS as _COLORCRAFT_DISPLAY_NAMES,
)


__version__ = "1.7.0"

# ComfyUI's unload-all-models (/free, OOM recovery, --disable-smart-memory)
# also releases the SAM3 bundle this pack keeps between runs. No-op outside
# ComfyUI (the app imports the relight module from this package).
_install_sam3_unload_release_hook()
# The same for the cached VAE DeGrid NAFNet, and ComfyUI's ``degrid`` model
# folder (ComfyUI/models/degrid, merged into an existing entry). Both no-op
# outside ComfyUI.
_install_degrid_unload_release_hook()
_register_degrid_folder()
# Sampler/scheduler names in ComfyUI's tables, in a fixed order: beta57 (``compat``, at its import)
# -> the nine Extra Schedulers -> the twelve Extra Samplers. A name another provider already holds
# is never replaced (skipped, logged once, left out of the nodes' ``owned_names``). With
# ``required=False`` neither call raises; outside ComfyUI both do nothing.
_install_extra_schedulers(required=False)
_install_extra_samplers(required=False)


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
    _DEGRID_NODES,
    _EXTRA_SCHEDULER_NODES,
    _EXTRA_SAMPLER_NODES,
    _SPEED_NODES,
    _COLORCRAFT_NODES,
)

NODE_DISPLAY_NAME_MAPPINGS = {
    node_id: node_id.replace("ForgeNeo", "Forge Neo ")
    for node_id in NODE_CLASS_MAPPINGS
}
NODE_DISPLAY_NAME_MAPPINGS.update(_SAM3_DISPLAY_NAMES)
NODE_DISPLAY_NAME_MAPPINGS.update(_ANIMA38_DISPLAY_NAMES)
NODE_DISPLAY_NAME_MAPPINGS.update(_ANIMA_LLLITE_DISPLAY_NAMES)
NODE_DISPLAY_NAME_MAPPINGS.update(_DEGRID_DISPLAY_NAMES)
NODE_DISPLAY_NAME_MAPPINGS.update(_EXTRA_SCHEDULER_DISPLAY_NAMES)
NODE_DISPLAY_NAME_MAPPINGS.update(_EXTRA_SAMPLER_DISPLAY_NAMES)
NODE_DISPLAY_NAME_MAPPINGS.update(_SPEED_DISPLAY_NAMES)
NODE_DISPLAY_NAME_MAPPINGS.update(_COLORCRAFT_DISPLAY_NAMES)


__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
