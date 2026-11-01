"""Which SAM3 files Easy SAM3 can read. No torch: only a safetensors header is opened."""
from __future__ import annotations

import json
import struct

TRANSFORMERS_PREFIX = "detector_model."  # the layout of Hugging Face transformers; the original one starts with "detector."
MAX_HEADER_BYTES = 64 * 1024 * 1024


def is_transformers_layout(path) -> bool:
    """A ``.safetensors`` SAM3 saved in transformers' layout.

    Easy SAM3 loads with ``strict=False``, so such a file leaves the model on random weights: it finds nothing, without
    an error. Anything else, a ``.pt`` or an unreadable header included, is left to the loader.
    """
    if not str(path).lower().endswith(".safetensors"):
        return False
    try:
        with open(path, "rb") as handle:
            size = struct.unpack("<Q", handle.read(8))[0]
            if size > MAX_HEADER_BYTES:
                return False
            header = json.loads(handle.read(size))
    except (OSError, ValueError, struct.error):
        return False
    return isinstance(header, dict) and any(key.startswith(TRANSFORMERS_PREFIX) for key in header)
