"""One shared helper for atomic file writes: a unique temp name next to the target, replaced in, removed on failure.

A fixed ``<name>.tmp`` collides when two requests write the same target at once (``os.replace`` fails on
Windows); the pid + thread id in the name keeps concurrent writers apart. No Qt and no ComfyUI imports.
"""
from __future__ import annotations

import os
import threading
from contextlib import contextmanager
from pathlib import Path


def unique_tmp(target) -> Path:
    target = Path(target)
    return target.with_name(f"{target.name}.{os.getpid()}.{threading.get_ident()}.tmp")


@contextmanager
def atomic_path(target):
    """Yield a unique temp path; on a clean exit ``os.replace`` it onto ``target``, on any error delete it."""
    target = Path(target)
    tmp = unique_tmp(target)
    try:
        yield tmp
        os.replace(tmp, target)
    except BaseException:
        try:
            tmp.unlink()
        except OSError:
            pass
        raise
