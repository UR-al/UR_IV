"""Training-sample lookup and thumbnails for the LoRA browser.

Finds the sample PNGs a training run wrote for one epoch (or the board's
four-in-one strip) and builds cached webp thumbnails. Read-only on the
run/board folders. No Qt or ComfyUI imports.
"""
import glob
import json
import logging
import os
import threading
from pathlib import Path

from PIL import Image

from .lora_atomic import atomic_path

log = logging.getLogger(__name__)

MOVE_LOG = "move_log.jsonl"
MAX_SAMPLES = 4
MAX_COVER_PIXELS = 40_000_000  # covers: a bigger source (an XYZ grid, an upscale) is skipped, never decoded

_MANIFEST_CACHE_MAX = 32
_MANIFEST_LOCK = threading.Lock()
_MANIFEST_CACHE: dict[str, tuple[int, int, dict[str, list[Path]]]] = {}

_MOVE_LOG_CACHE_MAX = 128
_MOVE_LOG_LOCK = threading.Lock()
_MOVE_LOG_CACHE: dict[str, tuple[int, int, tuple[Path, ...]]] = {}


def reset_manifest_cache() -> None:
    with _MANIFEST_LOCK:
        _MANIFEST_CACHE.clear()


def reset_move_log_cache() -> None:
    with _MOVE_LOG_LOCK:
        _MOVE_LOG_CACHE.clear()


def _json_lines(path: Path):
    """JSON objects of a .jsonl log; a line that is not UTF-8 JSON (cp949, UTF-16 …) is skipped, never an error."""
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as fh:
            for line in fh:
                try:
                    obj = json.loads(line)
                except ValueError:  # also a line whose bytes were not UTF-8 (now U+FFFD)
                    continue
                if isinstance(obj, dict):
                    yield obj
    except OSError:
        return


def _src_parent(src) -> Path | None:
    if not isinstance(src, str) or not src:
        return None
    return Path(src).parent


def _manifest_runs(manifest: Path) -> dict[str, list[Path]]:
    """Cached ``run -> [src, ...]`` (newest first) for ``manifest``, invalidated on mtime/size change."""
    key = str(manifest)
    try:
        st = os.stat(manifest)
    except OSError:
        with _MANIFEST_LOCK:
            _MANIFEST_CACHE.pop(key, None)
        return {}
    mtime_ns, size = st.st_mtime_ns, st.st_size
    with _MANIFEST_LOCK:
        cached = _MANIFEST_CACHE.get(key)
        if cached is not None and cached[0] == mtime_ns and cached[1] == size:
            return cached[2]
    by_run: dict[str, list[Path]] = {}
    for obj in _json_lines(manifest):
        run_name, src = obj.get("run"), obj.get("src")
        if isinstance(run_name, str) and isinstance(src, str) and src:
            by_run.setdefault(run_name, []).append(Path(src))
    for paths in by_run.values():
        paths.reverse()
    with _MANIFEST_LOCK:
        if len(_MANIFEST_CACHE) >= _MANIFEST_CACHE_MAX and key not in _MANIFEST_CACHE:
            _MANIFEST_CACHE.clear()
        _MANIFEST_CACHE[key] = (mtime_ns, size, by_run)
    return by_run


def _move_log_parents(move_log: Path) -> tuple[Path, ...]:
    """Cached ``_src_parent`` entries (newest first) for ``move_log``, invalidated on mtime/size change."""
    key = str(move_log)
    try:
        st = os.stat(move_log)
    except OSError:
        with _MOVE_LOG_LOCK:
            _MOVE_LOG_CACHE.pop(key, None)
        return ()
    mtime_ns, size = st.st_mtime_ns, st.st_size
    with _MOVE_LOG_LOCK:
        cached = _MOVE_LOG_CACHE.get(key)
        if cached is not None and cached[0] == mtime_ns and cached[1] == size:
            return cached[2]
    move_parents = [
        p for obj in _json_lines(move_log)
        if (p := _src_parent(obj.get("src"))) is not None
    ]
    move_parents.reverse()
    result = tuple(move_parents)
    with _MOVE_LOG_LOCK:
        if len(_MOVE_LOG_CACHE) >= _MOVE_LOG_CACHE_MAX and key not in _MOVE_LOG_CACHE:
            _MOVE_LOG_CACHE.clear()
        _MOVE_LOG_CACHE[key] = (mtime_ns, size, result)
    return result


def run_dirs(full_path: Path, run: str, anima_root: Path | str | None, manifest: Path) -> list[Path]:
    """Existing folders that may hold the run's ``sample/`` dir, best first."""
    cands: list[Path | None] = []
    cands.extend(_move_log_parents(Path(full_path).parent / MOVE_LOG))
    # Every manifest line of the run, newest first: a later line can name a folder that is gone (the 11-16 relocate lines
    # point at the old D:\LoRA\<algo>\<group>\<run>), and must not hide the training folder an earlier line names.
    cands.extend(_manifest_runs(Path(manifest)).get(run, ()))
    if anima_root and str(anima_root).strip():
        cands.append(Path(anima_root) / run)
    out: list[Path] = []
    for c in cands:
        try:
            if c is not None and c.is_dir() and c not in out:
                out.append(c)
        except OSError:
            continue
    return out


def epoch_samples(run_dirs: list[Path], run: str, epoch: int) -> list[Path]:
    """Newest sample per prompt index for ``epoch``, ordered by index, max 4."""
    pattern = f"{glob.escape(run)}_e{int(epoch):06d}_??_*.png"
    best: dict[int, Path] = {}
    for d in run_dirs:
        sample_dir = Path(d) / "sample"
        try:
            files = list(sample_dir.glob(pattern))
        except OSError:
            continue
        for f in files:
            rest = f.name[len(run) + len("_e000000_"):]
            if not rest[:2].isdigit():
                continue
            idx = int(rest[:2])
            # the 14-digit timestamp sorts lexically; name is the tiebreak
            if idx not in best or f.name[len(run):] > best[idx].name[len(run):]:
                best[idx] = f
    return [best[i] for i in sorted(best)][:MAX_SAMPLES]


def board_strip(board_root: Path | str | None, run: str, epoch: int) -> Path | None:
    if not board_root or not str(board_root).strip():
        return None
    name = run[1:] if run.startswith("@") else run
    path = Path(board_root) / "site" / "s" / name / f"e{int(epoch):03d}.webp"
    try:
        return path if path.is_file() else None
    except OSError:
        return None


def file_epoch(item: dict) -> int | None:
    epoch = item.get("epoch")
    if epoch is not None:
        return int(epoch)
    if item.get("final") and item.get("num_epochs") is not None:
        return int(item["num_epochs"])
    return None


def pixel_count(src) -> int:
    """Width x height from the image header; the pixels are not decoded."""
    with Image.open(src) as im:
        width, height = im.size
    return width * height


def make_thumb(src: Path, dst: Path, size: int, crop: tuple | None = None, max_pixels: int | None = None) -> Path:
    """Long side ``size`` webp of ``src`` (optionally cropped); reused if newer than src. A source above ``max_pixels``
    raises ValueError before its pixels are decoded."""
    src, dst = Path(src), Path(dst)
    try:
        if dst.is_file() and dst.stat().st_mtime >= src.stat().st_mtime:
            return dst
    except OSError:
        pass
    dst.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(src) as im:
        if max_pixels is not None and im.size[0] * im.size[1] > max_pixels:
            raise ValueError(f"image too large for a thumbnail: {im.size[0]}x{im.size[1]}")
        im.load()
        im = im.convert("RGB")
        if crop:
            im = im.crop(tuple(int(v) for v in crop))
        im.thumbnail((size, size), Image.LANCZOS)  # never upscales
        try:
            with atomic_path(dst) as tmp:
                im.save(tmp, "WEBP", quality=82)
        except OSError:
            if not dst.is_file():  # a concurrent writer that won the replace left a complete file: fine
                raise
    return dst


def strip_tiles(strip: Path) -> list[tuple]:
    """Crop boxes of the four samples in a board strip (vertical if taller than wide)."""
    with Image.open(strip) as im:
        w, h = im.size
    if h > w:
        step = h // MAX_SAMPLES
        return [(0, i * step, w, (i + 1) * step) for i in range(MAX_SAMPLES)]
    step = w // MAX_SAMPLES
    return [(i * step, 0, (i + 1) * step, h) for i in range(MAX_SAMPLES)]
