"""Training-sample lookup and thumbnails for the LoRA browser.

Finds the sample PNGs a training run wrote for one epoch (or the board's
four-in-one strip) and builds cached webp thumbnails. Read-only on the
run/board folders. No Qt or ComfyUI imports.
"""
import glob
import json
import logging
from pathlib import Path

from PIL import Image

from .lora_atomic import atomic_path

log = logging.getLogger(__name__)

MOVE_LOG = "move_log.jsonl"
MAX_SAMPLES = 4
MAX_COVER_PIXELS = 40_000_000  # covers: a bigger source (an XYZ grid, an upscale) is skipped, never decoded


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


def run_dirs(full_path: Path, run: str, anima_root: Path, manifest: Path) -> list[Path]:
    """Existing folders that may hold the run's ``sample/`` dir, best first."""
    cands: list[Path | None] = []
    for obj in _json_lines(Path(full_path).parent / MOVE_LOG):
        cands.append(_src_parent(obj.get("src")))
        break
    last = None
    for obj in _json_lines(Path(manifest)):
        if obj.get("run") == run and isinstance(obj.get("src"), str):
            last = obj["src"]
    cands.append(Path(last) if last else None)
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


def board_strip(board_root: Path, run: str, epoch: int) -> Path | None:
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
