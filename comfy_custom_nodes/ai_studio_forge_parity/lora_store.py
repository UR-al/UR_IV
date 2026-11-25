"""User-folder store for the LoRA browser: settings, prefs, custom previews.

Everything lives under ``<user_dir>/default/ai_studio`` (see ``store_dir``).
No Qt or ComfyUI imports; the caller passes ``folder_paths.get_user_directory()``.
"""
import hashlib
import json
import logging
import threading
from pathlib import Path

from . import lora_samples
from .lora_atomic import atomic_path

log = logging.getLogger(__name__)

_LOCK = threading.Lock()

SETTINGS_FILE = "lora_browser.json"
PREFS_FILE = "lora_prefs.json"
PREVIEWS_DIR = "previews"
PREVIEWS_INDEX = "previews.json"

DEFAULT_SETTINGS = {
    "anima_root": r"C:\Users\URL\Desktop\UR\ANIMA",
    "board_root": r"C:\Users\URL\Desktop\UR\board",
    "thumb_size": 384,
}
SORTS = ("name", "date", "group", "recent")
RECENT_MAX = 20


def store_dir(user_dir) -> Path:
    path = Path(user_dir) / "default" / "ai_studio"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _read_json(path: Path) -> dict:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return data if isinstance(data, dict) else {}


def _write_json(path: Path, data: dict) -> None:
    """Atomic save: write a unique temp file next to the target, then ``os.replace``."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with atomic_path(path) as tmp:
        tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def load_settings(dir) -> dict:
    settings = dict(DEFAULT_SETTINGS)
    raw = _read_json(Path(dir) / SETTINGS_FILE)
    for key, default in DEFAULT_SETTINGS.items():
        value = raw.get(key)
        if isinstance(default, int):
            if isinstance(value, int) and not isinstance(value, bool) and value > 0:
                settings[key] = value
        elif isinstance(value, str) and value.strip():
            settings[key] = value
    return settings


def _str_list(value, limit=None) -> list:
    if not isinstance(value, list):
        return []
    out = []
    for item in value:
        if isinstance(item, str) and item and item not in out:
            out.append(item)
    return out[:limit] if limit else out


def _normalise_prefs(prefs) -> dict:
    prefs = prefs if isinstance(prefs, dict) else {}

    def flag(key):
        return prefs.get(key) if isinstance(prefs.get(key), bool) else False

    sort = prefs.get("sort")
    return {
        "favorites": _str_list(prefs.get("favorites")),
        "recent": _str_list(prefs.get("recent"), RECENT_MAX),
        "sort": sort if sort in SORTS else "name",
        "show_all": flag("show_all"),
        "tree_open": flag("tree_open"),
        "close_after_add": flag("close_after_add"),
    }


def load_prefs(dir) -> dict:
    return _normalise_prefs(_read_json(Path(dir) / PREFS_FILE))


def save_prefs(dir, prefs: dict) -> dict:
    clean = _normalise_prefs(prefs)
    _write_json(Path(dir) / PREFS_FILE, clean)
    return clean


def _preview_name(run_key: str) -> str:
    return hashlib.sha1(run_key.encode("utf-8")).hexdigest() + ".webp"


def _previews(dir) -> dict:
    raw = _read_json(Path(dir) / PREVIEWS_INDEX)
    return {k: v for k, v in raw.items() if isinstance(k, str) and isinstance(v, str)}


def set_custom_preview(dir, run_key: str, image, size: int, *, max_pixels: int | None = None) -> str:
    from PIL import Image  # lazy: only needed when a preview is set

    dir = Path(dir)
    name = _preview_name(run_key)
    target = dir / PREVIEWS_DIR / name
    target.parent.mkdir(parents=True, exist_ok=True)
    limit = lora_samples.MAX_COVER_PIXELS if max_pixels is None else max_pixels
    with Image.open(image) as im:
        if limit is not None and im.size[0] * im.size[1] > limit:
            raise ValueError(f"image too large for a preview: {im.size[0]}x{im.size[1]}")
        im.load()
        im = im.convert("RGBA" if "A" in im.getbands() else "RGB")
        im.thumbnail((size, size))
        with atomic_path(target) as tmp:
            im.save(tmp, format="WEBP", quality=85)
    with _LOCK:
        index = _previews(dir)
        index[run_key] = name
        _write_json(dir / PREVIEWS_INDEX, index)
    return name


def clear_custom_preview(dir, run_key: str) -> None:
    dir = Path(dir)
    with _LOCK:
        index = _previews(dir)
        name = index.pop(run_key, None)
        if name is None:
            return
        _write_json(dir / PREVIEWS_INDEX, index)
    try:
        (dir / PREVIEWS_DIR / _preview_name(run_key)).unlink()
    except OSError:
        pass


def custom_preview_versions(dir) -> dict:
    """``{run_key: version}`` of the custom previews: the file's mtime in microseconds (new whenever one is set again;
    exact in a JS number), for a cover URL the browser has not cached."""
    dir = Path(dir)
    out = {}
    for run_key in _previews(dir):
        try:
            out[run_key] = (dir / PREVIEWS_DIR / _preview_name(run_key)).stat().st_mtime_ns // 1000
        except OSError:
            continue
    return out


def custom_preview(dir, run_key: str):
    dir = Path(dir)
    if run_key not in _previews(dir):
        return None
    path = dir / PREVIEWS_DIR / _preview_name(run_key)
    return path if path.is_file() else None
