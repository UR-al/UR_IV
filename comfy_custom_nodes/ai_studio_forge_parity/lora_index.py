"""LoRA index for the picker: names -> run/epoch grouping, safetensors header summary, model family, trigger word.

No Qt and no ComfyUI imports; the caller passes ``full_path`` (``folder_paths.get_full_path`` in practice). Only the
safetensors *header* is read (``degrid_files.safetensors_header``), results are cached per ``(name, size, mtime_ns)``.
``parse_lora_path`` mirrors ``web/ai_studio/lora_names.js`` ``parseLoraPath`` — keep both in step with
``js_tests/fixtures/lora_names_golden.json`` (test_lora_names_golden).
"""
from __future__ import annotations

import json
import logging
import math
import os
import re
from datetime import datetime
from pathlib import Path
from typing import Callable, Optional

from .degrid_files import safetensors_header
from .lora_atomic import atomic_path
from .studio_choices import unet_family

log = logging.getLogger("ai_studio.lora_index")

_CACHE_VERSION = 1
_TOP_TAGS = 10

_ALGOS = frozenset(("lokr", "loha", "locon", "lora", "dora", "tsm", "full"))
_OPTIMIZERS = frozenset(("came", "adamw", "adamw8bit", "prodigy", "adafactor"))
_SIZE = re.compile(r"[0-9]+(\.[0-9]+)?b")
_BATCH = re.compile(r"b[0-9]+")
_LR = re.compile(r"[0-9]+e[0-9]+")
_EPOCH_SUFFIX = re.compile(r"-([0-9]{6})\Z")
_SPACE_CLASS = r"[ \t\n\v\f\r\u00a0\u1680\u2000-\u200a\u2028\u2029\u202f\u205f\u3000\ufeff]*"  # = JS \s
_GROUP_TAG = re.compile(_SPACE_CLASS + r"\(@[^)]*\)" + _SPACE_CLASS + r"\Z")
_FAMILY_BASES = re.compile(r"anima(?!gine)|moss|krea", re.IGNORECASE)  # not Animagine XL (an SDXL model)
_JS_SPACE = (" \t\n\v\f\r\u00a0\u1680\u2000\u2001\u2002\u2003\u2004\u2005\u2006\u2007\u2008\u2009\u200a"
             "\u2028\u2029\u202f\u205f\u3000\ufeff")  # what JS String.trim removes (Python strip() differs)


def _trim(text: str) -> str:
    return text.strip(_JS_SPACE)


def is_recipe_token(token: str) -> bool:
    t = token.lower()
    return bool(_SIZE.fullmatch(t) or t in _ALGOS or t in _OPTIMIZERS or _BATCH.fullmatch(t) or _LR.fullmatch(t))


_MODEL_FOLDERS = frozenset(("anima", "moss", "sdxl", "krea2", "_unsorted"))  # D:\LoRA's top folders since 11-16


def _is_algo_folder(name: str) -> bool:
    return _trim(name).lower() in _ALGOS


def _is_model_folder(first: str, second: str) -> bool:
    """A known model folder name, or a first folder that is not an algorithm folder followed by one that is (so a new
    algorithm folder, or one with a stray space/BOM from a hand move, under ANIMA/MOSS/... still keeps its runs apart)."""
    return _trim(first).lower() in _MODEL_FOLDERS or (not _is_algo_folder(first) and _is_algo_folder(second))


def parse_lora_path(entry: str) -> dict:
    """Where a ComfyUI LoRA list entry sits: [model/]algo/[_downloaded/]group/[run folder/]file; see lora_names.js.

    The optional model folder (D:\\LoRA\\ANIMA\\LoKr\\... since 11-16) is a known model folder name, or a first folder that
    is not an algorithm folder followed by one that is (``_is_model_folder``); without this, ANIMA was read as the algo,
    LoKr as the group and the group folder as the run, so every run of a group fell into one card."""
    parts = [p for p in re.split(r"[\\/]+", str(entry or "")) if p != ""]
    file = parts.pop() if parts else ""
    dirs = parts
    model = dirs[0] if len(dirs) >= 2 and _is_model_folder(dirs[0], dirs[1]) else ""
    if model:
        dirs = dirs[1:]
    stem = re.sub(r"\.[^.]+\Z", "", file)
    algo = dirs[0] if dirs else ""
    downloaded = len(dirs) > 1 and dirs[1] == "_downloaded"
    group_at = 2 if downloaded else 1
    group = dirs[group_at] if len(dirs) > group_at else ""
    below = dirs[group_at + 1:]
    suffix = _EPOCH_SUFFIX.search(stem)
    epoch = int(suffix.group(1)) if suffix else None
    run = below[0] if below else _EPOCH_SUFFIX.sub("", stem)
    final = bool(below) and stem == run

    tokens = [t for t in (_trim(t) for t in run.split("_")) if t]
    recipe = [t for t in tokens if is_recipe_token(t)]
    first = next((i for i, t in enumerate(tokens) if is_recipe_token(t)), -1)
    if first < 0:
        words = tokens
    else:
        head = "_".join(tokens[:first])
        words = [head] + [t for t in tokens[first + 1:] if not is_recipe_token(t)]
    run_short = " ".join(w for w in words if w)

    group_label = _trim(_GROUP_TAG.sub("", group)) or group
    return {"model": model, "algo": algo, "group": group, "group_label": group_label, "run": run, "run_short": run_short,
            "recipe": recipe, "epoch": epoch, "final": final, "downloaded": downloaded,
            "run_key": "/".join((model, algo, group, run) if model else (algo, group, run))}


def _text(value) -> Optional[str]:
    if value is None:
        return None
    text = str(value).strip()
    return text if text and text.lower() not in ("none", "null") else None


def _int(value) -> Optional[int]:
    text = _text(value)
    if text is None:
        return None
    try:
        return int(float(text)) if re.fullmatch(r"[0-9]+(\.[0-9]+)?", text) else None
    except (ValueError, OverflowError):
        return None


def _jsonish(value):
    """kohya stores dicts as JSON strings; accept either form, else None."""
    if isinstance(value, (dict, list)):
        return value
    try:
        return json.loads(value) if isinstance(value, str) else None
    except ValueError:
        return None


def _date(meta: dict) -> Optional[str]:
    text = _text(meta.get("modelspec.date"))
    if text:
        return text
    started = _text(meta.get("ss_training_started_at"))
    if started is None:
        return None
    try:
        return datetime.fromtimestamp(float(started)).strftime("%Y-%m-%d %H:%M")
    except (ValueError, OverflowError, OSError):
        return started


def _top_tags(meta: dict) -> list:
    freq = _jsonish(meta.get("ss_tag_frequency"))
    totals: dict = {}
    if isinstance(freq, dict):
        for tags in freq.values():
            if not isinstance(tags, dict):
                continue
            for tag, count in tags.items():
                if isinstance(count, bool) or not isinstance(count, (int, float)) or not math.isfinite(count):
                    count = 0
                totals[str(tag)] = totals.get(str(tag), 0) + int(count)
    ranked = sorted(totals.items(), key=lambda kv: (-kv[1], kv[0]))
    return [[tag, count] for tag, count in ranked[:_TOP_TAGS]]


def _dataset_dirs(meta: dict) -> list:
    dirs = _jsonish(meta.get("ss_dataset_dirs"))
    return [str(k) for k in dirs] if isinstance(dirs, (dict, list)) else []


def _algo(meta: dict, network_args) -> Optional[str]:
    if isinstance(network_args, dict) and _text(network_args.get("algo")):
        return _text(network_args.get("algo")).lower()
    module = _text(meta.get("ss_network_module"))
    return module.rsplit(".", 1)[-1].lower() if module else None


def header_summary(header: Optional[dict]) -> dict:
    """The few training-metadata fields the picker shows; anything absent or malformed is None."""
    meta = header.get("__metadata__") if isinstance(header, dict) else None
    meta = meta if isinstance(meta, dict) and meta else None
    out = {"has_meta": meta is not None}
    meta = meta or {}
    network_args = _jsonish(meta.get("ss_network_args"))
    out.update({
        "epoch": _int(meta.get("ss_epoch")),
        "num_epochs": _int(meta.get("ss_num_epochs")),
        "steps": _int(meta.get("ss_steps")),
        "max_steps": _int(meta.get("ss_max_train_steps")),
        "base_model": _text(meta.get("ss_sd_model_name")),
        "algo": _algo(meta, network_args),
        "network_args": network_args if network_args is not None else _text(meta.get("ss_network_args")),
        "precision": _text(meta.get("ss_mixed_precision")),
        "optimizer": _text(meta.get("ss_optimizer")),
        "lr": _text(meta.get("ss_learning_rate")),
        "scheduler": _text(meta.get("ss_lr_scheduler")),
        "date": _date(meta),
        "dataset_dirs": _dataset_dirs(meta),
        "top_tags": _top_tags(meta),
        "sha": _text(meta.get("sshs_model_hash")) or _text(meta.get("ss_new_sd_model_hash")),
    })
    # metadata that yields no usable field (e.g. only ss_epoch="None") counts as no metadata
    out["has_meta"] = out["has_meta"] and any(v not in (None, [], "") for k, v in out.items() if k != "has_meta")
    return out


# D:\LoRA's model folders -> the family of a LoRA in them (= lora_names.js MODEL_FAMILIES). "anima" = an Anima LoRA of
# unknown size: the picker shows it for every Anima checkpoint (2.9B, 3.8B, base) and hides it for MOSS / Krea2 ones.
# _unsorted and any other folder give no family (shown everywhere).
_MODEL_FAMILIES = {"anima": "anima", "moss": "moss", "krea2": "krea2", "sdxl": "sdxl"}
# Folder families that decide before the header and the run name (lora_family): the folder is the user's own sorting.
_FOLDER_FIRST = frozenset(("moss", "krea2", "sdxl"))


def model_folder_family(model: str) -> Optional[str]:
    """The family a model folder name stands for, case-blind and trimmed like JS ``trim()``; None for _unsorted or an
    unknown folder. Mirrors ``lora_names.js`` ``familyOfModelFolder`` (golden rows with ``modelFamily``)."""
    return _MODEL_FAMILIES.get(_trim(str(model or "")).lower())


def lora_family(summary: dict, parsed: dict) -> Optional[str]:
    """Model family a LoRA was trained for. A MOSS / KREA2 / SDXL model folder (``model_folder_family``) decides first:
    the user sorted the file there, whatever its header or name says. Otherwise (an ANIMA folder, _unsorted, none) the
    header's base model if it names a known family (not "animagine"), else the run name's 2.9b / 3.8b, else the folder
    ("anima" for an ANIMA run of unknown size, None without one)."""
    folder = model_folder_family(parsed.get("model") or "")
    if folder in _FOLDER_FIRST:
        return folder
    base = summary.get("base_model")
    if base and _FAMILY_BASES.search(base):
        return unet_family(base)
    for token in parsed.get("recipe") or []:
        if token.lower() in ("2.9b", "3.8b"):
            return token.lower()
    return folder


def _before_bars(text) -> str:
    """A trigger up to the first ``|||`` (captions like ``@00rux ||| male focus``), stripped."""
    return str(text).split("|||", 1)[0].strip()


def trigger_of(summary: dict, board_run: Optional[dict]) -> str:
    """Board run trigger first, else the first ``@`` tag among the most frequent training tags, else empty; each cut at
    the first ``|||`` (an empty cut falls through to the next source)."""
    if isinstance(board_run, dict) and _text(board_run.get("trigger")):
        board = _before_bars(_text(board_run.get("trigger")))
        if board:
            return board
    for tag, _count in summary.get("top_tags") or []:
        if str(tag).startswith("@") and _before_bars(tag):
            return _before_bars(tag)
    return ""


def load_board(board_root) -> dict:
    """``{runs, marks}`` from ``<board_root>/board_db.json``; missing or broken file -> empty maps."""
    out = {"runs": {}, "marks": {}}
    if not board_root or not str(board_root).strip():
        return out
    try:
        data = json.loads((Path(board_root) / "board_db.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return out
    if isinstance(data, dict):
        for key in out:
            if isinstance(data.get(key), dict):
                out[key] = data[key]
    return out


class LoraIndex:
    """Index of LoRA files keyed by list entry; headers are re-read only for files whose size or mtime changed."""

    def __init__(self, cache_path):
        self.cache_path = Path(cache_path)
        self._files: dict = self._load()

    def _load(self) -> dict:
        try:
            data = json.loads(self.cache_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return {}
        files = data.get("files") if isinstance(data, dict) and data.get("version") == _CACHE_VERSION else None
        return files if isinstance(files, dict) else {}

    def _save(self) -> None:
        try:
            self.cache_path.parent.mkdir(parents=True, exist_ok=True)
            with atomic_path(self.cache_path) as tmp:
                tmp.write_text(json.dumps({"version": _CACHE_VERSION, "files": self._files}, ensure_ascii=False),
                               encoding="utf-8")
        except OSError as exc:
            log.warning("LoRA index cache not saved: %s", exc)

    def _summary(self, name: str, path: str, stat: os.stat_result, fresh: dict) -> dict:
        cached = self._files.get(name)
        if (isinstance(cached, dict) and cached.get("size") == stat.st_size
                and cached.get("mtime_ns") == stat.st_mtime_ns and isinstance(cached.get("summary"), dict)):
            fresh[name] = cached
            return cached["summary"]
        summary = header_summary(safetensors_header(path))
        fresh[name] = {"size": stat.st_size, "mtime_ns": stat.st_mtime_ns, "summary": summary}
        return summary

    def build(self, names: list, full_path: Callable[[str], Optional[str]], board: dict) -> list:
        runs = (board or {}).get("runs") if isinstance(board, dict) else None
        runs = runs if isinstance(runs, dict) else {}
        fresh: dict = {}
        items = []
        for name in names:
            try:
                path = full_path(name)
                stat = os.stat(path) if path else None
            except (OSError, ValueError, TypeError):
                stat = None
            if stat is None:
                continue
            parsed = parse_lora_path(name)
            summary = self._summary(name, path, stat, fresh)
            epoch = summary["epoch"] if summary["epoch"] is not None else parsed["epoch"]
            items.append({
                "name": name, "stem": Path(name.replace("\\", "/")).stem, "size": stat.st_size,
                "mtime": stat.st_mtime, "run_key": parsed["run_key"], "epoch": epoch, "final": parsed["final"],
                "num_epochs": summary["num_epochs"], "family": lora_family(summary, parsed),
                "algo": summary["algo"] or parsed["algo"].lower() or None, "precision": summary["precision"],
                "date": summary["date"], "trigger": trigger_of(summary, runs.get(parsed["run"])),
                "has_meta": summary["has_meta"], "downloaded": parsed["downloaded"], "model": parsed["model"],
                "group": parsed["group"],
                "group_label": parsed["group_label"], "run": parsed["run"], "run_short": parsed["run_short"],
                "recipe": parsed["recipe"],
            })
        if fresh != self._files:
            self._files = fresh
            self._save()
        return items
