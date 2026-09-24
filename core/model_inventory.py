"""Forge Neo / ComfyUI model-library inventory.

The runtime manager owns installation and sharing policy.  This module owns the
read-only catalog seam consumed by generation UI code:

* every distinct file from the selected primary library is retained;
* a secondary file is retained only when it is not the same physical file or a
  bounded sampled-content duplicate with the same normalized filename;
* backend choices are never decorated.  Source grouping is returned as
  metadata alongside the exact strings accepted by the active API.

Large model libraries make full hashing impractical.  Content comparison is
therefore deliberately restricted to secondary candidates with the same
normalized basename/stem *and* size.  At most the first, middle and last 64 KiB
of each candidate is read.
"""

from __future__ import annotations

import hashlib
import os
import re
import stat as _stat
from collections.abc import Mapping, Sequence
from pathlib import Path, PurePosixPath
from typing import Any


MODEL_CATEGORIES = (
    "checkpoints",
    "diffusion_models",
    "loras",
    "vae",
    "text_encoders",
)

_CATEGORY_ALIASES = {
    "checkpoint": "checkpoints",
    "checkpoints": "checkpoints",
    "model": "checkpoints",
    "models": "checkpoints",
    "diffusion_model": "diffusion_models",
    "diffusion_models": "diffusion_models",
    "unet": "diffusion_models",
    "lora": "loras",
    "loras": "loras",
    "vae": "vae",
    "te": "text_encoders",
    "text_encoder": "text_encoders",
    "text_encoders": "text_encoders",
}

_CATEGORY_EXTENSIONS = {
    "checkpoints": frozenset({".ckpt", ".safetensors", ".sft", ".gguf", ".pt", ".pth", ".bin"}),
    "diffusion_models": frozenset({".ckpt", ".safetensors", ".sft", ".gguf", ".pt", ".pth", ".bin"}),
    "loras": frozenset({".ckpt", ".safetensors", ".sft", ".pt", ".pth"}),
    "vae": frozenset({".ckpt", ".safetensors", ".sft", ".pt", ".pth", ".bin"}),
    "text_encoders": frozenset({".ckpt", ".safetensors", ".sft", ".gguf", ".pt", ".pth", ".bin"}),
}

_DEFAULT_SOURCE_NAMES = {
    "forge": "Forge Neo",
    "comfyui": "ComfyUI",
}

_HASH_SUFFIX = re.compile(r"\s+\[[0-9a-f]{6,}\]\s*$", re.IGNORECASE)
_SAMPLE_CHUNK_BYTES = 64 * 1024


def _canonical_engine(value: Any) -> str:
    engine = str(value or "").strip().casefold().replace("-", "_")
    if engine in {"forge", "forge_neo", "webui", "automatic1111"}:
        return "forge"
    if engine in {"comfy", "comfyui"}:
        return "comfyui"
    return engine


def _canonical_category(value: Any) -> str:
    category = str(value or "").strip().casefold().replace("-", "_")
    try:
        return _CATEGORY_ALIASES[category]
    except KeyError as exc:
        raise ValueError(f"unsupported model category: {value!r}") from exc


def _is_placeholder(value: str) -> bool:
    return value.strip().casefold() in {
        "",
        "none",
        "use same checkpoint",
        "use same vae",
        "use checkpoint default",
    }


def _without_suffix(value: str) -> str:
    path = PurePosixPath(value.replace("\\", "/"))
    return str(path.with_suffix("")) if path.suffix else str(path)


def _display_name(relative_name: str) -> str:
    return _without_suffix(relative_name)


def _native_runtime_name(engine: str, category: str, relative_name: str) -> str:
    # Forge's /sdapi/v1/loras names omit the model-file extension.  ComfyUI's
    # LoraLoader choices retain it.  Other module selectors use filenames.
    if engine == "forge" and category == "loras":
        return _without_suffix(relative_name)
    return relative_name


def _name_key(relative_name: str) -> str:
    """Bounded-dedupe key: normalized basename stem only."""
    basename = PurePosixPath(relative_name.replace("\\", "/")).name
    return _without_suffix(basename).casefold()


def _stable_id(source: str, category: str, resolved_path: str) -> str:
    material = f"{source}\0{category}\0{os.path.normcase(resolved_path)}"
    digest = hashlib.sha256(material.encode("utf-8", errors="surrogatepass")).hexdigest()[:20]
    return f"{source}:{category}:{digest}"


def _physical_key(
    path: Path, stat_result: os.stat_result, resolved: str | None = None
) -> tuple[Any, ...]:
    """Return a cheap same-file identity before any sampled I/O.

    inode 가 있으면(NTFS·ext4 등) 경로를 resolve 하지 않는다 — 파일마다 반복되던
    resolve 가 카탈로그 비용의 큰 몫이었다. ``resolved`` 를 이미 알면 재사용한다.
    """
    inode = int(getattr(stat_result, "st_ino", 0) or 0)
    device = int(getattr(stat_result, "st_dev", 0) or 0)
    if inode:
        return ("inode", device, inode)
    if resolved is None:
        resolved = os.path.realpath(path)
    return ("path", os.path.normcase(resolved))


def _sampled_fingerprint(path: Path, size: int) -> str | None:
    """Hash at most three fixed-size chunks, never an entire large model."""
    try:
        offsets = [0]
        if size > _SAMPLE_CHUNK_BYTES:
            offsets.append(max(0, (size - _SAMPLE_CHUNK_BYTES) // 2))
            offsets.append(max(0, size - _SAMPLE_CHUNK_BYTES))
        offsets = list(dict.fromkeys(offsets))

        digest = hashlib.sha256()
        digest.update(str(size).encode("ascii"))
        with path.open("rb") as stream:
            for offset in offsets:
                stream.seek(offset)
                chunk = stream.read(_SAMPLE_CHUNK_BYTES)
                digest.update(offset.to_bytes(8, "little", signed=False))
                digest.update(len(chunk).to_bytes(4, "little", signed=False))
                digest.update(chunk)
        return digest.hexdigest()
    except OSError:
        return None


def _iter_model_files(root: Path, category: str):
    extensions = _CATEGORY_EXTENSIONS[category]
    found: list[tuple[str, Path, os.stat_result]] = []
    try:
        for folder, directories, filenames in os.walk(root, followlinks=False):
            directories.sort(key=str.casefold)
            for filename in sorted(filenames, key=str.casefold):
                if Path(filename).suffix.casefold() not in extensions:
                    continue
                path = Path(folder) / filename
                # VAE sidecars beside checkpoints are not checkpoint choices.
                if category in {"checkpoints", "diffusion_models"} and filename.casefold().endswith(".vae.safetensors"):
                    continue
                try:
                    # stat 1회 — is_file() 을 따로 부르면 같은 파일을 한 번 더 stat 한다.
                    stat_result = os.stat(path)
                    if not _stat.S_ISREG(stat_result.st_mode):
                        continue
                    relative = path.relative_to(root).as_posix()
                except (OSError, ValueError):
                    continue
                found.append((relative, path, stat_result))
    except OSError:
        return
    found.sort(key=lambda item: (item[0].casefold(), item[0]))
    yield from found


def _clean_api_value(value: Any) -> str:
    cleaned = _HASH_SUFFIX.sub("", str(value or "").strip())
    return cleaned.replace("\\", "/")


def _api_values(item: Any) -> list[str]:
    if isinstance(item, Mapping):
        values = [
            item.get("runtimeName"),
            item.get("name"),
            item.get("alias"),
            item.get("title"),
            item.get("model_name"),
            item.get("filename"),
            item.get("path"),
        ]
    else:
        values = [item]
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        cleaned = _clean_api_value(value)
        if not cleaned or _is_placeholder(cleaned):
            continue
        key = cleaned.casefold()
        if key not in seen:
            result.append(cleaned)
            seen.add(key)
    return result


def _preferred_api_name(item: Any) -> str:
    if isinstance(item, Mapping):
        for key in ("runtimeName", "name", "alias", "title", "model_name", "filename"):
            # This value is sent back to the backend.  Keep its spelling,
            # separators and optional title hash exactly as the API supplied it.
            value = str(item.get(key) or "").strip()
            if value and not _is_placeholder(value):
                return value
        return ""
    value = str(item or "").strip()
    return "" if _is_placeholder(value) else value


def _value_variants(value: str) -> dict[str, int]:
    normalized = _clean_api_value(value).strip().lstrip("./").casefold()
    if not normalized:
        return {}
    variants = {normalized: 80}
    without_suffix = _without_suffix(normalized)
    variants[without_suffix] = max(variants.get(without_suffix, 0), 70)
    basename = PurePosixPath(normalized).name
    variants[basename] = max(variants.get(basename, 0), 50)
    stem = _without_suffix(basename)
    variants[stem] = max(variants.get(stem, 0), 40)
    return variants


def _resolved_path_key(value: str) -> str:
    try:
        return os.path.normcase(str(Path(value).expanduser().resolve(strict=False)))
    except (OSError, ValueError):
        return os.path.normcase(os.path.abspath(os.path.expanduser(value)))


def _existing_physical_key(value: str) -> tuple[Any, ...] | None:
    path = Path(value)
    try:
        return _physical_key(path, os.stat(path))
    except (OSError, ValueError):
        return None


class _EntryMatcher:
    """Indexed matcher for one immutable catalog snapshot.

    Matching used to compare every API item against every disk entry and call
    ``samefile`` for each absolute-path pair.  A large Forge library therefore
    performed millions of filesystem stats on the QWebChannel thread.  This
    internal seam indexes the same path/name variants once and limits scoring
    to the small collision buckets that can actually match.
    """

    def __init__(
        self,
        entries: Sequence[Mapping[str, Any]],
        active_engine: str,
        physical_keys: Mapping[str, tuple[Any, ...]] | None = None,
    ):
        self.entries = list(entries)
        self.active_engine = active_engine
        # 카탈로그가 이미 계산한 물리 키(정규화 경로 → 키). 없을 때만 다시 stat 한다.
        known_physical = physical_keys or {}
        self._variant_index: dict[str, list[tuple[int, int]]] = {}
        self._path_index: dict[str, list[int]] = {}
        self._physical_index: dict[tuple[Any, ...], list[int]] = {}
        self._query_physical_cache: dict[str, tuple[Any, ...] | None] = {}

        for index, entry in enumerate(self.entries):
            entry_path = str(entry.get("path") or "")
            entry_variants: dict[str, int] = {}
            for value in (
                entry.get("runtimeName"),
                entry.get("name"),
                entry.get("label"),
                entry_path,
            ):
                if not value:
                    continue
                for variant, rank in _value_variants(str(value)).items():
                    entry_variants[variant] = max(entry_variants.get(variant, 0), rank)
            for variant, rank in entry_variants.items():
                self._variant_index.setdefault(variant, []).append((index, rank))

            if entry_path and os.path.isabs(entry_path):
                # 카탈로그의 path 는 이미 resolve 된 값이라 다시 resolve 하지 않는다.
                path_key = os.path.normcase(entry_path)
                self._path_index.setdefault(path_key, []).append(index)
                physical = known_physical.get(path_key)
                if physical is None:
                    physical = _existing_physical_key(entry_path)
                if physical is not None:
                    self._physical_index.setdefault(physical, []).append(index)

    def best_match(self, item: Any) -> Mapping[str, Any] | None:
        scores: dict[int, int] = {}

        def offer(index: int, score: int) -> None:
            if score > scores.get(index, 0):
                scores[index] = score

        for api_value in _api_values(item):
            if os.path.isabs(api_value):
                path_key = _resolved_path_key(api_value)
                for index in self._path_index.get(path_key, ()):
                    offer(index, 950)

                if path_key not in self._query_physical_cache:
                    self._query_physical_cache[path_key] = _existing_physical_key(api_value)
                physical = self._query_physical_cache[path_key]
                if physical is not None:
                    for index in self._physical_index.get(physical, ()):
                        offer(index, 1000)

            for variant, api_rank in _value_variants(api_value).items():
                for index, entry_rank in self._variant_index.get(variant, ()):
                    offer(index, api_rank + entry_rank)

        best_index = -1
        best_score = 0
        for index, raw_score in scores.items():
            entry = self.entries[index]
            score = raw_score
            if bool(entry.get("primary")):
                score += 3
            if str(entry.get("source") or "") == self.active_engine:
                score += 1
            if score > best_score or (score == best_score and (best_index < 0 or index < best_index)):
                best_index = index
                best_score = score
        return self.entries[best_index] if best_index >= 0 else None


def _trigger_words(item: Any) -> list[str]:
    if not isinstance(item, Mapping):
        return []
    raw = item.get("trigger_words")
    if raw is None:
        raw = item.get("triggerWords")
    if isinstance(raw, str):
        values = [part.strip() for part in raw.split(",")]
    elif isinstance(raw, Sequence) and not isinstance(raw, (bytes, bytearray)):
        values = [str(part).strip() for part in raw]
    else:
        values = []
    return [value for value in values if value][:12]


class ModelInventory:
    """Lazy read-only catalog built from one backend-runtime snapshot."""

    def __init__(self, snapshot: Mapping[str, Any] | None = None, *, active_engine: str | None = None):
        self.snapshot = dict(snapshot or {})
        raw_engines = self.snapshot.get("engines")
        raw_engines = raw_engines if isinstance(raw_engines, Mapping) else {}
        self.engines: dict[str, Mapping[str, Any]] = {}
        for raw_key, value in raw_engines.items():
            engine = _canonical_engine(raw_key)
            if engine and isinstance(value, Mapping):
                self.engines[engine] = value

        requested_active = active_engine if active_engine is not None else self.snapshot.get("activeEngine")
        self.active_engine = _canonical_engine(requested_active)
        requested_primary = _canonical_engine(self.snapshot.get("primaryModelEngine"))
        if requested_primary not in self.engines:
            requested_primary = self.active_engine if self.active_engine in self.engines else ""
        if not requested_primary and self.engines:
            requested_primary = next(iter(self.engines))
        self.primary_engine = requested_primary
        self._catalog_cache: dict[str, list[dict[str, Any]]] = {}
        self._matcher_cache: dict[str, _EntryMatcher] = {}
        # 카테고리별 {정규화 경로: 물리 키} — entry dict 는 그대로 브리지 JSON 으로 나가므로
        # 내부 키를 거기에 싣지 않고 여기에 따로 둔다.
        self._physical_keys: dict[str, dict[str, tuple[Any, ...]]] = {}

    def _source_name(self, engine: str) -> str:
        data = self.engines.get(engine, {})
        return str(data.get("name") or _DEFAULT_SOURCE_NAMES.get(engine) or engine)

    def _engine_order(self) -> list[str]:
        order = list(self.engines)
        if self.primary_engine in order:
            order.remove(self.primary_engine)
            order.insert(0, self.primary_engine)
        return order

    def _catalog(self, category: str) -> list[dict[str, Any]]:
        category = _canonical_category(category)
        cached = self._catalog_cache.get(category)
        if cached is not None:
            return cached

        accepted: list[dict[str, Any]] = []
        physical_keys: dict[str, tuple[Any, ...]] = {}
        physical_seen: set[tuple[Any, ...]] = set()
        content_candidates: dict[tuple[str, int], list[dict[str, Any]]] = {}
        fingerprint_cache: dict[str, str | None] = {}

        for engine in self._engine_order():
            engine_data = self.engines[engine]
            model_paths = engine_data.get("modelPaths")
            model_paths = model_paths if isinstance(model_paths, Mapping) else {}
            roots = model_paths.get(category)
            if isinstance(roots, (str, os.PathLike)):
                roots = [roots]
            elif not isinstance(roots, Sequence):
                roots = []

            primary = engine == self.primary_engine
            root_seen: set[str] = set()
            for root_value in roots:
                root_text = str(root_value or "").strip()
                if not root_text:
                    continue
                root = Path(root_text).expanduser()
                root_key = os.path.normcase(str(root.resolve(strict=False)))
                if root_key in root_seen or not root.is_dir():
                    continue
                root_seen.add(root_key)

                for relative, path, stat_result in _iter_model_files(root, category):
                    physical = _physical_key(path, stat_result)
                    if physical in physical_seen:
                        continue
                    # 파일당 resolve 는 여기 한 번 — 중복 판정 캐시와 entry 가 같이 쓴다.
                    # Path.resolve(strict=False) 는 결과를 다시 stat 하므로 realpath 를 쓴다.
                    resolved = os.path.realpath(path)

                    size = int(stat_result.st_size)
                    duplicate_key = (_name_key(relative), size)
                    if not primary:
                        duplicate = False
                        secondary_fingerprint: str | None = None
                        secondary_fingerprint_computed = False
                        for candidate in content_candidates.get(duplicate_key, []):
                            candidate_path = str(candidate["path"])
                            candidate_fingerprint = fingerprint_cache.get(candidate_path)
                            if candidate_path not in fingerprint_cache:
                                candidate_fingerprint = _sampled_fingerprint(
                                    Path(candidate_path), int(candidate["size"])
                                )
                                fingerprint_cache[candidate_path] = candidate_fingerprint
                            if not secondary_fingerprint_computed:
                                secondary_fingerprint = _sampled_fingerprint(path, size)
                                fingerprint_cache[resolved] = secondary_fingerprint
                                secondary_fingerprint_computed = True
                            if (
                                secondary_fingerprint is not None
                                and candidate_fingerprint is not None
                                and secondary_fingerprint == candidate_fingerprint
                            ):
                                duplicate = True
                                break
                        if duplicate:
                            physical_seen.add(physical)
                            continue

                    runtime_name = _native_runtime_name(engine, category, relative)
                    entry = {
                        "id": _stable_id(engine, category, resolved),
                        "category": category,
                        "source": engine,
                        "sourceName": self._source_name(engine),
                        "primary": primary,
                        "group": "main" if primary else "secondary_unique",
                        "path": resolved,
                        "runtimeName": runtime_name,
                        "name": _display_name(relative),
                        "label": _display_name(relative),
                        "backendAvailable": False,
                        "nameConflict": False,
                        "size": size,
                    }
                    accepted.append(entry)
                    physical_seen.add(physical)
                    physical_keys[os.path.normcase(resolved)] = physical
                    content_candidates.setdefault(duplicate_key, []).append(entry)

        by_name: dict[str, list[dict[str, Any]]] = {}
        for entry in accepted:
            by_name.setdefault(_name_key(str(entry["runtimeName"])), []).append(entry)
        for conflicts in by_name.values():
            if len(conflicts) > 1:
                for entry in conflicts:
                    entry["nameConflict"] = True

        self._catalog_cache[category] = accepted
        self._physical_keys[category] = physical_keys
        return accepted

    def _matcher(self, category: str) -> _EntryMatcher:
        category = _canonical_category(category)
        matcher = self._matcher_cache.get(category)
        if matcher is None:
            entries = self._catalog(category)
            matcher = _EntryMatcher(
                entries, self.active_engine, self._physical_keys.get(category)
            )
            self._matcher_cache[category] = matcher
        return matcher

    def entries(self, category: str, *, backend_items: Sequence[Any] | None = None) -> list[dict[str, Any]]:
        """Return primary-all + secondary-unique catalog entries.

        When backend items are supplied, exact active-API matches are marked
        available and their ``runtimeName`` becomes the API's unmodified raw
        choice.  With no API list, availability means "belongs to active source".
        """
        category = _canonical_category(category)
        entries = [dict(entry) for entry in self._catalog(category)]
        if backend_items is None:
            for entry in entries:
                entry["backendAvailable"] = bool(
                    self.active_engine and entry["source"] == self.active_engine
                )
            return entries

        entries_by_id = {str(entry["id"]): entry for entry in entries}
        matcher = self._matcher(category)
        matched_ids: set[str] = set()
        for item in backend_items:
            base_match = matcher.best_match(item)
            if base_match is None:
                continue
            match_id = str(base_match["id"])
            if match_id in matched_ids:
                continue
            matched_ids.add(match_id)
            match = entries_by_id[match_id]
            match["backendAvailable"] = True
            runtime_name = _preferred_api_name(item)
            if runtime_name:
                match["runtimeName"] = runtime_name
        return entries

    def option_groups(self, category: str, raw_options: Sequence[Any]) -> list[dict[str, Any]]:
        """Group exact backend option strings without changing their values."""
        matcher = self._matcher(category)
        groups: dict[tuple[str, bool], dict[str, Any]] = {}
        first_seen: dict[tuple[str, bool], int] = {}
        seen_options: set[str] = set()
        seen_entry_ids: set[str] = set()

        for index, raw in enumerate(raw_options or []):
            option = str(raw or "")
            if not option or _is_placeholder(option):
                continue
            option_key = option.casefold()
            if option_key in seen_options:
                continue
            seen_options.add(option_key)

            match = matcher.best_match(raw)
            if match is not None:
                match_id = str(match["id"])
                if match_id in seen_entry_ids:
                    continue
                seen_entry_ids.add(match_id)
                source = str(match["source"])
                primary = bool(match["primary"])
                source_name = str(match["sourceName"])
            else:
                source = self.active_engine or self.primary_engine or "backend"
                primary = source == self.primary_engine
                source_name = self._source_name(source)
            key = (source, primary)
            if key not in groups:
                groups[key] = {
                    "label": source_name,
                    "source": source,
                    "primary": primary,
                    "options": [],
                }
                first_seen[key] = index
            # The option is intentionally not replaced by label/runtime metadata.
            groups[key]["options"].append(option)

        ordered_keys = sorted(groups, key=lambda key: (not key[1], first_seen[key]))
        return [groups[key] for key in ordered_keys]

    def merge_loras(self, api_loras: Sequence[Any] | None) -> list[dict[str, Any]]:
        """Merge backend LoRA metadata with disk source/category metadata."""
        api_loras = list(api_loras or [])
        disk_entries = [dict(entry) for entry in self._catalog("loras")]
        entries_by_id = {str(entry["id"]): entry for entry in disk_entries}
        matcher = self._matcher("loras")
        output: list[dict[str, Any]] = []
        used_ids: set[str] = set()
        seen_api_names: set[str] = set()

        for api_item in api_loras:
            runtime_name = _preferred_api_name(api_item)
            if not runtime_name:
                continue
            runtime_key = runtime_name.casefold()
            if runtime_key in seen_api_names:
                continue
            seen_api_names.add(runtime_key)

            base_match = matcher.best_match(api_item)
            if base_match is not None:
                match_id = str(base_match["id"])
                # One physical/catalog model may be exposed under several raw
                # aliases or paths by a backend.  Keep the first valid raw
                # choice, but never reintroduce a content-deduped secondary.
                if match_id in used_ids:
                    continue
                used_ids.add(match_id)
                item = dict(entries_by_id[match_id])
            else:
                source = self.active_engine or self.primary_engine or "backend"
                primary = source == self.primary_engine
                source_name = self._source_name(source)
                item = {
                    "id": f"api:{source}:{hashlib.sha256(runtime_key.encode('utf-8')).hexdigest()[:20]}",
                    "category": "loras",
                    "source": source,
                    "sourceName": source_name,
                    "primary": primary,
                    "group": "main" if primary else "secondary_unique",
                    "path": str(api_item.get("path") or "") if isinstance(api_item, Mapping) else "",
                    "nameConflict": False,
                    "size": 0,
                }

            alias = ""
            if isinstance(api_item, Mapping):
                alias = str(api_item.get("alias") or "").strip()
            item["runtimeName"] = runtime_name
            item["name"] = runtime_name
            item["label"] = alias or str(item.get("label") or _display_name(runtime_name))
            item["triggerWords"] = _trigger_words(api_item)
            item["backendAvailable"] = True
            output.append(item)

        for entry in disk_entries:
            if str(entry["id"]) in used_ids:
                continue
            item = dict(entry)
            item["triggerWords"] = []
            output.append(item)

        # Unmatched API items can introduce conflicts not present in the disk
        # catalog.  Mark every visible participant without merging either one.
        by_name: dict[str, list[dict[str, Any]]] = {}
        for item in output:
            by_name.setdefault(_name_key(str(item.get("runtimeName") or item.get("name") or "")), []).append(item)
        for conflicts in by_name.values():
            if len(conflicts) > 1:
                for item in conflicts:
                    item["nameConflict"] = True

        output.sort(key=lambda item: (
            not bool(item.get("primary")),
            str(item.get("sourceName") or "").casefold(),
            str(item.get("label") or item.get("name") or "").casefold(),
        ))
        return output


def get_model_inventory(
    snapshot: Mapping[str, Any] | None = None,
    *,
    active_engine: str | None = None,
) -> ModelInventory:
    """Create an inventory from an explicit or current runtime snapshot."""
    if snapshot is None:
        try:
            # Runtime import avoids a core.backend_runtime -> inventory cycle.
            from core.backend_runtime import get_backend_runtime_manager

            current = get_backend_runtime_manager().snapshot()
            snapshot = current if isinstance(current, Mapping) else {}
        except Exception:
            snapshot = {}
    return ModelInventory(snapshot, active_engine=active_engine)


__all__ = ["MODEL_CATEGORIES", "ModelInventory", "get_model_inventory"]
