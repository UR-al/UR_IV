"""Owned, bounded H3 conditioning cache. Torch/Comfy are imported only at use.

This is an independent implementation, not a copy of GemmaStudio's nodes.
Only CPU tensors and plain containers are serialized; neither model objects nor
Comfy's NestedTensor instances are accepted by the weights-only cache format.
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import os
import pickle
from pathlib import Path
import re
import threading
import sys
import time
import uuid


CACHE_SCHEMA = 1
_KEY = re.compile(r"[0-9a-f]{64}")
# 재시작 후 첫 H3 작업이 수십 GB 모델을 다시 해시하지 않도록 모델·엔진 digest 를
# 캐시 루트에 남긴다. 미디어 digest 는 넣지 않는다(업로드 이름은 내용이 아니다).
DIGEST_SIDECAR = "model_digests.json"
_DIGEST_SIDECAR_SCHEMA = 1
_DIGEST_SIDECAR_LIMIT = 256
# put() 의 임시 파일 '{key}.{uuid4 hex}.tmp' / '{key}.{uuid4 hex}.json.tmp' 와
# 사이드카 교체 중의 '{DIGEST_SIDECAR}.{uuid4 hex}.part'.
# 강제 종료(taskkill /F)로 남은 것만 이 엄격한 패턴으로 골라 지운다.
_ORPHAN = re.compile(
    r"[0-9a-f]{64}\.[0-9a-f]{32}\.(?:json\.)?tmp"
    r"|" + re.escape(DIGEST_SIDECAR) + r"\.[0-9a-f]{32}\.part"
)
# 다른 프로세스가 같은 폴더에 쓰는 중일 수 있는 최근 임시 파일은 put 시작 정리에서 제외한다.
_ORPHAN_MIN_AGE_SECONDS = 600
_LOCK = threading.RLock()
_FINGERPRINTS = {}
_MODEL_IDENTITIES = {}


class ConditioningCacheCancelled(RuntimeError):
    pass


def _digest_file(path, cancelled=lambda: False):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            if cancelled():
                raise ConditioningCacheCancelled("H3 캐시 작업이 취소되었습니다")
            digest.update(chunk)
    return digest.hexdigest()


def _cpu_value(value):
    import torch
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().contiguous()
    if isinstance(value, dict) and all(isinstance(k, (str, int)) for k in value):
        return {key: _cpu_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(_cpu_value(item) for item in value)
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    raise ValueError(f"H3 cache does not support {type(value).__name__}")


def _conditioning(value):
    import torch
    if not isinstance(value, (list, tuple)) or not value:
        raise ValueError("H3 cache must contain non-empty CONDITIONING")
    for pair in value:
        if (not isinstance(pair, (list, tuple)) or len(pair) != 2
                or not isinstance(pair[0], torch.Tensor) or not isinstance(pair[1], dict)):
            raise ValueError("H3 cache CONDITIONING schema is invalid")
    return _cpu_value(value)


class ConditioningCache:
    """CPU disk store with SHA integrity, count/byte limits and exact-scope clear."""

    def __init__(self, root, *, max_bytes=8 * 1024 ** 3, max_entries=32):
        self.root = Path(root).resolve()
        self.max_bytes, self.max_entries = int(max_bytes), int(max_entries)
        if not 1024 ** 2 <= self.max_bytes <= 128 * 1024 ** 3 or not 1 <= self.max_entries <= 1024:
            raise ValueError("Invalid H3 conditioning cache limits")
        if self.root == Path(self.root.anchor):
            raise ValueError("A filesystem root cannot be a conditioning cache")

    def _path(self, key, suffix=".pt"):
        if not isinstance(key, str) or not _KEY.fullmatch(key):
            raise ValueError("Invalid H3 cache key")
        path = self.root / (key + suffix)
        if path.is_symlink() or path.resolve().parent != self.root:
            raise ValueError("H3 cache path is outside its owned directory")
        return path

    def _files(self):
        if not self.root.is_dir():
            return []
        return [p for p in self.root.iterdir() if p.is_file() and not p.is_symlink()
                and p.suffix == ".pt" and _KEY.fullmatch(p.stem)]

    def _remove(self, key):
        for suffix in (".pt", ".json"):
            self._path(key, suffix).unlink(missing_ok=True)

    def _orphans(self, *, min_age_seconds=0.0):
        """Temporary files an interrupted put() left behind (never live entries)."""
        if not self.root.is_dir():
            return []
        cutoff = time.time() - float(min_age_seconds)
        found = []
        for path in self.root.iterdir():
            if not _ORPHAN.fullmatch(path.name) or path.is_symlink() or not path.is_file():
                continue
            try:
                if min_age_seconds and path.stat().st_mtime > cutoff:
                    continue
            except OSError:
                continue
            found.append(path)
        return found

    def _remove_orphans(self, *, min_age_seconds=0.0):
        removed = 0
        for path in self._orphans(min_age_seconds=min_age_seconds):
            try:
                removed += path.stat().st_size
                path.unlink(missing_ok=True)
            except OSError:
                pass
        return removed

    def _prune(self, keep):
        files = self._files()
        total, count = sum(p.stat().st_size for p in files), len(files)
        for stale in sorted((p for p in files if p.stem != keep), key=lambda p: p.stat().st_mtime_ns):
            if total <= self.max_bytes and count <= self.max_entries:
                break
            total -= stale.stat().st_size
            count -= 1
            self._remove(stale.stem)

    def stats(self):
        with _LOCK:
            files = self._files()
            return {"entries": len(files), "bytes": sum(p.stat().st_size for p in files),
                    "maxBytes": self.max_bytes, "maxEntries": self.max_entries,
                    "scope": "comfy_server"}

    def clear(self):
        with _LOCK:
            before = self.stats()
            for path in self._files():
                self._remove(path.stem)
            # _LOCK 안에서는 이 프로세스의 put 이 진행 중일 수 없으므로 임시 파일은
            # 전부 강제 종료가 남긴 고아다.
            orphan_bytes = self._remove_orphans()
            return {**self.stats(), "removedEntries": before["entries"],
                    "removedBytes": before["bytes"] + orphan_bytes}

    def get(self, key, *, cancelled=lambda: False):
        import torch
        with _LOCK:
            path, manifest = self._path(key), self._path(key, ".json")
            if not path.is_file() or not manifest.is_file():
                return None
            try:
                info = json.loads(manifest.read_text(encoding="utf-8"))
                if (info.get("schema") != CACHE_SCHEMA or info.get("key") != key
                        or info.get("bytes") != path.stat().st_size
                        or path.stat().st_size > self.max_bytes
                        or info.get("sha256") != _digest_file(path, cancelled)):
                    raise ValueError("H3 cache integrity mismatch")
                envelope = torch.load(path, map_location="cpu", weights_only=True)
                if envelope.get("schema") != CACHE_SCHEMA or envelope.get("key") != key:
                    raise ValueError("H3 cache identity mismatch")
                value = _conditioning(envelope["conditioning"])
                if cancelled():
                    raise ConditioningCacheCancelled("H3 캐시 작업이 취소되었습니다")
                os.utime(path, None)
                self._prune(key)
                return value
            except ConditioningCacheCancelled:
                raise
            except (OSError, ValueError, TypeError, KeyError, RuntimeError, EOFError,
                    AttributeError, pickle.UnpicklingError):
                self._remove(key)
                return None

    def put(self, key, conditioning, *, cancelled=lambda: False):
        import torch
        with _LOCK:
            destination = self._path(key)
            value = _conditioning(conditioning)
            if cancelled():
                raise ConditioningCacheCancelled("H3 캐시 작업이 취소되었습니다")
            self.root.mkdir(parents=True, exist_ok=True)
            # 자기 임시 파일을 만들기 전에, 오래된 고아만 치운다(다른 프로세스가
            # 막 쓰는 중일 수 있는 최근 파일은 남긴다).
            self._remove_orphans(min_age_seconds=_ORPHAN_MIN_AGE_SECONDS)
            nonce = uuid.uuid4().hex
            temporary = self.root / f"{key}.{nonce}.tmp"
            temporary_manifest = self.root / f"{key}.{nonce}.json.tmp"
            try:
                torch.save({"schema": CACHE_SCHEMA, "key": key, "conditioning": value}, temporary)
                size = temporary.stat().st_size
                if size > self.max_bytes:
                    raise ValueError("H3 conditioning이 캐시 용량 한도를 초과했습니다. 한도를 늘리거나 캐시를 끄세요")
                digest = _digest_file(temporary, cancelled)
                temporary_manifest.write_text(json.dumps({"schema": CACHE_SCHEMA, "key": key,
                    "sha256": digest, "bytes": size}), encoding="utf-8")
                if cancelled():
                    raise ConditioningCacheCancelled("H3 캐시 작업이 취소되었습니다")
                others = sorted((p for p in self._files() if p.stem != key), key=lambda p: p.stat().st_mtime_ns)
                total = sum(p.stat().st_size for p in others) + size
                while others and (len(others) + 1 > self.max_entries or total > self.max_bytes):
                    stale = others.pop(0)
                    total -= stale.stat().st_size
                    self._remove(stale.stem)
                os.replace(temporary, destination)
                os.replace(temporary_manifest, self._path(key, ".json"))
                return {"key": key, "bytes": size, "ready": True, "hit": False}
            finally:
                temporary.unlink(missing_ok=True)
                temporary_manifest.unlink(missing_ok=True)


def _read_digest_sidecar(sidecar):
    """Persisted model digests; any unreadable/foreign content is ignored."""
    try:
        data = json.loads(Path(sidecar).read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError):
        return {}
    if not isinstance(data, dict) or data.get("schema") != _DIGEST_SIDECAR_SCHEMA:
        return {}
    entries = data.get("entries")
    return entries if isinstance(entries, dict) else {}


def _sidecar_entry_matches(entry, stamp):
    return (isinstance(entry, dict)
            and [entry.get("size"), entry.get("mtime_ns"), entry.get("ctime_ns"),
                 entry.get("ino")] == list(stamp)
            and isinstance(entry.get("sha256"), str) and _KEY.fullmatch(entry["sha256"]))


def _write_digest_sidecar(sidecar, resolved, stamp, digest):
    """Best-effort atomic update; a failure only costs a re-hash next restart."""
    sidecar = Path(sidecar)
    with _LOCK:
        try:
            entries = _read_digest_sidecar(sidecar)
            entries.pop(resolved, None)
            entries[resolved] = {"size": stamp[0], "mtime_ns": stamp[1],
                                 "ctime_ns": stamp[2], "ino": stamp[3], "sha256": digest}
            while len(entries) > _DIGEST_SIDECAR_LIMIT:
                entries.pop(next(iter(entries)))
            sidecar.parent.mkdir(parents=True, exist_ok=True)
            temporary = sidecar.with_name(f"{sidecar.name}.{uuid.uuid4().hex}.part")
            try:
                temporary.write_text(json.dumps(
                    {"schema": _DIGEST_SIDECAR_SCHEMA, "entries": entries},
                    ensure_ascii=False), encoding="utf-8")
                os.replace(temporary, sidecar)
            finally:
                temporary.unlink(missing_ok=True)
        except OSError:
            pass


def content_identity(path, *, memoize=False, cancelled=lambda: False, sidecar=None):
    """Hash actual server-side bytes, rejecting files modified during hashing.

    Large model digests are reused only while path/size/mtime/ctime/inode match,
    in memory and — with ``sidecar`` — across ComfyUI restarts.  Media are
    always hashed; upload filenames are not their content identity.
    """
    path = Path(path).resolve()
    stat = path.stat()
    stamp = (stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns, stat.st_ino)
    memo_key = (str(path), stamp)
    if cancelled():
        raise ConditioningCacheCancelled("H3 캐시 작업이 취소되었습니다")
    digest = _FINGERPRINTS.get(memo_key) if memoize else None
    if digest is None and memoize and sidecar is not None:
        entry = _read_digest_sidecar(sidecar).get(str(path))
        if _sidecar_entry_matches(entry, stamp):
            digest = entry["sha256"]
            if len(_FINGERPRINTS) >= 256:
                _FINGERPRINTS.clear()
            _FINGERPRINTS[memo_key] = digest
    if digest is None:
        digest = _digest_file(path, cancelled)
        after = path.stat()
        if stamp != (after.st_size, after.st_mtime_ns, after.st_ctime_ns, after.st_ino):
            raise RuntimeError("H3 입력/모델 파일이 변경되었습니다. 변경이 끝난 뒤 다시 생성하세요")
        if memoize:
            if len(_FINGERPRINTS) >= 256:
                _FINGERPRINTS.clear()
            _FINGERPRINTS[memo_key] = digest
            if sidecar is not None:
                _write_digest_sidecar(sidecar, str(path), stamp, digest)
    return {"sha256": digest, "bytes": stat.st_size}


def conditioning_identity(descriptor, resolve_model, resolve_input, *, engine_files=(),
                          cancelled=lambda: False, sidecar=None):
    """Resolve graph filenames at the Comfy host, including remote installations."""
    if not isinstance(descriptor, str) or len(descriptor) > 1024 * 1024:
        raise ValueError("Invalid H3 cache descriptor")
    value = json.loads(descriptor)
    if value.get("schema") != CACHE_SCHEMA or not isinstance(value.get("conditioning"), dict):
        raise ValueError("Unsupported H3 cache descriptor schema")
    models = {"UNETLoader": ("diffusion_models", "unet_name"),
              "CLIPLoader": ("text_encoders", "clip_name"), "VAELoader": ("vae", "vae_name")}
    for node in [*value["conditioning"].values(), *value.get("models", [])]:
        kind, inputs = node["class_type"], node["inputs"]
        if kind in models:
            category, field = models[kind]
            inputs[field] = content_identity(resolve_model(category, inputs[field]),
                                              memoize=True, cancelled=cancelled,
                                              sidecar=sidecar)
        elif kind in {"LoadImage", "GemmaVideoReferencePreprocessor"}:
            field = "image" if kind == "LoadImage" else "file"
            inputs[field] = content_identity(resolve_input(inputs[field]), cancelled=cancelled)
    value["implementation"] = [content_identity(path, memoize=True, cancelled=cancelled,
                                                sidecar=sidecar)
                                for path in engine_files]
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     ensure_ascii=False).encode("utf-8")).hexdigest()


def _cancelled():
    manager = sys.modules.get("comfy.model_management")
    return bool(manager and manager.processing_interrupted())


def _loaded_diffusion_models(manager):
    """Loaded entries whose patcher wraps a Comfy ``BaseModel`` (UNET/DiT)."""
    base_model = getattr(sys.modules.get("comfy.model_base"), "BaseModel", None)
    if not isinstance(base_model, type):
        return None
    loaded = getattr(manager, "current_loaded_models", None)
    if not isinstance(loaded, list):
        return None
    kept = []
    for entry in list(loaded):
        patcher = getattr(entry, "model", None)
        if isinstance(getattr(patcher, "model", None), base_model):
            kept.append(entry)
    return kept


def _unload_encoder_models(*, keep_diffusion=False):
    """Synchronous execution-worker barrier, not the asynchronous /free flag.

    A cache miss just ran the text encoder, so everything is unloaded before
    the sample stage.  A cache hit ran no encoder: only non-diffusion models
    (TE/CLIP/VAE) are released and a resident diffusion model — usually the
    previous H3 sample stage's UNET — stays loaded instead of a 2-5 s reload.
    Returns whether a diffusion model was deliberately kept.
    """
    manager = sys.modules.get("comfy.model_management")
    if manager is None:
        raise RuntimeError("ComfyUI model management is unavailable for H3 cache unloading")
    if _cancelled():
        raise ConditioningCacheCancelled("H3 캐시 작업이 취소되었습니다")
    kept = _loaded_diffusion_models(manager) if keep_diffusion else None
    devices = getattr(manager, "get_all_torch_devices", None)
    free_memory = getattr(manager, "free_memory", None)
    if kept is not None and callable(devices) and callable(free_memory):
        for device in devices():
            free_memory(1e30, device, keep_loaded=kept)
    else:
        kept = None
        manager.unload_all_models()
    manager.soft_empty_cache()
    if _cancelled():
        raise ConditioningCacheCancelled("H3 캐시 작업이 취소되었습니다")
    return bool(kept)


def _cache_root(folder_paths):
    return Path(folder_paths.get_output_directory()) / "aistudio_cache" / "h3_conditioning"


def _runtime(descriptor, max_bytes, max_entries):
    import folder_paths

    root = _cache_root(folder_paths)
    sidecar = root / DIGEST_SIDECAR

    def relative(name):
        name = str(name).replace("\\", "/")
        if not name or name.startswith("/") or ":" in name or ".." in name.split("/"):
            raise ValueError("H3 cache dependency must be a safe relative filename")
        return name

    def model(category, name):
        path = folder_paths.get_full_path(category, relative(name))
        if not path:
            raise FileNotFoundError(f"H3 cache model identity is unavailable: {category}/{name}")
        # Native Comfy loaders cache model objects by filename, independently
        # of this disk cache. Never pair a new on-disk identity with an already
        # loaded, old model after an in-place replacement in this process.
        identity = content_identity(path, memoize=True, cancelled=_cancelled, sidecar=sidecar)
        resolved = str(Path(path).resolve())
        with _LOCK:
            previous = _MODEL_IDENTITIES.get(resolved)
            if previous is not None and previous != identity:
                raise RuntimeError("H3 모델 파일이 실행 중 변경되었습니다. ComfyUI 서버를 재시작한 뒤 다시 생성하세요")
            if previous is None and len(_MODEL_IDENTITIES) >= 256:
                raise RuntimeError("H3 모델 확인 기록 한도에 도달했습니다. ComfyUI 서버를 재시작하세요")
            _MODEL_IDENTITIES[resolved] = identity
        return path

    def media(name):
        root = Path(folder_paths.get_input_directory()).resolve()
        path = (root / relative(name)).resolve()
        if not path.is_relative_to(root):
            raise ValueError("H3 cache input is outside the Comfy input directory")
        return path

    base = Path(getattr(folder_paths, "base_path", ""))
    engine_files = [Path(__file__)]
    for name in ("comfyui_version.py", "comfy_extras/nodes_minimax_h3.py",
                 "comfy/text_encoders/minimax.py", "comfy/ldm/minimax/model.py"):
        candidate = base / name
        if candidate.is_file():
            engine_files.append(candidate)
    nodes = sys.modules.get("nodes")
    for node_type in sorted({node["class_type"] for node in json.loads(descriptor)["conditioning"].values()}):
        cls = getattr(nodes, "NODE_CLASS_MAPPINGS", {}).get(node_type)
        module = sys.modules.get(getattr(cls, "__module__", ""))
        source = getattr(module, "__file__", "")
        if source and Path(source).is_file() and Path(source) not in engine_files:
            engine_files.append(Path(source))
    key = conditioning_identity(descriptor, model, media, engine_files=engine_files,
                                cancelled=_cancelled, sidecar=sidecar)
    return ConditioningCache(root, max_bytes=max_bytes, max_entries=max_entries), key


class ForgeNeoH3ConditioningCachePrepare:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"descriptor": ("STRING", {"default": ""}),
            "max_bytes": ("INT", {"default": 8 * 1024 ** 3, "min": 1024 ** 2, "max": 128 * 1024 ** 3}),
            "max_entries": ("INT", {"default": 32, "min": 1, "max": 1024}),
            "conditioning": ("CONDITIONING", {"lazy": True})}}

    RETURN_TYPES = ()
    OUTPUT_NODE = True
    FUNCTION = "prepare"
    CATEGORY = "AI Studio/Creator"

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return float("nan")  # Validate disk integrity/model identity on every request.

    def check_lazy_status(self, descriptor, max_bytes=8 * 1024 ** 3, max_entries=32, conditioning=None):
        store, key = _runtime(descriptor, max_bytes, max_entries)
        pending = getattr(self, "_pending_identity", None)
        if pending is not None and pending != key:
            self._pending_identity = None
            self._cached = None
            raise RuntimeError("H3 모델/입력이 인코딩 중 변경되었습니다. 다시 생성하세요")
        self._identity = key
        self._cached = store.get(key, cancelled=_cancelled)
        if self._cached is None and conditioning is None:
            self._pending_identity = key
            return ["conditioning"]
        return []

    def prepare(self, descriptor, max_bytes=8 * 1024 ** 3, max_entries=32, conditioning=None):
        store, key = _runtime(descriptor, max_bytes, max_entries)
        expected = getattr(self, "_identity", None) or key
        self._pending_identity = None
        self._identity = None
        if expected != key:
            self._cached = None
            raise RuntimeError("H3 모델/입력이 인코딩 중 변경되었습니다. 다시 생성하세요")
        cached = getattr(self, "_cached", None)
        self._cached = None
        if cached is None:
            cached = store.get(key, cancelled=_cancelled)
        if cached is not None:
            receipt = {"ready": True, "hit": True, "key": key}
        elif conditioning is not None:
            receipt = store.put(key, conditioning, cancelled=_cancelled)
        else:
            raise RuntimeError("H3 conditioning cache miss: encoder input is required")
        # The encoder barrier runs on hit and miss alike; a hit keeps the
        # resident diffusion model that the following sample stage reuses.
        receipt["diffusion_model_kept"] = _unload_encoder_models(
            keep_diffusion=bool(receipt.get("hit")),
        )
        receipt["models_unloaded"] = True
        return {"ui": {"h3_conditioning_cache": [{**receipt, **store.stats()}]}, "result": ()}


class ForgeNeoH3ConditioningCacheLoad:
    @classmethod
    def INPUT_TYPES(cls):
        inputs = dict(ForgeNeoH3ConditioningCachePrepare.INPUT_TYPES()["required"])
        inputs.pop("conditioning")
        return {"required": inputs, "optional": {"expected_key": ("STRING", {"default": ""})}}

    RETURN_TYPES = ("CONDITIONING",)
    FUNCTION = "load"
    CATEGORY = "AI Studio/Creator"
    IS_CHANGED = ForgeNeoH3ConditioningCachePrepare.IS_CHANGED

    def load(self, descriptor, max_bytes=8 * 1024 ** 3, max_entries=32, expected_key=""):
        store, key = _runtime(descriptor, max_bytes, max_entries)
        if expected_key and key != expected_key:
            raise RuntimeError("H3 모델/입력이 캐시 준비 이후 변경되었습니다. 다시 생성하세요")
        value = store.get(key, cancelled=_cancelled)
        if value is None:
            raise RuntimeError("H3 conditioning cache is missing or corrupt. Run the encoding stage again.")
        return (value,)


NODE_CLASS_MAPPINGS = {cls.__name__: cls for cls in (
    ForgeNeoH3ConditioningCachePrepare, ForgeNeoH3ConditioningCacheLoad)}


def _register_routes():
    # Importing this file in the desktop app must not import a Comfy host.
    server = sys.modules.get("server")
    instance = getattr(getattr(server, "PromptServer", None), "instance", None)
    if instance is None or getattr(instance, "_aistudio_h3_cache_routes", False):
        return
    from aiohttp import web
    import folder_paths

    def store():
        return ConditioningCache(_cache_root(folder_paths))

    # The prompt worker holds _LOCK through torch.save/SHA-256/torch.load.
    # Waiting for it on the aiohttp loop thread would stall every request, so
    # the locked work runs on a worker thread and the loop only awaits it.
    @instance.routes.get("/aistudio/h3-cache/status")
    async def status(_request):
        return web.json_response(await asyncio.to_thread(lambda: store().stats()))

    def clear_when_idle():
        # Queue can change after this check; the disk lock still prevents a
        # partial write/delete overlap, and sample loads fail closed if evicted.
        with _LOCK:
            current, pending = instance.prompt_queue.get_current_queue()
            if current or pending:
                return None
            return store().clear()

    @instance.routes.post("/aistudio/h3-cache/clear")
    async def clear(_request):
        result = await asyncio.to_thread(clear_when_idle)
        if result is None:
            return web.json_response({"error": "ComfyUI is busy"}, status=409)
        return web.json_response(result)

    instance._aistudio_h3_cache_routes = True


_register_routes()
