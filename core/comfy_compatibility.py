"""Read-only Comfy capability recipes and explicitly saved local baselines.

Schema availability is not proof of a successful GPU generation. Upstream pins
are reference observations, never an instruction to downgrade an installation.
No installers, model imports, downloads or backend mutations live here.
"""
from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlsplit

import requests

from core.storage_paths import PROJECT_ROOT, config_file
from utils.atomic_json import atomic_write_json


LAKIS_REVISION = "19ec1be13414ea8c029782184121ee43b3662bea"
REFERENCE_SOURCE = f"https://github.com/Jeong-Luke/LAKIS/blob/{LAKIS_REVISION}/installer/Setup_LAKIS_Safe.cs#L145"
REFERENCES = [
    {"id": "comfy", "name": "ComfyUI", "version": "v0.21.1", "commit": "", "repoUrl": "https://github.com/Comfy-Org/ComfyUI"},
    {"id": "spectrum", "name": "Spectrum", "version": "", "commit": "b46a364aec3b161b889c9cc26cd976a49eb537ae", "repoUrl": "https://github.com/sorryhyun/ComfyUI-Spectrum-KSampler"},
    {"id": "ultimate", "name": "Ultimate SD Upscale", "version": "", "commit": "a5547db9e1d07d3318bb21e9e9c474f4c1e9c8df", "repoUrl": "https://github.com/ssitu/ComfyUI_UltimateSDUpscale"},
    {"id": "impact", "name": "Impact Pack", "version": "", "commit": "429d0159ad429e64d2b3916e6e7be9c22d025c3c", "repoUrl": "https://github.com/ltdrdata/ComfyUI-Impact-Pack"},
]
RECIPES = [
    # 기본 컴파일러가 실제로 만드는 노드만 적는다. 출력은 save_images 를 따른다 —
    # 메인 생성(save_images=True)은 코어 SaveImage, 후처리·채팅·손 재구성 등 나머지는 코어 PreviewImage.
    {"id": "forge-parity", "title": "기본 생성 · Forge 기능 연결", "scope": "앱 기본 Comfy 워크플로",
     "nodes": {"ForgeNeoLatentInput": {"vae": "VAE", "mode": "CHOICE", "width": "INT", "height": "INT"},
               "ForgeNeoKSamplerCNS": {"model": "MODEL", "steps": "INT", "cfg": "FLOAT", "latent_image": "LATENT"},
               "SaveImage": {"images": "IMAGE", "filename_prefix": "STRING"},
               "PreviewImage": {"images": "IMAGE"}},
     "models": ["diffusion", "text_encoder", "vae"], "repoUrl": "",
     "note": "번들 노드와 모델 목록을 확인합니다. Anima 버전별 TE/LoRA 호환성은 생성 전 컴파일 검증도 필요합니다."},
    {"id": "sam3", "title": "SAM3 영역 마스크 · 상세 보정", "scope": "앱 SAM3 생성/후처리",
     "nodes": {"ForgeNeoSAM3Mask": {"image": "IMAGE", "prompt": "STRING", "checkpoint": "STRING"},
               "ForgeNeoSAM3Detailer": {}, "ForgeNeoSAM3Refine": {}}, "models": [], "repoUrl": "",
     "note": "SAM3 모델 경로는 STRING 입력이라 API 목록만으로 파일·가중치 로드 성공을 검증할 수 없습니다. SAM3 설정과 실제 마스크 생성을 확인하세요."},
    {"id": "spectrum", "title": "Spectrum 가속", "scope": "실험 · 기본 생성 샘플러",
     "nodes": {"ForgeNeoKSamplerCNS": {"spectrum_enabled": "BOOLEAN", "spectrum_one_sampler_only": "BOOLEAN"},
               "DiTSpectrumPatch": {"model": "MODEL"}}, "models": [],
     "repoUrl": "https://github.com/sorryhyun/ComfyUI-Spectrum-KSampler",
     "note": "외부 노드가 필요합니다. 기준 커밋과 달라도 즉시 불호환은 아니며, A/B 속도·품질 검증 없이 가속을 보장하지 않습니다."},
    {"id": "relight", "title": "조명·그림자 보정", "scope": "실험 · 독립 번들 알고리즘",
     "nodes": {"AIStudioRelight": {"image": "IMAGE"}}, "models": [], "repoUrl": "",
     "note": "깊이·법선 맵은 선택 입력입니다. DepthAnything/DSINE 같은 외부 추정 노드는 사용자 워크플로에서 별도로 연결하며 자동 설치·추론하지 않습니다."},
]
MODEL_FIELDS = {
    "diffusion": [("UNETLoader", "unet_name"), ("CheckpointLoaderSimple", "ckpt_name")],
    "text_encoder": [("CLIPLoader", "clip_name"), ("DualCLIPLoader", "clip_name1")],
    "vae": [("VAELoader", "vae_name")],
}
MODEL_LABELS = {"diffusion": "생성 모델", "text_encoder": "텍스트 인코더", "vae": "VAE"}


def _digest(value) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def endpoint_identity(api_url: str) -> str:
    return _digest(api_url.rstrip("/"))[:24]


def _fields(node) -> dict:
    raw = node.get("input", {}) if isinstance(node, Mapping) else {}
    if not isinstance(raw, Mapping):
        return {}
    return {**(raw.get("required") if isinstance(raw.get("required"), dict) else {}),
            **(raw.get("optional") if isinstance(raw.get("optional"), dict) else {})}


def _choice_count(schema: Mapping, candidates: list) -> int | None:
    found, values = False, set()
    for node_id, field in candidates:
        spec = _fields(schema.get(node_id)).get(field)
        if isinstance(spec, (list, tuple)) and spec and isinstance(spec[0], list):
            found = True
            values.update(str(value) for value in spec[0] if isinstance(value, str) and value.strip())
    return len(values) if found else None


def check_recipes(schema: Mapping | None) -> list[dict]:
    """Compare declared types; an absent API response is unknown, not success."""
    results = []
    for recipe in RECIPES:
        checks = []
        for node_id, required in recipe["nodes"].items():
            if schema is None:
                checks.append({"label": node_id, "status": "unknown", "detail": "서버 응답 없음"})
                continue
            node = schema.get(node_id)
            if not isinstance(node, Mapping):
                checks.append({"label": node_id, "status": "missing", "detail": "노드 미등록 · 설치/재시작 확인"})
                continue
            if not isinstance(node.get("input"), Mapping):
                checks.append({"label": node_id, "status": "unknown", "detail": "입력 스키마 확인 불가"})
                continue
            fields = _fields(node)
            problems = []
            for key, expected in required.items():
                spec = fields.get(key)
                kind = spec[0] if isinstance(spec, (list, tuple)) and spec else None
                valid = isinstance(kind, list) and len(kind) > 0 if expected == "CHOICE" else kind == expected
                if not valid:
                    problems.append(f"{key}: {expected} 필요")
            checks.append({"label": node_id, "status": "mismatch" if problems else "available",
                           "detail": "; ".join(problems) if problems else "노드 및 주요 입력 타입 확인"})
        for kind in recipe["models"]:
            count = _choice_count(schema, MODEL_FIELDS[kind]) if schema is not None else None
            checkpoint_count = _choice_count(schema, [("CheckpointLoaderSimple", "ckpt_name")]) if schema else None
            if kind in {"text_encoder", "vae"} and not count and checkpoint_count:
                checks.append({"label": MODEL_LABELS[kind], "status": "unknown",
                               "detail": "체크포인트 내장 TE/VAE 사용 가능성 · 별도 모델이 필요한 Anima는 설정 확인"})
                continue
            checks.append({"label": MODEL_LABELS[kind], "status": "unknown" if count is None else "available" if count else "missing",
                           "detail": "모델 목록 API 확인 불가" if count is None else f"서버 목록 {count}개 · 파일 SHA/아키텍처 미검증"})
        if recipe["id"] == "sam3":
            checks.append({"label": "SAM3 가중치 로드", "status": "unknown", "detail": "파일 경로 입력 방식 · 실제 로드 테스트 필요"})
        status = "missing" if any(x["status"] in {"missing", "mismatch"} for x in checks) else "unknown" if any(x["status"] == "unknown" for x in checks) else "available"
        results.append({key: recipe[key] for key in ("id", "title", "scope", "note", "repoUrl")} | {"status": status, "checks": checks})
    return results


_BUNDLE_MAX_FILES = 256
# tokenizer.json 같은 런타임 자산도 지문에 들어가므로 상한은 넉넉히 둔다.
_BUNDLE_MAX_FILE_BYTES = 32 * 1024 * 1024


def bundle_fingerprint(root: Path | None = None) -> dict:
    """Pack digest for the compatibility screen — the installer's own algorithm.

    ``diskMatch`` compares this for the app source and the installed copy, so
    it must be exactly what ``install_bundled_node_pack`` decides a reinstall
    on (``.py``/``.json``/``.md``/``.txt``), not a ``.py``-only variant.
    """
    from core.comfy_node_pack import ComfyNodePackError, node_pack_fingerprint, pack_files

    root = Path(root) if root is not None else PROJECT_ROOT / "comfy_custom_nodes" / "ai_studio_forge_parity"
    try:
        fingerprint = node_pack_fingerprint(
            root, max_files=_BUNDLE_MAX_FILES, max_file_bytes=_BUNDLE_MAX_FILE_BYTES,
        )
        file_count = len(pack_files(root))
        source = (root / "__init__.py").read_text(encoding="utf-8")
        version = re.search(r'__version__\s*=\s*[\"\']([^\"\']+)', source)
        return {"version": version.group(1) if version else "", "fingerprint": fingerprint,
                "status": "source", "fileCount": file_count}
    except (ComfyNodePackError, OSError, ValueError):
        return {"version": "", "fingerprint": "", "status": "unknown"}


def compare_references(runtime: Mapping | None, server_version: str = "") -> list[dict]:
    refs = []
    for reference in REFERENCES:
        installed = None
        if reference["id"] == "comfy":
            current = str((runtime or {}).get("version") or server_version)
            matches = current.lstrip("v") == reference["version"].lstrip("v") if current else None
        else:
            wanted = reference["repoUrl"].lower().removesuffix(".git").rstrip("/")
            extensions = (runtime or {}).get("extensions", [])
            installed = next((item for item in extensions if isinstance(item, Mapping) and
                str(item.get("repoUrl", "")).lower().removesuffix(".git").rstrip("/") == wanted), None)
            current = str(installed.get("commit", "")) if installed else ""
            matches = current == reference["commit"] if current else None
        refs.append({**reference, "current": current,
                     "status": "unknown" if matches is None else "same" if matches else "different"})
    return refs


def _fetch_json(api_url: str, path: str, *, max_bytes: int) -> dict:
    chunks, length = [], 0
    with requests.get(api_url.rstrip("/") + path, timeout=(3, 8), stream=True, allow_redirects=False) as response:
        if response.status_code != 200:
            raise ValueError("backend status unavailable")
        for chunk in response.iter_content(64 * 1024):
            length += len(chunk)
            if length > max_bytes:
                raise ValueError("backend response limit")
            chunks.append(chunk)
    data = json.loads(b"".join(chunks))
    if not isinstance(data, dict):
        raise ValueError("invalid backend schema")
    return data


def inspect_compatibility(api_url: str, *, runtime_snapshot=None, fetch=None, baseline_path=None) -> dict:
    parsed = urlsplit(api_url)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise ValueError("ComfyUI API 주소를 먼저 설정하세요.")
    fetch = fetch or _fetch_json
    schema, server_version, warnings = None, "", []
    try:
        schema = fetch(api_url, "/object_info", max_bytes=32 * 1024 * 1024)
        if not schema or not isinstance(schema, Mapping):
            schema = None
            raise ValueError("empty backend schema")
    except (requests.RequestException, ValueError, OSError):
        warnings.append("ComfyUI 노드 정보를 읽지 못했습니다. 서버 연결과 API 주소를 확인하세요.")
    try:
        stats = fetch(api_url, "/system_stats", max_bytes=1024 * 1024)
        server_version = str(stats.get("system", {}).get("comfyui_version", ""))[:100]
    except (requests.RequestException, ValueError, OSError, AttributeError):
        warnings.append("서버 버전 정보는 확인되지 않았습니다.")
    if runtime_snapshot is None:
        from core.backend_runtime import get_backend_runtime_manager
        runtime_snapshot = get_backend_runtime_manager().snapshot()
    engine = runtime_snapshot.get("engines", {}).get("comfyui", {})
    # Never attribute this PC's installed files to an unrelated remote server.
    local = engine if engine.get("running") and str(engine.get("apiUrl", "")).rstrip("/") == api_url.rstrip("/") else None
    bundle = bundle_fingerprint()
    if local:
        installed_bundle = next((item for item in local.get("extensions", []) if item.get("id") == "ai_studio_forge_parity"), None)
        installed = bundle_fingerprint(Path(installed_bundle["path"])) if installed_bundle and installed_bundle.get("path") else {}
        bundle["installedFingerprint"] = installed.get("fingerprint", "")
        bundle["diskMatch"] = bool(bundle.get("fingerprint") and bundle.get("fingerprint") == installed.get("fingerprint")) if installed.get("fingerprint") else None
    relevant = sorted({node for recipe in RECIPES for node in recipe["nodes"]} | {node for specs in MODEL_FIELDS.values() for node, _ in specs})
    node_hashes = {node: _digest({"input": schema[node].get("input"), "output": schema[node].get("output")})
                   for node in relevant if schema and isinstance(schema.get(node), Mapping)}
    reference_results = compare_references(local, server_version)
    identity = endpoint_identity(api_url)
    snapshot = {"endpointId": identity, "serverVersion": server_version, "nodes": node_hashes,
                "bundleFingerprint": bundle.get("fingerprint", ""),
                "revisions": {item["id"]: item["current"] for item in reference_results}}
    try:
        baseline = _load_baselines(baseline_path).get(identity)
    except ValueError:
        baseline = None
        warnings.append("저장된 비교 기준 파일이 손상되었거나 형식이 다릅니다. 기존 파일은 보존하며 새 기준으로 덮어쓰지 않습니다.")
    drift = []
    if isinstance(baseline, Mapping):
        old = baseline.get("snapshot", {})
        for key in ("serverVersion", "bundleFingerprint", "revisions"):
            if old.get(key) != snapshot[key]:
                drift.append({"field": key, "detail": "기준과 달라졌습니다. 변경 또는 확인 불가 상태를 다시 검토하세요."})
        for node in sorted(set(old.get("nodes", {})) | set(node_hashes)):
            if old.get("nodes", {}).get(node) != node_hashes.get(node):
                drift.append({"field": node, "detail": "노드/입력 스키마/모델 목록 변경 또는 확인 불가"})
    return {"ok": True, "connected": schema is not None, "serverVersion": server_version,
            "localRevisionKnown": local is not None, "recipes": check_recipes(schema), "references": reference_results,
            "referenceSource": REFERENCE_SOURCE, "referenceLabel": "LAKIS v7.2.2 설치기 기준 · 이 앱에서 미검증",
            "bundled": bundle, "warnings": warnings, "snapshot": snapshot,
            "baseline": {"exists": isinstance(baseline, Mapping), "savedAt": baseline.get("savedAt", "") if isinstance(baseline, Mapping) else "", "drift": drift}}


def _load_baselines(path=None) -> dict:
    path = Path(path or config_file("comfy_compatibility_baselines.json"))
    if not path.exists():
        return {}
    try:
        if path.stat().st_size > 2 * 1024 * 1024:
            raise ValueError("baseline size")
        data = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(data, dict) or data.get("schemaVersion") != 1 or not isinstance(data.get("baselines"), dict):
            raise ValueError("baseline schema")
        for value in data["baselines"].values():
            if not isinstance(value, dict) or not isinstance(value.get("snapshot"), dict) or not isinstance(value["snapshot"].get("nodes"), dict):
                raise ValueError("baseline entry")
        return data["baselines"]
    except (OSError, ValueError) as exc:
        raise ValueError("저장된 호환 기준 파일을 읽을 수 없습니다. 기존 파일을 확인한 뒤 다시 저장하세요.") from exc


def save_baseline(report: Mapping, *, path=None) -> dict:
    if report.get("connected") is not True or not isinstance(report.get("snapshot"), Mapping):
        raise ValueError("서버에서 노드 정보를 읽은 뒤 기준으로 저장하세요.")
    snapshot = report["snapshot"]
    identity = snapshot.get("endpointId", "")
    if not re.fullmatch(r"[a-f0-9]{24}", identity):
        raise ValueError("호환 조합의 서버 식별자가 올바르지 않습니다.")
    path = path or config_file("comfy_compatibility_baselines.json")
    baselines = _load_baselines(path)
    saved = {"savedAt": datetime.now(timezone.utc).isoformat(), "snapshot": dict(snapshot)}
    baselines[identity] = saved
    if len(baselines) > 32:
        # Do not evict another user's baseline silently.
        raise ValueError("저장된 서버 기준이 32개입니다. 기존 기준 파일을 정리한 뒤 다시 저장하세요.")
    atomic_write_json(str(path), {"schemaVersion": 1, "baselines": baselines})
    return {"exists": True, "savedAt": saved["savedAt"], "drift": []}
