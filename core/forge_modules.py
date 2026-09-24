"""Forge Neo 모델 디렉터리 설정과 로컬 파일 검색.

Settings에서 체크포인트 / LoRA / VAE / Text Encoder 폴더를 각각 지정한다.
설정은 사용자 전용 ``config/forge_model_paths.json``에 저장되며 Git에는
포함되지 않는다.

우선순위는 개별 환경변수 → 저장된 Settings 값 → 감지된 Forge models 루트 →
앱 공용 ``user_data/models`` fallback이다. 기존 ``FORGE_MODELS_ROOT`` 및
``config/forge_models_root.txt``도 계속 지원한다. 외부 Forge가 전혀 없어도
fallback의 표준 하위 폴더를 만들어 Forge와 ComfyUI가 같은 모델 파일을 공유한다.
"""
from __future__ import annotations

import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from core.storage_paths import config_file

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_FORGE_MODELS_ROOT = r"C:\sd-webui-forge-neo\models"
FORGE_ROOT_CANDIDATES = (
    Path(DEFAULT_FORGE_MODELS_ROOT),
    Path(r"C:\sd-webui-forge-classic\models"),
)

FORGE_PATHS_FILE = config_file(
    "forge_model_paths.json",
    legacy_paths="user_data/forge_model_paths.json",
)
LEGACY_ROOT_FILE = PROJECT_ROOT / "config" / "forge_models_root.txt"

FORGE_PATH_KEYS = (
    "checkpoint_dir",
    "lora_dir",
    "vae_dir",
    "text_encoder_dir",
)

_DEFAULT_SUBDIRS = {
    "checkpoint_dir": "Stable-diffusion",
    "lora_dir": "Lora",
    "vae_dir": "VAE",
    "text_encoder_dir": "text_encoder",
}

APP_MODEL_SUBDIRS = {
    "checkpoints": "Stable-diffusion",
    "diffusion_models": "diffusion_models",
    "loras": "Lora",
    "vae": "VAE",
    "text_encoders": "text_encoder",
    "upscale_models": "upscale_models",
}

_ENV_KEYS = {
    "checkpoint_dir": "FORGE_CHECKPOINT_DIR",
    "lora_dir": "FORGE_LORA_DIR",
    "vae_dir": "FORGE_VAE_DIR",
    "text_encoder_dir": "FORGE_TEXT_ENCODER_DIR",
}

_CHECKPOINT_EXTS = {".ckpt", ".safetensors", ".gguf"}
_LORA_EXTS = {".pt", ".ckpt", ".safetensors"}
_MODULE_EXTS = {".safetensors", ".sft", ".gguf", ".pt", ".bin", ".ckpt", ".pth"}
_SAM3_EXTS = {".pt", ".safetensors"}
SAM3_SUBDIR = "sam3"


class ForgePathError(ValueError):
    """저장하려는 Forge 경로가 유효하지 않을 때 발생."""

    def __init__(self, errors: Mapping[str, str]):
        self.errors = dict(errors)
        super().__init__("; ".join(f"{key}: {msg}" for key, msg in self.errors.items()))


def _normalise_directory(value: str | os.PathLike[str]) -> Path:
    expanded = os.path.expandvars(os.path.expanduser(str(value).strip()))
    return Path(os.path.abspath(os.path.normpath(expanded)))


def get_app_models_root() -> Path:
    """Return the stable model-library fallback owned by this application."""
    return (PROJECT_ROOT / "user_data" / "models").resolve()


def get_app_model_paths(
    root: str | os.PathLike[str] | None = None,
) -> dict[str, Path]:
    """Return canonical shared folders in backend-runtime category names."""
    base = _normalise_directory(root) if root is not None else get_app_models_root()
    return {
        category: base / subdir
        for category, subdir in APP_MODEL_SUBDIRS.items()
    }


def ensure_app_model_layout(
    root: str | os.PathLike[str] | None = None,
) -> dict[str, Path]:
    """Idempotently create the empty shared library used as the final fallback."""
    paths = get_app_model_paths(root)
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)
    return paths


def get_forge_root(environ: Mapping[str, str] | None = None) -> Path:
    """Forge ``models`` root, falling back to the app-owned shared library."""
    env = os.environ if environ is None else environ
    configured = str(env.get("FORGE_MODELS_ROOT", "") or "").strip()
    if configured:
        return _normalise_directory(configured)

    try:
        if LEGACY_ROOT_FILE.is_file():
            configured = LEGACY_ROOT_FILE.read_text(encoding="utf-8").strip()
            if configured:
                return _normalise_directory(configured)
    except OSError:
        pass
    for candidate in FORGE_ROOT_CANDIDATES:
        if candidate.is_dir():
            return candidate.resolve()
    fallback = get_app_models_root()
    ensure_app_model_layout(fallback)
    return fallback


def get_default_forge_paths(
    *,
    environ: Mapping[str, str] | None = None,
) -> dict[str, Path]:
    root = get_forge_root(environ)
    return {key: root / subdir for key, subdir in _DEFAULT_SUBDIRS.items()}


def _read_path_config(config_path: Path | None = None) -> dict[str, str]:
    path = config_path or FORGE_PATHS_FILE
    from utils.atomic_json import load_json_safe

    data = load_json_safe(str(path), {})
    if not isinstance(data, dict):
        return {}
    raw = data.get("paths", data)
    if not isinstance(raw, dict):
        return {}
    return {
        key: str(raw.get(key, "") or "").strip()
        for key in FORGE_PATH_KEYS
        if str(raw.get(key, "") or "").strip()
    }


def get_forge_paths(
    *,
    config_path: Path | None = None,
    environ: Mapping[str, str] | None = None,
) -> dict[str, Path]:
    """4개 Forge 경로의 실제 적용값을 반환한다."""
    env = os.environ if environ is None else environ
    defaults = get_default_forge_paths(environ=env)
    configured = _read_path_config(config_path)
    result: dict[str, Path] = {}
    for key in FORGE_PATH_KEYS:
        env_value = str(env.get(_ENV_KEYS[key], "") or "").strip()
        value = env_value or configured.get(key, "")
        result[key] = _normalise_directory(value) if value else defaults[key]
    return result


def validate_forge_paths(payload: Mapping[str, Any]) -> dict[str, str]:
    """UI payload를 정규화한다. 빈 값은 해당 항목의 기본 경로를 뜻한다."""
    if not isinstance(payload, Mapping):
        raise ForgePathError({"payload": "경로 설정이 객체가 아닙니다"})

    normalised: dict[str, str] = {}
    errors: dict[str, str] = {}
    for key in FORGE_PATH_KEYS:
        raw = str(payload.get(key, "") or "").strip()
        if not raw:
            continue
        expanded = os.path.expandvars(os.path.expanduser(raw))
        if not Path(expanded).is_absolute():
            errors[key] = "절대 경로 또는 UNC 경로를 입력하세요"
            continue
        path = _normalise_directory(raw)
        if not path.exists():
            errors[key] = "폴더가 존재하지 않습니다"
        elif not path.is_dir():
            errors[key] = "폴더가 아닌 파일입니다"
        else:
            normalised[key] = str(path)
    if errors:
        raise ForgePathError(errors)
    return normalised


def save_forge_paths(
    payload: Mapping[str, Any],
    *,
    config_path: Path | None = None,
    environ: Mapping[str, str] | None = None,
) -> dict[str, Path]:
    """검증된 경로를 원자적으로 저장하고 실제 적용값을 반환한다."""
    paths = validate_forge_paths(payload)
    path = config_path or FORGE_PATHS_FILE
    env = os.environ if environ is None else environ
    # 환경변수로 잠긴 항목은 UI에 effective 값만 보인다. 이를 그대로 저장하면
    # 환경변수를 지운 뒤에도 일시 override가 설정 파일에 굳어 버리므로 기존값을 보존한다.
    existing = _read_path_config(path)
    for key, env_key in _ENV_KEYS.items():
        if not str(env.get(env_key, "") or "").strip():
            continue
        if key in existing:
            paths[key] = existing[key]
        else:
            paths.pop(key, None)
    path.parent.mkdir(parents=True, exist_ok=True)
    from utils.atomic_json import atomic_write_json

    atomic_write_json(
        str(path),
        {"schema_version": 1, "paths": paths},
        indent=2,
    )
    return get_forge_paths(config_path=path, environ=env)


def reset_forge_paths(*, config_path: Path | None = None) -> dict[str, Path]:
    """저장된 사용자 override를 지우고 기본/환경변수 경로로 복귀한다."""
    path = config_path or FORGE_PATHS_FILE
    try:
        path.unlink(missing_ok=True)
    except OSError as exc:
        raise ForgePathError({"config": f"설정 파일 삭제 실패: {exc}"}) from exc
    return get_forge_paths(config_path=path)


def _list_files(root: Path, extensions: set[str] | None = None) -> list[str]:
    """하위 폴더까지 검색해 루트 기준 상대 파일명을 정렬해 반환한다."""
    if not root.is_dir():
        return []
    allowed = extensions or _MODULE_EXTS
    result: list[str] = []
    try:
        for folder, dirnames, filenames in os.walk(root, followlinks=False):
            dirnames.sort(key=str.lower)
            for filename in sorted(filenames, key=str.lower):
                full_path = Path(folder) / filename
                if full_path.suffix.lower() not in allowed:
                    continue
                result.append(full_path.relative_to(root).as_posix())
    except OSError:
        return []
    return result


def _checkpoint_files(root: Path) -> list[str]:
    """Checkpoint 폴더의 모델 파일 — Forge 처럼 ``*.vae.*`` 사이드카는 제외한다."""
    return [
        name for name in _list_files(root, _CHECKPOINT_EXTS)
        if ".vae." not in Path(name).name.casefold()
    ]


def _lora_files(root: Path) -> list[str]:
    return _list_files(root, _LORA_EXTS)


def _module_basenames(root: Path) -> list[str]:
    """Forge module_list와 같은 파일명 key로 중복 없이 반환한다."""
    result: list[str] = []
    seen: set[str] = set()
    for relative in _list_files(root, _MODULE_EXTS):
        name = Path(relative).name
        key = name.casefold()
        if key in seen:
            continue
        seen.add(key)
        result.append(name)
    return result


def list_vae_files() -> list[str]:
    return _module_basenames(get_forge_paths()["vae_dir"])


def list_te_files() -> list[str]:
    return _module_basenames(get_forge_paths()["text_encoder_dir"])


def get_forge_path_state() -> dict[str, Any]:
    """Settings UI용 경로, 유효성, 파일 개수 스냅샷."""
    paths = get_forge_paths()
    defaults = get_default_forge_paths()
    env = os.environ
    state: dict[str, Any] = {
        "paths": {key: str(paths[key]) for key in FORGE_PATH_KEYS},
        "defaults": {key: str(defaults[key]) for key in FORGE_PATH_KEYS},
        "environmentLocked": {
            key: bool(str(env.get(_ENV_KEYS[key], "") or "").strip())
            for key in FORGE_PATH_KEYS
        },
        "entries": {},
    }
    scanners = {
        "checkpoint_dir": _checkpoint_files,
        "lora_dir": _lora_files,
        "vae_dir": _module_basenames,
        "text_encoder_dir": _module_basenames,
    }
    for key, path in paths.items():
        state["entries"][key] = {
            "exists": path.is_dir(),
            "count": len(scanners[key](path)),
        }
    return state


def list_sam3_checkpoints() -> list[str]:
    """SAM3 체크포인트를 models/sam3 및 models 루트에서 검색."""
    root = get_forge_root()
    seen: set[str] = set()
    result: list[str] = []

    def _consider(path: Path) -> None:
        name = path.name
        if name.lower().startswith("sam3") and path.suffix.lower() in _SAM3_EXTS:
            if name not in seen:
                seen.add(name)
                result.append(name)

    for candidate_root in (root / SAM3_SUBDIR, root):
        if not candidate_root.is_dir():
            continue
        try:
            for candidate in sorted(candidate_root.iterdir(), key=lambda item: item.name.lower()):
                if candidate.is_file():
                    _consider(candidate)
        except OSError:
            pass

    return result or ["sam3.pt"]


def resolve_sam3_checkpoint(name: str) -> str:
    """SAM3 체크포인트 이름 → 앱이 스캔한 폴더의 **절대 경로**. 못 찾으면 그대로.

    관리형 Forge 는 `--data-dir` 아래 models 를 보므로 이름만 보내면 확장이
    `<data>/models/sam3/<name>` 을 찾다 'SAM3 checkpoint not found' 로 죽는다. 파일은
    사용자의 Forge `models/sam3` 에 있고, 확장은 절대 경로면 그대로 쓴다
    (sam3ext/core.py::resolve_checkpoint_path). 'auto'/'huggingface' 는 확장의 키워드다.
    """
    value = str(name or "").strip() or "sam3.pt"
    if value.lower() in {"auto", "huggingface"}:
        return value
    candidate = Path(value)
    if candidate.is_absolute():
        return value
    root = get_forge_root()
    for folder in (root / SAM3_SUBDIR, root):
        target = folder / candidate.name
        try:
            if target.is_file():
                return str(target.resolve())
        except OSError:
            continue
    return value
