"""에디터 YOLO 검출 모델 목록 — Qt/cv2 없는 순수 로직.

모델 목록 = yolo_config.json 의 model_paths(명시 등록) + Editor_models/ 자동 감지 − disabled_models.

예전(숨은 PyQt 에디터의 tabs/editor/mosaic_panel.py, 은퇴해 삭제됨)에는 자동 감지분을 무조건 다시 넣어서 'YOLO 모델 초기화'가
config 만 [] 로 비우고 화면 라벨만 'No Model Loaded' 로 바꿨다. auto_detect/auto_censor 는
계속 폴더의 penis.pt·pussy.pt 를 썼고 에디터를 다시 열면 모델명이 되살아났다. 그래서 초기화는
지금 감지된 모델을 disabled_models 에 넣고, 추가는 그 모델을 disabled_models 에서 뺀다.

테스트가 패치하는 모듈 전역 이름(_PROJECT_ROOT/_EDITOR_MODELS_DIR/_YOLO_CONFIG_PATH)은
호출 시점에 읽는다.
"""

from __future__ import annotations

import json
import os
import shutil
from typing import Callable, Iterable

from utils.atomic_json import atomic_write_json

_PROJECT_ROOT = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
_EDITOR_MODELS_DIR = os.path.join(_PROJECT_ROOT, "Editor_models")
_YOLO_CONFIG_PATH = os.path.join(_EDITOR_MODELS_DIR, "yolo_config.json")

MODEL_EXTENSIONS = ('.pt', '.onnx', '.safetensors')
NO_MODEL_LABEL = "No Model Loaded"

# SAM 계열 모델 파일명 키워드 (YOLO 검출용으로 잘못 로드되는 것을 방지)
_SAM_KEYWORDS = (
    'mobile_sam', 'fastsam', 'fast_sam',
    'sam_vit', 'sam_hq', 'sam2', 'sam3',
)


def is_sam_file(fname: str) -> bool:
    """파일명이 SAM 계열인지 판정"""
    fl = os.path.basename(str(fname or '')).lower()
    return any(k in fl for k in _SAM_KEYWORDS)


def get_editor_models_dir() -> str:
    """Editor_models 디렉터리(없으면 만든다)."""
    try:
        os.makedirs(_EDITOR_MODELS_DIR, exist_ok=True)
    except OSError:
        pass
    return _EDITOR_MODELS_DIR


def _same_path_key(path: str) -> str:
    return os.path.normcase(os.path.abspath(path)) if path else ''


def resolve_model_path(path: str) -> str:
    """설정 경로를 현재 프로젝트 기준 절대경로로 변환한다.

    프로젝트 내부 경로는 다른 PC에서도 동작하도록 상대경로를 허용한다.
    이전 PC의 절대경로가 남아 있으면 Editor_models의 동일 파일명으로 이관한다.
    """
    value = os.path.expandvars(os.path.expanduser(str(path or '').strip()))
    if not value:
        return ''

    if not os.path.isabs(value):
        from_root = os.path.abspath(os.path.join(_PROJECT_ROOT, value))
        from_models = os.path.abspath(os.path.join(_EDITOR_MODELS_DIR, value))
        value = from_root if os.path.exists(from_root) else from_models
    else:
        value = os.path.abspath(value)

    if not os.path.exists(value):
        local_copy = os.path.join(_EDITOR_MODELS_DIR, os.path.basename(value))
        if os.path.exists(local_copy):
            value = local_copy
    return value


def portable_model_path(path: str) -> str:
    """프로젝트 내부 모델은 프로젝트 루트 상대경로로 저장한다."""
    resolved = resolve_model_path(path)
    if not resolved:
        return ''
    try:
        relative = os.path.relpath(resolved, _PROJECT_ROOT)
    except ValueError:  # Windows에서 드라이브가 다른 외부 경로
        return resolved
    if relative == os.pardir or relative.startswith(os.pardir + os.sep):
        return resolved
    return relative.replace(os.sep, '/')


def _read_config() -> tuple[list[str], list[str]]:
    """(model_paths, disabled_models) — 둘 다 설정 파일에 적힌 원문 경로."""
    try:
        with open(_YOLO_CONFIG_PATH, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except (OSError, ValueError):
        return [], []
    if not isinstance(data, dict):
        return [], []
    paths = data.get('model_paths') or []
    if not paths:
        old = data.get('model_path', '')
        paths = [old] if old else []
    disabled = data.get('disabled_models') or []
    clean = lambda items: [str(p) for p in items if isinstance(p, str) and p.strip()]  # noqa: E731
    return (clean(paths) if isinstance(paths, list) else []), (clean(disabled) if isinstance(disabled, list) else [])


def _unique_portable(paths: Iterable[str]) -> list[str]:
    out, seen = [], set()
    for p in paths:
        portable = portable_model_path(p)
        key = _same_path_key(resolve_model_path(portable)) if portable else ''
        if portable and key not in seen:
            seen.add(key)
            out.append(portable)
    return out


def _write_config(paths: Iterable[str], disabled: Iterable[str]) -> None:
    get_editor_models_dir()
    atomic_write_json(_YOLO_CONFIG_PATH, {
        'model_paths': _unique_portable(paths),
        'disabled_models': _unique_portable(disabled),
    })


def scan_models_dir() -> list[str]:
    """Editor_models/ 안의 검출 모델 후보(SAM 계열 제외) 절대경로."""
    models_dir = _EDITOR_MODELS_DIR
    try:
        names = sorted(os.listdir(models_dir))
    except OSError:
        return []
    out = []
    for name in names:
        fp = os.path.join(models_dir, name)
        if name.lower().endswith(MODEL_EXTENSIONS) and os.path.isfile(fp) and not is_sam_file(name):
            out.append(fp)
    return out


def load_model_paths() -> list[str]:
    """현재 쓸 YOLO 모델 절대경로 목록 (명시 등록 + 자동 감지 − 비활성, 존재하는 것만, SAM 제외)."""
    config_paths, disabled = _read_config()
    disabled_keys = {_same_path_key(resolve_model_path(p)) for p in disabled}
    out, seen = [], set()
    for p in [resolve_model_path(p) for p in config_paths] + scan_models_dir():
        key = _same_path_key(p)
        if not p or key in seen or key in disabled_keys:
            continue
        seen.add(key)
        if os.path.exists(p) and not is_sam_file(p):
            out.append(p)
    return out


def clear_models() -> list[str]:
    """모델 목록 초기화 — 지금 쓰이는 모델(자동 감지 포함)을 모두 비활성으로 기록한다."""
    _config_paths, disabled = _read_config()
    _write_config([], list(disabled) + load_model_paths() + scan_models_dir())
    return load_model_paths()


def add_models(paths: Iterable[str], copy: Callable[[str, str], object] = shutil.copy2) -> list[str]:
    """모델 추가 — Editor_models 밖의 파일은 복사하고(같은 이름이 있으면 그 파일을 쓴다),
    비활성 목록에서 빼고 명시 등록한다. 매번 설정·폴더를 새로 읽으므로 자동 감지분이
    config 에 영구 기록되지 않는다. 반환: 갱신된 모델 목록."""
    models_dir = get_editor_models_dir()
    models_key = _same_path_key(models_dir)
    config_paths, disabled = _read_config()
    added = []
    for raw in paths:
        p = os.path.abspath(str(raw or '').strip()) if str(raw or '').strip() else ''
        if not p:
            continue
        if _same_path_key(os.path.dirname(p)) != models_key:
            dst = os.path.join(models_dir, os.path.basename(p))
            if not os.path.exists(dst):
                copy(p, dst)
            p = dst
        added.append(p)
    added_keys = {_same_path_key(p) for p in added}
    disabled = [d for d in disabled if _same_path_key(resolve_model_path(d)) not in added_keys]
    _write_config(list(config_paths) + added, disabled)
    return load_model_paths()


def model_label(paths: list | None = None) -> str:
    """화면 라벨 — 모델 파일명 목록, 없으면 'No Model Loaded'."""
    items = load_model_paths() if paths is None else paths
    names = [os.path.basename(p) for p in items]
    return ", ".join(names) if names else NO_MODEL_LABEL
