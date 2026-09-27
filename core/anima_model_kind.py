# core/anima_model_kind.py
"""체크포인트 종류 — Anima 3.8B v2 번들 / 비 번들 Anima / 그 밖(비 Anima). 순수 로직(Qt·torch 없음).

sam-extra 의 Anima 3.8B 스크립트는 모델 종류마다 같은 인자를 다르게 쓴다(scripts/anima_3_8b.py process_batch):
v2 번들은 블록이 없어도 자동으로 켜지고(부정 커넥터·Bypass 만 의미가 있다), 비 번들 Anima 는 v1 ``enabled`` 일 때만
켜지고, 비 Anima 모델에 v1 을 켜면 ``off: install failed (RuntimeError)`` 와 트레이스백만 남긴다. 앱은 그래서 보낼
블록을 모델 종류로 정한다(core/anima38.plan).

- ``header_kind(path)``  safetensors 헤더(앞 8바이트 길이 + JSON)만 읽어 판정한다 — 확장과 같은 근거:
  메타데이터 ``architecture``/``anima_v2_bundle_format`` 가 번들이면 v2(sam3ext/anima38/files.py bundle_metadata),
  ``llm_adapter.`` 키가 있으면 Anima(Forge 는 이 키로 Anima 를 알아보고 텍스트 인코더로 옮긴다 —
  backend/loader.py process_anima), 없으면 other. 실측(연구 P9 §3.5): anima_baseV10·Anima-2.9B·UR_ANIMA =
  ``net.llm_adapter.*`` 118개, Anima-3.8B-v1.1 = 번들 메타데이터, krea2 = 0개. 바이너리로 열어 한글 경로도 된다.
  (realpath, 크기, mtime_ns) 로 캐시한다 — 파일을 바꾸면 다시 읽는다.
- ``name_kind(title)``   이름 휴리스틱(Comfy 컴파일러가 쓰던 규칙을 옮겼다). 이름만으로 other 라고 하지 않는다.
- ``classify(title, path=None, *, cached_only=False)``  제목 캐시(연결 때 데몬 스레드가 채운다 —
  ui/anima38_ui.prewarm_model_kinds) → 헤더 → 이름. GUI 스레드는 ``cached_only=True`` 로 디스크를 읽지 않는다.

캐시는 데몬 스레드가 쓰고 GUI 스레드가 읽으므로 락으로 지킨다(critic B16).
"""
from __future__ import annotations

import json
import os
import re
import struct
import threading
from typing import Any, Iterable, Mapping, Optional

KIND_V2, KIND_ANIMA, KIND_OTHER, KIND_UNKNOWN = "v2", "anima", "other", "unknown"
KINDS = (KIND_V2, KIND_ANIMA, KIND_OTHER, KIND_UNKNOWN)

# 확장 sam3ext/anima38/files.py 상수 — SEMANTIC_PINS(anima38_bundle_architecture·_bundle_format·_v1_architecture)가 지킨다
BUNDLE_ARCHITECTURE = "anima_3_8b_semantic_connector_v2_bundle"
BUNDLE_FORMAT = "1"
BUNDLE_FORMAT_KEY = "anima_v2_bundle_format"
V1_ADAPTER_ARCHITECTURE = "anima_progressive_qwen35_cross_adapter_v1"
# 실측한 Anima 키 접두어. Forge 는 'net.'·'model.diffusion_model.' 접두어를 떼고 'llm_adapter' 로 본다 — 판정은
# 경로 조각 'llm_adapter.' 가 키의 맨 앞이거나 '.' 뒤에 오는지로 한다(두 저장 형식 모두).
ANIMA_KEY_PREFIX = "net.llm_adapter."
_ANIMA_KEY_SEGMENT = "llm_adapter."
MAX_HEADER_BYTES = 64 * 1024 * 1024   # safetensors 헤더 상한(정상 체크포인트는 수십~수백 KB)

# 이름 휴리스틱 표지(core/comfy_workflow_compiler 의 옛 _looks_like_* 와 같은 규칙)
_V2_FAMILY_MARKERS = ("3.8b", "3-8b", "3_8b")
_V2_RELEASE_MARKERS = ("-v2", "_v2", ".v2", "-v1.1", "_v1.1", ".v1.1")
QWEN35_MARKERS = ("qwen35_4b", "qwen3.5-4b", "qwen3_5_4b")   # sam3ext/anima38/files.py QWEN35_MARKERS 와 같다
_HASH_SUFFIX_RE = re.compile(r"\s*\[[0-9a-fA-F]{6,}\]\s*$")

_lock = threading.Lock()
_header_cache: dict[str, tuple[int, int, Optional[str]]] = {}   # normcase(realpath) → (size, mtime_ns, kind|None)
_title_kinds: dict[str, str] = {}                                # 제목 → 헤더로 확인한 종류


def _filename(value: Any) -> str:
    return os.path.basename(str(value or "").replace("\\", "/")).strip()


def checkpoint_key(title: Any) -> str:
    """비교용 이름 — 폴더·뒤의 해시(' [abcd1234]')를 떼고 소문자. Forge 옵션 값과 모델 목록 제목이 해시 유무로 달라도 같다."""
    return _HASH_SUFFIX_RE.sub("", _filename(title)).strip().casefold()


def _stem(title: Any) -> str:
    key = checkpoint_key(title)
    for ext in (".safetensors", ".ckpt", ".gguf", ".pt", ".pth", ".bin", ".sft"):
        if key.endswith(ext):
            return key[: -len(ext)]
    return key


def same_checkpoint(first: Any, second: Any) -> bool:
    """두 체크포인트 이름이 같은 파일을 가리키나(폴더·해시·확장자·대소문자 무시). 한쪽이라도 비면 False(모름).

    확장자까지 떼는 이유: Forge 의 ``sd_model_checkpoint`` 는 체크포인트 드롭다운이 보여 주는 값이고,
    'Show filenames without folder'(``sd_checkpoint_dropdown_use_short``) 를 켜면 ``short_title`` =
    확장자 없는 이름 + 해시다(modules/sd_models.py ``short_title``·``checkpoint_tiles``). ``name_for_extra``
    (확장자 없는 이름)도 Forge 가 받는 별칭이다. 확장자까지 보면 같은 모델을 다른 모델로 보고 Anima38 블록을 뺀다."""
    left = _stem(first)
    return bool(left) and left == _stem(second)


# ── 헤더 ───────────────────────────────────────────────────────────────────────
def _is_anima_key(key: str) -> bool:
    return key.startswith(_ANIMA_KEY_SEGMENT) or f".{_ANIMA_KEY_SEGMENT}" in key


def kind_from_header(header: Any) -> Optional[str]:
    """safetensors 헤더 dict → 종류. 모양이 틀리면 None."""
    if not isinstance(header, Mapping):
        return None
    metadata = header.get("__metadata__")
    metadata = metadata if isinstance(metadata, Mapping) else {}
    if (metadata.get("architecture") == BUNDLE_ARCHITECTURE
            and metadata.get(BUNDLE_FORMAT_KEY) == BUNDLE_FORMAT):
        return KIND_V2
    if any(_is_anima_key(str(key)) for key in header if key != "__metadata__"):
        return KIND_ANIMA
    return KIND_OTHER


def _read_header_kind(path: str, size: int) -> Optional[str]:
    try:
        with open(path, "rb") as handle:
            prefix = handle.read(8)
            if len(prefix) != 8:
                return None
            (length,) = struct.unpack("<Q", prefix)
            if length <= 0 or length > MAX_HEADER_BYTES or length > max(0, size - 8):
                return None
            raw = handle.read(length)
        if len(raw) != length:
            return None
        return kind_from_header(json.loads(raw.decode("utf-8")))
    except (OSError, ValueError, UnicodeDecodeError, struct.error):
        return None


def header_kind(path: Any) -> Optional[str]:
    """safetensors 헤더로 판정한 종류, 읽지 못하면 None(없는 파일·다른 형식·깨진 헤더·64 MB 초과). 디스크를 읽는다 —
    GUI 스레드에서 부르지 않는다."""
    if not path:
        return None
    try:
        real = os.path.realpath(os.fspath(path))
        stat = os.stat(real)
    except (OSError, TypeError, ValueError):
        return None
    cache_key = os.path.normcase(real)
    with _lock:
        cached = _header_cache.get(cache_key)
    if cached is not None and cached[0] == stat.st_size and cached[1] == stat.st_mtime_ns:
        return cached[2]
    kind = _read_header_kind(real, stat.st_size)
    with _lock:
        _header_cache[cache_key] = (stat.st_size, stat.st_mtime_ns, kind)
    return kind


# ── 이름 ───────────────────────────────────────────────────────────────────────
def looks_like_v2_bundle_name(value: Any) -> bool:
    """번들 릴리스 이름(Anima 3.8B + v2/v1.1) — 모든 Anima UNET 을 v2 로 보지 않는다."""
    name = _filename(value).casefold()
    return ("anima" in name and any(m in name for m in _V2_FAMILY_MARKERS)
            and any(m in name for m in _V2_RELEASE_MARKERS))


def looks_like_anima_name(value: Any) -> bool:
    return "anima" in _filename(value).casefold()


def looks_like_qwen35(value: Any) -> bool:
    name = _filename(value).casefold()
    return any(marker in name for marker in QWEN35_MARKERS)


def looks_like_anima_adapter(value: Any) -> bool:
    name = _filename(value).casefold()
    return "anima" in name and any(marker in name for marker in ("adapter", "connector"))


def v1_adapter_modules(modules: Iterable[Any]) -> list[str]:
    """모듈 목록의 Anima v1 어댑터(이름으로 — Qwen3.5 로 보이는 것은 뺀다), 목록 순서 그대로."""
    return [str(item).strip() for item in (modules or ())
            if str(item or "").strip() and looks_like_anima_adapter(item) and not looks_like_qwen35(item)]


def has_v1_module_pair(modules: Iterable[Any]) -> bool:
    """모듈 목록에 Qwen3.5 인코더와 Anima 어댑터가 함께 있나 — Comfy 컴파일러가 예전에 v1 을 자동으로 켜던 조건."""
    items = [item for item in (modules or ()) if str(item or "").strip()]
    return any(looks_like_qwen35(item) for item in items) and bool(v1_adapter_modules(items))


def name_kind(title: Any) -> str:
    """이름으로만 — v2 / anima / unknown. 이름에 anima 가 없어도 Anima 일 수 있으므로 other 는 돌려주지 않는다."""
    if looks_like_v2_bundle_name(title):
        return KIND_V2
    if looks_like_anima_name(title):
        return KIND_ANIMA
    return KIND_UNKNOWN


# ── 제목 캐시 ──────────────────────────────────────────────────────────────────
def _title_key(title: Any) -> str:
    return str(title or "").strip()


def remember_kinds(kinds: Mapping[str, str], *, replace: bool = False) -> None:
    """헤더로 확인한 {제목: 종류} 를 기억한다(연결 때 데몬 스레드). ``replace`` 면 옛 목록을 버린다."""
    clean = {_title_key(title): kind for title, kind in (kinds or {}).items()
             if _title_key(title) and kind in (KIND_V2, KIND_ANIMA, KIND_OTHER)}
    with _lock:
        if replace:
            _title_kinds.clear()
        _title_kinds.update(clean)


def forget_kinds() -> None:
    with _lock:
        _title_kinds.clear()


def cached_kinds() -> dict[str, str]:
    """기억한 {제목: 종류} 사본(락 안에서 복사)."""
    with _lock:
        return dict(_title_kinds)


def clear_header_cache() -> None:
    with _lock:
        _header_cache.clear()


def classify(title: Any, path: Any = None, *, cached_only: bool = False) -> str:
    """제목(과 로컬 경로) → 종류. 제목 캐시 → (``cached_only`` 가 아니면) 헤더 → 이름.

    ``cached_only=True`` 는 디스크를 건드리지 않는다(GUI 스레드). 캐시가 없으면 이름으로만 — 이름은 other 를 모르므로
    확인 전의 비 Anima 모델은 unknown 이다(core/anima38.plan 은 unknown 에 앱 기본값을 보내지 않는다)."""
    key = _title_key(title)
    with _lock:
        known = _title_kinds.get(key)
    if known:
        return known
    if path and not cached_only:
        kind = header_kind(path)
        if kind:
            if key:
                with _lock:
                    _title_kinds[key] = kind
            return kind
    return name_kind(title)


def kind_for_model_name(name: Any) -> str:
    """infotext 'Model' 값(확장자·해시 없는 이름)처럼 제목과 모양이 다른 이름 → 종류. 기억한 제목과 이름 줄기가 같으면
    그 종류, 아니면 이름으로만."""
    stem = _stem(name)
    if stem:
        with _lock:
            for title, kind in _title_kinds.items():
                if _stem(title) == stem:
                    return kind
    return name_kind(name)


__all__ = [
    "ANIMA_KEY_PREFIX", "BUNDLE_ARCHITECTURE", "BUNDLE_FORMAT", "BUNDLE_FORMAT_KEY", "KINDS", "KIND_ANIMA",
    "KIND_OTHER", "KIND_UNKNOWN", "KIND_V2", "MAX_HEADER_BYTES", "QWEN35_MARKERS", "V1_ADAPTER_ARCHITECTURE",
    "cached_kinds", "checkpoint_key", "classify", "clear_header_cache", "forget_kinds", "has_v1_module_pair",
    "header_kind", "kind_for_model_name", "kind_from_header", "looks_like_anima_adapter", "looks_like_anima_name",
    "looks_like_qwen35", "looks_like_v2_bundle_name", "name_kind", "remember_kinds", "same_checkpoint",
    "v1_adapter_modules",
]
