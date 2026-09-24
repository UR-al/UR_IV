"""Authenticated generation gateway for local and approved remote backends.

The module deliberately keeps HTTP transport, job lifecycle, persistence and
backend routing behind :class:`GenerationApiManager`.  Callers submit one
normalised job; they never receive a backend object and cannot supply an
arbitrary URL, workflow graph, or local file path.

The service is disabled by default and binds to loopback when enabled.  Its
token is stored in ``user_data/generation_api.json`` (an ignored user-data
directory), while job manifests and artifacts live below
``user_data/generation_api/results``.
"""

from __future__ import annotations

import base64
import binascii
import copy
import functools
import hmac
import ipaddress
import json
import logging
import math
import mimetypes
import os
import queue
import re
import secrets
import socket
import threading
import time
import uuid
from collections import OrderedDict
from datetime import datetime, timezone
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from io import BytesIO
from pathlib import Path
from typing import Any, Callable, Mapping, Optional
from urllib.parse import parse_qs, urlparse

from backends.base import GenerationResult, MediaArtifact
# 기본 포트·설정 경로는 관리형 엔진 포트 선택(core/backend_runtime._choose_port)도 읽는다.
# DEFAULT_PORT 는 관리형 Forge(17860~17909)·ComfyUI(18188~18237) 자동 포트 범위 밖이다.
from core.generation_api_port import DEFAULT_PORT, default_config_path
from core.resource_coordinator import ResourceBusyError, get_generation_coordinator
from utils.atomic_json import atomic_write_json, load_json_safe


logger = logging.getLogger(__name__)

SCHEMA_VERSION = 2
DEFAULT_BIND_HOST = "127.0.0.1"
# schema 1 의 기본 포트 — 관리형 Forge 의 기본 포트와 같아 동시에 켜면 충돌했다.
LEGACY_DEFAULT_PORT = 17860
DEFAULT_MAX_QUEUE = 32
DEFAULT_MAX_BODY_BYTES = 32 * 1024 * 1024
MAX_HTTP_CONNECTIONS = 32
HTTP_SOCKET_TIMEOUT_SECONDS = 10
MAX_TARGETS = 32
MAX_RECENT_JOBS = 200
MAX_ARTIFACTS_PER_JOB = 64
MAX_ARTIFACT_BYTES_PER_JOB = 2 * 1024 * 1024 * 1024
MAX_A1111_RESPONSE_BYTES = 128 * 1024 * 1024
MAX_PERSISTED_REQUEST_BYTES = 256 * 1024
TERMINAL_STATES = frozenset({"completed", "failed", "cancelled"})
ALLOWED_MODES = frozenset({"txt2img", "img2img"})
ALLOWED_FAMILIES = frozenset({"standard", "krea2"})

_TARGET_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")
_SAFE_FILENAME = re.compile(r"[^A-Za-z0-9._-]+")
_SAFE_MIME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9.+-]{0,63}/[A-Za-z0-9][A-Za-z0-9.+-]{0,127}$")
_URL_IN_ERROR = re.compile(r"https?://[^\s\]\[()<>{}\"']+", re.IGNORECASE)
_WINDOWS_PATH_IN_ERROR = re.compile(r"(?<![A-Za-z0-9])[A-Za-z]:[\\/][^\r\n\"'<>|]+")
_UNC_PATH_IN_ERROR = re.compile(r"\\\\[^\s\\/]+[\\/][^\r\n\"'<>|]+")
_POSIX_PATH_IN_ERROR = re.compile(r"(?<![:\w])/(?:[^/\s]+/)+[^/\s]+")
_FORBIDDEN_REQUEST_KEYS = frozenset({
    "url", "api_url", "endpoint", "workflow", "workflow_path",
    "workflowpath", "img2img_workflow_path", "img2imgworkflowpath",
    "local_path", "filepath", "file_path",
    "script_name", "script_args", "alwayson_scripts",
    "callback", "callback_url", "webhook", "webhook_url",
    "output_path", "output_dir", "outdir", "outdir_samples", "outdir_grids",
    "save_path", "save_dir", "filename",
})
_IMAGE_PAYLOAD_KEYS = frozenset({
    "mask", "image", "input_image", "krea2_reference_image",
})
_IMAGE_LIST_PAYLOAD_KEYS = frozenset({"init_images", "controlnet_input_images"})
_PROMPT_TEXT_KEYS = frozenset({"prompt", "negative_prompt"})
_RASTER_MIME_BY_FORMAT = {
    "PNG": "image/png",
    "JPEG": "image/jpeg",
    "WEBP": "image/webp",
    "GIF": "image/gif",
    "BMP": "image/bmp",
    "TIFF": "image/tiff",
}
_INLINE_ARTIFACT_MIMES = frozenset(_RASTER_MIME_BY_FORMAT.values())


class GenerationApiError(RuntimeError):
    """Base error with an HTTP status suitable for the transport adapter."""

    status = HTTPStatus.BAD_REQUEST


class GenerationValidationError(GenerationApiError):
    status = HTTPStatus.BAD_REQUEST


class GenerationNotFoundError(GenerationApiError):
    status = HTTPStatus.NOT_FOUND


class GenerationQueueFullError(GenerationApiError):
    status = HTTPStatus.TOO_MANY_REQUESTS


class GenerationConflictError(GenerationApiError):
    status = HTTPStatus.CONFLICT


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def _clean_unicode(value: Any) -> str:
    """Return display-safe UTF-8 text even for hostile backend metadata."""

    return str(value).encode("utf-8", errors="replace").decode("utf-8")


def _json_safe(value: Any, *, depth: int = 0) -> Any:
    """Convert backend metadata to bounded JSON-compatible values."""

    if depth > 8:
        return _clean_unicode(value)[:1000]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if value is None or isinstance(value, (bool, int, float, str)):
        return value if not isinstance(value, str) else _clean_unicode(value)[:100_000]
    if isinstance(value, Mapping):
        return {
            _clean_unicode(key)[:200]: _json_safe(item, depth=depth + 1)
            for key, item in list(value.items())[:500]
        }
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(item, depth=depth + 1) for item in list(value)[:1000]]
    return _clean_unicode(value)[:10_000]


def _validate_json_text(value: Any, *, depth: int = 0) -> None:
    """Reject lone surrogates and pathological nesting in client JSON."""

    if depth > 32:
        raise GenerationValidationError("요청 JSON 중첩 깊이가 너무 큽니다")
    if isinstance(value, str):
        try:
            value.encode("utf-8")
        except UnicodeEncodeError as exc:
            raise GenerationValidationError("요청 문자열은 올바른 UTF-8이어야 합니다") from exc
        return
    if isinstance(value, Mapping):
        for key, item in value.items():
            _validate_json_text(str(key), depth=depth + 1)
            _validate_json_text(item, depth=depth + 1)
        return
    if isinstance(value, list):
        for item in value:
            _validate_json_text(item, depth=depth + 1)


def _without_data_uri(value: str) -> str:
    text = str(value or "").strip()
    if text.lower().startswith("data:"):
        if "," not in text:
            raise GenerationValidationError("이미지 data URL 형식이 올바르지 않습니다")
        header, text = text.split(",", 1)
        if ";base64" not in header.lower():
            raise GenerationValidationError("이미지는 base64 data URL이어야 합니다")
    return "".join(text.split())


def _validate_base64_image(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise GenerationValidationError(f"{label}은 base64 이미지여야 합니다")
    encoded = _without_data_uri(value)
    try:
        decoded = base64.b64decode(encoded, validate=True)
    except (ValueError, binascii.Error) as exc:
        raise GenerationValidationError(f"{label}의 base64 형식이 올바르지 않습니다") from exc
    if not decoded:
        raise GenerationValidationError(f"{label} 이미지가 비어 있습니다")
    if len(decoded) > DEFAULT_MAX_BODY_BYTES:
        raise GenerationValidationError(f"{label} 이미지는 32MB 이하여야 합니다")
    try:
        from PIL import Image

        with Image.open(BytesIO(decoded)) as image:
            image.verify()
            width, height = image.size
            image_format = str(image.format or "").upper()
    except Exception as exc:
        raise GenerationValidationError(f"{label}가 지원되는 이미지 파일이 아닙니다") from exc
    if image_format not in {"PNG", "JPEG", "WEBP", "BMP", "TIFF"}:
        raise GenerationValidationError(f"{label} 형식은 PNG/JPEG/WebP/BMP/TIFF만 지원합니다")
    if width < 1 or height < 1 or width > 16384 or height > 16384 or width * height > 64_000_000:
        raise GenerationValidationError(f"{label} 해상도는 최대 6400만 픽셀, 한 변 16384px 이하여야 합니다")
    return encoded


def _normalise_mime(value: Any) -> str:
    mime = str(value or "application/octet-stream").split(";", 1)[0].strip().lower()
    return mime if _SAFE_MIME.fullmatch(mime) else "application/octet-stream"


def _detect_raster_mime(data: Any = None, path: Optional[Path] = None) -> str:
    try:
        from PIL import Image

        source = BytesIO(bytes(data)) if isinstance(data, (bytes, bytearray, memoryview)) else path
        if source is None:
            return ""
        with Image.open(source) as image:
            image_format = str(image.format or "").upper()
            image.verify()
        return _RASTER_MIME_BY_FORMAT.get(image_format, "")
    except Exception:
        return ""


def _bounded_int(payload: Mapping[str, Any], key: str, minimum: int, maximum: int) -> Optional[int]:
    if key not in payload:
        return None
    raw = payload[key]
    if isinstance(raw, bool):
        raise GenerationValidationError(f"payload.{key}는 정수여야 합니다")
    if isinstance(raw, float) and (not math.isfinite(raw) or not raw.is_integer()):
        raise GenerationValidationError(f"payload.{key}는 정수여야 합니다")
    try:
        value = int(raw)
    except (TypeError, ValueError, OverflowError) as exc:
        raise GenerationValidationError(f"payload.{key}는 정수여야 합니다") from exc
    if not minimum <= value <= maximum:
        raise GenerationValidationError(f"payload.{key}는 {minimum}~{maximum} 범위여야 합니다")
    return value


def _bounded_float(payload: Mapping[str, Any], key: str, minimum: float, maximum: float) -> Optional[float]:
    if key not in payload:
        return None
    raw = payload[key]
    if isinstance(raw, bool):
        raise GenerationValidationError(f"payload.{key}는 숫자여야 합니다")
    try:
        value = float(raw)
    except (TypeError, ValueError) as exc:
        raise GenerationValidationError(f"payload.{key}는 숫자여야 합니다") from exc
    if not minimum <= value <= maximum:
        raise GenerationValidationError(f"payload.{key}는 {minimum}~{maximum} 범위여야 합니다")
    return value


def _validate_generation_payload(payload: dict[str, Any], *, family: str = "standard") -> None:
    for key in ("prompt", "negative_prompt"):
        if key in payload:
            if not isinstance(payload[key], str):
                raise GenerationValidationError(f"payload.{key}는 문자열이어야 합니다")
            if len(payload[key].encode("utf-8")) > 16 * 1024:
                raise GenerationValidationError(f"payload.{key}는 UTF-8 기준 16KiB 이하여야 합니다")

    width = _bounded_int(payload, "width", 64, 4096)
    height = _bounded_int(payload, "height", 64, 4096)
    for key, value in (("width", width), ("height", height)):
        if value is not None and value % 8:
            raise GenerationValidationError(f"payload.{key}는 8의 배수여야 합니다")
    _bounded_int(payload, "steps", 1, 150)
    batch_size = _bounded_int(payload, "batch_size", 1, 16) or 1
    n_iter = _bounded_int(payload, "n_iter", 1, 16)
    batch_count_alias = _bounded_int(payload, "batch_count", 1, 16)
    batch_count = max(n_iter or 1, batch_count_alias or 1)
    if batch_size * batch_count > 4:
        raise GenerationValidationError("batch_size × batch count는 최대 4여야 합니다")
    _bounded_float(payload, "cfg_scale", 0.0, 30.0)
    _bounded_float(payload, "cfg", 0.0, 30.0)
    _bounded_float(payload, "denoising_strength", 0.0, 1.0)
    # Krea2 는 앱 replay 범위(32비트)만 실행한다 — 받아 놓고 작업 단계에서 거부하지 않게
    # 검증 단계에서 family 별 상한을 적용한다(core/generation_family.seed_max_for_family).
    from core.generation_family import seed_max_for_family
    _bounded_int(payload, "seed", -1, seed_max_for_family(family))


def _sanitise_public_string(raw: Any, *, limit: int = 100_000) -> str:
    message = _clean_unicode(raw or "").replace("\x00", "")
    message = _URL_IN_ERROR.sub("[backend endpoint]", message)
    message = _UNC_PATH_IN_ERROR.sub("[local path]", message)
    message = _WINDOWS_PATH_IN_ERROR.sub("[local path]", message)
    message = _POSIX_PATH_IN_ERROR.sub("[local path]", message)
    return message[:limit]


def _sanitise_error(raw: Any) -> str:
    """Remove upstream endpoints and local paths before exposing an error."""

    return _sanitise_public_string(raw or "생성 실패", limit=4000)


def _redact_public_metadata(value: Any, *, depth: int = 0) -> Any:
    safe = _json_safe(value, depth=depth)
    if isinstance(safe, str):
        return _sanitise_public_string(safe)
    if isinstance(safe, dict):
        return {
            _sanitise_public_string(key, limit=200): _redact_public_metadata(item, depth=depth + 1)
            for key, item in safe.items()
        }
    if isinstance(safe, list):
        return [_redact_public_metadata(item, depth=depth + 1) for item in safe]
    return safe


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        if isinstance(value, bool):
            return default
        return int(value)
    except (TypeError, ValueError, OverflowError):
        return default


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if isinstance(value, bool):
            return default
        result = float(value)
        if result != result or result in {float("inf"), float("-inf")}:
            return default
        return result
    except (TypeError, ValueError, OverflowError):
        return default


def _normalise_url(raw: Any) -> str:
    value = str(raw or "").strip().rstrip("/")
    parsed = urlparse(value)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise GenerationValidationError("대상 URL은 http 또는 https 주소여야 합니다")
    if parsed.username or parsed.password or parsed.query or parsed.fragment:
        raise GenerationValidationError("대상 URL에 인증정보, 쿼리 또는 프래그먼트를 넣을 수 없습니다")
    if parsed.path not in {"", "/"}:
        raise GenerationValidationError("대상 URL은 경로가 없는 서버 루트 주소여야 합니다")
    try:
        port = parsed.port
    except ValueError as exc:
        raise GenerationValidationError("대상 URL 포트가 올바르지 않습니다") from exc
    if port is not None and not 1 <= port <= 65535:
        raise GenerationValidationError("대상 URL 포트가 올바르지 않습니다")
    host = parsed.hostname
    if ":" in host and not host.startswith("["):
        host = f"[{host}]"
    authority = host if port is None else f"{host}:{port}"
    return f"{parsed.scheme}://{authority}"


def _normalise_bind_host(raw: Any) -> str:
    value = str(raw or DEFAULT_BIND_HOST).strip()
    try:
        address = ipaddress.ip_address(value)
    except ValueError as exc:
        raise GenerationValidationError("API bindHost는 숫자 IP 주소여야 합니다") from exc
    if address.version != 4:
        raise GenerationValidationError("현재 API 서버는 IPv4 bindHost만 지원합니다")
    return str(address)


def _is_loopback_host(host: str) -> bool:
    value = str(host or "").strip().strip("[]").lower()
    if value == "localhost":
        return True
    try:
        return ipaddress.ip_address(value).is_loopback
    except ValueError:
        return False


def _local_ipv4_addresses() -> set[str]:
    """Return the IPv4 addresses that identify this host without shelling out."""

    addresses = {"127.0.0.1", "0.0.0.0"}
    names = {socket.gethostname(), socket.getfqdn(), "localhost"}
    for name in names:
        if not name:
            continue
        try:
            for item in socket.getaddrinfo(name, None, socket.AF_INET, socket.SOCK_STREAM):
                addresses.add(str(item[4][0]))
        except OSError:
            continue
    return addresses


def _host_is_local(host: str) -> bool:
    value = str(host or "").strip().strip("[]").lower()
    if not value:
        return False
    if _is_loopback_host(value) or value in {"0.0.0.0", socket.gethostname().lower(), socket.getfqdn().lower()}:
        return True
    local_addresses = _local_ipv4_addresses()
    try:
        return str(ipaddress.ip_address(value)) in local_addresses
    except ValueError:
        pass
    try:
        resolved = {
            str(item[4][0])
            for item in socket.getaddrinfo(value, None, socket.AF_INET, socket.SOCK_STREAM)
        }
    except OSError:
        return False
    return bool(resolved & local_addresses)


def _target_points_to_server(target_url: str, bind_host: str, port: int) -> bool:
    parsed = urlparse(target_url)
    target_port = parsed.port or (443 if parsed.scheme == "https" else 80)
    if target_port != port:
        return False
    target_host = str(parsed.hostname or "").lower()
    if bind_host == "0.0.0.0":
        return _host_is_local(target_host)
    if _is_loopback_host(bind_host):
        return _is_loopback_host(target_host)
    if target_host == bind_host:
        return True
    try:
        resolved = {
            str(item[4][0])
            for item in socket.getaddrinfo(target_host, None, socket.AF_INET, socket.SOCK_STREAM)
        }
    except OSError:
        return False
    return bind_host in resolved


def _normalise_workflow_path(raw: Any, label: str) -> str:
    value = str(raw or "").strip()
    if not value:
        return ""
    path = Path(value).expanduser()
    if not path.is_absolute():
        raise GenerationValidationError(f"{label}는 절대 경로여야 합니다")
    if path.suffix.lower() != ".json":
        raise GenerationValidationError(f"{label}는 JSON 파일이어야 합니다")
    if not path.is_file():
        raise GenerationValidationError(f"{label} 파일을 찾을 수 없습니다")
    return str(path.resolve())


def _normalise_target(raw: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(raw, Mapping):
        raise GenerationValidationError("대상 프로필은 객체여야 합니다")
    target_id = str(raw.get("id") or "").strip()
    if target_id == "active" or not _TARGET_ID.fullmatch(target_id):
        raise GenerationValidationError("대상 ID는 active 이외의 영문/숫자/./_/- 1~64자여야 합니다")
    engine = str(raw.get("engine") or raw.get("type") or "").strip().lower()
    if engine not in {"webui", "comfyui"}:
        raise GenerationValidationError("대상 engine은 webui 또는 comfyui여야 합니다")
    result: dict[str, Any] = {
        "id": target_id,
        "name": str(raw.get("name") or target_id).strip()[:100] or target_id,
        "engine": engine,
        # ``type`` is retained as a compatibility alias for older UI drafts.
        "type": engine,
        "url": _normalise_url(raw.get("url")),
        "enabled": bool(raw.get("enabled", True)),
    }
    if engine == "comfyui":
        result["workflowPath"] = _normalise_workflow_path(
            raw.get("workflowPath", raw.get("workflow_path", "")),
            "ComfyUI T2I workflowPath",
        )
        result["img2imgWorkflowPath"] = _normalise_workflow_path(
            raw.get("img2imgWorkflowPath", raw.get("img2img_workflow_path", "")),
            "ComfyUI I2I img2imgWorkflowPath",
        )
    return result


def _migrate_legacy_config(raw: Any) -> tuple[Any, bool]:
    """schema 1 설정을 현재 기본값으로 올린다. 반환: (설정, 바뀌었는지).

    schema 1 은 기본 포트가 관리형 Forge 기본 포트(17860)와 같았다. 사용자가 포트를 직접
    고른 적이 없는(=기본값 그대로이고 꺼져 있는) 설정만 새 기본 포트로 옮긴다. 켜 둔 설정은
    외부 클라이언트가 이미 그 주소를 쓰고 있을 수 있으니 그대로 둔다 — 이 경우 관리형 엔진의
    포트 선택(BackendRuntimeManager._choose_port)이 켜 둔 API 설정의 포트를 예약으로 보고
    건너뛴다(core/generation_api_port.reserved_ports). 앱 시작 때는 엔진 자동 시작이 API
    bind 보다 먼저라 bind 시험(core/port_probe.py)만으로는 못 피한다. schema 2 로 저장된
    뒤에는 사용자가 17860 을 다시 골라도 건드리지 않는다.
    """
    if not isinstance(raw, Mapping) or not raw:
        return raw, False
    try:
        version = int(raw.get("schemaVersion") or 1)
    except (TypeError, ValueError, OverflowError):
        version = 1
    if version >= SCHEMA_VERSION:
        return raw, False
    migrated = dict(raw)
    try:
        port = int(raw.get("port", LEGACY_DEFAULT_PORT))
    except (TypeError, ValueError, OverflowError):
        port = None
    if port == LEGACY_DEFAULT_PORT and not bool(raw.get("enabled", False)):
        migrated["port"] = DEFAULT_PORT
    migrated["schemaVersion"] = SCHEMA_VERSION
    return migrated, True


def _normalise_config(raw: Mapping[str, Any], previous: Optional[Mapping[str, Any]] = None) -> dict[str, Any]:
    if not isinstance(raw, Mapping):
        raise GenerationValidationError("API 설정은 객체여야 합니다")
    previous = previous or {}
    token = str(raw.get("token") or previous.get("token") or "").strip()
    if not token:
        token = secrets.token_urlsafe(32)
    if len(token) < 16 or len(token) > 512:
        raise GenerationValidationError("API token은 16~512자여야 합니다")
    bind_host = _normalise_bind_host(raw.get("bindHost", previous.get("bindHost", DEFAULT_BIND_HOST)))
    try:
        port = int(raw.get("port", previous.get("port", DEFAULT_PORT)))
    except (TypeError, ValueError, OverflowError) as exc:
        raise GenerationValidationError("API port는 정수여야 합니다") from exc
    if not 1024 <= port <= 65535:
        raise GenerationValidationError("API port는 1024~65535 범위여야 합니다")
    try:
        max_queue = int(raw.get("maxQueue", previous.get("maxQueue", DEFAULT_MAX_QUEUE)))
    except (TypeError, ValueError, OverflowError) as exc:
        raise GenerationValidationError("maxQueue는 정수여야 합니다") from exc
    if not 1 <= max_queue <= 256:
        raise GenerationValidationError("maxQueue는 1~256 범위여야 합니다")

    target_values = raw.get("targets", previous.get("targets", []))
    if not isinstance(target_values, list) or len(target_values) > MAX_TARGETS:
        raise GenerationValidationError(f"대상 프로필은 최대 {MAX_TARGETS}개까지 저장할 수 있습니다")
    targets = [_normalise_target(item) for item in target_values]
    ids = [item["id"] for item in targets]
    if len(ids) != len(set(ids)):
        raise GenerationValidationError("대상 프로필 ID가 중복되었습니다")
    for target in targets:
        if _target_points_to_server(target["url"], bind_host, port):
            raise GenerationValidationError("원격 대상이 이 API 서버 자신을 가리킬 수 없습니다")

    default_target = str(raw.get("defaultTarget", previous.get("defaultTarget", "active")) or "active").strip()
    if default_target != "active" and default_target not in ids:
        raise GenerationValidationError("defaultTarget이 저장된 대상 프로필에 없습니다")
    if default_target != "active" and not next(
        target["enabled"] for target in targets if target["id"] == default_target
    ):
        raise GenerationValidationError("비활성화된 대상은 defaultTarget으로 지정할 수 없습니다")
    return {
        "schemaVersion": SCHEMA_VERSION,
        "enabled": bool(raw.get("enabled", previous.get("enabled", False))),
        "bindHost": bind_host,
        "port": port,
        "token": token,
        "maxQueue": max_queue,
        "defaultTarget": default_target,
        "targets": targets,
    }


def _sanitise_request_payload(value: Any, *, key: str = "", depth: int = 0) -> Any:
    """Remove bulky image data before a job request is persisted."""

    if depth > 8:
        return "[truncated]"
    if isinstance(value, Mapping):
        return {
            str(k): _sanitise_request_payload(v, key=str(k), depth=depth + 1)
            for k, v in list(value.items())[:500]
        }
    if isinstance(value, list):
        return [_sanitise_request_payload(v, key=key, depth=depth + 1) for v in value[:1000]]
    if isinstance(value, str) and (
        key.lower() in _IMAGE_PAYLOAD_KEYS
        or key.lower() in {"init_images", "controlnet_input_images"}
    ):
        return f"[base64 image: {len(value)} chars]"
    return _json_safe(value, depth=depth)


def _bounded_request_summary(value: Mapping[str, Any]) -> dict[str, Any]:
    safe = _sanitise_request_payload(value)
    encoded = json.dumps(safe, ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode("utf-8")
    if len(encoded) <= MAX_PERSISTED_REQUEST_BYTES:
        return safe
    payload = value.get("payload") if isinstance(value.get("payload"), Mapping) else {}
    return {
        "target": str(value.get("target") or "active")[:100],
        "mode": str(value.get("mode") or "")[:40],
        "family": str(value.get("family") or "")[:40],
        "model": str(value.get("model") or "")[:500],
        "payload": {
            "_truncated": f"request summary exceeded {MAX_PERSISTED_REQUEST_BYTES} bytes",
            "keys": [str(key)[:200] for key in list(payload.keys())[:200]],
        },
    }


def _validate_no_external_references(value: Any, *, key_name: str = "", depth: int = 0) -> None:
    if depth > 10:
        raise GenerationValidationError("payload 중첩 깊이가 너무 큽니다")
    if isinstance(value, Mapping):
        for raw_key, item in value.items():
            key = str(raw_key).strip().lower()
            if key in _FORBIDDEN_REQUEST_KEYS or key.endswith(("_url", "_uri", "_path", "_dir", "_file")):
                raise GenerationValidationError(f"요청에서 외부 URL/경로/워크플로우 필드({raw_key})를 사용할 수 없습니다")
            _validate_no_external_references(item, key_name=key, depth=depth + 1)
    elif isinstance(value, list):
        for item in value:
            _validate_no_external_references(item, key_name=key_name, depth=depth + 1)
    elif isinstance(value, float) and not math.isfinite(value):
        raise GenerationValidationError("payload에는 NaN 또는 Infinity를 사용할 수 없습니다")
    elif isinstance(value, str) and key_name not in (
        _PROMPT_TEXT_KEYS | _IMAGE_PAYLOAD_KEYS | _IMAGE_LIST_PAYLOAD_KEYS
    ):
        text = value.strip()
        lowered = text.lower()
        looks_like_url = lowered.startswith(("http://", "https://", "file://"))
        looks_like_path = bool(
            _WINDOWS_PATH_IN_ERROR.match(text)
            or _UNC_PATH_IN_ERROR.match(text)
            or text.startswith("/")
        )
        if looks_like_url or looks_like_path:
            raise GenerationValidationError("요청에서 외부 URL 또는 절대 경로 값을 사용할 수 없습니다")


def _normalise_payload_images(value: Any, *, key_name: str = "payload", depth: int = 0) -> Any:
    """Validate every supported image-bearing field, including nested ControlNet input."""

    if depth > 10:
        raise GenerationValidationError("payload 중첩 깊이가 너무 큽니다")
    if isinstance(value, Mapping):
        result: dict[str, Any] = {}
        for raw_key, item in value.items():
            key = str(raw_key).strip().lower()
            if key in _IMAGE_LIST_PAYLOAD_KEYS:
                if not isinstance(item, list) or (key == "init_images" and not item):
                    raise GenerationValidationError(f"{raw_key}는 base64 이미지 목록이어야 합니다")
                if len(item) > 4:
                    raise GenerationValidationError(f"{raw_key}는 최대 4개까지 허용됩니다")
                result[raw_key] = [
                    _validate_base64_image(image, f"{raw_key}[{index}]")
                    for index, image in enumerate(item)
                ]
            elif key in _IMAGE_PAYLOAD_KEYS and item not in (None, ""):
                if isinstance(item, list):
                    if len(item) > 4:
                        raise GenerationValidationError(f"{raw_key}는 최대 4개까지 허용됩니다")
                    result[raw_key] = [
                        _validate_base64_image(image, f"{raw_key}[{index}]")
                        for index, image in enumerate(item)
                    ]
                else:
                    result[raw_key] = _validate_base64_image(item, str(raw_key))
            else:
                result[raw_key] = _normalise_payload_images(item, key_name=key, depth=depth + 1)
        return result
    if isinstance(value, list):
        return [_normalise_payload_images(item, key_name=key_name, depth=depth + 1) for item in value]
    return value


def _normalise_job_request(raw: Mapping[str, Any], config: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    if not isinstance(raw, Mapping):
        raise GenerationValidationError("생성 요청은 JSON 객체여야 합니다")
    allowed_top = {
        "target", "mode", "task", "operation", "model", "model_name", "payload",
        "family", "generation_family", "request_family",
    }
    unknown = sorted(str(key) for key in raw if key not in allowed_top)
    if unknown:
        raise GenerationValidationError("지원하지 않는 요청 필드: " + ", ".join(unknown[:10]))
    _validate_no_external_references(raw)

    target_id = str(raw.get("target") or config.get("defaultTarget") or "active").strip()
    profiles = {item["id"]: item for item in config.get("targets", [])}
    if target_id == "active":
        profile = {"id": "active", "name": "현재 앱 백엔드", "engine": "active", "type": "active"}
    elif target_id in profiles:
        profile = copy.deepcopy(profiles[target_id])
        if not profile.get("enabled", True):
            raise GenerationValidationError("비활성화된 대상 프로필입니다")
    else:
        raise GenerationValidationError("승인되지 않은 대상 프로필입니다")

    raw_mode = str(raw.get("mode") or raw.get("task") or raw.get("operation") or "txt2img").strip().lower()
    mode_aliases = {"t2i": "txt2img", "text2image": "txt2img", "i2i": "img2img", "image2image": "img2img"}
    mode = mode_aliases.get(raw_mode, raw_mode)
    if mode not in ALLOWED_MODES:
        raise GenerationValidationError("mode는 txt2img 또는 img2img여야 합니다")

    payload = raw.get("payload", {})
    if not isinstance(payload, Mapping):
        raise GenerationValidationError("payload는 JSON 객체여야 합니다")
    payload = copy.deepcopy(dict(payload))
    family = str(
        raw.get("family")
        or raw.get("generation_family")
        or raw.get("request_family")
        or payload.pop("_generation_family", "standard")
        or "standard"
    ).strip().lower()
    if family not in ALLOWED_FAMILIES:
        raise GenerationValidationError("generation family는 standard 또는 krea2여야 합니다")
    if family == "krea2" and profile.get("engine") == "webui":
        raise GenerationValidationError("Krea2는 ComfyUI 대상에서만 실행할 수 있습니다")

    if mode == "img2img":
        init_images = payload.get("init_images")
        if not isinstance(init_images, list) or not init_images:
            raise GenerationValidationError("img2img payload에는 init_images가 필요합니다")
    payload = _normalise_payload_images(payload)
    _validate_generation_payload(payload, family=family)
    if family == "krea2":
        batch_size = int(payload.get("batch_size", 1))
        batch_count = max(int(payload.get("n_iter", 1)), int(payload.get("batch_count", 1)))
        if batch_size * batch_count > 1:
            raise GenerationValidationError("Krea2 API 작업은 현재 batch 1만 지원합니다")

    model = str(raw.get("model") or raw.get("model_name") or payload.get("model") or "").strip()[:500]
    normalised = {
        "target": target_id,
        "mode": mode,
        "family": family,
        "model": model,
        "payload": payload,
    }
    return normalised, profile


def _profile_serial_key(profile: Mapping[str, Any]) -> str:
    """Collapse aliases of one endpoint onto the same mutation/cancel lane."""

    if profile.get("id") == "active" or profile.get("engine") == "active":
        return "active"
    engine = str(profile.get("engine") or profile.get("type") or "").lower()
    return _endpoint_serial_key(engine, str(profile.get("url") or ""))


def _endpoint_serial_key(engine: str, url: str) -> str:
    parsed = urlparse(str(url or "").strip())
    scheme = str(parsed.scheme or "http").lower()
    host = _canonical_endpoint_host(str(parsed.hostname or ""))
    port = parsed.port or (443 if scheme == "https" else 80)
    return f"{str(engine or '').lower()}:{scheme}://{host}:{port}"


def _canonical_endpoint_host(host: str) -> str:
    """Collapse common DNS/IP aliases so one backend keeps one serial lane."""

    value = str(host or "").strip().strip("[]").lower()
    if _is_loopback_host(value):
        return "127.0.0.1"
    try:
        return str(ipaddress.ip_address(value))
    except ValueError:
        pass
    try:
        return _resolve_canonical_dns_host(value)
    except (OSError, ValueError):
        # Resolution failures are intentionally not cached: a target configured
        # before DNS/network startup must converge onto its IP lane after DNS
        # recovers.
        return value


@functools.lru_cache(maxsize=128)
def _resolve_canonical_dns_host(host: str) -> str:
    addresses = sorted({
        str(ipaddress.ip_address(item[4][0]))
        for item in socket.getaddrinfo(host, None, socket.AF_INET, socket.SOCK_STREAM)
    })
    if not addresses:
        raise OSError("DNS address not found")
    return addresses[0]


# Test/diagnostic compatibility: callers historically cleared the public
# canonicalizer cache directly.
_canonical_endpoint_host.cache_clear = _resolve_canonical_dns_host.cache_clear  # type: ignore[attr-defined]


def _backend_serial_key(backend: Any) -> str:
    try:
        engine = str(backend.get_backend_type()).strip().lower()
    except Exception:
        return "active"
    api_url = str(getattr(backend, "api_url", "") or "").strip()
    if engine in {"webui", "comfyui"} and api_url:
        return _endpoint_serial_key(engine, api_url)
    return "active"


class GenerationApiManager:
    """Own configuration, jobs, backend routing and the optional HTTP server."""

    def __init__(
        self,
        *,
        config_path: str | os.PathLike[str] | None = None,
        storage_root: str | os.PathLike[str] | None = None,
        target_factory: Optional[Callable[[Mapping[str, Any]], Any]] = None,
        coordinator: Any = None,
        max_body_bytes: int = DEFAULT_MAX_BODY_BYTES,
    ) -> None:
        project_root = Path(__file__).resolve().parent.parent
        self.config_path = Path(config_path or default_config_path())
        self.storage_root = Path(storage_root or project_root / "user_data" / "generation_api" / "results")
        self.max_body_bytes = max(1024, int(max_body_bytes))
        self._target_factory = target_factory or self._default_target_factory
        self._coordinator = coordinator or get_generation_coordinator()

        self._lock = threading.RLock()
        self._condition = threading.Condition(self._lock)
        self._server_lock = threading.RLock()
        self._jobs: "OrderedDict[str, dict[str, Any]]" = OrderedDict()
        self._job_profiles: dict[str, dict[str, Any]] = {}
        self._job_backends: dict[str, Any] = {}
        self._previews: dict[str, bytes] = {}
        self._target_queues: dict[str, queue.Queue[Optional[str]]] = {}
        self._target_workers: dict[str, threading.Thread] = {}
        self._target_gates: dict[str, threading.Lock] = {}
        self._running_backends: dict[str, Any] = {}
        self._active_slots = 0
        self._shutdown = threading.Event()
        self._server: Optional[ThreadingHTTPServer] = None
        self._server_thread: Optional[threading.Thread] = None

        raw, migrated = _migrate_legacy_config(load_json_safe(str(self.config_path), {}))
        try:
            self._config = _normalise_config(raw)
        except GenerationValidationError:
            logger.warning("generation_api.json 설정이 올바르지 않아 안전한 기본값을 사용합니다")
            self._config = _normalise_config({})
            migrated = False
        if migrated or not self.config_path.exists():
            self._persist_config()
        self._load_manifests()

    # -- public manager/config API -------------------------------------------------

    def snapshot(self, include_secret: bool = False) -> dict[str, Any]:
        with self._lock:
            config = copy.deepcopy(self._config)
            if not include_secret:
                config["token"] = ""
            recent = [
                self._job_public(job, include_request=False)
                for job in list(self._jobs.values())[-10:]
            ][::-1]
        with self._server_lock:
            running = self._server is not None
            actual_port = int(self._server.server_address[1]) if self._server is not None else config["port"]
        display_host = config["bindHost"]
        if display_host == "0.0.0.0":
            display_host = "127.0.0.1"
        return {
            "config": config,
            "running": running,
            "listenUrl": f"http://{display_host}:{actual_port}",
            "recentJobs": recent,
            "nativeOperations": True,
            "supportedOperations": ["txt2img", "img2img", "krea2"],
        }

    def execute(self, action: str, payload: Optional[Mapping[str, Any]] = None) -> dict[str, Any]:
        operation = str(action or "").strip().lower()
        values = dict(payload or {})
        if operation == "save_config":
            state = self.save_config(values)
            return {"ok": True, "action": operation, "state": state}
        if operation == "start":
            state = self.start(persist_enabled=False)
            return {"ok": True, "action": operation, "state": state}
        if operation == "stop":
            state = self.stop(persist_enabled=False)
            return {"ok": True, "action": operation, "state": state}
        if operation == "rotate_token":
            with self._lock:
                self._config["token"] = secrets.token_urlsafe(32)
                self._persist_config()
            return {"ok": True, "action": operation, "state": self.snapshot(include_secret=True)}
        if operation == "test_target":
            draft = values.get("target")
            if isinstance(draft, Mapping):
                profile = _normalise_target(draft)
                with self._lock:
                    bind_host = str(self._config["bindHost"])
                    port = int(self._config["port"])
                if _target_points_to_server(str(profile["url"]), bind_host, port):
                    raise GenerationValidationError("원격 대상이 이 API 서버 자신을 가리킬 수 없습니다")
                target_id = str(profile["id"])
            else:
                target_id = str(values.get("targetId") or "").strip()
                profile = self._resolve_profile(target_id)
            backend = self._target_factory(copy.deepcopy(profile))
            ok = bool(backend.test_connection())
            return {
                "ok": ok,
                "action": operation,
                "targetId": target_id,
                "message": "연결 성공" if ok else "연결 실패",
                "state": self.snapshot(include_secret=False),
            }
        raise GenerationValidationError("지원하지 않는 generation API 작업입니다")

    def save_config(self, values: Mapping[str, Any]) -> dict[str, Any]:
        with self._lock:
            old_config = copy.deepcopy(self._config)
            new_config = _normalise_config(values, old_config)
        with self._server_lock:
            was_running = self._server is not None
            address_changed = (
                old_config["bindHost"], old_config["port"]
            ) != (
                new_config["bindHost"], new_config["port"]
            )
            if was_running and address_changed:
                self._stop_server_locked()
            with self._lock:
                self._config = new_config
                self._persist_config()
            try:
                if was_running and address_changed:
                    self._start_server_locked()
            except Exception:
                with self._lock:
                    self._config = old_config
                    self._persist_config()
                if was_running:
                    try:
                        self._start_server_locked()
                    except Exception:
                        logger.exception("기존 generation API 주소 복구에 실패했습니다")
                raise
        return self.snapshot(include_secret=True)

    def start_if_enabled(self) -> dict[str, Any]:
        with self._lock:
            enabled = bool(self._config.get("enabled"))
        if enabled:
            return self.start(persist_enabled=False)
        return self.snapshot(include_secret=False)

    def start(self, *, persist_enabled: bool = True) -> dict[str, Any]:
        with self._server_lock:
            if persist_enabled:
                with self._lock:
                    self._config["enabled"] = True
                    self._persist_config()
            try:
                self._start_server_locked()
            except Exception:
                if persist_enabled:
                    with self._lock:
                        self._config["enabled"] = False
                        self._persist_config()
                raise
        return self.snapshot(include_secret=True)

    def stop(self, *, persist_enabled: bool = True) -> dict[str, Any]:
        with self._server_lock:
            self._stop_server_locked()
            if persist_enabled:
                with self._lock:
                    self._config["enabled"] = False
                    self._persist_config()
        return self.snapshot(include_secret=True)

    # -- public job API -----------------------------------------------------------

    def submit(self, request: Mapping[str, Any]) -> dict[str, Any]:
        with self._lock:
            normalised, profile = _normalise_job_request(request, self._config)
            if self._shutdown.is_set():
                raise GenerationConflictError("generation API manager가 종료되었습니다")
            if self._active_slots >= int(self._config["maxQueue"]):
                raise GenerationQueueFullError("generation API 대기열이 가득 찼습니다")
            # The special ``active`` route means the backend selected when the
            # request is accepted.  Snapshot the adapter now so a later UI
            # backend switch cannot silently reroute a queued job.
            backend_snapshot = None
            if profile.get("id") == "active" or profile.get("engine") == "active":
                backend_snapshot = self._target_factory(copy.deepcopy(profile))
            serial_key = (
                _backend_serial_key(backend_snapshot)
                if backend_snapshot is not None
                else _profile_serial_key(profile)
            )
            job_id = uuid.uuid4().hex
            now = _utc_now()
            job = {
                "id": job_id,
                "state": "queued",
                "target": normalised["target"],
                "mode": normalised["mode"],
                "family": normalised["family"],
                "model": normalised["model"],
                "createdAt": now,
                "startedAt": "",
                "completedAt": "",
                "progress": 0.0,
                "currentStep": 0,
                "totalSteps": 0,
                "cancelRequested": False,
                "error": "",
                "info": {},
                "artifacts": [],
                "request": _bounded_request_summary(normalised),
                "_payload": normalised["payload"],
                "_serialKey": serial_key,
                "_slotReleased": False,
            }
            self._jobs[job_id] = job
            self._job_profiles[job_id] = profile
            if backend_snapshot is not None:
                self._job_backends[job_id] = backend_snapshot
            self._active_slots += 1
            self._trim_jobs_locked()
            target_queue = self._ensure_target_worker_locked(job["_serialKey"])
            self._persist_job(job)
            target_queue.put(job_id)
            self._condition.notify_all()
            return self._job_public(job)

    def inspect(self, job_id: str) -> dict[str, Any]:
        with self._lock:
            job = self._jobs.get(str(job_id))
            if job is None:
                raise GenerationNotFoundError("생성 작업을 찾을 수 없습니다")
            return self._job_public(job)

    def list_jobs(self, limit: int = 50) -> list[dict[str, Any]]:
        try:
            count = max(1, min(int(limit), 200))
        except (TypeError, ValueError):
            count = 50
        with self._lock:
            return [self._job_public(job) for job in list(self._jobs.values())[-count:]][::-1]

    def wait(self, job_id: str, timeout: Optional[float] = None) -> dict[str, Any]:
        if timeout is None:
            deadline = None
        else:
            try:
                timeout_value = float(timeout)
            except (TypeError, ValueError, OverflowError) as exc:
                raise GenerationValidationError("timeout은 0 이상의 유한한 숫자여야 합니다") from exc
            if not math.isfinite(timeout_value) or timeout_value < 0:
                raise GenerationValidationError("timeout은 0 이상의 유한한 숫자여야 합니다")
            deadline = time.monotonic() + timeout_value
        with self._condition:
            while True:
                job = self._jobs.get(str(job_id))
                if job is None:
                    raise GenerationNotFoundError("생성 작업을 찾을 수 없습니다")
                if job["state"] in TERMINAL_STATES:
                    return self._job_public(job)
                if deadline is not None:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        return self._job_public(job)
                    self._condition.wait(min(remaining, 0.5))
                else:
                    self._condition.wait(0.5)

    def cancel(self, job_id: str) -> dict[str, Any]:
        requested_id = str(job_id)
        with self._condition:
            job = self._jobs.get(requested_id)
            if job is None:
                raise GenerationNotFoundError("생성 작업을 찾을 수 없습니다")
            if job["state"] in TERMINAL_STATES:
                return self._job_public(job)
            if job["state"] == "queued":
                job["cancelRequested"] = True
                job["_payload"] = {}
                # Keep the bounded-queue slot until the target worker consumes
                # this tombstone. Otherwise submit/cancel loops could grow an
                # unbounded physical queue behind one long-running job.
                self._finish_locked(
                    job,
                    "cancelled",
                    error="사용자가 작업을 취소했습니다",
                    release_slot=False,
                )
                return self._job_public(job)
            serial_key = str(job.get("_serialKey", job["target"]))
            gate = self._target_gates.setdefault(serial_key, threading.Lock())

        # Serialize interrupt delivery with the target worker's terminal/start
        # transitions.  A delayed WebUI /interrupt must never reach the next
        # job that happens to use the same endpoint.
        with gate:
            with self._condition:
                job = self._jobs.get(requested_id)
                if job is None:
                    raise GenerationNotFoundError("생성 작업을 찾을 수 없습니다")
                if job["state"] in TERMINAL_STATES:
                    return self._job_public(job)
                job["cancelRequested"] = True
                backend = self._running_backends.get(serial_key)
                self._persist_job(job)
                self._condition.notify_all()
            try:
                if backend is not None:
                    backend.interrupt()
            except Exception:
                logger.debug("backend interrupt 실패", exc_info=True)
            with self._lock:
                current = self._jobs.get(requested_id)
                if current is None:
                    raise GenerationNotFoundError("생성 작업을 찾을 수 없습니다")
                return self._job_public(current)

    def artifact(self, job_id: str, index: int) -> tuple[bytes, str, str]:
        path, mime, filename = self.artifact_path(job_id, index)
        return path.read_bytes(), mime, filename

    def artifact_path(self, job_id: str, index: int) -> tuple[Path, str, str]:
        """Resolve a persisted artifact without loading it into memory."""

        with self._lock:
            job = self._jobs.get(str(job_id))
            if job is None:
                raise GenerationNotFoundError("생성 작업을 찾을 수 없습니다")
            try:
                item = job["artifacts"][int(index)]
            except (IndexError, TypeError, ValueError):
                raise GenerationNotFoundError("생성 결과 artifact를 찾을 수 없습니다")
            filename = Path(str(item.get("file") or "")).name
            mime = _normalise_mime(item.get("mime"))
        path = (self.storage_root / str(job_id) / filename).resolve()
        expected_parent = (self.storage_root / str(job_id)).resolve()
        if path.parent != expected_parent or not path.is_file():
            raise GenerationNotFoundError("생성 결과 artifact 파일을 찾을 수 없습니다")
        return path, mime, filename

    def cancel_target(self, target_id: str) -> int:
        with self._lock:
            ids = [
                job["id"] for job in self._jobs.values()
                if job["target"] == target_id and job["state"] not in TERMINAL_STATES
            ]
        for job_id in ids:
            self.cancel(job_id)
        return len(ids)

    def shutdown(self) -> None:
        with self._server_lock:
            self._stop_server_locked()
        self._shutdown.set()
        with self._lock:
            pending_ids = [job["id"] for job in self._jobs.values() if job["state"] not in TERMINAL_STATES]
            queues = list(self._target_queues.values())
        for job_id in pending_ids:
            try:
                self.cancel(job_id)
            except GenerationApiError:
                pass
        for target_queue in queues:
            target_queue.put(None)

    # -- backend execution --------------------------------------------------------

    @staticmethod
    def _default_target_factory(profile: Mapping[str, Any]) -> Any:
        engine = str(profile.get("engine") or profile.get("type") or "active")
        if engine == "active":
            from backends import get_backend

            return get_backend()
        if engine == "webui":
            from backends.webui_backend import WebUIBackend

            return WebUIBackend(str(profile["url"]))
        if engine == "comfyui":
            from backends.comfyui_backend import ComfyUIBackend

            return ComfyUIBackend(str(profile["url"]))
        raise GenerationValidationError("지원하지 않는 backend target입니다")

    def _ensure_target_worker_locked(self, serial_key: str) -> queue.Queue[Optional[str]]:
        target_queue = self._target_queues.get(serial_key)
        worker = self._target_workers.get(serial_key)
        self._target_gates.setdefault(serial_key, threading.Lock())
        if target_queue is None or worker is None or not worker.is_alive():
            target_queue = queue.Queue()
            safe_thread_name = re.sub(r"[^A-Za-z0-9_.-]+", "-", serial_key)[:80]
            worker = threading.Thread(
                target=self._target_worker,
                args=(serial_key, target_queue),
                name=f"generation-api-{safe_thread_name}",
                daemon=True,
            )
            self._target_queues[serial_key] = target_queue
            self._target_workers[serial_key] = worker
            worker.start()
        return target_queue

    def _target_worker(self, serial_key: str, target_queue: queue.Queue[Optional[str]]) -> None:
        while not self._shutdown.is_set():
            job_id = target_queue.get()
            try:
                if job_id is None:
                    return
                self._execute_job(job_id)
            finally:
                if job_id is not None:
                    # Also covers queued-cancel tombstones that return before
                    # _execute_job reaches its dispatch-level finally block.
                    with self._lock:
                        self._job_profiles.pop(job_id, None)
                        self._job_backends.pop(job_id, None)
                        self._previews.pop(job_id, None)
                        job = self._jobs.get(job_id)
                        if job is not None:
                            job["_payload"] = {}
                target_queue.task_done()

    def _execute_job(self, job_id: str) -> None:
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                return
            serial_key = str(job.get("_serialKey", job.get("target", "active")))
            gate = self._target_gates.setdefault(serial_key, threading.Lock())
        with gate:
            with self._condition:
                job = self._jobs.get(job_id)
                if job is None:
                    return
                if job["state"] != "queued":
                    self._release_slot_locked(job)
                    return
                if job["cancelRequested"]:
                    self._finish_locked(job, "cancelled", error="사용자가 작업을 취소했습니다")
                    return
                job["state"] = "running"
                job["startedAt"] = _utc_now()
                profile = copy.deepcopy(self._job_profiles.get(job_id) or self._resolve_profile(job["target"]))
                payload = copy.deepcopy(job["_payload"])
                job["_payload"] = {}
                self._persist_job(job)
                self._condition.notify_all()

        backend = None
        try:
            backend = self._job_backends.get(job_id)
            if backend is None:
                backend = self._target_factory(profile)

            def progress(current: int, total: int, preview: Optional[bytes] = None) -> None:
                self._update_progress(job_id, current, total, preview)

            def run() -> GenerationResult:
                # Register only once dispatch can really begin.  In particular,
                # a loopback target waiting for the shared GPU lease must not be
                # interruptible: its backend has not started, and interrupting
                # it could hit a different job already owning that endpoint/GPU.
                with self._lock:
                    current_job = self._jobs.get(job_id)
                    if current_job is None or current_job.get("cancelRequested"):
                        return GenerationResult(success=False, error="사용자가 작업을 취소했습니다")
                    self._running_backends[job["_serialKey"]] = backend
                try:
                    return self._dispatch_backend(backend, profile, job, payload, progress)
                finally:
                    with self._lock:
                        if self._running_backends.get(job["_serialKey"]) is backend:
                            self._running_backends.pop(job["_serialKey"], None)

            if self._uses_local_generation_lease(profile):
                result = self._run_with_generation_lease(job_id, run)
            else:
                result = run()
            if not isinstance(result, GenerationResult):
                raise TypeError("backend가 GenerationResult를 반환하지 않았습니다")

            with gate:
                with self._condition:
                    current_job = self._jobs.get(job_id)
                    if current_job is None or current_job["state"] in TERMINAL_STATES:
                        return
                    if current_job["cancelRequested"]:
                        self._finish_locked(current_job, "cancelled", error="사용자가 작업을 취소했습니다")
                        return
                    if not result.success:
                        self._finish_locked(current_job, "failed", error=str(result.error or "생성 실패"), info=result.info)
                        return
                    artifact_job = {"id": current_job["id"]}
            # Artifact IO can involve large Comfy videos. Hold neither the
            # global lock nor endpoint gate so cancel/shutdown stays responsive.
            artifacts = self._persist_result_artifacts(artifact_job, result)
            discard_artifacts = False
            with gate:
                with self._condition:
                    current_job = self._jobs.get(job_id)
                    if current_job is None or current_job["state"] in TERMINAL_STATES:
                        discard_artifacts = True
                    elif current_job["cancelRequested"]:
                        self._finish_locked(current_job, "cancelled", error="사용자가 작업을 취소했습니다")
                        discard_artifacts = True
                    else:
                        self._finish_locked(current_job, "completed", info=result.info, artifacts=artifacts)
            if discard_artifacts:
                self._discard_persisted_artifacts(job_id, artifacts)
        except Exception as exc:
            logger.exception("generation API 작업 실행 실패: %s", job_id)
            with gate:
                with self._condition:
                    current_job = self._jobs.get(job_id)
                    if current_job is not None and current_job["state"] not in TERMINAL_STATES:
                        state = "cancelled" if current_job["cancelRequested"] else "failed"
                        self._finish_locked(current_job, state, error=str(exc))
        finally:
            with self._lock:
                serial_key = job.get("_serialKey", job.get("target"))
                if self._running_backends.get(serial_key) is backend:
                    self._running_backends.pop(serial_key, None)
                self._job_profiles.pop(job_id, None)
                self._job_backends.pop(job_id, None)
                self._previews.pop(job_id, None)
                current_job = self._jobs.get(job_id)
                if current_job is not None:
                    current_job["_payload"] = {}

    def _dispatch_backend(
        self,
        backend: Any,
        profile: Mapping[str, Any],
        job: Mapping[str, Any],
        payload: dict[str, Any],
        progress: Callable[[int, int, Optional[bytes]], None],
    ) -> GenerationResult:
        cancel_check = lambda: self._job_cancel_requested(str(job["id"]))
        backend_type = ""
        try:
            backend_type = str(backend.get_backend_type()).strip().lower()
        except Exception:
            backend_type = str(profile.get("engine") or "").lower()
        if cancel_check():
            return GenerationResult(success=False, error="사용자가 작업을 취소했습니다")
        family = str(job["family"])
        operation = "t2i" if job["mode"] == "txt2img" else "i2i"
        if family == "krea2":
            if backend_type != "comfyui":
                return GenerationResult(success=False, error="Krea2는 ComfyUI 대상에서만 실행할 수 있습니다")
            from core.krea2_generation import run_krea2_generation

            return run_krea2_generation(
                backend,
                operation,
                payload,
                progress,
                cancel_check=cancel_check,
            )

        if backend_type == "comfyui" and profile.get("id") != "active":
            return self._run_named_comfy(
                backend,
                profile,
                str(job["mode"]),
                str(job["model"]),
                payload,
                progress,
                cancel_check,
            )
        method = backend.txt2img if job["mode"] == "txt2img" else backend.img2img
        return self._invoke_cancellable(
            method,
            str(job["model"]),
            payload,
            progress,
            cancel_check=cancel_check,
        )

    def _job_cancel_requested(self, job_id: str) -> bool:
        with self._lock:
            job = self._jobs.get(str(job_id))
            return job is None or bool(job.get("cancelRequested"))

    @staticmethod
    def _invoke_cancellable(
        method: Callable[..., Any],
        *args: Any,
        cancel_check: Callable[[], bool],
    ) -> Any:
        """Call newer adapters cooperatively without breaking older adapters."""

        if cancel_check():
            raise GenerationConflictError("사용자가 작업을 취소했습니다")
        # 시그니처 판정은 GUI 워커·채팅·Krea2 와 공용(core/cancellable_call.py).
        from core.cancellable_call import call_with_optional_cancel
        return call_with_optional_cancel(method, *args, cancel_check=cancel_check)

    @staticmethod
    def _load_workflow(path_value: str) -> dict[str, Any]:
        path = Path(path_value)
        if not path.is_file():
            raise RuntimeError(f"저장된 ComfyUI workflow를 찾을 수 없습니다: {path}")
        if path.stat().st_size > 16 * 1024 * 1024:
            raise RuntimeError("ComfyUI workflow JSON은 16MB 이하여야 합니다")
        with path.open("r", encoding="utf-8") as handle:
            workflow = json.load(handle)
        # 앱과 같은 규칙: ComfyUI API 포맷만 실행한다(웹 포맷은 Export (API) 안내).
        from core.comfy_workflow_format import require_api_workflow

        return require_api_workflow(workflow)

    def _run_named_comfy(
        self,
        backend: Any,
        profile: Mapping[str, Any],
        mode: str,
        model: str,
        payload: dict[str, Any],
        progress: Callable[[int, int, Optional[bytes]], None],
        cancel_check: Callable[[], bool],
    ) -> GenerationResult:
        path_key = "workflowPath" if mode == "txt2img" else "img2imgWorkflowPath"
        path_value = str(profile.get(path_key) or "")
        if not path_value:
            return GenerationResult(success=False, error=f"대상 프로필에 {path_key}가 설정되지 않았습니다")
        if cancel_check():
            return GenerationResult(success=False, error="사용자가 작업을 취소했습니다")
        workflow = self._load_workflow(path_value)
        if cancel_check():
            return GenerationResult(success=False, error="사용자가 작업을 취소했습니다")

        # ComfyUI targets are always backends.comfyui_backend.ComfyUIBackend
        # (_default_target_factory), whose generate_workflow applies the payload
        # with the app compiler's rules.  Anything else is a wiring error.
        generate_workflow = getattr(backend, "generate_workflow", None)
        if not callable(generate_workflow):
            return GenerationResult(
                success=False,
                error="이 ComfyUI 대상 adapter는 워크플로 생성(generate_workflow)을 지원하지 않습니다",
            )
        return self._invoke_cancellable(
            generate_workflow,
            mode,
            workflow,
            model,
            payload,
            progress,
            cancel_check=cancel_check,
        )

    def _run_with_generation_lease(self, job_id: str, callback: Callable[[], GenerationResult]) -> GenerationResult:
        while True:
            with self._lock:
                job = self._jobs.get(job_id)
                if job is None or job["cancelRequested"]:
                    return GenerationResult(success=False, error="사용자가 작업을 취소했습니다")
            try:
                with self._coordinator.reserve(
                    f"generation-api:{job_id[:12]}",
                    unload_llm=False,
                    timeout=0.25,
                ):
                    return callback()
            except ResourceBusyError:
                continue

    @staticmethod
    def _uses_local_generation_lease(profile: Mapping[str, Any]) -> bool:
        if profile.get("id") == "active" or profile.get("engine") == "active":
            return True
        parsed = urlparse(str(profile.get("url") or ""))
        return _host_is_local(str(parsed.hostname or ""))

    # -- lifecycle/persistence helpers -------------------------------------------

    def _update_progress(self, job_id: str, current: int, total: int, preview: Optional[bytes]) -> None:
        with self._condition:
            job = self._jobs.get(job_id)
            if job is None or job["state"] != "running":
                return
            try:
                current_value = max(0, int(current))
                total_value = max(0, int(total))
            except (TypeError, ValueError):
                return
            job["currentStep"] = max(job["currentStep"], current_value)
            job["totalSteps"] = max(job["totalSteps"], total_value)
            if total_value > 0:
                job["progress"] = max(job["progress"], min(1.0, current_value / total_value))
            if isinstance(preview, (bytes, bytearray)) and len(preview) <= 8 * 1024 * 1024:
                self._previews[job_id] = bytes(preview)
            self._condition.notify_all()

    def _finish_locked(
        self,
        job: dict[str, Any],
        state: str,
        *,
        error: str = "",
        info: Any = None,
        artifacts: Optional[list[dict[str, Any]]] = None,
        release_slot: bool = True,
    ) -> None:
        if job["state"] in TERMINAL_STATES:
            return
        if state not in TERMINAL_STATES:
            raise RuntimeError("terminal state가 아닙니다")
        job["state"] = state
        job["completedAt"] = _utc_now()
        job["error"] = _sanitise_error(error) if error else ""
        if info is not None:
            job["info"] = _redact_public_metadata(info)
        if artifacts is not None:
            job["artifacts"] = artifacts
        if state == "completed":
            job["progress"] = 1.0
        if release_slot:
            self._release_slot_locked(job)
        self._persist_job(job)
        self._condition.notify_all()

    def _release_slot_locked(self, job: dict[str, Any]) -> None:
        if not job.get("_slotReleased"):
            self._active_slots = max(0, self._active_slots - 1)
            job["_slotReleased"] = True
            self._condition.notify_all()

    def _persist_result_artifacts(self, job: Mapping[str, Any], result: GenerationResult) -> list[dict[str, Any]]:
        source_artifacts = list(result.artifacts or [])
        if not source_artifacts and result.image_data is not None:
            source_artifacts = [MediaArtifact(kind="image", data=result.image_data, filename="image.png", mime="image/png")]
        if len(source_artifacts) > MAX_ARTIFACTS_PER_JOB:
            raise RuntimeError(f"한 작업의 artifact는 최대 {MAX_ARTIFACTS_PER_JOB}개까지 저장할 수 있습니다")

        prepared = []
        total_bytes = 0
        for index, artifact in enumerate(source_artifacts):
            data = artifact.data
            source_path = None
            if data is None and artifact.path:
                candidate = Path(artifact.path)
                if candidate.is_file():
                    source_path = candidate
            if isinstance(data, (bytes, bytearray, memoryview)):
                artifact_size = len(data)
            elif source_path is not None:
                artifact_size = source_path.stat().st_size
            else:
                logger.warning("데이터 없는 artifact 건너뜀: job=%s index=%s", job["id"], index)
                continue
            if artifact_size < 0 or total_bytes + artifact_size > MAX_ARTIFACT_BYTES_PER_JOB:
                raise RuntimeError("한 작업의 artifact 총 용량은 2GiB 이하여야 합니다")
            total_bytes += artifact_size
            kind = str(artifact.kind or "file")[:40]
            if kind in {"image", "animated"}:
                mime = _detect_raster_mime(data, source_path)
                if not mime:
                    kind = "file"
                    mime = "application/octet-stream"
            else:
                mime = _normalise_mime(artifact.mime)
            raw_name = Path(str(artifact.filename or "")).name
            extension = Path(raw_name).suffix
            if not extension:
                extension = mimetypes.guess_extension(mime) or ".bin"
            stem = Path(raw_name).stem if raw_name else f"artifact_{index}"
            stem = _SAFE_FILENAME.sub("_", stem).strip("._")[:80] or f"artifact_{index}"
            extension = _SAFE_FILENAME.sub("", extension.lower())[:12] or ".bin"
            filename = f"{index:03d}_{stem}{extension}"
            prepared.append({
                "artifact": artifact,
                "data": data,
                "sourcePath": source_path,
                "size": artifact_size,
                "kind": kind,
                "mime": mime,
                "rawName": raw_name,
                "filename": filename,
            })

        target_dir = self.storage_root / str(job["id"])
        target_dir.mkdir(parents=True, exist_ok=True)
        persisted: list[dict[str, Any]] = []
        written_paths: list[Path] = []
        temporary_paths: list[Path] = []
        try:
            for item in prepared:
                target_path = target_dir / item["filename"]
                temporary_path = target_path.with_suffix(target_path.suffix + ".tmp")
                temporary_paths.append(temporary_path)
                if isinstance(item["data"], (bytes, bytearray, memoryview)):
                    temporary_path.write_bytes(bytes(item["data"]))
                else:
                    with item["sourcePath"].open("rb") as source, temporary_path.open("wb") as destination:
                        while True:
                            chunk = source.read(1024 * 1024)
                            if not chunk:
                                break
                            destination.write(chunk)
                os.replace(temporary_path, target_path)
                written_paths.append(target_path)
                artifact = item["artifact"]
                persisted.append({
                    "index": len(persisted),
                    "kind": item["kind"],
                    "file": item["filename"],
                    "filename": item["rawName"] or item["filename"],
                    "mime": item["mime"],
                    "size": item["size"],
                    "metadata": _redact_public_metadata(artifact.metadata or {}),
                })
        except Exception:
            for path in written_paths:
                try:
                    path.unlink(missing_ok=True)
                except OSError:
                    pass
            raise
        finally:
            # Only clean temporary artifacts created by this routine. The job
            # manifest writer may concurrently own ``job.json.tmp`` during a
            # cancellation transition.
            for temporary_path in temporary_paths:
                try:
                    temporary_path.unlink(missing_ok=True)
                except OSError:
                    pass
        return persisted

    def _discard_persisted_artifacts(self, job_id: str, artifacts: list[dict[str, Any]]) -> None:
        target_dir = (self.storage_root / str(job_id)).resolve()
        for item in artifacts:
            filename = Path(str(item.get("file") or "")).name
            path = (target_dir / filename).resolve()
            if filename and path.parent == target_dir:
                try:
                    path.unlink(missing_ok=True)
                except OSError:
                    logger.warning("취소된 generation artifact 정리 실패: %s", path)

    def _job_public(self, job: Mapping[str, Any], *, include_request: bool = True) -> dict[str, Any]:
        result = {
            key: copy.deepcopy(value)
            for key, value in job.items()
            if not str(key).startswith("_") and (include_request or key != "request")
        }
        result["artifacts"] = [
            {
                **{key: copy.deepcopy(value) for key, value in artifact.items() if key != "file"},
                "url": f"/api/v1/generations/{job['id']}/artifacts/{artifact['index']}",
            }
            for artifact in job.get("artifacts", [])
        ]
        return result

    def _persist_config(self) -> None:
        atomic_write_json(str(self.config_path), self._config, indent=2)

    def _persist_job(self, job: Mapping[str, Any]) -> None:
        try:
            target_dir = self.storage_root / str(job["id"])
            target_dir.mkdir(parents=True, exist_ok=True)
            public = self._job_public(job)
            # Artifact URLs are reconstructed, while the safe local filename is
            # required to serve results after an app restart.
            public["artifacts"] = copy.deepcopy(job.get("artifacts", []))
            atomic_write_json(str(target_dir / "job.json"), public, indent=2)
        except Exception:
            logger.exception("generation job manifest 저장 실패: %s", job.get("id"))

    def _load_manifests(self) -> None:
        if not self.storage_root.is_dir():
            return
        def modified_time(item: Path) -> float:
            try:
                return item.stat().st_mtime
            except OSError:
                return 0.0

        paths = sorted(self.storage_root.glob("*/job.json"), key=modified_time)[-MAX_RECENT_JOBS:]
        for path in paths:
            try:
                raw = load_json_safe(str(path), {})
                job_id = str(raw.get("id") or path.parent.name)
                if not re.fullmatch(r"[0-9a-f]{32}", job_id):
                    continue
                state = str(raw.get("state") or "failed")
                recovered_interrupted = state not in TERMINAL_STATES
                if recovered_interrupted:
                    state = "failed"
                    raw["error"] = "앱 재시작으로 이전 실행이 종료되었습니다"
                    raw["completedAt"] = _utc_now()
                artifacts = raw.get("artifacts") if isinstance(raw.get("artifacts"), list) else []
                safe_artifacts = []
                for item in artifacts:
                    if not isinstance(item, Mapping):
                        continue
                    filename = Path(str(item.get("file") or "")).name
                    if not filename:
                        continue
                    safe_artifacts.append({
                        "index": len(safe_artifacts),
                        "kind": str(item.get("kind") or "file")[:40],
                        "file": filename,
                        "filename": str(item.get("filename") or filename)[:300],
                        "mime": _normalise_mime(item.get("mime")),
                        "size": max(0, _safe_int(item.get("size"), 0)),
                        "metadata": _redact_public_metadata(item.get("metadata") or {}),
                    })
                target_id = str(raw.get("target") or "active")[:100]
                job = {
                    "id": job_id,
                    "state": state,
                    "target": target_id,
                    "mode": str(raw.get("mode") or "txt2img")[:40],
                    "family": str(raw.get("family") or "standard")[:40],
                    "model": str(raw.get("model") or "")[:500],
                    "createdAt": str(raw.get("createdAt") or "")[:100],
                    "startedAt": str(raw.get("startedAt") or "")[:100],
                    "completedAt": str(raw.get("completedAt") or "")[:100],
                    "progress": min(1.0, max(0.0, _safe_float(raw.get("progress"), 0.0))),
                    "currentStep": max(0, _safe_int(raw.get("currentStep"), 0)),
                    "totalSteps": max(0, _safe_int(raw.get("totalSteps"), 0)),
                    "cancelRequested": bool(raw.get("cancelRequested")),
                    "error": _sanitise_error(raw.get("error") or ""),
                    "info": _redact_public_metadata(raw.get("info") or {}),
                    "artifacts": safe_artifacts,
                    "request": _bounded_request_summary(raw.get("request") or {}),
                    "_payload": {},
                    # 복구된 작업은 위에서 모두 종료 상태가 되므로 직렬화 키를 읽는
                    # 경로(대기열 실행·취소 인터럽트)에 다시 들어가지 않는다. 여기서
                    # 프로필 → DNS 해석을 하면 GUI 스레드 시작이 실패한 조회 수만큼
                    # 멈추므로 대상 ID를 그대로 둔다.
                    "_serialKey": target_id,
                    "_slotReleased": True,
                }
                self._jobs[job_id] = job
                if recovered_interrupted:
                    self._persist_job(job)
            except Exception:
                # A single hand-edited or partially-written manifest must not
                # prevent the generation API manager from starting.
                logger.warning("손상된 generation job manifest를 건너뜁니다: %s", path, exc_info=True)

    def _trim_jobs_locked(self) -> None:
        while len(self._jobs) > MAX_RECENT_JOBS:
            removable_id = next(
                (
                    job_id
                    for job_id, job in self._jobs.items()
                    if job["state"] in TERMINAL_STATES and job.get("_slotReleased") is True
                ),
                None,
            )
            if removable_id is None:
                break
            self._jobs.pop(removable_id, None)
            self._job_profiles.pop(removable_id, None)
            self._job_backends.pop(removable_id, None)
            self._previews.pop(removable_id, None)

    def _resolve_profile(self, target_id: str) -> dict[str, Any]:
        value = str(target_id or "").strip()
        with self._lock:
            if value == "active":
                return {"id": "active", "name": "현재 앱 백엔드", "engine": "active", "type": "active"}
            for target in self._config.get("targets", []):
                if target["id"] == value:
                    return copy.deepcopy(target)
        raise GenerationValidationError("승인되지 않은 대상 프로필입니다")

    # -- HTTP server --------------------------------------------------------------

    def _start_server_locked(self) -> None:
        if self._server is not None:
            return
        with self._lock:
            host = str(self._config["bindHost"])
            port = int(self._config["port"])
        manager = self

        from core.port_probe import apply_exclusive_bind, exclusive_bind_supported

        class ApiServer(ThreadingHTTPServer):
            daemon_threads = True
            # Windows 의 SO_REUSEADDR 는 다른 SO_REUSEADDR 소켓과 같은 포트를 '공유'시킨다.
            # 그러면 관리형 Forge/ComfyUI 포트 선택이 이 서버의 포트를 비었다고 보거나,
            # 다른 프로세스가 같은 포트를 가로챌 수 있으므로 Windows 는 배타 bind 를 쓴다.
            allow_reuse_address = not exclusive_bind_supported()
            request_slots = threading.BoundedSemaphore(MAX_HTTP_CONNECTIONS)

            def server_bind(self):
                apply_exclusive_bind(self.socket)
                super().server_bind()

            def process_request(self, request, client_address):
                if not self.request_slots.acquire(blocking=False):
                    try:
                        request.shutdown(socket.SHUT_RDWR)
                    except OSError:
                        pass
                    request.close()
                    return
                try:
                    super().process_request(request, client_address)
                except Exception:
                    self.request_slots.release()
                    raise

            def process_request_thread(self, request, client_address):
                try:
                    super().process_request_thread(request, client_address)
                finally:
                    self.request_slots.release()

        class Handler(BaseHTTPRequestHandler):
            server_version = "AIStudioGenerationAPI/1"
            sys_version = ""

            def setup(self) -> None:
                super().setup()
                self.connection.settimeout(HTTP_SOCKET_TIMEOUT_SECONDS)

            def log_message(self, format_string: str, *args: Any) -> None:
                logger.debug("generation API HTTP: " + format_string, *args)

            def _headers(self, content_type: str, length: int) -> None:
                self.send_header("Content-Type", content_type)
                self.send_header("Content-Length", str(length))
                self.send_header("Cache-Control", "no-store")
                self.send_header("X-Content-Type-Options", "nosniff")
                self.send_header("Referrer-Policy", "no-referrer")

            def _json(self, status: int, value: Any) -> None:
                data = json.dumps(
                    value,
                    ensure_ascii=False,
                    separators=(",", ":"),
                    allow_nan=False,
                ).encode("utf-8")
                self.send_response(int(status))
                self._headers("application/json; charset=utf-8", len(data))
                self.end_headers()
                self.wfile.write(data)

            def _error(self, error: Exception) -> None:
                status = int(getattr(error, "status", HTTPStatus.INTERNAL_SERVER_ERROR))
                if status >= 500:
                    logger.exception("generation API HTTP 처리 실패", exc_info=error)
                if isinstance(error, GenerationApiError):
                    message = _sanitise_error(str(error)) or HTTPStatus(status).phrase
                else:
                    message = "내부 서버 오류"
                self._json(status, {"error": message})

            def _authenticated(self) -> bool:
                with manager._lock:
                    expected = str(manager._config["token"])
                header = str(self.headers.get("Authorization") or "")
                supplied = header[7:].strip() if header.lower().startswith("bearer ") else ""
                if supplied and hmac.compare_digest(supplied, expected):
                    return True
                self.send_response(HTTPStatus.UNAUTHORIZED)
                self.send_header("WWW-Authenticate", 'Bearer realm="AI Studio Generation API"')
                data = b'{"error":"Bearer token required"}'
                self._headers("application/json; charset=utf-8", len(data))
                self.end_headers()
                self.wfile.write(data)
                return False

            def _read_json(self) -> dict[str, Any]:
                length_header = self.headers.get("Content-Length")
                if length_header is None:
                    raise GenerationValidationError("Content-Length가 필요합니다")
                try:
                    length = int(length_header)
                except ValueError as exc:
                    raise GenerationValidationError("Content-Length가 올바르지 않습니다") from exc
                if length < 0 or length > manager.max_body_bytes:
                    error = GenerationValidationError("요청 본문이 허용 크기를 초과했습니다")
                    error.status = HTTPStatus.REQUEST_ENTITY_TOO_LARGE
                    raise error
                body = self.rfile.read(length)
                try:
                    value = json.loads(
                        body.decode("utf-8"),
                        parse_constant=lambda constant: (_ for _ in ()).throw(
                            ValueError(f"invalid JSON constant: {constant}")
                        ),
                    ) if body else {}
                except (UnicodeDecodeError, json.JSONDecodeError, ValueError, RecursionError) as exc:
                    raise GenerationValidationError("요청 본문이 올바른 JSON이 아닙니다") from exc
                if not isinstance(value, dict):
                    raise GenerationValidationError("요청 JSON 루트는 객체여야 합니다")
                _validate_json_text(value)
                return value

            def do_OPTIONS(self) -> None:  # noqa: N802
                # Deliberately omit CORS headers.  Bearer-authenticated browser
                # clients must be same-origin or use a trusted native client.
                self.send_response(HTTPStatus.NO_CONTENT)
                self.send_header("Allow", "GET, POST, DELETE, OPTIONS")
                self.end_headers()

            def do_GET(self) -> None:  # noqa: N802
                parsed = urlparse(self.path)
                if parsed.path == "/api/v1/health":
                    self._json(HTTPStatus.OK, {"status": "ok", "service": "ai-studio-generation-api"})
                    return
                if not self._authenticated():
                    return
                try:
                    if parsed.path == "/api/v1/targets":
                        state = manager.snapshot(False)
                        targets = [{"id": "active", "name": "현재 앱 백엔드", "engine": "active"}]
                        targets.extend({
                            "id": target["id"],
                            "name": target["name"],
                            "engine": target["engine"],
                            "enabled": target.get("enabled", True),
                        } for target in state["config"]["targets"])
                        self._json(HTTPStatus.OK, {"defaultTarget": state["config"]["defaultTarget"], "targets": targets})
                        return
                    if parsed.path == "/api/v1/generations":
                        query = parse_qs(parsed.query)
                        self._json(HTTPStatus.OK, {"jobs": manager.list_jobs(query.get("limit", [50])[0])})
                        return
                    if parsed.path == "/sdapi/v1/progress":
                        target = str(self.headers.get("X-AIStudio-Target") or manager._config["defaultTarget"])
                        self._json(HTTPStatus.OK, manager._a1111_progress(target))
                        return
                    if parsed.path == "/sdapi/v1/options":
                        self._json(HTTPStatus.OK, {"sd_model_checkpoint": "", "ai_studio_target": manager._config["defaultTarget"]})
                        return
                    match = re.fullmatch(r"/api/v1/generations/([0-9a-f]{32})", parsed.path)
                    if match:
                        self._json(HTTPStatus.OK, manager.inspect(match.group(1)))
                        return
                    match = re.fullmatch(r"/api/v1/generations/([0-9a-f]{32})/artifacts/(\d+)", parsed.path)
                    if match:
                        path, mime, filename = manager.artifact_path(match.group(1), int(match.group(2)))
                        length = path.stat().st_size
                        self.send_response(HTTPStatus.OK)
                        self._headers(mime, length)
                        disposition = "inline" if mime in _INLINE_ARTIFACT_MIMES else "attachment"
                        self.send_header("Content-Disposition", f'{disposition}; filename="{filename}"')
                        self.end_headers()
                        try:
                            with path.open("rb") as handle:
                                while True:
                                    chunk = handle.read(1024 * 1024)
                                    if not chunk:
                                        break
                                    self.wfile.write(chunk)
                        except (BrokenPipeError, ConnectionResetError, socket.timeout):
                            pass
                        return
                    raise GenerationNotFoundError("API endpoint를 찾을 수 없습니다")
                except Exception as exc:
                    self._error(exc)

            def do_POST(self) -> None:  # noqa: N802
                parsed = urlparse(self.path)
                if not self._authenticated():
                    return
                try:
                    if parsed.path in {"/api/v1/generations", "/api/v1/generate"}:
                        # 본문을 먼저 읽는다. 읽지 않은 본문을 남긴 채 400 을 보내고 소켓을 닫으면
                        # Windows 가 RST 를 보내 클라이언트가 응답 대신 연결 중단(10053)을 받는다.
                        # 작업 제출은 아래 검증 뒤라 잘못된 wait 는 여전히 아무 작업도 만들지 않는다.
                        request = self._read_json()
                        wait_value = parse_qs(parsed.query).get("wait", ["0"])[0]
                        try:
                            wait_seconds = float(wait_value)
                        except (TypeError, ValueError, OverflowError) as exc:
                            raise GenerationValidationError("wait 쿼리는 0~600초의 유한한 숫자여야 합니다") from exc
                        if not math.isfinite(wait_seconds) or not 0 <= wait_seconds <= 600:
                            raise GenerationValidationError("wait 쿼리는 0~600초의 유한한 숫자여야 합니다")
                        job = manager.submit(request)
                        if wait_seconds:
                            job = manager.wait(job["id"], wait_seconds)
                        status = HTTPStatus.OK if job["state"] in TERMINAL_STATES else HTTPStatus.ACCEPTED
                        self._json(status, job)
                        return
                    if parsed.path in {"/sdapi/v1/txt2img", "/sdapi/v1/img2img"}:
                        payload = self._read_json()
                        body_target = str(payload.pop("ai_studio_target", "") or "").strip()
                        target = str(self.headers.get("X-AIStudio-Target") or body_target or manager._config["defaultTarget"])
                        family = str(payload.pop("generation_family", payload.pop("request_family", "standard")) or "standard")
                        job = manager.submit({
                            "target": target,
                            "mode": "txt2img" if parsed.path.endswith("txt2img") else "img2img",
                            "model": str(payload.get("override_settings", {}).get("sd_model_checkpoint", ""))
                            if isinstance(payload.get("override_settings"), Mapping) else "",
                            "family": family,
                            "payload": payload,
                        })
                        finished = manager.wait(job["id"], 600)
                        if finished["state"] not in TERMINAL_STATES:
                            manager.cancel(job["id"])
                            error = GenerationConflictError(
                                f"생성 대기 시간이 600초를 초과해 취소를 요청했습니다 (job: {job['id']})"
                            )
                            error.status = HTTPStatus.GATEWAY_TIMEOUT
                            raise error
                        if finished["state"] != "completed":
                            error = GenerationConflictError(finished.get("error") or "생성 실패")
                            error.status = HTTPStatus.BAD_GATEWAY
                            raise error
                        image_artifacts: list[tuple[Path, str, str]] = []
                        response_bytes = 0
                        for artifact_info in finished.get("artifacts", []):
                            if artifact_info.get("kind") not in {"image", "animated"}:
                                continue
                            path, mime, filename = manager.artifact_path(
                                job["id"], artifact_info["index"]
                            )
                            response_bytes += path.stat().st_size
                            if response_bytes > MAX_A1111_RESPONSE_BYTES:
                                error = GenerationConflictError(
                                    "A1111 호환 응답 이미지 총 용량이 128MiB를 초과했습니다; native artifact API를 사용하세요"
                                )
                                error.status = HTTPStatus.BAD_GATEWAY
                                raise error
                            image_artifacts.append((path, mime, filename))
                        images = []
                        for path, _mime, _filename in image_artifacts:
                            data = path.read_bytes()
                            images.append(base64.b64encode(data).decode("ascii"))
                        if not images:
                            error = GenerationConflictError("백엔드가 이미지 결과를 반환하지 않았습니다")
                            error.status = HTTPStatus.BAD_GATEWAY
                            raise error
                        self._json(HTTPStatus.OK, {
                            "images": images,
                            "parameters": payload,
                            "info": json.dumps(finished.get("info") or {}, ensure_ascii=False),
                        })
                        return
                    if parsed.path == "/sdapi/v1/interrupt":
                        # Read and discard an optional empty JSON object without
                        # making Content-Length mandatory for compatibility.
                        target = str(self.headers.get("X-AIStudio-Target") or manager._config["defaultTarget"])
                        self._json(HTTPStatus.OK, {"cancelled": manager.cancel_target(target)})
                        return
                    raise GenerationNotFoundError("API endpoint를 찾을 수 없습니다")
                except Exception as exc:
                    self._error(exc)

            def do_DELETE(self) -> None:  # noqa: N802
                if not self._authenticated():
                    return
                try:
                    parsed = urlparse(self.path)
                    match = re.fullmatch(r"/api/v1/generations/([0-9a-f]{32})", parsed.path)
                    if not match:
                        raise GenerationNotFoundError("API endpoint를 찾을 수 없습니다")
                    self._json(HTTPStatus.OK, manager.cancel(match.group(1)))
                except Exception as exc:
                    self._error(exc)

        server = ApiServer((host, port), Handler)
        thread = threading.Thread(
            target=server.serve_forever,
            kwargs={"poll_interval": 0.25},
            name="generation-api-http",
            daemon=True,
        )
        self._server = server
        self._server_thread = thread
        thread.start()
        logger.info("generation API server started: %s:%s", host, port)

    def _stop_server_locked(self) -> None:
        server = self._server
        thread = self._server_thread
        self._server = None
        self._server_thread = None
        if server is None:
            return
        server.shutdown()
        server.server_close()
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout=2)
        logger.info("generation API server stopped")

    def _a1111_progress(self, target_id: str) -> dict[str, Any]:
        with self._lock:
            jobs = [job for job in self._jobs.values() if job["target"] == target_id]
            current = next((job for job in reversed(jobs) if job["state"] == "running"), None)
            if current is None:
                current = next((job for job in reversed(jobs) if job["state"] == "queued"), None)
            if current is None:
                return {"progress": 0.0, "eta_relative": 0.0, "state": {"job": ""}, "current_image": ""}
            preview = self._previews.get(current["id"])
            return {
                "progress": float(current["progress"]),
                "eta_relative": 0.0,
                "state": {
                    "job": current["id"],
                    "job_count": self._active_slots,
                    "sampling_step": current["currentStep"],
                    "sampling_steps": current["totalSteps"],
                },
                "current_image": base64.b64encode(preview).decode("ascii") if preview else "",
            }


_MANAGER: Optional[GenerationApiManager] = None
_MANAGER_LOCK = threading.Lock()


def get_generation_api_manager() -> GenerationApiManager:
    """Return the process-wide generation API gateway instance."""

    global _MANAGER
    with _MANAGER_LOCK:
        if _MANAGER is None:
            _MANAGER = GenerationApiManager()
        return _MANAGER


__all__ = [
    "GenerationApiManager",
    "GenerationApiError",
    "GenerationValidationError",
    "GenerationNotFoundError",
    "GenerationQueueFullError",
    "GenerationConflictError",
    "get_generation_api_manager",
]
