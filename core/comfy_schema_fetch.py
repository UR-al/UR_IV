"""Bounded, cancellable ``/object_info`` read for possibly remote ComfyUI targets.

Generation API profiles may name a remote ComfyUI.  Their schema read keeps the
same safety rules as ``core/comfy_compatibility._fetch_json`` (no redirects, a
size cap) instead of the local backend's retrying, uncapped
``get_object_info``, and it stays responsive to the job's cancel: one attempt
with short connect/read timeouts, ``cancel_check`` polled between chunks.
"""
from __future__ import annotations

import json
from typing import Any, Callable, Optional

import requests

# The full schema was measured at 1.8-9 MB; comfy_compatibility uses the same cap.
MAX_OBJECT_INFO_BYTES = 32 * 1024 * 1024
# (connect, read-gap) seconds: an unreachable target fails in seconds, a server
# still building a large schema gets time for its first byte.
DEFAULT_TIMEOUT = (5.0, 20.0)
_CHUNK_BYTES = 256 * 1024


class SchemaFetchCancelled(RuntimeError):
    """The job was cancelled while its ComfyUI schema was being read."""


def _check_cancel(cancel_check: Optional[Callable[[], bool]]) -> None:
    if cancel_check is not None and cancel_check():
        raise SchemaFetchCancelled("사용자가 작업을 취소했습니다.")


def fetch_object_info_bounded(
    api_url: str,
    *,
    cancel_check: Optional[Callable[[], bool]] = None,
    max_bytes: int = MAX_OBJECT_INFO_BYTES,
    timeout: Any = DEFAULT_TIMEOUT,
    get: Callable[..., Any] = requests.get,
) -> dict:
    """Return the target's ``/object_info`` document.

    Raises ``SchemaFetchCancelled`` when ``cancel_check`` turns true, and
    ``RuntimeError`` for a redirect, a non-200 status, a body over
    ``max_bytes`` or a body that is not a JSON object.  Network errors from
    ``requests`` propagate unchanged.
    """
    _check_cancel(cancel_check)
    url = str(api_url or "").rstrip("/") + "/object_info"
    chunks: list[bytes] = []
    total = 0
    with get(url, timeout=timeout, stream=True, allow_redirects=False) as response:
        status = int(getattr(response, "status_code", 0) or 0)
        if 300 <= status < 400:
            raise RuntimeError(
                f"ComfyUI /object_info가 다른 주소로 리디렉션했습니다(HTTP {status}). "
                "대상 프로필 URL을 실제 ComfyUI 주소로 고치세요."
            )
        if status != 200:
            raise RuntimeError(f"ComfyUI /object_info 응답 오류: HTTP {status}")
        headers = getattr(response, "headers", None) or {}
        try:
            declared = int(headers.get("Content-Length", "0") or 0)
        except (TypeError, ValueError):
            declared = 0
        if declared > max_bytes:
            raise RuntimeError("ComfyUI /object_info 응답이 허용 크기(32MB)를 넘습니다.")
        for chunk in response.iter_content(_CHUNK_BYTES):
            _check_cancel(cancel_check)
            if not chunk:
                continue
            total += len(chunk)
            if total > max_bytes:
                raise RuntimeError("ComfyUI /object_info 응답이 허용 크기(32MB)를 넘습니다.")
            chunks.append(chunk)
    _check_cancel(cancel_check)
    try:
        data = json.loads(b"".join(chunks))
    except ValueError as exc:
        raise RuntimeError("ComfyUI /object_info 응답이 JSON이 아닙니다.") from exc
    if not isinstance(data, dict):
        raise RuntimeError("ComfyUI /object_info 응답이 객체가 아닙니다.")
    return data


__all__ = [
    "DEFAULT_TIMEOUT", "MAX_OBJECT_INFO_BYTES", "SchemaFetchCancelled",
    "fetch_object_info_bounded",
]
