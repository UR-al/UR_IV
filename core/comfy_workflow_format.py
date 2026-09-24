"""Only ComfyUI API-format workflows are accepted by the app.

The editor's "Save" format (``{"nodes": [...], "links": [...]}``) carries
widget values positionally, keeps muted/bypassed nodes and UI-only nodes
(Note, Reroute, PrimitiveNode) and needs ComfyUI's own ``graphToPrompt`` to
become executable.  Re-implementing that conversion produced graphs that the
picker called valid but ComfyUI rejected, so every entry point (workflow
picker, generation, Generation API profiles) refuses the web format with the
same instruction instead.
"""
from __future__ import annotations

from typing import Any

WEB_WORKFLOW_MESSAGE = (
    "ComfyUI 편집기 저장 형식(웹 포맷) 워크플로는 실행할 수 없습니다. "
    "ComfyUI 메뉴 Workflow → Export (API)로 저장한 JSON 파일을 선택하세요."
)


class WorkflowFormatError(RuntimeError, ValueError):
    """The workflow JSON is not a ComfyUI API-format prompt graph.

    A RuntimeError for the backend/Generation API paths (which report it as a
    failed generation) and a ValueError for the queue snapshot paths, which
    already turn WorkflowControlError/ValueError into a user message.
    """


def is_web_workflow(data: Any) -> bool:
    return isinstance(data, dict) and isinstance(data.get("nodes"), list)


def require_api_workflow(data: Any) -> dict:
    """Return ``data`` when it is an API-format graph, else raise."""
    if is_web_workflow(data):
        raise WorkflowFormatError(WEB_WORKFLOW_MESSAGE)
    if not isinstance(data, dict):
        raise WorkflowFormatError("ComfyUI 워크플로 JSON 최상위 값은 객체여야 합니다.")
    return data


__all__ = [
    "WEB_WORKFLOW_MESSAGE", "WorkflowFormatError",
    "is_web_workflow", "require_api_workflow",
]
