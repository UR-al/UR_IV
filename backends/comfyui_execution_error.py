"""ComfyUI ``execution_error`` websocket event -> what the app shows and what it logs.

ComfyUI (0.36, ``execution.py`` ``handle_execution_error``) sends ``{"prompt_id", "node_id", "node_type",
"executed", "exception_message", "exception_type", "traceback", "current_inputs", "current_outputs",
"timestamp"}``: ``exception_message`` is ``"{exception}\\n{tips}"`` (tips empty unless out of memory or a CLIP
shape clash), ``traceback`` the ``traceback.format_tb`` list. The generation result carries the message only;
the traceback belongs in the app log (comfy-3 O4: a pack error such as the flat custom sigma list showed three
traceback frames under its own message).
"""
from __future__ import annotations

from typing import Any, Mapping

UNKNOWN_ERROR = "알 수 없는 오류"


def execution_error_message(data: Any) -> str:
    """The exception message of an ``execution_error`` event (ComfyUI's tips included, outer blank lines
    removed); ``UNKNOWN_ERROR`` when the event carries none."""
    raw = data.get("exception_message") if isinstance(data, Mapping) else None
    text = raw.strip() if isinstance(raw, str) else ("" if raw is None else str(raw).strip())
    return text or UNKNOWN_ERROR


def execution_error_traceback(data: Any) -> str:
    """The event's whole traceback as one text (``format_tb`` entries joined; a plain string kept), '' without."""
    raw = data.get("traceback") if isinstance(data, Mapping) else None
    if isinstance(raw, str):
        return raw.rstrip()
    if isinstance(raw, (list, tuple)):
        return "".join(str(line) for line in raw).rstrip()
    return ""


def execution_error_log(data: Any) -> str:
    """The app-log record: the message, the failing node and exception type, then the whole traceback."""
    message = execution_error_message(data)
    info = data if isinstance(data, Mapping) else {}
    where = ", ".join(
        f"{label} {info.get(key)}" for key, label in (("node_id", "node"), ("node_type", "type"),
                                                       ("exception_type", "exception"))
        if info.get(key) not in (None, "")
    )
    trace = execution_error_traceback(data)
    text = f"{message} ({where})" if where else message
    return f"{text}\nTraceback (ComfyUI):\n{trace}" if trace else text
