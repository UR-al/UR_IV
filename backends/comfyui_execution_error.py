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
    removed). An exception raised without text (a bare ``assert``, ``RuntimeError()``) gives an empty message:
    then the short exception type and the failing frame stand in for it; ``UNKNOWN_ERROR`` when the event carries
    none of them."""
    info = data if isinstance(data, Mapping) else {}
    raw = info.get("exception_message")
    text = raw.strip() if isinstance(raw, str) else ("" if raw is None else str(raw).strip())
    if text:
        return text
    kind = info.get("exception_type")
    kind = str(kind).rsplit(".", 1)[-1].strip() if kind not in (None, "") else ""
    where = _failing_frame(info.get("traceback"))
    if kind and where:
        return f"{kind} — {where}"
    return kind or where or UNKNOWN_ERROR


def _failing_frame(raw: Any) -> str:
    """``model_patcher.py:1766 (load): assert load_device is not None`` from the last ``format_tb`` entry."""
    if isinstance(raw, (list, tuple)) and raw:
        last = str(raw[-1])
    elif isinstance(raw, str) and raw.strip():
        last = raw.rstrip().rsplit('  File "', 1)[-1]
        last = '  File "' + last if not last.startswith('  File "') else last
    else:
        return ""
    lines = [line.strip() for line in last.strip().splitlines() if line.strip()]
    if not lines:
        return ""
    head, code = lines[0], (lines[1] if len(lines) > 1 else "")
    if head.startswith('File "') and '", line ' in head:
        path, rest = head[len('File "'):].split('", line ', 1)
        line_no, _, func = rest.partition(", in ")
        name = path.replace("\\", "/").rsplit("/", 1)[-1]
        head = f"{name}:{line_no.strip()}" + (f" ({func.strip()})" if func.strip() else "")
    return f"{head}: {code}" if code else head


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
