"""대화 탭 — Ollama 스트리밍 응답을 Qt 스레드에서 받아 토큰 단위로 올린다.

토큰마다 시그널을 쏘면 QWebChannel 이 초당 수십 번 깨어난다 — 그 정도는 문제없지만,
빠른 모델은 초당 수백 조각을 내므로 40ms 단위로 모아서 보낸다. 사용자 눈엔 같은
'타자 치는' 속도이고 브리지는 훨씬 덜 바쁘다.
"""
from __future__ import annotations

import json
import threading
import time
from typing import Any

from PyQt6.QtCore import QThread, pyqtSignal

from core.chat_store import inline_image_paths
from core.ollama_client import OllamaClient
from core.lmstudio_client import LMStudioClient
from core.structured_output import validate_output

#: 토큰을 모아 보내는 간격(초).
FLUSH_INTERVAL = 0.04


class ChatWorker(QThread):
    #: JSON {id, text?, thinking?} — 모아 보낸 조각 (text = 답, thinking = 생각)
    token = pyqtSignal(str)
    #: JSON {id, ok, content, thinking, stopped, error?, evalCount?, durationMs?}
    done = pyqtSignal(str)

    def __init__(self, request_id: str, base_url: str, model: str,
                 messages: list[dict[str, Any]], options: dict[str, Any] | None = None,
                 think: bool | str | None = None, parent=None, *, provider='ollama', schema=None):
        super().__init__(parent)
        self.request_id = request_id
        self.base_url = base_url
        self.model = model
        self.messages = messages
        self.options = options or {}
        self.think = think
        self.provider = provider
        self.schema = schema
        self._stop = threading.Event()

    def stop(self) -> None:
        self._stop.set()

    def run(self) -> None:
        client_type = LMStudioClient if self.provider == 'lmstudio' else OllamaClient
        buffer: list[str] = []
        thoughts: list[str] = []
        last_flush = time.monotonic()

        def flush() -> None:
            nonlocal last_flush
            if buffer or thoughts:
                packet: dict[str, Any] = {"id": self.request_id}
                if thoughts:
                    packet["thinking"] = "".join(thoughts)
                    thoughts.clear()
                if buffer:
                    packet["text"] = "".join(buffer)
                    buffer.clear()
                self.token.emit(json.dumps(packet, ensure_ascii=False))
            last_flush = time.monotonic()

        def on_token(piece: str) -> None:
            buffer.append(piece)
            if time.monotonic() - last_flush >= FLUSH_INTERVAL:
                flush()

        def on_thinking(piece: str) -> None:
            thoughts.append(piece)
            if time.monotonic() - last_flush >= FLUSH_INTERVAL:
                flush()

        started = time.monotonic()
        try:
            # 경로 이미지 → base64 는 여기(워커 스레드)서, 이미 최근 턴만 남은 메시지에만 한다.
            # 역할·본문은 그대로 두고 경로 이미지만 바꾼다(맨 base64 는 looks_like_image_path 가 거른다).
            self.messages = inline_image_paths(self.messages)
            client = client_type(base_url=self.base_url, model=self.model)
            kwargs = {'schema': self.schema} if self.schema is not None else {}
            result = client.chat_stream(
                self.messages, model=self.model, options=self.options, think=self.think,
                on_token=on_token, on_thinking=on_thinking, should_stop=self._stop.is_set, **kwargs,
            )
            flush()
            validation_error = ''
            if self.schema is not None:
                if result.get('stopped') or result.get('done_reason') not in ('stop', None):
                    validation_error = 'JSON 응답이 중지되거나 완료되지 않았습니다. 형식을 보장할 수 없습니다. 받은 내용은 유지됩니다.'
                else:
                    try:
                        validate_output(result.get('content', ''), self.schema)
                    except ValueError as exc:
                        validation_error = str(exc)
            self.done.emit(json.dumps({
                "id": self.request_id, "ok": not bool(validation_error),
                **({'error': validation_error} if validation_error else {}),
                "structured": self.schema is not None,
                "content": result.get("content", ""),
                "thinking": result.get("thinking", ""),
                "stopped": bool(result.get("stopped")),
                "evalCount": result.get("eval_count"),
                "doneReason": result.get("done_reason"),
                "durationMs": int((time.monotonic() - started) * 1000),
            }, ensure_ascii=False))
        except Exception as exc:  # noqa: BLE001 — 사용자에게 그대로 보여 준다
            flush()
            self.done.emit(json.dumps({
                "id": self.request_id, "ok": False,
                "content": "", "stopped": self._stop.is_set(),
                "error": str(exc)[:2000],
            }, ensure_ascii=False))
