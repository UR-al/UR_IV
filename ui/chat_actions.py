"""대화 탭 액션 — Vue `requestAction('chat_*')` 의 Python 쪽.

GeneratorMainUI 에 믹스인으로 얹힌다(`ui/creator_actions.py` 와 같은 방식).
브리지 계약: tests/test_bridge_contract.py 가 AST 로 아래 `action in (...)` 의 이름을
읽어 frontend 의 requestAction 과 양방향 대조한다 — 비교 대상은 리터럴이나 모듈·클래스 상수,
함수 안 지역 표로 둘 것(정적으로 못 읽는 식이면 그 검사가 실패한다).

시그널(ui/vue_bridge.py): chatToken {id,text} · chatDone {id,ok,content,stopped,error?} ·
chatThreads [threads].
"""
from __future__ import annotations

import json
import uuid
from typing import Any

import os
import re
import threading
from dataclasses import replace
from PyQt6.QtCore import QObject, pyqtSignal, Qt

from core.chat_store import ChatStore, build_ollama_messages, clean_options
from core.chat_generation import MediaGenerationJob, plan_chat_generation
from workers.chat_worker import ChatWorker
from core.ollama_client import DEFAULT_OLLAMA_URL


class _ChatMediaDispatch(QObject):
    prepared = pyqtSignal(object, object)


class ChatActionsMixin:
    _chat_worker: ChatWorker | None = None
    _chat_store: ChatStore | None = None

    def _ensure_chat_media(self):
        if hasattr(self, '_chat_media_dispatch'):
            return
        self._chat_media_job = None
        self._chat_creator_started = False
        self._chat_media_dispatch = _ChatMediaDispatch(self)
        self._chat_media_dispatch.prepared.connect(self._chat_dispatch_creator, Qt.ConnectionType.QueuedConnection)
        self.vue_bridge.chatGenerationEvent.connect(self._chat_media_finished)
        self.vue_bridge.creatorProgress.connect(self._chat_creator_progress)
        self.vue_bridge.creatorResult.connect(self._chat_creator_result)

    def _chat_emit_media(self, event):
        self.vue_bridge.chatGenerationEvent.emit(json.dumps(event, ensure_ascii=False))

    def _chat_reject(self, request_id, error):
        self.vue_bridge.chatDone.emit(json.dumps({'id': request_id, 'ok': False, 'content': '',
                                                'stopped': False, 'error': str(error)[:2000]}, ensure_ascii=False))

    def _handle_chat_action(self, action: str, payload: dict) -> bool:
        if action in ("chat_send", "chat_stop", "chat_load", "chat_save", "chat_export", "chat_model_info", "chat_models"):
            pass
        else:
            return False
        handlers = {
            "chat_send": self._chat_send,
            "chat_stop": self._chat_stop,
            "chat_load": self._chat_load,
            "chat_save": self._chat_save,
            "chat_export": self._chat_export,
            "chat_model_info": self._chat_model_info,
            "chat_models": self._chat_models,
        }
        handlers[action](payload or {})
        return True

    def _chat_model_info(self, payload):
        """Coalesce rapid model changes into one read-only metadata worker."""
        if not hasattr(self, '_chat_info_lock'):
            self._chat_info_lock = threading.Lock()
            self._chat_info_running = False
        request = (str(payload.get('id') or '')[:100],
                   str(payload.get('url') or DEFAULT_OLLAMA_URL)[:2000],
                   str(payload.get('model') or '')[:300], str(payload.get('provider') or 'ollama'))
        with self._chat_info_lock:
            self._chat_info_pending = request
            if self._chat_info_running:
                return
            self._chat_info_running = True

        def work():
            from core.ollama_client import OllamaClient
            from core.lmstudio_client import LMStudioClient
            while True:
                with self._chat_info_lock:
                    current = self._chat_info_pending
                    self._chat_info_pending = None
                    if current is None:
                        self._chat_info_running = False
                        return
                request_id, url, model, provider = current
                event = {'id': request_id, 'model': model}
                try:
                    if not model:
                        raise ValueError('모델을 선택해 주세요')
                    if provider not in ('ollama', 'lmstudio'):
                        raise ValueError('지원하지 않는 대화 서버입니다')
                    client = LMStudioClient if provider == 'lmstudio' else OllamaClient
                    event.update(ok=True, info=client(url, model).get_model_info())
                except Exception as exc:
                    event.update(ok=False, error=str(exc)[:500])
                try:
                    self.vue_bridge.chatModelInfo.emit(json.dumps(event, ensure_ascii=False))
                except RuntimeError:
                    return  # window was destroyed while metadata HTTP finished
        threading.Thread(target=work, daemon=True, name='chat-model-info').start()

    def _chat_models(self, payload):
        """LM Studio model discovery, bounded to one coalescing worker."""
        if not hasattr(self, '_chat_models_lock'):
            self._chat_models_lock = threading.Lock()
            self._chat_models_running = False
        with self._chat_models_lock:
            self._chat_models_pending = (str(payload.get('id') or '')[:100],
                                         str(payload.get('url') or 'http://localhost:1234')[:2000])
            if self._chat_models_running:
                return
            self._chat_models_running = True

        def work():
            from core.lmstudio_client import LMStudioClient
            while True:
                with self._chat_models_lock:
                    current = self._chat_models_pending
                    self._chat_models_pending = None
                    if current is None:
                        self._chat_models_running = False
                        return
                request_id, url = current
                event = {'id': request_id}
                try:
                    event.update(ok=True, models=LMStudioClient(url).list_models())
                except Exception as exc:
                    event.update(ok=False, error=str(exc)[:500])
                try:
                    self.vue_bridge.chatModelsReady.emit(json.dumps(event, ensure_ascii=False))
                except RuntimeError:
                    return
        threading.Thread(target=work, daemon=True, name='chat-models').start()

    # ── 저장 ──
    def _chat_store_instance(self) -> ChatStore:
        if self._chat_store is None:
            self._chat_store = ChatStore()
        return self._chat_store

    def _chat_load(self, _payload: dict) -> None:
        threads = self._chat_store_instance().load()
        self.vue_bridge.chatThreads.emit(json.dumps(threads, ensure_ascii=False))

    def _chat_save(self, payload: dict) -> None:
        self._chat_store_instance().save(payload.get("threads") or [])

    def _chat_export(self, payload: dict) -> None:
        """대화 하나를 Markdown 파일로 — 본문은 Vue 가 만들고 여기서는 저장 대화상자만."""
        from PyQt6.QtWidgets import QFileDialog
        title = re.sub(r'[\\/:*?"<>|]+', ' ', str(payload.get("title") or "")).strip() or "대화"
        path, _ = QFileDialog.getSaveFileName(self, "대화 내보내기", f"{title}.md", "Markdown (*.md)")
        if not path:
            return
        try:
            with open(path, "w", encoding="utf-8") as fh:
                fh.write(str(payload.get("markdown") or ""))
            self.vue_bridge.showNotification.emit("success", f"내보냄: {os.path.basename(path)}")
        except OSError as exc:
            self.vue_bridge.showNotification.emit("error", f"내보내기 실패: {exc}")

    # ── 대화 ──
    def _chat_send(self, payload: dict) -> None:
        request_id = str(payload.get("id") or uuid.uuid4().hex)
        if len(request_id) > 100:
            self._chat_reject(request_id, '요청 ID가 너무 깁니다')
            return
        self._ensure_chat_media()
        if self._chat_media_job is not None:
            if request_id != self._chat_media_job.id:
                self._chat_reject(request_id, '채팅 생성이 진행 중입니다. 완료 또는 중지 후 다시 시도해 주세요')
            return
        if self._chat_worker is not None and self._chat_worker.isRunning():
            if request_id != self._chat_worker.request_id:
                self._chat_reject(request_id, '채팅 응답이 진행 중입니다. 중지 후 다시 시도해 주세요')
            return
        try:
            plan = plan_chat_generation(payload)
            if plan is not None:
                self._chat_start_media(request_id, plan)
                return
        except Exception as exc:
            self._chat_reject(request_id, exc)
            return
        provider = payload.get('provider') or 'ollama'
        try:
            if provider not in ('ollama', 'lmstudio'):
                raise ValueError('지원하지 않는 대화 서버입니다')
            from core.structured_output import parse_schema
            schema = parse_schema(payload['schema']) if payload.get('schema') is not None else None
        except (ValueError, TypeError) as exc:
            self._chat_reject(request_id, exc)
            return
        url = str(payload.get("url") or "").strip() or ('http://localhost:1234' if provider == 'lmstudio' else DEFAULT_OLLAMA_URL)
        model = str(payload.get("model") or "").strip()
        if not model:
            self.vue_bridge.chatDone.emit(json.dumps({
                "id": request_id, "ok": False, "content": "", "stopped": False,
                "error": "모델이 선택되지 않았습니다 — 대화 설정에서 서버 연결을 확인하고 모델을 고르세요",
            }, ensure_ascii=False))
            return
        # GUI 스레드에서는 최근 턴만 남기는 일만 한다. 경로 이미지(히스토리·갤러리 드롭, 최대 20MB)
        # 읽기와 base64 인코딩은 ChatWorker.run 이 잘라 낸 뒤의 메시지에만 한다 — 잘려 나갈 옛 턴의
        # 파일까지 매번 읽으며 onAction(동기 슬롯)을 막지 않게.
        messages = build_ollama_messages(
            payload.get("messages") or [],
            system_prompt=str(payload.get("system") or ""),
        )
        options: dict[str, Any] = clean_options(payload.get("options"))
        # 한 번에 하나 — 새 요청이 오면 이전 것을 멈춘다. 둘이 동시에 흐르면 토큰이 섞인다.
        # thinking 모델은 기본으로 생각부터 한다 — 빠른 답변이 기본, '깊은 추론' 을 켰을 때만 생각을 흘린다
        think = payload.get("think")
        if not isinstance(think, bool) and think not in (None, 'low', 'medium', 'high', 'max'):
            self._chat_reject(request_id, '지원하지 않는 추론 설정입니다')
            return
        worker = ChatWorker(request_id, url, model, messages, options, think=think, parent=self,
                            provider=provider, schema=schema)
        worker.token.connect(self.vue_bridge.chatToken.emit)
        worker.done.connect(self._on_chat_done)
        worker.finished.connect(worker.deleteLater)   # 교체된 옛 워커도 스스로 정리된다
        self._chat_worker = worker
        worker.start()

    def _chat_stop(self, _payload: dict | None = None) -> None:
        request_id = str((_payload or {}).get('id') or '')
        job = getattr(self, '_chat_media_job', None)
        if job is not None and (not request_id or request_id == job.id):
            if job.cancel():
                job.event('stopping', message='이 채팅의 생성 작업을 중지하는 중')
                if self._chat_creator_started:
                    self._creator_cancel({'requestId': job.id})
        worker = self._chat_worker
        if worker is not None and worker.isRunning() and (not request_id or request_id == worker.request_id):
            worker.stop()

    def _chat_start_media(self, request_id, plan):
        from backends import get_backend
        from core.resource_coordinator import get_generation_coordinator

        # 모델 언로드 hold(HOLD_PHASE)는 생성이 아니다 — 거절하지 않고, 작업 스레드의
        # reserve_generation_lease 가 그 언로드가 끝나길 기다린 뒤 리스를 잡는다.
        if getattr(self, '_creator_running', False) or get_generation_coordinator().generation_active():
            raise RuntimeError('다른 이미지·영상 생성 작업이 실행 중입니다')
        model, snapshot = '', {'prompt': plan.prompt}
        if plan.kind == 'image' and plan.family == 'current':
            model, snapshot = self._chat_generation_snapshot(plan.prompt)
            if snapshot.get('_generation_family') == 'krea2':
                plan = replace(plan, family='krea2')
        creator = plan.kind == 'video' or plan.family == 'krea2'
        backend = None if creator else get_backend()  # immutable backend ownership for this request
        if backend is not None:
            from backends import BackendType, get_backend_type
            if get_backend_type() == BackendType.COMFYUI:
                from core.comfy_workflow_controls import snapshot_comfy_payload
                snapshot = snapshot_comfy_payload(backend, snapshot, 'img2img' if plan.image else 'txt2img')
                if plan.image:
                    # The quality preset is explicitly a T2I feature. Its
                    # session eye pass must not leak into reference-image edits.
                    snapshot.pop('_comfy_detail_passes', None)
        if creator:
            self._ensure_creator_runtime()
        job = MediaGenerationJob(request_id, plan, self._chat_emit_media)
        self._chat_media_job = job
        self._chat_creator_started = False
        job.event('queued', model=model or ('MiniMax H3' if plan.kind == 'video' else 'Krea2'),
                  message='영상 생성 요청을 받았습니다' if plan.kind == 'video' else '이미지 생성 요청을 받았습니다')

        def work():
            try:
                if creator:
                    prepared = job.prepare_creator(snapshot)
                    self._chat_media_dispatch.prepared.emit(job, prepared)
                else:
                    from config import OUTPUT_DIR
                    from pathlib import Path
                    # Reuse the existing preference and owned Ollama lifecycle.
                    # Settings reads and unload HTTP both stay on this worker.
                    coordinator = get_generation_coordinator(unload_llm=self._creator_unload_ollama)
                    job.run_current(backend, model, snapshot, Path(OUTPUT_DIR) / 'chat',
                                    coordinator=coordinator,
                                    unload_llm=self._creator_should_unload_ollama())
            except Exception as exc:
                job.terminal(error=str(exc)[:2000])
        threading.Thread(target=work, daemon=True, name='chat-media-' + request_id[:24]).start()

    def _chat_dispatch_creator(self, job, prepared):
        if self._chat_media_job is not job or job.cancelled.is_set():
            job.terminal(error='생성을 중지했습니다')
            return
        self._chat_creator_started = True
        try:
            job.event('dispatching', message='Creator 백엔드에 생성 요청을 전달하는 중')
            self._creator_start_generation(prepared)
        except Exception as exc:
            job.terminal(error=str(exc)[:2000])

    def _chat_creator_progress(self, raw):
        try:
            event = json.loads(raw)
            job = self._chat_media_job
            if job is None or event.get('requestId') != job.id or job.cancelled.is_set():
                return
            job.event(str(event.get('stage') or 'generating'), progress=event.get('percent', 0),
                      message=str(event.get('message') or '생성 중'))
        except (ValueError, TypeError):
            pass

    def _chat_creator_result(self, raw):
        try:
            event = json.loads(raw)
            job = self._chat_media_job
            if job is None or event.get('requestId') != job.id:
                return
            artifacts = [a for a in event.get('artifacts', []) if isinstance(a, dict) and a.get('path')]
            error = '' if event.get('ok') and artifacts else str(event.get('error') or '생성 결과 파일을 받지 못했습니다')
            job.terminal(error=error, artifacts=artifacts)
        except (ValueError, TypeError):
            pass

    def _chat_media_finished(self, raw):
        try:
            event = json.loads(raw)
            job = self._chat_media_job
            if job is None or event.get('id') != job.id or not event.get('done'):
                return
            self._chat_media_job = None
            self._chat_creator_started = False
            threading.Thread(target=job.close, daemon=True, name='chat-input-cleanup').start()
        except (ValueError, TypeError):
            pass

    def _on_chat_done(self, payload_json: str) -> None:
        self.vue_bridge.chatDone.emit(payload_json)
        # 끝난 것이 *현재* 워커일 때만 비운다 — 새 요청으로 교체된 옛 워커의 done 이 늦게 올 수 있다
        finished = self.sender() if hasattr(self, "sender") else None
        if finished is None or finished is self._chat_worker:
            self._chat_worker = None
