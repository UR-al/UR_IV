"""연관 태그 — Vue ``related_tags_*`` 액션의 Python 쪽. 로직은 core/related_tags_service.py(Qt 없음).

액션(브리지 계약 — tests/test_bridge_contract.py 가 아래 ``action in (...)`` 의 리터럴을 AST 로 읽는다):
  related_tags_query  {requestId, fields:{main, character, copyright, count, prefix, suffix}, negative, exclude, limit?}
                      → 워커 스레드 하나가 가장 새 요청만 처리한다. 등급은 메인 창의 등급 필터(_rating_filter)를 따른다.
  related_tags_build  {requestId, shards:['g'|'s'|'q'|'e', ...]} → 백그라운드에서 표를 만든다(한 번에 하나).
                      이미 돌고 있으면 그 진행 이벤트로 답한다.
  related_tags_cancel {requestId}(돌고 있는 build 의 id) → 배치 사이에서 멈춘다. 따로 답하지 않는다(build 가 canceled 로 끝난다).
시그널(ui/vue_bridge.py): ``relatedTagsEvent`` — JSON {action, requestId, ...}(core/related_tags_service.py 설명).

웹 모드도 쓴다 — 대화상자를 열지 않고 앱 캐시(cache/related_tags)에만 쓴다. 메인 프롬프트를 바꾸는 건
프론트가 기존 프록시(main_prompt_text)로 하므로 Forge·ComfyUI 에 똑같이 나간다.
"""
from __future__ import annotations

import json
import threading

_INIT_LOCK = threading.Lock()


def _user_message(exc: Exception) -> str:
    from core.error_handler import sanitize_for_ui

    return sanitize_for_ui(str(exc) or type(exc).__name__, 300)


class RelatedTagsActionsMixin:
    def _handle_related_tags_action(self, action, payload):
        if action in ("related_tags_query", "related_tags_build", "related_tags_cancel"):
            self._related_tags_dispatch(action, payload)
            return True
        return False

    # ── 상태 ──
    def _related_tags_service(self):
        with _INIT_LOCK:
            service = getattr(self, "_related_tags_service_obj", None)
            if service is None:
                factory = getattr(self, "_related_tags_service_factory", None)
                if factory is None:
                    from core.fetch_data import DATA_DIR
                    from core.related_tags_service import RelatedTagsService

                    def factory(emit):
                        return RelatedTagsService(emit, data_dir=DATA_DIR)
                service = factory(self._related_tags_emit)
                self._related_tags_service_obj = service
            return service

    def _related_tags_emit(self, event: dict) -> None:
        if getattr(self, "_related_tags_closed", False):
            return
        signal = getattr(getattr(self, "vue_bridge", None), "relatedTagsEvent", None)
        if signal is None:
            return
        try:
            signal.emit(json.dumps(event, ensure_ascii=False))
        except RuntimeError:
            pass  # 종료 중 QObject 가 이미 사라졌다

    def _related_tags_ratings(self):
        ratings = getattr(self, "_rating_filter", None)
        return set(ratings) if ratings else {"g", "s"}

    def _shutdown_related_tags(self):
        """종료 — 더 보내지 않고, 도는 표 만들기는 다음 배치에서 멈춘다(저장된 등급 표는 남는다)."""
        self._related_tags_closed = True
        service = getattr(self, "_related_tags_service_obj", None)
        if service is not None:
            service.close()

    # ── 액션 ──
    def _related_tags_dispatch(self, action, payload):
        if getattr(self, "_related_tags_closed", False):
            return
        from core.related_tags import parse_build_payload, parse_cancel_payload, parse_query_payload

        request = payload if isinstance(payload, dict) else {}
        raw_id = request.get("requestId", "")
        request_id = raw_id if isinstance(raw_id, str) else ""
        try:
            if action == "related_tags_cancel":
                self._related_tags_service().cancel(parse_cancel_payload(request))
                return
            if action == "related_tags_build":
                build = parse_build_payload(request)
                service = self._related_tags_service()
                if not service.start_build(build.request_id, build.shards):
                    running = service.build_event()
                    if running is not None:
                        self._related_tags_emit(running)
                return
            query = parse_query_payload(request)
            self._related_tags_service().submit_query(query, self._related_tags_ratings())
        except Exception as exc:
            if action == "related_tags_cancel":
                print(f"[RelatedTags] 취소 요청 무시: {exc}")
            elif action == "related_tags_build":
                self._related_tags_emit({"action": "related_tags_build", "requestId": request_id,
                                         "state": "error", "error": _user_message(exc)})
            else:
                self._related_tags_emit({
                    "action": "related_tags_query", "requestId": request_id, "state": "error",
                    "tags": [], "context": [], "ratings": [], "missing": [], "stale": [], "unavailable": [],
                    "estimateSec": 0, "build": None, "error": _user_message(exc)})
