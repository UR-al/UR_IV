"""XYZ capability discovery and validated queue snapshots (no backend mutations)."""
from __future__ import annotations

import copy
import json
import secrets
import threading

from core.xyz_capabilities import backend_identity, build_jobs, fetch_capabilities


class XYZActionsMixin:
    def _handle_xyz_action(self, action, payload):
        if action not in ("get_xyz_capabilities", "start_xyz_plot"):
            return False
        self._xyz_ensure()
        if action == "get_xyz_capabilities":
            self._xyz_request_capabilities(payload or {})
        elif action == "start_xyz_plot":
            self._xyz_start_plot(payload or {})
        return True

    def _xyz_ensure(self):
        if hasattr(self, "_xyz_lock"):
            return
        self._xyz_lock = threading.RLock()
        self._xyz_capabilities = None
        self._xyz_query_serial = 0
        self._xyz_seen_requests = set()
        from core.app_context import get_context, Events
        context = get_context()
        self._xyz_subscriptions = [context.subscribe(event, lambda _: self._xyz_invalidate())
                                   for event in (Events.BACKEND_CHANGED, Events.BACKEND_URL_CHANGED)]

    def _xyz_emit(self, signal, payload):
        bridge = getattr(self, "vue_bridge", None)
        target = getattr(bridge, signal, None)
        if target is not None:
            target.emit(json.dumps(payload, ensure_ascii=False))

    def _xyz_invalidate(self):
        with self._xyz_lock:
            self._xyz_query_serial += 1
            self._xyz_capabilities = None
        self._xyz_emit("xyzCapabilitiesReceived", {"ok": False, "invalidated": True, "axes": [],
            "error": "백엔드가 변경되었습니다. 축 목록을 다시 확인합니다."})

    def _xyz_context(self):
        hires = getattr(self, "hires_options_group", None)
        family_check = getattr(self, "_is_krea2_generation", None)
        return {"hires": bool(hires and hires.isChecked()),
                "family": "krea2" if family_check and family_check() else "standard"}

    def _xyz_request_capabilities(self, payload):
        from backends import get_backend, get_backend_type
        backend, kind = get_backend(), get_backend_type().value
        context = self._xyz_context()
        request_id = str(payload.get("requestId") or secrets.token_hex(12))[:100]
        with self._xyz_lock:
            self._xyz_query_serial += 1
            serial = self._xyz_query_serial
            self._xyz_capabilities = None
        def work():
            try:
                capability = fetch_capabilities(backend, kind, **context)
                event = {**capability, "ok": True, "requestId": request_id}
            except Exception as exc:
                event = {"ok": False, "requestId": request_id, "axes": [],
                         "error": str(exc) if isinstance(exc, ValueError) else "백엔드 기능을 읽지 못했습니다. 연결과 API 사용 설정을 확인하세요."}
                capability = None
            with self._xyz_lock:
                if (serial != self._xyz_query_serial or get_backend() is not backend
                        or get_backend_type().value != kind):
                    return  # An old backend may finish after the new one.
                self._xyz_capabilities = {"data": capability, "backend": backend, "context": context} if capability else None
            self._xyz_emit("xyzCapabilitiesReceived", event)
        threading.Thread(target=work, daemon=True, name="xyz-capabilities").start()

    def _xyz_start_plot(self, payload):
        from backends import get_backend, get_backend_type
        request_id = str(payload.get("requestId") or secrets.token_hex(12))[:100]
        event = {"requestId": request_id, "ok": False}
        try:
            with self._xyz_lock:
                if request_id in self._xyz_seen_requests:
                    return
                cached = self._xyz_capabilities
            if not cached or cached["data"]["capabilityId"] != payload.get("capabilityId"):
                raise ValueError("XYZ 축 목록을 새로고침한 뒤 다시 실행하세요")
            backend = get_backend()
            if cached["backend"] is not backend or cached["context"] != self._xyz_context():
                raise ValueError("백엔드 또는 생성 모드가 바뀌었습니다. 축 목록을 새로고침하세요")
            manager = self.queue_manager
            if not getattr(manager, "is_running", False):
                # Starting the queue replaces the main generation worker. Do
                # not implicitly cancel a manual job to start an XYZ plot.
                worker = getattr(self, "gen_worker", None)
                try:
                    active = worker is not None and worker.isRunning()
                except RuntimeError:
                    active = False  # A deleted Qt wrapper is no longer active.
                if active and getattr(worker, "_result_emitted", False) is not True:
                    raise ValueError("이미지 생성 중입니다. 현재 생성이 끝난 뒤 XYZ를 시작하세요")
            base, error = self._build_generation_payload(snapshot=True)
            if error or base is None:
                raise ValueError(error or "생성 설정을 확인하세요")
            # This click is the send site: queued copies no longer carry the
            # sampling blocks the build gate dropped (ui/sampling_blocks).
            from ui.sampling_blocks import freeze_sampling_payload, show_sampling_notice_list
            notices = freeze_sampling_payload(self, base)
            jobs = build_jobs(base, self.model_combo.currentText(), payload.get("axes"), cached["data"])
            kind = get_backend_type().value
            for index, job in enumerate(jobs):
                job["_xyz_backend_id"] = backend_identity(kind, backend)
                job["_xyz_info"].update(requestId=request_id, index=index, total=len(jobs))
                self.queue_panel.add_single_item(job)
            with self._xyz_lock:
                self._xyz_seen_requests.add(request_id)
                if len(self._xyz_seen_requests) > 128:
                    self._xyz_seen_requests = {request_id}
            show_sampling_notice_list(self, notices)   # before the queue can dispatch
            if getattr(manager, "is_running", False):
                manager.total_count += len(jobs)
            else:
                manager.start()
            event.update(ok=True, count=len(jobs))
        except Exception as exc:
            event["error"] = str(exc)[:500]
        self._xyz_emit("xyzPlotEvent", event)

    def _xyz_prepare_queue_generation(self, item):
        """Return the exact queued snapshot; never rebuild from later UI settings."""
        from backends import get_backend, get_backend_type
        backend = get_backend()
        identity = backend_identity(get_backend_type().value, backend)
        if item.get("_xyz_backend_id") != identity:
            raise ValueError("XYZ 작업을 만든 백엔드와 현재 백엔드가 다릅니다. 원래 백엔드를 선택하거나 XYZ를 다시 만드세요")
        payload = copy.deepcopy(item)
        model = str(payload.pop("_xyz_model", ""))
        for key in ("id", "group_id", "group_index", "group_total", "is_last_of_group", "_xyz_backend_id"):
            payload.pop(key, None)
        return payload, model, backend
