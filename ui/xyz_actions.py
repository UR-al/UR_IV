"""XYZ capability discovery and validated queue snapshots (no backend mutations)."""
from __future__ import annotations

import copy
import json
import secrets
import threading

from PyQt6.QtCore import QObject, Qt, pyqtSignal

from core.xyz_capabilities import (TEXT_ENCODER, ForgeTextEncoderListError, backend_identity, build_jobs,
                                   fetch_capabilities)


class _XYZGuiRelay(QObject):
    """작업자 스레드가 'GUI 스레드에서 이것을 불러 달라'고 넘기는 통로(큐 연결 — ui/memo_actions._MemoGuiRelay 와 같다)."""

    call = pyqtSignal(object)

    def __init__(self):
        super().__init__()
        self.call.connect(self._run, Qt.ConnectionType.QueuedConnection)

    def _run(self, fn):
        fn()


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
        # 창(QObject)이면 창의 스레드(GUI)로 넘기는 통로. 테스트 호스트(QObject 아님)는 부른 스레드에서 바로 부른다
        self._xyz_gui_relay = None
        if isinstance(self, QObject):
            relay = _XYZGuiRelay()
            relay.moveToThread(self.thread())   # 슬롯은 창의 스레드에서 돈다
            self._xyz_gui_relay = relay   # 부모 없이 만들었으니 참조를 쥔다
        from core.app_context import get_context, Events
        context = get_context()
        self._xyz_subscriptions = [context.subscribe(event, lambda _: self._xyz_invalidate())
                                   for event in (Events.BACKEND_CHANGED, Events.BACKEND_URL_CHANGED)]

    def _xyz_on_gui(self, fn):
        """작업자 스레드에서: ``fn`` 을 GUI 스레드에서 부른다(통로가 없는 테스트 호스트는 바로)."""
        relay = getattr(self, "_xyz_gui_relay", None)
        if relay is None:
            fn()
        else:
            relay.call.emit(fn)

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
                "family": "krea2" if family_check and family_check() else "standard",
                "text_encoders": self._xyz_text_encoder_choices()}

    def _xyz_text_encoder_choices(self):
        """Text Encoder 축 값 — Vue 가 지금 보여 주는 TE 칩 선택지(``te_main_input.items``, 연결 때 채운다). 모르면 ()."""
        getter = getattr(getattr(self, "vue_bridge", None), "last_widget_property", None)
        if not callable(getter):
            return ()
        try:
            items = getter("te_main_input", "items")
        except Exception:
            return ()
        return tuple(str(item) for item in items if str(item or "").strip()) if isinstance(items, list) else ()

    def _xyz_text_encoder_selection(self):
        """지금 TE 선택(TE 칩 값 — forge_additional_modules 의 TE 칸, ui/generator_generation._build_vae_te_override 와 같다)."""
        widget = getattr(self, "te_main_input", None)
        try:
            text = widget.text() if widget is not None and hasattr(widget, "text") else ""
        except Exception:
            text = ""
        from core.main_module_choices import split_te_selection
        return split_te_selection(text)

    @staticmethod
    def _xyz_forge_modules(backend, kind, capability):
        """칩이 빈 Forge 에서 Text Encoder 축 칸이 TE 를 바꿀 Forge 의 목록(WebUIBackend.forge_module_base — 보통 생성이 쓸
        목록). 첫 칸을 만들기 전에 TE 가 둘 이상인지 보는 데만 쓴다(core/xyz_capabilities.build_jobs). 축이 없거나 Forge 가
        아니거나 못 읽으면 None — 칸을 보낼 때 백엔드가 그때의 목록으로 다시 본다."""
        if kind != "webui" or not any(axis.get("id") == TEXT_ENCODER for axis in capability.get("axes", [])):
            return None
        reader = getattr(backend, "forge_module_base", None)
        if not callable(reader):
            return None
        try:
            names = reader()
        except Exception:
            return None
        return list(names) if isinstance(names, list) else None

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
            # 작업자 스레드에서 같이 읽는다 — 시작 클릭(GUI 스레드)은 네트워크를 타지 않는다
            forge_modules = self._xyz_forge_modules(backend, kind, capability) if capability else None
            with self._xyz_lock:
                if (serial != self._xyz_query_serial or get_backend() is not backend
                        or get_backend_type().value != kind):
                    return  # An old backend may finish after the new one.
                self._xyz_capabilities = {"data": capability, "backend": backend, "context": context,
                                          "forge_modules": forge_modules} if capability else None
            self._xyz_emit("xyzCapabilitiesReceived", event)
        threading.Thread(target=work, daemon=True, name="xyz-capabilities").start()

    def _xyz_reread_forge_modules(self, cached, backend, kind, payload):
        """축 목록을 받을 때 읽어 둔 Forge 목록이 시작을 거절했다(TE 둘 이상). 사용자가 그 뒤 Forge 에서 TE 를 하나로 고쳤을 수
        있으니 작업자 스레드에서 Forge 의 지금 목록을 다시 읽어 캐시에 두고, GUI 스레드에서 한 번 더 시작한다 — 그 목록도
        거절하면 그때 거절한다(GUI 스레드는 네트워크를 타지 않는다). 못 읽으면 읽어 둔 목록으로 다시 보고 그렇다고 알린다."""
        def work():
            fresh = self._xyz_forge_modules(backend, kind, cached["data"])
            with self._xyz_lock:
                if fresh is not None and self._xyz_capabilities is cached:   # 그사이 새로고침했으면 그 목록이 더 새것
                    self._xyz_capabilities = {**cached, "forge_modules": fresh}
            self._xyz_on_gui(lambda: self._xyz_start_plot(payload, forge_reread=fresh is not None))
        threading.Thread(target=work, daemon=True, name="xyz-forge-modules").start()

    def _xyz_start_plot(self, payload, *, forge_reread=None):
        """``forge_reread`` = 이 클릭을 Forge 목록을 다시 읽은 뒤 다시 부른 것인지(None = 클릭 그대로, True = 다시 읽음,
        False = 다시 읽지 못함 — _xyz_reread_forge_modules)."""
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
            kind = get_backend_type().value
            try:
                jobs = build_jobs(base, self.model_combo.currentText(), payload.get("axes"), cached["data"],
                                  text_encoders=self._xyz_text_encoder_selection(),
                                  forge_modules=cached.get("forge_modules"))
            except ForgeTextEncoderListError as exc:
                if forge_reread is None:
                    # 읽어 둔 목록이 낡았을 수 있다 — 지금 목록으로 다시 본 뒤 답한다(이 클릭은 아직 답하지 않는다)
                    self._xyz_reread_forge_modules(cached, backend, kind, payload)
                    return
                if forge_reread is False:
                    raise ValueError(f"{exc} (Forge 의 지금 목록을 다시 읽지 못해 축 목록을 받을 때 읽은 목록으로 "
                                     "봤습니다 — 연결을 확인하세요)") from exc
                raise
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
        if get_backend_type().value != "webui":
            # Text Encoder 축의 '되돌릴 모듈 변경' 표시와 칩이 빈 칸의 TE 선택은 Forge 옵션용 — ComfyUI 는 칸마다 그래프를
            # 새로 만든다
            from core.xyz_capabilities import FORGE_TE_CELL_KEY, TRANSIENT_MODULES_KEY
            payload.pop(TRANSIENT_MODULES_KEY, None)
            payload.pop(FORGE_TE_CELL_KEY, None)
        return payload, model, backend
