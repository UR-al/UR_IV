"""Creator Studio action module.

This mixin is the single seam between Vue actions and the Creator domain
modules.  Heavy work runs off the Qt thread; Vue receives transport-neutral
JSON events through :class:`ui.vue_bridge.VueBridge`.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, Optional
import base64
import copy
import json
import mimetypes
import os
import re
import secrets
import threading
import time
from contextlib import contextmanager


# UI 스레드가 뜬 컷별 T2I 스냅샷을 워커에 넘기는 내부 키. Vue 가 보낸 같은 키는 시작 시 버린다
# (원격 web 모드에서 임의 payload 를 Forge 로 주입하지 못하게).
_COMIC_T2I_KEY = "_comicT2I"


class CreatorActionsMixin:
    """Handle all Creator actions behind one small action-dispatch interface."""

    def _handle_creator_action(self, action: str, payload: dict) -> bool:
        # Vue 액션 → 처리 메서드. 이 표 한 벌이 게이트와 디스패치를 겸한다(이름 목록 사본 없음).
        # 정적 검사 두 곳이 이 모양에 기대므로 유지할 것:
        #   - tests/test_bridge_contract.py: `action not in handlers` 의 dict 키 = 계약 액션 이름
        #   - tests/test_web_action_policy.py: {'action': self._method} 값으로 대화상자 도달 분석
        handlers = {
            "creator_get_state": self._creator_request_state,
            "creator_select_media": self._creator_select_media,
            "creator_generate": self._creator_start_generation,
            "creator_cancel": self._creator_cancel,
            "creator_h3_cache_status": self._creator_cache_status,
            "creator_h3_cache_clear": self._creator_cache_clear,
            "comic_plan": self._comic_start_plan,
            "comic_save": self._comic_save,
            "comic_generate_all": self._comic_start_generate_all,
            "comic_animate_all": self._comic_start_animate_all,
            "comic_export_page": self._comic_export_page,
            "comic_export_living": self._comic_start_export_living,
        }
        if action not in handlers:
            return False
        self._ensure_creator_runtime()
        handlers[action](payload or {})
        return True

    def _creator_cache_status(self, payload: dict) -> None:
        self._creator_cache_request("status", payload)

    def _creator_cache_clear(self, payload: dict) -> None:
        self._creator_cache_request("clear", payload)

    def _ensure_creator_runtime(self) -> None:
        if hasattr(self, "_creator_coordinator"):
            return
        from core.resource_coordinator import get_generation_coordinator

        self._creator_cancel_event = threading.Event()
        self._creator_state_lock = threading.RLock()
        self._creator_running = False
        self._creator_mode = ""
        self._creator_active_job = None
        self._creator_job_local = threading.local()
        self._creator_coordinator = get_generation_coordinator(
            unload_llm=self._creator_unload_ollama,
            on_state=lambda state: self._creator_emit(
                "creatorStateChanged",
                {
                    "resourcePhase": state.phase,
                    "resourceOwner": state.owner,
                    "llmUnloaded": state.llm_unloaded,
                    "running": self._creator_running,
                    "busy": state.phase != "idle" or self._creator_running,
                    "ready": state.phase == "idle" and not self._creator_running,
                    "status": "running" if state.phase != "idle" or self._creator_running else "ready",
                    "mode": self._creator_mode,
                },
            ),
        )

    def _creator_emit(self, signal_name: str, payload: Dict[str, Any]) -> None:
        payload = dict(payload)
        if signal_name in {"creatorProgress", "creatorResult"}:
            local = getattr(self, "_creator_job_local", None)
            job = getattr(local, "job", None) or getattr(self, "_creator_active_job", None)
            payload.setdefault("requestId", job["requestId"] if job else "")
            if (job and payload["requestId"] == job["requestId"]
                    and job["cancel"].is_set() and signal_name == "creatorResult"
                    and payload.get("ok")):
                payload = {"ok": False, "canceled": True, "mode": payload.get("mode", ""),
                           "requestId": job["requestId"], "error": "생성이 취소되었습니다"}
        bridge = getattr(self, "vue_bridge", None)
        signal = getattr(bridge, signal_name, None)
        if signal is not None:
            signal.emit(json.dumps(payload, ensure_ascii=False))

    def _creator_set_running(self, running: bool, mode: str = "") -> None:
        with self._creator_state_lock:
            self._creator_running = bool(running)
            self._creator_mode = str(mode if running else "")
        self._creator_emit(
            "creatorStateChanged",
            {
                "running": self._creator_running,
                "busy": self._creator_running,
                "status": "running" if self._creator_running else "ready",
                "mode": self._creator_mode,
            },
        )

    def _creator_run_thread(self, mode: str, target, payload: dict) -> None:
        payload = dict(payload)
        request_id = str(payload.get("requestId") or secrets.token_hex(16))[:100]
        payload["requestId"] = request_id
        with self._creator_state_lock:
            if self._creator_running:
                if self._creator_active_job and self._creator_active_job["requestId"] == request_id:
                    # Retried transport delivery is not a terminal failure of
                    # the already accepted job with this identity.
                    return
                self._creator_emit(
                    "creatorResult",
                    {"ok": False, "mode": mode, "requestId": request_id,
                     "error": "다른 Creator 작업이 실행 중입니다"},
                )
                return
            job = {"requestId": request_id, "cancel": threading.Event(), "backend": None,
                   "backend_lock": threading.RLock()}
            self._creator_active_job = job
            self._creator_cancel_event = job["cancel"]
            self._creator_set_running(True, mode)

        def _work():
            self._creator_job_local.job = job
            try:
                target(payload)
            except Exception as exc:
                self._creator_emit(
                    "creatorResult",
                    {"ok": False, "mode": mode, "requestId": request_id,
                     "canceled": job["cancel"].is_set(), "error": str(exc)[:2000]},
                )
            finally:
                with self._creator_state_lock:
                    if self._creator_active_job is job:
                        self._creator_active_job = None
                        self._creator_set_running(False)
                self._creator_job_local.job = None

        threading.Thread(
            target=_work,
            daemon=True,
            name=f"creator-{re.sub(r'[^a-z0-9_-]', '-', mode.lower())}",
        ).start()

    # ── Creator state and generic generation ──────────────────────────────

    def _creator_request_state(self, payload: dict) -> None:
        def _work():
            state = {
                "running": self._creator_running,
                "busy": self._creator_running,
                "mode": self._creator_mode,
                "backend": "",
                "connected": False,
                "ready": False,
                "status": "checking",
            }
            try:
                from backends import get_backend, get_backend_type

                backend = get_backend()
                state["backend"] = get_backend_type().value
                # 상태 조회는 연결 확인만 한다. 노드 목록(/object_info 전체)은 생성
                # 경로가 필요할 때 따로 받으므로, 여기서 받으면 느린/실패한 요청이
                # 연결된 백엔드를 '오류'로 표시하고 Comic 복구 emit까지 늦춘다.
                state["connected"] = bool(backend.test_connection())
                state["ready"] = state["connected"]
                state["status"] = "ready" if state["connected"] else "offline"
            except Exception as exc:
                state["error"] = str(exc)
                state["status"] = "error"
            self._creator_emit("creatorStateChanged", state)

            comic = self._comic_studio()
            try:
                reconciled = comic.reconcile(payload.get("comicRecovery"))
                persistence = {"status": reconciled.status, "source": "backend"}
                if reconciled.conflict_path:
                    persistence["conflictPath"] = reconciled.conflict_path
                self._comic_emit_document(reconciled.document, **persistence)
            except Exception as exc:
                self._comic_emit_document(None, status="error", error=str(exc))

        threading.Thread(target=_work, daemon=True, name="creator-state").start()

    def _creator_select_media(self, payload: dict) -> None:
        from PyQt6.QtWidgets import QFileDialog

        slot = str(payload.get("slot", "source"))[:50]
        file_filter = (
            "Creator Media (*.png *.jpg *.jpeg *.webp *.bmp *.tif *.tiff "
            "*.mp4 *.webm *.mov *.mkv *.avi *.m4v);;All Files (*)"
        )
        path, _ = QFileDialog.getOpenFileName(self, "Creator 입력 파일 선택", "", file_filter)
        if path:
            self._creator_emit(
                "creatorMediaSelected",
                {"slot": slot, "path": path.replace("\\", "/")},
            )

    def _creator_start_generation(self, payload: dict) -> None:
        mode = str(payload.get("mode", "")).strip()
        if mode == "comic_panel":
            # 컷 T2I 는 메인 T2I 설정을 따른다 — 위젯은 UI 스레드에서만 읽으므로 여기서 스냅샷.
            request = self._comic_prepare_request(payload, mode)
            if request is not None:
                self._creator_run_thread(mode, self._comic_generate_panel, request)
            return
        self._creator_run_thread(mode or "creator", self._creator_generate, payload)

    def _creator_generate(self, payload: dict) -> None:
        from backends import BackendType, get_backend, get_backend_type
        from core.creator_workflows import build, canonical_mode as creator_canonical_mode

        if get_backend_type() is not BackendType.COMFYUI:
            raise RuntimeError("Creator 영상/Krea2 생성은 ComfyUI 백엔드가 필요합니다")
        backend = get_backend()
        if not hasattr(backend, "run_workflow"):
            raise RuntimeError("현재 ComfyUI adapter가 Creator 워크플로 실행을 지원하지 않습니다")

        requested_mode = str(payload.get("mode", ""))
        # 별칭 표는 creator_workflows 한 벌 — 지원하지 않는 mode 는 업로드 전에 실패한다.
        canonical_mode = creator_canonical_mode(requested_mode)
        params = self._creator_prepare_params(backend, dict(payload))
        object_info = self._creator_object_info(backend, self._creator_cancel_event.is_set)
        available = set(object_info.keys())
        self._creator_configure_h3_cache(canonical_mode, params, available)
        built = build(canonical_mode, params)
        self._creator_check_nodes(built, available)
        self._creator_resolve_comfy_choices(built, object_info)

        unload = self._creator_should_unload_ollama()
        all_artifacts = []
        info: Dict[str, Any] = {"passes": []}
        with self._creator_reserve(backend, requested_mode or "creator", unload):
            result = self._creator_run_workflow(backend, built, self._creator_progress_callback)
            if not result.success:
                raise RuntimeError(result.error or "Creator 생성에 실패했습니다")
            all_artifacts.extend(result.artifacts)
            info["passes"].append({"mode": canonical_mode, **(result.info or {})})

            if canonical_mode == "krea2_edit" and bool(params.get("hires")):
                primary = next(
                    (artifact for artifact in result.artifacts if artifact.kind in {"image", "animated"} and artifact.data),
                    None,
                )
                if primary is None:
                    raise RuntimeError("Krea2 hires 입력으로 사용할 이미지 결과가 없습니다")
                hires_name = self._creator_upload_bytes(
                    backend,
                    primary.data,
                    "krea2_hires_input",
                    self._creator_artifact_extension(primary.filename, primary.mime),
                    primary.mime or "image/png",
                )
                hires_params = {
                    "input_image": hires_name,
                    "source_size": self._creator_image_size(primary.data),
                    "prompt": params.get("prompt", ""),
                    "seed": params["seed"],
                    "scale": params.get("hiresScale", params.get("hires_scale", 2)),
                    "denoise": params.get("hiresDenoise", params.get("hires_denoise", 0.45)),
                }
                hires_built = build("krea2_hires", hires_params)
                self._creator_check_nodes(hires_built, available)
                self._creator_resolve_comfy_choices(hires_built, object_info)
                self._creator_check_cancelled()
                result = self._creator_run_workflow(backend, hires_built, self._creator_progress_callback)
                if not result.success:
                    raise RuntimeError(result.error or "Krea2 hires 생성에 실패했습니다")
                all_artifacts.extend(result.artifacts)
                info["passes"].append({"mode": "krea2_hires", **(result.info or {})})
        if self._creator_cancel_event.is_set():
            raise RuntimeError("생성이 취소되었습니다")
        artifacts = self._creator_save_artifacts(all_artifacts, requested_mode or "creator")
        primary = artifacts[-1] if artifacts else {}
        self._creator_emit(
            "creatorResult",
            {
                "ok": True,
                "mode": requested_mode,
                "path": primary.get("path", ""),
                "mediaType": primary.get("kind", ""),
                "artifacts": artifacts,
                "info": info,
            },
        )

    @contextmanager
    def _creator_reserve(self, backend, owner, unload):
        # 진행 중인 '생성 후 언로드'가 있으면 끝난 뒤 리스를 잡는다 — 언로드도 같은 리스를 쥐므로
        # 기다리지 않으면 그 틈에 시작한 Creator 작업이 '사용 중'으로 실패한다(작업 스레드에서 기다림).
        from core.post_generation import reserve_generation_lease
        cancel_event = getattr(self, "_creator_cancel_event", None)
        with reserve_generation_lease(owner, coordinator=self._creator_coordinator, unload_llm=unload,
                                      cancelled=cancel_event.is_set if cancel_event is not None else None):
            self._creator_claim_backend(backend)
            try:
                self._creator_check_cancelled()
                yield
            finally:
                job = getattr(self._creator_job_local, "job", None)
                if job is not None:
                    with job["backend_lock"], self._creator_state_lock:
                        if self._creator_active_job is job:
                            job["backend"] = None

    def _creator_claim_backend(self, backend) -> None:
        """Called only while this job owns the shared generation lease."""
        job = getattr(self._creator_job_local, "job", None)
        if job is not None:
            with job["backend_lock"], self._creator_state_lock:
                if self._creator_active_job is job:
                    job["backend"] = backend

    def _creator_check_cancelled(self) -> None:
        local = getattr(self, "_creator_job_local", None)
        job = getattr(local, "job", None)
        if job is not None and job["cancel"].is_set():
            raise RuntimeError("생성이 취소되었습니다")

    def _creator_cache_preferences(self):
        prefs = self._creator_prefs()
        enabled = prefs.get("h3ConditioningCacheEnabled", True) is True
        try:
            gb = max(1, min(64, int(prefs.get("h3ConditioningCacheMaxGB", 8))))
            entries = max(1, min(256, int(prefs.get("h3ConditioningCacheMaxEntries", 32))))
        except (ValueError, TypeError, OverflowError):
            gb, entries = 8, 32
        return enabled, gb * 1024 ** 3, entries

    def _creator_configure_h3_cache(self, mode, params, available):
        """H3 실행 전 서버 능력에 맞춘 캐시 설정 — Creator 생성과 Living Comic 이 같이 쓴다.

        - turbo T2V/I2V 는 사용자가 따로 정하지 않았으면 서버에 MiniMaxH3BlockCacheT8 가
          있을 때 block cache 를 켠다(예전엔 _creator_generate 에만 있어 Living Comic 이 못 썼다).
        - 조건부 인코딩 캐시(conditioning_cache)는 설정·노드 지원 여부로 정한다.
        """
        if not mode.startswith("h3_"):
            return
        if (
            mode in {"h3_t2v", "h3_i2v"}
            and str(params.get("quality", "turbo")).strip().lower() == "turbo"
            and "block_cache" not in params
            and "blockCache" not in params
        ):
            params["block_cache"] = "MiniMaxH3BlockCacheT8" in available
        from core.h3_conditioning_cache import CACHE_NODE_TYPES
        enabled, max_bytes, max_entries = self._creator_cache_preferences()
        enabled = params.get("conditioning_cache", params.get("h3ConditioningCacheEnabled", enabled)) is True
        supported = (CACHE_NODE_TYPES | {"EmptyMiniMaxH3LatentAV"}).issubset(available)
        params["conditioning_cache"] = enabled and supported
        params["conditioning_cache_max_bytes"] = max_bytes
        params["conditioning_cache_max_entries"] = max_entries
        if enabled and not supported:
            self._creator_emit("creatorProgress", {"stage": "cache_unavailable", "percent": 0,
                "cache": "unavailable", "message": "H3 캐시 노드가 없어 이번 작업은 캐시 없이 생성합니다. ComfyUI 노드 업데이트/재시작이 필요합니다."})

    def _creator_run_workflow(self, backend, built, progress):
        from core.h3_conditioning_cache import prepare_receipt

        stages = built.get("stages") or [{"name": "sample", "workflow": built["workflow"]}]
        receipt = None
        result = None
        for stage in stages:
            self._creator_check_cancelled()
            encode = stage["name"] == "encode"
            graph = copy.deepcopy(stage["workflow"])
            if receipt and not encode:
                graph["6"]["inputs"]["expected_key"] = receipt["key"]
            if encode:
                self._creator_emit("creatorProgress", {"stage": "cache_check", "percent": 0,
                    "message": "H3 인코딩 캐시와 서버의 모델·입력 파일을 확인하는 중"})

            def stage_progress(value, maximum, preview=None):
                if len(stages) == 1:
                    progress(value, maximum, preview)
                    return
                ratio = max(0.0, min(1.0, value / maximum if maximum else 0))
                progress(round((ratio * 25) if encode else (25 + ratio * 75)), 100, preview)

            kwargs = {"cancel_check": self._creator_cancel_event.is_set}
            if encode:
                kwargs["allow_empty_outputs"] = True
            result = backend.run_workflow(graph, stage_progress, **kwargs)
            self._creator_check_cancelled()
            if not result.success:
                raise RuntimeError(result.error or "Creator 워크플로 실행에 실패했습니다")
            if encode:
                receipt = prepare_receipt(result.info or {})
                self._creator_emit("creatorProgress", {"stage": "cache_hit" if receipt.get("hit") else "cache_saved",
                    "percent": 25, "cache": "hit" if receipt.get("hit") else "miss",
                    "message": "저장된 H3 인코딩을 재사용합니다 · 인코더 해제 완료" if receipt.get("hit") else "H3 인코딩 저장 · 인코더 해제 완료"})
                self._creator_check_cancelled()
        if receipt is not None:
            result.info = {**(result.info or {}), "conditioning_cache": receipt}
        return result

    def _creator_cache_request(self, operation, payload):
        def work():
            import requests
            event = {"operation": operation, "requestId": str(payload.get("requestId", ""))[:100],
                     "ok": False, "available": False, "scope": "comfy_server"}
            enabled, max_bytes, max_entries = self._creator_cache_preferences()
            event.update(enabled=enabled, maxBytes=max_bytes, maxEntries=max_entries)
            try:
                from backends import BackendType, get_backend, get_backend_type
                if get_backend_type() is not BackendType.COMFYUI:
                    raise RuntimeError("H3 캐시는 선택된 ComfyUI 서버에서 관리합니다")
                with self._creator_state_lock:
                    if operation == "clear" and self._creator_running:
                        raise RuntimeError("Creator 작업이 끝난 뒤 캐시를 비워 주세요")
                backend = get_backend()
                url = f"{backend.api_url.rstrip('/')}/aistudio/h3-cache/{operation}"
                response = (requests.post(url, json={}, timeout=15) if operation == "clear"
                            else requests.get(url, timeout=15))
                if response.status_code == 404:
                    raise RuntimeError("ComfyUI에 H3 캐시 노드가 없습니다. 노드 업데이트 후 서버를 재시작하세요")
                event["available"] = True
                if response.status_code == 409:
                    raise RuntimeError("ComfyUI 서버에서 다른 작업이 실행 중입니다. 완료 후 캐시를 비워 주세요")
                response.raise_for_status()
                data = response.json()
                # Never forward arbitrary server paths or other clients' metadata.
                for key in ("entries", "bytes", "removedEntries", "removedBytes"):
                    if key in data:
                        event[key] = max(0, int(data[key]))
                event["ok"] = True
            except Exception as exc:
                # Network errors may contain a URL; do not expose backend addresses.
                event["error"] = str(exc)[:300] if isinstance(exc, RuntimeError) else "H3 캐시 서버 요청에 실패했습니다"
            self._creator_emit("creatorCacheEvent", event)
        threading.Thread(target=work, daemon=True, name=f"creator-cache-{operation}").start()

    def _creator_cancel(self, payload: dict) -> None:
        requested = str(payload.get("requestId") or "")[:100]
        with self._creator_state_lock:
            job = self._creator_active_job
            if job is None or (requested and requested != job["requestId"]):
                self._creator_emit("creatorProgress", {
                    "stage": "cancel_ignored", "requestId": requested,
                    "message": "해당 Creator 작업은 실행 중이 아닙니다",
                })
                return
            if job["cancel"].is_set():
                return
            job["cancel"].set()

        def interrupt_owned():
            # Network latency must not hold the short UI state lock. Only the
            # worker's lease release waits on this separate ownership lock.
            with job["backend_lock"]:
                with self._creator_state_lock:
                    backend = job["backend"] if self._creator_active_job is job else None
                if backend is None:
                    return
                try:
                    backend.interrupt()
                except Exception:
                    pass
        threading.Thread(target=interrupt_owned, daemon=True, name="creator-cancel").start()
        self._creator_emit("creatorProgress", {"stage": "cancel", "requestId": job["requestId"],
                                               "message": "취소 요청 전송"})

    def _creator_progress_callback(self, value: int, maximum: int, _preview=None) -> None:
        percent = int(value if maximum == 100 else (value * 100 / maximum if maximum else 0))
        self._creator_emit(
            "creatorProgress",
            {
                "stage": "generate",
                "value": int(value),
                "maximum": int(maximum),
                "percent": max(0, min(100, percent)),
            },
        )

    @staticmethod
    def _creator_object_info(backend, cancel_check=None) -> dict:
        """ComfyUI /object_info — Krea2 T2I/I2I 러너와 같은 backend.get_object_info(재시도 3회).

        get_object_info 가 없는 duck-typed 어댑터(테스트 fake·옛 확장)만 직접 GET 으로 폴백한다.
        """
        getter = getattr(backend, "get_object_info", None)
        if callable(getter):
            from core.cancellable_call import call_with_optional_cancel

            data = call_with_optional_cancel(getter, cancel_check=cancel_check)
            if not isinstance(data, dict):
                raise RuntimeError("ComfyUI /object_info 응답이 올바른 객체가 아닙니다")
            return data
        import requests

        response = requests.get(f"{backend.api_url.rstrip('/')}/object_info", timeout=15)
        response.raise_for_status()
        data = response.json()
        return data if isinstance(data, dict) else {}

    @staticmethod
    def _creator_random_seed(mode: str) -> int:
        """-1 seed 의 실제 값. Krea2 는 앱 replay 범위(32비트) 안에서만 뽑는다."""
        from core.generation_family import KREA2_SEED_MAX

        if str(mode).startswith("krea2"):
            return secrets.randbelow(KREA2_SEED_MAX + 1)
        return secrets.randbits(63)

    @staticmethod
    def _creator_upload_extension(path: Path) -> str:
        """업로드 이름(내용 해시)에 붙일 확장자 — 없거나 영숫자가 아니면 업로드 전에 거부한다.

        추측하지 않는다: build() 의 입력 확장자 검사(이미지/영상)가 이 값에 기댄다.
        """
        extension = path.suffix.lstrip(".").lower()
        if not extension or not extension.isascii() or not extension.isalnum():
            raise ValueError(f"입력 파일의 확장자를 알 수 없습니다: {path}")
        return extension

    @staticmethod
    def _creator_artifact_extension(filename: Any, mime: Any) -> str:
        """ComfyUI 결과물을 다시 올릴 때의 확장자 — 결과 파일명 → MIME → ``png`` 순.

        이미지 MIME 은 표로 먼저 본다 — ``mimetypes`` 는 Windows 레지스트리에 기대 webp 등을
        모를 수 있다.
        """
        suffix = Path(str(filename or "")).suffix.lstrip(".").lower()
        if suffix and suffix.isascii() and suffix.isalnum():
            return suffix
        clean_mime = str(mime or "").split(";")[0].strip().lower()
        known = {"image/png": "png", "image/jpeg": "jpg", "image/webp": "webp",
                 "image/bmp": "bmp", "image/tiff": "tiff"}.get(clean_mime)
        if known:
            return known
        guessed = (mimetypes.guess_extension(clean_mime) or "").lstrip(".").lower()
        if guessed and guessed.isascii() and guessed.isalnum():
            return guessed
        return "png"

    def _creator_upload_bytes(self, backend, data: bytes, prefix: str, extension: str, mime: str) -> str:
        """Creator 의 모든 ComfyUI 업로드 — 내용 해시 이름 + ``overwrite=False``.

        예전엔 사용자 파일명(basename)으로 ``overwrite=true`` 업로드해, 다른 폴더의 동명
        원본·참조(0001.png 등)가 서로를 덮어써 Krea2 편집의 두 LoadImage 가 같은(참조) 그림을
        읽었고, ComfyUI/input 에 원래 있던 같은 이름의 파일도 조용히 덮였다. 같은 바이트는 같은
        이름이라 다시 올려도 input 폴더에 사본이 쌓이지 않는다(core/comfy_upload_names.py).
        """
        from core.comfy_upload_names import upload_content

        event = getattr(self, "_creator_cancel_event", None)
        return upload_content(
            backend, data, prefix, extension, mime,
            cancel_check=event.is_set if event is not None else None,
        )

    def _creator_upload_file(self, backend, path: Path, prefix: str) -> str:
        extension = self._creator_upload_extension(path)
        mime = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
        return self._creator_upload_bytes(backend, path.read_bytes(), prefix, extension, mime)

    def _creator_prepare_params(self, backend, params: Dict[str, Any]) -> Dict[str, Any]:
        """Normalize the Vue payload and upload only media used by the graph."""
        from core.creator_workflows import (
            CreatorWorkflowError,
            canonical_mode as creator_canonical_mode,
            media_inputs as creator_media_inputs,
        )
        from core.generation_family import KREA2_SEED_MAX

        mode = str(params.get("mode", "")).strip().lower()
        try:
            canonical = creator_canonical_mode(mode)
        except CreatorWorkflowError:
            canonical = mode  # 알 수 없는 mode 는 build() 가 같은 오류로 거부한다

        # seed 는 업로드 전에 검증한다 — 잘못된 요청이 ComfyUI/input 에 파일만 남기지 않게.
        seed = params.get("seed", -1)
        if isinstance(seed, bool) or not isinstance(seed, int):
            raise ValueError("seed는 정수여야 합니다")
        if canonical.startswith("krea2") and seed > KREA2_SEED_MAX:
            raise ValueError(f"seed는 -1 또는 0~{KREA2_SEED_MAX} 범위여야 합니다")
        params["seed"] = self._creator_random_seed(canonical) if seed < 0 else seed

        # 이하 mode 분기는 전부 정규 mode 기준 — 'h3-v2v'/'H3_V2V' 같은 별칭도 build('h3_v2v') 와
        # 같은 업로드 키·오디오·레퍼런스 지시를 받는다(원문 문자열로 가르면 input_image 로 올라가
        # 업로드 뒤에 확장자 오류로 실패했다).
        is_v2v = canonical == "h3_v2v"
        source_key = "input_video" if is_v2v else "input_image"
        # 슬롯 → (그래프 입력 키, 업로드 이름 접두어). 그 mode 의 그래프가 읽지 않는 입력은 올리지
        # 않는다(creator_workflows.media_inputs): 아이덴티티는 V2V 전용이라, V2V 에서 고른 사진이
        # 폼에 남은 채 I2V 로 바꾸면 예전엔 그 사진이 시작 프레임(input_image)을 덮어썼다. T2V/T2I
        # 에 남은 원본도 올리지 않는다. 같은 입력 키를 여러 슬롯이 채우면 앞 슬롯이 이긴다.
        try:
            accepted = creator_media_inputs(canonical)
        except CreatorWorkflowError:
            accepted = frozenset()  # 알 수 없는 mode — 아무것도 올리지 않고 build() 가 거부한다
        slots = (
            (("sourcePath", "source_path"), source_key, "creator_source"),
            (("identityPath", "identity_path"), "input_image" if is_v2v else "", "creator_identity"),
            (("referencePath", "reference_path"), "reference_image", "creator_reference"),
            (("imagePath", "image_path"), "input_image", "creator_image"),
            (("videoPath", "video_path"), "input_video", "creator_video"),
        )
        planned: list[tuple[str, str, Path]] = []
        for incoming_names, normalized, prefix in slots:
            values = [params.pop(incoming, "") for incoming in incoming_names]
            path_text = next((str(value) for value in values if value), "")
            if (not path_text or normalized not in accepted
                    or any(key == normalized for key, _prefix, _path in planned)):
                continue
            path = Path(path_text).expanduser().resolve()
            if not path.is_file():
                raise FileNotFoundError(f"입력 파일을 찾을 수 없습니다: {path}")
            self._creator_upload_extension(path)
            planned.append((normalized, prefix, path))
        # 모든 입력을 검증한 뒤에 올린다 — 두 번째 파일이 없거나 확장자가 없을 때 첫 파일만
        # ComfyUI/input 에 남지 않게.
        for normalized, prefix, path in planned:
            params[normalized] = self._creator_upload_file(backend, path, prefix)

        params["generate_audio"] = bool(params.get("includeAudio", params.get("generate_audio", False)))
        if is_v2v:
            params["include_reference_audio"] = bool(params["generate_audio"])

        audio_prompt = str(params.pop("audioPrompt", params.pop("audio_prompt", "")) or "").strip()
        dialogue = str(params.pop("dialogue", "") or "").strip()
        prompt_parts = [str(params.get("prompt", "")).strip()]
        if audio_prompt:
            prompt_parts.append(f"Overall soundscape: {audio_prompt}")
        if dialogue:
            prompt_parts.append(f"Dialogue: {dialogue}")
        negative = str(params.pop("negative", "") or "").strip()
        if negative:
            prompt_parts.append(f"Avoid: {negative}")
        if is_v2v:
            reference_instructions = []
            if params.get("input_image"):
                reference_instructions.append("Use <Picture 1> for subject identity and appearance.")
            reference_instructions.append("Use <Video 1> for motion, staging, and camera movement.")
            if params.get("include_reference_audio"):
                reference_instructions.append("Use <Audio 1> as the synchronized sound reference.")
            prompt_parts.insert(0, " ".join(reference_instructions))
        params["prompt"] = "\n".join(part for part in prompt_parts if part)
        return params

    @staticmethod
    def _creator_check_nodes(built: Dict[str, Any], available: set[str]) -> None:
        # Krea2 T2I/I2I 러너와 같은 한 벌 (core/comfy_choice_resolver).
        from core.comfy_choice_resolver import check_required_nodes
        check_required_nodes(built, available)

    @staticmethod
    def _creator_resolve_comfy_choices(
        built: Dict[str, Any], object_info: Dict[str, Any]
    ) -> None:
        """이식 가능한 모델 경로 → 서버의 정확한 combo 값 (H3 캐시 descriptor 포함).

        구현은 Krea2 러너와 공유하는 core/comfy_choice_resolver.resolve_choices 한 벌.
        """
        from core.comfy_choice_resolver import resolve_choices
        resolve_choices(built, object_info)

    @staticmethod
    def _creator_image_size(data: bytes) -> tuple[int, int]:
        from io import BytesIO
        from PIL import Image

        with Image.open(BytesIO(data)) as image:
            return image.size

    # ── Comic planning, generation, and export ───────────────────────────

    def _comic_studio(self):
        from core.comic_studio import ComicStudio
        from core.storage_paths import user_data_file

        return ComicStudio(
            user_data_file(
                "creator/comic_studio.json",
                legacy_paths="config/comic_studio.json",
            ),
            complete_json=self._comic_ollama_complete if self._creator_ollama_config()[1] else None,
        )

    def _comic_start_plan(self, payload: dict) -> None:
        self._creator_run_thread("comic_plan", self._comic_plan, payload)

    def _comic_plan(self, payload: dict) -> None:
        studio = self._comic_studio()
        document = studio.plan(
            str(payload.get("scene", "")),
            int(payload.get("panelCount", payload.get("panel_count", 3)) or 3),
            str(payload.get("style", payload.get("artStyle", "manga"))),
            str(payload.get("characterLock", payload.get("character_lock", ""))),
        )
        expected = payload.get("expectedRevision")
        document = studio.save(
            document,
            expected_revision=int(expected) if expected is not None else None,
        )
        self._creator_emit("comicStoryboardReady", document.to_dict())
        self._comic_emit_document(document, status="saved", source="plan")
        self._creator_emit("creatorResult", {"ok": True, "mode": "comic_plan"})

    def _comic_save(self, payload: dict) -> None:
        from core.comic_studio import ComicRevisionConflict

        request_id = str(payload.get("requestId", ""))[:100]
        try:
            expected = payload.get("expectedRevision")
            document = self._comic_studio().save(
                payload.get("document", payload),
                expected_revision=int(expected) if expected is not None else None,
            )
            self._comic_emit_document(
                document,
                status="saved",
                source="backend",
                requestId=request_id,
            )
        except ComicRevisionConflict as exc:
            self._comic_emit_document(
                exc.current_document,
                status="conflict",
                source="backend",
                requestId=request_id,
                expectedRevision=exc.expected_revision,
                actualRevision=exc.actual_revision,
                conflictPath=exc.conflict_path,
                error=str(exc),
            )
        except Exception as exc:
            self._comic_emit_document(
                None,
                status="error",
                source="backend",
                requestId=request_id,
                error=str(exc),
            )

    def _comic_emit_document(self, document, *, status: str, **persistence) -> None:
        payload = {
            "document": document.to_dict() if document is not None else None,
            "persistence": {"status": status, **persistence},
        }
        self._creator_emit("comicDocumentChanged", payload)

    def _comic_start_generate_all(self, payload: dict) -> None:
        # 컷 T2I 는 메인 T2I 설정을 따른다 — 위젯은 UI 스레드에서만 읽으므로 여기서 스냅샷.
        request = self._comic_prepare_request(payload, "comic_generate")
        if request is not None:
            self._creator_run_thread("comic_generate", self._comic_generate_all, request)

    @staticmethod
    def _comic_requested_panel_id(payload: dict) -> str:
        return str((payload.get("panel") or {}).get("id", payload.get("panelId", "")))

    def _comic_prepare_request(self, payload: dict, mode: str) -> Optional[dict]:
        """UI 스레드: 컷별 T2I 스냅샷을 떠 워커 payload 에 싣는다. 실패하면 결과 오류를 보내고 None."""
        request = {key: value for key, value in dict(payload or {}).items() if key != _COMIC_T2I_KEY}
        try:
            block = self._comic_t2i_snapshots(request, mode)
            # 워커는 UI 스레드가 정규화한 문서를 그대로 다시 읽는다 — 원문을 따로 정규화하면
            # 정리 후 비는 id('컷')가 매번 다른 무작위 id 를 받아 스냅샷과 어긋났다.
            request["document"] = block.pop("document")
            request[_COMIC_T2I_KEY] = block
        except Exception as exc:
            self._creator_emit(
                "creatorResult",
                {"ok": False, "mode": mode, "requestId": str(request.get("requestId") or "")[:100],
                 "error": str(exc)[:2000]},
            )
            return None
        return request

    def _comic_t2i_snapshots(self, request: dict, mode: str) -> dict:
        """컷마다 메인 T2I payload 빌더(_build_generation_payload)로 요청을 만든다.

        - 빌더가 컷 프롬프트에 LoRA 스택을 붙이고 sampler/steps/cfg·VAE/TE·Hires·NegPiP·
          ADetailer/SAM3·Comfy 프리셋·KREA2 family 를 T2I 와 똑같이 채운다.
        - snapshot=True: 와일드카드·프롬프트 훅은 워커에서(core.comic_generation.run_panel).
        - 랜덤 해상도여도 컷 크기가 흩어지지 않게 첫 스냅샷(또는 요청의 명시 크기)으로 고정한다.
        - ComfyUI 워크플로 상세 설정은 누른 시점 것으로 고정한다(채팅·대기열과 같은 규칙).
        - 스냅샷은 컷 인덱스 순서 목록에 컷 id 를 같이 싣는다(core.comic_generation.panel_entry).
          id 로 키를 잡으면 id 가 겹친 컷끼리 한 스냅샷으로 뭉쳐 다른 컷 프롬프트로 생성됐다.
        - 반환 dict 의 "document" 는 이 정규화 결과 — 호출자가 워커 요청의 문서로 바꿔 싣는다.
        """
        from core import comic_generation as comic_t2i
        from core.comic_studio import panel_generation_payloads

        raw_document = request.get("document", request)
        document = self._comic_studio().normalize(raw_document)
        # 겹친 컷 id 는 여기서 한 번 풀어 워커·저장·결과 panelId 가 모두 유일한 id 를 쓰게 한다
        panel_ids = comic_t2i.unique_panel_ids([panel.id for panel in document.panels])
        for panel, panel_id in zip(document.panels, panel_ids):
            panel.id = panel_id
        panel_payloads = list(panel_generation_payloads(document))
        indexes = list(range(len(panel_payloads)))
        target: Dict[str, Any] = {}
        if mode == "comic_panel":
            # 요청 id 는 같은 원문 문서의 컷 id 로 푼다(정규화로 바뀐 id 도 찾는다).
            # 없는 컷이면 워커가 '생성할 Comic 컷을 찾을 수 없습니다' 로 거부한다
            panel_index = comic_t2i.resolve_panel_index(
                raw_document, panel_ids, self._comic_requested_panel_id(request),
            )
            indexes = [] if panel_index is None else [panel_index]
            target = {"panelIndex": panel_index,
                      "panelId": panel_ids[panel_index] if panel_index is not None else ""}

        builder = getattr(self, "_build_generation_payload", None)
        size = comic_t2i.explicit_size(request)
        entries: list[Dict[str, Any]] = []
        for index in indexes:
            panel_payload = panel_payloads[index]
            if callable(builder):
                snapshot, error = builder(prompt_override=panel_payload["prompt"], snapshot=True)
                if error or snapshot is None:
                    raise ValueError(error or "T2I 설정으로 컷 생성 요청을 만들지 못했습니다")
            else:
                snapshot = comic_t2i.fallback_snapshot(panel_payload["prompt"])
            if size is None:
                size = (int(snapshot.get("width") or comic_t2i.DEFAULT_SIZE),
                        int(snapshot.get("height") or comic_t2i.DEFAULT_SIZE))
            entries.append(comic_t2i.panel_entry(
                index, panel_ids[index], comic_t2i.panel_request(snapshot, panel_payload, size=size),
            ))

        krea2 = any(comic_t2i.is_krea2_snapshot(entry["request"]) for entry in entries)
        model = str(request.get("model", "") or self._creator_current_model())
        needs_checkpoint = getattr(self, "_backend_needs_checkpoint", None)
        if callable(builder) and not krea2 and not model.strip() and callable(needs_checkpoint) and needs_checkpoint():
            raise ValueError("T2I에서 사용할 모델을 먼저 선택하세요.")
        if callable(builder) and entries and not krea2:
            from backends import BackendType, get_backend, get_backend_type

            if get_backend_type() == BackendType.COMFYUI:
                from core.comfy_workflow_controls import snapshot_comfy_payload

                frozen = snapshot_comfy_payload(get_backend(), {}, "txt2img")["_comfy_workflow_snapshot"]
                for entry in entries:
                    entry["request"]["_comfy_workflow_snapshot"] = copy.deepcopy(frozen)
        return {"model": model, "panels": entries, "document": document.to_dict(), **target}

    def _comic_panel_generation_payload(self, payload: dict, document, index: int, panel_payload: dict) -> dict:
        """워커: UI 스레드 스냅샷에서 이 컷의 요청을 꺼낸다.

        스냅샷이 아예 없을 때(T2I 빌더 없는 호스트·직접 호출)만 예전 최소 payload 로 폴백한다.
        스냅샷이 있는데 이 컷(인덱스 + id 둘 다)이 없으면 조용히 기본값·다른 컷 설정으로
        생성하지 않고 오류로 멈춘다.
        """
        from core import comic_generation as comic_t2i

        t2i = payload.get(_COMIC_T2I_KEY)
        if t2i is None:
            size = comic_t2i.explicit_size(payload) or (comic_t2i.DEFAULT_SIZE, comic_t2i.DEFAULT_SIZE)
            return comic_t2i.panel_request(
                comic_t2i.fallback_snapshot(panel_payload["prompt"]), panel_payload, size=size,
            )
        panel_id = document.panels[index].id if 0 <= index < len(document.panels) else ""
        request = comic_t2i.find_panel_request(
            t2i.get("panels") if isinstance(t2i, dict) else None, index, panel_id,
        )
        if request is None:
            raise RuntimeError(f"컷 {index + 1}의 T2I 설정 스냅샷이 없습니다. 다시 생성하세요")
        return request

    def _comic_target_panel_index(self, payload: dict, document) -> int:
        """워커: 단일 컷 생성 대상 인덱스.

        UI 스레드가 푼 인덱스가 있으면 그 자리의 컷 id 까지 맞는지 확인해 쓰고, 스냅샷 없는
        직접 호출이면 요청 원문 기준으로 푼다(core.comic_generation.resolve_panel_index).
        """
        from core import comic_generation as comic_t2i

        panel_ids = [panel.id for panel in document.panels]
        t2i = payload.get(_COMIC_T2I_KEY)
        if isinstance(t2i, dict) and "panelIndex" in t2i:
            index = t2i.get("panelIndex")
            valid = (isinstance(index, int) and not isinstance(index, bool)
                     and 0 <= index < len(panel_ids) and panel_ids[index] == str(t2i.get("panelId", "")))
        else:
            index = comic_t2i.resolve_panel_index(
                payload.get("document", payload), panel_ids, self._comic_requested_panel_id(payload),
            )
            valid = index is not None
        if not valid:
            raise ValueError("생성할 Comic 컷을 찾을 수 없습니다")
        return int(index)

    def _comic_generation_model(self, payload: dict) -> str:
        t2i = payload.get(_COMIC_T2I_KEY)
        if isinstance(t2i, dict) and "model" in t2i:
            return str(t2i.get("model") or "")
        return str(payload.get("model", "") or self._creator_current_model())

    def _comic_wait_for_pending_unload(self) -> None:
        """T2I 워커와 같은 규칙: 직전 '생성 후 언로드' 요청이 끝난 뒤에 보낸다(이 워커 스레드에서)."""
        from core.post_generation import wait_for_pending_unload

        wait_for_pending_unload(cancelled=self._creator_cancel_event.is_set)
        if self._creator_cancel_event.is_set():
            raise RuntimeError("Comic 컷 생성이 취소되었습니다")

    def _comic_generate_all(self, payload: dict) -> None:
        from backends import get_backend
        from core.comic_studio import panel_generation_payloads

        studio = self._comic_studio()
        document = studio.normalize(payload.get("document", payload))
        backend = get_backend()
        model = self._comic_generation_model(payload)
        panel_payloads = list(panel_generation_payloads(document))
        unload = self._creator_should_unload_ollama()
        self._comic_wait_for_pending_unload()
        with self._creator_reserve(backend, "comic_generate", unload):
            for index, panel_payload in enumerate(panel_payloads):
                if self._creator_cancel_event.is_set():
                    raise RuntimeError("Comic 컷 생성이 취소되었습니다")
                generation_payload = self._comic_panel_generation_payload(
                    payload, document, index, panel_payload,
                )

                def _progress(value, maximum, preview=None, panel_index=index):
                    local = value / maximum if maximum else 0
                    overall = int(((panel_index + local) / len(panel_payloads)) * 100)
                    self._creator_emit(
                        "creatorProgress",
                        {
                            "stage": "comic_image",
                            "panelIndex": panel_index,
                            "panelCount": len(panel_payloads),
                            "percent": overall,
                        },
                    )

                result = self._comic_txt2img(backend, model, generation_payload, _progress)
                # 취소로 끊긴 부분 결과를 컷 이미지로 저장하지 않는다
                if self._creator_cancel_event.is_set():
                    raise RuntimeError("Comic 컷 생성이 취소되었습니다")
                if not result.success or not result.image_data:
                    raise RuntimeError(result.error or f"컷 {index + 1} 생성 결과가 없습니다")
                path = self._creator_write_bytes(
                    result.image_data, f"comic_panel_{index + 1}.png", "comic"
                )
                document.panels[index].image_path = path
                document = studio.save(document)
                self._comic_emit_document(document, status="saved", source="generation")

        last_path = document.panels[-1].image_path if document.panels else ""
        self._creator_emit(
            "creatorResult",
            {
                "ok": True,
                "mode": "comic_generate",
                "path": last_path,
                "mediaType": "image",
                "document": document.to_dict(),
            },
        )

    def _comic_txt2img(self, backend, model: str, generation_payload: dict, progress_callback):
        """Comic 컷 T2I — T2I 워커와 같은 라우팅(core.comic_generation.run_panel).

        지연 프롬프트 훅을 풀고, 비공개 키를 떼고, KREA2 면 Krea2 러너·아니면 backend.txt2img.
        받을 수 있는 어댑터에는 이 작업의 취소 확인을 넘긴다 — 예전엔 cancel_check 없이 불러,
        WebUI 가 체크포인트를 바꾸는 동안(_switch_model_if_needed) 누른 취소가 사라졌다.
        """
        from core.comic_generation import run_panel

        return run_panel(
            backend, model, generation_payload, progress_callback,
            cancel_check=self._creator_cancel_event.is_set,
        )

    def _comic_generate_panel(self, payload: dict) -> None:
        from backends import get_backend
        from core.comic_studio import panel_generation_payloads

        studio = self._comic_studio()
        document = studio.normalize(payload.get("document", payload))
        panel_index = self._comic_target_panel_index(payload, document)
        # 결과의 panelId 는 함께 보내는 (정규화된) 문서의 id — 프론트가 그 문서로 컷을 찾는다
        panel_id = document.panels[panel_index].id
        panel_payload = list(panel_generation_payloads(document))[panel_index]
        backend = get_backend()
        generation_payload = self._comic_panel_generation_payload(payload, document, panel_index, panel_payload)
        model = self._comic_generation_model(payload)
        self._comic_wait_for_pending_unload()
        with self._creator_reserve(backend, "comic_panel", self._creator_should_unload_ollama()):
            result = self._comic_txt2img(
                backend,
                model,
                generation_payload,
                self._creator_progress_callback,
            )
        if self._creator_cancel_event.is_set():
            raise RuntimeError("Comic 컷 생성이 취소되었습니다")
        if not result.success or not result.image_data:
            raise RuntimeError(result.error or "Comic 컷 이미지 결과가 없습니다")
        path = self._creator_write_bytes(
            result.image_data, f"comic_panel_{panel_index + 1}.png", "comic"
        )
        document.panels[panel_index].image_path = path
        document = studio.save(document)
        self._comic_emit_document(document, status="saved", source="generation")
        self._creator_emit(
            "creatorResult",
            {
                "ok": True,
                "mode": "comic_panel",
                "panelId": panel_id,
                "path": path,
                "mediaType": "image",
                "document": document.to_dict(),
            },
        )

    def _comic_start_animate_all(self, payload: dict) -> None:
        self._creator_run_thread("comic_animate", self._comic_animate_all, payload)

    def _comic_animate_all(self, payload: dict) -> None:
        from backends import BackendType, get_backend, get_backend_type
        from core.creator_workflows import build

        if get_backend_type() is not BackendType.COMFYUI:
            raise RuntimeError("Living Comic은 ComfyUI 백엔드가 필요합니다")
        studio = self._comic_studio()
        document = studio.normalize(payload.get("document", payload))
        backend = get_backend()
        object_info = self._creator_object_info(backend, self._creator_cancel_event.is_set)
        available = set(object_info.keys())
        unload = self._creator_should_unload_ollama()
        with self._creator_reserve(backend, "comic_animate", unload):
            for index, panel in enumerate(document.panels):
                if self._creator_cancel_event.is_set():
                    raise RuntimeError("Comic 애니메이션이 취소되었습니다")
                if not panel.image_path:
                    raise RuntimeError(f"컷 {index + 1} 이미지가 없습니다")
                uploaded = self._creator_upload_file(backend, Path(panel.image_path), "comic_panel")
                params = dict(payload.get("videoSettings", {}))
                params.update(
                    {
                        "input_image": uploaded,
                        "prompt": panel.motion_prompt or panel.text,
                        "seed": self._creator_random_seed("h3_i2v") if panel.seed < 0 else panel.seed,
                    }
                )
                self._creator_configure_h3_cache("h3_i2v", params, available)
                built = build("h3_i2v", params)
                self._creator_check_nodes(built, available)
                self._creator_resolve_comfy_choices(built, object_info)

                def _progress(value, maximum, preview=None, panel_index=index):
                    local = value / maximum if maximum else 0
                    self._creator_emit(
                        "creatorProgress",
                        {
                            "stage": "comic_video",
                            "panelIndex": panel_index,
                            "panelCount": len(document.panels),
                            "percent": int(((panel_index + local) / len(document.panels)) * 100),
                        },
                    )

                result = self._creator_run_workflow(backend, built, _progress)
                if not result.success:
                    raise RuntimeError(result.error or f"컷 {index + 1} 영상 생성 실패")
                saved = self._creator_save_artifacts(result.artifacts, "comic_video")
                video = next((item for item in saved if item["kind"] in {"video", "animated"}), None)
                if not video:
                    raise RuntimeError(f"컷 {index + 1} 영상 결과가 없습니다")
                panel.video_path = video["path"]
                document = studio.save(document)
                self._comic_emit_document(document, status="saved", source="generation")
        last_path = document.panels[-1].video_path if document.panels else ""
        self._creator_emit(
            "creatorResult",
            {
                "ok": True,
                "mode": "comic_animate",
                "path": last_path,
                "mediaType": "video",
                "document": document.to_dict(),
            },
        )

    def _comic_export_page(self, payload: dict) -> None:
        try:
            data_url = str(payload.get("dataUrl", ""))
            fmt = "webp" if str(payload.get("format", "png")).lower() == "webp" else "png"
            match = re.fullmatch(
                rf"data:image/{fmt};base64,([A-Za-z0-9+/=]+)", data_url
            )
            if not match:
                raise ValueError("내보낼 Comic 이미지 데이터가 올바르지 않습니다")
            data = base64.b64decode(match.group(1), validate=True)
            if not data or len(data) > 50 * 1024 * 1024:
                raise ValueError("Comic 이미지가 비어 있거나 50MB 제한을 초과했습니다")
            path = self._creator_write_bytes(data, f"comic_page.{fmt}", "comic")
            if payload.get("document"):
                document = self._comic_studio().save(payload["document"])
                self._comic_emit_document(document, status="saved", source="export")
            self._creator_emit(
                "creatorResult",
                {
                    "ok": True,
                    "mode": "comic_export",
                    "path": path,
                    "mediaType": "image",
                    "artifacts": [{"kind": "image", "path": path}],
                },
            )
        except Exception as exc:
            self._creator_emit(
                "creatorResult", {"ok": False, "mode": "comic_export", "error": str(exc)}
            )

    def _comic_start_export_living(self, payload: dict) -> None:
        self._creator_run_thread("comic_living_export", self._comic_export_living, payload)

    def _comic_export_living(self, payload: dict) -> None:
        from core.living_comic import render_living_comic

        document = self._comic_studio().normalize(payload.get("document", payload))
        out_dir = self._creator_output_dir("comic")
        result = render_living_comic(
            document.to_dict(),
            out_dir,
            fps=int(payload.get("fps", 8) or 8),
            seconds=float(payload.get("seconds", 4) or 4),
            cancelled=self._creator_cancel_event.is_set,
        )
        artifacts = [
            {"kind": "video" if path.lower().endswith(".mp4") else "animated", "path": path}
            for path in result
        ]
        self._creator_emit(
            "creatorResult",
            {
                "ok": True,
                "mode": "comic_living_export",
                "path": artifacts[0]["path"] if artifacts else "",
                "mediaType": artifacts[0]["kind"] if artifacts else "",
                "artifacts": artifacts,
            },
        )

    # ── Shared helpers ───────────────────────────────────────────────────

    def _creator_save_artifacts(self, artifacts: Iterable, feature: str) -> list:
        saved = []
        for index, artifact in enumerate(artifacts or []):
            self._creator_check_cancelled()
            kind = str(getattr(artifact, "kind", "binary"))
            data = getattr(artifact, "data", None)
            filename = str(getattr(artifact, "filename", "") or f"{feature}_{index + 1}.bin")
            mime = str(getattr(artifact, "mime", "") or mimetypes.guess_type(filename)[0] or "")
            path = str(getattr(artifact, "path", "") or "")
            if data:
                path = self._creator_write_bytes(data, filename, feature)
            self._creator_check_cancelled()
            saved.append(
                {
                    "kind": kind,
                    "path": path.replace("\\", "/"),
                    "filename": Path(path or filename).name,
                    "mime": mime,
                    "metadata": getattr(artifact, "metadata", {}) or {},
                }
            )
        return saved

    def _creator_write_bytes(self, data: bytes, filename: str, feature: str) -> str:
        from core.file_naming import sanitize_filename

        safe_name = sanitize_filename(Path(filename).stem, fallback=feature)
        suffix = Path(filename).suffix.lower() or ".bin"
        directory = self._creator_output_dir(feature)
        stamp = time.strftime("%Y%m%d_%H%M%S")
        candidate = directory / f"{safe_name}_{stamp}{suffix}"
        serial = 2
        while candidate.exists():
            candidate = directory / f"{safe_name}_{stamp}_{serial}{suffix}"
            serial += 1
        temporary = candidate.with_suffix(candidate.suffix + ".writing")
        self._creator_check_cancelled()
        try:
            temporary.write_bytes(data)
            with self._creator_state_lock:
                self._creator_check_cancelled()
                os.replace(temporary, candidate)
        finally:
            temporary.unlink(missing_ok=True)
        return str(candidate).replace("\\", "/")

    @staticmethod
    def _creator_output_dir(feature: str) -> Path:
        from config import OUTPUT_DIR

        safe_feature = re.sub(r"[^a-z0-9_-]", "_", feature.lower())[:60] or "creator"
        directory = Path(OUTPUT_DIR) / "creator" / safe_feature
        directory.mkdir(parents=True, exist_ok=True)
        return directory

    def _creator_current_model(self) -> str:
        widget = getattr(self, "model_combo", None)
        if widget and hasattr(widget, "currentText"):
            return str(widget.currentText() or "")
        return ""

    @staticmethod
    def _creator_prefs() -> dict:
        # 경로·마이그레이션은 core.ui_prefs 한 곳 — 예전엔 여기서 경로를 조립해 json.load 로 따로 읽었다.
        from core.ui_prefs import read_ui_prefs

        return read_ui_prefs()

    def _creator_ollama_config(self) -> tuple[str, str]:
        from core.ollama_client import DEFAULT_OLLAMA_URL

        prefs = self._creator_prefs()
        return (
            str(prefs.get("ollamaUrl", DEFAULT_OLLAMA_URL) or DEFAULT_OLLAMA_URL).rstrip("/"),
            str(prefs.get("ollamaModel", "") or ""),
        )

    def _creator_should_unload_ollama(self) -> bool:
        # 수동 생성(_maybe_unload_ollama)과 같은 판단 규칙 한 벌
        from core.ui_prefs import ollama_unload_target

        return ollama_unload_target(self._creator_prefs()) is not None

    def _creator_unload_ollama(self) -> bool:
        from core.ollama_client import unload_configured_model

        url, model = self._creator_ollama_config()
        # Ollama 가 떠 있지 않으면(설치 목록이 비면) 점유한 VRAM 도 없다 → 언로드 성공으로 본다.
        # ui/generator_generation.py 의 _maybe_unload_ollama 와 같은 의미론
        # ("Ollama 미실행/미설정이면 조용히 무시"). 설정 이름이 설치되지 않은 태그면
        # 태그 강화·NL·Comic 이 실제로 올린 같은 계열의 설치 모델을 골라 내린다(resolve_model).
        return unload_configured_model(url, model)

    def _comic_ollama_complete(self, system: str, user: str) -> str:
        from core.ollama_client import OllamaClient, resolve_installed_model

        url, model = self._creator_ollama_config()
        if not model:
            raise RuntimeError("Settings에서 Comic Director용 Ollama 모델을 선택하세요")
        # 설정 이름이 설치되지 않은 태그(추천 카드만 고르고 pull 안 함)면 그대로 보내면 404 —
        # 태그 강화·NL 워커와 같은 규칙으로 설치 모델에 맞춘다(_creator_run_thread 백그라운드라 HTTP 가능)
        model = resolve_installed_model(url, model)
        # 추론형 모델이 num_predict 3600 을 사고에 다 써 JSON 본문이 비지 않게 think 를 능력에 맞춰 끈다
        return OllamaClient(url, model).complete_chat(
            [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            options={"temperature": 0.35, "num_predict": 3600},
            response_format="json",
            timeout=300,
        )
