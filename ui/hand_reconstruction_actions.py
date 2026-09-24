"""Experimental, opt-in hand reconstruction. No source paths or automatic edits.

An erased, masked context crop is sampled by the selected generation backend;
only pixels inside the user's mask are composited back locally. Candidates are
kept in memory until explicit export. This is not a hand anatomy detector.
"""
from __future__ import annotations

import base64
import copy
from datetime import datetime
import re
import secrets
import threading
import json

from core.hand_reconstruction import prepare_hand_repair, compose_hand_candidate
from core.local_image_io import export_exclusive_png, png_data_url

_ID = re.compile(r"^[A-Za-z0-9_-]{1,100}$")
MAX_CACHED_BYTES = 128 * 1024 * 1024
_GENERATION_KEYS = (
    "prompt", "negative_prompt", "sampler_name", "scheduler", "steps",
    "cfg_scale", "seed", "forge_additional_modules", "distilled_cfg_scale",
    "_chat_deferred_prompt", "alwayson_scripts",
)


def _sampling_scripts(snapshot):
    from core import anima38, anima_guidance
    allowed = {anima38.SCRIPT_NAME, *anima_guidance.SPECS, "NegPiP"}
    return {name: copy.deepcopy(block) for name, block in snapshot.get("alwayson_scripts", {}).items()
            if name in allowed}


def _data_url(png):
    return png_data_url(png)


def hand_generation_payload(prepared, snapshot, seed):
    """An allowlist prevents hires/detailers/custom workflows leaking into repair."""
    payload = {key: copy.deepcopy(snapshot[key]) for key in _GENERATION_KEYS if key in snapshot}
    payload.pop("_chat_deferred_prompt", None)
    payload.update({
        "init_images": [base64.b64encode(prepared.init_png).decode("ascii")],
        "mask": base64.b64encode(prepared.mask_png).decode("ascii"),
        "width": prepared.working_size[0], "height": prepared.working_size[1],
        "denoising_strength": prepared.settings["strength"],
        "seed": seed, "batch_size": 1, "n_iter": 1,
        # 'Original' now means the deliberately erased crop, NOT the bad hand.
        "inpainting_fill": 1, "inpainting_mask_invert": 0,
        "inpaint_full_res": False, "inpaint_full_res_padding": 0,
        "mask_blur": 0, "mask_dilation": 0, "grow_mask_by": 0,
        "resize_mode": 0, "send_images": True, "save_images": False,
        "do_not_save_samples": True, "do_not_save_grid": True,
        "alwayson_scripts": _sampling_scripts(snapshot),
    })
    return payload


def freeze_hand_backend(backend):
    """Do not silently replace a user's custom Comfy graph or mutate its config."""
    kind = backend.get_backend_type()
    if kind == "comfyui":
        if backend._configured_workflow_path("img2img"):
            raise ValueError("손 재구성 실험은 ComfyUI 기본 생성 경로만 지원합니다. 설정에서 I2I 사용자 워크플로를 해제한 뒤 실행하세요.")
        from backends.comfyui_backend import ComfyUIBackend
        # Freeze the explicit empty workflow: a later settings edit cannot
        # substitute a graph that ignores masks after preflight.
        return ComfyUIBackend(backend.api_url, workflow_path="", img2img_workflow_path="")
    if kind != "webui":
        raise ValueError("손 재구성 실험은 Forge/WebUI와 ComfyUI 기본 경로에서만 사용할 수 있습니다.")
    return backend


def export_hand_candidate(png, output_root):
    # datetime/secrets are resolved here at call time so tests can patch this module.
    return export_exclusive_png(
        png, output_root, subdir="hand_reconstruction", prefix="hand",
        now=lambda: datetime.now(), token=lambda: secrets.token_hex(6),
        outside_message="손 재구성 저장 폴더가 앱 출력 폴더 밖을 가리킵니다.",
        exhausted_message="새 결과 파일 이름을 만들지 못했습니다.",
    )


class HandReconstructionActionsMixin:
    def _hand_emit(self, packet):
        if getattr(self, "_hand_closed", False):
            return
        signal = getattr(getattr(self, "vue_bridge", None), "handReconstructionEvent", None)
        if signal is not None:
            try:
                signal.emit(json.dumps(packet, ensure_ascii=False))
            except RuntimeError:
                pass  # Window closed while a backend request was finishing.

    def _hand_error(self, packet, exc):
        from core.error_handler import handle_error, sanitize_for_ui
        handle_error("E020", "실험 손 재구성", exc, notify=False)
        self._hand_emit({**packet, "phase": "complete", "ok": False,
                         "error": sanitize_for_ui(str(exc), 600)})

    def _ensure_hand_runtime(self):
        if not hasattr(self, "_hand_lock"):
            self._hand_lock = threading.RLock()
            self._hand_job = None
            self._hand_preview = None

    def _shutdown_hand_reconstruction(self):
        self._ensure_hand_runtime()
        with self._hand_lock:
            self._hand_closed = True
            self._hand_preview = None
            if self._hand_job:
                self._hand_job["cancel"].set()

    def _hand_snapshot(self, request):
        """Capture widget proxies only on the UI thread; all image/HTTP work waits."""
        from backends import get_backend
        extra = request.get("prompt", "")
        if not isinstance(extra, str) or len(extra) > 4000:
            raise ValueError("손 재구성 추가 지침은 4,000자 이하로 입력하세요.")
        if self._is_krea2_generation():
            raise ValueError("Krea2는 별도 이미지 편집 경로를 사용하므로 이 손 재구성 실험에서 아직 지원하지 않습니다.")
        base_prompt = self.total_prompt_display.toPlainText().strip()
        prompt = "\n".join(part for part in (base_prompt, extra.strip()) if part)
        if not prompt:
            raise ValueError("손의 자세·동작을 설명하는 프롬프트를 입력하세요.")
        model, snapshot = self._chat_generation_snapshot(prompt)
        if snapshot.get("_generation_family") == "krea2":
            raise ValueError("Krea2는 이 손 재구성 실험에서 아직 지원하지 않습니다.")
        snapshot = {key: copy.deepcopy(snapshot[key]) for key in _GENERATION_KEYS if key in snapshot}
        snapshot["alwayson_scripts"] = _sampling_scripts(snapshot)
        negative = str(snapshot.get("negative_prompt") or "").strip()
        snapshot["negative_prompt"] = ", ".join(filter(None, (
            negative, "extra fingers, fused fingers, duplicated hands, malformed hands",
        )))
        return model, snapshot, freeze_hand_backend(get_backend())

    def _handle_hand_reconstruction_action(self, action, payload):
        if action not in {"hand_reconstruction_generate", "hand_reconstruction_export", "hand_reconstruction_cancel"}:
            return False
        if getattr(self, "_hand_closed", False):
            return True
        request = dict(payload) if isinstance(payload, dict) else {}
        request_id = request.get("requestId", "")
        packet = {"requestId": request_id if isinstance(request_id, str) else "", "action": action}
        try:
            if getattr(self, "web_mode", False):
                raise ValueError("손 재구성 실험은 로컬 앱에서만 사용할 수 있습니다.")
            if not isinstance(request_id, str) or not _ID.fullmatch(request_id):
                raise ValueError("작업 식별자가 올바르지 않습니다. 패널을 다시 열어 주세요.")
            self._ensure_hand_runtime()
            with self._hand_lock:
                if action == "hand_reconstruction_cancel":
                    if self._hand_job and self._hand_job["requestId"] == request_id:
                        self._hand_job["cancel"].set()
                    if self._hand_preview and self._hand_preview["requestId"] == request_id:
                        self._hand_preview = None
                    self._hand_emit({**packet, "ok": True, "phase": "cancel_requested"})
                    return True
                if self._hand_job:
                    if self._hand_job["requestId"] == request_id:
                        return True
                    raise ValueError("손 재구성 작업이 실행 중입니다. 현재 요청이 끝난 뒤 다시 시도하세요.")
                if action == "hand_reconstruction_generate":
                    if not isinstance(request.get("settings"), dict) or request["settings"].get("enabled") is not True:
                        raise ValueError("먼저 실험 기능 사용을 켜세요.")
                    model, snapshot, backend = self._hand_snapshot(request)
                    self._hand_preview = None
                    cached_png = output_root = None
                else:
                    cached = self._hand_preview
                    index = request.get("candidateIndex")
                    if (not cached or cached["requestId"] != request.get("previewRequestId")
                            or type(index) is not int or not 0 <= index < len(cached["candidates"])):
                        raise ValueError("현재 미리보기의 후보를 선택하세요. 이전 결과는 다시 생성해야 합니다.")
                    cached_png = cached["candidates"][index]["png"]
                    from config import OUTPUT_DIR
                    output_root = OUTPUT_DIR
                    model = snapshot = backend = None
                job = {"requestId": request_id, "cancel": threading.Event()}
                self._hand_job = job
        except Exception as exc:
            self._hand_error(packet, exc)
            return True

        def work():
            try:
                if action == "hand_reconstruction_export":
                    if not job["cancel"].is_set():
                        path = export_hand_candidate(cached_png, output_root)
                        self._hand_emit({**packet, "ok": True, "phase": "complete", "path": path})
                    return
                prepared = prepare_hand_repair(request)
                from core.chat_generation import prepare_prompt_payload
                values = prepare_prompt_payload(snapshot)
                seed = int(values.get("seed", -1))
                seed = secrets.randbits(32) if seed < 0 else seed % (2 ** 32)
                candidates, warning = [], ""
                # The cache holds only what this process keeps: the prepared crop
                # and composited candidates. The panel already owns the source.
                total_size = len(prepared.prepared_png)
                from core.post_generation import reserve_generation_lease
                from core.resource_coordinator import get_generation_coordinator
                # One lease spans ALL candidates and any in-flight cancellation.
                # Deliberately no global backend interrupt: an external Forge
                # user's work must never be canceled by this experimental panel.
                # 진행 중인 '생성 후 언로드'(같은 리스를 쥔다)는 끝날 때까지 이 작업 스레드에서 기다린다.
                with reserve_generation_lease("hand-reconstruction", coordinator=get_generation_coordinator(),
                                              cancelled=job["cancel"].is_set):
                    for index in range(prepared.settings["candidates"]):
                        if job["cancel"].is_set():
                            break
                        current_seed = (seed + index) % (2 ** 32)
                        params = hand_generation_payload(prepared, values, current_seed)

                        def progress(step, total, _preview=None):
                            if not job["cancel"].is_set():
                                self._hand_emit({**packet, "ok": True, "phase": "progress",
                                                 "candidate": index + 1, "count": prepared.settings["candidates"],
                                                 "step": int(step), "total": int(total)})

                        progress(0, int(params.get("steps", 28)))
                        try:
                            result = backend.img2img(model, params, progress_callback=progress)
                            if job["cancel"].is_set():
                                break  # Drop even a 'successful' late result.
                            if not result.success or not result.image_data:
                                raise ValueError(result.error or "백엔드가 이미지 후보를 반환하지 않았습니다.")
                            png = compose_hand_candidate(prepared, result.image_data, provenance={
                                "model": model, "backend": backend.get_backend_type(), "seed": current_seed,
                                "prompt": params.get("prompt", ""), "negative_prompt": params.get("negative_prompt", ""),
                                "sampler": params.get("sampler_name", ""), "scheduler": params.get("scheduler", ""),
                                "steps": params.get("steps"), "cfg_scale": params.get("cfg_scale"),
                                "modules": params.get("forge_additional_modules", []),
                                "anatomy_verified": False,
                            })
                            total_size += len(png)
                            if total_size > MAX_CACHED_BYTES:
                                raise ValueError("후보의 총 용량이 128 MB를 넘습니다. 원본 크기 또는 후보 수를 줄이세요.")
                            candidates.append({"index": index, "seed": current_seed, "png": png})
                        except Exception as exc:
                            if not candidates:
                                raise
                            from core.error_handler import handle_error, sanitize_for_ui
                            handle_error("E020", "손 재구성 후속 후보", exc, notify=False)
                            warning = sanitize_for_ui(str(exc), 600)
                            break
                with self._hand_lock:
                    if not getattr(self, "_hand_closed", False) and candidates:
                        self._hand_preview = {"requestId": request_id, "candidates": candidates}
                    self._hand_job = None
                    # No 'source': the panel compares against the exact image it
                    # sent with this requestId (re-encoding it here was display-only
                    # and made the event up to ~49 MB of base64).
                    self._hand_emit({**packet, "phase": "complete", "ok": bool(candidates),
                                     "canceled": job["cancel"].is_set(), "warning": warning,
                                     "error": "후보 생성을 취소했습니다." if not candidates else "",
                                     "prepared": _data_url(prepared.prepared_png),
                                     "candidates": [{"index": c["index"], "seed": c["seed"], "image": _data_url(c["png"])} for c in candidates]})
            except Exception as exc:
                self._hand_error(packet, exc)
            finally:
                with self._hand_lock:
                    if self._hand_job is job:
                        self._hand_job = None

        try:
            threading.Thread(target=work, daemon=True, name="ai-studio-hand-reconstruction").start()
        except Exception as exc:
            with self._hand_lock:
                if self._hand_job is job:
                    self._hand_job = None
            self._hand_error(packet, exc)
        return True
