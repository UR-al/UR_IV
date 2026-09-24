# ui/generator_generation.py
"""
이미지 생성 및 자동화 관련 로직
"""
import os
import copy
import time
import random
import json

from config import OUTPUT_DIR
from workers.generation_worker import GenerationFlowWorker
from utils.file_wildcard import resolve_file_wildcards, wildcards_enabled
from utils.wildcard import process_wildcards
from utils.app_logger import get_logger

#: '생성 후 모델 언로드 요청' 을 띄운 뒤 언로드 스레드가 건너뛰었을 때(판단 뒤에 시작된 생성·후처리
#: 작업이 GPU 를 쓰는 중) 그 문구를 바꿔 놓는 상태 한 줄. ⚠ 로 경고 수준(core/status_message.py).
POST_GEN_UNLOAD_SKIPPED_STATUS = "⚠ 생성 후 모델 언로드 건너뜀 — 다른 작업이 GPU 를 쓰는 중"


def _widget_text(w, fallback: str = '') -> str:
    """위젯에서 텍스트 값을 얻는다. proxy/LineEdit/ComboBox/PlainTextEdit 모두 대응.

    우선순위: .text() → _fallback_text → .toPlainText() → .currentText()
    """
    v = w.text() if hasattr(w, 'text') else ''
    if not v and hasattr(w, '_fallback_text'):
        v = w._fallback_text
    if not v and hasattr(w, 'toPlainText'):
        v = w.toPlainText()
    if not v and hasattr(w, 'currentText'):
        v = w.currentText()
    return v or fallback


def _widget_float(w, fallback: float) -> float:
    try:
        v = _widget_text(w)
        return float(v) if v else fallback
    except (ValueError, TypeError):
        return fallback


def _widget_int(w, fallback: int) -> int:
    try:
        v = _widget_text(w)
        return int(float(v)) if v else fallback
    except (ValueError, TypeError):
        return fallback


def _gen_btn_default_color() -> str:
    """생성 버튼 기본 색상"""
    return '#4A90E2'


def _gen_btn_style(bg_color: str) -> str:
    """생성 버튼 스타일"""
    return (
        f"QPushButton {{ font-size: 15px; font-weight: bold; "
        f"background-color: {bg_color}; color: white; "
        f"border: none; border-radius: 20px; padding: 4px; }}"
    )

_logger = get_logger('generation')


class GenerationMixin:
    """이미지 생성 관련 로직을 담당하는 Mixin"""

    def _is_krea2_generation(self) -> bool:
        combo = getattr(self, 'generation_family_combo', None)
        if combo is None or not hasattr(combo, 'currentText'):
            return False
        from core.generation_family import is_krea2_family
        return is_krea2_family(combo.currentText())
    
    def _maybe_unload_ollama(self):
        """생성 직전 Ollama LLM 언로드 (ui_prefs.ollamaUnloadOnGen 켜진 경우) → VRAM 양보.
        best-effort 비동기 — Ollama 미실행/미설정이면 조용히 무시. 자동화·수동 공통 경로.
        판단 규칙은 Creator 와 같은 core.ui_prefs.ollama_unload_target 하나다."""
        try:
            import threading as _th
            from core.ui_prefs import ollama_unload_target, read_ui_prefs
            target = ollama_unload_target(read_ui_prefs())
            if target is None:
                return
            url, model = target

            def _do():
                try:
                    # 설정 이름이 설치되지 않은 태그(추천 카드만 고름)면 태그 강화·NL 워커가 실제로 올린 건
                    # 같은 계열의 설치 모델이다 — 같은 규칙(resolve_model)으로 골라 내린다. /api/tags 도
                    # HTTP 라 이 스레드에서.
                    from core.ollama_client import unload_configured_model
                    unload_configured_model(url, model)
                except Exception:
                    pass
            _th.Thread(target=_do, daemon=True).start()
        except Exception:
            pass

    def start_generation(self, *, payload_override=None, model_override=None, backend_override=None):
        """이미지 생성 시작"""
        worker = getattr(self, 'gen_worker', None)
        try:
            active = worker is not None and worker.isRunning()
        except RuntimeError:
            active = False
        if active and getattr(worker, '_result_emitted', False) is not True:
            message = '이미지 생성 중입니다. 현재 작업이 끝난 뒤 다시 시작하세요.'
            self.show_status(message, 5000)
            if hasattr(self, 'vue_bridge'):
                self.vue_bridge.showNotification.emit('warning', message)
            return False
        # 1) 입력 파싱 + payload 구성 + 검증 — UI를 '생성 중'으로 전환하기 *전에* 수행.
        #    (검증 실패 시 버튼/Vue 스피너가 '생성 중'으로 영구 잔류하던 버그 방지)
        try:
            if payload_override is None:
                payload, err = self._build_generation_payload()
            else:
                from core.payload_validator import PayloadValidator
                payload = copy.deepcopy(payload_override)
                validation = PayloadValidator.validate(payload)
                err = " / ".join(validation.errors)
                if not validation.ok:
                    payload = None
        except Exception as e:
            _logger.exception("payload 구성 실패")
            payload, err = None, str(e)
        if payload is None:
            self._abort_generation(err or '생성 준비 실패')
            return False

        self._maybe_unload_ollama()
        self._gen_start_time = time.time()
        # 상태 표시 업데이트
        self.setWindowTitle("AI Studio - Pro [생성 중...]")
        self.btn_generate.setText("⏳ 생성 중...")
        self.btn_generate.setEnabled(False)
        self.btn_generate.setStyleSheet(_gen_btn_style('#e67e22'))

        # 상태바 업데이트
        self.show_status("🎨 이미지 생성 중...")

        # 뷰어(Vue)에 로딩 표시
        if hasattr(self, 'vue_bridge'):
            self.vue_bridge.send_start()

        _logger.info("Sending Payload to WebUI API")
        _logger.debug(f"프롬프트: {payload['prompt'][:100]}...")

        selected_model = self.model_combo.currentText() if model_override is None else model_override
        if not str(selected_model or '').strip() and self._backend_needs_checkpoint():
            self._abort_generation("체크포인트가 아직 선택되지 않았습니다 — 목록이 로딩 중이면 잠시 후 다시 누르세요")
            return False
        self._cleanup_gen_worker()
        # '생성 후 언로드' 요청이 아직 날아가는 중이면 워커가 run() 초입에서 기다린다
        # (core.post_generation.wait_for_pending_unload) — UI 스레드는 막지 않는다.
        if backend_override is None:
            self.gen_worker = GenerationFlowWorker(selected_model, payload)
        else:
            self.gen_worker = GenerationFlowWorker(selected_model, payload, backend=backend_override)
        # 통계는 UI 위젯이 아니라 실제 요청(hires 배율·Anima 가드·XYZ/Comfy override 반영)으로 기록
        from core.gen_stats import request_meta_from_payload
        self._gen_request_meta = request_meta_from_payload(selected_model, payload)
        self.gen_worker.finished.connect(self.on_generation_finished)
        self.gen_worker.progress.connect(self._on_generation_progress)

        self.gen_worker.start()
        return True

    @staticmethod
    def _backend_needs_checkpoint() -> bool:
        """Forge/A1111 은 요청에 체크포인트가 있어야 한다. ComfyUI 는 워크플로가 모델을 정하므로 비어도 된다."""
        try:
            from backends import BackendType, get_backend_type
            return get_backend_type() == BackendType.WEBUI
        except Exception:
            return True

    def _abort_generation(self, msg: str):
        """생성 시작 실패 — UI 복구 + 에러 통지 + 자동화 명시적 중지.
        (기존엔 bare return으로 스피너 잔류 + 자동화가 조용히 멈췄음)"""
        _logger.error("generation aborted: %s", msg)
        try:
            self._restore_generate_button()
        except Exception:
            pass
        self.show_status(f"설정 오류: {msg}", 5000)
        if hasattr(self, 'vue_bridge'):
            # generationError → App.vue가 에러 토스트 + isGenerating(스피너) 리셋
            self.vue_bridge.generationError.emit(f'설정 오류: {msg}')
        # 자동화 중 검증 실패는 다음 사이클도 같은 이유(설정 문제)로 실패 →
        # 무한 루프/조용한 정지 대신 사유를 보여주며 중지
        if getattr(self, 'is_automating', False):
            try:
                self._stop_automation(f"설정 오류로 자동화 중지: {msg}")
            except Exception:
                self.is_automating = False

    def _chat_generation_snapshot(self, prompt: str):
        """Capture current generation controls on the UI thread without writes.

        Chat resolves wildcard files and prompt hooks later in its own worker.
        It must remove ``_chat_deferred_prompt`` before handing off to an API.
        """
        import copy

        model = str(self.model_combo.currentText() or '').strip()
        if not model:
            raise ValueError("T2I에서 사용할 모델을 먼저 선택하세요.")
        payload, error = self._build_generation_payload(prompt_override=prompt, snapshot=True)
        if error:
            raise ValueError(error)
        return model, copy.deepcopy(payload)

    def _build_generation_payload(self, *, prompt_override=None, snapshot=False, comfy_workflow_snapshot=None):
        """입력 위젯 → 검증된 payload. 성공 시 (payload, None), 실패 시 (None, 사유).
        UI 상태는 건드리지 않음 — start_generation이 검증 통과 후에만 busy 전환."""
        # 해상도 결정
        if self.random_res_check.isChecked() and self.random_resolutions:
            width, height, _ = random.choice(self.random_resolutions)
            if not snapshot:
                self.width_input.setText(str(width))
                self.height_input.setText(str(height))
        else:
            width = _widget_int(self.width_input, 1024)
            height = _widget_int(self.height_input, 1024)

        # 고해상도/자동해상도 + 설정 가능한 ANIMA 면적/한변 캡 — 순수 함수로 분리.
        # 동작/테스트: core/resolution_guard.py, tests/test_resolution_guard.py
        hr_factor = getattr(self, '_high_res_factor', 1.0) or 1.0
        _auto_res = bool(getattr(self, 'auto_res_check', None)
                         and self.auto_res_check.isChecked())
        from core.resolution_guard import (
            ANIMA_MAX_AREA, ANIMA_MAX_SIDE, apply_anima_resolution,
        )
        _guard_enabled = bool(getattr(self, '_anima_guard_enabled', True))
        _guard_area = int(getattr(self, '_anima_guard_max_area', ANIMA_MAX_AREA))
        _guard_side = int(getattr(self, '_anima_guard_max_side', ANIMA_MAX_SIDE))
        _bw, _bh = width, height
        width, height = apply_anima_resolution(
            _bw, _bh, hr_factor, _auto_res,
            max_area=_guard_area, max_side=_guard_side,
            enabled=_guard_enabled,
        )
        if (width, height) != (_bw, _bh):
            _guard_desc = (f"area≤{_guard_area:,}, side≤{_guard_side}"
                           if _guard_enabled else "OFF")
            print(f"[HighRes] base {_bw}x{_bh} (hr={hr_factor}, auto={_auto_res}) "
                  f"→ {width}x{height} (Anima Guard {_guard_desc})")
        
        combined_neg_prompt = self.neg_prompt_text.toPlainText().strip()

        # 와일드카드 치환
        final_prompt = self.total_prompt_display.toPlainText() if prompt_override is None else str(prompt_override)
        wc_enabled = wildcards_enabled(self)
        if wc_enabled and not snapshot:
            final_prompt = resolve_file_wildcards(final_prompt)
            final_prompt = process_wildcards(final_prompt)
            combined_neg_prompt = resolve_file_wildcards(combined_neg_prompt)
            combined_neg_prompt = process_wildcards(combined_neg_prompt)

        # PR 1: PromptPipeline 컨버전스 — 등록된 모든 훅을 final_prompt에 적용
        # 기존 처리 *뒤*에 호출되므로 비파괴적. instant_wildcards($$name$$),
        # standard_dedupe, 사용자 추가 훅 등이 여기서 동작.
        try:
            if not snapshot:
                from core.standard_hooks import run_pipeline_on_text
                final_prompt = run_pipeline_on_text(final_prompt)
                combined_neg_prompt = run_pipeline_on_text(combined_neg_prompt)
        except Exception as e:
            _logger.warning(f"pipeline 실행 실패 (원본 유지): {e}")

        # LoRA 합침 — 단일 소스 _vue_lora_entries(배율 단위, Vue set_lora_stack·부팅 복원·프로파일
        # 적용이 갱신)에서 매번 파생한다. 스택이 비었거나 전부 꺼져 있으면 아무것도 붙이지 않는다.
        # 프롬프트에 같은 <lora:NAME...>이 이미 있으면(큐 항목의 EXIF 로라 등) 그 LoRA는 제외.
        if not self._is_krea2_generation():
            from core.lora_stack import UNIT_MULTIPLIER, append_lora_stack_to_prompt
            final_prompt = append_lora_stack_to_prompt(
                final_prompt, getattr(self, '_vue_lora_entries', None) or [], unit=UNIT_MULTIPLIER)

        # Payload 생성
        payload = {
            "prompt": final_prompt,
            "negative_prompt": combined_neg_prompt,
            "sampler_name": self.sampler_combo.currentText(),
            "scheduler": self.scheduler_combo.currentText(),
            "steps": _widget_int(self.steps_input, 28),
            "cfg_scale": _widget_float(self.cfg_input, 7.0),
            "seed": _widget_int(self.seed_input, -1),
            "width": width,
            "height": height,
            "send_images": True,
            # '저장해도 되는 결과'라는 요청일 뿐 — Forge 가 실제로 자기 output 에도 저장할지는
            # WebUI 백엔드가 설정 forgeSaveOutputs(기본 off)로 확정한다(core/forge_output_policy.py).
            "save_images": True,
            "alwayson_scripts": {}
        }

        # 메인 VAE / TE → Forge Neo의 forge_additional_modules로 통합 전달
        # (override_settings.sd_vae와 병행 시 Forge processing._vae_override 충돌)
        extra_modules = self._build_vae_te_override()
        if extra_modules:
            payload["forge_additional_modules"] = extra_modules

        # Shift (Distilled CFG Scale)
        shift_val = _widget_float(self.shift_input, 0.0)
        if shift_val > 0:
            payload["distilled_cfg_scale"] = shift_val

        # Hires.fix
        if self.hires_options_group.isChecked():
            hr_scale = _widget_float(self.hires_scale_input, 2.0)
            if hr_scale <= 0:
                hr_scale = 2.0
            # Hires 패스에서 모듈 처리:
            # - 베이스에 forge_additional_modules가 있으면 "Use same choices"로 reload 회피
            #   (실제 리스트를 넘기면 Forge processing.py:1404 modules_change가 True를
            #    반환해 forge_model_reload() 호출 → BufferError 발생)
            # - 없으면 빈 리스트 (Built-in)
            if payload.get("forge_additional_modules"):
                hr_modules = ["Use same choices"]
            else:
                hr_modules = []
            hr_payload = {
                "enable_hr": True,
                "hr_upscaler": self.upscaler_combo.currentText(),
                "hr_second_pass_steps": _widget_int(self.hires_steps_input, 0),
                "denoising_strength": _widget_float(self.hires_denoising_input, 0.5),
                "hr_scale": hr_scale,
                "hr_additional_modules": hr_modules,
            }
            hr_cfg = _widget_float(self.hires_cfg_input, 0.0)
            if hr_cfg > 0:
                hr_payload["hr_cfg"] = hr_cfg

            # Hires Checkpoint
            hr_ckpt = self.hires_checkpoint_combo.currentText()
            if hr_ckpt and hr_ckpt != "Use same checkpoint":
                hr_payload["hr_checkpoint_name"] = hr_ckpt

            # Hires Sampler
            hr_sampler = self.hires_sampler_combo.currentText()
            if hr_sampler and hr_sampler != "Use same sampler":
                hr_payload["hr_sampler_name"] = hr_sampler

            # Hires Scheduler
            hr_scheduler = self.hires_scheduler_combo.currentText()
            if hr_scheduler and hr_scheduler != "Use same scheduler":
                hr_payload["hr_scheduler"] = hr_scheduler

            # Hires Prompt / Negative Prompt
            hr_prompt = self.hires_prompt_text.toPlainText().strip()
            if hr_prompt:
                hr_payload["hr_prompt"] = hr_prompt
            hr_neg = self.hires_neg_prompt_text.toPlainText().strip()
            if hr_neg:
                hr_payload["hr_negative_prompt"] = hr_neg

            payload.update(hr_payload)

        # NegPiP
        if hasattr(self, 'negpip_group') and self.negpip_group.isChecked():
            payload["alwayson_scripts"]["NegPiP"] = {"args": [True]}

        self._apply_postprocess_chain(payload)

        # Opt-in Comfy sampler experiments are snapshotted with the queued job.
        # Never leak provider-specific switches into Forge or Krea2 requests.
        from backends import BackendType, get_backend_type, get_backend
        if get_backend_type() == BackendType.COMFYUI and not self._is_krea2_generation():
            from core.ui_prefs import read_ui_prefs
            from core.spectrum_settings import spectrum_payload_from_prefs
            from ui.comfy_workflow_actions import quality_preset_payload
            try:
                payload.update(spectrum_payload_from_prefs(read_ui_prefs()))
                payload.update(quality_preset_payload(payload, host=self, endpoint=get_backend().api_url))
                if prompt_override is None:
                    if comfy_workflow_snapshot is not None:
                        payload['_comfy_workflow_snapshot'] = copy.deepcopy(comfy_workflow_snapshot)
                    else:
                        from core.comfy_workflow_controls import snapshot_comfy_payload
                        payload = snapshot_comfy_payload(get_backend(), payload, 'txt2img')
                    payload['_comfy_model_snapshot'] = self.model_combo.currentText()
            except ValueError as exc:
                return None, str(exc)

        # payload 사전 검증 — 잘못된 값은 API 호출 전에 사용자에게 안내
        from core.payload_validator import PayloadValidator
        vr = PayloadValidator.validate(payload)
        for w in vr.warnings:
            _logger.warning("payload warning: %s", w)
        if not vr.ok:
            msg = " / ".join(vr.errors)
            _logger.error("payload invalid: %s", msg)
            return None, msg

        if self._is_krea2_generation():
            payload["_generation_family"] = "krea2"
        if snapshot:
            payload["_chat_deferred_prompt"] = {"wildcards": bool(wc_enabled)}
        return payload, None

    def _build_vae_te_override(self) -> list:
        """메인 VAE + TE 파일들을 Forge Neo의 ``forge_additional_modules``
        포맷(파일명 리스트)으로 합쳐서 반환.

        VAE와 TE 모두 비어있으면 빈 리스트 반환.

        주의: ``override_settings.sd_vae``를 같이 보내면 Forge processing.py에서
        ``_vae_override`` 튜플 언팩 에러가 발생하므로, VAE도 이 리스트로만 전달한다.
        Forge가 파일명을 ``models/VAE/`` 또는 ``models/text_encoder/``에서 자동 매칭.
        """
        modules: list[str] = []

        vae = ''
        try:
            vae = (self.vae_main_combo.currentText() or '').strip()
        except Exception:
            pass
        if vae and vae not in ("Use checkpoint default", "Use same VAE", ""):
            modules.append(vae)

        te_raw = ''
        try:
            te_raw = (self.te_main_input.text() or '').strip()
        except Exception:
            pass
        if te_raw:
            modules.extend(t.strip() for t in te_raw.split(',') if t.strip())

        return modules

    def _cleanup_gen_worker(self):
        """이전 생성 워커 정리 — 시그널을 시그널별로 분리 해제하여 부분 실패 시에도 누수 방지."""
        worker = getattr(self, 'gen_worker', None)
        if worker is None:
            return
        for sig_name in ('finished', 'progress'):
            try:
                sig = getattr(worker, sig_name, None)
                if sig is not None:
                    sig.disconnect()
            except (TypeError, RuntimeError):
                # 이미 연결 해제됐거나 C++ 객체가 삭제된 경우
                pass
        try:
            if worker.isRunning():
                # cancel() = 취소 플래그 + 백엔드 interrupt → 블로킹 HTTP가 곧 반환되어
                # run()이 자연 종료됨. terminate(스레드 강제 종료, 락 잡은 채 죽을 수
                # 있음)는 최후 수단으로만.
                if hasattr(worker, 'cancel') and getattr(worker, '_result_emitted', False) is not True:
                    worker.cancel()
                worker.quit()
                if not worker.wait(5000):
                    _logger.warning("gen_worker wait timeout — terminate 호출")
                    worker.terminate()
                    worker.wait(500)
        except RuntimeError:
            pass
        self.gen_worker = None

    def _on_generation_progress(self, step: int, total: int, preview):
        """생성 진행률 업데이트"""
        if total <= 0:
            return

        # ETA 계산 — worker._start_time 기준 경과시간으로 남은 시간 추정
        eta_str = ""
        try:
            import time
            start_time = getattr(self.gen_worker, '_start_time', None)
            if start_time and step > 0:
                elapsed = time.monotonic() - start_time
                per_step = elapsed / step
                remaining = max(0.0, per_step * (total - step))
                if remaining >= 60:
                    eta_str = f" · ETA {int(remaining // 60)}m{int(remaining % 60):02d}s"
                else:
                    eta_str = f" · ETA {remaining:.0f}s"
        except Exception:
            eta_str = ""

        pct = int(step / total * 100)
        self.setWindowTitle(f"AI Studio - Pro [{step}/{total} steps · {pct}%{eta_str}]")
        self.show_status(f"🎨 생성 중... {step}/{total} steps ({pct}%){eta_str}")

        # Vue에 진행률 전달 + 라이브 프리뷰(바뀐 것만 — 백엔드가 이미 같은 그림은 None 으로 준다)
        if hasattr(self, 'vue_bridge'):
            self.vue_bridge.generationProgress.emit(step, total)
            if isinstance(preview, str) and preview:
                self.vue_bridge.generationPreview.emit(preview)

    def _restore_generate_button(self):
        """생성 버튼/타이틀을 idle 상태로 복구 (완료/취소 공용)."""
        if self.btn_auto_toggle.isChecked():
            if self.is_automating:
                self.btn_generate.setText("⏸️ 자동화 중지")
                self.btn_generate.setStyleSheet(_gen_btn_style('#e74c3c'))
            else:
                self.btn_generate.setText("🚀 자동화 시작")
                self.btn_generate.setStyleSheet(_gen_btn_style('#27ae60'))
        else:
            self.btn_generate.setText("✨ 이미지 생성")
            self.btn_generate.setStyleSheet(_gen_btn_style(_gen_btn_default_color()))
        self.btn_generate.setEnabled(True)
        self.setWindowTitle("AI Studio - Pro")

    def _generation_matches_queue(self, gen_info):
        """Only a matching XYZ result may consume its queued snapshot.

        대기열 매니저가 결과를 기다리는 항목(보낸 항목 · 중지 뒤에도 워커가 만드는 항목)과 비교한다.
        중지된 매니저라도 그런 항목이 있으면 결과를 넘긴다 — 멈춘 채 끝난 장을 정리해야(만들어졌으면
        소비, 아니면 표시 해제) 다음 시작 때 같은 장을 또 만들지 않는다.
        """
        manager = getattr(self, 'queue_manager', None)
        if manager is None:
            return False
        expects = getattr(manager, 'expects_result', None)
        awaiting = bool(expects()) if callable(expects) else False
        if not manager.is_running and not awaiting:
            return False
        awaited = getattr(manager, 'awaited_item', None)
        current = awaited() if (awaiting and callable(awaited)) else None
        if current is None:
            current = manager.queue_panel.get_first_item()
        expected = current.get('_xyz_info') if isinstance(current, dict) else None
        actual = gen_info.get('_xyz_info') if isinstance(gen_info, dict) else None
        if expected or actual:
            return bool(
                isinstance(expected, dict) and isinstance(actual, dict)
                and expected.get('requestId')
                and actual.get('requestId') == expected.get('requestId')
                and actual.get('index') == expected.get('index')
            )
        return True

    def _generation_stats_meta(self) -> dict:
        """통계용 요청 메타 — start_generation 이 저장한 실제 요청값. 없으면 위젯 폴백(_widget_int)."""
        meta = getattr(self, '_gen_request_meta', None)
        if isinstance(meta, dict):
            return meta
        model_combo = getattr(self, 'model_combo', None)
        width_input = getattr(self, 'width_input', None)
        height_input = getattr(self, 'height_input', None)
        return {
            'model': model_combo.currentText() if model_combo is not None else '',
            'width': _widget_int(width_input, 0) if width_input is not None else 0,
            'height': _widget_int(height_input, 0) if height_input is not None else 0,
            'seed': None,
        }

    def on_generation_finished(self, result, gen_info):
        """생성 완료 처리"""
        # 버튼/타이틀 복구 (자동화 모드에 따라 다르게)
        self._restore_generate_button()

        # The request never reached a backend. This is not a failed image and
        # must not consume the queued snapshot or advance automation.
        if isinstance(gen_info, dict) and gen_info.get('_queue_deferred'):
            manager = getattr(self, 'queue_manager', None)
            by_automation = False
            if manager is not None and self._generation_matches_queue(gen_info):
                # 일시정지 + '보낸 항목' 표시 해제 — 항목은 남고 재개 때 다시 나간다
                deferred = getattr(manager, 'on_generation_deferred', None)
                (deferred if callable(deferred) else manager.pause)()
            elif getattr(self, 'is_automating', False):
                # 자동화 '큐 우선'이 보낸 동결 항목(시드 탐색·XYZ·ComfyUI 스냅숏) — 자동화 중엔 대기열
                # 매니저가 멈춰 있어 위 분기가 받지 않는다. 자동화가 같은 규칙(항목은 남기고 일시정지)으로
                # 맡아야 '실행 중'인 채 조용히 멈추지 않는다(재개하면 같은 항목부터 다시 낸다).
                on_deferred = getattr(self, '_automation_queue_item_deferred', None)
                by_automation = bool(callable(on_deferred) and on_deferred(str(result)))
            if not by_automation:   # 자동화 쪽은 _automation_queue_item_deferred 가 알렸다
                self.show_status(f"대기열 일시정지: {result}", 5000)
            if hasattr(self, 'vue_bridge'):
                self.vue_bridge.generationError.emit(str(result))
            return
        
        # 취소 분기 — 에러(E020)/실패 통계/자동화 재시도로 처리하지 않음
        if isinstance(gen_info, dict) and gen_info.get('cancelled'):
            self.show_status("⏹ 생성 취소됨", 3000)
            if hasattr(self, 'vue_bridge'):
                # Vue 스피너 리셋 (✕로 이미 리셋된 경우 무해)
                self.vue_bridge.generationError.emit('생성 취소됨')
            if self._generation_matches_queue(gen_info):
                self.queue_manager.on_generation_completed(False)
            return

        if isinstance(result, bytes):
            self._auto_retry_count = 0   # 성공 — 자동화 재시도 카운터 리셋
            self._process_new_image(result, gen_info)
            self.show_status("✅ 이미지 생성 완료!")

            # 프롬프트 히스토리 기록
            try:
                from utils.prompt_history import add_entry
                add_entry(
                    self.total_prompt_display.toPlainText(),
                    self.neg_prompt_text.toPlainText()
                )
            except Exception:
                pass

            # 생성 통계 기록 — 실제 요청 메타(start_generation 에서 저장) 기준
            try:
                from core.gen_stats import build_generation_record, get_gen_stats
                duration = round(time.time() - getattr(self, '_gen_start_time', time.time()), 1)
                get_gen_stats().record(build_generation_record(
                    success=True, duration_sec=duration,
                    request_meta=self._generation_stats_meta(), gen_info=gen_info,
                ))
            except Exception:
                _logger.debug("생성 통계 기록 실패", exc_info=True)

            # 비활성 창이면 알림 (단일 생성, 비자동화)
            if not self.is_automating and not self.isActiveWindow():
                self._notify_generation_done()

            # 자동화 중이면 카운트 증가 — 덱 장만 센다(큐 우선 항목 제외: _automation_after_generation)
            if self.is_automating:
                self._automation_after_generation(True)
                self.show_status(
                    f"🔄 자동 생성 중... ({self.auto_gen_count}장 완료)"
                )
        else:
            error_msg = f"[E020] 생성 실패: {result}"
            # 실패 통계 기록 (설계상 model 만 — 해상도·시드는 남기지 않는다)
            try:
                from core.gen_stats import build_generation_record, get_gen_stats
                duration = round(time.time() - getattr(self, '_gen_start_time', time.time()), 1)
                get_gen_stats().record(build_generation_record(
                    success=False, duration_sec=duration,
                    request_meta=self._generation_stats_meta(), gen_info=gen_info,
                ))
            except Exception:
                _logger.debug("실패 통계 기록 실패", exc_info=True)
            self.show_status(error_msg, 5000)
            print(f"\n[E020] Generation Failed: {result}")
            if hasattr(self, 'vue_bridge'):
                # generationError → App.vue가 에러 토스트 + isGenerating(스피너) 리셋
                # (단일 생성 실패 시 스피너가 안 풀리던 버그 수정)
                self.vue_bridge.generationError.emit(error_msg)
            # 자동화 실패 재시도 (max_retries + 지수 백오프) — 같은 장을 반복 카운터·자연어
            # 변환 없이 다시 낸다(generator_actions._automation_after_generation).
            if self.is_automating and self._automation_after_generation(False):
                return   # 큐 진행/다음 사이클 스킵 — 재시도가 이어받음

        # 대기열 매니저에 생성 완료 알림
        if self._generation_matches_queue(gen_info):
            self.queue_manager.on_generation_completed(isinstance(result, bytes))

        # ★★★ 자동화 계속 (generator_actions.py의 메서드 호출) ★★★
        if self.is_automating:
            self._continue_automation()
        else:
            # 설정 '생성 후 모델 언로드' — 연속 작업의 마지막 장 뒤에만 (core/post_generation)
            self._maybe_unload_models_after_generation()

    def _maybe_unload_models_after_generation(self):
        """ui_prefs.unloadModelsAfterGen 이 켜져 있고 자동화/대기열이 끝났으면 백엔드 모델을 내린다.

        호출 지점: 단일 생성 완료 · 자동화 종료(_stop_automation) · 대기열 종료(_on_queue_completed).
        HTTP 호출(Forge unload-checkpoint / ComfyUI /free)은 UI 스레드를 막지 않게 데몬
        스레드에서 보내고, 다음 생성은 생성 워커(T2I·PNG Info 즉시 생성·I2I/인페인트·채팅)가
        run() 초입에서 이 요청을 기다린 뒤 시작한다(샘플링 도중 언로드가 끼어드는 경쟁 방지 —
        core.post_generation.wait_for_pending_unload). 실패해도 생성 결과에는 영향 없다.
        ``gen_worker`` 밖의 생성(인페인트·I2I 탭·편집기·채팅·Creator·생성 API)이 공유 GPU 리스를
        쥐고 있으면 내리지 않는다 — 인페인트 중 T2I 를 눌러 곧바로 '사용 중' 실패한 경우나, 대기열
        끝 1초 사이·자동화 중지 직후 인페인트가 시작된 경우. 언로드 스레드도 리스를 잡은 동안에만
        HTTP 를 보내 그 사이에 시작된 작업과도 겹치지 않는다(core.post_generation).
        리스를 잡지 않는 후처리 작업(ADetailer·SAM3·Refine·배치 업스케일, backend_job)이 도는 중에도
        내리지 않는다(unload_blocked). 판단 뒤에 시작된 작업 때문에 언로드 스레드가 건너뛰면 앞서 띄운
        '요청' 문구를 건너뜀 문구로 바꾼다.
        """
        try:
            from core.post_generation import (
                UNLOAD_SKIPPED_BUSY, should_unload_after_generation, start_post_generation_unload,
            )
            from core.resource_coordinator import get_generation_coordinator
            from core.safe_print import safe_print
            from core.ui_prefs import read_ui_prefs
            # 완료 시점에 다시 읽는다 — 긴 대기열·자동화 도중 바뀐 unloadModelsAfterGen 을 따라야 한다
            prefs = read_ui_prefs()
            queue = getattr(self, 'queue_manager', None)
            worker = getattr(self, 'gen_worker', None)
            if not should_unload_after_generation(
                prefs,
                automating=bool(getattr(self, 'is_automating', False)),
                queue_running=bool(getattr(queue, 'is_running', False)),
                worker_running=bool(worker is not None and hasattr(worker, 'isRunning') and worker.isRunning()),
                generation_active=get_generation_coordinator().unload_blocked(),
            ):
                return
            from backends import get_backend
            backend = get_backend()

            def _report(ok):
                # 언로드 스레드에서 불린다 — show_status 는 워커 스레드에서도 안전하다(ui/status_line.py 가
                # GUI 스레드로 넘긴다). 콘솔 로그는 safe_print(cp949 파이프에서도 예외 없음).
                if ok is UNLOAD_SKIPPED_BUSY:
                    safe_print("[PostGen] 생성 후 모델 언로드 건너뜀 - 다른 작업이 GPU 를 쓰는 중", flush=True)
                    self.show_status(POST_GEN_UNLOAD_SKIPPED_STATUS, 5000)
                    return
                safe_print(f"[PostGen] 생성 후 모델 언로드 {'요청됨' if ok else '실패(무시)'}", flush=True)

            # 앞선 언로드가 아직 진행 중이면 새로 보내지 않는다(None)
            if start_post_generation_unload(backend, on_done=_report) is not None:
                self.show_status("생성 후 모델 언로드 요청", 3000)
        except Exception as e:
            print(f"[PostGen] unload check skipped: {e}")

    def _process_new_image(self, image_data, gen_info):
        """새 이미지 저장 → Vue 뷰어·히스토리에 알림 → XYZ 결과 통지.

        해상도(App.vue 해상도 표시)는 결과 바이트의 헤더만 읽는다(core.result_image) —
        gen_info 의 width/height 는 요청값이라 hires fix·업스케일 결과와 다르다(폴백 전용).
        예전엔 장마다 GUI 스레드에서 QPixmap 풀 디코드 + 스무스 축소를 no-op 뷰어 더미에 넘기고,
        아무도 읽지 않는 150px 평면 썸네일(PIL 재디코드)과 보이지 않는 ThumbnailItem(최대 100개)을
        만들고, 읽는 곳 없는 generation_data 를 쌓았다(장당 60~210ms). Vue 히스토리 썸네일은
        generateThumbnails(core.thumb_prefetch)가 작업자 스레드에서 따로 만든다.
        """
        filename = f"generated_{int(time.time())}_{random.randint(100,999)}.png"
        filepath = os.path.join(OUTPUT_DIR, filename)
        with open(filepath, "wb") as f:
            f.write(image_data)

        self.current_image_path = filepath
        info = gen_info if isinstance(gen_info, dict) else {}

        # Vue 뷰어에 이미지 전달
        if hasattr(self, 'vue_bridge'):
            from core.result_image import result_image_size
            w, h = result_image_size(image_data, info)
            self.vue_bridge.send_image(filepath, w, h, info.get('seed', 0))

        # XYZ Plot 결과 전달 — _xyz_info 는 XYZ 대기열 항목이 gen_info 에 싣는다(ui/xyz_actions)
        xyz_info = info.get('_xyz_info')
        if isinstance(xyz_info, dict) and xyz_info and hasattr(self, '_xyz_emit'):
            self._xyz_emit('xyzPlotEvent', {"type": "result", "ok": True,
                "requestId": xyz_info.get('requestId', ''), "path": filepath,
                "label": xyz_info.get('label', ''), "axes": xyz_info.get('axes', {})})

    def _build_adetailer_args(self):
        """활성화된 ADetailer 슬롯 args 생성"""
        args = [True, False]

        if self.ad_slot1_group.isChecked():
            args.append(self._build_adetailer_slot(self.s1_widgets))

        if self.ad_slot2_group.isChecked():
            args.append(self._build_adetailer_slot(self.s2_widgets))

        return args if len(args) > 2 else None

    def apply_alwayson_extensions(self, payload):
        """확장 전체(NegPiP + ADetailer + SAM3 + Anima)를 payload에 적용.

        t2i 뿐 아니라 **img2img / inpaint 경로에서도 쓰는 단일 진입점**이다.
        확장의 `show()`가 `AlwaysVisible`이라 img2img에서도 그대로 동작하므로,
        Forge Neo UI에서 i2i 탭에 SAM3/ADetailer 아코디언이 보이는 것과 결과가 같아진다.
        (CLAUDE.md '확장기능은 Forge Neo와 동일한 alwayson_scripts 방식' 원칙)

        예전에는 i2i/inpaint가 `"alwayson_scripts": {}`를 빈 채로 보내서
        같은 설정인데도 t2i와 결과가 달랐다.
        """
        if not isinstance(payload, dict):
            return payload
        payload.setdefault("alwayson_scripts", {})
        if hasattr(self, 'negpip_group') and self.negpip_group.isChecked():
            payload["alwayson_scripts"].setdefault("NegPiP", {"args": [True]})
        self._apply_postprocess_chain(payload)
        from backends import BackendType, get_backend_type, get_backend
        if get_backend_type() == BackendType.COMFYUI and not self._is_krea2_generation():
            from core.comfy_workflow_controls import snapshot_comfy_payload
            payload.update(snapshot_comfy_payload(get_backend(), payload, 'img2img'))
        return payload

    def _apply_postprocess_chain(self, payload):
        """Forge Neo와 동일: ADetailer + SAM3 + Anima Guidance 모두 alwayson_scripts로 적용"""
        payload.setdefault("alwayson_scripts", {})

        # ADetailer: Forge Neo와 동일하게 alwayson_scripts로 직접 적용
        if self.adetailer_group.isChecked():
            adetailer_args = self._build_adetailer_args()
            if adetailer_args:
                payload["alwayson_scripts"]["ADetailer"] = {"args": adetailer_args}
                _logger.info("ADetailer alwayson_scripts 적용됨 (Forge Neo 방식)")

        # SAM3: Forge Neo와 동일하게 alwayson_scripts로 직접 적용.
        # state dict 구성은 core/sam3_args로 일원화 — 배치/Refine 경로와 동일 로직.
        sam3_group = getattr(self, "sam3_group", None)
        if sam3_group is not None and sam3_group.isChecked() and hasattr(self, "_build_sam3_settings"):
            sam3_settings = self._build_sam3_settings(payload)
            if sam3_settings:
                from core import sam3_args
                sam3_args.apply_to_payload(payload, sam3_settings)
                _logger.info("SAM3 alwayson_scripts 적용됨 (Forge Neo 방식)")

        # Anima Guidance Suite: PAG/SEG/SLG · APG/CWM/SMC · Skimmed CFG ·
        # DCW/RDC/DAVE/CNS · Modulation · Detail Daemon.
        # 전부 꺼져 있으면 아무것도 넣지 않는다 (확장을 건드리지 않아야 결과가 동일).
        try:
            from core import anima_guidance
            anima_settings = self._build_anima_settings()
            anima_guidance.apply_to_payload(payload, anima_settings)
            summary = anima_guidance.describe_active(anima_settings)
            if summary:
                _logger.info("Anima Guidance 적용됨: %s", summary)
        except Exception as e:
            # guidance는 부가 기능 — 실패해도 생성 자체는 진행되어야 한다
            _logger.warning("Anima Guidance 적용 실패 (무시하고 생성 진행): %s", e)

    def _build_adetailer_slot(self, widgets):
        """ADetailer 슬롯 딕셔너리 생성 (공식 REST API 스펙 준수)

        슬롯의 상수 부분·기본값은 core/adetailer_args 한 벌이고, 여기서는 위젯 값만
        override 로 넘긴다. widgets dict에서 proxy를 통해 읽는다.
        """
        from core import adetailer_args as ad

        _txt, _float, _int = _widget_text, _widget_float, _widget_int
        defaults = ad.DEFAULT_SLOT

        model_name = _txt(widgets['model'], ad.DEFAULT_MODEL)
        if model_name == 'None' or not model_name.strip():
            model_name = ad.DEFAULT_MODEL

        confidence = _float(widgets['confidence'], ad.DEFAULT_CONFIDENCE)
        denoise = _float(widgets['denoise'], ad.DEFAULT_DENOISE)
        mask_blur = _int(widgets['mask_blur'], defaults['ad_mask_blur'])
        padding = _int(widgets['padding'], defaults['ad_inpaint_only_masked_padding'])
        prompt = widgets['prompt'].toPlainText() if hasattr(widgets['prompt'], 'toPlainText') else ''

        _logger.debug(f"AD Slot: model={model_name}, confidence={confidence}, "
                      f"denoise={denoise}, mask_blur={mask_blur}, prompt='{prompt[:30]}'")

        neg_prompt = widgets['neg_prompt'].toPlainText() if hasattr(widgets['neg_prompt'], 'toPlainText') else ''
        dilate_erode = (_int(widgets['dilate_erode'], defaults['ad_dilate_erode'])
                        if 'dilate_erode' in widgets else defaults['ad_dilate_erode'])
        mask_merge = (_txt(widgets['mask_merge_invert'], defaults['ad_mask_merge_invert'])
                      if 'mask_merge_invert' in widgets else defaults['ad_mask_merge_invert'])

        overrides = {
            "ad_model": model_name,
            "ad_prompt": prompt,
            "ad_negative_prompt": neg_prompt,
            "ad_confidence": confidence,
            "ad_dilate_erode": dilate_erode,
            "ad_mask_merge_invert": mask_merge,
            "ad_mask_blur": mask_blur,
            "ad_denoising_strength": denoise,
            "ad_inpaint_only_masked_padding": padding,
            "ad_use_inpaint_width_height": widgets['use_inpaint_size_check'].isChecked(),
            "ad_inpaint_width": _int(widgets['inpaint_width'], defaults['ad_inpaint_width']),
            "ad_inpaint_height": _int(widgets['inpaint_height'], defaults['ad_inpaint_height']),
            "ad_use_steps": widgets['use_steps_check'].isChecked(),
            "ad_steps": _int(widgets['steps'], defaults['ad_steps']),
            "ad_use_cfg_scale": widgets['use_cfg_check'].isChecked(),
            "ad_cfg_scale": _float(widgets['cfg'], defaults['ad_cfg_scale']),
            "ad_use_checkpoint": widgets['use_checkpoint_check'].isChecked(),
            "ad_use_vae": widgets['use_vae_check'].isChecked(),
            "ad_use_sampler": widgets['use_sampler_check'].isChecked(),
        }
        # use_* 가 켜져있을 때만 값을 오버라이드
        if widgets['use_checkpoint_check'].isChecked():
            ckpt = _txt(widgets['checkpoint_combo'])
            if ckpt:
                overrides["ad_checkpoint"] = ckpt
        if widgets['use_vae_check'].isChecked():
            vae = _txt(widgets['vae_combo'])
            if vae:
                overrides["ad_vae"] = vae
        if widgets['use_sampler_check'].isChecked():
            overrides["ad_sampler"] = _txt(widgets['sampler_combo'], ad.DEFAULT_SAMPLER)
            overrides["ad_scheduler"] = _txt(widgets['scheduler_combo'], ad.DEFAULT_SCHEDULER)

        return ad.build_slot(**overrides)
    
    def _notify_generation_done(self):
        """생성 완료 알림 (비활성 창일 때)"""
        # 트레이 알림
        if hasattr(self, '_tray_manager'):
            self._tray_manager.notify("생성 완료", "이미지 생성이 완료되었습니다!")

        # 작업 표시줄 깜박임 (Windows)
        try:
            import ctypes
            hwnd = int(self.winId())
            ctypes.windll.user32.FlashWindow(hwnd, True)
        except Exception:
            pass

        # 사운드 알림
        try:
            import winsound
            winsound.MessageBeep(winsound.MB_ICONASTERISK)
        except Exception:
            pass

    def _build_sam3_settings(self, payload):
        """SAM3 설정 딕셔너리 생성"""
        widgets = getattr(self, 'sam3_widgets', None)
        if not widgets:
            return None

        # SAM3 sampler/scheduler 결정 — 사용자 기대대로:
        # '별도 Sampler/Scheduler' 체크박스가 켜져있어도 콤보가 'Use same X'면
        # use_X=False로 강제하여 base의 ER SDE/Beta57 등이 상속되게.
        # Forge 확장 v0.6.0+는 sampler/scheduler 토글이 분리됨.
        _sampler_v = _widget_text(widgets['sampler'], 'Use same sampler')
        _scheduler_v = _widget_text(widgets['scheduler'], 'Use same scheduler')
        _use_sampler = (widgets['use_sampler_check'].isChecked()
                        and _sampler_v not in ('', 'Use same sampler'))
        _use_scheduler = (widgets['use_scheduler_check'].isChecked()
                          and _scheduler_v not in ('', 'Use same scheduler'))

        _txt, _float, _int = _widget_text, _widget_float, _widget_int

        def _plain(w):
            return w.toPlainText() if hasattr(w, 'toPlainText') else _txt(w, '')

        detect_prompt = _txt(widgets['detect_prompt'], 'face').strip() or 'face'
        return {
            "sam3_mode": _txt(widgets['mode'], 'Inpaint'),
            "sam3_mask_mode": _txt(widgets['mask_mode'], 'Individual'),
            "sam3_prompt": detect_prompt,
            "sam3_exclude_prompt": _plain(widgets['exclude_prompt']),
            "sam3_inpaint_prompt": _plain(widgets['inpaint_prompt']),
            "sam3_negative_prompt": _plain(widgets['neg_prompt']),
            "sam3_threshold": _float(widgets['threshold'], 0.4),
            "sam3_mask_dilation": _int(widgets['mask_dilation'], 0),
            "sam3_mask_hull": widgets['mask_hull'].isChecked(),
            "sam3_mask_outline_px": _int(widgets['mask_outline_px'], 0),
            "sam3_checkpoint": _txt(widgets['checkpoint'], 'sam3.pt'),
            "sam3_device": _txt(widgets['device'], 'cuda') or 'cuda',
            "sam3_mask_blur": _int(widgets['mask_blur'], 4),
            "sam3_denoising_strength": _float(widgets['denoise'], 0.4),
            "sam3_inpainting_fill": _txt(widgets['inpainting_fill'], 'original') or 'original',
            "sam3_inpaint_only_masked": widgets['inpaint_only_masked'].isChecked(),
            "sam3_inpaint_only_masked_padding": _int(widgets['padding'], 32),
            "sam3_use_inpaint_width_height": widgets['use_inpaint_size_check'].isChecked(),
            "sam3_inpaint_width": _int(widgets['inpaint_width'], 1024),
            "sam3_inpaint_height": _int(widgets['inpaint_height'], 1024),
            "sam3_use_steps": widgets['use_steps_check'].isChecked(),
            "sam3_steps": _int(widgets['steps'], 28),
            "sam3_use_cfg_scale": widgets['use_cfg_check'].isChecked(),
            "sam3_cfg_scale": _float(widgets['cfg'], 7.0),
            # sampler/scheduler 분리 — 위에서 계산한 _use_sampler / _use_scheduler
            "sam3_use_sampler": _use_sampler,
            "sam3_sampler": _sampler_v if _use_sampler else 'Use same sampler',
            "sam3_use_scheduler": _use_scheduler,
            "sam3_scheduler": _scheduler_v if _use_scheduler else 'Use same scheduler',
            "sam3_use_seed": widgets['use_seed_check'].isChecked(),
            "sam3_seed": _int(widgets['seed'], -1),
            "sam3_use_noise_multiplier": widgets['use_noise_multiplier_check'].isChecked(),
            "sam3_noise_multiplier": _float(widgets['noise_multiplier'], 1.0),
            "sam3_restore_face": widgets['restore_face'].isChecked(),
            "sam3_preview_overlay": widgets['preview_overlay'].isChecked(),
            "sam3_save_artifacts": widgets['save_artifacts'].isChecked(),
            # 검출 직후 SAM3(~3.5GB) VRAM 회수 — 16GB GPU 인페인트 OOM 방지.
            # UI 토글로 노출 (기본 ON). 끄면 인페인트 내내 상주 → 빠르지만 VRAM↑
            "sam3_unload_after": widgets['unload_after'].isChecked(),
            # ── ControlNet 주입 (확장의 SAM3 > ControlNet 아코디언)
            "sam3_cn_enable": widgets['cn_enable'].isChecked(),
            "sam3_cn_override_external": widgets['cn_override_external'].isChecked(),
            "sam3_cn_model": _txt(widgets['cn_model'], 'None') or 'None',
            "sam3_cn_module": _txt(widgets['cn_module'], 'inpaint_only') or 'inpaint_only',
            "sam3_cn_weight": _float(widgets['cn_weight'], 1.0),
            "sam3_cn_guidance_start": _float(widgets['cn_guidance_start'], 0.0),
            "sam3_cn_guidance_end": _float(widgets['cn_guidance_end'], 1.0),
            "sam3_cn_pixel_perfect": widgets['cn_pixel_perfect'].isChecked(),
            "sam3_cn_control_mode": _txt(widgets['cn_control_mode'], 'Balanced') or 'Balanced',
            "sam3_cn_resize_mode": (_txt(widgets['cn_resize_mode'], 'Crop and Resize')
                                    or 'Crop and Resize'),
            "sam3_cn_processor_res": _int(widgets['cn_processor_res'], 512),
            "sam3_cn_threshold_a": _float(widgets['cn_threshold_a'], -1.0),
            "sam3_cn_threshold_b": _float(widgets['cn_threshold_b'], -1.0),
        }

    def _build_anima_settings(self) -> dict:
        """Anima Guidance Suite 설정 dict — 프록시 값을 그대로 모은다.

        타입 강제/범위 클램프는 core/anima_guidance.build_args가 담당하므로
        여기서는 문자열 원본만 넘긴다 (변환을 두 군데서 하면 어긋난다).
        """
        widgets = getattr(self, 'anima_guidance_widgets', None) or {}
        return {key: _widget_text(proxy, '') for key, proxy in widgets.items()}
