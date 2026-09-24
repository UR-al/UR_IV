# ui/generator_main.py
"""
GeneratorMainUI - 메인 윈도우 클래스 (데이터 연동 및 안정화 최종판)
"""
import sys
import traceback
import json
import os
import random
import shutil
import subprocess
import threading

from PyQt6.QtWidgets import QMessageBox, QLineEdit, QTextEdit, QApplication, QHBoxLayout, QWidget, QFileDialog, QMenu
from PyQt6.QtCore import QTimer, QEvent, Qt, pyqtSignal, pyqtSlot

from core.path_safety import strip_file_url


def _clean_path(path: str) -> str:
    """file:// URL 이면 스킴 제거 + 퍼센트 디코딩, 원시 경로는 글자 그대로 + OS 구분자 정규화.

    예전에는 원시 경로까지 unquote 해, 이름에 ``%20`` 같은 글자가 든 파일의 즐겨찾기·메타데이터·삭제가
    다른 파일(디코드된 이름의 형제)을 가리켰다. 규칙은 core.path_safety.strip_file_url 하나를 쓴다.
    """
    if not path:
        return ''
    clean_path = strip_file_url(path)
    clean_path = clean_path.replace('/', os.sep)
    return clean_path

from ui.generator_base import GeneratorBase
from ui.generator_ui_setup import UISetupMixin
from ui.generator_prompts import PromptHandlingMixin
from ui.generator_generation import GenerationMixin
from ui.generator_settings import SettingsMixin
from ui.generator_actions import ActionsMixin
from ui.generator_webui import WebUIMixin
from ui.creator_actions import CreatorActionsMixin
from ui.chat_actions import ChatActionsMixin
from ui.model_download_actions import ModelDownloadActionsMixin
from ui.xyz_actions import XYZActionsMixin
from ui.comfy_workflow_actions import ComfyWorkflowActionsMixin
from ui.comfy_compatibility_actions import ComfyCompatibilityActionsMixin
from ui.relight_actions import RelightActionsMixin
from ui.hand_reconstruction_actions import HandReconstructionActionsMixin
from ui.event_search_actions import EventSearchActionsMixin
from widgets.queue_panel import QueuePanel
from widgets.queue_manager import QueueManager
from utils.prompt_cleaner import get_prompt_cleaner
from utils.theme_manager import get_theme_manager, get_color
from utils.tray_manager import TrayManager
from utils.atomic_json import atomic_write_json
from core.file_naming import sanitize_filename
from core.metadata_actions import apply_block_reason, infotext_ui_fields, prompt_transfer
from ui.image_metadata_actions import (
    metadata_for_action as _metadata_for_action,
    queue_item_from_metadata as _queue_item_from_metadata,
    start_generation_from_metadata as _start_generation_from_metadata,
    # 별칭 없이 — tests/test_web_action_policy 의 대화상자 도달 분석이 함수 이름으로 따라간다
    transplant_metadata_action,
)


def _same_api_endpoint(left: object, right: object) -> bool:
    """Compare configured backend endpoints without conflating backend types."""

    first = str(left or '').strip().rstrip('/').casefold()
    second = str(right or '').strip().rstrip('/').casefold()
    return bool(first and second and first == second)


from ui.settings_data_actions import handle_settings_data_action, should_skip_persist_action

class GeneratorMainUI(
    GeneratorBase,
    UISetupMixin,
    PromptHandlingMixin,
    GenerationMixin,
    SettingsMixin,
    ActionsMixin,
    WebUIMixin,
    CreatorActionsMixin, ChatActionsMixin, ModelDownloadActionsMixin, XYZActionsMixin,
    ComfyWorkflowActionsMixin, ComfyCompatibilityActionsMixin, RelightActionsMixin,
    HandReconstructionActionsMixin, EventSearchActionsMixin,
):
    _IMAGE_EXTS = ('.png', '.jpg', '.jpeg', '.webp', '.bmp')
    animaForgeImportReady = pyqtSignal(object)

    def __init__(self):
        try:
            super().__init__()
            print("[System] Initializing AI Studio Pro Engine...")
            self.setWindowTitle("AI Studio Pro")
            self.setAcceptDrops(True)

            # 1. 필수 속성 초기화 (ADetailer 슬롯 프록시 s1/s2_widgets 는 _setup_ui 가 만든다)
            self.is_programmatic_change = False
            self.filtered_results = []

            # 1-A. UI 상태 영속화 매니저 — 창 크기/위치 자동 저장/복원
            from core.ui_state_manager import UIStateManager
            from core.storage_paths import config_file
            self.ui_state = UIStateManager(config_file(
                "state/ui_state.json",
                legacy_paths="save/ui_state.json",
            ))
            self._register_ui_state_handlers()

            # 2. 아이콘 설정
            from PyQt6.QtGui import QIcon
            icon_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'assets', 'icons', 'app_icon.svg')
            if os.path.exists(icon_path):
                self.setWindowIcon(QIcon(icon_path))

            self.prompt_cleaner = get_prompt_cleaner()

            # 3. UI 및 프록시 레이어 구축
            self._setup_ui()
            self.animaForgeImportReady.connect(self._apply_anima_forge_import)
            self.vue_bridge.backendRuntimeEvent.connect(self._on_backend_runtime_event)

            # 에러 핸들러에 bridge 등록
            from core.error_handler import set_bridge
            set_bridge(self.vue_bridge)
            
            # 4. 시그널 및 설정 동기화
            self.connect_signals()
            self.load_settings()

            # 5. 백엔드 선택 + 선(先)로딩 + Vue 준비는 main()이 _run_startup_sequence로
            #   스플래시와 함께 구동한다 (백엔드 선택 → 로딩창 → 데이터 준비 → 완성된 UI 노출).
            #   __init__은 UI/시그널/대기열/트레이 구성까지만 — 다이얼로그/무거운 로딩은 안 함.

            # 8. 대기열 및 시스템 트레이
            self._setup_queue()
            self._setup_tray()

            # 7. 타이머 가동
            # 백엔드 실제 연결 여부 — get_backend()는 항상 객체를 돌려주므로(미설정 시
            # config로 WebUI 기본 생성) 'None 체크'로는 미연결을 구분할 수 없다. 이 플래그로
            # 미연결/건너뛰기 상태에선 VRAM 폴링이 헛되이 백엔드를 두드리지 않게 한다.
            # on_webui_info_loaded에서 True(그 자리에서 LoRA 캐시도 워커로 다시 채운다),
            # 실패/건너뛰기에서 False.
            self._backend_connected = False
            # VRAM 은 NVML 로 장치 전체를 직접 잰다(마이크로초) — 1초 간격이라 Ollama 가 모델을
            # 올리고 내리는 것이 바로 보인다. 예전 30초 + Forge 자기 메모리는 12.1 에 멈춰 있었다.
            # 느린 폴백(nvidia-smi·백엔드 HTTP)은 core.gpu_stats 가 5초 캐시로 묶는다.
            self._vram_timer = QTimer()
            self._vram_timer.setInterval(1000)
            self._vram_timer.timeout.connect(self._update_vram_status)
            self._vram_timer.start()
            QTimer.singleShot(1500, self._update_vram_status)

            self._clean_timer = QTimer()
            self._clean_timer.setSingleShot(True)
            self._clean_timer.setInterval(500)
            self._clean_timer.timeout.connect(self._deferred_clean_all)
            self._setup_realtime_cleaning()

            # 8. 초기값 강제 업데이트
            QTimer.singleShot(500, self.update_total_prompt_display)

            # 9. 조건식 + 기본값 + ui_prefs 적용 — 동기로. Vue 는 바인딩 직후 getInitialConfig 로
            #    같은 파일을 당겨 가므로(bridge.js _requestInitialConfig) 1회 emit 에 기대지 않는다.
            #    레거시 마이그레이션이 그 pull 보다 먼저 끝나야 하고, 늦은 타이머가 Vue 가 이미 보낸
            #    값을 덮어서도 안 된다(감사 #107).
            self._apply_saved_configs()

            # 10. UI 상태 복원 (150ms 지연 — 위젯 레이아웃 자리 잡은 후)
            #     showMaximized() 등 다른 main의 호출과 충돌 회피 위해 약간 더 늦춤
            QTimer.singleShot(200, lambda: self.ui_state.restore_all_delayed(150))

            # 11. PR 9 — 모드별 자동화 설정 영속 (webui/comfyui 분리)
            #     백엔드 전환 시 자동 저장/복원, Vue로 자동 전파
            from core.mode_aware_automation import ModeAwareAutomationSettings
            self.automation_persistence = ModeAwareAutomationSettings(self)
            # 동기로 파일 값을 호스트에 싣는다. Vue 는 마운트 때 getAutomationSettings 로 이 값을
            # 당겨 채운 뒤에 동기화를 보낸다 — 예전엔 1.5초 타이머의 1회 emit 보다 Vue 의 하드코딩
            # 기본값 푸시가 먼저 파일을 덮었다(감사 #42).
            try:
                self.automation_persistence.initialize()
            except Exception as e:   # 설정 파일 문제로 앱 시작이 막히지 않게 — 기본값으로 계속
                print(f"[Warning] 자동화 설정 복원 실패(기본값 사용): {e}")
            # 자동완성 데이터(태그 DB 2초 + 한국어 카탈로그) 백그라운드 예열 — 첫 키 입력 공백 제거
            QTimer.singleShot(2500, self.vue_bridge.warmTagSuggestions)

            # 12. PR 1 — PromptPipeline 표준 훅 등록
            #     기존 처리 뒤에 호출되어 비파괴적 — 회귀 위험 없음
            try:
                from core.standard_hooks import register_standard_hooks
                register_standard_hooks()
            except Exception as e:
                print(f"[Warning] 표준 훅 등록 실패: {e}")

            # 13. 옛 썸네일 캐시 폴더(image_cache/thumbs) 은퇴 — 시작 30초 뒤 데몬 스레드에서 한 번.
            #     현재 형식은 config.THUMB_DIR(thumbs_v2)로 옮기고 레거시 PyQt 갤러리 썸네일은 지운다.
            #     그 전에 히스토리·카드가 요청한 키는 조회가 먼저 옮겨 온다(ThumbnailPrefetcher·
            #     aithumb: 핸들러의 legacy_dir) — 여기서는 남은 것을 한꺼번에 치운다.
            try:
                from config import LEGACY_THUMB_DIR, THUMB_DIR
                from core.legacy_thumb_cache import start_legacy_thumb_retirement
                start_legacy_thumb_retirement(LEGACY_THUMB_DIR, THUMB_DIR)
            except Exception as e:
                print(f"[Warning] 옛 썸네일 캐시 정리 예약 실패: {e}")

            print("[System] Engine Ready.")

        except Exception as e:
            print(f"\n[Fatal] Error during boot: {e}")
            traceback.print_exc()
            QMessageBox.critical(None, "Boot Error", f"Fatal initialization error:\n{e}")
            sys.exit(1)

    # ========== Vue Bridge Action Handler (The Core) ==========

    def _handle_vue_action(self, action: str, payload: dict):
        """[중요] Vue에서 날아온 모든 액션을 분석하고 백엔드 로직에 주입"""
        # 페이로드 전체를 json.dumps 하지 않는다 — update_prompt_deck(수만 행)·chat_save
        # (base64 이미지) 같은 큰 페이로드에서 로그 100자 때문에 GUI 스레드가 멈췄다.
        # 로그 줄은 ASCII 이고, 출력 실패(cp949 리다이렉트·닫힌 stdout)도 액션을 막지 않는다.
        from core.action_log import log_action
        log_action(action, payload)

        # Creator Studio는 별도 deep module에서 처리한다. 이 seam을 먼저
        # 통과시켜 아래의 거대한 레거시 문자열 라우터를 더 키우지 않는다.
        if self._handle_model_download_action(action, payload):
            return
        if ComfyWorkflowActionsMixin._handle_comfy_workflow_action(self, action, payload):
            return
        if ComfyCompatibilityActionsMixin._handle_comfy_compatibility_action(self, action, payload):
            return
        if RelightActionsMixin._handle_relight_action(self, action, payload):
            return
        if HandReconstructionActionsMixin._handle_hand_reconstruction_action(self, action, payload):
            return
        if XYZActionsMixin._handle_xyz_action(self, action, payload):
            return
        if self._handle_chat_action(action, payload):
            return
        if self._handle_creator_action(action, payload):
            return
        # 설정 백업을 가져온 직후(재시작 전)에는 저장 액션이 가져온 파일을 덮어쓰지 않는다.
        if should_skip_persist_action(self, action):
            self.vue_bridge.showNotification.emit(
                'info', '가져온 설정을 적용하려고 재시작 중입니다 — 지금은 저장하지 않습니다')
            return

        try:
            # 1. 워크스페이스 제어
            if action == 'native_tab_switch':
                tab_id = payload.get('tab', 't2i')
                tab_map = {'web': 1, 'backend': 2}
                idx = tab_map.get(tab_id, 0)
                if hasattr(self, '_main_stack'): self._main_stack.setCurrentIndex(idx)
                # 네이티브 웹뷰는 탭을 열 때 싣는다 — 안 보는 페이지를 뒤에서 돌리지 않는다.
                if tab_id == 'backend' and hasattr(self, 'backend_ui_tab'):
                    self.backend_ui_tab.ensure_loaded()
                elif tab_id == 'web' and hasattr(self, 'web_tab'):
                    self.web_tab.ensure_loaded()
                # (탭 전환은 화면 자체가 알린다 — 상태줄에 'Workspace Switched' 를 띄우지 않는다)

            elif action == 'vue_tab_switch':
                # Vue SPA 내부 탭(T2I/I2I/Inpaint/Search 등) 전환 알림.
                # 라우팅 자체는 Vue Router가 처리하므로 Python은 스택을 SPA(0)로
                # 맞추고 tabChanged 시그널만 전달한다.
                tab_id = payload.get('tab', 't2i')
                if hasattr(self, '_main_stack'):
                    self._main_stack.setCurrentIndex(0)
                if hasattr(self, 'vue_bridge'):
                    self.vue_bridge.tabChanged.emit(tab_id)

            # 2. 이미지 생성 엔진
            elif action == 'generate':
                # Vue 데이터가 Store를 통해 Proxy에 이미 동기화되어 있어야 함
                self.on_generate_clicked()

            elif action == 'set_high_res_factor':
                # Vue 고해상도 토글 → 생성 시점에 width/height × factor 적용
                # generator_generation.py가 self._high_res_factor 읽음
                try:
                    enabled = bool(payload.get('enabled', False))
                    factor = float(payload.get('factor', 1.5))
                    if not enabled:
                        factor = 1.0
                    self._high_res_factor = max(1.0, min(4.0, factor))
                    self.show_status(f"고해상도: {'ON ' + str(round(self._high_res_factor, 2)) + 'x' if enabled else 'OFF'}")
                except Exception as e:
                    self.show_status(f"고해상도 설정 실패: {e}")

            elif action == 'cancel_generation':
                # 자동화 중 취소 = 자동화도 중지 (안 끄면 다음 사이클이 계속 돎)
                if getattr(self, 'is_automating', False):
                    try:
                        self._stop_automation('사용자 취소')
                    except Exception:
                        self.is_automating = False
                worker = getattr(self, 'gen_worker', None)
                if worker is not None and hasattr(worker, 'cancel') and worker.isRunning():
                    worker.cancel()   # 플래그 + 백엔드 interrupt → 곧 cancelled finished 도착
                    self.show_status("생성 취소 중...")
                    # 버튼/타이틀 즉시 복구 (finished의 cancelled 분기가 마무리)
                    if hasattr(self, '_restore_generate_button'):
                        self._restore_generate_button()
                else:
                    self.show_status("취소할 생성이 없습니다")
                    if hasattr(self, '_restore_generate_button'):
                        self._restore_generate_button()

            # 3. 탭 간 데이터 전송 (이미지 & 프롬프트)
            elif action in ('send_to_i2i', 'send_to_inpaint', 'send_to_editor'):
                path = payload.get('path', '')
                if not path: return
                # file:/// 제거 + 경로 정규화
                clean_path = _clean_path(path)
                if clean_path.startswith('/') and ':' in clean_path[1:3]:
                    clean_path = clean_path[1:]  # /C:/... → C:/...
                clean_path = os.path.normpath(clean_path)
                if not os.path.exists(clean_path):
                    print(f"[Send] File not found: {clean_path} (original: {path})")
                    return

                fwd_path = clean_path.replace('\\', '/')
                print(f"[Send] {action}: {fwd_path}")
                if action == 'send_to_i2i':
                    self.vue_bridge.tabChanged.emit('i2i')
                    QTimer.singleShot(100, lambda p=fwd_path: self.vue_bridge.i2iImageLoaded.emit(p))
                elif action == 'send_to_inpaint':
                    self.vue_bridge.tabChanged.emit('inpaint')
                    QTimer.singleShot(100, lambda p=fwd_path: self.vue_bridge.inpaintImageLoaded.emit(p))
                elif action == 'send_to_editor':
                    # 탭 전환 먼저 → 100ms 후 이미지 로드 (keep-alive 렌더링 대기)
                    self.vue_bridge.tabChanged.emit('editor')
                    QTimer.singleShot(100, lambda p=fwd_path: self.vue_bridge.editorImageLoaded.emit(p))
                self.show_status("Asset Transfer Successful.")
                tab_names = {'send_to_i2i': 'I2I', 'send_to_inpaint': 'Inpaint', 'send_to_editor': 'Editor'}
                if hasattr(self, 'vue_bridge'):
                    self.vue_bridge.showNotification.emit('success', f'{tab_names.get(action, "")}로 전송됨')

            # 4. 에디터 정밀 조작 (먹통 해결)
            elif action == 'editor_open_file':
                # 파일 다이얼로그 호출 (메인 스레드 보장)
                path, _ = QFileDialog.getOpenFileName(self, "Select Image", "", "Images (*.png *.jpg *.jpeg *.webp)")
                if path: self.vue_bridge.editorImageLoaded.emit(path.replace('\\', '/'))
            
            elif action in ('editor_save', 'editor_save_as'):
                # 저장(비파괴): 원본은 덮어쓰지 않고 원본 옆 <stem>_edited[_N] 사본(임시 원본이면
                # 기본 출력 폴더)에 메타데이터를 보존해 쓴다 — 그 문서의 다음 저장은 그 사본만 갱신.
                # 다른 이름으로 저장은 고른 확장자로 다시 인코딩한다. 병합 안 된 드로잉 레이어도
                # 합성하며, 결과는 editorSaveResult 로 돌려준다.
                from ui.editor_save_actions import handle_editor_save_action
                from core.web_action_policy import HOST_DIALOG_MESSAGE, host_dialogs_blocked
                save_kwargs = {}
                if host_dialogs_blocked(web_mode=bool(getattr(self, 'web_mode', False)),
                                        remote=bool(getattr(self, 'web_remote', True))):
                    # 원격 웹 모드: 저장 위치를 호스트 대화상자로 묻지 않고 editorSaveResult
                    # 오류로 돌려준다 — 프론트의 '저장 중' 상태가 풀리도록 브리지가 아니라
                    # 여기서 처리한다(web_action_policy.BRANCH_HANDLED_DIALOG_ACTIONS).
                    def _refuse_host_dialog(_title, _suggested):
                        raise RuntimeError(HOST_DIALOG_MESSAGE)
                    save_kwargs['ask_path'] = _refuse_host_dialog
                handle_editor_save_action(self, action, payload, **save_kwargs)

            elif action == 'editor_add_yolo_model':
                from PyQt6.QtWidgets import QFileDialog as _QFD
                from core import yolo_models
                paths, _ = _QFD.getOpenFileNames(
                    self, "YOLO 모델 선택", yolo_models.get_editor_models_dir(),
                    "YOLO Model (*.pt *.onnx *.safetensors);;All Files (*)"
                )
                if paths:
                    # 외부 파일은 Editor_models 로 복사하고(같은 이름이 있으면 그 파일), 초기화로
                    # 비활성된 모델이면 다시 켠다. 설정·폴더를 매번 새로 읽는다(core.yolo_models).
                    try:
                        model_paths = yolo_models.add_models(paths)
                    except Exception as e:
                        self.vue_bridge.showNotification.emit('error', f'YOLO 모델 추가 실패: {e}')
                        return
                    label = yolo_models.model_label(model_paths)
                    self.vue_bridge.yoloModelUpdated.emit(label)
                    self.show_status(f"YOLO Model loaded: {label}")

            # 6. 기타 스튜디오 도구
            elif action == 'show_prompt_history': self._show_prompt_history()
            elif action == 'save_settings':
                saved = self.save_settings()
                # 저장 후 load_settings 로 다시 읽지 않는다 — 방금 쓴 값이 곧 프록시 값이라 얻는 것이
                # 없고, 재로드는 같은 URL 백엔드 재생성(XYZ 축 초기화)·슬라이더 표류·Vue 정리 토글
                # 덮어쓰기만 남겼다(감사 #29).
                # defaults도 동시 업데이트(f90af4319 'Default 연동') — 다섯 키만 병합한다.
                # Settings 화면은 다시 열릴 때(onActivated)·전역 저장 뒤 이 값을 다시 읽는다.
                try:
                    from core.tab_defaults import sync_t2i_defaults
                    sync_t2i_defaults({
                        'steps': self.steps_input.text(), 'cfg': self.cfg_input.text(),
                        'width': self.width_input.text(), 'height': self.height_input.text(),
                        'seed': self.seed_input.text(),
                    })
                except Exception as exc:
                    print(f"[Config] tab_defaults 연동 실패: {exc}")
                # save_settings 는 실패하면 False — 예전엔 실패해도 '저장되었습니다' 토스트가 떴다.
                from ui.status_line import notify_user
                if saved is False:
                    notify_user(self, 'error', '설정을 저장하지 못했습니다 — 콘솔 로그를 확인하세요')
                else:
                    notify_user(self, 'success', '설정이 저장되었습니다')
            elif action == 'import_anima_from_forge':
                self._start_anima_forge_import()
            elif action == 'reset_anima_guidance':
                # 기본값의 단일 출처는 core/anima_guidance 스펙 — Vue 는 사본을 들지 않는다
                self._reset_anima_guidance()
            elif action == 'swap_resolution': self._swap_resolution()
            elif action == 'set_random_resolutions':
                lst = payload.get('list', [])
                self.random_resolutions = [(int(r[0]), int(r[1]), str(r[2])) for r in lst if len(r) >= 3]
            elif action == 'set_rating_filter':
                # 같은 필터 재전송(마운트·웹 탭마다)은 무시하고, 바뀌면 진행도를 지키며 덱을 맞춘 뒤
                # 저장·상태 알림(core.search_deck.set_owner_rating_filter).
                from core.search_deck import set_owner_rating_filter
                ratings = payload.get('ratings') if isinstance(payload, dict) else None
                set_owner_rating_filter(
                    self, ratings if isinstance(ratings, list) else ['g', 's', 'q', 'e'])
            elif action == 'update_prompt_deck':
                # Vue에서 필터링된 결과로 덱 업데이트. Vue 는 base 행 인덱스
                # ({indices, base_size, lineage})를 보내고, 구형 {results} 도 받는다.
                # lineage 가 현재 스냅숏과 다르면(늦게 도착한 옛 필터) 거부한다.
                # base 를 아직 모르면(loadFullResults 가 기록 못 함) 같은 규칙으로 한 번 읽어 본다.
                from core.search_result_store import SearchResultStore
                from core.search_session import resolve_prompt_deck_update
                deck_update = resolve_prompt_deck_update(
                    self, payload, store_factory=SearchResultStore,
                )
                if deck_update.error:
                    # show_status 는 하단 상태줄 한 줄이라 놓치기 쉽다 — Vue 는 이미 '필터 적용'
                    # 토스트를 띄웠으니, 자동화 덱이 그대로라는 걸 알림(토스트)으로도 알린다.
                    self.show_status(deck_update.error)
                    bridge = getattr(self, 'vue_bridge', None)
                    if bridge is not None:
                        bridge.showNotification.emit('warning', deck_update.error)
                    return
                deck = deck_update.rows
                if deck is not None:
                    from core.search_deck import refill_owner_deck
                    self.filtered_results = deck
                    # 덱 재구성(등급 필터 → 셔플 → 저장 → 상태)의 단일 경로
                    refill_owner_deck(self)
                    if not deck and getattr(self, 'is_automating', False):
                        self._stop_automation(
                            '검색 필터 결과가 없어 자동화를 중지했습니다.'
                        )
                    # 필터 적용 결과를 디스크에 저장(단일 쓰기 경로) → 재시작 시 '필터링된' 덱 복원.
                    #   full은 갱신 안 함(새 검색 때만) → '필터 해제' 베이스(last_full)는 유지.
                    self._persist_search_results(deck)
            elif action == 'reset_prompt_deck':
                # 덱 초기화 — filtered_results에서 덱을 가득 다시 채우고 셔플 (사용 0으로 리셋).
                # 저장과 덱 현황(남은/사용) 알림까지 core.search_deck 한 곳에서.
                from core.search_deck import refill_owner_deck
                refill_owner_deck(self)
            elif action == 'run_adetailer_single':
                self._run_adetailer_single(payload)
            elif action == 'run_adetailer_batch':
                self._run_adetailer_batch(payload)
            elif action == 'stop_adetailer_batch':
                # BatchView '중지' 버튼 — 워커는 stop()/_stop_event 지원하나
                # 액션 핸들러가 없어 무동작이었음. 연결.
                _w = getattr(self, '_ad_batch_worker', None)
                if _w is not None and _w.isRunning():
                    _w.stop()
                    self.vue_bridge.showNotification.emit('info', 'ADetailer 배치 중지 요청됨')
            elif action == 'run_sam3_single':
                self._run_sam3_single(payload)
            elif action == 'run_sam3_batch':
                self._run_sam3_batch(payload)
            elif action == 'run_refine':
                self._run_refine(payload)
            elif action == 'open_ad_files':
                paths, _ = QFileDialog.getOpenFileNames(self, "ADetailer 이미지 선택", "",
                    "Images (*.png *.jpg *.jpeg *.webp);;All Files (*)")
                if paths:
                    self.vue_bridge.batchFilesSelected.emit(
                        json.dumps([p.replace('\\', '/') for p in paths]))
            elif action == 'open_ad_folder':
                folder = QFileDialog.getExistingDirectory(self, "ADetailer 폴더 선택")
                if folder:
                    import glob
                    imgs = []
                    for ext in ('*.png', '*.jpg', '*.jpeg', '*.webp'):
                        imgs.extend(glob.glob(os.path.join(folder, ext)))
                    if imgs:
                        self.vue_bridge.batchFilesSelected.emit(
                            json.dumps([p.replace('\\', '/') for p in sorted(imgs)]))
                        self.vue_bridge.showNotification.emit('info', f'{len(imgs)}개 이미지 발견')
            elif action == 'random_prompt': self.apply_random_prompt()

            # 7. 검색 결과 → 프롬프트 적용
            elif action == 'apply_search_result':
                # Vue 검색 결과 행 → 덱과 같은 번들(태그 4종 + rating + 해상도 — 자동 해상도 포함)
                from core.search_deck import prompt_bundle_from_row
                bundle = prompt_bundle_from_row(payload)

                self.is_programmatic_change = True
                try:
                    self.apply_prompt_from_data(bundle)
                finally:
                    self.is_programmatic_change = False

                # 조건부 프롬프트 적용
                cond_pos = payload.get('cond_positive', [])
                cond_neg = payload.get('cond_negative', [])
                if cond_pos or cond_neg:
                    self._apply_vue_conditional_rules(cond_pos, cond_neg)

                self.update_total_prompt_display()

                if hasattr(self, 'vue_bridge'):
                    self.vue_bridge.tabChanged.emit('t2i')
                    self.vue_bridge.showNotification.emit('success', '프롬프트가 적용되었습니다')

            elif action == 'add_search_to_queue':
                # 먼저 프롬프트 적용하여 UI 채우기 (덱과 같은 번들 — 자동 해상도 포함)
                from core.search_deck import prompt_bundle_from_row
                bundle = prompt_bundle_from_row(payload)
                self.is_programmatic_change = True
                try:
                    self.apply_prompt_from_data(bundle)
                finally:
                    self.is_programmatic_change = False
                self.update_total_prompt_display()

                # 현재 UI 상태에서 payload 구성
                queue_payload = {
                    'prompt': self.total_prompt_display.toPlainText(),
                    'negative_prompt': self.neg_prompt_text.toPlainText(),
                    'sampler_name': self.sampler_combo.currentText(),
                    'steps': int(self.steps_input.text() or 20),
                    'cfg_scale': float(self.cfg_input.text() or 7),
                    'seed': int(self.seed_input.text() or -1),
                    'width': int(self.width_input.text() or 1024),
                    'height': int(self.height_input.text() or 1024),
                }
                if hasattr(self, 'queue_panel'):
                    self.queue_panel.add_single_item(queue_payload)
                if hasattr(self, 'vue_bridge'):
                    self.vue_bridge.showNotification.emit('success', '대기열에 추가되었습니다')

            # 8. 즐겨찾기
            elif action == 'add_favorite':
                path = payload.get('path', '')
                if path:
                    # core.favorites — 지금 없는 경로(분리된 드라이브)의 항목을 지우지 않고,
                    # 경로 표기 차이(/ vs \, 대소문자)는 같은 항목으로 본다.
                    # 파일을 잠깐 못 읽으면(공유 위반 등) 저장하지 않고 멈춘다 — '추가됨' 도 띄우지 않는다.
                    from core.favorites import FavoritesUnavailableError, add_favorite
                    try:
                        # (저장된 목록은 들고 있지 않는다 — 읽던 숨은 PyQt 즐겨찾기 스트립은 은퇴했고,
                        #  Vue 는 getFavorites 로 파일을 다시 읽는다)
                        added, _saved = add_favorite(_clean_path(path))
                    except (FavoritesUnavailableError, OSError) as fav_err:
                        from core.error_handler import sanitize_for_ui
                        if hasattr(self, 'vue_bridge'):
                            self.vue_bridge.showNotification.emit(
                                'error', f'즐겨찾기에 추가하지 못했습니다: {sanitize_for_ui(str(fav_err))}')
                    else:
                        self.show_status("Added to favorites.")
                        if hasattr(self, 'vue_bridge'):
                            if added:
                                self.vue_bridge.showNotification.emit('success', '즐겨찾기에 추가됨')
                            else:
                                self.vue_bridge.showNotification.emit('info', '이미 즐겨찾기에 있습니다')

            # artist lock
            elif action == 'set_artist_locked':
                locked = payload.get('locked', False)
                if hasattr(self, 'btn_lock_artist'):
                    self.btn_lock_artist.setChecked(locked)

            # 9. 이미지 삭제
            elif action == 'delete_image':
                path = payload.get('path', '')
                if path:
                    # 실제로 옮겼을 때만 '휴지통으로 이동됨' — send2trash 가 없거나 실패하면
                    # 영구 삭제로 넘어가지 않고 오류로 알린다(core.image_utils.move_to_trash).
                    # 결과는 경로별로 imageDeleteResult 로도 돌려준다 — 프론트는 removed 가
                    # 참일 때만 갤러리·폴더 캐시·히스토리에서 뺀다(실패하면 목록에 남는다).
                    # 확장자는 갤러리가 보여 주는 미디어 전부(영상·오디오 포함) — 예전엔 정지
                    # 이미지만 허용해서 영상 '삭제' 가 목록에서만 사라지고 파일은 남았다.
                    from core.image_delete import image_delete_result
                    from ui.vue_bridge import _GALLERY_MEDIA_EXTS
                    result = image_delete_result(
                        str(path), _clean_path(path), allowed_exts=_GALLERY_MEDIA_EXTS)
                    if result['ok']:
                        self.show_status("Moved to trash.")
                    if hasattr(self, 'vue_bridge'):
                        self.vue_bridge.showNotification.emit(result['level'], result['message'])
                        self.vue_bridge.imageDeleteResult.emit(json.dumps(result, ensure_ascii=False))

            # 10. 프리셋 — 저장·미리보기·불러오기·공유가 같은 PRESET_KEYS(core.generation_presets)
            elif action == 'save_preset_by_name':
                from core.generation_presets import write_preset
                try:
                    # 선행/후행/네거티브는 칸의 글이 아닌 base_* 템플릿(SettingsMixin._build_preset_settings)
                    saved = write_preset(payload.get('name', ''), self._build_preset_settings())
                    self.vue_bridge.showNotification.emit('success', f'프리셋 "{saved}" 저장됨')
                except Exception as e:
                    self.vue_bridge.showNotification.emit('error', f'저장 실패: {e}')

            elif action == 'load_preset_by_name':
                from core.generation_presets import preset_name, read_preset
                name = preset_name(payload.get('name', ''))
                preset = read_preset(name) if name else None
                if not preset:
                    self.vue_bridge.showNotification.emit('error', f'프리셋 "{name}"을(를) 읽지 못했습니다')
                    return
                warnings = self._apply_generation_preset(preset)
                if warnings:
                    self.vue_bridge.showNotification.emit(
                        'warning', f'프리셋 "{name}" 로드됨 — ' + ' · '.join(warnings))
                else:
                    self.vue_bridge.showNotification.emit('success', f'프리셋 "{name}" 로드됨')

            elif action == 'delete_preset':
                from core.generation_presets import delete_preset, preset_name
                try:
                    name = preset_name(payload.get('name', ''))
                    if name and delete_preset(name):
                        self.vue_bridge.showNotification.emit('success', f'프리셋 "{name}" 삭제됨')
                    else:
                        self.vue_bridge.showNotification.emit('info', f'프리셋 "{name}"이(가) 없습니다')
                except Exception as e:
                    self.vue_bridge.showNotification.emit('error', f'삭제 실패: {e}')

            # 설정 백업/복원·재시작·프리셋 공유 (Settings '데이터 · 백업' 카드 — ui/settings_data_actions.py)
            elif action in ('settings_export', 'settings_import', 'restart_app',
                            'presets_export', 'presets_import',
                            'character_presets_export', 'character_presets_import'):
                handle_settings_data_action(self, action, payload)

            # 11. I2I/Inpaint 생성 (Vue payload를 탭에 주입 후 실행)
            elif action == 'generate_i2i':
                if hasattr(self, 'i2i_tab'):
                    self.i2i_tab.main_window = self
                    self.i2i_tab.generate_from_payload(payload)
            elif action == 'generate_inpaint':
                # Vue 페이로드만으로 요청을 만든다 — 숨은 레거시 InpaintTab 을 거치지 않는다.
                from ui.inpaint_actions import start_vue_inpaint
                start_vue_inpaint(self, payload)

            # 12. 배치/업스케일 — 숨은 레거시 BatchTab/UpscaleTab 을 거치지 않는다.
            elif action == 'start_batch':
                from ui.batch_actions import start_vue_batch
                start_vue_batch(self, payload)
            elif action == 'start_upscale':
                from ui.upscale_actions import start_vue_upscale
                start_vue_upscale(self, payload)

            # PNG Info 파일 열기 — PngInfoView 전용 시그널. inpaintImageLoaded 를 쓰면
            # keep-alive 로 살아 있는 InpaintView 의 원본·마스크·undo 가 초기화된다.
            elif action == 'open_png_info_file':
                path, _ = QFileDialog.getOpenFileName(self, "PNG Info 이미지 선택", "", "Images (*.png *.jpg *.jpeg *.webp)")
                if path and hasattr(self, 'vue_bridge'):
                    self.vue_bridge.pngInfoImageLoaded.emit(path.replace('\\', '/'))

            # 13. 에디터 워터마크 이미지 로드
            elif action == 'editor_load_watermark_image':
                path, _ = QFileDialog.getOpenFileName(self, "워터마크 이미지 선택", "", "Images (*.png *.jpg *.jpeg *.webp)")
                if path:
                    self.show_status(f"Watermark image: {os.path.basename(path)}")
                    # 경로를 Vue 로 되돌려야 image_watermark 가 성립한다.
                    if hasattr(self, 'vue_bridge'):
                        self.vue_bridge.editorWatermarkImageLoaded.emit(path.replace('\\', '/'))

            # 15. Search parquet 저장/불러오기
            elif action == 'export_search_results':
                # Vue에서 필터링된 결과를 직접 받아서 저장
                path, _ = QFileDialog.getSaveFileName(self, "검색 결과 저장", "", "Parquet Files (*.parquet)")
                if path:
                    try:
                        import pandas as pd
                        data = payload.get('data')
                        if data and isinstance(data, list):
                            df = pd.DataFrame(data)
                        elif hasattr(self, 'filtered_results') and self.filtered_results:
                            df = pd.DataFrame(self.filtered_results)
                        else:
                            self.show_status("Export: no results")
                            from ui.status_line import notify_user
                            notify_user(self, 'warning', '내보낼 검색 결과가 없습니다')
                            return
                        df.to_parquet(path)
                        self.show_status(f"Exported {len(df)} results")
                        if hasattr(self, 'vue_bridge'):
                            self.vue_bridge.showNotification.emit('success', f'{len(df)}건 내보내기 완료')
                    except Exception as e:
                        self.show_status(f"Export failed: {e}")
                        if hasattr(self, 'vue_bridge'):
                            self.vue_bridge.showNotification.emit('error', f'내보내기 실패: {e}')

            elif action == 'import_search_results':
                path, _ = QFileDialog.getOpenFileName(self, "검색 결과 불러오기", "", "Parquet Files (*.parquet)")
                if path:
                    try:
                        import pandas as pd
                        from core.search_rows import SEARCH_RESULT_CAP, search_rows_from_frame
                        from core.search_session import publish_snapshot
                        df = pd.read_parquet(path)
                        # 검색 결과와 같은 정규화 규칙(core.search_rows) — tag_string_X 우선,
                        # 결측(NaN/None)은 ''/None(외부 parquet 의 'nan' 문자열 누출 방지),
                        # 해상도는 양의 int. to_dict('records') 라 iterrows 보다 수 배 빠르다.
                        # 행 상한도 검색과 같다(무작위 표본) — Search 의 '무제한' 모드면 끈다.
                        total_rows = len(df)
                        import_cap = (
                            None if (isinstance(payload, dict)
                                     and payload.get('disable_result_cap'))
                            else SEARCH_RESULT_CAP
                        )
                        out = search_rows_from_frame(df, cap=import_cap)
                        del df
                        if len(out) < total_rows:
                            print(f"[Search] import capped {total_rows:,} -> {len(out):,} rows")
                        # 가져온 결과는 기존 Search cache/deck snapshot과 섞지 않는다.
                        # 현재 manifest identity 아래 새 snapshot pair로 저장하면 이후
                        # 필터(active-only)도 같은 full base를 안전하게 유지한다.
                        import uuid as _uuid
                        from core.search_result_store import SearchResultStore
                        imported_identity = SearchResultStore().dataset_info()
                        imported_snapshot = _uuid.uuid4().hex

                        # Python filtered_results(+필터 base) + shuffled_prompt_deck 업데이트
                        from core.search_deck import refill_owner_deck
                        publish_snapshot(
                            self,
                            active=out,
                            base=out,
                            identity=imported_identity,
                            snapshot_id=imported_snapshot,
                        )
                        self._persist_search_results(
                            out,
                            full=out,
                            dataset_identity=imported_identity,
                            snapshot_id=imported_snapshot,
                        )
                        # 덱 재구성(등급 필터 → 셔플 → 저장 → 상태)의 단일 경로
                        refill_owner_deck(self)
                        # Vue로 결과 전달
                        imported_results_json = json.dumps(
                            out, ensure_ascii=False, separators=(',', ':')
                        )
                        imported_lineage_json = json.dumps({
                            **imported_identity,
                            'snapshot_id': imported_snapshot,
                        }, ensure_ascii=False, separators=(',', ':'))
                        self.vue_bridge.searchResultLineage.emit(
                            imported_lineage_json
                        )
                        self.vue_bridge.searchResultsReady.emit(
                            imported_results_json
                        )
                        from ui.status_line import notify_user
                        if len(out) < total_rows:
                            self.show_status(
                                f"Imported {len(out):,} of {total_rows:,} results (random sample)"
                            )
                            # 표본으로 잘렸다는 건 결과 목록만 봐서는 모른다 — 토스트로 알린다.
                            notify_user(self, 'info',
                                        f'{total_rows:,}건 중 무작위 {len(out):,}건을 불러왔습니다')
                        else:
                            self.show_status(f"Imported {len(out)} results")
                    except Exception as e:
                        self.show_status(f"Import failed: {e}")
                        from ui.status_line import notify_user
                        notify_user(self, 'error', f'검색 결과 불러오기 실패: {e}')

            # 16. 자동화 설정/토글
            elif action == 'set_automation_settings':
                # 보낸 키만 합친다 — 파일 값을 아직 못 받은 Vue 의 동기화가 빠진 키를 기본값으로 덮어
                # 파일에 쓰지 않게(R2b#1). maxRetries(PR 3)·cleanupEveryN(F2 대기열 정기 정리)·자동 NL·
                # 대기열 런타임 반영·모드별 저장(PR 9)은 core/mode_aware_automation.apply_automation_payload.
                from core.mode_aware_automation import apply_automation_payload
                apply_automation_payload(self, payload)

            elif action == 'toggle_automation':
                checked = payload.get('checked', False)
                # btn_auto_toggle 은 _init_button_proxies 가 액션 핸들러 등록(set_action_handler)보다
                # 먼저 만든다 — toggled → toggle_automation_ui 가드(생성 중 켜기 거부)를 늘 거친다.
                self.btn_auto_toggle.setChecked(bool(checked))
                # 자동화 모드 ON 시 덱 현황 즉시 전송 → '시작' 전에도 남은/사용 개수 표시
                # (자동화 종료 후·UI 재시작 후 복원된 덱 진행도를 바로 확인 가능)
                if checked and hasattr(self, '_emit_auto_status'):
                    self._emit_auto_status()
            elif action == 'stop_automation':
                if self.is_automating:
                    self._stop_automation("사용자가 자동화를 중지했습니다.")
            elif action == 'pause_automation':
                # 중지와 다르다 — 덱·카운트·반복 상태를 유지한 채 대기/생성 사이에서 선다.
                self._pause_automation()
            elif action == 'resume_automation':
                self._resume_automation()
            elif action == 'automation_override_next':
                # 다음 '한 장'에만 쓸 프롬프트 전문. 쓰고 나면 백엔드가 스스로 지운다.
                # 빈 문자열이면 덮어쓰기 취소 → 원래 조립된 프롬프트로 돌아간다.
                self._set_prompt_override(str(payload.get('prompt', '') or ''))

            # ═══════ 워크플로우 프로파일 ═══════
            elif action == 'workflow_profile_list':
                self._send_workflow_profiles_list()
            elif action == 'workflow_profile_save':
                name = str(payload.get('name', '')).strip()
                if name:
                    from core.workflow_profiles import (
                        collect_from_host, find_conflicting_profile, profile_saved_name, save_profile,
                    )
                    # 덮어쓰기는 Vue 가 사용자에게 확인받은 요청(overwrite: true)만 — 확인 없이 온 이름이
                    # 규칙상 기존 파일('Flux.' · 'flux' → Flux.json)이면 쓰지 않고 알린 뒤 목록을 다시 보낸다
                    # (Vue 의 목록이 낡았으면 다음 저장에서 확인을 묻게).
                    overwrite = payload.get('overwrite') is True
                    existing = None if overwrite else find_conflicting_profile(name)
                    if existing is not None:
                        self._send_workflow_profiles_list()
                        if hasattr(self, 'vue_bridge'):
                            self.vue_bridge.showNotification.emit(
                                'warning',
                                f"같은 파일 이름의 프로파일 '{existing}'이(가) 이미 있어 저장하지 않았습니다",
                            )
                    else:
                        # 알림은 실제로 기록되는 이름 — 'Flux?' 를 쳐도 저장되는 건 'Flux'
                        saved_name = profile_saved_name(name)
                        snap = collect_from_host(self)
                        ok = save_profile(name, snap['fields'], snap['lora_stack'], overwrite=overwrite)
                        if ok:
                            self._send_workflow_profiles_list()
                            if hasattr(self, 'vue_bridge'):
                                self.vue_bridge.showNotification.emit(
                                    'success', f'프로파일 저장: {saved_name}'
                                )
                        else:
                            if hasattr(self, 'vue_bridge'):
                                self.vue_bridge.showNotification.emit(
                                    'error', f'프로파일 저장 실패: {saved_name}'
                                )
            elif action == 'workflow_profile_load':
                name = str(payload.get('name', '')).strip()
                if name:
                    from core.workflow_profiles import load_profile, apply_profile_to_host
                    data = load_profile(name)
                    if data:
                        result = apply_profile_to_host(data, self)
                        # 전체 프롬프트 다시 업데이트
                        if hasattr(self, 'update_total_prompt_display'):
                            self.update_total_prompt_display()
                        if hasattr(self, 'vue_bridge'):
                            self.vue_bridge.showNotification.emit(
                                'success',
                                f'프로파일 적용: {name} '
                                f'({len(result["applied"])}개)'
                            )
                    else:
                        if hasattr(self, 'vue_bridge'):
                            self.vue_bridge.showNotification.emit(
                                'error', f'프로파일을 찾을 수 없음: {name}'
                            )
            elif action == 'workflow_profile_delete':
                name = str(payload.get('name', '')).strip()
                if name:
                    from core.workflow_profiles import delete_profile
                    delete_profile(name)
                    self._send_workflow_profiles_list()
            elif action == 'workflow_profile_rename':
                old = str(payload.get('old', '')).strip()
                new = str(payload.get('new', '')).strip()
                if old and new:
                    from core.workflow_profiles import rename_profile
                    if rename_profile(old, new):
                        self._send_workflow_profiles_list()

            # ═══════ 프롬프트 섹션 순서 (사용자 지정) ═══════
            elif action == 'prompt_order_list':
                self._send_prompt_order()
            elif action == 'prompt_order_save':
                new_order = payload.get('order', [])
                if isinstance(new_order, list):
                    from core.prompt_order import save_order
                    save_order([str(k) for k in new_order])
                    self._send_prompt_order()
                    self.update_total_prompt_display()  # 즉시 반영
            elif action == 'prompt_order_reset':
                from core.prompt_order import reset_to_default
                reset_to_default()
                self._send_prompt_order()
                self.update_total_prompt_display()

            # ═══════ PR 8: Instant Wildcards (JSON 인라인) ═══════
            elif action == 'instant_wildcards_list':
                self._send_instant_wildcards_list()
            elif action == 'instant_wildcards_save':
                name = str(payload.get('name', '')).strip()
                lines = payload.get('lines', [])
                if name and isinstance(lines, list):
                    iw = self._get_instant_wildcards()
                    iw.set(name, [str(l) for l in lines])
                    iw.save()
                    self._send_instant_wildcards_list()
            elif action == 'instant_wildcards_delete':
                name = str(payload.get('name', '')).strip()
                if name:
                    iw = self._get_instant_wildcards()
                    iw.delete(name)
                    iw.save()
                    self._send_instant_wildcards_list()

            # ═══════ 이벤트 생성 (EventGen) ═══════
            elif action == 'search_events':
                self._start_event_search(payload)
            elif action == 'export_event_results':
                self._export_event_results(payload)
            elif action == 'import_event_results':
                self._import_event_results()

            elif action == 'event_add_to_queue':
                self._handle_event_generation_request(payload, start_immediately=False)

            elif action == 'event_generate_now':
                self._handle_event_generation_request(payload, start_immediately=True)

            # ═══════ PNG Info 전송/생성 ═══════
            elif action == 'pnginfo_send_prompt':
                # Gallery 'T2I에서 사용'·History '당겨오기'와 같은 판정(core.metadata_actions.prompt_transfer) —
                # 파라미터만 있는 이미지(prompt·negative 모두 빈 값)는 T2I 칸을 비우지 않고 알린다.
                prompt, negative, reason = prompt_transfer(payload)
                if reason:
                    self.vue_bridge.showNotification.emit('warning', reason)
                    return
                self.handle_prompt_only_transfer(prompt, negative)

            elif action == 'pnginfo_generate':
                # PNG Info 가 표시한 core 파싱 결과(prompt/negative/parameters)로 즉시 생성한다.
                # WebUI·ComfyUI 모두 같은 경로 — 레거시 raw split 은 네거티브 없는 이미지의
                # Steps 줄을 프롬프트로 넣고 sampler/size 를 버렸다.
                info = _metadata_for_action(self, payload, _clean_path)
                reason = apply_block_reason(
                    info, comfy_message='여러 프롬프트 또는 해석하지 못한 노드가 있습니다. 내용을 확인해 직접 적용하세요.')
                if reason:
                    self.vue_bridge.showNotification.emit('warning', reason)
                    return
                _start_generation_from_metadata(self, info)

            elif action == 'pnginfo_transplant_meta':
                # 보고 있는 이미지의 메타 → 다른 이미지에 박아 새 PNG (대화상자 2개, 원본은 안 건드림)
                transplant_metadata_action(self, payload, _clean_path)

            # ═══════ History 우클릭 — 프롬프트 당겨오기 ═══════
            elif action == 'pull_prompt_from_image':
                path = _clean_path(payload.get('path', ''))
                if path and os.path.exists(path):
                    try:
                        info = json.loads(self.vue_bridge.getImageExif(path))
                        prompt, negative, reason = prompt_transfer(info)
                        if reason:
                            self.vue_bridge.showNotification.emit('warning', reason)
                        else:
                            self.handle_prompt_only_transfer(prompt, negative)
                            self.vue_bridge.showNotification.emit('success', '프롬프트를 당겨왔습니다')
                    except Exception as e:
                        self.vue_bridge.showNotification.emit('error', f'프롬프트 로드 실패: {e}')

            # ═══════ History 우클릭 — 다음 큐에 추가 ═══════
            elif action == 'add_image_to_queue':
                path = _clean_path(payload.get('path', ''))
                if path and os.path.exists(path):
                    try:
                        info = json.loads(self.vue_bridge.getImageExif(path))
                        qp = self._build_queue_payload_from_exif(info)
                        if qp and hasattr(self, 'queue_panel'):
                            self.queue_panel.add_single_item(qp)
                            self.vue_bridge.showNotification.emit('success', '다음 큐에 추가되었습니다')
                        else:
                            self.vue_bridge.showNotification.emit('warning', '이 이미지에 생성 정보가 없습니다')
                    except Exception as e:
                        self.vue_bridge.showNotification.emit('error', f'큐 추가 실패: {e}')

            # ═══════ 배치 파일 열기 ═══════
            elif action in ('open_batch_files', 'open_upscale_files'):
                title = "배치 처리할 이미지 선택" if 'batch' in action else "업스케일할 이미지 선택"
                paths, _ = QFileDialog.getOpenFileNames(self, title, "", "Images (*.png *.jpg *.jpeg *.webp);;All Files (*)")
                if paths:
                    file_list = [p.replace('\\', '/') for p in paths]
                    self.vue_bridge.batchFilesSelected.emit(json.dumps(file_list))
                    self.show_status(f"{len(file_list)} files selected.")

            # ═══════ 캡션 대상 선택 (다중 파일 + 폴더) ═══════
            elif action == 'caption_pick_files':
                paths, _ = QFileDialog.getOpenFileNames(self, "캡션할 이미지 선택", "", "Images (*.png *.jpg *.jpeg *.webp);;All Files (*)")
                if paths:
                    self.vue_bridge.captionFilesSelected.emit(
                        json.dumps([p.replace('\\', '/') for p in paths]))
            elif action == 'caption_pick_folder':
                folder = QFileDialog.getExistingDirectory(self, "캡션할 폴더 선택")
                if folder:
                    import glob
                    imgs = []
                    for ext in ('*.png', '*.jpg', '*.jpeg', '*.webp', '*.bmp'):
                        imgs.extend(glob.glob(os.path.join(folder, ext)))
                        imgs.extend(glob.glob(os.path.join(folder, ext.upper())))
                    imgs = sorted(set(p.replace('\\', '/') for p in imgs))
                    if imgs:
                        self.vue_bridge.captionFilesSelected.emit(json.dumps(imgs))
                        self.vue_bridge.showNotification.emit('info', f'{len(imgs)}개 이미지 발견')
                    else:
                        self.vue_bridge.showNotification.emit('warning', '이미지가 없습니다')
            elif action == 'caption_pick_outdir':
                folder = QFileDialog.getExistingDirectory(self, "캡션 저장 폴더 선택")
                if folder:
                    # 캡션 슬롯은 서버가 승인한 폴더에만 .txt 를 읽고 쓴다(core/caption_out_dir.py).
                    # 승인은 이 호스트 대화상자에서만 — 웹 클라이언트가 outDir 로 아무 폴더나 가리키지 못한다.
                    from core.caption_out_dir import CaptionOutDirError
                    try:
                        self.vue_bridge.approve_caption_out_dir(folder)
                    except CaptionOutDirError as exc:
                        from ui.status_line import notify_user
                        notify_user(self, 'warning', str(exc))
                    else:
                        self.vue_bridge.captionOutDirSelected.emit(folder.replace('\\', '/'))
            elif action == 'caption_pick_caformer_dir':
                folder = QFileDialog.getExistingDirectory(self, "CAFormer 모델 폴더 선택")
                if folder:
                    self.vue_bridge.captionModelDirSelected.emit(folder.replace('\\', '/'))

            # ═══════ 클립보드 복사 ═══════
            elif action == 'copy_to_clipboard':
                path = payload.get('path', '')
                if getattr(self, 'web_mode', False):
                    # 웹 클라이언트는 호스트 PC 클립보드를 쓰지 않는다 — 브리지 정책
                    # (core.web_action_policy.WEB_BLOCKED_ACTIONS)이 먼저 막지만, open_url 처럼
                    # 분기에서도 한 번 더 거른다. 브라우저가 자기 클립보드에 복사한다.
                    from core.web_action_policy import COPY_IMAGE_MESSAGE
                    from ui.status_line import notify_user
                    notify_user(self, 'warning', COPY_IMAGE_MESSAGE)
                elif path:
                    clean = _clean_path(path)
                    from PyQt6.QtGui import QPixmap
                    pix = QPixmap(clean)
                    # 버튼만 누르고 끝나는 동작이라 결과를 토스트로 알린다 — 예전엔 성공도 실패도 무음.
                    from ui.status_line import notify_user
                    if not pix.isNull():
                        QApplication.clipboard().setPixmap(pix)
                        self.show_status("Copied to clipboard.")
                        notify_user(self, 'success', '이미지를 클립보드에 복사했습니다')
                    else:
                        self.show_status("Copy to clipboard failed.")
                        notify_user(self, 'error', '이미지를 읽지 못해 클립보드에 복사하지 못했습니다')

            # ═══════ YOLO 모델 초기화 ═══════
            elif action == 'editor_clear_yolo_models':
                # 예전엔 config 만 [] 로 비워서 Editor_models 자동 감지분이 그대로 쓰였다 — 지금 감지된
                # 모델을 비활성으로 기록해야 auto_detect/auto_censor 와 라벨이 실제로 초기화된다.
                from core import yolo_models
                try:
                    model_paths = yolo_models.clear_models()
                except Exception as e:
                    self.vue_bridge.showNotification.emit('error', f'YOLO 모델 초기화 실패: {e}')
                    return
                self.vue_bridge.yoloModelUpdated.emit(yolo_models.model_label(model_paths))
                self.show_status("YOLO models cleared.")

            # ═══════ Gallery EXIF → T2I ═══════
            elif action == 'gallery_send_exif_to_t2i':
                # 확대 뷰가 보낸 metadata(편집한 프롬프트 포함)를 그대로 쓰고, 경로만 오면
                # (즐겨찾기·구 페이로드) getImageExif 로 읽는다. raw 를 다시 split 하지 않는다.
                info = _metadata_for_action(self, payload, _clean_path)
                prompt, negative, reason = prompt_transfer(
                    info, comfy_message='ComfyUI 프롬프트가 여러 갈래입니다. 메타데이터에서 내용을 확인하세요.')
                if reason:
                    self.vue_bridge.showNotification.emit('warning', reason)
                    return
                self.handle_prompt_only_transfer(prompt, negative)

            # ═══════ Gallery 폴더 열기 다이얼로그 ═══════
            elif action == 'gallery_open_folder':
                from config import OUTPUT_DIR
                last = self.vue_bridge.getLastGalleryFolder() or OUTPUT_DIR
                folder = QFileDialog.getExistingDirectory(self, "Gallery 폴더 선택", last)
                if folder:
                    self.vue_bridge._save_gallery_folder(folder)
                    self.vue_bridge.galleryFolderLoaded.emit(folder.replace('\\', '/'))

            # ═══════ 즐겨찾기 제거 ═══════
            elif action == 'remove_favorite':
                path = payload.get('path', '')
                if path:
                    # 이미 지운 이미지·분리된 드라이브의 항목도 지울 수 있어야 한다 —
                    # 예전엔 '존재하는 것만' 걸러 읽어서 깨진 항목이 영영 안 지워졌다.
                    # 파일을 잠깐 못 읽으면(공유 위반 등) 저장하지 않고 오류로 알린다.
                    from core.favorites import FavoritesUnavailableError, remove_favorite
                    try:
                        _removed, _saved = remove_favorite(_clean_path(path))
                    except (FavoritesUnavailableError, OSError) as fav_err:
                        from core.error_handler import sanitize_for_ui
                        if hasattr(self, 'vue_bridge'):
                            self.vue_bridge.showNotification.emit(
                                'error', f'즐겨찾기에서 제거하지 못했습니다: {sanitize_for_ui(str(fav_err))}')
                    else:
                        self.show_status("Removed from favorites.")

            # ═══════ API 관리자 ═══════
            elif action == 'show_api_manager':
                try:
                    # Settings에서 호출 시 X버튼=종료 방지
                    self._api_manager_mode = True
                    self._startup_backend_check()
                    self._apply_backend_startup_result()
                    self._api_manager_mode = False
                    self.vue_bridge.showNotification.emit('success', 'API 연결 확인 완료')
                except SystemExit:
                    # X버튼으로 다이얼로그 닫힘 — 앱 종료 안 함
                    self._api_manager_mode = False
                    self.vue_bridge.showNotification.emit('info', 'API 설정 취소됨')
                except Exception as e:
                    self._api_manager_mode = False
                    self.vue_bridge.showNotification.emit('error', f'API 연결 실패: {e}')

            # ═══════ 시작 백엔드 게이트 (창 위의 Vue 오버레이) ═══════
            # 옛 QDialog 가 하던 세 가지 — 감지 / 확정 / 워크플로 고르기 — 를
            # 그대로 액션으로 옮긴 것. 실제 로직은 WebUIMixin 이 갖고 있다.
            elif action == 'probe_backend':
                self._probe_backends_async(
                    str(payload.get('webuiUrl') or '').strip(),
                    str(payload.get('comfyUrl') or '').strip(),
                )

            elif action == 'select_backend':
                self._select_backend_from_gate(payload)

            elif action == 'pick_comfy_workflow':
                self._pick_comfy_workflow()

            # ═══════ 탭 순서 설정 ═══════
            elif action == 'set_tab_order':
                # 저장 자체는 앞서 온 save_ui_prefs(tabOrder)가 한다 — 여기선 하단 상태줄 확인 한 줄만.
                order = payload.get('order')
                if isinstance(order, list) and order:
                    self.show_status(f"탭 순서 저장됨 ({len(order)}개 탭)")

            # ═══════ 시드 탐색 (3x3 그리드) ═══════
            elif action == 'explore_seed':
                # 완성 payload 를 한 번 동결해 9개 변형(Forge: subseed, Comfy: 이웃 시드)을 넣는다.
                # 대기열은 그 payload 를 그대로 보낸다(_on_generation_requested) — ui/seed_explore_actions.
                from ui.seed_explore_actions import start_seed_explore
                start_seed_explore(self, payload)

            # ═══════ 비교 이미지 ═══════
            elif action == 'open_compare_image':
                slot = payload.get('slot', 'before')
                path, _ = QFileDialog.getOpenFileName(self, "비교 이미지 선택", "", "Images (*.png *.jpg *.jpeg *.webp)")
                if path:
                    self.vue_bridge.compareImageLoaded.emit(json.dumps({'slot': slot, 'path': path.replace('\\', '/')}))

            elif action == 'open_url':
                # Windows 의 webbrowser.open 은 os.startfile 이다 — 호스트가 있는 http/https 만
                # 열고(file:·드라이브·UNC·search-ms: 거부), 웹 모드에선 브라우저가 직접 연다.
                from core.url_safety import open_external_url
                opened, message = open_external_url(
                    payload.get('url', ''), web_mode=bool(getattr(self, 'web_mode', False)),
                )
                if not opened and message and hasattr(self, 'vue_bridge'):
                    self.vue_bridge.showNotification.emit('warning', message)

            elif action == 'send_to_compare':
                path = payload.get('path', '')
                slot = payload.get('slot', 'after')
                if path:
                    clean = _clean_path(path).replace('\\', '/')
                    if clean.startswith('/') and ':' in clean[1:3]: clean = clean[1:]
                    self.vue_bridge.tabChanged.emit('png')
                    QTimer.singleShot(100, lambda: self.vue_bridge.compareImageLoaded.emit(json.dumps({'slot': slot, 'path': clean})))

            # ═══════ UI 설정 저장 ═══════
            elif action == 'save_ui_prefs':
                try:
                    from core.config_migration import load_ui_prefs, save_ui_prefs
                    from core.ui_prefs import ui_prefs_path
                    prefs_path = ui_prefs_path()
                    prefs = load_ui_prefs(prefs_path)
                    for legacy_key in (
                        'hires_enabled',
                        'ad_enabled',
                        'sam3_enabled',
                        'ad_s1_enabled',
                        'ad_s2_enabled',
                        'negpip_enabled',
                    ):
                        prefs.pop(legacy_key, None)
                        payload.pop(legacy_key, None)
                    # 캡션 저장 폴더 승인 목록은 서버 전용이라 클라이언트가 쓰지 못하고, captionOutDir 은
                    # ''(이미지 옆)·승인된 폴더만 받는다(core/caption_out_dir.filter_client_caption_prefs).
                    from core.caption_out_dir import filter_client_caption_prefs
                    filter_client_caption_prefs(payload, self.vue_bridge.approved_caption_out_dirs())
                    prefs.update(payload)
                    # Anima Guard 값은 파일에 쓰기 전에 안전 범위/8배수로 정규화.
                    from core.resolution_guard import normalize_anima_guard_prefs
                    prefs.update(normalize_anima_guard_prefs(prefs))
                    save_ui_prefs(prefs_path, prefs)
                    # FIX: Vue의 LOGIC 토글을 prompt_cleaner에 즉시 적용
                    # (이전에는 저장만 되고 효과 없었음)
                    self._apply_ui_prefs_to_cleaner(prefs)
                    # 실행 중인 자동화도 다음 생성부터 새 제한값을 사용.
                    self._apply_anima_guard_prefs(prefs)
                    # Forge 출력 폴더 저장 설정 — 생성 워커는 파일이 아니라 이 메모리 값을 읽는다.
                    from core.forge_output_policy import update_forge_save_outputs_from_prefs
                    update_forge_save_outputs_from_prefs(prefs)
                    # 테마는 PyQt 쪽 색표에도 반영 — 다음에 뜨는 다이얼로그/스플래시가
                    # Vue 와 같은 색이어야 한다. (theme 키가 안 왔으면 no-op)
                    if 'theme' in payload or 'themeOverrides' in payload:
                        self._apply_theme_prefs(prefs)
                    # 재전송 하지 않음 — uiPrefsLoaded는 앱 시작 시에만 emit
                    # 재전송하면 watch → save → emit → watch 무한 루프 발생
                except Exception as e:
                    self.vue_bridge.showNotification.emit('error', f'UI 설정 저장 실패: {e}')

            # ═══════ 글로벌 가중치 저장 ═══════
            elif action == 'save_global_weights':
                try:
                    weights = payload.get('weights', [])
                    from core.storage_paths import config_file
                    atomic_write_json(str(config_file('global_weights.json')), weights)
                    self.vue_bridge.showNotification.emit('success', '가중치가 저장되었습니다')
                except Exception as e:
                    self.vue_bridge.showNotification.emit('error', f'가중치 저장 실패: {e}')

            # ═══════ LoRA 스택 (생성 LoRA 의 단일 소스) ═══════
            # 예전의 set_lora_text 미러는 '활성 LoRA 가 있을 때만' 갱신돼 전부 끄면 옛 LoRA 가
            # 계속 붙었다. 이제 생성은 매번 _vue_lora_entries 에서 텍스트를 파생한다
            # (core/lora_stack.append_lora_stack_to_prompt). Vue 는 빈 스택도 반드시 보낸다.
            elif action == 'set_lora_stack':
                from core.lora_stack import UNIT_MULTIPLIER, normalize_lora_entries
                entries = payload.get('entries', [])
                self._vue_lora_entries = normalize_lora_entries(
                    entries if isinstance(entries, list) else [], unit=UNIT_MULTIPLIER)

            # ═══════ 조건부 프롬프트 저장 ═══════
            elif action == 'save_cond_rules':
                # 제어 플래그 _manual 은 파일에 남기지 않는다(getInitialConfig 로 되돌아간다). 자동저장
                # (편집 800ms 뒤·복원 동기화)은 조용히 쓰고, '즉시 저장' 버튼만 토스트를 띄운다 —
                # 예전엔 부팅마다 복원이 자동저장을 불러 거짓 '저장되었습니다' 가 떴다(감사 #108).
                try:
                    from core.cond_rules_store import save_cond_rules_payload
                    if save_cond_rules_payload(payload):
                        self.vue_bridge.showNotification.emit('success', '조건식이 저장되었습니다')
                except Exception as e:
                    self.vue_bridge.showNotification.emit('error', f'조건식 저장 실패: {e}')

            # ═══════ 기본값 저장 ═══════
            elif action == 'save_tab_defaults':
                # 바뀐 키만 온다 — 기존 파일 위에 병합한다(core.tab_defaults). 통째로 쓰면 keep-alive
                # Settings 의 옛 값이 '전역 저장'이 갱신한 T2I 기본값을 되덮었다(audit #140).
                try:
                    from core.tab_defaults import save_tab_defaults_patch
                    save_tab_defaults_patch(payload)
                    self.vue_bridge.showNotification.emit('success', '기본값이 저장되었습니다')
                except Exception as e:
                    self.vue_bridge.showNotification.emit('error', f'기본값 저장 실패: {e}')

            # ═══════ 대기열 제어 ═══════
            elif action == 'start_queue':
                if hasattr(self, 'queue_manager'):
                    started = self.queue_manager.start()
                    if started is False:
                        # 눌렀는데 아무 일도 없으면 고장처럼 보인다 — 사유(자동화가 대기열을 먼저 처리 중 ·
                        # 빈 대기열)를 상태줄과 토스트로 알린다.
                        from ui.queue_coordination import announce
                        refusal = getattr(self.queue_manager, 'last_start_refusal', '') or ''
                        announce(self, refusal or '대기열이 비어 있어 시작할 항목이 없습니다')
                    elif not self.queue_manager.is_paused:
                        # 시작하자마자 멈췄으면(워커가 다른 생성 중) 매니저의 notice 가 이미 사유를 알렸다
                        self.show_status("Queue started.")
                    self._sync_queue_to_vue()

            elif action == 'stop_queue':
                if hasattr(self, 'queue_manager'):
                    self.queue_manager.stop()
                    self.show_status("Queue stopped.")
                    self._sync_queue_to_vue()

            elif action == 'unload_model_request':
                # VRAM 게이지 클릭으로 사용자가 수동 unload 요청 — 백엔드에 위임
                # 앱 프로세스가 직접 쥔 편집기 비전 모델(YOLO/SAM3 유휴 캐시)도 즉시 반납한다.
                # (백엔드 언로드만으로는 에디터 SAM3 번들 ~3.4GB가 그대로 남았다)
                try:
                    from core.model_cache import clear_all as _clear_editor_models
                    _freed = _clear_editor_models()
                    if _freed:
                        print(f"[VRAM] 편집기 비전 모델 {_freed}개 반납")
                except Exception as e:
                    print(f"[VRAM] 편집기 비전 모델 반납 실패: {e}")
                try:
                    # 백엔드 HTTP(WebUI cleanup 최대 ~60초)는 워커에서, 성공/실패는 메인
                    # 스레드에서 알린다 — ui/manual_model_unload.py
                    from backends import get_backend
                    from ui.manual_model_unload import request_manual_backend_unload
                    request_manual_backend_unload(self, get_backend())
                except Exception as e:
                    self.show_status(f"Unload failed: {e}")
                    from ui.status_line import notify_user
                    notify_user(self, 'error', f'모델 언로드 요청 실패: {e}')

            elif action == 'pause_queue':
                if hasattr(self, 'queue_manager'):
                    self.queue_manager.pause()
                    self.show_status("Queue paused.")
                    self._sync_queue_to_vue()

            elif action == 'resume_queue':
                if hasattr(self, 'queue_manager'):
                    self.queue_manager.resume()
                    self.show_status("Queue resumed.")
                    self._sync_queue_to_vue()

            elif action == 'remove_queue_items':
                # payload: { item_ids: [str, ...] }  — 여러 항목 일괄 삭제. 생성 중인 항목은 보호된다.
                # (Vue 동기화는 queue_changed → _sync_queue_to_vue 가 한 번으로 합쳐 보낸다)
                if hasattr(self, 'queue_panel'):
                    from ui.queue_coordination import RUNNING_ITEM_KEPT, announce, running_item_kept
                    ids = payload.get('item_ids', []) or []
                    n = self.queue_panel.remove_items_by_ids(ids)
                    # 중지해도 워커가 만들고 있는 항목은 끝날 때까지 보호된다 — '중지하면 지울 수 있다'가 아니다
                    if running_item_kept(self.queue_panel, ids):
                        announce(self, RUNNING_ITEM_KEPT)
                    else:
                        self.show_status(f"{n}개 항목 삭제")

            elif action == 'move_queue_item':
                # payload: { item_id: str, direction: 'up' | 'down' } — 생성 중인 항목과는 자리를 바꾸지 않는다
                if hasattr(self, 'queue_panel'):
                    iid = payload.get('item_id', '')
                    direction = payload.get('direction', 'up')
                    if direction == 'up':
                        self.queue_panel.move_item_up(iid)
                    else:
                        self.queue_panel.move_item_down(iid)

            elif action == 'update_queue_item':
                # payload: { item_id: str, prompt?: str, negative_prompt?: str } — 큐 항목 편집
                if hasattr(self, 'queue_panel'):
                    iid = payload.get('item_id', '')
                    fields = {key: payload.get(key, '') for key in ('prompt', 'negative_prompt')
                              if key in payload}
                    if self.queue_panel.update_item(iid, fields):
                        self.show_status("큐 항목 수정됨")

            elif action == 'clear_queue':
                # 전체 삭제 — 확인 다이얼로그 우회 (Vue 측에서 이미 확인 받음). 생성 중인 항목은 남는다 —
                # 조용히 한 줄이 남으면 고장처럼 보이므로 선택 삭제와 같은 안내를 띄운다.
                if hasattr(self, 'queue_panel'):
                    from ui.queue_coordination import RUNNING_ITEM_KEPT, announce, running_item_kept
                    removed = self.queue_panel.clear_items()
                    if running_item_kept(self.queue_panel):
                        announce(self, RUNNING_ITEM_KEPT)
                    else:
                        self.show_status(f"{removed}개 항목 삭제")

            elif action == 'sync_queue_state':
                # Vue 대기열 패널이 마운트될 때(페이지 로드·웹 재접속) 현재 상태를 요청한다 —
                # 시작 시 복구된 대기열은 Vue 가 뜨기 전에 알려져 그냥은 보이지 않았다.
                self._sync_queue_to_vue()

            # ═══════ Toast 표시 ═══════
            elif action == 'show_toast':
                t = payload.get('type', 'info')
                m = payload.get('msg', '')
                if hasattr(self, 'vue_bridge'):
                    self.vue_bridge.showNotification.emit(t, m)

            # ═══════ 미처리 액션 로그 ═══════
            else:
                print(f"[Bridge] Unhandled action: {action}")

        except Exception as e:
            from core.error_handler import handle_error
            handle_error('E010', f'Action: {action}', e)

    def _start_anima_forge_import(self) -> None:
        """Forge가 공개한 Anima script-info 값을 UI 멈춤 없이 가져온다."""
        if getattr(self, '_anima_forge_import_inflight', False):
            self.vue_bridge.showNotification.emit('info', 'Forge Anima 설정을 이미 가져오는 중입니다')
            return

        self._anima_forge_import_inflight = True
        self.vue_bridge.showNotification.emit('info', 'Forge에서 Anima 설정을 가져오는 중...')
        threading.Thread(
            target=self._fetch_anima_forge_import,
            daemon=True,
            name='anima-forge-import',
        ).start()

    def _fetch_anima_forge_import(self) -> None:
        result = {}
        try:
            from backends import BackendType, get_backend, get_backend_type
            if get_backend_type() is not BackendType.WEBUI:
                raise RuntimeError('현재 백엔드가 Forge/WebUI가 아닙니다')

            backend = get_backend()
            api_url = str(getattr(backend, 'api_url', '') or '').rstrip('/')
            if not api_url:
                raise RuntimeError('Forge API 주소가 비어 있습니다')

            import requests
            response = requests.get(f'{api_url}/sdapi/v1/script-info', timeout=8)
            response.raise_for_status()

            from core.anima_guidance import parse_forge_script_info
            settings, meta = parse_forge_script_info(response.json())
            result = {
                'settings': settings,
                'meta': meta,
                'api_url': api_url,
            }
        except Exception as exc:
            result = {'error': str(exc)[:400]}

        try:
            self.animaForgeImportReady.emit(result)
        except RuntimeError:
            # 종료 중 메인 QObject가 이미 정리된 경우에는 결과를 버린다.
            pass

    @pyqtSlot(object)
    def _apply_anima_forge_import(self, result) -> None:
        """worker 결과를 GUI 스레드에서 프록시와 Vue에 원자적으로 적용한다."""
        self._anima_forge_import_inflight = False
        if not isinstance(result, dict) or result.get('error'):
            message = result.get('error', '알 수 없는 오류') if isinstance(result, dict) else '잘못된 응답'
            self.vue_bridge.showNotification.emit('error', f'Forge 설정 가져오기 실패: {message}')
            return

        widgets = getattr(self, 'anima_guidance_widgets', None) or {}
        settings = result.get('settings') or {}
        updated = 0
        self.vue_bridge.beginBatchUpdate()
        try:
            for key, value in settings.items():
                proxy = widgets.get(key)
                if proxy is None or not hasattr(proxy, 'setText'):
                    continue
                if isinstance(value, bool):
                    text = 'true' if value else 'false'
                elif value is None:
                    text = ''
                else:
                    text = str(value)
                proxy.setText(text)
                updated += 1
        finally:
            self.vue_bridge.endBatchUpdate()

        meta = result.get('meta') or {}
        ignored = int(meta.get('ignored_trailing_args') or 0)
        missing = meta.get('missing_scripts') or []
        details = []
        if ignored:
            details.append(f'신버전 후행 인자 {ignored}개 제외')
        if missing:
            details.append(f'미설치 스크립트 {len(missing)}개')
        suffix = f" ({', '.join(details)})" if details else ''
        level = 'warning' if missing else 'success'
        self.vue_bridge.showNotification.emit(
            level,
            f'Forge Anima 설정 {updated}개를 적용했습니다{suffix} · 저장하려면 SAVE를 누르세요',
        )

    @pyqtSlot(str)
    def _on_backend_runtime_event(self, payload_json: str) -> None:
        """managed runtime worker 결과를 GUI 스레드의 기존 backend seam에 연결."""
        try:
            event = json.loads(payload_json or '{}')
        except Exception:
            return
        if not isinstance(event, dict):
            return

        event_type = str(event.get('type') or '')
        action = str(event.get('action') or '')
        engine = str(event.get('engine') or '')
        startup = bool(event.get('startup', False))

        if startup and event_type in {'started', 'progress'}:
            # 진행 문구는 Vue 설정(런타임 · 엔진)이 backendRuntimeEvent 로 직접 받는다.
            return

        if event_type == 'completed' and bool(event.get('ok')):
            if action in {
                'set_install_root', 'use_managed_install',
                'set_primary_model_engine',
            }:
                # Runtime topology changes invalidate both the API-backed model
                # choices and the merged disk LoRA catalog.  Re-query on the GUI
                # thread so Settings changes are reflected without an app restart.
                try:
                    from ui.lora_catalog_cache import invalidate as invalidate_lora_cache
                    invalidate_lora_cache(getattr(self, 'vue_bridge', None))
                except Exception:
                    pass
                if getattr(self, '_backend_connected', False):
                    QTimer.singleShot(0, self.load_webui_info)

            should_activate = bool(event.get('activate')) and action in {'start', 'use'}
            if should_activate and action == 'start':
                # START may replace the opposite app-owned runtime, but the app
                # can still be connected to an unrelated user.bat/external URL.
                # Only follow that replacement when the current connection is
                # the exact managed endpoint that was stopped. USE remains an
                # explicit request to switch regardless of the current URL.
                result = event.get('result') if isinstance(event.get('result'), dict) else {}
                replaced_engine = str(result.get('replacedEngine') or '')
                if replaced_engine:
                    snapshot = event.get('snapshot') if isinstance(event.get('snapshot'), dict) else {}
                    engines = snapshot.get('engines') if isinstance(snapshot.get('engines'), dict) else {}
                    replaced_state = engines.get(replaced_engine)
                    replaced_state = replaced_state if isinstance(replaced_state, dict) else {}
                    replaced_url = str(replaced_state.get('apiUrl') or '')
                    try:
                        from backends import get_backend
                        current_url = str(getattr(get_backend(), 'api_url', '') or '')
                    except Exception:
                        current_url = ''
                    should_activate = _same_api_endpoint(current_url, replaced_url)

            if should_activate:
                result = event.get('result') if isinstance(event.get('result'), dict) else {}
                state = event.get('state') if isinstance(event.get('state'), dict) else {}
                url = str(
                    result.get('apiUrl') or result.get('url') or state.get('apiUrl') or ''
                ).strip()
                if not url:
                    self._backend_connected = False
                    self._managed_runtime_startup_inflight = False
                    self._backend_startup_result = 'managed_failed'
                    self._managed_runtime_startup_error = 'runtime이 API URL을 반환하지 않았습니다'
                    self.vue_bridge.showNotification.emit(
                        'error', '백엔드는 시작됐지만 API URL을 확인하지 못했습니다'
                    )
                    return
                try:
                    from backends import BackendType, set_backend

                    backend_type = (
                        BackendType.WEBUI if engine == 'forge' else BackendType.COMFYUI
                    )
                    set_backend(backend_type, url)
                    self._backend_startup_result = 'managed_connected'
                    # startup 연결 실패는 modal 대신 offline으로 폴백해야 하므로
                    # info worker 결과가 올 때까지 이 표식을 유지한다.
                    self._managed_runtime_startup_inflight = startup
                    self._backend_connected = False
                    if hasattr(self, 'save_settings'):
                        self.save_settings()
                    # startup apply 전 성공은 _apply_backend_startup_result가 연결하고,
                    # apply가 이미 pending으로 지나간 뒤 성공하면 여기서 연결한다.
                    # 이 경계가 없으면 타이밍에 따라 info worker가 두 번 시작된다.
                    if not startup or getattr(self, '_managed_runtime_startup_apply_done', False):
                        self.load_webui_info()
                    if not startup:
                        label = 'Forge Neo' if engine == 'forge' else 'ComfyUI'
                        self.vue_bridge.showNotification.emit(
                            'success', f'{label}를 시작하고 현재 백엔드로 선택했습니다'
                        )
                except Exception as exc:
                    self._backend_connected = False
                    self._managed_runtime_startup_inflight = False
                    self._backend_startup_result = 'managed_failed'
                    self._managed_runtime_startup_error = str(exc)
                    self.vue_bridge.showNotification.emit(
                        'error', f'실행된 백엔드 연결 실패: {exc}'
                    )
                return

            if action == 'stop':
                try:
                    from backends import BackendType, get_backend, get_backend_type

                    stopped_type = (
                        BackendType.WEBUI if engine == 'forge' else BackendType.COMFYUI
                    )
                    result = event.get('result') if isinstance(event.get('result'), dict) else {}
                    state = event.get('state') if isinstance(event.get('state'), dict) else {}
                    stopped_url = str(
                        result.get('apiUrl') or result.get('url') or state.get('apiUrl') or ''
                    )
                    current_url = str(getattr(get_backend(), 'api_url', '') or '')
                    if (
                        result.get('stopped') is True
                        and result.get('owned') is True
                        and get_backend_type() == stopped_type
                        and _same_api_endpoint(current_url, stopped_url)
                    ):
                        self._backend_connected = False
                        self.btn_generate.setEnabled(False)
                except Exception:
                    pass
            return

        if event_type == 'error':
            error = event.get('error')
            if isinstance(error, dict):
                message = str(error.get('message') or error.get('error') or error)
                error_code = str(error.get('code') or '')
            else:
                message = str(error or event.get('message') or '알 수 없는 오류')
                error_code = ''
            if startup:
                self._backend_connected = False
                self._managed_runtime_startup_inflight = False
                self._backend_startup_result = 'managed_failed'
                self._managed_runtime_startup_error = message
                self.btn_generate.setEnabled(False)
            elif error_code == 'BACKEND_SWITCH_ROLLBACK_FAILED':
                self._backend_connected = False
                self.btn_generate.setEnabled(False)
            self.vue_bridge.showNotification.emit(
                'error', f'백엔드 {action} 실패: {message}'
            )

    def _persist_search_results(
        self,
        active,
        full=None,
        *,
        dataset_identity=None,
        snapshot_id=None,
    ):
        """검색 결과 디스크 영속화 — 단일 쓰기 경로(중복 제거).
        dataset label이 포함된 cache/search envelope가 디스크 단일 소스다.
        메모리(filtered_results/shuffled_prompt_deck)는 런타임 캐시,
          active: 표시/자동화 덱용(필터 적용) 셋 (항상)
          full  : '필터 해제' 베이스(전체) 셋 (새 검색 때만 전달)
        NOTE: os/json은 모듈 레벨 import — 여기서 지역 재import 금지(UnboundLocalError 회피).
        대용량(수십만 행) 직렬화+쓰기를 GUI 스레드에서 하면 검색 완료 시 UI가 멈추므로
        백그라운드 데몬 스레드로 수행한다. 요청 순번은 GUI 스레드에서 먼저 예약해
        늦게 시작한 구형 writer가 최신 결과를 덮지 못하게 한다."""
        from core.search_result_store import reserve_write_sequence
        expected_identity = dataset_identity or getattr(
            self,
            '_search_dataset_identity',
            None,
        )
        expected_snapshot = snapshot_id or getattr(
            self,
            '_search_snapshot_id',
            None,
        )
        write_sequence = reserve_write_sequence(expected_snapshot)

        def _write():
            try:
                from core.search_result_store import SearchResultStore
                saved = SearchResultStore().save_if_latest(
                    write_sequence,
                    active,
                    full=full,
                    snapshot_id=expected_snapshot,
                    expected_identity=expected_identity,
                )
                if not saved:
                    print(
                        f"[Search] 오래된 디스크 저장 요청 건너뜀: {write_sequence}"
                    )
            except Exception as e:
                print(f"[Search] 디스크 영속 실패: {e}")

        import threading
        threading.Thread(target=_write, daemon=True).start()

    def _restore_runtime_prefs(self, prefs: dict):
        """ui_prefs.json(단일 소스)에서 런타임 상태를 직접 복원.
        Vue가 set_rating_filter / set_high_res_factor 를 아직 안 보낸 시점(예: 재시작 직후
        자동화 즉시 시작)에도 올바른 rating 필터/고해상도 배율로 동작하도록 한다."""
        self._apply_anima_guard_prefs(prefs)
        try:
            rf = prefs.get('ratingFilter')
            if isinstance(rf, list) and len(rf) == 4:
                keys = ('g', 's', 'q', 'e')
                self._rating_filter = {keys[i] for i, on in enumerate(rf) if on}
        except Exception:
            pass

        try:
            if prefs.get('highResEnabled'):
                factor = float(prefs.get('highResFactor', 1.5) or 1.5)
                self._high_res_factor = max(1.0, min(4.0, factor))
            else:
                self._high_res_factor = 1.0
        except Exception:
            pass
        try:
            lora = prefs.get('loraStack')
            if isinstance(lora, list):
                # 생성 LoRA 의 단일 소스 — Vue가 set_lora_stack 을 아직 안 보낸 시점(재시작 직후
                # 자동화 즉시 시작 등)에도 올바른 LoRA로 생성되도록 한다. ui_prefs 는 정수 %,
                # _vue_lora_entries 는 set_lora_stack 과 같은 배율 — 반드시 /100 해서 넣는다
                # (섞이면 프로파일 저장·적용에서 LoRA 가 100배가 됐다).
                from core.lora_stack import UNIT_PERCENT, normalize_lora_entries
                self._vue_lora_entries = normalize_lora_entries(lora, unit=UNIT_PERCENT)
        except Exception:
            pass
        try:
            # Forge 출력 폴더 저장(save_images) 설정 — 생성 워커는 이 메모리 값만 읽는다
            # (워커 스레드가 ui_prefs.json 을 열면 GUI 의 os.replace 저장이 Windows 에서 실패).
            from core.forge_output_policy import update_forge_save_outputs_from_prefs
            update_forge_save_outputs_from_prefs(prefs)
        except Exception:
            pass

    def _apply_anima_guard_prefs(self, prefs: dict):
        """Anima Guard 설정을 생성 경로가 읽는 런타임 값으로 반영."""
        try:
            from core.resolution_guard import normalize_anima_guard_prefs
            guard = normalize_anima_guard_prefs(prefs)
            area_side = guard['animaGuardMaxAreaSide']
            self._anima_guard_enabled = guard['animaGuardEnabled']
            self._anima_guard_max_area_side = area_side
            self._anima_guard_max_area = area_side * area_side
            self._anima_guard_max_side = guard['animaGuardMaxSide']
        except Exception as e:
            print(f"[AnimaGuard] 설정 적용 실패, 기본값 유지: {e}")

    def _migrate_legacy_lora_stack(self, prefs: dict, prefs_path: str):
        """LoRA 스택 단일 소스(ui_prefs.loraStack) 통합 — ui_prefs에 loraStack이 없고
        옛 prompt_settings.active_loras가 있으면 1회 흡수(prefs를 제자리 수정 + 영속)."""
        try:
            if not isinstance(prefs, dict) or isinstance(prefs.get('loraStack'), list):
                return
            ps_path = os.path.join(os.path.dirname(os.path.dirname(__file__)),
                                   'config', 'prompt_settings.json')
            if not os.path.exists(ps_path):
                return
            with open(ps_path, 'r', encoding='utf-8') as f:
                ps = json.load(f)
            legacy = ps.get('active_loras') if isinstance(ps, dict) else None
            if isinstance(legacy, list) and legacy:
                prefs['loraStack'] = legacy
                atomic_write_json(prefs_path, prefs)
                print(f"[Config] Legacy active_loras migrated → ui_prefs.loraStack ({len(legacy)})")
        except Exception as e:
            print(f"[Config] Legacy lora migration skipped: {e}")

    def _migrate_legacy_cond_rules(self, cond_path: str):
        """옛 전역 조건식(prompt_settings.cond_rules_json) → config/cond_rules.json 1회 이관.
        cond_rules.json이 아직 없을 때만 실행하여 레거시 cond_block_editor 데이터 유실을 방지한다."""
        try:
            ps_path = os.path.join(os.path.dirname(os.path.dirname(__file__)),
                                   'config', 'prompt_settings.json')
            if not os.path.exists(ps_path):
                return
            with open(ps_path, 'r', encoding='utf-8') as f:
                ps = json.load(f)
            legacy = ps.get('cond_rules_json', '') if isinstance(ps, dict) else ''
            if not legacy:
                return
            from utils.condition_block import legacy_cond_rules_to_vue
            vue = legacy_cond_rules_to_vue(legacy)
            if not (vue.get('positive') or vue.get('negative')):
                return
            # updatedAt 은 일부러 찍지 않는다(=0, core/cond_rules_store.py 설명). 옛 cond_rules_json 은
            # 더 이상 저장되지 않으므로, 시각이 있는 브라우저 캐시는 늘 이 데이터보다 새 편집이다 —
            # 파일이 지워진 뒤라면 부팅 때 그 캐시가 이겨 최근 규칙이 파일로 복구된다.
            atomic_write_json(cond_path, vue)
            print(f"[Config] Legacy cond rules migrated → cond_rules.json "
                  f"({len(vue['positive'])}P + {len(vue['negative'])}N)")
        except Exception as e:
            print(f"[Config] Legacy cond migration skipped: {e}")

    def _apply_saved_configs(self):
        """앱 시작 시(__init__, 동기) 조건식·기본값·ui_prefs 를 Python 쪽에 적용한다.

        Vue 로 보내는 일은 하지 않는다. Qt·웹 모드 모두 브리지 바인딩 직후 getInitialConfig 로
        cond_rules·global_weights·ui_prefs 를 당겨 가고(bridge.js _requestInitialConfig), 그 응답이
        uiPrefsLoaded/condRulesLoaded/globalWeightsLoaded 로 배달된다. 예전엔 __init__ 기준 1초
        타이머가 한 번 emit 했는데, 스플래시·loadFinished 대기 중에 터지면 JS 가 connect 하기 전이라
        영구히 유실됐고(iconAnimationStyle·globalWeights 는 폴백도 없음), 늦게 터지면 Vue 가 먼저
        보낸 값을 덮었다(감사 #107). 레거시 마이그레이션은 그 pull 보다 먼저 끝나야 해서 동기다.
        """
        try:
            # 조건식 — config/cond_rules.json 단일 소스. 레거시 1회 마이그레이션만 여기서.
            from core.cond_rules_store import cond_rules_path
            cond_path = cond_rules_path()
            if not os.path.exists(cond_path):
                # 레거시 1회 마이그레이션: 옛 prompt_settings.cond_rules_json → cond_rules.json
                self._migrate_legacy_cond_rules(cond_path)
        except Exception as e:
            print(f"[Config] Failed to migrate cond rules: {e}")
        try:
            # 기본값 로드 + 적용
            # FIX: 이전엔 prompt_settings.json 없을 때만 적용 → 사실상 첫 실행만.
            # 이제 처음 실행이면 모든 필드, 이후엔 빈 위젯만 채우기.
            from core.tab_defaults import default_tab_defaults_path, load_tab_defaults
            if os.path.exists(default_tab_defaults_path()):
                defaults = load_tab_defaults()
                from config import PROMPT_SETTINGS_FILE
                first_run = not os.path.exists(PROMPT_SETTINGS_FILE)
                # 첫 실행이면 모든 defaults 적용
                if first_run:
                    if 'hires_enabled' in defaults:
                        self.hires_options_group.setChecked(bool(defaults.get('hires_enabled')))
                    if 'ad_enabled' in defaults:
                        self.adetailer_group.setChecked(bool(defaults.get('ad_enabled')))
                    if 'sam3_enabled' in defaults and hasattr(self, 'sam3_group'):
                        self.sam3_group.setChecked(bool(defaults.get('sam3_enabled')))
                    if 'ad_s1_enabled' in defaults and hasattr(self, 'ad_slot1_group'):
                        self.ad_slot1_group.setChecked(bool(defaults.get('ad_s1_enabled')))
                    if 'ad_s2_enabled' in defaults and hasattr(self, 'ad_slot2_group'):
                        self.ad_slot2_group.setChecked(bool(defaults.get('ad_s2_enabled')))
                    # NegPiP 은 상시 적용 — 기본값 토글이 없다(audit #97).

                # 첫 실행이든 아니든 — 핵심 생성 파라미터를 위젯이 비어있을 때만 채움
                # (사용자가 저장한 값을 덮어쓰지 않음)
                self._apply_tab_defaults_to_empty_widgets(defaults, first_run=first_run)
                print(f"[Config] Tab defaults loaded ({'first run — full' if first_run else 'empty fields only'})")
        except Exception as e:
            print(f"[Config] Failed to load defaults: {e}")
        try:
            from core.config_migration import load_ui_prefs
            from core.ui_prefs import ui_prefs_path
            prefs_path = ui_prefs_path()
            prefs = load_ui_prefs(prefs_path)
            # 레거시 흡수: ui_prefs에 loraStack이 없고 옛 prompt_settings.active_loras가 있으면 1회 이관
            self._migrate_legacy_lora_stack(prefs, prefs_path)
            # LOGIC 토글(clean*) — 키가 없으면 Vue 화면의 기본값. 저장된 적 없어도 화면과 같게 맞춘다.
            self._apply_ui_prefs_to_cleaner(prefs)
            if prefs:
                # PyQt 색표도 같은 prefs 로 맞춘다(ThemeManager 는 파일을 스스로도
                # 읽지만, 마이그레이션 후 값이면 여기서 온 게 최신이다).
                self._apply_theme_prefs(prefs)
                # 단일 소스(ui_prefs.json)에서 런타임 상태 직접 복원 —
                #   Vue가 set_rating_filter/set_high_res_factor를 아직 안 보낸 시점(자동화 즉시 시작 등)에도 올바른 값.
                self._restore_runtime_prefs(prefs)
            print("[Config] UI prefs applied (LOGIC toggles · theme · runtime)")
        except Exception as e:
            print(f"[Config] Failed to load UI prefs: {e}")

    def _apply_vue_conditional_rules(self, pos_rules: list, neg_rules: list):
        """Vue에서 전달된 조건부 프롬프트 규칙 적용 — 규칙 해석은 utils.condition_block.apply_prompt_rules.

        - 조건은 포지티브 전 칸(캐릭터/작품/선행/본문/후행) 기준, 콤마 다중 조건은 AND.
        - add/remove/replace 모두 태그(쉼표 토큰) 단위 — 부분문자열 치환이 아니라서
          'muscular→muscular male' 이 'muscular male' 을 'muscular male male' 로 만들지 않고,
          여러 번 적용해도 결과가 같다(검색 적용 경로는 파일 규칙 + Vue 규칙을 두 번 적용한다).
        - location='after_condition' → 조건 첫 태그가 있는 '그 칸'에서 바로 뒤에 삽입.
        - 바뀐 칸만 다시 쓴다. 쓰는 동안 is_programmatic_change 를 세워, 선행/후행/네거티브에
          붙인 조건부 태그가 on_base_prompts_changed 로 사용자 템플릿(base_*_prompt)에 스며들어
          다음 프롬프트까지 따라가지 않게 한다.
        """
        try:
            from utils.condition_block import apply_prompt_rules, split_tags

            widgets = {
                'character': self.character_input,
                'copyright': self.copyright_input,
                'prefix': self.prefix_prompt_text,
                'main': self.main_prompt_text,
                'suffix': self.suffix_prompt_text,
                'neg': self.neg_prompt_text,
            }
            def _get(w):
                return w.text() if hasattr(w, 'text') else w.toPlainText()
            def _set(w, v):
                (w.setText if hasattr(w, 'text') else w.setPlainText)(v)

            before = {key: split_tags(_get(w)) for key, w in widgets.items()}
            after = apply_prompt_rules(before, pos_rules or [], neg_rules or [])
            changed = [key for key in widgets if after.get(key, before[key]) != before[key]]
            if not changed:
                return
            prev = getattr(self, 'is_programmatic_change', False)
            self.is_programmatic_change = True
            try:
                for key in changed:
                    _set(widgets[key], ', '.join(after[key]))
            finally:
                self.is_programmatic_change = prev
        except Exception as e:
            print(f"[Error] Conditional rules: {e}")

    def _apply_generation_preset(self, preset: dict) -> list:
        """생성 프리셋을 위젯에 적용 — load_settings 와 같은 적용 함수(only_present=True)로
        파일에 있는 키만 바꾼다. 모델은 match_checkpoint 로 맞추고, 못 맞춘 항목은 경고로 돌려준다.
        배치 모드는 예외가 나도 반드시 푼다(예전엔 endBatchUpdate 를 건너뛰어 _batch_mode 가 남았다).

        프리셋의 선행/후행/네거티브는 새 ``base_*`` 템플릿이 된다(sync_base_prompts). 이 함수는
        is_programmatic_change 를 세워 on_base_prompts_changed 가 템플릿을 갱신하지 않으므로, 따로
        맞추지 않으면 다음 랜덤 프롬프트·자동화 사이클이 프리셋 이전 값으로 되돌렸다."""
        from ui.generation_settings_apply import (
            apply_generation_settings, apply_prompt_settings, sync_base_prompts,
        )

        warnings = []
        bridge = getattr(self, 'vue_bridge', None)
        prev = getattr(self, 'is_programmatic_change', False)
        if bridge is not None:
            bridge.beginBatchUpdate()
        self.is_programmatic_change = True
        try:
            apply_prompt_settings(self, preset, only_present=True)
            # 생성 설정 적용이 실패해도 이미 바꾼 프롬프트 칸과 템플릿은 서로 맞게 — 바로 여기서.
            sync_base_prompts(self, preset)
            warnings = apply_generation_settings(self, preset, only_present=True)
        finally:
            self.is_programmatic_change = prev
            if bridge is not None:
                bridge.endBatchUpdate()
        try:
            self.update_total_prompt_display()
        except Exception as e:
            print(f"[Preset] 프롬프트 표시 갱신 실패: {e}")
        return warnings

    # ========== PNG Info → 즉시 생성 ==========

    def _handle_immediate_generation_from_raw(self, raw: str):
        """(호환 래퍼) infotext 문자열 → core 파싱 → 'pnginfo_generate' 와 같은 즉시 생성.

        tests/test_comfy_metadata_actions.py 하네스가 이 이름을 클래스 본문에서 바인딩하므로
        이름은 유지하고, 파싱은 core.image_metadata 에 맡긴다(레거시 split 없음).
        """
        info = infotext_ui_fields(raw)
        reason = apply_block_reason(info)
        if reason:
            self.vue_bridge.showNotification.emit('warning', reason)
            return
        _start_generation_from_metadata(self, info)

    def _build_queue_payload_from_exif(self, info: dict, *, preserve_seed=False):
        """getImageExif 결과(dict)에서 큐 아이템 payload 구성.

        이미지의 프롬프트/샘플러/steps/cfg/해상도는 EXIF에서 그대로 가져오고,
        seed는 -1(새 변형)로 둔다 — 동일 시드 재생성은 같은 이미지라 무의미하므로
        '비슷한 걸 더 생성'이 자연스러운 기본값. 프롬프트 없으면 None 반환.
        파라미터는 core 가 파싱한 ``parameters`` dict 만 쓴다(core.metadata_actions).
        """
        return _queue_item_from_metadata(self, info, preserve_seed=preserve_seed)

    # ========== 유틸리티 메서드 ==========

    def show_status(self, message: str, timeout_ms: int = 5000):
        """상태 한 줄 — 콘솔 + Vue 하단 계기 스트립(statusMessage). timeout_ms=0 이면 다음 문구까지 유지.

        예전엔 더미 라벨(status_message_label = _D())에만 써서 화면에 아무것도 나오지 않았다.
        토스트가 아니다(스텝마다 오는 진행 문구도 있다) — 놓치면 안 되는 결과는 호출처가
        ui.status_line.notify_user 로 따로 토스트를 띄운다. 시그니처는 테스트 스텁과 맞춰 둔다.
        """
        from ui.status_line import publish_status
        publish_status(self, message, timeout_ms)

    def _setup_realtime_cleaning(self):
        def _schedule():
            if not self.is_programmatic_change: self._clean_timer.start()
        for w in [self.char_count_input, self.character_input, self.copyright_input, self.artist_input, self.prefix_prompt_text, self.main_prompt_text, self.suffix_prompt_text, self.neg_prompt_text]:
            if hasattr(w, 'textChanged'): w.textChanged.connect(_schedule)

    def _deferred_clean_all(self):
        if self.is_programmatic_change: return
        self.is_programmatic_change = True
        try:
            for w in [self.char_count_input, self.character_input, self.copyright_input]:
                cleaned = self.prompt_cleaner.clean(w.text())
                if w.text() != cleaned: w.setText(cleaned)
            for w in [self.artist_input, self.prefix_prompt_text, self.main_prompt_text, self.suffix_prompt_text, self.neg_prompt_text]:
                cleaned = self.prompt_cleaner.clean(w.toPlainText())
                if w.toPlainText() != cleaned: w.setPlainText(cleaned)
        finally: self.is_programmatic_change = False

    def _setup_queue(self):
        # 화면 없는 대기열 상태 저장소(widgets/queue_panel.py) — 보이는 대기열은 Vue QueuePanel.vue
        self.queue_panel = QueuePanel()
        self.queue_panel.setParent(None)
        self.queue_manager = QueueManager(self.queue_panel)
        self.queue_manager.generation_requested.connect(self._on_generation_requested)
        self.queue_manager.queue_completed.connect(self._on_queue_completed)
        # 재개 때 이미 GPU 에 있는 항목을 다시 보내지 않게 — 생성 워커가 결과를 냈는지로 판단
        self.queue_manager.generation_active = lambda: self._auto_generation_in_flight()
        # 자동화가 돌면 대기열을 따로 시작하지 않는다(자동화 '큐 우선'이 먼저 처리한다) — ui/queue_coordination.py
        from ui.queue_coordination import announce, queue_start_refusal
        self.queue_manager.start_blocker = lambda: queue_start_refusal(self)
        # 워커가 바빠 보내지 못하고 멈춘 사유(수동 생성 · 자동화) — 상태줄 + 토스트
        self.queue_manager.notice.connect(lambda message: announce(self, message))
        # 일시정지 상태 변경 시 Vue 동기화 (▶/⏸ 토글)
        if hasattr(self.queue_manager, 'paused_changed'):
            self.queue_manager.paused_changed.connect(lambda _p: self._sync_queue_to_vue())
        # 대기열 변경 (추가/삭제/순서/수정/실행 중 표시) 시 Vue 동기화 — 같은 턴의 변경은 한 번으로 합쳐진다
        if hasattr(self.queue_panel, 'queue_changed'):
            self.queue_panel.queue_changed.connect(lambda _n: self._sync_queue_to_vue())
        # add_single_item 래핑 — ComfyUI 워크플로 컨트롤 동결 + queueItemAdded(핀 강조).
        # 일괄 추가(XYZ·시드 탐색·이벤트 시나리오)도 이 래퍼를 한 건씩 거친다(우회 API 없음).
        _orig_add = self.queue_panel.add_single_item
        def _wrapped_add(item):
            # Search/EXIF/automation use the legacy abbreviated payload. Freeze
            # only custom workflow controls; keep their other UI-time semantics.
            from backends import BackendType, get_backend, get_backend_type
            if (isinstance(item, dict) and '_comfy_workflow_snapshot' not in item
                    and '_comfy_queued_controls' not in item
                    and get_backend_type() == BackendType.COMFYUI and not self._is_krea2_generation()):
                from core.comfy_workflow_controls import snapshot_comfy_payload
                frozen = snapshot_comfy_payload(get_backend(), {}, 'txt2img')
                item = {**item, '_comfy_queued_controls': frozen['_comfy_workflow_snapshot']}
            added_id = _orig_add(item)
            # 실제 생성된 항목(id 포함)을 전달 — 원본 item엔 id가 없어 Vue 중복 방지가
            # 동작하지 않던 문제 방지
            actual = None
            try:
                getter = getattr(self.queue_panel, 'get_item_by_id', None)
                if added_id and callable(getter):
                    actual = getter(added_id)
                if actual is None:
                    actual = self.queue_panel.queue_items[-1] if self.queue_panel.queue_items else item
            except Exception:
                actual = item
            self._sync_queue_item_added(actual)
            return added_id
        self.queue_panel.add_single_item = _wrapped_add

    def _queue_vue_sync(self):
        """대기열 → Vue 전송기(ui/queue_vue_sync.py) — 처음 쓸 때 만든다."""
        sync = getattr(self, '_queue_vue_sync_obj', None)
        if sync is None:
            from ui.queue_vue_sync import QueueVueSync
            sync = QueueVueSync(self)
            self._queue_vue_sync_obj = sync
        return sync

    def _sync_queue_item_added(self, item: dict):
        """대기열에 아이템 추가 시 Vue로 전달 — 같은 턴의 추가는 마지막 항목 한 번으로 합친다."""
        if hasattr(self, 'vue_bridge'):
            self._queue_vue_sync().item_added(item)

    def _sync_queue_to_vue(self):
        """전체 대기열 상태를 Vue로 전달(queueUpdated) — 다음 이벤트 루프 턴에 최신 상태 한 번."""
        if not (hasattr(self, 'vue_bridge') and hasattr(self, 'queue_panel')):
            return
        self._queue_vue_sync().request_state()

    def _flush_queue_state(self):
        """모아 둔 대기열 저장을 지금 쓴다 — 앱 종료 직전(_quit_app · 웹 모드 aboutToQuit)."""
        panel = getattr(self, 'queue_panel', None)
        flush = getattr(panel, 'flush_to_disk', None)
        if callable(flush):
            flush()

    def _apply_payload_to_ui(self, item: dict):
        """큐 아이템(payload dict)을 생성 UI에 적용 — 생성 직전 호출.

        큐 아이템의 'prompt'는 이미 완성된 '전체' 프롬프트이므로 total_prompt_display에
        직접 넣는다(start_generation이 total_prompt_display를 그대로 사용, 재계산 안 함).
        ★ 이 메서드가 이전 리팩터에서 누락돼 _on_generation_requested가 미정의 메서드를
        호출 → 큐 항목이 '직전 UI(선행 고정만)'로 생성되던 버그(⑦)의 원인이었음.
        """
        if not isinstance(item, dict):
            return
        self.is_programmatic_change = True
        try:
            if item.get('prompt') is not None:
                self.total_prompt_display.setPlainText(str(item.get('prompt') or ''))
            if item.get('negative_prompt') is not None:
                self.neg_prompt_text.setPlainText(str(item.get('negative_prompt') or ''))
            if item.get('sampler_name'):
                self.sampler_combo.setText(str(item['sampler_name']))
            if item.get('scheduler'):
                self.scheduler_combo.setText(str(item['scheduler']))
            if item.get('steps') is not None:
                self.steps_input.setText(str(item.get('steps')))
            if item.get('cfg_scale') is not None:
                self.cfg_input.setText(str(item.get('cfg_scale')))
            if item.get('seed') is not None:
                self.seed_input.setText(str(item.get('seed')))
            if item.get('width'):
                self.width_input.setText(str(item.get('width')))
            if item.get('height'):
                self.height_input.setText(str(item.get('height')))
        except Exception as e:
            print(f"[Queue] _apply_payload_to_ui 실패: {e}")
        finally:
            self.is_programmatic_change = False

    def _on_generation_requested(self, item: dict):
        # 동결 항목(시드 탐색 · XYZ · ComfyUI 스냅숏/컨트롤)은 굳힌 payload 그대로 보낸다 —
        # 준비 규칙은 ui/queue_item_dispatch 한 곳에 있고 자동화 '큐 우선' 경로도 같이 쓴다.
        # (UI 에서 다시 만들면 subseed·LoRA·hires 같은 동결 설정이 사라진다)
        from ui.queue_item_dispatch import prepare_frozen_generation, start_prepared_generation
        try:
            prepared = prepare_frozen_generation(self, item)
            if prepared is not None:
                if not start_prepared_generation(self, prepared):
                    self.queue_manager.pause()
                return
        except Exception as exc:
            self.queue_manager.pause()
            self._abort_generation(str(exc))
            return
        self._apply_payload_to_ui(item)
        if self.start_generation() is False:
            # 시작하지 못했다(다른 생성 중 · 검증 실패 · 체크포인트 없음) — 항목을 남기고 멈춘다.
            # 예전엔 '실행 중'으로 남아, 다른 생성이 끝날 때 이 항목이 생성 없이 지워졌다.
            self.queue_manager.pause()

    def _on_queue_completed(self, total_count: int):
        self._queue_completed_count = total_count
        # 자연 완료(큐 소진)와 사용자 수동 중지 구분 — 수동 중지면 '완료' 팝업/성공알림 생략
        natural = getattr(getattr(self, 'queue_manager', None), 'last_stop_natural', True)
        if hasattr(self, 'vue_bridge'):
            # Vue running 상태 리셋은 항상 (수동 중지에도 큐가 멈췄음을 알려야 함)
            self.vue_bridge.queueCompleted.emit(json.dumps({'total': total_count, 'natural': natural}))
            if natural:
                self.vue_bridge.showNotification.emit('success', f'{total_count}장 생성 완료')
        # 설정 '생성 후 모델 언로드' — 대기열/XYZ 마지막 장 뒤 (큐 stop 이 delay 뒤에 오므로 여기서)
        if hasattr(self, '_maybe_unload_models_after_generation'):
            self._maybe_unload_models_after_generation()
        # 웹 모드엔 호스트 창이 없다 — 위 토스트가 브라우저에 가고, 이 모달은 호스트 데스크톱에
        # 아무도 닫지 않는 창으로 남을 뿐이다.
        if natural and not getattr(self, 'web_mode', False):
            QMessageBox.information(self, "Task Complete", f"Successfully generated {total_count} images.")

    def _setup_tray(self):
        self._tray_manager = TrayManager(self)
        self._tray_manager.show_window_requested.connect(self.showNormal)
        # _quit_app 경유 — QApplication.quit 직결이면 설정/UI 상태 저장이 통째로 생략됨
        self._tray_manager.quit_requested.connect(self._quit_app)
        self._tray_manager.show()

    def _run_startup_sequence(self, app):
        """스플래시 로딩 화면과 함께 시작 구동 (main()이 showMaximized 전에 호출).

        순서: 스플래시 → Vue 로드(완료 대기) → 검색 덱 복원 →
              (게이트) 창을 띄우고 그 위에 백엔드 선택 오버레이 → 고르면 연결
              (게이트 아님: managed 자동 시작 등) → 백엔드 연결·모델 로딩(완료 대기)
              → 스플래시 닫고 메인 창 노출.

        **왜 백엔드 선택이 뒤로 갔나**: 예전엔 이게 1번이었다 — 앱 창을 보기도 전에
        별도 QDialog 가 떠서 결정을 요구했고, 시작 화면만 앱의 디자인 규칙 밖에 살았다.
        지금은 창을 먼저 띄우고 Vue 오버레이로 묻는다. 연결·모델 로딩은 고른 *뒤에*
        돌므로 이 함수는 게이트 경로에서 백엔드를 기다리지 않는다(그 대기는 오버레이가
        자기 안에서 보여 준다).

        안전망: 어느 단계가 실패해도 앱은 뜬다 — Vue 로드가 타임아웃이면 오버레이를
        그릴 방법이 없으므로 그때만 옛 QDialog(`_show_startup_selector`)로 떨어진다.
        except에서 핵심 단계(Vue 로드/백엔드 적용/덱 복원)를 폴백 보장하고, 모든 대기는
        타임아웃이 있어 무한 멈춤이 없다.
        시작 다이얼로그 X(SystemExit)만 그대로 전파해 종료한다."""
        splash = None
        try:
            # 1. 백엔드 선택 방식 결정 — 게이트면 플래그만 세우고 아무것도 안 띄운다.
            self._startup_backend_check()
            gate = bool(getattr(self, '_backend_gate_pending', False))

            # 2. 스플래시 표시 (이후 로딩 단계 시각화)
            try:
                from ui.splash_loader import SplashLoader
                splash = SplashLoader()
                splash.show()
                app.processEvents()
            except Exception as e:
                print(f"[Startup] splash 생성 실패(무시): {e}")
                splash = None

            def _step(msg, pct):
                if splash:
                    try:
                        splash.step(msg, pct)
                        app.processEvents()
                    except Exception:
                        pass

            # 3. Vue UI 로드 + 완료 대기 (창은 아직 숨김 — push가 도달하도록 먼저 로드)
            _step("UI 로딩 중…", 20)
            self._load_vue_ui()
            vue_ready = self._await_signal(
                getattr(self, 'vue_viewer', None), 'loadFinished', 12000, app)

            # 4. 검색 덱 디스크 복원은 무거운 JSON 파싱이라 시작(부트스트랩)을 막지 않게
            #    window 표시 *후*로 지연. (자동화는 즉시 필요 없고, 곧 준비되면 됨)
            from PyQt6.QtCore import QTimer as _QTimer
            _QTimer.singleShot(600, self._restore_search_deck)

            if gate and not vue_ready:
                # Vue 가 안 떴다 = 오버레이를 그릴 수 없다. 이때만 옛 모달로 떨어진다.
                # 이 경로가 없으면 사용자는 백엔드를 고를 방법 자체를 잃는다.
                print("[Startup] Vue 로드 실패 — 백엔드 선택을 비상 다이얼로그로 폴백")
                self._backend_gate_pending = False
                self._show_startup_selector()   # X 버튼 → SystemExit(종료)는 그대로
                gate = False

            if gate:
                # 5-a. 게이트 경로 — 연결은 사용자가 고른 뒤. 여기서는 오버레이를
                #      열라고만 알리고 시퀀스를 끝낸다(창은 main()이 곧 띄운다).
                _step("백엔드 선택", 100)
                self._managed_runtime_startup_apply_done = True
                self._emit_backend_selection_required()
            else:
                # 5-b. 기존 경로 — managed 자동 시작 / 비상 다이얼로그로 이미 확정됨.
                _step("백엔드 연결 · 모델 로딩 중…", 60)
                self._apply_backend_startup_result()
                self._managed_runtime_startup_apply_done = True
                self._await_signal(getattr(self, 'info_worker', None), 'info_ready', 8000, app,
                                   also='error_occurred')
                _step("완료", 100)
        except SystemExit:
            if splash is not None:
                try: splash.close()
                except Exception: pass
            raise  # 시작 다이얼로그 X → 종료
        except Exception as e:
            print(f"[Startup] 시퀀스 오류 — 기본 시작으로 폴백: {e}")
            # 폴백: 핵심 단계 누락 시 보장 (앱이 정상 동작하도록)
            try:
                if getattr(self, '_pending_vue_url', None) is not None:
                    self._load_vue_ui()
            except Exception: pass
            try:
                self._apply_backend_startup_result()
                self._managed_runtime_startup_apply_done = True
            except Exception: pass
            try:
                self._restore_search_deck()
            except Exception: pass
        finally:
            if splash is not None:
                try: splash.close()
                except Exception: pass

    def _await_signal(self, obj, signal_name, timeout_ms, app, also=None) -> bool:
        """obj.<signal_name>(또는 also)이 올 때까지 로컬 이벤트루프로 대기.
        반드시 타임아웃이 있어 무한 대기하지 않는다. obj가 None이거나 이미 끝난
        QThread면 즉시 반환. (app.exec() 진입 전 부트스트랩에서 쓰는 중첩 이벤트루프)

        돌려주는 값: 시그널이 실제로 왔으면(또는 기다릴 필요가 없었으면) True,
        타임아웃/대기 불가면 False. 시작 게이트가 "Vue 가 정말 떴는가"로 오버레이와
        비상 다이얼로그를 가르기 때문에 둘을 구분해야 한다 — 예전엔 이 함수가
        타임아웃과 성공을 똑같이 조용히 지나갔다."""
        if obj is None:
            return False
        try:
            from PyQt6.QtCore import QEventLoop, QTimer
            sig = getattr(obj, signal_name, None)
            if sig is None:
                return False
            # 이미 끝난 워커면 대기 불필요
            if hasattr(obj, 'isRunning') and not obj.isRunning():
                return True
            arrived = {'value': False}

            def _on_signal(*_args):
                arrived['value'] = True
                loop.quit()

            loop = QEventLoop()
            try:
                sig.connect(_on_signal)
            except Exception:
                return False
            if also:
                alt = getattr(obj, also, None)
                if alt is not None:
                    try: alt.connect(_on_signal)
                    except Exception: pass
            QTimer.singleShot(max(500, int(timeout_ms)), loop.quit)  # 타임아웃 폴백
            loop.exec()
            return arrived['value']
        except Exception as e:
            print(f"[Startup] await {signal_name} 실패(무시): {e}")
            return False

    def _restore_search_deck(self):
        """검색 결과 덱 디스크 복원 — window 표시 후 1회 지연 실행.
        큰 JSON 파싱이라 __init__ 동기 경로에서 빼내 시작 프리징을 없앴다.
        자동화가 Search 탭 방문 없이도 즉시 사용 가능하게 filtered_results/덱을 미리 채운다.
        (Vue SearchView onMounted도 별도로 호출하지만, 다른 탭에서 시작해도 준비되게 함)

        런타임 상태만 채운다 — 예전처럼 loadLastSearchResults 슬롯을 불러 20MB JSON 을
        만들어 버리지 않는다. 그 사이 Search 탭이 먼저 복원했거나 새 검색이 끝났으면
        (스냅숏이 이미 있으면) 디스크를 다시 읽지 않는다."""
        try:
            from core.search_result_store import SearchResultStore
            from core.search_session import restore_runtime_from_disk, runtime_lineage
            lineage = runtime_lineage(self)
            if lineage and lineage.get('snapshot_id'):
                return
            store = SearchResultStore()
            restored = restore_runtime_from_disk(self, store)
            if store.last_error:
                print(f"[Search] cache ignored: {store.last_error}")
            elif store.last_snapshot_id is not None:
                print(f"[Search] restored {len(restored):,} rows from disk → filtered_results")
        except Exception as e:
            print(f"[Search] 시작 시 덱 복원 실패: {e}")

    def _update_vram_status(self):
        """하단 VRAM 바 — 장치 전체 사용량(모든 프로세스).

        1순위 core.gpu_stats.read_vram(NVML → nvidia-smi): 백엔드 연결과 무관하게 잰다.
        2순위(GPU 툴이 전혀 없을 때) 연결된 백엔드의 /memory — 그 값은 백엔드 자신의
        메모리라 Ollama 등은 안 잡히지만, 없는 것보다 낫다.
        워커 스레드에서 수행(백엔드 HTTP 는 timeout=3 이라 메인 스레드면 UI 가 언다).
        vramUpdated 는 Qt 시그널 → 워커에서 emit 해도 queued connection 으로 안전.
        """
        import threading

        # Called only by the GUI timer: reserve before starting the worker so a
        # slow fallback query cannot enqueue another poll every second.
        if getattr(self, '_vram_poll_inflight', False):
            return
        self._vram_poll_inflight = True

        def _work():
            try:
                from core.gpu_stats import read_backend_vram, read_vram
                stats = read_vram()
                if not stats and getattr(self, '_backend_connected', False):
                    from backends import get_backend
                    backend = get_backend()
                    # 백엔드 HTTP 는 5초에 한 번만 — 그 사이엔 마지막 값을 다시 보낸다.
                    stats = read_backend_vram(backend.get_system_stats) if backend else None
                if stats and stats.get('vram_total', 0) > 0:
                    used = stats['vram_used'] / (1024**3)
                    total = stats['vram_total'] / (1024**3)
                    pct = int(used / total * 100) if total > 0 else 0
                    if hasattr(self, 'vue_bridge'):
                        self.vue_bridge.vramUpdated.emit(json.dumps({
                            'used': round(used, 1), 'total': round(total, 1), 'pct': pct,
                            'source': stats.get('source', ''),
                        }))
            except Exception:
                pass  # VRAM 모니터링은 비핵심 — GPU 미탑재/백엔드 미응답 시 무시
            finally:
                self._vram_poll_inflight = False

        try:
            threading.Thread(target=_work, daemon=True).start()
        except Exception:
            self._vram_poll_inflight = False

    def closeEvent(self, event):
        from PyQt6.QtWidgets import QMessageBox as _QMB
        msg = _QMB(self)
        msg.setWindowTitle("AI Studio Pro")
        msg.setText("앱을 종료하시겠습니까?")
        msg.setInformativeText("설정이 자동 저장됩니다.")
        msg.setStandardButtons(_QMB.StandardButton.Yes | _QMB.StandardButton.No)
        msg.setDefaultButton(_QMB.StandardButton.No)
        msg.setStyleSheet("""
            QMessageBox { background: #0D0D0D; color: #E8E8E8; }
            QLabel { color: #E8E8E8; font-size: 13px; }
            QPushButton { background: #1E1E1E; color: #E8E8E8; border: 1px solid #333;
                          border-radius: 6px; padding: 6px 20px; font-weight: 600; }
            QPushButton:hover { background: #333; }
            QPushButton:default { background: #FACC15; color: #000; border: none; }
        """)
        if msg.exec() == _QMB.StandardButton.Yes:
            self._quit_app()
        else:
            event.ignore()

    # ── 이벤트 생성 / 검색 ──

    def _handle_event_generation_request(self, payload, *, start_immediately):
        """Validate the Vue-owned selection and enqueue that exact snapshot."""
        from core.event_generation import (
            EventGenerationPlanError,
            plan_event_generation,
        )

        try:
            plan = plan_event_generation(payload)
        except EventGenerationPlanError as exc:
            if hasattr(self, 'vue_bridge'):
                self.vue_bridge.showNotification.emit('warning', str(exc))
            self.show_status(str(exc))
            return 0

        self.receive_event_scenarios(list(plan.scenarios))
        if start_immediately and hasattr(self, 'queue_manager'):
            self.queue_manager.start()
        return plan.count

    # _start_event_search / _auto_load_event_data / _on_event_data_loaded /
    # _run_event_search_worker 는 ui/event_search_actions.EventSearchActionsMixin 에 있다
    # (적재·검색 요청 순서, 옛 결과 버리기, eventLoadStatus 채널).

    def _export_event_results(self, payload):
        """이벤트 검색 결과를 .parquet로 내보내기"""
        events = payload.get('events', [])
        if not events:
            self.vue_bridge.showNotification.emit('error', '내보낼 이벤트가 없습니다')
            return
        path, _ = QFileDialog.getSaveFileName(self, "이벤트 내보내기", "event_results.parquet", "Parquet (*.parquet);;JSON (*.json)")
        if not path:
            return
        try:
            import pandas as pd
            if path.endswith('.parquet'):
                df = pd.DataFrame(events)
                # steps 컬럼은 JSON 문자열로 저장
                if 'steps' in df.columns:
                    df['steps'] = df['steps'].apply(lambda x: json.dumps(x, ensure_ascii=False) if isinstance(x, list) else x)
                df.to_parquet(path, index=False)
            else:
                with open(path, 'w', encoding='utf-8') as f:
                    json.dump(events, f, ensure_ascii=False, indent=2)
            self.vue_bridge.showNotification.emit('success', f'이벤트 {len(events)}건 내보내기 완료')
        except Exception as e:
            self.vue_bridge.showNotification.emit('error', f'내보내기 실패: {e}')

    def _import_event_results(self):
        """이벤트 결과 .parquet/.json 불러오기"""
        path, _ = QFileDialog.getOpenFileName(self, "이벤트 불러오기", "", "Parquet/JSON (*.parquet *.json)")
        if not path:
            return
        try:
            import pandas as pd
            if path.endswith('.parquet'):
                df = pd.read_parquet(path)
                events = df.to_dict('records')
                # steps 컬럼 JSON 파싱
                for ev in events:
                    if 'steps' in ev and isinstance(ev['steps'], str):
                        try:
                            ev['steps'] = json.loads(ev['steps'])
                        except Exception:
                            pass
            else:
                with open(path, 'r', encoding='utf-8') as f:
                    events = json.load(f)
            self.vue_bridge.eventImportResults.emit(json.dumps(events, ensure_ascii=False))
        except Exception as e:
            self.vue_bridge.showNotification.emit('error', f'불러오기 실패: {e}')

    # ── ADetailer 단독 실행 ──

    def _run_adetailer_single(self, payload):
        """단일 이미지에 ADetailer 적용"""
        from workers.adetailer_worker import ADetailerSingleWorker
        path = payload.get('path', '')
        settings = payload.get('settings', {})
        if not path:
            self.vue_bridge.showNotification.emit('error', '이미지 경로가 없습니다')
            return
        self._ad_worker = ADetailerSingleWorker(path, settings, self)
        self._ad_worker.finished.connect(lambda r: self.vue_bridge.adetailerResult.emit(r))
        self._ad_worker.start()
        self.vue_bridge.showNotification.emit('info', 'ADetailer 처리 중...')

    def _run_adetailer_batch(self, payload):
        """배치 이미지에 ADetailer 적용"""
        from workers.adetailer_worker import ADetailerBatchWorker
        paths = payload.get('paths', [])
        settings = payload.get('settings', {})
        if not paths:
            self.vue_bridge.showNotification.emit('error', '이미지가 없습니다')
            return
        self._ad_batch_worker = ADetailerBatchWorker(paths, settings, self)
        self._ad_batch_worker.progress.connect(
            lambda cur, tot: self.vue_bridge.adetailerProgress.emit(cur, tot))
        self._ad_batch_worker.single_done.connect(
            lambda r: self.vue_bridge.adetailerResult.emit(r))
        self._ad_batch_worker.all_done.connect(
            lambda: self.vue_bridge.showNotification.emit('success', f'ADetailer 배치 완료 ({len(paths)}장)'))
        self._ad_batch_worker.start()
        self.vue_bridge.showNotification.emit('info', f'ADetailer 배치 시작 ({len(paths)}장)')

    def _run_sam3_single(self, payload):
        """단일 이미지에 SAM3 적용"""
        from workers.sam3_worker import Sam3SingleWorker
        path = payload.get('path', '')
        settings = payload.get('settings', {})
        if not path:
            self.vue_bridge.showNotification.emit('error', '이미지 경로가 없습니다')
            return
        self._sam3_worker = Sam3SingleWorker(path, settings, self)
        self._sam3_worker.finished.connect(lambda r: self.vue_bridge.sam3Result.emit(r))
        self._sam3_worker.start()
        self.vue_bridge.showNotification.emit('info', 'SAM3 처리 중...')

    def _run_sam3_batch(self, payload):
        """배치 이미지에 SAM3 적용"""
        from workers.sam3_worker import Sam3BatchWorker
        paths = payload.get('paths', [])
        settings = payload.get('settings', {})
        if not paths:
            self.vue_bridge.showNotification.emit('error', '이미지가 없습니다')
            return
        self._sam3_batch_worker = Sam3BatchWorker(paths, settings, self)
        self._sam3_batch_worker.progress.connect(
            lambda cur, tot: self.vue_bridge.sam3Progress.emit(cur, tot))
        self._sam3_batch_worker.single_done.connect(
            lambda r: self.vue_bridge.sam3Result.emit(r))
        self._sam3_batch_worker.all_done.connect(
            lambda: self.vue_bridge.showNotification.emit('success', f'SAM3 배치 완료 ({len(paths)}장)'))
        self._sam3_batch_worker.start()
        self.vue_bridge.showNotification.emit('info', f'SAM3 배치 시작 ({len(paths)}장)')

    def _run_refine(self, payload):
        """SAM3 Refine — 기존 이미지를 Target/Replacement로 재손질 (sam-extra 워크플로 2).

        메인 프롬프트를 안 보냈으면 현재 t2i 프롬프트를 상속시킨다
        (확장 Refine 패널의 'Inherit main t2i prompt' 기본 ON과 동일).
        """
        from workers.refine_worker import RefineWorker
        path = payload.get('path', '')
        settings = dict(payload.get('settings', {}))
        if not path:
            self.vue_bridge.showNotification.emit('error', 'Refine: 이미지 경로가 없습니다')
            return

        if not settings.get('main_prompt') and hasattr(self, 'total_prompt_display'):
            try:
                settings['main_prompt'] = self.total_prompt_display.toPlainText()
            except Exception:
                pass
        if not settings.get('main_negative') and hasattr(self, 'neg_prompt_text'):
            try:
                settings['main_negative'] = self.neg_prompt_text.toPlainText()
            except Exception:
                pass

        self._refine_worker = RefineWorker(path, settings, self)
        self._refine_worker.finished.connect(lambda r: self.vue_bridge.refineResult.emit(r))
        self._refine_worker.start()
        self.vue_bridge.showNotification.emit('info', 'Refine 처리 중...')

    # ── DEFAULTS 탭 (Vue) → 위젯 적용 ──────────────────────
    def _apply_tab_defaults_to_empty_widgets(self, defaults: dict, first_run: bool = False) -> None:
        """tab_defaults.json의 값을 적절한 위젯에 적용.

        :param first_run: True면 모든 필드 적용 (덮어쓰기 포함).
                          False면 위젯이 비어있거나 "기본 placeholder"일 때만.
        """
        # (widget_id, defaults_key, type_converter) 매핑
        FIELD_MAP = [
            ("steps_input", "steps", str),
            ("cfg_input", "cfg", str),
            ("width_input", "width", str),
            ("height_input", "height", str),
            ("seed_input", "seed", str),
            ("sampler_combo", "sampler", str),
            ("scheduler_combo", "scheduler", str),
        ]
        for widget_id, key, conv in FIELD_MAP:
            if key not in defaults:
                continue
            widget = getattr(self, widget_id, None)
            if widget is None:
                continue
            new_val = conv(defaults[key])
            if not new_val or new_val == "":
                continue
            try:
                if first_run:
                    if hasattr(widget, "setText"):
                        widget.setText(new_val)
                    elif hasattr(widget, "setCurrentText"):
                        widget.setCurrentText(new_val)
                else:
                    # 위젯이 비었거나 placeholder 같으면 채움
                    current = ""
                    if hasattr(widget, "currentText"):
                        current = widget.currentText() or ""
                    elif hasattr(widget, "text"):
                        current = widget.text() or ""
                    if not current.strip():
                        if hasattr(widget, "setText"):
                            widget.setText(new_val)
                        elif hasattr(widget, "setCurrentText"):
                            widget.setCurrentText(new_val)
            except Exception as e:
                print(f"[Warning] DEFAULTS {key} 적용 실패: {e}")

    # ── LOGIC 탭 (Vue) → prompt_cleaner 연결 ──────────────
    def _apply_ui_prefs_to_cleaner(self, prefs: dict) -> None:
        """Vue Settings → LOGIC 토글 값을 prompt_cleaner에 반영.

        Vue 키 → cleaner 속성:
        - cleanDuplicates → remove_duplicates
        - cleanSpaces     → auto_space
        - cleanUnderscore → underscore_to_space

        호출 시점:
        - 앱 시작 시 (저장된 ui_prefs.json 적용)
        - save_ui_prefs 액션 받을 때마다 (변경 즉시 반영)

        세 옵션의 주인은 ui_prefs 하나다 — 키가 없으면 Vue 화면의 기본값(켜짐)을 쓴다. 레거시
        prompt_settings.cleaning_options 는 이 세 키를 더 이상 적용하지 않는다(core/prompt_cleaner_prefs).
        """
        cleaner = getattr(self, "prompt_cleaner", None)
        if cleaner is None:
            return
        try:
            from core.prompt_cleaner_prefs import cleaner_options_from_ui_prefs
            for attr, value in cleaner_options_from_ui_prefs(prefs).items():
                setattr(cleaner, attr, value)
        except Exception as e:
            print(f"[Warning] LOGIC 토글 적용 실패: {e}")

    # ── 테마 (Vue) → PyQt 위젯 ────────────────────────────
    def _apply_theme_prefs(self, prefs: dict) -> None:
        """Update native shell/Web/Backend chrome immediately without reloading pages."""
        if not isinstance(prefs, dict):
            return
        try:
            from utils.theme_manager import get_theme_manager
            from ui.native_dialogs import apply_native_shell_theme

            tm = get_theme_manager()
            tm.apply_prefs(prefs)
            apply_native_shell_theme(self, tm)
        except Exception as e:
            print(f"[Warning] 테마 적용 실패: {e}")

    # ── 워크플로우 프로파일 ────────────────────────────────
    def _send_workflow_profiles_list(self):
        """현재 프로파일 목록을 Vue로 전송."""
        import json as _json
        try:
            from core.workflow_profiles import list_profiles
            self.vue_bridge.workflowProfilesList.emit(
                _json.dumps(list_profiles())
            )
        except Exception:
            pass

    # ── 프롬프트 섹션 순서 (사용자 지정) ──────────────────
    def _send_prompt_order(self):
        """현재 prompt_order.json 내용 + 라벨을 Vue로 전송."""
        import json as _json
        try:
            from core.prompt_order import order_with_labels
            self.vue_bridge.promptOrderLoaded.emit(
                _json.dumps(order_with_labels())
            )
        except Exception:
            pass

    # ── PR 8: Instant Wildcards ────────────────────────────
    def _get_instant_wildcards(self):
        """프로세스 싱글톤 InstantWildcards — 생성 훅(부팅 시 register_standard_hooks 가 등록)과
        같은 객체라, 여기서 저장한 목록이 다음 생성에 바로 쓰인다. 훅 등록은 이름 기준 멱등이라
        부팅 등록이 실패했던 경우에만 여기서 채워진다(이중 등록 없음)."""
        from core.instant_wildcards import ensure_hook_registered, get_instant_wildcards
        iw = get_instant_wildcards()
        try:
            ensure_hook_registered(instance=iw)
        except Exception as exc:
            print(f"[Warning] 인스턴트 와일드카드 훅 등록 실패: {exc}")
        return iw

    def _send_instant_wildcards_list(self):
        """현재 인스턴트 와일드카드 목록을 Vue로 푸시."""
        import json as _json
        iw = self._get_instant_wildcards()
        items = [{"name": n, "lines": iw.get(n)} for n in iw.names()]
        try:
            self.vue_bridge.instantWildcardsList.emit(_json.dumps(items))
        except Exception:
            pass

    def _register_ui_state_handlers(self) -> None:
        """UIStateManager에 메인 윈도우 기하/상태 등록.

        다른 모듈(스플리터 등)이 자기 상태를 더 등록할 수 있음 — register는 멱등.
        """
        def _get_main_geometry() -> dict:
            return {
                "x": self.x(),
                "y": self.y(),
                "width": self.width(),
                "height": self.height(),
                "maximized": self.isMaximized(),
                "full_screen": self.isFullScreen(),
            }

        def _set_main_geometry(state: dict) -> None:
            """저장된 윈도우 기하 복원 — 타이틀바가 가려지지 않도록 보정."""
            # 웹 모드: 호스트 PyQt 창을 띄우지 않는다(브라우저로만 사용).
            # 저장된 상태가 maximized면 showMaximized가 창을 노출하던 문제 차단.
            if getattr(self, 'web_mode', False):
                return
            if not isinstance(state, dict):
                return
            try:
                from PyQt6.QtGui import QGuiApplication

                # 1) 최대화/풀스크린 상태면 setGeometry 건너뛰기
                #    main.py가 이미 showMaximized 호출했으므로 중복 호출하면 깜빡임
                if state.get("maximized"):
                    if not self.isMaximized():
                        self.showMaximized()
                    return
                if state.get("full_screen"):
                    if not self.isFullScreen():
                        self.showFullScreen()
                    return

                # 2) 일반 윈도우 — 현재 최대화돼있으면 먼저 normal로 풀기
                if self.isMaximized() or self.isFullScreen():
                    self.showNormal()

                # 3) 기하 + 화면 영역 보정 (타이틀바 가림 방지)
                w = max(int(state.get("width", 1280)), 600)
                h = max(int(state.get("height", 720)), 400)
                x = int(state.get("x", 100))
                y = int(state.get("y", 100))

                # 어느 모니터인지 — 중심점 기준
                target_screen = QGuiApplication.primaryScreen()
                for sc in QGuiApplication.screens():
                    if sc.geometry().contains(x + w // 2, y + h // 2):
                        target_screen = sc
                        break

                avail = target_screen.availableGeometry()
                # x: 좌측 이상, 우측에서 폭 빼고 이하
                # y: 상단 이상 — ★ 타이틀바 보임 보장
                x = max(avail.left(), min(x, avail.right() - w))
                y = max(avail.top(), min(y, avail.bottom() - h))

                self.setGeometry(x, y, w, h)
            except Exception as e:
                print(f"[Warning] UI 상태 복원 실패: {e}")

        self.ui_state.register("main_window", _get_main_geometry, _set_main_geometry)

    def _save_shutdown_state(self) -> bool:
        """Persist live settings unless a just-imported backup must win."""
        def mark_session_backup_clean():
            try:
                from core.session_backup import mark_session_clean
                mark_session_clean()
            except Exception as e:
                print(f"[Warning] 세션 백업 정상 종료 표시 실패: {e}")

        if getattr(self, '_preserve_imported_settings_on_quit', False):
            print("[Config] 가져온 설정 보존을 위해 종료 시 자동 저장을 건너뜁니다")
            # 설정 백업 가져오기는 의도한 교체라 크래시가 아니다 — 세션 백업(가져오기 전 프롬프트)을
            # 정상 종료로 표시한다. 두면 다음 부팅이 가져온 prompt_settings 와 다른 그 프롬프트를
            # '크래시 복구'로 제안하고, 누르면 방금 가져온 설정을 가져오기 전 프롬프트로 덮는다.
            mark_session_backup_clean()
            return False
        else:
            try:
                saved = self.save_settings()
            except Exception as e:
                saved = False
                print(f"[Warning] 종료 시 설정 저장 실패: {e}")
            if saved is not False:
                # 프롬프트가 prompt_settings.json 에 남았으니 크래시 복구 백업은 '정상 종료'로 표시 —
                # 다음 부팅이 복구를 제안하지 않는다. 저장이 실패했으면 백업이 유일한 사본이라 둔다.
                mark_session_backup_clean()
            try:
                # UI 상태 (창 크기/위치/스플리터 등) 저장
                if hasattr(self, "ui_state"):
                    self.ui_state.save_all()
            except Exception as e:
                print(f"[Warning] UI 상태 저장 실패: {e}")
        return True

    def _quit_app(self):
        # 덱 진행도는 뽑을 때마다 쓰지 않고 모아 둔다 — 종료 전에 남은 것을 저장한다.
        try:
            flush_deck = getattr(self, '_flush_deck_state', None)
            if callable(flush_deck):
                flush_deck()
        except Exception as exc:
            print(f"[Shutdown] 덱 진행도 저장 실패(계속 종료): {exc}")
        # 대기열 저장도 모아서 쓴다(widgets/queue_panel.py PERSIST_DELAY_MS) — os._exit 전에 남은 것을 쓴다.
        try:
            self._flush_queue_state()
        except Exception as exc:
            print(f"[Shutdown] 대기열 저장 실패(계속 종료): {exc}")
        self._save_shutdown_state()
        for stop in (self._shutdown_model_downloads, self._chat_stop,
                     self._shutdown_comfy_compatibility, self._shutdown_comfy_workflow_controls,
                     self._shutdown_relight, self._shutdown_hand_reconstruction):
            try:
                stop()
            except Exception as exc:
                print(f"[Shutdown] 생성/다운로드 정리 실패(계속 종료): {exc}")
        try:
            # 진행 중 생성 워커에 취소(+백엔드 interrupt) — GPU 작업 중단
            worker = getattr(self, 'gen_worker', None)
            if worker is not None and hasattr(worker, 'cancel') and worker.isRunning():
                worker.cancel()
        except Exception:
            pass
        # QThread 워커들 정지·짧게 대기 — Python 데몬 스레드가 아니므로 그냥 두고 os._exit하면
        # 실행 중 파괴로 Qt 경고/드문 크래시 가능. 각 ~0.5초만 기다리고 안 끝나면 포기(워치독).
        # (Ollama 워커는 브리지가 보관하고 취소할 수 없는 HTTP 라 기다려도 이득이 없다 — 대상 아님)
        for _name in ('gen_worker', '_search_worker', 'info_worker'):
            try:
                w = getattr(self, _name, None)
                if w is None or not hasattr(w, 'isRunning') or not w.isRunning():
                    continue
                for _stopper in ('cancel', 'stop'):
                    fn = getattr(w, _stopper, None)
                    if callable(fn):
                        try: fn()
                        except Exception: pass
                try: w.quit()
                except Exception: pass
                try: w.wait(500)
                except Exception: pass
            except Exception:
                pass
        # 검색 stale 워커들(브리지가 보관)도 동일 처리
        try:
            for w in list(getattr(self.vue_bridge, '_stale_search_workers', []) or []):
                try:
                    if w.isRunning():
                        w.stop(); w.wait(300)
                except Exception:
                    pass
        except Exception:
            pass
        self._stop_owned_backend_runtimes()
        try:
            from core.app_instance import unregister_app_instance

            unregister_app_instance()
        except Exception:
            pass
        # 최종 종료는 os._exit 유지 — QApplication.quit()은 QWebEngineProfile/Page 해체 순서
        # 크래시·행이 재발함(커밋 24d7856d6, e6f964c6f 이력). 위에서 설정 저장 + QThread
        # 정지·대기를 마쳤고, 에디터/캡션/영속 쓰기는 Python 데몬 스레드라 안전.
        os._exit(0)

    def _stop_owned_backend_runtimes(self) -> None:
        """앱이 시작한 process만 종료한다. 외부 URL/process는 절대 건드리지 않는다."""
        try:
            from core.backend_runtime import get_backend_runtime_manager

            manager = get_backend_runtime_manager()
            # manager는 이 프로세스가 만든 live Popen handle만 보관하므로 외부
            # Forge/ComfyUI를 건드릴 수 없다.
            manager.stop_all_owned()
        except Exception as exc:
            print(f"[Runtime] owned process 종료 실패(계속 종료): {exc}")
