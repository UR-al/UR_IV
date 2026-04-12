# ui/generator_main.py
"""
GeneratorMainUI - 메인 윈도우 클래스 (데이터 연동 및 안정화 최종판)
"""
import sys
import traceback
import json
import os
import shutil
import subprocess
from urllib.parse import unquote

from PyQt6.QtWidgets import QMessageBox, QLineEdit, QTextEdit, QApplication, QHBoxLayout, QWidget, QFileDialog, QMenu
from PyQt6.QtCore import QTimer, QEvent, Qt, pyqtSlot


def _clean_path(path: str) -> str:
    """file:/// 프리픽스 제거 + OS 경로 정규화"""
    if not path:
        return ''
    clean_path = unquote(str(path)).replace('file:///', '').replace('file://', '')
    clean_path = clean_path.replace('/', os.sep)
    return clean_path

from ui.generator_base import GeneratorBase
from ui.generator_ui_setup import UISetupMixin
from ui.generator_prompts import PromptHandlingMixin
from ui.generator_generation import GenerationMixin
from ui.generator_settings import SettingsMixin
from ui.generator_actions import ActionsMixin
from ui.generator_gallery import GalleryMixin
from ui.generator_webui import WebUIMixin
from ui.generator_search import SearchMixin
from widgets.queue_panel import QueuePanel
from widgets.queue_manager import QueueManager
from utils.prompt_cleaner import get_prompt_cleaner
from utils.theme_manager import get_theme_manager, get_color
from utils.tray_manager import TrayManager


class GeneratorMainUI(
    GeneratorBase,
    UISetupMixin,
    PromptHandlingMixin,
    GenerationMixin,
    SettingsMixin,
    ActionsMixin,
    GalleryMixin,
    WebUIMixin,
    SearchMixin
):
    _IMAGE_EXTS = ('.png', '.jpg', '.jpeg', '.webp', '.bmp')

    def __init__(self):
        try:
            super().__init__()
            print("[System] Initializing AI Studio Pro Engine...")
            self.setWindowTitle("AI Studio Pro")
            self.setAcceptDrops(True)

            # 1. 필수 속성 초기화
            self.s1_widgets = {'prompt': type('P',(),{'installEventFilter':lambda *a:None})()}
            self.s2_widgets = {'prompt': type('P',(),{'installEventFilter':lambda *a:None})()}
            self.is_programmatic_change = False
            self.is_automation_running = False
            self.generation_data = {}
            self.filtered_results = []

            # 2. 아이콘 설정
            from PyQt6.QtGui import QIcon
            icon_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'assets', 'icons', 'app_icon.svg')
            if os.path.exists(icon_path):
                self.setWindowIcon(QIcon(icon_path))

            self.prompt_cleaner = get_prompt_cleaner()

            # 3. UI 및 프록시 레이어 구축
            self._setup_ui()

            # 에러 핸들러에 bridge 등록
            from core.error_handler import set_bridge
            set_bridge(self.vue_bridge)
            
            # 4. 시그널 및 설정 동기화
            self.connect_signals()
            self.load_settings()

            # 5. 백엔드 브릿지 가동
            self._startup_backend_check()
            self._apply_backend_startup_result()
            
            # 6. 대기열 및 시스템 트레이
            self._setup_queue()
            self._setup_tray()

            # 7. 타이머 가동
            self._vram_timer = QTimer()
            self._vram_timer.setInterval(30000)
            self._vram_timer.timeout.connect(self._update_vram_status)
            self._vram_timer.start()

            self._clean_timer = QTimer()
            self._clean_timer.setSingleShot(True)
            self._clean_timer.setInterval(500)
            self._clean_timer.timeout.connect(self._deferred_clean_all)
            self._setup_realtime_cleaning()

            # 8. 초기값 강제 업데이트
            QTimer.singleShot(500, self.update_total_prompt_display)

            # 9. 조건식 + 기본값 로드
            QTimer.singleShot(1000, self._load_saved_configs)
            print("[System] Engine Ready.")

        except Exception as e:
            print(f"\n[Fatal] Error during boot: {e}")
            traceback.print_exc()
            QMessageBox.critical(None, "Boot Error", f"Fatal initialization error:\n{e}")
            sys.exit(1)

    # ========== XYZ Plot 핸들러 ==========
    
    def _on_xyz_add_to_queue(self, payloads: list):
        if hasattr(self, 'queue_panel'):
            for p in payloads: self.queue_panel.add_single_item(p)
            self.show_status(f"Added {len(payloads)} XYZ combinations to queue.")

    def _on_xyz_start_generation(self, payloads: list):
        if hasattr(self, 'queue_panel'):
            for p in payloads: self.queue_panel.add_single_item(p)
            self.show_status("Starting XYZ generation.")
            if hasattr(self, 'queue_manager'): self.queue_manager.start()
            
    # ========== Vue Bridge Action Handler (The Core) ==========

    def _handle_vue_action(self, action: str, payload: dict):
        """[중요] Vue에서 날아온 모든 액션을 분석하고 백엔드 로직에 주입"""
        print(f"[Bridge] Action Received: {action} | Payload: {json.dumps(payload)[:100]}...")
        
        try:
            # 1. 워크스페이스 제어
            if action in ('switch_tab', 'native_tab_switch'):
                tab_id = payload.get('tab', 't2i')
                tab_map = {'web': 1, 'backend': 2}
                idx = tab_map.get(tab_id, 0)
                if hasattr(self, '_main_stack'): self._main_stack.setCurrentIndex(idx)
                self.show_status(f"Workspace Switched: {tab_id}")

            # 2. 이미지 생성 엔진
            elif action == 'generate':
                # Vue 데이터가 Store를 통해 Proxy에 이미 동기화되어 있어야 함
                self.on_generate_clicked()

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
            
            elif action == 'editor_save':
                path = payload.get('path', '')
                if path:
                    src = _clean_path(path)
                    dst, _ = QFileDialog.getSaveFileName(self, "Export Edited Image", "", "PNG (*.png);;JPEG (*.jpg)")
                    if dst:
                        shutil.copy2(src, dst)
                        self.show_status(f"Exported to: {os.path.basename(dst)}")

            elif action == 'editor_change_tool':
                tool = payload.get('tool', 'box')
                if hasattr(self, 'mosaic_editor') and self.mosaic_editor.mosaic_panel:
                    panel = self.mosaic_editor.mosaic_panel
                    tool_map = {'box': 0, 'lasso': 1, 'brush': 2, 'eraser': 3}
                    btn_id = tool_map.get(tool, 0)
                    btn = panel.tool_group.button(btn_id)
                    if btn: btn.setChecked(True); panel.on_tool_group_clicked(btn)

            elif action == 'editor_apply_effect':
                if hasattr(self, 'mosaic_editor'): self.mosaic_editor._on_apply_effect()
            elif action == 'editor_apply_auto_censor':
                if hasattr(self, 'mosaic_editor'): self.mosaic_editor._on_auto_censor()
            elif action == 'editor_add_yolo_model':
                from PyQt6.QtWidgets import QFileDialog as _QFD
                from tabs.editor.mosaic_panel import get_editor_models_dir
                models_dir = get_editor_models_dir()
                paths, _ = _QFD.getOpenFileNames(
                    self, "YOLO 모델 선택", models_dir,
                    "YOLO Model (*.pt *.onnx *.safetensors);;All Files (*)"
                )
                if paths and hasattr(self, 'mosaic_editor') and self.mosaic_editor.mosaic_panel:
                    panel = self.mosaic_editor.mosaic_panel
                    from tabs.editor.mosaic_panel import _save_yolo_model_paths, get_editor_models_dir
                    models_dir = get_editor_models_dir()
                    for p in paths:
                        # editor_models/ 외부 파일이면 복사
                        if not p.startswith(models_dir.replace('\\', '/')) and not p.startswith(models_dir):
                            dst = os.path.join(models_dir, os.path.basename(p))
                            if not os.path.exists(dst):
                                shutil.copy2(p, dst)
                                p = dst
                        if p not in panel._yolo_model_paths:
                            panel._yolo_model_paths.append(p)
                    _save_yolo_model_paths(panel._yolo_model_paths)
                    panel._update_model_label()
                    # Vue에 모델 라벨 업데이트 전달
                    import os as _os2
                    names = [_os2.path.basename(p) for p in panel._yolo_model_paths]
                    label = ", ".join(names) if names else "No Model"
                    self.vue_bridge.yoloModelUpdated.emit(label)
                    self.show_status(f"YOLO Model loaded: {label}")

            # 5. 하이엔드 우클릭 메뉴
            elif action == 'context_menu':
                path = payload.get('path', '')
                if not path: return
                clean_path = _clean_path(path)
                if not os.path.exists(clean_path): return
                
                menu = QMenu(self)
                menu.setStyleSheet(f"QMenu {{ background: #121212; color: white; border: 1px solid #333; padding: 4px; }} QMenu::item:selected {{ background: {get_color('accent')}; color: black; }}")
                
                act_i2i = menu.addAction("🖼️ SEND TO I2I")
                act_inpaint = menu.addAction("🎨 SEND TO INPAINT")
                act_editor = menu.addAction("🎨 SEND TO EDITOR")
                menu.addSeparator()
                act_folder = menu.addAction("📁 SHOW IN EXPLORER")
                act_copy = menu.addAction("📋 COPY TO CLIPBOARD")
                act_del = menu.addAction("🗑️ DELETE TO TRASH")
                
                from PyQt6.QtGui import QCursor
                chosen = menu.exec(QCursor.pos())
                if chosen == act_i2i: self._handle_vue_action('send_to_i2i', {'path': clean_path})
                elif chosen == act_inpaint: self._handle_vue_action('send_to_inpaint', {'path': clean_path})
                elif chosen == act_editor: self._handle_vue_action('send_to_editor', {'path': clean_path})
                elif chosen == act_folder:
                    subprocess.run(['explorer', '/select,', clean_path])
                elif chosen == act_copy:
                    from PyQt6.QtGui import QPixmap
                    pix = QPixmap(clean_path)
                    if not pix.isNull(): QApplication.clipboard().setPixmap(pix); self.show_status("Copied.")
                elif chosen == act_del:
                    from core.image_utils import move_to_trash
                    move_to_trash(clean_path); self.show_status("Moved to Trash.")

            # 6. 기타 스튜디오 도구
            elif action == 'show_prompt_history': self._show_prompt_history()
            elif action == 'open_lora_manager': self._open_lora_manager()
            elif action == 'save_settings':
                self.save_settings()
                # 저장 후 즉시 재로드 (Vue에 반영)
                QTimer.singleShot(200, self.load_settings)
                # defaults도 동시 업데이트
                try:
                    defaults_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'config', 'tab_defaults.json')
                    if os.path.exists(defaults_path):
                        with open(defaults_path, 'r', encoding='utf-8') as f:
                            cur = json.load(f)
                        cur['steps'] = int(self.steps_input.text() or 20)
                        cur['cfg'] = float(self.cfg_input.text() or 7)
                        cur['width'] = int(self.width_input.text() or 1024)
                        cur['height'] = int(self.height_input.text() or 1024)
                        cur['seed'] = self.seed_input.text() or '-1'
                        with open(defaults_path, 'w', encoding='utf-8') as f:
                            json.dump(cur, f, ensure_ascii=False, indent=2)
                except Exception:
                    pass
                if hasattr(self, 'vue_bridge'):
                    self.vue_bridge.showNotification.emit('success', '설정이 저장되었습니다')
            elif action == 'swap_resolution': self._swap_resolution()
            elif action == 'set_random_resolutions':
                lst = payload.get('list', [])
                self.random_resolutions = [(int(r[0]), int(r[1]), str(r[2])) for r in lst if len(r) >= 3]
            elif action == 'set_rating_filter':
                self._rating_filter = set(payload.get('ratings', ['g', 's', 'q', 'e']))
                # 현재 deck을 rating 필터로 재필터링
                if hasattr(self, 'filtered_results') and self.filtered_results:
                    self.shuffled_prompt_deck = [
                        r for r in self.filtered_results
                        if r.get('rating', 'g') in self._rating_filter
                    ]
                    import random as _rnd
                    _rnd.shuffle(self.shuffled_prompt_deck)
            elif action == 'update_prompt_deck':
                # Vue에서 필터링된 결과로 덱 업데이트
                deck = payload.get('results', [])
                if deck:
                    import random as _rnd
                    rating_filter = getattr(self, '_rating_filter', {'g', 's', 'q', 'e'})
                    self.filtered_results = deck
                    self.shuffled_prompt_deck = [
                        r for r in deck if r.get('rating', 'g') in rating_filter
                    ]
                    _rnd.shuffle(self.shuffled_prompt_deck)
            elif action == 'run_adetailer_single':
                self._run_adetailer_single(payload)
            elif action == 'run_adetailer_batch':
                self._run_adetailer_batch(payload)
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
            elif action == 'shuffle': self._shuffle_main_prompt()
            elif action == 'ab_test': self._open_ab_test()
            elif action == 'random_prompt': self.apply_random_prompt()

            # 7. 검색 결과 → 프롬프트 적용
            elif action == 'apply_search_result':
                # Vue 검색 결과 필드명 → Python bundle 필드명 통일
                bundle = {
                    'character': payload.get('character', ''),
                    'copyright': payload.get('copyright', ''),
                    'artist': payload.get('artist', ''),
                    'general': payload.get('general', ''),
                }
                # nan 문자열 처리
                for k in bundle:
                    if str(bundle[k]).lower() == 'nan':
                        bundle[k] = ''

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
                # 먼저 프롬프트 적용하여 UI 채우기
                bundle = {
                    'character': payload.get('character', ''),
                    'copyright': payload.get('copyright', ''),
                    'artist': payload.get('artist', ''),
                    'general': payload.get('general', ''),
                }
                for k in bundle:
                    if str(bundle[k]).lower() == 'nan': bundle[k] = ''
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
                    clean = _clean_path(path)
                    self._load_favorites_from_file()
                    if clean not in self.favorites_list:
                        self.favorites_list.append(clean)
                        self._save_favorites_to_file()
                    self.show_status("Added to favorites.")
                    if hasattr(self, 'vue_bridge'):
                        self.vue_bridge.showNotification.emit('success', '즐겨찾기에 추가됨')

            # artist lock
            elif action == 'set_artist_locked':
                locked = payload.get('locked', False)
                if hasattr(self, 'btn_lock_artist'):
                    self.btn_lock_artist.setChecked(locked)

            # 9. 이미지 삭제
            elif action == 'delete_image':
                path = payload.get('path', '')
                if path:
                    clean_path = _clean_path(path)
                    from core.image_utils import move_to_trash
                    move_to_trash(clean_path)
                    self.show_status("Moved to trash.")
                    if hasattr(self, 'vue_bridge'):
                        self.vue_bridge.showNotification.emit('info', '휴지통으로 이동됨')

            # 10. 프리셋
            elif action == 'save_preset_by_name':
                try:
                    name = payload.get('name', '')
                    if name:
                        preset = self._build_settings_dict()
                        preset_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'presets')
                        os.makedirs(preset_dir, exist_ok=True)
                        with open(os.path.join(preset_dir, f"{name}.json"), 'w', encoding='utf-8') as f:
                            json.dump(preset, f, ensure_ascii=False, indent=2)
                        self.vue_bridge.showNotification.emit('success', f'프리셋 "{name}" 저장됨')
                except Exception as e:
                    self.vue_bridge.showNotification.emit('error', f'저장 실패: {e}')

            elif action == 'load_preset_by_name':
                try:
                    name = payload.get('name', '')
                    if name:
                        fp = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'presets', f"{name}.json")
                        with open(fp, 'r', encoding='utf-8') as f:
                            preset = json.load(f)
                        self._apply_settings_dict(preset)
                        self.update_total_prompt_display()
                        self.vue_bridge.showNotification.emit('success', f'프리셋 "{name}" 로드됨')
                except Exception as e:
                    self.vue_bridge.showNotification.emit('error', f'로드 실패: {e}')

            elif action == 'delete_preset':
                try:
                    name = payload.get('name', '')
                    fp = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'presets', f"{name}.json")
                    if os.path.exists(fp): os.remove(fp)
                    self.vue_bridge.showNotification.emit('success', f'프리셋 "{name}" 삭제됨')
                except Exception as e:
                    self.vue_bridge.showNotification.emit('error', f'삭제 실패: {e}')

            elif action == 'save_preset':
                try:
                    name, ok = QMessageBox.question(self, "프리셋", ""), None
                    from PyQt6.QtWidgets import QInputDialog
                    name, ok = QInputDialog.getText(self, "프리셋 저장", "프리셋 이름:")
                    if ok and name:
                        preset = self._build_settings_dict()
                        preset_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'presets')
                        os.makedirs(preset_dir, exist_ok=True)
                        path = os.path.join(preset_dir, f"{name}.json")
                        with open(path, 'w', encoding='utf-8') as f:
                            json.dump(preset, f, ensure_ascii=False, indent=2)
                        self.vue_bridge.showNotification.emit('success', f'프리셋 "{name}" 저장됨')
                except Exception as e:
                    self.vue_bridge.showNotification.emit('error', f'프리셋 저장 실패: {e}')

            elif action == 'load_preset':
                try:
                    from PyQt6.QtWidgets import QInputDialog
                    preset_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'presets')
                    os.makedirs(preset_dir, exist_ok=True)
                    files = [f.replace('.json', '') for f in os.listdir(preset_dir) if f.endswith('.json')]
                    if not files:
                        self.vue_bridge.showNotification.emit('info', '저장된 프리셋이 없습니다')
                        return
                    name, ok = QInputDialog.getItem(self, "프리셋 불러오기", "선택:", files, 0, False)
                    if ok and name:
                        path = os.path.join(preset_dir, f"{name}.json")
                        with open(path, 'r', encoding='utf-8') as f:
                            preset = json.load(f)
                        self.load_settings_from_dict(preset) if hasattr(self, 'load_settings_from_dict') else self._apply_settings_dict(preset)
                        self.vue_bridge.showNotification.emit('success', f'프리셋 "{name}" 로드됨')
                except Exception as e:
                    self.vue_bridge.showNotification.emit('error', f'프리셋 로드 실패: {e}')

            # 11. I2I/Inpaint 생성
            elif action == 'generate_i2i':
                if hasattr(self, 'i2i_tab'):
                    self.i2i_tab.start_generation()
            elif action == 'generate_inpaint':
                if hasattr(self, 'inpaint_tab'):
                    self.inpaint_tab.start_generation()

            # 12. 배치/업스케일
            elif action == 'start_batch':
                if hasattr(self, 'batch_tab'):
                    self.batch_tab.start_batch()
            elif action == 'start_upscale':
                if hasattr(self, 'upscale_tab'):
                    self.upscale_tab.start_upscale()

            # PNG Info 파일 열기
            elif action == 'open_png_info_file':
                path, _ = QFileDialog.getOpenFileName(self, "PNG Info 이미지 선택", "", "Images (*.png *.jpg *.jpeg *.webp)")
                if path and hasattr(self, 'vue_bridge'):
                    self.vue_bridge.inpaintImageLoaded.emit(path.replace('\\', '/'))

            # 13. 에디터 워터마크 이미지 로드
            elif action == 'editor_load_watermark_image':
                path, _ = QFileDialog.getOpenFileName(self, "워터마크 이미지 선택", "", "Images (*.png *.jpg *.jpeg *.webp)")
                if path:
                    self.show_status(f"Watermark image: {os.path.basename(path)}")

            # 14. 갤러리 폴더 변경
            elif action == 'gallery_change_folder':
                folder = payload.get('folder', '')
                if folder and hasattr(self, 'gallery_tab'):
                    self.gallery_tab.load_folder(folder)

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
                        df = pd.read_parquet(path)
                        out = []
                        for _, row in df.iterrows():
                            out.append({
                                'copyright': str(row.get('tag_string_copyright', row.get('copyright', ''))),
                                'character': str(row.get('tag_string_character', row.get('character', ''))),
                                'artist': str(row.get('tag_string_artist', row.get('artist', ''))),
                                'general': str(row.get('tag_string_general', row.get('general', ''))),
                                'rating': str(row.get('rating', '')),
                            })
                        self._last_search_results = out
                        # Python filtered_results + shuffled_prompt_deck 업데이트
                        import random as _rnd
                        self.filtered_results = out
                        self.shuffled_prompt_deck = out.copy()
                        _rnd.shuffle(self.shuffled_prompt_deck)
                        # Vue로 결과 전달
                        self.vue_bridge.searchResultsReady.emit(json.dumps(out))
                        self.show_status(f"Imported {len(out)} results")
                    except Exception as e:
                        self.show_status(f"Import failed: {e}")

            # 16. 자동화 설정/토글
            elif action == 'set_automation_settings':
                self._vue_automation_settings = {
                    'mode': str(payload.get('mode', 'count')),
                    'limit': payload.get('limit', 10),
                    'repeat': payload.get('repeat', 1),
                    'delay': payload.get('delay', 1.0),
                    'allowDupes': bool(payload.get('allowDupes', False)),
                }

            elif action == 'toggle_automation':
                checked = payload.get('checked', False)
                if hasattr(self, 'btn_auto_toggle'):
                    self.btn_auto_toggle.setChecked(bool(checked))
                else:
                    self.toggle_automation_ui(bool(checked))
            elif action == 'stop_automation':
                if self.is_automating:
                    self._stop_automation("사용자가 자동화를 중지했습니다.")

            # ═══════ 이벤트 생성 (EventGen) ═══════
            elif action == 'search_events':
                self._start_event_search(payload)
            elif action == 'export_event_results':
                self._export_event_results(payload)
            elif action == 'import_event_results':
                self._import_event_results()

            elif action == 'select_event':
                idx = payload.get('index', 0)
                if hasattr(self, 'event_gen_tab') and hasattr(self.event_gen_tab, 'result_list'):
                    if 0 <= idx < self.event_gen_tab.result_list.count():
                        self.event_gen_tab.result_list.setCurrentRow(idx)
                        self.event_gen_tab._on_result_clicked(idx)

            elif action == 'event_add_to_queue':
                if hasattr(self, 'event_gen_tab'):
                    scenarios = payload.get('scenarios', [])
                    if scenarios:
                        self.receive_event_scenarios(scenarios)
                    elif hasattr(self.event_gen_tab, '_build_scenarios'):
                        scenarios = self.event_gen_tab._build_scenarios()
                        self.receive_event_scenarios(scenarios)

            elif action == 'event_generate_now':
                if hasattr(self, 'event_gen_tab'):
                    if hasattr(self.event_gen_tab, '_build_scenarios'):
                        scenarios = self.event_gen_tab._build_scenarios()
                        self.receive_event_scenarios(scenarios)
                        if hasattr(self, 'queue_manager'):
                            self.queue_manager.start()

            # ═══════ PNG Info 전송/생성 ═══════
            elif action == 'pnginfo_send_prompt':
                prompt = payload.get('prompt', '')
                negative = payload.get('negative', '')
                self.handle_prompt_only_transfer(prompt, negative)

            elif action == 'pnginfo_generate':
                # EXIF 데이터에서 payload 구성하여 즉시 생성
                raw = payload.get('raw', payload.get('parameters', ''))
                if raw:
                    self._handle_immediate_generation_from_raw(raw)

            # ═══════ XYZ Plot 실행 ═══════
            elif action == 'start_xyz_plot':
                axes = payload.get('axes', [])
                combinations = payload.get('combinations', [])
                if combinations and hasattr(self, 'queue_panel'):
                    for combo in combinations:
                        p = self._build_xyz_payload(combo)
                        self.queue_panel.add_single_item(p)
                    self.show_status(f"XYZ Plot: {len(combinations)} jobs queued.")
                    if hasattr(self, 'queue_manager'):
                        self.queue_manager.start()

            # ═══════ 배치 파일 열기 ═══════
            elif action in ('open_batch_files', 'open_upscale_files'):
                title = "배치 처리할 이미지 선택" if 'batch' in action else "업스케일할 이미지 선택"
                paths, _ = QFileDialog.getOpenFileNames(self, title, "", "Images (*.png *.jpg *.jpeg *.webp);;All Files (*)")
                if paths:
                    file_list = [p.replace('\\', '/') for p in paths]
                    self.vue_bridge.batchFilesSelected.emit(json.dumps(file_list))
                    self.show_status(f"{len(file_list)} files selected.")

            # ═══════ 클립보드 복사 ═══════
            elif action == 'copy_to_clipboard':
                path = payload.get('path', '')
                if path:
                    clean = _clean_path(path)
                    from PyQt6.QtGui import QPixmap
                    pix = QPixmap(clean)
                    if not pix.isNull():
                        QApplication.clipboard().setPixmap(pix)
                        self.show_status("Copied to clipboard.")

            # ═══════ 캐릭터 프리셋 ═══════
            elif action == 'open_character_preset':
                try:
                    from widgets.character_preset_dialog import CharacterPresetDialog
                    dlg = CharacterPresetDialog(parent=self)
                    if dlg.exec():
                        preset = dlg.get_selected_preset()
                        if preset:
                            self.character_input.setText(preset.get('character', ''))
                            if preset.get('copyright'):
                                self.copyright_input.setText(preset['copyright'])
                            self.show_status("Character preset applied.")
                except Exception as e:
                    print(f"[Error] Character preset: {e}")

            # ═══════ 태그 가중치 편집기 ═══════
            elif action == 'open_tag_weight_editor':
                try:
                    from widgets.tag_weight_editor import TagWeightEditorDialog
                    text = self.main_prompt_text.toPlainText()
                    dlg = TagWeightEditorDialog(text, parent=self)
                    if dlg.exec():
                        self.main_prompt_text.setPlainText(dlg.get_result())
                        self.show_status("Tag weights updated.")
                except Exception as e:
                    print(f"[Error] Tag weight editor: {e}")

            # ═══════ YOLO 모델 초기화 ═══════
            elif action == 'editor_clear_yolo_models':
                if hasattr(self, 'mosaic_editor') and self.mosaic_editor.mosaic_panel:
                    panel = self.mosaic_editor.mosaic_panel
                    panel._yolo_model_paths.clear()
                    from tabs.editor.mosaic_panel import _save_yolo_model_paths
                    _save_yolo_model_paths([])
                    panel._update_model_label()
                    self.vue_bridge.yoloModelUpdated.emit("No Model Loaded")
                    self.show_status("YOLO models cleared.")

            # ═══════ Gallery EXIF → T2I ═══════
            elif action == 'gallery_send_exif_to_t2i':
                exif_raw = payload.get('exif', '')
                if exif_raw:
                    parts = exif_raw.split('\nNegative prompt: ')
                    prompt = parts[0].strip()
                    negative = ''
                    if len(parts) > 1:
                        sub = parts[1].split('\nSteps: ')
                        negative = sub[0].strip()
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
                    clean = _clean_path(path)
                    self._load_favorites_from_file()
                    if clean in self.favorites_list:
                        self.favorites_list.remove(clean)
                        self._save_favorites_to_file()
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

            # ═══════ 탭 순서 설정 ═══════
            elif action == 'set_tab_order':
                order = payload.get('order', [])
                if order:
                    self.show_status(f"Tab order updated: {len(order)} tabs")

            # ═══════ 시드 탐색 (3x3 그리드) ═══════
            elif action == 'explore_seed':
                base_seed = int(payload.get('seed', -1))
                if base_seed < 0:
                    import random
                    base_seed = random.randint(0, 2**32 - 1)
                # subseed_strength 0.05 단위로 9개 변형
                variations = []
                for i in range(9):
                    subseed = base_seed + i * 1000
                    strength = round(0.05 * (i % 5), 2) if i > 0 else 0
                    p = {
                        'prompt': self.total_prompt_display.toPlainText(),
                        'negative_prompt': self.neg_prompt_text.toPlainText(),
                        'sampler_name': self.sampler_combo.currentText(),
                        'steps': int(self.steps_input.text() or 20),
                        'cfg_scale': float(self.cfg_input.text() or 7),
                        'seed': base_seed,
                        'subseed': subseed,
                        'subseed_strength': strength,
                        'width': int(self.width_input.text() or 1024),
                        'height': int(self.height_input.text() or 1024),
                    }
                    variations.append(p)
                # 대기열에 추가
                if hasattr(self, 'queue_panel'):
                    for p in variations:
                        self.queue_panel.add_single_item(p)
                    if hasattr(self, 'queue_manager'):
                        self.queue_manager.start()
                self.vue_bridge.showNotification.emit('info', f'시드 탐색: {len(variations)}개 변형 생성 시작')

            # ═══════ 비교 이미지 ═══════
            elif action == 'open_compare_image':
                slot = payload.get('slot', 'before')
                path, _ = QFileDialog.getOpenFileName(self, "비교 이미지 선택", "", "Images (*.png *.jpg *.jpeg *.webp)")
                if path:
                    self.vue_bridge.compareImageLoaded.emit(json.dumps({'slot': slot, 'path': path.replace('\\', '/')}))

            elif action == 'open_url':
                url = payload.get('url', '')
                if url:
                    import webbrowser
                    webbrowser.open(url)

            elif action == 'send_to_compare':
                path = payload.get('path', '')
                slot = payload.get('slot', 'after')
                if path:
                    clean = _clean_path(path).replace('\\', '/')
                    if clean.startswith('/') and ':' in clean[1:3]: clean = clean[1:]
                    self.vue_bridge.tabChanged.emit('png')
                    QTimer.singleShot(100, lambda: self.vue_bridge.compareImageLoaded.emit(json.dumps({'slot': slot, 'path': clean})))
                    self.vue_bridge.tabChanged.emit('png')

            # ═══════ UI 설정 저장 ═══════
            elif action == 'save_ui_prefs':
                try:
                    prefs_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'config', 'ui_prefs.json')
                    os.makedirs(os.path.dirname(prefs_path), exist_ok=True)
                    prefs = {}
                    if os.path.exists(prefs_path):
                        with open(prefs_path, 'r', encoding='utf-8') as f:
                            loaded = json.load(f)
                            if isinstance(loaded, dict):
                                prefs = loaded
                    prefs.update(payload)
                    with open(prefs_path, 'w', encoding='utf-8') as f:
                        json.dump(prefs, f, ensure_ascii=False, indent=2)
                    # 재전송 하지 않음 — uiPrefsLoaded는 앱 시작 시에만 emit
                    # 재전송하면 watch → save → emit → watch 무한 루프 발생
                except Exception as e:
                    self.vue_bridge.showNotification.emit('error', f'UI 설정 저장 실패: {e}')

            # ═══════ 글로벌 가중치 저장 ═══════
            elif action == 'save_global_weights':
                try:
                    weights = payload.get('weights', [])
                    wpath = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'config', 'global_weights.json')
                    os.makedirs(os.path.dirname(wpath), exist_ok=True)
                    with open(wpath, 'w', encoding='utf-8') as f:
                        json.dump(weights, f, ensure_ascii=False, indent=2)
                    self.vue_bridge.showNotification.emit('success', '가중치가 저장되었습니다')
                except Exception as e:
                    self.vue_bridge.showNotification.emit('error', f'가중치 저장 실패: {e}')

            # ═══════ LoRA 텍스트 설정 ═══════
            elif action == 'set_lora_text':
                lora_text = payload.get('lora_text', '')
                self._vue_lora_text = lora_text

            elif action == 'set_lora_stack':
                entries = payload.get('entries', [])
                self._vue_lora_entries = entries if isinstance(entries, list) else []
                # Python lora_active_panel도 동기화
                if hasattr(self, 'lora_active_panel') and hasattr(self.lora_active_panel, 'set_entries'):
                    self.lora_active_panel.set_entries(self._vue_lora_entries)

            # ═══════ 조건부 프롬프트 저장 ═══════
            elif action == 'save_cond_rules':
                try:
                    cond_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'config', 'cond_rules.json')
                    os.makedirs(os.path.dirname(cond_path), exist_ok=True)
                    with open(cond_path, 'w', encoding='utf-8') as f:
                        json.dump(payload, f, ensure_ascii=False, indent=2)
                    self.vue_bridge.showNotification.emit('success', '조건식이 저장되었습니다')
                except Exception as e:
                    self.vue_bridge.showNotification.emit('error', f'조건식 저장 실패: {e}')

            # ═══════ 기본값 저장 ═══════
            elif action == 'save_tab_defaults':
                try:
                    defaults_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'config', 'tab_defaults.json')
                    os.makedirs(os.path.dirname(defaults_path), exist_ok=True)
                    with open(defaults_path, 'w', encoding='utf-8') as f:
                        json.dump(payload, f, ensure_ascii=False, indent=2)
                    self.vue_bridge.showNotification.emit('success', '기본값이 저장되었습니다')
                except Exception as e:
                    self.vue_bridge.showNotification.emit('error', f'기본값 저장 실패: {e}')

            # ═══════ 대기열 제어 ═══════
            elif action == 'start_queue':
                if hasattr(self, 'queue_manager'):
                    self.queue_manager.start()
                    self.show_status("Queue started.")

            elif action == 'stop_queue':
                if hasattr(self, 'queue_manager'):
                    self.queue_manager.stop()
                    self.show_status("Queue stopped.")

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

    def _load_saved_configs(self):
        """앱 시작 시 조건식 + 기본값 로드"""
        try:
            # 조건식 로드
            cond_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'config', 'cond_rules.json')
            if os.path.exists(cond_path):
                with open(cond_path, 'r', encoding='utf-8') as f:
                    rules = json.load(f)
                if hasattr(self, 'vue_bridge'):
                    self.vue_bridge.condRulesLoaded.emit(json.dumps(rules))
                print(f"[Config] Conditional rules loaded: {len(rules.get('positive',[]))}P + {len(rules.get('negative',[]))}N")
        except Exception as e:
            print(f"[Config] Failed to load cond rules: {e}")
        try:
            # 기본값 로드
            defaults_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'config', 'tab_defaults.json')
            if os.path.exists(defaults_path):
                with open(defaults_path, 'r', encoding='utf-8') as f:
                    defaults = json.load(f)
                # 확장 기본값을 proxy에 직접 적용
                if defaults.get('hires_enabled'):
                    self.hires_options_group.setChecked(True)
                if defaults.get('ad_enabled'):
                    self.adetailer_group.setChecked(True)
                if defaults.get('ad_s1_enabled') and hasattr(self, 'ad_slot1_group'):
                    self.ad_slot1_group.setChecked(True)
                if defaults.get('negpip_enabled'):
                    self.negpip_group.setChecked(True)
                print(f"[Config] Tab defaults loaded + extensions applied")
        except Exception as e:
            print(f"[Config] Failed to load defaults: {e}")
        try:
            wpath = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'config', 'global_weights.json')
            if os.path.exists(wpath):
                with open(wpath, 'r', encoding='utf-8') as f:
                    weights = json.load(f)
                if hasattr(self, 'vue_bridge'):
                    self.vue_bridge.globalWeightsLoaded.emit(json.dumps(weights))
                print(f"[Config] Global weights loaded: {len(weights)} tags")
        except Exception as e:
            print(f"[Config] Failed to load weights: {e}")
        try:
            prefs_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'config', 'ui_prefs.json')
            if os.path.exists(prefs_path):
                with open(prefs_path, 'r', encoding='utf-8') as f:
                    prefs = json.load(f)
                if hasattr(self, 'vue_bridge'):
                    self.vue_bridge.uiPrefsLoaded.emit(json.dumps(prefs))
                print(f"[Config] UI prefs loaded")
        except Exception as e:
            print(f"[Config] Failed to load UI prefs: {e}")

    def _apply_vue_conditional_rules(self, pos_rules: list, neg_rules: list):
        """Vue에서 전달된 조건부 프롬프트 규칙 적용"""
        try:
            current_tags = set(t.strip().lower() for t in self.main_prompt_text.toPlainText().split(',') if t.strip())

            for rule in pos_rules:
                cond = rule.get('condition', '').strip().lower()
                exists = rule.get('exists', True)
                target = rule.get('target', '').strip()
                action = rule.get('action', 'add')
                location = rule.get('location', 'main')
                if not cond or not target: continue

                cond_met = (cond in current_tags) if exists else (cond not in current_tags)
                if not cond_met: continue

                widget_map = {
                    'main': self.main_prompt_text,
                    'prefix': self.prefix_prompt_text,
                    'suffix': self.suffix_prompt_text,
                }
                widget = widget_map.get(location, self.main_prompt_text)
                text = widget.toPlainText()

                if action == 'add':
                    if target.lower() not in text.lower():
                        widget.setPlainText((text + ', ' + target).strip(', '))
                elif action == 'remove':
                    tags = [t.strip() for t in text.split(',')]
                    tags = [t for t in tags if t.lower() != target.lower()]
                    widget.setPlainText(', '.join(tags))
                elif action == 'replace':
                    widget.setPlainText(text.replace(cond, target))

            for rule in neg_rules:
                cond = rule.get('condition', '').strip().lower()
                exists = rule.get('exists', True)
                target = rule.get('target', '').strip()
                action = rule.get('action', 'add')
                if not cond or not target: continue

                cond_met = (cond in current_tags) if exists else (cond not in current_tags)
                if not cond_met: continue

                neg_text = self.neg_prompt_text.toPlainText()
                if action == 'add':
                    if target.lower() not in neg_text.lower():
                        self.neg_prompt_text.setPlainText((neg_text + ', ' + target).strip(', '))
                elif action == 'remove':
                    tags = [t.strip() for t in neg_text.split(',')]
                    tags = [t for t in tags if t.lower() != target.lower()]
                    self.neg_prompt_text.setPlainText(', '.join(tags))
        except Exception as e:
            print(f"[Error] Conditional rules: {e}")

    def _apply_settings_dict(self, settings: dict):
        """프리셋 딕셔너리를 UI에 적용"""
        try:
            if hasattr(self, 'vue_bridge'):
                self.vue_bridge.beginBatchUpdate()
            mapping = {
                'char_count': self.char_count_input,
                'character': self.character_input,
                'copyright': self.copyright_input,
                'main_prompt': self.main_prompt_text,
                'prefix_prompt': self.prefix_prompt_text,
                'suffix_prompt': self.suffix_prompt_text,
                'negative_prompt': self.neg_prompt_text,
                'steps': self.steps_input,
                'cfg': self.cfg_input,
                'seed': self.seed_input,
                'width': self.width_input,
                'height': self.height_input,
            }
            for key, widget in mapping.items():
                if key in settings:
                    val = str(settings[key])
                    if hasattr(widget, 'setPlainText'):
                        widget.setPlainText(val)
                    elif hasattr(widget, 'setText'):
                        widget.setText(val)
            if 'artist' in settings:
                self.artist_input.setPlainText(str(settings['artist']))
            if 'model' in settings and settings['model']:
                self.model_combo.setCurrentText(str(settings['model']))
            if 'sampler' in settings and settings['sampler']:
                self.sampler_combo.setCurrentText(str(settings['sampler']))
            if 'active_loras' in settings:
                self._vue_lora_entries = settings.get('active_loras', [])
                if hasattr(self, 'lora_active_panel'):
                    self.lora_active_panel.set_entries(self._vue_lora_entries)
                if hasattr(self, 'vue_bridge'):
                    self.vue_bridge.loraStackLoaded.emit(json.dumps(self._vue_lora_entries, ensure_ascii=False))
            if hasattr(self, 'vue_bridge'):
                self.vue_bridge.endBatchUpdate()
            self.update_total_prompt_display()
        except Exception as e:
            print(f"[Error] Apply settings dict: {e}")

    # ========== PNG Info → 즉시 생성 ==========

    def _handle_immediate_generation_from_raw(self, raw: str):
        """PNG Info raw 텍스트에서 payload 구성하여 즉시 생성"""
        try:
            parts = raw.split('\nNegative prompt: ')
            prompt = parts[0].strip()
            negative = ''
            params = {}
            if len(parts) > 1:
                sub = parts[1].split('\nSteps: ')
                negative = sub[0].strip()
                if len(sub) > 1:
                    for kv in ('Steps: ' + sub[1]).split(', '):
                        if ':' in kv:
                            k, v = kv.split(':', 1)
                            params[k.strip().lower().replace(' ', '_')] = v.strip()
            # 프롬프트 설정
            self.main_prompt_text.setPlainText(prompt)
            self.neg_prompt_text.setPlainText(negative)
            if 'steps' in params: self.steps_input.setText(params['steps'])
            if 'cfg_scale' in params: self.cfg_input.setText(params['cfg_scale'])
            if 'seed' in params: self.seed_input.setText(params['seed'])
            if 'size' in params:
                wh = params['size'].split('x')
                if len(wh) == 2:
                    self.width_input.setText(wh[0].strip())
                    self.height_input.setText(wh[1].strip())
            self.update_total_prompt_display()
            self.start_generation()
        except Exception as e:
            print(f"[Error] Immediate generation from raw: {e}")

    def _build_xyz_payload(self, combo: dict) -> dict:
        """XYZ 조합에서 생성 payload 구성"""
        payload = {
            'prompt': self.total_prompt_display.toPlainText(),
            'negative_prompt': self.neg_prompt_text.toPlainText(),
            'sampler_name': self.sampler_combo.currentText(),
            'scheduler': self.scheduler_combo.currentText(),
            'steps': int(self.steps_input.text() or 20),
            'cfg_scale': float(self.cfg_input.text() or 7),
            'seed': int(self.seed_input.text() or -1),
            'width': int(self.width_input.text() or 1024),
            'height': int(self.height_input.text() or 1024),
        }
        # XYZ 축 값 오버라이드
        for key, val in combo.items():
            if key == 'Steps': payload['steps'] = int(val)
            elif key == 'CFG Scale': payload['cfg_scale'] = float(val)
            elif key == 'Seed': payload['seed'] = int(val)
            elif key == 'Width': payload['width'] = int(val)
            elif key == 'Height': payload['height'] = int(val)
            elif key == 'Sampler': payload['sampler_name'] = val
            elif key == 'Scheduler': payload['scheduler'] = val
            elif key == 'Denoising': payload['denoising_strength'] = float(val)
            elif key == 'Prompt S/R':
                if ',' in val:
                    search, replace = val.split(',', 1)
                    payload['prompt'] = payload['prompt'].replace(search.strip(), replace.strip())
            elif key == 'Negative S/R':
                if ',' in val:
                    search, replace = val.split(',', 1)
                    payload['negative_prompt'] = payload['negative_prompt'].replace(search.strip(), replace.strip())
        return payload

    # ========== 유틸리티 메서드 ==========

    def show_status(self, message: str, timeout_ms: int = 5000):
        if hasattr(self, 'status_message_label') and self.status_message_label:
            self.status_message_label.setText(message.upper())
            if timeout_ms > 0: QTimer.singleShot(timeout_ms, lambda: self.status_message_label.clear())

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
        self.queue_panel = QueuePanel()
        self.queue_panel.setParent(None)
        self.queue_manager = QueueManager(self.queue_panel)
        self.queue_manager.generation_requested.connect(self._on_generation_requested)
        self.queue_manager.queue_completed.connect(self._on_queue_completed)
        # 대기열 상태를 Vue로 실시간 동기화
        if hasattr(self.queue_panel, 'item_added'):
            self.queue_panel.item_added.connect(self._sync_queue_to_vue)
        # add_single_item 래핑으로 Vue 동기화
        _orig_add = self.queue_panel.add_single_item
        def _wrapped_add(item):
            _orig_add(item)
            self._sync_queue_item_added(item)
        self.queue_panel.add_single_item = _wrapped_add

    def _sync_queue_item_added(self, item: dict):
        """대기열에 아이템 추가 시 Vue로 전달"""
        if hasattr(self, 'vue_bridge'):
            safe = {k: str(v)[:200] for k, v in item.items() if isinstance(v, (str, int, float, bool))}
            self.vue_bridge.queueItemAdded.emit(json.dumps(safe))

    def _sync_queue_to_vue(self):
        """전체 대기열 상태를 Vue로 전달"""
        if hasattr(self, 'vue_bridge') and hasattr(self, 'queue_panel'):
            try:
                items = []
                for i in range(self.queue_panel.list_widget.count()):
                    item_w = self.queue_panel.list_widget.item(i)
                    if item_w:
                        data = item_w.data(0x0100)  # Qt.UserRole
                        if data:
                            items.append({k: str(v)[:200] for k, v in data.items() if isinstance(v, (str, int, float, bool))})
                state = {
                    'items': items,
                    'running': hasattr(self, 'queue_manager') and self.queue_manager._running if hasattr(self.queue_manager, '_running') else False,
                    'completed': getattr(self, '_queue_completed_count', 0),
                }
                self.vue_bridge.queueUpdated.emit(json.dumps(state))
            except Exception:
                pass

    def _on_generation_requested(self, item: dict):
        self._apply_payload_to_ui(item)
        self.start_generation()

    def _on_queue_completed(self, total_count: int):
        self.is_automation_running = False
        self._queue_completed_count = total_count
        if hasattr(self, 'vue_bridge'):
            self.vue_bridge.queueCompleted.emit(json.dumps({'total': total_count}))
            self.vue_bridge.showNotification.emit('success', f'{total_count}장 생성 완료')
        QMessageBox.information(self, "Task Complete", f"Successfully generated {total_count} images.")

    def _setup_tray(self):
        self._tray_manager = TrayManager(self)
        self._tray_manager.show_window_requested.connect(self.showNormal)
        self._tray_manager.quit_requested.connect(QApplication.quit)
        self._tray_manager.show()

    def _update_vram_status(self):
        try:
            from backends import get_backend
            backend = get_backend()
            if backend:
                stats = backend.get_system_stats()
                if stats and stats.get('vram_total', 0) > 0:
                    used = stats['vram_used'] / (1024**3)
                    total = stats['vram_total'] / (1024**3)
                    pct = int(used / total * 100) if total > 0 else 0
                    if hasattr(self, 'vue_bridge'):
                        self.vue_bridge.vramUpdated.emit(json.dumps({
                            'used': round(used, 1), 'total': round(total, 1), 'pct': pct
                        }))
        except Exception:
            pass  # VRAM 모니터링은 비핵심 — GPU 미탑재 환경에서 실패 가능

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

    # ── 이벤트 검색 (비동기) ──

    def _start_event_search(self, payload):
        """이벤트 검색을 비동기 워커로 실행 (진행도 포함)"""
        loader = getattr(self.event_gen_tab, 'event_loader', None) if hasattr(self, 'event_gen_tab') else None

        if loader is None:
            # 데이터 자동 로드 후 검색
            self._pending_event_payload = payload
            self._auto_load_event_data(payload.get('ratings', ['g']))
            return

        self._run_event_search_worker(loader, payload)

    def _auto_load_event_data(self, ratings):
        """이벤트 데이터 자동 로드 (Vue 진행도 표시)"""
        from tabs.event_gen_tab import EventDataLoadWorker
        from config import EVENT_PARQUET_DIR

        self.vue_bridge.searchStatus.emit('데이터 로딩 중...')

        self._event_load_worker = EventDataLoadWorker(EVENT_PARQUET_DIR, ratings)
        self._event_load_worker.progress.connect(
            lambda msg: self.vue_bridge.searchStatus.emit(msg))
        self._event_load_worker.finished.connect(self._on_event_data_loaded)
        self._event_load_worker.start()

    def _on_event_data_loaded(self, result):
        """데이터 로드 완료 → 검색 시작"""
        if isinstance(result, str):
            self.vue_bridge.eventSearchResults.emit(json.dumps({'error': result}))
            return

        if hasattr(self, 'event_gen_tab'):
            self.event_gen_tab.event_loader = result

        payload = getattr(self, '_pending_event_payload', {})
        if payload:
            self._run_event_search_worker(result, payload)
            self._pending_event_payload = {}

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

    def _run_event_search_worker(self, loader, payload):
        """검색 워커 실행"""
        from workers.event_search_worker import EventSearchWorker
        self._event_search_worker = EventSearchWorker(loader, payload, self)
        self._event_search_worker.progress.connect(
            lambda cur, total: self.vue_bridge.eventSearchProgress.emit(cur, total))
        self._event_search_worker.finished.connect(
            lambda r: self.vue_bridge.eventSearchResults.emit(r))
        self._event_search_worker.start()
        self.show_status("이벤트 검색 시작...")

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

    def _quit_app(self):
        try:
            self.save_settings()
        except Exception as e:
            print(f"[Warning] 종료 시 설정 저장 실패: {e}")
        os._exit(0)
