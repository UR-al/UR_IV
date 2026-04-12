# ui/generator_actions.py
"""
UI 액션 및 이벤트 처리 로직
"""
import os
from PyQt6.QtWidgets import QListWidgetItem, QMessageBox
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QPixmap

from config import OUTPUT_DIR
from utils.theme_manager import get_color
from core.image_utils import exif_for_display
from utils.app_logger import get_logger
from ui.generator_generation import _gen_btn_style, _gen_btn_default_color

_logger = get_logger('actions')

class ActionsMixin:
    """UI 액션 관련 로직을 담당하는 Mixin"""
    
    def connect_signals(self):
        """시그널 연결"""
        # 생성 버튼
        self.btn_generate.clicked.connect(self.on_generate_clicked)
        self.btn_random_prompt.clicked.connect(self.apply_random_prompt)
        self.btn_save_settings.clicked.connect(self.save_settings)
        
        # 텍스트 변경 시 업데이트
        text_inputs = [
            self.char_count_input, self.character_input, 
            self.copyright_input, self.artist_input
        ]
        for inp in text_inputs:
            inp.textChanged.connect(self.on_input_changed)
        
        text_edits = [
            self.prefix_prompt_text, self.main_prompt_text, 
            self.suffix_prompt_text
        ]
        for edit in text_edits:
            edit.textChanged.connect(self.on_input_changed)
        
        # 베이스 프롬프트 변경 감지
        self.prefix_prompt_text.textChanged.connect(self.on_base_prompts_changed)
        self.suffix_prompt_text.textChanged.connect(self.on_base_prompts_changed)
        self.neg_prompt_text.textChanged.connect(self.on_base_prompts_changed)
        
        # 포커스 아웃 시 정리 (eventFilter로 처리, 디바운스 타이머의 보완)
        text_edits_to_clean = [
            self.prefix_prompt_text,
            self.main_prompt_text,
            self.suffix_prompt_text,
            self.neg_prompt_text,
            self.exclude_prompt_local_input,
            self.s1_widgets['prompt'],
            self.s2_widgets['prompt'],
        ]
        for widget in text_edits_to_clean:
            widget.installEventFilter(self)

        # 토글 버튼
        self.prefix_toggle_button.toggled.connect(
            lambda checked: self.prefix_prompt_text.setVisible(checked)
        )
        self.suffix_toggle_button.toggled.connect(
            lambda checked: self.suffix_prompt_text.setVisible(checked)
        )
        self.neg_toggle_button.toggled.connect(
            lambda checked: self.neg_prompt_text.setVisible(checked)
        )
        
        # ADetailer 토글
        self.ad_toggle_button.toggled.connect(
            lambda checked: self.ad_settings_container.setVisible(checked)
        )
        
        # ADetailer 슬롯 체크박스
        for slot_widgets in [self.s1_widgets, self.s2_widgets]:
            slot_widgets['use_inpaint_size_check'].toggled.connect(
                lambda checked, w=slot_widgets: 
                    w['inpaint_size_container'].setVisible(checked)
            )
            slot_widgets['use_steps_check'].toggled.connect(
                lambda checked, w=slot_widgets: 
                    w['steps'].setVisible(checked)
            )
            slot_widgets['use_cfg_check'].toggled.connect(
                lambda checked, w=slot_widgets: 
                    w['cfg'].setVisible(checked)
            )
            slot_widgets['use_checkpoint_check'].toggled.connect(
                lambda checked, w=slot_widgets: 
                    w['checkpoint_combo'].setVisible(checked)
            )
            slot_widgets['use_vae_check'].toggled.connect(
                lambda checked, w=slot_widgets: 
                    w['vae_combo'].setVisible(checked)
            )
            slot_widgets['use_sampler_check'].toggled.connect(
                lambda checked, w=slot_widgets: 
                    w['sampler_container'].setVisible(checked)
            )
        
        # 랜덤 해상도
        self.random_res_check.toggled.connect(self.toggle_random_resolution_editor)
        self.btn_add_res.clicked.connect(self.add_resolution_item)
        
        # 즐겨찾기
        self.btn_add_favorite.clicked.connect(self.add_to_favorites)
        if hasattr(self, 'btn_fav_refresh'):
            self.btn_fav_refresh.clicked.connect(self.refresh_favorites)
        if hasattr(self, 'btn_fav_clear'):
            self.btn_fav_clear.clicked.connect(self.clear_all_favorites)        
        
        # 이벤트 탭 시그널 연결
        if hasattr(self, 'event_gen_tab'):
            if hasattr(self.event_gen_tab, 'btn_load_base'):
                self.event_gen_tab.btn_load_base.clicked.connect(
                    self.load_base_prompt_to_event
                )
            if hasattr(self.event_gen_tab, 'send_to_queue_signal'):
                self.event_gen_tab.send_to_queue_signal.connect(
                    self.receive_event_scenarios
                )
        
        # PNG Info 시그널 연결
        if hasattr(self, 'png_info_tab'):
            self.png_info_tab.generate_signal.connect(
                lambda payload: self.handle_immediate_generation(payload)
            )
            self.png_info_tab.send_prompt_signal.connect(
                lambda p, n: self.handle_prompt_only_transfer(p, n)
            )
            # I2I/Inpaint 전송 시그널
            if hasattr(self.png_info_tab, 'send_to_i2i_signal') and hasattr(self, 'i2i_tab'):
                self.png_info_tab.send_to_i2i_signal.connect(
                    lambda payload: self._handle_send_to_i2i(payload)
                )
            if hasattr(self.png_info_tab, 'send_to_inpaint_signal') and hasattr(self, 'inpaint_tab'):
                self.png_info_tab.send_to_inpaint_signal.connect(
                    lambda payload: self._handle_send_to_inpaint(payload)
                )
            if hasattr(self.png_info_tab, 'send_to_queue_signal'):
                self.png_info_tab.send_to_queue_signal.connect(self._gallery_send_to_queue)

        # Gallery 시그널 연결
        if hasattr(self, 'gallery_tab'):
            self.gallery_tab.send_prompt_signal.connect(
                lambda p, n: self.handle_prompt_only_transfer(p, n)
            )
            self.gallery_tab.generate_signal.connect(
                lambda payload: self.handle_immediate_generation(payload)
            )
            self.gallery_tab.open_in_editor.connect(self._gallery_send_to_editor)
            self.gallery_tab.send_to_i2i.connect(self._gallery_send_to_i2i)
            self.gallery_tab.send_to_inpaint.connect(self._gallery_send_to_inpaint)
            self.gallery_tab.send_to_upscale.connect(self._gallery_send_to_upscale)
            self.gallery_tab.send_to_queue_signal.connect(self._gallery_send_to_queue)
            if hasattr(self.gallery_tab, 'send_to_compare'):
                self.gallery_tab.send_to_compare.connect(self._gallery_send_to_compare)

        if hasattr(self, 'xyz_plot_tab'):
            self.xyz_plot_tab.add_to_queue_requested.connect(self._on_xyz_add_to_queue)
            self.xyz_plot_tab.start_generation_requested.connect(self._on_xyz_start_generation)

        # T2I 뷰어 우클릭 메뉴
        self.setup_viewer_context_menu()
    
    def on_generate_clicked(self):
        """생성 버튼 클릭 (일반 생성 또는 자동화 시작/중지)"""
        # 자동화 모드가 켜져 있으면
        if self.btn_auto_toggle.isChecked():
            if self.is_automating:
                # 자동화 중지
                self._stop_automation("사용자가 자동화를 중지했습니다.")
            else:
                # 자동화 시작
                self._start_automation()
        else:
            # 일반 이미지 생성
            self.start_generation()
    
    def on_input_changed(self):
        """입력 변경 시 최종 프롬프트 업데이트"""
        if not self.is_programmatic_change:
            self.update_total_prompt_display()
    
    def toggle_automation_ui(self, checked):
        """자동화 모드 토글 (ON/OFF만 — 패널 표시는 별도 접이식)"""
        # 생성 중이면 토글 무시
        if hasattr(self, 'gen_worker') and self.gen_worker and self.gen_worker.isRunning():
            self.btn_auto_toggle.setChecked(not checked)
            QMessageBox.warning(self, "알림", "이미지 생성 중에는 자동화 모드를 변경할 수 없습니다.")
            return

        if checked:
            self.btn_auto_toggle.setText("AUTOMATION: ON")
            self.btn_auto_toggle.setStyleSheet(f"""
                QPushButton {{
                    background-color: {get_color('success')}; color: black;
                    border: none; border-radius: 5px; font-weight: bold;
                }}
                QPushButton:hover {{ background-color: {get_color('success')}; }}
            """)
            self.btn_generate.setText("자동화 시작")
        else:
            if self.is_automating:
                self._stop_automation("자동화가 중지되었습니다.")
            self.btn_auto_toggle.setText("AUTOMATION: OFF")
            self.btn_auto_toggle.setStyleSheet("")  # 테마 기본 스타일 복원
            self.btn_generate.setText("이미지 생성")
            
    def _on_automation_generation_finished(self, result, gen_info):
        """자동화 생성 완료"""
        if not hasattr(self, 'auto_gen_count'):
            self.auto_gen_count = 0
        
        if isinstance(result, bytes):
            self.auto_gen_count += 1
            self._process_new_image(result, gen_info)
            self.show_status(
                f"🔄 자동 생성 중... ({self.auto_gen_count}장 완료)"
            )
            self._emit_auto_status()
        else:
            self.show_status(f"⚠️ 생성 실패: {result}")
        
        # 다음 생성 계속
        if self.is_automating:
            self._continue_automation()
            
    def toggle_random_resolution_editor(self, checked):
        """랜덤 해상도 편집기 토글"""
        if checked:
            self.resolution_editor_container.show()
            self._update_resolution_list()
            self._update_random_res_label()
        else:
            self.resolution_editor_container.hide()
            self.random_res_label.clear()
    
    def add_resolution_item(self):
        """해상도 추가 (이름 자동 생성: WxH)"""
        try:
            width = int(self.res_width_input.text())
            height = int(self.res_height_input.text())

            desc = f"{width}x{height}"
            self.random_resolutions.append((width, height, desc))
            self._update_resolution_list()

            self.res_width_input.clear()
            self.res_height_input.clear()

        except ValueError:
            QMessageBox.warning(self, "오류", "올바른 숫자를 입력해주세요.")
    
    def _update_resolution_list(self):
        """해상도 리스트 업데이트 (Vue SPA에서는 프록시이므로 스킵)"""
        from PyQt6.QtWidgets import QListWidget, QListWidgetItem
        # LWProxy인 경우 (Vue 모드) — PyQt UI 업데이트 불필요
        if not isinstance(self.resolution_list_widget, QListWidget):
            return
        self.resolution_list_widget.clear()
        for i, (w, h, desc) in enumerate(self.random_resolutions):
            from widgets.common_widgets import ResolutionItemWidget
            item_widget = ResolutionItemWidget(w, h, desc, i)
            item_widget.delete_requested.connect(self.delete_resolution_item)
            item = QListWidgetItem(self.resolution_list_widget)
            item.setSizeHint(item_widget.sizeHint())
            self.resolution_list_widget.addItem(item)
            self.resolution_list_widget.setItemWidget(item, item_widget)
    
    def delete_resolution_item(self, index):
        """해상도 삭제"""
        if 0 <= index < len(self.random_resolutions):
            del self.random_resolutions[index]
            self._update_resolution_list()
    
    def _update_random_res_label(self):
        """랜덤 해상도 라벨 업데이트"""
        if self.random_resolutions:
            res_list = ", ".join([
                f"{desc}({w}x{h})" 
                for w, h, desc in self.random_resolutions
            ])
            self.random_res_label.setText(f"등록된 해상도: {res_list}")
        else:
            self.random_res_label.setText("등록된 해상도가 없습니다.")
    
    def on_favorite_item_clicked(self, item):
        """즐겨찾기 아이템 클릭"""
        path = item.data(Qt.ItemDataRole.UserRole)
        if os.path.exists(path):
            self.display_image(path)
    
    def _apply_next_automation_prompt(self) -> bool:
        """자동화용 다음 프롬프트 적용 (apply_random_prompt와 동일한 로직 사용)"""
        import random
        
        settings = self.auto_settings
        
        # 덱이 비었으면 리필
        if not self.shuffled_prompt_deck:
            if settings.get('allow_duplicates', False):
                # 중복 허용: 덱 리필
                self.shuffled_prompt_deck = self.filtered_results.copy()
                random.shuffle(self.shuffled_prompt_deck)
                self.show_status("🔄 덱을 다시 섞었습니다.")
            else:
                # 중복 불허: 종료
                return False
        
        # 덱에서 프롬프트 가져오기
        if settings.get('allow_duplicates', False):
            # 중복 허용: 랜덤 선택 (덱에서 제거 안 함)
            bundle = random.choice(self.shuffled_prompt_deck)
        else:
            # 중복 불허: 덱에서 제거
            bundle = self.shuffled_prompt_deck.pop()
            remaining = len(self.shuffled_prompt_deck)
            self.btn_random_prompt.setText(f"🎲 랜덤 프롬프트 ({remaining})")
        
        # ★★★ 핵심: apply_prompt_from_data 호출 ★★★
        # 이게 UI 업데이트 + 토글 적용 + 필터링 전부 처리함
        self.apply_prompt_from_data(bundle)
        
        return True

    def _start_automation(self):
        """자동화 시작"""
        if not self.filtered_results:
            QMessageBox.warning(self, "알림", "먼저 검색을 수행하세요.")
            return
        
        import time
        import random
        
        self.is_automating = True
        self.auto_gen_count = 0
        self.auto_current_repeat = 1

        settings = self.automation_widget.get_settings()
        self.auto_settings = settings
        self._emit_auto_status()
        
        # 시간 제한 모드면 시작 시간 기록
        if settings['termination_mode'] == 'timer':
            self.auto_start_time = time.time()
        
        # 덱 초기화
        self.shuffled_prompt_deck = self.filtered_results.copy()
        random.shuffle(self.shuffled_prompt_deck)
        self.btn_random_prompt.setText(f"🎲 랜덤 프롬프트 ({len(self.shuffled_prompt_deck)})")
        
        # 버튼 상태 변경
        self.btn_generate.setText("⏸️ 자동화 중지")
        self.btn_generate.setStyleSheet(_gen_btn_style('#e74c3c'))
        
        self.show_status("🔄 자동화 시작...")
        
        # ★★★ 첫 번째 프롬프트 적용 (apply_random_prompt 사용!) ★★★
        self.apply_random_prompt()
        
        # 첫 생성 시작
        from PyQt6.QtCore import QTimer
        delay_ms = int(settings['delay'] * 1000)
        QTimer.singleShot(delay_ms, self.start_generation)


    def _run_automation_cycle(self):
        """자동화 사이클"""
        if not self.is_automating:
            return
        
        import time
        from PyQt6.QtCore import QTimer
        
        settings = self.auto_settings
        
        # 종료 조건 확인
        if settings['termination_mode'] == 'count':
            if self.auto_gen_count >= settings['termination_limit']:
                self._stop_automation(f"✅ 자동화 완료: {self.auto_gen_count}장 생성")
                return
        else:  # timer
            elapsed = time.time() - self.auto_start_time
            if elapsed >= settings['termination_limit']:
                self._stop_automation(f"✅ 시간 종료: {self.auto_gen_count}장 생성")
                return
        
        # 반복 횟수 확인
        if self.auto_current_repeat >= settings['repeat_per_prompt']:
            self.auto_current_repeat = 0
            
            # ★★★ 새 프롬프트 적용 (apply_random_prompt 사용!) ★★★
            # 덱이 비었는지 확인
            if not self.shuffled_prompt_deck and not settings.get('allow_duplicates', False):
                self._stop_automation("✅ 모든 프롬프트 처리 완료!")
                return
            
            self.apply_random_prompt()
        
        # 대기 후 생성
        delay_ms = int(settings['delay'] * 1000)
        self._emit_auto_status(waiting=(delay_ms > 0))
        QTimer.singleShot(delay_ms, self._automation_generate)


    def _automation_generate(self):
        """자동화 이미지 생성"""
        if not self.is_automating:
            return
        
        self.auto_current_repeat += 1
        self.start_generation()


    def _continue_automation(self):
        """자동화 계속 (on_generation_finished에서 호출)"""
        if self.is_automating:
            self._run_automation_cycle()
            

    def _emit_auto_status(self, waiting=False):
        """Vue에 자동화 상태 전송"""
        if hasattr(self, 'vue_bridge'):
            import json
            self.vue_bridge.automationStatus.emit(json.dumps({
                'running': self.is_automating,
                'count': getattr(self, 'auto_gen_count', 0),
                'waiting': waiting,
            }))

    def _stop_automation(self, message=None):
        """자동화 중지"""
        self.is_automating = False
        self._emit_auto_status()
        
        # 버튼 상태 복구 (자동화 모드는 유지)
        if self.btn_auto_toggle.isChecked():
            self.btn_generate.setText("🚀 자동화 시작")
            self.btn_generate.setStyleSheet(_gen_btn_style('#27ae60'))
        else:
            self.btn_generate.setText("✨ 이미지 생성")
            self.btn_generate.setStyleSheet(_gen_btn_style(_gen_btn_default_color()))

        self.btn_generate.setEnabled(True)
        
        if message:
            self.show_status(message)
            QMessageBox.information(self, "자동화", message)
        else:
            self.show_status(f"✅ 자동화 완료: {self.auto_gen_count}장 생성됨")
            
    def receive_event_scenarios(self, scenarios):
        """이벤트 시나리오를 대기열에 추가"""
        added_count = 0
        for scenario in scenarios:
            payload = scenario.get('payload', {})

            if not payload or 'prompt' not in payload:
                _logger.warning(f"잘못된 시나리오: {scenario}")
                continue

            self.queue_panel.add_single_item(payload)
            added_count += 1

        self.show_status(f"✅ {added_count}개의 이벤트가 대기열에 추가됨")
        QMessageBox.information(
            self, "전송 완료",
            f"{added_count}개의 이벤트가 대기열에 추가되었습니다."
        )

    def _on_center_tab_changed(self, index):
        """중앙 탭 전환 시 처리"""
        if not hasattr(self, 'center_tabs'):
            return
        current_widget = self.center_tabs.widget(index)

        # 탭에 따라 왼쪽 패널 전환
        if hasattr(self, 'left_stack'):
            if hasattr(self, 'mosaic_editor') and current_widget == self.mosaic_editor:
                self.left_stack.setFixedWidth(460)
                self.left_stack.setCurrentIndex(1)  # 에디터 도구
            else:
                # Vue 탭이면 _handle_vue_action에서 처리
                pass

        # 즐겨찾기 탭으로 전환 시 자동 새로고침
        if hasattr(self, 'fav_tab') and current_widget == self.fav_tab:
            self.refresh_favorites()

        # Gallery 탭 전환 시 저장된 폴더 자동 로드
        if hasattr(self, 'gallery_tab') and current_widget == self.gallery_tab:
            if not self.gallery_tab._all_paths and self.gallery_tab._current_folder:
                self.gallery_tab.load_folder(self.gallery_tab._current_folder)
                
    def handle_prompt_only_transfer(self, prompt, negative):
        """PNG Info/Gallery에서 프롬프트만 전송"""
        classified = self.tag_classifier.classify_tags_for_event([t.strip() for t in prompt.split(',') if t.strip()])
        bundle = {
            'general': ', '.join(classified["costume"] + classified["appearance"] + classified["expression"] + classified["action"] + classified["background"] + classified["composition"] + classified["effect"] + classified["objects"] + classified["general"]),
            'character': ', '.join(classified["character"]),
            'copyright': ', '.join(classified["copyright"]),
            'artist': ''
        }
        self.apply_prompt_from_data(bundle)
        self.neg_prompt_text.setPlainText(negative)
        # Vue에서 T2I 탭으로 전환 유도
        if hasattr(self, 'vue_bridge'):
            self.vue_bridge.tabChanged.emit('t2i')
        self.show_status("✅ 프롬프트 전송 완료")

    def _handle_send_to_i2i(self, payload):
        """I2I 탭으로 전송 (Vue 호환)"""
        if hasattr(self, 'i2i_tab'):
            self.i2i_tab.load_from_payload(payload)
            if hasattr(self, 'vue_bridge'):
                self.vue_bridge.tabChanged.emit('i2i')
            self.show_status("✅ I2I 탭으로 전송 완료")

    def _handle_send_to_inpaint(self, payload):
        """Inpaint 탭으로 전송 (Vue 호환)"""
        if hasattr(self, 'inpaint_tab'):
            self.inpaint_tab.load_from_payload(payload)
            if hasattr(self, 'vue_bridge'):
                self.vue_bridge.tabChanged.emit('inpaint')
            self.show_status("✅ Inpaint 탭으로 전송 완료")

    def _gallery_send_to_editor(self, path: str):
        """에디터 탭으로 이미지 전송 (Vue 호환)"""
        if hasattr(self, 'vue_bridge'):
            self.vue_bridge.editorImageLoaded.emit(path.replace('\\', '/'))
            self.vue_bridge.tabChanged.emit('editor')
            self.show_status(f"✅ 에디터로 전송: {os.path.basename(path)}")

    def _gallery_send_to_i2i(self, path: str):
        """I2I 탭으로 이미지 전송 (Vue 호환)"""
        if hasattr(self, 'i2i_tab') and hasattr(self.i2i_tab, '_load_image'):
            self.i2i_tab._load_image(path)
            if hasattr(self, 'vue_bridge'):
                self.vue_bridge.tabChanged.emit('i2i')
            self.show_status(f"✅ I2I로 전송: {os.path.basename(path)}")

    def _gallery_send_to_inpaint(self, path: str):
        """Inpaint 탭으로 이미지 전송 (Vue 호환)"""
        if hasattr(self, 'inpaint_tab') and hasattr(self.inpaint_tab, '_load_image'):
            self.inpaint_tab._load_image(path)
            if hasattr(self, 'vue_bridge'):
                self.vue_bridge.tabChanged.emit('inpaint')
            self.show_status(f"✅ Inpaint로 전송: {os.path.basename(path)}")

    def _gallery_send_to_upscale(self, path: str):
        """Gallery에서 Upscale 탭으로 이미지 전송"""
        if hasattr(self, 'upscale_tab') and hasattr(self.upscale_tab, '_add_file'):
            self.upscale_tab._add_file(path)
            idx = self.center_tabs.indexOf(self.upscale_tab)
            if idx >= 0:
                self.center_tabs.setCurrentIndex(idx)
            self.show_status(f"✅ Upscale로 전송: {os.path.basename(path)}")

    def _gallery_send_to_queue(self, payload: dict):
        """Gallery/Favorites에서 대기열에 추가"""
        if hasattr(self, 'queue_panel'):
            self.queue_panel.add_single_item(payload)
            self.show_status("📋 대기열에 추가되었습니다.")

    def _gallery_send_to_compare(self, path_a: str, path_b: str):
        """Gallery에서 두 이미지를 PNG Info 비교 탭으로 전송"""
        if hasattr(self, 'png_info_tab'):
            self.png_info_tab.load_compare_images(path_a, path_b)
            idx = self.center_tabs.indexOf(self.png_info_tab)
            if idx >= 0:
                self.center_tabs.setCurrentIndex(idx)
            self.show_status(
                f"🔍 이미지 비교: {os.path.basename(path_a)} vs {os.path.basename(path_b)}"
            )