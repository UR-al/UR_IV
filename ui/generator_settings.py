# ui/generator_settings.py
"""
설정 저장 및 불러오기 관련 로직
"""
import os
import json
from PyQt6.QtWidgets import QMessageBox, QFileDialog
from config import PROMPT_SETTINGS_FILE
from utils.shortcut_manager import get_shortcut_manager

class SettingsMixin:
    """설정 관련 로직을 담당하는 Mixin"""

    @staticmethod
    def _sync_slider(line_edit):
        """QLineEdit의 텍스트 값으로 연결된 슬라이더 위치 동기화"""
        if hasattr(line_edit, '_slider') and hasattr(line_edit, '_multiplier'):
            try:
                value = float(line_edit.text())
                line_edit._slider.setValue(int(value * line_edit._multiplier))
            except (ValueError, AttributeError):
                pass

    def save_settings(self):
        """설정 저장"""
        try:
            settings = self._build_settings_dict()
            with open(PROMPT_SETTINGS_FILE, 'w', encoding='utf-8') as f:
                json.dump(settings, f, ensure_ascii=False, indent=4)
            self.show_status("✅ 설정이 저장되었습니다.", 3000)
        except Exception as e:
            from utils.app_logger import get_logger
            get_logger('settings').error(f"설정 저장 실패: {e}")
            QMessageBox.critical(self, "저장 실패", f"설정 저장 중 오류: {e}")

    def _get_existing_setting(self, key: str, default: str = "") -> str:
        """기존 설정 파일에서 특정 키의 값을 읽어옴 (콤보 빈값 가드용)"""
        try:
            if os.path.exists(PROMPT_SETTINGS_FILE):
                with open(PROMPT_SETTINGS_FILE, 'r', encoding='utf-8') as f:
                    return json.load(f).get(key, default)
        except Exception:
            pass
        return default

    def _build_settings_dict(self) -> dict:
        """설정 딕셔너리 구성"""
        # 콤보가 비어있으면 기존 파일의 값 유지 (백엔드 미연결 시 ""로 덮어쓰기 방지)
        model_val = self.model_combo.currentText() or self._get_existing_setting("model")
        sampler_val = self.sampler_combo.currentText() or self._get_existing_setting("sampler")
        scheduler_val = self.scheduler_combo.currentText() or self._get_existing_setting("scheduler")
        active_loras = getattr(self, '_vue_lora_entries', None)
        if not isinstance(active_loras, list):
            active_loras = self.lora_active_panel.get_entries() if hasattr(self, 'lora_active_panel') else []

        settings = {
            "char_count": self.char_count_input.text(),
            "character": self.character_input.text(),
            "copyright": self.copyright_input.text(),
            "artist": self.artist_input.toPlainText(),
            "artist_locked": self.btn_lock_artist.isChecked(),
            "prefix_prompt": self.prefix_prompt_text.toPlainText(),
            "main_prompt": self.main_prompt_text.toPlainText(),
            "suffix_prompt": self.suffix_prompt_text.toPlainText(),
            "negative_prompt": self.neg_prompt_text.toPlainText(),
            "exclude_prompt_local": self.exclude_prompt_local_input.toPlainText(),

            "model": model_val,
            "sampler": sampler_val,
            "scheduler": scheduler_val,
            "steps": self.steps_input.text(),
            "cfg": self.cfg_input.text(),
            "seed": self.seed_input.text(),
            "width": self.width_input.text(),
            "height": self.height_input.text(),

            "random_res_enabled": self.random_res_check.isChecked(),
            "random_resolutions": self.random_resolutions,
            "res_presets": self._res_presets if hasattr(self, '_res_presets') else [],
            "active_loras": active_loras,

            "hires_enabled": self.hires_options_group.isChecked(),
            "hires_upscaler": self.upscaler_combo.currentText() or self._get_existing_setting("hires_upscaler"),
            "hires_steps": self.hires_steps_input.text(),
            "hires_denoising": self.hires_denoising_input.text(),
            "hires_scale": self.hires_scale_input.text(),
            "hires_cfg": self.hires_cfg_input.text(),
            "hires_checkpoint": self.hires_checkpoint_combo.currentText() or self._get_existing_setting("hires_checkpoint"),
            "hires_sampler": self.hires_sampler_combo.currentText() or self._get_existing_setting("hires_sampler"),
            "hires_scheduler": self.hires_scheduler_combo.currentText() or self._get_existing_setting("hires_scheduler"),
            "hires_prompt": self.hires_prompt_text.toPlainText(),
            "hires_neg_prompt": self.hires_neg_prompt_text.toPlainText(),

            "negpip_enabled": self.negpip_group.isChecked() if hasattr(self, 'negpip_group') else False,

            "adetailer_enabled": self.adetailer_group.isChecked(),
            "adetailer_slot1_enabled": self.ad_slot1_group.isChecked(),
            "adetailer_slot2_enabled": self.ad_slot2_group.isChecked(),
            
            "adetailer_slot1": self._get_slot_settings(self.s1_widgets),
            "adetailer_slot2": self._get_slot_settings(self.s2_widgets),
            
            "prefix_toggle": self.prefix_toggle_button.isChecked(),
            "suffix_toggle": self.suffix_toggle_button.isChecked(),
            "neg_toggle": self.neg_toggle_button.isChecked(),
            "exclude_toggle": self.exclude_toggle_button.isChecked(),
            
            "remove_artist": self.chk_remove_artist.isChecked(),
            "remove_copyright": self.chk_remove_copyright.isChecked(),
            "remove_character": self.chk_remove_character.isChecked(),
            "remove_meta": self.chk_remove_meta.isChecked(),
            "remove_censorship": self.chk_remove_censorship.isChecked(),
            "remove_text": self.chk_remove_text.isChecked(),

            "auto_char_features": self.chk_auto_char_features.isChecked() if hasattr(self, 'chk_auto_char_features') else False,
            "char_feature_mode": self.combo_char_feature_mode.currentIndex() if hasattr(self, 'combo_char_feature_mode') else 0,

            "cond_prompt_enabled": self.cond_prompt_check.isChecked(),
            "cond_rules_json": self._get_combined_cond_rules_json(),
            "cond_prevent_dupe": self.cond_prevent_dupe_check.isChecked(),

            "base_prefix_prompt": self.base_prefix_prompt,
            "base_suffix_prompt": self.base_suffix_prompt,
            "base_neg_prompt": self.base_neg_prompt,
            
            # 웹 설정 추가!
            "web_home_url": self.web_tab.home_url if hasattr(self.web_tab, 'home_url') else "",

            "theme": self.settings_tab.theme_combo.currentText() if hasattr(self.settings_tab, 'theme_combo') else "다크",
            "font_family": self.settings_tab.font_combo.currentText() if hasattr(self.settings_tab, 'font_combo') else "Pretendard",
            "font_size": self.settings_tab.font_size_spin.value() if hasattr(self.settings_tab, 'font_size_spin') else 10.5,

            "cleaning_options": self.settings_tab.get_cleaning_options(),
            "editor_defaults": self.settings_tab.get_editor_defaults(),
            "bg_removal_model": self.mosaic_editor.mosaic_panel.bg_model_combo.currentData() if hasattr(self, 'mosaic_editor') else "u2net",
            "shortcuts": get_shortcut_manager().to_dict(),
            "wildcard_enabled": self.settings_tab.chk_wildcard_enabled.isChecked() if hasattr(self.settings_tab, 'chk_wildcard_enabled') else True,

            "parquet_dir": self.settings_tab.parquet_dir_input.text() if hasattr(self.settings_tab, 'parquet_dir_input') else "",
            "event_parquet_dir": self.settings_tab.event_parquet_dir_input.text() if hasattr(self.settings_tab, 'event_parquet_dir_input') else "",

            "gallery_folder": self.gallery_tab._current_folder if hasattr(self, 'gallery_tab') else "",

            # 백엔드 설정
            "backend_type": self._get_backend_type_str(),
            "comfyui_url": self._get_comfyui_url(),
            "comfyui_workflow_path": self._get_comfyui_workflow_path(),

            # I2I 탭 설정
            "i2i_settings": self._get_i2i_settings() if hasattr(self, 'i2i_tab') else {},

            # Inpaint 탭 설정
            "inpaint_settings": self._get_inpaint_settings() if hasattr(self, 'inpaint_tab') else {},

            # Search 탭 설정
            "search_criteria": self.search_tab._get_criteria_dict() if hasattr(self, 'search_tab') else {},
        }
        return settings
    
    def load_settings(self):
        """설정 불러오기"""
        if not os.path.exists(PROMPT_SETTINGS_FILE):
            return
        
        try:
            with open(PROMPT_SETTINGS_FILE, 'r', encoding='utf-8') as f:
                settings = json.load(f)
            
            self.is_programmatic_change = True
            
            # 기본 입력
            self.char_count_input.setText(settings.get("char_count", ""))
            self.character_input.setText(settings.get("character", ""))
            self.copyright_input.setText(settings.get("copyright", ""))
            self.artist_input.setPlainText(settings.get("artist", ""))
            self.btn_lock_artist.setChecked(settings.get("artist_locked", False))
            
            # 프롬프트
            self.prefix_prompt_text.setPlainText(settings.get("prefix_prompt", ""))
            self.main_prompt_text.setPlainText(settings.get("main_prompt", ""))
            self.suffix_prompt_text.setPlainText(settings.get("suffix_prompt", ""))
            self.neg_prompt_text.setPlainText(settings.get("negative_prompt", ""))
            self.exclude_prompt_local_input.setPlainText(settings.get("exclude_prompt_local", ""))
            
            # 생성 파라미터
            model_text = settings.get("model", "")
            idx = self.model_combo.findText(model_text)
            if idx >= 0: 
                self.model_combo.setCurrentIndex(idx)
            
            sampler_text = settings.get("sampler", "")
            idx = self.sampler_combo.findText(sampler_text)
            if idx >= 0: 
                self.sampler_combo.setCurrentIndex(idx)
            
            scheduler_text = settings.get("scheduler", "")
            idx = self.scheduler_combo.findText(scheduler_text)
            if idx >= 0: 
                self.scheduler_combo.setCurrentIndex(idx)
            
            self.steps_input.setText(settings.get("steps", "25"))
            self.cfg_input.setText(settings.get("cfg", "7.0"))
            self._sync_slider(self.steps_input)
            self._sync_slider(self.cfg_input)
            self.seed_input.setText(settings.get("seed", "-1"))
            self.width_input.setText(settings.get("width", "1024"))
            self.height_input.setText(settings.get("height", "1024"))

            # 해상도 프리셋 복원
            saved_presets = settings.get("res_presets", [])
            if saved_presets and hasattr(self, '_res_presets') and hasattr(self, '_res_preset_btns'):
                for i, preset in enumerate(saved_presets):
                    if i < len(self._res_presets) and i < len(self._res_preset_btns):
                        self._res_presets[i] = list(preset)
                        self._res_preset_btns[i].setText(preset[0])

            # LoRA 활성 목록 복원
            if hasattr(self, 'lora_active_panel') and "active_loras" in settings:
                self.lora_active_panel.set_entries(settings["active_loras"])
            self._vue_lora_entries = settings.get("active_loras", [])
            if hasattr(self, 'vue_bridge'):
                self.vue_bridge.loraStackLoaded.emit(json.dumps(self._vue_lora_entries, ensure_ascii=False))

            # 랜덤 해상도
            self.random_res_check.setChecked(settings.get("random_res_enabled", False))
            if "random_resolutions" in settings:
                self.random_resolutions = settings["random_resolutions"]
                self._update_resolution_list()
            
            # Hires.fix
            self.hires_options_group.setChecked(settings.get("hires_enabled", False))
            upscaler_text = settings.get("hires_upscaler", "")
            idx = self.upscaler_combo.findText(upscaler_text)
            if idx >= 0: 
                self.upscaler_combo.setCurrentIndex(idx)
            self.hires_steps_input.setText(settings.get("hires_steps", "0"))
            self.hires_denoising_input.setText(settings.get("hires_denoising", "0.4"))
            self.hires_scale_input.setText(settings.get("hires_scale", "2.0"))
            self.hires_cfg_input.setText(settings.get("hires_cfg", "0"))
            self._sync_slider(self.hires_steps_input)
            self._sync_slider(self.hires_denoising_input)
            self._sync_slider(self.hires_scale_input)
            self._sync_slider(self.hires_cfg_input)

            hr_ckpt = settings.get("hires_checkpoint", "")
            idx = self.hires_checkpoint_combo.findText(hr_ckpt)
            if idx >= 0:
                self.hires_checkpoint_combo.setCurrentIndex(idx)

            hr_sampler = settings.get("hires_sampler", "")
            idx = self.hires_sampler_combo.findText(hr_sampler)
            if idx >= 0:
                self.hires_sampler_combo.setCurrentIndex(idx)

            hr_scheduler = settings.get("hires_scheduler", "")
            idx = self.hires_scheduler_combo.findText(hr_scheduler)
            if idx >= 0:
                self.hires_scheduler_combo.setCurrentIndex(idx)

            self.hires_prompt_text.setPlainText(settings.get("hires_prompt", ""))
            self.hires_neg_prompt_text.setPlainText(settings.get("hires_neg_prompt", ""))

            # NegPiP
            if hasattr(self, 'negpip_group'):
                self.negpip_group.setChecked(settings.get("negpip_enabled", False))
            
            # ADetailer
            self.adetailer_group.setChecked(settings.get("adetailer_enabled", False))
            self.ad_slot1_group.setChecked(settings.get("adetailer_slot1_enabled", False))
            self.ad_slot2_group.setChecked(settings.get("adetailer_slot2_enabled", False))
            
            if "adetailer_slot1" in settings:
                self._set_slot_settings(self.s1_widgets, settings["adetailer_slot1"])
            if "adetailer_slot2" in settings:
                self._set_slot_settings(self.s2_widgets, settings["adetailer_slot2"])
            
            # 토글 상태
            self.prefix_toggle_button.setChecked(settings.get("prefix_toggle", True))
            self.suffix_toggle_button.setChecked(settings.get("suffix_toggle", True))
            self.neg_toggle_button.setChecked(settings.get("neg_toggle", True))
            self.exclude_toggle_button.setChecked(settings.get("exclude_toggle", True))
            
            # 제거 옵션
            self.chk_remove_artist.setChecked(settings.get("remove_artist", False))
            self.chk_remove_copyright.setChecked(settings.get("remove_copyright", False))
            self.chk_remove_character.setChecked(settings.get("remove_character", False))
            self.chk_remove_meta.setChecked(settings.get("remove_meta", False))
            self.chk_remove_censorship.setChecked(settings.get("remove_censorship", False))
            self.chk_remove_text.setChecked(settings.get("remove_text", False))

            # 웹 설정 불러오기
            web_home = settings.get("web_home_url", "")
            if web_home and hasattr(self.web_tab, 'set_home_url'):
                self.web_tab.set_home_url(web_home)
            
            # 캐릭터 특징 자동 추가
            if hasattr(self, 'chk_auto_char_features'):
                self.chk_auto_char_features.setChecked(settings.get("auto_char_features", False))
            if hasattr(self, 'combo_char_feature_mode'):
                self.combo_char_feature_mode.setCurrentIndex(settings.get("char_feature_mode", 0))

            # 조건부 프롬프트
            self.cond_prompt_check.setChecked(settings.get("cond_prompt_enabled", False))
            self.cond_prevent_dupe_check.setChecked(settings.get("cond_prevent_dupe", True))

            # 새 JSON 포맷 우선, 없으면 기존 텍스트 포맷 마이그레이션
            cond_json = settings.get("cond_rules_json", "")
            all_rules = []
            if cond_json:
                from utils.condition_block import rules_from_json
                all_rules = rules_from_json(cond_json)
            else:
                old_pos = settings.get("cond_prompt_rules", "")
                old_neg = settings.get("cond_neg_rules", "")
                if old_pos or old_neg:
                    from utils.condition_block import migrate_old_rules
                    if old_pos:
                        all_rules.extend(migrate_old_rules(old_pos))
                    if old_neg:
                        neg_rules = migrate_old_rules(old_neg)
                        for r in neg_rules:
                            r.location = "neg"
                        all_rules.extend(neg_rules)
            if all_rules:
                pos_rules = [r for r in all_rules if r.location != "neg"]
                neg_rules = [r for r in all_rules if r.location == "neg"]
                self.cond_block_editor_pos.set_rules(pos_rules)
                self.cond_block_editor_neg.set_rules(neg_rules)
            
            # 베이스 프롬프트
            self.base_prefix_prompt = settings.get("base_prefix_prompt", "")
            self.base_suffix_prompt = settings.get("base_suffix_prompt", "")
            self.base_neg_prompt = settings.get("base_neg_prompt", "")

            # 테마 복원
            theme_name = settings.get("theme", "다크")
            if hasattr(self.settings_tab, 'theme_combo'):
                self.settings_tab.theme_combo.setCurrentText(theme_name)
            if hasattr(self, 'set_theme'):
                self.set_theme(theme_name)

            # 글꼴 복원
            font_family = settings.get("font_family", "")
            font_size = settings.get("font_size", 10.5)
            if font_family:
                from utils.theme_manager import get_theme_manager
                tm = get_theme_manager()
                tm.set_font(font_family, font_size)
                if hasattr(self.settings_tab, 'font_combo'):
                    self.settings_tab.font_combo.set_current_font(font_family)
                if hasattr(self.settings_tab, 'font_size_spin'):
                    self.settings_tab.font_size_spin.setValue(font_size)
                if hasattr(self, 'apply_stylesheet'):
                    self.apply_stylesheet()

            # 클리닝 옵션
            cleaning_options = settings.get("cleaning_options", {})
            if cleaning_options:
                self.settings_tab.chk_auto_comma.setChecked(cleaning_options.get("auto_comma", True))
                self.settings_tab.chk_auto_space.setChecked(cleaning_options.get("auto_space", True))
                self.settings_tab.chk_auto_escape.setChecked(cleaning_options.get("auto_escape", False))
                self.settings_tab.chk_remove_duplicates.setChecked(cleaning_options.get("remove_duplicates", False))
                self.settings_tab.chk_underscore_to_space.setChecked(cleaning_options.get("underscore_to_space", True))
                self.prompt_cleaner.set_options(**cleaning_options)
            
            # 에디터 기본값
            ed = settings.get("editor_defaults", {})
            if ed:
                self.settings_tab.spin_def_tool_size.setValue(ed.get("tool_size", 20))
                self.settings_tab.spin_def_effect_strength.setValue(ed.get("effect_strength", 15))
                self.settings_tab.spin_def_bar_w.setValue(ed.get("bar_w", 50))
                self.settings_tab.spin_def_bar_h.setValue(ed.get("bar_h", 20))
                self.settings_tab.spin_def_detect_conf.setValue(ed.get("detect_conf", 25))
                self.settings_tab.chk_def_magnetic_lasso.setChecked(ed.get("magnetic_lasso", False))
                self.settings_tab.spin_def_snap_radius.setValue(ed.get("snap_radius", 12))
                self.settings_tab.spin_def_canny_low.setValue(ed.get("canny_low", 50))
                self.settings_tab.spin_def_canny_high.setValue(ed.get("canny_high", 150))
                self.settings_tab.spin_def_smooth_factor.setValue(ed.get("smooth_factor", 0.008))
                self.settings_tab.spin_def_rotation_step.setValue(ed.get("rotation_step", 5))
                self.settings_tab.spin_def_undo_limit.setValue(ed.get("undo_limit", 20))

            # 에디터 탭에 기본값 적용
            if hasattr(self, 'mosaic_editor'):
                self.mosaic_editor.apply_defaults(self.settings_tab.get_editor_defaults())

            # 배경 제거 모델 복원
            bg_model = settings.get("bg_removal_model", "u2net")
            if hasattr(self, 'mosaic_editor'):
                combo = self.mosaic_editor.mosaic_panel.bg_model_combo
                for i in range(combo.count()):
                    if combo.itemData(i) == bg_model:
                        combo.setCurrentIndex(i)
                        break

            # 단축키 로드
            shortcuts_data = settings.get("shortcuts", {})
            if shortcuts_data:
                sm = get_shortcut_manager()
                sm.from_dict(shortcuts_data)
                # UI 버튼 갱신
                if hasattr(self.settings_tab, '_key_capture_buttons'):
                    for btn in self.settings_tab._key_capture_buttons.values():
                        btn.refresh()

            # 와일드카드 활성화 상태
            if hasattr(self.settings_tab, 'chk_wildcard_enabled'):
                self.settings_tab.chk_wildcard_enabled.setChecked(
                    settings.get("wildcard_enabled", True)
                )

            # 데이터 경로 복원
            import config as _cfg
            parquet_dir = settings.get("parquet_dir", "")
            if parquet_dir and hasattr(self.settings_tab, 'parquet_dir_input'):
                self.settings_tab.parquet_dir_input.setText(parquet_dir)
                _cfg.PARQUET_DIR = parquet_dir
            event_parquet_dir = settings.get("event_parquet_dir", "")
            if event_parquet_dir and hasattr(self.settings_tab, 'event_parquet_dir_input'):
                self.settings_tab.event_parquet_dir_input.setText(event_parquet_dir)
                _cfg.EVENT_PARQUET_DIR = event_parquet_dir

            # 갤러리 폴더 복원 (경로만 기억, 탭 클릭 시 실제 로드)
            gallery_folder = settings.get("gallery_folder", "")
            if gallery_folder and os.path.isdir(gallery_folder) and hasattr(self, 'gallery_tab'):
                self.gallery_tab._current_folder = gallery_folder
                self.gallery_tab.label_folder.setText(gallery_folder)
                self.gallery_tab.label_folder.setStyleSheet("color: #DDD; font-size: 12px;")

            # I2I 탭 복원
            i2i_s = settings.get("i2i_settings", {})
            if i2i_s and hasattr(self, 'i2i_tab'):
                self._apply_i2i_settings(i2i_s)

            # Inpaint 탭 복원
            inp_s = settings.get("inpaint_settings", {})
            if inp_s and hasattr(self, 'inpaint_tab'):
                self._apply_inpaint_settings(inp_s)

            # Search 탭 복원
            search_s = settings.get("search_criteria", {})
            if search_s and hasattr(self, 'search_tab'):
                self.search_tab._apply_criteria_dict(search_s)

            # 백엔드 설정 복원
            if "backend_type" in settings:
                self._restore_backend_settings(settings)

            self.is_programmatic_change = False

            self.loaded_settings = settings
            self.show_status("✅ 설정을 불러왔습니다.", 3000)
            
        except Exception as e:
            import traceback
            traceback.print_exc()
            from utils.app_logger import get_logger
            get_logger('settings').error(f"설정 불러오기 실패: {e}")
    
    def _get_combined_cond_rules_json(self) -> str:
        """Positive/Negative 에디터의 규칙을 합쳐 JSON으로 반환"""
        from utils.condition_block import rules_to_json
        pos_rules = self.cond_block_editor_pos.get_rules()
        neg_rules = self.cond_block_editor_neg.get_rules()
        return rules_to_json(pos_rules + neg_rules)

    def _get_slot_settings(self, widgets):
        """ADetailer 슬롯 설정 가져오기"""
        return {
            "model": widgets['model'].text(),
            "prompt": widgets['prompt'].toPlainText(),
            "mask_blur": widgets['mask_blur'].text(),
            "denoise": widgets['denoise'].text(),
            "confidence": widgets['confidence'].text(),
            "padding": widgets['padding'].text(),
            
            "use_inpaint_size": widgets['use_inpaint_size_check'].isChecked(),
            "inpaint_width": widgets['inpaint_width'].text(),
            "inpaint_height": widgets['inpaint_height'].text(),
            
            "use_steps": widgets['use_steps_check'].isChecked(),
            "steps": widgets['steps'].text(),
            
            "use_cfg": widgets['use_cfg_check'].isChecked(),
            "cfg": widgets['cfg'].text(),
            
            "use_checkpoint": widgets['use_checkpoint_check'].isChecked(),
            "checkpoint": widgets['checkpoint_combo'].currentText(),
            
            "use_vae": widgets['use_vae_check'].isChecked(),
            "vae": widgets['vae_combo'].currentText(),
            
            "use_sampler": widgets['use_sampler_check'].isChecked(),
            "sampler": widgets['sampler_combo'].currentText(),
            "scheduler": widgets['scheduler_combo'].currentText(),
        }
    
    def _set_slot_settings(self, widgets, settings):
        """ADetailer 슬롯 설정 적용"""
        widgets['model'].setText(settings.get("model", "face_yolov8n.pt"))
        widgets['prompt'].setPlainText(settings.get("prompt", ""))
        widgets['mask_blur'].setText(settings.get("mask_blur", "8"))
        widgets['denoise'].setText(settings.get("denoise", "0.4"))
        widgets['confidence'].setText(settings.get("confidence", "0.3"))
        widgets['padding'].setText(settings.get("padding", "32"))
        
        widgets['use_inpaint_size_check'].setChecked(settings.get("use_inpaint_size", False))
        widgets['inpaint_width'].setText(settings.get("inpaint_width", "1024"))
        widgets['inpaint_height'].setText(settings.get("inpaint_height", "1024"))
        
        widgets['use_steps_check'].setChecked(settings.get("use_steps", False))
        widgets['steps'].setText(settings.get("steps", "32"))
        
        widgets['use_cfg_check'].setChecked(settings.get("use_cfg", False))
        widgets['cfg'].setText(settings.get("cfg", "5.0"))
        
        widgets['use_checkpoint_check'].setChecked(settings.get("use_checkpoint", False))
        checkpoint_text = settings.get("checkpoint", "")
        idx = widgets['checkpoint_combo'].findText(checkpoint_text)
        if idx >= 0: 
            widgets['checkpoint_combo'].setCurrentIndex(idx)
        
        widgets['use_vae_check'].setChecked(settings.get("use_vae", False))
        vae_text = settings.get("vae", "")
        idx = widgets['vae_combo'].findText(vae_text)
        if idx >= 0: 
            widgets['vae_combo'].setCurrentIndex(idx)
        
        widgets['use_sampler_check'].setChecked(settings.get("use_sampler", False))
        sampler_text = settings.get("sampler", "")
        idx = widgets['sampler_combo'].findText(sampler_text)
        if idx >= 0:
            widgets['sampler_combo'].setCurrentIndex(idx)

        scheduler_text = settings.get("scheduler", "")
        idx = widgets['scheduler_combo'].findText(scheduler_text)
        if idx >= 0:
            widgets['scheduler_combo'].setCurrentIndex(idx)

    # ── I2I 탭 설정 ──

    def _get_i2i_settings(self) -> dict:
        """I2I 탭 설정 가져오기"""
        tab = self.i2i_tab
        return {
            "prompt": tab.prompt_text.toPlainText(),
            "negative_prompt": tab.neg_prompt_text.toPlainText(),
            "denoising": tab.denoise_input.text(),
            "resize_mode": tab.resize_combo.currentIndex(),
            "width": tab.width_input.text(),
            "height": tab.height_input.text(),
            "steps": tab.steps_input.text(),
            "cfg": tab.cfg_input.text(),
            "seed": tab.seed_input.text(),
        }

    def _apply_i2i_settings(self, s: dict):
        """I2I 탭 설정 적용"""
        tab = self.i2i_tab
        tab.prompt_text.setPlainText(s.get("prompt", ""))
        tab.neg_prompt_text.setPlainText(s.get("negative_prompt", ""))
        tab.denoise_input.setText(s.get("denoising", "0.75"))
        tab.resize_combo.setCurrentIndex(s.get("resize_mode", 0))
        tab.width_input.setText(s.get("width", "1024"))
        tab.height_input.setText(s.get("height", "1024"))
        tab.steps_input.setText(s.get("steps", "20"))
        tab.cfg_input.setText(s.get("cfg", "7.0"))
        tab.seed_input.setText(s.get("seed", "-1"))

    # ── Inpaint 탭 설정 ──

    def _get_inpaint_settings(self) -> dict:
        """Inpaint 탭 설정 가져오기"""
        tab = self.inpaint_tab
        return {
            "prompt": tab.prompt_text.toPlainText(),
            "negative_prompt": tab.neg_prompt_text.toPlainText(),
            "denoising": tab.denoise_input.text(),
            "fill_mode": tab.fill_combo.currentIndex(),
            "mask_blur": tab.mask_blur_input.text(),
            "full_res": tab.chk_full_res.isChecked(),
            "padding": tab.padding_input.text(),
            "steps": tab.steps_input.text(),
            "cfg": tab.cfg_input.text(),
            "seed": tab.seed_input.text(),
            "brush_size": tab.brush_slider.value(),
        }

    def _apply_inpaint_settings(self, s: dict):
        """Inpaint 탭 설정 적용"""
        tab = self.inpaint_tab
        tab.prompt_text.setPlainText(s.get("prompt", ""))
        tab.neg_prompt_text.setPlainText(s.get("negative_prompt", ""))
        tab.denoise_input.setText(s.get("denoising", "0.75"))
        tab.fill_combo.setCurrentIndex(s.get("fill_mode", 1))
        tab.mask_blur_input.setText(s.get("mask_blur", "4"))
        tab.chk_full_res.setChecked(s.get("full_res", True))
        tab.padding_input.setText(s.get("padding", "32"))
        tab.steps_input.setText(s.get("steps", "20"))
        tab.cfg_input.setText(s.get("cfg", "7.0"))
        tab.seed_input.setText(s.get("seed", "-1"))
        tab.brush_slider.setValue(s.get("brush_size", 30))

    # ── 백엔드 설정 헬퍼 ──

    def _get_backend_type_str(self) -> str:
        """현재 백엔드 타입 문자열"""
        try:
            from backends import get_backend_type
            return get_backend_type().value
        except Exception:
            return "webui"

    def _get_comfyui_url(self) -> str:
        """ComfyUI API URL"""
        import config
        return getattr(config, 'COMFYUI_API_URL', 'http://127.0.0.1:8188')

    def _get_comfyui_workflow_path(self) -> str:
        """ComfyUI 워크플로우 경로"""
        import config
        return getattr(config, 'COMFYUI_WORKFLOW_PATH', '')

    def _restore_backend_settings(self, settings: dict):
        """백엔드 설정 복원"""
        import config
        backend_type_str = settings.get("backend_type", "webui")
        comfyui_url = settings.get("comfyui_url", "http://127.0.0.1:8188")
        workflow_path = settings.get("comfyui_workflow_path", "")

        config.COMFYUI_API_URL = comfyui_url
        config.COMFYUI_WORKFLOW_PATH = workflow_path

        from backends import set_backend, BackendType
        if backend_type_str == "comfyui":
            set_backend(BackendType.COMFYUI, comfyui_url)
        else:
            set_backend(BackendType.WEBUI, config.WEBUI_API_URL)
