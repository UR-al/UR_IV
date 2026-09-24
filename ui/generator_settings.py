# ui/generator_settings.py
"""
설정 저장 및 불러오기 관련 로직
"""
import os
import json
from PyQt6.QtWidgets import QMessageBox
from config import PROMPT_SETTINGS_FILE
from core.prompt_settings_extras import PromptSettingsExtras, extras_of


def migrate_legacy_gallery_folder(
    settings: dict,
    *,
    prompt_settings_path: str = PROMPT_SETTINGS_FILE,
    ui_prefs_path: str | None = None,
) -> str:
    """Move the old prompt-settings gallery path into ui_prefs exactly once.

    The current ui_prefs value always wins.  The legacy key is removed only
    after a valid destination has been read or written successfully.
    """
    from core.config_migration import load_ui_prefs, save_ui_prefs
    from core.ui_prefs import ui_prefs_path as default_ui_prefs_path
    from utils.atomic_json import atomic_write_json

    prefs_path = ui_prefs_path or default_ui_prefs_path()
    prefs = load_ui_prefs(prefs_path)
    current = str(prefs.get('galleryFolder', '') or '').strip()
    legacy = str(settings.get('gallery_folder', '') or '').strip()

    if not current and legacy:
        prefs['galleryFolder'] = legacy
        save_ui_prefs(prefs_path, prefs)
        current = legacy

    if 'gallery_folder' in settings and (current or not legacy):
        settings.pop('gallery_folder', None)
        atomic_write_json(prompt_settings_path, settings, indent=4)
    return current


class SettingsMixin:
    """설정 관련 로직을 담당하는 Mixin"""

    @staticmethod
    def _sync_slider(line_edit):
        """QLineEdit의 텍스트 값으로 연결된 슬라이더 위치 동기화.

        SliderProxy 는 입력칸과 슬라이더가 한 몸(``_slider is self``)이라 맞출 것이 없다 —
        예전엔 여기서 ``int(value*multiplier)`` 로 되써서 cfg 5.3→5.0, hires 디노이즈 0.29→0.28
        처럼 부팅·설정 복원마다 값이 한 칸씩 내려갔다. 별도 슬라이더가 붙은 옛 위젯만 반올림으로
        맞춘다(0.29*100 = 28.999… 을 버리면 28).
        """
        slider = getattr(line_edit, '_slider', None)
        multiplier = getattr(line_edit, '_multiplier', None)
        if slider is None or multiplier is None or slider is line_edit:
            return
        try:
            value = float(line_edit.text())
            slider.setValue(int(round(value * multiplier)))
        except (ValueError, TypeError, AttributeError):
            pass

    def save_settings(self) -> bool:
        """설정 저장 — 성공하면 True (종료 경로가 크래시 복구 백업을 정리할지 판단한다)."""
        # 설정 백업을 막 가져왔으면(재시작 전) 메모리 상태로 가져온 prompt_settings 를 덮지 않는다
        # — 백엔드 연결 전환 등 내부 저장 경로까지 한곳에서 막는다(ui/settings_data_actions.py).
        if getattr(self, '_preserve_imported_settings_on_quit', False):
            return False
        try:
            settings = self._build_settings_dict()
            # 원자적 쓰기 — 비정상 종료/디스크 오류 시 핵심 설정 파일 절단 방지
            from utils.atomic_json import atomic_write_json
            atomic_write_json(PROMPT_SETTINGS_FILE, settings, indent=4)
            self.show_status("✅ 설정이 저장되었습니다.", 3000)
            return True
        except Exception as e:
            from utils.app_logger import get_logger
            get_logger('settings').error(f"설정 저장 실패: {e}")
            QMessageBox.critical(self, "저장 실패", f"설정 저장 중 오류: {e}")
            return False

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
        # 모델·Hires 체크포인트는 연결 오류(on_webui_info_error)가 목록을 비운다 — VAE 처럼
        # preservedText(비우기 직전 선택) → 디스크 값 순. currentText 만 보면 끊긴 채 종료할 때 그 전에
        # 고른 모델 대신 옛 디스크 값이 저장돼 다음 시작이 옛 모델로 돌아갔다.
        model_val = self._persisted_combo_text(self.model_combo) or self._get_existing_setting("model")
        sampler_val = self.sampler_combo.currentText() or self._get_existing_setting("sampler")
        scheduler_val = self.scheduler_combo.currentText() or self._get_existing_setting("scheduler")
        # LoRA 스택은 config/ui_prefs.json(loraStack)이 단일 소스 — prompt_settings에 중복 저장하지 않음.

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
            # VAE 는 연결 오류(on_webui_info_error)가 목록을 비운다 — 비운 뒤 저장해도 선택을 잃지 않게
            # preservedText(비우기 직전 항목) → 디스크 값 순으로 읽는다.
            "vae_main": (self._persisted_combo_text(self.vae_main_combo) or self._get_existing_setting("vae_main"))
                        if hasattr(self, 'vae_main_combo') else "",
            "te_main": self.te_main_input.text() if hasattr(self, 'te_main_input') else "",
            "sampler": sampler_val,
            "scheduler": scheduler_val,
            "steps": self.steps_input.text(),
            "cfg": self.cfg_input.text(),
            "shift": self.shift_input.text(),
            "seed": self.seed_input.text(),
            "width": self.width_input.text(),
            "height": self.height_input.text(),

            "random_res_enabled": self.random_res_check.isChecked(),
            "random_resolutions": self.random_resolutions,

            "hires_enabled": self.hires_options_group.isChecked(),
            "hires_upscaler": self.upscaler_combo.currentText() or self._get_existing_setting("hires_upscaler"),
            "hires_steps": self.hires_steps_input.text(),
            "hires_denoising": self.hires_denoising_input.text(),
            "hires_scale": self.hires_scale_input.text(),
            "hires_cfg": self.hires_cfg_input.text(),
            "hires_checkpoint": (self._persisted_combo_text(self.hires_checkpoint_combo)
                                 or self._get_existing_setting("hires_checkpoint")),
            "hires_sampler": self.hires_sampler_combo.currentText() or self._get_existing_setting("hires_sampler"),
            "hires_scheduler": self.hires_scheduler_combo.currentText() or self._get_existing_setting("hires_scheduler"),
            "hires_prompt": self.hires_prompt_text.toPlainText(),
            "hires_neg_prompt": self.hires_neg_prompt_text.toPlainText(),

            "adetailer_enabled": self.adetailer_group.isChecked(),
            "adetailer_slot1_enabled": self.ad_slot1_group.isChecked(),
            "adetailer_slot2_enabled": self.ad_slot2_group.isChecked(),
            
            "adetailer_slot1": self._get_slot_settings(self.s1_widgets, "adetailer_slot1"),
            "adetailer_slot2": self._get_slot_settings(self.s2_widgets, "adetailer_slot2"),
            "sam3_enabled": self.sam3_group.isChecked() if hasattr(self, 'sam3_group') else False,
            "sam3_settings": self._get_sam3_settings(self.sam3_widgets) if hasattr(self, 'sam3_widgets') else {},
            "anima_guidance_settings": (
                self._get_anima_guidance_settings(self.anima_guidance_widgets)
                if hasattr(self, 'anima_guidance_widgets') else {}
            ),
            
            
            "remove_artist": self.chk_remove_artist.isChecked(),
            "remove_copyright": self.chk_remove_copyright.isChecked(),
            "remove_character": self.chk_remove_character.isChecked(),
            "remove_meta": self.chk_remove_meta.isChecked(),
            "remove_censorship": self.chk_remove_censorship.isChecked(),
            "remove_text": self.chk_remove_text.isChecked(),

            "prompt_focus": self.chk_prompt_focus.isChecked() if hasattr(self, 'chk_prompt_focus') else False,
            "auto_char_features": self.chk_auto_char_features.isChecked() if hasattr(self, 'chk_auto_char_features') else False,
            "auto_remove_char_features": self.chk_auto_remove_char_features.isChecked() if hasattr(self, 'chk_auto_remove_char_features') else False,
            "char_feature_mode": self.combo_char_feature_mode.currentIndex() if hasattr(self, 'combo_char_feature_mode') else 0,
            "char_feature_override": getattr(self, '_char_feature_override', {'hair_length': False, 'eye_color': False}),

            # 전역 조건식의 단일 소스는 config/cond_rules.json (Vue 모달이 저장) — 늘 적용된다.
            #   레거시 cond_block_editor 위젯은 Vue SPA에서 표시되지 않고 앱도 읽지 않으므로
            #   여기서 cond_rules_json을 더 이상 저장하지 않는다(이중 저장 제거).

            "base_prefix_prompt": self.base_prefix_prompt,
            "base_suffix_prompt": self.base_suffix_prompt,
            "base_neg_prompt": self.base_neg_prompt,
            
            # 웹 설정 추가!
            "web_home_url": self.web_tab.home_url if hasattr(self.web_tab, 'home_url') else "",

            # Vue 위젯이 주인이 아닌 키 — wildcard_enabled·cleaning_options(Vue 토글이 없는 두 옵션만;
            # 중복·공백·밑줄 정리는 ui_prefs(clean*)가 주인)·font_family·font_size. 한 곳:
            # core/prompt_settings_extras. theme·parquet_dir 같은 은퇴한 키는 쓰지 않는다(RETIRED_KEYS).
            **extras_of(self).to_settings(),

            # 백엔드 설정
            "backend_type": self._get_backend_type_str(),
            "webui_url": self._get_webui_url(),
            "comfyui_url": self._get_comfyui_url(),
            "comfyui_workflow_path": self._get_comfyui_workflow_path(),
            "comfyui_workflow_img2img_path": self._get_comfyui_workflow_img2img_path(),
            # 저장한 모델을 고른 때의 워크플로(_model_workflow_path) — comfyui_workflow_path 는 방금 고른
            # 워크플로라, 연결 전에 저장(게이트)하거나 끊긴 채 종료하면 둘이 갈라진다. 모르면 null.
            "comfyui_model_workflow_path": self._model_workflow_baseline(),

            # I2I 탭 설정
            "i2i_settings": self._get_i2i_settings() if hasattr(self, 'i2i_tab') else {},

            # Inpaint 탭 설정
            "inpaint_settings": self._get_inpaint_settings() if hasattr(self, 'inpaint_tab') else {},

        }
        return settings

    def _build_preset_settings(self) -> dict:
        """생성 프리셋 저장용(save_preset_by_name) — :meth:`_build_settings_dict` 에서 선행/후행/네거티브만
        칸의 글 대신 ``base_*`` 템플릿(ui/generation_settings_apply.preset_prompt_templates).

        prompt_settings.json 은 칸의 글과 템플릿을 둘 다 저장·복원하지만 프리셋은 칸 키만 가진다 —
        사이클 뒤의 해석된 와일드카드·조건부 태그가 프리셋 불러오기로 템플릿에 굳지 않게 한다.
        """
        from ui.generation_settings_apply import preset_prompt_templates
        return preset_prompt_templates(self, self._build_settings_dict())

    def load_settings(self):
        """설정 불러오기"""
        if not os.path.exists(PROMPT_SETTINGS_FILE):
            return
        
        try:
            with open(PROMPT_SETTINGS_FILE, 'r', encoding='utf-8') as f:
                settings = json.load(f)
            
            self.is_programmatic_change = True
            
            # 프롬프트 칸·생성 파라미터·Hires·ADetailer·SAM3·Anima — 프리셋 불러오기와 같은 적용
            # 함수(ui/generation_settings_apply.py)를 쓴다. 여기선 없는 키를 기본값으로 채운다.
            from ui.generation_settings_apply import apply_generation_settings, apply_prompt_settings
            apply_prompt_settings(self, settings, only_present=False)
            self.btn_lock_artist.setChecked(settings.get("artist_locked", False))
            apply_generation_settings(self, settings, only_present=False)

            # LoRA 활성 목록: config/ui_prefs.json(loraStack) 단일 소스로 이관됨
            #   (복원은 generator_main._restore_runtime_prefs + App.vue uiPrefsLoaded).
            #   레거시 prompt_settings.active_loras는 여기서 더 이상 읽지 않는다.

            # NegPiP 은 상시 적용(b8ef7901a) — 옛 prompt_settings/백업의 negpip_enabled=false 로
            # 조용히 꺼지지 않게 저장값을 읽지 않고 켠다(audit #97).
            if hasattr(self, 'negpip_group'):
                self.negpip_group.setChecked(True)

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
            if hasattr(self, 'chk_prompt_focus'):
                self.chk_prompt_focus.setChecked(settings.get("prompt_focus", False))
            if hasattr(self, 'chk_auto_char_features'):
                self.chk_auto_char_features.setChecked(settings.get("auto_char_features", False))
            if hasattr(self, 'chk_auto_remove_char_features'):
                self.chk_auto_remove_char_features.setChecked(settings.get("auto_remove_char_features", False))
            if hasattr(self, 'combo_char_feature_mode'):
                self.combo_char_feature_mode.setCurrentIndex(settings.get("char_feature_mode", 0))
            _ov = settings.get("char_feature_override") or {}
            self._char_feature_override = {
                'hair_length': bool(_ov.get('hair_length', False)),
                'eye_color': bool(_ov.get('eye_color', False)),
            }

            # 전역 조건식: config/cond_rules.json 단일 소스로 이관됨(늘 적용 — 옛 cond_prompt_enabled 는 읽지 않는다)
            #   (마이그레이션은 generator_main._apply_saved_configs, Vue 전달은 getInitialConfig pull).
            #   레거시 prompt_settings.cond_rules_json은 더 이상 cond_block_editor에 로드하지 않는다.

            # 베이스 프롬프트 — 키가 없는 옛 설정 파일은 칸의 글을 템플릿으로(그 전엔 ''라, 다음 사이클이
            # 칸을 비우고 프리셋 저장(_build_preset_settings)도 빈 선행/네거티브를 썼다).
            self.base_prefix_prompt = settings.get("base_prefix_prompt", settings.get("prefix_prompt", ""))
            self.base_suffix_prompt = settings.get("base_suffix_prompt", settings.get("suffix_prompt", ""))
            self.base_neg_prompt = settings.get("base_neg_prompt", settings.get("negative_prompt", ""))

            # Vue 위젯이 주인이 아닌 키(와일드카드 ON/OFF·레거시 클리너 두 옵션·글꼴) — 예전엔 숨은
            #   SettingsTab 위젯을 거쳤다(audit #178). theme·parquet_dir·cond_prompt_enabled 등 은퇴한 키는
            #   읽지 않는다(core/prompt_settings_extras.RETIRED_KEYS) — 데이터셋 경로는 config.PARQUET_DIR
            #   (= core.fetch_data.DATA_DIR) 한 곳이라, 옮긴 설치본·다른 설치본의 백업이 옛 경로를 되살리지 않는다.
            self._apply_prompt_settings_extras(PromptSettingsExtras.from_settings(settings))

            # (editor_defaults·bg_removal_model·shortcuts 는 숨은 PyQt 에디터(MosaicEditor)·설정 탭 전용이라
            #  은퇴와 함께 저장·복원을 뺐다. Vue 에디터 기본값은 tab_defaults.json, 배경 제거는 u2net 고정,
            #  Vue 단축키는 프론트가 정한다 — 옛 prompt_settings.json 의 세 키는 무시된다
            #  (core/prompt_settings_extras.RETIRED_KEYS, tests/test_legacy_editor_retirement.py).

            # 옛 중복 키를 ui_prefs 단일 소스로 1회 흡수. 현재 ui_prefs가 항상 우선.
            # (갤러리 폴더는 Vue 갤러리가 ui_prefs 에서 읽는다 — 숨은 PyQt 갤러리 탭은 은퇴했다)
            migrate_legacy_gallery_folder(settings)

            # I2I 탭 복원
            i2i_s = settings.get("i2i_settings", {})
            if i2i_s and hasattr(self, 'i2i_tab'):
                self._apply_i2i_settings(i2i_s)

            # Inpaint 탭 복원
            inp_s = settings.get("inpaint_settings", {})
            if inp_s and hasattr(self, 'inpaint_tab'):
                self._apply_inpaint_settings(inp_s)

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
            # 예전엔 로그에만 남아, 기본값으로 뜬 이유를 사용자가 알 수 없었다.
            self.show_status(f"❌ 설정 불러오기 실패: {e}", 0)
            from ui.status_line import notify_user
            notify_user(self, 'error', f'설정을 불러오지 못해 일부 값이 기본값입니다: {e}')

    def _apply_prompt_settings_extras(self, extras: PromptSettingsExtras) -> None:
        """불러온 Vue-위젯-없는 키를 적용한다(core/prompt_settings_extras).

        - 보관함을 바꾼다 — 생성 경로(utils.file_wildcard.wildcards_enabled)와 다음 저장이 이것을 읽는다.
        - 글꼴 → ThemeManager(남은 PyQt 창·네이티브 대화상자 QSS).
        - 클리너는 Vue 토글이 없는 두 옵션(auto_comma/auto_escape)만. 중복·공백·밑줄 정리는 ui_prefs(clean*)가
          주인이라 _apply_ui_prefs_to_cleaner 가 맡는다(예전 set_options(**cleaning_options) 는 Vue 에서
          고른 값을 덮었다 — 감사 #29).
        - Vue 와일드카드 토글(CheckBoxProxy 'wildcard_enabled')을 값에 맞춘다.
        """
        self.prompt_settings_extras = extras
        from utils.theme_manager import get_theme_manager
        get_theme_manager().set_font(extras.font_family, extras.font_size)
        self.prompt_cleaner.set_options(**extras.cleaning)
        proxy = getattr(self, 'wildcard_enabled_check', None)
        if proxy is not None:
            proxy.setChecked(extras.wildcard_enabled)

    @staticmethod
    def _persisted_combo_text(combo) -> str:
        """저장용 콤보 값 — 목록이 비어 있어도(연결 전·연결 오류로 비움) 되살릴 선택을 읽는다.

        ComboBoxProxy.preservedText(지금 항목 → 아직 못 고른 설정값 → 비우기 직전 항목)는 연결 때
        ui/combo_restore 가 읽는 값과 같다. **저장 경로 전용** — 생성 요청은 currentText 를 그대로
        쓴다(비운 콤보의 옛 이름이 다른 백엔드 요청으로 새면 안 된다).
        """
        reader = getattr(combo, 'preservedText', None)
        if not callable(reader):
            reader = combo.currentText
        return str(reader() or '')

    def _get_slot_settings(self, widgets, slot_key: str | None = None):
        """ADetailer 슬롯 설정 가져오기.

        체크포인트·VAE·샘플러·스케줄러는 백엔드 목록으로 채우는 콤보다. 연결 전(목록 도착 전)이나
        연결 오류가 비운 뒤 저장해도 값을 잃지 않게 :meth:`_persisted_combo_text` 로 읽고, 그래도
        비었으면 ``slot_key``(prompt_settings.json 의 'adetailer_slot1' 등)의 디스크 값을 유지한다 —
        메인 콤보의 ``currentText() or _get_existing_setting`` 가드와 같은 목적. 예전엔 시작 게이트가
        연결 전에 저장하면서 슬롯 값이 ''로 덮여, 연결 뒤 'Use same checkpoint' 등 첫 항목으로 돌아갔다.
        """
        existing: dict | None = None

        def backend_combo(combo_key: str, setting_key: str) -> str:
            nonlocal existing
            value = self._persisted_combo_text(widgets[combo_key])
            if value or not slot_key:
                return value
            if existing is None:   # 디스크는 슬롯마다 한 번만
                saved = self._get_existing_setting(slot_key, {})
                existing = saved if isinstance(saved, dict) else {}
            return str(existing.get(setting_key) or '')

        def optional(key: str, read) -> str:
            widget = widgets.get(key) if hasattr(widgets, 'get') else None
            return str(read(widget) or '') if widget is not None else ''

        return {
            "model": widgets['model'].text(),
            "prompt": widgets['prompt'].toPlainText(),
            # 생성 페이로드(_build_adetailer_slot)가 읽는 ad_negative_prompt·ad_dilate_erode·
            # ad_mask_merge_invert — 예전엔 저장·복원하지 않아 재시작·프리셋마다 초기화됐다.
            "neg_prompt": optional('neg_prompt', lambda w: w.toPlainText()),
            "dilate_erode": optional('dilate_erode', lambda w: w.text()),
            "mask_merge_invert": optional('mask_merge_invert', lambda w: w.text()),
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
            "checkpoint": backend_combo('checkpoint_combo', 'checkpoint'),

            "use_vae": widgets['use_vae_check'].isChecked(),
            "vae": backend_combo('vae_combo', 'vae'),

            "use_sampler": widgets['use_sampler_check'].isChecked(),
            "sampler": backend_combo('sampler_combo', 'sampler'),
            "scheduler": backend_combo('scheduler_combo', 'scheduler'),
        }

    def _get_sam3_settings(self, widgets):
        """SAM3 설정 가져오기 — Forge 확장 args.py와 1:1"""
        return {
            "detect_prompt": widgets['detect_prompt'].toPlainText(),
            "exclude_prompt": widgets['exclude_prompt'].toPlainText(),
            "inpaint_prompt": widgets['inpaint_prompt'].toPlainText(),
            "neg_prompt": widgets['neg_prompt'].toPlainText(),
            "mode": widgets['mode'].currentText(),
            "mask_mode": widgets['mask_mode'].currentText(),
            "threshold": widgets['threshold'].text(),
            "mask_dilation": widgets['mask_dilation'].text(),
            "mask_hull": widgets['mask_hull'].isChecked(),
            "mask_outline_px": widgets['mask_outline_px'].text(),
            "mask_blur": widgets['mask_blur'].text(),
            "denoise": widgets['denoise'].text(),
            "padding": widgets['padding'].text(),
            "checkpoint": widgets['checkpoint'].text(),
            "device": widgets['device'].currentText(),
            "inpainting_fill": widgets['inpainting_fill'].currentText(),
            "inpaint_only_masked": widgets['inpaint_only_masked'].isChecked(),
            "preview_overlay": widgets['preview_overlay'].isChecked(),
            "save_artifacts": widgets['save_artifacts'].isChecked(),
            "unload_after": widgets['unload_after'].isChecked(),
            "use_inpaint_size": widgets['use_inpaint_size_check'].isChecked(),
            "inpaint_width": widgets['inpaint_width'].text(),
            "inpaint_height": widgets['inpaint_height'].text(),
            "use_steps": widgets['use_steps_check'].isChecked(),
            "steps": widgets['steps'].text(),
            "use_cfg": widgets['use_cfg_check'].isChecked(),
            "cfg": widgets['cfg'].text(),
            "use_sampler": widgets['use_sampler_check'].isChecked(),
            "sampler": widgets['sampler'].currentText(),
            "use_scheduler": widgets['use_scheduler_check'].isChecked(),
            "scheduler": widgets['scheduler'].currentText(),
            "use_seed": widgets['use_seed_check'].isChecked(),
            "seed": widgets['seed'].text(),
            "use_noise_multiplier": widgets['use_noise_multiplier_check'].isChecked(),
            "noise_multiplier": widgets['noise_multiplier'].text(),
            "restore_face": widgets['restore_face'].isChecked(),
            # ControlNet 13필드 (cn_enable … cn_threshold_b) — core/sam3_controlnet 표 한 벌
            **self._get_sam3_cn_settings(widgets),
        }

    @staticmethod
    def _get_sam3_cn_settings(widgets) -> dict:
        from core.sam3_controlnet import read_settings
        return read_settings(widgets)

    def _get_anima_guidance_settings(self, widgets):
        """62/7/13 위치 계약의 원본 문자열 값을 key 기반 dict로 저장한다."""
        return {
            key: proxy.text()
            for key, proxy in widgets.items()
            if hasattr(proxy, 'text')
        }

    def _set_anima_guidance_settings(self, widgets, settings):
        """저장된 ANIMA 값만 복원하고 새 확장 필드는 코어 기본값으로 보완한다."""
        if not isinstance(settings, dict):
            return
        from core.anima_guidance import default_settings
        defaults = default_settings()
        for key, proxy in widgets.items():
            if not hasattr(proxy, 'setText'):
                continue
            value = settings.get(key, defaults.get(key, ''))
            if isinstance(value, bool):
                value = 'true' if value else 'false'
            elif value is None:
                value = ''
            else:
                value = str(value)
            proxy.setText(value)

    def _reset_anima_guidance(self) -> int:
        """ANIMA 가이던스 82칸을 확장 기본값으로 — Vue '전체 초기화'(reset_anima_guidance 액션).

        기본값의 단일 출처는 core/anima_guidance.SPECS(default_settings)다. 예전엔 Vue 패널이
        82개 리터럴 사본(DEFAULTS)을 들고 있어, 스펙 기본값이 바뀌거나 키가 늘면 초기화만
        옛 값을 쓰거나 새 키를 빠뜨렸다. 값은 배치 한 번으로 Vue 에 간다. 반환: 대상 칸 수.
        """
        widgets = getattr(self, 'anima_guidance_widgets', None) or {}
        if not widgets:
            return 0
        bridge = getattr(self, 'vue_bridge', None)
        begin = getattr(bridge, 'beginBatchUpdate', None)
        end = getattr(bridge, 'endBatchUpdate', None)
        if callable(begin):
            begin()
        try:
            self._set_anima_guidance_settings(widgets, {})
        finally:
            if callable(end):
                end()
        notify = getattr(bridge, 'showNotification', None)
        if notify is not None:
            try:
                notify.emit('success', 'ANIMA 가이던스를 확장 기본값으로 되돌렸습니다')
            except Exception:
                pass
        return len(widgets)

    def _set_slot_settings(self, widgets, settings):
        """ADetailer 슬롯 설정 적용 — 못 고른 콤보 값의 경고 문구 목록을 돌려준다.

        콤보는 ui/generation_settings_apply.apply_combo_value 규칙: 목록이 찼으면 같은 항목(체크포인트·
        VAE 는 해시·경로 무시, 샘플러/스케줄러는 Forge↔ComfyUI 별칭)을 고르고 없으면 현재 선택 유지.
        목록이 비어 있으면(시작 시 load_settings·연결 전 프리셋) 다섯 콤보 모두 setText fallback 으로
        미룬다 — 메인 VAE·샘플러·스케줄러와 같은 규칙. 그래야 연결 전 저장(시작 게이트·종료)이 그 값을
        currentText 로 읽어 지키고, 연결 때 snapshot_backend_combos → restore_backend_combos 가 같은
        matcher 로 다시 고른다. 예전엔 체크포인트·VAE·샘플러·스케줄러를 미루지 않아 값이 어디에도 남지
        않았고, 연결 전 저장이 ''를 써서 연결 뒤 첫 항목('Use same checkpoint' 등)으로 돌아갔다.
        """
        from ui.generation_settings_apply import (
            apply_combo_value, match_checkpoint_index, match_plain_index,
            match_sampler_index, match_scheduler_index,
        )

        misses: list[str] = []
        apply_combo_value(widgets['model'], str(settings.get("model", "face_yolov8n.pt") or ''), '검출 모델',
                          match_plain_index, misses, defer_when_empty=True)
        widgets['prompt'].setPlainText(settings.get("prompt", ""))
        # 네거티브·Dilate/Erode·마스크 병합 — 이 키가 없는 옛 설정·프리셋은 지금 값을 건드리지 않는다
        # (예전엔 저장하지 않던 값이라, 기본값으로 덮으면 옛 프리셋을 불러올 때 입력해 둔 값이 지워진다).
        # 마스크 병합 콤보는 선택지가 Vue 에만 있어 setText(fallback)로 둔다 — SAM3 mode 등과 같다.
        for key, setter in (('neg_prompt', 'setPlainText'), ('dilate_erode', 'setText'),
                            ('mask_merge_invert', 'setText')):
            widget = widgets.get(key) if hasattr(widgets, 'get') else None
            if key in settings and widget is not None:
                value = settings.get(key)
                getattr(widget, setter)('' if value is None else str(value))
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
        apply_combo_value(widgets['checkpoint_combo'], str(settings.get("checkpoint", "") or ''), '체크포인트',
                          match_checkpoint_index, misses, defer_when_empty=True)

        widgets['use_vae_check'].setChecked(settings.get("use_vae", False))
        apply_combo_value(widgets['vae_combo'], str(settings.get("vae", "") or ''), 'VAE',
                          match_checkpoint_index, misses, defer_when_empty=True)

        widgets['use_sampler_check'].setChecked(settings.get("use_sampler", False))
        apply_combo_value(widgets['sampler_combo'], str(settings.get("sampler", "") or ''), '샘플러',
                          match_sampler_index, misses, defer_when_empty=True)
        apply_combo_value(widgets['scheduler_combo'], str(settings.get("scheduler", "") or ''), '스케줄러',
                          match_scheduler_index, misses, defer_when_empty=True)
        return misses

    def _set_sam3_settings(self, widgets, settings):
        """SAM3 설정 적용 — 못 고른 콤보 값의 경고 문구 목록을 돌려준다.

        Python 이 선택지를 채우지 않는 콤보(mode·device·sampler 등 — 선택지는 Vue 에 있다)는 setText
        (fallback) 그대로다. 디스크 목록으로 채우는 체크포인트만 apply_combo_value 규칙 — 목록이 찼는데
        없는 파일이면 현재 선택 유지(예전 setText 는 Vue 에만 프리셋 값을 보내 화면과 요청이 갈라졌다).
        """
        from ui.generation_settings_apply import apply_combo_value, match_checkpoint_index

        misses: list[str] = []
        widgets['detect_prompt'].setPlainText(settings.get("detect_prompt", "face"))
        widgets['exclude_prompt'].setPlainText(settings.get("exclude_prompt", ""))
        widgets['inpaint_prompt'].setPlainText(settings.get("inpaint_prompt", ""))
        widgets['neg_prompt'].setPlainText(settings.get("neg_prompt", ""))

        widgets['mode'].setText(settings.get("mode", "Inpaint"))
        widgets['mask_mode'].setText(settings.get("mask_mode", "Individual"))

        widgets['threshold'].setText(settings.get("threshold", "0.40"))
        widgets['mask_dilation'].setText(settings.get("mask_dilation", "0"))
        widgets['mask_hull'].setChecked(settings.get("mask_hull", False))
        widgets['mask_outline_px'].setText(settings.get("mask_outline_px", "0"))
        widgets['mask_blur'].setText(settings.get("mask_blur", "4"))
        widgets['denoise'].setText(settings.get("denoise", "0.40"))
        widgets['padding'].setText(settings.get("padding", "32"))
        apply_combo_value(widgets['checkpoint'], str(settings.get("checkpoint", "sam3.pt") or ''), '체크포인트',
                          match_checkpoint_index, misses, defer_when_empty=True)
        widgets['device'].setText(settings.get("device", "cuda"))
        widgets['inpainting_fill'].setText(settings.get("inpainting_fill", "original"))
        widgets['inpaint_only_masked'].setChecked(settings.get("inpaint_only_masked", True))
        widgets['preview_overlay'].setChecked(settings.get("preview_overlay", False))
        widgets['save_artifacts'].setChecked(settings.get("save_artifacts", True))
        widgets['unload_after'].setChecked(settings.get("unload_after", True))
        widgets['use_inpaint_size_check'].setChecked(settings.get("use_inpaint_size", False))
        widgets['inpaint_width'].setText(settings.get("inpaint_width", "1024"))
        widgets['inpaint_height'].setText(settings.get("inpaint_height", "1024"))
        widgets['use_steps_check'].setChecked(settings.get("use_steps", False))
        widgets['steps'].setText(settings.get("steps", "28"))
        widgets['use_cfg_check'].setChecked(settings.get("use_cfg", False))
        widgets['cfg'].setText(settings.get("cfg", "7.0"))
        widgets['use_sampler_check'].setChecked(settings.get("use_sampler", False))
        widgets['sampler'].setText(settings.get("sampler", "Use same sampler"))
        widgets['use_scheduler_check'].setChecked(settings.get("use_scheduler", False))
        widgets['scheduler'].setText(settings.get("scheduler", "Use same scheduler"))
        widgets['use_seed_check'].setChecked(settings.get("use_seed", False))
        widgets['seed'].setText(settings.get("seed", "-1"))
        widgets['use_noise_multiplier_check'].setChecked(settings.get("use_noise_multiplier", False))
        widgets['noise_multiplier'].setText(settings.get("noise_multiplier", "1.0"))
        widgets['restore_face'].setChecked(settings.get("restore_face", False))
        # ControlNet 13필드 — 예전 저장 파일엔 키가 없으니 확장 기본값으로 채운다.
        from core.sam3_controlnet import apply_settings as _apply_sam3_cn
        _apply_sam3_cn(widgets, settings)
        return misses

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

    def _get_webui_url(self) -> str:
        """WebUI/Forge API URL"""
        import config
        return getattr(config, 'WEBUI_API_URL', 'http://127.0.0.1:7860')

    def _get_comfyui_workflow_path(self) -> str:
        """ComfyUI 워크플로우 경로"""
        import config
        return getattr(config, 'COMFYUI_WORKFLOW_PATH', '')

    def _get_comfyui_workflow_img2img_path(self) -> str:
        """ComfyUI img2img 워크플로우 경로"""
        import config
        return getattr(config, 'COMFYUI_WORKFLOW_IMG2IMG_PATH', '')

    def _model_workflow_baseline(self) -> str | None:
        """저장할 '모델을 고른 때의 워크플로' — load_settings(_restore_backend_settings)나 지난 연결
        (_auto_select_workflow_model)이 남긴 값. 아직 모르면 None(→ 다음 시작은 예전처럼
        comfyui_workflow_path 를 기준으로 삼는다)."""
        value = getattr(self, '_model_workflow_path', None)
        return value if isinstance(value, str) else None

    def _restore_backend_settings(self, settings: dict):
        """백엔드 설정 복원"""
        import config
        backend_type_str = settings.get("backend_type", "webui")
        webui_url = settings.get(
            "webui_url",
            getattr(config, "WEBUI_API_URL", "http://127.0.0.1:7860"),
        )
        comfyui_url = settings.get(
            "comfyui_url",
            getattr(config, "COMFYUI_API_URL", "http://127.0.0.1:8188"),
        )
        workflow_path = settings.get(
            "comfyui_workflow_path",
            getattr(config, "COMFYUI_WORKFLOW_PATH", ""),
        )
        img2img_workflow_path = settings.get(
            "comfyui_workflow_img2img_path",
            getattr(config, "COMFYUI_WORKFLOW_IMG2IMG_PATH", ""),
        )

        config.WEBUI_API_URL = webui_url
        config.COMFYUI_API_URL = comfyui_url
        config.COMFYUI_WORKFLOW_PATH = workflow_path
        config.COMFYUI_WORKFLOW_IMG2IMG_PATH = img2img_workflow_path
        # 저장된 모델을 고른 때의 워크플로 — 연결 때 _auto_select_workflow_model 이 이 경로와 달라졌을
        # 때만(게이트·설정에서 새 워크플로를 고름) 워크플로 모델로 바꾼다. 따로 저장한 값
        # (comfyui_model_workflow_path)이 있으면 그것 — 게이트가 새 워크플로를 연결 전에 저장한 뒤 연결이
        # 실패했거나 끊긴 채 워크플로를 바꾸고 종료했으면 comfyui_workflow_path 는 이미 새 워크플로라,
        # 그걸 기준으로 삼으면 옛 모델이 '같은 워크플로의 선택'으로 남아 새 워크플로 로더에 써졌다.
        # 옛 설정 파일(키 없음·null)은 예전처럼 저장된 워크플로 경로.
        saved_baseline = settings.get("comfyui_model_workflow_path")
        self._model_workflow_path = saved_baseline if isinstance(saved_baseline, str) else workflow_path

        from backends import BackendType, is_active_backend, set_backend
        if backend_type_str == "comfyui":
            backend_type, api_url = BackendType.COMFYUI, comfyui_url
        else:
            backend_type, api_url = BackendType.WEBUI, webui_url
        # 같은 타입·같은 URL 어댑터가 이미 있으면 다시 만들지 않는다. set_backend 는 매번 새
        # 인스턴스를 만들고 BACKEND_URL_CHANGED 를 발행해 XYZ 축 설정을 지우고 Comfy 어댑터의
        # 생성 문맥·preflight 표시를 초기화했다(감사 #29). 명시적 재연결은 set_backend 를 직접 부른다.
        if is_active_backend(backend_type, api_url):
            return
        set_backend(backend_type, api_url)
