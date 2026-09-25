# ui/generator_ui_setup.py
"""
GeneratorMainUI의 UI 구성 부분 (전체)
"""
from enum import Enum, auto
from pathlib import Path

from PyQt6.QtWidgets import QMenu, QMessageBox
from core.url_safety import is_safe_external_url
from utils.theme_manager import get_color


class _VueNavigationDecision(Enum):
    """Native QWebChannel이 붙은 페이지의 탐색 결정."""

    ALLOW = auto()
    OPEN_EXTERNALLY = auto()
    BLOCK = auto()


class _VueNavigationPolicy:
    """Vue 메인 문서를 정확한 로컬 빌드 진입점으로 고정한다.

    CSS/JS/이미지 등의 하위 리소스 로드는 navigation이 아니므로 영향을
    받지 않는다. 대신 iframe 같은 다른 문서는 로컬 파일이라도 차단해,
    QWebChannel이 실제 SPA 문서 외의 origin에 노출되지 않게 한다.
    """

    def __init__(self, frontend_index: str | Path):
        self._frontend_index = Path(frontend_index).resolve(strict=False)

    def decide(self, url, *, is_main_frame: bool) -> _VueNavigationDecision:
        if not is_main_frame:
            return _VueNavigationDecision.BLOCK

        try:
            if url.isLocalFile():
                requested_path = Path(url.toLocalFile()).resolve(strict=False)
                if requested_path == self._frontend_index:
                    return _VueNavigationDecision.ALLOW
                return _VueNavigationDecision.BLOCK

            # 외부로 넘기는 기준은 open_url 액션과 같은 core.url_safety — 호스트 있는
            # http/https 와 mailto 만. 퍼센트 인코딩한 표기로 검사한다(공백 등 포함 URL).
            encoded = bytes(url.toEncoded()).decode("ascii", errors="replace")
            if url.isValid() and is_safe_external_url(encoded, allow_mailto=True):
                return _VueNavigationDecision.OPEN_EXTERNALLY
        except (OSError, RuntimeError, TypeError, ValueError):
            # 잘못된 URL/파일 경로는 fail closed.
            pass

        return _VueNavigationDecision.BLOCK


class _VueAutomationSettings:
    """자동화 설정 읽기 창구 — Vue 자동화 패널이 보낸 값(``_vue_automation_settings``)을 정규화해 준다."""

    def __init__(self, read):
        self._read = read

    def get_settings(self) -> dict:
        return self._read()


class UISetupMixin:
    """UI 구성을 담당하는 Mixin 클래스"""
    
    def _setup_ui(self):
        """UI: QWebEngineView(Vue SPA) 하나만 전체 화면"""
        self.setWindowTitle("AI Studio - Pro")
        self.setGeometry(100, 100, 1600, 950)

        # ── QWebEngineView 생성 (Vue SPA) ──
        from ui.vue_bridge import VueBridge
        from PyQt6.QtWebEngineWidgets import QWebEngineView
        from PyQt6.QtWebEngineCore import QWebEngineSettings, QWebEngineScript, QWebEnginePage
        from PyQt6.QtWebChannel import QWebChannel
        from PyQt6.QtCore import QUrl
        import os

        app_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        frontend_path = os.path.join(app_root, 'frontend_dist', 'index.html')
        navigation_policy = _VueNavigationPolicy(frontend_path)

        self.vue_bridge = VueBridge(self)
        from core.studio_application import CallContext, StudioApplication
        from ui.studio_qwebchannel import DesktopNativeHost, StudioQWebChannelAdapter

        # 새 transport-neutral Interface는 기존 VueBridge와 나란히 제공한다.
        # legacy 프론트는 계속 ``backend``를 사용하고, 이행된 호출부만
        # ``studio``의 단일 invoke/event 계약을 사용한다.
        self.studio_native_host = DesktopNativeHost(self, self.vue_bridge)
        self.studio_application = StudioApplication(host=self.studio_native_host)
        # 데스크톱 native 권한 컨텍스트 — Vue 어댑터와 Python 내부 호출(시작 자동기동,
        # generator_webui._try_managed_backend_autostart)이 같은 Studio 경로를 쓴다.
        self.studio_native_context = CallContext(
            principal_id="desktop-ui",
            transport="qwebchannel",
            capabilities=frozenset({"native"}),
        )
        self.studio_transport = StudioQWebChannelAdapter(
            self.studio_application,
            self.studio_native_context,
            self,
        )

        from PyQt6.QtGui import QDesktopServices
        from ui.native_dialogs import ThemedWebDialogs, apply_native_shell_theme

        class _ExternalNavigationPage(QWebEnginePage):
            """target=_blank/window.open을 메인 페이지와 채널에서 격리한다."""

            def acceptNavigationRequest(self, url, navigation_type, is_main_frame):
                decision = navigation_policy.decide(url, is_main_frame=is_main_frame)
                if decision is _VueNavigationDecision.OPEN_EXTERNALLY:
                    QDesktopServices.openUrl(url)
                # 팝업용 WebEngine 문서는 절대 로드하지 않는다.
                return False

        class _DebugPage(ThemedWebDialogs, QWebEnginePage):
            def javaScriptConsoleMessage(self, level, message, line, source):
                print(f"[Vue] {message}")

            def acceptNavigationRequest(self, url, navigation_type, is_main_frame):
                decision = navigation_policy.decide(url, is_main_frame=is_main_frame)
                if decision is _VueNavigationDecision.OPEN_EXTERNALLY:
                    QDesktopServices.openUrl(url)
                    return False
                return decision is _VueNavigationDecision.ALLOW

            def createWindow(self, window_type):
                # 새 창에 native channel을 상속시키지 않고, 외부 URL만 OS에 위임한다.
                return _ExternalNavigationPage(self.profile(), self)

        from PyQt6.QtWebEngineCore import QWebEngineProfile
        
        # 캐시 및 데이터 경로 — 고정 경로(PID 미포함)로 localStorage 영속화.
        # 이전엔 tempdir/AIStudioPro_{pid} 라 재시작마다 새 빈 폴더 → localStorage
        # (UI 크기/탭 순서/갤러리/Ollama/고해상도/에디터 설정 등)가 매번 초기화되고
        # temp에 옛 PID 폴더가 누적됐음. 단일 인스턴스 전제로 repo/web_profile 고정 →
        # 모든 UI 설정이 재시작 후에도 유지. (2개 동시 실행 시에만 프로파일 락 위험)
        base_cache_path = os.path.join(
            app_root, 'web_profile')
        os.makedirs(base_cache_path, exist_ok=True)
        
        # 독립적인 프로필 생성 (defaultProfile 대신 사용)
        self.web_profile = QWebEngineProfile("AIStudioProfile", self)
        self.web_profile.setPersistentStoragePath(os.path.join(base_cache_path, "Storage"))
        self.web_profile.setCachePath(os.path.join(base_cache_path, "Cache"))
        self.web_profile.setPersistentCookiesPolicy(QWebEngineProfile.PersistentCookiesPolicy.AllowPersistentCookies)
        # 갤러리·즐겨찾기 카드 썸네일(aithumb:) — 원본 PNG 를 카드마다 풀해상도로 디코드하지 않게.
        # 스킴은 new_main_ui 가 QApplication 전에 등록한다. 등록 안 된 실행 경로면 None(원본 폴백).
        # 캐시 폴더는 config.THUMB_DIR 한 곳(히스토리 generateThumbnails 와 같은 캐시). 옛 폴더
        # (LEGACY_THUMB_DIR)의 같은 키는 렌더 전에 옮겨 온다 — 배경 정리(시작 30초 뒤)를 기다리지 않게.
        from config import LEGACY_THUMB_DIR, THUMB_DIR
        from ui.thumb_scheme import install_thumb_scheme_handler
        self._thumb_scheme_handler = install_thumb_scheme_handler(
            self.web_profile, THUMB_DIR, legacy_dir=LEGACY_THUMB_DIR)

        from PyQt6.QtGui import QColor
        # First paint and native chrome use the same current theme as Vue.
        background = get_color('bg_primary')
        self.vue_viewer = QWebEngineView()

        # 중요: 새로 만든 프로필로 페이지 생성
        page = _DebugPage(self.web_profile, self.vue_viewer)
        page.setBackgroundColor(QColor(background))
        self.vue_viewer.setPage(page)

        channel = QWebChannel(page)
        channel.registerObject('backend', self.vue_bridge)
        channel.registerObject('studio', self.studio_transport)
        page.setWebChannel(channel)
        self.web_channel = channel

        qwc = QWebEngineScript()
        qwc.setName("qwebchannel")
        qwc.setSourceUrl(QUrl("qrc:///qtwebchannel/qwebchannel.js"))
        qwc.setInjectionPoint(QWebEngineScript.InjectionPoint.DocumentCreation)
        qwc.setWorldId(QWebEngineScript.ScriptWorldId.MainWorld)
        qwc.setRunsOnSubFrames(False)
        page.scripts().insert(qwc)

        settings = page.settings()
        settings.setAttribute(QWebEngineSettings.WebAttribute.LocalContentCanAccessFileUrls, True)
        settings.setAttribute(QWebEngineSettings.WebAttribute.LocalContentCanAccessRemoteUrls, True)

        # Vue 로드(setUrl)는 백엔드 선택 *후* _load_vue_ui()에서 시작한다.
        # (다이얼로그가 떠 있는 동안 Vue/Chromium 로딩이 CPU를 경쟁해 선택이 버벅이던 문제 방지 —
        #  '백엔드 선택 → 로드 → UI' 순서)
        self._pending_vue_url = QUrl.fromLocalFile(frontend_path) if os.path.exists(frontend_path) else None

        # QStackedWidget: 0=Vue SPA, 1=Web, 2=Backend
        from PyQt6.QtWidgets import QStackedWidget
        self._main_stack = QStackedWidget()
        self._main_stack.setContentsMargins(0, 0, 0, 0)
        self._main_stack.addWidget(self.vue_viewer)  # index 0

        # Web Browser (별도 QWebEngineView — iframe 보안 우회)
        from tabs.browser_tab import BrowserTab
        self.web_tab = BrowserTab(self)
        self._main_stack.addWidget(self.web_tab)  # index 1

        # Backend UI (별도 QWebEngineView)
        from tabs.backend_ui_tab import BackendUITab
        self.backend_ui_tab = BackendUITab(self)
        self._main_stack.addWidget(self.backend_ui_tab)  # index 2

        self.setCentralWidget(self._main_stack)
        apply_native_shell_theme(self)

        # ── 프록시 위젯 초기화 ──
        self._init_prompt_proxies()
        self._init_settings_proxies()
        self._init_button_proxies()
        # LoRA 목록 프리로드는 여기서 하지 않는다 — 이 시점엔 백엔드가 아직 연결 전이다(예전
        # _preload_loras 는 2초 자고 미연결을 보고 늘 그냥 끝났다). 연결 성공 경계
        # (generator_webui.on_webui_info_loaded)가 캐시를 비운 직후 워커로 다시 채운다
        # (ui/lora_catalog_cache.prewarm_async).

        self.vue_bridge.set_action_handler(self._handle_vue_action)

        # 예전 PyQt 화면의 자리 채움 더미(어떤 속성이든 no-op 을 돌려주는 _D·_VLP 와 center_tabs·
        # viewer_label·gen_progress_bar 등)는 없앴다 — hasattr 가드를 늘 참으로 만들어 죽은 분기를
        # 숨겼다(audit #175). 상태 문구는 show_status(Vue 계기 스트립: ui/status_line.py), 진행률·
        # 결과는 vue_bridge 시그널로 간다. 히스토리·갤러리·즐겨찾기는 Vue 가 맡는다.
        # tests/test_legacy_dummies_retirement.py 가 __getattr__ 더미의 재등장을 막는다.

        # 숨은 PyQt 탭은 더 만들지 않는다 — 남은 PyQt 화면은 위에서 _main_stack 에 넣은 Web·Backend
        # 두 개뿐이다(다시 만들면 표시 중인 뷰와 설정/백엔드 로드 대상이 갈라지고 Chromium 프로필도
        # 두 벌 생긴다). EventGen·XYZ Plot·PNG Info·Gallery(tests/test_legacy_gallery_tabs_retirement.py)와
        # I2I·Inpaint·Upscale(tests/test_legacy_i2i_inpaint_upscale_retirement.py)은 Vue 뷰와
        # ui/i2i_actions·inpaint_actions·upscale_actions 로 대체돼 은퇴했다.

        # prompt_settings.json 에서 Vue 위젯이 주인이 아닌 키(와일드카드 ON/OFF·레거시 클리너 두 옵션·
        # 글꼴) — 예전 숨은 SettingsTab 의 위젯 대신 순수 값 객체가 든다(audit #178).
        from core.prompt_settings_extras import PromptSettingsExtras
        self.prompt_settings_extras = PromptSettingsExtras()
        self._bind_wildcard_enabled_proxy()

    # ──────────────────────────────────────
    #  프록시 위젯 초기화 (Vue SPA 연동)
    # ──────────────────────────────────────

    def _init_prompt_proxies(self):
        """프롬프트 영역 프록시 위젯 초기화"""
        from ui.widget_proxies import LineEditProxy, TextEditProxy, CheckBoxProxy

        b = self.vue_bridge
        p = self  # 모든 프록시의 QObject 부모 (GC 방지)
        self.char_count_input = LineEditProxy(b, 'char_count_input')
        self.character_input = LineEditProxy(b, 'character_input')
        self.copyright_input = LineEditProxy(b, 'copyright_input')
        self.artist_input = TextEditProxy(b, 'artist_input')
        self.prefix_prompt_text = TextEditProxy(b, 'prefix_prompt_text')
        self.main_prompt_text = TextEditProxy(b, 'main_prompt_text')
        self.suffix_prompt_text = TextEditProxy(b, 'suffix_prompt_text')
        self.neg_prompt_text = TextEditProxy(b, 'neg_prompt_text')
        self.exclude_prompt_local_input = TextEditProxy(b, 'exclude_prompt_local_input')
        self.total_prompt_display = TextEditProxy(b, 'total_prompt_display')

    def _bind_wildcard_enabled_proxy(self):
        """와일드카드 시스템 ON/OFF 를 Vue 에 노출 — CheckBoxProxy 'wildcard_enabled'.

        값의 주인은 ``self.prompt_settings_extras.wildcard_enabled``(core/prompt_settings_extras,
        prompt_settings.json 영속)이고 생성 경로는 utils.file_wildcard.wildcards_enabled 가 그것을
        읽는다. Vue 에서 끄고 켜면 프록시 toggled 가 곧바로 그 값을 바꾼다. 설정 불러오기
        (generator_settings._apply_prompt_settings_extras)는 값을 바꾼 뒤 프록시를 맞춘다 —
        setChecked 는 값이 바뀔 때만 신호를 내므로 되먹임하지 않는다.
        """
        if not hasattr(self, 'vue_bridge'):
            return
        from core.prompt_settings_extras import extras_of
        from ui.widget_proxies import CheckBoxProxy
        proxy = CheckBoxProxy(self.vue_bridge, 'wildcard_enabled')
        proxy.setChecked(extras_of(self).wildcard_enabled)
        proxy.toggled.connect(self._set_wildcards_enabled)
        self.wildcard_enabled_check = proxy

    def _set_wildcards_enabled(self, enabled) -> None:
        """Vue 와일드카드 토글(프록시 toggled) → 값 보관함. 다음 저장이 prompt_settings 에 쓴다."""
        from core.prompt_settings_extras import PromptSettingsExtras
        extras = getattr(self, 'prompt_settings_extras', None)
        if not isinstance(extras, PromptSettingsExtras):
            extras = PromptSettingsExtras()
            self.prompt_settings_extras = extras
        extras.wildcard_enabled = bool(enabled)

    def _init_settings_proxies(self):
        """설정 영역 프록시 위젯 초기화"""
        from ui.widget_proxies import (
            LineEditProxy, TextEditProxy, ComboBoxProxy, CheckBoxProxy,
            SliderProxy, GroupBoxProxy,
        )

        b = self.vue_bridge
        # 프롬프트 토큰 집중 — 더 구체적인 태그에 포함되는 광범위 태그 제거
        self.chk_prompt_focus = CheckBoxProxy(b, 'chk_prompt_focus')
        self.chk_auto_char_features = CheckBoxProxy(b, 'chk_auto_char_features')
        # 자동화 시 특징 자동추가의 'auto remove'(충돌 자동 처리) 토글
        self.chk_auto_remove_char_features = CheckBoxProxy(b, 'chk_auto_remove_char_features')
        self.combo_char_feature_mode = ComboBoxProxy(b, 'combo_char_feature_mode')
        self.combo_char_feature_mode.addItems(["핵심만", "핵심+의상"])
        # auto remove 시 override(머리길이/눈색 강제 교체) 설정
        if not hasattr(self, '_char_feature_override'):
            self._char_feature_override = {'hair_length': False, 'eye_color': False}

        # 서버 종류와 별개인 생성 family. Krea2는 ComfyUI 위에서 전용
        # 워크플로를 실행하므로 checkpoint 콤보와 분리해 관리한다.
        # 항목은 Vue 라벨(PromptPanel generationFamilyItems)과 글자 그대로 같아야 한다 —
        # ComboBoxProxy 는 대소문자를 구분해 맞지 않는 값을 버린다 (core/generation_family.py).
        from core.generation_family import GENERATION_FAMILY_ITEMS
        self.generation_family_combo = ComboBoxProxy(b, 'generation_family_combo')
        self.generation_family_combo.addItems(list(GENERATION_FAMILY_ITEMS))
        self.model_combo = ComboBoxProxy(b, 'model_combo')
        self.vae_main_combo = ComboBoxProxy(b, 'vae_main_combo')
        self.te_main_input = LineEditProxy(b, 'te_main_input')
        self.sampler_combo = ComboBoxProxy(b, 'sampler_combo')
        self.scheduler_combo = ComboBoxProxy(b, 'scheduler_combo')

        self.steps_input = SliderProxy(b, 'steps_input')
        self.steps_input.setText('25')
        self.cfg_input = SliderProxy(b, 'cfg_input', multiplier=2)
        self.cfg_input.setText('7')
        self.shift_input = SliderProxy(b, 'shift_input', multiplier=2)
        self.shift_input.setText('0')

        self.seed_input = LineEditProxy(b, 'seed_input')
        self.seed_input.setText('-1')
        self.width_input = LineEditProxy(b, 'width_input')
        self.width_input.setText('1024')
        self.height_input = LineEditProxy(b, 'height_input')
        self.height_input.setText('1024')

        # 해상도 관련
        self.random_res_check = CheckBoxProxy(b, 'random_res_check')
        self.auto_res_check = CheckBoxProxy(b, 'auto_res_check')
        # 랜덤 해상도 목록(random_resolutions)은 Vue 편집기가 set_random_resolutions 액션으로 보낸다.

        # Hires.fix
        self.hires_options_group = GroupBoxProxy(b, 'hires_options_group')
        self.upscaler_combo = ComboBoxProxy(b, 'upscaler_combo')
        self.hires_steps_input = SliderProxy(b, 'hires_steps_input')
        self.hires_denoising_input = SliderProxy(b, 'hires_denoising_input', multiplier=100)
        self.hires_denoising_input.setText('0.40')
        self.hires_scale_input = SliderProxy(b, 'hires_scale_input', multiplier=20)
        self.hires_scale_input.setText('2.00')
        self.hires_cfg_input = SliderProxy(b, 'hires_cfg_input', multiplier=2)
        self.hires_checkpoint_combo = ComboBoxProxy(b, 'hires_checkpoint_combo')
        self.hires_sampler_combo = ComboBoxProxy(b, 'hires_sampler_combo')
        self.hires_scheduler_combo = ComboBoxProxy(b, 'hires_scheduler_combo')
        self.hires_prompt_text = TextEditProxy(b, 'hires_prompt_text')
        self.hires_neg_prompt_text = TextEditProxy(b, 'hires_neg_prompt_text')

        # NegPiP / ADetailer
        self.negpip_group = GroupBoxProxy(b, 'negpip_group')
        # NegPiP 은 상시 적용(b8ef7901a 'NegPiP 토글 제거') — Python 이 단일 출처로 켠다.
        # 설정 복원(load_settings)도 저장값을 읽지 않고 True 를 유지한다(audit #97).
        self.negpip_group.setChecked(True)
        self.adetailer_group = GroupBoxProxy(b, 'adetailer_group')
        # ADetailer 슬롯 체크박스 (Vue 연동)
        self.ad_slot1_group = CheckBoxProxy(b, 'ad_slot1_group')
        self.ad_slot2_group = CheckBoxProxy(b, 'ad_slot2_group')
        # ADetailer 슬롯 위젯 더미 (전체 키)
        def _ad_slot(prefix):
            # SliderProxy 생성 후 기본값 설정 (Vue에도 push)
            confidence = SliderProxy(b, f'{prefix}_confidence')
            confidence.setText('0.3')
            mask_blur = SliderProxy(b, f'{prefix}_mask_blur')
            mask_blur.setText('4')
            denoise = SliderProxy(b, f'{prefix}_denoise')
            denoise.setText('0.4')
            padding = SliderProxy(b, f'{prefix}_padding')
            padding.setText('32')
            steps = SliderProxy(b, f'{prefix}_steps')
            steps.setText('28')
            cfg = SliderProxy(b, f'{prefix}_cfg')
            cfg.setText('7.0')
            dilate_erode = SliderProxy(b, f'{prefix}_dilate_erode')
            dilate_erode.setText('4')
            return {
                'prompt': TextEditProxy(b, f'{prefix}_prompt'),
                'neg_prompt': TextEditProxy(b, f'{prefix}_neg'),
                'model': ComboBoxProxy(b, f'{prefix}_model'),
                'confidence': confidence,
                'mask_blur': mask_blur,
                'denoise': denoise,
                'padding': padding,
                'steps': steps,
                'cfg': cfg,
                'dilate_erode': dilate_erode,
                'mask_merge_invert': ComboBoxProxy(b, f'{prefix}_mask_merge'),
                'use_inpaint_size_check': CheckBoxProxy(b, f'{prefix}_use_inp_size'),
                'use_steps_check': CheckBoxProxy(b, f'{prefix}_use_steps'),
                'use_cfg_check': CheckBoxProxy(b, f'{prefix}_use_cfg'),
                'use_checkpoint_check': CheckBoxProxy(b, f'{prefix}_use_ckpt'),
                'use_vae_check': CheckBoxProxy(b, f'{prefix}_use_vae'),
                'use_sampler_check': CheckBoxProxy(b, f'{prefix}_use_sampler'),
                'inpaint_width': LineEditProxy(b, f'{prefix}_inp_w'),
                'inpaint_height': LineEditProxy(b, f'{prefix}_inp_h'),
                'checkpoint_combo': ComboBoxProxy(b, f'{prefix}_ckpt'),
                'vae_combo': ComboBoxProxy(b, f'{prefix}_vae'),
                'sampler_combo': ComboBoxProxy(b, f'{prefix}_sampler'),
                'scheduler_combo': ComboBoxProxy(b, f'{prefix}_scheduler'),
            }
        self.s1_widgets = _ad_slot('_ad_s1')
        self.s1_widgets['model'].setText('face_yolov8n.pt')
        self.s2_widgets = _ad_slot('_ad_s2')
        self.s2_widgets['model'].setText('hand_yolov8n.pt')

        self.sam3_group = GroupBoxProxy(b, 'sam3_group')
        self.sam3_widgets = {
            'detect_prompt': TextEditProxy(b, '_sam3_detect_prompt'),
            'exclude_prompt': TextEditProxy(b, '_sam3_exclude_prompt'),
            'inpaint_prompt': TextEditProxy(b, '_sam3_inpaint_prompt'),
            'neg_prompt': TextEditProxy(b, '_sam3_neg_prompt'),
            'mode': ComboBoxProxy(b, '_sam3_mode'),
            'mask_mode': ComboBoxProxy(b, '_sam3_mask_mode'),
            'threshold': SliderProxy(b, '_sam3_threshold', multiplier=100),
            'mask_dilation': LineEditProxy(b, '_sam3_mask_dilation'),
            'mask_hull': CheckBoxProxy(b, '_sam3_mask_hull'),
            'mask_outline_px': LineEditProxy(b, '_sam3_mask_outline_px'),
            'mask_blur': SliderProxy(b, '_sam3_mask_blur'),
            'denoise': SliderProxy(b, '_sam3_denoise', multiplier=100),
            'padding': SliderProxy(b, '_sam3_padding'),
            'checkpoint': ComboBoxProxy(b, '_sam3_checkpoint'),
            'device': ComboBoxProxy(b, '_sam3_device'),
            'inpainting_fill': ComboBoxProxy(b, '_sam3_inpainting_fill'),
            'inpaint_only_masked': CheckBoxProxy(b, '_sam3_inpaint_only_masked'),
            'preview_overlay': CheckBoxProxy(b, '_sam3_preview_overlay'),
            'save_artifacts': CheckBoxProxy(b, '_sam3_save_artifacts'),
            'unload_after': CheckBoxProxy(b, '_sam3_unload_after'),
            'use_inpaint_size_check': CheckBoxProxy(b, '_sam3_use_inp_size'),
            'inpaint_width': LineEditProxy(b, '_sam3_inp_w'),
            'inpaint_height': LineEditProxy(b, '_sam3_inp_h'),
            'use_steps_check': CheckBoxProxy(b, '_sam3_use_steps'),
            'steps': SliderProxy(b, '_sam3_steps'),
            'use_cfg_check': CheckBoxProxy(b, '_sam3_use_cfg'),
            'cfg': SliderProxy(b, '_sam3_cfg', multiplier=10),
            'use_sampler_check': CheckBoxProxy(b, '_sam3_use_sampler'),
            'sampler': ComboBoxProxy(b, '_sam3_sampler'),
            'use_scheduler_check': CheckBoxProxy(b, '_sam3_use_scheduler'),
            'scheduler': ComboBoxProxy(b, '_sam3_scheduler'),
            'use_seed_check': CheckBoxProxy(b, '_sam3_use_seed'),
            'seed': LineEditProxy(b, '_sam3_seed'),
            'use_noise_multiplier_check': CheckBoxProxy(b, '_sam3_use_noise_mul'),
            'noise_multiplier': SliderProxy(b, '_sam3_noise_mul', multiplier=100),
            'restore_face': CheckBoxProxy(b, '_sam3_restore_face'),
            # ── ControlNet 주입 (Forge 확장의 SAM3 > ControlNet 아코디언과 1:1)
            # sam3_mode == 'Inpaint' + sd_forge_controlnet 로드 시에만 실제로 동작.
            'cn_enable': CheckBoxProxy(b, '_sam3_cn_enable'),
            'cn_override_external': CheckBoxProxy(b, '_sam3_cn_override_external'),
            # Model 은 선택지 없는 자유 입력 — ComboBoxProxy 는 빈 값('None' 으로 되돌리기)을
            # 무시하고 숫자만인 이름을 인덱스로 읽어 버려서 LineEditProxy 로 둔다.
            'cn_model': LineEditProxy(b, '_sam3_cn_model'),
            'cn_module': ComboBoxProxy(b, '_sam3_cn_module'),
            'cn_weight': SliderProxy(b, '_sam3_cn_weight', multiplier=100),
            'cn_guidance_start': SliderProxy(b, '_sam3_cn_guidance_start', multiplier=100),
            'cn_guidance_end': SliderProxy(b, '_sam3_cn_guidance_end', multiplier=100),
            'cn_pixel_perfect': CheckBoxProxy(b, '_sam3_cn_pixel_perfect'),
            'cn_control_mode': ComboBoxProxy(b, '_sam3_cn_control_mode'),
            'cn_resize_mode': ComboBoxProxy(b, '_sam3_cn_resize_mode'),
            'cn_processor_res': LineEditProxy(b, '_sam3_cn_processor_res'),
            'cn_threshold_a': LineEditProxy(b, '_sam3_cn_threshold_a'),
            'cn_threshold_b': LineEditProxy(b, '_sam3_cn_threshold_b'),
        }
        self.sam3_widgets['threshold'].setText('0.40')
        self.sam3_widgets['mask_dilation'].setText('0')
        self.sam3_widgets['mask_outline_px'].setText('0')
        self.sam3_widgets['mask_blur'].setText('4')
        self.sam3_widgets['denoise'].setText('0.40')
        self.sam3_widgets['padding'].setText('32')
        self.sam3_widgets['checkpoint'].setText('sam3.pt')
        self.sam3_widgets['device'].setText('cuda')  # auto는 CPU로 떨어질 수 있어 검출 느려짐
        self.sam3_widgets['inpainting_fill'].setText('original')
        self.sam3_widgets['seed'].setText('-1')
        self.sam3_widgets['steps'].setText('28')
        self.sam3_widgets['cfg'].setText('7.0')
        self.sam3_widgets['noise_multiplier'].setText('1.0')
        self.sam3_widgets['save_artifacts'].setChecked(True)
        # 16GB GPU 권장 기본값 (Forge 확장 v0.6.1+ 디폴트와 동일)
        self.sam3_widgets['inpaint_only_masked'].setChecked(True)
        self.sam3_widgets['unload_after'].setChecked(True)
        # ControlNet 13필드 — 선택지(전처리기·control/resize mode)와 기본값은 확장 Sam3Args
        # 스펙 한 벌에서(core/sam3_controlnet → core/sam3_args). Vue Sam3ControlNetPanel 이
        # 선택지를 getProperty(id, 'items') 로 읽는다.
        from core.sam3_controlnet import init_widgets as _init_sam3_cn_widgets
        _init_sam3_cn_widgets(self.sam3_widgets)

        # ── Anima Guidance Suite (PAG/SEG/SLG · APG/CWM/SMC · Skimmed · DCW/RDC/DAVE/CNS
        #    · Detail Daemon · Modulation). 위치 인자 계약은 core/anima_guidance.py 참조.
        self.anima_guidance_widgets = self._init_anima_guidance_proxies(b)

        # 제거 옵션
        self.chk_remove_artist = CheckBoxProxy(b, 'chk_remove_artist')
        self.chk_remove_copyright = CheckBoxProxy(b, 'chk_remove_copyright')
        self.chk_remove_character = CheckBoxProxy(b, 'chk_remove_character')
        self.chk_remove_character_features = CheckBoxProxy(b, 'chk_remove_character_features')
        self.chk_remove_meta = CheckBoxProxy(b, 'chk_remove_meta')
        self.chk_remove_censorship = CheckBoxProxy(b, 'chk_remove_censorship')
        self.chk_remove_text = CheckBoxProxy(b, 'chk_remove_text')

        # lock
        self.btn_lock_artist = CheckBoxProxy(b, 'btn_lock_artist')

    def _init_anima_guidance_proxies(self, b):
        """Anima Guidance Suite 프록시 — core/anima_guidance.py 스펙에서 자동 생성.

        키가 82개(62+7+13)라 하나씩 손으로 나열하면 스펙과 어긋나기 쉽다. 스펙을
        단일 출처로 삼아 루프로 만든다. 값은 Vue에서 전부 문자열로 오고
        (`'true'`/`'0.75'`), 타입 강제는 core 쪽 `build_args`가 담당하므로
        프록시는 LineEditProxy 하나로 통일한다.

        widget_id = '_' + 스펙 키 (예: '_guid_enabled', '_dd_preset')
        """
        from ui.widget_proxies import LineEditProxy
        from core.anima_guidance import SPECS

        widgets = {}
        for spec in SPECS.values():
            for key, _kind, default, _extra in spec:
                proxy = LineEditProxy(b, f'_{key}')
                # 기본값을 문자열로 심어 둔다 — Vue가 아직 값을 안 보냈어도
                # 생성이 확장 기본값과 동일하게 나가야 하므로.
                if isinstance(default, bool):
                    proxy.setText('true' if default else 'false')
                else:
                    proxy.setText(str(default))
                widgets[key] = proxy
        return widgets

    def _init_button_proxies(self):
        """하단 도구바 버튼 프록시 초기화"""
        from ui.widget_proxies import ButtonProxy, CheckBoxProxy

        b = self.vue_bridge

        self.btn_generate = ButtonProxy(b, 'btn_generate')
        self.btn_generate._text = "이미지 생성"
        self.btn_random_prompt = ButtonProxy(b, 'btn_random_prompt')
        self.btn_auto_toggle = ButtonProxy(b, 'btn_auto_toggle')
        self.btn_auto_toggle.setCheckable(True)
        self.btn_auto_toggle.toggled.connect(self.toggle_automation_ui)

        self.btn_save_settings = ButtonProxy(b, 'btn_save_settings')

        self._vue_automation_settings = {
            'mode': 'count',
            'limit': 10,
            'repeat': 1,
            'delay': 1.0,
            'allowDupes': False,
            'autoResetDeck': False,
            'maxRetries': 2,
        }

        def _get_vue_automation_settings():
            raw = getattr(self, '_vue_automation_settings', {}) or {}

            # PR 3: 'unlimited' 모드 추가 — 횟수/시간 제한 없이 사용자가 중지할 때까지
            mode = str(raw.get('mode', 'count'))
            if mode not in ('count', 'timer', 'unlimited'):
                mode = 'count'

            try:
                limit = float(raw.get('limit', 10))
            except (TypeError, ValueError):
                limit = 10.0

            try:
                repeat = int(raw.get('repeat', 1))
            except (TypeError, ValueError):
                repeat = 1

            try:
                delay = float(raw.get('delay', 1.0))
            except (TypeError, ValueError):
                delay = 1.0

            try:
                max_retries = int(raw.get('maxRetries', 2))
            except (TypeError, ValueError):
                max_retries = 2

            limit = max(1.0, limit)
            repeat = max(1, repeat)
            delay = max(0.0, delay)
            max_retries = max(0, min(max_retries, 10))

            if mode == 'unlimited':
                termination_limit = 0  # 무시됨
            else:
                termination_limit = int(limit) if mode == 'count' else limit * 60

            return {
                'termination_mode': mode,
                'termination_limit': termination_limit,
                'repeat_per_prompt': repeat,
                'delay': delay,
                'allow_duplicates': bool(raw.get('allowDupes', False)),
                'auto_reset_deck': bool(raw.get('autoResetDeck', False)),
                'max_retries': max_retries,
            }

        # 자동화 설정 창구 — 자동화 시작(generator_actions)이 get_settings() 로 읽는다.
        self.automation_widget = _VueAutomationSettings(_get_vue_automation_settings)

    def _show_prompt_history(self):
        """최근 프롬프트 히스토리 팝업"""
        from utils.prompt_history import get_history
        history = get_history()
        if not history:
            QMessageBox.information(self, "히스토리", "저장된 프롬프트가 없습니다.")
            return

        menu = QMenu(self)
        menu.setStyleSheet(f"""
            QMenu {{
                background-color: {get_color('bg_secondary')}; color: {get_color('text_primary')}; border: 1px solid {get_color('border')};
                padding: 4px;
            }}
            QMenu::item {{
                padding: 6px 12px; border-radius: 3px;
            }}
            QMenu::item:selected {{ background-color: {get_color('accent')}; }}
        """)
        for i, entry in enumerate(history[:30]):
            prompt_preview = entry.get("prompt", "")[:80]
            if len(entry.get("prompt", "")) > 80:
                prompt_preview += "..."
            action = menu.addAction(f"{i+1}. {prompt_preview}")
            action.setData(entry)

        from PyQt6.QtGui import QCursor
        chosen = menu.exec(QCursor.pos())
        if chosen:
            data = chosen.data()
            self.main_prompt_text.setPlainText(data.get("prompt", ""))
            self.neg_prompt_text.setPlainText(data.get("negative", ""))


    def _swap_resolution(self):
        """W ↔ H 해상도 교환"""
        w, h = self.width_input.text(), self.height_input.text()
        self.width_input.setText(h)
        self.height_input.setText(w)

    def _load_vue_ui(self):
        """백엔드 선택 완료 후 Vue SPA 로드 시작 (지연 로드).
        _setup_ui에서 미뤄둔 setUrl을 여기서 호출 — 선택 다이얼로그가 떠 있는 동안
        Vue/Chromium 로딩이 경쟁하지 않게 하여 선택을 매끄럽게 한다."""
        url = getattr(self, '_pending_vue_url', None)
        if url is not None and hasattr(self, 'vue_viewer'):
            self.vue_viewer.setUrl(url)
            self._pending_vue_url = None

    def _apply_character_features_result(self, char_name: str, tags: list, add_copyright=None):
        """캐릭터 이름 + 특징 태그를 프롬프트 위젯에 삽입 (Vue 모달 / 레거시 다이얼로그 공용).
        add_copyright: True/False 면 강제, None 이면 ui_prefs.autoAddCopyright(기본 on) 따름."""
        char_name = (char_name or "").strip()
        # 캐릭터 이름 설정
        if char_name:
            cur = self.character_input.text().strip()
            if cur:
                existing_chars = set()
                for c in cur.split(","):
                    n = c.strip().lower().replace("_", " ")
                    existing_chars.add(n)
                    existing_chars.add(n.replace(r"\(", "(").replace(r"\)", ")"))
                if char_name.lower().replace("_", " ") not in existing_chars:
                    self.character_input.setText(f"{cur}, {char_name}")
            else:
                self.character_input.setText(char_name)

        # ③ copyright(시리즈) 자동 추가 — 캐릭터→copyright (모달 override 우선, 없으면 pref)
        if char_name:
            do_copy = add_copyright if add_copyright is not None \
                else bool(self._get_ui_pref('autoAddCopyright', True))
            if do_copy:
                try:
                    from core.tag_intelligence import get_tag_intelligence
                    cp = get_tag_intelligence().copyright_of(char_name)
                    if cp:
                        self._add_copyright_tag(cp)
                except Exception:
                    pass

        # 특징 태그 삽입 (중복 제거)
        tags = tags or []
        if tags:
            all_existing: set[str] = set()
            for src in (self.main_prompt_text.toPlainText(),
                        self.prefix_prompt_text.toPlainText(),
                        self.suffix_prompt_text.toPlainText(),
                        self.character_input.text()):
                for t in src.split(","):
                    n = t.strip().lower().replace("_", " ")
                    if n:
                        all_existing.add(n)
                        all_existing.add(n.replace(r"\(", "(").replace(r"\)", ")"))

            new_tags = [
                t for t in tags
                if t.strip().lower().replace("_", " ").replace(r"\(", "(").replace(r"\)", ")") not in all_existing
            ]
            if new_tags:
                insert_str = ", ".join(new_tags)
                current = self.main_prompt_text.toPlainText().strip()
                if current:
                    self.main_prompt_text.setPlainText(f"{insert_str}, {current}")
                else:
                    self.main_prompt_text.setPlainText(insert_str)

        if hasattr(self, 'update_total_prompt_display'):
            self.update_total_prompt_display()

    def _get_ui_pref(self, key, default=None):
        """config/ui_prefs.json 에서 단일 설정 읽기 (best-effort) — 경로·로더는 core.ui_prefs 한 곳."""
        try:
            from core.ui_prefs import read_ui_prefs
            return read_ui_prefs().get(key, default)
        except Exception:
            return default

    def _add_copyright_tag(self, cp: str):
        """copyright_input 에 시리즈 태그 추가 (중복 방지 + 괄호 이스케이프)."""
        cp = (cp or "").strip()
        if not cp or not hasattr(self, 'copyright_input'):
            return
        target = cp.lower().replace("_", " ")
        cur = self.copyright_input.text().strip()
        existing = set()
        for c in cur.split(","):
            n = c.strip().lower().replace("_", " ").replace(r"\(", "(").replace(r"\)", ")")
            if n:
                existing.add(n)
        if target in existing:
            return
        cp_esc = cp.replace("(", r"\(").replace(")", r"\)")
        self.copyright_input.setText(f"{cur}, {cp_esc}" if cur else cp_esc)
