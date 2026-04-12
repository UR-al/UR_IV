# ui/generator_ui_setup.py
"""
GeneratorMainUI의 UI 구성 부분 (전체)
"""
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QTextEdit, QLineEdit, QGroupBox, QCheckBox, QTabWidget,
    QSplitter, QScrollArea, QListWidget, QMenu, QMessageBox,
    QSizePolicy, QListWidgetItem, QFrame, QStackedWidget
)
from PyQt6.QtCore import Qt, QSize
from PyQt6.QtGui import QFont, QPixmap
from widgets.common_widgets import (
    NoScrollComboBox, AutomationWidget, ResolutionItemWidget, FlowLayout
)
from widgets.sliders import NumericSlider
from widgets.favorite_tags import FavoriteTagsBar
from widgets.character_preset_dialog import CharacterPresetDialog
from widgets.common_widgets import NoScrollComboBox, AutomationWidget, ResolutionItemWidget
from config import OUTPUT_DIR
from widgets.tag_input import TagInputWidget
from utils.theme_manager import get_color

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

        self.vue_bridge = VueBridge(self)

        class _DebugPage(QWebEnginePage):
            def javaScriptConsoleMessage(self, level, message, line, source):
                print(f"[Vue] {message}")

        from PyQt6.QtWebEngineCore import QWebEngineProfile
        
        # 캐시 및 데이터 경로 설정 (프로세스 간 충돌 방지 및 액세스 거부 방지)
        # 고정된 경로가 아닌 앱 전용 독립 경로 사용
        import tempfile
        base_cache_path = os.path.join(tempfile.gettempdir(), f'AIStudioPro_{os.getpid()}')
        os.makedirs(base_cache_path, exist_ok=True)
        
        # 독립적인 프로필 생성 (defaultProfile 대신 사용)
        self.web_profile = QWebEngineProfile("AIStudioProfile", self)
        self.web_profile.setPersistentStoragePath(os.path.join(base_cache_path, "Storage"))
        self.web_profile.setCachePath(os.path.join(base_cache_path, "Cache"))
        self.web_profile.setPersistentCookiesPolicy(QWebEngineProfile.PersistentCookiesPolicy.AllowPersistentCookies)
        
        self.vue_viewer = QWebEngineView()
        self.vue_viewer.setStyleSheet("border: none; background: transparent; margin: 0px; padding: 0px;")
        
        # 중요: 새로 만든 프로필로 페이지 생성
        page = _DebugPage(self.web_profile, self.vue_viewer)
        self.vue_viewer.setPage(page)

        channel = QWebChannel(page)
        channel.registerObject('backend', self.vue_bridge)
        page.setWebChannel(channel)

        qwc = QWebEngineScript()
        qwc.setName("qwebchannel")
        qwc.setSourceUrl(QUrl("qrc:///qtwebchannel/qwebchannel.js"))
        qwc.setInjectionPoint(QWebEngineScript.InjectionPoint.DocumentCreation)
        qwc.setWorldId(QWebEngineScript.ScriptWorldId.MainWorld)
        page.scripts().insert(qwc)

        settings = page.settings()
        settings.setAttribute(QWebEngineSettings.WebAttribute.LocalContentCanAccessFileUrls, True)
        settings.setAttribute(QWebEngineSettings.WebAttribute.LocalContentCanAccessRemoteUrls, True)

        frontend_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'frontend_dist', 'index.html')
        if os.path.exists(frontend_path):
            self.vue_viewer.setUrl(QUrl.fromLocalFile(frontend_path))

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

        # Editor는 Vue에서 처리 (PyQt MosaicEditor 제거)
        from tabs.editor_tab import MosaicEditor
        self.mosaic_editor = MosaicEditor()
        self.mosaic_editor.setParent(None)  # 화면에 추가하지 않음

        self.setCentralWidget(self._main_stack)

        # ── 프록시 위젯 초기화 ──
        self._init_prompt_proxies()
        self._init_settings_proxies()
        self._init_button_proxies()

        # ── 호환성 더미 (Python 백엔드 코드에서 참조하는 속성들) ──
        self.vue_bridge.set_action_handler(self._handle_vue_action)

        _D = type('D', (), {
            '__getattr__': lambda s, n: lambda *a, **k: None
        })

        self.viewer_panel = self.vue_viewer
        self.center_tabs = _D()
        self.left_stack = _D()
        self._left_panel_container = _D()
        self.left_panel_scroll = _D()
        self.generator_panel = _D()
        self.editor_tools_scroll = _D()
        self._native_tab_bar = _D()
        self._native_tab_btns = {}
        self.history_panel = _D()
        self._tools_bar = _D()
        self._bottom_container = _D()
        self._bottom_layout = _D()
        self.status_message_label = _D()
        self.vram_label = _D()
        self.gallery_items = []
        self.gallery_layout = _D()

        # viewer_label 프록시
        class _VLP:
            def __getattr__(self, n): return lambda *a, **k: None
            def size(self):
                from PyQt6.QtCore import QSize
                return QSize(800, 600)
            class _Sig:
                def connect(self, *a): pass
            customContextMenuRequested = _Sig()
        self.viewer_label = _VLP()

        self.gen_progress_bar = _D()
        self.exif_display = _D()

        # 히스토리/갤러리 프록시
        from ui.widget_proxies import ButtonProxy
        b = self.vue_bridge
        self.btn_add_favorite = ButtonProxy(b, 'btn_add_favorite')
        self.btn_refresh_gallery = ButtonProxy(b, 'btn_refresh_gallery')

        # 기존 PyQt 탭 인스턴스 (Python 백엔드에서 참조)
        from tabs.settings_tab import SettingsTab
        from tabs.search_tab import SearchTab
        from tabs.event_gen_tab import EventGenTab
        from tabs.editor_tab import MosaicEditor
        from tabs.i2i_tab import Img2ImgTab
        from tabs.inpaint_tab import InpaintTab
        from tabs.upscale_tab import UpscaleTab
        from tabs.gallery_tab import GalleryTab
        from tabs.xyz_plot_tab import XYZPlotTab
        from tabs.pnginfo_tab import PngInfoTab
        from tabs.batch_tab import BatchTab
        from tabs.browser_tab import BrowserTab
        from tabs.backend_ui_tab import BackendUITab

        self.settings_tab = SettingsTab(self)
        self.settings_tab.setParent(None)
        self.search_tab = SearchTab(self)
        self.search_tab.setParent(None)
        self.event_gen_tab = EventGenTab(self)
        self.event_gen_tab.setParent(None)
        self.mosaic_editor = MosaicEditor()
        self.mosaic_editor.setParent(None)
        self.i2i_tab = Img2ImgTab(self)
        self.i2i_tab.setParent(None)
        self.inpaint_tab = InpaintTab(self)
        self.inpaint_tab.setParent(None)
        self.upscale_tab = UpscaleTab(self)
        self.upscale_tab.setParent(None)
        self.gallery_tab = GalleryTab(self)
        self.gallery_tab.setParent(None)
        self.xyz_plot_tab = XYZPlotTab(self)
        self.xyz_plot_tab.setParent(None)
        self.png_info_tab = PngInfoTab()
        self.png_info_tab.setParent(None)
        self.batch_tab = BatchTab(self)
        self.batch_tab.setParent(None)
        self.web_tab = BrowserTab(self)
        self.web_tab.setParent(None)
        self.backend_ui_tab = BackendUITab(self)
        self.backend_ui_tab.setParent(None)
        self._batch_upscale_tabs = _D()
        self.fav_tab = _D()

        # 설정 위젯 링크
        self.cond_prompt_check = self.settings_tab.cond_prompt_check
        self.cond_prevent_dupe_check = self.settings_tab.cond_prevent_dupe_check
        self.cond_block_editor_pos = self.settings_tab.cond_block_editor_pos
        self.cond_block_editor_neg = self.settings_tab.cond_block_editor_neg
        self.exclude_artist_checkbox = _D()
        self.exclude_copyright_checkbox = _D()

    def _create_left_panel(self):
        """왼쪽 패널: 스크롤 프롬프트 영역 + 고정 하단바"""
        container = QWidget()
        container_layout = QVBoxLayout(container)
        container_layout.setContentsMargins(0, 0, 0, 0)
        container_layout.setSpacing(0)

        # 스크롤 영역
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll.setStyleSheet("QScrollArea { border: none; }")

        scroll_content = QWidget()
        scroll_content.setMaximumWidth(460)
        self._prompt_layout = QVBoxLayout(scroll_content)
        self._prompt_layout.setContentsMargins(10, 10, 10, 10)
        self._prompt_layout.setSpacing(8)

        self._create_prompt_zone(self._prompt_layout)
        self._create_settings_container(self._prompt_layout)
        self._prompt_layout.addStretch()

        scroll.setWidget(scroll_content)
        self.left_panel_scroll = scroll
        container_layout.addWidget(scroll, 1)

        bottom_bar = self._create_bottom_toolbar()
        container_layout.addWidget(bottom_bar)

        self.generator_panel = container
        return container

    def _create_editor_tools_panel(self):
        """에디터 도구 패널 (왼쪽 패널에 표시)"""
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll.setStyleSheet(f"""
            QScrollArea {{
                background-color: {get_color('bg_primary')};
                border-right: 1px solid {get_color('border')};
                border: none;
            }}
            QScrollBar:vertical {{
                width: 10px; background: {get_color('bg_primary')};
            }}
            QScrollBar::handle:vertical {{
                background: {get_color('bg_button')}; border-radius: 5px;
            }}
        """)

        # mosaic_editor의 bottom_tabs를 스크롤 영역에 배치
        scroll.setWidget(self.mosaic_editor.bottom_tabs)
        return scroll

    def _get_tab_title(self, key: str) -> str:
        """테마에 따른 동적 탭 이름 반환"""
        from utils.theme_manager import get_theme_manager
        theme = get_theme_manager().current_theme_name
        use_minimal = theme == '모던'

        titles_minimal = {
            't2i': "T2I", 'i2i': "I2I", 'inpaint': "Inpaint",
            'event': "Event Gen", 'search': "Search", 'web': "Web",
            'editor': "Editor", 'batch': "Batch / Upscale",
            'gallery': "Gallery", 'xyz': "XYZ Plot", 'png': "PNG Info",
            'fav': "Favorites", 'backend': "Backend UI", 'settings': "Settings",
        }
        titles_emoji = {
            't2i': "🖼️ T2I", 'i2i': "🖼️ I2I", 'inpaint': "🎨 Inpaint",
            'event': "🎬 이벤트 생성", 'search': "🔍 Search", 'web': "🌐 Web",
            'editor': "🎨 Editor", 'batch': "📦 배치/업스케일",
            'gallery': "🖼️ Gallery", 'xyz': "📊 XYZ Plot", 'png': "ℹ️ PNG Info",
            'fav': "⭐ Favorites", 'backend': "🖥️ Backend UI", 'settings': "⚙️ Setting",
        }
        titles = titles_minimal if use_minimal else titles_emoji
        return titles.get(key, key)

    def _update_tab_titles(self):
        """테마 변경 시 모든 탭 이름 업데이트"""
        tab_keys = [
            't2i', 'i2i', 'inpaint', 'event', 'search', 'web', 'editor',
            'batch', 'gallery', 'xyz', 'png', 'fav', 'backend', 'settings'
        ]
        try:
            count = self.center_tabs.count()
            if count is None: return
            for i, key in enumerate(tab_keys):
                if i < count:
                    self.center_tabs.setTabText(i, self._get_tab_title(key))
        except (TypeError, AttributeError):
            pass  # Vue 모드에서는 center_tabs가 더미

    def _create_center_tabs(self):
        """중앙 탭 위젯 생성"""
        from tabs.browser_tab import BrowserTab
        from tabs.settings_tab import SettingsTab
        from tabs.pnginfo_tab import PngInfoTab
        from tabs.search_tab import SearchTab
        from tabs.editor_tab import MosaicEditor
        from tabs.upscale_tab import UpscaleTab
        from tabs.gallery_tab import GalleryTab
        from tabs.event_gen_tab import EventGenTab
        from tabs.xyz_plot_tab import XYZPlotTab
        from tabs.i2i_tab import Img2ImgTab
        from tabs.inpaint_tab import InpaintTab
        from tabs.backend_ui_tab import BackendUITab

        # ── 중앙 영역: Vue SPA만 (QTabWidget/네이티브 탭 제거) ──
        self.viewer_panel = self._create_viewer_panel()

        # center_tabs 호환성 더미
        self.center_tabs = self.viewer_panel
        self.center_tabs.setCurrentIndex = lambda i: None
        self.center_tabs.currentChanged = type('S', (), {'connect': lambda *a: None})()
        self.center_tabs.indexOf = lambda w: 0
        self.center_tabs.setTabEnabled = lambda i, v: None
        self.center_tabs.setTabToolTip = lambda i, t: None
        self.center_tabs.widget = lambda i: self.viewer_panel
        self.center_tabs.count = lambda: 1
        self.center_tabs.tabText = lambda i: 'AI Studio'
        self.center_tabs.setTabText = lambda i, t: None

        # mosaic_editor, web_tab, backend_ui_tab은 _main_stack에서 이미 생성됨

        # 기존 PyQt 탭들 — 인스턴스만 생성 (호환성 유지)
        self.i2i_tab = Img2ImgTab(self)
        self.inpaint_tab = InpaintTab(self)
        self.event_gen_tab = EventGenTab(self)
        self.search_tab = SearchTab(self)
        from tabs.batch_tab import BatchTab
        self.batch_tab = BatchTab(self)
        self.upscale_tab = UpscaleTab(self)
        self._batch_upscale_tabs = QTabWidget()
        self._batch_upscale_tabs.addTab(self.batch_tab, "배치 처리")
        self._batch_upscale_tabs.addTab(self.upscale_tab, "Upscale")
        self.gallery_tab = GalleryTab(self)
        self.xyz_plot_tab = XYZPlotTab(self)
        self.png_info_tab = PngInfoTab()
        self.fav_tab = self._create_favorites_tab()
        self.settings_tab = SettingsTab(self)

        # Vue 액션 핸들러 등록 (탭 전환 등)
        self.vue_bridge.set_action_handler(self._handle_vue_action)

        # Vue에서 탭 전환 시 center_tabs 인덱스 변경

        
        # 설정 위젯 링크 (조건부 프롬프트 등)
        self.cond_prompt_check = self.settings_tab.cond_prompt_check
        self.cond_prevent_dupe_check = self.settings_tab.cond_prevent_dupe_check
        self.cond_block_editor_pos = self.settings_tab.cond_block_editor_pos
        self.cond_block_editor_neg = self.settings_tab.cond_block_editor_neg
        
        # 검색 결과 디스플레이 링크
        self.exclude_artist_checkbox = QCheckBox() 
        self.exclude_copyright_checkbox = QCheckBox()
        
        # ★★★ 탭 전환 시그널 연결 ★★★
        self.center_tabs.currentChanged.connect(self._on_center_tab_changed)

        return self.center_tabs
    
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
        self.token_count_label = type('LabelProxy', (), {
            'setText': lambda self, t: None,
            'setAlignment': lambda self, a: None,
            'setStyleSheet': lambda self, s: None,
            'hide': lambda self: None,
            'show': lambda self: None,
            'setVisible': lambda self, v: None,
            'setFixedHeight': lambda self, h: None,
        })()

        # 즐겨찾기 태그 바 (Vue에서 렌더링 — 더미)
        self.fav_tags_bar = type('DummyFavBar', (), {
            'tag_insert_requested': type('Sig', (), {'connect': lambda *a: None})(),
            'hide': lambda self: None,
            'show': lambda self: None,
        })()

        # 호환성: 토글 버튼 더미
        class _AlwaysOn:
            def isChecked(self): return True
            def setChecked(self, v): pass
            toggled = type('', (), {'connect': lambda *a: None})()
        _d = _AlwaysOn()
        self.prefix_toggle_button = _d
        self.suffix_toggle_button = _d
        self.neg_toggle_button = _d
        self.exclude_toggle_button = _d

    def _init_settings_proxies(self):
        """설정 영역 프록시 위젯 초기화"""
        from ui.widget_proxies import (
            LineEditProxy, TextEditProxy, ComboBoxProxy, CheckBoxProxy,
            SliderProxy, GroupBoxProxy, ButtonProxy
        )

        b = self.vue_bridge
        self.chk_auto_char_features = CheckBoxProxy(b, 'chk_auto_char_features')
        self.combo_char_feature_mode = ComboBoxProxy(b, 'combo_char_feature_mode')
        self.combo_char_feature_mode.addItems(["핵심만", "핵심+의상"])
        self.btn_char_preset = ButtonProxy(b, 'btn_char_preset')

        self.model_combo = ComboBoxProxy(b, 'model_combo')
        self.sampler_combo = ComboBoxProxy(b, 'sampler_combo')
        self.scheduler_combo = ComboBoxProxy(b, 'scheduler_combo')

        self.steps_input = SliderProxy(b, 'steps_input')
        self.steps_input.setText('25')
        self.cfg_input = SliderProxy(b, 'cfg_input', multiplier=2)
        self.cfg_input.setText('7')

        self.seed_input = LineEditProxy(b, 'seed_input')
        self.seed_input.setText('-1')
        self.width_input = LineEditProxy(b, 'width_input')
        self.width_input.setText('1024')
        self.height_input = LineEditProxy(b, 'height_input')
        self.height_input.setText('1024')

        # 해상도 관련
        self.random_res_check = CheckBoxProxy(b, 'random_res_check')
        self.auto_res_check = CheckBoxProxy(b, 'auto_res_check')
        self.random_res_label = type('LblProxy', (), {
            'setText': lambda s, t: None, 'hide': lambda s: None, 'show': lambda s: None,
        })()
        self.resolution_editor_container = type('WProxy', (), {
            'hide': lambda s: None, 'show': lambda s: None,
        })()
        self.resolution_list_widget = type('LWProxy', (), {
            'clear': lambda s: None, 'addItem': lambda s, i: None,
            'count': lambda s: 0, 'item': lambda s, i: None,
            'setFixedHeight': lambda s, h: None,
        })()
        self.res_width_input = LineEditProxy(b, 'res_width_input')
        self.res_height_input = LineEditProxy(b, 'res_height_input')
        self.btn_add_res = ButtonProxy(b, 'btn_add_res')
        self._res_presets = [
            ["512 × 512", 512, 512], ["512 × 768", 512, 768], ["768 × 512", 768, 512],
            ["1024 × 1024", 1024, 1024], ["832 × 1216", 832, 1216], ["1216 × 832", 1216, 832],
        ]
        self._DEFAULT_RES_PRESETS = self._res_presets[:]
        self._res_preset_btns = []

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
        self.adetailer_group = GroupBoxProxy(b, 'adetailer_group')
        self.ad_toggle_button = ButtonProxy(b, 'ad_toggle_button')
        self.ad_settings_container = type('WProxy', (), {
            'hide': lambda s: None, 'show': lambda s: None, 'setVisible': lambda s, v: None,
        })()
        # ADetailer 슬롯 체크박스 (Vue 연동)
        self.ad_slot1_group = CheckBoxProxy(b, 'ad_slot1_group')
        self.ad_slot2_group = CheckBoxProxy(b, 'ad_slot2_group')
        # ADetailer 슬롯 위젯 더미 (전체 키)
        def _ad_slot(prefix):
            _W = type('WProxy', (), {
                'setVisible': lambda s, v: None, 'hide': lambda s: None,
                'show': lambda s: None, 'isVisible': lambda s: False,
            })
            # SliderProxy 생성 후 기본값 설정 (Vue에도 push)
            confidence = SliderProxy(b, f'{prefix}_confidence')
            confidence.setText('0.3')
            strength = SliderProxy(b, f'{prefix}_str')
            strength.setText('0.4')
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
                'strength': strength,
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
                'inpaint_size_container': _W(),
                'inpaint_width': LineEditProxy(b, f'{prefix}_inp_w'),
                'inpaint_height': LineEditProxy(b, f'{prefix}_inp_h'),
                'checkpoint_combo': ComboBoxProxy(b, f'{prefix}_ckpt'),
                'vae_combo': ComboBoxProxy(b, f'{prefix}_vae'),
                'sampler_combo': ComboBoxProxy(b, f'{prefix}_sampler'),
                'scheduler_combo': ComboBoxProxy(b, f'{prefix}_scheduler'),
                'sampler_container': _W(),
            }
        self.s1_widgets = _ad_slot('_ad_s1')
        self.s1_widgets['model'].setText('face_yolov8n.pt')
        self.s2_widgets = _ad_slot('_ad_s2')
        self.s2_widgets['model'].setText('hand_yolov8n.pt')

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
        self.btn_preset_save = ButtonProxy(b, 'btn_preset_save')
        self.btn_preset_load = ButtonProxy(b, 'btn_preset_load')
        self.btn_prompt_history = ButtonProxy(b, 'btn_prompt_history')
        self.btn_lora_manager = ButtonProxy(b, 'btn_lora_manager')
        self.btn_tag_weights = ButtonProxy(b, 'btn_tag_weights')
        self.btn_shuffle = ButtonProxy(b, 'btn_shuffle')
        self.btn_ab_test = ButtonProxy(b, 'btn_ab_test')
        self.btn_api_manager = None

        self._vue_automation_settings = {
            'mode': 'count',
            'limit': 10,
            'repeat': 1,
            'delay': 1.0,
            'allowDupes': False,
        }

        def _get_vue_automation_settings():
            raw = getattr(self, '_vue_automation_settings', {}) or {}

            mode = str(raw.get('mode', 'count'))
            if mode not in ('count', 'timer'):
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

            limit = max(1.0, limit)
            repeat = max(1, repeat)
            delay = max(0.0, delay)

            termination_limit = int(limit) if mode == 'count' else limit * 60

            return {
                'termination_mode': mode,
                'termination_limit': termination_limit,
                'repeat_per_prompt': repeat,
                'delay': delay,
                'allow_duplicates': bool(raw.get('allowDupes', False)),
            }

        # 자동화 위젯 (더미)
        self.automation_widget = type('AutoProxy', (), {
            'hide': lambda s: None, 'show': lambda s: None,
            'get_settings': lambda s: _get_vue_automation_settings(),
            'setVisible': lambda s, v: None,
        })()

        # LoRA 패널 (더미)
        from ui.widget_proxies import ButtonProxy as BP
        self.lora_active_panel = type('LoraProxy', (), {
            'hide': lambda s: None, 'show': lambda s: None,
            'get_active_loras': lambda s: [],
            'get_active_lora_text': lambda s: '',
            'clear_all': lambda s: None,
            'get_entries': lambda s: [],
            'set_entries': lambda s, e: None,
            'add_lora': lambda s, n, w: None,
            'parse_and_add_loras': lambda s, t: None,
        })()

    def _prompt_label(self, text: str, color: str = None) -> QLabel:
        """NAIS2식 작은 프롬프트 라벨"""
        lbl = QLabel(text)
        c = color or get_color('text_muted')
        lbl.setStyleSheet(f"color: {c}; font-size: 10px; font-weight: 600; margin-top: 2px; padding: 0;")
        return lbl

    def _create_prompt_zone(self, layout):
        """프롬프트 영역 (항상 표시) — NAIS2 메인 뷰"""
        _toggle_style = (
            f"QPushButton {{ background: transparent; color: {get_color('text_muted')}; "
            f"border: none; text-align: left; font-size: 10px; padding: 4px 0; }}"
        )

        # ── 1. 전체 프롬프트 (기본 펼침) ──
        layout.addWidget(self._prompt_label("최종 프롬프트"))
        self.total_prompt_display = QTextEdit()
        self.total_prompt_display.setPlaceholderText("Final output prompt...")
        self.total_prompt_display.setMinimumHeight(60)
        self.total_prompt_display.document().contentsChanged.connect(self._adjust_total_prompt_height)
        self.total_prompt_display.textChanged.connect(self._update_token_count)
        layout.addWidget(self.total_prompt_display)

        self.token_count_label = QLabel("TOKENS: 0 / 75")
        self.token_count_label.setAlignment(Qt.AlignmentFlag.AlignRight)
        self.token_count_label.setStyleSheet(f"font-size: 9px; color: {get_color('text_muted')};")
        layout.addWidget(self.token_count_label)

        # 접이식 토글 더미 (호환성)
        self._final_prompt_toggle = QPushButton()
        self._final_prompt_toggle.hide()

        # ── 2. 인물 수 / 캐릭터 / 작품명 (각 한 줄) ──
        layout.addWidget(self._prompt_label("인물 수"))
        self.char_count_input = QLineEdit()
        self.char_count_input.setPlaceholderText("예: 1girl, 2boys")
        layout.addWidget(self.char_count_input)

        layout.addWidget(self._prompt_label("캐릭터"))
        self.character_input = QLineEdit()
        self.character_input.setPlaceholderText("캐릭터 이름")
        layout.addWidget(self.character_input)

        layout.addWidget(self._prompt_label("작품명"))
        self.copyright_input = QLineEdit()
        self.copyright_input.setPlaceholderText("작품 (Copyright)")
        layout.addWidget(self.copyright_input)

        # ── 3. 작가 + 고정 버튼 ──
        artist_row = QHBoxLayout()
        artist_row.setSpacing(4)
        self.artist_input = TagInputWidget()
        self.artist_input.setPlaceholderText("작가 태그...")
        self.artist_input.setMinimumHeight(36)
        self.btn_lock_artist = QPushButton("🔒")
        self.btn_lock_artist.setCheckable(True)
        self.btn_lock_artist.setFixedSize(36, 36)
        self.btn_lock_artist.setToolTip("작가 고정")
        artist_row.addWidget(self.artist_input)
        artist_row.addWidget(self.btn_lock_artist)
        layout.addLayout(artist_row)

        # ── 4. 선행 프롬프트 ──
        layout.addWidget(self._prompt_label("선행 프롬프트"))
        self.prefix_prompt_text = TagInputWidget()
        self.prefix_prompt_text.setPlaceholderText("year 2025, masterpiece, best quality, ...")
        self.prefix_prompt_text.setMinimumHeight(50)
        layout.addWidget(self.prefix_prompt_text)

        # ── 5. 메인 프롬프트 ──
        layout.addWidget(self._prompt_label("메인 프롬프트"))
        self.main_prompt_text = TagInputWidget()
        self.main_prompt_text.setPlaceholderText("메인 태그 입력...")
        self.main_prompt_text.setMinimumHeight(80)
        layout.addWidget(self.main_prompt_text)

        # 즐겨찾기 태그 바
        self.fav_tags_bar = FavoriteTagsBar()
        self.fav_tags_bar.tag_insert_requested.connect(self._insert_fav_tag)
        layout.addWidget(self.fav_tags_bar)

        # ── 6. 후행 프롬프트 ──
        layout.addWidget(self._prompt_label("후행 프롬프트"))
        self.suffix_prompt_text = TagInputWidget()
        self.suffix_prompt_text.setPlaceholderText("후행 고정 태그...")
        self.suffix_prompt_text.setMinimumHeight(50)
        layout.addWidget(self.suffix_prompt_text)

        # ── 7. 네거티브 프롬프트 ──
        layout.addWidget(self._prompt_label("네거티브 프롬프트", get_color('error')))
        self.neg_prompt_text = TagInputWidget()
        self.neg_prompt_text.setPlaceholderText("lowres, bad quality, ...")
        self.neg_prompt_text.setMinimumHeight(50)
        layout.addWidget(self.neg_prompt_text)

        # ── 8. 제외 프롬프트 (접이식) ──
        self._exclude_toggle = QPushButton("▶ 제외 프롬프트")
        self._exclude_toggle.setCheckable(True)
        self._exclude_toggle.setChecked(False)
        self._exclude_toggle.setStyleSheet(_toggle_style)
        layout.addWidget(self._exclude_toggle)

        self.exclude_prompt_local_input = TagInputWidget()
        self.exclude_prompt_local_input.setMinimumHeight(50)
        self.exclude_prompt_local_input.setPlaceholderText("예: arms up, __hair, ~blue hair")
        self.exclude_prompt_local_input.hide()
        layout.addWidget(self.exclude_prompt_local_input)

        self._exclude_toggle.toggled.connect(
            lambda on: (self.exclude_prompt_local_input.setVisible(on),
                        self._exclude_toggle.setText(
                            "▼ 제외 프롬프트" if on else "▶ 제외 프롬프트"))
        )

        # 호환성: 기존 토글 버튼 속성 (새 구조에서는 항상 표시이므로 더미)
        class _AlwaysOnToggle:
            """항상 True를 반환하는 더미 토글"""
            def isChecked(self): return True
            def setChecked(self, v): pass
            def toggled(self): pass
            @staticmethod
            def connect(*a): pass
        _dummy = _AlwaysOnToggle()
        _dummy.toggled = type('', (), {'connect': lambda *a: None})()
        self.prefix_toggle_button = _dummy
        self.suffix_toggle_button = _dummy
        self.neg_toggle_button = _dummy
        self.exclude_toggle_button = self._exclude_toggle

    def _create_settings_container(self, layout):
        """설정 위젯들 (접이식 — 기본 접힘)"""

        # ── 접이식 설정 토글 ──
        self._settings_toggle = QPushButton("▶ 생성 설정")
        self._settings_toggle.setCheckable(True)
        self._settings_toggle.setChecked(False)
        self._settings_toggle.setStyleSheet(
            f"QPushButton {{ background: {get_color('bg_secondary')}; color: {get_color('text_secondary')}; "
            f"border: 1px solid {get_color('border')}; border-radius: 6px; "
            f"text-align: left; font-size: 10px; font-weight: bold; padding: 8px 12px; }}"
            f"QPushButton:checked {{ border-color: {get_color('accent')}; color: {get_color('text_primary')}; }}"
        )
        layout.addWidget(self._settings_toggle)

        self._settings_container = QWidget()
        sl = QVBoxLayout(self._settings_container)
        sl.setContentsMargins(0, 4, 0, 0)
        sl.setSpacing(8)

        # 캐릭터 옵션
        self.chk_auto_char_features = QCheckBox("특징 자동 추가")
        sl.addWidget(self.chk_auto_char_features)

        sl.addWidget(self._prompt_label("특징 모드"))
        self.combo_char_feature_mode = NoScrollComboBox()
        self.combo_char_feature_mode.addItems(["핵심만", "핵심+의상"])
        sl.addWidget(self.combo_char_feature_mode)

        self.btn_char_preset = QPushButton("특징 프리셋")
        self.btn_char_preset.clicked.connect(self._open_character_preset)
        sl.addWidget(self.btn_char_preset)

        # 모델
        sl.addWidget(self._prompt_label("모델"))
        self.model_combo = NoScrollComboBox()
        sl.addWidget(self.model_combo)

        # 샘플러
        sl.addWidget(self._prompt_label("샘플러"))
        self.sampler_combo = NoScrollComboBox()
        sl.addWidget(self.sampler_combo)

        # 스케줄러
        sl.addWidget(self._prompt_label("스케줄러"))
        self.scheduler_combo = NoScrollComboBox()
        sl.addWidget(self.scheduler_combo)

        # Steps / CFG
        self.steps_input, _ = self._create_param_slider(sl, "Steps", 1, 100, 25, 1)
        self.cfg_input, _ = self._create_param_slider(sl, "CFG", 1, 20, 7, 0.5)

        # Seed
        seed_row = QHBoxLayout()
        self.seed_input = QLineEdit("-1")
        btn_seed = QPushButton("🎲")
        btn_seed.setFixedSize(30, 30)
        btn_seed.clicked.connect(lambda: self.seed_input.setText("-1"))
        seed_row.addWidget(self.seed_input)
        seed_row.addWidget(btn_seed)
        sl.addWidget(self._prompt_label("Seed"))
        sl.addLayout(seed_row)

        # 해상도 (W × H 한 줄)
        sl.addWidget(self._prompt_label("해상도"))
        res_row = QHBoxLayout()
        res_row.setSpacing(6)
        self.width_input = QLineEdit("1024")
        self.width_input.setPlaceholderText("W")
        self.width_input.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.height_input = QLineEdit("1024")
        self.height_input.setPlaceholderText("H")
        self.height_input.setAlignment(Qt.AlignmentFlag.AlignCenter)
        _lbl_x = QLabel("×")
        _lbl_x.setAlignment(Qt.AlignmentFlag.AlignCenter)
        _lbl_x.setFixedWidth(16)
        btn_swap = QPushButton("<->")
        btn_swap.setFixedHeight(32)
        btn_swap.setToolTip("W ↔ H 교환")
        btn_swap.clicked.connect(self._swap_resolution)
        res_row.addWidget(self.width_input)
        res_row.addWidget(_lbl_x)
        res_row.addWidget(self.height_input)
        res_row.addWidget(btn_swap)
        sl.addLayout(res_row)

        # 해상도 퀵 프리셋 (각 한 줄)
        sl.addWidget(self._prompt_label("퀵 프리셋"))
        from PyQt6.QtWidgets import QInputDialog
        self._DEFAULT_RES_PRESETS = [
            ("512 × 512", 512, 512), ("512 × 768", 512, 768), ("768 × 512", 768, 512),
            ("1024 × 1024", 1024, 1024), ("832 × 1216", 832, 1216), ("1216 × 832", 1216, 832),
        ]
        self._res_presets = [list(p) for p in self._DEFAULT_RES_PRESETS]
        self._res_preset_btns = []
        for i, (_label, _w, _h) in enumerate(self._res_presets):
            _btn = QPushButton(_label)
            _btn.setFixedHeight(28)
            _btn.clicked.connect(
                lambda _, idx=i: (
                    self.width_input.setText(str(self._res_presets[idx][1])),
                    self.height_input.setText(str(self._res_presets[idx][2]))
                )
            )
            _btn.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
            _btn.customContextMenuRequested.connect(
                lambda pos, idx=i, b=_btn: self._on_res_preset_context(idx, b)
            )
            self._res_preset_btns.append(_btn)
            sl.addWidget(_btn)

        self.random_res_check = QCheckBox("랜덤 해상도")
        self.auto_res_check = QCheckBox("자동 해상도 (Parquet H/W)")
        sl.addWidget(self.random_res_check)
        sl.addWidget(self.auto_res_check)
        self.random_res_label = QLabel()
        sl.addWidget(self.random_res_label)

        # 해상도 편집기
        self.resolution_editor_container = QWidget()
        res_edit_layout = QVBoxLayout(self.resolution_editor_container)
        input_res_layout = QHBoxLayout()
        self.res_width_input = QLineEdit()
        self.res_width_input.setPlaceholderText("W")
        self.res_height_input = QLineEdit()
        self.res_height_input.setPlaceholderText("H")
        self.btn_add_res = QPushButton("+")
        input_res_layout.addWidget(QLabel("W:"))
        input_res_layout.addWidget(self.res_width_input)
        input_res_layout.addWidget(QLabel("H:"))
        input_res_layout.addWidget(self.res_height_input)
        input_res_layout.addWidget(self.btn_add_res)
        res_edit_layout.addLayout(input_res_layout)
        self.resolution_list_widget = QListWidget()
        self.resolution_list_widget.setFixedHeight(100)
        res_edit_layout.addWidget(self.resolution_list_widget)
        self.resolution_editor_container.hide()
        sl.addWidget(self.resolution_editor_container)

        # Hires.fix
        self.hires_options_group = QGroupBox("Hires.fix")
        self.hires_options_group.setCheckable(True)
        self.hires_options_group.setChecked(False)
        hl = QVBoxLayout(self.hires_options_group)
        self.upscaler_combo = NoScrollComboBox()
        hl.addWidget(self.upscaler_combo)
        self.hires_steps_input, _ = self._create_param_slider(hl, "Steps", 0, 50, 0, 1)
        self.hires_denoising_input, _ = self._create_param_slider(hl, "Denoise", 0, 1, 0.4, 0.01)
        self.hires_scale_input, _ = self._create_param_slider(hl, "Scale", 1, 4, 2, 0.05)
        self.hires_cfg_input, _ = self._create_param_slider(hl, "CFG", 0, 30, 0, 0.5)
        self.hires_checkpoint_combo = NoScrollComboBox()
        hl.addWidget(QLabel("Checkpoint"))
        hl.addWidget(self.hires_checkpoint_combo)
        self.hires_sampler_combo = NoScrollComboBox()
        self.hires_scheduler_combo = NoScrollComboBox()
        hl.addWidget(self._make_hbox([self.hires_sampler_combo, self.hires_scheduler_combo]))
        self.hires_prompt_text = QTextEdit()
        self.hires_prompt_text.setFixedHeight(50)
        self.hires_prompt_text.setPlaceholderText("비워두면 메인 프롬프트 사용")
        hl.addWidget(QLabel("Hires Prompt"))
        hl.addWidget(self.hires_prompt_text)
        self.hires_neg_prompt_text = QTextEdit()
        self.hires_neg_prompt_text.setFixedHeight(50)
        self.hires_neg_prompt_text.setPlaceholderText("비워두면 메인 네거티브 사용")
        hl.addWidget(QLabel("Hires Negative"))
        hl.addWidget(self.hires_neg_prompt_text)
        sl.addWidget(self.hires_options_group)

        # NegPiP
        self.negpip_group = QGroupBox("NegPiP 확장")
        self.negpip_group.setCheckable(True)
        self.negpip_group.setChecked(False)
        np_l = QVBoxLayout(self.negpip_group)
        np_l.addWidget(QLabel("(keyword:-1.0) 네거티브 가중치 문법"))
        sl.addWidget(self.negpip_group)

        # ADetailer
        self.adetailer_group = QGroupBox("ADetailer")
        self.adetailer_group.setCheckable(True)
        self.adetailer_group.setChecked(False)
        ad_l = QVBoxLayout(self.adetailer_group)
        self.ad_toggle_button = QPushButton("설정 보기")
        self.ad_toggle_button.setCheckable(True)
        ad_l.addWidget(self.ad_toggle_button)
        self.ad_settings_container = QWidget()
        self.ad_settings_container.hide()
        ad_sets = QVBoxLayout(self.ad_settings_container)
        self.ad_slot1_group, self.s1_widgets = self._create_adetailer_slot_ui("Slot 1", "face_yolov8n.pt")
        self.ad_slot2_group, self.s2_widgets = self._create_adetailer_slot_ui("Slot 2", "hand_yolov8n.pt")
        ad_sets.addWidget(self.ad_slot1_group)
        ad_sets.addWidget(self.ad_slot2_group)
        ad_l.addWidget(self.ad_settings_container)
        sl.addWidget(self.adetailer_group)

        # 제거 옵션
        remove_group = QGroupBox("태그 제거 옵션")
        rg_l = QVBoxLayout(remove_group)
        self.chk_remove_artist = QCheckBox("작가명 제거")
        self.chk_remove_copyright = QCheckBox("작품명 제거")
        self.chk_remove_character = QCheckBox("캐릭터 제거")
        self.chk_remove_character_features = QCheckBox("캐릭터 특징 제거")
        self.chk_remove_meta = QCheckBox("메타 제거")
        self.chk_remove_censorship = QCheckBox("검열 제거")
        self.chk_remove_text = QCheckBox("텍스트 제거")
        for chk in [self.chk_remove_artist, self.chk_remove_copyright, self.chk_remove_character,
                     self.chk_remove_character_features,
                     self.chk_remove_meta, self.chk_remove_censorship, self.chk_remove_text]:
            rg_l.addWidget(chk)
        sl.addWidget(remove_group)

        self._settings_container.hide()
        layout.addWidget(self._settings_container)

        self._settings_toggle.toggled.connect(
            lambda on: (self._settings_container.setVisible(on),
                        self._settings_toggle.setText(
                            "▼ 생성 설정" if on else "▶ 생성 설정"))
        )

    def _create_bottom_toolbar(self):
        """좌측 패널 고정 하단: 자동화(접이식) + 생성 버튼"""
        bar = QWidget()
        bar.setStyleSheet(f"background-color: {get_color('bg_secondary')}; border-top: 1px solid {get_color('border')};")
        bl = QVBoxLayout(bar)
        bl.setContentsMargins(10, 6, 10, 6)
        bl.setSpacing(4)

        # 자동화 ON/OFF 토글 (체크 = 자동화 활성)
        self.btn_auto_toggle = QPushButton("AUTOMATION: OFF")
        self.btn_auto_toggle.setCheckable(True)
        self.btn_auto_toggle.setFixedHeight(32)
        self.btn_auto_toggle.toggled.connect(self.toggle_automation_ui)
        bl.addWidget(self.btn_auto_toggle)

        # 자동화 설정 펼침/접기 (ON/OFF 무관하게 독립적으로 동작)
        self._auto_settings_toggle = QPushButton("▶ 자동화 설정")
        self._auto_settings_toggle.setCheckable(True)
        self._auto_settings_toggle.setChecked(False)
        self._auto_settings_toggle.setFixedHeight(28)
        self._auto_settings_toggle.setStyleSheet(
            f"QPushButton {{ background: transparent; color: {get_color('text_muted')}; "
            f"border: none; text-align: left; font-size: 10px; padding: 2px 0; }}"
        )
        bl.addWidget(self._auto_settings_toggle)

        self.automation_widget = AutomationWidget()
        self.automation_widget.hide()
        bl.addWidget(self.automation_widget)

        self._auto_settings_toggle.toggled.connect(
            lambda on: (self.automation_widget.setVisible(on),
                        self._auto_settings_toggle.setText(
                            "▼ 자동화 설정" if on else "▶ 자동화 설정"))
        )

        # 생성 버튼
        self.btn_generate = QPushButton("이미지 생성")
        self.btn_generate.setObjectName("primaryButton")
        self.btn_generate.setFixedHeight(44)
        self.btn_generate.setEnabled(False)
        bl.addWidget(self.btn_generate)

        return bar

    def _create_tools_bar(self):
        """대기열 위 전체 너비 도구 바 (2행: 백엔드 / 도구 버튼)"""
        bar = QWidget()
        bar.setStyleSheet(
            f"background-color: {get_color('bg_secondary')}; "
            f"border-top: 1px solid {get_color('border')}; "
            f"border-bottom: 1px solid {get_color('border')};"
        )
        vl = QVBoxLayout(bar)
        vl.setContentsMargins(8, 4, 8, 4)
        vl.setSpacing(4)

        # ── 행 1: 백엔드 상태 (풀 너비) ──
        from widgets.api_status_button import ApiStatusButton
        self.btn_api_manager = ApiStatusButton()
        self.btn_api_manager.setFixedHeight(36)
        self.btn_api_manager.clicked.connect(self._show_api_manager_popup)
        vl.addWidget(self.btn_api_manager)

        # ── 행 2: 도구 버튼들 (가로 나열) ──
        tools_row = QHBoxLayout()
        tools_row.setSpacing(4)

        def _sep():
            s = QFrame()
            s.setFrameShape(QFrame.Shape.VLine)
            s.setFixedWidth(1)
            s.setFixedHeight(24)
            s.setStyleSheet(f"color: {get_color('border')};")
            return s

        _btn_h = 30

        # 저장 / 프리셋
        self.btn_save_settings = QPushButton("  💾 저장  ")
        self.btn_save_settings.setToolTip("설정 저장")

        self._btn_preset = QPushButton("  📦 프리셋  ")
        self._btn_preset.setToolTip("프리셋 저장/불러오기")
        _preset_menu = QMenu(self._btn_preset)
        _preset_menu.addAction("  📥  프리셋 저장").triggered.connect(self._save_prompt_preset)
        _preset_menu.addAction("  📤  프리셋 불러오기").triggered.connect(self._load_prompt_preset)
        self._btn_preset.setMenu(_preset_menu)
        self.btn_preset_save = self._btn_preset
        self.btn_preset_load = self._btn_preset

        self.btn_prompt_history = QPushButton("  📋 히스토리  ")
        self.btn_prompt_history.setToolTip("프롬프트 히스토리")
        self.btn_prompt_history.clicked.connect(self._show_prompt_history)

        for btn in [self.btn_save_settings, self._btn_preset, self.btn_prompt_history]:
            btn.setFixedHeight(_btn_h)
            tools_row.addWidget(btn)

        tools_row.addWidget(_sep())

        # LoRA / 가중치
        self.btn_lora_manager = QPushButton("  LoRA  ")
        self.btn_lora_manager.setToolTip("LoRA 관리")
        self.btn_lora_manager.clicked.connect(self._open_lora_manager)
        self.btn_tag_weights = QPushButton("  ⚖ 가중치  ")
        self.btn_tag_weights.setToolTip("가중치 편집")
        self.btn_tag_weights.clicked.connect(self._open_tag_weight_editor)

        for btn in [self.btn_lora_manager, self.btn_tag_weights]:
            btn.setFixedHeight(_btn_h)
            tools_row.addWidget(btn)

        tools_row.addWidget(_sep())

        # 도구
        self.btn_shuffle = QPushButton("  🔀 셔플  ")
        self.btn_shuffle.setToolTip("태그 셔플")
        self.btn_shuffle.clicked.connect(self._shuffle_main_prompt)
        self.btn_ab_test = QPushButton("  A/B  ")
        self.btn_ab_test.setToolTip("A/B 비교")
        self.btn_ab_test.clicked.connect(self._open_ab_test)
        self.btn_random_prompt = QPushButton("  🎲 랜덤  ")
        self.btn_random_prompt.setToolTip("랜덤 프롬프트")
        self.btn_random_prompt.setEnabled(False)

        for btn in [self.btn_shuffle, self.btn_ab_test, self.btn_random_prompt]:
            btn.setFixedHeight(_btn_h)
            tools_row.addWidget(btn)

        tools_row.addStretch()
        vl.addLayout(tools_row)

        # ── 행 3: LoRA 활성 패널 (스택 공간) ──
        from widgets.lora_panel import LoraActivePanel
        self.lora_active_panel = LoraActivePanel()
        self.lora_active_panel.hide()
        vl.addWidget(self.lora_active_panel)

        return bar

    def _create_adetailer_slot_ui(self, title, default_model):
        """ADetailer 슬롯 UI 생성"""
        slot_group = QGroupBox(title)
        slot_group.setCheckable(True)
        slot_layout = QVBoxLayout(slot_group)
        widgets = {}
        
        # Model
        slot_layout.addWidget(QLabel("Model"))
        widgets['model'] = QLineEdit(default_model)
        slot_layout.addWidget(widgets['model'])
        
        # Prompt
        slot_layout.addWidget(QLabel("Prompt"))
        widgets['prompt'] = QTextEdit()
        widgets['prompt'].setFixedHeight(60)
        slot_layout.addWidget(widgets['prompt'])
        
        # Mask Blur & Denoise
        row1_layout = QHBoxLayout()
        widgets['mask_blur'], blur_widget = self._create_param_slider(
            None, "인페인트 마스크 블러", 0, 64, 8, 1
        )
        row1_layout.addWidget(blur_widget)
        
        widgets['denoise'], denoise_widget = self._create_param_slider(
            None, "디노이징 강도", 0.0, 1.0, 0.4, 0.01
        )
        row1_layout.addWidget(denoise_widget)
        slot_layout.addLayout(row1_layout)
        
        # Confidence & Padding
        widgets['confidence'], _ = self._create_param_slider(
            slot_layout, "Detection Confidence", 0.0, 1.0, 0.3, 0.01
        )
        widgets['padding'], _ = self._create_param_slider(
            slot_layout, "Inpaint Padding (px)", 0, 256, 32, 1
        )
        
        # Inpaint Size
        widgets['use_inpaint_size_check'] = QCheckBox("별도의 너비/높이 사용")
        slot_layout.addWidget(widgets['use_inpaint_size_check'])
        
        widgets['inpaint_size_container'] = QWidget()
        s_inpaint_size_layout = QHBoxLayout(widgets['inpaint_size_container'])
        s_inpaint_size_layout.setContentsMargins(20, 0, 0, 0)
        
        widgets['inpaint_width'] = QLineEdit("1024")
        widgets['inpaint_height'] = QLineEdit("1024")
        s_inpaint_size_layout.addWidget(QLabel("ㄴ 너비:"))
        s_inpaint_size_layout.addWidget(widgets['inpaint_width'])
        s_inpaint_size_layout.addWidget(QLabel("높이:"))
        s_inpaint_size_layout.addWidget(widgets['inpaint_height'])
        widgets['inpaint_size_container'].hide()
        slot_layout.addWidget(widgets['inpaint_size_container'])
        
        # Options
        options = [
            ('use_steps_check', '별도의 단계 사용', 'steps', QLineEdit("32")),
            ('use_cfg_check', '별도의 CFG 스케일 사용', 'cfg', QLineEdit("5.0")),
            ('use_checkpoint_check', 'Use separate checkpoint', 
             'checkpoint_combo', NoScrollComboBox()), 
            ('use_vae_check', 'Use separate VAE', 'vae_combo', NoScrollComboBox()), 
        ]
        
        for check_key, text, widget_key, widget in options:
            widgets[check_key] = QCheckBox(text)
            slot_layout.addWidget(widgets[check_key])
            widget.hide()
            widgets[widget_key] = widget
            slot_layout.addWidget(widget)
            
        # Sampler
        widgets['use_sampler_check'] = QCheckBox("별도의 샘플러 사용")
        slot_layout.addWidget(widgets['use_sampler_check'])
        
        widgets['sampler_container'] = QWidget()
        s_sampler_layout = QHBoxLayout(widgets['sampler_container'])
        s_sampler_layout.setContentsMargins(0, 0, 0, 0)
        
        widgets['sampler_combo'] = NoScrollComboBox() 
        widgets['scheduler_combo'] = NoScrollComboBox()
        
        s_sampler_layout.addWidget(widgets['sampler_combo'])
        s_sampler_layout.addWidget(widgets['scheduler_combo'])
        widgets['sampler_container'].hide()
        slot_layout.addWidget(widgets['sampler_container'])
        
        return slot_group, widgets
    
    def _create_viewer_panel(self):
        """뷰어 패널 생성 — QWebEngineView(Vue SPA) 전체 화면"""
        from PyQt6.QtWebEngineWidgets import QWebEngineView
        from PyQt6.QtWebEngineCore import QWebEngineSettings
        from PyQt6.QtWebChannel import QWebChannel
        import os

        # QWebEngineView (Vue 앱) — vue_bridge는 _setup_ui에서 이미 생성됨
        from PyQt6.QtWebEngineCore import QWebEngineScript
        from PyQt6.QtCore import QUrl

        from PyQt6.QtWebEngineCore import QWebEnginePage

        class _DebugPage(QWebEnginePage):
            def javaScriptConsoleMessage(self, level, message, line, source):
                print(f"[Vue JS] {message} (line {line}, {source})")

        self.vue_viewer = QWebEngineView()
        self.vue_viewer.setMinimumSize(400, 400)
        debug_page = _DebugPage(self.vue_viewer)
        self.vue_viewer.setPage(debug_page)

        page = self.vue_viewer.page()
        channel = QWebChannel(page)
        channel.registerObject('backend', self.vue_bridge)
        page.setWebChannel(channel)

        # qwebchannel.js를 스크립트 주입으로 로드 (qrc 프로토콜 대신)
        qwc_script = QWebEngineScript()
        qwc_script.setName("qwebchannel")
        qwc_script.setSourceUrl(QUrl("qrc:///qtwebchannel/qwebchannel.js"))
        qwc_script.setInjectionPoint(QWebEngineScript.InjectionPoint.DocumentCreation)
        qwc_script.setWorldId(QWebEngineScript.ScriptWorldId.MainWorld)
        page.scripts().insert(qwc_script)

        # 로컬 파일 접근 허용
        settings = page.settings()
        settings.setAttribute(QWebEngineSettings.WebAttribute.LocalContentCanAccessFileUrls, True)
        settings.setAttribute(QWebEngineSettings.WebAttribute.LocalContentCanAccessRemoteUrls, True)

        # Vue 빌드 결과 로드
        from PyQt6.QtCore import QUrl
        frontend_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'frontend_dist', 'index.html')
        if os.path.exists(frontend_path):
            self.vue_viewer.setUrl(QUrl.fromLocalFile(frontend_path))
        else:
            self.vue_viewer.setHtml('<h2 style="color:#787878;text-align:center;margin-top:40%">Vue 빌드 필요: cd frontend && npm run build</h2>')

        # 호환성 더미
        class _ViewerLabelProxy:
            def __init__(self, bridge): self._bridge = bridge
            def setPixmap(self, p): pass
            def setText(self, t): pass
            def setAlignment(self, a): pass
            def setFixedHeight(self, h): pass
            def setMinimumSize(self, *a): pass
            def setStyleSheet(self, s): pass
            def hide(self): pass
            def show(self): pass
            def size(self):
                from PyQt6.QtCore import QSize
                return QSize(800, 600)
            def setEnabled(self, e): pass
            def setContextMenuPolicy(self, p): pass
            def setAcceptDrops(self, v): pass
            class _DummySig:
                def connect(self, *a): pass
            customContextMenuRequested = _DummySig()
        self.viewer_label = _ViewerLabelProxy(self.vue_bridge)

        # 더미 (호환성)
        self.gen_progress_bar = type('D', (), {
            'setRange': lambda s,a,b: None, 'setValue': lambda s,v: None,
            'setFixedHeight': lambda s,h: None, 'setTextVisible': lambda s,v: None,
            'setFormat': lambda s,f: None, 'setStyleSheet': lambda s,st: None,
            'hide': lambda s: None, 'show': lambda s: None, 'setMaximum': lambda s,v: None,
        })()
        self.exif_display = type('D', (), {
            'setPlainText': lambda s,t: None, 'setReadOnly': lambda s,v: None,
            'setStyleSheet': lambda s,st: None, 'toPlainText': lambda s: '',
        })()

        return self.vue_viewer
    
    def _create_history_panel(self):
        """히스토리 패널 생성"""
        panel = QWidget()
        panel.setStyleSheet(
            f"background-color: {get_color('bg_primary')}; border-left: 1px solid {get_color('border')};"
        )
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        
        # 헤더
        header = QLabel("히스토리")
        header.setAlignment(Qt.AlignmentFlag.AlignCenter)
        header.setStyleSheet(
            f"padding: 15px; font-weight: bold; color: {get_color('accent')}; "
            f"font-size: 13px; background-color: {get_color('bg_secondary')}; text-transform: uppercase; letter-spacing: 2px;"
        )
        layout.addWidget(header)
        
        # 이전 버튼
        self.btn_history_up = QPushButton("PREV IMAGE")
        self.btn_history_up.clicked.connect(self.select_prev_image)
        self.btn_history_up.setStyleSheet(
            f"background-color: {get_color('bg_button')}; border: none; padding: 8px; color: {get_color('text_secondary')}; font-size: 10px; font-weight: bold;"
        )
        layout.addWidget(self.btn_history_up)


        # 갤러리 스크롤
        self.gallery_scroll_area = QScrollArea()
        self.gallery_scroll_area.setWidgetResizable(True)
        self.gallery_scroll_area.setHorizontalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAlwaysOff
        )
        self.gallery_scroll_area.setStyleSheet(
            "border: none; background: transparent;"
        )
        
        scroll_content = QWidget()
        self.gallery_layout = QVBoxLayout(scroll_content)
        self.gallery_layout.setAlignment(
            Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignHCenter
        )
        self.gallery_layout.setSpacing(10)
        self.gallery_layout.setContentsMargins(10, 10, 10, 10)
        self.gallery_scroll_area.setWidget(scroll_content)
        
        layout.addWidget(self.gallery_scroll_area, 1)  # ← stretch factor 추가!
        
        # 다음 버튼
        self.btn_history_down = QPushButton("▼ 다음 이미지")
        self.btn_history_down.clicked.connect(self.select_next_image)
        self.btn_history_down.setStyleSheet(
            f"background-color: {get_color('bg_button')}; border: none; padding: 8px; color: {get_color('text_secondary')};"
        )
        layout.addWidget(self.btn_history_down)

        # 즐겨찾기 추가 버튼 (토글 아님!)
        self.btn_add_favorite = QPushButton("ADD TO FAVORITES")
        self.btn_add_favorite.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_add_favorite.setFixedHeight(40)
        self.btn_add_favorite.setEnabled(False)  # ← 기본 비활성화!
        self.btn_add_favorite.setStyleSheet(f"""
            QPushButton {{
                background-color: {get_color('bg_button')};
                border: 1px solid {get_color('accent')};
                color: {get_color('accent')};
                font-weight: bold;
                border-radius: 4px;
                font-size: 10px;
            }}
            QPushButton:hover {{
                background-color: {get_color('accent')};
                color: black;
            }}
            QPushButton:disabled {{
                background-color: {get_color('disabled_bg')};
                border: 1px solid {get_color('border')};
                color: {get_color('disabled_text')};
            }}
        """)
        layout.addWidget(self.btn_add_favorite)

        
        # 새로고침 버튼 (클릭 피드백 추가)
        self.btn_refresh_gallery = QPushButton("🔄 목록 새로고침")
        self.btn_refresh_gallery.setFixedHeight(35)
        self.btn_refresh_gallery.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_refresh_gallery.setStyleSheet(f"""
            QPushButton {{
                background-color: {get_color('bg_tertiary')};
                border: none;
                padding: 8px;
                color: {get_color('text_secondary')};
                font-weight: bold;
            }}
            QPushButton:hover {{
                background-color: {get_color('bg_button')};
                color: {get_color('text_primary')};
            }}
            QPushButton:pressed {{
                background-color: {get_color('accent')};
                color: white;
            }}
        """)
        self.btn_refresh_gallery.clicked.connect(self._on_refresh_gallery)
        layout.addWidget(self.btn_refresh_gallery)
        
        return panel
    
    def _on_refresh_gallery(self):
        """갤러리 새로고침 (피드백 포함)"""
        self.btn_refresh_gallery.setText("🔄 새로고침 중...")
        self.btn_refresh_gallery.setEnabled(False)
        
        # 실제 새로고침 수행
        if hasattr(self, 'refresh_gallery'):
            self.refresh_gallery()
        
        # 버튼 복구 (0.5초 후)
        from PyQt6.QtCore import QTimer
        QTimer.singleShot(500, lambda: (
            self.btn_refresh_gallery.setText("🔄 목록 새로고침"),
            self.btn_refresh_gallery.setEnabled(True)
        ))
        
    def _save_prompt_preset(self):
        """현재 프롬프트를 프리셋으로 저장"""
        from PyQt6.QtWidgets import QInputDialog
        from utils.prompt_preset import save_preset, list_presets

        name, ok = QInputDialog.getText(self, "프리셋 저장", "프리셋 이름:")
        if not ok or not name.strip():
            return
        name = name.strip()

        data = {
            "character": self.character_input.text(),
            "copyright": self.copyright_input.text(),
            "artist": self.artist_input.toPlainText(),
            "main_prompt": self.main_prompt_text.toPlainText(),
            "prefix": self.prefix_prompt_text.toPlainText(),
            "suffix": self.suffix_prompt_text.toPlainText(),
            "negative": self.neg_prompt_text.toPlainText(),
        }
        save_preset(name, data)
        QMessageBox.information(self, "저장 완료", f"프리셋 '{name}'이 저장되었습니다.")

    def _load_prompt_preset(self):
        """저장된 프리셋 불러오기 (미리보기 다이얼로그)"""
        from utils.prompt_preset import list_presets
        from widgets.preset_preview_dialog import PresetPreviewDialog

        names = list_presets()
        if not names:
            QMessageBox.information(self, "프리셋", "저장된 프리셋이 없습니다.")
            return

        dlg = PresetPreviewDialog(self)
        if dlg.exec() != dlg.DialogCode.Accepted:
            return

        data = dlg.get_result()
        if not data:
            return

        _field_map = {
            "character":    lambda v: self.character_input.setText(v),
            "copyright":    lambda v: self.copyright_input.setText(v),
            "artist":       lambda v: self.artist_input.setPlainText(v),
            "main_prompt":  lambda v: self.main_prompt_text.setPlainText(v),
            "prefix":       lambda v: self.prefix_prompt_text.setPlainText(v),
            "suffix":       lambda v: self.suffix_prompt_text.setPlainText(v),
            "negative":     lambda v: self.neg_prompt_text.setPlainText(v),
        }
        for key, setter in _field_map.items():
            if key in data:
                setter(data[key])

        self.show_status("프리셋 적용됨")

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

    def _adjust_total_prompt_height(self):
        """최종 프롬프트 칸 내용에 맞춰 높이 자동 조절"""
        doc = self.total_prompt_display.document()
        doc_height = int(doc.size().height()) + 10  # 여백
        new_h = max(60, min(doc_height, 600))
        current_h = self.total_prompt_display.height()
        # 높이 차이가 3px 이상일 때만 업데이트 (진동 방지)
        if abs(current_h - new_h) > 3:
            self.total_prompt_display.setFixedHeight(new_h)

    def _adjust_artist_height(self):
        """작가 입력칸 내용에 맞춰 높이 자동 조절"""
        doc = self.artist_input.document()
        doc_height = int(doc.size().height()) + 10
        new_h = max(60, min(doc_height, 200))
        current_h = self.artist_input.height()
        if abs(current_h - new_h) > 3:
            self.artist_input.setFixedHeight(new_h)

    def _create_group(self, parent_layout, title, widget_or_layout):
        """그룹 생성 헬퍼"""
        parent_layout.addWidget(QLabel(title))
        if isinstance(widget_or_layout, QWidget):
            parent_layout.addWidget(widget_or_layout)
            return widget_or_layout
        elif isinstance(widget_or_layout, QHBoxLayout):
            parent_layout.addLayout(widget_or_layout)
            return widget_or_layout
    
    def _create_param_slider(self, parent_layout, name, min_val, max_val, 
                            default_val, step):
        """파라미터 슬라이더 생성"""
        is_float = isinstance(step, float)
        multiplier = int(1 / step) if is_float else 1
        
        container = QWidget()
        layout = QHBoxLayout(container)
        layout.setContentsMargins(0, 0, 0, 0)
        
        from PyQt6.QtWidgets import QSlider
        slider = QSlider(Qt.Orientation.Horizontal)
        slider.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        slider.wheelEvent = lambda event: event.ignore()
        slider.setRange(int(min_val * multiplier), int(max_val * multiplier))
        
        num_input = QLineEdit(
            f"{default_val:.2f}" if is_float else str(default_val)
        )
        num_input.setFixedWidth(60)
        num_input.setAlignment(Qt.AlignmentFlag.AlignCenter)
        
        def update_input():
            value = slider.value() / multiplier
            num_input.setText(
                f"{value:.2f}" if is_float else str(int(value))
            )
            
        def update_slider():
            try:
                value = float(num_input.text())
                slider.setValue(int(value * multiplier))
            except ValueError: 
                pass
            
        slider.valueChanged.connect(update_input)
        num_input.editingFinished.connect(update_slider)

        # 슬라이더 참조 저장 (load_settings 시 동기화용)
        num_input._slider = slider
        num_input._multiplier = multiplier

        slider.setValue(int(default_val * multiplier))
        if hasattr(self, 'wheel_filter'):
            slider.installEventFilter(self.wheel_filter)
        
        layout.addWidget(slider)
        layout.addWidget(num_input)

        if parent_layout is not None:
            if name: 
                parent_layout.addWidget(QLabel(name))
            parent_layout.addWidget(container)
            return num_input, container
        else:
            return num_input, container
    
    def _create_separator(self):
        """구분선 생성"""
        separator = QFrame()
        separator.setFrameShape(QFrame.Shape.HLine)
        separator.setFrameShadow(QFrame.Shadow.Sunken)
        return separator
    
    def _make_hbox(self, widgets):
        """HBox 컨테이너 생성"""
        container = QWidget()
        layout = QHBoxLayout(container)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(5)
        for w in widgets:
            layout.addWidget(w)
        return container

    def _on_prefix_toggle(self, checked):
        """선행 프롬프트 토글"""
        self.prefix_prompt_text.setVisible(checked)
        self.prefix_toggle_button.setText(
            "▼ 선행 고정 프롬프트" if checked else "▶ 선행 고정 프롬프트"
        )
    
    def _on_suffix_toggle(self, checked):
        """후행 프롬프트 토글"""
        self.suffix_prompt_text.setVisible(checked)
        self.suffix_toggle_button.setText(
            "▼ 후행 고정 프롬프트" if checked else "▶ 후행 고정 프롬프트"
        )
    
    def _on_neg_toggle(self, checked):
        """네거티브 프롬프트 토글"""
        self.neg_prompt_text.setVisible(checked)
        self.neg_toggle_button.setText(
            "▼ 부정 프롬프트 (Negative)" if checked else "▶ 부정 프롬프트 (Negative)"
        )

    def _on_exclude_toggle(self, checked):
        """제외 프롬프트 토글"""
        self.exclude_prompt_local_input.setVisible(checked)
        self.exclude_toggle_button.setText(
            "▼ 제외 프롬프트 (Local)" if checked else "▶ 제외 프롬프트 (Local)"
        )
        
    def _on_res_preset_context(self, idx: int, btn):
        """해상도 프리셋 우클릭 메뉴"""
        from PyQt6.QtWidgets import QMenu, QDialog, QDialogButtonBox, QSpinBox
        menu = QMenu(self)
        menu.setStyleSheet(
            f"QMenu {{ background-color: {get_color('bg_button')}; color: {get_color('text_primary')}; border: 1px solid {get_color('border')}; }}"
            f"QMenu::item {{ padding: 6px 16px; }}"
            f"QMenu::item:selected {{ background-color: {get_color('accent')}; }}"
        )
        act_edit = menu.addAction("✏️ 해상도 변경")
        act_reset = menu.addAction("↩️ 기본값 복원")
        chosen = menu.exec(btn.mapToGlobal(btn.rect().bottomLeft()))
        if not chosen:
            return
        if chosen == act_edit:
            dlg = QDialog(self)
            dlg.setWindowTitle("해상도 변경")
            dlg.setFixedSize(280, 120)
            dlg.setStyleSheet(f"background-color: {get_color('bg_secondary')}; color: {get_color('text_primary')};")
            dl = QVBoxLayout(dlg)
            row = QHBoxLayout()
            w_spin = QSpinBox()
            w_spin.setRange(64, 4096)
            w_spin.setSingleStep(64)
            w_spin.setValue(self._res_presets[idx][1])
            w_spin.setStyleSheet(f"background:{get_color('bg_button')}; color:{get_color('text_primary')}; border:1px solid {get_color('border')}; padding:4px;")
            h_spin = QSpinBox()
            h_spin.setRange(64, 4096)
            h_spin.setSingleStep(64)
            h_spin.setValue(self._res_presets[idx][2])
            h_spin.setStyleSheet(f"background:{get_color('bg_button')}; color:{get_color('text_primary')}; border:1px solid {get_color('border')}; padding:4px;")
            swap_btn = QPushButton("⇄")
            swap_btn.setFixedSize(32, 32)
            swap_btn.setToolTip("W ↔ H 교환")
            swap_btn.setStyleSheet(
                f"background:{get_color('accent')}; color:white; border:none; "
                f"border-radius:4px; font-weight:bold; font-size:16px;"
            )
            swap_btn.clicked.connect(lambda: (
                w_spin.setValue(h_spin.value()) or True) if (
                    _tw := w_spin.value()) and (w_spin.setValue(h_spin.value()) or True) and h_spin.setValue(_tw) is None else None
            )
            # simpler swap
            def _swap_wh():
                _w, _h = w_spin.value(), h_spin.value()
                w_spin.setValue(_h)
                h_spin.setValue(_w)
            swap_btn.clicked.disconnect()
            swap_btn.clicked.connect(_swap_wh)
            row.addWidget(QLabel("W:"))
            row.addWidget(w_spin)
            row.addWidget(swap_btn)
            row.addWidget(QLabel("H:"))
            row.addWidget(h_spin)
            dl.addLayout(row)
            bb = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
            bb.accepted.connect(dlg.accept)
            bb.rejected.connect(dlg.reject)
            dl.addWidget(bb)
            if dlg.exec() == QDialog.DialogCode.Accepted:
                w, h = w_spin.value(), h_spin.value()
                label = f"{w}x{h}" if w != h else f"{w}\u00b2"
                self._res_presets[idx] = [label, w, h]
                btn.setText(label)
        elif chosen == act_reset:
            if idx < len(self._DEFAULT_RES_PRESETS):
                default = list(self._DEFAULT_RES_PRESETS[idx])
                self._res_presets[idx] = default
                btn.setText(default[0])

    def _swap_resolution(self):
        """W ↔ H 해상도 교환"""
        w, h = self.width_input.text(), self.height_input.text()
        self.width_input.setText(h)
        self.height_input.setText(w)

    def _open_lora_manager(self):
        """LoRA 브라우저 다이얼로그 열기"""
        from widgets.lora_manager import LoraManagerDialog
        from backends import get_backend
        try:
            backend = get_backend()
        except Exception:
            backend = None
        dlg = LoraManagerDialog(backend=backend, parent=self)
        dlg.lora_inserted.connect(self._on_lora_inserted)
        dlg.loras_batch_inserted.connect(self._on_lora_batch_inserted)
        dlg.exec()

    def _on_lora_inserted(self, lora_text: str):
        """LoRA를 활성 패널에 추가 + Vue로 전달"""
        import re, json as _json
        # 트리거 워드 분리 (||TRIGGER:[...] 포맷)
        trigger_words = []
        if '||TRIGGER:' in lora_text:
            parts = lora_text.split('||TRIGGER:', 1)
            lora_text = parts[0]
            try:
                trigger_words = _json.loads(parts[1])
            except Exception:
                pass
        m = re.match(r'<lora:(.+?):([-\d.]+)>', lora_text)
        if m:
            name, weight = m.group(1), float(m.group(2))
            self.lora_active_panel.add_lora(name, weight)
            # Vue LoRA Stack으로 전달 (트리거 워드 포함)
            if hasattr(self, 'vue_bridge'):
                payload = {'name': name, 'weight': weight}
                if trigger_words:
                    payload['trigger_words'] = trigger_words
                self.vue_bridge.loraInserted.emit(_json.dumps(payload))

    def _on_lora_batch_inserted(self, text: str):
        """다이얼로그에서 일괄 붙여넣기된 LoRA 텍스트를 패널에 추가"""
        import re
        self.lora_active_panel.parse_and_add_loras(text)
        # Vue로도 전달
        if hasattr(self, 'vue_bridge'):
            import json as _json
            for m in re.finditer(r'<lora:(.+?):([-\d.]+)>', text):
                self.vue_bridge.loraInserted.emit(_json.dumps({'name': m.group(1), 'weight': float(m.group(2))}))

    def _update_token_count(self):
        """최종 프롬프트 토큰 수 추정 (CLIP 기준 근사)"""
        import re
        text = self.total_prompt_display.toPlainText().strip()

        if not text:
            self.token_count_label.setText("토큰: 0 / 75")
            self.token_count_label.setStyleSheet(
                f"color: {get_color('text_muted')}; font-size: 11px; font-weight: bold; padding: 0 4px;"
            )
            return

        # CLIP 토큰 근사: 단어/서브워드 기준 (영어 ~0.75 토큰/단어, 태그 ~1 토큰/태그)
        tags = [t.strip() for t in text.split(",") if t.strip()]
        token_est = 0
        for tag in tags:
            words = re.findall(r'[a-zA-Z]+|[^ ,():\[\]]+', tag)
            token_est += max(1, len(words))

        if token_est <= 75:
            color = "#4CAF50"
        elif token_est <= 150:
            color = "#FFA726"
        else:
            color = "#E74C3C"
        self.token_count_label.setText(f"토큰: ~{token_est} / 75")
        self.token_count_label.setStyleSheet(
            f"color: {color}; font-size: 11px; font-weight: bold; padding: 0 4px;"
        )

    def _open_tag_weight_editor(self):
        """태그 가중치 슬라이더 편집"""
        from widgets.tag_weight_editor import TagWeightEditorDialog
        text = self.main_prompt_text.toPlainText().strip()
        if not text:
            return
        dlg = TagWeightEditorDialog(text, parent=self)
        if dlg.exec() == dlg.DialogCode.Accepted:
            result = dlg.get_result()
            if result is not None:
                self.main_prompt_text.setPlainText(result)

    def _open_ab_test(self):
        """A/B 프롬프트 비교 테스트"""
        from widgets.ab_test_dialog import ABTestDialog
        prompt = self.main_prompt_text.toPlainText().strip()
        negative = self.neg_prompt_text.toPlainText().strip()
        dlg = ABTestDialog(prompt, negative, parent=self)
        if dlg.exec() != dlg.DialogCode.Accepted:
            return
        result = dlg.get_result()
        if not result:
            return
        # 프롬프트 A를 대기열에, 프롬프트 B를 대기열에 — 같은 시드
        for label, key in [("A", "prompt_a"), ("B", "prompt_b")]:
            payload = {
                'prompt': result[key],
                'negative_prompt': result['negative'],
                'seed': result['seed'],
            }
            if hasattr(self, 'queue_panel'):
                self.queue_panel.add_single_item(payload)
        self.show_status(f"A/B 테스트 대기열 추가 (시드: {result['seed']})")

    def _shuffle_main_prompt(self):
        """메인 프롬프트 태그 순서 랜덤 셔플"""
        import random
        text = self.main_prompt_text.toPlainText().strip()
        if not text:
            return
        tags = [t.strip() for t in text.split(",") if t.strip()]
        random.shuffle(tags)
        self.main_prompt_text.setPlainText(", ".join(tags))

    def _insert_fav_tag(self, tags: str):
        """즐겨찾기 태그를 메인 프롬프트에 삽입"""
        current = self.main_prompt_text.toPlainText().strip()
        if current:
            self.main_prompt_text.setPlainText(f"{current}, {tags}")
        else:
            self.main_prompt_text.setPlainText(tags)

    def _open_character_preset(self):
        """캐릭터 특징 프리셋 다이얼로그 열기"""
        # 기존 태그 수집 (중복 표시용) — 이스케이프/비이스케이프 모두 등록
        existing: set[str] = set()
        for src in (self.main_prompt_text.toPlainText(),
                    self.prefix_prompt_text.toPlainText(),
                    self.suffix_prompt_text.toPlainText(),
                    self.character_input.text()):
            for t in src.split(","):
                norm = t.strip().lower().replace("_", " ")
                if norm:
                    existing.add(norm)
                    # 이스케이프 제거 버전도 등록
                    unesc = norm.replace(r"\(", "(").replace(r"\)", ")")
                    if unesc != norm:
                        existing.add(unesc)

        current_char = self.character_input.text().strip()
        dlg = CharacterPresetDialog(
            existing_tags=existing,
            current_character=current_char,
            parent=self
        )
        if dlg.exec() != dlg.DialogCode.Accepted:
            return

        result = dlg.get_result()
        if not result:
            return

        # 캐릭터 이름 설정
        char_name = result.get("character_name", "")
        if char_name:
            cur = self.character_input.text().strip()
            if cur:
                existing_chars = set()
                for c in cur.split(","):
                    n = c.strip().lower().replace("_", " ")
                    existing_chars.add(n)
                    existing_chars.add(
                        n.replace(r"\(", "(").replace(r"\)", ")")
                    )
                if char_name.lower().replace("_", " ") not in existing_chars:
                    self.character_input.setText(f"{cur}, {char_name}")
            else:
                self.character_input.setText(char_name)

        # 특징 태그 삽입 (중복 제거)
        tags = result.get("tags", [])
        if tags:
            # 삽입 시점의 전체 태그 재수집
            all_existing: set[str] = set()
            for src in (self.main_prompt_text.toPlainText(),
                        self.prefix_prompt_text.toPlainText(),
                        self.suffix_prompt_text.toPlainText(),
                        self.character_input.text()):
                for t in src.split(","):
                    n = t.strip().lower().replace("_", " ")
                    if n:
                        all_existing.add(n)
                        all_existing.add(
                            n.replace(r"\(", "(").replace(r"\)", ")")
                        )

            new_tags = [
                t for t in tags
                if t.strip().lower().replace("_", " ") not in all_existing
            ]
            if new_tags:
                insert_str = ", ".join(new_tags)
                current = self.main_prompt_text.toPlainText().strip()
                if current:
                    self.main_prompt_text.setPlainText(
                        f"{insert_str}, {current}"
                    )
                else:
                    self.main_prompt_text.setPlainText(insert_str)

    def _create_favorites_tab(self):
        """즐겨찾기 탭 생성"""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(10)
        
        # 헤더
        header_layout = QHBoxLayout()
        header_label = QLabel("⭐ 즐겨찾기 목록")
        header_label.setStyleSheet(f"font-size: 16px; font-weight: bold; color: {get_color('accent')};")
        header_layout.addWidget(header_label)
        header_layout.addStretch()
        
        self.btn_fav_refresh = QPushButton("🔄 새로고침")
        self.btn_fav_refresh.clicked.connect(self.refresh_favorites)
        self.btn_fav_refresh.setStyleSheet(
            f"padding: 5px 10px; background-color: {get_color('bg_button')}; border-radius: 4px;"
        )
        header_layout.addWidget(self.btn_fav_refresh)
        
        self.btn_fav_clear = QPushButton("🗑️ 전체 삭제")
        self.btn_fav_clear.clicked.connect(self.clear_all_favorites)
        self.btn_fav_clear.setStyleSheet(
            "padding: 5px 10px; background-color: #8B0000; color: white; border-radius: 4px;"
        )
        header_layout.addWidget(self.btn_fav_clear)
        
        layout.addLayout(header_layout)
        
        # 스크롤 영역 (썸네일 그리드)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setStyleSheet(f"border: none; background: {get_color('bg_primary')};")
        
        scroll_content = QWidget()
        scroll_content_layout = QVBoxLayout(scroll_content)
        scroll_content_layout.setContentsMargins(10, 10, 10, 10)
        scroll_content_layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        
        # 플로우 레이아웃 (한 줄에 5개씩)
        self.fav_flow_widget = QWidget()
        self.fav_flow_layout = FlowLayout(self.fav_flow_widget)
        self.fav_flow_layout.setSpacing(10)
        
        scroll_content_layout.addWidget(self.fav_flow_widget)
        scroll_content_layout.addStretch()
        
        scroll.setWidget(scroll_content)
        layout.addWidget(scroll)
        
        return tab
