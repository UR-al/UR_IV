# tabs/browser_tab.py
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLineEdit
)
from PyQt6.QtCore import QUrl
from PyQt6.QtWebEngineWidgets import QWebEngineView
from PyQt6.QtWebEngineCore import QWebEngineSettings, QWebEnginePage, QWebEngineProfile
from ui.native_dialogs import ThemedWebDialogs

#: Web 탭 전용 Chromium 프로필 이름 — 이름이 있어야 디스크에 남는다(빈 이름 = off-the-record).
WEB_TAB_PROFILE_NAME = "web_tab"


def web_tab_profile_paths():
    """(영속 저장소, HTTP 캐시) 경로.

    쿠키·localStorage·로그인은 사용자 데이터(user_data/web_tab_profile)에, 다시 받을 수 있는
    HTTP 캐시는 cache/web_tab 에 둔다 — core.storage_paths 의 저장 경계를 따른다.
    """
    from core.storage_paths import storage_paths

    return (
        str(storage_paths.user_data_dir / "web_tab_profile"),
        str(storage_paths.cache_dir / "web_tab"),
    )


class _QuietPage(ThemedWebDialogs, QWebEnginePage):
    """외부 사이트의 JS 콘솔 잡음(CSP, Permissions-Policy, 코덱 등) 억제"""
    def javaScriptConsoleMessage(self, level, message, line, source):
        pass


class BrowserTab(QWidget):
    """내장 웹 브라우저 탭"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.default_url = "https://hijiribe.donmai.us/"
        self.home_url = self.default_url

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        
        # 네비게이션 바
        nav_bar = QHBoxLayout()
        nav_bar.setContentsMargins(5, 5, 5, 5)
        
        btn_back = QPushButton("◀")
        btn_back.setFixedWidth(40)
        btn_back.clicked.connect(self.go_back)
        
        btn_home = QPushButton("🏠") 
        btn_home.setFixedWidth(40)
        btn_home.clicked.connect(self.go_home)
        
        self.url_input = QLineEdit()
        self.url_input.setPlaceholderText("URL 입력 (예: https://google.com)")
        self.url_input.returnPressed.connect(self.navigate_to_url)
        
        btn_go = QPushButton("이동")
        btn_go.setFixedWidth(60)
        btn_go.clicked.connect(self.navigate_to_url)
        
        btn_vue_home = QPushButton("← AI Studio")
        btn_vue_home.setFixedWidth(100)
        btn_vue_home.clicked.connect(self._go_vue_home)

        nav_bar.addWidget(btn_vue_home)
        nav_bar.addWidget(btn_back)
        nav_bar.addWidget(btn_home)
        nav_bar.addWidget(self.url_input)
        nav_bar.addWidget(btn_go)
        layout.addLayout(nav_bar)
        
        # 웹뷰 — 전용 영속 프로필을 쓴다. 프로필 없이 만든 페이지는 Qt 6 의 off-the-record
        # 기본 프로필을 써서 캐시/저장 경로 설정이 무시되고, 쿠키·localStorage(로그인)가
        # 메모리에만 남아 재시작마다 사라졌다.
        storage_path, cache_path = web_tab_profile_paths()
        profile = QWebEngineProfile(WEB_TAB_PROFILE_NAME, self)
        profile.setHttpUserAgent(
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/120.0.0.0 Safari/537.36"
        )
        profile.setPersistentStoragePath(storage_path)
        profile.setCachePath(cache_path)
        profile.setPersistentCookiesPolicy(
            QWebEngineProfile.PersistentCookiesPolicy.ForcePersistentCookies
        )
        self._profile = profile

        self.web_view = QWebEngineView()
        self.web_view.setPage(_QuietPage(profile, self.web_view))

        # 웹 설정
        settings = self.web_view.settings()
        settings.setAttribute(QWebEngineSettings.WebAttribute.LocalStorageEnabled, True)
        settings.setAttribute(QWebEngineSettings.WebAttribute.JavascriptEnabled, True)
        settings.setAttribute(QWebEngineSettings.WebAttribute.PluginsEnabled, True)
        settings.setAttribute(
            QWebEngineSettings.WebAttribute.JavascriptCanOpenWindows, True
        )
        settings.setAttribute(
            QWebEngineSettings.WebAttribute.AllowRunningInsecureContent, True
        )
        settings.setAttribute(QWebEngineSettings.WebAttribute.AutoLoadImages, True)

        # 첫 URL 은 탭을 열 때 싣는다 — 앱을 켤 때마다 외부 사이트가 뒤에서 뜨면
        # 백엔드를 켠 것과 무관하게 '웹이 따로 켜진' 것처럼 보인다.
        self._loaded_once = False
        layout.addWidget(self.web_view)
        
        self.web_view.urlChanged.connect(
            lambda u: self.url_input.setText(u.toString())
        )
        self.apply_theme()

    def apply_theme(self):
        """Refresh native navigation colors without navigating the web page."""
        from PyQt6.QtGui import QColor
        from utils.theme_manager import get_theme_manager
        theme = get_theme_manager()
        self.setStyleSheet(theme.get_stylesheet())
        self.web_view.page().setBackgroundColor(QColor(theme.get_colors()['bg_primary']))

    def ensure_loaded(self):
        """탭을 열 때 부른다. 처음 한 번만 홈 URL 을 싣는다."""
        if not getattr(self, '_loaded_once', False):
            self._loaded_once = True
            self.web_view.setUrl(QUrl(self.home_url))

    def set_home_url(self, url):
        """홈 URL 설정"""
        if url:
            self.home_url = url
            self.url_input.setText(url)
            
    def get_home_url(self):
        """홈 URL 반환"""
        return self.home_url        

    def go_home(self):
        """홈으로 이동 (설정된 home_url 우선, 없으면 default_url)"""
        self.web_view.setUrl(QUrl(getattr(self, 'home_url', '') or self.default_url))

    def go_back(self):
        """뒤로 가기"""
        self.web_view.back()

    def _go_vue_home(self):
        """Vue SPA(AI Studio)로 복귀"""
        parent = self.parent()
        while parent:
            if hasattr(parent, '_main_stack'):
                parent._main_stack.setCurrentIndex(0)
                return
            parent = parent.parent()

    def navigate_to_url(self):
        """URL로 이동"""
        url = self.url_input.text().strip()
        if not url: 
            return
        if not url.startswith("http"):
            url = "https://" + url
        self.web_view.setUrl(QUrl(url))
