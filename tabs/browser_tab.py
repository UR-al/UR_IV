# tabs/browser_tab.py
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLineEdit
)
from PyQt6.QtCore import QUrl
from PyQt6.QtWebEngineWidgets import QWebEngineView
from PyQt6.QtWebEngineCore import QWebEngineSettings, QWebEnginePage


class _QuietPage(QWebEnginePage):
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
        
        # 웹뷰
        self.web_view = QWebEngineView()
        self.web_view.setPage(_QuietPage(self.web_view))

        page = self.web_view.page()
        profile = page.profile()
        
        # User Agent 설정
        new_user_agent = (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/120.0.0.0 Safari/537.36"
        )
        profile.setHttpUserAgent(new_user_agent)
        
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
        
        # 캐시 경로 설정
        import os
        from config import CURRENT_DIR
        cache_path = os.path.join(CURRENT_DIR, 'web_cache')
        profile.setCachePath(cache_path)
        profile.setPersistentStoragePath(cache_path)
        
        self.web_view.setUrl(QUrl(self.default_url))
        layout.addWidget(self.web_view)
        
        self.web_view.urlChanged.connect(
            lambda u: self.url_input.setText(u.toString())
        )

    def set_home_url(self, url):
        """홈 URL 설정"""
        if url:
            self.home_url = url
            self.url_input.setText(url)
            
    def get_home_url(self):
        """홈 URL 반환"""
        return self.home_url        

    def go_home(self):
        """홈으로 이동"""
        self.web_view.setUrl(QUrl(self.default_url))

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