# tabs/backend_ui_tab.py
"""백엔드 UI 확인 탭 — WebUI / ComfyUI 웹 인터페이스를 임베디드 웹뷰로 표시"""
import os
import json
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel, QLineEdit
)
from PyQt6.QtCore import QUrl, Qt, QTimer
from PyQt6.QtWebEngineWidgets import QWebEngineView
from PyQt6.QtWebEngineCore import QWebEngineSettings, QWebEngineProfile, QWebEnginePage

from config import CURRENT_DIR
from utils.theme_manager import get_color
from ui.native_dialogs import ThemedWebDialogs


class _QuietPage(ThemedWebDialogs, QWebEnginePage):
    """JS 콘솔 경고 억제 페이지"""
    def javaScriptConsoleMessage(self, level, message, line, source):
        pass


# Chromium net::ERR_ABORTED — 새로고침·다른 URL 로의 이동이 진행 중인 로드를 끊었을 때.
_ERR_ABORTED = -3


def load_outcome(status_name: str, error_code: int = 0):
    """``loadingChanged`` 의 로드 상태 → ``'succeeded'`` / ``'failed'`` / ``None``(무시).

    ``loadFinished(False)`` 는 진짜 실패와 '중단'을 가리지 않는다. 첫 로드 중에 🔄 를 누르거나
    백엔드 전환으로 새 페이지를 실으면 진행 중이던 로드가 ERR_ABORTED 로 끝나는데, 그걸 실패로
    보면 살아 있는 백엔드 UI 를 '불러오지 못했습니다' 안내로 덮고 새 로드까지 취소했다.
    - LoadStartedStatus: 아직 끝나지 않았다.
    - LoadStoppedStatus, 또는 ERR_ABORTED 로 끝난 LoadFailedStatus: 다른 이동이 끊은 것 — 무시.
    """
    if status_name == 'LoadSucceededStatus':
        return 'succeeded'
    try:
        code = int(error_code)
    except (TypeError, ValueError):
        code = 0
    if status_name == 'LoadFailedStatus' and code != _ERR_ABORTED:
        return 'failed'
    return None


class BackendUITab(QWidget):
    """백엔드 UI 확인 탭"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._current_url = ""
        self._load_workflow_after = False
        self._workflow_retry_count = 0

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        # ── 상단 바 ──
        top_bar = QHBoxLayout()
        top_bar.setContentsMargins(8, 6, 8, 6)
        top_bar.setSpacing(6)

        btn_vue_home = QPushButton("← AI Studio")
        btn_vue_home.setFixedWidth(100)
        btn_vue_home.clicked.connect(self._go_vue_home)
        top_bar.addWidget(btn_vue_home)

        self._status_label = QLabel("백엔드 UI")
        self._status_label.setStyleSheet(
            f"color: {get_color('text_secondary')}; font-weight: bold; font-size: 13px;"
        )
        top_bar.addWidget(self._status_label)

        top_bar.addStretch()

        self._url_display = QLineEdit()
        self._url_display.setReadOnly(True)
        self._url_display.setStyleSheet(
            f"background: {get_color('bg_secondary')}; color: {get_color('text_muted')}; border: 1px solid {get_color('bg_button_hover')}; "
            "border-radius: 4px; padding: 3px 8px; font-size: 11px;"
        )
        self._url_display.setFixedWidth(300)
        top_bar.addWidget(self._url_display)

        btn_reload = QPushButton("🔄")
        btn_reload.setFixedSize(36, 32)
        btn_reload.setToolTip("새로고침")
        btn_reload.clicked.connect(self._reload)
        top_bar.addWidget(btn_reload)

        btn_open = QPushButton("🌐")
        btn_open.setFixedSize(36, 32)
        btn_open.setToolTip("외부 브라우저로 열기")
        btn_open.clicked.connect(self._open_external)
        top_bar.addWidget(btn_open)

        layout.addLayout(top_bar)
        self._layout = layout

        # ── 웹뷰 — 처음 표시될 때(ensure_loaded) 만든다 ──
        # 숨은 탭이 시작 시 프로필·웹뷰를 만들고 placeholder 를 setHtml 하면 보이지도 않는
        # 페이지 때문에 렌더러 프로세스가 뜬다(탭을 여는 순간 _recreate_page 가 교체한다).
        self._web_view = None
        self._profile = None
        self._page = None
        self._showing_placeholder = False
        self.apply_theme()

    def _ensure_web_view(self):
        """백엔드 전용 프로필·웹뷰를 한 번만 만든다."""
        if self._web_view is not None:
            return self._web_view
        from PyQt6.QtGui import QColor

        self._web_view = QWebEngineView()
        self._profile = QWebEngineProfile("backend_ui", self)
        self._profile.setHttpUserAgent(
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/120.0.0.0 Safari/537.36"
        )
        cache_path = os.path.join(CURRENT_DIR, 'backend_ui_cache')
        self._profile.setCachePath(cache_path)
        self._profile.setPersistentStoragePath(cache_path)

        self._page = _QuietPage(self._profile, self._web_view)
        self._web_view.setPage(self._page)
        self._apply_web_settings()
        self._page.setBackgroundColor(QColor(get_color('bg_primary')))

        # URL 변경 추적 (한 번만 연결)
        self._web_view.urlChanged.connect(self._on_url_changed)
        self._layout.addWidget(self._web_view, 1)
        return self._web_view

    def apply_theme(self):
        """Refresh existing chrome, including formerly frozen inline colors."""
        from PyQt6.QtGui import QColor
        from utils.theme_manager import get_theme_manager
        theme = get_theme_manager()
        colors = theme.get_colors()
        self.setStyleSheet(theme.get_stylesheet())
        self._status_label.setStyleSheet(
            f"color: {colors['text_secondary']}; font-weight: bold; font-size: 13px;")
        self._url_display.setStyleSheet(
            f"background: {colors['bg_input']}; color: {colors['text_secondary']}; "
            f"border: 1px solid {colors['border']}; border-radius: 4px; padding: 3px 8px; font-size: 11px;")
        # 웹뷰는 처음 표시될 때 만든다 — 아직 없으면 배경색만 건너뛴다. placeholder 를 여기서
        # setHtml 하지 않는다(숨은 탭의 렌더러를 띄우고, 탭을 여는 순간 교체돼 보이지도 않는다).
        view = getattr(self, '_web_view', None)
        if view is not None:
            view.page().setBackgroundColor(QColor(colors['bg_primary']))

    def _on_url_changed(self, url: QUrl):
        """URL 변경 시 표시 업데이트 (로드 실패 안내 페이지의 about:blank 는 표시하지 않는다)"""
        if self._showing_placeholder and url.toString() in ('', 'about:blank'):
            return
        self._url_display.setText(url.toString())

    def _apply_web_settings(self):
        """웹뷰 설정 적용"""
        settings = self._web_view.settings()
        settings.setAttribute(QWebEngineSettings.WebAttribute.LocalStorageEnabled, True)
        settings.setAttribute(QWebEngineSettings.WebAttribute.JavascriptEnabled, True)
        settings.setAttribute(QWebEngineSettings.WebAttribute.PluginsEnabled, True)
        settings.setAttribute(QWebEngineSettings.WebAttribute.JavascriptCanOpenWindows, True)
        settings.setAttribute(QWebEngineSettings.WebAttribute.AllowRunningInsecureContent, True)
        settings.setAttribute(QWebEngineSettings.WebAttribute.AutoLoadImages, True)

    def _recreate_page(self):
        """새 페이지를 생성하여 이전 SPA/JS 상태를 완전히 제거"""
        old_page = self._page

        new_page = _QuietPage(self._profile, self._web_view)
        # 새 페이지도 테마 배경으로 시작한다(로드 전 흰 화면 번쩍임 방지).
        from PyQt6.QtGui import QColor
        new_page.setBackgroundColor(QColor(get_color('bg_primary')))
        self._web_view.setPage(new_page)
        self._page = new_page
        self._apply_web_settings()

        # 이전 페이지 정리
        if old_page is not None:
            try:
                old_page.deleteLater()
            except RuntimeError:
                pass

    # ── 공개 API ──

    def _go_vue_home(self):
        """Vue SPA로 복귀"""
        parent = self.parent()
        while parent:
            if hasattr(parent, '_main_stack'):
                parent._main_stack.setCurrentIndex(0)
                return
            parent = parent.parent()

    def mark_backend_changed(self):
        """백엔드가 바뀌었다 — 탭이 보이는 중이면 바로 갈아입히고, 아니면 다음에 열 때."""
        self._stale = True
        if self.isVisible():
            self.ensure_loaded()

    def ensure_loaded(self):
        """탭을 열 때 부른다. 로드가 필요할 때만 실제로 로드한다."""
        if getattr(self, '_stale', True) or not getattr(self, '_current_url', ''):
            self._stale = False
            self.load_backend_ui()

    def load_backend_ui(self):
        """현재 백엔드에 맞는 UI를 로드"""
        from backends import get_backend, get_backend_type, BackendType

        backend = get_backend()
        backend_type = get_backend_type()

        if backend_type == BackendType.COMFYUI:
            url = backend.api_url
            self._status_label.setText("🟣 ComfyUI")
            self._load_url(url, load_workflow=True)
        else:
            url = backend.api_url
            self._status_label.setText("🟠 WebUI")
            self._load_url(url)

    def _load_url(self, url: str, load_workflow: bool = False):
        """URL을 웹뷰에 로드 (페이지를 새로 생성하여 이전 상태 완전 제거)"""
        self._ensure_web_view()
        self._current_url = url
        self._url_display.setText(url)
        self._load_workflow_after = load_workflow
        self._workflow_retry_count = 0
        self._showing_placeholder = False

        # 페이지를 새로 생성하여 이전 JS/SPA 완전 제거
        self._recreate_page()

        # 로드 상태 연결 — 이 페이지를 붙잡은 연결이라, 교체된 옛 페이지가 늦게 보내는 결과는
        # _on_page_loaded 가 걸러 낸다(옛 페이지는 deleteLater 로 연결째 사라진다).
        page = self._page
        page.loadingChanged.connect(
            lambda info, page=page: self._on_loading_changed(page, info))

        # URL 로드
        self._web_view.setUrl(QUrl(url))

    def _on_loading_changed(self, page, info):
        """QWebEngineLoadingInfo → 로드 결과. 상태·오류 코드를 읽지 못하면 무시한다."""
        try:
            status = info.status()
            status_name = getattr(status, 'name', str(status))
            error_code = info.errorCode()
        except Exception:
            return
        self._on_page_loaded(page, load_outcome(status_name, error_code))

    def _on_page_loaded(self, page, outcome):
        """지금 페이지의 로드가 끝났다 — outcome: 'succeeded' / 'failed' / None(무시)."""
        if page is not self._page or outcome is None or self._showing_placeholder:
            # 교체된 옛 페이지 · 새로고침 등으로 중단된 로드 · 안내 페이지 자체의 로드
            return

        if outcome == 'failed':
            # 백엔드가 꺼져 있거나 주소가 틀리면 Chromium 오류 화면 대신 안내를 보인다.
            self._showing_placeholder = True
            self._load_workflow_after = False
            self._web_view.setHtml(self._placeholder_html(), QUrl('about:blank'))
            return

        # ComfyUI 모드: 워크플로우 자동 로드 (첫 성공 때 한 번)
        if self._load_workflow_after:
            self._load_workflow_after = False
            self._workflow_retry_count = 0
            # ComfyUI 프론트엔드 초기화 대기 후 워크플로우 주입
            QTimer.singleShot(2000, self._inject_comfyui_workflow)

    def _inject_comfyui_workflow(self):
        """ComfyUI 웹 인터페이스에 워크플로우 JSON을 주입"""
        import config
        workflow_path = getattr(config, 'COMFYUI_WORKFLOW_PATH', '')
        if not workflow_path or not os.path.exists(workflow_path):
            return

        try:
            with open(workflow_path, 'r', encoding='utf-8') as f:
                workflow_data = json.load(f)
        except Exception:
            return

        # JSON을 안전하게 이스케이프
        workflow_json_str = json.dumps(json.dumps(workflow_data, ensure_ascii=False))

        # ComfyUI app 객체를 확인하고 워크플로우 로드
        js_code = f"""
        (function() {{
            var workflowStr = {workflow_json_str};
            var workflow = JSON.parse(workflowStr);

            if (typeof app === 'undefined' || !app.graph) {{
                return 'NOT_READY';
            }}

            // API format 판별 (class_type 키가 있으면 API format)
            var isApiFormat = false;
            for (var key in workflow) {{
                if (workflow[key] && typeof workflow[key] === 'object' && workflow[key].class_type) {{
                    isApiFormat = true;
                    break;
                }}
            }}

            try {{
                if (isApiFormat && typeof app.loadApiJson === 'function') {{
                    app.loadApiJson(workflow);
                    return 'OK_API';
                }} else if (typeof app.loadGraphData === 'function') {{
                    app.loadGraphData(workflow);
                    return 'OK_GRAPH';
                }}
            }} catch(e) {{
                return 'ERROR:' + e.message;
            }}
            return 'NO_METHOD';
        }})();
        """
        self._page.runJavaScript(js_code, 0, self._on_workflow_inject_result)

    def _on_workflow_inject_result(self, result):
        """워크플로우 주입 결과 처리"""
        if result == 'NOT_READY':
            # ComfyUI 아직 초기화 안됨 — 재시도 (최대 10회)
            self._workflow_retry_count += 1
            if self._workflow_retry_count < 10:
                QTimer.singleShot(1000, self._inject_comfyui_workflow)

    def _reload(self):
        """현재 페이지 새로고침 — 아직 웹뷰가 없거나 로드 실패 안내 중이면 다시 싣는다."""
        if self._current_url and self._web_view is not None and not self._showing_placeholder:
            self._web_view.reload()
        else:
            self._stale = False
            self.load_backend_ui()

    def _open_external(self):
        """외부 브라우저로 열기"""
        if self._current_url:
            from PyQt6.QtGui import QDesktopServices
            QDesktopServices.openUrl(QUrl(self._current_url))

    def _placeholder_html(self) -> str:
        """백엔드 UI 로드가 실패했을 때만 보이는 안내 페이지."""
        from html import escape

        url = escape(self._current_url or '')
        return f"""
        <html>
        <body style="background:{get_color('bg_primary')}; color:{get_color('text_muted')}; display:flex;
                     align-items:center; justify-content:center; height:100vh;
                     font-family:sans-serif; margin:0;">
          <div style="text-align:center;">
            <div style="font-size:48px; margin-bottom:16px;">🖥️</div>
            <div style="font-size:16px;">백엔드 UI를 불러오지 못했습니다</div>
            <div style="font-size:13px; color:{get_color('text_secondary')}; margin-top:8px;">
              {url}<br>백엔드가 켜져 있는지 확인한 뒤 새로고침(🔄)하세요.
            </div>
          </div>
        </body>
        </html>
        """
