"""Web/Backend 네이티브 탭 — 영속 프로필과 숨은 탭의 지연 생성.

실제 QtWebEngine 은 만들지 않는다(렌더러/GPU 프로세스를 띄우지 않게). offscreen 하위
프로세스에서 PyQt6.QtWebEngineWidgets/QtWebEngineCore 를 기록용 가짜 모듈로 바꿔 끼운 뒤
실제 tabs/browser_tab.py · tabs/backend_ui_tab.py 를 import 한다.
"""
import os
import subprocess
import sys
import textwrap
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

_PRELUDE = '''
import sys, types
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
from PyQt6.QtCore import QObject, QUrl, pyqtSignal
from PyQt6.QtWidgets import QApplication, QWidget

core_mod = types.ModuleType('PyQt6.QtWebEngineCore')
widgets_mod = types.ModuleType('PyQt6.QtWebEngineWidgets')


class QWebEngineSettings:
    class WebAttribute:
        LocalStorageEnabled = 1
        JavascriptEnabled = 2
        PluginsEnabled = 3
        JavascriptCanOpenWindows = 4
        AllowRunningInsecureContent = 5
        AutoLoadImages = 6

    def __init__(self):
        self.attrs = {}

    def setAttribute(self, key, value):
        self.attrs[key] = value


class QWebEngineProfile(QObject):
    class PersistentCookiesPolicy:
        NoPersistentCookies = 0
        AllowPersistentCookies = 1
        ForcePersistentCookies = 2

    created = []

    def __init__(self, name='', parent=None):
        super().__init__(parent)
        self.name, self.ua, self.cache, self.storage, self.cookies = name, '', '', '', None
        QWebEngineProfile.created.append(self)

    def isOffTheRecord(self):
        return not self.name

    def setHttpUserAgent(self, value):
        self.ua = value

    def setCachePath(self, value):
        self.cache = value

    def setPersistentStoragePath(self, value):
        self.storage = value

    def setPersistentCookiesPolicy(self, value):
        self.cookies = value


class QWebEnginePage(QObject):
    loadFinished = pyqtSignal(bool)
    loadingChanged = pyqtSignal(object)   # QWebEngineLoadingInfo 대신 status()/errorCode() 가짜
    default_profile = QWebEngineProfile('')

    def __init__(self, *args):
        profile = args[0] if args and isinstance(args[0], QWebEngineProfile) else None
        parent = args[-1] if args and not isinstance(args[-1], QWebEngineProfile) else None
        super().__init__(parent)
        self._profile = profile or QWebEnginePage.default_profile
        self._settings = QWebEngineSettings()
        self.background = None

    def profile(self):
        return self._profile

    def settings(self):
        return self._settings

    def setBackgroundColor(self, color):
        self.background = color.name()

    def runJavaScript(self, *args):
        pass


class QWebEngineView(QWidget):
    urlChanged = pyqtSignal(QUrl)
    created = []

    def __init__(self, parent=None):
        super().__init__(parent)
        self._page = QWebEnginePage(self)
        self.urls, self.html = [], []
        QWebEngineView.created.append(self)

    def page(self):
        return self._page

    def setPage(self, page):
        self._page = page

    def settings(self):
        return self._page.settings()

    def setUrl(self, url):
        self.urls.append(url.toString())

    def setHtml(self, html, base=None):
        self.html.append(html)

    def reload(self):
        self.urls.append('reload')

    def back(self):
        pass


core_mod.QWebEngineSettings = QWebEngineSettings
core_mod.QWebEngineProfile = QWebEngineProfile
core_mod.QWebEnginePage = QWebEnginePage
widgets_mod.QWebEngineView = QWebEngineView
sys.modules['PyQt6.QtWebEngineCore'] = core_mod
sys.modules['PyQt6.QtWebEngineWidgets'] = widgets_mod
ui_package = types.ModuleType('ui')
ui_package.__path__ = [str(Path('ui').resolve())]
sys.modules['ui'] = ui_package
app = QApplication([])


def loading(status, error_code=0):
    """QWebEngineLoadingInfo 흉내 — status 이름과 Chromium 오류 코드만."""
    return SimpleNamespace(status=lambda: SimpleNamespace(name=status), errorCode=lambda: error_code)


FAILED = loading('LoadFailedStatus', -102)       # ERR_CONNECTION_REFUSED
ABORTED = loading('LoadFailedStatus', -3)        # ERR_ABORTED — 다른 이동이 끊었다
STOPPED = loading('LoadStoppedStatus', -3)
STARTED = loading('LoadStartedStatus')
SUCCEEDED = loading('LoadSucceededStatus')
'''


class WebTabsTests(unittest.TestCase):
    def probe(self, body):
        result = subprocess.run(
            [sys.executable, '-X', 'faulthandler', '-c', textwrap.dedent(_PRELUDE) + textwrap.dedent(body)],
            cwd=ROOT, capture_output=True, text=True, encoding='utf-8', errors='replace', timeout=60,
            env={**os.environ, 'QT_QPA_PLATFORM': 'offscreen', 'QT_OPENGL': 'software',
                 'PYTHONIOENCODING': 'utf-8'},
        )
        self.assertEqual(result.returncode, 0, result.stdout[-3000:] + result.stderr[-3000:])

    def test_web_tab_uses_a_named_persistent_profile_under_storage_boundaries(self):
        self.probe('''
from core.storage_paths import storage_paths
from tabs.browser_tab import BrowserTab, WEB_TAB_PROFILE_NAME
tab = BrowserTab()
profile = tab.web_view.page().profile()
assert profile.name == WEB_TAB_PROFILE_NAME and not profile.isOffTheRecord(), profile.name
assert profile.cookies == QWebEngineProfile.PersistentCookiesPolicy.ForcePersistentCookies
assert Path(profile.storage) == storage_paths.user_data_dir / 'web_tab_profile', profile.storage
assert Path(profile.cache) == storage_paths.cache_dir / 'web_tab', profile.cache
assert 'Chrome/' in profile.ua
assert tab.web_view.urls == [], '첫 URL 은 탭을 열 때만 싣는다'
tab.ensure_loaded()
tab.ensure_loaded()
assert tab.web_view.urls == [tab.home_url], tab.web_view.urls
''')

    def test_hidden_backend_tab_creates_no_web_view_until_first_shown(self):
        self.probe('''
from backends import BackendType
from tabs.backend_ui_tab import BackendUITab
tab = BackendUITab()
tab.apply_theme()
tab.mark_backend_changed()          # 숨어 있으면 싣지 않는다
assert tab._web_view is None and not QWebEngineView.created, QWebEngineView.created
assert not [p for p in QWebEngineProfile.created if p.name == 'backend_ui']
backend = SimpleNamespace(api_url='http://127.0.0.1:9')
with patch('backends.get_backend', return_value=backend), \\
        patch('backends.get_backend_type', return_value=BackendType.WEBUI):
    tab.ensure_loaded()
    view = tab._web_view
    assert view is not None and len(QWebEngineView.created) == 1
    assert [p.name for p in QWebEngineProfile.created if p.name] == ['backend_ui']
    assert view.urls == ['http://127.0.0.1:9'], view.urls
    assert view.html == [], 'placeholder 는 로드 실패 때만'
    assert tab._page.background, '지연 생성된 페이지도 테마 배경을 받는다'
    tab.ensure_loaded()             # 이미 실었으면 다시 싣지 않는다
    assert view.urls == ['http://127.0.0.1:9'], view.urls
    tab._page.loadingChanged.emit(FAILED)   # 백엔드가 꺼져 있으면 안내 페이지
    assert len(view.html) == 1 and '불러오지 못했습니다' in view.html[0]
    assert '127.0.0.1:9' in view.html[0]
    tab._page.loadingChanged.emit(SUCCEEDED)   # 안내 페이지 자체의 로드는 무시
    assert len(view.html) == 1
    tab._reload()                   # 안내 중 새로고침은 백엔드 UI 를 다시 싣는다
    assert view.urls[-1] == 'http://127.0.0.1:9' and len(QWebEngineView.created) == 1, view.urls
    tab._page.loadingChanged.emit(SUCCEEDED)
    tab._reload()
    assert view.urls[-1] == 'reload', view.urls
''')

    def test_aborted_or_superseded_loads_do_not_show_the_failure_placeholder(self):
        """loadFinished(False) 는 ERR_ABORTED(중단)도 포함한다 — 첫 로드 중 🔄·백엔드 전환이
        살아 있는 백엔드 UI 를 '불러오지 못했습니다'로 덮고 새 로드를 취소하던 회귀."""
        self.probe('''
from backends import BackendType
from tabs.backend_ui_tab import BackendUITab, load_outcome

assert load_outcome('LoadSucceededStatus') == 'succeeded'
assert load_outcome('LoadFailedStatus', -102) == 'failed'
assert load_outcome('LoadFailedStatus', -3) is None
assert load_outcome('LoadStoppedStatus', -3) is None
assert load_outcome('LoadStartedStatus') is None
assert load_outcome('LoadFailedStatus', 'garbage') == 'failed'

tab = BackendUITab()
backend = SimpleNamespace(api_url='http://127.0.0.1:9')
with patch('backends.get_backend', return_value=backend), \\
        patch('backends.get_backend_type', return_value=BackendType.WEBUI):
    tab.ensure_loaded()
    view, first = tab._web_view, tab._page
    first.loadingChanged.emit(STARTED)
    tab._reload()                          # 첫 로드 중 🔄 — 진행 중이던 로드는 중단된다
    assert view.urls[-1] == 'reload', view.urls
    first.loadingChanged.emit(ABORTED)
    first.loadingChanged.emit(STOPPED)
    assert view.html == [], '중단은 실패가 아니다'
    assert not tab._showing_placeholder

    tab.load_backend_ui()                  # 백엔드 전환(mark_backend_changed) — 새 페이지
    second = tab._page
    assert second is not first
    first.loadingChanged.emit(FAILED)      # 교체된 옛 페이지가 늦게 보낸 실패
    assert view.html == [], '옛 페이지의 결과로 새 페이지를 덮지 않는다'
    second.loadingChanged.emit(FAILED)     # 새 페이지의 연결은 살아 있다
    assert len(view.html) == 1 and '불러오지 못했습니다' in view.html[0], view.html

    tab._reload()                          # 안내 중 새로고침 → 새 페이지로 다시 싣는다
    third = tab._page
    assert third is not second and not tab._showing_placeholder
    third.loadingChanged.emit(SUCCEEDED)
    tab._reload()
    third.loadingChanged.emit(ABORTED)     # 성공한 페이지를 새로고침해 생긴 중단
    assert len(view.html) == 1, view.html
    third.loadingChanged.emit(FAILED)      # 새로고침이 진짜로 실패하면 안내한다
    assert len(view.html) == 2, view.html
''')

    def test_comfy_workflow_is_injected_once_after_the_first_real_success(self):
        self.probe('''
from backends import BackendType
from tabs.backend_ui_tab import BackendUITab
import tabs.backend_ui_tab as backend_ui_tab

scheduled = []
tab = BackendUITab()
backend = SimpleNamespace(api_url='http://127.0.0.1:9')
with patch('backends.get_backend', return_value=backend), \\
        patch('backends.get_backend_type', return_value=BackendType.COMFYUI), \\
        patch.object(backend_ui_tab.QTimer, 'singleShot', side_effect=lambda ms, fn: scheduled.append(ms)):
    tab.ensure_loaded()
    page = tab._page
    page.loadingChanged.emit(ABORTED)
    assert scheduled == [] and tab._load_workflow_after, '중단된 로드로 주입을 소비하지 않는다'
    page.loadingChanged.emit(SUCCEEDED)
    page.loadingChanged.emit(SUCCEEDED)
    assert scheduled == [2000], scheduled
''')


if __name__ == '__main__':
    unittest.main()
