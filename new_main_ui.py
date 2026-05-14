# main.py
"""
AI Studio - Pro
메인 실행 파일
"""
import sys
import os

# Chromium/QtWebEngine 네이티브 로그 억제 (QApplication 생성 전에 설정)
os.environ.setdefault(
    "QTWEBENGINE_CHROMIUM_FLAGS",
    "--disable-logging --log-level=3 --disable-features=WebRtcHideLocalIpsWithMdns",
)
os.environ.setdefault("QT_LOGGING_RULES", "qt.webenginecontext.debug=false")

from config import *
from ui.generator_main import GeneratorMainUI
from PyQt6.QtWidgets import QApplication, QPushButton
from PyQt6.QtCore import Qt, QEvent, QObject
from PyQt6.QtGui import QPalette, QColor, QCursor


class ButtonCursorFilter(QObject):
    """QPushButton에 마우스 올리면 포인터 커서로 변경"""
    def eventFilter(self, obj, event):
        if isinstance(obj, QPushButton) and obj.isEnabled():
            if event.type() == QEvent.Type.Enter:
                obj.setCursor(QCursor(Qt.CursorShape.PointingHandCursor))
            elif event.type() == QEvent.Type.Leave:
                obj.unsetCursor()
        return False


def main():
    """메인 실행 함수"""
    # 윈도우 작업 표시줄 아이콘 해결 (AppUserModelID 설정)
    if sys.platform == 'win32':
        import ctypes
        myappid = 'mycompany.myproduct.subproduct.version' # 임의의 고유 ID
        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(myappid)

    QApplication.setHighDpiScaleFactorRoundingPolicy(
        Qt.HighDpiScaleFactorRoundingPolicy.PassThrough
    )
    
    app = QApplication(sys.argv)
    app.setApplicationName("AI Studio Pro")
    app.setOrganizationName("AI Studio")

    # 앱 전역 아이콘 설정 (작업 표시줄 및 트레이 기본값)
    from PyQt6.QtGui import QIcon
    icon_path = os.path.join(os.path.dirname(__file__), 'assets', 'icons', 'app_icon.svg')
    if os.path.exists(icon_path):
        app.setWindowIcon(QIcon(icon_path))
    
    # PyQt 스타일 완전 제거 — Vue가 모든 UI 스타일링 담당
    app.setStyleSheet("")

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 예외 핸들러 — 크래시 원인 로깅
    import traceback
    def _excepthook(exc_type, exc_value, exc_tb):
        print("=" * 60)
        print("UNHANDLED EXCEPTION:")
        traceback.print_exception(exc_type, exc_value, exc_tb)
        print("=" * 60)
    sys.excepthook = _excepthook

    window = GeneratorMainUI()
    window.showMaximized()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()