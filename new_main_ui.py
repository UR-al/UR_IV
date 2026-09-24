# main.py
"""
AI Studio - Pro
메인 실행 파일
"""
import sys
import os

# Do not import project code while the detached updater is replacing the
# checkout.  A Windows kernel mutex is process-owned, so a crashed helper can
# never leave a stale startup block behind.
def _acquire_detached_update_gate(timeout_ms: int = 120_000):
    if os.name != "nt":
        return None
    import ctypes

    kernel32 = ctypes.windll.kernel32
    kernel32.CreateMutexW.restype = ctypes.c_void_p
    kernel32.CreateMutexW.argtypes = (ctypes.c_void_p, ctypes.c_bool, ctypes.c_wchar_p)
    kernel32.WaitForSingleObject.restype = ctypes.c_ulong
    kernel32.WaitForSingleObject.argtypes = (ctypes.c_void_p, ctypes.c_ulong)
    kernel32.ReleaseMutex.argtypes = (ctypes.c_void_p,)
    kernel32.CloseHandle.argtypes = (ctypes.c_void_p,)
    handle = kernel32.CreateMutexW(None, False, r"Local\AIStudioPro.UR_IV.Update")
    if not handle:
        raise RuntimeError("앱 업데이트 시작 잠금을 확인하지 못했습니다.")
    result = int(kernel32.WaitForSingleObject(handle, timeout_ms))
    if result not in {0x00000000, 0x00000080}:
        kernel32.CloseHandle(handle)
        raise RuntimeError("앱 업데이트가 아직 진행 중입니다. 잠시 후 다시 실행하세요.")
    return kernel32, handle


_startup_update_gate = _acquire_detached_update_gate() if __name__ == "__main__" else None
try:
    if __name__ == "__main__":
        from core.app_instance import register_app_instance

        register_app_instance(update_guarded=_startup_update_gate is not None)
finally:
    if _startup_update_gate is not None:
        _startup_update_gate[0].ReleaseMutex(_startup_update_gate[1])
        _startup_update_gate[0].CloseHandle(_startup_update_gate[1])

if __name__ == "__main__":
    from core.app_startup import prepare_application

    _prepare_result = prepare_application()
    if _prepare_result:
        from core.app_instance import unregister_app_instance

        unregister_app_instance()
        raise SystemExit(_prepare_result)

# Chromium/QtWebEngine 네이티브 로그 억제 (QApplication 생성 전에 설정)
os.environ.setdefault(
    "QTWEBENGINE_CHROMIUM_FLAGS",
    "--disable-logging --log-level=3 --disable-features=WebRtcHideLocalIpsWithMdns",
)
os.environ.setdefault("QT_LOGGING_RULES", "qt.webenginecontext.debug=false")

from config import *
from ui.generator_main import GeneratorMainUI
from PyQt6.QtWidgets import QApplication
from PyQt6.QtCore import Qt


def main():
    """메인 실행 함수"""
    # 윈도우 작업 표시줄 아이콘 해결 (AppUserModelID 설정)
    if sys.platform == 'win32':
        import ctypes
        from core.app_instance import APP_USER_MODEL_ID
        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(APP_USER_MODEL_ID)

    QApplication.setHighDpiScaleFactorRoundingPolicy(
        Qt.HighDpiScaleFactorRoundingPolicy.PassThrough
    )

    # 갤러리 카드 썸네일 스킴(aithumb:) — QWebEngineUrlScheme 는 QApplication 전에 등록해야 한다.
    from ui.thumb_scheme import register_thumb_scheme
    register_thumb_scheme()

    app = QApplication(sys.argv)
    app.setApplicationName("AI Studio Pro")
    app.setOrganizationName("AI Studio")
    try:
        with open(os.path.join(os.path.dirname(__file__), "VERSION"), encoding="utf-8") as version_file:
            app.setApplicationVersion(version_file.read().strip())
    except OSError:
        app.setApplicationVersion("0.0.0")

    # 앱 전역 아이콘 설정 (작업 표시줄 및 트레이 기본값)
    from PyQt6.QtGui import QIcon
    icon_path = os.path.join(os.path.dirname(__file__), 'assets', 'icons', 'app_icon.svg')
    if os.path.exists(icon_path):
        app.setWindowIcon(QIcon(icon_path))

    # PyQt 전역 스타일시트는 두지 않는다 — Vue가 모든 UI 스타일링 담당

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 예외 핸들러 — 크래시 원인 로깅 (콘솔 + 파일).
    # Python 예외와 네이티브 크래시(SIGSEGV/abort/0x80000003)를 모두
    # logs/last_crash.log 에 기록해, 콘솔 창이 닫혀도 사후 확인 가능.
    # 슬롯 예외가 PyQt 기본 동작(qFatal)으로 앱을 끝내지 않게도 한다 — 웹 모드와 공용(core/crash_hooks.py).
    from core.crash_hooks import install_crash_handlers
    _crash_fp = install_crash_handlers()  # noqa: F841 — 앱 수명 동안 파일을 열어 둔다(faulthandler)

    window = GeneratorMainUI()
    # 캡션 저장 폴더 승인 1회 이식 — 승인 규칙(core/caption_out_dir.py) 전에 대화상자로 고른 폴더를
    # 계속 쓰게 한다. 데스크톱 진입점에서만(웹 모드는 이 값을 클라이언트가 썼을 수 있다).
    try:
        window.vue_bridge.seed_caption_out_dir_approval_from_prefs()
    except Exception as exc:
        print(f"[Caption] 저장 폴더 승인 이식 건너뜀: {exc}")
    # 스플래시 로딩 시퀀스: 백엔드 선택 → 로딩창 → 데이터 준비 → 완성된 UI.
    # (실패해도 내부 폴백으로 앱은 뜸. 시작 다이얼로그 X면 SystemExit로 종료.)
    window._run_startup_sequence(app)

    # 생성 API는 설정에서 명시적으로 켜 둔 경우에만 로컬 서버를 연다.
    # 위젯에 참조를 보관해 앱 수명과 gateway 수명을 일치시킨다.
    try:
        from core.generation_api import get_generation_api_manager

        generation_api_manager = get_generation_api_manager()
        window._generation_api_manager = generation_api_manager
        generation_api_manager.start_if_enabled()
        app.aboutToQuit.connect(generation_api_manager.shutdown)
    except Exception as exc:
        print(f"[Generation API] startup skipped: {exc}")
    window.showMaximized()

    # run_main_loop = app.exec() + 메인 루프 표시 — 슬롯 안 Ctrl+C(KeyboardInterrupt)를 크래시 훅이
    # 앱 종료 요청으로 넘길 때, 메인 루프면 창 닫기 확인·저장을 타는 quit() 을 쓴다(core/console_interrupt).
    from core.console_interrupt import run_main_loop
    sys.exit(run_main_loop(app))


if __name__ == "__main__":
    main()
