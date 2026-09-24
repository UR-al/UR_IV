# core/console_interrupt.py
"""콘솔 Ctrl+C(SIGINT)를 Qt 앱의 정상 종료로 잇는다 — 데스크톱(new_main_ui)·웹 모드(web_main_ui) 공용.

왜 따로 두는가: PyQt6 는 Qt 가 부른 슬롯(타이머·시그널·QWebChannel)에서 난 파이썬 예외를
``sys.excepthook`` 에 넘긴다. 예전 웹 모드는 훅이 없어서, 기본 SIGINT 처리기가 1초 VRAM 타이머 같은
슬롯 안에서 낸 KeyboardInterrupt 를 PyQt 기본 동작(qFatal)이 받아 프로세스째 끝냈다 — Ctrl+C 가
'우연히' 서버를 멈추던 길이다. 공용 크래시 훅(core/crash_hooks)이 슬롯 예외를 기록만 하고 삼키게
되자 Ctrl+C 는 'UNHANDLED EXCEPTION' 만 남기고 서버는 계속 돌았다. 이 모듈이 그 의도를 명시적으로
되살린다(정리 단계까지 거쳐서).

- 이벤트 루프 안에서 받은 Ctrl+C 는 앱 종료 요청이다.
  - 메인 루프(:func:`run_main_loop`)가 돌면 ``app.quit()`` — 창 닫기 확인과 ``aboutToQuit`` 정리
    (창 상태·대기열 flush, 서버 종료)를 평소 종료와 똑같이 탄다.
  - 메인 루프 전(시작 대화상자·로딩 대기 같은 중첩 루프)에는 Qt 6 의 quit() 이 무시된다(exec 밖).
    그래서 요청을 기억하고 ``app.exit(0)`` 으로 중첩 루프를 모두 끝낸 뒤, 메인 루프가 시작되자마자
    quit() 한다.
- 이벤트 루프 밖(메인 스레드의 평범한 파이썬 코드)이면 기본 동작 그대로 KeyboardInterrupt 를 낸다.
- 파이썬 시그널 처리기는 파이썬 바이트코드가 돌 때만 실행된다. Qt 루프가 C++ 에서 쉬는 동안에도
  곧 반응하도록 :func:`install_sigint_quit` 이 빈 파이썬 타이머를 함께 돌린다.
"""
from __future__ import annotations

import signal
from typing import Any, Callable, Optional

KEEPALIVE_MS = 500

_main_loop_running = False
_quit_requested = False


def event_loop_level() -> int:
    """지금 스레드에서 돌고 있는 Qt 이벤트 루프 깊이(0 = 루프 밖). Qt 를 못 쓰면 0."""
    try:
        from PyQt6.QtCore import QThread

        return int(QThread.currentThread().loopLevel())
    except Exception:
        return 0


def _qt_app() -> Any:
    try:
        from PyQt6.QtCore import QCoreApplication

        return QCoreApplication.instance()
    except Exception:
        return None


def main_loop_running() -> bool:
    return _main_loop_running


def quit_requested() -> bool:
    """메인 루프 전에 받아 두고 아직 반영하지 않은 종료 요청이 있는가."""
    return _quit_requested


def request_quit(app: Any = None, *, loop_level: Optional[int] = None) -> bool:
    """이벤트 루프 안이면 앱 종료를 요청하고 True. 앱이 없거나 루프 밖이면 False(호출자가 기본 동작).

    메인 루프면 ``app.quit()``, 메인 루프 전의 중첩 루프면 요청을 기억하고 ``app.exit(0)``.
    """
    global _quit_requested
    if app is None:
        app = _qt_app()
    if app is None:
        return False
    level = event_loop_level() if loop_level is None else int(loop_level)
    if level <= 0:
        return False
    if _main_loop_running:
        app.quit()
    else:
        # Qt 6 은 exec 밖의 quit() 을 무시한다. 중첩 루프를 모두 끝내고(이후 exec 전까지 새 중첩 루프도
        # 곧바로 돌아온다) 메인 루프가 시작되면 run_main_loop 가 quit() 한다.
        _quit_requested = True
        app.exit(0)
    return True


def run_main_loop(app: Any) -> int:
    """``app.exec()`` 를 돌린다 — 메인 루프가 도는 동안을 표시하고, 시작 중 받은 종료 요청을 곧바로 반영한다.

    진입점은 ``sys.exit(app.exec())`` 대신 ``sys.exit(run_main_loop(app))`` 를 쓴다. 그래야
    :func:`request_quit` 이 창 닫기·aboutToQuit 정리를 타는 quit() 과 시작 중 요청을 가를 수 있다.
    """
    global _main_loop_running, _quit_requested
    if _quit_requested:
        _quit_requested = False
        from PyQt6.QtCore import QTimer

        QTimer.singleShot(0, app.quit)
    _main_loop_running = True
    try:
        return app.exec()
    finally:
        _main_loop_running = False


def make_sigint_handler(
    app: Any, *, loop_level: Optional[Callable[[], int]] = None, notify: Optional[Callable[[str], None]] = None,
) -> Callable[[int, Any], None]:
    """SIGINT 처리기 — 이벤트 루프 안이면 종료 요청, 밖이면 기본 처리기(KeyboardInterrupt)."""
    level_of = loop_level or event_loop_level
    say = notify or (lambda text: print(text, flush=True))

    def _on_sigint(signum: int, frame: Any) -> None:
        if request_quit(app, loop_level=level_of()):
            try:
                say("[app] Ctrl+C — 앱을 종료합니다")
            except Exception:
                pass
            return
        signal.default_int_handler(signum, frame)

    return _on_sigint


def install_sigint_quit(app: Any, *, keepalive_ms: int = KEEPALIVE_MS) -> Any:
    """Ctrl+C 를 앱 종료로 잇는다. 반환한 keepalive 타이머(부모 = app)는 앱 수명 동안 돈다.

    QApplication 을 만든 직후 부른다. keepalive_ms <= 0 이면 타이머를 만들지 않고 None.
    """
    signal.signal(signal.SIGINT, make_sigint_handler(app))
    if keepalive_ms <= 0:
        return None
    from PyQt6.QtCore import QTimer

    timer = QTimer(app)
    # 빈 파이썬 슬롯 — 인터프리터가 주기적으로 깨어나 대기 중인 SIGINT 처리기를 실행하게 한다.
    timer.timeout.connect(lambda: None)
    timer.start(int(keepalive_ms))
    return timer


__all__ = [
    "KEEPALIVE_MS",
    "event_loop_level",
    "install_sigint_quit",
    "main_loop_running",
    "make_sigint_handler",
    "quit_requested",
    "request_quit",
    "run_main_loop",
]
