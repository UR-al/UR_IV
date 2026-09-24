# core/crash_hooks.py
"""처리되지 않은 예외·네이티브 크래시 기록 — 데스크톱(new_main_ui)과 웹 모드(web_main_ui) 공용.

왜 두 진입점이 같이 써야 하는가: PyQt6 는 Qt 가 부른 슬롯(시그널 연결·타이머·QWebChannel)에서
난 파이썬 예외를, ``sys.excepthook`` 이 기본값이면 qFatal 로 처리해 프로세스를 끝낸다.
데스크톱만 훅을 걸고 웹 모드는 걸지 않아서, 웹 모드에서는 더미 프록시의 AttributeError 하나가
서버 전체를 죽일 수 있었다. 훅이 있으면 트레이스백만 남기고 이벤트 루프는 계속 돈다.

- 콘솔에는 구분선과 함께 트레이스백을 찍는다.
- 크래시 로그 파일에도 같은 트레이스백을 덧붙인다(콘솔 창이 닫혀도 사후 확인). 진입점마다 파일이
  다르다 — 데스크톱은 ``logs/last_crash.log``, 웹 모드는 ``logs/last_crash_web.log``
  (:func:`crash_log_name`). 두 모드는 동시에 뜰 수 있고(core.app_instance 는 PID 별 등록뿐) 파일을
  ``'w'`` 로 새로 열므로, 한 파일을 같이 쓰면 나중에 뜬 쪽이 앞선 쪽의 크래시 기록을 지우고
  faulthandler 덤프가 뒤섞였다.
- ``faulthandler`` 를 같은 파일에 켜서 SIGSEGV/abort 같은 네이티브 크래시의 전체 스레드 스택도 남긴다.
- KeyboardInterrupt(콘솔 Ctrl+C)는 크래시가 아니다 — 기록하지 않고 앱 종료를 요청한다
  (core/console_interrupt). 훅이 이것까지 삼키면 Ctrl+C 로 웹 서버를 멈출 수 없었다.
"""
from __future__ import annotations

import os
import re
import sys
import traceback
from typing import Callable, Optional, TextIO

ExceptHook = Callable[[type, BaseException, object], None]

DEFAULT_CRASH_LOG = "last_crash.log"
LEGACY_CRASH_LOG = "config/last_crash.log"


def crash_log_name(label: str = "") -> str:
    """진입점별 크래시 로그 파일 이름 — 라벨이 없으면(데스크톱) ``last_crash.log``, ``'web'`` 은 ``last_crash_web.log``."""
    slug = re.sub(r"[^a-z0-9_-]+", "_", str(label or "").strip().lower()).strip("_-")
    return f"last_crash_{slug}.log" if slug else DEFAULT_CRASH_LOG


def _default_interrupt_handler() -> bool:
    from core.console_interrupt import request_quit

    return request_quit()


def make_excepthook(
    crash_fp: Optional[TextIO], *, label: str = "", on_interrupt: Optional[Callable[[], bool]] = None,
) -> ExceptHook:
    """콘솔과 ``crash_fp``(있으면)에 트레이스백을 남기는 ``sys.excepthook`` 을 만든다.

    훅은 절대 예외를 밖으로 내지 않는다 — 기록 실패가 원래 예외를 가리거나 앱을 끝내지 않게.
    KeyboardInterrupt 는 ``on_interrupt()``(기본: core.console_interrupt.request_quit — 이벤트 루프
    안이면 앱 종료 요청)로 넘긴다. 요청하지 못했으면(루프 밖 — 인터프리터가 곧 끝난다) 파이썬 기본
    훅으로 알린다. 어느 쪽이든 크래시 로그에는 쓰지 않는다.
    """
    title = f"UNHANDLED EXCEPTION{f' ({label})' if label else ''}:"
    interrupt = on_interrupt or _default_interrupt_handler

    def _on_keyboard_interrupt(exc_type, exc_value, exc_tb):
        try:
            requested = bool(interrupt())
        except Exception:
            requested = False
        try:
            if requested:
                print("[app] Ctrl+C — 앱을 종료합니다", flush=True)
            else:
                sys.__excepthook__(exc_type, exc_value, exc_tb)
        except Exception:
            pass

    def _excepthook(exc_type, exc_value, exc_tb):
        if isinstance(exc_type, type) and issubclass(exc_type, KeyboardInterrupt):
            _on_keyboard_interrupt(exc_type, exc_value, exc_tb)
            return
        try:
            print("=" * 60)
            print(title)
            traceback.print_exception(exc_type, exc_value, exc_tb)
            print("=" * 60)
        except Exception:
            pass
        if crash_fp is not None:
            try:
                crash_fp.write("\n=== UNHANDLED PYTHON EXCEPTION ===\n")
                traceback.print_exception(exc_type, exc_value, exc_tb, file=crash_fp)
                crash_fp.flush()
            except Exception:
                pass

    return _excepthook


def open_crash_log(path: str) -> Optional[TextIO]:
    """크래시 기록 파일을 새로 연다(줄 버퍼, UTF-8). 실패하면 None — 기록 없이 계속 뜬다."""
    try:
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        return open(path, "w", encoding="utf-8", buffering=1)
    except Exception:
        return None


def install_crash_handlers(crash_log_path: Optional[str] = None, *, label: str = "") -> Optional[TextIO]:
    """``sys.excepthook`` 과 ``faulthandler`` 를 설치하고 열린 크래시 로그 파일을 돌려준다.

    ``crash_log_path`` 가 없으면 ``core.storage_paths.log_file(crash_log_name(label))`` 를 쓴다 —
    데스크톱(라벨 없음)은 ``last_crash.log``(옛 ``config/last_crash.log`` 이전 포함), 웹 모드는
    ``last_crash_web.log``. 호출자는 반환값을 앱 수명 동안 붙잡아 둔다(파일이 닫히면 faulthandler
    기록이 끊긴다).
    """
    if crash_log_path is None:
        try:
            from core.storage_paths import log_file

            name = crash_log_name(label)
            legacy = LEGACY_CRASH_LOG if name == DEFAULT_CRASH_LOG else None
            crash_log_path = str(log_file(name, legacy_paths=legacy))
        except Exception:
            crash_log_path = None
    crash_fp = open_crash_log(crash_log_path) if crash_log_path else None
    if crash_fp is not None:
        try:
            import faulthandler

            faulthandler.enable(crash_fp)
        except Exception:
            pass
    sys.excepthook = make_excepthook(crash_fp, label=label)
    return crash_fp


__all__ = [
    "DEFAULT_CRASH_LOG",
    "crash_log_name",
    "install_crash_handlers",
    "make_excepthook",
    "open_crash_log",
]
