"""
UIStateManager — 창 크기/스플리터/스크롤/패널 상태 자동 영속화.

NAIA 2.0의 ``core/ui_state_manager.py`` 패턴 참고.

설계
----
UI 요소가 자신의 상태를 어떻게 직렬화하는지 모름 — 호출자가 ``register()``로
이름과 ``get_state`` / ``set_state`` 콜백을 제공. 매니저는 그것을 ``config/state/ui_state.json``에
모아서 저장하고, 시작 시 복원.

복원은 150ms 지연 — 위젯 레이아웃이 자리 잡기를 기다림 (NAIA 패턴).

Qt 의존성 없음. 콜백은 PyQt/Vue/뭐든 사용 가능.

사용 예
-------
>>> mgr = UIStateManager(state_file=Path("config/state/ui_state.json"))
>>>
>>> # 메인 윈도우 등록
>>> mgr.register("main_window",
...              get_state=lambda: {"w": win.width(), "h": win.height()},
...              set_state=lambda s: win.resize(s["w"], s["h"]))
>>>
>>> # 시작 시:
>>> mgr.restore_all_delayed(delay_ms=150)
>>>
>>> # 종료 시:
>>> mgr.save_all()
"""
from __future__ import annotations

import json
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Optional

from utils.app_logger import get_logger

_logger = get_logger("ui_state")


@dataclass
class _Registration:
    name: str
    get_state: Callable[[], Any]
    set_state: Callable[[Any], None]


class UIStateManager:
    """UI 상태 영속화 — register 콜백 기반.

    저장 파일은 JSON. 각 등록 이름은 top-level 키.
    """

    def __init__(self, state_file: Path):
        self.state_file = Path(state_file)
        self._registrations: dict[str, _Registration] = {}
        self._cached_state: dict[str, Any] = {}  # 마지막 로드/저장된 상태
        self._lock = threading.RLock()
        self._delayed_restore_timer: Optional[threading.Timer] = None

        # 파일이 있으면 미리 로드 — restore 호출 시 즉시 사용 가능
        self._load_from_disk()

    # ────────────────────────────────────────
    # 등록 / 해제
    # ────────────────────────────────────────

    def register(self,
                 name: str,
                 get_state: Callable[[], Any],
                 set_state: Callable[[Any], None],
                 restore_now: bool = False) -> None:
        """위젯 등록.

        :param name: 고유 이름 (JSON 키)
        :param get_state: 현재 상태 반환 (직렬화 가능해야 함)
        :param set_state: 상태 적용
        :param restore_now: True면 등록 즉시 캐시된 상태 적용 시도
        """
        with self._lock:
            self._registrations[name] = _Registration(name, get_state, set_state)
            if restore_now and name in self._cached_state:
                self._apply_one(name)

    # ────────────────────────────────────────
    # 저장
    # ────────────────────────────────────────

    def collect_state(self) -> dict:
        """현재 상태 수집.

        등록된 항목의 ``get_state()`` 결과를 기존 캐시 위에 덮어씀.
        파일에서 읽었지만 이번 실행에 등록되지 않은 키도 보존됨(다음 실행에서 다시 쓸 수 있게).
        """
        with self._lock:
            result: dict[str, Any] = dict(self._cached_state)
            regs = list(self._registrations.values())
        for reg in regs:
            try:
                result[reg.name] = reg.get_state()
            except Exception:
                _logger.exception(f"get_state failed for {reg.name!r}")
        return result

    def save_all(self) -> bool:
        """모든 상태 수집 → 파일 저장."""
        try:
            state = self.collect_state()
            self.state_file.parent.mkdir(parents=True, exist_ok=True)
            from utils.atomic_json import atomic_write_json
            atomic_write_json(str(self.state_file), state, indent=2)
            with self._lock:
                self._cached_state = state
            return True
        except Exception:
            _logger.exception(f"save_all failed: {self.state_file}")
            return False

    # ────────────────────────────────────────
    # 복원
    # ────────────────────────────────────────

    def _load_from_disk(self) -> None:
        """파일 → 캐시. 등록된 항목에 적용은 안 함."""
        if not self.state_file.is_file():
            return
        try:
            data = json.loads(self.state_file.read_text(encoding="utf-8"))
            if isinstance(data, dict):
                with self._lock:
                    self._cached_state = data
        except Exception:
            _logger.exception(f"load_from_disk failed: {self.state_file}")

    def _apply_one(self, name: str) -> bool:
        with self._lock:
            reg = self._registrations.get(name)
            cached = self._cached_state.get(name)
        if reg is None or cached is None:
            return False
        try:
            reg.set_state(cached)
            return True
        except Exception:
            _logger.exception(f"set_state failed for {name!r}")
            return False

    def restore_all(self) -> int:
        """등록된 모든 항목에 캐시 적용 — 즉시. 적용된 개수 반환."""
        with self._lock:
            names = list(self._registrations.keys())
        return sum(1 for n in names if self._apply_one(n))

    def restore_all_delayed(self, delay_ms: int = 150) -> None:
        """``delay_ms`` 후 복원 — 반드시 GUI(메인) 스레드 이벤트 루프에 예약.

        Qt 시작 시 위젯 레이아웃이 자리 잡기 전에 setGeometry 호출하면
        레이아웃 매니저가 덮어쓸 수 있음 — 지연으로 회피.

        ★★ 반드시 메인 스레드에서 실행해야 함: 등록된 setter들이
        ``QWidget.setGeometry`` 등 Qt GUI를 직접 호출하므로, 백그라운드
        스레드(``threading.Timer``)에서 돌리면 즉시 native 크래시
        (Qt: GUI 조작은 메인 스레드 전용, abort 0x80000003)가 난다.
        따라서 ``QTimer.singleShot``으로 메인 스레드 이벤트 루프에 예약한다.
        (호출자는 GUI 스레드에서 호출해야 한다 — QTimer가 그 스레드의
        이벤트 루프에 콜백을 붙이기 때문.)
        """
        try:
            from PyQt6.QtCore import QTimer
            QTimer.singleShot(max(0, int(delay_ms)), self.restore_all)
        except Exception:
            # Qt 비가용 환경(헤드리스 테스트 등): GUI setter 자체가 없어
            # 스레드 타이머로 폴백해도 안전.
            with self._lock:
                if self._delayed_restore_timer is not None:
                    self._delayed_restore_timer.cancel()
                t = threading.Timer(delay_ms / 1000.0, self.restore_all)
                t.daemon = True
                self._delayed_restore_timer = t
                t.start()
