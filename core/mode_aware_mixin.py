"""
ModeAwareMixin — 백엔드 모드별 설정 자동 저장/로드.

목적
----
UR_IV는 WEBUI/COMFYUI 두 백엔드를 전환하며 사용한다.
사용자가 모드를 바꿀 때마다 모드별로 설정을 자동 영속화하는 표준 패턴.

저장 위치
---------
``<project_root>/config/automation/<base_filename>_<MODE>.json``

예: ``config/automation/automation_webui.json`` / ``..._comfyui.json``

사용 패턴
---------
>>> from core.mode_aware_mixin import ModeAwareMixin
>>> class AutomationModule(ModeAwareMixin):
...     settings_base_filename = "automation"
...
...     def collect_current_settings(self) -> dict:
...         return {"mode": self.mode, "limit": self.limit}
...
...     def apply_settings(self, s: dict) -> None:
...         self.mode = s.get("mode", "count")
...         self.limit = s.get("limit", 10)
>>>
>>> m = AutomationModule()
>>> m.load_mode_settings()           # 현재 백엔드 모드의 설정 로드
>>> m.save_mode_settings()           # 현재 모드로 저장
>>> m.auto_subscribe_to_mode_change()  # 자동 전환 (이전 모드 저장 → 새 모드 로드)
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

from utils.app_logger import get_logger

_logger = get_logger("mode_aware")


def _current_mode_value() -> str:
    """현재 백엔드 모드 문자열 ('webui' / 'comfyui'). 실패 시 'webui'."""
    try:
        from backends import get_backend_type
        return get_backend_type().value
    except Exception:
        return "webui"


class ModeAwareMixin:
    """믹스인 — 모드별 설정 영속화.

    상속 클래스가 구현해야 할 것:
    - ``settings_base_filename: str`` (클래스 속성). 예: ``"automation"``
    - ``collect_current_settings(self) -> dict``
    - ``apply_settings(self, settings: dict) -> None``

    구독을 자동화하려면 ``auto_subscribe_to_mode_change()`` 한 번 호출.
    """

    settings_base_filename: str = ""

    # ────────────────────────────────────────────────────
    # 구현 필수
    # ────────────────────────────────────────────────────

    def collect_current_settings(self) -> dict:
        raise NotImplementedError(
            f"{type(self).__name__} must implement collect_current_settings()"
        )

    def apply_settings(self, settings: dict) -> None:
        raise NotImplementedError(
            f"{type(self).__name__} must implement apply_settings()"
        )

    # ────────────────────────────────────────────────────
    # 영속화
    # ────────────────────────────────────────────────────

    def _settings_path(self, mode_value: Optional[str] = None) -> Path:
        base = self.settings_base_filename
        if not base:
            raise ValueError(
                f"{type(self).__name__}.settings_base_filename must be set "
                "(e.g., 'automation')"
            )
        m = mode_value or _current_mode_value()
        filename = f"{base}_{m}.json"
        from core.storage_paths import config_file
        return config_file(
            f"automation/{filename}",
            legacy_paths=f"save/{filename}",
        )

    def save_mode_settings(self, mode_value: Optional[str] = None) -> bool:
        """현재 (또는 지정한) 모드로 설정 저장. 성공 시 True.

        파일 내용이 이미 같으면 쓰지 않는다 — 같은 값을 다시 보내는 동기화(Vue 마운트·생성 직전·
        모드 전환 때의 이전 모드 저장)가 파일을 매번 다시 쓰지 않게(감사 #42).
        """
        try:
            path = self._settings_path(mode_value)
            data = self.collect_current_settings()
            if path.exists():
                try:
                    if json.loads(path.read_text(encoding="utf-8")) == data:
                        return True
                except (OSError, ValueError):
                    pass   # 깨진 파일은 아래에서 새로 쓴다
            from utils.atomic_json import atomic_write_json
            atomic_write_json(str(path), data, indent=2)
            _logger.debug(f"saved: {path.name}")
            return True
        except Exception:
            _logger.exception(
                f"save_mode_settings failed: {self.settings_base_filename}"
            )
            return False

    def load_mode_settings(self, mode_value: Optional[str] = None) -> bool:
        """현재 (또는 지정한) 모드 설정 로드 → ``apply_settings()`` 호출.

        파일 없으면 False (에러 아님 — 첫 실행 시 정상).
        """
        try:
            path = self._settings_path(mode_value)
        except (ValueError, OSError):
            _logger.exception("load_mode_settings: settings path unavailable")
            return False

        if not path.exists():
            _logger.debug(f"no saved settings: {path.name} (first run?)")
            return False

        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            self.apply_settings(data)
            _logger.debug(f"loaded: {path.name}")
            return True
        except json.JSONDecodeError:
            _logger.exception(f"corrupt JSON: {path}")
            return False
        except Exception:
            _logger.exception(
                f"load_mode_settings failed: {self.settings_base_filename}"
            )
            return False

    # ────────────────────────────────────────────────────
    # 자동 전환
    # ────────────────────────────────────────────────────

    def auto_subscribe_to_mode_change(self) -> Optional[int]:
        """AppContext의 ``BACKEND_CHANGED`` 이벤트 구독.

        모드 변경 시:
        1. 이전 모드로 현재 설정을 저장
        2. 새 모드의 설정을 로드 → ``apply_settings()`` 호출

        반환: 구독 handle (해제 시 ``AppContext.unsubscribe``에 전달). 실패 시 None.
        """
        try:
            from core.app_context import get_context, Events
        except ImportError:
            _logger.warning("AppContext 없음 — 자동 전환 비활성")
            return None

        # 시작 시점의 모드 기억
        self._mam_last_mode: str = _current_mode_value()

        def _on_backend_changed(new_mode_data):
            # data가 BackendType이면 .value, 문자열이면 그대로
            new_value = getattr(new_mode_data, "value", new_mode_data)
            if not isinstance(new_value, str):
                new_value = _current_mode_value()

            prev = getattr(self, "_mam_last_mode", None)
            if prev and prev != new_value:
                self.save_mode_settings(prev)
            self.load_mode_settings(new_value)
            self._mam_last_mode = new_value

        ctx = get_context()
        return ctx.subscribe(Events.BACKEND_CHANGED, _on_backend_changed)
