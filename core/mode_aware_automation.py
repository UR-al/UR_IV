"""
ModeAwareAutomationSettings — 자동화 설정을 백엔드 모드별로 영속.

ModeAwareMixin 어댑터. 호스트(보통 GeneratorMainUI)의
``_vue_automation_settings`` dict와 Vue UI 사이를 매개.

저장 파일:
- ``config/automation/automation_settings_webui.json``
- ``config/automation/automation_settings_comfyui.json``

흐름:
1. 시작 시(GeneratorMainUI.__init__, 동기) 현재 모드 설정 로드 → 호스트 dict 갱신
2. Vue 가 마운트 때 ``getAutomationSettings`` 슬롯(:meth:`snapshot`)으로 그 값을 당겨 화면을
   채운 뒤 set_automation_settings 로 동기화한다 — 예전엔 Vue 의 하드코딩 기본값 푸시가 1.5초
   타이머의 1회 emit 보다 먼저 와서 파일을 기본값으로 덮었다(감사 #42).
3. Vue가 사용자 입력 시 set_automation_settings 액션 호출
   → 호스트 dict 에 **보낸 키만** 합침(:func:`apply_automation_payload`) → ``save_mode_settings()``
   (내용이 같으면 쓰지 않음). 파일 값을 아직 못 받은 Vue 는 모르는 키를 보내지 않는다.
4. 백엔드 모드 전환 (BACKEND_CHANGED 이벤트):
   → 이전 모드 저장
   → 새 모드 로드 → 호스트 dict 갱신 → Vue 시그널 발행
"""
from __future__ import annotations

import json
import math
from typing import Any, Optional

from core.mode_aware_mixin import ModeAwareMixin
from utils.app_logger import get_logger

_logger = get_logger("mode_aware_auto")


# 기본값 — 디스크에 파일 없을 때 (Vue App.vue autoSettings 초기값과 같게 유지)
_DEFAULT_AUTOMATION_SETTINGS: dict[str, Any] = {
    "mode": "count",
    "limit": 10,
    "repeat": 1,
    "delay": 1.0,
    "allowDupes": False,
    "autoResetDeck": False,
    "maxRetries": 2,
    # 대기열 정기 정리 — N장마다 체크포인트를 VRAM 에서 내린다(Forge unload-checkpoint · ComfyUI /free).
    # 항목 사이에서만 하고 마지막 장 뒤엔 하지 않는다(0 = 안 함). 예전엔 영속 키에서 빠져 재시작마다 0이 됐다
    "cleanupEveryN": 0,
}


def _as_float(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _as_int(value: Any, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return int(default)


def normalize_automation_settings(settings: Any) -> dict:
    """직렬화할 자동화 설정 한 벌 — 알려진 키만, 타입 정규화, 없는 키는 기본값.

    저장(collect)·로드(apply)·Vue 응답(snapshot)이 모두 이 모양을 쓴다. 같은 값이면 같은 dict
    가 나오므로 save_mode_settings 의 '바뀐 경우에만 쓰기' 비교가 성립한다.
    """
    source = settings if isinstance(settings, dict) else {}
    merged = dict(_DEFAULT_AUTOMATION_SETTINGS)
    merged.update({key: source[key] for key in _DEFAULT_AUTOMATION_SETTINGS if key in source})
    return {
        "mode": str(merged["mode"]),
        "limit": _as_float(merged["limit"], 10.0),
        "repeat": _as_int(merged["repeat"], 1),
        "delay": _as_float(merged["delay"], 1.0),
        "allowDupes": bool(merged["allowDupes"]),
        "autoResetDeck": bool(merged["autoResetDeck"]),
        "maxRetries": _as_int(merged["maxRetries"], 2),
        "cleanupEveryN": max(0, _as_int(merged["cleanupEveryN"], 0)),
    }


def _finite_float(value: Any) -> Optional[float]:
    if isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return number if math.isfinite(number) else None


def _finite_int(value: Any) -> Optional[int]:
    number = _finite_float(value)
    return None if number is None else int(number)


def merge_automation_payload(current: Any, payload: Any) -> dict:
    """``set_automation_settings`` 페이로드를 현재 설정에 **들어 있는 키만** 합친다.

    Vue 는 파일 값을 아직 못 받았을 때(부팅 hydrate 가 늦거나 응답이 없을 때) 자기가 모르는 키를
    보내지 않는다. 예전 핸들러는 dict 를 통째로 바꾸며 빠진 키를 ``payload.get(k, 기본값)`` 으로
    채워, 그런 동기화 한 번에 파일의 사용자 값이 전부 기본값으로 저장됐다(R2b#1).
    빠진 키와 타입이 잘못된 값은 현재 값을 그대로 둔다.
    """
    merged = normalize_automation_settings(current)
    if not isinstance(payload, dict):
        return merged
    if "mode" in payload and isinstance(payload["mode"], str) and payload["mode"].strip():
        merged["mode"] = payload["mode"].strip()
    for key in ("limit", "delay"):
        if key in payload:
            number = _finite_float(payload[key])
            if number is not None:
                merged[key] = number
    for key in ("repeat", "maxRetries", "cleanupEveryN"):
        if key in payload:
            number = _finite_int(payload[key])
            if number is not None:
                merged[key] = max(0, number) if key == "cleanupEveryN" else number
    for key in ("allowDupes", "autoResetDeck"):
        if key in payload and payload[key] is not None:
            merged[key] = bool(payload[key])
    return merged


def apply_settings_to_queue(host: Any, settings: dict) -> None:
    """대기열 매니저가 직접 읽는 값(간격·정기 정리)을 설정에 맞춘다(없으면 무시)."""
    queue = getattr(host, "queue_manager", None)
    if queue is None:
        return
    try:
        queue.cleanup_every_n = int(settings.get("cleanupEveryN", 0))
        queue.delay_seconds = max(0.0, float(settings.get("delay", 1.0)))
    except Exception:
        _logger.exception("대기열 자동화 값 반영 실패")


def apply_automation_payload(host: Any, payload: Any) -> dict:
    """``set_automation_settings`` 액션 본체 (ui/generator_main._handle_vue_action 이 부른다).

    - 영속 키는 :func:`merge_automation_payload` 로 **보낸 키만** 합친다.
    - 자동 NL·Ollama 값도 페이로드에 있을 때만 바꾼다(없으면 이전 값 유지).
    - 대기열 런타임 값은 합친 결과로 맞추고, 모드별 파일에 저장한다(내용이 같으면 쓰지 않음).
    """
    data = payload if isinstance(payload, dict) else {}
    settings = merge_automation_payload(getattr(host, "_vue_automation_settings", None), data)
    host._vue_automation_settings = settings
    # 생성 시 태그→자연어 자동 변환 (자동화 루프에서 nl_caption 적용)
    if "autoNl" in data:
        host._auto_nl_enabled = bool(data.get("autoNl"))
    if "ollamaUrl" in data:
        from core.ollama_client import DEFAULT_OLLAMA_URL
        host._auto_nl_url = str(data.get("ollamaUrl") or DEFAULT_OLLAMA_URL)
    if "ollamaModel" in data:
        host._auto_nl_model = str(data.get("ollamaModel") or "")   # 비면 워커가 설치 모델로 정한다
    apply_settings_to_queue(host, settings)
    persistence = getattr(host, "automation_persistence", None)
    if persistence is not None:
        persistence.save_mode_settings()
    return settings


class ModeAwareAutomationSettings(ModeAwareMixin):
    """``_vue_automation_settings`` 어댑터.

    호스트 객체에 ``_vue_automation_settings`` (dict)와 ``vue_bridge``
    (automationSettingsLoaded 시그널 보유) 속성이 있어야 한다.
    """

    settings_base_filename = "automation_settings"

    def __init__(self, host: Any):
        self.host = host

    # ─────────────────────────────────────────
    # ModeAwareMixin 구현
    # ─────────────────────────────────────────

    def collect_current_settings(self) -> dict:
        """호스트의 ``_vue_automation_settings``에서 직렬화 가능한 값만 반환."""
        return normalize_automation_settings(
            getattr(self.host, "_vue_automation_settings", None) or {}
        )

    def snapshot(self) -> dict:
        """Vue 가 마운트 때 당겨 가는 현재 모드 설정(getAutomationSettings 슬롯)."""
        return self.collect_current_settings()

    def apply_settings(self, settings: dict, *, emit: bool = True) -> None:
        """디스크에서 로드된 설정 → 호스트 dict 갱신 + 대기열 런타임 반영 + (emit 이면) Vue로 전파."""
        if not isinstance(settings, dict):
            return
        merged = normalize_automation_settings(settings)

        # 호스트 dict 교체
        self.host._vue_automation_settings = merged
        self._apply_to_queue(merged)

        if not emit:
            return
        # Vue로 전파 (시그널 있을 때만 — 단위 테스트나 초기화 미완료 환경 보호)
        bridge = getattr(self.host, "vue_bridge", None)
        if bridge is not None and hasattr(bridge, "automationSettingsLoaded"):
            try:
                bridge.automationSettingsLoaded.emit(json.dumps(merged))
            except Exception:
                _logger.exception("automationSettingsLoaded emit 실패")

    def _apply_to_queue(self, settings: dict) -> None:
        """대기열 매니저가 직접 읽는 값(간격·정기 정리)도 같이 맞춘다.

        set_automation_settings 핸들러만 이 값을 넣었기 때문에, 파일에서 복원하거나 모드를 바꾼
        뒤 사용자가 자동화 패널을 건드리기 전까지 대기열은 옛 값으로 돌았다.
        """
        apply_settings_to_queue(self.host, settings)

    # ─────────────────────────────────────────
    # 편의 메서드
    # ─────────────────────────────────────────

    def initialize(self) -> None:
        """앱 시작 시(__init__, 동기) 1회 호출 — 현재 모드 설정 로드 + 자동 전환 구독.

        이 시점엔 Vue 가 아직 없으므로 시그널을 보내지 않는다 — Vue 는 getAutomationSettings 로
        당겨 간다. 파일이 없으면 (첫 실행) 기본값을 저장해 다음 실행부터 일관.
        """
        loaded = False
        try:
            path = self._settings_path()
            if path.exists():
                self.apply_settings(json.loads(path.read_text(encoding="utf-8")), emit=False)
                loaded = True
        except Exception:
            _logger.exception("automation settings load failed")
        if not loaded:
            self.apply_settings(_DEFAULT_AUTOMATION_SETTINGS, emit=False)
            self.save_mode_settings()

        # 백엔드 전환 자동화 (이전 모드 저장 + 새 모드 로드)
        self.auto_subscribe_to_mode_change()
