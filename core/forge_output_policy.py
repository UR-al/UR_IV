# core/forge_output_policy.py
"""Forge/A1111 쪽 이미지 저장(save_images) 정책 — 순수 로직 + ui_prefs 캐시.

앱은 생성 결과를 받은 뒤 generated_images 에 직접 저장한다. 그런데 T2I·I2I·Inpaint·
PNG Info·갤러리 즉시 생성 payload 가 ``save_images=True`` 를 보내면 Forge 도
``do_not_save_samples=False`` 로 받아 자기 data-dir/output(앱 사용자에겐 숨은 폴더)에 같은
이미지를 한 번 더 저장한다(4MB 가 넘으면 JPG 도 추가). 그래서 백엔드가 POST 직전에
이 정책으로 ``save_images`` 를 확정한다:

- 기본(설정 꺼짐): 항상 False — 중복 저장 없음
- 설정 ``forgeSaveOutputs`` 켜짐: 호출부가 요청한 값 그대로(True 를 보낸 경로만 Forge 도 저장)

호출부가 False 를 보낸 경로(채팅·손 재구성·백엔드 후처리)는 설정과 무관하게 False 다.
ComfyUI 백엔드는 이 설정을 쓰지 않는다. 앱 컴파일러가 payload 의 ``save_images`` 를 그대로
따라 True(메인 생성)면 코어 SaveImage(ComfyUI/output 사본), 그 밖에는 코어 PreviewImage(temp)로
결과를 돌려준다(core/comfy_workflow_compiler.py ``_add_output_image``).

설정값은 GUI 가 메모리로 밀어 넣는다(:func:`update_forge_save_outputs_from_prefs`). 생성
워커 스레드가 ui_prefs.json 을 열면 GUI 의 원자적 교체(os.replace)가 Windows 에서 실패한다.
"""
from __future__ import annotations

import json
import os
import threading
from typing import Any, Mapping, Optional

PREF_KEY = "forgeSaveOutputs"


def forge_save_outputs_enabled(prefs: Optional[Mapping]) -> bool:
    if not isinstance(prefs, Mapping):
        return False
    return prefs.get(PREF_KEY) is True


def effective_save_images(requested: Any, enabled: bool) -> bool:
    """Forge 에 보낼 save_images — 요청했고(True) 사용자가 Forge 저장을 켰을 때만 True."""
    return requested is True and bool(enabled)


def apply_save_policy(payload: Mapping[str, Any], enabled: bool) -> dict:
    """``save_images`` 를 정책대로 확정한 새 payload (입력은 건드리지 않는다)."""
    out = dict(payload or {})
    out["save_images"] = effective_save_images(out.get("save_images", False), enabled)
    return out


class _PrefsFlagCache:
    """ui_prefs.json 의 불리언 하나를 mtime 기준으로 캐시 — 생성마다 JSON 을 다시 읽지 않는다.

    Vue 저장·설정 백업 복원·웹 모드 등 어떤 경로로 파일이 바뀌어도 mtime 이 바뀌면 다시 읽는다.
    """

    def __init__(self, key: str, path_factory=None):
        self._key = key
        self._path_factory = path_factory
        self._lock = threading.Lock()
        self._stamp: Optional[tuple[str, float, int]] = None
        self._value = False

    def _path(self) -> str:
        if self._path_factory is not None:
            return str(self._path_factory())
        # 경로는 core.ui_prefs 한 곳 — 로더는 이 캐시의 raw 읽기(워커 스레드·mtime)를 유지한다
        from core.ui_prefs import ui_prefs_path
        return ui_prefs_path()

    def get(self) -> bool:
        try:
            path = self._path()
            stat = os.stat(path)
        except (OSError, ValueError):
            return False
        stamp = (path, stat.st_mtime, stat.st_size)
        with self._lock:
            if stamp == self._stamp:
                return self._value
        try:
            with open(path, "r", encoding="utf-8") as handle:
                prefs = json.load(handle)
        except (OSError, ValueError):
            return False
        value = prefs.get(self._key) is True if isinstance(prefs, dict) else False
        with self._lock:
            self._stamp, self._value = stamp, value
        return value


_cache = _PrefsFlagCache(PREF_KEY)


def forge_save_outputs_from_prefs_file() -> bool:
    """현재 ui_prefs.json 의 ``forgeSaveOutputs`` (없거나 못 읽으면 False).

    생성 경로는 이 함수가 아니라 :func:`forge_save_outputs_setting` 을 쓴다 — 여기는 GUI 가
    아직 값을 밀어 넣지 않았을 때의 폴백이다.
    """
    return _cache.get()


class _PushedFlag:
    """GUI 스레드가 밀어 넣는 설정값 — 생성 워커 스레드는 **메모리 값만** 읽는다.

    Windows 에선 읽기로 열려 있는 파일을 ``os.replace`` 로 교체할 수 없다(PermissionError,
    WinError 5). 생성 워커(GenerationFlowWorker·채팅·API 스레드)가 ``_generate`` 마다
    ui_prefs.json 을 열면, 그 순간 GUI 스레드의 save_ui_prefs(LoRA 슬라이더 드래그는 input
    이벤트마다 저장한다)가 교체에 실패해 설정 쓰기가 유실됐다. 그래서 값의 주인은
    save_ui_prefs 핸들러와 부팅 복원(_restore_runtime_prefs)이고, 파일 읽기는 한 번도
    값을 받지 못했을 때(부팅 복원 전·GUI 없는 실행)만 쓰는 폴백이다.
    """

    def __init__(self, fallback):
        self._fallback = fallback
        self._lock = threading.Lock()
        self._value: Optional[bool] = None

    def set(self, enabled: bool) -> None:
        with self._lock:
            self._value = enabled is True

    def reset(self) -> None:
        with self._lock:
            self._value = None

    def is_set(self) -> bool:
        with self._lock:
            return self._value is not None

    def get(self) -> bool:
        with self._lock:
            value = self._value
        if value is not None:
            return value
        return bool(self._fallback())


# 폴백은 호출 시점에 이름으로 찾는다(테스트가 모듈 함수를 바꿔 끼울 수 있게).
_setting = _PushedFlag(lambda: forge_save_outputs_from_prefs_file())


def set_forge_save_outputs(enabled: bool) -> None:
    """설정값을 메모리에 반영 (GUI 스레드: save_ui_prefs 핸들러·부팅 복원)."""
    _setting.set(enabled)


def update_forge_save_outputs_from_prefs(prefs: Optional[Mapping]) -> None:
    """병합된 ui_prefs dict 로 설정값 갱신 — 키가 없거나 True 가 아니면 꺼짐."""
    _setting.set(forge_save_outputs_enabled(prefs))


def forge_save_outputs_setting() -> bool:
    """생성 요청이 쓸 ``forgeSaveOutputs`` — 밀어 넣은 값, 없으면 파일 폴백."""
    return _setting.get()
