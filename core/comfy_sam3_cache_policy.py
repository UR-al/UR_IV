# core/comfy_sam3_cache_policy.py
"""ComfyUI SAM3 모델 보관 설정 — 순수 로직 + ui_prefs 캐시 (Qt 비의존).

번들 노드팩의 ``ForgeNeoSAM3Mask`` 는 불러온 SAM3 번들(약 3.4GB)을 다음 SAM3 작업까지
보관한다('사용 후 언로드'면 ComfyUI 의 CPU RAM, 끄면 장치에 그대로). Forge 확장의
``sam3_unload_keep_in_ram`` 설정과 같은 역할을 앱 설정 ``comfySam3KeepInRam`` 이 한다:

- 켜짐(기본): '사용 후 언로드' 뒤에도 CPU RAM 에 두었다가 다음 검출 때 장치로 옮기기만 한다
  (이미지마다 체크포인트를 다시 읽지 않는다). 결과 이미지는 같다.
- 꺼짐: '사용 후 언로드'면 매번 새로 읽고, 보관 중인 사본도 다음 SAM3 작업에서 해제한다.
- '사용 후 언로드'를 끈 SAM3 는 Forge 처럼 설정과 무관하게 장치에 둔다.

어느 경우든 ComfyUI 의 /free(앱의 '생성 후 모델 언로드')가 보관본을 해제한다 — 노드팩이
``comfy.model_management.unload_all_models`` 에 해제 훅을 건다. 컴파일러는
``ComfyWorkflowCompiler(sam3_keep_in_ram=...)`` 로 이 값을 받아 ``cache_model`` 에 싣는다.
"""
from __future__ import annotations

import json
import os
import threading
from typing import Any, Callable, Mapping, Optional

PREF_KEY = "comfySam3KeepInRam"
DEFAULT_KEEP_IN_RAM = True


def comfy_sam3_keep_in_ram_enabled(prefs: Optional[Mapping[str, Any]]) -> bool:
    """ui_prefs 의 ``comfySam3KeepInRam`` — 불리언이 아니거나 없으면 기본값(켜짐)."""
    if not isinstance(prefs, Mapping):
        return DEFAULT_KEEP_IN_RAM
    value = prefs.get(PREF_KEY)
    return value if isinstance(value, bool) else DEFAULT_KEEP_IN_RAM


class _KeepInRamPrefCache:
    """ui_prefs.json 을 mtime·크기 기준으로 캐시 — 워크플로 컴파일마다 JSON 을 다시 읽지 않는다.

    Vue 저장·설정 백업 복원·웹 모드 등 어떤 경로로 파일이 바뀌어도 stamp 가 바뀌면 다시 읽는다.
    파일이 없거나 읽지 못하면 기본값(켜짐)이다.
    """

    def __init__(self, path_factory: Optional[Callable[[], Any]] = None):
        self._path_factory = path_factory
        self._lock = threading.Lock()
        self._stamp: Optional[tuple[str, int, int]] = None
        self._value = DEFAULT_KEEP_IN_RAM

    def _path(self) -> str:
        if self._path_factory is not None:
            return str(self._path_factory())
        # 경로는 core.ui_prefs 한 곳 — 로더는 이 캐시의 raw 읽기(컴파일마다 mtime 비교)를 유지한다
        from core.ui_prefs import ui_prefs_path
        return ui_prefs_path()

    def get(self) -> bool:
        try:
            path = self._path()
            stat = os.stat(path)
        except (OSError, ValueError):
            return DEFAULT_KEEP_IN_RAM
        stamp = (path, int(stat.st_mtime_ns), int(stat.st_size))
        with self._lock:
            if stamp == self._stamp:
                return self._value
        try:
            with open(path, "r", encoding="utf-8") as handle:
                prefs = json.load(handle)
        except (OSError, ValueError):
            return DEFAULT_KEEP_IN_RAM
        value = comfy_sam3_keep_in_ram_enabled(prefs if isinstance(prefs, dict) else None)
        with self._lock:
            self._stamp, self._value = stamp, value
        return value


_cache = _KeepInRamPrefCache()


def comfy_sam3_keep_in_ram_from_prefs_file() -> bool:
    """현재 ui_prefs.json 의 ``comfySam3KeepInRam`` (없거나 못 읽으면 True)."""
    return _cache.get()
