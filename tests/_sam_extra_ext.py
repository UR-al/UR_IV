"""sam-extra 확장 가드 테스트 공용 — 설치 위치 찾기와 skip/강제 규칙 한 곳.

찾는 순서는 `core.sam_extra_scan.extension_dir_candidates`: 환경 변수 AISTUDIO_FORGE_EXTENSION_DIR →
앱이 쓰는 Forge 확장 폴더(config/backend_runtime.json) → 알려진 Forge 설치. 폴더 이름은
forge_sam3_extension 과 sam-extra 둘 다 본다. 환경 변수를 지정했으면 그 경로만 본다 — 틀린 경로가 조용히
다른 설치로 바뀌지 않게(`find_installed_extension`).

확장이 없으면 skip 하되 이유(찾아본 곳)를 남긴다. AISTUDIO_REQUIRE_FORGE_EXT=1 이면 skip 대신 실패한다
(확장이 꼭 있어야 하는 검증 — /verify 나 확장 동기화 작업 중).
"""
from __future__ import annotations

import os
import unittest

from core.sam_extra_scan import (
    EXTENSION_DIR_ENV,
    REQUIRE_EXTENSION_ENV,
    configured_extension_dir,
    extension_dir_candidates,
    find_installed_extension,
)

EXT_ROOT = find_installed_extension()
REQUIRED = os.environ.get(REQUIRE_EXTENSION_ENV, "").strip() == "1"


def skip_reason() -> str:
    configured = configured_extension_dir()
    if configured is not None:
        return (f"{EXTENSION_DIR_ENV}={configured} 은(는) sam-extra 확장 폴더가 아니다(scripts/ 와 sam3ext/ 가 "
                "있어야 한다) — 다른 설치로 대신하지 않는다. 경로를 고치거나 변수를 지워라")
    tried = ", ".join(str(path) for path in extension_dir_candidates()[:6])
    return f"sam-extra 확장 미설치 — 찾아본 곳: {tried} (다른 위치는 {EXTENSION_DIR_ENV} 로 지정)"


def requires_extension(cls):
    """테스트 클래스 데코레이터: 확장이 없으면 skip, 강제 모드면 setUpClass 에서 실패."""
    if EXT_ROOT is not None:
        return cls
    if not REQUIRED:
        return unittest.skip(skip_reason())(cls)

    def _fail(_klass):
        raise AssertionError(f"{REQUIRE_EXTENSION_ENV}=1 인데 {skip_reason()}")

    cls.setUpClass = classmethod(_fail)
    return cls
