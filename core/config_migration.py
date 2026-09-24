# core/config_migration.py
"""JSON 설정 파일의 스키마 버전 관리 + 마이그레이션.

앞으로 설정 필드가 추가/제거/이름변경될 때, 기존 사용자 파일을 조용히
현재 스키마로 올려주기 위한 공용 로더. 기존 파일이 schema_version을
포함하지 않으면 0으로 간주한다.
"""
from __future__ import annotations

import json
import logging
import os
import shutil
from typing import Callable

from utils.atomic_json import atomic_write_json

logger = logging.getLogger(__name__)

# 각 설정 파일별 마이그레이션 맵을 이 모듈에 중앙화.
# 키: 파일의 논리 이름 — 값: {from_version: migrator(dict) -> dict}
_UI_PREFS_MIGRATIONS: dict[int, Callable[[dict], dict]] = {
    # 향후 추가 예시:
    # 0: lambda d: {**d, 'new_field': False},
    # 1: lambda d: {...},
}
UI_PREFS_CURRENT_VERSION = 1

ICON_ANIMATION_STYLE_DEFAULT = "none"
ICON_ANIMATION_STYLES = frozenset({"none", "claude", "gpt"})


def normalize_icon_animation_style(value: object) -> str:
    """Return the closed icon-animation preference contract.

    The preference exists before any animation implementation, so an unknown
    value must stay inert instead of enabling an effect in an older app build.
    """
    if isinstance(value, str) and value in ICON_ANIMATION_STYLES:
        return value
    return ICON_ANIMATION_STYLE_DEFAULT


def _normalize_ui_prefs(data: dict) -> dict:
    normalized = dict(data)
    normalized["iconAnimationStyle"] = normalize_icon_animation_style(
        normalized.get("iconAnimationStyle")
    )
    return normalized


def load_and_migrate(
    path: str,
    migrations: dict[int, Callable[[dict], dict]],
    current_version: int,
) -> dict:
    """설정을 읽고 필요한 마이그레이션을 적용 후 반환.

    - 파일이 없으면 {} 반환
    - schema_version 누락 시 0으로 시작
    - 각 단계 실패 시 .bak 백업 생성 후 빈 dict로 fallback
    """
    if not os.path.exists(path):
        return {}

    try:
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError) as e:
        logger.error("config load failed (%s): %s", path, e)
        return {}

    if not isinstance(data, dict):
        return {}

    version = int(data.get('schema_version', 0) or 0)
    if version == current_version:
        return data

    if version > current_version:
        logger.warning(
            "config %s has newer schema_version %d > %d; using as-is",
            path, version, current_version,
        )
        return data

    # 단계별 마이그레이션
    bak = path + '.bak'
    try:
        shutil.copy2(path, bak)
    except OSError:
        # 백업 실패는 치명적이지 않음 — 진행은 계속
        logger.warning("backup failed for %s", path)

    while version < current_version:
        migrator = migrations.get(version)
        if migrator is None:
            logger.warning("no migrator from v%d — stopping", version)
            break
        try:
            data = migrator(data)
        except Exception as e:
            logger.exception("migration v%d → v%d failed for %s: %s",
                             version, version + 1, path, e)
            break
        version += 1
        data['schema_version'] = version
    return data


def save_with_version(path: str, data: dict, schema_version: int) -> None:
    """schema_version 필드를 주입하고 저장."""
    out = dict(data)
    out['schema_version'] = schema_version
    # 원자적 쓰기는 공용 구현 한 벌(fsync + 실패 시 tmp 정리)을 쓴다.
    atomic_write_json(path, out, indent=2)


def load_ui_prefs(path: str) -> dict:
    data = load_and_migrate(path, _UI_PREFS_MIGRATIONS, UI_PREFS_CURRENT_VERSION)
    return _normalize_ui_prefs(data)


def save_ui_prefs(path: str, data: dict) -> None:
    save_with_version(
        path,
        _normalize_ui_prefs(data),
        UI_PREFS_CURRENT_VERSION,
    )


def read_legacy_gallery_folder(txt_path: str) -> str:
    """옛 config/gallery_last_folder.txt 의 Gallery 폴더 — 이 PC 에 실제로 있을 때만.

    이 파일은 .gitignore 규칙보다 먼저 추적돼 저장소에 다른 PC 경로가 남아 있다. 새 클론은
    ui_prefs.galleryFolder 가 비어 있어 이 값을 흡수하는데, 존재 확인 없이 옮기면 갤러리가
    없는 폴더에 고정돼 빈 목록으로 남았다. generator_settings 의 레거시 흡수와 같이 isdir 만 받는다.
    """
    try:
        with open(txt_path, 'r', encoding='utf-8-sig') as f:
            folder = f.read().strip()
    except (OSError, UnicodeDecodeError):
        return ''
    return folder if folder and os.path.isdir(folder) else ''
