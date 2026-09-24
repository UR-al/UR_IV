"""Persist schema editor drafts, including incomplete JSON, without applying them."""
from __future__ import annotations

import json
import threading
from pathlib import Path

from core.config_migration import save_ui_prefs
from core.storage_paths import config_file
from core.ui_prefs import UI_PREFS_NAME

_LOCK = threading.RLock()


def save_schema_draft(text, path=None):
    if not isinstance(text, str) or len(text) > 64000:
        raise ValueError('스키마 입력은 64,000자 이하의 문자열이어야 합니다')
    target = Path(path if path is not None else config_file(UI_PREFS_NAME))
    with _LOCK:
        try:
            prefs = json.loads(target.read_text(encoding='utf-8')) if target.exists() else {}
            if not isinstance(prefs, dict) or not isinstance(prefs.get('chatSettingsV2', {}), dict):
                raise ValueError('잘못된 설정 형식')
        except (OSError, ValueError, RecursionError) as exc:
            raise ValueError('기존 설정 파일을 읽지 못해 저장하지 않았습니다. 입력 내용은 유지됩니다.') from exc
        settings = dict(prefs.get('chatSettingsV2', {}))
        settings['schemaText'] = text
        prefs['chatSettingsV2'] = settings
        save_ui_prefs(str(target), prefs)
    return text
