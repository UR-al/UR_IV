"""Named chat/assist instruction snapshots. CRUD never changes live settings."""
from __future__ import annotations

import json
import threading
import uuid
from pathlib import Path

from core.ai_assist_instructions import normalize_instructions
from core.storage_paths import config_file
from utils.atomic_json import atomic_write_json

_LOCK = threading.RLock()
MAX_PRESETS = 100


def _scope(scope):
    if scope not in ('chat', 'assist'):
        raise ValueError('프리셋 종류는 chat 또는 assist여야 합니다')
    return scope


def _content(scope, value):
    if scope == 'assist':
        return normalize_instructions(value, strict=True)
    if not isinstance(value, str) or len(value) > 32000:
        raise ValueError('대화 지침은 32,000자 이하의 문자열이어야 합니다')
    return value


def _path(path):
    return Path(path if path is not None else config_file('instruction_presets.json'))


def _read(path):
    if not path.exists():
        return {'version': 1, 'presets': []}
    try:
        if path.stat().st_size > 64_000_000:
            raise ValueError('파일 용량 초과')
        data = json.loads(path.read_text(encoding='utf-8'))
        if not isinstance(data, dict) or data.get('version') != 1 or not isinstance(data.get('presets'), list):
            raise ValueError('파일 형식 오류')
        ids = set()
        for item in data['presets']:
            if not isinstance(item, dict) or not isinstance(item.get('id'), str) or not item['id'] or item['id'] in ids:
                raise ValueError('중복 또는 잘못된 프리셋 ID')
            ids.add(item['id'])
            if not isinstance(item.get('name'), str) or not item['name'].strip() or len(item['name']) > 80:
                raise ValueError('잘못된 프리셋 이름')
            _content(_scope(item.get('scope')), item.get('instructions'))
        return data
    except (OSError, ValueError, TypeError, RecursionError) as exc:
        raise ValueError('기존 프리셋 파일을 읽지 못했습니다. 원본을 덮어쓰지 않았습니다.') from exc


def list_presets(scope, path=None):
    _scope(scope)
    with _LOCK:
        return [item for item in _read(_path(path))['presets'] if item['scope'] == scope]


def save_preset(scope, name, instructions, *, preset_id=None, path=None):
    _scope(scope)
    if not isinstance(name, str) or not name.strip() or len(name.strip()) > 80:
        raise ValueError('프리셋 이름은 1~80자로 입력해 주세요')
    content = _content(scope, instructions)
    with _LOCK:
        target = _path(path)
        data = _read(target)
        entries = data['presets']
        existing = next((p for p in entries if p['scope'] == scope and p['id'] == preset_id), None)
        if preset_id is not None and existing is None:
            raise ValueError('변경할 프리셋을 찾지 못했습니다. 목록을 새로 불러오세요')
        if any(p['scope'] == scope and p['name'].casefold() == name.strip().casefold() and p is not existing for p in entries):
            raise ValueError('같은 이름의 프리셋이 있습니다. 다른 이름을 사용하거나 선택한 프리셋을 덮어쓰세요')
        if existing is None and sum(p['scope'] == scope for p in entries) >= MAX_PRESETS:
            raise ValueError('종류별 최대 100개까지 저장할 수 있습니다')
        item = {'id': existing['id'] if existing else uuid.uuid4().hex, 'scope': scope,
                'name': name.strip(), 'instructions': content}
        if existing is not None:
            entries[entries.index(existing)] = item
        else:
            entries.append(item)
        atomic_write_json(str(target), data)
        return item


def delete_preset(scope, preset_id, path=None):
    _scope(scope)
    with _LOCK:
        target = _path(path)
        data = _read(target)
        kept = [p for p in data['presets'] if not (p['scope'] == scope and p['id'] == preset_id)]
        if len(kept) == len(data['presets']):
            raise ValueError('삭제할 프리셋을 찾지 못했습니다')
        data['presets'] = kept
        atomic_write_json(str(target), data)
