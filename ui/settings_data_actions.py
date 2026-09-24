"""Settings '데이터 · 백업' 카드의 데스크톱 액션 — 설정 백업/복원·앱 재시작·프리셋 공유.

이 기능들(core/settings_backup.py, 재시작, 캐릭터/프롬프트 프리셋 공유)은 setParent(None) 으로
숨겨진 레거시 SettingsTab 버튼에만 연결돼 추가된 날부터 사용자가 쓸 수 없었다(audit #179).
이제 Vue 설정 화면이 아래 액션을 부르고, 파일 대화상자·확인은 호스트(Qt)가, 결과는 토스트가 알린다.

- settings_export  : ZIP 내보내기(대화 기록은 payload.includeChat 일 때만). 가져온 백업을 보존 중이
                     아니면 먼저 현재 설정을 저장해 최신 상태를 담는다.
- settings_import  : ZIP 복원 → 종료 시 자동 저장을 끄고(_preserve_imported_settings_on_quit) 곧바로
                     재시작한다. 그 사이 설정 저장 액션은 generator_main 이 막는다(가져온 파일 보존).
                     복원할 항목이 하나도 없는 ZIP 이면 잠그지도 재시작하지도 않고 경고만 한다.
- restart_app      : 현재 설정 저장(가져온 백업 보존 중이면 건너뜀) 후 재시작(core.app_restart).
                     재시작이 시작되면(_restart_pending) 이 모듈의 액션은 모두 거절한다.
- presets_export/presets_import : 생성 프리셋(presets/*.json) 묶음 공유(core.generation_presets).
- character_presets_export/character_presets_import : 캐릭터 프리셋 공유(utils.character_presets).

웹 모드 권한은 core/web_action_policy.py — 복원·재시작은 웹에서 항상 거부, 나머지 대화상자
액션은 원격 웹 모드에서 거부한다.
"""

from __future__ import annotations

import json
import os
from datetime import datetime
from typing import Any, Callable

#: generator_main._handle_vue_action 이 이 모듈로 넘기는 액션(같은 이름을 리터럴로 비교한다).
SETTINGS_DATA_ACTIONS = (
    'settings_export', 'settings_import', 'restart_app',
    'presets_export', 'presets_import',
    'character_presets_export', 'character_presets_import',
)

#: 복원 직후 재시작까지 가져온 파일을 덮어쓰면 안 되는 저장 액션.
PERSIST_ACTIONS_LOCKED_AFTER_IMPORT = frozenset({
    'save_settings', 'save_ui_prefs', 'save_tab_defaults', 'save_cond_rules',
    'save_global_weights', 'save_preset_by_name', 'delete_preset',
})

RESTART_DELAY_MS = 1200


def persistence_locked(host) -> bool:
    """방금 가져온 백업이 재시작 전까지 우선하는가."""
    return bool(getattr(host, '_preserve_imported_settings_on_quit', False))


def restart_pending(host) -> bool:
    """재시작 대기 프로세스를 이미 띄워 곧 _quit_app 이 도는가.

    두 번째 재시작은 같은 PID 를 기다리는 대기 프로세스를 하나 더 띄워 앱이 두 개(같은 web_profile)
    뜨게 했다. 그 사이의 대화상자 액션도 _quit_app 이 모달 대화상자 도중에 앱을 끄므로 막는다.
    """
    return bool(getattr(host, '_restart_pending', False))


def should_skip_persist_action(host, action: str) -> bool:
    """복원 직후 재시작 전이면 저장 액션을 건너뛴다(가져온 파일을 메모리 상태로 덮지 않게)."""
    return persistence_locked(host) and str(action or '') in PERSIST_ACTIONS_LOCKED_AFTER_IMPORT


def _notify(host, level: str, message: str) -> None:
    bridge = getattr(host, 'vue_bridge', None)
    signal = getattr(bridge, 'showNotification', None)
    if signal is not None:
        signal.emit(level, message)


def _default_dialogs():
    from PyQt6.QtWidgets import QFileDialog

    def save(parent, title, suggested, filters):
        path, _ = QFileDialog.getSaveFileName(parent, title, suggested, filters)
        return path

    def open_(parent, title, filters):
        path, _ = QFileDialog.getOpenFileName(parent, title, '', filters)
        return path

    return save, open_


def _stamp() -> str:
    return datetime.now().strftime('%Y%m%d_%H%M')


def _read_json_file(path: str) -> Any:
    size = os.path.getsize(path)
    if size > 32 * 1024 * 1024:
        raise ValueError('파일이 너무 큽니다(최대 32MB)')
    with open(path, 'r', encoding='utf-8-sig') as f:
        return json.load(f)


def _write_json_file(path: str, data: Any) -> None:
    from utils.atomic_json import atomic_write_json

    atomic_write_json(path, data)


# ── 재시작 ──────────────────────────────────────────────────────────────────

def restart_app(host, *, spawn: Callable[[], object] | None = None,
                schedule: Callable[[int, Callable[[], None]], None] | None = None) -> bool:
    """재시작 도우미를 띄운 뒤 정상 종료 경로(_quit_app)로 끈다. 준비 실패면 앱을 끄지 않는다.

    이미 재시작을 시작했으면(:func:`restart_pending`) 대기 프로세스를 또 띄우지 않는다 — 두 대기
    프로세스가 같은 PID 를 기다렸다가 각자 앱을 띄웠다.
    """
    from core.app_restart import RestartError, spawn_restart_waiter

    if restart_pending(host):
        _notify(host, 'info', '이미 앱을 재시작하는 중입니다')
        return False
    try:
        (spawn or spawn_restart_waiter)()
    except RestartError as exc:
        _notify(host, 'error', f'앱을 재시작하지 못했습니다: {exc}')
        return False
    # 대기 프로세스가 떴을 때만 세운다 — 실패하면 다시 시도할 수 있어야 한다.
    host._restart_pending = True
    message = ('가져온 설정을 유지한 채 앱을 재시작합니다…' if persistence_locked(host)
               else '설정을 저장하고 앱을 재시작합니다…')
    _notify(host, 'info', message)
    if schedule is None:
        from PyQt6.QtCore import QTimer
        schedule = QTimer.singleShot
    # 토스트가 그려질 틈을 준 뒤 기존 종료 경로 — 저장(보존 중이면 건너뜀)·워커 정리·os._exit.
    schedule(RESTART_DELAY_MS, host._quit_app)
    return True


# ── 설정 백업/복원 ─────────────────────────────────────────────────────────

def export_settings(host, payload: dict, *, ask_save=None, export=None) -> str | None:
    from core.settings_backup import export_settings_archive

    save = ask_save or _default_dialogs()[0]
    path = save(host, '설정 백업 내보내기', f'ai_studio_backup_{_stamp()}.zip', 'ZIP (*.zip)')
    if not path:
        return None
    if not persistence_locked(host) and hasattr(host, 'save_settings'):
        try:
            host.save_settings()   # 지금 화면 상태를 담는다
        except Exception as exc:
            _notify(host, 'warning', f'현재 설정 저장 실패 — 마지막 저장본으로 백업합니다: {exc}')
    include_chat = bool((payload or {}).get('includeChat', False))
    count = (export or export_settings_archive)(path, include_chat=include_chat)
    extra = ' (대화 기록 포함)' if include_chat else ''
    _notify(host, 'success', f'설정 백업 완료 — {count}개 파일{extra}: {path}')
    return path


def import_settings(host, payload: dict, *, ask_open=None, importer=None, restart=None) -> int | None:
    from core.settings_backup import import_settings_archive

    open_ = ask_open or _default_dialogs()[1]
    path = open_(host, '설정 백업 가져오기', 'ZIP (*.zip)')
    if not path:
        return None
    count = (importer or import_settings_archive)(path)
    if count <= 0:
        # 허용 목록에 든 항목이 하나도 없는 ZIP(엉뚱한 파일)이면 아무것도 바뀌지 않았다 — 저장을
        # 잠그고 재시작하면 저장하지 않은 현재 설정만 잃는다.
        _notify(host, 'warning', '백업에서 복원할 설정 파일을 찾지 못했습니다 — 이 앱에서 내보낸 백업 ZIP 인지 확인하세요')
        return 0
    # 종료·재시작 때 메모리 상태를 저장하면 방금 가져온 파일을 덮어쓴다 — 이번 실행에서는 막는다.
    host._preserve_imported_settings_on_quit = True
    _notify(host, 'success', f'설정 {count}개를 복원했습니다 — 적용을 위해 앱을 재시작합니다')
    (restart or restart_app)(host)
    return count


# ── 생성 프리셋 공유 ────────────────────────────────────────────────────────

def export_presets(host, payload: dict, *, ask_save=None) -> str | None:
    from core.generation_presets import export_bundle

    bundle = export_bundle()
    if not bundle['presets']:
        _notify(host, 'info', '내보낼 생성 프리셋이 없습니다')
        return None
    save = ask_save or _default_dialogs()[0]
    path = save(host, '생성 프리셋 내보내기', f'generation_presets_{_stamp()}.json', 'JSON (*.json)')
    if not path:
        return None
    _write_json_file(path, bundle)
    _notify(host, 'success', f"생성 프리셋 {len(bundle['presets'])}개를 내보냈습니다: {path}")
    return path


def import_presets(host, payload: dict, *, ask_open=None) -> dict | None:
    from core.generation_presets import import_bundle

    open_ = ask_open or _default_dialogs()[1]
    path = open_(host, '생성 프리셋 가져오기', 'JSON (*.json)')
    if not path:
        return None
    result = import_bundle(_read_json_file(path), overwrite=bool((payload or {}).get('overwrite', False)))
    parts = []
    if result['added']:
        parts.append(f"추가 {len(result['added'])}")
    if result['replaced']:
        parts.append(f"교체 {len(result['replaced'])}")
    if result['skipped']:
        parts.append(f"같은 이름 유지 {len(result['skipped'])}")
    duplicates = result.get('duplicate') or []
    if duplicates:
        # 파일 안에서 대소문자만 다른 이름 — 앞 항목만 가져왔다(core.generation_presets.import_bundle)
        names = ', '.join(f"'{name}'" for name in duplicates[:3]) + (' …' if len(duplicates) > 3 else '')
        parts.append(f"파일 안 중복 이름 {len(duplicates)}개 건너뜀({names})")
    _notify(host, 'warning' if duplicates else 'success',
            '생성 프리셋 가져오기 — ' + (' · '.join(parts) or '변경 없음'))
    return result


# ── 캐릭터 프리셋 공유 ──────────────────────────────────────────────────────

def export_character_presets(host, payload: dict, *, ask_save=None) -> str | None:
    from utils.character_presets import export_character_presets as dump

    data = dump()
    if not data:
        _notify(host, 'info', '내보낼 캐릭터 프리셋이 없습니다')
        return None
    save = ask_save or _default_dialogs()[0]
    path = save(host, '캐릭터 프리셋 내보내기', 'character_presets.json', 'JSON (*.json)')
    if not path:
        return None
    _write_json_file(path, data)
    _notify(host, 'success', f'캐릭터 프리셋 {len(data)}개를 내보냈습니다: {path}')
    return path


def import_character_presets(host, payload: dict, *, ask_open=None) -> dict | None:
    from utils.character_presets import import_character_presets as load

    open_ = ask_open or _default_dialogs()[1]
    path = open_(host, '캐릭터 프리셋 가져오기', 'JSON (*.json)')
    if not path:
        return None
    replace = bool((payload or {}).get('replace', False))
    result = load(_read_json_file(path), replace=replace)
    mode = '전체 교체' if replace else '병합'
    _notify(host, 'success', f"캐릭터 프리셋 {result['imported']}개 가져옴({mode}) — 전체 {result['total']}개")
    return result


def handle_settings_data_action(host, action: str, payload: dict | None) -> None:
    """액션 하나를 실행하고 실패는 토스트로 알린다(앱은 계속 돈다).

    분기를 직접 부른다 — tests/test_web_action_policy 의 대화상자 도달 분석이 이 함수에서
    QFileDialog 까지 따라가야 web_action_policy 목록과 대조할 수 있다.
    """
    data = payload if isinstance(payload, dict) else {}
    if restart_pending(host):
        # 곧 _quit_app 이 돈다 — 대화상자를 여는 도중 앱이 꺼지거나 두 번째 재시작이 뜨지 않게 한다.
        _notify(host, 'info', '앱을 재시작하는 중이라 이 작업을 실행하지 않았습니다')
        return
    try:
        if action == 'settings_export':
            export_settings(host, data)
        elif action == 'settings_import':
            import_settings(host, data)
        elif action == 'restart_app':
            restart_app(host)
        elif action == 'presets_export':
            export_presets(host, data)
        elif action == 'presets_import':
            import_presets(host, data)
        elif action == 'character_presets_export':
            export_character_presets(host, data)
        elif action == 'character_presets_import':
            import_character_presets(host, data)
    except Exception as exc:  # 대화상자·파일·검증 오류 — 사용자에게 그대로 알린다
        _notify(host, 'error', f'작업 실패: {exc}')
