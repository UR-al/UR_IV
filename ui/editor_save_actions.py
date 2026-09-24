"""에디터 '저장' / '다른 이름으로 저장' 액션 — Qt 쪽 얇은 접착층.

순수 로직(대상 판정·메타데이터·인코딩·원자적 쓰기)은 ``core.editor_save`` 에 있고,
여기서는 (1) 필요할 때 저장 대화상자를 UI 스레드에서 띄우고 (2) 인코딩은 작업
스레드로 넘기고 (3) 결과를 ``editorSaveResult`` 시그널로 Vue 에 돌려준다.

Vue 는 이 결과를 받은 **뒤에만** '저장됨(깨끗함)' 상태로 바꾼다. 예전에는 요청을
보내자마자 isDirty=false 로 바꿔서, 실패해도 닫기 경고와 자동저장이 꺼졌다.

'저장'은 대화상자를 띄우지 않고, 사용자가 연 원본도 덮어쓰지 않는다(비파괴 저장 —
``core.editor_save`` 참고). 첫 저장은 원본 옆(임시 원본이면 기본 출력 폴더)에 사본을
새로 만들고, Vue 가 ``overwrite_source`` 로 '이 문서가 저장한 사본'이라고 알려 오면
그 파일이 정말 이번 실행에서 에디터가 쓴 것(``SavedCopyRegistry``)일 때만 덮어쓴다.

결과 페이로드: ``{request_id, mode, ok, path?, format?, width?, height?, unchanged?,
owned?, alpha_png?, cancelled?, error?, snapshot_path?, replaced_paths?}``
(``owned`` = 이 문서의 다음 '저장'이 ``path`` 를 덮어써도 되는지)
"""
from __future__ import annotations

import json
import os
import threading
from typing import Callable, Optional

from core.editor_save import (
    SAVE_DIALOG_FILTERS,
    EditorSaveError,
    SavedCopyRegistry,
    app_owned_dirs,
    edited_copy_path,
    ensure_extension,
    format_for_path,
    is_inside,
    non_user_dirs,
    save_edited_image,
    suggest_save_path,
    validate_chosen_target,
)
from core.path_safety import safe_input_path

APP_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EDITOR_TEMP_DIR = os.path.join(APP_ROOT, 'image_cache', 'editor_temp')

# 이번 실행에서 에디터가 쓴 파일 — '저장'이 덮어써도 되는 유일한 대상
SAVED_COPIES = SavedCopyRegistry()
# 사본 이름 고르기(unique_path)와 쓰기 사이에 다른 저장(웹 모드의 다른 탭)이 끼어들어
# 같은 이름을 고르지 않게 저장 작업을 한 줄로 세운다.
_SAVE_LOCK = threading.Lock()

AskPath = Callable[[str, str], tuple[str, str]]   # (title, suggested) -> (path, selected_filter)
Runner = Callable[[Callable[[], None]], None]


def _default_output_dir() -> str:
    try:
        import config
        return str(getattr(config, 'OUTPUT_DIR', '') or APP_ROOT)
    except Exception:
        return APP_ROOT


def _qt_ask_path(window) -> AskPath:
    def ask(title: str, suggested: str) -> tuple[str, str]:
        from PyQt6.QtWidgets import QFileDialog
        path, selected = QFileDialog.getSaveFileName(window, title, suggested, SAVE_DIALOG_FILTERS)
        return path or '', selected or ''
    return ask


def _thread_runner(fn: Callable[[], None]) -> None:
    threading.Thread(target=fn, daemon=True, name='ai-studio-editor-save').start()


def _emit(window, result: dict) -> None:
    bridge = getattr(window, 'vue_bridge', None)
    signal = getattr(bridge, 'editorSaveResult', None)
    if signal is not None:
        # 작업 스레드 emit 은 queued connection 이라 안전하다
        signal.emit(json.dumps(result, ensure_ascii=False))


def handle_editor_save_action(window, action: str, payload: dict, *,
                              ask_path: Optional[AskPath] = None,
                              run: Optional[Runner] = None,
                              app_root: str = APP_ROOT,
                              output_dir: Optional[str] = None,
                              snapshot_dir: Optional[str] = None,
                              registry: Optional[SavedCopyRegistry] = None) -> None:
    """``editor_save`` / ``editor_save_as`` 처리. 결과는 항상 한 번 emit 한다."""
    payload = payload if isinstance(payload, dict) else {}
    save_as = action == 'editor_save_as'
    base = {'request_id': payload.get('request_id'), 'mode': 'save_as' if save_as else 'save'}
    ask_path = ask_path or _qt_ask_path(window)
    run = run or _thread_runner
    registry = registry if registry is not None else SAVED_COPIES
    snapshot_dir = snapshot_dir or (
        EDITOR_TEMP_DIR if app_root == APP_ROOT else os.path.join(app_root, 'image_cache', 'editor_temp'))

    def fail(message: str, **extra) -> None:
        _emit(window, {**base, 'ok': False, 'error': message, **extra})

    try:
        edited = safe_input_path(str(payload.get('path') or ''))
        if not edited:
            fail('저장할 편집 이미지를 찾을 수 없습니다')
            return
        raw_source = str(payload.get('source_path') or '')
        source = safe_input_path(raw_source) if raw_source else None
        excluded = non_user_dirs(app_root)
        default_dir = output_dir or _default_output_dir()
        # None = 작업 스레드에서 사본 이름을 고른다(잠금 안에서 — 이름 경합 방지)
        target: Optional[str] = None

        if save_as:
            # 추천 이름은 원본 이름을 따른다(사용자 폴더면 그 옆, 임시면 기본 출력 폴더)
            suggested = suggest_save_path(source, default_dir, excluded)
            chosen, selected_filter = ask_path('다른 이름으로 저장', suggested)
            if not chosen:
                _emit(window, {**base, 'ok': False, 'cancelled': True})
                return
            target = os.path.abspath(ensure_extension(chosen, selected_filter))
            if format_for_path(target) is None:
                fail('PNG · JPEG · WebP 로만 저장할 수 있습니다')
                return
            if is_inside(target, app_owned_dirs(app_root)):
                # 앱이 스스로 정리하는 폴더 — 여기 저장하면 며칠 뒤 사라진다
                fail('앱 임시 폴더에는 저장할 수 없습니다 — 다른 폴더를 고르세요')
                return
            # 사용자가 대화상자로 고른 곳 — 시스템 폴더만 막는다(USB 드라이브 루트는 된다)
            problem = validate_chosen_target(target)
            if problem:
                fail(problem)
                return
        elif (payload.get('overwrite_source') is True and source and source in registry
              and format_for_path(source) and not is_inside(source, app_owned_dirs(app_root))):
            # 이 문서가 앞서 저장한 사본 — 그 사본만 갱신한다. 원본은 여기 올 수 없다
            # (registry 에는 에디터가 이번 실행에서 쓴 파일만 있다).
            target = source

        protect = payload.get('protect_paths')
        protect_paths = [p for p in protect if isinstance(p, str)] if isinstance(protect, list) else []
        overlay = payload.get('overlay_base64') or None
        try:
            opacity = float(payload.get('overlay_opacity', 100)) / 100.0
        except (TypeError, ValueError):
            opacity = 1.0
        # 생성 파라미터는 원본에서 가져온다(원본이 임시 파일이어도 담겨 있으면 이어받는다).
        # 덮어쓰는 경우 core 가 쓰기 **전에** 읽는다.
        metadata_path = source or None
        # 편집이 없으면 '변경 없음'으로 끝내도 되는 원본 — 사용자 폴더에 이미 있는 파일만
        keep_as_is = source if (not save_as and source and not is_inside(source, excluded)) else None
    except Exception as exc:   # 어떤 경우에도 Vue 가 결과를 못 받고 '저장 중'에 갇히면 안 된다
        fail(f'저장 준비 실패: {exc}')
        return

    def work() -> None:
        try:
            with _SAVE_LOCK:
                path = target
                if path is None:
                    # 원본 옆 <stem>_edited[_N] (임시 원본이면 기본 출력 폴더) — 원본은 그대로 둔다
                    path = edited_copy_path(source, default_dir, excluded)
                    try:
                        os.makedirs(os.path.dirname(path), exist_ok=True)
                    except OSError as exc:
                        raise EditorSaveError(
                            f'저장 폴더를 만들 수 없습니다 ({exc.strerror or exc})') from exc
                result = save_edited_image(
                    edited, path,
                    metadata_path=metadata_path,
                    overlay_base64=overlay,
                    overlay_opacity=opacity,
                    protect_paths=protect_paths,
                    snapshot_dir=snapshot_dir,
                    # '저장'을 사용자 원본을 열기만 하고 눌렀다 — 똑같은 사본을 만들지 않는다.
                    # 임시 원본(클립보드·복구 작업 사본)은 아직 어디에도 저장되지 않았으니 쓴다.
                    unchanged_source=keep_as_is,
                    # 자동으로 정한 대상만 — 사용자가 고른 JPEG 는 그대로 존중한다
                    alpha_to_png=not save_as,
                )
                written = result.get('path') or ''
                if written and (save_as or not result.get('unchanged')):
                    registry.add(written)
                # 이 문서의 다음 '저장'이 이 파일을 덮어써도 되는지 — 원본이면 False
                result['owned'] = bool(written) and written in registry
            _emit(window, {**base, **result})
        except EditorSaveError as exc:
            fail(str(exc))
        except Exception as exc:
            try:
                from core.error_handler import handle_error
                handle_error('E040', 'Editor: save', exc)
            except Exception:
                pass
            fail(f'저장 실패: {exc}')

    run(work)
