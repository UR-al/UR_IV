# core/image_utils.py
import os
import re

def normalize_windows_path(path):
    """Windows 경로 정규화"""
    if path.startswith('\\\\?\\') or path.startswith('//?/'):
        path = re.sub(r'^\\\\\?\\', '', path)
        path = re.sub(r'^//\?/', '', path)
    return os.path.normpath(path)

class TrashUnavailableError(RuntimeError):
    """send2trash 를 쓸 수 없어 휴지통 이동을 하지 않았다(영구 삭제로 대신하지 않는다)."""


def move_to_trash(path) -> bool:
    """파일을 휴지통으로 이동한다.

    반환: True = 휴지통으로 옮김, False = 파일이 이미 없음(아무것도 안 함).
    send2trash 를 불러올 수 없으면 TrashUnavailableError, 이동 자체가 실패하면
    send2trash 의 예외(OSError 계열)를 그대로 올린다.

    예전에는 send2trash import 가 실패하면 os.remove 로 **영구 삭제**했고, 호출부는
    결과와 상관없이 '휴지통으로 이동됨' 을 알렸다. 되돌릴 수 없는 삭제를 조용히
    하느니 실패를 알리는 쪽이 맞다 — 여기서도, 호출부에서도 os.remove 폴백은 두지 않는다.
    """
    path = normalize_windows_path(str(path))
    if not os.path.exists(path):
        return False
    try:
        from send2trash import send2trash
    except ImportError as exc:
        raise TrashUnavailableError(
            "휴지통 모듈(send2trash)을 불러오지 못해 파일을 지우지 않았습니다"
        ) from exc
    send2trash(path)
    return True


TRASH_ALREADY_GONE_MESSAGE = '파일이 이미 없어 휴지통으로 옮길 것이 없습니다'


def move_to_trash_result(path) -> dict:
    """move_to_trash 를 실행하고 결과를 dict 로 돌려준다.

    ``ok``      — 이번 호출로 휴지통에 옮겼다.
    ``removed`` — 그 경로에 더는 파일이 없다(옮겼거나 원래 없었다). 화면 목록에서
                  빼도 되는지는 이 값으로 정한다 — 실패했는데 목록에서 지우면
                  파일은 남았는데 갤러리·히스토리에서만 사라진다.
    ``level``/``message`` — 알림 레벨과 사용자 메시지('휴지통으로 이동됨' 은 실제로 옮겼을 때만).
    """
    try:
        moved = move_to_trash(path)
    except TrashUnavailableError as exc:
        return {'ok': False, 'removed': False, 'level': 'error', 'message': str(exc)}
    except Exception as exc:
        from core.error_handler import sanitize_for_ui
        return {
            'ok': False, 'removed': False, 'level': 'error',
            'message': f'휴지통으로 옮기지 못했습니다: {sanitize_for_ui(str(exc))}',
        }
    if moved:
        return {'ok': True, 'removed': True, 'level': 'info', 'message': '휴지통으로 이동됨'}
    return {'ok': False, 'removed': True, 'level': 'warning', 'message': TRASH_ALREADY_GONE_MESSAGE}


def move_to_trash_feedback(path) -> tuple[str, str]:
    """move_to_trash 를 실행하고 (알림 레벨, 사용자 메시지)를 돌려준다.

    '휴지통으로 이동됨' 은 실제로 옮겼을 때만 — 실패·파일 없음은 각각 알린다.
    """
    result = move_to_trash_result(path)
    return result['level'], result['message']