# core/image_delete.py
"""갤러리·히스토리 '휴지통으로 이동' 요청 한 건의 결과 (Qt 를 모르는 순수 경계).

예전에는 프론트가 결과를 기다리지 않고 목록에서 먼저 지웠다. 그래서 휴지통 이동이
실패하면(send2trash 없음·권한·잠김) 파일은 그대로인데 갤러리·폴더 캐시·히스토리에서만
사라졌고, 폴더를 다시 훑기 전까지 돌아오지 않았다. 여기서 만든 dict 를 generator_main 의
delete_image 분기가 ``imageDeleteResult`` 로 그대로 돌려주고, 프론트는 ``removed`` 가
참일 때만 목록에서 뺀다(frontend/src/utils/imageDeleteResult.ts).
"""
from __future__ import annotations

from core.image_utils import TRASH_ALREADY_GONE_MESSAGE, move_to_trash_result
from core.path_safety import missing_input_path, safe_input_path

REFUSED_PATH_MESSAGE = '허용되지 않은 이미지 경로입니다'


def image_delete_result(
    request_path: str,
    local_path: str,
    *,
    allowed_exts: frozenset[str] | None = None,
) -> dict:
    """삭제 요청 한 건을 처리하고 프론트로 보낼 결과를 만든다.

    request_path — 프론트가 보낸 경로 원문. 프론트가 자기 목록에서 같은 항목을 찾도록
                   그대로 돌려준다(정규화한 값을 돌려주면 '/' vs '\\' 로 어긋난다).
    local_path   — file:// 제거·구분자 정리를 마친 로컬 경로.
    allowed_exts — 지울 수 있는 확장자. None 이면 path_safety 기본(정지 이미지).

    반환: ``{path, ok, removed, level, message}`` — 뜻은 move_to_trash_result 와 같다.
    """
    exts_kwargs = {} if allowed_exts is None else {'allowed_exts': allowed_exts}
    clean = safe_input_path(local_path, **exts_kwargs)
    if clean:
        result = move_to_trash_result(clean)
    elif missing_input_path(local_path, **exts_kwargs):
        # 허용된 경로인데 파일이 이미 없다(다른 프로그램이 지움·드라이브 분리) —
        # 목록에 남겨 두면 다시 지울 방법이 없으니 빼도 된다고 알린다.
        result = {'ok': False, 'removed': True, 'level': 'warning',
                  'message': TRASH_ALREADY_GONE_MESSAGE}
    else:
        result = {'ok': False, 'removed': False, 'level': 'error',
                  'message': REFUSED_PATH_MESSAGE}
    return {'path': str(request_path), **result}
