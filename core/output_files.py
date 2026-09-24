"""결과 파일을 '새 파일로만' 쓰는 공용 헬퍼 (Qt 비의존).

배치·업스케일·인페인트 결과는 절대 기존 파일(원본 포함)을 덮어쓰면 안 된다. 예전 Vue
배치는 ``generated_images/<원본이름>`` 에 그대로 써서 원본과 생성 메타데이터를 날렸다.

``unique_path`` 로 비어 있는 이름을 고른 뒤 ``open(..., 'xb')``(O_EXCL)로 만든다 — 이름을
고른 뒤 쓰기 전에 다른 쓰기가 같은 이름을 차지해도 덮어쓰지 않고 다음 이름으로 간다.
"""
from __future__ import annotations

import os

from core.editor_save import unique_path


def same_file_path(a: str, b: str) -> bool:
    """대소문자·구분자·상대 경로 차이를 무시하고 같은 경로인지 (파일이 없어도 판정)."""
    def norm(path: str) -> str:
        return os.path.normcase(os.path.abspath(os.path.realpath(str(path))))
    try:
        return norm(a) == norm(b)
    except (OSError, ValueError):
        return False


def write_new_file(path: str, data: bytes, *, attempts: int = 1000) -> str:
    """``path`` 가 비어 있으면 거기, 아니면 ``<stem>_2``, ``_3`` … 에 새로 쓴다. 쓴 경로를 돌려준다.

    쓰기 도중 실패하면 이번에 만든 미완성 파일만 지운다.
    """
    folder = os.path.dirname(os.path.abspath(path))
    if folder:
        os.makedirs(folder, exist_ok=True)
    candidate = unique_path(path)
    for _ in range(max(1, int(attempts))):
        try:
            handle = open(candidate, 'xb')
        except FileExistsError:
            candidate = unique_path(candidate)
            continue
        try:
            with handle:
                handle.write(data)
        except Exception:
            try:
                os.remove(candidate)
            except OSError:
                pass
            raise
        return candidate
    raise OSError(f'비어 있는 파일 이름을 찾지 못했습니다: {path}')
