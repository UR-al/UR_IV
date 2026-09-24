"""관리형 백엔드 release 정리 — 수 GB 트리를 호출 스레드에서 지우지 않는다.

``BackendRuntimeManager`` 는 앱 시작 때(스플래시보다 먼저, GUI 스레드) 생성된다. 예전에는
그 생성자가 비활성 release 트리(venv 포함 수 GB)를 동기 ``rmtree`` 해서, 관리형 업데이트
다음 첫 실행에 창 없이 한참 멈췄다.

지금은 두 단계로 나눈다.

1. **이름 바꾸기(즉시)**: 비활성 release 를 같은 ``releases/`` 아래 ``.trash-<id>-<rand>`` 로
   rename 한다. 같은 볼륨 안의 rename 이라 크기와 무관하게 즉시 끝나고, 이후 누구도 그
   경로를 release 로 보지 않는다.
2. **삭제(백그라운드)**: 데몬 스레드가 ``.trash-*`` 만 지운다. 설치 중인 후보 release
   (``_prepare_release`` 가 만드는 ``releases/<새 id>``)는 이 접두사가 없으므로 절대 건드리지
   않는다 — 그래서 작업 게이트(_operation_gate)를 잡을 필요가 없고, 시작 자동기동이
   OPERATION_BUSY 로 거절되지도 않는다.

앱이 삭제 도중 종료돼도 남은 ``.trash-*`` 는 다음 실행에 다시 지운다. Windows 에서 파일이
열려 있어 rename 이 실패한 release 는 그대로 두고 다음 실행에 다시 시도한다(예전의
ignore_errors rmtree 처럼 반쯤 지운 release 를 남기지 않는다).

**링크는 따라가지 않는다**: ``releases/`` 안에 ``.trash-*`` 이름의 심볼릭 링크·정션이 있으면
(앱은 만들지 않지만 사람·외부 도구가 만들 수 있다) 그 대상 — 예: 활성 release — 을 지우지
않는다. 지울 대상은 "resolve 해도 이름이 그대로인 ``.trash-*`` 실제 폴더"뿐이다. release id 는
영숫자로 시작하므로(core.backend_runtime) 활성·후보 release 는 구조적으로 대상이 될 수 없다.
"""
from __future__ import annotations

import logging
import os
import shutil
import stat
import threading
import uuid
from pathlib import Path
from typing import Iterable, Optional

TRASH_PREFIX = ".trash-"
# IsReparseTagNameSurrogate — 다른 경로를 가리키는 reparse point(심볼릭 링크·정션).
# OneDrive 클라우드 파일 같은 다른 reparse point 는 링크가 아니므로 여기 해당하지 않는다.
_NAME_SURROGATE_BIT = 0x20000000

logger = logging.getLogger(__name__)


def is_trash(path: Path) -> bool:
    return Path(path).name.startswith(TRASH_PREFIX)


def _is_link(path: Path) -> bool:
    """심볼릭 링크·정션이면 True. lstat 이 실패하면 확인할 수 없으니 True(건드리지 않는다)."""
    try:
        info = os.lstat(path)
    except OSError:
        return True
    if stat.S_ISLNK(info.st_mode):
        return True
    # Windows: lstat 은 이름 대리 reparse point(정션 포함)만 열고 그 태그를 st_reparse_tag 로 준다.
    return bool((getattr(info, "st_reparse_tag", 0) or 0) & _NAME_SURROGATE_BIT)


def _direct_child(path: Path, root: Path) -> Optional[Path]:
    """``path`` 가 ``root`` 바로 아래의 실제 폴더면 resolve 된 경로, 아니면 None."""
    try:
        root_resolved = Path(root).resolve()
        resolved = Path(path).resolve()
    except OSError:
        return None
    if resolved.parent != root_resolved or not resolved.is_dir():
        return None
    return resolved


def move_to_trash(path: Path, releases_root: Path) -> Optional[Path]:
    """release 폴더를 ``releases_root/.trash-*`` 로 이름만 바꾼다. 실패하면 None.

    링크 자체를 넘기면 거절한다 — 링크를 따라가 대상 release 를 옮기지 않는다
    (현재 호출자는 resolve 된 실제 경로만 넘긴다).
    """
    if _is_link(Path(path)):
        return None
    resolved = _direct_child(path, releases_root)
    if resolved is None or is_trash(resolved):
        return None
    target = resolved.parent / f"{TRASH_PREFIX}{resolved.name}-{uuid.uuid4().hex[:8]}"
    try:
        resolved.rename(target)
    except OSError:
        # 실행 중인 프로세스가 파일을 쥐고 있으면(Windows) 다음 실행에 다시 시도한다.
        logger.info("release 정리 보류(사용 중): %s", resolved)
        return None
    return target


def _is_trash_dir(path: Path) -> bool:
    """rmtree 직전 재확인 — ``.trash-*`` 이름의 실제 폴더(링크 아님)."""
    return is_trash(path) and not _is_link(path)


def trash_entries(releases_root: Path) -> list[Path]:
    """``releases_root`` 바로 아래 ``.trash-*`` 실제 폴더(resolve 된 경로).

    ``.trash-*`` 이름의 링크·정션은 건너뛴다 — resolve 하면 활성 release 같은 형제 폴더가
    나와 rmtree 가 그것을 지우게 된다. resolve 결과의 이름도 원래 이름과 같아야 한다.
    """
    root = Path(releases_root)
    try:
        children = tuple(root.iterdir())
    except OSError:
        return []
    entries = []
    for child in children:
        if not is_trash(child):
            continue
        if _is_link(child):
            logger.debug("release 정리: 링크는 따라가지 않음 %s", child)
            continue
        resolved = _direct_child(child, root)
        if resolved is None or not is_trash(resolved):
            continue
        if os.path.normcase(resolved.name) != os.path.normcase(child.name):
            continue
        entries.append(resolved)
    return entries


def purge_trash(releases_roots: Iterable[Path]) -> int:
    """``.trash-*`` 폴더만 지운다. 반환: 완전히 지운 개수."""
    removed = 0
    for root in releases_roots:
        for entry in trash_entries(Path(root)):
            if not _is_trash_dir(entry):   # 목록을 만든 뒤 바뀌었으면 건드리지 않는다
                continue
            shutil.rmtree(entry, ignore_errors=True)
            if not entry.exists():
                removed += 1
    return removed


def _purge_logged(roots: list[Path]) -> None:
    try:
        removed = purge_trash(roots)
        if removed:
            logger.info("이전 release %d개를 백그라운드에서 정리했습니다", removed)
    except Exception:
        logger.warning("이전 release 백그라운드 정리 실패", exc_info=True)


def purge_trash_async(releases_roots: Iterable[Path]) -> Optional[threading.Thread]:
    """지울 ``.trash-*`` 가 있을 때만 데몬 스레드를 띄운다."""
    roots = [Path(root) for root in releases_roots if trash_entries(Path(root))]
    if not roots:
        return None
    thread = threading.Thread(
        target=_purge_logged, args=(roots,), name="runtime-release-trash", daemon=True
    )
    thread.start()
    return thread


__all__ = [
    "TRASH_PREFIX",
    "is_trash",
    "move_to_trash",
    "purge_trash",
    "purge_trash_async",
    "trash_entries",
]
