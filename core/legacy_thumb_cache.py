# core/legacy_thumb_cache.py
"""옛 썸네일 캐시 폴더(image_cache/thumbs) 은퇴 — Qt 비의존.

옛 폴더에는 두 종류가 섞여 있다.
  - 레거시 PyQt 갤러리(ThumbnailItem · workers/gallery_worker · GeneratorBase._create_thumbnail)가
    만든 ``sha1(resolve 한 경로)`` 150px JPEG — 그 코드가 모두 은퇴해 아무도 읽지 않는다
    (실측 32,302개 / 128.6MB, 캐시의 약 43%).
  - Vue 히스토리·갤러리(aithumb:)가 쓰는 ``sha1(normpath@폭)`` 썸네일(core.thumb_cache).
예전 평면→샤드 이관(migrate_flat_to_sharded)이 두 종류를 이름만 보고 같은 샤드 폴더로 옮겨,
파일 이름으로는 구분할 수 없다. 그래서 지금 캐시는 새 폴더(``config.THUMB_DIR`` =
image_cache/thumbs_v2)를 쓰고, 옛 폴더는 앱 시작 뒤 백그라운드에서 한 번 정리한다.

  - core.thumb_cache 가 쓴 썸네일 — JPEG 주석에 원본 서명(``aistudio-src:``)이 있다 — 은 새 폴더의
    같은 샤드 자리로 옮겨 다시 만들지 않게 한다. 서명이 지금 원본과 다르면 읽을 때
    thumb_is_stale 가 다시 만든다(옮겨도 옛 그림이 새것처럼 남지 않는다).
  - 나머지(서명 없는 레거시 썸네일, 서명 이전 형식 — 읽혀도 어차피 다시 만든다, 렌더 임시 파일)는
    지운다. 새 폴더에 같은 키가 이미 있으면 새 것을 두고 옛 것을 지운다.
  - 빈 폴더까지 지워 끝나면 옛 폴더 자체가 없어진다 — 다음 실행부터는 할 일이 없다.
    중간에 끊겨도(앱 종료) 남은 것만 다음 실행에서 이어서 한다.

배경 정리는 시작 30초 뒤에 도는데, 히스토리는 시작하자마자 목록 전체의 썸네일을 요청한다.
그 사이 새 폴더에서 빗나간 키를 원본에서 다시 만들면 옮기는 의미가 없고(첫 실행 재렌더 폭주 —
정리가 나중에 '새 폴더에 이미 있음'으로 옛 것을 지운다), 원본을 지운 히스토리 항목은 만들 원본이
없어 썸네일을 잃는다. 그래서 조회가 새 폴더에서 빗나가면 그 키 하나를 옛 자리에서 먼저 옮겨
온다(:func:`adopt_legacy_thumb` — core.thumb_cache.get_or_make_thumb·core.thumb_prefetch 의
``legacy_dir``). 배경 정리는 남은 것을 한꺼번에 치우는 일만 한다.

앱 자체 캐시라 코드로 지운다(사용자 데이터 아님). 옛 폴더와 새 폴더가 겹치면(같거나 한쪽이
다른 쪽 안) 아무것도 하지 않는다 — 새 캐시를 지우지 않게.
"""
from __future__ import annotations

import logging
import os
import re
import threading
import time
from dataclasses import dataclass
from typing import Callable, Optional

from core.cache_cleanup import ensure_shard_dir, read_thumb_signature, shard_path, source_signature

logger = logging.getLogger(__name__)

# 앱 시작 직후의 디스크·CPU 경쟁(Vue 로드·히스토리 썸네일)을 피해 이만큼 뒤에 시작한다.
DEFAULT_START_DELAY_SECONDS = 30.0

_THUMB_NAME = re.compile(r"^[0-9a-f]{40}\.jpg$", re.IGNORECASE)


@dataclass(frozen=True)
class LegacyThumbRetirement:
    moved: int = 0     # 새 폴더로 옮긴 현재 형식 썸네일
    removed: int = 0   # 지운 레거시·옛 형식·임시 파일
    failed: int = 0    # 옮기거나 지우지 못한 파일(잠김 등) — 다음 실행에서 다시 한다
    done: bool = False  # 옛 폴더가 사라졌다(또는 처음부터 없었다)


def _norm(path: str) -> str:
    return os.path.normcase(os.path.abspath(path))


def folders_overlap(first: str, second: str) -> bool:
    """두 폴더가 같거나 한쪽이 다른 쪽 안에 있는지."""
    a, b = _norm(first), _norm(second)
    try:
        common = os.path.commonpath([a, b])
    except ValueError:   # 다른 드라이브
        return False
    return common in (a, b)


def is_current_format_thumb(path: str) -> bool:
    """core.thumb_cache 가 쓴 썸네일(40자리 sha1 이름 + 원본 서명 주석)인지."""
    return bool(_THUMB_NAME.match(os.path.basename(path))) and read_thumb_signature(path) is not None


def _remove_file(path: str) -> bool:
    try:
        os.remove(path)
        return True
    except FileNotFoundError:
        return True
    except OSError:
        return False


_FILE_ATTRIBUTE_REPARSE_POINT = 0x400


def _is_link_dir(path: str) -> bool:
    """심볼릭 링크·정션(재분석 지점) 폴더 — 따라 들어가 대상 폴더를 지우면 안 된다. 모르면 링크로 친다."""
    try:
        if os.path.islink(path):
            return True
        isjunction = getattr(os.path, "isjunction", None)   # 3.12+
        if isjunction is not None and isjunction(path):
            return True
        attrs = getattr(os.lstat(path), "st_file_attributes", 0)
        return bool(attrs & _FILE_ATTRIBUTE_REPARSE_POINT)
    except OSError:
        return True


def _walk_bottom_up(top: str):
    """(폴더, [파일 경로]) 를 아래에서 위로. 링크·정션 폴더는 들어가지도 지우지도 않는다."""
    try:
        with os.scandir(top) as scan:
            entries = list(scan)
    except OSError:
        return
    files: list[str] = []
    for entry in entries:
        try:
            is_dir = entry.is_dir(follow_symlinks=False)
        except OSError:
            continue
        if is_dir:
            if not _is_link_dir(entry.path):
                yield from _walk_bottom_up(entry.path)
        else:
            files.append(entry.path)   # 파일 링크는 링크만 지워진다(대상은 그대로)
    yield top, files


def retire_legacy_thumb_dir(legacy_dir: str, new_dir: str) -> LegacyThumbRetirement:
    """옛 썸네일 폴더를 정리한다 — 현재 형식은 ``new_dir`` 로 옮기고 나머지는 지운 뒤 폴더를 없앤다."""
    if not legacy_dir or not os.path.isdir(legacy_dir):
        return LegacyThumbRetirement(done=True)
    if not new_dir or folders_overlap(legacy_dir, new_dir):
        logger.warning("옛 썸네일 폴더 정리 건너뜀 — 새 캐시 폴더와 겹친다: %s / %s", legacy_dir, new_dir)
        return LegacyThumbRetirement(done=False)
    if _is_link_dir(legacy_dir):
        logger.warning("옛 썸네일 폴더 정리 건너뜀 — 링크(정션) 폴더다: %s", legacy_dir)
        return LegacyThumbRetirement(done=False)

    moved = removed = failed = 0
    # 아래에서 위로 — 파일을 비운 폴더를 바로 지울 수 있다. 링크(정션)는 따라가지 않는다.
    for root, files in _walk_bottom_up(legacy_dir):
        for path in files:
            if is_current_format_thumb(path):
                stem = os.path.splitext(os.path.basename(path))[0].lower()
                target = shard_path(new_dir, stem, ".jpg")
                if os.path.exists(target):
                    # 새 폴더가 이미 같은 키를 만들었다 — 새 것이 맞다
                    if _remove_file(path):
                        removed += 1
                    else:
                        failed += 1
                    continue
                ensure_shard_dir(target)
                try:
                    os.replace(path, target)
                    moved += 1
                except OSError:
                    if _remove_file(path):
                        removed += 1
                    else:
                        failed += 1
                continue
            if _remove_file(path):
                removed += 1
            else:
                failed += 1
        # 비었으면 폴더도 지운다(마지막 차례가 옛 폴더 자신). 남은 것(잠긴 파일·링크)이 있으면 그대로.
        try:
            os.rmdir(root)
        except OSError:
            pass
    done = not os.path.exists(legacy_dir)
    return LegacyThumbRetirement(moved=moved, removed=removed, failed=failed, done=done)


def legacy_thumb_candidates(legacy_dir: str, key: str) -> tuple[str, str]:
    """새 캐시 키 ``key`` 가 옛 폴더에 있을 수 있는 자리 — 샤드(대부분), 평면(1f8198f13 이전 저장분)."""
    return shard_path(legacy_dir, key, ".jpg"), os.path.join(legacy_dir, f"{key}.jpg")


def _reaches_through_link(legacy_dir: str, candidate: str) -> bool:
    """후보 파일이나 그 폴더(샤드·옛 폴더)가 링크·정션인지 — 옛 폴더 밖의 파일을 캐시로 끌어오지 않게."""
    if os.path.islink(candidate):
        return True
    folders = {os.path.dirname(candidate), legacy_dir}
    return any(_is_link_dir(folder) for folder in folders)


def adopt_legacy_thumb(legacy_dir: Optional[str], dest: str, source: Optional[str] = None) -> bool:
    """새 캐시에 없는 썸네일 ``dest`` 를 옛 폴더의 같은 키에서 옮겨 온다. 옮겼으면 True.

    조회가 새 폴더에서 빗나갈 때 렌더 **전에** 부른다(배경 정리를 기다리지 않는다 — 모듈 설명).
    빗나감마다 옛 자리 두 곳을 여는 비용뿐이고, 옛 폴더가 사라진 뒤에는 바로 실패한다.

    옮기는 것은 core.thumb_cache 가 쓴 서명 있는 썸네일 가운데 그대로 쓸 수 있는 것뿐이다.
      - 원본 서명이 지금 원본과 같다 → 다시 만들 필요가 없다.
      - 원본을 읽을 수 없다(지운 히스토리 파일 등) → 만들 수 없으니 이 썸네일이 유일한 그림이다.
    서명이 다르면(그 뒤 원본이 바뀜) 옮기지 않는다 — 어차피 다시 만들고, 옛 것은 배경 정리가 치운다.
    그래서 다른 스레드가 방금 같은 자리에 만든 썸네일을 덮어써도 같은 원본 기준이라 해가 없다.
    서명 없는 레거시 PyQt 썸네일·옛 형식은 건드리지 않는다. 링크·정션을 거치는 후보는 옮기지 않는다.
    이미 ``dest`` 가 있거나, 옮기기에 실패하면(배경 정리·다른 스레드가 먼저 옮김, 잠김) False —
    호출자는 평소대로 신선도 판정·렌더를 이어 간다.
    """
    if not legacy_dir or not dest or os.path.exists(dest):
        return False
    name = os.path.basename(dest)
    if not _THUMB_NAME.match(name):
        return False
    key = os.path.splitext(name)[0].lower()
    checked = False
    current = None
    for candidate in legacy_thumb_candidates(legacy_dir, key):
        signature = read_thumb_signature(candidate)   # 없는 파일·서명 없는 JPEG → None
        if signature is None:
            continue
        if not checked:   # 원본 stat 은 서명 있는 후보를 찾았을 때만(옛 폴더가 사라진 뒤엔 0회)
            current = source_signature(source) if source else None
            checked = True
        if current is not None and signature != current:
            continue
        cache_dir = os.path.dirname(os.path.dirname(dest))
        if folders_overlap(legacy_dir, cache_dir) or _reaches_through_link(legacy_dir, candidate):
            return False
        ensure_shard_dir(dest)
        try:
            os.replace(candidate, dest)
        except OSError:
            return False
        return True
    return False


_start_lock = threading.Lock()
_started: set[str] = set()


def start_legacy_thumb_retirement(
    legacy_dir: str,
    new_dir: str,
    *,
    delay_seconds: float = DEFAULT_START_DELAY_SECONDS,
    on_done: Optional[Callable[[LegacyThumbRetirement], None]] = None,
) -> Optional[threading.Thread]:
    """옛 폴더가 있으면 데몬 스레드에서 ``delay_seconds`` 뒤 한 번 정리한다. 시작한 스레드(없으면 None).

    같은 프로세스에서 같은 폴더로는 한 번만 띄운다. 데몬이라 앱 종료를 붙잡지 않는다 —
    끊기면 남은 것은 다음 실행에서 이어서 한다.
    """
    if not legacy_dir or not os.path.isdir(legacy_dir):
        return None
    key = _norm(legacy_dir)
    with _start_lock:
        if key in _started:
            return None
        _started.add(key)

    def _run() -> None:
        if delay_seconds > 0:
            time.sleep(delay_seconds)
        try:
            result = retire_legacy_thumb_dir(legacy_dir, new_dir)
        except Exception as exc:   # 캐시 정리 실패가 앱을 흔들지 않게
            logger.warning("옛 썸네일 폴더 정리 실패 (무시): %s", exc)
            return
        logger.info(
            "옛 썸네일 폴더 정리: %s — 옮김 %d · 지움 %d · 실패 %d · 완료 %s",
            legacy_dir, result.moved, result.removed, result.failed, result.done,
        )
        if on_done is not None:
            try:
                on_done(result)
            except Exception:
                pass

    thread = threading.Thread(target=_run, name="legacy-thumb-retire", daemon=True)
    thread.start()
    return thread
