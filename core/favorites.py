# core/favorites.py
"""즐겨찾기 목록(user_data/favorites.json)의 단일 reader/writer.

예전 로더는 읽을 때 ``os.path.exists`` 로 목록을 거르고, 추가·삭제가 그 결과를
곧바로 저장했다. 그래서 외장/네트워크 드라이브가 빠진 순간 즐겨찾기를 하나
추가하면 그 드라이브의 즐겨찾기가 파일에서 영구히 사라졌고, 이미 지워진 이미지는
걸러진 목록에 없어 개별 삭제도 되지 않았다(다시 불러오면 되살아남).
JSON 이 깨지면 [] 로 읽고 다음 추가가 파일을 한 줄로 덮어썼다.

여기서는
  - 파일에 있는 항목을 **존재 여부와 무관하게** 그대로 보존하고
    (화면에서 없는 파일을 가릴지는 표시하는 쪽이 정한다),
  - 손상 파일은 ``.corrupt`` 로 옮겨 보존한 뒤 빈 목록으로 시작하며,
  - **읽지 못한 파일은 빈 목록으로 치지 않는다** — 백신·OneDrive·백업 도구가 잠깐
    잡고 있어 open 이 PermissionError(공유 위반)를 내면 ``FavoritesUnavailableError``
    를 올리고, 추가·삭제·토글은 저장하지 않고 멈춘다(예전엔 [] 로 읽혀 다음 추가가
    파일을 한 줄로 덮어썼다).
  - 경로 비교는 normcase(normpath) 로 해 ``/``·``\\``·대소문자 차이를 같은 항목으로 본다.
Qt 를 모르는 순수 모듈이다.
"""
from __future__ import annotations

import json
import os
import threading
from typing import Iterable

from utils.atomic_json import atomic_write_json

_LOCK = threading.RLock()


class FavoritesUnavailableError(RuntimeError):
    """즐겨찾기 파일이 있는데 읽지(또는 손상본을 격리하지) 못했다.

    이때 목록을 [] 로 치고 저장하면 파일이 통째로 덮어써진다 — 변경을 포기해야 한다.
    """


def _default_file() -> str:
    from config import FAVORITES_FILE
    return str(FAVORITES_FILE)


def _target_file(path) -> str:
    """str/PathLike 모두 받는다 — '.corrupt'·'.tmp' 접미사는 문자열 연결로 붙인다."""
    return os.fspath(path) if path else _default_file()


def favorite_key(path: str) -> str:
    """같은 파일을 가리키는 표기 차이(구분자·대소문자·..)를 하나로 모은 비교 키."""
    return os.path.normcase(os.path.normpath(str(path)))


def _clean_entries(raw: Iterable) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for item in raw:
        if not isinstance(item, str) or not item.strip():
            continue
        key = favorite_key(item)
        if key in seen:
            continue
        seen.add(key)
        out.append(item)
    return out


def _corrupt_backup_path(target: str) -> str:
    """``.corrupt`` 가 이미 있으면 지우지 않고 ``.corrupt.1``, ``.2`` … 로 비껴 간다."""
    candidate = target + ".corrupt"
    index = 1
    while os.path.lexists(candidate):
        candidate = f"{target}.corrupt.{index}"
        index += 1
    return candidate


def _quarantine(target: str) -> None:
    """형식이 틀린 파일을 .corrupt 로 옮겨 다음 저장이 덮어써도 원본이 남게 한다.

    옮기지 못하면(파일이 잠겨 있음 등) FavoritesUnavailableError — 여기서 [] 를 돌려주면
    다음 저장이 손상본을 백업 없이 덮어쓴다.
    """
    try:
        os.replace(target, _corrupt_backup_path(target))
    except FileNotFoundError:
        return
    except OSError as exc:
        raise FavoritesUnavailableError(
            "즐겨찾기 파일이 손상됐는데 백업(.corrupt)으로 옮기지 못했습니다"
        ) from exc


def _read_favorites_file(target: str) -> list:
    """파일 원본 목록. 없음 → [], 손상 → 격리 후 [], 읽기 실패 → FavoritesUnavailableError.

    utils.atomic_json.load_json_safe 는 OSError 도 default 로 삼키므로 쓰지 않는다 —
    '파일 없음' 과 '있는데 못 읽음' 을 구분해야 저장 전 읽기가 파일을 지우지 않는다.
    """
    try:
        # utf-8-sig: 메모장으로 고친 BOM 붙은 파일도 손상으로 오인해 격리하지 않는다(쓰기는 BOM 없음).
        with open(target, "r", encoding="utf-8-sig") as fh:
            data = json.load(fh)
    except FileNotFoundError:
        return []
    except (json.JSONDecodeError, UnicodeDecodeError):
        _quarantine(target)
        return []
    except OSError as exc:
        raise FavoritesUnavailableError(
            # 레거시 호출부가 print 한다 — cp949 콘솔에서 깨지는 문자(—)는 쓰지 않는다.
            f"즐겨찾기 파일을 읽지 못했습니다({exc.__class__.__name__}). 잠시 후 다시 시도하세요"
        ) from exc
    if not isinstance(data, list):
        _quarantine(target)
        return []
    return data


def load_favorites(path: str | None = None) -> list[str]:
    """즐겨찾기 목록. 없는 파일(분리된 드라이브 등)도 **거르지 않는다**.

    파일이 있는데 읽지 못하면 FavoritesUnavailableError — 화면 표시용 호출부는 잡아서
    빈 목록을 보여 주면 되지만, 저장 전 읽기는 반드시 멈춰야 한다.
    """
    target = _target_file(path)
    with _LOCK:
        data = _read_favorites_file(target)
    return _clean_entries(data)


def save_favorites(entries: Iterable[str], path: str | None = None) -> list[str]:
    target = _target_file(path)
    cleaned = _clean_entries(entries)
    with _LOCK:
        atomic_write_json(target, cleaned)
    return cleaned


def is_favorite(image_path: str, path: str | None = None) -> bool:
    key = favorite_key(image_path)
    return any(favorite_key(item) == key for item in load_favorites(path))


def add_favorite(image_path: str, path: str | None = None) -> tuple[bool, list[str]]:
    """추가. (새로 추가됐는지, 저장된 목록). 이미 있으면 파일을 건드리지 않는다."""
    if not isinstance(image_path, str) or not image_path.strip():
        raise ValueError("즐겨찾기에 추가할 경로가 비어 있습니다")
    with _LOCK:
        entries = load_favorites(path)
        key = favorite_key(image_path)
        if any(favorite_key(item) == key for item in entries):
            return False, entries
        entries.append(image_path)
        return True, save_favorites(entries, path)


def remove_favorite(image_path: str, path: str | None = None) -> tuple[bool, list[str]]:
    """삭제. 파일이 이미 지워졌거나 드라이브가 빠져 있어도 항목은 지운다."""
    with _LOCK:
        entries = load_favorites(path)
        key = favorite_key(image_path)
        kept = [item for item in entries if favorite_key(item) != key]
        if len(kept) == len(entries):
            return False, entries
        return True, save_favorites(kept, path)


def toggle_favorite(image_path: str, path: str | None = None) -> tuple[bool, list[str]]:
    """토글. (토글 후 즐겨찾기 상태, 저장된 목록)."""
    with _LOCK:
        removed, entries = remove_favorite(image_path, path)
        if removed:
            return False, entries
        _added, entries = add_favorite(image_path, path)
        return True, entries
