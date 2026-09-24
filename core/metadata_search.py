# core/metadata_search.py
"""Gallery·Favorites 'EXIF 검색'용 텍스트 — 파일당 소문자 한 덩어리. Qt 비의존.

예전 검색은 이미지마다 동기 슬롯 getImageExif 를 불러 전체 메타 dict(raw_workflow,
prompt_candidates, 표시 그룹 …)를 받았다. 검색에는 prompt·negative·raw 만 쓰므로 여기서
그 텍스트만 만들고, (경로, 수정 시각, 크기) 로 캐시해 'EXIF 저장' 뒤에도 옛 텍스트로
매칭되지 않게 한다. 브리지는 이 모듈을 백그라운드 스레드에서 부른다.
"""
from __future__ import annotations

import os
import threading
from collections import OrderedDict
from pathlib import Path
from typing import Iterable, Optional, Union

# 한 번 요청에 받는 경로 수 상한 — 프런트는 50개씩 보낸다.
MAX_PATHS_PER_REQUEST = 200
# 캐시 항목 수 상한 (텍스트는 보통 2~7KB → 최대 수십 MB)
DEFAULT_CACHE_ENTRIES = 4000


def metadata_search_text(path: Union[str, Path]) -> str:
    """검색용 텍스트: prompt + negative + 원문(A1111 parameters 또는 Comfy prompt/workflow JSON).

    getImageExif 의 ``raw`` 와 같은 원문 우선순위를 쓰되 workflow JSON 은 다른 원문이 없을 때만
    포함한다(같은 그래프가 두 번 들어가지 않게). 읽기 실패는 빈 문자열.
    """
    try:
        from PIL import Image
        from core.image_metadata import extract_from_pil
        with Image.open(path) as image:
            meta = extract_from_pil(image)
    except Exception:
        return ""
    raw = meta.raw_parameters or meta.raw_prompt or meta.raw_workflow
    return f"{meta.prompt} {meta.negative_prompt} {raw}".lower()


class SearchTextCache:
    """(정규화 경로) → (mtime_ns, size, text) LRU. 스레드 안전."""

    def __init__(self, max_entries: int = DEFAULT_CACHE_ENTRIES):
        self._max = max(1, int(max_entries))
        self._items: "OrderedDict[str, tuple[int, int, str]]" = OrderedDict()
        self._lock = threading.Lock()

    @staticmethod
    def _key(path: str) -> str:
        return os.path.normcase(os.path.abspath(path))

    def get(self, path: str, reader=metadata_search_text) -> str:
        try:
            stat = os.stat(path)
        except OSError:
            return ""
        key = self._key(path)
        stamp = (stat.st_mtime_ns, stat.st_size)
        with self._lock:
            hit = self._items.get(key)
            if hit is not None and hit[:2] == stamp:
                self._items.move_to_end(key)
                return hit[2]
        text = reader(path)
        with self._lock:
            self._items[key] = (stamp[0], stamp[1], text)
            self._items.move_to_end(key)
            while len(self._items) > self._max:
                self._items.popitem(last=False)
        return text

    def __len__(self) -> int:
        with self._lock:
            return len(self._items)


def search_texts_for(paths: Iterable[str], *, cache: Optional[SearchTextCache] = None,
                     is_allowed=lambda _p: True) -> dict[str, str]:
    """경로 목록 → {원래 경로: 검색 텍스트}. 허용되지 않은 경로는 '' (목록에서 빠지지 않게)."""
    out: dict[str, str] = {}
    for index, raw in enumerate(paths):
        if index >= MAX_PATHS_PER_REQUEST:
            break
        path = str(raw or "")
        if not path:
            continue
        resolved = is_allowed(path)
        if not resolved:
            out[path] = ""
            continue
        target = resolved if isinstance(resolved, str) else path
        out[path] = cache.get(target) if cache is not None else metadata_search_text(target)
    return out
