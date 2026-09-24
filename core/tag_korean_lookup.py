# core/tag_korean_lookup.py
"""태그 자동완성용 한국어 라벨·검색.

`tags_db/catalogs/korean_tag_catalog.parquet` 에는 태그마다 한국어 분류 경로(category),
설명(desc), 검색 키워드(keywords) 가 이미 들어 있는데 지금까지는 `tag` 열만 썼다.
여기서 두 가지를 제공한다.

- `label(tag)`   : 영문 태그 → {category, desc, keywords} (자동완성 팝업에 한 줄 표시용)
- `search(query)`: 한글 질의 → 키워드/설명 접두·포함 일치 태그 (인기도 count 내림차순)

순수 로직 — Qt 의존 없음. 데이터베이스는 `read_parquet(TagAsset)` 만 있으면 되므로
테스트에서는 DataFrame 을 돌려주는 가짜로 대체한다.
"""
from __future__ import annotations

import re
import threading
from dataclasses import dataclass
from typing import Iterable

from core.tag_database import TagAsset, get_tag_database

_HANGUL = re.compile(r"[ㄱ-ㆎ가-힣]")
_ANGLE = re.compile(r"[<>]")


def has_hangul(text: str) -> bool:
    return bool(_HANGUL.search(text or ""))


def _norm_tag(value: object) -> str:
    return str(value or "").strip().lower().replace(" ", "_")


@dataclass(frozen=True)
class KoreanTagEntry:
    tag: str
    category: str
    desc: str
    keywords: tuple[str, ...]
    count: int

    @property
    def ko(self) -> str:
        """팝업 한 줄용 — 첫 키워드(한국어 이름) 가 있으면 그것, 없으면 설명."""
        for keyword in self.keywords:
            if keyword and has_hangul(keyword):
                return keyword
        return self.desc

    def to_dict(self) -> dict:
        return {
            "tag": self.tag,
            "ko": self.ko,
            "category": self.category,
            "desc": self.desc,
            "count": self.count,
        }


class KoreanTagLookup:
    def __init__(self, database=None):
        self._database = database
        self._loaded = False
        self._by_tag: dict[str, KoreanTagEntry] = {}
        # (검색어 소문자, 태그 키) — 접두 검색은 정렬 + bisect 대신 선형이라도 충분히 빠르다
        # (키워드 ~10만 개, 한글 질의는 자주 오지 않는다).
        self._keyword_index: list[tuple[str, str]] = []
        # 예열 스레드와 GUI 슬롯이 동시에 첫 질의를 해도 한 번만 적재하고, 적재가 끝나기
        # 전의 반쪽 색인을 누구도 보지 않게 한다.
        self._load_lock = threading.RLock()

    # ── 적재 ──
    def _ensure(self) -> None:
        if self._loaded:          # 빠른 경로 — 적재가 **끝난 뒤에만** 참이 된다
            return
        with self._load_lock:
            if self._loaded:
                return
            try:
                by_tag, keyword_index = self._build_index()
                # 지역에서 다 만든 뒤 한꺼번에 공개 — 부분 색인 노출 없음
                self._by_tag = by_tag
                self._keyword_index = keyword_index
            finally:
                # 실패해도 표시한다 — 카탈로그가 없을 때 키 입력마다 재시도하지 않게.
                self._loaded = True

    def _build_index(self) -> tuple[dict[str, KoreanTagEntry], list[tuple[str, str]]]:
        by_tag: dict[str, KoreanTagEntry] = {}
        keyword_index: list[tuple[str, str]] = []
        database = self._database or get_tag_database()
        try:
            df = database.read_parquet(TagAsset.KOREAN_TAG_CATALOG)
        except Exception as exc:  # pragma: no cover - 데이터 없음
            print(f"[KoreanTag] catalog load failed: {exc}")
            return by_tag, keyword_index
        cols = set(df.columns)
        cats = df["category"] if "category" in cols else [""] * len(df)
        descs = df["desc"] if "desc" in cols else [""] * len(df)
        kws = df["keywords"] if "keywords" in cols else [""] * len(df)
        cnts = df["count"] if "count" in cols else [0] * len(df)
        for tag, cat, desc, kw, cnt in zip(df["tag"], cats, descs, kws, cnts):
            key = _norm_tag(tag)
            if not key:
                continue
            keywords = self._split_keywords(kw)
            try:
                count = int(cnt)
            except (TypeError, ValueError):
                count = 0
            entry = KoreanTagEntry(
                tag=str(tag).strip(),
                category=self._clean(cat),
                desc=self._clean(desc),
                keywords=keywords,
                count=count,
            )
            by_tag[key] = entry
            # 검색 색인은 키워드만 — 설명 문장까지 넣으면 음절 하나에 엉뚱한 태그가 우르르 뜬다
            for keyword in keywords:
                keyword_index.append((keyword.lower(), key))
        return by_tag, keyword_index

    def is_ready(self) -> bool:
        """적재가 끝났는가(기다리지 않는 확인 — 예열 중 GUI 슬롯이 막히지 않게)."""
        return self._loaded

    @staticmethod
    def _clean(value: object) -> str:
        if value is None:
            return ""
        try:
            import pandas as pd
            if pd.isna(value):
                return ""
        except (TypeError, ValueError):
            pass
        return str(value).strip()

    @classmethod
    def _split_keywords(cls, value: object) -> tuple[str, ...]:
        raw = cls._clean(value)
        if not raw:
            return ()
        plain: list[str] = []
        groups: list[str] = []
        for part in raw.split(","):
            part = part.strip()
            if not part:
                continue
            if _ANGLE.search(part):
                # "<헤어스타일>" 은 분류 표지 — 검색은 되게 두되 라벨로는 안 쓴다(뒤로)
                part = _ANGLE.sub("", part).strip()
                if part and part not in groups:
                    groups.append(part)
            elif part not in plain:
                plain.append(part)
        return tuple(plain + [g for g in groups if g not in plain])

    # ── 조회 ──
    def label(self, tag: str) -> KoreanTagEntry | None:
        self._ensure()
        return self._by_tag.get(_norm_tag(tag))

    def labels(self, tags: Iterable[str]) -> list[dict]:
        """자동완성 결과(영문 태그 목록)에 한국어 라벨을 붙인 dict 목록."""
        out: list[dict] = []
        for tag in tags:
            entry = self.label(tag)
            if entry is None:
                out.append({"tag": tag, "ko": "", "category": "", "desc": "", "count": 0})
            else:
                item = entry.to_dict()
                item["tag"] = tag   # 완성기가 준 표기(공백/밑줄)를 유지
                out.append(item)
        return out

    def search(self, query: str, limit: int = 10) -> list[dict]:
        """한글 질의 → 태그. 접두 일치를 포함 일치보다 앞에, 같은 급에서는 count 내림차순."""
        self._ensure()
        q = (query or "").strip().lower()
        if not q:
            return []
        prefix_hits: dict[str, KoreanTagEntry] = {}
        contains_hits: dict[str, KoreanTagEntry] = {}
        for keyword, key in self._keyword_index:
            if key in prefix_hits:
                continue
            if keyword.startswith(q):
                prefix_hits[key] = self._by_tag[key]
                contains_hits.pop(key, None)
            elif key not in contains_hits and q in keyword:
                contains_hits[key] = self._by_tag[key]
        ranked = sorted(prefix_hits.values(), key=lambda e: -e.count)
        ranked += sorted(contains_hits.values(), key=lambda e: -e.count)
        out = []
        for entry in ranked[:limit]:
            item = entry.to_dict()
            item["tag"] = entry.tag.replace(" ", "_")   # 영문 완성기와 같은 표기(long_hair)
            out.append(item)
        return out


_instance: KoreanTagLookup | None = None
_instance_lock = threading.Lock()


def get_korean_tag_lookup() -> KoreanTagLookup:
    """프로세스 싱글턴 — 예열 스레드와 GUI 가 동시에 불러도 한 벌만 만든다(이중 확인)."""
    global _instance
    instance = _instance
    if instance is not None:
        return instance
    with _instance_lock:
        if _instance is None:
            _instance = KoreanTagLookup()
        return _instance


def reset_korean_tag_lookup() -> None:
    global _instance
    with _instance_lock:
        _instance = None
