# utils/tag_completer.py
"""
태그 자동완성 시스템

TagData(parquet) 5개 카테고리(general / character / copyright / artist / meta) 모두
활용. 결과는 카테고리 우선순위 + 접두사 일치 우선으로 정렬.

기본 우선순위: general → character → copyright → artist → meta
(사용자가 쓰는 빈도 기준)
"""
import bisect
import csv
import threading
from pathlib import Path
from typing import List, Optional

from core.tag_database import TagAsset, TagDatabase, get_tag_database
from utils.tag_data import completion_key, completion_tag


# 카테고리 우선순위 — 작을수록 먼저 노출
CATEGORY_PRIORITY = {
    "general":   0,
    "character": 1,
    "copyright": 2,
    "artist":    3,
    "meta":      4,
}

DANBOORU_CATEGORY_MAP = {
    "0": "general",
    "1": "artist",
    "3": "copyright",
    "4": "character",
    "5": "meta",
    "general": "general",
    "artist": "artist",
    "copyright": "copyright",
    "character": "character",
    "meta": "meta",
}


class TagCompleter:
    """태그 자동완성 — 카테고리별 분리 인덱스로 우선순위 보장."""

    def __init__(self, tags_db_path: str = None):
        self.database = TagDatabase(Path(tags_db_path)) if tags_db_path else get_tag_database()
        self.tags_db_path = self.database.root
        # 적재 중에만 쓰는 (lower_key, original_tag) 쌍 — _build_indices 가 정렬 후 비운다
        self._cat_indices: dict[str, list[tuple[str, str]]] = {
            k: [] for k in CATEGORY_PRIORITY
        }
        # 카테고리별 정렬된 lower_keys (bisect용) 와 같은 위치의 표시형 태그
        self._cat_lower_keys: dict[str, list[str]] = {
            k: [] for k in CATEGORY_PRIORITY
        }
        self._cat_tags: dict[str, list[str]] = {
            k: [] for k in CATEGORY_PRIORITY
        }
        # 별칭 → 태그
        self.alias_map: dict[str, str] = {}
        # 태그 인기도 (completion_key → count) — 접두사 매칭 결과 정렬용.
        # TagData 에서 적재하면 TagData.tag_counts 를 **참조**한다(83만 항목을 다시 만들지 않음).
        self._counts: dict[str, int] = {}
        self._tags_set_cache: Optional[set] = None

        self._load_tags()
        self._build_indices()

    @staticmethod
    def _normalise_tag(value: object) -> str:
        """Return the canonical display form used by completion results."""

        return completion_tag(value)

    @classmethod
    def _tag_key(cls, value: object) -> str:
        return completion_key(value)

    def _add_tag(self, category: str, value: object, count: object = None) -> None:
        tag = self._normalise_tag(value)
        if not tag:
            return
        resolved_category = category if category in CATEGORY_PRIORITY else "general"
        key = tag.casefold()
        if key == tag:
            key = tag   # 이미 소문자인 태그(거의 전부)는 키와 표시형이 한 객체
        self._cat_indices[resolved_category].append((key, tag))
        if count is not None:
            try:
                self._counts[key] = int(count)
            except (TypeError, ValueError):
                pass

    # ────────────────────────────────────────
    # 로드
    # ────────────────────────────────────────

    def _load_tags(self):
        """TagData에서 5개 카테고리 모두 로드. 실패 시 CSV 폴백."""
        try:
            # 함수 안에서 import — tests/test_tag_completer.py 가 utils.tag_data.get_tag_data 를
            # patch 해 CSV 폴백을 강제한다. TagData 는 logging 만 쓰므로 stdout 을 가로채지
            # 않는다(예전 redirect_stdout 은 워밍업 스레드와 엇갈려 sys.stdout 을 잃을 수 있었다).
            from utils.tag_data import get_tag_data
            td = get_tag_data()
            if td.is_loaded:
                shared_counts = getattr(td, "tag_counts", None)
                if isinstance(shared_counts, dict):
                    # TagData 가 completion_key 로 만들어 둔 인기도 — 복사하지 않고 참조(읽기 전용)
                    self._counts = shared_counts
                counts: dict[str, int] = {}
                for cat in CATEGORY_PRIORITY:
                    tags = getattr(td, f"{cat}_tags", None) or []
                    for t in tags:
                        self._add_tag(cat, t)
                    counts[cat] = len(tags)
                if any(counts.values()):
                    parts = ", ".join(f"{k}={v:,}" for k, v in counts.items())
                    print(f"[TagCompleter] TagData categories loaded ({parts})")
                    self._load_aliases()
                    return
        except Exception as e:
            print(f"[TagCompleter] TagData load failed; using CSV fallback: {e}")

        # 폴백: 정식 자동완성 카탈로그의 category/count 열을 그대로 사용
        self._load_from_csv_fallback()

        self._load_aliases()

    def _load_aliases(self):
        """Load the manifest alias table, falling back to legacy CSV aliases."""

        try:
            aliases = self.database.load_tag_aliases()
        except Exception as exc:
            print(
                "[TagCompleter] manifest aliases unavailable; "
                f"using CSV fallback: {exc}"
            )
            self._load_aliases_from_csv()
            return

        for alias, canonical in aliases.items():
            alias_key = self._tag_key(alias)
            canonical_tag = self._normalise_tag(canonical)
            if alias_key and canonical_tag:
                self.alias_map[alias_key] = canonical_tag
        if self.alias_map:
            print(f"[TagCompleter] manifest aliases loaded: {len(self.alias_map):,}")

    def _load_aliases_from_csv(self):
        csv_file = self.database.path(TagAsset.AUTOCOMPLETE_CATALOG)
        if not csv_file.exists():
            return
        try:
            with open(csv_file, "r", encoding="utf-8-sig") as f:
                reader = csv.reader(f)
                for row in reader:
                    if not row:
                        continue
                    if len(row) >= 4 and row[3].strip():
                        tag_name = self._normalise_tag(row[0])
                        aliases = [a.strip() for a in row[3].split(",") if a.strip()]
                        for alias in aliases:
                            alias_key = self._tag_key(alias)
                            if alias_key and tag_name:
                                self.alias_map[alias_key] = tag_name
            if self.alias_map:
                print(f"[TagCompleter] aliases loaded: {len(self.alias_map):,}")
        except Exception:
            pass

    def _load_from_csv_fallback(self):
        """Load tags, Danbooru categories and counts from the CSV catalog."""
        csv_file = self.database.path(TagAsset.AUTOCOMPLETE_CATALOG)
        if not csv_file.exists():
            print(f"[TagCompleter] autocomplete catalog not found: {csv_file}")
            return
        try:
            with open(csv_file, "r", encoding="utf-8-sig") as f:
                reader = csv.reader(f)
                for row in reader:
                    if not row:
                        continue
                    tag_name = self._normalise_tag(row[0])
                    if not tag_name:
                        continue
                    raw_category = row[1].strip().casefold() if len(row) >= 2 else "0"
                    category = DANBOORU_CATEGORY_MAP.get(raw_category, "general")
                    count = row[2].strip() if len(row) >= 3 else None
                    self._add_tag(category, tag_name, count)
            counts = {
                category: len(self._cat_indices[category])
                for category in CATEGORY_PRIORITY
            }
            parts = ", ".join(f"{key}={value:,}" for key, value in counts.items())
            print(f"[TagCompleter] CSV tags loaded ({parts})")
        except Exception as e:
            print(f"[TagCompleter] autocomplete catalog load failed: {e}")

    def _build_indices(self):
        """카테고리별로 정렬 + 키/표시형 병렬 리스트로 분리 (bisect용).

        적재용 (key, tag) 튜플 83만 개는 정렬 뒤 버린다 — 이후 검색은 같은 위치의
        `_cat_lower_keys[cat][i]` / `_cat_tags[cat][i]` 로 본다(같은 문자열 객체 참조).
        """
        for cat in CATEGORY_PRIORITY:
            pairs = self._cat_indices[cat]
            pairs.sort(key=lambda x: x[0])
            self._cat_lower_keys[cat] = [p[0] for p in pairs]
            self._cat_tags[cat] = [p[1] for p in pairs]
            pairs.clear()
        self._tags_set_cache = None

    # ────────────────────────────────────────
    # 검색
    # ────────────────────────────────────────

    def get_suggestions(self, prefix: str, max_count: int = 10) -> List[str]:
        """접두사로 태그 추천.

        결과 순서: general → character → copyright → artist → meta
        각 카테고리 내에서는 접두사 일치 우선, 그 다음 별칭 일치, 그 다음 포함 일치.
        """
        if not prefix:
            return []
        prefix_lower = self._tag_key(prefix)
        if not prefix_lower:
            return []

        seen: set[str] = set()
        results: list[str] = []

        # 1. 카테고리별 접두사 매칭 (general → character → copyright → artist → meta)
        for cat in sorted(CATEGORY_PRIORITY, key=lambda c: CATEGORY_PRIORITY[c]):
            if len(results) >= max_count:
                return results
            results += self._prefix_match(cat, prefix_lower, max_count - len(results), seen)

        # 2. 별칭 매칭 (모든 카테고리 통합)
        if len(results) < max_count:
            for alias, tag_name in self.alias_map.items():
                if alias.startswith(prefix_lower):
                    tl = self._tag_key(tag_name)
                    if tl not in seen:
                        seen.add(tl)
                        results.append(tag_name)
                        if len(results) >= max_count:
                            return results

        # 3. 포함 매칭 (general에서만 — 너무 많아지지 않게)
        # 키는 적재 때 만든 것을 그대로 쓴다. 태그마다 _tag_key 를 다시 계산하던 것
        # (general 9만 개 × 드문 접두어 ≈ 35ms GUI 정지)과 결과가 같다 — 키 = _tag_key(태그).
        if len(results) < max_count:
            keys = self._cat_lower_keys["general"]
            tags = self._cat_tags["general"]
            for i, tag_lower in enumerate(keys):
                if prefix_lower in tag_lower and tag_lower not in seen:
                    seen.add(tag_lower)
                    results.append(tags[i])
                    if len(results) >= max_count:
                        return results

        return results

    # 접두사 매칭 후보 최대 스캔 수 — 인기도 정렬을 위해 후보를 모은 뒤 count로 정렬.
    # 너무 크면 1글자 접두사에서 느려지고, 너무 작으면 인기 태그를 놓침. 3000이면
    # 대부분의 접두사를 커버하면서 정렬 비용도 무시할 만함.
    _PREFIX_SCAN_CAP = 3000

    def _prefix_match(self, cat: str, prefix_lower: str,
                      remaining: int, seen: set) -> list[str]:
        """카테고리 내 접두사 매칭 — bisect로 후보 수집 후 인기도(count) 내림차순 정렬.

        기존엔 알파벳순이라 '1g' → '1girl'이 위로 안 왔음. count 정렬로 자주 쓰는
        태그가 먼저 노출됨.
        """
        if remaining <= 0:
            return []
        keys = self._cat_lower_keys[cat]
        if not keys:
            return []
        start = bisect.bisect_left(keys, prefix_lower)
        tags = self._cat_tags[cat]
        # 1) 접두사 일치 후보 수집 (최대 _PREFIX_SCAN_CAP) — (키, 표시형) 쌍, 키는 적재 때 계산됨
        matched: list[tuple[str, str]] = []
        for i in range(start, min(len(keys), start + self._PREFIX_SCAN_CAP)):
            key = keys[i]
            if not key.startswith(prefix_lower):
                break
            if key in seen:
                continue
            matched.append((key, tags[i]))
        if not matched:
            return []
        # 2) 인기도(count) 내림차순 정렬 — count 동률이면 짧은 태그 우선(더 일반적)
        c = self._counts
        matched.sort(key=lambda pair: (-c.get(pair[0], 0), len(pair[1])))
        # 3) 상위 remaining개 반환 + seen 갱신
        out: list[str] = []
        for key, orig in matched[:remaining]:
            seen.add(key)
            out.append(orig)
        return out

    # ────────────────────────────────────────
    # 유틸
    # ────────────────────────────────────────

    def is_valid_tag(self, tag: str) -> bool:
        """정렬된 카테고리 키에서 이진 탐색 — 83만 개짜리 set 을 따로 들지 않는다."""
        key = self._tag_key(tag)
        if not key:
            return False
        for keys in self._cat_lower_keys.values():
            i = bisect.bisect_left(keys, key)
            if i < len(keys) and keys[i] == key:
                return True
        return False

    @property
    def tags_set(self) -> set:
        """호환용 — 전체 키 집합을 처음 요청될 때 만든다(is_valid_tag 는 쓰지 않는다)."""
        if self._tags_set_cache is None:
            self._tags_set_cache = set().union(*self._cat_lower_keys.values())
        return self._tags_set_cache

    def count(self) -> int:
        return sum(len(self._cat_tags[c]) for c in CATEGORY_PRIORITY)


# 싱글톤 — 기동 워밍업 스레드와 GUI 슬롯이 동시에 불러도 한 번만 적재한다
_completer_instance: Optional[TagCompleter] = None
_completer_lock = threading.Lock()


def get_tag_completer() -> TagCompleter:
    global _completer_instance
    instance = _completer_instance
    if instance is not None:
        return instance
    with _completer_lock:
        if _completer_instance is None:
            _completer_instance = TagCompleter()
        return _completer_instance


def try_get_tag_completer() -> Optional[TagCompleter]:
    """막지 않는 접근 — 다른 스레드(기동 워밍업)가 적재 중이면 None.

    GUI 스레드 슬롯용이다. 적재가 끝나기를 수 초 기다리며 창을 멈추는 대신 빈 결과를
    돌려주게 한다. 아무도 적재하고 있지 않으면 호출한 스레드에서 적재한다.
    """
    global _completer_instance
    instance = _completer_instance
    if instance is not None:
        return instance
    if not _completer_lock.acquire(blocking=False):
        return None
    try:
        if _completer_instance is None:
            _completer_instance = TagCompleter()
        return _completer_instance
    finally:
        _completer_lock.release()


def is_tag_completer_ready() -> bool:
    """적재가 끝났는가 (기다리지 않는 확인)."""
    return _completer_instance is not None


def reset_tag_completer() -> None:
    """싱글톤 캐시 비우기 — 데이터 갱신 후 강제 재로드용."""
    global _completer_instance
    with _completer_lock:
        _completer_instance = None
