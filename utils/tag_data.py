# utils/tag_data.py
"""
tags_dictionary.parquet(구: danbooru2025-alltime-tag-counts.parquet) 기반 통합 태그 데이터.

프로세스에 한 벌만 둔다(get_tag_data). 83만 개 태그 어휘를 소비자마다 복사하지 않도록
- `*_tags` 리스트는 자동완성 표기(밑줄형, TagCompleter 가 쓰는 그대로)로 담고,
  원본 parquet 문자열과 같으면 그 객체를 그대로 쓴다.
- `tag_counts` 는 TagCompleter 의 비교 키(completion_key)로 담아 완성기가 참조만 한다.
- 이름 set(character/copyright/artist/meta)은 공백형 소문자다. TagClassifier 가
  복사하지 않고 그대로 참조하므로 **읽기 전용**으로 다룬다(meta 는 분류기가 복사해 보강).
"""
import logging
import os
import threading
from pathlib import Path
from typing import List, Set, Optional

logger = logging.getLogger(__name__)


def completion_tag(value: object) -> str:
    """자동완성 표시형 — 공백 묶음을 밑줄 하나로 ('long hair' → 'long_hair').

    이미 그 형태인 문자열은 같은 객체를 돌려준다(CPython split/join 단일 원소 경로).
    """
    return "_".join(str(value).strip().split())


def completion_key(value: object) -> str:
    """자동완성 비교 키 — completion_tag 의 casefold. TagCompleter 와 tag_counts 가 공유한다."""
    return completion_tag(value).casefold()


def _reuse(candidate: str, original: str) -> str:
    """값이 같으면 원본 객체를 써서 같은 문자열을 두 벌 들지 않는다."""
    return original if candidate == original else candidate


class TagData:
    """parquet 기반 통합 태그 데이터 (싱글톤)"""

    def __init__(self, parquet_path: str = None):
        if parquet_path is None:
            from core.fetch_data import DATA_DIR   # 데이터셋 폴더의 단일 출처(= config.PARQUET_DIR)
            base = Path(__file__).parent.parent
            # 우선순위: 신규 tags_dictionary.parquet (tag/category/color/count, 2026)
            #          → 구버전 danbooru2025-alltime-tag-counts.parquet (tag_string/tag_type/tag_count)
            candidates = [
                DATA_DIR / "tags_dictionary.parquet",
                base / "danbooru2025-alltime-tag-counts.parquet",
            ]
            parquet_path = str(next((c for c in candidates if c.exists()), candidates[-1]))
        self._parquet_path = parquet_path

        # tag_type별 리스트 (count 내림차순, 자동완성 표기 — utils/tag_completer.py 가 읽는다)
        self.general_tags: List[str] = []
        self.character_tags: List[str] = []
        self.copyright_tags: List[str] = []
        self.artist_tags: List[str] = []
        self.meta_tags: List[str] = []

        # 이름 set (공백형, strip+lower, 제거 토글·분류 lookup용 — TagClassifier 가 참조)
        self.character_set: Set[str] = set()
        self.copyright_set: Set[str] = set()
        self.artist_set: Set[str] = set()
        self.meta_set: Set[str] = set()

        # 태그 인기도 — completion_key → count (자동완성 인기순 정렬용, TagCompleter 가 참조)
        self.tag_counts: dict[str, int] = {}

        self._loaded = False
        self._load()

    def _load(self):
        """parquet 파일 로드 → tag_type별 분리"""
        # 로그는 logging 으로만 — 예전 이모지 print 는 CP949 콘솔에서 UnicodeEncodeError 를 내서
        # 소비자들이 redirect_stdout 으로 감쌌고, 그 전역 교체가 스레드끼리 엇갈렸다.
        if not os.path.exists(self._parquet_path):
            logger.warning("tag-counts parquet 없음: %s", self._parquet_path)
            return

        try:
            import pandas as pd
            import pyarrow.parquet as pq
            # 두 스키마 호환: 신규(tag/category/count) vs 구(tag_string/tag_type/tag_count)
            names = pq.read_schema(self._parquet_path).names
            c_tag = "tag" if "tag" in names else "tag_string"
            c_type = "category" if "category" in names else "tag_type"
            c_cnt = "count" if "count" in names else "tag_count"
            df = pd.read_parquet(self._parquet_path, columns=[c_tag, c_type, c_cnt])
            df = df.rename(columns={c_tag: "tag", c_type: "type", c_cnt: "count"})

            # count 내림차순 정렬 (자동완성 인기순)
            df = df.sort_values("count", ascending=False)
            raw_tags = df["tag"].astype(str).tolist()
            tag_types = df["type"].tolist()
            tag_counts = df["count"].tolist()
            del df

            type_map = {
                "general": (self.general_tags, None),
                "character": (self.character_tags, self.character_set),
                "copyright": (self.copyright_tags, self.copyright_set),
                "artist": (self.artist_tags, self.artist_set),
                "meta": (self.meta_tags, self.meta_set),
            }

            counts = self.tag_counts
            for raw, tag_type, cnt in zip(raw_tags, tag_types, tag_counts):
                target = type_map.get(tag_type)
                if target is not None:
                    tag_list, tag_set = target
                    display = raw.replace("_", " ")
                    # 자동완성 표기 — 예전 공백형 표시 문자열을 완성기가 정규화하던 결과와 같다
                    tag_list.append(_reuse(completion_tag(display), raw))
                    if tag_set is not None:
                        name = display.strip().lower()
                        if name:
                            tag_set.add(name)
                # 인기도 맵 — 완성기가 예전에 다시 만들던 키(completion_key)를 여기서 바로 쓴다
                if raw:
                    try:
                        count = int(cnt)
                    except (ValueError, TypeError):
                        continue
                    counts[_reuse(completion_key(raw.lower().replace(" ", "_")), raw)] = count

            self._loaded = True
            total = sum(len(v[0]) for v in type_map.values())
            logger.info(
                "TagData 로드: %s개 태그 (general=%s, character=%s, copyright=%s, "
                "artist=%s, meta=%s)",
                f"{total:,}",
                f"{len(self.general_tags):,}",
                f"{len(self.character_tags):,}",
                f"{len(self.copyright_tags):,}",
                f"{len(self.artist_tags):,}",
                f"{len(self.meta_tags):,}",
            )
        except Exception as e:
            logger.warning("TagData 로드 실패: %s", e)

    @property
    def is_loaded(self) -> bool:
        return self._loaded


# ── 싱글톤 ──
_tag_data_instance: Optional[TagData] = None
_tag_data_lock = threading.Lock()


def get_tag_data() -> TagData:
    """싱글톤 인스턴스 반환 — 워밍업 스레드와 GUI 스레드가 동시에 불러도 한 번만 적재한다."""
    global _tag_data_instance
    instance = _tag_data_instance
    if instance is not None:
        return instance
    with _tag_data_lock:
        if _tag_data_instance is None:
            _tag_data_instance = TagData()
        return _tag_data_instance


def reset_tag_data() -> None:
    """싱글톤 캐시 비우기 — 데이터 갱신 후 강제 재로드용(완성기·분류기도 함께 초기화할 것)."""
    global _tag_data_instance
    with _tag_data_lock:
        _tag_data_instance = None
