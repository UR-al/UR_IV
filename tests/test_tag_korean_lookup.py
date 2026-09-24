"""core.tag_korean_lookup — 자동완성 한국어 라벨과 한글 검색 (Qt 없이)."""
import unittest

import pandas as pd

from core.tag_database import TagAsset
from core.tag_korean_lookup import KoreanTagLookup, has_hangul


class _FakeDatabase:
    def __init__(self, frame):
        self.frame = frame
        self.reads = 0

    def read_parquet(self, asset):
        assert asset is TagAsset.KOREAN_TAG_CATALOG
        self.reads += 1
        return self.frame


def _catalog():
    return pd.DataFrame({
        "tag": ["long hair", "long_sleeves", "solo", "highres", "weird"],
        "count": [4_800_000, 900_000, 5_400_000, 5_900_000, 3],
        "category": ["패션 > 헤어스타일", "패션 > 의류", "인물 > 인원수", "메타 > 해상도", None],
        "desc": ["어깨를 넘어 허리까지 오는 길이의 머리임.", "긴 소매 옷.", "인물이 한 명만 존재하는 이미지임.", "고해상도 이미지임.", None],
        "keywords": ["<헤어스타일>, 긴 머리, 장발", "<의류>, 긴 소매", "<인원>, 솔로, 1인, 혼자", "<해상도>, 고해상도, 고화질", None],
    })


class KoreanTagLookupTests(unittest.TestCase):
    def setUp(self):
        self.db = _FakeDatabase(_catalog())
        self.lookup = KoreanTagLookup(self.db)

    def test_label_is_keyed_by_normalised_tag_and_loads_once(self):
        a = self.lookup.label("long hair")
        b = self.lookup.label("Long_Hair")
        self.assertIsNotNone(a)
        self.assertEqual(a, b, "공백/밑줄/대소문자는 같은 태그")
        self.assertEqual(a.category, "패션 > 헤어스타일")
        self.assertEqual(a.ko, "긴 머리", "첫 한국어 키워드가 팝업 라벨")
        self.assertEqual(a.keywords, ("긴 머리", "장발", "헤어스타일"), "<…> 분류 표지는 꺾쇠를 벗겨 맨 뒤로")
        self.assertEqual(self.db.reads, 1, "parquet 은 한 번만 읽는다")

    def test_labels_keep_completer_spelling_and_tolerate_unknown_tags(self):
        rows = self.lookup.labels(["long_hair", "nonexistent_tag"])
        self.assertEqual(rows[0]["tag"], "long_hair", "완성기가 준 표기 그대로")
        self.assertEqual(rows[0]["ko"], "긴 머리")
        self.assertEqual(rows[1], {"tag": "nonexistent_tag", "ko": "", "category": "", "desc": "", "count": 0})

    def test_hangul_search_prefers_prefix_then_popularity(self):
        hits = [h["tag"] for h in self.lookup.search("긴")]
        self.assertEqual(hits[:2], ["long_hair", "long_sleeves"], "접두 일치는 count 내림차순, 밑줄 표기로 통일")
        self.assertEqual([h["tag"] for h in self.lookup.search("장발")], ["long_hair"])
        self.assertEqual([h["tag"] for h in self.lookup.search("고화질")], ["highres"])
        # 포함 일치는 접두 일치 뒤에 온다 (키워드에서만 — 설명 문장은 색인하지 않는다)
        hits = [h["tag"] for h in self.lookup.search("머리")]
        self.assertIn("long_hair", hits)
        self.assertEqual(self.lookup.search("이미지임"), [], "설명 문장은 검색 대상이 아니다")
        self.assertEqual(self.lookup.search("   "), [])
        self.assertEqual(self.lookup.search("zzz"), [])

    def test_search_limit_and_result_shape(self):
        hits = self.lookup.search("고", limit=1)
        self.assertEqual(len(hits), 1)
        self.assertEqual(set(hits[0]), {"tag", "ko", "category", "desc", "count"})

    def test_rows_without_korean_data_are_harmless(self):
        entry = self.lookup.label("weird")
        self.assertIsNotNone(entry)
        self.assertEqual(entry.ko, "")
        self.assertEqual(entry.keywords, ())

    def test_has_hangul(self):
        self.assertTrue(has_hangul("장발"))
        self.assertTrue(has_hangul("long 머리"))
        self.assertFalse(has_hangul("long hair"))
        self.assertFalse(has_hangul(""))


if __name__ == "__main__":
    unittest.main()
