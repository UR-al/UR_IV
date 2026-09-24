"""Event 태그 텍스트 필터 — 제외 태그는 tag_matcher 공용 매처로 글자 그대로 거른다.

예전 복제 코드는 `.str.lower().str.contains(tag)` 를 형태마다 두 번(정규식 기본값) 돌려
- 컬럼을 두 번 소문자화·스캔했고
- '(' 같은 태그 글자를 정규식으로 해석해 `kafka_(honkai:_star_rail)` 을 못 잡았다.

숨은 레거시 EventGenTab 만 쓰던 child_include/child_exclude·search_events·build_steps·
get_event_summary 는 탭과 함께 지웠다(P13c) — Vue 이벤트 검색은 부모 제외 태그(exclude_tags)만 쓴다.
"""
from __future__ import annotations

import inspect
import tempfile
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from core.event_data_loader import EventDataLoader


def _row(post_id, general, *, parent_id=None, score=0):
    return {
        "id": post_id,
        "parent_id": parent_id,
        "has_children": parent_id is None,
        "has_visible_children": parent_id is None,
        "tag_string_general": general,
        "tag_string_character": "",
        "tag_string_copyright": "",
        "tag_string_artist": "",
        "tag_string_meta": "",
        "rating": "g",
        "score": score,
        "fav_count": 0,
        "image_width": 1024,
        "image_height": 1024,
    }


class EventTagTextFilterTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._tmp = tempfile.TemporaryDirectory()
        root = Path(cls._tmp.name)
        rows = [
            _row(10, "1girl smile", score=5),
            _row(11, "1girl", parent_id=10),
            _row(12, "1girl Smile", parent_id=10),
            _row(20, "1girl smile kafka_(honkai:_star_rail)", score=3),
            _row(21, "1girl", parent_id=20),
            _row(22, "1girl", parent_id=20),
            _row(30, "1girl smile long_hair", score=1),
            _row(31, "1girl", parent_id=30),
            _row(32, "1girl", parent_id=30),
        ]
        pd.DataFrame(rows).to_parquet(root / "danbooru_g.parquet", index=False)
        cls.loader = EventDataLoader(str(root))
        with redirect_stdout(StringIO()):
            cls.loader.load_parquets_by_rating(["g"])

    @classmethod
    def tearDownClass(cls):
        cls._tmp.cleanup()

    def _parent_ids(self, **kwargs):
        with redirect_stdout(StringIO()):
            results = self.loader.search_by_prompt("smile", min_children=2, **kwargs)
        return [int(entry["parent"]["id"]) for entry in results]

    def test_exclude_matches_parentheses_literally_in_both_spellings(self):
        self.assertEqual(self._parent_ids(), [10, 20, 30])
        for spelling in ("kafka (honkai: star rail)", "kafka_(honkai:_star_rail)"):
            with self.subTest(spelling=spelling):
                self.assertEqual(self._parent_ids(exclude_tags=spelling), [10, 30])

    def test_unbalanced_parenthesis_in_a_tag_does_not_crash(self):
        # 예전 정규식 해석은 '(honkai:' 에서 re.error 로 검색 전체가 죽었다
        self.assertEqual(self._parent_ids(exclude_tags="(honkai:"), [10, 30])

    def test_exclude_matches_both_space_and_underscore_forms(self):
        for spelling in ("long hair", "long_hair"):
            with self.subTest(spelling=spelling):
                self.assertEqual(self._parent_ids(exclude_tags=spelling), [10, 20])

    def test_exclude_fallback_scans_each_term_once(self):
        """매처가 질의를 못 읽을 때(_query_mask → None)의 폴백도 같은 헬퍼로 텀당 한 번만 훑는다."""
        import core.event_data_loader as module

        calls = []
        real = module.contains_tag_text

        def counting(series, text):
            calls.append(text)
            return real(series, text)

        with patch.object(EventDataLoader, "_query_mask", return_value=None), \
                patch.object(module, "contains_tag_text", side_effect=counting), \
                redirect_stdout(StringIO()):
            results = self.loader.search_by_prompt(
                "smile", exclude_tags="long hair, ", min_children=2
            )

        self.assertEqual(sorted(int(r["parent"]["id"]) for r in results), [10, 20])
        # 형태(공백/밑줄)마다 따로 스캔하지 않는다 — contains_tag_text 가 두 형태를 한 번에 본다
        self.assertEqual(calls, ["long hair"])

    def test_legacy_tab_only_api_stays_removed(self):
        for name in ("search_events", "build_steps", "get_event_summary"):
            self.assertFalse(hasattr(EventDataLoader, name), name)
        params = inspect.signature(EventDataLoader.search_by_prompt).parameters
        for name in ("child_include", "child_exclude", "min_score", "require_variant_set"):
            self.assertNotIn(name, params)


if __name__ == "__main__":
    unittest.main()
