"""Event Gen parquet shard regression tests."""

from __future__ import annotations

import tempfile
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path

import pandas as pd

from core.event_data_loader import EventDataLoader


def _event_row(post_id: int, *, parent_id=None, rating: str = "g") -> dict:
    return {
        "id": post_id,
        "parent_id": parent_id,
        "has_children": parent_id is None,
        "has_visible_children": parent_id is None,
        "tag_string_general": f"event_{post_id}",
        "tag_string_character": "",
        "tag_string_copyright": "",
        "tag_string_artist": "",
        "tag_string_meta": "",
        "rating": rating,
        "score": post_id,
        "fav_count": 0,
        "image_width": 1024,
        "image_height": 1024,
    }


class EventDataLoaderTests(unittest.TestCase):
    def test_cross_rating_parent_copies_are_deduplicated(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            parent = _event_row(10)
            pd.DataFrame([parent, _event_row(11, parent_id=10, rating="g")]).to_parquet(
                root / "danbooru_g.parquet", index=False
            )
            pd.DataFrame([parent, _event_row(12, parent_id=10, rating="s")]).to_parquet(
                root / "danbooru_s.parquet", index=False
            )

            loader = EventDataLoader(str(root))
            with redirect_stdout(StringIO()):
                frame = loader.load_parquets_by_rating(["g", "s"])

            self.assertEqual(len(frame), 3)
            self.assertEqual(frame["id"].nunique(), 3)
            self.assertEqual(set(loader.parents_df["id"]), {10})
            self.assertEqual(set(loader.parent_child_map[10]), {11, 12})


# ── search_by_prompt 회귀: 예전 구현(후보마다 children_df 전체 isin 스캔)을 기준으로 ──

def _reference_search(loader, prompt, exclude_tags="", min_children=2, max_children=20, limit=100):
    """최적화 전 search_by_prompt 의 순서·결과를 그대로 옮긴 오라클 (필터 문법은 같은 매처).

    child_include/child_exclude 는 숨은 레거시 EventGenTab 과 함께 지웠다(P13c) — 오라클에서도 뺐다.
    """
    from core.tag_matcher import filter_dataframe

    parse = EventDataLoader._parse_tags
    units = EventDataLoader._query_units(prompt)
    filtered = loader.parents_df.copy()
    if prompt.strip():
        filtered = filtered[filter_dataframe(filtered, 'tag_string_general', prompt)]
    if exclude_tags.strip():
        filtered = filtered[~filter_dataframe(filtered, 'tag_string_general', exclude_tags)]
    scored = []
    for _, parent in filtered.iterrows():
        parent_id = int(parent['id'])
        if parent_id not in loader.parent_child_map:
            continue
        tag_set = parent.get('_tag_set', set()) or parse(parent.get('tag_string_general', ''))
        # 매처가 통과시킨 부모는 질의를 만족한다 — 유사도는 순위만 정한다('최소 1개 일치'로 거르지 않음)
        similarity = EventDataLoader._unit_similarity(units, tag_set)[1] if units else 1.0
        children = loader.children_df[loader.children_df['id'].isin(loader.parent_child_map[parent_id])].copy()
        if len(children) < min_children or len(children) > max_children:
            continue
        children = children.sort_values('id')
        scored.append({
            'parent': parent.to_dict(),
            'children': children.to_dict('records'),
            'child_count': len(children),
            'similarity': round(similarity, 3),
        })
    scored.sort(key=lambda x: (x['similarity'], x['parent'].get('score', 0)), reverse=True)
    return scored[:limit]


def _row(post_id, general, *, parent_id=None, score=0, character="", copyright="", artist=""):
    row = _event_row(post_id, parent_id=parent_id)
    row.update({
        "tag_string_general": general,
        "tag_string_character": character,
        "tag_string_copyright": copyright,
        "tag_string_artist": artist,
        "score": score,
    })
    return row


def _synthetic_rows():
    rows = [
        # parent 10: miku, 3 children
        _row(10, "1girl smile school_uniform", score=5, character="hatsune_miku", copyright="vocaloid"),
        _row(11, "1girl smile school_uniform", parent_id=10, character="hatsune_miku"),
        _row(12, "1girl nude", parent_id=10, character="hatsune_miku"),
        _row(13, "1girl bra", parent_id=10, character="hatsune_miku"),
        # parent 20: rin, 2 children — 유사도 동률에서 score 로 줄 선다
        _row(20, "1girl smile school_uniform", score=9, character="kagamine_rin", copyright="vocaloid",
             artist="wlop"),
        _row(21, "1girl smile", parent_id=20),
        _row(22, "1girl blush", parent_id=20),
        # parent 30: 자식 1명 — min_children=2 에서 탈락
        _row(30, "1girl smile", score=100),
        _row(31, "1girl", parent_id=30),
        # parent 40: 1boy, 4 children
        _row(40, "1boy smile", score=3, character="kaito", copyright="vocaloid"),
        _row(41, "1boy", parent_id=40),
        _row(42, "1boy smile", parent_id=40),
        _row(43, "1boy nude", parent_id=40),
        _row(44, "1boy", parent_id=40),
        # parent 50: 일반 태그 결측(NaN) — 필터 전용 검색에서도 죽지 않아야
        _row(50, None, score=2, character="hatsune_miku", copyright="vocaloid"),
        _row(51, "1girl", parent_id=50),
        _row(52, "1girl smile", parent_id=50),
        # parent 60: 동률 유사도, 같은 score → 원래 순서
        _row(60, "1girl smile school_uniform", score=5, character="megurine_luka"),
        _row(61, "1girl", parent_id=60),
        _row(62, "1girl smile", parent_id=60),
    ]
    return rows


class EventSearchRegressionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._tmp = tempfile.TemporaryDirectory()
        root = Path(cls._tmp.name)
        pd.DataFrame(_synthetic_rows()).to_parquet(root / "danbooru_g.parquet", index=False)
        cls.loader = EventDataLoader(str(root))
        with redirect_stdout(StringIO()):
            cls.loader.load_parquets_by_rating(["g"])

    @classmethod
    def tearDownClass(cls):
        cls._tmp.cleanup()

    def search(self, *args, **kwargs):
        with redirect_stdout(StringIO()):
            return self.loader.search_by_prompt(*args, **kwargs)

    @staticmethod
    def _ids(results):
        return [(int(r['parent']['id']), [int(c['id']) for c in r['children']]) for r in results]

    def test_matches_the_previous_implementation(self):
        cases = [
            dict(prompt="1girl, smile"),
            dict(prompt="1girl, smile", limit=2),
            dict(prompt="1girl, smile", min_children=3),
            dict(prompt="1girl, smile", max_children=2),
            dict(prompt="smile", exclude_tags="school_uniform"),
            dict(prompt="smile", exclude_tags="1boy, school_uniform"),
            dict(prompt="1boy", min_children=1, max_children=10),
            dict(prompt="[1girl|1boy], smile"),
        ]
        for case in cases:
            with self.subTest(case=case):
                expected = _reference_search(self.loader, **case)
                actual = self.search(**case)
                self.assertEqual(self._ids(actual), self._ids(expected))
                self.assertEqual([r['similarity'] for r in actual], [r['similarity'] for r in expected])
                self.assertEqual([r['child_count'] for r in actual], [r['child_count'] for r in expected])
                self.assertEqual(
                    [r['parent']['tag_string_character'] for r in actual],
                    [r['parent']['tag_string_character'] for r in expected],
                )

    def test_ties_rank_by_score_then_original_order(self):
        results = self.search("1girl, smile, school_uniform")
        self.assertEqual([int(r['parent']['id']) for r in results], [20, 10, 60])

    def test_character_copyright_artist_filters_are_applied(self):
        # 일반 태그와 함께 — 필터가 조용히 빠지지 않는다
        both = self.search("1girl, smile", character="hatsune_miku")
        self.assertEqual({int(r['parent']['id']) for r in both}, {10})
        self.assertEqual(
            {int(r['parent']['id']) for r in self.search("smile", copyright="vocaloid")},
            {10, 20, 40},
        )
        self.assertEqual([int(r['parent']['id']) for r in self.search("smile", artist="wlop")], [20])
        self.assertEqual(self.search("smile", character="nobody_at_all"), [])

    def test_filter_only_search_needs_no_general_tags(self):
        results = self.search("", character="hatsune miku")
        # parent 50 은 일반 태그가 결측이어도 캐릭터로 찾힌다. score 순.
        self.assertEqual([int(r['parent']['id']) for r in results], [10, 50])
        self.assertTrue(all(r['similarity'] == 1.0 for r in results))
        self.assertEqual(self.search("", copyright="[vocaloid|nothing]", limit=2)[0]['parent']['id'], 20)
        self.assertEqual(self.search(""), [], "아무 조건도 없으면 검색하지 않는다")

    def test_cancel_check_stops_the_search(self):
        from core.event_data_loader import EventSearchCancelled

        with self.assertRaises(EventSearchCancelled):
            self.search("1girl", cancel_check=lambda: True)

    def test_child_positions_follow_parent_child_map(self):
        for parent_id, child_ids in self.loader.parent_child_map.items():
            rows = self.loader.children_df.iloc[self.loader._child_positions[parent_id]]
            self.assertEqual(rows['id'].tolist(), child_ids)

    def test_nan_general_tags_parse_to_empty(self):
        self.assertEqual(EventDataLoader._parse_tags(float("nan")), set())
        self.assertEqual(EventDataLoader._parse_tags(None), set())
        self.assertEqual(EventDataLoader._parse_tags(pd.NA), set())

    def test_syntax_only_queries_return_what_the_matcher_accepts(self):
        # 예전엔 연산자가 남은 토큰으로 유사도를 세어 '최소 1개 일치'에서 모두 탈락했다(0건)
        cases = {
            "[1girl|1boy]": {10, 20, 40, 60},
            "*1girl": {10, 20, 60},
            "*smile": {10, 20, 40, 60},
            "[1girl, smile]": {10, 20, 60},
            "_school_": {10, 20, 60},
            "_uniform": {10, 20, 60},
            "school_": {10, 20, 60},
            "[school uniform|nude]": {10, 20, 60},
        }
        for prompt, expected in cases.items():
            with self.subTest(prompt=prompt):
                self.assertEqual({int(r['parent']['id']) for r in self.search(prompt)}, expected)

    def test_or_group_is_not_counted_as_an_unmatched_tag(self):
        # [1girl|1boy], smile — 두 단위 모두 맞는다. 태그가 딱 맞는 40 이 1.0, 나머지는 score 순
        results = self.search("[1girl|1boy], smile")
        self.assertEqual([int(r['parent']['id']) for r in results], [40, 20, 10, 60])
        self.assertEqual([r['similarity'] for r in results], [1.0, 0.867, 0.867, 0.867])
        self.assertTrue(all(r['matched_tags'] == 2 and r['total_query_tags'] == 2 for r in results))
        self.assertEqual(self.search("*1girl")[0]['similarity'], 0.733)

    def test_query_units_follow_the_matcher_grammar(self):
        units = EventDataLoader._query_units
        self.assertEqual(units("[school uniform|nude], *1girl, _hair, long_, _x_"), [
            (('contains', 'school uniform'), ('contains', 'nude')),
            (('exact', '1girl'),),
            (('suffix', 'hair'),),
            (('prefix', 'long'),),
            (('contains', 'x'),),
        ])
        self.assertEqual(units("[a|b|], smile"), [(('contains', 'smile'),)], "항상 통과하는 [A|B|] 는 세지 않는다")
        self.assertEqual(units("[a, b_c]"), [(('contains', 'a'),), (('contains', 'b c'),)])
        self.assertEqual(units("1girl smile"), [(('contains', '1girl'),), (('contains', 'smile'),)])
        self.assertEqual(units("1girl, 1GIRL, 1girl"), [(('contains', '1girl'),)])
        self.assertEqual(units(""), [])
        self.assertEqual(units(None), [])

    def test_operator_term_with_spaces_is_one_unit_like_the_matcher(self):
        # 쉼표·대괄호 없이 연산자가 붙은 텀 — 매처는 통째로 한 텀('school uniform' 정확 일치)으로 읽는다.
        # 예전엔 공백으로 나눠 정확 'school' + 부분 'uniform' 두 단위가 되어 늘 1/2 로 세어졌다.
        units = EventDataLoader._query_units
        self.assertEqual(units("*school uniform"), [(('exact', 'school uniform'),)])
        self.assertEqual(units("_school uniform"), [(('suffix', 'school uniform'),)])
        self.assertEqual(units("school uniform_"), [(('prefix', 'school uniform'),)])
        self.assertEqual(units("_school uniform_"), [(('contains', 'school uniform'),)])
        # 연산자 없는 공백 질의는 예전처럼 danbooru 식으로 나눈다
        self.assertEqual(units("school uniform"), [(('contains', 'school'),), (('contains', 'uniform'),)])
        results = self.search("*school uniform")
        self.assertEqual({int(r['parent']['id']) for r in results}, {10, 20, 60})
        self.assertTrue(all(r['matched_tags'] == 1 and r['total_query_tags'] == 1 for r in results))
        # 태그가 셋인 부모 — overlap 1.0, jaccard 1/3 → 0.6 + 0.4/3
        self.assertTrue(all(r['similarity'] == 0.733 for r in results))

    def test_plain_tags_score_exactly_like_the_previous_formula(self):
        def legacy(query: set, target: set) -> float:
            matched, hit = 0, set()
            for q in query:
                if q in target:
                    matched += 1
                    hit.add(q)
                    continue
                for t in target:
                    if q in t:
                        matched += 1
                        hit.add(t)
                        break
            jaccard = len(hit) / len(query | target) if query and target else 0.0
            return round(0.6 * (matched / len(query)) + 0.4 * jaccard, 3)

        targets = [{'1girl', 'smile', 'school uniform'}, {'1boy', 'smile'}, {'1girl'}, set()]
        for prompt in ("1girl, smile", "smile", "boy", "girl, 1girl", "uniform, nude", "1girl smile school_uniform"):
            for target in targets:
                with self.subTest(prompt=prompt, target=sorted(target)):
                    got = EventDataLoader._unit_similarity(EventDataLoader._query_units(prompt), target)[1]
                    self.assertEqual(got, legacy(EventDataLoader._parse_tags(prompt), target))


class EventShardManifestTests(unittest.TestCase):
    def test_shards_that_disagree_with_the_manifest_are_reported(self):
        import json

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            shard_dir = root / "danbooru_sorted"
            shard_dir.mkdir()
            shard = shard_dir / "danbooru_g.parquet"
            pd.DataFrame([_event_row(10), _event_row(11, parent_id=10)]).to_parquet(shard, index=False)
            actual_size = shard.stat().st_size
            manifest = {
                "format_version": 1,
                "artifacts": [
                    {"path": "danbooru_sorted/danbooru_g.parquet", "size_bytes": actual_size + 1,
                     "kind": "event_graph"},
                ],
            }
            (root / "dataset_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

            loader = EventDataLoader(str(shard_dir))
            with redirect_stdout(StringIO()):
                loader.load_parquets_by_rating(["g"])
            self.assertEqual(loader.stale_shards, ["danbooru_sorted/danbooru_g.parquet"])

            manifest["artifacts"][0]["size_bytes"] = actual_size
            (root / "dataset_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
            fresh = EventDataLoader(str(shard_dir))
            with redirect_stdout(StringIO()):
                fresh.load_parquets_by_rating(["g"])
            self.assertEqual(fresh.stale_shards, [])


if __name__ == "__main__":
    unittest.main()
