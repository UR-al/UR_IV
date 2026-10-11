"""연관 태그 순수 로직 — 프롬프트 태그 추출 · 기준 태그 고르기 · 함의 · 순위 · 페이로드 검사(core/related_tags.py)."""
from __future__ import annotations

import unittest

from core import related_tags as rt


class ExtractPromptTagsTests(unittest.TestCase):
    def names(self, text):
        return [name for name, _w in rt.extract_prompt_tags(text)]

    def test_splits_normalizes_and_keeps_order(self):
        self.assertEqual(self.names("1girl, Long Hair\nblue_eyes,  holding   sword "),
                         ["1girl", "long_hair", "blue_eyes", "holding_sword"])

    def test_drops_lora_wildcards_alternatives_and_keywords(self):
        text = "<lora:style:0.8>, __hair_color__, {red|blue} dress, armor BREAK sword AND shield, <embedding>"
        self.assertEqual(self.names(text), ["dress", "armor", "sword", "shield"])

    def test_attention_and_weights(self):
        tags = rt.extract_prompt_tags("(red hair:1.2), ((smile)), [blush], (sword:-1.5), (armor, cape:0.9)")
        self.assertEqual(tags, [("red_hair", 1.2), ("smile", 1.0), ("blush", 1.0), ("sword", -1.5),
                                ("armor", 1.0), ("cape", 0.9)])

    def test_escaped_parentheses_stay_literal_and_emoticons_survive(self):
        self.assertEqual(self.names(r"nagato \(kancolle\), :3, :d, ^_^, :o"),
                         ["nagato_(kancolle)", ":3", ":d", "^_^", ":o"])
        # 이스케이프한 괄호 안의 :숫자 는 가중치가 아니라 글자다
        self.assertEqual(rt.extract_prompt_tags(r"\(foo:2\)"), [("(foo:2)", 1.0)])

    def test_qualifier_parentheses_inside_a_tag_are_part_of_the_tag(self):
        # 자동완성·연관 태그는 'ganyu (genshin impact)' · 'bow_(weapon)' 처럼 이스케이프 없이 넣는다 — 괄호를 지우면
        # 캐릭터·작품 기준이 사라지고, 이미 넣은 괄호 태그가 계속 추천됐다
        for text in ("ganyu (genshin impact)", "ganyu_(genshin_impact)", r"ganyu \(genshin impact\)"):
            with self.subTest(text=text):
                self.assertEqual(self.names(text), ["ganyu_(genshin_impact)"])
        self.assertEqual(rt.extract_prompt_tags("(ganyu (genshin impact):1.2)"), [("ganyu_(genshin_impact)", 1.2)])
        self.assertEqual(self.names("1girl, archery, arrow (projectile), ((bow (weapon))), >:("),
                         ["1girl", "archery", "arrow_(projectile)", "bow_(weapon)", ">:("])
        # 조각 전체를 감싼 괄호만 가중치·강조 문법이다
        self.assertEqual(rt.extract_prompt_tags("((smile)), [blush], {hat}, (armor, cape:0.9)"),
                         [("smile", 1.0), ("blush", 1.0), ("hat", 1.0), ("armor", 1.0), ("cape", 0.9)])

    def test_prompt_editing_body_survives_and_bad_input_is_empty(self):
        self.assertEqual(self.names("[cat:dog:0.5], [a|b]"), ["cat:dog", "a", "b"])
        self.assertEqual(rt.extract_prompt_tags(""), [])
        self.assertEqual(rt.extract_prompt_tags(None), [])
        self.assertEqual(rt.extract_prompt_tags(" , ,\n"), [])

    def test_prompt_comments_are_not_tags(self):
        # 주석(// · /* */)은 생성에 보내지 않는다 — 기준 태그도 '이미 있는 태그'도 아니다(core/prompt_comments)
        self.assertEqual(self.names("armor, // sword, shield\ncape, /* hat, scarf */ boots"),
                         ["armor", "cape", "boots"])
        # 주석이 아닌 슬래시(앞이 공백·쉼표·줄 처음이 아님)는 그대로 태그 글자다
        self.assertEqual(self.names("a//b, cape"), ["a//b", "cape"])

    def test_person_count_tags(self):
        for name in ("1girl", "2girls", "6+girls", "1boy", "3others", "multiple_girls", "solo", "solo_focus",
                     "male_focus", "no_humans", "Multiple Boys"):
            self.assertTrue(rt.is_person_count_tag(name), name)
        for name in ("girl", "1girl_(cosplay)", "boy_on_top", "solo_leveling", "armor"):
            self.assertFalse(rt.is_person_count_tag(name), name)

    def test_resolve_through_aliases(self):
        index = {"long_hair": 0, "hat": 1}
        aliases = {"longhair": 0}
        self.assertEqual(rt.resolve("Long Hair", index, aliases), 0)
        self.assertEqual(rt.resolve("longhair", index, aliases), 0)
        self.assertIsNone(rt.resolve("cape", index, aliases))


class ContextTests(unittest.TestCase):
    OCC = [900, 500, 100, 50, 10, 0, 30]

    def test_generic_tags_drop_only_when_something_specific_remains(self):
        self.assertEqual(rt.select_context([0, 2, 3], self.OCC, 1000), [2, 3])
        self.assertEqual(rt.select_context([0, 1], self.OCC, 1000), [0, 1])

    def test_dedupes_skips_unknown_and_keeps_the_rarest_in_prompt_order(self):
        ids = [3, 2, 3, 5, 4, 6, 99, -1]
        self.assertEqual(rt.select_context(ids, self.OCC, 1000), [3, 2, 4, 6])
        self.assertEqual(rt.select_context(ids, self.OCC, 1000, limit=2), [4, 6])
        self.assertEqual(rt.select_context([], self.OCC, 1000), [])
        self.assertEqual(rt.select_context([2], self.OCC, 0), [])

    def test_implied_closure_is_transitive_and_cycle_safe(self):
        implications = {1: [2], 2: [3], 3: [1], 7: [8]}
        self.assertEqual(rt.implied_closure([1], implications), {1, 2, 3})
        self.assertEqual(rt.implied_closure([7, 9], implications), {8})


class RankTests(unittest.TestCase):
    NAMES = ["armor", "sword", "weapon", "cape", "solo", "smile", "zeta"]
    OCC = [200, 300, 300, 100, 900, 600, 100]
    TABLE = {
        0: {1: 150, 2: 150, 3: 60, 4: 190, 5: 20, 6: 60},
        3: {1: 40, 0: 60, 5: 90},
    }

    def neighbors(self, tag_id):
        return self.TABLE.get(tag_id, {})

    def rank(self, context, **kw):
        return rt.rank_related(context, self.neighbors, self.OCC, 1000, self.NAMES, **kw)

    def test_coverage_then_added_value_then_count_then_name(self):
        result = self.rank([0, 3])
        # sword: 기준 둘 다(0.75−0.3 + 0.4−0.3). 나머지는 하나씩 — 더한 값 순 weapon .45 · smile .3(armor 쪽은
        # 0.1−0.6<0 이라 cape 에서만) · zeta .2 · solo .05. 기준 태그(cape)는 후보가 아니다.
        self.assertEqual([self.NAMES[r.id] for r in result], ["sword", "weapon", "smile", "zeta", "solo"])
        self.assertEqual(result[0].coverage, 2)
        self.assertEqual(result[0].via, (0, 3))
        self.assertAlmostEqual(result[0].score, (150 / 200 - 0.3) + (40 / 100 - 0.3))

    def test_min_pair_blocked_allowed_and_limit(self):
        result = self.rank([0], min_pair=100, blocked={1}, allowed=lambda g: g != 4, limit=5)
        self.assertEqual([self.NAMES[r.id] for r in result], ["weapon"])
        self.assertEqual(self.rank([0], limit=0), [])
        self.assertEqual(rt.rank_related([0], self.neighbors, self.OCC, 0, self.NAMES), [])

    def test_ties_sort_by_count_then_name(self):
        # 2진 분수라 더한 값이 정확히 같다: 64/128 − 128/1024 = 96/128 − 384/1024 = 0.375
        table = {0: {1: 64, 2: 96, 6: 64}}
        occ = [128, 128, 384, 0, 0, 0, 128]
        result = rt.rank_related([0], lambda c: table.get(c, {}), occ, 1024, self.NAMES)
        self.assertEqual([self.NAMES[r.id] for r in result], ["weapon", "sword", "zeta"])
        self.assertEqual(result[0].score, result[1].score)

    def test_context_tags_are_never_suggested(self):
        result = self.rank([0, 3])
        self.assertFalse({r.id for r in result} & {0, 3})


class PayloadTests(unittest.TestCase):
    FIELDS = {"main": "armor", "character": "", "copyright": "", "count": "1girl", "prefix": "", "suffix": ""}

    def test_query_payload_is_validated_and_clamped(self):
        req = rt.parse_query_payload({"requestId": "rtq-1", "fields": self.FIELDS, "negative": "nsfw",
                                      "exclude": "_armor", "limit": 999})
        self.assertEqual(req.request_id, "rtq-1")
        self.assertEqual(dict(req.fields)["main"], "armor")
        self.assertEqual([k for k, _ in req.fields], list(rt.QUERY_FIELDS))
        self.assertEqual(req.limit, rt.MAX_LIMIT)
        self.assertEqual(rt.parse_query_payload({"requestId": "a", "fields": self.FIELDS}).limit, rt.DEFAULT_LIMIT)
        self.assertEqual(rt.parse_query_payload({"requestId": "a", "fields": self.FIELDS, "limit": True}).limit,
                         rt.DEFAULT_LIMIT)
        self.assertEqual(rt.parse_query_payload({"requestId": "a", "fields": self.FIELDS, "limit": 0}).limit, 1)

    def test_bad_query_payloads_raise(self):
        bad = [None, {}, {"requestId": "../x", "fields": self.FIELDS}, {"requestId": "a"},
               {"requestId": "a", "fields": {**self.FIELDS, "main": 3}},
               {"requestId": "a", "fields": self.FIELDS, "negative": None},
               {"requestId": "a" * 121, "fields": self.FIELDS}]
        for payload in bad:
            with self.assertRaises(ValueError, msg=repr(payload)):
                rt.parse_query_payload(payload)

    def test_build_and_cancel_payloads(self):
        self.assertEqual(rt.parse_build_payload({"requestId": "b1", "shards": ["s", "G", "s"]}).shards, ("g", "s"))
        for shards in ([], ["x"], "gs", None, [1]):
            with self.assertRaises(ValueError):
                rt.parse_build_payload({"requestId": "b1", "shards": shards})
        self.assertEqual(rt.parse_cancel_payload({"requestId": "b1"}), "b1")
        with self.assertRaises(ValueError):
            rt.parse_cancel_payload({})

    def test_rating_shards_follow_the_filter_with_a_gs_default(self):
        self.assertEqual(rt.rating_shards({"e", "g"}), ("g", "e"))
        self.assertEqual(rt.rating_shards(["Q"]), ("q",))
        self.assertEqual(rt.rating_shards(set()), ("g", "s"))
        self.assertEqual(rt.rating_shards(None), ("g", "s"))
        self.assertEqual(rt.rating_shards({"x"}), ("g", "s"))


if __name__ == "__main__":
    unittest.main()
