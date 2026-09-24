"""조건부 v2 — 다중조건(AND) + after_condition 삽입 순수 로직 회귀."""
import unittest
from utils.condition_block import (
    ConditionRule, apply_prompt_rules, apply_rules, condition_terms, insert_after,
    multi_cond_met, norm_tag, remove_from_tags, replace_across, replace_in_tags,
    split_tags, split_target,
)


class TestMultiCondMet(unittest.TestCase):
    def test_single(self):
        self.assertTrue(multi_cond_met("fiery horns", {"fiery horns"}, True))
        self.assertFalse(multi_cond_met("fiery horns", {"other"}, True))

    def test_and_all_present(self):
        tags = {"tamamura gunzo", "fiery horns", "1boy"}
        self.assertTrue(multi_cond_met("tamamura gunzo, fiery horns", tags, True))

    def test_and_one_missing(self):
        tags = {"tamamura gunzo", "1boy"}
        self.assertFalse(multi_cond_met("tamamura gunzo, fiery horns", tags, True))

    def test_underscore_input_normalized(self):
        # 입력이 언더스코어여도 정규화돼 매칭
        self.assertTrue(multi_cond_met("tamamura_gunzo", {"tamamura gunzo"}, True))

    def test_not_exists(self):
        # exists=False → AND 조건이 충족 안 될 때 True
        self.assertTrue(multi_cond_met("x, y", {"x"}, False))        # 하나 빠짐
        self.assertFalse(multi_cond_met("x, y", {"x", "y"}, False))  # 다 있음

    def test_empty_condition(self):
        self.assertFalse(multi_cond_met("", {"a"}, True))


class TestInsertAfter(unittest.TestCase):
    def test_insert_after_anchor(self):
        tags = ["tamamura gunzo", "1boy", "fiery horns"]
        out = insert_after(tags, "tamamura gunzo", "wakan tanka")
        self.assertEqual(out, ["tamamura gunzo", "wakan tanka", "1boy", "fiery horns"])

    def test_already_present_no_change(self):
        tags = ["tamamura gunzo", "wakan tanka"]
        self.assertEqual(insert_after(tags, "tamamura gunzo", "wakan tanka"), tags)

    def test_anchor_missing_returns_none(self):
        self.assertIsNone(insert_after(["1boy", "smile"], "tamamura gunzo", "wakan tanka"))

    def test_underscore_data_matches(self):
        # 데이터 태그가 언더스코어여도 anchor(공백 정규화)와 매칭
        tags = ["tamamura_gunzo", "1boy"]
        out = insert_after(tags, "tamamura gunzo", "wakan tanka")
        self.assertEqual(out, ["tamamura_gunzo", "wakan tanka", "1boy"])

    def test_norm_tag(self):
        self.assertEqual(norm_tag(" Tamamura_Gunzo "), "tamamura gunzo")


class TestTagSplitting(unittest.TestCase):
    def test_top_level_commas_only(self):
        self.assertEqual(split_tags("a, (b, c:1.2), <lora:x:1>, d"),
                         ["a", "(b, c:1.2)", "<lora:x:1>", "d"])

    def test_escaped_parens_do_not_open_groups(self):
        self.assertEqual(split_tags(r"hatsune miku \(append\), smile"),
                         [r"hatsune miku \(append\)", "smile"])

    def test_norm_tag_unescapes_parens(self):
        self.assertEqual(norm_tag(r"Hatsune_Miku \(Append\)"), "hatsune miku (append)")
        self.assertEqual(condition_terms("a, A, b_c"), ["a", "b c"])
        self.assertEqual(split_target("x, X, y"), ["x", "y"])
        self.assertEqual(split_target(["x", "y, z"]), ["x", "y", "z"])

    def test_condition_matches_escaped_character_tags(self):
        tags = {norm_tag(t) for t in split_tags(r"hatsune miku \(append\), 1girl")}
        self.assertTrue(multi_cond_met("hatsune miku (append)", tags, True))


class TestTagLevelReplace(unittest.TestCase):
    def test_muscular_family_is_not_mangled(self):
        # 사용자 규칙 muscular -> muscular male (config/cond_rules.json 의 활성 규칙과 같은 모양)
        tags = split_tags("1boy, muscular, muscular male, muscular female, muscular arms, abs")
        self.assertEqual(replace_in_tags(tags, "muscular", "muscular male"),
                         ["1boy", "muscular male", "muscular female", "muscular arms", "abs"])

    def test_anchor_replaced_in_place_when_target_missing(self):
        self.assertEqual(replace_in_tags(["1boy", "muscular", "abs"], "muscular", "muscular male"),
                         ["1boy", "muscular male", "abs"])

    def test_replace_is_idempotent(self):
        once = replace_in_tags(["a", "muscular", "b"], "muscular", "muscular male")
        self.assertEqual(replace_in_tags(once, "muscular", "muscular male"), once)
        keep = replace_in_tags(["a", "muscular", "b"], "muscular", "muscular, abs")
        self.assertEqual(keep, ["a", "muscular", "abs", "b"])
        self.assertEqual(replace_in_tags(keep, "muscular", "muscular, abs"), keep)

    def test_multi_condition_anchors_collapse_to_one_target(self):
        self.assertEqual(replace_in_tags(["x", "cat ears", "y", "cat tail"], "cat ears, cat tail", "catgirl"),
                         ["x", "catgirl", "y"])

    def test_replace_across_fields_puts_target_once(self):
        fields = [["muscular"], ["a", "muscular"], ["muscular male"]]
        self.assertEqual(replace_across(fields, "muscular", "muscular male"),
                         [[], ["a"], ["muscular male"]])

    def test_escaped_anchor_is_found(self):
        self.assertEqual(replace_in_tags([r"hatsune miku \(append\)"], "hatsune miku (append)", "miku"),
                         ["miku"])

    def test_remove_accepts_comma_targets(self):
        self.assertEqual(remove_from_tags(["a", "b", "c", "d_e"], "b, c, d e"), ["a"])
        self.assertEqual(remove_from_tags(["a", "B"], "b"), ["a"])

    def test_insert_after_multi_target(self):
        self.assertEqual(insert_after(["a", "x", "b"], "x", "y, b, z"), ["a", "x", "y", "z", "b"])


def _rule(cond, target, action="add", location="main", exists=True):
    return {"enabled": True, "condition": cond, "exists": exists, "target": target,
            "action": action, "location": location}


class TestGlobalRuleEngine(unittest.TestCase):
    def test_user_rule_file_shape_is_read_as_is(self):
        fields = {"main": ["1boy", "muscular", "muscular male", "armpits"]}
        out = apply_prompt_rules(fields, [
            _rule("muscular", "muscular male", action="replace"),
            _rule("armpits", "armpit hair"),
        ], [])
        self.assertEqual(out["main"], ["1boy", "muscular male", "armpits", "armpit hair"])
        self.assertEqual(fields["main"], ["1boy", "muscular", "muscular male", "armpits"])  # 입력 불변

    def test_two_passes_equal_one_pass(self):
        rules = [
            _rule("muscular", "muscular male", action="replace"),
            _rule("shota", "aged down"),
            _rule("aged down", "shota"),
            _rule("beard", "beard", action="remove"),
            _rule("smile", "open mouth", location="after_condition"),
            _rule("hat", "bare head", exists=False, location="suffix"),
        ]
        neg = [_rule("muscular male", "bad anatomy, extra arms")]
        fields = {"character": ["hatsune miku"], "main": ["muscular", "shota", "smile", "beard"],
                  "suffix": ["best quality"], "neg": ["bad anatomy"]}
        once = apply_prompt_rules(fields, rules, neg)
        self.assertEqual(once["main"], ["muscular male", "shota", "smile", "open mouth", "aged down"])
        self.assertEqual(once["suffix"], ["best quality", "bare head"])
        self.assertEqual(once["neg"], ["bad anatomy"])   # 조건은 1차 적용 전 태그 기준
        twice = apply_prompt_rules(once, rules, neg)
        self.assertEqual(twice["main"], once["main"])
        self.assertEqual(twice["suffix"], once["suffix"])
        self.assertEqual(twice["neg"], ["bad anatomy", "extra arms"])   # 2차에서야 muscular male 이 보인다
        self.assertEqual(apply_prompt_rules(twice, rules, neg), twice)

    def test_remove_with_comma_target_and_all_positive_fields(self):
        out = apply_prompt_rules({"prefix": ["b"], "main": ["a", "b", "c"], "suffix": ["c"]},
                                 [_rule("a", "b, c", action="remove")], [])
        self.assertEqual((out["prefix"], out["main"], out["suffix"]), ([], ["a"], []))

    def test_add_skips_tags_already_in_another_field(self):
        out = apply_prompt_rules({"main": ["cat ears"], "suffix": ["animal ear fluff"]},
                                 [_rule("cat ears", "animal ear fluff, tail")], [])
        self.assertEqual(out["main"], ["cat ears", "tail"])

    def test_after_condition_without_anchor_appends_to_main(self):
        out = apply_prompt_rules({"main": ["a"]},
                                 [_rule("x", "y", location="after_condition", exists=False)], [])
        self.assertEqual(out["main"], ["a", "y"])

    def test_negative_field_is_not_part_of_the_condition(self):
        out = apply_prompt_rules({"main": ["a"], "neg": ["nsfw"]}, [_rule("nsfw", "safe")], [])
        self.assertEqual(out["main"], ["a"])

    def test_replace_when_absent_rule_does_nothing(self):
        out = apply_prompt_rules({"main": ["x"]}, [_rule("x, y", "z", action="replace", exists=False)], [])
        self.assertEqual(out["main"], ["x"])

    def test_invalid_rules_are_skipped(self):
        out = apply_prompt_rules({"main": ["a"]}, [None, {"condition": "a"}, _rule("", "b")], [])
        self.assertEqual(out["main"], ["a"])


class TestCharacterRuleEngineAlignment(unittest.TestCase):
    def _r(self, cond, tags, action="add", location="main", exists=True):
        return ConditionRule(condition_tag=cond, condition_exists=exists, target_tags=tags,
                             location=location, action=action)

    def test_comma_condition_is_and(self):
        rules = [self._r("cat ears, tail", ["catgirl"])]
        self.assertEqual(apply_rules(rules, {"cat ears", "tail"})["main"], ["catgirl"])
        self.assertEqual(apply_rules(rules, {"cat ears"})["main"], [])
        not_rule = [self._r("cat ears, tail", ["plain"], exists=False)]
        self.assertEqual(apply_rules(not_rule, {"cat ears", "tail"})["main"], [])
        self.assertEqual(apply_rules(not_rule, {"cat ears"})["main"], ["plain"])

    def test_neg_add_dedupes_against_negative_field(self):
        rules = [self._r("x", ["bad hands"], location="neg")]
        res = apply_rules(rules, {"x"}, current_by_location={"neg": ["bad_hands"]})
        self.assertEqual(res["neg"], [])
        res = apply_rules(rules, {"x", "bad hands"}, current_by_location={"neg": []})
        self.assertEqual(res["neg"], ["bad hands"])

    def test_replace_goes_to_its_bucket(self):
        res = apply_rules([self._r("x", ["y", "z"], action="replace"),
                           self._r("x", ["w"], action="replace", location="neg"),
                           self._r("x", ["v"], action="replace", exists=False)], {"x"})
        self.assertEqual(res["_replace"], [("x", "y"), ("x", "z")])
        self.assertEqual(res["_replace_neg"], [("x", "w")])


if __name__ == "__main__":
    unittest.main()
