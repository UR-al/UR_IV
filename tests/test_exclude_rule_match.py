"""core.exclude_rule_match — 제외 규칙 미리보기 매칭 (Qt 없이)."""
from __future__ import annotations

import unittest

from core.exclude_rule_match import match_exclude_rule, normalize_exclude_rule

VOCAB = frozenset({
    "long_hair", "short_hair", "hair_ornament", "hairband", "blue_eyes",
    "solo", "solo_focus", "1girl", "chair",
})


class _NoIterSet(frozenset):
    """완전 일치 규칙이 사전을 훑지 않는지 확인하는 집합 (반복하면 실패)."""

    def __iter__(self):
        raise AssertionError("exact rules must not scan the vocabulary")


class ExcludeRuleMatchTests(unittest.TestCase):
    def test_rule_normalisation(self):
        self.assertEqual(normalize_exclude_rule("  Long Hair "), "long_hair")
        self.assertEqual(normalize_exclude_rule(None), "")

    def test_each_rule_syntax(self):
        self.assertEqual(match_exclude_rule("*solo", VOCAB), ["solo"])
        self.assertEqual(match_exclude_rule("*Long Hair", VOCAB), ["long_hair"])
        self.assertEqual(match_exclude_rule("*missing", VOCAB), [])
        # 접미 규칙은 실제 제외 로직(generator_prompts)과 같이 단순 endswith 다 — chair 도 걸린다
        self.assertEqual(match_exclude_rule("_hair", VOCAB), ["chair", "long_hair", "short_hair"])
        self.assertEqual(match_exclude_rule("hair_", VOCAB), ["hair_ornament", "hairband"])
        self.assertEqual(
            match_exclude_rule("_hair_", VOCAB),
            ["chair", "hair_ornament", "hairband", "long_hair", "short_hair"],
        )
        self.assertEqual(
            match_exclude_rule("hair", VOCAB),
            ["chair", "hair_ornament", "hairband", "long_hair", "short_hair"],
        )

    def test_keep_rules_and_blanks_match_nothing(self):
        self.assertEqual(match_exclude_rule("~solo", VOCAB), [])
        self.assertEqual(match_exclude_rule("  ~_hair", VOCAB), [])
        self.assertEqual(match_exclude_rule("   ", VOCAB), [])
        self.assertEqual(match_exclude_rule(None, VOCAB), [])

    def test_exact_rule_is_a_set_lookup_not_a_scan(self):
        vocab = _NoIterSet(VOCAB)
        self.assertEqual(match_exclude_rule("*solo_focus", vocab), ["solo_focus"])
        self.assertEqual(match_exclude_rule("*nope", vocab), [])


if __name__ == "__main__":
    unittest.main()
