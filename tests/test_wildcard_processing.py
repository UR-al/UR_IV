"""utils.wildcard.process_wildcards — {A|B} · 범위 · 가중치 회귀(유일한 외부 진입점)."""
import random
import unittest
from unittest import mock

import utils.wildcard as wildcard_mod
from utils.wildcard import process_wildcards


class ProcessWildcardsTests(unittest.TestCase):
    def setUp(self):
        random.seed(1234)

    def test_plain_text_and_empty_are_unchanged(self):
        self.assertEqual(process_wildcards(""), "")
        self.assertEqual(process_wildcards("1girl, solo"), "1girl, solo")

    def test_range_and_step(self):
        for _ in range(50):
            self.assertIn(process_wildcards("{1-3}"), {"1", "2", "3"})
            self.assertIn(process_wildcards("{0-10:5}"), {"0", "5", "10"})
            self.assertIn(process_wildcards("{5-1}"), {"1", "2", "3", "4", "5"})

    def test_zero_step_does_not_raise(self):
        for _ in range(20):
            self.assertIn(process_wildcards("{1-3:0}"), {"1", "2", "3"})

    def test_nested_options_resolve_inside_out(self):
        for _ in range(50):
            self.assertIn(process_wildcards("{{red|blue} hair|ponytail}"),
                          {"red hair", "blue hair", "ponytail"})

    def test_zero_total_weight_falls_back_to_uniform(self):
        # 예전: random.choices 가 ValueError → 생성 준비가 통째로 멈췄다
        seen = {process_wildcards("{a:0|b:0}") for _ in range(100)}
        self.assertEqual(seen, {"a", "b"})

    def test_negative_weight_is_clipped_to_zero(self):
        seen = {process_wildcards("{a:-1|b:1}") for _ in range(100)}
        self.assertEqual(seen, {"b"})

    def test_weight_suffix_must_be_integer(self):
        # 'artist:foo' 의 콜론은 가중치가 아니다 — 옵션 원문 그대로
        seen = {process_wildcards("{artist:foo|artist:bar}") for _ in range(100)}
        self.assertEqual(seen, {"artist:foo", "artist:bar"})

    def test_weights_bias_selection(self):
        with mock.patch.object(wildcard_mod.random, "choices", wraps=random.choices) as choices:
            process_wildcards("{a:3|b:1}")
        _, kwargs = choices.call_args
        self.assertEqual(kwargs["weights"], [3, 1])

    def test_only_entry_point_is_exported(self):
        for dead in ("expand_wildcards", "count_wildcard_combinations"):
            self.assertFalse(hasattr(wildcard_mod, dead), dead)
        processor = wildcard_mod.get_wildcard_processor()
        for dead in ("expand_all", "count_combinations", "validate", "get_history", "history"):
            self.assertFalse(hasattr(processor, dead), dead)


if __name__ == "__main__":
    unittest.main()
