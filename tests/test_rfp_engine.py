"""core.rfp_engine — 생성 때마다 도는 FINAL 단계 중복 정리(standard_dedupe)의 순수 로직.

호출자 없던 조건부 명령 DSL·filter_contains·split/join_tags 를 지운 뒤(P13c) 남은 유일한 경로라,
dedupe·remove_collisions·make_dedupe_hook 의 동작을 여기서 고정한다.
"""
from __future__ import annotations

import unittest

from core import rfp_engine
from core.prompt_pipeline import HookPoint, PromptContext, PromptPipeline
from core.rfp_engine import dedupe, make_dedupe_hook, remove_collisions


class DedupeTests(unittest.TestCase):
    def test_keeps_first_occurrence_order(self):
        self.assertEqual(dedupe(["b", "a", "b", "c", "a"]), ["b", "a", "c"])

    def test_is_exact_match_only(self):
        # 대소문자·공백/밑줄 표기는 다른 태그로 본다(정규화는 앞 단계의 몫)
        self.assertEqual(dedupe(["Smile", "smile", "long_hair", "long hair"]),
                         ["Smile", "smile", "long_hair", "long hair"])

    def test_does_not_mutate_input(self):
        tags = ["a", "a"]
        dedupe(tags)
        self.assertEqual(tags, ["a", "a"])


class RemoveCollisionsTests(unittest.TestCase):
    def test_drops_main_tags_already_in_prefix_or_postfix(self):
        self.assertEqual(
            remove_collisions(["1girl", "smile", "masterpiece", "night"],
                              ["masterpiece"], ["night"]),
            ["1girl", "smile"],
        )

    def test_ignores_weight_brackets_on_the_blocking_side(self):
        self.assertEqual(
            remove_collisions(["best quality", "smile"], ["{best quality}"], ["[smile]"]),
            [],
        )

    def test_empty_prefix_postfix_keeps_everything(self):
        self.assertEqual(remove_collisions(["a", "b"], [], []), ["a", "b"])


class DedupeHookTests(unittest.TestCase):
    def test_hook_dedupes_then_removes_collisions_in_place(self):
        ctx = PromptContext(prefix_tags=["masterpiece"], main_tags=["smile", "masterpiece", "smile", "1girl"],
                            postfix_tags=["highres"])
        make_dedupe_hook()(ctx)
        self.assertEqual(ctx.main_tags, ["smile", "1girl"])
        self.assertEqual(ctx.prefix_tags, ["masterpiece"])
        self.assertEqual(ctx.postfix_tags, ["highres"])

    def test_standard_hooks_register_it_once_at_final(self):
        from core.standard_hooks import DEDUPE_HOOK_NAME, register_standard_hooks

        class _NoInstantWildcards:
            @staticmethod
            def make_hook():
                return lambda _ctx: None

        pipeline = PromptPipeline()
        self.assertEqual(register_standard_hooks(pipeline, instant_wildcards=_NoInstantWildcards()), 2)
        self.assertEqual(register_standard_hooks(pipeline, instant_wildcards=_NoInstantWildcards()), 0)
        self.assertTrue(pipeline.has_hook(HookPoint.FINAL, DEDUPE_HOOK_NAME))

        ctx = PromptContext(main_tags=["a", "b", "a"])
        pipeline.execute(HookPoint.FINAL, ctx)
        self.assertEqual(ctx.main_tags, ["a", "b"])

    def test_dead_conditional_dsl_stays_removed(self):
        for name in ("split_tags", "join_tags", "filter_contains", "parse_conditional_command",
                     "evaluate_condition", "execute_command", "_insert_after_keyword", "_eval_atom"):
            self.assertFalse(hasattr(rfp_engine, name), name)


if __name__ == "__main__":
    unittest.main()
