"""인스턴트 와일드카드($$name$$) — 가중치 파서, 멱등 훅 등록, 부팅 직후 치환 회귀."""
import json
import random
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from core import instant_wildcards as iw_mod
from core import prompt_pipeline as pp_mod
from core.instant_wildcards import (
    HOOK_NAME,
    InstantWildcards,
    ensure_hook_registered,
    parse_weighted_line,
)
from core.prompt_pipeline import HookPoint, PromptContext, PromptPipeline


class ParseWeightedLineGoldenTests(unittest.TestCase):
    def test_golden_values(self):
        self.assertEqual(parse_weighted_line("{100}:tag"), (100.0, "tag"))
        self.assertEqual(parse_weighted_line("{0}:x"), (1.0, "x"))      # 0 이하 → 기본 1
        self.assertEqual(parse_weighted_line("tag"), (1.0, "tag"))
        self.assertEqual(parse_weighted_line("{0.5}:a"), (0.5, "a"))
        self.assertEqual(parse_weighted_line("  {2}: spaced  "), (2.0, "spaced"))
        self.assertEqual(parse_weighted_line("{x}:not weight"), (1.0, "{x}:not weight"))


def _store(tmp: str, data: dict) -> Path:
    path = Path(tmp) / "instant_wildcards.json"
    path.write_text(json.dumps({"version": 1, "wildcards": data}), encoding="utf-8")
    return path


class InstantWildcardHookTests(unittest.TestCase):
    def test_pick_uses_weights_and_unknown_name_stays_literal(self):
        with tempfile.TemporaryDirectory() as tmp:
            iw = InstantWildcards(store_path=_store(tmp, {"mood": ["{0.0001}:sad", "{1000}:happy"]}),
                                  rng=random.Random(1))
            self.assertEqual(iw.resolve("a, $$mood$$"), "a, happy")
            self.assertEqual(iw.resolve("$$missing$$"), "$$missing$$")

    def test_hook_leaves_prompts_without_marker_untouched(self):
        iw = InstantWildcards()
        ctx = PromptContext(main_tags=["(a, b:1.2)", "c"])
        iw.make_hook()(ctx)
        self.assertEqual(ctx.main_tags, ["(a, b:1.2)", "c"])
        self.assertEqual(ctx.metadata, {})

    def test_ensure_hook_registered_is_idempotent_by_name(self):
        pl = PromptPipeline()
        iw = InstantWildcards()
        self.assertTrue(ensure_hook_registered(pl, iw))
        self.assertFalse(ensure_hook_registered(pl, iw))
        self.assertEqual(pl.hook_count(HookPoint.POST_PROCESSING), 1)
        self.assertTrue(pl.has_hook(HookPoint.POST_PROCESSING, HOOK_NAME))

    def test_removed_history_api_is_gone(self):
        iw = InstantWildcards()
        for name in ("history", "clear_history", "pin", "unpin", "is_pinned",
                     "set_override", "clear_override", "add_line", "remove_line", "rename"):
            self.assertFalse(hasattr(iw, name), name)


class BootRegistrationTests(unittest.TestCase):
    """register_standard_hooks() 한 번이면 '즉석 WC' 창을 열지 않아도 $$name$$ 이 풀린다."""

    def test_standard_hooks_resolve_instant_wildcards_right_after_boot(self):
        from core.standard_hooks import register_standard_hooks, run_pipeline_on_text

        with tempfile.TemporaryDirectory() as tmp:
            store = _store(tmp, {"mood": ["happy"]})
            pl = PromptPipeline()
            iw = InstantWildcards(store_path=store)
            with mock.patch.object(pp_mod, "_pipeline_instance", pl), \
                 mock.patch.object(iw_mod, "_instance", iw):
                self.assertEqual(register_standard_hooks(), 2)   # dedupe + instant
                self.assertEqual(register_standard_hooks(), 0)   # 멱등
                self.assertEqual(run_pipeline_on_text("a, $$mood$$"), "a, happy")
                # 매니저 액션 경로가 다시 불러도 훅이 두 번 걸리지 않는다
                self.assertFalse(ensure_hook_registered())
                self.assertEqual(pl.hook_count(HookPoint.POST_PROCESSING), 1)

    def test_store_failure_does_not_block_other_standard_hooks(self):
        from core.standard_hooks import register_standard_hooks

        pl = PromptPipeline()
        with mock.patch.object(iw_mod, "_instance", None), \
             mock.patch.object(iw_mod, "default_store_path", side_effect=OSError("no storage")):
            self.assertEqual(register_standard_hooks(pl), 1)
        self.assertEqual(pl.hook_count(HookPoint.FINAL), 1)
        self.assertEqual(pl.hook_count(HookPoint.POST_PROCESSING), 0)

    def test_manager_action_and_generation_share_one_instance(self):
        with mock.patch.object(iw_mod, "_instance", None), \
             mock.patch.object(iw_mod, "default_store_path", return_value=None):
            first = iw_mod.get_instant_wildcards()
            self.assertIs(first, iw_mod.get_instant_wildcards())


if __name__ == "__main__":
    unittest.main()
