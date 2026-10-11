"""프롬프트 주석이 사용자 글자가 태그 로직에 들어가는 자리마다 안전한지 — 섹션 합본·실시간 정리·조건부 규칙·
최적화·분류·LoRA 스택·와일드카드·Refine·SAM3·I2I/인페인트 폴백·Tile & Repair (core/prompt_comments).

공통 실패 모양: 줄 주석(``a // 메모``) 뒤에 쉼표로 무언가를 이으면 그 글자가 주석 안으로 들어가 조용히 사라진다.
그래서 이 자리들은 주석을 먼저 떼거나(보내는 글자), 주석을 앞 태그에 붙인 채로 다루고 줄바꿈으로 다시 잇는다(편집 칸).
"""
from __future__ import annotations

import base64
import io
import json
import os
import random
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from PIL import Image

from core.prompt_comments import has_prompt_comment, strip_prompt_comments
from core.prompt_order import compose_prompt


def _sent(text: str) -> str:
    return strip_prompt_comments(text)


class ComposePromptTests(unittest.TestCase):
    ORDER = ["char_count", "character", "prefix", "main", "suffix"]

    def test_a_trailing_line_comment_never_comments_out_the_next_section(self):
        total = compose_prompt({"prefix": "masterpiece // 품질", "main": "1girl", "suffix": "// 메모만"}, self.ORDER)
        self.assertEqual(total, "masterpiece, 1girl")

    def test_line_breaks_inside_a_section_stay(self):
        total = compose_prompt({"main": "1girl, // 메인\nsolo\nBREAK // x\nscenery"}, self.ORDER)
        self.assertEqual(total, "1girl,\nsolo\nBREAK\nscenery")

    def test_comment_free_sections_are_unchanged(self):
        sections = {"char_count": "1girl", "main": "  a, http://x.com/y, a//b  ", "suffix": "z"}
        self.assertEqual(compose_prompt(sections, self.ORDER), "1girl, a, http://x.com/y, a//b, z")


class RealtimeCleanerTests(unittest.TestCase):
    def setUp(self):
        from utils.prompt_cleaner import PromptCleaner
        self.cleaner = PromptCleaner()
        self.cleaner.underscore_to_space = True

    def test_the_line_after_a_comment_is_not_swallowed(self):
        cleaned = self.cleaner.clean("masterpiece, // quality_tags\nbest_quality, 1girl")
        self.assertEqual(cleaned, "masterpiece, // quality_tags\nbest quality, 1girl")
        self.assertEqual(_sent(cleaned), "masterpiece,\nbest quality, 1girl")

    def test_comment_lines_keep_their_own_lines_and_text(self):
        text = "1girl\n// 캐릭터_메모\n// 캐릭터_메모\nsolo, a_b /* x_y, (z) */ c_d"
        cleaned = self.cleaner.clean(text)
        self.assertEqual(cleaned, "1girl\n// 캐릭터_메모\n// 캐릭터_메모\nsolo, a b /* x_y, (z) */ c d")
        self.assertEqual(self.cleaner.clean(cleaned), cleaned)

    def test_identical_comments_are_never_deduplicated(self):
        self.cleaner.remove_duplicates = True
        cleaned = self.cleaner.clean("a // same\nb, c // same\nd")
        self.assertEqual(cleaned.count("// same"), 2)
        self.assertEqual(_sent(cleaned), "a,\nb, c,\nd")

    def test_crlf_and_trailing_comment(self):
        self.assertEqual(self.cleaner.clean("a_b\r\n// x\r\nc_d"), "a b\r\n// x\r\nc d")
        self.assertEqual(self.cleaner.clean("blue_eyes // why_x"), "blue eyes // why_x")

    def test_underscore_before_slashes_never_opens_a_comment(self):
        # Danbooru 저작권 태그 — 밑줄→공백이 '//' 앞을 공백으로 만들어 그 뒤(solo 까지)가 주석이 됐다
        text = "1girl, z.a.t.o._//_i_love_the_world_and_everything_in_it, solo"
        cleaned = self.cleaner.clean(text)
        self.assertEqual(cleaned, text)
        self.assertEqual(_sent(cleaned), text)
        self.assertEqual(self.cleaner.clean("a_b, x_/*y_z, c_d"), "a b, x_/*y_z, c d")
        # 빈 괄호를 지워 '//' 가 맨 앞에 오는 경우도 주석을 만들지 않는다
        self.assertFalse(has_prompt_comment(self.cleaner.clean("()// n_x")))

    def test_preserve_list_ignores_neighbouring_comments(self):
        # 주석 자리표시자가 옆 태그에 붙어 score_9 · source_anime 가 보존 목록에서 빠지던 문제
        for text in ("score_9, score_8_up, // pony 품질\nsource_anime, 1girl", "score_9 // 품질\nsource_anime",
                     "// x\nsource_anime", "score_9 // x", "score_9, /* 메모 */ source_anime"):
            with self.subTest(text=text):
                self.assertEqual(self.cleaner.clean(text), text)
        self.assertEqual(_sent(self.cleaner.clean("score_9 // 품질\nsource_anime")), "score_9,\nsource_anime")

    def test_dedupe_compares_tags_without_comments_and_keeps_the_comments(self):
        self.cleaner.remove_duplicates = True
        self.assertEqual(self.cleaner.clean("1girl, solo, // 메모\n1girl, smile"), "1girl, solo, // 메모\nsmile")
        self.assertEqual(self.cleaner.clean("1girl, solo, /* x */ 1girl, smile"), "1girl, solo, /* x */ smile")
        self.assertEqual(self.cleaner.clean("1girl, solo, 1girl /* x */"), "1girl, solo /* x */")

    def test_escape_sees_weights_next_to_comments(self):
        self.cleaner.auto_escape = True
        self.assertEqual(self.cleaner.clean("// 메모\n(red hair:1.2), ike (fire emblem)"),
                         "// 메모\n(red hair:1.2), ike \\(fire emblem\\)")
        self.assertEqual(self.cleaner.clean("(red hair:1.2) // 메모\nike (fire emblem)"),
                         "(red hair:1.2) // 메모\nike \\(fire emblem\\)")

    def test_cleaning_with_comments_sends_what_cleaning_without_them_sends(self):
        # 주석을 단 글을 정리해 보낸 태그 = 주석을 뗀 글을 정리한 태그(줄바꿈은 CLIP 에서 공백)
        from core.prompt_comments import comment_spans

        def tags(text):
            return [" ".join(t.split()) for t in text.split(",") if t.strip()]

        def comments(text):
            return [text[start:end] for start, end, _kind in comment_spans(text)]

        rng = random.Random(11)
        parts = ["a_b", "score_9", "solo", "solo", ", ", ",", " ", "\n", "// n_x", "/* c_d, e */", "(", ")", "()",
                 "__wc__", "x_/", "/y", "_//", "_/*", "(t:1.2)", "source_anime"]
        self.cleaner.remove_duplicates = True
        for _ in range(3000):
            text = "".join(rng.choice(parts) for _ in range(rng.randint(1, 9)))
            with self.subTest(text=text):
                cleaned = self.cleaner.clean(text)
                self.assertEqual(comments(cleaned), comments(text))   # 주석 글자는 그대로, 새 주석도 없다
                self.assertEqual(tags(_sent(cleaned)), tags(self.cleaner.clean(_sent(text))))


class ConditionRuleTests(unittest.TestCase):
    def test_split_keeps_comments_with_their_tag_and_join_restores_line_breaks(self):
        from utils.condition_block import join_tags, norm_tag, split_tags
        text = "1girl // 메인, 그래도\nsolo, (a, b // c\n:1.2), /* x, y */ hat"
        tags = split_tags(text)
        self.assertEqual(tags, ["1girl // 메인, 그래도", "solo", "(a, b // c\n:1.2)", "/* x, y */ hat"])
        self.assertEqual([norm_tag(t) for t in tags], ["1girl", "solo", "(a, b\n:1.2)", "hat"])
        self.assertEqual(join_tags(tags), "1girl // 메인, 그래도\nsolo, (a, b // c\n:1.2), /* x, y */ hat")
        self.assertEqual(join_tags(["a", "b"]), "a, b")
        self.assertEqual(norm_tag("// only"), "")
        self.assertEqual(split_tags("a, b\nc"), ["a", "b\nc"])   # 주석이 없으면 예전 그대로

    def test_global_rules_add_after_a_trailing_line_comment_without_swallowing(self):
        from tests.test_cond_rules_widgets import _Host, _rule
        from ui.generator_main import GeneratorMainUI
        host = _Host(main="armpits // 메모", neg="lowres // n")
        GeneratorMainUI._apply_vue_conditional_rules(
            host, [_rule("armpits", "armpit hair")], [_rule("armpits", "bad hands", location="neg")])
        self.assertEqual(host.main_prompt_text.value, "armpits // 메모\narmpit hair")
        self.assertEqual(host.neg_prompt_text.value, "lowres // n\nbad hands")
        self.assertEqual(_sent(host.main_prompt_text.value), "armpits,\narmpit hair")

    def test_conditions_ignore_text_inside_comments(self):
        from tests.test_cond_rules_widgets import _Host, _rule
        from ui.generator_main import GeneratorMainUI
        host = _Host(main="1girl, // armpits\nsolo")
        GeneratorMainUI._apply_vue_conditional_rules(host, [_rule("armpits", "armpit hair")], [])
        self.assertEqual(host.main_prompt_text.writes, 0)

    def test_character_rules_use_the_same_join(self):
        from tests.test_cond_rules_widgets import _Host
        from ui.generator_prompts import PromptHandlingMixin
        host = _Host(main="1girl // 메모", neg="lowres")
        PromptHandlingMixin._apply_condition_result(host, {"main": ["smile"], "neg": []})
        self.assertEqual(host.main_prompt_text.value, "1girl // 메모\nsmile")


class OptimizeAndClassifyTests(unittest.TestCase):
    def test_deep_clean_keeps_comments_and_dedupes_on_the_sent_text(self):
        from core.prompt_deep_clean import deep_clean_prompt
        out = deep_clean_prompt("smile, // 메모\nsolo, smile // dup\n1girl, // 메모, hat", context=["hat // ctx", "solo"])
        # 같은 주석 두 개는 지우지 않고, 'smile // dup' 은 smile 중복, 1girl 은 앞으로(인물수), 줄 주석 뒤는 줄바꿈
        self.assertEqual(out["optimized"], "1girl, smile, // 메모\n// 메모, hat")
        self.assertEqual((out["removed"], out["removed_from_context"], out["tag_count"]), (2, 1, 2))
        self.assertEqual(_sent(out["optimized"]), "1girl, smile,")

    def test_deep_clean_without_comments_is_unchanged(self):
        from core.prompt_deep_clean import deep_clean_prompt
        out = deep_clean_prompt("smile, 1girl, smile, masterpiece", context="hat")
        self.assertEqual(out["optimized"], "1girl, masterpiece, smile")
        self.assertEqual((out["removed"], out["tag_count"]), (1, 3))

    def test_separate_tags_classifies_the_sent_text_and_keeps_comments_in_rest(self):
        from ui.vue_bridge import VueBridge

        classified = []

        class _Intel:
            def split_by_categories(self, tags, cats):
                classified.append(list(tags))
                groups = {c: [] for c in cats}
                rest = []
                for t in tags:
                    (groups["expression"] if t == "smile" else rest).append(t)
                return {"rest": rest, "groups": groups}

        with mock.patch("core.tag_intelligence.get_tag_intelligence", return_value=_Intel()):
            out = json.loads(VueBridge.separateTags(None, "1girl, // 메모\nsmile // why\nhat", '["expression"]'))
            plain = json.loads(VueBridge.separateTags(None, "1girl, smile, hat", '["expression"]'))
        self.assertEqual(classified, [["1girl", "smile", "hat"], ["1girl", "smile", "hat"]])   # 주석은 분류하지 않는다
        self.assertEqual(out["groups"], {"expression": ["smile"]})
        self.assertEqual(out["rest"], "1girl, // 메모\nhat")
        self.assertEqual(plain["groups"], {"expression": ["smile"]})
        self.assertEqual(plain["rest"], "1girl, hat")


class LoraStackTests(unittest.TestCase):
    ENTRIES = [{"name": "style", "weight": 0.7, "enabled": True}]

    def test_stack_lora_is_not_swallowed_by_a_trailing_line_comment(self):
        from core.lora_stack import append_lora_stack_to_prompt
        out = append_lora_stack_to_prompt("1girl // 메모", self.ENTRIES)
        self.assertEqual(out, "1girl // 메모\n<lora:style:0.70>")
        self.assertEqual(_sent(out), "1girl,\n<lora:style:0.70>")
        self.assertEqual(append_lora_stack_to_prompt("1girl", self.ENTRIES), "1girl, <lora:style:0.70>")

    def test_a_commented_out_lora_does_not_suppress_the_stack(self):
        from core.lora_stack import append_lora_stack_to_prompt
        out = append_lora_stack_to_prompt("1girl, /* <lora:style:1> */ solo", self.ENTRIES)
        self.assertEqual(_sent(out), "1girl, solo, <lora:style:0.70>")
        self.assertEqual(append_lora_stack_to_prompt("1girl, <lora:style:1>", self.ENTRIES), "1girl, <lora:style:1>")


class WildcardTests(unittest.TestCase):
    def test_file_wildcards_drop_comments_including_multi_line_blocks(self):
        from utils.file_wildcard import FileWildcardManager
        with tempfile.TemporaryDirectory() as tmp:
            Path(tmp, "hair.txt").write_text(
                "# 머리색\nblue hair // 메모, red hair\n/* 쓰지 않음\ngreen hair\n*/\nwhite hair, /* off */ pink hair\n// 끝\n",
                encoding="utf-8")
            mgr = FileWildcardManager(tmp)
            self.assertEqual(mgr.load_wildcard("hair"), [["blue hair"], ["white hair", "pink hair"]])
            tree = {item["name"]: item for item in mgr.get_wildcard_tree()}
            self.assertEqual(tree["hair"]["tags"], ["blue hair", "white hair, pink hair"])
            self.assertIn("// 끝", tree["hair"]["lines"])   # 편집용 원문은 그대로

    def test_file_wildcards_split_lines_like_reading_a_file(self):
        # str.splitlines 는 U+2028·\\f 에서도 줄을 나눠 주석 없는 파일의 그룹이 바뀌었다
        from utils.file_wildcard import FileWildcardManager
        with tempfile.TemporaryDirectory() as tmp:
            Path(tmp, "pose.txt").write_text(
                "standing, sitting\u2028kneeling\nlying, running\x0cjumping\r\nsquatting // 메모\n", encoding="utf-8")
            mgr = FileWildcardManager(tmp)
            self.assertEqual(mgr.load_wildcard("pose"), [["standing", "sitting\u2028kneeling"],
                                                         ["lying", "running\x0cjumping"], ["squatting"]])
            tree = {item["name"]: item for item in mgr.get_wildcard_tree()}
            self.assertEqual(tree["pose"]["tags"],
                             ["standing, sitting\u2028kneeling", "lying, running\x0cjumping", "squatting"])

    def test_instant_wildcards_never_pick_comment_text(self):
        from core.instant_wildcards import InstantWildcards
        with tempfile.TemporaryDirectory() as tmp:
            store = Path(tmp) / "instant_wildcards.json"
            store.write_text(json.dumps({"version": 1, "wildcards": {
                "mood": ["// 설명 줄", "{1000}:happy // 자주", "sad"]}}), encoding="utf-8")
            iw = InstantWildcards(store_path=store, rng=random.Random(3))
            picks = {iw.pick("mood") for _ in range(30)}
        self.assertLessEqual(picks, {"happy", "sad"})
        self.assertIn("happy", picks)


class AuxPromptTests(unittest.TestCase):
    def test_refine_strips_every_input_before_normalizing(self):
        from core.refine_prompt import build_refine_prompts
        out = build_refine_prompts(main_prompt="1girl, // 메모\nshirt, smile", main_negative="bad // n\nworst",
                                   target="shirt // t", replacement="dress /* r */", negative="// only")
        self.assertEqual(out["prompt"], "1girl, dress, smile")
        self.assertEqual(out["negative_prompt"], "bad, worst")
        self.assertEqual(out["detect_tokens"], ["shirt"])

    def test_sam3_state_treats_comment_only_as_empty(self):
        from core import sam3_args
        with mock.patch("core.forge_modules.resolve_sam3_checkpoint", side_effect=lambda name: name):
            state = sam3_args.build_state(
                {"sam3_prompt": "// 검출 메모", "sam3_exclude_prompt": "hair /* x */",
                 "sam3_inpaint_prompt": "// only", "sam3_negative_prompt": "bad // n"},
                prompt="1girl // main", negative_prompt="unused")
            plain = sam3_args.build_state({"sam3_inpaint_prompt": ""}, prompt="  1girl  ")
        self.assertEqual(state["sam3_prompt"], "face")
        self.assertEqual(state["sam3_exclude_prompt"], "hair")
        self.assertEqual(state["sam3_inpaint_prompt"], "1girl")
        self.assertEqual(state["sam3_negative_prompt"], "bad")
        self.assertEqual(plain["sam3_inpaint_prompt"], "  1girl  ")   # 주석이 없으면 예전 그대로

    @staticmethod
    def _data_url(size=(8, 8)) -> str:
        out = io.BytesIO()
        Image.new("RGB", size, "gray").save(out, format="PNG")
        return "data:image/png;base64," + base64.b64encode(out.getvalue()).decode("ascii")

    def test_i2i_and_inpaint_comment_only_prompts_fall_back_to_t2i(self):
        from core.i2i_payload import build_i2i_request
        from core.inpaint_payload import build_inpaint_request
        image = self._data_url()
        i2i = build_i2i_request({"image": image, "prompt": "// 메모", "negative_prompt": "blur // n"},
                                main_prompt="1girl // m\nsolo", main_negative="lowres")
        self.assertEqual((i2i.payload["prompt"], i2i.payload["negative_prompt"]), ("1girl,\nsolo", "blur"))
        inpaint = build_inpaint_request({"image": image, "mask": self._data_url(), "prompt": "/* off */"},
                                        main_prompt="cat // m", main_negative="// n")
        self.assertEqual((inpaint.payload["prompt"], inpaint.payload["negative_prompt"]), ("cat", ""))

    def test_tile_repair_prompt(self):
        from core.tile_repair_request import normalize_settings
        self.assertEqual(normalize_settings({"prompt": "tiles // 메모", "negative_prompt": "/* x */ bad"})["prompt"],
                         "tiles")
        with self.assertRaises(ValueError):
            normalize_settings({"prompt": "// only"})


class SearchApplyTests(unittest.TestCase):
    def test_space_split_search_tags_never_open_a_comment(self):
        # 검색 데이터(공백 구분·밑줄)를 프롬프트로 넣을 때 밑줄→공백이 '_//' 를 주석으로 만들지 않는다
        from tests.test_apply_prompt_exclude_rules import _apply
        harness = _apply("", general="1girl z.a.t.o._//_i_love_the_world_and_everything_in_it solo long_hair")
        main = harness.main_prompt_text.toPlainText()
        self.assertIn("z.a.t.o._//_i_love_the_world_and_everything_in_it", main)
        self.assertIn("long hair", main)
        self.assertFalse(has_prompt_comment(main))
        self.assertIn("solo", _sent(main))


class AutoNlTests(unittest.TestCase):
    def test_auto_nl_appends_after_a_trailing_line_comment(self):
        from ui.generator_actions import ActionsMixin

        class _Text:
            def __init__(self):
                self.value = ""

            def setPlainText(self, value):
                self.value = value

        class _Host:
            is_automating = False
            _auto_nl_base = "1girl, solo // 메모"

            def __init__(self):
                self.main_prompt_text = _Text()

            def _auto_nl_result_is_current(self):
                return True

        host = _Host()
        ActionsMixin._on_auto_nl_done(host, json.dumps({"tags": "A girl stands."}))
        self.assertEqual(host.main_prompt_text.value, "1girl, solo // 메모\nA girl stands.")
        self.assertFalse(has_prompt_comment(_sent(host.main_prompt_text.value)))


if __name__ == "__main__":
    unittest.main()
