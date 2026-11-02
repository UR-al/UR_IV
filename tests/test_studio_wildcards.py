"""Anima studio wildcards: the app's file syntax plus Dynamic Prompts' {a|b}, seeded per image."""
import random
import tempfile
import unittest
from pathlib import Path

from comfy_custom_nodes.ai_studio_forge_parity import studio_state as ss
from comfy_custom_nodes.ai_studio_forge_parity import studio_wildcards as sw

FILES = {
    "hair": [["red hair", "blue hair"], ["ponytail", "braid"]],
    "one": [["only"]],
    "nest": [["__one__ tag"]],
    "loop": [["__loop__"]],
    "either": [["[left|right]"]],
    "styled": [["<lora:style:0.8>, painted"]],
}


def lookup(name):
    return FILES.get(name, [])


def expand(text, seed=1):
    missing = []
    return sw.expand_text(text, random.Random(seed), lookup, missing), missing


class ExpandTextTests(unittest.TestCase):
    def test_both_file_syntaxes_are_the_same_wildcard(self):
        self.assertEqual(expand("__one__")[0], "only")
        self.assertEqual(expand("~/one/~")[0], "only")

    def test_every_line_is_a_group_and_gives_one_option(self):
        text, _ = expand("__hair__")
        first, second = text.split(", ")
        self.assertIn(first, FILES["hair"][0])
        self.assertIn(second, FILES["hair"][1])

    def test_name_colon_n_takes_n_groups(self):
        self.assertEqual(len(expand("__hair:1__")[0].split(", ")), 1)
        self.assertEqual(len(expand("~/hair:1/~")[0].split(", ")), 1)
        self.assertEqual(len(expand("__hair:9__")[0].split(", ")), 2)

    def test_nested_files(self):
        self.assertEqual(expand("__nest__")[0], "only tag")

    def test_a_file_naming_itself_stops(self):
        text, _ = expand("__loop__")
        self.assertEqual(text, "__loop__")

    def test_a_missing_file_stays_and_is_reported_once(self):
        text, missing = expand("__nope__, ~/nope/~, 1girl")
        self.assertEqual(text, "__nope__, ~/nope/~, 1girl")
        self.assertEqual(missing, ["nope"])

    def test_or_groups_of_a_file_line_are_picked(self):
        self.assertIn(expand("__either__")[0], ("left", "right"))

    def test_or_groups_of_the_prompt_stay_for_prompt_editing(self):
        self.assertEqual(expand("[red|blue] hair, [cat:dog:0.5]")[0], "[red|blue] hair, [cat:dog:0.5]")

    def test_braces_pick_one_and_nest(self):
        self.assertIn(expand("{a|b}")[0], ("a", "b"))
        self.assertIn(expand("{a|{b|c}}")[0], ("a", "b", "c"))
        self.assertEqual(expand("{|}")[0], "")

    def test_brace_weights(self):
        picks = {expand("{0::a|5::b}", seed)[0] for seed in range(40)}
        self.assertEqual(picks, {"b"})

    def test_braces_without_a_bar_stay(self):
        self.assertEqual(expand("{just text}")[0], "{just text}")

    def test_escaped_braces_stay(self):
        self.assertEqual(expand(r"\{a|b\}")[0], r"\{a|b\}")

    def test_a_lora_tag_in_a_file_survives(self):
        self.assertEqual(expand("__styled__")[0], "<lora:style:0.8>, painted")

    def test_the_seed_decides(self):
        self.assertEqual(expand("{a|b|c|d|e|f}", 7)[0], expand("{a|b|c|d|e|f}", 7)[0])
        self.assertGreater(len({expand("{a|b|c|d|e|f}", seed)[0] for seed in range(30)}), 1)

    def test_underscored_words_are_not_wildcards(self):
        self.assertFalse(sw.has_wildcard("hair_style, long_hair"))
        self.assertFalse(sw.has_wildcard("^__^, >__<"))
        self.assertTrue(sw.has_wildcard("__hair_style__"))
        self.assertTrue(sw.has_wildcard("{a|b}"))
        self.assertTrue(sw.has_wildcard("~/x/~"))


class FileTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.files = sw.WildcardFiles(lambda: [self.root / "gone", self.root])

    def write(self, name, text):
        (self.root / f"{name}.txt").write_text(text, encoding="utf-8")

    def test_lines_are_groups_of_comma_options(self):
        self.write("a", "x, y\n\nz\n")
        self.assertEqual(self.files.groups("a"), [["x", "y"], ["z"]])

    def test_comments_are_left_out(self):
        self.write("a", "# note\nx // tail\n/* gone */ y\nhttp://site, b//c\n")
        self.assertEqual(self.files.groups("a"), [["x"], ["y"], ["http://site", "b//c"]])

    def test_a_bom_is_not_part_of_the_first_option(self):
        (self.root / "a.txt").write_bytes("﻿x\n".encode("utf-8"))
        self.assertEqual(self.files.groups("a"), [["x"]])

    def test_an_edited_file_is_read_again(self):
        self.write("a", "x\n")
        self.assertEqual(self.files.groups("a"), [["x"]])
        self.write("a", "yy\n")
        self.assertEqual(self.files.groups("a"), [["yy"]])

    def test_a_name_cannot_leave_the_folder(self):
        (self.root.parent / "outside_wildcard_test.txt").write_text("secret\n", encoding="utf-8")
        self.addCleanup((self.root.parent / "outside_wildcard_test.txt").unlink)
        for name in ("../outside_wildcard_test", "..\\outside_wildcard_test", "a/b", "c:x", ""):
            self.assertEqual(self.files.groups(name), [])

    def test_names_list_every_folder_once(self):
        self.write("b", "x\n")
        self.write("a", "x\n")
        (self.root / "skip.csv").write_text("x", encoding="utf-8")
        self.assertEqual(self.files.names(), ["a", "b"])

    def test_an_unreadable_file_counts_as_missing(self):
        (self.root / "a.txt").write_bytes(b"\xff\xfe\xff")
        self.assertEqual(self.files.groups("a"), [])


def state(**overrides):
    return ss.normalize_state({"seed": 5, **overrides})


class ExpandStateTests(unittest.TestCase):
    def test_a_state_without_wildcards_reports_nothing_to_do(self):
        self.assertFalse(sw.uses_wildcards(state(prompt="1girl, [a|b]")))
        self.assertEqual(sw.expand_run(state(prompt="1girl"), lookup), {"states": [], "missing": []})

    def test_prompt_boxes_join_before_they_expand(self):
        before = state(prompt="__one__", prompt_character="{a|a}")
        after = sw.expand_state(before, 5, lookup)[0]
        self.assertEqual(after["prompt"], "a, only")
        self.assertEqual(after["prompt_character"], "")
        self.assertEqual(after["prompt_template"], "{a|a}, __one__")

    def test_the_list_loras_stay_in_the_list(self):
        before = state(prompt="__one__", loras=[{"on": True, "name": "style", "strength": 0.5}])
        after = sw.expand_state(before, 5, lookup)[0]
        self.assertEqual(ss.effective_prompt(after), "only <lora:style:0.5>")

    def test_a_negative_with_a_wildcard_keeps_its_template(self):
        after = sw.expand_state(state(prompt="x", negative="__one__"), 5, lookup)[0]
        self.assertEqual((after["negative"], after["negative_template"]), ("only", "__one__"))
        self.assertEqual((after["prompt"], after["prompt_template"]), ("x", ""))

    def test_the_passes_own_prompts_expand_too(self):
        before = state(prompt="x", hires={"on": True, "prompt": "__one__"}, sam3={"on": True, "negative_prompt": "__one__"},
                       ext={"sam3_refine": {"on": True, "inpaint_prompt": "__one__"}})
        after = sw.expand_state(before, 5, lookup)[0]
        self.assertEqual(after["hires"]["prompt"], "only")
        self.assertEqual(after["sam3"]["negative_prompt"], "only")
        self.assertEqual(after["ext"]["sam3_refine"]["inpaint_prompt"], "only")

    def test_the_input_is_left_alone(self):
        before = state(prompt="__one__")
        sw.expand_state(before, 5, lookup)
        self.assertEqual(before["prompt"], "__one__")

    def test_a_batch_becomes_one_run_per_image(self):
        result = sw.expand_run(state(prompt="{a|b|c|d|e|f|g|h}", batch=6), lookup)
        self.assertEqual([one["seed"] for one in result["states"]], [5, 6, 7, 8, 9, 10])
        self.assertTrue(all(one["batch"] == 1 for one in result["states"]))
        self.assertGreater(len({one["prompt"] for one in result["states"]}), 1)

    def test_seed_minus_one_gets_a_real_seed_first(self):
        result = sw.expand_run(ss.normalize_state({"prompt": "__one__", "seed": -1, "batch": 2}), lookup, rng=random.Random(3))
        first, second = [one["seed"] for one in result["states"]]
        self.assertGreaterEqual(first, 0)
        self.assertEqual(second, first + 1)

    def test_missing_files_are_listed_once(self):
        result = sw.expand_run(state(prompt="__nope__, __one__", negative="__nope__", batch=2), lookup)
        self.assertEqual(result["missing"], ["nope"])
        self.assertEqual(result["states"][0]["prompt"], "__nope__, only")


class FalseFriendTests(unittest.TestCase):
    def test_lora_names_with_double_underscores_are_not_wildcards(self):
        self.assertFalse(sw.has_wildcard("<lora:a__v1:1> <lora:b__v2:0.5>, 1girl"))
        self.assertFalse(sw.has_wildcard("<lora:a__v1__x:1>"))
        self.assertEqual(expand("<lora:a__one__x:1>, __one__")[0], "<lora:a__one__x:1>, only")

    def test_faces_are_not_wildcards(self):
        for text in ("o__o ^__^ smile", "^__^", ">__<, x", "-__-"):
            self.assertFalse(sw.has_wildcard(text), text)

    def test_a_lora_tag_inside_braces_is_an_option(self):
        self.assertIn(expand("{<lora:a:1>|<lora:b:1>}")[0], ("<lora:a:1>", "<lora:b:1>"))

    def test_brackets_keep_their_bar_inside_braces(self):
        picks = {expand("{[red|blue] hair|short hair}", seed)[0] for seed in range(30)}
        self.assertEqual(picks, {"[red|blue] hair", "short hair"})

    def test_a_name_that_is_missing_changes_nothing_but_the_notice(self):
        self.assertEqual(sw.expand_run(state(prompt="a__b c__d", batch=3), lookup), {"states": [], "missing": ["b c"]})

    def test_a_prompt_of_a_pass_that_is_off_is_left_alone(self):
        before = state(prompt="x", hires={"on": False, "prompt": "__one__"}, batch=3)
        self.assertEqual(sw.expand_run(before, lookup), {"states": [], "missing": []})

    def test_each_prompt_draws_by_itself(self):
        """A seed gives a pass its picks whatever the main prompt holds: 같은 시드 runs without the main wildcards."""
        files = {"six": [["a", "b", "c", "d", "e", "f"]]}
        with_main = state(prompt="__six__, {1|2|3}", hires={"on": True, "prompt": "__six__"})
        without = state(prompt="fixed", hires={"on": True, "prompt": "__six__"})
        for seed in range(12):
            picks = [sw.expand_state(one, seed, lambda name: files[name])[0]["hires"]["prompt"] for one in (with_main, without)]
            self.assertEqual(picks[0], picks[1])


class AppFileRulesTests(FileTests):
    def test_form_feed_and_line_separator_stay_inside_a_line(self):
        self.write("a", "a\u2028b, c\x0cd\n")
        self.assertEqual(self.files.groups("a"), [["a\u2028b", "c\x0cd"]])

    def test_a_line_comment_hides_a_block_comment_opener(self):
        self.write("a", "// x /* y\nkeep */ z, w\n")
        self.assertEqual(self.files.groups("a"), [["keep */ z", "w"]])

    def test_a_name_may_carry_its_extension(self):
        self.write("a", "x\n")
        self.assertEqual(self.files.groups("a.txt"), [["x"]])


if __name__ == "__main__":
    unittest.main()
