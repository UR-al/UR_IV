"""The prompt card's on/off switches: every box and the negative, wherever a run's prompts are built.

Off, a box keeps its text and stays out of the assembled prompt; the negative off runs with an empty negative. A state
without the switches (a panel, workflow or image from before them) has them all on and gives the same prompts as before.
"""
import random
import unittest

from comfy_custom_nodes.ai_studio_forge_parity import studio_infotext
from comfy_custom_nodes.ai_studio_forge_parity import studio_state as ss
from comfy_custom_nodes.ai_studio_forge_parity import studio_wildcards as sw
from comfy_custom_nodes.ai_studio_forge_parity import xyz_plan as xp
from comfy_custom_nodes.ai_studio_forge_parity.studio_infotext_read import parse_infotext

SWITCHES = ("prompt_prefix_on", "prompt_char_count_on", "prompt_character_on", "prompt_copyright_on", "prompt_artist_on",
            "prompt_on", "prompt_suffix_on", "negative_on")
BOXES = {"prompt_prefix": "pre", "prompt_char_count": "1boy", "prompt_character": "hero", "prompt_copyright": "game",
         "prompt_artist": "@a", "prompt": "main", "prompt_suffix": "post"}
WHOLE = "pre, 1boy, hero, game, @a, main, post"


def _state(**changes):
    return ss.normalize_state({**BOXES, "negative": "bad hands", "unet": "anima.safetensors", **changes})


class SchemaTests(unittest.TestCase):
    def test_every_box_and_the_negative_have_a_switch(self):
        self.assertEqual(ss.PROMPT_SWITCHES, {key: f"{key}_on" for key in ss.PROMPT_SECTIONS})
        self.assertEqual(ss.NEGATIVE_SWITCH, "negative_on")
        self.assertEqual(tuple(ss.BOX_SWITCHES), (*ss.PROMPT_SECTIONS, "negative"))  # the card's rows, top to bottom
        self.assertEqual(tuple(ss.BOX_SWITCHES.values()), SWITCHES)

    def test_each_switch_defaults_on_and_keeps_only_booleans(self):
        for key in SWITCHES:
            with self.subTest(key=key):
                self.assertIs(ss.DEFAULT_STATE[key], True)
                self.assertIs(ss.normalize_state({})[key], True)  # a panel saved before the switches
                self.assertIs(ss.normalize_state({key: False})[key], False)
                for other in ("no", 0, None, "false"):
                    self.assertIs(ss.normalize_state({key: other})[key], True)

    def test_a_switched_off_box_keeps_its_text(self):
        state = _state(**dict.fromkeys(SWITCHES, False))
        self.assertEqual({key: state[key] for key in BOXES}, BOXES)
        self.assertEqual(state["negative"], "bad hands")


class AssembledPromptTests(unittest.TestCase):
    def test_each_switch_takes_its_own_box_out(self):
        for key, text in BOXES.items():
            with self.subTest(key=key):
                wanted = ", ".join(part for part in WHOLE.split(", ") if part != text)
                self.assertEqual(ss.assembled_prompt({**BOXES, ss.PROMPT_SWITCHES[key]: False}), wanted)

    def test_a_state_without_the_switches_is_the_state_with_all_of_them_on(self):
        on = dict.fromkeys(SWITCHES, True)
        for boxes in (BOXES, {"prompt": "1boy, \n"}, {"prompt": "1boy,", "prompt_artist": "@a,"}, {}):
            with self.subTest(boxes=boxes):
                self.assertEqual(ss.assembled_prompt({**boxes, **on}), ss.assembled_prompt(boxes))
                self.assertEqual(ss.effective_negative({"negative": "bad, \n", **on}), "bad, \n")
        self.assertEqual(ss.assembled_prompt(BOXES), WHOLE)
        self.assertEqual(ss.assembled_prompt({"prompt": "1boy, \n"}), "1boy, \n")  # byte for byte

    def test_the_main_box_off_leaves_the_other_boxes(self):
        self.assertEqual(ss.assembled_prompt({**BOXES, "prompt_on": False}), "pre, 1boy, hero, game, @a, post")
        self.assertEqual(ss.assembled_prompt({"prompt": "1boy, \n", "prompt_on": False}), "")  # nothing else to send
        self.assertEqual(ss.assembled_prompt({"prompt": "1boy", "prompt_artist": "@a", "prompt_on": False}), "@a")

    def test_with_the_other_boxes_off_the_prompt_is_as_written(self):
        off = {key: False for key in SWITCHES if key not in ("prompt_on", "negative_on")}
        self.assertEqual(ss.assembled_prompt({**BOXES, **off, "prompt": "1boy, \n"}), "1boy, \n")

    def test_everything_off_is_an_empty_prompt(self):
        self.assertEqual(ss.assembled_prompt({**BOXES, **dict.fromkeys(SWITCHES, False)}), "")

    def test_the_list_loras_do_not_follow_a_box(self):
        state = {"prompt": "1boy <lora:in_box:1>", "prompt_on": False, "loras": [{"on": True, "name": "l", "strength": 0.5}]}
        self.assertEqual(ss.effective_prompt(state), "<lora:l:0.5>")  # the box's own tag goes with the box


class EffectiveNegativeTests(unittest.TestCase):
    def test_off_is_an_empty_negative(self):
        self.assertEqual(ss.effective_negative({"negative": "bad hands"}), "bad hands")
        self.assertEqual(ss.effective_negative({"negative": "bad hands", "negative_on": True}), "bad hands")
        self.assertEqual(ss.effective_negative({"negative": "bad hands", "negative_on": False}), "")
        self.assertEqual(ss.effective_negative({}), "")

    def test_the_negative_switch_leaves_the_positive_alone(self):
        self.assertEqual(ss.assembled_prompt({**BOXES, "negative_on": False}), WHOLE)

    def test_one_prompt_in_the_main_box(self):
        written = ss.single_prompt("a, b")
        self.assertEqual(written, {**dict.fromkeys(ss.PROMPT_BOXES, ""), "prompt": "a, b", "prompt_on": True})
        state = _state(prompt_on=False, prompt_artist_on=False)
        state.update(ss.single_prompt(ss.assembled_prompt(state)))
        self.assertEqual(ss.assembled_prompt(state), "pre, 1boy, hero, game, post")  # the same text again
        self.assertIs(state["prompt_artist_on"], False)  # an empty box's switch stays the panel's


class InfotextTests(unittest.TestCase):
    def _text(self, **changes):
        return studio_infotext.build_infotext(_state(**changes), seed=1)

    def test_the_text_is_what_the_run_used(self):
        self.assertEqual(self._text().split("\n")[:2], [WHOLE, "Negative prompt: bad hands"])
        self.assertEqual(self._text(prompt_character_on=False, prompt_copyright_on=False).split("\n")[0], "pre, 1boy, @a, main, post")
        self.assertEqual(self._text(prompt_on=False).split("\n")[0], "pre, 1boy, hero, game, @a, post")

    def test_a_negative_switched_off_has_no_negative_line(self):
        lines = self._text(negative_on=False).split("\n")
        self.assertEqual(len(lines), 2)
        self.assertEqual(lines[0], WHOLE)
        self.assertTrue(lines[1].startswith("Steps: "))
        self.assertNotIn("bad hands", self._text(negative_on=False))

    def test_everything_off_still_has_its_settings_line(self):
        text = self._text(**dict.fromkeys(SWITCHES, False))
        self.assertTrue(text.startswith("Steps: "))


class InfotextReadTests(unittest.TestCase):
    def _settings(self, **changes):
        return parse_infotext(studio_infotext.build_infotext(_state(**changes), seed=7))["settings"]

    def test_the_prompt_read_back_is_switched_on(self):
        settings = self._settings(prompt_on=False, prompt_artist_on=False)
        self.assertEqual(settings["prompt"], "pre, 1boy, hero, game, post")
        self.assertIs(settings["prompt_on"], True)  # else the panel would keep the image's prompt out
        self.assertEqual([settings[key] for key in ss.PROMPT_BOXES], [""] * 6)
        for key in SWITCHES[:5] + ("prompt_suffix_on",):  # the boxes come back empty: their switches stay the panel's
            self.assertNotIn(key, settings)

    def test_the_negative_read_back_is_switched_on(self):
        self.assertEqual((self._settings()["negative"], self._settings()["negative_on"]), ("bad hands", True))
        off = self._settings(negative_on=False)
        self.assertEqual((off["negative"], off["negative_on"]), ("", True))  # made without one: an empty box, on
        forge = parse_infotext("1girl\nNegative prompt: lowres\nSteps: 20")["settings"]
        self.assertEqual((forge["prompt"], forge["prompt_on"], forge["negative"], forge["negative_on"]), ("1girl", True, "lowres", True))


FILES = {"one": [["only"]]}


def _lookup(name):
    return FILES.get(name, [])


class WildcardTests(unittest.TestCase):
    def test_a_box_that_is_off_is_not_drawn(self):
        off = _state(prompt="__one__", prompt_on=False, negative="{a|a}", negative_on=False)
        self.assertFalse(sw.uses_wildcards(off))
        after = sw.expand_state(off, 5, _lookup)[0]
        self.assertEqual((after["prompt"], after["negative"]), ("__one__", "{a|a}"))  # the text stays
        self.assertEqual((after["prompt_template"], after["negative_template"]), ("", ""))
        self.assertEqual(sw.expand_run(off, _lookup), {"states": [], "missing": []})

    def test_the_drawn_prompt_runs_whatever_the_main_switch_was(self):
        before = _state(prompt="main", prompt_character="__one__", prompt_on=False)
        after = sw.expand_state(before, 5, _lookup)[0]
        self.assertEqual(after["prompt"], "pre, 1boy, only, game, @a, post")
        self.assertIs(after["prompt_on"], True)  # the joined text sits in the main box now
        self.assertEqual(ss.assembled_prompt(after), "pre, 1boy, only, game, @a, post")
        self.assertEqual(after["prompt_template"], "pre, 1boy, __one__, game, @a, post")

    def test_the_negative_is_drawn_while_it_is_on(self):
        after = sw.expand_state(_state(negative="__one__"), 5, _lookup)[0]
        self.assertEqual((after["negative"], after["negative_template"], after["negative_on"]), ("only", "__one__", True))

    def test_a_negative_that_is_off_does_not_stop_the_positive(self):
        after = sw.expand_state(_state(prompt="__one__", negative="__one__", negative_on=False), 5, _lookup)[0]
        self.assertEqual(after["prompt"], "pre, 1boy, hero, game, @a, only, post")
        self.assertEqual((after["negative"], after["negative_on"], ss.effective_negative(after)), ("__one__", False, ""))


def _cells(state, axes=None, **more):
    plan = xp.plan_xyz(state, axes or [{"axis": "steps", "text": "10, 20"}], {}, rng=random.Random(1), families={}, **more)
    return plan, [cell.state for cell in plan.cells]


class XyzTests(unittest.TestCase):
    def test_a_cell_leaves_every_switched_off_box_out(self):
        _, cells = _cells({**BOXES, "prompt_character_on": False, "prompt_copyright_on": False})
        for cell in cells:
            self.assertEqual(cell["prompt"], "pre, 1boy, @a, main, post")
            self.assertEqual(ss.effective_prompt(cell), "pre, 1boy, @a, main, post")

    def test_a_cell_runs_its_joined_prompt_with_the_main_box_off(self):
        _, cells = _cells({**BOXES, "prompt_on": False})
        for cell in cells:
            self.assertEqual(cell["prompt"], "pre, 1boy, hero, game, @a, post")
            self.assertIs(cell["prompt_on"], True)
            self.assertEqual(ss.effective_prompt(cell), "pre, 1boy, hero, game, @a, post")

    def test_a_cell_of_a_negative_that_is_off_has_none(self):
        _, cells = _cells({"prompt": "cat", "negative": "blurry", "negative_on": False})
        for cell in cells:
            self.assertEqual((cell["negative"], ss.effective_negative(cell)), ("", ""))
        _, cells = _cells({"prompt": "cat", "negative": "blurry"})
        self.assertEqual([cell["negative"] for cell in cells], ["blurry", "blurry"])

    def test_prompt_sr_searches_what_the_run_uses(self):
        axes = [{"axis": "prompt_sr", "text": "blurry, sharp"}]
        plan, cells = _cells({"prompt": "cat", "negative": "blurry"}, axes)
        self.assertEqual((plan.errors, [cell["negative"] for cell in cells]), ([], ["blurry", "sharp"]))
        plan, cells = _cells({"prompt": "cat", "negative": "blurry", "negative_on": False}, axes)
        self.assertEqual(cells, [])
        self.assertIn("blurry", plan.errors[0])  # found only in a negative that does not run
        plan, cells = _cells({"prompt": "cat", "prompt_artist": "blurry", "prompt_artist_on": False}, axes)
        self.assertEqual(cells, [])

    def test_a_lora_tag_in_a_box_that_is_off_is_not_looked_for(self):
        state = {"prompt": "cat", "prompt_artist": "<lora:gone:1>", "negative": "<lora:gone_too:1>"}
        plan, _ = _cells(state, loras=["a.safetensors"])
        self.assertIn("gone, gone_too", plan.errors[0])
        plan, cells = _cells({**state, "prompt_artist_on": False, "negative_on": False}, loras=["a.safetensors"])
        self.assertEqual((plan.errors, len(cells)), ([], 2))
        self.assertEqual(xp.missing_loras([ss.normalize_state({**state, "negative_on": False})], ["a.safetensors"]), ["gone"])


if __name__ == "__main__":
    unittest.main()
