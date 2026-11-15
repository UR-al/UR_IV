"""Prompt S/R / Prompt order values that hold several tags: one value stays one value from the value box to the cell.

The XYZ panel draws these two axes as text blocks (web/ai_studio/xyz_blocks.js) and writes the value box one value a line,
CSV quoted; Forge's own way is quotes on one line.  Both must reach the planner as whole values, so a grid can swap
several tags at once.  Pure Python.
"""
import json
import random
import unittest
from pathlib import Path

from comfy_custom_nodes.ai_studio_forge_parity import xyz_plan as xp
from comfy_custom_nodes.ai_studio_forge_parity import xyz_values
from comfy_custom_nodes.ai_studio_forge_parity.studio_infotext import build_infotext
from comfy_custom_nodes.ai_studio_forge_parity.studio_infotext_read import _params

GOLDEN = (Path(__file__).resolve().parent.parent / "comfy_custom_nodes" / "ai_studio_forge_parity"
          / "js_tests" / "fixtures" / "xyz_values_golden.json")
STATE = {"prompt": "1girl, blue hair, long hair, smile", "prompt_artist": "@a", "negative": "bad hands, lowres", "seed": 5}


def _plan(state, *axes):
    entries = [{"axis": axis, "text": text} for axis, text in axes]
    return xp.plan_xyz(dict(state), entries, {}, rng=random.Random(0), families={})


def _prompts(plan):
    return [cell.state["prompt"] for cell in plan.cells]


class PromptSrWholeValueTests(unittest.TestCase):
    def test_quoted_values_on_one_line_swap_several_tags_at_once(self):
        """Forge's way: quotes protect the commas."""
        plan = _plan(STATE, ("prompt_sr", '"blue hair, long hair", "red hair, short hair", "green hair, twintails"'))
        self.assertEqual(plan.errors, [])
        self.assertEqual(_prompts(plan), ["@a, 1girl, blue hair, long hair, smile", "@a, 1girl, red hair, short hair, smile",
                                          "@a, 1girl, green hair, twintails, smile"])
        self.assertEqual(plan.labels["x"], ["blue hair, long hair", "red hair, short hair", "green hair, twintails"])

    def test_the_blocks_write_one_value_a_line(self):
        text = '"blue hair, long hair"\n"red hair, short hair"\nbob cut'
        plan = _plan(STATE, ("prompt_sr", text))
        self.assertEqual(plan.errors, [])
        self.assertEqual(_prompts(plan), ["@a, 1girl, blue hair, long hair, smile", "@a, 1girl, red hair, short hair, smile",
                                          "@a, 1girl, bob cut, smile"])
        self.assertEqual(plan.infotext_extra["X Values"], text)

    def test_without_quotes_every_comma_is_a_new_value(self):
        """What the plain text box does with ``blue hair, long hair``: two values, so only single tags could be swapped."""
        plan = _plan(STATE, ("prompt_sr", "blue hair, long hair\nred hair, short hair"))
        self.assertEqual(plan.labels["x"], ["blue hair", "long hair", "red hair", "short hair"])

    def test_a_search_text_may_span_the_prompt_boxes(self):
        """The boxes are joined with ", " before the axes run."""
        plan = _plan(STATE, ("prompt_sr", '"@a, 1girl"\n"@b, 1boy"'))
        self.assertEqual(plan.errors, [])
        self.assertEqual(_prompts(plan)[1], "@b, 1boy, blue hair, long hair, smile")

    def test_a_value_may_hold_a_line_break_and_quotes(self):
        state = {"prompt": '1girl,\nblue hair, "hi"', "seed": 5}
        plan = _plan(state, ("prompt_sr", '"1girl,\nblue hair"\n"1boy, ""yo"""'))
        self.assertEqual(plan.errors, [])
        self.assertEqual(_prompts(plan), ['1girl,\nblue hair, "hi"', '1boy, "yo", "hi"'])

    def test_an_empty_replacement_takes_the_text_out(self):
        plan = _plan(STATE, ("prompt_sr", '", long hair"\n""'))
        self.assertEqual(plan.errors, [])
        self.assertEqual(_prompts(plan), ["@a, 1girl, blue hair, long hair, smile", "@a, 1girl, blue hair, smile"])

    def test_an_empty_search_text_aborts_the_plan(self):
        """Forge would put the replacement between every two characters; a block left empty must not do that."""
        for text in ('""\nred hair', '""\n""', ", red hair"):
            with self.subTest(text=text):
                plan = _plan(STATE, ("prompt_sr", text))
                self.assertEqual(plan.errors, ["Prompt S/R: the first value (the text to search for) is empty"])
                self.assertEqual(plan.cells, [])

    def test_two_sr_axes_of_whole_values(self):
        plan = _plan(STATE, ("prompt_sr", '"blue hair, long hair"\n"red hair, short hair"'), ("prompt_sr", '"bad hands, lowres"\n"worst quality, blurry"'))
        self.assertEqual(plan.errors, [])
        self.assertEqual(len(plan.cells), 4)
        last = plan.cells[-1].state
        self.assertEqual((last["prompt"], last["negative"]), ("@a, 1girl, red hair, short hair, smile", "worst quality, blurry"))


class PromptOrderWholeValueTests(unittest.TestCase):
    def test_a_quoted_piece_moves_as_one(self):
        plan = _plan(STATE, ("prompt_order", '"blue hair, long hair"\nsmile'))
        self.assertEqual(plan.errors, [])
        self.assertEqual(_prompts(plan), ["@a, 1girl, blue hair, long hair, smile", "@a, 1girl, smile, blue hair, long hair"])

    def test_empty_blocks_are_skipped(self):
        plan = _plan(STATE, ("prompt_order", 'smile\n""\n1girl'))
        self.assertEqual(plan.errors, [])
        self.assertEqual(len(plan.cells), 2)


class BlocksTextTests(unittest.TestCase):
    """The value box the blocks write (golden rows marked ``joined``) and what is saved with the image."""

    def test_the_golden_has_values_of_several_tags(self):
        rows = [row for row in json.loads(GOLDEN.read_text(encoding="utf-8")) if row.get("joined")]
        several = [row for row in rows if any("," in field for field in row["fields"])]
        self.assertGreaterEqual(len(several), 5)
        for row in rows:
            with self.subTest(text=row["text"]):
                self.assertEqual(xyz_values.parse_axis_values("str", row["text"]), row["fields"])

    def test_the_value_box_survives_the_infotext(self):
        text = '"blue hair, long hair"\n"say ""hi"": ok"\n"x\ny"\nplain'
        info = build_infotext(xp.normalize_state({"prompt": "1girl"}), seed=5, extra={"X Type": "Prompt S/R", "X Values": text})
        self.assertEqual(_params(info.splitlines()[-1])["X Values"], text)


if __name__ == "__main__":
    unittest.main()
