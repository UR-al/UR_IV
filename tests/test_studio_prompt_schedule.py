"""studio_prompt_schedule.schedule_prompt — Forge Neo's prompt editing ([a:b:w], [a|b]) as (end step, text) segments.

Expected values are what Forge Neo's ``modules/prompt_parser.get_learned_conditioning_prompt_schedules`` returns for
the same prompt and step count (its doctests plus the brief's cases); the pack restates the behaviour, it carries no
Forge code.
"""
import unittest

from comfy_custom_nodes.ai_studio_forge_parity.studio_prompt_schedule import schedule_prompt


class BriefCases(unittest.TestCase):
    def test_fraction_switches_at_that_share_of_steps(self):
        self.assertEqual(schedule_prompt("[cat:dog:0.5]", 20), [(10, "cat"), (20, "dog")])

    def test_integer_is_a_step_count(self):
        self.assertEqual(schedule_prompt("[cat:dog:5]", 20), [(5, "cat"), (20, "dog")])

    def test_remove_after(self):
        self.assertEqual(schedule_prompt("[cat::0.25]", 8), [(2, "cat"), (8, "")])

    def test_alternation_switches_every_step(self):
        self.assertEqual(
            schedule_prompt("[cat|dog]", 4), [(1, "cat"), (2, "dog"), (3, "cat"), (4, "dog")]
        )

    def test_no_syntax_is_one_segment(self):
        self.assertEqual(schedule_prompt("1girl, solo, (smile:1.2)", 32), [(32, "1girl, solo, (smile:1.2)")])

    def test_escaped_brackets_stay_as_written(self):
        # Forge keeps the backslashes in the schedule text; the tokenizer then reads them (both apps alike).
        self.assertEqual(schedule_prompt(r"a \[x:y:2\] b", 10), [(10, r"a \[x:y:2\] b")])
        self.assertEqual(schedule_prompt(r"\[x\]", 10), [(10, r"\[x\]")])


class ForgeBehaviour(unittest.TestCase):
    """The doctest cases of Forge's prompt parser (10 steps)."""

    def check(self, prompt, expected, steps=10):
        self.assertEqual(schedule_prompt(prompt, steps), expected)

    def test_add_after(self):
        self.check("a [b:3]", [(3, "a "), (10, "a b")])
        self.check("a [b: 3]", [(3, "a "), (10, "a b")])

    def test_emphasis_inside_a_schedule_is_kept(self):
        self.check("a [[[b]]:2]", [(2, "a "), (10, "a [[b]]")])
        self.check("[(a:2):3]", [(3, ""), (10, "(a:2)")])

    def test_whitespace_around_the_parts(self):
        self.check("a [b : c : 1] d", [(1, "a b  d"), (10, "a  c  d")])

    def test_nested(self):
        self.check("a[b:[c:d:2]:1]e", [(1, "abe"), (2, "ace"), (10, "ade")])

    def test_unbalanced_is_plain_text(self):
        self.check("a [unbalanced", [(10, "a [unbalanced")])

    def test_leading_dot_fraction(self):
        self.check("a [b:.5] c", [(5, "a  c"), (10, "a b c")])

    def test_stray_brackets_at_top_level_are_text(self):
        self.check("((a][:b:c [d:3]", [(3, "((a][:b:c "), (10, "((a][:b:c d")])

    def test_alternation_with_weights(self):
        self.check("[a|(b:1.1)]", [(i, "a" if i % 2 else "(b:1.1)") for i in range(1, 11)])

    def test_alternation_empty_options(self):
        self.check("[fe|]male", [(i, "female" if i % 2 else "male") for i in range(1, 11)])
        self.check("[fe|||]male", [(i, "female" if i % 4 == 1 else "male") for i in range(1, 11)])


class ForgeEdgeRules(unittest.TestCase):
    def test_a_number_with_a_dot_is_always_a_fraction(self):
        # Forge reads "1.5" as 150 % of the steps (clipped to all steps), not as step 1.
        self.assertEqual(schedule_prompt("[cat:dog:1.5]", 10), [(10, "cat")])
        self.assertEqual(schedule_prompt("[cat:dog:2.0]", 10), [(10, "cat")])

    def test_step_beyond_the_run_keeps_the_first_prompt(self):
        self.assertEqual(schedule_prompt("[cat:dog:50]", 20), [(20, "cat")])

    def test_zero_or_negative_means_the_second_prompt_from_the_start(self):
        self.assertEqual(schedule_prompt("[cat:dog:0]", 20), [(20, "dog")])
        self.assertEqual(schedule_prompt("[cat:dog:-3]", 20), [(20, "dog")])

    def test_a_stray_pipe_disables_all_editing(self):
        # Forge cannot parse the whole prompt and then uses it as written, schedules included.
        self.assertEqual(schedule_prompt("a [b:c:2] | d", 10), [(10, "a [b:c:2] | d")])

    def test_three_parts_is_not_a_schedule(self):
        self.assertEqual(schedule_prompt("[a:b:c:2]", 10), [(10, "[a:b:c:2]")])

    def test_bad_number_is_text(self):
        self.assertEqual(schedule_prompt("[a:b:x]", 10), [(10, "[a:b:x]")])

    def test_schedule_inside_emphasis(self):
        self.assertEqual(
            schedule_prompt("(red [cat:dog:0.5]:1.2)", 4), [(2, "(red cat:1.2)"), (4, "(red dog:1.2)")]
        )

    def test_several_schedules_share_the_step_list(self):
        self.assertEqual(
            schedule_prompt("[a:b:2] [c:d:0.5]", 8),
            [(2, "a c"), (4, "b c"), (8, "b d")],
        )

    def test_one_step(self):
        self.assertEqual(schedule_prompt("[cat:dog:0.5]", 1), [(1, "dog")])

    def test_steps_must_be_positive(self):
        with self.assertRaises(ValueError):
            schedule_prompt("a", 0)

    def test_empty_prompt(self):
        self.assertEqual(schedule_prompt("", 5), [(5, "")])


if __name__ == "__main__":
    unittest.main()
