"""X/Y/Z value syntax (Forge-compatible ranges, CSV splitting, permutations)."""
import itertools
import unittest

from comfy_custom_nodes.ai_studio_forge_parity import xyz_values as xv


def _parse(kind, text):
    return xv.parse_axis_values(kind, text)


class SplitValuesTests(unittest.TestCase):
    def test_commas_and_blanks(self):
        self.assertEqual(xv.split_values("a,,b"), ["a", "", "b"])

    def test_quotes_protect_commas(self):
        self.assertEqual(xv.split_values('a, "b, c"'), ["a", "b, c"])

    def test_doubled_quote_is_a_literal_quote(self):
        self.assertEqual(xv.split_values('"say ""hi""", x'), ['say "hi"', "x"])

    def test_newlines_separate(self):
        self.assertEqual(xv.split_values("a\nb"), ["a", "b"])
        self.assertEqual(xv.split_values("a, b\r\nc"), ["a", "b", "c"])

    def test_strips_each_field(self):
        self.assertEqual(xv.split_values("  a  ,\tb "), ["a", "b"])

    def test_empty_text(self):
        self.assertEqual(xv.split_values(""), [])


class IntValueTests(unittest.TestCase):
    def test_inclusive_range(self):
        self.assertEqual(_parse("int", "1-5"), [1, 2, 3, 4, 5])

    def test_signed_step(self):
        self.assertEqual(_parse("int", "1-10 (+3)"), [1, 4, 7, 10])
        self.assertEqual(_parse("int", "0-9 (+4)"), [0, 4, 8])

    def test_negative_start(self):
        self.assertEqual(_parse("int", "-2-2"), [-2, -1, 0, 1, 2])

    def test_negative_step_loses_the_end_like_forge(self):
        self.assertEqual(_parse("int", "10-1 (-3)"), [10, 7, 4])
        self.assertEqual(_parse("int", "5-1 (-1)"), [5, 4, 3])

    def test_descending_without_step_is_empty(self):
        self.assertEqual(_parse("int", "5-1"), [])

    def test_count_syntax_truncates(self):
        self.assertEqual(_parse("int", "1-10 [5]"), [1, 3, 5, 7, 10])
        self.assertEqual(_parse("int", "0-100 [3]"), [0, 50, 100])
        self.assertEqual(_parse("int", "10-0 [3]"), [10, 5, 0])

    def test_plain_values_and_mixing(self):
        self.assertEqual(_parse("int", "-1, 5, 2-4"), [-1, 5, 2, 3, 4])

    def test_blank_fields_are_skipped(self):
        self.assertEqual(_parse("int", "1,, 2,\n"), [1, 2])
        self.assertEqual(_parse("int", ""), [])

    def test_spaces_inside_numbers(self):
        self.assertEqual(_parse("int", "- 2 - 2"), [-2, -1, 0, 1, 2])

    def test_garbage_raises(self):
        for text in ("abc", "1.5", "1-5 (3)", "1-5 (+0)"):
            with self.subTest(text=text):
                with self.assertRaises(ValueError):
                    _parse("int", text)


class FloatValueTests(unittest.TestCase):
    def assertFloats(self, got, want):
        self.assertEqual(len(got), len(want), got)
        for a, b in zip(got, want):
            self.assertAlmostEqual(a, b, places=9)
            self.assertIsInstance(a, float)

    def test_step_range(self):
        self.assertFloats(_parse("float", "1-2 (+0.25)"), [1.0, 1.25, 1.5, 1.75, 2.0])
        self.assertFloats(_parse("float", "1-3"), [1.0, 2.0, 3.0])

    def test_descending(self):
        self.assertFloats(_parse("float", "1-0 (-0.25)"), [1.0, 0.75, 0.5, 0.25, 0.0])

    def test_float_overshoot_matches_forge(self):
        got = _parse("float", "0.1-0.3 (+0.1)")
        self.assertFloats(got, [0.1, 0.2, 0.3, 0.4])

    def test_count_syntax(self):
        self.assertFloats(_parse("float", "0-1 [5]"), [0.0, 0.25, 0.5, 0.75, 1.0])
        self.assertFloats(_parse("float", "3-7 [3]"), [3.0, 5.0, 7.0])
        self.assertFloats(_parse("float", "2-9 [1]"), [2.0])
        self.assertEqual(_parse("float", "2-9 [0]"), [])

    def test_exponent_is_one_number_not_a_range(self):
        self.assertFloats(_parse("float", "1e-3"), [0.001])

    def test_plain_values(self):
        self.assertFloats(_parse("float", "3, .5, -2.5"), [3.0, 0.5, -2.5])

    def test_blank_skipped_and_garbage_raises(self):
        self.assertFloats(_parse("float", "1,,2"), [1.0, 2.0])
        for text in ("abc", "nan", "inf", "1-2 (0.5)", "1-2 (+0)"):
            with self.subTest(text=text):
                with self.assertRaises(ValueError):
                    _parse("float", text)


class StringKindsTests(unittest.TestCase):
    def test_str_keeps_blanks(self):
        self.assertEqual(_parse("str", "a,,b"), ["a", "", "b"])
        self.assertEqual(_parse("str", 'a, "b, c"'), ["a", "b, c"])
        self.assertEqual(_parse("str", "a\nb"), ["a", "b"])

    def test_choice_is_text_values(self):
        self.assertEqual(_parse("choice", "euler, dpmpp_2m"), ["euler", "dpmpp_2m"])

    def test_nothing_is_a_single_ignored_value(self):
        self.assertEqual(len(_parse("nothing", "")), 1)
        self.assertEqual(_parse("nothing", "whatever, 1-5"), _parse("nothing", ""))


class PermutationTests(unittest.TestCase):
    def test_all_orders_in_itertools_order(self):
        got = _parse("permutations", "a, b, c")
        self.assertEqual(len(got), 6)
        self.assertEqual([tuple(x) for x in got], list(itertools.permutations(["a", "b", "c"])))

    def test_single_token(self):
        self.assertEqual([tuple(x) for x in _parse("permutations", "a")], [("a",)])

    def test_empty(self):
        self.assertEqual(_parse("permutations", ""), [])

    def test_blank_tokens_dropped(self):
        self.assertEqual(len(_parse("permutations", "a,,b")), 2)


class SizeTests(unittest.TestCase):
    def test_width_by_height(self):
        self.assertEqual(_parse("size", "512x768, 832 x 1216"), [(512, 768), (832, 1216)])

    def test_bad_values_raise(self):
        for text in ("512X768", "512", "512x", "1x2x3", "axb"):
            with self.subTest(text=text):
                with self.assertRaises(ValueError):
                    _parse("size", text)


class BoolTests(unittest.TestCase):
    def test_words(self):
        self.assertEqual(_parse("bool", "True, false, 1, 0, ON, off"),
                         [True, False, True, False, True, False])

    def test_blank_skipped_and_garbage_raises(self):
        self.assertEqual(_parse("bool", "true,,false"), [True, False])
        with self.assertRaises(ValueError):
            _parse("bool", "maybe")


class UnknownKindTests(unittest.TestCase):
    def test_unknown_kind_raises(self):
        with self.assertRaises(ValueError):
            _parse("complex", "1")


if __name__ == "__main__":
    unittest.main()
