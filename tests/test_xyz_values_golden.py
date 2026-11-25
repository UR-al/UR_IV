"""split_values must agree with the JS splitValues (web/ai_studio/xyz_chips.js) on the shared golden table.

The XYZ panel draws a list axis' value box as blocks: it reads the text with ``splitValues`` and writes it back one value a
line.  The planner reads the same text with ``split_values``, so the two must split it alike.
"""
import json
import unittest
from pathlib import Path

from comfy_custom_nodes.ai_studio_forge_parity import xyz_values

GOLDEN = (Path(__file__).resolve().parent.parent / "comfy_custom_nodes" / "ai_studio_forge_parity"
          / "js_tests" / "fixtures" / "xyz_values_golden.json")


class XyzValuesGoldenTests(unittest.TestCase):
    def test_every_golden_row_matches(self):
        rows = json.loads(GOLDEN.read_text(encoding="utf-8"))
        self.assertGreaterEqual(len(rows), 40)
        for row in rows:
            with self.subTest(text=row["text"]):
                self.assertEqual(xyz_values.split_values(row["text"]), row["fields"])

    def test_the_blocks_text_parses_as_a_choice_axis(self):
        """A row the blocks wrote (``joined``) gives the planner the same values back."""
        rows = [row for row in json.loads(GOLDEN.read_text(encoding="utf-8")) if row.get("joined")]
        self.assertGreaterEqual(len(rows), 5)
        for row in rows:
            with self.subTest(text=row["text"]):
                self.assertEqual(xyz_values.parse_axis_values("choice", row["text"]), row["fields"])


if __name__ == "__main__":
    unittest.main()
