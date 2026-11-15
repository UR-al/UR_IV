"""parse_lora_path must agree with the JS parseLoraPath on the shared golden table."""
import json
import unittest
from pathlib import Path

from comfy_custom_nodes.ai_studio_forge_parity import lora_index

GOLDEN = (Path(__file__).resolve().parent.parent / "comfy_custom_nodes" / "ai_studio_forge_parity"
          / "js_tests" / "fixtures" / "lora_names_golden.json")


class LoraNamesGoldenTests(unittest.TestCase):
    def test_every_golden_row_matches(self):
        rows = json.loads(GOLDEN.read_text(encoding="utf-8"))
        self.assertGreaterEqual(len(rows), 30)
        for row in rows:
            want = row["expect"]
            got = lora_index.parse_lora_path(row["entry"])
            with self.subTest(entry=row["entry"]):
                self.assertEqual(got["group"], want["group"])
                self.assertEqual(got["run"], want["run"])
                self.assertEqual(got["run_short"], want["runShort"])
                self.assertEqual(got["epoch"], want["epoch"])
                self.assertEqual(got["final"], want["final"])
                self.assertEqual(got["downloaded"], want["downloaded"])


if __name__ == "__main__":
    unittest.main()
