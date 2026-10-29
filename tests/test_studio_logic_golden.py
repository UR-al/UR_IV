"""Golden cases shared with the studio JS (``web/ai_studio/studio_logic.js``).

``js_tests/fixtures/studio_logic_golden.json`` is read by this test and by ``js_tests/studio_logic.test.mjs``
(``cd frontend && npm run test:pack``), so the JS mirror of the state defaults, the effective prompt, the unet
family rule and the infotext reader cannot drift from the Python they copy.
"""
import json
import unittest
from pathlib import Path

from comfy_custom_nodes.ai_studio_forge_parity.studio_choices import unet_family
from comfy_custom_nodes.ai_studio_forge_parity.studio_infotext import build_infotext
from comfy_custom_nodes.ai_studio_forge_parity.studio_state import DEFAULT_STATE, effective_prompt, normalize_state

FIXTURE = (Path(__file__).resolve().parent.parent / "comfy_custom_nodes" / "ai_studio_forge_parity"
           / "js_tests" / "fixtures" / "studio_logic_golden.json")


class StudioLogicGoldenTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.golden = json.loads(FIXTURE.read_text(encoding="utf-8"))

    def test_default_state_matches(self):
        self.assertEqual(self.golden["default_state"], DEFAULT_STATE)

    def test_effective_prompt_cases(self):
        for case in self.golden["effective_prompt"]:
            with self.subTest(state=case["state"]):
                self.assertEqual(effective_prompt(normalize_state(case["state"])), case["expected"])

    def test_unet_family_cases(self):
        for case in self.golden["unet_family"]:
            with self.subTest(name=case["name"]):
                self.assertEqual(unet_family(case["name"]), case["expected"])

    def test_infotext_text(self):
        case = self.golden["infotext"]
        self.assertEqual(build_infotext(normalize_state(case["state"]), seed=case["seed"]), case["text"])


if __name__ == "__main__":
    unittest.main()
