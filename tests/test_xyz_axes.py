"""X/Y/Z axis list: Forge labels, order, costs, and state-field wiring."""
import unittest

from comfy_custom_nodes.ai_studio_forge_parity import studio_state as ss
from comfy_custom_nodes.ai_studio_forge_parity import xyz_axes as xa
from comfy_custom_nodes.ai_studio_forge_parity import xyz_values as xv

FORGE_LABELS = [
    "Nothing", "Seed", "Steps", "Hires steps", "Size", "CFG Scale", "Shift", "Prompt S/R", "Prompt order",
    "Sampler", "Hires sampler", "Schedule type", "Checkpoint name", "VAE", "Denoising", "Hires upscaler",
    "[Modules] Text Encoder", "[DoRA] Inference mode", "[SAM3] Enable", "[SAM3] Checkpoint",
    "[SAM3] Detect Prompt", "[SAM3] Denoising Strength", "[SAM3] Steps", "[SAM3] CFG Scale",
]
CHOICE_KINDS = {"samplers", "schedulers", "unets", "vaes", "tes", "upscalers", "sam3", "dora_modes"}


def _has_path(state: dict, path: str) -> bool:
    node = state
    for part in path.split("."):
        if not isinstance(node, dict) or part not in node:
            return False
        node = node[part]
    return True


class AxisListTests(unittest.TestCase):
    def test_axis_labels_match_forge(self):
        self.assertEqual([a.label for a in xa.AXES], FORGE_LABELS)

    def test_ids_are_unique(self):
        ids = [a.id for a in xa.AXES]
        self.assertEqual(len(ids), len(set(ids)))

    def test_costs(self):
        costs = {a.label: a.cost for a in xa.AXES}
        expected = {
            "Checkpoint name": 2.0, "[Modules] Text Encoder": 0.9, "[SAM3] Checkpoint": 0.9,
            "[DoRA] Inference mode": 0.8, "VAE": 0.7, "Prompt S/R": 0.1, "Prompt order": 0.1,
        }
        for label in FORGE_LABELS:
            with self.subTest(label=label):
                self.assertEqual(costs[label], expected.get(label, 0))

    def test_every_axis_field_exists_in_default_state(self):
        state = ss.normalize_state({})
        for axis in xa.AXES:
            if axis.kind == "nothing":
                self.assertEqual(axis.field, "")
                continue
            for path in axis.field.split(","):
                with self.subTest(axis=axis.label, path=path):
                    self.assertTrue(_has_path(state, path))

    def test_ruling_r4_fields(self):
        by = {a.label: a.field for a in xa.AXES}
        self.assertEqual(by["Size"], "width,height")
        self.assertEqual(by["Prompt S/R"], "prompt")
        self.assertEqual(by["Prompt order"], "prompt")
        self.assertEqual(by["Hires sampler"], "hires.sampler")
        self.assertEqual(by["[SAM3] Steps"], "sam3.steps")
        self.assertEqual(by["[SAM3] CFG Scale"], "sam3.cfg")
        self.assertEqual(by["[DoRA] Inference mode"], "dora_mode")
        self.assertEqual(by["Denoising"], "i2i.denoise")
        self.assertEqual(by["Checkpoint name"], "unet")
        self.assertEqual(by["[Modules] Text Encoder"], "te")
        self.assertEqual(by["Schedule type"], "scheduler")

    def test_kinds_are_parseable_and_choices_are_known_routes(self):
        for axis in xa.AXES:
            with self.subTest(axis=axis.label):
                self.assertIn(axis.kind, xv.KINDS)
                self.assertIn(axis.fmt, ("label_value", "value", "basename", "joined", "empty"))
                if axis.kind == "choice":
                    self.assertIn(axis.choices, CHOICE_KINDS)
                else:
                    self.assertIsNone(axis.choices)

    def test_formats(self):
        by = {a.label: a for a in xa.AXES}
        self.assertEqual(by["Nothing"].fmt, "empty")
        self.assertEqual(by["Prompt S/R"].fmt, "value")
        self.assertEqual(by["Sampler"].fmt, "value")
        self.assertEqual(by["Checkpoint name"].fmt, "basename")
        self.assertEqual(by["[SAM3] Checkpoint"].fmt, "basename")
        self.assertEqual(by["Prompt order"].fmt, "joined")
        self.assertEqual(by["Seed"].fmt, "label_value")

    def test_axis_by_id(self):
        for axis in xa.AXES:
            self.assertIs(xa.axis_by_id(axis.id), axis)
        with self.assertRaises(KeyError):
            xa.axis_by_id("nope")

    def test_sample_values_parse_for_every_kind(self):
        samples = {"int": "1", "float": "1.5", "str": "a", "choice": "a", "permutations": "a, b",
                   "size": "512x512", "nothing": "", "bool": "true"}
        for axis in xa.AXES:
            with self.subTest(axis=axis.label):
                self.assertTrue(xv.parse_axis_values(axis.kind, samples[axis.kind]))


if __name__ == "__main__":
    unittest.main()
