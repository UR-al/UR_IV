"""X/Y/Z axis list: Forge labels, order, costs, and state-field wiring."""
import unittest

from comfy_custom_nodes.ai_studio_forge_parity import studio_state as ss
from comfy_custom_nodes.ai_studio_forge_parity import xyz_axes as xa
from comfy_custom_nodes.ai_studio_forge_parity import xyz_values as xv

FORGE_LABELS = [
    "Nothing", "Seed", "Steps", "Hires steps", "Size", "CFG Scale", "Shift", "Prompt S/R", "Prompt order", "LoRA",
    "Sampler", "Hires sampler", "Schedule type", "Checkpoint name", "VAE", "Denoising", "Hires upscaler",
    "[Modules] Text Encoder", "[DoRA] Inference mode", "[Attention] Backend", "[SAM3] Enable", "[SAM3] Checkpoint",
    "[SAM3] Detect Prompt", "[SAM3] Denoising Strength", "[SAM3] Steps", "[SAM3] CFG Scale",
]
CHOICE_KINDS = {"samplers", "schedulers", "unets", "vaes", "tes", "upscalers", "sam3", "dora_modes", "attentions"}


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
            "[DoRA] Inference mode": 0.8, "VAE": 0.7, "LoRA": 0.5, "[Attention] Backend": 0.3, "Prompt S/R": 0.1, "Prompt order": 0.1,
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
                   "size": "512x512", "nothing": "", "bool": "true", "lora": ""}
        for axis in xa.AXES:
            with self.subTest(axis=axis.label):
                self.assertTrue(xv.parse_axis_values(axis.kind, samples[axis.kind]))


class ExtensionAxisTests(unittest.TestCase):
    def setUp(self):
        self.by_id = {axis.id: axis for axis in xa.EXT_AXES}

    def test_ids_and_labels_are_unique_and_clear_of_forges(self):
        ids = [axis.id for axis in (*xa.AXES, *xa.EXT_AXES)]
        labels = [axis.label for axis in (*xa.AXES, *xa.EXT_AXES)]
        self.assertEqual(len(set(ids)), len(ids))
        self.assertEqual(len(set(labels)), len(labels))

    def test_every_field_is_in_the_state(self):
        state = ss.normalize_state({})
        for axis in xa.EXT_AXES:
            with self.subTest(axis=axis.id):
                self.assertIn(axis.kind, xv.KINDS)
                self.assertTrue(_has_path(state, axis.field))
                self.assertEqual(xa.axis_by_id(axis.id), axis)
                for path, _value in axis.switches:
                    self.assertTrue(_has_path(state, path), path)

    def test_every_extension_has_a_switch_axis(self):
        for item in ss.EXT_SCHEMA["extensions"]:
            self.assertEqual(self.by_id[f"ext.{item['key']}.on"].kind, "bool")

    def test_prompts_paths_and_file_lists_are_left_out(self):
        for name in ("ext.guidance.guid_mod_base_prompt", "ext.guidance.guid_mod_adapter_path", "ext.sam3_refine.inpaint_prompt",
                     "ext.lora_block.lora_name", "ext.vae_2x.vae_name"):
            self.assertNotIn(name, self.by_id)

    def test_a_text_setting_keeps_its_commas(self):
        self.assertEqual(self.by_id["ext.speed.scales"].kind, "lines")
        self.assertEqual(xv.parse_axis_values("lines", "0.5,1.0\n\n 0.25,0.5,1.0 \n"), ["0.5,1.0", "0.25,0.5,1.0"])

    def test_a_fixed_list_comes_with_the_axis(self):
        self.assertEqual(self.by_id["ext.speed.transform"].values, ("dct", "dwt", "fft"))
        self.assertEqual(self.by_id["ext.speed.transform"].kind, "choice")

    def test_a_setting_turns_on_what_it_needs(self):
        switches = lambda name: dict(self.by_id[name].switches)
        self.assertEqual(switches("ext.detail_daemon.dd_amount"), {"ext.detail_daemon.on": True})
        self.assertEqual(switches("ext.guidance.guid_scale"), {"ext.guidance.on": True, "ext.guidance.guid_enabled": True})
        self.assertEqual(switches("ext.guidance.guid_s2_scale"),
                         {"ext.guidance.on": True, "ext.guidance.guid_slg_on": True, "ext.guidance.guid_slg_mode": "Stochastic (S²)"})
        self.assertEqual(switches("ext.guidance.guid_rdc_tau"),
                         {"ext.guidance.on": True, "ext.guidance.guid_dcw_enabled": True, "ext.guidance.guid_rdc_enabled": True})

    def test_an_axis_of_a_switch_stays_the_columns_choice(self):
        self.assertEqual(dict(self.by_id["ext.guidance.guid_enabled"].switches), {"ext.guidance.on": True})
        self.assertEqual(dict(self.by_id["ext.guidance.guid_smc_enabled"].switches), {"ext.guidance.on": True})  # not SMC's other switch
        self.assertEqual(dict(self.by_id["ext.guidance.guid_rdc_enabled"].switches),
                         {"ext.guidance.on": True, "ext.guidance.guid_dcw_enabled": True})
        self.assertEqual(self.by_id["ext.skim.on"].switches, ())

    def test_every_guidance_section_with_a_switch_is_in_the_table(self):
        sections = {field["section"] for field in ss.EXT_SCHEMA["extensions"][0]["fields"]}
        self.assertEqual(sections - set(xa._SECTION_SWITCHES), {"CFG 기준"})


if __name__ == "__main__":
    unittest.main()
