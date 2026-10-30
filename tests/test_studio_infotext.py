"""Forge-style ``parameters`` text for the Anima studio (labels, key order, app-reader round trip)."""
import unittest

from comfy_custom_nodes.ai_studio_forge_parity import studio_infotext
from comfy_custom_nodes.ai_studio_forge_parity.studio_state import normalize_state


def _state(**changes):
    raw = {"prompt": "1girl, solo", "negative": "bad hands", "unet": "anima-preview.safetensors",
           "vae": "qwen_image_vae.safetensors"}
    raw.update(changes)
    return normalize_state(raw)


class InfotextLayoutTests(unittest.TestCase):
    def test_infotext_txt2img_matches_forge_layout(self):
        lines = studio_infotext.build_infotext(_state(), seed=42).split("\n")
        self.assertEqual(lines[0], "1girl, solo")
        self.assertEqual(lines[1], "Negative prompt: bad hands")
        self.assertEqual(len(lines), 3)
        self.assertTrue(lines[2].startswith(
            "Steps: 32, Sampler: EXP Heun 2 x0 SDE, Schedule type: Flow Cosmos Dynamic, CFG scale: 3, Shift: 3, "
            "Seed: 42, Size: 832x1216, Model: anima-preview, VAE: qwen_image_vae"), lines[2])
        self.assertNotIn("Denoising strength", lines[2])
        self.assertNotIn("Hires", lines[2])

    def test_no_negative_line_and_no_empty_model_keys(self):
        text = studio_infotext.build_infotext(normalize_state({"prompt": "x"}), seed=1)
        self.assertEqual(text.split("\n")[0], "x")
        self.assertNotIn("Negative prompt", text)
        self.assertNotIn("Model:", text)
        self.assertNotIn("VAE:", text)
        self.assertTrue(text.split("\n")[-1].endswith("Size: 832x1216"))

    def test_prompt_line_is_the_effective_prompt_with_lora_tags(self):
        state = _state(loras=[{"on": True, "name": "meow", "strength": 0.9}, {"on": False, "name": "off", "strength": 1}])
        self.assertEqual(studio_infotext.build_infotext(state, seed=1).split("\n")[0], "1girl, solo <lora:meow:0.9>")

    def test_model_path_is_reduced_to_the_file_stem(self):
        text = studio_infotext.build_infotext(_state(unet="sub\dir/anima.v2.safetensors"), seed=1)
        self.assertIn("Model: anima.v2,", text)

    def test_values_with_commas_are_quoted_like_forge(self):
        text = studio_infotext.build_infotext(_state(unet="a, b.safetensors"), seed=1)
        self.assertIn('Model: "a, b",', text)

    def test_hires_keys(self):
        state = _state(hires={"on": True, "scale": 1.5, "steps": 16, "denoise": 0.4, "upscaler": "4x-UltraSharp.pth"})
        last = studio_infotext.build_infotext(state, seed=7).split("\n")[-1]
        self.assertIn("Hires upscale: 1.5, Hires steps: 16, Hires upscaler: 4x-UltraSharp", last)
        self.assertNotIn("4x-UltraSharp.pth", last)
        self.assertIn("Denoising strength: 0.4", last)
        self.assertLess(last.index("Denoising strength"), last.index("Hires upscale"))

    def test_hires_sampler_only_when_it_differs(self):
        same = _state(hires={"on": True, "sampler": "exp_heun_2_x0_sde"})
        other = _state(hires={"on": True, "sampler": "euler_ancestral"})
        self.assertNotIn("Hires sampler", studio_infotext.build_infotext(same, seed=1))
        self.assertIn("Hires sampler: Euler a", studio_infotext.build_infotext(other, seed=1))

    def test_size_comes_from_the_tab_in_use(self):
        text = studio_infotext.build_infotext(_state(tab="i2i", i2i={"width": 1248, "height": 1824}), seed=1)
        self.assertIn("Size: 1248x1824", text)

    def test_hires_off_for_i2i_and_denoise_comes_from_i2i(self):
        state = _state(tab="i2i", hires={"on": True}, i2i={"denoise": 0.55})
        last = studio_infotext.build_infotext(state, seed=1).split("\n")[-1]
        self.assertIn("Denoising strength: 0.55", last)
        self.assertNotIn("Hires", last)

    def test_extra_is_appended_at_the_end(self):
        text = studio_infotext.build_infotext(_state(), seed=1, extra={"XYZ": "cfg 4", "Note": "a, b"})
        self.assertTrue(text.endswith('XYZ: cfg 4, Note: "a, b"'))


class LabelTests(unittest.TestCase):
    def test_labels_and_unknown_names_pass_through(self):
        self.assertEqual(studio_infotext.forge_sampler_label("exp_heun_2_x0_sde"), "EXP Heun 2 x0 SDE")
        self.assertEqual(studio_infotext.forge_sampler_label("euler_ancestral"), "Euler a")
        self.assertEqual(studio_infotext.forge_sampler_label("something_new"), "something_new")
        self.assertEqual(studio_infotext.forge_scheduler_label("flow_cosmos_dynamic"), "Flow Cosmos Dynamic")
        self.assertEqual(studio_infotext.forge_scheduler_label("sgm_uniform"), "SGM Uniform")
        self.assertEqual(studio_infotext.forge_scheduler_label("something_new"), "something_new")


class PackTablesMatchAppTests(unittest.TestCase):
    """The pack cannot import ``core``; this test (app venv) keeps its tables in step with the compiler's."""

    def test_sampler_table_resolves_back_through_the_app_compiler(self):
        from core.comfy_workflow_compiler import _FORGE_SAMPLER_ALIASES, ComfyWorkflowCompiler

        for comfy_name, label in studio_infotext.SAMPLER_LABELS.items():
            with self.subTest(sampler=comfy_name):
                self.assertEqual(ComfyWorkflowCompiler._comfy_sampler(label), comfy_name)
        self.assertEqual(
            set(_FORGE_SAMPLER_ALIASES.values()) - set(studio_infotext.SAMPLER_LABELS), set())

    def test_scheduler_table_resolves_back_through_the_app_compiler(self):
        from core.comfy_workflow_compiler import _FORGE_SCHEDULER_ALIASES

        for comfy_name, label in studio_infotext.SCHEDULER_LABELS.items():
            with self.subTest(scheduler=comfy_name):
                self.assertEqual(_FORGE_SCHEDULER_ALIASES[label.lower()], comfy_name)
        self.assertEqual(
            set(_FORGE_SCHEDULER_ALIASES.values()) - set(studio_infotext.SCHEDULER_LABELS), set())

    def test_extra_sampler_and_scheduler_labels_match_the_sam_extra_tables(self):
        from core import extra_samplers, extra_schedulers

        for comfy_name, label in extra_samplers.LABEL_OF_COMFY.items():
            with self.subTest(sampler=comfy_name):
                self.assertEqual(studio_infotext.SAMPLER_LABELS[comfy_name], label)
        for comfy_name, label in extra_schedulers.SCHEDULER_LABELS.items():
            with self.subTest(scheduler=comfy_name):
                self.assertEqual(studio_infotext.SCHEDULER_LABELS[comfy_name], label)


if __name__ == "__main__":
    unittest.main()
