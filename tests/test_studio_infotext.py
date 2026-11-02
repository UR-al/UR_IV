"""Forge-style ``parameters`` text for the Anima studio (labels, key order, app-reader round trip)."""
import unittest

from comfy_custom_nodes.ai_studio_forge_parity import studio_infotext
from comfy_custom_nodes.ai_studio_forge_parity.studio_state import normalize_state


def _state(**changes):
    raw = {"prompt": "1girl, solo", "negative": "bad hands", "unet": "anima-preview.safetensors",
           "vae": "qwen_image_vae.safetensors"}
    raw.update(changes)
    return normalize_state(raw)


class PromptBoxTests(unittest.TestCase):
    def test_the_first_line_is_the_joined_prompt(self):
        state = _state(prompt_prefix="best", prompt_artist="@a", prompt="1boy", prompt_suffix="highres")
        self.assertEqual(studio_infotext.build_infotext(state, seed=1).split("\n")[0], "best, @a, 1boy, highres")


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

    def test_hires_settings_of_its_own_are_written_only_when_set(self):
        plain = studio_infotext.build_infotext(_state(hires={"on": True}), seed=1)
        for key in ("Hires CFG Scale", "Hires Shift", "Hires schedule type", "Hires prompt", "Hires Module 1", "Hires resize"):
            self.assertNotIn(key, plain)
        own = studio_infotext.build_infotext(_state(hires={
            "on": True, "cfg": 5, "shift": 4, "scheduler": "karras", "prompt": "a, b", "negative": "c", "vae": "x/y.safetensors",
        }), seed=1)
        self.assertIn("Hires CFG Scale: 5", own)
        self.assertIn("Hires Shift: 4", own)
        self.assertIn("Hires schedule type: Karras", own)
        self.assertIn('Hires prompt: "a, b"', own)
        self.assertIn("Hires negative prompt: c", own)
        self.assertIn("Hires Module 1: y", own)

    def test_hires_steps_0_means_the_main_steps_and_is_left_out(self):
        self.assertNotIn("Hires steps", studio_infotext.build_infotext(_state(hires={"on": True, "steps": 0}), seed=1))

    def test_hires_resize_replaces_the_scale(self):
        text = studio_infotext.build_infotext(_state(hires={"on": True, "resize_width": 1024, "size_step": 8}), seed=1)
        self.assertIn("Hires resize: 1024x1496", text)
        self.assertNotIn("Hires upscale:", text)

    def test_hires_resize_is_the_size_the_node_makes_on_moss(self):
        text = studio_infotext.build_infotext(
            _state(unet="moss-image.safetensors", hires={"on": True, "resize_width": 1000, "resize_height": 1000, "size_step": 8}), seed=1)
        self.assertIn("Hires resize: 992x992", text)

    def test_sam3_writes_forge_keys_only_when_on(self):
        self.assertNotIn("SAM3", studio_infotext.build_infotext(_state(), seed=1))
        text = studio_infotext.build_infotext(_state(sam3={"on": True, "prompt": "face", "threshold": 0.25, "checkpoint": r"C:\m\sam3.pt"}), seed=1)
        self.assertIn("SAM3 Enable: True, SAM3 Mode: Inpaint, SAM3 Mask Mode: Individual, SAM3 Prompt: face, SAM3 Threshold: 0.25, SAM3 Checkpoint: sam3.pt, SAM3 Device: auto", text)
        self.assertNotIn("SAM3 Mask Blur", text)
        self.assertNotIn("SAM3 Use", text)
        self.assertNotIn("C:", text.split("Size:")[1])

    def test_sam3_writes_forge_switch_beside_each_value_of_its_own(self):
        text = studio_infotext.build_infotext(_state(sam3={
            "on": True, "steps": 12, "cfg": 5, "sampler": "euler", "scheduler": "karras", "seed": 3141592653, "noise_multiplier": 1.5,
            "use_custom_size": True, "custom_width": 768, "custom_height": 640, "overlay": True,
        }), seed=1)
        for part in ("SAM3 Use Separate Steps: True, SAM3 Steps: 12", "SAM3 Use Separate CFG Scale: True, SAM3 CFG Scale: 5",
                     "SAM3 Use Separate Sampler: True, SAM3 Sampler: Euler", "SAM3 Use Separate Scheduler: True, SAM3 Scheduler: Karras",
                     "SAM3 Use Seed: True, SAM3 Seed: 3141592653", "SAM3 Use Noise Multiplier: True, SAM3 Noise Multiplier: 1.5",
                     "SAM3 Use Inpaint Width Height: True, SAM3 Inpaint Width: 768, SAM3 Inpaint Height: 640", "SAM3 Preview Overlay: True"):
            self.assertIn(part, text)

    def test_sam3_controlnet_and_face_crop_keys(self):
        text = studio_infotext.build_infotext(_state(sam3={
            "on": True, "cn_enable": True, "cn_model": "x/lllite.safetensors", "cn_strength": 0.8, "face_crop": True,
            "face_crop_model": r"C:\m\face_yolov8n.pt", "face_crop_confidence": 0.25,
        }), seed=1)
        self.assertIn("SAM3 CN Enable: True, SAM3 CN Model: lllite, SAM3 CN Module: inpaint_only, SAM3 CN Weight: 0.8", text)
        self.assertIn("SAM3 Face Crop: True, SAM3 Face Crop Model: face_yolov8n.pt, SAM3 Face Crop Confidence: 0.25", text)

    def test_latent_upscalers_carry_forge_names(self):
        text = lambda name: studio_infotext.build_infotext(_state(hires={"on": True, "upscaler": name}), seed=1)
        self.assertIn("Hires upscaler: Latent (bicubic antialiased)", text("latent:bicubic antialiased"))
        self.assertIn("Hires upscaler: Latent,", text("latent:bilinear") + ",")
        self.assertIn('Hires upscaler: "latent:bislerp"', text("latent:bislerp"))

    def test_upscale_writes_forge_extras_keys(self):
        self.assertNotIn("Postprocess", studio_infotext.build_infotext(_state(upscale={"on": True}), seed=1))
        by = studio_infotext.build_infotext(_state(upscale={"on": True, "model": "4x.pth", "scale": 2, "max_side": 4096}), seed=1)
        self.assertIn("Postprocess upscale by: 2, Max side length: 4096, Postprocess upscaler: 4x", by)
        to = studio_infotext.build_infotext(_state(upscale={
            "on": True, "model": "4x.pth", "mode": "Scale to", "width": 2048, "height": 1024, "model_2": "2x.pth", "visibility": 0.5,
        }), seed=1)
        self.assertIn("Postprocess upscale to: 2048x1024, Postprocess crop to: 2048x1024, Postprocess upscaler: 4x, Postprocess upscaler 2: 2x", to)

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


class StudioToolsTests(unittest.TestCase):
    def line(self, **changes):
        return studio_infotext.build_infotext(_state(**changes), seed=1).split("\n")[-1]

    def test_only_masked_writes_forges_keys(self):
        line = self.line(tab="i2i", i2i_mode="inpaint", i2i={"inpaint_area": "masked", "inpaint_padding": 40, "mask_blur": 6})
        self.assertIn("Mask blur: 6, Inpaint area: Only masked, Masked area padding: 40, Masked content: original", line)

    def test_a_whole_picture_inpaint_writes_no_area(self):
        line = self.line(tab="i2i", i2i_mode="inpaint")
        self.assertIn("Mask blur: 4, Masked content: original", line)
        self.assertNotIn("Inpaint area", line)

    def test_img2img_has_no_inpaint_keys(self):
        self.assertNotIn("Mask blur", self.line(tab="i2i", i2i={"inpaint_area": "masked"}))

    def test_attention_is_written_when_it_was_picked(self):
        self.assertNotIn("Attention", self.line())
        self.assertIn("Attention: sage", self.line(attention="sage"))
        self.assertIn("Attention: pytorch", self.line(forge_like={"master": True}))

    def test_wildcard_templates_use_dynamic_prompts_keys(self):
        line = self.line(prompt_template="__a__, x", negative_template="__b__")
        self.assertIn('Template: "__a__, x", Negative Template: __b__', line)


if __name__ == "__main__":
    unittest.main()
