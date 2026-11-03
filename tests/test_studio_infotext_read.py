"""Reading ``parameters`` text back into part of a studio state, for the EXIF panel's 설정 가져오기."""
import unittest

from comfy_custom_nodes.ai_studio_forge_parity import studio_infotext, studio_routes
from comfy_custom_nodes.ai_studio_forge_parity.studio_infotext_read import parse_infotext
from comfy_custom_nodes.ai_studio_forge_parity.studio_state import normalize_state

FORGE_LINE = (
    r'Steps: 50, Sampler: ER SDE, Schedule type: Beta57 (RES4LYF), CFG scale: 5.5, Shift: 3.0, '
    r'Seed: 113, Size: 2048x1144, Model: Anima-3.8B-v1.1, Model hash: 0123456789, '
    r'Module 1: vae_a, Module 2: te_b, RNG: CPU, SAM3 Enable: True, SAM3 Prompt: "face, eyes, hair", '
    r'Lora hashes: "style_a: 0x0000000001, style_b: 0x0000000002", Version: neo-2.29')


def _built(**changes):
    raw = {"prompt": "1girl, solo", "negative": "bad hands", "unet": "sub/anima.safetensors", "vae": "qwen_vae.safetensors"}
    raw.update(changes)
    return parse_infotext(studio_infotext.build_infotext(normalize_state(raw), seed=7))


class ForgeImageTests(unittest.TestCase):
    def setUp(self):
        self.read = parse_infotext(f"1girl, solo\nsecond line\nNegative prompt: lowres,\nbad hands\n{FORGE_LINE}")
        self.settings = self.read["settings"]

    def test_prompts_keep_their_lines(self):
        self.assertEqual(self.settings["prompt"], "1girl, solo\nsecond line")
        self.assertEqual(self.settings["negative"], "lowres,\nbad hands")

    def test_numbers_and_size(self):
        self.assertEqual((self.settings["steps"], self.settings["cfg"], self.settings["shift"]), (50, 5.5, 3.0))
        self.assertEqual((self.settings["width"], self.settings["height"]), (2048, 1144))

    def test_labels_become_comfy_names(self):
        self.assertEqual(self.settings["sampler"], "er_sde")
        self.assertEqual(self.settings["scheduler"], "beta57")  # Forge adds where the scheduler comes from

    def test_files_stay_stems_for_the_panel_to_match(self):
        self.assertEqual(self.read["model"], "Anima-3.8B-v1.1")
        self.assertEqual(self.read["modules"], ["vae_a", "te_b"])

    def test_the_prompt_holds_the_loras_so_the_boxes_and_list_empty(self):
        self.assertEqual(self.settings["loras"], [])
        self.assertEqual(self.settings["prompt_artist"], "")

    def test_no_hires_keys_turn_hires_off(self):
        self.assertEqual(self.settings["hires"], {"on": False})

    def test_seed_is_not_a_setting(self):
        self.assertNotIn("seed", self.settings)

    def test_an_unknown_label_stays_for_the_panel_to_report(self):
        read = parse_infotext("x\nSteps: 20, Sampler: Some New Sampler, Schedule type: Odd (pack)")
        self.assertEqual(read["settings"]["sampler"], "Some New Sampler")
        self.assertEqual(read["settings"]["scheduler"], "Odd (pack)")


class NotAnInfotextTests(unittest.TestCase):
    def test_text_without_a_steps_line_has_no_settings(self):
        for text in ("", "just a comment", "Steps: many", None, 3):
            self.assertIsNone(parse_infotext(text), text)

    def test_only_the_steps_line(self):
        read = parse_infotext("Steps: 20, Seed: 5")
        self.assertEqual(read["settings"]["prompt"], "")
        self.assertEqual(read["settings"]["negative"], "")
        self.assertEqual(read["settings"]["steps"], 20)

    def test_windows_line_ends(self):
        self.assertEqual(parse_infotext("a\r\nNegative prompt: b\r\nSteps: 20")["settings"]["negative"], "b")


class StudioRoundTripTests(unittest.TestCase):
    """What ``build_infotext`` writes comes back."""

    def test_main_settings(self):
        read = _built(sampler="euler", scheduler="simple", steps=20, cfg=4.5, shift=2.0, width=1024, height=768)
        settings = read["settings"]
        self.assertEqual(settings["prompt"], "1girl, solo")
        self.assertEqual(settings["negative"], "bad hands")
        self.assertEqual((settings["sampler"], settings["scheduler"]), ("euler", "simple"))
        self.assertEqual((settings["steps"], settings["cfg"], settings["shift"]), (20, 4.5, 2.0))
        self.assertEqual((settings["width"], settings["height"]), (1024, 768))
        self.assertEqual(read["model"], "anima")
        self.assertEqual(read["modules"], ["qwen_vae"])

    def test_list_loras_stay_in_the_prompt_text(self):
        read = _built(loras=[{"on": True, "name": "meow", "strength": 0.9}])
        self.assertEqual(read["settings"]["prompt"], "1girl, solo <lora:meow:0.9>")

    def test_hires(self):
        hires = {"on": True, "scale": 1.5, "steps": 16, "denoise": 0.4, "upscaler": "4x-UltraSharp.pth", "cfg": 4.0, "shift": 2.0,
                 "sampler": "euler", "scheduler": "simple", "vae": "other.safetensors", "prompt": "a, b", "negative": "c"}
        got = _built(hires=hires)["settings"]["hires"]
        self.assertEqual(got, {"on": True, "scale": 1.5, "steps": 16, "denoise": 0.4, "upscaler": "4x-UltraSharp", "cfg": 4.0, "shift": 2.0,
                               "sampler": "euler", "scheduler": "simple", "vae": "other", "prompt": "a, b", "negative": "c",
                               "resize_width": 0, "resize_height": 0})

    def test_hires_keys_the_image_left_out_clear_the_panel(self):
        got = _built(hires={"on": True, "scale": 2.0, "steps": 8, "denoise": 0.3, "upscaler": "latent:bicubic"})["settings"]["hires"]
        self.assertEqual(got["upscaler"], "latent:bicubic")
        self.assertEqual((got["cfg"], got["shift"], got["sampler"], got["vae"], got["prompt"]), (0, 0, "", "", ""))

    def test_hires_to_a_size(self):
        state = {"on": True, "scale": 1.5, "steps": 8, "denoise": 0.3, "upscaler": "x.pth", "resize_width": 1536, "resize_height": 2048,
                 "size_step": 64}
        got = _built(hires=state)["settings"]["hires"]
        self.assertEqual((got["resize_width"], got["resize_height"]), (1536, 2048))
        self.assertNotIn("scale", got)

    def test_an_i2i_image_fills_the_i2i_tab_and_leaves_the_t2i_size(self):
        settings = _built(tab="i2i", i2i={"denoise": 0.55, "width": 640, "height": 960})["settings"]
        self.assertEqual(settings["i2i"], {"denoise": 0.55, "width": 640, "height": 960})
        self.assertNotIn("width", settings)
        self.assertEqual(settings["hires"], {"on": False})

    def test_inpaint(self):
        i2i = {"denoise": 0.5, "width": 640, "height": 960, "mask_blur": 8, "inpaint_area": "masked", "inpaint_padding": 48, "inpaint_fill": "fill"}
        settings = _built(tab="i2i", i2i_mode="inpaint", i2i=i2i)["settings"]
        self.assertEqual(settings["i2i"], i2i)

    def test_whole_picture_inpaint(self):
        settings = _built(tab="i2i", i2i_mode="inpaint", i2i={"denoise": 0.5, "inpaint_area": "whole"})["settings"]
        self.assertEqual(settings["i2i"]["inpaint_area"], "whole")

    def test_wildcard_templates_win_over_the_drawn_text(self):
        read = _built(prompt="a red hat", prompt_template="a __color__ hat", negative="ugly", negative_template="{ugly|bad}")
        self.assertEqual(read["settings"]["prompt"], "a __color__ hat")
        self.assertEqual(read["settings"]["negative"], "{ugly|bad}")

    def test_attention(self):
        self.assertEqual(_built(attention="sage")["settings"]["attention"], "sage")
        self.assertNotIn("attention", _built()["settings"])
        self.assertNotIn("attention", parse_infotext("x\nSteps: 20, Attention: nonsense")["settings"])


class RouteTests(unittest.TestCase):
    def test_the_route_answers_the_parsed_settings(self):
        status, payload = studio_routes.handle_infotext_parse({"text": "a\nSteps: 20"})
        self.assertEqual(status, 200)
        self.assertEqual(payload["settings"]["steps"], 20)

    def test_text_without_settings_says_so(self):
        self.assertEqual(studio_routes.handle_infotext_parse({"text": "a"}), (200, {"settings": None}))

    def test_a_wrong_body_is_an_error(self):
        for body in (None, {}, {"text": 3}):
            self.assertEqual(studio_routes.handle_infotext_parse(body)[0], 400, body)


if __name__ == "__main__":
    unittest.main()
