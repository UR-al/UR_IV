"""Anima studio control-panel state schema: defaults, normalization, effective prompt."""
import copy
import unittest

from comfy_custom_nodes.ai_studio_forge_parity import studio_state as ss


class StudioStateTests(unittest.TestCase):
    def test_defaults_match_user_forge_settings(self):
        d = ss.DEFAULT_STATE
        self.assertEqual(d["version"], 1)
        self.assertEqual(d["sampler"], "exp_heun_2_x0_sde")
        self.assertEqual(d["scheduler"], "flow_cosmos_dynamic")
        self.assertEqual((d["steps"], d["cfg"], d["shift"]), (32, 3.0, 3.0))
        self.assertEqual((d["width"], d["height"]), (832, 1216))
        self.assertEqual(d["seed"], -1)
        self.assertEqual((d["hires"]["scale"], d["hires"]["denoise"], d["hires"]["steps"]), (1.5, 0.4, 16))
        self.assertEqual(d["hires"]["upscaler"], "4x-UltraSharp.pth")
        self.assertEqual((d["i2i"]["denoise"], d["i2i"]["mask_blur"]), (0.55, 4))
        self.assertEqual((d["sam3"]["prompt"], d["sam3"]["denoise"]), ("face", 0.4))

    def test_extra_keys_from_ruling_r2(self):
        d = ss.DEFAULT_STATE
        self.assertEqual(d["hires"]["sampler"], "")
        self.assertEqual((d["sam3"]["steps"], d["sam3"]["cfg"]), (0, 0.0))
        self.assertEqual(d["dora_mode"], "LyCORIS fp32 (as trained)")
        fl = d["forge_like"]
        self.assertTrue(all(fl[k] for k in ("master", "noise", "prompt_syntax", "dora")))
        self.assertFalse(fl["attention"])

    def test_negpip_defaults_on_and_keeps_only_booleans(self):
        self.assertIs(ss.normalize_state({})["negpip"], True)
        self.assertIs(ss.normalize_state({"negpip": False})["negpip"], False)
        self.assertIs(ss.normalize_state({"negpip": "no"})["negpip"], True)

    def test_normalize_default_is_identity_and_independent(self):
        out = ss.normalize_state({})
        self.assertEqual(out, ss.DEFAULT_STATE)
        out["hires"]["scale"] = 9
        out["loras"].append({})
        self.assertEqual(ss.DEFAULT_STATE["hires"]["scale"], 1.5)
        self.assertEqual(ss.DEFAULT_STATE["loras"], [])
        before = copy.deepcopy(ss.DEFAULT_STATE)
        self.assertEqual(ss.normalize_state(None), before)
        self.assertEqual(ss.normalize_state("junk"), before)

    def test_missing_and_bad_values_fall_back(self):
        out = ss.normalize_state({"steps": "abc", "cfg": 99, "foo": 1})
        self.assertEqual(out["steps"], 32)
        self.assertEqual(out["cfg"], 30.0)
        self.assertNotIn("foo", out)

    def test_clamping_ranges(self):
        out = ss.normalize_state({"steps": 999, "shift": -5, "width": 10, "height": 99999,
                                  "batch": 0, "seed": "-7"})
        self.assertEqual((out["steps"], out["shift"]), (150, 0.0))
        self.assertEqual((out["width"], out["height"]), (64, 4096))
        self.assertEqual(out["batch"], 1)
        self.assertEqual(out["seed"], -1)
        self.assertEqual(ss.normalize_state({"width": 835})["width"], 832)
        self.assertEqual(ss.normalize_state({"batch": 40})["batch"], 16)
        self.assertEqual(ss.normalize_state({"seed": 12345})["seed"], 12345)

    def test_nan_inf_bool_and_wrong_types_fall_back(self):
        out = ss.normalize_state({"cfg": float("nan"), "steps": True, "prompt": 5,
                                  "sampler": "", "scheduler": 3, "tab": "x", "i2i_mode": None})
        self.assertEqual(out["cfg"], 3.0)
        self.assertEqual(out["steps"], 32)
        self.assertEqual(out["prompt"], "")
        self.assertEqual(out["sampler"], "exp_heun_2_x0_sde")
        self.assertEqual(out["scheduler"], "flow_cosmos_dynamic")
        self.assertEqual((out["tab"], out["i2i_mode"]), ("t2i", "img2img"))

    def test_free_string_sampler_kept_and_enums_validated(self):
        out = ss.normalize_state({"sampler": "euler", "tab": "i2i", "i2i_mode": "inpaint",
                                  "i2i": {"inpaint_area": "whole"}})
        self.assertEqual((out["sampler"], out["tab"], out["i2i_mode"]), ("euler", "i2i", "inpaint"))
        self.assertEqual(out["i2i"]["inpaint_area"], "whole")
        self.assertEqual(ss.normalize_state({"i2i": {"inpaint_area": "x"}})["i2i"]["inpaint_area"], "whole")

    def test_nested_unknown_keys_dropped_and_partial_filled(self):
        out = ss.normalize_state({"hires": {"on": True, "scale": 2, "zzz": 1}, "forge_like": {"attention": True, "q": 1},
                                  "sam3": "bad"})
        self.assertEqual(out["hires"]["on"], True)
        self.assertEqual(out["hires"]["scale"], 2.0)
        self.assertEqual(out["hires"]["steps"], 16)
        self.assertNotIn("zzz", out["hires"])
        self.assertNotIn("q", out["forge_like"])
        self.assertTrue(out["forge_like"]["attention"] and out["forge_like"]["master"])
        self.assertEqual(out["sam3"], ss.DEFAULT_STATE["sam3"])

    def test_dora_mode_unknown_falls_back(self):
        self.assertEqual(ss.normalize_state({"dora_mode": "DoRA off (no magnitude)"})["dora_mode"],
                         "DoRA off (no magnitude)")
        self.assertEqual(ss.normalize_state({"dora_mode": "nope"})["dora_mode"], "LyCORIS fp32 (as trained)")

    def test_loras_normalized(self):
        out = ss.normalize_state({"loras": [
            {"on": True, "name": "a", "strength": 99, "extra": 1},
            "junk",
            {"name": 5},
            {"on": 0, "name": "b", "strength": "x"},
        ]})
        self.assertEqual(out["loras"], [
            {"on": True, "name": "a", "strength": 10.0},
            {"on": True, "name": "", "strength": 1.0},
            {"on": True, "name": "b", "strength": 1.0},
        ])
        self.assertEqual(ss.normalize_state({"loras": "x"})["loras"], [])

    def test_effective_prompt_appends_enabled_loras(self):
        st = {"prompt": "1boy", "loras": [{"on": True, "name": "a", "strength": 0.9},
                                          {"on": False, "name": "b", "strength": 1}]}
        self.assertEqual(ss.effective_prompt(st), "1boy <lora:a:0.9>")

    def test_effective_prompt_formats_and_empty_cases(self):
        st = {"prompt": "", "loras": [{"on": True, "name": "a", "strength": 1.0},
                                      {"on": True, "name": "c", "strength": -0.5},
                                      {"on": True, "name": "", "strength": 1}]}
        self.assertEqual(ss.effective_prompt(st), "<lora:a:1> <lora:c:-0.5>")
        self.assertEqual(ss.effective_prompt({"prompt": "x", "loras": []}), "x")

    def test_effective_prompt_skips_lora_already_tagged(self):
        st = {"prompt": "x <lora:a:0.5>", "loras": [{"on": True, "name": "a", "strength": 1},
                                                    {"on": True, "name": "b", "strength": 1}]}
        self.assertEqual(ss.effective_prompt(st), "x <lora:a:0.5> <lora:b:1>")
        st2 = {"prompt": "x <lora:a>", "loras": [{"on": True, "name": "a", "strength": 1}]}
        self.assertEqual(ss.effective_prompt(st2), "x <lora:a>")
        st3 = {"prompt": "x <lora:ab:1>", "loras": [{"on": True, "name": "a", "strength": 1}]}
        self.assertEqual(ss.effective_prompt(st3), "x <lora:ab:1> <lora:a:1>")


if __name__ == "__main__":
    unittest.main()
