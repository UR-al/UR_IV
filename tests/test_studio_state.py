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

    def test_i2i_has_its_own_size(self):
        state = ss.normalize_state({"width": 640, "i2i": {"width": 1250, "height": 9000}})
        self.assertEqual(state["width"], 640)
        self.assertEqual((state["i2i"]["width"], state["i2i"]["height"]), (1248, 4096))
        self.assertEqual(ss.generation_size(state | {"tab": "t2i"}), (640, 1216))
        self.assertEqual(ss.generation_size(state | {"tab": "i2i"}), (1248, 4096))

    def test_moss_stream_defaults_on_and_keeps_only_booleans(self):
        self.assertIs(ss.normalize_state({})["moss_stream"], True)
        self.assertIs(ss.normalize_state({"moss_stream": False})["moss_stream"], False)
        self.assertIs(ss.normalize_state({"moss_stream": "no"})["moss_stream"], True)

    def test_a_moss_size_runs_on_the_16_grid(self):
        self.assertEqual(ss.generation_size(ss.normalize_state({"unet": "moss_sft.safetensors", "width": 856, "height": 840})), (864, 848))
        self.assertEqual(ss.generation_size(ss.normalize_state({"unet": "anima.safetensors", "width": 856, "height": 840})), (856, 840))

    def test_family_memory_keeps_known_families_only(self):
        memory = ss.normalize_state({"family_memory": {
            "moss": {"te": "a", "vae": "b", "sampler": "euler", "scheduler": "simple", "steps": 999, "cfg": 3, "shift": 5,
                     "extra": 1},
            "other": {"te": "x"}, "krea2": "bad"}})["family_memory"]
        self.assertEqual(memory, {"moss": {"te": "a", "vae": "b", "sampler": "euler", "scheduler": "simple", "steps": 150,
                                           "cfg": 3.0, "shift": 5.0}})
        self.assertEqual(ss.normalize_state({"family_memory": "bad"})["family_memory"], {})

    def test_reference_defaults_and_clamps(self):
        self.assertEqual(ss.normalize_state({})["reference"], ss.DEFAULT_STATE["reference"])
        reference = ss.normalize_state({"reference": {
            "on": True, "image": "a.png", "method": "bad", "megapixels": 9,
            "snapref": {"mode": "bad", "strength": 9, "correction": "rescale", "extra": 1},
            "ref_adapter": {"strength": 5}, "krea2": {"position": "bad", "grounding_px": -1}}})["reference"]
        self.assertEqual((reference["on"], reference["image"], reference["method"], reference["megapixels"]),
                         (True, "a.png", "snapref", 2.0))
        self.assertEqual((reference["snapref"]["mode"], reference["snapref"]["strength"], reference["snapref"]["correction"]),
                         ("cs", 3.0, "rescale"))
        self.assertNotIn("extra", reference["snapref"])
        self.assertEqual(reference["ref_adapter"]["strength"], 2.0)
        self.assertEqual((reference["krea2"]["position"], reference["krea2"]["grounding_px"]), ("aligned (trained)", 0))

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

    def test_ext_defaults_come_from_the_table(self):
        schema = ss.EXT_SCHEMA["extensions"]
        self.assertEqual(sorted(ss.DEFAULT_STATE["ext"]), sorted(item["key"] for item in schema))
        for item in schema:
            with self.subTest(key=item["key"]):
                self.assertEqual(ss.DEFAULT_STATE["ext"][item["key"]],
                                 {"on": False, **{field["key"]: field["default"] for field in item["fields"]}})

    def test_ext_missing_or_junk_gives_defaults(self):
        for raw in ({}, {"ext": None}, {"ext": "junk"}, {"ext": []}, {"ext": {"skim": 5, "speed": []}}):
            with self.subTest(raw=raw):
                self.assertEqual(ss.normalize_state(raw)["ext"], ss.DEFAULT_STATE["ext"])

    def test_ext_normalize_is_idempotent_and_independent(self):
        out = ss.normalize_state({"ext": {"skim": {"on": True, "skim_skimming_cfg": 3}}})
        self.assertEqual(ss.normalize_state(out), out)
        out = ss.normalize_state({})
        out["ext"]["speed"]["on"] = True
        self.assertFalse(ss.DEFAULT_STATE["ext"]["speed"]["on"])

    def test_ext_values_coerced_clamped_and_unknown_dropped(self):
        out = ss.normalize_state({"ext": {
            "nope": {"on": True},
            "skim": {"on": "yes", "skim_skimming_cfg": "2.5", "skim_flip_at": 9, "skim_full_skim_negative": 1, "zzz": 1},
            "guidance": {"on": True, "guid_attn_method": "SEG", "guid_scale": float("nan"), "guid_adg_interval": 99.7,
                         "guid_rdc_tau": 10 ** 400, "guid_block_indices": 18, "guid_mod_start_layer": True},
            "speed": {"mode": "nope", "preset": "flux", "seed": "-5", "scales": "1/2,1"},
            "degrid": {"model": "my.safetensors", "mode": "Full (전체)"},
        }})["ext"]
        self.assertNotIn("nope", out)
        self.assertIs(out["skim"]["on"], False)
        self.assertEqual(out["skim"]["skim_skimming_cfg"], 2.5)
        self.assertEqual(out["skim"]["skim_flip_at"], 1.0)
        self.assertIs(out["skim"]["skim_full_skim_negative"], False)
        self.assertNotIn("zzz", out["skim"])
        guidance = out["guidance"]
        self.assertIs(guidance["on"], True)
        self.assertEqual(guidance["guid_attn_method"], "SEG")
        self.assertEqual(guidance["guid_scale"], 4.0)
        self.assertEqual(guidance["guid_adg_interval"], 10)
        self.assertEqual(guidance["guid_rdc_tau"], 0.0)
        self.assertEqual(guidance["guid_block_indices"], "18")
        self.assertEqual(guidance["guid_mod_start_layer"], 0)
        self.assertEqual(out["speed"]["mode"], "transition")
        self.assertEqual(out["speed"]["preset"], "flux")
        self.assertEqual(out["speed"]["seed"], -1)
        self.assertEqual(out["speed"]["scales"], "1/2,1")
        self.assertEqual(out["degrid"]["model"], "my.safetensors")  # a file list has no fixed choices
        self.assertEqual(out["degrid"]["mode"], "Full")

    def test_huge_integer_is_not_a_crash(self):
        self.assertEqual(ss.normalize_state({"cfg": 10 ** 400})["cfg"], 3.0)


if __name__ == "__main__":
    unittest.main()
