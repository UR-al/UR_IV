"""X/Y/Z planner: cells, loop order, seeds, Prompt S/R, labels and validation (pure Python)."""
import dataclasses
import json
import random
import unittest

from comfy_custom_nodes.ai_studio_forge_parity import dora_infer_mode as dim
from comfy_custom_nodes.ai_studio_forge_parity import xyz_plan as xp

OPTIONS = {"draw_legend": True, "keep_minus1": False, "vary_x": False, "vary_y": False, "vary_z": False,
           "include_sub_images": False, "include_sub_grids": False, "margin": 0, "row_count": 0}
FAMILIES = {"anima-2.9b.safetensors": "2.9b", "anima-base.safetensors": "base",
            "sub/anima-3.8b.safetensors": "3.8b"}


def _axes(x=("nothing", ""), y=("nothing", ""), z=("nothing", "")):
    return [{"axis": a, "text": t} for a, t in (x, y, z)]


def _plan(state=None, axes=None, rng_seed=0, families=FAMILIES, **options):
    return xp.plan_xyz(dict(state or {}), axes or _axes(), {**OPTIONS, **options},
                       rng=random.Random(rng_seed), families=families)


def _cell(plan, ix=0, iy=0, iz=0):
    hits = [c for c in plan.cells if (c.ix, c.iy, c.iz) == (ix, iy, iz)]
    assert len(hits) == 1, hits
    return hits[0]


class PromptSearchReplaceTests(unittest.TestCase):
    def test_lora_epoch_by_strength_12_cells(self):
        plan = _plan({"prompt": "1boy <lora:na-e10:1>", "seed": 5},
                     _axes(("prompt_sr", "na-e10, na-e14, na-e18, na-e20"), ("prompt_sr", ":1>, :0.9>, :1.1>")))
        self.assertEqual(plan.errors, [])
        self.assertEqual(len(plan.cells), 12)
        self.assertEqual(_cell(plan, 1, 2).state["prompt"], "1boy <lora:na-e14:1.1>")
        self.assertEqual(_cell(plan, 0, 0).state["prompt"], "1boy <lora:na-e10:1>")
        self.assertEqual(_cell(plan, 3, 1).state["prompt"], "1boy <lora:na-e20:0.9>")

    def test_cells_are_indexed_x_fastest(self):
        plan = _plan({"prompt": "a"}, _axes(("prompt_sr", "a, b, c"), ("cfg", "1, 2")))
        self.assertEqual([(c.ix, c.iy, c.iz) for c in plan.cells],
                         [(0, 0, 0), (1, 0, 0), (2, 0, 0), (0, 1, 0), (1, 1, 0), (2, 1, 0)])

    def test_sr_searches_list_loras(self):
        state = {"prompt": "1boy", "loras": [{"on": True, "name": "na-e10", "strength": 0.8}]}
        plan = _plan(state, _axes(("prompt_sr", "na-e10, na-e14")))
        self.assertEqual(plan.errors, [])
        self.assertEqual(_cell(plan, 0).state["prompt"], "1boy <lora:na-e10:0.8>")
        self.assertEqual(_cell(plan, 1).state["prompt"], "1boy <lora:na-e14:0.8>")
        for cell in plan.cells:
            self.assertEqual(cell.state["loras"], [])

    def test_sr_replaces_in_negative_too(self):
        plan = _plan({"prompt": "red hair", "negative": "red eyes"}, _axes(("prompt_sr", "red, blue")))
        self.assertEqual(_cell(plan, 1).state["prompt"], "blue hair")
        self.assertEqual(_cell(plan, 1).state["negative"], "blue eyes")

    def test_sr_found_only_in_negative_is_fine(self):
        plan = _plan({"prompt": "cat", "negative": "blurry"}, _axes(("prompt_sr", "blurry, sharp")))
        self.assertEqual(plan.errors, [])
        self.assertEqual(_cell(plan, 1).state["negative"], "sharp")

    def test_sr_is_case_sensitive(self):
        plan = _plan({"prompt": "Cat"}, _axes(("prompt_sr", "cat, dog")))
        self.assertEqual(plan.errors, ['Prompt S/R did not find "cat" in either prompt or negative prompt'])

    def test_sr_missing_aborts(self):
        plan = _plan({"prompt": "1girl", "negative": "worst"}, _axes(("prompt_sr", "na-e10, na-e14")))
        self.assertEqual(plan.errors, ['Prompt S/R did not find "na-e10" in either prompt or negative prompt'])
        self.assertEqual(plan.cells, [])
        self.assertEqual(plan.order, [])

    def test_sr_axes_chain_x_then_y(self):
        # Y searches the prompt X already rewrote: "b" only exists after X replaced "a" with "b".
        plan = _plan({"prompt": "a"}, _axes(("prompt_sr", "a, b"), ("prompt_sr", "b, c")))
        self.assertEqual(plan.errors, ['Prompt S/R did not find "b" in either prompt or negative prompt'])


class PromptOrderTests(unittest.TestCase):
    def test_reorders_tokens_in_positive_prompt(self):
        plan = _plan({"prompt": "cat, red hat, smile", "negative": "cat"},
                     _axes(("prompt_order", "cat, red hat, smile")))
        prompts = [c.state["prompt"] for c in plan.cells]
        self.assertEqual(len(prompts), 6)
        self.assertEqual(prompts[0], "cat, red hat, smile")
        self.assertIn("smile, red hat, cat", prompts)
        self.assertEqual({c.state["negative"] for c in plan.cells}, {"cat"})
        self.assertEqual(plan.labels["x"][-1], "smile, red hat, cat")

    def test_missing_token_is_an_error(self):
        plan = _plan({"prompt": "cat, smile"}, _axes(("prompt_order", "cat, dog")))
        self.assertEqual(plan.errors, ['Prompt order did not find "dog" in the prompt'])
        self.assertEqual(plan.cells, [])


class SeedTests(unittest.TestCase):
    def test_seed_fixed_once(self):
        plan = _plan({"seed": -1}, _axes(("cfg", "1, 2, 3"), ("steps", "10, 20")), rng_seed=0)
        seeds = {c.state["seed"] for c in plan.cells}
        self.assertEqual(seeds, {random.Random(0).randrange(4294967294)})

    def test_fixed_seed_kept(self):
        plan = _plan({"seed": 1234}, _axes(("cfg", "1, 2")))
        self.assertEqual({c.state["seed"] for c in plan.cells}, {1234})

    def test_keep_minus1_leaves_seed_random(self):
        plan = _plan({"seed": -1}, _axes(("cfg", "1, 2")), keep_minus1=True)
        self.assertEqual({c.state["seed"] for c in plan.cells}, {-1})

    def test_seed_axis_minus1_each_random(self):
        plan = _plan({"seed": 7}, _axes(("seed", "-1, -1, -1, 42")))
        seeds = [_cell(plan, i).state["seed"] for i in range(4)]
        self.assertEqual(len(set(seeds[:3])), 3)
        for seed in seeds[:3]:
            self.assertTrue(0 <= seed < 4294967294)
        self.assertEqual(seeds[3], 42)
        self.assertEqual(plan.fixed, {"x": seeds})
        self.assertEqual(plan.labels["x"], [f"Seed: {s}" for s in seeds])
        self.assertEqual(plan.infotext_extra["Fixed X Values"], ", ".join(str(s) for s in seeds))

    def test_seed_axis_keep_minus1(self):
        plan = _plan({}, _axes(("seed", "-1, 5")), keep_minus1=True)
        self.assertEqual([c.state["seed"] for c in plan.cells], [-1, 5])
        self.assertEqual(plan.fixed, {})
        self.assertNotIn("Fixed X Values", plan.infotext_extra)

    def test_vary_seeds(self):
        plan = _plan({"seed": 100}, _axes(("cfg", "1, 2, 3")), vary_x=True)
        self.assertEqual([_cell(plan, i).state["seed"] for i in range(3)], [100, 101, 102])

    def test_vary_y_and_z_multiply_by_varied_axes(self):
        plan = _plan({"seed": 100}, _axes(("cfg", "1, 2, 3"), ("steps", "10, 20"), ("shift", "1, 2")),
                     vary_x=True, vary_y=True, vary_z=True)
        self.assertEqual(_cell(plan, 2, 1, 1).state["seed"], 100 + 2 + 1 * 3 + 1 * 3 * 2)
        plan = _plan({"seed": 100}, _axes(("cfg", "1, 2, 3"), ("steps", "10, 20"), ("shift", "1, 2")),
                     vary_y=True, vary_z=True)
        self.assertEqual(_cell(plan, 2, 1, 1).state["seed"], 100 + 1 + 1 * 2)

    def test_vary_adds_to_seed_axis_value(self):
        plan = _plan({}, _axes(("seed", "10, 20")), vary_x=True)
        self.assertEqual([c.state["seed"] for c in plan.cells], [10, 21])

    def test_keep_minus1_vary_quirk(self):
        plan = _plan({"seed": -1}, _axes(("cfg", "1, 2")), keep_minus1=True, vary_x=True)
        self.assertEqual([c.state["seed"] for c in plan.cells], [-1, 0])


class LoopOrderTests(unittest.TestCase):
    def test_default_order_z_y_x(self):
        plan = _plan({}, _axes(("cfg", "1, 2"), ("steps", "1, 2"), ("shift", "1, 2")))
        self.assertEqual(plan.order, list(range(8)))

    def test_cost_order_checkpoint_outer(self):
        # Memo §3: X=Checkpoint(2.0), Y=CFG(0), Z=Nothing -> loop x (outer), z, y (inner).
        plan = _plan({}, _axes(("checkpoint", "anima-2.9b, anima-base"), ("cfg", "3, 4, 5")))
        self.assertEqual(plan.errors, [])
        nx = 2
        self.assertEqual(plan.order, [ix + iy * nx for ix in range(2) for iy in range(3)])
        self.assertEqual(plan.order, [0, 2, 4, 1, 3, 5])

    def test_prompt_sr_outer_over_cfg(self):
        plan = _plan({"prompt": "a"}, _axes(("prompt_sr", "a, b"), ("cfg", "1, 2")))
        self.assertEqual(plan.order, [0, 2, 1, 3])

    def test_y_outer_then_strictly_dearer_middle(self):
        # Y=Checkpoint outer; X=VAE(0.7) > Z=CFG(0) -> x middle, z inner.
        plan = _plan({}, _axes(("vae", "a, b"), ("checkpoint", "anima-2.9b, anima-base"), ("cfg", "1, 2")))
        nx, ny = 2, 2
        self.assertEqual(plan.order, [ix + iy * nx + iz * nx * ny
                                      for iy in range(2) for ix in range(2) for iz in range(2)])

    def test_z_outer_tie_keeps_y_middle(self):
        plan = _plan({}, _axes(("cfg", "1, 2"), ("steps", "1, 2"), ("checkpoint", "anima-2.9b, anima-base")))
        self.assertEqual(plan.order, list(range(8)))

    def test_tie_for_outer_keeps_default(self):
        plan = _plan({}, _axes(("checkpoint", "anima-2.9b, anima-base"), ("checkpoint", "anima-2.9b, anima-base")))
        self.assertEqual(plan.order, [0, 1, 2, 3])


class CheckpointTests(unittest.TestCase):
    def test_substring_match_picks_shortest_name(self):
        families = {"anima-2.9b-v2.safetensors": "2.9b", "anima-2.9b.safetensors": "2.9b"}
        plan = _plan({}, _axes(("checkpoint", "anima-2.9b")), families=families)
        self.assertEqual(plan.cells[0].state["unet"], "anima-2.9b.safetensors")

    def test_exact_name_wins(self):
        plan = _plan({}, _axes(("checkpoint", "sub/anima-3.8b.safetensors")))
        self.assertEqual(plan.cells[0].state["unet"], "sub/anima-3.8b.safetensors")

    def test_unknown_checkpoint(self):
        plan = _plan({}, _axes(("checkpoint", "anima-2.9b, nope")))
        self.assertEqual(plan.errors, ["Unknown Checkpoint: nope"])
        self.assertEqual(plan.cells, [])

    def test_mixed_families_rejected(self):
        plan = _plan({}, _axes(("checkpoint", "anima-2.9b, anima-3.8b")))
        self.assertEqual(plan.errors, [xp.MIXED_FAMILIES])
        self.assertEqual(plan.cells, [])

    def test_2_9b_and_base_may_mix(self):
        plan = _plan({}, _axes(("checkpoint", "anima-2.9b, anima-base")))
        self.assertEqual(plan.errors, [])
        self.assertEqual([c.state["unet"] for c in plan.cells],
                         ["anima-2.9b.safetensors", "anima-base.safetensors"])

    def test_moss_and_krea2_mix_with_nothing_but_their_own_family(self):
        families = {"anima-2.9b.safetensors": "2.9b", "anima-base.safetensors": "base",
                    "moss_sft.safetensors": "moss", "moss_base.safetensors": "moss", "krea2.safetensors": "krea2"}
        for pair, errors in (("anima-2.9b, anima-base", []), ("moss_sft, moss_base", []),
                             ("anima-2.9b, moss_sft", [xp.MIXED_FAMILIES]),
                             ("krea2, moss_sft", [xp.MIXED_FAMILIES])):
            with self.subTest(pair=pair):
                self.assertEqual(_plan({}, _axes(("checkpoint", pair)), families=families).errors, errors)

    def test_families_checked_across_axes(self):
        plan = _plan({}, _axes(("checkpoint", "anima-2.9b"), ("checkpoint", "anima-3.8b")))
        self.assertEqual(plan.errors, [xp.MIXED_FAMILIES])


class FieldTests(unittest.TestCase):
    def test_size_axis_on_the_i2i_tab_sets_the_i2i_size(self):
        plan = _plan({"tab": "i2i"}, _axes(x=("size", "640x960")))
        state = _cell(plan).state
        self.assertEqual((state["i2i"]["width"], state["i2i"]["height"]), (640, 960))
        self.assertEqual((state["width"], state["height"]), (832, 1216))

    def test_size_sets_width_and_height(self):
        plan = _plan({}, _axes(("size", "512x768, 1024 x 1024")))
        self.assertEqual([(c.state["width"], c.state["height"]) for c in plan.cells], [(512, 768), (1024, 1024)])
        self.assertEqual(plan.labels["x"], ["Size: 512x768", "Size: 1024x1024"])

    def test_nested_fields(self):
        plan = _plan({}, _axes(("hires_steps", "8, 12"), ("sam3_enable", "true, false")))
        self.assertEqual(_cell(plan, 1, 1).state["hires"]["steps"], 12)
        self.assertIs(_cell(plan, 1, 1).state["sam3"]["on"], False)

    def test_denoising_is_hires_denoise_in_t2i(self):
        plan = _plan({"tab": "t2i"}, _axes(("denoising", "0.3")))
        self.assertEqual(plan.cells[0].state["hires"]["denoise"], 0.3)
        self.assertEqual(plan.cells[0].state["i2i"]["denoise"], 0.55)

    def test_denoising_is_i2i_denoise_in_i2i(self):
        plan = _plan({"tab": "i2i"}, _axes(("denoising", "0.3")))
        self.assertEqual(plan.cells[0].state["i2i"]["denoise"], 0.3)
        self.assertEqual(plan.cells[0].state["hires"]["denoise"], 0.4)

    def test_dora_axis_maps_forge_labels_and_unmasks(self):
        plan = _plan({}, _axes(("dora_mode", "Forge (stock), Forge fp32, LyCORIS fp32, DoRA off (no magnitude)")))
        self.assertEqual(plan.errors, [])
        self.assertEqual([c.state["dora_mode"] for c in plan.cells],
                         [dim.MODE_LABELS[k] for k in (dim.MODE_FORGE, dim.MODE_FORGE_FP32, dim.MODE_LYCORIS,
                                                       dim.MODE_NO_MAGNITUDE)])
        self.assertEqual({c.state["forge_like"]["dora"] for c in plan.cells}, {False})
        self.assertEqual(plan.labels["x"][0], "[DoRA] Inference mode: Forge (stock)")

    def test_dora_axis_takes_node_labels_too(self):
        plan = _plan({}, _axes(("dora_mode", "Forge/Comfy fp32")))
        self.assertEqual(plan.cells[0].state["dora_mode"], "Forge/Comfy fp32")

    def test_dora_axis_unknown_mode(self):
        plan = _plan({}, _axes(("dora_mode", "Quantum")))
        self.assertEqual(plan.errors, ["Unknown DoRA mode: Quantum"])

    def test_other_fields_untouched_and_input_not_mutated(self):
        state = {"prompt": "x", "loras": [{"on": True, "name": "a", "strength": 1.0}], "cfg": 5}
        before = json.dumps(state, sort_keys=True)
        plan = _plan(state, _axes(("steps", "10")))
        self.assertEqual(json.dumps(state, sort_keys=True), before)
        self.assertEqual(plan.cells[0].state["cfg"], 5.0)
        self.assertEqual(plan.cells[0].state["steps"], 10)


class LabelTests(unittest.TestCase):
    def test_labels_format(self):
        plan = _plan({"prompt": "a"}, _axes(("cfg", "4, 4.5"), ("prompt_sr", "a, b"),
                                            ("checkpoint", "sub/anima-3.8b.safetensors")))
        self.assertEqual(plan.labels["x"], ["CFG Scale: 4.0", "CFG Scale: 4.5"])
        self.assertEqual(plan.labels["y"], ["a", "b"])
        self.assertEqual(plan.labels["z"], ["anima-3.8b.safetensors"])
        plan = _plan({}, _axes(("steps", "20")))
        self.assertEqual(plan.labels, {"x": ["Steps: 20"], "y": [""], "z": [""]})

    def test_more_label_forms(self):
        plan = _plan({}, _axes(("sampler", "euler"), ("text_encoder", r"te\qwen_3b.safetensors"),
                               ("cfg", "0.123456789")))
        self.assertEqual(plan.labels["x"], ["euler"])
        self.assertEqual(plan.labels["y"], ["Text Encoder: qwen_3b"])
        self.assertEqual(plan.labels["z"], ["CFG Scale: 0.12345679"])


class InfotextTests(unittest.TestCase):
    def test_keys_for_used_axes(self):
        plan = _plan({"prompt": "a"}, _axes(("prompt_sr", "a,  b"), ("nothing", ""), ("cfg", "3,4")))
        self.assertEqual(plan.infotext_extra, {
            "Script": "X/Y/Z plot",
            "X Type": "Prompt S/R", "X Values": "a,  b",
            "Z Type": "CFG Scale", "Z Values": "3,4",
        })


class ValidationTests(unittest.TestCase):
    def _assert_error(self, plan, text):
        self.assertEqual(plan.cells, [])
        self.assertEqual(plan.order, [])
        self.assertEqual(len(plan.errors), 1)
        self.assertIn(text, plan.errors[0])

    def test_grid_size_guard(self):
        # 832x1216 = 1.011776 MP; 98 cells -> round(99.15) = 99 passes, 99 cells -> round(100.17) = 100 fails.
        ok = _plan({}, _axes(("steps", "1-98")))
        self.assertEqual(ok.errors, [])
        self.assertEqual(len(ok.cells), 98)
        self._assert_error(_plan({}, _axes(("steps", "1-99"))), "too large")
        self._assert_error(_plan({}, _axes(("steps", "1-11"), ("cfg", "1-3"), ("shift", "1-3"))), "too large")

    def test_bad_value_reported(self):
        self._assert_error(_plan({}, _axes(("steps", "ten"))), "Steps")

    def test_too_many_values_reported(self):
        self._assert_error(_plan({}, _axes(("steps", "0-999999999999"))), "too many values")

    def test_empty_values_reported(self):
        self._assert_error(_plan({}, _axes(("cfg", "  "))), "CFG Scale")

    def test_unknown_axis_reported(self):
        self._assert_error(_plan({}, _axes(("nope", "1"))), "nope")

    def test_nothing_axes_make_one_cell(self):
        plan = _plan({}, _axes())
        self.assertEqual(len(plan.cells), 1)
        self.assertEqual(plan.order, [0])
        self.assertEqual(plan.infotext_extra, {"Script": "X/Y/Z plot"})

    def test_missing_axes_are_nothing(self):
        plan = xp.plan_xyz({}, [{"axis": "cfg", "text": "1, 2"}], OPTIONS, rng=random.Random(0), families={})
        self.assertEqual(len(plan.cells), 2)


class SerializationTests(unittest.TestCase):
    def test_plan_is_json_serializable(self):
        plan = _plan({"prompt": "a"}, _axes(("prompt_sr", "a, b"), ("size", "512x512")))
        data = json.loads(json.dumps(dataclasses.asdict(plan)))
        self.assertEqual(data["cells"][1]["ix"], 1)
        self.assertEqual(data["cells"][1]["state"]["prompt"], "b")
        self.assertEqual(set(data), {"cells", "order", "labels", "fixed", "infotext_extra", "errors"})



class LoraCheckTests(unittest.TestCase):
    """A <lora:...> tag of any cell that matches no file is an error before anything is queued (final review I5)."""

    AXES = _axes(("prompt_sr", "na-e10, na-e14, na-e18"))

    def test_mistyped_epoch_is_named(self):
        plan = _plan({"prompt": "1boy <lora:na-e10:1>"}, self.AXES)  # no loras= : no check (pure planner use)
        self.assertEqual(plan.errors, [])
        plan = xp.plan_xyz({"prompt": "1boy <lora:na-e10:1>"}, self.AXES, OPTIONS, rng=random.Random(0),
                           families=FAMILIES, loras=["na-e10.safetensors", "sub/NA-E18.safetensors"])
        self.assertEqual((plan.cells, plan.errors), ([], ["LoRA not found: na-e14"]))

    def test_list_loras_and_negative_tags_are_checked(self):
        state = {"prompt": "x", "negative": "<lora:neg:1>", "loras": [{"on": True, "name": "listed", "strength": 1}]}
        plan = xp.plan_xyz(state, _axes(("cfg", "1, 2")), OPTIONS, rng=random.Random(0), families=FAMILIES,
                           loras=["neg.safetensors"])
        self.assertEqual(plan.errors, ["LoRA not found: listed"])

    def test_list_is_read_only_when_a_cell_has_a_tag(self):
        def unreadable():
            raise OSError("no folder")

        plan = xp.plan_xyz({"prompt": "plain"}, _axes(("cfg", "1, 2")), OPTIONS, rng=random.Random(0),
                           families=FAMILIES, loras=unreadable)
        self.assertEqual((len(plan.cells), plan.errors), (2, []))
        plan = xp.plan_xyz({"prompt": "<lora:a:1>"}, _axes(("cfg", "1, 2")), OPTIONS, rng=random.Random(0),
                           families=FAMILIES, loras=unreadable)
        self.assertEqual(plan.cells, [])
        self.assertEqual(plan.errors, ["LoRA list could not be read: no folder"])

if __name__ == "__main__":
    unittest.main()
