"""Picker order: similar sampler / scheduler names sit next to each other, no name changes (no ComfyUI needed)."""
import random
import unittest
from types import SimpleNamespace
from unittest import mock

from comfy_custom_nodes.ai_studio_forge_parity import compat, extra_samplers_register, extra_schedulers, studio_choices, xyz_axes
from comfy_custom_nodes.ai_studio_forge_parity.studio_choices_order import group_similar, name_words

# ComfyUI 0.44.0 comfy/samplers.py: KSampler.SAMPLERS (= SAMPLER_NAMES) and SCHEDULER_NAMES, in its order.
CORE_SAMPLERS = [
    "euler", "euler_cfg_pp", "euler_ancestral", "euler_ancestral_cfg_pp", "heun", "heunpp2", "exp_heun_2_x0",
    "exp_heun_2_x0_sde", "dpm_2", "dpm_2_ancestral", "lms", "dpm_fast", "dpm_adaptive", "dpmpp_2s_ancestral",
    "dpmpp_2s_ancestral_cfg_pp", "dpmpp_sde", "dpmpp_sde_gpu", "dpmpp_2m", "dpmpp_2m_cfg_pp", "dpmpp_2m_sde",
    "dpmpp_2m_sde_gpu", "dpmpp_2m_sde_heun", "dpmpp_2m_sde_heun_gpu", "dpmpp_3m_sde", "dpmpp_3m_sde_gpu", "ddpm", "lcm",
    "ipndm", "ipndm_v", "deis", "cfgpp_ud10_ab", "res_multistep", "res_multistep_cfg_pp", "res_multistep_ancestral",
    "res_multistep_ancestral_cfg_pp", "gradient_estimation", "gradient_estimation_cfg_pp", "er_sde", "seeds_2", "seeds_3",
    "sa_solver", "sa_solver_pece", "ddim", "uni_pc", "uni_pc_bh2",
]
CORE_SCHEDULERS = ["simple", "sgm_uniform", "karras", "exponential", "ddim_uniform", "beta", "normal", "linear_quadratic",
                   "kl_optimal"]
PPM_SCHEDULERS = ["ays", "ays+", "ays_30", "ays_30+", "gits", "beta_1_1"]  # ComfyUI-ppm src/schedulers.py

# What the user's ComfyUI lists (core, then this pack, then ComfyUI-ppm) ...
REAL_SAMPLERS = [*CORE_SAMPLERS, *extra_samplers_register.PACK_SAMPLERS]
REAL_SCHEDULERS = [*CORE_SCHEDULERS, compat.BETA57_SCHEDULER, *extra_schedulers.SCHEDULER_NAMES, *PPM_SCHEDULERS]
# ... and what the pickers show.
GROUPED_SAMPLERS = [
    "euler", "euler_cfg_pp", "euler_ancestral", "euler_ancestral_cfg_pp", "euler_dy", "euler_dy_cfg_pp",
    "euler_smea_dy", "euler_smea_dy_cfg_pp",
    "heun", "heunpp2", "exp_heun_2_x0", "exp_heun_2_x0_sde",
    "dpm_2", "dpm_2_ancestral", "dpm_fast", "dpm_adaptive",
    "lms",
    "dpmpp_2s_ancestral", "dpmpp_2s_ancestral_cfg_pp", "dpmpp_sde", "dpmpp_sde_gpu",
    "dpmpp_2m", "dpmpp_2m_cfg_pp", "dpmpp_2m_sde", "dpmpp_2m_sde_gpu", "dpmpp_2m_sde_heun", "dpmpp_2m_sde_heun_gpu",
    "dpmpp_2m_flow_ode", "dpmpp_2m_heun_flow_ode",
    "dpmpp_3m_sde", "dpmpp_3m_sde_gpu", "dpmpp_3m_flow_ode", "dpmpp_4m_sde",
    "ddpm", "lcm",
    "ipndm", "ipndm_v", "ipndm_v_fixed",
    "deis", "cfgpp_ud10_ab",
    "res_multistep", "res_multistep_cfg_pp", "res_multistep_ancestral", "res_multistep_ancestral_cfg_pp",
    "gradient_estimation", "gradient_estimation_cfg_pp",
    "er_sde", "er_sde_reverse_time", "er_sde_ode", "er_sde_tunable",
    "seeds_2", "seeds_3", "sa_solver", "sa_solver_pece", "ddim", "uni_pc", "uni_pc_bh2",
    "restart_flow",
]
GROUPED_SCHEDULERS = [
    "simple", "sgm_uniform",
    "karras", "karras_dynamic",
    "exponential", "ddim_uniform",
    "beta", "beta57", "beta_1_1",
    "normal", "linear_quadratic", "kl_optimal",
    "cosine", "cosine_exponential",
    "phi", "laplace", "custom", "react_cosinusoidal_dynsf",
    "flow_cosmos_rho7", "flow_cosmos_dynamic",
    "ays", "ays+", "ays_30", "ays_30+",
    "gits",
]


def _family_runs(names) -> list[str]:
    """The first word of each run of neighbours that share it."""
    runs = []
    for name in names:
        word = name_words(name)[0]
        if not runs or runs[-1] != word:
            runs.append(word)
    return runs


class NameWordsTests(unittest.TestCase):
    def test_case_space_and_underscore_do_not_matter(self):
        self.assertEqual(name_words("karras_dynamic"), ("karras", "dynamic"))
        self.assertEqual(name_words("Karras Dynamic"), ("karras", "dynamic"))
        self.assertEqual(name_words("  KARRAS-dynamic "), ("karras", "dynamic"))

    def test_a_number_after_letters_is_its_own_word(self):
        self.assertEqual(name_words("beta57"), ("beta", "57"))
        self.assertEqual(name_words("beta_1_1"), ("beta", "1", "1"))
        self.assertEqual(name_words("flow_cosmos_rho7"), ("flow", "cosmos", "rho", "7"))

    def test_a_number_keeps_the_letters_after_it(self):
        self.assertEqual(name_words("dpmpp_2m_sde"), ("dpmpp", "2m", "sde"))
        self.assertEqual(name_words("dpmpp_2s_ancestral"), ("dpmpp", "2s", "ancestral"))

    def test_marks_are_left_out(self):
        self.assertEqual(name_words("ays_30+"), ("ays", "30"))
        self.assertEqual(name_words("DPM++ 2M"), ("dpm", "2m"))

    def test_a_name_without_letters_or_digits_is_one_word(self):
        self.assertEqual(name_words("+++"), ("+++",))
        self.assertEqual(name_words(""), ("",))


class GroupSimilarTests(unittest.TestCase):
    def test_the_users_examples(self):
        names = ["simple", "karras", "exponential", "beta", "normal", "beta57", "karras_dynamic", "custom", "beta_1_1"]
        self.assertEqual(group_similar(names),
                         ["simple", "karras", "karras_dynamic", "exponential", "beta", "beta57", "beta_1_1", "normal", "custom"])

    def test_dpm_2_and_its_ancestral(self):
        self.assertEqual(group_similar(["dpm_2", "lms", "dpm_2_ancestral", "dpmpp_2m", "dpm_fast"]),
                         ["dpm_2", "dpm_2_ancestral", "dpm_fast", "lms", "dpmpp_2m"])

    def test_display_names_group_like_ids(self):
        self.assertEqual(group_similar(["Karras", "Simple", "karras dynamic", "KARRAS_EXTRA"]),
                         ["Karras", "karras dynamic", "KARRAS_EXTRA", "Simple"])

    def test_families_keep_the_order_they_first_appear_in(self):
        self.assertEqual(group_similar(["zeta", "alpha", "zeta_2", "mid", "alpha_b"]), ["zeta", "zeta_2", "alpha", "alpha_b", "mid"])

    def test_a_family_groups_again_by_its_next_word(self):
        names = ["dpmpp_2m", "dpmpp_2m_sde", "dpmpp_3m_sde", "dpmpp_2m_flow_ode", "dpmpp_3m_flow_ode"]
        self.assertEqual(group_similar(names), ["dpmpp_2m", "dpmpp_2m_sde", "dpmpp_2m_flow_ode", "dpmpp_3m_sde", "dpmpp_3m_flow_ode"])

    def test_a_longer_first_word_is_another_family(self):
        self.assertEqual(group_similar(["heun", "x", "heunpp2", "dpm_2", "y", "dpmpp_2m"]), ["heun", "x", "heunpp2", "dpm_2", "y", "dpmpp_2m"])

    def test_names_with_the_same_words_stay_in_order(self):
        self.assertEqual(group_similar(["ays_30+", "gits", "ays_30", "x", "ays+", "ays"]), ["ays+", "ays", "ays_30+", "ays_30", "gits", "x"])

    def test_the_plain_name_leads_its_variants(self):
        # The pack registers euler_dy_cfg_pp before euler_dy; every core pair lists the plain sampler first.
        self.assertEqual(group_similar(["euler", "euler_cfg_pp", "lms", "euler_dy_cfg_pp", "euler_smea_dy_cfg_pp", "euler_dy", "euler_smea_dy"]),
                         ["euler", "euler_cfg_pp", "euler_dy", "euler_dy_cfg_pp", "euler_smea_dy", "euler_smea_dy_cfg_pp", "lms"])
        self.assertEqual(group_similar(["beta_1_1", "x", "beta57", "beta"]), ["beta", "beta_1_1", "beta57", "x"])

    def test_the_plain_name_keeps_its_familys_place(self):
        self.assertEqual(group_similar(["a_1", "b", "a"]), ["a", "a_1", "b"])
        self.assertEqual(group_similar(["b", "a_1", "c", "a"]), ["b", "a", "a_1", "c"])

    def test_a_grouped_list_does_not_change(self):
        for names in (GROUPED_SAMPLERS, GROUPED_SCHEDULERS, [], ["euler"], ["a", "a", "b"]):
            self.assertEqual(group_similar(names), list(names))

    def test_only_the_order_changes(self):
        rng = random.Random(7)
        for names in (REAL_SAMPLERS, REAL_SCHEDULERS):
            for _ in range(50):
                shuffled = rng.sample(names, len(names))
                grouped = group_similar(shuffled)
                self.assertEqual(sorted(grouped), sorted(shuffled))
                runs = _family_runs(grouped)
                self.assertEqual(len(runs), len(set(runs)), grouped)  # one run per family
                self.assertEqual(runs, list(dict.fromkeys(name_words(name)[0] for name in shuffled)))  # in first-seen order
                self.assertEqual(group_similar(grouped), grouped)
                for name in grouped:  # a name comes before the names that go on from it
                    words = name_words(name)
                    longer = [other for other in grouped if name_words(other)[:len(words)] == words and name_words(other) != words]
                    self.assertTrue(all(grouped.index(name) < grouped.index(other) for other in longer), (name, grouped))

    def test_a_copy_comes_back(self):
        names = ["b", "a", "b_1"]
        grouped = group_similar(names)
        self.assertIsNot(grouped, names)
        self.assertEqual(names, ["b", "a", "b_1"])
        self.assertEqual(group_similar(tuple(names)), ["b", "b_1", "a"])

    def test_same_name_twice_is_kept_twice(self):
        self.assertEqual(group_similar(["beta", "x", "beta"]), ["beta", "beta", "x"])


class RealListTests(unittest.TestCase):
    """The lists of ComfyUI 0.44.0 with this pack and ComfyUI-ppm, as the pickers show them."""

    maxDiff = None

    def test_samplers(self):
        self.assertEqual(group_similar(REAL_SAMPLERS), GROUPED_SAMPLERS)

    def test_schedulers(self):
        self.assertEqual(group_similar(REAL_SCHEDULERS), GROUPED_SCHEDULERS)

    def test_scheduler_order_does_not_depend_on_which_pack_loaded_first(self):
        ppm_first = [*CORE_SCHEDULERS, *PPM_SCHEDULERS, compat.BETA57_SCHEDULER, *extra_schedulers.SCHEDULER_NAMES]
        grouped = group_similar(ppm_first)
        self.assertEqual(grouped[:9], ["simple", "sgm_uniform", "karras", "karras_dynamic", "exponential", "ddim_uniform",
                                       "beta", "beta_1_1", "beta57"])
        self.assertEqual(sorted(grouped), sorted(GROUPED_SCHEDULERS))


class PickerListTests(unittest.TestCase):
    """The panel and the X/Y/Z picker both read ``/choices``: one place orders both."""

    def _choices(self, kind):
        ksampler = SimpleNamespace(SAMPLERS=list(REAL_SAMPLERS), SCHEDULERS=list(REAL_SCHEDULERS))
        with mock.patch.dict("sys.modules", {"comfy": SimpleNamespace(), "comfy.samplers": SimpleNamespace(KSampler=ksampler)}):
            return studio_choices.choices(kind)["items"]

    def test_the_route_lists_are_grouped(self):
        self.assertEqual(self._choices("samplers"), GROUPED_SAMPLERS)
        self.assertEqual(self._choices("schedulers"), GROUPED_SCHEDULERS)

    def test_the_xyz_axes_read_the_same_lists(self):
        kinds = {axis_id: xyz_axes.axis_by_id(axis_id).choices for axis_id in ("sampler", "hires_sampler", "scheduler")}
        self.assertEqual(kinds, {"sampler": "samplers", "hires_sampler": "samplers", "scheduler": "schedulers"})
        for axis_id in kinds:
            self.assertEqual(xyz_axes.axis_by_id(axis_id).values, ())  # no list of its own to fall out of step


if __name__ == "__main__":
    unittest.main()
