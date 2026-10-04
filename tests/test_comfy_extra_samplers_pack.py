"""Pack 1.7.0 Extra Samplers — the ComfyUI glue: registration, the settings node, the Hires carry.

Fake ComfyUI tables (``comfy.samplers`` lists and ``KSampler``, a ``comfy.k_diffusion.sampling`` module)
stand in for the host; no ComfyUI, no GPU. The registered names never overwrite another provider's
function (``compat.register_names``), every distinct choice list gets each owned name once, the two
discard-penultimate entries join ``KSampler.DISCARD_PENULTIMATE_SIGMA_SAMPLERS``, CNS and SPEED see through
the registered wrapper (the flow ODE trio, which draws nothing, takes no noise sampler — sam-extra's
signatures), the four Dy entries are guarded. ``ForgeNeoExtraSamplerSettings`` coerces its five
inputs like sam-extra into ``model_options`` and a Hires checkpoint override keeps them.

``EndToEndTests`` (torch + the extension oracle + Forge's files): node → model options → the registered
``sample_er_sde_tunable`` running on Forge's k-diffusion == sam-extra's sampler with the same values.
The real-ComfyUI tier is ``tests/test_comfy_extra_samplers_runtime.py``.
"""
from __future__ import annotations

import contextlib
import inspect
import types
import unittest
from unittest import mock

from comfy_custom_nodes.ai_studio_forge_parity import compat, extra_samplers_math as xs
from comfy_custom_nodes.ai_studio_forge_parity import extra_samplers_register as reg
from comfy_custom_nodes.ai_studio_forge_parity import guidance_cns, guidance_common, hires_carry
from tests._forge_k_diffusion import (
    Predictor,
    ReplayNoise,
    ToyModel,
    forge_k_sampling,
    import_ext,
    installed_k_sampling,
    stub_modules,
)
from tests._optional_deps import bind_torch, requires_torch
from tests._sam_extra_ext import requires_extension

torch = None  # filled by bind_torch() in the @requires_torch classes

# sam-extra's flow ODE entries: no eta, no noise_sampler (they never draw)
_FLOW_ODE = ("dpmpp_2m_flow_ode", "dpmpp_2m_heun_flow_ode", "dpmpp_3m_flow_ode")


def _stock_sampler(model, x, sigmas, extra_args=None, callback=None, disable=None):
    return x


def _fake_comfy(*, kd=None, shared_lists=False):
    """``comfy`` + ``comfy.samplers`` + ``comfy.k_diffusion.sampling`` stand-ins (dict for stub_modules)."""
    k_names = ["euler", "er_sde", "dpmpp_3m_sde"]
    samplers = types.ModuleType("comfy.samplers")
    samplers.KSAMPLER_NAMES = k_names
    samplers.SAMPLER_NAMES = k_names if shared_lists else k_names + ["ddim", "uni_pc", "uni_pc_bh2"]

    class KSampler:
        SAMPLERS = samplers.SAMPLER_NAMES
        DISCARD_PENULTIMATE_SIGMA_SAMPLERS = {"dpm_2", "dpm_2_ancestral", "uni_pc", "uni_pc_bh2"}

    samplers.KSampler = KSampler
    if kd is None:
        kd = types.ModuleType("comfy.k_diffusion.sampling")
        kd.sample_euler = _stock_sampler
        kd.sample_er_sde = _stock_sampler
    comfy = types.ModuleType("comfy")
    comfy.__path__ = []
    k_package = types.ModuleType("comfy.k_diffusion")
    k_package.__path__ = []
    k_package.sampling = kd
    comfy.samplers, comfy.k_diffusion = samplers, k_package
    return {"comfy": comfy, "comfy.samplers": samplers, "comfy.k_diffusion": k_package,
            "comfy.k_diffusion.sampling": kd}


@contextlib.contextmanager
def _with_comfy(modules):
    if modules is None:
        modules = {"comfy": None, "comfy.samplers": None, "comfy.k_diffusion": None, "comfy.k_diffusion.sampling": None}
    with stub_modules(modules):
        yield modules


class RegistrationTests(unittest.TestCase):
    """``install_extra_samplers`` in fake ComfyUI tables (no torch)."""

    def test_the_tables(self):
        self.assertEqual(len(reg.PACK_SAMPLERS), 12)
        self.assertEqual(set(reg.IMPLEMENTATIONS), set(reg.PACK_SAMPLERS))
        self.assertFalse(set(reg.PACK_SAMPLERS) & set(reg.CORE_SAMPLERS))
        self.assertLessEqual(set(reg.DY_SAMPLERS), set(reg.PACK_SAMPLERS))
        self.assertLessEqual(set(reg.DISCARD_PENULTIMATE_SIGMA), set(reg.PACK_SAMPLERS))
        self.assertEqual(set(xs.ER_SDE_SAMPLERS), set(reg.PACK_SAMPLERS[:3]))
        # every name has its implementation module in the pack
        self.assertEqual(reg.unavailable_names(), ())

    def test_registration_adds_every_available_name_once_and_tags_it(self):
        for shared in (False, True):
            with self.subTest(shared_lists=shared):
                modules = _fake_comfy(shared_lists=shared)
                samplers, kd = modules["comfy.samplers"], modules["comfy.k_diffusion.sampling"]
                with _with_comfy(modules):
                    first = reg.install_extra_samplers()
                    second = reg.install_extra_samplers()
                    owned = reg.owned_names()
                available = tuple(name for name in reg.PACK_SAMPLERS if name not in reg.unavailable_names())
                self.assertEqual((first.owned, first.inserted, first.foreign, first.published),
                                 (available, available, (), True))
                self.assertEqual((second.owned, second.inserted), (available, ()))
                self.assertEqual(owned, available)
                self.assertEqual(samplers.KSAMPLER_NAMES, ["euler", "er_sde", "dpmpp_3m_sde", *available])
                if not shared:
                    self.assertEqual(samplers.SAMPLER_NAMES,
                                     ["euler", "er_sde", "dpmpp_3m_sde", "ddim", "uni_pc", "uni_pc_bh2", *available])
                self.assertIs(samplers.KSampler.SAMPLERS, samplers.SAMPLER_NAMES)
                expected_discard = {"dpm_2", "dpm_2_ancestral", "uni_pc", "uni_pc_bh2"} | (
                    set(reg.DISCARD_PENULTIMATE_SIGMA) & set(available))
                self.assertEqual(samplers.KSampler.DISCARD_PENULTIMATE_SIGMA_SAMPLERS, expected_discard)
                self.assertIn("dpmpp_4m_sde", expected_discard)
                self.assertIn("dpmpp_3m_flow_ode", expected_discard)
                self.assertNotIn("dpmpp_2m_flow_ode", expected_discard)
                for name in available:
                    entry = getattr(kd, "sample_" + name)
                    self.assertEqual(compat.entry_owner(entry), reg.OWNER)
                    self.assertEqual(entry.__name__, "sample_" + name)
                    parameters = inspect.signature(entry).parameters
                    if name in _FLOW_ODE:
                        self.assertNotIn("noise_sampler", parameters)
                        self.assertNotIn("eta", parameters)
                    else:
                        self.assertIn("noise_sampler", parameters)
                self.assertIs(kd.sample_er_sde, _stock_sampler)

    def test_cns_and_speed_see_through_the_registered_wrapper(self):
        modules = _fake_comfy()
        kd = modules["comfy.k_diffusion.sampling"]
        with _with_comfy(modules):
            reg.install_extra_samplers()
        self.assertIs(guidance_cns._unwrapped(kd.sample_er_sde_tunable), xs.sample_er_sde_tunable)
        self.assertTrue(guidance_cns._takes_noise_sampler(kd.sample_dpmpp_4m_sde))
        for name in _FLOW_ODE:   # nothing drawn: CNS Path A runs them as they are
            self.assertFalse(guidance_cns._takes_noise_sampler(getattr(kd, "sample_" + name)))
        table = guidance_cns._kd_samplers(kd)
        self.assertTrue(table["sample_euler_dy_cfg_pp"][1])
        for name in reg.DY_SAMPLERS:
            with self.subTest(name=name):
                entry = getattr(kd, "sample_" + name)
                self.assertTrue(guidance_common.declares_keyword(entry, guidance_common.STEP_OFFSET_KWARG))
                self.assertNotIn("kd", inspect.signature(entry).parameters)
        for name in ("er_sde_reverse_time", "er_sde_ode", "er_sde_tunable", "dpmpp_4m_sde", *_FLOW_ODE, "restart_flow"):
            self.assertFalse(guidance_common.declares_keyword(getattr(kd, "sample_" + name),
                                                              guidance_common.STEP_OFFSET_KWARG))

    def test_dy_entries_run_through_the_guard(self):
        modules = _fake_comfy()
        kd = modules["comfy.k_diffusion.sampling"]
        with _with_comfy(modules):
            reg.install_extra_samplers()
        seen = []
        model = types.SimpleNamespace(inner_model=types.SimpleNamespace(conds={"positive": [{"area": (1, 1, 0, 0)}]}))
        from comfy_custom_nodes.ai_studio_forge_parity import extra_samplers_dy as dy
        from comfy_custom_nodes.ai_studio_forge_parity import extra_samplers_dy_host as host

        def fake_core(model, x, sigmas, **kwargs):
            seen.append(kwargs)
            return x

        for name, cfg_pp in reg.DY_SAMPLERS.items():
            with self.subTest(name=name), mock.patch.object(dy, "euler_dy", fake_core), \
                    mock.patch.object(host, "_logged", {f"{name}:regional"}):
                seen.clear()
                getattr(kd, "sample_" + name)(model, "x", "sigmas", extra_args={}, sam_extra_step_offset=2)
                self.assertEqual(len(seen), 1)
                call = seen[0]
                self.assertEqual((call["substeps"], call["cfg_pp"], call["sam_extra_step_offset"]), (False, cfg_pp, 2))
                self.assertEqual(call["smea"], "smea" in name)
                self.assertIs(call["substep_inputs"], dy.SubstepInputs)

    def test_a_foreign_name_is_never_replaced_or_published(self):
        modules = _fake_comfy()
        kd, samplers = modules["comfy.k_diffusion.sampling"], modules["comfy.samplers"]

        def theirs(model, x, sigmas, extra_args=None, callback=None, disable=None):
            return x

        kd.sample_euler_dy = theirs   # e.g. Koishi-Star's own node, loaded first
        kd.sample_dpmpp_4m_sde = theirs
        with _with_comfy(modules), mock.patch.object(compat, "_FOREIGN_LOGGED", set()), \
                self.assertLogs("ai_studio_forge_parity", level="WARNING") as logs:
            report = reg.install_extra_samplers()
            owned = reg.owned_names()
            choices = _node_class().INPUT_TYPES()["optional"][compat.OWNED_NAMES_INPUT][0]
        self.assertIs(kd.sample_euler_dy, theirs)
        self.assertEqual(report.foreign, ("dpmpp_4m_sde", "euler_dy"))
        for name in ("euler_dy", "dpmpp_4m_sde"):
            self.assertNotIn(name, report.owned)
            self.assertNotIn(name, owned)
            self.assertNotIn(name, choices)
            self.assertNotIn(name, samplers.SAMPLER_NAMES)
        self.assertNotIn("dpmpp_4m_sde", samplers.KSampler.DISCARD_PENULTIMATE_SIGMA_SAMPLERS)
        self.assertIn("euler_smea_dy", owned)
        self.assertTrue(any("'euler_dy'" in line for line in logs.output))

    def test_owned_names_are_read_live(self):
        modules = _fake_comfy()
        kd, samplers = modules["comfy.k_diffusion.sampling"], modules["comfy.samplers"]
        with _with_comfy(modules):
            reg.install_extra_samplers()
            kd.sample_er_sde_ode = _stock_sampler            # replaced afterwards
            samplers.SAMPLER_NAMES.remove("euler_smea_dy")   # no longer offered
            owned = reg.owned_names()
        self.assertNotIn("er_sde_ode", owned)
        self.assertNotIn("euler_smea_dy", owned)
        self.assertIn("er_sde_tunable", owned)

    def test_outside_comfyui_and_on_a_broken_host(self):
        empty = compat.NameRegistration()
        with _with_comfy(None):
            self.assertEqual(reg.install_extra_samplers(), empty)
            self.assertEqual(reg.owned_names(), ())
            with self.assertRaises(RuntimeError):
                reg.install_extra_samplers(required=True)
            choices = _node_class().INPUT_TYPES()["optional"][compat.OWNED_NAMES_INPUT][0]
        self.assertEqual(choices, [compat.NO_OWNED_NAMES])
        modules = _fake_comfy()
        modules["comfy.samplers"].KSAMPLER_NAMES = ("tuple", "not a list")
        modules["comfy.samplers"].SAMPLER_NAMES = None
        modules["comfy.samplers"].KSampler.SAMPLERS = None
        with _with_comfy(modules):
            self.assertEqual(reg.install_extra_samplers(), empty)   # nothing to publish into: nothing owned
            self.assertFalse(hasattr(modules["comfy.k_diffusion.sampling"], "sample_er_sde_tunable"))
            with self.assertRaises(RuntimeError):
                reg.install_extra_samplers(required=True)
        with _with_comfy(_fake_comfy()), mock.patch.object(reg, "register_names", side_effect=TypeError("boom")), \
                self.assertLogs("ai_studio_forge_parity", level="ERROR"):
            self.assertEqual(reg.install_extra_samplers(), empty)

    def test_unimplemented_names_are_skipped(self):
        real = reg.IMPLEMENTATIONS["er_sde_ode"]
        with mock.patch.dict(reg.IMPLEMENTATIONS, {"er_sde_ode": ("extra_samplers_not_there", "sample_x"),
                                                   "er_sde_tunable": (real[0], "missing_attribute")}):
            self.assertIn("er_sde_ode", reg.unavailable_names())
            self.assertIn("er_sde_tunable", reg.unavailable_names())
            modules = _fake_comfy()
            with _with_comfy(modules):
                report = reg.install_extra_samplers()
            self.assertNotIn("er_sde_ode", report.owned)
            self.assertFalse(hasattr(modules["comfy.k_diffusion.sampling"], "sample_er_sde_ode"))
            with self.assertRaises(LookupError):
                reg.sampler_entry("er_sde_ode")

    def test_a_broken_implementation_module_is_an_error_not_a_skip(self):
        # only the implementation module itself may be missing; a missing dependency inside it is a bug
        error = ModuleNotFoundError("No module named 'torch'", name="torch")
        with mock.patch.object(reg.importlib, "import_module", side_effect=error):
            with self.assertRaises(ModuleNotFoundError):
                reg.implementation("er_sde_ode")


def _node_class():
    from comfy_custom_nodes.ai_studio_forge_parity import extra_samplers_nodes

    return extra_samplers_nodes.ForgeNeoExtraSamplerSettings


class FakePatcher:
    """A ``ModelPatcher`` stand-in: ``clone`` copies the option dicts."""

    def __init__(self, options=None):
        self.model_options = {"transformer_options": {}} if options is None else options

    def clone(self):
        options = {key: (dict(value) if isinstance(value, dict) else value) for key, value in self.model_options.items()}
        return FakePatcher(options)


class SettingsNodeTests(unittest.TestCase):
    """``ForgeNeoExtraSamplerSettings`` (no torch)."""

    @classmethod
    def setUpClass(cls):
        from comfy_custom_nodes.ai_studio_forge_parity import extra_samplers_nodes

        cls.nodes = extra_samplers_nodes

    def test_inputs_are_the_block(self):
        spec = self.nodes.ForgeNeoExtraSamplerSettings.INPUT_TYPES()
        required = spec["required"]
        self.assertEqual(list(required), ["model", *xs.ARG_NAMES])
        self.assertEqual(required["max_stage"][0], "INT")
        self.assertEqual({k: required["max_stage"][1][k] for k in ("default", "min", "max", "step")},
                         {"default": 3, "min": 1, "max": 3, "step": 1})
        self.assertEqual({k: required["eta"][1][k] for k in ("default", "min", "max", "step")},
                         {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.01})
        self.assertEqual((required["noise_window"][0], required["noise_window"][1]["default"]), ("BOOLEAN", False))
        for name, default in (("noise_start", 0.2), ("noise_end", 0.8)):
            self.assertEqual({k: required[name][1][k] for k in ("default", "min", "max", "step")},
                             {"default": default, "min": 0.0, "max": 1.0, "step": 0.01})
        self.assertEqual(list(spec["optional"]), [compat.OWNED_NAMES_INPUT])
        node = self.nodes.ForgeNeoExtraSamplerSettings
        self.assertEqual((node.FUNCTION, node.RETURN_TYPES), ("patch", ("MODEL",)))
        self.assertEqual(self.nodes.NODE_CLASS_MAPPINGS, {"ForgeNeoExtraSamplerSettings": node})
        self.assertIn("ForgeNeoExtraSamplerSettings", self.nodes.NODE_DISPLAY_NAME_MAPPINGS)

    def test_patch_coerces_and_leaves_the_input_model_alone(self):
        original = FakePatcher()
        cases = (
            ((3, 1.0, False, 0.2, 0.8), xs.DEFAULTS),
            ((1, 0.37, True, 0.35, 0.6), xs.Settings(1, 0.37, True, 0.35, 0.6)),
            ((9, -2.0, "on", 1.5, float("nan")), xs.Settings(3, 0.0, True, 1.0, 0.8)),
            ((2.6, float("inf"), 2, True, "0.25"), xs.Settings(3, 1.0, False, 0.2, 0.25)),
        )
        for args, expected in cases:
            with self.subTest(args=args):
                (patched,) = self.nodes.ForgeNeoExtraSamplerSettings().patch(original, *args, owned_names="(none)")
                self.assertIsNot(patched, original)
                self.assertEqual(patched.model_options[xs.MODEL_OPTIONS_KEY], expected.as_dict())
                self.assertEqual(self.nodes.model_settings(patched), expected)
                self.assertEqual(xs.settings_from_model_options(patched.model_options), expected)
                self.assertNotIn(xs.MODEL_OPTIONS_KEY, original.model_options)
        self.assertIsNone(self.nodes.model_settings(original))
        with self.assertRaises(RuntimeError):
            self.nodes.apply_settings(object(), xs.DEFAULTS)

    def test_a_hires_override_keeps_the_values(self):
        self.assertIn(self.nodes.HIRES_CARRY_KEY, hires_carry.registered())
        (source,) = self.nodes.ForgeNeoExtraSamplerSettings().patch(FakePatcher(), 2, 0.5, True, 0.1, 0.9)
        override = FakePatcher()
        carried = hires_carry.apply(source, override, None)
        self.assertIsNot(carried, override)
        self.assertEqual(self.nodes.model_settings(carried), xs.Settings(2, 0.5, True, 0.1, 0.9))
        self.assertIsNone(self.nodes.model_settings(override))
        self.assertIs(hires_carry.apply(FakePatcher(), override, None), override)   # nothing to carry
        self.assertIsNone(self.nodes._carry_to_override(source, carried, None))      # already there


@requires_torch
@requires_extension
class EndToEndTests(unittest.TestCase):
    """Node → model options → the registered sampler on Forge's k-diffusion == sam-extra (bit for bit)."""

    @classmethod
    def setUpClass(cls):
        bind_torch(globals())
        cls.forge_kd = forge_k_sampling()
        with installed_k_sampling(cls.forge_kd):
            cls.er = import_ext("sam3ext.extra_samplers.er_sde")

    def test_settings_node_and_registered_sampler(self):
        from comfy_custom_nodes.ai_studio_forge_parity import extra_samplers_nodes

        kd = types.ModuleType("comfy.k_diffusion.sampling")
        kd.__dict__.update(self.forge_kd.__dict__)
        modules = _fake_comfy(kd=kd)
        settings = (2, 0.6, True, 0.2, 0.8)
        (patched,) = extra_samplers_nodes.ForgeNeoExtraSamplerSettings().patch(FakePatcher(), *settings)
        sigmas = 3.0 * torch.linspace(1, 0, 9, dtype=torch.float64) / (1 + 2.0 * torch.linspace(1, 0, 9, dtype=torch.float64))
        sigmas = sigmas.to(torch.float32)
        x = torch.randn((1, 4, 6, 5), generator=torch.Generator().manual_seed(9)).to(torch.float32)
        for name in ("er_sde_tunable", "er_sde_reverse_time", "er_sde_ode"):
            with self.subTest(name=name):
                with _with_comfy(modules):
                    reg.install_extra_samplers()
                    noise = ReplayNoise(x, 77)
                    ours = getattr(kd, "sample_" + name)(
                        ToyModel(Predictor(True)), x, sigmas, extra_args={"model_options": patched.model_options},
                        disable=True, noise_sampler=noise)
                theirs_noise = ReplayNoise(x, 77)
                kwargs = {"max_stage": 2}
                if name != "er_sde_ode":
                    kwargs.update(er_sde_eta=0.6, er_sde_window=(0.2, 0.8))
                with installed_k_sampling(self.forge_kd):
                    theirs = getattr(self.er, "sample_" + name)(
                        ToyModel(Predictor(True)), x, sigmas, extra_args={}, disable=True, noise_sampler=theirs_noise,
                        **kwargs)
                self.assertTrue(torch.equal(ours, theirs))
                self.assertEqual(noise.draws, theirs_noise.draws)


if __name__ == "__main__":
    unittest.main()
