r"""Pack 1.7.0 Extra Schedulers on the real ComfyUI runtime (CPU) = sam-extra v0.33.0.

The pack's scheduler handlers registered into ComfyUI's own ``comfy.samplers`` and fed by the
``ForgeNeoExtraSchedulers`` carrier on a real ``ModelPatcher`` + ``ModelSamplingDiscreteFlow``, through
every way a sampler asks for sigmas — ``calculate_sigmas`` (Impact's FaceDetailer calls it at module
level), ``KSampler`` (denoise slicing, discard-penultimate samplers), ``BasicScheduler`` — against the
installed extension's functions with the same settings, bit for bit. Also: ComfyUI's
``get_sigmas_laplace`` / ``loglinear_interp`` are what sam-extra's copy / Forge's function are, the
pack's flow shift (ComfyUI's ``ModelSamplingSD3``) keeps the values, and a device/dtype move of the
carrier copy leaves the checkpoint's own ``model_sampling`` alone. React Cosinusoidal DynSF (no values)
also runs on a MODEL without the carrier, flow and eps. Flow Cosmos rho7 (no values either): on ComfyUI's
flow ``model_sampling`` (``CONST``: Anima's, the carrier copy, after the pack's flow shift) the fixed flow list
== sam-extra's == the GPU A/B's custom expression == the upstream numbers; on eps/v ``model_sampling`` it is
ComfyUI's own "karras" schedule == sam-extra running Forge's Karras; through ``KSampler`` too; ComfyUI's
``get_sigmas_karras`` is Forge's function. Its three values (sam-extra v0.33.0 D19: ρ, σ̃ max, σ̃ min) set on
``ForgeNeoExtraSchedulers`` reach every sampler path like Laplace's μ/β: flow == sam-extra's list for those values,
eps/v == ComfyUI's Karras with that ρ (σ̃ unused), kept by the pack's flow shift; the one pair the node's
ranges let through that a flow model refuses (σ̃ min = σ̃ max = 1) stops ``calculate_sigmas`` with sam-extra's text.
Flow Cosmos Dynamic (sam-extra v0.33.0 D20, the same three values): on flow ``model_sampling`` Karras Dynamic's
per-step-ρ ramp over σ̃ in flow time == sam-extra's (the MODEL's own range unused, kept by the pack's flow shift); on
eps/v ``model_sampling`` Karras Dynamic with the node's ρ == sam-extra == ComfyUI's ``karras_dynamic`` handler of the
pack at ρ 7; through ``KSampler`` (plain, denoise slicing, discard-penultimate) and ``BasicScheduler``; a ρ it refuses
or a rising list stops the run with sam-extra's text; its own one-per-process log line.

Skipped unless ``AISTUDIO_COMFY_TEST_ROOT`` is set. Run it with the portable python, the repo (or
worktree) first on ``sys.path``, ``CUDA_VISIBLE_DEVICES=-1`` (ComfyUI is forced to ``--cpu`` before its
model management loads; no server, no GPU)::

    $env:CUDA_VISIBLE_DEVICES='-1'; $env:AISTUDIO_COMFY_TEST_ROOT='C:\ComfyUI_windows_portable\ComfyUI'
    $env:AISTUDIO_REQUIRE_FORGE_EXT='1'
    C:\ComfyUI_windows_portable\python_embeded\python.exe -X utf8 -c "import sys, unittest; sys.path.insert(0, r'<repo>'); unittest.main(module=None, argv=['x', 'tests.test_comfy_extra_schedulers_runtime', '-v'])"

Until sam-extra v0.33.0 is installed: ``AISTUDIO_FORGE_EXTENSION_DIR=<its work tree>`` and
``SAM3_FORGE_ROOT=C:\sd-webui-forge-classic`` (``tests/_sam_extra_ext.forge_root``).
"""
from __future__ import annotations

import ast
import importlib
import os
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest import mock

_COMFY_ROOT = os.environ.get("AISTUDIO_COMFY_TEST_ROOT")
if _COMFY_ROOT:  # force --cpu before anything loads ComfyUI's model management
    if _COMFY_ROOT not in sys.path:
        sys.path.append(_COMFY_ROOT)
    from comfy.cli_args import args as _comfy_args  # noqa: E402

    _comfy_args.cpu = True

from comfy_custom_nodes.ai_studio_forge_parity import extra_schedulers_math as xmath  # noqa: E402
from tests._forge_k_diffusion import forge_k_sampling, installed_k_sampling  # noqa: E402
from tests._optional_deps import bind_torch, requires_torch  # noqa: E402
from tests._sam_extra_ext import EXT_ROOT, FORGE_ROOT, requires_extension  # noqa: E402
# Flow Cosmos rho7's A/B expression and the upstream test reference (inline there, never shipped)
from tests.test_comfy_extra_schedulers_parity import COSMOS_AB_EXPRESSION, upstream_flow_cosmos_rho  # noqa: E402

torch = None  # filled by bind_torch()


def _ext(name: str):
    root = str(EXT_ROOT)
    if root not in sys.path:
        sys.path.append(root)
    return importlib.import_module(f"sam3ext.extra_schedulers.{name}")


def _forge_loglinear_interp():
    import numpy as np

    path = FORGE_ROOT / "modules" / "sd_schedulers.py"
    node = next(item for item in ast.parse(path.read_text(encoding="utf-8")).body
                if isinstance(item, ast.FunctionDef) and item.name == "_loglinear_interp")
    scope = {"np": np}
    exec(compile(ast.Module(body=[node], type_ignores=[]), f"{path.name}:_loglinear_interp", "exec"), scope)
    return scope["_loglinear_interp"]


def _bits(tensor) -> tuple:
    return str(tensor.dtype), tuple(tensor.shape), tuple(tensor.contiguous().view(torch.int32).tolist())


@requires_torch
@requires_extension
@unittest.skipUnless(_COMFY_ROOT, "Set AISTUDIO_COMFY_TEST_ROOT to run Extra Schedulers on the real ComfyUI runtime (CPU)")
class ExtraSchedulersOnRealComfyTests(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        bind_torch(globals())
        import comfy.model_management as management

        if management.get_torch_device().type != "cpu":
            raise unittest.SkipTest("ComfyUI did not start on CPU - refusing to touch the GPU")
        import comfy.model_patcher
        import comfy.model_sampling
        import comfy.samplers
        from comfy_extras.nodes_custom_sampler import BasicScheduler
        from comfy_extras.nodes_model_advanced import ModelSamplingSD3

        from comfy_custom_nodes.ai_studio_forge_parity import extra_schedulers, generation

        cls.samplers, cls.model_patcher, cls.model_sampling = comfy.samplers, comfy.model_patcher, comfy.model_sampling
        cls.basic_scheduler, cls.sd3 = BasicScheduler, ModelSamplingSD3
        cls.pack, cls.generation = extra_schedulers, generation
        cls.registration = extra_schedulers.install_extra_schedulers(required=True)
        cls.flow_class = type(
            "AnimaFlowSampling",
            (comfy.model_sampling.ModelSamplingDiscreteFlow, comfy.model_sampling.CONST), {},
        )
        cls.ext = _ext("schedulers")
        cls.ext_settings = _ext("settings")
        cls.ext_sigma_list = _ext("sigma_list")
        cls.interp = staticmethod(_forge_loglinear_interp())
        cls.forge_k = forge_k_sampling()   # Forge's k_diffusion.sampling (Flow Cosmos rho7's eps/v branch in sam-extra)
        # ComfyUI's model_base.model_sampling compositions for SD/SDXL eps and v, and Cosmos-Predict2's
        cls.eps_class = type("EpsSampling", (comfy.model_sampling.ModelSamplingDiscrete, comfy.model_sampling.EPS), {})
        cls.v_class = type("VSampling", (comfy.model_sampling.ModelSamplingDiscrete,
                                         comfy.model_sampling.V_PREDICTION), {})
        cls.cosmos_class = type("CosmosRFlowSampling", (comfy.model_sampling.ModelSamplingCosmosRFlow,
                                                        comfy.model_sampling.COSMOS_RFLOW), {})

    def setUp(self):
        for module in (self.pack, self.ext):   # the Flow Cosmos schedulers' one log line per process: quiet here
            for flag in ("_flow_cosmos_not_flow_logged", "_flow_cosmos_dynamic_not_flow_logged"):
                patcher = mock.patch.object(module, flag, True)
                patcher.start()
                self.addCleanup(patcher.stop)

    def tearDown(self):
        self.ext_settings.reset()

    def _patcher(self, shift: float = 3.0, multiplier: float = 1.0):
        flow_class = self.flow_class

        class Tiny(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.model_config = SimpleNamespace(sampling_settings={"shift": shift, "multiplier": multiplier})
                self.model_sampling = flow_class(self.model_config)
                self.weight = torch.nn.Parameter(torch.zeros(1))

        cpu = torch.device("cpu")
        return self.model_patcher.ModelPatcher(Tiny(), cpu, cpu)

    def _carried(self, base=None, **values):
        inputs = {"custom_mode": xmath.MODE_EXPRESSION, "custom_expression": xmath.DEFAULT_EXPRESSION,
                  "custom_sigmas": xmath.DEFAULT_SIGMAS, "laplace_mu": 0.0, "laplace_beta": 0.5,
                  "flow_cosmos_rho": 7.0, "flow_cosmos_sigma_max": 80.0, "flow_cosmos_sigma_min": 0.002, **values}
        return self.pack.ForgeNeoExtraSchedulers().patch(base or self._patcher(), **inputs)[0]

    def _expected(self, name: str, steps: int, sampling, settings: xmath.Settings):
        """The extension's function with Forge's arguments (model_wrap.sigmas[0]/[-1].item())."""
        functions = {"cosine": self.ext.cosine, "cosine_exponential": self.ext.cosine_exponential_blend,
                     "phi": self.ext.phi, "laplace": self.ext.laplace, "karras_dynamic": self.ext.karras_dynamic,
                     "custom": self.ext.custom, "react_cosinusoidal_dynsf": self.ext.react_cosinusoidal_dynsf,
                     "flow_cosmos_rho7": self.ext.flow_cosmos_rho7, "flow_cosmos_dynamic": self.ext.flow_cosmos_dynamic}
        self.ext_settings.set_active(self.ext_settings.ExtraSchedulerSettings(**settings._asdict()))
        extra = {}
        if name in ("flow_cosmos_rho7", "flow_cosmos_dynamic"):
            # Forge's model_wrap.predictor: "const" for a flow model, eps/v otherwise
            kind = "const" if isinstance(sampling, self.model_sampling.CONST) else (
                "v_prediction" if isinstance(sampling, self.model_sampling.V_PREDICTION) else "epsilon")
            extra["inner_model"] = SimpleNamespace(predictor=SimpleNamespace(prediction_type=kind))
        with mock.patch.object(self.ext_sigma_list, "forge_loglinear_interp", return_value=self.interp), \
                installed_k_sampling(self.forge_k):
            return functions[name](n=steps, sigma_min=sampling.sigmas[0].item(),
                                   sigma_max=sampling.sigmas[-1].item(), device="cpu", **extra)

    def _eps_patcher(self, kind: str = "eps"):
        sampling_class = self.eps_class if kind == "eps" else self.v_class

        class Tiny(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.model_config = SimpleNamespace(sampling_settings={})
                self.model_sampling = sampling_class(self.model_config)
                self.weight = torch.nn.Parameter(torch.zeros(1))

        cpu = torch.device("cpu")
        return self.model_patcher.ModelPatcher(Tiny(), cpu, cpu)

    def test_the_nine_names_are_comfyui_schedulers_of_the_pack(self):
        names = self.pack.SCHEDULER_NAMES
        self.assertEqual(len(names), 9)
        self.assertEqual(names[-3:], ("react_cosinusoidal_dynsf", "flow_cosmos_rho7", "flow_cosmos_dynamic"))
        self.assertEqual(self.registration.foreign, ())
        self.assertEqual(self.registration.owned, names)
        for name in names:
            self.assertIn(name, self.samplers.KSampler.SCHEDULERS)
            self.assertIn(name, self.samplers.SCHEDULER_NAMES)
            self.assertTrue(self.samplers.SCHEDULER_HANDLERS[name].use_ms)
        self.assertEqual(self.pack.owned_names(), names)
        choices = self.pack.ForgeNeoExtraSchedulers.INPUT_TYPES()["optional"]["owned_names"][0]
        self.assertEqual(choices, list(names))
        again = self.pack.install_extra_schedulers(required=True)
        self.assertEqual((again.owned, again.inserted), (names, ()))
        self.assertEqual(len(self.samplers.KSampler.SCHEDULERS), len(set(self.samplers.KSampler.SCHEDULERS)))

    def test_calculate_sigmas_is_the_extension_schedule(self):
        cases = (
            {},
            {"laplace_mu": 0.5, "laplace_beta": 1.0},
            {"laplace_mu": -2.0, "laplace_beta": 3.0},
            {"custom_expression": "m + (M - m) * (1 - x) ** 3"},
            {"custom_mode": "sigmas", "custom_sigmas": "[1.0, 0.6, 0.25, 0.1, 0.0]"},
            {"custom_mode": "sigmas", "custom_sigmas": "[0.99, 0.7, 0.31, 0.12, 0.004]"},
            {"flow_cosmos_rho": 4.5, "flow_cosmos_sigma_max": 240.0, "flow_cosmos_sigma_min": 0.006},
            {"flow_cosmos_rho": 12.0, "flow_cosmos_sigma_max": 999.0, "flow_cosmos_sigma_min": 0.5,
             "laplace_mu": 0.5, "laplace_beta": 1.0},
        )
        for shift in (1.0, 3.0, 5.0):
            for values in cases:
                carried = self._carried(self._patcher(shift), **values)
                sampling = carried.get_model_object("model_sampling")
                settings = self.pack.model_settings(carried)
                for name in self.pack.SCHEDULER_NAMES:
                    for steps in (1, 4, 12, 28):
                        with self.subTest(shift=shift, values=values, name=name, steps=steps):
                            got = self.samplers.calculate_sigmas(sampling, name, steps)
                            want = self._expected(name, steps, sampling, settings)
                            self.assertEqual(_bits(got), _bits(want))

    def test_react_cosinusoidal_dynsf_without_a_carrier(self):
        """``calculate_sigmas(ms, "react_cosinusoidal_dynsf", 28)`` — no values, so no
        ``ForgeNeoExtraSchedulers`` node is needed: a plain flow MODEL (Anima shift 3), an eps MODEL
        (SD/SDXL's discrete sampling) and KSampler's own path all give the extension's schedule."""
        flow = self._patcher(3.0).get_model_object("model_sampling")
        eps = self.model_sampling.ModelSamplingDiscrete(SimpleNamespace(sampling_settings={}))
        for sampling in (flow, eps):
            for steps in (1, 2, 3, 28, 50, 150):
                with self.subTest(sampling=type(sampling).__name__, steps=steps):
                    got = self.samplers.calculate_sigmas(sampling, "react_cosinusoidal_dynsf", steps)
                    self.assertEqual(_bits(got), _bits(self._expected("react_cosinusoidal_dynsf", steps, sampling,
                                                                      xmath.DEFAULTS)))
        anima = [float(value) for value in self.samplers.calculate_sigmas(flow, "react_cosinusoidal_dynsf", 28)[:-1]]
        self.assertEqual((sum(value > 0.5 for value in anima), sum(value < 0.05 for value in anima)), (17, 4))
        ksampler = self.samplers.KSampler(self._patcher(3.0), 28, "cpu", "euler", "react_cosinusoidal_dynsf").sigmas
        self.assertEqual(_bits(ksampler), _bits(self._expected("react_cosinusoidal_dynsf", 28, flow, xmath.DEFAULTS)))
        with self.assertRaisesRegex(ValueError, "React Cosinusoidal DynSF gives sigma"):
            self.samplers.calculate_sigmas(flow, "react_cosinusoidal_dynsf", 1000)

    def test_flow_cosmos_rho7_on_flow_and_eps_model_sampling(self):
        """``calculate_sigmas(ms, "flow_cosmos_rho7", n)`` — no values, so no ``ForgeNeoExtraSchedulers`` node is
        needed. Flow ``model_sampling`` (Anima shift 3, shift 1 and 5, the carrier's private copy, the pack's flow
        shift on top of it): the fixed flow list == sam-extra's == the A/B expression through the pack's
        ``custom`` == the upstream numbers. eps and v (SD/SDXL discrete sampling): ComfyUI's own "karras"
        schedule of that ``model_sampling`` == sam-extra running Forge's Karras on the same range."""
        steps_grid = (1, 2, 3, 28, 50, 150)
        carried = self._carried(self._patcher(3.0), custom_expression=COSMOS_AB_EXPRESSION)

        def provider(name, *, method, feature, args=(), kwargs=None):
            self.assertEqual(name, "ModelSamplingSD3")
            return self.sd3().patch(*args)

        with mock.patch.object(self.generation, "invoke_provider", side_effect=provider):
            shifted = self.generation._patch_flow_shift(carried.clone(), 5.0)
        flows = {f"shift {shift:g}": self._patcher(shift).get_model_object("model_sampling") for shift in (3.0, 1.0, 5.0)}
        flows["carrier copy"] = carried.get_model_object("model_sampling")
        flows["pack flow shift 5"] = shifted.get_model_object("model_sampling")
        reference = {}
        for label, sampling in flows.items():
            self.assertIsInstance(sampling, self.model_sampling.CONST, label)
            for steps in steps_grid:
                with self.subTest(model=label, steps=steps):
                    got = self.samplers.calculate_sigmas(sampling, "flow_cosmos_rho7", steps)
                    want = self._expected("flow_cosmos_rho7", steps, sampling, xmath.DEFAULTS)
                    self.assertEqual(_bits(got), _bits(want))
                    self.assertTrue(torch.equal(got, want))
                    self.assertTrue(torch.equal(got, reference.setdefault(steps, got)))   # independent of the range
                    by_expression = self.samplers.calculate_sigmas(flows["carrier copy"], "custom", steps)
                    self.assertTrue(torch.equal(got, by_expression))
                    if steps > 1:
                        self.assertTrue(torch.equal(got, torch.tensor(upstream_flow_cosmos_rho(steps),
                                                                      dtype=torch.float32)))
        for kind in ("eps", "v"):
            sampling = self._eps_patcher(kind).get_model_object("model_sampling")
            for steps in steps_grid:
                with self.subTest(model=kind, steps=steps):
                    got = self.samplers.calculate_sigmas(sampling, "flow_cosmos_rho7", steps)
                    self.assertTrue(torch.equal(got, self.samplers.calculate_sigmas(sampling, "karras", steps)))
                    want = self._expected("flow_cosmos_rho7", steps, sampling, xmath.DEFAULTS)
                    self.assertEqual(_bits(got), _bits(want))
                    self.assertTrue(torch.equal(got, want))
        self.assertAlmostEqual(float(sampling.sigma_max), 14.614642, places=5)

    def test_flow_cosmos_rho7_through_ksampler_and_the_log_line(self):
        """KSampler's own path (plain, denoise slicing, a discard-penultimate sampler) on a flow and an eps MODEL;
        an eps/v model logs sam-extra's line once per process, a flow model nothing."""
        flow_model, eps_model = self._patcher(3.0), self._eps_patcher("eps")
        flow = flow_model.get_model_object("model_sampling")
        eps = eps_model.get_model_object("model_sampling")
        for model, sampling in ((flow_model, flow), (eps_model, eps)):
            with self.subTest(model=type(sampling).__name__):
                plain = self.samplers.KSampler(model, 28, "cpu", "euler", "flow_cosmos_rho7").sigmas
                self.assertEqual(_bits(plain), _bits(self._expected("flow_cosmos_rho7", 28, sampling,
                                                                    xmath.DEFAULTS)))
                sliced = self.samplers.KSampler(model, 28, "cpu", "euler", "flow_cosmos_rho7", denoise=0.5).sigmas
                full = self._expected("flow_cosmos_rho7", int(28 / 0.5), sampling, xmath.DEFAULTS)
                self.assertEqual(_bits(sliced), _bits(full[-29:]))
                discard = self.samplers.KSampler(model, 10, "cpu", "dpm_2", "flow_cosmos_rho7").sigmas
                longer = self._expected("flow_cosmos_rho7", 11, sampling, xmath.DEFAULTS)
                self.assertEqual(_bits(discard), _bits(torch.cat([longer[:-2], longer[-1:]])))
        with mock.patch.object(self.pack, "_flow_cosmos_not_flow_logged", False):
            with self.assertNoLogs("ai_studio_forge_parity", level="INFO"):
                self.samplers.calculate_sigmas(flow, "flow_cosmos_rho7", 28)
            with self.assertLogs("ai_studio_forge_parity", level="INFO") as logs:
                self.samplers.calculate_sigmas(eps, "flow_cosmos_rho7", 28)
                self.samplers.KSampler(eps_model, 20, "cpu", "euler", "flow_cosmos_rho7")
                self.samplers.calculate_sigmas(self._eps_patcher("v").get_model_object("model_sampling"),
                                               "flow_cosmos_rho7", 12)
        self.assertEqual([record.getMessage() for record in logs.records],
                         ["[Extra Schedulers] Flow Cosmos rho7: not a flow model - using Karras rho 7 on the model's "
                          "own sigma range"])
        # the line names the ρ of the first eps/v run (sam-extra v0.33.0's template)
        knob = self._carried(self._eps_patcher("eps"), flow_cosmos_rho=3.5)
        with mock.patch.object(self.pack, "_flow_cosmos_not_flow_logged", False):
            with self.assertLogs("ai_studio_forge_parity", level="INFO") as logs:
                self.samplers.KSampler(knob, 20, "cpu", "euler", "flow_cosmos_rho7")
                self.samplers.calculate_sigmas(eps, "flow_cosmos_rho7", 28)
        self.assertEqual([record.getMessage() for record in logs.records],
                         ["[Extra Schedulers] Flow Cosmos rho7: not a flow model - using Karras rho 3.5 on the model's "
                          "own sigma range"])

    def test_flow_cosmos_rho7_values_through_the_node(self):
        """D19 on ComfyUI's runtime: ``ForgeNeoExtraSchedulers``' Flow Cosmos values reach ``calculate_sigmas``,
        ``KSampler`` (plain, denoise slicing, discard-penultimate) and ``BasicScheduler``. Flow (Anima shift 3, and
        after the pack's flow shift 5): sam-extra's list for those values, == the A/B expression with them through
        the pack's ``custom`` — whatever the MODEL's own range; eps/v: ComfyUI's ``get_sigmas_karras`` with that ρ
        (its built-in "karras" when ρ is 7) == sam-extra running Forge's Karras, σ̃ unused; with ρ 7 and σ̃ ×3 the flow
        list is the default one shifted by 3 in flow time (float32 rounding apart)."""
        from comfy_custom_nodes.ai_studio_forge_parity.compat import node_result
        from tests.test_comfy_extra_schedulers_parity import cosmos_expression

        import comfy.k_diffusion.sampling as k_sampling

        grid = ((4.5, 240.0, 0.006), (1.0, 1000.0, 0.0001), (15.0, 1.0, 0.0001), (9.3, 14.614642, 0.0291675),
                (7.0, 80.0, 0.002))

        def provider(name, *, method, feature, args=(), kwargs=None):
            self.assertEqual(name, "ModelSamplingSD3")
            return self.sd3().patch(*args)

        for rho, top, bottom in grid:
            values = {"flow_cosmos_rho": rho, "flow_cosmos_sigma_max": top, "flow_cosmos_sigma_min": bottom}
            flow_model = self._carried(self._patcher(3.0), custom_expression=cosmos_expression(rho, top, bottom),
                                       **values)
            with mock.patch.object(self.generation, "invoke_provider", side_effect=provider):
                shifted = self.generation._patch_flow_shift(flow_model.clone(), 5.0)
            flow = flow_model.get_model_object("model_sampling")
            settings = self.pack.model_settings(flow_model)
            self.assertEqual(tuple(settings)[5:], (rho, top, bottom))
            self.assertEqual(self.pack.model_settings(shifted), settings)
            for steps in (1, 2, 3, 28, 50, 151):
                with self.subTest(values=values, steps=steps, model="flow"):
                    got = self.samplers.calculate_sigmas(flow, "flow_cosmos_rho7", steps)
                    want = self._expected("flow_cosmos_rho7", steps, flow, settings)
                    self.assertEqual(_bits(got), _bits(want))
                    self.assertTrue(torch.equal(got, self.samplers.calculate_sigmas(flow, "custom", steps)))
                    after_shift = self.samplers.calculate_sigmas(shifted.get_model_object("model_sampling"),
                                                                 "flow_cosmos_rho7", steps)
                    self.assertTrue(torch.equal(after_shift, got))   # the MODEL's own range is not used
            plain = self.samplers.KSampler(flow_model, 28, "cpu", "euler", "flow_cosmos_rho7").sigmas
            self.assertEqual(_bits(plain), _bits(self._expected("flow_cosmos_rho7", 28, flow, settings)))
            sliced = self.samplers.KSampler(flow_model, 28, "cpu", "euler", "flow_cosmos_rho7", denoise=0.5).sigmas
            self.assertEqual(_bits(sliced), _bits(self._expected("flow_cosmos_rho7", 56, flow, settings)[-29:]))
            discard = self.samplers.KSampler(flow_model, 10, "cpu", "dpm_2", "flow_cosmos_rho7").sigmas
            longer = self._expected("flow_cosmos_rho7", 11, flow, settings)
            self.assertEqual(_bits(discard), _bits(torch.cat([longer[:-2], longer[-1:]])))
            basic = node_result(self.basic_scheduler.execute(flow_model, "flow_cosmos_rho7", 20, 1.0))[0]
            self.assertEqual(_bits(basic), _bits(self._expected("flow_cosmos_rho7", 20, flow, settings)))
            for kind in ("eps", "v"):
                model = self._carried(self._eps_patcher(kind), **values)
                sampling = model.get_model_object("model_sampling")
                for steps in (1, 2, 28, 151):
                    with self.subTest(values=values, steps=steps, model=kind):
                        got = self.samplers.calculate_sigmas(sampling, "flow_cosmos_rho7", steps)
                        self.assertEqual(_bits(got), _bits(self._expected("flow_cosmos_rho7", steps, sampling,
                                                                          settings)))
                        karras = k_sampling.get_sigmas_karras(steps, float(sampling.sigma_min),
                                                              float(sampling.sigma_max), rho=rho)
                        self.assertTrue(torch.equal(got, karras))
                        if rho == 7.0:
                            self.assertTrue(torch.equal(got, self.samplers.calculate_sigmas(sampling, "karras", steps)))
                        other_range = self._carried(self._eps_patcher(kind), flow_cosmos_rho=rho,
                                                    flow_cosmos_sigma_max=1.0, flow_cosmos_sigma_min=1.0)
                        self.assertTrue(torch.equal(self.samplers.calculate_sigmas(
                            other_range.get_model_object("model_sampling"), "flow_cosmos_rho7", steps), got))
        # σ̃ ×3 with ρ 7 is Anima's shift 3 of the default list in flow time (a maths fact, not a quality claim)
        default = self.samplers.calculate_sigmas(self._patcher(3.0).get_model_object("model_sampling"),
                                                 "flow_cosmos_rho7", 28).double()
        tripled = self._carried(self._patcher(3.0), flow_cosmos_sigma_max=240.0, flow_cosmos_sigma_min=0.006)
        tripled_list = self.samplers.calculate_sigmas(tripled.get_model_object("model_sampling"), "flow_cosmos_rho7", 28)
        shifted_default = (3.0 * default / (1.0 + 2.0 * default)).float()
        self.assertLessEqual(float((tripled_list - shifted_default).abs().max()), 2e-7)
        self.assertFalse(torch.equal(tripled_list, default.float()))

    def test_flow_cosmos_values_a_flow_model_refuses_stop_the_run(self):
        """The node's ranges meet at σ̃ min = σ̃ max = 1: carried as given, refused when a flow schedule is built (the
        sampler node's error, sam-extra's text); an eps/v MODEL does not use σ̃ and runs."""
        flow_model = self._carried(self._patcher(3.0), flow_cosmos_sigma_max=1.0, flow_cosmos_sigma_min=1.0)
        message = ("Extra Schedulers: Flow Cosmos rho7 needs 0 < sigma~ min < sigma~ max, both finite, got sigma~ min 1 "
                   "and sigma~ max 1 (Extra Schedulers accordion → Flow Cosmos rho / sigma max / sigma min)")
        with self.assertRaises(ValueError) as caught:
            self.samplers.calculate_sigmas(flow_model.get_model_object("model_sampling"), "flow_cosmos_rho7", 28)
        self.assertEqual(str(caught.exception), message)
        self.assertIsInstance(caught.exception, xmath.ExtraSchedulerError)
        with self.assertRaises(ValueError):
            self.samplers.KSampler(flow_model, 28, "cpu", "euler", "flow_cosmos_rho7")
        eps_model = self._carried(self._eps_patcher("eps"), flow_cosmos_sigma_max=1.0, flow_cosmos_sigma_min=1.0)
        sampling = eps_model.get_model_object("model_sampling")
        self.assertTrue(torch.equal(self.samplers.calculate_sigmas(sampling, "flow_cosmos_rho7", 28),
                                    self.samplers.calculate_sigmas(sampling, "karras", 28)))

    def test_flow_cosmos_rho7_on_comfyui_s_cosmos_predict2_sampling(self):
        """ComfyUI runs Cosmos-Predict2 with ``COSMOS_RFLOW`` (EDM sigmas 0.002–80, which the model maps to
        t = σ/(σ + 1) itself), not ``CONST``: it gets Karras ρ 7 on its own range — ComfyUI's "karras" schedule,
        the same EDM ramp. In flow time that is the flow list up to float32 rounding (at most 2.98e-7 over these
        step counts, measured 2026-10-04)."""
        sampling = self.cosmos_class(SimpleNamespace(sampling_settings={"sigma_data": 1.0, "sigma_max": 80.0,
                                                                         "sigma_min": 0.002}))
        self.assertNotIsInstance(sampling, self.model_sampling.CONST)
        flow = self._patcher(3.0).get_model_object("model_sampling")
        for steps in (2, 3, 28, 50, 150):
            with self.subTest(steps=steps):
                got = self.samplers.calculate_sigmas(sampling, "flow_cosmos_rho7", steps)
                self.assertTrue(torch.equal(got, self.samplers.calculate_sigmas(sampling, "karras", steps)))
                in_flow_time = got / (1 + got)
                flow_list = self.samplers.calculate_sigmas(flow, "flow_cosmos_rho7", steps)
                self.assertLessEqual(float((in_flow_time - flow_list).abs().max()), 3e-7)

    def _shifted(self, model, shift: float):
        """The pack's flow shift (ComfyUI's ``ModelSamplingSD3``) on a clone of ``model``."""
        def provider(name, *, method, feature, args=(), kwargs=None):
            self.assertEqual(name, "ModelSamplingSD3")
            return self.sd3().patch(*args)

        with mock.patch.object(self.generation, "invoke_provider", side_effect=provider):
            return self.generation._patch_flow_shift(model.clone(), shift)

    @staticmethod
    def _error_of(call) -> tuple[str, str] | None:
        try:
            call()
        except Exception as exc:  # the class and the text are what the sampler node shows
            return type(exc).__name__, str(exc)
        return None

    def test_flow_cosmos_dynamic_on_flow_and_eps_model_sampling(self):
        """``calculate_sigmas(ms, "flow_cosmos_dynamic", n)`` without the node (the defaults: ρ 7 on σ̃ 0.002 … 80).
        Flow ``model_sampling`` (Anima shift 3, shift 1 and 5, the carrier's private copy, the pack's flow shift on top
        of it): one list for every range == sam-extra's == the pack's float64 flow times as float32, starting and
        ending where Flow Cosmos rho7's list does. eps and v (SD/SDXL discrete sampling): Karras Dynamic ρ 7 on that
        ``model_sampling``'s range — the pack's own ``karras_dynamic`` handler — == sam-extra."""
        steps_grid = (1, 2, 3, 4, 28, 50, 150)
        carried = self._carried(self._patcher(3.0))
        flows = {f"shift {shift:g}": self._patcher(shift).get_model_object("model_sampling") for shift in (3.0, 1.0, 5.0)}
        flows["carrier copy"] = carried.get_model_object("model_sampling")
        flows["pack flow shift 5"] = self._shifted(carried, 5.0).get_model_object("model_sampling")
        reference = {}
        for label, sampling in flows.items():
            self.assertIsInstance(sampling, self.model_sampling.CONST, label)
            for steps in steps_grid:
                with self.subTest(model=label, steps=steps):
                    got = self.samplers.calculate_sigmas(sampling, "flow_cosmos_dynamic", steps)
                    want = self._expected("flow_cosmos_dynamic", steps, sampling, xmath.DEFAULTS)
                    self.assertEqual(_bits(got), _bits(want))
                    self.assertTrue(torch.equal(got, reference.setdefault(steps, got)))   # independent of the range
                    times, bad = xmath.flow_cosmos_dynamic_times(steps)
                    self.assertIsNone(bad)
                    self.assertTrue(torch.equal(got, torch.tensor([*times, 0.0], dtype=torch.float32)))
                    fixed = self.samplers.calculate_sigmas(sampling, "flow_cosmos_rho7", steps)
                    self.assertEqual((float(got[0]), float(got[-2])), (float(fixed[0]), float(fixed[-2])))
        for kind in ("eps", "v"):
            sampling = self._eps_patcher(kind).get_model_object("model_sampling")
            for steps in steps_grid:
                with self.subTest(model=kind, steps=steps):
                    got = self.samplers.calculate_sigmas(sampling, "flow_cosmos_dynamic", steps)
                    self.assertTrue(torch.equal(got, self.samplers.calculate_sigmas(sampling, "karras_dynamic", steps)))
                    want = self._expected("flow_cosmos_dynamic", steps, sampling, xmath.DEFAULTS)
                    self.assertEqual(_bits(got), _bits(want))
                    self.assertTrue(torch.equal(got, want))

    def test_flow_cosmos_dynamic_values_through_the_node(self):
        """D20 on ComfyUI's runtime: ``ForgeNeoExtraSchedulers``' Flow Cosmos values reach Flow Cosmos Dynamic through
        ``calculate_sigmas``, ``KSampler`` (plain, denoise slicing, discard-penultimate) and ``BasicScheduler``. Flow
        (Anima shift 3, and after the pack's flow shift 5): sam-extra's list for those values == the pack's float64
        times, whatever the MODEL's own range; eps/v: Karras Dynamic with that ρ on the MODEL's range == sam-extra (the
        pack's ``karras_dynamic`` handler when ρ is 7), σ̃ unused."""
        from comfy_custom_nodes.ai_studio_forge_parity.compat import node_result

        grid = ((4.5, 240.0, 0.006), (15.0, 1.0, 0.0001), (9.3, 14.614642, 0.0291675), (5.0, 1000.0, 0.0001),
                (7.0, 80.0, 0.002))
        for rho, top, bottom in grid:
            values = {"flow_cosmos_rho": rho, "flow_cosmos_sigma_max": top, "flow_cosmos_sigma_min": bottom}
            flow_model = self._carried(self._patcher(3.0), **values)
            shifted = self._shifted(flow_model, 5.0)
            flow = flow_model.get_model_object("model_sampling")
            settings = self.pack.model_settings(flow_model)
            self.assertEqual(tuple(settings)[5:], (rho, top, bottom))
            self.assertEqual(self.pack.model_settings(shifted), settings)
            for steps in (1, 2, 3, 28, 50, 151):
                with self.subTest(values=values, steps=steps, model="flow"):
                    got = self.samplers.calculate_sigmas(flow, "flow_cosmos_dynamic", steps)
                    self.assertEqual(_bits(got), _bits(self._expected("flow_cosmos_dynamic", steps, flow, settings)))
                    times, bad = xmath.flow_cosmos_dynamic_times(steps, rho, bottom, top)
                    self.assertIsNone(bad)
                    self.assertTrue(torch.equal(got, torch.tensor([*times, 0.0], dtype=torch.float32)))
                    after_shift = self.samplers.calculate_sigmas(shifted.get_model_object("model_sampling"),
                                                                 "flow_cosmos_dynamic", steps)
                    self.assertTrue(torch.equal(after_shift, got))   # the MODEL's own range is not used
            plain = self.samplers.KSampler(flow_model, 28, "cpu", "euler", "flow_cosmos_dynamic").sigmas
            self.assertEqual(_bits(plain), _bits(self._expected("flow_cosmos_dynamic", 28, flow, settings)))
            sliced = self.samplers.KSampler(flow_model, 28, "cpu", "euler", "flow_cosmos_dynamic", denoise=0.5).sigmas
            self.assertEqual(_bits(sliced), _bits(self._expected("flow_cosmos_dynamic", 56, flow, settings)[-29:]))
            discard = self.samplers.KSampler(flow_model, 10, "cpu", "dpm_2", "flow_cosmos_dynamic").sigmas
            longer = self._expected("flow_cosmos_dynamic", 11, flow, settings)
            self.assertEqual(_bits(discard), _bits(torch.cat([longer[:-2], longer[-1:]])))
            basic = node_result(self.basic_scheduler.execute(flow_model, "flow_cosmos_dynamic", 20, 1.0))[0]
            self.assertEqual(_bits(basic), _bits(self._expected("flow_cosmos_dynamic", 20, flow, settings)))
            for kind in ("eps", "v"):
                model = self._carried(self._eps_patcher(kind), **values)
                sampling = model.get_model_object("model_sampling")
                other_range = self._carried(self._eps_patcher(kind), flow_cosmos_rho=rho, flow_cosmos_sigma_max=1.0,
                                            flow_cosmos_sigma_min=1.0).get_model_object("model_sampling")
                for steps in (1, 2, 28, 151):
                    with self.subTest(values=values, steps=steps, model=kind):
                        got = self.samplers.calculate_sigmas(sampling, "flow_cosmos_dynamic", steps)
                        self.assertEqual(_bits(got), _bits(self._expected("flow_cosmos_dynamic", steps, sampling,
                                                                          settings)))
                        self.assertTrue(torch.equal(got, self.pack.karras_dynamic(
                            steps, float(sampling.sigma_min), float(sampling.sigma_max), rho=rho)))
                        if rho == 7.0:
                            self.assertTrue(torch.equal(got, self.samplers.calculate_sigmas(sampling, "karras_dynamic",
                                                                                            steps)))
                        self.assertTrue(torch.equal(self.samplers.calculate_sigmas(other_range, "flow_cosmos_dynamic",
                                                                                   steps), got))

    def test_flow_cosmos_dynamic_refusals_stop_the_run_and_its_log_line(self):
        """What Flow Cosmos Dynamic refuses stops ``calculate_sigmas`` and ``KSampler`` with sam-extra's class and text
        (the sampler node's error on ComfyUI): ρ 1 — the node's minimum; it needs ρ above 2 — on both models, a ρ whose
        list rises (3.0, from 4 steps on a flow model and 6 on SDXL's range), σ̃ min = σ̃ max = 1 on a flow model only.
        An eps/v model logs sam-extra's line once per process with the first such run's ρ (its own line, apart from
        Flow Cosmos rho7's); a flow model logs nothing."""
        low_rho = ("Extra Schedulers: Flow Cosmos Dynamic needs rho above 2 (its per-step exponent goes down to rho - 2), "
                   f"got 1. Set rho to about 4 or more: {xmath.FLOW_COSMOS_VALUES_SOURCE}.")
        # (model, MODEL, steps, class, text start, a step count that still runs)
        cases = (
            ("flow", self._carried(self._patcher(3.0), flow_cosmos_rho=1.0), 28, "KarrasDynamicError", low_rho, None),
            ("eps", self._carried(self._eps_patcher("eps"), flow_cosmos_rho=1.0), 28, "KarrasDynamicError", low_rho,
             None),
            ("flow", self._carried(self._patcher(3.0), flow_cosmos_rho=3.0), 4, "KarrasDynamicError",
             "Extra Schedulers: Flow Cosmos Dynamic with rho 3 does not fall over 4 steps — ", 3),
            ("eps", self._carried(self._eps_patcher("eps"), flow_cosmos_rho=3.0), 6, "KarrasDynamicError",
             "Extra Schedulers: Flow Cosmos Dynamic with rho 3 does not fall over 6 steps — ", 5),
            ("flow", self._carried(self._patcher(3.0), flow_cosmos_sigma_max=1.0, flow_cosmos_sigma_min=1.0), 28,
             "ExtraSchedulerError", "Extra Schedulers: Flow Cosmos Dynamic needs 0 < sigma~ min < sigma~ max, both "
             f"finite, got sigma~ min 1 and sigma~ max 1 ({xmath.FLOW_COSMOS_VALUES_SOURCE})", None),
        )
        for label, model, steps, kind, text, fewer in cases:
            with self.subTest(model=label, text=text):
                sampling = model.get_model_object("model_sampling")
                settings = self.pack.model_settings(model)
                error = self._error_of(lambda: self.samplers.calculate_sigmas(sampling, "flow_cosmos_dynamic", steps))
                self.assertEqual(error, self._error_of(lambda: self._expected("flow_cosmos_dynamic", steps, sampling,
                                                                              settings)))
                self.assertEqual(error[0], kind)
                self.assertTrue(error[1].startswith(text), error[1])
                with self.assertRaises(ValueError):
                    self.samplers.KSampler(model, steps, "cpu", "euler", "flow_cosmos_dynamic")
                if fewer is not None:   # a rising list is a matter of the step count: one fewer still falls
                    self.assertTrue(torch.equal(
                        self.samplers.calculate_sigmas(sampling, "flow_cosmos_dynamic", fewer),
                        self._expected("flow_cosmos_dynamic", fewer, sampling, settings)))
        eps_model = self._carried(self._eps_patcher("eps"), flow_cosmos_sigma_max=1.0, flow_cosmos_sigma_min=1.0)
        sampling = eps_model.get_model_object("model_sampling")
        self.assertTrue(torch.equal(self.samplers.calculate_sigmas(sampling, "flow_cosmos_dynamic", 28),
                                    self.samplers.calculate_sigmas(sampling, "karras_dynamic", 28)))   # σ̃ unused
        flow = self._patcher(3.0).get_model_object("model_sampling")
        eps = self._eps_patcher("eps").get_model_object("model_sampling")
        knob = self._carried(self._eps_patcher("v"), flow_cosmos_rho=5.5)
        with mock.patch.object(self.pack, "_flow_cosmos_dynamic_not_flow_logged", False), \
                mock.patch.object(self.pack, "_flow_cosmos_not_flow_logged", False):
            with self.assertNoLogs("ai_studio_forge_parity", level="INFO"):
                self.samplers.calculate_sigmas(flow, "flow_cosmos_dynamic", 28)
                refused = self._carried(self._eps_patcher("eps"), flow_cosmos_rho=1.0)
                with self.assertRaises(ValueError):   # ρ 1 is refused before the line
                    self.samplers.KSampler(refused, 20, "cpu", "euler", "flow_cosmos_dynamic")
            with self.assertLogs("ai_studio_forge_parity", level="INFO") as logs:
                self.samplers.KSampler(knob, 20, "cpu", "euler", "flow_cosmos_dynamic")
                self.samplers.calculate_sigmas(eps, "flow_cosmos_dynamic", 28)
                self.samplers.calculate_sigmas(eps, "flow_cosmos_rho7", 12)   # Flow Cosmos rho7 has its own line
                self.samplers.KSampler(self._eps_patcher("v"), 10, "cpu", "euler", "flow_cosmos_dynamic")
        self.assertEqual([record.getMessage() for record in logs.records],
                         ["[Extra Schedulers] Flow Cosmos Dynamic: not a flow model - using Karras Dynamic rho 5.5 on "
                          "the model's own sigma range",
                          "[Extra Schedulers] Flow Cosmos rho7: not a flow model - using Karras rho 7 on the model's "
                          "own sigma range"])

    def test_ksampler_denoise_slicing_and_discard_penultimate(self):
        carried = self._carried(laplace_mu=0.5, laplace_beta=1.0)
        sampling = carried.get_model_object("model_sampling")
        settings = self.pack.model_settings(carried)
        sliced = self.samplers.KSampler(carried, 28, "cpu", "euler", "laplace", denoise=0.35).sigmas
        full = self._expected("laplace", int(28 / 0.35), sampling, settings)
        self.assertEqual(_bits(sliced), _bits(full[-29:]))
        discard = self.samplers.KSampler(carried, 10, "cpu", "dpm_2", "custom").sigmas
        longer = self._expected("custom", 11, sampling, settings)
        self.assertEqual(_bits(discard), _bits(torch.cat([longer[:-2], longer[-1:]])))

    def test_impact_style_and_basic_scheduler_calls(self):
        carried = self._carried(custom_mode="sigmas", custom_sigmas="[14.6, 7, 3.1, 1.2, 0.03]")
        sampling = carried.get_model_object("model_sampling")
        settings = self.pack.model_settings(carried)
        # Impact Pack (impact_sampling.py): calculate_sigmas(model.get_model_object("model_sampling"), ...)
        impact = self.samplers.calculate_sigmas(carried.get_model_object("model_sampling"), "custom", 9)
        self.assertEqual(_bits(impact), _bits(self._expected("custom", 9, sampling, settings)))
        from comfy_custom_nodes.ai_studio_forge_parity.compat import node_result

        sigmas = node_result(self.basic_scheduler.execute(carried, "laplace", 20, 1.0))[0]
        self.assertEqual(_bits(sigmas), _bits(self._expected("laplace", 20, sampling, settings)))

    def test_values_reach_every_clone_and_the_pack_flow_shift(self):
        carried = self._carried(laplace_mu=1.0, laplace_beta=2.0, flow_cosmos_rho=5.0, flow_cosmos_sigma_max=300.0,
                                flow_cosmos_sigma_min=0.01)
        settings = self.pack.model_settings(carried)
        self.assertEqual((settings.laplace_mu, settings.laplace_beta), (1.0, 2.0))
        self.assertEqual(tuple(settings)[5:], (5.0, 300.0, 0.01))
        clone = carried.clone()
        self.assertIs(clone.get_model_object("model_sampling"), carried.get_model_object("model_sampling"))

        def provider(name, *, method, feature, args=(), kwargs=None):
            self.assertEqual(name, "ModelSamplingSD3")
            return self.sd3().patch(*args)

        with mock.patch.object(self.generation, "invoke_provider", side_effect=provider):
            shifted = self.generation._patch_flow_shift(clone, 5.0)
        sampling = shifted.get_model_object("model_sampling")
        self.assertEqual(sampling.shift, 5.0)
        self.assertEqual(self.pack.model_settings(shifted), settings)
        got = self.samplers.calculate_sigmas(sampling, "laplace", 16)
        self.assertEqual(_bits(got), _bits(self._expected("laplace", 16, sampling, settings)))
        default = self.samplers.calculate_sigmas(self._patcher(5.0).get_model_object("model_sampling"), "laplace", 16)
        self.assertNotEqual(_bits(got), _bits(default))   # the carried mu/beta are really used
        cosmos = self.samplers.calculate_sigmas(sampling, "flow_cosmos_rho7", 16)
        self.assertEqual(_bits(cosmos), _bits(self._expected("flow_cosmos_rho7", 16, sampling, settings)))
        self.assertNotEqual(_bits(cosmos), _bits(self.samplers.calculate_sigmas(
            self._patcher(5.0).get_model_object("model_sampling"), "flow_cosmos_rho7", 16)))   # ... and ρ / σ̃ too

    def test_the_carrier_copy_is_private(self):
        base = self._patcher()
        original = base.get_model_object("model_sampling")
        carried = self._carried(base, laplace_mu=0.3)
        duplicate = carried.get_model_object("model_sampling")
        self.assertIsInstance(duplicate, self.model_sampling.CONST)
        self.assertIs(type(duplicate), type(original))
        self.assertNotIn(self.pack.CARRIER_ATTR, vars(original))
        duplicate.to(torch.float64)
        self.assertEqual(duplicate.sigmas.dtype, torch.float64)
        self.assertEqual(original.sigmas.dtype, torch.float32)
        self.assertEqual(_bits(self.samplers.calculate_sigmas(original, "cosine", 8)),
                         _bits(self._expected("cosine", 8, original, xmath.DEFAULTS)))

    def test_host_functions_are_the_ones_the_extension_mirrors(self):
        from comfy_extras.nodes_align_your_steps import loglinear_interp

        import comfy.k_diffusion.sampling as k_sampling

        def body(path: Path, name: str) -> str:
            node = next(item for item in ast.parse(path.read_text(encoding="utf-8")).body
                        if isinstance(item, ast.FunctionDef) and item.name == name)
            statements = node.body[1:] if isinstance(node.body[0], ast.Expr) else node.body
            return ast.dump(ast.Module(body=statements, type_ignores=[])) + ast.dump(node.args)

        self.assertEqual(body(Path(k_sampling.__file__), "get_sigmas_laplace"),
                         body(EXT_ROOT / "sam3ext/extra_schedulers/schedulers.py", "get_sigmas_laplace"))
        self.assertEqual(body(Path(sys.modules[loglinear_interp.__module__].__file__), "loglinear_interp"),
                         body(FORGE_ROOT / "modules" / "sd_schedulers.py", "_loglinear_interp"))
        self.assertIs(self.pack.host_get_sigmas_laplace(), k_sampling.get_sigmas_laplace)
        # Flow Cosmos rho7 on eps/v: ComfyUI's Karras (its "karras" scheduler) is Forge's, which sam-extra calls
        forge_sampling = FORGE_ROOT / "modules_forge" / "packages" / "k_diffusion" / "sampling.py"
        for name in ("get_sigmas_karras", "append_zero"):
            self.assertEqual(body(Path(k_sampling.__file__), name), body(forge_sampling, name), name)
        self.assertIs(self.pack.host_get_sigmas_karras(), k_sampling.get_sigmas_karras)
        self.assertIs(self.samplers.SCHEDULER_HANDLERS["karras"].handler, k_sampling.get_sigmas_karras)


if __name__ == "__main__":
    unittest.main()
