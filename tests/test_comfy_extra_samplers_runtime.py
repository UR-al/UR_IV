r"""Pack 1.7.0 Extra Samplers on the real ComfyUI runtime (CPU) = sam-extra v0.33.0 on Forge's k-diffusion.

The pack registers its twelve samplers into ComfyUI's own ``comfy.k_diffusion.sampling`` / ``comfy.samplers``
tables (``extra_samplers_register.install_extra_samplers``) and runs them on ComfyUI's own helpers and solvers
(``sample_er_sde``, ``sample_dpmpp_2m_sde``/``3m_sde``, ``sigma_to_half_log_snr``, …). Here the registered
``sample_*`` entries run on **ComfyUI's** module with ComfyUI's own model-sampling classes (shift-3
``ModelSamplingDiscreteFlow`` + ``CONST``, ``ModelSamplingDiscrete`` + ``EPS``), and the installed extension's
functions on **Forge's** module (``tests/_forge_k_diffusion``) with a predictor that reads the same model sampling —
same toy model, sigmas and deterministic noise — and the latents must be equal bit for bit:

* registration in the real tables (each list once, the discard set, ``ksampler(name)`` → the pack entry, the six
  core names left to ComfyUI, a foreign function never replaced);
* ER SDE (Reverse-time) / (ODE) / (Tunable) incl. the noise window; ``er_sde_reverse_time`` == ComfyUI's own
  ``SamplerER_SDE("Reverse-time SDE", …)`` sampler and ``er_sde_tunable`` (window off) == ``SamplerER_SDE("ER-SDE",
  …)``; a full window == no window, an empty one == ``er_sde_ode`` (no noise drawn);
* DPM++ 4M SDE; Euler (SMEA) Dy and their CFG++ entries (the CFG++ capture through ``comfy.model_patcher``);
  ``euler_dy(substeps=False)`` == ComfyUI's ``sample_euler``, ``euler_dy_cfg_pp(substeps=False)`` ==
  ``sample_euler_cfg_pp``; ``DY_SUBSTEPS_OFF_KEY`` turns the registered entry's sub-steps off;
* the masked sub-step path with ComfyUI's ``KSamplerX0Inpaint`` (``latent_image``/``noise`` follow each sub-step and
  come back; an all-ones mask == no mask);
* the DPM++ flow ODE trio (== sam-extra and == ComfyUI's ``dpmpp_2m_sde``/``dpmpp_3m_sde`` at η 0, the noise stub
  never called); Restart (flow) (fewer than 20 steps == ComfyUI's ``sample_heun``);
* the free core names (X11): the extension's ``cfgpp_ud10_ab``, ``ipndm``, ``deis`` (Forge's k-diffusion and
  ``deis.py``) == ComfyUI's core functions bit for bit; ``dpmpp_2m_sde_heun`` too with a shared noise sampler;
* IPNDM_V: the pack's ``ipndm_v_fixed`` (sam-extra 0.33.1's fourth-order weight fix; ComfyUI core's ``ipndm_v`` keeps the
  upstream typo and stays ComfyUI's) == sam-extra 0.33.1's IPNDM_V on ComfyUI's schedules incl. Anima's Linear Quadratic
  (``tests/_ipndm_v_oracle`` — before the registry's sync the installed 0.33.0 file with its one line fixed), == core
  ``ipndm_v`` bit for bit where the step ratio is constant and the steps are exact in binary (within rounding on
  ComfyUI's float ``exponential`` list), and the constant-velocity probe through ComfyUI's tables (``ipndm_v_fixed``
  exact; core ``ipndm_v`` about −174 at step 15, 0-based).

Skipped unless ``AISTUDIO_COMFY_TEST_ROOT`` is set (and without the extension; ``AISTUDIO_REQUIRE_FORGE_EXT=1`` makes
that a failure). Run it with the portable python, the repo (or worktree) first on ``sys.path``,
``CUDA_VISIBLE_DEVICES=-1`` (ComfyUI is forced to ``--cpu`` before its model management loads; no server, no GPU)
and ``PYTHONDONTWRITEBYTECODE=1`` (nothing is written into the ComfyUI install). Until sam-extra v0.33.0 is
installed, point the oracle at its work tree (``AISTUDIO_FORGE_EXTENSION_DIR``, ``SAM3_FORGE_ROOT``)::

    $env:CUDA_VISIBLE_DEVICES='-1'; $env:PYTHONDONTWRITEBYTECODE='1'
    $env:AISTUDIO_COMFY_TEST_ROOT='C:\ComfyUI_windows_portable\ComfyUI'; $env:AISTUDIO_REQUIRE_FORGE_EXT='1'
    C:\ComfyUI_windows_portable\python_embeded\python.exe -X utf8 -c "import sys, unittest; sys.path.insert(0, r'<repo>'); unittest.main(module=None, argv=['x', 'tests.test_comfy_extra_samplers_runtime', '-v'])"
"""
from __future__ import annotations

import importlib.util
import itertools
import os
import sys
import types
import unittest
from unittest import mock

_COMFY_ROOT = os.environ.get("AISTUDIO_COMFY_TEST_ROOT")
if _COMFY_ROOT:  # force --cpu before anything loads ComfyUI's model management
    if _COMFY_ROOT not in sys.path:
        sys.path.append(_COMFY_ROOT)
    from comfy.cli_args import args as _comfy_args  # noqa: E402

    _comfy_args.cpu = True

from tests._forge_k_diffusion import (  # noqa: E402
    FORGE_ROOT,
    K_DIR,
    ComfyPatcher,
    ReplayNoise,
    ToyModel,
    forge_k_sampling,
    import_ext,
    installed_k_sampling,
    stub_modules,
)
from tests._optional_deps import bind_torch, requires_torch  # noqa: E402
from tests._sam_extra_ext import requires_extension  # noqa: E402

torch = None  # filled by bind_torch()

_SHAPES = ((1, 4, 6, 7), (1, 3, 2, 5, 6))


def _bits(tensor):
    return tensor.contiguous().view({torch.float32: torch.int32, torch.float64: torch.int64}[tensor.dtype])


def _impact_sampling():
    """Impact Pack's ``impact_sampling`` module from the ComfyUI install (read only, no bytecode, not registered)."""
    path = os.path.join(_COMFY_ROOT, "custom_nodes", "ComfyUI-Impact-Pack", "modules", "impact", "impact_sampling.py")
    if not os.path.isfile(path):
        raise unittest.SkipTest("ComfyUI-Impact-Pack is not installed next to this ComfyUI")
    spec = importlib.util.spec_from_file_location("_aistudio_impact_sampling", path)
    module = importlib.util.module_from_spec(spec)
    previous, sys.dont_write_bytecode = sys.dont_write_bytecode, True
    try:
        spec.loader.exec_module(module)
    finally:
        sys.dont_write_bytecode = previous
    return module


class _ForgePredictor:
    """Forge's predictor surface over a ComfyUI model-sampling object: the same ``percent_to_sigma``."""

    def __init__(self, sampling, flow: bool):
        self.prediction_type = "const" if flow else "eps"
        self.sampling = sampling

    def percent_to_sigma(self, percent):
        return self.sampling.percent_to_sigma(percent)


class _CfgModel(ToyModel):
    """A CFG denoiser: cond/uncond x0, CFG, then ComfyUI's post-CFG list with ComfyUI's argument dict
    (``comfy/samplers.py`` ``cfg_function``). Records each call's σ, shape and ``sam_extra_substep``."""

    def __init__(self, sampling, comfy_layout: bool = False, cfg: float = 2.0):
        super().__init__(sampling, comfy_layout=comfy_layout)
        self.cfg = cfg
        self.markers = []

    def __call__(self, x, sigma, **extra_args):
        options = extra_args.get("model_options") or {}
        self.markers.append(((options.get("transformer_options") or {}).get("sam_extra_substep"), tuple(x.shape)))
        self.calls.append((float(sigma.reshape(-1)[0]), tuple(x.shape), None))
        s = sigma.reshape([-1] + [1] * (x.ndim - 1)).to(x.dtype)
        cond = torch.tanh(x / (1.0 + s * s) ** 0.5) * 0.8 + 0.05 * torch.tanh(s)
        uncond = torch.tanh(0.7 * x / (1.0 + s * s) ** 0.5) * 0.6 - 0.02 * torch.tanh(s)
        denoised = uncond + (cond - uncond) * self.cfg
        for function in options.get("sampler_post_cfg_function", []):
            denoised = function({"denoised": denoised, "cond": [{}], "uncond": [{}], "cond_scale": self.cfg,
                                 "model": self, "uncond_denoised": uncond, "cond_denoised": cond, "sigma": sigma,
                                 "model_options": options, "input": x})
        return denoised


class _StableModel(ToyModel):
    """A bounded x0 at every σ (the ε runs start at σ ≈ 14.6)."""

    def __call__(self, x, sigma, **extra_args):
        self.calls.append((float(sigma.reshape(-1)[0]), tuple(x.shape), None))
        s = sigma.reshape([-1] + [1] * (x.ndim - 1)).to(x.dtype)
        return torch.tanh(x / (1.0 + s * s) ** 0.5) * 0.8 + self.bias * torch.tanh(s)


@requires_torch
@requires_extension
@unittest.skipUnless(_COMFY_ROOT, "Set AISTUDIO_COMFY_TEST_ROOT to run the Extra Samplers on the real ComfyUI runtime (CPU)")
class ExtraSamplersOnRealComfyTests(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        bind_torch(globals())
        import comfy.model_management as management

        if management.get_torch_device().type != "cpu":
            raise unittest.SkipTest("ComfyUI did not start on CPU - refusing to touch the GPU")
        import comfy.k_diffusion.sampling as ckd
        import comfy.model_base
        import comfy.model_sampling as ms
        import comfy.samplers
        from comfy_extras import nodes_custom_sampler

        from comfy_custom_nodes.ai_studio_forge_parity import extra_samplers_dy as dy
        from comfy_custom_nodes.ai_studio_forge_parity import extra_samplers_math as xs
        from comfy_custom_nodes.ai_studio_forge_parity import extra_samplers_register as reg
        from comfy_custom_nodes.ai_studio_forge_parity import guidance_common

        cls.ckd, cls.samplers, cls.model_base, cls.nodes = ckd, comfy.samplers, comfy.model_base, nodes_custom_sampler
        cls.dy, cls.xs, cls.reg, cls.common = dy, xs, reg, guidance_common

        class FlowSampling(ms.ModelSamplingDiscreteFlow, ms.CONST):
            pass

        class EpsSampling(ms.ModelSamplingDiscrete, ms.EPS):
            pass

        cls.sampling = {
            True: FlowSampling(types.SimpleNamespace(sampling_settings={"shift": 3.0, "multiplier": 1.0})),
            False: EpsSampling(types.SimpleNamespace(sampling_settings={})),
        }
        cls.registration = reg.install_extra_samplers(required=True)

        cls.fkd = forge_k_sampling()
        spec = importlib.util.spec_from_file_location("_aistudio_forge_k_diffusion_deis", FORGE_ROOT / K_DIR / "deis.py")
        cls.forge_deis = importlib.util.module_from_spec(spec)
        previous, sys.dont_write_bytecode = sys.dont_write_bytecode, True
        try:
            spec.loader.exec_module(cls.forge_deis)
        finally:
            sys.dont_write_bytecode = previous
        with cls._forge():
            cls.ext = {name: import_ext(f"sam3ext.extra_samplers.{name}") for name in (
                "er_sde", "dpmpp_4m_sde", "euler_dy", "dpmpp_flow", "restart_flow", "cfgpp_ud10_ab", "ipndm_deis")}
        from tests import _ipndm_v_oracle

        cls.ipndm_helpers = _ipndm_v_oracle
        function, cls.ipndm_v_how = _ipndm_v_oracle.ipndm_v_oracle(cls.ext["ipndm_deis"])
        cls.ipndm_v_oracle = staticmethod(function)

    @classmethod
    def _forge(cls):
        """sam-extra finds Forge's ``k_diffusion.sampling`` and ``k_diffusion.deis`` at call time."""
        stack = installed_k_sampling(cls.fkd)

        class _Both:
            def __enter__(self):
                stack.__enter__()
                self.deis = stub_modules({"k_diffusion.deis": cls.forge_deis})
                self.deis.__enter__()
                sys.modules["k_diffusion"].deis = cls.forge_deis

            def __exit__(self, *exc):
                self.deis.__exit__(*exc)
                return stack.__exit__(*exc)

        return _Both()

    # -- inputs ------------------------------------------------------------------------------------------

    def _sigmas(self, flow: bool, steps: int, scheduler: str = "simple"):
        return self.samplers.calculate_sigmas(self.sampling[flow], scheduler, steps).cpu()

    def _models(self, flow: bool, kind=_StableModel, **kwargs):
        """(ComfyUI-layout model over the ComfyUI model sampling, Forge-layout model over the same)."""
        sampling = self.sampling[flow]
        return kind(sampling, comfy_layout=True, **kwargs), kind(_ForgePredictor(sampling, flow), **kwargs)

    @staticmethod
    def _latent(shape, sigmas, seed=5):
        generator = torch.Generator().manual_seed(seed)
        return torch.randn(shape, generator=generator, dtype=torch.float64).to(torch.float32) * float(sigmas[0])

    def _entry(self, name):
        return getattr(self.ckd, "sample_" + name)

    def _pair(self, ours, theirs, *, flow, steps=10, shape=_SHAPES[0], kind=_StableModel, ours_kwargs=None,
              theirs_kwargs=None, noise=True, scheduler="simple", model_kwargs=None):
        """Run ``ours`` on ComfyUI's module and ``theirs`` (sam-extra or a ComfyUI function) and return both
        ``(latent, draws, model calls)``; ``theirs=None`` runs nothing on the Forge side."""
        sigmas = self._sigmas(flow, steps, scheduler)
        comfy_model, forge_model = self._models(flow, kind, **(model_kwargs or {}))
        results = []
        for function, model, kwargs, forge in ((ours, comfy_model, ours_kwargs, False),
                                               (theirs[0], theirs[1] == "forge" and forge_model or comfy_model,
                                                theirs_kwargs, theirs[1] == "forge")):
            x = self._latent(shape, sigmas)
            sampler = ReplayNoise(x, 4242) if noise else None
            call = dict(kwargs or {})
            if noise:
                call["noise_sampler"] = sampler
            if forge:
                with self._forge():
                    out = function(model, x, sigmas, extra_args={}, disable=True, **call)
            else:
                out = function(model, x, sigmas, extra_args={"seed": 3}, disable=True, **call)
            results.append((out, sampler.draws if sampler else None, [c[:2] for c in model.calls]))
            model.calls.clear()
        return results

    def _same(self, results, label, draws=True):
        (a, a_draws, a_calls), (b, b_draws, b_calls) = results
        self.assertEqual((a.dtype, a.shape), (b.dtype, b.shape), label)
        self.assertTrue(torch.equal(_bits(a), _bits(b)), f"{label}: latents differ")
        self.assertTrue(torch.isfinite(a).all(), label)
        if draws:
            self.assertEqual(a_draws, b_draws, f"{label}: noise draws")
        self.assertEqual(a_calls, b_calls, f"{label}: model calls")

    # -- registration --------------------------------------------------------------------------------------

    def test_registration_in_the_real_tables(self):
        reg, samplers = self.reg, self.samplers
        self.assertEqual(self.registration.owned, reg.PACK_SAMPLERS)
        self.assertEqual(reg.owned_names(), reg.PACK_SAMPLERS)
        self.assertEqual(reg.unavailable_names(), ())
        lists = {id(lst): lst for lst in (samplers.KSAMPLER_NAMES, samplers.SAMPLER_NAMES, samplers.KSampler.SAMPLERS)}
        for name in reg.PACK_SAMPLERS:
            with self.subTest(name=name):
                for names in lists.values():
                    self.assertEqual(names.count(name), 1)
                entry = self._entry(name)
                self.assertEqual(entry.__name__, "sample_" + name)
                self.assertIs(samplers.ksampler(name).sampler_function, entry)
                self.assertIs(samplers.sampler_object(name).sampler_function, entry)
        self.assertLessEqual({"dpmpp_4m_sde", "dpmpp_3m_flow_ode"}, samplers.KSampler.DISCARD_PENULTIMATE_SIGMA_SAMPLERS)
        self.assertNotIn("dpmpp_2m_flow_ode", samplers.KSampler.DISCARD_PENULTIMATE_SIGMA_SAMPLERS)
        for name in reg.CORE_SAMPLERS:   # ComfyUI's own (v0.35.0+ for cfgpp_ud10_ab): offered, never the pack's
            with self.subTest(core=name):
                self.assertIn(name, samplers.KSampler.SAMPLERS)
                samplers.sampler_object(name)          # resolves (uni_pc_bh2 is ComfyUI's own UniPC module)
                entry = getattr(self.ckd, "sample_" + name, None)
                if name != "uni_pc_bh2":
                    self.assertIsNotNone(entry)
                self.assertNotEqual(self.reg.entry_owner(entry), reg.OWNER)
        # ComfyUI core's ipndm_v (the upstream typo) is left alone and stays offered; the pack's fixed one is its own
        self.assertIn("ipndm_v", samplers.KSampler.SAMPLERS)
        self.assertNotEqual(self.reg.entry_owner(self.ckd.sample_ipndm_v), reg.OWNER)
        self.assertNotIn("ipndm_v", reg.PACK_SAMPLERS + reg.CORE_SAMPLERS)
        self.assertEqual(self.reg.entry_owner(self._entry("ipndm_v_fixed")), reg.OWNER)
        again = reg.install_extra_samplers(required=True)
        self.assertEqual((again.owned, again.inserted), (reg.PACK_SAMPLERS, ()))

    def test_impact_face_detailer_sigmas_drop_the_penultimate_sigma(self):
        # ADetailer / SAM3 restore face run through Impact's FaceDetailer, whose impact_sampling.calculate_sigmas
        # has its own literal discard list. With the pack's scheduler_func_opt, Impact's real separated_sample
        # samples dpmpp_4m_sde / dpmpp_3m_flow_ode on KSampler's sigmas (Forge's discard_next_to_last_sigma).
        impact = _impact_sampling()
        sampling = self.sampling[True]
        model = types.SimpleNamespace(get_model_object=lambda name: {"model_sampling": sampling}[name])
        latent = {"samples": torch.zeros(1, 4, 8, 8)}

        def run(name, scheduler, steps, denoise, scheduler_func):
            captured = []

            def sample(model, add_noise, seed, cfg, positive, negative, sampler, sigmas, latent_image, noise=None,
                       callback=None):
                captured.append((sampler, sigmas.clone()))
                return latent_image, latent_image

            with mock.patch.object(impact, "sample_with_custom_noise", side_effect=sample):
                impact.ksampler_wrapper(model, 1, steps, 5.0, name, scheduler, "positive", "negative", latent,
                                        denoise, scheduler_func=scheduler_func)
            (entry,) = captured
            return entry

        for name in self.reg.DISCARD_PENULTIMATE_SIGMA:
            for scheduler, steps, denoise in (("simple", 8, 1.0), ("exponential", 10, 0.4)):
                with self.subTest(sampler=name, scheduler=scheduler, denoise=denoise):
                    expected = self.samplers.KSampler(model, steps, "cpu", name, scheduler, denoise=denoise).sigmas
                    sampler, sigmas = run(name, scheduler, steps, denoise,
                                          self.reg.impact_scheduler_func(name, scheduler))
                    self.assertTrue(torch.equal(_bits(sigmas), _bits(expected)))
                    self.assertIs(sampler.sampler_function, self._entry(name))      # the pack's sampler runs
                    _sampler, kept = run(name, scheduler, steps, denoise, None)     # Impact's own list keeps it
                    self.assertEqual(kept.shape, expected.shape)
                    self.assertFalse(torch.equal(kept, expected))
        # Impact's own discard samplers stay with Impact (the pack hands it nothing for them)
        self.assertIsNone(self.reg.impact_scheduler_func("uni_pc_bh2", "simple"))

    def test_a_foreign_function_is_never_replaced(self):
        def foreign(model, x, sigmas, extra_args=None, callback=None, disable=None):
            return x

        with mock.patch.object(self.ckd, "sample_euler_dy", foreign):
            report = self.reg.install_extra_samplers(required=True)
            self.assertIs(self.ckd.sample_euler_dy, foreign)
            self.assertIn("euler_dy", report.foreign)
            self.assertNotIn("euler_dy", self.reg.owned_names())
        self.assertIn("euler_dy", self.reg.owned_names())

    # -- ER SDE --------------------------------------------------------------------------------------------

    def test_er_sde_trio_equals_the_extension(self):
        cases = 0
        for name, flow, shape, (stage, eta), window in itertools.product(
                ("er_sde_reverse_time", "er_sde_tunable", "er_sde_ode"), (True, False), _SHAPES,
                ((3, 1.0), (2, 0.37), (1, 2.5)), (None, (0.0, 1.0), (0.5, 0.5), (0.2, 0.8))):
            if name == "er_sde_ode" and window is not None:
                continue
            label = f"{name} flow={flow} {shape} stage={stage} eta={eta} window={window}"
            with self.subTest(label):
                kwargs = {"max_stage": stage}
                if name != "er_sde_ode":
                    kwargs.update(er_sde_eta=eta, er_sde_window=window)
                self._same(self._pair(self._entry(name), (getattr(self.ext["er_sde"], "sample_" + name), "forge"),
                                      flow=flow, shape=shape, ours_kwargs=kwargs, theirs_kwargs=kwargs), label)
                cases += 1
        self.assertEqual(cases, 2 * 2 * 3 * 4 * 2 + 2 * 2 * 3)

    def _node_sampler(self, solver, stage, eta, s_noise=1.0):
        output = self.nodes.SamplerER_SDE.execute(solver, stage, eta, s_noise)
        sampler = output.result[0] if hasattr(output, "result") else output[0]
        self.assertIs(sampler.sampler_function, self.ckd.sample_er_sde)
        return sampler

    def test_er_sde_entries_are_comfys_own_node(self):
        for (name, solver), flow, (stage, eta) in itertools.product(
                (("er_sde_reverse_time", "Reverse-time SDE"), ("er_sde_tunable", "ER-SDE")), (True, False),
                ((3, 1.0), (2, 0.5), (3, 0.0))):
            label = f"{name} flow={flow} stage={stage} eta={eta}"
            with self.subTest(label):
                node = self._node_sampler(solver, stage, eta)
                self._same(self._pair(self._entry(name), (node.sampler_function, "comfy"), flow=flow,
                                      ours_kwargs={"max_stage": stage, "er_sde_eta": eta, "er_sde_window": None},
                                      theirs_kwargs=dict(node.extra_options)), label)

    def test_tunable_window_full_is_off_and_empty_is_the_ode(self):
        tunable, ode = self._entry("er_sde_tunable"), self._entry("er_sde_ode")
        for flow in (True, False):
            with self.subTest(flow=flow):
                full = self._pair(tunable, (tunable, "comfy"), flow=flow,
                                  ours_kwargs={"max_stage": 3, "er_sde_eta": 1.0, "er_sde_window": (0.0, 1.0)},
                                  theirs_kwargs={"max_stage": 3, "er_sde_eta": 1.0, "er_sde_window": None})
                self._same(full, "full window")
                self.assertGreater(len(full[0][1]), 0)
                empty = self._pair(tunable, (ode, "comfy"), flow=flow,
                                   ours_kwargs={"max_stage": 3, "er_sde_eta": 1.0, "er_sde_window": (0.6, 0.6)},
                                   theirs_kwargs={"max_stage": 3})
                self._same(empty, "empty window", draws=False)
                self.assertEqual(empty[0][1], [])

    def test_values_from_the_settings_node_reach_the_sampler(self):
        """``ForgeNeoExtraSamplerSettings`` → ``model_options`` → the registered entry == explicit values."""
        from comfy_custom_nodes.ai_studio_forge_parity import extra_samplers_nodes

        class Patcher:   # the ModelPatcher surface the node uses
            def __init__(self, options=None):
                self.model_options = dict(options or {})

            def clone(self):
                return Patcher(self.model_options)

        (patched,) = extra_samplers_nodes.ForgeNeoExtraSamplerSettings().patch(Patcher(), 2, 0.5, True, 0.2, 0.8)
        options = patched.model_options
        self.assertIn(self.xs.MODEL_OPTIONS_KEY, options)
        sigmas = self._sigmas(True, 10)
        comfy_model, _ = self._models(True)
        runs = []
        for extra, kwargs in (({"model_options": options, "seed": 3}, {}),
                              ({"seed": 3}, {"max_stage": 2, "er_sde_eta": 0.5, "er_sde_window": (0.2, 0.8)})):
            x = self._latent(_SHAPES[0], sigmas)
            runs.append(self._entry("er_sde_reverse_time")(comfy_model, x, sigmas, extra_args=extra, disable=True,
                                                           noise_sampler=ReplayNoise(x, 9), **kwargs))
        self.assertTrue(torch.equal(_bits(runs[0]), _bits(runs[1])))

    # -- DPM++ 4M SDE, flow ODE trio, Restart ------------------------------------------------------------

    def test_dpmpp_4m_sde_equals_the_extension(self):
        for flow, eta, shape in itertools.product((True, False), (0.0, 1.0), _SHAPES):
            with self.subTest(flow=flow, eta=eta, shape=shape):
                kwargs = {"eta": eta, "s_noise": 1.0}
                self._same(self._pair(self._entry("dpmpp_4m_sde"),
                                      (self.ext["dpmpp_4m_sde"].sample_dpmpp_4m_sde, "forge"), flow=flow, shape=shape,
                                      ours_kwargs=kwargs, theirs_kwargs=kwargs, steps=12), f"flow={flow} eta={eta}")

    def test_flow_ode_trio_equals_the_extension_and_comfys_sde_at_eta_zero(self):
        hosts = {"dpmpp_2m_flow_ode": ("sample_dpmpp_2m_sde", "midpoint"),
                 "dpmpp_2m_heun_flow_ode": ("sample_dpmpp_2m_sde", "heun"),
                 "dpmpp_3m_flow_ode": ("sample_dpmpp_3m_sde", None)}
        for (name, (host, solver)), flow, shape in itertools.product(hosts.items(), (True, False), _SHAPES):
            with self.subTest(name=name, flow=flow, shape=shape):
                ours = self._entry(name)
                self._same(self._pair(ours, (getattr(self.ext["dpmpp_flow"], "sample_" + name), "forge"),
                                      flow=flow, shape=shape, noise=False, steps=12), "vs sam-extra", draws=False)
                host_kwargs = {"eta": 0.0, "s_noise": 1.0}
                if solver:
                    host_kwargs["solver_type"] = solver
                results = self._pair(lambda *a, noise_sampler=None, **k: ours(*a, **k),
                                     (getattr(self.ckd, host), "comfy"), flow=flow, shape=shape, steps=12,
                                     theirs_kwargs=host_kwargs)
                self._same(results, "vs ComfyUI at eta 0", draws=False)
                self.assertEqual(results[1][1], [], "ComfyUI's sampler drew no noise at eta 0")

    def test_restart_flow_equals_the_extension(self):
        for flow, steps, shape in itertools.product((True, False), (19, 28, 36), _SHAPES):
            with self.subTest(flow=flow, steps=steps, shape=shape):
                results = self._pair(self._entry("restart_flow"),
                                     (self.ext["restart_flow"].sample_restart_flow, "forge"), flow=flow, steps=steps,
                                     shape=shape, ours_kwargs={"s_noise": 0.7}, theirs_kwargs={"s_noise": 0.7})
                self._same(results, f"flow={flow} steps={steps}")
                plan = self.ext["restart_flow"].restart_plan(self._sigmas(flow, steps), flow)
                self.assertEqual(len(results[0][1]), 0 if plan is None else plan.times)

    def test_restart_flow_under_twenty_steps_is_comfys_heun(self):
        for flow in (True, False):
            with self.subTest(flow=flow):
                ours = self._entry("restart_flow")
                self._same(self._pair(lambda *a, noise_sampler=None, **k: ours(*a, **k),
                                      (self.ckd.sample_heun, "comfy"), flow=flow, steps=19, noise=False),
                           "heun", draws=False)

    # -- Euler (SMEA) Dy -----------------------------------------------------------------------------------

    def test_dy_entries_equal_the_extension(self):
        for name, flow, shape in itertools.product(self.reg.DY_SAMPLERS, (True, False), _SHAPES):
            with self.subTest(name=name, flow=flow, shape=shape):
                kwargs = {"s_churn": 0.0}
                results = self._pair(self._entry(name), (getattr(self.ext["euler_dy"], "sample_" + name), "forge"),
                                     flow=flow, shape=shape, kind=_CfgModel, ours_kwargs=kwargs, theirs_kwargs=kwargs)
                self._same(results, name)
        with self.subTest("churn"):
            kwargs = {"s_churn": 1.5}
            self._same(self._pair(self._entry("euler_smea_dy"), (self.ext["euler_dy"].sample_euler_smea_dy, "forge"),
                                  flow=False, kind=_CfgModel, ours_kwargs=kwargs, theirs_kwargs=kwargs), "churn")

    def test_dy_without_substeps_is_comfys_euler(self):
        for (name, host), flow in itertools.product((("euler_dy", "sample_euler"), ("euler_smea_dy", "sample_euler"),
                                                     ("euler_dy_cfg_pp", "sample_euler_cfg_pp"),
                                                     ("euler_smea_dy_cfg_pp", "sample_euler_cfg_pp")), (True, False)):
            with self.subTest(name=name, flow=flow):
                ours = getattr(self.dy, "sample_" + name)
                self._same(self._pair(lambda *a, noise_sampler=None, **k: ours(*a, substeps=False, **k),
                                      (getattr(self.ckd, host), "comfy"), flow=flow, kind=_CfgModel, noise=False),
                           name, draws=False)

    def test_the_substeps_off_flag_turns_the_registered_entry_plain(self):
        sigmas = self._sigmas(True, 8)
        runs = []
        for name, flag in (("euler_dy_cfg_pp", True), ("euler_dy_cfg_pp", False)):
            model, _ = self._models(True, _CfgModel)
            x = self._latent(_SHAPES[0], sigmas)
            options = {"transformer_options": {self.common.DY_SUBSTEPS_OFF_KEY: "Spectrum"}} if flag else {}
            if flag:
                runs.append(self._entry(name)(model, x, sigmas, extra_args={"model_options": options, "seed": 3},
                                              disable=True, noise_sampler=ReplayNoise(x, 1)))
            else:
                runs.append(self.dy.sample_euler_dy_cfg_pp(model, x, sigmas, extra_args={"seed": 3}, disable=True,
                                                           noise_sampler=ReplayNoise(x, 1), substeps=False))
            self.assertTrue(all(marker is None for marker, _shape in model.markers))
        self.assertTrue(torch.equal(_bits(runs[0]), _bits(runs[1])))

    def _inpaint_model(self, flow, shape, sigmas, mask_value=None):
        """ComfyUI's ``KSamplerX0Inpaint`` over a CFG toy (its ``inner_model.inner_model`` = a real
        ``BaseModel.scale_latent_inpaint`` on the model sampling)."""
        sampling = self.sampling[flow]
        cfg = _CfgModel(sampling, comfy_layout=True)
        base = types.SimpleNamespace(model_sampling=sampling)
        cfg.inner_model.inner_model = types.SimpleNamespace(
            scale_latent_inpaint=lambda **kw: self.model_base.BaseModel.scale_latent_inpaint(base, **kw))

        class Guider:   # the KSAMPLER-side model KSamplerX0Inpaint wraps (CFGGuider's surface)
            inner_model = cfg.inner_model.inner_model
            model_patcher = cfg.inner_model.model_patcher

            def __call__(self, x, sigma, model_options={}, seed=None):
                cfg.seen.append((tuple(x.shape), tuple(wrapper.latent_image.shape), tuple(wrapper.noise.shape)))
                return cfg(x, sigma, model_options=model_options, seed=seed)

        cfg.seen = []
        wrapper = self.samplers.KSamplerX0Inpaint(Guider(), sigmas)
        generator = torch.Generator().manual_seed(8)
        wrapper.latent_image = torch.randn(shape, generator=generator)
        wrapper.noise = torch.randn(shape, generator=generator)
        mask = torch.ones(shape) if mask_value == 1 else (torch.rand(shape, generator=generator) > 0.5).float()
        return wrapper, cfg, mask

    def test_masked_sub_steps_on_comfys_inpaint_wrapper(self):
        for name, flow, shape in itertools.product(self.reg.DY_SAMPLERS, (True, False), _SHAPES):
            with self.subTest(name=name, flow=flow, shape=shape):
                sigmas = self._sigmas(flow, 8)
                wrapper, cfg, mask = self._inpaint_model(flow, shape, sigmas)
                latent, noise = wrapper.latent_image, wrapper.noise
                x = self._latent(shape, sigmas)
                out = self._entry(name)(wrapper, x, sigmas, extra_args={"denoise_mask": mask, "seed": 3},
                                        disable=True, noise_sampler=ReplayNoise(x, 2))
                self.assertEqual(out.shape, x.shape)
                self.assertTrue(torch.isfinite(out).all())
                self.assertIs(wrapper.latent_image, latent)
                self.assertIs(wrapper.noise, noise)
                substeps = [seen for seen, (marker, _s) in zip(cfg.seen, cfg.markers) if marker]
                self.assertEqual(len(substeps), 2)
                for x_shape, latent_shape, noise_shape in substeps:
                    self.assertNotEqual(x_shape[-2:], shape[-2:])
                    self.assertEqual(latent_shape, x_shape)
                    self.assertEqual(noise_shape, x_shape)

    def test_an_all_ones_mask_is_no_mask(self):
        for name, flow in itertools.product(self.reg.DY_SAMPLERS, (True, False)):
            with self.subTest(name=name, flow=flow):
                sigmas = self._sigmas(flow, 8)
                runs = []
                for masked in (True, False):
                    wrapper, cfg, mask = self._inpaint_model(flow, _SHAPES[1], sigmas, mask_value=1)
                    x = self._latent(_SHAPES[1], sigmas)
                    extra = {"denoise_mask": mask, "seed": 3} if masked else {"seed": 3}
                    runs.append(self._entry(name)(wrapper if masked else cfg, x, sigmas, extra_args=extra,
                                                  disable=True, noise_sampler=ReplayNoise(x, 2)))
                self.assertTrue(torch.equal(_bits(runs[0]), _bits(runs[1])))

    # -- the free core names (X11) -------------------------------------------------------------------------

    def test_free_core_names_are_the_extensions_ports(self):
        cases = (
            ("cfgpp_ud10_ab", self.ext["cfgpp_ud10_ab"].sample_cfgpp_ud10_ab, _CfgModel, {}),
            ("ipndm", self.ext["ipndm_deis"].sample_ipndm, _StableModel, {}),
            ("deis", self.ext["ipndm_deis"].sample_deis, _StableModel, {}),
        )
        for (name, theirs, kind, kwargs), flow, steps, shape in itertools.product(
                cases, (True, False), (6, 13), _SHAPES):
            with self.subTest(name=name, flow=flow, steps=steps, shape=shape):
                ours = getattr(self.ckd, "sample_" + name)   # ComfyUI core's own function
                self._same(self._pair(lambda *a, noise_sampler=None, **k: ours(*a, **k),
                                      (lambda *a, noise_sampler=None, **k: theirs(*a, **k), "forge"),
                                      flow=flow, steps=steps, shape=shape, kind=kind, noise=False,
                                      ours_kwargs=kwargs, theirs_kwargs=kwargs), name, draws=False)

    def test_dpmpp_2m_sde_heun_is_the_extensions_with_a_shared_noise_sampler(self):
        for flow, eta, shape in itertools.product((True, False), (0.0, 0.5, 1.0), _SHAPES):
            with self.subTest(flow=flow, eta=eta, shape=shape):
                kwargs = {"eta": eta, "s_noise": 1.0}
                self._same(self._pair(self.ckd.sample_dpmpp_2m_sde_heun,
                                      (self.ext["dpmpp_flow"].sample_dpmpp_2m_sde_heun, "forge"), flow=flow,
                                      shape=shape, ours_kwargs=kwargs, theirs_kwargs=kwargs), f"eta={eta}")


    # -- IPNDM_V: sam-extra 0.33.1's fourth-order weight fix (the pack's ipndm_v_fixed) ---------------------------------

    def test_the_lq28_constant_is_comfyuis_linear_quadratic_list(self):
        self.assertTrue(torch.equal(_bits(self._sigmas(True, 28, "linear_quadratic")),
                                    _bits(torch.tensor(self.ipndm_helpers.LQ28, dtype=torch.float32))))

    def test_ipndm_v_fixed_is_sam_extra_0331s_ipndm_v(self):
        """The registered ``sample_ipndm_v_fixed`` on ComfyUI's module == sam-extra 0.33.1's IPNDM_V on Forge's (before the
        registry's sync: the installed 0.33.0 file with its one coeff4 line fixed — ``tests/_ipndm_v_oracle``), bit for
        bit, on ComfyUI's own schedules incl. Anima's Linear Quadratic."""
        theirs = self.ipndm_v_oracle
        cases = 0
        for scheduler, flow, steps, shape in itertools.product(
                ("linear_quadratic", "simple", "normal", "sgm_uniform", "beta", "karras", "exponential"), (True, False),
                (6, 13, 28), _SHAPES):
            with self.subTest(scheduler=scheduler, flow=flow, steps=steps, shape=shape, oracle=self.ipndm_v_how):
                ours = self._entry("ipndm_v_fixed")
                self._same(self._pair(lambda *a, noise_sampler=None, **k: ours(*a, **k),
                                      (lambda *a, noise_sampler=None, **k: theirs(*a, **k), "forge"),
                                      flow=flow, steps=steps, shape=shape, noise=False, scheduler=scheduler),
                           f"ipndm_v_fixed {scheduler}", draws=False)
                cases += 1
        self.assertEqual(cases, 84)

    def test_ipndm_v_fixed_is_comfys_ipndm_v_where_the_step_ratio_is_constant(self):
        """Only the typo differs from ComfyUI core's ``ipndm_v``: on lists whose step ratio is constant and whose steps
        are exact in binary (power-of-two equal steps, doubling and halving steps) the latents are bit for bit ComfyUI's;
        on a geometric list in floating point (``exponential``) only within rounding — equal ratios alone do not make
        the two coeff4 expressions round alike — and on Linear Quadratic they part."""
        core, ours = self.ckd.sample_ipndm_v, self._entry("ipndm_v_fixed")
        lists = []
        for n in (4, 8, 16, 32):
            equal = torch.arange(n, -1, -1, dtype=torch.float32) / n
            lists += [(f"equal {n}", equal), (f"equal {n} x8", equal * 8)]
        lists.append(("doubling", torch.tensor([2.0 ** k for k in range(4, -9, -1)] + [0.0])))
        lists.append(("halving", torch.tensor([1.0 - (2.0 ** k - 1) / 64 for k in range(7)] + [0.0])))
        for (label, sigmas), dtype, shape in itertools.product(lists, (torch.float32, torch.float64), _SHAPES):
            with self.subTest(sigmas=label, dtype=dtype, shape=shape):
                runs = []
                for function in (core, ours):
                    model = _StableModel(self.sampling[True], comfy_layout=True)
                    x = self._latent(shape, sigmas).to(dtype)
                    runs.append((function(model, x, sigmas.to(dtype), extra_args={}, disable=True), model.calls))
                self.assertTrue(torch.equal(_bits(runs[0][0]), _bits(runs[1][0])), label)
                self.assertEqual([c[:2] for c in runs[0][1]], [c[:2] for c in runs[1][1]])
        for scheduler, close in (("exponential", True), ("linear_quadratic", False)):
            with self.subTest(scheduler=scheduler):
                sigmas = self._sigmas(False, 20, scheduler) if scheduler == "exponential" else self._sigmas(True, 28, scheduler)
                outs = []
                for function in (core, ours):
                    model = _StableModel(self.sampling[scheduler == "linear_quadratic"], comfy_layout=True)
                    outs.append(function(model, self._latent(_SHAPES[0], sigmas).double(), sigmas.double(), extra_args={},
                                         disable=True))
                if close:
                    self.assertTrue(torch.allclose(outs[0], outs[1], rtol=1e-5, atol=1e-6))   # float32 schedule: ratios equal to ~1e-7
                else:
                    self.assertGreater(float((outs[0] - outs[1]).abs().max()), 1e-3)

    def test_the_constant_velocity_probe_through_comfyuis_tables(self):
        """``D = x − σ·v`` (``d = v``): the entry ComfyUI's KSampler runs for ``ipndm_v_fixed`` takes exact steps on Anima's
        Linear Quadratic list (weights summing to 1); ComfyUI's core ``ipndm_v`` sums to about −174 at step 15 (0-based;
        σ 0.968 → 0.952) — the green noise of the 10-05 GPU check."""
        sigmas = self._sigmas(True, 28, "linear_quadratic").double()
        for name in ("ipndm_v_fixed", "ipndm_v"):
            with self.subTest(sampler=name):
                function = self.samplers.sampler_object(name).sampler_function
                probe = self.ipndm_helpers.ConstantVelocity()
                sums = probe.step_sums(sigmas, function(probe, probe.x0.clone(), sigmas, extra_args={}, disable=True))
                if name == "ipndm_v_fixed":
                    self.assertIs(function, self._entry(name))
                    self.assertLess(max(abs(value - 1) for value in sums), 1e-9, sums)
                else:
                    self.assertTrue(-180 < sums[15] < -170, sums[15])


if __name__ == "__main__":
    unittest.main()
