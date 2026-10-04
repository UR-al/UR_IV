r"""Pack 1.7.0: the four sam-extra ports together on the real ComfyUI runtime (CPU).

Each feature has its own parity suite; this file checks what only shows when they meet (PLAN PB-INT2 and the
v0.33.0 amendment), on a tiny flow-matching model built from ComfyUI's own ``BaseModel`` (``ModelType.FLOW``,
shift 3) with a deterministic stand-in DiT that records every call, a real ``ModelPatcher`` and
``comfy.sample.sample_custom`` (the real ``CFGGuider``, ``KSAMPLER`` and SAMPLER_SAMPLE wrapper chain):

* **Euler (SMEA) Dy (CFG++) + Anima SPEED through CNS** — SPEED reads ``sam_extra_step_offset`` through CNS's
  ``_forge_neo_base`` and passes each segment's start, so the sub-steps stay on and run once, at the pass's own
  steps (the same sigmas as without SPEED), never again in a later segment.
* **The v0.33.0 samplers under SPEED** — Euler Dy, Euler SMEA Dy CFG++, ER SDE (Tunable) with the noise window,
  ER SDE (Reverse-time), the DPM++ flow-ODE trio, DPM++ 4M SDE and Restart (flow) give the same latents as the
  installed extension's ``run_speed`` placed in the same spot with the pack's host hooks.
* **Colorcraft + SPEED** — Colorcraft follows SPEED's re-published sigma list and grades the coarse-grid
  evaluations (masks and blur on the coarse grid); every evaluation equals the extension's Colorcraft callback.
* **Colorcraft + Dy** — the Dy/SMEA sub-step evaluations pass through ungraded, counted like the extension.
* **Extra Schedulers' carrier** survives ``ForgeNeoHiresFix`` with Shift > 0 (no override, a checkpoint
  override, a UNET override) — the pass's laplace/custom/flow_cosmos_rho7/flow_cosmos_dynamic schedule uses the
  carried values (the Flow Cosmos ρ / σ̃ max / σ̃ min since sam-extra v0.33.0 D19/D20) — and SPEED's flow-model
  (``CONST``) guard accepts the carrier's ``model_sampling`` copy.
* **All four carried onto a Hires override** (``hires_carry``) == the four nodes applied to the override MODEL
  directly: same sampled latents, Extra Samplers' values and Colorcraft (prepared with the override VAE) included.
* **The full stack** CNS + SPEED + Detail Daemon + Colorcraft: wrapper and post-CFG order, Detail Daemon scaling
  SPEED's hand-off sigma, and Colorcraft (pre-DD sigma, re-published list, Dy sub-steps) == the extension.

Skipped unless ``AISTUDIO_COMFY_TEST_ROOT`` is set. Run it with the portable python, the repo (or worktree) first on
``sys.path``, ``CUDA_VISIBLE_DEVICES=-1`` (ComfyUI is forced to ``--cpu`` before its model management loads; no
server, no GPU) and ``PYTHONDONTWRITEBYTECODE=1`` (nothing is written into the ComfyUI install)::

    $env:CUDA_VISIBLE_DEVICES='-1'; $env:PYTHONDONTWRITEBYTECODE='1'
    $env:AISTUDIO_COMFY_TEST_ROOT='C:\ComfyUI_windows_portable\ComfyUI'; $env:AISTUDIO_REQUIRE_FORGE_EXT='1'
    C:\ComfyUI_windows_portable\python_embeded\python.exe -X utf8 -c "import sys, unittest; sys.path.insert(0, r'<repo>'); unittest.main(module=None, argv=['x', 'tests.test_comfy_cross_features', '-v'])"
"""
from __future__ import annotations

import inspect
import io
import json
import os
import sys
import unittest
from types import SimpleNamespace
from unittest import mock

_COMFY_ROOT = os.environ.get("AISTUDIO_COMFY_TEST_ROOT")
if _COMFY_ROOT:  # force --cpu before anything loads ComfyUI's model management
    if _COMFY_ROOT not in sys.path:
        sys.path.append(_COMFY_ROOT)
    from comfy.cli_args import args as _comfy_args  # noqa: E402

    _comfy_args.cpu = True

from tests._optional_deps import bind_torch, requires_torch  # noqa: E402
from tests._sam_extra_ext import requires_extension  # noqa: E402

torch = None  # filled by bind_torch()

DY_SUBSTEP_STEPS = {"euler_dy": (2, 3), "euler_dy_cfg_pp": (2, 3), "euler_smea_dy": (0, 1),
                    "euler_smea_dy_cfg_pp": (0, 1)}
MANUAL = {"threshold": "manual", "manual_sigmas": "0.6"}
DD_VALUES = {"dd_amount": 0.4, "dd_start": 0.0, "dd_end": 1.0, "dd_bias": 0.5, "dd_smooth": True}


def _dit_class():
    class TinyDiT(torch.nn.Module):
        """A deterministic stand-in DiT (4-D and 5-D latents); records what every call saw."""

        def __init__(self, in_channels=16, mix_seed=5, device=None, operations=None, dtype=None, **kwargs):
            super().__init__()
            generator = torch.Generator().manual_seed(mix_seed)
            self.mix = torch.nn.Parameter(torch.randn(in_channels, in_channels, generator=generator) * 0.1)
            self.dtype = torch.float32
            self.calls = []

        def forward(self, x, t, context=None, control=None, transformer_options={}, **kwargs):
            self.calls.append({"shape": tuple(x.shape), "t": t.detach().clone(),
                               "marker": transformer_options.get("sam_extra_substep")})
            view = (-1,) + (1,) * (x.ndim - 1)
            mixed = torch.einsum("ij,bj...->bi...", self.mix, x)
            smooth = (x.roll(1, -1) + x.roll(-1, -1) + x.roll(1, -2) + x.roll(-1, -2)) * 0.25
            ctx = context.mean(dim=(1, 2)).view(view) if context is not None else 0.0
            return 0.6 * x + 0.2 * smooth + 0.1 * mixed + 0.05 * ctx + 0.001 * t.view(view)

    return TinyDiT


@requires_torch
@requires_extension
@unittest.skipUnless(_COMFY_ROOT, "Set AISTUDIO_COMFY_TEST_ROOT to run the cross-feature checks on ComfyUI (CPU)")
class CrossFeaturesOnRealComfyTests(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        bind_torch(globals())
        import comfy.model_management as management

        if management.get_torch_device().type != "cpu":
            raise unittest.SkipTest("ComfyUI did not start on CPU - refusing to touch the GPU")
        import comfy.latent_formats
        import comfy.model_base
        import comfy.model_patcher
        import comfy.model_sampling
        import comfy.ops
        import comfy.patcher_extension
        import comfy.sample
        import comfy.samplers
        from comfy_extras.nodes_model_advanced import ModelSamplingSD3

        import tests.test_comfy_colorcraft_parity as parity
        import tests.test_comfy_speed_runtime as speed_runtime
        from comfy_custom_nodes.ai_studio_forge_parity import (
            colorcraft_host,
            colorcraft_nodes,
            extra_samplers_dy,
            extra_samplers_math,
            extra_samplers_nodes,
            extra_samplers_register,
            extra_schedulers,
            generation,
            guidance_cns,
            guidance_common,
            guidance_dd,
            hires_carry,
            speed_nodes,
        )

        bind_torch(vars(parity))          # the shared stand-ins (fake VAE) use that module's torch
        bind_torch(vars(speed_runtime))   # its oracle wrapper uses that module's torch
        cls.management, cls.comfy_sample, cls.samplers = management, comfy.sample, comfy.samplers
        cls.model_base, cls.model_patcher, cls.ops = comfy.model_base, comfy.model_patcher, comfy.ops
        cls.model_sampling, cls.latent_formats, cls.sd3 = comfy.model_sampling, comfy.latent_formats, ModelSamplingSD3
        cls.sampler_sample = comfy.patcher_extension.WrappersMP.SAMPLER_SAMPLE
        cls.parity, cls.speed_runtime = parity, speed_runtime
        cls.cc_host, cls.cc_nodes, cls.dy, cls.xs_math = colorcraft_host, colorcraft_nodes, extra_samplers_dy, \
            extra_samplers_math
        cls.xs_nodes, cls.xs_register, cls.xsched = extra_samplers_nodes, extra_samplers_register, extra_schedulers
        cls.generation, cls.cns, cls.common, cls.dd, cls.carry = generation, guidance_cns, guidance_common, \
            guidance_dd, hires_carry
        cls.pack = cls.speed = speed_nodes          # the SPEED oracle helpers read ``self.pack``
        cls.ext_runner = speed_runtime._ext_speed("runner")
        cls.ext_fh = speed_runtime._ext_speed("forge_host")
        cls.hook = parity._ext_module("sam3ext.colorcraft.hook")
        cls.spec = parity._ext_module("sam3ext.colorcraft.spec")
        cls.panel = parity._ext_module("sam3ext.colorcraft.panel_state")
        cls.dave = parity._ext_module("sam3ext.guidance.dave_gate")
        cls.ext_sigmas = parity._ext_module("sam3ext.guidance.sigmas")
        cls.dit_class = _dit_class()
        # The pack's __init__ registers the names at import when ComfyUI is importable; make sure (idempotent).
        cls.xsched.install_extra_schedulers(required=True)
        cls.xs_register.install_extra_samplers(required=True)

    def setUp(self):
        self.cc_host.ANCHORS.clear()
        self.hook.clear_anchor_cache()
        self.dave.note_pre_dd_sigma(None, None)
        self.addCleanup(self.dave.note_pre_dd_sigma, None, None)

    # -- a tiny real model ---------------------------------------------------------------------------------------

    def _patcher(self, fmt=None, *, mix_seed: int = 5, shift: float = 3.0):
        fmt = fmt if fmt is not None else self.latent_formats.Flux()
        config = SimpleNamespace(
            unet_config={"in_channels": 16, "mix_seed": mix_seed}, latent_format=fmt, manual_cast_dtype=None,
            custom_operations=self.ops.disable_weight_init, optimizations={},
            sampling_settings={"shift": shift, "multiplier": 1.0}, memory_usage_factor=1.0,
        )
        cpu = torch.device("cpu")
        base = self.model_base.BaseModel(config, model_type=self.model_base.ModelType.FLOW, device=cpu,
                                         unet_model=self.dit_class)
        return self.model_patcher.ModelPatcher(base, load_device=cpu, offload_device=cpu)

    def _sigmas(self, model, steps: int = 14, scheduler: str = "simple"):
        return self.samplers.calculate_sigmas(model.get_model_object("model_sampling"), scheduler, steps).cpu()

    def _sample(self, model, sampler_name, sigmas, *, shape=(1, 16, 16, 16), seed=7, latent=None, noise_seed=2):
        generator = torch.Generator().manual_seed(noise_seed)
        noise = torch.randn(shape, generator=generator)
        latent = torch.zeros(shape) if latent is None else latent
        cond_gen = torch.Generator().manual_seed(1)
        positive = [[torch.randn(shape[0], 3, 8, generator=cond_gen), {}]]
        negative = [[torch.zeros(shape[0], 3, 8), {}]]
        steps = []
        out = self.comfy_sample.sample_custom(
            model, noise, 3.0, self.samplers.sampler_object(sampler_name), sigmas, positive, negative, latent,
            callback=lambda step, x0, x, total: steps.append(step), disable_pbar=True, seed=seed,
        )
        calls = list(model.model.diffusion_model.calls)
        model.model.diffusion_model.calls.clear()
        return out, calls, steps

    def _speed(self, model, **values):
        values = {"enabled": True, **MANUAL, "image_seed": 40, **values}
        return self.speed.ForgeNeoAnimaSpeed().patch(model, **values)[0]

    def _speed_status(self, model):
        (wrapper,) = model.wrappers[self.sampler_sample][self.speed.SPEED_WRAPPER_KEY]
        return wrapper._ai_studio_speed_status.text

    def _handoff(self, sigmas, shape=(1, 16, 16, 16), **values):
        settings = self.speed.coerce_settings({"enabled": True, **MANUAL, **values})
        return self.speed.plan_for(settings, torch.zeros(shape), sigmas).transitions[0]

    # -- Colorcraft: settings, recorder, replay through the extension's callback ---------------------------------

    def _cc_config(self):
        config = self.spec.default_config()
        config.enabled, config.masking = True, True
        config.modifiers[0].update({"active": True, "kind": "Advanced", "pass": "Both", "start": 0.0, "end": 0.9,
                                    "exposure": 0.6, "contrast": 0.8, "temperature": 1.1, "clarity": 1.2,
                                    "color_shift_amount": 0.5, "color_shift_red": 0.3, "mask": "M1"})
        config.modifiers[1].update({"active": True, "kind": "Punch", "pass": "Base", "start": 0.2, "end": 1.0,
                                    "contrast": -0.5, "sharpness": 0.9})
        config.leaves[0].update({"mask_axis": "exposure", "mask_mode": "highs", "blur": 12.0, "mask_hardness": 2.0})
        return config

    def _cc_json(self, config, hires=False):
        chain = self.spec.build_chain(config, hires)
        return json.dumps(self.parity._settings(*chain.entries, tags=chain.tags, pass_name="hires" if hires else "base",
                                                infotext=self.spec.to_infotext(config)))

    def _colorcraft(self, model, config, *, vae=None, hires=False, pre_dd=True):
        rank = 3 if model.model.latent_format.latent_dimensions == 3 else 2
        vae = vae if vae is not None else self.parity._FakeVAE(16, rank)
        return self.cc_nodes.ForgeNeoColorcraft().patch(model, vae, self._cc_json(config, hires), "v1", pre_dd)[0]

    def _record(self, model):
        """Wrap Colorcraft's post-CFG function on ``model`` in a recorder; returns the log (sigma lists by object)."""
        log = []
        functions = model.model_options["sampler_post_cfg_function"]
        index = next(i for i, fn in enumerate(functions) if self.cc_host.is_owned(fn))
        original = functions[index]

        def recorder(args):
            transformer = args["model_options"]["transformer_options"]
            note = transformer.get(self.common.PRE_DD_SIGMAS_KEY)
            entry = {"x0": args["denoised"].clone(), "sigma": args["sigma"].clone(),
                     "sigmas": transformer["sample_sigmas"], "marker": transformer.get("sam_extra_substep"),
                     "note": None if note is None else note.clone()}
            result = original(args)
            entry["passed"] = result is args["denoised"]
            entry["out"] = result.clone()
            log.append(entry)
            return result

        functions[index] = recorder
        return log

    def _replay(self, log, config, fmt, *, full_hw, pre_dd=True, hires=False, vae_seed=0):
        """The extension's Colorcraft callback on the recorded evaluations, each with the sigma list object the
        pack saw (SPEED's re-published copy carries the extension's tag) and its Dy marker; all results equal."""
        vae = self.parity._FakeVAE(16, 3 if fmt.latent_dimensions == 3 else 2, seed=vae_seed)
        p = SimpleNamespace(sd_model=SimpleNamespace(model_config=SimpleNamespace(latent_format=fmt),
                                                     forge_objects=SimpleNamespace(vae=vae,
                                                                                   unet=self.parity._FakeUnet())),
                            is_hr_pass=hires, extra_generation_params={}, sampler=None)
        args = [self.spec.to_infotext(config)] + self.panel.default_args()[1:]
        with mock.patch("sys.stderr", new=io.StringIO()):
            run = self.hook.process(p, args, full_hw=full_hw)
        run.pre_dd = pre_dd
        callback = p.sd_model.forge_objects.unet.model_options["sampler_post_cfg_function"][-1]
        for index, entry in enumerate(log):
            if entry["note"] is not None:
                self.dave.note_pre_dd_sigma(entry["note"], entry["sigma"])
            else:
                self.dave.note_pre_dd_sigma(None, None)
            transformer = {"sampling_sigmas": entry["sigmas"]}
            if entry["marker"]:
                transformer["sam_extra_substep"] = entry["marker"]
            x0 = entry["x0"].clone()
            out = callback({"denoised": x0, "sigma": entry["sigma"], "model_options": {"transformer_options": transformer}})
            self.assertEqual(out is x0, entry["passed"], f"eval {index}: pass-through differs")
            self.assertTrue(torch.equal(out, entry["out"]),
                            f"eval {index}: max |d| {(out.float() - entry['out'].float()).abs().max().item()}")
        return run

    def _compare_reports(self, model, run):
        report, status = self.cc_host.model_colorcraft_patch(model).last_report, run.status
        for name in ("evals", "edited", "idle", "foreign", "substeps", "errors", "nonfinite"):
            self.assertEqual(getattr(report, name), getattr(status, name), name)
        return report

    # -- 1. Dy + SPEED through CNS ---------------------------------------------------------------------------------

    def test_dy_substeps_stay_on_under_speed_through_cns(self):
        for name, steps in DY_SUBSTEP_STEPS.items():
            with self.subTest(sampler=name):
                plain = self.cns.apply_cns(self._patcher())
                sigmas = self._sigmas(plain)
                handoff = self._handoff(sigmas)
                self.assertGreater(handoff.step, max(steps) + 1, "the hand-off must come after the sub-steps")
                _, plain_calls, _ = self._sample(plain, name, sigmas)
                model = self.cns.apply_cns(self._speed(self._patcher()))
                self.assertEqual(list(model.wrappers[self.sampler_sample]),
                                 [self.cns.CNS_WRAPPER_KEY, self.speed.SPEED_WRAPPER_KEY])
                seen, real = [], self.dy.euler_dy

                def spy(*args, **kwargs):
                    seen.append((kwargs.get("sam_extra_step_offset"), kwargs.get("substeps"), len(args[2]) - 1))
                    return real(*args, **kwargs)

                with mock.patch.object(self.dy, "euler_dy", spy):
                    out, calls, callback_steps = self._sample(model, name, sigmas)
                self.assertTrue(self._speed_status(model).startswith("applied (transition)"), self._speed_status(model))
                n = len(sigmas) - 1
                self.assertEqual(seen, [(0, True, handoff.step), (handoff.step, True, n - handoff.step)])
                self.assertEqual(callback_steps, list(range(n)))
                marked = [call for call in calls if call["marker"]]
                plain_marked = [call for call in plain_calls if call["marker"]]
                self.assertEqual(len(marked), 2)
                self.assertEqual([call["marker"] for call in marked], [call["marker"] for call in plain_marked])
                for call, plain_call, step in zip(marked, plain_marked, steps):
                    self.assertTrue(torch.equal(call["t"], plain_call["t"]))
                    self.assertTrue(torch.equal(call["t"][:1], sigmas[step:step + 1].to(call["t"].dtype)))
                self.assertTrue(any(call["shape"][-1] == 8 for call in calls), "coarse calls reached the DiT")
                self.assertTrue(torch.isfinite(out).all())
        # CNS's own function does not name the keyword; SPEED reads it through ``_forge_neo_base``.
        coloured = self.cns.cns_sampler_function(self.samplers.ksampler("euler_dy").sampler_function, 1.0, 1.0, 1.0)
        self.assertNotIn(self.common.STEP_OFFSET_KWARG, inspect.signature(coloured).parameters)
        self.assertTrue(self.common.declares_keyword(coloured, self.common.STEP_OFFSET_KWARG))

    # -- 2. the v0.33.0 samplers under SPEED == the extension's runner ----------------------------------------------

    def _speed_vs_extension(self, sampler_name, *, values=None, steps=14, pre=None, sequential=True):
        values = {"enabled": True, **MANUAL, **(values or {})}
        pre = pre or (lambda model: model)
        pack_model = self.speed.ForgeNeoAnimaSpeed().patch(
            pre(self._patcher()), image_seed=40, **{k: v for k, v in values.items() if k in self.speed.ARG_NAMES})[0]
        oracle_model = pre(self._patcher()).clone()
        record: dict = {}
        oracle_model.add_wrapper_with_key(
            self.sampler_sample, "oracle",
            self.speed_runtime.SpeedOnRealComfyTests._oracle(self, values, "base", 40, True, record))
        sigmas = self._sigmas(pack_model, steps)
        got, got_calls, got_steps = self._sample(pack_model, sampler_name, sigmas)
        want, want_calls, want_steps = self._sample(oracle_model, sampler_name, sigmas)
        self.assertEqual(self._speed_status(pack_model),
                         f"applied ({values.get('mode', 'transition')}) - {record['summary']}")
        self.assertTrue(torch.isfinite(got).all())
        self.assertTrue(torch.equal(got, want), f"max diff {(got.double() - want.double()).abs().max()}")
        self.assertEqual([(c["shape"], c["marker"]) for c in got_calls], [(c["shape"], c["marker"]) for c in want_calls])
        self.assertTrue(all(torch.equal(g["t"], w["t"]) for g, w in zip(got_calls, want_calls)))
        self.assertEqual(got_steps, want_steps)
        if sequential:
            self.assertEqual(got_steps, list(range(len(sigmas) - 1)))
        self.assertTrue(any(call["shape"][-1] == 8 for call in got_calls), "coarse calls reached the DiT")
        return got_calls

    def test_v033_samplers_under_speed_equal_the_extension_runner(self):
        def window(model):
            return self.xs_nodes.ForgeNeoExtraSamplerSettings().patch(
                model, max_stage=2, eta=0.6, noise_window=True, noise_start=0.1, noise_end=0.9)[0]

        rows = [
            ("euler_dy", {}, None),
            ("euler_smea_dy_cfg_pp", {}, None),
            ("euler_dy", {"mode": "respace"}, None),
            ("er_sde_tunable", {}, window),
            ("er_sde_reverse_time", {}, window),
            ("er_sde_ode", {"mode": "respace"}, None),
            ("dpmpp_2m_flow_ode", {}, None),
            ("dpmpp_2m_heun_flow_ode", {}, None),
            ("dpmpp_3m_flow_ode", {"mode": "respace"}, None),
            ("dpmpp_4m_sde", {}, None),
        ]
        for sampler_name, values, pre in rows:
            with self.subTest(sampler=sampler_name, values=values, window=pre is not None):
                self._speed_vs_extension(sampler_name, values=values, pre=pre)
        with self.subTest(sampler="restart_flow"):
            # 30 steps: the coarse segment and the tail each re-plan Restart (sam-extra's per-segment rule).
            calls = self._speed_vs_extension("restart_flow", steps=30, sequential=False)
            self.assertGreater(len(calls), 2 * 30 - 1)

    # -- 3. Colorcraft + SPEED -----------------------------------------------------------------------------------

    def test_colorcraft_follows_speeds_republished_sigmas_on_the_coarse_grid(self):
        rows = [("Flux", (1, 16, 16, 24), "euler", {}), ("Flux", (1, 16, 16, 24), "dpmpp_2m", {"mode": "respace"}),
                ("Wan21", (1, 16, 1, 16, 24), "euler", {})]
        for fmt_name, shape, sampler_name, values in rows:
            with self.subTest(fmt=fmt_name, sampler=sampler_name, values=values):
                self.setUp()
                fmt = getattr(self.latent_formats, fmt_name)()
                config = self._cc_config()
                model = self._colorcraft(self._speed(self._patcher(fmt), **values), config)
                log = self._record(model)
                sigmas = self._sigmas(model)
                self._sample(model, sampler_name, sigmas, shape=shape)
                self.assertTrue(self._speed_status(model).startswith("applied"), self._speed_status(model))
                coarse = [entry for entry in log if entry["x0"].shape[-2:] != shape[-2:]]
                republished = [entry for entry in log if self.ext_sigmas.republished_from(entry["sigmas"]) is not None]
                self.assertTrue(coarse, "no coarse-grid evaluation was graded")
                self.assertTrue(republished, "SPEED never re-published the sigma list")
                self.assertTrue(all(self.ext_sigmas.republished_from(e["sigmas"]) is log[0]["sigmas"]
                                    for e in republished))
                run = self._replay(log, config, fmt, full_hw=shape[-2:])
                report = self._compare_reports(model, run)
                self.assertGreater(report.edited, len(log) // 2)
                self.assertTrue(any(not e["passed"] for e in coarse), "coarse evaluations were not graded")

    # -- 4. Colorcraft + Dy ----------------------------------------------------------------------------------------

    def test_colorcraft_passes_dy_substeps_through(self):
        for name in DY_SUBSTEP_STEPS:
            with self.subTest(sampler=name):
                self.setUp()
                fmt = self.latent_formats.Flux()
                config = self._cc_config()
                model = self._colorcraft(self._patcher(fmt), config)
                log = self._record(model)
                shape = (1, 16, 16, 24)
                self._sample(model, name, self._sigmas(model, 10), shape=shape)
                marked = [entry for entry in log if entry["marker"]]
                self.assertEqual(len(marked), 2)
                self.assertTrue(all(entry["passed"] for entry in marked))
                self.assertTrue(all(torch.equal(entry["out"], entry["x0"]) for entry in marked))
                report = self._compare_reports(model, self._replay(log, config, fmt, full_hw=shape[-2:]))
                self.assertEqual(report.substeps, 2)
                self.assertEqual(report.evals, len(log) - 2)

    # -- 5./6. Extra Schedulers' carrier: Hires fix with Shift > 0, overrides, SPEED's CONST guard ------------------

    XS_VALUES = {"custom_mode": "expression", "custom_expression": "m + (M - m) * (1 - x) ** 2",
                 "laplace_mu": 0.7, "laplace_beta": 0.35,
                 "flow_cosmos_rho": 4.5, "flow_cosmos_sigma_max": 240.0, "flow_cosmos_sigma_min": 0.006}
    # the schedulers that read the carried values (extra_schedulers.VALUE_SCHEDULERS)
    XS_SCHEDULERS = ("laplace", "custom", "flow_cosmos_rho7", "flow_cosmos_dynamic")

    def _xs(self, model, **values):
        return self.xsched.ForgeNeoExtraSchedulers().patch(model, **(values or self.XS_VALUES))[0]

    def _shifted(self, model, shift):
        return self.sd3().patch(model, shift, 1.0)[0]

    def _providers(self, *, checkpoint=None, unet=None):
        def provider(name, *, method, feature, args=(), kwargs=None):
            if name == "ModelSamplingSD3":
                return self.sd3().patch(*args)
            if name == "CheckpointLoaderSimple":
                return checkpoint
            if name == "UNETLoader":
                return (unet,)
            if name == "CLIPTextEncode":
                return (f"encoded:{args[1]}",)
            raise AssertionError(f"unexpected provider {name}")

        return provider

    def _hires(self, model, *, shift, checkpoint=None, unet=None, name=""):
        def get_full_path(kind, item):
            if kind == "checkpoints" and checkpoint is not None:
                return f"C:/models/{item}"
            if kind == "diffusion_models" and unet is not None:
                return f"C:/unet/{item}"
            return None

        captured = []
        with (
            mock.patch.object(self.generation, "invoke_provider",
                              side_effect=self._providers(checkpoint=checkpoint, unet=unet)),
            mock.patch.object(self.generation, "folder_paths_module",
                              return_value=SimpleNamespace(get_full_path=get_full_path)),
            mock.patch.object(self.generation, "_common_sample",
                              side_effect=lambda m, *a, **k: (captured.append(m) or {"samples": a[7]},)),
        ):
            self.generation.ForgeNeoHiresFix().run(
                model, "positive", "negative", {"samples": torch.zeros(1, 16, 8, 8)}, seed=3, enabled=True, steps=6,
                shift=shift, upscale_method="latent:nearest-exact", checkpoint_name=name, base_vae="base-vae",
                base_clip="base-clip",
            )
        (sampled,) = captured
        return sampled

    def _schedules(self, model, steps=10):
        sampling = model.get_model_object("model_sampling")
        return {name: self.samplers.calculate_sigmas(sampling, name, steps) for name in self.XS_SCHEDULERS}

    def _assert_schedules(self, got, want, default):
        self.assertEqual(self.XS_SCHEDULERS, self.xsched.VALUE_SCHEDULERS)
        for name in self.XS_SCHEDULERS:
            with self.subTest(scheduler=name):
                self.assertTrue(torch.equal(got[name], want[name]), f"{name}: {got[name]} != {want[name]}")
                self.assertFalse(torch.equal(got[name], default[name]), f"{name}: the carried values made no change")

    def test_extra_schedulers_carrier_survives_hires_fix_with_shift_and_overrides(self):
        attr = self.xsched.CARRIER_ATTR
        with self.subTest(case="no override, Shift 5"):
            sampled = self._hires(self._xs(self._patcher()), shift=5.0)
            sampling = sampled.get_model_object("model_sampling")
            self.assertEqual(sampling.shift, 5.0)
            self.assertIsInstance(sampling, self.model_sampling.CONST)
            self.assertEqual(getattr(sampling, attr), self.xsched.model_settings(self._xs(self._patcher())))
            self._assert_schedules(self._schedules(sampled), self._schedules(self._xs(self._shifted(self._patcher(), 5.0))),
                                   self._schedules(self._xs(self._shifted(self._patcher(), 5.0), laplace_mu=0.0)))
        with self.subTest(case="checkpoint override, Shift 3"):
            override = self._patcher(mix_seed=9, shift=1.0)
            sampled = self._hires(self._xs(self._patcher()), shift=3.0, checkpoint=(override, "clip", "vae"),
                                  name="other.safetensors")
            self.assertIs(sampled.model, override.model)
            self.assertEqual(sampled.get_model_object("model_sampling").shift, 3.0)
            expected_model = self._xs(self._shifted(self._patcher(mix_seed=9, shift=1.0), 3.0))
            self._assert_schedules(self._schedules(sampled), self._schedules(expected_model),
                                   self._schedules(self._shifted(self._patcher(mix_seed=9, shift=1.0), 3.0)))
            self.assertFalse(hasattr(override.get_model_object("model_sampling"), attr))
        with self.subTest(case="UNET override, no shift"):
            override = self._patcher(mix_seed=9, shift=2.0)
            sampled = self._hires(self._xs(self._patcher()), shift=0.0, unet=override, name="other_unet.safetensors")
            self.assertEqual(sampled.get_model_object("model_sampling").shift, 2.0)
            self._assert_schedules(self._schedules(sampled), self._schedules(self._xs(self._patcher(mix_seed=9, shift=2.0))),
                                   self._schedules(self._patcher(mix_seed=9, shift=2.0)))

    def test_speeds_flow_guard_accepts_the_extra_schedulers_carrier(self):
        for scheduler in self.XS_SCHEDULERS:
            with self.subTest(scheduler=scheduler):
                carried = self._speed(self._xs(self._patcher()))
                sampling = carried.get_model_object("model_sampling")
                self.assertIsInstance(sampling, self.model_sampling.CONST)
                self.assertIsNotNone(getattr(sampling, self.xsched.CARRIER_ATTR, None))
                sigmas = self._sigmas(carried, 14, scheduler)
                self.assertFalse(torch.equal(sigmas, self._sigmas(self._patcher(), 14, scheduler)))
                out, calls, _ = self._sample(carried, "euler", sigmas)
                self.assertTrue(self._speed_status(carried).startswith("applied (transition)"),
                                self._speed_status(carried))
                plain = self._speed(self._patcher())
                want, want_calls, _ = self._sample(plain, "euler", sigmas)
                self.assertTrue(torch.equal(out, want))
                self.assertEqual([c["shape"] for c in calls], [c["shape"] for c in want_calls])

    # -- 7. all four carried onto a Hires override == applied to it directly ---------------------------------------

    def _four(self, model, vae, config):
        model = self._xs(model)
        model = self.xs_nodes.ForgeNeoExtraSamplerSettings().patch(model, max_stage=2, eta=0.6, noise_window=True,
                                                                   noise_start=0.1, noise_end=0.9)[0]
        model = self._speed(model, pass_label="hires", manual_sigmas="0.5")
        return self._colorcraft(model, config, vae=vae, hires=True)

    def test_all_four_carried_onto_an_override_equal_the_nodes_applied_to_it(self):
        config = self._cc_config()
        base_vae, override_vae = self.parity._FakeVAE(16, 2), self.parity._FakeVAE(16, 2, seed=9)
        source = self._four(self._patcher(), base_vae, config)
        target = self._patcher(mix_seed=9)
        before = base_vae.calls
        carried = self.carry.apply(source, target, override_vae)
        self.assertIsNot(carried, target)
        self.assertEqual(base_vae.calls, before)
        self.assertGreater(override_vae.calls, 0)
        self.assertEqual(target.wrappers.get(self.sampler_sample, {}), {})
        sampling = carried.get_model_object("model_sampling")
        self.assertEqual(getattr(sampling, self.xsched.CARRIER_ATTR),
                         getattr(source.get_model_object("model_sampling"), self.xsched.CARRIER_ATTR))
        self.assertEqual(carried.model_options[self.xs_math.MODEL_OPTIONS_KEY],
                         source.model_options[self.xs_math.MODEL_OPTIONS_KEY])
        self.assertIn(self.speed.SPEED_WRAPPER_KEY, carried.wrappers[self.sampler_sample])
        self.assertTrue(self.cc_host.is_owned(carried.model_options["sampler_post_cfg_function"][-1]))
        patch = self.cc_host.model_colorcraft_patch(carried)
        self.assertEqual(patch.prepared.pass_name, "hires")

        direct = self._four(self._patcher(mix_seed=9), override_vae, config)
        bare = self._patcher(mix_seed=9)
        sigmas = self._sigmas(carried, 20, "laplace")[-10:]
        self.assertTrue(torch.equal(sigmas, self._sigmas(direct, 20, "laplace")[-10:]))
        latent = torch.randn(1, 16, 16, 16, generator=torch.Generator().manual_seed(6))
        got, got_calls, _ = self._sample(carried, "er_sde_tunable", sigmas, latent=latent)
        want, want_calls, _ = self._sample(direct, "er_sde_tunable", sigmas, latent=latent)
        self.assertTrue(self._speed_status(carried).startswith("applied (transition)"), self._speed_status(carried))
        self.assertIn("init=rescaled", self._speed_status(carried))
        self.assertTrue(torch.isfinite(got).all())
        self.assertTrue(torch.equal(got, want), f"max diff {(got - want).abs().max()}")
        self.assertEqual(len(got_calls), len(want_calls))
        self.assertGreater(patch.last_report.edited, 3)
        plain, _, _ = self._sample(bare, "er_sde_tunable", sigmas, latent=latent)
        self.assertFalse(torch.equal(got, plain))
        # the carried Extra Samplers values are the ones the sampler used
        without = carried.clone()                   # clone copies model_options' dicts and lists
        del without.model_options[self.xs_math.MODEL_OPTIONS_KEY]
        other, _, _ = self._sample(without, "er_sde_tunable", sigmas, latent=latent)
        self.assertFalse(torch.equal(got, other))

    # -- 8. the full stack: CNS + SPEED + Detail Daemon + Colorcraft --------------------------------------------------

    def test_full_stack_order_and_colorcraft_equal_the_extension(self):
        dd_json = json.dumps(DD_VALUES)
        for sampler_name in ("euler", "euler_ancestral", "euler_dy_cfg_pp"):
            with self.subTest(sampler=sampler_name):
                self.setUp()
                fmt = self.latent_formats.Flux()
                config = self._cc_config()
                dd_model = self.dd.ForgeNeoAnimaDetailDaemon().patch(self._patcher(fmt), True, dd_json)[0]
                model = self.cns.apply_cns(self._colorcraft(self._speed(dd_model), config, pre_dd=True))
                order = list(model.wrappers[self.sampler_sample])
                self.assertEqual(order[:2], [self.cns.CNS_WRAPPER_KEY, self.speed.SPEED_WRAPPER_KEY])
                self.assertLess(order.index(self.speed.SPEED_WRAPPER_KEY), order.index(self.dd.DD_WRAPPER_KEY))
                self.assertIn(self.cc_host.WRAPPER_KEY, order)
                self.assertTrue(self.cc_host.is_owned(model.model_options["sampler_post_cfg_function"][-1]))
                log = self._record(model)
                sigmas = self._sigmas(model)
                shape = (1, 16, 16, 16)
                out, calls, _ = self._sample(model, sampler_name, sigmas, shape=shape)
                self.assertTrue(self._speed_status(model).startswith("applied (transition)"), self._speed_status(model))
                self.assertTrue(torch.isfinite(out).all())
                self.assertTrue(all(entry["note"] is not None for entry in log), "Detail Daemon noted every sigma")
                self.assertTrue(any(self.ext_sigmas.republished_from(e["sigmas"]) is not None for e in log))
                if sampler_name == "euler_dy_cfg_pp":
                    self.assertEqual(sum(1 for entry in log if entry["marker"]), 2)
                report = self._compare_reports(model, self._replay(log, config, fmt, full_hw=shape[-2:]))
                self.assertGreater(report.edited, 5)
                self.assertGreater(report.pre_dd, 0)
                if sampler_name == "euler":
                    # Detail Daemon scales SPEED's aligned hand-off sigma with the hand-off step's schedule value.
                    values = self.dd.detail_daemon_node_values(DD_VALUES, 0.0)
                    schedule = torch.tensor(self.dd.detail_daemon_schedule(len(sigmas) - 1, values), dtype=torch.float32)
                    handoff = self._handoff(sigmas, shape)
                    first_full = next(call for call in calls if call["shape"][-1] == shape[-1])
                    expected = torch.tensor(handoff.aligned, dtype=torch.float32) * max(
                        1e-06, 1.0 - schedule[handoff.step].item() * 0.1 * 3.0)
                    self.assertTrue(torch.equal(first_full["t"][:1], expected.reshape(1)))


if __name__ == "__main__":
    unittest.main()
