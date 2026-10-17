r"""Pack 1.9.0 MOSS Spectrum on the real ComfyUI runtime (CPU) with a tiny MOSS-Image model.

The DiT is the installed ``moss-image`` node's own (``custom_nodes/moss-image``: ``loaders.MOSSImageConfig`` ->
``MOSSImage`` -> ``model.MOSSImageTransformer``; 3 blocks of width 64, seeded random weights), in a real
``ModelPatcher``, sampled through ``comfy.sample.sample_custom`` — the real ``CFGGuider`` (OUTER_SAMPLE wrappers,
``sample_sigmas``, cond/uncond batching with ``uuids``), ``KSAMPLER`` and ``BaseModel.apply_model``'s APPLY_MODEL
wrapper chain. A spy APPLY_MODEL wrapper ahead of the node's records every model call (input, sigma, branches, output,
whether the DiT ran):

* head: ``moss_head`` on the image tokens entering ``final_norm_video`` equals the model's own output bit for bit
  (one sample per call), and within float rounding for a CFG batch (the head runs batched, as upstream's);
* 50 Euler steps, CFG 3, ``flex_window`` 0.25 / 0.5 / 1.0: the DiT runs on 22 / 19 / 16 steps — exactly
  ``planned_actual_steps``; every call that ran the model returns bit for bit what the unpatched model returns for
  that input; every forecast call equals the unpatched model whose ``final_norm_video`` input is replaced by the
  explicit forecast (Chebyshev ridge + Taylor, ``tests/_moss_spectrum_oracle``) of each branch's features from the
  earlier calls; the run equals the plain run bit for bit until the first forecast; with nothing to forecast
  (warmup >= steps) the whole run is the plain run;
* a second run on the same patched MODEL (a 25-step denoise tail) starts at step 0 with its own step count; after
  each run no hook is left on ``final_norm_video``, no forecast tensor is alive, the MODEL's options carry no run;
* ``dpmpp_2s_ancestral``'s midpoint calls are off the grid: they run the model and the schedule stays a 20-step one;
* ``dpmpp_2m_sde``, ``er_sde`` and the pack's ``dpmpp_2m_flow_ode`` start at ComfyUI's ``offset_first_sigma_for_snr``
  sigma (just below 1.0): those calls are step 0, as upstream counts them, and 50 steps run the model on the planned
  19, nothing off the grid;
* with the pack's Detail Daemon the model gets scaled sigmas and the steps follow the sigma it noted (no off-grid
  call); with ``ModelSamplingSD3`` shift 5 and ``SkipLayerGuidanceDiTSimple`` (MOSS's pipeline) cond and uncond are
  separate calls with their own forecasters; a batch of 2 is forecast in batch order;
* the pack's Anima SPEED steps aside on this MODEL (the output is the run without SPEED) and Euler Dy's guard sees
  the wrapper in the run's real ``extra_args`` (and not a probe wrapper whose module says "spectrum"); a model
  without MOSS's head is returned unchanged with a warning;
* the installed moss-image ``model.py`` is the file whose SHA-256 THIRD_PARTY_NOTICES.md names for the restated head.

Skipped unless ``AISTUDIO_COMFY_TEST_ROOT`` is set and the moss-image node is installed. Run it with the portable
python, the repo first on ``sys.path``, ``CUDA_VISIBLE_DEVICES=-1`` (ComfyUI is forced to ``--cpu`` before its model
management loads; no server, no GPU) and ``PYTHONDONTWRITEBYTECODE=1`` (nothing is written into the ComfyUI
install)::

    $env:CUDA_VISIBLE_DEVICES='-1'; $env:PYTHONDONTWRITEBYTECODE='1'
    $env:AISTUDIO_COMFY_TEST_ROOT='C:\ComfyUI_windows_portable\ComfyUI'
    C:\ComfyUI_windows_portable\python_embeded\python.exe -X utf8 -c "import sys, unittest; sys.path.insert(0, r'<repo>'); unittest.main(module=None, argv=['x', 'tests.test_comfy_moss_spectrum_runtime', '-v'])"
"""
from __future__ import annotations

import gc
import hashlib
import importlib
import os
import re
import sys
import types
import unittest
import weakref
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

_COMFY_ROOT = os.environ.get("AISTUDIO_COMFY_TEST_ROOT")
if _COMFY_ROOT:  # force --cpu before anything loads ComfyUI's model management
    if _COMFY_ROOT not in sys.path:
        sys.path.append(_COMFY_ROOT)
    from comfy.cli_args import args as _comfy_args  # noqa: E402

    _comfy_args.cpu = True

from tests._moss_spectrum_oracle import explicit_forecast  # noqa: E402
from tests._optional_deps import bind_torch, requires_torch  # noqa: E402

torch = None  # filled by bind_torch()

MOSS_NODE = Path(_COMFY_ROOT or ".", "custom_nodes", "moss-image")
PACK_DIR = Path(__file__).resolve().parent.parent / "comfy_custom_nodes" / "ai_studio_forge_parity"
NOTICES = PACK_DIR / "THIRD_PARTY_NOTICES.md"
_PACKAGE = "_aistudio_test_moss_image"
SPY_KEY = "test.call_recorder"
LATENT = (16, 4, 6)          # channels, height, width of the tiny latent


def _moss_loaders():
    """moss-image's ``loaders`` from the installed node. The folder name has a hyphen, so it is imported under a
    package stub (its ``__init__``, which registers the node with ComfyUI, is not run)."""
    if _PACKAGE not in sys.modules:
        package = types.ModuleType(_PACKAGE)
        package.__path__ = [str(MOSS_NODE)]
        sys.modules[_PACKAGE] = package
    return importlib.import_module(f"{_PACKAGE}.loaders")


class _Recorder:
    """An APPLY_MODEL wrapper ahead of the node's: every model call's arguments, output and whether the DiT ran."""

    def __init__(self, dit):
        self.calls = []
        self.forwards = 0
        self._handle = dit.register_forward_pre_hook(self._count)

    def _count(self, module, inputs):
        self.forwards += 1

    def wrapper(self, executor, *args, **kwargs):
        before = self.forwards
        out = executor(*args, **kwargs)
        options = args[5]
        self.calls.append(SimpleNamespace(
            args=(args[0].clone(), args[1].clone()) + tuple(args[2:]), kwargs=dict(kwargs), out=out.clone(),
            ran=self.forwards > before, branches=list(options.get("cond_or_uncond") or [0]),
            uuids=[str(uid) for uid in options.get("uuids") or []], grid=options.get("sample_sigmas"),
            noted=options.get("ai_studio_pre_dd_sigmas"),
        ))
        return out

    def close(self):
        self._handle.remove()


@unittest.skipUnless(_COMFY_ROOT, "Set AISTUDIO_COMFY_TEST_ROOT to check the installed moss-image node")
@unittest.skipUnless((MOSS_NODE / "model.py").is_file(), "the moss-image ComfyUI node is not installed")
class InstalledMossImageSourceTests(unittest.TestCase):
    def test_model_py_is_the_file_the_notices_name(self):
        """``moss_spectrum.moss_head`` restates lines of this file (moss-image 0.1.1); a changed node fails here, not
        in the app's unit suite."""
        head = " ".join(NOTICES.read_text(encoding="utf-8").split("## MOSS-Image head (pack 1.9.0)", 1)[1].split())
        sha = re.search(r"`model.py`\s+SHA-256 `([0-9a-f]{64})`", head)
        self.assertIsNotNone(sha)
        self.assertEqual(hashlib.sha256((MOSS_NODE / "model.py").read_bytes()).hexdigest(), sha.group(1))


@requires_torch
@unittest.skipUnless(_COMFY_ROOT, "Set AISTUDIO_COMFY_TEST_ROOT to run MOSS Spectrum on the real ComfyUI runtime (CPU)")
@unittest.skipUnless((MOSS_NODE / "model.py").is_file(), "the moss-image ComfyUI node is not installed")
class MossSpectrumOnRealComfyTests(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        bind_torch(globals())
        import comfy.model_management as management

        if management.get_torch_device().type != "cpu":
            raise unittest.SkipTest("ComfyUI did not start on CPU - refusing to touch the GPU")
        import comfy.model_patcher
        import comfy.patcher_extension
        import comfy.sample
        import comfy.samplers

        from comfy_custom_nodes.ai_studio_forge_parity import (
            extra_samplers_dy_host,
            guidance_dd,
            moss_spectrum,
            moss_spectrum_nodes,
            speed_nodes,
        )

        cls.model_patcher, cls.comfy_sample, cls.samplers = comfy.model_patcher, comfy.sample, comfy.samplers
        cls.wrappers = comfy.patcher_extension.WrappersMP
        cls.ms, cls.node, cls.dd, cls.speed, cls.dy = (moss_spectrum, moss_spectrum_nodes.ForgeNeoMossSpectrum(),
                                                       guidance_dd, speed_nodes, extra_samplers_dy_host)
        cls.loaders = _moss_loaders()

    # -- a tiny real MOSS-Image model -----------------------------------------------------------------------

    def _patcher(self):
        config = self.loaders.MOSSImageConfig({
            "dim": 64, "num_heads": 4, "num_layers": 3, "ffn_dim": 96, "latent_channels": LATENT[0], "text_dim": 32,
            "freq_dim": 256, "rope_base": 10000.0,
        })
        config.set_inference_dtype(torch.float32, None)
        cpu = torch.device("cpu")
        base = config.get_model({}, device=cpu)
        self.assertEqual(type(base.diffusion_model).__name__, "MOSSImageTransformer")
        generator = torch.Generator().manual_seed(0)
        with torch.no_grad():
            for name, parameter in base.diffusion_model.named_parameters():
                if "norm" in name.split(".")[-2]:
                    parameter.copy_(1.0 + 0.1 * torch.randn(parameter.shape, generator=generator))
                else:
                    parameter.copy_(0.08 * torch.randn(parameter.shape, generator=generator))
        return self.model_patcher.ModelPatcher(base, load_device=cpu, offload_device=cpu)

    def _recorded(self, patcher):
        """A clone of ``patcher`` with the recorder outermost among APPLY_MODEL wrappers."""
        recorder = _Recorder(patcher.model.diffusion_model)
        self.addCleanup(recorder.close)
        clone = patcher.clone()
        clone.add_wrapper_with_key(self.wrappers.APPLY_MODEL, SPY_KEY, recorder.wrapper)
        return clone, recorder

    def _spectrum(self, model, **inputs):
        return self.node.patch(model, True, **inputs)[0]

    def _sigmas(self, model, steps=50, scheduler="simple"):
        return self.samplers.calculate_sigmas(model.get_model_object("model_sampling"), scheduler, steps).cpu()

    def _sample(self, model, sigmas, *, sampler="euler", batch=1, cfg=3.0, seed=7, negative_tokens=3):
        """A prompt of 5 tokens and a negative of 3 are two model calls per step (ComfyUI cannot batch their
        contexts); a 5-token negative is batched with the prompt into one call."""
        noise = torch.randn(batch, *LATENT, generator=torch.Generator().manual_seed(3))
        positive = [[torch.randn(batch, 5, 32, generator=torch.Generator().manual_seed(4)), {}]]
        negative = [[torch.randn(batch, negative_tokens, 32, generator=torch.Generator().manual_seed(5)), {}]]
        return self.comfy_sample.sample_custom(
            model, noise, cfg, self.samplers.sampler_object(sampler), sigmas, positive, negative,
            torch.zeros_like(noise), disable_pbar=True, seed=seed,
        )

    # -- the oracle -------------------------------------------------------------------------------------------

    @staticmethod
    def _sigma(call):
        return float((call.noted if call.noted is not None else call.args[1]).flatten()[0])

    @classmethod
    def _step(cls, call):
        grid = [float(value) for value in call.grid.tolist()]
        sigma = cls._sigma(call)
        matches = [index for index in range(len(grid) - 1) if grid[index] == sigma]
        return matches[0] if matches else None

    def _replay(self, base, call, features=None):
        """The unpatched model on the call's own input; ``features`` (B, H*W, dim) replace what enters
        ``final_norm_video`` (the model's own head on them). Returns (output, the features the model produced)."""
        norm = base.diffusion_model.final_norm_video
        produced = []

        def swap(module, inputs):
            produced.append(inputs[0].detach().clone())
            if features is not None:
                return (features[len(produced) - 1:len(produced)].to(inputs[0].dtype),)
            return None

        handle = norm.register_forward_pre_hook(swap)
        try:
            out = base._apply_model(*call.args, **call.kwargs)
        finally:
            handle.remove()
        return out, torch.cat(produced)

    def _check_run(self, base, recorder, settings):
        """Every call: ran the model -> the unpatched model's bits; forecast -> the model's head on the explicit
        forecast of each branch's history. Returns the steps whose calls were forecast."""
        history: dict = {}
        forecast_steps = set()
        grid = [float(value) for value in recorder.calls[0].grid.tolist()]
        steps = len(grid) - 1
        current, observed, start = -1, set(), None
        for number, call in enumerate(recorder.calls):
            step = self._step(call)
            if step is None and current < 0 and grid[1] < self._sigma(call) < grid[0]:
                start, step = self._sigma(call), 0                    # the sampler's offset first sigma: step 0
            elif step is None and current == 0 and self._sigma(call) == start:
                step = 0
            on_schedule = step is not None and step >= current      # a re-walk behind the step is not a step
            if on_schedule and step > current:
                current, observed = step, set()
            keys = list(zip(call.branches, call.uuids)) if call.uuids else list(call.branches)
            if call.ran:
                expected, produced = self._replay(base, call)
                with self.subTest(call=number, kind="model"):
                    self.assertTrue(torch.equal(call.out, expected))
                if on_schedule:                                       # one observation per branch and step
                    for key, chunk in zip(keys, produced.chunk(len(keys))):
                        if key not in observed:
                            observed.add(key)
                            history.setdefault(key, []).append((float(step), chunk))
                continue
            self.assertTrue(on_schedule)
            forecast_steps.add(step)
            feats = []
            for key in keys:
                ts = [t for t, _ in history[key]]
                hs = [h for _, h in history[key]]
                feats.append(explicit_forecast(torch, ts, hs, step, settings.cheby_degree, settings.ridge_lambda,
                                               settings.blend_w, steps).float())
            expected, _ = self._replay(base, call, torch.cat(feats))
            with self.subTest(call=number, kind="forecast", step=step):
                self.assertTrue(torch.allclose(call.out, expected, rtol=1e-5, atol=1e-5),
                                float((call.out - expected).abs().max()))
        return forecast_steps

    def _assert_released(self, patched, base):
        self.assertEqual(dict(base.diffusion_model.final_norm_video._forward_pre_hooks), {})
        self.assertNotIn(self.ms.RUN_KEY, patched.model_options.get("transformer_options", {}))

    # -- tests ----------------------------------------------------------------------------------------------

    def test_head_is_the_models_own(self):
        patcher = self._patcher()
        base = patcher.model
        dit = base.diffusion_model
        for batch in (1, 2):
            x = torch.randn(batch, *LATENT, generator=torch.Generator().manual_seed(8))
            sigma = torch.full((batch,), 0.62)
            context = torch.randn(batch, 5, 32, generator=torch.Generator().manual_seed(9))
            call = SimpleNamespace(args=(x, sigma, None, context, None, {}), kwargs={})
            expected, produced = self._replay(base, call)
            out = self.ms.moss_head(dit, base.model_sampling.timestep(sigma).float(), produced, LATENT[1], LATENT[2])
            denoised = base.model_sampling.calculate_denoised(sigma, out.float(), x)
            with self.subTest(batch=batch):
                if batch == 1:
                    self.assertTrue(torch.equal(denoised, expected))
                else:
                    self.assertTrue(torch.allclose(denoised, expected, rtol=1e-6, atol=1e-6))

    def test_fifty_euler_steps_follow_upstreams_schedule_and_the_explicit_forecast(self):
        patcher = self._patcher()
        sigmas = self._sigmas(patcher)
        plain_model, plain = self._recorded(patcher)
        plain_out = self._sample(plain_model, sigmas)
        for flex, count in ((0.25, 22), (0.5, 19), (1.0, 16)):
            with self.subTest(flex_window=flex):
                recorded, recorder = self._recorded(patcher)
                patched = self._spectrum(recorded, flex_window=flex)
                out = self._sample(patched, sigmas)
                settings = self.ms.SpectrumSettings(flex_window=flex)
                report = self.ms.model_spectrum_patch(patched).last_report
                planned = self.ms.planned_actual_steps(50, settings)
                self.assertEqual(len(planned), count)
                self.assertEqual(report.actual_steps(), planned)
                self.assertEqual((report.steps, report.evaluations, report.off_grid, report.passed, report.branches),
                                 (50, 100, 0, 0, 2))
                self.assertEqual({len(call.branches) for call in recorder.calls}, {1})   # cond, uncond: 2 calls
                self.assertEqual(sum(call.ran for call in recorder.calls), 2 * count)   # each step: cond, uncond
                forecast = self._check_run(patcher.model, recorder, settings)
                self.assertEqual(sorted(forecast), report.cached_steps())
                first = next(number for number, call in enumerate(recorder.calls) if not call.ran)
                self.assertEqual(first, 2 * min(forecast))
                for mine, theirs in zip(recorder.calls[:first], plain.calls[:first]):
                    self.assertTrue(torch.equal(mine.out, theirs.out))       # the same run until the first forecast
                self.assertFalse(torch.equal(out, plain_out))
                self.assertTrue(torch.isfinite(out).all())
                self._assert_released(patched, patcher.model)
        self.assertNotIn(self.ms.WRAPPER_KEY, patcher.wrappers.get(self.wrappers.APPLY_MODEL, {}))

    def test_nothing_to_forecast_is_the_plain_run(self):
        patcher = self._patcher()
        sigmas = self._sigmas(patcher, 12)
        plain = self._sample(patcher, sigmas)
        patched = self._spectrum(patcher, warmup_steps=12)
        self.assertTrue(torch.equal(self._sample(patched, sigmas), plain))
        self.assertEqual(self.ms.model_spectrum_patch(patched).last_report.actual_steps(), list(range(12)))
        self.assertIs(self.node.patch(patcher, False)[0], patcher)

    def test_every_run_starts_fresh_and_releases_its_tensors(self):
        patcher = self._patcher()
        patched = self._spectrum(patcher)
        refs = []
        original = self.ms.Forecaster.update

        def recording(forecaster, t, h):
            original(forecaster, t, h)
            refs.append(weakref.ref(forecaster.h[-1]))

        full = self._sigmas(patcher, 50)
        with mock.patch.object(self.ms.Forecaster, "update", recording):
            for sigmas in (full, full[25:], full[:13]):          # base, a denoise tail (hires), a 12-step run
                self._sample(patched, sigmas)
                report = self.ms.model_spectrum_patch(patched).last_report
                steps = len(sigmas) - 1
                with self.subTest(steps=steps):
                    self.assertEqual(report.steps, steps)
                    self.assertEqual(report.actual_steps(), self.ms.planned_actual_steps(steps))
                    gc.collect()
                    self.assertTrue(refs)
                    self.assertEqual([ref for ref in refs if ref() is not None], [])
                    self._assert_released(patched, patcher.model)
                refs.clear()
        for value in vars(report).values():
            self.assertNotIsInstance(value, torch.Tensor)

    def test_midpoint_calls_run_the_model_and_leave_the_schedule_alone(self):
        patcher = self._patcher()
        recorded, recorder = self._recorded(patcher)
        patched = self._spectrum(recorded)
        sigmas = self._sigmas(patcher, 20)
        self._sample(patched, sigmas, sampler="dpmpp_2s_ancestral")
        report = self.ms.model_spectrum_patch(patched).last_report
        midpoints = [call for call in recorder.calls if self._step(call) is None]
        self.assertGreater(len(midpoints), 10)
        self.assertTrue(all(call.ran for call in midpoints))
        self.assertEqual((report.off_grid, report.steps, sorted(report.step_modes)),
                         (len(midpoints), 20, list(range(20))))
        self.assertEqual(report.actual_steps(), self.ms.planned_actual_steps(20))
        self.assertEqual(sorted(self._check_run(patcher.model, recorder, self.ms.SpectrumSettings())),
                         report.cached_steps())

    def test_samplers_that_move_the_first_sigma_count_it_as_step_zero(self):
        """ComfyUI's offset_first_sigma_for_snr starts its SDE / exponential samplers (and the pack's flow ODE
        entries, which call dpmpp_2m_sde) at ``percent_to_sigma(1e-4)`` on a flow model: upstream counts those
        calls as step 0, so 50 steps run the model on the planned 19, with nothing off the grid."""
        patcher = self._patcher()
        sigmas = self._sigmas(patcher)
        for sampler in ("dpmpp_2m_sde", "er_sde", "dpmpp_2m_flow_ode"):
            with self.subTest(sampler=sampler):
                recorded, recorder = self._recorded(patcher)
                patched = self._spectrum(recorded)
                out = self._sample(patched, sigmas, sampler=sampler)
                report = self.ms.model_spectrum_patch(patched).last_report
                first = self._sigma(recorder.calls[0])
                self.assertTrue(float(sigmas[1]) < first < float(sigmas[0]), first)   # the sampler moved it
                self.assertEqual(len(report.actual_steps()), 19)
                self.assertEqual(report.actual_steps(), self.ms.planned_actual_steps(50))
                self.assertEqual((report.steps, report.evaluations, report.off_grid, report.passed,
                                  sorted(report.step_modes)), (50, 100, 0, 0, list(range(50))))
                self.assertEqual(sorted(self._check_run(patcher.model, recorder, self.ms.SpectrumSettings())),
                                 report.cached_steps())
                self.assertTrue(torch.isfinite(out).all())
                self._assert_released(patched, patcher.model)

    def test_second_order_and_restart_samplers(self):
        """Heun's corrector calls the next step's sigma (on the grid: it begins that step, whose own call follows
        without a second observation); the pack's restart_flow walks finished steps again (behind the step: the
        model, nothing recorded)."""
        patcher = self._patcher()
        for sampler, steps in (("heun", 16), ("restart_flow", 24)):
            with self.subTest(sampler=sampler):
                recorded, recorder = self._recorded(patcher)
                patched = self._spectrum(recorded)
                out = self._sample(patched, self._sigmas(patcher, steps), sampler=sampler)
                report = self.ms.model_spectrum_patch(patched).last_report
                self.assertEqual((report.steps, sorted(report.step_modes)), (steps, list(range(steps))))
                self.assertEqual(report.actual_steps(), self.ms.planned_actual_steps(steps))
                forecast = self._check_run(patcher.model, recorder, self.ms.SpectrumSettings())
                self.assertEqual(sorted(forecast), report.cached_steps())
                self.assertTrue(torch.isfinite(out).all())
                if sampler == "restart_flow":
                    self.assertGreater(report.off_grid, 0)        # the re-walked steps
                else:
                    self.assertEqual(report.off_grid, 0)
                    self.assertGreater(report.evaluations, 2 * steps)

    def test_detail_daemons_noted_sigma_drives_the_steps(self):
        patcher = self._patcher()
        settings = '{"dd_amount": 0.4, "dd_start": 0.0, "dd_end": 1.0, "dd_bias": 0.5, "dd_smooth": true}'
        recorded, recorder = self._recorded(patcher)
        patched = self._spectrum(self.dd.ForgeNeoAnimaDetailDaemon().patch(recorded, True, settings)[0])
        sigmas = self._sigmas(patcher, 30)
        self._sample(patched, sigmas)
        report = self.ms.model_spectrum_patch(patched).last_report
        scaled = [call for call in recorder.calls
                  if float(call.args[1].flatten()[0]) != float(call.noted.flatten()[0])]
        self.assertGreater(len(scaled), 10)                       # Detail Daemon did scale the model's sigma
        self.assertEqual((report.off_grid, report.steps), (0, 30))
        self.assertEqual(report.actual_steps(), self.ms.planned_actual_steps(30))
        self.assertEqual(sorted(self._check_run(patcher.model, recorder, self.ms.SpectrumSettings())),
                         report.cached_steps())

    def test_moss_pipeline_with_separate_cond_and_uncond_calls(self):
        from comfy_extras.nodes_model_advanced import ModelSamplingSD3
        from comfy_extras.nodes_slg import SkipLayerGuidanceDiTSimple

        patcher = self._patcher()
        shifted = ModelSamplingSD3().patch(patcher, 5.0)[0]
        recorded, recorder = self._recorded(shifted)
        slg = SkipLayerGuidanceDiTSimple.execute(recorded, 0.0, 1.0, double_layers="", single_layers="1").result[0]
        patched = self._spectrum(slg)
        sigmas = self._sigmas(shifted, 30, "sgm_uniform")
        self._sample(patched, sigmas)
        report = self.ms.model_spectrum_patch(patched).last_report
        self.assertEqual({len(call.branches) for call in recorder.calls}, {1})
        self.assertEqual((report.evaluations, report.branches, report.steps), (60, 2, 30))
        self.assertEqual(report.actual_steps(), self.ms.planned_actual_steps(30))
        self.assertEqual(sorted(self._check_run(shifted.model, recorder, self.ms.SpectrumSettings())),
                         report.cached_steps())

    def test_a_batch_is_forecast_in_batch_order(self):
        patcher = self._patcher()
        for batch in (1, 2):
            with self.subTest(batch=batch):
                self._batched_run(patcher, batch)

    def _batched_run(self, patcher, batch):
        recorded, recorder = self._recorded(patcher)
        patched = self._spectrum(recorded, flex_window=1.0)
        self._sample(patched, self._sigmas(patcher, 24), batch=batch, negative_tokens=5)
        self.assertEqual({call.args[0].shape[0] for call in recorder.calls}, {2 * batch})   # cond + uncond batched
        self.assertEqual({len(call.branches) for call in recorder.calls}, {2})
        report = self.ms.model_spectrum_patch(patched).last_report
        self.assertEqual((report.evaluations, report.branches), (24, 2))
        self.assertEqual(report.actual_steps(), self.ms.planned_actual_steps(24, self.ms.SpectrumSettings(
            flex_window=1.0)))
        self.assertEqual(sorted(self._check_run(patcher.model, recorder, self.ms.SpectrumSettings(flex_window=1.0))),
                         report.cached_steps())

    def test_speed_and_dy_see_the_wrapper(self):
        patcher = self._patcher()
        sigmas = self._sigmas(patcher, 16)
        patched = self._spectrum(patcher)
        seen = {}
        test = self

        def guard_probe(executor, *args, **kwargs):
            guider, run_sigmas, extra_args, noise = args[0], args[1], args[2], args[4]
            seen["speed"] = test.speed.comfy_runtime_guard(guider, noise, run_sigmas, extra_args)
            seen["dy"] = test.dy.blocking_reasons(SimpleNamespace(inner_model=guider), extra_args)
            return executor(*args, **kwargs)

        # the probes' modules say "spectrum" (a folder node's path, as ComfyUI names it): only the node's own
        # wrappers may count
        guard_probe.__module__ = r"C:\ComfyUI\custom_nodes\ComfyUI-Spectrum-KSampler.probe"
        probed = patched.clone()
        probed.add_wrapper_with_key(self.wrappers.SAMPLER_SAMPLE, SPY_KEY, guard_probe)
        without_speed = self._sample(probed, sigmas)
        self.assertEqual(seen, {"speed": self.speed.SPECTRUM_REASON, "dy": [self.dy.REASON_SPECTRUM]})
        speed_patch = self.speed.SpeedPatch(settings=self.speed.coerce_settings({"enabled": True}))
        with_speed = self.speed.apply_speed(patched, speed_patch)
        out = self._sample(with_speed, sigmas)
        status = self.speed._speed_wrappers(with_speed)[0]._ai_studio_speed_status.text
        self.assertEqual(status, f"skipped - {self.speed.SPECTRUM_REASON}")
        self.assertTrue(torch.equal(out, without_speed))
        plain_seen = {}

        def plain_probe(executor, *args, **kwargs):
            plain_seen["speed"] = test.speed.comfy_runtime_guard(args[0], args[4], args[1], args[2])
            plain_seen["dy"] = test.dy.blocking_reasons(SimpleNamespace(inner_model=args[0]), args[2])
            return executor(*args, **kwargs)

        plain_probe.__module__ = r"C:\ComfyUI\custom_nodes\ComfyUI-Spectrum-KSampler.probe"
        plain = patcher.clone()
        plain.add_wrapper_with_key(self.wrappers.SAMPLER_SAMPLE, SPY_KEY, plain_probe)
        self._sample(plain, sigmas)
        self.assertEqual(plain_seen, {"speed": None, "dy": []})

    def test_a_model_without_mosses_head_is_left_alone(self):
        import comfy.latent_formats
        import comfy.model_base
        import comfy.ops

        class OtherDiT(torch.nn.Module):
            def __init__(self, **kwargs):
                super().__init__()
                self.blocks = torch.nn.ModuleList([torch.nn.Linear(4, 4)])
                self.dtype = torch.float32

            def forward(self, x, t, context=None, **kwargs):
                return x * 0.5

        config = SimpleNamespace(
            unet_config={}, latent_format=comfy.latent_formats.LatentFormat(), manual_cast_dtype=None,
            custom_operations=comfy.ops.disable_weight_init, optimizations={},
            sampling_settings={"shift": 1.0, "multiplier": 1000}, memory_usage_factor=1.0,
        )
        cpu = torch.device("cpu")
        base = comfy.model_base.BaseModel(config, model_type=comfy.model_base.ModelType.FLOW, device=cpu,
                                          unet_model=OtherDiT)
        other = self.model_patcher.ModelPatcher(base, load_device=cpu, offload_device=cpu)
        with self.assertLogs("ai_studio_forge_parity", level="WARNING") as logs:
            self.assertIs(self._spectrum(other), other)
        self.assertEqual(len(logs.records), 1)
        self.assertEqual(other.wrappers.get(self.wrappers.APPLY_MODEL, {}), {})


if __name__ == "__main__":
    unittest.main()
