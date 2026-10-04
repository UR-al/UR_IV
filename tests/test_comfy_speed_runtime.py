r"""Pack 1.7.0 Anima SPEED on the real ComfyUI runtime (CPU) = sam-extra's ``run_speed`` (v0.33.0; same runner as 0.32.1).

A tiny flow-matching model built from ComfyUI's own ``BaseModel`` (``ModelType.FLOW`` -> ``CONST``
``ModelSamplingDiscreteFlow``, shift 3) with a deterministic stand-in DiT, wrapped in a real
``ModelPatcher`` and sampled through ``comfy.sample.sample_custom`` — the real ``CFGGuider``
(``process_conds``, cond/uncond batching, ``sample_sigmas``), ``KSAMPLER`` (flow noise scaling,
``KSamplerX0Inpaint``, callbacks) and ``WrapperExecutor`` — on ComfyUI's own k-diffusion samplers
(Euler, Euler ancestral -> ``_RF``, DPM++ 2M, DPM++ 2M SDE and DPM++ SDE GPU with real
``torchsde`` Brownian trees, Heun):

* ``ForgeNeoAnimaSpeed``'s wrapper == the installed extension's ``sam3ext.speed.runner.run_speed``
  run by a test wrapper in the same place with an equivalent host (the pack's ``ComfySpeedHost``
  hooks) and the extension's own plan — output bits, every DiT call (shape, timestep, the schedule
  it saw) and the status summary; both modes, DCT/FFT, batch 2, a 5-D latent, an img2img start.
* The run-wide noise stream is ComfyUI's ``default_noise_sampler`` (CPU ``seed + 1``); a txt2img
  Brownian tail tree is the tree a plain ComfyUI run builds.
* Guard reasons on real processed conds (``c_concat`` from ``noise_concat``, an area cond, an
  inpainting mask): SPEED steps aside and the output equals the plain run bit for bit.
* Detail Daemon + CNS + SPEED stacked on a real ``ModelPatcher``: wrapper order CNS -> SPEED -> DD,
  SPEED applied, Detail Daemon's scaled sigma at the hand-off uses the hand-off step's schedule
  value, the output is finite and full size; a Hires checkpoint override (``hires_carry``) keeps
  SPEED on a second real ``ModelPatcher``. ComfyUI's interrupt is a ``BaseException`` named as the
  wrapper expects.

Skipped unless ``AISTUDIO_COMFY_TEST_ROOT`` is set. Run it with the portable python, the repo (or
worktree) first on ``sys.path``, ``CUDA_VISIBLE_DEVICES=-1`` (ComfyUI is forced to ``--cpu`` before its
model management loads; no server, no GPU) and ``PYTHONDONTWRITEBYTECODE=1`` (nothing is written into
the ComfyUI install)::

    $env:CUDA_VISIBLE_DEVICES='-1'; $env:PYTHONDONTWRITEBYTECODE='1'
    $env:AISTUDIO_COMFY_TEST_ROOT='C:\ComfyUI_windows_portable\ComfyUI'; $env:AISTUDIO_REQUIRE_FORGE_EXT='1'
    C:\ComfyUI_windows_portable\python_embeded\python.exe -X utf8 -c "import sys, unittest; sys.path.insert(0, r'<repo>'); unittest.main(module=None, argv=['x', 'tests.test_comfy_speed_runtime', '-v'])"
"""
from __future__ import annotations

import importlib
import os
import sys
import unittest
from types import SimpleNamespace

_COMFY_ROOT = os.environ.get("AISTUDIO_COMFY_TEST_ROOT")
if _COMFY_ROOT:  # force --cpu before anything loads ComfyUI's model management
    if _COMFY_ROOT not in sys.path:
        sys.path.append(_COMFY_ROOT)
    from comfy.cli_args import args as _comfy_args  # noqa: E402

    _comfy_args.cpu = True

from tests._optional_deps import bind_torch, requires_torch  # noqa: E402
from tests._sam_extra_ext import EXT_ROOT, requires_extension  # noqa: E402

torch = None  # filled by bind_torch()


def _ext_speed(name: str):
    root = str(EXT_ROOT)
    if root not in sys.path:
        sys.path.append(root)
    return importlib.import_module(f"sam3ext.speed.{name}")


def _dit_class():
    class TinyDiT(torch.nn.Module):
        """A deterministic stand-in DiT on 4-D and 5-D latents; records what every call saw."""

        def __init__(self, in_channels=4, device=None, operations=None, dtype=None, **kwargs):
            super().__init__()
            generator = torch.Generator().manual_seed(5)
            self.mix = torch.nn.Parameter(torch.randn(in_channels, in_channels, generator=generator) * 0.1)
            self.dtype = torch.float32
            self.calls = []

        def forward(self, x, t, context=None, control=None, transformer_options={}, **kwargs):
            listed = transformer_options.get("sample_sigmas")
            self.calls.append((tuple(x.shape), t.detach().clone(), None if listed is None else listed.detach().clone()))
            view = (-1,) + (1,) * (x.ndim - 1)
            channels = self.mix.shape[0]
            latent, concat = x[:, :channels], x[:, channels:]      # c_concat rides on the channel axis
            mixed = torch.einsum("ij,bj...->bi...", self.mix, latent)
            smooth = (latent.roll(1, -1) + latent.roll(-1, -1) + latent.roll(1, -2) + latent.roll(-1, -2)) * 0.25
            ctx = context.mean(dim=(1, 2)).view(view) if context is not None else 0.0
            extra = concat.mean(dim=1, keepdim=True) * 0.01 if concat.shape[1] else 0.0
            return 0.6 * latent + 0.2 * smooth + 0.1 * mixed + 0.05 * ctx + 0.001 * t.view(view) + extra

    return TinyDiT


@requires_torch
@requires_extension
@unittest.skipUnless(_COMFY_ROOT, "Set AISTUDIO_COMFY_TEST_ROOT to run Anima SPEED on the real ComfyUI runtime (CPU)")
class SpeedOnRealComfyTests(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        bind_torch(globals())
        import comfy.model_management as management

        if management.get_torch_device().type != "cpu":
            raise unittest.SkipTest("ComfyUI did not start on CPU - refusing to touch the GPU")
        import comfy.k_diffusion.sampling as kd
        import comfy.latent_formats
        import comfy.model_base
        import comfy.model_patcher
        import comfy.ops
        import comfy.patcher_extension
        import comfy.sample
        import comfy.samplers

        from comfy_custom_nodes.ai_studio_forge_parity import guidance_cns, guidance_dd, speed_nodes, speed_schedule

        cls.management, cls.kd, cls.comfy_sample, cls.samplers = management, kd, comfy.sample, comfy.samplers
        cls.model_base, cls.model_patcher = comfy.model_base, comfy.model_patcher
        cls.latent_formats, cls.ops = comfy.latent_formats, comfy.ops
        cls.sampler_sample = comfy.patcher_extension.WrappersMP.SAMPLER_SAMPLE
        cls.pack, cls.schedule, cls.cns, cls.dd = speed_nodes, speed_schedule, guidance_cns, guidance_dd
        cls.ext_runner, cls.ext_fh = _ext_speed("runner"), _ext_speed("forge_host")
        cls.dit_class = _dit_class()

    # -- a tiny real model -----------------------------------------------------------------------------

    def _patcher(self, shift: float = 3.0):
        config = SimpleNamespace(
            unet_config={"in_channels": 4}, latent_format=self.latent_formats.LatentFormat(), manual_cast_dtype=None,
            custom_operations=self.ops.disable_weight_init, optimizations={},
            sampling_settings={"shift": shift, "multiplier": 1.0}, memory_usage_factor=1.0,
        )
        cpu = torch.device("cpu")
        base = self.model_base.BaseModel(config, model_type=self.model_base.ModelType.FLOW, device=cpu,
                                         unet_model=self.dit_class)
        return self.model_patcher.ModelPatcher(base, load_device=cpu, offload_device=cpu)

    def _sigmas(self, patcher, steps: int = 12, scheduler: str = "simple"):
        return self.samplers.calculate_sigmas(patcher.get_model_object("model_sampling"), scheduler, steps).cpu()

    @staticmethod
    def _conds(batch: int = 1):
        generator = torch.Generator().manual_seed(1)
        return ([[torch.randn(batch, 3, 8, generator=generator), {}]], [[torch.zeros(batch, 3, 8), {}]])

    def _sample(self, model, sampler_name, sigmas, noise, latent, *, seed=7, noise_mask=None, positive=None):
        default_positive, negative = self._conds(int(noise.shape[0]))
        steps = []
        out = self.comfy_sample.sample_custom(
            model, noise, 3.0, self.samplers.sampler_object(sampler_name), sigmas,
            default_positive if positive is None else positive, negative, latent, noise_mask=noise_mask,
            callback=lambda step, x0, x, total: steps.append((step, total)), disable_pbar=True, seed=seed,
        )
        calls = list(model.model.diffusion_model.calls)
        model.model.diffusion_model.calls.clear()
        return out, calls, steps

    # -- the oracle: the extension's run_speed in the same place, with the pack's host hooks ----------

    def _oracle(self, values: dict, label: str, image_seed: int, rescale: bool, record: dict):
        pack, ext_runner, ext_fh = self.pack, self.ext_runner, self.ext_fh
        pack_settings = pack.coerce_settings(values)
        ext_settings = ext_fh.coerce_settings([values])

        class HostHooks(ext_runner.SpeedHost):
            def __init__(self, inner):
                self.inner = inner

            def seeds(self, batch):
                return self.inner.seeds(batch)

            def noise_scope(self, full_shape):
                return self.inner.noise_scope(full_shape)

            def coarse_noise_sampler(self, template, x, stage, sigmas):
                return self.inner.coarse_noise_sampler(template, x, stage, sigmas)

            def publish_sigmas(self, sigmas):
                self.inner.publish_sigmas(sigmas)

            def restore_sigmas(self):
                self.inner.restore_sigmas()

            def log(self, message):
                self.inner.log(message)

        def wrapper(executor, *args, **kwargs):
            sampler = executor.class_obj
            original = sampler.sampler_function

            def ext_sample(model, x, sigmas, extra_args=None, callback=None, disable=None, **sampler_kwargs):
                plan = ext_fh._plan_for(ext_settings, x, sigmas)
                host = pack.ComfySpeedHost(
                    sampler_function=original, sigmas=sigmas, extra_args=extra_args, settings_seed=pack_settings.seed,
                    image_seed=image_seed, plan=pack.plan_for(pack_settings, x, sigmas),
                )
                run = ext_runner.SpeedRun(plan=plan, seeds=host.seeds(int(x.shape[0])),
                                          init_latent=label in ("hires", "img2img"), img2img_rescale=rescale)
                out, report = ext_runner.run_speed(
                    original, model, x, sigmas, run=run, host=HostHooks(host), extra_args=extra_args, callback=callback,
                    disable=disable, sampler_kwargs=host.sampler_kwargs(sampler_kwargs, x),
                )
                record["summary"] = report.summary(plan)
                return out

            sampler.sampler_function = ext_sample
            try:
                return executor(*args, **kwargs)
            finally:
                sampler.sampler_function = original

        return wrapper

    def _compare(self, sampler_name, values, *, shape=(1, 4, 16, 16), steps=12, label="base", rescale=True,
                 img2img=False, seed=7, image_seed=40):
        values = {"enabled": True, **values}
        pack_model = self.pack.ForgeNeoAnimaSpeed().patch(
            self._patcher(), pass_label=label, image_seed=image_seed, img2img_rescale=rescale,
            **{key: value for key, value in values.items() if key in self.pack.ARG_NAMES})[0]
        oracle_model = self._patcher().clone()
        record: dict = {}
        oracle_model.add_wrapper_with_key(self.sampler_sample, "oracle",
                                          self._oracle(values, label, image_seed, rescale, record))
        sigmas = self._sigmas(pack_model, steps)
        if img2img:
            sigmas = self._sigmas(pack_model, int(steps / 0.45))[-(steps + 1):]
        generator = torch.Generator().manual_seed(2)
        noise = torch.randn(shape, generator=generator)
        latent = torch.randn(shape, generator=generator) if img2img else torch.zeros(shape)
        got, got_calls, got_steps = self._sample(pack_model, sampler_name, sigmas, noise, latent, seed=seed)
        want, want_calls, want_steps = self._sample(oracle_model, sampler_name, sigmas, noise, latent, seed=seed)
        status = self._status(pack_model)
        self.assertEqual(status, f"applied ({values.get('mode', 'transition')}) - {record['summary']}")
        self.assertEqual((got.dtype, tuple(got.shape)), (want.dtype, tuple(want.shape)))
        self.assertTrue(torch.isfinite(got).all())
        self.assertTrue(torch.equal(got, want), f"max diff {(got.double() - want.double()).abs().max()}")
        self.assertEqual(len(got_calls), len(want_calls))
        for (g_shape, g_t, g_list), (w_shape, w_t, w_list) in zip(got_calls, want_calls):
            self.assertEqual(g_shape, w_shape)
            self.assertTrue(torch.equal(g_t, w_t))
            self.assertTrue(torch.equal(g_list, w_list))
        self.assertEqual(got_steps, want_steps)
        self.assertEqual([step for step, _ in got_steps], list(range(len(sigmas) - 1)))
        self.assertTrue(any(call[0][-2:] == (8, 8) for call in got_calls), "coarse calls reached the DiT")
        return status

    def _status(self, model):
        (wrapper,) = model.wrappers[self.sampler_sample][self.pack.SPEED_WRAPPER_KEY]
        return wrapper._ai_studio_speed_status.text

    # -- pack == extension ---------------------------------------------------------------------------------

    def test_samplers_and_modes_equal_the_extension(self):
        manual = {"threshold": "manual", "manual_sigmas": "0.8"}
        rows = [
            ("euler", dict(manual)),
            ("euler", dict(manual, mode="respace")),
            ("euler_ancestral", dict(manual)),
            ("euler_ancestral", dict(manual, mode="respace", transform="fft")),
            ("dpmpp_2m", dict(manual)),
            ("dpmpp_2m", dict(manual, mode="respace")),
            ("heun", dict(manual)),
            ("dpmpp_2m_sde", dict(manual)),
            ("dpmpp_2m_sde", dict(manual, mode="respace")),
            ("dpmpp_sde_gpu", dict(manual)),
            ("euler", dict(threshold="neo_shift", scales="0.25,0.5,1.0")),
            ("euler_ancestral", dict(manual, seed=123)),
        ]
        for sampler_name, values in rows:
            with self.subTest(sampler=sampler_name, values=values):
                self._compare(sampler_name, values)

    def test_batch_five_d_and_img2img(self):
        manual = {"threshold": "manual", "manual_sigmas": "0.8"}
        with self.subTest(case="batch 2"):
            self._compare("euler_ancestral", manual, shape=(2, 4, 16, 16))
        with self.subTest(case="5-D latent"):
            self._compare("euler", manual, shape=(1, 4, 1, 16, 16))
        for rescale in (True, False):
            with self.subTest(case="img2img", rescale=rescale):
                status = self._compare("euler_ancestral", {"threshold": "manual", "manual_sigmas": "0.6"},
                                       label="img2img", rescale=rescale, img2img=True)
                self.assertIn("init=rescaled" if rescale else "init=upstream", status)

    # -- ComfyUI's own noise ---------------------------------------------------------------------------

    def test_noise_stream_is_comfys_default_noise_sampler(self):
        patcher = self._patcher()
        sigmas = self._sigmas(patcher)
        for seed in (0, 7, 2 ** 40):
            x = torch.zeros(2, 4, 16, 16)
            host = self.pack.ComfySpeedHost(sampler_function=self.kd.sample_euler_ancestral, sigmas=sigmas,
                                            extra_args={"seed": seed})
            ours = host.sampler_kwargs({}, x)["noise_sampler"]
            comfy = self.kd.default_noise_sampler(x, seed=seed)
            for _ in range(3):
                self.assertTrue(torch.equal(ours(sigmas[0], sigmas[1]), comfy(sigmas[0], sigmas[1])))

    def test_txt2img_brownian_tail_is_the_plain_tree(self):
        patcher = self._patcher()
        sigmas = self._sigmas(patcher)
        plan = self.schedule.build_plan(mode="transition", transform="dct", sigmas=sigmas, full_grid=(16, 16),
                                        scales=(0.5, 1.0), threshold="manual", manual_sigmas=(0.8,))
        x = torch.zeros(1, 4, 16, 16)
        for name, cpu in (("sample_dpmpp_2m_sde", True), ("sample_dpmpp_3m_sde_gpu", False)):
            host = self.pack.ComfySpeedHost(sampler_function=getattr(self.kd, name), sigmas=sigmas,
                                            extra_args={"seed": 11}, plan=plan)
            ours = host.sampler_kwargs({}, x)["noise_sampler"]
            plain = self.kd.BrownianTreeNoiseSampler(x, sigmas[sigmas > 0].min(), sigmas.max(), seed=11, cpu=cpu)
            for i in range(len(sigmas) - 2):
                with self.subTest(sampler=name, step=i):
                    self.assertTrue(torch.equal(ours(sigmas[i], sigmas[i + 1]), plain(sigmas[i], sigmas[i + 1])))

    # -- guard reasons on real processed conds ---------------------------------------------------------

    def test_guard_on_real_conds_samples_plainly(self):
        positive, _negative = self._conds()
        concat = [[positive[0][0], {"noise_concat": torch.zeros(1, 2, 16, 16)}]]
        area = [[positive[0][0], {"area": (8, 8, 0, 0), "strength": 1.0}], positive[0]]
        mask = torch.ones(1, 1, 128, 128)
        rows = [
            ("c_concat", dict(positive=concat), self.pack.CONCAT_REASON),
            ("area", dict(positive=area), self.pack.AREA_REASON),
            ("mask", dict(noise_mask=mask), self.pack.MASK_REASON),
        ]
        for name, kwargs, reason in rows:
            with self.subTest(row=name):
                plain_model = self._patcher()
                sigmas = self._sigmas(plain_model)
                noise = torch.randn(1, 4, 16, 16, generator=torch.Generator().manual_seed(3))
                latent = torch.zeros(1, 4, 16, 16)
                plain, plain_calls, _ = self._sample(plain_model, "euler", sigmas, noise, latent, **kwargs)
                model = self.pack.ForgeNeoAnimaSpeed().patch(self._patcher(), enabled=True, threshold="manual",
                                                             manual_sigmas="0.8")[0]
                out, calls, _ = self._sample(model, "euler", sigmas, noise, latent, **kwargs)
                self.assertEqual(self._status(model), f"skipped - {reason}")
                self.assertTrue(torch.equal(out, plain))
                self.assertEqual([call[0] for call in calls], [call[0] for call in plain_calls])

    def test_interrupt_class(self):
        exc = self.management.InterruptProcessingException
        self.assertFalse(issubclass(exc, Exception))
        self.assertTrue(issubclass(exc, BaseException))
        self.assertTrue(self.pack._is_interrupt(exc()))

    # -- stacked with Detail Daemon and CNS; Hires override ------------------------------------------------

    def test_detail_daemon_cns_and_speed_stacked(self):
        dd_settings = '{"dd_amount": 0.4, "dd_start": 0.0, "dd_end": 1.0, "dd_bias": 0.5, "dd_smooth": true}'
        for sampler_name in ("euler", "euler_ancestral", "dpmpp_2m_sde"):
            with self.subTest(sampler=sampler_name):
                patcher = self._patcher()
                dd_model = self.dd.ForgeNeoAnimaDetailDaemon().patch(patcher, True, dd_settings)[0]
                speed_model = self.pack.ForgeNeoAnimaSpeed().patch(dd_model, enabled=True, threshold="manual",
                                                                   manual_sigmas="0.8", image_seed=40)[0]
                model = self.cns.apply_cns(speed_model)
                self.assertEqual(list(model.wrappers[self.sampler_sample]),
                                 [self.cns.CNS_WRAPPER_KEY, self.pack.SPEED_WRAPPER_KEY, self.dd.DD_WRAPPER_KEY])
                sigmas = self._sigmas(model)
                noise = torch.randn(1, 4, 16, 16, generator=torch.Generator().manual_seed(4))
                out, calls, _ = self._sample(model, sampler_name, sigmas, noise, torch.zeros(1, 4, 16, 16))
                self.assertTrue(self._status(model).startswith("applied (transition)"), self._status(model))
                self.assertEqual(tuple(out.shape), (1, 4, 16, 16))
                self.assertTrue(torch.isfinite(out).all())
                if sampler_name == "euler":
                    values = self.dd.detail_daemon_node_values(
                        {"dd_amount": 0.4, "dd_start": 0.0, "dd_end": 1.0, "dd_bias": 0.5, "dd_smooth": True}, 0.0)
                    schedule = torch.tensor(self.dd.detail_daemon_schedule(len(sigmas) - 1, values), dtype=torch.float32)
                    plan = self.pack.plan_for(self.pack.coerce_settings(
                        {"enabled": True, "threshold": "manual", "manual_sigmas": "0.8"}), out, sigmas)
                    step = plan.transitions[0].step
                    aligned = torch.tensor(plan.transitions[0].aligned, dtype=torch.float32)
                    first_full = next(call for call in calls if call[0][-1] == 16)
                    expected = aligned * max(1e-06, 1.0 - schedule[step].item() * 0.1 * 3.0)
                    self.assertTrue(torch.equal(first_full[1][:1], expected.reshape(1)))

    def test_hires_override_keeps_speed(self):
        source = self.pack.ForgeNeoAnimaSpeed().patch(self._patcher(), enabled=True, threshold="manual",
                                                      manual_sigmas="0.6", pass_label="hires", image_seed=40)[0]
        target = self._patcher()
        carried = self.pack._carry_to_override(source, target, None)
        self.assertIsNot(carried, target)
        self.assertEqual(target.wrappers.get(self.sampler_sample, {}), {})
        self.assertEqual(list(carried.wrappers[self.sampler_sample]), [self.pack.SPEED_WRAPPER_KEY])
        sigmas = self._sigmas(carried, int(12 / 0.45))[-13:]
        generator = torch.Generator().manual_seed(6)
        out, calls, _ = self._sample(carried, "euler", sigmas, torch.randn(1, 4, 16, 16, generator=generator),
                                     torch.randn(1, 4, 16, 16, generator=generator))
        self.assertTrue(self._status(carried).startswith("applied (transition)"), self._status(carried))
        self.assertIn("init=rescaled", self._status(carried))
        self.assertTrue(torch.isfinite(out).all())


if __name__ == "__main__":
    unittest.main()
