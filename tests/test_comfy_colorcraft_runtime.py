r"""Pack 1.7.0 Colorcraft on the real ComfyUI runtime (CPU) = sam-extra v0.32.1's Forge hook.

A tiny flow-matching model built from ComfyUI's own ``BaseModel`` (``ModelType.FLOW``) with ComfyUI's ``Wan21``
(5-D, the Anima/krea2 family) or ``Flux`` (4-D, zimage) latent format and a deterministic stand-in DiT, wrapped in
a real ``ModelPatcher`` and sampled through ``comfy.sample.sample_custom`` — the real ``CFGGuider``
(``sample_sigmas``, the SAMPLER_SAMPLE wrapper chain, ``cfg_function``'s post-CFG list), ``KSAMPLER`` and ComfyUI's
k-diffusion samplers:

* ``ForgeNeoColorcraft``'s post-CFG function sees every model evaluation of the run; each one is replayed through
  the installed extension's post-CFG callback (``hook.process`` on a Forge-shaped stand-in with the same latent
  format object and an identical VAE) with the same prediction, sigma and sigma list: the graded results are
  equal bit for bit, pass-throughs included, and so are the status counters;
* Colorcraft's post-CFG function runs after a post-CFG function patched in before it (Forge: appended last);
* stacked on the pack's Detail Daemon, the schedule is looked up by the sampler's own sigma Detail Daemon noted
  (``pre_dd_sigma`` on), and by the scaled one with it off — both equal to the extension;
* a Hires checkpoint override (``hires_carry``) keeps the hires pass's Colorcraft on a second real
  ``ModelPatcher``, prepared again for its latent format and with the override VAE.

Skipped unless ``AISTUDIO_COMFY_TEST_ROOT`` is set. Run it with the portable python, the repo (or worktree) first on
``sys.path``, ``CUDA_VISIBLE_DEVICES=-1`` (ComfyUI is forced to ``--cpu`` before its model management loads; no
server, no GPU) and ``PYTHONDONTWRITEBYTECODE=1`` (nothing is written into the ComfyUI install)::

    $env:CUDA_VISIBLE_DEVICES='-1'; $env:PYTHONDONTWRITEBYTECODE='1'
    $env:AISTUDIO_COMFY_TEST_ROOT='C:\ComfyUI_windows_portable\ComfyUI'; $env:AISTUDIO_REQUIRE_FORGE_EXT='1'
    C:\ComfyUI_windows_portable\python_embeded\python.exe -X utf8 -c "import sys, unittest; sys.path.insert(0, r'<repo>'); unittest.main(module=None, argv=['x', 'tests.test_comfy_colorcraft_runtime', '-v'])"
"""
from __future__ import annotations

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


def _dit_class():
    class TinyDiT(torch.nn.Module):
        """A deterministic stand-in DiT on 4-D and 5-D latents."""

        def __init__(self, in_channels=16, device=None, operations=None, dtype=None, **kwargs):
            super().__init__()
            generator = torch.Generator().manual_seed(5)
            self.mix = torch.nn.Parameter(torch.randn(in_channels, in_channels, generator=generator) * 0.1)
            self.dtype = torch.float32

        def forward(self, x, t, context=None, control=None, transformer_options={}, **kwargs):
            view = (-1,) + (1,) * (x.ndim - 1)
            mixed = torch.einsum("ij,bj...->bi...", self.mix, x)
            smooth = (x.roll(1, -1) + x.roll(-1, -1) + x.roll(1, -2) + x.roll(-1, -2)) * 0.25
            ctx = context.mean(dim=(1, 2)).view(view) if context is not None else 0.0
            return 0.6 * x + 0.2 * smooth + 0.1 * mixed + 0.05 * ctx + 0.001 * t.view(view)

    return TinyDiT


@requires_torch
@requires_extension
@unittest.skipUnless(_COMFY_ROOT, "Set AISTUDIO_COMFY_TEST_ROOT to run Colorcraft on the real ComfyUI runtime (CPU)")
class ColorcraftOnRealComfyTests(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        bind_torch(globals())
        import comfy.model_management as management

        if management.get_torch_device().type != "cpu":
            raise unittest.SkipTest("ComfyUI did not start on CPU - refusing to touch the GPU")
        import comfy.latent_formats
        import comfy.model_base
        import comfy.model_patcher
        import comfy.ops
        import comfy.sample
        import comfy.samplers

        import tests.test_comfy_colorcraft_parity as parity
        from comfy_custom_nodes.ai_studio_forge_parity import (
            colorcraft_host,
            colorcraft_nodes,
            guidance_common,
            guidance_dd,
            hires_carry,
        )

        bind_torch(vars(parity))          # the shared stand-ins (fake VAE, configs) use that module's torch
        cls.parity = parity
        cls.comfy_sample, cls.samplers = comfy.sample, comfy.samplers
        cls.model_base, cls.model_patcher, cls.ops = comfy.model_base, comfy.model_patcher, comfy.ops
        cls.latent_formats = comfy.latent_formats
        cls.host, cls.nodes, cls.common, cls.dd, cls.carry = (colorcraft_host, colorcraft_nodes, guidance_common,
                                                              guidance_dd, hires_carry)
        cls.hook = parity._ext_module("sam3ext.colorcraft.hook")
        cls.spec = parity._ext_module("sam3ext.colorcraft.spec")
        cls.panel = parity._ext_module("sam3ext.colorcraft.panel_state")
        cls.dave = parity._ext_module("sam3ext.guidance.dave_gate")
        cls.dit_class = _dit_class()

    def setUp(self):
        self.host.ANCHORS.clear()
        self.hook.clear_anchor_cache()
        self.dave.note_pre_dd_sigma(None, None)
        self.addCleanup(self.dave.note_pre_dd_sigma, None, None)

    # -- a tiny real model ---------------------------------------------------------------------------------

    def _patcher(self, fmt):
        config = SimpleNamespace(
            unet_config={"in_channels": 16}, latent_format=fmt, manual_cast_dtype=None,
            custom_operations=self.ops.disable_weight_init, optimizations={},
            sampling_settings={"shift": 3.0, "multiplier": 1.0}, memory_usage_factor=1.0,
        )
        cpu = torch.device("cpu")
        base = self.model_base.BaseModel(config, model_type=self.model_base.ModelType.FLOW, device=cpu,
                                         unet_model=self.dit_class)
        return self.model_patcher.ModelPatcher(base, load_device=cpu, offload_device=cpu)

    def _sample(self, model, sampler_name, steps=10, *, five_d, seed=7, hw=(24, 32)):
        sigmas = self.samplers.calculate_sigmas(model.get_model_object("model_sampling"), "simple", steps).cpu()
        shape = (1, 16, 1, *hw) if five_d else (1, 16, *hw)
        noise = torch.randn(*shape, generator=torch.Generator().manual_seed(seed))
        generator = torch.Generator().manual_seed(1)
        positive = [[torch.randn(1, 3, 8, generator=generator), {}]]
        negative = [[torch.zeros(1, 3, 8), {}]]
        return self.comfy_sample.sample_custom(model, noise, 3.0, self.samplers.sampler_object(sampler_name), sigmas,
                                               positive, negative, torch.zeros(*shape), disable_pbar=True, seed=seed)

    # -- recording the pack's evaluations and replaying them through the extension --------------------------

    def _config(self):
        config = self.spec.default_config()
        config.enabled, config.masking = True, True
        config.modifiers[0].update({"active": True, "kind": "Advanced", "pass": "Both", "start": 0.0, "end": 0.9,
                                    "exposure": 0.6, "contrast": 0.8, "temperature": 1.1, "clarity": 1.2,
                                    "color_shift_amount": 0.5, "color_shift_red": 0.3, "mask": "M1"})
        config.modifiers[1].update({"active": True, "kind": "Punch", "pass": "Base", "start": 0.2, "end": 1.0,
                                    "contrast": -0.5, "sharpness": 0.9})
        config.leaves[0].update({"mask_axis": "exposure", "mask_mode": "highs", "blur": 12.0, "mask_hardness": 2.0})
        return config

    def _settings_json(self, config, hires=False):
        chain = self.spec.build_chain(config, hires)
        return json.dumps(self.parity._settings(*chain.entries, tags=chain.tags,
                                                pass_name="hires" if hires else "base",
                                                infotext=self.spec.to_infotext(config)))

    def _record(self, model):
        """Replace the patch's post-CFG function on ``model`` by a recorder that calls it; returns the log."""
        log = []
        functions = model.model_options["sampler_post_cfg_function"]
        index = next(i for i, fn in enumerate(functions) if self.host.is_owned(fn))
        original = functions[index]

        def recorder(args):
            transformer = args["model_options"]["transformer_options"]
            note = transformer.get(self.common.PRE_DD_SIGMAS_KEY)
            entry = {"x0": args["denoised"].clone(), "sigma": args["sigma"].clone(),
                     "sigmas": transformer["sample_sigmas"].detach().cpu().clone(),
                     "note": None if note is None else note.clone(), "order": len(log)}
            result = original(args)
            entry["passed"] = result is args["denoised"]
            entry["out"] = result.clone()
            log.append(entry)
            return result

        functions[index] = recorder
        return log

    def _replay(self, log, config, fmt, *, pre_dd=True, hires=False, vae_seed=0):
        """The extension's callback on the recorded evaluations (an identical VAE); asserts every result equal."""
        vae = self.parity._FakeVAE(16, 3 if fmt.latent_dimensions == 3 else 2, seed=vae_seed)
        p = SimpleNamespace(sd_model=SimpleNamespace(model_config=SimpleNamespace(latent_format=fmt),
                                                     forge_objects=SimpleNamespace(vae=vae,
                                                                                   unet=self.parity._FakeUnet())),
                            is_hr_pass=hires, extra_generation_params={}, sampler=None)
        args = [self.spec.to_infotext(config)] + self.panel.default_args()[1:]
        with mock.patch("sys.stderr", new=io.StringIO()):
            run = self.hook.process(p, args, full_hw=tuple(log[0]["x0"].shape[-2:]))
        run.pre_dd = pre_dd
        callback = p.sd_model.forge_objects.unet.model_options["sampler_post_cfg_function"][-1]
        published = log[0]["sigmas"].clone()
        for index, entry in enumerate(log):
            self.assertTrue(torch.equal(entry["sigmas"], published), "the run's sigma list never changed")
            if entry["note"] is not None:
                self.dave.note_pre_dd_sigma(entry["note"], entry["sigma"])
            else:
                self.dave.note_pre_dd_sigma(None, None)
            x0 = entry["x0"].clone()
            out = callback({"denoised": x0, "sigma": entry["sigma"],
                            "model_options": {"transformer_options": {"sampling_sigmas": published}}})
            self.assertEqual(out is x0, entry["passed"], f"eval {index}: pass-through differs")
            self.assertTrue(torch.equal(out, entry["out"]),
                            f"eval {index}: max |d| {(out.float() - entry['out'].float()).abs().max().item()}")
        return run

    def _compare_reports(self, patch, run):
        report, status = patch.last_report, run.status
        for name in ("evals", "edited", "idle", "foreign", "substeps", "errors", "nonfinite"):
            self.assertEqual(getattr(report, name), getattr(status, name), name)

    # -- the cases ---------------------------------------------------------------------------------------------

    def test_every_evaluation_equals_the_extension(self):
        for fmt_name, five_d in (("Wan21", True), ("Flux", False)):
            for sampler_name in ("euler", "heun", "dpmpp_2m"):
                with self.subTest(fmt=fmt_name, sampler=sampler_name):
                    self.setUp()
                    fmt = getattr(self.latent_formats, fmt_name)()
                    config = self._config()
                    plain = self._patcher(fmt)
                    vae = self.parity._FakeVAE(16, 3 if five_d else 2)
                    model = self.nodes.ForgeNeoColorcraft().patch(plain, vae, self._settings_json(config))[0]
                    self.assertIsNot(model, plain)
                    log = self._record(model)
                    graded = self._sample(model, sampler_name, five_d=five_d)
                    reference = self._sample(plain, sampler_name, five_d=five_d)
                    self.assertFalse(torch.equal(graded, reference))       # Colorcraft changed the image
                    self.assertGreaterEqual(len(log), 10)
                    patch = self.host.model_colorcraft_patch(model)
                    run = self._replay(log, config, fmt)
                    self._compare_reports(patch, run)
                    self.assertGreater(patch.last_report.edited, 5)

    def test_runs_after_an_earlier_post_cfg_function(self):
        fmt = self.latent_formats.Flux()
        calls = []

        def earlier(args):
            calls.append("earlier")
            return args["denoised"]

        plain = self._patcher(fmt)
        plain.set_model_sampler_post_cfg_function(earlier)
        model = self.nodes.ForgeNeoColorcraft().patch(plain, self.parity._FakeVAE(16, 2),
                                                      self._settings_json(self._config()))[0]
        functions = model.model_options["sampler_post_cfg_function"]
        self.assertIs(functions[0], earlier)
        self.assertTrue(self.host.is_owned(functions[-1]))
        log = self._record(model)
        original = functions[-1]

        def ordered(args):
            calls.append("colorcraft")
            return original(args)

        functions[-1] = ordered
        self._sample(model, "euler", five_d=False)
        self.assertEqual(calls[:4], ["earlier", "colorcraft"] * 2)
        self.assertEqual(len(log), 10)

    def test_stacked_on_detail_daemon_follows_its_note(self):
        dd_settings = '{"dd_amount": 0.6, "dd_start": 0.0, "dd_end": 1.0, "dd_bias": 0.5, "dd_smooth": true}'
        for pre_dd in (True, False):
            with self.subTest(pre_dd=pre_dd):
                self.setUp()
                fmt = self.latent_formats.Wan21()
                config = self._config()
                dd_model = self.dd.ForgeNeoAnimaDetailDaemon().patch(self._patcher(fmt), True, dd_settings)[0]
                model = self.nodes.ForgeNeoColorcraft().patch(dd_model, self.parity._FakeVAE(16, 3),
                                                              self._settings_json(config), "v1", pre_dd)[0]
                log = self._record(model)
                self._sample(model, "euler", five_d=True)
                noted = [entry for entry in log if entry["note"] is not None]
                self.assertEqual(len(noted), len(log))
                self.assertTrue(any(not torch.equal(e["note"], e["sigma"]) for e in noted))  # DD scaled them
                patch = self.host.model_colorcraft_patch(model)
                self.assertEqual(patch.last_report.pre_dd > 0, pre_dd)
                run = self._replay(log, config, fmt, pre_dd=pre_dd)
                self._compare_reports(patch, run)

    def test_hires_override_keeps_colorcraft(self):
        config = self._config()
        base_vae, override_vae = self.parity._FakeVAE(16, 3), self.parity._FakeVAE(16, 2, seed=9)
        source = self.nodes.ForgeNeoColorcraft().patch(self._patcher(self.latent_formats.Wan21()), base_vae,
                                                       self._settings_json(config, hires=True))[0]
        self.assertIn(self.nodes.CARRY_KEY, self.carry.registered())
        target = self._patcher(self.latent_formats.Flux())
        before = base_vae.calls
        carried = self.carry.apply(source, target, override_vae)
        self.assertIsNot(carried, target)
        self.assertEqual(base_vae.calls, before)
        self.assertGreater(override_vae.calls, 0)          # anchors encoded again with the override VAE
        patch = self.host.model_colorcraft_patch(carried)
        self.assertEqual((patch.prepared.family, patch.prepared.pass_name), ("zimage", "hires"))
        log = self._record(carried)
        self._sample(carried, "euler", five_d=False)
        run = self._replay(log, config, self.latent_formats.Flux(), hires=True, vae_seed=9)
        self._compare_reports(patch, run)
        self.assertGreater(patch.last_report.edited, 5)

    def test_hires_vae_only_override_grades_with_the_override_vae(self):
        # Forge reloads the model with the Hires VAE (hr_additional_modules) and the hook encodes the colour
        # anchors with it; ForgeNeoHiresFix with only vae_name overridden must grade like that (Shift 0 and 3).
        from comfy_extras.nodes_model_advanced import ModelSamplingSD3

        from comfy_custom_nodes.ai_studio_forge_parity import generation

        config = self._config()
        fmt = self.latent_formats.Flux()
        for shift in (0.0, 3.0):
            with self.subTest(shift=shift):
                self.setUp()
                base_vae, override_vae = self.parity._FakeVAE(16, 2), self.parity._FakeVAE(16, 2, seed=9)
                source = self.nodes.ForgeNeoColorcraft().patch(self._patcher(fmt), base_vae,
                                                               self._settings_json(config, hires=True))[0]
                captured = []

                def provider(name, *, method, feature, args=(), kwargs=None):
                    if name == "VAELoader":
                        return (override_vae,)
                    if name == "ModelSamplingSD3":
                        return ModelSamplingSD3().patch(*args)
                    raise AssertionError(f"unexpected provider {name}")

                with (
                    mock.patch.object(generation, "invoke_provider", side_effect=provider),
                    mock.patch.object(generation, "_common_sample",
                                      side_effect=lambda m, *a, **k: (captured.append(m) or {"samples": a[7]},)),
                ):
                    _output, vae = generation.ForgeNeoHiresFix().run(
                        source, "positive", "negative", {"samples": torch.zeros(1, 16, 8, 8)}, seed=3, enabled=True,
                        steps=4, shift=shift, upscale_method="latent:nearest-exact", base_vae=base_vae,
                        vae_name="other_vae.safetensors",
                    )
                (sampled,) = captured
                self.assertIs(vae, override_vae)
                self.assertGreater(override_vae.calls, 0)          # anchors encoded again with the override VAE
                patch = self.host.model_colorcraft_patch(sampled)
                self.assertIsNot(patch, self.host.model_colorcraft_patch(source))
                log = self._record(sampled)
                self._sample(sampled, "euler", five_d=False)
                run = self._replay(log, config, fmt, hires=True, vae_seed=9)
                self._compare_reports(patch, run)
                self.assertGreater(patch.last_report.edited, 5)

    def test_nothing_to_grade_returns_the_model_itself(self):
        plain = self._patcher(self.latent_formats.Flux())
        empty = json.dumps(self.parity._settings())
        self.assertIs(self.nodes.ForgeNeoColorcraft().patch(plain, None, empty)[0], plain)
