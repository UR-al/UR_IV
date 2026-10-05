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
  ``ModelPatcher``, prepared again for its latent format and with the override VAE;
* the GE comfy-3 case ``cc_adv`` (finding F2, investigated as no defect): the node the app's compiler makes for it,
  sampled with ``res_multistep`` on the run's real ``linear_quadratic`` 28 list (ComfyUI's list == the one Forge
  walked), equals the extension's hook on every evaluation with equal counters and the run's status line
  (``28 evals, 7 edited``); fed one content per step in float32 and bfloat16, too. With Forge's
  ``models/VAE/qwen_image_vae.safetensors`` present, its colour anchor is bit-equal through both hosts' real VAE
  paths — ComfyUI's ``comfy.sd.VAE`` + the pack, the extension's ``encode_anchors`` on that VAE, and Forge's own
  ``WanVAE`` + the extension in Forge's python (``tests/_forge_colorcraft_anchor.py``, a CPU child process) —
  in fp32 and in bf16 with SDPA attention (the GPU runs), whole encoded latent included; the case then also runs
  with that real anchor. Skipped with the reason when the file (or Forge's python) is missing.

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
import subprocess
import sys
import tempfile
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
from tests._sam_extra_ext import EXT_ROOT, FORGE_ROOT, requires_extension  # noqa: E402

torch = None  # filled by bind_torch()

# -- the GE comfy-3 case cc_adv (Anima 3.8B, 832x1216, res_multistep, linear_quadratic 28, CFG 5, seed 11) ----------
# Forge got this one compact Colorcraft argument (its API fills the rest with the panel's defaults) ...
CC_ADV_ARG = "v1;mods=I;I.exposure=0.3;I.contrast=0.4;I.temperature=0.5;I.chroma_center=0.2"
# ... and ComfyUI the ForgeNeoColorcraft node the app compiled from it (part 3 graph cc_adv__r1.json, node 9;
# settings_json is these settings with sorted keys and compact separators).
CC_ADV_SETTINGS = {
    "entries": [{
        "kind": "advanced", "mask": None,
        "params": {"blue": 0.0, "brightness": 0.0, "chroma_center": 0.2, "chroma_contrast": 0.0,
                   "chroma_plane": "temp_tint", "chroma_plane_override": False, "clarity": 0.0, "color_shift": True,
                   "color_shift_amount": 0.0, "contrast": 0.4, "exposure": 0.3, "green": 0.0, "lab_a": 0.0,
                   "lab_a_minus_b": 0.0, "lab_a_plus_b": 0.0, "lab_b": 0.0, "max_chroma": 2.5,
                   "max_chroma_override": False, "mode": "default", "more_colors": True, "recenter": 0.5,
                   "recenter_override": False, "red": 0.0, "saturation": 0.0, "sharpness": 0.0,
                   "temp_minus_tint": 0.0, "temp_plus_tint": 0.0, "temperature": 0.5, "tint": 0.0,
                   "tone_compression": 0.0, "vibrance": 0.0},
        "schedule": {"bias": 0.5, "end": 0.75, "end_off": 0.0, "exponent": 0.0, "smooth": True, "start": 0.5,
                     "start_off": 0.0, "strength": 1.0},
    }],
    "infotext": CC_ADV_ARG, "pass": "base", "tags": ["I"], "v": 1,
}
# ComfyUI's linear_quadratic 28 on Anima's flow sampling (shift 3) == the list Forge walked in that run.
LINEAR_QUADRATIC_28 = (
    1.0, 0.9982143044471741, 0.9964285492897034, 0.9946428537368774, 0.9928571581840515, 0.9910714030265808,
    0.9892857074737549, 0.987500011920929, 0.9857142567634583, 0.9839285612106323, 0.9821428656578064,
    0.9803571701049805, 0.9785714149475098, 0.9767857193946838, 0.9750000238418579, 0.968367338180542,
    0.9520407915115356, 0.9260203838348389, 0.8903061151504517, 0.844897985458374, 0.7897959351539612,
    0.7250000238418579, 0.6505101919174194, 0.5663265585899353, 0.4724489748477936, 0.3688775599002838,
    0.25561225414276123, 0.13265305757522583, 0.0,
)
CC_ADV_STATUS = "base: krea2 (Wan21); mods I; 28 evals, 7 edited"     # both hosts' line in that run
CC_ADV_EDITED = list(range(14, 21))     # sigma 0.975 ... 0.790: Start 0.5 / End 0.75 of 28 steps, value 1
CC_ADV_HW = (38, 26)                    # a quarter of the run's 152 x 104 latent grid (the parity holds there)
QWEN_IMAGE_VAE = "qwen_image_vae.safetensors"   # Forge's models/VAE: the VAE both hosts loaded in that run
ANCHOR_VARIANTS = (("fp32", "float32", False), ("bf16_sdpa", "bfloat16", True))   # = _forge_colorcraft_anchor
_FORGE_ANCHOR_HELPER = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_forge_colorcraft_anchor.py")


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

    def _sample(self, model, sampler_name, steps=10, *, five_d, seed=7, hw=(24, 32), sigmas=None, cfg=3.0):
        if sigmas is None:
            sigmas = self.samplers.calculate_sigmas(model.get_model_object("model_sampling"), "simple", steps).cpu()
        shape = (1, 16, 1, *hw) if five_d else (1, 16, *hw)
        noise = torch.randn(*shape, generator=torch.Generator().manual_seed(seed))
        generator = torch.Generator().manual_seed(1)
        positive = [[torch.randn(1, 3, 8, generator=generator), {}]]
        negative = [[torch.zeros(1, 3, 8), {}]]
        return self.comfy_sample.sample_custom(model, noise, cfg, self.samplers.sampler_object(sampler_name), sigmas,
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

    def _extension_pass(self, args, fmt, vae, full_hw, *, hires=False):
        """The extension's ``hook.process`` for one pass on a Forge-shaped stand-in ``p`` with ``fmt`` and ``vae``:
        ``(Run, its post-CFG callback)``."""
        p = SimpleNamespace(sd_model=SimpleNamespace(model_config=SimpleNamespace(latent_format=fmt),
                                                     forge_objects=SimpleNamespace(vae=vae,
                                                                                   unet=self.parity._FakeUnet())),
                            is_hr_pass=hires, extra_generation_params={}, sampler=None)
        with mock.patch("sys.stderr", new=io.StringIO()):
            run = self.hook.process(p, args, full_hw=tuple(full_hw))
        self.assertIsNotNone(run)
        return run, p.sd_model.forge_objects.unet.model_options["sampler_post_cfg_function"][-1]

    def _replay(self, log, config, fmt, *, pre_dd=True, hires=False, vae_seed=0, args=None):
        """The extension's callback on the recorded evaluations (an identical VAE); asserts every result equal.
        ``args``: the script arguments Forge got (default: ``config``'s infotext + the panel's defaults)."""
        vae = self.parity._FakeVAE(16, 3 if fmt.latent_dimensions == 3 else 2, seed=vae_seed)
        if args is None:
            args = [self.spec.to_infotext(config)] + self.panel.default_args()[1:]
        run, callback = self._extension_pass(args, fmt, vae, log[0]["x0"].shape[-2:], hires=hires)
        run.pre_dd = pre_dd
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
                with self.assertLogs("ai_studio_forge_parity", level="INFO") as logs:
                    self._sample(model, "euler", five_d=True)
                noted = [entry for entry in log if entry["note"] is not None]
                self.assertEqual(len(noted), len(log))
                self.assertTrue(any(not torch.equal(e["note"], e["sigma"]) for e in noted))  # DD scaled them
                patch = self.host.model_colorcraft_patch(model)
                self.assertEqual(patch.last_report.pre_dd > 0, pre_dd)
                run = self._replay(log, config, fmt, pre_dd=pre_dd)
                self._compare_reports(patch, run)
                # comfy-3 F1 (cc_dd): the console line == the extension's status line; Forge's separate infotext
                # key (written because Detail Daemon scaled the sigma) is the pack's second line when it was used.
                self.assertEqual(run.params.get(self.hook.INFOTEXT_PRE_DD), str(pre_dd))
                lines = [r.getMessage() for r in logs.records if r.getMessage().startswith(self.host.LOG_PREFIX)]
                expected = [f"{self.host.LOG_PREFIX} {run.status.render()}"]
                if pre_dd:
                    expected.append(f"{self.host.LOG_PREFIX} base pass: {self.hook.INFOTEXT_PRE_DD}: True "
                                    f"(x{patch.last_report.pre_dd})")
                self.assertEqual(lines, expected)
                self.assertNotIn("pre-DD", lines[0])

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

    # -- comfy-3 F2: the GE case cc_adv end to end ---------------------------------------------------------------

    def _cc_adv_inputs(self):
        """The app's compiler on the request's Colorcraft block -> the ForgeNeoColorcraft node's inputs."""
        from core import colorcraft as app_colorcraft
        from core.comfy_workflow_compiler import ComfyWorkflowCompiler, _Graph

        compiler = ComfyWorkflowCompiler()
        warnings = []
        config = compiler._colorcraft_plan({"alwayson_scripts": {app_colorcraft.SCRIPT_NAME: {"args": [CC_ADV_ARG]}}},
                                           warnings)
        graph = _Graph()
        node = compiler._add_colorcraft(graph, ["4", 0], ["3", 0], config, hires=False)
        self.assertEqual(warnings, [])
        inputs = graph.nodes[node[0]]["inputs"]
        self.assertEqual(inputs["settings_json"],
                         json.dumps(CC_ADV_SETTINGS, ensure_ascii=False, sort_keys=True, separators=(",", ":")))
        self.assertEqual((inputs["contract"], inputs["pre_dd_sigma"]), ("v1", True))
        return inputs

    def _cc_adv_model(self, vae):
        """``(sigmas, plain patcher, graded MODEL, its ColorcraftPatch)`` for cc_adv on a Wan21 (krea2) model."""
        fmt = self.latent_formats.Wan21()
        inputs = self._cc_adv_inputs()
        plain = self._patcher(fmt)
        sigmas = self.samplers.calculate_sigmas(plain.get_model_object("model_sampling"), "linear_quadratic", 28)
        sigmas = sigmas.cpu()
        self.assertTrue(torch.equal(sigmas, torch.tensor(LINEAR_QUADRATIC_28, dtype=torch.float32)))
        model = self.nodes.ForgeNeoColorcraft().patch(plain, vae, inputs["settings_json"], inputs["contract"],
                                                      inputs["pre_dd_sigma"])[0]
        return sigmas, plain, model, self.host.model_colorcraft_patch(model)

    def _cc_adv_args(self):
        return [CC_ADV_ARG] + self.panel.default_args()[1:]

    def _feed_both(self, patch, callback, sigmas, latent):
        """The investigation's glue check: one evaluation per step of ``sigmas`` at the list's sigma with the same
        ``latent`` - the pack's SAMPLER_SAMPLE wrapper + post-CFG function on ComfyUI's ``sample_sigmas``, the
        extension's callback on Forge's ``sampling_sigmas``; every output bit-equal, pass-throughs included.
        Returns the edited step indices."""
        transformer = {"sample_sigmas": sigmas}
        extra_args = {"model_options": {"transformer_options": transformer}, "seed": 11}
        edited = []

        def sampler(guider, sigmas_, extra_args_, callback_, noise_, *rest):
            for index in range(len(sigmas) - 1):
                sigma = sigmas[index:index + 1].clone()
                x_ext, x_pack = latent.clone(), latent.clone()
                out_ext = callback({"denoised": x_ext, "sigma": sigma,
                                    "model_options": {"transformer_options": {"sampling_sigmas": sigmas}}})
                out_pack = patch.post_cfg({"denoised": x_pack, "sigma": sigma,
                                           "model_options": extra_args_["model_options"]})
                where = f"{latent.dtype} step {index}"
                self.assertEqual(out_pack is x_pack, out_ext is x_ext, f"{where}: pass-through differs")
                self.assertEqual(out_pack.dtype, out_ext.dtype, where)
                self.assertTrue(torch.equal(out_pack, out_ext),
                                f"{where}: max |d| {(out_pack.float() - out_ext.float()).abs().max().item()}")
                if out_pack is not x_pack:
                    edited.append(index)
            return "samples"

        noise = torch.zeros(1, 16, 1, *latent.shape[-2:])
        self.assertEqual(patch.wrapper(self.parity._Executor(sampler), None, sigmas, extra_args, None, noise),
                         "samples")
        return edited

    def _assert_cc_adv_feeds(self, patch, vae, sigmas):
        fmt = self.latent_formats.Wan21()
        for dtype in (torch.float32, torch.bfloat16):
            with self.subTest(dtype=str(dtype)):
                run, callback = self._extension_pass(self._cc_adv_args(), fmt, vae, CC_ADV_HW)
                self.assertEqual(list(run.anchors), list(patch.prepared.anchors))
                for key, anchor in patch.prepared.anchors.items():
                    self.assertTrue(torch.equal(run.anchors[key], anchor), key)
                latent = (torch.randn(1, 16, 1, *CC_ADV_HW, generator=torch.Generator().manual_seed(11)) * 0.8)
                self.assertEqual(self._feed_both(patch, callback, sigmas, latent.to(dtype)), CC_ADV_EDITED)
                self._compare_reports(patch, run)
                self.assertEqual(patch.last_report.render(), CC_ADV_STATUS)
                self.assertEqual(run.status.render(), CC_ADV_STATUS)

    def test_cc_adv_end_to_end_equals_the_extension_on_the_real_sigma_list(self):
        fmt = self.latent_formats.Wan21()
        vae = self.parity._FakeVAE(16, 3)
        sigmas, plain, model, patch = self._cc_adv_model(vae)
        self.assertEqual((patch.prepared.family_label, patch.prepared.tags), ("krea2 (Wan21)", ["I"]))
        log = self._record(model)
        with self.assertLogs("ai_studio_forge_parity", level="INFO") as logs:
            graded = self._sample(model, "res_multistep", five_d=True, seed=11, hw=CC_ADV_HW, sigmas=sigmas, cfg=5.0)
        reference = self._sample(plain, "res_multistep", five_d=True, seed=11, hw=CC_ADV_HW, sigmas=sigmas, cfg=5.0)
        self.assertFalse(torch.equal(graded, reference))
        self.assertTrue(bool(torch.isfinite(graded).all()))
        self.assertEqual(len(log), 28)                                          # one evaluation per step
        self.assertEqual([index for index, entry in enumerate(log) if not entry["passed"]], CC_ADV_EDITED)
        self.assertTrue(all(torch.equal(entry["sigma"].flatten()[:1], sigmas[index:index + 1])
                            for index, entry in enumerate(log)))
        run = self._replay(log, None, fmt, args=self._cc_adv_args())          # every evaluation bit-equal
        self._compare_reports(patch, run)
        self.assertEqual(patch.last_report.render(), CC_ADV_STATUS)
        self.assertEqual(run.status.render(), CC_ADV_STATUS)
        lines = [r.getMessage() for r in logs.records if r.getMessage().startswith(self.host.LOG_PREFIX)]
        self.assertEqual(lines, [f"{self.host.LOG_PREFIX} {CC_ADV_STATUS}"])
        self._assert_cc_adv_feeds(patch, self.parity._FakeVAE(16, 3), sigmas)

    # -- comfy-3 F2: cc_adv's colour anchor through both hosts' real VAE paths ----------------------------------

    @classmethod
    def _qwen_vae_file(cls):
        return None if FORGE_ROOT is None else FORGE_ROOT / "models" / "VAE" / QWEN_IMAGE_VAE

    def _qwen_vae_state(self):
        """``(state dict, metadata)`` of Forge's qwen_image_vae.safetensors, read once; skips when it is missing."""
        path = self._qwen_vae_file()
        if path is None or not path.is_file():
            self.skipTest(f"{path or QWEN_IMAGE_VAE} not found - the cc_adv anchor check needs the Qwen-Image/Wan VAE "
                          "both hosts loaded in the GE run (Forge's models/VAE; Forge is found as for the oracle: "
                          "SAM3_FORGE_ROOT or two levels above the extension)")
        cached = getattr(type(self), "_qwen_state", None)
        if cached is None:
            import comfy.utils

            cached = comfy.utils.load_torch_file(str(path), return_metadata=True)
            type(self)._qwen_state = cached
        return cached

    def _comfy_vae(self, dtype, sdpa):
        """ComfyUI's VAELoader path (``comfy.sd.VAE`` from the file) in ``dtype``; ``sdpa``: the Wan attention
        blocks on PyTorch SDPA, as ComfyUI picks on CUDA (CPU picks its own)."""
        import comfy.ldm.wan.vae as wan_vae
        import comfy.sd
        from comfy.ldm.modules.diffusionmodules import model as attention

        state, metadata = self._qwen_vae_state()
        vae = comfy.sd.VAE(sd=dict(state), metadata=metadata, dtype=dtype)
        self.assertEqual(vae.vae_dtype, dtype)
        if sdpa:
            for module in vae.first_stage_model.modules():
                if isinstance(module, wan_vae.AttentionBlock):
                    module.optimized_attention = attention.pytorch_attention
        return vae

    def _forge_side_anchors(self):
        """Forge's own VAE path + the extension, in Forge's python (``tests/_forge_colorcraft_anchor.py``, CPU child
        process, run once): ``{variant: {"anchors", "latent", ...}, "_latent_format": {...}}``; skips with the
        reason when Forge's python or its Anima VAE config is missing."""
        cached = getattr(type(self), "_forge_anchor_result", None)
        if cached is not None:
            return cached
        python = FORGE_ROOT / "venv" / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
        config = FORGE_ROOT / "backend" / "huggingface" / "circlestone-labs" / "Anima" / "vae"
        missing = [str(path) for path in (python, config) if not path.exists()]
        if missing:
            self.skipTest(f"Forge's side of the cc_adv anchor check needs {', '.join(missing)} (Forge's venv python "
                          "and its Anima VAE config)")
        with tempfile.TemporaryDirectory() as folder:
            out = os.path.join(folder, "forge_anchor.pt")
            env = {key: value for key, value in os.environ.items() if key not in ("PYTHONPATH", "PYTHONHOME")}
            env.update(CUDA_VISIBLE_DEVICES="-1", PYTHONDONTWRITEBYTECODE="1", PYTHONIOENCODING="utf-8")
            done = subprocess.run(
                [str(python), "-B", "-X", "utf8", _FORGE_ANCHOR_HELPER, str(FORGE_ROOT), str(EXT_ROOT),
                 str(self._qwen_vae_file()), CC_ADV_ARG, out],
                cwd=folder, env=env, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=900,
            )
            if done.returncode != 0:
                self.fail(f"Forge's anchor helper exited {done.returncode}:\n{done.stderr[-4000:]}")
            summary = json.loads([line for line in done.stdout.splitlines() if line.startswith("{")][-1])
            data = torch.load(out, weights_only=True)
        self.assertTrue(summary["ok"])
        type(self)._forge_anchor_result = data
        return data

    def test_cc_adv_anchor_is_bit_equal_through_both_hosts_vae_paths(self):
        self._qwen_vae_state()                                  # skips first when the VAE file is missing
        forge = self._forge_side_anchors()
        fmt = self.latent_formats.Wan21()
        for part in ("mean", "std"):                            # the Wan21 process_in constants of both hosts
            ours = getattr(fmt, f"latents_{part}").flatten()
            self.assertTrue(torch.equal(ours, forge["_latent_format"][part].flatten().to(ours.dtype)), part)
        engine = self.parity._ext_module("sam3ext.colorcraft.engine")
        colors = engine.needed_colors(self.spec.build_chain(self.spec.config_from_args([CC_ADV_ARG]), False).entries)
        self.assertEqual([tuple(c) for c in colors], [(0.0, 0.0, 0.0, 0.0)])      # cc_adv: the neutral anchor only
        for name, dtype_name, sdpa in ANCHOR_VARIANTS:
            with self.subTest(variant=name):
                self.setUp()
                vae = self._comfy_vae(getattr(torch, dtype_name), sdpa)
                latents = []
                encode = vae.encode

                def keeping(image, _encode=encode):
                    latent = _encode(image)
                    latents.append(latent.detach().float().clone())
                    return latent

                vae.encode = keeping
                sigmas, _plain, _model, patch = self._cc_adv_model(vae)        # ForgeNeoColorcraft encodes now
                pack = patch.prepared.anchors
                self.assertEqual(len(latents), 1)
                ext = self.hook.encode_anchors(vae, fmt, colors, {}, device=torch.device("cpu"))
                theirs = forge[name]
                self.assertEqual(sorted(repr(tuple(key)) for key in pack), sorted(theirs["anchors"]))
                for key, anchor in pack.items():
                    self.assertEqual(anchor.dtype, torch.float32)
                    self.assertTrue(torch.equal(anchor, ext[key]), f"{name}: pack vs extension on ComfyUI's VAE")
                    forge_anchor = theirs["anchors"][repr(tuple(key))]
                    self.assertTrue(torch.equal(anchor, forge_anchor),
                                    f"{name}: pack vs Forge's VAE path, max |d| "
                                    f"{(anchor - forge_anchor).abs().max().item()}")
                self.assertTrue(torch.equal(latents[0], theirs["latent"]), f"{name}: whole encoded latent")
                if name == "bf16_sdpa":                 # the GPU runs' VAE: cc_adv end to end with the real anchor
                    self._assert_cc_adv_feeds(patch, vae, sigmas)
