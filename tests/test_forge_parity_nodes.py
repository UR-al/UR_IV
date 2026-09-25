from __future__ import annotations

import json
from types import SimpleNamespace
import unittest
from unittest import mock

from comfy_custom_nodes.ai_studio_forge_parity import compat
from comfy_custom_nodes.ai_studio_forge_parity import generation
from comfy_custom_nodes.ai_studio_forge_parity import guidance
from comfy_custom_nodes.ai_studio_forge_parity import guidance_dave
from tests._optional_deps import load_torch, requires_torch


class _FakeModel:
    def __init__(self):
        self.model_options = {}

    def clone(self):
        clone = _FakeModel()
        clone.model_options = dict(self.model_options)
        return clone


class TestForgeParityContracts(unittest.TestCase):
    def test_required_node_mappings_are_exported_by_modules(self):
        expected_guidance = {
            "ForgeNeoNegPip", "ForgeNeoAnimaDAVE", "ForgeNeoAnimaModGuidance",
            "ForgeNeoSkimmedCFG", "ForgeNeoAnimaSafePAG", "ForgeNeoDCWCWMSMC",
            "ForgeNeoAnimaGuidanceSuite", "ForgeNeoAnimaDetailDaemon",
        }
        expected_generation = {
            "ForgeNeoKSamplerCNS", "ForgeNeoCNSSamplerPatch", "ForgeNeoModelSamplingShift",
            "ForgeNeoLatentInput", "ForgeNeoHiresFix",
            "ForgeNeoMaskSelector", "ForgeNeoLoraBlockWeight",
            "ForgeNeoCharacterReference", "ForgeNeoReferencePrompt",
            "ForgeNeoReferenceOutput", "ForgeNeoAnimaPiD", "ForgeNeoAnimaVAE2x",
            "ForgeNeoADetailer", "ForgeNeoSaveImage",
        }
        self.assertEqual(set(guidance.NODE_CLASS_MAPPINGS), expected_guidance)
        self.assertEqual(set(generation.NODE_CLASS_MAPPINGS), expected_generation)

    def test_compiler_facing_suite_and_daemon_contracts(self):
        suite = guidance.ForgeNeoAnimaGuidanceSuite
        daemon = guidance.ForgeNeoAnimaDetailDaemon
        self.assertEqual(
            tuple(suite.INPUT_TYPES()["required"]),
            ("model", "clip", "positive", "negative", "enabled", "settings_json"),
        )
        self.assertEqual(suite.RETURN_TYPES, ("MODEL",))
        self.assertEqual(
            tuple(daemon.INPUT_TYPES()["required"]),
            ("model", "enabled", "settings_json"),
        )
        self.assertEqual(daemon.RETURN_TYPES, ("MODEL",))

    def test_adetailer_contract_matches_workflow_compiler(self):
        node = generation.ForgeNeoADetailer
        self.assertEqual(
            tuple(node.INPUT_TYPES()["required"]),
            ("image", "model", "clip", "vae", "positive", "negative", "enabled", "settings_json"),
        )
        self.assertEqual(node.RETURN_TYPES, ("IMAGE", "MASK", "STRING"))
        self.assertEqual(node.RETURN_NAMES, ("image", "mask", "report"))

    def test_save_image_contract_keeps_all_forge_artifact_switches(self):
        inputs = generation.ForgeNeoSaveImage.INPUT_TYPES()
        self.assertEqual(
            tuple(inputs["required"]),
            (
                "images", "filename_prefix", "file_format", "metadata_mode",
                "quality", "webp_lossless", "collision_mode", "save_before_hires",
                "save_img2img_source", "save_inpaint_mask", "save_inpaint_composite",
            ),
        )
        self.assertEqual(
            tuple(inputs["optional"]),
            ("before_hires_images", "img2img_source", "inpaint_source", "inpaint_mask"),
        )


class TestForgeParityPureBehavior(unittest.TestCase):
    def test_beta57_scheduler_registration_pins_res4lyf_parameters(self):
        beta_scheduler = mock.Mock(return_value="sigmas")
        scheduler_names = ["simple", "beta"]

        def handler(function):
            return SimpleNamespace(handler=function, use_ms=True)

        fake_samplers = SimpleNamespace(
            SCHEDULER_HANDLERS={"simple": handler(object())},
            SCHEDULER_NAMES=scheduler_names,
            SchedulerHandler=handler,
            KSampler=SimpleNamespace(SCHEDULERS=scheduler_names),
            beta_scheduler=beta_scheduler,
        )
        with mock.patch.object(
            compat.importlib, "import_module", return_value=fake_samplers,
        ):
            self.assertTrue(compat.install_forge_scheduler_support(required=True))
            self.assertTrue(compat.install_forge_scheduler_support(required=True))

        self.assertEqual(scheduler_names.count("beta57"), 1)
        registered = fake_samplers.SCHEDULER_HANDLERS["beta57"].handler
        self.assertEqual(registered("model-sampling", 32), "sigmas")
        beta_scheduler.assert_called_once_with(
            "model-sampling", 32, alpha=0.5, beta=0.7,
        )

    def test_beta57_registration_never_removes_an_existing_provider(self):
        existing = object()
        fake_samplers = SimpleNamespace(
            SCHEDULER_HANDLERS={"beta57": existing},
            SchedulerHandler=lambda function: function,
            beta_scheduler=mock.Mock(),
            KSampler=SimpleNamespace(SCHEDULERS=()),
        )
        with mock.patch.object(
            compat.importlib, "import_module", return_value=fake_samplers,
        ):
            self.assertFalse(
                compat.install_forge_scheduler_support(required=False)
            )
        self.assertIs(fake_samplers.SCHEDULER_HANDLERS["beta57"], existing)

    def test_sampling_shift_preserves_anima_timestep_multiplier(self):
        model = mock.Mock()
        model.get_model_object.return_value = SimpleNamespace(multiplier=1.0)
        with mock.patch.object(
            generation, "invoke_provider", return_value=("patched",),
        ) as invoke:
            result = generation.ForgeNeoModelSamplingShift().patch(model, 3.0)

        self.assertEqual(result, ("patched",))
        invoke.assert_called_once_with(
            "ModelSamplingSD3",
            method="patch",
            feature="Forge flow shift",
            args=(model, 3.0, 1.0),
        )

    def test_parse_indices_supports_ranges_and_clamps(self):
        self.assertEqual(guidance.parse_indices("1, 3-5, 99, bad", 7), {1, 3, 4, 5})
        self.assertEqual(guidance.parse_indices("", 20, default="8-10"), {8, 9, 10})

    def test_detail_daemon_schedule_has_offsets_peak_and_fade(self):
        # 원본 노드 스케줄(스텝 단위) — 원본 대조는 tests/test_comfy_detail_daemon.py
        values = dict(
            start=0.2, end=0.8, bias=0.5, detail_amount=0.25, exponent=1.0,
            start_offset=0.02, end_offset=-0.01, fade=0.2, smooth=True,
        )
        schedule = guidance.detail_daemon_schedule(21, values)
        self.assertAlmostEqual(schedule[0], 0.016)
        self.assertAlmostEqual(schedule[10], 0.20)
        self.assertAlmostEqual(schedule[20], -0.008)

    def test_reference_box_and_prompt_are_deterministic(self):
        self.assertEqual(
            generation.reference_target_box(100, 80, "reference_left", 0.4, 2),
            (42, 0, 100, 80),
        )
        self.assertEqual(
            generation.compose_reference_prompt("one girl", True, "(split screen:1.2)"),
            "(split screen:1.2), one girl",
        )
        self.assertEqual(generation.compose_reference_prompt("one girl", False, "x"), "one girl")

    def test_adetailer_raw_slot_maps_to_impact_contract(self):
        resolved = generation.normalize_adetailer_settings({
            "ad_use_sampler": True,
            "ad_sampler": "DPM++ 2M Karras",
            "ad_scheduler": "Use same scheduler",
            "ad_use_inpaint_width_height": True,
            "ad_inpaint_width": 640,
            "ad_inpaint_height": 768,
            "ad_mask_blur": 7,
            "ad_denoising_strength": 0.31,
            "ad_confidence": 0.42,
            "ad_dilate_erode": -3,
            "ad_prompt": "detailed eyes",
        })
        self.assertEqual(resolved["sampler_name"], "dpmpp_2m")
        self.assertEqual(resolved["scheduler"], "karras")
        self.assertEqual(resolved["guide_size"], 768)
        self.assertEqual(resolved["max_size"], 768)
        self.assertEqual(resolved["feather"], 7)
        self.assertEqual(resolved["noise_mask_feather"], 7)
        self.assertEqual(resolved["denoise"], 0.31)
        self.assertEqual(resolved["bbox_threshold"], 0.42)
        self.assertEqual(resolved["bbox_dilation"], -3)
        self.assertEqual(resolved["wildcard"], "detailed eyes")

    def test_adetailer_drops_impact_bbox_no_segm_sentinel(self):
        class NoSegmDetector:
            pass

        class FakeMask:
            def __gt__(self, _other):
                return self

            def any(self, **_kwargs):
                return self

            def sum(self):
                return self

            def item(self):
                return 0

        image = mock.Mock(shape=(1, 8, 8, 3), dtype="float32", device="cpu")
        captured = {}

        def invoke(node_type, **kwargs):
            if node_type == "UltralyticsDetectorProvider":
                return object(), NoSegmDetector()
            if node_type == "FaceDetailer":
                captured.update(kwargs["kwargs"])
                return image, None, None, FakeMask()
            raise AssertionError(node_type)

        with (
            mock.patch.object(generation, "require_torch", return_value=object()),
            mock.patch.object(generation, "_image_tensor", return_value=image),
            mock.patch.object(generation, "invoke_provider", side_effect=invoke),
        ):
            generation.ForgeNeoADetailer().detail(
                image, object(), object(), object(), object(), object(), True,
                '{"ad_model":"face_yolov8n.pt"}',
            )

        self.assertIsNone(captured["segm_detector_opt"])

    def test_adetailer_fails_fast_for_unrepresentable_enabled_options(self):
        for settings, label in (
            ({"ad_x_offset": 4}, "offset"),
            ({"ad_mask_filter_method": "Confidence"}, "filter"),
            ({"ad_mask_merge_invert": "Merge"}, "merge"),
            ({"ad_use_checkpoint": True, "ad_checkpoint": "other.safetensors"}, "checkpoint"),
            ({"ad_use_vae": True, "ad_vae": "other.safetensors"}, "VAE"),
            ({"ad_use_clip_skip": True}, "CLIP"),
            ({"ad_use_noise_multiplier": True}, "noise"),
            ({"ad_restore_face": True}, "restore"),
            ({"ad_controlnet_model": "control.safetensors"}, "ControlNet"),
        ):
            with self.subTest(label=label):
                with self.assertRaisesRegex(RuntimeError, label):
                    generation.normalize_adetailer_settings(settings)

    def test_webui_plms_sampler_is_explicitly_rejected(self):
        with self.assertRaisesRegex(RuntimeError, "PLMS"):
            generation.normalize_adetailer_settings({
                "ad_use_sampler": True, "ad_sampler": "PLMS",
            })


class TestForgeParityDisabledPaths(unittest.TestCase):
    def test_disabled_guidance_paths_are_identity_and_do_not_parse_json(self):
        model, clip = object(), object()
        self.assertEqual(guidance.ForgeNeoNegPip().patch(model, clip, False), (model, clip))
        self.assertIs(guidance.ForgeNeoAnimaDAVE().patch(model, False)[0], model)
        self.assertIs(guidance.ForgeNeoSkimmedCFG().patch(model, False)[0], model)
        self.assertIs(guidance.ForgeNeoAnimaSafePAG().patch(model, False)[0], model)
        self.assertIs(guidance.ForgeNeoDCWCWMSMC().patch(model)[0], model)
        self.assertIs(
            guidance.ForgeNeoAnimaGuidanceSuite().patch(
                model, clip, object(), object(), False, "not json"
            )[0],
            model,
        )
        self.assertIs(
            guidance.ForgeNeoAnimaDetailDaemon().patch(model, False, "not json")[0],
            model,
        )

    def test_disabled_generation_patchers_are_identity(self):
        model, clip, positive, negative, vae, latent, image = (object() for _ in range(7))
        self.assertEqual(
            generation.ForgeNeoLoraBlockWeight().load(model, clip, False, block_vector="1,0"),
            (model, clip, "1,0"),
        )
        self.assertEqual(
            generation.ForgeNeoCharacterReference().patch(
                model, positive, negative, vae, False
            ),
            (model, positive, negative),
        )
        self.assertIs(generation.ForgeNeoReferenceOutput().crop(image, False)[0], image)
        self.assertIs(generation.ForgeNeoAnimaPiD().decode(image, latent, False)[0], image)
        self.assertIs(generation.ForgeNeoAnimaVAE2x().patch(vae, False)[0], vae)
        self.assertEqual(generation.ForgeNeoHiresFix().run(
            model, positive, negative, latent, enabled=False, base_vae=vae
        ), (latent, vae))

    def test_enabled_missing_provider_is_never_silently_ignored(self):
        error = compat.MissingComfyProvider("CLIPNegPip missing")
        with mock.patch.object(guidance, "invoke_provider", side_effect=error):
            with self.assertRaisesRegex(RuntimeError, "missing"):
                guidance.ForgeNeoNegPip().patch(object(), object(), True)

    def test_unavailable_reference_and_pid_options_fail_before_provider_use(self):
        with self.assertRaisesRegex(RuntimeError, "temporal_mask"):
            generation.ForgeNeoCharacterReference().patch(
                object(), object(), object(), object(), True,
                reference_method="temporal_mask", reference_image=object(),
            )
        with self.assertRaisesRegex(RuntimeError, "dtype override"):
            generation.ForgeNeoAnimaPiD().decode(
                object(), object(), True, ckpt_name="pid.safetensors", dtype="fp32",
            )
        with self.assertRaisesRegex(RuntimeError, "tile_latent"):
            generation.ForgeNeoAnimaPiD().decode(
                object(), object(), True, ckpt_name="pid.safetensors", tile_latent=96,
            )

    def test_suite_rejects_unsupported_seg_modes_before_runtime(self):
        # PAG 의 legacy·헤드 지정은 원본 노드가 받는다(tests/test_forge_parity_pag_origin.py 가
        # 스위트로 원본과 비트 단위 대조). SEG 는 팩 구현이 둘 다 없어 모델을 만지기 전에 막는다.
        for extra, message in (
            ({"guid_head_indices": "0,2"}, "Head-selective SEG"),
            ({"guid_legacy_attn": True}, "Legacy SEG"),
        ):
            payload = json.dumps({"guid_enabled": True, "guid_attn_method": "SEG", **extra})
            with self.subTest(extra=extra), self.assertRaisesRegex(RuntimeError, message):
                guidance.ForgeNeoAnimaGuidanceSuite().patch(
                    object(), object(), object(), object(), True, payload
                )

    def test_enabled_but_neutral_suite_and_disabled_inner_daemon_are_identity(self):
        model = _FakeModel()
        self.assertIs(
            guidance.ForgeNeoAnimaGuidanceSuite().patch(
                model, object(), object(), object(), True, "{}"
            )[0],
            model,
        )
        self.assertIs(
            guidance.ForgeNeoAnimaDetailDaemon().patch(
                model, True, '{"dd_enabled": false}'
            )[0],
            model,
        )


# ── DAVE = the original node (plan §4.3 C) ───────────────────────────────────
# origin: sorryhyun/ComfyUI-Anima-DAVE@83143e8d:nodes.py:91-106 (sigma -> schedule step),
#         :199-208 (tau gate), :163-172 (atten = clip(strength*w, 0, 1), > 1e-3, else no-op),
#         :75-88 (DC edit), dave_alpha.npz (weight 1 on flat blocks 8-18 of 28, else 0). MIT.
# Golden active-step counts: the ORIGINAL AnimaDAVE.patch wrapper, run on CPU with
# comfy.patcher_extension stubbed over Anima shift-3 simple schedules (float32 sample_sigmas);
# scratch generator origin_parity/dave/golden_gate.py, same numbers as the "original" column of
# origin_parity/dave/gate_sim.py. Steps 0..count-1 run DAVE, later steps do not.
_DAVE_ORIGIN_ACTIVE_STEPS = {
    0.05: {8: 1, 20: 1, 24: 1, 25: 1, 28: 1, 30: 2, 32: 2, 40: 2, 45: 2, 50: 2},
    0.10: {8: 1, 20: 2, 24: 2, 25: 2, 28: 3, 30: 3, 32: 3, 40: 4, 45: 4, 50: 5},
    0.15: {8: 1, 20: 3, 24: 4, 25: 4, 28: 4, 30: 4, 32: 5, 40: 6, 45: 7, 50: 8},
    0.30: {8: 2, 20: 6, 24: 7, 25: 8, 28: 8, 30: 9, 32: 10, 40: 12, 45: 14, 50: 15},
    1.00: {8: 8, 20: 20, 24: 24, 25: 25, 28: 28, 30: 30, 32: 32, 40: 40, 45: 45, 50: 50},
}
_DAVE_POOL = set(range(8, 19))


def _origin_dc_hook(atten):
    # origin: sorryhyun/ComfyUI-Anima-DAVE@83143e8d:nodes.py:75-88 (_dc_hook), verbatim math.
    # MIT, Copyright (c) 2026 Seunghyun Ji — full notice in the pack's guidance_dave.py.
    def hook(_module, _inputs, output):
        dims = tuple(range(1, output.ndim - 1)) or (1,)
        mu = output.float().mean(dim=dims, keepdim=True)
        return (output.float() - atten * mu).to(output.dtype)
    return hook


def _anima_simple_sigmas(steps: int, shift: float = 3.0) -> list[float]:
    """Comfy ``simple_scheduler`` over Anima's flow sigmas (shift 3), ending in 0 like sample_sigmas."""

    def sigma(t: float) -> float:
        return shift * t / (1 + (shift - 1) * t)

    table = [sigma((i + 1) / 1000) for i in range(1000)]
    stride = 1000 / steps
    return [table[-(1 + int(x * stride))] for x in range(steps)] + [0.0]


class _DaveBlock:
    def __init__(self, make_output):
        self._make_output = make_output

    def forward(self, *args, **kwargs):
        return self._make_output()


class _DaveBlockModel:
    """ModelPatcher stand-in: clone + add_object_patch over diffusion_model.blocks (Anima: 28)."""

    def __init__(self, count: int = 28, make_output=None):
        self.model = SimpleNamespace(diffusion_model=SimpleNamespace(
            blocks=[_DaveBlock(make_output) for _ in range(count)]
        ))
        self.object_patches = {}

    def clone(self):
        clone = _DaveBlockModel(0)
        clone.model = self.model
        clone.object_patches = dict(self.object_patches)
        return clone

    def add_object_patch(self, path, value):
        self.object_patches[path] = value


def _patched_blocks(model) -> set[int]:
    prefix, suffix = "diffusion_model.blocks.", ".forward"
    return {int(path[len(prefix):-len(suffix)]) for path in model.object_patches}


class TestAnimaDaveOriginParity(unittest.TestCase):
    @staticmethod
    def _active_steps(sigmas, tau) -> list[int]:
        return [
            index for index in range(len(sigmas) - 1)
            if guidance_dave.dave_gate_active({"sample_sigmas": sigmas, "sigmas": sigmas[index]}, tau)
        ]

    def test_tau_gate_matches_origin_step_tables(self):
        for tau, row in _DAVE_ORIGIN_ACTIVE_STEPS.items():
            for steps, count in row.items():
                with self.subTest(tau=tau, steps=steps):
                    self.assertEqual(
                        self._active_steps(_anima_simple_sigmas(steps), tau), list(range(count))
                    )
        # tau .1 spelled out; the old 1 - sigma gate ran 20 -> 0..5, 28 -> 0..7, 50 -> 0..12.
        self.assertEqual(self._active_steps(_anima_simple_sigmas(20), 0.1), [0, 1])
        self.assertEqual(self._active_steps(_anima_simple_sigmas(25), 0.1), [0, 1])
        self.assertEqual(self._active_steps(_anima_simple_sigmas(28), 0.1), [0, 1, 2])
        self.assertEqual(self._active_steps(_anima_simple_sigmas(30), 0.1), [0, 1, 2])
        self.assertEqual(self._active_steps(_anima_simple_sigmas(50), 0.1), [0, 1, 2, 3, 4])

    def test_tau_gate_sweep_matches_origin_cutoff(self):
        # origin: nodes.py:205-206 — k = max(1, min(n_steps, round(tau_f * n_steps))); gate = step < k
        # (Python round: 0.15 * 30 = 4.5 -> 4, 0.1 * 25 = 2.5 -> 2).
        for steps in range(1, 61):
            sigmas = _anima_simple_sigmas(steps)
            for tau in (0.01, 0.05, 0.1, 0.125, 0.15, 0.25, 0.3, 0.5, 0.75, 1.0):
                cutoff = max(1, min(steps, round(tau * steps)))
                with self.subTest(steps=steps, tau=tau):
                    self.assertEqual(self._active_steps(sigmas, tau), list(range(cutoff)))

    def test_step_lookup_reproduces_origin_quirks(self):
        step, gate = guidance_dave.dave_step, guidance_dave.dave_gate_active
        sigmas = _anima_simple_sigmas(30)
        self.assertEqual(step({"sample_sigmas": sigmas, "sigmas": sigmas[5]}), (5, 30))
        self.assertFalse(gate({"sample_sigmas": sigmas, "sigmas": sigmas[5]}, 0.1))
        # An off-schedule sigma (second-order mid-point, s_churn) matches nothing -> step 0 -> on.
        midpoint = (sigmas[5] + sigmas[6]) / 2
        self.assertEqual(step({"sample_sigmas": sigmas, "sigmas": midpoint}), (0, 30))
        self.assertTrue(gate({"sample_sigmas": sigmas, "sigmas": midpoint}, 0.1))
        # isclose(rtol=1e-4, atol=1e-6), tolerance relative to the current sigma (golden run:
        # sigma5 * (1 + 5e-5) -> off, sigma5 * (1 + 5e-4) -> on).
        self.assertEqual(step({"sample_sigmas": sigmas, "sigmas": sigmas[5] * (1 + 5e-5)}), (5, 30))
        self.assertEqual(step({"sample_sigmas": sigmas, "sigmas": sigmas[5] * (1 + 5e-4)}), (0, 30))
        # First matching index wins.
        self.assertEqual(step({"sample_sigmas": [1.0, 0.5, 0.5, 0.0], "sigmas": 0.5}), (1, 3))
        # A block never sees apply_model's t: without Comfy's 'sigmas' the sigma is unmatched.
        self.assertEqual(step({"sample_sigmas": sigmas}), (0, 30))

    def test_no_schedule_or_non_positive_tau_runs_every_step(self):
        late = _anima_simple_sigmas(30)[29]
        for options in (
            {}, {"sigmas": late}, {"sample_sigmas": None, "sigmas": late},
            {"sample_sigmas": [late], "sigmas": late},
        ):
            with self.subTest(options=options):
                self.assertEqual(guidance_dave.dave_step(options), (None, None))
                self.assertTrue(guidance_dave.dave_gate_active(options, 0.1))
        sigmas = _anima_simple_sigmas(30)
        for tau in (0.0, -0.25):
            self.assertEqual(self._active_steps(sigmas, tau), list(range(30)))

    def test_attenuation_is_clipped_and_tiny_or_negative_strength_is_identity(self):
        # Golden run of the original patch(): strength 0 / 0.001 / -0.5 -> model returned as is,
        # 0.0011 / 1.7 -> wrapper on blocks 8-18 (1.7 clipped to atten 1).
        self.assertEqual(guidance_dave.dave_attenuation(1.7), 1.0)
        self.assertEqual(guidance_dave.dave_attenuation(-0.5), 0.0)
        self.assertEqual(guidance_dave.dave_attenuation(0.3), 0.3)
        node = guidance.ForgeNeoAnimaDAVE()
        for strength in (0.0, 0.001, -0.5):
            with self.subTest(strength=strength):
                model = _DaveBlockModel()
                self.assertIs(node.patch(model, True, "dave_alpha.npz", strength, 0.1)[0], model)
        for strength in (0.0011, 1.7):
            with self.subTest(strength=strength):
                model = _DaveBlockModel()
                patched = node.patch(model, True, "dave_alpha.npz", strength, 0.1)[0]
                self.assertIsNot(patched, model)
                self.assertEqual(model.object_patches, {})
                self.assertEqual(_patched_blocks(patched), _DAVE_POOL)
        # SLG keeps its block when DAVE's attenuation rounds to nothing; DAVE never runs there.
        output = object()
        patched = guidance._patch_anima_blocks(
            _DaveBlockModel(make_output=lambda: output),
            dave_enabled=True, dave_blocks="8-18", dave_strength=0.0005, dave_tau=0.0,
            slg_enabled=True, slg_blocks="18",
        )
        self.assertEqual(_patched_blocks(patched), {18})
        wrapper = patched.object_patches["diffusion_model.blocks.18.forward"]
        self.assertIs(wrapper(None, transformer_options={}), output)

    def test_empty_block_field_is_the_origin_mask(self):
        node = guidance.ForgeNeoAnimaDAVE()
        for mask in ("dave_alpha.npz", "blocks:8-18", "blocks:", "blocks:  "):
            with self.subTest(mask=mask):
                patched = node.patch(_DaveBlockModel(), True, mask, 0.3, 0.1)[0]
                self.assertEqual(_patched_blocks(patched), _DAVE_POOL)
        suite = guidance.ForgeNeoAnimaGuidanceSuite()
        for blocks in ("", "   ", None):
            settings = {"guid_dave_enabled": True}
            if blocks is not None:
                settings["guid_dave_blocks"] = blocks
            with self.subTest(guid_dave_blocks=blocks):
                patched = suite.patch(
                    _DaveBlockModel(), object(), object(), object(), True, json.dumps(settings)
                )[0]
                self.assertEqual(_patched_blocks(patched), _DAVE_POOL)

    def test_missing_or_short_block_list_passes_through_like_origin(self):
        # origin: nodes.py:216-219 (no diffusion_model.blocks -> executor runs unchanged) and
        # :221-225 (only mask blocks with i < len(blocks) are hooked): DAVE never fails a graph.
        node = guidance.ForgeNeoAnimaDAVE()
        no_blocks = _DaveBlockModel()
        no_blocks.model = SimpleNamespace(diffusion_model=SimpleNamespace())
        for label, model in (("no blocks", no_blocks), ("0", _DaveBlockModel(0)), ("8", _DaveBlockModel(8))):
            with self.subTest(blocks=label):
                with self.assertLogs("ai_studio_forge_parity", "WARNING"):
                    self.assertIs(node.patch(model, True, "dave_alpha.npz", 0.3, 0.1)[0], model)
                self.assertEqual(model.object_patches, {})
        # 12 blocks: the original hooks the existing mask blocks 8..11.
        patched = node.patch(_DaveBlockModel(12), True, "dave_alpha.npz", 0.3, 0.1)[0]
        self.assertEqual(_patched_blocks(patched), {8, 9, 10, 11})
        # Suite: a block list with no block on this model is dropped the same way (the Forge
        # extension logs "no valid DAVE blocks — DAVE skipped."), and SLG keeps its own block.
        settings = {"guid_dave_enabled": True, "guid_dave_blocks": "30-40"}
        with self.assertLogs("ai_studio_forge_parity", "WARNING"):
            patched = guidance.ForgeNeoAnimaGuidanceSuite().patch(
                _DaveBlockModel(), object(), object(), object(), True, json.dumps(settings)
            )[0]
        self.assertEqual(_patched_blocks(patched), set())
        output = object()
        with self.assertLogs("ai_studio_forge_parity", "WARNING"):
            patched = guidance._patch_anima_blocks(
                _DaveBlockModel(make_output=lambda: output),
                dave_enabled=True, dave_blocks="30-40", dave_strength=0.3, dave_tau=0.0,
                slg_enabled=True, slg_blocks="18",
            )
        self.assertEqual(_patched_blocks(patched), {18})
        wrapper = patched.object_patches["diffusion_model.blocks.18.forward"]
        self.assertIs(wrapper(None, transformer_options={}), output)
        # SLG (not part of the DAVE node) still needs blocks, and a non-MODEL input still fails
        # like the original's model.clone() (nodes.py:182).
        with self.assertRaisesRegex(RuntimeError, "SLG requires"), \
                self.assertLogs("ai_studio_forge_parity", "WARNING"):
            guidance._patch_anima_blocks(
                no_blocks, dave_enabled=True, dave_blocks="8-18", dave_strength=0.3,
                dave_tau=0.1, slg_enabled=True, slg_blocks="18",
            )
        with self.assertRaisesRegex(RuntimeError, "requires a ComfyUI MODEL input"):
            node.patch(object(), True, "dave_alpha.npz", 0.3, 0.1)

    @requires_torch
    def test_block_wrapper_clips_strength_like_origin(self):
        # origin: nodes.py:165 — atten = np.clip(strength * weight, 0, 1), applied by the hook
        # (:222). Strength 1.7 removes the DC once (atten 1), never 1.7 times.
        torch = load_torch()
        generator = torch.Generator().manual_seed(2)
        block_out = torch.randn((2, 1, 4, 4, 8), generator=generator) * 2 + 1
        self.assertFalse(torch.equal(
            _origin_dc_hook(1.7)(None, None, block_out), _origin_dc_hook(1.0)(None, None, block_out)
        ))
        schedule = torch.tensor(_anima_simple_sigmas(30), dtype=torch.float32)
        options = {"sample_sigmas": schedule, "sigmas": schedule[0].reshape(1).repeat(2)}
        suite_settings = {"guid_dave_enabled": True, "guid_dave_strength": 1.7, "guid_dave_tau": 0.1}
        for strength, atten in ((1.7, 1.0), (1.0, 1.0), (0.8, 0.8)):
            node_model = guidance.ForgeNeoAnimaDAVE().patch(
                _DaveBlockModel(make_output=block_out.clone), True, "dave_alpha.npz", strength, 0.1,
            )[0]
            suite_settings["guid_dave_strength"] = strength
            suite_model = guidance.ForgeNeoAnimaGuidanceSuite().patch(
                _DaveBlockModel(make_output=block_out.clone), object(), object(), object(), True,
                json.dumps(suite_settings),
            )[0]
            expected = _origin_dc_hook(atten)(None, None, block_out)
            for source, patched in (("node", node_model), ("suite", suite_model)):
                for index in sorted(_DAVE_POOL):
                    wrapper = patched.object_patches[f"diffusion_model.blocks.{index}.forward"]
                    with self.subTest(source=source, strength=strength, block=index):
                        self.assertTrue(torch.equal(wrapper(None, transformer_options=options), expected))

    @requires_torch
    def test_block_edit_is_bit_identical_to_origin_hook(self):
        torch = load_torch()
        origin_hook = _origin_dc_hook
        generator = torch.Generator().manual_seed(0)
        base = torch.randn((2, 1, 6, 4, 8), generator=generator) * 3 + 1
        # 5-D predict2 block output, 3-D token layout, and the 2-D fallback dim (1,).
        layouts = (base, base.reshape(2, 24, 8), base.reshape(2, 192))
        for dtype in (torch.float32, torch.bfloat16, torch.float16):
            for value in layouts:
                value = value.to(dtype)
                for strength, atten in ((0.05, 0.05), (0.3, 0.3), (0.8, 0.8), (1.0, 1.0), (1.7, 1.0)):
                    with self.subTest(dtype=dtype, ndim=value.ndim, strength=strength):
                        mine = guidance_dave.apply_dave(value, guidance_dave.dave_attenuation(strength))
                        self.assertTrue(torch.equal(mine, origin_hook(atten)(None, None, value)))

    @requires_torch
    def test_tensor_schedule_lookup_matches_origin_tables(self):
        torch = load_torch()
        for tau, row in _DAVE_ORIGIN_ACTIVE_STEPS.items():
            for steps, count in row.items():
                schedule = torch.tensor(_anima_simple_sigmas(steps), dtype=torch.float32)
                active = [
                    index for index in range(steps)
                    if guidance_dave.dave_gate_active(
                        {"sample_sigmas": schedule, "sigmas": schedule[index].reshape(1).repeat(2)},
                        tau,
                    )
                ]
                with self.subTest(tau=tau, steps=steps):
                    self.assertEqual(active, list(range(count)))
        schedule = torch.tensor(_anima_simple_sigmas(30), dtype=torch.float32)
        step = guidance_dave.dave_step

        def at(sigma):
            return {"sample_sigmas": schedule, "sigmas": sigma.reshape(-1)}

        self.assertEqual(step(at(schedule[5] * (1 + 5e-5))), (5, 30))
        self.assertEqual(step(at(schedule[5] * (1 + 5e-4))), (0, 30))
        self.assertEqual(step(at((schedule[5] + schedule[6]) / 2)), (0, 30))
        # nodes.py:102 — the batch's first sigma decides.
        self.assertEqual(step(at(torch.stack([schedule[4], schedule[9]]))), (4, 30))

    @requires_torch
    def test_block_wrapper_runs_origin_gate_on_the_live_schedule(self):
        torch = load_torch()
        generator = torch.Generator().manual_seed(1)
        block_out = torch.randn((2, 1, 4, 4, 8), generator=generator)
        edited = (block_out.float() - 0.5 * block_out.float().mean(dim=(1, 2, 3), keepdim=True)).to(
            block_out.dtype
        )
        patched = guidance.ForgeNeoAnimaDAVE().patch(
            _DaveBlockModel(make_output=block_out.clone), True, "dave_alpha.npz", 0.5, 0.1,
        )[0]
        self.assertEqual(_patched_blocks(patched), _DAVE_POOL)
        wrapper = patched.object_patches["diffusion_model.blocks.8.forward"]
        schedule = torch.tensor(_anima_simple_sigmas(30), dtype=torch.float32)

        def options(sigma):
            return {"sample_sigmas": schedule, "sigmas": sigma.reshape(1).repeat(2)}

        # 30 steps at tau .1 -> steps 0..2 (golden table). Keyword and Cosmos positional (#7)
        # transformer_options are read the same way.
        for index in range(30):
            expected = edited if index < 3 else block_out
            with self.subTest(step=index):
                self.assertTrue(torch.equal(wrapper(*([None] * 6), options(schedule[index])), expected))
                self.assertTrue(torch.equal(
                    wrapper(None, transformer_options=options(schedule[index])), expected
                ))
        midpoint = (schedule[5] + schedule[6]) / 2
        self.assertTrue(torch.equal(wrapper(None, transformer_options=options(midpoint)), edited))
        late_no_schedule = {"sigmas": schedule[20].reshape(1)}
        self.assertTrue(torch.equal(wrapper(None, transformer_options=late_no_schedule), edited))


if __name__ == "__main__":
    unittest.main()
