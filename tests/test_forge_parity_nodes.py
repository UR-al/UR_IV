from __future__ import annotations

import json
from types import SimpleNamespace
import unittest
from unittest import mock

from comfy_custom_nodes.ai_studio_forge_parity import compat
from comfy_custom_nodes.ai_studio_forge_parity import generation
from comfy_custom_nodes.ai_studio_forge_parity import guidance
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
            "ForgeNeoKSamplerCNS", "ForgeNeoModelSamplingShift",
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
        args = dict(
            start=0.2, end=0.8, bias=0.5, amount=0.25, exponent=1.0,
            start_offset=0.02, end_offset=-0.01, fade=0.2, smooth=True,
        )
        self.assertAlmostEqual(guidance.detail_schedule_value(0.0, **args), 0.016)
        self.assertAlmostEqual(guidance.detail_schedule_value(0.5, **args), 0.20)
        self.assertAlmostEqual(guidance.detail_schedule_value(1.0, **args), -0.008)

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

    def test_suite_rejects_unsupported_head_selective_mode_before_runtime(self):
        payload = json.dumps({
            "guid_enabled": True,
            "guid_attn_method": "PAG",
            "guid_head_indices": "0,2",
        })
        with self.assertRaisesRegex(RuntimeError, "Head-selective"):
            guidance.ForgeNeoAnimaGuidanceSuite().patch(
                object(), object(), object(), object(), True, payload
            )

    @requires_torch
    def test_standalone_dave_shares_suite_wrapper_and_tau_cutoff(self):
        torch = load_torch()

        class _Block:
            def forward(self, *args, **kwargs):
                return torch.ones((1, 4, 2)) * 2.0

        class _BlockModel:
            def __init__(self, count):
                self.model = SimpleNamespace(diffusion_model=SimpleNamespace(
                    blocks=[_Block() for _ in range(count)]
                ))
                self.object_patches = {}

            def clone(self):
                clone = _BlockModel(0)
                clone.model = self.model
                clone.object_patches = dict(self.object_patches)
                return clone

            def add_object_patch(self, path, value):
                self.object_patches[path] = value

        model = _BlockModel(24)
        (patched,) = guidance.ForgeNeoAnimaDAVE().patch(
            model, True, "dave_alpha.npz", 0.5, 0.5,
        )
        self.assertIsNot(patched, model)
        self.assertEqual(model.object_patches, {})
        # 'dave_alpha.npz' keeps the saved COMBO valid and maps to Forge 8-18.
        self.assertEqual(
            set(patched.object_patches),
            {f"diffusion_model.blocks.{index}.forward" for index in range(8, 19)},
        )
        wrapper = patched.object_patches["diffusion_model.blocks.8.forward"]
        early = {"sigmas": torch.tensor([0.9])}   # progress 0.1 < tau 0.5
        late = {"sigmas": torch.tensor([0.1])}    # progress 0.9 >= tau 0.5
        # Keyword and Cosmos positional (#7) transformer_options are honoured
        # the same way — the old standalone copy only read the keyword.
        positional_early = wrapper(*([None] * 6), early)
        positional_late = wrapper(*([None] * 6), late)
        keyword_late = wrapper(None, transformer_options=late)
        self.assertTrue(torch.allclose(positional_early, torch.ones((1, 4, 2))))
        self.assertTrue(torch.allclose(positional_late, torch.full((1, 4, 2), 2.0)))
        self.assertTrue(torch.allclose(keyword_late, torch.full((1, 4, 2), 2.0)))

        (explicit,) = guidance.ForgeNeoAnimaDAVE().patch(
            _BlockModel(24), True, "blocks:8-18", 0.5, 0.0,
        )
        self.assertEqual(set(explicit.object_patches), set(patched.object_patches))
        self.assertIs(
            guidance.ForgeNeoAnimaDAVE().patch(model, True, "blocks:8-18", 0.0)[0],
            model,
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


if __name__ == "__main__":
    unittest.main()
