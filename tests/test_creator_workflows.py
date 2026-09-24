import json
import unittest

from core.creator_workflows import (
    CreatorWorkflowError,
    SUPPORTED_MODES,
    build,
)


class CreatorWorkflowTests(unittest.TestCase):
    def test_every_supported_mode_builds_json_safe_result(self):
        params_by_mode = {
            "h3_t2v": {"prompt": "a lighthouse in a storm"},
            "h3_i2v": {"prompt": "slow camera orbit", "input_image": "start.png"},
            "h3_v2v": {"prompt": "change the scene to winter", "input_video": "motion.mp4"},
            "krea2_t2i": {"prompt": "a glass sculpture on black velvet"},
            "krea2_edit": {"prompt": "change the jacket to red", "input_image": "person.png"},
            "krea2_hires": {"input_image": "result.png", "size": "2048x1536"},
        }
        for mode in SUPPORTED_MODES:
            with self.subTest(mode=mode):
                result = build(mode, params_by_mode[mode])
                json.dumps(result)
                self.assertIsInstance(result, dict)
                self.assertEqual(result["metadata"]["mode"], mode)
                self.assertTrue(result["workflow"])
                self.assertTrue(result["required_node_types"])
                self.assertIs(result.workflow, result["workflow"])
                self.assertIs(result.required_node_types, result["required_node_types"])
                self.assertTrue(result["output_node_ids"])
                self.assertEqual(
                    result["capability"]["required_node_types"],
                    result["required_node_types"],
                )

    def test_h3_t2v_graph_has_video_and_animated_webp_outputs(self):
        result = build(
            "h3-t2v",
            {
                "prompt": "a paper boat crossing a puddle",
                "size": "608x352",
                "frames": 121,
                "fps": 24,
                "seed": 42,
                "quality": "quality",
                "output_prefix": "Creator/H3/test",
            },
        )
        graph = result["workflow"]
        self.assertEqual(graph["6"]["class_type"], "MiniMaxH3ImageToVideo")
        self.assertNotIn("first_frame", graph["6"]["inputs"])
        self.assertEqual(graph["11"]["inputs"]["latent_image"], ["6", 1])
        self.assertEqual(graph["15"]["class_type"], "SaveVideo")
        self.assertEqual(graph["20"]["class_type"], "SaveAnimatedWEBP")
        self.assertEqual(result["output_node_ids"], ["15", "20"])
        self.assertEqual(result["metadata"]["frames"], 124)
        self.assertEqual(result["metadata"]["frames"] % 17, 5)
        self.assertEqual(result["metadata"]["seed"], 42)

    def test_h3_i2v_loads_first_frame(self):
        result = build(
            "h3_i2v",
            {"prompt": "the subject smiles", "input_image": "uploads/first.webp"},
        )
        graph = result["workflow"]
        self.assertEqual(graph["5"], {"class_type": "LoadImage", "inputs": {"image": "uploads/first.webp"}})
        self.assertEqual(graph["6"]["inputs"]["first_frame"], ["5", 0])

    def test_h3_v2v_has_video_preprocessor_and_optional_identity_and_audio(self):
        result = build(
            "h3_v2v",
            {
                "prompt": "preserve motion but move the scene outdoors",
                "input_image": "identity.png",
                "input_video": "references/motion.mp4",
                "include_reference_audio": True,
                "frames": 120,
                "fps": 24,
            },
        )
        graph = result["workflow"]
        self.assertEqual(graph["18"]["class_type"], "GemmaVideoReferencePreprocessor")
        self.assertEqual(graph["18"]["inputs"]["file"], "references/motion.mp4")
        self.assertEqual(graph["18"]["inputs"]["fps"], 24)
        self.assertLessEqual(graph["18"]["inputs"]["duration"], 15)
        self.assertTrue(graph["18"]["inputs"]["include_audio"])
        self.assertEqual(
            graph["1"]["inputs"]["unet_name"],
            "minimax_h3_ref2va_pruned_int8_convrot.safetensors",
        )
        self.assertEqual(graph["9"]["class_type"], "KSamplerSelect")
        self.assertEqual(graph["10"]["inputs"]["steps"], 20)
        self.assertEqual(graph["6"]["class_type"], "MiniMaxH3ReferenceToVideo")
        self.assertEqual(graph["6"]["inputs"]["ref_images.ref_image_0"], ["5", 0])
        self.assertEqual(graph["6"]["inputs"]["ref_videos.ref_video_0"], ["18", 0])
        self.assertEqual(graph["6"]["inputs"]["ref_video_audios.ref_video_audio_0"], ["18", 1])
        self.assertEqual(graph["14"]["inputs"]["audio"], ["13", 0])
        self.assertIn("GemmaVideoReferencePreprocessor", result["custom_node_types"])

    def test_model_preset_overrides_are_applied_without_mutating_defaults(self):
        custom = {
            "h3_unet": "custom/H3-v2.safetensors",
            "h3_clip": "custom/h3-clip.safetensors",
        }
        first = build("h3_t2v", {"prompt": "test", "model_preset": custom})
        second = build("h3_t2v", {"prompt": "test"})
        self.assertEqual(first["workflow"]["1"]["inputs"]["unet_name"], "custom/H3-v2.safetensors")
        self.assertEqual(first["workflow"]["2"]["inputs"]["clip_name"], "custom/h3-clip.safetensors")
        self.assertNotEqual(
            first["workflow"]["1"]["inputs"]["unet_name"],
            second["workflow"]["1"]["inputs"]["unet_name"],
        )

    def test_krea_mode_aliases_resolve_to_canonical_modes(self):
        edit_params = {"prompt": "make the coat blue", "input_image": "person.png"}
        hires_params = {"input_image": "result.png"}
        for alias in ("krea2", "krea_edit"):
            with self.subTest(alias=alias):
                self.assertEqual(build(alias, edit_params)["metadata"]["mode"], "krea2_edit")
        self.assertEqual(build("krea_hires", hires_params)["metadata"]["mode"], "krea2_hires")

    def test_krea2_t2i_matches_official_turbo_core_graph(self):
        result = build(
            "krea2_t2i",
            {"prompt": "a ceramic fox", "size": "1001x769", "seed": 44},
        )
        graph = result["workflow"]
        self.assertEqual(graph["1"]["class_type"], "UNETLoader")
        self.assertEqual(graph["2"]["inputs"]["type"], "krea2")
        self.assertEqual(graph["4"], {
            "class_type": "CLIPTextEncode",
            "inputs": {"text": "a ceramic fox", "clip": ["2", 0]},
        })
        self.assertEqual(graph["5"]["class_type"], "ConditioningZeroOut")
        self.assertEqual(graph["6"]["class_type"], "EmptyLatentImage")
        self.assertEqual(graph["6"]["inputs"]["width"], 1008)
        self.assertEqual(graph["6"]["inputs"]["height"], 776)
        self.assertEqual(graph["8"]["inputs"]["steps"], 8)
        self.assertEqual(graph["8"]["inputs"]["cfg"], 1)
        self.assertEqual(graph["8"]["inputs"]["sampler_name"], "euler")
        self.assertEqual(graph["8"]["inputs"]["scheduler"], "simple")
        self.assertEqual(graph["11"]["inputs"]["images"], ["10", 0])
        self.assertNotIn("LoraLoaderModelOnly", result["required_node_types"])

    def test_transport_param_aliases(self):
        i2v = build(
            "h3_i2v",
            {
                "prompt": "blink once",
                "source_image": "studio/upload.png",
                "includeAudio": True,
                "outputPrefix": "Creator/transport",
            },
        )
        self.assertEqual(i2v["workflow"]["5"]["inputs"]["image"], "studio/upload.png")
        self.assertEqual(i2v["workflow"]["15"]["inputs"]["filename_prefix"], "Creator/transport")
        self.assertIn("13", i2v["workflow"])

        v2v = build(
            "h3_v2v",
            {
                "prompt": "preserve the choreography",
                "source_image": "studio/motion.mp4",
                "includeAudio": True,
            },
        )
        self.assertEqual(v2v["workflow"]["18"]["inputs"]["file"], "studio/motion.mp4")
        self.assertTrue(v2v["workflow"]["18"]["inputs"]["include_audio"])

    def test_krea2_edit_graph_preserves_identity_contract(self):
        result = build(
            "krea2_edit",
            {
                "prompt": "replace the background with a studio wall",
                "input_image": "subject.jpg",
                "size": "1001x769",
                "fidelity": 7.5,
                "seed": 9,
            },
        )
        graph = result["workflow"]
        self.assertEqual(graph["4"]["inputs"]["model"], ["15", 0])
        self.assertEqual(graph["8"]["class_type"], "Krea2EditGroundedEncode")
        self.assertEqual(graph["10"]["class_type"], "Krea2EditModelPatch")
        self.assertEqual(graph["10"]["inputs"]["ref_boost"], 7.5)
        self.assertEqual(graph["10"]["inputs"]["source_latent"], ["6", 0])
        self.assertEqual(graph["10"]["inputs"]["target_latent"], ["7", 0])
        self.assertEqual(graph["7"]["inputs"]["width"], 1008)
        self.assertEqual(graph["7"]["inputs"]["height"], 784)
        self.assertEqual(graph["14"]["inputs"]["images"], ["13", 0])
        self.assertEqual(result["metadata"]["fidelity"], 7.5)

    def test_krea2_edit_can_use_distinct_identity_reference(self):
        result = build(
            "krea2",
            {
                "prompt": "keep the face and change the outfit",
                "input_image": "source.png",
                "reference_image": "identity.webp",
            },
        )
        graph = result["workflow"]
        self.assertEqual(graph["5"]["inputs"]["image"], "source.png")
        self.assertEqual(graph["16"]["inputs"]["image"], "identity.webp")
        self.assertEqual(graph["6"]["inputs"]["pixels"], ["5", 0])
        self.assertEqual(graph["17"]["inputs"]["pixels"], ["16", 0])
        self.assertEqual(graph["8"]["inputs"]["image"], ["5", 0])
        self.assertEqual(graph["8"]["inputs"]["image_b"], ["16", 0])
        self.assertEqual(graph["10"]["inputs"]["source_image"], ["5", 0])
        self.assertEqual(graph["10"]["inputs"]["source_image_b"], ["16", 0])
        self.assertEqual(graph["10"]["inputs"]["source_latent_b"], ["17", 0])
        self.assertTrue(result["metadata"]["uses_reference_image"])

    def test_krea2_hires_supports_source_size_and_scale(self):
        result = build(
            "krea2_hires",
            {
                "input_image": "generated/result.png",
                "source_size": [1152, 896],
                "scale": 2,
                "denoise": 0.35,
                "use_textfusion": False,
            },
        )
        graph = result["workflow"]
        self.assertEqual(result["metadata"]["width"], 2304)
        self.assertEqual(result["metadata"]["height"], 1792)
        self.assertEqual(graph["7"]["class_type"], "UpscaleModelLoader")
        self.assertEqual(graph["8"]["class_type"], "ImageUpscaleWithModel")
        self.assertEqual(graph["9"]["inputs"]["width"], 2304)
        self.assertEqual(graph["12"]["inputs"]["model"], ["1", 0])
        self.assertEqual(graph["12"]["inputs"]["denoise"], 0.35)
        self.assertNotIn("11", graph)

    def test_validation_rejects_bad_modes_files_sizes_frames_fps_seed_and_fidelity(self):
        invalid_cases = [
            ("unknown", {}, "unsupported creator mode"),
            ("h3_i2v", {"prompt": "x"}, "requires input_image"),
            ("h3_v2v", {"prompt": "x", "input_video": "../motion.mp4"}, "traversal"),
            ("h3_v2v", {"prompt": "x", "input_video": "motion.exe"}, "extensions"),
            ("h3_t2v", {"prompt": "x", "size": "610x352"}, "multiples of 32"),
            ("h3_t2v", {"prompt": "x", "frames": 4}, "frames"),
            ("h3_t2v", {"prompt": "x", "fps": 0}, "fps"),
            ("h3_t2v", {"prompt": "x", "fps": 30}, "fps"),
            (
                "h3_v2v",
                {"prompt": "x", "input_video": "motion.mp4", "quality": "turbo"},
                "requires quality mode",
            ),
            ("h3_t2v", {"prompt": "x", "seed": -1}, "seed"),
            (
                "krea2_edit",
                {"prompt": "x", "input_image": "input.png", "fidelity": 13},
                "fidelity",
            ),
            (
                "krea2_edit",
                {"prompt": "x", "input_image": "C:\\absolute\\input.png"},
                "relative",
            ),
            (
                "krea2_hires",
                {"input_image": "CON.png"},
                "reserved Windows filename",
            ),
        ]
        for mode, params, fragment in invalid_cases:
            with self.subTest(mode=mode, params=params):
                with self.assertRaisesRegex(CreatorWorkflowError, fragment):
                    build(mode, params)

    def test_model_preset_rejects_unknown_keys_and_unsafe_filenames(self):
        with self.assertRaisesRegex(CreatorWorkflowError, "unknown model_preset keys"):
            build("h3_t2v", {"prompt": "x", "model_preset": {"typo": "x.safetensors"}})
        with self.assertRaisesRegex(CreatorWorkflowError, "traversal"):
            build(
                "h3_t2v",
                {"prompt": "x", "model_preset": {"h3_unet": "../unsafe.safetensors"}},
            )


if __name__ == "__main__":
    unittest.main()
