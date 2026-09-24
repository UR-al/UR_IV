from __future__ import annotations

import unittest
from unittest import mock

from comfy_custom_nodes.ai_studio_forge_parity import anima_lora_nodes
from comfy_custom_nodes.ai_studio_forge_parity import generation


class _Diffusion:
    def __init__(self, blocks: int):
        self.blocks = [object() for _ in range(blocks)]


class _Config:
    unet_config = {"image_model": "anima"}


class _Base:
    def __init__(self, blocks: int):
        self.diffusion_model = _Diffusion(blocks)
        self.model_config = _Config()


class _Model:
    def __init__(self, blocks: int):
        self.model = _Base(blocks)


class _PatchModel(_Model):
    def __init__(self, blocks: int):
        super().__init__(blocks)
        self.patches: list[tuple[dict[object, object], float]] = []
        self.attachments: dict[str, object] = {}

    def clone(self):
        cloned = _PatchModel(len(self.model.diffusion_model.blocks))
        cloned.model = self.model
        return cloned

    def add_patches(self, patches, strength):
        self.patches.append((dict(patches), float(strength)))
        return tuple(patches)

    def set_attachments(self, key, value):
        self.attachments[str(key)] = value


class _ClipPatcher:
    def __init__(self):
        self.attachments: dict[str, object] = {}

    def set_attachments(self, key, value):
        self.attachments[str(key)] = value


class _Clip:
    def __init__(self):
        self.cond_stage_model = object()
        self.patcher = _ClipPatcher()
        self.patches: list[tuple[dict[object, object], float]] = []

    def clone(self):
        return _Clip()

    def add_patches(self, patches, strength):
        self.patches.append((dict(patches), float(strength)))
        return tuple(patches)


def _raw_lora(blocks: int) -> dict[str, object]:
    return {
        f"lora_unet_blocks_{index}_attn.lora_up.weight": object()
        for index in range(blocks)
    }


class AnimaLoraNodeTests(unittest.TestCase):
    def test_drop_in_node_contracts(self):
        loader = anima_lora_nodes.ForgeNeoAnimaLoraLoader
        model_only = anima_lora_nodes.ForgeNeoAnimaLoraLoaderModelOnly

        self.assertEqual(loader.RETURN_TYPES, ("MODEL", "CLIP"))
        self.assertEqual(loader.FUNCTION, "load_lora")
        self.assertEqual(model_only.RETURN_TYPES, ("MODEL",))
        self.assertEqual(model_only.FUNCTION, "load_lora_model_only")
        with mock.patch.object(anima_lora_nodes, "filename_choices", return_value=["x.safetensors"]):
            self.assertEqual(
                set(loader.INPUT_TYPES()["required"]),
                {"model", "clip", "lora_name", "strength_model", "strength_clip"},
            )

    def test_state_cache_keeps_raw_mapping_and_metadata(self):
        raw = {"key": object()}
        metadata = {"modelspec.architecture": "anima/lora"}
        comfy_utils = mock.Mock()
        comfy_utils.load_torch_file.return_value = (raw, metadata)
        folder_paths = mock.Mock()
        folder_paths.get_full_path_or_raise.return_value = "C:/models/x.safetensors"
        cache = anima_lora_nodes.AnimaLoraStateCache()

        with mock.patch.object(anima_lora_nodes, "folder_paths_module", return_value=folder_paths), mock.patch.object(
            anima_lora_nodes.importlib,
            "import_module",
            return_value=comfy_utils,
        ):
            first = cache.load("x.safetensors")
            second = cache.load("x.safetensors")

        self.assertIs(first[0], raw)
        self.assertIs(first[1], metadata)
        self.assertEqual(first, second)
        comfy_utils.load_torch_file.assert_called_once_with(
            "C:/models/x.safetensors",
            safe_load=True,
            return_metadata=True,
        )

    def test_comfy_loader_receives_remapped_copy_and_original_metadata(self):
        raw = _raw_lora(28)
        metadata = {"name": "kept"}
        cache = mock.Mock()
        cache.load.return_value = (raw, metadata)
        comfy_sd = mock.Mock()
        comfy_sd.load_lora_for_models.return_value = ("patched-model", "patched-clip")
        comfy_lora = mock.Mock()
        comfy_lora.model_lora_keys_unet.return_value = {}
        comfy_lora.model_lora_keys_clip.return_value = {}
        comfy_convert = mock.Mock()
        comfy_convert.convert_lora.side_effect = lambda value: value

        def import_module(name):
            return {
                "comfy.lora": comfy_lora,
                "comfy.lora_convert": comfy_convert,
                "comfy.sd": comfy_sd,
            }[name]

        with mock.patch.object(
            anima_lora_nodes.importlib,
            "import_module",
            side_effect=import_module,
        ):
            result = anima_lora_nodes.load_lora_for_models(
                _Model(40),
                _Clip(),
                "x.safetensors",
                0.8,
                0.4,
                cache=cache,
            )

        self.assertEqual(result, ("patched-model", "patched-clip"))
        passed = comfy_sd.load_lora_for_models.call_args.args[2]
        self.assertIsNot(passed, raw)
        self.assertEqual(len(raw), 28)
        self.assertEqual(
            len([key for key in passed if key.startswith("lora_unet_blocks_")]),
            40,
        )
        self.assertIs(
            comfy_sd.load_lora_for_models.call_args.kwargs["lora_metadata"],
            metadata,
        )

    def test_cached_raw_lora_can_be_reused_across_different_anima_depths(self):
        raw = _raw_lora(28)
        cache = mock.Mock()
        cache.load.return_value = (raw, {"kept": "yes"})
        comfy_sd = mock.Mock()
        comfy_sd.load_lora_for_models.side_effect = lambda model, clip, state, *_args, **_kwargs: (
            len([key for key in state if key.startswith("lora_unet_blocks_")]),
            clip,
        )
        comfy_lora = mock.Mock()
        comfy_lora.model_lora_keys_unet.return_value = {}
        comfy_lora.model_lora_keys_clip.return_value = {}
        comfy_convert = mock.Mock()
        comfy_convert.convert_lora.side_effect = lambda value: value

        def import_module(name):
            return {
                "comfy.lora": comfy_lora,
                "comfy.lora_convert": comfy_convert,
                "comfy.sd": comfy_sd,
            }[name]

        with mock.patch.object(
            anima_lora_nodes.importlib,
            "import_module",
            side_effect=import_module,
        ):
            clip = _Clip()
            first = anima_lora_nodes.load_lora_for_models(
                _Model(40), clip, "x.safetensors", 1.0, 1.0, cache=cache,
            )
            second = anima_lora_nodes.load_lora_for_models(
                _Model(52), clip, "x.safetensors", 1.0, 1.0, cache=cache,
            )

        self.assertEqual(first[0], 40)
        self.assertEqual(second[0], 52)
        self.assertEqual(len(raw), 28)
        self.assertEqual(cache.load.call_count, 2)

    def test_zero_strength_returns_without_loading(self):
        cache = mock.Mock()
        model, clip = object(), object()
        result = anima_lora_nodes.load_lora_for_models(
            model,
            clip,
            "x.safetensors",
            0.0,
            0.0,
            cache=cache,
        )
        self.assertEqual(result, (model, clip))
        cache.load.assert_not_called()

    def test_anima_block_weight_applies_each_block_ratio_and_preserves_metadata(self):
        raw = _raw_lora(52)
        raw["lora_unet_blocks_0_attn.diff_b"] = object()
        raw["lora_unet_final.lora_up.weight"] = object()
        raw["lora_te_x.lora_up.weight"] = object()
        metadata = {"kept": "yes"}
        cache = mock.Mock()
        cache.load.return_value = (raw, metadata)
        comfy_lora = mock.Mock()
        comfy_lora.model_lora_keys_unet.return_value = {
            "lora_unet_blocks_0_attn": "diffusion_model.blocks.0.attn.weight",
            "lora_unet_blocks_51_attn": "diffusion_model.blocks.51.attn.weight",
            "lora_unet_final": "diffusion_model.final.weight",
        }
        comfy_lora.model_lora_keys_clip.return_value = {
            "lora_te_x": "clip.transformer.x.weight",
        }
        comfy_lora.load_lora.return_value = {
            "diffusion_model.blocks.0.attn.weight": "block-0",
            "diffusion_model.blocks.0.attn.bias": "block-0-bias",
            "diffusion_model.blocks.51.attn.weight": "block-51",
            "diffusion_model.final.weight": "base",
            "clip.transformer.x.weight": "clip-base",
        }
        comfy_convert = mock.Mock()
        comfy_convert.convert_lora.side_effect = lambda value: value
        vector = ["0.5", *("1" for _ in range(52))]
        vector[1] = "0.25"
        vector[52] = "0.75"
        model, clip = _PatchModel(52), _Clip()

        def import_module(name):
            return {
                "comfy.lora": comfy_lora,
                "comfy.lora_convert": comfy_convert,
            }[name]

        with mock.patch.object(anima_lora_nodes.importlib, "import_module", side_effect=import_module), mock.patch.object(
            anima_lora_nodes,
            "provider_class",
            side_effect=AssertionError("ANIMA must not delegate to Inspire"),
        ):
            result = anima_lora_nodes.load_lora_block_weight(
                model,
                clip,
                "x.safetensors",
                2.0,
                3.0,
                False,
                7,
                4.0,
                1.0,
                ",".join(vector),
                cache=cache,
            )

        patched_model, patched_clip, populated = result
        self.assertEqual(
            [(next(iter(patch)), strength) for patch, strength in patched_model.patches],
            [
                ("diffusion_model.blocks.0.attn.weight", 0.5),
                ("diffusion_model.blocks.0.attn.bias", 0.5),
                ("diffusion_model.blocks.51.attn.weight", 1.5),
                ("diffusion_model.final.weight", 1.0),
            ],
        )
        self.assertEqual(
            [(next(iter(patch)), strength) for patch, strength in patched_clip.patches],
            [("clip.transformer.x.weight", 1.5)],
        )
        self.assertIs(patched_model.attachments["lora_metadata"], metadata)
        self.assertIs(patched_clip.patcher.attachments["lora_metadata"], metadata)
        self.assertEqual(model.attachments, {})
        self.assertEqual(clip.patcher.attachments, {})
        self.assertEqual(len(populated.split(",")), 53)
        self.assertEqual(len(raw), 55)

    def test_empty_anima_block_vector_is_uniform_for_every_supported_depth(self):
        for block_count in (28, 40, 52):
            with self.subTest(block_count=block_count):
                parsed = anima_lora_nodes._parse_anima_block_vector(
                    "",
                    block_count,
                    inverse=False,
                    seed=0,
                    a_value=4.0,
                    b_value=1.0,
                )
                self.assertEqual(parsed.base, 1.0)
                self.assertEqual(parsed.blocks, (1.0,) * block_count)
                self.assertEqual(len(parsed.populated.split(",")), block_count + 1)

                inverted = anima_lora_nodes._parse_anima_block_vector(
                    "",
                    block_count,
                    inverse=True,
                    seed=0,
                    a_value=4.0,
                    b_value=1.0,
                )
                self.assertEqual(inverted.base, 0.0)
                self.assertEqual(inverted.blocks, (0.0,) * block_count)

        block_input = generation.ForgeNeoLoraBlockWeight.INPUT_TYPES()[
            "required"
        ]["block_vector"]
        self.assertEqual(block_input[1]["default"], "")

    def test_anima_block_vector_rejects_legacy_short_vector(self):
        with self.assertRaisesRegex(RuntimeError, "52 block values"):
            anima_lora_nodes._parse_anima_block_vector(
                "1,0,0",
                52,
                inverse=False,
                seed=0,
                a_value=4.0,
                b_value=1.0,
            )

    def test_anima_block_weight_rejects_empty_or_unaccepted_comfy_patches(self):
        raw = _raw_lora(52)
        cache = mock.Mock()
        cache.load.return_value = (raw, {})
        model, clip = _PatchModel(52), _Clip()
        comfy_lora = mock.Mock()
        comfy_lora.model_lora_keys_unet.return_value = {
            "lora_unet_blocks_0_attn": "diffusion_model.blocks.0.attn.weight",
        }
        comfy_lora.model_lora_keys_clip.return_value = {}
        comfy_convert = mock.Mock()
        comfy_convert.convert_lora.side_effect = lambda value: value

        def import_module(name):
            return {
                "comfy.lora": comfy_lora,
                "comfy.lora_convert": comfy_convert,
            }[name]

        with mock.patch.object(
            anima_lora_nodes.importlib,
            "import_module",
            side_effect=import_module,
        ):
            comfy_lora.load_lora.return_value = {}
            with self.assertRaisesRegex(RuntimeError, "did not produce any Comfy patches"):
                anima_lora_nodes.load_lora_block_weight(
                    model, clip, "x.safetensors", 1.0, 1.0,
                    False, 0, 4.0, 1.0, "", cache=cache,
                )

            comfy_lora.load_lora.return_value = {
                "diffusion_model.blocks.0.attn.weight": "patch",
            }
            with mock.patch.object(
                _PatchModel,
                "add_patches",
                return_value=[],
            ), self.assertRaisesRegex(RuntimeError, "rejected by the active Comfy model"):
                anima_lora_nodes.load_lora_block_weight(
                    model, clip, "x.safetensors", 1.0, 1.0,
                    False, 0, 4.0, 1.0, "", cache=cache,
                )

    def test_comfy_destination_alias_collision_is_a_hard_failure(self):
        state = {
            "lora_unet_blocks_0_attn.lora_up.weight": object(),
            "diffusion_model.blocks.0.attn.lora_up.weight": object(),
        }
        key_map = {
            "lora_unet_blocks_0_attn": "diffusion_model.blocks.0.attn.weight",
            "diffusion_model.blocks.0.attn": "diffusion_model.blocks.0.attn.weight",
        }
        with self.assertRaisesRegex(RuntimeError, "same Comfy destination"):
            anima_lora_nodes._reject_comfy_alias_collisions(state, key_map)

    def test_alias_prefix_set_matches_the_linear_payload_scan_exactly(self):
        # 원래 조건(key == alias / alias + "." / alias + "_" 접두)을 오라클로 두고
        # set 기반 판정이 모든 경계 — 점이 여러 개인 별칭 포함 — 에서 같은지 본다.
        keys = (
            "diffusion_model.blocks.0.attn.lora_up.weight",
            "lora_unet_blocks_12_mlp_fc1.alpha",
            "text_encoders.qwen3.layer_3.lora_down.weight",
            "plain",
        )
        aliases = (
            "diffusion_model", "diffusion_model.blocks", "diffusion_model.blocks.0",
            "diffusion_model.blocks.0.attn", "diffusion_model.blocks.0.att",
            "diffusion_model.blocks.1.attn", "lora_unet", "lora_unet_blocks_12",
            "lora_unet_blocks_1", "lora_unet_blocks_12_mlp_fc1",
            "lora_unet_blocks_12_mlp_fc1.alpha", "lora_unet_blocks_12_mlp_fc",
            "text_encoders.qwen3.layer_3", "text_encoders.qwen3.layer",
            "plain", "pla", "plain.", "", "unrelated",
        )

        def oracle(alias):
            return any(
                key == alias or key.startswith(f"{alias}.") or key.startswith(f"{alias}_")
                for key in keys
            )

        prefixes = anima_lora_nodes._payload_alias_prefixes(keys)
        for alias in aliases:
            with self.subTest(alias=alias):
                self.assertEqual(alias in prefixes, oracle(alias))

    def test_alias_collision_ignores_aliases_without_payload(self):
        state = {"diffusion_model.blocks.0.attn.lora_up.weight": object(), 7: object()}
        key_map = {
            # 같은 목적지지만 두 번째 별칭은 payload가 없으므로 충돌이 아니다.
            "diffusion_model.blocks.0.attn": "diffusion_model.blocks.0.attn.weight",
            "lora_unet_blocks_0_attn": "diffusion_model.blocks.0.attn.weight",
            ("tuple", "alias"): "ignored",
        }
        anima_lora_nodes._reject_comfy_alias_collisions(state, key_map)

    def test_standard_anima_loader_checks_connector_alias_collisions(self):
        raw = {
            "lora_unet_anima_v2_connector_proj.lora_up.weight": object(),
            "diffusion_model.anima_v2_connector.proj.lora_up.weight": object(),
        }
        cache = mock.Mock()
        cache.load.return_value = (raw, {})
        model = _Model(52)
        model.model.diffusion_model.anima_v2_connector = object()
        comfy_lora = mock.Mock()
        comfy_lora.model_lora_keys_unet.return_value = {
            "lora_unet_anima_v2_connector_proj": (
                "diffusion_model.anima_v2_connector.proj.weight"
            ),
            "diffusion_model.anima_v2_connector.proj": (
                "diffusion_model.anima_v2_connector.proj.weight"
            ),
        }
        comfy_lora.model_lora_keys_clip.return_value = {}
        comfy_convert = mock.Mock()
        comfy_convert.convert_lora.side_effect = lambda value: value
        comfy_sd = mock.Mock()

        def import_module(name):
            return {
                "comfy.lora": comfy_lora,
                "comfy.lora_convert": comfy_convert,
                "comfy.sd": comfy_sd,
            }[name]

        with mock.patch.object(
            anima_lora_nodes.importlib,
            "import_module",
            side_effect=import_module,
        ), self.assertRaisesRegex(RuntimeError, "same Comfy destination"):
            anima_lora_nodes.load_lora_for_models(
                model,
                None,
                "x.safetensors",
                1.0,
                0.0,
                cache=cache,
            )
        comfy_sd.load_lora_for_models.assert_not_called()

    def test_block_weight_zero_strength_returns_without_loading(self):
        cache = mock.Mock()
        model, clip = object(), object()
        result = anima_lora_nodes.load_lora_block_weight(
            model,
            clip,
            "x.safetensors",
            0.0,
            0.0,
            False,
            0,
            4.0,
            1.0,
            "1,0",
            cache=cache,
        )
        self.assertEqual(result, (model, clip, ""))
        cache.load.assert_not_called()

    def test_non_anima_block_weight_fails_if_metadata_cannot_be_preserved(self):
        cache = mock.Mock()
        cache.load.return_value = ({"key": object()}, {"name": "must survive"})
        with self.assertRaisesRegex(RuntimeError, "cannot preserve safetensors metadata"):
            anima_lora_nodes.load_lora_block_weight(
                object(),
                object(),
                "x.safetensors",
                1.0,
                1.0,
                False,
                0,
                4.0,
                1.0,
                "1,0",
                cache=cache,
            )

    def test_non_anima_empty_vector_keeps_inspire_legacy_default(self):
        cache = mock.Mock()
        cache.load.return_value = ({"key": object()}, {})
        provider = mock.Mock()
        provider.load_lora_for_models.return_value = (
            "patched-model", "patched-clip", "populated",
        )
        with mock.patch.object(
            anima_lora_nodes,
            "provider_class",
            return_value=provider,
        ):
            result = anima_lora_nodes.load_lora_block_weight(
                object(),
                object(),
                "x.safetensors",
                1.0,
                1.0,
                False,
                0,
                4.0,
                1.0,
                "",
                cache=cache,
            )

        self.assertEqual(result, ("patched-model", "patched-clip", "populated"))
        self.assertEqual(
            provider.load_lora_for_models.call_args.args[-1],
            anima_lora_nodes._LEGACY_INSPIRE_BLOCK_VECTOR,
        )

    def test_character_reference_model_only_uses_shared_anima_loader(self):
        node = generation.ForgeNeoCharacterReference()
        with mock.patch.object(
            generation,
            "load_lora_model_only",
            return_value="patched",
        ) as apply_lora:
            result = node.patch(
                "model",
                "positive",
                "negative",
                "vae",
                enabled=True,
                lora_name="reference.safetensors",
                lora_strength=0.75,
                reference_method="split_screen",
                reference_image=object(),
            )

        self.assertEqual(result, ("patched", "positive", "negative"))
        apply_lora.assert_called_once_with(
            "model",
            "reference.safetensors",
            0.75,
            cache=node._anima_lora_cache,
        )

    def test_block_weight_node_uses_shared_remap_path(self):
        node = generation.ForgeNeoLoraBlockWeight()
        with mock.patch.object(
            generation,
            "load_lora_block_weight",
            return_value=("m", "c", "vector"),
        ) as apply_lora:
            result = node.load(
                "model",
                "clip",
                enabled=True,
                lora_name="style.safetensors",
                block_vector="1,0.5",
            )

        self.assertEqual(result, ("m", "c", "vector"))
        self.assertIs(
            apply_lora.call_args.kwargs["cache"],
            node._anima_lora_cache,
        )


if __name__ == "__main__":
    unittest.main()
