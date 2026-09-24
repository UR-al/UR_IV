from __future__ import annotations

import unittest

from comfy_custom_nodes.ai_studio_forge_parity.anima_lora import (
    ANIMA_29B_BLOCKS,
    ANIMA_38B_BLOCKS,
    ANIMA_BASE_BLOCKS,
    BLOCK_MAPPINGS,
    AnimaLoraCompatibilityError,
    inspect_anima_model,
    prepare_anima_lora_for_model,
    remap_anima_lora_state_dict,
)
from tests._optional_deps import load_torch, requires_torch


class _CloneValue:
    def __init__(self, lineage: int, clone_number: int = 0):
        self.lineage = lineage
        self.clone_number = clone_number

    def clone(self):
        return _CloneValue(self.lineage, self.clone_number + 1)


class _Diffusion:
    def __init__(self, blocks: int):
        self.blocks = [object() for _ in range(blocks)]


class _ModelConfig:
    def __init__(self, image_model: str):
        self.unet_config = {"image_model": image_model}


class _BaseModel:
    def __init__(self, blocks: int, image_model: str = "anima"):
        self.diffusion_model = _Diffusion(blocks)
        self.model_config = _ModelConfig(image_model)


class _ModelPatcher:
    def __init__(self, blocks: int, image_model: str = "anima"):
        self.model = _BaseModel(blocks, image_model)


def _lora(blocks: int, namespace: str = "kohya") -> dict[str, object]:
    if namespace == "kohya":
        return {
            f"lora_unet_blocks_{index}_attn.lora_up.weight": _CloneValue(index)
            for index in range(blocks)
        }
    if namespace == "native":
        return {
            f"diffusion_model.blocks.{index}.attn.lora_up.weight": _CloneValue(index)
            for index in range(blocks)
        }
    raise AssertionError(namespace)


def _lineages(state: dict[str, object], namespace: str = "kohya") -> list[int]:
    prefix = "lora_unet_blocks_" if namespace == "kohya" else "diffusion_model.blocks."
    separator = "_" if namespace == "kohya" else "."
    values: dict[int, int] = {}
    for key, value in state.items():
        if not key.startswith(prefix):
            continue
        index_text = key[len(prefix):].split(separator, 1)[0]
        values[int(index_text)] = value.lineage
    return [values[index] for index in sorted(values)]


class MappingTests(unittest.TestCase):
    def test_all_six_directed_pairs_are_present_and_valid(self):
        layouts = (ANIMA_BASE_BLOCKS, ANIMA_29B_BLOCKS, ANIMA_38B_BLOCKS)
        expected = {
            (source, target)
            for source in layouts
            for target in layouts
            if source != target
        }
        self.assertEqual(set(BLOCK_MAPPINGS), expected)
        for (source, target), mapping in BLOCK_MAPPINGS.items():
            self.assertEqual(len(mapping), target)
            self.assertTrue(all(0 <= index < source for index in mapping))

    def test_28_to_40_matches_published_expansion_manifest(self):
        mapping = BLOCK_MAPPINGS[(28, 40)]
        inserted_to_source = {
            2: 1, 5: 3, 8: 5, 11: 7, 14: 9, 17: 11,
            21: 14, 24: 16, 27: 18, 30: 20, 33: 22, 36: 24,
        }
        self.assertEqual(
            {target: mapping[target] for target in inserted_to_source},
            inserted_to_source,
        )
        self.assertEqual(
            [
                mapping[target]
                for target in range(40)
                if target not in inserted_to_source
            ],
            list(range(28)),
        )

    def test_40_to_52_uses_the_twelve_inserted_positions(self):
        mapping = BLOCK_MAPPINGS[(40, 52)]
        expected = {
            3: 2, 7: 5, 11: 8, 15: 11, 19: 14, 23: 17,
            27: 20, 31: 23, 35: 26, 39: 29, 43: 32, 47: 35,
        }
        self.assertEqual({index: mapping[index] for index in expected}, expected)

    def test_each_expansion_and_contraction_round_trips_lineage(self):
        for source, expanded in ((28, 40), (40, 52), (28, 52)):
            with self.subTest(source=source, expanded=expanded):
                original = _lora(source)
                up = remap_anima_lora_state_dict(original, expanded)
                down = remap_anima_lora_state_dict(up.state_dict, source)
                self.assertEqual(_lineages(down.state_dict), list(range(source)))
                self.assertEqual(up.report.duplicated_blocks, expanded - source)
                self.assertEqual(down.report.dropped_blocks, expanded - source)

    def test_both_key_namespaces_follow_the_same_mapping(self):
        for namespace in ("kohya", "native"):
            with self.subTest(namespace=namespace):
                result = remap_anima_lora_state_dict(_lora(40, namespace), 52)
                self.assertEqual(
                    _lineages(result.state_dict, namespace),
                    list(BLOCK_MAPPINGS[(40, 52)]),
                )
                self.assertEqual(result.report.namespaces, (namespace,))

    def test_matching_mixed_namespaces_are_remapped_before_live_alias_validation(self):
        raw = {
            **_lora(28, "kohya"),
            **{
                f"diffusion_model.blocks.{index}.mlp.lora_up.weight": _CloneValue(index)
                for index in range(28)
            },
        }
        result = remap_anima_lora_state_dict(raw, 40)
        self.assertEqual(result.report.namespaces, ("kohya", "native"))
        self.assertEqual(
            _lineages(result.state_dict, "kohya"),
            list(BLOCK_MAPPINGS[(28, 40)]),
        )
        self.assertEqual(
            _lineages(result.state_dict, "native"),
            list(BLOCK_MAPPINGS[(28, 40)]),
        )


class SafetyTests(unittest.TestCase):
    def test_input_mapping_is_never_mutated_and_result_is_a_shallow_copy(self):
        raw = _lora(28)
        raw["unrelated"] = _CloneValue(999)
        before = dict(raw)
        result = remap_anima_lora_state_dict(raw, 40)
        self.assertIsNot(result.state_dict, raw)
        self.assertEqual(raw, before)
        self.assertIs(result.state_dict["unrelated"], raw["unrelated"])

    def test_terminal_anchored_sparse_namespaces_are_valid(self):
        for source in (28, 40):
            with self.subTest(source=source):
                raw = {
                    key: value
                    for key, value in _lora(source).items()
                    if not key.startswith((
                        "lora_unet_blocks_0_", "lora_unet_blocks_1_",
                    ))
                }
                result = remap_anima_lora_state_dict(raw, source)
                self.assertEqual(result.report.source_blocks, source)
                self.assertEqual(result.report.target_blocks, source)
                self.assertEqual(len(result.state_dict), source - 2)
                self.assertEqual(
                    _lineages(result.state_dict), list(range(2, source)),
                )

    def test_real_native_peft_sparse_layouts_remap_across_supported_depths(self):
        """Lock the 2..27/2..39 PEFT layout used by the shipped rdbt LoRAs."""

        for source in (28, 40):
            raw = {
                f"diffusion_model.blocks.{index}.self_attn.to_q.lora_{side}.weight": (
                    _CloneValue(index)
                )
                for index in range(2, source)
                for side in ("A", "B")
            }
            for target in (28, 40, 52):
                if target == source:
                    continue
                with self.subTest(source=source, target=target):
                    result = remap_anima_lora_state_dict(raw, target)
                    mapping = BLOCK_MAPPINGS[(source, target)]
                    expected_lineage = [index for index in mapping if index >= 2]
                    self.assertEqual(
                        _lineages(result.state_dict, "native"),
                        expected_lineage,
                    )
                    self.assertEqual(
                        len(result.state_dict),
                        2 * len(expected_lineage),
                    )
                    self.assertEqual(result.report.source_blocks, source)
                    self.assertEqual(result.report.target_blocks, target)

    def test_sparse_namespace_without_a_supported_terminal_anchor_is_rejected(self):
        raw = _lora(28)
        del raw["lora_unet_blocks_27_attn.lora_up.weight"]
        with self.assertRaisesRegex(AnimaLoraCompatibilityError, "sparse"):
            remap_anima_lora_state_dict(raw, 40)

    def test_union_of_two_sparse_namespaces_is_not_mistaken_for_complete(self):
        raw = {
            **{
                f"lora_unet_blocks_{index}_x": _CloneValue(index)
                for index in range(14)
            },
            **{
                f"diffusion_model.blocks.{index}.x": _CloneValue(index)
                for index in range(14, 28)
            },
        }
        with self.assertRaisesRegex(AnimaLoraCompatibilityError, "namespace"):
            remap_anima_lora_state_dict(raw, 40)

    def test_complete_namespaces_with_different_layouts_are_rejected(self):
        raw = {**_lora(28, "kohya"), **_lora(40, "native")}
        with self.assertRaisesRegex(AnimaLoraCompatibilityError, "different layouts"):
            remap_anima_lora_state_dict(raw, 52)

    def test_normalized_destination_collision_is_a_hard_failure(self):
        raw = _lora(28)
        raw["lora_unet_blocks_00_attn.lora_up.weight"] = _CloneValue(1000)
        with self.assertRaisesRegex(AnimaLoraCompatibilityError, "collision"):
            remap_anima_lora_state_dict(raw, 40)

    def test_expanded_tensor_values_are_cloned_not_aliased(self):
        raw = _lora(40)
        original = raw["lora_unet_blocks_2_attn.lora_up.weight"]
        result = remap_anima_lora_state_dict(raw, 52)
        inherited = result.state_dict["lora_unet_blocks_2_attn.lora_up.weight"]
        inserted = result.state_dict["lora_unet_blocks_3_attn.lora_up.weight"]
        self.assertIs(inherited, original)
        self.assertIsNot(inserted, original)
        self.assertEqual(inserted.clone_number, 1)
        self.assertGreater(result.report.duplicated_tensor_keys, 0)

    def test_same_layout_preserves_value_objects(self):
        raw = _lora(40)
        result = remap_anima_lora_state_dict(raw, 40)
        self.assertTrue(result.report.passthrough)
        self.assertFalse(result.report.remapped)
        for key, value in raw.items():
            self.assertIs(result.state_dict[key], value)

    def test_unsupported_target_is_rejected_without_mutating_input(self):
        raw = _lora(28)
        before = dict(raw)
        with self.assertRaisesRegex(AnimaLoraCompatibilityError, "unsupported"):
            remap_anima_lora_state_dict(raw, 64)
        self.assertEqual(raw, before)

    def test_forge_llm_adapter_namespaces_are_left_unchanged(self):
        keys = (
            "diffusion_model.llm_adapter.out_proj.weight",
            "lora_unet_llm_adapter_blocks_0_x",
        )
        raw = {key: _CloneValue(index) for index, key in enumerate(keys)}
        result = remap_anima_lora_state_dict(raw, 28)
        self.assertEqual(set(result.state_dict), set(keys))
        self.assertTrue(result.report.passthrough)

    def test_downward_projection_drops_only_38b_connector_namespaces(self):
        raw = _lora(52)
        connector_keys = (
            "net.anima_v2_connector.anchor.weight",
            "diffusion_model.anima_v2_connector.gate.weight",
            "lora_unet_anima_v2_connector_proj.lora_up.weight",
        )
        for index, key in enumerate(connector_keys):
            raw[key] = _CloneValue(100 + index)
        qwen_key = "lora_te_qwen35_4b_blocks_0_attn.lora_up.weight"
        raw[qwen_key] = _CloneValue(500)
        result = remap_anima_lora_state_dict(raw, 40)
        self.assertEqual(result.report.dropped_connector_keys, 3)
        self.assertTrue(all(key not in result.state_dict for key in connector_keys))
        self.assertIn(qwen_key, result.state_dict)
        self.assertTrue(all(key in raw for key in connector_keys))

    def test_connector_keys_are_preserved_for_52_block_target(self):
        raw = _lora(52)
        key = "lora_unet_anima_v2_connector_proj.lora_up.weight"
        raw[key] = _CloneValue(100)
        result = remap_anima_lora_state_dict(raw, 52)
        self.assertIn(key, result.state_dict)
        self.assertEqual(result.report.dropped_connector_keys, 0)

    def test_connector_keys_require_a_live_connector_on_a_52_block_model(self):
        raw = _lora(52)
        key = "lora_unet_anima_v2_connector_proj.lora_up.weight"
        raw[key] = _CloneValue(100)
        model = _ModelPatcher(52)
        with self.assertRaisesRegex(
            AnimaLoraCompatibilityError,
            "has no anima_v2_connector",
        ):
            prepare_anima_lora_for_model(raw, model)

        model.model.diffusion_model.anima_v2_connector = object()
        result = prepare_anima_lora_for_model(raw, model)
        self.assertIn(key, result.state_dict)
        self.assertTrue(inspect_anima_model(model).has_38b_connector)

    def test_connector_only_downward_projection_is_rejected(self):
        key = "lora_unet_anima_v2_connector_proj.lora_up.weight"
        raw = {key: _CloneValue(100)}
        with self.assertRaisesRegex(AnimaLoraCompatibilityError, "connector-only"):
            remap_anima_lora_state_dict(raw, 40)
        self.assertEqual(set(raw), {key})

    @requires_torch
    def test_optional_torch_tensor_duplicate_is_a_real_clone(self):
        torch = load_torch()
        raw = {
            f"lora_unet_blocks_{index}_x": torch.tensor([float(index)])
            for index in range(40)
        }
        result = remap_anima_lora_state_dict(raw, 52)
        original = raw["lora_unet_blocks_2_x"]
        duplicate = result.state_dict["lora_unet_blocks_3_x"]
        self.assertTrue(torch.equal(original, duplicate))
        self.assertNotEqual(original.data_ptr(), duplicate.data_ptr())


class ModelInspectionTests(unittest.TestCase):
    def test_anima_model_block_count_comes_from_live_model(self):
        model = _ModelPatcher(52)
        info = inspect_anima_model(model)
        self.assertTrue(info.is_anima)
        self.assertEqual(info.block_count, 52)
        self.assertIn("model_config.unet_config.image_model", info.evidence)

    def test_non_anima_with_supported_depth_is_passed_through(self):
        raw = _lora(28)
        model = _ModelPatcher(28, image_model="other")
        result = prepare_anima_lora_for_model(raw, model)
        self.assertFalse(result.report.model_is_anima)
        self.assertEqual(result.report.action, "passthrough_non_anima")
        self.assertTrue(result.report.passthrough)
        self.assertIsNot(result.state_dict, raw)
        self.assertEqual(result.state_dict, raw)

    def test_anima_model_routes_through_remap(self):
        result = prepare_anima_lora_for_model(_lora(28), _ModelPatcher(40))
        self.assertTrue(result.report.model_is_anima)
        self.assertEqual(result.report.source_blocks, 28)
        self.assertEqual(result.report.target_blocks, 40)
        self.assertEqual(result.report.action, "expand")

    def test_marked_anima_without_blocks_is_rejected(self):
        base = type("Anima", (), {"diffusion_model": object()})()
        with self.assertRaisesRegex(AnimaLoraCompatibilityError, "does not expose"):
            prepare_anima_lora_for_model(_lora(28), base)

    def test_marked_anima_with_unknown_depth_is_rejected(self):
        with self.assertRaisesRegex(AnimaLoraCompatibilityError, "unsupported"):
            prepare_anima_lora_for_model(_lora(28), _ModelPatcher(41))

    def test_anima_lora_without_block_keys_is_a_shallow_passthrough(self):
        value = _CloneValue(1)
        raw = {"lora_te_text_model_encoder_x": value}
        result = prepare_anima_lora_for_model(raw, _ModelPatcher(40))
        self.assertEqual(result.report.action, "passthrough_no_blocks")
        self.assertTrue(result.report.passthrough)
        self.assertIs(result.state_dict["lora_te_text_model_encoder_x"], value)


if __name__ == "__main__":
    unittest.main()
