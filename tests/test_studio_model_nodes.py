"""Studio nodes over optional packs (MOSS stream, Anima and Krea2 references) and the flux-type shift they rely on."""
import json
import os
import tempfile
import time
import unittest
from pathlib import Path
from unittest import mock

from comfy_custom_nodes.ai_studio_forge_parity import anima_lora, generation, guidance, moss_lora_layout, reference_layout, studio_nodes
from comfy_custom_nodes.ai_studio_forge_parity import studio_model_nodes as nodes
from comfy_custom_nodes.ai_studio_forge_parity.guidance_common import ASYM_REFERENCE_KEY
from tests._optional_deps import load_torch, requires_torch


def fake_torch_zeros(shape):
    return shape


class ProviderCalls:
    """Stands in for ``invoke_provider``: records each call and returns the outputs a test gave that provider."""

    def __init__(self, **outputs):
        self.outputs = outputs
        self.calls = []

    def __call__(self, name, *, method=None, feature=None, args=(), kwargs=None):
        self.calls.append({"name": name, "method": method, "feature": feature, "args": args, "kwargs": kwargs or {}})
        result = self.outputs[name]
        return result(len(self.calls)) if callable(result) else result


class MossStreamTests(unittest.TestCase):
    def test_calls_the_toolkit_node(self):
        calls = ProviderCalls(MOSSStreamFP32=("streamed", "extra"))
        with mock.patch.object(nodes, "invoke_provider", calls):
            self.assertEqual(nodes.AIStudioMossStream().patch("model"), ("streamed",))
        self.assertEqual((calls.calls[0]["name"], calls.calls[0]["method"], calls.calls[0]["args"]),
                         ("MOSSStreamFP32", "patch", ("model",)))

    def test_the_stream_and_the_layer_skip_are_optional_switches_after_the_model(self):
        spec = nodes.AIStudioMossStream.INPUT_TYPES()
        self.assertEqual(list(spec["required"]), ["model"])  # a workflow saved before 1.20.0 has this socket alone
        self.assertEqual(list(spec["optional"]), ["stream", "layer_skip"])
        self.assertEqual((spec["optional"]["stream"][0], spec["optional"]["stream"][1]["default"]), ("BOOLEAN", True))
        self.assertEqual((spec["optional"]["layer_skip"][0], spec["optional"]["layer_skip"][1]["default"]), ("BOOLEAN", False))

    def test_the_stream_off_leaves_the_toolkit_out(self):
        calls = ProviderCalls(MOSSStreamFP32=("streamed",))
        with mock.patch.object(nodes, "invoke_provider", calls):
            self.assertEqual(nodes.AIStudioMossStream().patch("model", stream=False), ("model",))
        self.assertEqual(calls.calls, [])

    def test_the_layer_skip_goes_on_after_the_stream(self):
        streamed = FakeMossModel()
        calls = ProviderCalls(MOSSStreamFP32=(streamed,))
        with mock.patch.object(nodes, "invoke_provider", calls):
            (result,) = nodes.AIStudioMossStream().patch("model", stream=True, layer_skip=True)
        self.assertEqual(calls.calls[0]["args"], ("model",))
        self.assertIsNot(result, streamed)
        self.assertIn(SKIP_KEY, _dit_patches(result))
        self.assertEqual(_dit_patches(streamed), {})

    def test_the_layer_skip_alone_needs_no_toolkit(self):
        model = FakeMossModel()
        calls = ProviderCalls()
        with mock.patch.object(nodes, "invoke_provider", calls):
            (result,) = nodes.AIStudioMossStream().patch(model, stream=False, layer_skip=True)
        self.assertEqual(calls.calls, [])
        self.assertIn(SKIP_KEY, _dit_patches(result))


SKIP_KEY = ("single_block", 22)


class FakeMossModel:
    """A MODEL as far as the layer skip reads it: the inner model's class name, ``model_options`` and ComfyUI's
    copy-on-write ``set_model_patch_replace``."""

    def __init__(self, kind="MOSSImage", options=None):
        self.model = type(kind, (), {})()
        self.model_options = options if options is not None else {"transformer_options": {}}

    def clone(self):
        clone = FakeMossModel.__new__(FakeMossModel)
        clone.model = self.model
        clone.model_options = dict(self.model_options)
        return clone

    def set_model_patch_replace(self, patch, name, block_name, number):
        options = dict(self.model_options)
        transformer = dict(options.get("transformer_options") or {})
        replace = dict(transformer.get("patches_replace") or {})
        replace[name] = {**(replace.get(name) or {}), (block_name, number): patch}
        transformer["patches_replace"] = replace
        options["transformer_options"] = transformer
        self.model_options = options


def _dit_patches(model):
    return ((model.model_options.get("transformer_options") or {}).get("patches_replace") or {}).get("dit", {})


class Rows:
    """A batch tensor's rows for the tests without torch: a slice is a view, as in torch, until ``clone``."""

    def __init__(self, rows):
        self.rows = list(rows)

    @property
    def shape(self):
        return (len(self.rows),)

    def __getitem__(self, part):
        return RowsView(self, part)

    def __setitem__(self, part, value):
        self.rows[part] = value.rows


class RowsView:
    def __init__(self, owner, part):
        self.owner, self.part = owner, part

    @property
    def rows(self):
        return self.owner.rows[self.part]

    def clone(self):
        return Rows(self.rows)


class MossLayerSkipTests(unittest.TestCase):
    """OpenMOSS's sampling: the uncond pass without block 22 (core ``SkipLayerGuidanceDiTSimple`` with
    ``single_layers="22"``, 0..1), as a block replace of our own."""

    def _patch(self, model=None):
        patched = nodes.moss_layer_skip(model or FakeMossModel())
        return _dit_patches(patched)[SKIP_KEY]

    @staticmethod
    def _block(calls):
        def original_block(args):  # in place, as MOSS's blocks update the stream
            calls.append(list(args["img"].rows))
            args["img"].rows[:] = [row + 10 for row in args["img"].rows]
            return {"img": args["img"]}
        return original_block

    def _run(self, flags, rows=(1, 2, 3, 4), patch=None):
        calls = []
        args = {"img": Rows(rows), "transformer_options": {} if flags is None else {"cond_or_uncond": flags}}
        out = (patch or self._patch())(args, {"original_block": self._block(calls)})
        return out["img"].rows, calls, out is args

    def test_block_22_is_the_official_layer(self):
        self.assertEqual(nodes.MOSS_SKIP_LAYER, 22)

    def test_a_moss_model_gets_the_patch_on_a_clone_and_only_on_block_22(self):
        model = FakeMossModel()
        patched = nodes.moss_layer_skip(model)
        self.assertIsNot(patched, model)
        self.assertEqual(_dit_patches(model), {})
        self.assertEqual(list(_dit_patches(patched)), [SKIP_KEY])

    def test_any_other_model_is_left_as_it_is(self):
        for kind in ("Anima", "Krea2", "Flux"):
            with self.subTest(kind=kind):
                model = FakeMossModel(kind)
                self.assertIs(nodes.moss_layer_skip(model), model)
                self.assertEqual(_dit_patches(model), {})

    def test_the_uncond_pass_skips_the_block(self):
        rows, calls, same = self._run([1])
        self.assertEqual((rows, calls), ([1, 2, 3, 4], []))
        self.assertTrue(same)  # the core node's skip: the block's arguments come back untouched

    def test_the_cond_pass_runs_the_block(self):
        rows, calls, _ = self._run([0])
        self.assertEqual((rows, calls), ([11, 12, 13, 14], [[1, 2, 3, 4]]))

    def test_a_call_without_flags_runs_the_block(self):
        for flags in (None, []):
            with self.subTest(flags=flags):
                rows, calls, _ = self._run(flags)
                self.assertEqual((rows, len(calls)), ([11, 12, 13, 14], 1))

    def test_a_mixed_batch_keeps_the_uncond_rows_as_they_came_in(self):
        rows, calls, _ = self._run([0, 1])
        self.assertEqual((rows, len(calls)), ([11, 12, 3, 4], 1))
        rows, _, _ = self._run([1, 0])
        self.assertEqual(rows, [1, 2, 13, 14])

    def test_a_third_cond_of_a_guidance_pass_is_no_uncond(self):
        rows, _, _ = self._run([0, 1, 2], rows=(1, 2, 3))
        self.assertEqual(rows, [11, 2, 13])

    def _by_sample(self, flags, rows, per_chunk=1, options=None, patch=None):
        """A DiT that runs its blocks one sample at a time (the moss-image loader's, MOSS-Toolkit's reference pass):
        every call has one row and the whole batch's flags, in the forward's own copy of the options."""
        patch = patch or self._patch()
        options = {"cond_or_uncond": flags, "sigmas": Rows([0.5] * per_chunk)} if options is None else options
        out, calls = [], []
        for row in rows:
            out += patch({"img": Rows([row]), "transformer_options": options}, {"original_block": self._block(calls)})["img"].rows
        return out, calls

    def test_a_dit_that_runs_one_sample_at_a_time_skips_the_block_for_the_uncond_sample(self):
        rows, calls = self._by_sample([0, 1], (1, 2))
        self.assertEqual((rows, calls), ([11, 2], [[1]]))
        rows, calls = self._by_sample([1, 0], (1, 2))
        self.assertEqual((rows, calls), ([1, 12], [[2]]))

    def test_one_sample_at_a_time_with_a_batch_of_images_per_cond(self):
        rows, calls = self._by_sample([0, 1], (1, 2, 3, 4), per_chunk=2)
        self.assertEqual((rows, calls), ([11, 12, 3, 4], [[1], [2]]))
        rows, _ = self._by_sample([0, 1, 2], (1, 2, 3))
        self.assertEqual(rows, [11, 2, 13])

    def test_one_sample_at_a_time_starts_over_with_each_forward(self):
        patch = self._patch()
        shared = {"cond_or_uncond": [0, 1], "sigmas": Rows([0.5])}
        for _ in range(2):  # a copy of the options per forward, as both packs make
            rows, _ = self._by_sample(None, (1, 2), options=dict(shared), patch=patch)
            self.assertEqual(rows, [11, 2])
        for _ in range(2):  # and one dict for every forward
            rows, _ = self._by_sample(None, (1, 2), options=shared, patch=patch)
            self.assertEqual(rows, [11, 2])

    def test_the_whole_batch_in_one_call_reads_no_position(self):
        options = {"cond_or_uncond": [0, 1], "sigmas": Rows([0.5, 0.5])}
        for _ in range(2):
            calls = []
            out = self._patch()({"img": Rows((1, 2, 3, 4)), "transformer_options": options}, {"original_block": self._block(calls)})
            self.assertEqual(out["img"].rows, [11, 12, 3, 4])
        self.assertEqual(list(options), ["cond_or_uncond", "sigmas"])

    def test_rows_nobody_can_tell_apart_stop_the_run_instead_of_running_the_block_for_the_uncond(self):
        for options in ({"cond_or_uncond": [0, 1]}, {"cond_or_uncond": [0, 1], "sigmas": Rows([0.5])}):
            with self.subTest(options=list(options)):
                calls = []
                with self.assertRaisesRegex(RuntimeError, "MOSS layer skip"):
                    self._patch()({"img": Rows((1, 2, 3)), "transformer_options": options}, {"original_block": self._block(calls)})
                self.assertEqual(calls, [])

    def test_a_replace_that_was_on_the_block_runs_where_the_block_runs(self):
        seen = []

        def earlier(args, extra):
            seen.append("earlier")
            return extra["original_block"](args)
        model = FakeMossModel(options={"transformer_options": {"patches_replace": {"dit": {SKIP_KEY: earlier}}}})
        patch = self._patch(model)
        rows, calls, _ = self._run([0], patch=patch)
        self.assertEqual((rows, seen, len(calls)), ([11, 12, 13, 14], ["earlier"], 1))
        rows, calls, _ = self._run([1], patch=patch)
        self.assertEqual((rows, seen, calls), ([1, 2, 3, 4], ["earlier"], []))

    def test_a_second_application_adds_nothing(self):
        once = nodes.moss_layer_skip(FakeMossModel())
        twice = nodes.moss_layer_skip(once)
        self.assertIs(_dit_patches(twice)[SKIP_KEY], _dit_patches(once)[SKIP_KEY])

    @requires_torch
    def test_real_tensors_with_a_block_that_writes_in_place(self):
        torch = load_torch()
        patch = self._patch()

        def original_block(args):
            args["img"].add_(10.0)
            return {"img": args["img"]}
        img = torch.arange(8.0).reshape(4, 2)  # two chunks of two rows: cond, uncond
        out = patch({"img": img, "transformer_options": {"cond_or_uncond": [0, 1]}}, {"original_block": original_block})
        self.assertEqual(out["img"].tolist(), [[10.0, 11.0], [12.0, 13.0], [4.0, 5.0], [6.0, 7.0]])
        # ComfyUI's own options (sigmas: one per row of a chunk), whole batch and then one sample at a time
        options = {"cond_or_uncond": [1, 0], "sigmas": torch.ones(2)}
        img = torch.arange(8.0).reshape(4, 2)
        out = patch({"img": img, "transformer_options": dict(options)}, {"original_block": original_block})
        self.assertEqual(out["img"].tolist(), [[0.0, 1.0], [2.0, 3.0], [14.0, 15.0], [16.0, 17.0]])
        forward = dict(options)
        img = torch.arange(8.0).reshape(4, 2)
        outs = [patch({"img": img[row:row + 1], "transformer_options": forward}, {"original_block": original_block})["img"] for row in range(4)]
        self.assertEqual(torch.cat(outs).tolist(), [[0.0, 1.0], [2.0, 3.0], [14.0, 15.0], [16.0, 17.0]])


class AnimaReferenceTests(unittest.TestCase):
    values = {"model": "m", "clip": "c", "vae": "v", "image": "i", "file": "sub/ref.safetensors", "mode": "char",
              "strength": 0.5, "guidance_on": True, "guidance": 4.5, "correction": "apg", "ref_end": 0.6,
              "megapixels": 1.25, "caption": "1girl"}

    def _apply(self, method, asym=False, values=None, remap=(), native=frozenset(), **outputs):
        remap = dict(remap)
        calls = ProviderCalls(**outputs)
        self.native_seen = []

        def fit(_model, file, native=frozenset()):
            self.native_seen.append(native)
            return file if native else remap.get(file, file)
        with mock.patch.object(nodes, "invoke_provider", calls), mock.patch.object(nodes, "folder_paths_module"), \
                mock.patch.object(nodes, "_adapter_is_asym", return_value=asym), \
                mock.patch.object(nodes, "_snapref_native_layouts", return_value=native), \
                mock.patch.object(nodes, "adapter_for_model", side_effect=fit):
            result = nodes.AIStudioAnimaReference().apply(method=method, **(values or self.values))
        return result, calls.calls

    def test_a_third_reference_goes_with_its_weight(self):
        values = {**self.values, "image_2": "second", "image_3": "third", "weight_1": 0.8, "weight_2": 0.4, "weight_3": 0.6}
        _, calls = self._apply("snapref", values=values, SnapRef=("patched",))
        kwargs = calls[0]["kwargs"]
        self.assertEqual((kwargs["image_3"], kwargs["weight_3"], kwargs["weight_1"], kwargs["weight_2"]), ("third", 0.6, 0.8, 0.4))

    def test_a_third_reference_alone_still_sends_every_weight(self):
        _, calls = self._apply("snapref", values={**self.values, "image_3": "third", "weight_3": 0.6}, SnapRef=("patched",))
        kwargs = calls[0]["kwargs"]
        self.assertEqual((kwargs["image_3"], kwargs["weight_1"], kwargs["weight_3"]), ("third", 1.0, 0.6))
        self.assertNotIn("image_2", kwargs)

    def test_stitch_layout_goes_to_snapref_and_separate_stays_out(self):
        values = {**self.values, "image_2": "second"}
        _, calls = self._apply("snapref", values={**values, "multi_layout": "stitch"}, SnapRef=("patched",))
        self.assertEqual(calls[0]["kwargs"]["multi_layout"], "stitch")
        _, calls = self._apply("snapref", values={**values, "multi_layout": "separate"}, SnapRef=("patched",))
        self.assertNotIn("multi_layout", calls[0]["kwargs"])

    def test_the_layout_needs_more_than_one_picture(self):
        _, calls = self._apply("snapref", values={**self.values, "multi_layout": "stitch"}, SnapRef=("patched",))
        self.assertNotIn("multi_layout", calls[0]["kwargs"])

    def test_the_third_reference_and_layout_inputs_stay_optional(self):
        spec = nodes.AIStudioAnimaReference.INPUT_TYPES()
        self.assertEqual(spec["optional"]["image_3"][0], "IMAGE")
        self.assertEqual(spec["optional"]["weight_3"][0], "FLOAT")
        self.assertEqual(list(spec["optional"]["multi_layout"][0]), ["separate", "stitch"])
        # new inputs go after the old ones, so a saved workflow's widget values keep their places
        self.assertEqual(list(spec["optional"])[-3:], ["image_3", "weight_3", "multi_layout"])

    def test_snapref_hands_its_node_the_layouts_it_maps_itself(self):
        _, calls = self._apply("snapref", remap={"sub/ref.safetensors": "sub/ref__40b.safetensors"},
                               native=frozenset({(28, 40)}), SnapRef=("patched",))
        self.assertEqual(self.native_seen, [frozenset({(28, 40)})])
        self.assertEqual(calls[0]["kwargs"]["adapter"], "sub/ref.safetensors")

    def test_ref_adapter_still_gets_the_remapped_copy(self):
        _, calls = self._apply("ref_adapter", remap={"sub/ref.safetensors": "sub/ref__40b.safetensors"},
                               native=frozenset({(28, 40)}), AnimaRefAdapter=(FakeWrapped(),))
        self.assertEqual(self.native_seen, [frozenset()])
        self.assertEqual(calls[0]["kwargs"]["lora_name"], "sub/ref__40b.safetensors")

    def test_snapref_arguments(self):
        result, calls = self._apply("snapref", SnapRef=("patched",))
        self.assertEqual(result, ("patched",))
        self.assertEqual(calls[0]["name"], "SnapRef")
        self.assertEqual(calls[0]["kwargs"], {
            "model": "m", "clip": "c", "vae": "v", "image": "i", "adapter": "sub/ref.safetensors", "mode": "char",
            "ref_in_uncond": False, "ref_megapixels": 1.25, "strength": 0.5, "ref_caption": "1girl",
            "ref_guidance_on": True, "ref_guidance": 4.5, "correction": "apg", "ref_end": 0.6})

    def test_fidelity_below_one_goes_to_snapref(self):
        _, calls = self._apply("snapref", values={**self.values, "guidance_on": False, "fidelity": 0.6}, SnapRef=(FakeWrapped(),))
        self.assertEqual(calls[0]["kwargs"]["fidelity"], 0.6)

    def test_a_second_reference_goes_with_both_weights(self):
        values = {**self.values, "image_2": "second", "weight_1": 0.8, "weight_2": 0.4}
        _, calls = self._apply("snapref", values=values, SnapRef=("patched",))
        kwargs = calls[0]["kwargs"]
        self.assertEqual((kwargs["image_2"], kwargs["weight_1"], kwargs["weight_2"]), ("second", 0.8, 0.4))

    def test_weights_stay_out_without_a_second_reference(self):
        _, calls = self._apply("snapref", values={**self.values, "weight_1": 0.8, "weight_2": 0.4}, SnapRef=("patched",))
        self.assertFalse({"weight_1", "weight_2", "image_2"} & set(calls[0]["kwargs"]))

    def test_ref_adapter_takes_neither_second_image_nor_fidelity(self):
        values = {**self.values, "image_2": "second", "weight_2": 0.4, "fidelity": 0.5}
        _, calls = self._apply("ref_adapter", values=values, AnimaRefAdapter=(FakeWrapped(),))
        self.assertFalse({"image_2", "weight_2", "fidelity"} & set(calls[0]["kwargs"]))

    def test_an_old_snapref_node_gets_a_clear_message_for_the_new_settings(self):
        def old_node(_call):
            raise TypeError("SnapRef.apply() got an unexpected keyword argument 'fidelity'")
        with self.assertRaisesRegex(RuntimeError, "옛 버전"):
            self._apply("snapref", values={**self.values, "fidelity": 0.6}, SnapRef=old_node)

    def test_a_typeerror_from_snapref_itself_is_not_blamed_on_its_version(self):
        def broken(_call):
            raise TypeError("unsupported operand")
        with self.assertRaisesRegex(TypeError, "unsupported operand"):
            self._apply("snapref", values={**self.values, "fidelity": 0.6}, SnapRef=broken)

    def test_the_second_reference_input_stays_optional(self):
        spec = nodes.AIStudioAnimaReference.INPUT_TYPES()
        self.assertEqual(spec["optional"]["image_2"][0], "IMAGE")
        self.assertNotIn("image_2", spec["required"])

    def test_ref_adapter_arguments(self):
        result, calls = self._apply("ref_adapter", AnimaRefAdapter=("patched",))
        self.assertEqual(result, ("patched",))
        self.assertEqual(calls[0]["name"], "AnimaRefAdapter")
        self.assertEqual(calls[0]["kwargs"], {
            "model": "m", "vae": "v", "image": "i", "lora_name": "sub/ref.safetensors", "ref_guidance": 4.5,
            "correction": "apg", "strength": 0.5, "ref_end": 0.6, "ref_megapixels": 1.25})

    def test_empty_file_is_refused(self):
        with self.assertRaisesRegex(ValueError, "레퍼런스 파일"):
            nodes.AIStudioAnimaReference().apply(method="snapref", **{**self.values, "file": ""})

    def test_the_adapter_follows_the_models_block_layout(self):
        remap = {"sub/ref.safetensors": "sub/ref__40b.safetensors"}
        _, calls = self._apply("snapref", remap=remap, SnapRef=("patched",))
        self.assertEqual(calls[0]["kwargs"]["adapter"], "sub/ref__40b.safetensors")
        _, calls = self._apply("ref_adapter", remap=remap, AnimaRefAdapter=(FakeWrapped(),))
        self.assertEqual(calls[0]["kwargs"]["lora_name"], "sub/ref__40b.safetensors")

    def test_snapref_without_its_guidance_gets_the_batch_splitter_outermost(self):
        patcher = FakeWrapped()
        result, _ = self._apply("snapref", values={**self.values, "guidance_on": False}, SnapRef=(patcher,))
        self.assertEqual(list(result[0].wrappers["diffusion_model"]), [nodes.SPLIT_WRAPPER_KEY, "snapref", "negpip"])
        self.assertEqual(list(patcher.wrappers["diffusion_model"]), ["snapref", "negpip"])  # the input MODEL is untouched

    def test_snapref_with_its_guidance_needs_no_splitter(self):
        patcher = FakeWrapped()
        result, _ = self._apply("snapref", SnapRef=(patcher,))
        self.assertIs(result[0], patcher)

    def test_a_v2_adapter_marks_the_model_for_the_guidance_suite(self):
        result, _ = self._apply("ref_adapter", asym=True, AnimaRefAdapter=(FakeWrapped(),))
        self.assertTrue(result[0].model_options["transformer_options"][ASYM_REFERENCE_KEY])
        plain, _ = self._apply("ref_adapter", asym=False, AnimaRefAdapter=(FakeWrapped(),))
        self.assertNotIn(ASYM_REFERENCE_KEY, plain[0].model_options["transformer_options"])


class FakeWrapped:
    def __init__(self, wrappers=None):
        self.wrappers = wrappers or {"diffusion_model": {"snapref": ["a"], "negpip": ["b"]}}
        self.model_options = {"transformer_options": {}}

    def clone(self):
        return FakeWrapped({kind: dict(found) for kind, found in self.wrappers.items()})

    def add_wrapper_with_key(self, kind, key, wrapper):
        self.wrappers[kind][key] = [wrapper]


@requires_torch
class SplitBatchTests(unittest.TestCase):
    def test_each_chunk_gets_its_own_rows_flag_and_uuid(self):
        torch = load_torch()
        calls = []

        def executor(x, timesteps, context, fps, padding_mask, **kwargs):
            calls.append((x.shape[0], kwargs["c_ppm_negpip_mask"].shape[0], kwargs["transformer_options"], fps, padding_mask, kwargs["shared"]))
            return x * 2

        x = torch.arange(4.0).reshape(2, 2)
        options = {"cond_or_uncond": [0, 1], "uuids": ["c", "u"], "other": 1}
        out = nodes._split_mixed_batch(executor, x, torch.zeros(2), torch.zeros(2, 3), "fps", "pad",
                                       c_ppm_negpip_mask=torch.ones(2, 5), shared=torch.ones(7), transformer_options=options)
        self.assertTrue(torch.equal(out, x * 2))
        self.assertEqual([call[:2] for call in calls], [(1, 1), (1, 1)])
        self.assertEqual([call[2]["cond_or_uncond"] for call in calls], [[0], [1]])
        self.assertEqual([call[2]["uuids"] for call in calls], [["c"], ["u"]])
        self.assertEqual(calls[0][3:5], ("fps", "pad"))
        self.assertEqual(calls[0][5].shape[0], 7)  # not a per-row tensor
        self.assertEqual(options["cond_or_uncond"], [0, 1])  # the shared dict stays

    def test_one_kind_of_row_passes_straight_through(self):
        torch = load_torch()
        seen = []

        def executor(x, *_args, **_kwargs):
            seen.append(x.shape[0])
            return x

        nodes._split_mixed_batch(executor, torch.zeros(2, 2), torch.zeros(2), torch.zeros(2, 3),
                                 transformer_options={"cond_or_uncond": [0, 0]})
        self.assertEqual(seen, [2])


class TextEncoderLoaderTests(unittest.TestCase):
    def test_only_moss_takes_the_own_path(self):
        calls = ProviderCalls(CLIPLoader=("clip",))
        with mock.patch.object(nodes, "invoke_provider", calls), mock.patch.object(nodes, "load_moss_te") as moss:
            self.assertEqual(nodes.AIStudioTextEncoderLoader().load("te.safetensors", "krea2"), ("clip",))
            moss.assert_not_called()
        self.assertEqual(calls.calls[0]["args"], ("te.safetensors", "krea2"))

    def test_moss_reads_the_full_path(self):
        folders = mock.Mock()
        folders.get_full_path_or_raise.return_value = "D:/te.safetensors"
        with mock.patch.object(nodes, "folder_paths_module", return_value=folders), \
                mock.patch.object(nodes, "load_moss_te", return_value="clip") as moss:
            self.assertEqual(nodes.AIStudioTextEncoderLoader().load("te.safetensors", "moss_image"), ("clip",))
        moss.assert_called_once_with("D:/te.safetensors")

    def test_bare_keys_get_the_prefixes_core_detects(self):
        utils = mock.Mock()
        utils.load_torch_file.return_value = ({"language_model.a": 1}, {})
        utils.convert_old_quants.side_effect = lambda state, model_prefix, metadata: (state, metadata)
        with mock.patch.object(nodes.importlib, "import_module", return_value=utils):
            nodes._read_moss_te("x")
        utils.state_dict_prefix_replace.assert_called_once_with(
            {"language_model.a": 1}, {"language_model.": "model.language_model.", "visual.": "model.visual."})


class Krea2ReferenceTests(unittest.TestCase):
    def test_bypass_order_and_lazy_prompts(self):
        spec = nodes.AIStudioKrea2Reference.INPUT_TYPES()["required"]
        self.assertEqual(list(spec)[:3], ["model", "positive", "negative"])  # a bypassed output takes its own slot
        self.assertEqual(nodes.AIStudioKrea2Reference.RETURN_TYPES, ("MODEL", "CONDITIONING", "CONDITIONING"))
        self.assertTrue(spec["positive"][1]["lazy"] and spec["negative"][1]["lazy"])
        self.assertEqual(nodes.AIStudioKrea2Reference().check_lazy_status(positive=None, negative=None), [])

    def _apply(self, **changes):
        values = {"model": "m", "clip": "c", "vae": "v", "image": "i", "text": "prompt", "negative_text": "bad",
                  "file": "krea2/QuadView_krea2_v1.safetensors", "strength": 0.9, "grounding_px": 1024, "boost": 1.5,
                  "position": "beside (right of target)", "width": 1536, "height": 1024, **changes}
        calls = ProviderCalls(LoraLoaderModelOnly=("lora model",), Krea2EditModelPatch=("edit model",),
                              Krea2EditGroundedEncode=lambda index: (f"cond {index}",),
                              ConditioningSetTimestepRange=lambda index: ([f"range {index}"],))
        with mock.patch.object(nodes, "invoke_provider", calls), mock.patch.object(studio_nodes, "invoke_provider", calls), \
                mock.patch.object(nodes, "require_torch", return_value=mock.Mock(zeros=fake_torch_zeros)):
            return nodes.AIStudioKrea2Reference().apply(**values), calls.calls

    def test_lora_then_patch_then_both_prompts(self):
        result, calls = self._apply()
        self.assertEqual(result, ("edit model", "cond 3", "cond 4"))
        self.assertEqual([call["name"] for call in calls],
                         ["LoraLoaderModelOnly", "Krea2EditModelPatch", "Krea2EditGroundedEncode", "Krea2EditGroundedEncode"])
        self.assertEqual(calls[0]["args"], ("m", "krea2/QuadView_krea2_v1.safetensors", 0.9))
        patch = calls[1]["kwargs"]
        self.assertEqual(patch["model"], "lora model")
        self.assertEqual(patch["source_image"], "i")
        self.assertEqual(patch["vae"], "v")
        self.assertEqual(patch["fit_mode"], "fit")
        self.assertEqual((patch["ref_boost"], patch["ref_position"]), (1.5, "beside (right of target)"))
        self.assertEqual(patch["target_latent"]["samples"], (1, 16, 128, 192))  # the output size, in latent cells
        self.assertEqual(calls[2]["kwargs"], {"clip": "c", "prompt": "prompt", "image": "i", "grounding_px": 1024})
        self.assertEqual(calls[3]["kwargs"], {"clip": "c", "prompt": "bad", "image": "i", "grounding_px": 768})

    def test_empty_file_is_refused(self):
        with self.assertRaisesRegex(ValueError, "레퍼런스 파일"):
            self._apply(file="")

    def test_forge_prompt_editing_is_scheduled_like_the_plain_encode(self):
        _, calls = self._apply(text="a [cat:dog:0.5]", negative_text="", steps=10, forge_syntax=True)
        with_prompt = [call["kwargs"]["prompt"] for call in calls if call["name"] == "Krea2EditGroundedEncode"]
        self.assertEqual(with_prompt, ["a cat", "a dog", ""])
        ranges = [call["args"][1:] for call in calls if call["name"] == "ConditioningSetTimestepRange"]
        self.assertEqual(ranges, [(0.0, 0.5), (0.5, 1.0)])

    def test_prompt_editing_off_passes_the_text_as_it_is(self):
        _, calls = self._apply(text="a [cat:dog:0.5]", negative_text="", steps=10, forge_syntax=False)
        self.assertEqual([call["kwargs"]["prompt"] for call in calls if call["name"] == "Krea2EditGroundedEncode"],
                         ["a [cat:dog:0.5]", ""])


class FakeFlux:
    def __init__(self, config=None):
        self.config = config
        self.shift = None

    def set_parameters(self, shift=1.15):
        self.shift = shift


FakeFlux.__name__ = "ModelSamplingFlux"


class FakePatcher:
    def __init__(self, sampling):
        self.sampling = sampling
        self.model = mock.Mock(model_config="config")
        self.patches = {}

    def get_model_object(self, name):
        return self.patches.get(name, self.sampling) if name == "model_sampling" else None

    def clone(self):
        return FakePatcher(self.sampling)

    def add_object_patch(self, name, value):
        self.patches[name] = value


class FlowShiftTests(unittest.TestCase):
    def test_flux_type_sampling_keeps_its_class(self):
        class Advanced(FakeFlux):  # ComfyUI mixes the predictor into a local class, the core model_config one too
            pass

        original = Advanced()
        model = FakePatcher(original)
        with mock.patch.object(generation, "invoke_provider") as provider:
            patched = generation._patch_flow_shift(model, 1.5)
        provider.assert_not_called()  # ModelSamplingSD3 would multiply Krea2's timesteps by 1000
        sampling = patched.get_model_object("model_sampling")
        self.assertIsInstance(sampling, Advanced)
        self.assertEqual((sampling.config, sampling.shift), ("config", 1.5))
        self.assertIsNot(sampling, original)

    def test_discrete_flow_keeps_its_multiplier(self):
        class Discrete:
            multiplier = 1.0

        patched = FakePatcher(Discrete())
        with mock.patch.object(generation, "invoke_provider", return_value=(patched,)) as provider:
            generation._patch_flow_shift(patched, 5.0)
        self.assertEqual(provider.call_args.kwargs["args"][1:], (5.0, 1.0))


class ReferenceGuardTests(unittest.TestCase):
    def test_snapref_refuses_pag(self):
        model = mock.Mock(wrappers={"diffusion_model": {"snapref": []}}, model_options={})
        with self.assertRaisesRegex(RuntimeError, "PAG"):
            guidance._reference_guard(model, "pag", False, False)
        self.assertEqual(guidance._reference_guard(model, "seg", True, True), ("seg", True, True))

    def test_pag_at_scale_zero_installs_nothing_to_refuse(self):
        model = mock.Mock(wrappers={"diffusion_model": {"snapref": []}}, model_options={})
        self.assertEqual(guidance._reference_guard(model, "pag", False, False, 0.0), ("pag", False, False))

    def test_a_v2_adapter_loses_seg_slg_and_dave_but_keeps_pag(self):
        model = mock.Mock(wrappers={}, model_options={"transformer_options": {ASYM_REFERENCE_KEY: True}})
        self.assertEqual(guidance._reference_guard(model, "seg", True, True), (None, False, False))
        self.assertEqual(guidance._reference_guard(model, "pag", True, False), ("pag", False, False))

    def test_a_plain_model_keeps_everything(self):
        model = mock.Mock(wrappers={}, model_options={"transformer_options": {}})
        self.assertEqual(guidance._reference_guard(model, "pag", True, True), ("pag", True, True))


class FakeMossPatcher:
    model = type("MOSSImage", (), {})()


class MossLoraLayoutTests(unittest.TestCase):
    def _prepare(self, patches, layout=None):
        lora = mock.Mock(model_lora_keys_unet=mock.Mock(return_value={}), load_lora=mock.Mock(return_value=patches))
        convert = mock.Mock(convert_lora=lambda state: state)
        modules = {"comfy.lora": lora, "comfy.lora_convert": convert}
        with mock.patch.object(moss_lora_layout.importlib, "import_module", side_effect=modules.__getitem__), \
                mock.patch.object(moss_lora_layout, "_toolkit_layout", return_value=layout):
            return moss_lora_layout.prepare_moss_lora(FakeMossPatcher(), None, {"k": 1}, "name")

    def test_detects_the_core_model(self):
        self.assertTrue(moss_lora_layout.is_moss_model(FakeMossPatcher()))
        self.assertFalse(moss_lora_layout.is_moss_model(object()))

    def test_a_lora_that_patches_nothing_stops_the_run(self):
        with self.assertRaisesRegex(ValueError, "name"):
            self._prepare({})

    def test_a_text_encoder_only_file_counts(self):
        lora = mock.Mock(model_lora_keys_unet=mock.Mock(return_value={}), model_lora_keys_clip=mock.Mock(return_value={"te": 1}),
                         load_lora=mock.Mock(return_value={"patch": 1}))
        modules = {"comfy.lora": lora, "comfy.lora_convert": mock.Mock(convert_lora=lambda state: state)}
        clip = mock.Mock()
        with mock.patch.object(moss_lora_layout.importlib, "import_module", side_effect=modules.__getitem__), \
                mock.patch.object(moss_lora_layout, "_toolkit_layout", return_value=None):
            moss_lora_layout.prepare_moss_lora(FakeMossPatcher(), clip, {"k": 1}, "name")
        lora.model_lora_keys_clip.assert_called_once_with(clip.cond_stage_model, {})
        self.assertEqual(lora.load_lora.call_args.args[1], {"te": 1})

    def test_the_toolkit_converts_when_installed(self):
        layout = mock.Mock(model_is_fused=mock.Mock(return_value=False), to_model_layout=mock.Mock(return_value={"new": 1}))
        self.assertEqual(self._prepare({"patch": 1}, layout), {"new": 1})
        layout.to_model_layout.assert_called_once()

    def test_without_the_toolkit_the_file_goes_in_unchanged(self):
        self.assertEqual(self._prepare({"patch": 1}), {"k": 1})


class ReferenceLayoutTests(unittest.TestCase):
    def test_every_target_block_gets_its_lineage_source(self):
        keys = [f"anima_ref_kv.tgt.blocks.{index}.self_attn.k_proj.lora_down" for index in range(28)] + ["anima_ref_kv.mode_embed"]
        pairs = dict(reference_layout.lineage_keys(keys, 28, 40))
        self.assertEqual(len(pairs), 41)
        self.assertEqual(pairs["anima_ref_kv.mode_embed"], "anima_ref_kv.mode_embed")
        mapping = anima_lora.BLOCK_MAPPINGS[(28, 40)]
        for target, source in enumerate(mapping):
            new = f"anima_ref_kv.tgt.blocks.{target}.self_attn.k_proj.lora_down"
            self.assertEqual(pairs[new], f"anima_ref_kv.tgt.blocks.{source}.self_attn.k_proj.lora_down")

    def test_only_the_block_number_changes(self):
        pairs = dict(reference_layout.lineage_keys(["diffusion_model.blocks.27.cross_attn.k_proj.lora_up.weight"], 28, 52))
        self.assertTrue(all(new.endswith(".cross_attn.k_proj.lora_up.weight") for new in pairs))
        self.assertGreaterEqual(len(pairs), 1)

    def test_copy_name_stays_in_the_originals_folder(self):
        self.assertEqual(Path(reference_layout.remapped_name("LoRA/a/ref.safetensors", 40)), Path("LoRA/a/ref__40b.safetensors"))

    def _model(self, blocks):
        return mock.Mock(get_model_object=lambda _name: mock.Mock(blocks=[None] * blocks))

    def _for_model(self, blocks, file_blocks, source, native=frozenset()):
        folders = mock.Mock(get_full_path_or_raise=lambda _kind, _name: str(source))

        def write(_source, target, *_blocks):
            target.write_bytes(b"y")

        with mock.patch.object(reference_layout, "folder_paths_module", return_value=folders), \
                mock.patch.object(reference_layout, "_file_blocks", return_value=file_blocks), \
                mock.patch.object(reference_layout, "_write_remapped", side_effect=write) as writer:
            return reference_layout.adapter_for_model(self._model(blocks), "ref.safetensors", native=native), writer

    def test_a_layout_the_node_maps_itself_keeps_the_original(self):
        # SnapRef v2a's node fits a 28-block adapter to 2.9B on its own (11-10): no __40b copy
        name, writer = self._for_model(40, 28, "ref.safetensors", native=frozenset({(28, 40)}))
        self.assertEqual(name, "ref.safetensors")
        writer.assert_not_called()

    def test_native_layouts_do_not_cover_3_8b(self):
        with tempfile.TemporaryDirectory() as folder:
            source = Path(folder) / "ref.safetensors"
            source.write_bytes(b"x")
            name, writer = self._for_model(52, 28, source, native=frozenset({(28, 40)}))
            self.assertEqual(name, "ref__52b.safetensors")
            self.assertEqual(writer.call_count, 1)

    def test_a_file_on_its_own_layout_is_kept(self):
        for blocks in (28, 40, 52):
            name, writer = self._for_model(blocks, blocks, "ref.safetensors")
            self.assertEqual(name, "ref.safetensors")
            writer.assert_not_called()

    def test_a_published_2_9b_conversion_keeps_its_sparse_layout(self):
        kept = [index for index in range(40) if anima_lora.BLOCK_MAPPINGS[(28, 40)].index(anima_lora.BLOCK_MAPPINGS[(28, 40)][index]) == index]
        keys = [f"blocks.{index}.self_attn.k_proj.lora_down" for index in kept]
        self.assertEqual(len(keys), 28)
        self.assertEqual(reference_layout.layout_blocks(keys), 40)
        pairs = dict(reference_layout.lineage_keys(keys, 40, 52))
        self.assertEqual(set(pairs.values()), set(keys))

    def test_an_unknown_layout_is_refused(self):
        with self.assertRaisesRegex(ValueError, "블록 수"):
            self._for_model(40, 12, "ref.safetensors")

    def test_the_copy_follows_the_original_in_either_direction(self):
        with tempfile.TemporaryDirectory() as folder:
            source = Path(folder) / "ref.safetensors"
            source.write_bytes(b"x")
            name, writer = self._for_model(40, 28, source)
            self.assertEqual(name, "ref__40b.safetensors")
            self.assertEqual(writer.call_count, 1)
            self.assertEqual(self._for_model(40, 28, source)[1].call_count, 0)
            old = time.time() - 86400
            os.utime(source, (old, old))  # Explorer keeps a copied file's date: older than the copy, still another file
            self.assertEqual(self._for_model(40, 28, source)[1].call_count, 1)
            self.assertEqual(self._for_model(40, 28, source)[1].call_count, 0)


class ReferenceDocsTests(unittest.TestCase):
    def test_every_provider_is_named_in_the_parity_notes(self):
        notes = (Path(generation.__file__).parent / "studio_forge_parity.md").read_text(encoding="utf-8")
        for name in ("ComfyUI-MOSS-Toolkit", "SnapRef", "ComfyUI-Anima-Ref-Adapter", "comfyui-krea2edit"):
            with self.subTest(name=name):
                self.assertIn(name, notes)
        self.assertTrue(json.dumps(notes))


class AttentionNodeTests(unittest.TestCase):
    class Model:
        def __init__(self):
            self.attention = None

        def clone(self):
            return type(self)()

        def set_model_optimized_attention(self, function):
            self.attention = function

    def module(self, known):
        return mock.Mock(get_attention_function=lambda name, default=None: known.get(name, default))

    def test_launch_leaves_the_model_alone(self):
        model = self.Model()
        self.assertIs(nodes.AIStudioAttention().patch(model, "launch")[0], model)

    def test_a_backend_goes_on_a_clone(self):
        model = self.Model()
        with mock.patch.object(nodes.importlib, "import_module", return_value=self.module({"pytorch": "function"})) as loader:
            patched = nodes.AIStudioAttention().patch(model, "pytorch")[0]
        loader.assert_called_once_with("comfy.ldm.modules.attention")
        self.assertIsNot(patched, model)
        self.assertEqual((patched.attention, model.attention), ("function", None))

    def test_a_backend_this_comfyui_lacks_stops_the_run(self):
        with mock.patch.object(nodes.importlib, "import_module", return_value=self.module({})):
            with self.assertRaisesRegex(ValueError, "sage"):
                nodes.AIStudioAttention().patch(self.Model(), "sage")

    def test_the_mode_list_is_the_states(self):
        self.assertEqual(tuple(nodes.AIStudioAttention.INPUT_TYPES()["required"]["mode"][0]), nodes.ATTENTION_MODES)


if __name__ == "__main__":
    unittest.main()
