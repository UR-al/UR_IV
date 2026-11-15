"""Prompt adherence controls of the Anima studio (pack 1.17.0): LoRA control, CFG schedule, word emphasis."""
import math
import sys
import types
import unittest
from types import SimpleNamespace
from unittest import mock

from comfy_custom_nodes.ai_studio_forge_parity import generation
from comfy_custom_nodes.ai_studio_forge_parity import studio_adherence as sa
from comfy_custom_nodes.ai_studio_forge_parity import studio_adherence_nodes as nodes
from tests._optional_deps import bind_torch, requires_torch

CROSS = "diffusion_model.blocks.0.cross_attn.k_proj.weight"
CROSS_MOD = "diffusion_model.blocks.0.adaln_modulation_cross_attn.1.weight"
SELF = "diffusion_model.blocks.0.self_attn.q_proj.weight"
MLP = "diffusion_model.blocks.0.mlp.layer1.weight"
ADAPTER = "diffusion_model.llm_adapter.blocks.0.cross_attn.q_proj.weight"
ADAPTER_SELF = "diffusion_model.llm_adapter.blocks.0.self_attn.q_proj.weight"  # the adapter is all text path


class FakePatcher:
    """The parts of ComfyUI's ModelPatcher the nodes use."""

    def __init__(self, patches=None, model_options=None, model=None):
        self.patches = {key: list(items) for key, items in (patches or {}).items()}
        self.model_options = model_options if model_options is not None else {"transformer_options": {}}
        self.patches_uuid = "original"
        self.wrappers = {}
        self.object_patches = {}
        self.model = model

    def clone(self):
        other = FakePatcher(self.patches, {key: (dict(value) if isinstance(value, dict) else value)
                                           for key, value in self.model_options.items()}, self.model)
        other.patches_uuid = self.patches_uuid
        other.wrappers = {kind: {key: list(fns) for key, fns in by_key.items()} for kind, by_key in self.wrappers.items()}
        other.object_patches = dict(self.object_patches)
        return other

    def add_wrapper_with_key(self, kind, key, wrapper):
        self.wrappers.setdefault(kind, {}).setdefault(key, []).append(wrapper)

    def add_object_patch(self, name, obj):
        self.object_patches[name] = obj

    def set_model_attn2_patch(self, patch):
        patches = self.model_options.setdefault("transformer_options", {}).setdefault("patches", {})
        patches.setdefault("attn2_patch", []).append(patch)

    def get_model_object(self, name):
        return self.object_patches.get(name, getattr(self.model, name, None))


WRAPPERS = SimpleNamespace(PREDICT_NOISE="predict_noise", APPLY_MODEL="apply_model")


# ---------------------------------------------------------------------------------------------- pure rules

class CrossAttnKeyTests(unittest.TestCase):
    def test_modes(self):
        only, with_mod = sa.CROSS_ATTN_ONLY, sa.CROSS_ATTN_WITH_MOD
        for key, expected in ((CROSS, (False, True, True)), (CROSS_MOD, (False, False, True)), (SELF, (False, False, False)),
                              (MLP, (False, False, False)), (ADAPTER, (False, True, True)),
                              (ADAPTER_SELF, (False, True, True))):
            with self.subTest(key=key):
                self.assertEqual(tuple(sa.cross_attn_key(key, mode) for mode in (sa.CROSS_ATTN_OFF, only, with_mod)), expected)

    def test_a_bare_key_also_counts(self):
        self.assertTrue(sa.cross_attn_key("blocks.3.cross_attn.v_proj.weight", sa.CROSS_ATTN_ONLY))


class SpanTests(unittest.TestCase):
    def test_spans_are_the_tail_the_lora_nodes_added(self):
        before = {"a": ["old"], "c": ["kept"]}
        after = {"a": ["old", "l1", "l2"], "b": ["l3"], "c": ["kept"]}
        self.assertEqual(sa.lora_spans(before, after), {"a": (1, 3), "b": (0, 1)})

    def test_without_spans_drops_empty_keys(self):
        patches = {"a": ["old", "l1", "l2", "later"], "b": ["l3"], "c": ["kept"]}
        self.assertEqual(sa.without_spans(patches, {"a": (1, 3), "b": (0, 1)}), {"a": ["old", "later"], "c": ["kept"]})

    def test_drop_cross_attn_keeps_what_was_there_before(self):
        patches = {CROSS: ["base", "lora"], SELF: ["lora"], CROSS_MOD: ["lora"]}
        spans = {CROSS: (1, 2), SELF: (0, 1), CROSS_MOD: (0, 1)}
        out, left, keys = sa.drop_cross_attn(patches, spans, sa.CROSS_ATTN_ONLY)
        self.assertEqual(out, {CROSS: ["base"], SELF: ["lora"], CROSS_MOD: ["lora"]})
        self.assertEqual(left, {SELF: (0, 1), CROSS_MOD: (0, 1)})
        self.assertEqual(keys, 1)
        out, left, keys = sa.drop_cross_attn(patches, spans, sa.CROSS_ATTN_WITH_MOD)
        self.assertEqual(out, {CROSS: ["base"], SELF: ["lora"]})
        self.assertEqual((left, keys), ({SELF: (0, 1)}, 2))
        self.assertEqual(patches[CROSS], ["base", "lora"])  # the input is left alone

    def test_drop_cross_attn_off_changes_nothing(self):
        patches, spans = {CROSS: ["lora"]}, {CROSS: (0, 1)}
        self.assertEqual(sa.drop_cross_attn(patches, spans, sa.CROSS_ATTN_OFF), (patches, spans, 0))


class StepTests(unittest.TestCase):
    def test_base_steps_rounds(self):
        self.assertEqual(sa.base_steps(30, 32), 10)
        self.assertEqual(sa.base_steps(0, 32), 0)
        self.assertEqual(sa.base_steps(5, 8), 0)  # 0.4 of a step: no split
        self.assertEqual(sa.base_steps(90, 10), 9)
        self.assertEqual(sa.base_steps(90, 5), 4)  # below 100 % the last step keeps the LoRAs
        self.assertEqual(sa.base_steps(75, 2), 1)
        self.assertEqual(sa.base_steps(100, 32), 32)
        self.assertEqual(sa.base_steps(150, 32), 32)

    def test_progress_by_schedule_position(self):
        sigmas = [1.0, 0.8, 0.5, 0.2, 0.0]
        self.assertEqual(sa.step_progress(1.0, sigmas), 0.0)
        self.assertEqual(sa.step_progress(0.8, sigmas), 0.25)
        self.assertEqual(sa.step_progress(0.65, sigmas), 0.25)  # between entries: the step it belongs to
        self.assertEqual(sa.step_progress(0.5, sigmas), 0.5)
        self.assertEqual(sa.step_progress(0.0, sigmas), 0.75)  # never past the last step

    def test_progress_without_a_schedule_is_the_start(self):
        self.assertEqual(sa.step_progress(0.5, None), 0.0)
        self.assertEqual(sa.step_progress(0.5, [1.0]), 0.0)


class CfgScheduleTests(unittest.TestCase):
    def test_curves(self):
        cos, line, step = sa.CFG_CURVES
        self.assertEqual(sa.scheduled_cfg(0.0, 3.0, 6.0, 40, cos), 6.0)
        self.assertAlmostEqual(sa.scheduled_cfg(0.2, 3.0, 6.0, 40, cos), 4.5)
        self.assertAlmostEqual(sa.scheduled_cfg(0.1, 3.0, 6.0, 40, line), 5.25)
        self.assertEqual(sa.scheduled_cfg(0.39, 3.0, 6.0, 40, step), 6.0)
        for curve in sa.CFG_CURVES:
            self.assertEqual(sa.scheduled_cfg(0.4, 3.0, 6.0, 40, curve), 3.0, curve)
            self.assertEqual(sa.scheduled_cfg(0.0, 3.0, 6.0, 0, curve), 3.0, curve)

    def test_window(self):
        self.assertTrue(sa.in_window(0.0, 0, 40))
        self.assertTrue(sa.in_window(0.39, 0, 40))
        self.assertFalse(sa.in_window(0.4, 0, 40))
        self.assertFalse(sa.in_window(0.1, 20, 40))
        self.assertTrue(sa.in_window(1.0, 50, 100))


class EmphasisRuleTests(unittest.TestCase):
    def test_phrases(self):
        self.assertEqual(sa.emphasis_phrases("gripping,  metal   bar\n\n gripping ,"), ["gripping", "metal bar"])
        self.assertEqual(sa.emphasis_phrases(""), [])

    def test_positions(self):
        self.assertEqual(sa.find_positions([5, 7, 8, 9, 7, 8, 1], [[7, 8]]), [1, 2, 4, 5])
        self.assertEqual(sa.find_positions([5, 7, 8, 9], [[9], [], [5, 7]]), [0, 1, 3])
        self.assertEqual(sa.find_positions([5], [[5, 6]]), [])

    def test_logit(self):
        self.assertAlmostEqual(sa.emphasis_logit(2.0), math.log(2.0))
        self.assertLess(sa.emphasis_logit(0.5), 0)


# ---------------------------------------------------------------------------------------------- nodes

class LoraControlTests(unittest.TestCase):
    def setUp(self):
        self.before = FakePatcher({SELF: ["base"]})
        self.after = FakePatcher({SELF: ["base", "lora"], CROSS: ["lora"], CROSS_MOD: ["lora"], MLP: ["lora"]})

    def test_off_returns_the_model(self):
        self.assertIs(nodes.AIStudioLoraControl().apply(self.after, self.before, 0, sa.CROSS_ATTN_OFF)[0], self.after)

    def test_cross_attn_drops_only_lora_entries(self):
        (out,) = nodes.AIStudioLoraControl().apply(self.after, self.before, 0, sa.CROSS_ATTN_WITH_MOD)
        self.assertEqual(out.patches, {SELF: ["base", "lora"], MLP: ["lora"]})
        self.assertNotEqual(out.patches_uuid, "original")
        self.assertNotIn(sa.LORA_SCHEDULE_KEY, out.model_options)
        self.assertIn(CROSS, self.after.patches)  # the input MODEL is untouched

    def test_base_first_leaves_the_split_for_the_sampler(self):
        (out,) = nodes.AIStudioLoraControl().apply(self.after, self.before, 30, sa.CROSS_ATTN_ONLY)
        schedule = nodes.lora_schedule(out)
        self.assertEqual(schedule["base_percent"], 30)
        self.assertEqual(schedule["spans"], {SELF: (1, 2), CROSS_MOD: (0, 1), MLP: (0, 1)})
        base = nodes.base_model_of(out, schedule)
        self.assertEqual(base.patches, {SELF: ["base"]})
        self.assertIsNone(nodes.lora_schedule(base))
        self.assertEqual(out.patches[MLP], ["lora"])  # the LoRA pass keeps them

    def test_no_lora_no_split(self):
        (out,) = nodes.AIStudioLoraControl().apply(self.before.clone(), self.before, 30, sa.CROSS_ATTN_OFF)
        self.assertIsNone(nodes.lora_schedule(out))

    def test_unknown_mode_is_refused(self):
        with self.assertRaises(ValueError):
            nodes.AIStudioLoraControl().apply(self.after, self.before, 0, "nope")

    def test_has_wrapper(self):
        model = FakePatcher()
        model.add_wrapper_with_key("sampler_sample", "ai_studio_forge_parity.speed", lambda *a: None)
        self.assertTrue(nodes.has_wrapper(model, "ai_studio_forge_parity.speed"))
        self.assertFalse(nodes.has_wrapper(model, "other"))


class Executor:
    """A WrapperExecutor stand-in: records the guider's CFG when called."""

    def __init__(self, cfg=3.0, fail=False):
        self.class_obj = SimpleNamespace(cfg=cfg)
        self.seen = []
        self.fail = fail

    def __call__(self, *args, **kwargs):
        self.seen.append((self.class_obj.cfg, args, kwargs))
        if self.fail:
            raise RuntimeError("boom")
        return "noise"


class CfgScheduleNodeTests(unittest.TestCase):
    options = {"transformer_options": {"sample_sigmas": [1.0, 0.75, 0.5, 0.25, 0.0]}}

    def test_wrapper_sets_and_restores_the_guider_cfg(self):
        wrapper = nodes.cfg_schedule_wrapper(6.0, 50, sa.CFG_CURVES[2])
        executor = Executor()
        self.assertEqual(wrapper(executor, "x", 1.0, self.options, 7), "noise")
        self.assertEqual(executor.seen[0][0], 6.0)
        self.assertEqual(executor.seen[0][1], ("x", 1.0, self.options, 7))
        self.assertEqual(executor.class_obj.cfg, 3.0)
        wrapper(executor, "x", 0.5, self.options, 7)  # 50 %: the sampler's own
        self.assertEqual(executor.seen[1][0], 3.0)

    def test_wrapper_restores_after_an_error(self):
        executor = Executor(fail=True)
        with self.assertRaises(RuntimeError):
            nodes.cfg_schedule_wrapper(6.0, 50, sa.CFG_CURVES[0])(executor, "x", 1.0, self.options, 7)
        self.assertEqual(executor.class_obj.cfg, 3.0)

    @requires_torch
    def test_a_republished_tensor_schedule_is_followed(self):
        from comfy_custom_nodes.ai_studio_forge_parity.guidance_common import mark_republished
        from tests._optional_deps import load_torch

        torch = load_torch()
        whole = torch.tensor([1.0, 0.75, 0.5, 0.25, 0.0])
        published = mark_republished(torch.tensor([1.0, 0.6, 0.0]), whole)  # SPEED's patched copy
        executor = Executor()
        nodes.cfg_schedule_wrapper(6.0, 50, sa.CFG_CURVES[2])(executor, "x", torch.tensor([0.75]),
                                                               {"transformer_options": {"sample_sigmas": published}}, 7)
        self.assertEqual(executor.seen[0][0], 6.0)  # step 1 of the run's own 4 < 50 %, no bool() of a tensor

    def test_node(self):
        model = FakePatcher()
        with mock.patch.object(nodes, "_wrapper_types", return_value=WRAPPERS):
            self.assertIs(nodes.AIStudioCFGSchedule().patch(model, 6.0, 0, sa.CFG_CURVES[0])[0], model)
            (out,) = nodes.AIStudioCFGSchedule().patch(model, 6.0, 40, sa.CFG_CURVES[0])
        self.assertIn(nodes.CFG_WRAPPER_KEY, out.wrappers["predict_noise"])
        self.assertEqual(model.wrappers, {})
        with self.assertRaises(ValueError):
            nodes.AIStudioCFGSchedule().patch(model, 6.0, 40, "zigzag")


class FakeTokenizer:
    end_token, pad_token = 1, 0
    vocabulary = {"gripping": [10], "metal": [11], "bar": [12], "bars": [12, 13]}

    def tokenize_with_weights(self, text):
        ids = [token for word in text.split() for token in self.vocabulary.get(word, [99])]
        return [[(token, 1.0) for token in ids + [self.end_token]]]


class EmphasisNodeTests(unittest.TestCase):
    def test_phrase_ids_drop_the_end_token(self):
        self.assertEqual(nodes.phrase_token_ids(["gripping", "metal bar"], FakeTokenizer()), [[10], [11, 12]])

    def test_nothing_to_do_returns_the_model(self):
        model = FakePatcher()
        for words, multiplier, start, end in (("", 2.0, 0, 40), ("a", 1.0, 0, 40), ("a", 2.0, 40, 40)):
            self.assertIs(nodes.AIStudioWordEmphasis().patch(model, words, multiplier, start, end)[0], model)

    def test_other_models_are_left_out(self):
        model = FakePatcher(model=SimpleNamespace())
        self.assertIs(nodes.AIStudioWordEmphasis().patch(model, "gripping", 2.0, 0, 40)[0], model)

    def test_node_patches_conds_wrapper_and_attention(self):
        class Anima:
            def extra_conds(self, **kwargs):
                return {}

        previous = object()
        model = FakePatcher(model=Anima(), model_options={"transformer_options": {"optimized_attention_override": previous}})
        with mock.patch.object(nodes, "_wrapper_types", return_value=WRAPPERS), \
                mock.patch.object(nodes, "_t5_tokenizer", return_value=FakeTokenizer()):
            (out,) = nodes.AIStudioWordEmphasis().patch(model, "gripping", 2.0, 0, 40)
        self.assertIn("extra_conds", out.object_patches)
        self.assertIn(nodes.EMPHASIS_WRAPPER_KEY, out.wrappers["apply_model"])
        self.assertEqual(out.model_options["transformer_options"]["patches"]["attn2_patch"], [nodes.emphasis_attn2_patch])
        override = out.model_options["transformer_options"]["optimized_attention_override"]
        self.assertIsNot(override, previous)
        self.assertIs(model.model_options["transformer_options"]["optimized_attention_override"], previous)


# ---------------------------------------------------------------------------------------------- torch parts

def _conds_stub():
    conds = types.ModuleType("comfy.conds")

    class CONDRegular:
        def __init__(self, cond):
            self.cond = cond

    conds.CONDRegular = CONDRegular
    return conds


@requires_torch
class EmphasisTorchTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        bind_torch(globals())

    def _extra(self, **kwargs):
        def previous(**_kwargs):
            return {"c_crossattn": SimpleNamespace(cond=torch.zeros(1, 512, 4))}

        with mock.patch.dict(sys.modules, {"comfy.conds": _conds_stub()}):
            return nodes.emphasis_extra_conds(previous, [[11, 12]], math.log(2.0))(**kwargs)

    def test_positive_conds_get_the_positions(self):
        out = self._extra(t5xxl_ids=torch.tensor([5, 11, 12, 9, 1]), prompt_type="positive")
        bias = out[nodes.EMPHASIS_COND_KEY].cond
        self.assertEqual(tuple(bias.shape), (1, 512))
        self.assertEqual(torch.nonzero(bias[0]).flatten().tolist(), [1, 2])
        self.assertAlmostEqual(float(bias[0, 1]), math.log(2.0), places=6)

    def test_negative_conds_get_zeros_so_they_still_batch(self):
        bias = self._extra(t5xxl_ids=torch.tensor([11, 12]), prompt_type="negative")[nodes.EMPHASIS_COND_KEY].cond
        self.assertEqual(float(bias.abs().sum()), 0.0)

    def test_conds_without_ids_or_from_the_38b_connector_get_nothing(self):
        self.assertNotIn(nodes.EMPHASIS_COND_KEY, self._extra(prompt_type="positive"))
        out = self._extra(t5xxl_ids=torch.tensor([11, 12]), prompt_type="positive", qwen35_gated_run=3)
        self.assertNotIn(nodes.EMPHASIS_COND_KEY, out)

    def _apply(self, bias, sigma, flags=(0, 1)):
        seen = {}

        def executor(x, t, c_concat, c_crossattn, control, transformer_options, **kwargs):
            seen.update(options=transformer_options, kwargs=kwargs)
            return "out"

        options = {"sample_sigmas": [1.0, 0.75, 0.5, 0.25, 0.0], "cond_or_uncond": list(flags)}
        wrapper = nodes.emphasis_apply_wrapper(0, 50)
        self.assertEqual(wrapper(executor, "x", torch.tensor([sigma]), None, "ctx", None, options,
                                 **{nodes.EMPHASIS_COND_KEY: bias, "other": 1}), "out")
        self.assertEqual(seen["kwargs"], {"other": 1})  # the offsets never reach the model
        return seen["options"]

    def test_apply_wrapper_hands_the_offsets_over_in_the_window(self):
        bias = torch.tensor([[0.0, 0.5], [0.0, 0.0]])
        entry = self._apply(bias, 1.0)[nodes.EMPHASIS_OPTION_KEY]
        self.assertIs(entry["rows"], bias)
        self.assertEqual(entry["chunks"][0].tolist(), [[0.0, 0.5]])
        self.assertEqual(entry["chunks"][1].tolist(), [[0.0, 0.0]])
        self.assertNotIn(nodes.EMPHASIS_OPTION_KEY, self._apply(bias, 0.5))  # 50 %: outside

    def test_apply_wrapper_skips_all_zero_offsets(self):
        self.assertNotIn(nodes.EMPHASIS_OPTION_KEY, self._apply(torch.zeros(2, 2), 1.0))

    @staticmethod
    def _attention_stub():
        module = types.ModuleType("comfy.ldm.modules.attention")

        def attention_pytorch(q, k, v, heads, mask=None, skip_reshape=False, **kwargs):
            out = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=mask)
            return out.transpose(1, 2).reshape(q.shape[0], -1, heads * q.shape[-1])

        module.attention_pytorch = attention_pytorch
        return module

    def _qkv(self, batch=2, queries=6, keys=4):
        generator = torch.Generator().manual_seed(0)
        q = torch.randn(batch, 2, queries, 8, generator=generator)
        k = torch.randn(batch, 2, keys, 8, generator=generator)
        v = torch.randn(batch, 2, keys, 8, generator=generator)
        return q, k, v

    def _override(self, entry, q, k, v, flags=(0, 1), previous=None, cross=True):
        calls = []
        if entry is not None and cross:  # Cosmos runs the attn2 patch in a cross-attention's compute_qkv
            nodes.emphasis_attn2_patch(q, k, v, extra_options={nodes.EMPHASIS_OPTION_KEY: entry})

        def func(*args, **kwargs):
            calls.append("func")
            return "plain"

        options = {nodes.EMPHASIS_OPTION_KEY: entry, "cond_or_uncond": list(flags)} if entry else {}
        override = nodes.emphasis_attention_override(previous)
        with mock.patch.dict(sys.modules, {"comfy.ldm.modules.attention": self._attention_stub()}):
            result = override(func, q, k, v, 2, skip_reshape=True, transformer_options=options)
        return result, calls

    def test_cross_attention_gets_the_offsets_as_a_mask(self):
        q, k, v = self._qkv()
        bias = torch.tensor([[0.0, math.log(3.0), 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]])
        result, calls = self._override({"rows": bias, "chunks": {}}, q, k, v)
        self.assertEqual(calls, [])
        weights = torch.softmax(q @ k.transpose(-1, -2) / math.sqrt(8) + bias[:, None, None, :], dim=-1)
        expected = (weights @ v).transpose(1, 2).reshape(2, 6, 16)
        self.assertTrue(torch.allclose(result, expected, atol=1e-5))
        # token 1 of the positive row now draws three times its old weight before renormalizing
        plain = torch.softmax(q @ k.transpose(-1, -2) / math.sqrt(8), dim=-1)
        ratio = (weights[0, ..., 1] / weights[0, ..., 0]) / (plain[0, ..., 1] / plain[0, ..., 0])
        self.assertTrue(torch.allclose(ratio, torch.full_like(ratio, 3.0), atol=1e-4))

    def test_self_attention_and_calls_outside_the_window_pass_on(self):
        q, k, v = self._qkv(keys=4)
        bias = torch.zeros(2, 4)
        bias[0, 1] = 1.0
        entry = {"rows": bias, "chunks": {}}
        # no attn2 patch ran before this call: a self-attention, even with as many tokens as the text
        self.assertEqual(self._override(entry, q, k, v, cross=False)[1], ["func"])
        self.assertEqual(self._override(entry, q, k, v)[1], [])
        self.assertEqual(self._override(entry, q, k, v, cross=False)[1], ["func"])  # the mark is used up
        q2, k2, v2 = self._qkv(queries=4, keys=4)  # 512x256-like: image tokens == text tokens
        self.assertEqual(self._override(entry, q2, k2, v2)[1], [])
        q, k, v = self._qkv()
        self.assertEqual(self._override(None, q, k, v)[1], ["func"])
        previous = mock.Mock(return_value="picked")
        result, calls = self._override(None, q, k, v, previous=previous)
        self.assertEqual((result, calls), ("picked", []))
        self.assertEqual(previous.call_args.args[1:4], (q, k, v))

    def test_a_split_batch_uses_its_chunk(self):
        q, k, v = self._qkv(batch=1)
        rows = torch.tensor([[0.0, 2.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]])
        entry = {"rows": rows, "chunks": {0: rows[:1], 1: rows[1:]}}
        result, calls = self._override(entry, q, k, v, flags=(0,))
        self.assertEqual(calls, [])
        weights = torch.softmax(q @ k.transpose(-1, -2) / math.sqrt(8) + rows[:1, None, None, :], dim=-1)
        self.assertTrue(torch.allclose(result, (weights @ v).transpose(1, 2).reshape(1, 6, 16), atol=1e-5))


def _sample_stub(calls):
    sample = types.ModuleType("comfy.sample")

    def run(model, noise, steps, cfg, sampler, scheduler, positive, negative, latent, **kwargs):
        calls.append({"model": model, "noise": noise, "latent": latent, **kwargs})
        return f"latent-{len(calls)}"

    def run_custom(model, noise, cfg, sampler, sigmas, positive, negative, latent, **kwargs):
        calls.append({"model": model, "noise": noise, "sampler": sampler, "sigmas": sigmas, "latent": latent, **kwargs})
        return f"latent-{len(calls)}"

    sample.sample, sample.sample_custom = run, run_custom
    return sample


def _samplers_stub(function, sigmas):
    samplers = types.ModuleType("comfy.samplers")

    class KSAMPLER:
        def __init__(self, sampler_function, extra_options={}, inpaint_options={}):
            self.sampler_function, self.extra_options, self.inpaint_options = sampler_function, extra_options, inpaint_options

    class KSampler:
        def __init__(self, model, steps, device, sampler=None, scheduler=None, denoise=None, model_options={}):
            self.sigmas = sigmas

    samplers.KSAMPLER, samplers.KSampler = KSAMPLER, KSampler
    samplers.sampler_object = lambda name: KSAMPLER(function, {"eta": 1.0})
    return samplers


def _kd_stub(made):
    kd = types.ModuleType("comfy.k_diffusion.sampling")

    def default_noise_sampler(x, seed=None):
        made.append(("default", seed))
        return f"stream-{seed}"

    class BrownianTreeNoiseSampler:
        def __init__(self, x, sigma_min, sigma_max, seed=None, cpu=False):
            made.append(("tree", float(sigma_min), float(sigma_max), seed, cpu))

    kd.default_noise_sampler, kd.BrownianTreeNoiseSampler = default_noise_sampler, BrownianTreeNoiseSampler
    return kd


@requires_torch
class SplitSampleTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        bind_torch(globals())

    def setUp(self):
        self.model = FakePatcher({MLP: ["lora"], SELF: ["base", "lora"]})
        self.model.load_device = "cpu"
        self.schedule = {"base_percent": 25, "spans": {MLP: (0, 1), SELF: (1, 2)}}
        self.sigmas = torch.tensor([1.0, 0.8, 0.6, 0.4, 0.2, 0.0])

    @staticmethod
    def _function():
        seen = []

        def sample_test(model, x, sigmas, extra_args=None, callback=None, disable=None, noise_sampler=None,
                        sam_extra_step_offset=0):
            seen.append({"x": x, "sigmas": [round(float(v), 4) for v in sigmas], "noise_sampler": noise_sampler,
                         "offset": sam_extra_step_offset})
            if callback is not None:
                callback({"i": 0, "x": x, "denoised": x})
            return f"after-{len(sigmas) - 1}"

        return sample_test, seen

    def test_both_runs_see_the_whole_schedule_and_cut_inside(self):
        function, seen = self._function()
        made, ticks = [], []
        first, second = nodes.split_sampler_functions(function, 2, 5)
        with mock.patch.dict(sys.modules, {"comfy.k_diffusion.sampling": _kd_stub(made)}):
            out1 = first("m", "x0", self.sigmas, extra_args={"seed": 9}, callback=lambda info: ticks.append(info["i"]))
            out2 = second("m", "ignored", self.sigmas, extra_args={"seed": 9},
                          callback=lambda info: ticks.append(info["i"]))
        self.assertEqual((out1, out2), ("after-2", "after-3"))
        one, two = seen
        self.assertEqual(one["sigmas"], [1.0, 0.8, 0.6])
        self.assertEqual(two["sigmas"], [0.6, 0.4, 0.2, 0.0])
        self.assertEqual(two["x"], "after-2")  # the base run's latent, not the second run's own start
        self.assertEqual((one["noise_sampler"], two["noise_sampler"]), ("stream-9", "stream-9"))  # one stream
        self.assertEqual(made, [("default", 9)])
        self.assertEqual((one["offset"], two["offset"]), (0, 2))  # Dy sub-steps keep their place
        self.assertEqual(ticks, [0, 2])  # the preview counts the whole schedule

    def test_brownian_samplers_get_one_tree_over_the_whole_schedule(self):
        def sample_dpmpp_sde(model, x, sigmas, extra_args=None, callback=None, disable=None, noise_sampler=None):
            return x

        made = []
        first, second = nodes.split_sampler_functions(sample_dpmpp_sde, 2, 5)
        with mock.patch.dict(sys.modules, {"comfy.k_diffusion.sampling": _kd_stub(made)}):
            first("m", "x0", self.sigmas, extra_args={"seed": 4})
            second("m", "x", self.sigmas, extra_args={"seed": 4})
        self.assertEqual(len(made), 1)
        kind, low, high, seed, cpu = made[0]
        self.assertEqual((kind, seed, cpu), ("tree", 4, True))
        self.assertAlmostEqual(low, 0.2, places=6)
        self.assertAlmostEqual(high, 1.0, places=6)

    def test_a_given_noise_sampler_is_kept_across_the_cut(self):
        function, seen = self._function()
        first, second = nodes.split_sampler_functions(function, 2, 5)
        with mock.patch.dict(sys.modules, {"comfy.k_diffusion.sampling": _kd_stub([])}):
            first("m", "x0", self.sigmas, noise_sampler="coloured")
            second("m", "x", self.sigmas, noise_sampler="fresh")
        self.assertEqual([item["noise_sampler"] for item in seen], ["coloured", "coloured"])

    def test_a_schedule_of_another_length_is_refused(self):
        function, _seen = self._function()
        first, _second = nodes.split_sampler_functions(function, 2, 8)
        with mock.patch.dict(sys.modules, {"comfy.k_diffusion.sampling": _kd_stub([])}), \
                self.assertRaisesRegex(RuntimeError, "Base"):
            first("m", "x0", self.sigmas)

    def test_two_runs_base_then_lora(self):
        function, _seen = self._function()
        calls = []
        noise = torch.ones(1, 4, 2, 2)
        stubs = {"comfy.sample": _sample_stub(calls), "comfy.samplers": _samplers_stub(function, self.sigmas)}
        with mock.patch.dict(sys.modules, stubs):
            out = nodes.split_lora_sample(self.model, self.schedule, 2, noise, 5, 3.0, "euler", "simple", "p", "n",
                                          "latent", 1.0, callback="cb", seed=9)
        self.assertEqual(out, "latent-2")
        first, second = calls
        self.assertEqual(first["model"].patches, {SELF: ["base"]})
        self.assertIs(second["model"], self.model)
        self.assertIs(first["noise"], noise)
        self.assertEqual(float(second["noise"].abs().sum()), 0.0)
        for call in calls:
            self.assertIs(call["sigmas"], self.sigmas)  # both runs: the whole schedule
            self.assertEqual((call["latent"], call["callback"], call["seed"]), ("latent", "cb", 9))
            self.assertEqual(call["sampler"].extra_options, {"eta": 1.0})

    def test_samplers_that_cannot_be_cut_are_refused(self):
        def restart_sampler(model, x, sigmas, **kwargs):
            return x

        stubs = {"comfy.sample": _sample_stub([]), "comfy.samplers": _samplers_stub(restart_sampler, self.sigmas)}
        with mock.patch.dict(sys.modules, stubs), self.assertRaisesRegex(RuntimeError, "Restart"):
            nodes.split_lora_sample(self.model, self.schedule, 2, torch.ones(1), 5, 3.0, "restart", "simple", "p", "n",
                                    "latent", 1.0)


def _comfy_stubs(calls):
    comfy = types.ModuleType("comfy")
    sample = _sample_stub(calls)
    sample.fix_empty_latent_channels = lambda model, latent, *a: latent
    sample.prepare_noise = lambda latent, seed, inds: "noise"
    utils = types.ModuleType("comfy.utils")
    utils.PROGRESS_BAR_ENABLED = False
    preview = types.ModuleType("latent_preview")
    preview.prepare_callback = lambda model, steps: None
    comfy.sample, comfy.utils = sample, utils
    return {"comfy": comfy, "comfy.sample": sample, "comfy.utils": utils, "latent_preview": preview}


@requires_torch
class SamplerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        bind_torch(globals())

    def setUp(self):
        self.model = FakePatcher({MLP: ["lora"]})
        self.model.model_options[sa.LORA_SCHEDULE_KEY] = {"base_percent": 25, "spans": {MLP: (0, 1)}}
        self.latent = {"samples": SimpleNamespace(shape=(1, 16, 1, 8, 8))}

    def _sample(self, latent, lora_split=True, steps=8, split=None):
        calls = []
        with mock.patch.dict(sys.modules, _comfy_stubs(calls)), \
                mock.patch.object(generation, "split_lora_sample", side_effect=lambda *a, **k: "split") as split_call:
            (out,) = generation._common_sample(self.model, 5, steps, 3.0, "euler", "simple", "p", "n", latent, 1.0,
                                                lora_split=lora_split)
        return out["samples"], calls, split_call

    def test_main_sampler_splits(self):
        samples, calls, split = self._sample(self.latent)
        self.assertEqual((samples, calls), ("split", []))
        self.assertEqual(split.call_args.args[2], 2)

    def test_other_passes_do_not(self):
        samples, calls, split = self._sample(self.latent, lora_split=False)
        self.assertEqual(samples, "latent-1")
        self.assertIs(calls[0]["model"], self.model)
        split.assert_not_called()

    def test_a_masked_latent_is_not_split(self):
        samples, calls, split = self._sample({**self.latent, "noise_mask": "mask"})
        self.assertEqual(samples, "latent-1")
        split.assert_not_called()

    def test_every_step_base_runs_one_pass_without_the_loras(self):
        self.model.model_options[sa.LORA_SCHEDULE_KEY]["base_percent"] = 100
        samples, calls, split = self._sample(self.latent)
        self.assertEqual(calls[0]["model"].patches, {})
        split.assert_not_called()


class SamplerNodeTests(unittest.TestCase):
    def test_refuses_speed_and_spectrum(self):
        model = FakePatcher({MLP: ["lora"]})
        model.model_options[sa.LORA_SCHEDULE_KEY] = {"base_percent": 25, "spans": {MLP: (0, 1)}}
        sampler = generation.ForgeNeoKSamplerCNS()
        with mock.patch.object(generation, "with_pass_tag", side_effect=lambda m, tag: m), \
                mock.patch.object(generation, "_common_sample", side_effect=lambda *a, **k: ("sampled",)):
            # no split happens (5 % of 8 steps rounds to none): SPEED is fine
            model.model_options[sa.LORA_SCHEDULE_KEY]["base_percent"] = 5
            speedy = model.clone()
            speedy.add_wrapper_with_key("sampler_sample", "ai_studio_forge_parity.speed", lambda *a: None)
            self.assertEqual(sampler.sample(speedy, "p", "n", {"samples": None}, 1, 8, 3.0, "euler", "simple"),
                             ("sampled",))
            model.model_options[sa.LORA_SCHEDULE_KEY]["base_percent"] = 25
            for flags in ({"speed_enabled": True}, {"spectrum_enabled": True}):
                with self.subTest(flags=flags), self.assertRaisesRegex(RuntimeError, "SPEED"):
                    sampler.sample(model, "p", "n", {"samples": None}, 1, 8, 3.0, "euler", "simple", **flags)
            model.add_wrapper_with_key("sampler_sample", "ai_studio_forge_parity.speed", lambda *a: None)
            with self.assertRaisesRegex(RuntimeError, "SPEED"):
                sampler.sample(model, "p", "n", {"samples": None}, 1, 8, 3.0, "euler", "simple")

    def test_main_sampler_asks_for_the_split(self):
        model = FakePatcher()
        with mock.patch.object(generation, "_common_sample", side_effect=lambda *a, **k: (k,)) as sample, \
                mock.patch.object(generation, "with_pass_tag", side_effect=lambda m, tag: m):
            generation.ForgeNeoKSamplerCNS().sample(model, "p", "n", {"samples": None}, 1, 8, 3.0, "euler", "simple")
        self.assertTrue(sample.call_args.kwargs["lora_split"])



# ---------------------------------------------------------------------------------------------- state, infotext, X/Y/Z

class StateAndInfotextTests(unittest.TestCase):
    def test_state_clamps_the_lora_card(self):
        from comfy_custom_nodes.ai_studio_forge_parity.studio_state import DEFAULT_STATE, normalize_state

        self.assertEqual((DEFAULT_STATE["lora_base"], DEFAULT_STATE["lora_cross_attn"]), (0, sa.CROSS_ATTN_OFF))
        state = normalize_state({"lora_base": 95, "lora_cross_attn": sa.CROSS_ATTN_ONLY})
        self.assertEqual((state["lora_base"], state["lora_cross_attn"]), (sa.LORA_BASE_MAX, sa.CROSS_ATTN_ONLY))
        state = normalize_state({"lora_base": "x", "lora_cross_attn": "nope"})
        self.assertEqual((state["lora_base"], state["lora_cross_attn"]), (0, sa.CROSS_ATTN_OFF))

    def _text(self, **raw):
        from comfy_custom_nodes.ai_studio_forge_parity.studio_infotext import build_infotext
        from comfy_custom_nodes.ai_studio_forge_parity.studio_state import normalize_state

        return build_infotext(normalize_state({"prompt": "1girl", **raw}), seed=1)

    def test_infotext_names_what_was_on(self):
        plain = self._text()
        for key in ("LoRA base first", "LoRA cross-attn", "CFG schedule", "Word emphasis"):
            self.assertNotIn(key, plain)
        text = self._text(lora_base=30, lora_cross_attn=sa.CROSS_ATTN_WITH_MOD, cfg=3.0, ext={
            "cfg_schedule": {"on": True, "start_cfg": 6.0, "end_percent": 40, "curve": "코사인"},
            "emphasis": {"on": True, "words": "gripping,\nmetal bar", "multiplier": 2.0, "start_percent": 0, "end_percent": 40},
        })
        self.assertIn("LoRA base first: 30%", text)
        self.assertIn("LoRA cross-attn: 교차 어텐션+변조", text)
        self.assertIn("CFG schedule: 6 -> 3 by 40% (코사인)", text)
        self.assertIn('Word emphasis: "gripping, metal bar x2 0-40%"', text)

    def test_reader_gets_the_lora_card_back(self):
        from comfy_custom_nodes.ai_studio_forge_parity.studio_infotext_read import parse_infotext

        settings = parse_infotext(self._text(lora_base=30, lora_cross_attn=sa.CROSS_ATTN_ONLY))["settings"]
        self.assertEqual((settings["lora_base"], settings["lora_cross_attn"]), (30, sa.CROSS_ATTN_ONLY))
        settings = parse_infotext("1girl\nSteps: 20, Sampler: Euler, CFG scale: 3, Seed: 1, Size: 832x1216")["settings"]
        self.assertEqual((settings["lora_base"], settings["lora_cross_attn"]), (0, sa.CROSS_ATTN_OFF))


class XyzAxisTests(unittest.TestCase):
    def _plan(self, axis, text):
        import random

        from comfy_custom_nodes.ai_studio_forge_parity.xyz_plan import plan_xyz

        return plan_xyz({"prompt": "1girl"}, [{"axis": axis, "text": text}], {}, rng=random.Random(0), families={})

    def test_base_first_axis(self):
        plan = self._plan("lora_base", "0, 30")
        self.assertEqual(plan.errors, [])
        self.assertEqual([cell.state["lora_base"] for cell in plan.cells], [0, 30])
        self.assertEqual(plan.labels["x"], ["[LoRA] Base 먼저 %: 0", "[LoRA] Base 먼저 %: 30"])

    def test_cross_attn_axis_takes_its_fixed_values(self):
        plan = self._plan("lora_cross_attn", f"{sa.CROSS_ATTN_OFF}, {sa.CROSS_ATTN_ONLY}")
        self.assertEqual([cell.state["lora_cross_attn"] for cell in plan.cells], [sa.CROSS_ATTN_OFF, sa.CROSS_ATTN_ONLY])
        self.assertTrue(self._plan("lora_cross_attn", "nope").errors)

    def test_extension_axes_exist(self):
        from comfy_custom_nodes.ai_studio_forge_parity.xyz_axes import axis_by_id

        for axis_id in ("ext.cfg_schedule.on", "ext.cfg_schedule.start_cfg", "ext.cfg_schedule.curve", "ext.emphasis.on",
                        "ext.emphasis.words", "ext.emphasis.multiplier", "ext.emphasis.end_percent"):
            self.assertTrue(axis_by_id(axis_id))
        self.assertEqual(axis_by_id("ext.emphasis.words").kind, "lines")
        plan = self._plan("ext.cfg_schedule.start_cfg", "3, 6")
        self.assertEqual([cell.state["ext"]["cfg_schedule"]["on"] for cell in plan.cells], [True, True])



class EffectiveValueTests(unittest.TestCase):
    def _text(self, **raw):
        from comfy_custom_nodes.ai_studio_forge_parity.studio_infotext import build_infotext
        from comfy_custom_nodes.ai_studio_forge_parity.studio_state import normalize_state

        return build_infotext(normalize_state({"prompt": "1girl", **raw}), seed=1)

    def test_state_helpers(self):
        from comfy_custom_nodes.ai_studio_forge_parity import studio_state as ss

        inpaint = ss.normalize_state({"tab": "i2i", "i2i_mode": "inpaint", "lora_base": 30})
        self.assertEqual((ss.effective_lora_base(inpaint), ss.lora_split_steps(inpaint)), (0, 0))
        plain = ss.normalize_state({"lora_base": 30, "steps": 32})
        self.assertEqual((ss.effective_lora_base(plain), ss.lora_split_steps(plain)), (30, 10))
        self.assertEqual(ss.lora_split_steps(ss.normalize_state({"lora_base": 5, "steps": 8})), 0)  # rounds to none
        moss = ss.normalize_state({"unet": "moss_sft.safetensors", "lora_cross_attn": sa.CROSS_ATTN_ONLY})
        self.assertEqual(ss.effective_cross_attn(moss), sa.CROSS_ATTN_OFF)

    def test_infotext_leaves_out_what_did_not_act(self):
        self.assertNotIn("LoRA base first", self._text(tab="i2i", i2i_mode="inpaint", lora_base=30))
        self.assertNotIn("LoRA base first", self._text(lora_base=5, steps=8))
        self.assertNotIn("LoRA cross-attn", self._text(unet="krea2_raw.safetensors", lora_cross_attn=sa.CROSS_ATTN_ONLY))
        emphasis = {"emphasis": {"on": True, "words": "gripping"}}
        self.assertNotIn("Word emphasis", self._text(unet="anima_3.8b_v2.safetensors", ext=emphasis))
        self.assertIn("Word emphasis", self._text(unet="anima-2.9b.safetensors", ext=emphasis))

    def _plan(self, state, axis, text, families=None):
        import random

        from comfy_custom_nodes.ai_studio_forge_parity.xyz_plan import plan_xyz

        return plan_xyz({"prompt": "1girl", **state}, [{"axis": axis, "text": text}], {}, rng=random.Random(0),
                        families=families or {})

    def test_grids_that_would_not_change_are_refused(self):
        self.assertTrue(self._plan({"tab": "i2i", "i2i_mode": "inpaint"}, "lora_base", "0, 30").errors)
        self.assertFalse(self._plan({"tab": "i2i", "i2i_mode": "img2img"}, "lora_base", "0, 30").errors)
        self.assertTrue(self._plan({"unet": "moss_sft.safetensors"}, "lora_cross_attn", "끔, 교차 어텐션").errors)
        self.assertTrue(self._plan({"unet": "big.safetensors"}, "ext.emphasis.multiplier", "1, 2",
                                   families={"big.safetensors": "3.8b"}).errors)
        self.assertFalse(self._plan({"unet": "anima-2.9b.safetensors"}, "ext.emphasis.multiplier", "1, 2").errors)


if __name__ == "__main__":
    unittest.main()
