"""Pack 1.9.0 DoRA inference mode (``dora_infer_mode``, ``ForgeNeoDoraInferenceMode``) without ComfyUI.

* ``decompose`` against the LyCORIS training merge written out in ``tests/_dora_oracle`` (``lycoris/modules/lokr.py``
  ``get_merged_weight`` -> ``apply_weight_decompose`` operation for operation: ``vector_norm`` of ``W0 + dW`` plus
  the fp32 eps, ``1 * (m / norm - 1) + 1``, the product; lycoris is installed in neither the app venv nor ComfyUI's
  python), bit for bit on both DoRA axes, for LoRA, LoKr and LoHa deltas, linear and conv weights, strengths 1, 0.5
  and 0 (strength blends ``W0 + s*(W_dora - W0)`` like Forge and sam-extra), also on a conv input axis and with
  magnitudes outside [0.5, 2] of ``||W0 + dW||``, where ComfyUI's operations round differently; the Forge/Comfy
  formula in fp32 (``||W0||`` on the output axis); the merge written into the delta, in-place writes into views, the
  fp32 return rule, a failure that leaves the weight untouched.
* The installed sam-extra (``sam3ext/dora_infer_mode.py``) as the oracle: the Forge/Comfy fp32 branch of ``decompose``
  and ``_can_return_fp32`` give its bits, the LyCORIS branch gives its bits where both use the same operations, its
  mode keys are the pack's, its radio order and default are the node's, and its ``normalize_mode`` reads the node's
  labels as the same keys.
* The patch machinery on stand-ins with ComfyUI's adapter contract (``(loaded_keys, weights)``, the non-DoRA line
  ``weight += function(((strength * alpha) * diff).type(weight.dtype))``, errors logged and swallowed): DoRA
  adapters become mode subclasses whose mode survives ComfyUI's ``type(v)(v.loaded_keys, weights)`` rebuild, nothing
  shared is written, ``dW`` comes in fp32 whatever the merge dtype and leaves the adapter before its ``+=`` (a
  zero-stride stand-in weight, GLoRA's W0), the DoRA-off and stock modes, the stock fallback with the cache emptied for
  an OOM (also one the adapter swallows), and the per-run account and report.
* The node on a ModelPatcher stand-in: a clone with a new ``patches_uuid`` and one run-report wrapper only when
  something changes; registration, version, README and notices.

The same node on the real ComfyUI runtime (adapters, ModelPatcher, LowVramPatch, load_models_gpu, wrapper executor):
``tests/test_comfy_dora_infer_mode_runtime.py``.
"""
from __future__ import annotations

import ast
import importlib
import logging
import sys
import unittest
import uuid
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import comfy_custom_nodes.ai_studio_forge_parity as bundled_pack
from comfy_custom_nodes.ai_studio_forge_parity import dora_infer_mode as dim
from comfy_custom_nodes.ai_studio_forge_parity import dora_infer_mode_nodes as nodes
from core.comfy_node_pack import PACK_VERSION
from tests import _dora_oracle as oracle
from tests._optional_deps import bind_torch, requires_torch
from tests._sam_extra_ext import EXT_ROOT, requires_extension

torch = None  # filled by bind_torch()

PACK_DIR = Path(bundled_pack.__file__).resolve().parent
LYCORIS = dim.MODE_LABELS[dim.MODE_LYCORIS]
STOCK_MARK = 1000.0   # what the stand-ins' "stock DoRA merge" adds, so a stock merge is easy to see


def _identity(tensor):
    return tensor


# --------------------------------------------------------------------------------------------------
# The formulas, written out
# --------------------------------------------------------------------------------------------------


def lycoris_merge(w0, delta, dora_scale, wd_on_output, strength=1.0):
    """LyCORIS' training merge (``tests/_dora_oracle.lycoris_merge``), Forge's strength blend."""
    return oracle.lycoris_merge(torch, w0, delta, dora_scale, wd_on_output, strength)


def comfy_formula_fp32(w0, delta, dora_scale, wd_on_output, strength=1.0):
    """ComfyUI's ``weight_decompose`` formula in fp32 (``tests/_dora_oracle.comfy_formula_fp32``)."""
    return oracle.comfy_formula_fp32(torch, w0, delta, dora_scale, wd_on_output, strength)


_norm_dims = oracle.norm_dims


def _gen(seed):
    return torch.Generator().manual_seed(seed)


def _lora_delta(g, out, inp, kernel=(), rank=4, alpha=2.0):
    up = torch.randn(out, rank, *[1] * len(kernel), generator=g) * 0.1
    down = torch.randn(rank, inp, *kernel, generator=g) * 0.1
    return torch.mm(up.flatten(1), down.flatten(1)).reshape(out, inp, *kernel) * (alpha / rank)


def _lokr_delta(g, out, inp, kernel=(), rank=3, alpha=2.0):
    w1 = torch.randn(4, 2, generator=g) * 0.3
    w2 = torch.mm(torch.randn(out // 4, rank, generator=g), torch.randn(rank, (inp // 2) * max(1, _prod(kernel)),
                                                                          generator=g)) * 0.1
    if kernel:
        w2 = w2.reshape(out // 4, inp // 2, *kernel)
        w1 = w1.unsqueeze(2).unsqueeze(2)
    return torch.kron(w1, w2).reshape(out, inp, *kernel) * (alpha / rank)


def _loha_delta(g, out, inp, kernel=(), rank=4, alpha=2.0):
    flat = inp * max(1, _prod(kernel))
    m1 = torch.randn(out, rank, generator=g).mm(torch.randn(rank, flat, generator=g)) * 0.3
    m2 = torch.randn(out, rank, generator=g).mm(torch.randn(rank, flat, generator=g)) * 0.3
    return (m1 * m2).reshape(out, inp, *kernel) * (alpha / rank)


def _prod(values):
    result = 1
    for value in values:
        result *= value
    return result


DELTAS = {"lora": _lora_delta, "lokr": _lokr_delta, "loha": _loha_delta}


def _case(kind, wd_on_output, kernel=(), seed=0, out=24, inp=16):
    """``W0``, an alpha-scaled ``dW`` of ``kind`` (about 40 % of ``W0``, so the two formulas differ) and a trained
    magnitude (``||W0||`` along the DoRA axis, +-30 %)."""
    g = _gen(seed)
    w0 = torch.randn(out, inp, *kernel, generator=g) * 0.05
    delta = DELTAS[kind](g, out, inp, kernel)
    delta = delta * (0.4 * w0.abs().mean() / delta.abs().mean())
    norm = torch.linalg.vector_norm(w0, dim=_norm_dims(w0.dim(), wd_on_output), keepdim=True)
    return w0, delta, norm * (1.0 + 0.3 * (torch.rand(norm.shape, generator=g) - 0.5))


@requires_torch
class FormulaTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        bind_torch(globals())

    def _lycoris(self, scale, weight, delta, strength, function=_identity, **kwargs):
        return dim.decompose(scale, weight, delta, strength, function, lycoris=True, **kwargs)

    def test_lycoris_mode_is_the_lycoris_training_merge_bit_for_bit_on_both_axes(self):
        for kind in DELTAS:
            for wd_on_output in (True, False):
                for kernel in ((), (3, 3)):
                    for strength in (1.0, 0.5):
                        with self.subTest(kind=kind, output_axis=wd_on_output, kernel=kernel, strength=strength):
                            w0, delta, scale = _case(kind, wd_on_output, kernel, seed=len(kernel) + 7)
                            weight = w0.clone()
                            out = self._lycoris(scale, weight, delta.clone(), strength)
                            self.assertIs(out, weight)
                            expected = lycoris_merge(w0, delta, scale, wd_on_output, strength)
                            self.assertTrue(torch.equal(weight, expected),
                                            f"max |d| {(weight - expected).abs().max().item()}")
                            if strength == 1.0:  # LyCORIS' defining property: the norm along the axis is m
                                dims = _norm_dims(weight.dim(), wd_on_output)
                                torch.testing.assert_close(torch.linalg.vector_norm(weight, dim=dims, keepdim=True),
                                                           scale, rtol=1e-5, atol=1e-7)

    def test_lycoris_mode_follows_lycoris_where_its_operations_round_differently(self):
        """Review F1: LyCORIS reduces the norm with ``vector_norm`` (ComfyUI's and sam-extra's transpose-reshape norm
        of a conv input axis sums in another order) and always applies ``multiplier * (scale - 1) + 1``, which rounds
        when ``m / ||W0 + dW||`` is outside [0.5, 2]. Bit for bit in both cases, at strengths 1 and 0.5."""
        differs = {"conv input axis": 0, "far magnitude": 0}
        for kind in DELTAS:
            for wd_on_output in (True, False):
                for kernel in ((), (3, 3)):
                    for far in (False, True):
                        with self.subTest(kind=kind, output_axis=wd_on_output, kernel=kernel, far=far):
                            g = _gen(31 + len(kernel) + 2 * wd_on_output)
                            w0 = torch.randn(48, 32, *kernel, generator=g) * 0.05
                            delta = DELTAS[kind](g, 48, 32, kernel)
                            delta = delta * (0.4 * w0.abs().mean() / delta.abs().mean())
                            if far:
                                scale = oracle.far_magnitude(torch, w0 + delta, wd_on_output, g)
                            else:
                                scale = oracle.trained_magnitude(torch, w0 + delta, wd_on_output, g)
                            for strength in (1.0, 0.5):
                                out = self._lycoris(scale, w0.clone(), delta.clone(), strength, inplace=False)
                                expected = lycoris_merge(w0, delta, scale, wd_on_output, strength)
                                self.assertTrue(torch.equal(out, expected),
                                                f"s={strength} max |d| {(out - expected).abs().max().item()}")
                            expected = lycoris_merge(w0, delta, scale, wd_on_output)
                            if kernel and not wd_on_output and not far:   # ComfyUI's / sam-extra's norm operations
                                sam_extras_ops = dim.decompose(scale, w0.clone(), delta.clone(), 1.0, _identity,
                                                               lycoris=False, inplace=False)
                                differs["conv input axis"] += int(not torch.equal(sam_extras_ops, expected))
                            if far and not wd_on_output and not kernel:   # the same norm without the step
                                merged = w0 + delta
                                norm = torch.linalg.vector_norm(merged, dim=(0,), keepdim=True)
                                without_step = merged * (scale / (norm + torch.finfo(torch.float32).eps))
                                differs["far magnitude"] += int(not torch.equal(without_step, expected))
        # the cases are not vacuous: the other operations give other bits there
        self.assertGreater(differs["conv input axis"], 0)
        self.assertGreater(differs["far magnitude"], 0)

    def test_the_merge_is_written_into_the_delta(self):
        """Review F1 (memory): the delta is taken over, so no third fp32 copy of the layer is made."""
        w0, delta, scale = _case("lora", True, seed=13)
        temp = w0.to(torch.float16)
        owned = delta.clone()
        out = self._lycoris(scale, temp, owned, 1.0, inplace=False)
        self.assertIs(out, owned)
        self.assertTrue(torch.equal(out, lycoris_merge(temp, delta, scale, True)))
        # a function that returns a new tensor leaves the delta alone and the merge is its own tensor
        owned = delta.clone()
        out = self._lycoris(scale, temp, owned, 1.0, lambda t: t * 1.0, inplace=False)
        self.assertIsNot(out, owned)
        self.assertTrue(torch.equal(owned, delta))
        self.assertTrue(torch.equal(out, lycoris_merge(temp, delta, scale, True)))

    def test_forge_fp32_mode_is_comfys_formula_in_fp32(self):
        for wd_on_output in (True, False):
            for strength in (1.0, 0.6):
                with self.subTest(output_axis=wd_on_output, strength=strength):
                    w0, delta, scale = _case("lokr", wd_on_output, seed=3)
                    weight = w0.clone()
                    dim.decompose(scale, weight, delta.clone(), strength, _identity, lycoris=False)
                    torch.testing.assert_close(weight, comfy_formula_fp32(w0, delta, scale, wd_on_output, strength),
                                               rtol=1e-6, atol=1e-8)

    def test_the_output_axis_formulas_differ_and_the_input_axis_formulas_are_the_same(self):
        w0, delta, scale = _case("lora", True, seed=5)
        forge, lyco = w0.clone(), w0.clone()
        dim.decompose(scale, forge, delta.clone(), 1.0, _identity, lycoris=False)
        self._lycoris(scale, lyco, delta.clone(), 1.0)
        self.assertGreater((forge - lyco).abs().max().item(), 1e-4)
        w0, delta, scale = _case("lora", False, seed=6)
        forge, lyco = w0.clone(), w0.clone()
        dim.decompose(scale, forge, delta.clone(), 1.0, _identity, lycoris=False)
        self._lycoris(scale, lyco, delta.clone(), 1.0)
        self.assertTrue(torch.equal(forge, lyco))   # a linear weight, m / ||W0+dW|| inside [0.5, 2]: the same bits

    def test_strength_blends_linearly_and_zero_leaves_the_weight(self):
        w0, delta, scale = _case("loha", True, seed=8)
        full, zero, half = w0.clone(), w0.clone(), w0.clone()
        self._lycoris(scale, full, delta.clone(), 1.0)
        self._lycoris(scale, zero, delta.clone(), 0.0)
        self._lycoris(scale, half, delta.clone(), 0.5)
        self.assertTrue(torch.equal(zero, w0))
        torch.testing.assert_close(half, w0 + 0.5 * (full - w0), rtol=1e-6, atol=1e-8)
        # LyCORIS' own multiplier would keep all of dW at multiplier 0; the node does not follow it
        self.assertGreater((zero - (w0 + delta)).abs().max().item(), 1e-3)

    def test_function_is_applied_to_the_alpha_scaled_delta(self):
        w0, delta, scale = _case("lora", True, seed=9)
        weight = w0.clone()
        self._lycoris(scale, weight, delta.clone(), 1.0, lambda t: t * 2.0)
        self.assertTrue(torch.equal(weight, lycoris_merge(w0, delta * 2.0, scale, True)))

    def test_a_view_is_written_in_place_in_its_dtype_and_the_full_load_path_gets_fp32(self):
        w0, delta, scale = _case("lora", True, seed=11)
        fused = torch.zeros(3 * 24, 16, dtype=torch.float16)
        fused[24:48] = w0.to(torch.float16)
        view = fused.narrow(0, 24, 24)
        self.assertFalse(dim._can_return_fp32(view, torch.float32))
        out = self._lycoris(scale, view, delta.clone(), 1.0, inplace=True)
        self.assertIs(out, view)
        self.assertEqual(out.dtype, torch.float16)
        expected = lycoris_merge(w0.to(torch.float16), delta, scale, True)
        self.assertTrue(torch.equal(fused[24:48], expected.to(torch.float16)))
        self.assertEqual(fused[:24].abs().sum().item(), 0.0)
        self.assertEqual(fused[48:].abs().sum().item(), 0.0)

        temp = w0.to(torch.float16)
        before = temp.clone()
        self.assertTrue(dim._can_return_fp32(temp, torch.float32))
        out = self._lycoris(scale, temp, delta.clone(), 1.0, inplace=False)
        self.assertEqual(out.dtype, torch.float32)
        self.assertTrue(torch.equal(temp, before))
        self.assertTrue(torch.equal(out, expected))
        # LowVramPatch (intermediate = the weight's dtype) and an fp32 weight are written in place
        self.assertFalse(dim._can_return_fp32(w0.to(torch.bfloat16), torch.bfloat16))
        self.assertFalse(dim._can_return_fp32(w0.clone(), torch.float32))

    def test_a_failure_leaves_the_weight_untouched(self):
        w0, delta, _ = _case("lora", True, seed=12)
        for lycoris in (True, False):
            weight = w0.clone()
            with self.assertRaises(RuntimeError):
                dim.decompose(torch.ones(5, 1), weight, delta.clone(), 1.0, _identity, lycoris=lycoris)
            self.assertTrue(torch.equal(weight, w0))


# --------------------------------------------------------------------------------------------------
# sam-extra as the oracle
# --------------------------------------------------------------------------------------------------


def _ext_module():
    root = str(EXT_ROOT)
    if root not in sys.path:
        sys.path.insert(0, root)
    return importlib.import_module("sam3ext.dora_infer_mode")


def _ext_tree(relative: str):
    return ast.parse((EXT_ROOT / relative).read_text(encoding="utf-8"))


def _ext_strings(tree) -> dict:
    """Module-level ``NAME = "text"`` assignments."""
    strings = {}
    for item in tree.body:
        if isinstance(item, ast.Assign) and isinstance(item.value, ast.Constant) and isinstance(item.value.value, str):
            strings.update({target.id: item.value.value for target in item.targets if isinstance(target, ast.Name)})
    return strings


def _ext_label(node, strings: dict) -> str:
    if isinstance(node, ast.Name):
        return strings[node.id]
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    raise AssertionError(f"not a label: {ast.dump(node)}")


def _ext_label_list(relative: str, name: str) -> list:
    """A module-level list of labels (``LABEL_*`` names resolved to their strings)."""
    tree = _ext_tree(relative)
    strings = _ext_strings(tree)
    for item in tree.body:
        if isinstance(item, ast.Assign) and any(getattr(t, "id", None) == name for t in item.targets):
            return [_ext_label(element, strings) for element in item.value.elts]
    raise AssertionError(f"{relative} has no module-level {name}")


def _ext_radio_default(relative: str, choices: str) -> str:
    """``value=`` of the ``gr.Radio(choices=<choices>, ...)`` in ``relative``, resolved to its label."""
    tree = _ext_tree(relative)
    strings = _ext_strings(tree)
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and getattr(node.func, "attr", None) == "Radio":
            keywords = {keyword.arg: keyword.value for keyword in node.keywords}
            if getattr(keywords.get("choices"), "id", None) == choices:
                return _ext_label(keywords["value"], strings)
    raise AssertionError(f"{relative} has no gr.Radio(choices={choices})")


@requires_torch
@requires_extension
class SamExtraOracleTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        bind_torch(globals())
        cls.ext = _ext_module()

    def test_forge_fp32_gives_sam_extras_bits(self):
        """The Forge/Comfy fp32 mode is sam-extra's ``decompose(merged_norm=False)`` operation for operation (the pack
        hands it ``a*dW``, so alpha is 1 there), conv weights and both axes included."""
        cases = 0
        for kind in DELTAS:
            for wd_on_output in (True, False):
                for strength in (1.0, 0.7, 0.0):
                    for dtype in (torch.float32, torch.float16, torch.bfloat16):
                        for inplace in (True, False):
                            w0, delta, scale = _case(kind, wd_on_output, (3, 3) if kind == "lora" else (), seed=cases)
                            ours, theirs = w0.to(dtype), w0.to(dtype)
                            a = dim.decompose(scale, ours, delta.clone(), strength, _identity, lycoris=False,
                                              inplace=inplace)
                            b = self.ext.decompose(scale, theirs, delta.clone(), 1.0, strength, _identity,
                                                   merged_norm=False, inplace=inplace)
                            with self.subTest(kind=kind, axis=wd_on_output, s=strength, dtype=dtype, inplace=inplace):
                                self.assertEqual(a.dtype, b.dtype)
                                self.assertTrue(torch.equal(a, b))
                                self.assertTrue(torch.equal(ours, theirs))
                            cases += 1
        self.assertEqual(cases, 3 * 2 * 3 * 3 * 2)

    def test_lycoris_mode_is_sam_extras_except_where_lycoris_rounds_differently(self):
        """sam-extra's LyCORIS mode (``merged_norm=True``) uses ComfyUI's norm operations and no multiplier step: on a
        linear weight with ``m / ||W0 + dW||`` inside [0.5, 2] both give the same bits; on a conv input axis and outside
        that range the pack follows LyCORIS (``test_lycoris_mode_follows_lycoris_...``) and stays within fp32
        rounding of sam-extra."""
        for kind in DELTAS:
            for wd_on_output in (True, False):
                for strength in (1.0, 0.7):
                    for dtype in (torch.float32, torch.bfloat16):
                        with self.subTest(kind=kind, axis=wd_on_output, s=strength, dtype=dtype):
                            w0, delta, scale = _case(kind, wd_on_output, seed=40 + wd_on_output)
                            ours, theirs = w0.to(dtype), w0.to(dtype)
                            dim.decompose(scale, ours, delta.clone(), strength, _identity, lycoris=True)
                            self.ext.decompose(scale, theirs, delta.clone(), 1.0, strength, _identity, merged_norm=True)
                            self.assertTrue(torch.equal(ours, theirs))
        g = _gen(51)
        w0 = torch.randn(48, 32, 3, 3, generator=g) * 0.05
        delta = _lora_delta(g, 48, 32, (3, 3)) * 0.3
        for wd_on_output in (True, False):
            scale = oracle.far_magnitude(torch, w0 + delta, wd_on_output, g)
            ours = dim.decompose(scale, w0.clone(), delta.clone(), 1.0, _identity, lycoris=True, inplace=False)
            theirs = self.ext.decompose(scale, w0.clone(), delta.clone(), 1.0, 1.0, _identity, merged_norm=True,
                                        inplace=False)
            self.assertTrue(torch.equal(ours, lycoris_merge(w0, delta, scale, wd_on_output)))
            torch.testing.assert_close(ours, theirs, rtol=1e-6, atol=1e-8)

    def test_the_return_rule_is_sam_extras(self):
        base = torch.zeros(6, 4)
        for weight in (base, base.half(), base.bfloat16(), base.half().narrow(0, 1, 3), base.narrow(0, 0, 2)):
            for intermediate in (torch.float32, torch.float16, torch.bfloat16):
                with self.subTest(dtype=weight.dtype, view=weight._base is not None, intermediate=intermediate):
                    self.assertEqual(dim._can_return_fp32(weight, intermediate),
                                     self.ext._can_return_fp32(weight, intermediate))

    def test_mode_keys_and_labels_are_sam_extras(self):
        self.assertEqual(dim.MODES, self.ext.MODES)
        for mode, label in dim.MODE_LABELS.items():
            with self.subTest(label=label):
                self.assertEqual(self.ext.normalize_mode(label), mode)
        # the order of sam-extra's radio: scripts/dora_infer_mode.py MODE_CHOICES (LABEL_* names) -> keys
        radio = [self.ext.normalize_mode(label) for label in _ext_label_list("scripts/dora_infer_mode.py",
                                                                            "MODE_CHOICES")]
        self.assertEqual(radio, [dim.mode_key(label) for label in dim.MODE_CHOICES])
        self.assertEqual(radio, [dim.MODE_LYCORIS, dim.MODE_FORGE_FP32, dim.MODE_FORGE, dim.MODE_NO_MAGNITUDE])
        # and its default: gr.Radio(choices=MODE_CHOICES, value=LABEL_LYCORIS)
        default = _ext_radio_default("scripts/dora_infer_mode.py", "MODE_CHOICES")
        self.assertEqual(self.ext.normalize_mode(default), dim.DEFAULT_MODE)
        self.assertEqual(nodes.ForgeNeoDoraInferenceMode.INPUT_TYPES()["required"]["mode"][1]["default"],
                         dim.MODE_LABELS[self.ext.normalize_mode(default)])


# --------------------------------------------------------------------------------------------------
# Adapter stand-ins with ComfyUI's contract
# --------------------------------------------------------------------------------------------------


class _FakeAdapter:
    """``(loaded_keys, weights)``; ``calculate_weight`` like ComfyUI 0.36's adapters: the matrices cast to
    ``intermediate_dtype``, a DoRA merge when ``dora_scale`` is set (here: ``+STOCK_MARK``), otherwise
    ``weight += function(((strength * alpha) * diff).type(weight.dtype))``; errors are logged and swallowed."""

    name = "fake"
    DORA = 4
    calls: list = []
    handed: list = []   # the weight each call was handed

    def __init__(self, loaded_keys, weights):
        self.loaded_keys = loaded_keys
        self.weights = weights

    def _delta(self, weight, dtype):
        raise NotImplementedError

    def calculate_weight(self, weight, key, strength, strength_model, offset, function, intermediate_dtype=None,
                         original_weight=None):
        dtype = intermediate_dtype if intermediate_dtype is not None else torch.float32
        type(self).calls.append((key, dtype, self.weights[self.DORA] is not None))
        type(self).handed.append(weight)
        try:
            diff, alpha = self._delta(weight, dtype)
            if self.weights[self.DORA] is not None:
                weight += STOCK_MARK
            else:
                weight += function(((strength * alpha) * diff).type(weight.dtype))
        except Exception as exc:
            logging.getLogger("fake.comfy").error("ERROR %s %s %s", self.name, key, exc)
        return weight


class FakeLoRA(_FakeAdapter):
    """LoRAAdapter: ``(up, down, alpha, mid, dora_scale, reshape)``."""

    name = "lora"

    def _delta(self, weight, dtype):
        up, down, alpha = self.weights[0].to(dtype), self.weights[1].to(dtype), self.weights[2]
        if up.shape[0] < 0:
            raise RuntimeError("unreachable")
        return torch.mm(up.flatten(1), down.flatten(1)).reshape(weight.shape), alpha / down.shape[0]


class FakeLoKr(_FakeAdapter):
    """LoKrAdapter: ``(w1, w2, alpha, w1_a, w1_b, w2_a, w2_b, t2, dora_scale)``."""

    name = "lokr"
    DORA = 8

    def _delta(self, weight, dtype):
        w1 = self.weights[0].to(dtype)
        w2 = torch.mm(self.weights[5].to(dtype), self.weights[6].to(dtype))
        return torch.kron(w1, w2).reshape(weight.shape), self.weights[2] / self.weights[6].shape[0]


class FakeLoHa(_FakeAdapter):
    """LoHaAdapter: ``(w1a, w1b, alpha, w2a, w2b, t1, t2, dora_scale)``."""

    name = "loha"
    DORA = 7

    def _delta(self, weight, dtype):
        v = [w.to(dtype) if hasattr(w, "to") else w for w in self.weights]
        return (torch.mm(v[0], v[1]) * torch.mm(v[3], v[4])).reshape(weight.shape), self.weights[2] / v[1].shape[0]


class FakeGLoRA(_FakeAdapter):
    """GLoRAAdapter: ``(a1, a2, b1, b2, alpha, dora_scale)``; ``dW = W0 a1 a2 + b1 b2`` reads the weight."""

    name = "glora"
    DORA = 5

    def _delta(self, weight, dtype):
        a1, a2, b1, b2 = (w.to(dtype) for w in self.weights[:4])
        diff = torch.mm(torch.mm(weight.to(dtype), a1), a2) + torch.mm(b1, b2)
        return diff, self.weights[4] / a2.shape[0]


class FakeOFT(_FakeAdapter):
    """OFTAdapter: ``(blocks, rescale, alpha, dora_scale)`` - stays stock."""

    name = "oft"
    DORA = 3


class FakeLoRASubclass(FakeLoRA):
    """Another node's LoRAAdapter subclass: not re-classed (its calculate_weight may differ)."""


FAKE_MODULE = SimpleNamespace(LoRAAdapter=FakeLoRA, LoKrAdapter=FakeLoKr, LoHaAdapter=FakeLoHa,
                              GLoRAAdapter=FakeGLoRA, OFTAdapter=FakeOFT)


def _specs():
    return dim.adapter_specs(FAKE_MODULE)


class FakePatcher:
    """ModelPatcher's patch bookkeeping: ``clone`` copies the lists (tuples and adapters shared) and the uuid."""

    def __init__(self, patches=None):
        self.patches = patches if patches is not None else {}
        self.patches_uuid = uuid.uuid4()
        self.parent = None
        self.wrappers = {}

    def clone(self):
        clone = FakePatcher({key: items[:] for key, items in self.patches.items()})
        clone.patches_uuid = self.patches_uuid
        clone.parent = self
        clone.wrappers = {kind: {key: list(items) for key, items in keyed.items()}
                          for kind, keyed in self.wrappers.items()}
        return clone

    # ModelPatcher.add_wrapper_with_key / remove_wrappers_with_key
    def add_wrapper_with_key(self, wrapper_type, key, wrapper):
        self.wrappers.setdefault(wrapper_type, {}).setdefault(key, []).append(wrapper)

    def remove_wrappers_with_key(self, wrapper_type, key):
        self.wrappers.get(wrapper_type, {}).pop(key, None)

    def run_wrappers(self):
        return self.wrappers.get("outer_sample", {}).get(dim.WRAPPER_KEY, [])

    def add_patches(self, patches, strength_patch=1.0, strength_model=1.0):
        for key, adapter in patches.items():
            offset = function = None
            if not isinstance(key, str):
                key, offset = key[0], key[1]
            self.patches.setdefault(key, []).append((strength_patch, adapter, strength_model, offset, function))
        self.patches_uuid = uuid.uuid4()


def _lora(g, scale, out=24, inp=16, rank=4, alpha=2.0, reshape=None):
    up = torch.randn(out, rank, generator=g) * 0.1
    down = torch.randn(rank, inp, generator=g) * 0.1
    return FakeLoRA({"k.up", "k.down"}, (up, down, alpha, None, scale, reshape)), up.mm(down) * (alpha / rank)


@requires_torch
class AdapterTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        bind_torch(globals())

    def setUp(self):
        for cls in (_FakeAdapter, FakeLoRA, FakeLoKr, FakeLoHa, FakeGLoRA, FakeOFT, FakeLoRASubclass):
            cls.calls = []
            cls.handed = []
        dim.RUN.reset()

    def _w0_scale(self, seed=0, wd_on_output=True, out=24, inp=16):
        g = _gen(seed)
        w0 = torch.randn(out, inp, generator=g) * 0.05
        norm = torch.linalg.vector_norm(w0, dim=_norm_dims(2, wd_on_output), keepdim=True)
        return g, w0, norm * (1.0 + 0.3 * torch.rand(norm.shape, generator=g))

    def _mode(self, adapter, mode):
        spec = dim.SUPPORTED_ADAPTERS[{FakeLoRA: "LoRAAdapter", FakeLoKr: "LoKrAdapter", FakeLoHa: "LoHaAdapter",
                                       FakeGLoRA: "GLoRAAdapter"}[type(adapter)]]
        return dim.mode_class(type(adapter), spec, mode)(adapter.loaded_keys, adapter.weights)

    def test_specs_name_the_dora_scale_slot_of_each_adapter(self):
        specs = _specs()
        self.assertEqual({cls: (name, spec.dora_index, supported) for cls, (name, spec, supported) in specs.items()}, {
            FakeLoRA: ("LoRAAdapter", 4, True), FakeLoHa: ("LoHaAdapter", 7, True),
            FakeLoKr: ("LoKrAdapter", 8, True), FakeGLoRA: ("GLoRAAdapter", 5, True),
            FakeOFT: ("OFTAdapter", 3, False),
        })
        self.assertEqual(dim.SUPPORTED_ADAPTERS["LoRAAdapter"].reshape_index, 5)

    def test_the_mode_is_a_class_attribute_that_survives_comfys_rebuild(self):
        g, w0, scale = self._w0_scale()
        adapter, delta = _lora(g, scale)
        wrapped = self._mode(adapter, dim.MODE_LYCORIS)
        self.assertIsInstance(wrapped, FakeLoRA)
        self.assertIs(type(wrapped), dim.mode_class(FakeLoRA, dim.SUPPORTED_ADAPTERS["LoRAAdapter"], dim.MODE_LYCORIS))
        self.assertIs(wrapped.weights, adapter.weights)
        self.assertEqual(wrapped.name, "lora")
        # comfy/lora.py prefetch_prepared_value: type(v)(v.loaded_keys, prepared weights)
        rebuilt = type(wrapped)(wrapped.loaded_keys, tuple(wrapped.weights))
        self.assertEqual(getattr(type(rebuilt), dim.MODE_ATTR), dim.MODE_LYCORIS)
        weight = w0.clone()
        rebuilt.calculate_weight(weight, "k", 1.0, 1.0, None, _identity, torch.float32)
        self.assertTrue(torch.equal(weight, lycoris_merge(w0, delta, scale, True)))
        with self.assertRaises(ValueError):
            dim.mode_class(FakeLoRA, dim.SUPPORTED_ADAPTERS["LoRAAdapter"], dim.MODE_FORGE)

    def test_lycoris_merges_each_adapter_kind_in_fp32_on_both_axes(self):
        for wd_on_output in (True, False):
            g, w0, scale = self._w0_scale(seed=1, wd_on_output=wd_on_output)
            w1, w2a, w2b = torch.randn(4, 2, generator=g), torch.randn(6, 3, generator=g), torch.randn(3, 8, generator=g)
            lokr = FakeLoKr(set(), (w1 * 0.3, None, 3.0, None, None, w2a * 0.1, w2b * 0.1, None, scale))
            m = [torch.randn(*shape, generator=g) * 0.3 for shape in ((24, 4), (4, 16), (24, 4), (4, 16))]
            loha = FakeLoHa(set(), (m[0], m[1], 2.0, m[2], m[3], None, None, scale))
            a1, a2 = torch.randn(16, 4, generator=g) * 0.3, torch.randn(4, 16, generator=g) * 0.3
            b1, b2 = torch.randn(24, 4, generator=g) * 0.3, torch.randn(4, 16, generator=g) * 0.3
            glora = FakeGLoRA(set(), (a1, a2, b1, b2, 2.0, scale))
            lora, lora_delta = _lora(g, scale)
            deltas = {
                "lora": lora_delta,
                "lokr": torch.kron(w1 * 0.3, (w2a * 0.1).mm(w2b * 0.1)) * (3.0 / 3),
                "loha": (m[0].mm(m[1]) * m[2].mm(m[3])) * (2.0 / 4),
                "glora": (w0.mm(a1).mm(a2) + b1.mm(b2)) * (2.0 / 4),
            }
            for adapter in (lora, lokr, loha, glora):
                for strength in (1.0, 0.5):
                    with self.subTest(kind=adapter.name, output_axis=wd_on_output, strength=strength):
                        weight = w0.clone()
                        out = self._mode(adapter, dim.MODE_LYCORIS).calculate_weight(
                            weight, "k", strength, 1.0, None, _identity, torch.float32)
                        self.assertIs(out, weight)
                        self.assertTrue(torch.equal(
                            weight, lycoris_merge(w0, deltas[adapter.name], scale, wd_on_output, strength)))

    def test_delta_is_fp32_whatever_dtype_comfy_merges_in(self):
        g, w0, scale = self._w0_scale(seed=2)
        adapter, delta = _lora(g, scale)
        wrapped = self._mode(adapter, dim.MODE_LYCORIS)
        weight = w0.to(torch.bfloat16)
        base = weight.clone()
        out = wrapped.calculate_weight(weight, "k", 1.0, 1.0, None, _identity, torch.bfloat16)  # LowVramPatch
        self.assertIs(out, weight)
        self.assertEqual(out.dtype, torch.bfloat16)
        self.assertTrue(torch.equal(out, lycoris_merge(base, delta, scale, True).to(torch.bfloat16)))
        # the adapter's own code ran once, in fp32, without the dora_scale
        self.assertEqual(FakeLoRA.calls, [("k", torch.float32, False)])

    def test_the_full_load_path_gets_fp32_and_views_are_written_in_place(self):
        g, w0, scale = self._w0_scale(seed=3)
        adapter, delta = _lora(g, scale)
        wrapped = self._mode(adapter, dim.MODE_LYCORIS)
        temp = w0.to(torch.float16)
        before = temp.clone()
        out = wrapped.calculate_weight(temp, "k", 0.5, 1.0, None, _identity, torch.float32)
        self.assertEqual(out.dtype, torch.float32)
        self.assertTrue(torch.equal(temp, before))
        self.assertTrue(torch.equal(out, lycoris_merge(before, delta, scale, True, 0.5)))
        fused = torch.zeros(48, 16, dtype=torch.float16)
        fused[24:] = w0.to(torch.float16)
        view = fused.narrow(0, 24, 24)
        out = wrapped.calculate_weight(view, "k", 0.5, 1.0, (0, 24, 24), _identity, torch.float32)
        self.assertIs(out, view)
        self.assertTrue(torch.equal(fused[24:], lycoris_merge(before, delta, scale, True, 0.5).to(torch.float16)))
        self.assertEqual(fused[:24].abs().sum().item(), 0.0)

    def test_forge_fp32_and_no_magnitude_and_stock(self):
        g, w0, scale = self._w0_scale(seed=4)
        adapter, delta = _lora(g, scale)
        weight = w0.clone()
        self._mode(adapter, dim.MODE_FORGE_FP32).calculate_weight(weight, "k", 0.8, 1.0, None, _identity,
                                                                  torch.float32)
        torch.testing.assert_close(weight, comfy_formula_fp32(w0, delta, scale, True, 0.8), rtol=1e-6, atol=1e-8)
        # DoRA off: the adapter's own non-DoRA line, in the merge dtype
        weight = w0.to(torch.bfloat16)
        plain = FakeLoRA(set(), adapter.weights[:4] + (None,) + adapter.weights[5:])
        expected = plain.calculate_weight(w0.to(torch.bfloat16), "k", 0.8, 1.0, None, _identity, torch.bfloat16)
        FakeLoRA.calls = []
        out = self._mode(adapter, dim.MODE_NO_MAGNITUDE).calculate_weight(weight, "k", 0.8, 1.0, None, _identity,
                                                                         torch.bfloat16)
        self.assertTrue(torch.equal(out, expected))
        self.assertEqual(FakeLoRA.calls, [("k", torch.bfloat16, False)])
        # the stock class merges the stock way
        weight = w0.clone()
        adapter.calculate_weight(weight, "k", 1.0, 1.0, None, _identity, torch.float32)
        self.assertTrue(torch.equal(weight, w0 + STOCK_MARK))

    def test_nothing_shared_is_written(self):
        g, w0, scale = self._w0_scale(seed=5)
        adapter, _ = _lora(g, scale)
        snapshot = [t.clone() if hasattr(t, "clone") else t for t in adapter.weights]
        keys = set(adapter.loaded_keys)
        for mode in (dim.MODE_LYCORIS, dim.MODE_FORGE_FP32, dim.MODE_NO_MAGNITUDE):
            wrapped = self._mode(adapter, mode)
            for dtype in (torch.float32, torch.bfloat16):
                wrapped.calculate_weight(w0.to(dtype), "k", 0.7, 1.0, None, _identity, dtype)
        self.assertEqual(adapter.loaded_keys, keys)
        for now, then in zip(adapter.weights, snapshot):
            if hasattr(now, "clone"):
                self.assertTrue(torch.equal(now, then))
            else:
                self.assertEqual(now, then)

    def test_a_failure_merges_the_stock_way_and_is_logged_once_per_run(self):
        g, w0, scale = self._w0_scale(seed=6)
        broken = FakeLoRA(set(), (torch.ones(24, 4), torch.ones(5, 7), 1.0, None, scale, None))  # mm fails
        wrapped = self._mode(broken, dim.MODE_LYCORIS)
        before = dim.merge_stats().get("fallbacks", 0)
        with self.assertLogs("ai_studio_forge_parity", level="ERROR") as logs, \
                self.assertLogs("fake.comfy", level="ERROR"), mock.patch.object(dim, "_soft_empty_cache"):
            for _ in range(2):
                weight = w0.clone()
                out = wrapped.calculate_weight(weight, "k", 1.0, 1.0, None, _identity, torch.float32)
                self.assertIs(out, weight)
                self.assertTrue(torch.equal(out, w0))   # ComfyUI's own merge failed too: logged, weight unchanged
        # each run: the fp32 delta (no dora_scale), then the stock class with the dora_scale
        self.assertEqual(FakeLoRA.calls, [("k", torch.float32, False), ("k", torch.float32, True)] * 2)
        self.assertEqual(len(logs.records), 1)
        self.assertIn("merged the stock way", logs.output[0])
        self.assertEqual(dim.merge_stats()["fallbacks"] - before, 2)
        # review F2/F5: a new run (or a new clone of the node) logs it again instead of staying silent
        dim.RUN.reset()
        with self.assertLogs("ai_studio_forge_parity", level="ERROR") as logs, \
                self.assertLogs("fake.comfy", level="ERROR"), mock.patch.object(dim, "_soft_empty_cache"):
            wrapped.calculate_weight(w0.clone(), "k", 1.0, 1.0, None, _identity, torch.float32)
        self.assertEqual(len(logs.records), 1)

    def test_an_oom_empties_the_cache_before_the_stock_merge(self):
        g, w0, scale = self._w0_scale(seed=7)
        adapter, _ = _lora(g, scale)
        wrapped = self._mode(adapter, dim.MODE_LYCORIS)
        weight = w0.clone()
        # torch's own OOM class: inside ComfyUI ``comfy.model_management.is_oom`` decides, and it checks the class
        oom = getattr(torch, "OutOfMemoryError", None) or torch.cuda.OutOfMemoryError
        with mock.patch.object(dim, "decompose", side_effect=oom("CUDA out of memory. Tried to allocate")), \
                mock.patch.object(dim, "_soft_empty_cache") as empty, \
                self.assertLogs("ai_studio_forge_parity", level="ERROR") as logs:
            out = wrapped.calculate_weight(weight, "k", 1.0, 1.0, None, _identity, torch.float32)
        empty.assert_called_once_with()
        self.assertIn("OOM", logs.output[0])
        self.assertTrue(torch.equal(out, w0 + STOCK_MARK))

    def test_an_oom_that_comfys_adapter_swallows_still_empties_the_cache(self):
        """Review F2: ComfyUI's adapters build the full ``dW`` inside their own try/except, which logs an OOM and
        returns without a delta. That is handled as a possible OOM: the cache is emptied before the stock merge."""
        g, w0, scale = self._w0_scale(seed=8)
        adapter, _ = _lora(g, scale)
        wrapped = self._mode(adapter, dim.MODE_LYCORIS)
        oom = getattr(torch, "OutOfMemoryError", None) or torch.cuda.OutOfMemoryError
        real_delta = FakeLoRA._delta
        failures = iter([oom("CUDA out of memory. Tried to allocate 600.00 MiB")])

        def delta_once_oom(this, weight, dtype):
            failure = next(failures, None)
            if failure is not None:
                raise failure
            return real_delta(this, weight, dtype)

        with mock.patch.object(FakeLoRA, "_delta", delta_once_oom), \
                mock.patch.object(dim, "_soft_empty_cache") as empty, \
                self.assertLogs("fake.comfy", level="ERROR") as comfy_logs, \
                self.assertLogs("ai_studio_forge_parity", level="ERROR") as logs:
            out = wrapped.calculate_weight(w0.clone(), "k", 1.0, 1.0, None, _identity, torch.float32)
        self.assertIn("out of memory", comfy_logs.output[0])
        empty.assert_called_once_with()
        self.assertIn("possible OOM", logs.output[0])
        self.assertIn("built no weight delta", dim.RUN.first_cause[dim.MODE_LYCORIS])
        self.assertTrue(torch.equal(out, w0 + STOCK_MARK))   # the stock merge
        self.assertEqual(dim.RUN.fallbacks[dim.MODE_LYCORIS], 1)

    def test_comfys_adapter_never_writes_the_weight_it_is_handed_for_the_delta(self):
        """Review F1 (memory): the delta leaves the adapter before its ``+=``. The adapters that only read the weight's
        shape get a zero-stride stand-in (no fp32 copy of W0 while dW is built); GLoRA, whose dW reads W0, gets W0."""
        g, w0, scale = self._w0_scale(seed=9)
        adapter, delta = _lora(g, scale)
        weight = w0.to(torch.bfloat16)
        base = weight.clone()
        out = self._mode(adapter, dim.MODE_LYCORIS).calculate_weight(weight, "k", 1.0, 1.0, None, _identity,
                                                                      torch.bfloat16)
        self.assertTrue(torch.equal(out, lycoris_merge(base, delta, scale, True).to(torch.bfloat16)))
        (handed,) = FakeLoRA.handed
        self.assertEqual((tuple(handed.shape), handed.dtype, handed.stride()), ((24, 16), torch.float32, (0, 0)))
        a1, a2 = torch.randn(16, 4, generator=g) * 0.3, torch.randn(4, 16, generator=g) * 0.3
        b1, b2 = torch.randn(24, 4, generator=g) * 0.3, torch.randn(4, 16, generator=g) * 0.3
        glora = FakeGLoRA(set(), (a1, a2, b1, b2, 2.0, scale))
        self._mode(glora, dim.MODE_LYCORIS).calculate_weight(base.clone(), "k", 1.0, 1.0, None, _identity,
                                                             torch.bfloat16)
        (handed,) = FakeGLoRA.handed
        self.assertEqual(handed.dtype, torch.float32)
        self.assertTrue(torch.equal(handed, base.float()))
        # no "ERROR" from the adapter: the capture is not an Exception it swallows
        with self.assertNoLogs("fake.comfy", level="ERROR"):
            self._mode(adapter, dim.MODE_FORGE_FP32).calculate_weight(w0.clone(), "k", 1.0, 1.0, None, _identity,
                                                                      torch.float32)

    def test_the_run_account_reports_merges_and_fallbacks_per_run(self):
        """Review F2/F5: the OUTER_SAMPLE wrapper starts each sampling run with a fresh account and reports it."""
        g, w0, scale = self._w0_scale(seed=10)
        adapter, _ = _lora(g, scale)
        wrapped = self._mode(adapter, dim.MODE_LYCORIS)
        rebuilt = type(wrapped)(wrapped.loaded_keys, tuple(wrapped.weights))   # LowVramPatch's per-forward rebuild
        broken = self._mode(FakeLoRA({"b"}, (torch.ones(24, 4), torch.ones(5, 7), 1.0, None, scale, None)),
                            dim.MODE_LYCORIS)
        wrapper = dim.dora_outer_sample_wrapper()

        def sampling(merges, failures):
            for patch in (wrapped, rebuilt)[:merges] + (rebuilt,) * max(0, merges - 2):
                patch.calculate_weight(w0.to(torch.bfloat16), "k", 1.0, 1.0, None, _identity, torch.bfloat16)
            for _ in range(failures):
                broken.calculate_weight(w0.clone(), "b", 1.0, 1.0, None, _identity, torch.float32)
            return "latent"

        executor = mock.Mock(side_effect=sampling)
        dim.RUN.merged(dim.MODE_LYCORIS, "stale", adapter)   # from an earlier run: not reported
        with self.assertLogs("ai_studio_forge_parity", level="INFO") as logs, \
                self.assertLogs("fake.comfy", level="ERROR"), mock.patch.object(dim, "_soft_empty_cache"):
            self.assertEqual(wrapper(executor, 3, 2), "latent")
        executor.assert_called_once_with(3, 2)
        lines = [record.getMessage() for record in logs.records]
        self.assertEqual(len(lines), 3)
        self.assertIn("merged the stock way", lines[0])   # the first failure, when it happens
        self.assertEqual(lines[1], "[DoRA mode] LyCORIS fp32 (as trained): 1 DoRA patches merged this run "
                                   "(3 merges: LowVramPatch merges on every forward)")
        self.assertTrue(lines[2].startswith("[DoRA mode] LyCORIS fp32 (as trained): 2 merges fell back to the "
                                            "stock way this run - first cause: possible OOM: "), lines[2])
        self.assertEqual([record.levelname for record in logs.records], ["ERROR", "INFO", "WARNING"])
        # the next run starts over and logs its first failure again; a run that merges nothing reports nothing
        with self.assertLogs("ai_studio_forge_parity", level="INFO") as logs, \
                self.assertLogs("fake.comfy", level="ERROR"), mock.patch.object(dim, "_soft_empty_cache"):
            wrapper(mock.Mock(side_effect=lambda: sampling(1, 1)))
        self.assertEqual([record.levelname for record in logs.records], ["ERROR", "INFO", "WARNING"])
        self.assertIn("1 DoRA patches merged this run", logs.records[1].getMessage())
        self.assertNotIn("merges:", logs.records[1].getMessage())
        with self.assertNoLogs("ai_studio_forge_parity", level="INFO"):
            wrapper(mock.Mock(side_effect=lambda: sampling(0, 0)))
        # the report is written even when the run fails
        with self.assertLogs("ai_studio_forge_parity", level="INFO") as logs, self.assertRaises(KeyError):
            wrapper(mock.Mock(side_effect=lambda: (sampling(1, 0), {}["interrupted"])))
        self.assertIn("1 DoRA patches merged this run", logs.output[0])


# --------------------------------------------------------------------------------------------------
# Patch rewrite and node (stand-in tensors are not needed: only the dora_scale slot is read)
# --------------------------------------------------------------------------------------------------


def _adapter(cls, dora=True, reshape=None, tag="a"):
    size = {FakeLoRA: 6, FakeLoKr: 9, FakeLoHa: 8, FakeGLoRA: 6, FakeOFT: 4, FakeLoRASubclass: 6}[cls]
    weights = [f"{tag}{i}" for i in range(size)]
    weights[cls.DORA] = f"{tag}-dora" if dora else None
    if cls in (FakeLoRA, FakeLoRASubclass):
        weights[5] = reshape
    return cls({f"{tag}.keys"}, tuple(weights))


class RewriteTests(unittest.TestCase):
    def _patches(self):
        self.dora_lora = _adapter(FakeLoRA, tag="l")
        self.plain_lora = _adapter(FakeLoRA, dora=False, tag="p")
        self.lokr = _adapter(FakeLoKr, tag="k")
        self.loha = _adapter(FakeLoHa, tag="h")
        self.glora = _adapter(FakeGLoRA, tag="g")
        self.oft = _adapter(FakeOFT, tag="o")
        self.resized = _adapter(FakeLoRA, reshape=[24, 16], tag="r")
        self.foreign = _adapter(FakeLoRASubclass, tag="f")
        merge = [("merged-weight", _identity), (0.5, _adapter(FakeLoKr, tag="m"), 1.0, None, None)]
        return {
            "a.weight": [(1.0, self.dora_lora, 1.0, None, None), (0.7, self.plain_lora, 1.0, None, None)],
            "b.weight": [(0.5, self.lokr, 1.0, (0, 0, 8), "fn"), (1.0, self.loha, 0.9, None, None)],
            "c.weight": [(1.0, self.glora, 1.0, None, None), (1.0, self.oft, 1.0, None, None)],
            "d.weight": [(1.0, self.resized, 1.0, None, None), (1.0, self.foreign, 1.0, None, None)],
            "e.weight": [(1.0, merge, 1.0, None, None)],
            "f.bias": [(1.0, ("diff", ("tensor",)), 1.0, None, None)],
        }

    def test_dora_adapters_become_mode_subclasses_and_the_rest_stays(self):
        patches = self._patches()
        snapshot = {key: list(items) for key, items in patches.items()}
        plan = dim.rewrite_patches(patches, dim.MODE_LYCORIS, _specs())
        self.assertEqual(patches, snapshot)                       # the given lists are not changed
        self.assertEqual(set(plan.changes), {"a.weight", "b.weight", "c.weight", "e.weight"})
        self.assertEqual(plan.changed, 5)
        self.assertEqual(dict(plan.dora), {"LoRAAdapter": 1, "LoKrAdapter": 2, "LoHaAdapter": 1, "GLoRAAdapter": 1})
        self.assertEqual(dict(plan.stock), {"OFTAdapter (stock in sam-extra too)": 1,
                                            "LoRAAdapter with reshape_weight": 1})
        a = plan.changes["a.weight"]
        self.assertIsInstance(a[0][1], FakeLoRA)
        self.assertEqual(getattr(type(a[0][1]), dim.MODE_ATTR), dim.MODE_LYCORIS)
        self.assertIs(a[0][1].weights, self.dora_lora.weights)
        self.assertIs(a[0][1].loaded_keys, self.dora_lora.loaded_keys)
        self.assertIs(a[1], snapshot["a.weight"][1])             # the non-DoRA patch tuple itself
        b = plan.changes["b.weight"]
        self.assertEqual((b[0][0], b[0][2], b[0][3], b[0][4]), (0.5, 1.0, (0, 0, 8), "fn"))
        self.assertEqual((b[1][0], b[1][2]), (1.0, 0.9))
        self.assertIs(plan.changes["c.weight"][1][1], self.oft)
        e = plan.changes["e.weight"][0][1]
        self.assertIsInstance(e, list)
        self.assertIs(e[0], patches["e.weight"][0][1][0])
        self.assertEqual(getattr(type(e[1][1]), dim.MODE_ATTR), dim.MODE_LYCORIS)
        self.assertNotIn("d.weight", plan.changes)                # reshape LoRA and a foreign subclass stay

    def test_switching_and_undoing_the_mode(self):
        plan = dim.rewrite_patches(self._patches(), dim.MODE_LYCORIS, _specs())
        patched = {key: items for key, items in plan.changes.items()}
        again = dim.rewrite_patches(patched, dim.MODE_LYCORIS, _specs())
        self.assertEqual(again.changes, {})
        self.assertEqual(sum(again.dora.values()), 5)
        off = dim.rewrite_patches(patched, dim.MODE_NO_MAGNITUDE, _specs())
        self.assertEqual(off.changed, 5)
        self.assertEqual(getattr(type(off.changes["a.weight"][0][1]), dim.MODE_ATTR), dim.MODE_NO_MAGNITUDE)
        stock = dim.rewrite_patches(patched, dim.MODE_FORGE, _specs())
        self.assertIs(type(stock.changes["a.weight"][0][1]), FakeLoRA)
        self.assertIs(type(stock.changes["e.weight"][0][1][1][1]), FakeLoKr)
        self.assertEqual(dim.rewrite_patches(self._patches(), dim.MODE_FORGE, _specs()).changes, {})
        with self.assertRaises(ValueError):
            dim.rewrite_patches({}, "fp32", _specs())


class NodeTests(unittest.TestCase):
    def setUp(self):
        patcher = mock.patch.object(dim, "adapter_specs", return_value=_specs())
        patcher.start()
        self.addCleanup(patcher.stop)
        self.node = nodes.ForgeNeoDoraInferenceMode()

    def _model(self, dora=True):
        model = FakePatcher()
        model.add_patches({"a.weight": _adapter(FakeLoKr, dora=dora, tag="k"),
                           "b.weight": _adapter(FakeLoRA, dora=False, tag="p")}, 0.8)
        return model

    def test_off_returns_the_model(self):
        model = self._model()
        self.assertIs(self.node.patch(model, False, LYCORIS)[0], model)

    def test_lycoris_returns_a_clone_with_a_new_uuid_and_leaves_the_input(self):
        model = self._model()
        before = {key: [tuple(p) for p in items] for key, items in model.patches.items()}
        uid = model.patches_uuid
        with self.assertLogs("ai_studio_forge_parity", level="INFO") as logs:
            (patched,) = self.node.patch(model, True, LYCORIS)
        self.assertIsNot(patched, model)
        self.assertIs(patched.parent, model)
        self.assertNotEqual(patched.patches_uuid, uid)
        self.assertEqual(model.patches_uuid, uid)
        self.assertEqual({key: [tuple(p) for p in items] for key, items in model.patches.items()}, before)
        self.assertIs(type(model.patches["a.weight"][0][1]), FakeLoKr)
        self.assertEqual(getattr(type(patched.patches["a.weight"][0][1]), dim.MODE_ATTR), dim.MODE_LYCORIS)
        self.assertIs(patched.patches["b.weight"][0], model.patches["b.weight"][0])
        self.assertIn("LyCORIS fp32 (as trained): 1 DoRA patches", logs.output[0])
        self.assertIn("LoRAs added after this node are not affected", logs.output[0])
        # a LoRA added after the node stays stock
        later = patched.clone()
        later.add_patches({"a.weight": _adapter(FakeLoHa, tag="h")}, 1.0)
        self.assertIs(type(later.patches["a.weight"][1][1]), FakeLoHa)
        # the same mode again changes nothing; stock undoes it
        self.assertIs(self.node.patch(patched, True, LYCORIS)[0], patched)
        (stock,) = self.node.patch(patched, True, dim.MODE_LABELS[dim.MODE_FORGE])
        self.assertIsNot(stock, patched)
        self.assertIs(type(stock.patches["a.weight"][0][1]), FakeLoKr)
        self.assertNotEqual(stock.patches_uuid, patched.patches_uuid)

    def test_the_clone_carries_one_run_report_wrapper(self):
        """Review F2/F5: the per-run account rides on the node's clone (OUTER_SAMPLE, ``WRAPPER_KEY``): one per
        MODEL, replaced when the mode changes, gone in stock mode; the input MODEL never gets one."""
        model = self._model()
        with self.assertLogs("ai_studio_forge_parity", level="INFO"):
            (patched,) = self.node.patch(model, True, LYCORIS)
        self.assertEqual(model.wrappers, {})
        self.assertEqual(list(patched.wrappers), ["outer_sample"])   # WrappersMP.OUTER_SAMPLE outside ComfyUI
        (first,) = patched.run_wrappers()
        with self.assertLogs("ai_studio_forge_parity", level="INFO"):
            (fp32,) = self.node.patch(patched, True, dim.MODE_LABELS[dim.MODE_FORGE_FP32])
        (second,) = fp32.run_wrappers()
        self.assertIsNot(second, first)
        self.assertEqual(patched.run_wrappers(), [first])
        with self.assertLogs("ai_studio_forge_parity", level="INFO"):
            (stock,) = self.node.patch(fp32, True, dim.MODE_LABELS[dim.MODE_FORGE])
        self.assertEqual(stock.run_wrappers(), [])
        # a fresh account when the node builds a clone
        dim.RUN.merges[dim.MODE_LYCORIS] += 5
        with self.assertLogs("ai_studio_forge_parity", level="INFO"):
            self.node.patch(self._model(), True, LYCORIS)
        self.assertEqual(dim.RUN.merges, {})

    def test_a_model_without_dora_patches_is_returned_with_one_warning(self):
        model = self._model(dora=False)
        with self.assertLogs("ai_studio_forge_parity", level="WARNING") as logs:
            self.assertIs(self.node.patch(model, True, LYCORIS)[0], model)
        self.assertEqual(len(logs.records), 1)
        self.assertIn("no DoRA patch", logs.output[0])
        self.assertIn("after the LoRA loaders", logs.output[0])
        model.add_patches({"c.weight": _adapter(FakeOFT, tag="o")})
        with self.assertLogs("ai_studio_forge_parity", level="WARNING") as logs:
            self.assertIs(self.node.patch(model, True, LYCORIS)[0], model)
        self.assertIn("left stock: OFT (stock in sam-extra too) 1", logs.output[0])

    def test_stock_on_an_untouched_model_returns_it(self):
        model = self._model()
        with self.assertLogs("ai_studio_forge_parity", level="INFO") as logs:
            self.assertIs(self.node.patch(model, True, dim.MODE_LABELS[dim.MODE_FORGE])[0], model)
        self.assertIn("already merge this way", logs.output[0])

    def test_unknown_mode_and_non_model_inputs_raise(self):
        with self.assertRaises(ValueError):
            self.node.patch(self._model(), True, "LyCORIS")
        with self.assertRaises(RuntimeError):
            self.node.patch(object(), True, LYCORIS)
        self.assertEqual(dim.mode_key("lycoris"), dim.MODE_LYCORIS)


# --------------------------------------------------------------------------------------------------
# Registration, module rules, docs
# --------------------------------------------------------------------------------------------------


class RegistrationTests(unittest.TestCase):
    def test_node_is_registered_and_named(self):
        self.assertIs(bundled_pack.NODE_CLASS_MAPPINGS[nodes.NODE_ID], nodes.ForgeNeoDoraInferenceMode)
        self.assertEqual(nodes.NODE_ID, "ForgeNeoDoraInferenceMode")
        self.assertEqual(bundled_pack.NODE_DISPLAY_NAME_MAPPINGS[nodes.NODE_ID],
                         "Forge Neo DoRA Inference Mode (sam-extra)")
        self.assertEqual(nodes.NODE_CLASS_MAPPINGS, {nodes.NODE_ID: nodes.ForgeNeoDoraInferenceMode})
        self.assertEqual(nodes.ForgeNeoDoraInferenceMode.CATEGORY, "AI Studio/Forge Neo parity/Loaders")
        # the app's Anima custom-workflow check rejects a model-chain node whose ID says "lora"
        # (core/comfy_workflow_compiler.py) and follows only an input called "model"
        self.assertNotIn("lora", nodes.NODE_ID.casefold())

    def test_inputs(self):
        node = nodes.ForgeNeoDoraInferenceMode
        required = node.INPUT_TYPES()["required"]
        self.assertEqual(list(required), ["model", "enabled", "mode"])
        self.assertEqual(required["model"][0], "MODEL")
        self.assertEqual(required["enabled"][1]["default"], True)
        self.assertEqual(required["mode"][0], ["LyCORIS fp32 (as trained)", "Forge/Comfy fp32",
                                               "Forge/Comfy (stock)", "DoRA off (no magnitude)"])
        self.assertEqual(required["mode"][1]["default"], "LyCORIS fp32 (as trained)")
        self.assertEqual((node.RETURN_TYPES, node.FUNCTION), (("MODEL",), "patch"))

    def test_module_names_and_imports(self):
        names = sorted(path.name for path in PACK_DIR.glob("dora*.py"))
        self.assertEqual(names, ["dora_infer_mode.py", "dora_infer_mode_nodes.py"])
        heavy = []
        patched = []
        for name in names:
            tree = ast.parse((PACK_DIR / name).read_text(encoding="utf-8"))
            for item in tree.body:
                if isinstance(item, (ast.Import, ast.ImportFrom)):
                    modules = [alias.name for alias in item.names] if isinstance(item, ast.Import) else [item.module or ""]
                    heavy += [f"{name}:{m}" for m in modules if m.split(".")[0] in ("torch", "comfy")]
            for item in ast.walk(tree):  # no ComfyUI function is swapped (sam-extra's global hook is not ported)
                if isinstance(item, ast.Call) and getattr(item.func, "id", "") == "setattr":
                    patched.append(f"{name}:{item.lineno}")
                if isinstance(item, (ast.Assign, ast.AugAssign)):
                    targets = item.targets if isinstance(item, ast.Assign) else [item.target]
                    patched += [f"{name}:{item.lineno}" for t in targets
                                if isinstance(t, ast.Attribute) and t.attr in ("weight_decompose", "calculate_weight")]
        self.assertEqual(heavy, [])
        self.assertEqual(patched, [])

    def test_version_lockstep(self):
        self.assertEqual(bundled_pack.__version__, "1.9.0")
        self.assertEqual(PACK_VERSION, bundled_pack.__version__)
        source = (Path(__file__).resolve().parent.parent / "core" / "comfy_node_pack.py").read_text(encoding="utf-8")
        self.assertIn("# Same (unreleased) 1.9.0: new ForgeNeoDoraInferenceMode", source)

    def test_readme(self):
        readme = (PACK_DIR / "README.md").read_text(encoding="utf-8")
        custom = readme.split("### Custom ComfyUI workflows only", 1)[1].split("##", 1)[0]
        self.assertIn("`ForgeNeoDoraInferenceMode`", custom)
        changes = " ".join(readme.split("## 1.9.0 changes", 1)[1].split("\n## ", 1)[0].split())
        for text in ("New `ForgeNeoDoraInferenceMode`", "Forge Neo DoRA Inference Mode (sam-extra)",
                     "`LyCORIS fp32 (as trained)` (default)", "`Forge/Comfy fp32`", "`Forge/Comfy (stock)`",
                     "`DoRA off (no magnitude)`", "`W0 + s*(W_dora - W0)`", "`patches_uuid`",
                     "No ComfyUI function is patched", "after the LoRA loaders", "LoRAs loaded after it",
                     "OFT/BOFT", "`reshape_weight`", "returned unchanged with one warning",
                     "licensed to this pack under the MIT License by that author",
                     # review F1: LyCORIS' operations; F3: the W0 ComfyUI hands the patch; F1/F2: memory, OOM;
                     # F2/F5: what the console shows and when
                     "`vector_norm`", "`1 * (r - 1) + 1`", "an fp32-stored weight keeps fp16's rounding of `W0`",
                     "at most two fp32 copies", "an OOM empties the cache first, also one that ComfyUI's adapter logs "
                     "and swallows", "one line when the node runs", "ComfyUI caches the node",
                     "`ai_studio_forge_parity.dora_infer_mode`", "once per kind per run"):
            with self.subTest(text=text):
                self.assertIn(text, changes)
        dora_part = changes.split("New `ForgeNeoDoraInferenceMode`", 1)[1]   # MOSS Spectrum's text comes first
        self.assertNotIn("One console line per run", dora_part)
        self.assertNotIn("about four fp32 copies", dora_part)
        description = " ".join(nodes.ForgeNeoDoraInferenceMode.DESCRIPTION.split())
        self.assertIn("from the W0 ComfyUI hands the patch", description)
        self.assertIn("not for fp32-stored weights", description)

    def test_notices(self):
        notices = (PACK_DIR / "THIRD_PARTY_NOTICES.md").read_text(encoding="utf-8")
        section = " ".join(notices.split("## DoRA inference mode (pack 1.9.0)", 1)[1].split("\n## ", 1)[0].split())
        for text in ("`dora_infer_mode.py`", "`dora_infer_mode_nodes.py`", "`ForgeNeoDoraInferenceMode`",
                     "`sam3ext/dora_infer_mode.py`", "`decompose`", "`_can_return_fp32`", "`_is_oom`",
                     "licensed to this pack under the MIT License by that author", "`LICENSES/sam-extra-MIT.txt`",
                     "KohakuBlueleaf/LyCORIS", "Apache-2.0", "no LyCORIS code is included or imported",
                     "the pack never imports the extension", "`tests/test_comfy_dora_infer_mode.py`",
                     "`RunAccount`", "operation for operation", "`multiplier * (scale - 1) + 1`"):
            with self.subTest(section=text):
                self.assertIn(text, section)
        shared = notices.split("## Code shared with the sam-extra Forge extension", 1)[1].split("\n## ", 1)[0]
        row = next(line for line in shared.splitlines() if line.startswith("| `dora_infer_mode.py`"))
        for name in ("`decompose`", "`_can_return_fp32`", "`_is_oom`", "`sam3ext/dora_infer_mode.py`"):
            self.assertIn(name, row)
        self.assertIn("**DoRA weight decomposition** (`dora_infer_mode.decompose`)", shared)

    def test_module_docstring_names_the_origin(self):
        doc = " ".join((dim.__doc__ or "").split())
        for text in ("sam3ext/dora_infer_mode.py", "``decompose`` (lines 153-186)", "``_can_return_fp32`` (189-202)",
                     "``MODES`` (48-52)", "``_make_wrapper`` (243-278)", "659d354", "MIT grant",
                     "No torch or ComfyUI import at module level"):
            with self.subTest(text=text):
                self.assertIn(text, doc)


if __name__ == "__main__":
    unittest.main()
