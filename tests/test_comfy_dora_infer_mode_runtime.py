r"""Pack 1.9.0 DoRA inference mode on the real ComfyUI runtime (CPU): ComfyUI's own ``comfy.lora.load_lora`` and
adapter classes (LoRA, LoCon conv, LoKr, LoHa, GLoRA, OFT), ``ModelPatcher`` (``add_patches``, ``clone``,
``patch_model``/``unpatch_model``, ``load_models_gpu`` and its ``patches_uuid`` gating), ``LowVramPatch`` (with the
adapters its ``prepare`` rebuilds) and ``comfy.lora.calculate_weight`` (offset patches), on a tiny model:

* ``LyCORIS fp32 (as trained)`` merges every adapter kind with a ``dora_scale`` into the LyCORIS training merge
  (``tests/_dora_oracle``) bit for bit on both DoRA axes, at strengths 1 and 0.5, also where LyCORIS' operations
  round differently from ComfyUI's (a conv input axis, ``m / ||W0 + dW||`` outside [0.5, 2]); ``unpatch_model``
  restores the bits; ComfyUI's stock merge differs on the output axis and gives the same bits on a linear input axis
  (fp32 on CPU; a conv input axis within fp32 rounding - ComfyUI reduces its norm in another order);
* the full-load path with ComfyUI's GPU merge dtype (fp16, ``lora_compute_dtype`` mocked) rounds the fp32 result
  once to the stored bf16; LowVramPatch keeps the mode through ``prepare`` and writes the result in place in the
  layer's dtype; an offset patch (fused qkv) changes only its rows;
* the clone is isolated: the input MODEL's patch lists, tuples and adapters are unchanged and ComfyUI loads stock /
  LyCORIS / stock weights when it switches between the input, the node's clone and a sibling clone;
* stacked LoRAs (two DoRA files on both axes, a plain LoRA, a DoRA LoRA added after the node) merge in order, the
  plain one untouched and the later one stock; ``strength_model`` is ComfyUI's, applied before the adapter;
* ``Forge/Comfy fp32`` equals the stock merge in fp32, ``DoRA off`` equals the file without ``dora_scale``, OFT's
  DoRA stays stock, a MODEL without DoRA patches comes back unchanged, and no ComfyUI function is replaced;
* an OOM that ComfyUI's own adapter code swallows while building ``dW`` empties the cache before the stock merge,
  and the clone's OUTER_SAMPLE wrapper, run by ComfyUI's own wrapper executor around a model load, reports the run.

Skipped unless ``AISTUDIO_COMFY_TEST_ROOT`` is set. Run it with the portable python, the repo first on ``sys.path``,
``CUDA_VISIBLE_DEVICES=-1`` (ComfyUI is forced to ``--cpu`` before its model management loads; no server, no GPU) and
``PYTHONDONTWRITEBYTECODE=1`` (nothing is written into the ComfyUI install)::

    $env:CUDA_VISIBLE_DEVICES='-1'; $env:PYTHONDONTWRITEBYTECODE='1'
    $env:AISTUDIO_COMFY_TEST_ROOT='C:\ComfyUI_windows_portable\ComfyUI'
    C:\ComfyUI_windows_portable\python_embeded\python.exe -X utf8 -c "import sys, unittest; sys.path.insert(0, r'<repo>'); unittest.main(module=None, argv=['x', 'tests.test_comfy_dora_infer_mode_runtime', '-v'])"
"""
from __future__ import annotations

import os
import sys
import unittest
from unittest import mock

_COMFY_ROOT = os.environ.get("AISTUDIO_COMFY_TEST_ROOT")
if _COMFY_ROOT:  # force --cpu before anything loads ComfyUI's model management
    if _COMFY_ROOT not in sys.path:
        sys.path.append(_COMFY_ROOT)
    from comfy.cli_args import args as _comfy_args  # noqa: E402

    _comfy_args.cpu = True

from tests import _dora_oracle as oracle  # noqa: E402
from tests._optional_deps import bind_torch, requires_torch  # noqa: E402

torch = None  # filled by bind_torch()

LYCORIS = "LyCORIS fp32 (as trained)"
FORGE_FP32 = "Forge/Comfy fp32"
STOCK = "Forge/Comfy (stock)"
DORA_OFF = "DoRA off (no magnitude)"
OUT, IN, RANK, ALPHA = 24, 16, 4, 2.0
KINDS = ("lora", "locon", "lokr", "loha", "glora")
ADAPTER_MODULES = ("base", "lora", "lokr", "loha", "glora", "oft", "boft")


@requires_torch
@unittest.skipUnless(_COMFY_ROOT, "Set AISTUDIO_COMFY_TEST_ROOT to run the DoRA mode on the real ComfyUI runtime (CPU)")
class DoraModeOnRealComfyTests(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        bind_torch(globals())
        import comfy.model_management as management

        if management.get_torch_device().type != "cpu":
            raise unittest.SkipTest("ComfyUI did not start on CPU - refusing to touch the GPU")
        import importlib

        import comfy.lora
        import comfy.model_patcher
        import comfy.weight_adapter

        from comfy_custom_nodes.ai_studio_forge_parity import dora_infer_mode, dora_infer_mode_nodes

        cls.mm, cls.lora, cls.patcher_module, cls.wa = management, comfy.lora, comfy.model_patcher, comfy.weight_adapter
        cls.dim, cls.node = dora_infer_mode, dora_infer_mode_nodes.ForgeNeoDoraInferenceMode()
        cls.adapter_modules = {name: importlib.import_module(f"comfy.weight_adapter.{name}") for name in ADAPTER_MODULES}
        cls.cpu = torch.device("cpu")
        cls._threads = torch.get_num_threads()   # restored in tearDownClass (later modules' bit-equal tests)
        torch.set_num_threads(4)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls._threads)

    def setUp(self):
        self.dim.RUN.reset()

    def tearDown(self):
        self.mm.unload_all_models()

    # -- a tiny model and LyCORIS-style files ---------------------------------------------------------------------

    def _model(self, dtype=None, seed=0):
        g = torch.Generator().manual_seed(seed)

        class Tiny(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.a = torch.nn.Linear(IN, OUT, bias=False)
                self.c = torch.nn.Conv2d(8, 12, 3, bias=False)
                self.q = torch.nn.Linear(IN, 3 * OUT, bias=False)   # a fused qkv
                with torch.no_grad():
                    for param in self.parameters():
                        param.copy_(torch.randn(param.shape, generator=g) * 0.05)

        model = Tiny()
        if dtype is not None:
            model.to(dtype)
        return model

    def _file(self, kind, w0, wd_on_output, seed, dora=True, prefix="x", far=False):
        """A LyCORIS-style state dict for one module and its alpha-scaled dW in fp32 (``dW`` of GLoRA reads W0);
        ``far``: a magnitude whose ratio to ``||W0 + dW||`` leaves [0.5, 2] on some rows (columns)."""
        g = torch.Generator().manual_seed(100 + seed)

        def r(*shape, s=0.2):
            return torch.randn(*shape, generator=g) * s

        out, inp = w0.shape[0], w0.shape[1]
        sd = {f"{prefix}.alpha": torch.tensor(ALPHA)}
        if kind in ("lora", "locon"):
            kernel = tuple(w0.shape[2:])
            up, down = r(out, RANK, *[1] * len(kernel)), r(RANK, inp, *kernel)
            sd.update({f"{prefix}.lora_up.weight": up, f"{prefix}.lora_down.weight": down})
            delta = torch.mm(up.flatten(1), down.flatten(1)).reshape(w0.shape) * (ALPHA / RANK)
        elif kind == "lokr":
            w1, w2a, w2b = r(4, 2, s=0.5), r(out // 4, RANK), r(RANK, inp // 2)
            sd.update({f"{prefix}.lokr_w1": w1, f"{prefix}.lokr_w2_a": w2a, f"{prefix}.lokr_w2_b": w2b})
            delta = torch.kron(w1, w2a.mm(w2b)) * (ALPHA / RANK)
        elif kind == "loha":
            m = [r(out, RANK, s=0.4), r(RANK, inp, s=0.4), r(out, RANK, s=0.4), r(RANK, inp, s=0.4)]
            sd.update({f"{prefix}.hada_w1_a": m[0], f"{prefix}.hada_w1_b": m[1], f"{prefix}.hada_w2_a": m[2],
                       f"{prefix}.hada_w2_b": m[3]})
            delta = (m[0].mm(m[1]) * m[2].mm(m[3])) * (ALPHA / RANK)
        elif kind == "glora":   # the new GLoRA layout (out != in): dW = W0 a1 a2 + b1 b2
            a1, a2, b1, b2 = r(inp, RANK), r(RANK, inp), r(out, RANK), r(RANK, inp)
            sd.update({f"{prefix}.a1.weight": a1, f"{prefix}.a2.weight": a2, f"{prefix}.b1.weight": b1,
                       f"{prefix}.b2.weight": b2})
            delta = (w0.float().mm(a1).mm(a2) + b1.mm(b2)) * (ALPHA / RANK)
        else:
            raise AssertionError(kind)
        if far:
            scale = oracle.far_magnitude(torch, w0.float() + delta, wd_on_output, g)
        else:
            scale = oracle.trained_magnitude(torch, w0.float() + delta, wd_on_output, g)
        if dora:
            sd[f"{prefix}.dora_scale"] = scale
        return sd, delta, scale

    def _patcher(self, model, *files):
        """A ModelPatcher with each ``(state dict, {prefix: model key}, strength)`` added like LoraLoaderModelOnly."""
        patcher = self.patcher_module.ModelPatcher(model, self.cpu, self.cpu)
        for sd, key_map, strength in files:
            loaded = self.lora.load_lora(sd, key_map)
            self.assertEqual(len(loaded), len(key_map))
            patcher.add_patches(loaded, strength)
        return patcher

    def _merged(self, patcher, key):
        """The weight ComfyUI merges for ``key`` (patch_model), and the original bits back after unpatch_model."""
        before = self._param(patcher.model, key).detach().clone()
        patcher.patch_model()
        merged = self._param(patcher.model, key).detach().clone()
        patcher.unpatch_model()
        self.assertTrue(torch.equal(self._param(patcher.model, key), before), "unpatch_model did not restore")
        return merged

    @staticmethod
    def _param(model, key):
        module, name = key.rsplit(".", 1)
        return getattr(getattr(model, module), name)

    def _mode(self, patcher, label):
        (out,) = self.node.patch(patcher, True, label)
        return out

    def _assert_lycoris(self, got, expected):
        self.assertTrue(torch.equal(got, expected), f"max |d| {(got.float() - expected.float()).abs().max().item()}")

    # -- tests ------------------------------------------------------------------------------------------------------

    def test_every_adapter_kind_merges_into_the_lycoris_training_merge_on_both_axes(self):
        model = self._model()
        for kind in KINDS:
            key = "c.weight" if kind == "locon" else "a.weight"
            w0 = self._param(model, key).detach().clone()
            for wd_on_output in (True, False):
                for strength in (1.0, 0.5):
                    with self.subTest(kind=kind, output_axis=wd_on_output, strength=strength):
                        sd, delta, scale = self._file(kind, w0, wd_on_output, seed=len(kind) + wd_on_output)
                        stock = self._patcher(model, (sd, {"x": key}, strength))
                        adapter = stock.patches[key][0][1]
                        lyco = self._mode(stock, LYCORIS)
                        self.assertIsNot(lyco, stock)
                        self.assertIsInstance(lyco.patches[key][0][1], type(adapter))
                        self.assertIs(type(stock.patches[key][0][1]), type(adapter))
                        got = self._merged(lyco, key)
                        self._assert_lycoris(got, oracle.lycoris_merge(torch, w0, delta, scale, wd_on_output, strength))
                        plain = self._merged(stock, key)
                        if wd_on_output:   # ComfyUI divides by ||W0||
                            self.assertGreater((plain - got).abs().max().item(), 1e-4)
                        elif kind != "locon":   # the same formula in fp32 on CPU, a linear weight: the same bits
                            self.assertTrue(torch.equal(plain, got))
                        else:   # a conv input axis: ComfyUI's transpose-reshape norm sums in another order
                            torch.testing.assert_close(plain, got, rtol=1e-6, atol=1e-8)

    def test_lycoris_follows_lycoris_where_its_operations_round_differently(self):
        """Review F1: the magnitude step ``1 * (scale - 1) + 1`` with ratios outside [0.5, 2], on every adapter kind,
        both axes, strengths 1 and 0.5 - bit for bit, conv included."""
        model = self._model(seed=12)
        for kind in KINDS:
            key = "c.weight" if kind == "locon" else "a.weight"
            w0 = self._param(model, key).detach().clone()
            for wd_on_output in (True, False):
                for strength in (1.0, 0.5):
                    with self.subTest(kind=kind, output_axis=wd_on_output, strength=strength):
                        sd, delta, scale = self._file(kind, w0, wd_on_output, seed=20 + len(kind), far=True)
                        lyco = self._mode(self._patcher(model, (sd, {"x": key}, strength)), LYCORIS)
                        self._assert_lycoris(self._merged(lyco, key),
                                             oracle.lycoris_merge(torch, w0, delta, scale, wd_on_output, strength))

    def test_the_adapter_layout_is_the_one_the_node_reads(self):
        model = self._model()
        w0 = model.a.weight.detach()
        for kind in KINDS:
            with self.subTest(kind=kind):
                sd, _, _ = self._file(kind, model.c.weight.detach() if kind == "locon" else w0, True, seed=1)
                adapter = self.lora.load_lora(sd, {"x": "k"})["k"]
                name = type(adapter).__name__
                spec = self.dim.SUPPORTED_ADAPTERS[name]
                self.assertIs(adapter.weights[spec.dora_index], sd["x.dora_scale"])
                self.assertIn(type(adapter), self.dim.adapter_specs())
        oft_scale = torch.ones(OUT, 1)
        oft = self.lora.load_lora({"x.oft_blocks": torch.zeros(4, 6, 6), "x.dora_scale": oft_scale}, {"x": "k"})["k"]
        self.assertEqual(type(oft).__name__, "OFTAdapter")
        self.assertIs(oft.weights[self.dim.STOCK_ONLY_ADAPTERS["OFTAdapter"].dora_index], oft_scale)
        self.assertEqual(set(self.dim.SUPPORTED_ADAPTERS) | set(self.dim.STOCK_ONLY_ADAPTERS),
                         {cls.__name__ for cls in self.wa.adapters})

    def test_the_gpu_merge_dtype_rounds_the_fp32_result_once(self):
        model = self._model(dtype=torch.bfloat16, seed=3)
        w0 = model.a.weight.detach().clone()
        sd, delta, scale = self._file("lokr", w0, True, seed=3)
        stock = self._patcher(model, (sd, {"x": "a.weight"}, 1.0))
        lyco = self._mode(stock, LYCORIS)
        # ComfyUI merges in lora_compute_dtype: fp16 on most GPUs (model_management.should_use_fp16)
        with mock.patch.object(self.mm, "lora_compute_dtype", return_value=torch.float16):
            got = self._merged(lyco, "a.weight")
            plain = self._merged(stock, "a.weight")
        expected = oracle.lycoris_merge(torch, w0.to(torch.float16), delta, scale, True).to(torch.bfloat16)
        self.assertEqual(got.dtype, torch.bfloat16)
        self.assertTrue(torch.equal(got, expected))
        self.assertGreater((plain.float() - expected.float()).abs().max().item(), 1e-3)

    def test_lowvram_patch_keeps_the_mode_and_writes_in_the_layers_dtype(self):
        model = self._model()
        w0 = model.a.weight.detach().clone()
        sd, delta, scale = self._file("lora", w0, True, seed=4)
        stock = self._patcher(model, (sd, {"x": "a.weight"}, 0.8))
        lyco = self._mode(stock, LYCORIS)
        expected = oracle.lycoris_merge(torch, w0.to(torch.bfloat16), delta, scale, True, 0.8).to(torch.bfloat16)
        for prepare in (False, True):
            with self.subTest(prepare=prepare):
                patch = self.patcher_module.LowVramPatch("a.weight", lyco.patches)
                self.assertGreater(patch.memory_required(), 0)
                if prepare:   # comfy/lora.py prefetch_prepared_value: type(v)(v.loaded_keys, weights)
                    prepared = patch.prepare(None, None, copy=False)
                    self.assertIsNot(prepared[0][1], lyco.patches["a.weight"][0][1])
                    self.assertIs(type(prepared[0][1]), type(lyco.patches["a.weight"][0][1]))
                weight = w0.to(torch.bfloat16)
                out = patch(weight)
                self.assertIs(out, weight)
                self.assertTrue(torch.equal(out, expected))
        stock_out = self.patcher_module.LowVramPatch("a.weight", stock.patches)(w0.to(torch.bfloat16))
        self.assertFalse(torch.equal(stock_out, expected))

    def test_an_offset_patch_changes_only_its_rows(self):
        model = self._model(seed=5)
        fused = model.q.weight.detach().clone()
        rows = fused[OUT:2 * OUT]
        sd, delta, scale = self._file("loha", rows, True, seed=5)
        stock = self._patcher(model, (sd, {"x": ("q.weight", (0, OUT, OUT))}, 1.0))
        self.assertEqual(stock.patches["q.weight"][0][3], (0, OUT, OUT))
        lyco = self._mode(stock, LYCORIS)
        got = self._merged(lyco, "q.weight")
        self._assert_lycoris(got[OUT:2 * OUT], oracle.lycoris_merge(torch, rows, delta, scale, True))
        self.assertTrue(torch.equal(got[:OUT], fused[:OUT]))
        self.assertTrue(torch.equal(got[2 * OUT:], fused[2 * OUT:]))

    def test_the_clone_is_isolated_and_comfy_reloads_by_patches_uuid(self):
        model = self._model(seed=6)
        w0 = model.a.weight.detach().clone()
        sd, delta, scale = self._file("lokr", w0, True, seed=6)
        stock = self._patcher(model, (sd, {"x": "a.weight"}, 1.0))
        sibling = stock.clone()
        lists = {key: items for key, items in stock.patches.items()}
        tuples = {key: list(items) for key, items in stock.patches.items()}
        adapter = stock.patches["a.weight"][0][1]
        uid = stock.patches_uuid
        lyco = self._mode(stock, LYCORIS)
        self.assertEqual(stock.patches_uuid, uid)
        self.assertNotEqual(lyco.patches_uuid, uid)
        self.assertEqual(sibling.patches_uuid, uid)
        for key, items in stock.patches.items():
            self.assertIs(items, lists[key])
            self.assertEqual(items, tuples[key])
        self.assertIs(stock.patches["a.weight"][0][1], adapter)
        self.assertIs(type(sibling.patches["a.weight"][0][1]), type(adapter))
        self.assertIs(lyco.patches["a.weight"][0][1].weights, adapter.weights)
        expected = oracle.lycoris_merge(torch, w0, delta, scale, True)
        self.mm.load_models_gpu([stock])
        plain = model.a.weight.detach().clone()
        self.assertGreater((plain - expected).abs().max().item(), 1e-4)
        self.mm.load_models_gpu([lyco])
        self._assert_lycoris(model.a.weight.detach(), expected)
        self.mm.load_models_gpu([sibling])
        self.assertTrue(torch.equal(model.a.weight, plain))
        self.mm.unload_all_models()
        self.assertTrue(torch.equal(model.a.weight, w0))

    def test_stacked_loras_merge_in_order_and_a_later_lora_stays_stock(self):
        model = self._model(seed=7)
        w0 = model.a.weight.detach().clone()
        first, d1, m1 = self._file("lokr", w0, True, seed=71, prefix="x")
        w1 = oracle.lycoris_merge(torch, w0, d1, m1, True, 0.8)
        second, d2, m2 = self._file("lora", w1, False, seed=72, prefix="y")
        plain, d3, _ = self._file("lora", w0, True, seed=73, dora=False, prefix="z")
        stock = self._patcher(model, (first, {"x": "a.weight"}, 0.8), (second, {"y": "a.weight"}, 0.6),
                              (plain, {"z": "a.weight"}, 0.7))
        lyco = self._mode(stock, LYCORIS)
        self.assertIs(lyco.patches["a.weight"][2], stock.patches["a.weight"][2])   # the plain LoRA's tuple
        later_file, d4, m4 = self._file("lora", w0, True, seed=74, prefix="w")
        later = lyco.clone()
        later.add_patches(self.lora.load_lora(later_file, {"w": "a.weight"}), 1.0)
        self.assertIs(type(later.patches["a.weight"][3][1]), self.wa.LoRAAdapter)
        expected = oracle.lycoris_merge(torch, w1, d2, m2, False, 0.6) + 0.7 * d3
        self._assert_lycoris(self._merged(lyco, "a.weight"), expected)
        with_later = oracle.comfy_formula_fp32(torch, expected, d4, m4, True)   # stock, in fp32 on CPU
        torch.testing.assert_close(self._merged(later, "a.weight"), with_later, rtol=1e-5, atol=1e-7)

    def test_an_oom_inside_comfys_adapter_empties_the_cache_and_merges_the_stock_way(self):
        """Review F2: ComfyUI's LoRAAdapter builds the full ``dW`` (``torch.mm``) inside its own try/except, which logs
        the OOM and returns without a delta; the node empties the cache and merges that patch the stock way."""
        model = self._model(seed=13)
        w0 = model.a.weight.detach().clone()
        sd, _, _ = self._file("lora", w0, True, seed=13)
        stock = self._patcher(model, (sd, {"x": "a.weight"}, 1.0))
        lyco = self._mode(stock, LYCORIS)
        plain = self._merged(stock, "a.weight")
        real_mm = torch.mm
        failures = iter([torch.OutOfMemoryError("CUDA out of memory. Tried to allocate 600.00 MiB")])

        def mm_once_oom(*args, **kwargs):
            failure = next(failures, None)
            if failure is not None:
                raise failure
            return real_mm(*args, **kwargs)

        with mock.patch.object(torch, "mm", side_effect=mm_once_oom), \
                mock.patch.object(self.mm, "soft_empty_cache") as empty, \
                self.assertLogs(level="ERROR") as logs:
            got = self._merged(lyco, "a.weight")
        empty.assert_called_once_with()
        text = "\n".join(logs.output)
        self.assertIn("out of memory", text)                     # ComfyUI's own log line
        self.assertIn("possible OOM", text)                      # the node's
        self.assertTrue(torch.equal(got, plain))                 # that patch merged the stock way
        self.assertEqual(self.dim.RUN.fallbacks[self.dim.MODE_LYCORIS], 1)

    def test_the_run_report_wrapper_runs_in_comfys_wrapper_executor(self):
        """Review F2/F5: the clone's OUTER_SAMPLE wrapper reaches a run's model options the way CFGGuider.sample
        merges them (``prepare_model_patcher``) and, run by ComfyUI's WrapperExecutor around a model load, starts a
        fresh account and reports the merges of that run; the input MODEL has no such wrapper."""
        import comfy.patcher_extension
        import comfy.sampler_helpers

        model = self._model(seed=14)
        w0 = model.a.weight.detach().clone()
        sd, delta, scale = self._file("lokr", w0, True, seed=14)
        stock = self._patcher(model, (sd, {"x": "a.weight"}, 1.0))
        lyco = self._mode(stock, LYCORIS)
        outer = comfy.patcher_extension.WrappersMP.OUTER_SAMPLE
        self.assertEqual(stock.get_all_wrappers(outer), [])
        self.assertEqual(len(lyco.get_wrappers(outer, self.dim.WRAPPER_KEY)), 1)
        self.dim.RUN.merged(self.dim.MODE_LYCORIS, "stale", object())   # an earlier run's: not reported
        test = self

        class Guider:
            def outer_sample(self, *args):
                test.mm.load_models_gpu([lyco])
                return model.a.weight.detach().clone()

        options = self.patcher_module.create_model_options_clone(lyco.model_options)
        options.setdefault("transformer_options", {})
        comfy.sampler_helpers.prepare_model_patcher(lyco, {}, options)
        guider = Guider()
        executor = comfy.patcher_extension.WrapperExecutor.new_class_executor(
            guider.outer_sample, guider,
            comfy.patcher_extension.get_all_wrappers(outer, options, is_model_options=True))
        with self.assertLogs("ai_studio_forge_parity", level="INFO") as logs:
            loaded = executor.execute()
        self._assert_lycoris(loaded, oracle.lycoris_merge(torch, w0, delta, scale, True))
        self.assertEqual([record.getMessage() for record in logs.records],
                         ["[DoRA mode] LyCORIS fp32 (as trained): 1 DoRA patches merged this run"])

    def test_strength_model_is_applied_by_comfy_before_the_adapter(self):
        model = self._model(seed=8)
        w0 = model.a.weight.detach().clone()
        sd, delta, scale = self._file("lora", w0, True, seed=8)
        patcher = self.patcher_module.ModelPatcher(model, self.cpu, self.cpu)
        patcher.add_patches(self.lora.load_lora(sd, {"x": "a.weight"}), 0.5, 0.9)
        got = self._merged(self._mode(patcher, LYCORIS), "a.weight")
        self._assert_lycoris(got, oracle.lycoris_merge(torch, w0 * 0.9, delta, scale, True, 0.5))

    def test_forge_fp32_is_the_stock_formula_and_dora_off_is_the_file_without_dora_scale(self):
        model = self._model(seed=9)
        w0 = model.a.weight.detach().clone()
        for strength in (1.0, 0.5):
            with self.subTest(strength=strength):
                sd, delta, _ = self._file("loha", w0, True, seed=9)
                stock = self._patcher(model, (sd, {"x": "a.weight"}, strength))
                self.assertTrue(torch.equal(self._merged(self._mode(stock, FORGE_FP32), "a.weight"),
                                            self._merged(stock, "a.weight")))   # fp32 on CPU: ComfyUI's own bits
                without = {name: value for name, value in sd.items() if not name.endswith(".dora_scale")}
                bare = self._patcher(model, (without, {"x": "a.weight"}, strength))
                self.assertTrue(torch.equal(self._merged(self._mode(stock, DORA_OFF), "a.weight"),
                                            self._merged(bare, "a.weight")))
                torch.testing.assert_close(self._merged(bare, "a.weight"), w0 + strength * delta,
                                           rtol=1e-6, atol=1e-8)

    def test_stock_mode_undoes_the_node_and_oft_stays_stock(self):
        model = self._model(seed=10)
        w0 = model.a.weight.detach().clone()
        sd, _, _ = self._file("lora", w0, True, seed=10)
        oft = {"o.oft_blocks": torch.randn(4, 6, 6) * 0.01, "o.dora_scale": torch.ones(OUT, 1)}
        stock = self._patcher(model, (sd, {"x": "a.weight"}, 1.0), (oft, {"o": "a.weight"}, 1.0))
        lyco = self._mode(stock, LYCORIS)
        self.assertIs(lyco.patches["a.weight"][1], stock.patches["a.weight"][1])
        back = self._mode(lyco, STOCK)
        self.assertIs(type(back.patches["a.weight"][0][1]), type(stock.patches["a.weight"][0][1]))
        self.assertTrue(torch.equal(self._merged(back, "a.weight"), self._merged(stock, "a.weight")))
        only_oft = self._patcher(model, (oft, {"o": "a.weight"}, 1.0))
        with self.assertLogs("ai_studio_forge_parity", level="WARNING") as logs:
            self.assertIs(self._mode(only_oft, LYCORIS), only_oft)
        self.assertIn("OFT (stock in sam-extra too) 1", logs.output[0])

    def test_no_dora_patches_and_no_comfy_function_replaced(self):
        functions = {name: getattr(module, "weight_decompose") for name, module in self.adapter_modules.items()}
        methods = {cls: cls.__dict__.get("calculate_weight") for cls in self.wa.adapters}
        dispatcher = self.lora.calculate_weight
        model = self._model(seed=11)
        plain, _, _ = self._file("lora", model.a.weight.detach(), True, seed=11, dora=False)
        patcher = self._patcher(model, (plain, {"x": "a.weight"}, 1.0))
        with self.assertLogs("ai_studio_forge_parity", level="WARNING") as logs:
            self.assertIs(self._mode(patcher, LYCORIS), patcher)
        self.assertEqual(len(logs.records), 1)
        sd, _, _ = self._file("lokr", model.a.weight.detach(), True, seed=11)
        self._merged(self._mode(self._patcher(model, (sd, {"x": "a.weight"}, 1.0)), LYCORIS), "a.weight")
        self.assertEqual({name: getattr(module, "weight_decompose") for name, module in self.adapter_modules.items()},
                         functions)
        self.assertEqual({cls: cls.__dict__.get("calculate_weight") for cls in self.wa.adapters}, methods)
        self.assertIs(self.lora.calculate_weight, dispatcher)
        self.assertNotIn(self.dim.__name__, {cls.__module__ for cls in self.wa.adapters})


if __name__ == "__main__":
    unittest.main()
