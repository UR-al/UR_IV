"""Tensor regressions at the public Comfy node/provider boundaries."""
from __future__ import annotations

from contextlib import ExitStack
import copy
import json
import os
import sys
from types import SimpleNamespace
import unittest
from unittest import mock

from comfy_custom_nodes.ai_studio_forge_parity import generation, guidance
from tests._optional_deps import bind_torch, requires_torch

# torch 는 지연 import 한다(tests/_optional_deps) — 최상위 try-import 도 discovery 마다(= --quick 훅)
# torch 를 올렸다. 모든 클래스가 @requires_torch 이고 setUpClass 가 이 전역을 채운다.
torch = None


@requires_torch
class TestInpaintMaskFit(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        bind_torch(globals())

    def test_image_and_mask_use_the_same_crop_and_padding(self):
        class EncodeProvider:
            def encode(self, vae, pixels, mask, grow):
                # Expose the provider's observable encode/noise-mask result.
                return ({"samples": pixels.movedim(-1, 1), "noise_mask": mask},)

        image = torch.zeros(1, 8, 16, 3)
        image[:, :, 12:] = 1.0
        mask = image[..., 0]
        nodes = SimpleNamespace(NODE_CLASS_MAPPINGS={"VAEEncodeForInpaint": EncodeProvider})
        for fit in ("crop", "contain", "stretch"):
            with self.subTest(fit=fit), mock.patch.dict(sys.modules, {"nodes": nodes}):
                (latent,) = generation.ForgeNeoLatentInput().make(
                    object(), mode="inpaint", width=8, height=8, fit=fit,
                    mask_blur=0, grow_mask_by=0,
                    inpaint_image=image, inpaint_mask=mask,
                )
                visible = latent["samples"][:, 0] > 0.5
                selected = latent["noise_mask"] > 0.5
                self.assertTrue(torch.equal(visible, selected))


class _Patcher:
    """Comfy MODEL boundary: object patches activate only during execution."""
    def __init__(self, attention):
        self.model = SimpleNamespace(diffusion_model=SimpleNamespace(
            blocks=[SimpleNamespace(self_attn=attention)]
        ))
        self.model_options = {}
        self.object_patches = {}

    def clone(self):
        clone = copy.copy(self)
        clone.model_options = dict(self.model_options)
        clone.object_patches = dict(self.object_patches)
        return clone

    def get_model_object(self, path):
        if path in self.object_patches:
            return self.object_patches[path]
        value = self.model
        for part in path.split("."):
            value = value[int(part)] if part.isdigit() else getattr(value, part)
        return value

    def add_object_patch(self, path, value):
        self.object_patches[path] = value

    def set_model_sampler_post_cfg_function(self, callback, **_options):
        callbacks = list(self.model_options.get("sampler_post_cfg_function", []))
        self.model_options["sampler_post_cfg_function"] = callbacks + [callback]

    def predict(self, image, options):
        with ExitStack() as stack:
            for path, replacement in self.object_patches.items():
                parent, attr = path.rsplit(".", 1)
                stack.enter_context(mock.patch.object(self.get_model_object(parent), attr, replacement))
            attn = self.model.diffusion_model.blocks[0].self_attn
            transformer = {**options.get("transformer_options", {}), "block_index": 0}
            return attn(image, transformer_options=transformer)


@requires_torch
class TestPAGAttention(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        bind_torch(globals())

    def attention(self):
        # Small provider contract with the same q_proj -> RMSNorm -> attn_op
        # ordering as Cosmos. Real Cosmos is exercised by the subclass below.
        class Attention(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.q_proj = torch.nn.Linear(4, 4, bias=False)
                self.k_proj = torch.nn.Linear(4, 4, bias=False)
                self.v_proj = torch.nn.Linear(4, 4, bias=False)
                self.q_norm = torch.nn.RMSNorm(4, eps=1e-6)
                self.k_norm = torch.nn.RMSNorm(4, eps=1e-6)
                self.output_proj = torch.nn.Linear(4, 4, bias=False)
                self.attn_op = self.attention_op

            @staticmethod
            def attention_op(q, k, v, transformer_options=None):
                q, k, v = [item.flatten(1, -3).movedim(-2, 1) for item in (q, k, v)]
                out = torch.nn.functional.scaled_dot_product_attention(q, k, v)
                return out.movedim(1, -2).flatten(-2)

            def compute_qkv(self, x, transformer_options):
                q, k, v = x, x, x
                for patch in transformer_options.get("patches", {}).get("attn1_patch", []):
                    changed = patch(q, k, v, extra_options=transformer_options)
                    q, k, v = changed.get("q", q), changed.get("k", k), changed.get("v", v)
                return (self.q_norm(self.q_proj(q)).unsqueeze(-2),
                        self.k_norm(self.k_proj(k)).unsqueeze(-2), self.v_proj(v).unsqueeze(-2))

            def forward(self, x, transformer_options):
                q, k, v = self.compute_qkv(x, transformer_options)
                return self.output_proj(self.attn_op(q, k, v, transformer_options=transformer_options))

        return Attention()

    def test_pag_strength_blends_attention_toward_value_only_on_weak_pass(self):
        torch.manual_seed(31)
        device = os.environ.get("AISTUDIO_COMFY_TEST_DEVICE", "cpu")
        attention = self.attention().to(device)
        source = torch.tensor([[[[[1., 0., -1., 2.], [0., 2., 1., -1.], [2., -2., 0., 1.]]]]], device=device)
        model = _Patcher(attention)
        original = model.predict(source, {})
        value_target = attention.output_proj(attention.v_proj(source).flatten(1, -2))
        self.assertGreater((original - value_target).abs().max().item(), 0.1)
        for node_name in ("standalone", "suite"):
            for strength in (0.25, 0.75, 1.0):
                with self.subTest(node=node_name, strength=strength):
                    if node_name == "standalone":
                        (patched,) = guidance.ForgeNeoAnimaSafePAG().patch(
                            model, True, scale=1.0, block_indices="0",
                            perturbation_strength=strength, rescale=0.0,
                        )
                    else:
                        (patched,) = guidance.ForgeNeoAnimaGuidanceSuite().patch(
                            model, None, None, None, True, json.dumps({
                                "guid_enabled": True, "guid_attn_method": "PAG",
                                "guid_block_indices": "0", "guid_scale": 1.0,
                                "guid_official_strength": strength, "guid_rescale": 0.0,
                            }),
                        )
                    torch.testing.assert_close(patched.predict(source, {}), original)
                    samplers = SimpleNamespace(calc_cond_batch=lambda _m, _c, x, _s, options: (patched.predict(x, options),))
                    with mock.patch.dict(sys.modules, {"comfy": SimpleNamespace(samplers=samplers), "comfy.samplers": samplers}):
                        callback = patched.model_options["sampler_post_cfg_function"][-1]
                        actual = callback({
                            "model": patched.model, "model_options": patched.model_options,
                            "cond": [], "input": source, "sigma": torch.tensor([0.9]),
                            "denoised": original, "cond_denoised": original,
                        })
                    expected = original + strength * (original - value_target)
                    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)
                    # Applying weak guidance must not leave a process/global hook.
                    torch.testing.assert_close(model.predict(source, {}), original)


@unittest.skipUnless(os.environ.get("AISTUDIO_COMFY_TEST_ROOT"), "Set AISTUDIO_COMFY_TEST_ROOT for real Cosmos integration")
class TestPAGRealCosmos(TestPAGAttention):
    def attention(self):
        root = os.environ["AISTUDIO_COMFY_TEST_ROOT"]
        if root not in sys.path:
            sys.path.insert(0, root)
        from comfy.ldm.cosmos.predict2 import Attention
        return Attention(4, n_heads=1, head_dim=4, operations=torch.nn, dtype=torch.float32)


if __name__ == "__main__":
    unittest.main()
