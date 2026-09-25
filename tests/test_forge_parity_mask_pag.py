"""Tensor regressions at the public Comfy node/provider boundaries.

PAG 는 원본 노드를 그대로 부르게 바뀌어 tests/test_forge_parity_pag_origin.py 가 검증한다.
"""
from __future__ import annotations

import sys
from types import SimpleNamespace
import unittest
from unittest import mock

from comfy_custom_nodes.ai_studio_forge_parity import generation
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


if __name__ == "__main__":
    unittest.main()
