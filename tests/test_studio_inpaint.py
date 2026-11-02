"""The studio's inpaint: AIStudioInpaintDetailer, and what a drawn mask adds to ForgeNeoSAM3Detailer."""
import json
import unittest
from unittest import mock

from comfy_custom_nodes.ai_studio_forge_parity import sam3_nodes
from tests._optional_deps import load_torch, requires_torch


class NodeInputsTests(unittest.TestCase):
    NEW = {"mask_blur", "batch_size", "tell_empty_mask"}

    def test_the_new_inputs_are_the_inpaint_nodes_alone(self):
        self.assertEqual(self.NEW - set(sam3_nodes.AIStudioInpaintDetailer.INPUT_TYPES()["optional"]), set())
        # Refine and tile take their widgets as named arguments: an inherited input would stop them.
        for node in (sam3_nodes.ForgeNeoSAM3Detailer, sam3_nodes.ForgeNeoSAM3Refine, sam3_nodes.ForgeNeoSAM3TileRepair):
            self.assertEqual(self.NEW & set(node.INPUT_TYPES().get("optional", {})), set(), node.__name__)


@requires_torch
class DetailerInpaintInputsTests(unittest.TestCase):
    def run_detail(self, image, mask, **inputs):
        torch = load_torch()
        seen = {}

        def encode(vae, pixels, work_mask, grow, **kwargs):
            seen["pixels"], seen["mask"] = pixels.clone(), work_mask.clone()
            return {"samples": torch.zeros((pixels.shape[0], 16, pixels.shape[1] // 8, pixels.shape[2] // 8)), "shape": tuple(pixels.shape)}

        with (
            mock.patch.object(sam3_nodes, "_vae_encode_for_inpaint", side_effect=encode),
            mock.patch.object(sam3_nodes, "_sample_latent", side_effect=lambda model, seed, steps, cfg, sampler, scheduler, pos, neg, latent, *_: latent),
            mock.patch.object(sam3_nodes, "_vae_decode", side_effect=lambda vae, latent: torch.ones(latent["shape"])),
            mock.patch.object(sam3_nodes, "_pad_for_vae", side_effect=lambda pixels, work_mask, vae: (pixels, work_mask, (0, 0, 0, 0))),
        ):
            out, applied, report = sam3_nodes.AIStudioInpaintDetailer().detail(
                image, mask, object(), object(), object(), "positive", "negative", mask_mode="Combined", grow_mask_by=0, **inputs)
        return out, applied, json.loads(report), seen

    def scene(self):
        torch = load_torch()
        image = torch.zeros((1, 256, 512, 3))
        mask = torch.zeros((1, 256, 512))
        mask[:, 100:140, 300:360] = 1.0
        return image, mask

    def test_the_crop_keeps_the_picture_size_and_pastes_inside_the_mask(self):
        image, mask = self.scene()
        out, _, report, seen = self.run_detail(image, mask, target_width=128, target_height=128, mask_padding=8)
        self.assertEqual(tuple(out.shape), (1, 256, 512, 3))
        self.assertEqual(tuple(seen["pixels"].shape[1:3]), (128, 128))
        self.assertEqual(report["passes"][0]["status"], "sampled")
        self.assertAlmostEqual(float(out[0, 120, 330, 0]), 1.0, places=4)
        self.assertEqual(float(out[0, 10, 10, 0]), 0.0)

    def test_whole_picture_draws_at_the_tabs_size_from_the_original_pixels(self):
        image, mask = self.scene()
        out, _, report, seen = self.run_detail(image, mask, only_masked=False, target_width=256, target_height=128)
        self.assertEqual(tuple(out.shape), (1, 128, 256, 3))  # Forge resizes the picture first
        self.assertEqual(tuple(seen["pixels"].shape[1:3]), (128, 256))
        self.assertEqual(report["passes"][0]["status"], "sampled")
        self.assertEqual(float(out[0, 5, 5, 0]), 0.0)  # outside the mask nothing changes
        self.assertAlmostEqual(float(out[0, 60, 165, 0]), 1.0, places=4)

    def test_whole_picture_of_the_same_size_is_not_resampled(self):
        torch = load_torch()
        image, mask = self.scene()
        image = torch.rand_like(image)
        out, *_ = self.run_detail(image, mask, only_masked=False, target_width=512, target_height=256)
        self.assertTrue(torch.equal(out[0, :90, :, :], image[0, :90, :, :]))

    def test_fill_mode_reaches_the_encoder(self):
        image, mask = self.scene()
        fills = []

        def note(vae, pixels, work_mask, grow, **kwargs):
            fills.append(kwargs.get("fill_mode"))
            return {"samples": load_torch().zeros((1, 16, 16, 32)), "shape": tuple(pixels.shape)}

        with (
            mock.patch.object(sam3_nodes, "_vae_encode_for_inpaint", side_effect=note),
            mock.patch.object(sam3_nodes, "_sample_latent", side_effect=lambda model, seed, steps, cfg, sampler, scheduler, pos, neg, latent, *_: latent),
            mock.patch.object(sam3_nodes, "_vae_decode", side_effect=lambda vae, latent: load_torch().ones(latent["shape"])),
            mock.patch.object(sam3_nodes, "_pad_for_vae", side_effect=lambda pixels, work_mask, vae: (pixels, work_mask, (0, 0, 0, 0))),
        ):
            sam3_nodes.AIStudioInpaintDetailer().detail(
                image, mask, object(), object(), object(), "positive", "negative", mask_mode="Combined", only_masked=False,
                fill_mode="original", target_width=512, target_height=256)
        self.assertEqual(fills, ["original"])

    def test_mask_blur_feathers_the_edge_with_forges_width(self):
        image, mask = self.scene()
        sharp = self.run_detail(image, mask, target_width=128, target_height=128)[1]
        soft = self.run_detail(image, mask, target_width=128, target_height=128, mask_blur=8)[1]
        self.assertEqual(float(sharp[0, 120, 296]), 0.0)
        self.assertGreater(float(soft[0, 120, 296]), 0.2)  # a sigma of 8: 4 px outside the edge is about 0.31

    def test_batch_size_makes_that_many_pictures_of_one_input(self):
        image, mask = self.scene()
        out, applied, _, _ = self.run_detail(image, mask, target_width=128, target_height=128, batch_size=3)
        self.assertEqual(out.shape[0], 3)
        self.assertEqual(applied.shape[0], 3)

    def test_an_empty_mask_leaves_the_picture_and_says_so(self):
        image, _ = self.scene()
        torch = load_torch()
        with mock.patch.object(sam3_nodes, "_tell_empty_mask") as tell:
            out, _, report, _ = self.run_detail(image, torch.zeros((1, 64, 64)), tell_empty_mask=True, batch_size=3)
        tell.assert_called_once_with()
        self.assertEqual(report["status"], "no_detection")
        self.assertEqual(out.shape[0], 1)

    def test_the_sam3_passes_stay_silent_about_an_empty_mask(self):
        image, _ = self.scene()
        torch = load_torch()
        with mock.patch.object(sam3_nodes, "_tell_empty_mask") as tell:
            self.run_detail(image, torch.zeros((1, 64, 64)))
        tell.assert_not_called()


if __name__ == "__main__":
    unittest.main()
