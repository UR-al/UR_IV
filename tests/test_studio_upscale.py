"""Forge's upscaling arithmetic and the model loop, without ComfyUI (studio_upscale.py)."""
import unittest
from types import SimpleNamespace
from unittest import mock

import numpy as np

from comfy_custom_nodes.ai_studio_forge_parity import studio_upscale as su
from tests._optional_deps import load_torch, requires_torch


class SnapTests(unittest.TestCase):
    def test_snap_rounds_half_up_like_forge_sround(self):
        self.assertEqual(su.snap(1248, 64), 1280)
        self.assertEqual(su.snap(1216, 64), 1216)
        self.assertEqual(su.snap(1824, 64), 1856)
        self.assertEqual(su.snap(32, 64), 64)  # floor(0.5 + 0.5)
        self.assertEqual(su.snap(31.9, 64), 0)


class HiresSizeTests(unittest.TestCase):
    def test_scale_snaps_to_the_step(self):
        self.assertEqual(su.hires_size(832, 1216, 1.5, 0, 0, 64), (1280, 1856))
        self.assertEqual(su.hires_size(832, 1216, 1.5, 0, 0, 8), (1248, 1824))

    def test_width_alone_keeps_the_aspect(self):
        self.assertEqual(su.hires_size(832, 1216, 1.5, 1024, 0, 8), (1024, 1496))

    def test_height_alone_keeps_the_aspect(self):
        self.assertEqual(su.hires_size(832, 1216, 1.5, 0, 1216, 8), (832, 1216))

    def test_both_axes_are_taken_as_they_are(self):
        self.assertEqual(su.hires_size(832, 1216, 1.5, 1024, 1024, 64), (1024, 1024))

    def test_a_size_never_reaches_zero(self):
        self.assertEqual(su.hires_size(8, 8, 1.0, 0, 0, 64), (64, 64))

    def test_the_result_ends_on_the_latent_grid(self):
        self.assertEqual(su.hires_size(832, 1216, 1.5, 1000, 1000, 8, 16), (992, 992))
        self.assertEqual(su.hires_size(832, 1216, 1.5, 8, 8, 8, 16), (16, 16))


class ExtrasPlanTests(unittest.TestCase):
    def test_scale_by(self):
        plan = su.extras_plan(1000, 500, "Scale by", 2.0, 0, 1024, 1024, True)
        self.assertEqual(plan, su.ExtrasPlan(scale=2.0, crop_to=None))

    def test_scale_to_with_crop_covers_the_box_and_crops(self):
        plan = su.extras_plan(1000, 500, "Scale to", 2.0, 0, 1024, 1024, True)
        self.assertAlmostEqual(plan.scale, 2.048)
        self.assertEqual(plan.crop_to, (1024, 1024))

    def test_scale_to_without_crop_keeps_the_covering_size(self):
        plan = su.extras_plan(1000, 500, "Scale to", 2.0, 0, 1024, 1024, False)
        self.assertAlmostEqual(plan.scale, 2.048)
        self.assertIsNone(plan.crop_to)

    def test_max_side_limits_the_longest_side_and_turns_crop_off(self):
        plan = su.extras_plan(1000, 500, "Scale by", 4.0, 2000, 1024, 1024, True)
        self.assertAlmostEqual(plan.scale, 2.0)
        self.assertIsNone(plan.crop_to)

    def test_max_side_above_the_result_changes_nothing(self):
        self.assertEqual(su.extras_plan(1000, 500, "Scale by", 2.0, 4000, 1024, 1024, True).scale, 2.0)

    def test_limit_side_like_forge(self):
        self.assertEqual(su.limit_side(832, 1216, 1000), (684, 1000))
        self.assertEqual(su.limit_side(1216, 832, 1000), (1000, 684))
        self.assertEqual(su.limit_side(500, 400, 1000), (500, 400))


class FakeImage:
    """The ``shape`` of a ComfyUI IMAGE: [batch, height, width, channels]."""

    def __init__(self, width, height):
        self.shape = (1, height, width, 3)


class UpscaleLoopTests(unittest.TestCase):
    def run_upscale(self, image, name, scale, factor):
        passes = []

        def one_pass(model, source, tile, overlap):
            passes.append((source.shape[2], tile, overlap))
            return FakeImage(source.shape[2] * factor, source.shape[1] * factor)

        def resized(source, width, height, method):
            return SimpleNamespace(shape=(1, height, width, 3), method=method, source=source)

        with mock.patch.object(su, "load_model", return_value="model"), \
                mock.patch.object(su, "model_pass", side_effect=one_pass), \
                mock.patch.object(su, "resize", side_effect=resized):
            return su.upscale(image, name, scale, 256, 16), passes

    def test_the_model_runs_again_until_the_target_is_reached(self):
        out, passes = self.run_upscale(FakeImage(100, 100), "2x.pth", 4.0, 2)
        self.assertEqual(passes, [(100, 256, 16), (200, 256, 16)])
        self.assertEqual(out.shape[2], 400)

    def test_the_passes_stop_at_four(self):
        _, passes = self.run_upscale(FakeImage(16, 16), "1x.pth", 2.0, 1)
        self.assertEqual(len(passes), 1)  # a model that does not enlarge is not run again
        _, passes = self.run_upscale(FakeImage(8, 8), "2x.pth", 64.0, 2)
        self.assertEqual(len(passes), 4)

    def test_the_rest_is_lanczos_to_a_multiple_of_8(self):
        out, passes = self.run_upscale(FakeImage(101, 101), "4x.pth", 2.0, 4)
        self.assertEqual(len(passes), 1)
        self.assertEqual((out.shape[2], out.method), (200, "lanczos"))

    def test_lanczos_and_nearest_never_load_a_model(self):
        out, passes = self.run_upscale(FakeImage(100, 100), "Nearest", 2.0, 2)
        self.assertEqual(passes, [])
        self.assertEqual((out.shape[2], out.method), (200, "nearest-exact"))
        out, passes = self.run_upscale(FakeImage(100, 100), "Lanczos", 2.0, 2)
        self.assertEqual(passes, [])
        self.assertEqual((out.shape[2], out.method), (200, "lanczos"))


class UpscaleToTests(unittest.TestCase):
    def test_no_model_when_the_picture_is_big_enough(self):
        image = FakeImage(1000, 1000)
        with mock.patch.object(su, "upscale", side_effect=AssertionError("must not run")), \
                mock.patch.object(su, "resize", side_effect=lambda source, width, height, method: (width, height, method)):
            self.assertEqual(su.upscale_to(image, "4x.pth", 800, 800, 256, 16), (800, 800, "lanczos"))

    def test_an_exact_size_comes_back_as_it_is(self):
        image = FakeImage(1000, 1000)
        with mock.patch.object(su, "upscale", side_effect=AssertionError("must not run")), \
                mock.patch.object(su, "resize", side_effect=AssertionError("must not run")):
            self.assertIs(su.upscale_to(image, "4x.pth", 1000, 1000, 256, 16), image)

    def test_the_scale_is_the_larger_ratio(self):
        calls = []

        def upscale(image, name, scale, tile, overlap):
            calls.append(scale)
            return FakeImage(1600, 800)

        with mock.patch.object(su, "upscale", side_effect=upscale), \
                mock.patch.object(su, "resize", side_effect=lambda source, width, height, method: (width, height)):
            self.assertEqual(su.upscale_to(FakeImage(1000, 500), "4x.pth", 1600, 900, 256, 16), (1600, 900))
        self.assertEqual(calls, [1.8])


class ModelFilesTests(unittest.TestCase):
    def test_the_placeholder_of_an_empty_folder_is_not_a_model(self):
        with mock.patch.object(su, "filename_choices", return_value=["None"]):
            self.assertEqual(su.model_files(), [])
        with mock.patch.object(su, "filename_choices", return_value=["4x.pth"]):
            self.assertEqual(su.model_files(), ["4x.pth"])


class OpencvTests(unittest.TestCase):
    def test_a_missing_opencv_says_what_to_install(self):
        with mock.patch.dict("sys.modules", {"cv2": None}), self.assertRaisesRegex(RuntimeError, "opencv-python"):
            su.require_opencv()


@requires_torch
class CenterPasteTests(unittest.TestCase):
    def test_a_larger_picture_is_cropped_around_its_centre(self):
        torch = load_torch()
        image = torch.arange(36.0).reshape(1, 6, 6, 1).repeat(1, 1, 1, 3)
        out = su.paste_centered(image, 2, 2)
        self.assertEqual(tuple(out.shape), (1, 2, 2, 3))
        self.assertEqual(out[0, 0, 0, 0].item(), image[0, 2, 2, 0].item())

    def test_a_smaller_side_gets_black_edges_like_forge(self):
        torch = load_torch()
        out = su.paste_centered(torch.ones((1, 4, 4, 3)), 6, 4)
        self.assertEqual(tuple(out.shape), (1, 4, 6, 3))
        self.assertEqual(out[0, 0, 0, 0].item(), 0.0)
        self.assertEqual(out[0, 0, 1, 0].item(), 1.0)


class BlendTests(unittest.TestCase):
    def test_blend_is_forge_image_blend(self):
        first, second = np.array([0.0, 1.0]), np.array([1.0, 0.0])
        np.testing.assert_allclose(su.blend(first, second, 0.25), [0.25, 0.75])


class ColorMatchTests(unittest.TestCase):
    def test_a_tinted_picture_comes_back_to_the_source_colours(self):
        source = np.zeros((32, 32, 3), dtype=np.uint8)
        source[..., 0] = np.tile(np.arange(32, dtype=np.uint8) * 4, (32, 1))
        source[..., 2] = 40
        tinted = np.clip(source.astype(np.int16) + np.array([0, 60, 0]), 0, 255).astype(np.uint8)
        matched = su.match_colors_uint8(tinted, source)
        self.assertLess(np.abs(matched.astype(int) - source.astype(int)).mean(),
                        np.abs(tinted.astype(int) - source.astype(int)).mean() / 3)

    def test_the_histogram_match_is_what_forge_calls(self):
        exposure = pytest_skimage(self)
        generator = np.random.default_rng(7)
        image = generator.integers(0, 256, (48, 64, 3), dtype=np.uint8)
        reference = (generator.random((32, 32, 3)) ** 2 * 200 + 20).astype(np.uint8)
        expected = exposure.match_histograms(image, reference, channel_axis=2)
        matched = np.stack([su.match_channel(image[..., index], reference[..., index]) for index in range(3)], axis=2)
        np.testing.assert_array_equal(matched, expected)


def pytest_skimage(case):
    try:
        from skimage import exposure
    except ImportError:
        case.skipTest("scikit-image is not installed")
    return exposure


if __name__ == "__main__":
    unittest.main()
