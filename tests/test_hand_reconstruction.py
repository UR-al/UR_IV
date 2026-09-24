import base64
import hashlib
import io
import json
import unittest
from unittest.mock import patch

from PIL import Image, ImageDraw, PngImagePlugin

from core.hand_reconstruction import compose_hand_candidate, prepare_hand_repair


def encoded(image, *, fmt="PNG", **kwargs):
    stream = io.BytesIO()
    image.save(stream, format=fmt, **kwargs)
    return stream.getvalue()


def data_url(image, *, fmt="PNG", **kwargs):
    mime = {"PNG": "png", "JPEG": "jpeg", "WEBP": "webp", "BMP": "png"}[fmt]
    return "data:image/" + mime + ";base64," + base64.b64encode(encoded(image, fmt=fmt, **kwargs)).decode("ascii")


def pixels(data):
    with Image.open(io.BytesIO(data)) as source:
        return source.copy()


class HandReconstructionTests(unittest.TestCase):
    def setUp(self):
        self.source = Image.new("RGBA", (160, 112), (83, 124, 155, 173))
        ImageDraw.Draw(self.source).rectangle((59, 24, 94, 81), fill=(229, 171, 122, 224))
        self.mask = Image.new("L", self.source.size, 0)
        ImageDraw.Draw(self.mask).rectangle((55, 20, 99, 87), fill=255)

    def prepare(self, source=None, mask=None, **settings):
        return prepare_hand_repair({"image": data_url(source if source is not None else self.source),
                                    "mask": data_url(mask if mask is not None else self.mask),
                                    "settings": {"enabled": True, **settings}})

    def candidate(self, prepared, color=(26, 220, 72), **provenance):
        return compose_hand_candidate(prepared, encoded(Image.new("RGB", prepared.working_size, color)),
                                      provenance=provenance)

    def test_erasure_is_independent_of_sixth_finger_pixels_before_resampling(self):
        other = self.source.copy()
        ImageDraw.Draw(other).rectangle((70, 30, 84, 75), fill=(255, 0, 100, 30))
        first, second = self.prepare(), self.prepare(other)
        self.assertEqual(first.init_png, second.init_png)
        init, mask = pixels(first.init_png), pixels(first.mask_png)
        init_bytes = init.tobytes()
        for index, selected in enumerate(mask.tobytes()):
            if selected:
                self.assertEqual(init_bytes[index * 3:index * 3 + 3], bytes((127, 127, 127)))
        self.assertEqual(first.init_png, first.prepared_png)
        self.assertNotEqual(first.source_sha256, second.source_sha256)

    def test_composition_preserves_every_outside_rgba_pixel_and_original_alpha(self):
        for feather in (0, 4, 16):
            with self.subTest(feather=feather):
                prepared = self.prepare(feather=feather)
                before = prepared.source.tobytes()
                result = pixels(self.candidate(prepared))
                self.assertEqual(result.size, self.source.size)
                self.assertEqual(result.mode, "RGBA")
                original_bytes, result_bytes = self.source.tobytes(), result.tobytes()
                for index, selected in enumerate(self.mask.tobytes()):
                    original = original_bytes[index * 4:index * 4 + 4]
                    candidate = result_bytes[index * 4:index * 4 + 4]
                    self.assertEqual(original[3], candidate[3])
                    if not selected:
                        self.assertEqual(original, candidate)
                self.assertNotEqual(result.getpixel((75, 54))[:3], self.source.getpixel((75, 54))[:3])
                self.assertEqual(prepared.source.tobytes(), before)
                self.assertEqual(prepared.edit_mask.tobytes(), self.mask.tobytes())

    def test_final_feather_does_not_modify_unmasked_holes(self):
        mask = self.mask.copy()
        ImageDraw.Draw(mask).ellipse((64, 39, 80, 55), fill=0)
        prepared = self.prepare(mask=mask, feather=16)
        result = pixels(self.candidate(prepared))
        self.assertEqual(result.getpixel((71, 46)), self.source.getpixel((71, 46)))

    def test_inward_feather_starts_at_zero_and_reaches_full_interior_opacity(self):
        source = Image.new("RGBA", (96, 96), (0, 0, 0, 137))
        mask = Image.new("L", source.size, 0)
        ImageDraw.Draw(mask).rectangle((16, 16, 79, 79), fill=255)
        for feather in (1, 4, 16):
            with self.subTest(feather=feather):
                result = pixels(self.candidate(self.prepare(source, mask, feather=feather), color=(255, 255, 255)))
                self.assertEqual(result.getpixel((15, 48)), (0, 0, 0, 137))
                self.assertEqual(result.getpixel((16, 48)), (0, 0, 0, 137))
                samples = [result.getpixel((16 + depth, 48))[0] for depth in range(feather + 1)]
                self.assertEqual(samples, [round(255 * depth / feather) for depth in range(feather + 1)])
                self.assertEqual(result.getpixel((48, 48)), (255, 255, 255, 137))
                self.assertEqual(json.loads(result.info["ai_studio_hand_reconstruction"])["feather_mode"], "inward-distance-ramp")

    def test_zero_feather_keeps_hard_mask_boundary_editable(self):
        source = Image.new("RGB", self.source.size, (0, 0, 0))
        result = pixels(self.candidate(self.prepare(source, feather=0), color=(255, 255, 255)))
        self.assertEqual(result.getpixel((54, 50)), (0, 0, 0))
        self.assertEqual(result.getpixel((55, 50)), (255, 255, 255))

    def test_feather_protects_hole_boundaries_and_is_independent_of_crop_padding(self):
        source = Image.new("RGB", (96, 96), (0, 0, 0))
        mask = Image.new("L", source.size, 0)
        ImageDraw.Draw(mask).rectangle((16, 16, 79, 79), fill=255)
        ImageDraw.Draw(mask).rectangle((40, 40, 55, 55), fill=0)
        padded = pixels(self.candidate(self.prepare(source, mask, feather=4, padding=16), color=(255, 255, 255)))
        tight = pixels(self.candidate(self.prepare(source, mask, feather=4, padding=0), color=(255, 255, 255)))
        self.assertEqual(padded.tobytes(), tight.tobytes())
        self.assertEqual(tight.getpixel((40, 48)), (0, 0, 0))
        self.assertEqual(tight.getpixel((39, 48)), (0, 0, 0))
        self.assertEqual([tight.getpixel((39 - depth, 48))[0] for depth in range(5)], [0, 64, 128, 191, 255])

    def test_thin_mask_never_feathers_outwards_or_reintroduces_an_edge_step(self):
        source = Image.new("RGBA", (48, 48), (31, 42, 53, 107))
        mask = Image.new("L", source.size, 0)
        ImageDraw.Draw(mask).rectangle((15, 10, 17, 38), fill=255)
        result = pixels(self.candidate(self.prepare(source, mask, feather=16), color=(255, 255, 255)))
        for point in ((14, 24), (15, 24), (17, 24), (18, 24)):
            self.assertEqual(result.getpixel(point), source.getpixel(point))
        self.assertGreater(result.getpixel((16, 24))[0], source.getpixel((16, 24))[0])
        self.assertEqual(result.getchannel("A").tobytes(), source.getchannel("A").tobytes())

    def test_non_square_edge_crop_letterboxing_and_unpadding(self):
        source = Image.new("RGB", (801, 113), "blue")
        mask = Image.new("L", source.size, 0)
        ImageDraw.Draw(mask).rectangle((0, 0, 291, 60), fill=255)
        prepared = self.prepare(source, mask, padding=37, resolution=512, feather=0)
        self.assertEqual(prepared.bbox, (0, 0, 329, 98))
        self.assertEqual(prepared.working_size, (512, 192))
        content = prepared.content_box
        ratio = (content[2] - content[0]) / (content[3] - content[1])
        self.assertAlmostEqual(ratio, 329 / 98, delta=.03)
        generated = Image.new("RGB", prepared.working_size, "red")
        ImageDraw.Draw(generated).rectangle((content[0], content[1], content[2] - 1, content[3] - 1), fill="green")
        result = pixels(compose_hand_candidate(prepared, encoded(generated), provenance={}))
        self.assertEqual(result.getpixel((1, 1)), (0, 128, 0))
        self.assertEqual(result.getpixel((799, 111)), (0, 0, 255))

    def test_tall_crop_and_all_resolutions_stay_bounded(self):
        source = Image.new("RGB", (80, 701), "navy")
        mask = Image.new("L", source.size, 0)
        ImageDraw.Draw(mask).rectangle((10, 100, 65, 500), fill=255)
        for resolution in (512, 768, 1024):
            prepared = self.prepare(source, mask, resolution=resolution)
            self.assertEqual(max(prepared.working_size), resolution)
            self.assertTrue(all(side % 64 == 0 and side >= 64 for side in prepared.working_size))
            self.assertEqual(pixels(self.candidate(prepared)).size, source.size)

    def test_metadata_source_hash_and_trusted_provenance(self):
        metadata = PngImagePlugin.PngInfo()
        metadata.add_text("parameters", "original prompt\nNegative prompt: original negative")
        metadata.add_itxt("workflow", '{"original":true}')
        raw = encoded(self.source, pnginfo=metadata)
        prepared = prepare_hand_repair({"image": "data:image/png;base64," + base64.b64encode(raw).decode(),
                                        "mask": data_url(self.mask), "settings": {"enabled": True}})
        result = pixels(self.candidate(prepared, seed=17, source_sha256="untrusted"))
        self.assertEqual(result.info["parameters"], "original prompt\nNegative prompt: original negative")
        self.assertEqual(result.info["workflow"], '{"original":true}')
        record = json.loads(result.info["ai_studio_hand_reconstruction"])
        self.assertEqual(record["source_sha256"], hashlib.sha256(raw).hexdigest())
        self.assertEqual(record["run"]["seed"], 17)
        self.assertEqual(record["bbox"], list(prepared.bbox))
        self.assertEqual(record["settings"]["strength"], .9)
        self.assertNotIn("ai_studio_hand_reconstruction", prepared.source_metadata["text"])

    def test_exif_orientation_normalized_to_canvas_dimensions(self):
        source = Image.new("RGB", (40, 80), "orange")
        exif = Image.Exif()
        exif[274] = 6
        exif[315] = "original artist"
        mask = Image.new("L", (80, 40), 0)
        ImageDraw.Draw(mask).rectangle((20, 10, 40, 30), fill=255)
        prepared = prepare_hand_repair({"image": data_url(source, fmt="JPEG", exif=exif),
                                        "mask": data_url(mask), "settings": {"enabled": True}})
        self.assertEqual(prepared.source.size, (80, 40))
        result = pixels(self.candidate(prepared))
        self.assertEqual(result.getexif().get(315), "original artist")
        self.assertIn(result.getexif().get(274), (None, 1))

    def test_icc_is_kept_only_for_rgb_sources(self):
        """공용 정책(relight와 동일): 회색 ICC를 RGB PNG 결과에 붙이지 않는다."""
        gray = Image.new("L", self.source.size, 90)
        prepared = prepare_hand_repair({"image": data_url(gray, icc_profile=b"gray-profile"),
                                        "mask": data_url(self.mask), "settings": {"enabled": True}})
        self.assertIsNone(prepared.source_metadata["icc"])
        self.assertNotIn("icc_profile", pixels(self.candidate(prepared)).info)
        rgb = prepare_hand_repair({"image": data_url(self.source, icc_profile=b"rgb-profile"),
                                   "mask": data_url(self.mask), "settings": {"enabled": True}})
        self.assertEqual(pixels(self.candidate(rgb)).info["icc_profile"], b"rgb-profile")

    def test_transparent_mask_pixels_are_not_editable(self):
        mask = Image.new("RGBA", self.source.size, (255, 255, 255, 0))
        ImageDraw.Draw(mask).rectangle((60, 30, 80, 60), fill=(255, 255, 255, 255))
        prepared = self.prepare(mask=mask)
        self.assertEqual(prepared.edit_mask.getpixel((0, 0)), 0)
        self.assertEqual(prepared.edit_mask.getpixel((70, 40)), 255)

    def test_palette_transparency_is_preserved(self):
        source = Image.new("P", self.source.size, 0)
        source.putpalette([1, 2, 3, 50, 60, 70] + [0] * (768 - 6))
        source.info["transparency"] = 0
        source.paste(1, (20, 20, 80, 80))
        prepared = self.prepare(source)
        result = pixels(self.candidate(prepared))
        self.assertEqual(result.mode, "RGBA")
        self.assertEqual(result.getchannel("A").tobytes(), prepared.source.getchannel("A").tobytes())

    def test_empty_full_wrong_size_and_no_context_masks_fail(self):
        for mask in (Image.new("L", self.source.size, 0), Image.new("L", self.source.size, 255),
                     Image.new("L", (32, 48), 255)):
            with self.assertRaises(ValueError):
                self.prepare(mask=mask)
        with self.assertRaisesRegex(ValueError, "padding"):
            self.prepare(padding=0)

    def test_malformed_settings_reject_booleans_strings_nan_and_fractional_counts(self):
        for changes in ({"enabled": False}, {"enabled": 1}, {"strength": float("nan")},
                        {"strength": float("inf")}, {"strength": True}, {"strength": "0.9"},
                        {"strength": .64}, {"strength": 10 ** 1000}, {"candidates": 5}, {"candidates": 2.0},
                        {"padding": -1}, {"padding": True}, {"feather": 17}, {"resolution": 640}):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                self.prepare(**changes)
        with self.assertRaises(ValueError):
            prepare_hand_repair({"image": data_url(self.source), "mask": data_url(self.mask)})

    def test_non_uploads_unsupported_formats_corruption_and_mime_spoof_fail(self):
        for value in ("C:/some/image.png", "https://example.com/test.png", "file:///C:/image.png",
                      "data:image/svg+xml;base64,PHN2Zz4=", "data:image/png;base64,%%%%", "data:image/png;base64,YQ==",
                      data_url(self.source).replace("image/png", "image/jpeg"), data_url(self.source, fmt="BMP")):
            with self.subTest(value=value[:70]), self.assertRaises(ValueError):
                prepare_hand_repair({"image": value, "mask": data_url(self.mask), "settings": {"enabled": True}})

    def test_limits_fail_before_pixel_allocation_or_generation(self):
        with patch("core.hand_reconstruction.MAX_FILE_BYTES", 16):
            with self.assertRaises(ValueError): self.prepare()
        with patch("core.hand_reconstruction.MAX_REQUEST_CHARS", 16):
            with self.assertRaises(ValueError): self.prepare()
        with patch("core.hand_reconstruction.MAX_PIXELS", 100):
            with self.assertRaises(ValueError): self.prepare()
        with patch("core.hand_reconstruction.MAX_METADATA_BYTES", 1):
            info = PngImagePlugin.PngInfo()
            info.add_text("parameters", "too long")
            with self.assertRaises(ValueError):
                prepare_hand_repair({"image": data_url(self.source, pnginfo=info), "mask": data_url(self.mask),
                                     "settings": {"enabled": True}})

    def test_wrong_candidate_size_invalid_or_animated_raster_fail(self):
        prepared = self.prepare()
        for value in (b"not an image", encoded(Image.new("RGB", (64, 64))),
                      encoded(Image.new("RGB", prepared.working_size), fmt="BMP")):
            with self.assertRaises(ValueError): compose_hand_candidate(prepared, value, provenance={})
        stream = io.BytesIO()
        first = Image.new("RGB", prepared.working_size, "red")
        first.save(stream, format="PNG", save_all=True, append_images=[Image.new("RGB", prepared.working_size, "blue")])
        with self.assertRaises(ValueError): compose_hand_candidate(prepared, stream.getvalue(), provenance={})

    def test_provenance_rejects_non_json_and_nonfinite_values(self):
        prepared = self.prepare()
        candidate = encoded(Image.new("RGB", prepared.working_size))
        for value in (None, [], {"value": float("nan")}, {"value": object()}):
            with self.assertRaises(ValueError): compose_hand_candidate(prepared, candidate, provenance=value)

    def test_animated_source_is_not_silently_flattened(self):
        stream = io.BytesIO()
        self.source.save(stream, format="PNG", save_all=True, append_images=[Image.new("RGBA", self.source.size, "red")])
        with self.assertRaisesRegex(ValueError, "정지"):
            prepare_hand_repair({"image": "data:image/png;base64," + base64.b64encode(stream.getvalue()).decode(),
                                 "mask": data_url(self.mask), "settings": {"enabled": True}})

    def test_request_and_source_remain_unchanged_without_any_filesystem_access(self):
        settings = {"enabled": True, "candidates": 3}
        request = {"image": data_url(self.source), "mask": data_url(self.mask), "settings": settings}
        image_before, mask_before = self.source.tobytes(), self.mask.tobytes()
        with patch("builtins.open", side_effect=AssertionError("filesystem access is not allowed")):
            prepared = prepare_hand_repair(request)
            result = self.candidate(prepared)
        self.assertEqual(pixels(result).size, self.source.size)
        self.assertEqual(settings, {"enabled": True, "candidates": 3})
        self.assertEqual(self.source.tobytes(), image_before)
        self.assertEqual(self.mask.tobytes(), mask_before)


if __name__ == "__main__":
    unittest.main()
