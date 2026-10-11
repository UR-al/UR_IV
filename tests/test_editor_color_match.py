"""참조 색 맞추기(core.editor_color_match + VueBridge 배선) 테스트.

Reinhard 옮기기 골든 값, 마스크 밖·알파 바이트 보존, 마주 보는 두 띠(ring) 통계, 투명 픽셀 제외,
참조 통계 캐시, 브리지 페이로드를 고정한다.
"""
from __future__ import annotations

import json
import os
import re
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import cv2
import numpy as np

from core.cv_io import imread_unicode, imwrite_unicode
from core.editor_color_match import (
    MIN_PIXELS,
    RING_PX_MAX,
    RING_PX_MIN,
    MSG_NO_OUTER,
    MSG_NO_REFERENCE,
    MSG_RING_NEEDS_MASK,
    MSG_TOO_SMALL,
    MSG_UNREADABLE_REFERENCE,
    ColorMatchError,
    ReferenceStatsCache,
    color_match,
    color_match_ring,
    lab_stats,
    reference_stats,
    ring_bands,
    run_color_match,
    to_unit_float,
    transfer_lab,
)


FRONT = Path(__file__).resolve().parents[1] / 'frontend' / 'src'


def _noise(h=48, w=64, base=(90, 120, 150), spread=25, seed=0):
    rng = np.random.RandomState(seed)
    img = np.array(base, np.float32)[None, None, :] + rng.normal(0, spread, (h, w, 3))
    return np.clip(img, 0, 255).astype(np.uint8)


def _lab_of(img, region=None):
    unit, _ = to_unit_float(img[:, :, :3])
    return lab_stats(cv2.cvtColor(unit, cv2.COLOR_BGR2Lab), region)


class TransferTests(unittest.TestCase):
    def test_reinhard_formula_golden(self):
        lab = np.array([[[50.0, 10.0, -10.0], [70.0, 30.0, 10.0]]], np.float32)
        out = transfer_lab(lab, mu_t=[60, 20, 0], sd_t=[10, 10, 10], mu_r=[40, 0, 5], sd_r=[20, 5, 10])
        np.testing.assert_allclose(out, [[[20.0, -5.0, -5.0], [60.0, 5.0, 15.0]]], atol=1e-4)

    def test_ratio_is_clamped(self):
        lab = np.array([[[51.0, 0.0, 0.0]]], np.float32)
        hi = transfer_lab(lab, [50, 0, 0], [0.0, 1, 1], [50, 0, 0], [100, 1, 1])   # σt=0 → 비율 5 로 잘림
        lo = transfer_lab(lab, [50, 0, 0], [100, 1, 1], [50, 0, 0], [1, 1, 1])     # 0.01 → 0.2
        self.assertAlmostEqual(float(hi[0, 0, 0]), 55.0, places=4)
        self.assertAlmostEqual(float(lo[0, 0, 0]), 50.2, places=4)


class ReferenceModeTests(unittest.TestCase):
    def test_moves_statistics_towards_the_reference(self):
        # 색역 끝에서 잘리지 않을 만큼만 벌린다 — 잘리면 평균이 그만큼 어긋난다(옮기기 자체의 오차가 아니다)
        img = _noise(base=(60, 70, 160), spread=10, seed=1)
        ref = _noise(base=(170, 140, 90), spread=16, seed=2)
        ref_mean, ref_std = reference_stats(ref)
        out = color_match(img, ref_mean, ref_std)
        mean, std = _lab_of(out)
        before, _ = _lab_of(img)
        np.testing.assert_allclose(mean, ref_mean, atol=1.5)
        np.testing.assert_allclose(std, ref_std, rtol=0.15, atol=1.0)
        self.assertGreater(np.abs(before - ref_mean).max(), 10)

    def test_own_statistics_are_an_exact_identity(self):
        img = _noise(seed=3)
        mean, std = reference_stats(img)
        np.testing.assert_array_equal(color_match(img, mean, std), img)

    def test_strength_zero_and_half(self):
        img = _noise(seed=4)
        ref_mean, ref_std = reference_stats(_noise(base=(200, 30, 30), seed=5))
        np.testing.assert_array_equal(color_match(img, ref_mean, ref_std, strength=0.0), img)
        full = color_match(img, ref_mean, ref_std, strength=1.0).astype(int)
        half = color_match(img, ref_mean, ref_std, strength=0.5).astype(int)
        expected = img.astype(int) + (full - img.astype(int)) / 2.0
        self.assertLessEqual(float(np.abs(half - expected).max()), 1.0)

    def test_mask_limits_the_change_and_keeps_alpha(self):
        img = np.dstack([_noise(seed=6), np.random.RandomState(7).randint(0, 256, (48, 64)).astype(np.uint8)])
        mask = np.zeros((48, 64), np.uint8)
        mask[10:30, 20:50] = 255
        ref_mean, ref_std = reference_stats(_noise(base=(30, 200, 40), seed=8))
        out = color_match(img, ref_mean, ref_std, mask=mask)
        outside = mask <= 127
        np.testing.assert_array_equal(out[outside], img[outside])
        np.testing.assert_array_equal(out[:, :, 3], img[:, :, 3])
        self.assertFalse(np.array_equal(out[10:30, 20:50, :3], img[10:30, 20:50, :3]))

    def test_mask_of_other_size_is_matched_without_interpolation(self):
        img = _noise(seed=9)
        small = np.zeros((24, 32), np.uint8)
        small[5:15, 10:25] = 255
        ref_mean, ref_std = reference_stats(_noise(base=(10, 10, 240), seed=10))
        out = color_match(img, ref_mean, ref_std, mask=small)
        big = cv2.resize(small, (64, 48), interpolation=cv2.INTER_NEAREST) > 127
        np.testing.assert_array_equal(out[~big], img[~big])

    def test_transparent_pixels_are_left_out_of_the_statistics(self):
        img = _noise(seed=11)
        alpha = np.full((48, 64), 255, np.uint8)
        alpha[:, :20] = 0
        junk = img.copy()
        junk[:, :20] = (255, 0, 255)       # 투명한 곳의 색은 통계에 들어가면 안 된다
        ref_mean, ref_std = reference_stats(_noise(base=(40, 160, 90), seed=12))
        a = color_match(np.dstack([img, alpha]), ref_mean, ref_std)
        b = color_match(np.dstack([junk, alpha]), ref_mean, ref_std)
        np.testing.assert_array_equal(a[:, 20:], b[:, 20:])

    def test_reference_statistics_skip_transparent_pixels(self):
        ref = _noise(base=(100, 100, 100), spread=5, seed=13)
        alpha = np.full(ref.shape[:2], 255, np.uint8)
        alpha[:, :32] = 0
        polluted = ref.copy()
        polluted[:, :32] = (0, 255, 0)
        mean_a, std_a = reference_stats(np.dstack([ref, alpha]))
        mean_b, std_b = reference_stats(np.dstack([polluted, alpha]))
        np.testing.assert_allclose(mean_a, mean_b)
        np.testing.assert_allclose(std_a, std_b)

    def test_uint16_keeps_dtype(self):
        img = (_noise(seed=14).astype(np.uint16) * 257)
        ref_mean, ref_std = reference_stats(_noise(base=(220, 60, 20), seed=15))
        out = color_match(img, ref_mean, ref_std)
        self.assertEqual(out.dtype, np.uint16)
        self.assertGreater(int(out.max()), 255)

    def test_tiny_mask_is_refused(self):
        img = _noise(seed=16)
        mask = np.zeros((48, 64), np.uint8)
        mask[0, :MIN_PIXELS - 1] = 255
        with self.assertRaises(ColorMatchError) as ctx:
            color_match(img, *reference_stats(img), mask=mask)
        self.assertEqual(str(ctx.exception), MSG_TOO_SMALL)


class RingModeTests(unittest.TestCase):
    def _patched(self):
        img = _noise(base=(60, 140, 200), spread=10, seed=20)
        patch = _noise(base=(200, 80, 40), spread=10, seed=21)
        mask = np.zeros(img.shape[:2], np.uint8)
        mask[12:36, 16:48] = 255
        img[mask > 127] = patch[mask > 127]
        return img, mask

    def test_ring_bands_are_the_two_sides_of_the_edge(self):
        mask = np.zeros((40, 40), bool)
        mask[10:30, 10:30] = True
        inner, outer = ring_bands(mask, 3)
        self.assertFalse((inner & ~mask).any())
        self.assertFalse((outer & mask).any())
        self.assertTrue(inner[10, 15] and inner[12, 15] and not inner[14, 15] and not inner[20, 20])
        self.assertTrue(outer[9, 15] and outer[7, 15] and not outer[5, 15])

    def test_repainted_patch_takes_the_colour_of_its_surroundings(self):
        img, mask = self._patched()
        out = color_match_ring(img, mask, band_px=6)
        inside = mask > 127
        surround, _ = _lab_of(img, ring_bands(inside, 6)[1])
        patched, _ = _lab_of(out, inside)
        np.testing.assert_allclose(patched, surround, atol=2.0)
        np.testing.assert_array_equal(out[~inside], img[~inside])

    def test_mask_covering_everything_has_no_outer_band(self):
        img = _noise(seed=22)
        with self.assertRaises(ColorMatchError) as ctx:
            color_match_ring(img, np.full(img.shape[:2], 255, np.uint8))
        self.assertEqual(str(ctx.exception), MSG_NO_OUTER)

    def test_transparent_surroundings_have_no_outer_band(self):
        img, mask = self._patched()
        alpha = np.where(mask > 127, 255, 0).astype(np.uint8)
        with self.assertRaises(ColorMatchError) as ctx:
            color_match_ring(np.dstack([img, alpha]), mask, band_px=6)
        self.assertEqual(str(ctx.exception), MSG_NO_OUTER)

    def test_tiny_mask_is_refused(self):
        img = _noise(seed=23)
        mask = np.zeros(img.shape[:2], np.uint8)
        mask[20, 20:23] = 255
        with self.assertRaises(ColorMatchError) as ctx:
            color_match_ring(img, mask)
        self.assertEqual(str(ctx.exception), MSG_TOO_SMALL)

    def test_empty_mask_asks_for_painting(self):
        with self.assertRaises(ColorMatchError) as ctx:
            color_match_ring(_noise(seed=24), np.zeros((48, 64), np.uint8))
        self.assertEqual(str(ctx.exception), MSG_RING_NEEDS_MASK)

    def test_alpha_is_untouched(self):
        img, mask = self._patched()
        alpha = np.random.RandomState(25).randint(128, 256, mask.shape).astype(np.uint8)
        out = color_match_ring(np.dstack([img, alpha]), mask, band_px=6)
        np.testing.assert_array_equal(out[:, :, 3], alpha)


class ReferenceCacheTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.root = Path(self._tmp.name)

    def test_decodes_once_per_file_version_and_handles_korean_paths(self):
        path = self.root / '참조 이미지.png'
        imwrite_unicode(path, _noise(seed=30))
        cache = ReferenceStatsCache()
        first = cache.get(str(path))
        second = cache.get(str(path))
        self.assertEqual(cache.decode_count, 1)
        np.testing.assert_array_equal(first[0], second[0])
        imwrite_unicode(path, _noise(base=(10, 10, 10), seed=31))
        st = os.stat(path)
        os.utime(path, ns=(st.st_atime_ns, st.st_mtime_ns + 10_000_000))
        third = cache.get(str(path))
        self.assertEqual(cache.decode_count, 2)
        self.assertFalse(np.allclose(first[0], third[0]))

    def test_unreadable_reference_is_reported(self):
        bad = self.root / 'broken.png'
        bad.write_bytes(b'not an image')
        with self.assertRaises(ColorMatchError) as ctx:
            ReferenceStatsCache().get(str(bad))
        self.assertEqual(str(ctx.exception), MSG_UNREADABLE_REFERENCE)


class RunColorMatchTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.root = Path(self._tmp.name)

    def test_missing_reference_asks_to_load_one(self):
        for params in ({'mode': 'reference'}, {'mode': 'reference', 'reference_path': str(self.root / 'x.png')}):
            with self.subTest(params=params), self.assertRaises(ColorMatchError) as ctx:
                run_color_match(_noise(), params, None)
            self.assertEqual(str(ctx.exception), MSG_NO_REFERENCE)

    def test_ring_without_mask_asks_for_painting(self):
        with self.assertRaises(ColorMatchError) as ctx:
            run_color_match(_noise(), {'mode': 'ring'}, None)
        self.assertEqual(str(ctx.exception), MSG_RING_NEEDS_MASK)

    def test_ring_width_follows_the_preview_scale(self):
        mask = np.zeros((48, 64), np.uint8)
        mask[10:30, 10:40] = 255
        with mock.patch('core.editor_color_match.color_match_ring', return_value='ok') as ring:
            run_color_match(_noise(), {'mode': 'ring', 'ring_px': 32, 'strength': 50}, mask, pixel_scale=0.25)
            run_color_match(_noise(), {'mode': 'ring', 'ring_px': 4}, mask, pixel_scale=0.25)
        self.assertEqual(ring.call_args_list[0].kwargs, {'band_px': 8, 'strength': 0.5})
        self.assertEqual(ring.call_args_list[1].kwargs['band_px'], 2, '프리뷰 띠는 최소 2px')

    def test_reference_mode_uses_the_cache(self):
        path = self.root / 'ref.png'
        imwrite_unicode(path, _noise(base=(200, 50, 50), seed=40))
        cache = ReferenceStatsCache()
        img = _noise(seed=41)
        out = run_color_match(img, {'mode': 'reference', 'reference_path': str(path), 'strength': 100},
                              None, cache=cache)
        run_color_match(img, {'mode': 'reference', 'reference_path': str(path)}, None, cache=cache)
        self.assertEqual(cache.decode_count, 1)
        self.assertFalse(np.array_equal(out, img))

    def test_unknown_mode_is_an_error(self):
        with self.assertRaises(ColorMatchError):
            run_color_match(_noise(), {'mode': 'nope'}, None)


class FrontendContractTests(unittest.TestCase):
    """utils/colorMatch.ts 가 보내는 키·범위·문구가 백엔드와 같은지."""

    def setUp(self):
        self.ts = (FRONT / 'utils' / 'colorMatch.ts').read_text(encoding='utf-8')

    def test_payload_keys(self):
        self.assertIn("params: { mode: 'ring', strength: c.strength, ring_px: c.ringPx }", self.ts)
        self.assertIn("params: { mode: 'reference', strength: c.strength, reference_path: path }", self.ts)

    def test_ring_slider_stays_inside_the_backend_range(self):
        lo = int(re.search(r'export const RING_PX_MIN = (\d+)', self.ts).group(1))
        hi = int(re.search(r'export const RING_PX_MAX = (\d+)', self.ts).group(1))
        self.assertTrue(RING_PX_MIN <= lo < hi <= RING_PX_MAX)

    def test_front_messages_match_backend(self):
        self.assertIn(f"noReference: '{MSG_NO_REFERENCE}'", self.ts)
        self.assertIn(f"ringNeedsMask: '{MSG_RING_NEEDS_MASK}'", self.ts)


class BridgeColorMatchTests(unittest.TestCase):
    """VueBridge 가 color_match 를 core 로 넘기고 오류 문구를 그대로 돌려주는지."""

    def setUp(self):
        from ui.vue_bridge import VueBridge
        self.bridge = VueBridge()
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.root = Path(self._tmp.name)
        self.bridge._editor_temp_dir = str(self.root / 'editor_temp')

    def _run(self, path, params):
        with mock.patch('core.cache_cleanup.prune_editor_temp'):
            return json.loads(self.bridge._editor_process_impl(str(path), 'color_match', params))

    def test_commit_and_preview(self):
        src = self.root / '원본.png'
        ref = self.root / 'ref.png'
        imwrite_unicode(src, np.dstack([_noise(seed=50), np.full((48, 64), 200, np.uint8)]))
        imwrite_unicode(ref, _noise(base=(30, 30, 220), seed=51))
        result = self._run(src, {'mode': 'reference', 'reference_path': str(ref), 'strength': 100})
        self.assertIn('path', result, result)
        out = imread_unicode(result['path'], cv2.IMREAD_UNCHANGED)
        self.assertEqual(out.shape, (48, 64, 4))
        self.assertTrue(np.all(out[:, :, 3] == 200))
        preview = self._run(src, {'mode': 'reference', 'reference_path': str(ref), 'preview': True})
        self.assertTrue(preview.get('preview'), preview)

    def test_errors_are_returned_as_messages(self):
        src = self.root / 'a.png'
        imwrite_unicode(src, _noise(seed=52))
        self.assertEqual(self._run(src, {'mode': 'ring'}), {'error': MSG_RING_NEEDS_MASK})
        self.assertEqual(self._run(src, {'mode': 'reference'}), {'error': MSG_NO_REFERENCE})


if __name__ == '__main__':
    unittest.main()
