"""픽셀아트 필터(core.editor_pixel_art + editor_ops.FILTERS['pixelart']) 테스트.

블록 평균·가장자리 블록·알파 가중 평균·팔레트·dtype·프리뷰 배율을 골든 값으로 고정한다.
프론트 슬라이더 범위(utils/pixelArt.ts)와 백엔드 FILTER_OPTIONS 가 갈라지지 않는지도 본다.
"""
from __future__ import annotations

import base64
import json
import re
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import cv2
import numpy as np

from core.cv_io import imread_unicode, imwrite_unicode

from core import editor_ops
from core.editor_pixel_art import pixel_art

FRONT = Path(__file__).resolve().parents[1] / 'frontend' / 'src'


def _bgr(rows):
    """[[v, ...], ...] 회색값 → 세 채널이 같은 uint8 BGR."""
    a = np.array(rows, dtype=np.uint8)
    return np.dstack([a, a, a])


class BlockMeanTests(unittest.TestCase):
    def test_two_by_two_blocks_are_box_means(self):
        img = _bgr([[0, 10, 100, 100],
                    [20, 30, 100, 200],
                    [5, 5, 7, 7],
                    [5, 5, 7, 8]])
        out = pixel_art(img, pixel_size=2, colors=0)
        expected = np.array([[15, 15, 125, 125],
                             [15, 15, 125, 125],
                             [5, 5, 7, 7],
                             [5, 5, 7, 7]], dtype=np.uint8)   # 7.25 → 7
        np.testing.assert_array_equal(out[:, :, 0], expected)
        np.testing.assert_array_equal(out[:, :, 0], out[:, :, 2])

    def test_partial_edge_blocks_use_replicated_padding_and_stay_uniform(self):
        img = _bgr([[0, 2, 40],
                    [4, 6, 60],
                    [9, 9, 99]])
        out = pixel_art(img, pixel_size=2, colors=0)[:, :, 0]
        # 오른쪽 열 블록 = (40,40,60,60) 평균 50, 아래 행 블록 = (9,9,9,9), 모서리 = 99
        np.testing.assert_array_equal(out, np.array([[3, 3, 50], [3, 3, 50], [9, 9, 99]], np.uint8))

    def test_blocks_are_anchored_top_left(self):
        rng = np.random.RandomState(0)
        img = rng.randint(0, 256, (37, 29, 3)).astype(np.uint8)
        out = pixel_art(img, pixel_size=8, colors=0)
        for y in range(0, 37, 8):
            for x in range(0, 29, 8):
                block = out[y:y + 8, x:x + 8].reshape(-1, 3)
                self.assertTrue((block == block[0]).all(), f'블록 ({y},{x}) 이 한 색이 아니다')

    def test_same_shape_and_dtype(self):
        img = np.zeros((17, 23, 3), np.uint8)
        out = pixel_art(img, pixel_size=5, colors=0)
        self.assertEqual(out.shape, img.shape)
        self.assertEqual(out.dtype, np.uint8)

    def test_pixel_scale_shrinks_the_block_for_previews(self):
        rng = np.random.RandomState(1)
        img = rng.randint(0, 256, (32, 32, 3)).astype(np.uint8)
        np.testing.assert_array_equal(pixel_art(img, pixel_size=8, colors=0, pixel_scale=0.25),
                                      pixel_art(img, pixel_size=2, colors=0))

    def test_block_size_one_skips_blocking(self):
        rng = np.random.RandomState(2)
        img = rng.randint(0, 256, (16, 16, 3)).astype(np.uint8)
        np.testing.assert_array_equal(pixel_art(img, pixel_size=2, colors=0, pixel_scale=0.25), img)


class AlphaWeightTests(unittest.TestCase):
    def test_hidden_rgb_under_transparency_does_not_bleed(self):
        img = np.zeros((2, 2, 3), np.uint8)
        img[:] = (255, 0, 0)            # 파랑
        img[0, 0] = (0, 0, 255)         # 투명한 곳에 숨은 빨강
        alpha = np.full((2, 2), 255, np.uint8)
        alpha[0, 0] = 0
        out = pixel_art(img, pixel_size=2, colors=0, alpha=alpha)
        np.testing.assert_array_equal(out.reshape(-1, 3), np.tile([255, 0, 0], (4, 1)))

    def test_fully_transparent_block_falls_back_to_plain_mean(self):
        img = _bgr([[0, 100], [100, 200]])
        out = pixel_art(img, pixel_size=2, colors=0, alpha=np.zeros((2, 2), np.uint8))
        self.assertEqual(int(out[0, 0, 0]), 100)

    def test_apply_filter_keeps_alpha_bytes(self):
        rng = np.random.RandomState(3)
        bgra = rng.randint(0, 256, (24, 24, 4)).astype(np.uint8)
        out = editor_ops.apply_filter(bgra, 'pixelart', 1.0, options={'pixel_size': 4, 'colors': 0})
        self.assertEqual(out.shape, bgra.shape)
        np.testing.assert_array_equal(out[:, :, 3], bgra[:, :, 3])


class PaletteTests(unittest.TestCase):
    def _four_colours(self):
        img = np.zeros((16, 16, 3), np.uint8)
        img[:8, :8] = (10, 10, 10)
        img[:8, 8:] = (20, 20, 20)
        img[8:, :8] = (200, 0, 0)
        img[8:, 8:] = (0, 0, 220)
        return img

    def test_colour_count_is_limited(self):
        out = pixel_art(self._four_colours(), pixel_size=2, colors=2)
        self.assertLessEqual(len(np.unique(out.reshape(-1, 3), axis=0)), 2)

    def test_palette_is_deterministic(self):
        rng = np.random.RandomState(4)
        img = rng.randint(0, 256, (64, 64, 3)).astype(np.uint8)
        a = pixel_art(img, pixel_size=4, colors=8)
        b = pixel_art(img, pixel_size=4, colors=8)
        np.testing.assert_array_equal(a, b)
        self.assertLessEqual(len(np.unique(a.reshape(-1, 3), axis=0)), 8)

    def test_enough_colours_keep_the_blocks(self):
        img = self._four_colours()
        np.testing.assert_array_equal(pixel_art(img, pixel_size=2, colors=16),
                                      pixel_art(img, pixel_size=2, colors=0))

    def test_uint16_keeps_dtype_and_maps_to_8bit_steps(self):
        rng = np.random.RandomState(5)
        img = (rng.randint(0, 256, (20, 20, 3)) * 257).astype(np.uint16)
        out = pixel_art(img, pixel_size=4, colors=4)
        self.assertEqual(out.dtype, np.uint16)
        self.assertTrue(np.all(out % 257 == 0), '16비트 팔레트 색은 8비트 값 ×257 이어야 한다')
        self.assertLessEqual(len(np.unique(out.reshape(-1, 3), axis=0)), 4)

    def test_palette_ignores_hidden_rgb_under_full_transparency(self):
        # 배경을 지운 이미지 — 투명한 곳의 숨은 검정이 팔레트 한 칸을 차지해 보이는 색이 colors-1 개가 됐다
        h = w = 64
        x = np.linspace(0, 1, w)
        bgr = np.zeros((h, w, 3), np.uint8)
        bgr[..., 2] = 230
        bgr[..., 1] = (40 + 180 * x[None, :]).astype(np.uint8)
        bgr[..., 0] = 30
        yy, xx = np.mgrid[:h, :w]
        disc = (yy - 32) ** 2 + (xx - 32) ** 2 < 18 ** 2
        bgr[~disc] = 0
        img = np.dstack([bgr, np.where(disc, 255, 0).astype(np.uint8)])
        for k in (2, 3, 4):
            with self.subTest(colors=k):
                out = editor_ops.apply_filter(img, 'pixelart', 1.0, options={'pixel_size': 4, 'colors': k})
                visible = out[..., :3][out[..., 3] == 255]
                self.assertEqual(len(np.unique(visible, axis=0)), k)
        # 알파가 없거나 전부 보이면 예전과 같다
        np.testing.assert_array_equal(pixel_art(bgr, pixel_size=4, colors=4, alpha=np.full((h, w), 255, np.uint8)),
                                      pixel_art(bgr, pixel_size=4, colors=4))

    def test_uint16_without_palette_stays_16bit(self):
        img = np.array([[[1000, 2000, 3000], [3000, 2000, 1001]]], dtype=np.uint16)
        out = pixel_art(img, pixel_size=2, colors=0)
        self.assertEqual(out.dtype, np.uint16)
        np.testing.assert_array_equal(out[0, 0], [2000, 2000, 2000])


class StrengthBlendDtypeTests(unittest.TestCase):
    def test_16bit_strength_below_100_keeps_16bit_values(self):
        # 예전 _blend 는 uint8 로 잘라 16비트 PNG 가 세기 50% 에서 거의 흰색(BGRA 는 거의 검정)이 됐다
        rng = np.random.RandomState(9)
        img8 = rng.randint(0, 256, (32, 32, 3)).astype(np.uint8)
        img16 = img8.astype(np.uint16) * 257
        half8 = editor_ops.apply_filter(img8, 'pixelart', 0.5, options={'pixel_size': 4, 'colors': 0})
        half16 = editor_ops.apply_filter(img16, 'pixelart', 0.5, options={'pixel_size': 4, 'colors': 0})
        self.assertEqual(half16.dtype, np.uint16)
        self.assertLess(np.abs(half16.astype(np.float64) / 257 - half8).max(), 1.01)
        bgra16 = np.dstack([img16, np.full((32, 32), 65535, np.uint16)])
        out = editor_ops.apply_filter(bgra16, 'pixelart', 0.5, options={'pixel_size': 4, 'colors': 0})
        self.assertEqual(out.dtype, np.uint16)
        np.testing.assert_array_equal(out[..., :3], half16)
        self.assertTrue(np.all(out[..., 3] == 65535))


class FilterOptionTests(unittest.TestCase):
    def test_pixelart_is_a_filter_with_options(self):
        self.assertIn('pixelart', editor_ops.FILTERS)
        self.assertTrue(set(editor_ops.FILTER_OPTIONS) <= set(editor_ops.FILTERS))

    def test_options_are_clamped_and_defaulted(self):
        self.assertEqual(editor_ops._filter_options('pixelart', {'pixel_size': 999, 'colors': -3}),
                         {'pixel_size': 64, 'colors': 0})
        self.assertEqual(editor_ops._filter_options('pixelart', {'pixel_size': '12', 'junk': 1}),
                         {'pixel_size': 12, 'colors': 16})
        self.assertEqual(editor_ops._filter_options('pixelart', None), {'pixel_size': 8, 'colors': 16})
        self.assertEqual(editor_ops._filter_options('pixelart', ['x']), {'pixel_size': 8, 'colors': 16})
        self.assertEqual(editor_ops._filter_options('pixelart', {'pixel_size': True, 'colors': 'nan'}),
                         {'pixel_size': 8, 'colors': 16})

    def test_apply_filter_passes_options_and_preview_scale(self):
        rng = np.random.RandomState(6)
        img = rng.randint(0, 256, (32, 32, 3)).astype(np.uint8)
        via_filter = editor_ops.apply_filter(img, 'pixelart', 1.0, options={'pixel_size': 8, 'colors': 0},
                                             pixel_scale=0.5)
        np.testing.assert_array_equal(via_filter, pixel_art(img, pixel_size=4, colors=0))

    def test_other_filters_ignore_options(self):
        img = np.full((8, 8, 3), 100, np.uint8)
        np.testing.assert_array_equal(editor_ops.apply_filter(img, 'invert', 1.0, options={'pixel_size': 4}),
                                      255 - img)

    def test_strength_blends_like_the_other_filters(self):
        rng = np.random.RandomState(7)
        img = rng.randint(0, 256, (16, 16, 3)).astype(np.uint8)
        np.testing.assert_array_equal(editor_ops.apply_filter(img, 'pixelart', 0.0), img)
        half = editor_ops.apply_filter(img, 'pixelart', 0.5, options={'pixel_size': 4, 'colors': 0})
        full = editor_ops.apply_filter(img, 'pixelart', 1.0, options={'pixel_size': 4, 'colors': 0})
        self.assertFalse(np.array_equal(half, img))
        self.assertFalse(np.array_equal(half, full))

    def test_frontend_slider_ranges_match_backend(self):
        source = (FRONT / 'utils' / 'pixelArt.ts').read_text(encoding='utf-8')
        spec = editor_ops.FILTER_OPTIONS['pixelart']
        for key, ts_name in (('pixel_size', 'PIXEL_SIZE'), ('colors', 'COLORS')):
            default, lo, hi = spec[key]
            with self.subTest(option=key):
                for suffix, value in (('MIN', lo), ('MAX', hi), ('DEFAULT', default)):
                    m = re.search(rf'export const {ts_name}_{suffix}\s*=\s*(\d+)', source)
                    self.assertIsNotNone(m, f'{ts_name}_{suffix} 를 찾지 못했다')
                    self.assertEqual(int(m.group(1)), value, f'{ts_name}_{suffix} 가 백엔드와 다르다')

    def test_colour_panel_offers_the_preset(self):
        panel = (FRONT / 'components' / 'editor' / 'ColorPanel.vue').read_text(encoding='utf-8')
        self.assertIn("name: 'pixelart'", panel)


class BridgePixelArtTests(unittest.TestCase):
    """VueBridge 가 filter 의 options 를 넘기고, 프리뷰는 블록 크기에 축소 배율을 곱하는지."""

    def setUp(self):
        from ui.vue_bridge import VueBridge
        self.bridge = VueBridge()
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.root = Path(self._tmp.name)
        self.bridge._editor_temp_dir = str(self.root / 'editor_temp')

    def _run(self, path, params):
        with mock.patch('core.cache_cleanup.prune_editor_temp'):
            return json.loads(self.bridge._editor_process_impl(str(path), 'filter', params))

    def test_commit_uses_the_options(self):
        src = self.root / '픽셀.png'
        img = np.random.RandomState(8).randint(0, 256, (24, 24, 3)).astype(np.uint8)
        imwrite_unicode(src, img)
        result = self._run(src, {'filter': 'pixelart', 'strength': 100, 'options': {'pixel_size': 6, 'colors': 0}})
        self.assertIn('path', result, result)
        out = imread_unicode(result['path'], cv2.IMREAD_UNCHANGED)
        np.testing.assert_array_equal(out, pixel_art(img, pixel_size=6, colors=0))

    def test_preview_scales_the_block_with_the_preview(self):
        # 원본 16px 칸마다 다른 색 → 1024 축소본에서는 8px 칸. 프리뷰 블록이 16 × 0.5 = 8px 이면
        # 칸이 그대로 남고, 축소 배율을 안 곱해 16px 이면 네 칸이 섞인다.
        # 불투명 알파를 붙여 프리뷰가 무손실 PNG 로 오게 한다(JPEG 는 칸 색을 흐린다).
        src = self.root / 'big.png'
        cells = np.random.RandomState(9).randint(0, 256, (128, 128, 3)).astype(np.uint8)
        bgr = np.kron(cells, np.ones((16, 16, 1), np.uint8))
        imwrite_unicode(src, np.dstack([bgr, np.full(bgr.shape[:2], 255, np.uint8)]))
        preview = self._run(src, {'filter': 'pixelart', 'strength': 100, 'preview': True,
                                  'options': {'pixel_size': 16, 'colors': 0}})
        self.assertTrue(preview.get('preview'), preview)
        self.assertTrue(preview['image_base64'].startswith('data:image/png'))
        raw = base64.b64decode(preview['image_base64'].split(',', 1)[1])
        shown = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_UNCHANGED)
        self.assertEqual(shown.shape, (1024, 1024, 4))
        small = cv2.resize(bgr, (1024, 1024), interpolation=cv2.INTER_AREA)
        np.testing.assert_array_equal(shown[:, :, :3], pixel_art(small, pixel_size=8, colors=0))
        self.assertFalse(np.array_equal(shown[:, :, :3], pixel_art(small, pixel_size=16, colors=0)),
                         '프리뷰가 원본 px 블록(16) 그대로다 — pixel_scale 이 안 들어갔다')

if __name__ == '__main__':
    unittest.main()
