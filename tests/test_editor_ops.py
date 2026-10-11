"""에디터 이미지 연산 테스트.

회귀 방지 대상: 이 연산들은 예전에 백엔드에 **아예 구현이 없어서** 원본을 그대로
다시 저장했다(= 사용자에겐 무반응). "출력이 입력과 달라야 한다"를 명시적으로 검증한다.
알파 보존도 함께 검증한다 — 배경 제거 후 편집하면 투명도가 날아가던 버그의 짝.
"""
import unittest

try:
    import cv2
    import numpy as np
    _HAS_CV2 = True
except ImportError:  # pragma: no cover
    _HAS_CV2 = False

if _HAS_CV2:
    from core import editor_ops


def _sample_bgr(h=64, w=48):
    """가로 그라디언트 + 세로 그라디언트 + 노이즈. 상수 이미지는 필터가
    no-op처럼 보일 수 있어 피한다."""
    rng = np.random.RandomState(42)
    img = np.zeros((h, w, 3), dtype=np.float32)
    img[:, :, 0] = np.linspace(0, 255, w, dtype=np.float32)[None, :]
    img[:, :, 1] = np.linspace(0, 255, h, dtype=np.float32)[:, None]
    img[:, :, 2] = 128.0
    img += rng.normal(0, 12, img.shape)
    return np.clip(img, 0, 255).astype(np.uint8)


def _sample_bgra(h=64, w=48):
    bgr = _sample_bgr(h, w)
    alpha = np.zeros((h, w), dtype=np.uint8)
    alpha[h // 4: 3 * h // 4, w // 4: 3 * w // 4] = 255
    return np.dstack([bgr, alpha])


@unittest.skipUnless(_HAS_CV2, "opencv/numpy 미설치")
class TestAlphaHelpers(unittest.TestCase):
    def test_split_merge_roundtrip(self):
        bgra = _sample_bgra()
        bgr, alpha = editor_ops.split_alpha(bgra)
        self.assertEqual(bgr.shape[2], 3)
        self.assertIsNotNone(alpha)
        np.testing.assert_array_equal(editor_ops.merge_alpha(bgr, alpha), bgra)

    def test_split_on_bgr_returns_none_alpha(self):
        bgr, alpha = editor_ops.split_alpha(_sample_bgr())
        self.assertIsNone(alpha)
        self.assertEqual(bgr.shape[2], 3)


@unittest.skipUnless(_HAS_CV2, "opencv/numpy 미설치")
class TestAutoCorrect(unittest.TestCase):
    def test_changes_image(self):
        img = _sample_bgr()
        out = editor_ops.auto_correct(img)
        self.assertEqual(out.shape, img.shape)
        self.assertFalse(np.array_equal(out, img), "auto_correct가 무반응")

    def test_preserves_alpha(self):
        bgra = _sample_bgra()
        out = editor_ops.auto_correct(bgra)
        self.assertEqual(out.shape[2], 4)
        np.testing.assert_array_equal(out[:, :, 3], bgra[:, :, 3])


@unittest.skipUnless(_HAS_CV2, "opencv/numpy 미설치")
class TestAdvColor(unittest.TestCase):
    def test_identity_params_are_near_noop(self):
        img = _sample_bgr()
        out = editor_ops.adv_color(img, black_point=0, white_point=255,
                                   gamma=1.0, temperature=0, tint=0)
        # LUT 반올림 오차만 허용
        self.assertLessEqual(int(np.abs(out.astype(int) - img.astype(int)).max()), 1)

    def test_levels_increase_contrast(self):
        img = _sample_bgr()
        out = editor_ops.adv_color(img, black_point=60, white_point=200)
        self.assertGreater(out.std(), img.std())

    def test_gamma_brightens(self):
        img = _sample_bgr()
        bright = editor_ops.adv_color(img, gamma=2.0)
        dark = editor_ops.adv_color(img, gamma=0.5)
        self.assertGreater(bright.mean(), img.mean())
        self.assertLess(dark.mean(), img.mean())

    def test_temperature_shifts_red_vs_blue(self):
        img = _sample_bgr()
        warm = editor_ops.adv_color(img, temperature=60)
        self.assertGreater(warm[:, :, 2].mean(), img[:, :, 2].mean())   # R↑
        self.assertLess(warm[:, :, 0].mean(), img[:, :, 0].mean())      # B↓

    def test_zero_gamma_falls_back(self):
        img = _sample_bgr()
        out = editor_ops.adv_color(img, gamma=0)
        self.assertEqual(out.shape, img.shape)

    def test_inverted_levels_do_not_crash(self):
        img = _sample_bgr()
        out = editor_ops.adv_color(img, black_point=200, white_point=10)
        self.assertEqual(out.shape, img.shape)

    def test_preserves_alpha(self):
        bgra = _sample_bgra()
        out = editor_ops.adv_color(bgra, gamma=1.5)
        self.assertEqual(out.shape[2], 4)
        np.testing.assert_array_equal(out[:, :, 3], bgra[:, :, 3])


@unittest.skipUnless(_HAS_CV2, "opencv/numpy 미설치")
class TestFilters(unittest.TestCase):
    def test_filter_names_match_frontend_presets(self):
        # ColorPanel.vue presets[].name 과 정확히 일치해야 한다.
        # 이름이 어긋나면 필터가 조용히 죽는다 (예전 filter.name/undefined 버그)
        expected = {
            'grayscale', 'sepia', 'sharpen', 'warm', 'cool', 'soft',
            'invert', 'emboss', 'sketch', 'posterize', 'vignette', 'denoise',
            'pixelart',
        }
        self.assertEqual(set(editor_ops.FILTER_NAMES), expected)

    def test_every_filter_changes_image(self):
        img = _sample_bgr()
        for name in editor_ops.FILTER_NAMES:
            with self.subTest(filter=name):
                out = editor_ops.apply_filter(img, name, 1.0)
                self.assertEqual(out.shape, img.shape)
                self.assertFalse(np.array_equal(out, img), f"{name} 필터가 무반응")

    def test_unknown_filter_raises(self):
        with self.assertRaises(ValueError):
            editor_ops.apply_filter(_sample_bgr(), 'nope')

    def test_none_filter_raises_not_silently_ignored(self):
        # doOp(filter.name || filter.type) 가 undefined를 보내던 버그 방어
        with self.assertRaises(ValueError):
            editor_ops.apply_filter(_sample_bgr(), None)

    def test_strength_zero_is_original(self):
        img = _sample_bgr()
        np.testing.assert_array_equal(editor_ops.apply_filter(img, 'invert', 0.0), img)

    def test_strength_blends(self):
        img = _sample_bgr()
        half = editor_ops.apply_filter(img, 'invert', 0.5)
        full = editor_ops.apply_filter(img, 'invert', 1.0)
        self.assertFalse(np.array_equal(half, img))
        self.assertFalse(np.array_equal(half, full))

    def test_invert_is_exact(self):
        img = _sample_bgr()
        np.testing.assert_array_equal(editor_ops.apply_filter(img, 'invert', 1.0), 255 - img)

    def test_grayscale_channels_equal(self):
        out = editor_ops.apply_filter(_sample_bgr(), 'grayscale', 1.0)
        np.testing.assert_array_equal(out[:, :, 0], out[:, :, 1])
        np.testing.assert_array_equal(out[:, :, 1], out[:, :, 2])

    def test_sepia_is_warm(self):
        # BGR 채널 순서를 뒤집어 적용했는지 검증 — R이 B보다 커야 세피아
        out = editor_ops.apply_filter(np.full((8, 8, 3), 128, dtype=np.uint8), 'sepia', 1.0)
        self.assertGreater(out[:, :, 2].mean(), out[:, :, 0].mean())

    def test_case_insensitive(self):
        img = _sample_bgr()
        np.testing.assert_array_equal(
            editor_ops.apply_filter(img, 'GrayScale', 1.0),
            editor_ops.apply_filter(img, 'grayscale', 1.0))

    def test_preserves_alpha(self):
        bgra = _sample_bgra()
        for name in ('grayscale', 'sepia', 'vignette'):
            with self.subTest(filter=name):
                out = editor_ops.apply_filter(bgra, name, 1.0)
                self.assertEqual(out.shape[2], 4)
                np.testing.assert_array_equal(out[:, :, 3], bgra[:, :, 3])


@unittest.skipUnless(_HAS_CV2, "opencv/numpy 미설치")
class TestMoveRegion(unittest.TestCase):
    def _mask(self, h=64, w=48):
        m = np.zeros((h, w), dtype=np.uint8)
        m[20:40, 10:30] = 255
        return m

    def test_moves_pixels_and_fills_hole(self):
        img = _sample_bgr()
        mask = self._mask()
        out = editor_ops.move_region(img, mask, dx=10, dy=0, fill_color='black')
        self.assertEqual(out.shape, img.shape)
        # 원래 자리 왼쪽 끝(이동해서 비는 곳)은 검게 채워져야 한다
        self.assertEqual(int(out[30, 12].sum()), 0)
        self.assertFalse(np.array_equal(out, img))

    def test_white_fill(self):
        out = editor_ops.move_region(_sample_bgr(), self._mask(), dx=15, fill_color='white')
        np.testing.assert_array_equal(out[30, 12], [255, 255, 255])

    def test_inpaint_fill_is_not_solid(self):
        out = editor_ops.move_region(_sample_bgr(), self._mask(), dx=15, fill_color='inpaint')
        patch = out[25:35, 11:14]
        self.assertGreater(int(patch.max()), 0)

    def test_empty_mask_is_noop(self):
        img = _sample_bgr()
        out = editor_ops.move_region(img, np.zeros((64, 48), dtype=np.uint8), dx=10)
        np.testing.assert_array_equal(out, img)

    def test_none_mask_is_noop(self):
        img = _sample_bgr()
        np.testing.assert_array_equal(editor_ops.move_region(img, None, dx=10), img)

    def test_rotation_and_scale_run(self):
        out = editor_ops.move_region(_sample_bgr(), self._mask(),
                                     dx=5, dy=5, rotation=30, scale=150)
        self.assertEqual(out.shape, (64, 48, 3))

    def test_zero_scale_does_not_crash(self):
        out = editor_ops.move_region(_sample_bgr(), self._mask(), scale=0)
        self.assertEqual(out.shape, (64, 48, 3))

    def test_preserves_alpha_channel_count(self):
        out = editor_ops.move_region(_sample_bgra(), self._mask(), dx=8)
        self.assertEqual(out.shape[2], 4)


@unittest.skipUnless(_HAS_CV2, "opencv/numpy 미설치")
class TestPerspective(unittest.TestCase):
    def test_warps_to_requested_size(self):
        img = _sample_bgr(64, 48)
        out = editor_ops.perspective(img, [[0, 0], [47, 5], [47, 60], [0, 63]],
                                     width=40, height=50)
        self.assertEqual(out.shape[:2], (50, 40))

    def test_infers_size_when_omitted(self):
        img = _sample_bgr(64, 48)
        out = editor_ops.perspective(img, [[0, 0], [47, 0], [47, 63], [0, 63]])
        self.assertEqual(out.shape[:2], (63, 47))

    def test_bad_corner_count_raises(self):
        with self.assertRaises(ValueError):
            editor_ops.perspective(_sample_bgr(), [[0, 0], [1, 1]])

    def test_corner_order_is_tl_tr_br_bl(self):
        """꼭짓점 순서 계약 — 좌상 → 우상 → 우하 → 좌하.

        EditorCanvas.beginPerspective()가 이 순서로 만들어 보내므로, 여기가 어긋나면
        결과가 조용히 회전/반전된다(에러도 안 남). 사분면마다 다른 색을 칠해두고
        항등 변환 후에도 색 배치가 유지되는지로 검증한다.
        """
        img = np.zeros((40, 40, 3), dtype=np.uint8)
        img[:20, :20] = (255, 0, 0)      # 좌상 = 파랑(BGR)
        img[:20, 20:] = (0, 255, 0)      # 우상 = 초록
        img[20:, 20:] = (0, 0, 255)      # 우하 = 빨강
        img[20:, :20] = (255, 255, 255)  # 좌하 = 흰색

        out = editor_ops.perspective(
            img, [[0, 0], [39, 0], [39, 39], [0, 39]], width=40, height=40)

        self.assertEqual(tuple(int(v) for v in out[5, 5]), (255, 0, 0))
        self.assertEqual(tuple(int(v) for v in out[5, 34]), (0, 255, 0))
        self.assertEqual(tuple(int(v) for v in out[34, 34]), (0, 0, 255))
        self.assertEqual(tuple(int(v) for v in out[34, 5]), (255, 255, 255))

    def test_integer_corner_pairs_accepted(self):
        """프론트는 Math.round()로 정수 [[x,y] x4] 를 보낸다."""
        out = editor_ops.perspective(_sample_bgr(64, 48),
                                     [[2, 3], [45, 1], [46, 60], [1, 62]])
        self.assertEqual(out.ndim, 3)

    def test_trapezoid_is_straightened(self):
        """기울어진 사각형이 실제로 펴지는지 — 위쪽이 좁은 사다리꼴 입력."""
        img = np.zeros((60, 60, 3), dtype=np.uint8)
        img[:, :] = (30, 30, 30)
        img[10:50, 15:45] = (200, 200, 200)
        out = editor_ops.perspective(img, [[20, 10], [40, 10], [45, 49], [15, 49]],
                                     width=30, height=40)
        self.assertEqual(out.shape[:2], (40, 30))
        # 편 결과의 중앙은 밝은 영역에서 왔어야 한다
        self.assertGreater(int(out[20, 15].mean()), 100)

    def test_preserves_alpha(self):
        out = editor_ops.perspective(_sample_bgra(), [[0, 0], [47, 0], [47, 63], [0, 63]],
                                     width=20, height=20)
        self.assertEqual(out.shape[2], 4)


@unittest.skipUnless(_HAS_CV2, "opencv/numpy 미설치")
class TestFlatten(unittest.TestCase):
    def test_no_overlay_is_noop(self):
        img = _sample_bgr()
        np.testing.assert_array_equal(editor_ops.flatten(img, None), img)

    def test_opaque_overlay_replaces(self):
        img = _sample_bgr(32, 32)
        overlay = np.zeros((32, 32, 4), dtype=np.uint8)
        overlay[:, :, 2] = 255      # 빨강
        overlay[:, :, 3] = 255      # 완전 불투명
        out = editor_ops.flatten(img, overlay)
        np.testing.assert_array_equal(out[:, :, 2], np.full((32, 32), 255, dtype=np.uint8))
        np.testing.assert_array_equal(out[:, :, 0], np.zeros((32, 32), dtype=np.uint8))

    def test_transparent_overlay_is_noop(self):
        img = _sample_bgr(32, 32)
        overlay = np.zeros((32, 32, 4), dtype=np.uint8)
        overlay[:, :, 2] = 255
        # alpha = 0
        np.testing.assert_array_equal(editor_ops.flatten(img, overlay), img)

    def test_opacity_scales_blend(self):
        img = np.zeros((16, 16, 3), dtype=np.uint8)
        overlay = np.zeros((16, 16, 4), dtype=np.uint8)
        overlay[:, :, 2] = 200
        overlay[:, :, 3] = 255
        out = editor_ops.flatten(img, overlay, opacity=0.5)
        self.assertEqual(int(out[0, 0, 2]), 100)

    def test_overlay_is_resized_to_base(self):
        img = _sample_bgr(64, 48)
        overlay = np.zeros((16, 16, 4), dtype=np.uint8)
        overlay[:, :, 3] = 255
        out = editor_ops.flatten(img, overlay)
        self.assertEqual(out.shape[:2], (64, 48))


@unittest.skipUnless(_HAS_CV2, 'cv2 필요')
class HealTests(unittest.TestCase):
    """복원 브러시 — 칠한 자리를 주변 픽셀로 메운다.

    프론트는 마스크를 PNG 로 보낸다. `cv2.inpaint` 는 8UC1 만 받으므로 채널을
    줄이는 단계가 필요하고, 그걸 빼먹으면 조용히 예외가 나 '무반응'이 된다.
    """

    def _blot(self):
        """가운데에 튀는 얼룩을 넣은 이미지와 그 자리를 가리키는 마스크."""
        img = _sample_bgr(48, 48)
        img[20:28, 20:28] = (0, 0, 255)
        mask = np.zeros((48, 48), dtype=np.uint8)
        mask[20:28, 20:28] = 255
        return img, mask

    def test_marked_area_changes_and_rest_does_not(self):
        img, mask = self._blot()
        out = editor_ops.heal(img, mask, radius=3)
        self.assertFalse(np.array_equal(out[20:28, 20:28], img[20:28, 20:28]),
                         '칠한 자리가 그대로다 — inpaint 가 돌지 않았다')
        np.testing.assert_array_equal(out[:15, :15], img[:15, :15])

    def test_png_shaped_mask_is_reduced_to_one_channel(self):
        """프론트가 보내는 것은 3~4채널 PNG 다. 그대로 넘기면 cv2 가 거부한다."""
        img, mask = self._blot()
        gray = editor_ops.heal(img, mask)
        for channels in (3, 4):
            stacked = np.dstack([mask] * channels)
            if channels == 4:
                stacked[:, :, 3] = 255
            np.testing.assert_array_equal(editor_ops.heal(img, stacked), gray)

    def test_mask_of_different_size_is_matched_without_interpolation(self):
        """보간으로 줄이면 마스크 가장자리에 중간값이 생겨 칠하지 않은 곳까지 지운다."""
        img = _sample_bgr(48, 48)
        mask = np.zeros((24, 24), dtype=np.uint8)
        mask[10:14, 10:14] = 255
        out = editor_ops.heal(img, mask)
        self.assertEqual(out.shape, img.shape)
        np.testing.assert_array_equal(out[:8, :8], img[:8, :8])

    def test_empty_mask_returns_original(self):
        img = _sample_bgr(32, 32)
        np.testing.assert_array_equal(editor_ops.heal(img, np.zeros((32, 32), np.uint8)), img)
        np.testing.assert_array_equal(editor_ops.heal(img, None), img)

    def test_alpha_is_preserved(self):
        img, mask = self._blot()
        rgba = np.dstack([img, np.full((48, 48), 123, dtype=np.uint8)])
        out = editor_ops.heal(rgba, mask)
        self.assertEqual(out.shape[2], 4)
        self.assertTrue(np.all(out[:, :, 3] == 123))


if __name__ == '__main__':
    unittest.main()
