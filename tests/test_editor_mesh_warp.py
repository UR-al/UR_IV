"""3×3 격자 변형(core.editor_mesh_warp + VueBridge 배선) 테스트.

움직이지 않은 격자 = 정확한 항등, 끈 점으로 내용이 따라가는지, 가장자리 고정, 알파가 함께 움직이는지,
TPS 가 제어점을 정확히 지나는지, 성긴 격자 보간 오차, 페이로드 검증을 고정한다.
"""
from __future__ import annotations

import json
import math
import re
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import cv2
import numpy as np

from core.cv_io import imread_unicode, imwrite_unicode
from core.editor_mesh_warp import (
    GRID,
    MSG_NEED_POINTS,
    MSG_OVERLAP,
    MeshWarpError,
    anchor_points,
    control_points,
    edge_anchor_uv,
    edge_indices,
    grid_uv,
    mesh_warp,
    parse_mesh_params,
    tps_eval,
    tps_fit,
    uv_to_px,
    warp_maps,
)


FRONT = Path(__file__).resolve().parents[1] / 'frontend' / 'src'


def _textured(h=90, w=120, channels=3, dtype=np.uint8, seed=0):
    rng = np.random.RandomState(seed)
    img = rng.randint(0, 256, (h, w, channels))
    if dtype == np.uint16:
        img = img * 257
    return img.astype(dtype)


def _moved(index, du, dv):
    dst = grid_uv()
    dst[index] += (du, dv)
    return dst


class GridTests(unittest.TestCase):
    def test_grid_is_row_major_from_top_left(self):
        g = grid_uv()
        self.assertEqual(g.shape, (16, 2))
        np.testing.assert_allclose(g[0], (0, 0))
        np.testing.assert_allclose(g[1], (1 / 3, 0))
        np.testing.assert_allclose(g[4], (0, 1 / 3))
        np.testing.assert_allclose(g[15], (1, 1))

    def test_twelve_border_points_and_four_inner(self):
        edge = edge_indices()
        self.assertEqual(int(edge.sum()), 12)
        self.assertEqual(sorted(np.flatnonzero(~edge).tolist()), [5, 6, 9, 10])

    def test_anchors_skip_grid_positions(self):
        anchors = edge_anchor_uv()
        self.assertEqual(anchors.shape, (84, 2))
        g = grid_uv()
        d = np.sqrt(((anchors[:, None, :] - g[None, :, :]) ** 2).sum(-1))
        self.assertGreater(float(d.min()), 1 / 48)
        on_border = (np.isclose(anchors, 0) | np.isclose(anchors, 1)).any(axis=1)
        self.assertTrue(on_border.all())

    def test_unpinned_anchors_follow_the_moved_border(self):
        dst = _moved(1, 0.0, 0.1)   # 위쪽 변의 점을 아래로
        moved = anchor_points(dst)
        straight = edge_anchor_uv()
        # 위쪽 변 t=1/24 앵커(첫 번째) 는 (0,0)→(1/3,0.1) 선분 위 1/8 지점
        np.testing.assert_allclose(moved[0], (1 / 24, 0.1 / 8), atol=1e-12)
        np.testing.assert_allclose(moved[1], straight[1])   # 아래쪽 변은 그대로

    def test_uv_to_px_maps_edges_to_pixel_borders(self):
        np.testing.assert_allclose(uv_to_px(np.array([[0, 0], [1, 1]]), 100, 50), [[-0.5, -0.5], [99.5, 49.5]])


class TpsTests(unittest.TestCase):
    def test_interpolates_control_points_exactly(self):
        rng = np.random.RandomState(1)
        ctrl = rng.rand(20, 2)
        vals = rng.rand(20, 2)
        weights, affine = tps_fit(ctrl, vals)
        np.testing.assert_allclose(tps_eval(ctrl, weights, affine, ctrl), vals, atol=1e-9)

    def test_affine_moves_are_reproduced_exactly(self):
        w, h = 160, 120
        src, _ = control_points(grid_uv(), pin_edges=False)
        dst = src * 0.9 + 0.05
        map_x, map_y = warp_maps(w, h, uv_to_px(src, w, h), uv_to_px(dst, w, h))
        xs = np.arange(w)
        ys = np.arange(h)
        np.testing.assert_allclose(map_x[0], (xs + 0.5 - 0.05 * w) / 0.9 - 0.5, atol=1e-3)
        np.testing.assert_allclose(map_y[:, 0], (ys + 0.5 - 0.05 * h) / 0.9 - 0.5, atol=1e-3)

    def test_coarse_grid_interpolation_stays_sub_pixel(self):
        w, h = 1024, 768
        src, dst = control_points(_moved(5, 0.05, 0.03))
        src_px, dst_px = uv_to_px(src, w, h), uv_to_px(dst, w, h)
        map_x, map_y = warp_maps(w, h, src_px, dst_px)
        scale = float(max(w, h))
        weights, affine = tps_fit(dst_px / scale, src_px / scale)
        cx, cy = int(dst_px[5, 0]), int(dst_px[5, 1])
        yy, xx = np.mgrid[cy - 30:cy + 30, cx - 30:cx + 30]
        exact = tps_eval(dst_px / scale, weights, affine, np.stack([xx.ravel(), yy.ravel()], 1) / scale) * scale
        err = max(np.abs(map_x[yy, xx].ravel() - exact[:, 0]).max(), np.abs(map_y[yy, xx].ravel() - exact[:, 1]).max())
        self.assertLess(err, 0.15, f'끈 점 근처 보간 오차 {err:.3f}px')


class MeshWarpTests(unittest.TestCase):
    def test_untouched_grid_is_an_exact_identity(self):
        for img in (_textured(), _textured(channels=4), _textured(dtype=np.uint16)):
            with self.subTest(shape=img.shape, dtype=img.dtype):
                out = mesh_warp(img, grid_uv())
                np.testing.assert_array_equal(out, img)
                self.assertIsNot(out, img)

    def test_pinned_border_points_are_ignored(self):
        img = _textured()
        np.testing.assert_array_equal(mesh_warp(img, _moved(1, 0.1, 0.1), pin_edges=True), img)
        self.assertFalse(np.array_equal(mesh_warp(img, _moved(1, 0.1, 0.1), pin_edges=False), img))

    def test_content_follows_the_dragged_point(self):
        w, h = 240, 180
        img = np.zeros((h, w, 3), np.uint8)
        sx, sy = w / 3.0, h / 3.0                # 격자점 5 의 원래 자리(캔버스 좌표)
        cv2.circle(img, (int(round(sx - 0.5)), int(round(sy - 0.5))), 3, (255, 255, 255), -1)
        out = mesh_warp(img, _moved(5, 0.08, 0.05))
        ys, xs = np.nonzero(out[:, :, 0] > 127)
        self.assertGreater(len(xs), 0)
        target_x, target_y = (1 / 3 + 0.08) * w - 0.5, (1 / 3 + 0.05) * h - 0.5
        self.assertLess(math.hypot(xs.mean() - target_x, ys.mean() - target_y), 1.0)

    def test_pinned_borders_barely_move(self):
        w, h = 400, 300
        src, dst = control_points(_moved(5, 0.1, 0.08))
        map_x, map_y = warp_maps(w, h, uv_to_px(src, w, h), uv_to_px(dst, w, h))
        self.assertLess(float(np.abs(map_x[:, 0]).max()), 0.5)
        self.assertLess(float(np.abs(map_y[0, :]).max()), 0.5)
        self.assertLess(float(np.abs(map_x[:, -1] - (w - 1)).max()), 0.5)

    def test_alpha_moves_with_the_content(self):
        w, h = 120, 90
        img = np.zeros((h, w, 4), np.uint8)
        img[20:40, 30:50] = (0, 0, 255, 255)
        out = mesh_warp(img, _moved(5, 0.1, 0.0))
        self.assertEqual(out.shape, img.shape)
        np.testing.assert_array_equal(out[:, :, 3] > 127, out[:, :, 2] > 127)
        self.assertFalse(np.array_equal(out[:, :, 3], img[:, :, 3]))

    def test_dtype_and_shape_are_kept(self):
        img = _textured(dtype=np.uint16)
        out = mesh_warp(img, _moved(10, -0.05, 0.04))
        self.assertEqual(out.shape, img.shape)
        self.assertEqual(out.dtype, np.uint16)

    def test_coincident_points_are_refused(self):
        dst = grid_uv()
        dst[5] = dst[6] = (0.5, 1 / 3)
        with self.assertRaises(MeshWarpError) as ctx:
            mesh_warp(_textured(), dst)
        self.assertEqual(str(ctx.exception), MSG_OVERLAP)


class ParamTests(unittest.TestCase):
    def test_valid_payload(self):
        dst = grid_uv().tolist()
        dst[5] = [1.5, -0.2]
        pts, pin = parse_mesh_params({'grid': 4, 'dst': dst, 'pin_edges': False})
        self.assertEqual(pts.shape, (GRID * GRID, 2))
        np.testing.assert_allclose(pts[5], (1.0, 0.0))
        self.assertFalse(pin)
        self.assertTrue(parse_mesh_params({'dst': dst})[1], '가장자리 고정 기본값은 켜짐')

    def test_bad_payloads_need_sixteen_points(self):
        good = grid_uv().tolist()
        bad_cases = [
            {'dst': good[:15]},
            {'dst': good, 'grid': 3},
            {'dst': good[:15] + [[0.5, float('nan')]]},
            {'dst': good[:15] + [[True, 0.5]]},
            {'dst': good[:15] + [[0.5]]},
            {'dst': 'nope'},
            {},
        ]
        for params in bad_cases:
            with self.subTest(params=str(params)[:40]), self.assertRaises(MeshWarpError) as ctx:
                parse_mesh_params(params)
            self.assertEqual(str(ctx.exception), MSG_NEED_POINTS)


class FrontendContractTests(unittest.TestCase):
    """프론트가 보내는 격자 크기·페이로드 키가 백엔드와 같은지(어긋나면 '점 16개가 필요합니다' 만 뜬다)."""

    def test_grid_size_matches(self):
        ts = (FRONT / 'utils' / 'meshWarp.ts').read_text(encoding='utf-8')
        m = re.search(r'export const MESH_N = (\d+)', ts)
        self.assertIsNotNone(m)
        self.assertEqual(int(m.group(1)), GRID)

    def test_payload_keys_match(self):
        view = (FRONT / 'composables' / 'useEditorMeshWarp.ts').read_text(encoding='utf-8')
        self.assertIn(f'grid: {GRID}, dst, pin_edges:', view)
        self.assertIn("'mesh_warp'", view)


class BridgeMeshWarpTests(unittest.TestCase):
    def setUp(self):
        from ui.vue_bridge import VueBridge
        self.bridge = VueBridge()
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.root = Path(self._tmp.name)
        self.bridge._editor_temp_dir = str(self.root / 'editor_temp')

    def _run(self, path, params):
        with mock.patch('core.cache_cleanup.prune_editor_temp'):
            return json.loads(self.bridge._editor_process_impl(str(path), 'mesh_warp', params))

    def test_commit_keeps_size_and_alpha_channel(self):
        src = self.root / '격자.png'
        imwrite_unicode(src, _textured(channels=4))
        result = self._run(src, {'grid': 4, 'dst': _moved(6, 0.05, 0.05).tolist(), 'pin_edges': True})
        self.assertIn('path', result, result)
        self.assertEqual((result['width'], result['height']), (120, 90))
        self.assertEqual(imread_unicode(result['path'], cv2.IMREAD_UNCHANGED).shape, (90, 120, 4))

    def test_bad_payload_is_a_message(self):
        src = self.root / 'a.png'
        imwrite_unicode(src, _textured())
        self.assertEqual(self._run(src, {'grid': 4, 'dst': []}), {'error': MSG_NEED_POINTS})


if __name__ == '__main__':
    unittest.main()
