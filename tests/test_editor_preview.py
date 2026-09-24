"""에디터 실시간 프리뷰(core.editor_preview + VueBridge 프리뷰 경로) 회귀 테스트.

예전 프리뷰는 슬라이더를 멈출 때마다 원본을 통째로 다시 디코드하고, 결과를 PNG
base64(1.3~2.8MB)로 보냈다. 한글 경로에서는 cv2.imread 가 아예 실패했다.
"""
from __future__ import annotations

import base64
import json
import os
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from core.editor_preview import (
    PreviewSourceCache,
    downscale_for_preview,
    encode_preview,
    imread_unchanged,
    imwrite,
    normalize_channels,
    to_display_uint8,
)


def _write(path: Path, img: np.ndarray) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    ok, buf = cv2.imencode(path.suffix or '.png', img)
    assert ok
    buf.tofile(str(path))
    return path


def _gradient(h: int, w: int, channels: int = 3) -> np.ndarray:
    y, x = np.mgrid[0:h, 0:w]
    base = ((x + y) % 256).astype(np.uint8)
    img = np.dstack([base, base[::-1], (base // 2)])
    if channels == 4:
        img = np.dstack([img, np.full((h, w), 200, np.uint8)])
    return img


class ImageIoTests(unittest.TestCase):
    def test_unicode_path_round_trip(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / '한글 폴더' / '편집본.png'
            path.parent.mkdir()
            img = _gradient(12, 20)
            self.assertTrue(imwrite(str(path), img))
            back = imread_unchanged(str(path))
            self.assertIsNotNone(back)
            np.testing.assert_array_equal(back, img)

    def test_missing_or_empty_file_returns_none(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.assertIsNone(imread_unchanged(str(Path(tmp) / 'nope.png')))
            empty = Path(tmp) / 'empty.png'
            empty.write_bytes(b'')
            self.assertIsNone(imread_unchanged(str(empty)))

    def test_normalize_channels(self):
        self.assertEqual(normalize_channels(np.zeros((4, 5), np.uint8)).shape, (4, 5, 3))
        self.assertEqual(normalize_channels(np.zeros((4, 5, 2), np.uint8)).shape, (4, 5, 3))
        self.assertEqual(normalize_channels(np.zeros((4, 5, 4), np.uint8)).shape, (4, 5, 4))

    def test_downscale_matches_backend_formula(self):
        # SDXL 832×1216 → 긴 변 1024 (프론트 editorPreview.previewDims 와 같은 식)
        out = downscale_for_preview(np.zeros((1216, 832, 3), np.uint8), 1024)
        self.assertEqual(out.shape[:2], (1024, int(832 * (1024 / 1216))))
        small = np.zeros((500, 300, 3), np.uint8)
        self.assertIs(downscale_for_preview(small, 1024), small)


class PreviewSourceCacheTests(unittest.TestCase):
    def test_decodes_once_and_returns_independent_copies(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = _write(Path(tmp) / 'big.png', _gradient(1200, 2000))
            cache = PreviewSourceCache(1024)
            first = cache.get(str(path))
            self.assertEqual(first.shape[:2], (614, 1024))
            first[:] = 0   # 연산이 제자리에서 고쳐도 캐시는 오염되지 않아야 한다
            second = cache.get(str(path))
            self.assertEqual(cache.decode_count, 1, '같은 파일을 매 틱 다시 디코드했다')
            self.assertGreater(int(second.max()), 0)
            self.assertIsNot(first, second)

    def test_file_change_invalidates(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = _write(Path(tmp) / 'a.png', _gradient(40, 60))
            cache = PreviewSourceCache(1024)
            cache.get(str(path))
            _write(path, _gradient(50, 70))
            st = os.stat(path)
            os.utime(path, ns=(st.st_atime_ns, st.st_mtime_ns + 5_000_000_000))
            again = cache.get(str(path))
            self.assertEqual(cache.decode_count, 2)
            self.assertEqual(again.shape[:2], (50, 70))

    def test_other_file_replaces_single_entry(self):
        with tempfile.TemporaryDirectory() as tmp:
            a = _write(Path(tmp) / 'a.png', _gradient(10, 10))
            b = _write(Path(tmp) / 'b.png', _gradient(20, 20))
            cache = PreviewSourceCache(1024)
            cache.get(str(a))
            cache.get(str(b))
            cache.get(str(a))
            self.assertEqual(cache.decode_count, 3)
            self.assertIsNone(cache.get(str(Path(tmp) / 'missing.png')))


class EncodePreviewTests(unittest.TestCase):
    def test_opaque_preview_is_jpeg_and_much_smaller_than_png(self):
        img = _gradient(768, 1024)
        url = encode_preview(img)
        self.assertTrue(url.startswith('data:image/jpeg;base64,'))
        raw = base64.b64decode(url.split(',', 1)[1])
        decoded = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_UNCHANGED)
        self.assertEqual(decoded.shape, img.shape)
        ok, png = cv2.imencode('.png', img)
        self.assertLess(len(raw), len(png.tobytes()))

    def test_alpha_preview_stays_png(self):
        url = encode_preview(_gradient(16, 16, channels=4))
        self.assertTrue(url.startswith('data:image/png;base64,'))
        raw = base64.b64decode(url.split(',', 1)[1])
        decoded = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_UNCHANGED)
        self.assertEqual(decoded.shape[2], 4)

    @staticmethod
    def _decode(url: str) -> np.ndarray:
        raw = base64.b64decode(url.split(',', 1)[1])
        return cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_UNCHANGED)

    def test_sixteen_bit_preview_is_scaled_not_saturated(self):
        """예전: uint16 을 그대로 JPEG 로 넣어 OpenCV 가 255 로 잘랐다 — 프리뷰가 온통 흰색."""
        img16 = _gradient(64, 96).astype(np.uint16) * 257   # 8비트 그라디언트를 16비트로 늘린 것
        url = encode_preview(img16)
        self.assertTrue(url.startswith('data:image/jpeg;base64,'))
        decoded = self._decode(url)
        self.assertEqual(decoded.dtype, np.uint8)
        white = float((decoded >= 250).mean())
        self.assertLess(white, 0.1, f'16비트 프리뷰가 흰색으로 포화됐다 ({white:.1%})')
        # JPEG 손실 안에서 원래 8비트 그라디언트와 같아야 한다
        self.assertLess(float(np.abs(decoded.astype(int) - _gradient(64, 96).astype(int)).mean()), 4.0)

    def test_sixteen_bit_alpha_preview_is_eight_bit_png(self):
        img16 = _gradient(8, 8, channels=4).astype(np.uint16) * 257
        decoded = self._decode(encode_preview(img16))
        self.assertEqual(decoded.dtype, np.uint8)
        np.testing.assert_array_equal(decoded, _gradient(8, 8, channels=4))

    def test_to_display_uint8_conversions(self):
        u8 = np.array([[0, 128, 255]], np.uint8)
        self.assertIs(to_display_uint8(u8), u8)
        np.testing.assert_array_equal(to_display_uint8(np.array([[0, 32896, 65535]], np.uint16)), [[0, 128, 255]])
        np.testing.assert_array_equal(to_display_uint8(np.array([[0.0, 0.5, 1.0]], np.float32)), [[0, 128, 255]])
        np.testing.assert_array_equal(to_display_uint8(np.array([[-5.0, 100.0, 300.0]], np.float32)), [[0, 100, 255]])
        np.testing.assert_array_equal(to_display_uint8(np.array([[np.nan, 0.25]], np.float64)), [[0, 64]])
        np.testing.assert_array_equal(to_display_uint8(np.array([[-3, 7, 200]], np.int16)), [[0, 7, 200]])


class BridgePreviewPathTests(unittest.TestCase):
    """VueBridge._editor_process_impl 가 캐시·JPEG·한글 경로를 실제로 쓰는지."""

    def setUp(self):
        from ui.vue_bridge import VueBridge
        self.bridge = VueBridge()
        # 확정 작업은 결과를 editor_temp 에 쓰고 prune_editor_temp 를 돌린다 — 사용자의 실제
        # image_cache/editor_temp 를 건드리지 않게 테스트 전용 폴더로 돌린다.
        self._temp = tempfile.TemporaryDirectory()
        self.addCleanup(self._temp.cleanup)
        self.editor_temp = Path(self._temp.name) / 'editor_temp'
        self.bridge._editor_temp_dir = str(self.editor_temp)

    def test_preview_uses_cache_and_jpeg(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = _write(Path(tmp) / '원본 이미지.png', _gradient(1216, 832))
            params = {'preview': True, 'brightness': 10, 'contrast': 0, 'saturation': 0}
            r1 = json.loads(self.bridge._editor_process_impl(str(path), 'color_adjust', dict(params)))
            r2 = json.loads(self.bridge._editor_process_impl(str(path), 'color_adjust', dict(params)))
            self.assertTrue(r1.get('preview'), r1)
            self.assertTrue(r1['image_base64'].startswith('data:image/jpeg;base64,'))
            self.assertEqual((r1['width'], r1['height']), (int(832 * 1024 / 1216), 1024))
            self.assertEqual(r1['image_base64'], r2['image_base64'])
            self.assertEqual(self.bridge._editor_preview_cache.decode_count, 1)

    def test_preview_ops_do_not_corrupt_cached_source(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = _write(Path(tmp) / 'a.png', _gradient(64, 64))
            # 검은띠(ROI 제자리 대입) 프리뷰 뒤에도 다음 프리뷰는 원본 기준이어야 한다
            bar = {'preview': True, 'strength': 15, 'selection': {'x': 0, 'y': 0, 'w': 64, 'h': 64}}
            self.bridge._editor_process_impl(str(path), 'censor_bar', bar)
            cached = self.bridge._editor_preview_cache.get(str(path))
            self.assertGreater(int(cached.max()), 0)

    def test_commit_reads_unicode_path_and_writes_result(self):
        from unittest import mock
        with tempfile.TemporaryDirectory() as tmp:
            path = _write(Path(tmp) / '한글' / '그림.png', _gradient(20, 30))
            with mock.patch('core.cache_cleanup.prune_editor_temp') as prune:
                result = json.loads(self.bridge._editor_process_impl(str(path), 'flip_h', {}))
            self.assertIn('path', result, result)
            self.assertEqual(Path(result['path']).parent, self.editor_temp,
                             '테스트가 사용자의 image_cache/editor_temp 에 썼다')
            prune.assert_called_once()
            self.assertEqual(Path(prune.call_args.args[0]), self.editor_temp,
                             '테스트가 사용자의 editor_temp 를 정리(삭제)했다')
            out = imread_unchanged(result['path'])
            np.testing.assert_array_equal(out, cv2.flip(_gradient(20, 30), 1))
            self.assertEqual((result['width'], result['height']), (30, 20))

    def test_sixteen_bit_source_preview_is_not_white(self):
        with tempfile.TemporaryDirectory() as tmp:
            img16 = _gradient(48, 64).astype(np.uint16) * 257
            path = _write(Path(tmp) / 'deep.png', img16)
            self.assertEqual(imread_unchanged(str(path)).dtype, np.uint16)
            result = json.loads(self.bridge._editor_process_impl(str(path), 'flip_h', {'preview': True}))
            self.assertTrue(result.get('preview'), result)
            raw = base64.b64decode(result['image_base64'].split(',', 1)[1])
            decoded = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_UNCHANGED)
            self.assertLess(float((decoded >= 250).mean()), 0.1, '16비트 원본의 프리뷰가 흰색이다')


# ── EXIF 방향 · 흑백 tRNS: 에디터 배열은 브라우저가 보여 주는 그림과 같아야 한다 ──

_QUADRANTS_RGB = ((255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 255))   # 왼위·오위·왼아래·오아래


def _quadrant_image(mode: str = 'RGB', w: int = 40, h: int = 20):
    from PIL import Image
    img = Image.new(mode, (w, h))
    for index, color in enumerate(_QUADRANTS_RGB):
        x0, y0 = (index % 2) * (w // 2), (index // 2) * (h // 2)
        img.paste(color + ((255,) if mode == 'RGBA' else ()), (x0, y0, x0 + w // 2, y0 + h // 2))
    return img


def _save_oriented(path: Path, orientation: int, mode: str = 'RGB') -> Path:
    from PIL import Image
    exif = Image.Exif()
    exif[0x0112] = orientation
    path.parent.mkdir(parents=True, exist_ok=True)
    extra = {'quality': 95} if path.suffix.lower() in ('.jpg', '.jpeg') else {}
    _quadrant_image(mode).save(path, exif=exif, **extra)
    return path


def _upright_bgr(path: Path) -> np.ndarray:
    """브라우저·PIL exif_transpose 가 보여 주는 그림(BGR/BGRA)."""
    from PIL import Image, ImageOps
    with Image.open(path) as opened:
        up = ImageOps.exif_transpose(opened)
        arr = np.asarray(up.convert('RGBA' if up.mode == 'RGBA' else 'RGB'))
    return cv2.cvtColor(arr, cv2.COLOR_RGBA2BGRA if arr.shape[2] == 4 else cv2.COLOR_RGB2BGR)


class EditorOrientationDecodeTests(unittest.TestCase):
    def test_all_orientations_match_the_browser_view_for_png_and_jpeg(self):
        with tempfile.TemporaryDirectory() as tmp:
            for orientation in range(1, 9):
                with self.subTest(fmt='png', orientation=orientation):
                    path = _save_oriented(Path(tmp) / '한글 폴더' / f'o{orientation}.png', orientation, 'RGBA')
                    got = imread_unchanged(str(path))
                    self.assertEqual(got.shape[2], 4, '알파가 사라졌다')
                    np.testing.assert_array_equal(got, _upright_bgr(path))
                with self.subTest(fmt='jpeg', orientation=orientation):
                    path = _save_oriented(Path(tmp) / f'o{orientation}.jpg', orientation)
                    got = imread_unchanged(str(path)).astype(int)
                    want = _upright_bgr(path).astype(int)
                    self.assertEqual(got.shape, want.shape)
                    self.assertLess(int(np.abs(got - want).max()), 60)   # JPEG 디코더 차이만

    def test_png_exif_after_the_pixels_is_honoured(self):
        from tests.test_image_metadata import _insert_png_chunk_before_iend
        from PIL import Image
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'tail_exif.png'
            _quadrant_image().save(path)
            exif = Image.Exif()
            exif[0x0112] = 6
            raw = exif.tobytes()
            _insert_png_chunk_before_iend(path, b'eXIf', raw[6:] if raw.startswith(b'Exif\0\0') else raw)
            self.assertEqual(imread_unchanged(str(path)).shape, (40, 20, 3))
            np.testing.assert_array_equal(imread_unchanged(str(path)), _upright_bgr(path))

    def test_preview_cache_is_upright(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = _save_oriented(Path(tmp) / 'o6.jpg', 6)
            img, scale = PreviewSourceCache(max_edge=1024).get_scaled(str(path))
            self.assertEqual(img.shape[:2], (40, 20))
            self.assertEqual(scale, 1.0)

    def test_gray_png_transparency_key_becomes_alpha(self):
        from PIL import Image
        with tempfile.TemporaryDirectory() as tmp:
            cases = {
                'l8': (Image.new('L', (4, 3), 0), 0, np.uint8),
                'bit1': (Image.new('1', (4, 3), 0), 0, np.uint8),
                'l16': (Image.new('I;16', (4, 3), 0), 0, np.uint16),
            }
            for name, (image, key, dtype) in cases.items():
                with self.subTest(name=name):
                    image.putpixel((1, 1), 1 if image.mode == '1' else 200 if image.mode == 'L' else 40000)
                    path = Path(tmp) / f'{name}.png'
                    image.save(path, transparency=key)
                    got = imread_unchanged(str(path))
                    self.assertEqual((got.shape, got.dtype), ((3, 4, 4), np.dtype(dtype)))
                    opaque = np.iinfo(dtype).max
                    self.assertEqual(int(got[0, 0, 3]), 0)            # 키 색 → 투명
                    self.assertEqual(int(got[1, 1, 3]), opaque)       # 다른 색 → 불투명
                    self.assertGreater(int(got[1, 1, 0]), 0)
            plain = Path(tmp) / 'plain_gray.png'
            Image.new('L', (4, 3), 7).save(plain)
            self.assertEqual(imread_unchanged(str(plain)).ndim, 2)   # tRNS 없는 흑백은 그대로


class BridgeOrientationTests(unittest.TestCase):
    """Orientation 6(90° 시계) JPEG — 세운 좌표로 보낸 선택·마스크가 세운 그림에 적용된다."""

    def setUp(self):
        from ui.vue_bridge import VueBridge
        self.bridge = VueBridge()
        self._temp = tempfile.TemporaryDirectory()
        self.addCleanup(self._temp.cleanup)
        root = Path(self._temp.name)
        self.bridge._editor_temp_dir = str(root / 'editor_temp')
        self.path = _save_oriented(root / '폰 사진.jpg', 6)
        self.upright = _upright_bgr(self.path).astype(int)   # 20×40 (가로×세로)
        from unittest import mock
        patcher = mock.patch('core.cache_cleanup.prune_editor_temp')
        patcher.start()
        self.addCleanup(patcher.stop)

    def run_op(self, operation, params, path=None):
        return json.loads(self.bridge._editor_process_impl(str(path or self.path), operation, params))

    @staticmethod
    def mask_url(width, height, box):
        mask = np.zeros((height, width), np.uint8)
        x0, y0, x1, y1 = box
        mask[y0:y1, x0:x1] = 255
        ok, buf = cv2.imencode('.png', mask)
        return 'data:image/png;base64,' + base64.b64encode(buf.tobytes()).decode('ascii')

    def assert_close(self, got, want, tol=60):
        self.assertLess(int(np.abs(np.asarray(got, int) - np.asarray(want, int)).max()), tol, (got, want))

    def test_commit_and_preview_sizes_are_upright(self):
        flipped = self.run_op('flip_h', {})
        self.assertEqual((flipped['width'], flipped['height']), (20, 40))
        out = imread_unchanged(flipped['path']).astype(int)
        self.assert_close(out[5, 5], self.upright[5, 15])
        preview = self.run_op('color_adjust', {'preview': True, 'brightness': 0, 'contrast': 0, 'saturation': 0})
        self.assertEqual((preview['width'], preview['height']), (20, 40))

    def test_mask_censor_hits_the_painted_upright_quadrant_only(self):
        result = self.run_op('censor_bar', {'mask_base64': self.mask_url(20, 40, (0, 0, 10, 20)), 'strength': 15})
        out = imread_unchanged(result['path']).astype(int)
        self.assertEqual(out.shape[:2], (40, 20))
        self.assertLess(int(out[0:20, 0:10].max()), 5, '칠한 자리가 가려지지 않았다')
        for y, x in ((5, 15), (30, 5), (30, 15)):
            self.assert_close(out[y, x], self.upright[y, x])

    def test_crop_selection_uses_upright_coordinates(self):
        result = self.run_op('crop', {'selection': {'x': 0, 'y': 0, 'w': 10, 'h': 20}})
        self.assertEqual((result['width'], result['height']), (10, 20))
        out = imread_unchanged(result['path']).astype(int)
        self.assert_close(out[10, 5], self.upright[10, 5])

    def test_restore_from_the_rotated_original_after_an_edit(self):
        edited = self.run_op('censor_bar', {'mask_base64': self.mask_url(20, 40, (0, 0, 20, 40)), 'strength': 15})
        restored = self.run_op('restore', {'mask_base64': self.mask_url(20, 40, (0, 0, 20, 40)),
                                           'source_path': str(self.path)}, path=edited['path'])
        self.assertNotIn('error', restored)
        out = imread_unchanged(restored['path']).astype(int)
        self.assert_close(out[5, 5], self.upright[5, 5])
        self.assert_close(out[30, 15], self.upright[30, 15])


class AutoSaveRecoveryTests(unittest.TestCase):
    """복구본을 그대로 열면, 저장 뒤 복구본 정리가 편집 중인 그림을 지웠다."""

    def setUp(self):
        from unittest import mock
        from ui.vue_bridge import VueBridge
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        root = Path(self._tmp.name)
        patcher = mock.patch('tempfile.gettempdir', return_value=str(root / 'os_temp'))
        patcher.start()
        self.addCleanup(patcher.stop)
        self.autosave_dir = root / 'os_temp' / 'AIStudioPro_editor'
        self.bridge = VueBridge()
        self.editor_temp = root / 'editor_temp'
        self.bridge._editor_temp_dir = str(self.editor_temp)
        self.autosave_replies = []
        self.bridge.editorAutoSaveReady.connect(self.autosave_replies.append)

    def _request_autosave(self, *args, request_id: str = 'as-1'):
        """requestEditorAutoSave 를 부르고 띄운 워커 몫을 돌려준다(아직 돌리지 않았다).

        슬롯은 GUI 스레드에서 불린다 — 합성·쓰기는 워커에서만 해야 한다(예전 동기 editorAutoSave 는
        2048² 에 0.6초, 4K 에 1초 넘게 창을 멈췄다).
        """
        from types import SimpleNamespace
        from unittest import mock
        started = []
        with mock.patch('ui.vue_bridge.threading.Thread',
                        side_effect=lambda target, **_kw: SimpleNamespace(start=lambda: started.append(target))):
            self.bridge.requestEditorAutoSave(*args, request_id=request_id)
        self.assertEqual(len(started), 1, '자동저장이 워커를 띄우지 않았다')
        return started[0]

    def _autosave(self, *args, request_id: str = 'as-1') -> dict:
        """자동저장 한 건을 끝까지 돌리고 editorAutoSaveReady 응답을 돌려준다."""
        before = len(self.autosave_replies)
        self._request_autosave(*args, request_id=request_id)()
        self.assertEqual(len(self.autosave_replies), before + 1, '자동저장 응답이 한 번 와야 한다')
        reply = json.loads(self.autosave_replies[-1])
        self.assertEqual(reply['requestId'], request_id)
        return reply

    def test_recover_returns_a_working_copy_that_survives_clearing(self):
        autosave = _write(self.autosave_dir / '_autosave_session.png', _gradient(10, 12))
        before = autosave.read_bytes()
        result = json.loads(self.bridge.editorRecoverAutoSave())
        self.assertIn('path', result, result)
        work = Path(result['path'])
        self.assertEqual(work.parent, self.editor_temp)
        self.assertNotEqual(work.resolve(), autosave.resolve(), '복구본 파일 자체를 열면 저장 뒤 정리가 지운다')
        self.assertEqual(work.read_bytes(), before)
        # 저장 성공 → 복구본 정리. 편집 중인 작업 사본은 남아 있어야 한다
        self.assertTrue(json.loads(self.bridge.editorClearAutoSave())['cleared'])
        self.assertFalse(autosave.exists())
        self.assertTrue(work.is_file())

    def test_recover_without_autosave_reports_error(self):
        result = json.loads(self.bridge.editorRecoverAutoSave())
        self.assertIn('error', result)
        self.assertNotIn('path', result)

    @staticmethod
    def _overlay_b64(w: int, h: int, box: tuple, rgba: tuple) -> str:
        import io
        from PIL import Image
        over = Image.new('RGBA', (w, h), (0, 0, 0, 0))
        over.paste(rgba, box)
        buf = io.BytesIO()
        over.save(buf, 'PNG')
        return 'data:image/png;base64,' + base64.b64encode(buf.getvalue()).decode('ascii')

    @staticmethod
    def _rgb(path: str, xy: tuple) -> tuple:
        from PIL import Image
        with Image.open(path) as img:
            return img.convert('RGB').getpixel(xy)

    def test_autosave_includes_unmerged_drawing(self):
        """예전 자동저장은 확정 이미지 파일만 복사했다 — 그리기만 하고 크래시가 나면
        '자동저장 N분 전' 표시와 달리 복구본에 그림이 없었다."""
        orig = _write(Path(self._tmp.name) / '한글 폴더' / 'A.png', np.full((32, 32, 3), 40, np.uint8))
        before = orig.read_bytes()
        overlay = self._overlay_b64(32, 32, (0, 0, 8, 8), (255, 0, 0, 255))
        result = self._autosave(str(orig), overlay, 100.0)
        self.assertIn('path', result, result)
        self.assertTrue(result['drawing'])
        meta = json.loads((self.autosave_dir / '_autosave_session.meta.json').read_text(encoding='utf-8'))
        self.assertTrue(meta['drawing'])
        self.assertEqual(Path(meta['original']).name, 'A.png')
        recovered = json.loads(self.bridge.editorRecoverAutoSave())['path']
        self.assertEqual(self._rgb(recovered, (2, 2)), (255, 0, 0), '복구본에 병합 안 한 드로잉이 없다')
        self.assertEqual(self._rgb(recovered, (20, 20)), (40, 40, 40))
        self.assertEqual(orig.read_bytes(), before, '자동저장이 편집 중인 파일을 바꿨다')

    def test_autosave_applies_layer_opacity(self):
        orig = _write(Path(self._tmp.name) / 'B.png', np.full((16, 16, 3), 40, np.uint8))
        overlay = self._overlay_b64(16, 16, (0, 0, 8, 8), (255, 0, 0, 255))
        self.assertIn('path', self._autosave(str(orig), overlay, 50.0))
        r, g, _b = self._rgb(str(self.autosave_dir / '_autosave_session.png'), (2, 2))
        self.assertAlmostEqual(r, (255 + 40) / 2, delta=2)
        self.assertAlmostEqual(g, 40 / 2, delta=2)

    def test_autosave_without_drawing_copies_the_image(self):
        orig = _write(Path(self._tmp.name) / 'C.png', _gradient(10, 12))
        for args in ((str(orig), '', 100.0), (str(orig),)):   # 빈 레이어 / 경로만(기본값)
            result = self._autosave(*args)
            self.assertIn('path', result, result)
            self.assertFalse(result['drawing'])
            self.assertEqual((self.autosave_dir / '_autosave_session.png').read_bytes(), orig.read_bytes())

    def test_autosave_reports_a_broken_drawing_layer(self):
        orig = _write(Path(self._tmp.name) / 'D.png', _gradient(10, 12))
        result = self._autosave(str(orig), 'data:image/png;base64,bm90IGEgcG5n', 100.0)
        self.assertIn('error', result)
        self.assertNotIn('path', result, '레이어를 못 읽었는데 자동저장 성공으로 표시된다')

    def test_autosave_slot_does_not_composite_on_the_calling_thread(self):
        """슬롯(GUI 스레드)은 워커만 띄우고 돌아온다 — 합성·쓰기는 워커 몫이다."""
        orig = _write(Path(self._tmp.name) / 'E.png', _gradient(10, 12))
        overlay = self._overlay_b64(12, 10, (0, 0, 4, 4), (255, 0, 0, 255))
        work = self._request_autosave(str(orig), overlay, 100.0, request_id='as-slow')
        self.assertFalse((self.autosave_dir / '_autosave_session.png').exists(), '슬롯이 직접 복구본을 썼다')
        self.assertEqual(self.autosave_replies, [], '워커가 끝나기 전에 응답이 왔다')
        work()
        reply = json.loads(self.autosave_replies[-1])
        self.assertEqual(reply['requestId'], 'as-slow')
        self.assertTrue(reply['drawing'])
        self.assertTrue((self.autosave_dir / '_autosave_session.png').is_file())

    def test_clear_before_the_worker_runs_discards_the_write(self):
        """자동저장을 요청한 뒤 저장이 끝나 복구본을 폐기했다 — 저장 전 상태를 다시 쓰면 다음 시작 때
        이미 저장한 작업을 '복구할까요?' 하고 묻는다."""
        orig = _write(Path(self._tmp.name) / 'F.png', _gradient(10, 12))
        work = self._request_autosave(str(orig), '', 100.0, request_id='as-stale')
        self.assertTrue(json.loads(self.bridge.editorClearAutoSave())['cleared'])
        work()
        reply = json.loads(self.autosave_replies[-1])
        self.assertEqual(reply, {'requestId': 'as-stale', 'discarded': True})
        self.assertFalse((self.autosave_dir / '_autosave_session.png').exists())
        self.assertFalse((self.autosave_dir / '_autosave_session.meta.json').exists())

    def test_clear_while_writing_removes_what_was_written(self):
        from unittest import mock
        import core.editor_save as editor_save
        orig = _write(Path(self._tmp.name) / 'G.png', _gradient(10, 12))
        real = editor_save.write_autosave_snapshot

        def write_then_clear(*args, **kwargs):
            drawing = real(*args, **kwargs)
            # 워커가 그림을 쓴 직후(메타 전) GUI 스레드의 저장 완료 → 폐기
            self.assertTrue(json.loads(self.bridge.editorClearAutoSave())['cleared'])
            return drawing

        work = self._request_autosave(str(orig), '', 100.0, request_id='as-race')
        with mock.patch.object(editor_save, 'write_autosave_snapshot', side_effect=write_then_clear):
            work()
        self.assertTrue(json.loads(self.autosave_replies[-1])['discarded'])
        self.assertFalse((self.autosave_dir / '_autosave_session.png').exists(), '폐기 뒤에 복구본이 되살아났다')
        self.assertFalse((self.autosave_dir / '_autosave_session.meta.json').exists())

    def test_autosave_requested_after_a_clear_is_kept(self):
        orig = _write(Path(self._tmp.name) / 'H.png', _gradient(10, 12))
        self.bridge.editorClearAutoSave()
        result = self._autosave(str(orig), '', 100.0, request_id='as-new')
        self.assertIn('path', result, result)
        self.assertTrue(json.loads(self.bridge.editorCheckAutoSave())['exists'])


class EditorProcessDocTokenTests(unittest.TestCase):
    """editorProcess 결과가 요청한 문서의 세대(doc_gen)를 돌려주는지.

    프론트는 이 값으로 문서를 바꾼(열기·붙여넣기·닫기) 뒤 도착한 옛 문서의 결과를 버린다 —
    예전엔 job_id 만 봐서, A 의 느린 작업 결과가 새로 연 B 의 undo 히스토리로 들어갔다.
    """

    def setUp(self):
        from ui.vue_bridge import VueBridge
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        root = Path(self._tmp.name)
        self.bridge = VueBridge()
        self.bridge._editor_temp_dir = str(root / 'editor_temp')
        self.image = _write(root / '문서 A.png', _gradient(20, 30))

    def _process(self, operation: str, params: dict, impl_result: str | None = None):
        import contextlib
        from unittest import mock

        class _SyncThread:
            """작업 스레드를 그 자리에서 돌린다 — 결과 emit 까지 이 호출 안에서 끝난다."""
            def __init__(self, target=None, daemon=None, **_kw):
                self._target = target

            def start(self):
                self._target()

        emitted: list[str] = []
        self.bridge.editorResult.connect(emitted.append)
        with contextlib.ExitStack() as stack:
            stack.enter_context(mock.patch('threading.Thread', _SyncThread))
            stack.enter_context(mock.patch('core.cache_cleanup.prune_editor_temp'))
            if impl_result is not None:
                stack.enter_context(mock.patch.object(self.bridge, '_editor_process_impl', return_value=impl_result))
            started = json.loads(self.bridge.editorProcess(str(self.image), operation, json.dumps(params)))
        self.assertTrue(started.get('started'), started)
        self.assertEqual(len(emitted), 1, emitted)
        payload = json.loads(emitted[0])
        self.assertEqual(payload['job_id'], started['job_id'])
        return payload

    def test_commit_result_echoes_the_document_generation(self):
        gen = 1_790_000_000_000_123   # 프론트 initialDocGen 규모(창마다 다른 시작값)
        payload = self._process('flip_h', {'doc_gen': gen})
        self.assertIn('path', payload, payload)
        self.assertEqual(payload['doc_gen'], gen)
        self.assertEqual(payload['operation'], 'flip_h')

    def test_preview_result_keeps_its_own_tokens(self):
        payload = self._process('color_adjust', {
            'preview': True, 'preview_token': 3, 'doc_gen': 7, 'brightness': 10, 'contrast': 0, 'saturation': 0,
        })
        self.assertTrue(payload.get('preview'), payload)
        self.assertTrue(payload['preview_request'])
        self.assertEqual(payload['preview_token'], 3)
        self.assertEqual(payload['doc_gen'], 7)

    def test_error_and_mask_results_echo_the_generation(self):
        err = self._process('remove_bg', {'doc_gen': 9}, impl_result=json.dumps({'error': 'rembg 실패'}))
        self.assertEqual((err['error'], err['doc_gen']), ('rembg 실패', 9))
        mask = self._process('auto_detect', {'doc_gen': 10},
                             impl_result=json.dumps({'mask_base64': 'iVBOR', 'detect_count': 1}))
        self.assertEqual(mask['doc_gen'], 10)

    def test_non_numeric_generation_is_not_echoed(self):
        for bad in ('7', True, None, [7]):
            payload = self._process('flip_h', {'doc_gen': bad}, impl_result=json.dumps({'path': 'x.png'}))
            self.assertNotIn('doc_gen', payload, bad)
        self.assertNotIn('doc_gen', self._process('flip_h', {}, impl_result=json.dumps({'path': 'x.png'})))


if __name__ == '__main__':
    unittest.main()
