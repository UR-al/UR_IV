"""core.bg_removal — 에디터 배경 제거가 rembg 세션을 재사용하는지(감사 #60).

예전 Vue 경로는 ``rembg.remove(pil, **kw)`` 를 session 없이 불러 rembg 가 호출마다
``new_session('u2net')`` 으로 ONNX 세션을 새로 만들었다. 지금은 core.model_cache.REMBG_CACHE
(유휴 캐시)에서 lease 로 빌린다. 진짜 rembg 는 콜드 import 가 수십 초라 가짜 모듈을 쓴다.

알파 매팅 프리셋은 rembg 의 trimap 규칙(uint8 마스크 비교 → 침식)을 그대로 흉내 내 검사한다 —
옛 '품질' 프리셋(전경 270)은 전경이 늘 비어 매팅 없이 naive cutout 으로 떨어졌다.
"""
from __future__ import annotations

import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

import cv2
import numpy as np
from PIL import Image

from core import bg_removal
from core.model_cache import IdleModelCache


class _FakeSession:
    def __init__(self, name):
        self.name = name


class _FakeRembg:
    """rembg.remove/new_session 대역 — 받은 세션·입력·인자를 기록하고 마스크를 붙여 돌려준다."""

    def __init__(self, alpha: int = 128):
        self.alpha = alpha
        self.sessions_made = []
        self.calls = []

    def new_session(self, model_name, *args, **kwargs):
        session = _FakeSession(model_name)
        self.sessions_made.append(session)
        return session

    def remove(self, data, session=None, **kwargs):
        self.calls.append({'session': session, 'image': np.asarray(data).copy(),
                           'mode': data.mode, 'kwargs': dict(kwargs)})
        rgba = data.convert('RGBA')
        rgba.putalpha(self.alpha)
        return rgba

    def module(self):
        mod = types.ModuleType('rembg')
        mod.remove = self.remove
        mod.new_session = self.new_session
        return mod


def _cache():
    return IdleModelCache('rembg-test', idle_seconds=60.0, max_items=2)


def _bgr(h=6, w=8):
    img = np.zeros((h, w, 3), np.uint8)
    img[..., 0] = 10    # B
    img[..., 1] = 120   # G
    img[..., 2] = 250   # R
    return img


def _u2net_like_mask(size: int = 160, radius: float = 55.0, edge: float = 8.0) -> np.ndarray:
    """u2net 마스크 흉내 — 예측을 min-max 정규화해 uint8 로 만들므로 안쪽은 255 로 포화, 경계는 부드럽다."""
    yy, xx = np.mgrid[0:size, 0:size]
    dist = np.sqrt((yy - size / 2) ** 2 + (xx - size / 2) ** 2)
    return (np.clip((radius - dist) / edge, 0.0, 1.0) * 255).astype(np.uint8)


def _rembg_trimap(mask: np.ndarray, kwargs: dict):
    """rembg(2.0.x) bg.alpha_matting_cutout 의 trimap 규칙: uint8 비교 → 정사각형 침식.

    전경은 경계 밖을 0 으로, 배경은 1 로 보고 침식한다(scipy binary_erosion 의 border_value 와 같다).
    돌려주는 값: (확실한 전경, 확실한 배경) 불리언 마스크.
    """
    fg = mask > kwargs['alpha_matting_foreground_threshold']
    bg = mask < kwargs['alpha_matting_background_threshold']
    size = kwargs['alpha_matting_erode_size']
    if size > 0:
        kernel = np.ones((size, size), np.uint8)
        fg = cv2.erode(fg.astype(np.uint8), kernel, borderType=cv2.BORDER_CONSTANT, borderValue=0) > 0
        bg = cv2.erode(bg.astype(np.uint8), kernel, borderType=cv2.BORDER_CONSTANT, borderValue=1) > 0
    return fg, bg


class QualityPresetTests(unittest.TestCase):
    def test_preset_values(self):
        self.assertEqual(bg_removal.matting_kwargs('fast'), {})
        self.assertEqual(bg_removal.matting_kwargs('balanced'), {
            'alpha_matting': True, 'alpha_matting_foreground_threshold': 240,
            'alpha_matting_background_threshold': 10, 'alpha_matting_erode_size': 10})
        # 옛 인라인 값은 전경 270 이었다 — uint8 마스크로는 넘을 수 없어 매팅이 늘 실패했다(아래 테스트).
        self.assertEqual(bg_removal.matting_kwargs('quality'), {
            'alpha_matting': True, 'alpha_matting_foreground_threshold': 250,
            'alpha_matting_background_threshold': 20, 'alpha_matting_erode_size': 15})

    def test_matting_thresholds_fit_the_uint8_mask(self):
        """``mask > fg`` 가 참이 될 수 있어야 한다 — fg 가 255 이상이면 전경이 늘 비어 매팅이 조용히 빠진다."""
        for preset in bg_removal.QUALITIES:
            kwargs = bg_removal.matting_kwargs(preset)
            if not kwargs:
                continue
            with self.subTest(preset=preset):
                fg = kwargs['alpha_matting_foreground_threshold']
                bg = kwargs['alpha_matting_background_threshold']
                self.assertIsInstance(fg, int)
                self.assertIsInstance(bg, int)
                self.assertTrue(0 <= fg <= 254, f"전경 문턱 {fg} 는 uint8 마스크로 넘을 수 없다")
                self.assertTrue(1 <= bg <= 255, f"배경 문턱 {bg} 는 uint8 마스크로 밑돌 수 없다")
                self.assertLess(bg, fg)
                self.assertGreaterEqual(kwargs['alpha_matting_erode_size'], 0)

    def test_every_matting_preset_keeps_foreground_and_background_in_rembg_trimap(self):
        """전경·배경 중 하나라도 비면 pymatting 이 ValueError 를 내고 rembg 가 naive cutout 으로 떨어진다."""
        mask = _u2net_like_mask()
        for preset in (bg_removal.QUALITY_BALANCED, bg_removal.QUALITY_BEST):
            with self.subTest(preset=preset):
                fg, bg = _rembg_trimap(mask, bg_removal.matting_kwargs(preset))
                self.assertTrue(fg.any(), "trimap 에 확실한 전경이 없다 — 알파 매팅이 실행되지 않는다")
                self.assertTrue(bg.any(), "trimap 에 확실한 배경이 없다 — 알파 매팅이 실행되지 않는다")
                self.assertTrue((~(fg | bg)).any(), "매팅할 경계 띠가 없다")

        # 회귀 기준: 옛 'quality' 값(전경 270)은 같은 마스크에서 전경이 비었다.
        old = dict(bg_removal.matting_kwargs(bg_removal.QUALITY_BEST), alpha_matting_foreground_threshold=270)
        self.assertFalse(_rembg_trimap(mask, old)[0].any())

    def test_quality_mattes_a_wider_band_than_balanced(self):
        mask = _u2net_like_mask()
        widths = {}
        for preset in (bg_removal.QUALITY_BALANCED, bg_removal.QUALITY_BEST):
            fg, bg = _rembg_trimap(mask, bg_removal.matting_kwargs(preset))
            widths[preset] = int((~(fg | bg)).sum())
        self.assertGreater(widths[bg_removal.QUALITY_BEST], widths[bg_removal.QUALITY_BALANCED],
                           "'품질'은 '균형'보다 넓은 경계 띠를 매팅해야 한다")

    def test_trimap_rule_still_matches_installed_rembg(self):
        """위 _rembg_trimap 이 설치된 rembg 의 규칙과 같은지 소스로 확인한다(rembg 는 import 하지 않는다)."""
        import importlib.util
        spec = importlib.util.find_spec('rembg')
        if spec is None or not spec.origin:
            self.skipTest('rembg 미설치')
        bg_py = Path(spec.origin).with_name('bg.py')
        if not bg_py.is_file():
            self.skipTest(f'rembg/bg.py 없음: {bg_py}')
        source = bg_py.read_text(encoding='utf-8')
        for needle in ('mask_array > foreground_threshold', 'mask_array < background_threshold',
                       '(erode_structure_size, erode_structure_size)', 'border_value=1',
                       'naive_cutout(img, mask)'):
            self.assertIn(needle, source, f"rembg 의 trimap 규칙이 바뀌었다 — _rembg_trimap 을 다시 맞출 것: {needle}")

    def test_missing_quality_is_balanced_and_unknown_is_fast(self):
        self.assertEqual(bg_removal.normalize_quality(None), 'balanced')
        self.assertEqual(bg_removal.normalize_quality('  '), 'balanced')
        self.assertEqual(bg_removal.normalize_quality('Quality'), 'quality')
        self.assertEqual(bg_removal.normalize_quality('ultra'), 'fast')
        self.assertEqual(bg_removal.matting_kwargs('ultra'), {})

    def test_matting_kwargs_returns_a_copy(self):
        bg_removal.matting_kwargs('balanced')['alpha_matting_erode_size'] = 99
        self.assertEqual(bg_removal.matting_kwargs('balanced')['alpha_matting_erode_size'], 10)


class InputConversionTests(unittest.TestCase):
    def test_bgr_bgra_gray_and_16bit_become_rgb_uint8(self):
        rgb = bg_removal.to_rgb(_bgr())
        self.assertEqual(rgb.shape, (6, 8, 3))
        np.testing.assert_array_equal(rgb[0, 0], [250, 120, 10])

        bgra = np.dstack([_bgr(), np.full((6, 8), 7, np.uint8)])
        np.testing.assert_array_equal(bg_removal.to_rgb(bgra)[0, 0], [250, 120, 10])

        gray = np.full((4, 5), 33, np.uint8)
        np.testing.assert_array_equal(bg_removal.to_rgb(gray)[0, 0], [33, 33, 33])

        deep = _bgr().astype(np.uint16) * 257
        out = bg_removal.to_rgb(deep)
        self.assertEqual(out.dtype, np.uint8)
        np.testing.assert_array_equal(out[0, 0], [250, 120, 10])


class RemoveBackgroundTests(unittest.TestCase):
    def setUp(self):
        self.fake = _FakeRembg()
        self.cache = _cache()

    def _run(self, img, quality='balanced', **kwargs):
        return bg_removal.remove_background(
            img, quality, cache=self.cache, session_loader=self.fake.new_session,
            remove_fn=self.fake.remove, **kwargs)

    def test_session_is_created_once_and_reused(self):
        self._run(_bgr())
        self._run(_bgr(), 'fast')
        self._run(_bgr(), 'quality', refine_fn=lambda img: img)
        self.assertEqual(len(self.fake.sessions_made), 1, "클릭마다 세션을 새로 만들면 안 된다")
        sessions = {id(call['session']) for call in self.fake.calls}
        self.assertEqual(sessions, {id(self.fake.sessions_made[0])})
        self.assertEqual(self.fake.sessions_made[0].name, 'u2net')
        self.assertEqual(self.cache.keys(), ('u2net',))

    def test_rembg_receives_rgb_and_preset_kwargs(self):
        self._run(_bgr(), 'balanced')
        call = self.fake.calls[-1]
        self.assertEqual(call['mode'], 'RGB')
        np.testing.assert_array_equal(call['image'][0, 0], [250, 120, 10])
        self.assertTrue(call['kwargs']['alpha_matting'])
        self.assertEqual(call['kwargs']['alpha_matting_foreground_threshold'], 240)

    def test_result_is_bgra_with_rembg_alpha(self):
        out = self._run(_bgr(), 'fast')
        self.assertEqual(out.shape, (6, 8, 4))
        self.assertEqual(out.dtype, np.uint8)
        np.testing.assert_array_equal(out[0, 0], [10, 120, 250, 128])

    def test_transparent_input_is_segmented_again(self):
        bgra = np.dstack([_bgr(), np.zeros((6, 8), np.uint8)])
        out = self._run(bgra, 'fast')
        np.testing.assert_array_equal(out[0, 0], [10, 120, 250, 128])

    def test_quality_refines_edges_and_survives_refine_failure(self):
        seen = []

        def refine(img):
            seen.append(img.shape)
            img = img.copy()
            img[..., 3] = 255
            return img

        out = self._run(_bgr(), 'quality', refine_fn=refine)
        self.assertEqual(seen, [(6, 8, 4)])
        self.assertEqual(int(out[0, 0, 3]), 255)

        def broken(_img):
            raise RuntimeError('no ximgproc')

        with self.assertLogs('core.bg_removal', level='WARNING'):
            out = self._run(_bgr(), 'quality', refine_fn=broken)
        self.assertEqual(int(out[0, 0, 3]), 128, "다듬기가 실패하면 다듬지 않은 결과를 쓴다")

    def test_balanced_and_fast_do_not_refine(self):
        refine = mock.Mock(side_effect=AssertionError('refine 호출 금지'))
        self._run(_bgr(), 'balanced', refine_fn=refine)
        self._run(_bgr(), 'fast', refine_fn=refine)
        refine.assert_not_called()

    def test_session_is_leased_during_inference_and_clear_waits(self):
        observed = {}

        def remove(data, session=None, **kwargs):
            observed['leases'] = self.cache.stats()['keys'][0]['leases']
            observed['cleared_now'] = self.cache.clear()   # 수동 언로드가 추론 중에 온다
            observed['still_cached'] = len(self.cache)
            return self.fake.remove(data, session=session, **kwargs)

        bg_removal.remove_background(_bgr(), 'fast', cache=self.cache,
                                     session_loader=self.fake.new_session, remove_fn=remove)
        self.assertEqual(observed, {'leases': 1, 'cleared_now': 0, 'still_cached': 1},
                         "추론 중인 세션을 꺼내 가면 안 된다(반납 예약)")
        self.assertEqual(len(self.cache), 0, "lease 가 끝나면 예약된 반납이 일어난다")

    def test_remove_failure_propagates_and_releases_the_lease(self):
        def remove(*_a, **_k):
            raise RuntimeError('onnx boom')

        with self.assertRaisesRegex(RuntimeError, 'onnx boom'):
            bg_removal.remove_background(_bgr(), 'fast', cache=self.cache,
                                         session_loader=self.fake.new_session, remove_fn=remove)
        self.assertEqual(self.cache.stats()['keys'][0]['leases'], 0)

    def test_non_rgba_result_is_rejected(self):
        with self.assertRaises(ValueError):
            bg_removal.remove_background(
                _bgr(), 'fast', cache=self.cache, session_loader=self.fake.new_session,
                remove_fn=lambda data, session=None, **k: np.zeros((6, 8), np.uint8))

    def test_default_loader_and_remove_come_from_rembg(self):
        fake = _FakeRembg(alpha=90)
        cache = _cache()
        with mock.patch.dict(sys.modules, {'rembg': fake.module()}):
            out1 = bg_removal.remove_background(_bgr(), 'fast', cache=cache)
            out2 = bg_removal.remove_background(_bgr(), 'fast', cache=cache)
        self.assertEqual([s.name for s in fake.sessions_made], ['u2net'])
        self.assertIs(fake.calls[0]['session'], fake.calls[1]['session'])
        self.assertEqual(int(out1[0, 0, 3]), 90)
        np.testing.assert_array_equal(out1, out2)


class BridgeRemoveBgTests(unittest.TestCase):
    """VueBridge 'remove_bg' 가 core.bg_removal 과 전역 REMBG_CACHE 로 세션을 재사용하는지."""

    def setUp(self):
        from ui.vue_bridge import VueBridge
        self.bridge = VueBridge()
        self._temp = tempfile.TemporaryDirectory()
        self.addCleanup(self._temp.cleanup)
        root = Path(self._temp.name)
        self.bridge._editor_temp_dir = str(root / 'editor_temp')
        self.path = root / '배경 제거.png'
        ok, buf = cv2.imencode('.png', _bgr(12, 16))
        self.assertTrue(ok)
        buf.tofile(str(self.path))
        for target in ('core.cache_cleanup.prune_editor_temp',):
            patcher = mock.patch(target)
            patcher.start()
            self.addCleanup(patcher.stop)

    def test_two_clicks_share_one_session_and_keep_alpha(self):
        fake = _FakeRembg(alpha=77)
        cache = _cache()
        with mock.patch.dict(sys.modules, {'rembg': fake.module()}), \
                mock.patch('core.model_cache.REMBG_CACHE', cache):
            first = json.loads(self.bridge._editor_process_impl(str(self.path), 'remove_bg', {'quality': 'fast'}))
            second = json.loads(self.bridge._editor_process_impl(str(self.path), 'remove_bg', {'quality': 'balanced'}))
        self.assertIn('path', first, first)
        self.assertIn('path', second, second)
        self.assertEqual(len(fake.sessions_made), 1)
        self.assertEqual(len(fake.calls), 2)
        self.assertTrue(fake.calls[1]['kwargs'].get('alpha_matting'))
        from core.editor_preview import imread_unchanged
        saved = imread_unchanged(first['path'])
        self.assertEqual(saved.shape[2], 4, "배경 제거 결과가 알파를 잃었다")
        self.assertEqual(int(saved[0, 0, 3]), 77)

    def test_failure_is_reported_as_editor_error(self):
        cache = _cache()

        def boom(_name):
            raise RuntimeError('model download failed')

        with mock.patch('core.model_cache.REMBG_CACHE', cache), \
                mock.patch.object(bg_removal, 'load_session', boom), \
                mock.patch.dict(sys.modules, {'rembg': _FakeRembg().module()}):
            result = json.loads(self.bridge._editor_process_impl(str(self.path), 'remove_bg', {}))
        self.assertEqual(result, {'error': '배경 제거 실패: model download failed'})

    def test_bridge_no_longer_calls_rembg_without_a_session(self):
        source = (Path(__file__).resolve().parents[1] / 'ui' / 'vue_bridge.py').read_text(encoding='utf-8')
        self.assertNotIn('from rembg import', source, "rembg 는 core/bg_removal.py 에서만 부른다")
        self.assertIn('remove_background', source)


if __name__ == '__main__':
    unittest.main()
