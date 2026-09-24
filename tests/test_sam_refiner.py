"""SAM 정밀화 — 정직한 폴백·세션당 1회 경고·SAM3 exclude 인코딩 재사용.

GPU/실제 모델/네트워크 없이 돈다: torch·sam3·mobile_sam은 sys.modules의 가짜 모듈로 대체한다.

회귀 방지 대상
  · auto가 패키지 없는 MobileSAM을 골라 bbox를 '✓ Refined'로 보고하던 문제 (감사 #25)
  · auto가 패키지 없다고 SAM3(3.45GB)로 몰래 승격하는 퇴행 (b83190abe 결정)
  · SAM3 exclude 패스가 main pass에서 인코딩한 같은 이미지를 다시 set_image 하던 낭비 (#127)
"""
import base64
import contextlib
import gc
import io
import json
import os
import sys
import tempfile
import types
import unittest
import weakref
from types import SimpleNamespace
from unittest import mock

import cv2
import numpy as np
from PIL import Image

from core import model_cache, sam_refiner
from core.model_cache import IdleModelCache
from core.sam_refiner import (SamResolution, SamUnavailableError, _best_box_match, _segment_masks,
                              find_sam_model, missing_sam_runtime, notify_sam_unavailable,
                              refine_boxes_with_sam, resolve_sam_model)


def _fake_torch(cuda=False):
    return SimpleNamespace(
        cuda=SimpleNamespace(is_available=lambda: cuda),
        autocast=lambda **_kw: contextlib.nullcontext(),
        bfloat16='bfloat16',
    )


def _strict_cp949_stdout(test):
    """한국어 Windows에서 파이프·파일로 리다이렉트된 stdout과 같은 strict cp949 스트림.

    /verify·/ship은 run_tests.py를 파이프로 돌린다. PYTHONIOENCODING 같은 환경과 무관하게
    '진단 로그가 인코딩 때문에 흐름을 바꾸지 않는다'를 고정한다.
    """
    console = io.TextIOWrapper(io.BytesIO(), encoding='cp949', errors='strict', newline='\n')
    redirect = contextlib.redirect_stdout(console)
    redirect.__enter__()
    test.addCleanup(redirect.__exit__, None, None, None)
    return console


def _console_text(console) -> str:
    console.flush()
    return console.buffer.getvalue().decode('cp949')


def _models_dir(test, *names):
    temp = tempfile.TemporaryDirectory()
    test.addCleanup(temp.cleanup)
    for name in names:
        with open(os.path.join(temp.name, name), 'wb') as handle:
            handle.write(b'weights')
    return temp.name


def _only_missing(*sam_types):
    table = {'mobile_sam': 'mobile_sam', 'sam': 'segment_anything', 'fast_sam': 'ultralytics', 'sam3': 'sam3'}
    return lambda sam_type: table[sam_type] if sam_type in sam_types else None


class ResolveSamModelTests(unittest.TestCase):
    def setUp(self):
        sam_refiner._reset_unavailable_notices()
        self.addCleanup(sam_refiner._reset_unavailable_notices)

    def test_auto_does_not_promote_to_sam3_when_mobile_sam_package_is_missing(self):
        models = _models_dir(self, 'mobile_sam.pt', 'sam3.pt', 'penis.pt')
        with mock.patch.object(sam_refiner, 'missing_sam_runtime', _only_missing('mobile_sam')):
            resolution = resolve_sam_model(models, 'auto')
            legacy = find_sam_model(models, 'auto')
        self.assertEqual(resolution, SamResolution(None, None, 'mobile_sam', 'mobile_sam'))
        self.assertEqual(legacy, (None, None), "반환형은 그대로, 사용 불가면 (None, None)")

    def test_auto_uses_mobile_sam_when_package_present(self):
        models = _models_dir(self, 'mobile_sam.pt', 'sam3.pt')
        with mock.patch.object(sam_refiner, 'missing_sam_runtime', _only_missing()):
            resolution = resolve_sam_model(models, 'auto')
        self.assertEqual(resolution.sam_type, 'mobile_sam')
        self.assertEqual(os.path.basename(resolution.path), 'mobile_sam.pt')
        self.assertIsNone(resolution.missing_package)

    def test_auto_keeps_documented_sam3_fallback_only_when_no_mobile_file(self):
        models = _models_dir(self, 'sam3.pt')
        with mock.patch.object(sam_refiner, 'missing_sam_runtime', _only_missing('mobile_sam')):
            resolution = resolve_sam_model(models, 'auto')
        self.assertEqual(resolution.sam_type, 'sam3')

    def test_explicit_choices_and_off(self):
        models = _models_dir(self, 'mobile_sam.pt', 'sam3.pt')
        with mock.patch.object(sam_refiner, 'missing_sam_runtime', _only_missing('mobile_sam')):
            self.assertEqual(resolve_sam_model(models, 'sam3').sam_type, 'sam3')
            self.assertEqual(resolve_sam_model(models, 'mobile_sam'),
                             SamResolution(None, None, 'mobile_sam', 'mobile_sam'))
            self.assertEqual(resolve_sam_model(models, 'off'), SamResolution(None, None))
        self.assertEqual(resolve_sam_model(os.path.join(models, 'missing-dir'), 'auto'),
                         SamResolution(None, None))

    def test_yolo_detectors_are_never_picked_as_sam(self):
        models = _models_dir(self, 'nsfw_sam_detect.pt')
        with mock.patch.object(sam_refiner, 'missing_sam_runtime', _only_missing()):
            self.assertEqual(resolve_sam_model(models, 'auto'), SamResolution(None, None))

    def test_missing_sam_runtime_uses_find_spec_without_importing(self):
        with mock.patch.object(sam_refiner.importlib.util, 'find_spec',
                               side_effect=lambda name: None if name == 'mobile_sam' else object()) as spec:
            self.assertEqual(missing_sam_runtime('mobile_sam'), 'mobile_sam')
            self.assertIsNone(missing_sam_runtime('sam3'))
            self.assertIsNone(missing_sam_runtime('unknown'))
        spec.assert_any_call('mobile_sam')
        with mock.patch.object(sam_refiner.importlib.util, 'find_spec', side_effect=ValueError('no spec')):
            self.assertEqual(missing_sam_runtime('sam'), 'segment_anything')


class NotifyOnceTests(unittest.TestCase):
    def setUp(self):
        sam_refiner._reset_unavailable_notices()
        self.addCleanup(sam_refiner._reset_unavailable_notices)

    def test_toast_once_per_session_per_package(self):
        calls = []
        notify = lambda level, message: calls.append((level, message))
        self.assertTrue(notify_sam_unavailable(notify, 'mobile_sam', 'mobile_sam'))
        self.assertFalse(notify_sam_unavailable(notify, 'mobile_sam', 'mobile_sam'))
        self.assertTrue(notify_sam_unavailable(notify, 'sam3', 'sam3'))
        self.assertEqual([level for level, _ in calls], ['warning', 'warning'])
        self.assertIn('MobileSAM', calls[0][1])
        self.assertIn('mobile_sam', calls[0][1])

    def test_without_callback_does_not_consume_the_session_notice(self):
        self.assertFalse(notify_sam_unavailable(None, 'mobile_sam', 'mobile_sam'))
        calls = []
        self.assertTrue(notify_sam_unavailable(lambda *a: calls.append(a), 'mobile_sam', 'mobile_sam'))
        self.assertEqual(len(calls), 1)


class RefineHonestFallbackTests(unittest.TestCase):
    """SAM이 돌지 못하면 빈 마스크 → 호출자(vue_bridge)가 자기 YOLO 마스크를 쓰고 'Refined'로 보고하지 않는다."""

    def setUp(self):
        sam_refiner._reset_unavailable_notices()
        self.addCleanup(sam_refiner._reset_unavailable_notices)
        self.image = np.zeros((40, 50, 3), dtype=np.uint8)
        self.boxes = [(5, 6, 25, 30)]
        self.models = _models_dir(self, 'mobile_sam.pt')
        self.path = os.path.join(self.models, 'mobile_sam.pt')
        self.notices = []
        self.release = mock.patch.object(model_cache, 'release_torch_memory')
        self.released = self.release.start()
        self.addCleanup(self.release.stop)
        # 회귀: 이 클래스의 로그 줄에 '—'가 들어가 파이프 stdout(cp949)에서 UnicodeEncodeError
        self.console = _strict_cp949_stdout(self)

    def notify(self, level, message):
        self.notices.append((level, message))

    def refine(self, **kwargs):
        options = dict(sam_model_path=self.path, sam_type='mobile_sam', notify=self.notify)
        options.update(kwargs)
        return refine_boxes_with_sam(self.image, self.boxes, self.models, **options)

    def test_missing_mobile_sam_package_returns_empty_mask_and_warns_once(self):
        with mock.patch.dict(sys.modules, {'mobile_sam': None, 'torch': _fake_torch()}):
            first = self.refine()
            second = self.refine()
        self.assertFalse(first.any(), "bbox를 채워 정상값처럼 돌려주면 '✓ Refined' 거짓 보고가 된다")
        self.assertFalse(second.any())
        self.assertEqual(first.shape, (40, 50))
        self.assertEqual([level for level, _ in self.notices], ['warning'], "경고는 세션당 1회")
        self.released.assert_not_called()   # 모델을 올리지도 못했다 — gc/empty_cache 비용 없음

    def test_segment_anything_is_not_used_as_a_fake_mobile_sam_fallback(self):
        registry = mock.MagicMock()
        fake_sa = SimpleNamespace(sam_model_registry=registry, SamPredictor=mock.Mock())
        with mock.patch.dict(sys.modules, {'mobile_sam': None, 'segment_anything': fake_sa,
                                           'torch': _fake_torch()}):
            mask = self.refine()
        self.assertFalse(mask.any())
        registry.__getitem__.assert_not_called()

    def test_auto_detection_inside_refine_reports_missing_runtime(self):
        with mock.patch.object(sam_refiner, 'missing_sam_runtime', _only_missing('mobile_sam')):
            mask = refine_boxes_with_sam(self.image, self.boxes, self.models, notify=self.notify)
        self.assertFalse(mask.any())
        self.assertEqual(len(self.notices), 1)

    def test_no_model_returns_empty_mask(self):
        mask = refine_boxes_with_sam(self.image, self.boxes, self.models,
                                     sam_model_path=os.path.join(self.models, 'nope.pt'),
                                     sam_type='mobile_sam', notify=self.notify)
        self.assertFalse(mask.any())
        self.assertEqual(self.notices, [])
        self.assertIn('[SAM] No SAM model found', _console_text(self.console))

    def test_mobile_sam_runs_and_releases_cuda_cache_after_call(self):
        predicted = np.zeros((3, 40, 50), dtype=bool)
        predicted[1, 10:20, 8:18] = True

        class Predictor:
            def __init__(self, sam):
                self.sam = sam

            def set_image(self, rgb):
                self.rgb = rgb

            def predict(self, box, multimask_output):
                return predicted, np.array([0.1, 0.9, 0.2]), None

        registry = {'vit_t': lambda checkpoint: SimpleNamespace(to=lambda device: None, checkpoint=checkpoint)}
        fake_mobile = SimpleNamespace(sam_model_registry=registry, SamPredictor=Predictor)
        with mock.patch.dict(sys.modules, {'mobile_sam': fake_mobile, 'torch': _fake_torch()}):
            mask = self.refine()
        self.assertEqual(int((mask > 0).sum()), 100)
        self.assertTrue(mask[15, 10])
        self.released.assert_called_once_with()
        self.assertEqual(self.notices, [])

    def test_runtime_error_reports_and_returns_empty_mask(self):
        def boom(checkpoint):
            raise RuntimeError('corrupt checkpoint')

        fake_mobile = SimpleNamespace(sam_model_registry={'vit_t': boom}, SamPredictor=object)
        with mock.patch.dict(sys.modules, {'mobile_sam': fake_mobile, 'torch': _fake_torch()}), \
                contextlib.redirect_stderr(io.StringIO()):
            mask = self.refine()
        self.assertFalse(mask.any())
        self.assertEqual(self.notices[0][0], 'error')
        self.assertIn('corrupt checkpoint', self.notices[0][1])
        self.released.assert_called_once_with()
        self.assertIn('[SAM] Error: corrupt checkpoint', _console_text(self.console))

    def test_unencodable_error_text_still_reports_and_releases(self):
        """except 블록 안의 로그가 터지면 알림·VRAM 반납을 건너뛰고 엉뚱한 cp949 오류가
        호출자(vue_bridge)까지 새어 'SAM 정밀화 실패: cp949 codec…' 토스트가 뜨던 문제."""
        def boom(checkpoint):
            raise RuntimeError('CUDA out of memory — ✓ tried to allocate')

        fake_mobile = SimpleNamespace(sam_model_registry={'vit_t': boom}, SamPredictor=object)
        with mock.patch.dict(sys.modules, {'mobile_sam': fake_mobile, 'torch': _fake_torch()}), \
                contextlib.redirect_stderr(io.StringIO()):
            mask = self.refine()                    # 예외가 새어 나오면 안 된다
        self.assertFalse(mask.any())
        self.assertEqual([level for level, _ in self.notices], ['error'])
        self.assertIn('CUDA out of memory', self.notices[0][1])
        self.assertNotIn('cp949', self.notices[0][1])
        self.released.assert_called_once_with()
        self.assertIn('CUDA out of memory \\u2014 \\u2713', _console_text(self.console))


class FastSamMatchTests(unittest.TestCase):
    """벡터화한 IoU 매칭이 예전(박스×마스크마다 전체 해상도 resize/zeros) 계산과 같은지."""

    @staticmethod
    def _old_best(all_masks, box, w, h):
        bx1, by1, bx2, by2 = box
        best_iou, best_mask = 0, None
        for seg in all_masks:
            seg_resized = cv2.resize(seg.astype(np.float32), (w, h))
            box_mask = np.zeros((h, w), dtype=np.float32)
            box_mask[by1:by2, bx1:bx2] = 1.0
            inter = (seg_resized > 0.5) & (box_mask > 0.5)
            union = (seg_resized > 0.5) | (box_mask > 0.5)
            iou = inter.sum() / max(union.sum(), 1)
            if iou > best_iou:
                best_iou, best_mask = iou, seg_resized > 0.5
        return best_mask, best_iou

    def test_matches_previous_iou_selection(self):
        rng = np.random.default_rng(7)
        w, h = 64, 48
        raw = (rng.random((5, 24, 32)) > 0.6).astype(np.float32)    # 저해상도 → resize 경로
        segments = _segment_masks(raw, w, h)
        for box in ((0, 0, 64, 48), (10, 5, 30, 40), (50, 40, 64, 48), (3, 3, 4, 4)):
            with self.subTest(box=box):
                old_mask, old_iou = self._old_best(raw, box, w, h)
                new_mask, new_iou = _best_box_match(segments, box)
                self.assertAlmostEqual(float(new_iou), float(old_iou))
                if old_mask is None:
                    self.assertIsNone(new_mask)
                else:
                    np.testing.assert_array_equal(new_mask, old_mask)

    def test_segments_resize_once_and_keep_area(self):
        raw = np.zeros((2, 10, 10), dtype=np.float32)
        raw[0, :5, :5] = 1
        segments = _segment_masks(raw, 10, 10)
        self.assertEqual([area for _m, area in segments], [25, 0])
        self.assertEqual(segments[0][0].dtype, bool)


class _FakeSam3Processor:
    """sam3 Sam3Processor의 상태 계약만 흉내 낸다 (set_image → state, set_text_prompt → masks 덮어쓰기)."""

    def __init__(self, model, device='cuda'):
        self.model = model
        self.device = device
        self.set_image_calls = 0
        self.prompts = []
        self.threshold = None

    def set_confidence_threshold(self, threshold, state=None):
        self.threshold = threshold

    def set_image(self, image, state=None):
        self.set_image_calls += 1
        return {'backbone_out': {'image': image.size}, 'original_height': image.size[1],
                'original_width': image.size[0]}

    def set_text_prompt(self, prompt, state):
        if 'backbone_out' not in state:
            raise ValueError('set_image first')
        self.prompts.append(prompt)
        state['backbone_out']['language'] = prompt
        state['masks'] = self.model.masks_by_prompt.get(prompt, np.zeros((0, 60, 80), dtype=bool))
        return state


class Sam3ExcludeReuseTests(unittest.TestCase):
    def setUp(self):
        sam_refiner._reset_unavailable_notices()
        self.addCleanup(sam_refiner._reset_unavailable_notices)
        self.h, self.w = 60, 80
        body = np.zeros((1, self.h, self.w), dtype=bool)
        body[0, 10:50, 10:70] = True
        face = np.zeros((1, self.h, self.w), dtype=bool)
        face[0, 12:22, 30:45] = True
        hand = np.zeros((1, self.h, self.w), dtype=bool)
        hand[0, 40:48, 12:20] = True
        self.model = SimpleNamespace(masks_by_prompt={'person': body, 'face': face, 'hand': hand})
        self.processors = []

        def build_sam3_image_model(**kwargs):
            self.build_kwargs = kwargs
            return self.model

        def processor(model, device='cuda'):
            created = _FakeSam3Processor(model, device)
            self.processors.append(created)
            return created

        self.fake_modules = {
            'sam3': types.ModuleType('sam3'),
            'sam3.model_builder': SimpleNamespace(build_sam3_image_model=build_sam3_image_model),
            'sam3.model': types.ModuleType('sam3.model'),
            'sam3.model.sam3_image_processor': SimpleNamespace(Sam3Processor=processor),
        }
        self.cache = IdleModelCache('sam3-test', idle_seconds=90.0, max_items=1)
        patches = [
            mock.patch.object(model_cache, 'SAM3_CACHE', self.cache),
            mock.patch.object(sam_refiner, '_find_sam3_bpe_vocab', return_value='bpe.txt.gz'),
        ]
        for patcher in patches:
            patcher.start()
            self.addCleanup(patcher.stop)
        self.image = np.full((self.h, self.w, 3), 120, dtype=np.uint8)
        self.notices = []

    def refine(self, cuda=False, exclude='face, hand', modules=None):
        with mock.patch.dict(sys.modules, {**(modules or self.fake_modules), 'torch': _fake_torch(cuda)}):
            return refine_boxes_with_sam(self.image, [(8, 8, 72, 52)], '', sam_model_path=__file__,
                                         sam_type='sam3', text_prompt='person', exclude_prompt=exclude,
                                         notify=lambda *a: self.notices.append(a))

    def test_exclude_reuses_main_pass_image_encoding(self):
        for cuda in (False, True):
            with self.subTest(cuda=cuda):
                self.cache.clear()
                self.processors.clear()
                mask = self.refine(cuda=cuda)
                processor = self.processors[-1]
                self.assertEqual(processor.set_image_calls, 1, "exclude 패스가 같은 이미지를 다시 인코딩했다")
                self.assertEqual(processor.prompts, ['person', 'face', 'hand'])
                self.assertEqual(processor.threshold, 0.5)
                self.assertTrue(mask[30, 55], "main 마스크는 남는다")
                self.assertFalse(mask[16, 36], "face 영역은 빠진다")
                self.assertFalse(mask[44, 15], "hand 영역은 빠진다")
                self.assertEqual(self.build_kwargs['device'], 'cuda' if cuda else 'cpu')

    def test_main_mask_is_unchanged_without_exclude(self):
        mask = self.refine(exclude=None)
        self.assertEqual(self.processors[-1].set_image_calls, 1)
        self.assertEqual(self.processors[-1].prompts, ['person'])
        self.assertTrue(mask[16, 36])

    def test_bundle_is_cached_between_clicks(self):
        self.refine()
        self.refine()
        self.assertEqual(len(self.processors), 1, "연속 클릭은 3.45GB 번들을 다시 올리지 않는다")
        self.assertEqual(self.processors[0].set_image_calls, 2)

    def test_unimportable_sam3_warns_once_and_returns_empty_mask(self):
        broken = {**self.fake_modules, 'sam3.model_builder': None}
        first = self.refine(modules=broken)
        second = self.refine(modules=broken)
        self.assertFalse(first.any())
        self.assertFalse(second.any())
        self.assertEqual([level for level, _ in self.notices], ['warning'])
        self.assertIn('SAM3', self.notices[0][1])

    def test_missing_bpe_vocab_is_reported_not_bbox_filled(self):
        with mock.patch.object(sam_refiner, '_find_sam3_bpe_vocab', return_value=''):
            mask = self.refine()
        self.assertFalse(mask.any())
        self.assertIn('bpe_simple_vocab_16e6.txt.gz', self.notices[0][1])

    def test_sam3_does_not_call_per_call_release(self):
        with mock.patch.object(model_cache, 'release_torch_memory') as release:
            self.refine()
        release.assert_not_called()   # SAM3는 유휴 캐시가 관리한다 — 클릭마다 gc/empty_cache 금지


class _WeakModel:
    """weakref 가능한 가짜 SAM3 모델 (SimpleNamespace는 weakref가 안 된다)."""

    def __init__(self, masks_by_prompt):
        self.masks_by_prompt = masks_by_prompt


class Sam3LeaseReleaseTests(unittest.TestCase):
    """회귀: 생성 시작(release_for_generation)·reaper가 추론 중인 SAM3 번들을 빼 가면
    empty_cache가 작업의 지역 참조 때문에 헛돌고, 작업이 끝난 뒤에는 다시 부를 곳이 없어
    ~3.4GB가 PyTorch reserved로 남았다 (ANIMA+SAM3+Forge 16GB OOM 시나리오).

    이제 번들은 추론이 끝나는 즉시, 모델 참조가 모두 사라진 뒤에 반납돼야 한다.
    """

    def setUp(self):
        sam_refiner._reset_unavailable_notices()
        self.addCleanup(sam_refiner._reset_unavailable_notices)
        self.console = _strict_cp949_stdout(self)
        self.h, self.w = 60, 80
        body = np.zeros((1, self.h, self.w), dtype=bool)
        body[0, 10:50, 10:70] = True
        self.masks = {'person': body}
        self.refs = []
        self.after_calls = []
        self.release_calls = []
        self.on_prompt = None
        self.loads = 0
        test = self

        class Processor(_FakeSam3Processor):
            def set_text_prompt(self, prompt, state):
                if test.on_prompt is not None:
                    test.on_prompt(prompt)
                return super().set_text_prompt(prompt, state)

        def build_sam3_image_model(**_kwargs):
            test.loads += 1
            model = _WeakModel(test.masks)
            test.refs.append(weakref.ref(model))
            return model

        def processor(model, device='cuda'):
            created = Processor(model, device)
            test.refs.append(weakref.ref(created))
            return created

        self.modules = {
            'sam3': types.ModuleType('sam3'),
            'sam3.model_builder': SimpleNamespace(build_sam3_image_model=build_sam3_image_model),
            'sam3.model': types.ModuleType('sam3.model'),
            'sam3.model.sam3_image_processor': SimpleNamespace(Sam3Processor=processor),
            'torch': _fake_torch(False),
        }
        self.cache = IdleModelCache('sam3-lease', idle_seconds=90.0, max_items=1,
                                    after_evict=lambda: self.after_calls.append(self._bundle_dead()))

        def fake_release_torch_memory():
            gc.collect()                          # release_torch_memory와 같은 순서(gc → empty_cache)
            self.release_calls.append(self._bundle_dead())

        patches = [
            mock.patch.object(model_cache, 'SAM3_CACHE', self.cache),
            mock.patch.object(model_cache, 'release_torch_memory', side_effect=fake_release_torch_memory),
            mock.patch.object(sam_refiner, '_find_sam3_bpe_vocab', return_value='bpe.txt.gz'),
        ]
        for patcher in patches:
            patcher.start()
            self.addCleanup(patcher.stop)
        self.image = np.full((self.h, self.w, 3), 120, dtype=np.uint8)
        self.notices = []

    def _bundle_dead(self):
        return bool(self.refs) and all(ref() is None for ref in self.refs)

    def refine(self, text_prompt='person', exclude=None):
        with mock.patch.dict(sys.modules, self.modules):
            return refine_boxes_with_sam(self.image, [(8, 8, 72, 52)], '', sam_model_path=__file__,
                                         sam_type='sam3', text_prompt=text_prompt, exclude_prompt=exclude,
                                         notify=lambda *a: self.notices.append(a))

    def test_generation_start_mid_inference_releases_bundle_after_the_click(self):
        # 추론 도중 다른 스레드에서 생성이 시작된 상황 — 공유 코디네이터의 실제 훅 경로
        self.on_prompt = lambda _prompt: model_cache.release_for_generation()
        mask = self.refine()
        self.assertTrue(mask[30, 55], "진행 중이던 클릭은 같은 번들로 끝까지 돈다")
        self.assertEqual(self.after_calls, [True],
                         "추론이 끝난 뒤 한 번, 번들이 이미 죽은 상태에서 empty_cache")
        self.assertEqual(len(self.cache), 0)
        self.assertEqual(self.release_calls, [], "성공한 SAM3 클릭은 호출마다 gc/empty_cache를 하지 않는다")

    def test_bundle_stays_cached_without_release_request(self):
        self.refine()
        self.refine()
        self.assertEqual(self.loads, 1)
        self.assertEqual(self.after_calls, [])
        self.assertEqual(len(self.cache), 1)
        self.assertFalse(self.cache.stats()['keys'][0]['leases'], "클릭이 끝나면 lease도 끝난다")

    def test_inference_error_after_release_request_frees_bundle_once_frames_are_gone(self):
        def on_prompt(_prompt):
            model_cache.release_for_generation()
            raise RuntimeError('CUDA out of memory — tried to allocate')

        self.on_prompt = on_prompt
        with contextlib.redirect_stderr(io.StringIO()):
            mask = self.refine()
        self.assertFalse(mask.any())
        self.assertEqual([level for level, _ in self.notices], ['error'])
        self.assertEqual(len(self.cache), 0)
        self.assertEqual(len(self.after_calls), 1, "lease가 끝날 때 반납")
        self.assertEqual(self.release_calls, [True],
                         "예외 프레임이 사라진 뒤 한 번 더 반납해야 empty_cache가 블록을 돌려준다")

    def test_unencodable_prompts_do_not_break_the_click(self):
        mask = self.refine(text_prompt='nothing — ✓')      # 검출 없음 → bbox 폴백 로그('—')
        self.assertTrue(mask[20, 20], "검출이 없으면 YOLO bbox를 채운다")
        self.assertEqual(self.notices, [])
        self.assertIn("no detections for prompt 'nothing \\u2014 \\u2713'", _console_text(self.console))


class EditorAutoDetectWiringTests(unittest.TestCase):
    """vue_bridge._editor_process_impl → resolve_sam_model → 세션당 1회 경고 + YOLO bbox 마스크."""

    def setUp(self):
        sam_refiner._reset_unavailable_notices()
        self.addCleanup(sam_refiner._reset_unavailable_notices)
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.models = temp.name
        for name in ('mobile_sam.pt', 'sam3.pt', 'penis.pt'):
            with open(os.path.join(self.models, name), 'wb') as handle:
                handle.write(b'weights')
        self.image_path = os.path.join(temp.name, 'input.png')
        Image.new('RGB', (40, 30), (90, 90, 90)).save(self.image_path)

    def run_detect(self, host):
        from core import yolo_models
        from ui.vue_bridge import VueBridge

        box = SimpleNamespace(tolist=lambda: [5, 4, 25, 20])
        result = SimpleNamespace(masks=None, boxes=SimpleNamespace(xyxy=[box]))
        yolo = lambda _img, conf, verbose: [result]
        detector = os.path.join(self.models, 'penis.pt')
        with mock.patch.object(yolo_models, 'load_model_paths', return_value=[detector]), \
                mock.patch.object(yolo_models, 'get_editor_models_dir', return_value=self.models), \
                mock.patch.object(model_cache.YOLO_CACHE, 'get', return_value=yolo), \
                mock.patch.object(sam_refiner, 'missing_sam_runtime', _only_missing('mobile_sam')), \
                mock.patch.object(sam_refiner, 'refine_boxes_with_sam') as refine:
            raw = VueBridge._editor_process_impl(host, self.image_path, 'auto_detect', {'sam_model': 'auto'})
        return json.loads(raw), refine

    def test_missing_mobile_sam_warns_once_and_keeps_bbox_without_sam3(self):
        notices = []
        host = SimpleNamespace(showNotification=SimpleNamespace(emit=lambda *a: notices.append(a)),
                               _PREVIEW_MAX_EDGE=1024)
        first, refine = self.run_detect(host)
        second, _ = self.run_detect(host)
        refine.assert_not_called()                      # SAM3로 승격해서 돌리지 않는다
        self.assertEqual([level for level, _ in notices], ['warning'])
        self.assertIn('mobile_sam', notices[0][1])
        mask_bytes = base64.b64decode(first['mask_base64'].split(',', 1)[1])
        mask = np.array(Image.open(io.BytesIO(mask_bytes)))
        self.assertEqual(int((mask > 0).sum()), 20 * 16, "YOLO bbox 마스크를 그대로 쓴다")
        self.assertEqual(first['detect_count'], 1)
        self.assertEqual(second['detect_count'], 1)

    def test_refined_mask_has_no_false_error_toast_on_cp949_console(self):
        """'✓ Refined' 로그가 파이프 stdout(cp949)에서 UnicodeEncodeError를 내면, 이미 적용한
        SAM 마스크에 'SAM 정밀화 실패: cp949 codec…' 거짓 토스트가 붙던 문제."""
        from core import yolo_models
        from ui.vue_bridge import VueBridge

        console = _strict_cp949_stdout(self)
        notices = []
        host = SimpleNamespace(showNotification=SimpleNamespace(emit=lambda *a: notices.append(a)),
                               _PREVIEW_MAX_EDGE=1024)
        refined = np.zeros((30, 40), dtype=np.uint8)
        refined[6:12, 8:14] = 255
        box = SimpleNamespace(tolist=lambda: [5, 4, 25, 20])
        result = SimpleNamespace(masks=None, boxes=SimpleNamespace(xyxy=[box]))
        yolo = lambda _img, conf, verbose: [result]
        detector = os.path.join(self.models, 'penis.pt')
        with mock.patch.object(yolo_models, 'load_model_paths', return_value=[detector]), \
                mock.patch.object(yolo_models, 'get_editor_models_dir', return_value=self.models), \
                mock.patch.object(model_cache.YOLO_CACHE, 'get', return_value=yolo), \
                mock.patch.object(sam_refiner, 'missing_sam_runtime', _only_missing()), \
                mock.patch.object(sam_refiner, 'refine_boxes_with_sam', return_value=refined) as refine:
            raw = VueBridge._editor_process_impl(host, self.image_path, 'auto_detect', {'sam_model': 'auto'})
        data = json.loads(raw)
        refine.assert_called_once()
        self.assertEqual(refine.call_args.kwargs['sam_type'], 'mobile_sam')
        self.assertEqual(notices, [], "성공한 정밀화에 오류 토스트가 붙으면 안 된다")
        mask_bytes = base64.b64decode(data['mask_base64'].split(',', 1)[1])
        mask = np.array(Image.open(io.BytesIO(mask_bytes)))
        self.assertEqual(int((mask > 0).sum()), 6 * 6, "SAM 마스크가 적용된다")
        self.assertIn('[SAM] \\u2713 Refined mask applied (mobile_sam', _console_text(console))


if __name__ == '__main__':
    unittest.main()
