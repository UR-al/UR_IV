"""Forge 후처리 워커가 작업 직전에 편집기 SAM3 번들(~3.4GB)을 반납한다.

회귀 방지 대상 (감사 #8 잔여)
  · SAM3·ADetailer·Refine·배치 업스케일 워커는 GPU 리스(reserve)를 잡지 않아 생성 시작
    훅(before_generation → release_for_generation)이 돌지 않았다. 편집기에서 SAM3를 쓴 직후
    최대 90초 동안 앱의 번들과 Forge가 올리는 SAM3/ADetailer가 겹쳐 16GB를 넘겼다
    (ANIMA+SAM3+Forge OOM, 2e9e52b31).
  · 리스를 잡게 바꾸지는 않는다 — 이 작업들은 생성과 나란히 Forge 큐에 들어가 왔고,
    timeout=0 리스로 바꾸면 생성 중에는 실패하게 된다.
"""
import json
import os
import tempfile
import unittest
from unittest import mock

from core import model_cache, resource_coordinator
from core.model_cache import IdleModelCache
from core.resource_coordinator import release_before_backend_job

_PNG = b'\x89PNG\r\n\x1a\n' + b'\x00' * 32


class _Backend:
    """Forge 호출 순서만 기록하는 가짜 백엔드 (입력을 그대로 돌려준다)."""

    def __init__(self, events):
        self.events = events

    def _job(self, name, image_b64):
        self.events.append(f'forge:{name}')
        return image_b64

    def sam3(self, image_b64, _settings):
        return self._job('sam3', image_b64)

    def adetailer(self, image_b64, _settings):
        return self._job('adetailer', image_b64)

    def refine(self, image_b64, _settings):
        return self._job('refine', image_b64)

    def upscale(self, image_b64, _settings):
        return self._job('upscale', image_b64)


class WorkerReleaseOrderTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = temp.name
        self.out = os.path.join(self.root, 'out')
        self.paths = []
        for name in ('a.png', 'b.png'):
            path = os.path.join(self.root, name)
            with open(path, 'wb') as handle:
                handle.write(_PNG)
            self.paths.append(path)
        self.events = []
        self.backend = _Backend(self.events)
        patches = [
            mock.patch.object(model_cache, 'release_for_generation',
                              side_effect=lambda: self.events.append('release') or 0),
            mock.patch('backends.get_backend', return_value=self.backend),
        ]
        for patcher in patches:
            patcher.start()
            self.addCleanup(patcher.stop)

    def test_sam3_single(self):
        from workers.sam3_worker import Sam3SingleWorker
        results = []
        worker = Sam3SingleWorker(self.paths[0], {'output_folder': self.out})
        worker.finished.connect(results.append)
        worker.run()
        self.assertNotIn('error', json.loads(results[0]))
        self.assertEqual(self.events, ['release', 'forge:sam3'])

    def test_sam3_batch_releases_before_every_item(self):
        from workers.sam3_worker import Sam3BatchWorker
        results = []
        worker = Sam3BatchWorker(self.paths, {'output_folder': self.out})
        worker.single_done.connect(results.append)
        worker.run()
        self.assertEqual([('error' in json.loads(r)) for r in results], [False, False])
        self.assertEqual(self.events, ['release', 'forge:sam3'] * 2,
                         "배치 도중 편집기에서 SAM3를 다시 올려도 다음 항목 전에 반납")

    def test_adetailer_single(self):
        from workers.adetailer_worker import ADetailerSingleWorker
        results = []
        worker = ADetailerSingleWorker(self.paths[0], {'output_folder': self.out})
        worker.finished.connect(results.append)
        worker.run()
        self.assertNotIn('error', json.loads(results[0]))
        self.assertEqual(self.events, ['release', 'forge:adetailer'])

    def test_adetailer_batch_releases_before_every_item(self):
        from workers.adetailer_worker import ADetailerBatchWorker
        results = []
        worker = ADetailerBatchWorker(self.paths, {'output_folder': self.out})
        worker.single_done.connect(results.append)
        worker.run()
        self.assertEqual([('error' in json.loads(r)) for r in results], [False, False])
        self.assertEqual(self.events, ['release', 'forge:adetailer'] * 2)

    def test_refine(self):
        from workers.refine_worker import RefineWorker
        results = []
        worker = RefineWorker(self.paths[0], {'output_folder': self.out, 'main_prompt': '1girl'})
        worker.finished.connect(results.append)
        worker.run()
        self.assertNotIn('error', json.loads(results[0]))
        self.assertEqual(self.events, ['release', 'forge:refine'])

    def test_batch_upscale_releases_before_every_item(self):
        from workers import upscale_worker
        finished = []
        settings = {'mode': 'both', 'output_folder': self.out, 'ad_enabled': True, 'sam3_enabled': True}
        worker = upscale_worker.BatchUpscaleWorker(self.paths, settings)
        worker.single_finished.connect(lambda i, ok, msg: finished.append(ok))
        with mock.patch.object(upscale_worker, 'get_backend', return_value=self.backend):
            worker.run()
        self.assertEqual(finished, [True, True])
        self.assertEqual(self.events,
                         ['release', 'forge:upscale', 'forge:adetailer', 'forge:sam3'] * 2)

    def test_release_failure_never_blocks_the_forge_job(self):
        from workers.sam3_worker import Sam3SingleWorker
        results = []
        with mock.patch.object(model_cache, 'release_for_generation', side_effect=RuntimeError('boom')), \
                self.assertLogs('core.resource_coordinator', level='WARNING'):
            worker = Sam3SingleWorker(self.paths[0], {'output_folder': self.out})
            worker.finished.connect(results.append)
            worker.run()
        self.assertNotIn('error', json.loads(results[0]))
        self.assertEqual(self.events, ['forge:sam3'])


class ReleaseBeforeBackendJobTests(unittest.TestCase):
    def test_frees_cached_editor_bundle(self):
        released = []
        cache = IdleModelCache('sam3-test', idle_seconds=90.0, max_items=1,
                               after_evict=lambda: released.append(True))
        cache.get('sam3.pt|cuda', lambda _key: object())
        with mock.patch.object(model_cache, 'SAM3_CACHE', cache):
            self.assertTrue(release_before_backend_job('sam3'))
        self.assertEqual(len(cache), 0)
        self.assertEqual(released, [True])

    def test_bundle_in_use_by_the_editor_is_released_when_its_click_ends(self):
        released = []
        cache = IdleModelCache('sam3-test', idle_seconds=90.0, max_items=1,
                               after_evict=lambda: released.append(True))
        with mock.patch.object(model_cache, 'SAM3_CACHE', cache):
            with cache.lease('sam3.pt|cuda', lambda _key: object()):
                self.assertTrue(release_before_backend_job('adetailer'))
                self.assertEqual(released, [], "추론 중인 번들을 빼 가지 않는다")
            self.assertEqual(released, [True], "편집기 클릭이 끝나는 즉시 반납")
        self.assertEqual(len(cache), 0)

    def test_empty_cache_costs_nothing(self):
        released = []
        cache = IdleModelCache('sam3-test', idle_seconds=90.0, max_items=1,
                               after_evict=lambda: released.append(True))
        with mock.patch.object(model_cache, 'SAM3_CACHE', cache):
            self.assertTrue(release_before_backend_job())
        self.assertEqual(released, [], "비었으면 gc/empty_cache를 부르지 않는다")

    def test_failure_is_reported_not_raised(self):
        with mock.patch.object(resource_coordinator, 'release_in_process_vision_models',
                               side_effect=ImportError('no torch')), \
                self.assertLogs('core.resource_coordinator', level='WARNING') as logs:
            self.assertFalse(release_before_backend_job('refine'))
        self.assertIn('refine', logs.output[0])


if __name__ == '__main__':
    unittest.main()
