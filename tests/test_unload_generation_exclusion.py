"""'생성 후 모델 언로드'와 진행 중인 생성은 공유 GPU 리스로 서로 배타 — 회귀.

예전엔 언로드 판단이 ``gen_worker`` 하나만 봤다. 인페인트(_vue_inpaint_worker)·I2I 탭·편집기·채팅·
Creator·생성 API 가 샘플링하는 동안에도
  A) 대기열 끝 1초 지연 사이에 시작된 인페인트,
  B) 인페인트 중 T2I '생성'을 눌러 곧바로 '사용 중' 실패한 T2I 의 완료 처리,
  C) 자동화 중지
가 Forge unload-checkpoint(queue_lock 없이 모델을 내린다)를 보내 진행 중인 작업을 깨뜨릴 수 있었다.
이제 판단은 리스를 보고(generation_active), 언로드 스레드는 리스를 try_hold 로 잡은 동안에만
HTTP 를 보내며, 생성 쪽은 reserve_generation_lease 로 진행 중인 언로드를 기다린다.

리스를 잡지 않는 Forge 후처리 워커(ADetailer·SAM3 단일/배치·Refine·배치 업스케일)는 생성과 나란히
돌도록 리스 대신 backend_job 으로 등록한다 — 도는 동안엔 판단(unload_blocked)과 try_hold 가 모두
언로드를 건너뛰고, 진행 중인 언로드는 기다렸다 시작한다(예전엔 셋 다 이 작업들을 보지 못했다).
"""
from __future__ import annotations

import json
import os
import tempfile
import threading
import time
import unittest
from types import SimpleNamespace
from unittest import mock

from core import post_generation
from core.post_generation import (
    PREF_KEY,
    UNLOAD_HOLD_OWNER,
    UNLOAD_SKIPPED_BUSY,
    reserve_generation_lease,
    start_post_generation_unload,
)
from core.resource_coordinator import HOLD_PHASE, GenerationResourceCoordinator, ResourceBusyError
from ui.generator_generation import POST_GEN_UNLOAD_SKIPPED_STATUS, GenerationMixin


class _UnloadBackend:
    def __init__(self, *, block=False):
        self.calls = []
        self.entered = threading.Event()
        self.release = threading.Event()
        if not block:
            self.release.set()

    def unload_checkpoint(self):
        self.calls.append(threading.current_thread().name)
        self.entered.set()
        self.release.wait(5)
        return True


class _Held:
    """다른 스레드가 생성 리스를 쥐고 있는 상태(인페인트 샘플링 중)를 만든다."""

    def __init__(self, coordinator, owner='img2img'):
        self.coordinator = coordinator
        self.owner = owner
        self.entered = threading.Event()
        self.release = threading.Event()
        self.thread = threading.Thread(target=self._run, daemon=True)

    def _run(self):
        with self.coordinator.reserve(self.owner, unload_llm=False, timeout=0):
            self.entered.set()
            self.release.wait(5)

    def __enter__(self):
        self.thread.start()
        assert self.entered.wait(5)
        return self

    def __exit__(self, *_):
        self.release.set()
        self.thread.join(5)
        return False


class _PendingCase(unittest.TestCase):
    def setUp(self):
        post_generation._pending_unload = None
        grace = mock.patch.object(post_generation, 'UNLOAD_HOLD_GRACE_SECONDS', 0.05)
        grace.start()
        self.addCleanup(grace.stop)

    def tearDown(self):
        thread = post_generation._pending_unload
        if thread is not None:
            thread.join(5)
        post_generation._pending_unload = None


class CoordinatorHoldTests(unittest.TestCase):
    def test_hold_and_generation_exclude_each_other(self):
        states = []
        hooks = []
        coord = GenerationResourceCoordinator(on_state=states.append, before_generation=lambda: hooks.append(1))
        with coord.try_hold(UNLOAD_HOLD_OWNER) as state:
            self.assertEqual(state.phase, HOLD_PHASE)
            self.assertTrue(coord.is_busy())
            self.assertFalse(coord.generation_active(), '언로드 hold 는 생성이 아니다')
            with self.assertRaisesRegex(ResourceBusyError, '언로드'):
                with coord.reserve('txt2img', unload_llm=False, timeout=0):
                    self.fail('언로드 중에는 생성이 리스를 잡지 못한다')
        self.assertEqual((states, hooks), ([], []), 'hold 는 Creator 상태 UI·편집기 캐시 훅을 부르지 않는다')
        self.assertFalse(coord.is_busy())
        with coord.reserve('img2img', unload_llm=False, timeout=0):
            self.assertTrue(coord.generation_active())
            with self.assertRaises(ResourceBusyError):
                with coord.try_hold(UNLOAD_HOLD_OWNER):
                    self.fail('생성 중에는 언로드가 리스를 잡지 못한다')
        self.assertEqual(coord.state.phase, 'idle')


class UnloadThreadHoldsTheLeaseTests(_PendingCase):
    def test_unload_is_skipped_while_another_job_samples_and_runs_after(self):
        coord = GenerationResourceCoordinator()
        backend = _UnloadBackend()
        reported = []
        with _Held(coord):
            thread = start_post_generation_unload(backend, on_done=reported.append, coordinator=coord)
            thread.join(5)
        self.assertEqual(backend.calls, [], '샘플링 중에는 unload-checkpoint 를 보내지 않는다')
        self.assertEqual(len(reported), 1)
        self.assertIs(reported[0], UNLOAD_SKIPPED_BUSY)
        self.assertFalse(reported[0], '모르는 호출자에겐 실패(False)로 읽힌다')

        thread = start_post_generation_unload(backend, on_done=reported.append, coordinator=coord)
        thread.join(5)
        self.assertEqual(len(backend.calls), 1)
        self.assertIs(reported[-1], True)

    def test_generation_started_during_an_unload_waits_for_it(self):
        coord = GenerationResourceCoordinator()
        backend = _UnloadBackend(block=True)
        start_post_generation_unload(backend, coordinator=coord)
        self.assertTrue(backend.entered.wait(5))
        self.assertEqual(coord.state.owner, UNLOAD_HOLD_OWNER)
        order = []
        threading.Timer(0.1, lambda: (order.append('unload done'), backend.release.set())).start()
        with reserve_generation_lease('img2img', coordinator=coord):
            order.append('generation')
        self.assertEqual(order, ['unload done', 'generation'])

    def test_unload_that_grabs_the_lease_right_after_the_wait_is_waited_out(self):
        # 워커가 wait_for_pending_unload 를 지난 '직후' 언로드가 리스를 잡은 경합 — 예전엔
        # reserve(timeout=0) 가 곧바로 '사용 중'으로 실패했다(대기열 동결 항목은 보류로 멈춤).
        coord = GenerationResourceCoordinator()
        backend = _UnloadBackend(block=True)
        real_wait = post_generation.wait_for_pending_unload
        calls = []

        def racing_wait(*args, **kwargs):
            calls.append(len(calls))
            if len(calls) == 1:
                start_post_generation_unload(backend, coordinator=coord)
                assert backend.entered.wait(5)        # 언로드가 리스를 쥐었다
                threading.Timer(0.1, backend.release.set).start()
                return True                            # 워커는 '기다릴 것 없음'으로 지나간 뒤다
            return real_wait(*args, **kwargs)

        with mock.patch.object(post_generation, 'wait_for_pending_unload', side_effect=racing_wait):
            with reserve_generation_lease('txt2img', coordinator=coord) as state:
                self.assertEqual(state.owner, 'txt2img')
        self.assertEqual(len(calls), 2, '리스를 못 잡으면 언로드를 한 번 더 기다린다')
        self.assertEqual(len(backend.calls), 1)

    def test_a_real_generation_still_fails_fast(self):
        coord = GenerationResourceCoordinator()
        with _Held(coord, owner='inpaint'):
            started = time.monotonic()
            with self.assertRaisesRegex(ResourceBusyError, '다른 생성 작업'):
                with reserve_generation_lease('txt2img', coordinator=coord):
                    self.fail('다른 생성이 쥐고 있으면 잡지 못한다')
            self.assertLess(time.monotonic() - started, 1.0)


class _FakeImg2ImgBackend:
    def __init__(self):
        self.entered = threading.Event()
        self.release = threading.Event()

    def img2img(self, model, payload, progress_callback=None, cancel_check=None):
        self.entered.set()
        self.release.wait(5)
        return SimpleNamespace(success=True, image_data=b'png', info={}, error='')


class _GateHost:
    """_maybe_unload_models_after_generation 이 보는 것만 — gen_worker 는 끝났고 대기열·자동화는 멈췄다."""

    _maybe_unload_models_after_generation = GenerationMixin._maybe_unload_models_after_generation

    def __init__(self):
        self.queue_manager = SimpleNamespace(is_running=False)
        self.gen_worker = SimpleNamespace(isRunning=lambda: False)
        self.is_automating = False
        self.statuses = []

    def show_status(self, message, *_args):
        self.statuses.append(message)


class InpaintInFlightTests(_PendingCase):
    """실제 Img2ImgFlowWorker(인페인트)가 샘플링하는 동안의 '생성 후 언로드' 판단."""

    def setUp(self):
        super().setUp()
        self.coord = GenerationResourceCoordinator()
        self.unload_backend = _UnloadBackend()
        for patcher in (
            mock.patch('core.resource_coordinator.get_generation_coordinator', return_value=self.coord),
            mock.patch('workers.generation_worker.get_generation_coordinator', return_value=self.coord),
            mock.patch('core.ui_prefs.read_ui_prefs', return_value={PREF_KEY: True}),
            mock.patch('backends.get_backend', return_value=self.unload_backend),
        ):
            patcher.start()
            self.addCleanup(patcher.stop)

    def _start_inpaint(self):
        from workers.generation_worker import Img2ImgFlowWorker
        backend = _FakeImg2ImgBackend()
        worker = Img2ImgFlowWorker('model', {'prompt': 'inpaint'})
        patcher = mock.patch('workers.generation_worker.get_backend', return_value=backend)
        patcher.start()
        self.addCleanup(patcher.stop)
        thread = threading.Thread(target=worker.run, daemon=True)
        thread.start()
        self.assertTrue(backend.entered.wait(5))
        self.assertEqual((self.coord.state.phase, self.coord.state.owner), ('running', 'img2img'))
        return backend, thread

    def test_queue_end_or_stop_does_not_unload_under_a_running_inpaint(self):
        backend, thread = self._start_inpaint()
        host = _GateHost()
        host._maybe_unload_models_after_generation()
        self.assertIsNone(post_generation._pending_unload, '요청 자체를 보내지 않는다')
        self.assertNotIn('생성 후 모델 언로드 요청', host.statuses)

        # 판단 직후 인페인트가 시작된 경합 — 판단을 지나도 언로드 스레드가 리스를 못 잡아 건너뛴다.
        # 이미 띄운 '요청' 문구는 건너뜀 문구로 바뀐다(예전엔 콘솔에만 찍혀 '요청'이 그대로 남았다).
        # 유예를 늘려 '요청'(메인)이 건너뜀(언로드 스레드)보다 먼저 오는 실제 순서를 확정한다.
        with mock.patch.object(self.coord, 'generation_active', return_value=False), \
                mock.patch.object(post_generation, 'UNLOAD_HOLD_GRACE_SECONDS', 0.3):
            host._maybe_unload_models_after_generation()
            post_generation._pending_unload.join(5)
        self.assertEqual(self.unload_backend.calls, [])
        self.assertEqual(host.statuses, ['생성 후 모델 언로드 요청', POST_GEN_UNLOAD_SKIPPED_STATUS])

        backend.release.set()
        thread.join(5)
        post_generation._pending_unload = None
        host._maybe_unload_models_after_generation()        # 인페인트가 끝난 뒤엔 평소대로
        post_generation._pending_unload.join(5)
        self.assertEqual(len(self.unload_backend.calls), 1)
        self.assertEqual(host.statuses[-1], '생성 후 모델 언로드 요청', '성공한 언로드엔 건너뜀 문구가 없다')

    def test_busy_t2i_failure_during_an_inpaint_does_not_unload(self):
        # 경로 B — 인페인트 중 T2I 를 누르면 T2I 워커가 곧바로 '사용 중'으로 끝나고, 실패 분기가
        # 생성 후 언로드를 부른다(그 시점에 gen_worker 는 이미 끝났다).
        backend, thread = self._start_inpaint()
        host = _FinishHost()
        with mock.patch('core.gen_stats.get_gen_stats') as stats:
            host.on_generation_finished('이미지 생성 중 오류: 다른 생성 작업이 GPU 리소스를 사용 중입니다', {})
        stats.return_value.record.assert_called_once()        # 실패 분기를 실제로 탔다(기록은 가짜)
        self.assertIsNone(post_generation._pending_unload)
        self.assertEqual(self.unload_backend.calls, [])
        backend.release.set()
        thread.join(5)


class _FinishHost(_GateHost):
    on_generation_finished = GenerationMixin.on_generation_finished
    _generation_matches_queue = GenerationMixin._generation_matches_queue
    _generation_stats_meta = GenerationMixin._generation_stats_meta
    _restore_generate_button = GenerationMixin._restore_generate_button

    def __init__(self):
        super().__init__()
        self.queue_manager = None
        self.btn_auto_toggle = SimpleNamespace(isChecked=lambda: False)
        self.btn_generate = mock.Mock()
        self.vue_bridge = SimpleNamespace(generationError=mock.Mock())
        self._gen_request_meta = {'model': 'm', 'width': 8, 'height': 8, 'seed': None}

    def setWindowTitle(self, _title):
        pass


class ManualAndPeriodicUnloadTests(_PendingCase):
    def setUp(self):
        super().setUp()
        self.coord = GenerationResourceCoordinator()
        patcher = mock.patch('core.resource_coordinator.get_generation_coordinator', return_value=self.coord)
        patcher.start()
        self.addCleanup(patcher.stop)

    def _host(self):
        notes = []
        host = SimpleNamespace(statuses=[], vue_bridge=SimpleNamespace(showNotification=SimpleNamespace(
            emit=lambda level, text: notes.append((level, text)))))
        host.show_status = lambda message, *_a: host.statuses.append(message)
        return host, notes

    def test_manual_unload_is_refused_while_a_generation_runs(self):
        from ui.manual_model_unload import request_manual_backend_unload
        backend = SimpleNamespace(unload_models=mock.Mock(return_value=True))
        host, notes = self._host()
        with _Held(self.coord):
            self.assertIsNone(request_manual_backend_unload(host, backend, schedule=lambda ms, fn: None))
        backend.unload_models.assert_not_called()
        self.assertEqual(host.statuses, ['Model unload skipped: a generation is running.'])
        self.assertEqual(notes[-1][0], 'warning')

    def test_manual_unload_reports_a_generation_that_started_in_between(self):
        from ui.manual_model_unload import request_manual_backend_unload
        backend = SimpleNamespace(unload_models=mock.Mock(return_value=True))
        host, notes = self._host()
        scheduled = []
        with _Held(self.coord), mock.patch.object(self.coord, 'generation_active', return_value=False):
            thread = request_manual_backend_unload(host, backend, schedule=lambda ms, fn: scheduled.append(fn))
            thread.join(5)
        scheduled[0]()
        backend.unload_models.assert_not_called()
        self.assertEqual(host.statuses[-1], 'Model unload skipped: a generation is running.')
        self.assertEqual(notes[-1][0], 'warning')

    def test_periodic_queue_unload_is_skipped_while_another_owner_holds_the_lease(self):
        from widgets.queue_manager import QueueManager
        backend = _UnloadBackend()
        manager = QueueManager.__new__(QueueManager)   # _request_periodic_unload 는 두 카운터만 쓴다
        manager.generated_count, manager.cleanup_every_n = 4, 2
        with _Held(self.coord, owner='chat:abc'), \
                mock.patch('backends.get_backend', return_value=backend), \
                self.assertLogs('widgets.queue_manager', level='INFO') as logs:
            manager._request_periodic_unload()
            post_generation._pending_unload.join(5)
        self.assertEqual(backend.calls, [])
        self.assertTrue(any('건너뜀' in line for line in logs.output))


class _BackendJobThread:
    """다른 스레드에서 도는 후처리 백엔드 작업(backend_job) — release 전까지 붙잡는다."""

    def __init__(self, coordinator, *, hold_for=None):
        self.coordinator = coordinator
        self.entered = threading.Event()
        self.release = threading.Event()
        self.hold_for = hold_for
        self.thread = threading.Thread(target=self._run, daemon=True)

    def _run(self):
        with self.coordinator.backend_job('adetailer'):
            self.entered.set()
            if self.hold_for is not None:
                time.sleep(self.hold_for)
            else:
                self.release.wait(5)

    def __enter__(self):
        self.thread.start()
        assert self.entered.wait(5)
        return self

    def __exit__(self, *_):
        self.release.set()
        self.thread.join(5)
        return False


class BackendJobCoordinatorTests(unittest.TestCase):
    def test_backend_job_runs_beside_a_generation_but_excludes_the_unload_hold(self):
        coord = GenerationResourceCoordinator()
        with coord.backend_job('adetailer'):
            self.assertTrue(coord.backend_jobs_active())
            self.assertTrue(coord.unload_blocked())
            self.assertFalse(coord.generation_active(), '후처리는 생성 리스를 잡지 않는다')
            with coord.reserve('txt2img', unload_llm=False, timeout=0):
                self.assertEqual(coord.state.owner, 'txt2img', 'T2I 와 나란히 돈다')
            with self.assertRaisesRegex(ResourceBusyError, '후처리'):
                with coord.try_hold(UNLOAD_HOLD_OWNER):
                    self.fail('후처리 작업 중에는 언로드가 hold 를 잡지 못한다')
            self.assertEqual(coord.state.phase, 'idle', '실패한 hold 는 리스를 돌려놓는다')
            with coord.reserve('img2img', unload_llm=False, timeout=0):
                pass
        self.assertFalse(coord.backend_jobs_active())
        self.assertFalse(coord.unload_blocked())
        with coord.try_hold(UNLOAD_HOLD_OWNER) as state:
            self.assertEqual(state.phase, HOLD_PHASE)

    def test_jobs_are_counted_so_the_hold_waits_for_the_last_one(self):
        coord = GenerationResourceCoordinator()
        with coord.backend_job('sam3-batch'), coord.backend_job('refine'):
            pass
        with _BackendJobThread(coord), _BackendJobThread(coord) as second:
            second.release.set()
            second.thread.join(5)
            self.assertTrue(coord.backend_jobs_active(), '하나가 끝나도 다른 하나가 돈다')
            with self.assertRaises(ResourceBusyError):
                with coord.try_hold(UNLOAD_HOLD_OWNER):
                    self.fail('남은 작업이 있으면 hold 하지 않는다')
        self.assertFalse(coord.backend_jobs_active())

    def test_hold_waits_out_a_job_that_ends_within_its_grace(self):
        coord = GenerationResourceCoordinator()
        job = _BackendJobThread(coord, hold_for=0.05)
        with job:
            with coord.try_hold(UNLOAD_HOLD_OWNER, timeout=2) as state:
                self.assertEqual(state.phase, HOLD_PHASE)
                self.assertFalse(coord.backend_jobs_active())

    def test_job_started_during_a_hold_waits_for_it(self):
        coord = GenerationResourceCoordinator()
        order = []
        held, release = threading.Event(), threading.Event()

        def unload():
            with coord.try_hold(UNLOAD_HOLD_OWNER):
                held.set()
                release.wait(5)
                order.append('unload done')

        thread = threading.Thread(target=unload, daemon=True)
        thread.start()
        self.assertTrue(held.wait(5))
        threading.Timer(0.1, release.set).start()
        with coord.backend_job('sam3'):
            order.append('job')
        thread.join(5)
        self.assertEqual(order, ['unload done', 'job'])

    def test_a_stuck_hold_fails_the_job_instead_of_overlapping_it(self):
        """기다림은 유한하고, 시간이 지나도 hold 가 쥐어져 있으면 언로드와 겹쳐 돌지 않는다(fail-closed).

        예전엔 경고만 남기고 hold 가 쥐어진 채로 작업을 시작했다 — 언로드와 후처리가 동시에 돌았다.
        """
        coord = GenerationResourceCoordinator()
        held, release = threading.Event(), threading.Event()

        def unload():
            with coord.try_hold(UNLOAD_HOLD_OWNER):
                held.set()
                release.wait(5)

        thread = threading.Thread(target=unload, daemon=True)
        thread.start()
        self.assertTrue(held.wait(5))
        ran = []
        try:
            started = time.monotonic()
            with self.assertLogs('core.resource_coordinator', level='WARNING'), \
                    self.assertRaisesRegex(ResourceBusyError, '언로드'):
                with coord.backend_job('refine', wait_timeout=0.05):
                    ran.append('refine')
            self.assertLess(time.monotonic() - started, 2, '기다림은 wait_timeout 으로 끝난다')
            self.assertEqual(ran, [], 'hold 가 쥐어진 채로는 작업 본문을 돌리지 않는다')
            self.assertFalse(coord.backend_jobs_active(), '실패한 작업은 세지 않는다')
            self.assertEqual(coord.state.phase, HOLD_PHASE, 'hold 는 그대로다')
        finally:
            release.set()
            thread.join(5)
        with coord.backend_job('refine', wait_timeout=0.05):
            ran.append('after')
            self.assertTrue(coord.backend_jobs_active())
        self.assertEqual(ran, ['after'], 'hold 가 풀리면 평소대로 돈다')
        self.assertFalse(coord.backend_jobs_active())

    def test_default_wait_follows_the_module_constant_at_call_time(self):
        coord = GenerationResourceCoordinator()
        held, release = threading.Event(), threading.Event()

        def unload():
            with coord.try_hold(UNLOAD_HOLD_OWNER):
                held.set()
                release.wait(5)

        thread = threading.Thread(target=unload, daemon=True)
        thread.start()
        self.assertTrue(held.wait(5))
        try:
            with mock.patch('core.resource_coordinator.BACKEND_JOB_UNLOAD_WAIT_SECONDS', 0.05), \
                    self.assertLogs('core.resource_coordinator', level='WARNING'), \
                    self.assertRaises(ResourceBusyError):
                with coord.backend_job('sam3'):
                    self.fail('멈춘 hold 와 겹쳐 돌지 않는다')
        finally:
            release.set()
            thread.join(5)


class UnloadWaitOutlastsUnloadHttpTests(unittest.TestCase):
    """후처리의 언로드 대기는 언로드 HTTP 가 스스로 끝나는 상한보다 길다 — 정상적인 언로드는 늘
    먼저 hold 를 놓으므로, 대기 시간 초과(→ ResourceBusyError)는 멈춘 hold 에서만 난다."""

    def _unload_timeout(self, backend, patch_target):
        with mock.patch(patch_target) as post:
            post.return_value.raise_for_status.return_value = None
            self.assertTrue(backend.unload_checkpoint())
        self.assertEqual(post.call_count, 1)
        return post.call_args.kwargs['timeout']

    def test_the_wait_is_longer_than_connect_plus_read_of_every_unload_request(self):
        from backends.comfyui_backend import ComfyUIBackend
        from backends.webui_backend import WebUIBackend
        from core.resource_coordinator import (
            BACKEND_JOB_UNLOAD_WAIT_SECONDS,
            UNLOAD_HTTP_TIMEOUT_SECONDS,
        )
        forge = self._unload_timeout(WebUIBackend('http://127.0.0.1:7860'),
                                     'backends.webui_backend.requests.post')
        comfy = self._unload_timeout(ComfyUIBackend('http://127.0.0.1:8188'),
                                     'backends.comfyui_backend.requests.post')
        self.assertEqual(forge, UNLOAD_HTTP_TIMEOUT_SECONDS, 'Forge 언로드는 공유 상수를 쓴다')
        for timeout in (forge, comfy):
            self.assertIsInstance(timeout, (int, float), '연결·읽기 타임아웃이 모두 걸려 있어야 한다')
            # requests 는 스칼라 타임아웃을 연결과 읽기에 각각 쓴다 — hold 는 최대 약 2배를 쥔다
            self.assertGreater(BACKEND_JOB_UNLOAD_WAIT_SECONDS, 2 * timeout)


_PNG = b'\x89PNG\r\n\x1a\n' + b'\x00' * 32


class _Forge(_UnloadBackend):
    """같은 Forge — 후처리 HTTP(ADetailer·SAM3)는 job_release 전까지 샘플링 중, 언로드는 기록만."""

    def __init__(self, *, block_jobs=True):
        super().__init__()
        self.jobs = []
        self.job_entered = threading.Event()
        self.job_release = threading.Event()
        if not block_jobs:
            self.job_release.set()

    def _job(self, name, image_b64):
        self.jobs.append(name)
        self.job_entered.set()
        self.job_release.wait(5)
        return image_b64

    def adetailer(self, image_b64, _settings):
        return self._job('adetailer', image_b64)

    def sam3(self, image_b64, _settings):
        return self._job('sam3', image_b64)


class PostprocessWorkerUnloadTests(_PendingCase):
    """실제 ADetailer·SAM3 워커가 Forge 에서 샘플링하는 동안의 세 언로드 경로."""

    def setUp(self):
        super().setUp()
        self.coord = GenerationResourceCoordinator()
        self.forge = _Forge()
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.image = os.path.join(temp.name, 'a.png')
        with open(self.image, 'wb') as handle:
            handle.write(_PNG)
        self.out = os.path.join(temp.name, 'out')
        for patcher in (
            mock.patch('core.resource_coordinator.get_generation_coordinator', return_value=self.coord),
            mock.patch('core.resource_coordinator.release_in_process_vision_models'),
            mock.patch('core.ui_prefs.read_ui_prefs', return_value={PREF_KEY: True}),
            mock.patch('backends.get_backend', return_value=self.forge),
        ):
            patcher.start()
            self.addCleanup(patcher.stop)
        self.addCleanup(self.forge.job_release.set)

    def _run_worker(self, worker_cls):
        from PyQt6.QtCore import Qt
        worker = worker_cls(self.image, {'output_folder': self.out})
        results = []
        # run() 을 파이썬 스레드에서 돌린다 — 이벤트 루프 없이 받으려면 직접 연결
        worker.finished.connect(results.append, Qt.ConnectionType.DirectConnection)
        thread = threading.Thread(target=worker.run, daemon=True)
        thread.start()
        return thread, results

    def test_no_unload_trigger_fires_while_an_adetailer_job_samples(self):
        from ui.manual_model_unload import request_manual_backend_unload
        from widgets.queue_manager import QueueManager
        from workers.adetailer_worker import ADetailerSingleWorker
        thread, results = self._run_worker(ADetailerSingleWorker)
        self.assertTrue(self.forge.job_entered.wait(5))
        self.assertTrue(self.coord.backend_jobs_active())

        # ① 단일 T2I 가 끝나 '생성 후 언로드' — 판단에서 거른다(요청 문구도 없다)
        host = _GateHost()
        host._maybe_unload_models_after_generation()
        self.assertIsNone(post_generation._pending_unload, '요청 자체를 보내지 않는다')
        self.assertEqual(host.statuses, [])

        # ② 판단 직후 작업이 시작된 경합 — 언로드 스레드가 건너뛰고 '요청' 문구를 바꾼다
        with mock.patch.object(self.coord, 'unload_blocked', return_value=False), \
                mock.patch.object(post_generation, 'UNLOAD_HOLD_GRACE_SECONDS', 0.3):
            host._maybe_unload_models_after_generation()
            post_generation._pending_unload.join(5)
        self.assertEqual(host.statuses, ['생성 후 모델 언로드 요청', POST_GEN_UNLOAD_SKIPPED_STATUS])
        post_generation._pending_unload = None

        # ③ 대기열 매 N장 정리
        manager = QueueManager.__new__(QueueManager)
        manager.generated_count, manager.cleanup_every_n = 4, 2
        with self.assertLogs('widgets.queue_manager', level='INFO') as logs:
            manager._request_periodic_unload()
            post_generation._pending_unload.join(5)
        self.assertTrue(any('건너뜀' in line for line in logs.output))
        post_generation._pending_unload = None

        # ④ VRAM 게이지 수동 언로드
        manual = SimpleNamespace(statuses=[], vue_bridge=None)
        manual.show_status = lambda message, *_a: manual.statuses.append(message)
        self.assertIsNone(request_manual_backend_unload(manual, self.forge, schedule=lambda ms, fn: None))
        self.assertEqual(manual.statuses, ['Model unload skipped: a generation is running.'])

        self.assertEqual(self.forge.calls, [], '샘플링 중에는 unload-checkpoint 를 보내지 않는다')
        self.forge.job_release.set()
        thread.join(5)
        self.assertNotIn('error', json.loads(results[0]))
        self.assertFalse(self.coord.backend_jobs_active())

        host._maybe_unload_models_after_generation()        # 작업이 끝난 뒤엔 평소대로
        post_generation._pending_unload.join(5)
        self.assertEqual(len(self.forge.calls), 1)

    def test_job_started_during_an_unload_waits_for_it(self):
        from workers.sam3_worker import Sam3SingleWorker
        self.forge.job_release.set()
        unload = _UnloadBackend(block=True)
        start_post_generation_unload(unload, coordinator=self.coord)
        self.assertTrue(unload.entered.wait(5))
        thread, results = self._run_worker(Sam3SingleWorker)
        time.sleep(0.1)
        self.assertEqual(self.forge.jobs, [], '언로드가 끝나기 전에는 SAM3 를 보내지 않는다')
        unload.release.set()
        thread.join(5)
        self.assertEqual(self.forge.jobs, ['sam3'])
        self.assertNotIn('error', json.loads(results[0]))

    def test_a_stuck_unload_fails_the_job_instead_of_sampling_beside_it(self):
        """언로드 hold 가 대기 시간을 넘겨도 풀리지 않으면 ADetailer 를 Forge 에 보내지 않고 오류로 끝낸다."""
        from workers.adetailer_worker import ADetailerSingleWorker
        self.forge.job_release.set()
        unload = _UnloadBackend(block=True)
        self.addCleanup(unload.release.set)
        start_post_generation_unload(unload, coordinator=self.coord)
        self.assertTrue(unload.entered.wait(5))
        with mock.patch('core.resource_coordinator.BACKEND_JOB_UNLOAD_WAIT_SECONDS', 0.1), \
                self.assertLogs('core.resource_coordinator', level='WARNING'):
            thread, results = self._run_worker(ADetailerSingleWorker)
            thread.join(5)
        self.assertFalse(thread.is_alive())
        self.assertEqual(self.forge.jobs, [], '멈춘 언로드와 겹쳐 ADetailer 를 보내지 않는다')
        self.assertIn('언로드', json.loads(results[0])['error'])
        self.assertFalse(self.coord.backend_jobs_active())
        unload.release.set()
        post_generation._pending_unload.join(5)
        self.assertEqual(self.coord.state.phase, 'idle')


class PostprocessWorkersRegisterBackendJobsTests(unittest.TestCase):
    """리스를 잡지 않는 Forge 후처리 호출 여섯 곳이 모두 backend_job 안에서 나간다."""

    def test_every_forge_postprocess_call_is_a_backend_job(self):
        from workers import upscale_worker
        from workers.adetailer_worker import ADetailerBatchWorker, ADetailerSingleWorker
        from workers.refine_worker import RefineWorker
        from workers.sam3_worker import Sam3BatchWorker, Sam3SingleWorker
        coord = GenerationResourceCoordinator()
        seen = []

        class Forge:
            def _job(self, name, image_b64):
                seen.append((name, coord.backend_jobs_active(), coord.generation_active()))
                return image_b64

            def adetailer(self, image_b64, _settings):
                return self._job('adetailer', image_b64)

            def sam3(self, image_b64, _settings):
                return self._job('sam3', image_b64)

            def refine(self, image_b64, _settings):
                return self._job('refine', image_b64)

            def upscale(self, image_b64, _settings):
                return self._job('upscale', image_b64)

        forge = Forge()
        with tempfile.TemporaryDirectory() as root, \
                mock.patch('core.resource_coordinator.get_generation_coordinator', return_value=coord), \
                mock.patch('core.resource_coordinator.release_in_process_vision_models'), \
                mock.patch('backends.get_backend', return_value=forge), \
                mock.patch.object(upscale_worker, 'get_backend', return_value=forge):
            image = os.path.join(root, 'a.png')
            with open(image, 'wb') as handle:
                handle.write(_PNG)
            out = os.path.join(root, 'out')
            for worker in (
                Sam3SingleWorker(image, {'output_folder': out}),
                Sam3BatchWorker([image], {'output_folder': out}),
                ADetailerSingleWorker(image, {'output_folder': out}),
                ADetailerBatchWorker([image], {'output_folder': out}),
                RefineWorker(image, {'output_folder': out, 'main_prompt': '1girl'}),
                upscale_worker.BatchUpscaleWorker([image], {'mode': 'both', 'output_folder': out}),
            ):
                worker.run()
        self.assertEqual([name for name, _job, _gen in seen],
                         ['sam3', 'sam3', 'adetailer', 'adetailer', 'refine', 'upscale', 'adetailer', 'sam3'])
        self.assertTrue(all(job for _name, job, _gen in seen), seen)
        self.assertFalse(any(gen for _name, _job, gen in seen), '생성 리스는 잡지 않는다')
        self.assertFalse(coord.backend_jobs_active())


if __name__ == '__main__':
    unittest.main()
