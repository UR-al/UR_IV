"""모델 캐시 테스트 — 유휴 해제 + 로딩 1회 보장.

회귀 방지 대상:
  · YOLO가 클릭마다 재로딩되던 문제 (vue_bridge 루프 안에서 YOLO(path) 생성)
  · SAM3가 `finally: cache.clear()` 때문에 캐시가 무력화돼 3.45GB를 매번 다시 읽던 문제
"""
import sys
import threading
import time
import types
import unittest
import weakref
from unittest import mock

from core import model_cache
from core.model_cache import IdleModelCache, release_torch_memory


class _Clock:
    """테스트용 가짜 시계 — 실제로 기다리지 않는다."""
    def __init__(self):
        self.t = 1000.0

    def __call__(self):
        return self.t

    def advance(self, seconds):
        self.t += seconds


class TestIdleModelCache(unittest.TestCase):
    def setUp(self):
        self.clock = _Clock()
        self.loads = []
        self.evicted = []
        self.cache = IdleModelCache('test', idle_seconds=60.0, max_items=3,
                                    on_evict=self.evicted.append,
                                    time_fn=self.clock)

    def _loader(self, key):
        self.loads.append(key)
        return f"model:{key}"

    def test_loads_once_then_hits_cache(self):
        for _ in range(5):
            self.assertEqual(self.cache.get('a', self._loader), 'model:a')
        self.assertEqual(self.loads, ['a'], "캐시 히트여야 하는데 재로딩됨")

    def test_distinct_keys_load_separately(self):
        self.cache.get('a', self._loader)
        self.cache.get('b', self._loader)
        self.assertEqual(self.loads, ['a', 'b'])
        self.assertEqual(len(self.cache), 2)

    def test_idle_eviction(self):
        self.cache.get('a', self._loader)
        self.clock.advance(61)
        self.assertEqual(self.cache.sweep(), 1)
        self.assertEqual(len(self.cache), 0)
        self.assertEqual(self.evicted, ['model:a'])

    def test_not_evicted_before_idle(self):
        self.cache.get('a', self._loader)
        self.clock.advance(30)
        self.assertEqual(self.cache.sweep(), 0)
        self.assertEqual(len(self.cache), 1)

    def test_access_refreshes_idle_timer(self):
        self.cache.get('a', self._loader)
        self.clock.advance(50)
        self.cache.get('a', self._loader)      # 갱신
        self.clock.advance(50)                 # 최초 접근 기준 100초지만 갱신 후 50초
        self.assertEqual(self.cache.sweep(), 0)
        self.assertEqual(self.loads, ['a'])

    def test_get_sweeps_stale_entries(self):
        self.cache.get('a', self._loader)
        self.clock.advance(61)
        self.cache.get('b', self._loader)      # get 내부에서 sweep
        self.assertEqual(self.evicted, ['model:a'])

    def test_reload_after_eviction(self):
        self.cache.get('a', self._loader)
        self.clock.advance(61)
        self.cache.sweep()
        self.cache.get('a', self._loader)
        self.assertEqual(self.loads, ['a', 'a'])

    def test_capacity_evicts_oldest(self):
        for key in ('a', 'b', 'c'):
            self.cache.get(key, self._loader)
            self.clock.advance(1)
        self.cache.get('d', self._loader)
        self.assertEqual(len(self.cache), 3)
        self.assertNotIn('a', self.cache.keys())
        self.assertEqual(self.evicted, ['model:a'])

    def test_clear_evicts_everything(self):
        self.cache.get('a', self._loader)
        self.cache.get('b', self._loader)
        self.assertEqual(self.cache.clear(), 2)
        self.assertEqual(len(self.cache), 0)
        self.assertEqual(sorted(self.evicted), ['model:a', 'model:b'])

    def test_peek_does_not_load(self):
        self.assertIsNone(self.cache.peek('a'))
        self.assertEqual(self.loads, [])

    def test_evict_callback_error_is_swallowed(self):
        cache = IdleModelCache('boom', idle_seconds=1,
                               on_evict=lambda _m: (_ for _ in ()).throw(RuntimeError('x')),
                               time_fn=self.clock)
        cache.get('a', self._loader)
        self.clock.advance(2)
        self.assertEqual(cache.sweep(), 1)   # 예외가 새어나오면 안 됨

    def test_stats_shape(self):
        self.cache.get('a', self._loader)
        stats = self.cache.stats()
        self.assertEqual(stats['name'], 'test')
        self.assertEqual(stats['count'], 1)
        self.assertEqual(stats['keys'][0]['key'], 'a')

    def test_concurrent_get_returns_same_instance(self):
        """여러 스레드가 동시에 요청해도 결국 같은 객체를 쓴다 (락 밖 로딩이라
        중복 로딩 자체는 발생할 수 있지만, 저장되는 인스턴스는 하나여야 한다)."""
        results = []
        barrier = threading.Barrier(4)

        def worker():
            barrier.wait()
            results.append(self.cache.get('shared', lambda k: object()))

        threads = [threading.Thread(target=worker) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        self.assertEqual(len(set(id(r) for r in results)), 1)


class _Model:
    """weakref 가능한 가짜 모델 (문자열은 weakref가 안 된다)."""
    def __init__(self, key):
        self.key = key


class TestReleaseAfterReferencesDropped(unittest.TestCase):
    """회귀: empty_cache가 모델 참조가 살아 있는 동안 불려 VRAM이 reserved로 남던 문제.

    after_evict가 불리는 시점에 꺼낸 모델이 이미 죽어 있어야(weakref None) 한다.
    """

    def setUp(self):
        self.clock = _Clock()
        self.refs = {}
        self.on_evict_keys = []
        self.after_calls = []

    def _loader(self, key):
        model = _Model(key)
        self.refs[key] = weakref.ref(model)
        return model

    def _after(self):
        self.after_calls.append({k: r() is None for k, r in self.refs.items()})

    def _cache(self, **kwargs):
        options = dict(idle_seconds=60.0, max_items=3, time_fn=self.clock,
                       on_evict=lambda m: self.on_evict_keys.append(m.key),
                       after_evict=self._after)
        options.update(kwargs)
        return IdleModelCache('t', **options)

    def test_sweep_calls_after_evict_once_after_refs_dropped(self):
        cache = self._cache()
        cache.get('a', self._loader)
        cache.get('b', self._loader)
        self.clock.advance(61)
        self.assertEqual(cache.sweep(), 2)
        self.assertEqual(sorted(self.on_evict_keys), ['a', 'b'], "on_evict(model) 계약 유지")
        self.assertEqual(len(self.after_calls), 1, "after_evict는 배치당 한 번")
        self.assertEqual(self.after_calls[0], {'a': True, 'b': True},
                         "after_evict 시점에 모델 참조가 남아 있으면 empty_cache가 무력하다")

    def test_clear_releases_refs_before_after_evict(self):
        cache = self._cache()
        cache.get('a', self._loader)
        self.assertEqual(cache.clear(), 1)
        self.assertEqual(self.after_calls, [{'a': True}])

    def test_nothing_evicted_means_no_after_evict(self):
        cache = self._cache()
        self.assertEqual(cache.clear(), 0)
        self.assertEqual(cache.sweep(), 0)
        cache.get('a', self._loader)
        self.clock.advance(10)
        self.assertEqual(cache.sweep(), 0)
        self.assertEqual(self.after_calls, [], "비었을 때 gc/empty_cache 비용을 치르면 안 된다")

    def test_max_one_evicts_old_bundle_before_loading_new_key(self):
        """SAM3(max_items=1): 새 번들을 올리기 전에 옛 번들이 이미 풀려 있어야 한다(7GB 동시 상주 방지)."""
        cache = self._cache(max_items=1)
        cache.get('old', self._loader)
        seen = {}

        def loader(key):
            seen['old_dead'] = self.refs['old']() is None
            seen['released_before_load'] = len(self.after_calls)
            seen['len'] = len(cache)
            return self._loader(key)

        cache.get('new', loader)
        self.assertTrue(seen['old_dead'])
        self.assertEqual(seen['released_before_load'], 1)
        self.assertEqual(seen['len'], 0)
        self.assertEqual(cache.keys(), ('new',))
        self.assertEqual(self.on_evict_keys, ['old'])

    def test_capacity_eviction_runs_outside_the_lock(self):
        cache = self._cache(max_items=1)
        cache.get('a', self._loader)
        observed = []

        def after():
            # 다른 스레드가 락을 잡을 수 있어야 한다 — 락 안에서 gc.collect를 돌리면 멈춘다
            box = []
            probe = threading.Thread(target=lambda: box.append(cache.keys()))
            probe.start()
            probe.join(1.0)
            observed.append(not probe.is_alive() and bool(box))

        cache._after_evict = after
        cache.get('b', self._loader)
        self.assertEqual(observed, [True])

    def test_discarded_duplicate_load_is_released(self):
        cache = self._cache()
        first = _Model('first')

        def racing_loader(key):
            # 로딩하는 사이 다른 스레드가 같은 키를 먼저 넣은 상황
            cache.get(key, lambda _k: first)
            return self._loader(key)

        result = cache.get('k', racing_loader)
        self.assertIs(result, first)
        self.assertEqual(len(self.after_calls), 1)
        self.assertTrue(self.after_calls[0]['k'], "버린 중복본이 after_evict 전에 풀려야 한다")
        self.assertEqual(self.on_evict_keys, [], "캐시된 적 없는 중복본은 on_evict 대상이 아니다")

    def test_after_evict_error_is_swallowed(self):
        cache = self._cache(after_evict=lambda: (_ for _ in ()).throw(RuntimeError('x')))
        cache.get('a', self._loader)
        self.assertEqual(cache.clear(), 1)


class TestAutoSweep(unittest.TestCase):
    """회귀: sweep이 get() 안에서만 돌아 '90초 뒤 자동 반납'이 실제로는 일어나지 않던 문제."""

    def _make(self, idle=0.05):
        released = threading.Event()
        cache = IdleModelCache('auto', idle_seconds=idle, max_items=2,
                               after_evict=released.set, auto_sweep=True)
        self.addCleanup(cache.stop_auto_sweep)
        return cache, released

    def test_idle_entry_is_released_without_another_get(self):
        cache, released = self._make()
        cache.get('a', lambda k: _Model(k))
        self.assertTrue(released.wait(3.0), "다음 get 없이도 유휴 항목이 풀려야 한다")
        self.assertEqual(len(cache), 0)

    def test_reaper_sleeps_when_empty_and_wakes_on_next_load(self):
        cache, released = self._make()
        cache.get('a', lambda k: _Model(k))
        self.assertTrue(released.wait(3.0))
        released.clear()
        reaper = cache._reaper
        self.assertIsNotNone(reaper)
        self.assertTrue(reaper.is_alive(), "비어도 스레드는 잠든 채 남아 있어야 한다")
        cache.get('b', lambda k: _Model(k))
        self.assertTrue(released.wait(3.0), "다시 적재하면 reaper가 깨어나 만료시킨다")
        self.assertIs(cache._reaper, reaper, "스레드를 새로 만들지 않는다")

    def test_recent_use_is_not_released_early(self):
        cache, released = self._make(idle=30.0)
        cache.get('a', lambda k: _Model(k))
        self.assertFalse(released.wait(0.3))
        self.assertEqual(len(cache), 1)

    def test_auto_sweep_is_off_by_default(self):
        cache = IdleModelCache('manual', idle_seconds=0.01)
        cache.get('a', lambda k: _Model(k))
        self.assertIsNone(cache._reaper)

    def test_stop_auto_sweep_joins_thread(self):
        cache, _released = self._make(idle=30.0)
        cache.get('a', lambda k: _Model(k))
        reaper = cache._reaper
        cache.stop_auto_sweep()
        self.assertFalse(reaper.is_alive())
        cache.get('b', lambda k: _Model(k))
        self.assertIsNone(cache._reaper, "멈춘 뒤에는 다시 띄우지 않는다")


class _FakeCuda:
    def __init__(self, available, probe=None):
        self._available = available
        self.probe = probe
        self.calls = []

    def is_available(self):
        return self._available

    def empty_cache(self):
        self.calls.append(('empty_cache', self.probe() if self.probe else None))

    def ipc_collect(self):
        self.calls.append(('ipc_collect', None))


class TestReleaseTorchMemoryWithMockTorch(unittest.TestCase):
    """GPU 없이 — 가짜 torch 모듈로 empty_cache 호출 시점의 모델 생존 여부를 검증한다."""

    def _fake_torch(self, available=True, probe=None):
        return types.SimpleNamespace(cuda=_FakeCuda(available, probe))

    def test_empty_cache_and_ipc_collect_when_cuda_available(self):
        fake = self._fake_torch()
        with mock.patch.dict(sys.modules, {'torch': fake}):
            release_torch_memory()
        self.assertEqual([name for name, _ in fake.cuda.calls], ['empty_cache', 'ipc_collect'])

    def test_no_cuda_calls_without_cuda(self):
        fake = self._fake_torch(available=False)
        with mock.patch.dict(sys.modules, {'torch': fake}):
            release_torch_memory()
        self.assertEqual(fake.cuda.calls, [])

    def test_sam3_style_cache_frees_bundle_before_empty_cache(self):
        clock = _Clock()
        refs = []

        def load(key):
            bundle = (_Model(key), _Model(key + ':processor'))
            refs.extend(weakref.ref(part) for part in bundle)
            return bundle

        fake = self._fake_torch(probe=lambda: all(ref() is None for ref in refs))
        cache = IdleModelCache('sam3', idle_seconds=90.0, max_items=1, time_fn=clock,
                               after_evict=release_torch_memory)
        with mock.patch.dict(sys.modules, {'torch': fake}):
            model, processor = cache.get('sam3.pt|cuda', load)
            del model, processor                 # 호출자(_refine_with_sam3)의 지역 참조가 끝난 상태
            clock.advance(91)
            self.assertEqual(cache.sweep(), 1)
        self.assertEqual(fake.cuda.calls[0], ('empty_cache', True),
                         "empty_cache 시점에 번들이 살아 있으면 VRAM이 드라이버로 안 돌아간다")


class TestLease(unittest.TestCase):
    """회귀: 다른 스레드(생성 시작·reaper·용량 초과)가 추론 중인 SAM3 번들을 꺼내면
    after_evict(empty_cache)는 작업의 지역 참조 때문에 헛돌고, 작업이 끝나 모델이 죽어도
    다시 부를 곳이 없어 ~3.4GB가 reserved로 남았다.

    lease 동안에는 해제하지 않고 '반납 예약'만 → 마지막 lease가 끝날 때 on_evict + after_evict.
    """

    def setUp(self):
        self.clock = _Clock()
        self.refs = {}
        self.loads = []
        self.on_evict_keys = []
        self.after_calls = []

    def _loader(self, key):
        self.loads.append(key)
        model = _Model(key)
        self.refs[key] = weakref.ref(model)
        return model

    def _after(self):
        self.after_calls.append({k: r() is None for k, r in self.refs.items()})

    def _cache(self, **kwargs):
        options = dict(idle_seconds=60.0, max_items=3, time_fn=self.clock,
                       on_evict=lambda m: self.on_evict_keys.append(m.key),
                       after_evict=self._after)
        options.update(kwargs)
        return IdleModelCache('lease', **options)

    def test_clear_during_lease_releases_when_the_lease_ends_with_model_dead(self):
        cache = self._cache()
        with cache.lease('sam3', self._loader) as model:
            self.assertEqual(cache.clear(), 0, "사용 중인 번들은 지금 꺼내지 않는다(예약분은 개수에서 제외)")
            self.assertEqual(self.after_calls, [], "추론 중에 empty_cache를 헛돌리지 않는다")
            self.assertEqual(self.on_evict_keys, [])
            self.assertIs(cache.peek('sam3'), model, "작업은 같은 모델로 끝까지 돈다")
            self.assertTrue(cache.stats()['keys'][0]['release_pending'])
            del model                              # 빌린 쪽은 with가 끝나기 전에 참조를 놓는다
        self.assertEqual(self.on_evict_keys, ['sam3'])
        self.assertEqual(self.after_calls, [{'sam3': True}],
                         "lease가 끝날 때 한 번, 모델이 이미 죽은 뒤에 after_evict")
        self.assertEqual(len(cache), 0)

    def test_sweep_and_reaper_skip_leased_entries_and_lease_end_restarts_idle(self):
        cache = self._cache()
        with cache.lease('sam3', self._loader):
            self.clock.advance(600)                # 긴 CPU 작업
            self.assertEqual(cache.sweep(), 0, "사용 중인 항목을 유휴로 보고 만료시키면 안 된다")
            self.assertIsNone(cache._next_wait_locked(), "빌린 항목만 있으면 reaper는 잠든다")
        self.assertEqual(cache.sweep(), 0, "작업이 끝나자마자 만료되지 않는다")
        self.clock.advance(59)
        self.assertEqual(cache.sweep(), 0)
        self.clock.advance(2)
        self.assertEqual(cache.sweep(), 1)
        self.assertEqual(self.after_calls, [{'sam3': True}])

    def test_lease_without_release_request_keeps_model_cached(self):
        cache = self._cache()
        with cache.lease('sam3', self._loader):
            pass
        with cache.lease('sam3', self._loader):
            pass
        self.assertEqual(self.loads, ['sam3'], "연속 클릭은 캐시 히트")
        self.assertEqual(cache.keys(), ('sam3',))
        self.assertEqual(self.after_calls, [])
        self.assertEqual(cache._leases, {})

    def test_clear_while_lease_is_loading_releases_new_bundle_after_the_job(self):
        cache = self._cache()

        def loader(key):
            model = self._loader(key)
            cache.clear()                          # 로딩(수십 초) 도중 생성이 시작됐다
            return model

        with cache.lease('sam3', loader) as model:
            self.assertIs(cache.peek('sam3'), model)
            del model
        self.assertEqual(len(cache), 0, "생성이 도는 동안 90초씩 VRAM을 쥐고 있지 않는다")
        self.assertEqual(self.after_calls, [{'sam3': True}])

    def test_lease_joining_a_racing_load_after_clear_is_also_released(self):
        cache = self._cache()

        def racing(_key):
            model = _Model('racer')
            self.refs['racer'] = weakref.ref(model)
            return model

        def loader(key):
            cache.clear()                          # 생성 시작
            cache.get(key, racing)                 # 그 뒤 다른 스레드가 같은 키를 먼저 넣었다
            return self._loader(key)

        with cache.lease('sam3', loader) as model:
            self.assertEqual(model.key, 'racer', "먼저 들어간 인스턴스를 쓴다")
            del model
        self.assertEqual(len(cache), 0, "clear 뒤에 시작된 lease가 끝나면 반납")
        self.assertEqual(self.after_calls[-1], {'sam3': True, 'racer': True})

    def test_plain_get_ignores_clear_during_load(self):
        cache = self._cache()

        def loader(key):
            model = self._loader(key)
            cache.clear()
            return model

        cache.get('yolo', loader)
        self.assertEqual(cache.keys(), ('yolo',), "get()의 기존 의미는 그대로")

    def test_capacity_does_not_pull_a_leased_bundle(self):
        cache = self._cache(max_items=1)
        with cache.lease('old', self._loader) as model:
            cache.get('new', self._loader)
            self.assertEqual(set(cache.keys()), {'old', 'new'}, "사용 중인 옛 번들은 작업이 끝날 때까지 둔다")
            self.assertEqual(self.on_evict_keys, [])
            del model
        self.assertEqual(cache.keys(), ('new',))
        self.assertEqual(self.on_evict_keys, ['old'])
        self.assertEqual(self.after_calls, [{'old': True, 'new': False}])

    def test_second_lease_joins_pending_entry_and_last_one_releases(self):
        cache = self._cache()
        first = cache.lease('sam3', self._loader)
        first.__enter__()
        cache.clear()
        with cache.lease('sam3', self._loader) as again:
            self.assertEqual(self.loads, ['sam3'], "반납 예약된 번들도 재로딩 없이 이어 쓴다")
            first.__exit__(None, None, None)
            self.assertEqual(self.after_calls, [], "아직 다른 작업이 쓰는 중")
            self.assertEqual(len(cache), 1)
            del again
        self.assertEqual(len(cache), 0)
        self.assertEqual(self.after_calls, [{'sam3': True}])

    def test_exception_inside_lease_ends_it(self):
        cache = self._cache()
        with self.assertRaises(RuntimeError):
            with cache.lease('sam3', self._loader):
                raise RuntimeError('inference failed')
        self.assertEqual(cache._leases, {})
        self.clock.advance(61)
        self.assertEqual(cache.sweep(), 1, "끝난 lease는 다시 유휴 만료 대상")

    def test_loader_failure_leaves_no_lease(self):
        cache = self._cache()

        def broken(_key):
            raise OSError('checkpoint missing')

        with self.assertRaises(OSError):
            with cache.lease('sam3', broken):
                self.fail('loader가 실패하면 본문을 돌지 않는다')
        self.assertEqual(cache._leases, {})
        self.assertEqual(len(cache), 0)

    def test_stats_report_leases(self):
        cache = self._cache()
        with cache.lease('sam3', self._loader):
            entry = cache.stats()['keys'][0]
            self.assertEqual((entry['leases'], entry['release_pending']), (1, False))
        entry = cache.stats()['keys'][0]
        self.assertEqual((entry['leases'], entry['release_pending']), (0, False))


class TestLeaseWithReaper(unittest.TestCase):
    """실제 reaper 스레드: 빌린 항목만 남아도 바쁜 루프가 돌지 않고, lease가 끝나면 만료시킨다."""

    def test_reaper_waits_for_lease_then_expires(self):
        calls = {'now': 0}

        def counting_clock():
            calls['now'] += 1
            return time.monotonic()

        released = threading.Event()
        cache = IdleModelCache('reaper-lease', idle_seconds=0.05, max_items=1,
                               time_fn=counting_clock, after_evict=released.set, auto_sweep=True)
        self.addCleanup(cache.stop_auto_sweep)
        with cache.lease('sam3', lambda k: _Model(k)):
            self.assertFalse(released.wait(0.4), "사용 중에는 만료되지 않는다")
            self.assertLess(calls['now'], 200, "빌린 항목만 있을 때 reaper가 헛돌면 안 된다")
        self.assertTrue(released.wait(3.0), "lease가 끝나면 유휴 시간 뒤에 반납")
        self.assertEqual(len(cache), 0)


class TestGlobalCaches(unittest.TestCase):
    def test_global_caches_release_after_refs_and_auto_sweep(self):
        for cache in (model_cache.YOLO_CACHE, model_cache.SAM3_CACHE):
            with self.subTest(cache=cache.name):
                self.assertIs(cache._after_evict, model_cache.release_torch_memory)
                self.assertIsNone(cache._on_evict, "모델을 쥔 채 empty_cache를 부르는 on_evict 금지")
                self.assertTrue(cache._auto_sweep)

    def test_rembg_session_cache_releases_without_importing_torch(self):
        """rembg(ONNX) 세션 캐시 — gc 만 돌리고 torch 는 import 하지 않는다(감사 #60)."""
        cache = model_cache.REMBG_CACHE
        self.assertIs(cache._after_evict, model_cache.release_session_memory)
        self.assertIsNone(cache._on_evict)
        self.assertTrue(cache._auto_sweep)
        with mock.patch.dict('sys.modules', {'torch': None}), mock.patch('gc.collect') as collect:
            model_cache.release_session_memory()   # torch import 를 시도하면 ImportError
        collect.assert_called_once_with()

    def test_release_for_generation_clears_only_sam3(self):
        with mock.patch.object(model_cache.SAM3_CACHE, 'clear', return_value=1) as sam3, \
                mock.patch.object(model_cache.YOLO_CACHE, 'clear') as yolo, \
                mock.patch.object(model_cache.REMBG_CACHE, 'clear') as rembg:
            self.assertEqual(model_cache.release_for_generation(), 1)
        sam3.assert_called_once_with()
        yolo.assert_not_called()
        rembg.assert_not_called()

    def test_clear_all_clears_every_editor_cache(self):
        with mock.patch.object(model_cache.SAM3_CACHE, 'clear', return_value=1) as sam3, \
                mock.patch.object(model_cache.YOLO_CACHE, 'clear', return_value=2) as yolo, \
                mock.patch.object(model_cache.REMBG_CACHE, 'clear', return_value=4) as rembg:
            self.assertEqual(model_cache.clear_all(), 7)
        sam3.assert_called_once_with()
        yolo.assert_called_once_with()
        rembg.assert_called_once_with()


if __name__ == '__main__':
    unittest.main()
