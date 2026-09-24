# core/model_cache.py
"""무거운 모델(YOLO / SAM3)의 유휴 기반 캐시. 순수 로직이라 테스트 가능.

왜 필요한가
- YOLO: `ui/vue_bridge.py`가 auto_censor/auto_detect 때마다 루프 안에서 `YOLO(path)`를
  새로 만들었다. 등록 모델이 3개면 클릭 한 번에 3회 로딩.
- SAM3: `core/sam_refiner.py`가 `_SAM3_BUNDLE_CACHE`를 두고도 `finally`에서 매번
  `clear()` 해버려 캐시가 무력화됐다. 체크포인트가 3.45GB라 클릭마다 수십 초.

즉시 언로드는 VRAM을 아끼려던 의도였지만, 연속 작업에서는 매번 재로딩 비용을 물었다.
그래서 '쓰고 나서 N초 동안 아무도 안 쓰면 내린다'로 바꾼다. 연속 클릭은 캐시 히트,
자리를 비우면 알아서 반납.

실제로 반납되게 하는 두 가지 (예전 회귀: SAM3 3.4GB가 앱 종료까지 VRAM 점유)
- 유휴 만료는 `get()` 안에서만 돌았다 → 다음 클릭이 없으면 영영 안 내렸다.
  `auto_sweep=True`면 데몬 스레드가 만료 시각에 맞춰 sweep한다.
- `empty_cache`를 캐시에서 꺼낸 모델 참조를 쥔 채 불렀다 → 블록이 allocator에
  reserved로 남았다. 이제 참조를 모두 끊은 뒤 락 밖에서 `after_evict()`를 한 번 부른다.

사용 중인 모델은 빼 가지 않는다 (`lease()`)
- 생성 시작(release_for_generation)·reaper·용량 초과는 다른 스레드에서 온다. 추론 중인
  번들을 꺼내면 after_evict(empty_cache)는 작업의 지역 참조 때문에 아무것도 못 돌려주고,
  작업이 끝나 마지막 참조가 사라져도 다시 부를 곳이 없어 ~3.4GB가 reserved로 남았다.
- `with cache.lease(key, loader) as model:` 동안 그 항목은 sweep 대상이 아니고, clear·용량
  초과는 '반납 예약'만 한다. 마지막 lease가 끝날 때 on_evict + after_evict를 부른다.
  lease가 끝나면 유휴 시계도 다시 잰다(긴 CPU 작업이 끝나자마자 만료되지 않게).

이 모듈은 torch/ultralytics를 import하지 않는다 — 로더는 호출자가 콜백으로 준다.
"""
import threading
import time
from contextlib import contextmanager

# 기본 유휴 시간. SAM3는 로딩이 비싸고(수십 초) 덩치도 커서(3.5GB) 짧게,
# YOLO는 가벼워서(수십 MB) 길게 잡는다.
DEFAULT_IDLE_SECONDS = 90.0

# 자동 sweep 스레드: 만료 시각을 조금 넘겨 깨어나고(경계에서 헛돌지 않게),
# 시계가 어긋나도 최소 이 간격으로는 다시 계산한다.
_REAPER_SLACK_SECONDS = 0.05
_REAPER_MAX_WAIT_SECONDS = 60.0


class IdleModelCache:
    """key → 모델. `idle_seconds` 동안 접근이 없으면 해제한다.

    스레드 안전. `get()`은 캐시에 없으면 loader(key)로 만들고 저장한다.

    해제 순서 — VRAM이 실제로 드라이버에 돌아가게:
      1) 캐시에서 꺼낸 항목마다 `on_evict(model)` (모델을 넘기는 기존 계약)
      2) 캐시·리스트·지역 변수가 쥔 모델 참조를 모두 끊는다
      3) 락 밖에서 인자 없는 `after_evict()`를 **한 번** (예: release_torch_memory)
    꺼낸 것이 없으면 after_evict를 부르지 않는다 (gc/empty_cache 비용 없음).

    `auto_sweep=True`면 첫 적재 때 데몬 스레드를 띄워 가장 오래된 항목의 만료 시각에
    sweep한다. 캐시가 비면 다음 적재까지 잠든다. 가짜 시계 테스트를 위해 기본은 끈다.

    `lease(key, loader)`로 빌린 항목은 빌린 동안 해제되지 않는다. 그 사이 clear()나 용량
    초과가 오면 '반납 예약(release pending)'으로 표시하고, 마지막 lease가 끝날 때 해제한다.
    예약된 항목을 다른 lease가 이어 빌리면 같은 모델을 쓰고(재로딩 없음) 예약은 유지된다.
    lease가 로딩하는 사이 clear()가 왔으면 새로 올린 모델도 반납 예약한다.
    빌린 쪽은 with 블록이 끝나기 **전에** 모델 참조를 놓아야 한다 — 그래야 해제 시점의
    after_evict(empty_cache)가 블록을 드라이버로 돌려준다.
    """

    def __init__(self, name: str, idle_seconds: float = DEFAULT_IDLE_SECONDS,
                 max_items: int = 4, on_evict=None, time_fn=time.monotonic,
                 after_evict=None, auto_sweep: bool = False):
        self.name = name
        self.idle_seconds = float(idle_seconds)
        self.max_items = max(1, int(max_items))
        self._on_evict = on_evict
        self._after_evict = after_evict
        self._now = time_fn
        self._lock = threading.RLock()
        self._wake = threading.Condition(self._lock)
        self._auto_sweep = bool(auto_sweep)
        self._reaper = None
        self._reaper_stop = False
        self._items = {}        # key → (model, last_used)
        self._leases = {}       # key → 진행 중인 lease 수 (0이면 키 없음)
        self._pending = set()   # lease가 끝나면 해제할 키 (사용 중에 clear·용량 초과가 옴)
        self._clear_epoch = 0   # clear() 횟수 — lease 로딩 중에 clear가 왔는지 판별

    # ── 조회 ────────────────────────────────────────────────────────────────
    def get(self, key, loader):
        """캐시에서 꺼내거나 loader(key)로 만든다.

        loader는 락 **밖에서** 호출한다 — 모델 로딩이 수십 초라 락을 잡고 있으면
        다른 스레드(썸네일 생성 등)가 통째로 멈춘다.

        새 키를 올리기 **전에** 용량을 먼저 비운다. max_items=1(SAM3 3.4GB)에서 옛 번들과
        새 번들이 VRAM에 동시에 올라가는 순간(약 7GB)을 없애기 위해서다.
        (loader가 실패하면 비운 항목은 다음 사용 때 다시 올린다.)

        돌려준 모델은 '사용 중'으로 표시되지 않는다 — 다른 스레드의 clear/reaper가 쓰는
        도중에 꺼낼 수 있다. 무거운 모델을 오래 쓰면 `lease()`를 쓴다.
        """
        return self._acquire(key, loader, leased=False)

    @contextmanager
    def lease(self, key, loader):
        """get()처럼 꺼내거나 만들되, with 블록 동안 '사용 중'으로 표시한다.

        빌린 동안에는 reaper·clear·용량 초과가 이 항목을 해제하지 않는다(clear·용량 초과는
        반납 예약만). 마지막 lease가 끝날 때 예약돼 있으면 on_evict + after_evict로 해제하고,
        아니면 유휴 시계를 지금부터 다시 잰다. with 블록이 끝나기 전에 모델 참조를 놓아야
        after_evict(empty_cache)가 실제로 블록을 돌려준다.
        """
        model = self._acquire(key, loader, leased=True)
        try:
            yield model
        finally:
            model = None            # 제너레이터 프레임이 모델을 쥔 채 해제하지 않게
            self._end_lease(key)

    def _acquire(self, key, loader, leased: bool):
        self.sweep()
        with self._lock:
            entry = self._items.get(key)
            if entry is not None:
                self._items[key] = (entry[0], self._now())
                if leased:
                    self._leases[key] = self._leases.get(key, 0) + 1
                return entry[0]
            evicted = self._pop_over_capacity_locked(self.max_items - 1)
            epoch = self._clear_epoch
        self._release(evicted)

        model = loader(key)

        with self._lock:
            # 그 사이 다른 스레드가 먼저 넣었으면 그걸 쓴다 (중복 로딩 1회는 감수)
            existing = self._items.get(key)
            if existing is not None:
                result, discarded = existing[0], True
                self._items[key] = (existing[0], self._now())
                evicted = []
            else:
                self._items[key] = (model, self._now())
                result, discarded = model, False
                evicted = None
            if leased:
                self._leases[key] = self._leases.get(key, 0) + 1
                if self._clear_epoch != epoch:
                    # 로딩하는 사이 clear()(생성 시작·수동 언로드)가 왔다 — 이 작업만
                    # 끝내고 바로 반납한다. 90초 동안 VRAM을 쥐고 있지 않게.
                    self._pending.add(key)
            if evicted is None:
                evicted = self._pop_over_capacity_locked(self.max_items)
                self._wake_reaper_locked()
            existing = None
        # 버린 중복본은 여기서 마지막 참조가 끊긴다 — 그 뒤에 캐시 반납
        model = None
        if discarded:
            self._after_release()
        self._release(evicted)
        return result

    def _end_lease(self, key):
        """lease 하나를 끝낸다. 마지막이고 반납 예약돼 있으면 해제(락 밖에서)."""
        evicted = []
        with self._lock:
            remaining = self._leases.get(key, 0) - 1
            if remaining > 0:
                self._leases[key] = remaining
                return
            self._leases.pop(key, None)
            entry = self._items.get(key)
            if key in self._pending:
                self._pending.discard(key)
                if entry is not None:
                    evicted.append(self._items.pop(key)[0])
            elif entry is not None:
                # 긴 작업(예: CPU SAM3 + exclude 여러 개)이 끝나자마자 만료되지 않게
                self._items[key] = (entry[0], self._now())
                self._wake_reaper_locked()
            entry = None
        self._release(evicted)

    def peek(self, key):
        with self._lock:
            entry = self._items.get(key)
            return entry[0] if entry else None

    def touch(self, key):
        with self._lock:
            entry = self._items.get(key)
            if entry:
                self._items[key] = (entry[0], self._now())

    # ── 해제 ────────────────────────────────────────────────────────────────
    def sweep(self) -> int:
        """유휴 시간이 지난 항목 해제. 빌려 간(lease) 항목은 건너뛴다. 반환: 해제한 개수."""
        cutoff = self._now() - self.idle_seconds
        with self._lock:
            stale = [k for k, (_m, used) in self._items.items()
                     if not self._leases.get(k) and (used <= cutoff or k in self._pending)]
            evicted = [self._items.pop(k)[0] for k in stale]
            self._pending.difference_update(stale)
        return self._release(evicted)

    def clear(self) -> int:
        """전부 해제. 비어 있으면 아무것도 하지 않는다(after_evict도 안 부름).

        빌려 간(lease) 항목은 지금 꺼내지 않고 반납 예약한다 — 마지막 lease가 끝날 때
        해제된다. 반환: 지금 해제한 개수(예약분 제외).
        """
        with self._lock:
            self._clear_epoch += 1
            evicted = []
            for key in list(self._items):
                if self._leases.get(key):
                    self._pending.add(key)
                else:
                    evicted.append(self._items.pop(key)[0])
                    self._pending.discard(key)
        return self._release(evicted)

    def _pop_over_capacity_locked(self, limit: int) -> list:
        """반납 예약되지 않은 항목이 limit 이하가 될 때까지 가장 오래 안 쓴 것부터 꺼낸다.

        빌려 간(lease) 항목은 꺼내지 않고 반납 예약만 한다(lease가 끝날 때 해제).
        꺼내기만 한다 — on_evict/after_evict(gc·empty_cache)는 호출자가 락 밖에서
        `_release()`로 처리한다. 락을 쥔 채 gc.collect를 돌리지 않기 위해서다.
        """
        evicted = []
        limit = max(0, int(limit))
        active = [k for k in self._items if k not in self._pending]
        while len(active) > limit:
            oldest = min(active, key=lambda k: self._items[k][1])
            active.remove(oldest)
            if self._leases.get(oldest):
                self._pending.add(oldest)
            else:
                evicted.append(self._items.pop(oldest)[0])
        return evicted

    def _release(self, models: list) -> int:
        """꺼낸 모델마다 on_evict → 모든 참조를 끊고 → after_evict 1회. 락 밖에서만 부른다.

        `models` 리스트는 제자리에서 비운다 — 호출자 쪽 리스트가 모델을 붙잡고 있으면
        after_evict(empty_cache)가 그 블록을 드라이버로 돌려주지 못한다.
        """
        count = len(models)
        if not count:
            return 0
        models.reverse()                 # pop()이 원래 순서대로 꺼내도록
        while models:
            model = models.pop()
            self._evict(model)
            del model
        self._after_release()
        return count

    def _evict(self, model):
        if self._on_evict is None:
            return
        try:
            self._on_evict(model)
        except Exception:
            pass

    def _after_release(self):
        if self._after_evict is None:
            return
        try:
            self._after_evict()
        except Exception:
            pass

    # ── 자동 sweep (유휴 만료를 실제로 일으킨다) ───────────────────────────
    def _wake_reaper_locked(self):
        """auto_sweep이면 reaper를 (필요 시 띄우고) 깨운다. 락을 쥔 상태에서 부른다."""
        if not self._auto_sweep:
            return
        if self._reaper is None or not self._reaper.is_alive():
            self._reaper_stop = False
            self._reaper = threading.Thread(
                target=self._reap_loop, name=f"model-cache-{self.name}-reaper", daemon=True)
            self._reaper.start()
        self._wake.notify_all()

    def _next_wait_locked(self):
        """다음 만료까지 남은 초. 만료될 수 있는 항목이 없으면 None(무기한 대기),
        이미 지났으면 0 이하.

        빌려 간(lease) 항목은 셈에서 뺀다 — 그 항목만 남았을 때 '이미 만료'로 계산해
        sweep을 헛돌리는 바쁜 루프를 막는다. lease가 끝나면 _end_lease가 깨운다.
        """
        idle = [used for key, (_m, used) in self._items.items() if not self._leases.get(key)]
        if not idle:
            return None
        return min(idle) + self.idle_seconds - self._now()

    def _reap_loop(self):
        while True:
            with self._wake:
                if self._reaper_stop:
                    return
                delay = self._next_wait_locked()
                if delay is None:
                    self._wake.wait()            # 비었으면 get()이 깨울 때까지 잠든다
                    continue
                if delay > 0:
                    self._wake.wait(timeout=min(delay + _REAPER_SLACK_SECONDS,
                                                _REAPER_MAX_WAIT_SECONDS))
                    continue
            try:
                self.sweep()                     # on_evict/after_evict는 락 밖에서 돈다
            except Exception:
                with self._wake:
                    if not self._reaper_stop:
                        self._wake.wait(timeout=1.0)

    def stop_auto_sweep(self, timeout: float = 2.0):
        """reaper 스레드를 멈춘다(테스트·종료용). 이후 적재해도 다시 띄우지 않는다."""
        with self._wake:
            self._auto_sweep = False
            self._reaper_stop = True
            thread, self._reaper = self._reaper, None
            self._wake.notify_all()
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout)

    # ── 상태 ────────────────────────────────────────────────────────────────
    def __len__(self):
        with self._lock:
            return len(self._items)

    def keys(self):
        with self._lock:
            return tuple(self._items)

    def stats(self) -> dict:
        with self._lock:
            now = self._now()
            return {
                'name': self.name,
                'count': len(self._items),
                'idle_seconds': self.idle_seconds,
                'auto_sweep': self._auto_sweep,
                'keys': [
                    {'key': str(k), 'idle': round(now - used, 1),
                     'leases': self._leases.get(k, 0), 'release_pending': k in self._pending}
                    for k, (_m, used) in self._items.items()
                ],
            }


def release_torch_memory():
    """CUDA 캐시 반납. torch가 없으면 조용히 무시."""
    import gc
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            try:
                torch.cuda.ipc_collect()
            except Exception:
                pass
    except Exception:
        pass


def release_session_memory():
    """ONNX Runtime 세션(rembg) 반납 — 참조가 끊긴 세션을 gc로 정리한다.

    torch 캐시는 건드리지 않는다. rembg만 쓴 세션에서 반납하려고 torch를 import하면(수 초,
    수백 MB) 오히려 비싸진다. onnxruntime은 InferenceSession이 소멸할 때 자기 메모리를
    (CUDA provider를 포함해) 스스로 돌려준다.
    """
    import gc
    gc.collect()


# ── 전역 캐시 인스턴스 ──────────────────────────────────────────────────────
# release_torch_memory는 on_evict(모델 참조가 살아 있는 시점)가 아니라
# after_evict(참조를 모두 끊은 뒤)로 건다 — 그래야 empty_cache가 블록을 돌려준다.

# YOLO: 가볍고(수십 MB) 자주 쓰임 → 길게 잡는다
YOLO_CACHE = IdleModelCache('yolo', idle_seconds=300.0, max_items=8,
                            after_evict=release_torch_memory, auto_sweep=True)

# SAM3: 3.45GB. 연속 작업(클릭 여러 번)은 캐시 히트로 받고,
# 손을 떼면 90초 뒤 reaper 스레드가 VRAM을 돌려준다. 예전처럼 매번 clear() 하지 않는다.
SAM3_CACHE = IdleModelCache('sam3', idle_seconds=90.0, max_items=1,
                            after_evict=release_torch_memory, auto_sweep=True)

# rembg 배경 제거 세션(u2net ONNX, 약 170MB) — core/bg_removal.py가 lease로 빌린다.
# 예전 Vue 경로는 세션 없이 rembg.remove를 불러 클릭마다 새 세션을 만들었다.
# 연속 작업은 캐시 히트, 손을 떼면 3분 뒤 반납. 생성 시작 때는 YOLO처럼 남긴다.
REMBG_CACHE = IdleModelCache('rembg', idle_seconds=180.0, max_items=2,
                             after_evict=release_session_memory, auto_sweep=True)


def release_for_generation() -> int:
    """생성이 GPU 리스를 잡기 직전 — 무거운 SAM3 번들만 즉시 반납한다.

    core.resource_coordinator의 공유 코디네이터가 reserve() 때, 리스를 잡지 않는
    Forge 후처리 워커(SAM3·ADetailer·Refine·업스케일)는 작업 직전에 부른다. Forge/ComfyUI는
    별도 프로세스라 이 프로세스가 쥔 CUDA 캐시를 쓸 수 없다(ANIMA+SAM3+Forge 16GB OOM).
    비어 있으면 아무것도 하지 않는다. YOLO(수십 MB)는 다음 편집을 위해 남긴다.
    편집기가 지금 SAM3로 추론 중이면(lease) 그 작업이 끝나는 즉시 반납된다.
    반환: 지금 해제한 개수.
    """
    return SAM3_CACHE.clear()


def clear_all() -> int:
    """VRAM 게이지의 수동 언로드 — 앱이 쥔 편집기 비전 모델(YOLO·SAM3·rembg 세션)을 전부 반납."""
    return YOLO_CACHE.clear() + SAM3_CACHE.clear() + REMBG_CACHE.clear()
