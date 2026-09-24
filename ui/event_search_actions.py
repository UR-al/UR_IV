# ui/event_search_actions.py
"""Event Gen 검색 — 데이터 적재와 검색 워커의 요청 순서 관리 (GeneratorMainUI 믹스인).

한 번에 적재 하나, 검색 하나만 돈다.

- 적재 중에 온 검색 요청은 pending 으로 모아 두었다가, 적재가 끝나면 **마지막 요청**을
  다시 판단한다: 등급이 같으면 바로 검색하고, 바뀌었으면 그 등급을 새로 적재한다.
  같은 등급 parquet 을 두 번 동시에 읽어 메모리를 두 배로 쓰지 않는다.
- 검색 중에 새 요청이 오면 이전 워커를 취소하고, 요청 번호가 지난 결과·진행도는
  Vue 로 보내지 않는다. 늦게 끝난 옛 결과가 새 결과를 덮어쓰지 않게 한다.
- 적재 진행 문구는 Search 탭 전용 searchStatus 가 아니라 eventLoadStatus 로 보낸다.
  두 뷰가 keep-alive 로 함께 살아 있어, 공용 채널이면 서로의 로딩 문구를 덮어썼다.
- 적재·검색 워커는 메인 창을 Qt 부모로 두고, **스레드가 끝나면 deleteLater** 한다.
  부모가 있어야 참조를 바꿔 끼울 때 아직 끝나는 중인 QThread 가 파괴되지 않고
  (`QThread: Destroyed while thread is still running`), 끝난 뒤 지워야 창의 자식으로
  쌓이며 옛 등급 적재본(수백 MB)을 붙들지 않는다.

Qt 객체는 메서드 안에서만 만든다 — 모듈 자체는 Qt 없이 import 된다(테스트 하네스용).
"""
from __future__ import annotations

import json
from functools import partial

EVENT_RATINGS = ('g', 's', 'q', 'e')


def thread_finished_signal(thread):
    """QThread 자체의 finished() 바운드 시그널.

    EventDataLoadWorker 처럼 서브클래스가 같은 이름의 커스텀 시그널(`finished(object)`)을
    두면 `worker.finished` 로는 스레드 종료를 받을 수 없다 — 기반 클래스 디스크립터로 묶는다.
    """
    from PyQt6.QtCore import QThread

    return QThread.finished.__get__(thread, type(thread))


def normalize_event_ratings(raw) -> tuple[str, ...]:
    """Vue payload 의 ratings → 알려진 등급만 남긴 튜플 (비면 ('g',))."""
    if not isinstance(raw, (list, tuple)):
        return ('g',)
    ratings = tuple(
        rating for rating in raw
        if isinstance(rating, str) and rating in EVENT_RATINGS
    )
    return ratings or ('g',)


class EventSearchActionsMixin:
    """`search_events` 액션의 적재·검색 흐름."""

    def _start_event_search(self, payload):
        """이벤트 검색을 비동기 워커로 실행 (진행도 포함)"""
        if not isinstance(payload, dict):
            payload = {}
        ratings = normalize_event_ratings(payload.get('ratings', ['g']))

        loader = getattr(self, '_event_loader', None)
        loaded_ratings = getattr(self, '_event_loader_ratings', ())

        if loader is not None and frozenset(loaded_ratings) == frozenset(ratings):
            self._run_event_search_worker(loader, payload)
            return

        # 데이터 자동 로드 후 검색. 이 요청이 가장 새 것이므로, 돌고 있던 검색 결과는 낡았다.
        self._pending_event_payload = payload
        self._pending_event_ratings = ratings
        self._begin_event_search_request()
        if getattr(self, '_event_loading_ratings', None) is not None:
            # 이미 적재 중 — 끝나면 _on_event_data_loaded 가 위 pending 을 다시 판단한다
            return

        # 다른 등급으로 바꾸는 중이면 옛 적재본을 먼저 놓아 둘을 동시에 들고 있지 않게 한다
        self._event_loader = None
        self._event_loader_ratings = ()
        self._event_loading_ratings = ratings
        try:
            self._auto_load_event_data(ratings)
        except Exception as exc:
            self._event_loading_ratings = None
            self._pending_event_payload = {}
            self._pending_event_ratings = ()
            self.vue_bridge.eventSearchResults.emit(
                json.dumps({'error': f'이벤트 데이터 로드를 시작하지 못했습니다: {exc}'})
            )

    def _auto_load_event_data(self, ratings):
        """이벤트 데이터 자동 로드 (Vue 진행도 표시)"""
        from config import EVENT_PARQUET_DIR
        from workers.event_data_load_worker import EventDataLoadWorker

        # 첫 문구 '데이터 로딩 중...' 은 워커가 바로 보내고 Vue 도 같은 문구로 시작한다.
        # 부모를 둔다: 적재 중 등급이 바뀌면 _on_event_data_loaded 안에서 곧바로 새 적재를
        # 시작해 이 참조를 바꿔 끼우는데, 그때 옛 워커는 run() 을 막 빠져나오는 중일 수 있다.
        # 부모가 없으면 파이썬 참조가 사라지는 순간 실행 중인 QThread 가 파괴되어 앱이 죽는다.
        worker = EventDataLoadWorker(EVENT_PARQUET_DIR, ratings, self)
        worker.progress.connect(self.vue_bridge.eventLoadStatus)   # 시그널→시그널 (스레드 간 queued)
        worker.finished.connect(self._on_event_data_loaded)        # 커스텀 finished(object) = 적재 결과
        self._release_event_thread_when_done(worker, '_event_load_worker')
        self._event_load_worker = worker
        worker.start()

    def _on_event_data_loaded(self, result):
        """데이터 로드 완료 → 마지막 검색 요청 처리"""
        loaded_ratings = getattr(self, '_event_loading_ratings', None)
        self._event_loading_ratings = None

        if isinstance(result, str):
            self._pending_event_payload = {}
            self._pending_event_ratings = ()
            self.vue_bridge.eventSearchResults.emit(json.dumps({'error': result}))
            return

        if loaded_ratings is None:
            loaded_ratings = tuple(getattr(self, '_pending_event_ratings', ()) or ('g',))
        self._event_loader = result
        self._event_loader_ratings = tuple(loaded_ratings)
        self._notify_stale_event_shards(result)

        payload = getattr(self, '_pending_event_payload', {})
        self._pending_event_payload = {}
        self._pending_event_ratings = ()
        if payload:
            # 적재 중에 등급이 바뀌었으면 여기서 그 등급을 다시 적재한다
            self._start_event_search(payload)

    def _notify_stale_event_shards(self, loader) -> None:
        """manifest 와 크기가 다른 Event shard 를 조용히 쓰지 않도록 알린다."""
        stale = list(getattr(loader, 'stale_shards', None) or ())
        if not stale:
            return
        names = ', '.join(stale)
        self.vue_bridge.showNotification.emit(
            'warning',
            f'Event 데이터가 dataset_manifest 와 다릅니다(구버전일 수 있음): {names} '
            f'— 앱을 다시 시작하면 새 데이터를 받습니다.',
        )

    def _begin_event_search_request(self) -> int:
        """새 요청 번호를 발급하고 돌고 있는 검색을 취소한다 (그 결과는 버려진다)."""
        request_id = int(getattr(self, '_event_search_request_id', 0) or 0) + 1
        self._event_search_request_id = request_id
        worker = getattr(self, '_event_search_worker', None)
        if worker is not None:
            try:
                if worker.isRunning():
                    worker.cancel()
            except RuntimeError:
                # C++ 쪽 QThread 가 이미 정리됨
                pass
            # 밀려난 워커는 더 붙들지 않는다 — Qt 부모(메인 창)가 스레드가 끝날 때까지 살려 두고,
            # 끝나면 _on_event_thread_done 이 지운다. 등급 전환(적재 경로)에서 이 참조가 남아
            # 있으면 옛 적재본을 놓아도 워커를 통해 계속 상주했다.
            self._event_search_worker = None
        return request_id

    def _is_current_event_search(self, request_id: int) -> bool:
        return request_id == getattr(self, '_event_search_request_id', None)

    def _run_event_search_worker(self, loader, payload):
        """검색 워커 실행"""
        from workers.event_search_worker import EventSearchWorker

        request_id = self._begin_event_search_request()
        worker = EventSearchWorker(loader, payload, self)
        worker.progress.connect(partial(self._on_event_search_progress, request_id))
        worker.search_finished.connect(partial(self._on_event_search_finished, request_id))
        self._release_event_thread_when_done(worker, '_event_search_worker')
        self._event_search_worker = worker
        worker.start()
        self.show_status("이벤트 검색 시작...")

    def _release_event_thread_when_done(self, worker, attr: str) -> None:
        """`worker` 스레드가 끝나면 `attr` 참조를 비우고 C++ 객체를 지운다 (메인 스레드에서)."""
        from PyQt6.QtCore import Qt

        # QThread.finished 는 끝나는 스레드에서 나온다 — 속성 정리·deleteLater 는 메인 스레드로.
        # 결과 시그널(search_finished / finished(object))은 run() 안에서 먼저 큐에 들어가므로
        # 결과 처리가 지우기보다 앞선다.
        thread_finished_signal(worker).connect(
            partial(self._on_event_thread_done, attr, worker),
            Qt.ConnectionType.QueuedConnection,
        )

    def _on_event_thread_done(self, attr: str, worker) -> None:
        if getattr(self, attr, None) is worker:
            setattr(self, attr, None)
        try:
            worker.deleteLater()
        except RuntimeError:
            # 부모(메인 창)와 함께 이미 정리됨
            pass

    def _on_event_search_progress(self, request_id: int, current: int, total: int) -> None:
        if self._is_current_event_search(request_id):
            self.vue_bridge.eventSearchProgress.emit(current, total)

    def _on_event_search_finished(self, request_id: int, result_json: str) -> None:
        if not self._is_current_event_search(request_id):
            # 새 요청에 밀린 검색 — Vue 의 최신 결과를 덮어쓰지 않게 버린다
            return
        self.vue_bridge.eventSearchResults.emit(result_json)
