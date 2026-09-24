# widgets/queue_manager.py
"""
대기열 실행 관리 — 맨 앞 항목을 하나씩 생성 요청하고, 끝나면 **그 항목**을 지우고 다음으로 간다.

- 보낸 항목의 id(``_inflight_id``)를 기억해 완료 때 그 항목만 지운다(``consume_item``). 예전엔
  완료 때 무조건 맨 앞 항목을 지워, 실행 중 항목이 먼저 지워졌으면(보호 가드가 무력했다) 생성하지
  않은 다음 항목이 사라졌고, 일시정지 중 수동 생성이 끝나도 대기열 항목이 하나 없어졌다.
- '실행 중' 표시는 이 매니저의 이름(``QUEUE_OWNER``)으로 걸고 푼다. 자동화 '큐 우선'
  (``AUTOMATION_OWNER``)이 생성 중인 항목은 보내지도, 그 표시를 풀지도 않는다. 자동화가 도는 동안엔
  아예 시작하지 않는다(호스트가 넣는 ``start_blocker`` — 자동화가 대기열을 먼저 처리한다).
- 워커가 이미 다른 생성(수동 생성 · 자동화)을 하고 있으면 보내지 않고 일시정지한다(``notice`` 로
  사유를 알린다). 예전엔 보냈다가 start_generation 이 거절했고, 그 전에 항목 값을 UI 에 덮어썼다.
- 중지는 생성을 취소하지 않는다. 그때 GPU 에 있던 항목(``_orphan_id``)은 표시를 둔 채 기억해 두고
  - 곧바로 다시 시작하면 다시 보내지 않고 이어받는다(예전엔 같은 항목을 다시 보냈다가 거절돼
    일시정지했고, 원래 생성이 끝나도 추적을 잃어 재개 때 같은 장을 한 번 더 만들었다),
  - 멈춘 채 끝나면 만들어진 항목은 지우고(다음 시작 때 같은 장을 또 만들지 않게) 실패·취소면
    표시만 푼다(항목은 남는다).
- 다음 장까지의 지연 타이머는 실행 회차(``_run_epoch``)를 기억한다 — 기다리는 사이 중지→시작하면
  옛 타이머는 무시된다(예전엔 새 회차가 이미 보낸 항목을 한 번 더 보냈다).
- 요청이 백엔드에 닿지 못했으면(동기 디스패치 실패 · ``_queue_deferred``) 일시정지하고 보낸 표시를
  푼다 — 항목은 그대로 남아 재개 때 다시 나간다.
- 정기 정리(``cleanup_every_n``): N장마다 체크포인트를 VRAM 에서 내린다. 백엔드 공용 언로드
  (core.post_generation — Forge unload-checkpoint · ComfyUI /free)를 데몬 스레드로 보내고, 다음
  생성 워커가 run() 초입에서 그 요청을 기다린다(UI 스레드는 막지 않는다). 예전 ``cleanup_models`` 는
  LoRA 재스캔과 options POST 뿐이라 VRAM 을 회수하지 못했고 Forge config.json 을 매번 다시 썼다.
"""
import logging
from typing import Callable, Optional

from PyQt6.QtCore import QObject, QTimer, pyqtSignal

from core.queue_model import QUEUE_OWNER, periodic_unload_due

_logger = logging.getLogger(__name__)

#: 워커가 다른 생성(수동 생성 등)을 하고 있어 보내지 못하고 멈췄을 때
BUSY_NOTICE = '이미지 생성 중이라 대기열을 일시정지했습니다 — 현재 생성이 끝난 뒤 재개하세요'
#: 자동화 '큐 우선'이 대기열 항목을 생성하고 있어 멈췄을 때
AUTOMATION_NOTICE = '자동화가 대기열 항목을 생성하고 있어 대기열을 일시정지했습니다'


class QueueManager(QObject):
    """대기열 관리자"""

    # 시그널
    generation_requested = pyqtSignal(dict)
    queue_completed = pyqtSignal(int)

    # 일시정지 상태 변경 시그널 — UI 동기화용
    paused_changed = pyqtSignal(bool)

    # 대기열이 스스로 일시정지한 이유 — 호스트가 상태줄·토스트로 알린다
    notice = pyqtSignal(str)

    def __init__(self, queue_panel, parent=None):
        super().__init__(parent)
        self.queue_panel = queue_panel
        self.is_running = False
        self.is_paused = False  # ⏸ 일시정지 — 큐는 유지되고 _process_next만 멈춤
        self.generated_count = 0
        self.total_count = 0
        self.delay_seconds = 1.0
        # 정기 정리 주기 — N장마다 체크포인트 언로드. 0이면 끔. Vue 자동화 패널이
        # set_automation_settings 로 동기화한다.
        self.cleanup_every_n = 0
        self.last_stop_natural = True
        # 호스트가 넣는다: 생성 워커가 아직 결과를 내지 않았는가(= 보낸 항목이 GPU 에 있다).
        # 재개 때 이미 돌고 있는 항목을 두 번 보내지 않게 쓴다. 없으면 늘 다시 보낸다.
        self.generation_active: Optional[Callable[[], bool]] = None
        # 호스트가 넣는다: 지금 시작하면 안 되는 이유(빈 문자열이면 시작 가능). 자동화가 돌면 자동화가
        # 대기열을 먼저 처리한다 — 두 소비자가 같은 항목과 워커를 다투지 않게 한다.
        self.start_blocker: Optional[Callable[[], str]] = None
        # 마지막 start() 가 거절된 이유 — 빈 대기열이면 빈 문자열
        self.last_start_refusal = ''

        self._success_count: int = 0
        self._fail_count: int = 0
        # 생성을 요청했고 아직 결과가 오지 않은 항목의 id
        self._inflight_id = None
        # 중지했지만 워커가 아직 만들고 있는 항목의 id — 다시 시작하면 이어받고, 멈춘 채 끝나면 정리한다
        self._orphan_id = None
        # 실행 회차 — 시작·중지마다 는다. 옛 회차의 지연 타이머가 새 회차에 끼어들지 않게 한다
        self._run_epoch = 0

    def start(self) -> bool:
        """대기열 실행 시작. 시작하지 못하면 False — 사유는 ``last_start_refusal``(빈 대기열이면 빈 문자열).

        이미 돌고 있으면 새로 추가된 항목만 총계에 합치고(보낸 항목을 다시 보내지 않는다) True.
        자동화가 돌고 있으면(``start_blocker``) 시작하지 않는다 — 자동화가 '큐 우선'으로 대기열을
        먼저 처리한다. 예전엔 둘이 같은 항목을 보내고 서로의 '생성 중' 표시를 풀었다.
        """
        self.last_start_refusal = ''
        if self.is_running:
            self.total_count = max(self.total_count, self.generated_count + self.queue_panel.count())
            return True
        refusal = self._start_refusal()
        if refusal:
            self.last_start_refusal = refusal
            return False
        if self.queue_panel.is_empty():
            # 예전엔 빈 대기열로도 '실행 중'이 되어, 이후 추가한 항목이 영영 시작되지 않았다
            return False

        self.is_running = True
        self._run_epoch += 1
        self.generated_count = 0
        self.total_count = self.queue_panel.count()
        self._success_count = 0
        self._fail_count = 0
        self._inflight_id = None
        self._process_next()
        return True

    def stop(self, natural: bool = False):
        """자동화 중지. natural=True면 큐를 다 소진한 '정상 완료', False면 사용자 수동 중지.
        (수동 중지일 때 '완료' 팝업/성공 알림을 띄우지 않도록 _on_queue_completed가 읽음)

        생성은 취소하지 않는다 — 보낸 항목이 아직 GPU 에 있으면 '생성 중' 표시를 둔 채 기억해 둔다
        (``_orphan_id``: 곧바로 다시 시작하면 이어받고, 멈춘 채 끝나면 on_generation_completed 가
        정리한다). 이 매니저가 건 표시만 다룬다 — 자동화 '큐 우선'의 표시는 건드리지 않는다.
        """
        self._run_epoch += 1   # 걸려 있는 지연 타이머(continue_processing)를 무효로
        inflight, self._inflight_id = self._inflight_id, None
        self.is_running = False
        self.is_paused = False
        self.last_stop_natural = natural
        if inflight is not None:
            if self._generation_active():
                self._orphan_id = inflight
            else:
                self._release_mark(inflight)
        self.paused_changed.emit(False)
        self.queue_completed.emit(self.generated_count)

    def pause(self):
        """일시정지 — 큐는 유지하고 다음 아이템 처리만 막음."""
        if not self.is_running or self.is_paused:
            return
        self.is_paused = True
        self.paused_changed.emit(True)

    def resume(self):
        """일시정지 해제 — 멈춰있던 _process_next 재개.

        보낸 항목이 아직 GPU 에 있으면(일시정지 뒤 곧바로 재개) 다시 보내지 않는다 — 그 결과가
        오면 on_generation_completed 가 이어서 다음 항목으로 간다(_process_next 가 가린다).
        """
        if not self.is_running or not self.is_paused:
            return
        self.is_paused = False
        self.paused_changed.emit(False)
        self._process_next()

    # ── 결과를 기다리는 항목 (ui.generator_generation._generation_matches_queue 가 읽는다) ──

    def expects_result(self) -> bool:
        """이 매니저가 결과를 기다리는 항목이 있는가 — 보낸 항목, 또는 중지 뒤에도 워커가 만드는 항목."""
        return self._inflight_id is not None or self._orphan_id is not None

    def awaited_item(self):
        """결과를 기다리는 항목(dict). 없거나 대기열에서 빠졌으면 None."""
        item_id = self._inflight_id if self._inflight_id is not None else self._orphan_id
        if item_id is None:
            return None
        return self.queue_panel.get_item_by_id(item_id)

    # ── 내부 ──

    def _generation_active(self) -> bool:
        probe = self.generation_active
        if not callable(probe):
            return False
        try:
            return bool(probe())
        except Exception:
            return False

    def _start_refusal(self) -> str:
        probe = self.start_blocker
        if not callable(probe):
            return ''
        try:
            return str(probe() or '')
        except Exception:
            return ''

    def _claim(self, item_id) -> bool:
        """이 매니저 이름으로 '생성 중' 표시를 건다. 다른 주인(자동화)이 쥐고 있으면 False."""
        return bool(self.queue_panel.claim_processing(item_id, QUEUE_OWNER))

    def _release_mark(self, item_id) -> None:
        """이 매니저가 건 표시만 푼다 — 자동화 '큐 우선'의 표시는 그대로 둔다."""
        if item_id is not None:
            self.queue_panel.release_processing(item_id, QUEUE_OWNER)

    def _release_dispatch(self) -> None:
        """보낸 표시를 푼다 — 요청이 백엔드에 닿지 못했다(항목은 대기열에 남는다)."""
        item_id, self._inflight_id = self._inflight_id, None
        self._release_mark(item_id)

    def _hold(self, reason: str) -> None:
        """보내지 못하고 멈춘다 — 항목은 그대로 두고, 사유를 호스트에 알린다(재개하면 다시 본다)."""
        self.pause()
        if reason:
            self.notice.emit(reason)

    def _process_next(self):
        """다음 아이템 처리"""
        if not self.is_running:
            return
        if self.is_paused:
            # 일시정지 중 — 큐 그대로 두고 대기. resume()이 다시 호출
            return

        active = self._generation_active()
        if self._inflight_id is not None:
            if active:
                # 보낸 항목이 아직 GPU 에 있다(재개 · 옛 지연 타이머) — 그 결과가 오면 이어서 간다.
                # 예전엔 같은 항목을 다시 보냈다가 거절돼 일시정지했고, 그때 추적도 잃었다.
                return
            # 결과가 오지 않은 채 워커가 끝났다(짝이 맞지 않는 결과 등) — 표시를 풀고 다시 보낸다
            self._release_dispatch()

        if self._orphan_id is not None:
            orphan, self._orphan_id = self._orphan_id, None
            if active:
                # 중지 전에 보낸 항목을 워커가 아직 만든다 — 다시 보내지 않고 이어받는다
                self._inflight_id = orphan
                self._claim(orphan)
                return
            # 그 결과는 이미 지나갔다(놓친 완료) — 표시만 풀고 평소대로 맨 앞부터 본다
            self._release_mark(orphan)

        item = self.queue_panel.get_first_item()

        if not item:
            # 큐가 비었다 — 한 장이라도 만들었으면 자연 완료, 아니면(재개했더니 비어 있음 등)
            # 수동 중지로 끝낸다. 예전엔 '실행 중'에 갇혀 이후 추가한 항목이 시작되지 않았다.
            self.stop(natural=self.generated_count > 0)
            return

        # 다른 소비자가 대기열 항목을 만들고 있거나 워커가 다른 생성을 하고 있으면 보내지 않는다 —
        # 보냈다가 start_generation 이 거절하면 그 전에 항목 값이 UI 에 덮어써지고, 실패 처리가
        # 남의 '생성 중' 표시를 풀었다.
        if self.queue_panel.processing_held_by_other(QUEUE_OWNER):
            self._hold(AUTOMATION_NOTICE)
            return
        if active:
            self._hold(BUSY_NOTICE)
            return

        item_id = item['id']
        if not self._claim(item_id):
            self._hold(AUTOMATION_NOTICE)
            return
        self._inflight_id = item_id
        self.generation_requested.emit(item)
        # 슬롯(_on_generation_requested)은 같은 스레드에서 바로 돈다. 보내지 못했으면 거기서
        # pause() 한다 — 그 항목은 백엔드에 닿지 않았으니 보낸 표시를 풀어 둔다(재개 때 다시 보낸다).
        if self.is_paused and self._inflight_id == item_id:
            self._release_dispatch()

    def on_generation_deferred(self):
        """보낸 요청이 백엔드에 닿기 전에 멈췄다(_queue_deferred) — 소비하지 않고 일시정지."""
        if self._inflight_id is None:
            if self._orphan_id is not None:
                # 중지 뒤 남은 요청이 백엔드에 닿지 못했다 — 항목은 그대로, 표시만 푼다
                orphan, self._orphan_id = self._orphan_id, None
                self._release_mark(orphan)
            return
        if not self.is_running:
            return
        self.pause()
        self._release_dispatch()

    def _finish_orphan(self, success: bool) -> None:
        """중지 뒤에 끝난 항목 정리 — 만들어졌으면 지우고(다음 시작 때 같은 장을 또 만들지 않게),
        실패·취소면 표시만 푼다(항목은 남아 다음 시작 때 다시 나간다)."""
        orphan = self._orphan_id
        if orphan is None:
            return
        self._orphan_id = None
        if success:
            self.queue_panel.consume_item(orphan)
        else:
            self._release_mark(orphan)

    def on_generation_completed(self, success: bool):
        """생성 완료 콜백 — 이 매니저가 보낸 항목만 지우고 다음으로."""
        item_id = self._inflight_id
        if item_id is None:
            # 보낸 항목이 없다(일시정지 중 수동 생성 · 보내지 못한 항목 등) — 중지 뒤 끝난 항목이면
            # 그것만 정리하고, 아니면 대기열을 건드리지 않는다
            self._finish_orphan(success)
            return
        if not self.is_running:
            return
        self._inflight_id = None

        if success:
            self._success_count += 1
        else:
            self._fail_count += 1

        finished = self.queue_panel.consume_item(item_id)
        is_last_of_group = finished.get('is_last_of_group', True) if finished else True
        self.generated_count += 1

        # 정기 정리 — N장마다 체크포인트를 내려 다음 장이 새로 올린다(OOM 회피). 남은 항목이
        # 없으면 하지 않는다(끝난 뒤 언로드는 '생성 후 모델 언로드' 설정의 몫).
        if periodic_unload_due(self.generated_count, self.cleanup_every_n,
                               remaining=self.queue_panel.count()):
            self._request_periodic_unload()

        delay_ms = int(self.delay_seconds * 1000)
        epoch = self._run_epoch

        def continue_processing():
            if not self.is_running or epoch != self._run_epoch:
                # 기다리는 사이 중지(→다시 시작)됐다 — 옛 회차의 타이머는 새 회차를 건드리지 않는다
                return

            if self.queue_panel.is_empty() and is_last_of_group:
                # 큐가 비어있고 마지막 그룹 → 자연 완료
                self.stop(natural=True)
            else:
                self._process_next()

        if delay_ms > 0:
            QTimer.singleShot(delay_ms, continue_processing)
        else:
            continue_processing()

    def _request_periodic_unload(self) -> None:
        """백엔드 모델 언로드를 데몬 스레드로 보낸다 — 다음 생성 워커가 run() 초입에서 기다린다.

        언로드는 공유 GPU 리스를 잡은 동안에만 나간다 — 대기열 밖의 생성(인페인트·I2I·채팅 등)이
        쥐고 있으면 건너뛴다(core.post_generation.start_post_generation_unload).
        """
        try:
            from backends import get_backend
            from core.post_generation import UNLOAD_SKIPPED_BUSY, start_post_generation_unload

            count, every = self.generated_count, self.cleanup_every_n

            def _report(ok: bool) -> None:
                if ok is UNLOAD_SKIPPED_BUSY:
                    _logger.info("대기열 정기 정리(매 %d장, 현재 %d장): 다른 생성 작업이 GPU 를 쓰는 중이라 건너뜀",
                                 every, count)
                    return
                _logger.info("대기열 정기 정리(매 %d장, 현재 %d장): 체크포인트 언로드 %s",
                             every, count, "완료" if ok else "실패")

            if start_post_generation_unload(get_backend(), on_done=_report) is None:
                _logger.info("대기열 정기 정리: 이미 언로드가 진행 중이라 건너뜀")
        except Exception as exc:
            _logger.warning("대기열 정기 정리 실패(무시): %s", exc)
