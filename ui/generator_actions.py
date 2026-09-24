# ui/generator_actions.py
"""
UI 액션 및 이벤트 처리 로직
"""
import os
from PyQt6.QtWidgets import QMessageBox

from utils.theme_manager import get_color
from utils.app_logger import get_logger
from ui.generator_generation import _gen_btn_style, _gen_btn_default_color

_logger = get_logger('actions')

class ActionsMixin:
    """UI 액션 관련 로직을 담당하는 Mixin"""
    
    def connect_signals(self):
        """시그널 연결"""
        # 생성 버튼
        self.btn_generate.clicked.connect(self.on_generate_clicked)
        self.btn_random_prompt.clicked.connect(self.apply_random_prompt)
        self.btn_save_settings.clicked.connect(self.save_settings)
        
        # 텍스트 변경 시 업데이트
        text_inputs = [
            self.char_count_input, self.character_input, 
            self.copyright_input, self.artist_input
        ]
        for inp in text_inputs:
            inp.textChanged.connect(self.on_input_changed)
        
        text_edits = [
            self.prefix_prompt_text, self.main_prompt_text, 
            self.suffix_prompt_text
        ]
        for edit in text_edits:
            edit.textChanged.connect(self.on_input_changed)
        
        # 베이스 프롬프트 변경 감지
        self.prefix_prompt_text.textChanged.connect(self.on_base_prompts_changed)
        self.suffix_prompt_text.textChanged.connect(self.on_base_prompts_changed)
        self.neg_prompt_text.textChanged.connect(self.on_base_prompts_changed)
        
        # 포커스 아웃 시 정리 (eventFilter로 처리, 디바운스 타이머의 보완)
        text_edits_to_clean = [
            self.prefix_prompt_text,
            self.main_prompt_text,
            self.suffix_prompt_text,
            self.neg_prompt_text,
            self.exclude_prompt_local_input,
            self.s1_widgets['prompt'],
            self.s2_widgets['prompt'],
        ]
        for widget in text_edits_to_clean:
            widget.installEventFilter(self)

        # (선행/후행/네거티브·ADetailer·SAM3 표시 토글과 랜덤 해상도 편집기는 Vue 화면이 맡는다 —
        #  예전 setVisible 배선은 늘 켜짐 더미·no-op 프록시에 연결돼 아무 일도 하지 않았다(audit #175).
        #  랜덤 해상도 목록은 Vue 편집기가 set_random_resolutions 액션으로 보낸다.)
        # (즐겨찾기·이벤트·PNG Info·갤러리·XYZ 는 Vue 액션이 맡는다 — 숨은 레거시 PyQt 탭과 그
        #  시그널 배선은 은퇴했다. tests/test_legacy_gallery_tabs_retirement.py)

    def on_generate_clicked(self):
        """생성 버튼 클릭 (일반 생성 또는 자동화 시작/중지)"""
        # 자동화 모드가 켜져 있으면
        if self.btn_auto_toggle.isChecked():
            if self.is_automating:
                # 자동화 중지
                self._stop_automation("사용자가 자동화를 중지했습니다.")
            else:
                # 자동화 시작
                self._start_automation()
        else:
            # 일반 이미지 생성
            self.start_generation()
    
    def on_input_changed(self):
        """입력 변경 시 최종 프롬프트 업데이트"""
        if not self.is_programmatic_change:
            self.update_total_prompt_display()
    
    def toggle_automation_ui(self, checked):
        """자동화 모드 토글 (ON/OFF만 — 패널 표시는 별도 접이식)"""
        # 생성 중이면 토글 무시하고 되돌린다.
        # 되돌리는 setChecked 는 toggled 를 다시 쏘고 그게 곧 이 함수다 — 시그널을 막지 않으면
        # 되돌림 → 이 함수 → 되돌림 … 재귀가 한도까지 쌓였다가 풀리며 경고창이 수백 번 떴고,
        # 생성이 끝난 뒤에도 계속 떠서 강제 종료해야 했다. 경고도 모달 대신 토스트 한 번.
        if hasattr(self, 'gen_worker') and self.gen_worker and self.gen_worker.isRunning():
            btn = self.btn_auto_toggle
            was_blocked = btn.blockSignals(True)
            try:
                btn.setChecked(not checked)
            finally:
                btn.blockSignals(was_blocked)
            self._notify_automation_locked()
            return

        if checked:
            self.btn_auto_toggle.setText("AUTOMATION: ON")
            self.btn_auto_toggle.setStyleSheet(f"""
                QPushButton {{
                    background-color: {get_color('success')}; color: black;
                    border: none; border-radius: 5px; font-weight: bold;
                }}
                QPushButton:hover {{ background-color: {get_color('success')}; }}
            """)
            self.btn_generate.setText("자동화 시작")
        else:
            if self.is_automating:
                self._stop_automation("자동화가 중지되었습니다.")
            self.btn_auto_toggle.setText("AUTOMATION: OFF")
            self.btn_auto_toggle.setStyleSheet("")  # 테마 기본 스타일 복원
            self.btn_generate.setText("이미지 생성")
            
    def _notify_automation_locked(self):
        """'생성 중엔 못 바꾼다' 알림 — 토스트 한 번, 1.5초 안엔 반복하지 않는다."""
        import time as _t
        now = _t.monotonic()
        if now - float(getattr(self, '_auto_lock_notified_at', 0.0) or 0.0) < 1.5:
            return
        self._auto_lock_notified_at = now
        message = "이미지 생성 중에는 자동화 모드를 바꿀 수 없습니다"
        bridge = getattr(self, 'vue_bridge', None)
        if bridge is not None and hasattr(bridge, 'showNotification'):
            bridge.showNotification.emit('warning', message)
        else:
            QMessageBox.warning(self, "알림", message)

    # ── 덱 진행도 영속 (재시작 시 '얼마나 뽑았는지' 복원) ────────────────
    def _deck_state_path(self):
        from core.storage_paths import cache_file
        return str(cache_file(
            'search/last_deck.json',
            legacy_paths='config/last_deck.json',
        ))

    def _save_deck_state(self):
        """남은 덱을 filtered_results 인덱스 리스트로 저장.
        풀(filtered_results)은 last_search_results.json으로 따로 복원되므로,
        덱은 인덱스만 저장하면 충분(대용량에도 컴팩트). 소비/리필 때마다 호출."""
        try:
            import os, json
            fr = getattr(self, 'filtered_results', None) or []
            deck = getattr(self, 'shuffled_prompt_deck', None)
            snapshot_id = getattr(self, '_search_snapshot_id', None)
            if (
                deck is None
                or not isinstance(snapshot_id, str)
                or len(snapshot_id) != 32
                or any(ch not in '0123456789abcdefABCDEF' for ch in snapshot_id)
            ):
                return
            # id→index 맵 캐시 (filtered_results 객체가 교체되면 재생성)
            if getattr(self, '_deck_idmap_for', None) != id(fr):
                self._deck_idmap = {id(b): i for i, b in enumerate(fr)}
                self._deck_idmap_for = id(fr)
            idmap = self._deck_idmap
            idx = [idmap[id(b)] for b in deck if id(b) in idmap]
            path = self._deck_state_path()
            state = {
                'schema_version': 1,
                'snapshot_id': snapshot_id,
                'pool_size': len(fr),
                'remaining': idx,
            }
            # 덱을 만든 등급 필터 — 재시작 뒤 필터가 바뀌면 '새로 허용된 행'만 더하는 데 쓴다
            # (없는 옛 파일도 그대로 읽는다: core.search_deck.set_owner_rating_filter).
            built = getattr(self, '_deck_built_ratings', None)
            if built is not None:
                state['ratings'] = sorted(built)
            from utils.atomic_json import atomic_write_json
            atomic_write_json(path, state, indent=None)
            self._deck_throttle().mark_saved()
        except Exception as e:
            print(f"[Deck] save 실패: {e}")

    def _deck_throttle(self):
        """덱 저장 모으기(core.search_deck.DeckSaveThrottle) — 뽑을 때마다 전체를 쓰지 않는다."""
        throttle = getattr(self, '_deck_save_throttle', None)
        if throttle is None:
            from core.search_deck import DeckSaveThrottle
            throttle = self._deck_save_throttle = DeckSaveThrottle()
        return throttle

    def _note_deck_draw(self):
        """덱에서 한 장 뽑은 뒤 — 20번마다 · 60초마다 · 덱이 비면 저장한다(나머지는 모아 둔다).

        모아 두기만 하고 더 뽑지 않으면 interval 뒤 한 번 더 저장한다(_schedule_deck_flush) —
        종료 훅이 돌지 않는 끝(웹 모드 콘솔 닫기 · 강제 종료)에도 잃는 진행도가 그 시간으로 묶인다.
        """
        remaining = len(getattr(self, 'shuffled_prompt_deck', None) or [])
        throttle = self._deck_throttle()
        if throttle.note_draw(remaining):
            self._save_deck_state()
        elif throttle.dirty:
            self._schedule_deck_flush(throttle.interval)

    def _schedule_deck_flush(self, seconds: float):
        """모아 둔 덱 진행도를 seconds 뒤 한 번 저장한다(이미 걸려 있으면 그대로)."""
        if getattr(self, '_deck_flush_scheduled', False):
            return
        try:
            from PyQt6.QtCore import QCoreApplication, QTimer
            if QCoreApplication.instance() is None:
                return   # 이벤트 루프가 없다(단위 테스트 대역 등) — 중지·종료 flush 에 맡긴다
            self._deck_flush_scheduled = True
            QTimer.singleShot(max(0, int(seconds * 1000)), self._on_deck_flush_timer)
        except Exception:
            self._deck_flush_scheduled = False

    def _on_deck_flush_timer(self):
        self._deck_flush_scheduled = False
        self._flush_deck_state()

    def _flush_deck_state(self):
        """모아 둔 덱 진행도를 지금 저장 — 자동화 중지·앱 종료(데스크톱 _quit_app · 웹 모드
        aboutToQuit) 때, 그리고 모아 두고 더 안 뽑은 채 interval 이 지났을 때."""
        throttle = getattr(self, '_deck_save_throttle', None)
        if throttle is not None and throttle.dirty:
            self._save_deck_state()

    def _restore_deck_state(self) -> bool:
        """저장된 인덱스로 남은 덱을 재구성. 성공 시 True(호출자는 새 셔플 생략).
        풀 크기가 다르면(새 검색 등) False → 호출자가 새로 셔플."""
        try:
            import os, json
            fr = getattr(self, 'filtered_results', None) or []
            if not fr:
                return False
            path = self._deck_state_path()
            if not os.path.exists(path):
                return False
            with open(path, encoding='utf-8') as f:
                st = json.load(f)
            if st.get('schema_version') != 1:
                return False
            if st.get('snapshot_id') != getattr(self, '_search_snapshot_id', None):
                return False
            if st.get('pool_size') != len(fr):
                return False  # 풀이 바뀜 → 복원 불가
            n = len(fr)
            self.shuffled_prompt_deck = [fr[i] for i in st.get('remaining', [])
                                         if isinstance(i, int) and 0 <= i < n]
            saved_ratings = st.get('ratings')
            if isinstance(saved_ratings, list):
                from core.search_deck import normalize_ratings
                self._deck_built_ratings = normalize_ratings(saved_ratings)
            else:
                self._deck_built_ratings = None   # 옛 파일 — 어떤 필터로 만든 덱인지 모른다
            print(f"[Deck] 복원: 남은 {len(self.shuffled_prompt_deck):,} / 전체 {n:,}")
            return True
        except Exception as e:
            print(f"[Deck] restore 실패: {e}")
            return False

    def _start_automation(self):
        """자동화 시작"""
        if not self.filtered_results:
            QMessageBox.warning(self, "알림", "먼저 검색을 수행하세요.")
            return
        # 대기열이 돌고 있으면(일시정지 포함) 시작하지 않는다 — 둘 다 같은 대기열·워커를 쓴다.
        # 예전엔 자동화 '큐 우선'과 대기열 매니저가 같은 항목을 보내고 서로의 '생성 중' 표시를 풀었다.
        from ui.queue_coordination import announce, automation_start_refusal
        refusal = automation_start_refusal(self)
        if refusal:
            announce(self, refusal)
            return

        import time

        # 덱 초기화 — 이미 남은 덱이 있으면(부분 소비 포함) 유지하여 진행도 보존.
        # 비었을 때만(검색 직후/소진 후) 새로 채운다(등급 필터 → 셔플 → 저장: core.search_deck).
        # 필터를 통과한 행이 하나도 없으면 시작하지 않는다(예전엔 첫 pop 에서 IndexError).
        from core.search_deck import NO_ELIGIBLE_MESSAGE, refill_owner_deck
        if not self.shuffled_prompt_deck:
            refill_owner_deck(self, emit=False)
        if not self.shuffled_prompt_deck:
            QMessageBox.warning(self, "알림", NO_ELIGIBLE_MESSAGE)
            return

        self.is_automating = True
        # 새 회차 — 직전 회차의 재시도 타이머·재시도 횟수를 물려주지 않는다(_automation_after_generation)
        self._auto_run_epoch = getattr(self, '_auto_run_epoch', 0) + 1
        self._auto_retry_count = 0
        self.auto_gen_count = 0
        self.auto_current_repeat = 1
        # 일시정지 · 일회성 덮어쓰기 상태를 새 회차에 물려주지 않는다.
        # (직전 자동화에서 멈춘 채 끝났으면 새로 시작하자마자 또 멈춰 버린다)
        self._auto_paused = False
        self._auto_resume_pending = ''
        self._wait_paused_remaining_ms = None
        self._auto_prompt_override = ''
        self._auto_override_used = False
        # 직전 회차가 큐 항목 도중 멈췄으면(시작 실패 · 중지) '처리 중' 표시가 남아 있다 — 새 회차의
        # 첫 덱 장이 큐 항목으로 취급돼 세지 않고, 끝나면 남의 큐 항목을 지운다.
        self._auto_processing_queue = False
        self._queue_dirtied_prompt = False

        settings = self.automation_widget.get_settings()
        self.auto_settings = settings
        self._emit_auto_status()

        # 시간 제한 모드면 시작 시간 기록
        if settings['termination_mode'] == 'timer':
            self.auto_start_time = time.time()
        
        self.btn_random_prompt.setText(f"🎲 랜덤 프롬프트 ({len(self.shuffled_prompt_deck)})")

        # 버튼 상태 변경
        self.btn_generate.setText("⏸️ 자동화 중지")
        self.btn_generate.setStyleSheet(_gen_btn_style('#e74c3c'))

        self.show_status("🔄 자동화 시작...")

        # ★★★ 첫 번째 프롬프트 적용 (apply_random_prompt 사용!) ★★★
        if not self.apply_random_prompt():
            return   # 뽑을 프롬프트가 없다 — apply_random_prompt 가 자동화를 멈추고 알렸다
        # 위 호출이 총 프롬프트 상자를 다시 채운 '뒤'에 한 번 더 보낸다 —
        # 위쪽 첫 emit 은 아직 직전 회차의 프롬프트를 담고 있어 화면이 한 박자 어긋난다.
        self._emit_auto_status()

        # 첫 생성 시작 — 자동화 경로의 생성은 전부 _automation_start_generation 을 거친다
        # (일시정지 확인 + 일회성 덮어쓰기 소비가 여기 한 곳에만 있어야 새는 경로가 없다)
        from PyQt6.QtCore import QTimer
        delay_ms = int(settings['delay'] * 1000)
        QTimer.singleShot(delay_ms, self._automation_start_generation)


    def _run_automation_cycle(self):
        """자동화 사이클"""
        if not self.is_automating:
            return

        import time
        from PyQt6.QtCore import QTimer

        # 일시정지는 여기서 바로 서지 않는다 — 다음 장 프롬프트까지 준비(종료 판정 · 덱 뽑기)한
        # 뒤 아래에서 선다. 멈춘 이유가 '다음 장을 내보내기 전에 손보기'라서, 패널이 방금 생성한
        # 프롬프트가 아니라 실제로 나갈 프롬프트를 보여 줘야 그 편집이 맞는 장에 걸린다.
        # (예전엔 뽑기 전에 서서 이전 프롬프트를 '다음 프롬프트'로 보여 줬고, 거기 건 편집은
        # 재개하자마자 새로 뽑은 프롬프트 앞에서 버려져 본 적 없는 프롬프트가 나갔다)

        # 직전 회차에서 일회성 덮어쓰기를 썼다면 총 프롬프트 상자를 섹션에서 다시 조립한다.
        # 반복 생성(repeat_per_prompt>1) 회차는 새 프롬프트를 안 뽑아 상자를 갱신하지
        # 않으므로, 여기서 되돌리지 않으면 덮어쓴 값이 다음 장까지 따라간다 —
        # 사용자가 정한 '이번만' 이 깨지는 유일한 구멍이었다.
        if getattr(self, '_auto_override_used', False):
            self._auto_override_used = False
            try:
                self.update_total_prompt_display()
            except Exception:
                pass

        settings = self.auto_settings

        # 종료 조건 확인 (PR 3: 'unlimited' 모드 추가 — 종료 조건 없음)
        mode = settings.get('termination_mode', 'count')
        if mode == 'unlimited':
            pass  # 사용자가 중지할 때까지 계속
        elif mode == 'count':
            if self.auto_gen_count >= settings['termination_limit']:
                self._stop_automation(f"✅ 자동화 완료: {self.auto_gen_count}장 생성")
                return
        else:  # timer
            elapsed = time.time() - self.auto_start_time
            if elapsed >= settings['termination_limit']:
                self._stop_automation(f"✅ 시간 종료: {self.auto_gen_count}장 생성")
                return
        
        # 반복 횟수 확인
        if self.auto_current_repeat >= settings['repeat_per_prompt']:
            self.auto_current_repeat = 0
            
            # ★★★ 새 프롬프트 적용 (apply_random_prompt 사용!) ★★★
            # 덱이 비었는지 확인
            if not self.shuffled_prompt_deck and not settings.get('allow_duplicates', False):
                if settings.get('auto_reset_deck', False):
                    # 한 바퀴 완료 → 덱 자동 초기화(재셔플)하고 계속 (무한·공평: 모두 1회씩 후 다시)
                    from core.search_deck import refill_owner_deck
                    refill_owner_deck(self, emit=False)
                    self.show_status(f"🔄 덱 소진 → 자동 초기화 ({len(self.shuffled_prompt_deck)})")
                else:
                    self._stop_automation("✅ 모든 프롬프트 처리 완료!")
                    return

            # 이 장 전에 걸린 일회성 덮어쓰기는 '방금 끝난 프롬프트'를 고친 것이다 — 새로 뽑은
            # 다음 프롬프트를 그 편집본으로 바꿔치지 않게 여기서 버린다(패널은 새 프롬프트로 초기화).
            self._discard_stale_prompt_override()

            if not self.apply_random_prompt():
                return   # 등급 필터를 통과한 행이 없다 — apply_random_prompt 가 자동화를 멈췄다

        # 대기 후 생성 — 대기 동안 카운트다운 타이머로 남은 시간 표시 (Search % 느낌)
        import time
        delay_ms = int(settings['delay'] * 1000)
        if getattr(self, '_auto_paused', False):
            # 일시정지 — 다음 장 프롬프트를 준비해 보여 준 채 선다(패널에서 그 프롬프트를 손본다).
            # 사이 간격은 재개 뒤 온전히 센다: 얼린 대기로 두면 _resume_automation 이 그대로
            # 카운트다운을 이어 간다(간격이 0 이면 대기 없이 생성 직전에서 선다).
            if delay_ms > 0:
                self._wait_total_ms = delay_ms
                self._wait_end_time = 0
                self._wait_paused_remaining_ms = delay_ms
                self._emit_auto_status(waiting=True)
            else:
                self._wait_total_ms = 0
                self._wait_end_time = 0
                self._auto_resume_pending = 'generate'
                self._emit_auto_status()
            return
        if delay_ms > 0:
            self._wait_total_ms = delay_ms
            self._wait_end_time = time.time() + (delay_ms / 1000.0)
            self._ensure_wait_timer()
            self._wait_timer.start(100)  # 100ms마다 남은 시간 갱신
            self._emit_auto_status(waiting=True)
        else:
            self._wait_total_ms = 0
            self._wait_end_time = 0
            self._emit_auto_status(waiting=False)
            QTimer.singleShot(0, self._automation_generate)

    # ── 일시정지 / 재개 · 일회성 프롬프트 덮어쓰기 ───────────────────────
    def _auto_is_waiting(self) -> bool:
        """지금이 '사이 간격 대기' 중인가.

        카운트다운 타이머가 돌고 있거나, 일시정지로 남은 시간을 얼려 둔 상태면 대기다.
        (_emit_auto_status 를 대기 여부를 모르는 곳 — 액션 핸들러 — 에서 부를 때 쓴다)
        """
        if getattr(self, '_wait_paused_remaining_ms', None) is not None:
            return True
        t = getattr(self, '_wait_timer', None)
        return bool(t is not None and t.isActive())

    def _set_prompt_override(self, text: str):
        """다음 '한 장'에만 쓸 프롬프트 전문을 걸어 둔다. 빈 문자열이면 덮어쓰기 취소.

        Vue 는 추가/제거를 따로 추적하지 않고 편집 결과를 전문으로 보낸다 — 그래야
        와일드카드가 이미 풀린 문자열을 사람이 그대로 손볼 수 있다.
        걸자마자 총 프롬프트 상자에 반영하는 이유: 그 상자가 곧 API 로 나갈 문자열이고
        (_build_generation_payload 가 그대로 읽는다), 사용자가 방금 고친 게 화면에
        남아 있어야 '보이는 대로 나간다'가 성립한다.
        """
        text = (text or '').strip()
        self._auto_prompt_override = text
        try:
            if text:
                self.total_prompt_display.setPlainText(text)
            else:
                # 취소 → 섹션 위젯에서 원래 프롬프트를 다시 조립한다.
                self.update_total_prompt_display()
        except Exception:
            pass
        self._emit_auto_status(waiting=self._auto_is_waiting())

    def _consume_prompt_override(self):
        """걸린 덮어쓰기를 총 프롬프트 상자에 밀어 넣고 '즉시' 비운다.

        덱에는 손대지 않는다 — 덱 pop 은 _run_automation_cycle → apply_random_prompt
        가 따로 하므로, 이번 장을 손봤다고 덱 진행이 어긋나지 않는다.
        (큐 우선 항목도 같은 방식으로 총 프롬프트 상자에 직접 넣는다 — 검증된 경로)
        """
        ov = getattr(self, '_auto_prompt_override', '') or ''
        if not ov:
            return
        self._auto_prompt_override = ''      # '이번만' — 쓰는 순간 사라진다
        self._auto_override_used = True      # 다음 사이클에서 상자를 되돌리라는 표시
        try:
            self.total_prompt_display.setPlainText(ov)
        except Exception:
            pass

    def _discard_stale_prompt_override(self) -> bool:
        """새 덱 프롬프트를 뽑기 직전 — 아직 안 쓰인 덮어쓰기는 '이전' 프롬프트를 고친 것이라 버린다.

        패널은 생성 중·'continue' 일시정지 동안 방금 쓴 프롬프트를 보여 주되, 그 문자열이 다음 덱 장에
        쓰이지 않으면 prompt_is_next=False 로 편집을 막는다(core.automation_prompt_state). 여기는
        그 사이를 빠져나온 편집(편집 직후 생성이 시작된 경합 등)을 거르는 마지막 방어선이다 —
        새로 뽑은 프롬프트에 쓰면 그 덱 항목은 한 번도 안 나가고 이전 프롬프트의 편집본이 대신
        나간다. 반복(repeat) 중이라 새로 뽑지 않을 때는 같은 프롬프트라 그대로 쓴다.
        """
        if not (getattr(self, '_auto_prompt_override', '') or ''):
            return False
        self._auto_prompt_override = ''
        message = "✏️ 이전 프롬프트에 건 편집은 새로 뽑은 프롬프트에 쓰지 않고 버렸습니다"
        self.show_status(message)
        bridge = getattr(self, 'vue_bridge', None)
        if bridge is not None:
            try:
                bridge.showNotification.emit('info', message)
            except Exception:
                pass
        return True

    def _automation_start_generation(self):
        """자동화 경로의 유일한 생성 진입점 — 일시정지 확인 + 덮어쓰기 소비 후 생성.

        실패 재시도(_automation_after_generation)도 여기로 들어온다 — 반복 카운터·자연어 변환을
        다시 돌리지 않는다. 큐 우선 항목(과 그 재시도)은 '다음 덱 프롬프트'용 덮어쓰기를 먹지 않는다.
        """
        if not self.is_automating:
            return
        if getattr(self, '_auto_paused', False):
            # 반복 카운터·자연어 변환은 이미 끝난 지점이라 재개 시 여기로 되돌아온다.
            self._auto_resume_pending = 'start'
            self._emit_auto_status()
            return
        if getattr(self, '_auto_processing_queue', False):
            # 큐 우선 항목의 재시도 — 같은 항목을 같은 준비 규칙으로 다시 낸다. 동결 항목(시드 탐색 등)은
            # UI 에 올리지 않으므로 start_generation() 으로 UI 에서 만들면 덱 프롬프트가 대신 나간다.
            # 맨 앞이 아니라 **그 항목**(id)을 다시 낸다 — 위치로 찾으면 다른 항목이 대신 나갈 수 있다.
            qp = getattr(self, 'queue_panel', None)
            try:
                item = qp.get_item_by_id(self._auto_queue_item_id) if qp is not None else None
            except Exception:
                item = None
            if item:
                self._start_automation_queue_item(qp, item)
                return
            # 그 사이 항목이 대기열에서 빠졌다 — 재시도할 것이 없으니 평소 순서(큐 → 덱)로 잇는다
            self._auto_processing_queue = False
            self._auto_queue_item_id = None
            self._continue_automation()
            return
        self._consume_prompt_override()
        self.start_generation()
        # 생성이 들어갔다 — 패널이 보여 주는 프롬프트가 이제 '생성 중인 것'인지 다시 알린다
        # (반복이 남지 않았으면 다음 장은 새로 뽑으므로 편집을 막는다: prompt_is_next).
        if self.is_automating:
            self._emit_auto_status()

    def _pause_automation(self):
        """자동화 일시정지 — 덱·카운트·반복 상태를 모두 유지한다(_stop_automation 과 다름).

        진행 중인 생성은 중단하지 않는다. 이 화면에서 '멈춤'은 다음 장을 내보내기 전에
        프롬프트를 손보려는 것이라, 이미 GPU 에 넘어간 장을 버릴 이유가 없다.
        대기 시간 처리: 남은 시간을 그 자리에서 '얼린다'(흘려보내지 않는다). 사이 간격은
        API 호출 사이의 최소 간격이라 사람이 잠깐 세웠다고 건너뛰면 설정이 무의미해지고,
        얼려 봐야 재개 후 최대 delay 한 번(보통 1초)이라 체감 비용이 없다.
        """
        if not self.is_automating or getattr(self, '_auto_paused', False):
            return
        self._auto_paused = True
        t = getattr(self, '_wait_timer', None)
        if t is not None and t.isActive():
            import time as _t
            self._wait_paused_remaining_ms = max(
                0, int((getattr(self, '_wait_end_time', 0) - _t.time()) * 1000))
            t.stop()
            self._wait_end_time = 0
        self.show_status("⏸ 자동화 일시정지 — 덱·카운트는 그대로")
        self._emit_auto_status(waiting=self._auto_is_waiting())

    def _resume_automation(self):
        """멈춘 지점부터 잇는다 — 어디서 섰는지에 따라 되돌아갈 곳이 다르다."""
        if not self.is_automating or not getattr(self, '_auto_paused', False):
            return
        self._auto_paused = False
        from PyQt6.QtCore import QTimer

        # 1) 대기 중에 멈췄다 → 얼려 둔 잔여 시간부터 다시 센다.
        rem = getattr(self, '_wait_paused_remaining_ms', None)
        if rem is not None:
            import time as _t
            self._wait_paused_remaining_ms = None
            self._wait_end_time = _t.time() + (max(0, int(rem)) / 1000.0)
            self._ensure_wait_timer()
            self._wait_timer.start(100)
            self.show_status("▶ 자동화 재개")
            self._emit_auto_status(waiting=True)
            return

        # 2) 그 외 — 멈춘 단계로 되돌아간다.
        #    generate : 대기가 끝난 직후(반복 카운터 올리기 전) — 간격 0 이면 뽑은 직후도 여기
        #    start    : 반복 카운터·자연어 변환까지 끝난 직후
        #    continue : 생성이 끝난 뒤 큐 항목이 기다릴 때(큐 우선 → 덱 순서를 그대로 잇는다)
        #    (덱 장 사이에 멈췄으면 _run_automation_cycle 이 다음 프롬프트를 뽑은 뒤 대기를 얼려
        #     두므로 위 1) 로 이어진다)
        pending = getattr(self, '_auto_resume_pending', '') or ''
        self._auto_resume_pending = ''
        self.show_status("▶ 자동화 재개")
        self._emit_auto_status()
        target = {
            'generate': getattr(self, '_automation_generate', None),
            'start': getattr(self, '_automation_start_generation', None),
            'continue': getattr(self, '_continue_automation', None),
        }.get(pending)
        if target is not None:
            QTimer.singleShot(0, target)
        # pending 이 비어 있으면 생성이 아직 진행 중이라는 뜻 —
        # 그 생성이 끝나면 _continue_automation 이 평소대로 이어간다.


    def _automation_generate(self):
        """자동화 이미지 생성"""
        if not self.is_automating:
            return

        # 대기 타이머 정지 (생성 들어가면 카운트다운 종료)
        if getattr(self, '_wait_timer', None):
            self._wait_timer.stop()
        self._wait_end_time = 0

        # 대기가 끝난 바로 이 지점이 '대기와 생성 사이'다 — 일시정지는 여기서 선다.
        # 반복 카운터를 올리기 전이라, 재개하면 이 함수로 되돌아와 그대로 이어간다.
        if getattr(self, '_auto_paused', False):
            self._wait_paused_remaining_ms = None   # 대기는 이미 다 흘렀다
            self._auto_resume_pending = 'generate'
            self._emit_auto_status()
            return

        self.auto_current_repeat += 1

        # 생성 시 태그→자연어 자동 변환 (비동기 worker — UI 안 멈춤).
        # 이미 변환된 프롬프트(반복 생성 등)는 문자열 비교로 건너뜀 → 누적 방지.
        # 큐 우선 항목은 _automation_generate를 거치지 않으므로 자연스럽게 제외됨.
        # 사용자가 이번 장 프롬프트를 직접 고쳤으면(덮어쓰기) 변환을 건너뛴다 —
        # 고친 전문이 그대로 나가야 화면에 보인 것과 API 로 나간 것이 같아진다.
        if (getattr(self, '_auto_nl_enabled', False)
                and not getattr(self, '_auto_processing_queue', False)
                and not (getattr(self, '_auto_prompt_override', '') or '')):
            cur = self.main_prompt_text.toPlainText().strip()
            if cur and cur != getattr(self, '_auto_nl_last_output', None):
                if self._start_auto_nl_then_generate(cur):
                    return   # worker 완료 콜백에서 생성 호출
        self._automation_start_generation()

    def _start_auto_nl_then_generate(self, base_tags: str) -> bool:
        """태그→nl_caption 변환을 비동기로 시작. 성공 시 True(호출자는 start_generation 생략).

        설치 모델 대조(/api/tags)는 워커 스레드에서 한다 — Ollama 가 꺼져 있어도 자동화가
        장마다 UI 스레드에서 연결 시간초과를 기다리지 않게."""
        try:
            from core.ollama_client import DEFAULT_OLLAMA_URL
            from workers.ollama_worker import OllamaWorker, detach_result_signals, release_when_done
            url = getattr(self, '_auto_nl_url', '') or DEFAULT_OLLAMA_URL
            model = getattr(self, '_auto_nl_model', '') or ''   # 비면 워커가 설치 모델로 정한다
            # 자동화를 멈췄다 곧바로 다시 켜면 이전 변환이 아직 돌 수 있다 — 그 결과가
            # 새 사이클의 생성을 한 번 더 부르지 않게 결과 연결만 끊는다.
            detach_result_signals(getattr(self, '_auto_nl_worker', None))
            self._auto_nl_base = base_tags
            # 이 요청이 속한 자동화 회차 — 중지(·재시작) 뒤 늦게 온 결과는 _on_auto_nl_* 가 버린다
            self._auto_nl_epoch = getattr(self, '_auto_run_epoch', 0)
            w = OllamaWorker(
                url, model, base_tags, 'nl_caption', '', self, instruction_feature='auto_nl',
                resolve_installed=True,
            )
            w.finished.connect(self._on_auto_nl_done)
            w.error.connect(self._on_auto_nl_error)
            release_when_done(w, self, '_auto_nl_worker')
            self._auto_nl_worker = w
            if hasattr(self, 'show_status'):
                self.show_status("🅣→🅝 자연어 변환 중…")
            w.start()
            return True
        except Exception as e:
            print(f"[AutoNL] 변환 시작 실패(태그로 생성): {e}")
            return False

    def _auto_nl_result_is_current(self) -> bool:
        """도착한 자연어 변환 결과가 지금 도는 자동화 회차의 것인가.

        중지(_stop_automation)는 회차를 올리므로, 중지 뒤나 중지→재시작 뒤에 늦게 온 결과는 아니다.
        그런 결과가 프롬프트 상자를 덮으면 중지 뒤 사용자가 고친 프롬프트가 사라지고, 재시작한 회차의
        첫 장(또는 반복 장)이 옛 회차 프롬프트로 나가며 새로 뽑은 덱 장은 생성 없이 소비됐다.
        """
        return (bool(getattr(self, 'is_automating', False))
                and getattr(self, '_auto_nl_epoch', None) == getattr(self, '_auto_run_epoch', 0))

    def _on_auto_nl_done(self, result: str):
        """변환 완료 → 태그 뒤에 자연어 추가 → 생성 (UI 스레드에서 실행됨)."""
        # 로그는 safe_print — cp949 로 리다이렉트된 stdout(파이프·/verify)에서 print 가
        # UnicodeEncodeError 로 이 슬롯을 깨고 다음 생성을 건너뛰지 않게(core/safe_print.py).
        from core.safe_print import safe_print
        if not self._auto_nl_result_is_current():
            safe_print("[AutoNL] 중지·재시작된 회차의 변환 결과 - 버림")
            return
        try:
            import json as _json
            d = _json.loads(result)
            nl = (d.get('tags') or '').strip()
            base = getattr(self, '_auto_nl_base', '') or ''
            if nl:
                combined = (base + ', ' + nl) if base else nl
                self._auto_nl_last_output = combined
                self.main_prompt_text.setPlainText(combined)
                if hasattr(self, 'update_total_prompt_display'):
                    self.update_total_prompt_display()
        except Exception as e:
            safe_print(f"[AutoNL] 결과 처리 실패: {e}")
        if self.is_automating:
            # 변환이 update_total_prompt_display 로 상자를 다시 채웠으므로,
            # 덮어쓰기 소비는 반드시 그 '뒤'인 여기서 일어나야 한다.
            self._automation_start_generation()

    def _on_auto_nl_error(self, err: str):
        """변환 실패 → 원본 태그 그대로 생성."""
        # err 는 서버·예외 문구 그대로라 어떤 문자든 올 수 있다 — safe_print(위와 같은 이유)
        from core.safe_print import safe_print
        if not self._auto_nl_result_is_current():
            safe_print(f"[AutoNL] 중지·재시작된 회차의 변환 실패 - 버림: {err}")
            return
        safe_print(f"[AutoNL] 변환 실패(태그로 생성): {err}")
        self._automation_start_generation()

    def _ensure_wait_timer(self):
        """자동화 대기 카운트다운 타이머 보장 (100ms 간격)."""
        if getattr(self, '_wait_timer', None) is None:
            from PyQt6.QtCore import QTimer
            self._wait_timer = QTimer(self)
            self._wait_timer.setInterval(100)
            self._wait_timer.timeout.connect(self._on_wait_tick)

    def _on_wait_tick(self):
        """대기 중 남은 시간 갱신 → 0이 되면 생성 시작."""
        import time
        if not self.is_automating or getattr(self, '_auto_paused', False):
            # 일시정지는 _pause_automation 이 이미 타이머를 세웠다 — 여기는 방어선.
            if getattr(self, '_wait_timer', None):
                self._wait_timer.stop()
            return
        remaining = getattr(self, '_wait_end_time', 0) - time.time()
        if remaining <= 0:
            # _automation_generate가 타이머 정지 처리
            self._automation_generate()
            return
        self._emit_auto_status(waiting=True)


    def _automation_after_generation(self, success: bool) -> bool:
        """자동화 중 한 장이 끝난 직후(on_generation_finished) — 장 수 세기와 실패 재시도 판정.

        - 성공: 재시도 횟수를 비우고, 덱 장이면 auto_gen_count +1(큐 우선 항목은 세지 않는다).
        - 실패: max_retries 까지 지수 백오프로 **같은 장**을 다시 시도한다. 재시도는
          _automation_start_generation 으로 들어가 반복 카운터·자연어 변환을 다시 돌리지 않는다
          (예전엔 _automation_generate 로 들어가 repeat_per_prompt≥2 에서 그 프롬프트가 한 장씩
          모자랐고, 자연어 변환이 다시 돌며 사용자 덮어쓰기를 지웠다).
          재시도 예약은 회차(_auto_run_epoch)에 묶는다 — 멈췄다 곧바로 다시 시작해도 옛 타이머가
          새 회차에서 한 장을 더 내보내지 않는다.
        반환: 재시도를 예약했으면 True(호출자는 큐 알림·다음 사이클을 건너뛴다).
        """
        if not getattr(self, 'is_automating', False):
            return False
        if success:
            self._auto_retry_count = 0
            if not getattr(self, '_auto_processing_queue', False):
                self.auto_gen_count = getattr(self, 'auto_gen_count', 0) + 1
            return False

        from core.automation_retry import plan_retry
        max_retries = (getattr(self, 'auto_settings', {}) or {}).get('max_retries', 0)
        decision = plan_retry(getattr(self, '_auto_retry_count', 0), max_retries)
        if decision is None:
            self._auto_retry_count = 0   # 재시도 소진 — 이 장은 포기하고 다음으로
            return False
        self._auto_retry_count, backoff = decision
        try:
            limit = int(max_retries or 0)
        except (TypeError, ValueError):
            limit = self._auto_retry_count
        self.show_status(
            f"⚠️ 생성 실패 — {backoff:.1f}초 후 재시도 ({self._auto_retry_count}/{limit})")
        epoch = getattr(self, '_auto_run_epoch', 0)
        from PyQt6.QtCore import QTimer
        QTimer.singleShot(int(backoff * 1000), lambda: self._automation_retry(epoch))
        return True

    def _automation_retry(self, epoch: int):
        """예약된 재시도 — 그 사이 자동화가 멈췄거나 새 회차가 시작됐으면 버린다."""
        if epoch != getattr(self, '_auto_run_epoch', 0) or not getattr(self, 'is_automating', False):
            return
        self._automation_start_generation()

    def _continue_automation(self):
        """자동화 계속 (on_generation_finished에서 호출).

        큐 우선: 자동화 중 큐에 대기 항목이 있으면 자동화 덱보다 '먼저' 생성한다.
        - 큐 항목은 _automation_generate를 거치지 않으므로 반복 카운터(auto_current_repeat)
          가 보존됨 → 큐 처리 후 남은 반복을 그대로 이어감.
        - 큐 항목은 자동화 종료 횟수(auto_gen_count)에 미포함 — 성공 시점
          (_automation_after_generation)에 아예 세지 않는다. 예전엔 성공에서 +1 하고 여기서
          무조건 -1 해, 큐 항목이 실패(재시도 소진)하면 세지도 않은 걸 빼서 덱이 한 장 더 나갔다.
        - 큐 항목이 UI 프롬프트를 바꾸므로, 큐가 비면 현재 자동화 프롬프트로 복원.
        """
        if not self.is_automating:
            return
        qp = getattr(self, 'queue_panel', None)

        # 1) 직전 생성이 '큐 우선' 항목이었으면 정리: **그 항목**(id)을 큐에서 제거.
        #    예전엔 맨 앞 항목을 지워, 그 사이 항목이 바뀌었으면(표시가 풀려 지워진 뒤 등) 생성하지
        #    않은 다음 항목이 사라졌다. 이미 없으면 아무것도 지우지 않는다.
        if getattr(self, '_auto_processing_queue', False):
            self._auto_processing_queue = False
            item_id, self._auto_queue_item_id = getattr(self, '_auto_queue_item_id', None), None
            if qp is not None and item_id is not None:
                try:
                    qp.consume_item(item_id)
                except Exception:
                    pass

        # 2) 큐 우선 처리: 대기 항목이 있으면 다음 큐 항목 생성 (반복 상태 건드리지 않음)
        if qp is not None:
            try:
                item = qp.get_first_item()
            except Exception:
                item = None
            if item:
                # 2.5) 일시정지 — 방금 끝난 생성의 뒷정리까지만 하고 선다. 재개하면 이 함수 처음부터
                #      다시 들어온다(위 1단계는 _auto_processing_queue 가 이미 False 라 무해) → 큐 우선
                #      → 프롬프트 복원 → 덱 순서가 그대로 이어진다. 큐가 비어 있으면 여기서 서지 않고
                #      _run_automation_cycle 이 다음 덱 프롬프트를 뽑아 보여 준 뒤 선다.
                if getattr(self, '_auto_paused', False):
                    self._auto_resume_pending = 'continue'
                    self._emit_auto_status()
                    return
                self._start_automation_queue_item(qp, item)
                return

        # 3) 큐 비었음 → 자동화 덱 계속. 큐가 UI 프롬프트를 바꿨으면 현재 자동화 프롬프트 복원.
        if getattr(self, '_queue_dirtied_prompt', False):
            self._queue_dirtied_prompt = False
            b = getattr(self, '_current_auto_bundle', None)
            if b is not None:
                self.is_programmatic_change = True
                try:
                    self.apply_prompt_from_data(b)
                    self.update_total_prompt_display()
                finally:
                    self.is_programmatic_change = False

        self._run_automation_cycle()

    def _start_automation_queue_item(self, qp, item):
        """자동화 '큐 우선' — 대기열 항목 하나를 시작한다.

        준비 규칙은 대기열 매니저 경로(_on_generation_requested)와 같은 ui.queue_item_dispatch 다.
        동결 항목(시드 탐색 · XYZ · ComfyUI 스냅숏)은 굳힌 payload 그대로 나가고 seed 칸 같은 UI 는
        건드리지 않는다 — 예전엔 모든 항목을 UI 에서 다시 만들어, 멈춘 대기열에 남은 시드 탐색
        항목이 subseed 를 잃고 같은 이미지를 냈고 seed 칸이 기준 시드로 고정돼 이후 자동화 장이
        전부 그 시드로 나갔다.
        """
        from core.queue_model import AUTOMATION_OWNER
        from ui.queue_item_dispatch import (
            prepare_frozen_generation, start_prepared_generation, touches_ui,
        )
        item_id = item.get('id')
        # 이 항목을 자동화 이름으로 '생성 중' 표시한다 — 대기열에서 지우거나 옮길 수 없고, 대기열
        # 매니저도 보내거나 풀지 못한다. 자동화를 멈추면 _release_automation_queue_item 이 이 표시만
        # 푼다(항목은 소비되지 않고 남는다). 대기열 매니저가 이미 그 표시를 쥐고 있으면(그쪽이 생성
        # 중) 같은 항목·워커를 다투지 않도록 자동화를 멈춘다 — 시작 가드(ui/queue_coordination)가
        # 막는 경로라 여기까지 오면 조정이 어긋난 것이다.
        try:
            claimed = bool(qp.claim_processing(item_id, AUTOMATION_OWNER))
        except Exception:
            claimed = False
        if not claimed:
            from ui.queue_coordination import AUTOMATION_BLOCKED_BY_QUEUE
            self._auto_processing_queue = False
            self._auto_queue_item_id = None
            self._stop_automation(AUTOMATION_BLOCKED_BY_QUEUE)
            return
        self._auto_processing_queue = True
        self._auto_queue_item_id = item_id
        if touches_ui(item):
            # UI 에 항목 값을 올려 만드는 항목 — 큐가 비면 자동화 프롬프트로 되돌린다(3단계)
            self._queue_dirtied_prompt = True
        self.show_status("📋 큐 우선 처리 중...")
        try:
            prepared = prepare_frozen_generation(self, item)
            if prepared is None:
                self._apply_payload_to_ui(item)
                started = self.start_generation() is not False
            else:
                started = start_prepared_generation(self, prepared)
        except Exception as exc:
            # 만든 백엔드가 아님 · 스냅숏 손상 — 사유를 보이고 자동화를 멈춘다(_abort_generation).
            # 항목은 대기열에 남겨 둔다(백엔드를 되돌리면 다시 쓸 수 있다).
            started = False
            self._abort_generation(str(exc))
        if not started:
            # '처리 중'으로 남기면 다음 자동화의 첫 덱 장이 큐 항목으로 취급되고, 끝나면 이 항목이 지워진다.
            # 자동화가 건 표시만 푼다(_abort_generation → _stop_automation 이 먼저 풀었으면 할 일 없음) —
            # 예전 set_processing(False) 는 대기열 매니저의 표시까지 지웠다.
            self._auto_processing_queue = False
            self._auto_queue_item_id = None
            try:
                qp.release_processing(item_id, AUTOMATION_OWNER)
            except Exception:
                pass
            return
        self._emit_auto_status()

    def _release_automation_queue_item(self):
        """자동화가 '생성 중'으로 표시한 큐 우선 항목의 표시만 푼다(대기열 매니저의 표시는 그대로)."""
        from core.queue_model import AUTOMATION_OWNER
        item_id = getattr(self, '_auto_queue_item_id', None)
        self._auto_queue_item_id = None
        if item_id is None or not getattr(self, '_auto_processing_queue', False):
            return
        qp = getattr(self, 'queue_panel', None)
        release = getattr(qp, 'release_processing', None)
        if callable(release):
            try:
                release(item_id, AUTOMATION_OWNER)
            except Exception:
                pass

    def _automation_queue_item_deferred(self, reason: str = '') -> bool:
        """'큐 우선' 동결 항목이 백엔드에 닿기 전에 멈췄다(워커의 ``_queue_deferred`` — 다른 작업이 GPU
        리스를 쥐었거나 항목을 만든 백엔드가 아니다). 항목은 소비하지 않고 자동화를 일시정지한다.

        대기열 매니저의 on_generation_deferred 와 같은 규칙: 자동화가 건 '생성 중' 표시만 풀어 항목을
        대기열에 남기고, 재개하면 _continue_automation 이 큐 우선 → 덱 순서로 같은 항목부터 다시 잇는다.
        실패가 아니므로 재시도 횟수·장 수를 건드리지 않는다. 예전엔 이 결과가 (자동화 중엔 멈춰 있는)
        대기열 매니저에게만 가서 버려져, 자동화가 '실행 중'인 채 조용히 멈췄고 재개할 지점
        (_auto_resume_pending)도 없어 일시정지→재개로도 되살릴 수 없었다.
        반환: 자동화가 이 결과를 맡았으면 True.
        """
        if not getattr(self, 'is_automating', False) or not getattr(self, '_auto_processing_queue', False):
            return False
        self._release_automation_queue_item()   # _auto_processing_queue 가 True 일 때만 푼다 — 먼저 부른다
        self._auto_processing_queue = False
        self._auto_retry_count = 0
        self._auto_paused = True
        self._auto_resume_pending = 'continue'
        self._wait_paused_remaining_ms = None
        detail = str(reason or '').strip()
        self.show_status(
            "⏸ 자동화 일시정지 — 대기열 항목을 보내지 못했습니다" + (f": {detail}" if detail else ''), 5000)
        self._emit_auto_status()
        return True

    def _auto_generation_in_flight(self) -> bool:
        """생성 워커가 아직 결과를 내지 않았는가(= 지금 한 장이 GPU 에 있다)."""
        worker = getattr(self, 'gen_worker', None)
        try:
            active = worker is not None and worker.isRunning()
        except RuntimeError:
            active = False   # 지워진 Qt 래퍼는 돌고 있지 않다
        return bool(active and getattr(worker, '_result_emitted', False) is not True)

    def _emit_auto_status(self, waiting=False):
        """Vue에 자동화 상태 전송."""
        if hasattr(self, 'vue_bridge'):
            import json
            # 덱 현황 — 남은/전체/사용. 중복 허용 모드는 덱을 소모 안 하므로
            # remaining==total로 유지됨(무한). UI에서 그 경우 구분 표시 가능.
            # '전체'는 등급 필터를 통과한 풀 행 수다 — 필터 전 풀 크기로 세면 시작하자마자
            # 걸러진 행만큼 '사용'으로 잡혔다(core.search_deck.deck_pool_size, 캐시됨).
            from core.search_deck import deck_pool_size
            _deck = getattr(self, 'shuffled_prompt_deck', None) or []
            _remaining = len(_deck)
            _total = deck_pool_size(self)
            _used = max(0, _total - _remaining)
            # 대기 카운트다운 (다음 생성까지 남은 시간) — Search % 바 느낌
            import time as _t
            _wait_total = int(getattr(self, '_wait_total_ms', 0) or 0)
            _wait_remaining = 0
            _frozen = getattr(self, '_wait_paused_remaining_ms', None)
            if _frozen is not None:
                # 일시정지가 대기 중에 걸렸다 — 얼려 둔 잔여 시간을 그대로 비춘다.
                waiting = True
                _wait_remaining = max(0, int(_frozen))
            elif waiting:
                _wait_remaining = max(0, int((getattr(self, '_wait_end_time', 0) - _t.time()) * 1000))
            # 다음 생성에 나갈 프롬프트.
            #   total_prompt_display 는 _build_generation_payload 가 '그대로 읽어' API 로
            #   보내는 바로 그 문자열이다. 덱에서 뽑는 순간(apply_prompt_from_data)에
            #   제외 프롬프트 · 캐릭터 특징 · 조건식(1·2차) · 와일드카드 치환까지 모두
            #   끝나 이 상자에 들어오므로, 대기 중에도 이미 확정된 값을 보낼 수 있다.
            #   아직 반영되지 않는 것 = 생성 직전에야 붙는 3가지:
            #     · run_pipeline_on_text 훅(인스턴트 와일드카드 $$name$$, 중복 제거)
            #     · LoRA 스택 텍스트(<lora:...>) 꼬리
            #     · 태그→자연어 자동 변환(켰을 때만, 생성 직전 비동기)
            #   이 셋은 여기서 미리 계산할 수 없다(훅·와일드카드는 매 호출 결과가 달라져
            #   미리 돌리면 실제로 나갈 값과 어긋난다). 그래서 '지금 알 수 있는 가장
            #   가까운 값'인 이 상자를 보낸다 — 사람이 손볼 대상도 이 문자열이다.
            # 덮어쓰기가 걸려 있으면 그 값이 우선 — 방금 고친 게 그대로 보여야 한다.
            _override = getattr(self, '_auto_prompt_override', '') or ''
            _prompt = _override
            if not _prompt:
                try:
                    _prompt = self.total_prompt_display.toPlainText()
                except Exception:
                    _prompt = ''
            # 그 프롬프트가 '다음 덱 장'에 그대로 쓰이는가 — 아니면(생성 중인 마지막 반복 등) 패널이
            # 편집을 막는다. 거기 건 편집은 새로 뽑는 순간 버려진다(core.automation_prompt_state).
            from core.automation_prompt_state import prompt_is_next
            try:
                _rpp = int((getattr(self, 'auto_settings', None) or {}).get('repeat_per_prompt', 1) or 1)
            except (TypeError, ValueError):
                _rpp = 1
            _prompt_is_next = bool(self.is_automating) and prompt_is_next(
                generating=self._auto_generation_in_flight(),
                held_after_generation=(getattr(self, '_auto_resume_pending', '') == 'continue'),
                processing_queue=bool(getattr(self, '_auto_processing_queue', False)),
                queue_dirtied=bool(getattr(self, '_queue_dirtied_prompt', False)),
                override_pending=bool(_override),
                override_used=bool(getattr(self, '_auto_override_used', False)),
                current_repeat=getattr(self, 'auto_current_repeat', 0) or 0,
                repeat_per_prompt=_rpp,
            )
            self.vue_bridge.automationStatus.emit(json.dumps({
                'running': self.is_automating,
                'count': getattr(self, 'auto_gen_count', 0),
                'waiting': waiting,
                'wait_remaining_ms': _wait_remaining,
                'wait_total_ms': _wait_total,
                'deck_remaining': _remaining,
                'deck_total': _total,
                'deck_used': _used,
                'allow_duplicates': bool(getattr(self, 'auto_settings', {}).get('allow_duplicates', False)),
                'paused': bool(getattr(self, '_auto_paused', False)),
                'prompt': _prompt or '',
                'prompt_is_next': _prompt_is_next,
            }))

    def _stop_automation(self, message=None):
        """자동화 중지"""
        self.is_automating = False
        # 이 회차에 예약된 실패 재시도는 무효 — 곧바로 다시 시작해도 새 회차로 새지 않는다.
        self._auto_run_epoch = getattr(self, '_auto_run_epoch', 0) + 1
        self._auto_retry_count = 0
        # 아직 도는 태그→자연어 변환의 결과 연결을 끊는다 — 중지 뒤 도착한 결과가 프롬프트를 덮거나
        # 재시작한 회차의 생성을 부르지 않게(이미 이벤트 큐에 들어간 결과는 회차 비교로 버린다:
        # _auto_nl_result_is_current). 진행 중인 HTTP 자체는 취소할 수 없다.
        try:
            from workers.ollama_worker import detach_result_signals
            detach_result_signals(getattr(self, '_auto_nl_worker', None))
        except Exception:
            pass
        # 모아 둔 덱 진행도(뽑을 때마다 쓰지 않는다)를 지금 저장 — 재시작 때 이어서 뽑게.
        self._flush_deck_state()
        # 진행 중 생성도 실제로 중단 (cancel → 백엔드 interrupt).
        # 사이클 사이에 불리면 워커가 없어 no-op.
        worker = getattr(self, 'gen_worker', None)
        if worker is not None and hasattr(worker, 'cancel') and worker.isRunning():
            try:
                worker.cancel()
            except Exception:
                pass
        # 큐 우선 항목 도중에 멈췄으면 그 항목은 소비되지 않고 남는다 — '생성 중' 표시를 풀어
        # 대기열에서 다시 지우거나 옮길 수 있게 한다.
        self._release_automation_queue_item()
        # 대기 카운트다운 정지
        if getattr(self, '_wait_timer', None):
            self._wait_timer.stop()
        self._wait_end_time = 0
        self._wait_total_ms = 0
        # 일시정지·일회성 덮어쓰기는 자동화 한 회차짜리 상태다 — 중지와 함께 사라진다.
        # (남겨 두면 다음 '시작'이 멈춘 채로 뜨거나 남의 프롬프트로 첫 장이 나간다)
        self._auto_paused = False
        self._auto_resume_pending = ''
        self._wait_paused_remaining_ms = None
        if getattr(self, '_auto_prompt_override', '') or getattr(self, '_auto_override_used', False):
            self._auto_prompt_override = ''
            self._auto_override_used = False
            try:
                self.update_total_prompt_display()   # 덮어쓴 상자를 원래대로
            except Exception:
                pass
        self._emit_auto_status()
        # 설정 '생성 후 모델 언로드' — 자동화 마지막 장 뒤. 워커가 아직 돌면(수동 중지 직후) 건너뛴다.
        if hasattr(self, '_maybe_unload_models_after_generation'):
            self._maybe_unload_models_after_generation()

        # 버튼 상태 복구 (자동화 모드는 유지)
        if self.btn_auto_toggle.isChecked():
            self.btn_generate.setText("🚀 자동화 시작")
            self.btn_generate.setStyleSheet(_gen_btn_style('#27ae60'))
        else:
            self.btn_generate.setText("✨ 이미지 생성")
            self.btn_generate.setStyleSheet(_gen_btn_style(_gen_btn_default_color()))

        self.btn_generate.setEnabled(True)
        
        if message:
            self.show_status(message)
            QMessageBox.information(self, "자동화", message)
        else:
            self.show_status(f"✅ 자동화 완료: {self.auto_gen_count}장 생성됨")
            
    def receive_event_scenarios(self, scenarios):
        """이벤트 시나리오를 대기열에 추가"""
        added_count = 0
        for scenario in scenarios:
            payload = scenario.get('payload', {})

            if not payload or 'prompt' not in payload:
                _logger.warning(f"잘못된 시나리오: {scenario}")
                continue

            self.queue_panel.add_single_item(payload)
            added_count += 1

        self.show_status(f"✅ {added_count}개의 이벤트가 대기열에 추가됨")
        QMessageBox.information(
            self, "전송 완료",
            f"{added_count}개의 이벤트가 대기열에 추가되었습니다."
        )

    def _greedy_merge_words(self, words):
        """공백 구분 단어들을 알려진 다중단어 태그로 greedy 결합 (최장 우선).
        예: [tokyo, afterschool, summoners] → [tokyo afterschool summoners].
        DB에 없는 조합은 단일 단어로 유지."""
        try:
            from core.tag_intelligence import get_tag_intelligence
            ti = get_tag_intelligence()
        except Exception:
            return list(words)
        out, i, n = [], 0, len(words)
        while i < n:
            took = 1
            for j in range(min(5, n - i), 1, -1):   # 5..2 단어 조합을 길이순으로
                cand = ' '.join(words[i:i + j])
                if ti.is_known(cand) or ti.is_character(cand) or ti.is_copyright(cand):
                    out.append(cand)
                    took = j
                    break
            else:
                out.append(words[i])
            i += took
        return out

    def handle_prompt_only_transfer(self, prompt, negative):
        """PNG Info/Gallery에서 프롬프트만 전송"""
        import re
        # ② LoRA/LyCO/hypernet 토큰 제거 — main에 들어가면 LoRA STACK과 충돌
        def _strip_lora(s):
            s = re.sub(r'<(?:lora|lyco|lycoris|hypernet|lokr|loha|ip-?adapter):[^>]*>', '',
                       s or '', flags=re.IGNORECASE)
            return re.sub(r'\s*,\s*,\s*', ', ', s).strip().strip(',').strip()
        prompt = _strip_lora(prompt)
        negative = _strip_lora(negative)
        # ① 스마트 토큰화: 콤마 있으면 콤마, 없으면 공백(언더스코어→공백) 기준으로 각 태그를
        #    개별 분류 (공백 구분 프롬프트가 통째로 1덩어리→general(main)로 새던 버그 수정)
        raw = (prompt or '').strip()
        if ',' in raw:
            tokens = [t.strip() for t in raw.split(',') if t.strip()]
        else:
            words = [w for w in raw.split() if w.strip()]
            if any('_' in w for w in words):
                # 언더스코어 태그가 공백 구분 (white_hair blue_eyes) → 그대로 변환
                tokens = [w.replace('_', ' ').strip() for w in words]
            else:
                # 순수 공백 구분 단어 → 알려진 다중단어 태그로 greedy 결합
                # (tokyo afterschool summoners 가 3개로 쪼개지던 버그 수정)
                tokens = self._greedy_merge_words(words)
        classified = self.tag_classifier.classify_tags_for_event(tokens)
        # 분류 보강: wiki 분류기(265k)가 못 잡은 캐릭터/작품을 저장된 캐릭터 프리셋 +
        # tag_intelligence(character_profiles 34k)로 재분류 → general(main) 누수 방지
        try:
            from core.tag_intelligence import get_tag_intelligence
            from utils.character_presets import list_character_presets
            ti = get_tag_intelligence()
            saved = set(list_character_presets() or [])

            def _n(t):
                return t.strip().lower().replace("_", " ").replace(r"\(", "(").replace(r"\)", ")")
            leftover = []
            for t in classified.get("general", []):
                n = _n(t)
                if n in saved or ti.is_character(t):
                    classified["character"].append(t)
                elif ti.is_copyright(t):
                    classified["copyright"].append(t)
                else:
                    leftover.append(t)
            classified["general"] = leftover
        except Exception:
            pass
        bundle = {
            # count(인물수)를 general 앞에 포함 → apply_prompt_from_data가 인물수
            # 섹션(char_count_input)으로 분리. (과거엔 count를 빼서 1girl/2boys 등이
            # 당겨오기 시 통째로 누락됐음.)
            'general': ', '.join(classified.get("count", []) + classified["costume"] + classified["appearance"] + classified["expression"] + classified["action"] + classified["background"] + classified["composition"] + classified["effect"] + classified["objects"] + classified["general"]),
            'character': ', '.join(classified["character"]),
            'copyright': ', '.join(classified["copyright"]),
            'artist': ''
        }
        # preserve_locked=True → 선행/후행/작가 칸은 덮어쓰지 않고, 그 칸들과 겹치는
        # 태그는 당겨오지 않음 (인물수/캐릭터/main은 그대로 override)
        # comma_only=True → 당겨오기는 항상 콤마 구분이므로, 단일 다중단어 태그
        # ('genshin impact')가 공백으로 쪼개지지(망가지지) 않게 함
        self.apply_prompt_from_data(bundle, preserve_locked=True, comma_only=True)
        self.neg_prompt_text.setPlainText(negative)
        # Vue에서 T2I 탭으로 전환 유도
        if hasattr(self, 'vue_bridge'):
            self.vue_bridge.tabChanged.emit('t2i')
        self.show_status("✅ 프롬프트 전송 완료")
