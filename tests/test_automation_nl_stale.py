"""자동화 태그→자연어 변환 결과가 중지·재시작 뒤에 늦게 도착할 때 — 회귀.

예전엔 _stop_automation 이 변환 워커를 떼지 않았고 _on_auto_nl_done 은 회차를 보지 않고 곧바로
프롬프트 상자를 덮었다. 그래서
  - 중지 뒤 사용자가 고친 프롬프트가 옛 변환 결과로 사라졌고(_auto_nl_last_output 도 오염),
  - 재시작한 회차의 첫 장(또는 반복 장)이 옛 회차 프롬프트로 나갔으며, 새로 뽑은 덱 장은 생성
    없이 소비됐고, 재시작 자신의 첫 생성은 '이미지 생성 중' 경고로 거절됐다.
이제 요청마다 회차(_auto_nl_epoch)를 적고, 중지가 결과 연결을 끊으며, 이미 큐에 들어간 결과는
회차 비교(_auto_nl_result_is_current)로 버린다.
"""
import io
import json
import unittest
from unittest import mock

from PyQt6.QtCore import QTimer

from tests.test_automation_retry import _Host


class _FakeSignal:
    """pyqtSignal 대역 — detach_result_signals 가 쓰는 인자 없는 disconnect() 까지."""

    def __init__(self):
        self.slots = []

    def connect(self, fn):
        self.slots.append(fn)

    def disconnect(self, *_):
        if not self.slots:
            raise TypeError('disconnect() failed between signal and all slots')
        self.slots.clear()

    def emit(self, *args):
        for fn in list(self.slots):
            fn(*args)


class _FakeOllamaWorker:
    """OllamaWorker 대역 — 결과는 테스트가 원하는 때에 낸다(HTTP 없음)."""

    def __init__(self, url, model, text, mode, extra, parent=None, **_kw):
        self.text = text
        self.finished = _FakeSignal()
        self.error = _FakeSignal()
        self.started = False

    def start(self):
        self.started = True


def _nl(text):
    return json.dumps({'tags': text})


class _NlHost(_Host):
    """메인 프롬프트 상자(main_prompt_text) = 총 프롬프트 상자로 단순화한 자동화 호스트.

    start_generation 은 실제 앱처럼 생성 중이면 거절한다('이미지 생성 중입니다' 경고 경로).
    """

    def __init__(self, prompts, **settings):
        super().__init__(prompts, **settings)
        self._auto_nl_enabled = True
        self.generating = False
        self.refused = []

    def _show(self, text):
        self.main_prompt_text.setPlainText(text)
        self.total_prompt_display.setPlainText(text)

    def apply_random_prompt(self):
        if not self.shuffled_prompt_deck:
            return False
        bundle = self.shuffled_prompt_deck.pop()
        self._current_auto_bundle = bundle
        self._show(bundle['general'])
        return True

    def apply_prompt_from_data(self, bundle):
        self._show(bundle['general'])

    def update_total_prompt_display(self):
        self.total_prompt_display.setPlainText(self.main_prompt_text.toPlainText())

    def start_generation(self):
        prompt = self.total_prompt_display.toPlainText()
        if self.generating:
            self.refused.append(prompt)
            return False
        self.generating = True
        self.started.append((prompt, bool(getattr(self, '_auto_processing_queue', False))))
        return True

    def finish(self, success):
        self.generating = False
        super().finish(success)


class AutomationNlStaleResultTests(unittest.TestCase):
    def setUp(self):
        self.host = None
        patches = (
            mock.patch.object(QTimer, 'singleShot',
                              side_effect=lambda ms, fn: self.host.timers.append((ms, fn))),
            mock.patch('ui.generator_actions.QMessageBox'),
            mock.patch('workers.ollama_worker.OllamaWorker', _FakeOllamaWorker),
            mock.patch('workers.ollama_worker.release_when_done', lambda *a, **k: None),
        )
        for patcher in patches:
            patcher.start()
            self.addCleanup(patcher.stop)

    def _run_until_nl_pending(self, host):
        """P1(첫 장 — 변환 없이) 생성 → 끝 → P2 를 뽑아 변환 요청이 걸린 상태까지."""
        self.host = host
        host._start_automation()
        host.run_timers()
        self.assertEqual(host.started, [('P1', False)])
        host.finish(True)
        host.run_timers()
        worker = host._auto_nl_worker
        self.assertIsInstance(worker, _FakeOllamaWorker)
        self.assertEqual(worker.text, 'P2')
        self.assertEqual(len(host.started), 1, '변환이 끝나야 P2 가 나간다')
        return worker

    def test_result_after_stop_does_not_overwrite_the_users_edit(self):
        host = _NlHost(['P1', 'P2', 'P3'])
        worker = self._run_until_nl_pending(host)
        host._stop_automation()
        self.assertEqual(worker.finished.slots, [], '중지가 결과 연결을 끊는다')
        self.assertEqual(worker.error.slots, [])
        host._show('USER EDIT after stop')
        worker.finished.emit(_nl('A bird. It flies.'))       # 끊겨서 닿지 않는다
        host._on_auto_nl_done(_nl('A bird. It flies.'))      # 이미 큐에 들어가 있던 결과
        self.assertEqual(host.main_prompt_text.toPlainText(), 'USER EDIT after stop')
        self.assertEqual(host.total_prompt_display.toPlainText(), 'USER EDIT after stop')
        self.assertIsNone(getattr(host, '_auto_nl_last_output', None))
        self.assertEqual(len(host.started), 1)
        self.assertEqual(host.refused, [])

    def test_restart_with_delay_uses_the_newly_drawn_card(self):
        host = _NlHost(['P1', 'P2', 'P3'], delay=1)
        self.host = host                           # delay>0 은 대기 타이머 경유 — 직접 구동한다
        host._start_automation()
        host.run_timers()
        host.finish(True)                          # P2 를 뽑고 대기 타이머
        host._automation_generate()                # 대기 끝 → P2 변환 요청
        worker = host._auto_nl_worker
        self.assertEqual(worker.text, 'P2')
        host._stop_automation()
        host._start_automation()                   # 새 회차 — P3 를 뽑고 delay 뒤 첫 생성 예약
        self.assertEqual(host.main_prompt_text.toPlainText(), 'P3')
        host._on_auto_nl_done(_nl('A bird. It flies.'))      # 옛 회차 결과가 첫 생성 전에 도착
        self.assertEqual(host.main_prompt_text.toPlainText(), 'P3')
        host.run_timers()                          # 재시작의 첫 생성
        self.assertEqual([p for p, _q in host.started], ['P1', 'P3'])
        self.assertEqual(host.refused, [], "'이미지 생성 중' 거절이 없어야 한다")

    def test_restart_with_repeat_keeps_every_repeat_on_the_new_prompt(self):
        host = _NlHost(['P1', 'P2', 'P3'], repeat_per_prompt=1)
        stale = self._run_until_nl_pending(host)
        host._stop_automation()
        host.settings['repeat_per_prompt'] = 2
        host._start_automation()
        host.run_timers()                          # 새 회차 첫 장 P3 생성 중
        self.assertEqual(host.started[-1], ('P3', False))
        host._on_auto_nl_done(_nl('A bird. It flies.'))      # 옛 결과가 생성 도중 도착
        self.assertEqual(host.main_prompt_text.toPlainText(), 'P3')
        self.assertEqual(host.refused, [])
        host.finish(True)                          # 반복 2번째 — 이 회차의 변환을 요청
        host.run_timers()
        fresh = host._auto_nl_worker
        self.assertIsNot(fresh, stale)
        self.assertEqual(fresh.text, 'P3')
        fresh.finished.emit(_nl('A cat.'))
        self.assertEqual([p for p, _q in host.started], ['P1', 'P3', 'P3, A cat.'])
        self.assertEqual(host.refused, [])

    def test_stale_error_after_restart_does_not_start_an_extra_generation(self):
        host = _NlHost(['P1', 'P2', 'P3'], delay=1)
        self.host = host
        host._start_automation()
        host.run_timers()
        host.finish(True)
        host._automation_generate()
        host._stop_automation()
        host._start_automation()
        host._on_auto_nl_error('timeout')          # 옛 회차의 실패 — 태그로 생성하지 않는다
        self.assertEqual([p for p, _q in host.started], ['P1'])
        host.run_timers()
        self.assertEqual([p for p, _q in host.started], ['P1', 'P3'])
        self.assertEqual(host.refused, [])

    def test_result_of_the_current_run_is_still_applied_once(self):
        host = _NlHost(['P1', 'P2', 'P3'])
        worker = self._run_until_nl_pending(host)
        worker.finished.emit(_nl('A bird. It flies.'))
        self.assertEqual(host.main_prompt_text.toPlainText(), 'P2, A bird. It flies.')
        self.assertEqual(host._auto_nl_last_output, 'P2, A bird. It flies.')
        self.assertEqual([p for p, _q in host.started], ['P1', 'P2, A bird. It flies.'])

    def test_error_of_the_current_run_generates_with_the_tags(self):
        host = _NlHost(['P1', 'P2', 'P3'])
        worker = self._run_until_nl_pending(host)
        worker.error.emit('Ollama down')
        self.assertEqual([p for p, _q in host.started], ['P1', 'P2'])

    def test_logs_never_break_the_handlers_on_a_cp949_pipe(self):
        # /verify·/ship 은 run_tests.py 를 PYTHONIOENCODING 없이 파이프로 돌린다 — 한국어 Windows 에서
        # stdout 은 cp949(strict)이고 print('—') 는 UnicodeEncodeError 로 슬롯을 깼다. 이 회차의
        # 실패 문구(err)에 그런 문자가 있으면 태그로 생성하는 다음 장까지 건너뛰었다.
        pipe = io.TextIOWrapper(io.BytesIO(), encoding='cp949', errors='strict')
        with mock.patch('sys.stdout', pipe):
            host = _NlHost(['P1', 'P2', 'P3'])
            worker = self._run_until_nl_pending(host)
            worker.error.emit('Ollama 응답 없음 — 시간 초과')    # 이 회차의 실패 → 태그로 생성
            self.assertEqual([p for p, _q in host.started], ['P1', 'P2'])
            host._stop_automation()
            host._on_auto_nl_done(_nl('A bird.'))                   # 중지 뒤 결과 — 버린다
            host._on_auto_nl_error('timeout — late')           # 중지 뒤 실패 — 버린다
            pipe.flush()
            logged = pipe.buffer.getvalue().decode('cp949')
        self.assertIn('[AutoNL] 변환 실패(태그로 생성): Ollama 응답 없음 \\u2014 시간 초과', logged)
        self.assertIn('[AutoNL] 중지·재시작된 회차의 변환 결과 - 버림', logged)
        self.assertIn('[AutoNL] 중지·재시작된 회차의 변환 실패 - 버림: timeout \\u2014 late', logged)
        self.assertEqual([p for p, _q in host.started], ['P1', 'P2'], '버린 결과는 생성을 부르지 않는다')


if __name__ == '__main__':
    unittest.main()
