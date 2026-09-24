"""Headless Qt/public action tests with an external generation backend fake."""
import json
import tempfile
import threading
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from PyQt6.QtCore import QCoreApplication, QObject, pyqtSignal
from ui.chat_actions import ChatActionsMixin


class Bridge(QObject):
    chatGenerationEvent = pyqtSignal(str)
    creatorProgress = pyqtSignal(str)
    creatorResult = pyqtSignal(str)
    chatDone = pyqtSignal(str)


class Host(QObject, ChatActionsMixin):
    def __init__(self):
        super().__init__()
        self.vue_bridge = Bridge(self)
        self.unload_enabled = False
        self.unload_threads = []
        self.policy_threads = []

    def _creator_should_unload_ollama(self):
        self.policy_threads.append(threading.get_ident())
        return self.unload_enabled

    def _creator_unload_ollama(self):
        self.unload_threads.append(threading.get_ident())
        return True

    def _chat_generation_snapshot(self, prompt):
        # UI settings boundary, no widgets or backend state are mutated.
        return 'selected-anima', {'prompt': prompt, 'steps': 8}


class Backend:
    def __init__(self, blocked=False):
        self.started = threading.Event()
        self.release = threading.Event()
        if not blocked:
            self.release.set()
        self.interrupts = 0
        self.worker_thread = None

    def txt2img(self, model, payload, progress_callback=None):
        self.worker_thread = threading.get_ident()
        self.started.set()
        self.release.wait(3)
        return SimpleNamespace(success=True, image_data=b'fake-result', artifacts=[], info={})

    def interrupt(self):
        self.interrupts += 1
        self.release.set()


class BlockingUnload:
    """Forge unload-checkpoint 대역 — release 전까지 공유 리스를 hold(HOLD_PHASE)로 쥔다."""

    def __init__(self):
        self.entered = threading.Event()
        self.release = threading.Event()

    def unload_checkpoint(self):
        self.entered.set()
        self.release.wait(5)
        return True


class ChatGenerationActionsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QCoreApplication.instance() or QCoreApplication([])

    def wait_for(self, predicate):
        end = time.monotonic() + 4
        while not predicate() and time.monotonic() < end:
            self.app.processEvents()
            time.sleep(.002)
        self.assertTrue(predicate(), 'request did not complete')

    @staticmethod
    def request(request_id):
        return {'id': request_id, 'model': '', 'generation': {'mode': 'image', 'family': 'current'},
                'messages': [{'role': 'user', 'content': 'cat'}]}

    def test_direct_generation_without_llm_runs_off_ui_and_publishes_owned_media(self):
        host, backend, events = Host(), Backend(), []
        host.vue_bridge.chatGenerationEvent.connect(lambda raw: events.append(json.loads(raw)))
        with tempfile.TemporaryDirectory() as directory, patch('backends.get_backend', return_value=backend), patch('config.OUTPUT_DIR', directory):
            self.assertTrue(host._handle_chat_action('chat_send', self.request('one')))
            self.wait_for(lambda: any(e.get('done') for e in events))
            final = next(e for e in events if e.get('done'))
            self.assertTrue(final['ok'], final.get('error'))
            self.assertTrue(Path(final['artifacts'][0]['path']).is_file())
            self.assertNotEqual(backend.worker_thread, threading.get_ident())
            self.assertTrue(all(e['id'] == 'one' for e in events))

    def test_current_generation_reads_unload_policy_and_runs_hook_off_ui(self):
        for enabled in (False, True):
            host, backend, events = Host(), Backend(), []
            host.unload_enabled = enabled
            host.vue_bridge.chatGenerationEvent.connect(lambda raw: events.append(json.loads(raw)))
            with self.subTest(enabled=enabled), tempfile.TemporaryDirectory() as directory, patch('backends.get_backend', return_value=backend), patch('config.OUTPUT_DIR', directory):
                host._handle_chat_action('chat_send', self.request('policy'))
                self.wait_for(lambda: any(e.get('done') for e in events))
                self.assertTrue(next(e for e in events if e.get('done'))['ok'])
                self.assertEqual(len(host.policy_threads), 1)
                self.assertNotEqual(host.policy_threads[0], threading.get_ident())
                self.assertEqual(len(host.unload_threads), 1 if enabled else 0)
                if enabled:
                    self.assertEqual(host.unload_threads[0], backend.worker_thread)

    def test_request_during_a_model_unload_waits_for_it_instead_of_being_rejected(self):
        # 언로드 hold(HOLD_PHASE)는 생성이 아니다. 예전엔 사전 검사가 phase != 'idle' 을 봐서 생성 후
        # 언로드·대기열 정기 정리·VRAM 수동 언로드 직후(최대 30초) 채팅 이미지·영상 요청을 '다른 생성
        # 작업이 실행 중'으로 거절했다. 이제 받아 두고 작업 스레드가 그 언로드를 기다렸다 생성한다.
        from core import post_generation
        from core.resource_coordinator import HOLD_PHASE, get_generation_coordinator
        unload = BlockingUnload()
        host, backend, events, rejected = Host(), Backend(), [], []
        host.vue_bridge.chatGenerationEvent.connect(lambda raw: events.append(json.loads(raw)))
        host.vue_bridge.chatDone.connect(lambda raw: rejected.append(json.loads(raw)))
        thread = post_generation.start_post_generation_unload(unload)
        self.assertIsNotNone(thread, '앞선 테스트의 언로드가 남아 있으면 안 된다')
        try:
            self.assertTrue(unload.entered.wait(5))
            self.assertEqual(get_generation_coordinator().state.phase, HOLD_PHASE)
            with tempfile.TemporaryDirectory() as directory, patch('backends.get_backend', return_value=backend), \
                    patch('config.OUTPUT_DIR', directory):
                host._handle_chat_action('chat_send', self.request('during-unload'))
                self.assertEqual(rejected, [], '언로드 중이라고 거절하지 않는다')
                time.sleep(0.05)
                self.assertFalse(backend.started.is_set(), '언로드가 끝나기 전에는 샘플링하지 않는다')
                unload.release.set()
                self.wait_for(lambda: any(e.get('done') for e in events))
                final = next(e for e in events if e.get('done'))
                self.assertTrue(final['ok'], final.get('error'))
                self.assertTrue(Path(final['artifacts'][0]['path']).is_file())
        finally:
            unload.release.set()
            thread.join(5)

    def test_stop_and_results_are_owned_and_duplicate_request_does_not_replace_job(self):
        host, backend, events, rejected = Host(), Backend(blocked=True), [], []
        host.vue_bridge.chatGenerationEvent.connect(lambda raw: events.append(json.loads(raw)))
        host.vue_bridge.chatDone.connect(lambda raw: rejected.append(json.loads(raw)))
        with tempfile.TemporaryDirectory() as directory, patch('backends.get_backend', return_value=backend), patch('config.OUTPUT_DIR', directory):
            try:
                host._handle_chat_action('chat_send', self.request('owned'))
                self.wait_for(backend.started.is_set)
                host.vue_bridge.creatorResult.emit(json.dumps({'requestId': 'manual-creator', 'ok': True,
                                                               'artifacts': [{'kind': 'image', 'path': 'unrelated.png'}]}))
                host._handle_chat_action('chat_stop', {'id': 'unrelated'})
                host._handle_chat_action('chat_send', self.request('second'))
                self.assertEqual(backend.interrupts, 0)
                self.assertFalse(any(e.get('done') for e in events))
                self.assertEqual(rejected[0]['id'], 'second')
                host._handle_chat_action('chat_stop', {'id': 'owned'})
                self.wait_for(lambda: any(e.get('done') for e in events))
                final = next(e for e in events if e.get('done'))
                self.assertTrue(final['stopped'])
                self.assertEqual(final['artifacts'], [])
                self.assertEqual(backend.interrupts, 1)
                self.assertFalse(list(Path(directory).rglob('chat_*')))
            finally:
                backend.release.set()
