"""설치 모델 대조(resolve_model)와 OllamaWorker 의 워커-스레드 대조 — 가짜 HTTP 만.

예전엔 같은 11줄 블록이 ollamaEnhance·convertPromptToNl·자동화 3곳에 복제돼 GUI 스레드에서
/api/tags(timeout 5)를 동기로 불렀고, 'gemma3:4b' 요청에 'gemma3:12b' 만 설치돼 있으면
계열이 같다는 이유로 요청 이름을 그대로 보내 404('빈 응답')가 났다.
"""
from __future__ import annotations

import json
import sys
import threading
import unittest
from unittest.mock import patch

from PyQt6.QtCore import QCoreApplication, QObject

from core.ollama_client import DEFAULT_OLLAMA_MODEL, DEFAULT_OLLAMA_URL, resolve_installed_model, resolve_model
# ui.* 는 QtWebEngine 을 끌어오므로 QCoreApplication 보다 먼저 import 해야 한다
from ui.generator_actions import ActionsMixin
from ui.vue_bridge import VueBridge


class _Resp:
    status_code = 200

    def __init__(self, data):
        self._data = data

    def raise_for_status(self):
        return None

    def json(self):
        return self._data


class ResolveModelTests(unittest.TestCase):
    def test_exact_name_wins(self):
        self.assertEqual(resolve_model('gemma3:4b', ['llava:7b', 'gemma3:4b']), 'gemma3:4b')

    def test_same_model_ignores_latest_tag_and_case(self):
        self.assertEqual(resolve_model('qwen3', ['llava:7b', 'qwen3:latest']), 'qwen3:latest')
        self.assertEqual(resolve_model('Qwen3:Latest', ['qwen3']), 'qwen3')

    def test_other_tag_of_same_family_is_substituted_instead_of_passing_through(self):
        # 4b 요청, 12b 만 설치 → 설치된 12b (예전엔 'gemma3:4b' 그대로 보내 404)
        self.assertEqual(resolve_model('gemma3:4b', ['llava:7b', 'gemma3:12b']), 'gemma3:12b')

    def test_unknown_or_empty_request_falls_back_to_first_installed(self):
        self.assertEqual(resolve_model('mistral:7b', ['llava:7b', 'gemma3:12b']), 'llava:7b')
        self.assertEqual(resolve_model('', ['llava:7b']), 'llava:7b')
        self.assertEqual(resolve_model('  ', ['llava:7b']), 'llava:7b')

    def test_without_installed_list_keeps_request_or_default(self):
        self.assertEqual(resolve_model('custom:1b', []), 'custom:1b')
        self.assertEqual(resolve_model('', []), DEFAULT_OLLAMA_MODEL)
        self.assertEqual(resolve_model('', None, 'x:1'), 'x:1')

    def test_registry_port_and_hf_names_are_not_split_on_the_wrong_colon(self):
        installed = ['host:5000/team/model', 'hf.co/org/repo:Q4_K_M']
        self.assertEqual(resolve_model('host:5000/team/model:latest', installed), 'host:5000/team/model')
        self.assertEqual(resolve_model('hf.co/org/repo:BF16', installed), 'hf.co/org/repo:Q4_K_M')

    def test_resolve_installed_model_reads_tags_and_tolerates_a_dead_server(self):
        import requests
        with patch('core.ollama_client.requests.get',
                   return_value=_Resp({'models': [{'name': 'gemma3:12b'}]})) as get:
            self.assertEqual(resolve_installed_model('http://fake.local', 'gemma3:4b'), 'gemma3:12b')
        self.assertEqual(get.call_args.args[0], 'http://fake.local/api/tags')
        with patch('core.ollama_client.requests.get', side_effect=requests.ConnectionError('down')):
            self.assertEqual(resolve_installed_model('', 'wanted:1b'), 'wanted:1b')
        self.assertEqual(DEFAULT_OLLAMA_URL, 'http://localhost:11434')


class DefaultsContractTests(unittest.TestCase):
    """기본값 단일 출처 — 파이썬 상수와 프론트(utils/ollamaPrefs.ts · Settings 추천 'best')가 같아야 한다."""

    def _read(self, *parts):
        from pathlib import Path
        return Path(__file__).resolve().parents[1].joinpath(*parts).read_text(encoding='utf-8')

    def test_frontend_default_url_matches_backend(self):
        import re
        source = self._read('frontend', 'src', 'utils', 'ollamaPrefs.ts')
        match = re.search(r"export const DEFAULT_OLLAMA_URL = '([^']+)'", source)
        self.assertIsNotNone(match)
        self.assertEqual(match.group(1), DEFAULT_OLLAMA_URL)

    def test_backend_last_resort_model_is_the_settings_best_recommendation(self):
        import re
        source = self._read('frontend', 'src', 'views', 'SettingsView.vue')
        best = re.findall(r"\{\s*name:\s*'([^']+)',\s*best:\s*true", source)
        self.assertEqual(best, [DEFAULT_OLLAMA_MODEL])

    def test_duplicated_install_check_blocks_are_gone(self):
        import inspect
        for fn in (VueBridge.ollamaEnhance, VueBridge.convertPromptToNl, ActionsMixin._start_auto_nl_then_generate):
            with self.subTest(fn=fn.__name__):
                source = inspect.getsource(fn)
                self.assertNotIn('list_models', source, 'GUI 스레드에서 /api/tags 를 부르지 않는다')
                self.assertNotIn(".split(':')[0]", source)
                self.assertNotIn("'gemma3:4b'", source)
                self.assertNotIn('.wait(', source, '이전 워커를 기다리며 GUI 를 막지 않는다')
                self.assertNotIn('.quit(', source)
                self.assertIn('resolve_installed=True', source)


class OllamaWorkerResolutionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QCoreApplication.instance() or QCoreApplication(sys.argv)

    def setUp(self):
        from core.ollama_client import clear_thinking_mode_cache
        clear_thinking_mode_cache()
        self.addCleanup(clear_thinking_mode_cache)
        thinking = patch('core.ollama_client.OllamaClient.thinking_mode', return_value='unknown')
        thinking.start()
        self.addCleanup(thinking.stop)

    def _run(self, **kwargs):
        from workers.ollama_worker import OllamaWorker
        results, errors, models, threads = [], [], [], []

        def post(url, json=None, **_):
            models.append(json['model'])
            return _Resp({'message': {'content': 'blue_bird'}})

        def get(url, **_):
            threads.append(threading.current_thread().name)
            return _Resp({'models': [{'name': 'gemma3:12b'}]})

        worker = OllamaWorker('http://fake.local', kwargs.pop('model', 'gemma3:4b'), 'bird', 'expand',
                              instructions={}, **kwargs)
        worker.finished.connect(results.append)
        worker.error.connect(errors.append)
        with patch('core.ollama_client.requests.get', side_effect=get), \
                patch('core.ollama_client.requests.post', side_effect=post):
            worker.start()
            self.assertTrue(worker.wait(5000))
            self.app.processEvents()
        return results, errors, models, threads

    def test_opt_in_resolution_runs_on_the_worker_thread(self):
        results, errors, models, threads = self._run(resolve_installed=True)
        self.assertEqual(errors, [])
        self.assertEqual(json.loads(results[0])['tags'], 'blue_bird')
        self.assertEqual(models, ['gemma3:12b'])
        self.assertEqual(len(threads), 1)
        self.assertNotEqual(threads[0], threading.main_thread().name)

    def test_default_worker_sends_the_given_model_without_listing(self):
        results, errors, models, threads = self._run()
        self.assertEqual(errors, [])
        self.assertEqual(models, ['gemma3:4b'])
        self.assertEqual(threads, [], '옵트인이 아니면 /api/tags 를 부르지 않는다')
        _r, _e, models, _t = self._run(model='')
        self.assertEqual(models, [DEFAULT_OLLAMA_MODEL])


class BridgeSlotTests(unittest.TestCase):
    """슬롯은 워커만 띄우고 돌아온다 — GUI 스레드에서 /api/tags 를 부르지 않는다."""

    @classmethod
    def setUpClass(cls):
        cls.app = QCoreApplication.instance() or QCoreApplication(sys.argv)

    def setUp(self):
        from core.ollama_client import clear_thinking_mode_cache
        clear_thinking_mode_cache()
        self.addCleanup(clear_thinking_mode_cache)
        thinking = patch('core.ollama_client.OllamaClient.thinking_mode', return_value='unknown')
        thinking.start()
        self.addCleanup(thinking.stop)
        instructions = patch('core.ai_assist_instructions.load_instructions', return_value={})
        instructions.start()
        self.addCleanup(instructions.stop)

    def _drain(self, worker):
        self.assertTrue(worker.wait(5000))
        for _ in range(3):
            self.app.processEvents()

    def test_enhance_and_nl_slots_resolve_off_the_gui_thread_and_clean_up(self):
        from ui.vue_bridge import VueBridge
        bridge = VueBridge()
        get_threads, sent_models = [], []
        results, nl_results = [], []
        bridge.ollamaResult.connect(results.append)
        bridge.genNlResult.connect(nl_results.append)

        def get(url, **_):
            get_threads.append(threading.current_thread() is threading.main_thread())
            return _Resp({'models': [{'name': 'gemma3:12b'}]})

        def post(url, json=None, **_):
            sent_models.append(json['model'])
            return _Resp({'message': {'content': 'A blue bird rests. It is calm.'}})

        with patch('core.ollama_client.requests.get', side_effect=get), \
                patch('core.ollama_client.requests.post', side_effect=post):
            bridge.ollamaEnhance('bird', 'expand', json.dumps({'url': 'http://fake.local', 'model': 'gemma3:4b'}))
            worker = bridge._ollama_worker
            self._drain(worker)
            bridge.convertPromptToNl('bird', json.dumps({'url': '', 'model': ''}))
            nl_worker = bridge._gennl_worker
            self._drain(nl_worker)
        self.assertEqual(get_threads, [False, False], '설치 목록 조회는 워커 스레드에서만')
        self.assertEqual(sent_models, ['gemma3:12b', 'gemma3:12b'])
        self.assertEqual(len(results), 1)
        self.assertEqual(len(nl_results), 1)
        self.assertIsNone(bridge._ollama_worker, '끝난 워커 참조는 비워진다')
        self.assertIsNone(bridge._gennl_worker)

    def test_replacing_a_running_request_detaches_it_without_blocking(self):
        from ui.vue_bridge import VueBridge
        bridge = VueBridge()
        release = threading.Event()
        results = []
        bridge.ollamaResult.connect(results.append)

        def slow_post(url, json=None, **_):
            if json['messages'][-1]['content'] == 'first':
                release.wait(5)
                return _Resp({'message': {'content': 'old_tag'}})
            return _Resp({'message': {'content': 'new_tag'}})

        with patch('core.ollama_client.requests.get', return_value=_Resp({'models': []})), \
                patch('core.ollama_client.requests.post', side_effect=slow_post):
            bridge.ollamaEnhance('first', 'expand', json.dumps({'url': 'http://fake.local', 'model': 'm'}))
            first = bridge._ollama_worker
            import time
            started = time.monotonic()
            bridge.ollamaEnhance('second', 'expand', json.dumps({'url': 'http://fake.local', 'model': 'm'}))
            self.assertLess(time.monotonic() - started, 0.5, '이전 워커를 wait 하며 GUI 를 막지 않는다')
            second = bridge._ollama_worker
            self.assertIsNot(first, second)
            self._drain(second)
            release.set()
            self._drain(first)
        self.assertEqual([json.loads(r)['tags'] for r in results], ['new_tag'], '교체된 요청의 결과는 버린다')

    def test_detach_tolerates_workers_without_connections(self):
        from workers.ollama_worker import OllamaWorker, detach_result_signals
        detach_result_signals(None)
        detach_result_signals(OllamaWorker('http://fake.local', 'm', 't', 'expand', instructions={}))


class AutomationNlTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QCoreApplication.instance() or QCoreApplication(sys.argv)

    def test_automation_restart_drops_the_stale_conversion_result(self):
        from ui.generator_actions import ActionsMixin

        class Host(QObject, ActionsMixin):
            _auto_nl_url = 'http://fake.local'
            _auto_nl_model = ''

            def __init__(self):
                super().__init__()
                self.done = []

            def _on_auto_nl_done(self, raw):
                self.done.append(json.loads(raw)['tags'])

            def _on_auto_nl_error(self, error):
                self.done.append('error:' + error)

        host = Host()
        release = threading.Event()

        def post(url, json=None, **_):
            if 'old' in json['messages'][-1]['content']:
                release.wait(5)
            return _Resp({'message': {'content': 'A bird. It flies.'}})

        with patch('core.ollama_client.OllamaClient.thinking_mode', return_value='unknown'), \
                patch('core.ai_assist_instructions.load_instructions', return_value={}), \
                patch('core.ollama_client.requests.get', return_value=_Resp({'models': [{'name': 'm:1'}]})), \
                patch('core.ollama_client.requests.post', side_effect=post):
            self.assertTrue(host._start_auto_nl_then_generate('old tags'))
            old = host._auto_nl_worker
            self.assertTrue(host._start_auto_nl_then_generate('new tags'))
            new = host._auto_nl_worker
            self.assertTrue(new.wait(5000))
            release.set()
            self.assertTrue(old.wait(5000))
            for _ in range(3):
                self.app.processEvents()
        self.assertEqual(len(host.done), 1, '멈췄다 다시 켠 뒤 옛 변환이 생성을 한 번 더 부르지 않는다')


class _SyncThread:
    """threading.Thread 대역 — start() 에서 바로 실행해 결과를 결정적으로 확인한다."""

    def __init__(self, target=None, args=(), kwargs=None, daemon=None, **_):
        self._target, self._args, self._kwargs = target, args, dict(kwargs or {})

    def start(self):
        self._target(*self._args, **self._kwargs)


class ConfiguredModelConsistencyTests(unittest.TestCase):
    """ui_prefs.ollamaModel 을 Ollama 에 보내는 모든 경로가 같은 규칙(resolve_model)으로 설치 모델을 고른다.

    Settings 추천 카드는 pull 안내를 위해 설치 안 된 이름(gemma3:4b)을 그대로 저장한다. 태그 강화·NL
    워커는 같은 계열의 설치 모델(gemma3:12b)을 올리므로, 생성 전 언로드와 Comic Director 도 그 모델을
    가리켜야 한다 — 예전엔 언로드가 keep_alive=0 을 없는 이름으로 보내(404) VRAM 이 그대로 남았고
    Comic 은 404 로 실패했다.
    """

    def setUp(self):
        from core.ollama_client import clear_thinking_mode_cache
        clear_thinking_mode_cache()
        self.addCleanup(clear_thinking_mode_cache)
        thinking = patch('core.ollama_client.OllamaClient.thinking_mode', return_value='unknown')
        thinking.start()
        self.addCleanup(thinking.stop)
        self.installed = ['llava:7b', 'gemma3:12b']   # None = 서버 꺼짐
        self.posts = []

    def _get(self, url, **_):
        import requests
        if self.installed is None:
            raise requests.ConnectionError('refused')
        return _Resp({'models': [{'name': name} for name in self.installed]})

    def _post(self, url, json=None, **_):
        self.posts.append((url, dict(json or {})))
        return _Resp({'message': {'content': '{"ok": 1}'}})

    def _http(self):
        from contextlib import ExitStack
        stack = ExitStack()
        stack.enter_context(patch('core.ollama_client.requests.get', side_effect=self._get))
        stack.enter_context(patch('core.ollama_client.requests.post', side_effect=self._post))
        return stack

    def test_unload_configured_model_targets_the_model_actually_loaded(self):
        from core.ollama_client import unload_configured_model
        with self._http():
            self.assertTrue(unload_configured_model('http://fake.local', 'gemma3:4b'))
        self.assertEqual(self.posts, [('http://fake.local/api/generate', {'model': 'gemma3:12b', 'keep_alive': 0})])

    def test_unload_configured_model_without_server_or_model_has_nothing_to_unload(self):
        from core.ollama_client import unload_configured_model
        self.installed = None
        with self._http():
            self.assertTrue(unload_configured_model('http://fake.local', 'gemma3:4b'), '서버가 꺼져 있으면 VRAM 도 없다')
            self.assertTrue(unload_configured_model('http://fake.local', '  '))
        self.assertEqual(self.posts, [])

    def test_generation_unload_uses_the_resolved_model(self):
        from ui.generator_generation import GenerationMixin
        prefs = {'ollamaUnloadOnGen': True, 'ollamaModel': 'gemma3:4b', 'ollamaUrl': 'http://fake.local'}
        # ui_prefs 는 core.ui_prefs.read_ui_prefs 한 곳에서 읽는다 — 실제 config 파일을 열지 않게 그 경계를 바꾼다
        with self._http(), \
                patch('core.ui_prefs.read_ui_prefs', return_value=prefs), \
                patch('threading.Thread', _SyncThread):
            GenerationMixin._maybe_unload_ollama(object())
        self.assertEqual(self.posts, [('http://fake.local/api/generate', {'model': 'gemma3:12b', 'keep_alive': 0})])

    def test_comic_director_requests_the_resolved_model(self):
        from ui.creator_actions import CreatorActionsMixin

        class Host(CreatorActionsMixin):
            def _creator_ollama_config(self):
                return ('http://fake.local', 'gemma3:4b')

        with self._http():
            self.assertEqual(Host()._comic_ollama_complete('sys', 'user'), '{"ok": 1}')
            self.assertTrue(Host()._creator_unload_ollama())
        self.assertEqual([body['model'] for url, body in self.posts if url.endswith('/api/chat')], ['gemma3:12b'])
        self.assertEqual([body['model'] for url, body in self.posts if url.endswith('/api/generate')], ['gemma3:12b'])


class ModelListRefreshTests(unittest.TestCase):
    """모델 목록 재조회(Settings 연결 테스트·자동 로드, 부팅, 대화·캡션 탭) = 재연결·pull 뒤 —
    그 서버의 think 능력·거부 기억을 비워 /api/show 를 새로 읽는다(다른 서버 몫은 그대로)."""

    @classmethod
    def setUpClass(cls):
        cls.app = QCoreApplication.instance() or QCoreApplication(sys.argv)

    def setUp(self):
        from core.ollama_client import clear_thinking_mode_cache
        clear_thinking_mode_cache()
        self.addCleanup(clear_thinking_mode_cache)

    def test_reloading_models_refreshes_think_state_for_that_server_only(self):
        from core.ollama_client import THINK_REJECTED, OllamaClient
        shows = []

        def post(url, json=None, **_):
            shows.append(url)
            return _Resp({'capabilities': ['completion', 'thinking'], 'details': {'family': 'gemma3'},
                          'model_info': {'general.architecture': 'gemma3'}})

        with patch('core.ollama_client.requests.post', side_effect=post), \
                patch('core.ollama_client.requests.get', return_value=_Resp({'models': [{'name': 'reasoner'}]})):
            here = OllamaClient('http://fake.local', 'reasoner')
            there = OllamaClient('http://other.local', 'reasoner')
            self.assertEqual(here.thinking_mode(), 'boolean')
            self.assertEqual(there.thinking_mode(), 'boolean')
            here._mark_think_rejected()
            there._mark_think_rejected()
            self.assertEqual(json.loads(VueBridge()._load_ollama_models_json('http://fake.local/')), ['reasoner'])
            self.assertEqual(here.thinking_mode(), 'boolean', '재연결 뒤엔 능력을 새로 읽는다')
            self.assertEqual(there.thinking_mode(), THINK_REJECTED, '다른 서버의 기억은 그대로')
        self.assertEqual(len(shows), 3)


if __name__ == '__main__':
    unittest.main()
