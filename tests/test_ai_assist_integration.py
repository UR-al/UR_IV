"""AI assist settings -> existing UI entrypoints -> actual HTTP request contracts.

Only temporary settings and mocked HTTP are used. No Ollama server/model starts.
"""
from __future__ import annotations

import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from PyQt6.QtCore import QCoreApplication

from ui.vue_bridge import VueBridge


class _Response:
    status_code = 200

    def __init__(self, data):
        self.data = data

    def raise_for_status(self):
        return None

    def json(self):
        return self.data


class AiAssistIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QCoreApplication.instance() or QCoreApplication([])

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.prefs = self.root / 'ui_prefs.json'
        self.prefs.write_text(json.dumps({'unrelatedSetting': 'keep-me'}), encoding='utf-8')
        # The module's default-path seam prevents any access to personal prefs.
        self.path_patch = patch('core.ai_assist_instructions.config_file', return_value=self.prefs)
        self.path_patch.start()
        self.addCleanup(self.path_patch.stop)
        # /api/show 능력 조회(think 제어)는 (서버, 모델)별로 캐시된다 — 테스트 순서에 따라
        # 조회 요청이 끼거나 빠지지 않게 매번 비운다. 요청은 URL 로 골라 본다.
        from core.ollama_client import clear_thinking_mode_cache
        clear_thinking_mode_cache()
        self.addCleanup(clear_thinking_mode_cache)
        self.bridge = VueBridge()
        self.instructions = {
            'common': 'AI_ASSIST_COMMON_SENTINEL',
            'features': {key: f'FEATURE_{key.upper()}_SENTINEL' for key in (
                'expand', 'suggest', 'nl2tags', 'nl_caption', 'nl_scene',
                'translate', 'creative', 'negative', 'auto_nl')},
        }
        self.posts = []

    def _save(self, instructions=None):
        value = self.instructions if instructions is None else instructions
        saved = json.loads(self.bridge.saveAiAssistInstructions(json.dumps(value)))
        self.assertTrue(saved.get('ok'), saved)
        return saved['instructions']

    def _post(self, url, **kwargs):
        self.posts.append((url, copy.deepcopy(kwargs['json'])))
        return _Response({'message': {'content': 'A small cat sits beside a window.'}})

    def _wait_worker(self, worker):
        self.assertTrue(worker.wait(5000), 'mocked HTTP worker should finish promptly')
        self.app.processEvents()

    def _assert_instructions(self, payload, *features):
        content = '\n'.join(message['content'] for message in payload['messages'])
        self.assertEqual(content.count(self.instructions['common']), 1)
        for feature, instruction in self.instructions['features'].items():
            self.assertEqual(content.count(instruction), 1 if feature in features else 0, feature)

    def test_saved_instructions_reach_every_manual_ai_menu_without_other_features(self):
        self._save()
        loaded = json.loads(self.bridge.getAiAssistInstructions())
        self.assertTrue(loaded['ok'])
        self.assertEqual(loaded['instructions'], self.instructions)
        errors = []
        self.bridge.ollamaResult.connect(lambda raw: errors.append(json.loads(raw).get('error')))
        with patch('core.ollama_client.requests.get', return_value=_Response({'models': [{'name': 'fixture'}]})), \
             patch('core.ollama_client.requests.post', side_effect=self._post):
            for feature in self.instructions['features']:
                if feature == 'auto_nl':
                    continue
                with self.subTest(feature=feature):
                    self.bridge.ollamaEnhance('cat, window', feature,
                                              json.dumps({'url': 'http://test-only', 'model': 'fixture'}))
                    self._wait_worker(self.bridge._ollama_worker)
                    self._assert_instructions(self.posts[-1][1], feature)
        self.assertEqual(errors, [None] * 8)
        self.assertEqual(json.loads(self.prefs.read_text(encoding='utf-8'))['unrelatedSetting'], 'keep-me')

    def test_saved_assist_instructions_never_enter_chat_or_image_caption_http(self):
        from PIL import Image
        from core.ollama_client import OllamaClient
        from workers.chat_worker import ChatWorker
        self._save()
        image = self.root / 'fixture.png'
        Image.new('RGB', (2, 2), 'navy').save(image)

        class StreamResponse(_Response):
            def __enter__(self):
                return self

            def __exit__(self, *_args):
                return False

            def iter_lines(self, **_kwargs):
                yield json.dumps({'message': {'content': 'Chat reply.'}, 'done': True})

        def post(url, **kwargs):
            self.posts.append((url, copy.deepcopy(kwargs['json'])))
            return StreamResponse({}) if kwargs['json'].get('stream') else _Response({'response': 'A cat sits.'})

        done = []
        messages = [{'role': 'system', 'content': 'CHAT_PRIVATE_SYSTEM'}, {'role': 'user', 'content': 'hello'}]
        worker = ChatWorker('chat-isolation', 'http://test-only', 'fixture', messages)
        worker.done.connect(lambda raw: done.append(json.loads(raw)))
        with patch('core.ollama_client.requests.post', side_effect=post):
            worker.run()
            self.assertEqual(OllamaClient('http://test-only', 'fixture').caption_image(str(image)), 'A cat sits.')
            self.assertEqual(OllamaClient('http://test-only', 'fixture').caption_image(
                str(image), system_prompt='BATCH_PRIVATE_SYSTEM'), 'A cat sits.')
        self.assertTrue(done[0]['ok'])
        self.assertEqual(self.posts[0][1]['messages'], messages)
        captions = [payload for url, payload in self.posts if url.endswith('/api/generate')]
        self.assertEqual(len(captions), 2)
        self.assertEqual(captions[1]['system'], 'BATCH_PRIVATE_SYSTEM')
        for _url, payload in self.posts:
            self.assertNotIn(self.instructions['common'], json.dumps(payload))
            for instruction in self.instructions['features'].values():
                self.assertNotIn(instruction, json.dumps(payload))

    def test_both_automatic_caption_entrypoints_apply_common_caption_and_auto_once(self):
        from PyQt6.QtCore import QObject
        from ui.generator_actions import ActionsMixin

        class AutomationHost(QObject, ActionsMixin):
            _auto_nl_url = 'http://test-only'
            _auto_nl_model = 'fixture'

            def __init__(self):
                super().__init__()
                self.results = []

            def _on_auto_nl_done(self, raw):
                self.results.append(json.loads(raw))

            def _on_auto_nl_error(self, error):
                self.results.append({'error': error})

        self._save()
        results = []
        self.bridge.genNlResult.connect(lambda raw: results.append(json.loads(raw)))
        host = AutomationHost()
        with patch('core.ollama_client.requests.get', return_value=_Response({'models': [{'name': 'fixture'}]})), \
             patch('core.ollama_client.requests.post', side_effect=self._post):
            self.bridge.convertPromptToNl('cat, window', json.dumps({'url': 'http://test-only', 'model': 'fixture'}))
            self._wait_worker(self.bridge._gennl_worker)
            self._assert_instructions(self.posts[-1][1], 'nl_caption', 'auto_nl')
            self.assertTrue(host._start_auto_nl_then_generate('cat, window'))
            self._wait_worker(host._auto_nl_worker)
            self._assert_instructions(self.posts[-1][1], 'nl_caption', 'auto_nl')
        self.assertEqual(len(results), 1)
        self.assertEqual(len(host.results), 1)
        for result in results + host.results:
            self.assertNotIn('error', result)
            self.assertEqual(result['mode'], 'nl_caption')

    def test_saving_changes_affects_the_next_worker_not_an_already_created_request(self):
        from workers.ollama_worker import OllamaWorker
        self._save()
        first = OllamaWorker('http://test-only', 'fixture', 'cat', 'expand')
        changed = copy.deepcopy(self.instructions)
        changed['common'] = 'NEW_COMMON_SENTINEL'
        changed['features']['expand'] = 'NEW_EXPAND_SENTINEL'
        self._save(changed)
        with patch('core.ollama_client.requests.post', side_effect=self._post):
            first.run()
            self._assert_instructions(self.posts[-1][1], 'expand')
            second = OllamaWorker('http://test-only', 'fixture', 'cat', 'expand')
            second.run()
        content = json.dumps(self.posts[-1][1])
        self.assertEqual(content.count('NEW_COMMON_SENTINEL'), 1)
        self.assertEqual(content.count('NEW_EXPAND_SENTINEL'), 1)
        self.assertNotIn(self.instructions['common'], content)
        self.assertNotIn(self.instructions['features']['expand'], content)

    def test_clearing_saved_instructions_restores_the_original_system_prompt(self):
        from core.ollama_client import SYSTEM_PROMPTS
        from workers.ollama_worker import OllamaWorker
        self._save()
        self._save({'common': '', 'features': {}})
        with patch('core.ollama_client.requests.post', side_effect=self._post):
            OllamaWorker('http://test-only', 'fixture', 'cat', 'expand').run()
        self.assertEqual(self.posts[-1][1]['messages'][0], {'role': 'system', 'content': SYSTEM_PROMPTS['expand']})

    def test_invalid_save_payload_preserves_previous_instructions(self):
        self._save()
        before = self.prefs.read_bytes()
        for invalid in ('not JSON', '{"common": null}', '{"features": {"chat": "not permitted"}}'):
            with self.subTest(invalid=invalid):
                result = json.loads(self.bridge.saveAiAssistInstructions(invalid))
                self.assertFalse(result['ok'])
                self.assertEqual(self.prefs.read_bytes(), before)

    def test_bridge_returns_save_failure_even_when_error_reporting_also_throws(self):
        self._save()
        before = self.prefs.read_bytes()
        changed = copy.deepcopy(self.instructions)
        changed['common'] = 'unsaved edit'
        with patch('core.config_migration.os.replace', side_effect=PermissionError('test denial')), \
             patch('core.error_handler.handle_error', side_effect=RuntimeError('test logging failure')):
            result = json.loads(self.bridge.saveAiAssistInstructions(json.dumps(changed)))
        self.assertFalse(result['ok'])
        self.assertIn('저장하지 못했습니다', result['error'])
        self.assertEqual(self.prefs.read_bytes(), before)

    def test_bridge_returns_load_failure_even_when_error_reporting_also_throws(self):
        with patch('core.ai_assist_instructions.config_file', side_effect=RuntimeError('test path failure')), \
             patch('core.error_handler.handle_error', side_effect=RuntimeError('test logging failure')):
            result = json.loads(self.bridge.getAiAssistInstructions())
        self.assertFalse(result['ok'])
        self.assertIn('불러오지 못했습니다', result['error'])


if __name__ == '__main__':
    unittest.main()
