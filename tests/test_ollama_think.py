"""추론형 모델의 think 제어 — 태그 강화·자연어·캡션·Comic 요청 (가짜 HTTP 만, Ollama 없음).

- /api/show 의 thinkingMode 로 think 를 정한다: boolean→False, levels→'low', 그 밖→필드 생략
- think 를 보낸 요청이 400 이거나 200 인데 빈 답이면 think 없이 **호출당 한 번만** 다시 보낸다
- think 탓인 400 이면 그 호출의 남은 폴백 단계는 think 없이, 빈 답(재시도도 빔)이면 think 유지
- 영구 기억(THINK_REJECTED)은 '<모델> does not support thinking' 400 이거나 빈 답이 연속될 때만
- 태그 모드에서는 사고(thinking) 원문을 답으로 쓰지 않는다(콤마로 쪼개져 태그에 섞임)
"""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from PIL import Image

from core.ollama_client import (
    THINK_EMPTY_STRIKE_LIMIT,
    THINK_REJECTED,
    OllamaClient,
    clear_thinking_mode_cache,
    is_think_unsupported_error,
    think_value_for,
)


class _Resp:
    """requests.Response 대역 — status_code 를 주지 않으면 속성 자체가 없다(200 으로 봐야 함)."""

    def __init__(self, data, status=None):
        self._data = data
        if status is not None:
            self.status_code = status

    def raise_for_status(self):
        status = getattr(self, 'status_code', 200)
        if status >= 400:
            import requests
            raise requests.HTTPError(f"{status} error")

    def json(self):
        return self._data


def _show(capabilities, architecture='gemma3'):
    return _Resp({'capabilities': capabilities, 'details': {'family': architecture},
                  'model_info': {'general.architecture': architecture}})


class _FakeOllama:
    """URL 별로 응답을 정해 두고 보낸 요청을 기록한다."""

    def __init__(self, show, answers, installed=()):
        self.show = show
        self.answers = list(answers)
        self.installed = list(installed)
        self.calls = []

    def post(self, url, json=None, timeout=None, **_kwargs):
        self.calls.append((url, dict(json or {})))
        if url.endswith('/api/show'):
            if isinstance(self.show, Exception):
                raise self.show
            return self.show
        return self.answers.pop(0)

    def get(self, url, timeout=None, **_kwargs):
        self.calls.append((url, {}))
        return _Resp({'models': [{'name': name} for name in self.installed]})

    def requests_to(self, suffix):
        return [body for url, body in self.calls if url.endswith(suffix)]


class ThinkValueTests(unittest.TestCase):
    def test_mode_to_request_value(self):
        self.assertIs(think_value_for('boolean'), False)
        self.assertEqual(think_value_for('levels'), 'low')
        for mode in ('none', 'unknown', THINK_REJECTED, '', None):
            self.assertIsNone(think_value_for(mode))

    def test_only_the_explicit_capability_error_counts_as_unsupported(self):
        for detail in ('"gemma3:4b" does not support thinking',
                       'registry.ollama.ai/library/gemma3:4b does not support thinking.',
                       '  "x" Does Not Support Thinking  '):
            self.assertTrue(is_think_unsupported_error(detail), detail)
        for detail in ('think value invalid', 'invalid think value "max"', '', None,
                       'model "x" does not support thinking with format json'):
            self.assertFalse(is_think_unsupported_error(detail), detail)


class ThinkingModeCacheTests(unittest.TestCase):
    def setUp(self):
        clear_thinking_mode_cache()
        self.addCleanup(clear_thinking_mode_cache)

    def test_mode_is_read_once_per_server_and_model(self):
        fake = _FakeOllama(_show(['completion', 'thinking']), [])
        with patch('core.ollama_client.requests.post', side_effect=fake.post):
            client = OllamaClient('http://fake.local', 'reasoner')
            self.assertEqual(client.thinking_mode(), 'boolean')
            self.assertEqual(OllamaClient('http://fake.local/', 'reasoner').thinking_mode(), 'boolean')
            self.assertEqual(len(fake.requests_to('/api/show')), 1)
            fake.show = _show(['completion'])
            self.assertEqual(OllamaClient('http://fake.local', 'plain').thinking_mode(), 'none')
            self.assertEqual(len(fake.requests_to('/api/show')), 2, '모델이 다르면 따로 읽는다')

    def test_lookup_failures_are_not_cached(self):
        import requests
        fake = _FakeOllama(requests.ConnectionError('refused'), [])
        with patch('core.ollama_client.requests.post', side_effect=fake.post):
            client = OllamaClient('http://fake.local', 'late-server')
            self.assertEqual(client.thinking_mode(), 'unknown')
            fake.show = _show(['completion', 'thinking'], 'gptoss')
            self.assertEqual(client.thinking_mode(), 'levels', '서버가 뜬 뒤엔 다시 읽는다')

    def test_clearing_one_server_forgets_only_its_capabilities_and_rejections(self):
        fake = _FakeOllama(_show(['completion', 'thinking']), [])
        with patch('core.ollama_client.requests.post', side_effect=fake.post):
            here = OllamaClient('http://a.local', 'reasoner')
            there = OllamaClient('http://b.local', 'reasoner')
            self.assertEqual(here.thinking_mode(), 'boolean')
            self.assertEqual(there.thinking_mode(), 'boolean')
            here._mark_think_rejected()
            there._mark_think_rejected()
            clear_thinking_mode_cache('http://a.local/')
            self.assertEqual(here.thinking_mode(), 'boolean', '재연결·pull 뒤엔 능력을 새로 읽는다')
            self.assertEqual(there.thinking_mode(), THINK_REJECTED, '다른 서버의 기억은 그대로')
            self.assertEqual(len(fake.requests_to('/api/show')), 3)

    def test_clearing_also_resets_the_empty_answer_streak(self):
        fake = _FakeOllama(_show(['completion', 'thinking']), [])
        with patch('core.ollama_client.requests.post', side_effect=fake.post):
            client = OllamaClient('http://fake.local', 'reasoner')
            for _ in range(THINK_EMPTY_STRIKE_LIMIT - 1):
                client._note_think_empty_strike()
            clear_thinking_mode_cache('http://fake.local')
            client._note_think_empty_strike()
            self.assertEqual(client.thinking_mode(), 'boolean' if THINK_EMPTY_STRIKE_LIMIT > 1 else THINK_REJECTED)


class EnhanceThinkTests(unittest.TestCase):
    def setUp(self):
        clear_thinking_mode_cache()
        self.addCleanup(clear_thinking_mode_cache)

    def _enhance(self, fake, mode='expand', model='reasoner'):
        with patch('core.ollama_client.requests.post', side_effect=fake.post):
            return OllamaClient('http://fake.local', model).enhance('blue bird', mode)

    def test_boolean_thinking_model_gets_think_false_on_tag_requests(self):
        fake = _FakeOllama(_show(['completion', 'thinking']),
                           [_Resp({'message': {'content': 'blue_bird, sky'}})])
        self.assertEqual(self._enhance(fake), 'blue_bird, sky')
        chat = fake.requests_to('/api/chat')
        self.assertEqual(len(chat), 1)
        self.assertIs(chat[0]['think'], False)

    def test_gpt_oss_levels_uses_low_and_plain_models_omit_think(self):
        fake = _FakeOllama(_show(['completion', 'thinking'], 'gptoss'),
                           [_Resp({'message': {'content': 'blue_bird'}})])
        self._enhance(fake, model='gpt-oss:20b')
        self.assertEqual(fake.requests_to('/api/chat')[0]['think'], 'low')
        fake = _FakeOllama(_show(['completion']), [_Resp({'message': {'content': 'blue_bird'}})])
        self._enhance(fake, model='plain')
        self.assertNotIn('think', fake.requests_to('/api/chat')[0])
        fake = _FakeOllama(_Resp({'details': {}}), [_Resp({'message': {'content': 'blue_bird'}})])
        self._enhance(fake, model='old-server')
        self.assertNotIn('think', fake.requests_to('/api/chat')[0], '능력을 모르면 보내지 않는다')

    def _mode(self, fake, model='reasoner'):
        with patch('core.ollama_client.requests.post', side_effect=fake.post):
            return OllamaClient('http://fake.local', model).thinking_mode()

    def test_explicit_unsupported_400_retries_without_think_and_is_remembered(self):
        fake = _FakeOllama(_show(['completion', 'thinking']), [
            _Resp({'error': '"reasoner" does not support thinking'}, status=400),
            _Resp({'message': {'content': 'blue_bird'}}),
            _Resp({'message': {'content': 'red_bird'}}),
        ])
        self.assertEqual(self._enhance(fake), 'blue_bird')
        chat = fake.requests_to('/api/chat')
        self.assertIs(chat[0]['think'], False)
        self.assertNotIn('think', chat[1])
        self.assertEqual(self._mode(fake), THINK_REJECTED, '능력이 없다는 명시적 거부는 모델 성질이다')
        self.assertEqual(self._enhance(fake), 'red_bird')
        self.assertNotIn('think', fake.requests_to('/api/chat')[2], '거부한 모델엔 다시 보내지 않는다')
        self.assertEqual(len(fake.requests_to('/api/show')), 1)

    def test_generic_400_drops_think_for_the_rest_of_this_call_only(self):
        fake = _FakeOllama(_show(['completion', 'thinking']), [
            _Resp({'error': 'think value invalid'}, status=400),   # 1단계 think → 400
            _Resp({'message': {'content': ''}}),                    # 1단계 think 없이 → 200(빈 답)
            _Resp({'message': {'content': 'blue_bird'}}),           # 2단계
            _Resp({'message': {'content': 'red_bird'}}),            # 다음 호출
        ])
        self.assertEqual(self._enhance(fake), 'blue_bird')
        chat = fake.requests_to('/api/chat')
        self.assertEqual(len(chat), 3)
        self.assertIs(chat[0]['think'], False)
        self.assertNotIn('think', chat[1])
        self.assertNotIn('think', chat[2], 'think 탓인 400 이면 남은 폴백 단계는 think 없이')
        self.assertEqual(self._mode(fake), 'boolean', '명시적 능력 거부가 아닌 400 은 기억하지 않는다')
        self.assertEqual(self._enhance(fake), 'red_bird')
        self.assertIs(fake.requests_to('/api/chat')[3]['think'], False, '다음 호출은 다시 사고를 끈다')

    def test_400_that_persists_without_think_is_not_blamed_on_think(self):
        fake = _FakeOllama(_show(['completion', 'thinking']), [
            _Resp({'error': 'invalid options'}, status=400),   # 1단계 think
            _Resp({'error': 'invalid options'}, status=400),   # 1단계 think 없이 — 역시 400
            _Resp({'error': 'invalid options'}, status=400),   # 2단계 think — 재시도는 이미 씀
            _Resp({'response': 'blue_bird'}),                   # 3단계 generate
        ])
        self.assertEqual(self._enhance(fake), 'blue_bird')
        chat = fake.requests_to('/api/chat')
        self.assertEqual(len(chat), 3, 'think 없는 재시도는 호출당 한 번')
        self.assertNotIn('think', chat[1])
        self.assertIs(chat[2]['think'], False, 'think 를 빼도 400 이면 think 탓이 아니다 — 사고는 계속 끈다')
        self.assertIs(fake.requests_to('/api/generate')[0]['think'], False)
        self.assertEqual(self._mode(fake), 'boolean')

    def test_empty_answer_keeps_think_for_later_stages_without_more_retries(self):
        fake = _FakeOllama(_show(['completion', 'thinking']), [
            _Resp({'message': {'content': ''}}),   # 1단계 think → 빈 답
            _Resp({'message': {'content': ''}}),   # 1단계 think 없이 → 역시 빈 답(think 탓 아님)
            _Resp({'message': {'content': ''}}),   # 2단계 think → 빈 답, 재시도는 이미 씀
            _Resp({'response': 'blue_bird'}),      # 3단계 generate think
        ])
        self.assertEqual(self._enhance(fake), 'blue_bird')
        chat = fake.requests_to('/api/chat')
        self.assertEqual(len(chat), 3, 'think 없는 재시도는 호출당 한 번 — 요청 수가 두 배가 되지 않는다')
        self.assertIs(chat[0]['think'], False)
        self.assertNotIn('think', chat[1])
        self.assertIs(chat[2]['think'], False, '재시도도 비었다면 think 탓이 아니다 — 다음 단계도 사고를 끈 채로')
        generate = fake.requests_to('/api/generate')
        self.assertEqual(len(generate), 1)
        self.assertIs(generate[0]['think'], False)
        self.assertEqual(self._mode(fake), 'boolean')

    def test_one_transient_empty_answer_is_not_remembered(self):
        fake = _FakeOllama(_show(['completion', 'thinking']), [
            _Resp({'message': {'content': ''}}),            # 1회차 think → 빈 답
            _Resp({'message': {'content': 'blue_bird'}}),   # 1회차 think 없이 → 답
            _Resp({'message': {'content': 'red_bird'}}),    # 2회차 think → 답 (연속이 끊김)
            _Resp({'message': {'content': ''}}),            # 3회차 think → 빈 답
            _Resp({'message': {'content': 'green_bird'}}),  # 3회차 think 없이 → 답
        ])
        self.assertEqual(self._enhance(fake), 'blue_bird')
        self.assertEqual(self._mode(fake), 'boolean', '한 번의 빈 답으로 think 제어를 영영 끄지 않는다')
        self.assertEqual(self._enhance(fake), 'red_bird')
        self.assertIs(fake.requests_to('/api/chat')[2]['think'], False)
        self.assertEqual(self._enhance(fake), 'green_bird')
        self.assertEqual(self._mode(fake), 'boolean', '사이에 think 로 답을 받았으면 연속이 아니다')

    def test_repeated_empty_answers_with_think_are_remembered(self):
        self.assertGreaterEqual(THINK_EMPTY_STRIKE_LIMIT, 2, '한 번은 샘플링 우연일 수 있다')
        answers = []
        for _ in range(THINK_EMPTY_STRIKE_LIMIT):
            answers += [_Resp({'message': {'content': ''}}), _Resp({'message': {'content': 'blue_bird'}})]
        answers.append(_Resp({'message': {'content': 'red_bird'}}))
        fake = _FakeOllama(_show(['completion', 'thinking']), answers)
        for attempt in range(THINK_EMPTY_STRIKE_LIMIT):
            self.assertEqual(self._enhance(fake), 'blue_bird')
            expected = THINK_REJECTED if attempt == THINK_EMPTY_STRIKE_LIMIT - 1 else 'boolean'
            self.assertEqual(self._mode(fake), expected)
        self.assertEqual(self._enhance(fake), 'red_bird')
        self.assertNotIn('think', fake.requests_to('/api/chat')[-1], '연속으로 빈 답이면 그 모델엔 보내지 않는다')

    def test_tag_mode_never_turns_thinking_into_tags(self):
        thought = _Resp({'message': {'content': '', 'thinking': 'Okay, the user wants tags, maybe sky'}})
        fake = _FakeOllama(_show(['completion']), [thought, thought, _Resp({'response': ''})])
        with self.assertRaisesRegex(RuntimeError, '빈 응답'):
            self._enhance(fake)

    def test_nl_mode_still_recovers_a_caption_from_thinking(self):
        fake = _FakeOllama(_show(['completion']), [
            _Resp({'message': {'content': '', 'thinking': 'A blue bird sits on a branch. It looks calm.'}}),
        ])
        self.assertEqual(self._enhance(fake, mode='nl_caption'), 'A blue bird sits on a branch. It looks calm.')

    def test_response_without_status_code_is_treated_as_success(self):
        fake = _FakeOllama(_show(['completion', 'thinking']), [_Resp({'message': {'content': 'blue_bird'}})])
        self.assertFalse(hasattr(fake.answers[0], 'status_code'))
        self.assertEqual(self._enhance(fake), 'blue_bird')


class CaptionAndComicThinkTests(unittest.TestCase):
    def setUp(self):
        clear_thinking_mode_cache()
        self.addCleanup(clear_thinking_mode_cache)
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.image = Path(temp.name) / 'image.png'
        Image.new('RGB', (4, 4), 'navy').save(self.image)

    def test_caption_sends_think_false_and_retries_empty_answer_without_it(self):
        fake = _FakeOllama(_show(['completion', 'vision', 'thinking'], 'qwen3vl'), [
            _Resp({'response': ''}),
            _Resp({'response': 'A navy square fills the frame.'}),
        ])
        with patch('core.ollama_client.requests.post', side_effect=fake.post):
            text = OllamaClient('http://fake.local', 'qwen3-vl').caption_image(str(self.image))
        self.assertEqual(text, 'A navy square fills the frame.')
        generate = fake.requests_to('/api/generate')
        self.assertIs(generate[0]['think'], False)
        self.assertNotIn('think', generate[1])

    def test_comic_complete_chat_uses_think_and_returns_json_content(self):
        fake = _FakeOllama(_show(['completion', 'thinking']),
                           [_Resp({'message': {'content': '{"pages": []}', 'thinking': 'plan'}})])
        with patch('core.ollama_client.requests.post', side_effect=fake.post):
            text = OllamaClient('http://fake.local', 'reasoner').complete_chat(
                [{'role': 'user', 'content': 'x'}], options={'num_predict': 3600}, response_format='json')
        self.assertEqual(text, '{"pages": []}')
        body = fake.requests_to('/api/chat')[0]
        self.assertIs(body['think'], False)
        self.assertEqual(body['format'], 'json')
        self.assertEqual(body['options'], {'num_predict': 3600})
        self.assertFalse(body['stream'])

    def test_creator_comic_path_goes_through_the_think_aware_client(self):
        from ui.creator_actions import CreatorActionsMixin

        class Host(CreatorActionsMixin):
            def _creator_ollama_config(self):
                return ('http://fake.local', 'reasoner')

        fake = _FakeOllama(_show(['completion', 'thinking']), [_Resp({'message': {'content': '{"ok": 1}'}})],
                           installed=['reasoner'])
        # Comic 도 설치 모델 대조(/api/tags)를 거친다 — 가짜 GET 으로 실제 네트워크를 타지 않게
        with patch('core.ollama_client.requests.post', side_effect=fake.post), \
                patch('core.ollama_client.requests.get', side_effect=fake.get):
            self.assertEqual(Host()._comic_ollama_complete('sys', 'user'), '{"ok": 1}')
        body = fake.requests_to('/api/chat')[0]
        self.assertIs(body['think'], False)
        self.assertEqual(body['model'], 'reasoner')
        self.assertEqual(body['messages'], [{'role': 'system', 'content': 'sys'}, {'role': 'user', 'content': 'user'}])


if __name__ == '__main__':
    unittest.main()
