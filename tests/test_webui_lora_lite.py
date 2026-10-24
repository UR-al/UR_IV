"""WebUIBackend.get_loras — sam-extra 의 가벼운 GET /sam3-lora/lite 와 /sdapi/v1/loras 폴백.

/sdapi/v1/loras 는 LoRA 마다 safetensors 메타데이터 전체를 싣는다(809개 라이브러리에서 143MB JSON, 따뜻한
캐시 ~5초·차가운 캐시 72초, 2026-10-24 실측). 예전 timeout 10초로는 빈 LoRA 목록이 나왔다.
lite 경로는 같은 순서·같은 name/alias/path 에 activation_text 와 정렬된 tag_frequency_top(앞 32개)만 싣는다.

여기서는 하나의 가짜 메타데이터 묶음에서 두 응답을 모두 만들고(lite 는 sam-extra 계약대로 이 파일의
``_lite_item`` 이 만든다), lite 경로 결과가 **예전 get_loras 코드 그대로**(``_legacy_parse``)의 결과와
같은지 본다. 폴백(404·응답 지연·연결 실패·모양 틀린 JSON)과, 404 를 기억하지 않고 다음 호출에 lite 를 다시
묻는지도 본다(막 뜬 Forge 의 404 는 잠깐이다 — test_404_falls_back_and_is_not_remembered).
"""
import json
import unittest
from unittest import mock

import requests

from backends.webui_backend import WebUIBackend

API = 'http://127.0.0.1:7860'
LITE_URL = f'{API}/sam3-lora/lite'
FULL_URL = f'{API}/sdapi/v1/loras'


# ── 예전 get_loras 의 파싱(2026-10-24 이전 코드 그대로) — 동작 동일성의 기준 ──────────────
def _legacy_parse(data):
    result = []
    for item in data:
        lora = {
            'name': item.get('name', ''),
            'alias': item.get('alias', item.get('name', '')),
            'path': item.get('path', ''),
            'trigger_words': [],
        }
        metadata = item.get('metadata', {})
        if metadata:
            act_text = (metadata.get('activation text', '')
                        or metadata.get('ss_activation_text', ''))
            if act_text:
                lora['trigger_words'] = [
                    t.strip() for t in act_text.split(',') if t.strip()
                ][:8]
            if not lora['trigger_words']:
                tag_freq = metadata.get('ss_tag_frequency', {})
                if isinstance(tag_freq, str):
                    try:
                        tag_freq = json.loads(tag_freq)
                    except Exception:
                        tag_freq = {}
                if isinstance(tag_freq, dict):
                    all_tags = {}
                    for ds_tags in tag_freq.values():
                        if isinstance(ds_tags, dict):
                            all_tags.update(ds_tags)
                    if all_tags:
                        sorted_tags = sorted(all_tags.items(), key=lambda x: x[1], reverse=True)
                        lora['trigger_words'] = [t[0] for t in sorted_tags[:5]]
        result.append(lora)
    return result


# ── sam-extra 쪽 계약(GET /sam3-lora/lite 항목)을 그대로 흉내 낸다 ─────────────────────────
def _lite_item(item):
    metadata = item.get('metadata') or {}
    activation = metadata.get('activation text', '') or metadata.get('ss_activation_text', '') or ''
    tag_freq = metadata.get('ss_tag_frequency', {})
    if isinstance(tag_freq, str):
        try:
            tag_freq = json.loads(tag_freq)
        except Exception:
            tag_freq = {}
    merged = {}
    if isinstance(tag_freq, dict):
        for ds_tags in tag_freq.values():
            if isinstance(ds_tags, dict):
                merged.update(ds_tags)
    top = sorted(merged.items(), key=lambda kv: kv[1], reverse=True)[:32]
    return {
        'name': item['name'], 'alias': item['alias'], 'path': item['path'],
        'activation_text': activation if isinstance(activation, str) else '',
        'tag_frequency_top': [[tag, count] for tag, count in top],
    }


def _lite_body(full):
    items = [_lite_item(item) for item in full]
    return {'version': 1, 'count': len(items), 'items': items}


def _many_tags():
    # 40개 태그 — 서버가 32개로 자르고, 5번째 자리에서 동점이 갈린다(안정 정렬 = 합친 순서).
    tags = {f'tag{i:02d}': 100 - i for i in range(4)}           # 100, 99, 98, 97
    tags.update({f'tie{i:02d}': 50 for i in range(6)})          # 동점 6개 — 앞 1개만 5위 안
    tags.update({f'low{i:02d}': 1 for i in range(30)})
    return tags


FULL = [
    {'name': 'act', 'alias': 'act_alias', 'path': 'D:/LoRA/a/act.safetensors',
     'metadata': {'activation text': ' girl,  red hair ,, smile ,'}},
    {'name': 'ss_act', 'alias': 'ss_act', 'path': 'D:/LoRA/b/ss_act.safetensors',
     'metadata': {'activation text': '', 'ss_activation_text': 'x, y'}},
    {'name': 'many_act', 'alias': 'many_act', 'path': 'D:/LoRA/c/many_act.safetensors',
     'metadata': {'activation text': ','.join(f'w{i}' for i in range(12)),
                  'ss_tag_frequency': {'ds': {'ignored': 9}}}},
    {'name': 'blank_act_json', 'alias': 'blank_act_json', 'path': 'D:/LoRA/d/blank.safetensors',
     'metadata': {'activation text': ' , ,',
                  'ss_tag_frequency': json.dumps({'1_a': {'cat': 3, 'dog': 7}, '2_b': {'cat': 9, 'fish': 7}})}},
    {'name': 'dict_ties', 'alias': 'dict_ties', 'path': 'D:/LoRA/e/ties.safetensors',
     'metadata': {'ss_tag_frequency': {'1_x': _many_tags(), '2_y': 'not a dict'}}},
    {'name': 'json_ties', 'alias': 'json_ties', 'path': 'D:/LoRA/e/ties2.safetensors',
     'metadata': {'ss_tag_frequency': json.dumps({'only': _many_tags()})}},
    {'name': 'bad_json', 'alias': 'bad_json', 'path': 'D:/LoRA/f/bad.safetensors',
     'metadata': {'ss_tag_frequency': '{not json'}},
    {'name': 'list_freq', 'alias': 'list_freq', 'path': 'D:/LoRA/f/list.safetensors',
     'metadata': {'ss_tag_frequency': ['a', 'b']}},
    {'name': 'empty_freq', 'alias': 'empty_freq', 'path': 'D:/LoRA/f/emptyf.safetensors',
     'metadata': {'ss_tag_frequency': {'ds': {}}}},
    {'name': 'empty_meta', 'alias': 'empty_meta', 'path': 'D:/LoRA/g/empty.safetensors', 'metadata': {}},
    {'name': 'none_meta', 'alias': 'none_meta', 'path': 'D:/LoRA/g/none.safetensors', 'metadata': None},
    {'name': 'no_meta', 'alias': 'renamed', 'path': 'D:/LoRA/g/no.safetensors'},
]


def _response(status=200, body=None, raw=None):
    response = requests.Response()
    response.status_code = status
    response.url = ''
    if raw is not None:
        response._content = raw
    else:
        response._content = json.dumps(body if body is not None else {}).encode('utf-8')
    return response


class _Server:
    """URL → 응답(또는 예외). 받은 (url, timeout) 을 기록한다."""

    def __init__(self, lite, full):
        self.lite, self.full = lite, full
        self.calls = []

    def get(self, url, headers=None, timeout=None, **_kw):
        self.calls.append((url, timeout))
        reply = self.lite if url == LITE_URL else self.full if url == FULL_URL else None
        if reply is None:
            raise AssertionError(f'예상 밖 URL {url}')
        if isinstance(reply, BaseException):
            raise reply
        return reply

    def urls(self):
        return [url for url, _ in self.calls]


class LoraLiteTests(unittest.TestCase):
    def setUp(self):
        self.backend = WebUIBackend(API)
        self.expected = _legacy_parse(FULL)

    def _run(self, server):
        with mock.patch('backends.webui_backend.requests.get', side_effect=server.get):
            return self.backend.get_loras()

    def test_fixture_covers_both_methods(self):
        """기준 자체가 의미 있는지 — Method 1·2·빈 결과가 모두 나온다."""
        by_name = {lora['name']: lora['trigger_words'] for lora in self.expected}
        self.assertEqual(by_name['act'], ['girl', 'red hair', 'smile'])
        self.assertEqual(by_name['ss_act'], ['x', 'y'])
        self.assertEqual(len(by_name['many_act']), 8)
        self.assertEqual(by_name['blank_act_json'], ['cat', 'dog', 'fish'])   # 뒤 데이터셋 cat=9 가 이긴다, 7 동점은 합친 순서
        self.assertEqual(by_name['dict_ties'], ['tag00', 'tag01', 'tag02', 'tag03', 'tie00'])
        self.assertEqual(by_name['json_ties'], by_name['dict_ties'])
        for name in ('bad_json', 'list_freq', 'empty_freq', 'empty_meta', 'none_meta', 'no_meta'):
            self.assertEqual(by_name[name], [], name)
        self.assertEqual(self.expected[-1]['alias'], 'renamed')

    def test_lite_path_matches_the_legacy_full_path(self):
        server = _Server(_response(body=_lite_body(FULL)), AssertionError('전체 목록을 부르면 안 된다'))
        self.assertEqual(self._run(server), self.expected)
        self.assertEqual(server.calls, [(LITE_URL, 30)])

    def test_new_full_path_matches_the_legacy_parse(self):
        server = _Server(_response(404, {'detail': 'Not Found'}), _response(body=FULL))
        self.assertEqual(self._run(server), self.expected)
        self.assertEqual(server.calls, [(LITE_URL, 30), (FULL_URL, 120)])

    def test_lite_success_logs_one_line(self):
        server = _Server(_response(body=_lite_body(FULL)), None)
        with self.assertLogs('backends.webui_backend', level='INFO') as logs:
            self._run(server)
        lines = [line for line in logs.output if 'LoRA 목록' in line]
        self.assertEqual(len(lines), 1)
        self.assertIn('/sam3-lora/lite', lines[0])
        self.assertIn(f'{len(FULL)}개', lines[0])

    def test_404_falls_back_and_is_not_remembered(self):
        """Forge 는 앱이 준비 확인에 쓰는 /sdapi/v1/samplers 를 연 뒤에야(create_api) 확장 라우트를 붙인다
        (app_started — 내장 LoRA 의 /sdapi/v1/loras 가 sam-extra 보다 먼저). 그 틈의 404 를 기억하면 그 백엔드 객체가
        사는 동안 143MB 경로에 묶인다 — 매번 lite 부터 묻는다(없는 라우트에 한 번 더 묻는 값은 수 ms)."""
        server = _Server(_response(404, {'detail': 'Not Found'}), _response(body=FULL))
        with self.assertLogs('backends.webui_backend', level='INFO') as logs:
            self.assertEqual(self._run(server), self.expected)
        self.assertEqual(server.calls, [(LITE_URL, 30), (FULL_URL, 120)])
        self.assertTrue(any('/sdapi/v1/loras' in line and 'lite 404' in line for line in logs.output))

        server.calls.clear()
        server.lite = _response(body=_lite_body(FULL))   # 그새 sam-extra 가 라우트를 붙였다
        self.assertEqual(self._run(server), self.expected)
        self.assertEqual(server.urls(), [LITE_URL], '앞의 404 를 기억해 lite 를 건너뛰면 안 된다')

    def test_timeout_falls_back_and_asks_lite_again(self):
        server = _Server(requests.exceptions.ReadTimeout('slow'), _response(body=FULL))
        self.assertEqual(self._run(server), self.expected)
        self.assertEqual(server.calls, [(LITE_URL, 30), (FULL_URL, 120)])
        server.calls.clear()
        self._run(server)
        self.assertEqual(server.urls(), [LITE_URL, FULL_URL], '다음 호출도 lite 부터 묻는다')

    def test_connection_error_falls_back(self):
        server = _Server(requests.exceptions.ConnectionError('refused'), _response(body=FULL))
        self.assertEqual(self._run(server), self.expected)
        self.assertEqual(server.urls(), [LITE_URL, FULL_URL])

    def test_server_error_falls_back(self):
        server = _Server(_response(500, {'detail': 'boom'}), _response(body=FULL))
        self.assertEqual(self._run(server), self.expected)
        self.assertEqual(server.urls(), [LITE_URL, FULL_URL])

    def test_bad_lite_json_falls_back(self):
        bad_bodies = [
            _response(raw=b'<html>not json</html>'),
            _response(body=[]),                                                   # 객체가 아님
            _response(body={'version': 2, 'count': 0, 'items': []}),              # 모르는 버전
            _response(body={'version': 1, 'count': 1}),                           # items 없음
            _response(body={'version': 1, 'count': 1, 'items': ['x']}),           # 항목이 객체가 아님
            _response(body={'version': 1, 'count': 1, 'items': [
                {'name': 'a', 'alias': 'a', 'path': 'p', 'activation_text': '', 'tag_frequency_top': 'x'}]}),
        ]
        for lite in bad_bodies:
            with self.subTest(body=lite._content[:60]):
                self.backend = WebUIBackend(API)
                server = _Server(lite, _response(body=FULL))
                self.assertEqual(self._run(server), self.expected)
                self.assertEqual(server.urls(), [LITE_URL, FULL_URL])

    def test_both_paths_failing_gives_empty_list(self):
        server = _Server(requests.exceptions.ConnectionError('down'), requests.exceptions.ConnectionError('down'))
        with self.assertLogs('backends.webui_backend', level='WARNING'):
            self.assertEqual(self._run(server), [])

    def test_empty_lite_list_is_a_valid_answer(self):
        """LoRA 가 하나도 없는 Forge — lite 의 빈 목록을 그대로 믿고 143MB 경로로 가지 않는다."""
        server = _Server(_response(body={'version': 1, 'count': 0, 'items': []}), AssertionError('폴백 금지'))
        self.assertEqual(self._run(server), [])
        self.assertEqual(server.urls(), [LITE_URL])

    def test_lite_order_is_kept(self):
        reversed_full = list(reversed(FULL))
        server = _Server(_response(body=_lite_body(reversed_full)), None)
        self.assertEqual([lora['name'] for lora in self._run(server)],
                         [item['name'] for item in reversed_full])


class LiteContractRegistryTests(unittest.TestCase):
    """앱이 묻는 라우트·그 확장 파일이 sam-extra 계약 레지스트리에 mapped 로 있다.

    없으면 그 라우트가 든 sam-extra 를 설치하는 순간 tests/test_sam_extra_contract.py 의 test_routes·
    test_module_units 가 '분류되지 않은 새 항목' 으로 실패한다(2026-10-24 작업 트리로 확인).
    """

    def test_route_and_module_are_mapped(self):
        from backends import webui_backend as wb
        from core import sam_extra_contract as reg
        route = reg.ROUTES[f'GET {wb._LORA_LITE_PATH}']
        self.assertEqual(route['status'], reg.MAPPED)
        self.assertIn('backends/webui_backend.py:_LORA_LITE_PATH', route['app'])
        self.assertEqual(reg.MODULES['sam3ext/lora_lite.py']['status'], reg.MAPPED)


if __name__ == '__main__':
    unittest.main()
