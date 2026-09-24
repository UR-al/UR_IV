"""core.action_log — Vue 액션 로그 요약은 페이로드를 직렬화하지 않는다."""
from __future__ import annotations

import io
import json
import unittest
from unittest.mock import patch

from core.action_log import format_action_log, log_action, summarize_action_payload


class ActionLogSummaryTests(unittest.TestCase):
    def test_large_payload_is_summarised_by_keys_and_sizes(self) -> None:
        payload = {
            'results': [{'general': 'x' * 200}] * 40_000,
            'lineage': {'label': '2026_07', 'fingerprint': 'f' * 64, 'snapshot_id': 'a' * 32},
            'data': 'b' * 5_000_000,
            'count': 3,
        }
        with patch('json.dumps', side_effect=AssertionError('must not serialise')):
            text = summarize_action_payload(payload)
        self.assertIn('results=[40000]', text)
        self.assertIn('lineage={3}', text)
        self.assertIn('data=str(5000000)', text)
        self.assertIn('count=3', text)
        self.assertLessEqual(len(text), 160)

    def test_non_dict_payloads_do_not_raise(self) -> None:
        # onAction 의 except 가 TypeError 를 삼키면 액션이 조용히 사라진다 — 어떤 값도 예외 없이
        self.assertEqual(summarize_action_payload(None), 'None')
        self.assertEqual(summarize_action_payload([1, 2, 3]), '[3]')
        self.assertEqual(summarize_action_payload('ok'), "'ok'")
        self.assertEqual(summarize_action_payload(object()), 'object')

    def test_many_keys_are_truncated(self) -> None:
        payload = {f'k{i}': i for i in range(20)}
        text = summarize_action_payload(payload)
        self.assertIn('+12 keys', text)
        self.assertNotIn('k19', text)

    def test_long_summary_is_capped(self) -> None:
        payload = {('key_' + 'x' * 60 + str(i)): 'v' for i in range(8)}
        self.assertLessEqual(len(summarize_action_payload(payload, limit=100)), 100)

    def test_small_payload_stays_readable(self) -> None:
        self.assertEqual(
            summarize_action_payload({'ratings': ['g', 's'], 'enabled': True}),
            '{ratings=[2], enabled=True}',
        )
        json.dumps({})  # 모듈 자체는 json 을 쓰지 않아도 테스트 환경은 정상


class ActionLogAsciiTests(unittest.TestCase):
    """cp949 로 리다이렉트된 stdout 에서도 로그 print 가 실패하지 않아야 한다(액션 유실 방지)."""

    def test_summary_is_pure_ascii_for_non_ascii_values_and_keys(self) -> None:
        text = summarize_action_payload({
            'type': 'info',
            'msg': 'Pokémon 저장',
            'emoji': 'café 😀',
            '키': '—✓',
        })
        self.assertTrue(text.isascii(), text)
        text.encode('cp949')   # 예외 없음
        # 값은 ascii() 표기(escape 를 되돌리면 원문) — 로그로 원래 글자를 알아볼 수 있다
        self.assertIn('msg=' + ascii('Pokémon 저장'), text)
        self.assertIn('emoji=' + ascii('café 😀'), text)
        self.assertIn("msg='Pok" + chr(92) + "xe9mon ", text)
        escaped_key = '키'.encode('ascii', 'backslashreplace').decode('ascii')
        self.assertIn(escaped_key + '=', text)
        self.assertEqual(escaped_key.encode('ascii').decode('unicode_escape'), '키')
        self.assertTrue(summarize_action_payload('ü').isascii())

    def test_action_name_is_escaped_too(self) -> None:
        line = format_action_log('액션', {'k': 1})
        self.assertTrue(line.isascii(), line)
        prefix, _, rest = line.partition(' | Payload: ')
        self.assertEqual(rest, '{k=1}')
        name = prefix.removeprefix('[Bridge] Action Received: ')
        self.assertEqual(name.encode('ascii').decode('unicode_escape'), '액션')
        self.assertEqual(format_action_log('show_toast', None),
                         '[Bridge] Action Received: show_toast | Payload: None')

    def test_log_action_writes_to_a_cp949_stream(self) -> None:
        stream = io.TextIOWrapper(io.BytesIO(), encoding='cp949')
        log_action('show_toast', {'type': 'info', 'msg': 'Pokémon 😀'}, stream=stream)
        stream.flush()
        self.assertIn(b"msg='Pok\\xe9mon \\U0001f600'", stream.buffer.getvalue())

    def test_log_failures_never_propagate(self) -> None:
        closed = io.StringIO()
        closed.close()
        log_action('x', {'a': 1}, stream=closed)   # ValueError: I/O on closed file

        class Broken(io.StringIO):
            def write(self, _text):
                raise OSError('broken pipe')

        log_action('x', {'a': 1}, stream=Broken())

        class Strict(io.StringIO):
            def write(self, text):
                text.encode('ascii', 'strict')
                return super().write(text)

        stream = Strict()
        log_action('동작', {'m': '한글'}, stream=stream)
        self.assertTrue(stream.getvalue().isascii())


if __name__ == '__main__':
    unittest.main()
