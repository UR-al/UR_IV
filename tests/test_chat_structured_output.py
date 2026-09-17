"""Offline chat contracts, plus a real loopback SSE round trip. No model runs."""
import base64
import json
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from unittest.mock import patch

from core.lmstudio_client import LMStudioClient
from core.ollama_client import OllamaClient
from core.structured_output import parse_schema, validate_output
from workers.chat_worker import ChatWorker
from tests.test_chat_feature import _FakeResponse, _chunks

SCHEMA = {'type': 'object', 'properties': {'answer': {'type': 'string'}},
          'required': ['answer'], 'additionalProperties': False}


def sse(content='{"answer":"안녕"}', reason='stop', done=True):
    result = ['data: ' + json.dumps({'choices': [{'index': 0, 'delta': {'content': content}, 'finish_reason': None}]}, ensure_ascii=False),
              'data: ' + json.dumps({'choices': [{'index': 0, 'delta': {}, 'finish_reason': reason}],
                                    'usage': {'completion_tokens': 12}})]
    return result + (['data: [DONE]'] if done else [])


class StructuredOutputTests(unittest.TestCase):
    def test_inline_schema_and_exact_output(self):
        self.assertEqual(parse_schema(json.dumps(SCHEMA)), SCHEMA)
        validate_output('{"answer":"안녕"}', SCHEMA)
        for content in ('```json\n{"answer":"ok"}\n```', '{"answer":3}', '{"answer":"ok","extra":1}', '{', 'NaN'):
            with self.subTest(content=content), self.assertRaises(ValueError):
                validate_output(content, SCHEMA)

    def test_invalid_oversized_and_external_schemas_fail_before_http(self):
        for schema in ('not json', '[]', '{}', {'type': 'invalid'}, {'$ref': 'http://127.0.0.1/private'},
                       {'$ref': '#'}, {'type': 'string', 'description': 'x' * 64000}, {'minimum': float('nan')}):
            with self.subTest(schema=str(schema)[:60]), patch('requests.post') as post:
                for client in (OllamaClient(), LMStudioClient()):
                    with self.assertRaises(ValueError):
                        client.chat_stream([], schema=schema)
                post.assert_not_called()

    def test_ollama_schema_and_finite_fallback_only_in_structured_mode(self):
        options = {'num_predict': -1, 'temperature': .2}
        with patch('core.ollama_client.requests.post', return_value=_FakeResponse(_chunks('{"answer":"yes"}'))) as post:
            OllamaClient().chat_stream([], schema=SCHEMA, options=options)
            self.assertEqual(post.call_args.kwargs['json']['format'], SCHEMA)
            self.assertEqual(post.call_args.kwargs['json']['options']['num_predict'], 4096)
            self.assertEqual(options['num_predict'], -1)
            OllamaClient().chat_stream([], options=options)
            self.assertNotIn('format', post.call_args.kwargs['json'])
            self.assertEqual(post.call_args.kwargs['json']['options']['num_predict'], -1)

    def test_lmstudio_payload_stream_image_and_utf8(self):
        image = base64.b64encode(b'\x89PNG\r\n\x1a\n' + b'x' * 30).decode()
        pieces = []
        with patch('core.lmstudio_client.requests.post', return_value=_FakeResponse(sse())) as post:
            result = LMStudioClient('http://localhost:1234/v1/', 'fixture').chat_stream(
                [{'role': 'user', 'content': '설명', 'images': [image]}],
                schema=SCHEMA, options={'temperature': .3, 'num_ctx': 4096, 'num_predict': 1024},
                think=True, on_token=pieces.append)
        self.assertEqual(post.call_args.args[0], 'http://localhost:1234/v1/chat/completions')
        payload = post.call_args.kwargs['json']
        self.assertEqual(payload['response_format']['json_schema']['schema'], SCHEMA)
        self.assertIs(payload['response_format']['json_schema']['strict'], True)
        self.assertEqual(payload['max_tokens'], 1024)
        self.assertNotIn('think', payload)
        self.assertNotIn('num_ctx', payload)
        self.assertTrue(payload['messages'][0]['content'][1]['image_url']['url'].startswith('data:image/png;base64,'))
        self.assertEqual(result['eval_count'], 12)
        self.assertEqual(''.join(pieces), '{"answer":"안녕"}')

    def test_lmstudio_eof_error_stop_and_no_silent_schema_downgrade(self):
        for response in (_FakeResponse(sse(done=False)), _FakeResponse(sse(reason=None)),
                         _FakeResponse(['data: {"error":{"message":"unsupported schema"}}']),
                         _FakeResponse([], status=400)):
            with patch('core.lmstudio_client.requests.post', return_value=response) as post:
                with self.assertRaises(RuntimeError):
                    LMStudioClient().chat_stream([], schema=SCHEMA)
                self.assertEqual(post.call_count, 1)
        with patch('core.lmstudio_client.requests.post') as post:
            self.assertTrue(LMStudioClient().chat_stream([], should_stop=lambda: True)['stopped'])
            post.assert_not_called()

    def test_worker_validates_complete_responses_and_preserves_invalid_content(self):
        for content, reason, stopped, expected in (
            ('{"answer":"yes"}', 'stop', False, True),
            ('{"answer":5}', 'stop', False, False),
            ('{"answer":"yes"}', 'length', False, False),
            ('{"answer":', None, True, False),
        ):
            with self.subTest(content=content, reason=reason):
                worker = ChatWorker('test', 'http://test.invalid', 'fixture', [], provider='lmstudio', schema=SCHEMA)
                completed = []
                worker.done.connect(lambda raw: completed.append(json.loads(raw)))
                with patch('workers.chat_worker.LMStudioClient.chat_stream', return_value={
                    'content': content, 'done_reason': reason, 'stopped': stopped,
                }):
                    worker.run()
                self.assertEqual(completed[0]['ok'], expected)
                self.assertEqual(completed[0]['content'], content)
                self.assertTrue(completed[0]['structured'])

    def test_real_http_loopback_sse_contract(self):
        captured = []

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *_):
                pass

            def do_POST(self):
                captured.append((self.path, json.loads(self.rfile.read(int(self.headers['Content-Length'])))))
                self.send_response(200)
                self.send_header('Content-Type', 'text/event-stream')
                self.end_headers()
                self.wfile.write(('\n\n'.join(sse()) + '\n\n').encode('utf-8'))

        server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            result = LMStudioClient(f'http://127.0.0.1:{server.server_port}', 'fixture').chat_stream(
                [{'role': 'user', 'content': 'JSON으로 답해줘'}], schema=SCHEMA)
            validate_output(result['content'], SCHEMA)
            self.assertEqual(captured[0][0], '/v1/chat/completions')
            self.assertEqual(captured[0][1]['response_format']['type'], 'json_schema')
        finally:
            server.shutdown(); server.server_close(); thread.join(2)
