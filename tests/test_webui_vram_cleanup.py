"""Forge VRAM 정리 경로 — 실제로 VRAM 을 회수하는 요청만 보내고 Forge 설정을 건드리지 않는가.

회귀: cleanup_models(대기열 정기 정리 · 수동 언로드)가 POST refresh-loras(LoRA 폴더 재스캔)와
POST options {'memmon_poll_rate': 8} 를 보냈다. 둘 다 메모리를 정리하지 않고, options POST 는
사용자의 Forge 설정을 5→8 로 영구히 바꾸며 config.json 을 매번 다시 썼다. VRAM 을 실제로
회수하는 것은 unload-checkpoint 뿐이다.
"""
import json
import unittest
from types import SimpleNamespace
from unittest import mock

import requests

from backends.webui_backend import WebUIBackend


def _response(status=200):
    response = requests.Response()
    response.status_code = status
    response._content = b'{}'
    return response


class WebUIUnloadTests(unittest.TestCase):
    def setUp(self):
        self.backend = WebUIBackend('http://127.0.0.1:7860')

    def _posted_urls(self, post):
        return [call.kwargs.get('url') or call.args[0] for call in post.call_args_list]

    def test_unload_models_only_unloads_the_checkpoint(self):
        with mock.patch('backends.webui_backend.requests.post', return_value=_response()) as post:
            self.assertTrue(self.backend.unload_models())
        self.assertEqual(self._posted_urls(post), ['http://127.0.0.1:7860/sdapi/v1/unload-checkpoint'])
        for call in post.call_args_list:
            self.assertNotIn('json', call.kwargs)                   # 설정(options)을 쓰지 않는다

    def test_cleanup_models_is_gone(self):
        self.assertFalse(hasattr(self.backend, 'cleanup_models'))

    def test_unload_checkpoint_reports_http_errors(self):
        with mock.patch('backends.webui_backend.requests.post', return_value=_response(404)):
            self.assertFalse(self.backend.unload_checkpoint())       # 엔드포인트 없는 옛 버전
        with mock.patch('backends.webui_backend.requests.post',
                        side_effect=requests.exceptions.ConnectionError('down')):
            self.assertFalse(self.backend.unload_checkpoint())

    def test_refresh_loras_is_an_explicit_rescan(self):
        with mock.patch('backends.webui_backend.requests.post', return_value=_response()) as post:
            self.assertTrue(self.backend.refresh_loras())
        self.assertEqual(self._posted_urls(post), ['http://127.0.0.1:7860/sdapi/v1/refresh-loras'])
        with mock.patch('backends.webui_backend.requests.post', return_value=_response(500)):
            self.assertFalse(self.backend.refresh_loras())


class LoraRescanTests(unittest.TestCase):
    """LoRA 매니저 '목록 다시 스캔'(getLoras mode='force')만 Forge 재스캔을 요청한다."""

    def _get_loras(self, mode, backend):
        from ui.vue_bridge import VueBridge
        from ui import lora_catalog_cache

        previous = lora_catalog_cache.raw_loras()
        lora_catalog_cache._raw_loras = [{'name': 'cached'}]
        try:
            with mock.patch('backends.get_backend', return_value=backend), \
                    mock.patch('backends.get_backend_type', return_value=SimpleNamespace(value='webui')), \
                    mock.patch('ui.lora_catalog_cache.merged_json',
                               side_effect=lambda loras, engine: json.dumps(loras)):
                return json.loads(VueBridge.getLoras(None, mode))
        finally:
            lora_catalog_cache._raw_loras = previous

    def test_force_refresh_rescans_before_listing(self):
        order = []
        backend = SimpleNamespace(
            refresh_loras=lambda: order.append('refresh') or True,
            get_loras=lambda: order.append('list') or [{'name': 'new'}],
        )
        self.assertEqual(self._get_loras('force', backend), [{'name': 'new'}])
        self.assertEqual(order, ['refresh', 'list'])

    def test_plain_open_does_not_rescan(self):
        backend = SimpleNamespace(refresh_loras=mock.Mock(), get_loras=mock.Mock(return_value=[]))
        self._get_loras('', backend)
        backend.refresh_loras.assert_not_called()

    def test_backend_without_rescan_still_lists(self):
        backend = SimpleNamespace(get_loras=lambda: [{'name': 'comfy'}])
        self.assertEqual(self._get_loras('force', backend), [{'name': 'comfy'}])


class AsyncLoraRescanTests(unittest.TestCase):
    """LoRA 매니저가 쓰는 비동기 경로(requestLoras → lorasReady) — 재스캔 POST(최대 20초)가 GUI 스레드를
    막지 않고 워커에서 목록보다 먼저 나간다. 예전 동기 getLoras('force')는 QWebChannel 슬롯 안에서
    refresh-loras 와 get_loras 를 GUI 스레드로 보내 큰 라이브러리에서 창(웹 모드는 WebSocket)이 멈췄다."""

    def test_force_rescan_runs_in_the_worker_before_listing(self):
        from ui.vue_bridge import VueBridge
        from ui import lora_catalog_cache

        order = []
        backend = SimpleNamespace(
            refresh_loras=lambda: order.append('refresh') or True,
            get_loras=lambda: order.append('list') or [{'name': 'new'}],
        )
        emitted = []
        bridge = SimpleNamespace(
            _async_lookup_lock=__import__('threading').Lock(), _async_lookup_inflight=set(),
            _merged_lora_cache=None, lorasReady=SimpleNamespace(emit=emitted.append),
        )
        bridge._run_async_lookup = lambda key, loader, signal: VueBridge._run_async_lookup(
            bridge, key, loader, signal)
        started = []
        previous = lora_catalog_cache.raw_loras()
        lora_catalog_cache._raw_loras = [{'name': 'cached'}]
        try:
            with mock.patch('backends.get_backend', return_value=backend), \
                    mock.patch('backends.get_backend_type', return_value=SimpleNamespace(value='webui')), \
                    mock.patch('ui.lora_catalog_cache.merged_json',
                               side_effect=lambda loras, engine: json.dumps(loras)), \
                    mock.patch('ui.vue_bridge.threading.Thread',
                               side_effect=lambda target, **_kw: SimpleNamespace(
                                   start=lambda: started.append(target))):
                VueBridge.requestLoras(bridge, 'force', 'r1')
                self.assertEqual(order, [])                      # 슬롯은 HTTP 없이 곧바로 돌아온다
                self.assertEqual(len(started), 1)
                started[0]()                                     # 워커 스레드 몫
        finally:
            lora_catalog_cache._raw_loras = previous
        self.assertEqual(order, ['refresh', 'list'])
        reply = json.loads(emitted[0])
        self.assertEqual((reply['requestId'], reply['mode'], reply['loras']), ('r1', 'force', [{'name': 'new'}]))


if __name__ == '__main__':
    unittest.main()
