"""WebUI 연결 정보 로드 — 꺼진 백엔드는 곧바로 실패를 알리고, 옛 워커는 메인 스레드를 막지 않는다."""
from __future__ import annotations

import time
import unittest
from types import SimpleNamespace
from unittest import mock

import requests

from backends.webui_backend import WebUIBackend


class _Response:
    def __init__(self, status_code=200, payload=None):
        self.status_code = status_code
        self._payload = payload if payload is not None else []

    def json(self):
        return self._payload

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError(f"{self.status_code}")


class WebUIGetInfoTests(unittest.TestCase):
    def setUp(self):
        self.backend = WebUIBackend("http://127.0.0.1:7860")

    def test_refused_connection_fails_immediately_without_backoff(self):
        with mock.patch("backends.webui_backend.requests.get",
                        side_effect=requests.exceptions.ConnectionError("refused")) as get, \
                mock.patch("backends.webui_backend.get_with_retry") as retry:
            started = time.monotonic()
            with self.assertRaises(requests.exceptions.ConnectionError):
                self.backend.get_info()
        self.assertLess(time.monotonic() - started, 0.5)
        retry.assert_not_called()
        self.assertEqual(1, get.call_count)
        # 루프백은 connect 만 짧게 끊는다(read 5초는 그대로).
        self.assertEqual((0.3, 5.0), get.call_args.kwargs["timeout"])

    def _count_sd_models_requests(self, sd_models_outcomes):
        """sd-models 요청을 requests 계층(requests.get + http_retry 의 requests.request)에서 센다.

        get_with_retry 를 통째로 가짜로 바꾸면 그 안의 재시도 횟수(retries+1)가 안 보여서,
        '한 번 더'가 실제로는 3회였던 것을 못 잡았다.
        """
        outcomes = list(sd_models_outcomes)
        sd_models = []

        def respond(url):
            if url.endswith("/sd-models"):
                sd_models.append(url)
                outcome = outcomes.pop(0)
                if isinstance(outcome, Exception):
                    raise outcome
                return outcome
            return _Response(200, {} if url.endswith("/options") else [])

        with mock.patch("backends.webui_backend.requests.get",
                        side_effect=lambda url, **_kw: respond(url)), \
                mock.patch("core.http_retry.requests.request",
                           side_effect=lambda _method, url, **_kw: respond(url)), \
                mock.patch("backends.webui_backend._SD_MODELS_RETRY_PAUSE_S", 0.0), \
                mock.patch("core.http_retry.time.sleep") as sleep:
            # time 모듈은 하나라 이 mock 이 webui_backend 의 5xx 대기(0초로 패치)도 받는다.
            try:
                info = self.backend.get_info()
            except Exception as exc:   # noqa: BLE001 — 호출자가 결과를 본다
                info = exc
        backoff = [c.args[0] for c in sleep.call_args_list if c.args and c.args[0] > 0]
        return info, len(sd_models), backoff

    def test_read_timeout_and_5xx_retry_exactly_once(self):
        cases = {
            "read-timeout": requests.exceptions.ReadTimeout("slow"),
            "503": _Response(503),
        }
        for name, first in cases.items():
            with self.subTest(first=name):
                info, count, backoff = self._count_sd_models_requests(
                    [first, _Response(200, [{"title": "m.safetensors", "model_name": "m"}])]
                )
                self.assertNotIsInstance(info, Exception)
                self.assertEqual(["m.safetensors"], info.models)
                self.assertEqual(2, count, "첫 요청 + 딱 한 번 더")
                self.assertEqual([], backoff, "공유 http_retry 의 지수 백오프를 타지 않는다")

    def test_second_transient_failure_is_not_retried_again(self):
        for name, outcomes in {
            "read-timeout-twice": [requests.exceptions.ReadTimeout("slow")] * 3,
            "503-twice": [_Response(503)] * 3,
            "timeout-then-refused": [requests.exceptions.ReadTimeout("slow"),
                                     requests.exceptions.ConnectionError("refused"),
                                     _Response(200, [])],
        }.items():
            with self.subTest(case=name):
                info, count, _backoff = self._count_sd_models_requests(outcomes)
                self.assertIsInstance(info, (requests.exceptions.RequestException,))
                self.assertEqual(2, count, "hung 백엔드도 5+5초에서 끝난다(예전엔 3회·약 16초)")

    def test_5xx_pauses_before_the_single_retry(self):
        with mock.patch("backends.webui_backend.requests.get",
                        side_effect=[_Response(503), _Response(200, [])]), \
                mock.patch("backends.webui_backend.get_with_retry",
                           return_value=_Response(200, [])), \
                mock.patch("backends.webui_backend.time.sleep") as pause:
            self.backend.get_info()
        pause.assert_called_once()
        self.assertGreater(pause.call_args.args[0], 0)

    def test_missing_scheduler_endpoint_falls_back_to_automatic(self):
        def fake_retry(url, **_kwargs):
            if url.endswith("/schedulers"):
                return _Response(404, {"detail": "Not Found"})
            if url.endswith("/samplers"):
                return _Response(200, [{"name": "Euler"}])
            return _Response(200, [] if not url.endswith("/options") else {})

        with mock.patch("backends.webui_backend.requests.get",
                        return_value=_Response(200, [{"title": "m.safetensors", "model_name": "m"}])), \
                mock.patch("backends.webui_backend.get_with_retry", side_effect=fake_retry):
            info = self.backend.get_info()
        self.assertEqual(["m.safetensors"], info.models)
        self.assertEqual(["Euler"], info.samplers)
        self.assertEqual(["Automatic"], info.schedulers)

    def test_remote_backends_keep_the_plain_timeout(self):
        backend = WebUIBackend("http://192.168.0.20:7860")
        with mock.patch("backends.webui_backend.requests.get",
                        side_effect=requests.exceptions.ConnectionError("down")) as get:
            with self.assertRaises(requests.exceptions.ConnectionError):
                backend.get_info()
        self.assertEqual(5.0, get.call_args.kwargs["timeout"])


class _Signal:
    def __init__(self):
        self.slots = []
        self.disconnects = 0

    def connect(self, slot):
        self.slots.append(slot)

    def disconnect(self):
        self.disconnects += 1
        if not self.slots:
            raise TypeError("no connections")
        self.slots.clear()


class _FakeWorker:
    created = []

    def __init__(self, parent=None):
        self.parent = parent
        self.info_ready = _Signal()
        self.error_occurred = _Signal()
        self.finished = _Signal()
        self.started = False
        self.deleted = False
        _FakeWorker.created.append(self)

    def start(self):
        self.started = True

    def deleteLater(self):
        self.deleted = True

    def quit(self):
        raise AssertionError("quit() 은 이벤트 루프 없는 워커에 효과가 없다")

    def wait(self, *_args):
        raise AssertionError("wait() 로 메인 스레드를 막지 않는다")

    def isRunning(self):
        return True


class LoadWebUIInfoWorkerTests(unittest.TestCase):
    def test_new_worker_is_parented_and_old_one_is_detached_without_blocking(self):
        from ui.generator_webui import WebUIMixin

        _FakeWorker.created.clear()
        host = SimpleNamespace(
            btn_generate=SimpleNamespace(setEnabled=lambda _value: None),
            btn_random_prompt=SimpleNamespace(setEnabled=lambda _value: None),
            on_webui_info_loaded=lambda _info: None,
            on_webui_info_error=lambda _error: None,
        )
        with mock.patch("ui.generator_webui.WebUIInfoWorker", _FakeWorker), \
                mock.patch("ui.generator_webui.get_backend_type", return_value=None):
            WebUIMixin.load_webui_info(host)
            first = host.info_worker
            WebUIMixin.load_webui_info(host)   # 옛 워커가 아직 돌고 있어도 막지 않는다
        second = host.info_worker
        self.assertIsNot(first, second)
        for worker in (first, second):
            self.assertIs(worker.parent, host)
            self.assertTrue(worker.started)
            self.assertEqual([worker.deleteLater], worker.finished.slots)
        self.assertEqual([], first.info_ready.slots, "옛 워커의 결과는 더 이상 받지 않는다")
        self.assertEqual([], first.error_occurred.slots)
        self.assertEqual(1, len(second.info_ready.slots))


if __name__ == "__main__":
    unittest.main()
