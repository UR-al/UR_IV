"""core.http_retry — 5xx·연결 오류만 재시도하고 4xx 는 바로 돌려준다."""
from __future__ import annotations

import unittest
from unittest import mock

import requests

from core import http_retry


def _response(status: int) -> requests.Response:
    response = requests.Response()
    response.status_code = status
    return response


class RequestWithRetryTests(unittest.TestCase):
    def setUp(self):
        self.sleep = mock.patch("core.http_retry.time.sleep").start()
        self.random = mock.patch("core.http_retry.random.random", return_value=0.5).start()
        self.addCleanup(mock.patch.stopall)

    def test_server_error_is_retried_until_success(self):
        with mock.patch("core.http_retry.requests.request",
                        side_effect=[_response(503), _response(502), _response(200)]) as request:
            response = http_retry.request_with_retry("GET", "http://127.0.0.1:7860/x", retries=3, timeout=5)
        self.assertEqual(response.status_code, 200)
        self.assertEqual(request.call_count, 3)
        request.assert_called_with("GET", "http://127.0.0.1:7860/x", timeout=5)
        # 지수 백오프 + jitter(random 0.5 × 기본 0.2 = 0.1초)
        self.assertEqual([c.args[0] for c in self.sleep.call_args_list], [1.0 + 0.1, 2.0 + 0.1])

    def test_client_error_returns_immediately_without_retry(self):
        with mock.patch("core.http_retry.requests.request", return_value=_response(404)) as request:
            response = http_retry.get_with_retry("http://127.0.0.1:7860/missing")
        self.assertEqual(response.status_code, 404)
        self.assertEqual(request.call_count, 1)
        self.sleep.assert_not_called()

    def test_last_server_error_response_is_returned_when_retries_run_out(self):
        with mock.patch("core.http_retry.requests.request", return_value=_response(500)) as request:
            response = http_retry.request_with_retry("GET", "http://h/x", retries=2)
        self.assertEqual(response.status_code, 500)
        self.assertEqual(request.call_count, 3)   # retries + 1

    def test_connection_errors_are_retried_then_reraised(self):
        error = requests.exceptions.ConnectionError("refused")
        with mock.patch("core.http_retry.requests.request", side_effect=error) as request:
            with self.assertRaises(requests.exceptions.ConnectionError):
                http_retry.request_with_retry("GET", "http://h/x", retries=2)
        self.assertEqual(request.call_count, 3)
        self.assertEqual(self.sleep.call_count, 2)

    def test_jitter_is_bounded_by_the_jitter_argument(self):
        self.random.return_value = 0.999
        with mock.patch("core.http_retry.requests.request", side_effect=[_response(503), _response(200)]):
            http_retry.request_with_retry("GET", "http://h/x", retries=1, backoff_factor=2.0, jitter=0.2)
        (wait,), _ = self.sleep.call_args
        self.assertGreaterEqual(wait, 1.0)
        self.assertLess(wait, 1.0 + 0.2)

    def test_unused_post_helper_is_gone(self):
        self.assertFalse(hasattr(http_retry, "post_with_retry"))


if __name__ == "__main__":
    unittest.main()
