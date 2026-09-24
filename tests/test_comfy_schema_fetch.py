"""Bounded, cancellable /object_info read for Generation API ComfyUI profiles.

No network: ``requests.get`` is replaced by a fake streaming response.
"""
from __future__ import annotations

import json
import unittest
from unittest import mock

from core.comfy_schema_fetch import (
    MAX_OBJECT_INFO_BYTES, SchemaFetchCancelled, fetch_object_info_bounded,
)


class _Response:
    def __init__(self, status=200, chunks=(b"{}",), headers=None):
        self.status_code = status
        self.headers = headers or {}
        self._chunks = list(chunks)
        self.closed = False

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.closed = True
        return False

    def iter_content(self, _size):
        yield from self._chunks


class FetchObjectInfoBoundedTests(unittest.TestCase):
    def _fetch(self, response, **kwargs):
        get = mock.Mock(return_value=response)
        return get, fetch_object_info_bounded("http://remote.example:8188/", get=get, **kwargs)

    def test_streams_without_redirects_and_returns_the_document(self):
        body = json.dumps({"KSampler": {"input": {}}}).encode("utf-8")
        response = _Response(chunks=(body[:5], b"", body[5:]))
        get, data = self._fetch(response)
        self.assertEqual(data, {"KSampler": {"input": {}}})
        get.assert_called_once()
        self.assertEqual(get.call_args.args[0], "http://remote.example:8188/object_info")
        self.assertIs(get.call_args.kwargs["allow_redirects"], False)
        self.assertIs(get.call_args.kwargs["stream"], True)
        self.assertTrue(response.closed)

    def test_redirect_and_error_status_are_refused(self):
        with self.assertRaisesRegex(RuntimeError, "리디렉션"):
            self._fetch(_Response(status=302, headers={"Location": "http://elsewhere/"}))
        with self.assertRaisesRegex(RuntimeError, "HTTP 500"):
            self._fetch(_Response(status=500))

    def test_declared_or_streamed_size_over_the_cap_is_refused(self):
        with self.assertRaisesRegex(RuntimeError, "허용 크기"):
            self._fetch(_Response(headers={"Content-Length": str(MAX_OBJECT_INFO_BYTES + 1)}))
        with self.assertRaisesRegex(RuntimeError, "허용 크기"):
            self._fetch(_Response(chunks=(b"{" + b" " * 8, b" " * 8)), max_bytes=10)

    def test_non_json_or_non_object_bodies_are_refused(self):
        with self.assertRaisesRegex(RuntimeError, "JSON"):
            self._fetch(_Response(chunks=(b"<html>",)))
        with self.assertRaisesRegex(RuntimeError, "객체"):
            self._fetch(_Response(chunks=(b"[1, 2]",)))

    def test_cancel_before_the_request_and_between_chunks(self):
        get = mock.Mock()
        with self.assertRaises(SchemaFetchCancelled):
            fetch_object_info_bounded("http://remote.example:8188", get=get, cancel_check=lambda: True)
        get.assert_not_called()

        calls = iter([False, False, True])  # before request, first chunk, second chunk
        response = _Response(chunks=(b'{"a"', b": 1}"))
        with self.assertRaises(SchemaFetchCancelled):
            self._fetch(response, cancel_check=lambda: next(calls))
        self.assertTrue(response.closed)


if __name__ == "__main__":
    unittest.main()
