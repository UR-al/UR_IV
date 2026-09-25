"""core.forge_memo_client — sam-extra 메모 라우트 HTTP 클라이언트(요청은 가짜로 대신한다)."""
import unittest

from core.forge_memo_client import (
    MEMO_API_PATH, ForgeMemoClient, MemoConflictError, MemoNotFoundError, MemoRejectedError,
    MemoRemoteError, MemoRoutesUnavailable,
)

TS = "2026-09-25T00:00:10+00:00"


def memo(memo_id="m1", **extra):
    base = {"id": memo_id, "title": "제목", "text": "본문", "created_at": TS, "updated_at": TS,
            "deleted": False}
    base.update(extra)
    return base


class Response:
    def __init__(self, status, body=None, *, bad_json=False):
        self.status_code = status
        self._body = body
        self._bad = bad_json

    def json(self):
        if self._bad:
            raise ValueError("not json")
        return self._body


class FakeRequest:
    def __init__(self, *responses, error=None):
        self.responses = list(responses)
        self.error = error
        self.calls = []

    def __call__(self, method, url, **kwargs):
        self.calls.append((method, url, kwargs))
        if self.error is not None:
            raise self.error
        return self.responses.pop(0)


class ClientTests(unittest.TestCase):
    def client(self, *responses, error=None, url="http://127.0.0.1:7860/"):
        self.fake = FakeRequest(*responses, error=error)
        return ForgeMemoClient(url, request=self.fake)

    def test_list_sends_notebook_header_and_parses(self):
        client = self.client(Response(200, {"revision": 7, "memos": [
            memo("a"), {"id": "../evil"}, "junk", memo("a"), memo("b", deleted=True)]}))
        result = client.list_memos()
        self.assertEqual(result.revision, 7)
        self.assertEqual([m["id"] for m in result.memos], ["a", "b"])
        self.assertTrue(result.memos[1]["deleted"])
        self.assertNotIn("dirty", result.memos[0])
        method, url, kwargs = self.fake.calls[0]
        self.assertEqual((method, url), ("GET", "http://127.0.0.1:7860" + MEMO_API_PATH))
        self.assertEqual(kwargs["headers"]["X-SAM3-Notebook"], "1")
        self.assertEqual(kwargs["params"], {"include_deleted": "1"})
        # 루프백은 connect 만 짧게(core.backend_probe.request_timeout)
        self.assertIsInstance(kwargs["timeout"], tuple)

    def test_old_extension_404_means_unavailable(self):
        client = self.client(Response(404, {"detail": "Not Found"}), Response(404, {"detail": "Not Found"}))
        with self.assertRaises(MemoRoutesUnavailable):
            client.list_memos()
        self.assertFalse(client.probe())

    def test_probe_true_when_routes_exist(self):
        client = self.client(Response(200, {"revision": 0, "memos": []}))
        self.assertTrue(client.probe())
        self.assertNotIn("params", self.fake.calls[0][2])

    def test_errors_are_user_facing_and_never_leak_the_url(self):
        client = self.client(error=ConnectionError("http://127.0.0.1:7860 refused secret"))
        with self.assertRaises(MemoRemoteError) as ctx:
            client.list_memos()
        self.assertNotIn("7860", str(ctx.exception))
        for status in (401, 403, 500, 413, 400):
            client = self.client(Response(status, {"detail": "nope"}))
            with self.assertRaises(MemoRemoteError) as raised:
                client.list_memos()
            if status in (401, 403):   # 401 은 --gradio-auth 로그인과 --api·--nowebui 의 --api-auth 둘 다에서 온다
                for flag in ("--gradio-auth", "--api-auth"):
                    self.assertIn(flag, str(raised.exception))
        with self.assertRaises(MemoRemoteError):
            self.client(Response(200, bad_json=True)).list_memos()
        with self.assertRaises(MemoRemoteError):
            self.client(Response(200, {"revision": 1})).list_memos()
        with self.assertRaises(MemoRemoteError):
            self.client(Response(200, [])).list_memos()

    def test_put_body_url_encoding_and_result(self):
        stored = memo("ns:1", title="t", text="x")
        client = self.client(Response(200, {"revision": 3, "memo": stored}))
        revision, saved = client.put_memo("ns:1", title="t", text="x", updated_at=TS, base_updated_at=None)
        self.assertEqual((revision, saved["id"]), (3, "ns:1"))
        method, url, kwargs = self.fake.calls[0]
        self.assertEqual(method, "PUT")
        self.assertTrue(url.endswith(MEMO_API_PATH + "/ns%3A1"))
        self.assertEqual(kwargs["json"], {"title": "t", "text": "x", "updated_at": TS, "base_updated_at": None})
        self.assertEqual(kwargs["headers"]["X-SAM3-Notebook"], "1")

    def test_put_conflict_carries_the_stored_memo(self):
        client = self.client(Response(409, {"detail": "newer", "memo": memo("m1", text="server")}))
        with self.assertRaises(MemoConflictError) as ctx:
            client.put_memo("m1", title="t", text="mine", updated_at=TS, base_updated_at=TS)
        self.assertEqual(ctx.exception.memo["text"], "server")

    def test_put_rejections_and_schema_409(self):
        with self.assertRaises(MemoRejectedError):
            self.client(Response(413, {"detail": "Memo storage holds at most 500 memos"})).put_memo(
                "m1", title="", text="")
        with self.assertRaises(MemoRejectedError) as ctx:
            self.client(Response(400, {"detail": "bad title"})).put_memo("m1", title="", text="")
        self.assertIn("bad title", str(ctx.exception))
        # 메모 없는 409 = 서버 저장소가 더 새 schema — 충돌 사본을 만들 일이 아니다
        with self.assertRaises(MemoRemoteError) as ctx:
            self.client(Response(409, {"detail": "newer schema"})).put_memo("m1", title="", text="")
        self.assertNotIsInstance(ctx.exception, MemoConflictError)
        with self.assertRaises(MemoRemoteError):
            self.client(Response(409, {"detail": "newer schema"})).list_memos()

    def test_put_sends_created_at_when_known(self):
        client = self.client(Response(200, {"revision": 1, "memo": memo("m1")}))
        client.put_memo("m1", title="t", text="x", updated_at=TS, created_at="2026-09-01T00:00:00+00:00")
        self.assertEqual(self.fake.calls[0][2]["json"]["created_at"], "2026-09-01T00:00:00+00:00")

    def test_put_404_is_old_extension_and_mismatched_reply_is_rejected(self):
        with self.assertRaises(MemoRoutesUnavailable):
            self.client(Response(404, {"detail": "Not Found"})).put_memo("m1", title="", text="")
        with self.assertRaises(MemoRemoteError):
            self.client(Response(200, {"revision": 1, "memo": memo("other")})).put_memo("m1", title="", text="")

    def test_delete(self):
        client = self.client(Response(200, {"revision": 4, "memo": memo("m1", text="", deleted=True)}))
        revision, tomb = client.delete_memo("m1")
        self.assertEqual((revision, tomb["deleted"]), (4, True))
        self.assertEqual(self.fake.calls[0][0], "DELETE")
        self.assertNotIn("json", self.fake.calls[0][2])
        with self.assertRaises(MemoNotFoundError):
            self.client(Response(404, {"detail": "unknown memo"})).delete_memo("m1")

    def test_invalid_ids_and_urls_never_send(self):
        client = self.client()
        with self.assertRaises(ValueError):
            client.put_memo("../x", title="", text="")
        with self.assertRaises(ValueError):
            client.delete_memo("")
        self.assertEqual(self.fake.calls, [])
        with self.assertRaises(ValueError):
            ForgeMemoClient("  ")
        self.assertEqual(ForgeMemoClient("http://host:7860//").base_url, "http://host:7860")
        self.assertEqual(ForgeMemoClient("http://192.168.0.5:7860")._timeout, 5.0)


if __name__ == "__main__":
    unittest.main()
