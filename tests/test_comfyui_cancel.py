"""ComfyUI cancel must only ever touch this adapter's own prompt (UR-01).

A body-less POST /interrupt is a global interrupt on the ComfyUI host, so a
check-then-interrupt sequence can stop another client's prompt that started
in the gap. The host's /api/jobs/{id}/cancel does the check and the interrupt
atomically under the queue mutex; older hosts fall back to targeted calls.
"""
from __future__ import annotations

import unittest
from unittest import mock

from backends.comfyui_backend import ComfyUIBackend


def response(status=200, payload=None):
    resp = mock.Mock()
    resp.status_code = status
    resp.json.return_value = payload if payload is not None else {}

    def raise_for_status():
        if status >= 400:
            raise RuntimeError(f"HTTP {status}")

    resp.raise_for_status.side_effect = raise_for_status
    return resp


class ComfyCancelTests(unittest.TestCase):
    def setUp(self):
        self.backend = ComfyUIBackend("http://127.0.0.1:8188")
        # Never let a regression reach a real ComfyUI on 8188.
        self.get = mock.patch("backends.comfyui_backend.requests.get").start()
        self.addCleanup(mock.patch.stopall)

    def _assert_no_global_interrupt(self, post):
        for call in post.call_args_list:
            if call.args[0].endswith("/interrupt"):
                self.assertEqual(call.kwargs.get("json"), {"prompt_id": "ours"})

    def test_uses_atomic_job_cancel_api(self):
        with mock.patch("backends.comfyui_backend.requests.post",
                        return_value=response(200, {"cancelled": True})) as post:
            self.backend._cancel_prompt("ours")
        self.assertEqual(post.call_count, 1)
        self.assertEqual(post.call_args.args[0],
                         "http://127.0.0.1:8188/api/jobs/ours/cancel")
        self.get.assert_not_called()

    def test_job_id_is_url_quoted(self):
        with mock.patch("backends.comfyui_backend.requests.post",
                        return_value=response(200, {"cancelled": False})) as post:
            self.backend._cancel_prompt("a/b c")
        self.assertEqual(post.call_args.args[0],
                         "http://127.0.0.1:8188/api/jobs/a%2Fb%20c/cancel")
        self.get.assert_not_called()

    def test_finished_job_is_not_followed_by_any_interrupt(self):
        # {"cancelled": false} means it already finished: no fallback call at all.
        with mock.patch("backends.comfyui_backend.requests.post",
                        return_value=response(200, {"cancelled": False})) as post:
            self.backend._cancel_prompt("ours")
        self.assertEqual(post.call_count, 1)
        self.get.assert_not_called()

    def test_old_host_running_prompt_gets_targeted_interrupt(self):
        posts = [response(404), response(200), response(200)]
        queue = {"queue_running": [[1, "ours", {}, {}, []]], "queue_pending": []}
        self.get.return_value = response(200, queue)
        with mock.patch("backends.comfyui_backend.requests.post",
                        side_effect=posts) as post:
            self.backend._cancel_prompt("ours")
        urls = [c.args[0] for c in post.call_args_list]
        self.assertIn("http://127.0.0.1:8188/interrupt", urls)
        self._assert_no_global_interrupt(post)

    def test_old_host_running_other_prompt_is_never_interrupted(self):
        posts = [response(404), response(200)]
        queue = {"queue_running": [[2, "theirs", {}, {}, []]],
                 "queue_pending": [[3, "ours", {}, {}, []]]}
        self.get.return_value = response(200, queue)
        with mock.patch("backends.comfyui_backend.requests.post",
                        side_effect=posts) as post:
            self.backend._cancel_prompt("ours")
        urls = [c.args[0] for c in post.call_args_list]
        self.assertNotIn("http://127.0.0.1:8188/interrupt", urls)
        self.assertIn(mock.call("http://127.0.0.1:8188/queue",
                                json={"delete": ["ours"]}, timeout=5),
                      post.call_args_list)

    def test_old_host_dequeues_before_reading_the_running_prompt(self):
        # Read after the dequeue: a prompt that started before it is seen running,
        # and once dequeued it can no longer start unnoticed.
        calls = []
        queue = {"queue_running": [[1, "ours", {}, {}, []]], "queue_pending": []}

        def post(url, **kwargs):
            calls.append(("post", url))
            return response(404 if url.endswith("/cancel") else 200)

        def get(url, **kwargs):
            calls.append(("get", url))
            return response(200, queue)

        self.get.side_effect = get
        with mock.patch("backends.comfyui_backend.requests.post", side_effect=post):
            self.backend._cancel_prompt("ours")
        self.assertEqual(calls, [
            ("post", "http://127.0.0.1:8188/api/jobs/ours/cancel"),
            ("post", "http://127.0.0.1:8188/queue"),
            ("get", "http://127.0.0.1:8188/queue"),
            ("post", "http://127.0.0.1:8188/interrupt"),
        ])

    def test_cancel_errors_are_swallowed(self):
        with mock.patch("backends.comfyui_backend.requests.post",
                        side_effect=OSError("down")):
            self.backend._cancel_prompt("ours")


if __name__ == "__main__":
    unittest.main()
