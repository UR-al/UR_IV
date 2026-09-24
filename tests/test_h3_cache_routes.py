"""Exercise registered cache endpoints without starting an HTTP or Comfy host."""
import json
import tempfile
import threading
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from comfy_custom_nodes.ai_studio_forge_parity import h3_cache_nodes


class CacheRouteTests(unittest.IsolatedAsyncioTestCase):
    async def test_clear_rejects_queue_work_and_only_removes_owned_entries(self):
        # Only the external JSON response constructor is replaced; execute the
        # actual registered handlers and disk store without an HTTP listener.
        web = SimpleNamespace(json_response=lambda data, status=200:
            SimpleNamespace(body=json.dumps(data).encode("utf-8"), status=status))
        handlers = {}
        def register(path):
            def save(function):
                handlers[path] = function
                return function
            return save
        queue = [[], []]
        server = SimpleNamespace(routes=SimpleNamespace(get=register, post=register),
                                 prompt_queue=SimpleNamespace(get_current_queue=lambda: queue))
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "aistudio_cache" / "h3_conditioning"
            root.mkdir(parents=True)
            owned = root / ("a" * 64 + ".pt")
            owned.write_bytes(b"synthetic owned entry")
            foreign = root / "original.txt"
            foreign.write_text("preserve", encoding="utf-8")
            with mock.patch.dict("sys.modules", {
                "server": SimpleNamespace(PromptServer=SimpleNamespace(instance=server)),
                "folder_paths": SimpleNamespace(get_output_directory=lambda: tmp),
                "aiohttp": SimpleNamespace(web=web),
            }):
                h3_cache_nodes._register_routes()
                h3_cache_nodes._register_routes()  # Idempotent extension load.
                self.assertEqual(len(handlers), 2)
                response = await handlers["/aistudio/h3-cache/status"](None)
                self.assertEqual(json.loads(response.body)["entries"], 1)
                queue[1] = ["another client's pending job"]
                response = await handlers["/aistudio/h3-cache/clear"](None)
                self.assertEqual(response.status, 409)
                self.assertTrue(owned.exists())
                queue[1] = []
                response = await handlers["/aistudio/h3-cache/clear"](None)
                self.assertEqual(json.loads(response.body)["removedEntries"], 1)
                self.assertFalse(owned.exists())
                self.assertEqual(foreign.read_text(encoding="utf-8"), "preserve")


    async def test_status_and_clear_wait_for_the_disk_lock_off_the_event_loop(self):
        import asyncio

        web = SimpleNamespace(json_response=lambda data, status=200:
            SimpleNamespace(body=json.dumps(data).encode("utf-8"), status=status))
        handlers = {}

        def register(path):
            def save(function):
                handlers[path] = function
                return function
            return save

        server = SimpleNamespace(routes=SimpleNamespace(get=register, post=register),
                                 prompt_queue=SimpleNamespace(get_current_queue=lambda: ([], [])))
        with tempfile.TemporaryDirectory() as tmp, mock.patch.dict("sys.modules", {
            "server": SimpleNamespace(PromptServer=SimpleNamespace(instance=server)),
            "folder_paths": SimpleNamespace(get_output_directory=lambda: tmp),
            "aiohttp": SimpleNamespace(web=web),
        }):
            h3_cache_nodes._register_routes()
            for path in ("/aistudio/h3-cache/status", "/aistudio/h3-cache/clear"):
                with self.subTest(path=path):
                    held, release = threading.Event(), threading.Event()

                    def hold_lock():
                        # prompt_worker 가 put/get 중 _LOCK 을 쥐고 있는 상황
                        with h3_cache_nodes._LOCK:
                            held.set()
                            release.wait(5)

                    holder = threading.Thread(target=hold_lock)
                    holder.start()
                    self.assertTrue(held.wait(2))
                    try:
                        task = asyncio.ensure_future(handlers[path](None))
                        await asyncio.sleep(0.05)
                        # 루프가 멈추지 않았고, 핸들러는 락을 워커 스레드에서 기다린다.
                        self.assertFalse(task.done())
                    finally:
                        release.set()
                    response = await asyncio.wait_for(task, 5)
                    holder.join(5)
                    self.assertEqual(response.status, 200)


class CacheActionTests(unittest.TestCase):
    def test_status_uses_strict_disabled_pref_and_does_not_forward_server_paths(self):
        from ui.creator_actions import CreatorActionsMixin
        from backends import BackendType
        actions = CreatorActionsMixin()
        ready = threading.Event()
        events = []
        def emit(value):
            events.append(json.loads(value))
            ready.set()
        actions.vue_bridge = SimpleNamespace(creatorCacheEvent=SimpleNamespace(emit=emit))
        response = SimpleNamespace(status_code=200, raise_for_status=lambda: None,
            json=lambda: {"entries": 2, "bytes": 256, "path": "private/server/path", "maxBytes": 5})
        with (mock.patch("backends.get_backend_type", return_value=BackendType.COMFYUI),
              mock.patch("backends.get_backend", return_value=SimpleNamespace(api_url="http://synthetic.invalid")),
              mock.patch("requests.get", return_value=response),
              mock.patch.object(actions, "_creator_prefs", return_value={
                  "h3ConditioningCacheEnabled": "false", "h3ConditioningCacheMaxGB": 80,
                  "h3ConditioningCacheMaxEntries": -1})):
            actions._handle_creator_action("creator_h3_cache_status", {"requestId": "status-test"})
            self.assertTrue(ready.wait(3))
        event = events[-1]
        self.assertTrue(event["ok"])
        self.assertFalse(event["enabled"])
        self.assertEqual(event["requestId"], "status-test")
        self.assertEqual(event["maxBytes"], 64 * 1024 ** 3)
        self.assertEqual(event["maxEntries"], 1)
        self.assertEqual(event["entries"], 2)
        self.assertNotIn("path", event)


if __name__ == "__main__":
    unittest.main()
