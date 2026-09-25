"""Tile & Repair 액션(ui/tile_repair_actions.py) — 창·Forge·GPU 없이 브리지 경계만 본다.

가짜 클라이언트로 sam-extra 라우트를 대신한다. 결과 PNG 는 출력 폴더의 tile_repair/ 에 새 파일로 쓴다.
"""
from __future__ import annotations

import base64
import io
import json
import sys
import tempfile
import threading
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from PIL import Image

from backends import BackendType
from core.forge_tile_repair_client import TileRepairError, TileRepairResult
from core.sam_extra_capabilities import HttpResult, build_capabilities, EP_TILE_REPAIR
from ui import tile_repair_actions as actions
from ui.tile_repair_actions import TileRepairActionsMixin


def png(size=(24, 16)) -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", size, (90, 100, 120)).save(buffer, format="PNG")
    return buffer.getvalue()


def data_url() -> str:
    return "data:image/png;base64," + base64.b64encode(png()).decode()


class Signal:
    def __init__(self):
        self.items = []
        self.ready = threading.Event()

    def emit(self, raw):
        self.items.append(json.loads(raw))
        self.ready.set()


class FakeClient:
    def __init__(self, url, *, result=None, error=None, gate=None):
        self.url = url
        self.result = result or TileRepairResult(png((64, 96)), "p\nSeed: 7", 7, 64, 96,
                                                 "animaTileRepair_v20.safetensors")
        self.error = error
        self.gate = gate
        self.bodies = []
        self.stops = 0
        self.stop_answers = None     # e.g. [False, True] — Forge 가 요청을 아직 못 받았다가 받은 경우
        self.on_stop = None
        self.active = set()          # Forge 에 닿아 아직 답하지 않은 run 호출 번호(1부터)
        self.stopped_calls = set()   # stop 이 멈춘 run 호출 번호

    def options(self):
        if self.error:
            raise self.error
        return {"models": ["animaTileRepair_v20.safetensors"], "defaults": {}}

    def run(self, body):
        self.bodies.append(body)
        call = len(self.bodies)
        self.active.add(call)
        try:
            gate = self.gate
            if gate is not None:
                gate["entered"].set()
                gate["release"].wait(5)
            if self.error:
                raise self.error
            if call in self.stopped_calls:
                return TileRepairResult(b"", "", None, None, None, self.result.model, interrupted=True)
            return self.result
        finally:
            self.active.discard(call)

    def stop(self):
        self.stops += 1
        if self.on_stop is not None:
            self.on_stop()
        # sam-extra 의 stop 은 요청을 가리지 않는다 — 처리되는 순간 와 있는 route 요청을 전부 멈춘다(cancel_all).
        self.stopped_calls |= self.active
        if self.stop_answers is not None:
            return self.stop_answers.pop(0) if self.stop_answers else False
        return True


class Host(TileRepairActionsMixin):
    def __init__(self, output_dir, client):
        self.vue_bridge = SimpleNamespace(tileRepairResult=Signal())
        self.web_mode = False
        self.client = client
        self._tile_repair_client_factory = lambda url: (setattr(client, "url", url), client)[1]
        self._output_dir = output_dir

    def _tile_repair_output_root(self):
        return self._output_dir


class TileRepairActionTests(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.dir.cleanup)
        self.client = FakeClient("")
        self.host = Host(self.dir.name, self.client)
        self.signal = self.host.vue_bridge.tileRepairResult
        for target, value in (("backends.get_backend_type", BackendType.WEBUI),
                              ("backends.get_backend", SimpleNamespace(api_url="http://127.0.0.1:7860"))):
            patcher = mock.patch(target, return_value=value)
            patcher.start()
            self.addCleanup(patcher.stop)

    def wait(self):
        self.assertTrue(self.signal.ready.wait(5), "tile repair worker did not answer")
        self.signal.ready.clear()
        return self.signal.items[-1]

    def run_action(self, action, payload):
        self.assertTrue(self.host._handle_tile_repair_action(action, payload))

    def test_unrelated_actions_are_not_handled(self):
        self.assertFalse(self.host._handle_tile_repair_action("relight_preview", {}))

    def test_run_saves_a_new_png_and_answers_with_its_path(self):
        self.run_action("tile_repair_run", {"requestId": "run_1", "image_path": "", "image": data_url(),
                                            "settings": {"multiplier": -2.0, "model": ""}})
        event = self.wait()
        self.assertTrue(event["ok"], event)
        self.assertEqual((event["action"], event["requestId"]), ("tile_repair_run", "run_1"))
        path = Path(event["path"])
        self.assertEqual(path.parent, Path(self.dir.name).resolve() / "tile_repair")
        self.assertEqual(path.read_bytes(), self.client.result.png)
        self.assertEqual((event["width"], event["height"], event["seed"]), (64, 96, 7))
        self.assertEqual(self.client.url, "http://127.0.0.1:7860")
        body = self.client.bodies[0]
        self.assertEqual(base64.b64decode(body["image"]), png())
        self.assertEqual((body["multiplier"], body["steps"], body["cfg_scale"]), (-2.0, 50, 3.5))
        self.assertNotIn("model", body)                       # '' = 확장 기본값(최신 Tile & Repair)
        self.assertIsNone(self.host._tile_repair_job)

    def test_web_mode_bad_id_and_non_forge_backends_never_start_a_worker(self):
        with mock.patch.object(actions.threading, "Thread") as thread:
            self.host.web_mode = True
            self.run_action("tile_repair_run", {"requestId": "web", "image": data_url()})
            self.assertIn("로컬 앱", self.signal.items[-1]["error"])
            self.host.web_mode = False
            self.run_action("tile_repair_run", {"requestId": "../bad", "image": data_url()})
            self.assertFalse(self.signal.items[-1]["ok"])
            with mock.patch("backends.get_backend_type", return_value=BackendType.COMFYUI):
                self.run_action("tile_repair_options", {"requestId": "opt"})
            self.assertIn("Forge(WebUI)", self.signal.items[-1]["error"])
            thread.assert_not_called()
        self.assertEqual(self.client.bodies, [])

    def test_known_old_extension_is_refused_up_front(self):
        self.host.sam_extra_capabilities = build_capabilities({
            "/sdapi/v1/scripts": HttpResult(200, {"txt2img": [], "img2img": []}),
            "/sdapi/v1/script-info": HttpResult(200, []),
            EP_TILE_REPAIR: HttpResult(404),
        })
        self.run_action("tile_repair_run", {"requestId": "old", "image": data_url()})
        self.assertIn("업데이트", self.signal.items[-1]["error"])
        self.assertEqual(self.client.bodies, [])

    def test_bad_settings_or_source_answer_with_an_error(self):
        self.run_action("tile_repair_run", {"requestId": "r1", "image": data_url(), "settings": {"multiplier": 11}})
        self.assertIn("multiplier", self.wait()["error"])
        self.run_action("tile_repair_run", {"requestId": "r2", "image_path": "", "image": ""})
        self.assertIn("이미지", self.wait()["error"])
        self.assertEqual(self.client.bodies, [])
        self.assertIsNone(self.host._tile_repair_job)

    def test_route_errors_are_shown_without_paths(self):
        self.client.error = TileRepairError(r"Forge Tile & Repair 준비가 안 됐습니다: C:\secret\model.safetensors")
        self.run_action("tile_repair_run", {"requestId": "r", "image": data_url()})
        event = self.wait()
        self.assertFalse(event["ok"])
        self.assertIn("준비", event["error"])
        self.assertNotIn("secret", event["error"])
        self.assertEqual(list(Path(self.dir.name).glob("**/*.png")), [])

    def test_one_run_at_a_time_duplicate_delivery_and_cancel(self):
        gate = {"entered": threading.Event(), "release": threading.Event()}
        self.client.gate = gate
        self.run_action("tile_repair_run", {"requestId": "first", "image": data_url()})
        self.assertTrue(gate["entered"].wait(5))
        self.run_action("tile_repair_run", {"requestId": "first", "image": data_url()})   # 재전송
        self.assertEqual(self.signal.items, [])
        self.run_action("tile_repair_run", {"requestId": "second", "image": data_url()})
        self.assertIn("실행 중", self.wait()["error"])
        self.run_action("tile_repair_cancel", {"requestId": "first"})
        stop = self.wait()
        self.assertEqual((stop["action"], stop["ok"], stop["stopped"]), ("tile_repair_cancel", True, True))
        self.assertEqual(self.client.stops, 1)
        gate["release"].set()
        done = self.wait()
        self.assertEqual((done["requestId"], done["ok"], done["canceled"]), ("first", False, True))
        self.assertEqual(len(self.client.bodies), 1)
        self.assertEqual(list(Path(self.dir.name).glob("**/*.png")), [])   # 취소한 결과는 저장하지 않는다
        self.assertIsNone(self.host._tile_repair_job)

    def _wait_for(self, count):
        for _ in range(500):
            if len(self.signal.items) >= count:
                return self.signal.items
            threading.Event().wait(0.01)
        self.fail(f"expected {count} tile repair events, got {self.signal.items}")

    def test_cancel_before_the_request_reaches_forge_asks_again_until_it_stops(self):
        # 확장은 요청이 도착한 순간부터 멈춘다. 그 전(본문 직렬화·연결 중)엔 stopped=false 라 다시 묻는다.
        gate = {"entered": threading.Event(), "release": threading.Event()}
        self.client.gate = gate
        self.client.stop_answers = [False, False, True]
        self.run_action("tile_repair_run", {"requestId": "r", "image": data_url()})
        self.assertTrue(gate["entered"].wait(5))
        with mock.patch.object(actions, "STOP_RETRY_DELAY", 0.01):
            self.run_action("tile_repair_cancel", {"requestId": "r"})
            stop = self.wait()
        self.assertEqual((stop["action"], stop["stopped"], self.client.stops), ("tile_repair_cancel", True, 3))
        gate["release"].set()
        self.assertTrue(self.wait()["canceled"])

    def test_cancel_stops_asking_once_the_run_has_answered(self):
        # Forge 가 멈출 요청을 못 찾았는데 그사이 run 이 끝났다 — 더 묻지 않고 stopped=false 로 답한다.
        gate = {"entered": threading.Event(), "release": threading.Event()}
        self.client.gate = gate
        self.client.stop_answers = []

        def run_finishes_meanwhile():
            gate["release"].set()
            for _ in range(500):
                if self.host._tile_repair_job is None:
                    return
                threading.Event().wait(0.01)

        self.client.on_stop = run_finishes_meanwhile
        self.run_action("tile_repair_run", {"requestId": "r", "image": data_url()})
        self.assertTrue(gate["entered"].wait(5))
        with mock.patch.object(actions, "STOP_RETRY_DELAY", 5):     # 다시 물었다면 테스트가 5초 멈춘다
            self.run_action("tile_repair_cancel", {"requestId": "r"})
            events = {event["action"]: event for event in self._wait_for(2)}
        self.assertEqual(self.client.stops, 1)
        self.assertEqual((events["tile_repair_cancel"]["ok"], events["tile_repair_cancel"]["stopped"]), (True, False))
        self.assertTrue(events["tile_repair_run"]["canceled"])            # 취소했으니 결과는 버린다
        self.assertEqual(list(Path(self.dir.name).glob("**/*.png")), [])

    def _until(self, predicate):
        for _ in range(500):
            if predicate():
                return
            threading.Event().wait(0.01)
        self.fail("condition not reached")

    def _event(self, action, request_id):
        found = []
        self._until(lambda: found.extend(e for e in self.signal.items
                                         if (e["action"], e["requestId"]) == (action, request_id)) or found)
        return found[0]

    def test_a_late_retry_of_an_old_cancel_never_stops_the_next_run(self):
        # 이전 취소가 재시도를 기다리는 사이 그 run 이 끝나고 다음 run 이 Forge 에 닿았다. 이전 취소가 그때
        # 다시 물으면 stop 은 요청을 가리지 않으니 다음 run 이 멈춘다 — 더 묻지 않아야 한다.
        old_gate = {"entered": threading.Event(), "release": threading.Event()}
        new_gate = {"entered": threading.Event(), "release": threading.Event()}
        self.client.gate = old_gate
        self.client.stop_answers = [False]          # 첫 stop 땐 old 요청이 아직 Forge 에 닿기 전

        def old_run_ends_and_the_next_starts():
            threading.Event().wait(0.05)            # 취소 워커가 재시도를 기다리기 시작한 뒤
            old_gate["release"].set()
            self._until(lambda: self.host._tile_repair_job is None)
            self.client.gate = new_gate
            self.host._handle_tile_repair_action("tile_repair_run", {"requestId": "new", "image": data_url()})

        self.client.on_stop = lambda: threading.Thread(target=old_run_ends_and_the_next_starts, daemon=True).start()
        self.run_action("tile_repair_run", {"requestId": "old", "image": data_url()})
        self.assertTrue(old_gate["entered"].wait(5))
        with mock.patch.object(actions, "STOP_RETRY_DELAY", 1.0):
            self.run_action("tile_repair_cancel", {"requestId": "old"})
            self.assertTrue(new_gate["entered"].wait(5))
            self.assertTrue(self._event("tile_repair_run", "old")["canceled"])
            cancel = self._event("tile_repair_cancel", "old")   # 이전 취소 워커가 끝날 때까지
        new_gate["release"].set()
        new = self._event("tile_repair_run", "new")
        self.assertTrue(new["ok"], new)
        self.assertEqual((cancel["ok"], cancel["stopped"]), (True, False))
        self.assertEqual(self.client.stops, 1)
        self.assertNotIn(2, self.client.stopped_calls)

    def test_the_next_run_waits_for_an_old_stop_still_on_its_way(self):
        # 이전 취소의 stop 이 가는 중에 그 run 이 끝났다. 다음 run 을 곧바로 보내면 그 stop 이 다음 run 을 멈출 수
        # 있다 — 다음 run 은 이전 취소가 끝난 뒤에 보낸다.
        old_gate = {"entered": threading.Event(), "release": threading.Event()}
        new_gate = {"entered": threading.Event(), "release": threading.Event()}
        self.client.gate = old_gate
        self.client.stop_answers = [False]
        reached_forge_during_old_stop = []

        def old_run_ends_while_the_stop_travels():
            old_gate["release"].set()
            self._until(lambda: self.host._tile_repair_job is None)
            self.client.gate = new_gate
            self.host._handle_tile_repair_action("tile_repair_run", {"requestId": "new", "image": data_url()})
            reached_forge_during_old_stop.append(new_gate["entered"].wait(0.3))

        self.client.on_stop = old_run_ends_while_the_stop_travels
        self.run_action("tile_repair_run", {"requestId": "old", "image": data_url()})
        self.assertTrue(old_gate["entered"].wait(5))
        self.run_action("tile_repair_cancel", {"requestId": "old"})
        self.assertTrue(new_gate["entered"].wait(5))
        new_gate["release"].set()
        new = self._event("tile_repair_run", "new")
        self.assertEqual(reached_forge_during_old_stop, [False])
        self.assertTrue(new["ok"], new)
        self.assertNotIn(2, self.client.stopped_calls)
        self.assertEqual(self._event("tile_repair_cancel", "old")["stopped"], False)

    def test_cancelling_a_run_that_waits_for_an_old_stop_does_not_hold_it_up(self):
        # 다음 run 이 이전 취소를 기다리는 동안 그 run 도 취소했다 — 자기 취소 때문에 막히지 않고 canceled 로 답한다.
        old_gate = {"entered": threading.Event(), "release": threading.Event()}
        self.client.gate = old_gate
        self.client.stop_answers = []
        second_cancelled = threading.Event()

        def old_run_ends_then_the_next_is_cancelled_too():
            old_gate["release"].set()
            self._until(lambda: self.host._tile_repair_job is None)
            self.client.gate = None
            self.host._handle_tile_repair_action("tile_repair_run", {"requestId": "new", "image": data_url()})
            self._until(lambda: self.host._tile_repair_job is not None)
            self.host._handle_tile_repair_action("tile_repair_cancel", {"requestId": "new"})
            second_cancelled.set()

        self.client.on_stop = lambda: (self.client.stops == 1 and old_run_ends_then_the_next_is_cancelled_too())
        self.run_action("tile_repair_run", {"requestId": "old", "image": data_url()})
        self.assertTrue(old_gate["entered"].wait(5))
        with mock.patch.object(actions, "STOP_RETRY_DELAY", 5):     # 자기 취소를 기다렸다면 다음 run 답이 5초 넘게 늦는다
            self.run_action("tile_repair_cancel", {"requestId": "old"})
            self.assertTrue(second_cancelled.wait(5))
            new = self._event("tile_repair_run", "new")
            self._event("tile_repair_cancel", "new")
        self.assertTrue(new["canceled"], new)
        self.assertEqual(len(self.client.bodies), 1)          # 취소된 다음 run 은 Forge 에 보내지 않는다
        self._until(lambda: not self.host._tile_repair_stopping)

    def test_cancel_without_a_matching_run_says_nothing_was_stopped(self):
        self.run_action("tile_repair_cancel", {"requestId": "ghost"})
        event = self.wait()
        self.assertEqual((event["ok"], event["stopped"]), (True, False))
        self.assertEqual(self.client.stops, 0)

    def test_interrupted_run_is_reported_as_canceled(self):
        self.client.result = TileRepairResult(b"", "", None, None, None, "m", interrupted=True)
        self.run_action("tile_repair_run", {"requestId": "r", "image": data_url()})
        event = self.wait()
        self.assertTrue(event["canceled"])
        self.assertFalse(event["ok"])

    def test_options_answer_and_failure(self):
        self.run_action("tile_repair_options", {"requestId": "opt_1"})
        event = self.wait()
        self.assertEqual((event["action"], event["ok"]), ("tile_repair_options", True))
        self.assertEqual(event["options"]["models"], ["animaTileRepair_v20.safetensors"])
        self.client.error = TileRepairError("연결된 Forge 의 sam-extra 가 Tile & Repair 라우트를 지원하지 않습니다")
        self.run_action("tile_repair_options", {"requestId": "opt_2"})
        self.assertIn("지원하지", self.wait()["error"])

    def test_closed_host_stays_silent(self):
        self.host._shutdown_tile_repair()
        self.run_action("tile_repair_options", {"requestId": "late"})
        self.assertEqual(self.signal.items, [])

    def test_export_never_overwrites(self):
        a = Path(actions.export_tile_repair_png(b"one", self.dir.name))
        b = Path(actions.export_tile_repair_png(b"two", self.dir.name))
        self.assertNotEqual(a, b)
        self.assertEqual((a.read_bytes(), b.read_bytes()), (b"one", b"two"))
        self.assertEqual(a.parent.name, "tile_repair")


class OutputRootTests(unittest.TestCase):
    def test_output_root_comes_from_config(self):
        with mock.patch.dict(sys.modules, {"config": SimpleNamespace(OUTPUT_DIR="D:/out")}):
            self.assertEqual(TileRepairActionsMixin()._tile_repair_output_root(), "D:/out")


if __name__ == "__main__":
    unittest.main()
