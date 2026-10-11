"""연관 태그 — 캐시 상태·질의·표 만들기 조율(core/related_tags_store.py · core/related_tags_service.py).

작은 parquet 샤드로 실제 표를 만들고 질의한다. 스레드는 주입한 starter 로 그 자리에서 돌려 기다리지 않는다.
"""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from core import related_tags_store as store
from core.related_tags import QUERY_FIELDS, parse_query_payload
from core.related_tags_service import NO_DATA_MESSAGE, RelatedTagsService
from core.storage_paths import StoragePaths
from tests import _related_tags_fixture as fx


def request(request_id="rtq-1", limit=16, negative="", exclude="", **fields):
    payload_fields = {key: "" for key in QUERY_FIELDS}
    payload_fields.update(fields)
    return parse_query_payload({"requestId": request_id, "fields": payload_fields, "negative": negative,
                                "exclude": exclude, "limit": limit})


def names(event):
    return [item["tag"] for item in event["tags"]]


class _Harness(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self.data = fx.write_dataset(self.root)
        self.events: list[dict] = []
        self.logs: list[str] = []
        self.pending: list = []
        self.korean = fx.FakeKorean()
        self.storage = StoragePaths(self.root / "app")
        self.service = self.make_service()

    def tearDown(self):
        self._tmp.cleanup()

    def make_service(self, **kw):
        kw.setdefault("start_thread", lambda work, _name: work())
        kw.setdefault("tag_db", fx.FakeTagDatabase(self.data["catalog"]))
        kw.setdefault("log", self.logs.append)
        return RelatedTagsService(self.events.append, data_dir=self.data["data_dir"], storage=self.storage,
                                  korean=self.korean, idle_seconds=0, **kw)

    def build(self, shards=("g", "s")):
        self.assertTrue(self.service.start_build("rtb-1", shards))
        return self.events[-1]


class StoreTests(_Harness):
    def test_cache_paths_live_under_the_cache_folder(self):
        paths = store.cache_paths(self.storage)
        self.assertEqual(paths.root, (self.root / "app" / "cache" / "related_tags" / "v1").resolve())
        self.assertEqual(paths.shards["q"].name, "shard_q.npz")
        self.assertTrue(paths.root.is_dir())

    def test_estimate_uses_manifest_rows(self):
        manifest = {"artifacts": [
            {"path": "a.parquet", "kind": "search", "rating_shard": "g", "rows": 4_000_000},
            {"path": "b.parquet", "kind": "search", "rating_shard": "s", "rows": 2_000_000},
            {"path": "c.parquet", "kind": "event", "rating_shard": "q", "rows": 9_000_000}]}
        self.assertEqual(store.estimate_seconds(manifest, ["g", "s"]), 90)
        self.assertEqual(store.estimate_seconds(manifest, ["q", "e"]), 0)
        self.assertEqual(store.estimate_seconds(None, ["g"]), 0)

    def test_states_without_tables(self):
        paths = store.cache_paths(self.storage)
        manifest = self.data["manifest"]
        states = store.shard_states(manifest, self.data["data_dir"], paths, None)
        self.assertEqual(states, {"g": "missing", "s": "missing", "q": "missing", "e": "unavailable"})
        (self.data["data_dir"] / "danbooru_2026_07_q.parquet").write_bytes(b"x")    # 크기가 다르다
        self.assertEqual(store.shard_states(manifest, self.data["data_dir"], paths, "fp", ["q"]),
                         {"q": "unavailable"})

    def test_state_of_a_table_follows_its_meta(self):
        manifest = self.data["manifest"]
        artifact = store.shard_artifact(manifest, "g")[1]
        meta = {"format": 1, "shard": "g", "vocab": "fp", **store.artifact_identity(artifact)}
        data_dir = self.data["data_dir"]
        self.assertEqual(store.shard_state(meta, manifest, data_dir, "g", "fp"), "ready")
        self.assertEqual(store.shard_state(meta, manifest, data_dir, "g", "other"), "missing")
        self.assertEqual(store.shard_state({**meta, "format": 2}, manifest, data_dir, "g", "fp"), "missing")
        self.assertEqual(store.shard_state({**meta, "sha256": "old"}, manifest, data_dir, "g", "fp"), "stale")
        # 데이터가 사라져도 만든 표는 쓴다
        self.assertEqual(store.shard_state(meta, {"artifacts": []}, data_dir, "g", "fp"), "stale")


class QueryTests(_Harness):
    def test_missing_before_any_build_and_nothing_is_built_by_a_query(self):
        event = self.service.query(request(main="armor"), {"g", "s"})
        self.assertEqual(event["state"], "missing")
        self.assertEqual(event["missing"], ["g", "s"])
        self.assertEqual(event["ratings"], ["g", "s"])
        self.assertIsNone(event["build"])
        self.assertEqual(event["tags"], [])
        self.assertFalse(store.cache_paths(self.storage).vocab.exists())
        self.assertEqual(self.service.query(request(), {"e"})["state"], "unavailable")
        self.assertEqual(self.service.query(request(), None)["ratings"], ["g", "s"])

    def test_build_then_query_ranks_by_association(self):
        final = self.build()
        self.assertEqual(final, {"action": "related_tags_build", "requestId": "rtb-1", "state": "done",
                                 "shards": ["g", "s"]})
        building = [e for e in self.events if e["state"] == "building"]
        self.assertTrue(building)
        self.assertTrue(all(e["progress"]["count"] == 2 for e in building))
        paths = store.cache_paths(self.storage)
        self.assertTrue(paths.vocab.is_file() and paths.shards["g"].is_file() and paths.shards["s"].is_file())
        self.assertFalse(paths.shards["q"].exists())
        self.assertEqual(list(paths.root.glob("*.tmp")), [])
        self.assertTrue(any(str(paths.shards["g"].resolve()) in line for line in self.logs))

        event = self.service.query(request(main="1girl, armor", count=""), {"g", "s"})
        self.assertEqual(event["state"], "ready")
        self.assertEqual(event["context"], ["armor"])     # 1girl 은 흔한 태그라 기준에서 빠진다
        self.assertEqual(names(event), ["sword", "cape", "holding_sword", "shoulder_armor", "weapon"])
        self.assertEqual(event["tags"][0], {"tag": "sword", "ko": "검", "count": 200, "via": ["armor"]})
        self.assertEqual((event["missing"], event["stale"], event["unavailable"]), ([], [], []))
        self.assertEqual(len(self.service.query(request(main="armor", limit=2), {"g", "s"})["tags"]), 2)
        # G 만 고르면 S 표(cape)는 반영되지 않는다
        self.assertNotIn("cape", names(self.service.query(request(main="armor"), {"g"})))

    def test_present_implied_negative_negpip_and_exclude_rules_are_never_suggested(self):
        self.build()
        q = self.service.query
        self.assertEqual(names(q(request(main="holding sword"), {"g", "s"})), ["shoulder_armor", "armor"])
        event = q(request(main="armor, (sword:-1)", negative="weapon", exclude="_armor"), {"g", "s"})
        self.assertEqual(names(event), ["cape", "holding_sword"])
        self.assertEqual(names(q(request(main="nurse cap"), {"g", "s"})), ["smile"])     # hat 은 함의
        self.assertEqual(names(q(request(prefix="longhair"), {"g", "s"})), ["hat", "smile"])   # 별칭
        self.assertEqual(q(request(main="<lora:x:1>, __wild__"), {"g", "s"})["tags"], [])

    def test_character_and_copyright_fields_are_context_with_korean_labels(self):
        self.build()
        event = self.service.query(request(character="hatsune miku", copyright="vocaloid"), {"g", "s"})
        self.assertEqual(event["context"], ["hatsune_miku", "vocaloid"])
        self.assertEqual([(t["tag"], t["ko"], t["via"]) for t in event["tags"]], [
            ("aqua_hair", "청록 머리", ["hatsune_miku", "vocaloid"]),
            ("necktie", "", ["hatsune_miku", "vocaloid"]),
            ("twintails", "트윈테일", ["hatsune_miku", "vocaloid"]),
        ])

    def test_partial_stale_and_corrupt_tables(self):
        self.build(("g",))
        event = self.service.query(request(main="armor"), {"g", "q"})
        self.assertEqual((event["state"], event["missing"]), ("ready", ["q"]))
        self.assertEqual(names(event), ["holding_sword", "shoulder_armor", "sword", "weapon"])

        manifest = json.loads(json.dumps(self.data["manifest"]))
        manifest["artifacts"][0]["sha256"] = "0" * 64       # 새 릴리스
        fx.write_manifest(self.data["data_dir"], manifest)
        event = self.service.query(request(main="armor"), {"g"})
        self.assertEqual((event["state"], event["stale"]), ("ready", ["g"]))
        self.assertTrue(event["tags"])

        import numpy as np

        path = store.cache_paths(self.storage).shards["g"]
        with np.load(path, allow_pickle=False) as archive:
            arrays = {k: archive[k] for k in archive.files}
        arrays["indptr"] = arrays["indptr"][:3]           # 메타는 맞고 배열이 깨졌다
        with open(path, "wb") as handle:
            np.savez(handle, **arrays)
        service = self.make_service()
        event = service.query(request(main="armor"), {"g"})
        self.assertEqual((event["state"], event["missing"], event["stale"]), ("missing", ["g"], []))
        path.write_bytes(b"garbage")
        self.assertEqual(self.make_service().query(request(main="armor"), {"g"})["missing"], ["g"])

    def test_query_worker_answers_only_the_newest_request(self):
        started = []
        service = self.make_service(start_thread=lambda work, name: started.append((work, name)))
        for n in range(3):
            service.submit_query(request(f"rtq-{n}", main="armor"), {"g"})
        self.assertEqual(len(started), 1)
        self.assertEqual(started[0][1], "related-tags-query")
        started[0][0]()
        self.assertEqual([e["requestId"] for e in self.events], ["rtq-2"])
        service.submit_query(request("rtq-3"), {"g"})       # 워커가 끝났다 — 다시 띄운다
        self.assertEqual(len(started), 2)
        service.close()
        started[1][0]()
        service.submit_query(request("rtq-4"), {"g"})
        self.assertEqual(len(started), 2)
        self.assertEqual([e["requestId"] for e in self.events], ["rtq-2"])

    def test_query_errors_become_error_events(self):
        service = self.make_service()
        service.query = lambda *_a: (_ for _ in ()).throw(RuntimeError("C:\\secret\\path boom"))
        service.submit_query(request("rtq-9"), {"g"})
        self.assertEqual(self.events[-1]["state"], "error")
        self.assertEqual(self.events[-1]["requestId"], "rtq-9")
        self.assertNotIn("secret", self.events[-1]["error"])


class BuildTests(_Harness):
    def test_one_build_at_a_time_cancel_and_progress_in_queries(self):
        held = []
        service = self.make_service(start_thread=lambda work, _name: held.append(work))
        self.assertTrue(service.start_build("rtb-1", ("g", "s")))
        self.assertFalse(service.start_build("rtb-2", ("q",)))
        self.assertEqual(service.build_event()["requestId"], "rtb-1")
        event = service.query(request(main="armor"), {"g"})
        self.assertEqual(event["state"], "building")
        self.assertEqual((event["build"]["count"], event["build"]["requestId"]), (2, "rtb-1"))
        self.assertFalse(service.cancel("other"))
        self.assertTrue(service.cancel("rtb-1"))
        held[0]()
        self.assertEqual(self.events[-1]["state"], "canceled")
        self.assertIsNone(service.build_event())
        self.assertFalse(store.cache_paths(self.storage).shards["g"].exists())
        self.assertTrue(service.start_build("rtb-3", ("g",)))

    def test_cancel_mid_build_keeps_finished_shards(self):
        service = self.make_service(progress_interval=0)

        def emit(event):
            self.events.append(event)
            progress = event.get("progress") or {}
            if progress.get("shard") == "s":
                service.cancel()

        service._emit_raw = emit
        service.start_build("rtb-1", ("g", "s"))
        self.assertEqual(self.events[-1], {"action": "related_tags_build", "requestId": "rtb-1",
                                           "state": "canceled", "shards": ["g"]})
        paths = store.cache_paths(self.storage)
        self.assertTrue(paths.shards["g"].is_file())
        self.assertFalse(paths.shards["s"].exists())
        self.assertEqual(service.query(request(main="armor"), {"g", "s"})["missing"], ["s"])

    def test_progress_is_throttled(self):
        now = [0.0]
        service = self.make_service(clock=lambda: now[0])
        service.start_build("rtb-1", ("g", "s"))
        building = [e for e in self.events if e["state"] == "building"]
        # 시계가 멈춰 있으면 강제 알림(시작 · 샤드마다)만 나간다
        self.assertEqual([(e["progress"]["shard"], e["progress"]["index"]) for e in building],
                         [("g", 0), ("g", 0), ("s", 1)])
        self.assertEqual(building[-1]["progress"]["overall"], 0.5)

    def test_errors_and_missing_data(self):
        self.service.start_build("rtb-1", ("e",))
        self.assertEqual(self.events[-1]["state"], "error")
        self.assertEqual(self.events[-1]["error"], NO_DATA_MESSAGE)
        self.events.clear()
        service = self.make_service(tag_db=fx.FakeTagDatabase(self.data["catalog"], fail=True))
        service.start_build("rtb-2", ("g",))
        self.assertEqual(self.events[-1]["state"], "done")
        self.assertTrue(any("별칭" in line for line in self.logs))
        # 별칭 없이 만든 어휘 — longhair 는 모른다
        self.assertEqual(service.query(request(main="longhair"), {"g"})["context"], [])

    def test_a_console_that_cannot_print_does_not_fail_the_build(self):
        printed = []

        def cp949_console(message):
            if "—" in message or "표" in message:
                raise UnicodeEncodeError("cp949", message, 0, 1, "illegal multibyte sequence")
            printed.append(message)

        service = self.make_service(log=cp949_console)
        service.start_build("rtb-1", ("g",))
        self.assertEqual(self.events[-1]["state"], "done")
        self.assertTrue(any("RelatedTags" in line and "\\" in line for line in printed))   # ASCII 로 다시 찍었다

    def test_closed_service_emits_nothing_and_refuses_builds(self):
        self.service.close()
        self.assertFalse(self.service.start_build("rtb-1", ("g",)))
        self.service.submit_query(request(), {"g"})
        self.assertEqual(self.events, [])


if __name__ == "__main__":
    unittest.main()
