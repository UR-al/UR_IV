"""메모 교차 검토(R4)에서 나온 데이터 손실·되살아남 경로 — 네 가지 모두 수정 전에는 실패했다.

1. 두 편집기(웹 클라이언트 둘 · 탭 둘)가 같은 낡은 base 로 다른 글을 저장하면 둘째가 첫째의 충돌 사본을
   덮었다 — 사본은 (메모, base, 편집기) 별로 모은다. 편집기를 모르는 저장은 다른 글을 사본에 덮지 않는다.
2. Forge A 에 보내 dirty 가 풀린 메모를, 같은 id 의 다른 판을 가진 Forge B 와 처음 맞추면 로컬 편집이 사본 없이
   B 판으로 바뀌고 A 까지 덮였다 — 기준(base)이 없으면 내용이 다를 때 로컬도 바뀐 것으로 본다. 깨끗한 삭제
   표시는 빼는데, 서버에서 사라진 메모를 따라 지운 삭제 표시(ADOPT_DELETE)가 다른 기기가 다시 올린 편집을
   사본 없이 지웠기 때문이다(재검토에서 찾은 회귀).
3. 충돌 처리를 기다리는 동안 지운 메모가 서버의 옛 편집으로 되살아났다 — 더 새 삭제 표시는 지키고 보낸다.
4. 한 서버(A)에만 전달된 삭제 표시가 500개 한도 정리에서 지워져 B 에서 되살아났다 — 모든 서버가 본 삭제
   표시만 정리하고, 아직 못 본 서버가 있는 것은 한도에 세지 않는다.
"""
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from core.memo_store import (
    ADOPT_REMOTE, CONFLICT_COPY, CONFLICT_SUFFIX, MAX_MEMOS, NOOP, PUSH_DELETE, MemoStore,
    plan_memo_merge,
)
from core.memo_sync import run_memo_sync
from core.memo_sync_service import MemoSyncService, MemoSyncTarget
from tests.test_memo_store import Clock, Ids, memo, ts
from tests.test_memo_sync import FakeForge

A = "http://127.0.0.1:7860"
B = "http://127.0.0.1:7861"


class Case(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.path = Path(self.tmp.name) / "memos.json"
        self.clock = Clock()
        self.store = MemoStore(self.path, clock=self.clock, id_factory=Ids())

    def texts(self):
        return sorted(m["text"] for m in self.store.list_public())


class TwoEditorsSameStaleBaseTests(Case):
    def setUp(self):
        super().setUp()
        self.first = self.store.save("m1", "t", "v1").memo
        self.store.save("m1", "t", "newer elsewhere", self.first["updated_at"])

    def test_two_editors_each_keep_their_own_copy(self):
        x = self.store.save("m1", "t", "client X text", self.first["updated_at"], editor="X")
        y = self.store.save("m1", "t", "client Y text", self.first["updated_at"], editor="Y")
        self.assertNotEqual(x.memo["id"], y.memo["id"])
        self.assertEqual(self.texts(), ["client X text", "client Y text", "newer elsewhere"])

    def test_same_editor_continuation_and_resend_still_share_one_copy(self):
        one = self.store.save("m1", "t", "typing", self.first["updated_at"], editor="X")
        again = self.store.save("m1", "t", "typing", self.first["updated_at"], editor="X")
        more = self.store.save("m1", "t", "typing more", self.first["updated_at"], editor="X")
        self.assertEqual({one.memo["id"], again.memo["id"], more.memo["id"]}, {one.memo["id"]})
        self.assertFalse(again.changed)
        self.assertEqual(self.texts(), ["newer elsewhere", "typing more"])

    def test_unknown_editor_never_overwrites_a_copy_with_different_text(self):
        x = self.store.save("m1", "t", "client X text", self.first["updated_at"])
        y = self.store.save("m1", "t", "client Y text", self.first["updated_at"])
        self.assertNotEqual(x.memo["id"], y.memo["id"])
        self.assertEqual(self.texts(), ["client X text", "client Y text", "newer elsewhere"])
        # 같은 글을 다시 보내면(확인이 늦은 재전송) 사본을 또 만들지 않는다
        self.assertFalse(self.store.save("m1", "t", "client Y text", self.first["updated_at"]).changed)

    def test_service_passes_the_editor_and_reports_which_memo_the_save_landed_in(self):
        states = []
        service = MemoSyncService(self.store, target_provider=MemoSyncTarget.local, emit=states.append,
                                  timer_factory=lambda delay, fn: _NoTimer(), start_thread=lambda fn: fn())
        base = self.first["updated_at"]
        service.handle_save({"id": "m1", "title": "t", "text": "X", "base_updated_at": base,
                             "editor": "ed-x", "request": "rq-1"})
        saved_x = states[-1]["saved"]
        service.handle_save({"id": "m1", "title": "t", "text": "Y", "base_updated_at": base,
                             "editor": "ed-y", "request": "rq-2"})
        saved_y = states[-1]["saved"]
        self.assertEqual(saved_x["request"], "rq-1")
        self.assertEqual(saved_x["conflict_of"], "m1")
        self.assertEqual(saved_y["request"], "rq-2")
        self.assertNotEqual(saved_x["id"], saved_y["id"])
        self.assertEqual(self.store.get(saved_x["id"])["text"], "X")
        self.assertEqual(self.store.get(saved_y["id"])["text"], "Y")
        # 제자리에 저장된 것은 conflict_of 가 비어 있다
        service.handle_save({"id": "m1", "title": "t", "text": "in place",
                             "base_updated_at": self.store.get("m1")["updated_at"],
                             "editor": "ed-x", "request": "rq-3"})
        self.assertEqual(states[-1]["saved"], {"request": "rq-3", "id": "m1", "conflict_of": ""})
        service.handle_list()
        self.assertNotIn("saved", states[-1])


class _NoTimer:
    def start(self):
        pass

    def cancel(self):
        pass


class FirstContactWithAnotherServerTests(Case):
    def test_plan_without_base_treats_differing_clean_local_as_changed(self):
        mine = memo("a", text="E1", updated=30)                  # 다른 Forge 에 보내 깨끗해졌다
        self.assertEqual(plan_memo_merge(mine, memo("a", text="B version", updated=20), None), CONFLICT_COPY)
        self.assertEqual(plan_memo_merge(mine, memo("a", text="E1", updated=20), None), ADOPT_REMOTE)
        # 살아 있는 로컬과 서버 삭제 표시는 더 새것이 이긴다(사본 없이 덮지 않는다)
        self.assertEqual(plan_memo_merge(mine, memo("a", updated=10, deleted=True), None), "push")
        self.assertEqual(plan_memo_merge(memo("a", text="E1", updated=5), memo("a", updated=10, deleted=True),
                                         None), ADOPT_REMOTE)
        # 깨끗한 로컬 삭제 표시는 바뀐 것으로 보지 않는다 — 기준 없는 서버의 살아 있는 판을 들인다. 서버에서
        # 사라진 메모를 따라 지운 삭제 표시(ADOPT_DELETE)는 따라 지운 시각이 찍혀, 다른 기기가 다시 올린 그 전
        # 편집보다 늘 새것이다 — 이기게 두면 그 편집이 사본 없이 모든 곳에서 지워진다.
        tomb = memo("a", updated=40, deleted=True)
        self.assertEqual(plan_memo_merge(tomb, memo("a", text="old live", updated=20), None), ADOPT_REMOTE)
        self.assertEqual(plan_memo_merge(memo("a", updated=40, deleted=True, dirty=True),
                                         memo("a", text="old live", updated=20), None), PUSH_DELETE)
        self.assertEqual(plan_memo_merge(memo("a", updated=40, deleted=True),
                                         memo("a", updated=20, deleted=True), None), ADOPT_REMOTE)
        self.assertNotEqual(plan_memo_merge(mine, memo("a", text="E1", updated=30), None), NOOP)

    def test_edit_sent_to_a_survives_first_sync_with_b_holding_another_version(self):
        a, b = FakeForge(self.clock), FakeForge(self.clock)
        v0 = self.store.save("m1", "t", "v0").memo
        run_memo_sync(self.store, a, A)
        b.memos["m1"] = dict(a.memos["m1"], text="B version", updated_at=self.clock())
        self.store.save("m1", "t", "E1", v0["updated_at"])
        run_memo_sync(self.store, a, A)
        self.assertFalse(self.store.get("m1")["dirty"])
        report = run_memo_sync(self.store, b, B)
        self.assertEqual(report.conflicts, ["m1"])
        self.assertEqual(self.texts(), ["B version", "E1"])
        run_memo_sync(self.store, a, A)
        self.assertEqual(sorted(m["text"] for m in a.memos.values() if not m["deleted"]), ["B version", "E1"])
        self.assertEqual(sorted(m["text"] for m in b.memos.values() if not m["deleted"]), ["B version", "E1"])


def _fill_with_tombstones(server, count, prefix="old"):
    for index in range(count):
        memo_id = f"{prefix}-{index}"
        server.memos[memo_id] = {"id": memo_id, "title": "", "text": "", "created_at": ts(1),
                                 "updated_at": ts(2), "deleted": True}


class ReuploadAfterAdoptedDeleteTests(Case):
    def test_offline_edit_reuploaded_after_a_pruned_delete_survives_on_every_device(self):
        server = FakeForge(self.clock)
        other = MemoStore(Path(self.tmp.name) / "other.json", clock=self.clock, id_factory=Ids())
        self.store.save("m1", "t", "v0")
        run_memo_sync(self.store, server, A)
        run_memo_sync(other, server, A)
        other.save("m1", "t", "offline edit", other.get("m1")["updated_at"])   # 다른 기기가 오프라인에서 고친다
        # 그사이 다른 곳에서 지웠고, 가득 찬 서버가 그 삭제 표시를 정리했다
        del server.memos["m1"]
        _fill_with_tombstones(server, MAX_MEMOS)
        server.revision += MAX_MEMOS
        run_memo_sync(self.store, server, A)
        self.assertTrue(self.store.get("m1")["deleted"])                    # 따라 지운다(ADOPT_DELETE)
        run_memo_sync(other, server, A)                                     # 서버가 모르는 메모 — 다시 올린다
        self.assertEqual(server.memos["m1"]["text"], "offline edit")

        run_memo_sync(self.store, server, A)                                # 기준을 지웠으니 새 메모로 들인다
        self.assertFalse(server.memos["m1"]["deleted"])
        self.assertEqual(server.memos["m1"]["text"], "offline edit")
        self.assertEqual(self.store.get("m1")["text"], "offline edit")
        self.assertFalse(self.store.get("m1")["deleted"])
        run_memo_sync(other, server, A)
        self.assertEqual(other.get("m1")["text"], "offline edit")
        self.assertFalse(other.get("m1")["deleted"])


class DeleteDuringConflictTests(Case):
    def test_memo_deleted_while_its_conflict_waits_stays_deleted_and_reaches_the_server(self):
        server = FakeForge(self.clock)
        m0 = self.store.save("m0", "t", "zero").memo
        v0 = self.store.save("m1", "t", "v0").memo
        run_memo_sync(self.store, server, A)
        server.edit("m1", "REMOTE")
        self.store.save("m1", "t", "LOCAL", v0["updated_at"])
        self.store.save("m0", "t", "zero edited", m0["updated_at"])   # 이 PUT 이 m1 충돌 처리보다 먼저

        def delete_m1_during_the_m0_put(memo_id):
            if memo_id == "m0":
                self.store.delete("m1")
        server.before_put = delete_m1_during_the_m0_put
        run_memo_sync(self.store, server, A)
        self.assertTrue(self.store.get("m1")["deleted"])
        self.assertTrue(server.memos["m1"]["deleted"])
        self.assertNotIn("REMOTE", self.texts())
        self.assertNotIn("LOCAL", self.texts())                  # 지운 것이라 사본도 만들지 않는다
        run_memo_sync(self.store, server, A)                     # 다음 동기화도 되살리지 않는다
        self.assertTrue(self.store.get("m1")["deleted"])
        self.assertTrue(server.memos["m1"]["deleted"])

    def test_remote_edit_newer_than_the_delete_still_wins(self):
        server = FakeForge(self.clock)
        v0 = self.store.save("m1", "t", "v0").memo
        run_memo_sync(self.store, server, A)
        self.store.save("m1", "t", "LOCAL", v0["updated_at"])
        self.store.delete("m1")
        self.clock.back(-100)                                     # 서버 편집이 삭제보다 뒤
        server.edit("m1", "REMOTE later")
        self.store.resolve_conflict(A, server.memos["m1"])
        self.assertEqual(self.store.get("m1")["text"], "REMOTE later")
        self.assertFalse(self.store.get("m1")["deleted"])


def _server_memo(memo_id, **kw):
    m = memo(memo_id, **kw)
    m.pop("dirty")
    return m


class TombstonePruningTests(Case):
    def seed(self, memos, servers):
        self.path.write_text(json.dumps({"schema_version": 1, "memos": memos,
                                         "sync": {"servers": servers}}), encoding="utf-8")
        self.store = MemoStore(self.path, clock=self.clock, id_factory=Ids())

    def test_tombstone_not_yet_seen_by_b_is_kept_and_later_delivered(self):
        live = [memo(f"live-{i}", updated=100 + i) for i in range(MAX_MEMOS - 1)]
        self.seed(live + [memo("t1", deleted=True, updated=50)],
                  {A: {"base": {"t1": ts(50)}}, B: {"base": {"t1": ts(5)}}})
        b = FakeForge(self.clock)
        b.memos["t1"] = _server_memo("t1", text="deleted content", updated=5)
        self.store.save("fresh", "f", "x")                        # 한도 — 여기서 t1 을 지우면 안 된다
        self.assertTrue(self.store.get("t1")["deleted"])
        run_memo_sync(self.store, b, B)
        self.assertFalse(any(m["id"] == "t1" for m in self.store.list_public()))
        self.assertTrue(b.memos["t1"]["deleted"])

    def test_tombstone_delivered_to_both_servers_becomes_prunable(self):
        a, b = FakeForge(self.clock), FakeForge(self.clock)
        self.store.save("t1", "t", "doomed")
        run_memo_sync(self.store, a, A)
        run_memo_sync(self.store, b, B)
        self.store.delete("t1")
        run_memo_sync(self.store, a, A)
        run_memo_sync(self.store, b, B)                           # 두 서버의 삭제 시각은 서로 다르다
        self.assertTrue(a.memos["t1"]["deleted"] and b.memos["t1"]["deleted"])
        memos = self.store.all_memos() + [memo(f"live-{i}", updated=100 + i) for i in range(MAX_MEMOS - 1)]
        data = json.loads(self.path.read_text(encoding="utf-8"))
        data["memos"] = memos
        self.path.write_text(json.dumps(data), encoding="utf-8")
        self.store = MemoStore(self.path, clock=self.clock, id_factory=Ids())
        self.store.save("fresh", "f", "x")
        self.assertIsNone(self.store.get("t1"))                   # 모두 본 삭제 표시는 정리한다
        self.assertEqual(len(self.store.all_memos()), MAX_MEMOS)

    def test_tombstone_for_a_server_that_lost_the_memo_keeps_its_base_so_a_stale_reupload_is_deleted(self):
        # 서버가 메모를 잃은 뒤 지운 삭제 표시는 그 서버 기준(살아 있던 판)을 지니고 붙잡힌다(held). 그 기준이
        # 다른 기기가 다시 올린 옛 판을 '안 바뀐 서버'로 보게 해 삭제가 이긴다 — 기준을 지우면(또는 정리하면)
        # 모르는 메모로 들여 지운 메모가 모든 기기에서 되살아난다.
        server = FakeForge(self.clock)
        other = MemoStore(Path(self.tmp.name) / "other.json", clock=self.clock, id_factory=Ids())
        self.store.save("m1", "t", "v0")
        run_memo_sync(self.store, server, A)
        run_memo_sync(other, server, A)
        del server.memos["m1"]                                              # 서버가 잃었다(가득 차지 않음)
        self.store.delete("m1")
        run_memo_sync(self.store, server, A)                                # MARK_CLEAN — 기준은 그대로
        self.assertIn("m1", self.store.sync_bases(A))
        run_memo_sync(other, server, A)                                     # 옛 판을 다시 올린다
        self.assertFalse(server.memos["m1"]["deleted"])
        run_memo_sync(self.store, server, A)
        run_memo_sync(other, server, A)
        self.assertTrue(server.memos["m1"]["deleted"])
        self.assertTrue(self.store.get("m1")["deleted"])
        self.assertTrue(other.get("m1")["deleted"])

    def test_room_check_skips_the_per_server_scan_while_there_is_room(self):
        # held 는 넘침을 줄이기만 한다 — 칸이 남으면(대부분의 들이기·저장) 삭제 표시마다 서버 기록을 훑지 않는다
        self.seed([memo(f"t{i}", deleted=True, updated=50 + i) for i in range(20)],
                  {A: {"base": {f"t{i}": ts(1) for i in range(20)}}})
        server = FakeForge(self.clock)
        for i in range(5):
            server.edit(f"r{i}", f"remote {i}")
        with mock.patch.object(MemoStore, "_tomb_unsettled",
                               side_effect=MemoStore._tomb_unsettled) as scan:
            report = run_memo_sync(self.store, server, A)
            self.store.save("fresh", "f", "x")
        self.assertEqual(report.pulled, 5)
        self.assertEqual(scan.call_count, 0)
        self.assertEqual(len(self.store.all_memos()), 26)

    def test_held_tombstones_do_not_block_new_memos(self):
        live = [memo(f"live-{i}", updated=100 + i) for i in range(MAX_MEMOS - 1)]
        held = [memo(f"t{i}", deleted=True, updated=50 + i) for i in range(3)]
        self.seed(live + held, {B: {"base": {f"t{i}": ts(1) for i in range(3)}}})
        self.store.save("fresh", "f", "x")
        ids = {m["id"] for m in self.store.all_memos()}
        self.assertTrue({"fresh", "t0", "t1", "t2"} <= ids)


if __name__ == "__main__":
    unittest.main()
