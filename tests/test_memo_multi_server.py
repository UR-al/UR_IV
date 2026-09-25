"""한 저장소를 두 Forge(설치 A·B, 또는 WebUI 주소를 바꿔 가며)와 맞출 때.

'로컬이 바뀌었나'는 서버마다 따진다(로컬 updated_at vs 그 서버의 base). 하나뿐인 dirty 로 따지면 A 에
보낸 편집 · 삭제는 B 에 영영 안 가고, B 의 편집이 로컬 편집을 사본 없이 덮는다.
"""
import tempfile
import unittest
from pathlib import Path

from core.memo_store import ADOPT_DELETE, ADOPT_REMOTE, NOOP, PUSH, MemoStore, plan_memo_merge
from core.memo_sync import run_memo_sync
from tests.test_memo_store import Clock, Ids, memo, ts
from tests.test_memo_sync import FakeForge

A = "http://127.0.0.1:7860"
B = "http://127.0.0.1:7861"


class PerServerPlanTests(unittest.TestCase):
    def test_clean_local_that_moved_past_this_servers_base_is_pushed(self):
        # 다른 서버에서 들였거나 그리로 보낸 판(dirty 아님)이 이 서버의 base 보다 새것
        self.assertEqual(plan_memo_merge(memo("a", text="E1", updated=30), memo("a", updated=10), ts(10)), PUSH)
        # 이 서버가 그 뒤 따로 바뀌었으면 둘 다 바뀐 것 — 충돌 규칙으로
        self.assertNotEqual(plan_memo_merge(memo("a", text="E1", updated=30),
                                            memo("a", text="B", updated=40), ts(10)), ADOPT_REMOTE)
        # 이 서버와 같은 판이면 서버 쪽 변경만 들인다
        self.assertEqual(plan_memo_merge(memo("a", updated=10), memo("a", text="B", updated=40), ts(10)),
                         ADOPT_REMOTE)
        self.assertEqual(plan_memo_merge(memo("a", updated=10), memo("a", updated=10), ts(10)), NOOP)

    def test_pruned_delete_is_adopted_only_when_local_matches_this_server(self):
        self.assertEqual(plan_memo_merge(memo("a", updated=10), None, ts(10), remote_may_prune=True),
                         ADOPT_DELETE)
        self.assertEqual(plan_memo_merge(memo("a", updated=30), None, ts(10), remote_may_prune=True), PUSH)


class TwoServerSyncTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.clock = Clock()
        self.a, self.b = FakeForge(self.clock), FakeForge(self.clock)
        self.store = MemoStore(Path(self.tmp.name) / "memos.json", clock=self.clock, id_factory=Ids())
        v0 = self.store.save("m1", "t", "v0").memo
        self.sync(self.a, A)
        self.sync(self.b, B)
        self.v0 = v0

    def sync(self, server, key):
        return run_memo_sync(self.store, server, key)

    def text(self, memo_id="m1"):
        return self.store.get(memo_id)["text"]

    def test_edit_synced_to_a_reaches_b(self):
        self.store.save("m1", "t", "E1", self.v0["updated_at"])
        self.sync(self.a, A)
        self.assertFalse(self.store.get("m1")["dirty"])
        report = self.sync(self.b, B)
        self.assertEqual(report.pushed, 1)
        self.assertEqual(self.b.memos["m1"]["text"], "E1")
        self.assertEqual(report.conflicts, [])

    def test_edit_made_on_b_travels_to_a_without_losing_anything(self):
        self.store.save("m1", "t", "E1", self.v0["updated_at"])
        self.sync(self.a, A)
        self.sync(self.b, B)
        self.b.edit("m1", "B-side edit")
        self.sync(self.b, B)
        self.assertEqual(self.text(), "B-side edit")
        self.sync(self.a, A)
        self.assertEqual(self.a.memos["m1"]["text"], "B-side edit")
        self.assertEqual({m["text"] for m in self.store.list_public()}, {"B-side edit"})

    def test_b_edit_against_an_unsent_local_edit_keeps_both(self):
        self.store.save("m1", "t", "E1", self.v0["updated_at"])
        self.sync(self.a, A)                    # E1 은 A 에만 갔다
        self.b.edit("m1", "B-side edit")        # B 는 v0 위에서 따로 고쳤다
        report = self.sync(self.b, B)
        self.assertEqual(report.conflicts, ["m1"])
        self.assertEqual(sorted(m["text"] for m in self.store.list_public()), ["B-side edit", "E1"])
        self.sync(self.a, A)
        self.assertEqual(sorted(m["text"] for m in self.a.memos.values() if not m["deleted"]),
                         ["B-side edit", "E1"])

    def test_delete_reaches_both_servers(self):
        self.store.delete("m1")
        self.sync(self.a, A)
        self.sync(self.b, B)
        self.assertTrue(self.a.memos["m1"]["deleted"])
        self.assertTrue(self.b.memos["m1"]["deleted"])

    def test_quiet_once_converged(self):
        self.store.save("m1", "t", "E1", self.v0["updated_at"])
        for _ in range(2):
            self.sync(self.a, A)
            self.sync(self.b, B)
        writes = (list(self.a.writes), list(self.b.writes))
        self.sync(self.a, A)
        self.sync(self.b, B)
        self.assertEqual((self.a.writes, self.b.writes), writes)


if __name__ == "__main__":
    unittest.main()
