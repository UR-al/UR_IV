"""메모 저장의 낡은 base — 편집 중인 화면이 옛 판에 기대 저장할 때.

- 낡았다 = 저장본의 updated_at(또는 내용이 같은 옛 시각)이 아니다. 저장본이 더 새것인지는 따지지 않는다 —
  동기화가 충돌로 서버 판을 id 에 두면 그 판이 화면의 base(로컬 편집)보다 옛 시각일 수 있고, 덮으면
  서버 판이 앱과 Forge 양쪽에서 사라진다.
- 같은 (메모, base, 편집기) 로 다시 오는 저장(확인이 늦어 다시 보냄 · 이어 친 글)은 그 저장이 만든 사본 하나에
  모은다. 편집기를 모르면 같은 글의 재전송만 모으고 다른 글은 새 사본으로(tests/test_memo_data_loss_guards.py).
"""
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from core.memo_store import CONFLICT_SUFFIX, MemoStore, is_newer
from core.memo_sync import run_memo_sync
from core.memo_sync_service import CONFLICT_MESSAGE, MemoSyncService, MemoSyncTarget
from tests.test_memo_store import SERVER, Clock, Ids, memo, ts
from tests.test_memo_sync import FakeForge


class StaleCase(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.clock = Clock()
        self.server = FakeForge(self.clock)
        self.store = MemoStore(Path(self.tmp.name) / "memos.json", clock=self.clock, id_factory=Ids())

    def copies(self):
        return [m for m in self.store.list_public() if m["title"].endswith(CONFLICT_SUFFIX)]

    def sync(self):
        return run_memo_sync(self.store, self.server, SERVER)


class OlderRemoteKeptTests(StaleCase):
    def test_save_on_top_of_a_replaced_local_edit_is_a_conflict_even_if_remote_is_older(self):
        first = self.store.save("m1", "t", "v1").memo
        self.sync()
        self.server.edit("m1", "REMOTE R")                        # Forge 편집이 먼저(더 옛 시각)
        local = self.store.save("m1", "t", "local L2", first["updated_at"]).memo
        self.assertTrue(is_newer(local["updated_at"], self.server.memos["m1"]["updated_at"]))
        self.assertEqual(self.sync().conflicts, ["m1"])          # R 이 id 를, L2 는 사본으로

        # 열려 있던 편집기가 base=L2 로 이어 저장한다
        result = self.store.save("m1", "t", "local L2 + more", local["updated_at"])
        self.assertTrue(result.conflict)
        self.assertEqual(result.conflict_of, "m1")
        self.assertEqual(self.store.get("m1")["text"], "REMOTE R")
        self.sync()
        self.assertEqual(self.server.memos["m1"]["text"], "REMOTE R")
        server_texts = sorted(m["text"] for m in self.server.memos.values())
        self.assertEqual(server_texts, ["REMOTE R", "local L2", "local L2 + more"])
        self.assertEqual(sorted(m["text"] for m in self.store.list_public()), server_texts)

    def test_adopted_older_version_is_not_overwritten_by_a_newer_base(self):
        # 로컬은 ts(30) 판을 보고 있었는데, 동기화가 더 옛 시각 ts(20) 의 다른 판을 들였다(다른 Forge 등)
        self.store.adopt_remote(SERVER, memo("m", text="mine", updated=30), expected=None)
        self.store.adopt_remote("http://other", memo("m", text="theirs", updated=20),
                                expected=self.store.get("m"))
        result = self.store.save("m", "t", "typed on mine", ts(30))
        self.assertTrue(result.conflict)
        self.assertEqual(self.store.get("m")["text"], "theirs")

    def test_base_in_z_notation_is_the_same_version(self):
        saved = self.store.save("m", "t", "v1").memo
        z_base = saved["updated_at"].replace("+00:00", "Z")
        result = self.store.save("m", "t", "v2", z_base)
        self.assertFalse(result.conflict)
        self.assertEqual(self.store.get("m")["text"], "v2")


class OneCopyPerStaleSaveTests(StaleCase):
    def setUp(self):
        super().setUp()
        self.first = self.store.save("m1", "t", "v1").memo
        self.store.save("m1", "t", "forge newer", self.first["updated_at"])   # 다른 곳이 먼저 바꿨다

    def test_resending_the_same_stale_save_keeps_one_copy(self):
        results = [self.store.save("m1", "t", "typing", self.first["updated_at"]) for _ in range(3)]
        self.assertEqual(len(self.copies()), 1)
        self.assertEqual([r.changed for r in results], [True, False, False])
        self.assertTrue(all(r.conflict for r in results))
        self.assertEqual({r.memo["id"] for r in results}, {"copy-1"})

    def test_more_typing_with_the_same_stale_base_updates_that_copy(self):
        one = self.store.save("m1", "t", "typing", self.first["updated_at"], editor="ed-1")
        two = self.store.save("m1", "t", "typing more", self.first["updated_at"].replace("+00:00", "Z"),
                              editor="ed-1")
        self.assertEqual(two.memo["id"], one.memo["id"])
        self.assertTrue(two.changed)
        self.assertTrue(is_newer(two.memo["updated_at"], one.memo["updated_at"]))
        self.assertEqual([m["text"] for m in self.copies()], ["typing more"])
        self.assertEqual(self.store.get("m1")["text"], "forge newer")

    def test_copy_edited_on_its_own_is_not_overwritten_by_a_late_stale_save(self):
        copy_memo = self.store.save("m1", "t", "typing", self.first["updated_at"], editor="ed-1").memo
        # 화면이 사본으로 옮겨 가 이어 쓴다
        self.store.save(copy_memo["id"], copy_memo["title"], "typing, then the copy", copy_memo["updated_at"],
                        editor="ed-1")
        late = self.store.save("m1", "t", "typing (late resend)", self.first["updated_at"], editor="ed-1")
        self.assertTrue(late.conflict)
        self.assertNotEqual(late.memo["id"], copy_memo["id"])
        self.assertEqual(self.store.get(copy_memo["id"])["text"], "typing, then the copy")

    def test_a_different_stale_base_gets_its_own_copy(self):
        second = self.store.get("m1")["updated_at"]
        self.store.save("m1", "t", "third", second)
        self.store.save("m1", "t", "from first", self.first["updated_at"])
        self.store.save("m1", "t", "from second", second)
        self.assertEqual(sorted(m["text"] for m in self.copies()), ["from first", "from second"])

    def test_forgotten_after_reload(self):
        self.store.save("m1", "t", "typing", self.first["updated_at"])
        self.store.reload()
        self.store.save("m1", "t", "typing more", self.first["updated_at"])
        self.assertEqual(len(self.copies()), 2)   # 기억은 메모리에만 — 글은 잃지 않는다


class ServiceNotifiesOnceTests(StaleCase):
    def test_resent_stale_save_does_not_toast_again(self):
        notes, states = [], []
        service = MemoSyncService(
            self.store, target_provider=MemoSyncTarget.local, emit=states.append,
            notify=lambda level, msg: notes.append((level, msg)),
            timer_factory=lambda delay, fn: mock.Mock(), start_thread=lambda fn: fn())
        first = self.store.save("m1", "t", "v1").memo
        self.store.save("m1", "t", "v2", first["updated_at"])
        stale = {"id": "m1", "title": "t", "text": "mine", "base_updated_at": first["updated_at"]}
        service.handle_save(stale)
        service.handle_save(dict(stale))
        self.assertEqual(notes.count(("info", CONFLICT_MESSAGE)), 1)
        self.assertEqual(sorted(m["text"] for m in states[-1]["memos"]), ["mine", "v2"])


if __name__ == "__main__":
    unittest.main()
