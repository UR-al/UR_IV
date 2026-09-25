"""core.memo_store — 로컬 메모 저장소, dirty 표시, 공유 메모 계약의 병합 규칙."""
import json
import os
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest import mock

from core import memo_store as ms
from core.memo_store import (
    ADOPT_REMOTE, CONFLICT_COPY, CONFLICT_SUFFIX, MARK_CLEAN, MAX_MEMOS, MAX_TEXT_LENGTH,
    MAX_TITLE_LENGTH, NOOP, PUSH, PUSH_DELETE, MemoStore, MemoStoreUnavailableError,
    MemoValidationError, conflict_title, is_newer, parse_timestamp, plan_memo_merge,
    timestamps_equal,
)

SERVER = "http://127.0.0.1:7860"


class Clock:
    """호출마다 1초씩 가는 시계(ISO8601 UTC). ``back`` 으로 되돌릴 수도 있다."""

    def __init__(self, start="2026-09-25T00:00:00+00:00"):
        self.now = datetime.fromisoformat(start)

    def __call__(self):
        self.now += timedelta(seconds=1)
        return self.now.isoformat()

    def back(self, seconds):
        self.now -= timedelta(seconds=seconds)


def ts(second):
    return (datetime(2026, 9, 25, tzinfo=timezone.utc) + timedelta(seconds=second)).isoformat()


def memo(memo_id, *, title="t", text="x", updated=10, deleted=False, dirty=False, created=1):
    return {"id": memo_id, "title": title, "text": "" if deleted else text,
            "created_at": ts(created), "updated_at": ts(updated), "deleted": deleted, "dirty": dirty}


class Ids:
    def __init__(self):
        self.n = 0

    def __call__(self):
        self.n += 1
        return f"copy-{self.n}"


class StoreCase(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.path = Path(self.tmp.name) / "memos.json"
        self.clock = Clock()
        self.store = self.new_store()

    def new_store(self):
        return MemoStore(self.path, clock=self.clock, id_factory=Ids())

    def on_disk(self):
        return json.loads(self.path.read_text(encoding="utf-8"))


class TimestampTests(unittest.TestCase):
    def test_z_suffix_and_offset_are_the_same_instant(self):
        self.assertTrue(timestamps_equal("2026-09-25T01:00:00Z", "2026-09-25T01:00:00+00:00"))
        self.assertTrue(timestamps_equal("2026-09-25T10:00:00+09:00", "2026-09-25T01:00:00+00:00"))
        self.assertFalse(timestamps_equal("2026-09-25T01:00:00Z", "2026-09-25T01:00:01Z"))
        self.assertTrue(timestamps_equal(None, ""))
        self.assertFalse(timestamps_equal("garbage", "2026-09-25T01:00:00Z"))

    def test_unparseable_is_oldest_and_naive_is_utc(self):
        self.assertTrue(is_newer("2020-01-01T00:00:00Z", "garbage"))
        self.assertFalse(is_newer("garbage", "2020-01-01T00:00:00Z"))
        self.assertEqual(parse_timestamp("2026-09-25T01:00:00"), parse_timestamp("2026-09-25T01:00:00Z"))
        self.assertIsNone(parse_timestamp(42))


class SaveDeleteTests(StoreCase):
    def test_new_memo_is_dirty_persisted_utf8_without_bom(self):
        result = self.store.save("memo-1", "제목", "한국어 본문 — 日本語", None)
        self.assertTrue(result.changed)
        self.assertFalse(result.conflict)
        self.assertEqual(result.memo["title"], "제목")
        self.assertEqual(set(result.memo), {"id", "title", "text", "created_at", "updated_at"})
        raw = self.path.read_bytes()
        self.assertFalse(raw.startswith(b"\xef\xbb\xbf"))
        data = self.on_disk()
        self.assertEqual(data["schema_version"], 1)
        self.assertEqual(data["memos"][0]["dirty"], True)
        self.assertEqual(data["memos"][0]["deleted"], False)
        reloaded = self.new_store()
        self.assertEqual(reloaded.list_public()[0]["text"], "한국어 본문 — 日本語")
        self.assertTrue(reloaded.has_dirty())

    def test_same_content_does_not_bump_or_dirty(self):
        first = self.store.save("m", "a", "b").memo
        self.store.mark_clean("m", expected=self.store.get("m"))
        again = self.store.save("m", "a", "b", first["updated_at"])
        self.assertFalse(again.changed)
        self.assertEqual(again.memo["updated_at"], first["updated_at"])
        self.assertFalse(self.store.has_dirty())

    def test_edit_bumps_updated_at_and_marks_dirty(self):
        first = self.store.save("m", "a", "b").memo
        self.store.mark_clean("m", expected=self.store.get("m"))
        edited = self.store.save("m", "a", "c", first["updated_at"])
        self.assertTrue(edited.changed)
        self.assertTrue(is_newer(edited.memo["updated_at"], first["updated_at"]))
        self.assertTrue(self.store.get("m")["dirty"])

    def test_clock_going_backwards_still_moves_forward(self):
        first = self.store.save("m", "a", "b").memo
        self.clock.back(3600)
        second = self.store.save("m", "a", "c", first["updated_at"]).memo
        self.assertTrue(is_newer(second["updated_at"], first["updated_at"]))

    def test_stale_base_keeps_both(self):
        first = self.store.save("m", "원래", "v1").memo
        newer = self.store.save("m", "원래", "v2", first["updated_at"]).memo  # 다른 곳(동기화)이 먼저 바꿨다
        result = self.store.save("m", "원래", "내 편집", first["updated_at"])
        self.assertTrue(result.conflict)
        self.assertEqual(result.conflict_of, "m")
        self.assertEqual(result.memo["id"], "copy-1")
        self.assertEqual(result.memo["title"], "원래" + CONFLICT_SUFFIX)
        self.assertEqual(result.memo["text"], "내 편집")
        self.assertEqual(self.store.get("m")["text"], "v2")
        self.assertEqual(self.store.get("m")["updated_at"], newer["updated_at"])
        self.assertEqual({m["id"] for m in self.store.list_public()}, {"m", "copy-1"})

    def test_stale_base_with_identical_text_is_not_a_conflict(self):
        first = self.store.save("m", "a", "v1").memo
        self.store.save("m", "a", "v2", first["updated_at"])
        result = self.store.save("m", "a", "v2", first["updated_at"])
        self.assertFalse(result.conflict)
        self.assertFalse(result.changed)

    def test_saving_over_a_tombstone_resurrects_newest_write_wins(self):
        first = self.store.save("m", "a", "v1").memo
        self.store.delete("m")
        result = self.store.save("m", "a", "살림", first["updated_at"])
        self.assertFalse(result.conflict)
        current = self.store.get("m")
        self.assertFalse(current["deleted"])
        self.assertEqual(current["text"], "살림")

    def test_validation(self):
        with self.assertRaises(MemoValidationError):
            self.store.save("../bad", "t", "x")
        with self.assertRaises(MemoValidationError):
            self.store.save(None, "t", "x")
        with self.assertRaises(MemoValidationError):
            self.store.save("m", "t", "x" * (MAX_TEXT_LENGTH + 1))
        self.assertFalse(self.path.exists())
        saved = self.store.save("m", "가" * (MAX_TITLE_LENGTH + 30), "x" * MAX_TEXT_LENGTH).memo
        self.assertEqual(len(saved["title"]), MAX_TITLE_LENGTH)
        self.assertEqual(self.store.save("n", None, None).memo["text"], "")

    def test_delete_is_a_dirty_tombstone(self):
        self.store.save("m", "제목 유지", "지워질 본문")
        self.store.mark_clean("m", expected=self.store.get("m"))
        tomb = self.store.delete("m")
        self.assertEqual((tomb["deleted"], tomb["text"], tomb["title"], tomb["dirty"]),
                         (True, "", "제목 유지", True))
        self.assertEqual(self.store.list_public(), [])
        self.assertIsNone(self.store.delete("m"))
        self.assertIsNone(self.store.delete("unknown"))
        self.assertIsNone(self.store.delete(5))

    def test_list_is_newest_updated_first(self):
        self.store.save("a", "a", "1")
        self.store.save("b", "b", "1")
        first_a = self.store.get("a")
        self.store.save("a", "a", "2", first_a["updated_at"])
        self.assertEqual([m["id"] for m in self.store.list_public()], ["a", "b"])


class FileSafetyTests(StoreCase):
    def test_corrupt_file_is_quarantined_and_store_starts_empty(self):
        self.path.write_text("{not json", encoding="utf-8")
        self.assertEqual(self.store.list_public(), [])
        self.assertTrue(Path(str(self.path) + ".corrupt").exists())
        self.store.save("m", "a", "b")
        self.assertEqual(len(self.on_disk()["memos"]), 1)

    def test_wrong_root_type_is_quarantined(self):
        self.path.write_text("[1, 2]", encoding="utf-8")
        self.assertEqual(self.store.list_public(), [])
        self.assertTrue(Path(str(self.path) + ".corrupt").exists())

    def test_unreadable_file_refuses_to_overwrite(self):
        self.path.write_text(json.dumps({"schema_version": 1, "memos": [memo("keep")]}), encoding="utf-8")
        before = self.path.read_bytes()
        with mock.patch("builtins.open", side_effect=PermissionError("locked")):
            with self.assertRaises(MemoStoreUnavailableError):
                self.store.save("m", "a", "b")
        self.assertEqual(self.path.read_bytes(), before)
        # 잠금이 풀리면 정상으로 돌아온다
        self.store.save("m", "a", "b")
        self.assertEqual({m["id"] for m in self.on_disk()["memos"]}, {"keep", "m"})

    def test_newer_schema_is_read_but_never_written(self):
        self.path.write_text(json.dumps({"schema_version": 2, "memos": [memo("keep")]}), encoding="utf-8")
        before = self.path.read_bytes()
        self.assertEqual([m["id"] for m in self.store.list_public()], ["keep"])
        with self.assertRaises(MemoStoreUnavailableError):
            self.store.save("m", "a", "b")
        self.assertEqual(self.path.read_bytes(), before)

    def test_bom_and_bad_entries_are_tolerated(self):
        payload = {"schema_version": 1, "memos": [memo("ok"), {"id": "../x"}, "junk", memo("ok")]}
        self.path.write_bytes(b"\xef\xbb\xbf" + json.dumps(payload).encode("utf-8"))
        self.assertEqual([m["id"] for m in self.store.list_public()], ["ok"])

    def test_default_path_is_user_data_memos_json(self):
        from core.storage_paths import StoragePaths
        paths = StoragePaths(self.tmp.name)
        with mock.patch("core.storage_paths.user_data_file", paths.user_data_file):
            store = MemoStore(clock=self.clock)
            self.assertEqual(store.path, Path(self.tmp.name).resolve() / "user_data" / "memos.json")


class CapacityTests(StoreCase):
    def seed(self, memos):
        self.path.write_text(json.dumps({"schema_version": 1, "memos": memos}), encoding="utf-8")
        self.store = self.new_store()

    def test_oldest_clean_tombstones_are_pruned_first(self):
        live = [memo(f"live-{i}", updated=100 + i) for i in range(MAX_MEMOS - 3)]
        tombs = [memo("tomb-old", updated=1, deleted=True), memo("tomb-new", updated=50, deleted=True),
                 memo("tomb-dirty", updated=0, deleted=True, dirty=True)]
        self.seed(live + tombs)
        self.store.save("fresh", "f", "x")
        ids = {m["id"] for m in self.store.all_memos()}
        self.assertNotIn("tomb-old", ids)
        self.assertIn("tomb-new", ids)
        self.assertIn("tomb-dirty", ids)
        self.assertEqual(len(ids), MAX_MEMOS)

    def test_unsent_tombstones_are_never_pruned(self):
        """오프라인 삭제(dirty 삭제 표시)를 지우면 다음 동기화가 서버 판을 되살린다 — 새 메모를 거절한다."""
        live = [memo(f"live-{i}") for i in range(MAX_MEMOS - 1)]
        self.seed(live + [memo("offline-delete", deleted=True, dirty=True, updated=1)])
        with self.assertRaises(MemoValidationError):
            self.store.save("fresh", "f", "x")
        self.assertTrue(self.store.get("offline-delete")["deleted"])
        self.assertEqual(len(self.store.all_memos()), MAX_MEMOS)

    def test_refused_new_memo_prunes_nothing(self):
        live = [memo(f"live-{i}") for i in range(MAX_MEMOS - 2)]
        self.seed(live + [memo("sent", deleted=True, updated=1),
                          memo("unsent", deleted=True, dirty=True, updated=2)])
        self.store.save("one", "t", "x")            # 보낸 삭제 표시 하나만 정리해 들어간다
        self.assertIsNone(self.store.get("sent"))
        before = self.store.all_memos()
        with self.assertRaises(MemoValidationError):
            self.store.save("two", "t", "x")
        self.assertEqual(self.store.all_memos(), before)

    def test_tombstones_every_server_has_seen_go_first(self):
        # 아직 못 받은 서버가 있는 삭제 표시(old-unseen)는 한도에 세지 않는다 — 살아 있는 499 + new-seen 이 가득
        live = [memo(f"live-{i}", updated=100 + i) for i in range(MAX_MEMOS - 1)]
        tombs = [memo("old-unseen", deleted=True, updated=1), memo("new-seen", deleted=True, updated=50)]
        # 한 서버는 old-unseen 을 삭제 전 판(ts 0)으로만 안다 — 지우면 그 서버에서 되살아난다
        sync = {"servers": {SERVER: {"base": {"old-unseen": ts(0), "new-seen": ts(50)}}}}
        self.path.write_text(json.dumps({"schema_version": 1, "memos": live + tombs, "sync": sync}),
                             encoding="utf-8")
        self.store = self.new_store()
        self.store.save("fresh", "f", "x")
        ids = {m["id"] for m in self.store.all_memos()}
        self.assertNotIn("new-seen", ids)
        self.assertIn("old-unseen", ids)
        self.assertNotIn("new-seen", self.store.sync_bases(SERVER))

    def test_full_of_live_memos_rejects_new_memo(self):
        self.seed([memo(f"live-{i}") for i in range(MAX_MEMOS)])
        with self.assertRaises(MemoValidationError):
            self.store.save("one-more", "t", "x")
        # 기존 메모 편집은 된다
        self.store.save("live-0", "t", "edited")

    def test_conflict_title_fits(self):
        title = conflict_title("가" * 200)
        self.assertEqual(len(title), MAX_TITLE_LENGTH)
        self.assertTrue(title.endswith(CONFLICT_SUFFIX))


class PlanTests(unittest.TestCase):
    def test_one_sided_changes(self):
        self.assertEqual(plan_memo_merge(None, None, None), NOOP)
        self.assertEqual(plan_memo_merge(None, memo("a"), None), ADOPT_REMOTE)
        self.assertEqual(plan_memo_merge(None, memo("a", deleted=True), None), NOOP)
        self.assertEqual(plan_memo_merge(memo("a", dirty=True), None, None), PUSH)
        self.assertEqual(plan_memo_merge(memo("a"), None, ts(10)), PUSH)   # 서버가 잃어버림 → 다시 올린다
        self.assertEqual(plan_memo_merge(memo("a", deleted=True, dirty=True), None, None), MARK_CLEAN)
        self.assertEqual(plan_memo_merge(memo("a", deleted=True), None, ts(10)), NOOP)
        # 로컬 그대로, 서버만 바뀜 → 서버 것
        self.assertEqual(plan_memo_merge(memo("a"), memo("a", text="new", updated=20), ts(10)), ADOPT_REMOTE)
        self.assertEqual(plan_memo_merge(memo("a"), memo("a", updated=20, deleted=True), ts(10)), ADOPT_REMOTE)
        # 둘 다 그대로
        self.assertEqual(plan_memo_merge(memo("a"), memo("a"), ts(10)), NOOP)
        # 서버 기준이 'Z' 로 적혀 있어도 같은 순간
        self.assertEqual(plan_memo_merge(memo("a"), memo("a"), ts(10).replace("+00:00", "Z")), NOOP)
        # 로컬만 바뀜 → 올린다
        self.assertEqual(plan_memo_merge(memo("a", text="mine", updated=30, dirty=True), memo("a"), ts(10)), PUSH)
        self.assertEqual(plan_memo_merge(memo("a", updated=30, deleted=True, dirty=True), memo("a"), ts(10)),
                         PUSH_DELETE)

    def test_both_changed(self):
        base = ts(10)
        mine = memo("a", text="mine", updated=30, dirty=True)
        theirs = memo("a", text="theirs", updated=20)
        self.assertEqual(plan_memo_merge(mine, theirs, base), CONFLICT_COPY)
        # 같은 내용으로 바뀌었으면 충돌이 아니다
        self.assertEqual(plan_memo_merge(memo("a", text="same", updated=30, dirty=True),
                                         memo("a", text="same", updated=20), base), ADOPT_REMOTE)
        # 기준이 없는 첫 만남(같은 id) — 둘 다 바뀐 것으로 본다
        self.assertEqual(plan_memo_merge(mine, theirs, None), CONFLICT_COPY)

    def test_tombstones_win_if_newer(self):
        base = ts(10)
        # 서버 삭제가 더 새것 → 삭제
        self.assertEqual(plan_memo_merge(memo("a", text="edit", updated=20, dirty=True),
                                         memo("a", updated=30, deleted=True), base), ADOPT_REMOTE)
        # 로컬 편집이 더 새것 → 되살려 올린다
        self.assertEqual(plan_memo_merge(memo("a", text="edit", updated=40, dirty=True),
                                         memo("a", updated=30, deleted=True), base), PUSH)
        # 로컬 삭제가 더 새것 → 삭제를 보낸다
        self.assertEqual(plan_memo_merge(memo("a", updated=40, deleted=True, dirty=True),
                                         memo("a", text="edit", updated=30), base), PUSH_DELETE)
        # 서버 편집이 더 새것 → 되살린다
        self.assertEqual(plan_memo_merge(memo("a", updated=20, deleted=True, dirty=True),
                                         memo("a", text="edit", updated=30), base), ADOPT_REMOTE)
        # 같은 시각이면 삭제가 이긴다
        self.assertEqual(plan_memo_merge(memo("a", updated=30, deleted=True, dirty=True),
                                         memo("a", text="edit", updated=30), base), PUSH_DELETE)
        # 둘 다 삭제
        self.assertEqual(plan_memo_merge(memo("a", updated=30, deleted=True, dirty=True),
                                         memo("a", updated=20, deleted=True), base), ADOPT_REMOTE)


class SyncSupportTests(StoreCase):
    def test_adopt_remote_respects_concurrent_local_edit(self):
        self.store.save("m", "a", "local")
        snapshot = self.store.get("m")
        self.store.save("m", "a", "typed during sync", snapshot["updated_at"])
        adopted = self.store.adopt_remote(SERVER, memo("m", text="remote", updated=99), expected=snapshot)
        self.assertFalse(adopted)
        self.assertEqual(self.store.get("m")["text"], "typed during sync")
        self.assertEqual(self.store.sync_bases(SERVER), {})

    def test_adopt_remote_sets_clean_and_base(self):
        self.assertTrue(self.store.adopt_remote(SERVER, memo("m", updated=42), expected=None))
        self.assertFalse(self.store.get("m")["dirty"])
        self.assertEqual(self.store.sync_bases(SERVER), {"m": ts(42)})
        self.assertEqual(self.new_store().sync_bases(SERVER), {"m": ts(42)})
        # 아직 없던 메모인데 그사이 로컬에 생겼으면 들이지 않는다
        self.store.save("n", "local", "x")
        self.assertFalse(self.store.adopt_remote(SERVER, memo("n", updated=50), expected=None))

    def test_record_push_moves_base_even_when_local_changed(self):
        self.store.save("m", "a", "v1")
        pushed = self.store.get("m")
        self.store.save("m", "a", "v2", pushed["updated_at"])
        server_copy = dict(pushed, updated_at=pushed["updated_at"], dirty=False)
        self.assertFalse(self.store.record_push(SERVER, server_copy, expected=pushed))
        self.assertEqual(self.store.sync_bases(SERVER)["m"], pushed["updated_at"])
        self.assertTrue(self.store.get("m")["dirty"])
        self.assertEqual(self.store.get("m")["text"], "v2")

    def test_server_rewritten_timestamp_is_not_a_conflict(self):
        """시계 차이로 서버가 updated_at 을 고쳐 돌려줘도, 화면이 든 옛 base 는 같은 판본이다."""
        saved = self.store.save("m", "t", "v1").memo
        pushed = self.store.get("m")
        server_copy = dict(pushed, updated_at=ts(10_000))      # 서버 시각이 앞서 있다
        self.assertTrue(self.store.record_push(SERVER, server_copy, expected=pushed))
        result = self.store.save("m", "t", "v2", saved["updated_at"])
        self.assertFalse(result.conflict)
        self.assertEqual(self.store.get("m")["text"], "v2")
        self.assertTrue(is_newer(result.memo["updated_at"], ts(10_000)))
        # 내용이 바뀐 뒤에는 옛 별칭이 더 이상 통하지 않는다
        self.store.adopt_remote(SERVER, dict(self.store.get("m"), text="forge", updated_at=ts(20_000)),
                                expected=self.store.get("m"))
        stale = self.store.save("m", "t", "mine", saved["updated_at"])
        self.assertTrue(stale.conflict)

    def test_resolve_conflict_copies_current_local_text(self):
        self.store.save("m", "제목", "mine")
        copy_memo = self.store.resolve_conflict(SERVER, memo("m", title="제목", text="theirs", updated=99))
        self.assertEqual(copy_memo["title"], "제목" + CONFLICT_SUFFIX)
        self.assertEqual(copy_memo["text"], "mine")
        self.assertTrue(copy_memo["dirty"])
        self.assertEqual(self.store.get("m")["text"], "theirs")
        self.assertFalse(self.store.get("m")["dirty"])
        # 내용이 같으면 사본을 만들지 않는다
        self.store.save("s", "x", "same")
        self.assertIsNone(self.store.resolve_conflict(SERVER, memo("s", title="x", text="same", updated=99)))

    def test_servers_are_tracked_separately_and_bounded(self):
        self.store.adopt_remote("http://a", memo("m", updated=1), expected=None)
        self.store.finish_sync("http://a", revision=3)
        self.assertEqual(self.store.sync_bases("http://b"), {})
        self.assertIsNone(self.store.last_synced_at("http://b"))
        self.assertIsNotNone(self.store.last_synced_at("http://a"))
        for index in range(ms.MAX_SYNC_SERVERS + 3):
            self.store.finish_sync(f"http://s{index}", revision=0)
        servers = self.on_disk()["sync"]["servers"]
        self.assertEqual(len(servers), ms.MAX_SYNC_SERVERS)
        self.assertIn(f"http://s{ms.MAX_SYNC_SERVERS + 2}", servers)
        self.assertNotIn("http://a", servers)

    def test_flush_writes_deferred_changes(self):
        self.store.adopt_remote(SERVER, memo("m"), expected=None, persist=False)
        self.assertFalse(self.path.exists())
        self.store.flush()
        self.assertEqual(self.on_disk()["memos"][0]["id"], "m")

    def test_reload_rereads_disk(self):
        self.store.save("m", "a", "b")
        data = self.on_disk()
        data["memos"][0]["text"] = "outside"
        self.path.write_text(json.dumps(data), encoding="utf-8")
        self.store.reload()
        self.assertEqual(self.store.get("m")["text"], "outside")


if __name__ == "__main__":
    unittest.main()
