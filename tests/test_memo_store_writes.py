"""core.memo_store — 파일 쓰기가 실패할 때 · 쓸 수 없는 글자 · 파일 모양.

- 외톨이 서로게이트('\\ud83d', 이모지 가운데서 잘린 JS 문자열)는 U+FFFD 로 바꾼다(확장과 같은 규칙) —
  그대로 두면 UTF-8 로 쓰지 못해 그 뒤의 모든 저장(다른 메모까지)이 실패한다.
- 파일에 못 쓴 변경은 메모리에서도 되돌린다 — memoState 가 저장된 것으로 보여 주면 화면은 다시 보내지
  않고, 그 편집은 앱을 닫을 때 사라진다.
"""
import json
import unittest
from unittest import mock

from core.memo_store import MemoStoreUnavailableError, sanitize_memo, strip_lone_surrogates
from tests.test_memo_store import SERVER, StoreCase, memo

LONE = json.loads('"emoji cut \\ud83d"')          # requestAction 의 JSON.stringify 가 만드는 모양


class SurrogateTests(StoreCase):
    def test_lone_surrogate_is_replaced_and_the_file_is_written(self):
        saved = self.store.save("m1", LONE, LONE).memo
        self.assertEqual(saved["text"], "emoji cut �")
        self.assertEqual(saved["title"], "emoji cut �")
        self.assertEqual(self.on_disk()["memos"][0]["text"], "emoji cut �")
        self.store.save("m2", "t", "다른 메모")                  # 뒤의 저장도 멀쩡하다
        self.assertEqual(len(self.on_disk()["memos"]), 2)

    def test_pairs_are_kept_and_remote_memos_are_cleaned(self):
        self.assertEqual(strip_lone_surrogates("🙂 ok"), "🙂 ok")
        self.assertEqual(strip_lone_surrogates(""), "")
        remote = sanitize_memo(dict(memo("r"), text=LONE))
        self.assertEqual(remote["text"], "emoji cut �")
        self.assertTrue(self.store.adopt_remote(SERVER, dict(memo("r"), text=LONE), expected=None))
        self.assertEqual(self.on_disk()["memos"][0]["text"], "emoji cut �")


class FailedWriteRollsBackTests(StoreCase):
    def failing_write(self):
        return mock.patch("core.memo_store.atomic_write_json", side_effect=OSError("disk full"))

    def test_failed_save_leaves_memory_as_on_disk(self):
        first = self.store.save("m1", "t", "v1").memo
        with self.failing_write(), self.assertRaises(OSError):
            self.store.save("m1", "t", "v2", first["updated_at"])
        self.assertEqual(self.store.get("m1")["text"], "v1")
        with self.failing_write(), self.assertRaises(OSError):
            self.store.save("new", "t", "x")
        self.assertIsNone(self.store.get("new"))
        # 다시 보내면(화면은 확인이 없으면 같은 저장을 다시 보낸다) 그대로 저장된다
        again = self.store.save("m1", "t", "v2", first["updated_at"])
        self.assertFalse(again.conflict)
        self.assertEqual(self.on_disk()["memos"][0]["text"], "v2")

    def test_failed_delete_and_stale_copy_are_rolled_back(self):
        first = self.store.save("m1", "t", "v1").memo
        self.store.save("m1", "t", "v2", first["updated_at"])
        with self.failing_write(), self.assertRaises(OSError):
            self.store.delete("m1")
        self.assertFalse(self.store.get("m1")["deleted"])
        with self.failing_write(), self.assertRaises(OSError):
            self.store.save("m1", "t", "stale", first["updated_at"])
        self.assertEqual([m["id"] for m in self.store.list_public()], ["m1"])
        # 실패한 사본은 기억하지 않는다 — 다시 보내면 사본을 새로 만든다
        retry = self.store.save("m1", "t", "stale", first["updated_at"])
        self.assertTrue(retry.conflict and retry.changed)
        self.assertEqual(len(self.store.list_public()), 2)

    def test_failed_sync_write_keeps_bases_unchanged(self):
        self.store.save("m1", "t", "v1")
        pushed = self.store.get("m1")
        with self.failing_write(), self.assertRaises(OSError):
            self.store.record_push(SERVER, pushed, expected=pushed)
        self.assertEqual(self.store.sync_bases(SERVER), {})
        self.assertTrue(self.store.get("m1")["dirty"])

    def test_newer_schema_shows_what_is_on_disk_after_a_refused_save(self):
        self.path.write_text(json.dumps({"schema_version": 2, "memos": [memo("keep", text="on disk")]}),
                             encoding="utf-8")
        with self.assertRaises(MemoStoreUnavailableError):
            self.store.save("keep", "t", "edited", self.store.get("keep")["updated_at"])
        with self.assertRaises(MemoStoreUnavailableError):
            self.store.save("new", "t", "x")
        self.assertEqual([(m["id"], m["text"]) for m in self.store.list_public()], [("keep", "on disk")])


class FileFormatTests(StoreCase):
    def test_file_is_compact_json(self):
        self.store.save("m1", "제목", "본문\n둘째 줄")
        raw = self.path.read_text(encoding="utf-8")
        self.assertNotIn("\n", raw)           # 자동 저장마다 통째로 쓰는 파일 — 들여쓰기 없이
        self.assertEqual(json.loads(raw)["memos"][0]["text"], "본문\n둘째 줄")


if __name__ == "__main__":
    unittest.main()
