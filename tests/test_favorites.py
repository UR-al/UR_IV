"""즐겨찾기 저장소(core.favorites) 회귀 테스트.

예전 로더는 os.path.exists 로 걸러 읽고 추가·삭제가 그 결과를 곧바로 저장해서,
외장 드라이브가 빠진 채 즐겨찾기를 하나 추가하면 그 드라이브 항목이 영구 삭제됐다.
지워진 이미지는 걸러진 목록에 없어 개별 삭제도 안 됐고, JSON 이 깨지면 다음 추가가
파일을 한 줄로 덮어썼다.
"""
from __future__ import annotations

import builtins
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from core.favorites import (
    FavoritesUnavailableError,
    add_favorite,
    favorite_key,
    is_favorite,
    load_favorites,
    remove_favorite,
    save_favorites,
    toggle_favorite,
)


class FavoritesStoreTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.root = Path(self._tmp.name)
        self.file = str(self.root / "user_data" / "favorites.json")
        self.present = self.root / "a.png"
        self.present.write_bytes(b"png")
        # 분리된 드라이브/지운 이미지 — 지금은 없는 경로
        self.detached = r"Z:\external\gallery\b.png"

    def _write(self, data):
        os.makedirs(os.path.dirname(self.file), exist_ok=True)
        with open(self.file, "w", encoding="utf-8") as fh:
            json.dump(data, fh, ensure_ascii=False)

    def _read(self):
        with open(self.file, encoding="utf-8") as fh:
            return json.load(fh)

    def test_missing_paths_survive_load_and_add(self):
        self._write([self.detached, str(self.present)])
        self.assertEqual(load_favorites(self.file), [self.detached, str(self.present)])
        new_image = str(self.root / "c.png")
        added, entries = add_favorite(new_image, self.file)
        self.assertTrue(added)
        self.assertEqual(self._read(), [self.detached, str(self.present), new_image])
        self.assertEqual(entries, self._read())

    def test_broken_entry_can_be_removed_individually(self):
        self._write([self.detached, str(self.present)])
        removed, entries = remove_favorite(self.detached, self.file)
        self.assertTrue(removed)
        self.assertEqual(self._read(), [str(self.present)])
        self.assertEqual(entries, [str(self.present)])
        # 다시 불러와도 되살아나지 않는다.
        self.assertEqual(load_favorites(self.file), [str(self.present)])

    def test_path_spelling_differences_match_the_same_entry(self):
        stored = r"C:\Users\Me\Pictures\x.png"
        self._write([stored])
        self.assertTrue(is_favorite("C:/Users/Me/Pictures/x.png", self.file) if os.name == "nt"
                        else is_favorite(stored, self.file))
        added, _ = add_favorite(stored, self.file)
        self.assertFalse(added)
        self.assertEqual(self._read(), [stored])
        if os.name == "nt":
            removed, _ = remove_favorite("c:/users/me/pictures/X.PNG", self.file)
            self.assertTrue(removed)
            self.assertEqual(self._read(), [])

    def test_duplicate_add_does_not_rewrite_file(self):
        self._write([str(self.present)])
        before = os.stat(self.file).st_mtime_ns
        added, _ = add_favorite(str(self.present), self.file)
        self.assertFalse(added)
        self.assertEqual(os.stat(self.file).st_mtime_ns, before)

    def test_corrupt_json_is_preserved_not_overwritten(self):
        os.makedirs(os.path.dirname(self.file), exist_ok=True)
        with open(self.file, "w", encoding="utf-8") as fh:
            fh.write('["C:/a.png", ')
        self.assertEqual(load_favorites(self.file), [])
        self.assertTrue(os.path.exists(self.file + ".corrupt"))
        with open(self.file + ".corrupt", encoding="utf-8") as fh:
            self.assertEqual(fh.read(), '["C:/a.png", ')
        add_favorite(str(self.present), self.file)
        self.assertEqual(self._read(), [str(self.present)])

    def test_wrong_root_type_is_quarantined_too(self):
        self._write({"not": "a list"})
        self.assertEqual(load_favorites(self.file), [])
        self.assertTrue(os.path.exists(self.file + ".corrupt"))
        self.assertFalse(os.path.exists(self.file))

    def test_junk_entries_and_duplicates_are_dropped_on_load(self):
        self._write([str(self.present), 5, None, "", "  ", str(self.present)])
        self.assertEqual(load_favorites(self.file), [str(self.present)])

    def test_missing_file_is_empty_and_toggle_round_trips(self):
        self.assertEqual(load_favorites(self.file), [])
        state, entries = toggle_favorite(str(self.present), self.file)
        self.assertTrue(state)
        self.assertEqual(entries, [str(self.present)])
        state, entries = toggle_favorite(str(self.present), self.file)
        self.assertFalse(state)
        self.assertEqual(entries, [])

    def test_save_is_atomic_and_clean(self):
        save_favorites([str(self.present), str(self.present)], self.file)
        self.assertEqual(self._read(), [str(self.present)])
        self.assertFalse(os.path.exists(self.file + ".tmp"))

    def test_add_rejects_empty_path(self):
        with self.assertRaises(ValueError):
            add_favorite("", self.file)

    def test_favorite_key_normalizes_separators(self):
        self.assertEqual(favorite_key("a/b/../c.png"), favorite_key(os.path.join("a", "c.png")))


class FavoritesUnreadableFileTests(unittest.TestCase):
    """파일이 있는데 한 번 못 읽었을 때(백신·OneDrive 공유 위반) 저장하면 안 된다."""

    ORIGINAL = [r"D:\a.png", r"D:\b.png", r"E:\c.png"]

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.file = str(Path(self._tmp.name) / "favorites.json")
        with open(self.file, "w", encoding="utf-8") as fh:
            json.dump(self.ORIGINAL, fh)
        with open(self.file, "rb") as fh:
            self.original_bytes = fh.read()

    def _deny_first_read(self, error: OSError | None = None):
        """favorites.json 을 읽기 모드로 여는 첫 호출만 실패시킨다(쓰기는 그대로)."""
        real_open = builtins.open
        target = os.path.normcase(os.path.abspath(self.file))
        state = {"denied": 0}

        def flaky_open(file, mode="r", *args, **kwargs):
            reading = "r" in mode and not any(flag in mode for flag in "wax+")
            same = isinstance(file, (str, os.PathLike)) and \
                os.path.normcase(os.path.abspath(os.fspath(file))) == target
            if reading and same and state["denied"] == 0:
                state["denied"] += 1
                raise error or PermissionError(13, "sharing violation", self.file)
            return real_open(file, mode, *args, **kwargs)

        return mock.patch("builtins.open", flaky_open), state

    def _assert_untouched(self):
        with open(self.file, "rb") as fh:
            self.assertEqual(fh.read(), self.original_bytes)
        self.assertFalse(os.path.exists(self.file + ".corrupt"))
        self.assertFalse(os.path.exists(self.file + ".tmp"))

    def test_add_aborts_without_saving(self):
        patcher, state = self._deny_first_read()
        with patcher, self.assertRaises(FavoritesUnavailableError):
            add_favorite(r"C:\new.png", self.file)
        self.assertEqual(state["denied"], 1)
        self._assert_untouched()
        # 다음 시도(파일이 풀린 뒤)는 원래 항목을 모두 보존한 채 추가된다.
        added, entries = add_favorite(r"C:\new.png", self.file)
        self.assertTrue(added)
        self.assertEqual(entries, self.ORIGINAL + [r"C:\new.png"])

    def test_remove_aborts_without_saving(self):
        patcher, state = self._deny_first_read()
        with patcher, self.assertRaises(FavoritesUnavailableError):
            remove_favorite(r"D:\a.png", self.file)
        self.assertEqual(state["denied"], 1)
        self._assert_untouched()

    def test_toggle_aborts_without_saving(self):
        patcher, state = self._deny_first_read()
        with patcher, self.assertRaises(FavoritesUnavailableError):
            toggle_favorite(r"C:\new.png", self.file)
        self.assertEqual(state["denied"], 1)
        self._assert_untouched()

    def test_any_os_error_is_unavailable_not_empty(self):
        patcher, _ = self._deny_first_read(OSError(5, "I/O error"))
        with patcher, self.assertRaises(FavoritesUnavailableError):
            load_favorites(self.file)
        self._assert_untouched()

    def test_missing_file_is_still_just_empty(self):
        os.remove(self.file)
        self.assertEqual(load_favorites(self.file), [])

    def test_corrupt_file_that_cannot_be_quarantined_is_not_overwritten(self):
        with open(self.file, "w", encoding="utf-8") as fh:
            fh.write('["C:/a.png", ')
        with mock.patch("core.favorites.os.replace", side_effect=PermissionError(13, "locked")):
            with self.assertRaises(FavoritesUnavailableError):
                add_favorite(r"C:\new.png", self.file)
        with open(self.file, encoding="utf-8") as fh:
            self.assertEqual(fh.read(), '["C:/a.png", ')

    def test_bom_prefixed_file_is_read_not_quarantined(self):
        with open(self.file, "w", encoding="utf-8-sig") as fh:
            json.dump(self.ORIGINAL, fh)
        self.assertEqual(load_favorites(self.file), self.ORIGINAL)
        self.assertFalse(os.path.exists(self.file + ".corrupt"))
        add_favorite(r"C:\new.png", self.file)
        with open(self.file, "rb") as fh:
            self.assertFalse(fh.read().startswith(b"\xef\xbb\xbf"), "저장은 BOM 없는 UTF-8")

    def test_pathlike_target_is_accepted(self):
        target = Path(self.file)
        added, entries = add_favorite(r"C:\new.png", target)
        self.assertTrue(added)
        self.assertEqual(entries, self.ORIGINAL + [r"C:\new.png"])
        with open(self.file, "w", encoding="utf-8") as fh:
            fh.write("{broken")
        self.assertEqual(load_favorites(target), [])
        self.assertTrue(os.path.exists(self.file + ".corrupt"))

    def test_existing_corrupt_backup_is_kept(self):
        with open(self.file + ".corrupt", "w", encoding="utf-8") as fh:
            fh.write("older backup")
        with open(self.file, "w", encoding="utf-8") as fh:
            fh.write("{broken")
        self.assertEqual(load_favorites(self.file), [])
        with open(self.file + ".corrupt", encoding="utf-8") as fh:
            self.assertEqual(fh.read(), "older backup")
        with open(self.file + ".corrupt.1", encoding="utf-8") as fh:
            self.assertEqual(fh.read(), "{broken")


class FavoritesCallerWiringTests(unittest.TestCase):
    """읽기 실패 시 호출부가 '추가됨' 을 띄우거나 옛 목록을 저장하지 않게."""

    ROOT = Path(__file__).resolve().parents[1]

    @staticmethod
    def _call_names(node):
        import ast
        return {
            (call.func.attr if isinstance(call.func, ast.Attribute) else getattr(call.func, "id", ""))
            for call in ast.walk(node) if isinstance(call, ast.Call)
        }

    def _action_branch(self, tree, action):
        import ast
        for node in ast.walk(tree):
            if isinstance(node, ast.If) and isinstance(node.test, ast.Compare):
                right = node.test.comparators[0] if node.test.comparators else None
                if isinstance(right, ast.Constant) and right.value == action:
                    return node
        self.fail(f"{action} 분기를 찾지 못했습니다")

    def test_vue_actions_catch_unavailable_error_before_success_toast(self):
        import ast
        source = (self.ROOT / "ui" / "generator_main.py").read_text(encoding="utf-8")
        tree = ast.parse(source)
        for action, writer in (("add_favorite", "add_favorite"), ("remove_favorite", "remove_favorite")):
            with self.subTest(action=action):
                branch = self._action_branch(tree, action)
                tries = [
                    node for node in ast.walk(ast.Module(body=branch.body, type_ignores=[]))
                    if isinstance(node, ast.Try)
                    and writer in self._call_names(ast.Module(body=node.body, type_ignores=[]))
                ]
                self.assertTrue(tries, f"{action}: {writer} 호출이 try 로 감싸져 있지 않습니다")
                handled = {
                    getattr(elt, "id", getattr(elt, "attr", ""))
                    for handler in tries[0].handlers
                    for elt in (handler.type.elts if isinstance(handler.type, ast.Tuple) else [handler.type])
                }
                self.assertIn("FavoritesUnavailableError", handled)
                # 성공 알림은 try 본문이 아니라 else 에만 — 실패하면 '추가됨' 이 뜨지 않는다.
                body_src = "\n".join(ast.get_source_segment(source, stmt) or "" for stmt in tries[0].body)
                self.assertNotIn("즐겨찾기에 추가됨", body_src)

    def test_only_core_favorites_reads_or_writes_the_favorites_file(self):
        """즐겨찾기 파일의 reader/writer 는 core.favorites 하나다.

        옛 PyQt 갤러리 믹스인(ui/generator_gallery.py)·갤러리 탭의 사본 로직(옛 목록을 통째로 저장)은
        숨은 레거시 탭과 함께 은퇴했다 — UI 계층이 파일 경로를 직접 다루면 다시 두 벌이 된다.
        """
        self.assertFalse((self.ROOT / "ui" / "generator_gallery.py").exists())
        offenders = []
        for package in ("ui", "tabs", "widgets", "workers"):
            for path in sorted((self.ROOT / package).rglob("*.py")):
                source = path.read_text(encoding="utf-8")
                if "FAVORITES_FILE" in source or "favorites.json" in source:
                    offenders.append(str(path.relative_to(self.ROOT)))
        self.assertEqual(offenders, [])


if __name__ == "__main__":
    unittest.main()
