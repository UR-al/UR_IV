"""휴지통 이동 안전장치 회귀 테스트.

예전 move_to_trash 는 send2trash import 가 실패하면 os.remove 로 **영구 삭제**했고
반환값이 없었다. 호출부는 결과와 무관하게 '휴지통으로 이동됨' 을 알렸고, 레거시
갤러리·폴더 정리 도구는 모든 예외에 os.remove 폴백을 따로 두었다.
실제 휴지통을 건드리지 않도록 send2trash 는 전부 가짜 모듈로 바꿔 끼운다.
"""
from __future__ import annotations

import ast
import pathlib
import sys
import tempfile
import types
import unittest
from unittest import mock

import json
import os
import re

from core.image_delete import REFUSED_PATH_MESSAGE, image_delete_result
from core.image_utils import (
    TrashUnavailableError,
    move_to_trash,
    move_to_trash_feedback,
    move_to_trash_result,
)

ROOT = pathlib.Path(__file__).resolve().parents[1]


def _fake_send2trash(calls, error=None):
    module = types.ModuleType("send2trash")

    def send2trash(path):
        calls.append(path)
        if error is not None:
            raise error

    module.send2trash = send2trash
    return module


class MoveToTrashTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.file = pathlib.Path(self._tmp.name) / "a.png"
        self.file.write_bytes(b"png")

    def test_moves_through_send2trash_and_reports_true(self):
        calls = []
        with mock.patch.dict(sys.modules, {"send2trash": _fake_send2trash(calls)}):
            self.assertTrue(move_to_trash(str(self.file)))
        self.assertEqual([pathlib.Path(p) for p in calls], [self.file])

    def test_missing_file_reports_false_without_touching_trash(self):
        calls = []
        with mock.patch.dict(sys.modules, {"send2trash": _fake_send2trash(calls)}):
            self.assertFalse(move_to_trash(str(self.file) + ".gone"))
        self.assertEqual(calls, [])

    def test_missing_send2trash_fails_instead_of_permanent_delete(self):
        with mock.patch.dict(sys.modules, {"send2trash": None}):
            with self.assertRaises(TrashUnavailableError):
                move_to_trash(str(self.file))
        self.assertTrue(self.file.exists(), "send2trash 가 없다고 파일을 영구 삭제하면 안 된다")

    def test_send2trash_failure_propagates_and_keeps_file(self):
        calls = []
        with mock.patch.dict(sys.modules, {"send2trash": _fake_send2trash(calls, PermissionError("denied"))}):
            with self.assertRaises(PermissionError):
                move_to_trash(str(self.file))
        self.assertTrue(self.file.exists())


class FeedbackTests(unittest.TestCase):
    def test_levels_follow_the_real_outcome(self):
        with mock.patch("core.image_utils.move_to_trash", return_value=True):
            self.assertEqual(move_to_trash_feedback("x"), ("info", "휴지통으로 이동됨"))
        with mock.patch("core.image_utils.move_to_trash", return_value=False):
            level, message = move_to_trash_feedback("x")
            self.assertEqual(level, "warning")
            self.assertNotIn("이동됨", message)
        with mock.patch("core.image_utils.move_to_trash", side_effect=TrashUnavailableError("모듈 없음")):
            self.assertEqual(move_to_trash_feedback("x"), ("error", "모듈 없음"))
        with mock.patch(
            "core.image_utils.move_to_trash",
            side_effect=PermissionError(r"denied: C:\Users\me\secret\a.png"),
        ):
            level, message = move_to_trash_feedback("x")
            self.assertEqual(level, "error")
            self.assertNotIn(r"C:\Users\me", message)

    def test_result_says_whether_the_list_entry_may_go(self):
        with mock.patch("core.image_utils.move_to_trash", return_value=True):
            r = move_to_trash_result("x")
            self.assertEqual((r["ok"], r["removed"], r["level"]), (True, True, "info"))
        with mock.patch("core.image_utils.move_to_trash", return_value=False):
            r = move_to_trash_result("x")
            self.assertEqual((r["ok"], r["removed"], r["level"]), (False, True, "warning"))
        for error in (TrashUnavailableError("모듈 없음"), PermissionError("denied")):
            with mock.patch("core.image_utils.move_to_trash", side_effect=error):
                r = move_to_trash_result("x")
                self.assertEqual((r["ok"], r["removed"], r["level"]), (False, False, "error"))


class ImageDeleteResultTests(unittest.TestCase):
    """delete_image 한 건의 결과 — 프론트는 removed 가 참일 때만 목록에서 뺀다."""

    GALLERY_EXTS = frozenset({".png", ".mp4"})

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.root = pathlib.Path(self._tmp.name)
        self.file = self.root / "a.png"
        self.file.write_bytes(b"png")
        # 프론트가 보내는 표기('/' 구분자) — 결과에 그대로 돌아와야 목록에서 찾는다.
        self.request = self.file.as_posix()

    def _run(self, request, local, calls, error=None, **kwargs):
        with mock.patch.dict(sys.modules, {"send2trash": _fake_send2trash(calls, error)}):
            return image_delete_result(request, local, **kwargs)

    def test_moved_file_is_reported_removed_and_request_path_is_echoed(self):
        calls = []
        result = self._run(self.request, str(self.file), calls)
        self.assertEqual(result, {
            "path": self.request, "ok": True, "removed": True,
            "level": "info", "message": "휴지통으로 이동됨",
        })
        self.assertEqual(len(calls), 1)
        json.dumps(result, ensure_ascii=False)   # 시그널로 그대로 나간다

    def test_trash_failure_keeps_the_entry(self):
        calls = []
        result = self._run(self.request, str(self.file), calls, PermissionError("locked"))
        self.assertFalse(result["ok"])
        self.assertFalse(result["removed"], "실패했는데 목록에서 빼면 파일은 남고 화면에서만 사라진다")
        self.assertEqual(result["level"], "error")
        self.assertTrue(self.file.exists())

    def test_missing_send2trash_keeps_the_entry_and_the_file(self):
        with mock.patch.dict(sys.modules, {"send2trash": None}):
            result = image_delete_result(self.request, str(self.file))
        self.assertEqual((result["ok"], result["removed"], result["level"]), (False, False, "error"))
        self.assertTrue(self.file.exists())

    def test_already_missing_file_may_leave_the_list(self):
        calls = []
        gone = self.root / "gone.png"
        result = self._run(gone.as_posix(), str(gone), calls)
        self.assertEqual((result["ok"], result["removed"], result["level"]), (False, True, "warning"))
        self.assertEqual(calls, [])

    def test_refused_paths_stay_and_never_reach_the_trash(self):
        calls = []
        text = self.root / "notes.txt"
        text.write_text("x", encoding="utf-8")
        refused = [str(text), ""]
        if os.name == "nt":
            windir = os.environ.get("SystemRoot") or r"C:\Windows"
            refused.append(os.path.join(windir, "Web", "Wallpaper", "Windows", "img0.jpg"))
        for local in refused:
            with self.subTest(local=local):
                result = self._run(local or "x", local, calls)
                self.assertEqual((result["ok"], result["removed"]), (False, False))
                self.assertEqual(result["message"], REFUSED_PATH_MESSAGE)
        self.assertEqual(calls, [])
        self.assertTrue(text.exists())

    def test_gallery_media_extensions_are_deletable_when_allowed(self):
        calls = []
        video = self.root / "clip.mp4"
        video.write_bytes(b"mp4")
        refused = self._run(video.as_posix(), str(video), calls)
        self.assertFalse(refused["removed"])        # 기본(정지 이미지)만 허용이면 거부
        moved = self._run(video.as_posix(), str(video), calls, allowed_exts=self.GALLERY_EXTS)
        self.assertTrue(moved["ok"])
        self.assertEqual(len(calls), 1)


class DeleteResultWiringTests(unittest.TestCase):
    """프론트가 결과를 기다리지 않고 목록에서 먼저 지우던 문제의 회귀 가드."""

    def test_delete_branch_emits_a_per_path_result(self):
        tree = ast.parse((ROOT / "ui" / "generator_main.py").read_text(encoding="utf-8"))
        handler = next(
            fn for fn in ast.walk(tree)
            if isinstance(fn, ast.FunctionDef) and fn.name == "_handle_vue_action"
        )
        branch = NoPermanentDeleteFallbackTests._delete_branch(handler)
        self.assertIsNotNone(branch)
        attrs = {node.attr for node in ast.walk(branch) if isinstance(node, ast.Attribute)}
        names = {
            (node.func.attr if isinstance(node.func, ast.Attribute) else getattr(node.func, "id", ""))
            for node in ast.walk(branch) if isinstance(node, ast.Call)
        }
        self.assertIn("imageDeleteResult", attrs)
        self.assertIn("image_delete_result", names)
        bridge = (ROOT / "ui" / "vue_bridge.py").read_text(encoding="utf-8")
        self.assertRegex(bridge, r"(?m)^\s+imageDeleteResult\s*=\s*pyqtSignal\(str\)")
        import web_main_ui
        self.assertIn("imageDeleteResult", web_main_ui._WEB_SIGNALS)

    def test_frontend_removes_only_after_the_result(self):
        gallery = (ROOT / "frontend" / "src" / "views" / "GalleryView.vue").read_text(encoding="utf-8")
        ctx_body = re.search(r"function ctx\([^)]*\)\s*\{(.*?)\n\}", gallery, re.S)
        self.assertIsNotNone(ctx_body)
        self.assertNotIn("images.value =", ctx_body.group(1))
        self.assertNotIn("_cache", ctx_body.group(1))
        self.assertIn("onBackendEvent('imageDeleteResult'", gallery)

        app = (ROOT / "frontend" / "src" / "App.vue").read_text(encoding="utf-8")
        delete_body = re.search(r"const ctxDelete = \(\) => \{(.*?)\n\}", app, re.S)
        self.assertIsNotNone(delete_body)
        self.assertNotIn("historyImages", delete_body.group(1))
        self.assertNotIn("currentImage.value =", delete_body.group(1))
        self.assertIn("onBackendEvent('imageDeleteResult'", app)


class NoPermanentDeleteFallbackTests(unittest.TestCase):
    """move_to_trash 를 부르는 함수에 os.remove/unlink 폴백이 다시 생기지 않게."""

    FILES = (
        "core/image_utils.py",
        "core/image_delete.py",
        "ui/generator_main.py",
        # (옛 ui/generator_gallery.py·tabs/gallery_tab.py·widgets/folder_organizer.py 는
        #  숨은 레거시 갤러리와 함께 은퇴했다 — tests/test_legacy_gallery_tabs_retirement.py)
    )
    FORBIDDEN = {"remove", "unlink"}

    def test_trash_callers_never_fall_back_to_permanent_delete(self):
        offenders = []
        for rel in self.FILES:
            tree = ast.parse((ROOT / rel).read_text(encoding="utf-8"))
            for fn in ast.walk(tree):
                if not isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    continue
                names = {
                    (node.func.attr if isinstance(node.func, ast.Attribute) else getattr(node.func, "id", ""))
                    for node in ast.walk(fn)
                    if isinstance(node, ast.Call)
                }
                if not names & {"move_to_trash", "move_to_trash_feedback",
                                "move_to_trash_result", "image_delete_result"}:
                    continue
                if fn.name == "_handle_vue_action":
                    # 거대 디스패처 — delete_image 분기만 본다.
                    body = self._delete_branch(fn)
                    names = {
                        (node.func.attr if isinstance(node.func, ast.Attribute) else getattr(node.func, "id", ""))
                        for node in ast.walk(body)
                        if isinstance(node, ast.Call)
                    } if body is not None else set()
                if names & self.FORBIDDEN:
                    offenders.append(f"{rel}:{fn.name}")
        self.assertEqual(offenders, [])

    @staticmethod
    def _delete_branch(fn):
        for node in ast.walk(fn):
            if isinstance(node, ast.If) and isinstance(node.test, ast.Compare):
                right = node.test.comparators[0] if node.test.comparators else None
                if isinstance(right, ast.Constant) and right.value == "delete_image":
                    return ast.Module(body=node.body, type_ignores=[])
        return None


class DeleteLabelTests(unittest.TestCase):
    def test_context_menus_say_trash_not_permanent_delete(self):
        gallery = (ROOT / "frontend" / "src" / "views" / "GalleryView.vue").read_text(encoding="utf-8")
        self.assertNotIn("완전 삭제", gallery)
        self.assertRegex(gallery, r"ctx\('delete_image'\)[^\n]*휴지통으로 이동")
        app = (ROOT / "frontend" / "src" / "App.vue").read_text(encoding="utf-8")
        self.assertRegex(app, r'@click="ctxDelete"[^\n]*휴지통으로 이동')


if __name__ == "__main__":
    unittest.main()
