"""core.editor_autosave — 워커에서 쓰는 크래시 복구본의 순서·폐기 경합.

예전 동기 슬롯(editorAutoSave)은 레이어 합성을 GUI 스레드에서 해 5분마다 창을 멈췄다. 워커로
옮기면 (1) 두 쓰기가 한 복구본을 섞어 쓰거나 (2) 저장 성공 → 폐기 뒤에 끝난 쓰기가 이미 저장한
작업의 복구본을 되살릴 수 있다. 브리지 경유 흐름은 tests/test_editor_preview.py 가 본다.
"""
from __future__ import annotations

import json
import tempfile
import threading
import unittest
from pathlib import Path
from unittest import mock

from PIL import Image

import core.editor_save as editor_save
from core.editor_autosave import (
    AUTOSAVE_IMAGE, AUTOSAVE_META, AutosaveWriter, autosave_dir, remove_autosave_files,
)


class AutosaveWriterTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        root = Path(self._tmp.name)
        self.folder = root / 'autosave'
        self.src = root / '한글 폴더' / '원본.png'
        self.src.parent.mkdir(parents=True)
        Image.new('RGB', (8, 6), (10, 20, 30)).save(self.src)
        self.writer = AutosaveWriter()

    def _write(self, ticket=None):
        ticket = self.writer.ticket() if ticket is None else ticket
        return self.writer.write(str(self.src), None, 1.0, ticket, folder=str(self.folder))

    def test_write_copies_the_image_and_records_the_original_path(self):
        result = self._write()
        self.assertEqual(result, {'path': str(self.folder / AUTOSAVE_IMAGE).replace('\\', '/'), 'drawing': False})
        self.assertEqual((self.folder / AUTOSAVE_IMAGE).read_bytes(), self.src.read_bytes())
        meta = json.loads((self.folder / AUTOSAVE_META).read_text(encoding='utf-8'))
        self.assertEqual(meta['original'], str(self.src))
        self.assertFalse(meta['drawing'])
        self.assertGreater(meta['saved_at'], 0)

    def test_a_write_requested_before_invalidate_leaves_nothing(self):
        ticket = self.writer.ticket()
        self.writer.invalidate()
        self.assertEqual(self._write(ticket), {'discarded': True})
        self.assertFalse(self.folder.exists(), '폐기된 요청이 폴더·파일을 만들었다')

    def test_invalidate_during_the_write_removes_what_was_written(self):
        real = editor_save.write_autosave_snapshot

        def write_then_invalidate(*args, **kwargs):
            drawing = real(*args, **kwargs)
            self.writer.invalidate()   # 저장 완료 → 폐기가 쓰기 도중에 왔다
            return drawing

        with mock.patch.object(editor_save, 'write_autosave_snapshot', side_effect=write_then_invalidate):
            self.assertEqual(self._write(), {'discarded': True})
        self.assertFalse((self.folder / AUTOSAVE_IMAGE).exists())
        self.assertFalse((self.folder / AUTOSAVE_META).exists())

    def test_invalidate_does_not_wait_for_a_running_write(self):
        """폐기는 GUI 스레드에서 온다 — 쓰기(최대 1초 넘게)를 기다리면 다시 창이 멈춘다."""
        entered, release = threading.Event(), threading.Event()
        real = editor_save.write_autosave_snapshot

        def slow_write(*args, **kwargs):
            entered.set()
            release.wait(5)
            return real(*args, **kwargs)

        with mock.patch.object(editor_save, 'write_autosave_snapshot', side_effect=slow_write):
            worker = threading.Thread(target=self._write)
            worker.start()
            try:
                self.assertTrue(entered.wait(5))
                done = threading.Event()
                threading.Thread(target=lambda: (self.writer.invalidate(), done.set())).start()
                self.assertTrue(done.wait(1), 'invalidate 가 쓰기 잠금을 기다린다')
            finally:
                release.set()
                worker.join(5)
        self.assertFalse((self.folder / AUTOSAVE_IMAGE).exists())

    def test_writes_are_serialized(self):
        """복구본은 한 장 — 두 쓰기가 겹치면 그림과 메타가 다른 요청의 것이 될 수 있다."""
        active = []
        overlap = []
        real = editor_save.write_autosave_snapshot

        def tracked(*args, **kwargs):
            active.append(1)
            if len(active) > 1:
                overlap.append(True)
            try:
                threading.Event().wait(0.05)
                return real(*args, **kwargs)
            finally:
                active.pop()

        with mock.patch.object(editor_save, 'write_autosave_snapshot', side_effect=tracked):
            threads = [threading.Thread(target=self._write) for _ in range(3)]
            for t in threads:
                t.start()
            for t in threads:
                t.join(5)
        self.assertEqual(overlap, [], '자동저장 쓰기가 겹쳤다')

    def test_remove_autosave_files_tolerates_missing_files(self):
        self.assertTrue(remove_autosave_files(str(self.folder)))
        self._write()
        self.assertTrue(remove_autosave_files(str(self.folder)))
        self.assertFalse((self.folder / AUTOSAVE_IMAGE).exists())
        self.assertFalse((self.folder / AUTOSAVE_META).exists())

    def test_autosave_dir_follows_the_os_temp_dir(self):
        with mock.patch('tempfile.gettempdir', return_value=str(Path(self._tmp.name) / 'os')):
            self.assertEqual(Path(autosave_dir()), Path(self._tmp.name) / 'os' / 'AIStudioPro_editor')


if __name__ == '__main__':
    unittest.main()
