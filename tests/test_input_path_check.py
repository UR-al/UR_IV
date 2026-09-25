"""core.path_safety.check_input_path — safe_input_path 와 같은 판정 + 실패 이유.

예전엔 None 하나라서 호출부(core/i2i_payload)가 막힌 확장자(.tif)·시스템 폴더도
'파일을 찾을 수 없습니다'로 알렸다. 이유는 달라도 판정(경로/None)은 safe_input_path 와 한 벌이어야 한다.
"""
from __future__ import annotations

import os
import tempfile
import unittest
from unittest import mock

from core import path_safety
from core.path_safety import (
    INPUT_BLOCKED_EXT,
    INPUT_EMPTY,
    INPUT_FORBIDDEN,
    INPUT_MISSING,
    INPUT_NOT_FILE,
    INPUT_OK,
    INPUT_UNRESOLVABLE,
    check_input_path,
    safe_input_path,
)


class CheckInputPathTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = self._tmp.name
        self.image = os.path.join(self.root, '한글 이미지.png')
        with open(self.image, 'wb') as handle:
            handle.write(b'\x89PNG')
        self.text = os.path.join(self.root, 'notes.txt')
        with open(self.text, 'w', encoding='utf-8') as handle:
            handle.write('x')
        self.folder = os.path.join(self.root, 'folder.png')
        os.mkdir(self.folder)

    def tearDown(self):
        self._tmp.cleanup()

    def test_reasons(self):
        cases = (
            ('', INPUT_EMPTY),
            (self.image, INPUT_OK),
            ('file:///' + self.image.replace('\\', '/'), INPUT_OK),
            (os.path.join(self.root, 'gone.png'), INPUT_MISSING),
            (self.text, INPUT_BLOCKED_EXT),
            (self.folder, INPUT_NOT_FILE),
        )
        for raw, reason in cases:
            with self.subTest(raw=raw):
                resolved, got = check_input_path(raw)
                self.assertEqual(got, reason)
                self.assertEqual(resolved is not None, reason == INPUT_OK)

    def test_allowed_exts_none_skips_the_extension_check(self):
        resolved, reason = check_input_path(self.text, allowed_exts=None)
        self.assertEqual(reason, INPUT_OK)
        self.assertEqual(os.path.normcase(resolved), os.path.normcase(os.path.realpath(self.text)))

    def test_forbidden_and_unresolvable(self):
        with mock.patch.object(path_safety, '_is_forbidden', return_value=True):
            self.assertEqual(check_input_path(self.image), (None, INPUT_FORBIDDEN))
        with mock.patch.object(path_safety.Path, 'resolve', side_effect=OSError('bad')):
            self.assertEqual(check_input_path(self.image), (None, INPUT_UNRESOLVABLE))

    def test_blocked_extension_is_reported_before_existence(self):
        # 없는 .txt 도 '없음'이 아니라 '막힌 형식' — safe_input_path 와 같은 검사 순서
        self.assertEqual(check_input_path(os.path.join(self.root, 'gone.txt')), (None, INPUT_BLOCKED_EXT))

    def test_safe_input_path_is_the_same_verdict(self):
        for raw in ('', self.image, self.text, self.folder, os.path.join(self.root, 'gone.png')):
            for exts in (None, frozenset({'.png'}), frozenset({'.txt'})):
                with self.subTest(raw=raw, exts=exts):
                    self.assertEqual(safe_input_path(raw, allowed_exts=exts),
                                     check_input_path(raw, allowed_exts=exts)[0])

    def test_missing_verdict_agrees_with_missing_input_path(self):
        for raw in (os.path.join(self.root, 'gone.png'), self.folder, self.text, self.image):
            with self.subTest(raw=raw):
                self.assertEqual(check_input_path(raw)[1] == INPUT_MISSING, path_safety.missing_input_path(raw))


if __name__ == '__main__':
    unittest.main()
