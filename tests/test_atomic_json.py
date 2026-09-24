"""utils/atomic_json.py — 원자적 쓰기 + 손상 백업 로드 테스트."""
import json
import os
import pathlib
import tempfile
import unittest
from unittest import mock

from utils.atomic_json import (
    atomic_write_bytes,
    atomic_write_json,
    atomic_write_text,
    load_json_safe,
)


class TestAtomicWriteJson(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        self.path = os.path.join(self.dir.name, 'sub', 'data.json')

    def tearDown(self):
        self.dir.cleanup()

    def test_write_and_read_roundtrip(self):
        atomic_write_json(self.path, {'a': 1, '한글': [1, 2]})
        with open(self.path, encoding='utf-8') as f:
            self.assertEqual(json.load(f), {'a': 1, '한글': [1, 2]})

    def test_no_tmp_leftover(self):
        atomic_write_json(self.path, [1, 2, 3])
        self.assertFalse(os.path.exists(self.path + '.tmp'))

    def test_overwrite_existing(self):
        atomic_write_json(self.path, {'v': 1})
        atomic_write_json(self.path, {'v': 2})
        self.assertEqual(load_json_safe(self.path, {}), {'v': 2})

    def test_indent_none_compact(self):
        atomic_write_json(self.path, {'a': 1}, indent=None)
        with open(self.path, encoding='utf-8') as f:
            self.assertNotIn('\n', f.read())

    def test_separators_pass_through(self):
        atomic_write_json(self.path, {'a': [1, 2]}, indent=None, separators=(',', ':'))
        with open(self.path, encoding='utf-8') as f:
            self.assertEqual(f.read(), '{"a":[1,2]}')

    def test_accepts_pathlike(self):
        atomic_write_json(pathlib.Path(self.path), {'p': 1})
        self.assertEqual(load_json_safe(self.path, {}), {'p': 1})
        self.assertFalse(os.path.exists(self.path + '.tmp'))

    def test_fsyncs_before_replace_by_default(self):
        order = []
        real_fsync, real_replace = os.fsync, os.replace
        with mock.patch('utils.atomic_json.os.fsync',
                        side_effect=lambda fd: (order.append('fsync'), real_fsync(fd))), \
             mock.patch('utils.atomic_json.os.replace',
                        side_effect=lambda a, b: (order.append('replace'), real_replace(a, b))):
            atomic_write_json(self.path, {'v': 1})
        self.assertEqual(order, ['fsync', 'replace'])

    def test_durable_false_skips_fsync(self):
        with mock.patch('utils.atomic_json.os.fsync') as fsync:
            atomic_write_json(self.path, {'v': 1}, durable=False)
        fsync.assert_not_called()
        self.assertEqual(load_json_safe(self.path, {}), {'v': 1})

    def test_serialization_failure_keeps_original_and_removes_tmp(self):
        atomic_write_json(self.path, {'v': 'original'})
        with self.assertRaises(TypeError):
            atomic_write_json(self.path, {'bad': object()})
        self.assertEqual(load_json_safe(self.path, {}), {'v': 'original'})
        self.assertFalse(os.path.exists(self.path + '.tmp'))

    def test_replace_failure_keeps_original_and_removes_tmp(self):
        atomic_write_json(self.path, {'v': 'original'})
        with mock.patch('utils.atomic_json.os.replace', side_effect=PermissionError('locked')):
            with self.assertRaises(PermissionError):
                atomic_write_json(self.path, {'v': 'new'})
        self.assertEqual(load_json_safe(self.path, {}), {'v': 'original'})
        self.assertFalse(os.path.exists(self.path + '.tmp'))


class TestAtomicWriteBytesAndText(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        self.path = os.path.join(self.dir.name, 'nested', 'out.bin')

    def tearDown(self):
        self.dir.cleanup()

    def test_bytes_roundtrip_with_custom_tmp_suffix(self):
        seen = []
        real_replace = os.replace
        with mock.patch('utils.atomic_json.os.replace',
                        side_effect=lambda a, b: (seen.append(a), real_replace(a, b))):
            atomic_write_bytes(self.path, b'{"a":1}\n', tmp_suffix='.writing')
        self.assertEqual(seen, [self.path + '.writing'])
        with open(self.path, 'rb') as f:
            self.assertEqual(f.read(), b'{"a":1}\n')  # 바이트 그대로 — 개행 변환 없음
        self.assertFalse(os.path.exists(self.path + '.writing'))

    def test_bytes_write_failure_removes_tmp(self):
        with mock.patch('utils.atomic_json.os.fsync', side_effect=OSError('disk full')):
            with self.assertRaises(OSError):
                atomic_write_bytes(self.path, b'data')
        self.assertFalse(os.path.exists(self.path))
        self.assertFalse(os.path.exists(self.path + '.tmp'))

    def test_text_newline_empty_keeps_lf(self):
        atomic_write_text(self.path, 'a\nb', newline='')
        with open(self.path, 'rb') as f:
            self.assertEqual(f.read(), b'a\nb')

    def test_text_utf8_without_bom(self):
        atomic_write_text(self.path, '한글')
        with open(self.path, 'rb') as f:
            self.assertEqual(f.read(), '한글'.encode('utf-8'))


class TestLoadJsonSafe(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        self.path = os.path.join(self.dir.name, 'data.json')

    def tearDown(self):
        self.dir.cleanup()

    def test_missing_returns_default(self):
        self.assertEqual(load_json_safe(self.path, []), [])
        self.assertEqual(load_json_safe(self.path, {'x': 1}), {'x': 1})

    def test_corrupt_backed_up_not_silently_lost(self):
        # 손상 파일 → default 반환 + .corrupt 백업 (다음 저장이 덮어써도 원본 보존)
        with open(self.path, 'w', encoding='utf-8') as f:
            f.write('{"truncated": ')
        self.assertEqual(load_json_safe(self.path, {}), {})
        self.assertFalse(os.path.exists(self.path))
        self.assertTrue(os.path.exists(self.path + '.corrupt'))
        with open(self.path + '.corrupt', encoding='utf-8') as f:
            self.assertEqual(f.read(), '{"truncated": ')

    def test_corrupt_backup_disabled(self):
        with open(self.path, 'w', encoding='utf-8') as f:
            f.write('not json')
        self.assertEqual(load_json_safe(self.path, [], backup_corrupt=False), [])
        self.assertTrue(os.path.exists(self.path))  # 원본 유지

    def test_wrong_root_type_returns_default(self):
        atomic_write_json(self.path, {'a': 1})
        self.assertEqual(load_json_safe(self.path, []), [])  # dict인데 list 기대 → default

    def test_valid_load(self):
        atomic_write_json(self.path, [{'p': 'x'}])
        self.assertEqual(load_json_safe(self.path, []), [{'p': 'x'}])


if __name__ == "__main__":
    unittest.main()
