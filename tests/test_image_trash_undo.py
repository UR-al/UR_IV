"""History undo uses disposable test files; never touch the real recycle bin."""
from __future__ import annotations

import os
from pathlib import Path
import shutil
import stat
import tempfile
import unittest
from unittest.mock import patch

from core.image_delete import image_delete_result
from core.image_trash_undo import ImageTrashUndoManager


class ImageTrashUndoTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.manager = ImageTrashUndoManager(max_entries=3, max_bytes=1024)
        self.addCleanup(self.manager.close)
        self.calls = []

    def make_file(self, name='image.png', content=b'PNG pixels and EXIF'):
        path = self.root / name
        path.write_bytes(content)
        return path

    def fake_trash(self, path):
        self.calls.append(path)
        # Move into our own test fixture, never invoke send2trash or OS recycle bin.
        shutil.move(path, self.root / f'test-bin-{len(self.calls)}.png')
        return {'ok': True, 'removed': True, 'level': 'info', 'message': '휴지통으로 이동됨'}

    def delete(self, path, **kwargs):
        with patch('core.image_delete.move_to_trash_result', side_effect=self.fake_trash):
            return image_delete_result(path.as_posix(), str(path), undo_manager=self.manager, **kwargs)

    def test_success_restores_exact_bytes_original_path_and_timestamp(self):
        content = b'PNG\x00binary metadata\xff' * 4
        path = self.make_file('이름%20 image.png', content)
        original_mtime = path.stat().st_mtime_ns
        deleted = self.delete(path)
        self.assertTrue(deleted['ok'])
        self.assertFalse(path.exists())
        self.assertNotIn(str(path), deleted['undo_token'])
        result = self.manager.restore(deleted['undo_token'])
        self.assertTrue(result['ok'])
        self.assertTrue(result['restored'])
        self.assertEqual(result['path'], path.as_posix())
        self.assertEqual(path.read_bytes(), content)
        self.assertEqual(path.stat().st_mtime_ns, original_mtime)
        self.assertTrue((self.root / 'test-bin-1.png').exists(), 'OS recycle-bin item is not guessed or removed')
        self.assertEqual(self.manager._bytes, 0)
        self.assertEqual(list(Path(self.manager._temporary.name).iterdir()), [])
        duplicate = self.manager.restore(deleted['undo_token'])
        self.assertFalse(duplicate['ok'])
        self.assertFalse(duplicate['retryable'])

    def test_restore_never_overwrites_new_file_and_token_can_be_retried(self):
        path = self.make_file(content=b'original')
        deleted = self.delete(path)
        path.write_bytes(b'new file')
        result = self.manager.restore(deleted['undo_token'])
        self.assertFalse(result['restored'])
        self.assertTrue(result['retryable'])
        self.assertEqual(path.read_bytes(), b'new file')
        path.unlink()
        self.assertTrue(self.manager.restore(deleted['undo_token'])['restored'])
        self.assertEqual(path.read_bytes(), b'original')

    def test_readonly_original_keeps_snapshot_writable_and_does_not_block_future_deletes(self):
        path = self.make_file(content=b'readonly image')
        path.chmod(stat.S_IREAD)
        deleted = self.delete(path)
        self.assertTrue(deleted['ok'])
        snapshot = self.manager._entries[deleted['undo_token']].snapshot
        self.assertTrue(snapshot.stat().st_mode & stat.S_IWRITE)
        result = self.manager.restore(deleted['undo_token'])
        self.assertTrue(result['restored'])
        self.assertFalse(path.stat().st_mode & stat.S_IWRITE, 'original read-only attribute is restored')
        self.assertFalse(snapshot.exists())
        self.assertFalse(self.manager._cleanup_failed)
        self.assertTrue(self.delete(self.make_file('next.png'))['ok'])

    def test_readonly_original_snapshot_can_be_evicted_at_capacity(self):
        self.manager.max_entries = 1
        path = self.make_file()
        path.chmod(stat.S_IREAD)
        deleted = self.delete(path)
        snapshot = self.manager._entries[deleted['undo_token']].snapshot
        self.assertTrue(self.delete(self.make_file('next.png'))['ok'])
        self.assertFalse(snapshot.exists())
        self.assertFalse(self.manager.restore(deleted['undo_token'])['retryable'])

    def test_race_creating_destination_is_still_exclusive(self):
        path = self.make_file(content=b'original')
        deleted = self.delete(path)
        real_open = Path.open

        def raced_open(target, mode='r', *args, **kwargs):
            if target == path and mode == 'xb':
                path.write_bytes(b'arrived concurrently')
            return real_open(target, mode, *args, **kwargs)

        with patch.object(Path, 'open', new=raced_open):
            result = self.manager.restore(deleted['undo_token'])
        self.assertFalse(result['restored'])
        self.assertTrue(result['retryable'])
        self.assertEqual(path.read_bytes(), b'arrived concurrently')

    def test_invalid_token_cannot_select_arbitrary_file_or_leak_path(self):
        path = self.make_file()
        self.delete(path)
        for token in [str(path), '../image.png', '', None, {'path': str(path)}]:
            result = self.manager.restore(token)
            self.assertFalse(result['restored'])
            self.assertFalse(result['retryable'])
            self.assertEqual(result['path'], '')
        self.assertFalse(path.exists())

    def test_size_limit_preserves_original_without_trash_call(self):
        path = self.make_file(content=b'x' * 1025)
        result = self.delete(path)
        self.assertFalse(result['removed'])
        self.assertNotIn('undo_token', result)
        self.assertEqual(self.calls, [])
        self.assertTrue(path.exists())

    def test_space_or_snapshot_failure_preserves_original(self):
        path = self.make_file()
        with patch('core.image_trash_undo.shutil.disk_usage', return_value=shutil._ntuple_diskusage(100, 100, 0)):
            self.assertFalse(self.delete(path)['removed'])
        with patch('core.image_trash_undo.tempfile.TemporaryDirectory', side_effect=PermissionError()):
            self.manager.close()
            self.assertFalse(self.delete(path)['removed'])
        self.assertTrue(path.exists())
        self.assertEqual(self.calls, [])

    def test_trash_failure_does_not_offer_undo_and_removes_snapshot(self):
        path = self.make_file()
        failed = {'ok': False, 'removed': False, 'level': 'error', 'message': 'locked'}
        with patch('core.image_delete.move_to_trash_result', return_value=failed):
            result = image_delete_result(path.as_posix(), str(path), undo_manager=self.manager)
        self.assertNotIn('undo_token', result)
        self.assertTrue(path.exists())
        self.assertEqual(list(Path(self.manager._temporary.name).iterdir()), [])

    def test_unexpected_trash_exception_with_original_intact_cleans_snapshot(self):
        path = self.make_file()
        with patch('core.image_delete.move_to_trash_result', side_effect=RuntimeError('unexpected')), \
                self.assertLogs('core.image_trash_undo', level='ERROR'):
            result = image_delete_result(path.as_posix(), str(path), undo_manager=self.manager)
        self.assertFalse(result['removed'])
        self.assertTrue(path.exists())
        self.assertEqual(list(Path(self.manager._temporary.name).iterdir()), [])
        self.assertEqual(self.manager._bytes, 0)

    def test_unexpected_exception_after_move_keeps_backup_within_accounted_budget(self):
        path = self.make_file(content=b'original')

        def move_then_raise(local_path):
            self.fake_trash(local_path)
            raise RuntimeError('failed after moving')

        with patch('core.image_delete.move_to_trash_result', side_effect=move_then_raise), \
                self.assertLogs('core.image_trash_undo', level='ERROR'):
            result = image_delete_result(path.as_posix(), str(path), undo_manager=self.manager)
        self.assertFalse(result['removed'])
        self.assertNotIn('undo_token', result)
        self.assertEqual(self.manager._bytes, len(b'original'))
        self.assertEqual(len(self.manager._entries), 1)
        self.assertEqual(next(iter(self.manager._entries.values())).snapshot.read_bytes(), b'original')

    def test_cleanup_failure_is_reported_without_unbounded_new_snapshots(self):
        path = self.make_file()
        failed = {'ok': False, 'removed': False, 'level': 'error', 'message': 'locked'}
        with patch('core.image_delete.move_to_trash_result', return_value=failed), \
                patch.object(Path, 'unlink', side_effect=PermissionError()):
            result = image_delete_result(path.as_posix(), str(path), undo_manager=self.manager)
        self.assertFalse(result['removed'])
        self.assertTrue(path.exists())
        self.assertFalse(self.delete(path)['removed'])
        self.assertEqual(self.calls, [])

    def test_cleanup_failure_after_restore_still_reports_success_but_invalidates_token(self):
        path = self.make_file()
        deleted = self.delete(path)
        with patch.object(Path, 'unlink', side_effect=PermissionError()):
            result = self.manager.restore(deleted['undo_token'])
        self.assertTrue(result['restored'])
        self.assertTrue(path.exists())
        self.assertFalse(self.manager.restore(deleted['undo_token'])['retryable'])
        self.assertFalse(self.delete(path)['removed'])
        self.assertEqual(len(self.calls), 1)

    def test_entry_and_total_byte_limits_evict_oldest_tokens(self):
        tokens = [self.delete(self.make_file(f'{i}.png', b'x' * 300))['undo_token'] for i in range(4)]
        self.assertEqual(len(self.manager._entries), 3)
        self.assertEqual(self.manager._bytes, 900)
        expired = self.manager.restore(tokens[0])
        self.assertFalse(expired['restored'])
        self.assertFalse(expired['retryable'])
        large = self.delete(self.make_file('large.png', b'x' * 700))
        self.assertTrue(large['ok'])
        self.assertLessEqual(self.manager._bytes, 1024)
        self.assertTrue(self.manager.restore(tokens[-1])['restored'])
        self.assertTrue(self.manager.restore(large['undo_token'])['restored'])

    def test_missing_original_has_no_undo_token(self):
        path = self.root / 'already-gone.png'
        result = self.delete(path)
        self.assertTrue(result['removed'])
        self.assertFalse(result['ok'])
        self.assertNotIn('undo_token', result)
        self.assertEqual(self.calls, [])

    def test_unsupported_extension_is_refused_before_snapshot(self):
        path = self.make_file('notes.txt')
        self.assertFalse(self.delete(path)['removed'])
        self.assertEqual(self.calls, [])
        self.assertIsNone(self.manager._temporary)

    def test_explicit_media_whitelist_survives_restore(self):
        path = self.make_file('clip.mp4')
        result = self.delete(path, allowed_exts=frozenset({'.mp4'}))
        self.assertTrue(self.manager.restore(result['undo_token'])['restored'])

    def test_changed_source_during_snapshot_is_not_deleted(self):
        path = self.make_file()
        real_fsync = os.fsync

        def change_source(fd):
            path.write_bytes(b'something changed while we copied')
            real_fsync(fd)

        with patch('core.image_trash_undo.os.fsync', side_effect=change_source):
            result = self.delete(path)
        self.assertFalse(result['removed'])
        self.assertEqual(self.calls, [])
        self.assertTrue(path.exists())

    def test_partial_restore_failure_removes_only_partial_output_and_allows_retry(self):
        path = self.make_file(content=b'original bytes')
        deleted = self.delete(path)

        def partial_copy(reader, writer, **kwargs):
            writer.write(b'partial')
            raise OSError('disk full')

        with patch('core.image_trash_undo.shutil.copyfileobj', side_effect=partial_copy):
            result = self.manager.restore(deleted['undo_token'])
        self.assertFalse(result['restored'])
        self.assertTrue(result['retryable'])
        self.assertFalse(path.exists())
        self.assertTrue(self.manager.restore(deleted['undo_token'])['restored'])
        self.assertEqual(path.read_bytes(), b'original bytes')

    def test_snapshot_corruption_never_restores_bad_bytes(self):
        path = self.make_file()
        deleted = self.delete(path)
        self.manager._entries[deleted['undo_token']].snapshot.write_bytes(b'corrupted')
        result = self.manager.restore(deleted['undo_token'])
        self.assertFalse(result['restored'])
        self.assertFalse(result['retryable'])
        self.assertFalse(path.exists())

    def test_changed_parent_target_is_rejected(self):
        path = self.make_file()
        deleted = self.delete(path)
        with patch('core.image_trash_undo.resolve_missing_input_path', return_value=str(self.root / 'other.png')):
            result = self.manager.restore(deleted['undo_token'])
        self.assertFalse(result['restored'])
        self.assertTrue(result['retryable'])
        self.assertFalse(path.exists())

    def test_close_removes_only_owned_snapshot_directory_and_expires_tokens(self):
        path = self.make_file()
        deleted = self.delete(path)
        directory = Path(self.manager._temporary.name)
        unrelated = self.make_file('unrelated.png')
        self.manager.close()
        self.assertFalse(directory.exists())
        self.assertTrue(unrelated.exists())
        self.assertFalse(self.manager.restore(deleted['undo_token'])['retryable'])


if __name__ == '__main__':
    unittest.main()
