"""Execute the real dispatcher branches without starting the GUI or touching user files."""
import ast
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest import mock

from core.path_safety import strip_file_url

ROOT = Path(__file__).resolve().parents[1]


def action_branch(name):
    tree = ast.parse((ROOT / 'ui/generator_main.py').read_text(encoding='utf-8'))
    handler = next(node for node in ast.walk(tree)
                   if isinstance(node, ast.FunctionDef) and node.name == '_handle_vue_action')
    for node in ast.walk(handler):
        if (isinstance(node, ast.If) and isinstance(node.test, ast.Compare)
                and isinstance(node.test.left, ast.Name) and node.test.left.id == 'action'
                and any(isinstance(value, ast.Constant) and value.value == name
                        for value in node.test.comparators)):
            return compile(ast.Module(body=node.body, type_ignores=[]), 'history-action', 'exec')
    raise AssertionError(f'Missing action: {name}')


class HistoryTrashActionTests(unittest.TestCase):
    def setUp(self):
        self.host = SimpleNamespace(vue_bridge=SimpleNamespace(
            imageDeleteResult=mock.Mock(), imageRestoreResult=mock.Mock(), showNotification=mock.Mock()),
            show_status=mock.Mock())
        self.bridge_module = SimpleNamespace(_GALLERY_MEDIA_EXTS=frozenset({'.png', '.mp4'}))

    def run_action(self, name, payload):
        with mock.patch.dict(sys.modules, {'ui.vue_bridge': self.bridge_module}):
            exec(action_branch(name), {'self': self.host, 'payload': payload,
                                       'json': json, '_clean_path': strip_file_url})

    def result(self, signal):
        return json.loads(getattr(self.host.vue_bridge, signal).emit.call_args.args[0])

    def test_history_delete_passes_manager_and_echoes_request_id(self):
        moved = {'path': 'D:/sample.png', 'ok': True, 'removed': True,
                 'undo_token': 'server-token', 'level': 'info', 'message': 'moved'}
        with mock.patch('core.image_trash_undo.ImageTrashUndoManager') as factory, \
                mock.patch('core.image_delete.image_delete_result', return_value=moved) as delete:
            self.run_action('delete_image', {'path': 'D:/sample.png', 'undoable': True, 'request_id': 'own-1'})
        factory.assert_called_once_with(max_entries=30)
        self.assertIs(delete.call_args.kwargs['undo_manager'], factory.return_value)
        self.assertIs(self.host._history_trash_undo, factory.return_value)
        self.assertEqual(self.result('imageDeleteResult')['undo_token'], 'server-token')
        self.assertEqual(self.result('imageDeleteResult')['request_id'], 'own-1')

    def test_gallery_delete_stays_on_existing_no_snapshot_path(self):
        moved = {'path': 'D:/sample.png', 'ok': True, 'removed': True, 'level': 'info', 'message': 'moved'}
        with mock.patch('core.image_trash_undo.ImageTrashUndoManager') as factory, \
                mock.patch('core.image_delete.image_delete_result', return_value=moved) as delete:
            self.run_action('delete_image', {'path': 'D:/sample.png', 'undoable': 'true'})
        factory.assert_not_called()
        self.assertIsNone(delete.call_args.kwargs['undo_manager'])
        self.assertNotIn('undo_token', self.result('imageDeleteResult'))

    def test_delete_failure_still_releases_frontend_request_without_claiming_removal(self):
        with mock.patch('core.image_delete.image_delete_result', side_effect=OSError('disk failed')):
            self.run_action('delete_image', {'path': 'D:/sample.png', 'request_id': 'own-2'})
        result = self.result('imageDeleteResult')
        self.assertEqual((result['ok'], result['removed'], result['request_id']), (False, False, 'own-2'))
        self.host.show_status.assert_not_called()

    def test_restore_uses_only_server_token_never_client_destination(self):
        manager = mock.Mock()
        manager.restore.return_value = {'path': 'D:/sample.png', 'undo_token': 'server-token',
                                        'ok': True, 'restored': True, 'level': 'info', 'message': 'restored'}
        self.host._history_trash_undo = manager
        self.run_action('restore_image', {'undo_token': 'server-token', 'request_id': 'undo-1',
                                          'path': 'C:/Windows/overwritten.png'})
        manager.restore.assert_called_once_with('server-token')
        self.assertEqual(self.result('imageRestoreResult')['path'], 'D:/sample.png')
        self.assertEqual(self.result('imageRestoreResult')['request_id'], 'undo-1')

    def test_expired_token_gets_a_terminal_correlated_result(self):
        self.run_action('restore_image', {'undo_token': 'expired', 'request_id': 'undo-2'})
        self.addCleanup(self.host._history_trash_undo.close)
        result = self.result('imageRestoreResult')
        self.assertEqual((result['restored'], result['retryable'], result['path']), (False, False, ''))
        self.assertEqual((result['undo_token'], result['request_id']), ('expired', 'undo-2'))

    def test_unexpected_restore_failure_gets_a_retryable_correlated_result(self):
        manager = self.host._history_trash_undo = mock.Mock()
        manager.restore.side_effect = OSError('disk failed')
        self.run_action('restore_image', {'undo_token': 'retry', 'request_id': 'undo-3'})
        result = self.result('imageRestoreResult')
        self.assertEqual((result['restored'], result['retryable'], result['request_id']), (False, True, 'undo-3'))
        self.assertEqual(result['undo_token'], 'retry')


if __name__ == '__main__':
    unittest.main()
