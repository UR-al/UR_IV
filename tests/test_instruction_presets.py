import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from PyQt6.QtCore import QCoreApplication
from core.ai_assist_instructions import normalize_instructions
from core.instruction_presets import list_presets, save_preset, delete_preset
from ui.vue_bridge import VueBridge


class InstructionPresetsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QCoreApplication.instance() or QCoreApplication([])

    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.path = Path(temp.name) / 'instruction_presets.json'

    def test_round_trip_scope_isolation_unicode_names_and_delete(self):
        chat = save_preset('chat', '태그 설명', '한국어로 설명', path=self.path)
        assist = normalize_instructions({'common': '보이는 것만', 'features': {'expand': '배경 유지'}})
        save_preset('assist', '태그 설명', assist, path=self.path)
        save_preset('chat', '수정본', '정확하게', preset_id=chat['id'], path=self.path)
        self.assertEqual(list_presets('chat', self.path)[0]['instructions'], '정확하게')
        self.assertEqual(list_presets('assist', self.path)[0]['instructions'], assist)
        delete_preset('chat', chat['id'], self.path)
        self.assertEqual(list_presets('chat', self.path), [])
        self.assertEqual(len(list_presets('assist', self.path)), 1)

    def test_invalid_duplicate_and_cross_scope_updates_do_not_write(self):
        item = save_preset('chat', 'one', 'original', path=self.path)
        original = self.path.read_bytes()
        for kwargs in (
            dict(scope='chat', name='ONE', instructions='duplicate'),
            dict(scope='chat', name='', instructions='empty name'),
            dict(scope='chat', name='two', instructions='x' * 32001),
            dict(scope='assist', name='two', instructions={}, preset_id=item['id']),
            dict(scope='assist', name='two', instructions={'features': {'unknown': 'x'}}),
            dict(scope='anything', name='two', instructions='x'),
        ):
            with self.subTest(kwargs=repr(kwargs)[:80]), self.assertRaises(ValueError):
                save_preset(**kwargs, path=self.path)
            self.assertEqual(self.path.read_bytes(), original)
        with self.assertRaises(ValueError):
            delete_preset('assist', item['id'], self.path)
        self.assertEqual(self.path.read_bytes(), original)

    def test_schema_presets_preserve_format_and_support_rename_update_delete(self):
        source = '{\n  "type": "object", "properties": {"tags": {"type": "string"}}\n}'
        save_preset('chat', '태그', 'existing chat', path=self.path)
        schema = save_preset('schema', '태그', source, path=self.path)
        self.assertEqual(list_presets('schema', self.path)[0]['instructions'], source)
        updated = save_preset('schema', '태그 설명', '{"type":"string"}', preset_id=schema['id'], path=self.path)
        self.assertEqual(updated['id'], schema['id'])
        self.assertEqual(list_presets('schema', self.path)[0]['name'], '태그 설명')
        delete_preset('schema', schema['id'], self.path)
        self.assertEqual(list_presets('schema', self.path), [])
        self.assertEqual(list_presets('chat', self.path)[0]['instructions'], 'existing chat')

    def test_invalid_schema_preset_does_not_overwrite_existing_presets(self):
        item = save_preset('schema', 'valid', '{"type":"object"}', path=self.path)
        original = self.path.read_bytes()
        for value in ('', '{unfinished', '{}', '{"$ref":"https://example.invalid/schema"}', '{"type":"invalid"}', 'x' * 64001, {}):
            with self.subTest(value=str(value)[:40]), self.assertRaises(ValueError):
                save_preset('schema', 'valid', value, preset_id=item['id'], path=self.path)
            self.assertEqual(self.path.read_bytes(), original)

    def test_corrupt_and_future_files_are_not_overwritten(self):
        for text in ('broken', '{"version":9,"presets":[]}', '{"version":1,"presets":[{}]}'):
            self.path.write_text(text, encoding='utf-8')
            with self.assertRaises(ValueError):
                save_preset('chat', 'new', 'do not lose existing', path=self.path)
            self.assertEqual(self.path.read_text(encoding='utf-8'), text)

    def test_failed_atomic_replace_preserves_original(self):
        save_preset('chat', 'one', 'original', path=self.path)
        original = self.path.read_bytes()
        with patch('utils.atomic_json.os.replace', side_effect=PermissionError('denied')):
            with self.assertRaises(PermissionError):
                save_preset('chat', 'two', 'new', path=self.path)
        self.assertEqual(self.path.read_bytes(), original)

    def test_bridge_crud_ack_events_and_live_instruction_independence(self):
        bridge = VueBridge()
        changed = []
        bridge.instructionPresetsChanged.connect(lambda raw: changed.append(json.loads(raw)))
        with patch('core.instruction_presets.config_file', return_value=self.path), patch('core.ai_assist_instructions.save_instructions') as save_live:
            response = json.loads(bridge.saveInstructionPreset(json.dumps({'scope': 'chat', 'name': '내 지침', 'instructions': 'chat only'})))
            self.assertTrue(response['ok'])
            self.assertEqual(len(changed), 1)
            self.assertEqual(json.loads(bridge.getInstructionPresets('chat'))['presets'][0]['instructions'], 'chat only')
            self.assertFalse(json.loads(bridge.deleteInstructionPreset(json.dumps({'scope': 'assist', 'id': response['preset']['id']})))['ok'])
            self.assertEqual(len(changed), 1)
            self.assertTrue(json.loads(bridge.deleteInstructionPreset(json.dumps({'scope': 'chat', 'id': response['preset']['id']})))['ok'])
            save_live.assert_not_called()

    def test_assist_save_broadcasts_only_after_success(self):
        bridge = VueBridge()
        events = []
        bridge.aiAssistInstructionsChanged.connect(lambda raw: events.append(json.loads(raw)))
        prefs = self.path.with_name('ui_prefs.json')
        with patch('core.ai_assist_instructions.config_file', return_value=prefs):
            self.assertTrue(json.loads(bridge.saveAiAssistInstructions(json.dumps({'common': 'shared', 'features': {}})))['ok'])
            self.assertEqual(events[0]['instructions']['common'], 'shared')
            self.assertFalse(json.loads(bridge.saveAiAssistInstructions('invalid'))['ok'])
            self.assertEqual(len(events), 1)

    def test_bridge_schema_presets_never_apply_live_settings(self):
        bridge = VueBridge()
        with patch('core.instruction_presets.config_file', return_value=self.path), patch('core.chat_schema_settings.save_schema_draft') as save_live:
            reply = json.loads(bridge.saveInstructionPreset(json.dumps({'scope': 'schema', 'name': '태그', 'instructions': '{"type":"object"}'})))
            self.assertTrue(reply['ok'])
            self.assertEqual(json.loads(bridge.getInstructionPresets('schema'))['presets'][0]['name'], '태그')
            save_live.assert_not_called()
