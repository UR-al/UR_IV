import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from PyQt6.QtCore import QCoreApplication
from core.chat_schema_settings import save_schema_draft
from ui.vue_bridge import VueBridge


class ChatSchemaDraftTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QCoreApplication.instance() or QCoreApplication([])

    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.path = Path(temp.name) / 'ui_prefs.json'

    def test_saves_incomplete_blank_and_unicode_drafts_without_touching_other_settings(self):
        original = {'theme': 'light', 'aiAssistInstructions': {'common': 'keep'},
                    'chatSettingsV2': {'provider': 'lmstudio', 'systemPrompt': '내 지침', 'structuredEnabled': True}}
        self.path.write_text(json.dumps(original), encoding='utf-8')
        for text in ('{"type":', '', '미완성 😀'):
            self.assertEqual(save_schema_draft(text, self.path), text)
            saved = json.loads(self.path.read_text(encoding='utf-8'))
            self.assertEqual(saved['chatSettingsV2'], {**original['chatSettingsV2'], 'schemaText': text})
            self.assertEqual(saved['aiAssistInstructions'], original['aiAssistInstructions'])
            self.assertEqual(saved['theme'], 'light')

    def test_creates_missing_settings_file_and_accepts_character_limit(self):
        text = '한' * 64000
        save_schema_draft(text, self.path)
        self.assertEqual(json.loads(self.path.read_text(encoding='utf-8'))['chatSettingsV2']['schemaText'], text)
        original = self.path.read_bytes()
        for value in ('x' * 64001, None, {}, 123):
            with self.assertRaises(ValueError):
                save_schema_draft(value, self.path)
            self.assertEqual(self.path.read_bytes(), original)

    def test_corrupt_settings_are_never_replaced_by_a_draft(self):
        for original in ('broken', '[]', '{"chatSettingsV2":null}', '{"chatSettingsV2":[]}'):
            self.path.write_text(original, encoding='utf-8')
            with self.assertRaises(ValueError):
                save_schema_draft('{draft', self.path)
            self.assertEqual(self.path.read_text(encoding='utf-8'), original)

    def test_bridge_acknowledges_disk_write_not_merely_valid_json(self):
        bridge = VueBridge()
        with patch('core.chat_schema_settings.config_file', return_value=self.path):
            reply = json.loads(bridge.saveChatSchemaDraft('{unfinished'))
            self.assertEqual(reply, {'ok': True, 'schemaText': '{unfinished'})
            self.assertEqual(json.loads(self.path.read_text(encoding='utf-8'))['chatSettingsV2']['schemaText'], '{unfinished')
            self.assertFalse(json.loads(bridge.saveChatSchemaDraft('x' * 64001))['ok'])
            original = self.path.read_bytes()
            with patch('core.chat_schema_settings.save_ui_prefs', side_effect=PermissionError('denied')), patch('core.error_handler.handle_error'):
                self.assertFalse(json.loads(bridge.saveChatSchemaDraft('new draft'))['ok'])
            self.assertEqual(self.path.read_bytes(), original)

    def test_atomic_replace_failure_preserves_previous_file(self):
        save_schema_draft('original', self.path)
        original = self.path.read_bytes()
        with patch('core.config_migration.os.replace', side_effect=PermissionError('denied')):
            with self.assertRaises(PermissionError):
                save_schema_draft('new', self.path)
        self.assertEqual(self.path.read_bytes(), original)
