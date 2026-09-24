"""원자적 JSON 쓰기 사본들을 utils.atomic_json 한 벌로 통일한 호출처 회귀 테스트.

각 호출처가 (1) 공용 구현으로 쓰고 (2) 교체 실패 시 원본을 보존하며 tmp 를 남기지 않는지 본다.
"""
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock


class PromptOrderAtomicSaveTests(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        self.path = Path(self.dir.name) / 'config' / 'prompt_order.json'
        patcher = mock.patch('core.prompt_order._config_path', return_value=self.path)
        patcher.start()
        self.addCleanup(patcher.stop)
        self.addCleanup(self.dir.cleanup)

    def test_save_creates_folder_and_roundtrips_cleaned_order(self):
        from core import prompt_order
        self.assertTrue(prompt_order.save_order(['main', 'bogus', 'main', 'artist']))
        data = json.loads(self.path.read_text(encoding='utf-8'))
        self.assertEqual(data['version'], 1)
        self.assertEqual(data['order'][:2], ['main', 'artist'])
        self.assertEqual(sorted(data['order']), sorted(prompt_order.DEFAULT_ORDER))
        self.assertEqual(prompt_order.load_order(), data['order'])
        self.assertFalse(Path(str(self.path) + '.tmp').exists())

    def test_load_does_not_create_config_folder(self):
        from core import prompt_order
        self.assertEqual(prompt_order.load_order(), prompt_order.DEFAULT_ORDER)
        self.assertFalse(self.path.parent.exists())

    def test_failed_replace_keeps_previous_order(self):
        from core import prompt_order
        prompt_order.save_order(['suffix'])
        before = self.path.read_bytes()
        with mock.patch('utils.atomic_json.os.replace', side_effect=PermissionError('locked')):
            self.assertFalse(prompt_order.save_order(['prefix']))
        self.assertEqual(self.path.read_bytes(), before)
        self.assertFalse(Path(str(self.path) + '.tmp').exists())

    def test_non_object_file_falls_back_to_default(self):
        from core import prompt_order
        self.path.parent.mkdir(parents=True)
        self.path.write_text('["main"]', encoding='utf-8')
        self.assertEqual(prompt_order.load_order(), prompt_order.DEFAULT_ORDER)


class ConfigMigrationSaveTests(unittest.TestCase):
    def test_save_with_version_is_atomic_and_versioned(self):
        from core.config_migration import load_and_migrate, save_with_version
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, 'sub', 'prefs.json')
            save_with_version(path, {'a': 1}, 3)
            self.assertEqual(load_and_migrate(path, {}, 3), {'a': 1, 'schema_version': 3})
            with mock.patch('utils.atomic_json.os.replace', side_effect=OSError('locked')):
                with self.assertRaises(OSError):
                    save_with_version(path, {'a': 2}, 3)
            with open(path, encoding='utf-8') as f:
                self.assertEqual(json.load(f)['a'], 1)
            self.assertFalse(os.path.exists(path + '.tmp'))


class GenStatsSaveTests(unittest.TestCase):
    def test_record_persists_compact_json_through_shared_writer(self):
        from core.gen_stats import GenStats
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, 'stats', 'generation.json')
            stats = GenStats(path)
            with mock.patch('core.gen_stats.atomic_write_json',
                            wraps=__import__('utils.atomic_json', fromlist=['x']).atomic_write_json) as writer:
                stats.record({'success': True, 'model': 'm'})
            writer.assert_called_once()
            self.assertIsNone(writer.call_args.kwargs.get('indent'))
            with open(path, encoding='utf-8') as f:
                text = f.read()
            self.assertNotIn('\n', text)
            self.assertEqual(json.loads(text)[0]['model'], 'm')
            self.assertEqual(GenStats(path)._records[0]['model'], 'm')

    def test_save_failure_is_logged_not_raised(self):
        from core.gen_stats import GenStats
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, 'generation.json')
            stats = GenStats(path)
            with mock.patch('utils.atomic_json.os.replace', side_effect=OSError('locked')):
                stats.record({'success': False, 'model': 'x'})  # 예외 없이 넘어가야 한다
            self.assertFalse(os.path.exists(path + '.tmp'))


class ComicStudioAtomicWriteTests(unittest.TestCase):
    def test_document_write_uses_writing_suffix_and_cleans_up_on_failure(self):
        from core.comic_studio import ComicStudio
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'comic.json'
            studio = ComicStudio(path)
            document = studio.plan('첫 장면. 두 번째 장면.', 2)
            studio.save(document)
            before = path.read_bytes()
            self.assertTrue(before.endswith(b'}'))
            self.assertNotIn(b'\r\n', before)  # 바이트 그대로 기록 — CRLF 변환 없음
            with mock.patch('utils.atomic_json.os.replace', side_effect=PermissionError('locked')):
                with self.assertRaises(PermissionError):
                    studio._write_document_unlocked(document)
            self.assertEqual(path.read_bytes(), before)
            self.assertFalse(path.with_suffix('.json.writing').exists())


class CaptionAtomicWriteTests(unittest.TestCase):
    def test_caption_sidecar_keeps_exact_newlines(self):
        from ui.vue_bridge import VueBridge
        with tempfile.TemporaryDirectory() as tmp:
            target = os.path.join(tmp, 'a.txt')
            VueBridge._write_caption_atomic(target, 'tag1, tag2\nline2')
            with open(target, 'rb') as f:
                self.assertEqual(f.read(), b'tag1, tag2\nline2')
            with mock.patch('utils.atomic_json.os.replace', side_effect=PermissionError('locked')):
                with self.assertRaises(PermissionError):
                    VueBridge._write_caption_atomic(target, 'new')
            with open(target, 'rb') as f:
                self.assertEqual(f.read(), b'tag1, tag2\nline2')
            self.assertFalse(os.path.exists(target + '.tmp'))


if __name__ == '__main__':
    unittest.main()
