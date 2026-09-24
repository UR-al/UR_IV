"""옛 config/gallery_last_folder.txt 흡수 — 이 PC 에 없는 폴더는 받지 않는다(audit #106)."""
import os
import tempfile
import unittest
from unittest.mock import patch

from core import config_migration
from core.config_migration import read_legacy_gallery_folder


class ReadLegacyGalleryFolderTests(unittest.TestCase):
    def test_existing_folder_is_returned(self):
        with tempfile.TemporaryDirectory() as temp:
            txt = os.path.join(temp, 'gallery_last_folder.txt')
            with open(txt, 'w', encoding='utf-8') as f:
                f.write(f'  {temp}\n')
            self.assertEqual(read_legacy_gallery_folder(txt), temp)

    def test_other_pc_path_and_missing_file_are_ignored(self):
        with tempfile.TemporaryDirectory() as temp:
            txt = os.path.join(temp, 'gallery_last_folder.txt')
            with open(txt, 'w', encoding='utf-8') as f:
                f.write('C:/Users/KMJ/Desktop/AI zzal-does-not-exist')
            self.assertEqual(read_legacy_gallery_folder(txt), '')
            self.assertEqual(read_legacy_gallery_folder(os.path.join(temp, 'missing.txt')), '')

    def test_bom_and_file_path_instead_of_folder(self):
        with tempfile.TemporaryDirectory() as temp:
            txt = os.path.join(temp, 'gallery_last_folder.txt')
            with open(txt, 'w', encoding='utf-8-sig') as f:
                f.write(temp)
            self.assertEqual(read_legacy_gallery_folder(txt), temp)
            with open(txt, 'w', encoding='utf-8') as f:
                f.write(txt)   # 파일 경로는 폴더가 아니다
            self.assertEqual(read_legacy_gallery_folder(txt), '')


class GetLastGalleryFolderTests(unittest.TestCase):
    def _call(self, prefs, legacy):
        from ui.vue_bridge import VueBridge

        saved = []
        host = type('Host', (), {'_save_gallery_folder': lambda _self, folder: saved.append(folder)})()
        with patch.object(config_migration, 'load_ui_prefs', return_value=prefs), \
                patch.object(config_migration, 'read_legacy_gallery_folder', return_value=legacy):
            result = VueBridge.getLastGalleryFolder(host)
        return result, saved

    def test_invalid_legacy_path_is_not_persisted_and_falls_back_to_output(self):
        from config import OUTPUT_DIR

        result, saved = self._call({}, '')
        self.assertEqual(result, OUTPUT_DIR)
        self.assertEqual(saved, [], '없는 폴더를 ui_prefs 에 영구 저장하면 안 된다')

    def test_valid_legacy_folder_is_absorbed_once(self):
        with tempfile.TemporaryDirectory() as temp:
            result, saved = self._call({}, temp)
        self.assertEqual(result, temp)
        self.assertEqual(saved, [temp])

    def test_ui_prefs_value_wins(self):
        result, saved = self._call({'galleryFolder': 'D:/mine'}, 'D:/legacy')
        self.assertEqual((result, saved), ('D:/mine', []))


if __name__ == '__main__':
    unittest.main()
