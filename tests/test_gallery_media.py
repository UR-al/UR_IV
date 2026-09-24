import os
import tempfile
import unittest
from pathlib import Path

from ui.vue_bridge import _scan_gallery_media


class TestGalleryMediaScan(unittest.TestCase):
    def test_includes_creator_media_and_ignores_unrelated_files(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            names = [
                'still.PNG', 'animated.webp', 'clip.mp4', 'preview.WEBM',
                'sound.wav', 'music.FLAC',
            ]
            for index, name in enumerate(names, start=1):
                path = root / name
                path.write_bytes(b'x')
                os.utime(path, (index, index))
            (root / 'notes.txt').write_text('ignore', encoding='utf-8')
            (root / 'fake.mp4').mkdir()

            result = _scan_gallery_media(temp_dir)

            self.assertEqual([Path(path).name for path in result], list(reversed(names)))

    def test_supports_animated_and_audio_formats(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            names = ['page.apng', 'animation.gif', 'voice.ogg', 'score.m4a', 'speech.opus']
            for name in names:
                (root / name).write_bytes(b'x')

            result_names = {Path(path).name for path in _scan_gallery_media(temp_dir)}

            self.assertEqual(result_names, set(names))

    def test_optional_creator_root_is_scanned_recursively(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            creator = root / 'creator'
            nested = creator / 'h3_v2v'
            nested.mkdir(parents=True)
            (nested / 'result.mp4').write_bytes(b'video')
            (root / 'regular.png').write_bytes(b'image')

            result_names = {
                Path(path).name
                for path in _scan_gallery_media(str(root), (str(creator),))
            }

            self.assertEqual(result_names, {'regular.png', 'result.mp4'})

    def test_versions_follow_the_files_and_change_when_a_file_is_overwritten(self):
        # 카드 썸네일 URL 버전 — 같은 경로에 덮어쓴 파일은 버전이 바뀌어야 브라우저가 새로 읽는다
        from ui.vue_bridge import _scan_gallery_media_versions
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            for index, name in enumerate(('old.png', 'new.png'), start=1):
                (root / name).write_bytes(b'x' * index)
                os.utime(root / name, (index * 100, index * 100))
            files, versions = _scan_gallery_media_versions(temp_dir)
            self.assertEqual([Path(p).name for p in files], ['new.png', 'old.png'])
            self.assertEqual(files, _scan_gallery_media(temp_dir))
            self.assertEqual(len(versions), len(files))
            self.assertTrue(all(versions) and len(set(versions)) == 2)
            # 같은 크기·같은 mtime 순서를 지키며 덮어써도(내용만 바뀜) mtime_ns 가 달라지면 버전이 바뀐다
            # (NTFS mtime 해상도는 100ns — 1ms 차이로 둔다)
            before = dict(zip(files, versions))
            (root / 'old.png').write_bytes(b'y')
            os.utime(root / 'old.png', ns=(100 * 10**9 + 10**6, 100 * 10**9 + 10**6))
            after = dict(zip(*_scan_gallery_media_versions(temp_dir)))
            old = next(p for p in files if p.endswith('old.png'))
            new = next(p for p in files if p.endswith('new.png'))
            self.assertNotEqual(after[old], before[old])
            self.assertEqual(after[new], before[new])


if __name__ == '__main__':
    unittest.main()
