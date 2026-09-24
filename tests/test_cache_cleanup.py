"""이미지 캐시 정리 테스트.

실측(2026-07) image_cache 812MB / thumbs 96,641개 — 정리 루틴이 아예 없었다.
여기서 검증하는 핵심 안전장치: **undo 히스토리가 참조하는 최근 파일은 지우지 않는다.**
"""
import os
import shutil
import tempfile
import time
import unittest

from core.cache_cleanup import (
    SHARD_PREFIX_LEN,
    prune_by_total_size,
    prune_editor_temp,
    read_thumb_signature,
    shard_path,
    source_signature,
    thumb_is_stale,
    thumb_signature_comment,
)


class _TmpDir(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.mkdtemp(prefix='cachetest_')

    def tearDown(self):
        shutil.rmtree(self.dir, ignore_errors=True)

    def _touch(self, name, size=16, age_hours=0.0):
        path = os.path.join(self.dir, name)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'wb') as fh:
            fh.write(b'x' * size)
        if age_hours:
            when = time.time() - age_hours * 3600
            os.utime(path, (when, when))
        return path


class TestShardPath(_TmpDir):
    def test_prefix_split(self):
        p = shard_path('/base', 'abcdef123', '.jpg')
        self.assertEqual(os.path.basename(p), 'abcdef123.jpg')
        self.assertEqual(os.path.basename(os.path.dirname(p)),
                         'abcdef123'[:SHARD_PREFIX_LEN])

    def test_short_digest_does_not_crash(self):
        self.assertTrue(shard_path('/base', '', '.jpg').endswith('.jpg'))


class TestThumbIsStale(_TmpDir):
    """캐시 키가 경로@폭이라, 원본이 덮어써져도 옛 썸네일이 계속 나왔다."""

    def test_missing_thumb_is_stale(self):
        src = self._touch('a.png')
        self.assertTrue(thumb_is_stale(src, os.path.join(self.dir, 'thumb.jpg')))

    def _stamped_thumb(self, src, name='thumb.jpg'):
        """원본 서명을 주석에 담은 썸네일 JPEG(렌더러가 쓰는 모양)."""
        from PIL import Image
        path = os.path.join(self.dir, name)
        Image.new('RGB', (4, 4)).save(path, 'JPEG', comment=thumb_signature_comment(source_signature(src)))
        return path

    def test_overwritten_source_makes_thumb_stale(self):
        src = self._touch('a.png', age_hours=2)
        thumb = self._stamped_thumb(src)
        self.assertEqual(read_thumb_signature(thumb), source_signature(src))
        self.assertFalse(thumb_is_stale(src, thumb))
        self._touch('a.png')   # 에디터 '저장'이 사본을 갱신했다
        self.assertTrue(thumb_is_stale(src, thumb))

    def test_source_replaced_by_an_older_file_is_stale(self):
        # copy2·탐색기 덮어쓰기는 원본 mtime 을 유지한다 — 썸네일보다 옛 mtime 이어도 다른 파일이다
        src = self._touch('a.png', size=16)
        thumb = self._stamped_thumb(src)
        self._touch('a.png', size=32, age_hours=48)
        self.assertLess(os.stat(src).st_mtime_ns, os.stat(thumb).st_mtime_ns)
        self.assertTrue(thumb_is_stale(src, thumb))

    def test_thumb_without_a_signature_is_rebuilt_once(self):
        # 서명이 없는 예전 썸네일(순서 비교 시절)이나 깨진 파일은 다시 만든다
        src = self._touch('a.png', age_hours=2)
        self.assertTrue(thumb_is_stale(src, self._touch('legacy.jpg')))
        self.assertIsNone(read_thumb_signature(self._touch('junk.jpg')))

    def test_missing_source_keeps_existing_thumb(self):
        thumb = self._touch('thumb.jpg')
        self.assertFalse(thumb_is_stale(os.path.join(self.dir, 'gone.png'), thumb))


class TestPruneEditorTemp(_TmpDir):
    def test_keeps_recent_even_when_old(self):
        # undo 스택이 참조하는 최근 파일은 나이와 무관하게 보존돼야 한다
        for i in range(5):
            self._touch(f'edited_{i}.png', age_hours=100)
        removed = prune_editor_temp(self.dir, keep=5, max_age_hours=1)
        self.assertEqual(removed, 0)
        self.assertEqual(len(os.listdir(self.dir)), 5)

    def test_removes_old_beyond_keep(self):
        for i in range(10):
            self._touch(f'edited_{i:02d}.png', age_hours=100 - i)
        removed = prune_editor_temp(self.dir, keep=3, max_age_hours=1)
        self.assertEqual(removed, 7)
        self.assertEqual(len(os.listdir(self.dir)), 3)

    def test_recent_files_survive_age_filter(self):
        for i in range(10):
            self._touch(f'edited_{i:02d}.png', age_hours=0)
        # 전부 방금 만든 파일 → keep 초과분이어도 max_age 미달이라 안 지움
        self.assertEqual(prune_editor_temp(self.dir, keep=3, max_age_hours=24), 0)

    def test_under_keep_is_noop(self):
        self._touch('edited_a.png')
        self.assertEqual(prune_editor_temp(self.dir, keep=10), 0)

    def test_missing_dir_is_safe(self):
        self.assertEqual(prune_editor_temp(os.path.join(self.dir, 'nope'), keep=1), 0)

    def test_ignores_non_images(self):
        self._touch('notes.txt', age_hours=100)
        for i in range(5):
            self._touch(f'edited_{i}.png', age_hours=100)
        prune_editor_temp(self.dir, keep=0, max_age_hours=1)
        self.assertIn('notes.txt', os.listdir(self.dir))

    def test_keep_zero_removes_all_old(self):
        for i in range(4):
            self._touch(f'edited_{i}.png', age_hours=100)
        self.assertEqual(prune_editor_temp(self.dir, keep=0, max_age_hours=1), 4)


class TestPruneBySize(_TmpDir):
    def test_removes_oldest_until_under_limit(self):
        for i in range(10):
            self._touch(f'f{i:02d}.jpg', size=100, age_hours=100 - i)
        removed = prune_by_total_size(self.dir, max_bytes=500)
        self.assertGreaterEqual(removed, 5)
        total = sum(os.path.getsize(os.path.join(self.dir, f))
                    for f in os.listdir(self.dir))
        self.assertLessEqual(total, 500)

    def test_newest_survive(self):
        for i in range(5):
            self._touch(f'f{i}.jpg', size=100, age_hours=100 - i)
        prune_by_total_size(self.dir, max_bytes=200)
        remaining = sorted(os.listdir(self.dir))
        self.assertIn('f4.jpg', remaining)

    def test_under_limit_is_noop(self):
        self._touch('a.jpg', size=10)
        self.assertEqual(prune_by_total_size(self.dir, max_bytes=10_000), 0)

    def test_recursive_walks_shards(self):
        for i in range(6):
            self._touch(os.path.join(f'{i:02d}', f'{i:02d}abc.jpg'),
                        size=100, age_hours=100 - i)
        removed = prune_by_total_size(self.dir, max_bytes=250, recursive=True)
        self.assertGreaterEqual(removed, 3)


# (평면→샤드 이관 migrate_flat_to_sharded 와 그 테스트는 은퇴했다 — 캐시는 처음부터 샤딩된
#  image_cache/thumbs_v2 를 쓰고, 옛 폴더 정리는 tests/test_legacy_thumb_cache.py 가 검증한다.)


if __name__ == '__main__':
    unittest.main()
