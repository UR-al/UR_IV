"""core.git_metadata — .git 파일만 읽는 표시용 Git 정보 (git 프로세스 없음)."""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from core.git_metadata import common_dir, git_dir, head_info, read_ref, remote_url, remote_urls


class GitMetadataTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.git = self.root / ".git"
        (self.git / "refs" / "heads").mkdir(parents=True)

    def test_detached_head_has_no_branch(self):
        (self.git / "HEAD").write_text("C" * 40 + "\n", encoding="ascii")
        self.assertEqual(("c" * 40, ""), head_info(self.root))

    def test_unsafe_or_foreign_refs_are_ignored(self):
        outside = self.root / "outside"
        outside.write_text("d" * 40, encoding="ascii")
        self.assertEqual("", read_ref(self.git, "refs/../../outside"))
        self.assertEqual("", read_ref(self.git, "HEAD"))
        (self.git / "HEAD").write_text("ref: refs/heads/../../../outside\n", encoding="ascii")
        self.assertEqual(("", ""), head_info(self.root))

    def test_missing_checkout_and_unreadable_head(self):
        self.assertIsNone(git_dir(self.root / "nowhere"))
        self.assertEqual(("", ""), head_info(self.root))   # HEAD 파일 없음

    def test_remote_urls_keep_every_remote(self):
        (self.git / "config").write_text(
            '[remote "origin"]\n\turl = https://example.invalid/a.git\n'
            '[remote "upstream"]\n\turl = https://github.com/UR-al/UR_IV.git\n'
            '[branch "main"]\n\tremote = origin\n',
            encoding="utf-8",
        )
        self.assertEqual(
            {"origin": "https://example.invalid/a.git", "upstream": "https://github.com/UR-al/UR_IV.git"},
            remote_urls(self.root),
        )
        self.assertEqual("https://example.invalid/a.git", remote_url(self.root))

    def test_common_dir_falls_back_to_the_git_dir(self):
        self.assertEqual(self.git, common_dir(self.git))
        (self.git / "commondir").write_text("does-not-exist\n", encoding="ascii")
        self.assertEqual(self.git, common_dir(self.git))


if __name__ == "__main__":
    unittest.main()
