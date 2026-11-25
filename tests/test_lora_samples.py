import json
import os
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from comfy_custom_nodes.ai_studio_forge_parity import lora_samples as ls

RUN = "@작가 (v2) [x]_2.9b"


def _png(path: Path, size=(8, 8), color=(200, 0, 0)):
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", size, color).save(path)
    return path


class RunDirsTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self.addCleanup(self._tmp.cleanup)
        self.anima = self.root / "ANIMA"
        self.manifest = self.root / "_manifest.jsonl"
        self.weights = self.root / "D" / "grp"
        self.weights.mkdir(parents=True)
        self.full = self.weights / f"{RUN}-000001.safetensors"

    def test_run_dir_from_move_log_then_manifest_then_root(self):
        log_dir = self.root / "C" / "log_run"
        man_dir = self.root / "C" / "man_run"
        root_dir = self.anima / RUN
        for d in (log_dir, man_dir, root_dir):
            d.mkdir(parents=True)
        # 1) move_log only
        (self.weights / "move_log.jsonl").write_text(
            json.dumps({"at": "x", "src": str(log_dir / "a.safetensors")}) + "\n", encoding="utf-8")
        self.assertEqual(ls.run_dirs(self.full, RUN, self.anima, self.manifest)[0], log_dir)
        # 2) manifest: last line with same run wins
        (self.weights / "move_log.jsonl").unlink()
        lines = [{"run": RUN, "src": str(self.root / "gone")}, {"run": "other", "src": str(root_dir)},
                 {"run": RUN, "src": str(man_dir)}]
        self.manifest.write_text("\n".join(json.dumps(x) for x in lines) + "\nnot json\n", encoding="utf-8")
        got = ls.run_dirs(self.full, RUN, self.anima, self.manifest)
        self.assertEqual(got, [man_dir, root_dir])
        # 3) root only
        self.manifest.unlink()
        self.assertEqual(ls.run_dirs(self.full, RUN, self.anima, self.manifest), [root_dir])
        # nothing exists
        root_dir.rmdir()
        self.assertEqual(ls.run_dirs(self.full, RUN, self.anima, self.manifest), [])

    def test_logs_not_in_utf8_do_not_break_the_lookup(self):
        man_dir = self.root / "C" / "man_run"
        man_dir.mkdir(parents=True)
        root_dir = self.anima / RUN
        root_dir.mkdir(parents=True)
        # a move_log written as UTF-16 and a manifest with one cp949 line among UTF-8 ones
        (self.weights / "move_log.jsonl").write_text(json.dumps({"src": "C:\\x\\a.safetensors"}) + "\n", encoding="utf-16")
        cp949 = json.dumps({"run": "다른런", "src": "C:\\옛 폴더"}, ensure_ascii=False).encode("cp949")
        good = json.dumps({"run": RUN, "src": str(man_dir)}, ensure_ascii=False).encode("utf-8")
        self.manifest.write_bytes(cp949 + b"\n" + good + b"\n" + b"\xff\xfe{broken\n")
        self.assertEqual(ls.run_dirs(self.full, RUN, self.anima, self.manifest), [man_dir, root_dir])

    def test_a_later_manifest_line_to_a_gone_folder_does_not_hide_an_earlier_one(self):
        # 11-16: relocate lines (src = the old D:\LoRA\LoKr\<group>\<run>, gone) became the last line of 35 runs; the
        # training folder named by an earlier line must still be found, newest existing line first
        older = self.root / "C" / "train_run"
        newer = self.root / "C" / "train_run_again"
        older.mkdir(parents=True)
        newer.mkdir(parents=True)
        lines = [{"run": RUN, "src": str(older)}, {"run": RUN, "src": str(newer)},
                 {"run": RUN, "src": str(self.root / "D" / "LoKr" / "grp" / RUN), "relocate": True}]
        self.manifest.write_text("\n".join(json.dumps(x) for x in lines) + "\n", encoding="utf-8")
        self.assertEqual(ls.run_dirs(self.full, RUN, self.anima, self.manifest), [newer, older])

    def test_run_dirs_deduplicates(self):
        d = self.anima / RUN
        d.mkdir(parents=True)
        self.manifest.write_text(json.dumps({"run": RUN, "src": str(d)}) + "\n", encoding="utf-8")
        self.assertEqual(ls.run_dirs(self.full, RUN, self.anima, self.manifest), [d])


class EpochSamplesTest(unittest.TestCase):
    def test_epoch_samples_pick_latest_per_prompt(self):
        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp) / RUN
            s = d / "sample"
            old = _png(s / f"{RUN}_e000010_01_20261111021005_1.png")
            new = _png(s / f"{RUN}_e000010_01_20261111030000_1.png")
            _png(s / f"{RUN}_e000010_00_20261111021005_1.png")
            _png(s / f"{RUN}_e000001_02_20261111021005_1.png")  # other epoch
            _png(s / f"{RUN}_e000010_03_20261111021005_1.png")
            got = ls.epoch_samples([d], RUN, 10)
            self.assertEqual([p.name.split("_")[-3] for p in got], ["00", "01", "03"])
            self.assertIn(new, got)
            self.assertNotIn(old, got)
            self.assertFalse(any("e000001" in p.name for p in got))
            self.assertEqual(ls.epoch_samples([d], RUN, 7), [])

    def test_max_four_and_multiple_dirs(self):
        with tempfile.TemporaryDirectory() as tmp:
            a, b = Path(tmp) / "a", Path(tmp) / "b"
            for i in range(3):
                _png(a / "sample" / f"{RUN}_e000002_0{i}_20261111000000_1.png")
            for i in range(2, 6):
                _png(b / "sample" / f"{RUN}_e000002_0{i}_20261111000000_1.png")
            got = ls.epoch_samples([a, b], RUN, 2)
            self.assertEqual(len(got), 4)
            self.assertEqual([p.name.split("_")[-3] for p in got], ["00", "01", "02", "03"])


class BoardTest(unittest.TestCase):
    def test_board_strip_name_without_at(self):
        with tempfile.TemporaryDirectory() as tmp:
            board = Path(tmp)
            self.assertIsNone(ls.board_strip(board, "@abc d", 3))
            f = _png(board / "site" / "s" / "abc d" / "e003.webp", (40, 10))
            self.assertEqual(ls.board_strip(board, "@abc d", 3), f)
            self.assertIsNone(ls.board_strip(board, "@abc d", 4))

    def test_strip_tiles_horizontal_and_vertical(self):
        with tempfile.TemporaryDirectory() as tmp:
            h = _png(Path(tmp) / "h.png", (40, 10))
            v = _png(Path(tmp) / "v.png", (10, 40))
            self.assertEqual(ls.strip_tiles(h), [(0, 0, 10, 10), (10, 0, 20, 10), (20, 0, 30, 10), (30, 0, 40, 10)])
            self.assertEqual(ls.strip_tiles(v), [(0, 0, 10, 10), (0, 10, 10, 20), (0, 20, 10, 30), (0, 30, 10, 40)])


class FileEpochTest(unittest.TestCase):
    def test_final_file_uses_num_epochs(self):
        self.assertEqual(ls.file_epoch({"epoch": 5, "final": False, "num_epochs": 15}), 5)
        self.assertEqual(ls.file_epoch({"epoch": None, "final": True, "num_epochs": 15}), 15)
        self.assertIsNone(ls.file_epoch({"epoch": None, "final": False, "num_epochs": 15}))
        self.assertIsNone(ls.file_epoch({"epoch": None, "final": True, "num_epochs": None}))


class ThumbTest(unittest.TestCase):
    def test_concurrent_thumb_writes_same_target(self):
        import threading

        with tempfile.TemporaryDirectory() as tmp:
            src = _png(Path(tmp) / "s.png", (400, 300))
            dst = Path(tmp) / "out" / "t.webp"
            errors, start = [], threading.Barrier(8)

            def work():
                try:
                    start.wait(5)
                    ls.make_thumb(src, dst, 64)
                except Exception as exc:
                    errors.append(exc)

            threads = [threading.Thread(target=work) for _ in range(8)]
            for t in threads:
                t.start()
            for t in threads:
                t.join(20)
            self.assertEqual(errors, [])
            with Image.open(dst) as im:
                im.load()
            self.assertEqual(list(dst.parent.glob("*.tmp")), [])

    def test_thumb_is_cached(self):
        with tempfile.TemporaryDirectory() as tmp:
            src = _png(Path(tmp) / "s.png", (200, 100))
            dst = Path(tmp) / "out" / "t.webp"
            os.utime(src, (1000, 1000))
            self.assertEqual(ls.make_thumb(src, dst, 64), dst)
            with Image.open(dst) as im:
                self.assertEqual(im.size, (64, 32))
            os.utime(dst, (2000, 2000))
            ls.make_thumb(src, dst, 64)
            self.assertEqual(int(dst.stat().st_mtime), 2000)
            os.utime(src, (3000, 3000))  # src newer -> rebuilt
            ls.make_thumb(src, dst, 64)
            self.assertNotEqual(int(dst.stat().st_mtime), 2000)

    def test_thumb_refuses_a_huge_source_before_decoding(self):
        from unittest import mock

        from PIL import ImageFile

        with tempfile.TemporaryDirectory() as tmp:
            src = _png(Path(tmp) / "big.png", (50, 40))  # 2000 px
            dst = Path(tmp) / "t.webp"
            with mock.patch.object(ImageFile.ImageFile, "load", side_effect=AssertionError("pixels decoded")):
                with self.assertRaises(ValueError):
                    ls.make_thumb(src, dst, 16, max_pixels=1999)
            self.assertFalse(dst.exists())
            self.assertEqual(ls.make_thumb(src, dst, 16, max_pixels=2000), dst)  # at the limit: fine
            self.assertEqual(ls.MAX_COVER_PIXELS, 40_000_000)

    def test_thumb_crop(self):
        with tempfile.TemporaryDirectory() as tmp:
            src = _png(Path(tmp) / "s.png", (400, 100))
            dst = Path(tmp) / "t.webp"
            ls.make_thumb(src, dst, 50, crop=(0, 0, 100, 100))
            with Image.open(dst) as im:
                self.assertEqual(im.size, (50, 50))


if __name__ == "__main__":
    unittest.main()
