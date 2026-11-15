import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from PIL import Image, PngImagePlugin

from comfy_custom_nodes.ai_studio_forge_parity import lora_previews as lp


def _png(path: Path, parameters=None, mtime=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    info = None
    if parameters is not None:
        info = PngImagePlugin.PngInfo()
        info.add_text("parameters", parameters)
    Image.new("RGB", (4, 4)).save(path, pnginfo=info)
    if mtime is not None:
        os.utime(path, (mtime, mtime))
    return path


class LoraPreviewsTests(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.out = Path(tmp.name) / "output"
        self.out.mkdir()
        self.cache = Path(tmp.name) / "user" / "previews.json"

    def _opens(self):
        return mock.patch.object(lp.Image, "open", wraps=lp.Image.open)

    def test_names_from_infotext(self):
        self.assertEqual(lp.lora_names_in_infotext("a <lora:x\\y-000010:0.8> b <lora:z:1>"), ["y-000010", "z"])
        self.assertEqual(lp.lora_names_in_infotext("<lora:a/b c (@d) 한글:1> <lora:a/b c (@d) 한글:0.5>"),
                         ["b c (@d) 한글"])
        self.assertEqual(lp.lora_names_in_infotext(""), [])

    def test_scan_reads_only_new_files(self):
        _png(self.out / "sub" / "a.png", "p <lora:x:1>", 1000)
        scan = lp.OutputScan(self.cache)
        with self._opens() as op:
            res = scan.update(self.out)
        self.assertEqual(op.call_count, 1)
        self.assertEqual(res["pending"], 0)
        self.assertEqual(res["latest"]["x"]["mtime"], 1000)
        with self._opens() as op:
            scan.update(self.out)
        self.assertEqual(op.call_count, 0)
        _png(self.out / "b.png", "<lora:y:1>", 2000)
        with self._opens() as op:
            res = scan.update(self.out)
        self.assertEqual(op.call_count, 1)
        self.assertEqual(set(res["latest"]), {"x", "y"})
        with self._opens() as op:  # a fresh instance reuses the cache file
            lp.OutputScan(self.cache).update(self.out)
        self.assertEqual(op.call_count, 0)

    def test_scan_skips_broken_and_textless_png(self):
        _png(self.out / "plain.png")
        (self.out / "broken.png").write_bytes(b"not a png")
        (self.out / "note.txt").write_text("<lora:t:1>", encoding="utf-8")
        _png(self.out / "ok.png", "<lora:ok:1>")
        res = lp.OutputScan(self.cache).update(self.out)
        self.assertEqual(list(res["latest"]), ["ok"])
        self.assertEqual(res["pending"], 0)

    def test_latest_for_picks_newest_across_epochs(self):
        _png(self.out / "a.png", "<lora:r-000005:1>", 1000)
        _png(self.out / "b.png", "<lora:r-000010:1>", 3000)
        _png(self.out / "c.png", "<lora:r-000005:1>", 2000)
        _png(self.out / "d.png", "<lora:other:1>", 9000)
        scan = lp.OutputScan(self.cache)
        scan.update(self.out)
        hit = scan.latest_for(["r-000005", "r-000010", "missing"])
        self.assertEqual(hit["path"], str(self.out / "b.png"))
        self.assertEqual(scan.latest_for(["r-000005"])["path"], str(self.out / "c.png"))
        self.assertIsNone(scan.latest_for(["nope"]))

    def test_time_limit_leaves_pending(self):
        for i in range(3):
            _png(self.out / f"{i}.png", f"<lora:s{i}:1>", 1000 + i)
        ticks = iter(range(0, 1000, 10))
        scan = lp.OutputScan(self.cache, clock=lambda: next(ticks))
        res = scan.update(self.out, limit_seconds=5)
        self.assertEqual(len(res["latest"]), 1)
        self.assertEqual(res["pending"], 2)
        self.assertEqual(list(res["latest"]), ["s2"])  # newest file first
        res = scan.update(self.out, limit_seconds=1000)
        self.assertEqual(res["pending"], 0)
        self.assertEqual(len(res["latest"]), 3)

    def test_xyz_grid_folders_are_not_scanned(self):
        # XYZ grids (output/AIStudio/xyz, up to ~110 MP) must never become automatic covers
        _png(self.out / "AIStudio" / "xyz" / "grid.png", "<lora:g:1>", 3000)
        _png(self.out / "AIStudio" / "XYZ" / "deep" / "grid2.png", "<lora:g2:1>", 3000)
        _png(self.out / "AIStudio" / "a.png", "<lora:a:1>", 1000)
        _png(self.out / "xyz_notes" / "b.png", "<lora:b:1>", 1000)  # only a folder named exactly "xyz" is skipped
        res = lp.OutputScan(self.cache).update(self.out)
        self.assertEqual(set(res["latest"]), {"a", "b"})

    def test_removed_files_drop_out(self):
        p = _png(self.out / "a.png", "<lora:x:1>")
        scan = lp.OutputScan(self.cache)
        scan.update(self.out)
        p.unlink()
        self.assertEqual(scan.update(self.out)["latest"], {})
        self.assertIsNone(scan.latest_for(["x"]))


if __name__ == "__main__":
    unittest.main()
