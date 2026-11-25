"""LoRA browser user-folder store: settings, prefs, custom previews."""
import json
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from comfy_custom_nodes.ai_studio_forge_parity import lora_store


class LoraStoreTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.user = Path(self._tmp.name) / "user"
        self.dir = lora_store.store_dir(self.user)

    def test_store_dir_created(self):
        self.assertEqual(self.dir, self.user / "default" / "ai_studio")
        self.assertTrue(self.dir.is_dir())

    def test_settings_defaults_and_override(self):
        s = lora_store.load_settings(self.dir)
        self.assertEqual(s, {"anima_root": r"C:\Users\URL\Desktop\UR\ANIMA",
                             "board_root": r"C:\Users\URL\Desktop\UR\board", "thumb_size": 384})
        (self.dir / "lora_browser.json").write_text(
            json.dumps({"anima_root": r"X:\a", "thumb_size": 256, "bogus": 1}), encoding="utf-8")
        s = lora_store.load_settings(self.dir)
        self.assertEqual(s["anima_root"], r"X:\a")
        self.assertEqual(s["thumb_size"], 256)
        self.assertNotIn("bogus", s)
        self.assertEqual(s["board_root"], r"C:\Users\URL\Desktop\UR\board")

    def test_prefs_defaults(self):
        self.assertEqual(lora_store.load_prefs(self.dir), {
            "favorites": [], "recent": [], "sort": "name", "show_all": False,
            "tree_open": False, "close_after_add": False})

    def test_prefs_normalise_bad_values(self):
        out = lora_store.save_prefs(self.dir, {
            "recent": [f"s{i}" for i in range(25)], "sort": "x", "favorites": "nope",
            "show_all": "yes", "tree_open": True, "close_after_add": 1})
        self.assertEqual(len(out["recent"]), 20)
        self.assertEqual(out["recent"][0], "s0")
        self.assertEqual(out["sort"], "name")
        self.assertEqual(out["favorites"], [])
        self.assertIs(out["show_all"], False)
        self.assertIs(out["tree_open"], True)
        self.assertIs(out["close_after_add"], False)
        self.assertEqual(lora_store.load_prefs(self.dir), out)

    def test_custom_preview_round_trip(self):
        src = Path(self._tmp.name) / "a.png"
        Image.new("RGB", (8, 8), (200, 10, 10)).save(src)
        self.assertIsNone(lora_store.custom_preview(self.dir, "run1"))
        name = lora_store.set_custom_preview(self.dir, "run1", src, 4)
        self.assertTrue(name.endswith(".webp"))
        p = lora_store.custom_preview(self.dir, "run1")
        self.assertTrue(p.is_file())
        self.assertEqual(p.parent, self.dir / "previews")
        with Image.open(p) as im:
            self.assertLessEqual(max(im.size), 4)
        lora_store.clear_custom_preview(self.dir, "run1")
        self.assertIsNone(lora_store.custom_preview(self.dir, "run1"))
        self.assertFalse(p.exists())

    def test_store_never_writes_outside_dir(self):
        src = Path(self._tmp.name) / "src" / "a.png"
        src.parent.mkdir()
        Image.new("RGB", (8, 8)).save(src)
        before = {p for p in Path(self._tmp.name).rglob("*")}
        lora_store.save_prefs(self.dir, {"sort": "date"})
        lora_store.set_custom_preview(self.dir, "../../evil", src, 4)
        new = {p for p in Path(self._tmp.name).rglob("*") if p.is_file()} - before
        self.assertTrue(new)
        for p in new:
            self.assertTrue(p.resolve().is_relative_to(self.dir.resolve()), p)


if __name__ == "__main__":
    unittest.main()
