"""LoRA picker HTTP routes: pure handlers (no server), fake folder_paths, temp dirs only."""
import json
import os
import struct
import tempfile
import threading
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock
from urllib.parse import parse_qsl, quote

from PIL import Image

from comfy_custom_nodes.ai_studio_forge_parity import lora_routes, lora_samples, studio_routes

GROUP = "@grp (@grp)"
RUN = "run_a_2.9b_b10"
KEY = f"LoKr/{GROUP}/{RUN}"


def _entry(epoch=None):
    stem = RUN if epoch is None else f"{RUN}-{epoch:06d}"
    return f"LoKr\\{GROUP}\\{RUN}\\{stem}.safetensors"


def _safetensors(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    head = json.dumps({"__metadata__": {"ss_num_epochs": "10"}}).encode("utf-8")
    path.write_bytes(struct.pack("<Q", len(head)) + head)


def _png(path: Path, size=(40, 40), color=(200, 30, 30)):
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", size, color).save(path)
    return path


class FakeScan:
    updates = 0
    gate = None
    hit = None

    def __init__(self, cache_path):
        self.cache_path = cache_path

    def update(self, output_dir, limit_seconds=20.0):
        type(self).updates += 1
        if type(self).gate is not None:
            type(self).gate.wait(5)
        return {}

    def latest_for(self, stems):
        return type(self).hit


class LoraRoutesBase(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)
        self.loras = self.root / "loras"
        self.user = self.root / "user"
        self.output = self.root / "output"
        self.input = self.root / "input"
        self.anima = self.root / "ANIMA"
        self.board = self.root / "board"
        for d in (self.output, self.input, self.anima, self.board):
            d.mkdir()
        self.entries = [_entry(5), _entry(10), _entry()]
        for entry in self.entries:
            _safetensors(self.loras / entry.replace("\\", "/"))
        settings_dir = self.user / "default" / "ai_studio"
        settings_dir.mkdir(parents=True)
        (settings_dir / "lora_browser.json").write_text(
            json.dumps({"anima_root": str(self.anima), "board_root": str(self.board), "thumb_size": 64}),
            encoding="utf-8")
        self.store = settings_dir
        self.fp = self._folder_paths(lambda: list(self.entries))
        FakeScan.updates, FakeScan.gate, FakeScan.hit = 0, None, None
        self.addCleanup(lambda: (lora_routes._SCANS.clear(), setattr(lora_routes, "_SCAN_THREAD", None)))
        lora_routes._SCANS.clear()
        lora_routes._SCAN_THREAD = None
        for target in (mock.patch.object(lora_routes, "folder_paths_module", lambda: self.fp),
                       mock.patch.object(studio_routes, "folder_paths_module", lambda: self.fp),
                       mock.patch.object(lora_routes, "OutputScan", FakeScan)):
            target.start()
            self.addCleanup(target.stop)

    def _folder_paths(self, names):
        def full_path(kind, name):
            path = self.loras / name.replace("\\", "/")
            return str(path) if path.is_file() else None

        return SimpleNamespace(
            get_filename_list=lambda kind: names(), get_full_path=full_path,
            get_folder_paths=lambda kind: [str(self.loras)],
            get_user_directory=lambda: str(self.user), get_output_directory=lambda: str(self.output),
            get_directory_by_type=lambda kind: str({"output": self.output, "input": self.input}.get(kind, "")) or None)

    def sample(self, epoch, index, color=(10, 200, 10)):
        return _png(self.anima / RUN / "sample" / f"{RUN}_e{epoch:06d}_{index:02d}_20261111000000_1.png",
                    color=color)

    def write_board(self, stars=(5,), prompts=("a", "b")):
        (self.board / "board_db.json").write_text(json.dumps({
            "runs": {RUN: {"trigger": "@trig", "prompts": [{"label": p, "text": p} for p in prompts]}},
            "marks": {RUN: {"stars": list(stars)}}}), encoding="utf-8")

    def join_scan(self):
        thread = lora_routes._SCAN_THREAD
        if thread is not None:
            thread.join(5)


class IndexTests(LoraRoutesBase):
    def test_index_payload_shape(self):
        self.write_board()
        status, body = lora_routes.handle_index({})
        self.join_scan()
        self.assertEqual(status, 200)
        self.assertEqual(sorted(body), ["board_stars", "cover_versions", "items", "prefs", "prompt_labels"])
        self.assertEqual(body["cover_versions"], {})
        self.assertEqual(len(body["items"]), 3)
        self.assertEqual({i["run_key"] for i in body["items"]}, {KEY})
        self.assertEqual(body["prefs"]["sort"], "name")
        self.assertEqual(body["board_stars"], {RUN: [5]})
        self.assertEqual(body["prompt_labels"], {RUN: ["a", "b"]})
        self.assertTrue((self.store / "lora_index.json").is_file())

    def test_index_starts_one_background_scan(self):
        FakeScan.gate = threading.Event()
        lora_routes.handle_index({})
        lora_routes.handle_index({"fresh": "1"})
        FakeScan.gate.set()
        self.join_scan()
        self.assertEqual(FakeScan.updates, 1)
        lora_routes.handle_index({})  # the first scan has finished: a new one may start
        self.join_scan()
        self.assertEqual(FakeScan.updates, 2)


class FakeComfyList:
    """ComfyUI-like ``get_filename_list``: the list is cached under ``filename_list_cache["loras"]`` and kept while the folder
    mtimes look unchanged (exFAT does not bump them for a new file); a folder that vanished makes the cached check raise."""

    def __init__(self, names):
        self.names = names
        self.filename_list_cache = {}
        self.broken = False  # the drive is gone: every listing fails
        self.walks = 0

    def get_filename_list(self, kind):
        if self.broken:
            raise OSError("[WinError 21] The device is not ready: 'D:\\LoRA'")
        cached = self.filename_list_cache.get(kind)
        if cached is not None:
            if cached[1].get("vanished"):
                raise FileNotFoundError("[WinError 3] The system cannot find the path specified: 'D:\\LoRA\\gone'")
            return list(cached[0])
        self.walks += 1
        self.filename_list_cache[kind] = (self.names(), {}, 0.0)
        return list(self.filename_list_cache[kind][0])

    def vanish(self):
        self.filename_list_cache["loras"][1]["vanished"] = True


class ListCacheTests(LoraRoutesBase):
    """ComfyUI's LoRA list cache: dropped by the refresh button (fresh) and after an OSError, never a 500."""

    def setUp(self):
        super().setUp()
        lora_routes.reset_index_cache()
        self.addCleanup(lora_routes.reset_index_cache)
        self.comfy = FakeComfyList(lambda: list(self.entries))
        self.fp.get_filename_list = self.comfy.get_filename_list
        self.fp.filename_list_cache = self.comfy.filename_list_cache

    def test_fresh_drops_comfyui_list_cache_so_a_new_epoch_shows(self):
        self.assertEqual(len(lora_routes.handle_index({})[1]["items"]), 3)
        new = _entry(15)
        _safetensors(self.loras / new.replace("\\", "/"))
        self.entries.append(new)
        self.assertEqual(len(lora_routes.handle_index({})[1]["items"]), 3)  # ComfyUI's cache still answers
        status, body = lora_routes.handle_index({"fresh": "1"})
        self.join_scan()
        self.assertEqual(status, 200)
        self.assertIn(new, [i["name"] for i in body["items"]])
        self.assertNotIn("error", body)

    def test_vanished_folder_lists_again_instead_of_500(self):
        lora_routes.handle_index({})
        gone = self.entries.pop(0)
        (self.loras / gone.replace("\\", "/")).unlink()
        self.comfy.vanish()
        with self.assertLogs("ai_studio.lora_routes", "WARNING"):
            status, body = lora_routes.handle_index({})
        self.assertEqual(status, 200)
        self.assertEqual(sorted(i["name"] for i in body["items"]), sorted(self.entries))
        self.assertNotIn("error", body)
        self.comfy.vanish()
        with self.assertLogs("ai_studio.lora_routes", "WARNING"):
            self.assertEqual(lora_routes.handle_detail({"name": self.entries[0]})[0], 200)
        self.comfy.vanish()
        with self.assertLogs("ai_studio.lora_routes", "WARNING"):
            self.assertEqual(lora_routes.thumb_response({"name": gone, "kind": "cover"})[0], 404)
        self.join_scan()

    def test_unreadable_drive_gives_an_empty_list_and_a_readable_error(self):
        self.comfy.broken = True
        with self.assertLogs("ai_studio.lora_routes", "WARNING"):
            status, body = lora_routes.handle_index({})
        self.join_scan()
        self.assertEqual(status, 200)
        self.assertEqual(body["items"], [])
        self.assertIsInstance(body.get("error"), str)
        self.assertIn("device is not ready", body["error"])
        with self.assertLogs("ai_studio.lora_routes", "WARNING"):
            self.assertEqual(lora_routes.handle_detail({"name": self.entries[0]})[0], 404)
        with self.assertLogs("ai_studio.lora_routes", "WARNING"):
            self.assertEqual(lora_routes.thumb_response({"name": self.entries[0], "kind": "cover"})[0], 404)

    def test_a_folder_paths_without_the_cache_still_works(self):
        del self.fp.filename_list_cache
        self.assertEqual(len(lora_routes.handle_index({"fresh": "1"})[1]["items"]), 3)
        self.join_scan()


class IndexReuseTests(LoraRoutesBase):
    def setUp(self):
        super().setUp()
        self.now = [100.0]
        patch = mock.patch.object(lora_routes, "_clock", lambda: self.now[0])
        patch.start()
        self.addCleanup(patch.stop)
        lora_routes.reset_index_cache()
        self.addCleanup(lora_routes.reset_index_cache)
        real = lora_routes.LoraIndex.build
        self.builds = mock.patch.object(lora_routes.LoraIndex, "build", autospec=True, side_effect=real)
        self.spy = self.builds.start()
        self.addCleanup(self.builds.stop)

    def test_thumb_reuses_recent_index(self):
        self.sample(10, 0)
        query = {"name": self.entries[1], "kind": "sample", "epoch": "10", "i": "0"}
        self.assertEqual(lora_routes.thumb_response(query)[0], 200)
        self.assertEqual(lora_routes.thumb_response(query)[0], 200)
        self.assertEqual(self.spy.call_count, 1)
        self.now[0] += lora_routes.INDEX_TTL_SECONDS + 1
        self.assertEqual(lora_routes.thumb_response(query)[0], 200)
        self.assertEqual(self.spy.call_count, 2)

    def test_index_route_always_rebuilds(self):
        lora_routes.handle_index({})
        lora_routes.handle_index({})
        self.join_scan()
        self.assertEqual(self.spy.call_count, 2)

    def test_detail_and_preview_share_the_cached_index(self):
        lora_routes.handle_detail({"name": self.entries[0]})
        lora_routes.handle_preview({"run_key": KEY, "clear": True})
        self.assertEqual(self.spy.call_count, 1)

    def test_concurrent_cold_callers_share_one_rebuild(self):
        errors = []

        def work():
            try:
                lora_routes.handle_detail({"name": self.entries[0]})
            except Exception as exc:  # pragma: no cover
                errors.append(exc)

        threads = [threading.Thread(target=work) for _ in range(6)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(10)
        self.assertEqual(errors, [])
        self.assertEqual(self.spy.call_count, 1)


class DetailTests(LoraRoutesBase):
    def test_detail_lists_epochs_of_the_run_with_stars(self):
        self.write_board(stars=(5,))
        self.sample(10, 0)
        self.sample(10, 1)
        status, body = lora_routes.handle_detail({"name": self.entries[0]})
        self.assertEqual(status, 200)
        self.assertEqual(body["item"]["name"], self.entries[0])
        self.assertEqual(body["trigger"], "@trig")
        self.assertEqual([(e["epoch"], e["samples"], e["starred"]) for e in body["epochs"]],
                         [(5, 0, True), (10, 2, False)])
        self.assertEqual(body["epochs"][1]["name"], self.entries[1])
        self.assertEqual(body["prompt_labels"], ["a", "b"])
        self.assertEqual(body["cover"], "sample")
        self.assertIn("summary", body)

    def test_unknown_or_escaping_names_are_404(self):
        for name in ("..\\..\\x", "C:\\x", "../x.safetensors", "nope.safetensors", "", None):
            self.assertEqual(lora_routes.handle_detail({"name": name})[0], 404, name)
            self.assertEqual(lora_routes.thumb_response({"name": name, "kind": "cover"})[0], 404, name)
        self.assertEqual(lora_routes.handle_preview({"run_key": "LoKr/x/..", "clear": True})[0], 404)

    def test_unicode_names_round_trip(self):
        names = ["LoKr\\묵밥 (@Bjunbae)\\r_2.9b\\r_2.9b-000001.safetensors",
                 "LoKr\\໊ (@9an9ta)\\r_2.9b\\r_2.9b-000001.safetensors"]
        for name in names:
            _safetensors(self.loras / name.replace("\\", "/"))
        self.entries.extend(names)
        for name in names:
            query = dict(parse_qsl("name=" + quote(name), keep_blank_values=True))
            status, body = lora_routes.handle_detail(query)
            self.assertEqual(status, 200, name)
            self.assertEqual(body["item"]["name"], name)


class ThumbTests(LoraRoutesBase):
    def test_sample_thumb_falls_back_to_board_strip_tile(self):
        colors = [(250, 0, 0), (0, 250, 0), (0, 0, 250), (250, 250, 0)]
        strip = Image.new("RGB", (400, 100))
        for n, color in enumerate(colors):
            strip.paste(Image.new("RGB", (100, 100), color), (n * 100, 0))
        path = self.board / "site" / "s" / RUN / "e010.webp"
        path.parent.mkdir(parents=True)
        strip.save(path, "WEBP", lossless=True)
        query = {"name": self.entries[1], "kind": "sample", "epoch": "10", "i": "2"}
        status, thumb = lora_routes.thumb_response(query)
        self.assertEqual(status, 200)
        with Image.open(thumb) as im:
            self.assertEqual(im.size, (64, 64))  # thumb_size 64
            self.assertGreater(im.convert("RGB").getpixel((50, 50))[2], 200)
        status, other = lora_routes.thumb_response({**query, "i": "3"})
        self.assertNotEqual(thumb, other)  # a different tile never reuses a thumbnail name
        self.assertEqual(lora_routes.thumb_response({**query, "i": "4"})[0], 404)
        self.assertEqual(lora_routes.thumb_response({**query, "i": "x"})[0], 400)
        self.assertEqual(lora_routes.thumb_response({**query, "epoch": "7"})[0], 404)

    def test_training_sample_wins_over_board_strip(self):
        self.sample(10, 0, color=(0, 250, 0))
        status, thumb = lora_routes.thumb_response(
            {"name": self.entries[1], "kind": "sample", "epoch": "10", "i": "0"})
        self.assertEqual(status, 200)
        with Image.open(thumb) as im:
            self.assertGreater(im.convert("RGB").getpixel((5, 5))[1], 200)

    def test_thumb_falls_back_custom_sample_output(self):
        cover = {"name": self.entries[0], "kind": "cover"}
        self.assertEqual(lora_routes.thumb_response(cover)[0], 404)
        self.assertIsNone(lora_routes.handle_detail({"name": self.entries[0]})[1]["cover"])
        out = _png(self.output / "gen.png", (50, 50), (0, 0, 250))
        FakeScan.hit = {"path": str(out), "mtime": 1.0}
        status, thumb = lora_routes.thumb_response(cover)
        self.assertEqual(status, 200)
        self.assertEqual(lora_routes.handle_detail({"name": self.entries[0]})[1]["cover"], "output")
        self.sample(10, 0)  # last epoch sample 0 beats the output image
        status, from_sample = lora_routes.thumb_response(cover)
        self.assertNotEqual(from_sample, thumb)
        self.assertEqual(lora_routes.handle_detail({"name": self.entries[0]})[1]["cover"], "sample")
        _png(self.output / "pick.png", (30, 30), (250, 0, 250))
        status, body = lora_routes.handle_preview(
            {"run_key": KEY, "image": {"filename": "pick.png", "subfolder": "", "type": "output"}})
        self.assertEqual((status, body["cover"]), (200, "custom"))
        status, custom = lora_routes.thumb_response(cover)
        self.assertEqual(status, 200)
        self.assertEqual(custom.parent.name, "previews")
        self.assertEqual(lora_routes.handle_detail({"name": self.entries[0]})[1]["cover"], "custom")
        self.assertEqual(lora_routes.handle_preview({"run_key": KEY, "clear": True})[0], 200)
        self.assertEqual(lora_routes.handle_detail({"name": self.entries[0]})[1]["cover"], "sample")

    def test_huge_output_image_is_not_a_cover(self):
        out = _png(self.output / "big.png", (50, 40), (0, 0, 250))  # 2000 px: over a lowered limit
        FakeScan.hit = {"path": str(out), "mtime": 1.0}
        cover = {"name": self.entries[0], "kind": "cover"}
        with mock.patch.object(lora_samples, "MAX_COVER_PIXELS", 1999):
            self.assertEqual(lora_routes.thumb_response(cover)[0], 404)
            self.assertIsNone(lora_routes.handle_detail({"name": self.entries[0]})[1]["cover"])
        self.assertEqual(lora_routes.thumb_response(cover)[0], 200)

    def test_cover_version_changes_when_a_custom_preview_is_set_or_cleared(self):
        _png(self.output / "pick.png", (30, 30), (250, 0, 250))
        ref = {"filename": "pick.png", "subfolder": "", "type": "output"}
        self.assertEqual(lora_routes.handle_index({})[1]["cover_versions"], {})
        self.assertEqual(lora_routes.handle_preview({"run_key": KEY, "image": ref})[0], 200)
        first = lora_routes.handle_index({})[1]["cover_versions"]
        self.assertEqual(list(first), [KEY])
        self.assertIsInstance(first[KEY], int)
        self.assertLess(first[KEY], 2 ** 53)  # exact in a JS number
        preview = next((self.store / "previews").iterdir())
        os.utime(preview, (1_000_000_000, 1_000_000_000))
        old = lora_routes.handle_index({})[1]["cover_versions"][KEY]
        self.assertEqual(lora_routes.handle_preview({"run_key": KEY, "image": ref})[0], 200)  # set again
        self.assertNotEqual(lora_routes.handle_index({})[1]["cover_versions"][KEY], old)
        self.assertEqual(lora_routes.handle_preview({"run_key": KEY, "clear": True})[0], 200)
        self.assertEqual(lora_routes.handle_index({})[1]["cover_versions"], {})
        self.join_scan()

    def test_corrupt_sample_is_404_not_500(self):
        bad = self.anima / RUN / "sample" / f"{RUN}_e000010_00_20261111000000_1.png"
        bad.parent.mkdir(parents=True)
        bad.write_bytes(b"not a png")
        with self.assertLogs("ai_studio.lora_routes", "WARNING"):
            status, body = lora_routes.thumb_response({"name": self.entries[1], "kind": "sample", "epoch": "10"})
        self.assertEqual(status, 404)
        self.assertIn("error", body)


class PreviewAndPrefsTests(LoraRoutesBase):
    def test_preview_rejects_images_outside_output_input(self):
        outside = _png(self.root / "secret.png")
        for ref in ({"filename": "..\\secret.png", "subfolder": "", "type": "output"},
                    {"filename": "secret.png", "subfolder": "..", "type": "input"},
                    {"filename": str(outside), "subfolder": "", "type": "output"},
                    {"filename": "x.png", "subfolder": "", "type": "temp"},
                    {"filename": "missing.png", "subfolder": "", "type": "output"},
                    "secret.png", None):
            status, body = lora_routes.handle_preview({"run_key": KEY, "image": ref})
            self.assertEqual(status, 400, ref)
            self.assertIn("error", body)
        self.assertFalse((self.store / "previews.json").exists())

    def test_unreadable_preview_image_is_400(self):
        (self.output / "bad.png").write_bytes(b"junk")
        with self.assertLogs("ai_studio.lora_routes", "WARNING"):
            status, _ = lora_routes.handle_preview(
                {"run_key": KEY, "image": {"filename": "bad.png", "subfolder": "", "type": "output"}})
        self.assertEqual(status, 400)

    def test_prefs_round_trip_normalised(self):
        self.assertEqual(lora_routes.handle_prefs_get(None)[1]["sort"], "name")
        status, saved = lora_routes.handle_prefs_set({"favorites": ["a", "a"], "sort": "bogus", "show_all": True})
        self.assertEqual((status, saved["favorites"], saved["sort"], saved["show_all"]), (200, ["a"], "name", True))
        self.assertEqual(lora_routes.handle_prefs_get(None)[1], saved)
        self.assertEqual(lora_routes.handle_prefs_set([])[0], 400)


class RegistrationTests(unittest.TestCase):
    def test_handlers_registered_under_prefix(self):
        saved = dict(studio_routes._HANDLERS)
        self.addCleanup(lambda: (studio_routes._HANDLERS.clear(), studio_routes._HANDLERS.update(saved)))
        studio_routes._HANDLERS.clear()
        registered = []

        def route(method, path):
            return lambda handler: registered.append((method, path)) or handler

        instance = SimpleNamespace(routes=SimpleNamespace(route=route))
        modules = {"server": SimpleNamespace(PromptServer=SimpleNamespace(instance=instance))}
        lora_routes.register_lora_handlers()
        with mock.patch.dict("sys.modules", modules):
            self.assertTrue(studio_routes.register_routes())
        self.assertEqual(sorted(registered), sorted((m, "/aistudio/studio" + p) for m, p in (
            ("GET", "/loras/index"), ("GET", "/loras/detail"), ("GET", "/loras/thumb"),
            ("GET", "/loras/prefs"), ("POST", "/loras/prefs"), ("POST", "/loras/preview"))))


if __name__ == "__main__":
    unittest.main()
