"""Regression and coverage tests for comfy_custom_nodes/ai_studio_forge_parity (11-26 review: B-22, B-24, P-03, M-05)."""
from __future__ import annotations

import copy
import gc
import json
import os
import struct
import sys
import tempfile
import threading
import types
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np
from PIL import Image, ImageFile

from comfy_custom_nodes.ai_studio_forge_parity import (
    anima38_cache,
    lora_atomic,
    lora_index,
    lora_routes,
    lora_samples,
    lora_store,
    sam3_face_crop,
    sam3_face_crop_run,
    studio_routes,
)


def _png(path: Path, size=(16, 16), color=(180, 40, 40)) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", size, color).save(path)
    return path


def _safetensors(path: Path, metadata=None) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    header = {} if metadata is None else {"__metadata__": metadata}
    raw = json.dumps(header).encode("utf-8")
    path.write_bytes(struct.pack("<Q", len(raw)) + raw)
    return path


class B22TrimConsistencyTests(unittest.TestCase):
    def test_model_and_algo_folders_trim_whitespace_and_bom(self):
        cases = [
            ("\ufeffMOSS\\OFT\\g\\r\\r-000001.safetensors", "\ufeffMOSS", "OFT", "g", "r", "moss"),
            (" SDXL \\styles\\x.safetensors", " SDXL ", "styles", "", "x", "sdxl"),
            ("\u3000KREA2\u3000\\OFT\\g\\r.safetensors", "\u3000KREA2\u3000", "OFT", "g", "r", "krea2"),
            ("FLUX\\\ufeffLoKr\\g\\r\\r-000002.safetensors", "FLUX", "\ufeffLoKr", "g", "r", None),
            (" LoRA \\LoKr\\x.safetensors", "", " LoRA ", "LoKr", "x", None),
        ]
        none_summary = lora_index.header_summary(None)
        for entry, model, algo, group, run, family in cases:
            with self.subTest(entry=entry):
                parsed = lora_index.parse_lora_path(entry)
                self.assertEqual((parsed["model"], parsed["algo"], parsed["group"], parsed["run"]),
                                 (model, algo, group, run))
                self.assertEqual(lora_index.model_folder_family(parsed["model"]), family)
                self.assertEqual(lora_index.lora_family(none_summary, parsed), family)


class B24MoveLogAndEmptyRootsTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.root = Path(self._tmp.name)
        lora_samples.reset_manifest_cache()
        lora_samples.reset_move_log_cache()
        self.addCleanup(lora_samples.reset_manifest_cache)
        self.addCleanup(lora_samples.reset_move_log_cache)

    def test_move_log_reads_all_lines_newest_first(self):
        weights = self.root / "weights"
        weights.mkdir()
        full = weights / "run_a-000001.safetensors"
        older = self.root / "train_old"
        newer = self.root / "train_new"
        older.mkdir()
        newer.mkdir()
        gone = self.root / "train_gone"
        lines = [
            {"src": str(older / "a.safetensors")},
            {"src": str(newer / "a.safetensors")},
            {"src": str(gone / "a.safetensors")},
        ]
        (weights / lora_samples.MOVE_LOG).write_text(
            "\n".join(json.dumps(x) for x in lines) + "\n", encoding="utf-8"
        )
        dirs = lora_samples.run_dirs(full, "run_a", "", self.root / "_manifest.jsonl")
        self.assertEqual(dirs, [newer, older])

    def test_empty_anima_and_board_roots_skip_cwd_lookup(self):
        # Default paths keep user's configured defaults; explicit empty/None overrides must not hit CWD.
        self.assertEqual(lora_store.DEFAULT_SETTINGS["anima_root"], r"C:\Users\URL\Desktop\UR\ANIMA")
        self.assertEqual(lora_store.DEFAULT_SETTINGS["board_root"], r"C:\Users\URL\Desktop\UR\board")
        cwd_backup = os.getcwd()
        os.chdir(self.root)
        self.addCleanup(os.chdir, cwd_backup)
        # Put decoys in CWD that Path("") would erroneously hit
        (self.root / "board_db.json").write_text(
            json.dumps({"runs": {"r": {"trigger": "@cwd"}}, "marks": {"r": {"stars": [1]}}}),
            encoding="utf-8",
        )
        (self.root / "r").mkdir()
        _png(self.root / "site" / "s" / "r" / "e001.webp")
        self.assertEqual(lora_index.load_board(""), {"runs": {}, "marks": {}})
        self.assertEqual(lora_index.load_board("   "), {"runs": {}, "marks": {}})
        self.assertIsNone(lora_samples.board_strip("", "r", 1))
        self.assertIsNone(lora_samples.board_strip(None, "r", 1))
        self.assertEqual(lora_samples.run_dirs(self.root / "w" / "r.safetensors", "r", "", self.root / "m.jsonl"), [])
        self.assertEqual(lora_samples.run_dirs(self.root / "w" / "r.safetensors", "r", None, self.root / "m.jsonl"), [])


class B24LoraStoreTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.root = Path(self._tmp.name)
        self.store = lora_store.store_dir(self.root / "user")

    def test_set_custom_preview_enforces_max_cover_pixels_before_decode(self):
        src = _png(self.root / "big.png", size=(50, 40))  # 2000 px
        with mock.patch.object(lora_samples, "MAX_COVER_PIXELS", 1999), \
                mock.patch.object(ImageFile.ImageFile, "load", side_effect=AssertionError("pixels decoded")):
            with self.assertRaises(ValueError):
                lora_store.set_custom_preview(self.store, "run/a", src, 32)
        self.assertIsNone(lora_store.custom_preview(self.store, "run/a"))
        with mock.patch.object(lora_samples, "MAX_COVER_PIXELS", 2000):
            name = lora_store.set_custom_preview(self.store, "run/a", src, 32)
        self.assertTrue(name.endswith(".webp"))
        self.assertIsNotNone(lora_store.custom_preview(self.store, "run/a"))

    def test_concurrent_set_and_clear_custom_preview_preserves_index(self):
        src = _png(self.root / "ok.png", size=(16, 16))
        barrier = threading.Barrier(8)
        errors: list[Exception] = []

        def worker(i: int):
            try:
                barrier.wait(5)
                lora_store.set_custom_preview(self.store, f"run/{i}", src, 16)
            except Exception as exc:
                errors.append(exc)

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(15)
        self.assertEqual(errors, [])
        for i in range(8):
            self.assertIsNotNone(lora_store.custom_preview(self.store, f"run/{i}"), i)

        # Concurrently clear half while keeping the other half
        barrier2 = threading.Barrier(4)

        def clearer(i: int):
            try:
                barrier2.wait(5)
                lora_store.clear_custom_preview(self.store, f"run/{i}")
            except Exception as exc:
                errors.append(exc)

        threads2 = [threading.Thread(target=clearer, args=(i,)) for i in range(4)]
        for t in threads2:
            t.start()
        for t in threads2:
            t.join(15)
        self.assertEqual(errors, [])
        for i in range(4):
            self.assertIsNone(lora_store.custom_preview(self.store, f"run/{i}"))
        for i in range(4, 8):
            self.assertIsNotNone(lora_store.custom_preview(self.store, f"run/{i}"))


class B24StudioRoutesTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.root = Path(self._tmp.name).resolve()
        self.dirs = {name: self.root / name for name in ("output", "temp", "input")}
        for path in self.dirs.values():
            path.mkdir()
        self.fp = SimpleNamespace(
            get_output_directory=lambda: str(self.dirs["output"]),
            get_temp_directory=lambda: str(self.dirs["temp"]),
            get_input_directory=lambda: str(self.dirs["input"]),
            get_directory_by_type=lambda kind: str(self.dirs[kind]) if kind in self.dirs else None,
        )
        patcher = mock.patch.dict("sys.modules", {"folder_paths": self.fp})
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_load_cells_enforces_total_pixel_budget_before_decode(self):
        _png(self.dirs["temp"] / "a.png", size=(40, 40))  # 1600 px
        _png(self.dirs["temp"] / "b.png", size=(40, 40))  # 1600 px -> total 3200 px
        body = {
            "nx": 2, "ny": 1, "nz": 1,
            "labels": {"x": ["a", "b"], "y": [""], "z": [""]},
            "options": {"draw_legend": False, "margin": 0, "row_count": 0},
            "cells": [
                {"ix": 0, "iy": 0, "iz": 0, "image": {"filename": "a.png", "subfolder": "", "type": "temp"}},
                {"ix": 1, "iy": 0, "iz": 0, "image": {"filename": "b.png", "subfolder": "", "type": "temp"}},
            ],
            "infotext": "",
        }
        with mock.patch.object(studio_routes, "MAX_GRID_TOTAL_PIXELS", 2000):
            status, payload = studio_routes.handle_grid(body)
        self.assertEqual(status, 400)
        self.assertIn("pixel limit", payload["error"])

        # Single image exceeding budget must not decode pixels
        with mock.patch.object(studio_routes, "MAX_GRID_TOTAL_PIXELS", 1599), \
                mock.patch.object(ImageFile.ImageFile, "load", side_effect=AssertionError("pixels decoded")):
            status, _payload = studio_routes.handle_grid(body)
        self.assertEqual(status, 400)

    async def test_open_output_route_rejects_non_loopback_remote(self):
        web = SimpleNamespace(json_response=lambda data, status=200: (data, status))
        handler = studio_routes._HANDLERS[("POST", "/open_output")]
        opened: list[str] = []

        class Req:
            def __init__(self, remote):
                self.remote = remote
                self.rel_url = SimpleNamespace(query={})

        with mock.patch.dict("sys.modules", {"aiohttp": SimpleNamespace(web=web)}), \
                mock.patch.object(studio_routes, "_is_windows", return_value=True), \
                mock.patch.object(os, "startfile", opened.append, create=True):
            for bad_remote in ("192.168.1.50", "10.0.0.2", "203.0.113.7"):
                data, status = await handler(Req(bad_remote))
                self.assertEqual(status, 403, bad_remote)
                self.assertIn("error", data)
            self.assertEqual(opened, [])

            for ok_remote in ("127.0.0.1", "::1", "::ffff:127.0.0.1", None):
                data, status = await handler(Req(ok_remote))
                self.assertEqual(status, 200, ok_remote)
                self.assertEqual(data["path"], str(self.dirs["output"]))
            self.assertEqual(len(opened), 4)


class P03PerformanceTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.root = Path(self._tmp.name)
        lora_samples.reset_manifest_cache()
        lora_samples.reset_move_log_cache()
        lora_routes.reset_index_cache()
        self.addCleanup(lora_samples.reset_manifest_cache)
        self.addCleanup(lora_samples.reset_move_log_cache)
        self.addCleanup(lora_routes.reset_index_cache)

    def test_move_log_cache_reuses_and_invalidates(self):
        weights = self.root / "weights"
        weights.mkdir()
        full = weights / "r1-000001.safetensors"
        move_log = weights / lora_samples.MOVE_LOG
        dir_a = self.root / "dir_a"
        dir_b = self.root / "dir_b"
        dir_a.mkdir()
        dir_b.mkdir()
        move_log.write_text(json.dumps({"src": str(dir_a / "a.safetensors")}) + "\n", encoding="utf-8")

        real_json_lines = lora_samples._json_lines
        move_log_reads = 0

        def counting_json_lines(path: Path):
            nonlocal move_log_reads
            if Path(path) == move_log:
                move_log_reads += 1
            return real_json_lines(path)

        manifest = self.root / "_manifest.jsonl"
        with mock.patch.object(lora_samples, "_json_lines", side_effect=counting_json_lines):
            self.assertEqual(lora_samples.run_dirs(full, "r1", "", manifest), [dir_a])
            self.assertEqual(lora_samples.run_dirs(full, "r1", "", manifest), [dir_a])
            self.assertEqual(move_log_reads, 1)

            # Update move_log -> mtime/size changes -> cache invalidates and reloads
            move_log.write_text(
                json.dumps({"src": str(dir_a / "a.safetensors")}) + "\n"
                + json.dumps({"src": str(dir_b / "a.safetensors")}) + "\n",
                encoding="utf-8",
            )
            self.assertEqual(lora_samples.run_dirs(full, "r1", "", manifest), [dir_b, dir_a])
            self.assertEqual(move_log_reads, 2)

            # Delete move_log -> cache evicted, returns empty
            move_log.unlink()
            self.assertEqual(lora_samples.run_dirs(full, "r1", "", manifest), [])

    def test_manifest_cache_reuses_and_invalidates(self):
        manifest = self.root / "_manifest.jsonl"
        dir_a = self.root / "dir_a"
        dir_b = self.root / "dir_b"
        dir_a.mkdir()
        dir_b.mkdir()
        manifest.write_text(json.dumps({"run": "r1", "src": str(dir_a)}) + "\n", encoding="utf-8")
        full = self.root / "w" / "r1-000001.safetensors"

        real_json_lines = lora_samples._json_lines
        manifest_reads = 0

        def counting_json_lines(path: Path):
            nonlocal manifest_reads
            if Path(path) == manifest:
                manifest_reads += 1
            return real_json_lines(path)

        with mock.patch.object(lora_samples, "_json_lines", side_effect=counting_json_lines):
            self.assertEqual(lora_samples.run_dirs(full, "r1", "", manifest), [dir_a])
            self.assertEqual(lora_samples.run_dirs(full, "r1", "", manifest), [dir_a])
            self.assertEqual(lora_samples.run_dirs(full, "r2", "", manifest), [])
            self.assertEqual(manifest_reads, 1)

            # Update manifest -> size/mtime changes -> cache refreshes
            manifest.write_text(
                json.dumps({"run": "r1", "src": str(dir_a)}) + "\n"
                + json.dumps({"run": "r1", "src": str(dir_b)}) + "\n",
                encoding="utf-8",
            )
            self.assertEqual(lora_samples.run_dirs(full, "r1", "", manifest), [dir_b, dir_a])
            self.assertEqual(manifest_reads, 2)

            # Delete manifest -> returns empty
            manifest.unlink()
            self.assertEqual(lora_samples.run_dirs(full, "r1", "", manifest), [])

    def test_context_by_run_key_and_thumb_pruning(self):
        loras = self.root / "loras"
        user = self.root / "user"
        entries = [
            "LoKr\\g\\run1\\run1-000001.safetensors",
            "LoKr\\g\\run1\\run1-000002.safetensors",
            "LoKr\\g\\run2\\run2-000001.safetensors",
        ]
        for entry in entries:
            _safetensors(loras / entry.replace("\\", "/"), {"ss_num_epochs": "2"})

        fp = SimpleNamespace(
            get_filename_list=lambda kind: list(entries),
            get_full_path=lambda kind, name: str(loras / name.replace("\\", "/")),
            get_folder_paths=lambda kind: [str(loras)],
            get_user_directory=lambda: str(user),
            get_output_directory=lambda: str(self.root / "out"),
        )
        with mock.patch.object(lora_routes, "folder_paths_module", lambda: fp):
            ctx = lora_routes._Context(reuse=False)
            self.assertEqual(len(ctx.by_run_key), 2)
            r1 = ctx.by_name[entries[0]]
            self.assertEqual([i["name"] for i in ctx.run_items(r1)], entries[:2])

            # Thumb pruning keeps newest MAX_THUMB_FILES webp files
            thumbs_dir = ctx.store / lora_routes.THUMBS_DIR
            with mock.patch.object(lora_routes, "MAX_THUMB_FILES", 3):
                created = []
                for idx in range(5):
                    src = _png(self.root / "imgs" / f"i{idx}.png", size=(32, 32))
                    out = lora_routes._thumb_path(ctx, src, None)
                    t = 1_700_000_000 + idx * 10
                    os.utime(out, (t, t))
                    created.append(out)
                remaining = sorted(thumbs_dir.glob("*.webp"))
                self.assertEqual(len(remaining), 3)
                # Oldest 2 pruned, newest 3 kept
                self.assertNotIn(created[0], remaining)
                self.assertNotIn(created[1], remaining)
                for keep in created[2:]:
                    self.assertIn(keep, remaining)


class M05LoraAtomicTests(unittest.TestCase):
    def test_atomic_path_replaces_on_success_and_cleans_on_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / "data.json"
            target.write_text("old", encoding="utf-8")

            with lora_atomic.atomic_path(target) as tmp_path:
                self.assertNotEqual(tmp_path, target)
                tmp_path.write_text("new", encoding="utf-8")
            self.assertEqual(target.read_text(encoding="utf-8"), "new")
            self.assertEqual(list(Path(tmp).glob("*.tmp")), [])

            with self.assertRaises(RuntimeError):
                with lora_atomic.atomic_path(target) as tmp_path:
                    tmp_path.write_text("partial", encoding="utf-8")
                    raise RuntimeError("boom")
            self.assertEqual(target.read_text(encoding="utf-8"), "new")
            self.assertEqual(list(Path(tmp).glob("*.tmp")), [])

    def test_unique_tmp_differs_across_threads(self):
        target = Path("sample.webp")
        names: list[Path] = []
        lock = threading.Lock()

        def grab():
            p = lora_atomic.unique_tmp(target)
            with lock:
                names.append(p)

        threads = [threading.Thread(target=grab) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(5)
        self.assertEqual(len(set(names)), 4)


class M05Anima38CacheTests(unittest.TestCase):
    def test_registry_managed_vs_legacy_and_copy_lifetime(self):
        reg = anima38_cache._RunRegistry({1: "pre1"}, [1])
        self.assertIn(1, reg.legacy)
        # Legacy FIFO pruning caps at 64 entries
        for i in range(2, 70):
            reg[i] = f"legacy_{i}"
        self.assertEqual(len(reg.legacy), 64)
        self.assertNotIn(1, reg)

        # Managed run cannot be popped by legacy FIFO code
        pending: list[int] = []
        token = reg.pending.set(pending)
        try:
            rid = reg.register("managed_payload")
        finally:
            reg.pending.reset(token)
        self.assertEqual(pending, [rid])
        self.assertEqual(reg.pop(rid, None), "managed_payload")
        self.assertIn(rid, reg)

        # _RunOwner shallow/deep copy preserves single lease until all refs drop
        owner = anima38_cache._RunOwner(reg, rid)
        owner_copy = copy.copy(owner)
        owner_deep = copy.deepcopy(owner)
        self.assertIs(owner_copy, owner)
        self.assertIs(owner_deep, owner)
        del owner_copy, owner_deep
        gc.collect()
        self.assertIn(rid, reg)
        del owner
        gc.collect()
        self.assertNotIn(rid, reg)

    def test_install_conditioning_cache_support_and_exception_cleanup(self):
        anima_cls = type("Anima", (), {"_qwen35_gated_runs": {}, "_qwen35_gated_run_counter": [0]})
        comfy_pkg = types.ModuleType("comfy")
        ldm_pkg = types.ModuleType("comfy.ldm")
        anima_pkg = types.ModuleType("comfy.ldm.anima")
        model_mod = types.ModuleType("comfy.ldm.anima.model")
        model_mod.Anima = anima_cls

        runtime = types.ModuleType("fake_pkg.semantic_v2_runtime")
        runtime._GATED_RUNS = {}
        runtime._GATED_RUN_COUNTER = [0]

        def _orig_register(run):
            runtime._GATED_RUN_COUNTER[0] += 1
            rid = runtime._GATED_RUN_COUNTER[0]
            runtime._GATED_RUNS[rid] = run
            return rid

        def emit_conditioning(payload, fail=False):
            rid = runtime._register_run(payload)
            if fail:
                raise ValueError("emit failed")
            return [[("tensor", {"qwen35_gated_run": rid})]]

        runtime._register_run = _orig_register
        runtime.emit_conditioning = emit_conditioning

        consumer = types.ModuleType("fake_pkg.consumer")
        consumer.emit_conditioning = emit_conditioning

        fake_modules = {
            "comfy": comfy_pkg,
            "comfy.ldm": ldm_pkg,
            "comfy.ldm.anima": anima_pkg,
            "comfy.ldm.anima.model": model_mod,
            "fake_pkg.semantic_v2_runtime": runtime,
            "fake_pkg.consumer": consumer,
        }
        with mock.patch.dict(sys.modules, fake_modules):
            anima38_cache.install_conditioning_cache_support(runtime)
            # Consumer alias was updated to wrapped emitter
            self.assertIs(consumer.emit_conditioning, runtime.emit_conditioning)
            # Keep >64 conditioning outputs alive: none may be evicted
            kept = [consumer.emit_conditioning(f"run_{i}") for i in range(70)]
            reg = anima_cls._qwen35_gated_runs
            self.assertEqual(len(reg), 70)
            # Exception after registration does not leak
            with self.assertRaises(ValueError):
                consumer.emit_conditioning("leaked", fail=True)
            self.assertEqual(len(reg), 70)
            # Releasing all conditioning outputs frees registry entries
            kept.clear()
            gc.collect()
            self.assertEqual(len(reg), 0)

            # Missing emitter raises RuntimeError
            bad_rt = types.ModuleType("fake_pkg.bad")
            with self.assertRaises(RuntimeError):
                anima38_cache.install_conditioning_cache_support(bad_rt)


class M05Sam3FaceCropRunTests(unittest.TestCase):
    def test_pure_mask_helpers(self):
        self.assertEqual(sam3_face_crop_run._split_masks(np.zeros((0, 4, 4)), 4, 4), [])
        m2d = np.zeros((4, 4), dtype=np.float32)
        m2d[1:3, 1:3] = 1.0
        split2d = sam3_face_crop_run._split_masks(m2d, 4, 4)
        self.assertEqual(len(split2d), 1)
        self.assertEqual(sam3_face_crop_run._mask_bbox(split2d[0]), (1, 1, 3, 3))
        self.assertIsNone(sam3_face_crop_run._mask_bbox(np.zeros((4, 4), dtype=bool)))
        self.assertEqual(sam3_face_crop_run._intersection_area((0, 0, 4, 4), (2, 2, 6, 6)), 4)
        self.assertEqual(sam3_face_crop_run._intersection_area((0, 0, 2, 2), (3, 3, 5, 5)), 0)

        # clean_pieces clips stray pixels outside the best matching box + pad
        mask = np.zeros((40, 40), dtype=bool)
        mask[5:10, 5:10] = True
        mask[38, 38] = True  # far outside box [5, 5, 10, 10] + pad(8)
        cleaned = sam3_face_crop_run.clean_pieces(mask[None, ...], [[5.0, 5.0, 10.0, 10.0]], 40, 40)
        self.assertEqual(len(cleaned), 1)
        self.assertTrue(cleaned[0][6, 6])
        self.assertFalse(cleaned[0][38, 38])

    def test_continuing_result_drops_non_overlapping_pieces_and_boxes(self):
        earlier = np.zeros((10, 10), dtype=bool)
        earlier[1:4, 1:4] = True
        overlap = np.zeros((10, 10), dtype=bool)
        overlap[2:5, 2:5] = True
        disjoint = np.zeros((10, 10), dtype=bool)
        disjoint[7:9, 7:9] = True
        found = ([overlap, disjoint], [[2, 2, 5, 5], [7, 7, 9, 9]], [0.9, 0.4], [0, 1])
        (pieces, boxes, scores), dropped = sam3_face_crop_run._continuing_result(found, [earlier])
        self.assertEqual(dropped, 1)
        self.assertEqual(len(pieces), 1)
        self.assertEqual(boxes, [[2, 2, 5, 5]])
        self.assertEqual(scores, [0.9])

    def test_run_face_crop_empty_and_single_face_with_growth(self):
        # Empty crops returns empty outcome without calling segment
        empty_info = sam3_face_crop.FaceCropInfo(
            model="face_yolov8n.pt", confidence=0.5, boxes=[], scores=[], crops=[],
        )
        called = []
        outcome = sam3_face_crop_run.run_face_crop(
            empty_info, [["hair"]], ["eye"],
            lambda tok, crop: called.append((tok, crop)) or ([], [], []),
            lambda m: m, image_size=(64, 64), margin=0,
        )
        self.assertEqual(called, [])
        self.assertEqual(outcome.group_masks, [])

        # Single face whose initial piece touches crop inner edge -> triggers crop growth
        face_box = [16.0, 16.0, 40.0, 40.0]
        crop0 = sam3_face_crop.CropBox(10, 10, 50, 50)
        info = sam3_face_crop.FaceCropInfo(
            model="face_yolov8n.pt", confidence=0.5, boxes=[face_box], scores=[0.9], crops=[crop0],
        )
        logs: list[str] = []

        def fake_segment(token: str, crop: sam3_face_crop.CropBox):
            piece = np.zeros((crop.height, crop.width), dtype=bool)
            if token == "eye":
                piece[10:14, 10:14] = True
                return [piece], [[10.0, 10.0, 14.0, 14.0]], [0.8]
            if crop == crop0:
                # Touch right inner edge (x = crop.width - 1) so growth triggers rightward
                piece[10:25, 10:crop.width] = True
                return [piece], [[10.0, 10.0, float(crop.width), 25.0]], [0.9]
            # Grown crop: overlapping piece that stays inside (no longer touches inner edge)
            piece[10:25, 10:42] = True
            return [piece], [[10.0, 10.0, 42.0, 25.0]], [0.95]

        res = sam3_face_crop_run.run_face_crop(
            info, [["hair"]], ["eye"], fake_segment, lambda m: m,
            image_size=(80, 80), margin=0, log=logs.append,
        )
        self.assertEqual(len(res.group_masks), 1)
        self.assertTrue(np.any(res.group_masks[0]))
        self.assertTrue(len(res.exclude_pieces) >= 1)
        self.assertEqual(len(info.growth), 1)
        self.assertEqual(info.growth[0].rounds, 1)
        self.assertTrue(any("reached the crop edge" in line for line in logs))


if __name__ == "__main__":
    unittest.main()
