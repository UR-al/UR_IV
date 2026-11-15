"""LoRA index: header summary, family, trigger, mtime cache, board loading."""
import json
import os
import struct
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from comfy_custom_nodes.ai_studio_forge_parity import lora_index


def write_safetensors(path: Path, metadata=None, raw_header=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    if raw_header is None:
        header = {} if metadata is None else {"__metadata__": metadata}
        raw_header = json.dumps(header).encode("utf-8")
    path.write_bytes(struct.pack("<Q", len(raw_header)) + raw_header)


class SummaryTests(unittest.TestCase):
    def test_summary_reads_epoch_and_base_model(self):
        header = {"__metadata__": {"ss_epoch": "14", "ss_num_epochs": "15",
                                   "ss_sd_model_name": "Anima-2.9B-preview-v1.safetensors"}}
        summary = lora_index.header_summary(header)
        parsed = lora_index.parse_lora_path("LoKr\\g\\r\\r-000014.safetensors")
        self.assertEqual(summary["epoch"], 14)
        self.assertEqual(summary["num_epochs"], 15)
        self.assertTrue(summary["has_meta"])
        self.assertEqual(lora_index.lora_family(summary, parsed), "2.9b")

    def test_family_falls_back_to_recipe_then_none(self):
        parsed = lora_index.parse_lora_path("LoKr\\g\\x_3.8b_lokr\\x_3.8b_lokr-000001.safetensors")
        self.assertEqual(lora_index.lora_family(lora_index.header_summary(None), parsed), "3.8b")
        plain = lora_index.parse_lora_path("LoKr\\g\\x\\x-000001.safetensors")
        self.assertIsNone(lora_index.lora_family(lora_index.header_summary(None), plain))
        other = lora_index.header_summary({"__metadata__": {"ss_sd_model_name": "sdxl_base.safetensors"}})
        self.assertIsNone(lora_index.lora_family(other, plain))

    def test_bad_headers_do_not_break_the_index(self):
        cases = [None, {}, {"__metadata__": None}, {"__metadata__": "oops"}, {"__metadata__": {}},
                 {"__metadata__": {"ss_epoch": "None"}}, {"__metadata__": {"ss_tag_frequency": "{not json"}}]
        for header in cases:
            with self.subTest(header=header):
                summary = lora_index.header_summary(header)
                self.assertFalse(summary["has_meta"])
                self.assertIsNone(summary["epoch"])
                self.assertEqual(summary["top_tags"], [])
        self.assertFalse(lora_index.header_summary(None)["has_meta"])
        self.assertFalse(lora_index.header_summary({"a": 1})["has_meta"])

    def test_non_finite_tag_counts_are_skipped(self):
        raw = '{"x": {"@a": NaN, "b": Infinity, "c": -Infinity, "d": 3, "e": "7"}}'
        summary = lora_index.header_summary({"__metadata__": {"ss_tag_frequency": raw}})
        self.assertEqual(summary["top_tags"][0], ["d", 3])
        self.assertTrue(summary["has_meta"])

    def test_trigger_prefers_board_then_at_tag(self):
        freq = json.dumps({"x": {"@gt dune": 70, "1boy": 90}})
        summary = lora_index.header_summary({"__metadata__": {"ss_tag_frequency": freq}})
        self.assertEqual(summary["top_tags"][0], ["1boy", 90])
        self.assertEqual(lora_index.trigger_of(summary, {"trigger": "@board"}), "@board")
        self.assertEqual(lora_index.trigger_of(summary, None), "@gt dune")
        self.assertEqual(lora_index.trigger_of(lora_index.header_summary(None), {}), "")

    def test_trigger_is_cut_at_the_first_triple_bar(self):
        # captions like "@00rux ||| male focus": only the part before "|||" goes into the 작가 칸
        freq = json.dumps({"x": {"@00rux ||| male focus": 70, "1boy": 90}})
        summary = lora_index.header_summary({"__metadata__": {"ss_tag_frequency": freq}})
        self.assertEqual(lora_index.trigger_of(summary, None), "@00rux")
        self.assertEqual(lora_index.trigger_of(summary, {"trigger": " @board  ||| a ||| b"}), "@board")
        self.assertEqual(lora_index.trigger_of(summary, {"trigger": "||| only after"}), "@00rux")  # empty -> next source
        self.assertEqual(lora_index.trigger_of(lora_index.header_summary(None), {"trigger": "@a|||"}), "@a")

    def test_source_spells_invisible_whitespace_as_escapes(self):
        source = Path(lora_index.__file__).read_text(encoding="utf-8")
        hidden = sorted({f"U+{ord(c):04X}" for c in source
                         if ord(c) > 127 and (c.isspace() or c in "​‌‍⁠﻿")})
        self.assertEqual(hidden, [], "write these as \\u escapes")
        # behaviour unchanged: the JS String.trim set and the group-tag pattern
        js_space = " \t\n\v\f\r  " + "".join(chr(c) for c in range(0x2000, 0x200B)) + "    　﻿"
        self.assertEqual(lora_index._JS_SPACE, js_space)
        self.assertEqual(lora_index.parse_lora_path("LoKr\\묵밥　(@Bjunbae)﻿\\r.safetensors")["group_label"], "묵밥")
        self.assertEqual(lora_index.parse_lora_path("LoKr\\묵밥 (@Bjunbae) \\r.safetensors")["group_label"], "묵밥")


class IndexTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.root = Path(self._tmp.name)
        self.names = ["LoKr\\g\\run_2.9b\\run_2.9b-000001.safetensors", "LoKr\\g\\run_2.9b\\run_2.9b.safetensors"]
        for name in self.names:
            write_safetensors(self.root / name.replace("\\", os.sep), {"ss_epoch": "1"})
        self.cache = self.root / "cache" / "lora_index.json"
        self.index = lora_index.LoraIndex(self.cache)

    def full_path(self, name):
        path = self.root / name.replace("\\", os.sep)
        return str(path) if path.exists() else None

    def build(self, names=None):
        return self.index.build(self.names if names is None else names, self.full_path, {"runs": {}, "marks": {}})

    def test_index_cache_save_leaves_no_tmp_on_error(self):
        with mock.patch.object(lora_index.os, "replace", side_effect=OSError("locked")),                 self.assertLogs(lora_index.log, "WARNING"):
            items = self.build()
        self.assertEqual(len(items), 2)
        self.assertEqual(list(self.cache.parent.glob("*.tmp")), [])

    def test_build_item_keys(self):
        items = self.build()
        self.assertEqual(len(items), 2)
        want = {"name", "stem", "size", "mtime", "run_key", "epoch", "final", "num_epochs", "family", "algo",
                "precision", "date", "trigger", "has_meta", "downloaded", "group", "group_label", "run",
                "run_short", "recipe"}
        self.assertEqual(set(items[0]), want)
        self.assertEqual(items[0]["family"], "2.9b")

    def test_build_survives_malformed_files(self):
        bad = {
            "LoKr\\g\\r\\bad-000001.safetensors": dict(raw_header=b"not json at all"),
            "LoKr\\g\\r\\bad-000002.safetensors": dict(raw_header=b""),
            "LoKr\\g\\r\\bad-000003.safetensors": dict(metadata=None),
            "LoKr\\g\\r\\bad-000004.safetensors": dict(metadata={"ss_epoch": "None"}),
        }
        for name, kwargs in bad.items():
            write_safetensors(self.root / name.replace("\\", os.sep), **kwargs)
        (self.root / "LoKr" / "g" / "r" / "empty-000005.safetensors").write_bytes(b"")
        names = list(bad) + ["LoKr\\g\\r\\empty-000005.safetensors"]
        items = self.build(names + self.names)
        self.assertEqual(len(items), len(names) + 2)
        by_name = {i["name"]: i for i in items}
        for name in names:
            self.assertFalse(by_name[name]["has_meta"], name)
            self.assertIsNone(by_name[name]["family"], name)
        for n in range(1, 5):
            self.assertEqual(by_name[f"LoKr\\g\\r\\bad-00000{n}.safetensors"]["epoch"], n)  # falls back to the path
        self.assertTrue(by_name[self.names[0]]["has_meta"])

    def test_cache_rereads_only_changed_files(self):
        with mock.patch.object(lora_index, "safetensors_header", wraps=lora_index.safetensors_header) as spy:
            self.build()
            self.assertEqual(spy.call_count, 2)
            spy.reset_mock()
            self.build()
            self.assertEqual(spy.call_count, 0)
            path = self.root / self.names[0].replace("\\", os.sep)
            stat = path.stat()
            os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns + 5_000_000_000))
            self.build()
            self.assertEqual(spy.call_count, 1)
        again = lora_index.LoraIndex(self.cache)
        with mock.patch.object(lora_index, "safetensors_header") as spy:
            again.build(self.names, self.full_path, {})
            self.assertEqual(spy.call_count, 0)
        self.assertFalse(self.cache.with_name("lora_index.json.tmp").exists())

    def test_missing_files_drop_out(self):
        self.build()
        self.assertEqual(len(self.build(self.names[:1])), 1)
        self.build()
        (self.root / self.names[0].replace("\\", os.sep)).unlink()
        items = self.build()
        self.assertEqual([i["name"] for i in items], [self.names[1]])
        self.assertEqual(self.build(["nope.safetensors"]), [])
        saved = json.loads(self.cache.read_text(encoding="utf-8"))
        self.assertNotIn("run_2.9b-000001", json.dumps(saved, ensure_ascii=False))

    def test_bad_cache_file_is_ignored(self):
        self.cache.parent.mkdir(parents=True, exist_ok=True)
        self.cache.write_text("{broken", encoding="utf-8")
        self.assertEqual(len(lora_index.LoraIndex(self.cache).build(self.names, self.full_path, {})), 2)

    def test_load_board_tolerates_missing_and_broken_files(self):
        empty = {"runs": {}, "marks": {}}
        self.assertEqual(lora_index.load_board(self.root / "nothing"), empty)
        (self.root / "board_db.json").write_text("{broken", encoding="utf-8")
        self.assertEqual(lora_index.load_board(self.root), empty)
        (self.root / "board_db.json").write_text("[1]", encoding="utf-8")
        self.assertEqual(lora_index.load_board(self.root), empty)
        (self.root / "board_db.json").write_text(json.dumps({"runs": {"a": {"trigger": "@a"}}, "plan": 1}),
                                                 encoding="utf-8")
        self.assertEqual(lora_index.load_board(self.root), {"runs": {"a": {"trigger": "@a"}}, "marks": {}})


if __name__ == "__main__":
    unittest.main()
