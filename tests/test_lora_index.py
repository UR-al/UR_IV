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


class ModelFolderTests(unittest.TestCase):
    def test_model_folder_is_only_taken_before_an_algorithm_folder(self):
        cases = {
            # entry: (model, algo, group, run, downloaded, run_key)
            "ANIMA\\LoKr\\g\\r\\r-000003.safetensors": ("ANIMA", "LoKr", "g", "r", False, "ANIMA/LoKr/g/r"),
            "_unsorted\\LoRA\\_downloaded\\M3\\m3.safetensors": ("_unsorted", "LoRA", "M3", "m3", True, "_unsorted/LoRA/M3/m3"),
            "SDXL\\LoRA\\HUSI\\husi_v1.safetensors": ("SDXL", "LoRA", "HUSI", "husi_v1", False, "SDXL/LoRA/HUSI/husi_v1"),
            "LoKr\\g\\r\\r-000003.safetensors": ("", "LoKr", "g", "r", False, "LoKr/g/r"),  # old layout unchanged
            "LoRA\\LoKr\\x.safetensors": ("", "LoRA", "LoKr", "x", False, "LoRA/LoKr/x"),  # first folder is an algo
            "ANIMA\\x.safetensors": ("", "ANIMA", "", "x", False, "ANIMA//x"),  # one folder: no model
            "x.safetensors": ("", "", "", "x", False, "//x"),
        }
        for entry, want in cases.items():
            with self.subTest(entry=entry):
                p = lora_index.parse_lora_path(entry)
                self.assertEqual((p["model"], p["algo"], p["group"], p["run"], p["downloaded"], p["run_key"]), want)

    def test_model_folder_gives_the_family_when_header_and_recipe_do_not(self):
        # 11-16 review item 2: KREA2 / SDXL runs had no family (always shown), so the filter could not hide them
        none = lora_index.header_summary(None)
        cases = {
            "MOSS\\LoKr\\@enaa97\\@enaa97_moss\\@enaa97_moss-000002.safetensors": "moss",
            "KREA2\\LoRA\\krea2\\QuadView_krea2_v1.safetensors": "krea2",
            "SDXL\\LoRA\\HUSI\\HUSI.safetensors": "sdxl",
            # an ANIMA run without 2.9b/3.8b ("40b" is not a family size): an Anima LoRA of unknown size
            "ANIMA\\LoRA\\_downloaded\\snapref-v1a\\snapref_v1a__40b.safetensors": "anima",
            "anima\\lokr\\g\\r\\r-000001.safetensors": "anima",  # case-blind
            "﻿MOSS\\LoKr\\g\\r\\r.safetensors": "moss",  # a BOM from a hand move is trimmed (JS String.trim)
            "_unsorted\\LoRA\\_downloaded\\M3_Unlocked\\M3_Unlocked_V2.safetensors": None,
            "FLUX\\LoRA\\g\\r.safetensors": None,  # a model folder by place only: no family known
            "LoKr\\g\\r\\r-000001.safetensors": None,  # old layout: no model folder
        }
        for entry, want in cases.items():
            with self.subTest(entry=entry):
                self.assertEqual(lora_index.lora_family(none, lora_index.parse_lora_path(entry)), want)

    @staticmethod
    def _base(name):
        return lora_index.header_summary({"__metadata__": {"ss_sd_model_name": name}})

    def test_moss_krea2_sdxl_folders_win_over_the_header_and_the_recipe(self):
        # 11-16 review: the model folder is the user's own sorting, so a MOSS / KREA2 / SDXL folder beats what the header's
        # base model or the run name's size says (a MOSS run whose header names an Anima base, a 2.9b recipe under SDXL)
        anima38 = self._base("Anima-3.8B-v1.1.safetensors")
        cases = [
            (anima38, "MOSS\\LoKr\\g\\r\\r-000001.safetensors", "moss"),
            (lora_index.header_summary(None), "MOSS\\LoKr\\g\\x_2.9b_lokr\\x_2.9b_lokr-000001.safetensors", "moss"),
            (anima38, "SDXL\\LoRA\\HUSI\\x_2.9b.safetensors", "sdxl"),
            (self._base("NoobAI-XL-v1.1.safetensors"), "SDXL\\LoRA\\HUSI\\HUSI.safetensors", "sdxl"),
            (self._base("moss-image-v1.safetensors"), "KREA2\\LoRA\\krea2\\QuadView_krea2_v1.safetensors", "krea2"),
            (anima38, "﻿sdxl\\LoRA\\g\\x_3.8b.safetensors", "sdxl"),  # case-blind and trimmed (model_folder_family)
        ]
        for summary, entry, want in cases:
            with self.subTest(entry=entry, base=summary["base_model"]):
                self.assertEqual(lora_index.lora_family(summary, lora_index.parse_lora_path(entry)), want)

    def test_header_base_model_and_recipe_size_still_come_first_in_an_anima_folder_or_none(self):
        anima38 = self._base("Anima-3.8B-v1.1.safetensors")
        none = lora_index.header_summary(None)
        cases = [
            # ANIMA folder: header base model -> 2.9b / 3.8b recipe -> "anima"
            (anima38, "ANIMA\\LoKr\\g\\x_2.9b_lokr\\x_2.9b_lokr-000001.safetensors", "3.8b"),
            (self._base("moss-image-v1.safetensors"), "ANIMA\\LoKr\\g\\r\\r-000001.safetensors", "moss"),
            (none, "ANIMA\\LoKr\\g\\x_2.9b_lokr\\x_2.9b_lokr-000001.safetensors", "2.9b"),
            (none, "ANIMA\\LoKr\\g\\r\\r-000001.safetensors", "anima"),
            # no model folder (old layout) or one that says nothing (_unsorted): header -> recipe -> unknown
            (anima38, "LoKr\\g\\x_2.9b\\x_2.9b-000001.safetensors", "3.8b"),
            (none, "LoKr\\g\\x_2.9b\\x_2.9b-000001.safetensors", "2.9b"),
            (anima38, "_unsorted\\LoRA\\M3\\m3.safetensors", "3.8b"),
            (none, "_unsorted\\LoRA\\M3\\m3.safetensors", None),
        ]
        for summary, entry, want in cases:
            with self.subTest(entry=entry, base=summary["base_model"]):
                self.assertEqual(lora_index.lora_family(summary, lora_index.parse_lora_path(entry)), want)

    def test_an_animagine_base_model_is_not_anima(self):
        # "animagine-xl-3.1" (an SDXL model) has "anima" in it: its LoRAs were counted 2.9b
        plain = lora_index.parse_lora_path("LoKr\\g\\r\\r-000001.safetensors")
        for name in ("animagine-xl-3.1", "animagine-xl-3.1.safetensors", "Animagine-XL-4.0.safetensors",
                     "ANIMAGINEXL_V3.safetensors"):
            with self.subTest(base=name):
                self.assertIsNone(lora_index.lora_family(self._base(name), plain))
        animagine = self._base("animagine-xl-3.1.safetensors")
        cases = {
            "LoKr\\g\\x_3.8b\\x_3.8b-000001.safetensors": "3.8b",  # the recipe answers next
            "ANIMA\\LoRA\\g\\r.safetensors": "anima",
            "SDXL\\LoRA\\g\\r.safetensors": "sdxl",
        }
        for entry, want in cases.items():
            with self.subTest(entry=entry):
                self.assertEqual(lora_index.lora_family(animagine, lora_index.parse_lora_path(entry)), want)
        # Anima base names keep matching
        anima = {"anima-preview2.safetensors": "2.9b", "Anima-2.9B-preview-v1.safetensors": "2.9b",
                 "anima_baseV10.safetensors": "base", "anima-3.8b-v2.safetensors": "3.8b",
                 "animagine_to_anima_merge.safetensors": "2.9b"}
        for name, want in anima.items():
            with self.subTest(base=name):
                self.assertEqual(lora_index.lora_family(self._base(name), plain), want)

    def test_model_folder_family_table(self):
        self.assertEqual({name: lora_index.model_folder_family(name) for name in
                          ("ANIMA", "MOSS", "KREA2", "SDXL", "Sdxl", " KREA2　", "_unsorted", "", "FLUX", "get")},
                         {"ANIMA": "anima", "MOSS": "moss", "KREA2": "krea2", "SDXL": "sdxl", "Sdxl": "sdxl",
                          " KREA2　": "krea2", "_unsorted": None, "": None, "FLUX": None, "get": None})


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
                "precision", "date", "trigger", "has_meta", "downloaded", "model", "group", "group_label", "run",
                "run_short", "recipe"}
        self.assertEqual(set(items[0]), want)
        self.assertEqual(items[0]["family"], "2.9b")
        self.assertEqual(items[0]["model"], "")

    def test_model_folder_layout_keeps_runs_of_one_group_apart(self):
        # 11-16 D:\LoRA\<MODEL>\<algo>\<group>\<run>: the three 나은 runs were one card with every epoch three times
        group = "ANIMA\\LoKr\\\u0eca (@9an9ta)\\"
        runs = ["@na-eun_2.9b_lokr_came_b10_2e06", "@na-eun_2.9b_lokr_came_b10_2e06_fp16",
                "@na-eun_2.9b_lokr_came_b10_2e06_fp16_v2"]
        names = [f"{group}{r}\\{r}-{n:06d}.safetensors" for r in runs for n in (1, 2)]
        names.append("MOSS\\LoKr\\\u0eca (@9an9ta)\\@na-eun_2.9b_lokr_came_b10_2e06\\x-000001.safetensors")
        for name in names:
            write_safetensors(self.root / name.replace("\\", os.sep), {"ss_epoch": "1"})
        items = self.build(names)
        keys = {}
        for item in items:
            keys.setdefault(item["run_key"], []).append(item["epoch"])
        self.assertEqual(len(keys), 4)  # three ANIMA runs + the same run name under MOSS
        self.assertEqual(sorted(len(v) for v in keys.values()), [1, 2, 2, 2])
        first = items[0]
        self.assertEqual((first["model"], first["group"], first["run"]), ("ANIMA", "\u0eca (@9an9ta)", runs[0]))
        self.assertEqual(first["run_key"], f"ANIMA/LoKr/\u0eca (@9an9ta)/{runs[0]}")
        self.assertEqual(first["family"], "2.9b")  # the recipe comes from the run name again, not the group folder

    def test_items_take_the_family_of_their_model_folder(self):
        names = {
            "ANIMA\\LoRA\\_downloaded\\snapref-v1a\\snapref_v1a__52b.safetensors": "anima",
            "ANIMA\\LoKr\\g\\x_3.8b\\x_3.8b-000001.safetensors": "3.8b",
            "MOSS\\LoKr\\g\\@x_moss\\@x_moss-000001.safetensors": "moss",
            "KREA2\\Full\\krea2\\krea2_3ajoxt.safetensors": "krea2",
            "SDXL\\LoCon\\shinur\\shinur-v0.4.safetensors": "sdxl",
            "_unsorted\\LoRA\\_downloaded\\M3_Unlocked\\M3_Unlocked_V2.safetensors": None,
        }
        for name in names:
            write_safetensors(self.root / name.replace("\\", os.sep), {"ss_epoch": "1"})  # no base model in the header
        items = {i["name"]: i for i in self.build(list(names))}
        self.assertEqual({name: items[name]["family"] for name in names}, names)

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
