"""Prompt-box presets of the Anima studio: the user-folder store (studio_presets) and its routes (preset_routes).

Temp dirs only; no ComfyUI.  The match / name rules share ``js_tests/fixtures/preset_logic_golden.json`` with
``web/ai_studio/preset_logic.js``.
"""
import json
import tempfile
import threading
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from comfy_custom_nodes.ai_studio_forge_parity import preset_routes, studio_presets, studio_routes, studio_state

GOLDEN = (Path(__file__).resolve().parent.parent / "comfy_custom_nodes" / "ai_studio_forge_parity"
          / "js_tests" / "fixtures" / "preset_logic_golden.json")
SECTIONS = ("prompt_prefix", "prompt_char_count", "prompt_character", "prompt_copyright", "prompt_artist", "prompt",
            "prompt_suffix", "negative")


class _StoreCase(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)
        self.dir = self.root / "user" / "default" / "ai_studio"
        self.dir.mkdir(parents=True)
        self.file = self.dir / "prompt_presets.json"
        for module in (studio_presets, preset_routes):  # the store's own warnings (broken file, backup) stay out of the run
            patch = mock.patch.object(module.log, "warning")
            patch.start()
            self.addCleanup(patch.stop)

    def add(self, section="negative", name="n", text="worst quality", now=100):
        return studio_presets.apply(self.dir, {"op": "add", "section": section, "name": name, "text": text}, now=now)

    def names(self, section="negative"):
        return [item["name"] for item in studio_presets.load(self.dir)["sections"][section]]

    def on_disk(self):
        return json.loads(self.file.read_text(encoding="utf-8"))


class ShapeTests(_StoreCase):
    def test_sections_are_the_prompt_boxes_plus_negative(self):
        self.assertEqual(studio_presets.SECTIONS, SECTIONS)
        self.assertEqual(studio_presets.SECTIONS, (*studio_state.PROMPT_SECTIONS, "negative"))

    def test_edge_is_the_one_a_joined_prompt_box_loses(self):
        self.assertEqual(studio_presets.EDGE, studio_state._PROMPT_EDGE)

    def test_missing_file_is_an_empty_document_and_reading_writes_nothing(self):
        doc = studio_presets.load(self.dir)
        self.assertEqual(doc, {"version": 1, "sections": {key: [] for key in SECTIONS},
                               "prefs": {"close_after_insert": False}})
        self.assertEqual(list(self.dir.iterdir()), [])

    def test_file_name_and_place(self):
        self.add()
        self.assertEqual([p.name for p in self.dir.iterdir()], ["prompt_presets.json"])
        self.assertEqual(studio_presets.PRESET_FILE, "prompt_presets.json")
        self.assertEqual(self.on_disk()["version"], studio_presets.VERSION)


class AddTests(_StoreCase):
    def test_add_appends_in_order_and_round_trips(self):
        first = self.add(name="A", text="a, b", now=100)
        second = self.add(name="B", text="c", now=200)
        self.assertNotEqual(first["id"], second["id"])
        items = second["doc"]["sections"]["negative"]
        self.assertEqual(items, [
            {"id": first["id"], "name": "A", "text": "a, b", "created": 100, "used": 0},
            {"id": second["id"], "name": "B", "text": "c", "created": 200, "used": 0}])
        self.assertEqual(studio_presets.load(self.dir), second["doc"])
        self.assertEqual(self.on_disk(), second["doc"])
        self.assertEqual(second["doc"]["sections"]["prompt"], [])

    def test_text_is_kept_exactly_and_the_name_is_tidied(self):
        out = self.add(name="  내   네거티브\n 3 ", text=" 1boy,\nsolo, \n")
        item = out["doc"]["sections"]["negative"][0]
        self.assertEqual((item["name"], item["text"]), ("내 네거티브 3", " 1boy,\nsolo, \n"))

    def test_korean_is_written_as_utf8_text(self):
        self.add(name="글자 막기", text="한글, text")
        raw = self.file.read_bytes()
        self.assertIn("글자 막기".encode("utf-8"), raw)
        self.assertFalse(raw.startswith(b"\xef\xbb\xbf"))

    def test_bad_requests_are_refused_and_write_nothing(self):
        bad = [
            {"op": "add", "section": "bogus", "name": "n", "text": "t"},
            {"op": "add", "section": "negative", "name": "", "text": "t"},
            {"op": "add", "section": "negative", "name": " \n ", "text": "t"},
            {"op": "add", "section": "negative", "name": "x" * (studio_presets.NAME_MAX + 1), "text": "t"},
            {"op": "add", "section": "negative", "name": 5, "text": "t"},
            {"op": "add", "section": "negative", "name": "n", "text": ""},
            {"op": "add", "section": "negative", "name": "n", "text": " ,\n, "},
            {"op": "add", "section": "negative", "name": "n", "text": "x" * (studio_presets.TEXT_MAX + 1)},
            {"op": "add", "section": "negative", "name": "n", "text": ["t"]},
            {"op": "add", "section": "negative", "name": "n"},
            {"op": "nope", "section": "negative"},
            {"section": "negative", "name": "n", "text": "t"},
            "add",
        ]
        for op in bad:
            with self.subTest(op=op), self.assertRaises(studio_presets.PresetError):
                studio_presets.apply(self.dir, op)
        self.assertFalse(self.file.exists())

    def test_limits_are_inclusive(self):
        self.add(name="x" * studio_presets.NAME_MAX, text="y" * studio_presets.TEXT_MAX)
        self.assertEqual(len(self.names()[0]), studio_presets.NAME_MAX)

    def test_a_section_holds_at_most_max_presets(self):
        with mock.patch.object(studio_presets, "MAX_PER_SECTION", 3):
            for index in range(3):
                self.add(name=f"n{index}")
            with self.assertRaises(studio_presets.PresetError):
                self.add(name="one too many")
            self.add(section="prompt", name="another box is fine")
        self.assertEqual(self.names(), ["n0", "n1", "n2"])


class ChangeTests(_StoreCase):
    def setUp(self):
        super().setUp()
        self.ids = [self.add(name=name, text=f"text {name}", now=10)["id"] for name in "ABC"]

    def op(self, op, now=500, **fields):
        return studio_presets.apply(self.dir, {"op": op, "section": "negative", **fields}, now=now)

    def test_update_name_text_or_both(self):
        self.op("update", id=self.ids[1], name=" B2 ")
        self.op("update", id=self.ids[2], text="new c")
        doc = self.op("update", id=self.ids[0], name="A2", text="new a")["doc"]
        got = [(i["name"], i["text"], i["created"], i["used"]) for i in doc["sections"]["negative"]]
        self.assertEqual(got, [("A2", "new a", 10, 0), ("B2", "text B", 10, 0), ("C", "new c", 10, 0)])

    def test_update_refuses_bad_values_and_unknown_presets(self):
        for fields in ({"id": self.ids[0]}, {"id": self.ids[0], "name": ""}, {"id": self.ids[0], "text": " , "},
                       {"id": self.ids[0], "name": 3}, {"name": "x"}):
            with self.subTest(fields=fields), self.assertRaises(studio_presets.PresetError):
                self.op("update", **fields)
        with self.assertRaises(studio_presets.PresetError) as caught:
            self.op("update", id="gone", name="x")
        self.assertEqual(caught.exception.status, 404)
        self.assertEqual(self.names(), ["A", "B", "C"])

    def test_delete_is_idempotent(self):
        self.op("delete", id=self.ids[1])
        doc = self.op("delete", id=self.ids[1])["doc"]  # another tab already removed it
        self.assertEqual([i["name"] for i in doc["sections"]["negative"]], ["A", "C"])

    def test_move_steps_and_stops_at_the_ends(self):
        self.op("move", id=self.ids[2], delta=-1)
        self.assertEqual(self.names(), ["A", "C", "B"])
        self.op("move", id=self.ids[2], delta=-5)
        self.assertEqual(self.names(), ["C", "A", "B"])
        self.op("move", id=self.ids[2], delta=1)
        self.assertEqual(self.names(), ["A", "C", "B"])
        self.op("move", id=self.ids[1], delta=9)
        self.assertEqual(self.names(), ["A", "C", "B"])
        for delta in (0, 1.5, "1", True, None):
            with self.subTest(delta=delta), self.assertRaises(studio_presets.PresetError):
                self.op("move", id=self.ids[0], delta=delta)
        with self.assertRaises(studio_presets.PresetError):
            self.op("move", id="gone", delta=1)

    def test_use_stamps_the_time(self):
        doc = self.op("use", id=self.ids[1], now=777)["doc"]
        self.assertEqual([i["used"] for i in doc["sections"]["negative"]], [0, 777, 0])
        self.assertEqual(self.op("use", id="gone", now=888)["doc"], doc)  # deleted elsewhere: nothing to stamp

    def test_insert_puts_a_deleted_preset_back_where_it_was(self):
        before = studio_presets.load(self.dir)["sections"]["negative"]
        self.op("delete", id=self.ids[1])
        other = self.add(name="other tab", now=20)["id"]  # saved elsewhere meanwhile: it stays
        doc = self.op("insert", item=before[1], index=1)["doc"]
        self.assertEqual(doc["sections"]["negative"][1], before[1])
        self.assertEqual([i["id"] for i in doc["sections"]["negative"]], [self.ids[0], self.ids[1], self.ids[2], other])
        self.assertEqual(self.on_disk(), doc)

    def test_insert_stops_at_the_ends_and_leaves_a_listed_preset_alone(self):
        item = {"id": "zz", "name": "Z", "text": "z", "created": 3, "used": 4}
        self.assertEqual(self.op("insert", item=item, index=99)["doc"]["sections"]["negative"][-1], item)
        self.op("delete", id="zz")
        self.assertEqual(self.op("insert", item=item, index=-5)["doc"]["sections"]["negative"][0], item)
        self.op("insert", item={**item, "name": "again", "text": "other"}, index=2)  # already back (another tab)
        self.assertEqual(self.names(), ["Z", "A", "B", "C"])

    def test_insert_refuses_what_is_not_a_whole_preset(self):
        good = {"id": "zz", "name": "Z", "text": "z", "created": 1, "used": 0}
        for item, index in ((None, 0), ("x", 0), ({**good, "name": ""}, 0), ({**good, "text": " , "}, 0),
                            ({**good, "id": ""}, 0), ({**good, "id": 7}, 0), (good, None), (good, "1"), (good, True),
                            (good, 1.5)):
            with self.subTest(item=item, index=index), self.assertRaises(studio_presets.PresetError):
                self.op("insert", item=item, index=index)
        with mock.patch.object(studio_presets, "MAX_PER_SECTION", 3), self.assertRaises(studio_presets.PresetError):
            self.op("insert", item=good, index=0)
        self.assertEqual(self.names(), ["A", "B", "C"])

    def test_a_whole_list_cannot_be_replaced(self):
        """No 'restore': one tab's copy of a list never overwrites what another tab saved meanwhile."""
        with self.assertRaises(studio_presets.PresetError):
            self.op("restore", items=[])
        self.assertNotIn("restore", studio_presets._OPS)
        self.assertEqual(self.names(), ["A", "B", "C"])

    def test_prefs_keep_only_known_flags(self):
        doc = studio_presets.apply(self.dir, {"op": "prefs", "prefs": {"close_after_insert": True, "bogus": 1}})["doc"]
        self.assertEqual(doc["prefs"], {"close_after_insert": True})
        doc = studio_presets.apply(self.dir, {"op": "prefs", "prefs": {"close_after_insert": "yes"}})["doc"]
        self.assertEqual(doc["prefs"], {"close_after_insert": True})  # not a flag: left as it was
        with self.assertRaises(studio_presets.PresetError):
            studio_presets.apply(self.dir, {"op": "prefs", "prefs": []})
        self.assertEqual(self.names(), ["A", "B", "C"])


class DamagedFileTests(_StoreCase):
    def baks(self):
        return sorted(p.name for p in self.dir.iterdir() if ".bak" in p.name)

    def test_corrupt_file_starts_empty_and_is_kept_as_bak_before_the_first_write(self):
        self.file.write_bytes(b'{"version": 1, "sections": {"negative": [{"id"')
        self.assertEqual(studio_presets.load(self.dir)["sections"]["negative"], [])
        self.assertEqual(self.baks(), [])  # reading never writes
        self.add(name="fresh")
        self.assertEqual(self.baks(), ["prompt_presets.json.bak"])
        self.assertEqual((self.dir / "prompt_presets.json.bak").read_bytes(),
                         b'{"version": 1, "sections": {"negative": [{"id"')
        self.assertEqual(self.names(), ["fresh"])

    def test_not_an_object_counts_as_corrupt(self):
        for raw in ("[1, 2]", '"text"', "null", ""):
            with self.subTest(raw=raw):
                self.file.write_text(raw, encoding="utf-8")
                self.assertEqual(studio_presets.load(self.dir), studio_presets.empty_doc())
                self.assertEqual(studio_presets.load_state(self.dir), (studio_presets.empty_doc(), True))

    def test_only_a_file_that_cannot_be_parsed_is_called_broken(self):
        self.assertEqual(studio_presets.load_state(self.dir), (studio_presets.empty_doc(), False))  # no file
        self.file.write_text('{"sections": {"negative": [{"name": "hand made", "text": "a, b"}, "junk"]}}', encoding="utf-8")
        doc, broken = studio_presets.load_state(self.dir)
        self.assertEqual(([i["name"] for i in doc["sections"]["negative"]], broken), (["hand made"], False))
        self.file.write_bytes(b"\xff\xfe not utf-8")
        self.assertEqual(studio_presets.load_state(self.dir), (studio_presets.empty_doc(), True))

    def test_a_file_with_a_utf8_bom_is_read(self):
        """Windows PowerShell 5.1 and old Notepad put a BOM in front: the presets are there, nothing is set aside."""
        self.add(name="가", text="하나, 둘")
        self.add(name="나", text="셋")
        clean = self.file.read_bytes()
        self.file.write_bytes(b"\xef\xbb\xbf" + clean)
        self.assertEqual(self.names(), ["가", "나"])
        self.assertEqual(studio_presets.load_state(self.dir)[1], False)
        self.add(name="다")
        self.assertEqual(self.names(), ["가", "나", "다"])
        self.assertEqual(self.baks(), [])  # the same document: nothing to keep
        self.assertFalse(self.file.read_bytes().startswith(b"\xef\xbb\xbf"))  # written back without the BOM

    def test_a_file_that_cannot_be_opened_is_not_an_empty_one(self):
        """A sharing violation / access denied is not 'corrupt': reading fails and no change overwrites the file."""
        ids = [self.add(name=name)["id"] for name in "ABC"]
        whole = self.file.read_bytes()
        real = Path.read_bytes

        def locked(path):
            if path.name == studio_presets.PRESET_FILE:
                raise PermissionError(13, "sharing violation")
            return real(path)

        with mock.patch.object(Path, "read_bytes", locked):
            with self.assertRaises(OSError):
                studio_presets.load(self.dir)
            for op in ({"op": "use", "section": "negative", "id": ids[0]},
                       {"op": "add", "section": "negative", "name": "new", "text": "t"}):
                with self.subTest(op=op["op"]), self.assertRaises(OSError):
                    studio_presets.apply(self.dir, op)
        self.assertEqual(self.file.read_bytes(), whole)
        self.assertEqual(self.names(), ["A", "B", "C"])
        self.assertEqual(self.baks(), [])

    def test_a_damaged_file_is_not_overwritten_when_its_backup_cannot_be_written(self):
        self.file.write_text("broken {", encoding="utf-8")
        real = studio_presets.atomic_path

        def no_backup(target):
            if ".bak" in Path(target).name:
                raise PermissionError("no room")
            return real(target)

        with mock.patch.object(studio_presets, "atomic_path", no_backup):
            with self.assertRaises(OSError):
                self.add()
        self.assertEqual(self.file.read_text(encoding="utf-8"), "broken {")
        self.assertEqual(self.baks(), [])
        self.add(name="later")  # the disk is fine again: backed up, then written
        self.assertEqual((self.dir / "prompt_presets.json.bak").read_text(encoding="utf-8"), "broken {")
        self.assertEqual(self.names(), ["later"])

    def test_the_backup_is_the_file_that_was_read(self):
        """One read serves both: what is parsed and what is kept, so no other file can slip in between."""
        self.file.write_text("broken {", encoding="utf-8")
        real, reads = Path.read_bytes, []

        def counting(path):
            reads.append(path.name)
            return real(path)

        with mock.patch.object(Path, "read_bytes", counting):
            self.add()
        self.assertEqual(reads.count(studio_presets.PRESET_FILE), 1)
        self.assertEqual((self.dir / "prompt_presets.json.bak").read_text(encoding="utf-8"), "broken {")

    def test_entries_that_cannot_be_read_are_dropped_and_the_file_is_backed_up_once(self):
        good = {"id": "g1", "name": "Good", "text": "a, b", "created": 5, "used": 6}
        raw = {"version": 1, "prefs": {"close_after_insert": True, "x": 1}, "sections": {
            "negative": [good, {"id": "g1", "name": "same id", "text": "c"}, {"name": "", "text": "x"}, "junk",
                         {"id": "g3", "name": "no text"}, {"id": 7, "name": "Odd", "text": "d", "created": -1, "used": "x"}],
            "prompt": "not a list", "unknown_box": [good]}}
        self.file.write_text(json.dumps(raw), encoding="utf-8")
        doc = studio_presets.load(self.dir)
        items = doc["sections"]["negative"]
        self.assertEqual(items[0], good)
        self.assertEqual([(i["name"], i["text"], i["created"], i["used"]) for i in items[1:]],
                         [("same id", "c", 0, 0), ("Odd", "d", 0, 0)])
        self.assertEqual(len({i["id"] for i in items}), 3)
        self.assertTrue(all(isinstance(i["id"], str) and i["id"] for i in items))
        self.assertEqual(doc["sections"]["prompt"], [])
        self.assertNotIn("unknown_box", doc["sections"])
        self.assertEqual(doc["prefs"], {"close_after_insert": True})
        self.add(name="next")
        self.add(name="again")
        self.assertEqual(self.baks(), ["prompt_presets.json.bak"])  # the file was clean before the second write
        self.assertEqual(json.loads((self.dir / "prompt_presets.json.bak").read_text(encoding="utf-8")), raw)
        self.assertEqual(self.names(), ["Good", "same id", "Odd", "next", "again"])

    def test_a_file_of_a_newer_version_is_backed_up_before_it_is_rewritten(self):
        raw = {"version": 2, "sections": {"negative": [{"id": "a", "name": "A", "text": "t", "created": 1, "used": 0}]},
               "prefs": {"close_after_insert": False}, "future": {"x": 1}}
        self.file.write_text(json.dumps(raw), encoding="utf-8")
        self.assertEqual(self.names(), ["A"])
        self.add(name="B")
        self.assertEqual(json.loads((self.dir / "prompt_presets.json.bak").read_text(encoding="utf-8")), raw)
        self.assertEqual(self.on_disk()["version"], 1)

    def test_a_clean_file_is_never_backed_up(self):
        self.add(name="A")
        self.add(name="B")
        studio_presets.apply(self.dir, {"op": "prefs", "prefs": {"close_after_insert": True}})
        self.assertEqual(self.baks(), [])

    def test_a_second_damaged_file_gets_its_own_bak_and_the_same_one_is_not_copied_twice(self):
        self.file.write_text("first {", encoding="utf-8")
        self.add()
        self.file.write_text("first {", encoding="utf-8")
        self.add()
        self.assertEqual(self.baks(), ["prompt_presets.json.bak"])
        self.file.write_text("second {", encoding="utf-8")
        self.add()
        self.assertEqual(self.baks(), ["prompt_presets.json.bak", "prompt_presets.json.bak1"])
        self.assertEqual((self.dir / "prompt_presets.json.bak").read_text(encoding="utf-8"), "first {")
        self.assertEqual((self.dir / "prompt_presets.json.bak1").read_text(encoding="utf-8"), "second {")


class AtomicWriteTests(_StoreCase):
    def test_a_failed_write_leaves_the_old_file_whole_and_no_temp_file(self):
        self.add(name="kept")
        before = self.file.read_bytes()
        with mock.patch.object(studio_presets.json, "dumps", side_effect=OSError("disk full")):
            with self.assertRaises(OSError):
                self.add(name="lost")
        self.assertEqual(self.file.read_bytes(), before)
        self.assertEqual([p.name for p in self.dir.iterdir()], ["prompt_presets.json"])

    def test_a_write_cut_short_never_shows_as_the_file(self):
        self.add(name="kept")
        before = self.file.read_bytes()
        real = Path.write_text

        def half(path, data, *args, **kwargs):
            real(path, data[: len(data) // 2], *args, **kwargs)
            raise OSError("power cut")

        with mock.patch.object(Path, "write_text", half), self.assertRaises(OSError):
            self.add(name="lost")
        self.assertEqual(self.file.read_bytes(), before)
        self.assertEqual([p.name for p in self.dir.iterdir()], ["prompt_presets.json"])

    def test_writers_at_once_lose_nothing(self):
        errors = []

        def work(tag):
            try:
                for index in range(10):
                    self.add(section="negative" if index % 2 else "prompt", name=f"{tag}-{index}", text=f"t{index}")
                    studio_presets.load(self.dir)
            except Exception as exc:  # pragma: no cover - the assertion below reports it
                errors.append(exc)

        threads = [threading.Thread(target=work, args=(f"w{n}",)) for n in range(8)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(30)
        self.assertEqual(errors, [])
        doc = self.on_disk()
        names = self.names("negative") + self.names("prompt")
        self.assertEqual(sorted(names), sorted(f"w{n}-{i}" for n in range(8) for i in range(10)))
        ids = [item["id"] for key in ("negative", "prompt") for item in doc["sections"][key]]
        self.assertEqual(len(set(ids)), 80)
        self.assertEqual([p.name for p in self.dir.iterdir()], ["prompt_presets.json"])

    def test_nothing_is_written_outside_the_store_folder(self):
        before = set(self.root.rglob("*"))
        self.file.write_text("{ broken", encoding="utf-8")
        self.add(name="../../evil", text="x")
        new = {p for p in self.root.rglob("*") if p.is_file()} - before
        self.assertEqual({p.name for p in new}, {"prompt_presets.json", "prompt_presets.json.bak"})
        for path in new:
            self.assertEqual(path.parent, self.dir)


class GoldenTests(unittest.TestCase):
    """The same table drives ``preset_logic.js`` (js_tests/preset_logic.test.mjs)."""

    @classmethod
    def setUpClass(cls):
        cls.golden = json.loads(GOLDEN.read_text(encoding="utf-8"))

    def test_edge(self):
        self.assertEqual(self.golden["edge"], studio_presets.EDGE)

    def test_match_rows(self):
        self.assertGreaterEqual(len(self.golden["match"]), 8)
        for row in self.golden["match"]:
            with self.subTest(case=row["case"]):
                hit = studio_presets.match_preset(row["list"], row["text"])
                self.assertEqual(hit["id"] if hit else None, row["id"])

    def test_match_ignores_what_is_not_a_preset(self):
        self.assertIsNone(studio_presets.match_preset(None, "x"))
        self.assertIsNone(studio_presets.match_preset([{"text": 5}, "x"], "x"))
        self.assertIsNone(studio_presets.match_preset([{"id": "a", "text": "x"}], None))

    def test_name_rows(self):
        for row in self.golden["names"]:
            with self.subTest(name=row["in"]):
                if row["out"] is None:
                    with self.assertRaises(studio_presets.PresetError):
                        studio_presets.clean_name(row["in"])
                else:
                    self.assertEqual(studio_presets.clean_name(row["in"]), row["out"])

    def test_text_rows(self):
        for row in self.golden["texts"]:
            with self.subTest(text=row["in"]):
                if row["ok"]:
                    self.assertEqual(studio_presets.clean_text(row["in"]), row["in"])
                else:
                    with self.assertRaises(studio_presets.PresetError):
                        studio_presets.clean_text(row["in"])


class RouteTests(_StoreCase):
    def setUp(self):
        super().setUp()
        fake = SimpleNamespace(get_user_directory=lambda: str(self.root / "user"))
        patch = mock.patch.object(preset_routes, "folder_paths_module", lambda: fake)
        patch.start()
        self.addCleanup(patch.stop)

    def test_get_answers_the_document_and_the_limits(self):
        status, body = preset_routes.handle_presets_get(None)
        self.assertEqual(status, 200)
        self.assertEqual(body, {"doc": studio_presets.empty_doc(), "broken": False,
                                "limits": {"name": studio_presets.NAME_MAX, "text": studio_presets.TEXT_MAX,
                                           "count": studio_presets.MAX_PER_SECTION}})

    def test_get_says_when_the_file_could_not_be_parsed(self):
        self.file.write_text("broken {", encoding="utf-8")
        status, body = preset_routes.handle_presets_get(None)
        self.assertEqual((status, body["broken"], body["doc"]), (200, True, studio_presets.empty_doc()))
        preset_routes.handle_presets_post({"op": "add", "section": "negative", "name": "n", "text": "t"})
        self.assertEqual(preset_routes.handle_presets_get(None)[1]["broken"], False)  # kept as .bak, written anew

    def test_get_is_a_500_when_the_file_cannot_be_opened(self):
        with mock.patch.object(studio_presets, "load_state", side_effect=PermissionError("locked")):
            status, payload = preset_routes.handle_presets_get(None)
        self.assertEqual(status, 500)
        self.assertIn("locked", payload["error"])

    def test_get_makes_the_store_folder_but_no_file(self):
        fresh = self.root / "other"
        with mock.patch.object(preset_routes, "folder_paths_module",
                               lambda: SimpleNamespace(get_user_directory=lambda: str(fresh))):
            self.assertEqual(preset_routes.handle_presets_get(None)[0], 200)
        self.assertEqual(list((fresh / "default" / "ai_studio").iterdir()), [])

    def test_post_applies_one_change_and_answers_the_new_document(self):
        status, body = preset_routes.handle_presets_post({"op": "add", "section": "prompt_artist", "name": "Ena",
                                                          "text": "@enaa97"})
        self.assertEqual(status, 200)
        self.assertEqual(sorted(body), ["doc", "id", "limits"])
        item = body["doc"]["sections"]["prompt_artist"][0]
        self.assertEqual((item["id"], item["name"], item["text"], item["used"]), (body["id"], "Ena", "@enaa97", 0))
        self.assertGreater(item["created"], 1_700_000_000)
        self.assertEqual(preset_routes.handle_presets_get(None)[1]["doc"], body["doc"])
        self.assertTrue(self.file.is_file())
        status, body = preset_routes.handle_presets_post({"op": "delete", "section": "prompt_artist", "id": body["id"]})
        self.assertEqual((status, sorted(body), body["doc"]["sections"]["prompt_artist"]), (200, ["doc", "limits"], []))

    def test_refusals_are_400_with_the_reason_and_an_unknown_preset_is_404(self):
        for body in ([], None, "x", {"op": "add", "section": "negative", "name": "", "text": "t"}, {"op": "what"}):
            with self.subTest(body=body):
                status, payload = preset_routes.handle_presets_post(body)
                self.assertEqual(status, 400)
                self.assertTrue(payload["error"])
        status, payload = preset_routes.handle_presets_post({"op": "update", "section": "negative", "id": "gone",
                                                             "name": "x"})
        self.assertEqual(status, 404)
        self.assertTrue(payload["error"])

    def test_a_disk_error_is_a_500_with_the_reason(self):
        with mock.patch.object(studio_presets, "apply", side_effect=PermissionError("locked")):
            status, payload = preset_routes.handle_presets_post({"op": "add", "section": "negative", "name": "n",
                                                                 "text": "t"})
        self.assertEqual(status, 500)
        self.assertIn("locked", payload["error"])


class RegistrationTests(unittest.TestCase):
    def test_register_adds_get_and_post_presets(self):
        added = []
        route = lambda handle, read="json": (handle, read)
        preset_routes.register_preset_handlers(lambda method, path, handler: added.append((method, path, handler)), route)
        self.assertEqual(added, [("GET", "/presets", (preset_routes.handle_presets_get, "none")),
                                 ("POST", "/presets", (preset_routes.handle_presets_post, "json"))])

    def test_the_studio_registers_them_with_its_own_routes(self):
        saved = dict(studio_routes._HANDLERS)
        self.addCleanup(lambda: (studio_routes._HANDLERS.clear(), studio_routes._HANDLERS.update(saved)))
        studio_routes._HANDLERS.clear()
        studio_routes.register_studio_handlers()
        self.assertIn(("GET", "/presets"), studio_routes._HANDLERS)
        self.assertIn(("POST", "/presets"), studio_routes._HANDLERS)
        self.assertIn(("GET", "/choices"), studio_routes._HANDLERS)

    def test_the_routes_are_mounted_under_the_studio_prefix(self):
        saved = dict(studio_routes._HANDLERS)
        self.addCleanup(lambda: (studio_routes._HANDLERS.clear(), studio_routes._HANDLERS.update(saved)))
        studio_routes._HANDLERS.clear()
        registered = []
        instance = SimpleNamespace(routes=SimpleNamespace(
            route=lambda method, path: lambda handler: registered.append((method, path)) or handler))
        modules = {"server": SimpleNamespace(PromptServer=SimpleNamespace(instance=instance))}
        preset_routes.register_preset_handlers(studio_routes.add_handler, studio_routes._route)
        with mock.patch.dict("sys.modules", modules):
            self.assertTrue(studio_routes.register_routes())
        self.assertEqual(sorted(registered), [("GET", "/aistudio/studio/presets"), ("POST", "/aistudio/studio/presets")])


if __name__ == "__main__":
    unittest.main()
