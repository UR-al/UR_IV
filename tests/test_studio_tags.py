"""Anima studio prompt tag autocomplete: the pure search (studio_tags), its route, and the generated list."""
import csv
import importlib.util
import tempfile
import threading
import unittest
from pathlib import Path
from unittest import mock

from comfy_custom_nodes.ai_studio_forge_parity import studio_routes, studio_tags
from comfy_custom_nodes.ai_studio_forge_parity.studio_tags import TagIndex

ROOT = Path(__file__).resolve().parents[1]
BUILDER = ROOT / "tools" / "build_studio_tag_list.py"
PACK_TAGS = ROOT / "comfy_custom_nodes" / "ai_studio_forge_parity" / "tags" / "danbooru_tags.csv"

# name, category, count, aliases, ko
ROWS = [
    ("1girl", "0", "8359716", "1girls,sole_female", "한 명|소녀"),
    ("long_hair", "0", "6168790", "/lh,longhair", "긴 머리|장발"),
    ("long_sleeves", "0", "2000000", "", "긴팔"),
    ("longing", "0", "50", "", ""),
    ("blue_hair", "0", "900000", "bluehair", "파란 머리"),
    ("very_long_hair", "0", "1500000", "", "아주 긴 머리"),
    ("hatsune_miku", "4", "300000", "miku", ""),
    ("hatsune_miku_(cosplay)", "0", "1200", "", ""),
    ("^_^", "0", "40000", "", ""),
    ("greg_rutkowski", "1", "700", "", ""),
    ("mikumo", "0", "300", "", ""),
]


def names(results):
    return [item["name"] for item in results]


class SearchTests(unittest.TestCase):
    def setUp(self):
        self.index = TagIndex(ROWS)

    def test_prefix_first_then_count_descending(self):
        # long_hair/long_sleeves/longing start with "long"; very_long_hair only contains it (after the prefix hits)
        self.assertEqual(names(self.index.search("long")), ["long_hair", "long_sleeves", "longing", "very_long_hair"])

    def test_contains_matches_are_ordered_by_count(self):
        results = self.index.search("hair")  # nothing starts with "hair": all contains matches
        self.assertEqual(names(results), ["long_hair", "very_long_hair", "blue_hair"])
        counts = [item["count"] for item in results]
        self.assertEqual(counts, sorted(counts, reverse=True))

    def test_alias_is_found_and_reported_but_never_duplicates_the_tag(self):
        results = self.index.search("sole")
        self.assertEqual(names(results), ["1girl"])
        self.assertEqual(results[0]["alias"], "sole_female")
        # the name matches too, so no alias is reported
        self.assertNotIn("alias", self.index.search("1gir")[0])
        self.assertEqual(names(self.index.search("1girls")), ["1girl"])

    def test_alias_prefix_ranks_with_name_prefix_by_count(self):
        # "miku" is an alias of hatsune_miku (300000); mikumo (300) is a name prefix match
        self.assertEqual(names(self.index.search("miku"))[:2], ["hatsune_miku", "mikumo"])

    def test_case_underscore_and_space_do_not_matter(self):
        expected = names(self.index.search("long_hair"))
        self.assertEqual(expected[0], "long_hair")
        self.assertEqual(names(self.index.search("Long Hair")), expected)
        self.assertEqual(names(self.index.search("  LONG_HAIR")), expected)
        self.assertEqual(names(self.index.search("long   hair")), expected)

    def test_limit_and_empty_query(self):
        self.assertEqual(len(self.index.search("long", limit=2)), 2)
        self.assertEqual(self.index.search("long", limit=0), [])
        self.assertEqual(self.index.search(""), [])
        self.assertEqual(self.index.search("   "), [])
        self.assertEqual(self.index.search("zzzzzz"), [])

    def test_result_shape(self):
        item = self.index.search("blue_h")[0]
        self.assertEqual(item, {"name": "blue_hair", "category": 0, "count": 900000, "ko": "파란 머리"})
        self.assertEqual(self.index.search("greg")[0]["category"], 1)
        self.assertEqual(self.index.search("hatsune_miku")[0]["category"], 4)
        self.assertNotIn("ko", self.index.search("greg")[0])

    def test_special_character_tags_are_searchable(self):
        self.assertEqual(names(self.index.search("^")), ["^_^"])

    def test_korean_query_matches_korean_keywords(self):
        results = self.index.search("긴")
        # prefix: "긴 머리" (long_hair 6.1M), "긴팔" (long_sleeves 2M); contains: "아주 긴 머리" (very_long_hair)
        self.assertEqual(names(results), ["long_hair", "long_sleeves", "very_long_hair"])
        self.assertEqual(results[0]["ko"], "긴 머리")
        self.assertEqual(names(self.index.search("장")), ["long_hair"])
        self.assertEqual(self.index.search("없는말"), [])

    def test_ko_label_is_the_first_keyword(self):
        self.assertEqual(self.index.search("1girl")[0]["ko"], "한 명")

    def test_loads_from_csv_with_or_without_header(self):
        header = ["name", "category", "count", "aliases", "ko"]
        for with_header in (True, False):
            with self.subTest(with_header=with_header), tempfile.TemporaryDirectory() as folder:
                path = Path(folder) / "tags.csv"
                with open(path, "w", encoding="utf-8", newline="") as handle:
                    writer = csv.writer(handle, lineterminator="\n")
                    if with_header:
                        writer.writerow(header)
                    writer.writerows(ROWS)
                index = TagIndex.from_csv(path)
                self.assertEqual(len(index), len(ROWS))
                self.assertEqual(names(index.search("blue"))[0], "blue_hair")

    def test_missing_or_short_rows_do_not_break_the_index(self):
        index = TagIndex([("solo", "0", "100"), ("bad", "x", "y", "", ""), ("", "0", "1", "", ""), ("ok", "0", "5", "", "")])
        self.assertEqual(names(index.search("so")), ["solo"])
        self.assertEqual(names(index.search("ok")), ["ok"])
        self.assertEqual(index.search("bad"), [])
        self.assertEqual(len(TagIndex.from_csv(Path("no_such_tags.csv"))), 0)


class DefaultIndexTests(unittest.TestCase):
    def test_default_index_is_read_once_even_from_many_threads(self):
        calls = []
        real = TagIndex.from_csv

        def counting(path):
            calls.append(path)
            return real(path)

        with mock.patch.object(studio_tags, "_default", None), mock.patch.object(TagIndex, "from_csv", staticmethod(counting)):
            threads = [threading.Thread(target=studio_tags.search_tags, args=("long",)) for _ in range(8)]
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join()
        self.assertEqual(len(calls), 1)

    def test_module_has_no_comfyui_or_torch_import(self):
        source = Path(studio_tags.__file__).read_text(encoding="utf-8")
        for word in ("import torch", "folder_paths", "import server", "comfy."):
            self.assertNotIn(word, source)


class HandleTagsTests(unittest.TestCase):
    def test_empty_query_is_an_empty_list(self):
        for query in ({}, {"q": ""}, {"q": "  "}, None):
            self.assertEqual(studio_routes.handle_tags(query), (200, {"items": []}))

    def test_search_is_called_with_the_clamped_limit(self):
        cases = {None: 12, "5": 5, "0": 1, "-3": 1, "99": 30, "30": 30, "1": 1}
        for limit, expected in cases.items():
            query = {"q": "lo"} if limit is None else {"q": "lo", "limit": limit}
            with self.subTest(limit=limit), mock.patch.object(studio_routes.studio_tags, "search_tags", return_value=[{"name": "x"}]) as search:
                self.assertEqual(studio_routes.handle_tags(query), (200, {"items": [{"name": "x"}]}))
                search.assert_called_once_with("lo", expected)

    def test_bad_limit_is_400(self):
        for limit in ("abc", "1.5", ""):
            with self.subTest(limit=limit):
                status, payload = studio_routes.handle_tags({"q": "lo", "limit": limit})
                self.assertEqual(status, 400)
                self.assertIn("error", payload)

    def test_real_list_answers(self):
        status, payload = studio_routes.handle_tags({"q": "long_h", "limit": "3"})
        self.assertEqual(status, 200)
        self.assertEqual(payload["items"][0]["name"], "long_hair")
        self.assertEqual(len(payload["items"]), 3)

    def test_route_is_registered(self):
        self.assertIn(("GET", "/tags"), studio_routes._HANDLERS)


class GeneratedListTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        spec = importlib.util.spec_from_file_location("build_studio_tag_list", BUILDER)
        cls.builder = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cls.builder)

    def test_build_rows_keeps_count_20_and_over_and_attaches_korean(self):
        catalog = [
            ["1girl", "0", "8359716", "1girls,sole_female"],
            ["rare_tag", "0", "19", ""],
            ["edge_tag", "5", "20", "e"],
            ["broken", "0", "many", ""],
            ["", "0", "100", ""],
            ["long_hair", "0", "6168790", "/lh,longhair"],
        ]
        korean = {"long_hair": "긴 머리|장발"}
        rows = self.builder.build_rows(catalog, korean)
        self.assertEqual(rows, [
            ["1girl", "0", "8359716", "1girls,sole_female", ""],
            ["long_hair", "0", "6168790", "/lh,longhair", "긴 머리|장발"],
            ["edge_tag", "5", "20", "e", ""],
        ])

    def test_korean_labels_put_plain_keywords_before_category_markers(self):
        labels = self.builder.korean_labels([("long hair", "<헤어스타일>, 긴 머리, 장발"), ("solo", "<인원>, 솔로, 1인"), ("x", "")])
        self.assertEqual(labels, {"long_hair": "긴 머리|장발|헤어스타일", "solo": "솔로|1인|인원"})

    def test_pack_list_is_the_generated_one(self):
        with tempfile.TemporaryDirectory() as folder:
            made = self.builder.write_tag_list(Path(folder) / "danbooru_tags.csv")
            self.assertFalse(made.read_bytes().startswith(b"\xef\xbb\xbf"))
            # as text: a checkout with autocrlf may turn the saved file's LF into CRLF
            self.assertEqual(made.read_text(encoding="utf-8"), PACK_TAGS.read_text(encoding="utf-8"))

    def test_pack_list_content(self):
        with open(PACK_TAGS, encoding="utf-8", newline="") as handle:
            rows = list(csv.reader(handle))
        self.assertEqual(rows[0], ["name", "category", "count", "aliases", "ko"])
        body = rows[1:]
        self.assertGreater(len(body), 100000)
        self.assertTrue(all(len(row) == 5 and int(row[2]) >= 20 for row in body))
        self.assertEqual(body[0][0], "1girl")
        self.assertTrue(any(row[4] for row in body))
        self.assertLess(PACK_TAGS.stat().st_size, 8 * 1024 * 1024)

    def test_real_searches(self):
        index = TagIndex.from_csv(PACK_TAGS)
        self.assertEqual(names(index.search("1gir"))[0], "1girl")
        self.assertEqual(names(index.search("long_h"))[0], "long_hair")
        self.assertIn("long_hair", names(index.search("긴 머리")))


if __name__ == "__main__":
    unittest.main()
