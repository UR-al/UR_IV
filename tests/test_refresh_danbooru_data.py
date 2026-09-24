"""Regression tests for the Search/Event dataset refresh transaction."""

from __future__ import annotations

import json
import csv
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

from tools.refresh_danbooru_data import EVENT_COLUMNS, SEARCH_COLUMNS, main, validate_manifest


def _post(post_id: int, rating: str, parent_id=None) -> dict:
    return {
        "id": post_id,
        "parent_id": parent_id,
        "rating": rating,
        "image_width": 1024,
        "image_height": 768,
        "score": post_id,
        "fav_count": post_id // 2,
        "tag_string_general": f"post_{post_id}",
        "tag_string_character": "test_character",
        "tag_string_copyright": "test_series",
        "tag_string_artist": "test_artist",
        "tag_string_meta": "",
        "has_children": parent_id is None,
        "has_visible_children": parent_id is None,
    }


class DanbooruDatasetRefreshTests(unittest.TestCase):
    def test_legacy_archive_rejects_non_csv_output_without_overwriting_it(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            protected = root / "dataset_manifest.json"
            original = b'{"keep": true}'
            protected.write_bytes(original)

            with redirect_stdout(StringIO()), redirect_stderr(StringIO()):
                result = main(
                    [
                        "archive-legacy-tags",
                        "--input-dir",
                        str(root),
                        "--latest-label",
                        "latest",
                        "--legacy-labels",
                        "old",
                        "--output",
                        str(protected),
                    ]
                )

            self.assertEqual(result, 1)
            self.assertEqual(protected.read_bytes(), original)

    def test_builds_search_and_cross_rating_complete_event_shards(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "posts.parquet"
            output = root / "runtime"
            frame = pd.DataFrame(
                [
                    _post(1, "g"),
                    _post(2, "s", 1),
                    _post(3, "q"),
                    _post(4, "q", 3),
                    _post(5, "e"),
                    _post(6, "e", 5),
                    _post(7, "g", 1),
                ]
            )
            frame["parent_id"] = frame["parent_id"].astype("Int64")
            frame.to_parquet(source, index=False)

            with redirect_stdout(StringIO()):
                result = main(
                    [
                        "posts",
                        "--source",
                        str(source),
                        "--output-dir",
                        str(output),
                        "--dataset-label",
                        "test_release",
                        "--source-url",
                        "https://example.test/posts.parquet",
                        "--source-revision",
                        "immutable-test-revision",
                        "--snapshot-at",
                        "2026-07-13T00:00:00Z",
                        "--batch-rows",
                        "3",
                        "--row-group-rows",
                        "2",
                    ]
                )
            self.assertEqual(result, 0)

            manifest_path = output / "dataset_manifest.json"
            manifest = validate_manifest(manifest_path, output)
            self.assertEqual(len(manifest["artifacts"]), 8)
            self.assertEqual(manifest["source"]["revision"], "immutable-test-revision")

            search_s = pq.read_table(output / "danbooru_test_release_s.parquet")
            self.assertEqual(tuple(search_s.column_names), SEARCH_COLUMNS)
            self.assertEqual(search_s.num_rows, 1)

            event_s = pq.read_table(output / "danbooru_sorted" / "danbooru_s.parquet")
            self.assertEqual(tuple(event_s.column_names), EVENT_COLUMNS)
            self.assertEqual(set(event_s["id"].to_pylist()), {1, 2})
            self.assertEqual(
                set(event_s["parent_id"].drop_null().to_pylist()),
                {1},
            )

            stored = json.loads(manifest_path.read_text(encoding="utf-8"))
            self.assertTrue(stored["build"]["cross_rating_parent_duplication"])

    def test_tag_columns_are_published_lower_case(self):
        """Search 워커는 소문자 사본 없이 원본 태그 컬럼으로 매칭하므로 빌드가 소문자를 보장한다."""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "posts.parquet"
            output = root / "runtime"
            mixed = _post(1, "g")
            mixed.update({
                "tag_string_general": "Long_Hair SMILE",
                "tag_string_character": "Hatsune_Miku",
                "tag_string_copyright": "VOCALOID",
                "tag_string_artist": "SomeArtist",
                "tag_string_meta": "HighRes",
            })
            missing = _post(2, "g", 1)   # post 1 의 자식 — Event 그래프에도 두 행이 실린다
            missing["tag_string_meta"] = None
            frame = pd.DataFrame([mixed, missing])
            frame["parent_id"] = frame["parent_id"].astype("Int64")
            frame.to_parquet(source, index=False)

            with redirect_stdout(StringIO()):
                result = main(
                    [
                        "posts",
                        "--source",
                        str(source),
                        "--output-dir",
                        str(output),
                        "--dataset-label",
                        "case_release",
                        "--source-url",
                        "https://example.test/posts.parquet",
                        "--source-revision",
                        "immutable-test-revision",
                        "--snapshot-at",
                        "2026-07-13T00:00:00Z",
                    ]
                )
            self.assertEqual(result, 0)

            search_g = pq.read_table(output / "danbooru_case_release_g.parquet").to_pylist()
            by_general = {row["general"]: row for row in search_g}
            row = by_general["long_hair smile"]
            self.assertEqual(row["character"], "hatsune_miku")
            self.assertEqual(row["copyright"], "vocaloid")
            self.assertEqual(row["artist"], "someartist")
            self.assertEqual(row["meta"], "highres")
            self.assertIsNone(by_general["post_2"]["meta"])   # null 은 null 그대로
            for record in search_g:
                for column in ("general", "character", "copyright", "artist", "meta"):
                    value = record[column]
                    if value is not None:
                        self.assertEqual(value, value.lower(), (column, value))

            event_g = pq.read_table(
                output / "danbooru_sorted" / "danbooru_g.parquet"
            ).to_pylist()
            event_row = next(r for r in event_g if r["id"] == 1)
            self.assertEqual(event_row["tag_string_general"], "long_hair smile")

    def test_archives_unique_tags_missing_from_latest_release(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)

            def write_release(label, *, comma_separated, rows):
                for rating in ("g", "s", "q", "e"):
                    selected = rows if rating == "g" else []
                    frame = pd.DataFrame(
                        selected,
                        columns=(
                            "general",
                            "character",
                            "copyright",
                            "artist",
                            "metadata" if comma_separated else "meta",
                        ),
                    )
                    frame.insert(0, "rating", rating)
                    frame.to_parquet(root / f"danbooru_{label}_{rating}.parquet", index=False)

            write_release(
                "latest",
                comma_separated=False,
                rows=[
                    {
                        "general": "shared_tag latest_tag tag_moved_category",
                        "character": "latest_character",
                        "copyright": "",
                        "artist": "",
                        "meta": "",
                    }
                ],
            )
            write_release(
                "old_a",
                comma_separated=True,
                rows=[
                    {
                        "general": "shared tag, old only tag, tag moved category",
                        "character": "old character",
                        "copyright": "",
                        "artist": "repeat old tag",
                        "metadata": "legacy metadata",
                    }
                ],
            )
            write_release(
                "old_b",
                comma_separated=False,
                rows=[
                    {
                        "general": "old_only_tag second_old_tag",
                        "character": "",
                        "copyright": "repeat_old_tag",
                        "artist": "",
                        "meta": "",
                    }
                ],
            )
            output = root / "legacy.csv"

            with redirect_stdout(StringIO()):
                result = main(
                    [
                        "archive-legacy-tags",
                        "--input-dir",
                        str(root),
                        "--latest-label",
                        "latest",
                        "--legacy-labels",
                        "old_a",
                        "old_b",
                        "--output",
                        str(output),
                    ]
                )

            self.assertEqual(result, 0)
            with output.open("r", encoding="utf-8-sig", newline="") as handle:
                rows = list(csv.DictReader(handle))
            by_tag = {row["tag"]: row for row in rows}
            self.assertNotIn("shared_tag", by_tag)
            self.assertNotIn("tag_moved_category", by_tag)
            self.assertEqual(
                by_tag["old_only_tag"]["legacy_releases"],
                "old_a|old_b",
            )
            self.assertEqual(
                by_tag["repeat_old_tag"]["legacy_categories"],
                "copyright|artist",
            )
            self.assertEqual(
                list(by_tag),
                sorted(by_tag),
            )


if __name__ == "__main__":
    unittest.main()
