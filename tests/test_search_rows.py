"""core.search_rows — 검색 워커·bridge·parquet 가져오기가 공유하는 행 정규화."""
from __future__ import annotations

import json
import math
import unittest

import numpy as np
import pandas as pd

from core.search_rows import (
    SEARCH_RESULT_CAP,
    SEARCH_ROW_KEYS,
    NormalizedSearchRows,
    dimension_value,
    is_missing,
    normalize_search_row,
    normalize_search_rows_in_place,
    search_rows_from_frame,
    text_value,
)


class SearchRowNormalizationTests(unittest.TestCase):
    def test_contract_keys_and_cap(self) -> None:
        self.assertEqual(
            SEARCH_ROW_KEYS,
            ('copyright', 'character', 'artist', 'general', 'rating',
             'image_width', 'image_height'),
        )
        self.assertEqual(SEARCH_RESULT_CAP, 500_000)

    def test_missing_values_never_leak_as_nan_strings(self) -> None:
        row = normalize_search_row({
            'general': float('nan'),
            'character': None,
            'copyright': pd.NA,
            'artist': np.float64('nan'),
            'rating': float('nan'),
            'image_width': float('nan'),
            'image_height': pd.NA,
        })
        self.assertEqual(row, {
            'copyright': '', 'character': '', 'artist': '', 'general': '',
            'rating': '', 'image_width': None, 'image_height': None,
        })
        # NaN 이 남으면 JSON 이 NaN 리터럴을 내고 JS JSON.parse 가 깨진다
        self.assertNotIn('NaN', json.dumps(row))

    def test_tag_string_column_wins_unless_empty(self) -> None:
        row = normalize_search_row({
            'tag_string_general': 'from_tag_string',
            'general': 'plain',
            'tag_string_character': '',
            'character': 'fallback_char',
            'tag_string_artist': float('nan'),
            'artist': 'fallback_artist',
        })
        self.assertEqual(row['general'], 'from_tag_string')
        self.assertEqual(row['character'], 'fallback_char')
        self.assertEqual(row['artist'], 'fallback_artist')
        self.assertEqual(row['copyright'], '')

    def test_dimensions_are_positive_ints_or_none(self) -> None:
        self.assertEqual(dimension_value(1024), 1024)
        self.assertEqual(dimension_value('768'), 768)
        self.assertEqual(dimension_value(832.0), 832)
        self.assertEqual(dimension_value(np.int64(640)), 640)
        self.assertIsNone(dimension_value(0))
        self.assertIsNone(dimension_value(-5))
        self.assertIsNone(dimension_value(''))
        self.assertIsNone(dimension_value('abc'))
        self.assertIsNone(dimension_value(float('inf')))
        self.assertIsNone(dimension_value(None))
        self.assertIs(type(dimension_value(np.int64(640))), int)

    def test_scalar_helpers(self) -> None:
        self.assertTrue(is_missing(None))
        self.assertTrue(is_missing(math.nan))
        self.assertTrue(is_missing(pd.NaT))
        self.assertFalse(is_missing(''))
        self.assertFalse(is_missing(0))
        self.assertFalse(is_missing([1, 2]))
        self.assertEqual(text_value('g'), 'g')
        self.assertEqual(text_value(3), '3')
        self.assertEqual(text_value(math.nan), '')

    def test_in_place_normalization_reuses_list_and_dicts(self) -> None:
        first = {'general': 'a', 'extra': 'dropped', 'image_width': '512'}
        second = {'general': 'b', 'rating': 's'}
        rows = [first, 'not-a-dict', second, None]

        out = normalize_search_rows_in_place(rows)

        self.assertIs(out, rows)
        self.assertEqual(len(rows), 2)
        self.assertIs(rows[0], first)
        self.assertIs(rows[1], second)
        self.assertEqual(list(first), list(SEARCH_ROW_KEYS))
        self.assertEqual(first['image_width'], 512)
        self.assertEqual(second['rating'], 's')

    def test_normalization_is_idempotent(self) -> None:
        once = normalize_search_row({'general': 'a', 'rating': 'g', 'image_width': 64})
        self.assertEqual(normalize_search_row(once), once)

    def test_frame_import_handles_pandas3_missing_strings(self) -> None:
        frame = pd.DataFrame({
            'general': pd.array(['1girl solo', None], dtype='str'),
            'character': pd.array([None, 'hatsune_miku'], dtype='str'),
            'copyright': ['original', 'vocaloid'],
            'artist': ['', 'x'],
            'rating': pd.array(['g', None], dtype='str'),
            'image_width': pd.array([1024, None], dtype='Int64'),
            'image_height': [768.0, float('nan')],
        })

        rows = search_rows_from_frame(frame)

        self.assertEqual(rows, [
            {'copyright': 'original', 'character': '', 'artist': '',
             'general': '1girl solo', 'rating': 'g',
             'image_width': 1024, 'image_height': 768},
            {'copyright': 'vocaloid', 'character': 'hatsune_miku', 'artist': 'x',
             'general': '', 'rating': '',
             'image_width': None, 'image_height': None},
        ])
        self.assertNotIn('nan', json.dumps(rows))

    def test_frame_import_prefers_tag_string_columns(self) -> None:
        frame = pd.DataFrame({
            'tag_string_general': ['from_tag_string', ''],
            'general': ['plain', 'fallback'],
        })
        rows = search_rows_from_frame(frame)
        self.assertEqual([row['general'] for row in rows], ['from_tag_string', 'fallback'])

    def test_frame_import_applies_the_search_cap_as_a_random_sample(self) -> None:
        frame = pd.DataFrame({'general': [f'tag_{i}' for i in range(50)]})

        capped = search_rows_from_frame(frame, cap=10)
        uncapped = search_rows_from_frame(frame, cap=None)
        small = search_rows_from_frame(frame.head(5), cap=10)

        self.assertEqual(len(capped), 10)
        self.assertEqual(len({row['general'] for row in capped}), 10)
        self.assertTrue({row['general'] for row in capped} <= set(frame['general']))
        self.assertEqual(len(uncapped), 50)
        self.assertEqual(len(small), 5)

    def test_marker_list_is_a_plain_list_for_consumers(self) -> None:
        rows = NormalizedSearchRows([{'general': 'a'}])
        self.assertIsInstance(rows, list)
        self.assertEqual(json.loads(json.dumps(rows)), [{'general': 'a'}])
        self.assertIs(type(rows.copy()), list)


if __name__ == '__main__':
    unittest.main()
