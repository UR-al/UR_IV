from __future__ import annotations

import json
import unittest
from unittest.mock import patch

import pandas as pd

from ui.vue_bridge import VueBridge


class _DatabaseFixture:
    def __init__(self):
        self.catalog_reads = 0

    def read_lines(self, _asset):
        return []

    def read_parquet(self, _asset, columns=None):
        self.catalog_reads += 1
        frame = pd.DataFrame({"tag": ["1girl", "abandoned", "long hair", "short hair"]})
        return frame if columns is None else frame[list(columns)]

    def all_group_tags(self):
        return set()


class _ClassifierFixture:
    tag_to_category = {"hair ornament": []}
    characters = {"hatsune miku"}
    copyrights = set()
    artists = set()


class _CountingSet(set):
    """사전 전체를 몇 번 훑었는지 센다."""

    scans = 0

    def __iter__(self):
        type(self).scans += 1
        return super().__iter__()


class VueBridgeTagDatabaseTests(unittest.TestCase):
    def _patched(self, database=None, classifier=None):
        return (
            patch(
                "core.tag_database.get_tag_database",
                return_value=database or _DatabaseFixture(),
            ),
            # 브리지는 메인 창과 같은 공유 분류기를 쓴다 — 전역 싱글톤을 오염시키지 않게
            # 접근자 자체를 바꿔 끼운다.
            patch(
                "core.tag_classifier.get_tag_classifier",
                return_value=classifier or _ClassifierFixture(),
            ),
        )

    def test_exclude_preview_keeps_general_korean_catalog_tags(self):
        db_patch, cls_patch = self._patched()
        with db_patch, cls_patch:
            bridge = VueBridge()
            matches = json.loads(bridge.getExcludeMatches("*1girl"))

        self.assertEqual(matches, ["1girl"])

    def test_vocabulary_merges_catalog_groups_and_shared_classifier_names(self):
        db_patch, cls_patch = self._patched()
        with db_patch, cls_patch:
            bridge = VueBridge()
            self.assertEqual(
                json.loads(bridge.getExcludeMatches("hair")),
                ["hair_ornament", "long_hair", "short_hair"],
            )
            self.assertEqual(json.loads(bridge.getExcludeMatches("*Hatsune Miku")), ["hatsune_miku"])
            self.assertEqual(json.loads(bridge.getExcludeMatches("~hair")), [])

    def test_results_are_memoised_per_rule_and_vocabulary_is_built_once(self):
        database = _DatabaseFixture()
        db_patch, cls_patch = self._patched(database=database)
        with db_patch, cls_patch:
            bridge = VueBridge()
            first = bridge.getExcludeMatches("_hair")
            bridge._all_tags_set = _CountingSet(bridge._all_tags_set)
            _CountingSet.scans = 0
            again = bridge.getExcludeMatches("_hair")
            same_rule_other_spelling = bridge.getExcludeMatches("  _HAIR ")
            exact = bridge.getExcludeMatches("*long hair")

        self.assertEqual(first, again)
        self.assertEqual(first, same_rule_other_spelling)
        self.assertEqual(json.loads(exact), ["long_hair"])
        self.assertEqual(_CountingSet.scans, 0, "같은 규칙 재클릭·완전 일치는 사전을 훑지 않는다")
        self.assertEqual(database.catalog_reads, 1, "태그 사전은 한 번만 만든다")

    def test_failed_vocabulary_build_is_not_cached_half_done(self):
        with patch("core.tag_database.get_tag_database", side_effect=RuntimeError("db down")):
            bridge = VueBridge()
            result = json.loads(bridge.getExcludeMatches("hair"))
        self.assertIn("error", result)
        self.assertFalse(hasattr(bridge, "_all_tags_set"), "실패한 사전을 캐시로 남기지 않는다")

        db_patch, cls_patch = self._patched()
        with db_patch, cls_patch:
            self.assertEqual(json.loads(bridge.getExcludeMatches("*1girl")), ["1girl"])

    def test_bridge_uses_the_shared_classifier(self):
        shared = _ClassifierFixture()
        _db_patch, cls_patch = self._patched(classifier=shared)
        with cls_patch:
            bridge = VueBridge()
            self.assertIs(bridge._get_tag_classifier(), shared)


if __name__ == "__main__":
    unittest.main()
