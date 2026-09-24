from __future__ import annotations

import unittest
from pathlib import Path

import pandas as pd

from core.tag_classifier import TagClassifier
from core.tag_database import TagAsset
from core.tag_intelligence import TagIntelligence


class FakeTagDatabase:
    """Small in-memory adapter exercising the same interface as TagDatabase."""

    def __init__(self):
        self.calls: list[tuple[str, TagAsset]] = []
        self.lines = {
            TagAsset.CLOTHING_TAGS_CURATED: ["formal_dress"],
            TagAsset.CLOTHING_TAGS_EXTENDED: ["festival_jacket"],
            TagAsset.APPEARANCE_TAGS_CURATED: ["freckles"],
            TagAsset.APPEARANCE_TAGS_EXTENDED: ["star_pupils"],
            TagAsset.COLOR_TERMS_CURATED: ["blue"],
            TagAsset.COLOR_TERMS_EXTENDED: ["chartreuse"],
        }
        self.json = {
            TagAsset.RATING_COUNTS: {
                "_meta": {"total_posts": 10},
                "ornate_gown": [5, 1, 3, 1],
            },
            TagAsset.CHARACTER_PROFILES: [
                {"tag": "profile_hero", "copyright": "profile_series"},
            ],
            TagAsset.CURATED_CHARACTER_SERIES: {
                "curated_series": {
                    "girl": [
                        {"name": "curated_hero", "aliases": ["curated_alias"]},
                        {"name": "profile_hero", "aliases": []},
                    ]
                }
            },
            TagAsset.CLOTHING_REGIONS: {
                "regions": {"FULL_BODY": ["ornate_gown"]},
            },
            TagAsset.EXPRESSION_TAGS: {"tags": ["smile"]},
            TagAsset.LOCATION_TAGS: {"tags": ["city_street"]},
            TagAsset.POSE_ACTION_TAGS: {
                "categories": {"body": ["standing"]},
                "uncategorized": ["floating"],
            },
            TagAsset.OBJECT_TAGS: {"tags": ["umbrella"]},
            TagAsset.META_TAGS: {"tags": ["highres"]},
        }
        self.frames = {
            TagAsset.KOREAN_TAG_CATALOG: pd.DataFrame(
                {
                    "tag": ["ornate_gown", "star_pupils"],
                    "category": ["패션 > 의류", "신체 > 눈"],
                    "count": [300, 20],
                }
            ),
            TagAsset.EXTENDED_CHARACTER_SERIES: pd.DataFrame(
                {
                    "character": ["profile_hero", "curated_hero", "extended_hero"],
                    "copyright": ["wrong_profile", "wrong_curated", "extended_series"],
                    "post_count": [999, 999, 100],
                    "first_seen": ["2020", "2020", "2020"],
                }
            ),
        }
        self.groups = {
            "tag_group:attire": {"formal_dress", "evening_gown"},
            "tag_group:face_tags": {"bright_smile"},
            "tag_group:censorship": {"pixel_censoring"},
            "tag_group:text": {"speech_bubble"},
            "tag_group:fine_art_parody": {"after_artist"},
            "tag_group:piercings": {"ear_piercing"},
            "tag_group:pixiv_projects": {"pixiv_fantasia"},
            "tag_group:subjective": {"adorable"},
            "tag_group:verbs_and_gerunds": {"aiming"},
            "tag_group:year_tags": {"2024"},
            # An unclassified direct group must not block a useful implication.
            "tag_group:unknown_fixture": {"blue_evening_gown", "uncategorized_child"},
        }
        self.implications = {
            "blue_evening_gown": {"evening_gown"},
            "ornate_gown": {"gown"},
            "gown": {"clothes"},
            "cycle_a": {"cycle_b"},
            "cycle_b": {"cycle_a"},
        }

    def path(self, asset):
        self.calls.append(("path", asset))
        return Path("fixture") / f"{asset.value}.parquet"

    def read_lines(self, asset):
        self.calls.append(("lines", asset))
        return list(self.lines[asset])

    def read_json(self, asset):
        self.calls.append(("json", asset))
        return self.json[asset]

    def read_parquet(self, asset, columns=None):
        self.calls.append(("parquet", asset))
        frame = self.frames[asset]
        return frame if columns is None else frame[list(columns)]

    def load_tag_groups(self):
        self.calls.append(("groups", TagAsset.TAG_GROUPS))
        return self.groups

    def load_active_implications(self):
        self.calls.append(("implications", TagAsset.TAG_IMPLICATIONS))
        return self.implications

    def all_group_tags(self):
        self.calls.append(("all_group_tags", TagAsset.TAG_GROUPS))
        return set().union(*self.groups.values())


class TagClassifierDatabaseTests(unittest.TestCase):
    def test_uses_consolidated_groups_variants_and_nearest_implication(self):
        database = FakeTagDatabase()

        classifier = TagClassifier(database=database)

        self.assertEqual(classifier.classify_tag("formal dress"), "clothing")
        self.assertEqual(classifier.classify_tag("evening_gown"), "clothing")
        self.assertEqual(classifier.classify_tag("blue evening gown"), "clothing")
        self.assertEqual(classifier.classify_tag("bright_smile"), "expression")
        self.assertEqual(classifier.classify_tag("ear_piercing"), "clothing")
        self.assertEqual(classifier.classify_tag("pixiv_fantasia"), "art_style")
        self.assertEqual(classifier.classify_tag("adorable"), "character_trait")
        self.assertEqual(classifier.classify_tag("aiming"), "pose")
        self.assertEqual(classifier.classify_tag("2024"), "effect")
        self.assertEqual(classifier.classify_tag("profile hero"), "character")
        self.assertEqual(classifier.classify_tag("profile_series"), "copyright")
        self.assertEqual(classifier.classify_tag("extended hero"), "character")
        self.assertEqual(classifier.classify_tag("extended_series"), "copyright")
        self.assertTrue(classifier.is_meta_tag("after artist"))
        self.assertTrue(classifier.is_meta_tag("highres"))
        self.assertTrue(classifier.is_censorship_tag("pixel censoring"))
        self.assertTrue(classifier.is_text_tag("speech_bubble"))

        parquet_assets = [asset for method, asset in database.calls if method == "parquet"]
        self.assertNotIn(TagAsset.AUTOCOMPLETE_CATALOG, parquet_assets)
        self.assertNotIn(TagAsset.KOREAN_TAG_CATALOG, parquet_assets)

    def test_implication_cycle_with_no_categorized_parent_is_general(self):
        classifier = TagClassifier(database=FakeTagDatabase())

        self.assertEqual(classifier.classify_tag("cycle_a"), "general")


class TagIntelligenceDatabaseTests(unittest.TestCase):
    def test_is_lazy_and_preserves_character_series_priority(self):
        database = FakeTagDatabase()
        intelligence = TagIntelligence(database=database)
        self.assertEqual(database.calls, [])

        self.assertEqual(intelligence.copyright_of("profile hero"), "profile series")
        self.assertEqual(intelligence.copyright_of("curated_alias"), "curated series")
        self.assertEqual(intelligence.copyright_of("extended_hero"), "extended series")
        self.assertTrue(intelligence.is_clothing("formal dress"))
        self.assertTrue(intelligence.is_clothing("festival_jacket"))
        self.assertTrue(intelligence.is_appearance("freckles"))
        self.assertTrue(intelligence.is_appearance("star pupils"))
        self.assertTrue(intelligence.is_color("chartreuse"))

        for tag in (
            "smile", "city_street", "standing", "floating", "umbrella",
            "highres", "chartreuse", "extended_hero", "extended_series",
            "evening_gown", "score_9", "score_8_up", "score_7_up",
        ):
            self.assertTrue(intelligence.is_known(tag), tag)
        self.assertEqual(intelligence.tag_group("score_9"), "quality")

    def test_dead_prompt_helpers_stay_removed(self):
        # 호출자 없던 슬롯(pairColors·getClothingRegions·refineToSpecificTags)과 함께 지운 전용
        # 로직 — 되살리지 말 것(감사 #135).
        self.assertFalse(hasattr(TagIntelligence, "pair_colors"))
        self.assertFalse(hasattr(TagIntelligence, "group_by_region"))
        self.assertFalse(hasattr(TagIntelligence, "remove_redundant_subtags"))

    def test_lookups_never_load_the_implication_table(self):
        """활성 implication 표(실데이터 약 4.6만 행 parquet)는 TagIntelligence 가 올리지 않는다.

        예전엔 공용 적재(_ensure) 8단계가 호출자 없던 remove_redundant_subtags 하나를 위해
        copyright_of·tag_group 같은 조회마다 올렸다. 상위 태그 추론은 TagClassifier 몫이다.
        """
        database = FakeTagDatabase()
        intelligence = TagIntelligence(database=database)

        self.assertEqual(intelligence.copyright_of("profile hero"), "profile series")
        self.assertEqual(intelligence.tag_group("score_9"), "quality")
        self.assertTrue(intelligence.is_known("smile"))
        intelligence.split_by_categories(["smile", "gown"], ["expression", "clothing"])

        self.assertTrue(database.calls)  # 공용 사전은 실제로 적재됐다(공허한 통과 방지)
        self.assertNotIn("implications", {method for method, _asset in database.calls})
        self.assertFalse(hasattr(intelligence, "_implications"))

        # 같은 표를 쓰는 TagClassifier 는 그대로 읽는다(TagDatabase.load_active_implications 는 살아 있다)
        classifier_db = FakeTagDatabase()
        TagClassifier(database=classifier_db).classify_tag("evening_gown")
        self.assertIn("implications", {method for method, _asset in classifier_db.calls})


if __name__ == "__main__":
    unittest.main()
