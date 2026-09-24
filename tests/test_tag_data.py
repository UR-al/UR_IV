"""utils.tag_data / tag_completer / tag_classifier — 태그 어휘 한 벌 공유와 스레드 안전 싱글톤."""
from __future__ import annotations

import json
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

import core.tag_classifier as tag_classifier_module
import utils.tag_completer as tag_completer_module
import utils.tag_data as tag_data_module
from utils.tag_completer import TagCompleter
from utils.tag_data import TagData, completion_key, completion_tag

ROWS = [
    # tag, category, count
    ("long_hair", "general", 500),
    ("Long_Hair_Alt", "general", 5),      # 대문자 — 키는 casefold
    ("x__x", "general", 7),               # 연속 밑줄 — 예전 표시형(공백 두 칸) 정규화와 같아야
    ("hatsune_miku", "character", 300),
    ("genshin_impact", "copyright", 200),
    ("wlop", "artist", 100),
    ("highres", "meta", 400),
    ("mystery_tag", "unknown_type", 3),   # 목록엔 없고 인기도 맵에만
    ("broken_count", "general", None),
]


def _write_dictionary(root: Path) -> Path:
    path = root / "tags_dictionary.parquet"
    pd.DataFrame(ROWS, columns=["tag", "category", "count"]).to_parquet(path, index=False)
    return path


def _write_tags_db(root: Path) -> Path:
    """별칭이 비어 있는 최소 tags_db (완성기가 실제 사전의 별칭을 끌어오지 않게)."""
    db = root / "tags_db"
    (db / "taxonomy").mkdir(parents=True)
    pd.DataFrame({"alias": ["zz_alias"], "canonical": ["wlop"]}).to_parquet(
        db / "taxonomy" / "aliases.parquet", index=False
    )
    (db / "manifest.json").write_text(
        json.dumps({
            "version": 1,
            "assets": {
                "tag_aliases": {
                    "path": "taxonomy/aliases.parquet",
                    "format": "parquet",
                    "description": "alias fixture",
                    "columns": ["alias", "canonical"],
                }
            },
        }),
        encoding="utf-8",
    )
    return db


def _legacy_completion_counts(frame: pd.DataFrame) -> dict:
    """예전 경로: TagData(lower+replace 키) → TagCompleter 가 _tag_key 로 다시 만든 dict."""
    first: dict = {}
    for tag, cnt in zip(frame["tag"].astype(str), frame["count"]):
        if tag:
            try:
                first[tag.lower().replace(" ", "_")] = int(cnt)
            except (TypeError, ValueError):
                pass
    return {completion_key(k): v for k, v in first.items()}


class TagDataSharingTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.path = _write_dictionary(Path(self._tmp.name))
        self.td = TagData(str(self.path))

    def tearDown(self):
        self._tmp.cleanup()

    def test_lists_use_completion_form_and_reuse_parquet_strings(self):
        self.assertTrue(self.td.is_loaded)
        # count 내림차순 — count 가 없는 태그도 목록엔 남는다(인기도 맵에서만 빠짐)
        self.assertEqual(
            self.td.general_tags, ["long_hair", "x_x", "Long_Hair_Alt", "broken_count"]
        )
        self.assertEqual(self.td.character_tags, ["hatsune_miku"])
        self.assertNotIn("mystery_tag", self.td.general_tags)
        # 예전: 공백형 표시 문자열을 완성기가 정규화 → 같은 결과
        for raw in ("long_hair", "x__x", "Long_Hair_Alt"):
            self.assertIn(completion_tag(raw.replace("_", " ")), self.td.general_tags)

    def test_name_sets_are_space_form_lower_and_classifier_ready(self):
        self.assertEqual(self.td.character_set, {"hatsune miku"})
        self.assertEqual(self.td.copyright_set, {"genshin impact"})
        self.assertEqual(self.td.artist_set, {"wlop"})
        self.assertEqual(self.td.meta_set, {"highres"})
        self.assertFalse(hasattr(self.td, "general_set"), "소비자 없는 general_set 은 들지 않는다")

    def test_counts_match_the_completer_rebuild_it_replaces(self):
        frame = pd.DataFrame(ROWS, columns=["tag", "category", "count"]).sort_values(
            "count", ascending=False
        )
        self.assertEqual(self.td.tag_counts, _legacy_completion_counts(frame))
        self.assertEqual(self.td.tag_counts["long_hair_alt"], 5)
        self.assertEqual(self.td.tag_counts["mystery_tag"], 3)
        self.assertNotIn("broken_count", self.td.tag_counts)

    def test_completer_references_tag_data_instead_of_copying(self):
        db = _write_tags_db(Path(self._tmp.name))
        with patch("utils.tag_data.get_tag_data", return_value=self.td):
            completer = TagCompleter(str(db))
        self.assertIs(completer._counts, self.td.tag_counts)
        keys = completer._cat_lower_keys["general"]
        tags = completer._cat_tags["general"]
        self.assertEqual(keys, ["broken_count", "long_hair", "long_hair_alt", "x_x"])
        self.assertIs(keys[1], tags[1], "소문자 태그는 키와 표시형이 한 객체")
        # 접두 일치는 인기도순, 포함 일치(general)는 그 뒤
        self.assertEqual(completer.get_suggestions("long"), ["long_hair", "Long_Hair_Alt"])
        self.assertEqual(completer.get_suggestions("hair"), ["long_hair", "Long_Hair_Alt"])
        self.assertEqual(completer.get_suggestions("miku"), [])
        self.assertEqual(completer.get_suggestions("hatsu"), ["hatsune_miku"])
        self.assertEqual(completer.get_suggestions("zz_al"), ["wlop"], "별칭 일치")
        self.assertTrue(completer.is_valid_tag("Long Hair"))
        self.assertTrue(completer.is_valid_tag("hatsune miku"))
        self.assertFalse(completer.is_valid_tag("mystery_tag"))
        self.assertFalse(completer.is_valid_tag(""))
        self.assertIn("wlop", completer.tags_set)
        self.assertEqual(completer.count(), 8)
        # 호출자 없던 get_all_tags(전 목록 복사)는 지웠다(P13c) — 개수는 count() 로 본다
        self.assertFalse(hasattr(completer, "get_all_tags"))

    def test_classifier_shares_name_sets_but_copies_meta(self):
        with patch("utils.tag_data.get_tag_data", return_value=self.td):
            classifier = tag_classifier_module.TagClassifier()
        self.assertIs(classifier.characters, self.td.character_set)
        self.assertIs(classifier.copyrights, self.td.copyright_set)
        self.assertIs(classifier.artists, self.td.artist_set)
        self.assertIsNot(classifier.meta_tags, self.td.meta_set)
        before = set(self.td.meta_set)
        classifier.meta_tags.add("manifest_only_meta")
        self.assertEqual(self.td.meta_set, before, "분류기 meta 보강이 공유 TagData 를 오염시키지 않는다")
        self.assertEqual(classifier.classify_tag("hatsune_miku"), "character")
        self.assertEqual(classifier.classify_tag("Genshin Impact"), "copyright")


class _SlowFactory:
    """생성에 시간이 걸리는 가짜 — 동시에 불려도 한 번만 만들어졌는지 센다."""

    def __init__(self):
        self.built = 0
        self.lock = threading.Lock()

    def __call__(self, *args, **kwargs):
        with self.lock:
            self.built += 1
        time.sleep(0.05)
        return object()


class SingletonThreadSafetyTests(unittest.TestCase):
    def _hammer(self, getter, n=8):
        results = []
        barrier = threading.Barrier(n)

        def run():
            barrier.wait()
            results.append(getter())

        threads = [threading.Thread(target=run) for _ in range(n)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        return results

    def test_tag_data_singleton_builds_once_under_concurrency(self):
        factory = _SlowFactory()
        with patch.object(tag_data_module, "TagData", factory), \
                patch.object(tag_data_module, "_tag_data_instance", None):
            results = self._hammer(tag_data_module.get_tag_data)
        self.assertEqual(factory.built, 1)
        self.assertEqual(len({id(r) for r in results}), 1)

    def test_completer_singleton_builds_once_and_try_get_does_not_block(self):
        factory = _SlowFactory()
        with patch.object(tag_completer_module, "TagCompleter", factory), \
                patch.object(tag_completer_module, "_completer_instance", None):
            self.assertFalse(tag_completer_module.is_tag_completer_ready())
            # 다른 스레드(워밍업)가 적재 중이면 GUI 쪽 try_get 은 기다리지 않고 None
            with tag_completer_module._completer_lock:
                self.assertIsNone(tag_completer_module.try_get_tag_completer())
            results = self._hammer(tag_completer_module.get_tag_completer)
            self.assertIs(tag_completer_module.try_get_tag_completer(), results[0])
            self.assertTrue(tag_completer_module.is_tag_completer_ready())
        self.assertEqual(factory.built, 1)
        self.assertEqual(len({id(r) for r in results}), 1)

    def test_classifier_singleton_is_shared_and_resettable(self):
        factory = _SlowFactory()
        with patch.object(tag_classifier_module, "TagClassifier", factory), \
                patch.object(tag_classifier_module, "_shared_classifier", None):
            results = self._hammer(tag_classifier_module.get_tag_classifier)
            self.assertEqual(factory.built, 1)
            self.assertEqual(len({id(r) for r in results}), 1)
            tag_classifier_module.reset_tag_classifier()
            self.assertIsNot(tag_classifier_module.get_tag_classifier(), results[0])
        self.assertEqual(factory.built, 2)


if __name__ == "__main__":
    unittest.main()
