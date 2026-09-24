"""태그 싱글턴·지연 로더 동시성 — 예열 스레드와 GUI 슬롯이 겹쳐도

- 팩토리는 인스턴스를 한 벌만 만들고(락 + 이중 확인),
- _ensure 는 한 번만 적재하며, 적재가 끝나기 전의 반쪽 색인을 누구도 보지 않고,
- 적재가 실패해도 조회마다 재시도하지 않는다.
"""
from __future__ import annotations

import threading
import time
import unittest
from unittest.mock import patch

import pandas as pd

from core.tag_database import TagAsset


def _run_threads(target, count=8):
    barrier = threading.Barrier(count)
    results = [None] * count
    errors = []

    def worker(index):
        try:
            barrier.wait(2)
            results[index] = target()
        except Exception as exc:   # pragma: no cover - 실패 진단용
            errors.append(exc)

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(count)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(5)
    return results, errors


class _SlowCtor:
    """생성에 시간이 걸리는 가짜 — 락이 없으면 여러 스레드가 함께 만든다."""

    created = 0
    lock = threading.Lock()

    def __init__(self, *args, **kwargs):
        with type(self).lock:
            type(self).created += 1
        time.sleep(0.05)


class _GatedKoreanDatabase:
    def __init__(self, *, fail=False):
        self.reads = 0
        self.entered = threading.Event()
        self.release = threading.Event()
        self.fail = fail

    def read_parquet(self, asset):
        assert asset is TagAsset.KOREAN_TAG_CATALOG
        self.reads += 1
        self.entered.set()
        self.release.wait(5)
        if self.fail:
            raise OSError("catalog missing")
        return pd.DataFrame({
            "tag": ["long hair", "solo"],
            "count": [10, 20],
            "category": ["패션 > 헤어스타일", "인물 > 인원수"],
            "desc": ["긴 머리", "혼자"],
            "keywords": ["긴 머리, 장발", "솔로"],
        })


class KoreanTagLookupLoadingTests(unittest.TestCase):
    def test_concurrent_first_queries_wait_for_one_complete_load(self):
        from core.tag_korean_lookup import KoreanTagLookup

        db = _GatedKoreanDatabase()
        lookup = KoreanTagLookup(db)
        seen = {}

        first = threading.Thread(target=lambda: seen.setdefault("a", lookup.label("long hair")))
        first.start()
        self.assertTrue(db.entered.wait(2))
        second = threading.Thread(target=lambda: seen.setdefault("b", lookup.search("장발")))
        second.start()
        second.join(0.2)

        # 적재 중에는 반쪽 색인으로 빈 결과를 돌려주지 않고 기다린다
        self.assertTrue(second.is_alive())
        self.assertFalse(lookup.is_ready())

        db.release.set()
        first.join(2)
        second.join(2)

        self.assertEqual(db.reads, 1)
        self.assertTrue(lookup.is_ready())
        self.assertIsNotNone(seen["a"])
        self.assertEqual([hit["tag"] for hit in seen["b"]], ["long_hair"])

    def test_failed_load_is_not_retried_on_every_keystroke(self):
        from core.tag_korean_lookup import KoreanTagLookup

        db = _GatedKoreanDatabase(fail=True)
        db.release.set()
        lookup = KoreanTagLookup(db)

        self.assertIsNone(lookup.label("long hair"))
        self.assertEqual(lookup.search("장발"), [])
        self.assertEqual(db.reads, 1)
        self.assertTrue(lookup.is_ready())

    def test_factory_builds_one_instance_under_contention(self):
        import core.tag_korean_lookup as module

        class Slow(_SlowCtor):
            created = 0

        module.reset_korean_tag_lookup()
        try:
            with patch.object(module, "KoreanTagLookup", Slow):
                results, errors = _run_threads(module.get_korean_tag_lookup)
        finally:
            module.reset_korean_tag_lookup()

        self.assertEqual(errors, [])
        self.assertEqual(Slow.created, 1)
        self.assertEqual(len({id(r) for r in results}), 1)


class TagIntelligenceLoadingTests(unittest.TestCase):
    def _gated_database(self):
        from tests.test_tag_database_consumers import FakeTagDatabase

        class Gated(FakeTagDatabase):
            def __init__(self):
                super().__init__()
                self.catalog_reads = 0
                self.entered = threading.Event()
                self.release = threading.Event()

            def read_parquet(self, asset, *args, **kwargs):
                if asset is TagAsset.KOREAN_TAG_CATALOG:
                    self.catalog_reads += 1
                    self.entered.set()
                    self.release.wait(5)
                return super().read_parquet(asset, *args, **kwargs)

        return Gated()

    def test_queries_during_load_see_the_complete_lexicon(self):
        from core.tag_intelligence import TagIntelligence

        db = self._gated_database()
        intelligence = TagIntelligence(database=db)
        seen = {}

        first = threading.Thread(
            target=lambda: seen.setdefault("known", intelligence.is_known("profile_series"))
        )
        first.start()
        self.assertTrue(db.entered.wait(2))
        second = threading.Thread(
            target=lambda: seen.setdefault("copyright", intelligence.is_copyright("profile_series"))
        )
        second.start()
        second.join(0.2)
        self.assertTrue(second.is_alive(), "적재 중 조회는 완료를 기다려야 한다")

        db.release.set()
        first.join(2)
        second.join(2)

        self.assertEqual(db.catalog_reads, 1)
        # 예전에는 적재 도중 is_known 이 부분 _copyright_vals 를 세션 내내 캐시할 수 있었다
        self.assertTrue(seen["known"])
        self.assertTrue(seen["copyright"])

    def test_factory_builds_one_instance_under_contention(self):
        import core.tag_intelligence as module

        class Slow(_SlowCtor):
            created = 0

        saved = module._instance
        module._instance = None
        try:
            with patch.object(module, "TagIntelligence", Slow):
                results, errors = _run_threads(module.get_tag_intelligence)
        finally:
            module._instance = saved

        self.assertEqual(errors, [])
        self.assertEqual(Slow.created, 1)
        self.assertEqual(len({id(r) for r in results}), 1)


class CharacterFeatureLoadingTests(unittest.TestCase):
    def test_concurrent_lookups_load_once(self):
        import utils.character_features as module

        class Gated:
            def __init__(self):
                self.profile_reads = 0
                self.entered = threading.Event()
                self.release = threading.Event()

            def read_json(self, asset):
                assert asset is TagAsset.CHARACTER_PROFILES
                self.profile_reads += 1
                self.entered.set()
                self.release.wait(5)
                return [{"tag": "hatsune_miku", "core_tags": ["aqua_hair", "twintails"],
                         "post_count": 5, "copyright": "vocaloid"}]

            def read_parquet(self, asset, columns=None):
                return pd.DataFrame({"character": [], "features": [], "post_count": []})

        db = Gated()
        with patch.object(module, "get_tag_database", return_value=db):
            lookup = module.CharacterFeatureLookup()

        seen = {}
        first = threading.Thread(target=lambda: seen.setdefault("a", lookup.all_keys()))
        first.start()
        self.assertTrue(db.entered.wait(2))
        second = threading.Thread(target=lambda: seen.setdefault("b", lookup.all_keys()))
        second.start()
        second.join(0.2)
        self.assertTrue(second.is_alive(), "적재 중 조회는 완료를 기다려야 한다")

        db.release.set()
        first.join(2)
        second.join(2)

        self.assertEqual(db.profile_reads, 1)
        self.assertEqual(seen["a"], ["hatsune miku"])
        self.assertEqual(seen["b"], ["hatsune miku"])

    def test_factory_builds_one_instance_under_contention(self):
        import utils.character_features as module

        class Slow(_SlowCtor):
            created = 0

        saved = module._instance
        module._instance = None
        try:
            with patch.object(module, "CharacterFeatureLookup", Slow):
                results, errors = _run_threads(module.get_character_features)
        finally:
            module._instance = saved

        self.assertEqual(errors, [])
        self.assertEqual(Slow.created, 1)
        self.assertEqual(len({id(r) for r in results}), 1)


if __name__ == "__main__":
    unittest.main()
