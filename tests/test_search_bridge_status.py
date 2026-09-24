from __future__ import annotations

import json
import unittest
from unittest.mock import patch

from PyQt6.QtCore import QObject, pyqtSignal

from ui.vue_bridge import VueBridge


class _FakeSearchWorker(QObject):
    # 실제 PandasSearchWorker 와 같은 object 시그니처 — list(QVariantList)면 emit 마다
    # 행 전체가 복사된다.
    results_ready = pyqtSignal(object, int)
    status_update = pyqtSignal(str)

    def __init__(self, *_args, **_kwargs) -> None:
        super().__init__()
        self._running = False

    def start(self) -> None:
        self._running = True

    def isRunning(self) -> bool:
        return self._running

    def stop(self) -> None:
        self._running = False

    def wait(self, _milliseconds: int) -> bool:
        return True


class SearchBridgeStatusTests(unittest.TestCase):
    def test_worker_error_status_reaches_the_search_view_channel(self) -> None:
        bridge = VueBridge()
        statuses = []
        bridge.searchStatus.connect(statuses.append)

        with patch(
            "workers.search_worker.PandasSearchWorker",
            _FakeSearchWorker,
        ):
            bridge.searchDanbooru(json.dumps({"ratings": ["g"]}))
            bridge._search_worker.status_update.emit(
                "❌ 활성 검색 shard 검증 실패"
            )

        self.assertIn("❌ 활성 검색 shard 검증 실패", statuses)

    def test_replaced_worker_cannot_publish_a_late_error_status(self) -> None:
        bridge = VueBridge()
        statuses = []
        bridge.searchStatus.connect(statuses.append)

        with patch(
            "workers.search_worker.PandasSearchWorker",
            _FakeSearchWorker,
        ):
            bridge.searchDanbooru(json.dumps({"ratings": ["g"]}))
            first = bridge._search_worker
            bridge.searchDanbooru(json.dumps({"ratings": ["s"]}))
            first.status_update.emit("❌ 오래된 worker 오류")

        self.assertNotIn("❌ 오래된 worker 오류", statuses)

    def test_rejected_result_cannot_be_overwritten_by_late_success_status(self) -> None:
        bridge = VueBridge()
        statuses = []
        bridge.searchStatus.connect(statuses.append)

        with patch(
            "workers.search_worker.PandasSearchWorker",
            _FakeSearchWorker,
        ):
            bridge.searchDanbooru(json.dumps({"ratings": ["g"]}))
            worker = bridge._search_worker
            worker.results_ready.emit([{"general": "unverified"}], 1)
            worker.status_update.emit("✅ 검색 완료: 1건")

        self.assertTrue(statuses[-1].startswith("❌"), statuses)
        self.assertNotIn("✅ 검색 완료: 1건", statuses)

    def test_manifest_change_during_result_processing_prevents_publication_and_runtime_mutation(
        self,
    ) -> None:
        verified_identity = {"label": "2026_07", "fingerprint": "a" * 64}
        changed_identity = {"label": "2026_07", "fingerprint": "b" * 64}
        previous_identity = {"label": "previous", "fingerprint": "c" * 64}

        class Parent(QObject):
            def __init__(self) -> None:
                super().__init__()
                self.filtered_results = [{"general": "existing"}]
                self.shuffled_prompt_deck = [{"general": "existing-deck"}]
                self._search_dataset_identity = previous_identity
                self._search_snapshot_id = "d" * 32
                self.persist_calls = []
                self.deck_save_calls = 0

            def _persist_search_results(self, *args, **kwargs) -> None:
                self.persist_calls.append((args, kwargs))

            def _save_deck_state(self) -> None:
                self.deck_save_calls += 1

        class Store:
            dataset_info_calls = 0
            changed_during_serialization = False

            def dataset_info(self):
                type(self).dataset_info_calls += 1
                return (
                    changed_identity
                    if type(self).changed_during_serialization
                    else verified_identity
                )

        real_dumps = json.dumps

        def dumps_and_change(value, *args, **kwargs):
            encoded = real_dumps(value, *args, **kwargs)
            if isinstance(value, list):
                Store.changed_during_serialization = True
            return encoded

        parent = Parent()
        bridge = VueBridge(parent)
        statuses = []
        published = []
        lineages = []
        bridge.searchStatus.connect(statuses.append)
        bridge.searchResultsReady.connect(lambda payload: published.append(json.loads(payload)))
        bridge.searchResultLineage.connect(lambda payload: lineages.append(json.loads(payload)))
        query_json = real_dumps({"ratings": ["g"]})

        with (
            patch("workers.search_worker.PandasSearchWorker", _FakeSearchWorker),
            patch("core.search_result_store.SearchResultStore", Store),
            patch("ui.vue_bridge.json.dumps", side_effect=dumps_and_change),
        ):
            bridge.searchDanbooru(query_json)
            worker = bridge._search_worker
            worker.dataset_identity = verified_identity
            worker.results_ready.emit([{"general": "new-result"}], 1)
            worker.status_update.emit("✅ 검색 완료: 1건")

        self.assertEqual(Store.dataset_info_calls, 2)
        self.assertEqual(published, [])
        self.assertEqual(lineages, [])
        self.assertTrue(statuses[-1].startswith("❌"), statuses)
        self.assertNotIn("✅ 검색 완료: 1건", statuses)
        self.assertEqual(parent.filtered_results, [{"general": "existing"}])
        self.assertEqual(parent.shuffled_prompt_deck, [{"general": "existing-deck"}])
        self.assertEqual(parent._search_dataset_identity, previous_identity)
        self.assertEqual(parent._search_snapshot_id, "d" * 32)
        self.assertEqual(parent.persist_calls, [])
        self.assertEqual(parent.deck_save_calls, 0)

    def test_accepted_result_publishes_lineage_before_rows(self) -> None:
        identity = {"label": "2026_07", "fingerprint": "a" * 64}

        class Store:
            @staticmethod
            def dataset_info():
                return identity

            @staticmethod
            def save(*_args, **_kwargs):
                return None

        bridge = VueBridge()
        events = []
        bridge.searchResultLineage.connect(
            lambda payload: events.append(("lineage", json.loads(payload)))
        )
        bridge.searchResultsReady.connect(
            lambda payload: events.append(("results", json.loads(payload)))
        )

        with (
            patch("workers.search_worker.PandasSearchWorker", _FakeSearchWorker),
            patch("core.search_result_store.SearchResultStore", Store),
        ):
            bridge.searchDanbooru(json.dumps({"ratings": ["g"]}))
            worker = bridge._search_worker
            worker.dataset_identity = identity
            worker.results_ready.emit([{"general": "accepted"}], 1)

        self.assertEqual(events[0][0], "lineage")
        self.assertEqual(events[0][1]["label"], "2026_07")
        self.assertEqual(events[0][1]["fingerprint"], "a" * 64)
        self.assertRegex(events[0][1]["snapshot_id"], r"^[0-9a-f]{32}$")
        self.assertEqual(events[1], ("results", [{
            "copyright": "",
            "character": "",
            "artist": "",
            "general": "accepted",
            "rating": "",
            "image_width": None,
            "image_height": None,
        }]))

    def test_valid_empty_cache_restores_runtime_snapshot_lineage(self) -> None:
        identity = {"label": "2026_07", "fingerprint": "f" * 64}

        class Parent(QObject):
            def __init__(self) -> None:
                super().__init__()
                self.filtered_results = [{"general": "stale"}]
                self.shuffled_prompt_deck = [{"general": "stale"}]

            @staticmethod
            def _restore_deck_state():
                return False

        class Store:
            last_error = None
            last_snapshot_id = "a" * 32
            last_dataset_identity = identity

            @staticmethod
            def load_active():
                return []

        parent = Parent()
        bridge = VueBridge(parent)
        lineages = []
        bridge.searchResultLineage.connect(
            lambda payload: lineages.append(json.loads(payload))
        )

        with patch("core.search_result_store.SearchResultStore", Store):
            self.assertEqual(json.loads(bridge.loadLastSearchResults()), [])

        self.assertEqual(parent._search_snapshot_id, "a" * 32)
        self.assertEqual(parent._search_dataset_identity, identity)
        self.assertEqual(parent.filtered_results, [])
        self.assertEqual(parent.shuffled_prompt_deck, [])
        self.assertEqual(lineages, [{
            **identity,
            "snapshot_id": "a" * 32,
        }])


_IDENTITY = {"label": "2026_07", "fingerprint": "a" * 64}
_SNAPSHOT = "b" * 32


class _PublishingParent(QObject):
    def __init__(self) -> None:
        super().__init__()
        self.filtered_results = []
        self.shuffled_prompt_deck = []
        self.persisted = []

    def _persist_search_results(self, *args, **kwargs) -> None:
        self.persisted.append((args, kwargs))

    def _save_deck_state(self) -> None:
        return None


class _IdentityStore:
    def dataset_info(self):
        return dict(_IDENTITY)

    def save(self, *_args, **_kwargs):
        return None


class SearchResultPublishingTests(unittest.TestCase):
    def test_worker_normalized_rows_skip_the_gui_thread_loop_and_become_the_base(self) -> None:
        from core.search_rows import NormalizedSearchRows

        parent = _PublishingParent()
        bridge = VueBridge(parent)
        published = []
        bridge.searchResultsReady.connect(lambda payload: published.append(json.loads(payload)))
        row = {"copyright": "", "character": "", "artist": "", "general": "ready",
               "rating": "g", "image_width": 512, "image_height": None}
        rows = NormalizedSearchRows([row])

        with (
            patch("workers.search_worker.PandasSearchWorker", _FakeSearchWorker),
            patch("core.search_result_store.SearchResultStore", _IdentityStore),
            patch(
                "core.search_rows.normalize_search_rows_in_place",
                side_effect=AssertionError("worker rows must not be normalised again"),
            ),
        ):
            bridge.searchDanbooru(json.dumps({"ratings": ["g"]}))
            worker = bridge._search_worker
            worker.dataset_identity = dict(_IDENTITY)
            worker.results_ready.emit(rows, 1)

        self.assertEqual(published, [[row]])
        # 같은 dict 객체가 그대로 런타임 풀·필터 base 가 된다(QVariant 왕복 복사 없음)
        self.assertIs(parent.filtered_results[0], row)
        self.assertIs(type(parent.filtered_results), list)
        self.assertIs(parent._search_base_results, parent.filtered_results)
        self.assertEqual(parent._search_base_snapshot_id, parent._search_snapshot_id)
        self.assertEqual(parent._search_dataset_identity, _IDENTITY)

    def test_object_signal_delivers_the_same_list_object(self) -> None:
        received = []
        worker = _FakeSearchWorker()
        worker.results_ready.connect(lambda rows, total: received.append(rows))
        rows = [{"general": "x"}]
        worker.results_ready.emit(rows, 1)
        self.assertIs(received[0], rows)

    def test_non_list_results_are_reported_as_errors_not_empty_success(self) -> None:
        bridge = VueBridge()
        statuses = []
        published = []
        bridge.searchStatus.connect(statuses.append)
        bridge.searchResultsReady.connect(lambda payload: published.append(json.loads(payload)))

        with (
            patch("workers.search_worker.PandasSearchWorker", _FakeSearchWorker),
            patch("core.search_result_store.SearchResultStore", _IdentityStore),
        ):
            bridge.searchDanbooru(json.dumps({"ratings": ["g"]}))
            worker = bridge._search_worker
            worker.dataset_identity = dict(_IDENTITY)
            worker.results_ready.emit({"general": "not a list"}, 1)

        self.assertTrue(statuses[-1].startswith("❌"), statuses)
        self.assertIn("error", published[-1])


class SearchCacheRestoreSlotTests(unittest.TestCase):
    def _parent_with_snapshot(self, active, base=None):
        from core.search_session import publish_snapshot

        parent = _PublishingParent()
        publish_snapshot(
            parent,
            active=active,
            base=base,
            identity=dict(_IDENTITY),
            snapshot_id=_SNAPSHOT,
        )
        return parent

    def test_load_last_serialises_memory_without_reparsing_the_cache(self) -> None:
        active = [{"general": "in-memory"}]
        parent = self._parent_with_snapshot(active)
        parent.shuffled_prompt_deck = ["progress"]
        bridge = VueBridge(parent)
        lineages = []
        bridge.searchResultLineage.connect(lambda p: lineages.append(json.loads(p)))

        class Store(_IdentityStore):
            def load_active(self):
                raise AssertionError("active cache must not be parsed again")

        with patch("core.search_result_store.SearchResultStore", Store):
            self.assertEqual(json.loads(bridge.loadLastSearchResults()), active)

        self.assertEqual(lineages, [{**_IDENTITY, "snapshot_id": _SNAPSHOT}])
        self.assertEqual(parent.shuffled_prompt_deck, ["progress"])   # 덱 재구성 없음
        self.assertIs(parent.filtered_results, active)

    def test_load_last_falls_back_to_disk_when_the_dataset_changed(self) -> None:
        parent = self._parent_with_snapshot([{"general": "old dataset"}])

        class Store:
            last_error = "search result dataset fingerprint does not match"
            last_snapshot_id = None
            last_dataset_identity = None

            @staticmethod
            def dataset_info():
                return {"label": "2026_07", "fingerprint": "c" * 64}

            @staticmethod
            def load_active():
                return []

        bridge = VueBridge(parent)
        with patch("core.search_result_store.SearchResultStore", Store):
            self.assertEqual(json.loads(bridge.loadLastSearchResults()), [])

    def test_load_full_returns_the_memory_base(self) -> None:
        base = [{"general": "a"}, {"general": "b"}]
        parent = self._parent_with_snapshot([base[0]], base=base)
        bridge = VueBridge(parent)

        class Store(_IdentityStore):
            def load_full(self, **_kwargs):
                raise AssertionError("memory base must be used")

            def load_active(self):
                raise AssertionError("memory base must be used")

        with patch("core.search_result_store.SearchResultStore", Store):
            self.assertEqual(json.loads(bridge.loadFullResults()), base)

    def test_load_full_reads_only_the_full_file_by_snapshot_and_records_the_base(self) -> None:
        parent = self._parent_with_snapshot([{"general": "a"}], base=None)
        bridge = VueBridge(parent)
        full = [{"general": "a"}, {"general": "b"}]
        calls = []

        class Store(_IdentityStore):
            last_error = None

            def load_full(self, *, expected_snapshot_id=None):
                calls.append(expected_snapshot_id)
                return full

            def load_active(self):
                raise AssertionError("active must not be re-parsed for the snapshot check")

        with patch("core.search_result_store.SearchResultStore", Store):
            self.assertEqual(json.loads(bridge.loadFullResults()), full)
            self.assertEqual(json.loads(bridge.loadFullResults()), full)

        self.assertEqual(calls, [_SNAPSHOT])
        self.assertIs(parent._search_base_results, full)

    def test_startup_restore_fills_runtime_without_serialising(self) -> None:
        from ui.generator_main import GeneratorMainUI

        active = [{"general": "restored"}]

        class Store:
            last_error = None
            last_snapshot_id = _SNAPSHOT
            last_dataset_identity = dict(_IDENTITY)

            @staticmethod
            def load_active():
                return active

        subject = _PublishingParent()
        subject._restore_deck_state = lambda: False
        dumped = []
        real_dumps = json.dumps

        def recording_dumps(value, *args, **kwargs):
            dumped.append(value)
            return real_dumps(value, *args, **kwargs)

        with (
            patch("core.search_result_store.SearchResultStore", Store),
            patch("json.dumps", side_effect=recording_dumps),
        ):
            GeneratorMainUI._restore_search_deck(subject)

        self.assertFalse(any(value is active for value in dumped), "시작 복원은 결과를 직렬화하지 않는다")
        self.assertIs(subject.filtered_results, active)
        self.assertEqual(subject._search_snapshot_id, _SNAPSHOT)
        self.assertEqual(subject.shuffled_prompt_deck, active)

    def test_startup_restore_is_skipped_when_a_snapshot_already_exists(self) -> None:
        from ui.generator_main import GeneratorMainUI

        subject = self._parent_with_snapshot([{"general": "fresh search"}])
        constructed = []

        class Store:
            def __init__(self):
                constructed.append(self)   # _restore_search_deck 가 예외를 삼키므로 기록으로 검증

            @staticmethod
            def load_active():
                return [{"general": "stale disk"}]

        with patch("core.search_result_store.SearchResultStore", Store):
            GeneratorMainUI._restore_search_deck(subject)

        self.assertEqual(constructed, [])
        self.assertEqual(subject.filtered_results, [{"general": "fresh search"}])


class TagSuggestionWarmupTests(unittest.TestCase):
    """예열 스레드가 적재 중일 때 GUI 슬롯은 락을 기다리지 않는다."""

    def setUp(self) -> None:
        import threading

        self._release = threading.Event()
        self._warm = threading.Thread(target=self._release.wait, daemon=True)
        self._warm.start()

    def tearDown(self) -> None:
        self._release.set()
        self._warm.join(1)

    class _Lookup:
        def __init__(self, ready: bool) -> None:
            self.ready = ready

        def is_ready(self) -> bool:
            return self.ready

        def search(self, _prefix, limit=10):
            raise AssertionError("must not block on the Korean catalogue while warming")

        def labels(self, tags):
            if not self.ready:
                raise AssertionError("must not block on the Korean catalogue while warming")
            return [{"tag": t, "ko": "라벨", "category": "", "desc": "", "count": 0} for t in tags]

    class _Completer:
        @staticmethod
        def get_suggestions(prefix, max_count=10):
            return [f"{prefix}_hair"]

    def _bridge(self) -> VueBridge:
        bridge = VueBridge()
        bridge._tag_warm_thread = self._warm
        return bridge

    def test_rich_returns_empty_while_the_completer_is_loading(self) -> None:
        bridge = self._bridge()
        with (
            patch("core.tag_korean_lookup.get_korean_tag_lookup", return_value=self._Lookup(False)),
            patch("utils.tag_completer.try_get_tag_completer", return_value=None),
        ):
            self.assertEqual(json.loads(bridge.getTagSuggestionsRich("long")), [])
        # 문자열 목록만 주던 옛 슬롯은 제거됐다 — 자동완성은 Rich 하나로만 간다(웹 화이트리스트 포함)
        self.assertFalse(hasattr(VueBridge, "getTagSuggestions"))

    def test_rich_returns_unlabelled_tags_while_only_korean_labels_load(self) -> None:
        bridge = self._bridge()
        with (
            patch("core.tag_korean_lookup.get_korean_tag_lookup", return_value=self._Lookup(False)),
            patch("utils.tag_completer.try_get_tag_completer", return_value=self._Completer()),
        ):
            self.assertEqual(json.loads(bridge.getTagSuggestionsRich("long")), [
                {"tag": "long_hair", "ko": "", "category": "", "desc": "", "count": 0}
            ])
            self.assertEqual(json.loads(bridge.getTagSuggestionsRich("장발")), [])

    def test_labels_are_attached_once_the_catalogue_is_ready(self) -> None:
        bridge = self._bridge()
        with (
            patch("core.tag_korean_lookup.get_korean_tag_lookup", return_value=self._Lookup(True)),
            patch("utils.tag_completer.try_get_tag_completer", return_value=self._Completer()),
        ):
            self.assertEqual(json.loads(bridge.getTagSuggestionsRich("long"))[0]["ko"], "라벨")


if __name__ == "__main__":
    unittest.main()
