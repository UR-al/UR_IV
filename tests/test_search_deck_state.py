from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from ui.generator_actions import ActionsMixin
from ui.generator_main import GeneratorMainUI
from ui.model_download_actions import ModelDownloadActionsMixin


_TEST_IDENTITY = {"label": "2026_07", "fingerprint": "f" * 64}
_TEST_SNAPSHOT = "f" * 32
_TEST_LINEAGE = {**_TEST_IDENTITY, "snapshot_id": _TEST_SNAPSHOT}


class _DeckSubject(ActionsMixin):
    def __init__(self, path: Path) -> None:
        self._path = path

    def _deck_state_path(self):
        return str(self._path)


class SearchDeckStateTests(unittest.TestCase):
    def test_deck_round_trip_requires_the_same_search_snapshot(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "last_deck.json"
            rows = [{"general": "first"}, {"general": "second"}]
            snapshot_id = "a" * 32

            subject = _DeckSubject(path)
            subject.filtered_results = rows
            subject.shuffled_prompt_deck = [rows[1]]
            subject._search_snapshot_id = snapshot_id
            subject._save_deck_state()

            stored = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(stored["snapshot_id"], snapshot_id)

            restored = _DeckSubject(path)
            restored.filtered_results = rows
            restored._search_snapshot_id = snapshot_id
            self.assertTrue(restored._restore_deck_state())
            self.assertEqual(restored.shuffled_prompt_deck, [rows[1]])

    def test_same_sized_deck_from_another_search_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "last_deck.json"
            path.write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "snapshot_id": "a" * 32,
                        "pool_size": 2,
                        "remaining": [1],
                    }
                ),
                encoding="utf-8",
            )
            subject = _DeckSubject(path)
            subject.filtered_results = [
                {"general": "different first"},
                {"general": "different second"},
            ]
            subject._search_snapshot_id = "b" * 32

            self.assertFalse(subject._restore_deck_state())

    def test_empty_vue_filter_clears_runtime_deck_and_persists_active_cache(self) -> None:
        class Subject(ModelDownloadActionsMixin):
            filtered_results = [{"general": "old"}]
            shuffled_prompt_deck = [{"general": "old"}]
            _rating_filter = {"g", "s", "q", "e"}
            _search_dataset_identity = _TEST_IDENTITY
            _search_snapshot_id = _TEST_SNAPSHOT

            def __init__(self) -> None:
                self.persisted = []
                self.deck_saves = 0
                self.statuses = []

            @staticmethod
            def _handle_creator_action(_action, _payload):
                return False

            @staticmethod
            def _handle_chat_action(_action, _payload):
                return False

            def _persist_search_results(self, active, full=None):
                self.persisted.append((active, full))

            def _save_deck_state(self):
                self.deck_saves += 1

            def show_status(self, message):
                self.statuses.append(message)

        subject = Subject()

        GeneratorMainUI._handle_vue_action(
            subject,
            "update_prompt_deck",
            {"results": [], "lineage": _TEST_LINEAGE},
        )

        self.assertEqual(subject.filtered_results, [])
        self.assertEqual(subject.shuffled_prompt_deck, [])
        self.assertEqual(subject.persisted, [([], None)])
        self.assertEqual(subject.deck_saves, 1)

    def test_empty_vue_filter_stops_running_automation(self) -> None:
        class Subject(ModelDownloadActionsMixin):
            filtered_results = [{"general": "old"}]
            shuffled_prompt_deck = [{"general": "old"}]
            _rating_filter = {"g", "s", "q", "e"}
            _search_dataset_identity = _TEST_IDENTITY
            _search_snapshot_id = _TEST_SNAPSHOT
            is_automating = True

            def __init__(self) -> None:
                self.stop_messages = []

            @staticmethod
            def _handle_creator_action(_action, _payload):
                return False

            @staticmethod
            def _handle_chat_action(_action, _payload):
                return False

            @staticmethod
            def _persist_search_results(_active, full=None):
                return None

            @staticmethod
            def _save_deck_state():
                return None

            def _stop_automation(self, message=None):
                self.is_automating = False
                self.stop_messages.append(message)

        subject = Subject()

        GeneratorMainUI._handle_vue_action(
            subject,
            "update_prompt_deck",
            {"results": [], "lineage": _TEST_LINEAGE},
        )

        self.assertFalse(subject.is_automating)
        self.assertEqual(
            subject.stop_messages,
            ["검색 필터 결과가 없어 자동화를 중지했습니다."],
        )

    def test_delayed_filter_from_another_snapshot_is_rejected(self) -> None:
        current_identity = {"label": "2026_07", "fingerprint": "b" * 64}
        stale_lineage = {
            "label": "2026_07",
            "fingerprint": "a" * 64,
            "snapshot_id": "a" * 32,
        }

        class Subject(ModelDownloadActionsMixin):
            filtered_results = [{"general": "current_b"}]
            shuffled_prompt_deck = [{"general": "current_b"}]
            _rating_filter = {"g", "s", "q", "e"}
            _search_dataset_identity = current_identity
            _search_snapshot_id = "b" * 32

            def __init__(self) -> None:
                self.persisted = []
                self.statuses = []

            @staticmethod
            def _handle_creator_action(_action, _payload):
                return False

            @staticmethod
            def _handle_chat_action(_action, _payload):
                return False

            def _persist_search_results(self, active, full=None):
                self.persisted.append((active, full))

            @staticmethod
            def _save_deck_state():
                return None

            def show_status(self, message):
                self.statuses.append(message)

        subject = Subject()

        GeneratorMainUI._handle_vue_action(
            subject,
            "update_prompt_deck",
            {
                "results": [{"general": "stale_a"}],
                "lineage": stale_lineage,
            },
        )

        self.assertEqual(subject.filtered_results, [{"general": "current_b"}])
        self.assertEqual(subject.shuffled_prompt_deck, [{"general": "current_b"}])
        self.assertEqual(subject.persisted, [])
        self.assertTrue(any("일치하지" in item for item in subject.statuses))

    def test_imported_results_start_a_new_cache_and_deck_snapshot(self) -> None:
        identity = {"label": "2026_07", "fingerprint": "f" * 64}

        class Signal:
            def __init__(self) -> None:
                self.values = []

            def emit(self, value) -> None:
                self.values.append(value)

        class Bridge:
            searchResultsReady = Signal()
            searchResultLineage = Signal()

        class Store:
            @staticmethod
            def dataset_info():
                return identity

        class Subject(ModelDownloadActionsMixin):
            vue_bridge = Bridge()
            _search_snapshot_id = "a" * 32

            def __init__(self) -> None:
                self.persisted = []
                self.deck_saves = 0
                self.statuses = []

            @staticmethod
            def _handle_creator_action(_action, _payload):
                return False

            @staticmethod
            def _handle_chat_action(_action, _payload):
                return False

            def _persist_search_results(self, active, full=None, **kwargs):
                self.persisted.append((active, full, kwargs))

            def _save_deck_state(self):
                self.deck_saves += 1

            def show_status(self, message):
                self.statuses.append(message)

        frame = pd.DataFrame(
            [
                {
                    "general": "imported_tag",
                    "character": "",
                    "copyright": "",
                    "artist": "",
                    "rating": "g",
                }
            ]
        )
        subject = Subject()

        with (
            patch(
                "ui.generator_main.QFileDialog.getOpenFileName",
                return_value=("import.parquet", "Parquet Files (*.parquet)"),
            ),
            patch("pandas.read_parquet", return_value=frame),
            patch("core.search_result_store.SearchResultStore", Store),
        ):
            GeneratorMainUI._handle_vue_action(
                subject,
                "import_search_results",
                {},
            )

        self.assertEqual(len(subject.filtered_results), 1)
        self.assertNotEqual(subject._search_snapshot_id, "a" * 32)
        self.assertEqual(subject._search_dataset_identity, identity)
        active, full, kwargs = subject.persisted[0]
        self.assertEqual(active, full)
        self.assertEqual(kwargs["dataset_identity"], identity)
        self.assertEqual(kwargs["snapshot_id"], subject._search_snapshot_id)
        self.assertEqual(subject.deck_saves, 1)
        self.assertEqual(
            json.loads(subject.vue_bridge.searchResultLineage.values[0]),
            {
                **identity,
                "snapshot_id": subject._search_snapshot_id,
            },
        )


class _DeckActionSubject(ModelDownloadActionsMixin):
    """update_prompt_deck / import_search_results 만 도는 메인 윈도우 대역."""

    _rating_filter = {"g", "s", "q", "e"}

    def __init__(self) -> None:
        self.persisted = []
        self.deck_saves = 0
        self.statuses = []

    @staticmethod
    def _handle_creator_action(_action, _payload):
        return False

    @staticmethod
    def _handle_chat_action(_action, _payload):
        return False

    def _persist_search_results(self, active, full=None, **kwargs):
        self.persisted.append((active, full, kwargs))

    def _save_deck_state(self):
        self.deck_saves += 1

    def show_status(self, message):
        self.statuses.append(message)


class SearchDeckIndexPayloadTests(unittest.TestCase):
    def _subject_with_base(self, base):
        from core.search_session import publish_snapshot

        subject = _DeckActionSubject()
        publish_snapshot(
            subject,
            active=base,
            base=base,
            identity=dict(_TEST_IDENTITY),
            snapshot_id=_TEST_SNAPSHOT,
        )
        subject.shuffled_prompt_deck = list(base)
        return subject

    def test_index_filter_rebuilds_the_deck_from_the_python_base(self) -> None:
        base = [
            {"general": "a", "rating": "g"},
            {"general": "b", "rating": "e"},
            {"general": "c", "rating": "g"},
        ]
        subject = self._subject_with_base(base)
        subject._rating_filter = {"g"}

        GeneratorMainUI._handle_vue_action(
            subject,
            "update_prompt_deck",
            {"indices": [1, 2], "base_size": 3, "lineage": _TEST_LINEAGE},
        )

        self.assertEqual(len(subject.filtered_results), 2)
        self.assertIs(subject.filtered_results[0], base[1])
        self.assertIs(subject.filtered_results[1], base[2])
        # rating 필터(g)는 덱에만 적용 — 풀은 필터 결과 그대로
        self.assertEqual(subject.shuffled_prompt_deck, [base[2]])
        # base 는 필터 해제용으로 그대로 남는다
        self.assertIs(subject._search_base_results, base)
        self.assertEqual(subject.persisted, [(subject.filtered_results, None, {})])
        self.assertEqual(subject.deck_saves, 1)

    def test_index_filter_for_a_different_base_is_rejected_without_mutation(self) -> None:
        base = [{"general": "a"}, {"general": "b"}]
        subject = self._subject_with_base(base)

        GeneratorMainUI._handle_vue_action(
            subject,
            "update_prompt_deck",
            {"indices": [0], "base_size": 5, "lineage": _TEST_LINEAGE},
        )

        self.assertIs(subject.filtered_results, base)
        self.assertEqual(subject.persisted, [])
        self.assertEqual(subject.deck_saves, 0)
        self.assertTrue(any("기준 목록" in s for s in subject.statuses), subject.statuses)

    def test_index_filter_with_a_stale_lineage_is_rejected(self) -> None:
        base = [{"general": "a"}]
        subject = self._subject_with_base(base)

        GeneratorMainUI._handle_vue_action(
            subject,
            "update_prompt_deck",
            {
                "indices": [0],
                "base_size": 1,
                "lineage": {**_TEST_LINEAGE, "snapshot_id": "e" * 32},
            },
        )

        self.assertEqual(subject.persisted, [])
        self.assertTrue(any("일치하지" in s for s in subject.statuses))

    def test_large_filter_payload_is_not_serialised_for_the_action_log(self) -> None:
        base = [{"general": str(i)} for i in range(3)]
        subject = self._subject_with_base(base)
        payload = {"results": list(base), "lineage": _TEST_LINEAGE}

        with patch("ui.generator_main.json.dumps", side_effect=AssertionError("log must not dump payload")):
            GeneratorMainUI._handle_vue_action(subject, "update_prompt_deck", payload)

        self.assertEqual(subject.filtered_results, base)

    def test_none_payload_does_not_break_the_action_log(self) -> None:
        subject = _DeckActionSubject()
        # 예전 json.dumps(payload)[:100] 은 dict 가 아닌 payload 에서도 돌았지만, 요약 로그가
        # 예외를 내면 onAction 의 except 에 먹혀 액션이 사라진다 — None 도 안전해야 한다.
        GeneratorMainUI._handle_vue_action(subject, "reset_prompt_deck", None)

    def test_non_ascii_payload_on_a_cp949_stdout_still_runs_the_action(self) -> None:
        import io

        base = [{"general": "a"}, {"general": "b"}]
        subject = self._subject_with_base(base)
        stdout = io.TextIOWrapper(io.BytesIO(), encoding="cp949")
        payload = {
            "indices": [1],
            "base_size": 2,
            "lineage": _TEST_LINEAGE,
            "note": "Pokémon café 😀",   # cp949 로 인코딩할 수 없는 글자
        }

        with patch("sys.stdout", stdout):
            GeneratorMainUI._handle_vue_action(subject, "update_prompt_deck", payload)

        self.assertEqual(subject.filtered_results, [base[1]])
        stdout.flush()
        self.assertIn(b"Pok\\xe9mon", stdout.buffer.getvalue())

    def test_rejected_filter_is_shown_as_a_notification(self) -> None:
        from core.search_session import BASE_MISMATCH_MESSAGE

        class Signal:
            def __init__(self) -> None:
                self.values = []

            def emit(self, *args) -> None:
                self.values.append(args)

        class Bridge:
            def __init__(self) -> None:
                self.showNotification = Signal()

        base = [{"general": "a"}, {"general": "b"}]
        subject = self._subject_with_base(base)
        subject.vue_bridge = Bridge()

        GeneratorMainUI._handle_vue_action(
            subject,
            "update_prompt_deck",
            {"indices": [0], "base_size": 5, "lineage": _TEST_LINEAGE},
        )

        # Vue 모드의 show_status 는 더미 라벨 — 사용자는 알림으로만 본다
        self.assertEqual(
            subject.vue_bridge.showNotification.values,
            [("warning", BASE_MISMATCH_MESSAGE)],
        )
        self.assertIn(BASE_MISMATCH_MESSAGE, subject.statuses)
        self.assertIs(subject.filtered_results, base)

    def test_index_filter_after_disk_restore_loads_the_base_lazily(self) -> None:
        from core.search_session import restore_runtime_from_disk, runtime_base_rows

        full = [{"general": "a", "rating": "g"}, {"general": "b", "rating": "g"},
                {"general": "c", "rating": "g"}]

        class Store:
            last_error = None
            last_snapshot_id = _TEST_SNAPSHOT
            last_dataset_identity = dict(_TEST_IDENTITY)

            @staticmethod
            def dataset_info():
                return dict(_TEST_IDENTITY)

            @staticmethod
            def load_active():
                return [{"general": "a", "rating": "g"}, {"general": "c", "rating": "g"}]

            @staticmethod
            def load_full(*, expected_snapshot_id=None):
                assert expected_snapshot_id == _TEST_SNAPSHOT
                return full

        subject = _DeckActionSubject()
        subject.filtered_results = []
        subject.shuffled_prompt_deck = []
        restore_runtime_from_disk(subject, Store())
        self.assertIsNone(runtime_base_rows(subject))   # loadFullResults 가 기록 못 한 상태

        with patch("core.search_result_store.SearchResultStore", Store):
            GeneratorMainUI._handle_vue_action(
                subject,
                "update_prompt_deck",
                {"indices": [1], "base_size": 3, "lineage": _TEST_LINEAGE},
            )

        self.assertEqual(subject.statuses, [])
        self.assertEqual(len(subject.filtered_results), 1)
        self.assertIs(subject.filtered_results[0], full[1])
        self.assertIs(runtime_base_rows(subject), full)


class ImportSearchResultsNormalizationTests(unittest.TestCase):
    def _import(self, frame, payload=None):
        identity = {"label": "2026_07", "fingerprint": "f" * 64}

        class Signal:
            def __init__(self) -> None:
                self.values = []

            def emit(self, value) -> None:
                self.values.append(value)

        class Bridge:
            def __init__(self) -> None:
                self.searchResultsReady = Signal()
                self.searchResultLineage = Signal()

        class Store:
            @staticmethod
            def dataset_info():
                return identity

        subject = _DeckActionSubject()
        subject.vue_bridge = Bridge()
        with (
            patch(
                "ui.generator_main.QFileDialog.getOpenFileName",
                return_value=("import.parquet", "Parquet Files (*.parquet)"),
            ),
            patch("pandas.read_parquet", return_value=frame),
            patch("core.search_result_store.SearchResultStore", Store),
        ):
            GeneratorMainUI._handle_vue_action(
                subject, "import_search_results", payload if payload is not None else {}
            )
        return subject

    def test_import_follows_the_search_result_cap_unless_unlimited(self) -> None:
        frame = pd.DataFrame({"general": [f"tag_{i}" for i in range(12)]})

        with patch("core.search_rows.SEARCH_RESULT_CAP", 5):
            capped = self._import(frame)
            unlimited = self._import(frame, {"disable_result_cap": True})

        self.assertEqual(len(capped.filtered_results), 5)
        self.assertTrue(any("random sample" in s for s in capped.statuses), capped.statuses)
        self.assertEqual(len(unlimited.filtered_results), 12)

    def test_missing_values_become_empty_strings_not_nan(self) -> None:
        frame = pd.DataFrame({
            "general": pd.array(["1girl", None], dtype="str"),
            "character": pd.array([None, "miku"], dtype="str"),
            "copyright": ["original", "vocaloid"],
            "artist": ["", "x"],
            "rating": pd.array([None, "s"], dtype="str"),
            "image_width": [1024.0, float("nan")],
            "image_height": pd.array([768, None], dtype="Int64"),
        })

        subject = self._import(frame)

        self.assertEqual(subject.filtered_results, [
            {"copyright": "original", "character": "", "artist": "", "general": "1girl",
             "rating": "", "image_width": 1024, "image_height": 768},
            {"copyright": "vocaloid", "character": "miku", "artist": "x", "general": "",
             "rating": "s", "image_width": None, "image_height": None},
        ])
        published = subject.vue_bridge.searchResultsReady.values[0]
        self.assertNotIn("nan", published)
        self.assertNotIn("NaN", published)
        self.assertEqual(json.loads(published), subject.filtered_results)

    def test_import_prefers_non_empty_tag_string_columns_like_search(self) -> None:
        frame = pd.DataFrame({
            "tag_string_general": ["from_tag_string", ""],
            "general": ["plain", "fallback"],
        })

        subject = self._import(frame)

        self.assertEqual(
            [row["general"] for row in subject.filtered_results],
            ["from_tag_string", "fallback"],
        )

    def test_imported_rows_become_the_filter_base(self) -> None:
        subject = self._import(pd.DataFrame({"general": ["a", "b"]}))

        self.assertIs(subject._search_base_results, subject.filtered_results)
        self.assertEqual(subject._search_base_snapshot_id, subject._search_snapshot_id)
        self.assertFalse(hasattr(subject, "_last_search_results"))


if __name__ == "__main__":
    unittest.main()
