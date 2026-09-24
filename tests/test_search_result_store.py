from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from core.search_result_store import (
    SCHEMA_VERSION,
    SearchResultStore,
    reserve_write_sequence,
)
from core.storage_paths import StoragePaths


class SearchResultStoreTests(unittest.TestCase):
    def setUp(self) -> None:
        self._temporary = tempfile.TemporaryDirectory()
        self.project_root = Path(self._temporary.name).resolve()
        self.manifest_path = (
            self.project_root / "danbooru_optimized" / "dataset_manifest.json"
        )
        self.manifest_path.parent.mkdir(parents=True)
        self._write_manifest("2026_07")

    def tearDown(self) -> None:
        self._temporary.cleanup()

    def _write_manifest(self, label: str) -> None:
        self.manifest_path.write_text(
            json.dumps({"dataset_label": label}),
            encoding="utf-8",
        )

    def test_active_and_full_results_round_trip_with_dataset_provenance(self) -> None:
        store = SearchResultStore(project_root=self.project_root)
        active = [{"general": "1girl", "score": 10}]
        full = active + [{"general": "solo", "score": 7}]

        store.save(active, full=full)

        self.assertEqual(store.load_active(), active)
        self.assertEqual(store.load_full(), full)
        self.assertEqual(store.dataset_info()["label"], "2026_07")
        expected_active_envelope = {
            "schema_version": SCHEMA_VERSION,
            "dataset_label": "2026_07",
            "dataset_fingerprint": store._dataset_identity()[1],
            "snapshot_id": json.loads(store.full_path.read_text(encoding="utf-8"))[
                "snapshot_id"
            ],
            "results": active,
        }
        self.assertEqual(
            json.loads(
                (self.project_root / "cache/search/last_search_results.json")
                .read_text(encoding="utf-8")
            ),
            expected_active_envelope,
        )

    def test_results_from_a_different_dataset_label_are_rejected(self) -> None:
        store = SearchResultStore(project_root=self.project_root)
        store.save([{"general": "old active"}], full=[{"general": "old full"}])

        self._write_manifest("2026_08")

        self.assertEqual(store.load_active(), [])
        self.assertIn("dataset label", store.last_error.lower())
        self.assertEqual(store.load_full(), [])
        self.assertIn("dataset label", store.last_error.lower())

    def test_missing_manifest_rejects_cache_save_and_load(self) -> None:
        self.manifest_path.unlink()
        store = SearchResultStore(project_root=self.project_root)

        with self.assertRaisesRegex(ValueError, "manifest is required"):
            store.save([{"general": "unverified dataset"}])

        store.active_path.write_text("{}", encoding="utf-8")
        self.assertEqual(store.load_active(), [])
        self.assertIn("manifest is required", store.last_error)

    def test_explicit_dataset_root_controls_cache_provenance(self) -> None:
        custom_root = self.project_root / "external-search-data"
        custom_root.mkdir()
        (custom_root / "dataset_manifest.json").write_text(
            json.dumps({"dataset_label": "next_release"}),
            encoding="utf-8",
        )
        store = SearchResultStore(
            project_root=self.project_root,
            dataset_root=custom_root,
        )

        store.save([{"general": "custom root"}])

        envelope = json.loads(store.active_path.read_text(encoding="utf-8"))
        self.assertEqual(envelope["dataset_label"], "next_release")
        self.assertEqual(store.load_active(), [{"general": "custom root"}])

    def test_same_label_with_a_different_manifest_rejects_old_cache(self) -> None:
        first_root = self.project_root / "first-data"
        second_root = self.project_root / "second-data"
        first_root.mkdir()
        second_root.mkdir()
        (first_root / "dataset_manifest.json").write_text(
            json.dumps({"dataset_label": "2026_07", "source_revision": "first"}),
            encoding="utf-8",
        )
        (second_root / "dataset_manifest.json").write_text(
            json.dumps({"dataset_label": "2026_07", "source_revision": "second"}),
            encoding="utf-8",
        )
        first_store = SearchResultStore(
            project_root=self.project_root,
            dataset_root=first_root,
        )
        first_store.save([{"general": "first dataset"}])

        second_store = SearchResultStore(
            project_root=self.project_root,
            dataset_root=second_root,
        )
        self.assertEqual(second_store.load_active(), [])
        self.assertIn("fingerprint", second_store.last_error.lower())

    def test_legacy_bare_lists_are_migrated_but_rejected_without_provenance(self) -> None:
        legacy_active = self.project_root / "config/last_search_results.json"
        legacy_full = self.project_root / "config/last_full_results.json"
        legacy_active.parent.mkdir(parents=True)
        legacy_active.write_text('[{"general":"legacy active"}]', encoding="utf-8")
        legacy_full.write_text('[{"general":"legacy full"}]', encoding="utf-8")

        store = SearchResultStore(storage=StoragePaths(self.project_root))

        self.assertFalse(legacy_active.exists())
        self.assertFalse(legacy_full.exists())
        self.assertTrue(store.active_path.is_file())
        self.assertTrue(store.full_path.is_file())
        self.assertEqual(store.load_active(), [])
        self.assertIn("provenance", store.last_error.lower())
        self.assertEqual(store.load_full(), [])
        self.assertIn("provenance", store.last_error.lower())

    def test_malformed_manifest_returns_empty_results_with_an_explicit_error(self) -> None:
        store = SearchResultStore(project_root=self.project_root)
        store.save([{"general": "safe"}], full=[{"general": "safe full"}])
        self.manifest_path.write_text('{"dataset_label":', encoding="utf-8")

        self.assertEqual(store.load_active(), [])
        self.assertIn("manifest", store.last_error.lower())
        self.assertEqual(store.load_full(), [])
        self.assertIn("manifest", store.last_error.lower())

    def test_unsafe_manifest_label_is_rejected(self) -> None:
        store = SearchResultStore(project_root=self.project_root)
        store.save([{"general": "safe"}])
        self._write_manifest("../other-dataset")

        self.assertEqual(store.load_active(), [])
        self.assertIn("unsafe", store.last_error.lower())

    def test_existing_manifest_without_a_dataset_label_is_rejected(self) -> None:
        store = SearchResultStore(project_root=self.project_root)
        store.save([{"general": "safe"}])
        self.manifest_path.write_text("{}", encoding="utf-8")

        self.assertEqual(store.load_active(), [])
        self.assertIn("dataset label", store.last_error.lower())

    def test_schema_version_must_be_exactly_integer_one(self) -> None:
        store = SearchResultStore(project_root=self.project_root)
        store.active_path.write_text(
            json.dumps(
                {
                    "schema_version": True,
                    "dataset_label": "2026_07",
                    "dataset_fingerprint": store._dataset_identity()[1],
                    "snapshot_id": "0" * 32,
                    "results": [{"general": "ambiguous schema"}],
                }
            ),
            encoding="utf-8",
        )

        self.assertEqual(store.load_active(), [])
        self.assertIn("schema", store.last_error.lower())

    def test_active_only_and_full_only_updates_preserve_the_other_result_set(self) -> None:
        store = SearchResultStore(project_root=self.project_root)
        original_active = [{"general": "active one"}]
        original_full = [{"general": "full one"}, {"general": "full two"}]
        store.save(original_active, full=original_full)

        updated_active = [{"general": "filtered active"}]
        store.save(updated_active)
        self.assertEqual(store.load_active(), updated_active)
        self.assertEqual(store.load_full(), original_full)

        updated_full = [{"general": "replacement full"}]
        store.save_full(updated_full)
        self.assertEqual(store.load_active(), updated_active)
        self.assertEqual(store.load_full(), updated_full)

    def test_invalid_full_payload_does_not_replace_a_valid_active_cache(self) -> None:
        store = SearchResultStore(project_root=self.project_root)
        original_active = [{"general": "keep me"}]
        store.save(original_active)

        with self.assertRaises(TypeError):
            store.save([{"general": "must not publish"}], full={"not": "a list"})

        self.assertEqual(store.load_active(), original_active)

    def test_non_object_rows_are_rejected_on_save_and_load(self) -> None:
        store = SearchResultStore(project_root=self.project_root)
        with self.assertRaises(TypeError):
            store.save(["not an object"])

        store.active_path.write_text(
            json.dumps(
                {
                    "schema_version": SCHEMA_VERSION,
                    "dataset_label": "2026_07",
                    "dataset_fingerprint": store._dataset_identity()[1],
                    "snapshot_id": "0" * 32,
                    "results": ["not an object"],
                }
            ),
            encoding="utf-8",
        )
        self.assertEqual(store.load_active(), [])
        self.assertIn("objects", store.last_error.lower())

    def test_partial_pair_publish_never_mixes_two_search_snapshots(self) -> None:
        store = SearchResultStore(project_root=self.project_root)
        old_active = [{"general": "old active"}]
        old_full = [{"general": "old full"}]
        store.save(old_active, full=old_full)

        from core.search_result_store import atomic_write_json as real_write

        calls = 0

        def fail_second_write(path, data, **kwargs):
            nonlocal calls
            calls += 1
            if calls == 2:
                raise OSError("simulated active publish failure")
            return real_write(path, data, **kwargs)

        with patch(
            "core.search_result_store.atomic_write_json",
            side_effect=fail_second_write,
        ):
            with self.assertRaises(OSError):
                store.save(
                    [{"general": "new active"}],
                    full=[{"general": "new full"}],
                )

        self.assertEqual(store.load_active(), old_active)
        self.assertEqual(store.load_full(), [])
        self.assertIn("different snapshots", store.last_error.lower())

    def test_out_of_order_background_write_cannot_replace_newer_results(self) -> None:
        store = SearchResultStore(project_root=self.project_root)
        old_sequence = reserve_write_sequence()
        new_sequence = reserve_write_sequence()

        self.assertTrue(
            store.save_if_latest(new_sequence, [{"general": "newest"}])
        )
        self.assertFalse(
            store.save_if_latest(old_sequence, [{"general": "stale"}])
        )
        self.assertEqual(store.load_active(), [{"general": "newest"}])

    def test_manifest_change_before_background_write_rejects_old_results(self) -> None:
        store = SearchResultStore(project_root=self.project_root)
        searched_identity = store.dataset_info()
        snapshot_id = "a" * 32
        self.manifest_path.write_text(
            json.dumps(
                {
                    "dataset_label": "2026_07",
                    "source_revision": "changed after search",
                }
            ),
            encoding="utf-8",
        )

        with self.assertRaisesRegex(ValueError, "changed before"):
            store.save(
                [{"general": "old result"}],
                full=[{"general": "old result"}],
                snapshot_id=snapshot_id,
                expected_identity=searched_identity,
            )

        self.assertFalse(store.active_path.exists())
        self.assertFalse(store.full_path.exists())

    def test_newer_filter_write_does_not_discard_its_pending_full_base(self) -> None:
        store = SearchResultStore(project_root=self.project_root)
        snapshot_id = "a" * 32
        full_sequence = reserve_write_sequence(snapshot_id)
        filter_sequence = reserve_write_sequence(snapshot_id)

        self.assertTrue(
            store.save_if_latest(
                filter_sequence,
                [{"general": "filtered active"}],
                snapshot_id=snapshot_id,
            )
        )
        self.assertTrue(
            store.save_if_latest(
                full_sequence,
                [{"general": "unfiltered active"}],
                full=[
                    {"general": "unfiltered active"},
                    {"general": "second result"},
                ],
                snapshot_id=snapshot_id,
            )
        )

        self.assertEqual(store.load_active(), [{"general": "filtered active"}])
        self.assertEqual(
            store.load_full(),
            [
                {"general": "unfiltered active"},
                {"general": "second result"},
            ],
        )

    def test_empty_active_filter_keeps_a_valid_full_restore_base(self) -> None:
        store = SearchResultStore(project_root=self.project_root)
        store.save(
            [],
            full=[{"general": "full result"}],
            snapshot_id="a" * 32,
        )

        self.assertEqual(store.load_active(), [])
        self.assertEqual(store.load_full(), [{"general": "full result"}])

    def test_orphaned_full_cache_is_not_restored(self) -> None:
        store = SearchResultStore(project_root=self.project_root)
        store.save_full([{"general": "orphan"}])

        self.assertEqual(store.load_full(), [])
        self.assertIn("no matching active", store.last_error)

    def test_expected_snapshot_loads_full_without_reparsing_active(self) -> None:
        store = SearchResultStore(project_root=self.project_root)
        store.save(
            [{"general": "active"}],
            full=[{"general": "active"}, {"general": "base only"}],
            snapshot_id="a" * 32,
        )
        reads = []
        real_read = SearchResultStore._read_envelope

        def tracking_read(path, *args):
            reads.append(Path(path).name)
            return real_read(path, *args)

        with patch.object(
            SearchResultStore, "_read_envelope", staticmethod(tracking_read)
        ):
            loaded = store.load_full(expected_snapshot_id="A" * 32)

        self.assertEqual(
            loaded, [{"general": "active"}, {"general": "base only"}]
        )
        self.assertEqual(reads, ["last_full_results.json"])
        self.assertIsNone(store.last_error)
        self.assertEqual(store.last_snapshot_id, "a" * 32)

    def test_expected_snapshot_rejects_a_full_base_from_another_snapshot(self) -> None:
        store = SearchResultStore(project_root=self.project_root)
        store.save([{"general": "a"}], full=[{"general": "a"}], snapshot_id="a" * 32)

        self.assertEqual(store.load_full(expected_snapshot_id="b" * 32), [])
        self.assertIn("different snapshot", store.last_error)
        self.assertIsNone(store.last_snapshot_id)

    def test_expected_snapshot_still_validates_dataset_and_id_format(self) -> None:
        store = SearchResultStore(project_root=self.project_root)
        store.save([{"general": "a"}], full=[{"general": "a"}], snapshot_id="a" * 32)

        self.assertEqual(store.load_full(expected_snapshot_id="../bad"), [])
        self.assertIn("invalid", store.last_error)

        self._write_manifest("2026_08")
        self.assertEqual(store.load_full(expected_snapshot_id="a" * 32), [])
        self.assertIn("dataset label", store.last_error.lower())

    def test_expected_snapshot_without_full_file_is_empty_not_an_error(self) -> None:
        store = SearchResultStore(project_root=self.project_root)

        self.assertEqual(store.load_full(expected_snapshot_id="a" * 32), [])
        self.assertIsNone(store.last_error)


if __name__ == "__main__":
    unittest.main()
