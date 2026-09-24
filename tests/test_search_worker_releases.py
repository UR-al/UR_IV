from __future__ import annotations

import hashlib
import json
import os
import tempfile
import unittest
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

from workers.search_worker import PandasSearchWorker


class SearchWorkerReleaseTests(unittest.TestCase):
    def setUp(self):
        self._temp_dir = tempfile.TemporaryDirectory()
        self.data_dir = Path(self._temp_dir.name)
        self._reset_cache()

    def tearDown(self):
        self._reset_cache()
        self._temp_dir.cleanup()

    @staticmethod
    def _reset_cache():
        PandasSearchWorker.cached_df = None
        PandasSearchWorker.loaded_ratings = set()
        PandasSearchWorker.loaded_year = ""
        if hasattr(PandasSearchWorker, "loaded_file_signature"):
            PandasSearchWorker.loaded_file_signature = ()
        if hasattr(PandasSearchWorker, "verified_artifact_signatures"):
            PandasSearchWorker.verified_artifact_signatures.clear()

    def _write_release(
        self,
        year: str,
        rating: str,
        general: str,
        *,
        drop_columns=(),
        **overrides,
    ):
        row = {
            "rating": rating,
            "general": general,
            "character": "release_tester",
            "copyright": "search_contract",
            "artist": "test_artist",
            "meta": "",
            "image_width": 1024,
            "image_height": 1024,
        }
        row.update(overrides)
        for column in drop_columns:
            row.pop(column, None)
        path = self.data_dir / f"danbooru_{year}_{rating}.parquet"
        pd.DataFrame([row]).to_parquet(path, index=False)
        return path

    @staticmethod
    def _sha256(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()

    def _artifact(self, year: str, rating: str, **overrides):
        path = self.data_dir / f"danbooru_{year}_{rating}.parquet"
        if path.is_file():
            parquet = pq.ParquetFile(path)
            schema = [
                {
                    "name": field.name,
                    "type": str(field.type),
                    "nullable": bool(field.nullable),
                }
                for field in parquet.schema_arrow
            ]
            artifact = {
                "kind": "search",
                "format": "parquet",
                "path": path.name,
                "rating_shard": rating,
                "rows": parquet.metadata.num_rows,
                "schema": schema,
                "sha256": self._sha256(path),
                "size_bytes": path.stat().st_size,
            }
        else:
            artifact = {
                "kind": "search",
                "format": "parquet",
                "path": path.name,
                "rating_shard": rating,
                "rows": 0,
                "schema": [],
                "sha256": "0" * 64,
                "size_bytes": 0,
            }
        artifact.update(overrides)
        return artifact

    def _write_manifest(self, year: str, ratings=("g",), *, artifacts=None):
        if artifacts is None:
            artifacts = [self._artifact(year, rating) for rating in ratings]
        payload = {
            "format_version": 1,
            "dataset_label": year,
            "artifacts": artifacts,
        }
        (self.data_dir / "dataset_manifest.json").write_text(
            json.dumps(payload, ensure_ascii=False),
            encoding="utf-8",
        )

    def _run(self, *, ratings=("g",)):
        emissions = []
        worker = PandasSearchWorker(
            str(self.data_dir),
            ratings,
            queries={},
        )
        worker.results_ready.connect(
            lambda rows, total: emissions.append((rows, total))
        )
        worker.run()
        self.assertEqual(len(emissions), 1)
        return worker, emissions[0]

    def _run_failure(self, *, ratings=("g",)):
        emissions = []
        statuses = []
        worker = PandasSearchWorker(str(self.data_dir), ratings, queries={})
        worker.results_ready.connect(
            lambda rows, total: emissions.append((rows, total))
        )
        worker.status_update.connect(statuses.append)
        worker.run()
        self.assertEqual(emissions, [])
        self.assertTrue(any(status.startswith("❌") for status in statuses), statuses)
        return worker, statuses

    def test_only_active_2026_07_release_is_loaded(self):
        self._write_release("2026_07", "g", "new_release_tag")
        self._write_release("2026_06", "g", "previous_release_tag")
        self._write_manifest("2026_07")

        worker, (rows, total) = self._run()

        self.assertEqual(worker.dataset_year, "2026_07")
        self.assertEqual(total, 1)
        self.assertEqual(rows[0]["general"], "new_release_tag")

    def test_old_release_is_not_used_when_latest_is_missing(self):
        self._write_release("2026_06", "g", "fallback_release_tag")
        self._write_manifest("2026_07")

        worker, _statuses = self._run_failure()

        self.assertEqual(worker.dataset_year, "2026_07")

    def test_incomplete_active_release_does_not_mix_old_shards(self):
        self._write_release("2026_07", "g", "partial_latest_tag")
        self._write_release("2026_06", "g", "fallback_general_tag")
        self._write_release("2026_06", "s", "fallback_sensitive_tag")
        self._write_manifest("2026_07", ("g", "s"))

        worker, _statuses = self._run_failure(ratings=("g", "s"))

        self.assertEqual(worker.dataset_year, "2026_07")

    def test_active_release_missing_required_schema_fails_as_a_whole(self):
        self._write_release(
            "2026_07",
            "g",
            "invalid_latest_tag",
            drop_columns=("artist",),
        )
        self._write_release("2026_06", "g", "valid_fallback_tag")
        self._write_manifest("2026_07")

        worker, _statuses = self._run_failure()

        self.assertEqual(worker.dataset_year, "2026_07")

    def test_manifest_selects_the_single_active_release(self):
        self._write_release("next_release", "g", "manifest_release_tag")
        self._write_manifest("next_release")
        manifest_bytes = (self.data_dir / "dataset_manifest.json").read_bytes()

        worker, (rows, total) = self._run()

        self.assertEqual(worker.dataset_year, "next_release")
        self.assertEqual(worker.dataset_identity, {
            "label": "next_release",
            "fingerprint": hashlib.sha256(manifest_bytes).hexdigest(),
        })
        self.assertEqual(total, 1)
        self.assertEqual(rows[0]["general"], "manifest_release_tag")

    def test_invalid_manifest_never_falls_back_to_an_old_release(self):
        self._write_release("2026_06", "g", "old_release_tag")
        (self.data_dir / "dataset_manifest.json").write_text(
            json.dumps({
                "format_version": 1,
                "dataset_label": "../unsafe",
                "artifacts": [{}],
            }),
            encoding="utf-8",
        )
        statuses = []
        worker = PandasSearchWorker(str(self.data_dir), ("g",), queries={})
        worker.status_update.connect(statuses.append)
        emissions = []
        worker.results_ready.connect(
            lambda rows, total: emissions.append((rows, total))
        )

        worker.run()

        self.assertEqual(emissions, [])
        self.assertTrue(
            any("manifest 오류" in status for status in statuses),
            statuses,
        )

    def test_non_string_manifest_label_is_rejected(self):
        self._write_release("2026_07", "g", "current_release_tag")
        (self.data_dir / "dataset_manifest.json").write_text(
            json.dumps({
                "format_version": 1,
                "dataset_label": 202607,
                "artifacts": [{}],
            }),
            encoding="utf-8",
        )
        statuses = []
        worker = PandasSearchWorker(str(self.data_dir), ("g",), queries={})
        worker.status_update.connect(statuses.append)
        emissions = []
        worker.results_ready.connect(
            lambda rows, total: emissions.append((rows, total))
        )

        worker.run()

        self.assertEqual(emissions, [])
        self.assertTrue(any("문자열" in status for status in statuses), statuses)

    def test_replacing_a_release_file_invalidates_the_dataframe_cache(self):
        self._write_release("2026_07", "g", "before_update")
        self._write_manifest("2026_07")
        _, (first_rows, _) = self._run()
        path = self.data_dir / "danbooru_2026_07_g.parquet"
        previous_mtime = path.stat().st_mtime_ns

        self._write_release("2026_07", "g", "after_update")
        os.utime(
            path,
            ns=(path.stat().st_atime_ns, previous_mtime + 2_000_000_000),
        )
        self._write_manifest("2026_07")
        _, (second_rows, _) = self._run()

        self.assertEqual(first_rows[0]["general"], "before_update")
        self.assertEqual(second_rows[0]["general"], "after_update")

    def test_invalid_release_emits_an_error_without_replacing_results(self):
        self._write_release(
            "2026_07",
            "g",
            "invalid_release_tag",
            drop_columns=("copyright",),
        )
        self._write_manifest("2026_07")

        worker, _statuses = self._run_failure()
        self.assertIsNone(worker.dataset_identity)

    def test_manifest_is_required_at_runtime(self):
        self._write_release("2026_07", "g", "unverified_tag")
        statuses = []
        worker = PandasSearchWorker(str(self.data_dir), ("g",), queries={})
        worker.status_update.connect(statuses.append)
        emissions = []
        worker.results_ready.connect(
            lambda rows, total: emissions.append((rows, total))
        )

        worker.run()

        self.assertEqual(emissions, [])
        self.assertTrue(any("manifest 없음" in item for item in statuses), statuses)

    def test_missing_selected_manifest_artifact_fails_closed(self):
        self._write_release("2026_07", "g", "current_release_tag")
        self._write_manifest("2026_07", artifacts=[{
            "kind": "event_graph",
            "rating_shard": "g",
        }])
        statuses = []
        worker = PandasSearchWorker(str(self.data_dir), ("g",), queries={})
        worker.status_update.connect(statuses.append)
        emissions = []
        worker.results_ready.connect(
            lambda rows, total: emissions.append((rows, total))
        )

        worker.run()

        self.assertEqual(emissions, [])
        self.assertTrue(any("정확히 1개" in item for item in statuses), statuses)

    def test_selected_manifest_artifact_path_must_match_active_release(self):
        self._write_release("2026_07", "g", "current_release_tag")
        artifact = self._artifact(
            "2026_07", "g", path="danbooru_2026_06_g.parquet"
        )
        self._write_manifest("2026_07", artifacts=[artifact])
        statuses = []
        worker = PandasSearchWorker(str(self.data_dir), ("g",), queries={})
        worker.status_update.connect(statuses.append)
        emissions = []
        worker.results_ready.connect(
            lambda rows, total: emissions.append((rows, total))
        )

        worker.run()

        self.assertEqual(emissions, [])
        self.assertTrue(any("경로 불일치" in item for item in statuses), statuses)

    def test_manifest_size_hash_and_row_mismatches_fail_closed(self):
        self._write_release("2026_07", "g", "verified_release_tag")
        valid_artifact = self._artifact("2026_07", "g")
        cases = {
            "size": {"size_bytes": valid_artifact["size_bytes"] + 1},
            "hash": {"sha256": "0" * 64},
            "rows": {"rows": valid_artifact["rows"] + 1},
        }

        for label, override in cases.items():
            with self.subTest(label=label):
                self._reset_cache()
                artifact = dict(valid_artifact)
                artifact.update(override)
                self._write_manifest("2026_07", artifacts=[artifact])
                statuses = []
                worker = PandasSearchWorker(
                    str(self.data_dir), ("g",), queries={}
                )
                worker.status_update.connect(statuses.append)
                emissions = []
                worker.results_ready.connect(
                    lambda rows, total: emissions.append((rows, total))
                )

                worker.run()

                self.assertEqual(emissions, [])
                self.assertTrue(
                    any("불일치" in item for item in statuses),
                    statuses,
                )

    def test_manifest_change_during_artifact_validation_fails_closed(self):
        self._write_release("2026_07", "g", "verified_release_tag")
        self._write_manifest("2026_07")
        manifest_path = self.data_dir / "dataset_manifest.json"
        statuses = []
        emissions = []
        worker = PandasSearchWorker(str(self.data_dir), ("g",), queries={})
        worker.status_update.connect(statuses.append)
        worker.results_ready.connect(
            lambda rows, total: emissions.append((rows, total))
        )
        original_sha256 = worker._sha256

        def mutate_manifest_after_hash(path):
            result = original_sha256(path)
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["changed_during_validation"] = True
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            return result

        worker._sha256 = mutate_manifest_after_hash

        worker.run()

        self.assertEqual(emissions, [])
        self.assertIsNone(worker.dataset_identity)
        self.assertTrue(
            any(item.startswith("❌") and "변경" in item for item in statuses),
            statuses,
        )

    def test_previously_verified_shard_replaced_by_later_validation_is_rejected(self):
        self._write_release("2026_07", "g", "verified_g")
        self._write_release("2026_07", "s", "verified_s")
        self._write_manifest("2026_07", ("g", "s"))
        statuses = []
        emissions = []
        worker = PandasSearchWorker(
            str(self.data_dir), ("g", "s"), queries={}
        )
        worker.status_update.connect(statuses.append)
        worker.results_ready.connect(
            lambda rows, total: emissions.append((rows, total))
        )
        original_validate = worker._validate_manifest_artifact

        def replace_g_after_s_validation(artifact, **kwargs):
            result = original_validate(artifact, **kwargs)
            if kwargs["rating"] == "s":
                g_path = self.data_dir / "danbooru_2026_07_g.parquet"
                replacement = self.data_dir / "replacement_g.parquet"
                frame = pd.read_parquet(g_path)
                frame = pd.concat([frame, frame], ignore_index=True)
                frame.loc[0, "general"] = "unverified_1"
                frame.loc[1, "general"] = "unverified_2"
                frame.to_parquet(replacement, index=False)
                os.replace(replacement, g_path)
            return result

        worker._validate_manifest_artifact = replace_g_after_s_validation

        worker.run()

        self.assertEqual(emissions, [])
        self.assertIsNone(worker.dataset_identity)
        self.assertTrue(
            any(item.startswith("❌") and "변경" in item for item in statuses),
            statuses,
        )

    # ── 워커가 내보내는 행 계약 — 정규화는 워커 스레드에서 끝난다 ──

    def test_results_are_normalised_rows_on_an_object_signal(self):
        from core.search_rows import SEARCH_ROW_KEYS, NormalizedSearchRows

        self._write_release(
            "2026_07", "g", "1girl solo",
            image_width=0, meta="highres",
        )
        self._write_manifest("2026_07")

        _worker, (rows, total) = self._run()

        self.assertIsInstance(rows, NormalizedSearchRows)
        self.assertEqual(total, 1)
        self.assertEqual(tuple(rows[0]), SEARCH_ROW_KEYS)
        self.assertEqual(rows[0]["general"], "1girl solo")
        self.assertIsNone(rows[0]["image_width"])     # 0 → 자동 해상도 폴백
        self.assertEqual(rows[0]["image_height"], 1024)
        self.assertNotIn("meta", rows[0])
        # list(QVariantList) 시그니처면 emit 마다 행 전체가 QVariant 로 왕복 복사된다
        meta = PandasSearchWorker.staticMetaObject
        signatures = {
            bytes(meta.method(i).methodSignature()).decode()
            for i in range(meta.methodCount())
        }
        self.assertIn("results_ready(PyQt_PyObject,int)", signatures)
        self.assertNotIn("results_ready(QVariantList,int)", signatures)

    def test_empty_release_emits_an_empty_normalised_list(self):
        from core.search_rows import NormalizedSearchRows

        path = self.data_dir / "danbooru_2026_07_g.parquet"
        frame = pd.DataFrame([{
            "rating": "g", "general": "x", "character": "", "copyright": "",
            "artist": "", "meta": "", "image_width": 1, "image_height": 1,
        }]).iloc[0:0]
        frame.to_parquet(path, index=False)
        self._write_manifest("2026_07")

        _worker, (rows, total) = self._run()

        self.assertIsInstance(rows, NormalizedSearchRows)
        self.assertEqual((rows, total), ([], 0))

    def test_queries_match_the_loaded_column_without_a_lowercase_copy(self):
        self._write_release("2026_07", "g", "long_hair smile")
        self._write_manifest("2026_07")
        emissions = []
        worker = PandasSearchWorker(
            str(self.data_dir), ("g",), queries={"general": "Long Hair"}
        )
        worker.results_ready.connect(lambda rows, total: emissions.append((rows, total)))

        worker.run()

        self.assertEqual(emissions[0][1], 1)
        self.assertFalse(hasattr(PandasSearchWorker, "cached_col_lower"))
        self.assertFalse(hasattr(PandasSearchWorker, "_get_col_lower"))

    def test_result_cap_is_the_shared_constant(self):
        from core.search_rows import SEARCH_RESULT_CAP

        self.assertEqual(PandasSearchWorker.DEFAULT_RESULT_CAP, SEARCH_RESULT_CAP)


if __name__ == "__main__":
    unittest.main()
