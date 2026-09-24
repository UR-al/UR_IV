"""Dataset-aware persistence for Search tab result caches."""

from __future__ import annotations

import hashlib
import json
import re
import threading
import uuid
from pathlib import Path

from core.storage_paths import PROJECT_ROOT, StoragePaths
from utils.atomic_json import atomic_write_json, load_json_safe


SCHEMA_VERSION = 2
_SAFE_DATASET_LABEL = re.compile(r"\A[A-Za-z0-9][A-Za-z0-9._-]*\Z")
_SAFE_SNAPSHOT_ID = re.compile(r"\A[A-Fa-f0-9]{32}\Z")
_STORE_LOCK = threading.RLock()
_WRITE_SEQUENCE_LOCK = threading.Lock()
_LATEST_WRITE_SEQUENCE = 0
_LATEST_WRITE_SNAPSHOT: str | None = None


class DatasetManifestError(ValueError):
    """Raised when the active dataset manifest cannot identify a safe label."""


def reserve_write_sequence(snapshot_id: str | None = None) -> int:
    """Reserve a monotonic persistence order before starting a writer thread."""
    global _LATEST_WRITE_SEQUENCE, _LATEST_WRITE_SNAPSHOT
    with _WRITE_SEQUENCE_LOCK:
        _LATEST_WRITE_SEQUENCE += 1
        _LATEST_WRITE_SNAPSHOT = snapshot_id
        return _LATEST_WRITE_SEQUENCE


def _latest_write_request() -> tuple[int, str | None]:
    with _WRITE_SEQUENCE_LOCK:
        return _LATEST_WRITE_SEQUENCE, _LATEST_WRITE_SNAPSHOT


class SearchResultStore:
    """Read and write the active and unfiltered Search result sets."""

    def __init__(
        self,
        project_root: str | Path = PROJECT_ROOT,
        *,
        storage: StoragePaths | None = None,
        dataset_root: str | Path | None = None,
    ) -> None:
        self.storage = storage or StoragePaths(project_root)
        self.project_root = self.storage.project_root
        if dataset_root is None:
            # 앱의 데이터셋 경로는 core.fetch_data.DATA_DIR 한 곳(= config.PARQUET_DIR)이다. 다른 루트
            # (테스트의 임시 프로젝트)는 같은 상대 위치를 쓴다. 가벼운 fetch_data 를 직접 읽는다 —
            # config 는 PyQt·QtWebEngine 을 끌어온다.
            from core.fetch_data import DATA_DIR
            dataset_root = DATA_DIR if self.project_root == PROJECT_ROOT else self.project_root / DATA_DIR.name
        self.dataset_root = Path(dataset_root).expanduser().resolve(strict=False)
        self.last_error: str | None = None
        self.last_snapshot_id: str | None = None
        self.last_dataset_identity: dict[str, str] | None = None
        with _STORE_LOCK:
            self.active_path = self.storage.cache_file(
                "search/last_search_results.json",
                legacy_paths=self.project_root / "config/last_search_results.json",
            )
            self.full_path = self.storage.cache_file(
                "search/last_full_results.json",
                legacy_paths=self.project_root / "config/last_full_results.json",
            )

    def save(
        self,
        active: list,
        *,
        full: list | None = None,
        snapshot_id: str | None = None,
        expected_identity: dict[str, str] | None = None,
    ) -> None:
        self._require_results_list(active)
        if full is not None:
            self._require_results_list(full)
        with _STORE_LOCK:
            self._save_locked(
                active,
                full=full,
                snapshot_id=snapshot_id,
                expected_identity=expected_identity,
            )

    def save_if_latest(
        self,
        sequence: int,
        active: list,
        *,
        full: list | None = None,
        snapshot_id: str | None = None,
        expected_identity: dict[str, str] | None = None,
    ) -> bool:
        """Persist only if no newer GUI request was queued in the meantime."""
        if type(sequence) is not int or sequence < 1:
            raise ValueError("search result write sequence must be a positive integer")
        self._require_results_list(active)
        if full is not None:
            self._require_results_list(full)
        with _STORE_LOCK:
            latest_sequence, latest_snapshot = _latest_write_request()
            if sequence != latest_sequence:
                # A newer active-only filter update from the same Search
                # snapshot still depends on this request's full base. Publish
                # only that base, never the older active view.
                if (
                    full is None
                    or snapshot_id is None
                    or snapshot_id != latest_snapshot
                ):
                    return False
                dataset_label, dataset_fingerprint = self._validated_save_identity(
                    expected_identity
                )
                self._require_snapshot_id(snapshot_id)
                self._save(
                    self.full_path,
                    full,
                    dataset_label,
                    dataset_fingerprint,
                    snapshot_id,
                )
                self.last_error = None
                self.last_snapshot_id = snapshot_id
                self.last_dataset_identity = {
                    "label": dataset_label,
                    "fingerprint": dataset_fingerprint,
                }
                return True
            self._save_locked(
                active,
                full=full,
                snapshot_id=snapshot_id,
                expected_identity=expected_identity,
            )
            return True

    def save_active(self, results: list) -> None:
        self._require_results_list(results)
        with _STORE_LOCK:
            label, fingerprint = self._dataset_identity()
            snapshot_id = self._current_snapshot_id(label, fingerprint)
            self._save(
                self.active_path,
                results,
                label,
                fingerprint,
                snapshot_id,
            )

    def save_full(self, results: list) -> None:
        self._require_results_list(results)
        with _STORE_LOCK:
            label, fingerprint = self._dataset_identity()
            snapshot_id = self._current_snapshot_id(label, fingerprint)
            self._save(
                self.full_path,
                results,
                label,
                fingerprint,
                snapshot_id,
            )

    def load_active(self) -> list:
        with _STORE_LOCK:
            return self._load(self.active_path)

    def load_full(self, *, expected_snapshot_id: str | None = None) -> list:
        """Load the unfiltered base of a Search snapshot.

        Without ``expected_snapshot_id`` the active envelope is read as well to
        prove both files belong to one snapshot.  A caller that already holds
        the active snapshot id (from a validated active load or its own save)
        passes it instead, so the 20MB+ active file is not parsed again just to
        compare one field.  The pairing guarantee is the same: the full base is
        returned only when its snapshot id equals the expected one.
        """
        with _STORE_LOCK:
            if expected_snapshot_id is None:
                return self._load_full()
            return self._load_full_for_snapshot(expected_snapshot_id)

    def dataset_info(self) -> dict[str, str]:
        """Return the active manifest identity used to validate cache files."""
        with _STORE_LOCK:
            label, fingerprint = self._dataset_identity()
            return {
                "label": label,
                "fingerprint": fingerprint,
            }

    def _save(
        self,
        path: Path,
        results: list,
        dataset_label: str,
        dataset_fingerprint: str,
        snapshot_id: str,
    ) -> None:
        envelope = {
            "schema_version": SCHEMA_VERSION,
            "dataset_label": dataset_label,
            "dataset_fingerprint": dataset_fingerprint,
            "snapshot_id": snapshot_id,
            "results": results,
        }
        atomic_write_json(str(path), envelope, indent=None)

    def _save_locked(
        self,
        active: list,
        *,
        full: list | None,
        snapshot_id: str | None,
        expected_identity: dict[str, str] | None,
    ) -> None:
        dataset_label, dataset_fingerprint = self._validated_save_identity(
            expected_identity
        )
        if full is None:
            if snapshot_id is None:
                snapshot_id = self._current_snapshot_id(
                    dataset_label,
                    dataset_fingerprint,
                )
            self._require_snapshot_id(snapshot_id)
            self._save(
                self.active_path,
                active,
                dataset_label,
                dataset_fingerprint,
                snapshot_id,
            )
        else:
            snapshot_id = snapshot_id or uuid.uuid4().hex
            self._require_snapshot_id(snapshot_id)
            # Publish the full base first and the active view second.  A crash
            # between them leaves different snapshot ids; load_full then
            # rejects the pair instead of combining two searches.
            self._save(
                self.full_path,
                full,
                dataset_label,
                dataset_fingerprint,
                snapshot_id,
            )
            self._save(
                self.active_path,
                active,
                dataset_label,
                dataset_fingerprint,
                snapshot_id,
            )
        self.last_error = None
        self.last_snapshot_id = snapshot_id
        self.last_dataset_identity = {
            "label": dataset_label,
            "fingerprint": dataset_fingerprint,
        }

    @staticmethod
    def _require_results_list(results: object) -> None:
        if not isinstance(results, list):
            raise TypeError("search results must be a list")
        if any(not isinstance(row, dict) for row in results):
            raise TypeError("each search result must be an object")

    def _load(self, path: Path) -> list:
        if not path.is_file():
            self.last_error = None
            self.last_snapshot_id = None
            self.last_dataset_identity = None
            return []
        try:
            current_label, current_fingerprint = self._dataset_identity()
        except DatasetManifestError as exc:
            self.last_error = str(exc)
            self.last_snapshot_id = None
            self.last_dataset_identity = None
            return []
        envelope, error = self._read_envelope(
            path,
            current_label,
            current_fingerprint,
        )
        if envelope is None:
            self.last_error = error
            self.last_snapshot_id = None
            self.last_dataset_identity = None
            return []
        self.last_error = None
        self.last_snapshot_id = envelope["snapshot_id"]
        self.last_dataset_identity = {
            "label": current_label,
            "fingerprint": current_fingerprint,
        }
        return envelope["results"]

    def _load_full(self) -> list:
        if not self.full_path.is_file():
            self.last_error = None
            self.last_snapshot_id = None
            self.last_dataset_identity = None
            return []
        if not self.active_path.is_file():
            self.last_error = "full search cache has no matching active snapshot"
            self.last_snapshot_id = None
            self.last_dataset_identity = None
            return []
        try:
            current_label, current_fingerprint = self._dataset_identity()
        except DatasetManifestError as exc:
            self.last_error = str(exc)
            self.last_snapshot_id = None
            self.last_dataset_identity = None
            return []
        full, error = self._read_envelope(
            self.full_path,
            current_label,
            current_fingerprint,
        )
        if full is None:
            self.last_error = error
            self.last_snapshot_id = None
            self.last_dataset_identity = None
            return []
        active, active_error = self._read_envelope(
            self.active_path,
            current_label,
            current_fingerprint,
        )
        if active is None:
            self.last_error = (
                "active search cache could not validate the full snapshot: "
                f"{active_error}"
            )
            self.last_snapshot_id = None
            self.last_dataset_identity = None
            return []
        if active["snapshot_id"] != full["snapshot_id"]:
            self.last_error = (
                "active and full search caches belong to different snapshots"
            )
            self.last_snapshot_id = None
            self.last_dataset_identity = None
            return []
        self.last_error = None
        self.last_snapshot_id = full["snapshot_id"]
        self.last_dataset_identity = {
            "label": current_label,
            "fingerprint": current_fingerprint,
        }
        return full["results"]

    def _load_full_for_snapshot(self, expected_snapshot_id: str) -> list:
        def _reject(error: str | None) -> list:
            self.last_error = error
            self.last_snapshot_id = None
            self.last_dataset_identity = None
            return []

        if not isinstance(expected_snapshot_id, str) or not _SAFE_SNAPSHOT_ID.fullmatch(
            expected_snapshot_id
        ):
            return _reject("expected search snapshot id is invalid")
        if not self.full_path.is_file():
            return _reject(None)
        try:
            current_label, current_fingerprint = self._dataset_identity()
        except DatasetManifestError as exc:
            return _reject(str(exc))
        full, error = self._read_envelope(
            self.full_path,
            current_label,
            current_fingerprint,
        )
        if full is None:
            return _reject(error)
        if full["snapshot_id"].lower() != expected_snapshot_id.lower():
            return _reject(
                "full search cache belongs to a different snapshot than the active view"
            )
        self.last_error = None
        self.last_snapshot_id = full["snapshot_id"]
        self.last_dataset_identity = {
            "label": current_label,
            "fingerprint": current_fingerprint,
        }
        return full["results"]

    @staticmethod
    def _read_envelope(
        path: Path,
        current_label: str,
        current_fingerprint: str,
    ) -> tuple[dict | None, str | None]:
        envelope = load_json_safe(str(path), None)
        if isinstance(envelope, list):
            return None, (
                "legacy search result list has no dataset provenance and was rejected: "
                f"{path}"
            )
        if not isinstance(envelope, dict):
            return None, f"invalid search result envelope: {path}"
        schema_version = envelope.get("schema_version")
        if type(schema_version) is not int or schema_version != SCHEMA_VERSION:
            return None, f"unsupported search result schema: {path}"
        if envelope.get("dataset_label") != current_label:
            return None, (
                "search result dataset label does not match the active dataset: "
                f"{path}"
            )
        if envelope.get("dataset_fingerprint") != current_fingerprint:
            return None, (
                "search result dataset fingerprint does not match the active dataset: "
                f"{path}"
            )
        snapshot_id = envelope.get("snapshot_id")
        if not isinstance(snapshot_id, str) or not _SAFE_SNAPSHOT_ID.fullmatch(
            snapshot_id
        ):
            return None, f"invalid search result snapshot id: {path}"
        results = envelope.get("results")
        if not isinstance(results, list) or any(
            not isinstance(row, dict) for row in results
        ):
            return None, f"search result payload is not a list of objects: {path}"
        return envelope, None

    def _current_snapshot_id(
        self,
        current_label: str,
        current_fingerprint: str,
    ) -> str:
        # The full base owns the snapshot lineage for active-only filter
        # updates.  Fall back to active when no full base has been saved yet.
        for path in (self.full_path, self.active_path):
            if not path.is_file():
                continue
            envelope, _error = self._read_envelope(
                path,
                current_label,
                current_fingerprint,
            )
            if envelope is not None:
                return envelope["snapshot_id"]
        return uuid.uuid4().hex

    @staticmethod
    def _require_snapshot_id(snapshot_id: str) -> None:
        if not isinstance(snapshot_id, str) or not _SAFE_SNAPSHOT_ID.fullmatch(
            snapshot_id
        ):
            raise ValueError("search result snapshot id must be 32 hexadecimal characters")

    def _validated_save_identity(
        self,
        expected_identity: dict[str, str] | None,
    ) -> tuple[str, str]:
        current = self._dataset_identity()
        if expected_identity is None:
            return current
        if not isinstance(expected_identity, dict):
            raise TypeError("expected dataset identity must be an object")
        label = expected_identity.get("label")
        fingerprint = expected_identity.get("fingerprint")
        if not isinstance(label, str) or not _SAFE_DATASET_LABEL.fullmatch(label):
            raise DatasetManifestError("expected dataset identity has an unsafe label")
        if (
            not isinstance(fingerprint, str)
            or not re.fullmatch(r"[A-Fa-f0-9]{64}", fingerprint)
        ):
            raise DatasetManifestError(
                "expected dataset identity has an invalid fingerprint"
            )
        expected = (label, fingerprint.lower())
        if current != expected:
            raise DatasetManifestError(
                "active dataset changed before search results could be persisted"
            )
        return expected

    def _dataset_identity(self) -> tuple[str, str]:
        manifest_path = self.dataset_root / "dataset_manifest.json"
        if not manifest_path.is_file():
            raise DatasetManifestError(
                f"dataset manifest is required for Search cache provenance: {manifest_path}"
            )
        try:
            raw_manifest = manifest_path.read_bytes()
            manifest = json.loads(raw_manifest.decode("utf-8-sig"))
        except (json.JSONDecodeError, UnicodeDecodeError, OSError) as exc:
            raise DatasetManifestError(
                f"dataset manifest could not be read safely: {manifest_path}: {exc}"
            ) from exc
        if not isinstance(manifest, dict):
            raise DatasetManifestError(
                f"dataset manifest root must be an object: {manifest_path}"
            )
        label = manifest.get("dataset_label")
        if not isinstance(label, str) or not _SAFE_DATASET_LABEL.fullmatch(label):
            raise DatasetManifestError(
                f"dataset manifest contains an unsafe dataset label: {manifest_path}"
            )
        return label, hashlib.sha256(raw_manifest).hexdigest()


__all__ = [
    "DatasetManifestError",
    "SCHEMA_VERSION",
    "SearchResultStore",
    "reserve_write_sequence",
]
