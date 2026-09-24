"""core.fetch_data / core.dataset_artifacts — manifest 기준 누락·구버전 shard 받기."""
from __future__ import annotations

import io
import json
import os
import subprocess
import sys
import tempfile
import textwrap
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

from core import fetch_data
from core.dataset_artifacts import (
    artifact_relpath,
    manifest_revision,
    read_manifest,
    stale_among,
    stale_artifacts,
)


def _write(path: Path, size: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"x" * size)


def _manifest(root: Path, artifacts, **extra) -> dict:
    manifest = {"format_version": 1, "dataset_label": "2026_08", "artifacts": artifacts, **extra}
    (root / "dataset_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return manifest


ARTIFACTS = [
    {"path": "danbooru_2026_08_g.parquet", "size_bytes": 3, "kind": "search"},
    {"path": "danbooru_sorted/danbooru_g.parquet", "size_bytes": 5, "kind": "event_graph"},
]


class DatasetArtifactTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)

    def tearDown(self):
        self._tmp.cleanup()

    def test_missing_and_size_mismatched_artifacts_are_stale(self):
        manifest = _manifest(self.root, ARTIFACTS)
        self.assertEqual(
            stale_artifacts(manifest, self.root),
            ["danbooru_2026_08_g.parquet", "danbooru_sorted/danbooru_g.parquet"],
        )
        _write(self.root / "danbooru_2026_08_g.parquet", 3)
        _write(self.root / "danbooru_sorted" / "danbooru_g.parquet", 4)   # 옛 릴리스 크기
        self.assertEqual(stale_artifacts(manifest, self.root), ["danbooru_sorted/danbooru_g.parquet"])
        self.assertEqual(stale_artifacts(manifest, self.root, kind="search"), [])
        _write(self.root / "danbooru_sorted" / "danbooru_g.parquet", 5)
        self.assertEqual(stale_artifacts(manifest, self.root), [])

    def test_unsafe_paths_are_never_targets(self):
        for raw in ("../evil.parquet", "/abs.parquet", "C:\\x.parquet", "a/../../b", "", None, 3):
            self.assertIsNone(artifact_relpath({"path": raw}), raw)
        self.assertEqual(artifact_relpath({"path": "sub\\file.parquet"}), "sub/file.parquet")
        manifest = _manifest(self.root, [{"path": "../evil.parquet", "size_bytes": 1}])
        self.assertEqual(stale_artifacts(manifest, self.root), [])

    def test_manifest_reading_and_revision(self):
        self.assertIsNone(read_manifest(self.root))
        (self.root / "dataset_manifest.json").write_text("{broken", encoding="utf-8")
        self.assertIsNone(read_manifest(self.root))
        manifest = _manifest(self.root, ARTIFACTS, hf_revision=" abc123 ")
        self.assertEqual(read_manifest(self.root), manifest)
        self.assertEqual(manifest_revision(manifest), "abc123")
        self.assertIsNone(manifest_revision({"artifacts": []}))

    def test_stale_among_only_judges_listed_files(self):
        manifest = _manifest(self.root, ARTIFACTS)
        loaded = self.root / "danbooru_sorted" / "danbooru_g.parquet"
        _write(loaded, 9)
        unknown = self.root / "danbooru_sorted" / "danbooru_x.parquet"
        _write(unknown, 1)
        self.assertEqual(
            stale_among(manifest, self.root, [loaded, unknown], kind="event_graph"),
            ["danbooru_sorted/danbooru_g.parquet"],
        )


class EnsureDataTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name) / "danbooru_optimized"
        self.root.mkdir()
        self._dir_patch = patch.object(fetch_data, "DATA_DIR", self.root)
        self._dir_patch.start()

    def tearDown(self):
        self._dir_patch.stop()
        self._tmp.cleanup()

    def _run(self, download):
        with patch.object(fetch_data, "_download", side_effect=download) as mocked, \
                redirect_stdout(io.StringIO()):
            code = fetch_data.ensure_data()
        return code, mocked

    def _complete_dataset(self):
        _write(self.root / "danbooru_2026_08_g.parquet", 3)
        _write(self.root / "danbooru_sorted" / "danbooru_g.parquet", 5)
        _write(self.root / "tags_dictionary.parquet", 1)

    def test_up_to_date_data_is_not_downloaded(self):
        _manifest(self.root, ARTIFACTS)
        self._complete_dataset()
        code, mocked = self._run(lambda *a: True)
        self.assertEqual(code, 0)
        mocked.assert_not_called()

    def test_manifest_update_downloads_only_the_stale_paths_at_the_pinned_revision(self):
        _manifest(self.root, ARTIFACTS, hf_revision="rev1")
        self._complete_dataset()
        _write(self.root / "danbooru_sorted" / "danbooru_g.parquet", 4)   # 옛 Event shard

        def download(patterns, revision=None):
            _write(self.root / "danbooru_sorted" / "danbooru_g.parquet", 5)
            return True

        code, mocked = self._run(download)
        self.assertEqual(code, 0)
        mocked.assert_called_once_with(["danbooru_sorted/danbooru_g.parquet"], "rev1")

    def test_missing_tag_dictionary_is_fetched_with_stale_shards(self):
        _manifest(self.root, ARTIFACTS)
        self._complete_dataset()
        (self.root / "tags_dictionary.parquet").unlink()
        code, mocked = self._run(lambda *a: True)
        self.assertEqual(code, 0)
        mocked.assert_called_once_with(["tags_dictionary.parquet"], None)

    def test_failed_refresh_keeps_the_app_starting(self):
        _manifest(self.root, ARTIFACTS)
        self._complete_dataset()
        _write(self.root / "danbooru_2026_08_g.parquet", 2)
        code, _mocked = self._run(lambda *a: False)
        self.assertEqual(code, 0, "쓰던 데이터가 있으면 기동을 막지 않는다")

    def test_fresh_install_with_manifest_downloads_exact_paths_and_fails_hard(self):
        _manifest(self.root, ARTIFACTS)
        code, mocked = self._run(lambda *a: False)
        self.assertEqual(code, 1, "데이터가 하나도 없는 첫 설치 실패는 예전처럼 중단")
        mocked.assert_called_once_with(
            ["danbooru_2026_08_g.parquet", "danbooru_sorted/danbooru_g.parquet",
             "tags_dictionary.parquet"],
            None,
        )

    def test_without_manifest_legacy_rule_applies(self):
        code, mocked = self._run(lambda *a: True)
        self.assertEqual(code, 0)
        mocked.assert_called_once_with(fetch_data.PATTERNS)

        _write(self.root / "something.parquet", 1)
        code, mocked = self._run(lambda *a: True)
        self.assertEqual(code, 0)
        mocked.assert_not_called()

        (self.root / "something.parquet").unlink()
        code, _mocked = self._run(lambda *a: False)
        self.assertEqual(code, 1)

    def test_older_release_files_are_not_a_fresh_install_when_the_refresh_fails(self):
        # 현재 manifest 경로는 하나도 없지만 쓰던 데이터(사전 + 옛 릴리스 이름 shard)가 있다 —
        # 오프라인 등으로 받기에 실패해도 기동을 막지 않는다 (예전 규칙도 이 경우 앱을 띄웠다)
        _manifest(self.root, ARTIFACTS)
        _write(self.root / "tags_dictionary.parquet", 1)
        _write(self.root / "danbooru_2026_07_g.parquet", 2)
        code, mocked = self._run(lambda *a: False)
        self.assertEqual(code, 0)
        mocked.assert_called_once_with(
            ["danbooru_2026_08_g.parquet", "danbooru_sorted/danbooru_g.parquet"], None,
        )

    def test_only_the_tag_dictionary_present_is_not_a_fresh_install(self):
        _manifest(self.root, ARTIFACTS)
        _write(self.root / "tags_dictionary.parquet", 1)
        code, _mocked = self._run(lambda *a: False)
        self.assertEqual(code, 0)

    def test_partial_fresh_download_that_fails_still_stops_startup(self):
        # 첫 설치 판정은 받기 **전** 상태로 한다 — 받다 끊겨 일부만 생겨도 첫 설치 실패다
        _manifest(self.root, ARTIFACTS)

        def download(patterns, revision=None):
            _write(self.root / "danbooru_2026_08_g.parquet", 3)
            return False

        code, _mocked = self._run(download)
        self.assertEqual(code, 1)


class ScriptEntryTests(unittest.TestCase):
    def test_running_the_file_directly_can_import_the_core_package(self):
        # README 의 수동 복구 `python core\fetch_data.py` 는 sys.path[0] 이 core/ 다 — 그 상태를
        # 새 인터프리터에서 재현한다. run_name 을 __main__ 이 아닌 값으로 두어 ensure_data()
        # (= 실제 다운로드)는 돌리지 않고 모듈 최상위 import 만 검증한다.
        script = Path(fetch_data.__file__).resolve()
        core_dir = script.parent
        child = textwrap.dedent(f"""
            import importlib.util, os, runpy, sys
            root = os.path.normcase({str(core_dir.parent)!r})
            sys.path[:] = [p for p in sys.path
                           if os.path.normcase(os.path.abspath(p or os.getcwd())) != root]
            sys.path.insert(0, {str(core_dir)!r})
            print("core importable before:", importlib.util.find_spec("core") is not None)
            ns = runpy.run_path({str(script)!r}, run_name="fetch_data_as_script")
            print("ensure_data callable:", callable(ns["ensure_data"]))
            print("artifacts module:", ns["stale_artifacts"].__module__)
        """)
        env = {key: value for key, value in os.environ.items() if key != "PYTHONPATH"}
        env["PYTHONIOENCODING"] = "utf-8"
        proc = subprocess.run(
            [sys.executable, "-c", child], cwd=str(core_dir), env=env,
            capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=60,
        )
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertIn("core importable before: False", proc.stdout, "재현 전제: 프로젝트 루트가 경로에 없다")
        self.assertIn("ensure_data callable: True", proc.stdout)
        self.assertIn("artifacts module: core.dataset_artifacts", proc.stdout)

    def test_readme_manual_recovery_uses_module_form(self):
        readme = (Path(fetch_data.PROJECT_ROOT) / "README.md").read_text(encoding="utf-8")
        self.assertIn("python -m core.fetch_data", readme)
        self.assertNotIn("python core\\fetch_data.py", readme)


if __name__ == "__main__":
    unittest.main()
