from __future__ import annotations

import json
import shutil
import subprocess
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from core.app_updater import (
    AppUpdateError,
    AppUpdateManager,
    GitSource,
    normalise_version,
    parse_version,
)


def _github_release(version: str, *, notes: str = "release notes") -> dict[str, Any]:
    return {
        "tag_name": f"v{version}",
        "name": f"Release {version}",
        "html_url": f"https://github.com/UR-al/UR_IV/releases/tag/v{version}",
        "published_at": "2026-09-01T00:00:00Z",
        "body": notes,
        "draft": False,
        "prerelease": False,
    }


def _cached_release(version: str) -> dict[str, str]:
    return {
        "version": version,
        "tagName": f"v{version}",
        "name": f"Release {version}",
        "url": f"https://github.com/UR-al/UR_IV/releases/tag/v{version}",
        "publishedAt": "2026-09-01T00:00:00Z",
        "notes": "cached notes",
    }


class _FakeReleaseClient:
    def __init__(self, release: dict[str, Any]):
        self.release = release
        self.calls = 0

    def fetch_latest(self) -> dict[str, Any]:
        self.calls += 1
        return dict(self.release)


class _FakeGitSource:
    def __init__(self) -> None:
        self.checkout = True
        self.head_value = "a" * 40
        self.branch_value = "main"
        self.describe_value = "v2.9.0-0-gaaaaaaaaa"
        self.remote_value = "origin"
        self.target_value = "b" * 40
        self.ancestor = True
        self.changed: set[str] = set()
        self.dirty: set[str] = set()
        self.ignored: set[str] = set()
        self.calls: list[tuple[Any, ...]] = []

    def is_checkout(self) -> bool:
        self.calls.append(("is_checkout",))
        return self.checkout

    def head(self) -> str:
        self.calls.append(("head",))
        return self.head_value

    def branch(self) -> str:
        self.calls.append(("branch",))
        return self.branch_value

    def describe(self) -> str:
        self.calls.append(("describe",))
        return self.describe_value

    def trusted_remote(self) -> str:
        self.calls.append(("trusted_remote",))
        return self.remote_value

    def fetch_release_tag(self, remote: str, tag_name: str) -> str:
        self.calls.append(("fetch_release_tag", remote, tag_name))
        return self.target_value

    def is_ancestor(self, ancestor: str, descendant: str) -> bool:
        self.calls.append(("is_ancestor", ancestor, descendant))
        return self.ancestor

    def changed_paths(self, base: str, target: str) -> set[str]:
        self.calls.append(("changed_paths", base, target))
        return set(self.changed)

    def dirty_paths(self) -> set[str]:
        self.calls.append(("dirty_paths",))
        return set(self.dirty)

    def ignored_paths(self, candidates: set[str]) -> set[str]:
        self.calls.append(("ignored_paths", tuple(sorted(candidates))))
        return set(self.ignored)


class AppUpdaterTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)
        self.root = Path(self.temp_dir.name).resolve()
        (self.root / "VERSION").write_text("2.9.0\n", encoding="utf-8")
        (self.root / "new_run_main_ui.bat").write_text("@echo off\n", encoding="utf-8")
        self.settings_path = self.root / "config" / "app_update.json"
        self.plan_dir = self.root / "cache" / "updates"
        self.result_path = self.root / "logs" / "updates" / "last_result.json"
        self.now = datetime(2026, 9, 2, 12, 0, 0, tzinfo=timezone.utc)
        self.release_client = _FakeReleaseClient(_github_release("2.10.0"))
        self.git = _FakeGitSource()
        self.launched_paths: list[Path] = []
        self.launched_plans: list[dict[str, Any]] = []

    def _launch_helper(self, plan_path: Path) -> int:
        self.launched_paths.append(plan_path)
        self.launched_plans.append(json.loads(plan_path.read_text(encoding="utf-8")))
        return 9001

    def _manager(
        self,
        *,
        process_launcher=None,
        instance_scanner=None,
        update_lock_probe=None,
    ) -> AppUpdateManager:
        return AppUpdateManager(
            project_root=self.root,
            settings_path=self.settings_path,
            plan_dir=self.plan_dir,
            result_path=self.result_path,
            release_client=self.release_client,
            git_source=self.git,
            process_launcher=process_launcher or self._launch_helper,
            clock=lambda: self.now,
            current_pid=lambda: 4242,
            instance_scanner=instance_scanner or (lambda *_args, **_kwargs: []),
            update_lock_probe=update_lock_probe or (lambda: False),
        )

    def _write_settings(self, **overrides: Any) -> None:
        values: dict[str, Any] = {
            "schema": 1,
            "autoCheck": True,
            "intervalHours": 12,
            "lastCheckedAt": "",
            "skippedVersion": "",
            "pendingVersion": "",
            "latestRelease": {},
        }
        values.update(overrides)
        self.settings_path.parent.mkdir(parents=True, exist_ok=True)
        self.settings_path.write_text(json.dumps(values), encoding="utf-8")

    def test_version_parsing_is_strict_and_comparison_is_numeric(self) -> None:
        self.assertEqual((2, 10, 0), parse_version("v2.10.0"))
        self.assertEqual("2.10.0", normalise_version("  v2.10.0  "))
        for value in ("2.10", "2.10.0-rc1", "release-2.10.0", "2.10.0.1", ""):
            with self.subTest(value=value):
                self.assertIsNone(parse_version(value))
                self.assertEqual("", normalise_version(value))

        self._write_settings(latestRelease=_cached_release("2.10.0"))
        snapshot = self._manager().snapshot()
        self.assertEqual("2.9.0", snapshot["currentVersion"])
        self.assertEqual("2.10.0", snapshot["latestVersion"])
        self.assertTrue(snapshot["updateAvailable"])

    def test_check_persists_sanitised_cache_and_snapshot_never_hits_network(self) -> None:
        self.release_client.release = _github_release("2.10.0", notes="safe\x00 notes")
        manager = self._manager()

        checked = manager.execute("check")
        cached = json.loads(self.settings_path.read_text(encoding="utf-8"))

        self.assertEqual(1, self.release_client.calls)
        self.assertEqual("2026-09-02T12:00:00Z", cached["lastCheckedAt"])
        self.assertEqual("2.10.0", cached["latestRelease"]["version"])
        self.assertEqual("safe notes", cached["latestRelease"]["notes"])
        self.assertTrue(checked["snapshot"]["updateAvailable"])

        first = manager.snapshot()
        second = self._manager().snapshot()
        self.assertEqual(1, self.release_client.calls)
        self.assertEqual("2.10.0", first["latestVersion"])
        self.assertEqual(first["latestVersion"], second["latestVersion"])
        self.assertEqual("safe notes", second["notes"])

    def test_invalid_cached_release_is_ignored_without_network(self) -> None:
        poisoned = _cached_release("2.10.0")
        poisoned["url"] = "https://example.invalid/releases/tag/v2.10.0"
        self._write_settings(latestRelease=poisoned)

        snapshot = self._manager().snapshot()

        self.assertEqual(0, self.release_client.calls)
        self.assertEqual("", snapshot["latestVersion"])
        self.assertFalse(snapshot["updateAvailable"])

    def test_auto_check_becomes_due_at_the_configured_interval(self) -> None:
        last_checked = self.now - timedelta(hours=12) + timedelta(seconds=1)
        self._write_settings(lastCheckedAt=last_checked.isoformat().replace("+00:00", "Z"))
        manager = self._manager()

        self.assertFalse(manager.snapshot()["shouldAutoCheck"])

        self.now += timedelta(seconds=1)
        self.assertTrue(manager.snapshot()["shouldAutoCheck"])

        self._write_settings(
            autoCheck=False,
            lastCheckedAt=(self.now - timedelta(days=7)).isoformat().replace("+00:00", "Z"),
        )
        self.assertFalse(manager.snapshot()["shouldAutoCheck"])

    def test_missing_or_malformed_check_time_is_due(self) -> None:
        manager = self._manager()
        self.assertTrue(manager.snapshot()["shouldAutoCheck"])

        self._write_settings(lastCheckedAt="not-a-date")
        self.assertTrue(manager.snapshot()["shouldAutoCheck"])

    def test_git_describe_marks_commits_ahead_of_tag_as_development_build(self) -> None:
        self.git.describe_value = "v2.6.0-41-gabcdef123"
        self._write_settings(latestRelease=_cached_release("2.6.0"))

        snapshot = self._manager().snapshot()

        self.assertEqual("2.6.0", snapshot["currentVersion"])
        self.assertEqual("v2.6.0 + 41개 변경 (aaaaaaaaa)", snapshot["currentDisplay"])
        self.assertEqual("aaaaaaaaa", snapshot["currentRevision"])
        self.assertTrue(snapshot["developmentBuild"])
        self.assertFalse(snapshot["updateAvailable"])

    def test_dirty_tag_and_unknown_git_identity_are_never_presented_as_clean_release(self) -> None:
        self.git.describe_value = "v2.9.0-0-gaaaaaaaaa-dirty"
        self._write_settings(latestRelease=_cached_release("2.10.0"))
        dirty = self._manager().snapshot()
        self.assertTrue(dirty["developmentBuild"])
        self.assertTrue(dirty["identityKnown"])
        self.assertIn("로컬 수정", dirty["currentDisplay"])

        self.git.describe_value = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
        unknown = self._manager().snapshot()
        self.assertTrue(unknown["developmentBuild"])
        self.assertFalse(unknown["identityKnown"])
        self.assertFalse(unknown["canInstall"])
        self.assertIn("기준 릴리스", unknown["installReason"])

    def test_install_preflight_creates_a_constrained_plan_for_non_overlapping_changes(self) -> None:
        self.git.changed = {"core/app_updater.py"}
        self.git.dirty = {"config/gallery_last_folder.txt"}
        manager = self._manager()

        result = manager.execute("install")

        self.assertTrue(result["restartRequired"])
        self.assertEqual(9001, result["helperPid"])
        self.assertEqual("2.10.0", result["targetVersion"])
        self.assertEqual(1, len(self.launched_plans))
        plan = self.launched_plans[0]
        ready_path = Path(str(plan.pop("readyPath")))
        self.assertEqual(self.plan_dir.resolve(), ready_path.parent)
        self.assertTrue(ready_path.name.startswith("ready-"))
        self.assertEqual(".json", ready_path.suffix)
        self.assertEqual(
            {
                "schema": 1,
                "projectRoot": str(self.root),
                "expectedHead": "a" * 40,
                "targetCommit": "b" * 40,
                "tagName": "v2.10.0",
                "remoteName": "origin",
                "branch": "main",
                "launcher": str((self.root / "new_run_main_ui.bat").resolve()),
                "parentPid": 4242,
                "resultPath": str(self.result_path.resolve()),
            },
            plan,
        )
        self.assertTrue(self.launched_paths[0].is_file())
        saved = json.loads(self.settings_path.read_text(encoding="utf-8"))
        self.assertEqual("2.10.0", saved["pendingVersion"])
        self.assertIn(("fetch_release_tag", "origin", "v2.10.0"), self.git.calls)

    def test_install_rejects_diverged_checkout_before_writing_plan(self) -> None:
        self.git.ancestor = False

        with self.assertRaises(AppUpdateError) as caught:
            self._manager().execute("install")

        self.assertEqual("UPDATE_DIVERGED", caught.exception.code)
        self.assertEqual([], self.launched_paths)
        self.assertFalse(self.plan_dir.exists())

    def test_install_rejects_only_dirty_paths_that_overlap_the_update(self) -> None:
        self.git.changed = {"core/app_updater.py", "frontend/src/App.vue"}
        self.git.dirty = {"frontend/src/App.vue", "config/user.json"}

        with self.assertRaises(AppUpdateError) as caught:
            self._manager().execute("install")

        self.assertEqual("UPDATE_LOCAL_CHANGES", caught.exception.code)
        self.assertEqual({"paths": ["frontend/src/App.vue"]}, caught.exception.details)
        self.assertEqual([], self.launched_paths)
        self.assertFalse(self.plan_dir.exists())

    def test_install_rejects_ignored_untracked_release_destination(self) -> None:
        self.git.changed = {"config/새_기본값.json"}
        self.git.ignored = {"CONFIG/새_기본값.json"}

        with self.assertRaises(AppUpdateError) as caught:
            self._manager().execute("install")

        self.assertEqual("UPDATE_LOCAL_CHANGES", caught.exception.code)
        self.assertEqual({"paths": ["config/새_기본값.json"]}, caught.exception.details)
        self.assertEqual([], self.launched_paths)

    def test_install_rejects_detached_branch_before_fetch_or_launch(self) -> None:
        self.git.branch_value = ""

        with self.assertRaises(AppUpdateError) as caught:
            self._manager().execute("install")

        self.assertEqual("UPDATE_UNAVAILABLE", caught.exception.code)
        self.assertFalse(any(call[0] == "fetch_release_tag" for call in self.git.calls))
        self.assertEqual([], self.launched_paths)

    def test_install_lock_collision_is_reported_as_busy(self) -> None:
        with self.assertRaises(AppUpdateError) as caught:
            self._manager(update_lock_probe=lambda: True).execute("install")

        self.assertEqual("OPERATION_BUSY", caught.exception.code)
        self.assertTrue(caught.exception.retryable)
        self.assertEqual([], self.launched_paths)

    def test_install_rejects_another_live_app_instance(self) -> None:
        with self.assertRaises(AppUpdateError) as caught:
            self._manager(
                instance_scanner=lambda *_args, **_kwargs: [111, 222],
            ).execute("install")

        self.assertEqual("UPDATE_OTHER_INSTANCE", caught.exception.code)
        self.assertEqual({"count": 2}, caught.exception.details)
        self.assertEqual([], self.launched_paths)

    def test_snapshot_exposes_only_safe_last_update_receipt(self) -> None:
        self.result_path.parent.mkdir(parents=True)
        self.result_path.write_text(json.dumps({
            "schema": 1,
            "ok": False,
            "message": "simulated failure",
            "tagName": "v2.10.0",
            "finishedAt": "2026-09-02T12:01:00Z",
            "unexpectedPath": "C:/private/file",
        }), encoding="utf-8")

        result = self._manager().snapshot()["lastResult"]

        self.assertEqual({
            "ok": False,
            "message": "simulated failure",
            "tagName": "v2.10.0",
            "finishedAt": "2026-09-02T12:01:00Z",
        }, result)

    def test_helper_launch_failure_removes_plan(self) -> None:
        attempted: list[Path] = []

        def fail_to_launch(plan_path: Path) -> int:
            attempted.append(plan_path)
            raise OSError("injected process failure")

        with self.assertRaises(AppUpdateError) as caught:
            self._manager(process_launcher=fail_to_launch).execute("install")

        self.assertEqual("UPDATE_LAUNCH_FAILED", caught.exception.code)
        self.assertTrue(caught.exception.retryable)
        self.assertEqual(1, len(attempted))
        self.assertFalse(attempted[0].exists())

    @unittest.skipUnless(shutil.which("git"), "Git is required for the Unicode path regression")
    def test_git_source_preserves_ignored_unicode_paths_with_nul_output(self) -> None:
        repo = self.root / "unicode-repo"
        repo.mkdir()
        subprocess.run(["git", "init", "--quiet"], cwd=repo, check=True)
        (repo / ".gitignore").write_text("config/*.json\n", encoding="utf-8")
        target = repo / "config" / "설정.json"
        target.parent.mkdir()
        target.write_text("{}", encoding="utf-8")

        found = GitSource(repo).ignored_paths({"config/설정.json"})

        self.assertEqual({"config/설정.json"}, found)


class _DescribeOnlyRunner:
    """git 프로세스 대역 — describe 만 답하고 나머지 호출은 기록만 한다."""

    def __init__(self, describe: str | None = "v2.9.0-0-gaaaaaaaaa") -> None:
        self.describe = describe
        self.calls: list[tuple[str, ...]] = []

    def __call__(self, argv, **_kwargs):
        args = tuple(argv[1:])
        self.calls.append(args)
        if args and args[0] == "describe":
            if self.describe is None:
                raise OSError("git not installed")
            return subprocess.CompletedProcess(argv, 0, stdout=self.describe + "\n", stderr="")
        return subprocess.CompletedProcess(argv, 1, stdout="", stderr="unexpected git call")


class FileBasedSourceIdentityTests(unittest.TestCase):
    """표시용 정체성은 .git 파일에서 읽고, git 은 describe 1회만 실행한다."""

    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)
        self.root = Path(self.temp_dir.name).resolve()
        (self.root / "VERSION").write_text("2.9.0\n", encoding="utf-8")

    def _git_layout(self, *, packed: bool = False, remotes=None) -> None:
        git = self.root / ".git"
        (git / "refs" / "heads").mkdir(parents=True)
        (git / "HEAD").write_text("ref: refs/heads/main\n", encoding="ascii")
        if packed:
            (git / "packed-refs").write_text(
                "# pack-refs with: peeled fully-peeled sorted\n"
                + "a" * 40 + " refs/heads/main\n", encoding="ascii")
        else:
            (git / "refs" / "heads" / "main").write_text("A" * 40 + "\n", encoding="ascii")
        remotes = remotes if remotes is not None else {
            "fork": "https://github.com/someone/UR_IV.git",
            "upstream": "https://github.com/UR-al/UR_IV.git",
        }
        config = "[core]\n\trepositoryformatversion = 0\n"
        for name, url in remotes.items():
            config += f'[remote "{name}"]\n\turl = {url}\n\tfetch = +refs/heads/*:refs/remotes/{name}/*\n'
        (git / "config").write_text(config, encoding="utf-8")

    def _manager(self, runner) -> AppUpdateManager:
        return AppUpdateManager(
            project_root=self.root,
            settings_path=self.root / "config" / "app_update.json",
            plan_dir=self.root / "cache" / "updates",
            result_path=self.root / "logs" / "updates" / "last_result.json",
            release_client=_FakeReleaseClient(_github_release("2.10.0")),
            git_source=GitSource(self.root, runner=runner),
            process_launcher=lambda _path: 1,
            clock=lambda: datetime(2026, 9, 2, tzinfo=timezone.utc),
            current_pid=lambda: 4242,
            instance_scanner=lambda *_args, **_kwargs: [],
            update_lock_probe=lambda: False,
        )

    def test_local_identity_reads_head_branch_and_trusted_remote_without_git(self) -> None:
        for packed in (False, True):
            with self.subTest(packed=packed):
                shutil.rmtree(self.root / ".git", ignore_errors=True)
                self._git_layout(packed=packed)
                runner = _DescribeOnlyRunner()
                identity = GitSource(self.root, runner=runner).local_identity()
                self.assertEqual(
                    {"checkout": True, "head": "a" * 40, "branch": "main", "trustedRemote": "upstream"},
                    identity,
                )
                self.assertEqual([], runner.calls)

    def test_snapshot_runs_git_only_for_describe(self) -> None:
        self._git_layout()
        runner = _DescribeOnlyRunner("v2.6.0-41-gabcdef123-dirty")
        manager = self._manager(runner)
        snapshot = manager.snapshot()
        self.assertEqual("git", snapshot["mode"])
        self.assertEqual(["describe"], [call[0] for call in runner.calls])
        self.assertEqual("main", snapshot["branch"])
        self.assertEqual("2.6.0", snapshot["currentVersion"])
        self.assertTrue(snapshot["developmentBuild"])
        state = manager._current_source_state()
        self.assertEqual("upstream", state["trustedRemote"])
        self.assertEqual("a" * 9, state["revision"])

    def test_missing_dot_git_is_a_manual_copy_without_any_git_process(self) -> None:
        runner = _DescribeOnlyRunner()
        snapshot = self._manager(runner).snapshot()
        self.assertEqual("manual", snapshot["mode"])
        self.assertEqual([], runner.calls)

    def test_checkout_without_a_working_git_stays_manual(self) -> None:
        self._git_layout()
        snapshot = self._manager(_DescribeOnlyRunner(describe=None)).snapshot()
        self.assertEqual("manual", snapshot["mode"])
        self.assertTrue(snapshot["identityKnown"])

    def test_worktree_gitdir_file_uses_common_dir_refs_and_config(self) -> None:
        common = self.root / "main-repo" / ".git"
        worktree_git = common / "worktrees" / "feature"
        worktree_git.mkdir(parents=True)
        (common / "refs" / "heads").mkdir(parents=True)
        (common / "refs" / "heads" / "feature").write_text("b" * 40 + "\n", encoding="ascii")
        (common / "config").write_text(
            '[remote "origin"]\n\turl = git@github.com:UR-al/UR_IV.git\n', encoding="utf-8")
        (worktree_git / "HEAD").write_text("ref: refs/heads/feature\n", encoding="ascii")
        (worktree_git / "commondir").write_text("../..\n", encoding="ascii")
        (self.root / ".git").write_text(f"gitdir: {worktree_git}\n", encoding="utf-8")
        runner = _DescribeOnlyRunner()
        identity = GitSource(self.root, runner=runner).local_identity()
        self.assertEqual(
            {"checkout": True, "head": "b" * 40, "branch": "feature", "trustedRemote": "origin"},
            identity,
        )
        self.assertEqual([], runner.calls)


if __name__ == "__main__":
    unittest.main()
