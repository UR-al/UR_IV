"""Detached helper that applies a validated Git fast-forward after app exit."""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping

from utils.atomic_json import atomic_write_json

# wait_for_exit 의 기본 인자로 쓰이므로 모듈 수준 import 여야 한다(구현은 한 벌만 둔다).
from core.app_instance import process_exists


_COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")
_TAG_RE = re.compile(r"^v\d+\.\d+\.\d+$")
_REMOTE_RE = re.compile(
    r"^(?:https://github\.com/|git@github\.com:|ssh://git@github\.com/)"
    r"UR-al/UR_IV(?:\.git)?/?$",
    re.IGNORECASE,
)


class InstallerError(RuntimeError):
    pass


def _within(candidate: Path, root: Path, label: str) -> Path:
    resolved = candidate.resolve()
    try:
        resolved.relative_to(root.resolve())
    except ValueError as exc:
        raise InstallerError(f"{label} is outside the project") from exc
    return resolved


def load_plan(plan_path: str | os.PathLike[str]) -> dict[str, Any]:
    path = Path(plan_path).resolve()
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError) as exc:
        raise InstallerError("update plan could not be read") from exc
    if not isinstance(raw, Mapping) or raw.get("schema") != 1:
        raise InstallerError("unsupported update plan")
    root = Path(str(raw.get("projectRoot") or "")).resolve()
    if not root.is_dir() or not (root / ".git").exists():
        raise InstallerError("project checkout is missing")
    _within(path, root / "cache" / "updates", "plan")
    expected = str(raw.get("expectedHead") or "").lower()
    target = str(raw.get("targetCommit") or "").lower()
    tag = str(raw.get("tagName") or "")
    remote = str(raw.get("remoteName") or "")
    branch = str(raw.get("branch") or "")
    if not _COMMIT_RE.fullmatch(expected) or not _COMMIT_RE.fullmatch(target):
        raise InstallerError("invalid update commit")
    if not _TAG_RE.fullmatch(tag) or not remote or not branch:
        raise InstallerError("invalid Git update identity")
    if any(char in remote + branch for char in "\x00\r\n"):
        raise InstallerError("invalid Git name")
    launcher = _within(Path(str(raw.get("launcher") or "")), root, "launcher")
    if launcher != (root / "new_run_main_ui.bat").resolve() or not launcher.is_file():
        raise InstallerError("invalid launcher")
    result_path = _within(Path(str(raw.get("resultPath") or "")), root / "logs", "result")
    ready_path = _within(Path(str(raw.get("readyPath") or "")), root / "cache" / "updates", "ready")
    parent_pid = int(raw.get("parentPid") or 0)
    if parent_pid <= 0:
        raise InstallerError("invalid parent process")
    return {
        **dict(raw),
        "projectRoot": str(root),
        "launcher": str(launcher),
        "resultPath": str(result_path),
        "readyPath": str(ready_path),
        "expectedHead": expected,
        "targetCommit": target,
        "parentPid": parent_pid,
    }


def wait_for_exit(
    pid: int,
    *,
    exists: Callable[[int], bool] = process_exists,
    timeout: float = 180.0,
    poll_interval: float = 0.25,
) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if not exists(pid):
            return True
        time.sleep(poll_interval)
    return not exists(pid)


def _run_git(root: Path, *args: str, timeout: int = 180) -> str:
    git_env = {
        key: value
        for key, value in os.environ.items()
        if not key.upper().startswith("GIT_")
    }
    git_env.update({"GIT_TERMINAL_PROMPT": "0", "GCM_INTERACTIVE": "Never"})
    try:
        completed = subprocess.run(
            ["git", *args],
            cwd=str(root),
            env=git_env,
            text=True,
            encoding="utf-8",
            errors="replace",
            capture_output=True,
            check=False,
            timeout=timeout,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise InstallerError("Git command failed to start") from exc
    if completed.returncode != 0:
        raise InstallerError((completed.stderr or "Git command failed").strip()[:500])
    return completed.stdout or ""


def _paths(value: str) -> set[str]:
    return {path.replace("\\", "/") for path in value.split("\x00") if path}


def _ignored_release_paths(
    root: Path,
    changed: set[str],
    run_git: Callable[..., str],
) -> set[str]:
    found: set[str] = set()
    ordered = sorted(changed)
    for offset in range(0, len(ordered), 80):
        pathspecs = [f":(literal){path}" for path in ordered[offset:offset + 80]]
        found |= _paths(run_git(
            root,
            "ls-files",
            "-z",
            "--others",
            "--ignored",
            "--exclude-standard",
            "--",
            *pathspecs,
        ))
    return found


def apply_plan(plan: Mapping[str, Any], run_git: Callable[..., str] = _run_git) -> dict[str, Any]:
    root = Path(str(plan["projectRoot"]))
    expected = str(plan["expectedHead"])
    target = str(plan["targetCommit"])
    remote = str(plan["remoteName"])
    from core.app_instance import live_app_instance_pids

    other_instances = live_app_instance_pids(root)
    if other_instances:
        raise InstallerError("another AI Studio Pro instance is still running")
    try:
        top_level = Path(run_git(root, "rev-parse", "--show-toplevel").strip()).resolve()
    except OSError as exc:
        raise InstallerError("checkout root could not be verified") from exc
    if top_level != root.resolve():
        raise InstallerError("checkout root changed after update approval")
    if run_git(root, "rev-parse", "HEAD").strip().lower() != expected:
        raise InstallerError("checkout changed after update approval")
    if run_git(root, "branch", "--show-current").strip() != str(plan["branch"]):
        raise InstallerError("checkout branch changed after update approval")
    remote_url = run_git(root, "remote", "get-url", remote).strip()
    if not _REMOTE_RE.fullmatch(remote_url):
        raise InstallerError("untrusted Git remote")
    resolved_target = run_git(root, "rev-parse", f"{plan['tagName']}^{{commit}}").strip().lower()
    if resolved_target != target:
        raise InstallerError("release tag changed after update approval")
    for required_path in (
        "VERSION",
        "new_main_ui.py",
        "new_run_main_ui.bat",
        "frontend_dist/index.html",
    ):
        run_git(root, "cat-file", "-e", f"{target}:{required_path}")
    target_version = run_git(root, "show", f"{target}:VERSION").strip()
    if f"v{target_version}" != str(plan["tagName"]):
        raise InstallerError("release VERSION does not match its tag")
    run_git(root, "merge-base", "--is-ancestor", expected, target)
    changed = _paths(run_git(root, "diff", "--name-only", "--no-renames", "-z", expected, target))
    dirty = set()
    dirty |= _paths(run_git(root, "diff", "--name-only", "-z"))
    dirty |= _paths(run_git(root, "diff", "--cached", "--name-only", "-z"))
    dirty |= _paths(run_git(root, "ls-files", "-z", "--others", "--exclude-standard"))
    dirty |= _ignored_release_paths(root, changed, run_git)
    if {path.casefold() for path in changed} & {path.casefold() for path in dirty}:
        raise InstallerError("local files changed after update approval")
    run_git(root, "-c", "core.hooksPath=NUL", "merge", "--ff-only", target)
    actual = run_git(root, "rev-parse", "HEAD").strip().lower()
    if actual != target:
        raise InstallerError("updated commit could not be verified")
    return {"ok": True, "tagName": str(plan["tagName"]), "commit": actual}


def launch_app(path: Path) -> None:
    if os.name == "nt":
        os.startfile(str(path))  # type: ignore[attr-defined]
        return
    subprocess.Popen([str(path)], cwd=str(path.parent), start_new_session=True)


def _receipt(ok: bool, message: str, plan: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "schema": 1,
        "ok": ok,
        "message": message[:1000],
        "tagName": str(plan.get("tagName") or ""),
        "finishedAt": datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
    }


def run_installer(
    plan_path: str | os.PathLike[str],
    *,
    waiter: Callable[[int], bool] | None = None,
    applier: Callable[[Mapping[str, Any]], Mapping[str, Any]] = apply_plan,
    launcher: Callable[[Path], None] = launch_app,
    lock_factory: Callable[[Path], Any] | None = None,
) -> int:
    raw_for_cleanup: dict[str, Any] = {}
    parent_exited = False
    plan_file = Path(plan_path).resolve()
    try:
        plan = load_plan(plan_file)
        raw_for_cleanup = plan
        if lock_factory is None:
            from core.app_update_lock import acquire_update_lock

            lock_factory = lambda root: acquire_update_lock(root, timeout=0.0)
        with lock_factory(Path(str(plan["projectRoot"]))):
            atomic_write_json(str(plan["readyPath"]), {
                "ready": True,
                "pid": os.getpid(),
                "targetCommit": str(plan["targetCommit"]),
            })
            wait = waiter or (lambda pid: wait_for_exit(pid))
            if not wait(int(plan["parentPid"])):
                raise InstallerError("app did not exit before the update timeout")
            parent_exited = True
            result = dict(applier(plan))
            receipt = _receipt(True, f"{plan['tagName']} 업데이트를 설치했습니다.", plan)
            receipt.update({key: value for key, value in result.items() if key in {"commit"}})
            atomic_write_json(str(plan["resultPath"]), receipt)
            launcher(Path(str(plan["launcher"])))
            return 0
    except Exception as exc:
        if raw_for_cleanup:
            try:
                atomic_write_json(
                    str(raw_for_cleanup["resultPath"]),
                    _receipt(False, str(exc) or type(exc).__name__, raw_for_cleanup),
                )
            except Exception:
                pass
            # No reset/stash is attempted. Relaunch whichever verified checkout
            # remains so a failed source update does not strand the user.
            if parent_exited:
                try:
                    launcher(Path(str(raw_for_cleanup["launcher"])))
                except Exception:
                    pass
        return 1
    finally:
        if raw_for_cleanup:
            try:
                Path(str(raw_for_cleanup["readyPath"])).unlink(missing_ok=True)
            except OSError:
                pass
        try:
            plan_file.unlink(missing_ok=True)
        except OSError:
            pass


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Apply a prepared AI Studio Pro update")
    parser.add_argument("--plan", required=True)
    args = parser.parse_args(argv)
    return run_installer(args.plan)


if __name__ == "__main__":
    sys.exit(main())


__all__ = ["InstallerError", "apply_plan", "load_plan", "run_installer", "wait_for_exit"]
