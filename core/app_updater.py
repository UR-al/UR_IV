"""GitHub Releases based application update service.

The service deliberately keeps network, Git and process details behind one
small ``snapshot``/``execute`` interface.  A source checkout can update itself
with a guarded fast-forward.  Copies without Git metadata still receive update
notifications and release notes, but are never overwritten in place.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import threading
import time
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable, Mapping

from core.storage_paths import PROJECT_ROOT
from utils.atomic_json import atomic_write_json, load_json_safe


REPOSITORY = "UR-al/UR_IV"
REPOSITORY_URL = f"https://github.com/{REPOSITORY}"
RELEASES_URL = f"{REPOSITORY_URL}/releases"
LATEST_RELEASE_API = f"https://api.github.com/repos/{REPOSITORY}/releases/latest"
CONFIG_SCHEMA = 1
DEFAULT_INTERVAL_HOURS = 12
_VERSION_RE = re.compile(r"^v?(\d+)\.(\d+)\.(\d+)$")
_COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")
_REMOTE_RE = re.compile(
    r"^(?:https://github\.com/|git@github\.com:|ssh://git@github\.com/)"
    r"UR-al/UR_IV(?:\.git)?/?$",
    re.IGNORECASE,
)


class AppUpdateError(RuntimeError):
    """Structured updater failure suitable for StudioApplication."""

    def __init__(
        self,
        code: str,
        message: str,
        *,
        retryable: bool = False,
        details: Mapping[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.code = str(code)
        self.message = str(message)
        self.retryable = bool(retryable)
        self.details = dict(details or {})

    def as_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "code": self.code,
            "message": self.message,
            "retryable": self.retryable,
        }
        if self.details:
            result["details"] = dict(self.details)
        return result


def parse_version(value: Any) -> tuple[int, int, int] | None:
    match = _VERSION_RE.fullmatch(str(value or "").strip())
    if not match:
        return None
    return tuple(int(part) for part in match.groups())  # type: ignore[return-value]


def normalise_version(value: Any) -> str:
    parsed = parse_version(value)
    return ".".join(str(part) for part in parsed) if parsed else ""


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _iso_time(value: datetime) -> str:
    return value.astimezone(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _parse_time(value: Any) -> datetime | None:
    text = str(value or "").strip()
    if not text:
        return None
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


class GitHubReleaseClient:
    """Minimal client for the public latest-release endpoint."""

    def fetch_latest(self) -> dict[str, Any]:
        import requests

        try:
            response = requests.get(
                LATEST_RELEASE_API,
                headers={
                    "Accept": "application/vnd.github+json",
                    "X-GitHub-Api-Version": "2022-11-28",
                    "User-Agent": "AI-Studio-Pro-Updater",
                },
                timeout=(8, 20),
            )
            response.raise_for_status()
            raw = response.json()
        except requests.Timeout as exc:
            raise AppUpdateError(
                "UPDATE_TIMEOUT",
                "GitHub 업데이트 확인 시간이 초과되었습니다.",
                retryable=True,
            ) from exc
        except requests.RequestException as exc:
            raise AppUpdateError(
                "UPDATE_NETWORK",
                "GitHub 릴리스 정보를 가져오지 못했습니다.",
                retryable=True,
            ) from exc
        except (ValueError, TypeError) as exc:
            raise AppUpdateError(
                "UPDATE_RESPONSE",
                "GitHub 릴리스 응답 형식이 올바르지 않습니다.",
                retryable=True,
            ) from exc
        if not isinstance(raw, Mapping):
            raise AppUpdateError("UPDATE_RESPONSE", "GitHub 릴리스 응답이 비어 있습니다.")
        return dict(raw)


class GitSource:
    """Strict argument-list Git adapter scoped to one project checkout."""

    def __init__(self, root: Path, runner: Callable[..., subprocess.CompletedProcess[str]] | None = None):
        self.root = root.resolve()
        self._runner = runner or subprocess.run

    def _execute(self, *args: str, timeout: int = 60) -> str:
        git_env = {
            key: value
            for key, value in os.environ.items()
            if not key.upper().startswith("GIT_")
        }
        git_env.update({"GIT_TERMINAL_PROMPT": "0", "GCM_INTERACTIVE": "Never"})
        try:
            completed = self._runner(
                ["git", *args],
                cwd=str(self.root),
                env=git_env,
                text=True,
                encoding="utf-8",
                errors="replace",
                capture_output=True,
                timeout=timeout,
                check=False,
            )
        except (OSError, subprocess.SubprocessError) as exc:
            raise AppUpdateError(
                "GIT_UNAVAILABLE", "Git 명령을 실행하지 못했습니다.", retryable=True
            ) from exc
        if int(getattr(completed, "returncode", 1)) != 0:
            message = str(getattr(completed, "stderr", "") or "").strip()
            raise AppUpdateError(
                "GIT_FAILED",
                message[:500] or "Git 작업을 완료하지 못했습니다.",
                retryable=True,
            )
        return str(getattr(completed, "stdout", "") or "")

    def run(self, *args: str, timeout: int = 60) -> str:
        return self._execute(*args, timeout=timeout).strip()

    def run_paths(self, *args: str, timeout: int = 60) -> set[str]:
        return {
            path.replace("\\", "/")
            for path in self._execute(*args, timeout=timeout).split("\x00")
            if path
        }

    def is_checkout(self) -> bool:
        try:
            if self.run("rev-parse", "--is-inside-work-tree") != "true":
                return False
            return Path(self.run("rev-parse", "--show-toplevel")).resolve() == self.root
        except (AppUpdateError, OSError):
            return False

    def head(self) -> str:
        return self.run("rev-parse", "HEAD").lower()

    def branch(self) -> str:
        return self.run("branch", "--show-current")

    def describe(self) -> str:
        try:
            return self.run(
                "describe",
                "--tags",
                "--long",
                "--dirty",
                "--match",
                "v[0-9]*",
                "--always",
            )
        except AppUpdateError:
            return ""

    def local_identity(self) -> dict[str, Any] | None:
        """표시용 체크아웃 정체성을 git 프로세스 없이 .git 파일에서 읽는다.

        반환: ``{"checkout": False}`` (이 프로젝트 루트에 .git 이 없음) 또는
        ``{"checkout": True, "head", "branch", "trustedRemote"}``. 파일로 판단할 수 없으면
        ``None`` — 호출자는 git 명령으로 되돌아간다. 업데이트 설치는 이 값을 믿지 않고
        git 으로 다시 확인한다(_install).
        """
        from core.git_metadata import git_dir, head_info, remote_urls

        if git_dir(self.root) is None:
            return {"checkout": False}
        head, branch = head_info(self.root)
        if not _COMMIT_RE.fullmatch(head):
            return None
        remote = ""
        # `git remote` 와 같은 이름순으로 공식 원격을 고른다.
        for name, url in sorted(remote_urls(self.root).items()):
            if _REMOTE_RE.fullmatch(url.strip()):
                remote = name
                break
        if not remote:
            # includeIf·insteadOf 같은 설정은 파일만으로 알 수 없다 — 이때만 git 에 묻는다.
            try:
                remote = self.trusted_remote()
            except AppUpdateError:
                remote = ""
        return {"checkout": True, "head": head, "branch": branch, "trustedRemote": remote}

    def trusted_remote(self) -> str:
        for name in self.run("remote").splitlines():
            candidate = name.strip()
            if not candidate:
                continue
            try:
                url = self.run("remote", "get-url", candidate)
            except AppUpdateError:
                continue
            if _REMOTE_RE.fullmatch(url.strip()):
                return candidate
        return ""

    def fetch_release_tag(self, remote: str, tag_name: str) -> str:
        if not remote or not _VERSION_RE.fullmatch(tag_name):
            raise AppUpdateError("UPDATE_INVALID", "업데이트 태그 또는 원격 저장소가 올바르지 않습니다.")
        self.run("fetch", "--quiet", remote, "tag", tag_name, timeout=180)
        target = self.run("rev-parse", f"{tag_name}^{{commit}}").lower()
        if not _COMMIT_RE.fullmatch(target):
            raise AppUpdateError("UPDATE_INVALID", "릴리스 커밋을 확인하지 못했습니다.")
        return target

    def is_ancestor(self, ancestor: str, descendant: str) -> bool:
        try:
            self.run("merge-base", "--is-ancestor", ancestor, descendant)
            return True
        except AppUpdateError:
            return False

    def changed_paths(self, base: str, target: str) -> set[str]:
        return self.run_paths("diff", "--name-only", "--no-renames", "-z", base, target)

    def dirty_paths(self) -> set[str]:
        return (
            self.run_paths("diff", "--name-only", "-z")
            | self.run_paths("diff", "--cached", "--name-only", "-z")
            | self.run_paths("ls-files", "-z", "--others", "--exclude-standard")
        )

    def ignored_paths(self, candidates: set[str]) -> set[str]:
        """Return ignored, untracked files that a release would overwrite.

        Query only release paths so large ignored model/output trees are not
        enumerated merely to update the application.
        """

        found: set[str] = set()
        ordered = sorted(path for path in candidates if path)
        for offset in range(0, len(ordered), 80):
            pathspecs = [f":(literal){path}" for path in ordered[offset:offset + 80]]
            found.update(self.run_paths(
                "ls-files",
                "-z",
                "--others",
                "--ignored",
                "--exclude-standard",
                "--",
                *pathspecs,
            ))
        return found


class AppUpdateManager:
    """Deep updater interface used by desktop and web transports."""

    def __init__(
        self,
        *,
        project_root: str | os.PathLike[str] = PROJECT_ROOT,
        settings_path: str | os.PathLike[str] | None = None,
        plan_dir: str | os.PathLike[str] | None = None,
        result_path: str | os.PathLike[str] | None = None,
        release_client: Any = None,
        git_source: GitSource | None = None,
        process_launcher: Callable[[Path], int] | None = None,
        clock: Callable[[], datetime] = _utc_now,
        current_pid: Callable[[], int] = os.getpid,
        instance_scanner: Callable[..., list[int]] | None = None,
        update_lock_probe: Callable[[], bool] | None = None,
    ) -> None:
        self.project_root = Path(project_root).resolve()
        self.settings_path = Path(settings_path) if settings_path else self.project_root / "config" / "app_update.json"
        self.plan_dir = Path(plan_dir) if plan_dir else self.project_root / "cache" / "updates"
        self.result_path = Path(result_path) if result_path else self.project_root / "logs" / "updates" / "last_result.json"
        self.release_client = release_client or GitHubReleaseClient()
        self.git = git_source or GitSource(self.project_root)
        self._process_launcher = process_launcher or self._launch_installer
        self._clock = clock
        self._current_pid = current_pid
        if instance_scanner is None:
            from core.app_instance import live_app_instance_pids

            instance_scanner = live_app_instance_pids
        self._instance_scanner = instance_scanner
        if update_lock_probe is None:
            from core.app_update_lock import is_update_in_progress

            update_lock_probe = lambda: is_update_in_progress(self.project_root)
        self._update_lock_probe = update_lock_probe
        self._lock = threading.RLock()
        self._busy_action = ""

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            settings = self._load_settings()
            busy_action = self._busy_action
        current = self._current_source_state()
        release = self._normalise_cached_release(settings.get("latestRelease"))
        current_tuple = parse_version(current["version"]) or (0, 0, 0)
        latest_tuple = parse_version(release.get("version")) if release else None
        update_available = bool(latest_tuple and latest_tuple > current_tuple)
        skipped_version = normalise_version(settings.get("skippedVersion"))
        skipped = bool(update_available and skipped_version == release.get("version"))
        last_checked = _parse_time(settings.get("lastCheckedAt"))
        interval = max(1, min(168, int(settings.get("intervalHours") or DEFAULT_INTERVAL_HOURS)))
        auto_check = bool(settings.get("autoCheck", True))
        should_check = auto_check and (
            last_checked is None or self._clock() - last_checked >= timedelta(hours=interval)
        )
        trusted_remote = current.get("trustedRemote", "")
        can_install = bool(
            update_available
            and current["mode"] == "git"
            and current.get("identityKnown")
            and current.get("branch")
            and trusted_remote
            and not busy_action
        )
        install_reason = ""
        if update_available and not can_install:
            if current["mode"] != "git":
                install_reason = "이 복사본은 Git 설치가 아니므로 릴리스 페이지에서 받아야 합니다."
            elif not current.get("identityKnown"):
                install_reason = "현재 Git 버전의 기준 릴리스를 확인할 수 없어 자동 업데이트하지 않습니다."
            elif not current.get("branch"):
                install_reason = "분리된 Git 커밋에서는 자동 업데이트할 수 없습니다."
            elif not trusted_remote:
                install_reason = "공식 GitHub 원격 저장소를 확인하지 못했습니다."
            elif busy_action:
                install_reason = "다른 업데이트 작업이 진행 중입니다."
        return {
            "ok": True,
            "repository": REPOSITORY,
            "repositoryUrl": REPOSITORY_URL,
            "releasesUrl": RELEASES_URL,
            "currentVersion": current["version"],
            "currentDisplay": current["display"],
            "currentRevision": current.get("revision", ""),
            "developmentBuild": bool(current.get("developmentBuild")),
            "identityKnown": bool(current.get("identityKnown")),
            "mode": current["mode"],
            "branch": current.get("branch", ""),
            "latestVersion": release.get("version", "") if release else "",
            "tagName": release.get("tagName", "") if release else "",
            "releaseName": release.get("name", "") if release else "",
            "releaseUrl": release.get("url", "") if release else "",
            "publishedAt": release.get("publishedAt", "") if release else "",
            "notes": release.get("notes", "") if release else "",
            "updateAvailable": update_available,
            "notificationAvailable": bool(update_available and not skipped),
            "skipped": skipped,
            "skippedVersion": skipped_version,
            "autoCheck": auto_check,
            "intervalHours": interval,
            "lastCheckedAt": str(settings.get("lastCheckedAt") or ""),
            "shouldAutoCheck": should_check,
            "busy": bool(busy_action),
            "busyAction": busy_action,
            "canInstall": can_install,
            "installReason": install_reason,
            "pendingVersion": normalise_version(settings.get("pendingVersion")),
            "lastResult": self._load_last_result(),
        }

    def execute(
        self,
        action: str,
        payload: Mapping[str, Any] | None = None,
        on_progress: Callable[[Mapping[str, Any]], None] | None = None,
    ) -> dict[str, Any]:
        action = str(action or "").strip().lower()
        values = dict(payload or {})
        if action not in {"check", "configure", "skip", "install"}:
            raise AppUpdateError("INVALID_ARGUMENT", f"지원하지 않는 업데이트 작업입니다: {action}")
        with self._lock:
            if self._busy_action:
                raise AppUpdateError(
                    "OPERATION_BUSY",
                    "다른 업데이트 작업이 진행 중입니다.",
                    retryable=True,
                    details={"action": self._busy_action},
                )
            self._busy_action = action
        try:
            if action == "check":
                result = self._check(on_progress)
            elif action == "configure":
                result = self._configure(values)
            elif action == "skip":
                result = self._skip(values)
            else:
                result = self._install(on_progress)
            result.setdefault("ok", True)
            result.setdefault("action", action)
            return result
        finally:
            with self._lock:
                self._busy_action = ""

    def _check(self, on_progress: Callable[[Mapping[str, Any]], None] | None) -> dict[str, Any]:
        if on_progress:
            on_progress({"stage": "checking", "message": "GitHub 릴리스를 확인하는 중입니다."})
        raw = self.release_client.fetch_latest()
        release = self._normalise_release(raw)
        with self._lock:
            settings = self._load_settings()
            settings["lastCheckedAt"] = _iso_time(self._clock())
            settings["latestRelease"] = release
            self._save_settings(settings)
        snapshot = self.snapshot()
        return {
            "message": (
                f"v{release['version']} 업데이트를 사용할 수 있습니다."
                if snapshot["updateAvailable"]
                else "현재 버전이 최신입니다."
            ),
            "snapshot": snapshot,
        }

    def _configure(self, values: Mapping[str, Any]) -> dict[str, Any]:
        if set(values) - {"autoCheck"} or "autoCheck" not in values or not isinstance(values["autoCheck"], bool):
            raise AppUpdateError("INVALID_ARGUMENT", "autoCheck는 true 또는 false여야 합니다.")
        with self._lock:
            settings = self._load_settings()
            settings["autoCheck"] = values["autoCheck"]
            self._save_settings(settings)
        return {"message": "자동 업데이트 확인 설정을 저장했습니다.", "snapshot": self.snapshot()}

    def _skip(self, values: Mapping[str, Any]) -> dict[str, Any]:
        if set(values) - {"version"}:
            raise AppUpdateError("INVALID_ARGUMENT", "알 수 없는 건너뛰기 설정입니다.")
        requested = normalise_version(values.get("version"))
        snapshot = self.snapshot()
        latest = normalise_version(snapshot.get("latestVersion"))
        if requested and requested != latest:
            raise AppUpdateError("INVALID_ARGUMENT", "현재 최신 버전만 건너뛸 수 있습니다.")
        with self._lock:
            settings = self._load_settings()
            settings["skippedVersion"] = requested
            self._save_settings(settings)
        message = f"v{requested} 알림을 건너뜁니다." if requested else "건너뛴 업데이트 알림을 다시 표시합니다."
        return {"message": message, "snapshot": self.snapshot()}

    def _install(self, on_progress: Callable[[Mapping[str, Any]], None] | None) -> dict[str, Any]:
        # Refresh immediately before resolving a mutable Git tag.
        checked = self._check(on_progress)
        snapshot = checked["snapshot"]
        if not snapshot.get("updateAvailable"):
            raise AppUpdateError("NO_UPDATE", "설치할 새 버전이 없습니다.")
        if snapshot.get("mode") != "git":
            raise AppUpdateError("MANUAL_UPDATE_REQUIRED", str(snapshot.get("installReason") or "수동 업데이트가 필요합니다."))
        if not snapshot.get("identityKnown"):
            raise AppUpdateError("UPDATE_UNAVAILABLE", str(snapshot.get("installReason") or "현재 Git 버전을 확인할 수 없습니다."))
        remote = str(self.git.trusted_remote() or "")
        branch = str(self.git.branch() or "")
        tag_name = str(snapshot.get("tagName") or "")
        if not remote or not branch:
            raise AppUpdateError("UPDATE_UNAVAILABLE", str(snapshot.get("installReason") or "자동 업데이트를 사용할 수 없습니다."))
        other_instances = self._instance_scanner(
            self.project_root,
            exclude_pid=int(self._current_pid()),
        )
        if other_instances:
            raise AppUpdateError(
                "UPDATE_OTHER_INSTANCE",
                "다른 AI Studio Pro 창이 실행 중입니다. 다른 창을 닫고 다시 시도하세요.",
                retryable=True,
                details={"count": len(other_instances)},
            )
        try:
            update_active = bool(self._update_lock_probe())
        except Exception as exc:
            raise AppUpdateError(
                "UPDATE_UNAVAILABLE",
                "업데이트 실행 상태를 확인하지 못했습니다.",
                retryable=True,
            ) from exc
        if update_active:
            raise AppUpdateError(
                "OPERATION_BUSY",
                "다른 업데이트 작업이 이미 진행 중입니다.",
                retryable=True,
            )
        if on_progress:
            on_progress({"stage": "fetching", "message": f"{tag_name} 파일을 준비하는 중입니다."})
        head = self.git.head()
        target = self.git.fetch_release_tag(remote, tag_name)
        if head == target:
            raise AppUpdateError("NO_UPDATE", "이미 해당 릴리스가 설치되어 있습니다.")
        if not self.git.is_ancestor(head, target):
            raise AppUpdateError(
                "UPDATE_DIVERGED",
                "현재 코드와 릴리스가 갈라져 있어 자동 업데이트하지 않았습니다. 사용자 변경은 그대로 유지됩니다.",
            )
        changed_paths = self.git.changed_paths(head, target)
        dirty_paths = self.git.dirty_paths() | self.git.ignored_paths(changed_paths)
        changed_by_key = {path.casefold(): path for path in changed_paths}
        dirty_keys = {path.casefold() for path in dirty_paths}
        conflicts = sorted(
            path for key, path in changed_by_key.items() if key in dirty_keys
        )
        if conflicts:
            raise AppUpdateError(
                "UPDATE_LOCAL_CHANGES",
                "업데이트 파일과 겹치는 수정 사항이 있어 자동 업데이트하지 않았습니다.",
                details={"paths": conflicts[:50]},
            )
        launcher = (self.project_root / "new_run_main_ui.bat").resolve()
        try:
            launcher.relative_to(self.project_root)
        except ValueError as exc:
            raise AppUpdateError("UPDATE_INVALID", "재시작 파일 경로가 올바르지 않습니다.") from exc
        if not launcher.is_file():
            raise AppUpdateError("UPDATE_UNAVAILABLE", "앱 재시작 파일을 찾지 못했습니다.")
        self.plan_dir.mkdir(parents=True, exist_ok=True)
        plan_path = self.plan_dir / f"pending-{uuid.uuid4().hex}.json"
        ready_path = self.plan_dir / f"ready-{uuid.uuid4().hex}.json"
        plan = {
            "schema": 1,
            "projectRoot": str(self.project_root),
            "expectedHead": head,
            "targetCommit": target,
            "tagName": tag_name,
            "remoteName": remote,
            "branch": branch,
            "launcher": str(launcher),
            "parentPid": int(self._current_pid()),
            "resultPath": str(self.result_path.resolve()),
            "readyPath": str(ready_path.resolve()),
        }
        try:
            with self._lock:
                settings = self._load_settings()
                settings["pendingVersion"] = str(snapshot.get("latestVersion") or "")
                self._save_settings(settings)
            atomic_write_json(str(plan_path), plan)
            helper_pid = int(self._process_launcher(plan_path))
        except Exception as exc:
            try:
                plan_path.unlink(missing_ok=True)
            except OSError:
                pass
            try:
                ready_path.unlink(missing_ok=True)
            except OSError:
                pass
            try:
                with self._lock:
                    settings = self._load_settings()
                    settings["pendingVersion"] = ""
                    self._save_settings(settings)
            except OSError:
                pass
            raise AppUpdateError("UPDATE_LAUNCH_FAILED", "업데이트 도우미를 시작하지 못했습니다.", retryable=True) from exc
        return {
            "message": "앱을 종료한 뒤 업데이트하고 자동으로 다시 시작합니다.",
            "restartRequired": True,
            "helperPid": helper_pid,
            "targetVersion": str(snapshot.get("latestVersion") or ""),
        }

    def _local_identity(self) -> dict[str, Any] | None:
        reader = getattr(self.git, "local_identity", None)
        if not callable(reader):
            return None
        try:
            identity = reader()
        except Exception:
            return None
        return identity if isinstance(identity, Mapping) else None

    def _current_source_state(self) -> dict[str, Any]:
        fallback = self._version_file()
        manual = {
            "mode": "manual",
            "version": fallback,
            "display": f"v{fallback}",
            "identityKnown": True,
        }
        # 표시용 정체성(HEAD·브랜치·원격)은 .git 파일에서 읽는다 — 예전에는 앱 마운트와
        # Settings 첫 진입마다 메인 스레드 QWebChannel 슬롯에서 git 을 7번 실행했다.
        # git 프로세스는 `describe --dirty` 1회만 남는다(태그 거리와 작업 트리 수정 여부).
        identity = self._local_identity()
        if identity is not None and not identity.get("checkout"):
            return manual
        try:
            if identity is None:
                if not self.git.is_checkout():
                    return manual
                revision = self.git.head()
                branch = self.git.branch()
                remote = self.git.trusted_remote()
            else:
                revision = str(identity.get("head") or "")
                branch = str(identity.get("branch") or "")
                remote = str(identity.get("trustedRemote") or "")
            described = self.git.describe()
        except AppUpdateError:
            return {**manual, "identityKnown": False}
        if identity is not None and not described:
            # `describe --always` 는 git 이 동작하면 항상 무언가를 돌려준다 — 빈 값은 git 을
            # 실행할 수 없다는 뜻이다(예전 is_checkout 실패와 같은 수동 모드).
            return manual
        version = fallback
        ahead = 0
        dirty = described.endswith("-dirty")
        clean_description = described.removesuffix("-dirty")
        match = re.fullmatch(r"(v\d+\.\d+\.\d+)-(\d+)-g([0-9a-f]+)", clean_description)
        identity_known = False
        if match and parse_version(match.group(1)):
            version = normalise_version(match.group(1))
            ahead = int(match.group(2))
            identity_known = True
        short = revision[:9]
        display = f"v{version}"
        if ahead:
            display += f" + {ahead}개 변경 ({short})"
        elif dirty:
            display += f" · 로컬 수정 ({short})"
        elif not identity_known:
            display += f" · 기준 릴리스 미확인 ({short})"
        return {
            "mode": "git",
            "version": version,
            "display": display,
            "revision": short,
            "developmentBuild": bool(ahead or dirty or not identity_known),
            "identityKnown": identity_known,
            "dirty": dirty,
            "ahead": ahead,
            "branch": branch,
            "trustedRemote": remote,
        }

    def _version_file(self) -> str:
        try:
            value = normalise_version((self.project_root / "VERSION").read_text(encoding="utf-8"))
        except OSError:
            value = ""
        return value or "0.0.0"

    def _load_last_result(self) -> dict[str, Any]:
        raw = load_json_safe(str(self.result_path), {})
        if not isinstance(raw, Mapping) or raw.get("schema") != 1:
            return {}
        return {
            "ok": bool(raw.get("ok")),
            "message": str(raw.get("message") or "")[:1000],
            "tagName": str(raw.get("tagName") or "")[:100],
            "finishedAt": str(raw.get("finishedAt") or "")[:100],
        }

    def _load_settings(self) -> dict[str, Any]:
        raw = load_json_safe(str(self.settings_path), {})
        values = dict(raw) if isinstance(raw, Mapping) else {}
        return {
            "schema": CONFIG_SCHEMA,
            "autoCheck": bool(values.get("autoCheck", True)),
            "intervalHours": max(1, min(168, int(values.get("intervalHours") or DEFAULT_INTERVAL_HOURS))),
            "lastCheckedAt": str(values.get("lastCheckedAt") or ""),
            "skippedVersion": normalise_version(values.get("skippedVersion")),
            "pendingVersion": normalise_version(values.get("pendingVersion")),
            "latestRelease": self._normalise_cached_release(values.get("latestRelease")),
        }

    def _save_settings(self, settings: Mapping[str, Any]) -> None:
        self.settings_path.parent.mkdir(parents=True, exist_ok=True)
        atomic_write_json(str(self.settings_path), dict(settings))

    @staticmethod
    def _normalise_release(raw: Mapping[str, Any]) -> dict[str, str]:
        if bool(raw.get("draft")) or bool(raw.get("prerelease")):
            raise AppUpdateError("UPDATE_RESPONSE", "정식 릴리스가 아닌 응답은 사용할 수 없습니다.")
        tag = str(raw.get("tag_name") or "").strip()
        version = normalise_version(tag)
        url = str(raw.get("html_url") or "").strip()
        if not version or not url.startswith(f"{REPOSITORY_URL}/releases/"):
            raise AppUpdateError("UPDATE_RESPONSE", "릴리스 태그 또는 주소가 올바르지 않습니다.")
        return {
            "version": version,
            "tagName": f"v{version}",
            "name": str(raw.get("name") or f"v{version}").replace("\x00", "")[:300],
            "url": url[:1000],
            "publishedAt": str(raw.get("published_at") or "")[:100],
            "notes": str(raw.get("body") or "").replace("\x00", "")[:50_000],
        }

    @staticmethod
    def _normalise_cached_release(raw: Any) -> dict[str, str]:
        if not isinstance(raw, Mapping):
            return {}
        version = normalise_version(raw.get("version"))
        tag = str(raw.get("tagName") or "")
        url = str(raw.get("url") or "")
        if not version or tag != f"v{version}" or not url.startswith(f"{REPOSITORY_URL}/releases/"):
            return {}
        return {
            "version": version,
            "tagName": tag,
            "name": str(raw.get("name") or tag)[:300],
            "url": url[:1000],
            "publishedAt": str(raw.get("publishedAt") or "")[:100],
            "notes": str(raw.get("notes") or "")[:50_000],
        }

    def _launch_installer(self, plan_path: Path) -> int:
        command = [sys.executable, "-m", "core.app_update_installer", "--plan", str(plan_path)]
        kwargs: dict[str, Any] = {
            "cwd": str(self.project_root),
            "stdin": subprocess.DEVNULL,
            "stdout": subprocess.DEVNULL,
            "stderr": subprocess.DEVNULL,
            "close_fds": True,
        }
        if os.name == "nt":
            kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.DETACHED_PROCESS
        else:
            kwargs["start_new_session"] = True
        process = subprocess.Popen(command, **kwargs)
        try:
            plan = json.loads(plan_path.read_text(encoding="utf-8"))
            ready_path = Path(str(plan.get("readyPath") or ""))
        except (OSError, ValueError, TypeError) as exc:
            process.terminate()
            raise RuntimeError("업데이트 도우미 확인 정보를 읽지 못했습니다.") from exc
        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline:
            if ready_path.is_file():
                try:
                    ready = json.loads(ready_path.read_text(encoding="utf-8"))
                except (OSError, ValueError, TypeError):
                    ready = {}
                if (
                    isinstance(ready, Mapping)
                    and ready.get("ready") is True
                    and int(ready.get("pid") or 0) == process.pid
                    and str(ready.get("targetCommit") or "") == str(plan.get("targetCommit") or "")
                ):
                    return int(process.pid)
            if process.poll() is not None:
                break
            time.sleep(0.05)
        try:
            process.terminate()
            process.wait(timeout=2)
        except (OSError, subprocess.SubprocessError):
            pass
        raise RuntimeError("업데이트 도우미가 준비 상태를 확인하지 못했습니다.")


_manager: AppUpdateManager | None = None
_manager_lock = threading.Lock()


def get_app_update_manager() -> AppUpdateManager:
    global _manager
    if _manager is None:
        with _manager_lock:
            if _manager is None:
                _manager = AppUpdateManager()
    return _manager


__all__ = [
    "AppUpdateError",
    "AppUpdateManager",
    "GitHubReleaseClient",
    "GitSource",
    "get_app_update_manager",
    "normalise_version",
    "parse_version",
]
