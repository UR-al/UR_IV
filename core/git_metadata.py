"""Git 체크아웃 메타데이터를 프로세스 없이 파일에서 읽는다 (Qt·git 실행 파일 불필요).

표시용 정보(현재 커밋·브랜치·원격 URL)만 다룬다. 원격과 통신하거나 작업 트리를 바꾸는
결정(업데이트 설치 등)은 여전히 ``git`` 명령으로 다시 확인해야 한다.

지원하는 배치:
- 일반 체크아웃: ``<root>/.git/`` 폴더
- ``.git`` 파일(``gitdir: ...``) — 서브모듈·``git worktree``. worktree 는 ``commondir`` 가
  가리키는 공용 저장소에 refs·packed-refs·config 가 있으므로 거기까지 찾아본다.
"""
from __future__ import annotations

import configparser
import re
from pathlib import Path
from typing import Optional

_SHA = re.compile(r"[0-9a-fA-F]{40,64}")
_REMOTE_SECTION = re.compile(r'^remote "(.+)"$')


def _is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def git_dir(source: Path) -> Optional[Path]:
    """``source`` 체크아웃의 git 디렉터리(없으면 None)."""
    dot_git = Path(source) / ".git"
    if dot_git.is_dir():
        return dot_git.resolve()
    if not dot_git.is_file():
        return None
    try:
        marker = dot_git.read_text(encoding="utf-8", errors="replace").strip()
        if not marker.casefold().startswith("gitdir:"):
            return None
        candidate = Path(marker.split(":", 1)[1].strip())
        if not candidate.is_absolute():
            candidate = Path(source) / candidate
        candidate = candidate.resolve()
        return candidate if candidate.is_dir() else None
    except OSError:
        return None


def common_dir(directory: Path) -> Path:
    """worktree 의 공용 git 디렉터리(``commondir``), 없으면 자기 자신."""
    try:
        value = (directory / "commondir").read_text(encoding="utf-8", errors="replace").strip()
    except OSError:
        return directory
    if not value:
        return directory
    candidate = Path(value)
    if not candidate.is_absolute():
        candidate = directory / candidate
    try:
        candidate = candidate.resolve()
    except OSError:
        return directory
    return candidate if candidate.is_dir() else directory


def _read_loose_ref(directory: Path, ref: str) -> str:
    loose = (directory / Path(ref)).resolve()
    if not _is_relative_to(loose, directory):
        return ""
    try:
        value = loose.read_text(encoding="ascii", errors="ignore").strip()
    except OSError:
        return ""
    return value.lower() if _SHA.fullmatch(value) else ""


def _read_packed_ref(directory: Path, ref: str) -> str:
    try:
        lines = (directory / "packed-refs").read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return ""
    for line in lines:
        if not line or line.startswith(("#", "^")):
            continue
        commit, _, packed_ref = line.partition(" ")
        if packed_ref.strip() == ref and _SHA.fullmatch(commit):
            return commit.lower()
    return ""


def read_ref(directory: Path, ref: str) -> str:
    """``refs/...`` 의 커밋(소문자 SHA). 없거나 안전하지 않은 이름이면 ""."""
    if not ref.startswith("refs/") or ".." in Path(ref).parts:
        return ""
    for base in dict.fromkeys((directory, common_dir(directory))):
        value = _read_loose_ref(base, ref) or _read_packed_ref(base, ref)
        if value:
            return value
    return ""


def head_info(source: Path) -> tuple[str, str]:
    """(현재 커밋 SHA, 브랜치 이름). 분리 HEAD 면 브랜치는 "". 읽지 못하면 ("", "")."""
    directory = git_dir(source)
    if directory is None:
        return "", ""
    try:
        head = (directory / "HEAD").read_text(encoding="ascii", errors="ignore").strip()
    except OSError:
        return "", ""
    if head.startswith("ref:"):
        ref = head.split(":", 1)[1].strip()
        if not ref.startswith("refs/") or ".." in Path(ref).parts:
            return "", ""
        branch = ref[len("refs/heads/"):] if ref.startswith("refs/heads/") else ""
        return read_ref(directory, ref), branch
    if _SHA.fullmatch(head):
        return head.lower(), ""
    return "", ""


def remote_urls(source: Path) -> dict[str, str]:
    """``[remote "이름"] url`` 목록(설정 파일 순서). 읽지 못하면 {}."""
    directory = git_dir(source)
    if directory is None:
        return {}
    parser = configparser.RawConfigParser(strict=False)
    try:
        parser.read(common_dir(directory) / "config", encoding="utf-8")
    except (OSError, configparser.Error):
        return {}
    urls: dict[str, str] = {}
    for section in parser.sections():
        match = _REMOTE_SECTION.match(section.strip())
        if not match:
            continue
        url = str(parser.get(section, "url", fallback="") or "").strip()
        if url:
            urls[match.group(1)] = url
    return urls


def remote_url(source: Path, name: str = "origin") -> str:
    return remote_urls(source).get(name, "")


__all__ = ["common_dir", "git_dir", "head_info", "read_ref", "remote_url", "remote_urls"]
