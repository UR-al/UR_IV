"""메모 로컬 저장소(user_data/memos.json)와 병합 규칙 — Qt 를 모르는 순수 모듈.

sam-extra 확장의 메모 저장소(<Forge data>/sam-extra/memos.json)와 같은 메모 모양을 쓴다.
메모 = {id, title, text, created_at, updated_at, deleted} + 로컬 전용 ``dirty``(아직 Forge 로
보내지 않은 변경). 파일 모양::

    {"schema_version": 1,
     "memos": [memo...],
     "sync": {"servers": {"<Forge 주소>": {"base": {id: 마지막 동기화 때 본 서버 updated_at},
                                          "tombs": {id: base 판이 삭제 표시일 때 그 updated_at},
                                          "last_synced_at": str|null, "revision": int|null}}}}

``base`` 는 '양쪽이 모두 바뀌었나'를 가르는 기준이다. 서버 주소별로 따로 둔다 — 다른 Forge 에
붙었을 때 남의 서버 기준으로 판단하지 않게. '로컬이 바뀌었나'도 서버마다 따진다: 로컬 updated_at 이
그 서버의 base 와 다르면 그 서버에는 아직 안 간 변경이다(``dirty`` 는 '어느 서버에도 안 보낸 변경'
표시일 뿐이다 — Forge A 에 보내 지워진 dirty 로 Forge B 를 판단하면 B 에는 영영 안 간다).

병합 규칙(공유 메모 계약, :func:`plan_memo_merge`):
  - 한쪽만 바뀌었으면 바뀐 쪽이 이긴다(마지막 기록 우선). 이 서버와 맞춘 기준이 없으면(처음 붙은 서버) 내용이
    다른 로컬 메모는 dirty 가 아니어도 바뀐 것으로 본다 — 다른 Forge 에 보낸 편집을 사본 없이 덮지 않게.
  - 둘 다 바뀌었는데 한쪽이 삭제 표시(tombstone)면 updated_at 이 더 새것(같으면 삭제)이 이긴다.
  - 둘 다 내용이 바뀌었으면 둘 다 남긴다 — 서버 것이 id 를 갖고, 로컬 것은 새 id 로
    ``"<제목> (충돌 사본)"`` 이 된다.
  - 서버가 마지막으로 맞춘 판보다 옛 판이면(.bak 복구 등) 로컬 것을 다시 올린다.
  - 맞췄던 메모가 서버에서 사라졌으면 다시 올린다 — 단, 가득 찬 서버(삭제 표시 정리)면 지운 것으로 본다.

화면의 저장(:meth:`MemoStore.save`)은 ``base_updated_at`` 이 지금 저장본(또는 내용이 같은 옛 시각)이
아니면 낡은 것으로 본다 — 저장본을 덮지 않고 충돌 사본으로 남긴다. 같은 낡은 base 로 다시 오는 저장
(확인이 늦어 다시 보낸 것 · 이어 친 글)은 그 저장이 만든 사본 하나에 모은다 — 같은 편집기(``editor``)의
저장만. 다른 편집기의 글이나 편집기를 모르는 다른 글은 새 사본으로 남긴다(서로의 사본을 덮지 않는다).

파일이 있는데 읽지 못하면(백신·동기화 도구가 잠금) :class:`MemoStoreUnavailableError` 로 멈춘다 —
빈 목록으로 치고 저장하면 파일이 통째로 덮어써진다. JSON 이 깨졌으면 ``.corrupt`` 로 옮겨 두고
빈 저장소로 시작한다. 더 새 schema_version 이면 읽기만 하고 쓰지 않는다. 파일에 못 쓴 변경은 메모리에서도
되돌린다 — 화면이 저장된 것으로 믿고 끝에 잃지 않게.
"""
from __future__ import annotations

import copy
import json
import os
import re
import threading
import uuid
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Optional

from utils.atomic_json import atomic_write_json

SCHEMA_VERSION = 1
MAX_MEMOS = 500                 # 삭제 표시 포함 — 넘치면 모든 서버가 받은 오래된 삭제 표시부터 지운다
MAX_TITLE_LENGTH = 120
MAX_TEXT_LENGTH = 100_000
MAX_SYNC_SERVERS = 8
CONFLICT_SUFFIX = " (충돌 사본)"
MEMO_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,79}$")
PUBLIC_FIELDS = ("id", "title", "text", "created_at", "updated_at")
# 낡은 base 저장이 만든 충돌 사본을 (메모 id, base, 편집기) 별로 기억하는 개수(메모리에만 — 화면 초안과 같은 수명).
STALE_COPY_MEMORY = 64

_EPOCH = datetime(1970, 1, 1, tzinfo=timezone.utc)
# json.loads 는 외톨이 "\ud83d" 이스케이프(이모지 가운데서 잘린 JS 문자열)를 서로게이트 코드 포인트로 만든다 —
# UTF-8 로 쓰지 못해(UnicodeEncodeError) 그 뒤 모든 저장이 실패한다. 확장(notebook_memos)과 같은 규칙.
_LONE_SURROGATE_RE = re.compile("[\ud800-\udfff]")


class MemoValidationError(ValueError):
    """사용자에게 그대로 보여 줄 수 있는 입력 오류(한국어 문구)."""


class MemoStoreUnavailableError(RuntimeError):
    """메모 파일이 있는데 읽지(또는 손상본을 격리하지) 못했다 — 저장하면 파일을 덮어쓴다."""


# ── 시각 ──────────────────────────────────────────────────────────────────

def utc_now_iso() -> str:
    """확장(notebook_store._utc_now)과 같은 모양: ``2026-09-25T01:02:03.456789+00:00``."""
    return datetime.now(timezone.utc).isoformat()


def parse_timestamp(value: Any) -> Optional[datetime]:
    """ISO8601 → UTC datetime. 'Z' 접미·시간대 없는 값(UTC 로 본다)도 받는다. 못 읽으면 None."""
    if not isinstance(value, str) or not value.strip():
        return None
    text = value.strip()
    if text[-1] in "Zz":
        text = text[:-1] + "+00:00"
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def timestamps_equal(left: Any, right: Any) -> bool:
    """같은 순간인가 — 표기 차이('Z' vs '+00:00')는 같게 본다. 못 읽는 값은 글자 그대로 비교."""
    first, second = parse_timestamp(left), parse_timestamp(right)
    if first is not None and second is not None:
        return first == second
    return (left or None) == (right or None)


def is_newer(left: Any, right: Any) -> bool:
    """left 가 right 보다 엄격히 새것인가. 못 읽는 시각은 가장 오래된 것으로 친다."""
    return (parse_timestamp(left) or _EPOCH) > (parse_timestamp(right) or _EPOCH)


def _sort_key(memo: dict) -> tuple:
    return (parse_timestamp(memo.get("updated_at")) or _EPOCH,
            parse_timestamp(memo.get("created_at")) or _EPOCH, memo.get("id", ""))


def _timestamp_key(value: str) -> str:
    """같은 순간을 같은 키로('Z' vs '+00:00'). 못 읽는 값은 글자 그대로."""
    parsed = parse_timestamp(value)
    return parsed.isoformat() if parsed is not None else value


def new_memo_id() -> str:
    return f"memo-{uuid.uuid4().hex}"


# ── 메모 모양 ─────────────────────────────────────────────────────────────

def strip_lone_surrogates(value: str) -> str:
    """외톨이 서로게이트를 U+FFFD 로(확장과 같은 규칙). 짝이 맞는 문자는 그대로."""
    return _LONE_SURROGATE_RE.sub("�", value) if value else value


def sanitize_memo(raw: Any, *, keep_dirty: bool = False) -> Optional[dict]:
    """메모 한 건을 계약 모양으로. id 가 틀렸으면 None(버린다). 글자 수는 자르지 않는다."""
    if not isinstance(raw, dict):
        return None
    memo_id = raw.get("id")
    if not isinstance(memo_id, str) or not MEMO_ID_RE.fullmatch(memo_id):
        return None
    title = raw.get("title")
    text = raw.get("text")
    created_at = raw.get("created_at")
    updated_at = raw.get("updated_at")
    memo = {
        "id": memo_id,
        "title": strip_lone_surrogates(title) if isinstance(title, str) else "",
        "text": strip_lone_surrogates(text) if isinstance(text, str) else "",
        "created_at": created_at if isinstance(created_at, str) else "",
        "updated_at": updated_at if isinstance(updated_at, str) else "",
        "deleted": raw.get("deleted") is True,
    }
    if not memo["created_at"]:
        memo["created_at"] = memo["updated_at"]
    if keep_dirty:
        memo["dirty"] = raw.get("dirty") is True
    return memo


def public_memo(memo: dict) -> dict:
    """Vue 로 보내는 모양(memoState.memos[i])."""
    return {key: memo.get(key, "") for key in PUBLIC_FIELDS}


def memo_fingerprint(memo: Optional[dict]) -> Optional[tuple]:
    """동기화 도중 사용자가 같은 메모를 또 고쳤는지 가르는 값."""
    if memo is None:
        return None
    return (memo.get("updated_at"), memo.get("title"), memo.get("text"),
            bool(memo.get("deleted")), bool(memo.get("dirty")))


def same_content(left: dict, right: dict) -> bool:
    if bool(left.get("deleted")) and bool(right.get("deleted")):
        return True
    return (bool(left.get("deleted")) == bool(right.get("deleted"))
            and left.get("title") == right.get("title")
            and left.get("text") == right.get("text"))


def conflict_title(title: str) -> str:
    room = MAX_TITLE_LENGTH - len(CONFLICT_SUFFIX)
    return f"{(title or '')[:room]}{CONFLICT_SUFFIX}"


# ── 병합 판단 ─────────────────────────────────────────────────────────────

NOOP = "noop"
ADOPT_REMOTE = "adopt_remote"      # 서버 것을 로컬에(깨끗한 상태로)
PUSH = "push"                      # 로컬 것을 PUT
PUSH_DELETE = "push_delete"        # 로컬 삭제를 DELETE
MARK_CLEAN = "mark_clean"          # 서버에 없는 로컬 삭제 표시 — 보낼 것 없음
CONFLICT_COPY = "conflict_copy"    # 서버 것이 id 를 갖고 로컬 것은 충돌 사본으로
ADOPT_DELETE = "adopt_delete"      # 서버가 삭제 표시까지 정리(prune)한 메모 — 로컬도 삭제 표시로


def plan_memo_merge(local: Optional[dict], remote: Optional[dict], base: Optional[str], *,
                    remote_may_prune: bool = False) -> str:
    """메모 한 건의 병합 동작. ``base`` = 이 서버와 마지막으로 맞췄을 때 본 서버 updated_at.

    로컬이 '이 서버와 맞춘 뒤 바뀌었나'는 서버마다 따진다 — 어느 서버에도 안 보냈거나(dirty),
    로컬 updated_at 이 이 서버의 base 와 다르면(다른 Forge 에서 들였거나 그리로 보낸 판) 바뀐 것이다.
    맞출 때마다 로컬과 서버의 updated_at 은 같아진다(서버는 클라이언트 시각을 쓰고, 고쳐 돌려주면 로컬도
    그 판으로 바뀐다) — 그래서 시각 비교로 충분하다.

    ``remote_may_prune`` — 서버 저장소가 오래된 삭제 표시를 지웠을 수 있다
    (:meth:`MemoStore.remote_may_have_pruned`).
    """
    if local is None:
        if remote is None or remote.get("deleted"):
            return NOOP              # 모르는 메모의 삭제 표시는 들일 필요가 없다
        return ADOPT_REMOTE
    if base is None:
        # 이 서버와 맞춘 적이 없다(처음 붙은 서버 · 주소가 바뀐 같은 설치 · 기록이 밀려난 서버). dirty 가 풀린
        # 메모도 다른 Forge 에 보낸 판일 수 있다 — 내용이 다르면 바뀐 것으로 보고 병합 규칙으로(사본 없이
        # 서버 판으로 바꾸면 그 편집을 잃고, 다음 동기화가 그 서버에까지 서버 판을 올린다).
        # 깨끗한 삭제 표시는 예외 — 잃을 내용이 없고, 서버에서 사라진 메모를 따라 지운 것(ADOPT_DELETE)은 따라
        # 지운 시각이 찍혀 다른 기기가 다시 올린 그 전 편집보다 늘 새것이다. 바뀐 것으로 보면 그 편집을 사본 없이
        # 서버에서 지우고, 그 삭제가 다른 기기까지 번진다.
        local_changed = bool(local.get("dirty")) or (
            remote is not None and not local.get("deleted") and not same_content(local, remote))
    else:
        local_changed = bool(local.get("dirty")) or not timestamps_equal(local.get("updated_at"), base)
    if remote is None:
        if local.get("deleted"):
            return MARK_CLEAN if local.get("dirty") else NOOP
        if remote_may_prune and base is not None and not local_changed:
            # 이 서버와 맞췄던, 그 뒤 안 고친 메모가 가득 찬 서버에서 사라졌다 — 다른 곳에서 지운 뒤
            # 그 삭제 표시가 정리된 것이다. 다시 올리면 지운 메모가 되살아난다(삭제 표시가 이긴다).
            return ADOPT_DELETE
        # 새 메모, 처음 붙은 서버, 또는 서버가 잃어버린 메모 — 내용을 잃지 않게 올린다.
        return PUSH
    if base is not None and is_newer(base, remote.get("updated_at")):
        # 서버가 마지막으로 맞춘 판보다 옛 판으로 돌아갔다(손상 뒤 .bak 복구 등). 마지막 기록 우선 —
        # 더 새 로컬 것을 다시 올린다. 옛 판을 들이면 그 뒤의 편집을 양쪽에서 잃는다.
        if local.get("deleted") and remote.get("deleted"):
            return MARK_CLEAN if local.get("dirty") else NOOP
        return PUSH_DELETE if local.get("deleted") else PUSH
    remote_changed = base is None or not timestamps_equal(remote.get("updated_at"), base)
    if not local_changed:
        # 같은 내용이어도 들인다 — 그래야 이 서버 기준(base)이 서버 시각으로 맞춰진다.
        return ADOPT_REMOTE if remote_changed else NOOP
    if not remote_changed:
        return PUSH_DELETE if local.get("deleted") else PUSH
    # 양쪽 모두 바뀌었다.
    if same_content(local, remote):
        return ADOPT_REMOTE
    if bool(local.get("deleted")) != bool(remote.get("deleted")):
        tomb, live = (local, remote) if local.get("deleted") else (remote, local)
        tomb_wins = not is_newer(live.get("updated_at"), tomb.get("updated_at"))
        if local.get("deleted"):
            return PUSH_DELETE if tomb_wins else ADOPT_REMOTE
        return ADOPT_REMOTE if tomb_wins else PUSH
    return CONFLICT_COPY


# ── 저장소 ────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class SaveResult:
    memo: dict               # 저장된 메모(public 모양) — 충돌이면 충돌 사본
    changed: bool            # 파일이 바뀌었나(같은 내용 재저장이면 False)
    conflict: bool = False   # base 가 낡아 충돌 사본으로 저장했나
    conflict_of: str = ""    # 충돌 사본의 원래 id


def _empty_data() -> dict:
    return {"schema_version": SCHEMA_VERSION, "memos": [], "sync": {"servers": {}}}


def _corrupt_backup_path(target: str) -> str:
    candidate = target + ".corrupt"
    index = 1
    while os.path.lexists(candidate):
        candidate = f"{target}.corrupt.{index}"
        index += 1
    return candidate


def _clean_server_state(raw: Any) -> dict:
    raw = raw if isinstance(raw, dict) else {}
    base = raw.get("base") if isinstance(raw.get("base"), dict) else {}
    revision = raw.get("revision")
    last = raw.get("last_synced_at")
    tombs = raw.get("tombs") if isinstance(raw.get("tombs"), dict) else {}
    base = {k: v for k, v in base.items()
            if isinstance(k, str) and MEMO_ID_RE.fullmatch(k) and isinstance(v, str)}
    return {
        "base": base,
        # base 판이 삭제 표시인 id(→ 그 판의 updated_at) — 이 서버가 삭제를 받았나(:meth:`MemoStore._make_room`)
        "tombs": {k: v for k, v in tombs.items() if k in base and isinstance(v, str)},
        "last_synced_at": last if isinstance(last, str) and last else None,
        "revision": revision if isinstance(revision, int) and not isinstance(revision, bool) else None,
    }


def _drop_base(state: dict, memo_id: str) -> None:
    state["base"].pop(memo_id, None)
    state.setdefault("tombs", {}).pop(memo_id, None)


class MemoStore:
    """스레드 안전한 메모 저장소. 모든 변경은 잠금 안에서 메모리 → 파일(원자적 쓰기) 순서다."""

    def __init__(self, path: str | os.PathLike | None = None, *,
                 clock: Callable[[], str] = utc_now_iso,
                 id_factory: Callable[[], str] = new_memo_id) -> None:
        self._path = Path(path) if path is not None else None
        self._clock = clock
        self._new_id = id_factory
        self._lock = threading.RLock()
        self._data: Optional[dict] = None
        self._read_only = False
        # 내용은 그대로인데 시각만 바뀐 경우(서버가 시계 차이로 updated_at 을 고쳐 돌려줌)의 옛 시각.
        # 화면이 옛 시각을 base 로 보내도 충돌 사본을 만들지 않게 한다(메모리에만 둔다).
        self._aliases: dict[str, set] = {}
        # (메모 id, 낡은 base, 편집기 — 모르면 "") → (그 저장이 만든 충돌 사본 id, 사본에 마지막으로 쓴 updated_at).
        self._stale_copies: dict[tuple[str, str, str], tuple[str, str]] = {}

    # ── 파일 ──
    @property
    def path(self) -> Path:
        if self._path is None:
            from core.storage_paths import user_data_file
            self._path = user_data_file("memos.json")
        return self._path

    def _quarantine(self, target: str) -> None:
        try:
            os.replace(target, _corrupt_backup_path(target))
        except FileNotFoundError:
            return
        except OSError as exc:
            raise MemoStoreUnavailableError(
                "메모 파일이 손상됐는데 백업(.corrupt)으로 옮기지 못했습니다") from exc

    def _read_file(self) -> dict:
        target = os.fspath(self.path)
        try:
            with open(target, "r", encoding="utf-8-sig") as fh:
                raw = json.load(fh)
        except FileNotFoundError:
            return _empty_data()
        except (json.JSONDecodeError, UnicodeDecodeError):
            self._quarantine(target)
            return _empty_data()
        except OSError as exc:
            raise MemoStoreUnavailableError(
                f"메모 파일을 읽지 못했습니다({exc.__class__.__name__}). 잠시 후 다시 시도하세요") from exc
        if not isinstance(raw, dict) or not isinstance(raw.get("memos", []), list):
            self._quarantine(target)
            return _empty_data()
        version = raw.get("schema_version", SCHEMA_VERSION)
        self._read_only = not isinstance(version, int) or version > SCHEMA_VERSION
        memos, seen = [], set()
        for item in raw.get("memos", []):
            memo = sanitize_memo(item, keep_dirty=True)
            if memo is None or memo["id"] in seen:
                continue
            seen.add(memo["id"])
            memos.append(memo)
        sync = raw.get("sync") if isinstance(raw.get("sync"), dict) else {}
        servers = sync.get("servers") if isinstance(sync.get("servers"), dict) else {}
        return {
            "schema_version": SCHEMA_VERSION,
            "memos": memos,
            "sync": {"servers": {key: _clean_server_state(value)
                                 for key, value in servers.items() if isinstance(key, str) and key}},
        }

    def _load(self) -> dict:
        if self._data is None:
            self._data = self._read_file()
        return self._data

    def _persist(self) -> None:
        if self._read_only:
            raise MemoStoreUnavailableError(
                "메모 파일이 더 새 버전의 앱에서 저장됐습니다 — 이 버전에서는 바꾸지 않습니다")
        # 들여쓰기 없이 — 자동 저장(입력이 멈출 때마다)이 통째로 다시 쓰는 파일이다.
        atomic_write_json(self.path, self._data, indent=None, separators=(",", ":"))

    def _snapshot(self) -> tuple:
        """변경 전 모습 — 파일에 못 쓰면 이것으로 되돌린다.

        메모 dict 는 제자리에서 바뀌므로 하나씩 얕게 복사한다(글자열은 공유 — 큰 본문도 싸다).
        """
        data = self._load()
        servers = {key: {k: (dict(v) if isinstance(v, dict) else v) for k, v in state.items()}
                   for key, state in data["sync"]["servers"].items()}
        return ([dict(m) for m in data["memos"]], servers,
                {key: set(value) for key, value in self._aliases.items()})

    def _commit(self, snapshot: tuple) -> None:
        """파일에 쓴다. 못 쓰면(잠금·디스크·읽기 전용) 메모리도 변경 전으로 되돌리고 예외를 올린다 —
        화면이 저장된 것으로 보고 넘어가면 그 편집은 앱을 닫을 때 사라진다(화면은 확인이 없으면 다시 보낸다)."""
        try:
            self._persist()
        except Exception:
            memos, servers, aliases = snapshot
            if self._data is not None:
                self._data["memos"] = memos
                self._data["sync"]["servers"] = servers
            self._aliases = aliases
            raise

    def reload(self) -> None:
        """다음 접근 때 파일을 다시 읽는다(설정 복원 등 밖에서 파일이 바뀐 뒤)."""
        with self._lock:
            self._data = None
            self._read_only = False
            self._aliases.clear()
            self._stale_copies.clear()

    def flush(self) -> None:
        """persist=False 로 모은 동기화 변경을 쓴다. 못 쓰면 예외만 올린다 — 서버에서 온 판이라 다음
        쓰기(또는 다음 동기화)가 다시 맞춘다."""
        with self._lock:
            if self._data is not None:
                self._persist()

    # ── 조회 ──
    def _find(self, memo_id: str) -> Optional[dict]:
        for memo in self._load()["memos"]:
            if memo["id"] == memo_id:
                return memo
        return None

    def list_public(self) -> list[dict]:
        """삭제 안 된 메모, 최근 수정 순."""
        with self._lock:
            live = [m for m in self._load()["memos"] if not m.get("deleted")]
            return [public_memo(m) for m in sorted(live, key=_sort_key, reverse=True)]

    def all_memos(self) -> list[dict]:
        """삭제 표시·dirty 까지 포함한 사본."""
        with self._lock:
            return copy.deepcopy(self._load()["memos"])

    def get(self, memo_id: str) -> Optional[dict]:
        with self._lock:
            memo = self._find(memo_id)
            return copy.deepcopy(memo) if memo is not None else None

    def has_dirty(self) -> bool:
        with self._lock:
            return any(m.get("dirty") for m in self._load()["memos"])

    def _is_current_version(self, memo: dict, stamp: Any) -> bool:
        """stamp 가 이 메모의 지금 판인가 — updated_at, 또는 내용이 같은 옛 시각(별칭)."""
        known = (memo.get("updated_at"), *self._aliases.get(memo["id"], ()))
        return any(timestamps_equal(k, stamp) for k in known)

    # ── 시각 ──
    def _stamp_after(self, *previous: Any) -> str:
        """지금 시각. 시계가 뒤로 갔어도 이전 updated_at 보다는 늦게(마지막 기록 우선이 뒤집히지 않게)."""
        now = self._clock()
        latest = max((parse_timestamp(p) or _EPOCH for p in previous), default=_EPOCH)
        parsed = parse_timestamp(now)
        if parsed is None or parsed <= latest:
            return (latest + timedelta(milliseconds=1)).isoformat()
        return now

    # ── 로컬 편집 ──
    def save(self, memo_id: Any, title: Any, text: Any,
             base_updated_at: Any = None, *, editor: Any = None) -> SaveResult:
        """메모 저장(없으면 만든다). base_updated_at 이 지금 판이 아니면 충돌 사본으로 따로 남긴다.

        낡았다 = 저장본의 updated_at(또는 내용이 같은 옛 시각)이 아니다 — 저장본이 더 새것인지는 따지지
        않는다. 동기화가 편집 중인 메모를 서버 판으로 바꾸면(충돌 · 다른 Forge 의 판) 그 서버 판이 화면의
        base 보다 옛 시각일 수 있고, 그때 덮으면 서버 판이 앱과 Forge 양쪽에서 사라진다.

        ``editor`` = 저장을 보낸 편집기(화면 인스턴스) 식별자. 같은 낡은 base 의 이어 친 글을 그 편집기가 만든
        사본 하나에 모으는 데만 쓴다 — 다른 편집기(웹 클라이언트 둘 · 탭 둘)의 글은 서로의 사본을 덮지 않는다.
        """
        if not isinstance(memo_id, str) or not MEMO_ID_RE.fullmatch(memo_id):
            raise MemoValidationError("메모 id 형식이 올바르지 않습니다")
        editor = editor if isinstance(editor, str) and 0 < len(editor) <= 80 else None
        title = strip_lone_surrogates("" if title is None else str(title))
        text = strip_lone_surrogates("" if text is None else str(text))
        if len(text) > MAX_TEXT_LENGTH:
            raise MemoValidationError(f"메모 본문은 {MAX_TEXT_LENGTH:,}자까지 저장할 수 있습니다")
        title = title[:MAX_TITLE_LENGTH]
        base = base_updated_at if isinstance(base_updated_at, str) and base_updated_at else None
        with self._lock:
            data = self._load()
            stored = self._find(memo_id)
            if stored is None:
                snapshot = self._snapshot()
                self._make_room(adding=1)
                now = self._stamp_after()
                memo = {"id": memo_id, "title": title, "text": text, "created_at": now,
                        "updated_at": now, "deleted": False, "dirty": True}
                data["memos"].append(memo)
                self._commit(snapshot)
                return SaveResult(public_memo(memo), True)
            stale = base is not None and not self._is_current_version(stored, base)
            if stale and not stored.get("deleted"):
                if stored.get("title") == title and stored.get("text") == text:
                    return SaveResult(public_memo(stored), False)
                return self._save_stale_locked(memo_id, base, title, text, editor)
            if not stored.get("deleted") and stored.get("title") == title and stored.get("text") == text:
                return SaveResult(public_memo(stored), False)
            # 내용이 바뀌었거나 삭제된 메모를 되살린다(지금 쓴 것이 가장 새 기록이다).
            snapshot = self._snapshot()
            stored.update(title=title, text=text, deleted=False, dirty=True,
                          updated_at=self._stamp_after(stored.get("updated_at")))
            self._aliases.pop(memo_id, None)
            self._commit(snapshot)
            return SaveResult(public_memo(stored), True)

    def _save_stale_locked(self, memo_id: str, base: str, title: str, text: str,
                           editor: Optional[str]) -> SaveResult:
        """낡은 base 의 저장 — 저장본은 두고 충돌 사본에 쓴다.

        같은 (메모, base, 편집기) 로 다시 오는 저장은 그 저장이 만든 사본 하나에 모은다: 화면은 확인이 늦으면
        같은 저장을 그대로 다시 보내고(같은 내용 → 그대로 돌려준다), 그 사이 더 친 글도 같은 base 로 온다(사본을
        고친다). 사본이 그 뒤 따로 고쳐졌으면(화면이 사본으로 옮겨 가 편집) 덮지 않고 새 사본을 만든다.
        편집기를 모르면 이어 친 글인지 다른 편집기의 글인지 가를 수 없다 — 다른 글은 늘 새 사본으로(덮지 않는다).
        """
        key = (memo_id, _timestamp_key(base), editor or "")
        entry = self._stale_copies.get(key)
        target = self._find(entry[0]) if entry else None
        copy_title = conflict_title(title)
        if target is not None and not target.get("deleted"):
            if target.get("title") == copy_title and target.get("text") == text:
                self._remember_stale_copy(key, target, entry[1])
                return SaveResult(public_memo(target), False, conflict=True, conflict_of=memo_id)
            if editor and self._is_current_version(target, entry[1]):
                snapshot = self._snapshot()
                target.update(title=copy_title, text=text, dirty=True,
                              updated_at=self._stamp_after(target.get("updated_at")))
                self._aliases.pop(target["id"], None)
                self._commit(snapshot)
                self._remember_stale_copy(key, target)
                return SaveResult(public_memo(target), True, conflict=True, conflict_of=memo_id)
        snapshot = self._snapshot()
        copy_memo = self._add_copy_locked(title, text)
        self._commit(snapshot)
        self._remember_stale_copy(key, copy_memo)
        return SaveResult(public_memo(copy_memo), True, conflict=True, conflict_of=memo_id)

    def _remember_stale_copy(self, key: tuple[str, str, str], memo: dict,
                             written_at: Optional[str] = None) -> None:
        self._stale_copies.pop(key, None)
        self._stale_copies[key] = (memo["id"], written_at or memo["updated_at"])
        while len(self._stale_copies) > STALE_COPY_MEMORY:
            self._stale_copies.pop(next(iter(self._stale_copies)))

    def delete(self, memo_id: Any) -> Optional[dict]:
        """삭제 표시(tombstone)로 바꾼다. 모르는 id·이미 삭제된 메모면 None."""
        if not isinstance(memo_id, str):
            return None
        with self._lock:
            stored = self._find(memo_id)
            if stored is None or stored.get("deleted"):
                return None
            snapshot = self._snapshot()
            stored.update(text="", deleted=True, dirty=True,
                          updated_at=self._stamp_after(stored.get("updated_at")))
            self._aliases.pop(memo_id, None)
            self._commit(snapshot)
            return copy.deepcopy(stored)

    def _add_copy_locked(self, title: str, text: str) -> dict:
        now = self._stamp_after()
        memo_id = self._new_id()
        while self._find(memo_id) is not None:
            memo_id = self._new_id()
        memo = {"id": memo_id, "title": conflict_title(title), "text": text, "created_at": now,
                "updated_at": now, "deleted": False, "dirty": True}
        # 충돌 사본은 사용자의 글이다 — 칸이 모자라도 거절하지 않는다(보낸 삭제 표시만 정리).
        self._make_room(adding=1, strict=False)
        self._load()["memos"].append(memo)
        return memo

    def _make_room(self, *, adding: int, strict: bool = True) -> None:
        """MAX_MEMOS 를 넘지 않게, 기억하는 모든 서버가 받은 삭제 표시를 오래된 것부터 지운다.

        지우면 안 되는 삭제 표시 — 지우면 다음 동기화가 그 서버의 살아 있는 판을 모르는 메모로 보고 들여,
        지운 메모가 되살아난다:
          - 아직 어디에도 안 보낸 것(dirty). 한도에 센다 — strict 면 칸이 모자랄 때 아무것도 지우지 않고 거절한다.
          - 한 서버에는 보냈지만 그 메모를 아는 다른 서버가 아직 못 받은 것(:meth:`_tomb_unsettled`).
            한도에 세지 않는다 — 한동안 안 붙은 서버 때문에 새 메모가 막히지 않게. 그 서버가 알던 메모 수를
            넘지 않고, 받으면(또는 서버 기록이 밀려나면) 정리 대상이 된다.
        """
        data = self._load()
        memos = data["memos"]
        if len(memos) + adding - MAX_MEMOS <= 0:
            return   # 붙잡힌(held) 삭제 표시는 넘침을 줄이기만 한다 — 칸이 남으면 서버 기록을 훑을 필요가 없다
        servers = list(data["sync"]["servers"].values())
        clean_tombs = [m for m in memos if m.get("deleted") and not m.get("dirty")]
        held = {m["id"] for m in clean_tombs if self._tomb_unsettled(m, servers)}
        overflow = len(memos) - len(held) + adding - MAX_MEMOS
        if overflow <= 0:
            return
        candidates = sorted((m for m in clean_tombs if m["id"] not in held), key=_sort_key)
        doomed = {m["id"] for m in candidates[:overflow]}
        if strict and len(doomed) < overflow:
            raise MemoValidationError(f"메모는 {MAX_MEMOS}개까지 저장할 수 있습니다 — 안 쓰는 메모를 지워 주세요")
        if not doomed:
            return
        data["memos"] = [m for m in memos if m["id"] not in doomed]
        for state in servers:
            for memo_id in doomed:
                _drop_base(state, memo_id)

    @staticmethod
    def _tomb_unsettled(memo: dict, servers: list[dict]) -> bool:
        """이 삭제 표시를 아직 못 받은 서버가 있나 — 그 메모를 아는(base 가 있는) 서버 중 base 판이 이 삭제
        표시도, 서버가 돌려준 삭제 표시(``tombs``)도 아닌 곳. 서버마다 삭제 시각을 따로 찍으므로(확장은 받은 시각)
        두 서버에 보낸 삭제 표시는 로컬 시각이 한쪽 base 와만 같다 — 그래서 ``tombs`` 로 가른다."""
        memo_id = memo["id"]
        for state in servers:
            seen = state["base"].get(memo_id)
            if seen is None or timestamps_equal(seen, memo.get("updated_at")):
                continue
            tomb = state.get("tombs", {}).get(memo_id)
            if tomb is not None and timestamps_equal(tomb, seen):
                continue
            return True
        return False

    # ── 동기화 지원(core.memo_sync 가 부른다) ──
    def _server(self, server: str) -> dict:
        servers = self._load()["sync"]["servers"]
        state = servers.get(server)
        if state is None:
            state = servers[server] = _clean_server_state({})
            # 오래전에 붙었던 서버 기록부터 버린다(마지막 동기화 시각 순).
            while len(servers) > MAX_SYNC_SERVERS:
                oldest = min((k for k in servers if k != server),
                             key=lambda k: parse_timestamp(servers[k].get("last_synced_at")) or _EPOCH)
                servers.pop(oldest)
        return state

    def _set_base(self, server: str, memo: dict) -> None:
        """이 서버와 맞춘 판을 적는다 — 그 판이 삭제 표시면 ``tombs`` 에도(서버가 삭제를 받았다는 표시)."""
        state = self._server(server)
        state["base"][memo["id"]] = memo["updated_at"]
        tombs = state.setdefault("tombs", {})
        if memo.get("deleted"):
            tombs[memo["id"]] = memo["updated_at"]
        else:
            tombs.pop(memo["id"], None)

    def sync_bases(self, server: str) -> dict:
        with self._lock:
            return dict(self._load()["sync"]["servers"].get(server, {}).get("base", {}))

    def last_synced_at(self, server: str) -> Optional[str]:
        with self._lock:
            state = self._load()["sync"]["servers"].get(server)
            return state.get("last_synced_at") if state else None

    def _replace_locked(self, memo: dict) -> None:
        memos = self._load()["memos"]
        for index, current in enumerate(memos):
            if current["id"] == memo["id"]:
                self._note_replacement(current, memo)
                memos[index] = memo
                return
        self._aliases.pop(memo["id"], None)
        memos.append(memo)

    def _note_replacement(self, current: dict, memo: dict) -> None:
        """내용이 같고 시각만 바뀌면 옛 시각을 같은 판본의 별칭으로 기억한다(최대 8개)."""
        if not same_content(current, memo) or current.get("deleted") or memo.get("deleted"):
            self._aliases.pop(memo["id"], None)
            return
        if timestamps_equal(current.get("updated_at"), memo.get("updated_at")):
            return
        aliases = self._aliases.setdefault(memo["id"], set())
        aliases.add(current.get("updated_at"))
        while len(aliases) > 8:
            aliases.pop()

    def adopt_remote(self, server: str, remote: dict, *, expected: Optional[dict],
                     persist: bool = True) -> bool:
        """서버 것을 로컬로(깨끗한 상태). 그사이 사용자가 고쳤으면(expected 와 다르면) 손대지 않는다."""
        clean = sanitize_memo(remote)
        if clean is None:
            return False
        with self._lock:
            if memo_fingerprint(self._find(clean["id"])) != memo_fingerprint(expected):
                return False
            snapshot = self._snapshot() if persist else None
            clean["dirty"] = False
            self._replace_locked(clean)
            self._set_base(server, clean)
            self._make_room(adding=0, strict=False)
            if persist:
                self._commit(snapshot)
            return True

    def record_push(self, server: str, remote: dict, *, expected: Optional[dict],
                    persist: bool = True) -> bool:
        """PUT/DELETE 성공. 서버 기준은 늘 옮기고, 그사이 로컬이 안 바뀌었으면 서버 응답으로 맞춘다.

        보내는 동안 사용자가 또 고쳤으면 그 변경은 dirty 로 남아 다음 동기화가 이 기준 위에 올린다.
        """
        clean = sanitize_memo(remote)
        if clean is None:
            return False
        with self._lock:
            snapshot = self._snapshot() if persist else None
            self._set_base(server, clean)
            current = self._find(clean["id"])
            applied = memo_fingerprint(current) == memo_fingerprint(expected)
            if applied:
                clean["dirty"] = False
                self._replace_locked(clean)
            if persist:
                self._commit(snapshot)
            return applied

    def mark_clean(self, memo_id: str, *, expected: Optional[dict], persist: bool = True) -> bool:
        with self._lock:
            current = self._find(memo_id)
            if current is None or memo_fingerprint(current) != memo_fingerprint(expected):
                return False
            snapshot = self._snapshot() if persist else None
            current["dirty"] = False
            if persist:
                self._commit(snapshot)
            return True

    def remote_may_have_pruned(self, server: str, *, remote_count: int,
                               remote_revision: Optional[int]) -> bool:
        """서버가 오래된 삭제 표시를 지웠을 수 있나(:data:`ADOPT_DELETE` 의 조건).

        확장은 저장소가 MAX_MEMOS(삭제 표시 포함)를 넘을 때만 정리하고, 정리한 뒤에는 늘 가득 차 있다 —
        덜 찼으면 사라진 메모는 지운 것이 아니라 서버가 잃어버린 것이다(데이터 폴더를 새로 만든 경우 등).
        revision 이 지난 동기화보다 작으면 저장소가 새로 만들어졌거나 옛 판(.bak)으로 돌아간 것이라
        역시 잃어버린 쪽으로 본다.
        """
        if remote_count < MAX_MEMOS or not isinstance(remote_revision, int):
            return False
        with self._lock:
            state = self._load()["sync"]["servers"].get(server)
            last = state.get("revision") if state else None
        return isinstance(last, int) and remote_revision >= last

    def adopt_remote_delete(self, server: str, memo_id: str, *, expected: Optional[dict],
                            persist: bool = True) -> bool:
        """삭제 표시까지 정리돼 서버에서 사라진 메모를 로컬에서도 (깨끗한) 삭제 표시로.

        그사이 사용자가 고쳤으면(expected 와 다르면) 손대지 않는다 — 다음 동기화가 그 편집을 올린다.
        """
        with self._lock:
            current = self._find(memo_id)
            if current is None or memo_fingerprint(current) != memo_fingerprint(expected):
                return False
            snapshot = self._snapshot() if persist else None
            current.update(text="", deleted=True, dirty=False,
                           updated_at=self._stamp_after(current.get("updated_at")))
            self._aliases.pop(memo_id, None)
            # 서버에 없는 메모다 — 기준을 지워야 다른 기기가 다시 올리면 새 메모로 들인다.
            _drop_base(self._server(server), memo_id)
            if persist:
                self._commit(snapshot)
            return True

    def resolve_conflict(self, server: str, remote: dict, *, persist: bool = True) -> Optional[dict]:
        """둘 다 바뀐 메모: 서버 것이 id 를 갖고, 지금 로컬 내용은 충돌 사본(새 id, dirty)으로.

        반환: 만든 충돌 사본(사본이 필요 없으면 None — 로컬도 삭제됐거나 내용이 같을 때).
        dirty 가 아니어도 사본을 만든다 — 다른 Forge 로 이미 보낸 판(깨끗함)도 이 서버에는 처음이다.
        편집 중이던 화면이 옛 base 로 이어 저장하면 :meth:`save` 가 낡은 저장으로 보고 사본을 따로 만든다
        (서버 판이 로컬 편집보다 옛 시각이어도 덮지 않는다).

        계획한 뒤(다른 메모를 보내는 동안) 지웠으면 :func:`plan_memo_merge` 와 같은 규칙 — 삭제 표시가 서버
        편집보다 새것(같으면 삭제)이면 삭제 표시를 지키고 서버 판을 기준으로만 적는다. 서버 판을 들이면 지운
        메모가 되살아나고 dirty 도 풀려 삭제가 영영 안 간다. 삭제는 core.memo_sync 가 곧바로 보낸다
        (못 보내도 로컬 판이 이 기준과 달라 다음 동기화가 보낸다).
        """
        clean = sanitize_memo(remote)
        if clean is None:
            return None
        with self._lock:
            snapshot = self._snapshot() if persist else None
            current = self._find(clean["id"])
            if (current is not None and current.get("deleted") and not clean.get("deleted")
                    and not is_newer(clean.get("updated_at"), current.get("updated_at"))):
                self._set_base(server, clean)
                if persist:
                    self._commit(snapshot)
                return None
            copy_memo = None
            if (current is not None and not current.get("deleted")
                    and not same_content(current, clean)):
                copy_memo = self._add_copy_locked(current.get("title", ""), current.get("text", ""))
            clean["dirty"] = False
            self._replace_locked(clean)
            self._set_base(server, clean)
            if persist:
                self._commit(snapshot)
            return copy.deepcopy(copy_memo) if copy_memo is not None else None

    def finish_sync(self, server: str, *, revision: Optional[int], synced_at: Optional[str] = None,
                    persist: bool = True) -> str:
        with self._lock:
            snapshot = self._snapshot() if persist else None
            state = self._server(server)
            state["last_synced_at"] = synced_at or self._clock()
            state["revision"] = revision if isinstance(revision, int) else None
            self._make_room(adding=0, strict=False)
            if persist:
                self._commit(snapshot)
            return state["last_synced_at"]


def iter_ids(*groups: Iterable[dict]) -> list[str]:
    """여러 메모 목록의 id 합집합(처음 본 순서 유지)."""
    seen: dict[str, None] = {}
    for group in groups:
        for memo in group:
            seen.setdefault(memo["id"], None)
    return list(seen)


__all__ = [
    "ADOPT_DELETE", "ADOPT_REMOTE", "CONFLICT_COPY", "CONFLICT_SUFFIX", "MARK_CLEAN", "MAX_MEMOS",
    "MAX_TEXT_LENGTH", "MAX_TITLE_LENGTH", "MEMO_ID_RE", "MemoStore",
    "MemoStoreUnavailableError", "MemoValidationError", "NOOP", "PUSH", "PUSH_DELETE",
    "SaveResult", "conflict_title", "is_newer", "iter_ids", "memo_fingerprint",
    "new_memo_id", "parse_timestamp", "plan_memo_merge", "public_memo", "same_content",
    "sanitize_memo", "strip_lone_surrogates", "timestamps_equal", "utc_now_iso",
]
