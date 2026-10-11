# core/related_tags_service.py
"""연관 태그 — 질의·표 만들기 조율(Qt 없음, 의존성 주입). ui/related_tags_actions.py 가 브리지에 잇는다.

질의(related_tags_query): 워커 스레드 하나가 화면(client)마다 가장 새 요청만 처리한다(입력 중 쌓인 그 화면의
  옛 요청은 버린다 — 웹 모드의 다른 탭 요청은 버리지 않는다). 기다리는 화면이 여럿이면 먼저 온 순서로.
  표·어휘는 처음 쓸 때 읽고 등급 조합마다 기억한다(표를 새로 만들면 잊는다).
  어휘의 원본(자동완성 카탈로그·별칭·함의 파일)이 바뀌었으면 쓸 수 있는 표도 stale 로 알려, '다시 만들기'가
  어휘부터 새로 만든다.
표 만들기(related_tags_build): 한 번에 하나, 백그라운드 스레드. 사용자가 버튼을 눌렀을 때만 돈다.
  샤드마다 원자적으로 저장하므로 끊겨도 다음에 남은 샤드부터 만든다. 배치 사이에서 취소할 수 있다.
  진행은 0.5초에 한 번만 보낸다.

이벤트(emit 에 넘기는 dict — 브리지가 JSON 으로 relatedTagsEvent 에 싣는다):
  {action:'related_tags_query', requestId, state, tags, context, ratings, missing, stale, unavailable,
   estimateSec, build, error?}
  {action:'related_tags_build', requestId, state:'building'|'done'|'canceled'|'error', progress?, error?}
"""
from __future__ import annotations

import hashlib
import threading
import time
from pathlib import Path
from typing import Callable, Iterable

from core.dataset_artifacts import read_manifest, stale_artifacts
from core.related_tags import (
    FORMAT_VERSION, GENERIC_SHARE, MAX_CONTEXT, MIN_PAIR_QUERY, QueryRequest,
    extract_prompt_tags, implied_closure, is_person_count_tag, rank_related, rating_shards, resolve,
    select_context,
)
from core.related_tags_store import (
    MISSING, READY, STALE, UNAVAILABLE, USABLE_STATES, LoadedTables, artifact_identity, build_params, cache_paths,
    estimate_seconds, shard_artifact, shard_states,
)

QUERY_ACTION = "related_tags_query"
BUILD_ACTION = "related_tags_build"
PROGRESS_INTERVAL = 0.5
MAX_PENDING_CLIENTS = 16   # 기다리는 질의를 둘 화면 수 — 넘으면 가장 오래 기다린 것을 버린다
NO_DATA_MESSAGE = "Danbooru 데이터가 없어 표를 만들 수 없습니다. 데이터를 받은 뒤 다시 시도하세요."


def _user_message(exc: BaseException) -> str:
    from core.error_handler import sanitize_for_ui

    return sanitize_for_ui(str(exc) or type(exc).__name__, 300)


def _same_source(stored: object, current: dict) -> bool:
    """어휘에 적힌 원본과 지금 원본이 같은가 — 지금 쪽의 키가 하나라도 다르거나 없으면 다르다."""
    return isinstance(stored, dict) and all(stored.get(key) == value for key, value in current.items())


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


class RelatedTagsService:
    def __init__(self, emit: Callable[[dict], None], *, data_dir: Path | str, storage=None, tag_db=None,
                 korean=None, clock: Callable[[], float] = time.monotonic,
                 start_thread: Callable[[Callable[[], None], str], None] | None = None,
                 log: Callable[[str], None] = print, progress_interval: float = PROGRESS_INTERVAL,
                 idle_seconds: float = 30.0):
        self._emit_raw = emit
        self._data_dir = Path(data_dir)
        self._storage = storage
        self._tag_db = tag_db
        self._korean = korean
        self._clock = clock
        self._start_thread = start_thread or self._default_start
        self._log_raw = log
        self._progress_interval = progress_interval
        self._idle_seconds = idle_seconds
        self._lock = threading.RLock()
        self._paths = None
        self._vocab = None
        self._vocab_key = None
        self._vocab_maps = None
        self._loaded: dict[tuple[str, ...], LoadedTables | None] = {}
        # 등급 표는 한 번만 읽어 조합끼리 나눠 쓴다 — 표 하나가 수십 MB 라 조합마다 따로 읽으면 메모리가 겹친다
        self._shard_tables: dict[str, object] = {}
        self._build: dict | None = None
        self._closed = False
        # 질의 워커 — 화면(client)마다 가장 새 요청 하나만 기다린다(먼저 온 화면부터)
        self._query_wake = threading.Condition(self._lock)
        self._query_pending: dict[str, tuple[QueryRequest, tuple[str, ...]]] = {}
        self._query_worker = False
        self._sha_cache: dict[str, tuple[tuple[int, int], str]] = {}

    @staticmethod
    def _default_start(work: Callable[[], None], name: str) -> None:
        threading.Thread(target=work, daemon=True, name=name).start()

    def _log(self, message: str) -> None:
        """로그 한 줄 — 출력 실패(cp949 콘솔·닫힌 stdout)가 표 만들기를 실패로 바꾸지 않게 삼킨다."""
        try:
            self._log_raw(message)
        except Exception:
            try:
                self._log_raw(message.encode("ascii", "backslashreplace").decode("ascii"))
            except Exception:
                pass

    def _emit(self, event: dict) -> None:
        if self._closed:
            return
        self._emit_raw(event)

    # ── 공용 ──
    def paths(self):
        with self._lock:
            if self._paths is None:
                self._paths = cache_paths(self._storage)
            return self._paths

    def _database(self):
        if self._tag_db is None:
            from core.tag_database import get_tag_database

            self._tag_db = get_tag_database()
        return self._tag_db

    def _korean_lookup(self):
        if self._korean is None:
            from core.tag_korean_lookup import get_korean_tag_lookup

            self._korean = get_korean_tag_lookup()
        return self._korean

    def _invalidate(self) -> None:
        with self._lock:
            self._loaded.clear()
            self._shard_tables.clear()
            self._vocab = None
            self._vocab_key = None
            self._vocab_maps = None

    def _current_vocab(self):
        """캐시의 어휘와 (색인, 별칭, 함의) — 파일이 바뀌었으면 다시 읽는다. 없으면 (None, None).

        둘을 한 번에 돌려준다 — 표 만들기 스레드가 그 사이 캐시를 비워도 질의는 같은 벌을 쓴다."""
        from core.related_tags_build import load_vocab

        path = self.paths().vocab
        try:
            stat = path.stat()
            key = (stat.st_mtime_ns, stat.st_size)
        except OSError:
            key = None
        with self._lock:
            if key is None:
                self._vocab, self._vocab_key, self._vocab_maps = None, None, None
                self._loaded.clear()
                self._shard_tables.clear()
                return None, None
            if key != self._vocab_key:
                self._vocab = load_vocab(path)
                self._vocab_key = key
                self._vocab_maps = None
                self._loaded.clear()
                self._shard_tables.clear()
            if self._vocab is not None and self._vocab_maps is None:
                self._vocab_maps = (self._vocab.index(), self._vocab.aliases(), self._vocab.implications())
            return self._vocab, self._vocab_maps

    def _tables(self, shards: tuple[str, ...], vocab) -> LoadedTables | None:
        from core.related_tags_build import load_shard

        with self._lock:
            if shards in self._loaded:
                return self._loaded[shards]
            tables = []
            for shard in shards:
                if shard not in self._shard_tables:
                    table = load_shard(self.paths().shards[shard])
                    usable = table is not None and len(table.occ) == len(vocab.names)
                    self._shard_tables[shard] = table if usable else None
                if self._shard_tables[shard] is not None:
                    tables.append(self._shard_tables[shard])
            loaded = LoadedTables(tables) if tables else None
            self._loaded[shards] = loaded
            return loaded

    # ── 질의 ──
    def submit_query(self, request: QueryRequest, ratings: Iterable[str] | None) -> None:
        """질의를 워커에 맡긴다 — 워커가 바쁘면 이 요청이 같은 화면(client)에서 기다리던 옛 요청을 대신한다.

        다른 화면의 요청은 남긴다 — 웹 모드는 탭마다 자기 마지막 requestId 만 기다리므로, 한 슬롯을 모두가
        나눠 쓰면 다른 탭이 보낸 질의에 밀려 답을 못 받고 20초 뒤 '응답이 없습니다'가 떴다."""
        shards = rating_shards(ratings)
        with self._lock:
            if self._closed:
                return
            pending = self._query_pending
            pending.pop(request.client, None)
            pending[request.client] = (request, shards)
            while len(pending) > MAX_PENDING_CLIENTS:
                pending.pop(next(iter(pending)))
            self._query_wake.notify_all()
            if self._query_worker:
                return
            self._query_worker = True
        try:
            self._start_thread(self._query_loop, "related-tags-query")
        except Exception:
            with self._lock:
                self._query_worker = False
            raise

    def _query_loop(self) -> None:
        while True:
            with self._lock:
                while not self._query_pending and not self._closed:
                    self._query_wake.wait(self._idle_seconds)
                    if not self._query_pending:
                        # 한동안 할 일이 없으면 스레드를 끝낸다(다음 요청이 다시 띄운다)
                        self._query_worker = False
                        return
                if self._closed:
                    self._query_worker = False
                    return
                request, shards = self._query_pending.pop(next(iter(self._query_pending)))
            try:
                event = self.query(request, shards)
            except Exception as exc:
                event = self._query_event(request, shards, state="error", error=_user_message(exc))
            self._emit(event)

    def _query_event(self, request: QueryRequest, shards, **values) -> dict:
        event = {
            "action": QUERY_ACTION, "requestId": request.request_id, "state": "ready", "tags": [],
            "context": [], "ratings": list(shards), "missing": [], "stale": [], "unavailable": [],
            "estimateSec": 0, "build": self.build_progress(),
        }
        event.update(values)
        return event

    def query(self, request: QueryRequest, ratings: Iterable[str] | None) -> dict:
        """요청 하나 → 이벤트 본문. 표를 만들지는 않는다."""
        shards = rating_shards(ratings)
        manifest = read_manifest(self._data_dir)
        vocab, maps = self._current_vocab()
        states = shard_states(manifest, self._data_dir, self.paths(), vocab.fingerprint if vocab else None, shards)
        if vocab is not None and not self._vocab_is_current(vocab):
            # 데이터 갱신으로 카탈로그·별칭·함의가 바뀌었다 — '다시 만들기'가 어휘부터 새로 만들게 알린다
            states = {s: (STALE if state == READY else state) for s, state in states.items()}
        usable = tuple(s for s in shards if states[s] in USABLE_STATES)
        missing = [s for s in shards if states[s] == MISSING]
        stale = [s for s in shards if states[s] == STALE]
        unavailable = [s for s in shards if states[s] == UNAVAILABLE]
        tables = self._tables(usable, vocab) if usable and vocab is not None else None
        if usable and tables is None:
            # 메타는 맞는데 배열을 못 읽었다(깨진 파일) — 다시 만들 수 있게 missing 으로 알린다
            missing = [s for s in shards if s in usable or s in missing]
            stale = []
        building = self.build_progress()
        status = {"missing": missing, "stale": stale, "unavailable": unavailable,
                  "estimateSec": estimate_seconds(manifest, missing), "build": building}
        if tables is None:
            state = "building" if building else ("missing" if missing else "unavailable")
            return self._query_event(request, shards, state=state, **status)

        index, aliases, implications = maps
        names = vocab.names
        occ = tables.occ
        total = tables.total_posts
        positive: list[int] = []
        weighted_out: set[int] = set()
        for _field, text in request.fields:
            for name, weight in extract_prompt_tags(text):
                tag_id = resolve(name, index, aliases)
                if tag_id is None:
                    continue
                if weight < 0:
                    weighted_out.add(tag_id)       # NegPiP 음수 가중치 — 넣지 않을 태그
                else:
                    positive.append(tag_id)
        negative = {tag_id for name, _w in extract_prompt_tags(request.negative)
                    if (tag_id := resolve(name, index, aliases)) is not None}
        present = set(positive)
        blocked = present | implied_closure(present, implications) | negative | weighted_out
        context = select_context(positive, occ, total, limit=MAX_CONTEXT, generic_share=GENERIC_SHARE)

        from core.exclude_rules import parse_exclude_rules

        rules = parse_exclude_rules(request.exclude)
        n_candidates = vocab.n_candidates

        def allowed(tag_id: int) -> bool:
            name = names[tag_id]
            return (tag_id < n_candidates and not is_person_count_tag(name)
                    and not (rules and rules.excludes(name)))

        ranked = rank_related(context, tables.neighbors, occ, total, names, blocked=blocked, allowed=allowed,
                              min_pair=MIN_PAIR_QUERY, limit=request.limit)
        labels = self._labels([names[r.id] for r in ranked])
        tags = [{"tag": names[r.id], "ko": labels.get(names[r.id], ""), "count": int(occ[r.id]),
                 "via": [names[v] for v in r.via[:3]]} for r in ranked]
        return self._query_event(request, shards, state="ready", tags=tags,
                                 context=[names[c] for c in context], **status)

    def _labels(self, tags: list[str]) -> dict[str, str]:
        if not tags:
            return {}
        out: dict[str, str] = {}
        try:
            lookup = self._korean_lookup()
            for tag in tags:
                entry = lookup.label(tag)
                out[tag] = entry.ko if entry is not None else ""
        except Exception as exc:   # 한국어 카탈로그가 없어도 추천은 보인다
            self._log(f"[RelatedTags] 한국어 라벨 실패: {exc}")
        return out

    # ── 표 만들기 ──
    def build_progress(self) -> dict | None:
        """질의 답에 싣는 진행 — 도는 build 의 requestId 를 함께(다시 연 패널도 취소할 수 있게)."""
        with self._lock:
            job = self._build
            return {**job["progress"], "requestId": job["requestId"]} if job is not None else None

    def build_event(self) -> dict | None:
        """지금 도는 표 만들기의 진행 이벤트(없으면 None) — 두 번째 요청에 대한 답."""
        with self._lock:
            job = self._build
            if job is None:
                return None
            return {"action": BUILD_ACTION, "requestId": job["requestId"], "state": "building",
                    "progress": dict(job["progress"])}

    def start_build(self, request_id: str, shards: Iterable[str]) -> bool:
        """표 만들기를 시작한다. 이미 돌고 있거나 닫혔으면 False(호출자가 build_event 를 보낸다)."""
        shards = tuple(shards)
        with self._lock:
            if self._closed or self._build is not None:
                return False
            job = {"requestId": request_id, "shards": shards, "cancel": threading.Event(),
                   "progress": {"shard": shards[0] if shards else "", "index": 0, "count": len(shards),
                                "progress": 0.0, "overall": 0.0},
                   "emitted": None}
            self._build = job
        try:
            self._start_thread(lambda: self._run_build(job), "related-tags-build")
        except Exception:
            with self._lock:
                if self._build is job:
                    self._build = None
            raise
        return True

    def cancel(self, request_id: str | None = None) -> bool:
        with self._lock:
            job = self._build
            if job is None or (request_id is not None and job["requestId"] != request_id):
                return False
            job["cancel"].set()
            return True

    def close(self) -> None:
        with self._lock:
            self._closed = True
            if self._build is not None:
                self._build["cancel"].set()
            self._query_pending.clear()
            self._query_wake.notify_all()

    def _set_progress(self, job: dict, shard: str, index: int, count: int, fraction: float,
                      *, force: bool = False) -> None:
        fraction = max(0.0, min(1.0, float(fraction)))
        progress = {"shard": shard, "index": index, "count": count, "progress": round(fraction, 4),
                    "overall": round((index + fraction) / max(1, count), 4)}
        now = self._clock()
        with self._lock:
            job["progress"] = progress
            last = job["emitted"]
            if not force and last is not None and now - last < self._progress_interval:
                return
            job["emitted"] = now
        self._emit({"action": BUILD_ACTION, "requestId": job["requestId"], "state": "building",
                    "progress": progress})

    def _sha256(self, path: Path) -> str:
        """파일 sha256 — 크기·수정 시각이 같으면 앞서 잰 값을 쓴다(질의마다 다시 읽지 않게)."""
        stat = path.stat()
        key = (stat.st_size, stat.st_mtime_ns)
        with self._lock:
            cached = self._sha_cache.get(str(path))
        if cached is not None and cached[0] == key:
            return cached[1]
        digest = _file_sha256(path)
        with self._lock:
            self._sha_cache[str(path)] = (key, digest)
        return digest

    def _vocab_source(self) -> dict:
        """어휘를 만드는 원본의 지금 모습 — 자동완성 카탈로그(이름·크기·sha256)와 별칭·함의 파일(크기·sha256).

        카탈로그가 없으면 OSError. 별칭·함의 파일을 못 읽으면 그 값은 None(만들 때도 빼고 만든다)."""
        from core.tag_database import TagAsset

        database = self._database()
        catalog = Path(database.path(TagAsset.AUTOCOMPLETE_CATALOG))
        source: dict = {"catalog": catalog.name, "catalog_size": catalog.stat().st_size,
                        "catalog_sha256": self._sha256(catalog)}
        for key, asset in (("aliases", TagAsset.TAG_ALIASES), ("implications", TagAsset.TAG_IMPLICATIONS)):
            try:
                path = Path(database.path(asset))
                source[f"{key}_size"] = path.stat().st_size
                source[f"{key}_sha256"] = self._sha256(path)
            except Exception:
                source[f"{key}_size"] = None
                source[f"{key}_sha256"] = None
        return source

    def _vocab_is_current(self, vocab) -> bool:
        """캐시 어휘가 지금 원본으로 만든 것인가. 원본을 볼 수 없으면(카탈로그 없음 등) 참으로 둔다."""
        try:
            return _same_source(vocab.source, self._vocab_source())
        except Exception:
            return True

    def _ensure_vocab(self):
        from core.related_tags_build import load_vocab, read_catalog_rows, save_vocab, select_vocab

        path = self.paths().vocab
        source = self._vocab_source()
        vocab = load_vocab(path)
        if vocab is not None and _same_source(vocab.source, source):
            return vocab
        if vocab is not None:
            self._log("[RelatedTags] 어휘 원본(자동완성 카탈로그·별칭·함의)이 바뀌어 어휘를 다시 만든다")
        from core.tag_database import TagAsset

        database = self._database()
        catalog = Path(database.path(TagAsset.AUTOCOMPLETE_CATALOG))
        aliases, implications = {}, {}
        try:
            aliases = database.load_tag_aliases()
        except Exception as exc:
            self._log(f"[RelatedTags] 별칭 표를 읽지 못해 빼고 만든다: {exc}")
        try:
            implications = database.load_active_implications()
        except Exception as exc:
            self._log(f"[RelatedTags] 함의 표를 읽지 못해 빼고 만든다: {exc}")
        vocab = select_vocab(read_catalog_rows(catalog), aliases=aliases, implications=implications, source=source)
        save_vocab(path, vocab)
        self._invalidate()
        self._log(f"[RelatedTags] 어휘 저장: {path} - 태그 {len(vocab.names):,} · 후보 {vocab.n_candidates:,}")
        return load_vocab(path) or vocab

    def _run_build(self, job: dict) -> None:
        from core.related_tags_build import BuildCanceled, build_shard_table, save_shard, vocab_params
        from core.related_tags import BATCH_ROWS

        final: dict = {"state": "error", "error": "표를 만들지 못했습니다."}
        written: list[str] = []
        try:
            manifest = read_manifest(self._data_dir)
            outdated = set(stale_artifacts(manifest, self._data_dir, kind="search"))
            plan = []
            for shard in job["shards"]:
                found = shard_artifact(manifest, shard)
                if found is not None and found[0] not in outdated:
                    plan.append((shard, found))
            if not plan:
                raise ValueError(NO_DATA_MESSAGE)
            count = len(plan)
            self._set_progress(job, plan[0][0], 0, count, 0.0, force=True)
            if job["cancel"].is_set():
                raise BuildCanceled()
            vocab = self._ensure_vocab()
            for index, (shard, (relpath, artifact)) in enumerate(plan):
                if job["cancel"].is_set():
                    raise BuildCanceled()
                self._set_progress(job, shard, index, count, 0.0, force=True)
                started = self._clock()
                table = build_shard_table(
                    self._data_dir / relpath, vocab,
                    progress=lambda fraction, s=shard, i=index: self._set_progress(job, s, i, count, fraction),
                    should_stop=job["cancel"].is_set)
                table.meta.update({
                    "format": FORMAT_VERSION, "shard": shard, "vocab": vocab.fingerprint, "artifact": relpath,
                    **artifact_identity(artifact),
                    "params": {**vocab_params(), "batch_rows": BATCH_ROWS, **build_params()},
                })
                target = self.paths().shards[shard]
                save_shard(target, table)
                written.append(shard)
                self._invalidate()
                self._log(f"[RelatedTags] {shard.upper()} 표 저장: {target.resolve()} - 게시물 {table.posts:,} · "
                          f"건너뜀 {table.skipped:,} · 항목 {len(table.nbr):,} · {self._clock() - started:.1f}초")
            final = {"state": "done", "shards": written}
        except BuildCanceled:
            final = {"state": "canceled", "shards": written}
        except Exception as exc:
            self._log(f"[RelatedTags] 표 만들기 실패: {exc}")
            final = {"state": "error", "error": _user_message(exc), "shards": written}
        finally:
            with self._lock:
                if self._build is job:
                    self._build = None
            self._invalidate()
        self._log(f"[RelatedTags] 표 만들기 끝: {final['state']} - 만든 등급 {', '.join(written) or '없음'} "
                  f"({len(written)}/{len(job['shards'])})")
        self._emit({"action": BUILD_ACTION, "requestId": job["requestId"], **final})


__all__ = ["BUILD_ACTION", "NO_DATA_MESSAGE", "QUERY_ACTION", "RelatedTagsService"]
