# core/thumb_prefetch.py
"""히스토리 스트립 썸네일 미리 만들기 — ``VueBridge.generateThumbnails`` 의 Qt 비의존 본체.

예전 슬롯은
  - 호출(40장 청크)마다 daemon 스레드를 새로 띄워, 부팅 때 히스토리 1,300장이면 스레드 ~34개가
    동시에 PIL 디코드·LANCZOS 를 돌렸고(첫 실행·미리보기 폭 변경 시 CPU 포화, 순간 메모리 ~0.5GB),
  - 캐시가 이미 있어도 경로마다 ``thumbnailReady`` 를 쏴서 부팅마다 emit/JSON.parse 가 1,300번,
  - 이관·정리(migrate/prune)를 락 없는 check-then-set 으로 돌려 첫 청크들과 겹칠 수 있었고,
  - 응답에 폭이 없어 다른 폭으로 요청한 화면이 서로의 썸네일을 받았다.

여기서는
  - **작업자 수 고정(기본 2) 풀**: 요청은 큐에 쌓이고 작업자 2개가 차례로 처리한다. 작업자는 daemon
    이라 앱 종료를 붙잡지 않는다(표준 ThreadPoolExecutor 는 종료 때 남은 큐를 다 돌 때까지 기다린다).
  - **대기 집합**: 이미 큐에 있거나 처리 중인 (경로, 폭)은 다시 넣지 않는다 — 그 작업의 응답이 온다.
  - **청크 단위 emit**: 캐시 적중은 한 번에 모아 1회 보내고, 새로 만든 것만 만들 때마다 보낸다.
    페이로드: ``{"width": 256, "items": [{"path": 원본, "thumb": "file:///…jpg" 또는 "", "v": "렌더 시각"}]}``
    (``v`` 는 썸네일이 있을 때만 — 캐시 파일 경로가 같아도 다시 만든 썸네일을 새로 읽게 URL 에 붙인다)
  - **처리 중 재요청 = 재실행**: 처리 중인 (경로, 폭)을 다시 요청하면(원본을 덮어써 프런트가 무효화)
    끝난 뒤 한 번 더 돌려 새 원본 기준 결과를 다시 알린다.
  - **정리 1회 + 락**: 캐시 용량 정리는 첫 작업 전에 락 안에서 한 번만 돈다(다른 작업자는 기다림).
  - 썸네일 렌더·원자적 저장·원본 서명 무효화는 ``core.thumb_cache`` 를 그대로 쓴다(갤러리 aithumb:·웹과 같은 캐시).
  - **경로 관문** (``is_allowed``): 다른 썸네일 입구(웹 /thumbnail·aithumb:·requestImageSearchTexts)와
    같은 규칙(:func:`default_thumb_source_allowed`)을 작업자에서 적용한다. 예전 슬롯은 웹 클라이언트가
    준 경로를 그대로 PIL 에 넘기고 캐시 URL(/file 이 내준다)을 돌려줘, C:\\Windows 등 시스템 폴더 이미지의
    축소본을 받아 갈 수 있었다. 거부한 경로는 캐시 적중이어도 ``thumb: ""`` 로 답한다.
    관문이 돌려준 **검사한 정규화 경로**로 렌더·캐시 키를 만든다(원문은 응답 짝 맞추기에만) —
    원문을 OS 에 다시 넘기면 URL 디코딩·'..' 접기가 검사 때와 달라 관문을 비껴갔다.
  - **옛 캐시 폴더 입양** (``legacy_dir``): 새 폴더에서 빗나간 키는 관문을 통과한 뒤, 적중 판정 전에
    옛 폴더(image_cache/thumbs)의 같은 키를 옮겨 온다(core.legacy_thumb_cache.adopt_legacy_thumb).
    시작 30초 뒤의 배경 정리를 기다리지 않으므로 업그레이드 뒤 첫 실행에도 히스토리 전체를 다시
    렌더하지 않고(옮긴 것은 적중 묶음으로 한 번에 알린다), 원본을 지운 항목도 썸네일을 잃지 않는다.
"""
from __future__ import annotations

import logging
import os
import queue
import threading
from typing import Callable, Iterable, Optional

from core.cache_cleanup import read_thumb_signature, thumb_is_stale
from core.legacy_thumb_cache import adopt_legacy_thumb
from core.thumb_cache import clamp_thumb_width, get_or_make_thumb, thumb_path

logger = logging.getLogger(__name__)

DEFAULT_MAX_WORKERS = 2
# 할 일이 없으면 작업자가 이만큼 기다렸다가 스스로 끝난다(다음 요청 때 다시 띄운다).
WORKER_IDLE_SECONDS = 30.0

Emit = Callable[[dict], None]


def thumb_file_url(path: str) -> str:
    """캐시 파일 → 프런트가 mediaUrl 로 감싸 쓰는 file:/// URL."""
    return "file:///" + str(path).replace("\\", "/")


def thumb_version(path: str) -> str:
    """썸네일 버전 문자열. 못 읽으면 ''.

    캐시 파일 경로(sha1(path@width))는 다시 만들어도 같아서, 프런트가 이 값을 URL 에 붙여야
    WebEngine 메모리 캐시의 옛 그림 대신 다시 만든 썸네일을 읽는다.
    값은 ``렌더 시각 st_mtime_ns`` 에 썸네일 주석에 적힌 **원본 서명**(mtime_ns, 크기)을 더한 것이다.
    렌더 시각만 쓰면 원본을 덮어쓴 뒤 같은 시계 틱 안에 다시 만든 썸네일이 옛 버전과 같은 값을
    가져 화면이 옛 그림에 머물 수 있었다 — 원본이 바뀌면 서명이 바뀌어 버전도 반드시 바뀐다.
    (JS 수 정밀도를 넘으므로 문자열로 보낸다.)
    """
    try:
        rendered = os.stat(path).st_mtime_ns
    except OSError:
        return ""
    signature = read_thumb_signature(path)
    if signature is None:
        return str(rendered)
    return f"{rendered}-{signature[0]}-{signature[1]}"


def pending_key(path: str, width: int) -> tuple[str, int]:
    return os.path.normcase(os.path.normpath(str(path))), int(width)


def default_thumb_source_allowed(path: str) -> Optional[str]:
    """썸네일 원본으로 받을 경로면 **검사한 정규화 절대 경로**, 아니면 None — 참/거짓으로도 쓴다.

    다른 썸네일 입구(aithumb:·웹 /thumbnail)와 같은 관문이다. 시스템 폴더·장치 이름공간·관리 공유·
    이미지가 아닌 확장자는 거부한다. 다만 **지워진** 파일(허용 위치·확장자인데 파일만 없음)은 받는다
    — 히스토리의 지운 이미지는 캐시 썸네일을 계속 보여 준다.

    프리페처는 돌려받은 경로로 렌더하고 캐시 키를 만든다(원문 아님). 검사는 file:// 를 떼고
    URL 디코딩한 뒤 resolve 한 경로로 하는데, 원문을 그대로 PIL 에 넘기면 OS 는 '%5C'·'%2F' 를
    이름 글자로 보고 '..' 를 다르게 접는다 — 'gallery\\a%5Cb\\..\\..\\sys\\x.jpg' 가 검사에선
    'gallery\\sys\\x.jpg'(없는 파일)로 통과하고 실제로는 'sys\\x.jpg' 를 열었다(Codex R3 재검토 #1).
    다른 입구도 검증한 경로를 get_or_make_thumb 에 넘긴다.
    """
    from core.path_safety import resolve_missing_input_path, safe_input_path
    from core.thumb_cache import THUMB_SOURCE_EXTS

    return (
        safe_input_path(path, allowed_exts=THUMB_SOURCE_EXTS)
        or resolve_missing_input_path(path, allowed_exts=THUMB_SOURCE_EXTS)
    )


def maintain_thumb_cache(cache_dir: str, max_bytes: int) -> int:
    """용량 상한 정리(오래된 것부터). 지운 수.

    예전의 평면 → 샤딩 이관(migrate_flat_to_sharded)은 없앴다 — 캐시가 처음부터 샤딩된 새 폴더
    (config.THUMB_DIR)로 옮겨 갔고, 옛 폴더는 core.legacy_thumb_cache 가 한 번 정리한다.
    """
    from core.cache_cleanup import prune_thumbs

    removed = prune_thumbs(cache_dir, max_bytes)
    if removed:
        logger.info("썸네일 캐시 정리: %d개 삭제", removed)
    return removed


class ThumbnailPrefetcher:
    """(경로 목록, 폭) 요청을 작업자 풀에서 처리하고 결과를 ``emit(payload)`` 로 알린다."""

    def __init__(
        self,
        cache_dir: str,
        emit: Emit,
        *,
        max_workers: int = DEFAULT_MAX_WORKERS,
        maintenance: Optional[Callable[[str], object]] = None,
        idle_seconds: float = WORKER_IDLE_SECONDS,
        is_allowed: Optional[Callable[[str], object]] = None,
        legacy_dir: Optional[str] = None,
    ) -> None:
        self._cache_dir = cache_dir
        self._emit = emit
        self._max_workers = max(1, int(max_workers))
        self._maintenance = maintenance
        # 경로 관문(None = 모두 허용). resolve·is_file I/O 라 GUI 스레드(submit)가 아닌 작업자에서 부른다.
        self._is_allowed = is_allowed
        # 옛 캐시 폴더(None = 입양 안 함) — 빗나간 키를 렌더 전에 옮겨 온다(모듈 설명)
        self._legacy_dir = legacy_dir or None
        self._idle_seconds = max(0.05, float(idle_seconds))
        self._queue: "queue.Queue[tuple[list[tuple[str, tuple[str, int]]], int]]" = queue.Queue()
        self._lock = threading.Lock()
        self._pending: set[tuple[str, int]] = set()
        # 작업자가 지금 처리 중인 청크의 키 / 처리 중에 다시 요청돼 끝나면 한 번 더 돌릴 키
        self._inflight: set[tuple[str, int]] = set()
        self._rerun: set[tuple[str, int]] = set()
        self._workers: list[threading.Thread] = []
        self._maintenance_lock = threading.Lock()
        self._maintenance_done = maintenance is None
        self._closed = False

    # ── 요청 ─────────────────────────────────────────────────────────────
    def submit(self, paths: Iterable[object], width: object) -> int:
        """새로 큐에 넣은 경로 수. 이미 대기·처리 중인 (경로, 폭)은 건너뛴다.

        이미 **처리 중인** 키를 다시 요청하면(원본을 덮어써 프런트가 무효화한 경우) 버리지 않고
        표시해 둔다 — 그 처리가 끝나면 한 번 더 돌려 새 원본 기준 결과를 다시 알린다. 예전엔
        조용히 버려, 덮어쓰기 전 원본으로 만든 썸네일이 마지막 응답으로 남았다.
        """
        w = clamp_thumb_width(width)
        fresh: list[tuple[str, tuple[str, int]]] = []
        with self._lock:
            if self._closed:
                return 0
            for raw in paths or ():
                if not isinstance(raw, str) or not raw.strip():
                    continue
                key = pending_key(raw, w)
                if key in self._pending:
                    if key in self._inflight:
                        self._rerun.add(key)
                    continue
                self._pending.add(key)
                fresh.append((raw, key))
            if not fresh:
                return 0
            self._queue.put((fresh, w))
            self._ensure_workers_locked()
        return len(fresh)

    def pending_count(self) -> int:
        with self._lock:
            return len(self._pending)

    def worker_threads(self) -> list[threading.Thread]:
        """지금 살아 있는 이 인스턴스의 작업자 스레드."""
        with self._lock:
            return [t for t in self._workers if t.is_alive()]

    def wait_idle(self, timeout: float = 10.0) -> bool:
        """큐가 빌 때까지 기다린다(테스트용). 시간 안에 비면 True."""
        done = threading.Event()

        def _join() -> None:
            self._queue.join()
            done.set()

        threading.Thread(target=_join, daemon=True).start()
        return done.wait(timeout)

    def close(self) -> None:
        """새 요청을 받지 않고, 큐에 남은 작업은 버린다(처리 중인 한 건은 끝낸다)."""
        with self._lock:
            self._closed = True
            while True:
                try:
                    items, _w = self._queue.get_nowait()
                except queue.Empty:
                    break
                for _path, key in items:
                    self._pending.discard(key)
                self._queue.task_done()

    # ── 작업자 ───────────────────────────────────────────────────────────
    def _ensure_workers_locked(self) -> None:
        self._workers = [t for t in self._workers if t.is_alive()]
        while len(self._workers) < self._max_workers:
            thread = threading.Thread(
                target=self._worker_loop,
                name=f"thumb-prefetch-{len(self._workers) + 1}",
                daemon=True,
            )
            self._workers.append(thread)
            thread.start()

    def _worker_loop(self) -> None:
        me = threading.current_thread()
        while True:
            try:
                job = self._queue.get(timeout=self._idle_seconds)
            except queue.Empty:
                with self._lock:
                    # 큐가 빈 채로 기다리다 끝난다 — 그 사이 들어온 일이 있으면 계속 돈다.
                    if self._queue.empty():
                        if me in self._workers:
                            self._workers.remove(me)
                        return
                continue
            try:
                items, width = job
                self._run_chunk(items, width)
            except Exception as exc:  # 작업자는 죽지 않는다
                logger.warning("썸네일 청크 처리 실패 (무시): %s", exc)
            finally:
                self._queue.task_done()

    def _maintain_once(self) -> None:
        if self._maintenance_done:
            return
        with self._maintenance_lock:
            if self._maintenance_done:
                return
            try:
                if self._maintenance is not None:
                    self._maintenance(self._cache_dir)
            except Exception as exc:
                logger.warning("썸네일 캐시 정리 실패 (무시): %s", exc)
            finally:
                self._maintenance_done = True

    def _checked_source(self, path: str) -> Optional[str]:
        """경로 관문 → 렌더·캐시 키에 쓸 원본 경로. None = 거부.

        관문이 문자열을 돌려주면 **그 검사한 경로**로 렌더한다 — 원문을 다시 OS 에 넘기면 검사 때와
        다르게 해석될 수 있다(:func:`default_thumb_source_allowed`). 관문이 없거나 참/거짓만 돌려주면
        원문 그대로. 관문이 예외를 던지면 거부(fail closed).
        """
        if self._is_allowed is None:
            return path
        try:
            verdict = self._is_allowed(path)
        except Exception as exc:
            logger.debug("썸네일 원본 경로 검사 실패(거부): %s", exc)
            return None
        if isinstance(verdict, str):
            return verdict or None
        return path if verdict else None

    def _adopt_legacy(self, dest: str, source: str) -> bool:
        """옛 캐시 폴더의 같은 키를 ``dest`` 로 옮겨 왔는지. 실패는 빗나감으로 친다(렌더로 이어 간다)."""
        try:
            return adopt_legacy_thumb(self._legacy_dir, dest, source)
        except Exception as exc:   # 입양 실패가 청크 전체를 버리지 않게
            logger.debug("옛 썸네일 입양 실패(무시): %s", exc)
            return False

    def _finish(self, path: str, key: tuple[str, int], width: int) -> None:
        """키 하나를 끝낸다. 처리 중에 다시 요청됐으면 대기 집합에 둔 채 큐에 다시 넣는다(재실행)."""
        with self._lock:
            self._inflight.discard(key)
            if key in self._rerun and not self._closed:
                self._rerun.discard(key)
                self._queue.put(([(path, key)], width))
                return
            self._rerun.discard(key)
            self._pending.discard(key)

    def _send(self, width: int, items: list[dict]) -> None:
        if not items:
            return
        try:
            self._emit({"width": width, "items": items})
        except Exception as exc:
            logger.debug("thumbnailReady emit 실패: %s", exc)

    def _run_chunk(self, items: list[tuple[str, tuple[str, int]]], width: int) -> None:
        released: set[tuple[str, int]] = set()
        try:
            if self._closed:
                return
            with self._lock:
                self._inflight.update(key for _path, key in items)
            self._maintain_once()
            try:
                os.makedirs(self._cache_dir, exist_ok=True)
            except OSError:
                pass
            hits: list[dict] = []
            misses: list[tuple[str, str, tuple[str, int]]] = []
            for path, key in items:
                # 관문은 캐시 조회보다 먼저 — 거부한 경로의 예전 캐시 파일도 내주지 않는다.
                # 응답은 적중과 같은 묶음으로(원문 경로 그대로 — 프런트가 이 경로로 짝을 찾는다).
                source = self._checked_source(path)
                if source is None:
                    hits.append({"path": path, "thumb": ""})
                    self._finish(path, key, width)
                    released.add(key)
                    continue
                # 캐시 키·신선도·렌더는 모두 관문이 검사한 경로로(원문은 응답 짝 맞추기에만)
                dest = thumb_path(self._cache_dir, source, width)
                present = os.path.isfile(dest)
                if not present and self._legacy_dir:
                    # 옛 폴더의 같은 키를 먼저 옮겨 온다 — 옮긴 것은 이 적중 묶음으로 바로 알린다
                    present = self._adopt_legacy(dest, source)
                if present and not thumb_is_stale(source, dest):
                    hits.append({"path": path, "thumb": thumb_file_url(dest), "v": thumb_version(dest)})
                    self._finish(path, key, width)
                    released.add(key)
                else:
                    misses.append((path, source, key))
            # 캐시 적중은 한 번에 — 경로마다 쏘던 수백 건의 emit 을 청크당 1건으로
            self._send(width, hits)
            for path, source, key in misses:
                if self._closed:
                    return
                try:
                    # 옛 폴더 입양은 위 적중 판정에서 이미 해 봤다 — 여기서는 렌더만(legacy_dir 없이)
                    made = get_or_make_thumb(source, width, self._cache_dir)
                except Exception:
                    made = None
                # 대기 집합에서 먼저 빼고 알린다 — 알림 직후 같은 경로를 다시 요청하면 새로 받는다
                self._finish(path, key, width)
                released.add(key)
                item = {"path": path, "thumb": ""}
                if made:
                    item.update(thumb=thumb_file_url(made), v=thumb_version(made))
                self._send(width, [item])
        finally:
            with self._lock:
                for _path, key in items:
                    if key in released:
                        # _finish 가 이미 정리했다. 그 뒤로 이 키는 이 청크 것이 아니다 — 재실행으로 다시
                        # 큐에 넣었거나 새 요청이 들어와, 다른 작업자가 지금 처리 중일 수 있다. 여기서
                        # _inflight 를 지우면 그동안 '대기 중인데 처리 중 아님'이 되어 원본을 또 덮어쓴 뒤의
                        # 재요청이 재실행 표시 없이 버려졌다(Codex S5 #2).
                        continue
                    self._inflight.discard(key)
                    self._rerun.discard(key)
                    self._pending.discard(key)
