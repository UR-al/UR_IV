"""히스토리 썸네일 미리 만들기(core.thumb_prefetch) — #53.

작업자 풀·대기 집합·청크 단위 emit·정리 1회·폭이 담긴 페이로드를 검증한다.
Qt·GPU 는 쓰지 않는다(PIL 로 작은 PNG 를 만든다).
"""
import os
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest import mock

from PIL import Image

from core import thumb_prefetch
from core.thumb_cache import thumb_path
from core.thumb_prefetch import ThumbnailPrefetcher, maintain_thumb_cache, pending_key, thumb_file_url


def expected_version(dest: str) -> str:
    """썸네일 버전 = 렌더 시각 + 썸네일에 적힌 원본 서명(원본이 바뀌면 같은 틱이어도 달라진다)."""
    from core.cache_cleanup import read_thumb_signature
    mtime_ns, size = read_thumb_signature(dest)
    return f"{os.stat(dest).st_mtime_ns}-{mtime_ns}-{size}"


class ThumbnailPrefetcherTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.dir = Path(temp.name)
        self.cache = str(self.dir / "thumbs")
        self.sources = []
        for i in range(5):
            path = str(self.dir / f"그림 {i}.png")
            Image.new("RGB", (640, 480), (i * 40, 10, 10)).save(path)
            self.sources.append(path)
        self.events: list[dict] = []
        self.events_lock = threading.Lock()

    def emit(self, payload):
        with self.events_lock:
            self.events.append(payload)

    def items(self):
        with self.events_lock:
            return [item for ev in self.events for item in ev["items"]]

    def make(self, **kw):
        kw.setdefault("idle_seconds", 0.2)   # 테스트 뒤 작업자가 오래 남지 않게
        prefetcher = ThumbnailPrefetcher(self.cache, self.emit, **kw)
        self.addCleanup(prefetcher.close)
        return prefetcher

    def test_generates_once_with_width_in_every_payload(self):
        p = self.make()
        self.assertEqual(p.submit(self.sources, 256), 5)
        self.assertTrue(p.wait_idle(20))
        got = {item["path"]: item["thumb"] for item in self.items()}
        self.assertEqual(set(got), set(self.sources))
        for src, url in got.items():
            dest = thumb_path(self.cache, src, 256)
            self.assertEqual(url, thumb_file_url(dest))
            self.assertTrue(os.path.isfile(dest))
            with Image.open(dest) as img:
                self.assertEqual(max(img.size), 256)
        self.assertTrue(all(ev["width"] == 256 for ev in self.events))
        self.assertEqual(p.pending_count(), 0)

    def test_cache_hits_are_sent_as_one_batch_per_chunk(self):
        p = self.make()
        p.submit(self.sources, 256)
        self.assertTrue(p.wait_idle(20))
        self.events.clear()
        # 따뜻한 캐시: 5건이 경로마다가 아니라 한 번에 온다
        p.submit(self.sources, 256)
        self.assertTrue(p.wait_idle(20))
        self.assertEqual(len(self.events), 1)
        self.assertEqual(len(self.events[0]["items"]), 5)

    def _signed_legacy_thumb(self, legacy: Path, source: str, width: int = 256) -> Path:
        """옛 폴더(image_cache/thumbs)에 core.thumb_cache 가 예전에 쓴 모양의 썸네일."""
        from core.cache_cleanup import shard_path, source_signature, thumb_signature_comment
        path = Path(shard_path(str(legacy), Path(thumb_path(self.cache, source, width)).stem))
        path.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (8, 6), (0, 255, 0)).save(
            path, "JPEG", comment=thumb_signature_comment(source_signature(source)))
        return path

    def test_legacy_thumbs_are_adopted_as_one_hit_batch_without_rendering(self):
        # 업그레이드 뒤 첫 실행: 히스토리는 시작하자마자 목록 전체를 요청하고, 옛 폴더 배경 정리는
        # 30초 뒤에 돈다. 그 사이 새 폴더에서 빗나간 키를 다시 렌더하지 않고 옮겨 온다.
        legacy = self.dir / "legacy"
        moved = [self._signed_legacy_thumb(legacy, src) for src in self.sources[:3]]
        os.remove(self.sources[2])   # 원본을 지운 히스토리 항목도 썸네일을 받는다
        with mock.patch.object(thumb_prefetch, "get_or_make_thumb",
                               side_effect=AssertionError("렌더하면 안 된다")) as render:
            p = self.make(max_workers=1, legacy_dir=str(legacy))
            self.assertEqual(p.submit(self.sources[:3], 256), 3)
            self.assertTrue(p.wait_idle(20))
        render.assert_not_called()
        self.assertEqual(len(self.events), 1, "옮겨 온 것은 적중 묶음 한 번으로 알린다")
        got = {item["path"]: item for item in self.events[0]["items"]}
        for src in self.sources[:3]:
            dest = thumb_path(self.cache, src, 256)
            self.assertEqual(got[src]["thumb"], thumb_file_url(dest))
            self.assertEqual(got[src]["v"], expected_version(dest))
        for path in moved:
            self.assertFalse(path.exists())

    def test_deleted_source_without_legacy_dir_gets_no_thumb(self):
        # 대조군 — legacy_dir 가 없으면 옛 폴더를 보지 않는다(원본이 없어 만들 수 없음)
        legacy = self.dir / "legacy"
        kept = self._signed_legacy_thumb(legacy, self.sources[0])
        os.remove(self.sources[0])
        p = self.make()
        p.submit(self.sources[:1], 256)
        self.assertTrue(p.wait_idle(20))
        self.assertEqual([item["thumb"] for item in self.items()], [""])
        self.assertTrue(kept.exists())

    def test_overwritten_source_is_rebuilt(self):
        p = self.make()
        p.submit(self.sources[:1], 256)
        self.assertTrue(p.wait_idle(20))
        dest = thumb_path(self.cache, self.sources[0], 256)
        old = os.stat(dest).st_mtime_ns
        # 원본을 덮어쓴다(에디터 저장 등) — mtime 이 썸네일보다 새로워진다
        Image.new("RGB", (640, 480), (0, 200, 0)).save(self.sources[0])
        os.utime(self.sources[0], ns=(old + 5_000_000_000, old + 5_000_000_000))
        self.events.clear()
        p.submit(self.sources[:1], 256)
        self.assertTrue(p.wait_idle(20))
        # 다시 만들었는지는 썸네일에 적힌 원본 서명으로 판정한다. 파일 mtime 비교는 두 렌더가
        # 같은 시계 틱 안에 끝나면 같아져 간헐적으로 실패했다(구현의 갱신 판정도 서명 일치다).
        from core.cache_cleanup import read_thumb_signature, source_signature
        self.assertEqual(read_thumb_signature(dest), source_signature(self.sources[0]))
        with Image.open(dest) as img:
            r, g, _b = img.convert("RGB").getpixel((5, 5))
            self.assertGreater(g, r)

    def test_every_thumbnail_carries_its_file_version(self):
        p = self.make()
        p.submit(self.sources[:2], 256)
        self.assertTrue(p.wait_idle(20))
        p.submit(self.sources[:2], 256)   # 이번엔 캐시 적중 묶음
        self.assertTrue(p.wait_idle(20))
        for item in self.items():
            dest = thumb_path(self.cache, item["path"], 256)
            self.assertEqual(item["v"], expected_version(dest))

    def test_request_for_an_overwritten_source_while_in_flight_is_rerun(self):
        # 렌더를 마친 직후(알리기 전) 원본이 덮어써지고 프런트가 같은 경로를 다시 요청한다.
        # 예전엔 그 요청을 버려 마지막 응답이 덮어쓰기 전 그림이었다.
        started, gate = threading.Event(), threading.Event()
        real = thumb_prefetch.get_or_make_thumb
        calls = []

        def paused(path, width, cache_dir):
            made = real(path, width, cache_dir)
            calls.append(path)
            if len(calls) == 1:
                started.set()
                gate.wait(10)
            return made

        with mock.patch.object(thumb_prefetch, "get_or_make_thumb", paused):
            p = self.make(max_workers=1)
            self.assertEqual(p.submit(self.sources[:1], 256), 1)
            self.assertTrue(started.wait(10))
            Image.new("RGB", (640, 480), (0, 200, 0)).save(self.sources[0])
            st = os.stat(self.sources[0])
            os.utime(self.sources[0], ns=(st.st_atime_ns, st.st_mtime_ns + 5_000_000_000))
            self.assertEqual(p.submit(self.sources[:1], 256), 0)   # 대기 중 — 끝나면 다시 돈다
            gate.set()
            self.assertTrue(p.wait_idle(20))
        mine = [item for item in self.items() if item["path"] == self.sources[0]]
        self.assertGreaterEqual(len(mine), 2)
        self.assertNotEqual(mine[0]["v"], mine[-1]["v"], "다시 만든 썸네일의 URL 버전이 같다")
        with Image.open(mine[-1]["thumb"][len("file:///"):]) as img:
            r, g, _b = img.convert("RGB").getpixel((5, 5))
        self.assertGreater(g, r, "마지막 응답이 덮어쓰기 전 그림이다")
        self.assertEqual(p.pending_count(), 0)

    def _overwrite(self, path: str, color, bump_ns: int) -> None:
        Image.new("RGB", (640, 480), color).save(path)
        st = os.stat(path)
        os.utime(path, ns=(st.st_atime_ns, st.st_mtime_ns + bump_ns))

    def test_rerun_on_the_second_worker_stays_in_flight_after_the_first_chunk_ends(self):
        # 청크 A = [K, X]. K 가 처리 중에 다시 요청돼 재실행 작업이 되고, 놀던 두 번째 작업자가 그걸
        # 곧바로 잡는다. 예전엔 청크 A 가 X 까지 끝낸 뒤 finally 에서 K 를 _inflight 에서 지워,
        # 두 번째 렌더 도중의 세 번째 덮어쓰기·재요청이 재실행 표시 없이 버려졌다 — 마지막 썸네일이
        # 둘째 그림(초록)에 머물고 원본 서명과도 달랐다(Codex S5 #2).
        from core.cache_cleanup import read_thumb_signature, source_signature
        k, x = self.sources[0], self.sources[1]
        started1, gate1 = threading.Event(), threading.Event()
        started2, gate3 = threading.Event(), threading.Event()
        gate_x, chunk_a_done = threading.Event(), threading.Event()
        real = thumb_prefetch.get_or_make_thumb
        k_calls = []

        def paused(path, width, cache_dir):
            if path == x:
                gate_x.wait(10)
                return real(path, width, cache_dir)
            made = real(path, width, cache_dir)
            k_calls.append(path)
            if len(k_calls) == 1:
                started1.set()
                gate1.wait(10)
            elif len(k_calls) == 2:
                started2.set()
                gate3.wait(10)
            return made

        with mock.patch.object(thumb_prefetch, "get_or_make_thumb", paused):
            # 두 작업자가 테스트 도중 쉬다 끝나지 않게 idle 을 넉넉히
            p = self.make(max_workers=2, idle_seconds=10)
            run_chunk = p._run_chunk

            def tracked(items, width):
                try:
                    run_chunk(items, width)
                finally:
                    if any(path == x for path, _key in items):
                        chunk_a_done.set()

            p._run_chunk = tracked
            key = pending_key(k, 256)
            self.assertEqual(p.submit([k, x], 256), 2)
            self.assertTrue(started1.wait(10))
            self._overwrite(k, (0, 200, 0), 5_000_000_000)
            self.assertEqual(p.submit([k], 256), 0)          # 처리 중 → 재실행 표시
            gate1.set()
            self.assertTrue(started2.wait(10), "재실행이 두 번째 작업자에서 돌지 않았다")
            gate_x.set()
            self.assertTrue(chunk_a_done.wait(10))
            with p._lock:
                self.assertIn(key, p._inflight, "재실행 청크가 처리 중인 키를 첫 청크가 지웠다")
            self._overwrite(k, (0, 0, 200), 10_000_000_000)
            self.assertEqual(p.submit([k], 256), 0)          # 여전히 처리 중 → 다시 재실행 표시
            gate3.set()
            self.assertTrue(p.wait_idle(20))
        self.assertEqual(len(k_calls), 3)
        mine = [item for item in self.items() if item["path"] == k]
        with Image.open(mine[-1]["thumb"][len("file:///"):]) as img:
            r, g, b = img.convert("RGB").getpixel((5, 5))
        self.assertGreater(b, max(r, g), "마지막 응답이 마지막 덮어쓰기 전 그림이다")
        dest = thumb_path(self.cache, k, 256)
        self.assertEqual(read_thumb_signature(dest), source_signature(k))
        self.assertEqual(p.pending_count(), 0)

    def test_duplicate_requests_while_pending_are_skipped(self):
        gate = threading.Event()
        started = threading.Event()

        def slow_maintenance(_dir):
            started.set()
            gate.wait(10)

        p = self.make(max_workers=1, maintenance=slow_maintenance)
        self.assertEqual(p.submit(self.sources[:3], 256), 3)
        self.assertTrue(started.wait(10))
        # 처리 중인 (경로, 폭)은 다시 넣지 않는다 — 같은 경로라도 다른 폭은 별개
        self.assertEqual(p.submit(self.sources[:3], 256), 0)
        self.assertEqual(p.submit([self.sources[0].replace("\\", "/")], 256), 0 if os.name == "nt" else 1)
        self.assertEqual(p.submit(self.sources[:1], 384), 1)
        gate.set()
        self.assertTrue(p.wait_idle(20))
        widths = sorted({ev["width"] for ev in self.events})
        self.assertEqual(widths, [256, 384])
        self.assertEqual(p.pending_count(), 0)
        # 끝난 뒤에는 다시 요청할 수 있다
        self.assertEqual(p.submit(self.sources[:1], 256), 1)
        self.assertTrue(p.wait_idle(20))

    def test_maintenance_runs_once_before_any_work_even_with_two_workers(self):
        calls = []
        seen_work_before = []

        def maintenance(directory):
            seen_work_before.append(bool(self.items()))
            calls.append(directory)
            time.sleep(0.2)

        p = self.make(max_workers=2, maintenance=maintenance)
        p.submit(self.sources[:2], 256)
        p.submit(self.sources[2:], 256)
        self.assertTrue(p.wait_idle(20))
        self.assertEqual(calls, [self.cache])
        self.assertEqual(seen_work_before, [False])
        self.assertEqual(len(self.items()), 5)

    def test_worker_count_is_bounded(self):
        gate = threading.Event()
        p = self.make(max_workers=2, maintenance=lambda _d: gate.wait(10))
        for i in range(10):
            p.submit([str(self.dir / f"missing-{i}.png")], 256)
        alive = p.worker_threads()
        self.assertEqual(len(alive), 2)                  # 청크 10개여도 작업자는 2개
        self.assertTrue(all(t.daemon for t in alive))   # 앱 종료를 붙잡지 않는다
        gate.set()
        self.assertTrue(p.wait_idle(20))

    def test_missing_source_reports_empty_thumb_and_bad_input_is_ignored(self):
        p = self.make()
        missing = str(self.dir / "없음.png")
        self.assertEqual(p.submit([missing, "", None, 3, "  "], 256), 1)
        self.assertTrue(p.wait_idle(20))
        self.assertEqual(self.items(), [{"path": missing, "thumb": ""}])

    def test_emit_failure_does_not_kill_the_worker(self):
        def bad_emit(_payload):
            raise RuntimeError("wrapped C/C++ object has been deleted")

        p = ThumbnailPrefetcher(self.cache, bad_emit, idle_seconds=0.2)
        self.addCleanup(p.close)
        p.submit(self.sources[:2], 256)
        self.assertTrue(p.wait_idle(20))
        self.assertEqual(p.pending_count(), 0)

    def test_close_drops_queued_work_and_rejects_new_requests(self):
        gate = threading.Event()
        p = self.make(max_workers=1, maintenance=lambda _d: gate.wait(10))
        p.submit(self.sources[:1], 256)
        p.submit(self.sources[1:], 256)
        p.close()
        self.assertEqual(p.submit(self.sources, 512), 0)
        gate.set()
        self.assertTrue(p.wait_idle(20))
        self.assertEqual(p.pending_count(), 0)

    def test_width_is_clamped_like_the_cache(self):
        p = self.make()
        p.submit(self.sources[:1], 5)
        self.assertTrue(p.wait_idle(20))
        self.assertEqual(self.events[0]["width"], 64)

    def test_rejected_path_is_answered_empty_and_never_rendered(self):
        """Codex R3 #1 — 관문이 거부한 경로는 PIL 에 넘기지 않고 thumb "" 로 답한다."""
        rejected = self.sources[0]
        p = self.make(is_allowed=lambda path: path != rejected)
        with mock.patch.object(thumb_prefetch, "get_or_make_thumb",
                               wraps=thumb_prefetch.get_or_make_thumb) as render:
            self.assertEqual(p.submit(self.sources[:2], 256), 2)
            self.assertTrue(p.wait_idle(20))
        got = {item["path"]: item["thumb"] for item in self.items()}
        self.assertEqual(got[rejected], "")
        self.assertTrue(got[self.sources[1]].startswith("file:///"))
        self.assertFalse(os.path.exists(thumb_path(self.cache, rejected, 256)))
        self.assertNotIn(rejected, [call.args[0] for call in render.call_args_list])
        self.assertEqual(p.pending_count(), 0)

    def test_existing_cache_is_not_served_for_a_rejected_path(self):
        """캐시 적중 경로도 관문 뒤 — 거부된 원본의 예전 캐시 파일 URL 을 내주지 않는다."""
        rejected = self.sources[0]
        dest = Path(thumb_path(self.cache, rejected, 256))
        dest.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (256, 192), "red").save(dest, "JPEG")
        p = self.make(is_allowed=lambda path: path != rejected)
        p.submit([rejected], 256)
        self.assertTrue(p.wait_idle(20))
        self.assertEqual(self.items(), [{"path": rejected, "thumb": ""}])

    def test_a_raising_gate_rejects_and_the_worker_survives(self):
        def gate(path):
            if path == self.sources[0]:
                raise OSError("resolve failed")
            return True

        p = self.make(max_workers=1, is_allowed=gate)
        p.submit(self.sources[:2], 256)
        self.assertTrue(p.wait_idle(20))
        got = {item["path"]: item["thumb"] for item in self.items()}
        self.assertEqual(got[self.sources[0]], "")
        self.assertTrue(got[self.sources[1]])
        self.assertEqual(p.pending_count(), 0)

    def test_default_gate_matches_the_other_thumbnail_entry_points(self):
        allowed = thumb_prefetch.default_thumb_source_allowed
        self.assertTrue(allowed(self.sources[0]))
        self.assertTrue(allowed(Path(self.sources[0]).as_uri()))
        self.assertTrue(allowed(str(self.dir / "지운 파일.png")), "지운 히스토리 파일은 캐시를 계속 보여 준다")
        text = self.dir / "notes.txt"
        text.write_text("x", encoding="utf-8")
        self.assertFalse(allowed(str(text)))
        self.assertFalse(allowed(""))
        with mock.patch("core.path_safety._is_forbidden", return_value=True):
            self.assertFalse(allowed(self.sources[0]))
            self.assertFalse(allowed(str(self.dir / "지운 파일.png")))
        if os.name == "nt":
            windir = os.environ.get("SystemRoot") or r"C:\Windows"
            self.assertFalse(allowed(os.path.join(windir, "Web", "Wallpaper", "Windows", "img0.jpg")))
            self.assertFalse(allowed(os.path.join(windir, "win.ini")))

    def test_default_gate_returns_the_normalised_path_it_checked(self):
        allowed = thumb_prefetch.default_thumb_source_allowed
        resolved = str(Path(self.sources[0]).resolve())
        self.assertEqual(allowed(self.sources[0]), resolved)
        self.assertEqual(allowed(Path(self.sources[0]).as_uri()), resolved)
        gone = self.dir / "sub" / ".." / "지운 파일.png"
        self.assertEqual(allowed(str(gone)), str((self.dir / "지운 파일.png").resolve()))
        self.assertIsNone(allowed(""))

    def _encoded_separator_escape(self):
        """검사(URL 디코딩 후 resolve)와 OS(원문 그대로)가 다른 파일을 가리키는 원문 목록.

        'gallery\\a%2Fb\\..\\..\\sys\\wallpaper.jpg' — OS 는 'a%2Fb' 를 이름 한 칸으로 보고 '..' 둘로
        self.dir 까지 올라가 sys\\wallpaper.jpg(막힌 폴더의 실제 파일)를 연다. 검사는 'a/b' 두 칸으로 봐
        gallery\\sys\\wallpaper.jpg(없는 파일 = 지운 히스토리)로 판정해 통과시켰다.
        """
        sys_dir = self.dir / "sys"
        sys_dir.mkdir()
        forbidden = sys_dir / "wallpaper.jpg"
        Image.new("RGB", (320, 200), "blue").save(forbidden)
        encodings = ["%2F", "%2f"] + (["%5C", "%5c"] if os.name == "nt" else [])
        raws = []
        for enc in encodings:
            # 윈도는 없는 칸도 '..' 로 글자 그대로 접지만, POSIX 는 칸이 있어야 따라간다 — 만들어 둔다.
            carrier = self.dir / "gallery" / f"a{enc}b"
            carrier.mkdir(parents=True, exist_ok=True)
            raws.append(str(carrier / ".." / ".." / "sys" / "wallpaper.jpg"))
        from core import path_safety
        real_is_forbidden = path_safety._is_forbidden
        sys_root = sys_dir.resolve()

        def fake_is_forbidden(resolved):
            return Path(resolved).is_relative_to(sys_root) or real_is_forbidden(resolved)

        return forbidden, raws, mock.patch.object(path_safety, "_is_forbidden", side_effect=fake_is_forbidden)

    def test_encoded_separator_cannot_render_a_forbidden_file(self):
        """Codex R3 재검토 #1 — 관문이 검사한 경로와 PIL 이 여는 경로가 같아야 한다."""
        forbidden, raws, forbid_sys = self._encoded_separator_escape()
        for raw in raws:
            # 전제: OS 는 이 원문으로 막힌 폴더의 실제 파일을 연다
            self.assertTrue(os.path.isfile(raw), raw)
            self.assertTrue(os.path.samefile(raw, forbidden), raw)
        p = self.make(max_workers=1, is_allowed=thumb_prefetch.default_thumb_source_allowed)
        with forbid_sys, mock.patch.object(thumb_prefetch, "get_or_make_thumb",
                                           wraps=thumb_prefetch.get_or_make_thumb) as render:
            # 대기 키(normpath)는 표기마다 같아 한 번에 넣으면 하나로 합쳐진다 — 하나씩 끝까지 돌린다
            for raw in raws:
                self.assertEqual(p.submit([raw], 256), 1)
                self.assertTrue(p.wait_idle(20))
            rendered = [call.args[0] for call in render.call_args_list]
        self.assertEqual([(item["path"], item["thumb"]) for item in self.items()],
                         [(raw, "") for raw in raws])
        for source in rendered:
            self.assertNotIn("%", source, "렌더는 원문이 아니라 검사한 정규화 경로로")
            self.assertFalse(os.path.exists(source))
        cached = [name for _root, _dirs, files in os.walk(self.cache) for name in files]
        self.assertEqual(cached, [], "막힌 폴더 원본의 축소본이 캐시에 생기지 않는다")
        self.assertEqual(p.pending_count(), 0)

    def test_render_and_cache_key_use_the_checked_path(self):
        """관문이 돌려준 경로로 렌더·키를 만든다 — file:// 원문도 검사한 로컬 경로로 그린다."""
        uri = Path(self.sources[0]).as_uri()
        p = self.make(is_allowed=thumb_prefetch.default_thumb_source_allowed)
        p.submit([uri], 256)
        self.assertTrue(p.wait_idle(20))
        dest = thumb_path(self.cache, str(Path(self.sources[0]).resolve()), 256)
        self.assertEqual([(item["path"], item["thumb"]) for item in self.items()],
                         [(uri, thumb_file_url(dest))])
        self.assertTrue(os.path.isfile(dest))
        # 두 번째는 같은 키로 캐시 적중(관문이 돌려준 경로 기준)
        self.events.clear()
        p.submit([uri], 256)
        self.assertTrue(p.wait_idle(20))
        self.assertEqual([item["thumb"] for item in self.items()], [thumb_file_url(dest)])

    def test_boolean_gate_keeps_the_raw_path(self):
        p = self.make(is_allowed=lambda _path: True)
        p.submit(self.sources[:1], 256)
        self.assertTrue(p.wait_idle(20))
        self.assertEqual(self.items()[0]["thumb"], thumb_file_url(thumb_path(self.cache, self.sources[0], 256)))
        empty = self.make(is_allowed=lambda _path: "")
        self.events.clear()
        empty.submit(self.sources[1:2], 256)
        self.assertTrue(empty.wait_idle(20))
        self.assertEqual(self.items(), [{"path": self.sources[1], "thumb": ""}])

    def test_pending_key_normalises_case_and_separators_on_windows(self):
        a = pending_key("C:/Out/A.png", 256)
        b = pending_key("c:\\out\\a.png", 256)
        if os.name == "nt":
            self.assertEqual(a, b)
        self.assertNotEqual(pending_key("C:/Out/A.png", 256), pending_key("C:/Out/A.png", 384))


class MaintainThumbCacheTests(unittest.TestCase):
    def test_prunes_with_the_budget_and_no_longer_migrates_flat_files(self):
        import core.cache_cleanup as cache_cleanup
        with mock.patch("core.cache_cleanup.prune_thumbs", return_value=2) as prune:
            self.assertEqual(maintain_thumb_cache("X", 123), 2)
        prune.assert_called_once_with("X", 123)
        # 평면→샤드 이관은 은퇴했다 — 새 캐시 폴더는 처음부터 샤딩, 옛 폴더는 core.legacy_thumb_cache
        self.assertFalse(hasattr(cache_cleanup, "migrate_flat_to_sharded"))


class VueBridgeThumbnailSlotTests(unittest.TestCase):
    """generateThumbnails 슬롯은 프리페처 하나를 지연 생성해 재사용하고, 페이로드를 JSON 으로 쏜다."""

    def test_slot_reuses_one_prefetcher_and_emits_json_with_width(self):
        import json
        from ui.vue_bridge import VueBridge

        bridge = VueBridge()
        created = []

        class FakePrefetcher:
            def __init__(self, cache_dir, emit, **kw):
                self.cache_dir, self.emit, self.kw = cache_dir, emit, kw
                self.calls = []
                created.append(self)

            def submit(self, paths, width):
                self.calls.append((list(paths), width))
                return len(paths)

        sent = []
        bridge.thumbnailReady.connect(sent.append)
        with mock.patch.object(thumb_prefetch, "ThumbnailPrefetcher", FakePrefetcher):
            bridge.generateThumbnails(json.dumps(["a.png", "b.png"]), 256)
            bridge.generateThumbnails(json.dumps(["c.png"]), 384)
            bridge.generateThumbnails("not json", 256)
            bridge.generateThumbnails(json.dumps([]), 256)
        self.assertEqual(len(created), 1)
        self.assertEqual(created[0].calls, [(["a.png", "b.png"], 256), (["c.png"], 384)])
        created[0].emit({"width": 256, "items": [{"path": "a.png", "thumb": ""}]})
        self.assertEqual(json.loads(sent[-1]), {"width": 256, "items": [{"path": "a.png", "thumb": ""}]})
        # 웹에도 공개된 슬롯이라 aithumb:·/thumbnail 과 같은 경로 관문을 건다(Codex R3 #1)
        self.assertIs(created[0].kw.get("is_allowed"), thumb_prefetch.default_thumb_source_allowed)
        # 옛 캐시 폴더의 같은 키는 렌더 전에 옮겨 온다(배경 정리를 기다리지 않는다)
        import config
        self.assertEqual(created[0].kw.get("legacy_dir"), config.LEGACY_THUMB_DIR)

    def test_web_facade_cannot_thumbnail_a_system_folder_image(self):
        """웹 facade 로 부른 generateThumbnails 가 시스템 폴더 원본의 축소본을 만들지 않는다."""
        import json
        from ui.vue_bridge import VueBridge
        from web_main_ui import WebBridgeFacade

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            forbidden = root / "sys" / "wallpaper.jpg"
            forbidden.parent.mkdir()
            Image.new("RGB", (320, 200), "blue").save(forbidden)
            normal = root / "gallery" / "ok.png"
            normal.parent.mkdir()
            Image.new("RGB", (320, 200), "green").save(normal)
            cache = root / "thumbs"

            from core import path_safety
            real_is_forbidden = path_safety._is_forbidden

            def fake_is_forbidden(resolved):
                return (Path(resolved) == forbidden.resolve()) or real_is_forbidden(resolved)

            # thumbnailReady 는 작업자 스레드에서 쏘여 이벤트 루프 없는 테스트엔 안 온다 —
            # 보내는 묶음을 프리페처에서 직접 기록한다(관문·렌더는 실제 코드 그대로).
            recorded: list[dict] = []

            class RecordingPrefetcher(ThumbnailPrefetcher):
                def _send(self, width, items):
                    recorded.extend(items)
                    super()._send(width, items)

            bridge = VueBridge()
            facade = WebBridgeFacade(bridge)
            # 옛 캐시 폴더도 임시 폴더로 — 실제 사용자 캐시(image_cache/thumbs)를 건드리지 않는다
            with mock.patch("config.THUMB_DIR", str(cache)), \
                    mock.patch("config.LEGACY_THUMB_DIR", str(root / "legacy_thumbs")), \
                    mock.patch.object(thumb_prefetch, "ThumbnailPrefetcher", RecordingPrefetcher), \
                    mock.patch.object(path_safety, "_is_forbidden", side_effect=fake_is_forbidden):
                response = json.loads(facade.invoke(
                    "generateThumbnails",
                    json.dumps([json.dumps([str(forbidden), str(normal)]), 256])))
                self.assertTrue(response.get("ok"), response)
                self.assertTrue(bridge._thumbnail_prefetcher().wait_idle(20))
            bridge._thumbnail_prefetcher().close()
            items = {item["path"]: item["thumb"] for item in recorded}
            self.assertEqual(items[str(forbidden)], "")
            self.assertTrue(items[str(normal)].startswith("file:///"))
            self.assertFalse(os.path.exists(thumb_path(str(cache), str(forbidden), 256)))


if __name__ == "__main__":
    unittest.main()
