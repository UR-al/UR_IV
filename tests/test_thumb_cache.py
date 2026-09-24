"""갤러리 썸네일 캐시(core.thumb_cache)와 Qt 모드 aithumb: 스킴 핸들러 계약.

WebEngine 페이지·GPU 는 쓰지 않는다 — 가짜 요청 job(QObject)으로 핸들러만 돌린다.
"""
import os
import tempfile
import time
import unittest
from pathlib import Path

from PIL import Image

from core.thumb_cache import (
    GALLERY_THUMB_BUCKETS,
    bucket_thumb_width,
    build_thumb_url,
    get_or_make_thumb,
    parse_thumb_url,
    thumb_key,
    thumb_path,
)
# QtWebEngine 모듈은 QCoreApplication 보다 먼저 import 돼야 한다(ui 패키지가 config 를 거쳐 올린다).
from ui.thumb_scheme import ThumbSchemeHandler, initiator_allowed


class ThumbCacheTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.dir = Path(temp.name)
        self.cache = str(self.dir / "cache")
        self.source = str(self.dir / "원본 이미지+1.png")
        Image.new("RGBA", (1024, 1536), (255, 0, 0, 0)).save(self.source)

    def test_buckets_round_up_and_clamp(self):
        self.assertEqual(GALLERY_THUMB_BUCKETS, (192, 256, 384, 512, 768))
        self.assertEqual(bucket_thumb_width(100), 192)
        self.assertEqual(bucket_thumb_width(200), 256)
        self.assertEqual(bucket_thumb_width(257), 384)
        self.assertEqual(bucket_thumb_width(760), 768)   # 380px × DPR 2
        self.assertEqual(bucket_thumb_width(5000), 768)
        self.assertEqual(bucket_thumb_width("nope"), 384)

    def test_url_round_trip_keeps_unicode_spaces_and_plus(self):
        url = build_thumb_url(self.source, 200)
        self.assertTrue(url.startswith("aithumb:thumb?path="))
        self.assertEqual(parse_thumb_url(url), (self.source, 256))
        # 프런트 encodeURIComponent 결과 모양 그대로
        self.assertEqual(parse_thumb_url("aithumb:thumb?path=C%3A%2Fimg%2Fa%20b%2B1.png&width=384"),
                         ("C:/img/a b+1.png", 384))
        # 카드 URL 의 내용 버전(v=, frontend utils/mediaVersions)은 캐시 키만 바꾼다 — 핸들러는 무시한다
        self.assertEqual(parse_thumb_url("aithumb:thumb?path=C%3A%2Fimg%2Fx.png&width=256&v=18a-3f.1700000000000"),
                         ("C:/img/x.png", 256))
        for bad in ("", "file:///x.png", "aithumb:other?path=x", "aithumb:thumb?width=10"):
            self.assertIsNone(parse_thumb_url(bad))

    def test_thumbnail_is_made_once_and_refreshed_when_source_changes(self):
        thumb = get_or_make_thumb(self.source, 256, self.cache)
        self.assertEqual(thumb, thumb_path(self.cache, self.source, 256))
        self.assertIn(os.sep + thumb_key(self.source, 256)[:2] + os.sep, thumb)
        with Image.open(thumb) as img:
            self.assertEqual(img.format, "JPEG")
            self.assertEqual(max(img.size), 256)
            # 투명한 부분은 앱 배경색(어두운 면)으로 합성된다 — 검게/빨갛게 새지 않는다
            self.assertTrue(all(channel < 30 for channel in img.getpixel((5, 5))))
        first_mtime = os.stat(thumb).st_mtime_ns
        self.assertEqual(get_or_make_thumb(self.source, 256, self.cache), thumb)
        self.assertEqual(os.stat(thumb).st_mtime_ns, first_mtime)
        time.sleep(0.02)
        Image.new("RGB", (512, 512), "blue").save(self.source)
        os.utime(self.source, ns=(first_mtime + 10_000_000_000, first_mtime + 10_000_000_000))
        self.assertEqual(get_or_make_thumb(self.source, 256, self.cache), thumb)
        with Image.open(thumb) as img:
            self.assertGreater(img.getpixel((5, 5))[2], 200)
        self.assertEqual([p for p in Path(thumb).parent.iterdir() if p.suffix == ".tmp"], [])

    def test_unreadable_source_returns_none_without_leaving_files(self):
        broken = self.dir / "broken.png"
        broken.write_bytes(b"not an image")
        self.assertIsNone(get_or_make_thumb(str(broken), 256, self.cache))
        self.assertIsNone(get_or_make_thumb(str(self.dir / "missing.png"), 256, self.cache))
        leftovers = [p for p in Path(self.cache).rglob("*") if p.is_file()]
        self.assertEqual(leftovers, [])

    # ── 렌더 중 원본이 바뀌는 경쟁 — 옛 그림이 새 원본의 '최신' 썸네일로 남으면 안 된다 ──

    def _blue_png_bytes(self):
        import io
        buf = io.BytesIO()
        Image.new("RGB", (1024, 1536), (0, 0, 255)).save(buf, "PNG")
        return buf.getvalue()

    def test_thumbnail_records_the_source_signature(self):
        from core.cache_cleanup import read_thumb_signature, source_signature, thumb_is_stale
        thumb = get_or_make_thumb(self.source, 256, self.cache)
        self.assertEqual(read_thumb_signature(thumb), source_signature(self.source))
        self.assertFalse(thumb_is_stale(self.source, thumb))

    def test_source_overwritten_during_render_is_rebuilt(self):
        # 에디터 원자 저장이 렌더 도중(픽셀을 읽은 뒤) 원본을 파랑으로 바꾼다
        from unittest import mock
        from core.cache_cleanup import thumb_is_stale
        from core.editor_save import atomic_write_bytes
        real = Image.Image.thumbnail
        calls = []

        def racing(img, *args, **kwargs):
            if not calls:
                atomic_write_bytes(self.source, self._blue_png_bytes())
            calls.append(1)
            return real(img, *args, **kwargs)

        with mock.patch.object(Image.Image, "thumbnail", racing):
            thumb = get_or_make_thumb(self.source, 256, self.cache)
        self.assertEqual(len(calls), 2, "바뀐 원본으로 다시 만들지 않았다")
        with Image.open(thumb) as img:
            r, _g, b = img.convert("RGB").getpixel((5, 5))
        self.assertGreater(b, 200)
        self.assertLess(r, 60)
        self.assertFalse(thumb_is_stale(self.source, thumb))

    def test_tmp_written_before_render_and_replaced_after_is_stale(self):
        # 에디터 저장 순서: 임시 파일 쓰기 → (그 사이 썸네일 렌더) → os.replace. 원본은 임시 파일의
        # 옛 mtime 을 가져 예전 '원본 mtime > 썸네일 mtime' 규칙으로는 새것처럼 보였다.
        import os as _os
        from core.cache_cleanup import thumb_is_stale
        tmp = self.dir / ".pending_save.tmp"
        tmp.write_bytes(self._blue_png_bytes())
        time.sleep(0.02)
        thumb = get_or_make_thumb(self.source, 256, self.cache)    # 아직 빨강 원본으로 만든다
        _os.replace(tmp, self.source)
        self.assertLess(_os.stat(self.source).st_mtime_ns, _os.stat(thumb).st_mtime_ns)
        self.assertTrue(thumb_is_stale(self.source, thumb))
        with Image.open(get_or_make_thumb(self.source, 256, self.cache)) as img:
            self.assertGreater(img.convert("RGB").getpixel((5, 5))[2], 200)

    def test_source_that_keeps_changing_returns_none_not_a_stale_thumb(self):
        from unittest import mock
        from core import thumb_cache
        with mock.patch.object(thumb_cache, "render_thumbnail",
                               side_effect=thumb_cache.SourceChangedError("busy")) as render:
            self.assertIsNone(get_or_make_thumb(self.source, 256, self.cache))
        self.assertEqual(render.call_count, thumb_cache.RENDER_ATTEMPTS)

    # ── 렌더가 다른 이유로 실패해도 원본이 있으면 옛 서명의 썸네일을 내주지 않는다 ──

    def _make_stale_thumb(self):
        thumb = get_or_make_thumb(self.source, 256, self.cache)
        stamp = os.stat(self.source).st_mtime_ns + 10_000_000_000
        Image.new("RGB", (1024, 1536), (0, 0, 255)).save(self.source)
        os.utime(self.source, ns=(stamp, stamp))
        return thumb

    def test_replace_blocked_by_open_thumb_returns_none_not_the_old_thumb(self):
        # Windows: 다른 스레드(스킴 핸들러·웹 스트리밍·서명 읽기)가 캐시 JPEG 를 열고 있으면
        # os.replace 가 PermissionError(winerror 5). 예전엔 옛 빨강 썸네일을 그대로 돌려줬다.
        from unittest import mock
        from core import thumb_cache
        from core.cache_cleanup import thumb_is_stale
        thumb = self._make_stale_thumb()
        self.assertTrue(thumb_is_stale(self.source, thumb))
        with mock.patch.object(thumb_cache.os, "replace", side_effect=PermissionError(13, "in use")):
            self.assertIsNone(get_or_make_thumb(self.source, 256, self.cache))
        self.assertEqual([p for p in Path(thumb).parent.iterdir() if p.suffix == ".tmp"], [])
        # 막힘이 풀리면 다음 요청이 새 원본으로 다시 만든다
        with Image.open(get_or_make_thumb(self.source, 256, self.cache)) as img:
            self.assertGreater(img.convert("RGB").getpixel((5, 5))[2], 200)

    def test_truncated_source_returns_none_not_the_old_thumb(self):
        # 비원자 쓰기(탐색기 복사·Forge) 도중의 잘린 원본 — PIL 이 잘림 오류를 낸다
        import io
        thumb = self._make_stale_thumb()
        buf = io.BytesIO()
        Image.effect_noise((1024, 1536), 64).convert("RGB").save(buf, "PNG")
        Path(self.source).write_bytes(buf.getvalue()[: len(buf.getvalue()) // 2])
        self.assertTrue(os.path.isfile(thumb))
        self.assertIsNone(get_or_make_thumb(self.source, 256, self.cache))

    def test_failed_render_still_serves_a_thumb_another_thread_just_made(self):
        # 같은 캐시 파일을 다른 스레드가 방금 지금 원본으로 만들었고, 이 스레드의 replace 만 막혔다
        from unittest import mock
        from core import thumb_cache
        self._make_stale_thumb()
        real = thumb_cache.render_thumbnail

        def made_by_other_then_blocked(source, dest, width, **kwargs):
            real(source, dest, width, **kwargs)
            raise PermissionError(13, "in use")

        with mock.patch.object(thumb_cache, "render_thumbnail", side_effect=made_by_other_then_blocked):
            thumb = get_or_make_thumb(self.source, 256, self.cache)
        self.assertIsNotNone(thumb)
        with Image.open(thumb) as img:
            self.assertGreater(img.convert("RGB").getpixel((5, 5))[2], 200)

    def test_failed_render_after_source_vanished_keeps_the_existing_thumb(self):
        # 원본이 렌더 도중 사라지면(지워짐·분리된 드라이브) 있는 썸네일을 그대로 쓴다
        from unittest import mock
        from core import thumb_cache
        thumb = self._make_stale_thumb()

        def vanish(source, dest, width, **kwargs):
            os.remove(source)
            raise FileNotFoundError(source)

        with mock.patch.object(thumb_cache, "render_thumbnail", side_effect=vanish):
            self.assertEqual(get_or_make_thumb(self.source, 256, self.cache), thumb)

    # ── 옛 캐시 폴더(legacy_dir) 입양 — 배경 정리(시작 30초 뒤) 전에 요청된 키 ──

    def _signed_legacy_thumb(self, width=256):
        """옛 폴더(image_cache/thumbs)에 core.thumb_cache 가 예전에 쓴 모양의 썸네일(초록)."""
        from core.cache_cleanup import shard_path, source_signature, thumb_signature_comment
        legacy = self.dir / "legacy"
        path = Path(shard_path(str(legacy), thumb_key(self.source, width)))
        path.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (8, 6), (0, 255, 0)).save(
            path, "JPEG", comment=thumb_signature_comment(source_signature(self.source)))
        return str(legacy), path

    def test_signed_legacy_thumb_is_adopted_without_rendering(self):
        from unittest import mock
        from core import thumb_cache
        legacy, legacy_thumb = self._signed_legacy_thumb()
        payload = legacy_thumb.read_bytes()
        with mock.patch.object(thumb_cache, "render_thumbnail",
                               side_effect=AssertionError("렌더하면 안 된다")) as render:
            thumb = get_or_make_thumb(self.source, 256, self.cache, legacy_dir=legacy)
        render.assert_not_called()
        self.assertEqual(thumb, thumb_path(self.cache, self.source, 256))
        self.assertEqual(Path(thumb).read_bytes(), payload)
        self.assertFalse(legacy_thumb.exists())

    def test_legacy_thumb_is_returned_when_the_source_was_deleted(self):
        legacy, _legacy_thumb = self._signed_legacy_thumb()
        _other_legacy, _ = self._signed_legacy_thumb(384)
        os.remove(self.source)
        self.assertEqual(get_or_make_thumb(self.source, 256, self.cache, legacy_dir=legacy),
                         thumb_path(self.cache, self.source, 256))
        # 옛 폴더를 보지 않으면 원본이 없는 키는 만들 수 없다 — 예전 첫 실행에 깨지던 카드
        self.assertIsNone(get_or_make_thumb(self.source, 384, self.cache))

    def test_legacy_thumb_of_a_changed_source_is_rebuilt_not_adopted(self):
        legacy, legacy_thumb = self._signed_legacy_thumb()
        stamp = os.stat(self.source).st_mtime_ns + 10_000_000_000
        Image.new("RGB", (1024, 1536), (0, 0, 255)).save(self.source)
        os.utime(self.source, ns=(stamp, stamp))
        thumb = get_or_make_thumb(self.source, 256, self.cache, legacy_dir=legacy)
        with Image.open(thumb) as img:
            self.assertGreater(img.convert("RGB").getpixel((5, 5))[2], 200)
        self.assertTrue(legacy_thumb.exists())   # 옛 것은 배경 정리가 치운다

    def test_without_legacy_dir_the_old_folder_is_not_touched(self):
        _legacy, legacy_thumb = self._signed_legacy_thumb()
        thumb = get_or_make_thumb(self.source, 256, self.cache)
        self.assertTrue(legacy_thumb.exists())
        with Image.open(thumb) as img:
            self.assertEqual(max(img.size), 256)   # 옛 8×6 이 아니라 새로 렌더한 것


class ThumbSchemeHandlerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from PyQt6.QtCore import QCoreApplication
        cls.app = QCoreApplication.instance() or QCoreApplication([])

    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.dir = Path(temp.name)
        self.source = str(self.dir / "a b.png")
        Image.new("RGB", (800, 600), "green").save(self.source)

    def make_job(self, url, initiator="file:///C:/app/index.html"):
        from PyQt6.QtCore import QObject, QUrl

        class FakeJob(QObject):
            def __init__(self):
                super().__init__()
                self.replied = None
                self.failed = None

            def initiator(self):
                return QUrl(initiator)

            def requestUrl(self):
                return QUrl(url)

            def fail(self, error):
                self.failed = error

            def reply(self, content_type, device):
                self.replied = (bytes(content_type), bytes(device.readAll()))

        return FakeJob()

    def wait_for(self, predicate, timeout=5.0):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline and not predicate():
            self.app.processEvents()
            time.sleep(0.01)
        return predicate()

    def test_valid_request_replies_with_jpeg_from_the_cache(self):
        handler = ThumbSchemeHandler(str(self.dir / "cache"), max_threads=1)
        job = self.make_job(build_thumb_url(self.source, 300))
        handler.requestStarted(job)
        self.assertTrue(self.wait_for(lambda: job.replied or job.failed is not None))
        self.assertIsNone(job.failed)
        content_type, data = job.replied
        self.assertEqual(content_type, b"image/jpeg")
        self.assertTrue(data.startswith(b"\xff\xd8"))

    def test_card_thumb_left_in_the_legacy_folder_is_served_without_rendering(self):
        # 업그레이드 뒤 첫 실행 — 배경 정리(시작 30초 뒤) 전에 보인 갤러리·즐겨찾기 카드
        from unittest import mock
        from core import thumb_cache
        from core.cache_cleanup import shard_path, source_signature, thumb_signature_comment
        from core.path_safety import safe_input_path
        from core.thumb_cache import THUMB_SOURCE_EXTS
        checked = safe_input_path(self.source, allowed_exts=THUMB_SOURCE_EXTS)   # 핸들러가 키를 만드는 경로
        legacy = self.dir / "legacy"
        legacy_thumb = Path(shard_path(str(legacy), thumb_key(checked, 256)))
        legacy_thumb.parent.mkdir(parents=True)
        Image.new("RGB", (8, 6), (0, 255, 0)).save(
            legacy_thumb, "JPEG", comment=thumb_signature_comment(source_signature(checked)))
        payload = legacy_thumb.read_bytes()
        handler = ThumbSchemeHandler(str(self.dir / "cache"), max_threads=1, legacy_dir=str(legacy))
        with mock.patch.object(thumb_cache, "render_thumbnail",
                               side_effect=AssertionError("렌더하면 안 된다")) as render:
            job = self.make_job(build_thumb_url(self.source, 256))
            handler.requestStarted(job)
            self.assertTrue(self.wait_for(lambda: job.replied or job.failed is not None))
        self.assertIsNone(job.failed)
        self.assertEqual(job.replied[1], payload)
        render.assert_not_called()
        self.assertFalse(legacy_thumb.exists())

    def test_bad_requests_fail_fast(self):
        # 문자열만 보고 거를 수 있는 요청은 GUI 스레드에서 바로 실패한다
        from PyQt6.QtWebEngineCore import QWebEngineUrlRequestJob
        handler = ThumbSchemeHandler(str(self.dir / "cache"), max_threads=1)
        blocked_ext = self.dir / "clip.mp4"
        blocked_ext.write_bytes(b"x")
        cases = [
            (build_thumb_url(str(blocked_ext), 256), "file:///app", QWebEngineUrlRequestJob.Error.UrlNotFound),
            ("aithumb:thumb?width=256", "file:///app", QWebEngineUrlRequestJob.Error.UrlNotFound),
            (build_thumb_url(self.source, 256), "https://evil.example/page", QWebEngineUrlRequestJob.Error.RequestDenied),
        ]
        for url, initiator, expected in cases:
            with self.subTest(url=url, initiator=initiator):
                job = self.make_job(url, initiator)
                handler.requestStarted(job)
                self.assertEqual(job.failed, expected)
                self.assertIsNone(job.replied)

    def test_path_checks_that_touch_the_disk_fail_from_the_worker(self):
        # 없는 파일·시스템 폴더는 워커의 safe_input_path 가 거르고 404 로 답한다
        from PyQt6.QtWebEngineCore import QWebEngineUrlRequestJob
        handler = ThumbSchemeHandler(str(self.dir / "cache"), max_threads=1)
        system_root = os.environ.get("SystemRoot", r"C:\Windows")
        for raw in (str(self.dir / "missing.png"), os.path.join(system_root, "not-a-thumb.png")):
            with self.subTest(raw=raw):
                job = self.make_job(build_thumb_url(raw, 256))
                handler.requestStarted(job)
                self.assertTrue(self.wait_for(lambda: job.replied or job.failed is not None))
                self.assertEqual(job.failed, QWebEngineUrlRequestJob.Error.UrlNotFound)
                self.assertIsNone(job.replied)

    def test_path_validation_never_runs_on_the_gui_thread(self):
        # NAS·매핑 드라이브에서 resolve/is_file 은 네트워크 왕복이다 — 카드마다 GUI 를 막으면 안 된다
        import threading
        from unittest import mock
        import ui.thumb_scheme as thumb_scheme
        calls = []
        real = thumb_scheme.safe_input_path

        def recording(*args, **kwargs):
            calls.append(threading.current_thread() is threading.main_thread())
            return real(*args, **kwargs)

        handler = ThumbSchemeHandler(str(self.dir / "cache"), max_threads=1)
        with mock.patch.object(thumb_scheme, "safe_input_path", recording):
            job = self.make_job(build_thumb_url(self.source, 256))
            handler.requestStarted(job)
            self.assertEqual(calls, [])  # requestStarted 는 경로를 검증하지 않고 넘긴다
            self.assertTrue(self.wait_for(lambda: job.replied or job.failed is not None))
        self.assertIsNone(job.failed)
        self.assertEqual(calls, [False])

    def test_ext_precheck_is_string_only_and_permissive(self):
        from ui.thumb_scheme import thumb_source_ext_allowed
        self.assertTrue(thumb_source_ext_allowed(r"\\nas\share\a.PNG"))
        self.assertTrue(thumb_source_ext_allowed("C:/img/a.tif"))
        self.assertTrue(thumb_source_ext_allowed("C:/img/a.png."))  # Windows 가 끝 점을 떼고 연다 — 최종 판정은 워커
        self.assertFalse(thumb_source_ext_allowed("C:/img/clip.mp4"))
        self.assertFalse(thumb_source_ext_allowed("C:/img/noext"))
        self.assertFalse(thumb_source_ext_allowed(""))

    def test_scheme_registration_runs_before_any_qt_app(self):
        # 별도 프로세스 — 등록은 QApplication 전에만 유효하고, 테스트 프로세스엔 이미 앱이 있을 수 있다.
        import subprocess
        import sys
        code = (
            "import sys, types; from pathlib import Path\n"
            "ui = types.ModuleType('ui'); ui.__path__ = [str(Path('ui').resolve())]; sys.modules['ui'] = ui\n"
            "from ui.thumb_scheme import register_thumb_scheme, thumb_scheme_registered, install_thumb_scheme_handler\n"
            "assert not thumb_scheme_registered()\n"
            "assert register_thumb_scheme() and register_thumb_scheme()\n"
            "assert thumb_scheme_registered()\n"
            "assert install_thumb_scheme_handler(None, 'unused') is None\n"
            "print('ok')\n"
        )
        root = Path(__file__).resolve().parents[1]
        result = subprocess.run([sys.executable, "-c", code], cwd=root, capture_output=True, text=True,
                                encoding="utf-8", timeout=60)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("ok", result.stdout)

    def test_initiator_policy(self):
        self.assertTrue(initiator_allowed("file", ""))
        self.assertTrue(initiator_allowed("", ""))
        self.assertTrue(initiator_allowed("http", "localhost"))
        self.assertFalse(initiator_allowed("https", "example.com"))


if __name__ == "__main__":
    unittest.main()
