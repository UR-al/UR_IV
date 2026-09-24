"""GUI 스레드를 막던 동기 슬롯의 비동기 짝 — requestCharacterTagsOnline / requestCompareGif / requestLoras.

예전 fetchCharacterTagsOnline(danbooru HTTPS 최대 2회, 각 12초)·exportCompareGif(풀해상도 LANCZOS·
16프레임 blend·optimize 저장)·getLoras(백엔드 HTTP + 디스크 병합)는 ``@pyqtSlot(result=str)`` 동기
슬롯이라 QWebChannel 이 GUI 스레드에서 끝까지 실행해 창(웹 모드는 WebSocket)이 수 초 멈췄다.
지금은 슬롯이 워커 스레드만 띄우고 곧바로 돌아오며, 결과는 요청 id 를 동봉한 *Ready 시그널로 온다.
순수 로직(core.danbooru_character_tags · core.compare_gif · ui.lora_catalog_cache)도 여기서 본다.
"""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from PIL import Image

from core import compare_gif
from core import danbooru_character_tags as danbooru

try:
    from ui.vue_bridge import VueBridge
    _IMPORT_ERROR = None
except Exception as exc:  # pragma: no cover - PyQt6 없는 환경
    VueBridge = None
    _IMPORT_ERROR = exc


class _Response:
    def __init__(self, posts=None, error=None):
        self._posts = posts
        self._error = error

    def raise_for_status(self):
        if self._error is not None:
            raise self._error

    def json(self):
        return self._posts


def _capture_threads():
    """ui.vue_bridge 가 띄우는 워커 스레드를 붙잡는다 — 슬롯이 곧바로 돌아오는지 보고, 몫은 손으로 돌린다."""
    started = []
    patcher = mock.patch(
        "ui.vue_bridge.threading.Thread",
        side_effect=lambda target, **_kw: SimpleNamespace(start=lambda: started.append(target)),
    )
    return patcher, started


def _png_with_corrupt_idat(width: int = 8, height: int = 8) -> bytes:
    """위 절반(빨강)만 정상이고 그다음 zlib 블록이 손상된 RGB PNG — CRC 는 맞다.

    IDAT = zlib 헤더 + stored 블록(위 절반 스캔라인) + 예약된 블록 타입(BTYPE=11) → inflate 가
    'invalid block type' 로 멈춰 Pillow 디코더가 err_code<0 을 낸다(잘린 파일과 달리 tile 을 비운 뒤 던진다).
    """
    import struct
    import zlib

    def chunk(kind: bytes, data: bytes) -> bytes:
        return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data) & 0xFFFFFFFF)

    rows = (b"\x00" + b"\xff\x00\x00" * width) * (height // 2)
    stream = b"\x78\x01" + b"\x00" + struct.pack("<HH", len(rows), len(rows) ^ 0xFFFF) + rows + b"\x07"
    header = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    return b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", header) + chunk(b"IDAT", stream) + chunk(b"IEND", b"")


# ── core.danbooru_character_tags ────────────────────────────────────────────


class DanbooruCharacterTagsTests(unittest.TestCase):
    def test_solo_query_first_and_tags_ranked_by_frequency(self):
        calls = []

        def get(url, **kwargs):
            calls.append((url, kwargs))
            return _Response([
                {"tag_string_general": "long_hair smile"},
                {"tag_string_general": "long_hair blue_eyes"},
                {"tag_string_general": ""},          # 태그 없는 게시물은 표본에서 뺀다
                "not-a-dict",
            ])

        result = danbooru.fetch_character_tags_online("  Hatsune  Miku ", get=get)
        self.assertEqual(result["sampled"], 2)
        self.assertEqual(result["tags"][0], "long hair")
        self.assertEqual(set(result["tags"]), {"long hair", "smile", "blue eyes"})
        self.assertEqual(len(calls), 1, "solo 질의에 결과가 있으면 두 번째 요청을 보내지 않는다")
        url, kwargs = calls[0]
        self.assertEqual(url, danbooru.DANBOORU_POSTS_URL)
        self.assertEqual(kwargs["params"]["tags"], "hatsune_miku solo")
        self.assertEqual(kwargs["timeout"], (5, 10))
        self.assertIn("User-Agent", kwargs["headers"])

    def test_falls_back_to_the_plain_tag_when_solo_fails_or_is_empty(self):
        for first in (_Response(error=RuntimeError("503 Server Error")), _Response([])):
            with self.subTest(first=first):
                queries = []
                replies = iter([first, _Response([{"tag_string_general": "a b"}])])

                def get(url, **kwargs):
                    queries.append(kwargs["params"]["tags"])
                    return next(replies)

                result = danbooru.fetch_character_tags_online("x", get=get)
                self.assertEqual(queries, ["x solo", "x"])
                self.assertEqual(result, {"tags": ["a", "b"], "sampled": 1})

    def test_network_failure_is_an_error_reply_never_an_exception(self):
        def get(url, **kwargs):
            raise ConnectionError("offline")

        result = danbooru.fetch_character_tags_online("miku", get=get)
        self.assertEqual(set(result), {"error"})
        self.assertIn("offline", result["error"])

    def test_blank_name_and_tagless_posts(self):
        self.assertIn("error", danbooru.fetch_character_tags_online("   ", get=mock.Mock()))
        tagless = danbooru.fetch_character_tags_online(
            "x", get=lambda url, **kw: _Response([{"tag_string_general": ""}]))
        self.assertEqual(tagless, {"error": "태그 없음"})

    def test_result_is_capped(self):
        posts = [{"tag_string_general": " ".join(f"t{i}" for i in range(80))}]
        tags, sampled = danbooru.rank_general_tags(posts)
        self.assertEqual(sampled, 1)
        self.assertEqual(len(tags), danbooru.RESULT_LIMIT)
        self.assertEqual(danbooru.rank_general_tags("nope"), ([], 0))


# ── core.compare_gif ─────────────────────────────────────────────────────────


class CompareGifCoreTests(unittest.TestCase):
    def test_target_size_uses_the_smaller_image_and_caps_the_long_side(self):
        self.assertEqual(compare_gif.target_size((800, 600), (1000, 700)), (800, 600))
        self.assertEqual(compare_gif.target_size((4000, 3000), (4096, 3072)), (1024, 768))
        self.assertEqual(compare_gif.target_size((3000, 4000), (3000, 4000), 1024), (768, 1024))
        self.assertEqual(compare_gif.target_size((0, 5), (3, 0)), (1, 1))
        self.assertEqual(compare_gif.target_size((5000, 10), (5000, 10), 0), (5000, 10))   # 0 = 제한 없음

    def test_blend_alphas_go_there_and_back_without_repeating_the_ends(self):
        alphas = compare_gif.blend_alphas()
        self.assertEqual(len(alphas), 16)
        self.assertEqual(alphas[0], 0.0)
        self.assertEqual(alphas[8], 1.0)
        self.assertEqual(alphas[9], 0.875)
        self.assertEqual(alphas[-1], 0.125)
        self.assertEqual(alphas.count(1.0), 1)
        self.assertEqual(alphas.count(0.0), 1)

    def test_rgba_and_rgb_mix_is_blended_and_downscaled(self):
        """예전: RGBA(배경 제거 결과)와 RGB 를 섞으면 Image.blend 가 'images do not match' 로 실패."""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            before, after, palette = root / "전.png", root / "후.png", root / "p.png"
            Image.new("RGBA", (40, 30), (255, 0, 0, 0)).save(before)
            Image.new("RGB", (48, 36), (0, 0, 255)).save(after)
            with Image.new("P", (40, 30)) as indexed:
                indexed.info["transparency"] = 0
                indexed.save(palette, transparency=0)
            data, frames = compare_gif.build_compare_gif(str(before), str(after), 5, 0, max_side=20)
            self.assertEqual(frames, 16)
            with Image.open(__import__("io").BytesIO(data)) as gif:
                self.assertEqual(gif.format, "GIF")
                self.assertEqual(gif.size, (20, 15))
                self.assertEqual(getattr(gif, "n_frames", 1), 16)
                self.assertGreaterEqual(gif.info.get("duration", 0), compare_gif.MIN_DURATION_MS)
                gif.seek(0)
                # 투명 RGBA 는 흰 배경에 합성된다(검정이 아니다)
                self.assertEqual(gif.convert("RGB").getpixel((5, 5)), (255, 255, 255))
            data2, frames2 = compare_gif.build_compare_gif(str(palette), str(after), 80, 0)
            self.assertEqual(frames2, 16)
            self.assertTrue(data2.startswith(b"GIF"))

    def test_exif_orientation_is_applied_like_the_viewer(self):
        """예전: 폰·카메라 JPEG(Orientation=6)가 옆으로 누운 채 나오고, 똑바른 이미지와 섞으면
        공통 크기가 (20, 20) 으로 찌그러졌다. 비교 슬라이더의 <img> 는 EXIF 방향을 적용한다."""
        import io

        def near(pixel, want):
            return all((c > 200) if w else (c < 60) for c, w in zip(pixel, want))

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            rotated, upright = root / "회전.jpg", root / "후.png"
            # 저장된 픽셀: 60x20, 왼쪽 빨강·오른쪽 파랑 → 90° 시계 방향으로 돌려 보이면 20x60, 위 빨강.
            with Image.new("RGB", (60, 20), (0, 0, 255)) as raw:
                raw.paste((255, 0, 0), (0, 0, 30, 20))
                exif = Image.Exif()
                exif[0x0112] = 6
                raw.save(rotated, "JPEG", quality=95, exif=exif.tobytes())
            with Image.new("RGB", (20, 60), (0, 0, 255)) as after:
                after.paste((0, 255, 0), (0, 0, 20, 30))
                after.save(upright)

            data, frames = compare_gif.build_compare_gif(str(rotated), str(upright), 80, 0)
            self.assertEqual(frames, 16)
            with Image.open(io.BytesIO(data)) as gif:
                self.assertEqual(gif.size, (20, 60))
                gif.seek(0)                                   # Before = 똑바로 선 JPEG
                first = gif.convert("RGB")
                self.assertTrue(near(first.getpixel((10, 8)), (1, 0, 0)), first.getpixel((10, 8)))
                self.assertTrue(near(first.getpixel((10, 52)), (0, 0, 1)), first.getpixel((10, 52)))
                gif.seek(8)                                   # After 끝 프레임
                last = gif.convert("RGB")
                self.assertTrue(near(last.getpixel((10, 8)), (0, 1, 0)), last.getpixel((10, 8)))

            same, _frames = compare_gif.build_compare_gif(str(rotated), str(rotated), 80, 0)
            with Image.open(io.BytesIO(same)) as gif:
                self.assertEqual(gif.size, (20, 60))

    def test_broken_exif_falls_back_to_the_stored_pixels(self):
        # 방향 읽기(진짜 EXIF)나 적용(exif_transpose) 어느 쪽이 깨져도 저장된 픽셀 그대로
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            before, after = root / "a.png", root / "b.png"
            exif = Image.Exif()
            exif[0x0112] = 6   # 읽고 적용할 수 있었다면 세워졌을 방향(적용 단계까지 가게)
            Image.new("RGB", (8, 6), (255, 0, 0)).save(before, exif=exif)
            Image.new("RGB", (8, 6), (0, 255, 0)).save(after)
            for target in ("core.cv_io.image_exif_orientation", "PIL.ImageOps.exif_transpose"):
                with self.subTest(broken=target):
                    with mock.patch(target, side_effect=SyntaxError("bad exif")) as broken:
                        data, frames = compare_gif.build_compare_gif(str(before), str(after), 80, 0)
                    broken.assert_called()
                    self.assertEqual(frames, 16)
                    self.assertTrue(data.startswith(b"GIF"))
                    with Image.open(__import__("io").BytesIO(data)) as gif:
                        gif.seek(0)
                        self.assertEqual(gif.size, (8, 6))
                        self.assertEqual(gif.convert("RGB").getpixel((4, 5)), (255, 0, 0))   # 저장된 픽셀 전부

    _XMP6 = ('<x:xmpmeta xmlns:x="adobe:ns:meta/"><rdf:RDF '
             'xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#"><rdf:Description '
             'xmlns:tiff="http://ns.adobe.com/tiff/1.0/" tiff:Orientation="6"/></rdf:RDF></x:xmpmeta>')

    @staticmethod
    def _left_red_right_blue():
        stored = Image.new("RGB", (60, 20), (0, 0, 255))
        stored.paste((255, 0, 0), (0, 0, 30, 20))
        return stored

    def test_xmp_only_orientation_is_ignored_like_the_slider(self):
        """Chromium <img>(비교 슬라이더)는 XMP tiff:Orientation 을 무시한다 — GIF 도 저장된 그대로.
        예전엔 Pillow exif_transpose 가 XMP 를 따라 GIF 만 20x60 으로 돌았고, 똑바른 60x20 짝과
        섞으면 공통 크기가 20x20 으로 찌그러졌다(Codex S5 #4)."""
        import io
        from PIL import PngImagePlugin

        def near(pixel, want):
            return all((c > 200) if w else (c < 60) for c, w in zip(pixel, want))

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            stored = self._left_red_right_blue()
            jpg, png, upright = root / "xmp방향.jpg", root / "xmp방향.png", root / "똑바로.png"
            stored.save(jpg, "JPEG", quality=95, xmp=self._XMP6.encode("utf-8"))
            info = PngImagePlugin.PngInfo()
            info.add_itxt("XML:com.adobe.xmp", self._XMP6)
            stored.save(png, pnginfo=info)
            stored.save(upright)
            for path in (jpg, png):
                with self.subTest(path=path.name):
                    with Image.open(path) as opened:
                        self.assertEqual(opened.getexif().get(0x0112), 6, "전제: Pillow 가 XMP 방향을 읽는다")
                    data, _frames = compare_gif.build_compare_gif(str(path), str(path), 80, 0)
                    with Image.open(io.BytesIO(data)) as gif:
                        self.assertEqual(gif.size, (60, 20))
                        first = gif.convert("RGB")
                        self.assertTrue(near(first.getpixel((5, 10)), (1, 0, 0)), first.getpixel((5, 10)))
                        self.assertTrue(near(first.getpixel((55, 10)), (0, 0, 1)), first.getpixel((55, 10)))
                    mixed, _frames = compare_gif.build_compare_gif(str(path), str(upright), 80, 0)
                    with Image.open(io.BytesIO(mixed)) as gif:
                        self.assertEqual(gif.size, (60, 20), "똑바른 짝과 섞어도 찌그러지지 않는다")

    def test_real_exif_wins_over_xmp_and_png_exif_after_the_image_data_counts(self):
        import io
        from PIL import PngImagePlugin
        from core.png_chunks import make_chunk
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            stored = self._left_red_right_blue()
            # EXIF 1 이 명시되면 XMP 6 은 무시(Pillow·Chromium 모두) — 저장된 그대로
            explicit = root / "exif1_xmp6.jpg"
            exif = Image.Exif()
            exif[0x0112] = 1
            stored.save(explicit, "JPEG", quality=95, exif=exif, xmp=self._XMP6.encode("utf-8"))
            # PNG 의 IDAT 뒤 eXIf(방향 6) — 진짜 EXIF 라 세운다(load() 가 info['exif'] 로 채운 뒤 읽는다)
            trailing = root / "trailing_exif6.png"
            stored.save(trailing, pnginfo=PngImagePlugin.PngInfo())
            exif6 = Image.Exif()
            exif6[0x0112] = 6
            raw = exif6.tobytes()
            raw = raw[6:] if raw.startswith(b"Exif\0\0") else raw
            data = trailing.read_bytes()
            iend = data.rindex(b"IEND") - 4
            trailing.write_bytes(data[:iend] + make_chunk(b"eXIf", raw) + data[iend:])
            # TIFF(방향 6) — Pillow 가 load() 에서 세우고 태그를 지운다. 두 번 돌리지 않는다
            tiff = root / "exif6.tiff"
            stored.save(tiff, exif=exif6)
            for path, size in ((explicit, (60, 20)), (trailing, (20, 60)), (tiff, (20, 60))):
                with self.subTest(path=path.name):
                    gif_bytes, _frames = compare_gif.build_compare_gif(str(path), str(path), 80, 0)
                    with Image.open(io.BytesIO(gif_bytes)) as gif:
                        self.assertEqual(gif.size, size)

    def test_grayscale_rotated_tiff_and_xmp_only_tiff_match_the_editor(self):
        """(Codex S5 #1-a) 경로로 연 흑백(L) 무압축 TIFF + 방향 6 — Pillow mmap 지름길이 세운 크기로 매핑해
        뒤섞인 60x20 을 섞었다(RGB 는 mmap 모드가 아니라 위 테스트로는 안 보였다).
        (Codex S5 #1-b) XMP 에만 방향 6 인 TIFF — Pillow load() 가 돌려 GIF 만 20x60 이 됐다."""
        import io
        from PIL import TiffImagePlugin
        bright, dark = 220, 30
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            stored = Image.new("L", (60, 20), dark)
            stored.paste(bright, (0, 0, 30, 20))   # 저장된 픽셀: 왼쪽 밝음·오른쪽 어두움
            rotated, xmp_only = root / "흑백_방향6.tiff", root / "xmp방향6.tiff"
            exif6 = Image.Exif()
            exif6[0x0112] = 6
            stored.save(rotated, exif=exif6)
            info = TiffImagePlugin.ImageFileDirectory_v2()
            info[700] = self._XMP6.encode("utf-8")
            info.tagtype[700] = 1
            stored.save(xmp_only, tiffinfo=info)
            with Image.open(xmp_only) as probe:
                self.assertEqual(probe.getexif().get(0x0112), 6, "전제: Pillow 가 XMP 방향을 읽는다")
            # 6: 왼쪽 밝음이 위 절반으로 / XMP 만: 저장된 그대로
            for path, size, corners in (
                    (rotated, (20, 60), {(1, 1): bright, (18, 1): bright, (1, 58): dark, (18, 58): dark}),
                    (xmp_only, (60, 20), {(1, 1): bright, (1, 18): bright, (58, 1): dark, (58, 18): dark})):
                with self.subTest(path=path.name):
                    gif_bytes, _frames = compare_gif.build_compare_gif(str(path), str(path), 80, 0)
                    with Image.open(io.BytesIO(gif_bytes)) as gif:
                        self.assertEqual(gif.size, size)
                        gif.seek(0)
                        first = gif.convert("L")
                        for xy, want in corners.items():
                            self.assertLessEqual(abs(first.getpixel(xy) - want), 8, xy)

    def test_corrupt_pixel_data_is_an_error_not_a_half_black_gif(self):
        """예전: _upright 의 넓은 except 가 exif_transpose 안 load() 의 디코더 오류까지 삼켰다.
        Pillow 는 tile 을 비운 뒤 던지므로 뒤의 convert() 가 조용히 통과해, IDAT 가 손상된 PNG 로
        위는 빨강·아래는 검정(0,0,0)인 16프레임 GIF 가 '성공'으로 gif/ 에 쓰였다."""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            broken, good = root / "손상.png", root / "b.png"
            broken.write_bytes(_png_with_corrupt_idat())
            Image.new("RGB", (8, 8), (0, 255, 0)).save(good)
            with Image.open(broken) as probe:                  # 전제: Pillow 가 이 파일을 디코드 오류로 본다
                self.assertRaises(OSError, probe.load)
            for pair in ((broken, good), (good, broken)):
                with self.subTest(before=pair[0].name):
                    with self.assertRaises(OSError):
                        compare_gif.build_compare_gif(str(pair[0]), str(pair[1]), 80, 0)
            out_dir = root / "gif"
            with self.assertRaises(OSError):
                compare_gif.export_compare_gif(str(broken), str(good), 80, 0, str(out_dir), now=1_700_000_000)
            self.assertFalse(out_dir.exists() and any(out_dir.iterdir()), "반쯤 디코드된 GIF 를 쓰지 않는다")

    def test_upright_rotates_in_place_without_an_extra_copy(self):
        """태그 없는 흔한 입력에 exif_transpose 가 돌려주던 풀해상도 사본을 만들지 않는다."""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            plain, rotated = root / "plain.png", root / "rotated.jpg"
            Image.new("RGB", (6, 4), (255, 0, 0)).save(plain)
            exif = Image.Exif()
            exif[0x0112] = 6
            Image.new("RGB", (6, 4), (255, 0, 0)).save(rotated, "JPEG", exif=exif.tobytes())
            with Image.open(plain) as opened:
                self.assertIs(compare_gif._upright(opened), opened)
            with Image.open(rotated) as opened:
                upright = compare_gif._upright(opened)
                self.assertIs(upright, opened)
                self.assertEqual(upright.size, (4, 6))
                self.assertNotIn(0x0112, upright.getexif())

    def test_export_never_overwrites_within_the_same_second(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            before, after = root / "a.png", root / "b.png"
            Image.new("RGB", (8, 8), (255, 0, 0)).save(before)
            Image.new("RGB", (8, 8), (0, 255, 0)).save(after)
            out_dir = root / "gif"
            first = compare_gif.export_compare_gif(str(before), str(after), 80, 0, str(out_dir), now=1_700_000_000)
            second = compare_gif.export_compare_gif(str(before), str(after), 80, 0, str(out_dir), now=1_700_000_000)
            self.assertNotEqual(first["path"], second["path"])
            self.assertNotIn("\\", first["path"])
            self.assertEqual(sorted(p.name for p in out_dir.iterdir()),
                             sorted([Path(first["path"]).name, Path(second["path"]).name]))
            self.assertEqual(first["frames"], 16)


# ── VueBridge 비동기 슬롯 ────────────────────────────────────────────────────


@unittest.skipIf(VueBridge is None, f"PyQt6 unavailable: {_IMPORT_ERROR}")
class RequestCharacterTagsOnlineTests(unittest.TestCase):
    def setUp(self):
        self.bridge = VueBridge()
        self.emitted = []
        self.bridge.characterTagsOnlineReady.connect(self.emitted.append)

    def test_slot_returns_before_any_network_and_the_reply_carries_id_and_name(self):
        patcher, started = _capture_threads()
        with mock.patch("core.danbooru_character_tags.fetch_character_tags_online",
                        return_value={"tags": ["long hair"], "sampled": 3}) as fetch:
            with patcher:
                self.bridge.requestCharacterTagsOnline("Hatsune Miku", "danbooru-1")
            fetch.assert_not_called()              # 슬롯(GUI 스레드)은 HTTP 를 하지 않는다
            self.assertEqual(self.emitted, [])
            self.assertEqual(len(started), 1)
            started[0]()                           # 워커 스레드 몫
        fetch.assert_called_once_with("Hatsune Miku")
        self.assertEqual(json.loads(self.emitted[0]), {
            "tags": ["long hair"], "sampled": 3, "requestId": "danbooru-1", "name": "Hatsune Miku"})

    def test_every_request_gets_its_own_worker(self):
        """키에 요청 id 를 넣어 다른 캐릭터·다시 누르기가 중복 제거로 버려지지 않는다."""
        patcher, started = _capture_threads()
        with patcher:
            self.bridge.requestCharacterTagsOnline("A", "r1")
            self.bridge.requestCharacterTagsOnline("B", "r2")
            self.bridge.requestCharacterTagsOnline("A", "r3")
        self.assertEqual(len(started), 3)

    def test_loader_failure_still_answers_with_an_error_reply(self):
        patcher, started = _capture_threads()
        with mock.patch("core.danbooru_character_tags.fetch_character_tags_online",
                        side_effect=RuntimeError("boom")):
            with patcher:
                self.bridge.requestCharacterTagsOnline("A", "r9")
            started[0]()
        reply = json.loads(self.emitted[0])
        self.assertEqual((reply["requestId"], reply["name"]), ("r9", "A"))
        self.assertIn("boom", reply["error"])


@unittest.skipIf(VueBridge is None, f"PyQt6 unavailable: {_IMPORT_ERROR}")
class RequestCompareGifTests(unittest.TestCase):
    def setUp(self):
        self.bridge = VueBridge()
        self.emitted = []
        self.bridge.compareGifReady.connect(self.emitted.append)

    def test_gif_is_built_in_the_worker_and_announced_with_the_request_id(self):
        import ui.vue_bridge as vue_bridge_module

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            before, after = root / "전.png", root / "후.png"
            Image.new("RGBA", (16, 12), (255, 0, 0, 128)).save(before)
            Image.new("RGB", (16, 12), (0, 0, 255)).save(after)
            patcher, started = _capture_threads()
            with mock.patch.object(vue_bridge_module, "__file__", str(root / "ui" / "vue_bridge.py")):
                with patcher:
                    self.bridge.requestCompareGif(str(before), str(after), 80, 0, "gif-1")
                self.assertFalse((root / "gif").exists(), "슬롯(GUI 스레드)은 이미지를 만들지 않는다")
                self.assertEqual(len(started), 1)
                started[0]()
            reply = json.loads(self.emitted[0])
            self.assertEqual(reply["requestId"], "gif-1")
            self.assertEqual(reply["frames"], 16)
            self.assertEqual(Path(reply["path"]).parent, root / "gif")
            self.assertTrue(Path(reply["path"]).is_file())

    def test_bad_paths_answer_with_an_error_reply(self):
        reply = json.loads(self.bridge._compare_gif_json("", "", 80, 0, "gif-2"))
        self.assertEqual(reply["requestId"], "gif-2")
        self.assertIn("error", reply)

    def test_decode_failure_is_reported_without_absolute_paths(self):
        with tempfile.TemporaryDirectory() as tmp:
            broken = Path(tmp) / "broken.png"
            broken.write_bytes(b"not an image")
            reply = json.loads(self.bridge._compare_gif_json(str(broken), str(broken), 80, 0, "gif-3"))
        self.assertEqual(reply["requestId"], "gif-3")
        self.assertIn("error", reply)
        self.assertNotIn(tmp, reply["error"])

    def test_corrupt_pixel_data_answers_with_an_error_and_writes_no_gif(self):
        """IDAT 가 손상된 PNG — 예전엔 성공 응답과 함께 아래 절반이 검정인 GIF 가 gif/ 에 쓰였다."""
        import ui.vue_bridge as vue_bridge_module

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            broken, good = root / "손상.png", root / "b.png"
            broken.write_bytes(_png_with_corrupt_idat())
            Image.new("RGB", (8, 8), (0, 255, 0)).save(good)
            with mock.patch.object(vue_bridge_module, "__file__", str(root / "ui" / "vue_bridge.py")):
                reply = json.loads(self.bridge._compare_gif_json(str(broken), str(good), 80, 0, "gif-4"))
            self.assertEqual(reply["requestId"], "gif-4")
            self.assertIn("error", reply)
            self.assertNotIn("path", reply)
            gif_dir = root / "gif"
            self.assertFalse(gif_dir.exists() and any(gif_dir.iterdir()))


# ── LoRA 카탈로그 (requestLoras → ui.lora_catalog_cache.load_catalog_json) ───


class LoadCatalogJsonTests(unittest.TestCase):
    def setUp(self):
        from ui import lora_catalog_cache

        self.cache = lora_catalog_cache
        self.addCleanup(setattr, lora_catalog_cache, "_raw_loras", lora_catalog_cache.raw_loras())
        self.bridge = SimpleNamespace(_merged_lora_cache=None)
        self.merges = []

    def _patches(self, backend):
        def merged(loras, engine):
            self.merges.append(list(loras))
            return json.dumps({"engine": engine, "names": [item["name"] for item in loras]})

        return (
            mock.patch("backends.get_backend", return_value=backend),
            mock.patch("backends.get_backend_type", return_value=SimpleNamespace(value="forge")),
            mock.patch("ui.lora_catalog_cache.merged_json", side_effect=merged),
        )

    def _load(self, backend, mode=""):
        from ui.lora_catalog_cache import load_catalog_json

        p1, p2, p3 = self._patches(backend)
        with p1, p2, p3:
            return json.loads(load_catalog_json(self.bridge, mode))

    def test_cache_hit_needs_neither_http_nor_a_disk_merge(self):
        from ui.lora_catalog_cache import cache_entry, raw_signature

        raw = [{"name": "a"}]
        self.cache._raw_loras = raw
        self.bridge._merged_lora_cache = cache_entry("forge", raw_signature(raw), '{"hit": true}', raw=raw)
        backend = mock.Mock()
        self.assertEqual(self._load(backend), {"hit": True})
        backend.get_loras.assert_not_called()
        self.assertEqual(self.merges, [])

    def test_raw_without_merge_only_merges(self):
        raw = [{"name": "a"}, {"name": "b"}]
        self.cache._raw_loras = raw
        backend = mock.Mock()
        self.assertEqual(self._load(backend), {"engine": "forge", "names": ["a", "b"]})
        backend.get_loras.assert_not_called()
        self.assertIs(self.bridge._merged_lora_cache["raw"], raw)

    def test_cold_cache_fetches_once_and_fills_both_caches(self):
        self.cache._raw_loras = []
        backend = mock.Mock()
        backend.get_loras.return_value = [{"name": "x"}]
        self.assertEqual(self._load(backend), {"engine": "forge", "names": ["x"]})
        backend.get_loras.assert_called_once_with()
        self.assertEqual(self.cache.raw_loras(), [{"name": "x"}])
        self.assertIs(self.bridge._merged_lora_cache["raw"], self.cache.raw_loras())
        # 다음 열기는 캐시 적중 — 백엔드를 다시 읽지 않는다
        self.assertEqual(self._load(backend), {"engine": "forge", "names": ["x"]})
        backend.get_loras.assert_called_once_with()

    def test_force_rescans_then_refetches_even_with_a_warm_cache(self):
        from ui.lora_catalog_cache import cache_entry, raw_signature

        raw = [{"name": "old"}]
        self.cache._raw_loras = raw
        self.bridge._merged_lora_cache = cache_entry("forge", raw_signature(raw), "[]", raw=raw)
        order = []
        backend = SimpleNamespace(refresh_loras=lambda: order.append("refresh"),
                                  get_loras=lambda: order.append("list") or [{"name": "new"}])
        self.assertEqual(self._load(backend, "force"), {"engine": "forge", "names": ["new"]})
        self.assertEqual(order, ["refresh", "list"])
        self.assertEqual(self.cache.raw_loras(), [{"name": "new"}])

    def test_backend_switch_during_fetch_does_not_cache_the_old_list(self):
        """받는 사이 연결 전환(무효화)이 끼면 옛 목록을 캐시하지 않고 새 상태로 다시 받는다."""
        from ui.lora_catalog_cache import invalidate

        self.cache._raw_loras = []
        replies = iter([[{"name": "old-backend"}], [{"name": "new-backend"}]])

        def get_loras():
            reply = next(replies)
            if reply[0]["name"] == "old-backend":
                invalidate(self.bridge)
            return reply

        backend = SimpleNamespace(get_loras=get_loras)
        self.assertEqual(self._load(backend), {"engine": "forge", "names": ["new-backend"]})
        self.assertEqual(self.cache.raw_loras(), [{"name": "new-backend"}])
        self.assertEqual(json.loads(self.bridge._merged_lora_cache["json"])["names"], ["new-backend"])

    def test_without_a_bridge_only_the_raw_cache_is_updated(self):
        """브리지 없는 파이썬 호출(VueBridge.getLoras(None, ...))도 같은 규칙 — 병합 캐시만 둘 곳이 없다."""
        from ui.lora_catalog_cache import load_catalog_json

        self.cache._raw_loras = [{"name": "raw-only"}]
        backend = mock.Mock()
        p1, p2, p3 = self._patches(backend)
        with p1, p2, p3:
            self.assertEqual(json.loads(load_catalog_json(None, "")), {"engine": "forge", "names": ["raw-only"]})
            backend.get_loras.assert_not_called()
            backend.get_loras.return_value = [{"name": "fresh"}]
            self.assertEqual(json.loads(load_catalog_json(None, "force")), {"engine": "forge", "names": ["fresh"]})
        self.assertEqual(self.cache.raw_loras(), [{"name": "fresh"}])


class SyncGetLorasDelegationTests(unittest.TestCase):
    """동기 getLoras 는 requestLoras 와 같은 load_catalog_json 을 부르는 얇은 래퍼다.

    예전 getLoras 에는 캐시 알고리즘 사본이 따로 있었고, 그 사본의 force/cold 분기는 raw 캐시를
    무조건 교체하고 병합 캐시를 락·객체 확인 없이 대입했다 — 운영(requestLoras)에서는 돌지 않는 사본을
    캐시 계약 테스트들이 고정하고 있었다."""

    def setUp(self):
        from ui import lora_catalog_cache

        self.cache = lora_catalog_cache
        self.addCleanup(setattr, lora_catalog_cache, "_raw_loras", lora_catalog_cache.raw_loras())

    @unittest.skipIf(VueBridge is None, f"PyQt6 unavailable: {_IMPORT_ERROR}")
    def test_get_loras_delegates_to_load_catalog_json(self):
        bridge = SimpleNamespace(_merged_lora_cache=None)
        with mock.patch("ui.lora_catalog_cache.load_catalog_json", return_value='["x"]') as load:
            self.assertEqual(VueBridge.getLoras(bridge, "force"), '["x"]')
        load.assert_called_once_with(bridge, "force")

    @unittest.skipIf(VueBridge is None, f"PyQt6 unavailable: {_IMPORT_ERROR}")
    def test_get_loras_failure_is_an_error_json(self):
        with mock.patch("ui.lora_catalog_cache.load_catalog_json", side_effect=OSError("backend down")):
            reply = json.loads(VueBridge.getLoras(None, ""))
        self.assertIn("backend down", reply["error"])

    @unittest.skipIf(VueBridge is None, f"PyQt6 unavailable: {_IMPORT_ERROR}")
    def test_force_refresh_across_a_backend_switch_does_not_cache_the_old_list(self):
        """force 로 받는 사이 연결 전환(무효화)이 끼면, 옛 백엔드 목록을 raw·병합 캐시에 넣지 않는다."""
        from ui.lora_catalog_cache import invalidate

        bridge = SimpleNamespace(_merged_lora_cache=None)
        self.cache._raw_loras = [{"name": "stale"}]

        class Backend:
            calls = 0

            def get_loras(self):
                Backend.calls += 1
                if Backend.calls == 1:
                    invalidate(bridge)   # 백엔드 전환 경계(on_webui_info_loaded)가 캐시를 비웠다
                    return [{"name": "old-backend"}]
                return [{"name": "new-backend"}]

        with (
            mock.patch("backends.get_backend", return_value=Backend()),
            mock.patch("backends.get_backend_type", return_value=SimpleNamespace(value="forge")),
            mock.patch("ui.lora_catalog_cache.merged_json",
                       side_effect=lambda loras, engine: json.dumps([item["name"] for item in loras])),
        ):
            served = json.loads(VueBridge.getLoras(bridge, "force"))
        self.assertEqual(served, ["new-backend"])
        self.assertEqual(2, Backend.calls)
        self.assertEqual(self.cache.raw_loras(), [{"name": "new-backend"}])
        self.assertIs(bridge._merged_lora_cache["raw"], self.cache.raw_loras())
        self.assertEqual(json.loads(bridge._merged_lora_cache["json"]), ["new-backend"])


class LoraPrewarmWiringTests(unittest.TestCase):
    """예전 _preload_loras 는 기동 2초 뒤 연결 여부를 한 번 보고, 모든 시작 경로에서 그때는 연결 전이라
    늘 그냥 끝났다(연결 성공은 캐시를 비우기만 했다). 이제 연결 성공 경계가 비운 직후 워커로 채운다."""

    def test_connection_success_invalidates_then_prewarms(self):
        import ast

        root = Path(__file__).resolve().parents[1]
        tree = ast.parse((root / "ui" / "generator_webui.py").read_text(encoding="utf-8"))
        handler = next(node for node in ast.walk(tree)
                       if isinstance(node, ast.FunctionDef) and node.name == "on_webui_info_loaded")

        def call_lines(name):
            return sorted(node.lineno for node in ast.walk(handler)
                          if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == name)

        invalidated, prewarmed = call_lines("invalidate_lora_cache"), call_lines("prewarm_async")
        self.assertEqual(len(invalidated), 1)
        self.assertEqual(len(prewarmed), 1)
        self.assertLess(invalidated[0], prewarmed[0], "비운 뒤에 채워야 옛 연결의 목록이 남지 않는다")

    def test_startup_no_longer_starts_a_preconnection_preload_thread(self):
        from ui.generator_ui_setup import UISetupMixin

        self.assertFalse(hasattr(UISetupMixin, "_preload_loras"))


@unittest.skipIf(VueBridge is None, f"PyQt6 unavailable: {_IMPORT_ERROR}")
class RequestLorasTests(unittest.TestCase):
    def setUp(self):
        self.bridge = VueBridge()
        self.emitted = []
        self.bridge.lorasReady.connect(self.emitted.append)

    def test_reply_envelope_is_valid_json_with_id_mode_and_catalog(self):
        patcher, started = _capture_threads()
        catalog = json.dumps([{"name": 'q"uote', "path": "C:/x"}])
        with mock.patch("ui.lora_catalog_cache.load_catalog_json", return_value=catalog) as load:
            with patcher:
                self.bridge.requestLoras("", 'loras-"1"')
            load.assert_not_called()
            started[0]()
        load.assert_called_once_with(self.bridge, "")
        reply = json.loads(self.emitted[0])
        self.assertEqual(reply, {"requestId": 'loras-"1"', "mode": "", "loras": [{"name": 'q"uote', "path": "C:/x"}]})

    def test_failure_answers_with_an_error_reply(self):
        patcher, started = _capture_threads()
        with mock.patch("ui.lora_catalog_cache.load_catalog_json", side_effect=OSError("backend down")):
            with patcher:
                self.bridge.requestLoras("force", "r2")
            started[0]()
        reply = json.loads(self.emitted[0])
        self.assertEqual((reply["requestId"], reply["mode"]), ("r2", "force"))
        self.assertIn("backend down", reply["error"])

    def test_force_is_never_swallowed_by_an_inflight_plain_request(self):
        patcher, started = _capture_threads()
        with patcher:
            self.bridge.requestLoras("", "r1")
            self.bridge.requestLoras("force", "r2")
        self.assertEqual(len(started), 2)

    def test_sync_get_loras_is_not_a_webchannel_slot(self):
        """동기 getLoras 는 파이썬 전용 — QWebChannel 에 노출되면 프론트가 다시 GUI 스레드를 막을 수 있다."""
        meta = VueBridge.staticMetaObject
        methods = {bytes(meta.method(i).name()).decode() for i in range(meta.methodCount())}
        self.assertNotIn("getLoras", methods)
        self.assertIn("requestLoras", methods)
        self.assertNotIn("exportCompareGif", methods)
        self.assertNotIn("fetchCharacterTagsOnline", methods)


class ModelListLookupTests(unittest.TestCase):
    """업스케일러·Ollama·ADetailer 목록 — request* 만 남았다(Codex R3 #3).

    동기 getUpscalers·ollamaListModels·getADetailerModels 는 HTTP(timeout 5초)를 GUI 스레드에서 돌렸다.
    프론트는 늘 request* 를 먼저 써서 죽은 폴백이었지만, 웹 facade 로는 직접 부를 수 있었다.
    """

    def test_sync_model_list_slots_are_gone_and_async_partners_remain(self):
        meta = VueBridge.staticMetaObject
        methods = {bytes(meta.method(i).name()).decode() for i in range(meta.methodCount())}
        for name in ("getUpscalers", "ollamaListModels", "getADetailerModels"):
            self.assertNotIn(name, methods)
            self.assertFalse(hasattr(VueBridge, name), name)
        for name in ("requestUpscalers", "requestOllamaModels", "requestADetailerModels"):
            self.assertIn(name, methods)

    def test_request_slots_return_before_any_lookup_and_answer_through_signals(self):
        bridge = VueBridge()
        got = {"up": [], "ollama": [], "ad": []}
        bridge.upscalersReady.connect(got["up"].append)
        bridge.ollamaModelsReady.connect(got["ollama"].append)
        bridge.adetailerModelsReady.connect(got["ad"].append)
        patcher, started = _capture_threads()
        with patcher, \
                mock.patch.object(VueBridge, "_load_upscalers_json", return_value='["Lanczos"]') as up, \
                mock.patch.object(VueBridge, "_load_ollama_models_json", return_value='["m:8b"]') as ollama, \
                mock.patch.object(VueBridge, "_load_adetailer_models_json",
                                  return_value='["face_yolov8n.pt"]') as ad:
            bridge.requestUpscalers()
            bridge.requestOllamaModels("http://127.0.0.1:11434")
            bridge.requestADetailerModels()
            # 슬롯은 워커만 띄우고 돌아온다 — 조회(HTTP)는 아직 한 번도 돌지 않았다
            up.assert_not_called()
            ollama.assert_not_called()
            ad.assert_not_called()
            self.assertEqual(len(started), 3)
            for work in started:
                work()
        self.assertEqual(got["up"], ['["Lanczos"]'])
        self.assertEqual(json.loads(got["ollama"][0]),
                         {"url": "http://127.0.0.1:11434", "models": ["m:8b"]})
        self.assertEqual(got["ad"], ['["face_yolov8n.pt"]'])


if __name__ == "__main__":
    unittest.main()
