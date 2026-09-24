"""한글·일본어 경로 안전 OpenCV 입출력(core.cv_io)과 자석 올가미 엣지맵(core.edge_map).

Windows(ACP 949)의 cv2.imread/imwrite 는 비ASCII 경로에서 조용히 실패한다(None/False).
에디터 연산·restore·자석 올가미가 한글 폴더 이미지에서 전부 실패하던 회귀를 막는다.
"""
from __future__ import annotations

import ast
import base64
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from core.cv_io import imread_unicode, imwrite_unicode
from core.edge_map import compute_edge_map, edge_map_data_url

ROOT = Path(__file__).resolve().parents[1]


def _pattern(h: int = 24, w: int = 32, channels: int = 3) -> np.ndarray:
    y, x = np.mgrid[0:h, 0:w]
    base = ((x * 7 + y * 3) % 256).astype(np.uint8)
    img = np.dstack([base, base[::-1], 255 - base])
    if channels == 4:
        img = np.dstack([img, np.where(x < w // 2, 0, 255).astype(np.uint8)])
    return img


class UnicodePathRoundTripTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.dir = Path(self._tmp.name) / '한글_日本語 폴더'
        self.dir.mkdir()

    def test_png_round_trip_keeps_alpha_with_unchanged(self):
        img = _pattern(channels=4)
        path = self.dir / '테스트_テスト.png'
        self.assertTrue(imwrite_unicode(path, img))
        back = imread_unicode(path, cv2.IMREAD_UNCHANGED)
        self.assertIsNotNone(back)
        self.assertEqual(back.shape, (24, 32, 4))
        np.testing.assert_array_equal(back, img)

    def test_default_flag_is_color_like_cv2_imread(self):
        path = self.dir / '알파.png'
        self.assertTrue(imwrite_unicode(str(path), _pattern(channels=4)))
        color = imread_unicode(str(path))
        self.assertEqual(color.shape, (24, 32, 3), 'flags 생략 시 cv2.imread 처럼 BGR 3채널이어야 한다')
        gray = imread_unicode(str(path), cv2.IMREAD_GRAYSCALE)
        self.assertEqual(gray.shape, (24, 32))

    def test_write_params_are_applied(self):
        img = _pattern(64, 64)
        low = self.dir / '저화질.jpg'
        high = self.dir / '고화질.jpg'
        self.assertTrue(imwrite_unicode(low, img, [cv2.IMWRITE_JPEG_QUALITY, 10]))
        self.assertTrue(imwrite_unicode(high, img, [cv2.IMWRITE_JPEG_QUALITY, 100]))
        self.assertLess(low.stat().st_size, high.stat().st_size)

    def test_missing_extension_writes_png(self):
        path = self.dir / '확장자없음'
        self.assertTrue(imwrite_unicode(path, _pattern()))
        self.assertTrue(path.read_bytes().startswith(b'\x89PNG'))

    def test_failures_keep_the_cv2_contract(self):
        self.assertIsNone(imread_unicode(self.dir / '없는파일.png'))
        empty = self.dir / '빈파일.png'
        empty.write_bytes(b'')
        self.assertIsNone(imread_unicode(empty))
        junk = self.dir / '깨진.png'
        junk.write_bytes(b'not an image at all')
        self.assertIsNone(imread_unicode(junk))
        self.assertIsNone(imread_unicode(None))
        self.assertIsNone(imread_unicode(self.dir), '폴더는 None 이어야 한다')
        self.assertFalse(imwrite_unicode(self.dir / '없는폴더' / 'x.png', _pattern()))
        self.assertFalse(imwrite_unicode(self.dir / 'x.unknownext', _pattern()))
        self.assertFalse(imwrite_unicode(self.dir / 'y.png', None))


class ExifOrientationTests(unittest.TestCase):
    """에디터용 opt-in 방향 보정 — PIL ImageOps.exif_transpose 와 같은 결과여야 한다."""

    def test_apply_matches_pil_exif_transpose_for_all_orientations(self):
        import io
        from PIL import Image, ImageOps
        from core.cv_io import apply_exif_orientation, exif_orientation
        rgb = _pattern(20, 40)[:, :, ::-1]   # BGR → RGB 로 PIL 에 넘긴다
        for orientation in range(1, 9):
            with self.subTest(orientation=orientation):
                exif = Image.Exif()
                exif[0x0112] = orientation
                buf = io.BytesIO()
                Image.fromarray(np.ascontiguousarray(rgb)).save(buf, 'PNG', exif=exif)
                data = buf.getvalue()
                self.assertEqual(exif_orientation(data), orientation)
                self.assertEqual(exif_orientation(np.frombuffer(data, np.uint8)), orientation)
                with Image.open(io.BytesIO(data)) as opened:
                    want = np.asarray(ImageOps.exif_transpose(opened).convert('RGB'))[:, :, ::-1]
                got = apply_exif_orientation(np.ascontiguousarray(rgb[:, :, ::-1]), orientation)
                np.testing.assert_array_equal(got, want)

    def test_missing_or_bad_orientation_is_one(self):
        from core.cv_io import apply_exif_orientation, exif_orientation
        ok, buf = cv2.imencode('.png', _pattern())
        self.assertEqual(exif_orientation(buf.tobytes()), 1)
        self.assertEqual(exif_orientation(b'not an image'), 1)
        img = _pattern()
        self.assertIs(apply_exif_orientation(img, 1), img)
        self.assertIs(apply_exif_orientation(img, 42), img)
        self.assertIsNone(apply_exif_orientation(None, 6))

    # Pillow getexif()·exif_transpose 는 EXIF 에 방향이 없으면 XMP tiff:Orientation 까지 읽는다.
    # Chromium 캔버스·OpenCV(IMREAD_COLOR, edge_map)는 XMP 를 무시한다 — 에디터 배열도 무시해야 한다.
    _XMP6 = ('<x:xmpmeta xmlns:x="adobe:ns:meta/"><rdf:RDF '
             'xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#"><rdf:Description '
             'xmlns:tiff="http://ns.adobe.com/tiff/1.0/" tiff:Orientation="6"/></rdf:RDF></x:xmpmeta>')

    def _xmp_only(self, fmt):
        import io
        from PIL import Image, PngImagePlugin
        buf = io.BytesIO()
        img = Image.new('RGB', (40, 20), (50, 60, 70))
        if fmt == 'PNG':
            info = PngImagePlugin.PngInfo()
            info.add_itxt('XML:com.adobe.xmp', self._XMP6)
            img.save(buf, 'PNG', pnginfo=info)
        else:
            img.save(buf, fmt, xmp=self._XMP6.encode('utf-8'))
        data = buf.getvalue()
        with Image.open(io.BytesIO(data)) as opened:
            self.assertEqual(opened.getexif().get(0x0112), 6, 'Pillow 가 XMP 방향을 읽는 전제가 깨졌다')
        return data

    def test_xmp_only_orientation_is_ignored_like_the_canvas(self):
        from core.cv_io import exif_orientation
        from core.editor_preview import imread_unchanged
        with tempfile.TemporaryDirectory() as tmp:
            for fmt, ext in (('JPEG', '.jpg'), ('PNG', '.png')):
                with self.subTest(fmt=fmt):
                    data = self._xmp_only(fmt)
                    self.assertEqual(exif_orientation(data), 1)
                    path = Path(tmp) / f'xmp방향{ext}'
                    path.write_bytes(data)
                    color = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)   # edge_map 이 쓰는 디코드
                    self.assertEqual(imread_unchanged(str(path)).shape[:2], color.shape[:2])
                    self.assertEqual(color.shape[:2], (20, 40))

    def test_real_exif_orientation_is_still_applied(self):
        import io
        from PIL import Image
        from core.cv_io import exif_orientation, image_exif_orientation
        cases = (('JPEG', 6), ('WEBP', 3), ('TIFF', 8))
        for fmt, orientation in cases:
            with self.subTest(fmt=fmt):
                exif = Image.Exif()
                exif[0x0112] = orientation
                buf = io.BytesIO()
                # EXIF 방향과 다른 XMP 방향이 같이 있어도 EXIF 가 이긴다(Pillow 도 EXIF 가 있으면 XMP 를 안 본다)
                extra = {} if fmt == 'TIFF' else {'xmp': self._XMP6.replace('"6"', '"2"').encode('utf-8')}
                Image.new('RGB', (40, 20), (50, 60, 70)).save(buf, fmt, exif=exif, **extra)
                self.assertEqual(exif_orientation(buf.getvalue()), orientation)
                with Image.open(io.BytesIO(buf.getvalue())) as opened:
                    self.assertEqual(image_exif_orientation(opened), orientation)
        # OpenCV IMREAD_COLOR 도 같은 방향으로 세운다(JPEG)
        exif = Image.Exif()
        exif[0x0112] = 6
        buf = io.BytesIO()
        Image.new('RGB', (40, 20), (50, 60, 70)).save(buf, 'JPEG', exif=exif)
        color = cv2.imdecode(np.frombuffer(buf.getvalue(), np.uint8), cv2.IMREAD_COLOR)
        self.assertEqual(color.shape[:2], (40, 20))

    def test_tiff_decoders_already_apply_the_orientation(self):
        """전제 고정(Codex S5 #1) — OpenCV·Pillow 의 TIFF 디코더는 방향을 스스로 적용한다.

        그래서 에디터 디코드(imread_unchanged)·load_image_array 는 TIFF 에 방향을 다시 적용하지 않는다.
        라이브러리 업그레이드로 이 전제가 바뀌면 조용히 돌아간 배열 대신 여기서 실패한다.
        """
        import io
        from PIL import Image
        from core.cv_io import is_tiff_bytes
        exif = Image.Exif()
        exif[0x0112] = 6
        buf = io.BytesIO()
        Image.new('RGB', (40, 20), (50, 60, 70)).save(buf, 'TIFF', exif=exif)
        data = buf.getvalue()
        self.assertTrue(is_tiff_bytes(data))
        self.assertTrue(is_tiff_bytes(np.frombuffer(data, np.uint8)))
        unchanged = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_UNCHANGED)
        self.assertEqual(unchanged.shape[:2], (40, 20), 'OpenCV TIFF IMREAD_UNCHANGED 가 방향을 적용하지 않는다')
        with Image.open(io.BytesIO(data)) as opened:
            self.assertEqual(opened.tag_v2.get(0x0112), 6)
            opened.load()
            self.assertEqual(opened.size, (20, 40), 'Pillow TIFF load() 가 방향을 적용하지 않는다')
            self.assertNotIn(0x0112, opened.tag_v2)
        # 다른 포맷은 TIFF 가 아니다
        ok, png = cv2.imencode('.png', _pattern())
        self.assertFalse(is_tiff_bytes(png.tobytes()))
        for junk in (b'', b'II', None, b'MM\x00*'[:3]):
            self.assertFalse(is_tiff_bytes(junk))
        self.assertTrue(is_tiff_bytes(b'MM\x00*rest'))
        self.assertTrue(is_tiff_bytes(b'II+\x00bigtiff'))


def save_tiff(image, path, *, orientation=None, xmp=None, compression=None):
    """TIFF 저장 — IFD0 방향 태그(진짜 EXIF)·XMP(태그 700)·압축을 골라 넣는다."""
    from PIL import Image, TiffImagePlugin
    extra = {}
    if xmp is not None:
        # tiffinfo 를 주면 Pillow 는 exif= 를 무시한다 — 방향도 같은 IFD 에 넣는다
        info = TiffImagePlugin.ImageFileDirectory_v2()
        info[700] = xmp.encode('utf-8')
        info.tagtype[700] = 1   # BYTE
        if orientation is not None:
            info[0x0112] = orientation
        extra['tiffinfo'] = info
    elif orientation is not None:
        exif = Image.Exif()
        exif[0x0112] = orientation
        extra['exif'] = exif
    if compression:
        extra['compression'] = compression
    image.save(path, format='TIFF', **extra)


class OpenPilImageTests(unittest.TestCase):
    """Pillow 로 픽셀을 읽는 곳(load_image_array·배치 변환·비교 GIF)의 TIFF 방향 (Codex S5 #1-a·#1-b).

    a) 경로로 연 방향 5~8 무압축 한 스트립 TIFF(L·P·RGBA·CMYK·I;16 — Pillow mmap 모드)는 mmap 지름길이
       세운 크기로 버퍼를 매핑해 픽셀이 뒤섞였다(20x40 L 이 (20,40) 쓰레기). 에디터(cv2)는 맞게 읽는다.
    b) Pillow TIFF load() 가 XMP 에만 적힌 방향으로 돌았다(에디터·브라우저 규칙은 진짜 EXIF 만).
    """

    _XMP6 = ExifOrientationTests._XMP6

    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name) / '한글 폴더'
        self.root.mkdir()

    @staticmethod
    def _stored(mode):
        """40x20 저장 픽셀 — 왼쪽·오른쪽 위·오른쪽 아래가 다른 값(뒤집기·회전이 모두 구별된다)."""
        from PIL import Image
        gray = np.zeros((20, 40), np.uint8)
        gray[:, :20] = 200
        gray[:10, 20:] = 90
        gray[10:, 20:] = 30
        if mode == 'I;16':
            return Image.fromarray(gray.astype(np.uint16) * 257)
        if mode == 'RGBA':
            return Image.fromarray(np.dstack([gray, gray[::-1], 255 - gray, np.full_like(gray, 255)]))
        return Image.fromarray(gray).convert(mode)

    def test_rotated_uncompressed_memory_map_modes_decode_like_the_editor(self):
        from PIL import Image
        from core.cv_io import apply_exif_orientation, open_pil_image
        from core.editor_preview import imread_unchanged
        for mode in ('L', 'P', 'RGBA', 'CMYK', 'I;16'):
            stored = self._stored(mode)
            self.assertEqual(stored.mode, mode)
            for orientation in range(1, 9):
                with self.subTest(mode=mode, orientation=orientation):
                    path = self.root / f'방향{orientation}_{mode.replace(";", "")}.tiff'
                    save_tiff(stored, path, orientation=orientation)
                    with Image.open(path) as probe:
                        self.assertEqual(len(probe.tile), 1, '전제: 무압축 한 스트립(mmap 지름길 대상)')
                    want = apply_exif_orientation(np.asarray(stored), orientation)
                    with open_pil_image(path) as opened:
                        opened.load()
                        got = np.asarray(opened)
                    np.testing.assert_array_equal(got, want)
                    self.assertEqual(imread_unchanged(str(path)).shape[:2], want.shape[:2], '에디터(cv2)와 같은 크기')

    def test_xmp_only_tiff_orientation_is_not_applied_but_real_exif_is(self):
        from PIL import Image
        from core.cv_io import apply_exif_orientation, open_pil_image
        from core.editor_preview import imread_unchanged
        stored = self._stored('RGB')
        base = np.asarray(stored)
        cases = (
            # (이름, IFD0 방향, XMP, 기대 방향)
            ('xmp6', None, self._XMP6, 1),
            ('exif8_xmp6', 8, self._XMP6, 8),   # 진짜 EXIF 가 있으면 그것만(Pillow·cv2 모두)
        )
        for compression in (None, 'tiff_lzw'):
            for name, orientation, xmp, want_orientation in cases:
                with self.subTest(name=name, compression=compression):
                    path = self.root / f'{name}_{compression}.tiff'
                    save_tiff(stored, path, orientation=orientation, xmp=xmp, compression=compression)
                    with Image.open(path) as probe:
                        self.assertEqual(probe.getexif().get(0x0112), orientation or 6,
                                         '전제: Pillow getexif 가 XMP 방향을 읽는다')
                    want = apply_exif_orientation(base, want_orientation)
                    with open_pil_image(path) as opened:
                        opened.load()
                        np.testing.assert_array_equal(np.asarray(opened), want)
                    np.testing.assert_array_equal(imread_unchanged(str(path))[:, :, ::-1], want)

    def test_suppress_only_touches_xmp_orientation_of_tiffs(self):
        import io
        from PIL import Image
        from core.cv_io import suppress_xmp_orientation
        # TIFF 가 아니면(JPEG 의 load 는 방향을 적용하지 않는다) getexif 를 건드리지 않는다
        buf = io.BytesIO()
        Image.new('RGB', (4, 2)).save(buf, 'JPEG', xmp=self._XMP6.encode('utf-8'))
        with Image.open(io.BytesIO(buf.getvalue())) as opened:
            suppress_xmp_orientation(opened)
            self.assertEqual(opened.getexif().get(0x0112), 6)
        # 진짜 IFD0 방향은 그대로 — load 가 적용한다
        path = self.root / 'exif3.tiff'
        save_tiff(Image.new('RGB', (4, 2)), path, orientation=3, xmp=self._XMP6)
        with Image.open(path) as opened:
            suppress_xmp_orientation(opened)
            self.assertEqual(opened.getexif().get(0x0112), 3)
        # 방향도 XMP 도 없는 TIFF·이미지가 아닌 값 — 조용히 넘어간다
        plain = self.root / 'plain.tiff'
        save_tiff(Image.new('RGB', (4, 2)), plain)
        with Image.open(plain) as opened:
            suppress_xmp_orientation(opened)
            self.assertNotIn(0x0112, opened.getexif())
        suppress_xmp_orientation(None)

    def test_open_releases_the_file_and_keeps_the_path_error_wording(self):
        from PIL import Image, UnidentifiedImageError
        from core.cv_io import open_pil_image
        good = self.root / '좋은.tiff'
        save_tiff(Image.new('L', (4, 2), 7), good, orientation=6)
        with open_pil_image(good) as opened:
            opened.load()
            self.assertEqual(opened.size, (2, 4))
        good.unlink()   # Windows: 핸들이 남아 있으면 지울 수 없다
        junk = self.root / '깨진.tiff'
        junk.write_bytes(b'not an image at all')
        with self.assertRaises(UnidentifiedImageError) as caught:
            with open_pil_image(junk):
                pass
        self.assertEqual(str(caught.exception), f'cannot identify image file {str(junk)!r}')
        junk.unlink()
        with self.assertRaises(FileNotFoundError):
            with open_pil_image(self.root / '없음.tiff'):
                pass


class EdgeMapTests(unittest.TestCase):
    def test_unicode_path_edge_map_is_not_empty(self):
        with tempfile.TemporaryDirectory() as tmp:
            img = np.zeros((40, 60, 3), np.uint8)
            img[:, 30:] = 255            # 세로 경계 하나
            path = Path(tmp) / '자석_올가미' / 'エッジ.png'
            path.parent.mkdir()
            self.assertTrue(imwrite_unicode(path, img))
            edges = compute_edge_map(str(path))
            self.assertEqual(edges.shape, (40, 60))
            self.assertGreater(int((edges > 0).sum()), 0)
            url = edge_map_data_url(str(path), 50, 150)
            self.assertTrue(url.startswith('data:image/png;base64,'))
            decoded = cv2.imdecode(np.frombuffer(base64.b64decode(url.split(',', 1)[1]), np.uint8),
                                   cv2.IMREAD_UNCHANGED)
            np.testing.assert_array_equal(decoded, edges)

    def test_unreadable_image_returns_empty_string(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.assertEqual(edge_map_data_url(str(Path(tmp) / '없음.png')), '')

    def test_inverted_thresholds_do_not_crash(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'a.png'
            imwrite_unicode(path, _pattern())
            self.assertIsNotNone(compute_edge_map(str(path), 200, 10))


class NoPathBasedCv2IoInAppCodeTests(unittest.TestCase):
    """앱 코드가 경로 기반 cv2.imread/imwrite 를 다시 쓰지 않는다 (comfy_custom_nodes 제외).

    커스텀 노드는 ComfyUI 프로세스 안에서 돌아 이 앱의 헬퍼를 import 할 수 없다.
    """

    APP_DIRS = ('core', 'ui', 'workers', 'utils', 'tabs', 'widgets', 'backends')

    def test_no_cv2_imread_or_imwrite_calls(self):
        offenders = []
        files = [p for d in self.APP_DIRS for p in (ROOT / d).rglob('*.py')]
        files += list(ROOT.glob('*.py'))
        for path in files:
            try:
                tree = ast.parse(path.read_text(encoding='utf-8'))
            except (SyntaxError, UnicodeDecodeError):
                continue
            for node in ast.walk(tree):
                if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                        and node.func.attr in ('imread', 'imwrite')
                        and isinstance(node.func.value, ast.Name) and node.func.value.id == 'cv2'):
                    offenders.append(f'{path.relative_to(ROOT)}:{node.lineno}')
        self.assertEqual(offenders, [], '경로 기반 cv2.imread/imwrite 대신 core.cv_io 를 쓰세요')


if __name__ == '__main__':
    unittest.main()
