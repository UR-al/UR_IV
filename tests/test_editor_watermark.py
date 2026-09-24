"""에디터 워터마크(core.editor_watermark + VueBridge 배선) 회귀 테스트.

예전: BGRA 가 BGR 로 떨어져 투명도가 사라졌고, 글꼴 표시명이 전부 OSError 라 늘 10px 기본
글꼴이었으며, 이미지 워터마크 100% 가 1% 로 줄었고, 반투명 워터마크 불투명도가 제곱이 됐다.
그 뒤: 일본 한자('込'·'働')가 맑은 고딕에 없어 두부로 나왔고, 회전 타일이 대각선 패딩 캔버스
(8K ≈ 439MB + 회전 사본)를 잡았으며, clamp 를 끈 500% 이미지 워터마크가 통째로 리사이즈됐다.
"""
from __future__ import annotations

import json
import math
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import cv2
import numpy as np
from PIL import Image, ImageFont

from core.cv_io import imread_unicode, imwrite_unicode
from core.editor_watermark import (
    CJK_FONT_FILES,
    JAPANESE_FONT_FILES,
    KOREAN_FONT_FILES,
    WatermarkError,
    blend_over,
    centered_position,
    cjk_font_files,
    covered_chars,
    fallback_candidates,
    fallback_font_paths,
    font_candidates,
    needs_cjk_font,
    open_font,
    parse_hex_color,
    placed_alpha,
    plan_font_runs,
    rasterize_text,
    render_image_watermark,
    render_text_watermark,
    resize_visible,
    resolve_font_path,
    rotated_extent,
    rotated_origin,
    sample_affine,
    script_preference,
    tiled_alpha,
)


def _ink(before: np.ndarray, after: np.ndarray) -> np.ndarray:
    """바뀐 픽셀 마스크(H×W bool)."""
    diff = np.abs(after.astype(np.int16) - before.astype(np.int16))
    return diff.reshape(diff.shape[0], diff.shape[1], -1).max(axis=2) > 0


def _bbox(mask: np.ndarray):
    ys, xs = np.nonzero(mask)
    return int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())


def _text(**kw):
    params = {'text': 'WM', 'fontFamily': 'Arial', 'fontSize': 40, 'color': '#FFFFFF',
              'opacity': 1.0, 'xPct': 50, 'yPct': 50, 'rotation': 0, 'tile': False, 'clamp': True}
    params.update(kw)
    return params


class FontResolutionTests(unittest.TestCase):
    def test_cjk_detection(self):
        for text in ('홍길동', 'テスト', '漢字', 'A가', 'ㄱ'):
            with self.subTest(text=text):
                self.assertTrue(needs_cjk_font(text))
        for text in ('Watermark', '© 2026 Studio', '', 'Ünïcödé'):
            with self.subTest(text=text):
                self.assertFalse(needs_cjk_font(text))

    def test_display_names_map_to_font_files(self):
        self.assertEqual(font_candidates('Times New Roman')[0], 'times.ttf')
        self.assertEqual(font_candidates('courier new')[0], 'cour.ttf')
        self.assertEqual(font_candidates('Georgia')[0], 'georgia.ttf')
        self.assertEqual(font_candidates('맑은 고딕')[0], 'malgun.ttf')

    def test_unknown_or_path_like_family_never_becomes_a_file_to_open(self):
        for family in ('C:/Windows/win.ini', '..\\..\\secret', 'NoSuchFont', '', None):
            with self.subTest(family=family):
                cands = font_candidates(family)
                self.assertEqual(cands, font_candidates('Arial'))
                self.assertFalse(any('/' in c or '\\' in c for c in cands))
        for prefer in ('ko', 'ja'):
            for cjk_char in (True, False):
                with self.subTest(prefer=prefer, cjk_char=cjk_char):
                    self.assertFalse(any('/' in c or '\\' in c for c in fallback_candidates(prefer, cjk_char)))

    def test_script_preference_puts_japanese_fonts_first_for_kana(self):
        self.assertEqual(script_preference('申し込み'), 'ja')
        self.assertEqual(script_preference('ｱｲｳ'), 'ja', '반각 가타카나')
        self.assertEqual(script_preference('働'), 'ko', '가나가 없으면 한국어 우선')
        self.assertEqual(script_preference('홍길동'), 'ko')
        self.assertEqual(cjk_font_files('ko'), CJK_FONT_FILES)
        self.assertEqual(cjk_font_files('ko')[:len(KOREAN_FONT_FILES)], KOREAN_FONT_FILES)
        self.assertEqual(cjk_font_files('ja')[:len(JAPANESE_FONT_FILES)], JAPANESE_FONT_FILES)
        self.assertEqual(set(cjk_font_files('ja')), set(cjk_font_files('ko')))

    def test_fallback_order_depends_on_the_character(self):
        cjk = fallback_candidates('ko', True)
        other = fallback_candidates('ko', False)
        self.assertEqual(cjk[0], KOREAN_FONT_FILES[0], '한중일 글자는 한국어 글꼴부터')
        self.assertEqual(other[0], font_candidates('Arial')[0], '기호·라틴 확장은 기본 글꼴부터')
        self.assertEqual(set(cjk), set(other))
        self.assertEqual(len(cjk), len({c.lower() for c in cjk}), '중복 없음')

    def test_requested_size_is_kept_even_when_no_font_file_is_found(self):
        font = open_font(None, 33)
        self.assertEqual(getattr(font, 'size', None), 33, '폴백이 10px 기본 글꼴로 떨어졌다')
        font = open_font('Z:/no/such/font.ttf', 21)
        self.assertEqual(getattr(font, 'size', None), 21)

    def test_real_font_files_are_found_on_this_machine(self):
        path = resolve_font_path('Times New Roman')
        if path is None:
            self.skipTest('시스템 글꼴 없음')
        self.assertTrue(Path(path).is_file())
        font = open_font(path, 48)
        self.assertEqual(font.size, 48)
        self.assertEqual(Path(font.path).name.lower(), Path(path).name.lower())


def _is_notdef(path, ch: str) -> bool:
    """``path`` 글꼴에서 ``ch`` 가 .notdef(두부)로 그려지는지 — 모듈 판정과 독립적으로 Pillow 로 직접 본다."""
    font = ImageFont.truetype(path, 40) if path else ImageFont.load_default(size=40)

    def sig(c):
        mask = font.getmask(c)
        return mask.size, bytes(mask)

    return sig(ch) == sig('\U0010FFFF')


class GlyphFallbackTests(unittest.TestCase):
    """글자마다 실제 글리프가 있는 글꼴로 그려지는지 (예전: 맑은 고딕 고정이라 일본 한자가 두부)."""

    def _need(self, *chars):
        for ch in chars:
            if not any(ch in covered_chars(p, ch) for p in fallback_font_paths('ja', True)):
                self.skipTest(f'{ch!r} 를 담은 글꼴이 이 PC 에 없다')

    def test_coverage_probe_detects_missing_glyphs(self):
        arial = resolve_font_path('Arial')
        if arial is None or Path(arial).name.lower() != 'arial.ttf':
            self.skipTest('Arial 없음')
        self.assertEqual(covered_chars(arial, 'AB가込'), frozenset('AB'))

    def test_japanese_text_is_never_drawn_as_tofu(self):
        for text in ('申し込み', '働く', '込'):
            self._need(*text)
            with self.subTest(text=text):
                runs = plan_font_runs('Arial', text)
                self.assertEqual(''.join(run for _, run in runs), text)
                for path, run in runs:
                    for ch in run:
                        self.assertFalse(_is_notdef(path, ch), f'{ch!r} 가 {path} 에서 두부')
                self.assertEqual(len(runs), 1, '가나가 있는 일본어 한 단어는 한 글꼴로(자형이 섞이지 않게)')

    def test_malgun_family_still_falls_back_for_kanji_it_lacks(self):
        self._need('込')
        malgun = resolve_font_path('맑은 고딕')
        if malgun is None or Path(malgun).name.lower() != 'malgun.ttf':
            self.skipTest('맑은 고딕 없음')
        runs = plan_font_runs('맑은 고딕', '가込')
        self.assertEqual(runs[0], (malgun, '가'))
        self.assertNotEqual(runs[1][0], malgun)
        self.assertFalse(_is_notdef(runs[1][0], '込'))

    def test_mixed_korean_and_japanese_uses_two_fonts(self):
        self._need('働', 'く', '홍')
        runs = plan_font_runs('Arial', '홍길동 働く')
        self.assertEqual(''.join(run for _, run in runs), '홍길동 働く')
        self.assertEqual(len(runs), 2)
        for path, run in runs:
            for ch in run.strip():
                self.assertFalse(_is_notdef(path, ch), f'{ch!r} 가 {path} 에서 두부')

    def test_latin_keeps_the_chosen_font_next_to_hangul(self):
        arial = resolve_font_path('Arial')
        self._need('가')
        runs = plan_font_runs('Arial', 'Hello 가나다')
        self.assertEqual(runs[0], (arial, 'Hello '), '공백은 앞 글자의 글꼴에 붙는다')
        self.assertEqual(runs[1][1], '가나다')
        self.assertNotEqual(runs[1][0], arial)

    def test_combining_mark_stays_with_its_base_letter(self):
        arial = resolve_font_path('Arial')
        if arial is None:
            self.skipTest('Arial 없음')
        self.assertEqual(plan_font_runs('Arial', 'e\u0301'), ((arial, 'e\u0301'),))

    def test_rendered_kanji_differ_from_each_other(self):
        self._need('込', '働')
        a = rasterize_text('込', 'Arial', 48)
        b = rasterize_text('働', 'Arial', 48)
        self.assertFalse(a.shape == b.shape and np.array_equal(a, b), '두 한자가 같은 두부 글리프로 그려졌다')

    def test_multiline_text_stacks_lines(self):
        if resolve_font_path('Arial') is None:
            self.skipTest('Arial 없음')
        one = rasterize_text('Line', 'Arial', 30)
        two = rasterize_text('Line\nLine', 'Arial', 30)
        self.assertGreater(two.shape[0], one.shape[0] * 1.8)
        self.assertEqual(two.shape[1], one.shape[1])


class FrontendContractTests(unittest.TestCase):
    """패널이 보내는 글꼴 표시명·이미지 크기 단위가 백엔드와 맞는지."""

    FRONT = Path(__file__).resolve().parents[1] / 'frontend' / 'src'

    def test_every_panel_font_is_a_known_display_name(self):
        import re
        from core.editor_watermark import FONT_FILES
        source = (self.FRONT / 'utils' / 'watermark.ts').read_text(encoding='utf-8')
        block = re.search(r'WATERMARK_FONTS[^=]*=\s*\[(.*?)\]', source, re.S)
        self.assertIsNotNone(block, 'WATERMARK_FONTS 를 찾지 못했다')
        names = re.findall(r"'([^']+)'", block.group(1))
        self.assertGreaterEqual(len(names), 5)
        for name in names:
            with self.subTest(font=name):
                self.assertIn(name.lower(), FONT_FILES, f"'{name}' 가 백엔드 FONT_FILES 에 없다 — 늘 기본 글꼴이 된다")

    def test_panel_sends_image_scale_in_percent(self):
        panel = (self.FRONT / 'components' / 'editor' / 'WatermarkPanel.vue').read_text(encoding='utf-8')
        self.assertIn('scale: imgScale.value,', panel)
        self.assertNotIn('imgScale.value / 100', panel)

    def test_editor_passes_the_watermark_image_to_the_panel(self):
        import re
        view = (self.FRONT / 'views' / 'EditorView.vue').read_text(encoding='utf-8')
        self.assertRegex(view, re.compile(r'<WatermarkPanel[^>]*:image-path="wmImagePath"', re.S))


class HelperTests(unittest.TestCase):
    def test_parse_hex_color(self):
        self.assertEqual(parse_hex_color('#FF8000'), (255, 128, 0))
        self.assertEqual(parse_hex_color('abc'), (170, 187, 204))
        self.assertEqual(parse_hex_color('nope'), (255, 255, 255))
        self.assertEqual(parse_hex_color(None, (1, 2, 3)), (1, 2, 3))

    def test_centered_position_and_clamp(self):
        self.assertEqual(centered_position(50, 100, 20, True), 40)
        self.assertEqual(centered_position(100, 100, 20, True), 80)
        self.assertEqual(centered_position(100, 100, 20, False), 90)
        self.assertEqual(centered_position(0, 100, 20, True), 0)
        self.assertEqual(centered_position(50, 10, 20, True), 0, '이미지보다 큰 워터마크는 0 에 붙는다')

    def test_rotated_extent(self):
        self.assertEqual(rotated_extent(80, 12, 0), (80, 12))
        w, h = rotated_extent(80, 12, 90)
        self.assertAlmostEqual(w, 12)
        self.assertAlmostEqual(h, 80)
        w, h = rotated_extent(80, 12, 45)
        self.assertAlmostEqual(w, 92 / math.sqrt(2), places=6)
        self.assertAlmostEqual(h, 92 / math.sqrt(2), places=6)
        for a, b in zip(rotated_extent(80, 12, -135), rotated_extent(80, 12, 45)):
            self.assertAlmostEqual(a, b, places=6)

    def test_rotated_origin_keeps_the_rotated_box_inside(self):
        # 돌린 상자(폭 60)가 [0, 200] 안에 들도록 **중심**을 민다 — 돌리기 전 도장 폭은 80
        for pct in (0, 5, 50, 95, 100):
            with self.subTest(pct=pct):
                x = rotated_origin(pct, 200, 80, 60, True)
                centre = x + 40
                self.assertGreaterEqual(centre - 30, 0)
                self.assertLessEqual(centre + 30, 200)
        self.assertEqual(rotated_origin(50, 200, 80, 60, True), 60)
        self.assertEqual(rotated_origin(100, 200, 80, 60, False), 160, 'clamp 를 끄면 밀지 않는다')
        self.assertEqual(rotated_origin(0, 40, 80, 60, True), -20, '상자가 이미지보다 크면 가운데')


class TextWatermarkTests(unittest.TestCase):
    def setUp(self):
        if resolve_font_path('Arial') is None:
            self.skipTest('시스템 글꼴 없음')

    def test_bgra_input_keeps_transparency(self):
        img = np.zeros((80, 160, 4), np.uint8)           # 완전 투명
        out = render_text_watermark(img, _text())
        self.assertEqual(out.shape, (80, 160, 4), 'BGRA 가 BGR 로 떨어졌다')
        self.assertEqual(int(out[0, 0, 3]), 0, '글자 밖 투명 픽셀이 불투명해졌다')
        self.assertGreater(int(out[:, :, 3].max()), 200, '글자가 그려지지 않았다')

    def test_bgr_input_stays_bgr(self):
        img = np.zeros((60, 120, 3), np.uint8)
        self.assertEqual(render_text_watermark(img, _text()).shape, (60, 120, 3))

    def test_font_size_is_applied(self):
        img = np.zeros((200, 400, 3), np.uint8)
        small = _bbox(_ink(img, render_text_watermark(img, _text(fontSize=12))))
        big = _bbox(_ink(img, render_text_watermark(img, _text(fontSize=72))))
        self.assertGreater(big[3] - big[1], 3 * (small[3] - small[1]))

    def test_font_family_is_applied(self):
        if resolve_font_path('Courier New') is None:
            self.skipTest('Courier New 없음')
        img = np.zeros((120, 400, 3), np.uint8)
        a = render_text_watermark(img, _text(text='Watermark', fontFamily='Arial'))
        b = render_text_watermark(img, _text(text='Watermark', fontFamily='Courier New'))
        self.assertFalse(np.array_equal(a, b), '글꼴 선택이 결과에 반영되지 않는다')

    def test_korean_text_is_not_tofu(self):
        if not fallback_font_paths('ko', True):
            self.skipTest('CJK 글꼴 없음')
        img = np.zeros((120, 400, 3), np.uint8)
        a = render_text_watermark(img, _text(text='가나다'))
        b = render_text_watermark(img, _text(text='라마바'))
        self.assertFalse(np.array_equal(a, b), '한글이 모두 같은 두부 글리프로 그려졌다')

    def test_japanese_kanji_watermark_is_not_tofu(self):
        if not any('込' in covered_chars(p, '込') for p in fallback_font_paths('ja', True)):
            self.skipTest('일본 한자 글꼴 없음')
        img = np.zeros((120, 400, 3), np.uint8)
        a = render_text_watermark(img, _text(text='申し込み'))
        b = render_text_watermark(img, _text(text='申し働み'))
        self.assertFalse(np.array_equal(a, b), "'込'·'働' 이 같은 두부 글리프로 그려졌다")

    def test_color_and_no_dark_fringe(self):
        white = np.full((80, 200, 3), 255, np.uint8)
        out = render_text_watermark(white, _text(color='#FFFFFF', opacity=1.0))
        self.assertEqual(int(out.min()), 255, '흰 바탕의 흰 글자 가장자리가 어두워졌다(알파 이중 적용)')
        black = np.zeros((80, 200, 3), np.uint8)
        red = render_text_watermark(black, _text(color='#FF0000'))
        ink = _ink(black, red)
        self.assertGreater(int(red[..., 2][ink].max()), 200)
        self.assertEqual(int(red[..., 0].max()), 0)

    def test_opacity_scales_alpha(self):
        black = np.zeros((80, 200, 3), np.uint8)
        full = render_text_watermark(black, _text(opacity=1.0))
        half = render_text_watermark(black, _text(opacity=0.5))
        self.assertAlmostEqual(int(half.max()), 128, delta=2)
        self.assertEqual(int(full.max()), 255)

    def test_clamp_keeps_corner_text_inside(self):
        img = np.zeros((100, 300, 3), np.uint8)
        centred = int(_ink(img, render_text_watermark(img, _text(text='CORNER'))).sum())
        clamped = int(_ink(img, render_text_watermark(img, _text(text='CORNER', xPct=100, yPct=100))).sum())
        cut = int(_ink(img, render_text_watermark(img, _text(text='CORNER', xPct=100, yPct=100, clamp=False))).sum())
        self.assertAlmostEqual(clamped, centred, delta=centred * 0.05)
        self.assertLess(cut, centred * 0.6)

    def test_clamp_keeps_rotated_corner_text_inside(self):
        # 예전엔 돌리기 전 크기로 밀어서, 모서리의 돌린 글자 끝이 잘렸다(45° 에서 20% 남짓)
        img = np.zeros((200, 200, 3), np.uint8)

        def ink(**kw):
            return int(render_text_watermark(img, _text(text='CORNER', fontSize=16, **kw)).astype(np.int64).sum())

        for rotation in (30, 45, 90, -45, 135):
            centred = ink(rotation=rotation)
            for x_pct, y_pct in ((0, 0), (100, 100), (0, 100), (100, 0), (95, 95), (5, 5)):
                with self.subTest(rotation=rotation, x=x_pct, y=y_pct):
                    self.assertAlmostEqual(ink(rotation=rotation, xPct=x_pct, yPct=y_pct), centred,
                                           delta=centred * 0.03)
        # clamp 를 끄면 잘린다 — 위 결과가 clamp 덕분임을 확인
        self.assertLess(ink(rotation=45, xPct=100, yPct=100, clamp=False), ink(rotation=45) * 0.8)

    def test_unrotated_and_half_turn_clamp_is_unchanged(self):
        # 0°·180° 는 예전 centered_position 경로 그대로(픽셀 단위로 같은 자리)
        img = np.zeros((120, 300, 3), np.uint8)
        for rotation in (0, 180, -180):
            with self.subTest(rotation=rotation):
                box = _bbox(_ink(img, render_text_watermark(img, _text(text='EDGE', xPct=100, yPct=100,
                                                                         rotation=rotation))))
                self.assertLessEqual(box[2], 299)
                self.assertLessEqual(box[3], 119)
                self.assertGreater(box[2], 290, '0° 경로가 필요 이상으로 안쪽으로 밀렸다')

    def test_rotation_turns_around_the_text_not_the_image(self):
        img = np.zeros((400, 400, 3), np.uint8)
        params = _text(text='ROT', xPct=85, yPct=85)
        straight = _bbox(_ink(img, render_text_watermark(img, params)))
        turned = _bbox(_ink(img, render_text_watermark(img, dict(params, rotation=90))))
        c0 = ((straight[0] + straight[2]) / 2, (straight[1] + straight[3]) / 2)
        c1 = ((turned[0] + turned[2]) / 2, (turned[1] + turned[3]) / 2)
        self.assertLess(abs(c0[0] - c1[0]) + abs(c0[1] - c1[1]), 12)
        self.assertGreater(turned[3] - turned[1], turned[2] - turned[0], '90° 회전인데 여전히 가로로 길다')

    def test_rotated_tiles_cover_every_corner(self):
        img = np.zeros((240, 240, 3), np.uint8)
        ink = _ink(img, render_text_watermark(img, _text(fontSize=20, tile=True, rotation=45)))
        for name, quad in (('tl', ink[:40, :40]), ('tr', ink[:40, -40:]),
                           ('bl', ink[-40:, :40]), ('br', ink[-40:, -40:])):
            with self.subTest(corner=name):
                self.assertTrue(quad.any(), f'{name} 모서리에 타일이 없다')

    def test_pixel_scale_matches_downscaled_preview(self):
        big = np.zeros((800, 800, 3), np.uint8)
        small = np.zeros((200, 200, 3), np.uint8)
        b = _bbox(_ink(big, render_text_watermark(big, _text(fontSize=120))))
        s = _bbox(_ink(small, render_text_watermark(small, _text(fontSize=120), pixel_scale=0.25)))
        self.assertAlmostEqual((s[3] - s[1]) / (b[3] - b[1]), 0.25, delta=0.05)

    def test_empty_text_is_an_error(self):
        with self.assertRaises(WatermarkError):
            render_text_watermark(np.zeros((10, 10, 3), np.uint8), _text(text='   '))

    def test_sixteen_bit_input_does_not_crash(self):
        img = np.zeros((40, 80, 3), np.uint16)
        self.assertEqual(render_text_watermark(img, _text(fontSize=12)).dtype, np.uint8)

    def test_rotated_tiles_do_not_build_a_padded_canvas(self):
        """예전: (w+2·pad)×(h+2·pad) RGBA 캔버스를 통째로 돌렸다 — 8K 에서 ≈439MB + 회전 사본."""
        width, height = 320, 90
        created = []
        real_new = Image.new

        def spy_new(mode, size, *args, **kwargs):
            created.append((mode, tuple(size)))
            return real_new(mode, size, *args, **kwargs)

        with mock.patch.object(Image, 'new', side_effect=spy_new), \
                mock.patch.object(Image.Image, 'rotate', side_effect=AssertionError('이미지 크기 캔버스를 돌렸다')):
            out = render_text_watermark(np.zeros((height, width, 3), np.uint8),
                                        _text(fontSize=16, tile=True, rotation=33))
        self.assertTrue(out.any())
        for mode, (w, h) in created:
            with self.subTest(mode=mode, size=(w, h)):
                self.assertLess(w * h, width * height, 'PIL 로 이미지 크기 이상의 캔버스를 만들었다')


def _test_stamp() -> np.ndarray:
    """좌우·상하가 비대칭인 합성 도장(12×30) — 회전 방향이 틀리면 무늬가 달라진다."""
    stamp = np.zeros((12, 30), np.uint8)
    stamp[2:10, 3:27] = 255
    stamp[4:8, 10:14] = 0
    stamp[2:5, 3:8] = 128
    return stamp


GAP_X, GAP_Y = 7, 5   # 가로·세로 간격을 다르게 — 축이 뒤바뀌면 무늬가 달라진다


def _reference_tiles(stamp: np.ndarray, width: int, height: int, rotation: float) -> np.ndarray:
    """예전 구현(대각선 패딩 캔버스를 깔고 PIL 로 돌린 뒤 잘라 냄) — 무늬 비교 기준."""
    th, tw = stamp.shape
    src = Image.fromarray(stamp)
    step_x, step_y = tw + GAP_X, th + GAP_Y
    pad = 0
    if rotation % 360:
        pad = int(math.ceil((math.hypot(width, height) - min(width, height)) / 2.0)) + max(tw, th)
    canvas = Image.new('L', (width + 2 * pad, height + 2 * pad), 0)
    first_x = -tw - int(math.ceil(pad / step_x)) * step_x
    first_y = -th - int(math.ceil(pad / step_y)) * step_y
    for ty in range(first_y, height + th + pad, step_y):
        for tx in range(first_x, width + tw + pad, step_x):
            canvas.paste(src, (tx + pad, ty + pad))
    if pad:
        canvas = canvas.rotate(-rotation, resample=Image.BICUBIC, center=(pad + width / 2.0, pad + height / 2.0))
        canvas = canvas.crop((pad, pad, pad + width, pad + height))
    return np.asarray(canvas)


def _reference_placed(stamp: np.ndarray, width: int, height: int, x: int, y: int, rotation: float) -> np.ndarray:
    """예전 구현(이미지 크기 오버레이에 붙이고 도장 중심으로 PIL 회전)."""
    th, tw = stamp.shape
    overlay = Image.new('L', (width, height), 0)
    overlay.paste(Image.fromarray(stamp), (x, y))
    if rotation % 360:
        overlay = overlay.rotate(-rotation, resample=Image.BICUBIC, center=(x + tw / 2.0, y + th / 2.0))
    return np.asarray(overlay)


def _iou(a: np.ndarray, b: np.ndarray, threshold: int = 60) -> float:
    ia, ib = a > threshold, b > threshold
    union = int((ia | ib).sum())
    return 1.0 if union == 0 else int((ia & ib).sum()) / union


class TileSamplingTests(unittest.TestCase):
    """회전 타일·배치를 출력 픽셀에서 거꾸로 샘플링 — 메모리 O(출력), 무늬는 예전과 같다."""

    def _tiles(self, w, h, rotation):
        stamp = _test_stamp()
        got = tiled_alpha(stamp, w, h, stamp.shape[1] + GAP_X, stamp.shape[0] + GAP_Y, rotation)
        return got, _reference_tiles(stamp, w, h, rotation)

    def test_unrotated_and_quarter_turn_tiles_match_the_old_canvas_exactly(self):
        # w−h 가 짝수면 90° 회전이 픽셀 중심을 픽셀 중심으로 보낸다(보간 커널 차이가 안 끼어든다).
        # 홀수면 반 픽셀 자리를 읽어 PIL(a=−0.5)·OpenCV(a=−0.75) 3차 커널 차이만큼 다를 수 있다.
        for (w, h) in ((130, 76), (96, 64), (333, 211)):
            for rotation in (0, 90, -90, 180, 270):
                with self.subTest(size=(w, h), rotation=rotation):
                    got, ref = self._tiles(w, h, rotation)
                    self.assertTrue(np.array_equal(got, ref))
        got, ref = self._tiles(130, 77, 90)
        self.assertGreater(_iou(got, ref), 0.95)

    def test_rotated_tiles_match_the_old_canvas_pattern(self):
        for (w, h, rotation) in ((200, 120, 30), (150, 260, -17), (240, 240, 45)):
            with self.subTest(size=(w, h), rotation=rotation):
                got, ref = self._tiles(w, h, rotation)
                self.assertGreater(_iou(got, ref), 0.9)
                self.assertLess(float(np.abs(got.astype(int) - ref).mean()), 3.0)

    def test_long_thin_image_is_covered_to_every_corner(self):
        """한 축 패딩을 두 축에 다 쓰던 예전 캔버스 대신 — 긴 축 끝 모서리도 비지 않아야 한다."""
        stamp = _test_stamp()
        alpha = tiled_alpha(stamp, 900, 40, stamp.shape[1] + GAP_X, stamp.shape[0] + GAP_Y, 45)
        for name, quad in (('tl', alpha[:20, :40]), ('tr', alpha[:20, -40:]),
                           ('bl', alpha[-20:, :40]), ('br', alpha[-20:, -40:])):
            with self.subTest(corner=name):
                self.assertTrue(quad.any(), f'{name} 모서리에 타일이 없다')

    def test_sampling_does_not_depend_on_the_block_size(self):
        # 블록마다 좌표 원점만 옮겨 float32 로 넘기므로, 1/32 픽셀 양자화 경계에서 ±1 까지만 다를 수 있다
        rng = np.random.default_rng(3)
        src = rng.integers(0, 256, (37, 53), dtype=np.uint8)
        c, s = math.cos(math.radians(23)), math.sin(math.radians(23))
        matrix = (c, s, -40.3, -s, c, 71.9)
        for wrap in (True, False):
            with self.subTest(wrap=wrap):
                small = sample_affine(src, (0, 0, 150, 90), matrix, wrap=wrap, block=16)
                large = sample_affine(src, (0, 0, 150, 90), matrix, wrap=wrap, block=4096)
                diff = np.abs(small.astype(int) - large)
                self.assertLessEqual(int(diff.max()), 2)
                self.assertLess(float((diff > 0).mean()), 0.01)
                self.assertTrue(small.any())

    def test_wrapped_sampling_is_periodic(self):
        rng = np.random.default_rng(4)
        src = rng.integers(0, 256, (9, 13), dtype=np.uint8)
        whole = sample_affine(src, (0, 0, 13 * 3, 9 * 2), (1, 0, 0, 0, 1, 0), wrap=True)
        self.assertTrue(np.array_equal(whole, np.tile(src, (2, 3))), '정수 이동은 주기 복사와 같아야 한다')
        shifted = sample_affine(src, (0, 0, 13, 9), (1, 0, 13 * 5 + 2, 0, 1, -9 * 4), wrap=True)
        self.assertTrue(np.array_equal(shifted, np.roll(src, -2, axis=1)))

    def test_placed_stamp_matches_the_old_overlay_rotation(self):
        stamp = _test_stamp()
        for rotation in (0, 90, -90):
            with self.subTest(rotation=rotation):
                alpha, (ox, oy) = placed_alpha(stamp, 120, 80, 50, 20, rotation)
                full = np.zeros((80, 120), np.uint8)
                full[oy:oy + alpha.shape[0], ox:ox + alpha.shape[1]] = alpha
                self.assertTrue(np.array_equal(full, _reference_placed(stamp, 120, 80, 50, 20, rotation)))
        alpha, (ox, oy) = placed_alpha(stamp, 120, 80, 50, 20, 30)
        full = np.zeros((80, 120), np.uint8)
        full[oy:oy + alpha.shape[0], ox:ox + alpha.shape[1]] = alpha
        self.assertGreater(_iou(full, _reference_placed(stamp, 120, 80, 50, 20, 30)), 0.9)
        self.assertLessEqual(alpha.shape[0] * alpha.shape[1], 50 * 50, '도장이 닿는 사각형만 계산해야 한다')

    def test_placed_stamp_outside_the_image_is_empty(self):
        alpha, _origin = placed_alpha(_test_stamp(), 40, 40, 500, 500, 15)
        self.assertEqual(alpha.size, 0)

    def test_stamps_wider_than_16bit_coordinates_still_work(self):
        """OpenCV remap 은 좌표를 16비트로 다룬다 — 창을 잘라 넘기므로 아주 긴 글자도 된다."""
        stamp = np.zeros((3, 40000), np.uint8)
        stamp[:, 35000] = 255
        alpha, (ox, oy) = placed_alpha(stamp, 64, 16, -34990, 0, 0)
        self.assertEqual(int(alpha[1 - oy, 10 - ox]), 255)
        self.assertEqual(int(alpha.sum()), 255 * 3)
        tiled = tiled_alpha(stamp, 64, 8, 40001, 4, 0)
        self.assertEqual(tiled.shape, (8, 64))


class BlendOverTests(unittest.TestCase):
    def test_opaque_base_is_an_exact_over(self):
        rng = np.random.default_rng(5)
        base = rng.integers(0, 256, (40, 70, 3), dtype=np.uint8)
        alpha = rng.integers(0, 256, (40, 70), dtype=np.uint8)
        colour = rng.integers(0, 256, (40, 70, 3), dtype=np.uint8)
        a = alpha.astype(np.float64)[..., None]
        for src in ((30, 200, 90), colour):
            with self.subTest(per_pixel=isinstance(src, np.ndarray)):
                out = base.copy()
                blend_over(out, (0, 0), alpha, src)
                c = np.asarray(src, np.float64)
                expected = np.rint((base * (255.0 - a) + c * a) / 255.0).astype(np.uint8)
                self.assertTrue(np.array_equal(out, expected))

    def test_transparent_base_matches_pil_alpha_composite(self):
        rng = np.random.default_rng(6)
        base = rng.integers(0, 256, (30, 50, 4), dtype=np.uint8)
        alpha = rng.integers(0, 256, (30, 50), dtype=np.uint8)
        out = base.copy()
        blend_over(out, (0, 0), alpha, (10, 120, 240))
        dst = Image.fromarray(cv2.cvtColor(base, cv2.COLOR_BGRA2RGBA))
        src = np.zeros((30, 50, 4), np.uint8)
        src[..., :3] = (240, 120, 10)
        src[..., 3] = alpha
        ref = cv2.cvtColor(np.asarray(Image.alpha_composite(dst, Image.fromarray(src))), cv2.COLOR_RGBA2BGRA)
        self.assertLessEqual(int(np.abs(out[..., 3].astype(int) - ref[..., 3]).max()), 1)
        visible = ref[..., 3] > 8
        self.assertLessEqual(int(np.abs(out[..., :3].astype(int) - ref[..., :3])[visible].max()), 2)

    def test_zero_alpha_and_offset_leave_other_pixels_untouched(self):
        base = np.zeros((20, 20, 4), np.uint8)
        base[..., :3] = 77                      # 투명 픽셀 아래 숨은 색 — 그대로 남아야 한다
        alpha = np.zeros((4, 5), np.uint8)
        alpha[1, 2] = 255
        out = base.copy()
        blend_over(out, (7, 5), alpha, (255, 255, 255))
        changed = np.argwhere(np.any(out != base, axis=2))
        self.assertEqual(changed.tolist(), [[6, 9]])
        self.assertEqual(out[6, 9].tolist(), [255, 255, 255, 255])


class ImageWatermarkTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.wm_path = Path(self._tmp.name) / '워터마크_ロゴ.png'
        Image.new('RGBA', (20, 10), (255, 255, 255, 255)).save(self.wm_path)

    def _params(self, **kw):
        params = {'watermark_path': str(self.wm_path), 'scale': 100, 'opacity': 1.0,
                  'xPct': 50, 'yPct': 50, 'clamp': True}
        params.update(kw)
        return params

    def test_scale_is_a_percentage(self):
        base = np.zeros((100, 100, 3), np.uint8)
        full = _ink(base, render_image_watermark(base, self._params(scale=100)))
        half = _ink(base, render_image_watermark(base, self._params(scale=50)))
        self.assertEqual(int(full.sum()), 200, '100% 가 원래 크기가 아니다(예전엔 1%)')
        self.assertEqual(int(half.sum()), 50)

    def test_opacity_is_not_squared(self):
        base = np.zeros((100, 100, 3), np.uint8)
        out = render_image_watermark(base, self._params(opacity=0.5))
        self.assertAlmostEqual(int(out.max()), 128, delta=2, msg='불투명도가 제곱으로 떨어졌다(예전 64)')

    def test_bgra_base_keeps_transparency(self):
        base = np.zeros((100, 100, 4), np.uint8)
        out = render_image_watermark(base, self._params())
        self.assertEqual(out.shape, (100, 100, 4))
        self.assertEqual(int(out[0, 0, 3]), 0)
        self.assertEqual(int(out[50, 50, 3]), 255)

    def test_pixel_scale_and_clamp(self):
        base = np.zeros((100, 100, 3), np.uint8)
        scaled = _ink(base, render_image_watermark(base, self._params(), pixel_scale=0.5))
        self.assertEqual(int(scaled.sum()), 50)
        tiny = np.zeros((5, 5, 3), np.uint8)
        fit = _ink(tiny, render_image_watermark(tiny, self._params(scale=500)))
        self.assertTrue(fit.any())
        corner = _ink(base, render_image_watermark(base, self._params(xPct=100, yPct=100)))
        self.assertEqual(_bbox(corner), (80, 90, 99, 99))

    def test_missing_or_broken_watermark_is_an_error(self):
        base = np.zeros((10, 10, 3), np.uint8)
        with self.assertRaises(WatermarkError):
            render_image_watermark(base, self._params(watermark_path=''))
        broken = Path(self._tmp.name) / 'broken.png'
        broken.write_bytes(b'not a png')
        with self.assertRaises(WatermarkError):
            render_image_watermark(base, self._params(watermark_path=str(broken)))

    def _noise_watermark(self, size=(300, 200)) -> tuple[Path, Image.Image]:
        rng = np.random.default_rng(11)
        pixels = rng.integers(0, 256, (size[1], size[0], 4), dtype=np.uint8)
        wm = Image.fromarray(pixels, 'RGBA')
        path = Path(self._tmp.name) / '노이즈.png'
        wm.save(path)
        return path, wm

    def test_unclamped_large_scale_matches_the_full_resize_in_view(self):
        """clamp 를 끈 500% — 보이는 부분만 리사이즈해도 전체를 키워 자른 것과 같아야 한다."""
        path, wm = self._noise_watermark()
        rng = np.random.default_rng(12)
        base = rng.integers(0, 256, (80, 100, 3), dtype=np.uint8)
        params = self._params(watermark_path=str(path), scale=500, clamp=False, xPct=30, yPct=70, opacity=0.8)
        out = render_image_watermark(base, params)

        full = np.asarray(wm.resize((1500, 1000), Image.LANCZOS)).astype(np.float64)
        x = centered_position(30, 100, 1500, False)
        y = centered_position(70, 80, 1000, False)
        view = full[-y:-y + 80, -x:-x + 100]
        a = np.rint(view[..., 3] * 0.8)[..., None]
        colour = view[..., 2::-1]
        expected = np.rint((base * (255.0 - a) + colour * a) / 255.0).astype(np.uint8)
        self.assertLessEqual(int(np.abs(out.astype(int) - expected).max()), 1)

    def test_unclamped_resize_never_exceeds_the_base_image(self):
        path, _wm = self._noise_watermark((1200, 900))
        base = np.zeros((60, 80, 3), np.uint8)
        sizes = []
        real_resize = Image.Image.resize

        def spy(img, size, *args, **kwargs):
            sizes.append(tuple(size))
            return real_resize(img, size, *args, **kwargs)

        with mock.patch.object(Image.Image, 'resize', autospec=True, side_effect=spy):
            render_image_watermark(base, self._params(watermark_path=str(path), scale=500, clamp=False))
        self.assertTrue(sizes, '리사이즈가 불리지 않았다')
        for w, h in sizes:
            self.assertLessEqual(w, 80, f'보이지 않는 부분까지 {w}×{h} 로 키웠다(예전 6000×4500)')
            self.assertLessEqual(h, 60)

    def test_resize_visible_matches_a_crop_of_the_full_resize(self):
        _path, wm = self._noise_watermark((90, 70))
        for size, crop in (((450, 350), (37, 11, 120, 90)), ((45, 35), (3, 4, 40, 30)),
                           ((90, 70), (10, 10, 50, 60)), ((200, 150), (0, 0, 200, 150))):
            with self.subTest(size=size, crop=crop):
                part = np.asarray(resize_visible(wm, size, crop)).astype(int)
                ref = np.asarray(wm.resize(size, Image.LANCZOS).crop(crop)).astype(int)
                self.assertEqual(part.shape, ref.shape)
                self.assertLessEqual(int(np.abs(part - ref).max()), 1)


class BridgeWatermarkWiringTests(unittest.TestCase):
    """VueBridge 가 알파를 보존하고 프리뷰 배율을 워터마크에 넘기는지."""

    def setUp(self):
        if resolve_font_path('Arial') is None:
            self.skipTest('시스템 글꼴 없음')
        from ui.vue_bridge import VueBridge
        self.bridge = VueBridge()
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.root = Path(self._tmp.name)
        self.bridge._editor_temp_dir = str(self.root / 'editor_temp')

    def _commit(self, path: Path, op: str, params: dict) -> dict:
        with mock.patch('core.cache_cleanup.prune_editor_temp'):
            return json.loads(self.bridge._editor_process_impl(str(path), op, params))

    def test_commit_on_transparent_png_keeps_alpha(self):
        src = self.root / '투명 원본.png'
        imwrite_unicode(src, np.zeros((64, 128, 4), np.uint8))
        result = self._commit(src, 'text_watermark', _text(fontSize=20))
        self.assertIn('path', result, result)
        out = imread_unicode(result['path'], cv2.IMREAD_UNCHANGED)
        self.assertEqual(out.shape, (64, 128, 4))
        self.assertEqual(int(out[0, 0, 3]), 0)
        self.assertGreater(int(out[..., 3].max()), 200)

    def test_preview_text_is_scaled_like_the_commit(self):
        src = self.root / 'big.png'
        base = np.zeros((2048, 2048, 3), np.uint8)
        imwrite_unicode(src, base)
        params = _text(fontSize=200)
        committed = imread_unicode(self._commit(src, 'text_watermark', dict(params))['path'])
        full_h = _bbox(_ink(base, committed))
        preview = json.loads(self.bridge._editor_process_impl(str(src), 'text_watermark', dict(params, preview=True)))
        self.assertTrue(preview.get('preview'), preview)
        import base64
        raw = base64.b64decode(preview['image_base64'].split(',', 1)[1])
        shown = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
        self.assertEqual(shown.shape[:2], (1024, 1024))
        small = np.zeros_like(shown)
        pv = _bbox(_ink(small, shown) & (shown.max(axis=2) > 60))
        ratio = (pv[3] - pv[1]) / (full_h[3] - full_h[1])
        self.assertAlmostEqual(ratio, 0.5, delta=0.08, msg='프리뷰 글자 크기가 원본 비율과 다르다')

    def test_empty_text_is_reported_not_crashing(self):
        src = self.root / 'a.png'
        imwrite_unicode(src, np.zeros((16, 16, 3), np.uint8))
        result = self._commit(src, 'text_watermark', _text(text=''))
        self.assertIn('error', result)


if __name__ == '__main__':
    unittest.main()
