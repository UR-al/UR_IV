"""에디터 저장(core.editor_save) 회귀 테스트.

예전 '저장'은 파일을 쓰지 않았고, '다른 이름으로 저장'은 JPEG/WebP 를 골라도 PNG 바이트를
그대로 복사했으며, 병합하지 않은 드로잉 레이어는 두 경로 모두에서 빠졌다.
"""
from __future__ import annotations

import base64
import io
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
from PIL import Image, PngImagePlugin

from core import editor_save
from core.editor_save import (
    EditorSaveError,
    SavedCopyRegistry,
    alpha_safe_target,
    edited_copy_path,
    ensure_extension,
    has_transparency,
    is_inside,
    is_user_image,
    read_source_metadata,
    save_edited_image,
    suggest_save_path,
    unique_path,
    validate_chosen_target,
)
from core.image_metadata import extract_from_file

PARAMS = '1girl, 한글 태그, solo\nNegative prompt: lowres\nSteps: 20, Sampler: Euler a, CFG scale: 5, Seed: 42, Size: 64x48'
WORKFLOW = '{"nodes": [{"id": 1, "type": "KSampler"}]}'


def _png_with_meta(path: Path, color=(10, 20, 30), size=(64, 48), mode='RGB') -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    info = PngImagePlugin.PngInfo()
    info.add_text('parameters', PARAMS)
    info.add_text('workflow', WORKFLOW)
    fill = color if mode == 'RGB' else (*color, 255)
    Image.new(mode, size, fill).save(path, pnginfo=info)
    return path


def _plain_png(path: Path, color=(200, 100, 50), size=(64, 48), mode='RGB') -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fill = color if mode == 'RGB' else (*color, 128)
    Image.new(mode, size, fill).save(path)
    return path


def _overlay_b64(size=(64, 48), box=(0, 0, 8, 8), color=(255, 0, 0, 255)) -> str:
    over = Image.new('RGBA', size, (0, 0, 0, 0))
    over.paste(color, box)
    buf = io.BytesIO()
    over.save(buf, 'PNG')
    return 'data:image/png;base64,' + base64.b64encode(buf.getvalue()).decode('ascii')


class PathRulesTests(unittest.TestCase):
    def test_is_inside_and_user_image(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cache = root / 'image_cache'
            user = _plain_png(root / 'photos' / 'a.png')
            cached = _plain_png(cache / 'editor_temp' / 'edited_x.png')
            self.assertTrue(is_inside(str(cached), [str(cache)]))
            self.assertFalse(is_inside(str(user), [str(cache)]))
            self.assertTrue(is_user_image(str(user), [str(cache)]))
            self.assertFalse(is_user_image(str(cached), [str(cache)]))
            # 쓸 수 없는 포맷·없는 파일은 사용자 이미지로 덮어쓰지 않는다
            bmp = root / 'photos' / 'b.bmp'
            Image.new('RGB', (4, 4)).save(bmp)
            self.assertFalse(is_user_image(str(bmp), [str(cache)]))
            self.assertFalse(is_user_image(str(root / 'nope.png'), [str(cache)]))
            self.assertFalse(is_user_image(None, [str(cache)]))

    def test_system_temp_is_not_a_user_location(self):
        dirs = editor_save.non_user_dirs('C:/app')
        self.assertIn(tempfile.gettempdir(), dirs)
        self.assertTrue(any(d.replace('\\', '/').endswith('image_cache') for d in dirs))

    def test_ensure_extension_follows_selected_filter(self):
        self.assertEqual(ensure_extension('C:/x/out', 'JPEG (*.jpg *.jpeg)'), 'C:/x/out.jpg')
        self.assertEqual(ensure_extension('C:/x/out', 'WebP (*.webp)'), 'C:/x/out.webp')
        self.assertEqual(ensure_extension('C:/x/out', ''), 'C:/x/out.png')
        self.assertEqual(ensure_extension('C:/x/out.JPG', 'PNG (*.png)'), 'C:/x/out.JPG')
        self.assertEqual(ensure_extension('C:/x/out.bmp', 'PNG (*.png)'), 'C:/x/out.bmp.png')

    def test_suggest_keeps_source_name_and_avoids_collisions(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            src = _plain_png(root / 'gen' / '00012-12345.png')
            suggested = suggest_save_path(str(src), str(root / 'out'), [str(root / 'image_cache')])
            self.assertEqual(Path(suggested).name, '00012-12345_edited.png')
            _plain_png(Path(suggested))
            again = suggest_save_path(str(src), str(root / 'out'), [str(root / 'image_cache')])
            self.assertEqual(Path(again).name, '00012-12345_edited_2.png')
            # 임시 원본(클립보드 등)은 기본 폴더에 날짜 이름
            temp_src = _plain_png(root / 'image_cache' / 'clip.png')
            fallback = suggest_save_path(str(temp_src), str(root / 'out'), [str(root / 'image_cache')], now=0)
            self.assertEqual(Path(fallback).parent, root / 'out')
            self.assertRegex(Path(fallback).name, r'^edited_\d{8}_\d{6}\.png$')
            # 사용자 폴더의 BMP — 덮어쓸 수는 없지만 추천은 원본 옆에 PNG 로
            bmp = root / 'gen' / 'scan.bmp'
            Image.new('RGB', (4, 4)).save(bmp)
            beside = suggest_save_path(str(bmp), str(root / 'out'), [str(root / 'image_cache')])
            self.assertEqual(Path(beside), root / 'gen' / 'scan_edited.png')

    def test_unique_path(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / 'a.png'
            self.assertEqual(unique_path(str(p)), str(p))
            p.write_bytes(b'x')
            self.assertEqual(Path(unique_path(str(p))).name, 'a_2.png')

    def test_edited_copy_of_an_edited_copy_bumps_the_number(self):
        """저장된 사본을 다시 열어 저장하면 '_edited_edited' 가 아니라 번호가 오른다."""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            excluded = [str(root / 'image_cache')]
            _plain_png(root / 'gen' / 'a.png')
            first = _plain_png(root / 'gen' / 'a_edited.png')
            self.assertEqual(Path(edited_copy_path(str(first), str(root), excluded)).name, 'a_edited_2.png')
            second = _plain_png(root / 'gen' / 'a_edited_2.png')
            self.assertEqual(Path(edited_copy_path(str(second), str(root), excluded)).name, 'a_edited_3.png')
            # 이름 자체가 '_edited' 뿐인 파일도 빈 이름이 되지 않는다
            bare = _plain_png(root / 'gen' / '_edited.png')
            self.assertEqual(Path(edited_copy_path(str(bare), str(root), excluded)).name, '_edited_edited.png')

    def test_validate_chosen_target_allows_drive_root_but_not_system_dirs(self):
        with tempfile.TemporaryDirectory() as tmp:
            anchor = Path(tmp).anchor   # 예: 'C:\\'
            self.assertIsNone(validate_chosen_target(os.path.join(anchor, 'edited.png')),
                              '드라이브 루트(USB 등)는 사용자가 고른 정당한 위치다')
            self.assertIsNone(validate_chosen_target(str(Path(tmp) / 'x.png')))
            self.assertEqual(validate_chosen_target(str(Path(tmp) / 'missing' / 'x.png')), '저장할 폴더가 없습니다')
        if os.name == 'nt':
            windows_dir = os.environ.get('SystemRoot') or 'C:\\Windows'
            self.assertEqual(validate_chosen_target(os.path.join(windows_dir, 'x.png')),
                             '이 폴더에는 저장할 수 없습니다')

    def test_saved_copy_registry_normalizes_and_is_bounded(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / 'Out' / 'a_edited.png'
            reg = SavedCopyRegistry(limit=2)
            reg.add(str(p))
            self.assertIn(str(p).replace('\\', '/'), reg)
            self.assertIn('file:///' + str(p).replace('\\', '/'), reg)
            if os.name == 'nt':
                self.assertIn(str(p).upper(), reg)
            self.assertNotIn(str(Path(tmp) / 'Out' / 'a.png'), reg)
            self.assertNotIn('', reg)
            self.assertNotIn(None, reg)
            reg.add(str(Path(tmp) / 'b.png'))
            reg.add(str(Path(tmp) / 'c.png'))
            self.assertEqual(len(reg), 2)
            self.assertNotIn(str(p), reg, '가장 오래된 항목이 밀려나야 한다')

    def test_saved_copy_registry_decodes_file_urls_but_keeps_raw_percent(self):
        # core.path_safety.strip_file_url 과 같은 규칙: file URL 만 퍼센트 디코드,
        # 원시 경로의 '%' 는 이름의 일부다. 예전엔 'file:///' 만 떼고 디코드하지 않았다.
        with tempfile.TemporaryDirectory() as tmp:
            spaced = Path(tmp) / 'Out' / 'a b.png'          # 'a%20b.png' 로 인코딩되는 이름
            literal = Path(tmp) / 'Out' / 'a%20b.png'       # 이름에 '%20' 이 글자 그대로 있는 형제
            reg = SavedCopyRegistry()
            reg.add(str(spaced))
            self.assertIn(spaced.as_uri(), reg)
            self.assertIn('FILE:///' + str(spaced).replace('\\', '/').replace(' ', '%20'), reg)
            self.assertNotIn(str(literal), reg, '원시 경로의 %20 을 디코드하면 다른 파일이 된다')
            reg.add(str(literal))
            self.assertIn(str(literal), reg)
            self.assertIn(literal.as_uri(), reg)            # ...%2520b.png → a%20b.png
            # 등록하지 않은 이름이 디코드로 등록된 이름과 섞이지 않는다
            other = SavedCopyRegistry()
            other.add(str(literal))
            self.assertNotIn(spaced.as_uri(), other)
            self.assertNotIn(str(spaced), other)


class SaveEditedImageTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self.snap_dir = str(self.root / 'image_cache' / 'editor_temp')

    def tearDown(self):
        self._tmp.cleanup()

    def test_overwrite_png_preserves_generation_metadata(self):
        source = _png_with_meta(self.root / 'generated_images' / 'gen.png')
        edited = _plain_png(self.root / 'image_cache' / 'editor_temp' / 'edited_1.png', color=(1, 2, 3))
        result = save_edited_image(str(edited), str(source), metadata_path=str(source),
                                   snapshot_dir=self.snap_dir)
        self.assertTrue(result['ok'])
        self.assertEqual(result['format'], 'PNG')
        with Image.open(source) as img:
            img.load()
            self.assertEqual(img.text.get('parameters'), PARAMS)
            self.assertEqual(img.text.get('workflow'), WORKFLOW)
            self.assertEqual(img.convert('RGB').getpixel((5, 5)), (1, 2, 3))
        meta = extract_from_file(source)
        self.assertEqual(meta.prompt, '1girl, 한글 태그, solo')
        self.assertEqual(meta.parameters.get('Seed'), 42)

    def test_save_as_jpeg_writes_real_jpeg_with_parameters(self):
        source = _png_with_meta(self.root / 'generated_images' / 'gen.png')
        edited = _plain_png(self.root / 'image_cache' / 'editor_temp' / 'e.png', color=(0, 128, 255),
                            mode='RGBA')   # 알파 128 — JPEG 는 흰 바탕에 합성해야 한다
        target = self.root / 'out' / '저장본.jpg'
        target.parent.mkdir()
        result = save_edited_image(str(edited), str(target), metadata_path=str(source),
                                   snapshot_dir=self.snap_dir)
        self.assertEqual(result['format'], 'JPEG')
        self.assertEqual(target.read_bytes()[:2], b'\xff\xd8', 'JPEG 를 골랐는데 PNG 바이트가 쓰였다')
        with Image.open(target) as img:
            self.assertEqual(img.format, 'JPEG')
            r, g, b = img.getpixel((5, 5))
            # 반투명 파랑이 흰 바탕과 섞였다 (검게 뭉개지지 않는다)
            self.assertGreater(r, 100)
            self.assertGreater(g, 150)
        meta = extract_from_file(target)
        self.assertEqual(meta.raw_parameters, PARAMS)

    def test_save_as_webp_writes_real_webp_with_parameters(self):
        source = _png_with_meta(self.root / 'generated_images' / 'gen.png')
        edited = _plain_png(self.root / 'image_cache' / 'editor_temp' / 'e.png')
        target = self.root / 'out.webp'
        save_edited_image(str(edited), str(target), metadata_path=str(source), snapshot_dir=self.snap_dir)
        data = target.read_bytes()
        self.assertEqual(data[:4], b'RIFF')
        self.assertEqual(data[8:12], b'WEBP')
        self.assertEqual(extract_from_file(target).raw_parameters, PARAMS)

    def test_jpeg_source_parameters_move_into_png_text_chunk(self):
        jpeg = self.root / 'src.jpg'
        exif = Image.Exif()
        exif.get_ifd(0x8769)[0x9286] = b'UNICODE\x00' + PARAMS.encode('utf-16-be')
        Image.new('RGB', (64, 48), (5, 5, 5)).save(jpeg, exif=exif.tobytes())
        edited = _plain_png(self.root / 'image_cache' / 'editor_temp' / 'e.png')
        target = self.root / 'from_jpeg.png'
        save_edited_image(str(edited), str(target), metadata_path=str(jpeg), snapshot_dir=self.snap_dir)
        with Image.open(target) as img:
            img.load()
            self.assertEqual(img.text.get('parameters'), PARAMS)

    def test_unmerged_draw_layer_is_composited(self):
        edited = _plain_png(self.root / 'image_cache' / 'editor_temp' / 'e.png', color=(0, 0, 0))
        target = self.root / 'drawn.png'
        save_edited_image(str(edited), str(target), overlay_base64=_overlay_b64(), overlay_opacity=1.0,
                          snapshot_dir=self.snap_dir)
        with Image.open(target) as img:
            rgb = img.convert('RGB')
            self.assertEqual(rgb.getpixel((2, 2)), (255, 0, 0), '드로잉 레이어가 저장본에 없다')
            self.assertEqual(rgb.getpixel((30, 30)), (0, 0, 0))

    def test_overlay_opacity_is_applied(self):
        edited = _plain_png(self.root / 'image_cache' / 'editor_temp' / 'e.png', color=(0, 0, 0))
        target = self.root / 'half.png'
        save_edited_image(str(edited), str(target), overlay_base64=_overlay_b64(), overlay_opacity=0.5,
                          snapshot_dir=self.snap_dir)
        with Image.open(target) as img:
            r, _g, _b = img.convert('RGB').getpixel((2, 2))
            self.assertTrue(120 <= r <= 135, r)

    def test_transparent_overlay_counts_as_no_overlay(self):
        empty = Image.new('RGBA', (64, 48), (0, 0, 0, 0))
        buf = io.BytesIO()
        empty.save(buf, 'PNG')
        self.assertIsNone(editor_save.decode_overlay_base64(base64.b64encode(buf.getvalue()).decode()))

    def test_unchanged_source_is_not_reencoded(self):
        source = _png_with_meta(self.root / 'gen.png')
        before = source.read_bytes()
        result = save_edited_image(str(source), str(source), metadata_path=str(source),
                                   protect_paths=[str(source)], snapshot_dir=self.snap_dir)
        self.assertTrue(result['unchanged'])
        self.assertNotIn('snapshot_path', result)
        self.assertEqual(source.read_bytes(), before)

    def test_overwriting_a_history_path_snapshots_previous_content(self):
        source = _png_with_meta(self.root / 'gen.png', color=(9, 9, 9))
        original_bytes = source.read_bytes()
        vue_path = str(source).replace('\\', '/')   # 프론트가 들고 있는 표기
        result = save_edited_image(str(source), str(source), metadata_path=str(source),
                                   overlay_base64=_overlay_b64(), protect_paths=[vue_path, 'other.png'],
                                   snapshot_dir=self.snap_dir)
        self.assertEqual(result['replaced_paths'], [vue_path])
        snap = Path(result['snapshot_path'])
        self.assertEqual(snap.parent, Path(self.snap_dir))
        self.assertEqual(snap.read_bytes(), original_bytes, '사본이 덮어쓰기 전 내용이 아니다')
        with Image.open(source) as img:
            self.assertEqual(img.convert('RGB').getpixel((2, 2)), (255, 0, 0))
            img.load()
            self.assertEqual(img.text.get('parameters'), PARAMS)

    def test_protect_paths_match_decoded_file_urls_not_raw_percent_names(self):
        # 히스토리가 file URL 을 들고 있어도 같은 파일로 알아본다(퍼센트 디코드). 원시 경로의
        # '%20' 은 이름의 일부라 'a b.png' 와 다른 파일이다 — 스냅숏 대상으로 오인하지 않는다.
        source = _png_with_meta(self.root / 'a b.png', color=(9, 9, 9))
        as_url = source.as_uri()                                     # ...a%20b.png
        raw_percent = str(self.root / 'a%20b.png')                   # 글자 그대로의 다른 이름
        result = save_edited_image(str(source), str(source), metadata_path=str(source),
                                   overlay_base64=_overlay_b64(),
                                   protect_paths=[raw_percent, as_url, 'other.png'],
                                   snapshot_dir=self.snap_dir)
        self.assertEqual(result['replaced_paths'], [as_url])
        self.assertTrue(Path(result['snapshot_path']).is_file())

    def test_unchanged_source_is_reported_without_writing_a_copy(self):
        source = _png_with_meta(self.root / 'gen' / 'gen.png')
        before = source.read_bytes()
        copy = self.root / 'gen' / 'gen_edited.png'
        result = save_edited_image(str(source), str(copy), metadata_path=str(source),
                                   unchanged_source=str(source), snapshot_dir=self.snap_dir)
        self.assertTrue(result['unchanged'])
        self.assertEqual(Path(result['path']).resolve(), source.resolve())
        self.assertFalse(copy.exists(), '아무것도 안 바꿨는데 똑같은 사본을 만들었다')
        self.assertEqual(source.read_bytes(), before)
        # 드로잉이 있으면 변경이다 — 사본에 합성한다
        result = save_edited_image(str(source), str(copy), metadata_path=str(source),
                                   overlay_base64=_overlay_b64(), unchanged_source=str(source),
                                   snapshot_dir=self.snap_dir)
        self.assertNotIn('unchanged', result)
        self.assertTrue(copy.exists())
        self.assertEqual(source.read_bytes(), before)

    def test_alpha_to_png_keeps_transparency_for_auto_jpeg_target(self):
        edited = self.root / 'image_cache' / 'editor_temp' / 'cut.png'
        edited.parent.mkdir(parents=True)
        cut = Image.new('RGBA', (64, 48), (10, 200, 10, 255))
        cut.paste((0, 0, 0, 0), (0, 0, 32, 48))
        cut.save(edited)
        jpeg_target = self.root / 'photo_edited.jpg'
        result = save_edited_image(str(edited), str(jpeg_target), alpha_to_png=True, snapshot_dir=self.snap_dir)
        self.assertTrue(result['alpha_png'])
        self.assertEqual(result['format'], 'PNG')
        self.assertEqual(Path(result['path']).name, 'photo_edited.png')
        self.assertFalse(jpeg_target.exists())
        with Image.open(result['path']) as img:
            self.assertEqual(img.getpixel((5, 5))[3], 0)
        # 사용자가 JPEG 를 고른 경우(alpha_to_png=False)는 흰 바탕 합성 — 기존 규칙 그대로
        chosen = self.root / 'chosen.jpg'
        result = save_edited_image(str(edited), str(chosen), snapshot_dir=self.snap_dir)
        self.assertEqual(result['format'], 'JPEG')
        self.assertNotIn('alpha_png', result)
        self.assertEqual(chosen.read_bytes()[:2], b'\xff\xd8')

    def test_alpha_safe_target_rules(self):
        opaque = np.zeros((4, 4, 4), np.uint8)
        opaque[..., 3] = 255
        clear = opaque.copy()
        clear[0, 0, 3] = 0
        self.assertFalse(has_transparency(opaque))
        self.assertFalse(has_transparency(np.zeros((4, 4, 3), np.uint8)))
        self.assertTrue(has_transparency(clear))
        jpg = str(self.root / 'x.jpg')
        self.assertEqual(alpha_safe_target(jpg, opaque), jpg)
        self.assertEqual(Path(alpha_safe_target(jpg, clear)).name, 'x.png')
        webp = str(self.root / 'x.webp')
        self.assertEqual(alpha_safe_target(webp, clear), webp, 'WebP 는 알파를 담을 수 있다')
        _plain_png(self.root / 'x.png')
        self.assertEqual(Path(alpha_safe_target(jpg, clear)).name, 'x_2.png', '기존 PNG 를 덮어쓰면 안 된다')

    def test_snapshot_prefix(self):
        source = _png_with_meta(self.root / 'r.png')
        snap = editor_save.snapshot_file(str(source), self.snap_dir, prefix='recovered')
        self.assertTrue(Path(snap).name.startswith('recovered_'))
        self.assertEqual(Path(snap).read_bytes(), source.read_bytes())

    def test_snapshot_mtime_is_fresh_so_prune_keeps_it(self):
        source = _png_with_meta(self.root / 'old.png')
        os.utime(source, (1_000_000, 1_000_000))   # 아주 오래된 원본
        snap = editor_save.snapshot_file(str(source), self.snap_dir)
        self.assertGreater(os.path.getmtime(snap), 1_000_000 + 86400)

    def test_failed_replace_keeps_original_and_cleans_temp(self):
        source = _png_with_meta(self.root / 'keep.png')
        before = source.read_bytes()
        edited = _plain_png(self.root / 'image_cache' / 'editor_temp' / 'e.png')
        with mock.patch('core.editor_save.os.replace', side_effect=PermissionError(13, 'locked')):
            with self.assertRaises(EditorSaveError):
                save_edited_image(str(edited), str(source), metadata_path=str(source),
                                  snapshot_dir=self.snap_dir)
        self.assertEqual(source.read_bytes(), before)
        leftovers = [p for p in source.parent.iterdir() if p.name.endswith('.tmp')]
        self.assertEqual(leftovers, [])

    def test_korean_paths_round_trip(self):
        source = _png_with_meta(self.root / '생성 결과' / '캐릭터.png')
        edited = _plain_png(self.root / '편집' / '임시.png', color=(7, 8, 9))
        result = save_edited_image(str(edited), str(source), metadata_path=str(source),
                                   snapshot_dir=self.snap_dir)
        self.assertTrue(result['ok'])
        self.assertEqual(read_source_metadata(str(source)).parameters, PARAMS)

    def test_exif_orientation_is_applied_and_reset(self):
        src = self.root / 'rot.jpg'
        exif = Image.Exif()
        exif[0x0112] = 6   # 90° 회전해서 보여 줘야 하는 사진
        Image.new('RGB', (40, 20), (50, 60, 70)).save(src, exif=exif.tobytes())
        target = self.root / 'rot_out.jpg'
        save_edited_image(str(src), str(target), metadata_path=str(src), overlay_base64=_overlay_b64(size=(20, 40)),
                          snapshot_dir=self.snap_dir)
        with Image.open(target) as img:
            self.assertEqual(img.size, (20, 40), '화면(브라우저)처럼 세운 픽셀로 저장해야 한다')
            self.assertEqual(img.getexif().get(0x0112), 1)

    def test_unsupported_target_format_is_rejected(self):
        edited = _plain_png(self.root / 'e.png')
        with self.assertRaises(EditorSaveError):
            save_edited_image(str(edited), str(self.root / 'x.bmp'), snapshot_dir=self.snap_dir)
        with self.assertRaises(EditorSaveError):
            save_edited_image(str(self.root / 'missing.png'), str(self.root / 'y.png'), snapshot_dir=self.snap_dir)

    def test_sixteen_bit_grayscale_is_scaled_not_clipped(self):
        arr = (np.arange(64 * 48, dtype=np.uint32).reshape(48, 64) * 20).astype(np.uint16)
        src = self.root / 'g16.png'
        Image.fromarray(arr).save(src)
        loaded = editor_save.load_image_array(str(src))
        self.assertEqual(loaded.shape, (48, 64, 3))
        self.assertLess(int(loaded[0, 0, 0]), 5)
        self.assertLess(int(loaded[-1, -1, 0]), 255)

    def _keyed_sources(self):
        """tRNS 투명색 PNG — RGB(마젠타 키)·흑백(검정 키)·16비트 흑백. 왼쪽 절반이 투명하다."""
        rgb = Image.new('RGB', (64, 48), (10, 200, 10))
        rgb.paste((255, 0, 255), (0, 0, 32, 48))
        gray = Image.new('L', (64, 48), 180)
        gray.paste(0, (0, 0, 32, 48))
        deep = Image.new('I;16', (64, 48), 40000)
        deep.paste(0, (0, 0, 32, 48))
        out = {}
        for name, image, key in (('rgb', rgb, (255, 0, 255)), ('gray', gray, 0), ('gray16', deep, 0)):
            path = self.root / f'{name}_trns.png'
            image.save(path, transparency=key)
            out[name] = path
        return out

    def test_transparency_key_png_is_loaded_with_alpha(self):
        for name, path in self._keyed_sources().items():
            with self.subTest(name=name):
                arr = editor_save.load_image_array(str(path))
                self.assertEqual(arr.shape, (48, 64, 4))
                self.assertEqual(int(arr[5, 5, 3]), 0, '투명색이 불투명해졌다')
                self.assertEqual(int(arr[5, 50, 3]), 255)

    def test_transparency_key_survives_save_as_png_and_webp(self):
        # 예전엔 RGB·흑백+tRNS 를 저장하면 투명하던 곳에 키 색(마젠타·검정)이 드러났다
        for name, path in self._keyed_sources().items():
            for suffix in ('.png', '.webp'):
                with self.subTest(name=name, suffix=suffix):
                    target = self.root / f'{name}_out{suffix}'
                    save_edited_image(str(path), str(target), snapshot_dir=self.snap_dir)
                    with Image.open(target) as img:
                        rgba = img.convert('RGBA')
                        self.assertEqual(rgba.getpixel((5, 5))[3], 0)
                        self.assertEqual(rgba.getpixel((50, 5))[3], 255)

    def test_transparency_key_switches_an_auto_jpeg_target_to_png(self):
        path = self._keyed_sources()['rgb']
        result = save_edited_image(str(path), str(self.root / 'keyed.jpg'), alpha_to_png=True,
                                   snapshot_dir=self.snap_dir)
        self.assertTrue(result['alpha_png'])
        self.assertEqual(result['format'], 'PNG')

    # ── Pillow convert('RGBA') 가 틀리는 tRNS 키 — 에디터(cv2) 화면과 같은 투명도여야 한다 ──

    @staticmethod
    def _raw_png(path: Path, width: int, depth: int, color_type: int, row: bytes, trns: bytes) -> Path:
        """손으로 짠 한 줄짜리 PNG(Pillow 가 직접 못 쓰는 저비트 흑백·16비트 RGB + tRNS)."""
        import struct
        import zlib
        from core.png_chunks import PNG_SIGNATURE, make_chunk
        ihdr = struct.pack('>IIBBBBB', width, 1, depth, color_type, 0, 0, 0)
        path.write_bytes(PNG_SIGNATURE + make_chunk(b'IHDR', ihdr) + make_chunk(b'tRNS', trns)
                         + make_chunk(b'IDAT', zlib.compress(b'\x00' + row)) + make_chunk(b'IEND', b''))
        return path

    def _low_depth_gray(self, depth: int, values: list, key: int) -> Path:
        import struct
        bits = ''.join(format(v, f'0{depth}b') for v in values)
        bits += '0' * (-len(bits) % 8)
        row = int(bits, 2).to_bytes(len(bits) // 8, 'big')
        return self._raw_png(self.root / f'gray{depth}_key{key}.png', len(values), depth, 0, row,
                             struct.pack('>H', key))

    def _assert_alpha_like_editor(self, path: Path, want_alpha: list):
        from core.editor_preview import imread_unchanged, to_display_uint8
        arr = editor_save.load_image_array(str(path))
        self.assertEqual(arr.shape[-1], 4)
        self.assertEqual(arr[0, :, 3].tolist(), want_alpha)
        editor = to_display_uint8(imread_unchanged(str(path)))
        self.assertEqual(editor[0, :, 3].tolist(), want_alpha, '에디터 화면 기준이 바뀌었다')

    def test_low_bit_gray_key_is_scaled_like_the_editor(self):
        # 2·4비트 흑백 + 0 이 아닌 키: Pillow 는 늘리지 않은 키(3·15)를 0/85/170/255 픽셀과 비교해 투명한 곳이 없었다
        self._assert_alpha_like_editor(self._low_depth_gray(2, [0, 1, 2, 3], 3), [255, 255, 255, 0])
        self._assert_alpha_like_editor(self._low_depth_gray(4, [0, 5, 15, 7], 15), [255, 255, 0, 255])

    def test_sixteen_bit_rgb_key_is_compared_at_full_depth(self):
        # 16비트 RGB + 키 (0,0,0): Pillow 는 8비트로 줄인 뒤 비교해 (200,200,200) 까지 투명해졌다
        import struct
        row = b''.join(struct.pack('>HHH', *p) for p in ((0, 0, 0), (200, 200, 200), (65535, 0, 0)))
        path = self._raw_png(self.root / 'rgb16_key.png', 3, 16, 2, row, struct.pack('>HHH', 0, 0, 0))
        self._assert_alpha_like_editor(path, [0, 255, 255])
        arr = editor_save.load_image_array(str(path))
        self.assertEqual(arr[0, 2, :3].tolist(), [255, 0, 0], 'RGB 채널 순서')

    def test_save_as_keeps_the_editor_transparency_for_keyed_pngs(self):
        import struct
        row = b''.join(struct.pack('>HHH', *p) for p in ((0, 0, 0), (200, 200, 200), (65535, 0, 0)))
        sources = {
            'gray2': (self._low_depth_gray(2, [0, 1, 2, 3], 3), [255, 255, 255, 0]),
            'rgb16': (self._raw_png(self.root / 'rgb16_key.png', 3, 16, 2, row, struct.pack('>HHH', 0, 0, 0)),
                      [0, 255, 255]),
        }
        for name, (src, want) in sources.items():
            with self.subTest(name=name):
                target = self.root / f'{name}_saved.png'
                save_edited_image(str(src), str(target), snapshot_dir=self.snap_dir)
                with Image.open(target) as img:
                    self.assertEqual([img.convert('RGBA').getpixel((x, 0))[3] for x in range(len(want))], want)

    def test_xmp_only_orientation_is_not_applied(self):
        # 에디터 배열(cv2·진짜 EXIF 만)은 XMP tiff:Orientation 을 무시한다 — 저장본도 돌리면 안 된다
        xmp =('<x:xmpmeta xmlns:x="adobe:ns:meta/"><rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">'
               '<rdf:Description xmlns:tiff="http://ns.adobe.com/tiff/1.0/" tiff:Orientation="6"/></rdf:RDF></x:xmpmeta>')
        jpg = self.root / 'xmp6.jpg'
        Image.new('RGB', (40, 20), (50, 60, 70)).save(jpg, xmp=xmp.encode('utf-8'))
        png = self.root / 'xmp6.png'
        info = PngImagePlugin.PngInfo()
        info.add_itxt('XML:com.adobe.xmp', xmp)
        Image.new('RGB', (40, 20), (50, 60, 70)).save(png, pnginfo=info)
        for path in (jpg, png):
            with self.subTest(path=path.name):
                self.assertEqual(editor_save.load_image_array(str(path)).shape[:2], (20, 40))

    @staticmethod
    def _quadrants() -> np.ndarray:
        """12x20 RGB — 왼쪽 빨강, 오른쪽 위 초록·아래 파랑(뒤집기·회전이 모두 구별된다)."""
        base = np.zeros((12, 20, 3), np.uint8)
        base[:, :10] = (255, 0, 0)
        base[:6, 10:] = (0, 255, 0)
        base[6:, 10:] = (0, 0, 255)
        return base

    def test_tiff_orientation_is_applied_exactly_once(self):
        # Pillow(TiffImageFile.load)·OpenCV(libtiff, IMREAD_UNCHANGED 여도) TIFF 디코더는 방향을 이미
        # 적용한다. 예전엔 그 위에 또 돌려 3·2·4… 는 저장된 그대로, 6·8 은 180° 뒤집혀 누웠다(Codex S5 #1).
        from core.cv_io import apply_exif_orientation
        from core.editor_preview import imread_unchanged
        base = self._quadrants()
        for compression in (None, 'tiff_lzw'):
            for orientation in range(1, 9):
                with self.subTest(compression=compression, orientation=orientation):
                    path = self.root / f'방향_{orientation}_{compression}.tiff'
                    exif = Image.Exif()
                    exif[0x0112] = orientation
                    extra = {'compression': compression} if compression else {}
                    Image.fromarray(base).save(path, exif=exif, **extra)
                    want = apply_exif_orientation(base, orientation)   # 저장된 픽셀에 방향을 한 번
                    np.testing.assert_array_equal(editor_save.load_image_array(str(path)), want)
                    np.testing.assert_array_equal(imread_unchanged(str(path))[:, :, ::-1], want)

    def test_tiff_save_as_png_is_upright(self):
        src = self.root / '한글 폴더' / '회전6.tiff'
        src.parent.mkdir()
        exif = Image.Exif()
        exif[0x0112] = 6
        Image.new('RGB', (40, 20), (50, 60, 70)).save(src, exif=exif)
        target = self.root / 'tiff_out.png'
        save_edited_image(str(src), str(target), snapshot_dir=self.snap_dir)
        with Image.open(target) as img:
            self.assertEqual(img.size, (20, 40), '화면처럼 세운 픽셀로 저장해야 한다')

    @staticmethod
    def _editor_rgb(path: Path) -> np.ndarray:
        """에디터 화면의 픽셀(cv2 디코드 + 프리뷰 8비트 변환)을 RGB/RGBA 로."""
        from core.editor_preview import imread_unchanged, to_display_uint8
        img = to_display_uint8(imread_unchanged(str(path)))
        if img.ndim == 2:
            return np.dstack([img, img, img])
        return img[:, :, [2, 1, 0, 3]] if img.shape[2] == 4 else img[:, :, ::-1]

    def test_rotated_uncompressed_tiff_in_pillow_mmap_modes_matches_the_editor(self):
        # 경로로 연 방향 5~8 무압축 한 스트립 TIFF(L·RGBA·I;16 — Pillow mmap 모드)는 mmap 지름길이 세운
        # 크기로 버퍼를 매핑해 뒤섞인 채 (12,20) 으로 읽혔다. 에디터(cv2)는 (20,12) — '다른 이름으로 저장'·
        # 배치 변환이 쓰레기를 디스크에 썼다(Codex S5 #1-a). RGB 만 보던 테스트는 이걸 못 잡았다.
        from core.cv_io import apply_exif_orientation
        quads = self._quadrants()
        gray = quads[:, :, 0] // 2 + quads[:, :, 1] // 3 + quads[:, :, 2] // 5   # 세 구역 127·85·51
        # 불투명 알파 — OpenCV 는 연관 안 된(unassociated) 알파 TIFF 의 색을 알파로 곱해(premultiply) 읽어
        # 반투명 픽셀은 에디터와 Pillow 가 원래 다르다. 여기선 방향만 본다.
        alpha = np.full((12, 20), 255, np.uint8)
        sources = {
            'L': (Image.fromarray(gray), np.dstack([gray, gray, gray]), 0),
            'RGBA': (Image.fromarray(np.dstack([quads, alpha])), np.dstack([quads, alpha]), 0),
            # 16비트 → 8비트: 저장 경로는 v/65535*255 버림, 에디터 프리뷰는 반올림 — 1 차이까지
            'I;16': (Image.fromarray(gray.astype(np.uint16) * 257), np.dstack([gray, gray, gray]), 1),
        }
        for mode, (image, stored, tolerance) in sources.items():
            self.assertEqual(image.mode, mode)
            for orientation in range(1, 9):
                with self.subTest(mode=mode, orientation=orientation):
                    path = self.root / f'무압축_{mode.replace(";", "")}_{orientation}.tiff'
                    exif = Image.Exif()
                    exif[0x0112] = orientation
                    image.save(path, exif=exif)
                    want = apply_exif_orientation(stored, orientation).astype(int)
                    got = editor_save.load_image_array(str(path))
                    self.assertEqual(got.shape, want.shape)
                    self.assertLessEqual(int(np.abs(got.astype(int) - want).max()), tolerance)
                    editor = self._editor_rgb(path)
                    self.assertEqual(editor.shape, want.shape, '에디터 화면 기준이 바뀌었다')
                    self.assertLessEqual(int(np.abs(editor.astype(int) - want).max()), tolerance)
        # '다른 이름으로 저장'(PNG)도 세운 그대로
        src = self.root / '무압축_L_6.tiff'
        target = self.root / 'l6_out.png'
        save_edited_image(str(src), str(target), snapshot_dir=self.snap_dir)
        with Image.open(target) as img:
            np.testing.assert_array_equal(np.asarray(img.convert('RGB')),
                                          apply_exif_orientation(sources['L'][1], 6))

    def test_xmp_only_tiff_orientation_is_not_applied(self):
        # Pillow TIFF load() 는 XMP 에만 적힌 방향으로도 돌았다(load_end 의 exif_transpose → getexif 가 XMP 를
        # 읽는다). 에디터(cv2)·브라우저는 무시한다 — 저장본도 저장된 그대로(Codex S5 #1-b)
        from PIL import TiffImagePlugin
        xmp = ('<x:xmpmeta xmlns:x="adobe:ns:meta/"><rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">'
               '<rdf:Description xmlns:tiff="http://ns.adobe.com/tiff/1.0/" tiff:Orientation="6"/></rdf:RDF></x:xmpmeta>')
        base = self._quadrants()
        for compression in (None, 'tiff_lzw'):
            with self.subTest(compression=compression):
                path = self.root / f'xmp방향6_{compression}.tiff'
                info = TiffImagePlugin.ImageFileDirectory_v2()
                info[700] = xmp.encode('utf-8')
                info.tagtype[700] = 1
                extra = {'compression': compression} if compression else {}
                Image.fromarray(base).save(path, tiffinfo=info, **extra)
                with Image.open(path) as probe:
                    self.assertEqual(probe.getexif().get(0x0112), 6, '전제: Pillow 가 XMP 방향을 읽는다')
                np.testing.assert_array_equal(editor_save.load_image_array(str(path)), base)
                np.testing.assert_array_equal(self._editor_rgb(path), base)
                target = self.root / f'xmp_out_{compression}.png'
                save_edited_image(str(path), str(target), snapshot_dir=self.snap_dir)
                with Image.open(target) as img:
                    self.assertEqual(img.size, (20, 12))


class AutosaveSnapshotTests(unittest.TestCase):
    """크래시 복구본(write_autosave_snapshot) — 예전 자동저장은 확정 이미지만 복사해
    병합 안 한 드로잉이 복구본에서 빠졌다('자동저장 N분 전' 표시는 떴는데도)."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.root = Path(self._tmp.name)
        self.src = _png_with_meta(self.root / '한글 폴더' / '원본.png', color=(40, 40, 40))
        self.dst = self.root / 'autosave' / '_autosave_session.png'
        self.dst.parent.mkdir()

    def _pixel(self, x: int, y: int) -> tuple:
        with Image.open(self.dst) as img:
            return img.convert('RGB').getpixel((x, y))

    def test_without_overlay_is_a_byte_identical_copy(self):
        self.assertFalse(editor_save.write_autosave_snapshot(str(self.src), str(self.dst)))
        self.assertEqual(self.dst.read_bytes(), self.src.read_bytes())
        # 빈 문자열(프론트가 레이어가 없을 때 보내는 값)도 같다
        self.dst.unlink()
        self.assertFalse(editor_save.write_autosave_snapshot(str(self.src), str(self.dst), '', 1.0))
        self.assertEqual(self.dst.read_bytes(), self.src.read_bytes())

    def test_fully_transparent_overlay_falls_back_to_a_copy(self):
        blank = _overlay_b64(color=(0, 0, 0, 0))
        self.assertFalse(editor_save.write_autosave_snapshot(str(self.src), str(self.dst), blank, 1.0))
        self.assertEqual(self.dst.read_bytes(), self.src.read_bytes())

    def test_unmerged_drawing_is_composited(self):
        drawn = editor_save.write_autosave_snapshot(
            str(self.src), str(self.dst), _overlay_b64(box=(0, 0, 8, 8)), 1.0)
        self.assertTrue(drawn)
        self.assertEqual(self._pixel(2, 2), (255, 0, 0), '그린 획이 복구본에 없다')
        self.assertEqual(self._pixel(40, 30), (40, 40, 40), '그리지 않은 곳이 바뀌었다')
        # 원본 메타데이터(생성 정보)는 복구본에도 남는다
        self.assertEqual(read_source_metadata(str(self.dst)).parameters, PARAMS)
        # 원본은 그대로다
        with Image.open(self.src) as img:
            self.assertEqual(img.convert('RGB').getpixel((2, 2)), (40, 40, 40))

    def test_opacity_is_applied_like_the_manual_save(self):
        editor_save.write_autosave_snapshot(str(self.src), str(self.dst), _overlay_b64(box=(0, 0, 8, 8)), 0.5)
        r, g, b = self._pixel(2, 2)
        self.assertAlmostEqual(r, (255 + 40) / 2, delta=2)
        self.assertAlmostEqual(g, 40 / 2, delta=2)
        # 범위 밖 불투명도는 수동 저장처럼 0~1 로 자른다
        editor_save.write_autosave_snapshot(str(self.src), str(self.dst), _overlay_b64(box=(0, 0, 8, 8)), 7.0)
        self.assertEqual(self._pixel(2, 2), (255, 0, 0))

    def test_broken_overlay_is_reported(self):
        with self.assertRaises(EditorSaveError):
            editor_save.write_autosave_snapshot(str(self.src), str(self.dst), 'data:image/png;base64,bm90IGEgcG5n', 1.0)


if __name__ == '__main__':
    unittest.main()
