"""Vue 일괄 처리(리사이즈/포맷) — 원본 비파괴 + 메타데이터 보존 + 단일 워커 (#13, #72).

예전 Vue 배치는 generated_images/<원본이름> 에 충돌 검사 없이 cv2 로 다시 써서 원본과
생성 파라미터를 날렸고, 재클릭하면 워커 두 개가 같은 경로에 동시에 썼으며, 끝날 때마다
숨은 탭의 QMessageBox 모달이 떴다.
"""
from __future__ import annotations

import json
import os
import tempfile
import unittest

from PIL import Image, PngImagePlugin

from core.batch_image_ops import (
    BATCH_SUBDIR,
    BatchJob,
    BatchJobError,
    batch_item_result,
    batch_output_dir,
    parse_batch_job,
    planned_output_name,
    process_batch_file,
)
from core.batch_job_state import JobProgress, completion_notice
from core.image_metadata import extract_from_file

PARAMS = 'a cat, masterpiece\nNegative prompt: lowres\nSteps: 20, Sampler: Euler a, CFG scale: 7, Seed: 42, Size: 64x48'


def _write_png(path: str, size=(64, 48), mode='RGB', params=PARAMS) -> bytes:
    info = PngImagePlugin.PngInfo()
    if params:
        info.add_text('parameters', params)
    color = (200, 10, 10, 128) if mode == 'RGBA' else (200, 10, 10)
    Image.new(mode, size, color).save(path, format='PNG', pnginfo=info)
    with open(path, 'rb') as handle:
        return handle.read()


class ParseTests(unittest.TestCase):
    def test_resize_and_format_payloads(self):
        self.assertEqual(parse_batch_job({'operation': 'resize', 'settings': {'width': '512', 'height': 768}}),
                         BatchJob('resize', width=512, height=768))
        self.assertEqual(parse_batch_job({'operation': 'format', 'settings': {'format': 'JPEG'}}),
                         BatchJob('format', target_format='JPEG'))
        self.assertEqual(parse_batch_job({'operation': 'format', 'settings': {'format': 'webp'}}).target_format, 'WEBP')

    def test_bad_payloads_raise_user_errors(self):
        for payload in (
            {'operation': 'resize', 'settings': {'width': 'abc', 'height': 10}},
            {'operation': 'resize', 'settings': {'width': 0, 'height': 10}},
            {'operation': 'resize', 'settings': {'width': 99999, 'height': 10}},
            {'operation': 'format', 'settings': {'format': 'GIF'}},
            {'operation': 'watermark'},
        ):
            with self.subTest(payload=payload), self.assertRaises(BatchJobError):
                parse_batch_job(payload)

    def test_output_goes_to_a_batch_subfolder_with_descriptive_names(self):
        self.assertEqual(batch_output_dir('C:/out'), os.path.join('C:/out', BATCH_SUBDIR))
        self.assertEqual(planned_output_name('C:/g/gen_1.png', BatchJob('resize', width=512, height=512)),
                         'gen_1_512x512.png')
        self.assertEqual(planned_output_name('C:/g/photo.bmp', BatchJob('resize', width=10, height=10)),
                         'photo_10x10.png')   # BMP 는 쓸 수 없는 포맷 → PNG
        self.assertEqual(planned_output_name('C:/g/gen_1.png', BatchJob('format', target_format='JPEG')),
                         'gen_1.jpg')


class ProcessTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = self._tmp.name
        # 생성 이미지는 출력 폴더 바로 아래에 있다 — 예전엔 여기로 같은 이름을 다시 썼다
        self.source = os.path.join(self.root, 'generated_1.png')
        self.original_bytes = _write_png(self.source)
        self.out_dir = batch_output_dir(self.root)

    def tearDown(self):
        self._tmp.cleanup()

    def test_png_to_png_never_touches_the_original_and_keeps_parameters(self):
        written = process_batch_file(self.source, self.out_dir, BatchJob('format', target_format='PNG'))
        with open(self.source, 'rb') as handle:
            self.assertEqual(handle.read(), self.original_bytes)
        self.assertNotEqual(os.path.normcase(written), os.path.normcase(self.source))
        self.assertEqual(os.path.dirname(written), self.out_dir)
        self.assertEqual(extract_from_file(written).prompt, 'a cat, masterpiece')

    def test_resize_keeps_extension_size_metadata_and_alpha(self):
        rgba = os.path.join(self.root, '투명.png')
        _write_png(rgba, mode='RGBA')
        written = process_batch_file(rgba, self.out_dir, BatchJob('resize', width=32, height=24))
        self.assertTrue(written.endswith('투명_32x24.png'))
        with Image.open(written) as image:
            self.assertEqual(image.size, (32, 24))
            self.assertEqual(image.mode, 'RGBA')
            self.assertIn('parameters', image.info)

    def test_transparency_key_png_keeps_its_transparent_area(self):
        # RGB+tRNS(키 색 투명) — 예전엔 불투명 RGB 로 바뀌어 키 색(마젠타)이 드러났다
        keyed = os.path.join(self.root, 'keyed.png')
        image = Image.new('RGB', (64, 48), (10, 200, 10))
        image.paste((255, 0, 255), (0, 0, 32, 48))
        image.save(keyed, transparency=(255, 0, 255))
        for name, job in (('resize', BatchJob('resize', width=32, height=24)),
                          ('webp', BatchJob('format', target_format='WEBP'))):
            with self.subTest(job=name):
                written = process_batch_file(keyed, self.out_dir, job)
                with Image.open(written) as out:
                    rgba = out.convert('RGBA')
                    self.assertEqual(rgba.getpixel((2, 2))[3], 0)
                    self.assertEqual(rgba.getpixel((rgba.width - 2, 2))[3], 255)

    def test_repeated_runs_get_new_names(self):
        job = BatchJob('resize', width=16, height=16)
        first = process_batch_file(self.source, self.out_dir, job)
        second = process_batch_file(self.source, self.out_dir, job)
        self.assertNotEqual(first, second)
        self.assertTrue(second.endswith('generated_1_16x16_2.png'))

    def test_source_already_in_the_output_folder_is_not_overwritten(self):
        os.makedirs(self.out_dir)
        inside = os.path.join(self.out_dir, 'x.png')
        data = _write_png(inside)
        written = process_batch_file(inside, self.out_dir, BatchJob('format', target_format='PNG'))
        self.assertTrue(written.endswith('x_2.png'))
        with open(inside, 'rb') as handle:
            self.assertEqual(handle.read(), data)

    def test_jpeg_output_moves_parameters_to_exif(self):
        written = process_batch_file(self.source, self.out_dir, BatchJob('format', target_format='JPEG'))
        self.assertTrue(written.endswith('.jpg'))
        self.assertEqual(extract_from_file(written).prompt, 'a cat, masterpiece')

    def test_rotated_tiff_is_converted_upright_once(self):
        # Pillow TIFF 디코더가 방향을 이미 적용한다 — 예전엔 또 돌려 6 이 40x20 으로 누웠고,
        # 3 은 저장된(뒤집힌) 그대로 나왔다(Codex S5 #1)
        red, blue = (255, 0, 0), (0, 0, 255)
        # 6(시계 90°): 왼쪽 빨강이 위로 / 3(180°): 오른쪽 파랑이 왼쪽으로
        for orientation, size, top_left, bottom_right in ((6, (20, 40), red, blue), (3, (40, 20), blue, red)):
            with self.subTest(orientation=orientation):
                source = os.path.join(self.root, f'한글_방향{orientation}.tiff')
                stored = Image.new('RGB', (40, 20), blue)
                stored.paste(red, (0, 0, 20, 20))   # 저장된 픽셀: 왼쪽 빨강·오른쪽 파랑
                exif = Image.Exif()
                exif[0x0112] = orientation
                stored.save(source, exif=exif)
                ok, written = batch_item_result(source, self.out_dir, BatchJob('format', target_format='PNG'),
                                                resolve_path=lambda raw: raw)
                self.assertTrue(ok, written)
                with Image.open(written) as out:
                    self.assertEqual(out.size, size)
                    rgb = out.convert('RGB')
                    self.assertEqual(rgb.getpixel((1, 1)), top_left)
                    self.assertEqual(rgb.getpixel((size[0] - 2, size[1] - 2)), bottom_right)

    def test_grayscale_rotated_tiff_and_xmp_only_tiff_convert_like_the_editor(self):
        # (Codex S5 #1-a) 흑백(L) 무압축 TIFF + 방향 6 — Pillow mmap 지름길이 세운 크기로 매핑해 뒤섞인
        # 40x20 쓰레기를 PNG 로 썼다(RGB 는 mmap 모드가 아니라 위 테스트로는 안 보였다).
        # (Codex S5 #1-b) XMP 에만 방향 6 — 에디터·브라우저는 무시하는데 Pillow load() 만 돌려 20x40 이 됐다.
        from PIL import TiffImagePlugin
        bright, dark = 220, 30
        stored = Image.new('L', (40, 20), dark)
        stored.paste(bright, (0, 0, 20, 20))   # 저장된 픽셀: 왼쪽 밝음·오른쪽 어두움
        rotated = os.path.join(self.root, '한글_흑백_방향6.tiff')
        exif = Image.Exif()
        exif[0x0112] = 6
        stored.save(rotated, exif=exif)
        xmp_only = os.path.join(self.root, '한글_xmp방향6.tiff')
        info = TiffImagePlugin.ImageFileDirectory_v2()
        info[700] = (b'<x:xmpmeta xmlns:x="adobe:ns:meta/"><rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-'
                     b'syntax-ns#"><rdf:Description xmlns:tiff="http://ns.adobe.com/tiff/1.0/" tiff:Orientation="6"/>'
                     b'</rdf:RDF></x:xmpmeta>')
        info.tagtype[700] = 1
        stored.save(xmp_only, tiffinfo=info)
        # 6(시계 90°): 왼쪽 밝음이 위 절반으로 / XMP 만: 저장된 그대로(왼쪽 절반 밝음). 네 모서리로 확인
        for source, size, corners in (
                (rotated, (20, 40), {(1, 1): bright, (18, 1): bright, (1, 38): dark, (18, 38): dark}),
                (xmp_only, (40, 20), {(1, 1): bright, (1, 18): bright, (38, 1): dark, (38, 18): dark})):
            with self.subTest(source=os.path.basename(source)):
                ok, written = batch_item_result(source, self.out_dir, BatchJob('format', target_format='PNG'),
                                                resolve_path=lambda raw: raw)
                self.assertTrue(ok, written)
                with Image.open(written) as out:
                    self.assertEqual(out.size, size)
                    gray = out.convert('L')
                    self.assertEqual({xy: gray.getpixel(xy) for xy in corners}, corners)

    def test_unreadable_input_is_a_batch_error(self):
        broken = os.path.join(self.root, 'broken.png')
        with open(broken, 'wb') as handle:
            handle.write(b'not a png')
        with self.assertRaises(BatchJobError):
            process_batch_file(broken, self.out_dir, BatchJob('format', target_format='PNG'))

    def _assert_no_host_path(self, message: str):
        self.assertNotIn(self.root, message)
        self.assertNotIn(self.root.replace('\\', '/'), message)
        self.assertNotRegex(message, r'[A-Za-z]:[\\/]')
        self.assertNotIn('편집 이미지', message)   # 에디터 문구가 배치 토스트에 새지 않는다
        self.assertNotIn('cannot identify', message)

    def test_unreadable_input_message_is_batch_worded_without_the_host_path(self):
        """손상 파일 한 개짜리 배치의 최종 토스트 — 예전엔
        '편집 이미지를 읽을 수 없습니다: cannot identify image file 'C:\\…\\x.png'' 였다."""
        broken = os.path.join(self.root, '깨진 파일.png')
        with open(broken, 'wb') as handle:
            handle.write(b'not a png')
        with self.assertRaises(BatchJobError) as caught:
            process_batch_file(broken, self.out_dir, BatchJob('resize', width=8, height=8))
        message = str(caught.exception)
        self.assertEqual(message, '배치: 이미지를 읽을 수 없습니다 (깨진 파일.png)')
        self._assert_no_host_path(message)

        ok, item_message = batch_item_result(broken, self.out_dir, BatchJob('resize', width=8, height=8),
                                             resolve_path=lambda raw: raw)
        self.assertFalse(ok)
        self.assertEqual(item_message, message)
        progress = JobProgress(job='batch', total=1)
        progress.record(False)
        kind, toast = completion_notice(progress, item_message)
        self.assertEqual(kind, 'error')
        self.assertIn('이미지를 읽을 수 없습니다 (깨진 파일.png)', toast)
        self._assert_no_host_path(toast)

    def test_item_result_success_and_rejected_path(self):
        job = BatchJob('format', target_format='PNG')
        ok, written = batch_item_result(self.source, self.out_dir, job, resolve_path=lambda raw: raw)
        self.assertTrue(ok)
        self.assertTrue(os.path.isfile(written))
        self.assertNotIn('\\', written)

        missing = os.path.join(self.root, 'secret', 'gone.png')
        ok, message = batch_item_result(missing, self.out_dir, job, resolve_path=lambda raw: None)
        self.assertFalse(ok)
        self.assertIn('(gone.png)', message)
        self._assert_no_host_path(message)

    def test_unexpected_failures_are_sanitized(self):
        job = BatchJob('format', target_format='PNG')

        def _boom(raw):
            raise RuntimeError(f"permission problem at '{os.path.join(self.root, 'x.png')}'")

        ok, message = batch_item_result(self.source, self.out_dir, job, resolve_path=_boom)
        self.assertFalse(ok)
        self.assertTrue(message.startswith('배치: 처리하지 못했습니다 (generated_1.png)'))
        self.assertIn('[path]', message)
        self._assert_no_host_path(message)

    def test_write_failure_is_batch_worded(self):
        # 출력 폴더 자리에 파일이 있으면 makedirs 가 실패한다 (OSError)
        blocker = os.path.join(self.root, 'blocked')
        with open(blocker, 'wb') as handle:
            handle.write(b'x')
        with self.assertRaises(BatchJobError) as caught:
            process_batch_file(self.source, os.path.join(blocker, 'batch'), BatchJob('format', target_format='PNG'))
        message = str(caught.exception)
        self.assertTrue(message.startswith('배치: 결과 파일을 쓸 수 없습니다 (generated_1.png)'), message)
        self._assert_no_host_path(message)

    def test_worker_uses_the_sanitizing_entry_point(self):
        from pathlib import Path
        source = (Path(__file__).resolve().parents[1] / 'workers' / 'batch_image_worker.py').read_text(encoding='utf-8')
        self.assertIn('batch_item_result(', source)
        self.assertNotIn('str(exc)', source)   # 원문 예외 문구를 그대로 내보내지 않는다


class CompletionNoticeTests(unittest.TestCase):
    def _progress(self, total, success, failed, done=None):
        progress = JobProgress(job='batch', total=total, output_dir='C:/out/batch')
        for _ in range(success):
            progress.record(True)
        for _ in range(failed):
            progress.record(False)
        if done is not None:
            progress.done = done
        progress.finish()
        return progress

    def test_all_success(self):
        kind, message = completion_notice(self._progress(2, 2, 0))
        self.assertEqual(kind, 'success')
        self.assertIn('C:/out/batch', message)

    def test_all_failed_is_error_with_reason(self):
        kind, message = completion_notice(self._progress(2, 0, 2), '이미지를 읽을 수 없습니다')
        self.assertEqual(kind, 'error')
        self.assertIn('이미지를 읽을 수 없습니다', message)

    def test_mixed_and_stopped_are_warnings(self):
        self.assertEqual(completion_notice(self._progress(3, 2, 1))[0], 'warning')
        stopped = self._progress(4, 1, 0)
        self.assertTrue(stopped.stopped)
        kind, message = completion_notice(stopped)
        self.assertEqual(kind, 'warning')
        self.assertIn('건너뜀 3', message)
        self.assertEqual(completion_notice(self._progress(2, 0, 0))[0], 'warning')

    def test_state_json_shape(self):
        state = json.loads(self._progress(2, 1, 1).to_json())
        self.assertEqual(set(state), {'job', 'total', 'output_dir', 'running', 'done', 'success', 'failed', 'stopped'})
        self.assertFalse(state['running'])


class _Signal:
    def __init__(self):
        self.calls = []
        self.slots = []

    def emit(self, *args):
        self.calls.append(args)
        for slot in list(self.slots):
            slot(*args)

    def connect(self, slot):
        self.slots.append(slot)

    def disconnect(self):
        self.slots.clear()


class _Window:
    def __init__(self):
        self.vue_bridge = type('B', (), {})()
        self.vue_bridge.showNotification = _Signal()
        self.vue_bridge.batchJobState = _Signal()


class BatchActionTests(unittest.TestCase):
    def test_real_worker_run_end_to_end_with_one_notice(self):
        from ui.batch_actions import WORKER_ATTR, start_vue_batch
        from workers.batch_image_worker import BatchImageWorker

        class _SyncWorker(BatchImageWorker):
            def start(self):   # 테스트에서는 같은 스레드에서 돌린다
                self.run()

        with tempfile.TemporaryDirectory() as root:
            src = os.path.join(root, 'generated_7.png')
            original = _write_png(src)
            missing = os.path.join(root, 'missing.png')
            window = _Window()
            started = start_vue_batch(window, {
                'files': [src, missing, src],
                'operation': 'resize',
                'settings': {'width': '20', 'height': '10', 'format': 'PNG'},
            }, worker_factory=_SyncWorker, output_root=root)
            self.assertTrue(started)
            self.assertIsInstance(getattr(window, WORKER_ATTR), BatchImageWorker)
            with open(src, 'rb') as handle:
                self.assertEqual(handle.read(), original)
            outputs = os.listdir(os.path.join(root, 'batch'))
        self.assertEqual(outputs, ['generated_7_20x10.png'])
        notices = window.vue_bridge.showNotification.calls
        final = [call for call in notices if call[0] in ('success', 'warning', 'error')]
        self.assertEqual(len(final), 1)
        self.assertEqual(final[0][0], 'warning')   # 1개 성공 · 1개 실패(없는 파일)
        state = json.loads(window.vue_bridge.batchJobState.calls[-1][0])
        self.assertEqual((state['job'], state['running'], state['success'], state['failed']), ('batch', False, 1, 1))

    def test_running_guard_and_bad_settings(self):
        from ui.batch_actions import WORKER_ATTR, start_vue_batch

        class _Busy:
            def isRunning(self):
                return True

        window = _Window()
        setattr(window, WORKER_ATTR, _Busy())
        self.assertFalse(start_vue_batch(window, {'files': ['a.png']}, output_root=tempfile.gettempdir()))
        self.assertEqual(window.vue_bridge.showNotification.calls[-1][0], 'warning')

        window = _Window()
        self.assertFalse(start_vue_batch(window, {'files': ['a.png'], 'operation': 'resize',
                                                  'settings': {'width': 'x', 'height': 1}},
                                         output_root=tempfile.gettempdir()))
        self.assertEqual(window.vue_bridge.showNotification.calls[-1][0], 'error')
        self.assertFalse(start_vue_batch(window, {'files': []}, output_root=tempfile.gettempdir()))

    def test_new_code_does_not_depend_on_legacy_tabs(self):
        from pathlib import Path
        root = Path(__file__).resolve().parents[1]
        for rel in ('ui/batch_actions.py', 'ui/upscale_actions.py', 'ui/inpaint_actions.py', 'ui/batch_jobs.py',
                    'workers/batch_image_worker.py', 'core/batch_image_ops.py'):
            source = (root / rel).read_text(encoding='utf-8')
            self.assertNotIn('from tabs', source, rel)
            self.assertNotIn('import tabs', source, rel)
            # 애플리케이션 모달을 띄우지 않는다 (알림은 Vue 토스트 한 번)
            self.assertNotRegex(source, r'QMessageBox\.|import[^\n]*QMessageBox', rel)


if __name__ == '__main__':
    unittest.main()
