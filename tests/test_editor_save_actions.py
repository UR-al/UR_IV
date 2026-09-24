"""에디터 저장 액션(ui.editor_save_actions) 회귀 테스트.

Qt 없이 돈다 — 저장 대화상자(ask_path)와 작업 스레드(run)를 주입한다.
결과는 항상 editorSaveResult 로 정확히 한 번 돌아와야 한다(Vue 가 '저장 중'에 갇히지 않게).

'저장'은 비파괴다 — 사용자가 연 원본(검열 전 생성 원본 등)은 절대 덮어쓰지 않고,
첫 저장은 원본 옆 <stem>_edited[_N] 사본을 만들며, 그 문서의 다음 저장은 그 사본만 갱신한다.
"""
from __future__ import annotations

import base64
import io
import json
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from PIL import Image, PngImagePlugin

from core.editor_save import SavedCopyRegistry
from ui.editor_save_actions import handle_editor_save_action

PARAMS = 'masterpiece, 1girl\nSteps: 28, Sampler: Euler a, Seed: 7'


class _Signal:
    def __init__(self):
        self.payloads: list[dict] = []

    def emit(self, text):
        self.payloads.append(json.loads(text))


def _png(path: Path, color=(10, 20, 30), params: str | None = None) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    info = PngImagePlugin.PngInfo()
    if params:
        info.add_text('parameters', params)
    Image.new('RGB', (32, 24), color).save(path, pnginfo=info)
    return path


class EditorSaveActionTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self.app_root = self.root / 'app'
        self.output_dir = self.root / 'generated_images'
        self.output_dir.mkdir(parents=True)
        # 테스트 파일이 전부 시스템 임시 폴더 안이라, '임시 폴더는 사용자 위치가 아니다'
        # 규칙을 별도 폴더로 돌려 둔다(core.editor_save 안에서만).
        fake_temp = self.root / 'os_temp'
        fake_temp.mkdir()
        patcher = mock.patch('core.editor_save.tempfile')
        self.addCleanup(patcher.stop)
        patched = patcher.start()
        patched.gettempdir.return_value = str(fake_temp)
        self.fake_temp = fake_temp
        self.signal = _Signal()
        self.window = SimpleNamespace(vue_bridge=SimpleNamespace(editorSaveResult=self.signal))
        self.asked: list[tuple[str, str]] = []
        self.answer: tuple[str, str] = ('', '')
        # 테스트마다 새 레지스트리 — 모듈 전역(SAVED_COPIES)을 오염시키지 않는다
        self.registry = SavedCopyRegistry()

    def tearDown(self):
        self._tmp.cleanup()

    def _ask(self, title, suggested):
        self.asked.append((title, suggested))
        return self.answer

    def _run(self, action, payload):
        self.signal.payloads.clear()
        handle_editor_save_action(
            self.window, action, payload, ask_path=self._ask, run=lambda fn: fn(),
            app_root=str(self.app_root), output_dir=str(self.output_dir), registry=self.registry,
        )
        self.assertEqual(len(self.signal.payloads), 1, '결과는 정확히 한 번 와야 한다')
        return self.signal.payloads[0]

    def _edited(self, color=(1, 2, 3), name='edited_x.png') -> Path:
        return _png(self.app_root / 'image_cache' / 'editor_temp' / name, color=color)

    def test_first_save_writes_edited_copy_beside_source_and_keeps_original(self):
        source = _png(self.output_dir / 'gen.png', params=PARAMS)
        before = source.read_bytes()
        edited = self._edited()
        result = self._run('editor_save', {
            'request_id': 77, 'path': str(edited), 'source_path': str(source).replace('\\', '/'),
        })
        self.assertEqual(self.asked, [], '저장이 대화상자를 띄웠다')
        self.assertTrue(result['ok'], result)
        self.assertEqual(result['request_id'], 77)
        self.assertEqual(result['mode'], 'save')
        self.assertTrue(result['owned'], '문서가 방금 쓴 사본은 다음 저장이 덮어써도 된다')
        out = Path(result['path'])
        self.assertEqual(out.parent.resolve(), source.parent.resolve())
        self.assertEqual(out.name, 'gen_edited.png')
        self.assertEqual(source.read_bytes(), before, '원본(생성 원본)이 덮어써졌다')
        with Image.open(out) as img:
            img.load()
            self.assertEqual(img.text.get('parameters'), PARAMS, '사본에 생성 파라미터가 없다')
            self.assertEqual(img.convert('RGB').getpixel((1, 1)), (1, 2, 3))

    def test_second_save_overwrites_only_the_documents_own_copy(self):
        source = _png(self.output_dir / 'gen.png', params=PARAMS)
        before = source.read_bytes()
        first = self._run('editor_save', {'request_id': 1, 'path': str(self._edited((1, 1, 1))),
                                          'source_path': str(source)})
        copy = first['path']
        second = self._run('editor_save', {
            'request_id': 2, 'path': str(self._edited((200, 0, 0), name='edited_y.png')),
            'source_path': copy, 'overwrite_source': True,
        })
        self.assertTrue(second['ok'], second)
        self.assertEqual(Path(second['path']).resolve(), Path(copy).resolve(), '사본을 갱신하지 않고 또 만들었다')
        self.assertTrue(second['owned'])
        with Image.open(copy) as img:
            img.load()
            self.assertEqual(img.convert('RGB').getpixel((1, 1)), (200, 0, 0))
            self.assertEqual(img.text.get('parameters'), PARAMS, '다시 저장하며 메타데이터를 잃었다')
        self.assertEqual(source.read_bytes(), before)
        self.assertEqual(sorted(p.name for p in self.output_dir.iterdir()), ['gen.png', 'gen_edited.png'])

    def test_overwrite_flag_never_reaches_a_file_the_editor_did_not_write(self):
        """프론트가 '내 사본'이라고 우겨도, 이번 실행에서 에디터가 쓴 파일이 아니면 덮어쓰지 않는다."""
        source = _png(self.output_dir / 'gen.png', params=PARAMS)
        before = source.read_bytes()
        result = self._run('editor_save', {'request_id': 3, 'path': str(self._edited()),
                                           'source_path': str(source), 'overwrite_source': True})
        self.assertTrue(result['ok'], result)
        self.assertEqual(Path(result['path']).name, 'gen_edited.png')
        self.assertEqual(source.read_bytes(), before)

    def test_reopened_edited_copy_gets_the_next_number(self):
        """이전 실행에서 저장한 사본을 다시 열어 저장하면 _edited_edited 가 아니라 _edited_2."""
        _png(self.output_dir / 'gen.png')
        old_copy = _png(self.output_dir / 'gen_edited.png', color=(9, 9, 9))
        before = old_copy.read_bytes()
        result = self._run('editor_save', {'request_id': 4, 'path': str(self._edited()),
                                           'source_path': str(old_copy)})
        self.assertEqual(Path(result['path']).name, 'gen_edited_2.png')
        self.assertEqual(old_copy.read_bytes(), before)

    def test_unchanged_original_is_not_copied(self):
        source = _png(self.output_dir / 'gen.png', params=PARAMS)
        result = self._run('editor_save', {'request_id': 5, 'path': str(source), 'source_path': str(source)})
        self.assertTrue(result['ok'], result)
        self.assertTrue(result['unchanged'])
        self.assertFalse(result['owned'], '원본을 이 문서의 사본으로 여기면 다음 저장이 원본을 덮어쓴다')
        self.assertEqual(Path(result['path']).resolve(), source.resolve())
        self.assertEqual([p.name for p in self.output_dir.iterdir()], ['gen.png'])

    def test_alpha_edit_of_jpeg_source_is_saved_as_png(self):
        source = self.output_dir / 'photo.jpg'
        Image.new('RGB', (32, 24), (90, 90, 90)).save(source, quality=90)
        before = source.read_bytes()
        cut = Image.new('RGBA', (32, 24), (0, 200, 0, 255))
        cut.paste((0, 0, 0, 0), (0, 0, 16, 24))   # 배경 제거로 왼쪽 절반이 투명
        edited = self.app_root / 'image_cache' / 'editor_temp' / 'edited_cut.png'
        edited.parent.mkdir(parents=True, exist_ok=True)
        cut.save(edited)
        result = self._run('editor_save', {'request_id': 6, 'path': str(edited), 'source_path': str(source)})
        self.assertTrue(result['ok'], result)
        self.assertTrue(result.get('alpha_png'))
        self.assertEqual(result['format'], 'PNG')
        out = Path(result['path'])
        self.assertEqual(out.name, 'photo_edited.png')
        with Image.open(out) as img:
            self.assertEqual(img.mode, 'RGBA')
            self.assertEqual(img.getpixel((2, 2))[3], 0, '투명 영역이 흰색으로 뭉개졌다')
        self.assertEqual(source.read_bytes(), before)

    def test_opaque_edit_of_jpeg_source_stays_jpeg(self):
        source = self.output_dir / 'photo.jpg'
        Image.new('RGB', (32, 24), (90, 90, 90)).save(source, quality=90)
        result = self._run('editor_save', {'request_id': 7, 'path': str(self._edited()), 'source_path': str(source)})
        self.assertEqual(Path(result['path']).name, 'photo_edited.jpg')
        self.assertEqual(result['format'], 'JPEG')
        self.assertNotIn('alpha_png', result)

    def test_save_with_temp_source_writes_to_output_dir_without_dialog(self):
        clip = _png(self.fake_temp / 'AIStudioPro_editor' / 'clipboard_1.png')
        before = clip.read_bytes()
        result = self._run('editor_save', {'request_id': 1, 'path': str(self._edited()), 'source_path': str(clip)})
        self.assertEqual(self.asked, [], "'저장'은 대화상자를 띄우지 않는다 — 위치는 기본 출력 폴더")
        self.assertTrue(result['ok'], result)
        out = Path(result['path'])
        self.assertEqual(out.parent.resolve(), self.output_dir.resolve())
        self.assertRegex(out.name, r'^edited_\d{8}_\d{6}(_\d+)?\.png$')
        self.assertTrue(result['owned'])
        self.assertEqual(clip.read_bytes(), before)

    def test_unedited_temporary_source_is_still_written(self):
        """복구 작업 사본·클립보드를 편집 없이 저장해도 '변경 없음'으로 끝내면 안 된다 —
        아직 어디에도 저장되지 않은 그림이다."""
        recovered = _png(self.app_root / 'image_cache' / 'editor_temp' / 'recovered_ab.png', color=(4, 5, 6))
        result = self._run('editor_save', {'request_id': 14, 'path': str(recovered),
                                           'source_path': str(recovered)})
        self.assertTrue(result['ok'], result)
        self.assertNotIn('unchanged', result)
        out = Path(result['path'])
        self.assertEqual(out.parent.resolve(), self.output_dir.resolve())
        with Image.open(out) as img:
            self.assertEqual(img.convert('RGB').getpixel((1, 1)), (4, 5, 6))

    def test_save_creates_missing_output_dir(self):
        missing = self.root / 'not_yet' / 'generated'
        edited = self._edited()
        self.signal.payloads.clear()
        handle_editor_save_action(
            self.window, 'editor_save', {'request_id': 8, 'path': str(edited), 'source_path': ''},
            ask_path=self._ask, run=lambda fn: fn(), app_root=str(self.app_root),
            output_dir=str(missing), registry=self.registry,
        )
        result = self.signal.payloads[0]
        self.assertTrue(result['ok'], result)
        self.assertEqual(Path(result['path']).parent.resolve(), missing.resolve())

    def test_save_as_to_a_drive_root_is_allowed(self):
        """대화상자로 USB 루트(D:\\) 같은 곳을 고른 건 정당하다 — 출력 폴더 규칙으로 막지 않는다."""
        edited = self._edited()
        root_target = os.path.join(Path(self.root).anchor, 'edited.png')
        self.answer = (root_target, 'PNG (*.png)')
        captured = {}

        def fake_save(edited_path, target_path, **kwargs):
            captured['target'] = target_path
            return {'ok': True, 'path': target_path.replace('\\', '/'), 'format': 'PNG', 'width': 1, 'height': 1}

        with mock.patch('ui.editor_save_actions.save_edited_image', side_effect=fake_save):
            result = self._run('editor_save_as', {'request_id': 12, 'path': str(edited), 'source_path': ''})
        self.assertTrue(result['ok'], result)
        self.assertEqual(os.path.normcase(captured['target']), os.path.normcase(root_target))

    def test_save_as_into_system_folder_is_refused(self):
        edited = self._edited()
        windows_dir = os.environ.get('SystemRoot') or 'C:\\Windows'
        self.answer = (os.path.join(windows_dir, 'edited.png'), 'PNG (*.png)')
        result = self._run('editor_save_as', {'request_id': 13, 'path': str(edited), 'source_path': ''})
        self.assertFalse(result['ok'])
        self.assertIn('저장할 수 없습니다', result['error'])

    def test_save_as_registers_target_so_following_save_updates_it(self):
        source = _png(self.output_dir / 'gen.png', params=PARAMS)
        chosen = self.root / 'exports' / 'final.png'
        chosen.parent.mkdir()
        self.answer = (str(chosen), 'PNG (*.png)')
        first = self._run('editor_save_as', {'request_id': 20, 'path': str(self._edited()), 'source_path': str(source)})
        self.assertTrue(first['owned'])
        second = self._run('editor_save', {
            'request_id': 21, 'path': str(self._edited((5, 6, 7), name='edited_z.png')),
            'source_path': first['path'], 'overwrite_source': True,
        })
        self.assertEqual(Path(second['path']).resolve(), chosen.resolve())
        with Image.open(chosen) as img:
            self.assertEqual(img.convert('RGB').getpixel((1, 1)), (5, 6, 7))

    def test_save_as_uses_selected_filter_format_and_source_metadata(self):
        source = _png(self.output_dir / 'gen.png', params=PARAMS)
        edited = self._edited(color=(200, 10, 10))
        chosen = self.root / 'exports' / 'final'
        chosen.parent.mkdir()
        self.answer = (str(chosen), 'JPEG (*.jpg *.jpeg)')
        result = self._run('editor_save_as', {'request_id': 5, 'path': str(edited), 'source_path': str(source)})
        self.assertTrue(result['ok'], result)
        self.assertEqual(result['mode'], 'save_as')
        self.assertEqual(result['format'], 'JPEG')
        out = Path(result['path'])
        self.assertEqual(out.suffix, '.jpg')
        self.assertEqual(out.read_bytes()[:2], b'\xff\xd8')
        # 추천 이름은 원본 이름을 따른다 (예전: edited_<uuid>_edited.png)
        self.assertEqual(Path(self.asked[0][1]).name, 'gen_edited.png')
        from core.image_metadata import extract_from_file
        self.assertEqual(extract_from_file(out).raw_parameters, PARAMS)
        # 원본은 그대로
        with Image.open(source) as img:
            self.assertEqual(img.convert('RGB').getpixel((1, 1)), (10, 20, 30))

    def test_save_as_into_app_cache_is_refused(self):
        edited = self._edited()
        self.answer = (str(self.app_root / 'image_cache' / 'editor_temp' / 'mine.png'), 'PNG (*.png)')
        result = self._run('editor_save_as', {'request_id': 2, 'path': str(edited), 'source_path': ''})
        self.assertFalse(result['ok'])
        self.assertIn('임시 폴더', result['error'])

    def test_missing_edited_image_reports_error_without_dialog(self):
        result = self._run('editor_save', {'request_id': 3, 'path': str(self.root / 'nope.png')})
        self.assertFalse(result['ok'])
        self.assertIn('찾을 수 없습니다', result['error'])
        self.assertEqual(self.asked, [])

    def test_worker_failure_still_answers(self):
        source = _png(self.output_dir / 'gen.png')
        edited = self._edited()
        with mock.patch('ui.editor_save_actions.save_edited_image', side_effect=RuntimeError('boom')), \
                mock.patch('core.error_handler.handle_error'):
            result = self._run('editor_save', {'request_id': 9, 'path': str(edited), 'source_path': str(source)})
        self.assertFalse(result['ok'])
        self.assertIn('boom', result['error'])

    @staticmethod
    def _overlay_payload() -> str:
        over = Image.new('RGBA', (32, 24), (0, 0, 0, 0))
        over.paste((0, 255, 0, 255), (0, 0, 4, 4))
        buf = io.BytesIO()
        over.save(buf, 'PNG')
        return 'data:image/png;base64,' + base64.b64encode(buf.getvalue()).decode()

    def test_save_with_only_a_draw_layer_keeps_the_original(self):
        """원본을 열고 레이어에만 그려 저장해도 원본은 그대로, 합성본은 사본으로."""
        source = _png(self.output_dir / 'gen.png', color=(40, 40, 40))
        before = source.read_bytes()
        vue_path = str(source).replace('\\', '/')
        result = self._run('editor_save', {
            'request_id': 10, 'path': vue_path, 'source_path': vue_path,
            'overlay_base64': self._overlay_payload(), 'overlay_opacity': 100, 'protect_paths': [vue_path],
        })
        self.assertTrue(result['ok'], result)
        self.assertNotIn('replaced_paths', result, '원본을 덮어쓰지 않으니 히스토리를 바꿀 일이 없다')
        self.assertEqual(source.read_bytes(), before)
        with Image.open(result['path']) as img:
            self.assertEqual(img.convert('RGB').getpixel((1, 1)), (0, 255, 0))

    def test_protected_history_path_is_snapshotted(self):
        """'다른 이름으로 저장'으로 지금 연 파일을 골라 덮어쓰면, 히스토리용 사본을 먼저 뜬다."""
        source = _png(self.output_dir / 'gen.png', color=(40, 40, 40))
        before = source.read_bytes()
        vue_path = str(source).replace('\\', '/')
        self.answer = (str(source), 'PNG (*.png)')
        result = self._run('editor_save_as', {
            'request_id': 11, 'path': vue_path, 'source_path': vue_path,
            'overlay_base64': self._overlay_payload(), 'overlay_opacity': 100, 'protect_paths': [vue_path],
        })
        self.assertTrue(result['ok'], result)
        self.assertEqual(result['replaced_paths'], [vue_path])
        self.assertEqual(Path(result['snapshot_path']).read_bytes(), before)
        with Image.open(source) as img:
            self.assertEqual(img.convert('RGB').getpixel((1, 1)), (0, 255, 0))


if __name__ == '__main__':
    unittest.main()
