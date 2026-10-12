"""SAM3 '얼굴 안에서만 찾기'(sam-extra 0.34.0) — 앱 쪽 표·위젯·저장/복원·전송·모델 목록·ComfyUI 거절 회귀 테스트.

확장은 Sam3Args 맨 뒤에 sam3_face_crop(False)·sam3_face_crop_model('face_yolov8n.pt')·sam3_face_crop_confidence(0.3)를
더했다. 앱은 켤 때만 세 키를 보낸다(core/sam3_args.apply_face_crop — tests/test_sam3_args.py 가 요청 모양을 본다). 여기서는
T2I 프록시 3개(generator_ui_setup)·저장/복원(_get/_set_sam3_settings)·전송(_build_sam3_settings)·TS 거울(utils/sam3FaceCrop.ts)·
얼굴 모델 추천 목록(설치된 확장 face_detect.find_face_models 와 대조)·ComfyUI 컴파일러 거절을 본다.
"""
from __future__ import annotations

import importlib.util
import re
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from core import sam3_args
from core import sam3_face_crop as fc
from ui.widget_proxies import CheckBoxProxy, ComboBoxProxy, LineEditProxy, SliderProxy
from tests._sam_extra_ext import EXT_ROOT, requires_extension

ROOT = Path(__file__).resolve().parents[1]
TS_FILE = ROOT / 'frontend' / 'src' / 'utils' / 'sam3FaceCrop.ts'


class _Bridge:
    def __init__(self):
        self.values: dict[str, str] = {}
        self.properties: dict[tuple[str, str], object] = {}

    def _register_proxy(self, widget_id, proxy):
        pass

    def pushWidgetValue(self, widget_id, value):
        self.values[widget_id] = value

    def pushWidgetProperty(self, widget_id, prop, value):
        self.properties[(widget_id, prop)] = value


_PROXY_CLASSES = {'CheckBoxProxy': CheckBoxProxy, 'ComboBoxProxy': ComboBoxProxy,
                  'LineEditProxy': LineEditProxy, 'SliderProxy': SliderProxy}
_REGISTRATION_RE = re.compile(r"'(face_crop[a-z_]*)': (\w+Proxy)\(b, '(_sam3_face_crop[a-z_]*)'\)")


def _registered() -> dict:
    source = (ROOT / 'ui' / 'generator_ui_setup.py').read_text(encoding='utf-8')
    return {key: (cls, wid) for key, cls, wid in _REGISTRATION_RE.findall(source)}


def _widgets(bridge) -> dict:
    return {key: _PROXY_CLASSES[cls](bridge, wid) for key, (cls, wid) in _registered().items()}


class TableTests(unittest.TestCase):
    def test_three_fields_are_the_last_three_spec_keys(self):
        self.assertEqual(fc.FACE_CROP_FIELD_KEYS, ('face_crop', 'face_crop_model', 'face_crop_confidence'))
        self.assertEqual(tuple(fc.spec_key(key) for key in fc.FACE_CROP_FIELD_KEYS), sam3_args.FACE_CROP_KEYS)
        self.assertEqual(sam3_args.SAM3_KEYS[-3:], sam3_args.FACE_CROP_KEYS)

    def test_defaults_follow_the_spec(self):
        self.assertEqual(fc.default_values(), {'face_crop': False, 'face_crop_model': 'face_yolov8n.pt',
                                               'face_crop_confidence': '0.3'})

    def test_generator_ui_setup_registers_the_widgets_with_matching_kinds(self):
        registered = _registered()
        self.assertEqual(set(registered), set(fc.FACE_CROP_FIELD_KEYS))
        allowed = {'check': {'CheckBoxProxy'}, 'text': {'LineEditProxy'}}
        for key, kind in fc.FACE_CROP_FIELDS:
            cls, wid = registered[key]
            self.assertIn(cls, allowed[kind], key)
            self.assertEqual(wid, fc.widget_id(key))
        source = (ROOT / 'ui' / 'generator_ui_setup.py').read_text(encoding='utf-8')
        self.assertIn('_init_sam3_face_crop_widgets(self.sam3_widgets)', source)

    def test_ts_mirror_has_the_same_fields_and_defaults(self):
        text = TS_FILE.read_text(encoding='utf-8')
        body = text[text.index('SAM3_FACE_CROP_FIELDS'):]
        rows = re.findall(r"\{ key: '([a-z_]+)', kind: '([a-z]+)', def: '([^']*)' \}", body[: body.index('] as const')])
        kinds = {'check': 'bool', 'text': None}
        self.assertEqual([key for key, _kind, _def in rows], list(fc.FACE_CROP_FIELD_KEYS))
        defaults = fc.default_values()
        for key, kind, default in rows:
            with self.subTest(key=key):
                py_kind = dict(fc.FACE_CROP_FIELDS)[key]
                if kinds[py_kind]:
                    self.assertEqual(kind, kinds[py_kind])
                expected = defaults[key]
                self.assertEqual(default, ('true' if expected else 'false') if isinstance(expected, bool) else expected)
        self.assertIn(f"FACE_CROP_DEFAULT_MODEL = '{sam3_args.FACE_CROP_DEFAULT_MODEL}'", text)
        self.assertIn(f"FACE_CROP_DEFAULT_CONFIDENCE = {sam3_args.FACE_CROP_DEFAULT_CONFIDENCE}", text)


class WidgetPersistenceTests(unittest.TestCase):
    def setUp(self):
        self.bridge = _Bridge()
        self.widgets = _widgets(self.bridge)
        fc.init_widgets(self.widgets)

    def test_init_pushes_the_default_model_item_and_defaults(self):
        self.assertEqual(self.bridge.properties[('_sam3_face_crop_model', 'items')], ['face_yolov8n.pt'])
        self.assertFalse(self.widgets['face_crop'].isChecked())
        self.assertEqual(self.widgets['face_crop_model'].text(), 'face_yolov8n.pt')
        self.assertEqual(self.widgets['face_crop_confidence'].text(), '0.3')

    def test_vue_edits_round_trip_and_old_settings_fall_back_to_off(self):
        w = self.widgets
        w['face_crop']._on_vue_changed('true')
        w['face_crop_model']._on_vue_changed('face_yolov8s.pt')
        w['face_crop_confidence']._on_vue_changed('0.45')
        saved = fc.read_settings(w)
        self.assertEqual(saved, {'face_crop': True, 'face_crop_model': 'face_yolov8s.pt', 'face_crop_confidence': '0.45'})
        fresh = _widgets(_Bridge())
        fc.init_widgets(fresh)
        fc.apply_settings(fresh, saved)
        self.assertEqual(fc.read_settings(fresh), saved)
        fc.apply_settings(fresh, {'threshold': '0.5'})     # 0.34.0 전 저장 파일 — 키 없음 → 꺼짐·기본값
        self.assertEqual(fc.read_settings(fresh), {'face_crop': False, 'face_crop_model': 'face_yolov8n.pt',
                                                   'face_crop_confidence': '0.3'})

    def test_settings_mixin_saves_and_restores_the_fields(self):
        from ui.generator_settings import SettingsMixin
        self.widgets['face_crop'].setChecked(True)
        self.assertEqual(SettingsMixin._get_sam3_face_crop_settings(self.widgets)['face_crop'], True)
        source = (ROOT / 'ui' / 'generator_settings.py').read_text(encoding='utf-8')
        self.assertIn('**self._get_sam3_face_crop_settings(widgets)', source)
        self.assertIn('_apply_sam3_face_crop(widgets, settings)', source)

    def test_build_sam3_settings_sends_the_fields_and_build_state_drops_them_when_off(self):
        from ui.generator_generation import GenerationMixin

        class _Any:
            def text(self): return ''
            def toPlainText(self): return ''
            def currentText(self): return ''
            def isChecked(self): return False

        class _Widgets(dict):
            def __missing__(self, key):
                self[key] = _Any()
                return self[key]

        host = SimpleNamespace(sam3_widgets=_Widgets(self.widgets))
        settings = GenerationMixin._build_sam3_settings(host, {})
        self.assertEqual({key: settings[key] for key in sam3_args.FACE_CROP_KEYS},
                         {'sam3_face_crop': False, 'sam3_face_crop_model': 'face_yolov8n.pt',
                          'sam3_face_crop_confidence': '0.3'})
        self.assertFalse(set(sam3_args.FACE_CROP_KEYS) & set(sam3_args.build_state(settings)))
        self.widgets['face_crop']._on_vue_changed('true')
        self.widgets['face_crop_confidence']._on_vue_changed('0.5')
        state = sam3_args.build_state(GenerationMixin._build_sam3_settings(host, {}))
        self.assertEqual([state[key] for key in sam3_args.FACE_CROP_KEYS], [True, 'face_yolov8n.pt', 0.5])


class ModelChoiceTests(unittest.TestCase):
    def test_extension_dropdown_rules(self):
        names = ['None', 'mediapipe_face_full', 'hand_yolov8n.pt', 'face_yolov8s.pt', 'yolov8x-worldv2.pt',
                 'person_yolov8n-seg.pt', 'face_yolov8n.pt', 'Face_Anime.PT', 'sub/face_extra.pt', 'face_yolov8s.pt']
        self.assertEqual(fc.model_choices(names), ['face_yolov8n.pt', 'Face_Anime.PT', 'face_extra.pt', 'face_yolov8s.pt',
                                                   'hand_yolov8n.pt', 'person_yolov8n-seg.pt'])
        self.assertEqual(fc.model_choices([]), ['face_yolov8n.pt'])
        self.assertEqual(fc.model_choices(['mediapipe_face_short', 'None']), ['face_yolov8n.pt'])

    def test_bridge_pushes_the_choices_when_adetailer_models_arrive(self):
        import json
        from ui.vue_bridge import VueBridge
        bridge = _Bridge()
        proxy = LineEditProxy(bridge, '_sam3_face_crop_model')
        host = SimpleNamespace(_proxies={'_sam3_face_crop_model': proxy})
        VueBridge._apply_adetailer_models_json(host, json.dumps(['hand_yolov8n.pt', 'face_yolov8n.pt']))
        self.assertEqual(bridge.properties[('_sam3_face_crop_model', 'items')], ['face_yolov8n.pt', 'hand_yolov8n.pt'])


@requires_extension
class InstalledExtensionTests(unittest.TestCase):
    """설치된 확장의 face_detect.find_face_models 와 같은 폴더 내용에서 같은 목록 — 모듈은 표준 라이브러리만 불러온다."""

    @classmethod
    def setUpClass(cls):
        import sys
        name = '_samx_face_detect_oracle'
        path = Path(EXT_ROOT) / 'sam3ext' / 'face_detect.py'
        spec = importlib.util.spec_from_file_location(name, path)
        cls.module = importlib.util.module_from_spec(spec)
        sys.modules[name] = cls.module     # @dataclass 가 모듈을 sys.modules 에서 찾는다
        try:
            spec.loader.exec_module(cls.module)
        finally:
            sys.modules.pop(name, None)

    def test_defaults_match(self):
        self.assertEqual(self.module.DEFAULT_MODEL, sam3_args.FACE_CROP_DEFAULT_MODEL)
        self.assertEqual(self.module.DEFAULT_CONFIDENCE, sam3_args.FACE_CROP_DEFAULT_CONFIDENCE)
        self.assertEqual(self.module.MODEL_SUFFIX, fc.MODEL_SUFFIX)

    def test_choices_match_find_face_models(self):
        files = ('hand_yolov8n.pt', 'face_yolov8s.pt', 'yolov8x-worldv2.pt', 'person_yolov8n-seg.pt',
                 'Face_Anime.PT', 'face_yolov8n.pt', 'readme.txt', 'zz_model.pt')
        cases = (files, ('hand_yolov8n.pt',), ('yolov8x-worldv2.pt', 'notes.md'), ())
        for case in cases:
            with self.subTest(files=case), tempfile.TemporaryDirectory() as folder:
                for name in case:
                    (Path(folder) / name).write_bytes(b'')
                original = self.module.model_dirs
                self.module.model_dirs = lambda folder=folder: [Path(folder)]
                try:
                    expected = self.module.find_face_models()
                finally:
                    self.module.model_dirs = original
                self.assertEqual(fc.model_choices(case), expected)


class ComfyCompilerTests(unittest.TestCase):
    """ComfyUI 팩에는 얼굴 crop 이 아직 없다 — 켠 요청은 조용히 전체 이미지에서 찾지 않고 컴파일에서 멈춘다."""

    def test_face_crop_on_is_rejected_and_off_compiles(self):
        from core.comfy_workflow_compiler import ComfyWorkflowCompiler, WorkflowCompileError
        from tests.test_comfy_workflow_compiler import _capabilities
        compiler = ComfyWorkflowCompiler(_capabilities())
        for settings in ({'sam3_mode': 'Inpaint'}, {'sam3_mode': 'Mask only'}):
            with self.subTest(settings=settings):
                off = {'prompt': 'portrait', 'alwayson_scripts': sam3_args.build_alwayson(settings)}
                on = {'prompt': 'portrait', 'alwayson_scripts': sam3_args.build_alwayson(
                    {**settings, 'sam3_face_crop': True})}
                compiler.compile('txt2img', 'checkpoint.safetensors', off)
                compiler.compile_sam3_mask_only(off, uploaded_image='source.png')
                with self.assertRaisesRegex(WorkflowCompileError, '얼굴 안에서만 찾기'):
                    compiler.compile('txt2img', 'checkpoint.safetensors', on)
                with self.assertRaisesRegex(WorkflowCompileError, '얼굴 안에서만 찾기'):
                    compiler.compile_sam3_mask_only(on, uploaded_image='source.png')


if __name__ == '__main__':
    unittest.main()
