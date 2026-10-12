"""SAM3 '얼굴 안에서만 찾기'(sam-extra 0.34.0) — 앱 쪽 표·위젯·저장/복원·전송·모델 목록·ComfyUI 거절 회귀 테스트.

확장은 Sam3Args 맨 뒤에 sam3_face_crop(False)·sam3_face_crop_model('face_yolov8n.pt')·sam3_face_crop_confidence(0.3)를
더했다. 앱은 켤 때만 세 키를 보낸다(core/sam3_args.apply_face_crop — tests/test_sam3_args.py 가 요청 모양을 본다). 여기서는
T2I 프록시 3개(generator_ui_setup)·저장/복원(_get/_set_sam3_settings)·전송(_build_sam3_settings)·TS 거울(utils/sam3FaceCrop.ts)·
얼굴 모델 추천 목록(설치된 확장 face_detect.find_face_models 와 대조)·ComfyUI(앱 노드 팩 1.8.0 — 켠 요청만 노드 입력 셋, 옛 팩·
ultralytics 없는 ComfyUI 는 이유와 함께 멈춤, 모델 절대 경로, 연결 때 카드로 보내는 상태)를 본다. 팩 자체의 계산은
tests/test_comfy_sam3_face_crop.py 가 sam-extra 와 대조한다.
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
        from unittest import mock
        from backends import BackendType
        from ui.vue_bridge import VueBridge
        bridge = _Bridge()
        proxy = LineEditProxy(bridge, '_sam3_face_crop_model')
        host = SimpleNamespace(_proxies={'_sam3_face_crop_model': proxy})
        with mock.patch('backends.get_backend_type', return_value=BackendType.WEBUI):
            VueBridge._apply_adetailer_models_json(host, json.dumps(['hand_yolov8n.pt', 'face_yolov8n.pt']))
        self.assertEqual(bridge.properties[('_sam3_face_crop_model', 'items')], ['face_yolov8n.pt', 'hand_yolov8n.pt'])

    def test_comfyui_suggests_the_forge_adetailer_files_the_compiler_sends(self):
        """ComfyUI 에는 ADetailer 모델 API 가 없다(앱의 기본값 대체 목록) — 추천은 resolve_model_path 가 찾는 앱 Forge
        models/adetailer 의 파일(확장 find_face_models 규칙)."""
        import json
        from unittest import mock
        from backends import BackendType
        from ui.vue_bridge import VueBridge
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / 'adetailer' / 'sub').mkdir(parents=True)
            for name in ('face_yolov8m.pt', 'hand_yolov8n.pt', 'yolov8x-worldv2.pt', 'notes.txt'):
                (root / 'adetailer' / name).write_bytes(b'')
            (root / 'adetailer' / 'sub' / 'face_yolov8n.pt').write_bytes(b'')
            self.assertEqual(fc.local_model_names(root), ['face_yolov8m.pt', 'face_yolov8n.pt', 'hand_yolov8n.pt'])
            self.assertEqual(fc.local_model_names(root / 'missing'), [])
            bridge = _Bridge()
            host = SimpleNamespace(_proxies={'_sam3_face_crop_model': LineEditProxy(bridge, '_sam3_face_crop_model')})
            fallback = ['face_yolov8n.pt', 'hand_yolov8n.pt', 'person_yolov8n-seg.pt', 'mediapipe_face_full']
            with mock.patch('backends.get_backend_type', return_value=BackendType.COMFYUI), \
                    mock.patch('core.forge_modules.get_forge_root', return_value=root):
                VueBridge._apply_adetailer_models_json(host, json.dumps(fallback))
            self.assertEqual(bridge.properties[('_sam3_face_crop_model', 'items')],
                             ['face_yolov8n.pt', 'face_yolov8m.pt', 'hand_yolov8n.pt'])
        webui = (ROOT / 'ui' / 'generator_webui.py').read_text(encoding='utf-8')
        loaded = webui[webui.index('def on_webui_info_loaded'):webui.index('def on_webui_info_error')]
        self.assertIn("push_comfy_model_items((getattr(self, 'sam3_widgets', None) or {}).get('face_crop_model'))", loaded)


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


def _mask_node(graph) -> dict:
    nodes = [node for node in graph.values() if node.get('class_type') == fc.COMFY_NODE_CLASS]
    assert len(nodes) == 1, nodes
    return nodes[0]['inputs']


def _pack_schema(detector: bool = True) -> dict:
    """The real pack node's /object_info entry (JSON round trip of INPUT_TYPES), ultralytics found or not."""
    import json
    from unittest import mock
    from comfy_custom_nodes.ai_studio_forge_parity import sam3_face_detect, sam3_nodes

    with mock.patch.object(sam3_face_detect.importlib.util, 'find_spec', return_value=object() if detector else None):
        return {'input': json.loads(json.dumps(sam3_nodes.ForgeNeoSAM3Mask.INPUT_TYPES()))}


class ComfySupportTests(unittest.TestCase):
    """core/sam3_face_crop.comfy_support — /object_info → ready · no_detector · old_pack · 모름(None)."""

    def test_states_from_the_node_schema(self):
        self.assertIsNone(fc.comfy_support(None))
        self.assertIsNone(fc.comfy_support({fc.COMFY_NODE_CLASS: {'input': {'required': {}}}}), '클래스 이름만')
        self.assertEqual(fc.comfy_support({}), fc.COMFY_OLD_PACK, '노드 없음')
        self.assertEqual(fc.comfy_support({fc.COMFY_NODE_CLASS: _pack_schema()}), fc.COMFY_READY)
        self.assertEqual(fc.comfy_support({fc.COMFY_NODE_CLASS: _pack_schema(detector=False)}), fc.COMFY_NO_DETECTOR)
        old = _pack_schema()
        for key in (*fc.COMFY_INPUTS, fc.COMFY_DETECTOR_INPUT):
            old['input']['optional'].pop(key)
        self.assertEqual(fc.comfy_support({fc.COMFY_NODE_CLASS: old}), fc.COMFY_OLD_PACK, '1.7.0 팩')
        new_style = _pack_schema()
        new_style['input']['optional'][fc.COMFY_DETECTOR_INPUT] = ['COMBO', {'options': ['ultralytics']}]
        self.assertEqual(fc.comfy_support({fc.COMFY_NODE_CLASS: new_style}), fc.COMFY_READY)

    def test_inputs_and_detector_names_are_the_packs(self):
        optional = _pack_schema()['input']['optional']
        self.assertTrue(set(fc.COMFY_INPUTS) <= set(optional))
        self.assertEqual(optional[fc.COMFY_DETECTOR_INPUT][0], [fc.COMFY_DETECTOR_READY])
        self.assertEqual(tuple(fc.spec_key(key) for key in fc.COMFY_INPUTS), sam3_args.FACE_CROP_KEYS)

    def test_pack_version_is_at_least_the_feature_pack(self):
        from core.comfy_node_pack import PACK_VERSION

        def as_tuple(text):
            return tuple(int(part) for part in text.split('.'))

        self.assertGreaterEqual(as_tuple(PACK_VERSION), as_tuple(fc.COMFY_MIN_PACK_VERSION))

    def test_model_path_from_forge_adetailer(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / 'adetailer' / 'sub').mkdir(parents=True)
            direct = root / 'adetailer' / 'face_yolov8n.pt'
            direct.write_bytes(b'')
            nested = root / 'adetailer' / 'sub' / 'Face_YOLOv8s.pt'
            nested.write_bytes(b'')
            (root / 'adetailer' / 'yolov8x-worldv2.pt').write_bytes(b'')
            self.assertEqual(fc.resolve_model_path('face_yolov8n.pt', root), str(direct.resolve()))
            self.assertEqual(fc.resolve_model_path('', root), str(direct.resolve()), '빈 값 = 기본 모델')
            self.assertEqual(fc.resolve_model_path('face_yolov8s.pt', root), str(nested.resolve()))
            self.assertEqual(fc.resolve_model_path('yolov8x-worldv2.pt', root), 'yolov8x-worldv2.pt', 'YOLO-World 제외')
            self.assertEqual(fc.resolve_model_path('missing.pt', root), 'missing.pt', '못 찾으면 이름 그대로')
            absolute = str(Path(folder) / 'elsewhere.pt')
            self.assertEqual(fc.resolve_model_path(absolute, root), absolute)

    def test_push_to_the_card(self):
        bridge = _Bridge()
        host = SimpleNamespace(vue_bridge=bridge)
        fc.push_comfy_support(host, fc.COMFY_NO_DETECTOR)
        self.assertEqual(bridge.properties[('_sam3_face_crop', 'comfySupport')], 'no_detector')
        fc.push_comfy_support(host, None)
        self.assertIsNone(bridge.properties[('_sam3_face_crop', 'comfySupport')])
        fc.push_comfy_support(SimpleNamespace(), 'ready')       # 브리지 없음 — 조용히

    def test_ts_mirror_names(self):
        text = TS_FILE.read_text(encoding='utf-8')
        self.assertIn(f"COMFY_SUPPORT_PROPERTY = '{fc.COMFY_SUPPORT_PROPERTY}'", text)
        self.assertIn(f"COMFY_MIN_PACK_VERSION = '{fc.COMFY_MIN_PACK_VERSION}'", text)
        self.assertIn(f"'{fc.COMFY_READY}' | '{fc.COMFY_NO_DETECTOR}' | '{fc.COMFY_OLD_PACK}'", text)

    def test_backend_info_and_connection_wiring(self):
        from backends.base import BackendInfo
        self.assertIsNone(BackendInfo().sam3_face_crop_comfy)
        backend = (ROOT / 'backends' / 'comfyui_backend.py').read_text(encoding='utf-8')
        self.assertIn('info.sam3_face_crop_comfy = comfy_support(obj_info)', backend)
        worker = (ROOT / 'workers' / 'generation_worker.py').read_text(encoding='utf-8')
        self.assertIn("'sam3_face_crop_comfy': getattr(info, 'sam3_face_crop_comfy', None)", worker)
        webui = (ROOT / 'ui' / 'generator_webui.py').read_text(encoding='utf-8')
        self.assertIn("push_comfy_support(self, info.get('sam3_face_crop_comfy'))", webui)
        self.assertIn('watch_comfy_support(self)', webui)
        self.assertIn('push_comfy_support(self, None)', webui)


class ComfySupportRefreshTests(unittest.TestCase):
    """연결 뒤에도 ComfyUI 가 /object_info 를 새로 받을 때마다 카드 상태가 바뀐다 — 팩 설치·재시작 뒤 생성의 새 스키마, 카드 ↻,
    재연결(ComfyUIBackend.get_object_info → publish_comfy_support → watch_comfy_support 가 등록한 GUI). 예전엔 연결 때만 보내
    팩 갱신·ultralytics 설치 뒤에도 '옛 팩'·'검출기 없음' 이 다시 연결할 때까지 남았다."""

    def setUp(self):
        from unittest import mock
        from backends.comfyui_backend import ComfyUIBackend

        self.mock = mock
        self.bridge = _Bridge()
        self.host = SimpleNamespace(vue_bridge=self.bridge)
        self.backend = ComfyUIBackend('http://127.0.0.1:1', workflow_path='')
        self.addCleanup(fc.set_comfy_support_listener, None)
        fc.watch_comfy_support(self.host)
        patcher = mock.patch('backends.get_backend', return_value=self.backend)
        patcher.start()
        self.addCleanup(patcher.stop)

    def _fetch(self, document, backend=None):
        response = SimpleNamespace(json=lambda: document, content=b'{}')
        with self.mock.patch('core.http_retry.get_with_retry', return_value=response):
            return (backend or self.backend).get_object_info()

    def _card(self):
        return self.bridge.properties.get(('_sam3_face_crop', 'comfySupport'), 'unset')

    def test_a_fresh_schema_updates_the_card(self):
        self._fetch({fc.COMFY_NODE_CLASS: {'input': {'required': {'image': ['IMAGE']}, 'optional': {}}}})
        self.assertEqual(self._card(), fc.COMFY_OLD_PACK, 'pack 1.6.0 at connect')
        self._fetch({fc.COMFY_NODE_CLASS: _pack_schema(detector=False)})
        self.assertEqual(self._card(), fc.COMFY_NO_DETECTOR, 'after the pack install and restart')
        self._fetch({fc.COMFY_NODE_CLASS: _pack_schema(detector=True)})
        self.assertEqual(self._card(), fc.COMFY_READY, 'after installing ultralytics, re-checked')

    def test_unknown_documents_and_other_backends_do_not_change_the_card(self):
        from backends.comfyui_backend import ComfyUIBackend

        self._fetch({fc.COMFY_NODE_CLASS: _pack_schema()})
        self._fetch({fc.COMFY_NODE_CLASS: {'input': {'required': {}}}})          # class-only: unknown
        self.assertEqual(self._card(), fc.COMFY_READY)
        old = ComfyUIBackend('http://127.0.0.1:2', workflow_path='')
        self._fetch({}, backend=old)                                              # not the active backend
        self.assertEqual(self._card(), fc.COMFY_READY)
        fc.set_comfy_support_listener(lambda *_: (_ for _ in ()).throw(RuntimeError('GUI gone')))
        self.assertIsInstance(self._fetch({}), dict, 'a failing listener never breaks the fetch')

    def test_the_compile_after_a_pack_install_and_the_card_refresh_both_publish(self):
        """팩 설치 뒤 컴파일(_compile_object_info — 캐시가 비어 새로 받는다)과 카드 ↻(vae_degrid_ui.refresh_comfy_models)."""
        from ui import vae_degrid_ui

        response = SimpleNamespace(json=lambda: {fc.COMFY_NODE_CLASS: _pack_schema(detector=False)}, content=b'{}')
        with self.mock.patch('core.http_retry.get_with_retry', return_value=response):
            self.backend._object_info_cache.invalidate()
            self.backend._compile_object_info()
        self.assertEqual(self._card(), fc.COMFY_NO_DETECTOR)
        response = SimpleNamespace(json=lambda: {fc.COMFY_NODE_CLASS: _pack_schema(detector=True)}, content=b'{}')
        with self.mock.patch('core.http_retry.get_with_retry', return_value=response):
            thread = vae_degrid_ui.refresh_comfy_models(SimpleNamespace(vue_bridge=_Bridge()))
            thread.join(5)
        self.assertEqual(self._card(), fc.COMFY_READY)


class ComfyCompilerTests(unittest.TestCase):
    """앱 노드 팩 1.8.0: 켠 요청만 ForgeNeoSAM3Mask 에 face_crop 입력 셋(모델은 Forge models/adetailer 의 절대 경로), 끈 요청
    그래프는 그대로, 옛 팩·ultralytics 없는 ComfyUI 는 조용히 전체 이미지에서 찾지 않고 이유와 함께 멈춘다."""

    def _payloads(self, settings):
        off = {'prompt': 'portrait', 'seed': 1234, 'alwayson_scripts': sam3_args.build_alwayson(settings)}
        on = {'prompt': 'portrait', 'seed': 1234, 'alwayson_scripts': sam3_args.build_alwayson(
            {**settings, 'sam3_face_crop': True, 'sam3_face_crop_model': 'face_yolov8s.pt',
             'sam3_face_crop_confidence': 0.45})}
        return off, on

    def _compile(self, compiler, payload, mode):
        if mode == 'Mask only':
            return compiler.compile_sam3_mask_only(payload, uploaded_image='source.png')
        return compiler.compile('txt2img', 'checkpoint.safetensors', payload)

    def test_off_is_unchanged_and_on_adds_the_three_inputs(self):
        from unittest import mock
        from core.comfy_workflow_compiler import ComfyWorkflowCompiler
        from tests.test_comfy_workflow_compiler import _capabilities

        with tempfile.TemporaryDirectory() as folder:
            (Path(folder) / 'adetailer').mkdir()
            model = Path(folder) / 'adetailer' / 'face_yolov8s.pt'
            model.write_bytes(b'')
            for schema in (None, _pack_schema()):
                capabilities = _capabilities()
                if schema is not None:
                    capabilities[fc.COMFY_NODE_CLASS] = schema
                compiler = ComfyWorkflowCompiler(capabilities)
                for mode in ('Inpaint', 'Mask only'):
                    with self.subTest(schema=schema is not None, mode=mode), \
                            mock.patch('core.forge_modules.get_forge_root', return_value=Path(folder)):
                        off, on = self._payloads({'sam3_mode': mode})
                        off_inputs = _mask_node(self._compile(compiler, off, mode))
                        self.assertFalse({*fc.COMFY_INPUTS, fc.COMFY_DETECTOR_INPUT} & set(off_inputs))
                        on_inputs = _mask_node(self._compile(compiler, on, mode))
                        self.assertEqual({key: on_inputs[key] for key in fc.COMFY_INPUTS},
                                         {'face_crop': True, 'face_crop_model': str(model.resolve()),
                                          'face_crop_confidence': 0.45})
                        self.assertNotIn(fc.COMFY_DETECTOR_INPUT, on_inputs, '정보용 COMBO 는 보내지 않는다')
                        self.assertEqual({k: v for k, v in on_inputs.items() if k not in fc.COMFY_INPUTS}, off_inputs)

    def test_old_pack_or_no_detector_stops_with_the_reason(self):
        from core.comfy_workflow_compiler import ComfyWorkflowCompiler, WorkflowCompileError
        from tests.test_comfy_workflow_compiler import _capabilities

        old = _pack_schema()
        for key in (*fc.COMFY_INPUTS, fc.COMFY_DETECTOR_INPUT):
            old['input']['optional'].pop(key)
        cases = ((old, '1.8.0 이상'), (_pack_schema(detector=False), 'ultralytics'))
        for schema, reason in cases:
            capabilities = _capabilities()
            capabilities[fc.COMFY_NODE_CLASS] = schema
            compiler = ComfyWorkflowCompiler(capabilities)
            for mode in ('Inpaint', 'Mask only'):
                with self.subTest(reason=reason, mode=mode):
                    off, on = self._payloads({'sam3_mode': mode})
                    self._compile(compiler, off, mode)
                    with self.assertRaisesRegex(WorkflowCompileError, f"얼굴 안에서만 찾기.*{reason}"):
                        self._compile(compiler, on, mode)


if __name__ == '__main__':
    unittest.main()
