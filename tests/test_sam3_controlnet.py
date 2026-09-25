"""T2I SAM3 ControlNet 13필드 — 위젯 초기값·선택지·설정 저장/복원·Vue 배선 회귀 테스트 (#47).

예전엔 ``_sam3_cn_*`` 프록시 13개가 있었지만 Vue 어디에도 바인딩되지 않아 T2I SAM3 가
늘 CN off 로 나갔고, ``_get_sam3_settings``/``_set_sam3_settings`` 에 cn 키가 없어 값이
재시작마다 초기화됐다.
"""
from __future__ import annotations

import json
import re
import unittest
from pathlib import Path

from core import sam3_args
from core import sam3_controlnet as cn
from ui.generator_settings import SettingsMixin
from ui.widget_proxies import CheckBoxProxy, ComboBoxProxy, LineEditProxy, SliderProxy

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / 'frontend' / 'src'


class _Bridge:
    def __init__(self):
        self.values: dict[str, str] = {}
        self.properties: dict[tuple[str, str], object] = {}
        self.proxies: dict[str, object] = {}

    def _register_proxy(self, widget_id, proxy):
        self.proxies[widget_id] = proxy

    def pushWidgetValue(self, widget_id, value):
        self.values[widget_id] = value

    def pushWidgetProperty(self, widget_id, prop, value):
        self.properties[(widget_id, prop)] = value


_PROXY_CLASSES = {
    'CheckBoxProxy': CheckBoxProxy,
    'ComboBoxProxy': ComboBoxProxy,
    'LineEditProxy': LineEditProxy,
    'SliderProxy': SliderProxy,
}
_REGISTRATION_RE = re.compile(
    r"'(cn_[a-z_]+)': (\w+Proxy)\(b, '(_sam3_cn_[a-z_]+)'(?:, multiplier=(\d+))?\)")


def _registered_cn_proxies() -> dict:
    """generator_ui_setup.py 가 실제로 등록하는 (프록시 클래스, widget id, multiplier)."""
    source = (ROOT / 'ui' / 'generator_ui_setup.py').read_text(encoding='utf-8')
    return {
        key: (cls_name, wid, int(multiplier) if multiplier else None)
        for key, cls_name, wid, multiplier in _REGISTRATION_RE.findall(source)
    }


def _cn_widgets(bridge) -> dict:
    """generator_ui_setup.py 와 **같은 프록시 종류**로 13개를 만든다 (소스에서 읽어 어긋나지 않게)."""
    widgets = {}
    for key, (cls_name, wid, multiplier) in _registered_cn_proxies().items():
        cls = _PROXY_CLASSES[cls_name]
        widgets[key] = cls(bridge, wid, multiplier=multiplier) if multiplier else cls(bridge, wid)
    return widgets


class _AnyProxy:
    """_get/_set_sam3_settings 의 나머지(비 CN) 위젯용 범용 가짜."""

    def __init__(self):
        self.value = ''
        self.checked = False

    def text(self):
        return self.value

    def setText(self, value):
        self.value = value

    def toPlainText(self):
        return self.value

    def setPlainText(self, value):
        self.value = value

    def currentText(self):
        return self.value

    def isChecked(self):
        return self.checked

    def setChecked(self, value):
        self.checked = bool(value)


class _Widgets(dict):
    def __missing__(self, key):
        proxy = _AnyProxy()
        self[key] = proxy
        return proxy


def _build_sam3(cn_widgets: dict) -> dict:
    """실제 전송 경로 — GenerationMixin._build_sam3_settings 가 CN 프록시를 읽는 그대로."""
    from ui.generator_generation import GenerationMixin
    host = type('_Host', (), {})()
    host.sam3_widgets = _Widgets(cn_widgets)
    return GenerationMixin._build_sam3_settings(host, {})


class SpecTableTests(unittest.TestCase):
    def test_thirteen_fields_match_the_extension_spec(self):
        self.assertEqual(len(cn.CN_FIELDS), 13)
        spec_cn = [key for key in sam3_args.SAM3_KEYS if key.startswith('sam3_cn_')]
        self.assertEqual(sorted(cn.spec_key(key) for key in cn.CN_KEYS), sorted(spec_cn))

    def test_generator_ui_setup_registers_the_same_widget_ids(self):
        source = (ROOT / 'ui' / 'generator_ui_setup.py').read_text(encoding='utf-8')
        for key in cn.CN_KEYS:
            self.assertIn(f"'{key}':", source)
            self.assertIn(f"'{cn.widget_id(key)}'", source)
        # 기본값·선택지는 표 한 벌에서 — 손으로 쓴 기본값 줄이 되살아나지 않게
        self.assertIn('init_widgets', source)
        self.assertNotIn("['cn_module'].setText('inpaint_only')", source)

    def test_registered_proxy_kinds_match_the_table(self):
        """표의 종류와 실제 프록시가 같아야 한다 — 자유 입력(Model)을 ComboBoxProxy 로 두면
        빈 값(=None 으로 되돌리기)을 무시하고 숫자만인 이름을 인덱스로 읽는다."""
        registered = _registered_cn_proxies()
        self.assertEqual(set(registered), set(cn.CN_KEYS))
        allowed = {
            'check': {'CheckBoxProxy'},
            'combo': {'ComboBoxProxy'},
            'text': {'LineEditProxy', 'SliderProxy'},
        }
        for key, kind in cn.CN_FIELDS:
            cls_name, wid, _multiplier = registered[key]
            self.assertIn(cls_name, allowed[kind], key)
            self.assertEqual(wid, cn.widget_id(key), key)
        # 고정 선택지만 콤보 — 나머지 콤보는 자유 입력을 망가뜨린다. 전처리기는 라이브 목록(정적 목록에
        # 없는 이름 포함)에서 고르므로 자유 입력 + 정적 폴백 추천 목록(items)이다(P3).
        combos = {key for key, kind in cn.CN_FIELDS if kind == 'combo'}
        self.assertEqual(combos, set(cn.choice_items()) - {'cn_module'})
        self.assertEqual(registered['cn_model'][0], 'LineEditProxy')
        self.assertEqual(registered['cn_module'][0], 'LineEditProxy')

    def test_default_values_follow_sam3_spec(self):
        defaults = cn.default_values()
        self.assertIs(defaults['cn_enable'], False)
        self.assertIs(defaults['cn_pixel_perfect'], True)
        self.assertEqual(defaults['cn_module'], 'inpaint_only')
        self.assertEqual(float(defaults['cn_weight']), 1.0)
        self.assertEqual(float(defaults['cn_threshold_a']), -1.0)
        self.assertEqual(int(defaults['cn_processor_res']), 512)

    def test_choice_items_single_source(self):
        items = cn.choice_items()
        self.assertEqual(items['cn_module'], list(sam3_args.CN_MODULES))
        self.assertIn('ControlNet is more important', items['cn_control_mode'])
        self.assertIn('Crop and Resize', items['cn_resize_mode'])
        self.assertNotIn('cn_model', items)   # 설치 모델에 따라 달라 자유 입력


class WidgetInitAndPersistenceTests(unittest.TestCase):
    def setUp(self):
        self.bridge = _Bridge()
        self.widgets = _cn_widgets(self.bridge)
        cn.init_widgets(self.widgets)

    def test_init_pushes_items_then_defaults(self):
        self.assertEqual(self.bridge.properties[('_sam3_cn_module', 'items')], list(sam3_args.CN_MODULES))
        self.assertEqual(self.widgets['cn_module'].text(), 'inpaint_only')
        self.assertEqual(self.bridge.values['_sam3_cn_module'], 'inpaint_only')
        self.assertEqual(self.widgets['cn_resize_mode'].currentText(), 'Crop and Resize')
        self.assertEqual(self.bridge.values['_sam3_cn_resize_mode'], 'Crop and Resize')
        self.assertEqual(self.widgets['cn_control_mode'].currentText(), 'Balanced')
        self.assertEqual(self.widgets['cn_model'].text(), 'None')
        self.assertEqual(self.bridge.values['_sam3_cn_model'], 'None')
        self.assertTrue(self.widgets['cn_pixel_perfect'].isChecked())
        self.assertFalse(self.widgets['cn_enable'].isChecked())

    def test_vue_edits_round_trip_through_save_and_restore(self):
        w = self.widgets
        # Vue 에서 바뀐 값 (widget store 는 문자열을 보낸다)
        w['cn_enable']._on_vue_changed('true')
        w['cn_override_external']._on_vue_changed('true')
        w['cn_model']._on_vue_changed('control_v11p_sd15_inpaint')
        w['cn_module']._on_vue_changed('depth_anything')
        w['cn_weight']._on_vue_changed('0.65')
        w['cn_control_mode']._on_vue_changed('ControlNet is more important')
        w['cn_threshold_a']._on_vue_changed('64')
        saved = cn.read_settings(w)
        self.assertEqual(set(saved), set(cn.CN_KEYS))
        self.assertIs(saved['cn_enable'], True)
        self.assertEqual(saved['cn_module'], 'depth_anything')

        fresh_bridge = _Bridge()
        fresh = _cn_widgets(fresh_bridge)
        cn.init_widgets(fresh)          # 재시작 — 기본값
        cn.apply_settings(fresh, saved)  # 설정 복원
        self.assertEqual(cn.read_settings(fresh), saved)
        self.assertEqual(fresh_bridge.values['_sam3_cn_model'], 'control_v11p_sd15_inpaint')
        self.assertEqual(fresh_bridge.values['_sam3_cn_enable'], 'true')

    def test_old_settings_without_cn_keys_restore_defaults(self):
        self.widgets['cn_enable'].setChecked(True)
        self.widgets['cn_weight'].setText('1.7')
        cn.apply_settings(self.widgets, {'detect_prompt': 'face'})
        self.assertFalse(self.widgets['cn_enable'].isChecked())
        self.assertEqual(float(self.widgets['cn_weight'].text()), 1.0)

    def test_clearing_the_model_field_goes_back_to_none(self):
        """Model 칸을 비우면(placeholder 'None') 이전 모델이 남아 계속 전송·저장되면 안 된다."""
        w = self.widgets
        w['cn_model']._on_vue_changed('control_v11p_sd15_inpaint')
        self.assertEqual(_build_sam3(w)['sam3_cn_model'], 'control_v11p_sd15_inpaint')

        w['cn_model']._on_vue_changed('')
        self.assertEqual(w['cn_model'].text(), '')
        self.assertEqual(_build_sam3(w)['sam3_cn_model'], 'None')

        saved = cn.read_settings(w)
        self.assertEqual(saved['cn_model'], '')
        restored_bridge = _Bridge()
        restored = _cn_widgets(restored_bridge)
        cn.init_widgets(restored)
        cn.apply_settings(restored, saved)   # 재시작 후 복원 — 옛 모델이 되살아나지 않는다
        self.assertEqual(restored['cn_model'].text(), '')
        self.assertEqual(_build_sam3(restored)['sam3_cn_model'], 'None')

    def test_digit_only_model_name_is_text_not_an_index(self):
        self.widgets['cn_model']._on_vue_changed('12345')
        self.assertEqual(self.widgets['cn_model'].text(), '12345')
        self.assertEqual(_build_sam3(self.widgets)['sam3_cn_model'], '12345')
        self.assertEqual(cn.read_settings(self.widgets)['cn_model'], '12345')


class SettingsMixinIntegrationTests(unittest.TestCase):
    def test_get_and_set_sam3_settings_include_all_cn_keys(self):
        mixin = SettingsMixin()
        bridge = _Bridge()
        widgets = _Widgets(_cn_widgets(bridge))
        cn.init_widgets(widgets)
        widgets['cn_enable']._on_vue_changed('true')
        widgets['cn_resize_mode']._on_vue_changed('Just Resize')

        saved = mixin._get_sam3_settings(widgets)
        for key in cn.CN_KEYS:
            self.assertIn(key, saved)
        self.assertIs(saved['cn_enable'], True)
        self.assertEqual(saved['cn_resize_mode'], 'Just Resize')

        restored = _Widgets(_cn_widgets(_Bridge()))
        cn.init_widgets(restored)
        mixin._set_sam3_settings(restored, saved)
        self.assertTrue(restored['cn_enable'].isChecked())
        self.assertEqual(restored['cn_resize_mode'].currentText(), 'Just Resize')


class VueWiringTests(unittest.TestCase):
    def test_t2i_sam3_card_binds_the_panel_to_the_widget_store(self):
        # T2I 의 SAM3 카드는 App.vue 분할(④)로 components/params/Sam3MaskCard.vue 가 됐다.
        card = (SRC / 'components' / 'params' / 'Sam3MaskCard.vue').read_text(encoding='utf-8')
        self.assertRegex(card, r'<Sam3ControlNetPanel :widgets="storeWidgets"')
        self.assertIn("import Sam3ControlNetPanel from '../Sam3ControlNetPanel.vue'", card)
        # storeWidgets 는 위젯 스토어 그 자체여야 한다(예전 App 의 storeWidgets 와 같은 객체)
        self.assertIn("const storeWidgets = useWidgetStore().widgets", card)
        app = (SRC / 'App.vue').read_text(encoding='utf-8')
        self.assertIn("<Sam3MaskCard />", app)
        self.assertIn("import Sam3MaskCard from './components/params/Sam3MaskCard.vue'", app)

    def test_refine_and_batch_sam3_send_the_thirteen_fields(self):
        refine = (SRC / 'components' / 'RefinePanel.vue').read_text(encoding='utf-8')
        batch = (SRC / 'views' / 'BatchView.vue').read_text(encoding='utf-8')
        for source in (refine, batch):
            self.assertIn('<Sam3ControlNetPanel', source)
            self.assertIn('...sam3CnSettings(', source)

    def test_typescript_defaults_match_python(self):
        ts = (SRC / 'utils' / 'sam3ControlNet.ts').read_text(encoding='utf-8')
        entries = dict(
            (key, (kind, default)) for key, kind, default in re.findall(
                r"\{ key: '(cn_[a-z_]+)', kind: '(bool|number|text)', def: '([^']*)' \}", ts)
        )
        self.assertEqual(set(entries), set(cn.CN_KEYS))
        defaults = cn.default_values()
        kinds = dict(cn.CN_FIELDS)
        for key, (kind, default) in entries.items():
            expected = defaults[key]
            if kind == 'bool':
                self.assertEqual(kinds[key], 'check', key)
                self.assertEqual(default, 'true' if expected else 'false', key)
            elif kind == 'number':
                self.assertAlmostEqual(float(default), float(expected), msg=key)
            else:
                self.assertEqual(default, expected, key)

    def test_refine_panel_no_longer_duplicates_the_module_list(self):
        refine = (SRC / 'components' / 'RefinePanel.vue').read_text(encoding='utf-8')
        self.assertNotIn("'inpaint_only+lama'", refine)

    def test_widget_store_loads_the_property_snapshot_on_connect(self):
        store = (SRC / 'stores' / 'widgetStore.js').read_text(encoding='utf-8')
        self.assertIn('backend.getAllWidgetProperties(', store)


class LateClientSyncTests(unittest.TestCase):
    """선택지는 _setup_ui(Vue 로드 전)에 한 번 push 된다 — 실제 VueBridge 로, 나중에 붙은
    클라이언트가 받는 경로(getAllWidgetProperties)에 선택지가 있어야 한다.

    가짜 _Bridge 는 push 를 dict 에 적어 두지만 실제 브리지는 예전엔 emit 만 해서, 페이지가
    뜨기 전의 push 가 사라지고 세 드롭다운이 현재 값 하나만 보였다.
    """

    def setUp(self):
        from ui.vue_bridge import VueBridge
        self.bridge = VueBridge()
        self.widgets = _cn_widgets(self.bridge)
        cn.init_widgets(self.widgets)   # generator_ui_setup._init_settings_proxies 와 같은 시점

    def test_snapshot_carries_the_choices_pushed_before_any_client(self):
        snapshot = json.loads(self.bridge.getAllWidgetProperties())
        for key, items in cn.choice_items().items():
            self.assertEqual(snapshot[cn.widget_id(key)]['items'], items, key)
        self.assertNotIn('_sam3_cn_model', snapshot)   # 자유 입력 — 선택지 없음
        values = json.loads(self.bridge.getAllWidgetValues())
        self.assertEqual(values['_sam3_cn_module'], 'inpaint_only')
        self.assertEqual(values['_sam3_cn_model'], 'None')

    def test_snapshot_keeps_the_latest_push(self):
        received = []
        self.bridge.widgetPropertyChanged.connect(lambda *args: received.append(args))
        self.widgets['cn_resize_mode'].addItems(['Just Resize', 'Crop and Resize'])
        self.widgets['cn_resize_mode'].addItem('Resize and Fill')
        snapshot = json.loads(self.bridge.getAllWidgetProperties())
        self.assertEqual(snapshot['_sam3_cn_resize_mode']['items'],
                         ['Just Resize', 'Crop and Resize', 'Resize and Fill'])
        # 연결된 클라이언트에는 여전히 push 로 간다
        self.assertEqual(received[-1][0:2], ('_sam3_cn_resize_mode', 'items'))
        self.assertEqual(json.loads(received[-1][2]), ['Just Resize', 'Crop and Resize', 'Resize and Fill'])

    def test_free_text_suggestions_also_reach_the_snapshot(self):
        """전처리기는 자유 입력(LineEditProxy)이지만 정적 폴백 목록은 같은 'items' 로 늦은 클라이언트에 간다."""
        self.widgets['cn_module'].setSuggestions(['None', 'inpaint_noobai'])
        snapshot = json.loads(self.bridge.getAllWidgetProperties())
        self.assertEqual(snapshot['_sam3_cn_module']['items'], ['None', 'inpaint_noobai'])

    def test_web_clients_can_read_the_snapshot(self):
        try:
            import web_main_ui
        except ModuleNotFoundError as exc:   # QtWebEngine 없는 환경
            self.skipTest(str(exc))
        self.assertIn('getAllWidgetProperties', web_main_ui._WEB_METHODS)
        facade = web_main_ui.WebBridgeFacade(self.bridge)
        reply = json.loads(facade.invoke('getAllWidgetProperties', '[]'))
        self.assertTrue(reply['ok'])
        snapshot = json.loads(reply['value'])
        self.assertEqual(snapshot['_sam3_cn_module']['items'], list(sam3_args.CN_MODULES))


if __name__ == '__main__':
    unittest.main()
