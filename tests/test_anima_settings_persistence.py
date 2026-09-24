"""ANIMA 설정 저장/복원 회귀 테스트."""

from pathlib import Path
import unittest

from core.anima_guidance import default_settings
from ui.generator_settings import SettingsMixin


class _Proxy:
    def __init__(self, value=''):
        self.value = str(value)

    def text(self):
        return self.value

    def setText(self, value):
        self.value = value


class TestAnimaSettingsPersistence(unittest.TestCase):
    def setUp(self):
        self.mixin = SettingsMixin()

    def test_collects_every_anima_widget_value(self):
        widgets = {
            key: _Proxy(f'value-{index}')
            for index, key in enumerate(default_settings())
        }
        saved = self.mixin._get_anima_guidance_settings(widgets)
        self.assertEqual(len(saved), 82)
        self.assertEqual(saved['guid_smc_preset'], widgets['guid_smc_preset'].text())
        self.assertEqual(saved['guid_rdc_tau'], widgets['guid_rdc_tau'].text())

    def test_old_saved_settings_gain_new_forge_defaults(self):
        widgets = {key: _Proxy('stale') for key in default_settings()}
        self.mixin._set_anima_guidance_settings(widgets, {'guid_enabled': True})
        self.assertEqual(widgets['guid_enabled'].text(), 'true')
        self.assertEqual(widgets['guid_smc_preset'].text(), 'Auto')
        self.assertEqual(widgets['guid_smc_master_enabled'].text(), 'false')
        self.assertEqual(widgets['guid_rdc_tau'].text(), '0.15')
        self.assertEqual(widgets['guid_rdc_alpha_ll'].text(), '0.03')

    def test_reset_restores_every_spec_default_in_one_batch(self):
        """Vue '전체 초기화' — 기본값은 스펙(default_settings) 한 곳에서만 온다(#99)."""
        calls = []

        class _Signal:
            def emit(self, *args):
                calls.append(('notify',) + args)

        class _Bridge:
            showNotification = _Signal()

            def beginBatchUpdate(self):
                calls.append(('begin',))

            def endBatchUpdate(self):
                calls.append(('end',))

        class _Proxy2(_Proxy):
            def setText(self, value):
                calls.append(('set', value))
                super().setText(value)

        defaults = default_settings()
        widgets = {key: _Proxy2('user-value') for key in defaults}
        self.mixin.anima_guidance_widgets = widgets
        self.mixin.vue_bridge = _Bridge()

        self.assertEqual(self.mixin._reset_anima_guidance(), 82)
        for key, value in defaults.items():
            expected = ('true' if value else 'false') if isinstance(value, bool) else str(value)
            self.assertEqual(widgets[key].text(), expected, key)
        # 82개 값이 배치 안에서 한 번에 Vue 로 간다
        kinds = [c[0] for c in calls]
        self.assertEqual(kinds[0], 'begin')
        self.assertEqual(kinds.index('end'), 1 + kinds.count('set'))
        self.assertEqual(kinds[-1], 'notify')

    def test_reset_without_widgets_is_a_no_op(self):
        self.mixin.anima_guidance_widgets = {}
        self.assertEqual(self.mixin._reset_anima_guidance(), 0)

    def test_vue_panel_keeps_no_copy_of_the_defaults(self):
        panel = (Path(__file__).resolve().parents[1] / 'frontend' / 'src' / 'components'
                 / 'AnimaGuidancePanel.vue').read_text(encoding='utf-8')
        self.assertIn("requestAction('reset_anima_guidance')", panel)
        self.assertNotRegex(panel, r"const\s+DEFAULTS\b")
        main = (Path(__file__).resolve().parents[1] / 'ui' / 'generator_main.py').read_text(encoding='utf-8')
        self.assertIn("action == 'reset_anima_guidance'", main)

    def test_save_and_load_paths_include_anima_settings(self):
        root = Path(__file__).resolve().parents[1] / 'ui'
        # 저장은 generator_settings._build_settings_dict, 복원은 load_settings 와 프리셋 불러오기가
        # 함께 쓰는 ui/generation_settings_apply.apply_generation_settings(audit #154).
        save_source = (root / 'generator_settings.py').read_text(encoding='utf-8')
        load_source = (root / 'generation_settings_apply.py').read_text(encoding='utf-8')
        self.assertGreaterEqual(save_source.count('"anima_guidance_settings"'), 1)
        self.assertGreaterEqual(load_source.count("'anima_guidance_settings'"), 2)
        self.assertIn('apply_generation_settings(self, settings', save_source)
        from core.generation_presets import PRESET_KEYS
        self.assertIn('anima_guidance_settings', PRESET_KEYS)


if __name__ == '__main__':
    unittest.main()
