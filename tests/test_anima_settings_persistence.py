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
        # 62 + 7 + Detail Daemon 11 (14칸 중 자리만 남은 preset·multiplier·cfg_couple 은 설정이 아니다)
        self.assertEqual(len(saved), 80)
        self.assertEqual(saved['guid_smc_preset'], widgets['guid_smc_preset'].text())
        self.assertEqual(saved['guid_rdc_tau'], widgets['guid_rdc_tau'].text())

    def test_old_saved_settings_gain_new_forge_defaults(self):
        widgets = {key: _Proxy('stale') for key in default_settings()}
        self.mixin._set_anima_guidance_settings(widgets, {'guid_enabled': True})
        self.assertEqual(widgets['guid_enabled'].text(), 'true')
        self.assertEqual(widgets['guid_smc_preset'].text(), 'Auto')
        self.assertEqual(widgets['guid_smc_master_enabled'].text(), 'false')
        # 저장에 없던 칸은 새 기본값(원본 노드 값)을 받는다 — RDC tau 0 = 끔
        # (origin: namemechan/ComfyUI-DCW@66aaf9dd:dcw_node.py:761-765 — 숫자만)
        self.assertEqual(widgets['guid_rdc_tau'].text(), '0.0')
        self.assertEqual(widgets['guid_rdc_alpha_ll'].text(), '0.03')
        self.assertEqual(widgets['guid_cns_gamma_scale'].text(), '2.0')
        self.assertEqual(widgets['guid_dcw_lambda_low'].text(), '0.05')

    def test_saved_values_are_loaded_untouched(self):
        """새 기본값(원본 노드)은 저장에 없는 칸에만 — 예전 기본값을 그대로 저장해 둔 값은 바꾸지 않는다."""
        old_saved = {'guid_rdc_enabled': 'false', 'guid_rdc_tau': '0.15', 'guid_dcw_lambda_low': '0.1',
                     'guid_dcw_lambda_high': '0.02', 'guid_cwm_alpha_low': '0.3',
                     'guid_cwm_alpha_high': '0.15', 'guid_cns_gamma_scale': '3.0'}
        widgets = {key: _Proxy('stale') for key in default_settings()}
        self.mixin._set_anima_guidance_settings(widgets, dict(old_saved))
        for key, value in old_saved.items():
            self.assertEqual(widgets[key].text(), value, key)
        self.assertEqual(self.mixin._get_anima_guidance_settings(widgets)['guid_rdc_tau'], '0.15')

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

        self.assertEqual(self.mixin._reset_anima_guidance(), 80)
        for key, value in defaults.items():
            expected = ('true' if value else 'false') if isinstance(value, bool) else str(value)
            self.assertEqual(widgets[key].text(), expected, key)
        # 80개 값이 배치 안에서 한 번에 Vue 로 간다
        kinds = [c[0] for c in calls]
        self.assertEqual(kinds[0], 'begin')
        self.assertEqual(kinds.index('end'), 1 + kinds.count('set'))
        self.assertEqual(kinds[-1], 'notify')

    def test_reset_without_widgets_is_a_no_op(self):
        self.mixin.anima_guidance_widgets = {}
        self.assertEqual(self.mixin._reset_anima_guidance(), 0)

    def test_vue_panel_keeps_no_copy_of_the_defaults(self):
        components = Path(__file__).resolve().parents[1] / 'frontend' / 'src' / 'components'
        panel_path = components / 'AnimaGuidancePanel.vue'
        panel = panel_path.read_text(encoding='utf-8')
        self.assertIn("requestAction('reset_anima_guidance')", panel)
        # 칸은 기능별 섹션(components/guidance/*Section.vue, P0-B 분할)에 있다 — 섹션·공용 헬퍼에도 사본 금지
        guidance = components / 'guidance'
        sections = sorted(guidance.glob('*.vue'))
        self.assertTrue(sections, guidance)
        helpers = sorted(p for p in guidance.glob('*.ts') if not p.name.endswith('.test.ts'))
        for path in [panel_path] + sections + helpers:
            with self.subTest(file=path.name):
                self.assertNotRegex(path.read_text(encoding='utf-8'), r"const\s+DEFAULTS\b")
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
