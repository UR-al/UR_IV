"""회귀 방지: 숨은 SettingsTab·만능 더미·PyQt LoRA 매니저가 다시 생기지 않는다 (audit #175·#177·#178).

- 숨은 SettingsTab(위젯 약 200개)은 prompt_settings 키 몇 개의 보관함일 뿐이었다 — 그 키는
  core/prompt_settings_extras 가 맡고, 은퇴한 키(theme·parquet_dir 등)는 읽지도 쓰지도 않는다.
- 어떤 속성이든 no-op 을 돌려주는 ``__getattr__`` 더미(_D·_VLP)는 hasattr 가드를 늘 참으로 만들어
  죽은 분기를 숨겼다. 더미 자리의 호출은 지웠고, 상태 문구는 show_status·vue_bridge 로 간다.
- 호출자가 없던 PyQt LoRA 매니저(LoraManagerDialog)·open_lora_manager·loraInserted 는 지웠고,
  LoRA raw 캐시는 ui/lora_catalog_cache 한 곳에 있다.
"""
from __future__ import annotations

import ast
import io
import json
import re
import tempfile
import tokenize
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]


def _read(rel: str) -> str:
    return (ROOT / rel).read_text(encoding="utf-8")


def _ui_sources() -> dict[str, str]:
    return {p.name: p.read_text(encoding="utf-8") for p in sorted((ROOT / "ui").glob("*.py"))}


def _code_names(text: str) -> set[str]:
    """주석·문자열을 뺀 코드의 이름(식별자)과 속성 문자열 리터럴 — 은퇴 사유를 적은 주석은 걸리지 않게."""
    names: set[str] = set()
    for tok in tokenize.generate_tokens(io.StringIO(text).readline):
        if tok.type == tokenize.NAME:
            names.add(tok.string)
        elif tok.type == tokenize.STRING:
            inner = tok.string.lstrip('rbuRBUfF').strip('\'"')
            if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", inner):
                names.add(inner)   # getattr(self, 'viewer_label') 같은 이름 문자열
    return names


class SettingsTabRetirementTests(unittest.TestCase):
    def test_settings_tab_and_its_dedicated_widgets_are_gone(self):
        for rel in ("tabs/settings_tab.py", "widgets/condition_block_editor.py", "widgets/font_combo.py"):
            self.assertFalse((ROOT / rel).exists(), rel)

    def test_nothing_reads_the_hidden_settings_tab(self):
        pattern = re.compile(r"\.settings_tab\b|['\"]settings_tab['\"]|import SettingsTab|cond_prompt_check|"
                             r"cond_block_editor_(pos|neg)")
        sources = {**_ui_sources(), "file_wildcard.py": _read("utils/file_wildcard.py")}
        offenders = [name for name, text in sources.items() if pattern.search(text)]
        self.assertEqual(offenders, [])

    def test_load_ignores_retired_keys_and_save_never_writes_them(self):
        """옛 prompt_settings.json(다른 설치본 백업 포함)의 parquet_dir 가 데이터셋 경로를 덮지 않는다."""
        import config
        from core.fetch_data import DATA_DIR
        from core.prompt_settings_extras import PromptSettingsExtras, RETIRED_KEYS
        from ui.generator_settings import SettingsMixin

        class _Host(SettingsMixin):
            def __getattr__(self, name):   # 이 테스트가 보지 않는 위젯은 MagicMock
                if name.startswith('__'):
                    raise AttributeError(name)
                value = mock.MagicMock(name=name)
                setattr(self, name, value)
                return value

        self.assertEqual(config.PARQUET_DIR, str(DATA_DIR), '데이터셋 경로의 단일 출처')
        self.assertEqual(Path(config.EVENT_PARQUET_DIR), DATA_DIR / 'danbooru_sorted')
        saved = {
            'parquet_dir': r'C:\moved install\danbooru_optimized',
            'event_parquet_dir': r'C:\moved install\danbooru_optimized\danbooru_sorted',
            'theme': '다크', 'cond_prompt_enabled': True, 'prefix_toggle': False, 'res_presets': [['x', 1, 2]],
            'wildcard_enabled': False, 'cleaning_options': {'auto_comma': False, 'auto_escape': True},
            'font_family': 'Noto Sans KR', 'font_size': 12.0,
        }
        before = (config.PARQUET_DIR, config.EVENT_PARQUET_DIR)
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'prompt_settings.json'
            path.write_text(json.dumps(saved), encoding='utf-8')
            host = _Host()
            with mock.patch('ui.generator_settings.PROMPT_SETTINGS_FILE', str(path)), \
                    mock.patch('ui.generator_settings.migrate_legacy_gallery_folder', return_value=''), \
                    mock.patch('ui.generation_settings_apply.apply_generation_settings', return_value=[]), \
                    mock.patch('utils.theme_manager.get_theme_manager') as tm:
                SettingsMixin.load_settings(host)
        self.assertEqual((config.PARQUET_DIR, config.EVENT_PARQUET_DIR), before)
        self.assertEqual(host.prompt_settings_extras, PromptSettingsExtras(
            wildcard_enabled=False, cleaning={'auto_comma': False, 'auto_escape': True},
            font_family='Noto Sans KR', font_size=12.0))
        tm.return_value.set_font.assert_called_once_with('Noto Sans KR', 12.0)
        host.prompt_cleaner.set_options.assert_called_with(auto_comma=False, auto_escape=True)
        host.wildcard_enabled_check.setChecked.assert_called_with(False)

        written = SettingsMixin._build_settings_dict(host)
        self.assertFalse(set(written) & set(RETIRED_KEYS), sorted(set(written) & set(RETIRED_KEYS)))
        self.assertIs(written['wildcard_enabled'], False)
        self.assertEqual(written['cleaning_options'], {'auto_comma': False, 'auto_escape': True})
        self.assertEqual((written['font_family'], written['font_size']), ('Noto Sans KR', 12.0))

    def test_legacy_theme_name_api_is_gone(self):
        import utils.theme_manager as tm
        for name in ('LEGACY_THEME_NAME', 'THEMES', 'MODERN_THEME'):
            self.assertFalse(hasattr(tm, name), name)
        for name in ('current_theme_name', 'available_themes'):
            self.assertFalse(hasattr(tm.ThemeManager, name), name)


class BlanketDummyRetirementTests(unittest.TestCase):
    def test_no_getattr_catch_all_dummies_in_ui(self):
        offenders = []
        for name, text in _ui_sources().items():
            tree = ast.parse(text)
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef) and node.name == '__getattr__':
                    offenders.append(f"{name}:{node.lineno} def __getattr__")
                if isinstance(node, ast.Dict):
                    for key in node.keys:
                        if isinstance(key, ast.Constant) and key.value == '__getattr__':
                            offenders.append(f"{name}:{node.lineno} {{'__getattr__': ...}}")
        self.assertEqual(offenders, [])

    def test_dead_placeholders_do_not_come_back(self):
        names = ('_AlwaysOn', 'LoraProxy', '_get_tab_title', 'btn_api_manager', 'lora_active_panel',
                 'viewer_label', 'gen_progress_bar', 'center_tabs', 'token_count_label', 'fav_tags_bar',
                 'ad_settings_container', 'sam3_settings_container', 'inpaint_size_container',
                 'sampler_container', 'resolution_list_widget', 'toggle_random_resolution_editor',
                 '_res_preset_btns', '_auto_adjust_text_edit_height')
        found = [f"{file}: {n}" for file, text in _ui_sources().items()
                 for n in sorted(set(names) & _code_names(text))]
        self.assertEqual(found, [])
        widget_names = _code_names(_read('widgets/common_widgets.py'))
        # 사용처(settings_tab·batch_tab·mosaic_panel·event_gen_tab·gallery_tab)가 은퇴한 위젯과
        # 그 위젯만 쓰던 Qt import 도 남기지 않는다. NoScrollComboBox 는 inpaint_tab 이 쓴다.
        retired = {'ResolutionItemWidget', 'NoScrollDoubleSpinBox', 'NoScrollSpinBox', 'FlowLayout',
                   'QSpinBox', 'QLayout', 'QRect', 'QSize'}
        self.assertEqual(sorted(retired & widget_names), [])
        self.assertIn('NoScrollComboBox', widget_names)

    def test_proxies_have_no_noop_visibility_or_fake_document(self):
        from ui.widget_proxies import SliderProxy, TextEditProxy, _ProxyBase
        for cls in (_ProxyBase, SliderProxy, TextEditProxy):
            for name in ('setVisible', 'hide', 'show'):
                self.assertFalse(hasattr(cls, name), f'{cls.__name__}.{name}')
        self.assertFalse(hasattr(TextEditProxy, 'document'))


class LoraManagerDialogRetirementTests(unittest.TestCase):
    def test_dialog_chain_and_signal_are_gone(self):
        self.assertFalse((ROOT / 'widgets' / 'lora_manager.py').exists())
        for rel in ('ui/generator_main.py', 'ui/generator_ui_setup.py', 'ui/vue_bridge.py', 'web_main_ui.py',
                    'core/web_action_policy.py', 'ui/lora_catalog_cache.py', 'frontend/src/App.vue',
                    'frontend/src/types/bridge.d.ts', 'frontend/src/utils/hostDialogs.ts'):
            text = _read(rel)
            with self.subTest(rel=rel):
                self.assertNotIn('loraInserted', text)
                self.assertNotIn('open_lora_manager', text)
                self.assertIsNone(re.search(r'from widgets\.lora_manager import|_on_lora_(batch_)?inserted', text))

    def test_raw_cache_lives_in_lora_catalog_cache(self):
        from ui import lora_catalog_cache

        previous = lora_catalog_cache.raw_loras()
        self.addCleanup(setattr, lora_catalog_cache, '_raw_loras', previous)
        lora_catalog_cache._raw_loras = [{'name': 'x'}]
        bridge = mock.Mock(_merged_lora_cache={'json': '[]'})
        lora_catalog_cache.invalidate(bridge)
        self.assertEqual(lora_catalog_cache.raw_loras(), [])
        self.assertIsNone(bridge._merged_lora_cache)


if __name__ == '__main__':
    unittest.main()
