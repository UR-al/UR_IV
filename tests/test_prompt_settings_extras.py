"""core.prompt_settings_extras — 숨은 SettingsTab 을 대신하는 prompt_settings 키 보관함 (audit #178)."""
from __future__ import annotations

import json
import unittest
from types import SimpleNamespace

from core.prompt_cleaner_prefs import LEGACY_ONLY_DEFAULTS
from core.prompt_settings_extras import (
    DEFAULT_FONT_FAMILY_NAME,
    DEFAULT_FONT_SIZE_PT,
    PromptSettingsExtras,
    RETIRED_KEYS,
    extras_of,
)


class FromSettingsTests(unittest.TestCase):
    def test_missing_keys_use_the_old_settings_tab_defaults(self):
        extras = PromptSettingsExtras.from_settings({})
        self.assertTrue(extras.wildcard_enabled)
        self.assertEqual(extras.cleaning, dict(LEGACY_ONLY_DEFAULTS))
        self.assertEqual((extras.font_family, extras.font_size), (DEFAULT_FONT_FAMILY_NAME, DEFAULT_FONT_SIZE_PT))
        self.assertEqual(PromptSettingsExtras.from_settings(None), PromptSettingsExtras())

    def test_saved_values_are_read(self):
        extras = PromptSettingsExtras.from_settings({
            'wildcard_enabled': False,
            'cleaning_options': {'auto_comma': False, 'auto_escape': True, 'remove_duplicates': True},
            'font_family': 'Noto Sans KR', 'font_size': 12.5,
        })
        self.assertFalse(extras.wildcard_enabled)
        # Vue 토글이 있는 세 옵션은 ui_prefs 가 주인 — 여기엔 레거시 두 옵션만
        self.assertEqual(extras.cleaning, {'auto_comma': False, 'auto_escape': True})
        self.assertEqual((extras.font_family, extras.font_size), ('Noto Sans KR', 12.5))

    def test_malformed_values_fall_back_instead_of_breaking_the_load(self):
        extras = PromptSettingsExtras.from_settings({
            'wildcard_enabled': 'no', 'cleaning_options': 'bad', 'font_family': '   ', 'font_size': 'huge',
        })
        self.assertEqual(extras, PromptSettingsExtras())
        self.assertEqual(PromptSettingsExtras.from_settings({'font_size': float('nan')}).font_size,
                         DEFAULT_FONT_SIZE_PT)

    def test_font_size_is_clamped_to_the_old_spin_box_range(self):
        self.assertEqual(PromptSettingsExtras.from_settings({'font_size': 3}).font_size, 8.0)
        self.assertEqual(PromptSettingsExtras.from_settings({'font_size': '40'}).font_size, 20.0)
        self.assertEqual(PromptSettingsExtras.from_settings({'font_size': True}).font_size, DEFAULT_FONT_SIZE_PT)

    def test_retired_keys_are_ignored_and_never_written(self):
        saved = {'theme': '다크', 'parquet_dir': r'C:\old install\danbooru_optimized',
                 'event_parquet_dir': r'C:\old install\danbooru_optimized\danbooru_sorted',
                 'cond_prompt_enabled': True, 'prefix_toggle': False, 'res_presets': [['x', 1, 2]],
                 'editor_defaults': {'tool_size': 5}, 'bg_removal_model': 'isnet',
                 'shortcuts': {'undo': 'Ctrl+Z'}}
        self.assertTrue(set(saved) <= set(RETIRED_KEYS))
        self.assertEqual(PromptSettingsExtras.from_settings(saved), PromptSettingsExtras())
        self.assertFalse(set(PromptSettingsExtras().to_settings()) & set(RETIRED_KEYS))


class RoundTripTests(unittest.TestCase):
    def test_to_settings_round_trips_through_json(self):
        extras = PromptSettingsExtras(
            wildcard_enabled=False, cleaning={'auto_comma': False, 'auto_escape': True},
            font_family='Malgun Gothic', font_size=9.5,
        )
        written = json.loads(json.dumps(extras.to_settings()))
        self.assertEqual(set(written), {'wildcard_enabled', 'cleaning_options', 'font_family', 'font_size'})
        self.assertEqual(PromptSettingsExtras.from_settings(written), extras)

    def test_to_settings_normalizes_values_changed_at_runtime(self):
        extras = PromptSettingsExtras()
        extras.wildcard_enabled = 0
        extras.font_size = 99
        extras.cleaning = {'auto_comma': 0, 'extra': True}
        written = extras.to_settings()
        self.assertIs(written['wildcard_enabled'], False)
        self.assertEqual(written['font_size'], 20.0)
        self.assertEqual(written['cleaning_options'], {'auto_comma': False, 'auto_escape': False})


class ExtrasOfTests(unittest.TestCase):
    def test_returns_the_owner_store_or_a_default(self):
        store = PromptSettingsExtras(wildcard_enabled=False)
        self.assertIs(extras_of(SimpleNamespace(prompt_settings_extras=store)), store)
        self.assertEqual(extras_of(SimpleNamespace()), PromptSettingsExtras())
        self.assertEqual(extras_of(SimpleNamespace(prompt_settings_extras=object())), PromptSettingsExtras())


class FontDefaultsMatchThemeManagerTests(unittest.TestCase):
    def test_defaults_match_the_theme_manager(self):
        from utils.theme_manager import DEFAULT_FONT_FAMILY, DEFAULT_FONT_SIZE, ThemeManager

        tm = ThemeManager()
        self.assertEqual(tm.get_font_family_name(), DEFAULT_FONT_FAMILY_NAME)
        self.assertEqual(tm.get_font_size_value(), DEFAULT_FONT_SIZE_PT)
        self.assertTrue(DEFAULT_FONT_FAMILY.startswith(f"'{DEFAULT_FONT_FAMILY_NAME}'"))
        self.assertEqual(DEFAULT_FONT_SIZE, f'{DEFAULT_FONT_SIZE_PT}pt')


if __name__ == '__main__':
    unittest.main()
