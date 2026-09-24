"""프리셋 값을 지금 백엔드 콤보 항목에 맞추기(core.preset_choice)와 TE 선택 맞추기(audit #154 후속)."""
import unittest

from core.comfy_workflow_compiler import _FORGE_SAMPLER_ALIASES, _FORGE_SCHEDULER_ALIASES
from core.main_module_choices import resolve_te_selection
from core.preset_choice import (
    fold_choice,
    match_choice,
    match_sampler,
    match_scheduler,
    sampler_key,
    scheduler_key,
)

COMFY_SAMPLERS = ['euler', 'euler_ancestral', 'dpmpp_2m', 'dpmpp_2m_sde', 'uni_pc', 'er_sde']
FORGE_SAMPLERS = ['Euler', 'Euler a', 'DPM++ 2M', 'DPM++ 2M SDE', 'UniPC', 'ER SDE']


class MatchChoiceTests(unittest.TestCase):
    def test_exact_then_case_and_separator_insensitive(self):
        self.assertEqual(match_choice('Euler', ['euler', 'Euler']), 1, '정확히 같은 항목이 먼저')
        self.assertEqual(match_choice('SGM Uniform', ['normal', 'sgm_uniform']), 1)
        self.assertEqual(fold_choice('  DPM++_2M   Karras '), 'dpm++ 2m karras')

    def test_empty_value_or_list_never_matches(self):
        self.assertEqual(match_choice('', ['a']), -1)
        self.assertEqual(match_choice('a', []), -1)
        self.assertEqual(match_choice(None, None), -1)

    def test_no_alias_without_key(self):
        self.assertEqual(match_choice('Euler a', COMFY_SAMPLERS), -1)


class SamplerAliasTests(unittest.TestCase):
    def test_forge_labels_pick_comfy_names_and_back(self):
        for forge, comfy in zip(FORGE_SAMPLERS, COMFY_SAMPLERS):
            with self.subTest(forge=forge):
                self.assertEqual(COMFY_SAMPLERS[match_sampler(forge, COMFY_SAMPLERS)], comfy)
                self.assertEqual(FORGE_SAMPLERS[match_sampler(comfy, FORGE_SAMPLERS)], forge)

    def test_combined_label_is_not_split_so_the_scheduler_is_not_lost(self):
        self.assertEqual(match_sampler('DPM++ 2M Karras', COMFY_SAMPLERS), -1)

    def test_unknown_sampler_is_a_miss(self):
        self.assertEqual(match_sampler('Restart', COMFY_SAMPLERS), -1)

    def test_placeholders_are_only_matched_exactly(self):
        items = ['Use same sampler', 'euler']
        self.assertEqual(match_sampler('Use same sampler', items), 0)
        self.assertEqual(sampler_key('Use same sampler'), '')
        self.assertEqual(match_sampler('Restart', ['Use same sampler']), -1)

    def test_keys_agree_with_the_comfy_compiler_tables(self):
        """ComfyUI 생성이 Forge 표기를 바꾸는 표와 같은 결과 — 두 곳이 갈라지지 않는다."""
        for forge, comfy in _FORGE_SAMPLER_ALIASES.items():
            self.assertEqual(sampler_key(forge), sampler_key(comfy), forge)
        for forge, comfy in _FORGE_SCHEDULER_ALIASES.items():
            self.assertEqual(scheduler_key(forge), scheduler_key(comfy), forge)


class SchedulerAliasTests(unittest.TestCase):
    def test_forge_schedulers_pick_comfy_names(self):
        comfy = ['normal', 'karras', 'exponential', 'sgm_uniform', 'beta57']
        self.assertEqual(match_scheduler('Karras', comfy), 1)
        self.assertEqual(match_scheduler('SGM Uniform', comfy), 3)
        self.assertEqual(match_scheduler('Beta57 (RES4LYF)', comfy), 4)
        self.assertEqual(match_scheduler('karras', ['Automatic', 'Karras']), 1)

    def test_automatic_is_not_silently_turned_into_normal(self):
        self.assertEqual(match_scheduler('Automatic', ['normal', 'karras']), -1)

    def test_use_same_scheduler_is_not_an_alias_target(self):
        self.assertEqual(match_scheduler('Automatic', ['Use same scheduler', 'normal']), -1)
        self.assertEqual(match_scheduler('Use same scheduler', ['Use same scheduler', 'normal']), 0)


class ResolveTeSelectionTests(unittest.TestCase):
    def test_same_file_with_other_folder_or_hash_uses_the_choice_spelling(self):
        text, dropped = resolve_te_selection(
            'qwen_3_06b.safetensors [abc123], CLIP_L.safetensors, gone.safetensors',
            ['text_encoders/qwen_3_06b.safetensors', 'clip_l.safetensors'])
        self.assertEqual(text, 'text_encoders/qwen_3_06b.safetensors, clip_l.safetensors')
        self.assertEqual(dropped, ['gone.safetensors'])

    def test_duplicates_collapse_and_empty_choices_drop_everything(self):
        self.assertEqual(resolve_te_selection('a.safetensors, a.safetensors', ['a.safetensors']),
                         ('a.safetensors', []))
        self.assertEqual(resolve_te_selection('a.safetensors', []), ('', ['a.safetensors']))
        self.assertEqual(resolve_te_selection('', ['a.safetensors']), ('', []))


if __name__ == '__main__':
    unittest.main()
