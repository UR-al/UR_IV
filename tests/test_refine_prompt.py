"""Refine 프롬프트 수술 테스트.

핵심 요구: Target 토큰이 든 **콤마 세그먼트를 통째로** 지워야 한다.
substring replace로 하면 "white shirt"에서 'shirt'만 빠져 "white"가 고아로 남는다.
확장 sam3ext/ui_refine.py 와 동일한 결과가 나와야 앱/Forge UI 출력이 일치한다.
"""
import unittest

from core.refine_prompt import (
    apply_prompt_sr,
    build_refine_prompts,
    normalize_prompt,
    parse_detect_tokens,
    strip_patterns_with_replacement,
)


class TestNormalize(unittest.TestCase):
    def test_collapses_whitespace_and_commas(self):
        self.assertEqual(normalize_prompt('a,  b,,,  c'), 'a, b, c')

    def test_trims_edges(self):
        self.assertEqual(normalize_prompt(' , a, b , '), 'a, b')

    def test_empty(self):
        self.assertEqual(normalize_prompt(''), '')
        self.assertEqual(normalize_prompt(None), '')


class TestParseTokens(unittest.TestCase):
    def test_splits_on_sam3_separators(self):
        self.assertEqual(parse_detect_tokens('face, eyes / hand; hair\nfoot'),
                         ['face', 'eyes', 'hand', 'hair', 'foot'])

    def test_empty(self):
        self.assertEqual(parse_detect_tokens(''), [])
        self.assertEqual(parse_detect_tokens('  ,  '), [])


class TestStripPatterns(unittest.TestCase):
    def test_removes_whole_segment_not_substring(self):
        # 핵심 회귀 — 'white'가 고아로 남으면 안 된다
        out = strip_patterns_with_replacement(
            '1boy, white shirt, black necktie', ['shirt'], 'nude')
        self.assertEqual(out, '1boy, nude, black necktie')
        self.assertNotIn('white', out)

    def test_replacement_inserted_once_for_multiple_matches(self):
        out = strip_patterns_with_replacement(
            '1boy, white shirt, black necktie, belt', ['shirt', 'necktie', 'belt'], 'nude')
        self.assertEqual(out, '1boy, nude')
        self.assertEqual(out.count('nude'), 1)

    def test_replacement_at_first_match_position(self):
        out = strip_patterns_with_replacement(
            'a, shirt, b, necktie, c', ['shirt', 'necktie'], 'X')
        self.assertEqual(out, 'a, X, b, c')

    def test_empty_replacement_deletes_only(self):
        out = strip_patterns_with_replacement('a, shirt, b', ['shirt'], '')
        self.assertEqual(out, 'a, b')

    def test_no_match_returns_original_untouched(self):
        original = 'a,  b,,  c'
        self.assertEqual(strip_patterns_with_replacement(original, ['zzz'], 'X'), original)

    def test_no_patterns_returns_original(self):
        self.assertEqual(strip_patterns_with_replacement('a, b', [], 'X'), 'a, b')

    def test_preserves_lora_and_score_tags(self):
        out = strip_patterns_with_replacement(
            '1boy, solo, white shirt, score_9, <lora:detail:0.8>', ['shirt'], 'nude')
        self.assertEqual(out, '1boy, solo, nude, score_9, <lora:detail:0.8>')


class TestApplyPromptSR(unittest.TestCase):
    def test_single_rule(self):
        self.assertEqual(apply_prompt_sr('a, shirt, b', 'shirt = nude'), 'a, nude, b')

    def test_multi_pattern_rule_collapses(self):
        out = apply_prompt_sr('1boy, white shirt, black necktie, belt',
                              'shirt, necktie, belt = nude')
        self.assertEqual(out, '1boy, nude')

    def test_multiple_rule_lines(self):
        out = apply_prompt_sr('a, shirt, hat, b', 'shirt = nude\nhat = crown')
        self.assertEqual(out, 'a, nude, crown, b')

    def test_lines_without_equals_ignored(self):
        self.assertEqual(apply_prompt_sr('a, b', 'no equals here'), 'a, b')

    def test_no_match_returns_original(self):
        self.assertEqual(apply_prompt_sr('a, b', 'zzz = X'), 'a, b')

    def test_empty_inputs(self):
        self.assertEqual(apply_prompt_sr('', 'a = b'), '')
        self.assertEqual(apply_prompt_sr('a', ''), 'a')


class TestBuildRefinePrompts(unittest.TestCase):
    MAIN = '1boy, solo, white shirt, black necktie, belt, score_9, <lora:detailedAnatomy:0.8>'

    def test_readme_example(self):
        # README 워크플로 2 예시와 동일한 결과가 나와야 한다
        out = build_refine_prompts(
            main_prompt=self.MAIN,
            target='shirt, necktie, belt',
            replacement='nude',
        )
        self.assertEqual(out['prompt'],
                         '1boy, solo, nude, score_9, <lora:detailedAnatomy:0.8>')

    def test_lora_preserved(self):
        out = build_refine_prompts(main_prompt=self.MAIN, target='shirt', replacement='nude')
        self.assertIn('<lora:detailedAnatomy:0.8>', out['prompt'])

    def test_negative_does_not_receive_replacement(self):
        out = build_refine_prompts(
            main_prompt='a, shirt', main_negative='bad hands, shirt',
            target='shirt', replacement='nude')
        self.assertNotIn('nude', out['negative_prompt'])
        self.assertEqual(out['negative_prompt'], 'bad hands')

    def test_inherit_off_uses_replacement_only(self):
        out = build_refine_prompts(
            main_prompt=self.MAIN, target='shirt', replacement='nude',
            inherit_main=False)
        self.assertEqual(out['prompt'], 'nude')

    def test_no_target_appends_replacement(self):
        out = build_refine_prompts(main_prompt='1girl, solo', replacement='detailed face')
        self.assertEqual(out['prompt'], '1girl, solo, detailed face')

    def test_unmatched_target_appends_replacement(self):
        # 리뷰 2026-11-04 SAM-05 — 예전엔 'portrait, 1girl' 로 Replacement 가 조용히 사라졌다(확장 handle_refine_click 과 같게)
        out = build_refine_prompts(main_prompt='portrait, 1girl', target='shirt', replacement='red dress')
        self.assertEqual(out['prompt'], 'portrait, 1girl, red dress')
        self.assertTrue(out['replacement_appended'])

    def test_matched_target_equal_to_replacement_is_not_appended_twice(self):
        out = build_refine_prompts(main_prompt='red dress, 1girl', target='red dress', replacement='red dress')
        self.assertEqual(out['prompt'], 'red dress, 1girl', '매칭은 됐다 — 결과가 원문과 같아도 덧붙이지 않는다')
        self.assertFalse(out['replacement_appended'])

    def test_matched_target_is_not_flagged_as_appended(self):
        out = build_refine_prompts(main_prompt=self.MAIN, target='shirt', replacement='nude')
        self.assertFalse(out['replacement_appended'])

    def test_unmatched_target_without_replacement_keeps_main(self):
        for replacement in ('', '   '):
            out = build_refine_prompts(main_prompt='portrait, 1girl', target='shirt', replacement=replacement)
            self.assertEqual(out['prompt'], 'portrait, 1girl')
            self.assertFalse(out['replacement_appended'])

    def test_unmatched_target_with_inherit_off_uses_replacement_only(self):
        out = build_refine_prompts(main_prompt='portrait, 1girl', target='shirt', replacement='red dress',
                                   inherit_main=False)
        self.assertEqual(out['prompt'], 'red dress')
        self.assertFalse(out['replacement_appended'])

    def test_no_target_no_replacement_keeps_main(self):
        out = build_refine_prompts(main_prompt='1girl, solo')
        self.assertEqual(out['prompt'], '1girl, solo')

    def test_negative_inherit_off(self):
        out = build_refine_prompts(
            main_prompt='a', main_negative='bad hands',
            negative='blurry', inherit_negative=False)
        self.assertEqual(out['negative_prompt'], 'blurry')

    def test_negative_inherit_on_appends(self):
        out = build_refine_prompts(
            main_prompt='a', main_negative='bad hands', negative='blurry')
        self.assertEqual(out['negative_prompt'], 'bad hands, blurry')

    def test_detect_tokens_reported(self):
        out = build_refine_prompts(main_prompt='a', target='face / hand')
        self.assertEqual(out['detect_tokens'], ['face', 'hand'])

    def test_changed_flag(self):
        self.assertTrue(build_refine_prompts(
            main_prompt='a, shirt', target='shirt', replacement='nude')['changed'])
        self.assertFalse(build_refine_prompts(
            main_prompt='a, b', target='zzz', replacement='nude')['changed'])

    def test_slash_target_splits_like_sam3(self):
        out = build_refine_prompts(
            main_prompt='1boy, shirt, necktie', target='shirt / necktie', replacement='nude')
        self.assertEqual(out['prompt'], '1boy, nude')

    def test_output_is_normalized(self):
        out = build_refine_prompts(main_prompt='a,,  shirt ,  b', target='shirt', replacement='X')
        self.assertEqual(out['prompt'], 'a, X, b')

    def test_all_empty_is_safe(self):
        out = build_refine_prompts()
        self.assertEqual(out['prompt'], '')
        self.assertEqual(out['negative_prompt'], '')


if __name__ == '__main__':
    unittest.main()
