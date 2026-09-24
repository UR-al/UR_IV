"""실시간 프롬프트 정리(PromptCleaner)가 와일드카드 토큰을 망가뜨리지 않는지 — 회귀.

Vue 와일드카드 관리자의 '사용' 버튼이 '__이름__, ' 을 넣으면 textChanged → 0.5초 뒤
_deferred_clean_all → PromptCleaner.clean 이 돈다. 밑줄→공백(기본 켜짐)이 토큰을 평문
'hairstyle' 로 바꿔 해석기가 못 찾고 그 낱말이 그대로 Forge/ComfyUI 로 나갔다.
"""
import os
import random
import tempfile
import unittest

from utils.file_wildcard import FileWildcardManager
from utils.prompt_cleaner import PromptCleaner


class CleanerKeepsWildcardTokensTests(unittest.TestCase):
    def setUp(self):
        self.cleaner = PromptCleaner()
        self.cleaner.underscore_to_space = True   # 사용자 기본값(prompt_settings.json)

    def test_both_syntaxes_survive_underscore_conversion(self):
        self.assertEqual(self.cleaner.clean('1girl, __hair_style__, ~/my_set/~'),
                         '1girl, __hair_style__, ~/my_set/~')

    def test_vue_inserted_token_survives_the_trailing_comma_trim(self):
        # App.vue useWcSyntax: cur + ', ' + '__이름__' + ', '
        self.assertEqual(self.cleaner.clean('1girl, __hairstyle__, '), '1girl, __hairstyle__')

    def test_ordinary_tags_are_still_converted(self):
        self.assertEqual(self.cleaner.clean('blue_eyes, __hair_style__, long_hair'),
                         'blue eyes, __hair_style__, long hair')
        self.assertEqual(self.cleaner.clean('score_9, a_b'), 'score_9, a b')   # 보존 목록 유지

    def test_n_pick_weighted_and_nested_tokens_are_kept_verbatim(self):
        for text in ('__hair_style:2__', '(__hair_style__:1.2)', '__a ~/b_c/~ d__',
                     '~/my_set:3/~', 'x, ~/a,b/~'):
            self.assertEqual(self.cleaner.clean(text), text)

    def test_other_rules_still_apply_around_tokens(self):
        self.assertEqual(self.cleaner.clean('a ,  __x_y__ ,, b_c ,'), 'a, __x_y__, b c')

    def test_duplicate_removal_treats_identical_tokens_as_duplicates(self):
        self.cleaner.remove_duplicates = True
        self.assertEqual(self.cleaner.clean('__x__, __x__, a_b, a b'), '__x__, a b')

    def test_instant_wildcard_tokens_survive_underscore_conversion(self):
        # 즉석 와일드카드 이름엔 공백이 허용되지 않는다 — '$$hair style$$' 은 풀리지 않고 그대로 전송됐다
        for text in ('1girl, $$hair_style$$, smile', '$$outfit/summer_dress$$', '$$a__b__c$$',
                     '(__a $$b_c$$ d__:1.1)', '~/$$w_v$$/~'):
            self.assertEqual(self.cleaner.clean(text), text)
        self.assertEqual(self.cleaner.clean('blue_eyes, $$hair_style$$, long_hair'),
                         'blue eyes, $$hair_style$$, long hair')
        # Vue 삽입 후 꼬리 쉼표 정리도 그대로
        self.assertEqual(self.cleaner.clean('1girl, $$hair_style$$, '), '1girl, $$hair_style$$')

    def test_duplicate_instant_tokens_follow_the_file_token_rule(self):
        self.cleaner.remove_duplicates = True
        self.assertEqual(self.cleaner.clean('$$x_y$$, $$x_y$$, a_b'), '$$x_y$$, a b')

    def test_placeholder_characters_in_input_are_left_alone(self):
        # 자리표시자와 같은 글자가 이미 있으면 가리지 않는다 — 엉뚱한 토큰으로 되돌리지 않는다
        out = self.cleaner.clean('a0_b, c_d')
        self.assertEqual(out, 'a0 b, c d')


class CleanedPromptResolvesTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.mgr = FileWildcardManager(os.path.join(self._tmp.name, 'wildcards'))
        for name, body in (('hairstyle', '# 주석\nponytail'), ('hair_style', 'twintails'),
                           ('my_set', 'smile')):
            with open(os.path.join(self.mgr.wildcards_dir, name + '.txt'), 'w', encoding='utf-8') as f:
                f.write(body)
        random.seed(3)

    def tearDown(self):
        self._tmp.cleanup()

    def test_inserted_tokens_are_substituted_after_realtime_cleaning(self):
        cleaned = PromptCleaner().clean('1girl, __hairstyle__, __hair_style__, ~/my_set/~, ')
        self.assertEqual(self.mgr.resolve(cleaned), '1girl, ponytail, twintails, smile')

    def test_instant_wildcard_is_substituted_after_realtime_cleaning(self):
        from core.instant_wildcards import InstantWildcards
        instant = InstantWildcards(rng=random.Random(0))   # store_path 없음 — 사용자 파일을 건드리지 않는다
        instant.set('hair_style', ['twintails'])
        instant.set('outfit/summer_dress', ['sundress'])
        cleaned = PromptCleaner().clean('1girl, $$hair_style$$, $$outfit/summer_dress$$, ')
        self.assertEqual(instant.resolve(cleaned), '1girl, twintails, sundress')


if __name__ == '__main__':
    unittest.main()
