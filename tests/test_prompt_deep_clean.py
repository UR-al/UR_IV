"""프롬프트 [최적화] 딥 클리너(core.prompt_deep_clean) — #38.

최적화는 메인 태그만 다루고, 다른 6칸은 context 로 '이미 있는 태그'와 충돌 검사에만 쓴다.
예전엔 7칸 합본을 메인에 통째로 써서 캐릭터·작가·접두·접미 태그가 메인에 영구 복제됐다.
"""
import json
import unittest

from core.prompt_deep_clean import deep_clean_prompt, deep_clean_request, tag_key


class DeepCleanPromptTests(unittest.TestCase):
    def test_result_never_contains_context_tags(self):
        main = 'smile, long hair, hatsune miku, blue sky'
        context = ['1girl', 'hatsune miku', 'vocaloid', 'artist_x', 'masterpiece', 'simple background']
        out = deep_clean_prompt(main, context)
        self.assertEqual(out['optimized'], 'smile, long hair, blue sky')
        for tag in ('1girl', 'vocaloid', 'artist_x', 'masterpiece', 'simple background'):
            self.assertNotIn(tag, out['optimized'])
        self.assertEqual(out['removed'], 1)
        self.assertEqual(out['removed_from_context'], 1)
        self.assertEqual(out['tag_count'], 3)

    def test_duplicates_inside_main_and_against_context_use_the_same_normalisation(self):
        out = deep_clean_prompt('Long_Hair, long hair, smile, Hatsune_Miku', 'hatsune miku, 1girl')
        self.assertEqual(out['optimized'], 'Long_Hair, smile')
        self.assertEqual(out['removed'], 2)
        self.assertEqual(out['removed_from_context'], 1)

    def test_conflicts_are_checked_across_main_and_context(self):
        out = deep_clean_prompt('blonde hair, smile', ['black_hair', 'outdoors', 'indoors'])
        groups = {c['group']: c['tags'] for c in out['conflicts']}
        self.assertEqual(groups['머리색'], ['black_hair', 'blonde_hair'])
        self.assertEqual(groups['장소'], ['indoors', 'outdoors'])
        # 결과에는 여전히 메인 태그만
        self.assertEqual(out['optimized'], 'blonde hair, smile')

    def test_reorders_count_then_quality_then_rest(self):
        out = deep_clean_prompt('smile, masterpiece, 1girl, highres, sky')
        self.assertEqual(out['optimized'], '1girl, masterpiece, highres, smile, sky')

    def test_all_main_tags_duplicated_elsewhere_gives_empty_result(self):
        out = deep_clean_prompt('1girl, solo', ['1girl', 'solo'])
        self.assertEqual(out['optimized'], '')
        self.assertEqual(out['removed'], 2)
        self.assertEqual(out['tag_count'], 0)

    def test_legacy_payload_without_context_still_works(self):
        out = deep_clean_request({'prompt': 'a, b, a'})
        self.assertEqual(out['optimized'], 'a, b')
        self.assertEqual(out['removed_from_context'], 0)
        # 잘못된 context 는 무시
        self.assertEqual(deep_clean_prompt('a', 42)['optimized'], 'a')
        self.assertEqual(deep_clean_prompt('a', [None, 3, 'a'])['optimized'], '')
        with self.assertRaises(ValueError):
            deep_clean_request(['not', 'a', 'dict'])

    def test_tag_key(self):
        self.assertEqual(tag_key(' Long Hair '), 'long_hair')


class DeepCleanBridgeSlotTests(unittest.TestCase):
    def test_slot_accepts_context_and_reports_errors_as_json(self):
        from ui.vue_bridge import VueBridge

        bridge = VueBridge()
        reply = json.loads(bridge.deepCleanPrompt(json.dumps({
            'prompt': 'smile, hatsune miku', 'context': ['hatsune miku'],
        })))
        self.assertEqual(reply['optimized'], 'smile')
        self.assertEqual(reply['removed_from_context'], 1)
        self.assertIn('error', json.loads(bridge.deepCleanPrompt('[1, 2]')))
        self.assertIn('error', json.loads(bridge.deepCleanPrompt('{not json')))


if __name__ == '__main__':
    unittest.main()
