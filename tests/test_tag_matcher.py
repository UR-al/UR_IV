"""tag_matcher.filter_dataframe — 결합 모드(AND/OR) + 공백/언더스코어 양쪽 매칭 회귀.

- combine_mode가 'or'면 콤마 리스트가 OR로, 'and'면 AND로 결합된다(검색 모드를 따름).
- 입력은 공백(monkey d. luffy)인데 데이터는 언더스코어(monkey_d._luffy)여도 매칭된다
  (matcher가 양쪽 형식을 모두 시도 — 일반검색이 이 경로를 씀. 심층검색은 별도 JS 경로).
"""
import unittest

try:
    import pandas as pd
    from core.tag_matcher import filter_dataframe
    _HAS_PANDAS = True
except Exception:
    _HAS_PANDAS = False


@unittest.skipUnless(_HAS_PANDAS, "pandas/tag_matcher 미설치")
class TestTagMatcherExclude(unittest.TestCase):
    def _df(self):
        return pd.DataFrame({'character': [
            'monkey_d._luffy, sanji_(one_piece)',   # luffy 포함
            'kaeya_(genshin_impact)',                # kaeya 포함
            'random_char, other_tag',                # 둘 다 없음
        ]})

    def test_space_input_matches_underscore_data(self):
        # 공백 입력이 언더스코어 데이터와 매칭 (단일 태그)
        mask = filter_dataframe(self._df(), 'character', 'monkey d. luffy', default_combine='and')
        self.assertEqual(list(mask), [True, False, False])

    def test_or_combine_matches_any(self):
        # OR 결합이면 어느 하나라도 매칭하면 True
        q = 'monkey d. luffy, kaeya (genshin impact)'
        mask_or = filter_dataframe(self._df(), 'character', q, default_combine='or')
        self.assertEqual(list(mask_or), [True, True, False])

    def test_and_combine_requires_all_terms(self):
        # AND 결합이면 나열한 텀을 모두 동시에 가진 행만 매칭 (여기선 없어 전부 False)
        q = 'monkey d. luffy, kaeya (genshin impact)'
        mask_and = filter_dataframe(self._df(), 'character', q, default_combine='and')
        self.assertEqual(list(mask_and), [False, False, False])


@unittest.skipUnless(_HAS_PANDAS, "pandas/tag_matcher 미설치")
class TestTagMatcherOrGroup(unittest.TestCase):
    def test_or_group_batched_matches_any(self):
        # [A|B|C] 큰 OR 그룹 — plain 텀들이 정규식 1회 스캔으로 묶여도 결과 동일.
        # 공백 입력 → 언더스코어 데이터 매칭. 어느 하나라도 가진 행이 True.
        df = pd.DataFrame({'character': [
            'phainon_(honkai:_star_rail)',
            'mydei_(honkai:_star_rail)',
            'random_char',
            'sanji_(one_piece), nami',
        ]})
        q = '[phainon (honkai: star rail)|sanji (one piece)|nobody here]'
        mask = filter_dataframe(df, 'character', q, default_combine='and')
        self.assertEqual(list(mask), [True, False, False, True])

    def test_or_group_mixes_plain_and_operator_terms(self):
        # plain + 연산자(*완전일치) 텀 혼합 OR 그룹도 정확히 동작
        df = pd.DataFrame({'general': ['1girl, smile', 'big smile, blush', '1boy, frown']})
        # [frown|*1girl] → 'frown' 부분일치 OR '1girl' 완전일치
        mask = filter_dataframe(df, 'general', '[frown|*1girl]', default_combine='and')
        self.assertEqual(list(mask), [True, False, True])


@unittest.skipUnless(_HAS_PANDAS, "pandas/tag_matcher 미설치")
class TestPlainTermSingleScan(unittest.TestCase):
    """plain(연산자 없는) 텀은 공백형·밑줄형을 한 번의 컬럼 스캔으로 찾는다."""

    def _series(self):
        return pd.Series([
            'long_hair smile',
            'letter long hair ornament',
            'short_hair',
            None,
            'kafka_(honkai:_star_rail) solo',
            'c++ programming',
        ], dtype='str')

    def _count_contains(self, series, fn):
        from unittest.mock import patch
        calls = []
        real = type(series.str).contains

        def counting(accessor, *args, **kwargs):
            calls.append(args[0] if args else kwargs.get('pat'))
            return real(accessor, *args, **kwargs)

        with patch.object(type(series.str), 'contains', counting):
            mask = fn()
        return mask, calls

    def test_default_and_explicit_contains_scan_once(self):
        from core.tag_matcher import _apply_pattern

        series = self._series()
        for pattern in ('long hair', 'long_hair', '_long hair_', 'smile'):
            with self.subTest(pattern=pattern):
                mask, calls = self._count_contains(
                    series, lambda p=pattern: _apply_pattern(series, p)
                )
                self.assertEqual(len(calls), 1, calls)
                # 'long_hair smile' 은 두 텀 모두, 'letter long hair ornament' 는 hair 텀만 잡힌다
                expected_second = pattern != 'smile'
                self.assertEqual(
                    list(mask),
                    [True, expected_second, False, False, False, False],
                )

    def test_mask_is_identical_to_the_old_two_scan_or(self):
        from core.tag_matcher import contains_tag_text

        series = self._series()
        for term in ('long hair', 'hair', 'star rail', 'honkai:', 'LONG_HAIR'):
            t = term.lower()
            old = (
                series.str.contains(t.replace('_', ' '), regex=False, na=False)
                | series.str.contains(t.replace(' ', '_'), regex=False, na=False)
            )
            with self.subTest(term=term):
                self.assertEqual(list(contains_tag_text(series, term)), list(old))

    def test_regex_metacharacters_are_matched_literally(self):
        from core.tag_matcher import contains_tag_text

        series = self._series()
        self.assertEqual(
            list(contains_tag_text(series, 'kafka (honkai: star rail)')),
            [False, False, False, False, True, False],
        )
        self.assertEqual(
            list(contains_tag_text(series, 'c++')),
            [False, False, False, False, False, True],
        )

    def test_filter_dataframe_keeps_space_and_underscore_semantics(self):
        df = pd.DataFrame({'general': list(self._series())})
        mask = filter_dataframe(df, 'general', 'long hair', col_lower=df['general'])
        self.assertEqual(list(mask), [True, True, False, False, False, False])


if __name__ == "__main__":
    unittest.main()
