"""core.request_numbers.parse_seed_checked — 상한을 넘는 seed 는 자르지 않고 SeedRangeError.

parse_seed(인페인트가 쓴다)는 0..2^32-1 로 자른다. I2I 는 예전 탭이 int() 값을 그대로 보냈으므로,
자르면 사용자가 적은 것과 다른 seed 로 알리지 않고 생성된다(ComfyUI 는 uint64 seed 를 그대로 쓴다).
"""
from __future__ import annotations

import unittest

from core.request_numbers import MAX_SEED, SeedRangeError, parse_seed, parse_seed_checked

UINT64_MAX = 2 ** 64 - 1


class ParseSeedCheckedTests(unittest.TestCase):
    def test_blank_text_and_negative_are_random(self):
        for raw in (None, '', '  ', 'abc', 'nan', 'inf', '1e400', True, False, -1, '-1', -7, '-5000000000'):
            with self.subTest(raw=raw):
                self.assertEqual(parse_seed_checked(raw, MAX_SEED), -1)

    def test_values_in_range_are_kept_exactly(self):
        for raw, expected in ((0, 0), ('42', 42), (' 42 ', 42), ('12.0', 12), (12.9, 12), ('1e3', 1000),
                              (MAX_SEED, MAX_SEED), (str(UINT64_MAX), UINT64_MAX)):
            with self.subTest(raw=raw):
                self.assertEqual(parse_seed_checked(raw, UINT64_MAX), expected)

    def test_values_above_the_limit_raise(self):
        for raw, limit in ((MAX_SEED + 1, MAX_SEED), (str(2 ** 40), MAX_SEED), (UINT64_MAX + 1, UINT64_MAX)):
            with self.subTest(raw=raw):
                with self.assertRaises(SeedRangeError) as caught:
                    parse_seed_checked(raw, limit)
                self.assertEqual((caught.exception.seed, caught.exception.maximum), (int(raw), limit))
                self.assertIsInstance(caught.exception, ValueError)

    def test_parse_seed_still_clamps_for_inpaint(self):
        self.assertEqual(parse_seed(2 ** 40), MAX_SEED)
        self.assertEqual(parse_seed('abc'), -1)


if __name__ == '__main__':
    unittest.main()
