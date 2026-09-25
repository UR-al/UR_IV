"""core.exclude_rules — 제외 규칙 문법의 단일 출처 (Qt 없이).

* 나누기: 쉼표·줄바꿈만. 공백으로 나누지 않고 밑줄을 파싱 때 바꾸지 않는다.
  (옛 generator_prompts 는 태그용 to_list 로 규칙을 나눠, 쉼표가 없으면 `_short` 를 `short` 로 바꿨다.)
* 분류: 9종 — 프론트(utils/excludeRules.ts)와 같은 골든 파일(excludeRules.cases.json)로 묶인다.
* 적용: 쉼표 목록에서는 옛 인라인 로직과 같은 결과(골든).
* 미리보기(core.exclude_rule_match)와 적용이 같은 규칙을 같은 태그에 건다.
"""
from __future__ import annotations

import json
import unittest
from pathlib import Path

from core.exclude_rule_match import match_exclude_rule
from core.exclude_rules import (
    CONTAINS, EXACT, PREFIX, RULE_FORMS, SUFFIX,
    ExcludeRuleSet, keyword_matches, normalize_exclude_tag,
    parse_exclude_rule, parse_exclude_rules, split_exclude_rules,
)

ROOT = Path(__file__).resolve().parents[1]
CASES = json.loads(
    (ROOT / "frontend" / "src" / "utils" / "excludeRules.cases.json").read_text(encoding="utf-8"))

GENERAL = [
    "1girl", "short hair", "very_short_hair", "short", "too short", "short pants", "long hair",
    "blue hair", "Blue_Hair", "hair ornament", "tank top", "blue tank top", "tank", "tanktop",
    "solo", "solo focus", "wakan tanka",
]
PROPER = ["wakan tanka", "tank", "Tank_Girl", "hatsune miku", "blue_archive", "short (artist)"]


def _without(tags, *removed):
    gone = set(removed)
    return [t for t in tags if t not in gone]


class SharedGoldenCasesTests(unittest.TestCase):
    """프론트 구현과 같은 파일 — 여기서 실패하면 두 파서가 갈라진 것이다."""

    def test_split_cases(self):
        self.assertGreater(len(CASES["split"]), 5)
        for case in CASES["split"]:
            with self.subTest(text=case["text"]):
                self.assertEqual(split_exclude_rules(case["text"]), case["rules"])

    def test_parse_cases(self):
        seen_forms = set()
        for case in CASES["parse"]:
            with self.subTest(rule=case["rule"]):
                parsed = parse_exclude_rule(case["rule"])
                if case["form"] is None:
                    self.assertIsNone(parsed)
                    continue
                self.assertIsNotNone(parsed)
                self.assertEqual(parsed.form, case["form"])
                self.assertEqual(parsed.keyword, case["keyword"])
                self.assertEqual(parsed.text, case["rule"].strip())
                seen_forms.add(parsed.form)
        self.assertEqual(seen_forms, set(RULE_FORMS), "골든 사례가 9종을 모두 덮어야 한다")

    def test_kept_cases(self):
        # 관리 창의 예외 표시(buildExcludeKeepIndex)와 적용이 같은 태그를 유지로 본다 —
        # 규칙 끝에 그 태그의 완전 일치 제외(*태그)를 더해도 지워지지 않으면 '유지'다
        self.assertGreater(len(CASES["kept"]), 5)
        for case in CASES["kept"]:
            with self.subTest(text=case["text"], tag=case["tag"]):
                rules = parse_exclude_rules(split_exclude_rules(case["text"]) + ["*" + case["tag"]])
                self.assertEqual(not rules.excludes(case["tag"]), case["kept"])


class SplitTests(unittest.TestCase):
    def test_never_splits_on_spaces_or_rewrites_underscores(self):
        self.assertEqual(split_exclude_rules("long hair"), ["long hair"])
        self.assertEqual(split_exclude_rules("_tank_top_"), ["_tank_top_"])
        self.assertEqual(split_exclude_rules("~_tank_top"), ["~_tank_top"])

    def test_non_text_is_empty(self):
        self.assertEqual(split_exclude_rules(None), [])
        self.assertEqual(split_exclude_rules(12), [])

    def test_rule_list_input(self):
        rules = parse_exclude_rules(["_short", "  ", "~short pants", None])
        self.assertEqual([r.form for r in rules.rules], ["suffix", "keep_exact"])
        self.assertEqual(len(parse_exclude_rules(None)), 0)


class RuleModelTests(unittest.TestCase):
    def test_form_table(self):
        self.assertEqual(len(RULE_FORMS), 9)
        self.assertEqual(
            {form: match for form, (match, _keep) in RULE_FORMS.items()},
            {
                "contains": CONTAINS, "contains_explicit": CONTAINS, "exact": EXACT,
                "prefix": PREFIX, "suffix": SUFFIX,
                "keep_exact": EXACT, "keep_contains": CONTAINS,
                "keep_prefix": PREFIX, "keep_suffix": SUFFIX,
            },
        )
        self.assertEqual({f for f, (_m, keep) in RULE_FORMS.items() if keep},
                         {"keep_exact", "keep_contains", "keep_prefix", "keep_suffix"})

    def test_keyword_matches(self):
        self.assertTrue(keyword_matches(EXACT, "blue hair", "blue hair"))
        self.assertFalse(keyword_matches(EXACT, "blue hairband", "blue hair"))
        self.assertTrue(keyword_matches(CONTAINS, "very short hair", "short"))
        self.assertTrue(keyword_matches(PREFIX, "short pants", "short"))
        self.assertFalse(keyword_matches(PREFIX, "too short", "short"))
        self.assertTrue(keyword_matches(SUFFIX, "too short", "short"))
        with self.assertRaises(ValueError):
            keyword_matches("bogus", "a", "a")

    def test_normalize_tag(self):
        self.assertEqual(normalize_exclude_tag("  Blue_Hair "), "blue hair")
        self.assertEqual(normalize_exclude_tag("_hair_"), "hair")


class LegacyGoldenTests(unittest.TestCase):
    """쉼표 목록 — 옛 인라인 로직(1bf0b2c)이 낸 결과를 그대로 박아 둔다."""

    def assertFiltered(self, text, general, proper):
        rules = parse_exclude_rules(text)
        self.assertEqual(rules.filter_tags(GENERAL), general, f"general: {text!r}")
        self.assertEqual(rules.filter_tags(PROPER, proper=True), proper, f"proper: {text!r}")

    def test_contains_and_exact(self):
        self.assertFiltered(
            "short, *blue hair",
            ["1girl", "long hair", "hair ornament", "tank top", "blue tank top", "tank", "tanktop",
             "solo", "solo focus", "wakan tanka"],
            PROPER,
        )

    def test_exception_beats_contains(self):
        self.assertFiltered(
            "short, ~short pants",
            ["1girl", "short pants", "long hair", "blue hair", "Blue_Hair", "hair ornament", "tank top",
             "blue tank top", "tank", "tanktop", "solo", "solo focus", "wakan tanka"],
            PROPER,
        )

    def test_suffix_and_prefix_apply_to_proper_nouns(self):
        self.assertFiltered(
            "_short, short_",
            ["1girl", "very_short_hair", "long hair", "blue hair", "Blue_Hair", "hair ornament", "tank top",
             "blue tank top", "tank", "tanktop", "solo", "solo focus", "wakan tanka"],
            ["wakan tanka", "tank", "Tank_Girl", "hatsune miku", "blue_archive"],
        )

    def test_explicit_contains_with_exact_exception(self):
        self.assertFiltered(
            "_tank_top_, ~blue tank top",
            _without(GENERAL, "tank top"),
            PROPER,
        )

    def test_keep_suffix_and_keep_prefix(self):
        self.assertFiltered("hair, ~_hair, ~short_", _without(GENERAL, "hair ornament"), PROPER)

    def test_proper_nouns_only_lose_whole_tag_contains_matches(self):
        # 'tank' 가 캐릭터 'wakan tanka' 를 부분일치로 지우지 않는다 (general 에서는 지운다)
        self.assertFiltered(
            "tank, ~_tank_top_",
            _without(GENERAL, "tank", "tanktop", "wakan tanka"),
            _without(PROPER, "tank"),
        )

    def test_case_and_underscore_insensitive(self):
        self.assertFiltered("*Blue_Hair, SOLO_", _without(GENERAL, "blue hair", "Blue_Hair", "solo", "solo focus"),
                            PROPER)

    def test_padding_and_trailing_commas(self):
        self.assertFiltered(" tank ,  *solo , ", _without(GENERAL, "tank top", "blue tank top", "tank",
                                                        "tanktop", "wakan tanka", "solo"),
                            _without(PROPER, "tank"))

    def test_old_trailing_comma_workaround_still_works(self):
        self.assertFiltered("_short,", _without(GENERAL, "short", "too short"), PROPER)
        self.assertFiltered("short_,", _without(GENERAL, "short hair", "short", "short pants"),
                            _without(PROPER, "short (artist)"))
        self.assertFiltered("~_tank_top,", GENERAL, PROPER)
        self.assertFiltered("long hair,", _without(GENERAL, "long hair"), PROPER)


class SingleRuleWithoutCommaTests(unittest.TestCase):
    """결함 재현 — 쉼표 없는 규칙 하나가 종류를 잃지 않는다 (9종 모두)."""

    def _form(self, text):
        rules = parse_exclude_rules(text).rules
        self.assertEqual(len(rules), 1, f"{text!r} 는 규칙 하나다")
        return rules[0]

    def test_all_nine_forms_keep_their_type(self):
        expected = {
            "_short": ("suffix", "short"),
            "short_": ("prefix", "short"),
            "*blue hair": ("exact", "blue hair"),
            "_tank_top_": ("contains_explicit", "tank top"),
            "~blue hair": ("keep_exact", "blue hair"),
            "~_tank_top": ("keep_suffix", "tank top"),
            "~tank_": ("keep_prefix", "tank"),
            "~_x_": ("keep_contains", "x"),
            "long hair": ("contains", "long hair"),
        }
        for text, (form, keyword) in expected.items():
            with self.subTest(rule=text):
                rule = self._form(text)
                self.assertEqual((rule.form, rule.keyword), (form, keyword))

    def test_single_rule_equals_the_trailing_comma_form(self):
        for text in ("_short", "short_", "*blue hair", "_tank_top_", "~blue hair", "~_tank_top",
                     "~tank_", "~_x_", "long hair"):
            with self.subTest(rule=text):
                alone = parse_exclude_rules(text)
                with_comma = parse_exclude_rules(text + ",")
                self.assertEqual(alone.filter_tags(GENERAL), with_comma.filter_tags(GENERAL))
                self.assertEqual(alone.filter_tags(PROPER, proper=True),
                                 with_comma.filter_tags(PROPER, proper=True))

    def test_suffix_rule_no_longer_becomes_contains(self):
        # 옛 동작: '_short' → 'short'(포함) → short hair·short pants 까지 지웠다
        self.assertEqual(parse_exclude_rules("_short").filter_tags(GENERAL), _without(GENERAL, "short", "too short"))

    def test_prefix_rule_no_longer_becomes_contains(self):
        self.assertEqual(parse_exclude_rules("short_").filter_tags(GENERAL),
                         _without(GENERAL, "short hair", "short", "short pants"))

    def test_multi_word_rule_is_one_rule(self):
        # 옛 동작: 'long hair' → 'long', 'hair' 두 포함 규칙 → hair 가 든 태그가 모두 사라졌다
        self.assertEqual(parse_exclude_rules("long hair").filter_tags(GENERAL), _without(GENERAL, "long hair"))

    def test_keep_rule_alone_removes_nothing(self):
        self.assertEqual(parse_exclude_rules("~_tank_top").filter_tags(GENERAL), GENERAL)


class NewlineSeparationTests(unittest.TestCase):
    def test_one_rule_per_line(self):
        text = "_short\nshort_\r\n*blue hair\n~blue tank top\n_tank_top_"
        rules = parse_exclude_rules(text)
        self.assertEqual([r.form for r in rules.rules], ["suffix", "prefix", "exact", "keep_exact", "contains_explicit"])
        self.assertEqual(
            rules.filter_tags(GENERAL),
            ["1girl", "very_short_hair", "long hair", "hair ornament", "blue tank top",
             "tank", "tanktop", "solo", "solo focus", "wakan tanka"],
        )

    def test_pasted_reference_lines_do_not_merge_across_the_line_break(self):
        # config/default_excludes.txt 의 두 줄을 붙여 넣은 모양 — 줄 끝 규칙과 다음 줄 첫 규칙이 합쳐지지 않는다
        text = "watermark, *qr code\nborder, *2koma"
        self.assertEqual(split_exclude_rules(text), ["watermark", "*qr code", "border", "*2koma"])
        tags = ["qr code", "border", "2koma", "watermark", "solo"]
        self.assertEqual(parse_exclude_rules(text).filter_tags(tags), ["solo"])


class LoneMarkerLineTests(unittest.TestCase):
    """줄 끝에 표시만 있는 조각(`~`, `*`, `_` …)은 다음 줄 조각과 한 규칙이다.

    옛 적용(1bf0b2c)은 쉼표 목록 안의 줄바꿈을 규칙 안 공백으로 봤다: `x, ~\\nsolo` = `~solo` 유지.
    줄바꿈 구분만 더하면 표시가 버려지고 `solo` 가 포함 제외가 되어, 유지하려던 태그와
    `solo focus` 까지 지운다(유지 → 제외로 뒤집힘).
    """

    TAGS = ["solo", "solo focus", "x", "long hair", "hair ornament"]

    def assertSameAs(self, text, equivalent):
        self.assertEqual(parse_exclude_rules(text).filter_tags(self.TAGS),
                         parse_exclude_rules(equivalent).filter_tags(self.TAGS), f"{text!r} ≡ {equivalent!r}")
        self.assertEqual(parse_exclude_rules(text).filter_tags(GENERAL),
                         parse_exclude_rules(equivalent).filter_tags(GENERAL), f"{text!r} ≡ {equivalent!r}")

    def test_keep_marker_line_still_keeps(self):
        self.assertEqual(parse_exclude_rules("x, solo, ~\nsolo").filter_tags(self.TAGS),
                         ["solo", "long hair", "hair ornament"])
        self.assertSameAs("x, solo, ~\nsolo", "x, solo, ~solo")
        self.assertSameAs("solo, ~\n\nsolo", "solo, ~solo")          # 빈 줄을 건너도 쉼표 전이면 잇는다
        self.assertSameAs("solo, ~ \r\n  solo  ", "solo, ~solo")

    def test_exact_and_affix_marker_lines_keep_their_kind(self):
        self.assertEqual(parse_exclude_rules("x, *\nsolo").filter_tags(self.TAGS),
                         ["solo focus", "long hair", "hair ornament"])   # 포함이 아니라 완전 일치
        self.assertSameAs("x, *\nsolo", "x, *solo")
        self.assertSameAs("x, _\nhair", "x, _hair")
        self.assertSameAs("x, ~\n_\nhair, hair", "x, ~_hair, hair")    # 표시만 있는 줄이 이어져도

    def test_marker_without_following_rule_is_dropped(self):
        # 쉼표 앞이나 끝에 있는 표시만 있는 조각은 규칙이 없다 — 옛 적용도 빈 키워드라 아무것도 안 했다
        self.assertSameAs("x, ~,\nsolo", "x, solo")
        self.assertSameAs("x, ~\n,solo", "x, solo")
        self.assertSameAs("solo\n~", "solo")
        # 앞 줄로는 잇지 않는다 — 줄 머리의 `_` 는 다음 규칙의 접미 표시일 수도 있어 어느 쪽인지 모른다
        self.assertSameAs("solo\n_", "solo")

    def test_normal_lines_are_never_joined(self):
        self.assertEqual(split_exclude_rules("short_\nhair"), ["short_", "hair"])
        self.assertEqual(split_exclude_rules("~solo\nsolo"), ["~solo", "solo"])


class DegenerateRuleTests(unittest.TestCase):
    def test_empty_keyword_rules_are_dropped_instead_of_matching_everything(self):
        # 옛 코드: '__'(빈 접미)·'___'(빈 포함)가 모든 태그를 지웠고, '~__' 는 모든 제외를 무력화했다
        for text in ("__", "___", "_", "*", "~", "~_", "~__", "_ _"):
            with self.subTest(rule=text):
                self.assertEqual(len(parse_exclude_rules(text)), 0)
        self.assertEqual(parse_exclude_rules("__, ___, short").filter_tags(GENERAL),
                         parse_exclude_rules("short").filter_tags(GENERAL))
        self.assertEqual(parse_exclude_rules("~__, short").filter_tags(GENERAL),
                         parse_exclude_rules("short").filter_tags(GENERAL))

    def test_empty_rule_set_is_identity(self):
        rules = ExcludeRuleSet([])
        self.assertFalse(rules)
        self.assertEqual(rules.filter_tags(iter(GENERAL)), GENERAL)


class ProperNounTests(unittest.TestCase):
    def test_contains_forms_match_whole_tag_only(self):
        for text in ("tank", "_tank_"):
            with self.subTest(rule=text):
                rules = parse_exclude_rules(text)
                self.assertTrue(rules.excludes("Tank", proper=True))
                self.assertFalse(rules.excludes("wakan_tanka", proper=True))
                self.assertTrue(rules.excludes("wakan_tanka"))

    def test_exact_prefix_suffix_apply_to_proper_nouns(self):
        self.assertTrue(parse_exclude_rules("*hatsune miku").excludes("Hatsune_Miku", proper=True))
        self.assertTrue(parse_exclude_rules("blue_").excludes("blue_archive", proper=True))
        self.assertTrue(parse_exclude_rules("_(artist)").excludes("short (artist)", proper=True))

    def test_keep_rules_apply_to_proper_nouns_too(self):
        rules = parse_exclude_rules("_tanka, ~_wakan_")
        self.assertFalse(rules.excludes("wakan tanka", proper=True))
        self.assertTrue(rules.excludes("tanka", proper=True))


class ReferenceSemanticsTests(unittest.TestCase):
    """ExcludeRuleSet 의 빠른 경로(종류별 키워드 묶음) = 규칙 하나씩 본 정의(ExcludeRule.matches)."""

    RULE_POOL = (
        "short", "long hair", "*blue hair", "_short", "short_", "_tank_top_", "~blue tank top",
        "~_hair", "~short_", "~_top_", "tank", "*Tank", "_(artist)", "blue_", "~tank_", "hair",
    )
    TAGS = GENERAL + PROPER + ["hair", "blue", "Short_Hair", "tank_top_", "_top"]

    @staticmethod
    def _reference(rules, tag, proper):
        nt = normalize_exclude_tag(tag)
        if any(r.keep and r.matches(nt) for r in rules):
            return False
        return any(not r.keep and r.matches(nt, proper=proper) for r in rules)

    def test_fast_path_equals_reference_for_rule_combinations(self):
        import itertools
        import random
        rng = random.Random(7)
        combos = [list(c) for n in (1, 2) for c in itertools.combinations(self.RULE_POOL, n)]
        combos += [rng.sample(self.RULE_POOL, rng.randint(3, len(self.RULE_POOL))) for _ in range(200)]
        for texts in combos:
            rule_set = parse_exclude_rules(texts)
            for proper in (False, True):
                for tag in self.TAGS:
                    self.assertEqual(
                        rule_set.excludes(tag, proper=proper),
                        self._reference(rule_set.rules, tag, proper),
                        f"{texts} / {tag!r} / proper={proper}",
                    )


class PreviewAgreementTests(unittest.TestCase):
    """미리보기(사전 = 소문자·밑줄 표기)와 적용(공백 표기 비교)이 같은 태그를 지운다."""

    VOCAB = frozenset({
        "long_hair", "short_hair", "very_short_hair", "hair_ornament", "hairband", "blue_eyes",
        "solo", "solo_focus", "1girl", "chair", "tank_top", "blue_tank_top", "tank", "tanktop",
        "wakan_tanka", "short", "too_short", "short_pants", "blue_hair",
    })
    RULES = (
        "short", "long hair", "*blue hair", "*Blue_Hair", "_short", "short_", "_tank_top_", "_hair",
        "hair_", "_ hair", "* long hair", "hair _", "tank", "_x_", "__", "_", "*", "~short", "~_hair",
        "Solo_", "blue", "_top",
        "*\nsolo", "_\nhair", "~\nsolo",   # 표시만 있는 줄 + 다음 줄 = 규칙 하나 (split_exclude_rules)
    )

    def test_preview_equals_application_for_every_rule(self):
        for rule in self.RULES:
            with self.subTest(rule=rule):
                applied = parse_exclude_rules([rule])
                removed = sorted(tag for tag in self.VOCAB if applied.excludes(tag))
                self.assertEqual(match_exclude_rule(rule, self.VOCAB), removed)

    def test_rule_is_classified_identically(self):
        # 미리보기가 따로 파싱하던 시절의 불일치 사례: '_ hair'(적용=접미 hair), '_'(적용=규칙 없음)
        self.assertEqual(match_exclude_rule("_ hair", self.VOCAB),
                         ["blue_hair", "chair", "long_hair", "short_hair", "very_short_hair"])
        self.assertEqual(match_exclude_rule("_", self.VOCAB), [])


if __name__ == "__main__":
    unittest.main()
