"""ui.generator_prompts.apply_prompt_from_data — `제외 (로컬)` 칸이 규칙 종류를 잃지 않는다.

결함: 규칙 텍스트를 태그용 to_list 로 나눠, 쉼표가 없으면 공백으로 쪼개고 밑줄을 공백으로 바꿨다.
그래서 규칙 하나짜리 `_short`(접미)가 `short`(포함)가 되어 short hair·short pants 까지 지웠다.
이제 규칙은 core.exclude_rules 가 쉼표·줄바꿈으로만 나눈다 — comma_only 플래그와도 무관하다.
"""
from __future__ import annotations

import unittest

from ui.generator_prompts import PromptHandlingMixin


class _Check:
    def __init__(self, checked: bool = False):
        self._checked = checked

    def isChecked(self) -> bool:
        return self._checked


class _Text:
    def __init__(self, text: str = ""):
        self._text = text

    def toPlainText(self) -> str:
        return self._text

    def setPlainText(self, text: str) -> None:
        self._text = text

    def text(self) -> str:
        return self._text

    def setText(self, text: str) -> None:
        self._text = text


class _Harness(PromptHandlingMixin):
    """apply_prompt_from_data 가 읽는 위젯만 갖춘 대역 (제거 토글은 모두 꺼짐)."""

    def __init__(self, exclude_text: str):
        self._tag_classifier = None
        self.exclude_prompt_local_input = _Text(exclude_text)
        self.btn_lock_artist = _Check()
        for name in ("chk_remove_artist", "chk_remove_copyright", "chk_remove_character",
                     "chk_remove_meta", "chk_remove_censorship", "chk_remove_text"):
            setattr(self, name, _Check())
        for name in ("prefix_prompt_text", "suffix_prompt_text", "neg_prompt_text", "artist_input",
                     "char_count_input", "character_input", "copyright_input", "main_prompt_text"):
            setattr(self, name, _Text())
        self.is_programmatic_change = False

    @property
    def tag_classifier(self):  # 토글이 꺼져 있으면 닿지 않는다
        raise AssertionError("classifier must not be built")

    def _apply_conditional_prompts(self):
        pass

    def update_total_prompt_display(self):
        pass


GENERAL = "1girl, short, too short, short hair, short pants, long hair, tank top, blue tank top"


def _apply(exclude_text: str, *, comma_only: bool = False, **bundle) -> _Harness:
    harness = _Harness(exclude_text)
    data = {"general": GENERAL, "character": "", "copyright": "", "artist": ""}
    data.update(bundle)
    harness.apply_prompt_from_data(data, comma_only=comma_only)
    return harness


class ApplyPromptExcludeRulesTests(unittest.TestCase):
    def test_single_suffix_rule_without_comma(self):
        for comma_only in (False, True):
            with self.subTest(comma_only=comma_only):
                h = _apply("_short", comma_only=comma_only)
                self.assertEqual(h.main_prompt_text.toPlainText(),
                                 "short hair, short pants, long hair, tank top, blue tank top")

    def test_single_prefix_rule_without_comma(self):
        h = _apply("short_")
        self.assertEqual(h.main_prompt_text.toPlainText(), "too short, long hair, tank top, blue tank top")

    def test_single_keep_suffix_rule_without_comma_removes_nothing(self):
        # 옛 동작: '~_tank_top' → '~ tank top' 로 바뀌어 예외 완전일치 '~ tank top' 이 됐다
        h = _apply("~_tank_top")
        self.assertEqual(h.main_prompt_text.toPlainText(),
                         "short, too short, short hair, short pants, long hair, tank top, blue tank top")

    def test_multi_word_rule_is_one_rule(self):
        # 옛 동작: 'long hair' → 'long','hair' → short hair 까지 지웠다
        h = _apply("long hair")
        self.assertEqual(h.main_prompt_text.toPlainText(),
                         "short, too short, short hair, short pants, tank top, blue tank top")

    def test_trailing_comma_workaround_gives_the_same_result(self):
        for rule in ("_short", "short_", "*tank top", "_tank_top_", "~_tank_top"):
            with self.subTest(rule=rule):
                self.assertEqual(_apply(rule).main_prompt_text.toPlainText(),
                                 _apply(rule + ",").main_prompt_text.toPlainText())

    def test_rules_on_separate_lines(self):
        h = _apply("_short\n_tank_top_\n~blue tank top")
        self.assertEqual(h.main_prompt_text.toPlainText(),
                         "short hair, short pants, long hair, blue tank top")
        self.assertEqual(h.char_count_input.text(), "1girl")

    def test_proper_noun_fields_only_lose_whole_tag_contains_matches(self):
        h = _apply("tank", character="wakan tanka, tank", copyright="blue archive")
        self.assertEqual(h.character_input.text(), "wakan tanka")
        self.assertEqual(h.main_prompt_text.toPlainText(), "short, too short, short hair, short pants, long hair")
        # 접두 규칙은 고유명사에도 걸린다
        h = _apply("blue_", copyright="blue archive, original")
        self.assertEqual(h.copyright_input.text(), "")
        self.assertEqual(h.main_prompt_text.toPlainText(),
                         "short, too short, short hair, short pants, long hair, tank top")

    def test_empty_exclude_text_keeps_everything(self):
        h = _apply("  \n ")
        self.assertEqual(h.main_prompt_text.toPlainText(),
                         "short, too short, short hair, short pants, long hair, tank top, blue tank top")


if __name__ == "__main__":
    unittest.main()
