"""조건부 규칙 → 위젯 적용 경로 회귀 (전역 _apply_vue_conditional_rules · 캐릭터 _apply_condition_result).

- 바뀐 칸만 다시 쓴다.
- 쓰는 동안 is_programmatic_change 가 서 있어 선행/후행/네거티브의 조건부 태그가 사용자
  템플릿(on_base_prompts_changed → base_*_prompt)에 스며들지 않는다.
"""
import unittest

from ui.generator_main import GeneratorMainUI
from ui.generator_prompts import PromptHandlingMixin as PromptsMixin


class _Line:
    """LineEditProxy 대역 (text/setText)."""

    def __init__(self, owner, value=""):
        self.owner, self.value, self.writes = owner, value, 0

    def text(self):
        return self.value

    def setText(self, value):
        self.value = value
        self.writes += 1


class _Plain:
    """TextEditProxy 대역 — setPlainText 가 textChanged 처럼 base 갱신 훅을 부른다."""

    def __init__(self, owner, name, value=""):
        self.owner, self.name, self.value, self.writes = owner, name, value, 0

    def toPlainText(self):
        return self.value

    def setPlainText(self, value):
        self.value = value
        self.writes += 1
        if not self.owner.is_programmatic_change:
            self.owner.base_writes.append(self.name)


class _Host(PromptsMixin):
    def __init__(self, **values):
        self.is_programmatic_change = False
        self.base_writes = []
        self.character_input = _Line(self, values.get("character", ""))
        self.copyright_input = _Line(self, values.get("copyright", ""))
        self.prefix_prompt_text = _Plain(self, "prefix", values.get("prefix", ""))
        self.main_prompt_text = _Plain(self, "main", values.get("main", ""))
        self.suffix_prompt_text = _Plain(self, "suffix", values.get("suffix", ""))
        self.neg_prompt_text = _Plain(self, "neg", values.get("neg", ""))


def _rule(cond, target, action="add", location="main", exists=True):
    return {"enabled": True, "condition": cond, "exists": exists, "target": target,
            "action": action, "location": location}


class GlobalRuleWidgetTests(unittest.TestCase):
    def apply(self, host, pos, neg=()):
        GeneratorMainUI._apply_vue_conditional_rules(host, list(pos), list(neg))

    def test_replace_is_tag_level_and_repeatable(self):
        host = _Host(main="1boy, muscular, muscular male, abs")
        rules = [_rule("muscular", "muscular male", action="replace")]
        self.apply(host, rules)
        self.apply(host, rules)   # 검색 적용 경로는 두 번 적용한다
        self.assertEqual(host.main_prompt_text.value, "1boy, muscular male, abs")

    def test_only_changed_fields_are_written(self):
        host = _Host(main="armpits", prefix="masterpiece,\nbest quality", neg="lowres")
        self.apply(host, [_rule("armpits", "armpit hair")])
        self.assertEqual(host.main_prompt_text.value, "armpits, armpit hair")
        self.assertEqual(host.prefix_prompt_text.writes, 0)
        self.assertEqual(host.neg_prompt_text.writes, 0)
        self.assertEqual(host.prefix_prompt_text.value, "masterpiece,\nbest quality")

    def test_conditional_tags_do_not_leak_into_base_templates(self):
        host = _Host(main="armpits", suffix="best quality", neg="lowres")
        self.apply(host, [_rule("armpits", "detailed", location="suffix")],
                   [_rule("armpits", "bad anatomy")])
        self.assertEqual(host.suffix_prompt_text.value, "best quality, detailed")
        self.assertEqual(host.neg_prompt_text.value, "lowres, bad anatomy")
        self.assertEqual(host.base_writes, [])
        self.assertFalse(host.is_programmatic_change)

    def test_weighted_groups_are_kept_whole(self):
        host = _Host(main="(smile, blush:1.2), beard")
        self.apply(host, [_rule("beard", "beard", action="remove")])
        self.assertEqual(host.main_prompt_text.value, "(smile, blush:1.2)")


class CharacterRuleWidgetTests(unittest.TestCase):
    def test_replace_uses_every_target_and_skips_existing(self):
        host = _Host(main="1boy, muscular, abs")
        PromptsMixin._apply_condition_result(host, {
            "_replace": [("muscular", "muscular male"), ("muscular", "abs")],
        })
        self.assertEqual(host.main_prompt_text.value, "1boy, muscular male, abs")

    def test_remove_from_positive_location_covers_all_positive_fields(self):
        host = _Host(prefix="beard, masterpiece", main="beard, 1boy", neg="beard")
        PromptsMixin._apply_condition_result(host, {"_remove_main": ["beard"]})
        self.assertEqual(host.prefix_prompt_text.value, "masterpiece")
        self.assertEqual(host.main_prompt_text.value, "1boy")
        self.assertEqual(host.neg_prompt_text.value, "beard")
        self.assertEqual(host.base_writes, [])

    def test_collect_all_tags_is_positive_only_and_unescaped(self):
        host = _Host(character=r"hatsune miku \(append\)", main="Smile", neg="nsfw")
        self.assertEqual(PromptsMixin._collect_all_tags(host), {"hatsune miku (append)", "smile"})


if __name__ == "__main__":
    unittest.main()
