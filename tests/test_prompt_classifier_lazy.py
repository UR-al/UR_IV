"""ui.generator_prompts.apply_prompt_from_data — 제거 토글이 꺼져 있으면 태그 분류기를 만들지 않는다.

`tag_classifier` 는 지연 로드 프로퍼티라 hasattr·속성 접근만으로 분류기(GUI 스레드 약 0.5초 +
tag_groups·implications parquet)를 만든다. 예전 디버그 로그가 `isEnabledFor(DEBUG)` 뒤에서
개수를 세며 프로퍼티를 건드렸는데, utils/app_logger 가 루트를 DEBUG 로 두어 그 검사는 늘 참이었다.
"""
from __future__ import annotations

import logging
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


class _FakeClassifier:
    censorship_tags = {"censored"}
    text_tags = set()

    def is_meta_tag(self, tag: str) -> bool:
        return tag == "highres"

    def is_censorship_tag(self, tag: str) -> bool:
        return tag in self.censorship_tags

    def is_text_tag(self, tag: str) -> bool:
        return tag in self.text_tags


_TOGGLES = (
    "chk_remove_artist", "chk_remove_copyright", "chk_remove_character",
    "chk_remove_meta", "chk_remove_censorship", "chk_remove_text",
)


class _Harness(PromptHandlingMixin):
    def __init__(self, **checked):
        self._tag_classifier = None          # GeneratorBase 와 같은 지연 로드 슬롯
        self.classifier_accesses = 0
        self.exclude_prompt_local_input = _Text()
        self.btn_lock_artist = _Check()
        for name in _TOGGLES:
            setattr(self, name, _Check(checked.get(name, False)))
        for name in ("prefix_prompt_text", "suffix_prompt_text", "neg_prompt_text", "artist_input",
                     "char_count_input", "character_input", "copyright_input", "main_prompt_text"):
            setattr(self, name, _Text())
        self.is_programmatic_change = False

    @property
    def tag_classifier(self):
        # GeneratorBase.tag_classifier 대역 — 접근하면 분류기를 만든다
        self.classifier_accesses += 1
        if self._tag_classifier is None:
            self._tag_classifier = _FakeClassifier()
        return self._tag_classifier

    def _apply_conditional_prompts(self):
        pass

    def update_total_prompt_display(self):
        pass


BUNDLE = {"general": "1girl, smile, censored, highres", "character": "", "copyright": "", "artist": ""}


class PromptClassifierLazyTests(unittest.TestCase):
    def setUp(self):
        self._logger = logging.getLogger("prompts")
        self._old_level = self._logger.level
        # 실제 앱과 같은 조건 — DEBUG 가 켜져 있어도 분류기를 만들지 않아야 한다
        self._logger.setLevel(logging.DEBUG)

    def tearDown(self):
        self._logger.setLevel(self._old_level)

    def test_toggles_off_never_build_the_classifier_even_with_debug_logging(self):
        self.assertTrue(self._logger.isEnabledFor(logging.DEBUG))
        harness = _Harness()
        with self.assertLogs("prompts", level="DEBUG") as logs:
            harness.apply_prompt_from_data(BUNDLE)
        self.assertEqual(harness.classifier_accesses, 0, "제거 토글이 모두 꺼져 있으면 분류기를 만들지 않는다")
        self.assertIsNone(harness._tag_classifier)
        self.assertFalse(any("censorship_tags" in line for line in logs.output))
        self.assertEqual(harness.char_count_input.text(), "1girl")
        self.assertEqual(harness.main_prompt_text.toPlainText(), "smile, censored, highres")

    def test_removal_toggles_still_use_the_classifier(self):
        harness = _Harness(chk_remove_meta=True, chk_remove_censorship=True)
        harness.apply_prompt_from_data(BUNDLE)
        self.assertGreater(harness.classifier_accesses, 0)
        self.assertEqual(harness.main_prompt_text.toPlainText(), "smile")

    def test_existing_classifier_counts_are_logged_without_touching_the_property(self):
        harness = _Harness()
        harness._tag_classifier = _FakeClassifier()
        with self.assertLogs("prompts", level="DEBUG") as logs:
            harness.apply_prompt_from_data(BUNDLE)
        self.assertEqual(harness.classifier_accesses, 0)
        self.assertTrue(any("censorship_tags 개수: 1, text_tags 개수: 0" in line for line in logs.output))


if __name__ == "__main__":
    unittest.main()
