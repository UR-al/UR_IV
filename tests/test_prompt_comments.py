"""core.prompt_comments — 프롬프트 주석(``//`` 줄 · ``/* */`` 블록) 문법과 요청 방어선.

* 골든 사례는 프론트(utils/promptComments.ts)와 같은 파일(promptComments.cases.json) — 두 구현이 갈라지면
  둘 중 한쪽이 실패한다. 주석은 줄 처음·공백·쉼표·블록 끝 뒤에서만 시작한다(URL·LoRA 경로·와일드카드 안전).
* 성질: 멱등, 결과에 주석 없음, 입력보다 길어지지 않음, ``/`` 가 없으면 같은 객체.
* 요청 방어선(strip_request_comments): 최상위·Hires·ADetailer 슬롯 전부·SAM3 state 의 프롬프트 키, 입력 불변,
  뗄 게 없으면 같은 객체. 컴파일러·인자 빌더가 읽는 '*prompt' 키는 모두 지키는 키 목록에 있어야 한다(정적 검사).
"""
from __future__ import annotations

import ast
import copy
import json
import random
import unittest
from pathlib import Path

from core.prompt_comments import (
    ADETAILER_PROMPT_KEYS, GUARDED_KEYS, PROMPT_KEYS, SAM3_PROMPT_KEYS,
    append_prompt_text, comment_code, comment_spans, ends_in_line_comment,
    filter_tokens_keep_comments, has_prompt_comment, strip_prompt_comments, strip_request_comments,
    underscores_to_spaces,
)

ROOT = Path(__file__).resolve().parents[1]
CASES = json.loads(
    (ROOT / "frontend" / "src" / "utils" / "promptComments.cases.json").read_text(encoding="utf-8"))


class SharedGoldenCasesTests(unittest.TestCase):
    """프론트 구현과 같은 파일 — 여기서 실패하면 두 구현이 갈라진 것이다."""

    def test_cases_are_present(self):
        self.assertGreater(len(CASES["strip"]), 20)
        self.assertGreater(len(CASES["endsInLineComment"]), 4)

    def test_spans(self):
        for case in CASES["strip"]:
            with self.subTest(text=case["text"]):
                self.assertEqual([list(s) for s in comment_spans(case["text"])], case["spans"])

    def test_stripped(self):
        for case in CASES["strip"]:
            with self.subTest(text=case["text"]):
                self.assertEqual(strip_prompt_comments(case["text"]), case["stripped"])

    def test_ends_in_line_comment(self):
        for case in CASES["endsInLineComment"]:
            with self.subTest(text=case["text"]):
                self.assertIs(ends_in_line_comment(case["text"]), case["ends"])

    def test_underscores_to_spaces(self):
        self.assertGreater(len(CASES["underscoresToSpaces"]), 4)
        for case in CASES["underscoresToSpaces"]:
            with self.subTest(text=case["text"]):
                out = underscores_to_spaces(case["text"])
                self.assertEqual(out, case["out"])
                # 바꾼 글자가 새 주석을 만들지 않는다
                self.assertEqual(has_prompt_comment(out), has_prompt_comment(case["text"]))

    def test_literal_slashes_never_start_a_comment(self):
        for text in ("http://e.com/a, red", "a//b", ">//<", "__x/*__", "~/x/~", "<lora:d/n:1>", "file:///p"):
            with self.subTest(text=text):
                self.assertFalse(has_prompt_comment(text))
                self.assertIs(strip_prompt_comments(text), text)


class PropertyTests(unittest.TestCase):
    ALPHABET = "a /*,\n:\\()\r\t"

    def test_random_strings(self):
        rng = random.Random(20261011)
        for _ in range(2000):
            text = "".join(rng.choice(self.ALPHABET) for _ in range(rng.randint(0, 24)))
            with self.subTest(text=text):
                once = strip_prompt_comments(text)
                self.assertEqual(strip_prompt_comments(once), once, "멱등이 아니다")
                self.assertEqual(comment_spans(once), [], "뗀 결과에 주석이 남았다")
                self.assertLessEqual(len(once), len(text))
                if "/" not in text:
                    self.assertIs(once, text)

    def test_lines_without_comments_are_untouched(self):
        text = "masterpiece,  best quality\n\t1girl ,solo\nBREAK\n// note\nscenery"
        out = strip_prompt_comments(text)
        self.assertEqual(out, "masterpiece,  best quality\n\t1girl ,solo\nBREAK\nscenery")

    def test_line_comment_ends_the_tag_like_the_block_view(self):
        # 블록 칸(split_tags·splitPromptBlocks)은 줄 주석 뒤 줄바꿈을 태그 경계로 본다 — 보내는 글자도 같게
        from utils.condition_block import split_tags
        for text in ("1girl // 주인공\nsolo, smile", "1girl, solo // 메모\r\nsmile", "a /* open\nb, c",
                     "1girl // a\n// b\n\n/* c */ solo"):
            with self.subTest(text=text):
                sent = strip_prompt_comments(text)
                blocks = [strip_prompt_comments(t).strip() for t in split_tags(text)]
                self.assertEqual([t.strip() for t in sent.split(",")], [b for b in blocks if b])
        # 이미 쉼표가 있거나, 괄호 안이거나, 다음 내용이 없거나, BREAK 줄이면 그대로
        self.assertEqual(strip_prompt_comments("a, // x\nb"), "a,\nb")
        self.assertEqual(strip_prompt_comments("(a, b // x\n:1.2), c"), "(a, b\n:1.2), c")
        self.assertEqual(strip_prompt_comments("a // x\n// y"), "a")
        self.assertEqual(strip_prompt_comments("a // x\n, b"), "a\n, b")
        self.assertEqual(strip_prompt_comments("solo\nBREAK // x\nscenery"), "solo\nBREAK\nscenery")

    def test_large_text_is_linear(self):
        text = "\n".join(f"tag{i}, // note {i}" for i in range(2000))
        out = strip_prompt_comments(text)
        self.assertEqual(out.count("\n"), 1999)
        self.assertNotIn("//", out)


class HelperTests(unittest.TestCase):
    def test_append_prompt_text(self):
        self.assertEqual(append_prompt_text("a", "b"), "a, b")
        self.assertEqual(append_prompt_text("a // note", "b"), "a // note\nb")
        self.assertEqual(append_prompt_text("a /* open", "b"), "a /* open\nb")
        self.assertEqual(append_prompt_text("a /* x */", "b"), "a /* x */, b")
        self.assertEqual(append_prompt_text("", "b"), "b")
        self.assertEqual(append_prompt_text("a", ""), "a")
        self.assertEqual(append_prompt_text("a", "b", sep=" "), "a b")
        # 붙인 글자는 주석에 먹히지 않고, 줄 주석이 끝낸 태그와 쉼표로 갈린다(블록 칸 경계와 같게)
        self.assertEqual(strip_prompt_comments(append_prompt_text("a // note", "b")), "a,\nb")

    def test_comment_code_and_filter(self):
        self.assertEqual(comment_code("  // only "), "")
        self.assertEqual(comment_code("smile // keep"), "smile")
        tokens = ["// header", "smile // why", "blue sky", "a /* x */ b"]
        self.assertEqual(filter_tokens_keep_comments(tokens, {"blue sky"}), ["// header", "blue sky"])
        self.assertEqual(filter_tokens_keep_comments(tokens, {"smile", "a b"}),
                         ["// header", "smile // why", "a /* x */ b"])


def _payload():
    return {
        "prompt": "1girl, // main\nsolo",
        "negative_prompt": "bad /* x */ hands",
        "hr_prompt": "detailed // hr",
        "hr_negative_prompt": "blurry",
        "prompt_id": "a // b",
        "steps": 20,
        "alwayson_scripts": {
            "ADetailer": {"args": [True, False,
                                   {"ad_model": "face", "ad_prompt": "face // a", "ad_negative_prompt": "x"},
                                   {"ad_model": "hand", "ad_prompt": "hand", "ad_negative_prompt": "/* n */ bad"}]},
            "SAM3 Mask": {"args": [{"sam3_prompt": "face // detect", "sam3_exclude_prompt": "hair /* x */",
                                    "sam3_inpaint_prompt": "smile // i", "sam3_negative_prompt": "// only",
                                    "sam3_source_image": "init // keep"}]},
            "Other": {"args": [{"prompt": "x // y"}]},
        },
    }


class StripRequestCommentsTests(unittest.TestCase):
    def test_all_prompt_keys(self):
        original = _payload()
        snapshot = copy.deepcopy(original)
        out = strip_request_comments(original)
        self.assertEqual(original, snapshot, "입력을 고쳤다")
        self.assertEqual(out["prompt"], "1girl,\nsolo")
        self.assertEqual(out["negative_prompt"], "bad hands")
        self.assertEqual(out["hr_prompt"], "detailed")
        self.assertEqual(out["hr_negative_prompt"], "blurry")
        self.assertEqual(out["prompt_id"], "a // b", "프롬프트가 아닌 키는 그대로")
        ad = out["alwayson_scripts"]["ADetailer"]["args"]
        self.assertEqual(ad[:2], [True, False])
        self.assertEqual((ad[2]["ad_prompt"], ad[2]["ad_negative_prompt"]), ("face", "x"))
        self.assertEqual((ad[3]["ad_prompt"], ad[3]["ad_negative_prompt"]), ("hand", "bad"))
        sam = out["alwayson_scripts"]["SAM3 Mask"]["args"][0]
        self.assertEqual(sam["sam3_prompt"], "face")
        self.assertEqual(sam["sam3_exclude_prompt"], "hair")
        self.assertEqual(sam["sam3_inpaint_prompt"], "smile")
        self.assertEqual(sam["sam3_negative_prompt"], "")
        self.assertEqual(sam["sam3_source_image"], "init // keep")
        self.assertEqual(out["alwayson_scripts"]["Other"], original["alwayson_scripts"]["Other"])
        self.assertIs(out["alwayson_scripts"]["Other"], original["alwayson_scripts"]["Other"])

    def test_titles_casefold_and_non_dict_args(self):
        payload = {"alwayson_scripts": {
            " adetailer ": {"args": [True, "ad_prompt // x", None, {"ad_prompt": "a // b"}]},
            "sam3 MASK": {"args": [{"sam3_prompt": "p"}, {"sam3_prompt": "q // only first arg"}]},
            "NegPiP": {"args": [True]},
            "Broken": "not a block",
        }}
        out = strip_request_comments(payload)
        args = out["alwayson_scripts"][" adetailer "]["args"]
        self.assertEqual(args[:3], [True, "ad_prompt // x", None])
        self.assertEqual(args[3], {"ad_prompt": "a"})
        self.assertIs(out["alwayson_scripts"]["sam3 MASK"], payload["alwayson_scripts"]["sam3 MASK"])

    def test_identity_when_nothing_to_strip(self):
        payload = {"prompt": "http://x, a//b", "negative_prompt": "", "alwayson_scripts": {
            "ADetailer": {"args": [True, False, {"ad_prompt": "face"}]},
            "SAM3 Mask": {"args": [{"sam3_prompt": "face"}]},
        }}
        self.assertIs(strip_request_comments(payload), payload)
        empty: dict = {}
        self.assertIs(strip_request_comments(empty), empty)
        self.assertIsNone(strip_request_comments(None))

    def test_key_sets(self):
        self.assertEqual(PROMPT_KEYS, ("prompt", "negative_prompt", "hr_prompt", "hr_negative_prompt"))
        self.assertEqual(set(ADETAILER_PROMPT_KEYS), {"ad_prompt", "ad_negative_prompt"})
        self.assertEqual(len(SAM3_PROMPT_KEYS), 4)


class GuardedKeysStaticTests(unittest.TestCase):
    """요청을 읽어 그래프·확장 인자를 만드는 곳이 새 프롬프트 키를 읽기 시작하면, 방어선도 그 키를 떼야 한다."""

    SOURCES = ("core/comfy_workflow_compiler.py", "core/sam3_args.py", "core/adetailer_args.py")

    @staticmethod
    def _read_prompt_keys(path: Path) -> set[str]:
        tree = ast.parse(path.read_text(encoding="utf-8"))
        keys: set[str] = set()
        for node in ast.walk(tree):
            literal = None
            if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                    and node.func.attr in ("get", "pop", "setdefault") and node.args):
                literal = node.args[0]
            elif isinstance(node, ast.Subscript) and isinstance(node.ctx, ast.Load):
                literal = node.slice
            if (isinstance(literal, ast.Constant) and isinstance(literal.value, str)
                    and literal.value.endswith("prompt")):
                keys.add(literal.value)
        return keys

    def test_every_read_prompt_key_is_guarded(self):
        seen: set[str] = set()
        for rel in self.SOURCES:
            keys = self._read_prompt_keys(ROOT / rel)
            seen |= keys
            with self.subTest(source=rel):
                self.assertLessEqual(keys, set(GUARDED_KEYS), f"방어선 밖의 프롬프트 키: {keys - set(GUARDED_KEYS)}")
        # 검사가 실제로 무언가를 본다(파일 이동·이름 변경으로 빈 검사가 되지 않게)
        self.assertIn("prompt", seen)
        self.assertIn("ad_prompt", seen)
        self.assertIn("sam3_inpaint_prompt", seen)


if __name__ == "__main__":
    unittest.main()
