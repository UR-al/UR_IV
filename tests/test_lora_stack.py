"""LoRA 스택 → 생성 텍스트 변환 · 단위 정규화 테스트.
(단일 소스: 생성 LoRA 는 _vue_lora_entries(배율)에서만 파생 — 포맷은 Vue 와 같은 소수 2자리)"""
import unittest
from core.lora_stack import (
    UNIT_AUTO, UNIT_MULTIPLIER, UNIT_PERCENT, append_lora_stack_to_prompt, build_lora_text,
    detect_weight_unit, normalize_lora_entries, prompt_lora_names, to_percent_entries,
)


class TestBuildLoraText(unittest.TestCase):
    def test_basic_format_weight_percent_to_decimal(self):
        # weight는 정수%로 저장 → /100, 소수 2자리 (App.vue doGenerate와 동일)
        out = build_lora_text([{"name": "styleA", "weight": 80, "enabled": True}])
        self.assertEqual(out, "<lora:styleA:0.80>")

    def test_multiple_joined_by_comma(self):
        out = build_lora_text([
            {"name": "a", "weight": 100, "enabled": True},
            {"name": "b", "weight": 65, "enabled": True},
        ])
        self.assertEqual(out, "<lora:a:1.00>, <lora:b:0.65>")

    def test_disabled_excluded(self):
        out = build_lora_text([
            {"name": "a", "weight": 100, "enabled": False},
            {"name": "b", "weight": 50, "enabled": True},
        ])
        self.assertEqual(out, "<lora:b:0.50>")

    def test_missing_name_skipped(self):
        out = build_lora_text([{"name": "", "weight": 100, "enabled": True}])
        self.assertEqual(out, "")

    def test_enabled_defaults_true(self):
        # enabled 키 없으면 활성으로 간주
        out = build_lora_text([{"name": "a", "weight": 90}])
        self.assertEqual(out, "<lora:a:0.90>")

    def test_empty_and_garbage_inputs(self):
        self.assertEqual(build_lora_text([]), "")
        self.assertEqual(build_lora_text(None), "")
        self.assertEqual(build_lora_text(["not a dict", 5]), "")

    def test_bad_weight_falls_back(self):
        out = build_lora_text([{"name": "a", "weight": "xx", "enabled": True}])
        self.assertEqual(out, "<lora:a:1.00>")

    def test_multiplier_unit_is_not_divided_again(self):
        # set_lora_stack 엔트리(0.85)를 퍼센트로 다시 /100 하면 <lora:x:0.01> 이 된다.
        out = build_lora_text([{"name": "a", "weight": 0.85, "enabled": True}], unit=UNIT_MULTIPLIER)
        self.assertEqual(out, "<lora:a:0.85>")

    def test_nan_and_infinite_weights_fall_back(self):
        out = build_lora_text([{"name": "a", "weight": float("nan")},
                               {"name": "b", "weight": float("inf")}], unit=UNIT_MULTIPLIER)
        self.assertEqual(out, "<lora:a:1.00>, <lora:b:1.00>")


class TestLoraWeightUnits(unittest.TestCase):
    def test_percent_entries_normalize_to_multiplier(self):
        entries = normalize_lora_entries(
            [{"name": "a", "weight": 85, "enabled": True, "triggerWords": ["tw"]}], unit=UNIT_PERCENT)
        self.assertEqual(entries, [{"name": "a", "weight": 0.85, "enabled": True, "triggerWords": ["tw"]}])

    def test_normalize_does_not_mutate_input_and_fills_defaults(self):
        source = [{"name": "a", "weight": 30}, "junk", {"weight": "bad", "enabled": False, "triggerWords": "x"}]
        entries = normalize_lora_entries(source, unit=UNIT_PERCENT)
        self.assertEqual(source[0]["weight"], 30)
        self.assertEqual(entries, [
            {"name": "a", "weight": 0.3, "enabled": True, "triggerWords": []},
            {"name": "", "weight": 1.0, "enabled": False, "triggerWords": []},
        ])

    def test_unknown_unit_is_rejected(self):
        with self.assertRaises(ValueError):
            normalize_lora_entries([], unit="permille")

    def test_legacy_profile_detection_is_list_wide(self):
        # 실제 ANIMA.json(version 1) 값 — 퍼센트로 저장돼 적용 시 100배가 되던 목록
        anima = [{"name": n, "weight": w, "enabled": True}
                 for n, w in (("a", 95), ("b", 10), ("c", 100), ("d", 100), ("e", 30))]
        self.assertEqual(detect_weight_unit(anima), UNIT_PERCENT)
        self.assertEqual(
            [e["weight"] for e in normalize_lora_entries(anima, unit=UNIT_AUTO)],
            [0.95, 0.1, 1.0, 1.0, 0.3],
        )
        # 5% 이하 LoRA 나 음수가 섞여도 목록에 |w|>3 이 하나라도 있으면 전체가 퍼센트
        mixed = [{"name": "a", "weight": 2}, {"name": "b", "weight": -40}]
        self.assertEqual([e["weight"] for e in normalize_lora_entries(mixed, unit=UNIT_AUTO)], [0.02, -0.4])
        # 배율로 저장된 옛 목록은 그대로
        multiplier = [{"name": "a", "weight": 0.8}, {"name": "b", "weight": -1.5}, {"name": "c", "weight": 3}]
        self.assertEqual(detect_weight_unit(multiplier), UNIT_MULTIPLIER)
        self.assertEqual([e["weight"] for e in normalize_lora_entries(multiplier, unit=UNIT_AUTO)], [0.8, -1.5, 3.0])

    def test_to_percent_entries_round_trips_vue_storage(self):
        entries = normalize_lora_entries([{"name": "a", "weight": 65}, {"name": "b", "weight": 5}], unit=UNIT_PERCENT)
        self.assertEqual([e["weight"] for e in to_percent_entries(entries)], [65, 5])


class TestAppendLoraStackToPrompt(unittest.TestCase):
    STACK = [
        {"name": "styleA", "weight": 0.8, "enabled": True},
        {"name": "off", "weight": 1.0, "enabled": False},
        {"name": "Detail", "weight": 0.5, "enabled": True},
    ]

    def test_appends_only_enabled_entries(self):
        self.assertEqual(
            append_lora_stack_to_prompt("1girl", self.STACK),
            "1girl, <lora:styleA:0.80>, <lora:Detail:0.50>",
        )

    def test_empty_or_all_disabled_stack_adds_nothing(self):
        # 감사 #2: 전부 끄면 LoRA 없는 프롬프트여야 한다(옛 미러가 계속 붙던 버그)
        self.assertEqual(append_lora_stack_to_prompt("1girl", []), "1girl")
        self.assertEqual(append_lora_stack_to_prompt("1girl", None), "1girl")
        disabled = [dict(e, enabled=False) for e in self.STACK]
        self.assertEqual(append_lora_stack_to_prompt("1girl", disabled), "1girl")

    def test_prompt_lora_with_same_name_is_not_duplicated(self):
        self.assertEqual(
            append_lora_stack_to_prompt("1girl, <lora:stylea:0.9>", self.STACK),
            "1girl, <lora:stylea:0.9>, <lora:Detail:0.50>",
        )
        self.assertEqual(prompt_lora_names("<LORA:A b:1>, <lora:c>"), {"a b", "c"})

    def test_empty_prompt_gets_only_lora_text(self):
        self.assertEqual(append_lora_stack_to_prompt("", self.STACK[:1]), "<lora:styleA:0.80>")


if __name__ == "__main__":
    unittest.main()
