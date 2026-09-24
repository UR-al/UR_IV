"""ADetailer 슬롯 필드 — Vue(AdSlotFields) 와 Python 프록시(_ad_slot)의 키가 같은지 (App.vue 분할 ④).

예전 App.vue 는 슬롯 1·2 의 23개 필드를 `_ad_s1_` / `_ad_s2_` 만 바꿔 두 번 베껴 적었다. 이제 슬롯 하나를
components/params/AdSlotFields.vue 가 그리고 키 목록은 utils/adSlotKeys.ts 의 AD_SLOT_FIELDS 한 곳에 있다.
그 목록이 Python `ui/generator_ui_setup.py` 의 `_ad_slot(prefix)` 가 만드는 위젯 id 와 어긋나면 예외 없이
그 칸만 저장·생성에서 빠진다 — 이 저장소의 단골 실패 방식이라 정적으로 잡는다(Qt 비의존).
"""
from __future__ import annotations

import pathlib
import re
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]
KEYS_TS = ROOT / "frontend" / "src" / "utils" / "adSlotKeys.ts"
SLOT_VUE = ROOT / "frontend" / "src" / "components" / "params" / "AdSlotFields.vue"
CARD_VUE = ROOT / "frontend" / "src" / "components" / "params" / "AdetailerCard.vue"
UI_SETUP = ROOT / "ui" / "generator_ui_setup.py"


def _ts_fields() -> list[str]:
    text = KEYS_TS.read_text(encoding="utf-8")
    body = text.split("export const AD_SLOT_FIELDS = [", 1)[1].split("] as const", 1)[0]
    return re.findall(r"'([a-z_]+)'", body)


def _python_fields() -> set[str]:
    text = UI_SETUP.read_text(encoding="utf-8")
    body = text.split("def _ad_slot(prefix):", 1)[1].split("self.s1_widgets", 1)[0]
    return set(re.findall(r"f'\{prefix\}_([a-z_]+)'", body))


class AdSlotFieldsContractTests(unittest.TestCase):
    def test_vue_fields_match_the_python_proxies(self):
        ts = _ts_fields()
        self.assertEqual(len(ts), len(set(ts)), "AD_SLOT_FIELDS 에 중복이 있다")
        self.assertEqual(len(ts), 23)
        self.assertEqual(set(ts), _python_fields())

    def test_slot_component_binds_every_field(self):
        text = SLOT_VUE.read_text(encoding="utf-8")
        template = text.split("<template>", 1)[1].split("<script", 1)[0]
        used = set(re.findall(r"\b(?:k|isOn|setOn)\('([a-z_]+)'", template))
        self.assertEqual(used, set(_ts_fields()))
        # 슬롯 번호가 키에 박혀 있으면 슬롯 2 가 슬롯 1 값을 쓴다
        self.assertNotRegex(template, r"_ad_s[12]_")

    def test_card_renders_both_slots(self):
        text = CARD_VUE.read_text(encoding="utf-8")
        self.assertEqual(re.findall(r'<AdSlotFields :slot-no="(\d)"', text), ["1", "2"])
        for group in ("ad_slot1_group", "ad_slot2_group"):
            self.assertIn(group, UI_SETUP.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
