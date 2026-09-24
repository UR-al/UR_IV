"""T2I 메인 프롬프트 대체 텍스트(Hires/ADetailer/SAM3) 떼기·SAM3 폴백 — core/main_prompt_overrides."""
from __future__ import annotations

import copy
import unittest

from core import main_prompt_overrides as mpo


def _payload():
    return {
        "prompt": "a castle, <lora:ink:0.7>", "negative_prompt": "lowres, text",
        "enable_hr": True, "hr_scale": 2.0, "hr_checkpoint_name": "other.safetensors",
        "hr_prompt": "1girl, red dress", "hr_negative_prompt": "bad",
        "alwayson_scripts": {
            "NegPiP": {"args": [True]},
            "ADetailer": {"args": [True, False,
                                   {"ad_model": "face_yolov8n.pt", "ad_prompt": "1girl", "ad_negative_prompt": "x"},
                                   {"ad_model": "hand_yolov8n.pt", "ad_prompt": "", "ad_confidence": 0.3}]},
            "SAM3 Mask": {"args": [{"sam3_prompt": "face", "sam3_exclude_prompt": "eyes",
                                    "sam3_inpaint_prompt": "T2I prompt", "sam3_negative_prompt": "T2I negative",
                                    "sam3_enable": True}]},
        },
    }


class ClearOverridesTests(unittest.TestCase):
    def test_script_names_match_the_payload_builders(self):
        from core.sam3_args import SCRIPT_SAM3

        self.assertEqual(mpo.SAM3_SCRIPT, SCRIPT_SAM3)
        self.assertEqual(mpo.ADETAILER_SCRIPT, "ADetailer")

    def test_clears_only_prompt_text_and_keeps_every_setting(self):
        payload = _payload()
        before = copy.deepcopy(payload)
        result = mpo.clear_main_prompt_overrides(payload)
        self.assertIs(result, payload)
        self.assertNotIn("hr_prompt", payload)
        self.assertNotIn("hr_negative_prompt", payload)
        for key in ("prompt", "negative_prompt", "enable_hr", "hr_scale", "hr_checkpoint_name"):
            self.assertEqual(payload[key], before[key])
        face, hand = payload["alwayson_scripts"]["ADetailer"]["args"][2:]
        self.assertEqual((face["ad_prompt"], face["ad_negative_prompt"], face["ad_model"]),
                         ("", "", "face_yolov8n.pt"))
        self.assertNotIn("ad_negative_prompt", hand)  # 없던 키는 만들지 않는다
        self.assertEqual(hand["ad_confidence"], 0.3)
        self.assertEqual(payload["alwayson_scripts"]["ADetailer"]["args"][:2], [True, False])
        sam = payload["alwayson_scripts"]["SAM3 Mask"]["args"][0]
        self.assertEqual((sam["sam3_inpaint_prompt"], sam["sam3_negative_prompt"]), ("", ""))
        self.assertEqual((sam["sam3_prompt"], sam["sam3_exclude_prompt"], sam["sam3_enable"]), ("face", "eyes", True))
        self.assertEqual(payload["alwayson_scripts"]["NegPiP"], before["alwayson_scripts"]["NegPiP"])

    def test_tolerates_payloads_without_extensions_or_with_odd_shapes(self):
        for payload in ({}, {"prompt": "x"}, {"alwayson_scripts": None}, {"alwayson_scripts": {"ADetailer": None}},
                        {"alwayson_scripts": {"ADetailer": {"args": "bad"}, "SAM3 Mask": {"args": [None, 3]}}}):
            with self.subTest(payload=payload):
                expected = copy.deepcopy(payload)
                mpo.clear_main_prompt_overrides(payload)
                self.assertEqual(payload, expected)


class Sam3FallbackTests(unittest.TestCase):
    def test_fills_only_empty_sam3_prompts_from_the_final_payload_text(self):
        payload = mpo.clear_main_prompt_overrides(_payload())
        mpo.fill_sam3_prompt_fallback(payload)
        sam = payload["alwayson_scripts"]["SAM3 Mask"]["args"][0]
        self.assertEqual(sam["sam3_inpaint_prompt"], "a castle, <lora:ink:0.7>")
        self.assertEqual(sam["sam3_negative_prompt"], "lowres, text")
        # ADetailer/Hires 는 비운 채로 둔다(백엔드가 빈 값 = 메인 프롬프트)
        self.assertEqual(payload["alwayson_scripts"]["ADetailer"]["args"][2]["ad_prompt"], "")
        self.assertNotIn("hr_prompt", payload)

    def test_keeps_explicit_sam3_prompts_and_skips_empty_main_text(self):
        payload = _payload()
        mpo.fill_sam3_prompt_fallback(payload)
        sam = payload["alwayson_scripts"]["SAM3 Mask"]["args"][0]
        self.assertEqual((sam["sam3_inpaint_prompt"], sam["sam3_negative_prompt"]), ("T2I prompt", "T2I negative"))
        empty = {"prompt": "", "negative_prompt": "",
                 "alwayson_scripts": {"SAM3 Mask": {"args": [{"sam3_inpaint_prompt": ""}]}}}
        mpo.fill_sam3_prompt_fallback(empty)
        self.assertEqual(empty["alwayson_scripts"]["SAM3 Mask"]["args"][0], {"sam3_inpaint_prompt": ""})

    def test_matches_the_sam3_args_build_state_fallback_rule(self):
        from core.sam3_args import build_state

        state = build_state({}, prompt="a castle", negative_prompt="lowres")
        payload = {"prompt": "a castle", "negative_prompt": "lowres",
                   "alwayson_scripts": {"SAM3 Mask": {"args": [dict(state, sam3_inpaint_prompt="",
                                                                    sam3_negative_prompt="")]}}}
        mpo.fill_sam3_prompt_fallback(payload)
        filled = payload["alwayson_scripts"]["SAM3 Mask"]["args"][0]
        self.assertEqual(filled["sam3_inpaint_prompt"], state["sam3_inpaint_prompt"])
        self.assertEqual(filled["sam3_negative_prompt"], state["sam3_negative_prompt"])


if __name__ == "__main__":
    unittest.main()
