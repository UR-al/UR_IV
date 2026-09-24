"""생성 통계 레코드 — UI 위젯이 아니라 실제 요청값으로 기록 (감사 #112).

예전엔 model_combo/width_input 을 읽어 고해상도 배율·Anima 해상도 가드·XYZ 축·Comfy 스냅샷
큐의 실제 요청과 다른 값이 남았고, width 칸이 비면 int('') 예외로 성공 레코드가 빠졌다.
"""
from __future__ import annotations

import unittest
from types import SimpleNamespace

from core.gen_stats import build_generation_record, request_meta_from_payload
from ui.generator_generation import GenerationMixin


class _Text:
    def __init__(self, value):
        self.value = value

    def text(self):
        return self.value

    currentText = text


class GenStatsRecordTests(unittest.TestCase):
    def test_request_meta_uses_payload_after_guard_and_override(self):
        # UI 는 1024x1024 인데 Anima 가드/hires 로 요청은 1536x1536, XYZ 가 모델을 override
        meta = request_meta_from_payload(
            "xyz-model.safetensors [abc]", {"width": 1536, "height": 1536, "seed": -1})
        self.assertEqual(meta, {"model": "xyz-model.safetensors [abc]",
                                "width": 1536, "height": 1536, "seed": -1})

    def test_request_meta_tolerates_missing_or_bad_values(self):
        meta = request_meta_from_payload(None, {"width": "", "height": "abc"})
        self.assertEqual(meta, {"model": "", "width": None, "height": None, "seed": None})
        self.assertEqual(request_meta_from_payload("m", None)["width"], None)

    def test_krea2_request_is_not_credited_to_the_hidden_checkpoint(self):
        # Krea2 모드에선 체크포인트 콤보가 숨겨져 있어도 currentText() 는 Standard 체크포인트 title 이다.
        # 실제 요청은 Krea2 ComfyUI 워크플로 → 통계 모델은 Krea2 라벨.
        payload = {"width": 1024, "height": 1024, "seed": 5, "_generation_family": "krea2"}
        meta = request_meta_from_payload("hidden-standard.safetensors [abc]", payload)
        self.assertEqual(meta, {"model": "Krea2", "width": 1024, "height": 1024, "seed": 5})
        self.assertEqual(payload["_generation_family"], "krea2")   # 워커가 쓸 키는 건드리지 않는다
        self.assertEqual(request_meta_from_payload("m", {"_generation_family": " KREA2 "})["model"], "Krea2")
        for family in ("standard", "", None):
            self.assertEqual(
                request_meta_from_payload("std.safetensors", {"_generation_family": family})["model"],
                "std.safetensors")

    def test_success_record_prefers_backend_seed(self):
        meta = request_meta_from_payload("m", {"width": 832, "height": 1216, "seed": -1})
        record = build_generation_record(success=True, duration_sec=3.2, request_meta=meta,
                                         gen_info={"seed": 12345})
        self.assertEqual(record, {"success": True, "duration_sec": 3.2, "model": "m",
                                  "seed": 12345, "width": 832, "height": 1216})

    def test_success_record_never_drops_on_empty_resolution(self):
        record = build_generation_record(success=True, duration_sec=1.0,
                                         request_meta={"model": "m", "width": None, "height": None,
                                                       "seed": 7}, gen_info={})
        self.assertEqual((record["width"], record["height"], record["seed"]), (0, 0, 7))

    def test_failure_record_keeps_design_of_model_only(self):
        record = build_generation_record(success=False, duration_sec=0.5,
                                         request_meta={"model": "m", "width": 1, "height": 2, "seed": 3})
        self.assertEqual(record, {"success": False, "duration_sec": 0.5, "model": "m"})

    def test_host_meta_prefers_saved_request_and_falls_back_to_widgets(self):
        saved = {"model": "req", "width": 1536, "height": 1536, "seed": 1}
        host = SimpleNamespace(_gen_request_meta=saved, model_combo=_Text("ui"),
                               width_input=_Text("1024"), height_input=_Text("1024"))
        self.assertIs(GenerationMixin._generation_stats_meta(host), saved)
        fallback = SimpleNamespace(model_combo=_Text("ui"), width_input=_Text(""), height_input=_Text("768"))
        self.assertEqual(GenerationMixin._generation_stats_meta(fallback),
                         {"model": "ui", "width": 0, "height": 768, "seed": None})


if __name__ == "__main__":
    unittest.main()
