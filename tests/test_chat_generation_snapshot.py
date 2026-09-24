"""Chat captures the complete T2I request without changing that tab's controls."""
import unittest
from types import SimpleNamespace
from unittest import mock

from ui.generator_generation import GenerationMixin


class ReadOnlyWidget:
    def __init__(self, value):
        self.value = value

    def text(self):
        return str(self.value)

    currentText = text
    toPlainText = text

    def isChecked(self):
        return bool(self.value)

    def setText(self, _value):
        raise AssertionError("Chat must not write T2I controls")


class SnapshotHost(GenerationMixin):
    def __init__(self):
        values = {
            "model_combo": "Anima-3.8B-v1.1.safetensors", "generation_family_combo": "Anima",
            "random_res_check": True, "width_input": "768", "height_input": "1024",
            "neg_prompt_text": "__negative__", "total_prompt_display": "T2I remains unchanged",
            "sampler_combo": "Euler", "scheduler_combo": "Beta", "steps_input": "28",
            "cfg_input": "5", "seed_input": "123", "shift_input": "1.2",
            "vae_main_combo": "qwen_image_vae.safetensors", "te_main_input": "qwen35_4b.safetensors",
            "hires_options_group": True, "hires_scale_input": "1.5",
            "upscaler_combo": "Latent", "hires_steps_input": "10", "hires_denoising_input": "0.4",
            "hires_cfg_input": "4", "hires_checkpoint_combo": "Use same checkpoint",
            "hires_sampler_combo": "Use same sampler", "hires_scheduler_combo": "Use same scheduler",
            "hires_prompt_text": "", "hires_neg_prompt_text": "", "negpip_group": True,
        }
        for name, value in values.items():
            setattr(self, name, ReadOnlyWidget(value))
        self.random_resolutions = [(1024, 768, "landscape")]
        # 와일드카드 ON/OFF 의 주인(core/prompt_settings_extras) — 예전 숨은 SettingsTab 체크박스 자리
        from core.prompt_settings_extras import PromptSettingsExtras
        self.prompt_settings_extras = PromptSettingsExtras(wildcard_enabled=True)
        # 생성 LoRA 의 단일 소스 — set_lora_stack 과 같은 배율 단위
        self._vue_lora_entries = [
            {"name": "style", "weight": 0.7, "enabled": True, "triggerWords": []},
            {"name": "disabled", "weight": 1.0, "enabled": False, "triggerWords": []},
        ]
        self.postprocess = {"ADetailer": {"args": [True, {"ad_model": "face.pt"}]} }

    def _apply_postprocess_chain(self, payload):
        payload["alwayson_scripts"].update(self.postprocess)


class ChatSnapshotTests(unittest.TestCase):
    def test_captures_controls_lora_modules_and_postprocess_without_mutation(self):
        host = SnapshotHost()
        with mock.patch("ui.generator_generation.resolve_file_wildcards") as files, \
             mock.patch("ui.generator_generation.process_wildcards") as wildcards, \
             mock.patch("core.standard_hooks.run_pipeline_on_text") as pipeline:
            model, payload = host._chat_generation_snapshot("__subject__")
        self.assertEqual(model, "Anima-3.8B-v1.1.safetensors")
        self.assertEqual(payload["prompt"], "__subject__, <lora:style:0.70>")
        self.assertEqual(payload["negative_prompt"], "__negative__")
        self.assertEqual((payload["width"], payload["height"]), (1024, 768))
        self.assertEqual(payload["forge_additional_modules"], ["qwen_image_vae.safetensors", "qwen35_4b.safetensors"])
        self.assertEqual(payload["hr_additional_modules"], ["Use same choices"])
        self.assertTrue(payload["enable_hr"])
        self.assertEqual(payload["alwayson_scripts"]["NegPiP"], {"args": [True]})
        self.assertEqual(payload["_chat_deferred_prompt"], {"wildcards": True})
        self.assertEqual(host.total_prompt_display.value, "T2I remains unchanged")
        self.assertEqual(host.width_input.value, "768")
        payload["alwayson_scripts"]["ADetailer"]["args"][1]["ad_model"] = "changed"
        self.assertEqual(host.postprocess["ADetailer"]["args"][1]["ad_model"], "face.pt")
        files.assert_not_called()
        wildcards.assert_not_called()
        pipeline.assert_not_called()

    def test_existing_prompt_lora_is_not_duplicated(self):
        _, payload = SnapshotHost()._chat_generation_snapshot("portrait, <lora:style:0.9>")
        self.assertEqual(payload["prompt"].count("<lora:style:"), 1)

    def test_all_disabled_or_empty_stack_sends_no_lora(self):
        # 감사 #2: 전부 끄거나 지운 스택이 예전 미러(_vue_lora_text) 때문에 계속 붙던 버그
        host = SnapshotHost()
        host._vue_lora_text = "<lora:stale:1.00>"   # 옛 미러가 남아 있어도 읽지 않는다
        for entry in host._vue_lora_entries:
            entry["enabled"] = False
        _, payload = host._chat_generation_snapshot("portrait")
        self.assertEqual(payload["prompt"], "portrait")
        host._vue_lora_entries = []
        _, payload = host._chat_generation_snapshot("portrait")
        self.assertEqual(payload["prompt"], "portrait")

    def test_krea_route_keeps_family_and_does_not_add_anima_lora(self):
        host = SnapshotHost()
        host.generation_family_combo = ReadOnlyWidget("KREA2")
        _, payload = host._chat_generation_snapshot("portrait")
        self.assertEqual(payload["_generation_family"], "krea2")
        self.assertEqual(payload["prompt"], "portrait")

    def test_no_selected_model_is_actionable_error(self):
        host = SnapshotHost()
        host.model_combo = ReadOnlyWidget("")
        with self.assertRaisesRegex(ValueError, "모델"):
            host._chat_generation_snapshot("portrait")

    def test_invalid_controls_are_rejected_before_any_worker(self):
        host = SnapshotHost()
        host.steps_input = ReadOnlyWidget("0")
        with self.assertRaises(ValueError):
            host._chat_generation_snapshot("portrait")


if __name__ == "__main__":
    unittest.main()
