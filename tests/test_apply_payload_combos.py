"""큐 항목·'이 설정으로 생성'의 샘플러·스케줄러 적용(GeneratorMainUI._apply_payload_to_ui, 계획 X3a).

Forge infotext 의 'Schedule type' 은 표기(label)이고 Forge 스케줄러 콤보 값은 이름(name)이다. 예전에는 정확히 같은
항목만 고르는 ``ComboBoxProxy.setText`` 를 써서 'Karras'·'SGM Uniform'·'Laplace'·'CosineExponential blend' 가 오면
Vue 에만 그 값을 보이고 Python 은 이전 선택으로 생성했다. 이제 프리셋 불러오기와 같은 매처(정확 → 접기 → 별칭·Forge
표기, core.preset_choice)로 고르고, 찾지 못하면 현재 선택을 두고 경고 토스트를 띄운다. Qt·네트워크 없이 돈다.
"""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from PIL import Image, PngImagePlugin

from tests.test_comfy_metadata_actions import _Harness
from ui.generation_settings_apply import apply_sampler_scheduler
from ui.image_metadata_actions import start_generation_from_metadata

FORGE_SAMPLERS = ["Euler", "Euler a", "DPM++ 2M", "DPM++ 2M SDE"]
FORGE_SCHEDULERS = ["automatic", "karras", "exponential", "sgm_uniform", "simple", "beta", "flow_match",
                    "cosine", "cosine_exponential", "phi", "laplace", "karras_dynamic", "custom"]


class _Combo:
    """ComboBoxProxy 가 이 경로에서 쓰는 면만 — 항목·선택 index·미뤄 둔 값(setText fallback)."""

    def __init__(self, items, current=""):
        self._items = list(items)
        self._index = self._items.index(current) if current in self._items else -1
        self.fallback = ""

    def currentText(self):
        return self._items[self._index] if 0 <= self._index < len(self._items) else self.fallback

    text = currentText

    def setCurrentIndex(self, index):
        self._index = index
        self.fallback = ""

    def setText(self, value):
        if value in self._items:
            self._index = self._items.index(value)
        else:
            self.fallback = value


def _host(*, sampler_items=FORGE_SAMPLERS, scheduler_items=FORGE_SCHEDULERS, sampler="Euler", scheduler="simple"):
    host = _Harness()
    host.sampler_combo = _Combo(sampler_items, sampler)
    host.scheduler_combo = _Combo(scheduler_items, scheduler)
    return host


class ApplyPayloadComboTests(unittest.TestCase):
    def test_forge_schedule_type_labels_select_the_name_valued_item(self):
        for label, name in (("Karras", "karras"), ("SGM Uniform", "sgm_uniform"), ("Laplace", "laplace"),
                            ("CosineExponential blend", "cosine_exponential"), ("Karras Dynamic", "karras_dynamic"),
                            ("FlowMatchEulerDiscrete", "flow_match"), ("custom", "custom")):
            with self.subTest(label=label):
                host = _host()
                host._apply_payload_to_ui({"scheduler": label})
                self.assertEqual(host.scheduler_combo.currentText(), name)
                self.assertEqual(host.scheduler_combo.fallback, "")
                self.assertEqual(host.vue_bridge.showNotification.calls, [])

    def test_comfy_names_select_the_forge_items(self):
        host = _host()
        host._apply_payload_to_ui({"sampler_name": "euler_ancestral", "scheduler": "sgm_uniform"})
        self.assertEqual((host.sampler_combo.currentText(), host.scheduler_combo.currentText()),
                         ("Euler a", "sgm_uniform"))
        host._apply_payload_to_ui({"sampler_name": "dpmpp_2m_sde", "scheduler": "laplace"})
        self.assertEqual((host.sampler_combo.currentText(), host.scheduler_combo.currentText()),
                         ("DPM++ 2M SDE", "laplace"))
        self.assertEqual(host.vue_bridge.showNotification.calls, [])

    def test_unknown_values_keep_the_current_choice_and_warn_once(self):
        host = _host(sampler="DPM++ 2M", scheduler="beta")
        host._apply_payload_to_ui({"sampler_name": "Restart", "scheduler": "Align Your Steps 32", "steps": 30})
        self.assertEqual((host.sampler_combo.currentText(), host.scheduler_combo.currentText()), ("DPM++ 2M", "beta"))
        self.assertEqual(host.steps_input.text(), "30")             # 나머지 칸은 그대로 적용된다
        [(level, message)] = host.vue_bridge.showNotification.calls
        self.assertEqual(level, "warning")
        self.assertIn("샘플러 'Restart'", message)
        self.assertIn("'DPM++ 2M' 유지", message)
        self.assertIn("스케줄러 'Align Your Steps 32'", message)
        self.assertIn("'beta' 유지", message)

    def test_exact_queue_values_and_empty_values_are_unchanged(self):
        host = _host()
        host._apply_payload_to_ui({"sampler_name": "DPM++ 2M", "scheduler": "karras"})
        self.assertEqual((host.sampler_combo.currentText(), host.scheduler_combo.currentText()), ("DPM++ 2M", "karras"))
        host._apply_payload_to_ui({"sampler_name": "", "scheduler": None})
        self.assertEqual((host.sampler_combo.currentText(), host.scheduler_combo.currentText()), ("DPM++ 2M", "karras"))
        self.assertEqual(host.vue_bridge.showNotification.calls, [])

    def test_an_empty_list_defers_like_before(self):
        """연결 전(목록이 빈 콤보)에는 예전처럼 setText 로 미룬다 — 목록이 오면 프록시가 그 값을 고른다."""
        host = _host(sampler_items=[], scheduler_items=[])
        self.assertEqual(apply_sampler_scheduler(host, "Euler a", "Karras"), [])
        self.assertEqual((host.sampler_combo.fallback, host.scheduler_combo.fallback), ("Euler a", "Karras"))

    def test_generate_from_forge_metadata_uses_the_label_mapped_scheduler(self):
        """'이 설정으로 생성' — Forge PNG 의 Schedule type 표기가 Forge 이름 콤보에서 골라져 그대로 생성된다."""
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "forge.png"
            metadata = PngImagePlugin.PngInfo()
            metadata.add_text("parameters", "a cat\nNegative prompt: bad\nSteps: 28, Sampler: Euler a, "
                                            "Schedule type: CosineExponential blend, CFG scale: 5, Seed: 7, "
                                            "Size: 832x1216")
            Image.new("RGB", (8, 8), "blue").save(path, pnginfo=metadata)
            host = _host()
            info = json.loads(host.vue_bridge.getImageExif(str(path)))
        self.assertTrue(start_generation_from_metadata(host, info))
        [started] = host.started
        self.assertEqual((started["sampler"], started["scheduler"]), ("Euler a", "cosine_exponential"))
        self.assertEqual((started["steps"], started["seed"]), ("28", "7"))
        self.assertEqual(host.vue_bridge.showNotification.calls, [])


if __name__ == "__main__":
    unittest.main()
