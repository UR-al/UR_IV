"""'이 설정으로 생성'(ui/image_metadata_actions.start_generation_from_metadata, 계획 A-R1·X3b) — sam-extra Extra
Schedulers·Extra Samplers 값을 그 스케줄러·샘플러와 **함께** 되살린다.

규칙: 이미지의 메인('Schedule type'·'Sampler')·Hires('Hires schedule type'·'Hires sampler')가 그 값을 읽을 때만 —
custom(식·목록과 모드)·Laplace(μ/β)·Flow Cosmos rho7·Dynamic(ρ·σ̃ max·σ̃ min), ER SDE 셋(차수)과 Reverse-time·Tunable
(η·잡음 구간 — Reverse-time 이미지의 Forge 'Eta' 대체 포함). 쓰지 않으면 키가 있어도 칸을 건드리지 않는다(글자도 그대로).
값은 생성이 시작되기 전에 칸에 있다. 실제 PNG(PIL)·실제 메타데이터 파서·실제 액션 분기를 거친다. Qt·네트워크·GPU 없음.
"""
from __future__ import annotations

import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

from PIL import Image, PngImagePlugin

from core import extra_samplers as xsa
from core import extra_schedulers as xs
from tests.test_apply_payload_combos import _Combo
from tests.test_comfy_metadata_actions import _Harness
from ui import extra_samplers_ui as xsamp_ui
from ui import extra_schedulers_ui as xsched_ui
from ui.image_metadata_actions import restore_sampling_values, start_generation_from_metadata

RT, ODE, TUN = xsa.LABEL_ER_SDE_REVERSE_TIME, xsa.LABEL_ER_SDE_ODE, xsa.LABEL_ER_SDE_TUNABLE
SAMPLERS = ["Euler", "Euler a", "DPM++ 2M", RT, ODE, TUN]
SCHEDULERS = ["automatic", "karras", "simple", "laplace", "custom", "karras_dynamic", "flow_cosmos_rho7",
              "flow_cosmos_dynamic"]
# 사용자가 칸에 둔 글자 — 정규화 전 모양(앞뒤 공백·'3.0')도 그대로 남아야 한다
XSCHED_TEXT = {"custom_mode": "sigmas", "custom_expression": "  M * 0.5 ", "custom_sigmas": "[2, 1, 0]",
               "laplace_mu": "-1.0", "laplace_beta": "4", "flow_cosmos_rho": "3.0", "flow_cosmos_sigma_max": "500",
               "flow_cosmos_sigma_min": "0.01"}
XSAMP_TEXT = {"max_stage": "1", "eta": "4.50", "noise_window": "1", "noise_start": "0.3", "noise_end": "0.7"}


class _Proxy:
    """LineEditProxy 의 이 경로 면 — 글자와 setText 호출 기록."""

    def __init__(self, value):
        self.value = value
        self.writes = []

    def text(self):
        return self.value

    def setText(self, value):
        self.writes.append(value)
        self.value = value


class _Host(_Harness):
    """하네스 + 샘플러·스케줄러 콤보(Forge 이름 목록) + 두 값 칸. 생성이 시작될 때의 칸 값을 같이 적는다."""

    def __init__(self, *, sampler="Euler", scheduler="simple"):
        super().__init__()
        self.sampler_combo = _Combo(SAMPLERS, sampler)
        self.scheduler_combo = _Combo(SCHEDULERS, scheduler)
        self.extra_schedulers_widgets = {key: _Proxy(XSCHED_TEXT[key]) for key in xs.WIDGET_KEYS}
        self.extra_samplers_widgets = {key: _Proxy(XSAMP_TEXT[key]) for key in xsa.WIDGET_KEYS}

    def start_generation(self):
        super().start_generation()
        self.started[-1].update(xsched=xsched_ui.current_settings(self), xsamp=xsamp_ui.current_settings(self),
                                xsched_text=self.texts(self.extra_schedulers_widgets),
                                xsamp_text=self.texts(self.extra_samplers_widgets))

    @staticmethod
    def texts(widgets):
        return {key: proxy.text() for key, proxy in widgets.items()}


USER_XSCHED = xs.parse_settings(XSCHED_TEXT)
USER_XSAMP = xsa.parse_settings(XSAMP_TEXT)


class GenerateFromMetadataRestoreTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.path = Path(temp.name) / "forge.png"

    def info(self, host, line):
        """Forge 가 쓰는 모양의 PNG(parameters 청크) → getImageExif 결과."""
        metadata = PngImagePlugin.PngInfo()
        metadata.add_text("parameters", "a cat\nNegative prompt: bad\n" + line)
        Image.new("RGB", (8, 8), "blue").save(self.path, pnginfo=metadata)
        return json.loads(host.vue_bridge.getImageExif(str(self.path)))

    def generate(self, host, line):
        self.assertTrue(start_generation_from_metadata(host, self.info(host, line)))
        [started] = host.started
        return started

    def assert_untouched(self, host, started, *, xsched=True, xsamp=True):
        if xsched:
            self.assertEqual(started["xsched_text"], XSCHED_TEXT)
            self.assertEqual([p.writes for p in host.extra_schedulers_widgets.values()], [[]] * len(xs.WIDGET_KEYS))
        if xsamp:
            self.assertEqual(started["xsamp_text"], XSAMP_TEXT)
            self.assertEqual([p.writes for p in host.extra_samplers_widgets.values()], [[]] * len(xsa.WIDGET_KEYS))

    def test_laplace_and_reverse_time_values_come_back_with_their_names(self):
        host = _Host()
        started = self.generate(host, (
            "Steps: 28, Sampler: ER SDE (Reverse-time), Schedule type: Laplace, CFG scale: 5, Seed: 7, Size: 832x1216, "
            "Laplace mu: 0.25, Laplace beta: 1.5, ER SDE max stage: 2, ER SDE eta: 0.5, ER SDE noise window: 0.1-0.9"))
        self.assertEqual((started["sampler"], started["scheduler"]), (RT, "laplace"))
        self.assertEqual(started["xsched"], replace(USER_XSCHED, laplace_mu=0.25, laplace_beta=1.5))
        # μ/β 칸만 썼다 — custom·Flow Cosmos 칸의 글자는 그대로(정규화도 하지 않는다)
        self.assertEqual({k: v for k, v in started["xsched_text"].items() if k not in ("laplace_mu", "laplace_beta")},
                         {k: v for k, v in XSCHED_TEXT.items() if k not in ("laplace_mu", "laplace_beta")})
        self.assertEqual(started["xsamp"], xsa.ExtraSamplersSettings(max_stage=2, eta=0.5, noise_window=True,
                                                                     noise_start=0.1, noise_end=0.9))
        self.assertEqual(host.vue_bridge.showNotification.calls, [])

    def test_custom_expression_and_sigma_list(self):
        host = _Host()
        started = self.generate(host, "Steps: 20, Sampler: Euler a, Schedule type: custom, CFG scale: 5, Seed: 1, "
                                      "Size: 512x512, Custom scheduler expression: m + (M - m) * (1 - x) ** 3")
        self.assertEqual(started["scheduler"], "custom")
        self.assertEqual((started["xsched"].custom_mode, started["xsched"].custom_expression),
                         (xs.MODE_EXPRESSION, "m + (M - m) * (1 - x) ** 3"))
        self.assertEqual(started["xsched"].custom_sigmas, USER_XSCHED.custom_sigmas)
        self.assertEqual(started["xsched"].laplace_mu, USER_XSCHED.laplace_mu)
        self.assert_untouched(host, started, xsched=False)

        host = _Host()
        started = self.generate(host, 'Steps: 20, Sampler: Euler a, Schedule type: custom, CFG scale: 5, Seed: 1, '
                                      'Size: 512x512, Custom scheduler sigmas: "[1.0, 0.5, 0.25, 0.0]"')
        self.assertEqual(started["xsched"], xs.ExtraSchedulerSettings(
            xs.MODE_SIGMAS, USER_XSCHED.custom_expression, "[1.0, 0.5, 0.25, 0.0]", -1.0, 4.0, 3.0, 500.0, 0.01))
        self.assertEqual(started["xsched_text"]["custom_expression"], "  M * 0.5 ")

    def test_flow_cosmos_values_for_either_label(self):
        for label, name in (("Flow Cosmos rho7", "flow_cosmos_rho7"), ("Flow Cosmos Dynamic", "flow_cosmos_dynamic")):
            with self.subTest(label=label):
                host = _Host()
                started = self.generate(host, (
                    f"Steps: 30, Sampler: Euler, Schedule type: {label}, CFG scale: 4, Seed: 3, Size: 1024x1024, "
                    "Flow Cosmos rho: 5.5, Flow Cosmos sigma max: 240.0, Flow Cosmos sigma min: 0.006, Laplace mu: 2"))
                self.assertEqual(started["scheduler"], name)
                self.assertEqual(started["xsched"], xs.ExtraSchedulerSettings(
                    "sigmas", "M * 0.5", "[2, 1, 0]", -1.0, 4.0, 5.5, 240.0, 0.006))
                self.assertEqual(started["xsched_text"]["laplace_mu"], "-1.0")
                self.assert_untouched(host, started, xsched=False)

    def test_hires_pass_values_and_the_forge_eta_fallback(self):
        """메인은 값을 읽지 않고 Hires 가 읽는다 — Hires 칸은 이 경로가 되살리지 않지만 값은 이미지를 따른다(Forge 붙여 넣기와
        같은 범위). Reverse-time Hires 이미지의 Forge 'Eta' 가 η 가 된다."""
        host = _Host()
        started = self.generate(host, (
            "Steps: 30, Sampler: Euler a, Schedule type: Karras, CFG scale: 4, Seed: 3, Size: 1024x1024, Eta: 0.4, "
            "Hires sampler: ER SDE (Reverse-time), Hires schedule type: Flow Cosmos Dynamic, Flow Cosmos rho: 4.5, "
            "Flow Cosmos sigma max: 80.0, Flow Cosmos sigma min: 0.002"))
        self.assertEqual((started["sampler"], started["scheduler"]), ("Euler a", "karras"))
        self.assertEqual((started["xsched"].flow_cosmos_rho, started["xsched"].flow_cosmos_sigma_max,
                          started["xsched"].flow_cosmos_sigma_min), (4.5, 80.0, 0.002))
        self.assertEqual(started["xsamp"], xsa.ExtraSamplersSettings(eta=0.4))

    def test_tunable_does_not_read_the_forge_eta_and_ode_reads_only_the_stage(self):
        host = _Host()
        started = self.generate(host, "Steps: 20, Sampler: ER SDE (Tunable), Schedule type: Karras, CFG scale: 5, "
                                      "Seed: 1, Size: 512x512, Eta: 0.7")
        self.assertEqual(started["xsamp"], xsa.APP_DEFAULTS)
        self.assert_untouched(host, started, xsamp=False)

        host = _Host()
        started = self.generate(host, "Steps: 20, Sampler: ER SDE (ODE), Schedule type: Karras, CFG scale: 5, Seed: 1, "
                                      "Size: 512x512, ER SDE max stage: 2, ER SDE eta: 0.5, ER SDE noise window: 0.1-0.9")
        self.assertEqual(started["sampler"], ODE)
        self.assertEqual(started["xsamp"], replace(USER_XSAMP, max_stage=2))
        self.assertEqual(started["xsamp_text"], {**XSAMP_TEXT, "max_stage": "2"})

    def test_values_present_but_not_used_leave_every_field_alone(self):
        host = _Host(sampler="DPM++ 2M", scheduler="karras")
        started = self.generate(host, (
            'Steps: 20, Sampler: Euler a, Schedule type: Karras, CFG scale: 5, Seed: 1, Size: 512x512, '
            'Custom scheduler expression: x, Custom scheduler sigmas: "[1.0, 0.0]", Laplace mu: 0.25, Laplace beta: 1.5, '
            'Flow Cosmos rho: 5.5, Flow Cosmos sigma max: 240.0, Flow Cosmos sigma min: 0.006, ER SDE max stage: 2, '
            'ER SDE eta: 0.5, ER SDE noise window: 0.1-0.9, Eta: 0.3, Hires schedule type: Use same scheduler'))
        self.assertEqual((started["sampler"], started["scheduler"]), ("Euler a", "karras"))
        self.assertEqual((started["xsched"], started["xsamp"]), (USER_XSCHED, USER_XSAMP))
        self.assert_untouched(host, started)

    def test_a_scheduler_without_values_and_an_image_without_keys(self):
        host = _Host()
        started = self.generate(host, "Steps: 20, Sampler: Euler, Schedule type: Karras Dynamic, CFG scale: 5, "
                                      "Seed: 1, Size: 512x512")
        self.assertEqual(started["scheduler"], "karras_dynamic")
        self.assert_untouched(host, started)

        host = _Host()   # Laplace·Reverse-time 인데 값 키가 없다 — 확장 붙여 넣기처럼 그 칸은 기본값
        started = self.generate(host, "Steps: 20, Sampler: ER SDE (Reverse-time), Schedule type: Laplace, CFG scale: 5, "
                                      "Seed: 1, Size: 512x512")
        self.assertEqual((started["xsched"].laplace_mu, started["xsched"].laplace_beta),
                         (xs.LAPLACE_MU_DEFAULT, xs.LAPLACE_BETA_DEFAULT))
        self.assertEqual(started["xsched_text"]["custom_expression"], "  M * 0.5 ")
        self.assertEqual(started["xsamp"], xsa.APP_DEFAULTS)

    def test_the_bridge_action_and_a_legacy_raw_payload_take_the_same_path(self):
        line = ("Steps: 20, Sampler: ER SDE (Reverse-time), Schedule type: Laplace, CFG scale: 5, Seed: 1, "
                "Size: 512x512, Laplace mu: 0.25, Laplace beta: 1.5, ER SDE eta: 0.5")
        host = _Host()
        host._handle_vue_action("pnginfo_generate", self.info(host, line))
        [started] = host.started
        self.assertEqual((started["xsched"].laplace_mu, started["xsamp"].eta), (0.25, 0.5))

        host = _Host()   # 구 프런트 — parameters dict 없이 raw 만
        self.assertTrue(start_generation_from_metadata(host, {"source": "webui", "prompt": "a cat",
                                                              "raw": "a cat\n" + line}))
        [started] = host.started
        self.assertEqual((started["xsched"].laplace_beta, started["xsamp"].eta), (1.5, 0.5))

    def test_no_generation_info_or_no_value_widgets_change_nothing(self):
        host = _Host()
        self.assertFalse(start_generation_from_metadata(host, {"source": "webui", "prompt": "",
                                                               "parameters": {"Schedule type": "Laplace",
                                                                              "Laplace mu": "1"}}))
        self.assertEqual(host.started, [])
        self.assertEqual(_Host.texts(host.extra_schedulers_widgets), XSCHED_TEXT)
        bare = _Harness()   # 값 칸이 없는 호스트(옛 호스트·테스트 더블) — 아무것도 하지 않는다
        restore_sampling_values(bare, {"parameters": {"Schedule type": "Laplace", "Sampler": RT, "Laplace mu": "1"}})
        self.assertFalse(hasattr(bare, "extra_schedulers_widgets"))


if __name__ == "__main__":
    unittest.main()
