"""Comfy Detail Daemon — 팩 노드가 원본 ComfyUI 노드와 똑같이 도는가, 컴파일러가 패스마다 맞는 모델을 주는가 (DD-C).

기준(사용자 결정): Jonseed/ComfyUI-Detail-Daemon 의 "Detail Daemon Sampler" 노드를 그대로 따른다 —
×0.1×cfg 늘, 프리셋 없음, [0.05, 3] 클램프 없음(노드 자체의 하한 1e-6 만), 모델 호출 σ 를 샘플러 σ 목록에서
찾아 이웃 사이를 보간. 노드에 없는 Forge 전용 Hires Pass 는 muerrilla/sd-webui-detail-daemon 을 따른다(기본 끔 =
base 패스만, 켜면 hires 패스만, cfg 는 base cfg_scale). 생성 안의 디테일러도 muerrilla·확장과 같다: ADetailer 는
마지막 본 패스의 모델을 이어받고, SAM3 패스는 Hires Pass 가 꺼져 있을 때 자기 cfg·샘플러로 다시 건다
(CompilerPassModelTests). 단독 후처리에는 없다.

아래 ``node_*`` 와 ``NODE_INPUTS`` 는 원본(MIT)에서 그대로 옮긴 황금 기준이다 — 각 origin 줄 참고.
  ComfyUI-Detail-Daemon — Copyright (c) 2024 Jonseed — MIT License
  (전문: comfy_custom_nodes/ai_studio_forge_parity/LICENSES/ComfyUI-Detail-Daemon-MIT.txt)
``node_make_schedule`` 은 Jonseed 가 muerrilla 의 make_schedule(ORIGIN_FORGE:309-338)에서 옮긴 함수다.
  sd-webui-detail-daemon — Copyright (c) 2024 Sahand Ahmadian — MIT License
  (전문: comfy_custom_nodes/ai_studio_forge_parity/LICENSES/sd-webui-detail-daemon-MIT.txt)
팩의 복사본이 원본 파일과 글자까지 같은지, 설치된 ComfyUI 의 실제 KSAMPLER·WrapperExecutor 로 원본 노드와 팩 노드를
12개 샘플러 × 536회 돌려 모델이 받는 σ 와 결과 latent 가 비트까지 같은지는 세션 스크래치
(origin_parity/dd_c/verbatim_check.py, comfy_parity.py — CPU, GPU 숨김)에서 확인했다. 여기서는 torch·ComfyUI 없이
도는 계약(스케줄·입력·그래프)과, torch 가 있으면 CPU 에서 σ 조회·샘플러 래퍼를 본다.
"""
from __future__ import annotations

import json
import sys
import types
import unittest
from unittest import mock

import numpy as np

from comfy_custom_nodes.ai_studio_forge_parity import guidance, guidance_dd
from core import anima_guidance, sam3_args
from core.comfy_workflow_compiler import ComfyWorkflowCompiler, WorkflowCompileError
from tests._optional_deps import load_torch, requires_torch

ORIGIN = "Jonseed/ComfyUI-Detail-Daemon@3394e44afea04ed0188fb37b21f0d9952469766b:detail_daemon_node.py"
ORIGIN_FORGE = "muerrilla/sd-webui-detail-daemon@19479998340831d7804fca8efd3f262b54b6373f:scripts/detail_daemon.py"


# origin: Jonseed/ComfyUI-Detail-Daemon@3394e44:detail_daemon_node.py:25-67 (make_detail_daemon_schedule, 원문 그대로)
# — Jonseed 가 옮겨 온 muerrilla/sd-webui-detail-daemon@19479998:scripts/detail_daemon.py:309-338 make_schedule
#   (Copyright (c) 2024 Sahand Ahmadian, MIT)
def node_make_schedule(steps, start, end, bias, amount, exponent, start_offset, end_offset, fade, smooth):
    start = min(start, end)
    mid = start + bias * (end - start)
    multipliers = np.zeros(steps)

    start_idx, mid_idx, end_idx = [
        int(round(x * (steps - 1))) for x in [start, mid, end]
    ]

    start_values = np.linspace(0, 1, mid_idx - start_idx + 1)
    if smooth:
        start_values = 0.5 * (1 - np.cos(start_values * np.pi))
    start_values = start_values**exponent
    if start_values.any():
        start_values *= amount - start_offset
        start_values += start_offset

    end_values = np.linspace(1, 0, end_idx - mid_idx + 1)
    if smooth:
        end_values = 0.5 * (1 - np.cos(end_values * np.pi))
    end_values = end_values**exponent
    if end_values.any():
        end_values *= amount - end_offset
        end_values += end_offset

    multipliers[start_idx : mid_idx + 1] = start_values
    multipliers[mid_idx : end_idx + 1] = end_values
    multipliers[:start_idx] = start_offset
    multipliers[end_idx + 1 :] = end_offset
    multipliers *= 1 - fade

    return multipliers


# origin: Jonseed/ComfyUI-Detail-Daemon@3394e44:detail_daemon_node.py:226-262 (get_dd_schedule, 원문 그대로; torch 는 호출 때 넘긴다)
def node_get_dd_schedule(torch, sigma, sigmas, dd_schedule):
    sched_len = len(dd_schedule)
    if (
        sched_len < 2
        or len(sigmas) < 2
        or sigma <= 0
        or not (sigmas[-1] <= sigma <= sigmas[0])
    ):
        return 0.0
    deltas = (sigmas[:-1] - sigma).abs()
    idx = int(deltas.argmin())
    if (
        (idx == 0 and sigma >= sigmas[0])
        or (idx == sched_len - 1 and sigma <= sigmas[-2])
        or deltas[idx] == 0
    ):
        return dd_schedule[idx].item()
    idxlow, idxhigh = (idx, idx - 1) if sigma > sigmas[idx] else (idx + 1, idx)
    nlow, nhigh = sigmas[idxlow], sigmas[idxhigh]
    if nhigh - nlow == 0:
        return dd_schedule[idxlow]
    ratio = ((sigma - nlow) / (nhigh - nlow)).clamp(0, 1)
    return torch.lerp(dd_schedule[idxlow], dd_schedule[idxhigh], ratio).item()


# origin: Jonseed/ComfyUI-Detail-Daemon@3394e44:detail_daemon_node.py:288, :291-296 (model_wrapper 의 σ 조정)
def node_adjusted_sigma(torch, sigma, sigmas_cpu, dd_schedule, cfg_scale):
    sigma_max, sigma_min = float(sigmas_cpu[0]), float(sigmas_cpu[-1]) + 1e-05
    sigma_float = float(sigma.max().detach().cpu())
    if not (sigma_min <= sigma_float <= sigma_max):
        return sigma
    dd_adjustment = node_get_dd_schedule(torch, sigma_float, sigmas_cpu, dd_schedule) * 0.1
    return sigma * max(1e-06, 1.0 - dd_adjustment * cfg_scale)


# origin: Jonseed/ComfyUI-Detail-Daemon@3394e44:detail_daemon_node.py:324-367 (DetailDaemonSamplerNode.INPUT_TYPES 의 값 칸)
NODE_INPUTS = {
    "detail_amount": ("FLOAT", {"default": 0.1, "min": -5.0, "max": 5.0, "step": 0.01}),
    "start": ("FLOAT", {"default": 0.2, "min": 0.0, "max": 1.0, "step": 0.01}),
    "end": ("FLOAT", {"default": 0.8, "min": 0.0, "max": 1.0, "step": 0.01}),
    "bias": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
    "exponent": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.05}),
    "start_offset": ("FLOAT", {"default": 0.0, "min": -1.0, "max": 1.0, "step": 0.01}),
    "end_offset": ("FLOAT", {"default": 0.0, "min": -1.0, "max": 1.0, "step": 0.01}),
    "fade": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.05}),
    "smooth": ("BOOLEAN", {"default": True}),
    "cfg_scale_override": ("FLOAT", {"default": 0, "min": 0.0, "max": 100.0, "step": 0.5, "round": 0.01}),
}
NODE_DEFAULTS = {name: spec[1]["default"] for name, spec in NODE_INPUTS.items()}

SHAPES = [
    dict(start=0.2, end=0.8, bias=0.5, amount=0.1, exponent=1.0, start_offset=0.0, end_offset=0.0,
         fade=0.0, smooth=True),
    dict(start=0.1, end=0.9, bias=0.3, amount=-2.5, exponent=2.0, start_offset=0.2, end_offset=-0.4,
         fade=0.25, smooth=False),
    dict(start=0.8, end=0.2, bias=1.0, amount=5.0, exponent=0.0, start_offset=-1.0, end_offset=1.0,
         fade=0.0, smooth=True),      # start>end 는 노드가 min 으로 접는다, exponent 0 은 0**0 = 1
    dict(start=0.0, end=1.0, bias=0.0, amount=1.0, exponent=10.0, start_offset=0.5, end_offset=0.5,
         fade=1.0, smooth=True),
    dict(start=0.5, end=0.5, bias=0.5, amount=0.3, exponent=1.0, start_offset=0.0, end_offset=0.1,
         fade=0.05, smooth=True),
]


def _values(shape):
    """SHAPES 한 줄 → 팩 detail_daemon_schedule 이 받는 노드 입력 dict."""
    names = {"amount": "detail_amount"}
    return {names.get(key, key): value for key, value in shape.items()}


def _flow_sigmas(steps, shift=3.0):
    """Anima flow σ (shift 3) — σ = s·t/(1+(s−1)·t), t = 1 → 0."""
    t = np.linspace(1.0, 0.0, steps + 1)
    return shift * t / (1.0 + (shift - 1.0) * t)


# ── 1) 원본 공식 (numpy, torch 없음) ───────────────────────────────────────────────
class NodeScheduleParityTests(unittest.TestCase):
    def test_schedule_is_the_nodes_bit_for_bit(self):
        for shape in SHAPES:
            for steps in (1, 2, 3, 7, 10, 20, 28, 30, 50):
                with self.subTest(shape=shape, steps=steps):
                    expected = node_make_schedule(steps, **shape)
                    got = guidance_dd.detail_daemon_schedule(steps, _values(shape))
                    self.assertTrue(np.array_equal(got, expected), (got, expected))

    def test_default_amount_peaks_at_the_nodes_sigma_factor(self):
        # 노드 기본값 0.10, cfg 5: 가장 센 스텝의 σ 배율 = 1 − 0.1·0.1·5 = 0.95 (parity_plan §1.0 표)
        schedule = guidance_dd.detail_daemon_schedule(28, dict(NODE_DEFAULTS))
        peak = float(max(schedule))
        self.assertEqual(peak, 0.1)
        self.assertAlmostEqual(max(1e-06, 1.0 - peak * guidance_dd.DD_SIGMA_SCALE * 5.0), 0.95, places=12)
        self.assertEqual(guidance.DD_SIGMA_SCALE, 0.1)          # origin :169, :294

    def test_extreme_amount_hits_only_the_nodes_floor(self):
        # 옛 팩은 [0.05, 3] 으로 잘랐다. 노드는 max(1e-06, …) 뿐 — amount 5, cfg 12 → 1 − 6 → 1e-6
        schedule = guidance_dd.detail_daemon_schedule(20, dict(NODE_DEFAULTS, detail_amount=5.0))
        factor = max(1e-06, 1.0 - float(max(schedule)) * 0.1 * 12.0)
        self.assertEqual(factor, 1e-06)
        negative = guidance_dd.detail_daemon_schedule(20, dict(NODE_DEFAULTS, detail_amount=-5.0))
        self.assertAlmostEqual(1.0 - float(min(negative)) * 0.1 * 12.0, 7.0, places=12)   # 위로도 자르지 않는다


class NodeInputTests(unittest.TestCase):
    def test_input_table_is_the_nodes(self):
        spec = guidance_dd.origin_input_types()["required"]
        self.assertEqual(spec["sampler"], ("SAMPLER",))
        for name, (kind, options) in NODE_INPUTS.items():
            with self.subTest(name=name):
                self.assertEqual(spec[name][0], kind)
                for key, value in options.items():
                    self.assertEqual(spec[name][1][key], value)
        self.assertEqual(set(spec) - {"sampler"}, set(NODE_INPUTS))

    def test_settings_keys_map_to_every_node_value_input(self):
        self.assertEqual({name for _key, name in guidance_dd.SETTING_KEYS},
                         set(NODE_INPUTS) - {"cfg_scale_override"})
        # 컴파일러가 보내는 키 = dd_enabled + 팩이 읽는 키 (preset·multiplier·cfg_couple·dd_hires 는 안 보낸다)
        self.assertEqual(ComfyWorkflowCompiler._DD_NODE_KEYS,
                         ("dd_enabled",) + tuple(key for key, _name in guidance_dd.SETTING_KEYS))

    def test_missing_settings_take_the_nodes_defaults(self):
        self.assertEqual(guidance_dd.detail_daemon_node_values({}), NODE_DEFAULTS)
        self.assertEqual(guidance_dd.detail_daemon_node_values({}, None)["cfg_scale_override"], 0)

    def test_values_pass_unchanged_presets_and_old_markers_are_ignored(self):
        settings = {"dd_amount": 0.3, "dd_start_offset": -0.2, "dd_end_offset": 0.4, "dd_smooth": "false"}
        plain = guidance_dd.detail_daemon_node_values(settings, 7.0)
        self.assertEqual((plain["detail_amount"], plain["start_offset"], plain["end_offset"]), (0.3, -0.2, 0.4))
        self.assertIs(plain["smooth"], False)
        self.assertEqual(plain["cfg_scale_override"], 7.0)
        noisy = dict(settings, dd_preset="Strong", dd_amount_scale="old", dd_schedule="steps",
                     dd_multiplier=2.0, dd_cfg_couple=False, dd_hires=True)
        self.assertEqual(guidance_dd.detail_daemon_node_values(noisy, 7.0), plain)
        self.assertEqual(guidance_dd.detail_daemon_node_values({"dd_preset": "Strong"})["detail_amount"], 0.1)

    def test_values_outside_the_nodes_ranges_are_refused_like_comfy_validation(self):
        # ComfyUI execution.validate_inputs: FLOAT 은 float() 뒤 min/max 밖이면 거부(자르지 않는다)
        for key, bad in (("dd_amount", 5.01), ("dd_amount", -5.01), ("dd_start", -0.01), ("dd_end", 1.01),
                         ("dd_bias", 1.5), ("dd_exponent", 10.05), ("dd_exponent", -0.1),
                         ("dd_start_offset", -1.01), ("dd_end_offset", 1.01), ("dd_fade", 1.01)):
            with self.subTest(key=key, value=bad):
                with self.assertRaisesRegex(ValueError, "Detail Daemon .*(smaller than min|bigger than max)"):
                    guidance_dd.detail_daemon_node_values({key: bad})
        for cfg in (-0.5, 100.5):
            with self.assertRaises(ValueError):
                guidance_dd.detail_daemon_node_values({}, cfg)
        with self.assertRaisesRegex(ValueError, "FLOAT"):
            guidance_dd.detail_daemon_node_values({"dd_amount": "strong"})
        edges = {"dd_amount": -5.0, "dd_start": 0.0, "dd_end": 1.0, "dd_exponent": 10.0,
                 "dd_start_offset": 1.0, "dd_end_offset": -1.0, "dd_fade": 1.0}
        self.assertEqual(guidance_dd.detail_daemon_node_values(edges, 100.0)["detail_amount"], -5.0)


# ── 2) 노드 계약과 래퍼 등록 (가짜 ComfyUI 모듈) ─────────────────────────────────────
class _FakePatcher:
    """ModelPatcher 의 wrapper 저장 방식만 흉내 낸다(add_wrapper_with_key, clone)."""

    def __init__(self):
        self.wrappers = {}

    def clone(self):
        clone = _FakePatcher()
        clone.wrappers = {kind: {key: list(items) for key, items in table.items()}
                          for kind, table in self.wrappers.items()}
        return clone

    def add_wrapper_with_key(self, kind, key, wrapper):
        self.wrappers.setdefault(kind, {}).setdefault(key, []).append(wrapper)


class _FakeKSampler:
    """comfy.samplers.KSAMPLER 자리 — 생성 인자와 sample() 경로만."""

    def __init__(self, sampler_function, extra_options={}, inpaint_options={}):
        self.sampler_function = sampler_function
        self.extra_options = extra_options
        self.inpaint_options = inpaint_options

    def sample(self, model, x, sigmas):
        return self.sampler_function(model, x, sigmas, **self.extra_options)


class _FakeExecutor:
    """comfy.patcher_extension.WrapperExecutor 가 wrapper 에 주는 것(class_obj, wrappers, idx)만."""

    def __init__(self, original, class_obj, wrappers, idx=0):
        self.original, self.class_obj, self.wrappers, self.idx = original, class_obj, list(wrappers), idx

    @classmethod
    def new_class_executor(cls, original, class_obj, wrappers, idx=0):
        return cls(original, class_obj, wrappers, idx)

    def execute(self, *args, **kwargs):
        if self.idx == len(self.wrappers):
            return self.original(*args, **kwargs)
        return self.wrappers[self.idx](self, *args, **kwargs)

    def __call__(self, *args, **kwargs):
        return type(self)(self.original, self.class_obj, self.wrappers, self.idx + 1).execute(*args, **kwargs)


def _fake_comfy_modules():
    comfy = types.ModuleType("comfy")
    samplers = types.ModuleType("comfy.samplers")
    samplers.KSAMPLER = _FakeKSampler
    extension = types.ModuleType("comfy.patcher_extension")
    extension.WrappersMP = types.SimpleNamespace(SAMPLER_SAMPLE="sampler_sample")
    extension.WrapperExecutor = _FakeExecutor
    comfy.samplers, comfy.patcher_extension = samplers, extension
    return {"comfy": comfy, "comfy.samplers": samplers, "comfy.patcher_extension": extension}


class NodeContractTests(unittest.TestCase):
    def test_inputs_keep_the_compiler_contract_and_add_the_nodes_cfg_override(self):
        node = guidance.ForgeNeoAnimaDetailDaemon
        types_ = node.INPUT_TYPES()
        self.assertEqual(tuple(types_["required"]), ("model", "enabled", "settings_json"))
        self.assertEqual(types_["optional"], {"cfg_scale_override": guidance_dd.origin_input_types()["required"]["cfg_scale_override"]})
        self.assertEqual(node.RETURN_TYPES, ("MODEL",))

    def test_disabled_is_the_same_model(self):
        model = _FakePatcher()
        daemon = guidance.ForgeNeoAnimaDetailDaemon()
        self.assertIs(daemon.patch(model, False, "not json")[0], model)
        self.assertIs(daemon.patch(model, True, '{"dd_enabled": false}')[0], model)

    def test_enabled_registers_one_sampler_wrapper_on_a_clone(self):
        model = _FakePatcher()
        with mock.patch.dict(sys.modules, _fake_comfy_modules()):
            patched = guidance.ForgeNeoAnimaDetailDaemon().patch(model, True, '{"dd_amount": 0}', 0.0)[0]
        self.assertIsNot(patched, model)
        self.assertEqual(model.wrappers, {})
        table = patched.wrappers["sampler_sample"]
        self.assertEqual(list(table), [guidance_dd.DD_WRAPPER_KEY])
        self.assertEqual(len(table[guidance_dd.DD_WRAPPER_KEY]), 1)
        # amount 0 도 원본 노드처럼 샘플러를 감싼다(σ×1.0 은 같은 값) — 건너뛰는 지름길은 없다

    def test_out_of_range_settings_fail_the_node(self):
        with mock.patch.dict(sys.modules, _fake_comfy_modules()):
            with self.assertRaisesRegex(ValueError, "detail_amount"):
                guidance.ForgeNeoAnimaDetailDaemon().patch(_FakePatcher(), True, '{"dd_amount": 7}', 0.0)
            with self.assertRaisesRegex(ValueError, "cfg_scale_override"):
                guidance.ForgeNeoAnimaDetailDaemon().patch(_FakePatcher(), True, "{}", 101.0)

    def test_comfy_without_sampler_wrappers_is_refused(self):
        without = {"comfy": types.ModuleType("comfy"), "comfy.patcher_extension": None}
        with mock.patch.dict(sys.modules, without):
            with self.assertRaisesRegex(RuntimeError, "sampler wrappers"):
                guidance.ForgeNeoAnimaDetailDaemon().patch(_FakePatcher(), True, "{}", 0.0)


# ── 3) torch 가 있으면: σ 조회와 샘플러 래퍼 (CPU) ─────────────────────────────────
class _RecordingModel:
    """KSamplerX0Inpaint 자리 — inner_model(guider).cfg 와 호출 σ 기록."""

    def __init__(self, cfg=5.0):
        self.inner_model = types.SimpleNamespace(cfg=cfg)
        self.sigmas_seen = []

    def __call__(self, x, sigma, **extra):
        self.sigmas_seen.append(sigma.clone())
        return x * 0.9


def _euler(model, x, sigmas, **extra):
    """1차 샘플러 모양: 스텝마다 σ_i 로 한 번."""
    s_in = x.new_ones([x.shape[0]])
    for i in range(len(sigmas) - 1):
        denoised = model(x, sigmas[i] * s_in)
        x = denoised + (x - denoised) * (sigmas[i + 1] / sigmas[i])
    return x


def _heun(model, x, sigmas, **extra):
    """2차 샘플러 모양: σ_i 와 두 σ 사이 중간점(마지막 스텝 제외)."""
    s_in = x.new_ones([x.shape[0]])
    for i in range(len(sigmas) - 1):
        model(x, sigmas[i] * s_in)
        if sigmas[i + 1] > 0:
            model(x, (sigmas[i] + sigmas[i + 1]) * 0.5 * s_in)
    return x


@requires_torch
class SigmaLookupTorchTests(unittest.TestCase):
    def test_lookup_matches_the_node_at_steps_midpoints_and_outside(self):
        torch = load_torch()
        for steps in (1, 4, 20, 28):
            sigmas = torch.tensor(_flow_sigmas(steps), dtype=torch.float32)
            schedule = torch.tensor(node_make_schedule(steps, **SHAPES[1]), dtype=torch.float32)
            probes = [float(s) for s in sigmas] + [float(sigmas[0]) + 0.5, -0.1, 0.0]
            probes += [float((sigmas[i] + sigmas[i + 1]) / 2) for i in range(steps)]
            probes += [float(sigmas[i] * 0.75 + sigmas[i + 1] * 0.25) for i in range(steps)]
            for probe in probes:
                with self.subTest(steps=steps, sigma=probe):
                    guidance_dd._bind_torch()
                    got = guidance_dd.get_dd_schedule(probe, sigmas, schedule)
                    self.assertEqual(got, node_get_dd_schedule(torch, probe, sigmas, schedule))

    def test_midpoint_is_interpolated_between_neighbours(self):
        torch = load_torch()
        guidance_dd._bind_torch()
        sigmas = torch.tensor([1.0, 0.5, 0.0])
        schedule = torch.tensor([0.0, 1.0])
        self.assertAlmostEqual(guidance_dd.get_dd_schedule(0.75, sigmas, schedule), 0.5, places=6)
        self.assertEqual(guidance_dd.get_dd_schedule(0.5, sigmas, schedule), 1.0)        # 정확히 스텝 σ
        self.assertEqual(guidance_dd.get_dd_schedule(0.25, sigmas, schedule), 1.0)       # 마지막 칸 뒤는 끝값
        self.assertEqual(guidance_dd.get_dd_schedule(1.5, sigmas, schedule), 0.0)        # 범위 밖


@requires_torch
class SamplerWrapperTorchTests(unittest.TestCase):
    def _run(self, sampler_function, values, cfg, steps=12):
        torch = load_torch()
        sigmas = torch.tensor(_flow_sigmas(steps), dtype=torch.float32)
        original = _FakeKSampler(sampler_function, extra_options={})
        model = _RecordingModel(cfg)
        seen_by_later_wrapper = []

        def later_wrapper(executor, *args, **kwargs):
            seen_by_later_wrapper.append(executor.class_obj)
            return executor(*args, **kwargs)

        with mock.patch.dict(sys.modules, _fake_comfy_modules()):
            wrapper = guidance_dd.detail_daemon_sampler_sample_wrapper(values)
            executor = _FakeExecutor.new_class_executor(original.sample, original, [wrapper, later_wrapper])
            executor.execute(model, torch.zeros(1, 4, 2, 2), sigmas)
        return torch, sigmas, original, model, seen_by_later_wrapper

    def test_chain_continues_with_the_nodes_ksampler(self):
        values = dict(NODE_DEFAULTS, cfg_scale_override=6.5)
        _torch, _sigmas, original, _model, later = self._run(_euler, values, cfg=5.0)
        (dd_sampler,) = later
        self.assertIsInstance(dd_sampler, _FakeKSampler)
        self.assertIs(dd_sampler.sampler_function, guidance_dd.detail_daemon_sampler)
        self.assertIs(dd_sampler.extra_options["dds_wrapped_sampler"], original)
        self.assertEqual(dd_sampler.extra_options["dds_cfg_scale_override"], 6.5)
        self.assertEqual(dd_sampler.inpaint_options, {})          # 원본 노드의 KSAMPLER 도 기본(빈) inpaint 옵션
        self.assertEqual(list(dd_sampler.extra_options["dds_make_schedule"](9)),
                         list(node_make_schedule(9, **SHAPES[0])))

    def test_model_sees_the_nodes_adjusted_sigma(self):
        for sampler_function in (_euler, _heun):
            for shape in SHAPES[:3]:
                for cfg, override in ((5.0, 0.0), (4.0, 7.0)):
                    with self.subTest(sampler=sampler_function.__name__, shape=shape, cfg=cfg, override=override):
                        values = dict(_values(shape), cfg_scale_override=override)
                        torch, sigmas, _original, model, _later = self._run(sampler_function, values, cfg)
                        schedule = torch.tensor(node_make_schedule(len(sigmas) - 1, **shape),
                                                dtype=torch.float32, device="cpu")
                        cfg_used = override if override > 0 else cfg        # origin :275-281
                        plain = _RecordingModel(cfg)
                        sampler_function(plain, torch.zeros(1, 4, 2, 2), sigmas)
                        self.assertEqual(len(model.sigmas_seen), len(plain.sigmas_seen))
                        for got, raw in zip(model.sigmas_seen, plain.sigmas_seen):
                            expected = node_adjusted_sigma(torch, raw, sigmas, schedule, cfg_used)
                            self.assertTrue(torch.equal(got, expected), (got, expected, raw))

    def test_cfg_falls_back_to_one_without_a_guider_cfg(self):
        torch = load_torch()
        values = dict(NODE_DEFAULTS, detail_amount=1.0)
        sigmas = torch.tensor(_flow_sigmas(8), dtype=torch.float32)
        model = _RecordingModel()
        model.inner_model = types.SimpleNamespace()               # cfg 없음 → 1.0 (origin :278-281)
        original = _FakeKSampler(_euler)
        with mock.patch.dict(sys.modules, _fake_comfy_modules()):
            guidance_dd.detail_daemon_ksampler(original, values).sample(model, torch.zeros(1, 1, 1, 1), sigmas)
        schedule = torch.tensor(node_make_schedule(8, **dict(SHAPES[0], amount=1.0)), dtype=torch.float32)
        plain = _RecordingModel()
        _euler(plain, torch.zeros(1, 1, 1, 1), sigmas)
        for got, raw in zip(model.sigmas_seen, plain.sigmas_seen):
            self.assertTrue(torch.equal(got, node_adjusted_sigma(torch, raw, sigmas, schedule, 1.0)))


# ── 4) 컴파일러 그래프: 패스마다 받는 모델 ────────────────────────────────────────────
def _dd_payload(*, hires_pass=False, enable_hr=True, image_scripts=True, sam3=None, **dd):
    settings = anima_guidance.default_settings()
    settings.update({"dd_enabled": True, "dd_amount": 0.3, "dd_start_offset": -0.2, "dd_hires": hires_pass})
    settings.update(dd)
    scripts = anima_guidance.build_alwayson(settings)
    if image_scripts:
        scripts["ADetailer"] = {"args": [True, False, {"ad_tab_enable": True, "ad_model": "face.pt"}]}
        scripts.update(sam3_args.build_alwayson({"sam3_prompt": "face", "sam3_mode": "Inpaint", **(sam3 or {})}))
    payload = {"prompt": "portrait", "negative_prompt": "bad", "cfg_scale": 5.0, "steps": 20,
               "sampler_name": "Euler", "scheduler": "Simple", "alwayson_scripts": scripts}
    if enable_hr:
        payload.update({"enable_hr": True, "hr_scale": 1.5, "hr_cfg": 3.0})
    return payload


def _nodes(graph, class_type):
    return [(node_id, node) for node_id, node in graph.items()
            if isinstance(node, dict) and node.get("class_type") == class_type]


def _one(graph, class_type):
    matches = _nodes(graph, class_type)
    if len(matches) != 1:
        raise AssertionError(f"{class_type}: {len(matches)} nodes")
    return matches[0]


BASE_DD = "Anima detail daemon"
HIRES_DD = "Anima detail daemon (hires pass)"
SAM3_DD = "Anima detail daemon (SAM3 detailer)"


def _dd_by_title(graph):
    """제목 → (id, 노드). 패스마다 DD 노드는 하나씩이다(순차 SAM3 패스는 따로 센다)."""
    found = {}
    for node_id, node in _nodes(graph, "ForgeNeoAnimaDetailDaemon"):
        title = node.get("_meta", {}).get("title")
        if title in found:
            raise AssertionError(f"DD 노드 제목 중복: {title}")
        found[title] = (node_id, node)
    return found


class CompilerPassModelTests(unittest.TestCase):
    """패스마다 받는 모델 — muerrilla(원본)와 확장이 실제로 하는 것.

    ADetailer: muerrilla 의 on_cfg_denoiser 콜백은 process() 에서 걸리고(origin ORIGIN_FORGE:256-258) postprocess()
    에서야 풀린다(:269-272). Forge 는 postprocess_image(ADetailer) 를 p.scripts.postprocess 보다 먼저 부르고
    (modules/processing.py:1068 < :1180), ADetailer 의 i2i 는 고른 스크립트만 돌려(기본 ad_script_names 에 DD 없음)
    DD 를 다시 판정하지 않는다 — 그래서 콜백은 is_hires_pass 가 마지막 본 패스 값인 채로 불리고
    daemon['hires'] 와 같을 때 base cfg 로 건다(:276, :259, :304). 확장도 _DD['on'] 이 마지막 본 패스 값으로 남는다.
    → ADetailer 는 마지막 본 패스의 모델을 받는다.
    SAM3: 확장의 SAM3 인페인트(sam3ext/inpaint_core.py build_i2i → script_filter)는 SAM3 만 뺀 alwayson 스크립트를
    p2(img2img)로 다시 돌려 DD 를 p2 로 다시 판정한다(is_hr_pass False·p2.cfg_scale·p2.sampler_name).
    → Hires Pass 가 꺼져 있을 때만, 그 패스의 cfg·샘플러로.
    """

    def _compile(self, payload, **kwargs):
        return ComfyWorkflowCompiler().compile("txt2img", "model.safetensors", payload, **kwargs)

    def test_base_pass_by_default_adetailer_follows_the_last_pass_sam3_reruns_it(self):
        graph = self._compile(_dd_payload())                    # Hires fix 켬, Hires Pass 끔
        daemons = _dd_by_title(graph)
        self.assertEqual(set(daemons), {BASE_DD, SAM3_DD})
        base_id, base = daemons[BASE_DD]
        without_dd = base["inputs"]["model"]
        self.assertEqual(_one(graph, "ForgeNeoKSamplerCNS")[1]["inputs"]["model"], [base_id, 0])
        self.assertEqual(_one(graph, "ForgeNeoHiresFix")[1]["inputs"]["model"], without_dd)
        # 마지막 본 패스(hires)에 DD 가 없으니 ADetailer 도 없다(ORIGIN_FORGE:276, is_hires_pass True ≠ hires False)
        self.assertEqual(_one(graph, "ForgeNeoADetailer")[1]["inputs"]["model"], without_dd)
        sam_id, sam = daemons[SAM3_DD]
        self.assertEqual(sam["inputs"]["model"], without_dd)
        self.assertEqual(_one(graph, "ForgeNeoSAM3Detailer")[1]["inputs"]["model"], [sam_id, 0])
        self.assertEqual(sam["inputs"]["settings_json"], base["inputs"]["settings_json"])
        self.assertEqual(sam["inputs"]["cfg_scale_override"], 5.0)

    def test_without_hires_fix_adetailer_gets_the_base_pass_model(self):
        graph = self._compile(_dd_payload(enable_hr=False))
        daemons = _dd_by_title(graph)
        self.assertEqual(set(daemons), {BASE_DD, SAM3_DD})
        base_id, base = daemons[BASE_DD]
        self.assertEqual(_one(graph, "ForgeNeoKSamplerCNS")[1]["inputs"]["model"], [base_id, 0])
        self.assertEqual(_one(graph, "ForgeNeoADetailer")[1]["inputs"]["model"], [base_id, 0])
        self.assertEqual(base["inputs"]["cfg_scale_override"], 5.0)         # ADetailer 의 cfg 가 아니라 base cfg
        sam_id, _sam = daemons[SAM3_DD]
        self.assertEqual(_one(graph, "ForgeNeoSAM3Detailer")[1]["inputs"]["model"], [sam_id, 0])

    def test_hires_pass_only_with_dd_hires_and_cfg_is_the_base_cfg(self):
        graph = self._compile(_dd_payload(hires_pass=True))
        daemons = _dd_by_title(graph)
        self.assertEqual(set(daemons), {HIRES_DD})
        dd_id, dd = daemons[HIRES_DD]
        without_dd = dd["inputs"]["model"]
        self.assertEqual(_one(graph, "ForgeNeoHiresFix")[1]["inputs"]["model"], [dd_id, 0])
        hires_cfg = _one(graph, "ForgeNeoHiresFix")[1]["inputs"]["cfg"]
        self.assertEqual(hires_cfg, 3.0)
        # muerrilla: DD 의 cfg 는 hr_cfg 가 아니라 p.cfg_scale (origin ORIGIN_FORGE:259, :304)
        self.assertEqual(dd["inputs"]["cfg_scale_override"], 5.0)
        self.assertEqual(_one(graph, "ForgeNeoKSamplerCNS")[1]["inputs"]["model"], without_dd)
        # ADetailer 는 마지막 본 패스(hires)의 모델을 이어받는다(ORIGIN_FORGE:276, is_hires_pass True)
        self.assertEqual(_one(graph, "ForgeNeoADetailer")[1]["inputs"]["model"], [dd_id, 0])
        # SAM3 p2 는 img2img(is_hr_pass False)라 Hires Pass 데몬이 돌지 않는다
        self.assertEqual(_one(graph, "ForgeNeoSAM3Detailer")[1]["inputs"]["model"], without_dd)

    def test_sam3_pass_uses_its_own_cfg_and_sampler(self):
        # 확장: p2.cfg_scale(SAM3 cfg 를 따로 쓰면 그 값)·p2.sampler_name 으로 다시 판정한다
        graph = self._compile(_dd_payload(enable_hr=False, sam3={"sam3_use_cfg_scale": True, "sam3_cfg_scale": 4.0}))
        daemons = _dd_by_title(graph)
        self.assertEqual(daemons[SAM3_DD][1]["inputs"]["cfg_scale_override"], 4.0)
        self.assertEqual(daemons[BASE_DD][1]["inputs"]["cfg_scale_override"], 5.0)
        # SAM3 샘플러만 DPM adaptive → SAM3 패스만 끈다
        graph = self._compile(_dd_payload(
            enable_hr=False, sam3={"sam3_use_sampler": True, "sam3_sampler": "DPM adaptive"}))
        daemons = _dd_by_title(graph)
        self.assertEqual(set(daemons), {BASE_DD})
        self.assertEqual(_one(graph, "ForgeNeoSAM3Detailer")[1]["inputs"]["model"],
                         daemons[BASE_DD][1]["inputs"]["model"])
        # base 만 DPM adaptive → base·ADetailer 는 끄고 SAM3(자기 샘플러 Euler)는 건다
        graph = self._compile(dict(
            _dd_payload(enable_hr=False, sam3={"sam3_use_sampler": True, "sam3_sampler": "Euler"}),
            sampler_name="DPM adaptive"))
        daemons = _dd_by_title(graph)
        self.assertEqual(set(daemons), {SAM3_DD})
        sam_id, sam = daemons[SAM3_DD]
        self.assertEqual(_one(graph, "ForgeNeoADetailer")[1]["inputs"]["model"], sam["inputs"]["model"])
        self.assertEqual(_one(graph, "ForgeNeoSAM3Detailer")[1]["inputs"]["model"], [sam_id, 0])

    def test_sequential_sam3_passes_each_get_dd(self):
        payload = dict(_dd_payload(enable_hr=False), _comfy_detail_passes=["eyes"])
        graph = self._compile(payload)
        detailers = _nodes(graph, "ForgeNeoSAM3Detailer")
        self.assertEqual(len(detailers), 2)
        sam3_daemons = [node_id for node_id, node in _nodes(graph, "ForgeNeoAnimaDetailDaemon")
                        if node["_meta"]["title"] == SAM3_DD]
        self.assertEqual(len(sam3_daemons), 2)
        self.assertEqual(sorted(tuple(node["inputs"]["model"]) for _node_id, node in detailers),
                         sorted((node_id, 0) for node_id in sam3_daemons))
        # Hires Pass 데몬이면 순차 SAM3 패스에도 없다
        payload = dict(_dd_payload(hires_pass=True, enable_hr=False), _comfy_detail_passes=["eyes"])
        self.assertEqual(_nodes(self._compile(payload), "ForgeNeoAnimaDetailDaemon"), [])

    def test_settings_are_the_node_values_unchanged(self):
        graph = self._compile(_dd_payload(image_scripts=False))
        _dd_id, dd = _one(graph, "ForgeNeoAnimaDetailDaemon")
        settings = json.loads(dd["inputs"]["settings_json"])
        self.assertEqual(tuple(sorted(settings)), tuple(sorted(ComfyWorkflowCompiler._DD_NODE_KEYS)))
        self.assertEqual(settings["dd_amount"], 0.3)           # 옛 컴파일러는 0.03(×0.1) 으로 바꿨다
        self.assertEqual(settings["dd_start_offset"], -0.2)
        self.assertEqual(dd["inputs"]["cfg_scale_override"], 5.0)
        # 팩 노드가 이 settings 로 읽는 값 = 원본 노드 입력
        values = guidance_dd.detail_daemon_node_values(settings, dd["inputs"]["cfg_scale_override"])
        self.assertEqual((values["detail_amount"], values["start_offset"], values["cfg_scale_override"]),
                         (0.3, -0.2, 5.0))

    def test_no_node_when_the_chosen_pass_does_not_run_or_dd_is_a_no_op(self):
        cases = {
            "hires pass without Hires fix": _dd_payload(hires_pass=True, enable_hr=False),
            "disabled": _dd_payload(dd_enabled=False),
            "cfg 0 (muerrilla ×cfg = no-op)": dict(_dd_payload(), cfg_scale=0.0),
        }
        for label, payload in cases.items():
            with self.subTest(label):
                self.assertEqual(_nodes(self._compile(payload), "ForgeNeoAnimaDetailDaemon"), [])

    def test_dpm_adaptive_and_heunpp2_base_sampler_turn_it_off_like_muerrilla(self):
        # origin: ORIGIN_FORGE:197-199 — p.sampler_name(base) 로 생성 전체를 끈다
        for label in ("DPM adaptive", "HeunPP2", "dpm_adaptive"):
            with self.subTest(sampler=label):
                graph = self._compile(dict(_dd_payload(image_scripts=False), sampler_name=label))
                self.assertEqual(_nodes(graph, "ForgeNeoAnimaDetailDaemon"), [])
        # hires 샘플러만 HeunPP2 면 base 판정이라 hires 패스 DD 는 그대로다
        graph = self._compile(dict(_dd_payload(hires_pass=True, image_scripts=False), hr_sampler_name="HeunPP2"))
        dd_id, _dd = _one(graph, "ForgeNeoAnimaDetailDaemon")
        self.assertEqual(_one(graph, "ForgeNeoHiresFix")[1]["inputs"]["model"], [dd_id, 0])

    def test_postprocess_detailers_never_get_dd(self):
        # Forge 의 단독 ADetailer/SAM3 요청(webui_backend._build_postprocess_payload)은 그 스크립트 인자만 보내
        # DD 는 UI 기본값(끔)으로 돈다 — 저장된 생성 문맥에 DD 가 켜져 있어도 걸지 않는다.
        for kind in ("ForgeNeoSAM3Detailer", "ForgeNeoSAM3Refine"):
            with self.subTest(kind=kind):
                graph = ComfyWorkflowCompiler().compile_postprocess(
                    "model.safetensors", _dd_payload(enable_hr=False), uploaded_image="source.png",
                    sam3_detailer_class=kind,
                )
                self.assertEqual(_nodes(graph, "ForgeNeoAnimaDetailDaemon"), [])
                self.assertTrue(_nodes(graph, "ForgeNeoADetailer"))
                self.assertTrue(_nodes(graph, kind))

    def test_custom_workflow_follows_the_same_pass_rules(self):
        from tests.test_comfy_workflow_compiler import _custom_workflow

        for hires_pass in (False, True):
            with self.subTest(hires_pass=hires_pass):
                graph = self._compile(_dd_payload(hires_pass=hires_pass), workflow=_custom_workflow())
                daemons = _dd_by_title(graph)
                main_id, main = daemons[HIRES_DD if hires_pass else BASE_DD]
                without_dd = main["inputs"]["model"]
                base = graph["5"]["inputs"]["model"]
                hires = _one(graph, "ForgeNeoHiresFix")[1]["inputs"]["model"]
                self.assertEqual((base, hires),
                                 (without_dd, [main_id, 0]) if hires_pass else ([main_id, 0], without_dd))
                # ADetailer = 마지막 본 패스(hires)의 모델, SAM3 = Hires Pass 가 꺼져 있을 때만 자기 DD
                self.assertEqual(_one(graph, "ForgeNeoADetailer")[1]["inputs"]["model"],
                                 [main_id, 0] if hires_pass else without_dd)
                sam3_model = _one(graph, "ForgeNeoSAM3Detailer")[1]["inputs"]["model"]
                if hires_pass:
                    self.assertEqual(set(daemons), {HIRES_DD})
                    self.assertEqual(sam3_model, without_dd)
                else:
                    self.assertEqual(set(daemons), {BASE_DD, SAM3_DD})
                    self.assertEqual(sam3_model, [daemons[SAM3_DD][0], 0])

    def test_stale_pack_without_cfg_scale_override_is_refused_before_queueing(self):
        from tests.test_comfy_workflow_compiler import _capabilities

        payload = _dd_payload(image_scripts=False, enable_hr=False)
        current = _capabilities()
        current["ForgeNeoAnimaDetailDaemon"] = {"input": guidance.ForgeNeoAnimaDetailDaemon.INPUT_TYPES()}
        graph = ComfyWorkflowCompiler(current).compile("txt2img", "checkpoint.safetensors", payload)
        self.assertEqual(len(_nodes(graph, "ForgeNeoAnimaDetailDaemon")), 1)
        stale = _capabilities()
        stale["ForgeNeoAnimaDetailDaemon"] = {"input": {"required": {        # 팩 1.3.0 의 INPUT_TYPES
            "model": ["MODEL"], "enabled": ["BOOLEAN", {"default": False}],
            "settings_json": ["STRING", {"default": "{}", "multiline": True}],
        }}}
        with self.assertRaisesRegex(WorkflowCompileError, "노드 계약.*cfg_scale_override"):
            ComfyWorkflowCompiler(stale).compile("txt2img", "checkpoint.safetensors", payload)


if __name__ == "__main__":
    unittest.main()
