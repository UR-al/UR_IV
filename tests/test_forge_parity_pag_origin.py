"""ComfyUI 팩의 Anima Safe PAG = 원본 노드 그대로 (계획 §2.3 C, 패키지 PAG-C).

origin: iljung1106/comfyui-anima-safe-pag@905b0107d1f924fc6acbcac3b6a879b566ff671c:__init__.py (MIT)

팩은 이 파일을 ``vendor/comfyui_anima_safe_pag/__init__.py`` 로 바이트 그대로 싣고,
``ForgeNeoAnimaSafePAG`` 와 스위트(``_patch_perturbation_guidance``)가 그 ``AnimaSafePAG().patch`` 를
부른다. 여기서 지키는 것:

- 벤더 파일이 원본 그대로다(SHA-256) — 그래서 아래의 '원본 모듈'은 곧 원본이다.
- 노드 입력(기본값·범위·step)이 원본 INPUT_TYPES(:197-210)와 같다.
- 가짜 Anima 모델(CPU)로 샘플링 루프를 돌리면 팩 노드의 post-CFG 결과가 원본 모듈과 비트 단위로 같다.
  모델은 한 스텝에 한 번(추가 행 포함 한 배치)만 불리고 ``disable_cfg1_optimization`` 은 켜지지 않는다.
- Anima(flow, shift 3)에서 PAG 가 걸리는 스텝 수가 원본 σ창대로다: 20/28/30 스텝 → 15/20/22,
  denoise .5 → 9/20, .35 → 3/20. (숫자 출처: scratchpad/origin_parity/pag_range_check.py,
  pag_range_denoise.py — 원본 :15-34 의 percent_to_sigma 창을 흉내 낸 CPU 계산)
- 컴파일러와 스위트가 PAG legacy(legacy strength)/헤드 지정을 받아 원본 노드로 넘기고, SEG 의
  legacy/헤드는 둘 다 막는다(스위트는 모델을 만지기 전에).
- 스위트: ADG 를 PAG 보다 먼저 건다 — 원본이 ADG 를 previous_calc 로 잇고(:237, :255-269), ADG 가
  건너뛰는 스텝에는 추가 행까지 cond 예측을 돌려줘 PAG 항이 0 이다(sam-extra 의 ADG 스텝과 같다).
  rescale_mode 는 sam-extra 처럼 읽는다('partial' 이 아니면 full). 단독 노드는 원본처럼 그대로 넘긴다.
- SEG/SLG 는 원본이 없지만 PAG 와 같은 σ창으로 켜진다(sam-extra ``_pert_in_range`` 와 같다).
- AISTUDIO_COMFY_TEST_ROOT 를 주면 설치된 ComfyUI 의 진짜 ``comfy.ldm.cosmos.predict2.Attention`` 으로
  원본 패치가 돌고 되돌려지는지 본다(CPU 전용, 기본은 건너뜀).

torch 는 지연 import 한다(tests/_optional_deps). torch 를 쓰는 클래스는 @requires_torch.
"""
from __future__ import annotations

import ast
import contextlib
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import sys
import types
import unittest
from unittest import mock

from comfy_custom_nodes.ai_studio_forge_parity import guidance, guidance_pag
from core.comfy_workflow_compiler import ComfyWorkflowCompiler, WorkflowCompileError
from tests._optional_deps import bind_torch, requires_torch

torch = None  # @requires_torch 클래스의 setUpClass 가 bind_torch 로 채운다

VENDORED = Path(guidance_pag.__file__).resolve().parent / "vendor" / "comfyui_anima_safe_pag" / "__init__.py"
# origin: iljung1106/comfyui-anima-safe-pag@905b0107:__init__.py — git blob(LF) 의 SHA-256
ORIGIN_SHA256 = "e894ce01f7fcebe957eb3e682f959fa7b7826579af4534e001539aacd246bf37"
VENDORED_MODULE = f"{guidance_pag.__package__}.vendor.comfyui_anima_safe_pag"


def _original_input_types() -> dict:
    """원본 ``AnimaSafePAG.INPUT_TYPES`` 의 반환 dict — import 없이 AST 로 읽는다(torch·comfy 불필요)."""
    tree = ast.parse(VENDORED.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == "AnimaSafePAG":
            for item in node.body:
                if isinstance(item, ast.FunctionDef) and item.name == "INPUT_TYPES":
                    returns = [n for n in ast.walk(item) if isinstance(n, ast.Return)]
                    return ast.literal_eval(returns[0].value)
    raise AssertionError("AnimaSafePAG.INPUT_TYPES not found in the vendored original")


class TestVendoredOriginal(unittest.TestCase):
    def test_vendored_file_is_the_pinned_original(self):
        self.assertEqual(hashlib.sha256(VENDORED.read_bytes()).hexdigest(), ORIGIN_SHA256)

    def test_node_inputs_are_the_originals(self):
        # origin: iljung1106/comfyui-anima-safe-pag@905b0107:__init__.py:197-210
        original = _original_input_types()["required"]
        pack = guidance.ForgeNeoAnimaSafePAG.INPUT_TYPES()["required"]
        self.assertEqual(list(pack), ["model", "enabled", *list(original)[1:]])
        self.assertEqual({k: v for k, v in pack.items() if k != "enabled"}, original)
        self.assertEqual(original["scale"][1]["max"], 100.0)

    def test_pack_does_not_register_the_original_node(self):
        self.assertNotIn("AnimaSafePAG", guidance.NODE_CLASS_MAPPINGS)
        self.assertIn("ForgeNeoAnimaSafePAG", guidance.NODE_CLASS_MAPPINGS)


class TestIndexParsing(unittest.TestCase):
    def test_reversed_block_range_is_swapped(self):
        # origin: iljung1106/comfyui-anima-safe-pag@905b0107:__init__.py:48-51 (end < start → swap)
        self.assertEqual(guidance.parse_indices("20-18", 28), {18, 19, 20})
        self.assertEqual(guidance.parse_indices("3-1,5", 8), {1, 2, 3, 5})
        self.assertEqual(guidance.parse_indices("18-20", 28), {18, 19, 20})


class TestCompilerAndSuiteGuards(unittest.TestCase):
    validate = staticmethod(ComfyWorkflowCompiler._validate_anima_guidance_settings)

    def test_pag_legacy_and_heads_are_no_longer_rejected(self):
        for settings in (
            {"guid_enabled": True, "guid_attn_method": "PAG", "guid_legacy_attn": True},
            {"guid_enabled": True, "guid_attn_method": "PAG", "guid_head_indices": "0,2"},
            {"guid_enabled": True, "guid_attn_method": "PAG", "guid_legacy_attn": True,
             "guid_head_indices": "7-4"},
        ):
            with self.subTest(settings=settings):
                self.validate(settings)

    def test_seg_legacy_and_heads_are_still_rejected(self):
        with self.assertRaisesRegex(WorkflowCompileError, "legacy SEG"):
            self.validate({"guid_enabled": True, "guid_attn_method": "SEG", "guid_legacy_attn": True})
        with self.assertRaisesRegex(WorkflowCompileError, "head-selective SEG"):
            self.validate({"guid_enabled": True, "guid_attn_method": "SEG", "guid_head_indices": "1"})

    def test_suite_accepts_pag_legacy_and_heads(self):
        # 모델을 만지기 전 검사(reject_unsupported_seg)는 PAG 를 통과시킨다 — 실제 적용은 아래 torch 테스트.
        for method in ("pag", None):
            guidance_pag.reject_unsupported_seg(method, legacy_attn=True, head_indices="0,2")

    def test_rescale_mode_is_read_like_sam_extra(self):
        # sam-extra scripts/anima_safe_pag.py:3632-3635, :3736-3738 — strip·lower, 'partial' 이 아니면 full.
        for value, expected in (
            ("full", "full"), ("Full", "full"), (" full ", "full"), ("FULL", "full"),
            ("bogus", "full"), ("", "full"),
            ("partial", "partial"), (" Partial", "partial"), ("PARTIAL", "partial"),
        ):
            with self.subTest(value=value):
                self.assertEqual(guidance_pag.normalize_rescale_mode(value), expected)

    def test_pack_seg_rejects_legacy_and_heads_before_touching_the_model(self):
        common = dict(
            attention_method="seg", attention_scale=3.0, attention_blocks="18",
            attention_strength=0.75, seg_sigma=10.0, slg_enabled=False, slg_scale=0.0,
            start_percent=0.0, end_percent=0.7, rescale=0.2, rescale_mode="full",
        )
        with self.assertRaisesRegex(RuntimeError, "Legacy SEG"):
            guidance_pag._patch_perturbation_guidance(object(), legacy_attn=True, **common)
        with self.assertRaisesRegex(RuntimeError, "Head-selective SEG"):
            guidance_pag._patch_perturbation_guidance(object(), head_indices="0", **common)


# ── 가짜 ComfyUI 런타임 ─────────────────────────────────────────────────────────
# 원본이 쓰는 표면만 흉내 낸다: ModelPatcher(clone·get_model_object·set_model_sampler_*),
# comfy.samplers.calc_cond_batch(여러 cond 를 한 배치로, cond_or_uncond = cond 번호),
# sampling_function(cfg 1 최적화·CFG·post-CFG 체인), Anima flow σ(shift 3).

def _time_snr_shift(shift, t):
    # 공식: σ = s·t / (1 + (s−1)·t)  (계획 §8.2 '공통', Comfy model_sampling.time_snr_shift)
    return shift * t / (1 + (shift - 1) * t)


class _FlowSampling:
    """Anima model_sampling: flow, shift 3, timestep multiplier 1."""

    def __init__(self, shift=3.0, timesteps=1000):
        self.shift = shift
        self.sigmas = _time_snr_shift(shift, torch.arange(1, timesteps + 1, 1) / timesteps)

    def percent_to_sigma(self, percent):
        if percent <= 0.0:
            return 1.0
        if percent >= 1.0:
            return 0.0
        return _time_snr_shift(self.shift, 1.0 - percent)


def _simple_sigmas(sampling, steps, denoise=1.0):
    """Comfy simple 스케줄러(+ KSampler denoise 처리: int(steps/denoise) 스텝의 끝 steps+1 개)."""
    total = steps if denoise >= 1.0 else int(steps / denoise)
    table = sampling.sigmas
    ss = len(table) / total
    sigmas = [float(table[-(1 + int(x * ss))]) for x in range(total)] + [0.0]
    return torch.FloatTensor(sigmas[-(steps + 1):])


def _calc_cond_batch(model, conds, x_in, timestep, model_options):
    ran = [index for index, cond in enumerate(conds) if cond is not None]
    out = [torch.zeros_like(x_in) for _ in conds]
    if not ran:
        return out
    transformer_options = dict(model_options.get("transformer_options", {}) or {})
    transformer_options["cond_or_uncond"] = ran[:]
    transformer_options["sigmas"] = timestep
    c = {
        "c_crossattn": torch.cat([conds[index]["cross"] for index in ran]),
        "transformer_options": transformer_options,
    }
    input_x = torch.cat([x_in] * len(ran))
    timestep_ = torch.cat([timestep] * len(ran))
    wrapper = model_options.get("model_function_wrapper")
    if wrapper is not None:
        output = wrapper(model.apply_model, {
            "input": input_x, "timestep": timestep_, "c": c, "cond_or_uncond": ran,
        })
    else:
        output = model.apply_model(input_x, timestep_, **c)
    for chunk, index in zip(output.chunk(len(ran)), ran):
        out[index] = chunk
    return out


@contextlib.contextmanager
def _comfy_runtime():
    """가짜 comfy.samplers 를 깔고, 원본 파일을 별도 모듈(기준)로 올려 준다.

    팩 쪽 vendored 모듈도 매번 새로 import 되게 sys.modules 에서 뺀다 — 그 모듈은 import 시점의
    ``comfy`` 를 전역으로 잡는다.
    """
    samplers = types.ModuleType("comfy.samplers")
    samplers.calc_cond_batch = _calc_cond_batch
    comfy = types.ModuleType("comfy")
    comfy.samplers = samplers
    names = ("comfy", "comfy.samplers", VENDORED_MODULE)
    saved = {name: sys.modules.get(name) for name in names}
    sys.modules.pop(VENDORED_MODULE, None)
    sys.modules.update({"comfy": comfy, "comfy.samplers": samplers})
    try:
        spec = importlib.util.spec_from_file_location("_anima_safe_pag_origin_reference", VENDORED)
        reference = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(reference)
        with contextlib.redirect_stdout(io.StringIO()):  # 원본의 첫 스텝 로그
            yield reference
    finally:
        for name, module in saved.items():
            if module is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = module


def _build_model(blocks=3, dim=8, heads=2, seed=7, attention=None):
    """Cosmos/Predict2 모양의 작은 DiT: blocks[i].self_attn 에 n_heads·compute_attention·output_proj.

    ``attention(dim, heads)`` 를 주면 그 모듈(예: 진짜 Cosmos Attention)을 쓴다. 블록은
    ``block(x, transformer_options=...)`` 로 불려 SLG 의 블록 object patch 가 걸린다.
    """
    torch.manual_seed(seed)
    functional = torch.nn.functional

    class Attention(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.n_heads, self.head_dim = heads, dim // heads
            self.q_proj = torch.nn.Linear(dim, dim, bias=False)
            self.k_proj = torch.nn.Linear(dim, dim, bias=False)
            self.v_proj = torch.nn.Linear(dim, dim, bias=False)
            self.output_proj = torch.nn.Linear(dim, dim, bias=False)
            self.output_dropout = torch.nn.Identity()

        @staticmethod
        def attn_op(q, k, v, transformer_options=None):  # [B,S,H,D] -> [B,S,H*D]
            out = functional.scaled_dot_product_attention(
                q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
            ).transpose(1, 2)
            return out.reshape(*out.shape[:-2], -1)

        def compute_attention(self, q, k, v, transformer_options={}):
            result = self.attn_op(q, k, v, transformer_options=transformer_options)
            return self.output_dropout(self.output_proj(result))

        def forward(self, x, transformer_options):
            b, s, _ = x.shape
            q, k, v = (
                proj(x).view(b, s, self.n_heads, self.head_dim)
                for proj in (self.q_proj, self.k_proj, self.v_proj)
            )
            return self.compute_attention(q, k, v, transformer_options=transformer_options)

    class Block(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.self_attn = Attention() if attention is None else attention(dim, heads)

        def forward(self, x, transformer_options=None):
            return x + self.self_attn(x, transformer_options=transformer_options or {})

    class DiT(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.blocks = torch.nn.ModuleList(Block() for _ in range(blocks))

        def forward(self, x, timestep, context, transformer_options):
            hidden = x + context
            for block in self.blocks:
                hidden = block(hidden, transformer_options=transformer_options)
            return hidden * (1.0 + timestep.view(-1, 1, 1))

    class BaseModel:
        def __init__(self):
            self.diffusion_model = DiT().eval()
            self.model_sampling = _FlowSampling()
            self.batch_sizes = []

        def apply_model(self, x, t, c_crossattn=None, transformer_options=None, **_kwargs):
            self.batch_sizes.append(int(x.shape[0]))
            with torch.no_grad():
                return self.diffusion_model(x, t, c_crossattn, transformer_options or {})

    class ModelPatcher:
        def __init__(self, model):
            self.model = model
            self.model_options = {"transformer_options": {}}
            self.object_patches = {}

        def clone(self):
            clone = ModelPatcher(self.model)
            clone.model_options = dict(self.model_options)
            clone.object_patches = dict(self.object_patches)
            return clone

        def add_object_patch(self, path, value):
            self.object_patches[path] = value

        @contextlib.contextmanager
        def patched(self):
            """Comfy 처럼 object patch 를 샘플링 동안만 건다."""
            with contextlib.ExitStack() as stack:
                for path, value in self.object_patches.items():
                    parent, _, attr = path.rpartition(".")
                    stack.enter_context(mock.patch.object(self.get_model_object(parent), attr, value))
                yield

        def get_model_object(self, name):
            value = self.model
            for part in name.split("."):
                value = value[int(part)] if part.isdigit() else getattr(value, part)
            return value

        def set_model_sampler_calc_cond_batch_function(self, function):
            self.model_options["sampler_calc_cond_batch_function"] = function

        def set_model_sampler_post_cfg_function(self, function, disable_cfg1_optimization=False):
            self.model_options["sampler_post_cfg_function"] = (
                self.model_options.get("sampler_post_cfg_function", []) + [function]
            )
            if disable_cfg1_optimization:
                self.model_options["disable_cfg1_optimization"] = True

    return ModelPatcher(BaseModel())


def _sample(patcher, sigmas, cfg, seed=11):
    """sampling_function 한 스텝씩: (스텝별 post-CFG 결과, 스텝별 post-CFG 가 값을 바꿨나)."""
    generator = torch.Generator().manual_seed(seed)
    shape = (1, 6, 8)
    noise = torch.randn(shape, generator=generator)
    cond = {"cross": torch.randn(shape, generator=generator)}
    uncond = {"cross": torch.randn(shape, generator=generator)}
    options = patcher.model_options
    results, changed = [], []
    with patcher.patched():
        _sample_steps(patcher, options, sigmas, cfg, noise, cond, uncond, results, changed)
    return results, changed


def _sample_steps(patcher, options, sigmas, cfg, noise, cond, uncond, results, changed):
    for value in sigmas[:-1]:
        sigma = value.reshape(1)
        x = noise * sigma
        uncond_ = (
            None if cfg == 1.0 and not options.get("disable_cfg1_optimization", False) else uncond
        )
        conds = [cond, uncond_]
        calc = options.get("sampler_calc_cond_batch_function")
        if calc is not None:
            out = calc({
                "conds": conds, "input": x, "sigma": sigma,
                "model": patcher.model, "model_options": options,
            })
        else:
            out = _calc_cond_batch(patcher.model, conds, x, sigma, options)
        cond_pred, uncond_pred = out[0], out[1]
        before = uncond_pred + (cond_pred - uncond_pred) * cfg
        result = before
        for post in options.get("sampler_post_cfg_function", []):
            result = post({
                "denoised": result, "cond": cond, "uncond": uncond_, "model": patcher.model,
                "uncond_denoised": uncond_pred, "cond_denoised": cond_pred, "sigma": sigma,
                "model_options": options, "input": x,
            })
        results.append(result)
        changed.append(not torch.equal(result, before))


def _as_suite_kwargs(values):
    return dict(
        attention_method="pag",
        attention_scale=values["scale"],
        attention_blocks=values["block_indices"],
        attention_strength=values["perturbation_strength"],
        head_indices=values["head_indices"],
        seg_sigma=0.0, slg_enabled=False, slg_scale=0.0,
        start_percent=values["start_percent"], end_percent=values["end_percent"],
        rescale=values["rescale"], rescale_mode=values["rescale_mode"],
    )


# (노드 값, cfg). 역순 블록·헤드·퍼센트, partial rescale, cfg 1 최적화, strength 0 을 섞는다.
CASES = (
    (dict(scale=4.0, block_indices="1", perturbation_strength=0.75, head_indices="",
          start_percent=0.0, end_percent=0.7, rescale=0.2, rescale_mode="full"), 5.0),
    (dict(scale=2.5, block_indices="2-0", perturbation_strength=0.5, head_indices="1",
          start_percent=0.9, end_percent=0.1, rescale=0.6, rescale_mode="partial"), 1.0),
    (dict(scale=6.0, block_indices="0,2", perturbation_strength=0.0, head_indices="1-0",
          start_percent=0.0, end_percent=1.0, rescale=0.0, rescale_mode="full"), 3.0),
)


@requires_torch
class TestOriginalPagParity(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        bind_torch(globals())

    def _assert_same(self, actual, expected):
        self.assertEqual(len(actual), len(expected))
        for step, (left, right) in enumerate(zip(actual, expected)):
            self.assertTrue(torch.equal(left, right), f"step {step} differs from the original")

    def test_pack_post_cfg_matches_the_original_bit_for_bit(self):
        for values, cfg in CASES:
            with self.subTest(values=values, cfg=cfg), _comfy_runtime() as reference:
                base = _build_model()
                sigmas = _simple_sigmas(base.model.model_sampling, 8)
                (original,) = reference.AnimaSafePAG().patch(base, **values)
                expected, expected_changed = _sample(original, sigmas, cfg)
                self.assertTrue(any(expected_changed) or values["perturbation_strength"] == 0.0)

                base.model.batch_sizes.clear()
                (standalone,) = guidance.ForgeNeoAnimaSafePAG().patch(base, True, **values)
                actual, actual_changed = _sample(standalone, sigmas, cfg)
                self._assert_same(actual, expected)
                self.assertEqual(actual_changed, expected_changed)
                # 추가 행은 같은 배치 안에서 돈다: 스텝마다 모델 호출 1번. cfg 1 이면 uncond 를 건너뛴다.
                self.assertEqual(base.model.batch_sizes, [2 if cfg == 1.0 else 3] * (len(sigmas) - 1))
                self.assertNotIn("disable_cfg1_optimization", standalone.model_options)

                suite = guidance_pag._patch_perturbation_guidance(base, **_as_suite_kwargs(values))
                self._assert_same(_sample(suite, sigmas, cfg)[0], expected)

    def test_guidance_suite_node_matches_the_original(self):
        values = CASES[0][0]
        settings = {
            "guid_enabled": True, "guid_attn_method": "PAG",
            "guid_scale": values["scale"], "guid_block_indices": values["block_indices"],
            "guid_official_strength": values["perturbation_strength"],
            "guid_start_percent": values["start_percent"], "guid_end_percent": values["end_percent"],
            "guid_rescale": values["rescale"], "guid_rescale_mode": values["rescale_mode"],
        }
        with _comfy_runtime() as reference:
            base = _build_model()
            sigmas = _simple_sigmas(base.model.model_sampling, 8)
            (original,) = reference.AnimaSafePAG().patch(base, **values)
            (suite,) = guidance.ForgeNeoAnimaGuidanceSuite().patch(
                base, None, None, None, True, json.dumps(settings)
            )
            self._assert_same(_sample(suite, sigmas, 5.0)[0], _sample(original, sigmas, 5.0)[0])

    def test_legacy_pag_uses_legacy_strength_like_sam_extra(self):
        # sam-extra scripts/anima_safe_pag.py:3624 — strength = legacy_strength if legacy_attn else official
        values = dict(CASES[0][0])
        with _comfy_runtime() as reference:
            base = _build_model()
            sigmas = _simple_sigmas(base.model.model_sampling, 8)
            for legacy_attn, used in ((True, 0.3), (False, values["perturbation_strength"])):
                with self.subTest(legacy_attn=legacy_attn):
                    (original,) = reference.AnimaSafePAG().patch(
                        base, **{**values, "perturbation_strength": used}
                    )
                    pack = guidance_pag._patch_perturbation_guidance(
                        base, legacy_attn=legacy_attn, legacy_strength=0.3,
                        **_as_suite_kwargs(values),
                    )
                    self._assert_same(_sample(pack, sigmas, 5.0)[0], _sample(original, sigmas, 5.0)[0])

    def test_invalid_indices_fail_where_the_original_fails(self):
        # origin :55-58 — 유효한 번호가 없으면 RuntimeError. 블록은 patch 때, 헤드는 샘플링 중에.
        values = dict(CASES[0][0])
        with _comfy_runtime() as reference:
            base = _build_model()
            for node in (
                lambda **kw: reference.AnimaSafePAG().patch(base, **kw),
                lambda **kw: guidance.ForgeNeoAnimaSafePAG().patch(base, True, **kw),
            ):
                with self.assertRaisesRegex(RuntimeError, "No valid block indices"):
                    node(**{**values, "block_indices": "99"})
                (patched,) = node(**{**values, "head_indices": "5"})
                with self.assertRaisesRegex(RuntimeError, "No valid block indices"):
                    _sample(patched, _simple_sigmas(base.model.model_sampling, 4), 5.0)

    def test_active_steps_follow_the_original_sigma_window_under_anima_shift_3(self):
        # origin :15-34 (percent_to_sigma 창, 양끝 포함). 기대값: pag_range_check.py / pag_range_denoise.py
        values = dict(CASES[0][0])  # start 0.0, end 0.7 → σ 1.0 … 0.5625
        with _comfy_runtime():
            base = _build_model()
            (patched,) = guidance.ForgeNeoAnimaSafePAG().patch(base, True, **values)
            sampling = base.model.model_sampling
            for steps, denoise, active in (
                (20, 1.0, 15), (28, 1.0, 20), (30, 1.0, 22), (20, 0.5, 9), (20, 0.35, 3),
            ):
                with self.subTest(steps=steps, denoise=denoise):
                    sigmas = _simple_sigmas(sampling, steps, denoise)
                    _results, changed = _sample(patched, sigmas, 5.0)
                    self.assertEqual(len(changed), steps)
                    self.assertEqual(sum(changed), active)
                    # 창은 앞쪽(큰 σ)부터 이어진다.
                    self.assertEqual(changed, [True] * active + [False] * (steps - active))



def _suite_settings(values, **extra):
    """노드 값 → 스위트 settings_json (앱이 보내는 키)."""
    return {
        "guid_enabled": True, "guid_attn_method": "PAG",
        "guid_scale": values["scale"], "guid_block_indices": values["block_indices"],
        "guid_official_strength": values["perturbation_strength"],
        "guid_head_indices": values["head_indices"],
        "guid_start_percent": values["start_percent"], "guid_end_percent": values["end_percent"],
        "guid_rescale": values["rescale"], "guid_rescale_mode": values["rescale_mode"],
        **extra,
    }


def _suite(base, settings):
    (patched,) = guidance.ForgeNeoAnimaGuidanceSuite().patch(
        base, None, None, None, True, json.dumps(settings)
    )
    return patched


@requires_torch
class TestSuiteComposition(unittest.TestCase):
    """스위트가 원본 PAG 노드를 다른 기능과 엮는 방식."""

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        bind_torch(globals())

    def _assert_same(self, actual, expected):
        self.assertEqual(len(actual), len(expected))
        for step, (left, right) in enumerate(zip(actual, expected)):
            self.assertTrue(torch.equal(left, right), f"step {step} differs from the original")

    def test_suite_passes_heads_and_legacy_strength_to_the_original(self):
        # 헤드 지정(역순 포함)은 원본 :61-64, :119-129 그대로. legacy 는 sam-extra :3624 처럼 legacy strength.
        for values, cfg in CASES:
            with self.subTest(values=values), _comfy_runtime() as reference:
                base = _build_model()
                sigmas = _simple_sigmas(base.model.model_sampling, 8)
                (original,) = reference.AnimaSafePAG().patch(base, **values)
                suite = _suite(base, _suite_settings(values))
                self._assert_same(_sample(suite, sigmas, cfg)[0], _sample(original, sigmas, cfg)[0])
        values = dict(CASES[0][0], head_indices="1")
        with _comfy_runtime() as reference:
            base = _build_model()
            sigmas = _simple_sigmas(base.model.model_sampling, 8)
            (original,) = reference.AnimaSafePAG().patch(base, **{**values, "perturbation_strength": 0.3})
            suite = _suite(base, _suite_settings(
                values, guid_legacy_attn=True, guid_legacy_strength=0.3,
            ))
            self._assert_same(_sample(suite, sigmas, 5.0)[0], _sample(original, sigmas, 5.0)[0])

    def test_adg_runs_before_pag_and_zeroes_pag_on_skipped_steps(self):
        # 8 simple 스텝(shift 3)의 퍼센트는 0, .125, …, .875. PAG 창(end .7)은 앞 6스텝.
        # ADG start .3: .375 부터 건너뛴다. interval 2 면 그중 홀수 스텝만(ADG 스텝 번호 = 스텝 순번).
        # 원본은 ADG 를 previous_calc 로 잇는다(:237, :255-269) — 건너뛴 스텝에는 cond 만 돌고(배치 1)
        # 추가 행도 cond 예측이라 PAG 항이 0 이다. sam-extra 도 ADG 스텝에서는 섭동을 더하지 않는다
        # (scripts/anima_safe_pag.py _post_cfg 의 adg_skipped).
        values = dict(CASES[0][0])
        F, T = False, True
        for interval, skipped in ((0, [F, F, F, T, T, T, T, T]), (2, [F, F, F, T, F, T, F, T])):
            with self.subTest(interval=interval), _comfy_runtime() as reference:
                base = _build_model()
                sigmas = _simple_sigmas(base.model.model_sampling, 8)
                (original,) = reference.AnimaSafePAG().patch(base, **values)
                expected, expected_changed = _sample(original, sigmas, 5.0)
                self.assertEqual(expected_changed, [T] * 6 + [F] * 2)

                base.model.batch_sizes.clear()
                suite = _suite(base, _suite_settings(
                    values, guid_adg_enabled=True, guid_adg_start=0.3, guid_adg_interval=interval,
                ))
                actual, changed = _sample(suite, sigmas, 5.0)
                self.assertEqual(base.model.batch_sizes, [1 if skip else 3 for skip in skipped])
                for step, skip in enumerate(skipped):
                    if skip:
                        self.assertFalse(changed[step], f"PAG changed ADG-skipped step {step}")
                    else:
                        self.assertTrue(torch.equal(actual[step], expected[step]), f"step {step}")

    def test_suite_reads_rescale_mode_like_sam_extra_and_the_node_passes_it_raw(self):
        values = dict(CASES[0][0], rescale=0.6)
        with _comfy_runtime() as reference:
            base = _build_model()
            sigmas = _simple_sigmas(base.model.model_sampling, 8)
            by_mode = {
                mode: _sample(
                    reference.AnimaSafePAG().patch(base, **{**values, "rescale_mode": mode})[0],
                    sigmas, 5.0,
                )[0]
                for mode in ("full", "partial")
            }
            self.assertFalse(all(torch.equal(a, b) for a, b in zip(by_mode["full"], by_mode["partial"])))
            for value, mode in (
                ("Full", "full"), (" full ", "full"), ("bogus", "full"),
                (" Partial", "partial"), ("PARTIAL", "partial"),
            ):
                with self.subTest(value=value):
                    suite = _suite(base, _suite_settings({**values, "rescale_mode": value}))
                    self._assert_same(_sample(suite, sigmas, 5.0)[0], by_mode[mode])
            # 단독 노드는 원본처럼 값을 그대로 넘긴다: 원본은 정확히 "full" 이 아니면 partial(:186).
            (standalone,) = guidance.ForgeNeoAnimaSafePAG().patch(
                base, True, **{**values, "rescale_mode": "Full"}
            )
            self._assert_same(_sample(standalone, sigmas, 5.0)[0], by_mode["partial"])


def _slg_settings(with_pag, **extra):
    settings = {
        "guid_slg_on": True, "guid_slg_blocks": "2", "guid_slg_scale": 3.0,
        "guid_start_percent": 0.0, "guid_end_percent": 0.7, "guid_rescale": 0.0,
    }
    if with_pag:
        settings.update(
            guid_enabled=True, guid_attn_method="PAG", guid_block_indices="1", guid_scale=4.0,
        )
    settings.update(extra)
    return settings


@requires_torch
class TestSegSlgWindow(unittest.TestCase):
    """SEG/SLG 는 원본이 없다(계획 §0.3 9번). 창만은 PAG 와 같다 — sam-extra ``_pert_in_range`` 가
    PAG·SEG·SLG 를 한 σ창으로 가른다(origin 창: :15-34). 예전 팩은 1−σ 로 재서 18/25/27 이었다."""

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        bind_torch(globals())

    def test_slg_alone_and_with_pag_is_active_on_the_pag_steps(self):
        with _comfy_runtime():
            base = _build_model()
            sampling = base.model.model_sampling
            for with_pag in (False, True):
                suite = _suite(base, _slg_settings(with_pag))
                for steps, denoise, active in (
                    (20, 1.0, 15), (28, 1.0, 20), (30, 1.0, 22), (20, 0.5, 9),
                ):
                    with self.subTest(with_pag=with_pag, steps=steps, denoise=denoise):
                        changed = _sample(suite, _simple_sigmas(sampling, steps, denoise), 5.0)[1]
                        self.assertEqual(changed, [True] * active + [False] * (steps - active))

    def test_reversed_percents_are_swapped_like_pag(self):
        # origin :26-29 — start > end 이면 바꾼다. (예전 팩 SLG 는 창이 비어 한 번도 켜지지 않았다.)
        with _comfy_runtime():
            base = _build_model()
            sigmas = _simple_sigmas(base.model.model_sampling, 20)
            forward = _sample(_suite(base, _slg_settings(False)), sigmas, 5.0)[0]
            reverse = _sample(_suite(base, _slg_settings(
                False, guid_start_percent=0.7, guid_end_percent=0.0,
            )), sigmas, 5.0)[0]
            self.assertTrue(all(torch.equal(a, b) for a, b in zip(forward, reverse)))

    def test_pag_and_slg_terms_add_up_when_rescale_is_0(self):
        # rescale 0 이면 sam-extra 의 한 번 합산(_apply_perturbation)과 같다. rescale > 0 에서 PAG 와 SLG 가
        # 따로 rescale 되는 것은 호스트 차이다(guidance_pag 모듈 docstring, 계획 §2.4).
        with _comfy_runtime():
            base = _build_model()
            sigmas = _simple_sigmas(base.model.model_sampling, 12)
            plain = _sample(base, sigmas, 5.0)[0]
            pag_settings = {k: v for k, v in _slg_settings(True).items() if not k.startswith("guid_slg")}
            pag_only = _sample(_suite(base, pag_settings), sigmas, 5.0)[0]
            slg_only = _sample(_suite(base, _slg_settings(False)), sigmas, 5.0)[0]
            both = _sample(_suite(base, _slg_settings(True)), sigmas, 5.0)[0]
            self.assertTrue(any(not torch.equal(a, b) for a, b in zip(slg_only, plain)))
            for step, (b, p, s, n) in enumerate(zip(both, pag_only, slg_only, plain)):
                torch.testing.assert_close(b - p, s - n, rtol=0.0, atol=1e-5, msg=f"step {step}")


def _real_cosmos_attention():
    """설치된 ComfyUI 의 ``comfy.ldm.cosmos.predict2.Attention`` (CPU).

    루트는 sys.path 끝에 붙인다 — 앞에 두면 ComfyUI 의 ``tests``/``utils`` 가 앱 것을 가린다.
    학습 중일 수 있어 GPU 는 건드리지 않는다: model_management 를 처음 올리기 전에 --cpu 로 둔다.
    comfy_kitchen 등 ComfyUI 의존성이 있는 파이썬(예: 포터블 python_embeded)에서 돌린다.
    """
    root = os.environ["AISTUDIO_COMFY_TEST_ROOT"]
    if root not in sys.path:
        sys.path.append(root)
    if "comfy.model_management" not in sys.modules:
        from comfy.cli_args import args

        args.cpu = True
    from comfy.ldm.cosmos.predict2 import Attention

    return Attention


@requires_torch
@unittest.skipUnless(
    os.environ.get("AISTUDIO_COMFY_TEST_ROOT"),
    "Set AISTUDIO_COMFY_TEST_ROOT to run the original PAG on real ComfyUI Cosmos attention",
)
class TestOriginalPagOnRealCosmosAttention(unittest.TestCase):
    """원본이 기대는 Attention 표면(compute_attention(q,k,v, transformer_options=)·n_heads·
    output_proj·output_dropout — vendored :95-106, :160-173)을 진짜 ComfyUI 코드로 확인한다."""

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        bind_torch(globals())
        cls.Attention = staticmethod(_real_cosmos_attention())

    def _attention(self, dim, heads):
        return self.Attention(
            dim, n_heads=heads, head_dim=dim // heads, operations=torch.nn, dtype=torch.float32,
        ).eval()

    def test_weak_row_is_the_value_lerp_and_the_patch_is_restored(self):
        with _comfy_runtime() as reference:
            base = _build_model(blocks=1, attention=self._attention)
            block = base.model.diffusion_model.blocks[0]
            attn = block.self_attn
            class_compute = type(attn).compute_attention
            x = torch.randn(2, 5, 8, generator=torch.Generator().manual_seed(3))
            with torch.no_grad():
                normal = attn(x, transformer_options={})
                patched = reference._patch_anima_attention([block], [0], 1, 1.0, "")
                self.assertIsNot(attn.compute_attention.__func__, class_compute)
                try:
                    mixed = attn(x, transformer_options={"cond_or_uncond": [0, 1]})
                finally:
                    reference._restore_attention(patched)
                _q, _k, v = attn.compute_qkv(x, transformer_options={})
                value_row = attn.output_dropout(attn.output_proj(v.reshape(*v.shape[:-2], -1)))
            self.assertIs(attn.compute_attention.__func__, class_compute)
            torch.testing.assert_close(mixed[0], normal[0])
            torch.testing.assert_close(mixed[1], value_row[1])  # strength 1: SDPA 결과가 v 로 완전히 간다
            self.assertGreater((normal[1] - value_row[1]).abs().max().item(), 1e-3)

    def test_pack_node_matches_the_original_through_the_sampling_loop(self):
        values = dict(CASES[0][0], block_indices="0")
        with _comfy_runtime() as reference:
            base = _build_model(blocks=1, attention=self._attention)
            attn = base.model.diffusion_model.blocks[0].self_attn
            class_compute = type(attn).compute_attention
            sigmas = _simple_sigmas(base.model.model_sampling, 8)
            (original,) = reference.AnimaSafePAG().patch(base, **values)
            expected, expected_changed = _sample(original, sigmas, 5.0)
            base.model.batch_sizes.clear()
            (pack,) = guidance.ForgeNeoAnimaSafePAG().patch(base, True, **values)
            actual, changed = _sample(pack, sigmas, 5.0)
            self.assertEqual(changed, [True] * 6 + [False] * 2)  # 패치가 돌았다 = PAG 항이 0 이 아니다
            self.assertEqual(changed, expected_changed)
            for step, (left, right) in enumerate(zip(actual, expected)):
                self.assertTrue(torch.equal(left, right), f"step {step}")
            self.assertEqual(base.model.batch_sizes, [3] * 8)
            self.assertIs(attn.compute_attention.__func__, class_compute)


if __name__ == "__main__":
    unittest.main()
