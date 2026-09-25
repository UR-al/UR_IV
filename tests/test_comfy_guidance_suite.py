"""ComfyUI 팩 가이던스 스위트의 합성 순서 = sam-extra (계획 §8.3 SUITE).

sam-extra 는 post-CFG 하나(``scripts/anima_safe_pag.py`` ``_post_cfg``)에서 고정 순서로 돈다:
CFG 단계(SMC → APG → CWM) → PAG/SEG/SLG → DCW/RDC. 팩 스위트(``guidance.ForgeNeoAnimaGuidanceSuite``)는
같은 순서를 ComfyUI 훅으로 건다 — CFG 단계는 ``sampler_cfg_function`` 하나, 그 뒤 post-CFG 목록이
[원본 PAG, SEG/SLG, DCW]. 여기서 지키는 것:

- 등록 순서와 숫자: 한 스텝의 결과가 DCW(PAG(CFG 단계)) 이다.
- ADG 는 PAG 보다 먼저 걸린다. 원본 PAG 가 ADG 를 previous_calc 로 이어(origin:
  iljung1106/comfyui-anima-safe-pag@905b0107:__init__.py:237, :255-269) ADG 가 건너뛴 스텝에서는 추가 행도
  cond 예측이라 PAG 항이 0 이다. SEG/SLG 도 그 스텝엔 약한 패스 없이 항을 더하지 않는다 — sam-extra
  ``_post_cfg`` 의 ``has_pert = not adg_skipped and ...``. 그 스텝의 CFG 단계는 cond 를 그대로 내고
  (uncond := cond), APG 모멘텀은 비우고 SMC e_prev 는 둔다. DCW 는 원본 post-CFG 처럼 그 스텝에도 돈다
  (계획 §3.3 F, sam-extra DCW-F).
- PAG rescale 자동 끄기는 APG 가 실제로 걸렸을 때만(sam-extra ``_apply_perturbation`` 의 ``apg_governs``).
- RDC 스위치 키가 없으면 켠 것으로 읽는다(원본엔 스위치가 없고 sam-extra arg 58 기본 True).
- CFG≈1: SMC/CWM 은 원본처럼 돈다(``disable_cfg1_optimization``, origin:
  namemechan/ComfyUI-DCW@66aaf9dd:dcw_node.py:877-878). APG 만 sam-extra 처럼 건너뛰고, APG 만 켜면
  uncond 패스를 강제하지 않는다. ADG 도 강제하지 않는다.
- CNS 는 스위트가 ``guidance_cns.apply_cns`` 래퍼(가장 바깥 SAMPLER_SAMPLE)로 건다 — 옛 ``forge_neo_cns``
  넘김 dict 는 없다.
- 키가 없을 때의 값은 원본 노드 기본값이다(아래 ORIGIN_* 표, 출처 줄 번호).

DCW(+a) 원본은 GPL-3.0 이라 코드는 옮기지 않았다. 아래 표는 원본 INPUT_TYPES 의 기본값(숫자)이다.
torch 는 지연 import 한다(tests/_optional_deps). torch 를 쓰는 클래스는 @requires_torch (CPU 전용).
"""
from __future__ import annotations

import json
import math
import unittest
from unittest import mock

from comfy_custom_nodes.ai_studio_forge_parity import guidance, guidance_cns, guidance_dcw
from tests import test_forge_parity_pag_origin as pag_harness
from tests._optional_deps import bind_torch, requires_torch

torch = None  # @requires_torch 클래스의 setUpClass 가 bind_torch 로 채운다

# origin: namemechan/ComfyUI-DCW@66aaf9dd:dcw_node.py — INPUT_TYPES 기본값
#   :637-638 lambda_l 0.05, :653-654 lambda_h 0.01, :675-676 alpha_l 0.0, :691-692 alpha_h 0.0,
#   :713-716 smc_preset "Off", :732-733 smc_lambda 6.0, :744-745 smc_k 0.1, :761-762 rdc_tau 0.0,
#   :784-785 rdc_alpha_ll 0.03, :800-801 rdc_alpha_hh 0.0
ORIGIN_DCW_DEFAULTS = {
    "lambda_l": 0.05, "lambda_h": 0.01, "alpha_l": 0.0, "alpha_h": 0.0,
    "smc_lambda": 6.0, "smc_k": 0.1, "rdc_tau": 0.0, "rdc_alpha_ll": 0.03, "rdc_alpha_hh": 0.0,
}
# origin: namemechan/comfyui-cns_sampler_patch@42278b13:cns_sampler_patch.py:396-437 (:397, :410, :424)
ORIGIN_CNS_DEFAULTS = {"strength": 1.0, "gamma_power": 0.5, "gamma_scale": 2.0}

# 훅 함수 이름(__qualname__) — 어느 기능이 어느 자리에 걸렸는지 본다.
CFG_STAGE = "_patch_cfg_stage.<locals>.cfg_stage"
PAG_POST = "AnimaSafePAG.patch.<locals>.post_cfg_function"   # 원본 PAG 노드(벤더)
SEG_SLG_POST = "_patch_seg_slg_guidance.<locals>.post_cfg"
DCW_POST = "_dcw_hook.<locals>.dcw_post_cfg"
ADG_CALC = "_patch_adaptive_guidance.<locals>.calculate"
PAG_CALC = "AnimaSafePAG.patch.<locals>.calc_cond_batch_with_pag"


def _names(functions):
    return [function.__qualname__ for function in functions]


class _Patcher:
    """ModelPatcher 표면(torch 없음): clone 은 Comfy 처럼 model_options 를 중첩 dict 까지 복사한다."""

    def __init__(self):
        self.model = None
        self.model_options = {"transformer_options": {}}
        self.wrappers = {}

    def clone(self):
        clone = _Patcher()
        clone.model_options = _copy_nested(self.model_options)
        clone.wrappers = {kind: {key: list(items) for key, items in keyed.items()}
                          for kind, keyed in self.wrappers.items()}
        return clone

    def add_wrapper_with_key(self, wrapper_type, key, wrapper):
        self.wrappers.setdefault(wrapper_type, {}).setdefault(key, []).append(wrapper)

    def remove_wrappers_with_key(self, wrapper_type, key):
        self.wrappers.get(wrapper_type, {}).pop(key, None)


def _copy_nested(options):
    """ComfyUI comfy/patcher_extension.py copy_nested_dicts (샘플링 실행마다 model_options 를 이렇게 복사)."""
    copied = dict(options)
    for key, value in options.items():
        if isinstance(value, dict):
            copied[key] = _copy_nested(value)
        elif isinstance(value, list):
            copied[key] = list(value)
    return copied


def _suite(model, **settings):
    (patched,) = guidance.ForgeNeoAnimaGuidanceSuite().patch(
        model, None, None, None, True, json.dumps(settings)
    )
    return patched


class TestSuiteWiringWithoutTorch(unittest.TestCase):
    """등록 모양·폴백·CNS 래퍼 — torch 없이(--quick 에서 돈다)."""

    def test_cfg_stage_adg_and_dcw_slots(self):
        patched = _suite(
            _Patcher(),
            guid_cwm_enabled=True, guid_cwm_alpha_low=0.3, guid_cwm_alpha_high=0.15,
            guid_smc_enabled=True, guid_smc_preset="Custom", guid_apg_enabled=True,
            guid_adg_enabled=True, guid_dcw_enabled=True,
        )
        options = patched.model_options
        self.assertEqual(options["sampler_cfg_function"].__qualname__, CFG_STAGE)
        self.assertEqual(_names(options["sampler_post_cfg_function"]), [DCW_POST])
        self.assertEqual(options["sampler_calc_cond_batch_function"].__qualname__, ADG_CALC)
        self.assertEqual(options[guidance.ADG_STEP_KEY], {"skipped": False})
        # SMC/CWM 이 있으면 원본처럼 CFG 1 에서도 uncond 를 돌린다(dcw_node.py:877-878).
        self.assertTrue(options["disable_cfg1_optimization"])

    def test_missing_keys_fall_back_to_the_original_defaults(self):
        with mock.patch.object(guidance, "patch_dcw", wraps=guidance_dcw.patch_dcw) as patch_dcw:
            _suite(_Patcher(), guid_cwm_enabled=True, guid_smc_enabled=True,
                   guid_dcw_enabled=True, guid_rdc_enabled=True)
        cfg_call, dcw_call = (call.kwargs for call in patch_dcw.call_args_list)
        self.assertFalse(cfg_call["dcw_enabled"])
        self.assertEqual(
            {key: cfg_call[key] for key in ("alpha_l", "alpha_h", "smc_lambda", "smc_k")},
            {key: ORIGIN_DCW_DEFAULTS[key] for key in ("alpha_l", "alpha_h", "smc_lambda", "smc_k")},
        )
        # 원본 기본값 Off 를 SMC 스위치가 켜면 sam-extra 처럼 Custom(λ, k)이다.
        self.assertEqual(cfg_call["smc_preset"], "Custom")
        self.assertTrue(dcw_call["dcw_enabled"])
        self.assertEqual(
            {key: dcw_call[key] for key in ("lambda_l", "lambda_h", "rdc_tau", "rdc_alpha_ll", "rdc_alpha_hh")},
            {key: ORIGIN_DCW_DEFAULTS[key]
             for key in ("lambda_l", "lambda_h", "rdc_tau", "rdc_alpha_ll", "rdc_alpha_hh")},
        )
        patched = _suite(_Patcher(), guid_cns_enabled=True)
        self.assertEqual(guidance_cns.model_cns_settings(patched), ORIGIN_CNS_DEFAULTS)

    def test_smc_preset_is_read_like_sam_extra(self):
        # sam3ext/guidance/cwm_smc.py normalize_smc_preset: 모르는 이름 → Off.
        # scripts/anima_safe_pag.py: SMC 가 켜졌는데 Off 면 Custom.
        for value, expected in (
            ("Auto", "Auto"), (" auto ", "Auto"), ("SDXL", "SDXL"), ("cosmos / wan", "Cosmos / Wan"),
            ("Custom", "Custom"), ("Off", "Custom"), ("bogus", "Custom"), ("", "Custom"), (None, "Custom"),
        ):
            with self.subTest(value=value):
                self.assertEqual(guidance._suite_smc_preset(value), expected)

    def test_rdc_runs_only_inside_dcw(self):
        # origin: namemechan/ComfyUI-DCW@66aaf9dd:dcw_node.py:855-856 — rdc_on = rdc_tau > 0, DCW 훅 안에서만.
        # 원본엔 RDC 스위치가 없다. sam-extra arg 58(scripts/anima_safe_pag.py rdc_switch)은 기본 True 인
        # 거부 전용 자리 — 스위트도 키가 없으면 켜고, 명시한 False 만 막는다.
        base = _Patcher()
        self.assertIs(_suite(base, guid_rdc_enabled=True, guid_rdc_tau=0.2), base)
        self.assertIs(_suite(base, guid_rdc_tau=0.2), base)
        with mock.patch.object(guidance, "patch_dcw", wraps=guidance_dcw.patch_dcw) as patch_dcw:
            _suite(base, guid_dcw_enabled=True, guid_rdc_tau=0.2)
            _suite(base, guid_dcw_enabled=True, guid_rdc_enabled=True, guid_rdc_tau=0.2)
            _suite(base, guid_dcw_enabled=True, guid_rdc_enabled=False, guid_rdc_tau=0.2)
            _suite(base, guid_dcw_enabled=True, guid_rdc_enabled=None, guid_rdc_tau=0.2)
        self.assertEqual([call.kwargs["rdc_tau"] for call in patch_dcw.call_args_list], [0.2, 0.2, 0.0, 0.2])

    def test_pag_rescale_auto_off_needs_apg_to_be_installed(self):
        # sam-extra _apply_perturbation: apg_governs = _APG["on"] and apg_autooff_rescale and not base_skipped
        # — APG 가 실제로 돌 때만 rescale 을 끈다. 팩은 앞 노드의 sampler_cfg_function 이 있으면 APG 를 걸지
        # 않으므로(DCW(+a) 처럼 비킴) 그때는 rescale 을 살린다.
        def earlier(args):
            return args["cond"]

        pag = {"guid_enabled": True, "guid_attn_method": "PAG", "guid_rescale": 0.2}
        for settings, earlier_cfg, expected in (
            ({}, False, 0.2),
            ({"guid_apg_enabled": True}, False, 0.0),
            ({"guid_cfg_mode": "APG"}, False, 0.0),
            ({"guid_apg_enabled": True, "guid_apg_autooff": False}, False, 0.2),
            ({"guid_apg_enabled": True}, True, 0.2),
            ({"guid_apg_enabled": True, "guid_cwm_enabled": True, "guid_cwm_alpha_low": 0.3}, True, 0.2),
        ):
            with self.subTest(settings=settings, earlier_cfg=earlier_cfg):
                base = _Patcher()
                if earlier_cfg:
                    base.model_options["sampler_cfg_function"] = earlier
                with mock.patch.object(
                    guidance, "_patch_perturbation_guidance", side_effect=lambda model, **_kw: model
                ) as perturbation:
                    if earlier_cfg:
                        with self.assertLogs("ai_studio_forge_parity", "WARNING"):
                            _suite(base, **pag, **settings)
                    else:
                        _suite(base, **pag, **settings)
                self.assertEqual(perturbation.call_args.kwargs["rescale"], expected)

    def test_cfg_1_optimization_is_only_disabled_for_smc_and_cwm(self):
        for settings, forced in (
            ({"guid_adg_enabled": True}, False),
            ({"guid_apg_enabled": True}, False),
            ({"guid_apg_enabled": True, "guid_adg_enabled": True}, False),
            ({"guid_cwm_enabled": True, "guid_cwm_alpha_low": 0.3}, True),
            ({"guid_smc_enabled": True, "guid_smc_preset": "Auto", "guid_apg_enabled": True}, True),
        ):
            with self.subTest(settings=settings):
                options = _suite(_Patcher(), **settings).model_options
                self.assertEqual(bool(options.get("disable_cfg1_optimization")), forced)
        # 앞 노드가 켠 플래그는 APG 만 켜도 그대로 둔다.
        base = _Patcher()
        base.model_options["disable_cfg1_optimization"] = True
        self.assertTrue(_suite(base, guid_apg_enabled=True).model_options["disable_cfg1_optimization"])

    def test_an_earlier_cfg_function_is_left_alone(self):
        # DCW(+a): 다른 노드의 sampler_cfg_function 이 있으면 경고하고 CWM/SMC(와 팩 APG)를 건너뛴다.
        base = _Patcher()

        def earlier(args):
            return args["cond"]

        base.model_options["sampler_cfg_function"] = earlier
        with self.assertLogs("ai_studio_forge_parity", "WARNING"):
            patched = _suite(base, guid_cwm_enabled=True, guid_cwm_alpha_low=0.3,
                             guid_apg_enabled=True, guid_dcw_enabled=True)
        self.assertIs(patched.model_options["sampler_cfg_function"], earlier)
        self.assertEqual(_names(patched.model_options["sampler_post_cfg_function"]), [DCW_POST])

    def test_cns_wrapper_is_installed_by_the_suite(self):
        base = _Patcher()
        base.add_wrapper_with_key("sampler_sample", "other.pack", object())
        values = {"strength": 0.5, "gamma_power": 0.75, "gamma_scale": 3.0}
        patched = _suite(base, guid_cns_enabled=True, guid_dcw_enabled=True,
                         **{f"guid_cns_{name}": value for name, value in values.items()})
        self.assertEqual(guidance_cns.model_cns_settings(patched), values)
        self.assertEqual(list(patched.wrappers["sampler_sample"])[0], guidance_cns.CNS_WRAPPER_KEY)
        self.assertEqual(len(patched.wrappers["sampler_sample"][guidance_cns.CNS_WRAPPER_KEY]), 1)
        self.assertNotIn("forge_neo_cns", patched.model_options)
        self.assertEqual(_names(patched.model_options["sampler_post_cfg_function"]), [DCW_POST])
        self.assertIsNone(guidance_cns.model_cns_settings(base))
        self.assertIs(_suite(base, guid_cns_enabled=False), base)


# ── torch(CPU) 로 샘플링 한 스텝씩 ─────────────────────────────────────────────
# 가짜 Anima 모델·원본 PAG 모듈 로딩은 tests/test_forge_parity_pag_origin 의 것을 쓴다.

def _sampling_function(model, x, sigma, cond, uncond, cfg, options, trace):
    """ComfyUI comfy/samplers.py sampling_function(:609-627) + cfg_function(:592-605), ComfyUI 387f98aa.

    ``trace`` 에 스텝의 예측, CFG 단계 결과, post-CFG 함수마다의 결과를 남긴다.
    """
    optimized = math.isclose(cfg, 1.0) and not options.get("disable_cfg1_optimization", False)
    uncond_ = None if optimized else uncond
    conds = [cond, uncond_]
    calc = options.get("sampler_calc_cond_batch_function")
    if calc is not None:
        out = calc({"conds": conds, "input": x, "sigma": sigma, "model": model, "model_options": options})
    else:
        out = pag_harness._calc_cond_batch(model, conds, x, sigma, options)
    cond_pred, uncond_pred = out[0], out[1]
    if "sampler_cfg_function" in options:
        args = {
            "cond": x - cond_pred, "uncond": x - uncond_pred, "cond_scale": cfg, "timestep": sigma,
            "input": x, "sigma": sigma, "cond_denoised": cond_pred, "uncond_denoised": uncond_pred,
            "model": model, "model_options": options, "input_cond": cond, "input_uncond": uncond_,
        }
        result = x - options["sampler_cfg_function"](args)
    else:
        result = uncond_pred + (cond_pred - uncond_pred) * cfg
    trace.update(x=x, sigma=sigma, cond_pred=cond_pred, uncond_pred=uncond_pred, cfg=result, post=[])
    for function in options.get("sampler_post_cfg_function", []):
        result = function({
            "denoised": result, "cond": cond, "uncond": uncond_, "cond_scale": cfg, "model": model,
            "uncond_denoised": uncond_pred, "cond_denoised": cond_pred, "sigma": sigma,
            "model_options": options, "input": x,
        })
        trace["post"].append(result)
    trace["result"] = result
    return result


def _run(patcher, sigmas, cfg, seed=11, after_step=None):
    """한 샘플링 실행: model_options 를 실행마다 복사하고(Comfy), 스텝별 trace 목록을 돌려준다."""
    generator = torch.Generator().manual_seed(seed)
    shape = (1, 6, 8)
    noise = torch.randn(shape, generator=generator)
    cond = {"cross": torch.randn(shape, generator=generator)}
    uncond = {"cross": torch.randn(shape, generator=generator)}
    options = _copy_nested(patcher.model_options)
    traces = []
    with patcher.patched():
        for value in sigmas[:-1]:
            sigma = value.reshape(1)
            trace = {}
            _sampling_function(patcher.model, noise * sigma, sigma, cond, uncond, cfg, options, trace)
            if after_step is not None:
                after_step(options, trace)
            traces.append(trace)
    return traces


PAG = {
    "guid_enabled": True, "guid_attn_method": "PAG", "guid_scale": 4.0, "guid_block_indices": "1",
    "guid_official_strength": 0.75, "guid_start_percent": 0.0, "guid_end_percent": 0.7,
    "guid_rescale": 0.2, "guid_rescale_mode": "full",
}
SMC_CWM = {
    "guid_smc_enabled": True, "guid_smc_preset": "Custom", "guid_smc_lambda": 6.0, "guid_smc_k": 0.2,
    "guid_cwm_enabled": True, "guid_cwm_alpha_low": 0.3, "guid_cwm_alpha_high": 0.15,
}
APG = {"guid_apg_enabled": True, "guid_apg_eta": 0.3, "guid_apg_norm": 15.0, "guid_apg_momentum": 0.5}
DCW = {"guid_dcw_enabled": True, "guid_dcw_lambda_low": 0.05, "guid_dcw_lambda_high": 0.01}


def _dcw(value, trace, low=0.05, high=0.01):
    return guidance_dcw.apply_dcw(value, trace["x"], trace["sigma"], low, high)


@requires_torch
class TestSuiteOrderMatchesSamExtra(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        bind_torch(globals())
        bind_torch(vars(pag_harness))

    def test_hooks_register_in_sam_extra_order(self):
        # sam-extra _post_cfg: _apply_cfg_base(SMC→APG→CWM) → _apply_perturbation(PAG/SEG/SLG) → apply_dcw.
        with pag_harness._comfy_runtime():
            patched = _suite(
                pag_harness._build_model(),
                **PAG, **SMC_CWM, **APG, **DCW,
                guid_slg_on=True, guid_slg_blocks="2", guid_slg_scale=3.0,
                guid_adg_enabled=True, guid_rdc_enabled=True, guid_rdc_tau=0.1,
            )
        options = patched.model_options
        self.assertEqual(options["sampler_cfg_function"].__qualname__, CFG_STAGE)
        self.assertEqual(_names(options["sampler_post_cfg_function"]), [PAG_POST, SEG_SLG_POST, DCW_POST])
        # 원본 PAG 의 calc 가 바깥, ADG 가 그 previous_calc (origin :237).
        self.assertEqual(options["sampler_calc_cond_batch_function"].__qualname__, PAG_CALC)
        previous = [
            cell.cell_contents for cell in options["sampler_calc_cond_batch_function"].__closure__
            if callable(cell.cell_contents) and getattr(cell.cell_contents, "__qualname__", "") == ADG_CALC
        ]
        self.assertEqual(len(previous), 1)

    def test_each_step_is_dcw_of_pag_of_the_cfg_stage(self):
        with pag_harness._comfy_runtime():
            base = pag_harness._build_model()
            sigmas = pag_harness._simple_sigmas(base.model.model_sampling, 8)
            patched = _suite(base, **PAG, **SMC_CWM, **APG, **DCW)
            traces = _run(patched, sigmas, 5.0)
        state = {}   # SMC/APG 상태를 따로 굴려 CFG 단계를 다시 계산한다
        pag_active = 0
        for step, trace in enumerate(traces):
            with self.subTest(step=step):
                x, sigma = trace["x"], trace["sigma"]
                apg_state = state.setdefault("apg", {})

                def apg(error):
                    return guidance_dcw._apply_apg_error(
                        error, trace["cond_pred"], eta=0.3, norm_threshold=15.0, momentum=0.5,
                        sigma=sigma, state=apg_state,
                    )

                cfg_noise = guidance_dcw._guided_noise(
                    x - trace["cond_pred"], x - trace["uncond_pred"], sigma, 5.0,
                    alpha_low=0.3, alpha_high=0.15, smc_lambda=6.0, smc_k=0.2,
                    smc_state=state.setdefault("smc", {}), apg=apg,
                )
                self.assertTrue(torch.equal(trace["cfg"], x - cfg_noise), "CFG stage (SMC→APG→CWM)")
                pag_out, final = trace["post"]
                pag_active += not torch.equal(pag_out, trace["cfg"])
                self.assertTrue(torch.equal(final, _dcw(pag_out, trace)), "DCW runs on the PAG result")
                self.assertFalse(torch.equal(final, pag_out))
        # PAG 창(end .7): 8 simple 스텝 중 앞 6 스텝 (test_forge_parity_pag_origin 과 같은 표).
        self.assertEqual(pag_active, 6)

    def test_adg_skipped_step_zeroes_pag_and_keeps_dcw(self):
        # 8 simple 스텝, ADG start .3 interval 2 → 3·5·7 번 스텝이 cond 만 돈다(test_forge_parity_pag_origin).
        skipped = [False, False, False, True, False, True, False, True]
        apg_filled, e_prev = [], []

        def after_step(options, _trace):
            apg_filled.append(bool(options.get(guidance_dcw.APG_STATE_KEY)))
            e_prev.append(options[guidance_dcw.SMC_STATE_KEY].get("e_prev"))

        settings = dict(**PAG, **SMC_CWM, **APG, **DCW, guid_adg_enabled=True,
                        guid_adg_start=0.3, guid_adg_interval=2)
        with pag_harness._comfy_runtime():
            base = pag_harness._build_model()
            sigmas = pag_harness._simple_sigmas(base.model.model_sampling, 8)
            traces = _run(_suite(base, **settings), sigmas, 5.0, after_step=after_step)
            self.assertEqual(base.model.batch_sizes, [1 if skip else 3 for skip in skipped])
        for step, (trace, skip) in enumerate(zip(traces, skipped)):
            x = trace["x"]
            pag_out, final = trace["post"]
            with self.subTest(step=step, skipped=skip):
                self.assertTrue(torch.equal(final, _dcw(pag_out, trace)))   # DCW 는 매 스텝
                if skip:
                    self.assertIs(trace["uncond_pred"], trace["cond_pred"])  # uncond := cond
                    self.assertTrue(torch.equal(trace["cfg"], x - (x - trace["cond_pred"])))
                    self.assertTrue(torch.equal(pag_out, trace["cfg"]), "PAG term must be 0")
                    self.assertFalse(apg_filled[step], "APG momentum cleared")
                    self.assertIs(e_prev[step], e_prev[step - 1], "SMC e_prev kept")
                else:
                    self.assertTrue(apg_filled[step])
                    self.assertIsNot(e_prev[step], e_prev[step - 1] if step else None)
                    if step < 3:   # PAG 창 안의 CFG 스텝
                        self.assertFalse(torch.equal(pag_out, trace["cfg"]))

    def test_adg_skipped_step_runs_no_seg_or_slg(self):
        # sam-extra _post_cfg: has_pert = not adg_skipped and ... — ADG 가 cond 만 돌린 스텝엔 SEG/SLG 항도,
        # 약한 패스도 없다(interval 0 이면 _apply_attach 가 창을 ADG 시작점에서 자르는 것과 같다).
        # 8 simple 스텝, start .3(σ 진행률로 3번 스텝부터): interval 2 → 3·5·7, interval 0 → 3~7 이 cond 만.
        # 가짜 모델의 attention 은 attn1_patch 를 부르지 않아 SEG 항은 0 이지만 약한 패스(batch 1)는 돈다.
        settings = dict(
            **DCW, guid_enabled=True, guid_attn_method="SEG", guid_scale=4.0, guid_block_indices="1",
            guid_official_strength=0.75, guid_slg_on=True, guid_slg_blocks="2", guid_slg_scale=3.0,
            guid_start_percent=0.0, guid_end_percent=1.0, guid_rescale=0.0,
            guid_adg_enabled=True, guid_adg_start=0.3,
        )
        for interval, skipped in ((2, [False, False, False, True, False, True, False, True]),
                                  (0, [False, False, False] + [True] * 5)):
            with self.subTest(interval=interval), pag_harness._comfy_runtime():
                base = pag_harness._build_model()
                sigmas = pag_harness._simple_sigmas(base.model.model_sampling, 8)
                patched = _suite(base, **settings, guid_adg_interval=interval)
                self.assertEqual(_names(patched.model_options["sampler_post_cfg_function"]),
                                 [SEG_SLG_POST, DCW_POST])
                self.assertTrue(hasattr(patched.model_options["sampler_post_cfg_function"][0], "__wrapped__"))
                traces = _run(patched, sigmas, 5.0)
                # CFG 스텝: cond+uncond(2) → SEG 약한 패스(1) → SLG 약한 패스(1). 건너뛴 스텝: cond(1) 하나.
                expected = []
                for skip in skipped:
                    expected += [1] if skip else [2, 1, 1]
                self.assertEqual(base.model.batch_sizes, expected)
                for step, (trace, skip) in enumerate(zip(traces, skipped)):
                    seg_slg_out, final = trace["post"]
                    with self.subTest(step=step, skipped=skip):
                        self.assertEqual(trace["uncond_pred"] is trace["cond_pred"], skip)
                        if skip:
                            self.assertTrue(torch.equal(trace["cfg"], trace["cond_pred"]))
                            self.assertIs(seg_slg_out, trace["cfg"], "no SEG/SLG term")
                            self.assertTrue(torch.equal(final, _dcw(trace["cond_pred"], trace)), "DCW(cond)")
                        else:
                            self.assertFalse(torch.equal(seg_slg_out, trace["cfg"]), "SLG term")
                            self.assertTrue(torch.equal(final, _dcw(seg_slg_out, trace)))

    def test_cfg_1_skips_apg_and_runs_smc_and_cwm(self):
        with pag_harness._comfy_runtime():
            base = pag_harness._build_model()
            sigmas = pag_harness._simple_sigmas(base.model.model_sampling, 6)
            cwm = {k: v for k, v in SMC_CWM.items() if k.startswith("guid_cwm")}
            for cfg, same in ((1.0, True), (5.0, False)):
                with self.subTest(cfg=cfg):
                    with_apg = _run(_suite(base, **cwm, **APG), sigmas, cfg)
                    without = _run(_suite(base, **cwm), sigmas, cfg)
                    equal = [torch.equal(a["result"], b["result"]) for a, b in zip(with_apg, without)]
                    self.assertEqual(all(equal), same)
            # CWM 은 CFG 1 에서도 돈다(원본): 그냥 CFG 1 결과와 다르다.
            plain = _run(base, sigmas, 1.0)
            cwm_only = _run(_suite(base, **cwm), sigmas, 1.0)
            self.assertFalse(all(torch.equal(a["result"], b["result"]) for a, b in zip(plain, cwm_only)))
            # APG 만: CFG 1 에서 uncond 패스가 없고 결과는 cond 예측이다.
            base.model.batch_sizes.clear()
            apg_only = _run(_suite(base, **APG), sigmas, 1.0)
            self.assertEqual(base.model.batch_sizes, [1] * (len(sigmas) - 1))
            for step, (a, b) in enumerate(zip(apg_only, plain)):
                torch.testing.assert_close(a["result"], b["result"], rtol=0.0, atol=1e-6, msg=f"step {step}")


if __name__ == "__main__":
    unittest.main()
