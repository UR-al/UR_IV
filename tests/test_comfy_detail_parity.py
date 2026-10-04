"""ComfyUI 팩 1.6.0 디테일 가이던스 = sam-extra v0.30.0 — 같은 텐서, 같은 수 (CPU).

팩은 확장의 식을 ComfyUI post-CFG 훅용으로 다시 썼다(comfy_custom_nodes/ai_studio_forge_parity/guidance_detail.py ·
guidance_s2.py · guidance_dcw 의 Adaptive SMC · guidance_optimal_scale.py). 여기서는 설치된 확장의 sam3ext.guidance
모듈(Forge 없이 import 된다 — torch 만 쓴다)과 팩을 같은 입력으로 돌려 결과를 대조한다. 확장이 없으면 skip,
AISTUDIO_REQUIRE_FORGE_EXT=1 이면 실패(tests/_sam_extra_ext). 확장 의존이 없는 팩 동작(창·패스 표시·노드 규칙)은
아래 PackBehaviourTests 가 본다.

팩 1.7.0: Euler (SMEA) Dy(CFG++ 판과 보통 판)의 보조 스텝 평가(transformer_options["sam_extra_substep"])는
스텝 단위 상태를 읽기만 하고 갱신·초기화하지 않는다 — 확장 scripts/anima_safe_pag.py 의 규칙(SubstepRuleSourceTests
가 그 자리를 확인한다)과 tests/test_extra_samplers_guidance.py 의 시나리오. SubstepCfgStateTests(SMC·APG·RDC·ADG),
PlainEulerDySubstepTests(실제 팩 Dy 샘플러로 샘플링 — 본 스텝들 = 보조 스텝 없이 다시 돌린 것).

torch 는 지연 import 한다(tests/_optional_deps).
"""
import ast
import importlib
import json
import sys
import types
import unittest
from unittest import mock

from tests._optional_deps import bind_torch, requires_torch
from tests._sam_extra_ext import EXT_ROOT, requires_extension

torch = None  # @requires_torch 클래스의 setUpClass 가 bind_torch 로 채운다


def _ext_module(name: str):
    root = str(EXT_ROOT)
    if root not in sys.path:
        sys.path.insert(0, root)
    return importlib.import_module(f"sam3ext.guidance.{name}")


def _ext_function(relative: str, name: str):
    """확장 스크립트의 순수 함수 하나 — Forge·gradio 를 import 하지 않게 그 함수 정의만 떼어 실행한다."""
    source = (EXT_ROOT / relative).read_text(encoding="utf-8")
    node = next(item for item in ast.parse(source).body
                if isinstance(item, ast.FunctionDef) and item.name == name)
    namespace = {"torch": torch}
    exec(compile(ast.Module(body=[node], type_ignores=[]), f"{relative}:{name}", "exec"), namespace)
    return namespace[name]


def _same(case, got, want):
    torch.testing.assert_close(got, want, rtol=0, atol=0)


@requires_torch
@requires_extension
class DetailMathParityTests(unittest.TestCase):
    """팩 함수 = 확장 함수(비트 단위). 입력은 Anima 꼴 5-D 잠재 [B, C, T, H, W]."""

    @classmethod
    def setUpClass(cls):
        bind_torch(globals())
        from comfy_custom_nodes.ai_studio_forge_parity import guidance_detail, guidance_s2
        cls.pack, cls.pack_s2 = guidance_detail, guidance_s2
        cls.tsr, cls.history = _ext_module("tsr"), _ext_module("history")
        cls.hiflow, cls.trajectory = _ext_module("hiflow"), _ext_module("trajectory")
        cls.s2 = _ext_module("s2")

    def _randn(self, *shape, seed):
        return torch.randn(*shape, generator=torch.Generator().manual_seed(seed))

    def test_tsr(self):
        x, d = self._randn(2, 16, 1, 8, 8, seed=0), self._randn(2, 16, 1, 8, 8, seed=1)
        sigmas = (0.03, 0.3, 0.7, 0.99, 1.0, torch.tensor([0.4, 0.85]), torch.tensor([1.0, 0.5]))
        for flow in (True, False):
            for sigma in sigmas:
                for k, tsr_sigma in ((0.95, 1.0), (0.9, 3.0), (1.2, 0.5), (1.0, 1.0)):
                    with self.subTest(flow=flow, sigma=sigma, k=k):
                        want = self.tsr.apply_tsr(d, x, sigma, k=k, tsr_sigma=tsr_sigma, flow=flow)
                        got = self.pack.apply_tsr(d, x, sigma, k=k, tsr_sigma=tsr_sigma, flow=flow)
                        _same(self, got, want)

    def test_momentum_and_higs_over_a_run(self):
        """Euler 처럼 스텝마다 한 번 + Heun 처럼 같은 σ 반복 + 다음 실행(σ 상승) — 역할·기록·출력이 같다."""
        ext_state, pack_state = self.history.HistoryState(), self.pack.HistoryState()
        sigmas = (0.99, 0.9, 0.9, 0.8, 0.6, 0.6, 0.4, 0.2, 0.05, 0.99, 0.7)
        for index, sigma in enumerate(sigmas):
            x = self._randn(1, 4, 1, 16, 16, seed=100 + index)
            d = self._randn(1, 4, 1, 16, 16, seed=200 + index)
            for flow in (True,):
                role = self.history.step_role(ext_state, sigma, True)
                self.assertEqual(self.pack.step_role(pack_state, sigma, True), role)
                active = 0.30 <= sigma <= 0.95
                want = self.history.apply_mg(d, x, sigma, ext_state, alpha=0.5, beta=0.6, normalize=True,
                                             active=active, role=role)
                got = self.pack.apply_mg(d, x, sigma, pack_state, alpha=0.5, beta=0.6, normalize=True,
                                         active=active, role=role)
                _same(self, got, want)
                want = self.history.apply_higs(want, sigma, ext_state, weight=1.75, eta=0.25, alpha=0.75,
                                               cutoff=0.05, t_min=0.4, t_max=1.0, flow=flow, role=role)
                got = self.pack.apply_higs(got, sigma, pack_state, weight=1.75, eta=0.25, alpha=0.75,
                                           cutoff=0.05, t_min=0.4, t_max=1.0, flow=flow, role=role)
                _same(self, got, want)
        self.assertEqual(pack_state.counters, ext_state.counters)

    def test_hiflow_trajectory_and_alignment(self):
        ext_traj, pack_traj = self.trajectory.Trajectory(), self.pack.Trajectory()
        for index, sigma in enumerate((1.0, 0.8, 0.6, 0.4, 0.2, 0.6)):   # 같은 σ 다시 기록 = 바꿔 쓴다
            record = self._randn(1, 4, 1, 8, 8, seed=300 + index)
            ext_traj.record(sigma, record)
            pack_traj.record(sigma, record)
        self.assertEqual(pack_traj.sigmas, ext_traj.sigmas)
        for sigma in (1.2, 0.95, 0.7, 0.5, 0.21, 0.05):
            _same(self, pack_traj.at(sigma), ext_traj.at(sigma))
        ext_state, pack_state = self.hiflow.HiFlowState(), self.pack.HiFlowState()
        walked = [0.6, 0.45, 0.3, 0.15, 0.0]
        evaluations = [(0, 0.6, True), (1, 0.45, True), (1, 0.45, False), (2, 0.3, True), (3, 0.15, True)]
        for index, (k, sigma, step_start) in enumerate(evaluations):
            d = self._randn(1, 4, 1, 16, 16, seed=400 + index)
            ref_e = self.hiflow.resize_latent(ext_traj.at(sigma), (16, 16))
            ref_p = self.pack.resize_latent(pack_traj.at(sigma), (16, 16))
            _same(self, ref_p, ref_e)
            position, steps, _on = self.pack.hiflow_position(walked, sigma)
            self.assertEqual((position, steps), (float(k), 4))
            weight = self.hiflow.step_weight(position, steps)
            self.assertEqual(self.pack.step_weight(position, steps), weight)
            want = self.hiflow.apply_hiflow(d, sigma, ref_e, ext_state, alpha=1.0, beta=0.5, cutoff=0.2,
                                            weight=weight, step_start=step_start)
            got = self.pack.apply_hiflow(d, sigma, ref_p, pack_state, alpha=1.0, beta=0.5, cutoff=0.2,
                                         weight=weight, step_start=step_start)
            _same(self, got, want)
        self.assertEqual(pack_state.counters, ext_state.counters)

    def test_s2_draws_are_the_extensions(self):
        for blocks in (28, 40, 52):
            eligible = self.s2.default_eligible(blocks)
            self.assertEqual(self.pack_s2.default_eligible(blocks), eligible)
            for ratio in (0.01, 0.05, 0.2, 0.5):
                self.assertEqual(self.pack_s2.count_for(ratio, len(eligible)), self.s2.count_for(ratio, len(eligible)))
                for seed in (0, 11, 2 ** 33 + 5):
                    for tag in ("base", "hires"):
                        for draw in range(6):
                            self.assertEqual(self.pack_s2.draw_blocks(eligible, ratio, seed, tag, draw),
                                             self.s2.draw_blocks(eligible, ratio, seed, tag, draw))


@requires_torch
@requires_extension
class AdaptiveSmcAndOptimalScaleParityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        bind_torch(globals())
        from comfy_custom_nodes.ai_studio_forge_parity import guidance_dcw, guidance_optimal_scale
        cls.dcw, cls.optimal = guidance_dcw, guidance_optimal_scale
        cls.cwm_smc = _ext_module("cwm_smc")

    def test_adaptive_smc_in_comfy_noise_space_equals_the_extension_in_x0_space(self):
        """팩은 Comfy 의 noise 공간 cond−uncond(= −(x0_c − x0_u))에, 확장은 x0 공간에 같은 제어기를 건다 — 홀함수라
        결과 x0 가 같다(부동소수 반올림만큼)."""
        generator = torch.Generator().manual_seed(7)
        ext_prev, pack_state = None, {}
        for sigma in (0.98, 0.85, 0.7, 0.55, 0.4):
            x = torch.randn(2, 4, 1, 16, 16, generator=generator)
            cond = torch.randn(2, 4, 1, 16, 16, generator=generator)
            uncond = torch.randn(2, 4, 1, 16, 16, generator=generator)
            want, ext_prev = self.cwm_smc.compose_cfg(
                cond, uncond, sigma, 4.5, "smc", 0.0, 0.0, 5.0, 0.0, ext_prev,
                smc_mode=self.cwm_smc.SMC_MODE_ADAPTIVE, smc_sigma=sigma, smc_alpha=0.2)
            noise = self.dcw._guided_noise(
                x - cond, x - uncond, sigma, 4.5, alpha_low=0.0, alpha_high=0.0, smc_lambda=5.0, smc_k=0.0,
                smc_state=pack_state, smc_mode=self.dcw.SMC_MODE_ADAPTIVE, smc_alpha=0.2, smc_sigma=sigma)
            torch.testing.assert_close(x - noise, want, rtol=1e-5, atol=1e-5)

    def test_optimal_scale_residual_is_the_extensions(self):
        residual = _ext_function("scripts/anima_cfg_optimal_scale.py", "_optimal_scale_residual")
        generator = torch.Generator().manual_seed(9)
        x, cond, uncond = (torch.randn(2, 16, 1, 8, 8, generator=generator) for _ in range(3))
        incoming = uncond + (cond - uncond) * 4.0
        sigma = torch.tensor([0.7, 0.7])
        for blend in (0.0, 0.25, 1.0):
            want = residual(incoming, x, cond, uncond, 4.0, blend, sigma)
            got = self.optimal.optimal_scale_residual(incoming, x, cond, uncond, 4.0, blend, sigma)
            torch.testing.assert_close(got, want, rtol=0, atol=0)


class _Patcher:
    """ModelPatcher 표면(clone·model_options) — 팩 패치가 쓰는 것만."""

    def __init__(self, model=None):
        self.model = model
        self.model_options = {"transformer_options": {}}

    def clone(self):
        clone = _Patcher(self.model)
        clone.model_options = json.loads(json.dumps({k: v for k, v in self.model_options.items()
                                                     if isinstance(v, (dict, bool))}))
        for key, value in self.model_options.items():
            if isinstance(value, list):
                clone.model_options[key] = list(value)
        return clone


@requires_torch
class PackBehaviourTests(unittest.TestCase):
    """확장 없이 보는 팩 규칙 — S² 창·뽑기 순번, 패스 표시, 디테일 함수 자리, Optimal Scale 건너뛰기."""

    @classmethod
    def setUpClass(cls):
        bind_torch(globals())
        from comfy_custom_nodes.ai_studio_forge_parity import (
            guidance, guidance_detail, guidance_optimal_scale, guidance_s2,
        )
        cls.guidance, cls.detail, cls.s2, cls.optimal = guidance, guidance_detail, guidance_s2, guidance_optimal_scale

    def _args(self, sigma, schedule, *, tag=None, noted=None):
        transformer = {"sample_sigmas": torch.tensor(schedule)}
        if tag:
            transformer[self.detail.PASS_KEY] = tag
        if noted is not None:
            from comfy_custom_nodes.ai_studio_forge_parity.guidance_common import PRE_DD_SIGMAS_KEY
            transformer[PRE_DD_SIGMAS_KEY] = torch.tensor([noted])
        return {"sigma": torch.tensor([sigma]), "model_options": {"transformer_options": transformer}}

    def test_s2_window_is_forges_step_counter_one_step_behind(self):
        schedule = [1.0, 0.9, 0.75, 0.5, 0.25, 0.0]   # 5 스텝
        fractions = [self.s2.forge_step_fraction(self._args(s, schedule)) for s in schedule[:-1]]
        self.assertEqual(fractions, [0.0, 0.0, 0.25, 0.5, 0.75])
        # Detail Daemon 이 모델 σ 를 줄여도 샘플러 σ(노트)로 찾는다
        self.assertEqual(self.s2.forge_step_fraction(self._args(0.4, schedule, noted=0.5)), 0.5)

    def test_s2_draw_index_counts_every_evaluation_and_restarts_per_pass(self):
        settings = self.s2.S2Settings(scale=0.25, ratio=0.05, blocks="", start=0.1, end=0.9, seed=11)
        run = self.s2.S2Run(settings, self.s2.default_eligible(28))
        schedule = [1.0, 0.8, 0.6, 0.0]
        drawn = [run.draw(self._args(s, schedule)) for s in (1.0, 0.8, 0.8, 0.6)]
        self.assertEqual(run.draws, 4)
        self.assertEqual(drawn, [frozenset(self.s2.draw_blocks(run.eligible, 0.05, 11, "base", i)) for i in range(4)])
        hires = run.draw(self._args(0.5, [0.5, 0.25, 0.0], tag=self.detail.PASS_HIRES))
        self.assertEqual(hires, frozenset(self.s2.draw_blocks(run.eligible, 0.05, 11, "hires", 0)))
        self.assertEqual(run.draws, 1)
        run.draw(self._args(1.0, schedule))   # 다음 생성(σ 상승) — 처음부터
        self.assertEqual(run.draws, 1)

    def test_s2_needs_omega_and_a_block(self):
        eligible = self.s2.default_eligible(28)
        self.assertTrue(self.s2.s2_active(self.s2.S2Settings(0.25, 0.05, "", 0.1, 0.9, 0), eligible))
        self.assertFalse(self.s2.s2_active(self.s2.S2Settings(0.0, 0.05, "", 0.1, 0.9, 0), eligible))
        self.assertFalse(self.s2.s2_active(self.s2.S2Settings(0.25, 0.0, "", 0.1, 0.9, 0), eligible))
        self.assertEqual(self.s2.eligible_blocks(self.s2.S2Settings(0.25, 0.05, "20-18, 99", 0.1, 0.9, 0), 28),
                         {18, 19, 20})
        self.assertIsNone(self.s2.s2_settings({"guid_slg_mode": "Fixed"}))
        self.assertIsNotNone(self.s2.s2_settings({"guid_slg_mode": " stochastic (S²) "}))

    def test_pass_tag_only_on_a_pass_aware_model(self):
        plain = _Patcher()
        self.assertIs(self.detail.with_pass_tag(plain, self.detail.PASS_BASE), plain)
        aware = self.detail.mark_pass_aware(plain)
        tagged = self.detail.with_pass_tag(aware, self.detail.PASS_HIRES)
        self.assertIsNot(tagged, aware)
        self.assertEqual(tagged.model_options["transformer_options"][self.detail.PASS_KEY], "hires")
        self.assertNotIn(self.detail.PASS_KEY, aware.model_options["transformer_options"])

    def _suite(self, **settings):
        (patched,) = self.guidance.ForgeNeoAnimaGuidanceSuite().patch(
            _Patcher(), None, None, None, True, json.dumps(settings))
        return patched

    def test_detail_functions_sit_between_the_perturbation_term_and_dcw(self):
        patched = self._suite(guid_tsr_enabled=True, guid_mg_enabled=True, guid_dcw_enabled=True,
                              guid_hiflow_enabled=True)
        names = [fn.__name__ for fn in patched.model_options["sampler_post_cfg_function"]]
        self.assertEqual(names, ["detail_stages", "dcw_post_cfg", "hiflow_record"])
        self.assertTrue(patched.model_options[self.detail.PASS_AWARE_KEY])
        # 디테일만 켜도 스위트가 붙고, HiFlow·S² 가 없으면 패스 표시를 하지 않는다
        patched = self._suite(guid_higs_enabled=True)
        self.assertEqual([fn.__name__ for fn in patched.model_options["sampler_post_cfg_function"]],
                         ["detail_stages"])
        self.assertNotIn(self.detail.PASS_AWARE_KEY, patched.model_options)
        # 끈 값(TSR k 1 · HiGS w 0)은 아무것도 걸지 않는다(확장의 _TSR on·higs_on 규칙)
        patched = self._suite(guid_tsr_enabled=True, guid_tsr_k=1.0, guid_higs_enabled=True, guid_higs_weight=0.0)
        self.assertNotIn("sampler_post_cfg_function", patched.model_options)

    def test_detail_stages_apply_in_the_extensions_order(self):
        """MG → HiGS → TSR — 같은 함수를 손으로 그 순서대로 돌린 값과 비트 단위로 같다."""
        detail = self.detail
        settings = detail.detail_settings({
            "guid_tsr_enabled": True, "guid_tsr_k": 0.9, "guid_mg_enabled": True,
            "guid_higs_enabled": True, "guid_higs_weight": 1.0})
        run, by_hand = detail.DetailRun(settings), detail.HistoryState()
        model = type("M", (), {"model_sampling": type("CONST", (), {})()})()
        schedule = [1.0, 0.8, 0.6, 0.0]
        generator = torch.Generator().manual_seed(3)
        for sigma in (1.0, 0.8, 0.6):
            x = torch.randn(1, 4, 1, 8, 8, generator=generator)
            d = torch.randn(1, 4, 1, 8, 8, generator=generator)
            args = {**self._args(sigma, schedule), "input": x, "model": model}
            got = detail.apply_detail_stages(run, args, d)
            sampled = detail.sampler_sigma(args)   # float32 텐서에서 읽은 값 — 0.8 이 아니라 0.800000012
            role = detail.step_role(by_hand, sampled, True)
            level = detail.noise_level(sampled, True)
            want = detail.apply_mg(d, x, sampled, by_hand, alpha=settings.mg_alpha, beta=settings.mg_beta,
                                   normalize=settings.mg_normalize,
                                   active=settings.mg_min <= level <= settings.mg_max, role=role)
            want = detail.apply_higs(want, sampled, by_hand, weight=settings.higs_weight, eta=settings.higs_eta,
                                     alpha=settings.higs_alpha, cutoff=settings.higs_cutoff,
                                     t_min=settings.higs_t_min, t_max=settings.higs_t_max, flow=True, role=role)
            want = detail.apply_tsr(want, x, args["sigma"], k=0.9, tsr_sigma=settings.tsr_sigma, flow=True)
            torch.testing.assert_close(got, want, rtol=0, atol=0)
        self.assertEqual(run.tsr_steps, 2)   # flow σ 1.0(순수 노이즈)은 TSR 이 건드리지 않는다 — 확장과 같다
        self.assertEqual(run.history.counters, by_hand.counters)

    def test_optimal_scale_skip_rules(self):
        callback_counts = {"applied": 0, "skipped": 0, "last_skip": ""}
        callback = self.optimal.make_callback(0.25, 0.0, 1.0, callback_counts)

        class Sampling:   # CONST(flow) — percent_to_sigma 는 Comfy 의 discrete flow 와 같은 끝값
            def percent_to_sigma(self, percent):
                return 1.0 if percent <= 0.0 else 0.0 if percent >= 1.0 else 1.0 - percent

        Sampling.__name__ = "CONST"
        model = type("Anima", (), {"model_sampling": Sampling()})()
        generator = torch.Generator().manual_seed(5)
        x, cond, uncond = (torch.randn(1, 4, 1, 8, 8, generator=generator) for _ in range(3))

        def args(**overrides):
            base = {"denoised": uncond + (cond - uncond) * 4.0, "cond_scale": 4.0, "input": x,
                    "cond_denoised": cond, "uncond_denoised": uncond, "uncond": [{}], "cond": [{}],
                    "model": model, "sigma": torch.tensor([0.5]), "model_options": {}}
            base.update(overrides)
            return base

        applied = callback(args())
        torch.testing.assert_close(applied, self.optimal.optimal_scale_residual(
            args()["denoised"], x, cond, uncond, 4.0, 0.25, torch.tensor([0.5])))
        for overrides, reason in (
            ({"model_options": {"sampler_cfg_function": object()}}, "custom CFG function"),
            ({"cond_scale": 1.0, "denoised": cond}, "CFG <= 1 or skipped negative step"),
            ({"denoised": uncond + (cond - uncond) * 4.0 + 0.1}, "prior post-CFG correction (APG/PAG/DCW/etc.)"),
            ({"uncond": None}, "negative not evaluated or identical predictions"),
        ):
            with self.subTest(reason=reason):
                call = args(**overrides)
                self.assertIs(callback(call), call["denoised"])
                self.assertEqual(callback_counts["last_skip"], reason)
        self.assertEqual(callback_counts["applied"], 1)

    def test_optimal_scale_node_patches_only_anima_and_a_valid_window(self):
        node = self.optimal.ForgeNeoAnimaOptimalScale()
        anima = _Patcher(type("Anima", (), {})())
        other = _Patcher(type("SDXL", (), {})())
        (patched,) = node.patch(anima, True, 0.25, 0.0, 1.0)
        self.assertEqual(len(patched.model_options["sampler_post_cfg_function"]), 1)
        for model, values in ((anima, (False, 0.25, 0.0, 1.0)), (anima, (True, 0.0, 0.0, 1.0)),
                              (anima, (True, 0.25, 0.6, 0.6)), (other, (True, 0.25, 0.0, 1.0))):
            with self.subTest(values=values, model=type(model.model).__name__):
                self.assertIs(node.patch(model, *values)[0], model)


# ── 팩 1.7.0: Euler (SMEA) Dy 보조 스텝(sam_extra_substep) 과 스텝 단위 상태 ─────────────────────────
# 보조 스텝은 한 스텝 안에서 다른 해상도(절반·×1.25)로 모델을 한 번 더 부른다. 확장(scripts/anima_safe_pag.py)은
# 그 평가가 스텝 단위 상태 — SMC e_prev, APG 모멘텀, RDC 이동 평균, ADG 스텝 수 — 를 읽기만 하고 갱신·초기화하지
# 않게 한다(keep_smc · _apply_apg · rdc_state=None if substep · _model_wrapper_inner/_post_cfg 의 ADG 초기화).
# 팩도 같은 표시(guidance_common.SUBSTEP_MARKER)로 같은 규칙을 지킨다(guidance_dcw · guidance).

SUBSTEP_FULL = (1, 4, 1, 8, 8)
SUBSTEP_HALF = (1, 4, 1, 4, 4)
SUBSTEP_SCALE = 4.0


def _copy_nested_options(options):
    """ComfyUI comfy/patcher_extension.py copy_nested_dicts — 샘플링 실행마다·clone 마다 model_options 복사."""
    copied = dict(options)
    for key, value in options.items():
        if isinstance(value, dict):
            copied[key] = _copy_nested_options(value)
        elif isinstance(value, list):
            copied[key] = list(value)
    return copied


class _NestedPatcher:
    """ModelPatcher 표면: clone 은 Comfy 처럼 model_options 를 중첩 dict 까지 복사한다(함수는 그대로)."""

    def __init__(self):
        self.model = None
        self.model_options = {"transformer_options": {}}
        self.wrappers = {}

    def clone(self):
        clone = _NestedPatcher()
        clone.model_options = _copy_nested_options(self.model_options)
        clone.wrappers = {kind: {key: list(items) for key, items in keyed.items()}
                          for kind, keyed in self.wrappers.items()}
        return clone

    def add_wrapper_with_key(self, wrapper_type, key, wrapper):
        self.wrappers.setdefault(wrapper_type, {}).setdefault(key, []).append(wrapper)


class _Const:
    """BaseModel.model_sampling 대역 — 클래스 이름 CONST(flow, guidance_detail.is_flow)."""


_Const.__name__ = "CONST"


def _comfy_cfg(options, x, sigma, cond_pred, uncond_pred, scale, model):
    """ComfyUI comfy/samplers.py cfg_function(v0.36 :592-605): sampler_cfg_function 의 인자 dict, 그 뒤 post-CFG 목록."""
    if "sampler_cfg_function" in options:
        args = {"cond": x - cond_pred, "uncond": x - uncond_pred, "cond_scale": scale, "timestep": sigma,
                "input": x, "sigma": sigma, "cond_denoised": cond_pred, "uncond_denoised": uncond_pred,
                "model": model, "model_options": options, "input_cond": None, "input_uncond": None}
        result = x - options["sampler_cfg_function"](args)
    else:
        result = uncond_pred + (cond_pred - uncond_pred) * scale
    for function in options.get("sampler_post_cfg_function", []):
        result = function({"denoised": result, "cond": None, "uncond": None, "cond_scale": scale, "model": model,
                           "uncond_denoised": uncond_pred, "cond_denoised": cond_pred, "sigma": sigma,
                           "model_options": options, "input": x})
    return result


def _toy_predictions(x, sigma):
    s = sigma.reshape([-1] + [1] * (x.ndim - 1)).to(x.dtype)
    cond = torch.tanh(x) * 0.8 / (1.0 + s) + 0.05 * s - 0.03 * x * x
    uncond = torch.tanh(0.7 * x) * 0.6 / (1.0 + s) - 0.02 * s
    return cond, uncond


def _marked_options(run_options, kind):
    """보조 스텝이 넘기는 model_options — 팩 샘플러의 extra_samplers_dy.with_substep_marker 그대로."""
    from comfy_custom_nodes.ai_studio_forge_parity.extra_samplers_dy import with_substep_marker
    return with_substep_marker({"model_options": run_options}, kind)["model_options"]


def _same_state(case, got, want, label):
    """중첩 dict/텐서 상태가 같다(텐서는 비트 단위)."""
    if isinstance(want, dict):
        case.assertIsInstance(got, dict, label)
        case.assertEqual(set(got), set(want), label)
        for key in want:
            _same_state(case, got[key], want[key], f"{label}.{key}")
    elif torch.is_tensor(want):
        case.assertTrue(torch.is_tensor(got) and torch.equal(got, want), label)
    else:
        case.assertEqual(got, want, label)


_CFG_STATE_CONFIGS = (
    ("SMC unit-L2", {"guid_smc_enabled": True, "guid_smc_preset": "Custom", "guid_smc_lambda": 6.0,
                     "guid_smc_k": 0.1}),
    ("SMC adaptive", {"guid_smc_enabled": True, "guid_smc_mode": "Adaptive sign",
                      "guid_smc_adaptive_alpha": 0.2, "guid_smc_adaptive_lambda": 5.0}),
    ("APG momentum", {"guid_apg_enabled": True, "guid_apg_eta": 0.5, "guid_apg_norm": 0.0,
                      "guid_apg_momentum": 0.6}),
    ("SMC + APG momentum", {"guid_smc_enabled": True, "guid_smc_preset": "Custom", "guid_apg_enabled": True,
                            "guid_apg_eta": 0.5, "guid_apg_norm": 0.0, "guid_apg_momentum": 0.6}),
    ("RDC", {"guid_dcw_enabled": True, "guid_dcw_lambda_low": 0.0, "guid_dcw_lambda_high": 0.0,
             "guid_rdc_tau": 0.5, "guid_rdc_alpha_ll": 0.3, "guid_rdc_alpha_hh": 0.2}),
    ("DCW + RDC", {"guid_dcw_enabled": True, "guid_dcw_lambda_low": 0.1, "guid_dcw_lambda_high": 0.02,
                   "guid_rdc_tau": 0.5, "guid_rdc_alpha_ll": 0.3, "guid_rdc_alpha_hh": 0.2}),
    ("all of them", {"guid_smc_enabled": True, "guid_smc_mode": "Adaptive sign", "guid_apg_enabled": True,
                     "guid_apg_eta": 0.5, "guid_apg_norm": 0.0, "guid_apg_momentum": 0.6,
                     "guid_dcw_enabled": True, "guid_dcw_lambda_low": 0.1, "guid_dcw_lambda_high": 0.02,
                     "guid_rdc_tau": 0.5, "guid_rdc_alpha_ll": 0.3, "guid_rdc_alpha_hh": 0.2}),
)


@requires_torch
class SubstepCfgStateTests(unittest.TestCase):
    """CFG 단계(SMC → APG)·DCW/RDC 의 실행 상태와 ADG: 표시된 보조 스텝은 읽기만 하고 남기지 않는다."""

    @classmethod
    def setUpClass(cls):
        bind_torch(globals())
        from comfy_custom_nodes.ai_studio_forge_parity import guidance, guidance_dcw
        cls.guidance, cls.dcw = guidance, guidance_dcw
        cls.state_keys = (guidance_dcw.SMC_STATE_KEY, guidance_dcw.APG_STATE_KEY, guidance_dcw.RDC_STATE_KEY)
        cls.model = type("Model", (), {"model_sampling": _Const()})()

    def _run_options(self, **settings):
        (patched,) = self.guidance.ForgeNeoAnimaGuidanceSuite().patch(
            _NestedPatcher(), None, None, None, True, json.dumps(settings))
        return _copy_nested_options(patched.model_options)

    def _call(self, options, sigma, seed, shape=SUBSTEP_FULL, substep=None):
        generator = torch.Generator().manual_seed(seed)
        cond, uncond, x = (torch.randn(shape, generator=generator) for _ in range(3))
        if substep:
            options = _marked_options(options, substep)
        return _comfy_cfg(options, x, torch.tensor([sigma]), cond, uncond, SUBSTEP_SCALE, self.model)

    def _state(self, options):
        return {key: dict(options.get(key) or {}) for key in self.state_keys}

    def _assert_untouched(self, before, after, label):
        for key, entries in before.items():
            self.assertEqual(set(after[key]), set(entries), f"{label}: {key}")
            for name, value in entries.items():
                self.assertIs(after[key][name], value, f"{label}: {key}.{name}")

    def test_a_marked_substep_leaves_no_trace(self):
        for name, settings in _CFG_STATE_CONFIGS:
            with self.subTest(name):
                plain_options = self._run_options(**settings)
                plain = [self._call(plain_options, sigma, seed) for sigma, seed in ((0.9, 1), (0.7, 2), (0.5, 3))]
                plain_state = self._state(plain_options)
                self.assertTrue(any(plain_state.values()), "the configuration keeps no state - nothing to test")

                options = self._run_options(**settings)
                mixed = [self._call(options, 0.9, 1)]
                before = self._state(options)
                self._call(options, 0.9, 11, SUBSTEP_HALF, "smea")
                self._assert_untouched(before, self._state(options), "smea sub-step")
                mixed.append(self._call(options, 0.7, 2))
                before = self._state(options)
                self._call(options, 0.7, 12, SUBSTEP_HALF, "dy")
                self._assert_untouched(before, self._state(options), "dy sub-step")
                mixed.append(self._call(options, 0.5, 3))

                for step, (a, b) in enumerate(zip(plain, mixed)):
                    self.assertTrue(torch.equal(a, b), f"step {step}")
                _same_state(self, self._state(options), plain_state, name)

    def test_an_unmarked_resolution_change_still_restarts_the_state(self):
        """표시가 막는 것: 표시 없는 다른 크기의 평가는 상태를 그 크기로 새로 시작한다(원본 DCW(+a) 규칙)."""
        dcw = self.dcw
        options = self._run_options(**dict(_CFG_STATE_CONFIGS)["all of them"])
        self._call(options, 0.9, 1)
        before = self._state(options)
        self._call(options, 0.9, 11, SUBSTEP_HALF)
        after = self._state(options)
        self.assertIsNot(after[dcw.SMC_STATE_KEY]["adaptive"], before[dcw.SMC_STATE_KEY]["adaptive"])
        self.assertEqual(tuple(after[dcw.SMC_STATE_KEY]["adaptive"]["e"].shape), SUBSTEP_HALF)
        self.assertEqual(tuple(after[dcw.APG_STATE_KEY]["average"].shape), SUBSTEP_HALF)
        self.assertEqual(tuple(after[dcw.RDC_STATE_KEY]["LL"].shape[-2:]), (2, 2))   # Haar 띠 = 절반의 절반

    def test_a_substep_reads_the_stored_state(self):
        """읽기는 한다: 같은 크기의 표시된 평가는 저장된 e_prev·모멘텀으로 계산한다(첫 스텝처럼 계산하지 않는다)."""
        settings = dict(_CFG_STATE_CONFIGS)["SMC + APG momentum"]
        options = self._run_options(**settings)
        self._call(options, 0.9, 1)
        with_history = self._call(options, 0.9, 5, SUBSTEP_FULL, "dy")
        fresh = self._call(self._run_options(**settings), 0.9, 5, SUBSTEP_FULL, "dy")
        self.assertFalse(torch.equal(with_history, fresh))

    def test_a_substep_rdc_call_is_dcw_only(self):
        """RDC 이동 평균은 스텝 해상도의 것 — 보조 스텝은 DCW 만(λ 0 이면 prediction 그대로)."""
        dcw = self.dcw
        for name in ("RDC", "DCW + RDC"):
            with self.subTest(name):
                settings = dict(_CFG_STATE_CONFIGS)[name]
                options = self._run_options(**settings)
                self._call(options, 0.9, 1)
                generator = torch.Generator().manual_seed(21)
                cond, uncond, x = (torch.randn(SUBSTEP_HALF, generator=generator) for _ in range(3))
                sigma = torch.tensor([0.9])
                got = _comfy_cfg(_marked_options(options, "dy"), x, sigma, cond, uncond, SUBSTEP_SCALE, self.model)
                plain = uncond + (cond - uncond) * SUBSTEP_SCALE
                want = dcw.apply_dcw(plain, x, sigma, settings["guid_dcw_lambda_low"],
                                     settings["guid_dcw_lambda_high"])
                self.assertTrue(torch.equal(got, want))

    def test_a_substep_logs_no_smc_restart(self):
        """보조 스텝은 아무것도 다시 시작하지 않으므로 'SMC state restarted' 를 남기지 않는다(표시 없는 크기 변경은 남긴다)."""
        options = self._run_options(**dict(_CFG_STATE_CONFIGS)["SMC unit-L2"])
        self._call(options, 0.9, 1)
        with self.assertNoLogs("ai_studio_forge_parity", level="INFO"):
            self._call(options, 0.9, 11, SUBSTEP_HALF, "dy")
        with self.assertLogs("ai_studio_forge_parity", level="INFO") as logged:
            self._call(options, 0.9, 12, SUBSTEP_HALF)
        self.assertTrue(any("SMC state restarted" in line for line in logged.output))

    def test_an_adaptive_guidance_substep_does_not_reset_apg_momentum(self):
        """확장 tests/test_extra_samplers_guidance.py 의 같은 이름 시나리오: cond-only(ADG) 표시 평가는 APG 모멘텀을
        비우지 않고, 표시 없는 cond-only 평가는 예전처럼 비운다."""
        guidance, dcw = self.guidance, self.dcw
        options = self._run_options(**dict(_CFG_STATE_CONFIGS)["APG momentum"],
                                    guid_adg_enabled=True, guid_adg_start=0.5, guid_adg_interval=0)
        options[guidance.ADG_STEP_KEY]["skipped"] = False
        self._call(options, 0.9, 1)
        momentum = options[dcw.APG_STATE_KEY]["average"]
        options[guidance.ADG_STEP_KEY]["skipped"] = True          # 이 평가는 ADG 가 cond 만 돌렸다
        self._call(options, 0.9, 11, SUBSTEP_HALF, "dy")
        self.assertIs(options[dcw.APG_STATE_KEY]["average"], momentum)
        self._call(options, 0.7, 2)
        self.assertEqual(options[dcw.APG_STATE_KEY], {})          # 표시 없는 cond-only 스텝은 비운다

    def test_a_substep_does_not_move_the_adaptive_guidance_step_count(self):
        """ADG 의 스텝 수는 샘플러 스텝을 따른다 — 보조 스텝은 자기 스텝으로 판단되고 수를 옮기지 않는다
        (확장: 보조 스텝은 Forge 스텝 수 그대로 평가되고 수를 되돌린다, euler_dy 항목 6). 간격 2: 0·2 스텝은 uncond 유지."""
        guidance = self.guidance
        options = self._run_options(guid_adg_enabled=True, guid_adg_start=0.0, guid_adg_interval=2)
        calculate = options["sampler_calc_cond_batch_function"]
        samplers = types.ModuleType("comfy.samplers")
        samplers.calc_cond_batch = lambda model, conds, x, sigma, run_options: [torch.zeros_like(x) for _ in conds]
        comfy = types.ModuleType("comfy")
        comfy.samplers = samplers
        decided = []
        with mock.patch.dict(sys.modules, {"comfy": comfy, "comfy.samplers": samplers}):
            for sigma, substep in ((0.9, None), (0.85, "dy"), (0.8, None), (0.7, None)):
                run = _marked_options(options, substep) if substep else options
                calculate({"conds": [{}, {}], "input": torch.zeros(SUBSTEP_FULL), "sigma": torch.tensor([sigma]),
                           "model": None, "model_options": run})
                decided.append(options[guidance.ADG_STEP_KEY]["skipped"])
        self.assertEqual(decided, [False, False, True, False])


@requires_extension
class SubstepRuleSourceTests(unittest.TestCase):
    """팩이 따르는 규칙이 설치된 확장에 그대로 있는지 — scripts/anima_safe_pag.py 의 함수마다(AST)."""

    @classmethod
    def setUpClass(cls):
        cls.source = source = (EXT_ROOT / "scripts" / "anima_safe_pag.py").read_text(encoding="utf-8")
        cls.functions = {node.name: ast.get_source_segment(source, node) for node in ast.walk(ast.parse(source))
                         if isinstance(node, ast.FunctionDef)}

    def _uses_the_rule(self, name):
        return "_is_extra_sampler_substep(" in self.functions[name]

    def test_the_cfg_state_rule(self):
        for name in ("_model_wrapper_inner", "_apply_apg", "_apply_cfg_base", "_post_cfg"):
            with self.subTest(name):
                self.assertTrue(self._uses_the_rule(name))
        self.assertIn("keep_smc = smc_on and not _is_extra_sampler_substep(args)", self.functions["_apply_cfg_base"])
        self.assertIn("rdc_state=None if substep", self.functions["_post_cfg"])
        self.assertIn("transformer_options.get(EXTRA_SAMPLER_SUBSTEP)", self.functions["_is_extra_sampler_substep"])
        from comfy_custom_nodes.ai_studio_forge_parity.guidance_common import SUBSTEP_MARKER
        self.assertIn(f'EXTRA_SAMPLER_SUBSTEP = "{SUBSTEP_MARKER}"', self.source)

    def test_the_detail_stage_rule(self):
        """HiFlow 기록·정렬과 MG/HiGS 는 보조 스텝을 건너뛰고, TSR 은 그대로 적용한다."""
        for name in ("_hiflow_record", "_apply_detail_stages"):
            with self.subTest(name):
                self.assertTrue(self._uses_the_rule(name))
        stages = self.functions["_apply_detail_stages"]
        self.assertIn("_hiflow_run_is_pass() and not substep", stages)
        self.assertIn('if (_HIST["mg_on"] or _HIST["higs_on"]) and not substep:', stages)
        tsr = stages[stages.index('if _TSR["on"]:'):]
        self.assertNotIn("substep", tsr)
        self.assertIn("or _is_extra_sampler_substep(args):", self.functions["_hiflow_record"])


_DY_E2E_SIGMAS = (1.0, 0.94, 0.86, 0.75, 0.6, 0.42, 0.24, 0.1, 0.0)
_DY_E2E_ENTRIES = ("euler_dy", "euler_smea_dy", "euler_dy_cfg_pp", "euler_smea_dy_cfg_pp")


class _DyCfgModel:
    """팩 Dy 샘플러가 부르는 모델: ComfyUI 의 CFG(_comfy_cfg)를 실행 model_options 로 돌리고 평가마다 남긴다.
    ``strip_marker`` 는 표시를 떼고 넘긴다(표시가 없으면 어떻게 되는지 보는 대조군)."""

    def __init__(self, predictor, model, strip_marker=False):
        self.inner_model = types.SimpleNamespace(predictor=predictor)
        self.model = model
        self.strip_marker = strip_marker
        self.evaluations = []
        self.run_options = None   # the run's own model_options (CFG++ entries sample a copy of the given dict)

    def __call__(self, x, sigma, model_options=None, seed=None):
        from comfy_custom_nodes.ai_studio_forge_parity.guidance_common import SUBSTEP_MARKER
        options = model_options or {}
        marker = (options.get("transformer_options") or {}).get(SUBSTEP_MARKER)
        if not marker:
            self.run_options = options
        if marker and self.strip_marker:
            options = dict(options)
            options["transformer_options"] = {key: value for key, value in options["transformer_options"].items()
                                              if key != SUBSTEP_MARKER}
        cond, uncond = _toy_predictions(x, sigma)
        out = _comfy_cfg(options, x, sigma, cond, uncond, SUBSTEP_SCALE, self.model)
        self.evaluations.append((marker, x.clone(), sigma.clone(), out.clone()))
        return out


@requires_torch
@requires_extension
class PlainEulerDySubstepTests(unittest.TestCase):
    """실제 팩 Euler (SMEA) Dy(보통 판 포함)로 샘플링: 보조 스텝이 끼어도 본 스텝들의 가이던스 결과·최종 상태가
    보조 스텝 없이 같은 평가만 다시 돌린 것과 비트 단위로 같다. 표시를 떼면 달라진다(대조군)."""

    SETTINGS = dict(_CFG_STATE_CONFIGS)["all of them"]

    @classmethod
    def setUpClass(cls):
        bind_torch(globals())
        from comfy_custom_nodes.ai_studio_forge_parity import extra_samplers_dy, guidance, guidance_dcw
        from tests._forge_k_diffusion import Predictor, ReplayNoise, forge_k_sampling
        cls.dy, cls.guidance, cls.dcw = extra_samplers_dy, guidance, guidance_dcw
        cls.kd, cls.predictor, cls.noise = forge_k_sampling(), Predictor(True), ReplayNoise
        cls.model = type("Model", (), {"model_sampling": _Const()})()

    def _suite(self, settings):
        (patched,) = self.guidance.ForgeNeoAnimaGuidanceSuite().patch(
            _NestedPatcher(), None, None, None, True, json.dumps(settings))
        return patched

    def _suite_options(self, settings, sigmas, tag=None, patched=None):
        patched = self._suite(settings) if patched is None else patched
        options = _copy_nested_options(patched.model_options)
        options["transformer_options"]["sample_sigmas"] = sigmas
        if tag:
            from comfy_custom_nodes.ai_studio_forge_parity.guidance_detail import PASS_KEY
            options["transformer_options"][PASS_KEY] = tag
        return options

    def _sample(self, entry, options, sigmas, *, strip_marker=False, shape=SUBSTEP_FULL):
        model = _DyCfgModel(self.predictor, self.model, strip_marker=strip_marker)
        x = torch.randn(shape, generator=torch.Generator().manual_seed(31)) * float(sigmas[0])
        sampler = getattr(self.dy, "sample_" + entry)
        sampler(model, x, sigmas, extra_args={"model_options": options, "seed": 0}, disable=True,
                noise_sampler=self.noise(x, 2468), kd=self.kd)
        return model.evaluations, model.run_options

    def _replay(self, evaluations, options):
        """같은 본 스텝 평가(표시 없는 것)만 새 스위트로 차례로 — (결과가 같은지, 그 실행 options)."""
        same = []
        for marker, x, sigma, out in evaluations:
            if marker:
                continue
            cond, uncond = _toy_predictions(x, sigma)
            same.append(torch.equal(_comfy_cfg(options, x, sigma, cond, uncond, SUBSTEP_SCALE, self.model), out))
        return same

    def _states(self, options):
        return {key: options.get(key) for key in (self.dcw.SMC_STATE_KEY, self.dcw.APG_STATE_KEY,
                                                  self.dcw.RDC_STATE_KEY)}

    def test_sub_steps_leave_the_steps_as_without_them(self):
        sigmas = torch.tensor(_DY_E2E_SIGMAS)
        expected_markers = {"euler_dy": [("dy", (4, 4)), ("dy", (4, 4))],
                            "euler_smea_dy": [("smea", (10, 10)), ("dy", (4, 4))]}
        for entry in _DY_E2E_ENTRIES:
            with self.subTest(entry):
                evaluations, run_options = self._sample(entry, self._suite_options(self.SETTINGS, sigmas), sigmas)
                markers = [(marker, tuple(x.shape[-2:])) for marker, x, _s, _o in evaluations if marker]
                self.assertEqual(markers, expected_markers[entry.replace("_cfg_pp", "")])
                replay = self._suite_options(self.SETTINGS, sigmas)
                same = self._replay(evaluations, replay)
                self.assertEqual(len(same), len(_DY_E2E_SIGMAS) - 1)
                self.assertTrue(all(same), same)
                self.assertTrue(all(value for value in self._states(replay).values()))
                _same_state(self, self._states(run_options), self._states(replay), entry)

    @staticmethod
    def _detail_run(patched):
        """스위트의 디테일 post-CFG 함수가 쥔 DetailRun(이력·HiFlow 궤적·정렬 상태)."""
        import inspect
        runs = {inspect.getclosurevars(function).nonlocals.get("run")
                for function in patched.model_options["sampler_post_cfg_function"]
                if function.__name__ in ("detail_stages", "hiflow_record")}
        (run,) = runs
        return run

    def _same_detail_run(self, got, want, label):
        _same_state(self, {"mg_m": got.history.mg_m, "higs_g": got.history.higs_g,
                           "last": got.history.last_sigma, "counters": dict(got.history.counters)},
                    {"mg_m": want.history.mg_m, "higs_g": want.history.higs_g,
                     "last": want.history.last_sigma, "counters": dict(want.history.counters)}, label + " history")
        self.assertEqual(got.trajectory.sigmas, want.trajectory.sigmas, label)
        for sigma in want.trajectory.sigmas:
            self.assertTrue(torch.equal(got.trajectory.at(sigma), want.trajectory.at(sigma)), f"{label} {sigma}")
        _same_state(self, {"prev_sigma": got.hiflow.prev_sigma, "prev_out": got.hiflow.prev_out,
                           "prev_ref": got.hiflow.prev_ref, "counters": dict(got.hiflow.counters)},
                    {"prev_sigma": want.hiflow.prev_sigma, "prev_out": want.hiflow.prev_out,
                     "prev_ref": want.hiflow.prev_ref, "counters": dict(want.hiflow.counters)}, label + " hiflow")
        self.assertEqual((got.hiflow_last_sigma, got.record_last_sigma),
                         (want.hiflow_last_sigma, want.record_last_sigma), label)

    def test_detail_stages_through_a_base_and_a_hires_pass(self):
        """MG·HiGS·TSR·HiFlow(기본 패스 기록 → Hires 패스 정렬)까지 켜고 두 패스를 Dy 로 샘플링: 본 스텝들의 결과,
        실행 상태, DetailRun(이력·궤적·정렬)이 보조 스텝 없이 다시 돌린 것과 비트 단위로 같다. TSR 만 보조 스텝에도
        걸린다(확장과 같이 tsr_steps 에 센다)."""
        from comfy_custom_nodes.ai_studio_forge_parity.guidance_detail import PASS_BASE, PASS_HIRES
        settings = dict(self.SETTINGS, guid_mg_enabled=True, guid_higs_enabled=True, guid_tsr_enabled=True,
                        guid_tsr_k=0.9, guid_hiflow_enabled=True)
        base_sigmas = torch.tensor(_DY_E2E_SIGMAS)
        hires_sigmas = torch.tensor((0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.0))
        for entry in _DY_E2E_ENTRIES:
            with self.subTest(entry):
                patched, replayed = self._suite(settings), self._suite(settings)
                passes = []
                for tag, sigmas, shape in ((PASS_BASE, base_sigmas, SUBSTEP_FULL),
                                           (PASS_HIRES, hires_sigmas, (1, 4, 1, 12, 12))):
                    evaluations, run_options = self._sample(
                        entry, self._suite_options(settings, sigmas, tag, patched=patched), sigmas, shape=shape)
                    replay = self._suite_options(settings, sigmas, tag, patched=replayed)
                    same = self._replay(evaluations, replay)
                    self.assertEqual(len(same), len(sigmas) - 1)
                    self.assertTrue(all(same), (tag, same))
                    _same_state(self, self._states(run_options), self._states(replay), f"{entry} {tag}")
                    passes.append(evaluations)
                run, reference = self._detail_run(patched), self._detail_run(replayed)
                self.assertEqual(len(reference.trajectory), len(_DY_E2E_SIGMAS) - 1)
                self.assertGreater(reference.hiflow.counters["direction"], 0)
                self.assertGreater(reference.history.counters["mg"], 0)
                self._same_detail_run(run, reference, entry)
                marked = [float(sigma.reshape(-1)[0]) for evaluations in passes
                          for marker, _x, sigma, _out in evaluations if marker]
                self.assertEqual(len(marked), 4)
                self.assertEqual(run.tsr_steps - reference.tsr_steps, sum(1 for sigma in marked if sigma < 1.0))

    def test_without_the_marker_the_sub_steps_would_change_the_steps(self):
        sigmas = torch.tensor(_DY_E2E_SIGMAS)
        for entry in ("euler_dy", "euler_smea_dy"):
            with self.subTest(entry):
                evaluations, _run = self._sample(entry, self._suite_options(self.SETTINGS, sigmas), sigmas,
                                                 strip_marker=True)
                same = self._replay(evaluations, self._suite_options(self.SETTINGS, sigmas))
                self.assertFalse(all(same), same)

@requires_torch
class SubstepDetailStageTests(unittest.TestCase):
    """디테일 단계(HiFlow·MG/HiGS·TSR): 표시된 보조 스텝은 HiFlow 기록·정렬과 MG/HiGS 이력을 건드리지 않고, 상태 없는
    TSR 만 받는다 — 확장 tests/test_extra_samplers_guidance.py SubstepDetailStageTests 의 시나리오."""

    @classmethod
    def setUpClass(cls):
        bind_torch(globals())
        from comfy_custom_nodes.ai_studio_forge_parity import guidance_detail
        from comfy_custom_nodes.ai_studio_forge_parity.guidance_common import SUBSTEP_MARKER
        cls.detail, cls.marker = guidance_detail, SUBSTEP_MARKER
        cls.model = type("Model", (), {"model_sampling": _Const()})()

    def _run(self, **settings):
        return self.detail.DetailRun(self.detail.detail_settings(settings))

    def _args(self, sigma, *, seed=0, shape=SUBSTEP_FULL, substep=None, schedule=None, tag=None):
        generator = torch.Generator().manual_seed(seed)
        cond, uncond, x = (torch.randn(shape, generator=generator) for _ in range(3))
        transformer = {}
        if schedule is not None:
            transformer["sample_sigmas"] = torch.tensor(schedule)
        if tag:
            transformer[self.detail.PASS_KEY] = tag
        if substep:
            transformer[self.marker] = substep
        return {"denoised": uncond + SUBSTEP_SCALE * (cond - uncond), "input": x, "sigma": torch.tensor([sigma]),
                "model": self.model, "model_options": {"transformer_options": transformer}}

    def _post(self, run, args, *, adg_skipped=False):
        return self.detail.apply_detail_stages(run, args, args["denoised"], adg_skipped=adg_skipped)

    def _base_pass(self, run, sigmas=(1.0, 0.75, 0.5, 0.25)):
        for index, sigma in enumerate(sigmas):
            args = self._args(sigma, seed=index, tag=self.detail.PASS_BASE)
            self.detail.record_hiflow(run, args, args["denoised"])

    def test_hiflow_base_trajectory_keeps_its_records_through_substeps(self):
        detail = self.detail
        run = self._run(guid_hiflow_enabled=True)
        for index, sigma in enumerate((1.0, 0.75, 0.5)):
            args = self._args(sigma, seed=index)
            detail.record_hiflow(run, args, args["denoised"])
            if sigma == 0.75:   # Euler Dy 의 2·3 스텝 / SMEA Dy 의 0·1 스텝: 스텝 안의 추가 평가
                marked = self._args(sigma, seed=10 + index, shape=SUBSTEP_HALF, substep="dy")
                detail.record_hiflow(run, marked, marked["denoised"])
        args = self._args(0.25, seed=3)
        detail.record_hiflow(run, args, args["denoised"])
        self.assertEqual(run.trajectory.sigmas, (0.25, 0.5, 0.75, 1.0))
        self.assertEqual(run.trajectory.shape, SUBSTEP_FULL)
        self.assertEqual(run.record_last_sigma, 0.25)

    def test_an_unmarked_resolution_change_still_restarts_the_trajectory(self):
        """표시가 막는 것: Trajectory.record 는 다른 모양의 기록에서 새로 시작한다."""
        run = self._run(guid_hiflow_enabled=True)
        self._base_pass(run, (1.0, 0.75, 0.5))
        args = self._args(0.5, seed=9, shape=SUBSTEP_HALF)
        self.detail.record_hiflow(run, args, args["denoised"])
        self.assertEqual(len(run.trajectory), 1)

    def test_hiflow_alignment_skips_substeps_in_the_hires_pass(self):
        detail = self.detail
        run = self._run(guid_hiflow_enabled=True)
        self._base_pass(run)
        schedule = [0.6, 0.4, 0.2, 0.0]
        args = self._args(0.6, seed=20, schedule=schedule, tag=detail.PASS_HIRES)
        self.assertFalse(torch.equal(self._post(run, args), args["denoised"]))
        state = run.hiflow
        before = (dict(state.counters), state.prev_sigma, state.prev_out, state.prev_ref, run.hiflow_last_sigma)
        for shape in (SUBSTEP_HALF, SUBSTEP_FULL):
            with self.subTest(shape=shape):
                marked = self._args(0.6, seed=21, shape=shape, substep="smea", schedule=schedule,
                                    tag=detail.PASS_HIRES)
                self.assertIs(self._post(run, marked), marked["denoised"])
                after = (dict(state.counters), state.prev_sigma, state.prev_out, state.prev_ref,
                         run.hiflow_last_sigma)
                self.assertEqual(after[:2], before[:2])
                self.assertIs(after[2], before[2])
                self.assertIs(after[3], before[3])
                self.assertEqual(after[4], before[4])

    def test_momentum_and_higs_history_ignore_substeps(self):
        detail = self.detail
        run = self._run(guid_mg_enabled=True, guid_higs_enabled=True)
        schedule = [1.0, 0.8, 0.6, 0.3, 0.2, 0.0]
        self._post(run, self._args(0.8, seed=1, schedule=schedule))
        self._post(run, self._args(0.6, seed=2, schedule=schedule))
        history = run.history
        before = (history.mg_m, history.higs_g, history.last_sigma, dict(history.counters))
        # 절반 크기(실제 보조 스텝), 같은 크기(표시 없으면 REPEAT 로 이력을 적용했을 것), ADG cond-only(표시 없으면 초기화)
        for shape, adg in ((SUBSTEP_HALF, False), (SUBSTEP_FULL, False), (SUBSTEP_FULL, True)):
            with self.subTest(shape=shape, adg_skipped=adg):
                marked = self._args(0.6, seed=3, shape=shape, substep="dy", schedule=schedule)
                self.assertIs(self._post(run, marked, adg_skipped=adg), marked["denoised"])
                self.assertIs(history.mg_m, before[0])
                self.assertIs(history.higs_g, before[1])
                self.assertEqual((history.last_sigma, dict(history.counters)), before[2:])
        self._post(run, self._args(0.3, seed=4, schedule=schedule))      # 다음 스텝은 이어 간다
        self.assertEqual(history.counters["mg"], before[3]["mg"] + 1)
        self.assertIsNot(history.mg_m, before[0])

    def test_tsr_still_rescales_a_substep(self):
        detail = self.detail
        run = self._run(guid_tsr_enabled=True, guid_mg_enabled=True)
        args = self._args(0.5, seed=5, shape=SUBSTEP_HALF, substep="dy")
        out = self._post(run, args)
        settings = run.settings
        expected = detail.apply_tsr(args["denoised"], args["input"], args["sigma"], k=settings.tsr_k,
                                    tsr_sigma=settings.tsr_sigma, flow=True)
        self.assertTrue(torch.equal(out, expected))
        self.assertFalse(torch.equal(out, args["denoised"]))
        self.assertEqual(run.tsr_steps, 1)
        self.assertIsNone(run.history.last_sigma)

if __name__ == "__main__":
    unittest.main()
