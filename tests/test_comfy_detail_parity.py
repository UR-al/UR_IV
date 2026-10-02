"""ComfyUI 팩 1.6.0 디테일 가이던스 = sam-extra v0.30.0 — 같은 텐서, 같은 수 (CPU).

팩은 확장의 식을 ComfyUI post-CFG 훅용으로 다시 썼다(comfy_custom_nodes/ai_studio_forge_parity/guidance_detail.py ·
guidance_s2.py · guidance_dcw 의 Adaptive SMC · guidance_optimal_scale.py). 여기서는 설치된 확장의 sam3ext.guidance
모듈(Forge 없이 import 된다 — torch 만 쓴다)과 팩을 같은 입력으로 돌려 결과를 대조한다. 확장이 없으면 skip,
AISTUDIO_REQUIRE_FORGE_EXT=1 이면 실패(tests/_sam_extra_ext). 확장 의존이 없는 팩 동작(창·패스 표시·노드 규칙)은
아래 PackBehaviourTests 가 본다.

torch 는 지연 import 한다(tests/_optional_deps).
"""
import ast
import importlib
import json
import sys
import unittest

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


if __name__ == "__main__":
    unittest.main()
