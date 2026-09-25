"""ForgeNeoSkimmedCFG = 원본 Skimmed CFG 노드 그대로 (SKIM-C, parity_plan §6.3 C).

origin: Extraltodeus/Skimmed_CFG@d83005832ac42783adfd6f4ae96f6ef6406d1a74:skimmed_CFG.py
  :5-6 (MAX_SCALE·STEP_STEP), :83-134 (입력), :152-201 (패치 시점 σ 변환·pre_cfg_patch·등록),
  :204-281 (Timed flip / Clean Skim 프리셋).

팩은 원본 파일을 vendor/skimmed_cfg/ 에 고치지 않고 넣고(해시 고정) 그 ``execute`` 를 부른다.
그래서 아래 비교의 '원본'은 같은 벤더 파일이다 — 해시 고정 테스트가 그 파일이 원본 커밋의 바이트
(줄끝만 LF)임을 보증하고, 동등성 테스트는 팩 노드가 입력을 원본 인자에 제대로 옮기는지 본다.
게이트 기대값(29/30, 18, flip 9)은 읽기 전용 원본 클론을 스크래치에서 직접 돌려 얻은 수다
(scratchpad/origin_parity/skim_c/gate_counts.py). parity_plan §6.2 #2·#13 의 표와 같다.
"""
from __future__ import annotations

import contextlib
import hashlib
import importlib
import io
import json
import sys
import types
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from comfy_custom_nodes.ai_studio_forge_parity import guidance_skim
from tests._optional_deps import bind_torch, requires_torch

ROOT = Path(__file__).resolve().parents[1]
PACK_ROOT = ROOT / "comfy_custom_nodes" / "ai_studio_forge_parity"
VENDOR_ROOT = PACK_ROOT / "vendor" / "skimmed_cfg"
UPSTREAM_COMMIT = "d83005832ac42783adfd6f4ae96f6ef6406d1a74"
UPSTREAM_PACKAGE = f"{guidance_skim.__package__}.vendor.skimmed_cfg"
UPSTREAM_MODULE = f"{UPSTREAM_PACKAGE}.skimmed_CFG"

# 원본 blob(CRLF)을 LF 로 바꾼 바이트의 SHA-256 — vendor/skimmed_cfg/UPSTREAM.md 표와 같다.
EXPECTED_VENDOR_SHA256 = {
    "__init__.py": "263b752ecca74b1ae9587931ed7a02860dbd60559cd1bb3c8c458386282303c2",
    "skimmed_CFG.py": "a7471ebe04fd8925d8634d26e9488508a389d6dd4b67c88cd1794833c98aadde",
}
# 원본 LICENSE blob(LF) 그대로.
LICENSE_SHA256 = "cfc7749b96f63bd31c3c42b5c471bf756814053e847c10f3eb003417bc523d30"

SHIFT = 3.0   # Anima flow shift
STEPS = 30
SCALE = 5.0


def _sha256_lf(path: Path) -> str:
    return hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def _time_snr_shift(shift: float, t: float) -> float:
    # 공식: σ = s·t/(1+(s−1)·t) (parity_plan §8.2 '공통' 행, Comfy model_sampling.py:331-336).
    return shift * t / (1 + (shift - 1) * t)


class _FlowSampling:
    """Anima(flow, shift 3)의 ``percent_to_sigma`` — p ≤ 0 → 1.0, p ≥ 1 → 0.0, 그 사이 σ(1 − p)."""

    def percent_to_sigma(self, percent):
        if percent <= 0.0:
            return 1.0
        if percent >= 1.0:
            return 0.0
        return _time_snr_shift(SHIFT, 1.0 - percent)


class _Patcher:
    """ModelPatcher 대역: clone·get_model_object·set_model_sampler_*_function 만."""

    def __init__(self, sampling=None, lookups=None):
        self.sampling = sampling or _FlowSampling()
        self.model = SimpleNamespace(model_sampling=self.sampling)
        self.model_options: dict = {}
        self.lookups = [] if lookups is None else lookups

    def clone(self):
        clone = _Patcher(self.sampling, self.lookups)
        clone.model_options = {
            key: list(value) if isinstance(value, list) else value
            for key, value in self.model_options.items()
        }
        return clone

    def get_model_object(self, name):
        self.lookups.append(name)
        if name != "model_sampling":
            raise AttributeError(name)
        return self.sampling

    def _append(self, key, function, disable_cfg1_optimization):
        self.model_options[key] = list(self.model_options.get(key, [])) + [function]
        if disable_cfg1_optimization:
            self.model_options["disable_cfg1_optimization"] = True

    def set_model_sampler_pre_cfg_function(self, function, disable_cfg1_optimization=False):
        self._append("sampler_pre_cfg_function", function, disable_cfg1_optimization)

    def set_model_sampler_post_cfg_function(self, function, disable_cfg1_optimization=False):
        self._append("sampler_post_cfg_function", function, disable_cfg1_optimization)

    def set_model_sampler_cfg_function(self, function, disable_cfg1_optimization=False):
        self.model_options["sampler_cfg_function"] = function
        if disable_cfg1_optimization:
            self.model_options["disable_cfg1_optimization"] = True


def _comfy_api_stub() -> dict:
    """원본 모듈이 import 할 때 쓰는 이름만 있는 ``comfy_api.latest`` 대역.

    import 시점에는 io.ComfyNode(부모 클래스)·io.Schema·io.NodeOutput(반환 표기)·ComfyExtension 만
    평가되고, execute 는 io.NodeOutput(m) 만 부른다(진짜 NodeOutput 도 인자를 ``args`` 에 둔다).
    """

    class NodeOutput:
        def __init__(self, *args, **_kwargs):
            self.args = args

    latest = types.ModuleType("comfy_api.latest")
    latest.io = SimpleNamespace(
        ComfyNode=type("ComfyNode", (), {}), Schema=object, NodeOutput=NodeOutput,
    )
    latest.ComfyExtension = type("ComfyExtension", (), {})
    package = types.ModuleType("comfy_api")
    package.latest = latest
    return {"comfy_api": package, "comfy_api.latest": latest}


def _patch_pack(model, **settings):
    with contextlib.redirect_stdout(io.StringIO()):   # 원본은 flip σ 를 print 한다
        return guidance_skim.ForgeNeoSkimmedCFG().patch(model, True, **settings)[0]


class TestSkimmedVendoring(unittest.TestCase):
    def test_vendored_files_match_pinned_upstream(self):
        copied = {
            path.relative_to(VENDOR_ROOT).as_posix()
            for path in VENDOR_ROOT.rglob("*.py")
            if "__pycache__" not in path.parts
        }
        self.assertEqual(copied, set(EXPECTED_VENDOR_SHA256))
        self.assertEqual(
            {name: _sha256_lf(VENDOR_ROOT / name) for name in EXPECTED_VENDOR_SHA256},
            EXPECTED_VENDOR_SHA256,
        )
        manifest = (VENDOR_ROOT / "UPSTREAM.md").read_text(encoding="utf-8")
        self.assertIn(UPSTREAM_COMMIT, manifest)
        for digest in EXPECTED_VENDOR_SHA256.values():
            self.assertIn(digest, manifest)

    def test_apache_license_and_notice_ship_with_the_pack(self):
        self.assertEqual(
            _sha256_lf(PACK_ROOT / "LICENSES" / "Skimmed_CFG-Apache-2.0.txt"), LICENSE_SHA256,
        )
        notice = (PACK_ROOT / "THIRD_PARTY_NOTICES.md").read_text(encoding="utf-8")
        self.assertIn(UPSTREAM_COMMIT, notice)
        self.assertIn("LICENSES/Skimmed_CFG-Apache-2.0.txt", notice)

    def test_inputs_match_original_node_ranges(self):
        # origin: skimmed_CFG.py:5-6, :93-134 — skimming_cfg 7.0 [0, MAX_SCALE 10] step 1/STEP_STEP,
        # start/end/flip 0/1/0 [0, 1] step .01. min −1: 원본 툴팁과 프리셋(:238, :275)이 쓰는 '현재 CFG'.
        inputs = guidance_skim.ForgeNeoSkimmedCFG.INPUT_TYPES()["required"]
        self.assertEqual(tuple(inputs), (
            "model", "enabled", "skimming_cfg", "full_skim_negative", "disable_flipping_filter",
            "start_percent", "end_percent", "flip_percent",
        ))
        skim = inputs["skimming_cfg"][1]
        self.assertEqual(
            (skim["default"], skim["min"], skim["max"], skim["step"]), (7.0, -1.0, 10.0, 0.5),
        )
        for name, default in (("start_percent", 0.0), ("end_percent", 1.0), ("flip_percent", 0.0)):
            spec = inputs[name][1]
            self.assertEqual(
                (spec["default"], spec["min"], spec["max"], spec["step"]), (default, 0.0, 1.0, 0.01),
            )
        for name in ("enabled", "full_skim_negative", "disable_flipping_filter"):
            self.assertIs(inputs[name][1]["default"], False)

    def test_enabled_node_requires_a_comfy_model(self):
        with self.assertRaisesRegex(RuntimeError, "ComfyUI MODEL"):
            guidance_skim.ForgeNeoSkimmedCFG().patch(object(), True)


@requires_torch
class TestSkimmedOriginParity(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        bind_torch(globals())
        stubs = _comfy_api_stub()
        saved = {name: sys.modules.get(name) for name in stubs}
        sys.modules.update(stubs)

        def restore():
            for name in (UPSTREAM_MODULE, UPSTREAM_PACKAGE):
                sys.modules.pop(name, None)
            for name, module in saved.items():
                if module is None:
                    sys.modules.pop(name, None)
                else:
                    sys.modules[name] = module

        cls.addClassCleanup(restore)
        cls.upstream = importlib.import_module(UPSTREAM_MODULE)

    # ── helpers ───────────────────────────────────────────────────────────
    def _original(self, **arguments):
        """원본 노드(CFG_Skimming_Single_Scale_Pre_CFG.execute)를 새 대역 모델에 직접 부른 결과."""
        with contextlib.redirect_stdout(io.StringIO()):
            return self.upstream.CFG_Skimming_Single_Scale_Pre_CFG.execute(
                model=_Patcher(), **arguments,
            ).args[0]

    @staticmethod
    def _hook(model):
        (hook,) = model.model_options["sampler_pre_cfg_function"]
        return hook

    @staticmethod
    def _sigmas():
        # Comfy simple 스케줄(Anima flow 표 1000칸에서 등간격) — 모델 호출 σ 30개.
        table = torch.tensor(
            [_time_snr_shift(SHIFT, i / 1000.0) for i in range(1, 1001)], dtype=torch.float32,
        )
        stride = len(table) / STEPS
        return [float(table[-(1 + int(x * stride))]) for x in range(STEPS)]

    @staticmethod
    def _call(hook, x, cond, uncond, sigma, scale=SCALE):
        conds_out = [cond.clone(), uncond.clone()]   # 원본은 conds_out 텐서를 제자리에서 고친다
        sigma_t = torch.tensor([sigma])
        return hook({
            "conds": [None, None], "conds_out": conds_out, "cond_scale": scale,
            "timestep": sigma_t, "input": x, "sigma": sigma_t, "model": None, "model_options": {},
        })

    @staticmethod
    def _latents(seed):
        generator = torch.Generator().manual_seed(seed)
        x = torch.randn(1, 4, 8, 8, generator=generator)
        cond = torch.randn(1, 4, 8, 8, generator=generator) * 0.8
        uncond = cond + torch.randn(1, 4, 8, 8, generator=generator) * 0.3
        return x, cond, uncond

    @staticmethod
    def _comfy_step(options, x, sigma, cond_pred, uncond_pred, scale=SCALE):
        """ComfyUI sampling_function → cfg_function 의 훅 호출 순서
        (ComfyUI@387f98aa comfy/samplers.py:592-627): calc_cond_batch 결과 → sampler_pre_cfg_function
        (conds_out) → sampler_cfg_function(없으면 선형 CFG) → sampler_post_cfg_function."""
        sigma_t = torch.tensor([sigma])
        model = SimpleNamespace(model_sampling=_FlowSampling())
        out = [cond_pred.clone(), uncond_pred.clone()]
        for function in options.get("sampler_pre_cfg_function", []):
            out = function({
                "conds": [None, None], "conds_out": out, "cond_scale": scale, "timestep": sigma_t,
                "input": x, "sigma": sigma_t, "model": model, "model_options": options,
            })
        cond_out, uncond_out = out
        if "sampler_cfg_function" in options:
            result = x - options["sampler_cfg_function"]({
                "cond": x - cond_out, "uncond": x - uncond_out, "cond_scale": scale,
                "timestep": sigma_t, "input": x, "sigma": sigma_t, "cond_denoised": cond_out,
                "uncond_denoised": uncond_out, "model": model, "model_options": options,
                "input_cond": None, "input_uncond": None,
            })
        else:
            result = uncond_out + (cond_out - uncond_out) * scale
        for function in options.get("sampler_post_cfg_function", []):
            result = function({
                "denoised": result, "cond": None, "uncond": None, "cond_scale": scale,
                "model": model, "uncond_denoised": uncond_out, "cond_denoised": cond_out,
                "sigma": sigma_t, "model_options": options, "input": x,
            })
        return result

    # ── tests ─────────────────────────────────────────────────────────────
    def test_registers_original_pre_cfg_patch_on_a_clone_without_cfg1_flag(self):
        base = _Patcher()
        patched = _patch_pack(base)
        self.assertIsNot(patched, base)
        self.assertEqual(base.model_options, {})
        hook = self._hook(patched)
        # origin: skimmed_CFG.py:160-200 — 원본 모듈의 pre_cfg_patch 하나만, cfg1 플래그·post-CFG 없음.
        self.assertEqual((hook.__module__, hook.__name__), (UPSTREAM_MODULE, "pre_cfg_patch"))
        self.assertEqual(set(patched.model_options), {"sampler_pre_cfg_function"})
        # origin: skimmed_CFG.py:152-155 — model_sampling 은 노드가 돌 때 한 번만 찾는다.
        self.assertEqual(base.lookups, ["model_sampling"])
        x, cond, uncond = self._latents(0)
        self._call(hook, x, cond, uncond, self._sigmas()[12])
        self.assertEqual(base.lookups, ["model_sampling"])

    def test_pre_cfg_output_equals_original_node(self):
        sigmas = self._sigmas()
        probes = [sigmas[index] for index in (0, 1, 5, 9, 10, 15, 24, 25, 29)]
        windows = ((0.0, 1.0, 0.0), (0.2, 0.8, 0.0), (0.0, 1.0, 0.3), (0.8, 0.2, 0.0))
        seed = 0
        for skimming_cfg in (7.0, -1.0, 2.5, 0.0):
            for full_skim_negative in (False, True):
                for disable_flipping_filter in (False, True):
                    for start, end, flip in windows:
                        pack = self._hook(_patch_pack(
                            _Patcher(), skimming_cfg=skimming_cfg,
                            full_skim_negative=full_skim_negative,
                            disable_flipping_filter=disable_flipping_filter,
                            start_percent=start, end_percent=end, flip_percent=flip,
                        ))
                        original = self._hook(self._original(
                            skimming_cfg=skimming_cfg, full_skim_negative=full_skim_negative,
                            disable_flipping_filter=disable_flipping_filter,
                            start_at_percentage=start, end_at_percentage=end,
                            flip_at_percentage=flip,
                        ))
                        for sigma in probes:
                            seed += 1
                            x, cond, uncond = self._latents(seed)
                            with self.subTest(skimming_cfg=skimming_cfg, full=full_skim_negative,
                                              flip_off=disable_flipping_filter,
                                              window=(start, end, flip), sigma=sigma):
                                got = self._call(pack, x, cond, uncond, sigma)
                                want = self._call(original, x, cond, uncond, sigma)
                                self.assertTrue(torch.equal(got[0], want[0]))
                                self.assertTrue(torch.equal(got[1], want[1]))
                                if start > end:
                                    # 원본은 start > end 를 바꾸지 않는다 — 아무 스텝도 깎지 않는다.
                                    self.assertTrue(torch.equal(got[0], cond))
                                    self.assertTrue(torch.equal(got[1], uncond))

    def test_timed_flip_and_clean_skim_presets_equal_original_nodes(self):
        # origin: skimmed_CFG.py:233-244 (SkimFlipPreCFG), :268-281 (ConstantSkimPreCFG).
        cases = []
        for reverse in (False, True):
            with contextlib.redirect_stdout(io.StringIO()):
                original = self.upstream.SkimFlipPreCFG.execute(
                    model=_Patcher(), flip_at=0.3, reverse=reverse,
                ).args[0]
            cases.append((f"timed flip reverse={reverse}", original, dict(
                skimming_cfg=-1.0, full_skim_negative=True, disable_flipping_filter=reverse,
                start_percent=0.0, end_percent=1.0, flip_percent=0.3,
            )))
        original = self.upstream.ConstantSkimPreCFG.execute(model=_Patcher(), enabled=True).args[0]
        cases.append(("clean skim", original, dict(
            skimming_cfg=-1.0, full_skim_negative=True, disable_flipping_filter=False,
            start_percent=0.0, end_percent=1.0, flip_percent=0.0,
        )))
        for label, original, settings in cases:
            pack = self._hook(_patch_pack(_Patcher(), **settings))
            for index, sigma in enumerate(self._sigmas()):
                x, cond, uncond = self._latents(100 + index)
                with self.subTest(label, step=index):
                    got = self._call(pack, x, cond, uncond, sigma)
                    want = self._call(self._hook(original), x, cond, uncond, sigma)
                    self.assertTrue(torch.equal(got[0], want[0]))
                    self.assertTrue(torch.equal(got[1], want[1]))

    def test_window_and_flip_follow_sigma_like_the_original(self):
        # origin: skimmed_CFG.py:168-179 — sigma <= end_σ 또는 sigma >= start_σ 면 그대로(엄격 부등호),
        # flip 은 sigma > flip_σ 일 때. 기대값은 원본 클론을 돌린 gate_counts.py 결과.
        expectations = (
            ((0.0, 1.0, 0.0), list(range(1, 30)), []),        # 29/30 — Anima 첫 스텝(σ 1.0) 제외
            ((0.2, 0.8, 0.0), list(range(7, 25)), []),        # 18
            ((0.0, 1.0, 0.3), list(range(1, 30)), list(range(1, 10))),   # flip 9
        )
        for (start, end, flip), applied_steps, flipped_steps in expectations:
            printed = io.StringIO()
            with contextlib.redirect_stdout(printed):
                hook = self._hook(guidance_skim.ForgeNeoSkimmedCFG().patch(
                    _Patcher(), True, 7.0, False, False, start, end, flip,
                )[0])
            applied, flipped = [], []
            with mock.patch.object(
                self.upstream, "skimmed_CFG", wraps=self.upstream.skimmed_CFG,
            ) as spy:
                for index, sigma in enumerate(self._sigmas()):
                    before = spy.call_count
                    x, cond, uncond = self._latents(200 + index)
                    self._call(hook, x, cond, uncond, sigma)
                    if spy.call_count > before:
                        self.assertEqual(spy.call_count - before, 2)   # uncond 먼저, 그다음 cond
                        applied.append(index)
                        if spy.call_args_list[before].args[5]:
                            flipped.append(index)
            with self.subTest(window=(start, end, flip)):
                self.assertEqual(applied, applied_steps)
                self.assertEqual(flipped, flipped_steps)
                if flip:
                    self.assertIn("Flip at sigma: 0.87", printed.getvalue())   # 원본 :157-158

    def test_cfg_one_keeps_comfys_zero_uncond(self):
        # cfg1 플래그가 없으니 CFG 1 에서 Comfy 는 음성 패스를 건너뛰고 0 텐서를 넘긴다.
        # origin: skimmed_CFG.py:168-173 — uncond 가 전부 0 이면 그대로 돌려준다.
        patched = _patch_pack(_Patcher())
        self.assertNotIn("disable_cfg1_optimization", patched.model_options)
        x, cond, _ = self._latents(7)
        zeros = torch.zeros_like(cond)
        out = self._call(self._hook(patched), x, cond, zeros, self._sigmas()[12], scale=1.0)
        self.assertTrue(torch.equal(out[0], cond))
        self.assertTrue(torch.equal(out[1], zeros))

    def test_cfg_one_with_a_forced_uncond_divides_by_zero_like_the_original(self):
        # 다른 노드(팩의 SEG/SLG·ADG·CWM/SMC/APG, 서드파티)가 cfg1 플래그를 켜면 Comfy 는 CFG 1 에서도
        # 진짜 uncond 를 계산한다(ComfyUI@387f98aa comfy/samplers.py:610-613). 원본은 그때 양성을
        # cond_scale − 1 = 0 으로 깎으며 0 으로 나눈다 — 팩은 이 원본 동작을 그대로 둔다(Forge 확장의
        # CFG 1 가드는 Forge 전용 호스트 차이).
        # origin: skimmed_CFG.py:181-196 (uncond 는 cond_scale, cond 는 cond_scale − 1), :51-53 (÷ cond_scale).
        settings = dict(
            skimming_cfg=7.0, full_skim_negative=False, disable_flipping_filter=False,
            start_percent=0.0, end_percent=1.0, flip_percent=0.0,
        )
        patched = _patch_pack(_Patcher(), **settings)
        patched.model_options["disable_cfg1_optimization"] = True   # 다른 노드가 켠 상태
        original = self._hook(self._original(
            skimming_cfg=7.0, full_skim_negative=False, disable_flipping_filter=False,
            start_at_percentage=0.0, end_at_percentage=1.0, flip_at_percentage=0.0,
        ))
        sigma = self._sigmas()[12]
        x, cond, uncond = self._latents(21)
        got = self._call(self._hook(patched), x, cond, uncond, sigma, scale=1.0)
        want = self._call(original, x, cond, uncond, sigma, scale=1.0)
        for index in (0, 1):
            torch.testing.assert_close(got[index], want[index], rtol=0, atol=0, equal_nan=True)
        # uncond 단계는 cond_scale 1 로 나눠 유한하다.
        self.assertTrue(bool(torch.isfinite(got[1]).all()))
        # cond 단계: 이미 깎인 uncond 에 대한 원본 마스크(cond_scale − 1 = 0) 안은 전부 inf/NaN, 밖은 그대로.
        mask = self.upstream.get_skimming_mask(x, cond, got[1], 0.0)
        self.assertTrue(bool(mask.any()))
        self.assertFalse(bool(torch.isfinite(got[0][mask]).any()))
        self.assertTrue(torch.equal(got[0][~mask], cond[~mask]))
        # 선형 CFG(cfg 함수 없음)는 그 값을 denoised 까지 옮긴다.
        denoised = self._comfy_step(patched.model_options, x, sigma, cond, uncond, scale=1.0)
        self.assertFalse(bool(torch.isfinite(denoised).all()))
        # CFG 가 1 이 아니면 같은 입력에서 유한하다.
        for scale in (1.5, SCALE):
            with self.subTest(scale=scale):
                out = self._call(self._hook(patched), x, cond, uncond, sigma, scale=scale)
                self.assertTrue(bool(torch.isfinite(out[0]).all() and torch.isfinite(out[1]).all()))

    def test_suite_cwm_cfg_function_sees_skimmed_predictions(self):
        # 옛 팩은 post-CFG 맨 앞에서 선형 CFG 를 다시 계산해 CWM 결과를 버렸다(§6.2 #12).
        from comfy_custom_nodes.ai_studio_forge_parity import guidance

        settings = json.dumps({
            "guid_cwm_enabled": True, "guid_cwm_alpha_low": 0.3, "guid_cwm_alpha_high": 0.15,
        })

        def suite(model):
            return guidance.ForgeNeoAnimaGuidanceSuite().patch(
                model, object(), object(), object(), True, settings,
            )[0]

        sigma = self._sigmas()[12]
        x, cond, uncond = self._latents(11)
        skimmed = self._call(self._hook(self._original(
            skimming_cfg=7.0, full_skim_negative=False, disable_flipping_filter=False,
            start_at_percentage=0.0, end_at_percentage=1.0, flip_at_percentage=0.0,
        )), x, cond, uncond, sigma)
        # 기대값: 원본 skim 이 낸 예측을 스위트(CWM)만 있는 모델에 넣은 결과.
        expected = self._comfy_step(suite(_Patcher()).model_options, x, sigma, *skimmed)
        unskimmed = self._comfy_step(suite(_Patcher()).model_options, x, sigma, cond, uncond)
        linear = skimmed[1] + (skimmed[0] - skimmed[1]) * SCALE
        for order in ("suite→skim", "skim→suite"):
            model = (
                _patch_pack(suite(_Patcher())) if order == "suite→skim"
                else suite(_patch_pack(_Patcher()))
            )
            with self.subTest(order=order):
                got = self._comfy_step(model.model_options, x, sigma, cond, uncond)
                self.assertFalse(torch.allclose(got, linear))       # CWM 결과를 버리지 않는다
                self.assertFalse(torch.allclose(got, unskimmed))    # CWM 이 깎인 예측을 받는다
                self.assertTrue(torch.equal(got, expected))
                self.assertEqual(len(model.model_options["sampler_pre_cfg_function"]), 1)


if __name__ == "__main__":
    unittest.main()
