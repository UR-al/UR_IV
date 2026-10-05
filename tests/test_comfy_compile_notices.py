"""ComfyUI compile warnings of the sam-extra features reach the user (GC review round 2, c6-routing).

Since plan C6 the ComfyUI compiler leaves Anima SPEED out with a warning (Spectrum, inpaint/mask, ControlNet, one
scale, invalid settings) and notes Euler (SMEA) Dy (CFG++) running without its sub-steps under Spectrum.  Before the fix
those warnings were only logged: ``ComfyUIBackend`` turned compile warnings into notices for VAE DeGrid only, so the
SPEED card's "the reason is shown as a notice" promise did not hold on ComfyUI.  ``core/comfy_compile_notices`` maps
the warnings Forge also reports (through its status infotext) to the same notices; the backend attaches them to
``info['_sam_extra_notices']`` like Forge results.  No ComfyUI is contacted (``/object_info`` and the queue are stubbed).
"""
from __future__ import annotations

import unittest

from backends.base import GenerationResult
from core import anima_speed as asp
from core import comfy_compile_notices as ccn
from core import extra_samplers as xsa
from core import sam_extra_notices as sn
from core.comfy_workflow_compiler import ComfyWorkflowCompiler
from tests.test_comfy_degrid_backend import _Harness
from tests.test_comfy_object_info_cache import _png
from tests.test_comfy_sampler_aliases import _owner_caps
from tests.test_comfy_speed_compiler import block, caps as speed_caps, payload, speed_nodes_of


def _dy_caps() -> dict:
    result = _owner_caps(samplers=(*xsa.PACK_SAMPLERS, *xsa.CORE_SAMPLERS))
    result["DiTSpectrumPatch"] = {"input": {"required": {}}}
    return result


def _dy_payload(sampler: str = "Euler SMEA Dy CFG++", **extra) -> dict:
    return {"prompt": "p", "negative_prompt": "n", "seed": 1, "steps": 20, "sampler_name": sampler, "cfg_scale": 5.0,
            "alwayson_scripts": {xsa.SCRIPT_NAME: xsa.plan(xsa.APP_DEFAULTS, target="t2i", backend="comfyui").block},
            **extra}


def _warnings(body: dict, *, capabilities: dict, mode: str = "txt2img") -> list:
    found: list = []
    uploads = {} if mode == "txt2img" else {"uploaded_image": "in.png"}
    if mode == "inpaint":
        uploads["uploaded_mask"] = "mask.png"
    ComfyWorkflowCompiler(capabilities).compile(mode, "checkpoint.safetensors", body, warnings=found, **uploads)
    return found


def _shown(result) -> list:
    return [(n.code, n.level, n.feature, n.detail) for n in sn.notices_from_info(result.info)]


class CompileWarningNoticeTests(unittest.TestCase):
    """The pure mapping: the warnings the compiler really writes -> the Forge notices."""

    def test_every_static_speed_skip_becomes_the_pre_generation_skip_notice(self):
        cases = (({"spectrum_enabled": True}, asp.SKIP_COMFY_SPECTRUM),
                 ({"use_image_alpha_as_mask": True}, asp.SKIP_INPAINT),
                 ({"scripts": {"ControlNet": {"args": [{"enabled": True, "module": "canny"}]}}}, asp.SKIP_CONTROLNET))
        for extra, code in cases:
            with self.subTest(code=code):
                warnings = _warnings(payload(speed_block=block(), **extra), capabilities=speed_caps())
                self.assertEqual([w["code"] for w in warnings], [sn.CODE_SPEED_SKIPPED])
                [notice] = ccn.compile_warning_notices(warnings)
                self.assertEqual(notice, sn.speed_static_skip_notice(code))
                self.assertEqual((notice.code, notice.level, notice.feature, notice.detail),
                                 (sn.CODE_SPEED_SKIPPED, sn.LEVEL_INFO, "speed", code))
                self.assertIn(asp.SKIP_MESSAGES[code], notice.message)

    def test_a_single_scale_skip_has_the_key_of_the_contributors_notice(self):
        # the T2I contributor already says this before generation (ui/anima_speed_ui.contribute); the same key lets
        # NoticeThrottle show it once
        single = block(asp.SpeedSettings(enabled=True, scales="1.0"))
        [notice] = ccn.compile_warning_notices(_warnings(payload(speed_block=single), capabilities=speed_caps()))
        self.assertEqual(notice.key, sn.speed_static_skip_notice(asp.SKIP_NO_COARSE_STAGE).key)

    def test_invalid_speed_settings_are_a_warning_with_the_compilers_reason(self):
        warnings = _warnings(payload(speed_block={"args": [{"enabled": True, "scales": "0.5,0.9"}]}),
                             capabilities=speed_caps())
        [notice] = ccn.compile_warning_notices(warnings)
        self.assertEqual((notice.code, notice.level, notice.feature), (sn.CODE_SPEED_INVALID, sn.LEVEL_WARNING, "speed"))
        self.assertEqual(notice.detail, warnings[0]["reason"])
        self.assertTrue(notice.message.startswith(warnings[0]["reason"]))
        self.assertIn("SPEED 칸을 고치세요", notice.message)

    def test_dy_without_sub_steps_under_spectrum_is_the_forge_info_notice(self):
        warnings = _warnings(_dy_payload(spectrum_enabled=True), capabilities=_dy_caps())
        self.assertEqual([w["code"] for w in warnings],
                         [sn.CODE_EXTRA_SAMPLERS_SUBSTEPS_SKIPPED, "extra_samplers_cfg_pp"])
        [notice] = ccn.compile_warning_notices(warnings)          # CFG++ above 2: Forge only logs it too
        self.assertEqual((notice.code, notice.level, notice.feature, notice.detail),
                         (sn.CODE_EXTRA_SAMPLERS_SUBSTEPS_SKIPPED, sn.LEVEL_INFO, "extra_samplers",
                          xsa.DY_REASON_SPECTRUM))
        self.assertIn(ccn.COMFY_DY_SPECTRUM_TEXT, notice.message)

    def test_what_forge_does_not_report_stays_a_log_line(self):
        quiet = (
            {"code": "extra_samplers_cfg_pp", "reason": "CFG++ …", "feature": "extra_samplers", "cause": None},
            {"code": "colorcraft_no_entries", "reason": "Colorcraft …", "feature": "colorcraft", "cause": None},
            {"code": sn.CODE_DEGRID_COMFY_UNAVAILABLE, "reason": "x", "feature": "degrid", "cause": "node_missing"},
            {"code": sn.CODE_SPEED_SKIPPED, "reason": "x", "feature": "colorcraft", "cause": None},
            "not a dict", None,
        )
        self.assertEqual(ccn.compile_warning_notices(quiet), [])
        self.assertEqual(ccn.compile_warning_notices(None), [])

    def test_repeats_collapse_and_order_follows_the_warnings(self):
        skip = {"code": sn.CODE_SPEED_SKIPPED, "reason": asp.SKIP_MESSAGES[asp.SKIP_INPAINT], "feature": "speed"}
        dy = {"code": sn.CODE_EXTRA_SAMPLERS_SUBSTEPS_SKIPPED, "reason": "base", "feature": "extra_samplers"}
        dy_hires = {**dy, "reason": "hires"}
        notices = ccn.compile_warning_notices([dy, skip, dy_hires, skip])
        self.assertEqual([(n.code, n.detail) for n in notices],
                         [(sn.CODE_EXTRA_SAMPLERS_SUBSTEPS_SKIPPED, xsa.DY_REASON_SPECTRUM),
                          (sn.CODE_SPEED_SKIPPED, asp.SKIP_INPAINT)])


class BackendResultNoticeTests(_Harness, unittest.TestCase):
    """End to end through ``ComfyUIBackend.txt2img``/``img2img``: the notices land in the generation info."""

    def test_speed_left_out_under_comfy_spectrum_is_told(self):
        backend = self.backend(speed_caps())
        result = backend.txt2img("checkpoint.safetensors", payload(speed_block=block(), spectrum_enabled=True))
        self.assertTrue(result.success, result.error)
        self.assertEqual(speed_nodes_of(self.queued(backend)), [])
        self.assertEqual(_shown(result), [(sn.CODE_SPEED_SKIPPED, sn.LEVEL_INFO, "speed", asp.SKIP_COMFY_SPECTRUM)])
        self.assertEqual(result.info["seed"], 7)                                      # seed report kept

    def test_speed_on_an_inpaint_request_is_told(self):
        backend = self.backend(speed_caps())
        body = {**payload(speed_block=asp.plan(asp.SpeedSettings(enabled=True, apply_img2img=True), target="i2i",
                                               backend="comfyui").block),
                "init_images": [_png()], "mask": _png(), "denoising_strength": 0.5}
        result = backend.img2img("checkpoint.safetensors", body)
        self.assertTrue(result.success, result.error)
        self.assertEqual(speed_nodes_of(self.queued(backend)), [])
        self.assertEqual(_shown(result), [(sn.CODE_SPEED_SKIPPED, sn.LEVEL_INFO, "speed", asp.SKIP_INPAINT)])

    def test_invalid_speed_settings_are_told(self):
        backend = self.backend(speed_caps())
        result = backend.txt2img("checkpoint.safetensors",
                                 payload(speed_block={"args": [{"enabled": True, "mode": "zoom"}]}))
        self.assertTrue(result.success, result.error)
        self.assertEqual([code for code, *_rest in _shown(result)], [sn.CODE_SPEED_INVALID])

    def test_dy_under_spectrum_is_told_once(self):
        backend = self.backend(_dy_caps())
        result = backend.txt2img("checkpoint.safetensors", _dy_payload(spectrum_enabled=True))
        self.assertTrue(result.success, result.error)
        self.assertEqual(_shown(result), [(sn.CODE_EXTRA_SAMPLERS_SUBSTEPS_SKIPPED, sn.LEVEL_INFO, "extra_samplers",
                                           xsa.DY_REASON_SPECTRUM)])

    def test_nothing_to_tell_adds_nothing(self):
        for label, caps, body in (("speed runs", speed_caps(), payload(speed_block=block())),
                                  ("no speed", speed_caps(), payload()),
                                  ("dy without spectrum", _dy_caps(), _dy_payload(cfg_scale=1.5))):
            with self.subTest(case=label):
                backend = self.backend(caps)
                result = backend.txt2img("checkpoint.safetensors", body)
                self.assertTrue(result.success, result.error)
                self.assertNotIn(sn.INFO_KEY, result.info)

    def test_a_failed_generation_gets_no_notice(self):
        backend = self.backend(speed_caps())
        backend._queue_and_wait.side_effect = None
        backend._queue_and_wait.return_value = GenerationResult(success=False, error="boom")
        result = backend.txt2img("checkpoint.safetensors", payload(speed_block=block(), spectrum_enabled=True))
        self.assertEqual((result.success, result.error, result.info), (False, "boom", {}))


if __name__ == "__main__":
    unittest.main()
