"""ComfyUI VAE DeGrid results become the same notices as Forge's (core/comfy_degrid_report, A9).

Node reports (``/history`` outputs ``ui.ai_studio_degrid``) are turned into
Forge's infotext keys and read by the same rules (per image, grouped by error
kind); compile warnings (node omitted) become their own notices.  Pure — no
ComfyUI, no Qt.
"""
from __future__ import annotations

import unittest

from core import comfy_degrid_report as report
from core import sam_extra_notices as sn
from core import vae_degrid as vdg

UI = vdg.COMFY_UI_KEY


def sent(**values) -> dict:
    return {"prompt": "p", "alwayson_scripts": {vdg.SCRIPT_NAME: vdg.as_block(vdg.DegridSettings(enabled=True, **values))}}


def ok(model="qwenVAEDegridNafnet_v11", strength=1.0, tile=512, precision="fp32") -> dict:
    return {"status": "ok", "model": model, "mode": "Full", "strength": strength, "tile": tile,
            "precision": precision, "error": ""}


def skipped(error: str) -> dict:
    return {"status": "skipped", "model": "auto", "mode": "Full", "strength": 1.0, "tile": 512,
            "precision": "-", "error": error}


def outputs(*reports, node="12") -> dict:
    return {"9": {"images": [{"filename": "a.png", "type": "output"}]}, node: {UI: list(reports)}}


class ReportReadingTests(unittest.TestCase):
    def test_reports_are_read_in_node_order_and_bad_shapes_are_ignored(self):
        data = {"10": {UI: [ok(tile=256)]}, "9": {UI: [ok(tile=128), "junk"]}, "x": {UI: "nope"},
                "11": {"images": []}, "2": None}
        self.assertEqual([r["tile"] for r in report.reports_from_outputs(data)], [128, 256])
        for empty in (None, [], "x", {"1": {}}):
            self.assertEqual(report.reports_from_outputs(empty), [])

    def test_params_are_forge_infotext_keys(self):
        params = report.params_from_outputs(outputs(ok(strength=0.85, tile=384), skipped("model not found: auto"),
                                                    {"status": "off"}))
        self.assertEqual(params, [
            {vdg.KEY_MODEL: "qwenVAEDegridNafnet_v11", vdg.KEY_MODE: "Full", vdg.KEY_STRENGTH: "0.85",
             vdg.KEY_TILE: 384, vdg.KEY_PRECISION: "fp32"},
            {vdg.KEY_ERROR: "model not found: auto"},
        ])


class ResultNoticeTests(unittest.TestCase):
    def test_success_and_strength_zero_are_quiet(self):
        self.assertEqual(report.result_notices(sent(), outputs(ok(), ok())), [])
        zero = {**ok(strength=0.0), "status": "skipped", "precision": "-"}
        self.assertEqual(report.result_notices(sent(strength=0.0), outputs(zero)), [])

    def test_not_requested_is_quiet_even_with_a_user_node_in_the_workflow(self):
        self.assertEqual(report.result_notices({"prompt": "p"}, outputs(skipped("model not found: auto"))), [])

    def test_no_reports_say_nothing(self):
        self.assertEqual(report.result_notices(sent(), {"9": {"images": []}}), [])
        self.assertEqual(report.result_notices(sent(), None), [])

    def test_skips_are_grouped_by_kind_with_comfy_hints(self):
        notices = report.result_notices(sent(), outputs(
            skipped("not a DeGrid residual model: corr=0.97 mean=0.31"),
            ok(),
            skipped("not a DeGrid residual model: corr=0.95 mean=0.29"),
        ))
        self.assertEqual(len(notices), 1)
        notice = notices[0]
        self.assertEqual((notice.code, notice.feature, notice.detail),
                         (sn.CODE_DEGRID_ERROR, "degrid", vdg.ERROR_NOT_RESIDUAL))
        self.assertIn("2/3장이", notice.message)

    def test_model_not_found_and_oom_hints_speak_comfyui(self):
        notices = report.result_notices(sent(), outputs(skipped("model not found: auto")))
        self.assertIn("models/upscale_models", notices[0].hint)
        self.assertNotIn("Forge", notices[0].hint)
        oom = report.result_notices(sent(), outputs(skipped("OutOfMemoryError: CUDA out of memory")))
        self.assertIn("타일", oom[0].hint)
        self.assertNotIn("설정 › Forge", oom[0].hint)
        other = report.result_notices(sent(), outputs(skipped("ValueError: boom")))
        self.assertIn("ComfyUI 콘솔", other[0].hint)
        self.assertEqual(other[0].detail, "ValueError")

    def test_forge_hints_are_unchanged(self):
        notices = sn.degrid_result_notices(sent(), [{vdg.KEY_ERROR: "model not found: auto"}])
        self.assertIn("Forge 의 models/ESRGAN", notices[0].hint)
        self.assertEqual(sn.degrid_error_kind("OutOfMemoryError: x")[1], sn._DEGRID_OOM_HINT)


class CompileWarningTests(unittest.TestCase):
    def test_model_missing_warning_reads_like_the_forge_result(self):
        warnings = [{"code": sn.CODE_DEGRID_ERROR, "reason": "model not found: gone", "feature": "degrid"}]
        notices = report.result_notices(sent(model="gone"), {}, warnings)
        self.assertEqual(len(notices), 1)
        self.assertEqual((notices[0].code, notices[0].detail), (sn.CODE_DEGRID_ERROR, "model not found: gone"))
        self.assertIn("ComfyUI 의 models/upscale_models", notices[0].message)

    def test_unavailable_warning(self):
        reason = f"ComfyUI 에 {vdg.COMFY_NODE_CLASS} 노드가 없습니다"
        warnings = [{"code": sn.CODE_DEGRID_COMFY_UNAVAILABLE, "reason": reason, "feature": "degrid"},
                    {"code": sn.CODE_DEGRID_COMFY_UNAVAILABLE, "reason": reason, "feature": "degrid"},
                    {"code": "other", "reason": "x", "feature": "lora"}, "junk"]
        notices = report.result_notices(sent(), outputs(ok()), warnings)
        self.assertEqual([n.code for n in notices], [sn.CODE_DEGRID_COMFY_UNAVAILABLE])   # deduplicated
        self.assertIn(reason, notices[0].message)
        self.assertIn(sn.CODE_DEGRID_COMFY_UNAVAILABLE, sn.NOTICE_MIN_TTL_S)

    def test_merge_into_info_keeps_existing_notices_first(self):
        existing = sn.Notice("x", sn.LEVEL_INFO, "earlier").to_dict()
        new = report.result_notices(sent(), outputs(skipped("model not found: auto")))
        merged = report.merge_into_info({"seed": 3, sn.INFO_KEY: [existing]}, new)
        self.assertEqual(merged["seed"], 3)
        self.assertEqual([item["code"] for item in merged[sn.INFO_KEY]], ["x", sn.CODE_DEGRID_ERROR])
        self.assertEqual(report.merge_into_info({"a": 1}, []), {"a": 1})
        self.assertEqual([n.code for n in sn.notices_from_info(merged)], ["x", sn.CODE_DEGRID_ERROR])


    def test_compile_warnings_are_the_only_notices_even_when_a_user_node_reported(self):
        """컴파일 경고가 있으면 앱의 DeGrid 노드는 그래프에 없었다 — 사용자 워크플로에 원래 있던 같은 노드의 리포트를
        앱 DeGrid 결과로 읽으면 엉뚱한 알림이 하나 더 뜬다(넣을 자리가 없어 뺀 경우가 바로 그 모양이다)."""
        user_node = outputs(skipped("model not found: someone-elses"), skipped("output blew up: mean 2.0/255"))
        placement = [{"code": sn.CODE_DEGRID_COMFY_UNAVAILABLE, "feature": "degrid", "cause": vdg.COMFY_OMIT_PLACEMENT,
                      "reason": "사용자 워크플로의 선택 분기에 서로 다른 이미지 출력이 여러 개라 넣을 자리를 정할 수 없습니다"}]
        notices = report.result_notices(sent(), user_node, placement)
        self.assertEqual([(n.code, n.detail) for n in notices],
                         [(sn.CODE_DEGRID_COMFY_UNAVAILABLE, placement[0]["reason"])])
        missing = [{"code": sn.CODE_DEGRID_ERROR, "feature": "degrid", "cause": vdg.COMFY_OMIT_MODEL_MISSING,
                    "reason": "model not found: gone"}]
        notices = report.result_notices(sent(model="gone"), user_node, missing)
        self.assertEqual([(n.code, n.detail) for n in notices], [(sn.CODE_DEGRID_ERROR, "model not found: gone")])
        # 같은 리포트라도 경고가 없으면(앱 노드가 돌았다) 리포트를 읽는다 — 위의 배타성이 리포트 무시가 아님을 보인다
        self.assertEqual(len(report.result_notices(sent(), user_node)), 2)


class SchemaFixableCauseTests(unittest.TestCase):
    """백엔드는 새 /object_info 가 바꿀 수 있는 경고만 보고 스키마를 다시 받는다(comfyui_backend._compile_graph)."""

    def test_only_node_and_model_causes_are_fixable_and_each_reason_is_distinct(self):
        def warning(cause, reason, feature="degrid"):
            return {"code": sn.CODE_DEGRID_COMFY_UNAVAILABLE, "reason": reason, "feature": feature, "cause": cause}
        warnings = [
            warning(vdg.COMFY_OMIT_NODE_MISSING, "no node"),
            warning(vdg.COMFY_OMIT_NODE_CONTRACT, "contract  differs"),
            warning(vdg.COMFY_OMIT_MODEL_MISSING, "model not found: a"),
            warning(vdg.COMFY_OMIT_MODEL_MISSING, "model not found: b"),
            warning(vdg.COMFY_OMIT_PLACEMENT, "no output"),
            warning(vdg.COMFY_OMIT_NODE_MISSING, "lora thing", feature="lora"),
            {"code": sn.CODE_DEGRID_ERROR, "reason": "model not found: old shape", "feature": "degrid"},
            "junk", None,
        ]
        self.assertEqual(report.schema_fixable_causes(warnings), frozenset({
            (vdg.COMFY_OMIT_NODE_MISSING, "no node"), (vdg.COMFY_OMIT_NODE_CONTRACT, "contract differs"),
            (vdg.COMFY_OMIT_MODEL_MISSING, "model not found: a"), (vdg.COMFY_OMIT_MODEL_MISSING, "model not found: b"),
        }))
        for empty in (None, [], [warning(vdg.COMFY_OMIT_PLACEMENT, "no decode")]):
            self.assertEqual(report.schema_fixable_causes(empty), frozenset())
        self.assertEqual(vdg.COMFY_SCHEMA_FIXABLE, frozenset({
            vdg.COMFY_OMIT_NODE_MISSING, vdg.COMFY_OMIT_NODE_CONTRACT, vdg.COMFY_OMIT_MODEL_MISSING}))


class NoticeWordingTests(unittest.TestCase):
    """알림 문장은 "… DeGrid 없이 저장됐습니다 — {확인할 것}. (원인: …)" 한 줄표다. ComfyUI 에서 건너뛴 것은 앱 노드다."""

    REASONS = ("model not found: auto", "not a DeGrid residual model: corr=0.97", "output blew up: mean 183.0/255",
               "OutOfMemoryError: CUDA out of memory", "ValueError: boom")

    def test_every_hint_reads_as_one_dash_clause_on_both_backends(self):
        for comfy in (False, True):
            for reason in self.REASONS:
                with self.subTest(comfy=comfy, reason=reason):
                    [notice] = sn.degrid_result_notices(sent(), [{vdg.KEY_ERROR: reason}], comfy=comfy)
                    self.assertNotIn("—", notice.hint)
                    self.assertEqual(notice.message.count(" — "), 1, notice.message)
                    self.assertNotIn("원본을 그대로", notice.message)          # '저장됐습니다' 와 겹치는 말 없음

    def test_comfy_skips_are_the_node_not_the_extension(self):
        for reason in ("not a DeGrid residual model: corr=0.97", "output blew up: mean 183.0/255"):
            with self.subTest(reason=reason):
                [forge] = sn.degrid_result_notices(sent(), [{vdg.KEY_ERROR: reason}])
                [comfy] = report.result_notices(sent(), outputs(skipped(reason)))
                self.assertIn("확장이 적용하지 않았습니다", forge.hint)
                self.assertNotIn("확장", comfy.hint)
                self.assertIn("ComfyUI 노드가 이 단계를 건너뛰었습니다", comfy.hint)
                self.assertEqual(forge.detail, comfy.detail)                   # 묶음·억제 키는 같다


if __name__ == "__main__":
    unittest.main()
