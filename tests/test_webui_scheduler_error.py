"""Forge HTTP 500 본문의 Extra Schedulers 오류가 사용자에게 닿는다(GPU 확인 F1, 2026-10-05) — 가짜 requests.post.

예전에는 앱 옵션을 넣지 않은 500 은 본문을 읽지 않아 'API 요청 실패: 500 Server Error …' 만 보였다(본문에는 확장의
'Extra Schedulers: the custom scheduler's sigma list gives sigma 1 at steps 0 and 1; …' 가 있었다). 이제 5xx 본문을 한 번
읽고 ``core/sam_extra_notices.explain_scheduler_error`` 가 한 줄을 주면 그 줄로 실패한다 — 다시 보내도 같으므로 재시도
없음(앱 옵션을 넣은 요청도). 그 밖의 500 은 지금과 같은 오류다. 메인 생성(txt2img·img2img)과 보조 패스(Refine 등) 모두.
"""
from __future__ import annotations

import unittest

import core.forge_override_settings as fos
from core import sam_extra_notices as sn
from tests.test_webui_forge_options import DEDUP, SAM3_OK, _ForgeCase, _Response, ok

# REPORT.md part forge-1 F1 — 테스트 Forge 7862 가 custom 시그마 목록 [1, 1, 0.5, 0] 에 돌려준 본문 그대로
MESSAGE = ("Extra Schedulers: the custom scheduler's sigma list gives sigma 1 at steps 0 and 1; a step that keeps sigma "
           "where it was is wasted with Euler and makes multistep samplers such as Res Multistep divide by zero (NaN, a "
           "black image)")
REPORT_BODY = {"error": "CustomSchedulerError", "detail": "", "body": "", "message": MESSAGE}
# 설치된 Forge 의 api_middleware.handle_exception 모양({"error", "detail", "body", "errors"})
ERRORS_BODY = {"error": "CustomSchedulerError", "detail": "", "body": "", "errors": MESSAGE}
LINE = sn.explain_scheduler_error(MESSAGE)


def scheduler_error(body=REPORT_BODY):
    return _Response(dict(body), 500)


class MainGenerationTests(_ForgeCase):
    def payload(self):
        return {"prompt": "1girl", "steps": 28, "scheduler": "custom", "alwayson_scripts": {}}

    def test_the_report_body_becomes_the_error_without_a_retry(self):
        response = scheduler_error()
        result = self.run_main(self.payload(), [response, ok()])
        self.assertFalse(result.success)
        self.assertEqual(result.error, LINE)
        self.assertTrue(result.error.startswith(sn.XSCHED_ERROR_LEAD + ": "), result.error)
        self.assertIn("gives sigma 1 at steps 0 and 1", result.error)
        self.assertNotIn("API 요청 실패", result.error)
        self.assertEqual(len(self.sent), 1)                       # 재시도 없음
        self.assertTrue(response.closed)                         # 본문을 읽고 닫았다
        self.refreshed.assert_not_called()

    def test_the_installed_forge_body_shape_and_img2img(self):
        for method, body in (("txt2img", ERRORS_BODY), ("img2img", REPORT_BODY)):
            with self.subTest(method=method):
                self.sent = []
                result = self.run_main(self.payload(), [scheduler_error(body), ok()], method=method)
                self.assertEqual((result.success, result.error, len(self.sent)), (False, LINE, 1))

    def test_with_app_options_the_scheduler_error_is_not_retried_either(self):
        fos.set_forge_option_overrides({DEDUP: False})
        result = self.run_main(self.payload(), [scheduler_error(), ok()])
        self.assertIn("override_settings", self.sent[0])
        self.assertEqual((result.success, result.error, len(self.sent)), (False, LINE, 1))
        self.refreshed.assert_not_called()

    def test_other_500_bodies_keep_the_old_error(self):
        for body in ({"error": "OutOfMemoryError", "detail": "", "body": "", "errors": "CUDA out of memory"},
                     {"error": "RuntimeError", "message": "Extra Samplers: something else"}, "Internal Server Error"):
            with self.subTest(body=body):
                self.sent = []
                result = self.run_main(self.payload(), [_Response(body, 500), ok()])
                self.assertFalse(result.success)
                self.assertTrue(result.error.startswith("API 요청 실패: 500"), result.error)
                self.assertEqual(len(self.sent), 1)


class AuxPassTests(_ForgeCase):
    SETTINGS = {"target": "face", "sam3_prompt": "face", "seed": 7}

    def test_aux_requests_fail_with_the_line(self):
        for method in ("refine", "sam3", "adetailer"):
            for options in ({}, {DEDUP: False}):
                with self.subTest(method=method, options=bool(options)):
                    fos.set_forge_option_overrides(options)
                    self.sent = []
                    with self.assertRaises(RuntimeError) as raised:
                        self.run_aux(method, dict(self.SETTINGS), [scheduler_error(), ok(SAM3_OK)],
                                     cancel_check=lambda: False)
                    self.assertEqual(str(raised.exception), LINE)
                    self.assertEqual(len(self.sent), 1)


if __name__ == "__main__":
    unittest.main()
