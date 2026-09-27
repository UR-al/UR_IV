"""앱이 덧붙인 부분을 Forge 가 거절했을 때 다시 보낼 계획 — core/forge_optional_parts (P7 422 가지, P10 500 가지)."""
import copy
import json
import unittest

from core import sam_extra_notices as sn
from core.forge_optional_parts import (
    MAX_ATTEMPTS_CAP, REASON_FROZEN, REASON_KEY_ERROR, OptionalParts, max_attempts, plan_retry, rejected_option_reason,
)

PAG, DD = "Anima Perturbation Guidance", "Anima Detail Daemon"


def payload():
    return {"prompt": "p", "init_images": ["b64"],
            "alwayson_scripts": {"SAM3 Mask": {"args": [{"sam3_prompt": "face"}]},
                                 PAG: {"args": [True]}, DD: {"args": [True]}}}


class PlanRetryTests(unittest.TestCase):
    parts = OptionalParts(titles=(PAG, DD))

    def test_both_422_body_shapes_remove_only_the_named_title(self):
        bodies = ({"error": "HTTPException", "detail": f"always on script {DD} not found"},
                  {"detail": f"Script '{DD}' not found"},
                  json.dumps({"detail": f"Script '{DD.lower()}' not found"}),
                  json.dumps({"detail": f"always on script {DD} not found"}).encode("utf-8"))
        for body in bodies:
            with self.subTest(body=body):
                original = payload()
                before = copy.deepcopy(original)
                plan = plan_retry(422, body, original, self.parts)
                self.assertIsNotNone(plan)
                self.assertEqual(plan.code, sn.CODE_PROPAGATION_RETRIED)
                self.assertEqual(plan.removed, (DD,))
                self.assertEqual(list(plan.payload["alwayson_scripts"]), ["SAM3 Mask", PAG])   # 하나만(A3)
                self.assertEqual(plan.remaining, OptionalParts(titles=(PAG,)))
                self.assertEqual(original, before)                                           # 입력 불변

    def test_not_ours_or_not_422_is_none(self):
        cases = (
            (422, {"detail": "Script 'SAM3 Mask' not found"}),        # 보조 경로 자신의 블록 — 지금처럼 실패
            (422, {"detail": "always on script ADetailer not found"}),
            (422, {"detail": "Script 'NegPiP' not found"}),           # 전달하지 않은 제목
            (422, {"detail": [{"loc": ["body", "steps"], "msg": "bad"}]}),
            (500, {"detail": f"Script '{PAG}' not found"}),
            (None, {"detail": f"Script '{PAG}' not found"}),
            ("x", {}),
        )
        for status, body in cases:
            with self.subTest(status=status, body=body):
                self.assertIsNone(plan_retry(status, body, payload(), self.parts))
        self.assertIsNone(plan_retry(422, {"detail": f"Script '{PAG}' not found"}, payload(), OptionalParts()))
        self.assertIsNone(plan_retry(422, {"detail": f"Script '{PAG}' not found"}, payload(), None))

    def test_parts_shrink_until_exhausted(self):
        plan = plan_retry(422, {"detail": f"Script '{PAG}' not found"}, payload(), self.parts)
        plan2 = plan_retry(422, {"detail": f"Script '{DD}' not found"}, plan.payload, plan.remaining)
        self.assertEqual(plan2.remaining, OptionalParts())
        self.assertEqual(list(plan2.payload["alwayson_scripts"]), ["SAM3 Mask"])
        self.assertIsNone(plan_retry(422, {"detail": f"Script '{DD}' not found"}, plan2.payload, plan2.remaining))

    def test_max_attempts(self):
        self.assertEqual(max_attempts(OptionalParts()), 1)
        self.assertEqual(max_attempts(None), 1)
        self.assertEqual(max_attempts(OptionalParts(titles=(PAG, DD))), 3)
        self.assertEqual(max_attempts(OptionalParts(titles=(PAG,), option_keys=("a", "b"))), 3)   # 옵션은 한 종류
        self.assertEqual(max_attempts(OptionalParts(titles=tuple(str(i) for i in range(20)))), MAX_ATTEMPTS_CAP)
        self.assertFalse(OptionalParts())
        self.assertTrue(OptionalParts(option_keys=("k",)))

    def test_missing_script_title(self):
        self.assertEqual(sn.missing_script_title({"detail": "Script 'X Y' not found"}), "X Y")
        self.assertEqual(sn.missing_script_title('{"detail": "always on script X not found"}'), "X")
        self.assertIsNone(sn.missing_script_title({"detail": "something else"}))
        self.assertIsNone(sn.missing_script_title({"detail": [{"msg": "x"}]}))
        self.assertIsNone(sn.missing_script_title(None))


# ── P10: 앱이 넣은 Forge 옵션(override_settings)의 500 ─────────────────────────────
DEDUP, DAVE = "sam3_guidance_pag_prefix_dedup", "sam3_guidance_dave_pre_dd_sigma"


def forge_500(error, errors):
    """modules/api/api.py handle_exception 의 본문 모양."""
    return {"error": error, "detail": "", "body": "", "errors": errors}


def option_payload(caller=None):
    overrides = dict(caller or {})
    overrides.update({DEDUP: False, DAVE: False})
    return {"prompt": "p", "override_settings": overrides}


class OptionRetryTests(unittest.TestCase):
    parts = OptionalParts(option_keys=(DEDUP, DAVE))

    def test_key_error_on_an_app_key_removes_every_app_key_once(self):
        bodies = (forge_500("KeyError", f"'{DAVE}'"), json.dumps(forge_500("KeyError", f"'{DAVE}'")),
                  json.dumps(forge_500("KeyError", f'"{DAVE}"')).encode("utf-8"))
        for body in bodies:
            with self.subTest(body=body):
                original = option_payload()
                before = copy.deepcopy(original)
                plan = plan_retry(500, body, original, self.parts)
                self.assertIsNotNone(plan)
                self.assertEqual((plan.code, plan.reason), (sn.CODE_FORGE_OPTION_REJECTED, REASON_KEY_ERROR))
                self.assertEqual(plan.removed, (DEDUP, DAVE))
                self.assertEqual(plan.payload, {"prompt": "p"})      # 앱이 만든 override_settings 는 통째로 없앤다
                self.assertEqual(plan.remaining, OptionalParts())
                self.assertEqual(original, before)

    def test_caller_keys_survive_the_retry(self):
        plan = plan_retry(500, forge_500("KeyError", f"'{DEDUP}'"), option_payload({"sd_vae": "v"}), self.parts)
        self.assertEqual(plan.payload["override_settings"], {"sd_vae": "v"})

    def test_frozen_settings(self):
        for text in ("changing settings is disabled",
                     f"not possible to set '{DEDUP}' because this setting is frozen with --freeze-specific-settings",
                     f"not possible to set '{DAVE}' because settings in section 'SAM Extra Guidance' "
                     "(sam3_guidance) are frozen with --freeze-settings-in-sections"):
            with self.subTest(text=text):
                plan = plan_retry(500, forge_500("AssertionError", text), option_payload(), self.parts)
                self.assertEqual((plan.code, plan.reason), (sn.CODE_FORGE_OPTION_REJECTED, REASON_FROZEN))
        # 호출자가 보낸 키가 잠겼으면 앱 키를 빼도 소용없다 — 지금처럼 실패
        caller_frozen = forge_500("AssertionError", "not possible to set 'sd_vae' because this setting is frozen "
                                                    "with --freeze-specific-settings")
        self.assertIsNone(plan_retry(500, caller_frozen, option_payload({"sd_vae": "v"}), self.parts))

    def test_a_partial_freeze_names_only_the_key_forge_named(self):
        """(P10 검토 4) --freeze-settings-in-sections·--freeze-specific-settings 는 앱 옵션 일부만 잠근다(확장 옵션은 섹션이
        여럿이다). 재시도는 여전히 앱 키를 한 번에 모두 빼지만(``removed``), 잠겼다고 적는 것(``frozen`` — 기억할 키)은
        Forge 가 이름을 댄 키뿐이다. 이름 없는 전역 잠금('changing settings is disabled')만 앱 키 전부다."""
        cases = (
            ("changing settings is disabled", (DEDUP, DAVE)),
            (f"not possible to set '{DEDUP}' because this setting is frozen with --freeze-specific-settings", (DEDUP,)),
            (f"not possible to set '{DAVE}' because settings in section 'SAM Extra Guidance' (sam3_guidance) are "
             "frozen with --freeze-settings-in-sections", (DAVE,)),
        )
        for text, locked in cases:
            with self.subTest(text=text):
                plan = plan_retry(500, forge_500("AssertionError", text), option_payload(), self.parts)
                self.assertEqual(plan.removed, (DEDUP, DAVE))            # 재시도 한 번 — 앱 키를 모두 뺀다
                self.assertEqual(plan.payload, {"prompt": "p"})
                self.assertEqual(plan.frozen, locked)
        key_error = plan_retry(500, forge_500("KeyError", f"'{DAVE}'"), option_payload(), self.parts)
        self.assertEqual(key_error.frozen, ())                           # KeyError 는 잠금이 아니다
        # 'frozen' 이 들어 있어도 Forge 의 잠금 문구(이름을 댄 키·전역 문구)가 아니면 앱 옵션 탓이 아니다 — 지금처럼 실패
        for text in ("parameter is frozen", "frozen module state mismatch"):
            with self.subTest(text=text):
                self.assertIsNone(plan_retry(500, forge_500("AssertionError", text), option_payload(), self.parts))
                self.assertIsNone(rejected_option_reason(forge_500("AssertionError", text), self.parts.option_keys))

    def test_other_500s_are_not_ours(self):
        cases = (
            forge_500("KeyError", "'sam3_something_else'"),        # 앱이 넣지 않은 키
            forge_500("KeyError", "'model'"),
            forge_500("RuntimeError", "CUDA out of memory"),
            forge_500("AssertionError", "shape mismatch"),
            "Internal Server Error",
            None,
        )
        for body in cases:
            with self.subTest(body=body):
                self.assertIsNone(plan_retry(500, body, option_payload(), self.parts))
                self.assertIsNone(rejected_option_reason(body, self.parts.option_keys))
        self.assertIsNone(plan_retry(500, forge_500("KeyError", f"'{DEDUP}'"), option_payload(), OptionalParts()))
        self.assertIsNone(plan_retry(500, forge_500("KeyError", f"'{DEDUP}'"), option_payload(),
                                     OptionalParts(titles=(PAG,))))
        self.assertIsNone(plan_retry(422, {"detail": f"Script '{PAG}' not found"}, payload(), self.parts))

    def test_422_title_then_500_option_shrinks_to_nothing(self):
        parts = OptionalParts(titles=(PAG,), option_keys=(DEDUP,))
        self.assertEqual(max_attempts(parts), 3)
        start = {**payload(), "override_settings": {DEDUP: False}}
        first = plan_retry(422, {"detail": f"Script '{PAG}' not found"}, start, parts)
        self.assertEqual(first.remaining, OptionalParts(option_keys=(DEDUP,)))
        self.assertIn("override_settings", first.payload)
        second = plan_retry(500, forge_500("KeyError", f"'{DEDUP}'"), first.payload, first.remaining)
        self.assertEqual(second.remaining, OptionalParts())
        self.assertNotIn("override_settings", second.payload)
        self.assertNotIn(PAG, second.payload["alwayson_scripts"])
        self.assertIsNone(plan_retry(500, forge_500("KeyError", f"'{DEDUP}'"), second.payload, second.remaining))


if __name__ == "__main__":
    unittest.main()
