"""sam-extra Forge 옵션 요청별 덮어쓰기 — core/forge_override_settings (P10, 사용자 결정 D3, critic B11·B14·C)."""
from __future__ import annotations

import copy
import json
import os
import re
import tempfile
import unittest
from pathlib import Path
from types import MappingProxyType, SimpleNamespace
from unittest import mock

import core.forge_override_settings as fos
from core import sam_extra_contract as reg
from core import sam_extra_notices as sn
from core import vae_degrid as vdg
from core.sam_extra_capabilities import SamExtraCapabilities, _freeze, unknown_capabilities

REPO = Path(__file__).resolve().parent.parent
TS = REPO / "frontend" / "src" / "utils" / "forgeOptionOverrides.ts"
CARD = REPO / "frontend" / "src" / "components" / "ForgeOptionOverridesSettings.vue"
SETTINGS_VIEW = REPO / "frontend" / "src" / "views" / "SettingsView.vue"

DEDUP, SEP, DAVE = fos.OPT_PREFIX_DEDUP, fos.OPT_SEG_SEPARABLE, fos.OPT_DAVE_PRE_DD
KEEP_RESIDENT, KEEP_IN_RAM, SPARSE = fos.OPT_KEEP_RESIDENT, fos.OPT_UNLOAD_KEEP_IN_RAM, fos.OPT_SPARSE_FORGE_GUESS
DEVICE, PRECISION, KEEP_LOADED = fos.OPT_DEGRID_DEVICE, fos.OPT_DEGRID_GPU_PRECISION, fos.OPT_DEGRID_KEEP_LOADED


def known_caps(options=None, *, options_known=True):
    """/config 를 읽은 스냅샷 — ``options`` 에 있는 키만 이 Forge 에 등록돼 있다."""
    return SamExtraCapabilities(status="ok", installed=True, options=_freeze(dict(options or {})),
                                options_known=options_known)


def isolate_pushed_setting(case: unittest.TestCase) -> None:
    """밀어 넣은 값을 테스트마다 비우고 끝나면 원래 상태(tests/__init__ 의 기본 '따름')로 되돌린다(critic B11).
    Forge 설정 잠금 기억(주소별, P10 검토 3)도 비우고 끝나면 다시 비운다."""
    previous = fos._setting._value

    def restore():
        with fos._setting._lock:
            fos._setting._value = previous

    case.addCleanup(restore)
    case.addCleanup(fos.forget_frozen_options)
    fos.reset_forge_option_overrides()
    fos.forget_frozen_options()


class SpecTests(unittest.TestCase):
    def test_options_in_card_order(self):
        """체크박스(bool)와 라디오(선택지 문자열) 11개. 기본값은 스펙이 받는 값이고, 컴포넌트는 선택지 유무로 정해진다."""
        self.assertEqual(len(fos.SPECS), 11)
        self.assertEqual(fos.OPTION_KEYS, tuple(spec.key for spec in fos.SPECS))
        self.assertEqual(len(set(fos.OPTION_KEYS)), 11)
        for spec in fos.SPECS:
            with self.subTest(key=spec.key):
                self.assertTrue(spec.accepts(spec.default), spec.default)
                self.assertIn(spec.effect, fos.GROUPS)
                self.assertTrue(spec.label.strip())
                self.assertIs(fos.SPEC_BY_KEY[spec.key], spec)
                if spec.choices:
                    self.assertEqual(spec.component, "Radio")
                    self.assertIsInstance(spec.default, str)
                    self.assertIn(spec.default, spec.values)
                    self.assertEqual(len(set(spec.values)), len(spec.values))
                    for value, label in spec.choices:
                        self.assertIsInstance(value, str)
                        self.assertTrue(label.strip())
                else:
                    self.assertEqual(spec.component, "Checkbox")
                    self.assertIsInstance(spec.default, bool)
                    self.assertEqual(spec.values, (True, False))
        # 묶음 순서대로 붙어 있다(카드가 묶음 → 스펙 순서로 그린다)
        order = [fos.GROUPS.index(spec.effect) for spec in fos.SPECS]
        self.assertEqual(order, sorted(order))
        self.assertEqual({s.key for s in fos.SPECS if s.choices}, {DEVICE, PRECISION})

    def test_dave_pre_dd_is_the_eighth_option_in_the_result_group(self):
        spec = fos.SPEC_BY_KEY[DAVE]
        self.assertEqual((spec.effect, spec.default, spec.infotext), (fos.RESULT, True, "Anima DAVE pre-DD sigma"))
        self.assertEqual(fos.SPEC_BY_KEY[SPARSE].effect, fos.RESULT)
        self.assertEqual({s.key for s in fos.SPECS if s.effect == fos.RESULT_MINOR}, {DEDUP, SEP, PRECISION})
        self.assertEqual([s.key for s in fos.SPECS if s.effect == fos.RESULT], [SPARSE, DAVE])

    def test_degrid_options_mirror_core_vae_degrid(self):
        """VAE DeGrid 옵션 셋 — 키·기본값·선택지 값은 core/vae_degrid 한 곳에서 온다. GPU 정밀도는 결과가 아주 미세하게
        다른 묶음이고, 결과 기록 'Anima DeGrid precision' 은 확장 런타임이 쓰는 값이라 OptionInfo infotext 가 없다
        (critic C1 — 계약 테스트가 설치된 소스의 infotext '' 와 대조한다). ComfyUI 는 같은 확장 기본값으로 돈다(D6)."""
        self.assertEqual((DEVICE, PRECISION, KEEP_LOADED), (vdg.OPT_DEVICE, vdg.OPT_GPU_PRECISION, vdg.OPT_KEEP_LOADED))
        device, precision, keep = (fos.SPEC_BY_KEY[key] for key in (DEVICE, PRECISION, KEEP_LOADED))
        self.assertEqual((device.effect, precision.effect, keep.effect), (fos.MEMORY, fos.RESULT_MINOR, fos.MEMORY))
        self.assertEqual({device.feature, precision.feature, keep.feature}, {"degrid"})
        self.assertEqual((device.values, precision.values), (vdg.DEVICE_CHOICES, vdg.PRECISION_CHOICES))
        self.assertEqual({"device": device.default, "precision": precision.default, "keep_loaded": keep.default},
                         dict(vdg.EXTENSION_OPTION_DEFAULTS))
        self.assertEqual(dict(vdg.COMFY_OPTIONS), dict(vdg.EXTENSION_OPTION_DEFAULTS))
        self.assertEqual((device.infotext, precision.infotext, keep.infotext), ("", "", ""))
        self.assertNotIn(vdg.KEY_PRECISION, {spec.infotext for spec in fos.SPECS})
        self.assertFalse(any(spec.onchange for spec in (device, precision, keep)))
        # DeGrid 행은 묶음 안에서 기존 행 뒤에 붙는다(카드 순서)
        self.assertEqual(fos.OPTION_KEYS.index(DEVICE), fos.OPTION_KEYS.index(KEEP_IN_RAM) + 1)
        self.assertEqual(fos.OPTION_KEYS.index(KEEP_LOADED), fos.OPTION_KEYS.index(DEVICE) + 1)
        self.assertEqual(fos.OPTION_KEYS.index(PRECISION), fos.OPTION_KEYS.index(SEP) + 1)

    def test_accepts_and_choice_labels(self):
        device, keep = fos.SPEC_BY_KEY[DEVICE], fos.SPEC_BY_KEY[KEEP_LOADED]
        for value, want in (("auto", True), ("cpu", True), ("CPU", False), ("gpu", False), ("", False),
                            (True, False), (False, False), (None, False), (0, False)):
            with self.subTest(radio=value):
                self.assertIs(device.accepts(value), want)
        for value, want in ((True, True), (False, True), ("false", False), ("True", False), (1, False),
                            (None, False), ("auto", False)):
            with self.subTest(checkbox=value):
                self.assertIs(keep.accepts(value), want)
        self.assertEqual(device.choice_label("cpu"), "CPU (VRAM 안 씀, 느림)")
        self.assertEqual(device.choice_label("weird"), "weird")
        self.assertEqual((keep.choice_label(True), keep.choice_label(False)), ("켬", "끔"))

    def test_only_keep_in_ram_has_an_onchange_callback(self):
        self.assertEqual({s.key for s in fos.SPECS if s.onchange}, {KEEP_IN_RAM})

    def test_contract_registry_maps_exactly_the_specs(self):
        mapped = {key for key, entry in reg.OPTIONS.items() if entry["status"] == reg.MAPPED}
        self.assertEqual(mapped, set(fos.OPTION_KEYS))
        for key in fos.OPTION_KEYS:
            with self.subTest(key=key):
                self.assertIn("core/forge_override_settings.py:SPECS", reg.OPTIONS[key]["app"])
        deferred = {key: entry["package"] for key, entry in reg.OPTIONS.items() if entry["status"] == reg.DEFERRED}
        # VAE DeGrid 옵션 셋은 P10 이 맡는다(라디오 둘은 선택지 문자열) — 남은 보류는 레퍼런스 IP-Adapter 둘(P20)과
        # 2026-10-02 검토 제안 옵션 둘(PAG 강도 곡선·내장 NegPiP 스위치 — HOLD, 앱은 Forge 설정을 따른다)
        self.assertEqual(deferred, {"sam3_ipa_duplicate_policy": "P20", "sam3_anima38_reference_ipa": "P20",
                                    "sam3_guidance_pag_cosine_envelope": "HOLD",
                                    "sam3_builtin_negpip_enabled": "HOLD"})
        for key, const in ((DEVICE, "OPT_DEGRID_DEVICE"), (PRECISION, "OPT_DEGRID_GPU_PRECISION"),
                           (KEEP_LOADED, "OPT_DEGRID_KEEP_LOADED")):
            with self.subTest(key=key):
                self.assertEqual(getattr(fos, const), key)
                self.assertIn(f"core/forge_override_settings.py:{const}", reg.OPTIONS[key]["app"])
                self.assertIn("Forge 전용", reg.OPTIONS[key]["note"])
        self.assertNotIn("P10", {entry.get("package") for entry in reg.OPTIONS.values()})

    def test_sparse_infotext_is_the_same_key_the_result_notice_reads(self):
        self.assertEqual(fos.INFOTEXT_SPARSE_GUESS, sn.KEY_SPARSE_LORA_GUESS)


class NormalizeTests(unittest.TestCase):
    def test_only_real_booleans_on_spec_keys_survive_in_spec_order(self):
        raw = {DAVE: False, DEDUP: True, KEEP_IN_RAM: "false", SEP: 1, SPARSE: None,
               "sd_model_checkpoint": "x.safetensors", "sam3_unknown": True}
        before = copy.deepcopy(raw)
        self.assertEqual(fos.normalize_overrides(raw), {DEDUP: True, DAVE: False})
        self.assertEqual(list(fos.normalize_overrides(raw)), [DEDUP, DAVE])
        self.assertEqual(raw, before)
        for bad in (None, [], "x", 3, [(DEDUP, True)]):
            with self.subTest(raw=bad):
                self.assertEqual(fos.normalize_overrides(bad), {})
        self.assertEqual(fos.normalize_overrides(MappingProxyType({SEP: False})), {SEP: False})

    def test_enum_options_keep_only_their_choice_strings(self):
        """라디오 옵션은 선택지 값 문자열만 — bool·모르는 문자열·대소문자가 다른 값은 버린다(= 따름). 체크박스에 문자열은
        여전히 버린다. 순서는 스펙 순서."""
        raw = {PRECISION: "fp16", DEVICE: "cpu", KEEP_LOADED: True, DEDUP: "auto"}
        self.assertEqual(fos.normalize_overrides(raw), {DEVICE: "cpu", KEEP_LOADED: True, PRECISION: "fp16"})
        self.assertEqual(list(fos.normalize_overrides(raw)), [DEVICE, KEEP_LOADED, PRECISION])
        for bad in ({DEVICE: True}, {DEVICE: "gpu"}, {DEVICE: "CPU"}, {DEVICE: ""}, {DEVICE: None},
                    {PRECISION: "fp64"}, {PRECISION: False}, {KEEP_LOADED: "true"}, {KEEP_LOADED: "auto"}):
            with self.subTest(raw=bad):
                self.assertEqual(fos.normalize_overrides(bad), {})

    def test_old_prefs_with_bool_values_only_are_unchanged(self):
        """P10 을 bool 만 받던 때 저장된 파일 — 그대로 읽힌다(DeGrid 키가 없으면 따름)."""
        old = {KEEP_RESIDENT: False, DEDUP: True, SEP: False, SPARSE: True, DAVE: False, KEEP_IN_RAM: True}
        self.assertEqual(fos.normalize_overrides(old), {KEEP_RESIDENT: False, KEEP_IN_RAM: True, DEDUP: True,
                                                        SEP: False, SPARSE: True, DAVE: False})


class PlanTests(unittest.TestCase):
    def test_empty_settings_is_an_empty_plan(self):
        for caps in (None, known_caps({DEDUP: True})):
            plan = fos.plan_overrides({}, caps)
            self.assertFalse(plan)
            self.assertEqual((dict(plan.send), plan.missing, plan.unverified), ({}, (), ()))

    def test_unknown_snapshot_sends_nothing(self):
        """may_use 와 반대 — 모르는 키는 요청 전체를 500 으로 실패시키므로 확인 전에는 보내지 않는다."""
        wanted = {DEDUP: False, DAVE: True}
        for caps in (None, unknown_capabilities(), unknown_capabilities("unreachable"), object(),
                     SimpleNamespace(known=True)):
            with self.subTest(caps=caps):
                plan = fos.plan_overrides(wanted, caps)
                self.assertEqual(dict(plan.send), {})
                self.assertEqual(plan.unverified, (DEDUP, DAVE))
                self.assertEqual(plan.reason, fos.UNVERIFIED_UNKNOWN)

    def test_unread_config_sends_nothing_and_says_why(self):
        plan = fos.plan_overrides({DEDUP: False}, known_caps({}, options_known=False))
        self.assertEqual((dict(plan.send), plan.unverified, plan.reason), ({}, (DEDUP,), fos.UNVERIFIED_NO_CONFIG))

    def test_per_key_existence(self):
        plan = fos.plan_overrides({DEDUP: False, DAVE: False, KEEP_RESIDENT: False}, known_caps({DEDUP: True}))
        self.assertEqual(dict(plan.send), {DEDUP: False})
        self.assertEqual(plan.missing, (KEEP_RESIDENT, DAVE))
        self.assertEqual(plan.unverified, ())

    def test_enum_values_pass_through_like_bools(self):
        """계획·잠금·알림은 값을 보지 않는다 — 라디오 문자열도 bool 과 같은 규칙(키 존재 확인)으로 간다. /config 의 Radio
        시작값은 문자열이라 스냅샷 options 에 문자열로 온다."""
        wanted = {DEVICE: "cpu", PRECISION: "fp16", KEEP_LOADED: True}
        plan = fos.plan_overrides(wanted, known_caps({DEVICE: "auto", KEEP_LOADED: False}))
        self.assertEqual(dict(plan.send), {DEVICE: "cpu", KEEP_LOADED: True})
        self.assertEqual(plan.missing, (PRECISION,))
        (missing,) = fos.plan_notices(plan)
        self.assertIn("VAE DeGrid GPU 정밀도", missing.message)
        early = fos.plan_overrides(wanted, None)
        self.assertEqual((dict(early.send), early.unverified), ({}, (DEVICE, KEEP_LOADED, PRECISION)))
        locked = fos.plan_overrides(wanted, known_caps(dict.fromkeys(wanted, "x")), frozen={DEVICE})
        self.assertEqual((dict(locked.send), locked.frozen), ({KEEP_LOADED: True, PRECISION: "fp16"}, (DEVICE,)))
        merged, sent = fos.merge_into_payload({"prompt": "p"}, plan)
        self.assertEqual(merged["override_settings"], {DEVICE: "cpu", KEEP_LOADED: True})
        self.assertEqual(sent, (DEVICE, KEEP_LOADED))
        self.assertEqual(fos.without_app_keys(merged, sent), {"prompt": "p"})


class FrozenVerdictTests(unittest.TestCase):
    """(P10 검토 3) --freeze-settings 는 Forge 시작 인자라 기능 스냅샷(script-info·/config)에 드러나지 않는다 — 잠금으로
    거절된 키를 주소별로 기억해 보내지 않는다. 백엔드 변경·재연결·수동 새로고침이 잊게 한다."""

    def setUp(self):
        isolate_pushed_setting(self)

    def test_remembered_per_normalized_address_until_forgotten(self):
        fos.remember_frozen_options("http://127.0.0.1:7860/", [DEDUP, "not_a_spec"])
        self.assertEqual(fos.frozen_options("http://127.0.0.1:7860"), frozenset({DEDUP}))
        self.assertEqual(fos.frozen_options(" http://127.0.0.1:7860/ "), frozenset({DEDUP}))
        self.assertEqual(fos.frozen_options("http://127.0.0.1:7861"), frozenset())
        fos.remember_frozen_options("http://127.0.0.1:7860", [DAVE])
        self.assertEqual(fos.frozen_options("http://127.0.0.1:7860"), frozenset({DEDUP, DAVE}))
        fos.remember_frozen_options("http://127.0.0.1:7861", [DEDUP])
        fos.forget_frozen_options("http://127.0.0.1:7860/")
        self.assertEqual(fos.frozen_options("http://127.0.0.1:7860"), frozenset())
        self.assertEqual(fos.frozen_options("http://127.0.0.1:7861"), frozenset({DEDUP}))
        fos.forget_frozen_options()
        self.assertEqual(fos.frozen_options("http://127.0.0.1:7861"), frozenset())
        fos.remember_frozen_options("", [DEDUP])                 # 주소가 없으면 기억하지 않는다
        self.assertEqual(fos.frozen_options(""), frozenset())

    def test_frozen_keys_are_not_sent_whatever_the_snapshot_says(self):
        for snapshot in (known_caps({DEDUP: True, DAVE: True}), None):
            with self.subTest(known=snapshot is not None):
                plan = fos.plan_overrides({DEDUP: False, DAVE: False}, snapshot, frozen={DEDUP})
                self.assertEqual(plan.frozen, (DEDUP,))
                self.assertNotIn(DEDUP, plan.send)
                self.assertNotIn(DEDUP, (*plan.missing, *plan.unverified))
        only = fos.plan_overrides({DEDUP: False}, known_caps({DEDUP: True}), frozen=frozenset({DEDUP}))
        self.assertEqual((dict(only.send), only.frozen, only.missing, only.unverified), ({}, (DEDUP,), (), ()))
        self.assertTrue(only)
        self.assertEqual(fos.plan_overrides({}, known_caps({DEDUP: True}), frozen={DEDUP}), fos.OverridePlan())

    def test_frozen_notice_says_why_and_is_throttled_without_refreshing(self):
        (notice,) = fos.plan_notices(fos.OverridePlan(frozen=(DEDUP,)))
        self.assertEqual((notice.code, notice.level), (sn.CODE_FORGE_OPTION_FROZEN, sn.LEVEL_WARNING))
        self.assertIn("--freeze-settings", notice.message)
        self.assertIn(fos.SPEC_BY_KEY[DEDUP].label, notice.message)
        self.assertIn("다시 연결", notice.message)
        self.assertEqual(sn.NOTICE_MIN_TTL_S[sn.CODE_FORGE_OPTION_FROZEN], sn.PRE_GENERATION_NOTICE_TTL_S)
        self.assertNotIn(sn.CODE_FORGE_OPTION_FROZEN, sn.REFRESH_CAPABILITIES_CODES)   # 새로 받아도 소용없다
        rejected = sn.forge_option_rejected_notice(["x"], frozen=True)
        self.assertIn("다시 연결할 때까지", rejected.message)
        self.assertNotIn("다시 연결할 때까지", sn.forge_option_rejected_notice(["x"], frozen=False).message)


class MergeTests(unittest.TestCase):
    def plan(self, **send):
        return fos.OverridePlan(send=send)

    def test_nothing_to_send_leaves_the_payload_byte_identical(self):
        payload = {"prompt": "p", "steps": 20, "alwayson_scripts": {"X": {"args": [1]}}}
        before = json.dumps(payload)
        for plan in (fos.OverridePlan(), fos.plan_overrides({}, known_caps({DEDUP: True})),
                     fos.plan_overrides({DEDUP: False}, None)):
            with self.subTest(plan=plan):
                merged, sent = fos.merge_into_payload(payload, plan)
                self.assertEqual(sent, ())
                self.assertNotIn("override_settings", merged)
                self.assertEqual(json.dumps(merged), before)
                self.assertEqual(json.dumps(payload), before)

    def test_app_keys_are_added_and_the_restore_flag_is_never_written(self):
        payload = {"prompt": "p"}
        merged, sent = fos.merge_into_payload(payload, self.plan(**{DEDUP: False, DAVE: True}))
        self.assertEqual(merged["override_settings"], {DEDUP: False, DAVE: True})
        self.assertEqual(sent, (DEDUP, DAVE))
        self.assertNotIn("override_settings_restore_afterwards", merged)
        self.assertEqual(payload, {"prompt": "p"})

    def test_caller_keys_win_and_are_not_mutated(self):
        caller = {"sd_vae": "x.safetensors", DEDUP: True}
        payload = {"override_settings": caller}
        merged, sent = fos.merge_into_payload(payload, self.plan(**{DEDUP: False, SEP: False}))
        self.assertEqual(merged["override_settings"], {"sd_vae": "x.safetensors", DEDUP: True, SEP: False})
        self.assertEqual(sent, (SEP,))
        self.assertEqual(caller, {"sd_vae": "x.safetensors", DEDUP: True})
        self.assertIsNot(merged["override_settings"], caller)

    def test_restore_false_or_null_means_no_app_keys(self):
        for flag in (False, None, 0, "true"):
            with self.subTest(flag=flag):
                payload = {"override_settings_restore_afterwards": flag}
                merged, sent = fos.merge_into_payload(payload, self.plan(**{DEDUP: False}))
                self.assertEqual((sent, "override_settings" in merged), ((), False))
        merged, sent = fos.merge_into_payload({"override_settings_restore_afterwards": True},
                                              self.plan(**{DEDUP: False}))
        self.assertEqual(sent, (DEDUP,))

    def test_out_of_spec_keys_and_non_bool_values_can_not_be_sent(self):
        plan = fos.OverridePlan(send={"sd_model_checkpoint": "x", "forge_additional_modules": [],
                                      DEDUP: "False", DEVICE: True, PRECISION: "fp64", KEEP_LOADED: "true"})
        merged, sent = fos.merge_into_payload({}, plan)
        self.assertEqual((sent, "override_settings" in merged), ((), False))

    def test_a_valid_enum_value_is_sent_next_to_rejected_ones(self):
        plan = fos.OverridePlan(send={DEVICE: "cpu", PRECISION: "FP16", DEDUP: False})
        merged, sent = fos.merge_into_payload({"override_settings": {"sd_vae": "v"}}, plan)
        self.assertEqual(merged["override_settings"], {"sd_vae": "v", DEVICE: "cpu", DEDUP: False})
        self.assertEqual(sent, (DEVICE, DEDUP))

    def test_non_mapping_caller_value_is_left_alone(self):
        merged, sent = fos.merge_into_payload({"override_settings": "weird"}, self.plan(**{DEDUP: False}))
        self.assertEqual((merged["override_settings"], sent), ("weird", ()))

    def test_without_app_keys_restores_the_callers_shape(self):
        merged, sent = fos.merge_into_payload({"prompt": "p"}, self.plan(**{DEDUP: False}))
        self.assertEqual(fos.without_app_keys(merged, sent), {"prompt": "p"})
        merged, sent = fos.merge_into_payload({"override_settings": {"sd_vae": "v"}}, self.plan(**{DEDUP: False}))
        stripped = fos.without_app_keys(merged, sent)
        self.assertEqual(stripped, {"override_settings": {"sd_vae": "v"}})
        self.assertEqual(merged["override_settings"], {"sd_vae": "v", DEDUP: False})   # 입력 불변


class NoticeTests(unittest.TestCase):
    def test_plan_notices(self):
        self.assertEqual(fos.plan_notices(fos.OverridePlan(send={DEDUP: False})), [])
        missing, unverified = fos.plan_notices(fos.OverridePlan(missing=(DAVE,), unverified=(DEDUP,),
                                                                reason=fos.UNVERIFIED_NO_CONFIG))
        self.assertEqual((missing.code, missing.level), (sn.CODE_FORGE_OPTION_MISSING, sn.LEVEL_WARNING))
        self.assertIn(fos.SPEC_BY_KEY[DAVE].label, missing.message)
        self.assertEqual((unverified.code, unverified.level), (sn.CODE_FORGE_OPTION_UNVERIFIED, sn.LEVEL_INFO))
        self.assertIn("/config", unverified.message)
        (early,) = fos.plan_notices(fos.OverridePlan(unverified=(DEDUP,), reason=fos.UNVERIFIED_UNKNOWN))
        self.assertIn("확인이 아직", early.message)

    def test_new_codes_are_throttled_like_pre_generation_warnings(self):
        """배치 100장이 30초마다 같은 알림을 다시 띄우지 않게(critic B14)."""
        for code in (sn.CODE_APP_BLOCK_RETRIED, sn.CODE_FORGE_OPTION_MISSING, sn.CODE_FORGE_OPTION_UNVERIFIED,
                     sn.CODE_FORGE_OPTION_REJECTED):
            with self.subTest(code=code):
                self.assertEqual(sn.NOTICE_MIN_TTL_S[code], sn.PRE_GENERATION_NOTICE_TTL_S)

    def test_rejections_refresh_the_gui_snapshot(self):
        """재시도는 캐시를 버리지 않는다 — 알림을 띄울 때 GUI 가 스냅샷을 다시 받는다(critic A1)."""
        self.assertTrue({sn.CODE_APP_BLOCK_RETRIED, sn.CODE_FORGE_OPTION_REJECTED} <= sn.REFRESH_CAPABILITIES_CODES)
        self.assertNotIn(sn.CODE_FORGE_OPTION_UNVERIFIED, sn.REFRESH_CAPABILITIES_CODES)
        rejected = sn.forge_option_rejected_notice(["x"], frozen=True, where="Refine")
        self.assertIn("--freeze-settings", rejected.message)
        self.assertIn("Refine 을(를) 다시 실행", rejected.message)
        self.assertIn("다시 생성했습니다", sn.forge_option_rejected_notice(["x"], frozen=False).message)
        retried = sn.app_block_retried_notice("DoRA Inference Mode")
        self.assertEqual(retried.level, sn.LEVEL_INFO)   # 앱 기본값일 수 있다 — 경고로 띄우지 않는다(A6)


class PushedSettingTests(unittest.TestCase):
    """생성 워커는 GUI 가 밀어 넣은 값만 읽는다. 폴백은 실제 config/ui_prefs.json 을 읽지 않게 막고 본다(critic B11)."""

    def setUp(self):
        isolate_pushed_setting(self)

    def test_pushed_value_never_touches_the_prefs_file(self):
        fos.update_forge_option_overrides_from_prefs({fos.PREF_KEY: {DEDUP: False, SEP: "no"}, "theme": "dark"})
        with mock.patch.object(fos, "forge_option_overrides_from_prefs_file",
                               side_effect=AssertionError("worker must not open ui_prefs.json")), \
             mock.patch("builtins.open", side_effect=AssertionError("worker must not open files")):
            self.assertEqual(fos.forge_option_overrides_setting(), {DEDUP: False})
            fos.update_forge_option_overrides_from_prefs({"theme": "dark"})   # 키 없음 = 전부 따름
            self.assertEqual(fos.forge_option_overrides_setting(), {})
            fos.update_forge_option_overrides_from_prefs(None)
            self.assertEqual(fos.forge_option_overrides_setting(), {})

    def test_file_fallback_only_before_the_first_push(self):
        with mock.patch.object(fos, "forge_option_overrides_from_prefs_file",
                               return_value={DAVE: False, "junk": True}) as fallback:
            self.assertEqual(fos.forge_option_overrides_setting(), {DAVE: False})
            fallback.assert_called_once_with()
            fos.set_forge_option_overrides({DEDUP: True})
            self.assertEqual(fos.forge_option_overrides_setting(), {DEDUP: True})
            fallback.assert_called_once_with()

    def test_returned_value_is_a_copy(self):
        fos.set_forge_option_overrides({DEDUP: False})
        fos.forge_option_overrides_setting()[DAVE] = True
        self.assertEqual(fos.forge_option_overrides_setting(), {DEDUP: False})

    def test_prefs_file_cache_reads_a_real_file_normalized(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "ui_prefs.json")
            cache = fos._PrefsOverridesCache(lambda: path)
            self.assertEqual(cache.get(), {})                              # 파일 없음
            with open(path, "w", encoding="utf-8") as fh:
                json.dump({fos.PREF_KEY: {SEP: False, KEEP_IN_RAM: "x"}}, fh)
            self.assertEqual(cache.get(), {SEP: False})
            with open(path, "w", encoding="utf-8") as fh:
                fh.write("{broken")
            self.assertEqual(cache.get(), {})

    def test_boot_restore_pushes_the_setting(self):
        from ui.generator_main import GeneratorMainUI
        host = SimpleNamespace(_apply_anima_guard_prefs=lambda _prefs: None)
        with mock.patch("core.forge_output_policy.update_forge_save_outputs_from_prefs"):
            GeneratorMainUI._restore_runtime_prefs(host, {fos.PREF_KEY: {DAVE: False}})
        self.assertTrue(fos._setting.is_set())
        self.assertEqual(fos.forge_option_overrides_setting(), {DAVE: False})

    def test_save_ui_prefs_normalizes_the_file_and_pushes_the_merged_prefs(self):
        from tests.test_vue_bridge_caption import _PrefsHost
        with tempfile.TemporaryDirectory() as tmp:
            prefs_path = Path(tmp) / "config" / "ui_prefs.json"
            prefs_path.parent.mkdir(parents=True)
            prefs_path.write_text(json.dumps({"theme": "dark", fos.PREF_KEY: {DEDUP: False}}), encoding="utf-8")
            bridge = SimpleNamespace(approved_caption_out_dirs=lambda: [], showNotification=mock.Mock())
            host = _PrefsHost(bridge)
            # 파일 폴백은 막는다 — 워커가 보는 값은 핸들러가 밀어 넣은 것이어야 한다
            with mock.patch("core.ui_prefs.ui_prefs_path", return_value=str(prefs_path)), \
                 mock.patch("core.forge_output_policy.update_forge_save_outputs_from_prefs"), \
                 mock.patch.object(fos, "forge_option_overrides_from_prefs_file", return_value={}):
                self.assertFalse(fos._setting.is_set())
                host.handle("save_ui_prefs", {fos.PREF_KEY: {DAVE: False, SEP: "false", "sd_vae": True}})
                stored = json.loads(prefs_path.read_text(encoding="utf-8"))
                self.assertEqual(stored[fos.PREF_KEY], {DAVE: False})      # Vue 가 보낸 dict 로 교체·정규화
                self.assertEqual(stored["theme"], "dark")
                self.assertTrue(fos._setting.is_set())
                self.assertEqual(fos.forge_option_overrides_setting(), {DAVE: False})
                host.handle("save_ui_prefs", {fos.PREF_KEY: {}})           # 모두 따름 → 키를 지운다
                stored = json.loads(prefs_path.read_text(encoding="utf-8"))
                self.assertNotIn(fos.PREF_KEY, stored)
                self.assertEqual(fos.forge_option_overrides_setting(), {})
            bridge.showNotification.emit.assert_not_called()

    def test_save_ui_prefs_keeps_enum_choices_and_drops_bad_values(self):
        from tests.test_vue_bridge_caption import _PrefsHost
        with tempfile.TemporaryDirectory() as tmp:
            prefs_path = Path(tmp) / "config" / "ui_prefs.json"
            prefs_path.parent.mkdir(parents=True)
            prefs_path.write_text(json.dumps({"theme": "dark"}), encoding="utf-8")
            bridge = SimpleNamespace(approved_caption_out_dirs=lambda: [], showNotification=mock.Mock())
            host = _PrefsHost(bridge)
            with mock.patch("core.ui_prefs.ui_prefs_path", return_value=str(prefs_path)), \
                 mock.patch("core.forge_output_policy.update_forge_save_outputs_from_prefs"), \
                 mock.patch.object(fos, "forge_option_overrides_from_prefs_file", return_value={}):
                host.handle("save_ui_prefs", {fos.PREF_KEY: {DEVICE: "cpu", PRECISION: "fp64", KEEP_LOADED: True,
                                                             DEDUP: "auto"}})
                stored = json.loads(prefs_path.read_text(encoding="utf-8"))
                self.assertEqual(stored[fos.PREF_KEY], {DEVICE: "cpu", KEEP_LOADED: True})
                self.assertEqual(fos.forge_option_overrides_setting(), {DEVICE: "cpu", KEEP_LOADED: True})
            bridge.showNotification.emit.assert_not_called()

    def test_workers_read_only_the_pushed_setting(self):
        source = (REPO / "backends" / "webui_backend.py").read_text(encoding="utf-8")
        self.assertIn("forge_option_overrides_setting()", source)
        self.assertIsNone(re.search(r"forge_option_overrides_from_prefs_file\(\)", source))

    def test_tests_package_pushes_the_follow_default(self):
        init = (REPO / "tests" / "__init__.py").read_text(encoding="utf-8")
        self.assertIn("set_forge_option_overrides({})", init)


class FrontendMirrorTests(unittest.TestCase):
    """Vue 카드의 표(frontend/src/utils/forgeOptionOverrides.ts)는 파이썬 SPECS 의 거울이다."""

    @classmethod
    def setUpClass(cls):
        cls.ts = TS.read_text(encoding="utf-8")
        cls.card = CARD.read_text(encoding="utf-8")
        cls.view = SETTINGS_VIEW.read_text(encoding="utf-8")

    @staticmethod
    def _ts_default(value):
        return ("true" if value else "false") if isinstance(value, bool) else f"'{value}'"

    def test_ts_specs_match_python(self):
        """한 줄 모양: { key, label, group, default: true|false|'값', infotext, choices: [['값', '라벨'], …] }."""
        rows = re.findall(r"\{ key: '([^']+)', label: '([^']+)', group: '([^']+)', default: (true|false|'[^']*'), "
                          r"infotext: '([^']*)', choices: \[(.*?)\] \}", self.ts)
        parsed = [(key, label, group, default, infotext, tuple(re.findall(r"\['([^']+)', '([^']+)'\]", choices)))
                  for key, label, group, default, infotext, choices in rows]
        expected = [(s.key, s.label, s.effect, self._ts_default(s.default), s.infotext, tuple(s.choices))
                    for s in fos.SPECS]
        self.assertEqual(parsed, expected)
        groups = re.findall(r"\{ id: '(memory|result_minor|result)', title: '[^']+' \}", self.ts)
        self.assertEqual(tuple(groups), fos.GROUPS)
        self.assertIn(f"export const PREF_KEY = '{fos.PREF_KEY}'", self.ts)
        for key in fos.OPTION_KEYS:
            self.assertRegex(self.ts, rf"(?m)^  {key}: ['\"]", f"DESCRIPTIONS 에 {key} 가 없다")

    def test_card_builds_each_row_from_its_spec(self):
        """라디오 행은 선택지 라벨을 그린다 — 카드가 줄마다 rowChoices(spec)로 목록을 만들고 spec 으로 값을 바꾼다."""
        self.assertIn("rowChoices(spec)", self.card)
        self.assertIn("setChoice(spec, $event)", self.card)
        self.assertIn("export function rowChoices(", self.ts)
        self.assertIn("export type Overrides = Record<string, boolean | string>", self.ts)

    def test_card_uses_the_existing_prefs_bridge_only(self):
        """SpectrumSettings 와 같은 방식(critic C): save_ui_prefs + getUiPrefs + uiPrefsLoaded, 새 브리지 이름 없음."""
        self.assertIn('<script setup lang="ts">', self.card)
        self.assertIn("requestAction('save_ui_prefs'", self.card)
        self.assertIn("getUiPrefs", self.card)
        self.assertIn("onBackendEvent('uiPrefsLoaded'", self.card)
        self.assertIn("useSamExtraCapabilities", self.card)
        self.assertNotIn("persistUiPrefs", self.card)

    def test_settings_view_hosts_the_card_on_the_forge_tab(self):
        forge_tab = self.view.split('data-settings-tab="forge"', 1)[1].split("data-settings-tab=", 1)[0]
        self.assertIn("<ForgeOptionOverridesSettings />", forge_tab)
        self.assertIn("import ForgeOptionOverridesSettings from '../components/ForgeOptionOverridesSettings.vue'",
                      self.view)
        self.assertNotIn("Forge 는 Forge 설정의 SAM3 항목을 따릅니다", self.view)
        self.assertIn("Forge 탭의 sam-extra 설정", self.view)


if __name__ == "__main__":
    unittest.main()
