"""sam-extra 런타임 기능 스냅샷 (core/sam_extra_capabilities.py · core/sam_extra_probe.py).

픽스처 tests/fixtures/sam_extra_live_*.json 은 2026-09-25 실행 중이던 Forge classic 에 GET 으로
받은 응답을 줄인 것이다(sam-extra 0.30.0 작업 사본). script-info 는 계약 가드와 같은
tests/fixtures/sam_extra_script_info.json 의 ``scripts`` 를 쓴다(tools/refresh_sam_extra_fixture.py 가
갱신). 네트워크·Forge·torch 없이 돈다.
"""
from __future__ import annotations

import copy
import json
import os
import re
import threading
import unittest
from unittest import mock

from core import anima_guidance, sam3_args
from core import sam_extra_capabilities as caps_mod
from core import sam_extra_probe as probe
from core.sam_extra_capabilities import (
    EP_CN_MODELS, EP_CN_MODULES, EP_CONTRACT, EP_EXTENSIONS, EP_GRADIO_CONFIG, EP_LORA_CONFIG,
    EP_MEMOS, EP_REFERENCE, EP_SCRIPTS, EP_SCRIPT_INFO, EP_TIPO, HttpResult, SamExtraCapabilities,
    build_capabilities, unknown_capabilities,
)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIXTURES = os.path.join(ROOT, "tests", "fixtures")
BASE = "http://127.0.0.1:7860"


def _fixture(name):
    with open(os.path.join(FIXTURES, f"sam_extra_live_{name}.json"), encoding="utf-8") as f:
        return json.load(f)


def _script_info_fixture():
    with open(os.path.join(FIXTURES, "sam_extra_script_info.json"), encoding="utf-8") as f:
        return _with_current_dd_shape(json.load(f)["scripts"])


# 확장 Detail Daemon ui() 인자 13 — Hires Pass 체크박스(맨 뒤 append, 기본 끔)의 script-info 모양
_DD_HIRES_ARG = {"label": "Hires Pass", "value": False, "minimum": None, "maximum": None, "step": None,
                 "choices": None}


def _with_current_dd_shape(script_info):
    """녹화(sam-extra 0.30.0)는 Detail Daemon Hires Pass(인자 13) 추가 전이다. 픽스처를 다시 녹화하기 전까지
    지금 확장 모양(14개)으로 맞춘다 — 이미 14개면 그대로 둔다. 13개 빌드는 test_detail_daemon_without_hires 가 본다."""
    for entry in script_info:
        if entry.get("name") == "anima detail daemon" and len(entry.get("args") or []) == 13:
            entry["args"].append(dict(_DD_HIRES_ARG))
    return script_info


def live_bodies():
    """경로 → 녹화된 본문 (없는 라우트는 상태 코드만)."""
    status = _fixture("probe_status")["status"]
    return {
        EP_SCRIPTS: _fixture("scripts"),
        EP_SCRIPT_INFO: _script_info_fixture(),
        EP_EXTENSIONS: _fixture("extensions"),
        EP_CN_MODELS: _fixture("cn_models"),
        EP_CN_MODULES: _fixture("cn_modules"),
        EP_LORA_CONFIG: _fixture("lora_config"),
        EP_GRADIO_CONFIG: _fixture("gradio_config"),
        EP_MEMOS: status[EP_MEMOS],
        EP_TIPO: status[EP_TIPO],
        EP_REFERENCE: status[EP_REFERENCE],
        EP_CONTRACT: status[EP_CONTRACT],
    }


class FakeGet:
    """주입용 GET. 값이 int 면 그 상태 코드(본문 없음), 예외면 연결 실패."""

    def __init__(self, bodies):
        self.bodies = bodies
        self.calls = []
        self.lock = threading.Lock()

    def __call__(self, url, *, headers=None, timeout=None, want_body=True):
        path = url[len(BASE):]
        with self.lock:
            self.calls.append((path, dict(headers or {}), want_body))
        value = self.bodies.get(path, 404)
        if isinstance(value, Exception):
            return HttpResult(None, None, type(value).__name__)
        if isinstance(value, int):
            return HttpResult(value)
        return HttpResult(200, copy.deepcopy(value) if want_body else None)

    def paths(self):
        return [path for path, _h, _b in self.calls]


def build(bodies):
    fake = FakeGet(bodies)
    return probe.fetch_capabilities(BASE, http_get=fake), fake


def _script(bodies, title, *, img2img=False):
    return next(s for s in bodies[EP_SCRIPT_INFO] if s["name"] == title and s["is_img2img"] == img2img)


def _codes(caps):
    return [w["code"] for w in caps.warnings]


class LiveFixtureTests(unittest.TestCase):
    """녹화된 v0.30.0 응답 → 모든 기능이 보이고 의미 판정이 맞는다."""

    @classmethod
    def setUpClass(cls):
        cls.caps, cls.fake = build(live_bodies())

    def test_status_and_features(self):
        c = self.caps
        self.assertEqual(c.status, "ok")
        self.assertTrue(c.known and c.installed)
        for flag in ("sam3", "anima_guidance", "skimmed_cfg", "detail_daemon", "anima38", "dora",
                     "vae2x", "lora_manager"):
            self.assertTrue(getattr(c, flag), flag)
        # 녹화 시점엔 아직 없는 라우트 (메모·TIPO·레퍼런스·계약)
        for flag in ("memo_routes", "tipo_route", "reference_route", "contract_route"):
            self.assertFalse(getattr(c, flag), flag)
        self.assertEqual(c.anima_guidance_argc, 62)
        self.assertIs(c.detail_daemon_hires, True)       # 인자 13(Hires Pass)이 있다

    def test_script_details_match_app_specs(self):
        pag = self.caps.script(anima_guidance.SCRIPT_PERTURBATION)
        self.assertEqual((pag["live_argc"], pag["spec_argc"]), (62, len(anima_guidance.PERTURBATION_SPEC)))
        self.assertEqual(pag["trailing_unmapped"], ())
        self.assertTrue(pag["img2img"])
        self.assertEqual(self.caps.script("Anima Detail Daemon")["live_argc"], 14)
        self.assertEqual(self.caps.script("dora inference mode")["spec_argc"], None)  # 앱이 아직 안 보냄
        for code in ("args_fewer", "args_more", "pag_smc_auto_old_build",
                     "extension_missing", "script_missing"):
            self.assertNotIn(code, _codes(self.caps))

    def test_sam3_keys_compare_without_enable_flag(self):
        self.assertEqual(self.caps.sam3_keys["missing_in_extension"], ())
        self.assertEqual(self.caps.sam3_keys["unknown_to_app"], ())

    def test_live_choices(self):
        choices = self.caps.choices
        self.assertEqual(choices["controlnet_modules"][0], "None")   # 대문자 'None' — S6
        self.assertIn("anima-lllite-inpainting-v2", choices["controlnet_models"])
        self.assertEqual(choices["clip_l"], ())                       # arg45 choices=[] (value None)
        self.assertIn("Anima-3.8B-expanded_adapter.safetensors", choices["anima38_adapters"])
        self.assertEqual(choices["vae2x_decoders"][0], "None")
        self.assertEqual(choices["smc_presets"][0], "Auto")
        self.assertNotIn("dd_presets", choices)                     # 프리셋은 원본에 없다(앱도 안 읽는다)
        self.assertEqual(len(choices["dora_modes"]), 4)
        self.assertEqual(len(choices["dora_weak_scopes"]), 3)

    def test_cn_module_case_warning_matches_app_list(self):
        unknown = [m for m in sam3_args.CN_MODULES if m not in self.caps.choices["controlnet_modules"]]
        warned = [w for w in self.caps.warnings if w["code"] == "sam3_cn_module_not_live"]
        self.assertEqual(bool(unknown), bool(warned))
        if unknown:
            for name in unknown:
                self.assertIn(name, warned[0]["message"])

    def test_options_from_gradio_config(self):
        self.assertTrue(self.caps.options_known)
        self.assertEqual(len(self.caps.options), 14)
        self.assertIs(self.caps.options["sam3_unload_keep_in_ram"], True)
        self.assertTrue(self.caps.has_option("sam3_anima38_keep_resident"))
        self.assertFalse(self.caps.has_option("sam3_not_an_option"))
        self.assertIn("expand", self.caps.gradio_api)
        self.assertIn("handle_anima_reference_click", self.caps.gradio_api)
        self.assertNotIn("refresh_model_list", self.caps.gradio_api)

    def test_version_hints_from_extensions(self):
        v = self.caps.version
        self.assertEqual(v["folder"], "forge_sam3_extension")
        self.assertEqual(v["commit"], "818b8fee")
        self.assertIs(v["enabled"], True)
        self.assertIsNone(v["extension_version"])   # 계약 라우트 없음

    def test_probe_uses_notebook_header_and_skips_bodies(self):
        calls = {path: (headers, want) for path, headers, want in self.fake.calls}
        self.assertEqual(calls[EP_MEMOS][0].get("X-SAM3-Notebook"), "1")
        self.assertFalse(calls[EP_MEMOS][1])         # 메모 본문은 받지 않는다(존재 확인만)
        self.assertEqual(calls[EP_LORA_CONFIG][0].get("X-SAM3-Notebook"), "1")   # LoRA 라우트도 같은 헤더
        self.assertTrue(calls[EP_LORA_CONFIG][1])    # available 을 읽어야 해서 본문은 받는다
        self.assertFalse(calls[EP_REFERENCE][1])
        self.assertNotIn("/sam3-lora/spawn", self.fake.paths())   # 부작용 있는 라우트 금지
        self.assertTrue(all(not p.startswith("/sdapi/v1/options") for p in self.fake.paths()))

    def test_to_dict_is_json_without_backend_url(self):
        data = self.caps.to_dict()
        text = json.dumps(data, ensure_ascii=False)
        self.assertNotIn("127.0.0.1", text)
        self.assertEqual(set(data["features"]), set(caps_mod.FEATURE_FLAGS))
        self.assertTrue(data["known"])
        self.assertIsInstance(data["scripts"]["anima perturbation guidance"]["trailing_unmapped"], list)

    def test_snapshot_is_read_only(self):
        with self.assertRaises(Exception):
            self.caps.sam3 = False  # type: ignore[misc]
        with self.assertRaises(TypeError):
            self.caps.options["sam3_unload_keep_in_ram"] = False  # type: ignore[index]


class VariantTests(unittest.TestCase):
    def test_missing_extension_skips_gradio_config(self):
        bodies = live_bodies()
        bodies[EP_SCRIPTS] = {"txt2img": ["controlnet", "seed"], "img2img": ["controlnet"]}
        bodies[EP_SCRIPT_INFO] = [s for s in bodies[EP_SCRIPT_INFO] if s["name"] not in caps_mod.SAM_EXTRA_TITLES]
        bodies[EP_EXTENSIONS] = [e for e in bodies[EP_EXTENSIONS] if e["name"] != "forge_sam3_extension"]
        bodies[EP_LORA_CONFIG] = 404
        caps, fake = build(bodies)
        self.assertEqual(caps.status, "ok")
        self.assertFalse(caps.installed or caps.sam3 or caps.lora_manager)
        self.assertNotIn(EP_GRADIO_CONFIG, fake.paths())
        self.assertFalse(caps.options_known)
        self.assertIsNone(caps.has_option("sam3_unload_keep_in_ram"))
        self.assertIn("extension_missing", _codes(caps))
        self.assertFalse(caps.may_use("sam3"))
        self.assertEqual(caps.version["folder"], None)

    def test_unreachable_never_blocks(self):
        caps, _ = build({path: ConnectionError() for path in live_bodies()})
        self.assertEqual(caps.status, "unreachable")
        self.assertFalse(caps.known)
        self.assertTrue(caps.may_use("sam3") and caps.may_use("dora"))
        self.assertEqual(caps.errors[EP_SCRIPTS], "ConnectionError")
        with self.assertRaises(KeyError):
            caps.may_use("no_such_feature")

    def test_http_errors_are_error_status(self):
        bodies = live_bodies()
        bodies[EP_SCRIPTS] = 500
        bodies[EP_SCRIPT_INFO] = 500
        caps, _ = build(bodies)
        self.assertEqual(caps.status, "error")
        self.assertEqual(caps.errors[EP_SCRIPTS], "HTTP 500")

    def test_script_info_alone_is_enough(self):
        bodies = live_bodies()
        bodies[EP_SCRIPTS] = 404
        caps, _ = build(bodies)
        self.assertEqual(caps.status, "ok")
        self.assertTrue(caps.sam3 and caps.dora)

    def test_old_pag_build_warns_about_smc_and_dropped_args(self):
        bodies = live_bodies()
        pag = _script(bodies, "anima perturbation guidance")
        pag["args"] = pag["args"][:57]
        caps, _ = build(bodies)
        self.assertEqual(caps.anima_guidance_argc, 57)
        self.assertIn("pag_smc_auto_old_build", _codes(caps))
        detail = caps.script("anima perturbation guidance")
        self.assertEqual(list(detail["trailing_unmapped"]),
                         [key for key, *_ in anima_guidance.PERTURBATION_SPEC[57:]])
        fewer = next(w for w in caps.warnings if w["code"] == "args_fewer")
        self.assertIn("RDC", fewer["message"])
        self.assertIn("SMC", fewer["message"])
        self.assertIs(caps.detail_daemon_hires, True)

    def test_detail_daemon_without_hires(self):
        # Hires Pass(인자 13) 추가 전 빌드: 잘려서 무시될 칸을 알리고, 앱은 dd_hires 생성에 경고한다
        bodies = live_bodies()
        for img2img in (False, True):
            dd = _script(bodies, "anima detail daemon", img2img=img2img)
            dd["args"] = dd["args"][:13]
        caps, _ = build(bodies)
        self.assertIs(caps.detail_daemon_hires, False)
        self.assertEqual(list(caps.script("anima detail daemon")["trailing_unmapped"]), ["dd_hires"])
        fewer = next(w for w in caps.warnings if w["code"] == "args_fewer")
        self.assertEqual(fewer["feature"], "detail_daemon")
        self.assertIn("HIRES", fewer["message"])
        self.assertIn("Hires Pass",
                      anima_guidance.detail_daemon_hires_note({"dd_enabled": True, "dd_hires": True}, caps))
        self.assertIsNone(anima_guidance.detail_daemon_hires_note({"dd_enabled": True}, caps))
        self.assertIs(caps.to_dict()["detail_daemon_hires"], False)
        self.assertNotIn("detail_daemon_scale", caps.to_dict())

    def test_detail_daemon_hires_unknown_without_the_script(self):
        bodies = live_bodies()
        bodies[EP_SCRIPT_INFO] = [s for s in bodies[EP_SCRIPT_INFO] if s["name"] != "anima detail daemon"]
        bodies[EP_SCRIPTS] = {mode: [t for t in titles if t != "anima detail daemon"]
                              for mode, titles in bodies[EP_SCRIPTS].items()}
        caps, _ = build(bodies)
        self.assertFalse(caps.detail_daemon)
        self.assertIsNone(caps.detail_daemon_hires)
        self.assertIsNone(unknown_capabilities().detail_daemon_hires)

    def test_newer_extension_with_more_args(self):
        bodies = live_bodies()
        pag = _script(bodies, "anima perturbation guidance")
        pag["args"].append({"label": "New toggle", "value": False})
        caps, _ = build(bodies)
        self.assertEqual(caps.script("anima perturbation guidance")["extra_live_args"], 1)
        self.assertIn("args_more", _codes(caps))

    def test_sam3_key_drift(self):
        bodies = live_bodies()
        state = _script(bodies, "sam3 mask")["args"][1]["value"]
        state.pop("sam3_seed")
        state["sam3_brand_new"] = 1
        caps, _ = build(bodies)
        self.assertEqual(caps.sam3_keys["missing_in_extension"], ("sam3_seed",))
        self.assertEqual(caps.sam3_keys["unknown_to_app"], ("sam3_brand_new",))
        self.assertIn("sam3_keys_unknown_to_extension", _codes(caps))
        self.assertIn("sam3_keys_unknown_to_app", _codes(caps))

    def test_partial_install_reports_missing_script(self):
        bodies = live_bodies()
        bodies[EP_SCRIPTS]["txt2img"].remove("dora inference mode")
        caps, _ = build(bodies)
        self.assertTrue(caps.installed)
        self.assertFalse(caps.dora)
        missing = [w for w in caps.warnings if w["code"] == "script_missing"]
        self.assertEqual([w["feature"] for w in missing], ["dora"])

    def test_new_routes_are_detected(self):
        bodies = live_bodies()
        bodies[EP_MEMOS] = {"revision": 0, "memos": []}
        bodies[EP_TIPO] = {"ready": True}
        bodies[EP_REFERENCE] = 405          # POST 전용 라우트에 GET → 405 = 존재
        bodies[EP_CONTRACT] = {"version": "0.31.0", "scripts": {}}
        caps, _ = build(bodies)
        self.assertTrue(caps.memo_routes and caps.tipo_route and caps.reference_route and caps.contract_route)
        self.assertEqual(caps.version["extension_version"], "0.31.0")
        self.assertNotIn(EP_REFERENCE, caps.errors)

    # 라우트 확인이 실패(타임아웃·연결 끊김·5xx·408·429)한 것은 '없음'이 아니다 — 스크립트로 설치가
    # 확인됐으면 있다고 보고 보낸다(모르면 지금처럼 보냄). 404 만 확정된 부재다.
    _ROUTES = ((EP_MEMOS, "memo_routes"), (EP_TIPO, "tipo_route"), (EP_REFERENCE, "reference_route"),
               (caps_mod.EP_TILE_REPAIR, "tile_repair_route"))

    def test_route_probe_failure_is_not_absence(self):
        for failure in (HttpResult(None, None, "ReadTimeout"), HttpResult(None, None, "ConnectionError"),
                        HttpResult(500), HttpResult(502), HttpResult(503), HttpResult(408), HttpResult(429)):
            for path, flag in self._ROUTES:
                with self.subTest(failure=failure, flag=flag):
                    responses = {p: HttpResult(404) for p, _f in self._ROUTES}
                    responses.update({EP_SCRIPTS: HttpResult(200, {"txt2img": ["sam3 mask"], "img2img": []}),
                                      EP_SCRIPT_INFO: HttpResult(200, []), path: failure})
                    caps = build_capabilities(responses)
                    self.assertTrue(caps.known and caps.installed)
                    self.assertTrue(getattr(caps, flag))
                    self.assertTrue(caps.may_use(flag))
                    self.assertTrue(caps.to_dict()["features"][flag])
                    self.assertIn(path, caps.errors)            # 실패 사유는 그대로 남는다
                    unverified = [w for w in caps.warnings if w["code"] == "route_unverified"]
                    self.assertEqual([w["feature"] for w in unverified], [flag])
                    for other_path, other in self._ROUTES:      # 확정된 404 는 그대로 '없음'
                        if other != flag:
                            self.assertFalse(getattr(caps, other), other)

    def test_route_404_or_missing_extension_still_blocks(self):
        scripts = {EP_SCRIPTS: HttpResult(200, {"txt2img": ["sam3 mask"], "img2img": []}),
                   EP_SCRIPT_INFO: HttpResult(200, [])}
        for path, flag in self._ROUTES:
            with self.subTest(flag=flag):
                caps = build_capabilities({**scripts, path: HttpResult(404)})
                self.assertFalse(getattr(caps, flag))
                self.assertFalse(caps.may_use(flag))
                self.assertNotIn("route_unverified", _codes(caps))
                # 스크립트 목록이 sam-extra 가 없다고 확정했으면 라우트 타임아웃도 '없음'이다
                caps = build_capabilities({EP_SCRIPTS: HttpResult(200, {"txt2img": ["seed"], "img2img": []}),
                                           EP_SCRIPT_INFO: HttpResult(200, []),
                                           path: HttpResult(None, None, "ReadTimeout")})
                self.assertFalse(caps.installed)
                self.assertFalse(caps.may_use(flag))
                self.assertNotIn("route_unverified", _codes(caps))
                # 수집기가 묻지 않은 경로는 확인 실패가 아니다
                self.assertFalse(build_capabilities(scripts).may_use(flag))

    def test_lora_manager_vendor_missing(self):
        bodies = live_bodies()
        bodies[EP_LORA_CONFIG] = {"available": False, "replace": False, "port": 8765}
        caps, _ = build(bodies)
        self.assertFalse(caps.lora_manager)
        self.assertIn("lora_manager_unavailable", _codes(caps))

    def test_lora_config_behind_same_origin_header(self):
        # 헤더를 요구하는 확장: 헤더 없는 GET 은 403. 수집기는 메모처럼 헤더를 보내므로 200 → 있음.
        class GuardedGet(FakeGet):
            def __call__(self, url, *, headers=None, timeout=None, want_body=True):
                if url[len(BASE):] == EP_LORA_CONFIG and (headers or {}).get("X-SAM3-Notebook") != "1":
                    return HttpResult(403)
                return super().__call__(url, headers=headers, timeout=timeout, want_body=want_body)

        caps = probe.fetch_capabilities(BASE, http_get=GuardedGet(live_bodies()))
        self.assertTrue(caps.lora_manager and caps.may_use("lora_manager"))
        self.assertNotIn(EP_LORA_CONFIG, caps.errors)
        self.assertNotIn("route_unverified", _codes(caps))

    def test_lora_config_refusal_is_not_reported_as_missing(self):
        # 401(로그인)·403(같은 출처 헤더 거절): 라우트는 있지만 앱이 못 쓴다 — False, 사유는 errors.
        # '벤더 미설치'·'확장 없음'·'확인 못 함' 경고는 내지 않는다(메모 라우트 플래그와 같은 판정).
        for status in (401, 403):
            with self.subTest(status=status):
                bodies = live_bodies()
                bodies[EP_LORA_CONFIG] = status
                caps, _ = build(bodies)
                self.assertTrue(caps.known and caps.installed)
                self.assertFalse(caps.lora_manager)
                self.assertFalse(caps.may_use("lora_manager"))
                self.assertEqual(caps.errors[EP_LORA_CONFIG], f"HTTP {status}")
                for code in ("lora_manager_unavailable", "route_unverified", "extension_missing", "script_missing"):
                    self.assertNotIn(code, _codes(caps))

    def test_lora_config_404_is_absence(self):
        bodies = live_bodies()
        bodies[EP_LORA_CONFIG] = 404          # LoRA 라우트가 없는 옛 확장
        caps, _ = build(bodies)
        self.assertTrue(caps.installed)
        self.assertFalse(caps.lora_manager)
        self.assertNotIn(EP_LORA_CONFIG, caps.errors)
        self.assertNotIn("lora_manager_unavailable", _codes(caps))
        self.assertNotIn("route_unverified", _codes(caps))

    def test_lora_config_probe_failure_is_not_absence(self):
        # 라우트 플래그와 같은 규칙: 확인 실패(타임아웃·연결 끊김·5xx·408·429)는 설치가 확인됐으면 있다고 본다
        for failure in (TimeoutError(), ConnectionError(), 500, 503, 408, 429):
            with self.subTest(failure=failure):
                bodies = live_bodies()
                bodies[EP_LORA_CONFIG] = failure
                caps, _ = build(bodies)
                self.assertTrue(caps.lora_manager and caps.may_use("lora_manager"))
                self.assertIn(EP_LORA_CONFIG, caps.errors)          # 실패 사유는 그대로 남는다
                unverified = [w for w in caps.warnings if w["code"] == "route_unverified"]
                self.assertEqual([w["feature"] for w in unverified], ["lora_manager"])
                self.assertIn("LoRA Manager", unverified[0]["message"])
                self.assertNotIn("lora_manager_unavailable", _codes(caps))
        # 스크립트 목록이 sam-extra 가 없다고 확정했으면 타임아웃도 '없음'이다
        bodies = self._without_sam_extra_scripts(live_bodies())
        bodies[EP_LORA_CONFIG] = TimeoutError()
        caps, _ = build(bodies)
        self.assertFalse(caps.installed or caps.lora_manager)
        self.assertNotIn("route_unverified", _codes(caps))

    @staticmethod
    def _without_sam_extra_scripts(bodies):
        bodies[EP_SCRIPTS] = {"txt2img": ["controlnet", "seed"], "img2img": ["controlnet"]}
        bodies[EP_SCRIPT_INFO] = [s for s in bodies[EP_SCRIPT_INFO] if s["name"] not in caps_mod.SAM_EXTRA_TITLES]
        return bodies

    def test_disabled_extension_entry(self):
        bodies = self._without_sam_extra_scripts(live_bodies())
        entry = next(e for e in bodies[EP_EXTENSIONS] if e["name"] == "forge_sam3_extension")
        entry["enabled"] = False
        entry["name"] = "sam-extra"
        caps, _ = build(bodies)
        self.assertIn("extension_disabled", _codes(caps))
        self.assertNotIn("extension_missing", _codes(caps))
        self.assertEqual(caps.version["folder"], "sam-extra")

    def test_disabled_entry_is_ignored_while_scripts_are_live(self):
        # 확장 목록은 오래 캐시한다 — 스크립트가 보이면 그 목록의 '꺼짐' 으로 경고하지 않는다
        bodies = live_bodies()
        next(e for e in bodies[EP_EXTENSIONS] if e["name"] == "forge_sam3_extension")["enabled"] = False
        caps, _ = build(bodies)
        self.assertTrue(caps.sam3 and caps.installed)
        self.assertNotIn("extension_disabled", _codes(caps))
        self.assertNotIn("script_missing", _codes(caps))

    def test_duplicate_folders_prefer_enabled_entry(self):
        # 수동 클론(꺼짐)이 이름순으로 앞에, 앱이 건 sam-extra 연결(켜짐)이 뒤에 온다
        bodies = live_bodies()
        clone = next(e for e in bodies[EP_EXTENSIONS] if e["name"] == "forge_sam3_extension")
        old = dict(clone, enabled=False, version="0badc0de")
        linked = dict(clone, name="sam-extra", enabled=True)
        bodies[EP_EXTENSIONS] = [old] + [e for e in bodies[EP_EXTENSIONS] if e is not clone] + [linked]
        caps, _ = build(bodies)
        self.assertNotIn("extension_disabled", _codes(caps))
        self.assertEqual((caps.version["folder"], caps.version["enabled"]), ("sam-extra", True))
        self.assertEqual(caps.version["commit"], clone["version"])
        # 둘 다 꺼져 있고 스크립트도 없으면 그때만 '꺼짐'
        bodies = self._without_sam_extra_scripts(bodies)
        bodies[EP_EXTENSIONS][-1]["enabled"] = False
        caps, _ = build(bodies)
        self.assertIn("extension_disabled", _codes(caps))
        self.assertEqual(caps.version["folder"], "forge_sam3_extension")

    def test_remote_credentials_are_not_exposed(self):
        bodies = live_bodies()
        entry = next(e for e in bodies[EP_EXTENSIONS] if e["name"] == "forge_sam3_extension")
        entry["name"] = "my-sam"   # 이름이 달라도 remote 로 찾는다 — 찾는 데는 원래 주소를 쓴다
        entry["remote"] = "https://someone:ghp_SECRET123@github.com:443/UR-MJ/sam-extra.git"
        caps, _ = build(bodies)
        self.assertEqual(caps.version["folder"], "my-sam")
        self.assertEqual(caps.version["remote"], "https://github.com:443/UR-MJ/sam-extra.git")
        text = json.dumps(caps.to_dict(), ensure_ascii=False)
        self.assertNotIn("SECRET", text)
        self.assertNotIn("someone", text)

    def test_public_remote_shapes(self):
        public = caps_mod._public_remote
        self.assertEqual(public("https://github.com/UR-MJ/sam-extra"), "https://github.com/UR-MJ/sam-extra")
        self.assertEqual(public("https://oauth2:tok@gitlab.example/x/sam-extra.git?private_token=t#f"),
                         "https://gitlab.example/x/sam-extra.git")
        self.assertEqual(public("ssh://git@github.com/UR-MJ/sam-extra.git"),
                         "ssh://github.com/UR-MJ/sam-extra.git")
        self.assertEqual(public("git@github.com:UR-MJ/sam-extra.git"), "github.com:UR-MJ/sam-extra.git")
        for local in ("C:\\sd\\extensions\\sam-extra", "C:/sd/sam-extra", "/home/u/sam-extra",
                      "file:///C:/sd/sam-extra", "https://u:p@host:notaport/x", "", None, 3):
            self.assertIsNone(public(local), local)

    def test_injected_get_exception_becomes_result(self):
        def boom(url, **_kw):
            raise RuntimeError("x")
        caps = probe.fetch_capabilities(BASE, http_get=boom)
        self.assertEqual(caps.status, "unreachable")

    def test_build_with_no_responses(self):
        self.assertEqual(build_capabilities({}).status, "unreachable")

    def test_unknown_capabilities_shape(self):
        data = unknown_capabilities("not_applicable").to_dict()
        self.assertEqual(data["status"], "not_applicable")
        self.assertFalse(data["known"] or data["installed"])
        self.assertEqual(data["warnings"], [])


class BoundedReadTests(unittest.TestCase):
    def test_declared_and_streamed_limits(self):
        class Response:
            def __init__(self, length, chunks):
                self.headers = {"Content-Length": str(length)}
                self._chunks = chunks

            def iter_content(self, chunk_size):
                return iter(self._chunks)

        with self.assertRaises(probe._TooLarge):
            probe._read_bounded(Response(probe.MAX_BODY_BYTES + 1, []))
        with mock.patch.object(probe, "MAX_BODY_BYTES", 4):
            with self.assertRaises(probe._TooLarge):
                probe._read_bounded(Response(0, [b"abc", b"de"]))
            self.assertEqual(probe._read_bounded(Response(0, [b"ab", b"", b"c"])), b"abc")


class CacheTests(unittest.TestCase):
    def setUp(self):
        self.now = 1000.0
        self.fetches = []

        def fetch(url):
            self.fetches.append(url)
            return SamExtraCapabilities(status="ok", checked_at=str(len(self.fetches)))
        self.cache = probe.CapabilityCache(fetch, ttl=60, failed_ttl=5, clock=lambda: self.now)

    def test_ttl_and_refresh(self):
        first = self.cache.get(BASE + "/")
        self.assertIs(self.cache.get(BASE), first)            # 끝 '/' 는 같은 키
        self.now += 59
        self.assertIs(self.cache.get(BASE), first)
        self.now += 2
        self.assertIsNot(self.cache.get(BASE), first)         # TTL 지남
        self.assertEqual(len(self.fetches), 2)
        self.cache.get(BASE, refresh=True)
        self.assertEqual(len(self.fetches), 3)
        self.assertEqual(self.cache.peek(BASE).checked_at, "3")
        self.assertIsNone(self.cache.peek("http://other:7860"))

    def test_failed_snapshot_uses_short_ttl(self):
        cache = probe.CapabilityCache(lambda url: unknown_capabilities("unreachable"),
                                      ttl=60, failed_ttl=5, clock=lambda: self.now)
        first = cache.get(BASE)
        self.now += 4
        self.assertIs(cache.get(BASE), first)
        self.now += 2
        self.assertIsNot(cache.get(BASE), first)

    def test_concurrent_requests_share_one_fetch(self):
        gate = threading.Event()
        started = threading.Event()
        calls = []

        def slow(url):
            calls.append(url)
            started.set()
            gate.wait(5)
            return SamExtraCapabilities(status="ok")
        cache = probe.CapabilityCache(slow, ttl=60)
        results = []
        workers = [threading.Thread(target=lambda: results.append(cache.get(BASE))) for _ in range(4)]
        workers[0].start()
        started.wait(5)
        for worker in workers[1:]:
            worker.start()
        gate.set()
        for worker in workers:
            worker.join(5)
        self.assertEqual(len(calls), 1)
        self.assertEqual(len(results), 4)
        self.assertTrue(all(r is results[0] for r in results))

    def test_invalidate_drops_in_flight_result(self):
        gate = threading.Event()
        started = threading.Event()

        def slow(url):
            started.set()
            gate.wait(5)
            return SamExtraCapabilities(status="ok")
        cache = probe.CapabilityCache(slow, ttl=60)
        worker = threading.Thread(target=lambda: cache.get(BASE))
        worker.start()
        started.wait(5)
        cache.invalidate(BASE)
        gate.set()
        worker.join(5)
        self.assertIsNone(cache.peek(BASE))


class ExtensionsCacheTests(unittest.TestCase):
    """/sdapi/v1/extensions 는 Forge 가 확장 목록을 비우고 다시 스캔하는 GET — 연결마다 보내지 않는다."""

    def setUp(self):
        self.now = 1000.0
        self.cache = probe.ExtensionsCache(ttl=600, clock=lambda: self.now)

    def _count(self, fake):
        return fake.paths().count(EP_EXTENSIONS)

    def test_forced_probes_reuse_the_extensions_list(self):
        fake = FakeGet(live_bodies())
        for _ in range(3):
            caps = probe.fetch_capabilities(BASE, http_get=fake, extensions_cache=self.cache)
            self.assertEqual(caps.version["folder"], "forge_sam3_extension")
        self.assertEqual(self._count(fake), 1)
        self.assertEqual(fake.paths().count(EP_SCRIPTS), 3)          # 나머지는 매번 묻는다
        self.now += 601                                              # TTL 지남
        probe.fetch_capabilities(BASE, http_get=fake, extensions_cache=self.cache)
        self.assertEqual(self._count(fake), 2)
        self.cache.invalidate(BASE + "/")                            # 수동 새로고침
        probe.fetch_capabilities(BASE, http_get=fake, extensions_cache=self.cache)
        self.assertEqual(self._count(fake), 3)

    def test_connection_failure_is_not_cached(self):
        bodies = live_bodies()
        bodies[EP_EXTENSIONS] = ConnectionError()
        fake = FakeGet(bodies)
        probe.fetch_capabilities(BASE, http_get=fake, extensions_cache=self.cache)
        fake.bodies = live_bodies()
        caps = probe.fetch_capabilities(BASE, http_get=fake, extensions_cache=self.cache)
        self.assertEqual(self._count(fake), 2)
        self.assertEqual(caps.version["folder"], "forge_sam3_extension")

    def test_invalidate_drops_in_flight_result(self):
        gate, started = threading.Event(), threading.Event()

        def slow():
            started.set()
            gate.wait(5)
            return HttpResult(200, [])
        worker = threading.Thread(target=lambda: self.cache.get(BASE, slow))
        worker.start()
        started.wait(5)
        self.cache.invalidate()
        gate.set()
        worker.join(5)
        self.assertEqual(self.cache.get(BASE, lambda: HttpResult(200, [{"name": "x"}])).body, [{"name": "x"}])

    def test_app_cache_shares_extensions_and_backend_change_keeps_them(self):
        fake = FakeGet(live_bodies())
        with mock.patch.object(probe, "requests_get", fake), \
                mock.patch.object(probe, "_EXTENSIONS_CACHE", probe.ExtensionsCache()), \
                mock.patch.object(probe, "_DEFAULT_CACHE", probe.CapabilityCache(probe._fetch_with_shared_extensions)):
            probe.get_capabilities(BASE, refresh=True)
            probe.invalidate_capabilities()                          # 백엔드 변경(연결마다 온다)
            probe.get_capabilities(BASE, refresh=True)
            self.assertEqual(self._count(fake), 1)
            self.assertEqual(fake.paths().count(EP_SCRIPTS), 2)
            probe.invalidate_extensions(BASE)                        # 수동 새로고침만 다시 묻는다
            probe.get_capabilities(BASE, refresh=True)
            self.assertEqual(self._count(fake), 2)


class FrontendMirrorTests(unittest.TestCase):
    """Vue 토스트의 '켜짐' 판단 표가 파이썬 활성화 키와 같아야 한다."""

    def test_guidance_activation_keys_are_mirrored(self):
        path = os.path.join(ROOT, "frontend", "src", "utils", "samExtraCapabilities.ts")
        with open(path, encoding="utf-8") as f:
            text = f.read()
        block = re.search(r"anima_guidance:\s*\[(.*?)\]", text, re.S).group(1)
        ts_keys = set(re.findall(r"'_(\w+)'", block))
        py_keys = set(anima_guidance._ACTIVATION_KEYS[anima_guidance.SCRIPT_PERTURBATION])
        self.assertEqual(ts_keys, py_keys)
        for title, feature in ((anima_guidance.SCRIPT_SKIMMED_CFG, "skimmed_cfg"),
                               (anima_guidance.SCRIPT_DETAIL_DAEMON, "detail_daemon")):
            keys = re.search(rf"{feature}:\s*\[(.*?)\]", text, re.S).group(1)
            self.assertEqual(set(re.findall(r"'_(\w+)'", keys)), set(anima_guidance._ACTIVATION_KEYS[title]))

    def test_feature_flags_match_typescript_union(self):
        with open(os.path.join(ROOT, "frontend", "src", "types", "bridge.d.ts"), encoding="utf-8") as f:
            text = f.read()
        union = re.search(r"export type SamExtraFeature =(.*?)\n\n", text, re.S).group(1)
        self.assertEqual(tuple(re.findall(r"'(\w+)'", union)), caps_mod.FEATURE_FLAGS)


class ActionsMixinTests(unittest.TestCase):
    """연결 훅·Vue 액션 → 워커 → samExtraCapabilities 이벤트."""

    def _host(self, fetch):
        from ui.sam_extra_capabilities_actions import SamExtraCapabilitiesActionsMixin

        class Signal:
            def __init__(self):
                self.sent = []

            def emit(self, text):
                self.sent.append(json.loads(text))

        class Bridge:
            samExtraCapabilities = Signal()

        class Host(SamExtraCapabilitiesActionsMixin):
            pass

        host = Host()
        host.vue_bridge = Bridge()
        host._sam_extra_fetch = fetch
        return host, host.vue_bridge.samExtraCapabilities

    def _wait(self, host):
        worker = getattr(host, "_sam_extra_worker", None)
        if worker is not None:
            worker.join(5)

    def test_refresh_stores_and_emits_snapshot(self):
        from backends import BackendType
        seen = []

        def fetch(url, force):
            seen.append((url, force))
            return probe.fetch_capabilities(BASE, http_get=FakeGet(live_bodies()))
        host, signal = self._host(fetch)
        backend = mock.Mock(api_url=BASE)
        with mock.patch("backends.get_backend_type", return_value=BackendType.WEBUI), \
                mock.patch("backends.get_backend", return_value=backend):
            self.assertTrue(host._handle_sam_extra_capabilities_action("sam_extra_capabilities_get",
                                                                        {"refresh": True}))
            self._wait(host)
        self.assertEqual(seen, [(BASE, True)])
        self.assertTrue(host.sam_extra_capabilities.installed)
        self.assertEqual(signal.sent[-1]["status"], "ok")
        self.assertTrue(signal.sent[-1]["features"]["sam3"])
        # refresh 없이 부르면 네트워크 없이 마지막 값을 다시 보낸다
        host._handle_sam_extra_capabilities_action("sam_extra_capabilities_get", {})
        self.assertEqual(len(seen), 1)
        self.assertEqual(signal.sent[-1]["checked_at"], signal.sent[-2]["checked_at"])
        self.assertFalse(host._handle_sam_extra_capabilities_action("memo_list", {}))

    def test_replay_before_connect_does_not_probe(self):
        host, signal = self._host(lambda url, force: self.fail("연결 전에는 서버를 두드리지 않는다"))
        self.assertTrue(host._handle_sam_extra_capabilities_action("sam_extra_capabilities_get",
                                                                    {"refresh": False}))
        self.assertEqual(signal.sent[-1]["status"], "unknown")

    def test_replay_after_connect_without_snapshot_uses_cache(self):
        from backends import BackendType
        seen = []

        def fetch(url, force):
            seen.append(force)
            return SamExtraCapabilities(status="ok")
        host, signal = self._host(fetch)
        host._backend_connected = True
        with mock.patch("backends.get_backend_type", return_value=BackendType.WEBUI), \
                mock.patch("backends.get_backend", return_value=mock.Mock(api_url=BASE)):
            host._handle_sam_extra_capabilities_action("sam_extra_capabilities_get", {})
            self._wait(host)
        self.assertEqual(seen, [False])
        self.assertEqual(signal.sent[-1]["status"], "ok")

    def test_comfyui_backend_is_not_applicable(self):
        from backends import BackendType
        host, signal = self._host(lambda url, force: self.fail("ComfyUI 에서 Forge 를 조회하면 안 된다"))
        with mock.patch("backends.get_backend_type", return_value=BackendType.COMFYUI):
            host._refresh_sam_extra_capabilities(force=True)
        self.assertEqual(signal.sent[-1]["status"], "not_applicable")
        self.assertFalse(signal.sent[-1]["known"])

    def test_invalidation_discards_late_result(self):
        from backends import BackendType
        gate = threading.Event()

        def fetch(url, force):
            gate.wait(5)
            return SamExtraCapabilities(status="ok", installed=True)
        host, signal = self._host(fetch)
        with mock.patch("backends.get_backend_type", return_value=BackendType.WEBUI), \
                mock.patch("backends.get_backend", return_value=mock.Mock(api_url=BASE)):
            host._refresh_sam_extra_capabilities(force=True)
            host._invalidate_sam_extra_capabilities()     # 백엔드가 바뀌었다
            gate.set()
            self._wait(host)
        self.assertIsNone(host.sam_extra_capabilities)
        self.assertEqual(signal.sent[-1]["status"], "unknown")

    def _shared_cache_host(self, fetch_impl):
        """앱과 같은 경로: 호스트 → probe.get_capabilities → (테스트용) 공용 CapabilityCache."""
        cache = probe.CapabilityCache(fetch_impl, ttl=300)
        host, signal = self._host(lambda url, force: probe.get_capabilities(url, refresh=force))
        host._backend_connected = True
        return host, signal, cache

    def test_replay_during_forced_probe_keeps_its_result(self):
        """재연결 중 붙은 웹 클라이언트의 재생 요청이 연결 훅의 강제 확인을 밀어내지 않는다."""
        from backends import BackendType
        gate = threading.Event()
        calls = []

        def fetch_impl(url):
            calls.append(url)
            if len(calls) == 1:
                return SamExtraCapabilities(status="ok", checked_at="OLD", sam3=False)
            gate.wait(5)                                   # Forge 재시작 뒤 강제 확인(느림)
            return SamExtraCapabilities(status="ok", checked_at="NEW", sam3=True)
        host, signal, cache = self._shared_cache_host(fetch_impl)
        with mock.patch.object(probe, "_DEFAULT_CACHE", cache), \
                mock.patch("backends.get_backend_type", return_value=BackendType.WEBUI), \
                mock.patch("backends.get_backend", return_value=mock.Mock(api_url=BASE)):
            host._refresh_sam_extra_capabilities(force=True)
            self._wait(host)                               # 첫 연결 → OLD
            host._invalidate_sam_extra_capabilities()      # set_backend(같은 URL) — 재연결
            self.assertIsNone(cache.peek(BASE))            # 바뀌기 전 스냅샷을 캐시에서도 버린다
            host._refresh_sam_extra_capabilities(force=True)   # 연결 훅(막혀 있음)
            forced = host._sam_extra_worker
            host._handle_sam_extra_capabilities_action("sam_extra_capabilities_get", {})
            host._handle_sam_extra_capabilities_action("sam_extra_capabilities_get", {"refresh": True})
            self.assertIs(host._sam_extra_worker, forced)  # 새 확인을 시작하지 않았다
            self.assertEqual(signal.sent[-1]["status"], "unknown")   # 확인 중 — 지금 값만 다시
            gate.set()
            forced.join(5)
        self.assertEqual(len(calls), 2)
        self.assertEqual(host.sam_extra_capabilities.checked_at, "NEW")
        self.assertTrue(host.sam_extra_capabilities.sam3)
        self.assertEqual(signal.sent[-1]["checked_at"], "NEW")

    def test_replay_after_invalidation_does_not_serve_pre_change_snapshot(self):
        from backends import BackendType
        calls = []

        def fetch_impl(url):
            calls.append(url)
            return SamExtraCapabilities(status="ok", checked_at=f"#{len(calls)}")
        host, signal, cache = self._shared_cache_host(fetch_impl)
        with mock.patch.object(probe, "_DEFAULT_CACHE", cache), \
                mock.patch("backends.get_backend_type", return_value=BackendType.WEBUI), \
                mock.patch("backends.get_backend", return_value=mock.Mock(api_url=BASE)):
            host._refresh_sam_extra_capabilities(force=True)
            self._wait(host)
            host._invalidate_sam_extra_capabilities()
            host._handle_sam_extra_capabilities_action("sam_extra_capabilities_get", {})  # 대기 중인 확인 없음
            self._wait(host)
        self.assertEqual(len(calls), 2)                    # TTL 안이어도 캐시의 옛 값을 쓰지 않는다
        self.assertEqual(signal.sent[-1]["checked_at"], "#2")

    def test_action_refresh_is_rate_limited_and_only_it_rescans_extensions(self):
        from backends import BackendType
        from ui import sam_extra_capabilities_actions as actions_mod
        seen = []
        now = [1000.0]

        def fetch(url, force):
            seen.append(force)
            return SamExtraCapabilities(status="ok", checked_at=str(len(seen)))
        host, signal = self._host(fetch)
        host._backend_connected = True
        with mock.patch("backends.get_backend_type", return_value=BackendType.WEBUI), \
                mock.patch("backends.get_backend", return_value=mock.Mock(api_url=BASE)), \
                mock.patch.object(actions_mod, "_clock", lambda: now[0]), \
                mock.patch("core.sam_extra_probe.invalidate_extensions") as rescan:
            host._refresh_sam_extra_capabilities(force=True)       # 연결 훅
            self._wait(host)
            rescan.assert_not_called()                              # 연결은 확장 목록 캐시를 쓴다
            host._handle_sam_extra_capabilities_action("sam_extra_capabilities_get", {"refresh": True})
            self._wait(host)
            rescan.assert_called_once_with(BASE)
            now[0] += actions_mod.ACTION_REFRESH_MIN_INTERVAL_S - 1
            host._handle_sam_extra_capabilities_action("sam_extra_capabilities_get", {"refresh": True})
            self._wait(host)
            self.assertEqual(seen, [True, True])                    # 간격 안 — 다시 묻지 않는다
            self.assertEqual(signal.sent[-1]["checked_at"], "2")    # 마지막 스냅샷을 다시 보냈다
            now[0] += 2
            host._handle_sam_extra_capabilities_action("sam_extra_capabilities_get", {"refresh": True})
            self._wait(host)
        self.assertEqual(seen, [True, True, True])
        self.assertEqual(rescan.call_count, 2)

    def test_fetch_exception_becomes_error_snapshot(self):
        from backends import BackendType

        def fetch(url, force):
            raise OSError("boom")
        host, signal = self._host(fetch)
        with mock.patch("backends.get_backend_type", return_value=BackendType.WEBUI), \
                mock.patch("backends.get_backend", return_value=mock.Mock(api_url=BASE)):
            host._refresh_sam_extra_capabilities(force=False)
            self._wait(host)
        self.assertEqual(signal.sent[-1]["status"], "error")
        self.assertEqual(signal.sent[-1]["errors"], {"probe": "OSError"})


if __name__ == "__main__":
    unittest.main()
