"""SAM3 ControlNet 전처리기·모델 이름 — 대소문자 정규화와 라이브 목록 선택 (P3, gap matrix S6).

Forge ControlNet 은 ``supported_preprocessors[name]`` · ``controlnet_filename_dict[name]`` 로 이름을
대소문자까지 그대로 찾는다. 예전 정적 목록의 소문자 'none' 은 KeyError 로 SAM3 패스를 조용히 실패시켰다.

픽스처 ``tests/fixtures/sam_extra_live_cn_modules.json`` · ``..._cn_models.json`` 은 2026-09-25 실행 중이던
Forge classic(sam-extra 0.30.0)의 ``/controlnet/module_list`` · ``/controlnet/model_list`` 응답이다.
네트워크·Forge·torch 없이 돈다.
"""
from __future__ import annotations

import base64
import io
import json
import os
import unittest
from unittest import mock

from core import sam3_args, sam3_cn_names as names
from core import sam3_controlnet as cn
from core.sam_extra_capabilities import (
    EP_CN_MODELS, EP_CN_MODULES, EP_SCRIPTS, EP_SCRIPT_INFO, STATUS_NOT_APPLICABLE, HttpResult,
    build_capabilities, unknown_capabilities,
)
# 확장 위치 찾기·skip/강제(AISTUDIO_REQUIRE_FORGE_EXT=1) 규칙은 공용 헬퍼 — 폴더 이름 sam-extra 도 찾는다.
from tests._sam_extra_ext import EXT_ROOT, requires_extension

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIXTURES = os.path.join(ROOT, "tests", "fixtures")


def _fixture(name):
    with open(os.path.join(FIXTURES, name), encoding="utf-8") as f:
        return json.load(f)


LIVE_MODULES = tuple(_fixture("sam_extra_live_cn_modules.json")["module_list"])
LIVE_MODELS = tuple(_fixture("sam_extra_live_cn_models.json")["model_list"])


def live_capabilities(*, cn_modules=True, cn_models=True):
    """녹화된 v0.30.0 응답으로 만든 기능 스냅샷 (CN 목록은 골라서 뺄 수 있다 — 404 = CN 확장 없음)."""
    responses = {
        EP_SCRIPTS: HttpResult(200, _fixture("sam_extra_live_scripts.json")),
        EP_SCRIPT_INFO: HttpResult(200, _fixture("sam_extra_script_info.json")["scripts"]),
        EP_CN_MODULES: HttpResult(200, {"module_list": list(LIVE_MODULES)}) if cn_modules else HttpResult(404),
        EP_CN_MODELS: HttpResult(200, {"model_list": list(LIVE_MODELS)}) if cn_models else HttpResult(404),
    }
    return build_capabilities(responses, checked_at="2026-09-25T00:00:00Z")


class StaticFallbackListTests(unittest.TestCase):
    def test_every_static_module_exists_in_the_live_forge_list_case_sensitively(self):
        missing = [m for m in sam3_args.CN_MODULES if m not in LIVE_MODULES]
        self.assertEqual(missing, [], "Forge 는 대소문자까지 그대로 찾는다 — 정적 폴백 목록이 라이브와 다르다")

    def test_none_is_capitalised(self):
        self.assertIn("None", sam3_args.CN_MODULES)
        self.assertNotIn("none", sam3_args.CN_MODULES)
        self.assertEqual(LIVE_MODULES[0], "None")
        self.assertEqual(LIVE_MODELS[0], "None")

    def test_spec_defaults_are_live_names(self):
        defaults = sam3_args.default_settings()
        self.assertIn(defaults["sam3_cn_module"], LIVE_MODULES)
        self.assertIn(defaults["sam3_cn_model"], LIVE_MODELS)
        self.assertEqual(names.DEFAULT_MODULE, defaults["sam3_cn_module"])
        self.assertEqual(names.NONE, defaults["sam3_cn_model"])


class NormalizeTests(unittest.TestCase):
    def test_module_case_and_empty(self):
        static = sam3_args.CN_MODULES
        self.assertEqual(names.normalize_module("none", static), "None")
        self.assertEqual(names.normalize_module(" NONE ", static), "None")
        self.assertEqual(names.normalize_module("none", ()), "None")        # 목록 없이도 'None'
        self.assertEqual(names.normalize_module("Inpaint_Only", static), "inpaint_only")
        self.assertEqual(names.normalize_module("", static), "inpaint_only")
        self.assertEqual(names.normalize_module(None, static), "inpaint_only")

    def test_module_outside_the_list_is_kept_not_substituted(self):
        # 정적 목록에 없는 라이브 이름은 그대로 — 다른 전처리기로 바꿔치지 않는다
        self.assertEqual(names.normalize_module("inpaint_noobai", sam3_args.CN_MODULES), "inpaint_noobai")
        self.assertEqual(names.normalize_module("INPAINT_NOOBAI", LIVE_MODULES), "inpaint_noobai")
        self.assertEqual(names.normalize_module("my_custom", LIVE_MODULES), "my_custom")

    def test_exact_match_wins_over_case_fold(self):
        self.assertEqual(names.canonical("abc", ("ABC", "abc")), "abc")
        self.assertEqual(names.canonical("Abc", ("ABC", "abc")), "ABC")
        self.assertIsNone(names.canonical("", ("None",)))

    def test_model(self):
        for raw in ("", None, "none", "NONE", " None "):
            self.assertEqual(names.normalize_model(raw, LIVE_MODELS), "None", raw)
            self.assertEqual(names.normalize_model(raw, None), "None", raw)
        self.assertEqual(names.normalize_model("Anima-LLLite-Inpainting-V2", LIVE_MODELS),
                         "anima-lllite-inpainting-v2")
        self.assertEqual(names.normalize_model("Anima-LLLite-Inpainting-V2", None),
                         "Anima-LLLite-Inpainting-V2")   # 목록을 모르면 그대로
        self.assertEqual(names.normalize_model("12345", LIVE_MODELS), "12345")

    def test_normalize_saved_only_touches_cn_names_and_keeps_empty(self):
        self.assertEqual(names.normalize_saved("cn_module", "none"), "None")
        self.assertEqual(names.normalize_saved("cn_model", "none"), "None")
        self.assertEqual(names.normalize_saved("cn_model", ""), "")          # 비운 모델 칸 보존
        self.assertEqual(names.normalize_saved("cn_module", "depth_leres"), "depth_leres")
        self.assertEqual(names.normalize_saved("cn_control_mode", "none"), "none")


class LiveListSelectionTests(unittest.TestCase):
    def test_live_snapshot_supplies_both_lists(self):
        lists = names.live_lists(live_capabilities())
        self.assertTrue(lists.known and lists.live_modules)
        self.assertEqual(lists.modules, LIVE_MODULES)
        self.assertEqual(lists.models, LIVE_MODELS)
        self.assertIn("inpaint_noobai", lists.modules)                 # 정적 목록에 없는 라이브 전처리기
        self.assertIn("anima-lllite-inpainting-v2", lists.models)      # sam3ext/ui.py 가 등록한 models/sam3 LLLite

    def test_unknown_snapshot_falls_back_to_static_modules_and_free_models(self):
        for caps in (None, unknown_capabilities(), unknown_capabilities(STATUS_NOT_APPLICABLE)):
            with self.subTest(caps=getattr(caps, "status", None)):
                lists = names.live_lists(caps)
                self.assertFalse(lists.known or lists.live_modules)
                self.assertEqual(lists.modules, tuple(sam3_args.CN_MODULES))
                self.assertIsNone(lists.models)

    def test_known_snapshot_without_controlnet_lists(self):
        lists = names.live_lists(live_capabilities(cn_modules=False, cn_models=False))
        self.assertTrue(lists.known)
        self.assertFalse(lists.live_modules)
        self.assertEqual(lists.modules, tuple(sam3_args.CN_MODULES))
        self.assertIsNone(lists.models)


class WarningTests(unittest.TestCase):
    def _state(self, **over):
        state = sam3_args.default_settings()
        state.update({"sam3_cn_enable": True, **over})
        return state

    def test_disabled_controlnet_never_warns(self):
        state = self._state(sam3_cn_enable=False, sam3_cn_module="nope")
        self.assertEqual(names.name_warnings(state, names.live_lists(None)), [])

    def test_unknown_snapshot_warns_that_the_static_list_is_used(self):
        got = []
        names.normalize_state(self._state(sam3_cn_module="none"), None, warn=got.append)
        self.assertEqual(len(got), 1)
        self.assertIn("정적", got[0])

    def test_known_snapshot_flags_names_forge_does_not_have(self):
        got = []
        state = names.normalize_state(
            self._state(sam3_cn_module="not_a_module", sam3_cn_model="not_a_model"),
            live_capabilities(), warn=got.append)
        self.assertEqual(state["sam3_cn_module"], "not_a_module")    # 바꿔치지 않는다
        self.assertEqual(len(got), 2)
        self.assertTrue(any("'not_a_module'" in m for m in got))
        self.assertTrue(any("'not_a_model'" in m for m in got))

    def test_unlisted_model_warning_does_not_claim_a_certain_failure(self):
        """모델 목록은 연결 때 것 — 확장은 SAM3 패스마다 models/sam3 를 다시 스캔해 filename dict 에 더한다.

        그래서 Forge 시작 뒤 models/sam3 에 넣은 LLLite 는 목록에 없어도 생성 때 찾는다. 전처리기는 시작 때
        정해지므로 목록 밖이면 실패가 맞다.
        """
        got = []
        names.normalize_state(self._state(sam3_cn_model="new-lllite-inpaint"), live_capabilities(),
                              warn=got.append)
        self.assertEqual(len(got), 1)
        self.assertIn("models/sam3", got[0])
        self.assertIn("연결 때", got[0])
        self.assertNotIn("목록에서 다시 고르세요", got[0])
        got.clear()
        names.normalize_state(self._state(sam3_cn_module="not_a_module"), live_capabilities(), warn=got.append)
        self.assertEqual(len(got), 1)
        self.assertIn("KeyError 로 실패합니다", got[0])

    def test_known_snapshot_with_valid_names_is_quiet(self):
        got = []
        names.normalize_state(self._state(sam3_cn_module="NONE", sam3_cn_model="Anima-LLLite-Inpainting-V2"),
                              live_capabilities(), warn=got.append)
        self.assertEqual(got, [])

    def test_missing_controlnet_extension_is_reported(self):
        got = []
        names.normalize_state(self._state(), live_capabilities(cn_modules=False, cn_models=False),
                              warn=got.append)
        self.assertEqual(len(got), 1)
        self.assertIn("sd_forge_controlnet", got[0])

    def test_default_warning_is_logged_once_per_message(self):
        with mock.patch.object(names, "_warned", set()):
            with self.assertLogs("core.sam3_cn_names", level="WARNING") as logs:
                for _ in range(3):   # 배치 SAM3 가 같은 설정으로 여러 번 부른다
                    sam3_args.build_state({"sam3_cn_enable": True, "sam3_cn_module": "none"})
            self.assertEqual(len(logs.records), 1)


class BuildStateTests(unittest.TestCase):
    """페이로드 빌더 — 세 Forge 경로(T2I apply_to_payload, Refine·배치 webui_backend)가 모두 build_state 를 탄다."""

    def test_saved_lowercase_none_is_sent_as_forge_casing_without_snapshot(self):
        with mock.patch.object(names, "_warned", set()), self.assertLogs("core.sam3_cn_names", "WARNING"):
            state = sam3_args.build_state({"sam3_cn_enable": True, "sam3_cn_module": "none",
                                           "sam3_cn_model": "none"})
        self.assertEqual(state["sam3_cn_module"], "None")
        self.assertEqual(state["sam3_cn_model"], "None")

    def test_live_snapshot_fixes_case_of_live_only_names(self):
        # 4채널 LLLite 인페인트 + inpaint_* 전처리기는 LLLite 가드가 None 으로 바꾼다(LLLiteModuleGuardTests) —
        # 대소문자 확인은 가드가 건드리지 않는 라이브 전용 전처리기(depth_leres)와 모델 없음으로 나눠 한다.
        caps = live_capabilities()
        state = sam3_args.build_state({"sam3_cn_enable": True, "sam3_cn_module": "Depth_Leres",
                                       "sam3_cn_model": "ANIMA-LLLITE-INPAINTING-V2"}, capabilities=caps)
        self.assertEqual(state["sam3_cn_module"], "depth_leres")
        self.assertEqual(state["sam3_cn_model"], "anima-lllite-inpainting-v2")
        state = sam3_args.build_state({"sam3_cn_enable": True, "sam3_cn_module": "Inpaint_NoobAI"},
                                      capabilities=caps)
        self.assertEqual(state["sam3_cn_module"], "inpaint_noobai")

    def test_disabled_controlnet_still_sends_valid_names(self):
        state = sam3_args.build_state({"sam3_cn_module": "none"})
        self.assertIs(state["sam3_cn_enable"], False)
        self.assertEqual(state["sam3_cn_module"], "None")

    def test_apply_to_payload_and_build_alwayson_pass_the_snapshot(self):
        caps = live_capabilities()
        payload = {"prompt": "p"}
        sam3_args.apply_to_payload(payload, {"sam3_cn_module": "INPAINT_NOOBAI"}, capabilities=caps)
        self.assertEqual(payload["alwayson_scripts"][sam3_args.SCRIPT_SAM3]["args"][0]["sam3_cn_module"],
                         "inpaint_noobai")
        block = sam3_args.build_alwayson({"sam3_cn_module": "INPAINT_NOOBAI"}, capabilities=caps)
        self.assertEqual(block[sam3_args.SCRIPT_SAM3]["args"][0]["sam3_cn_module"], "inpaint_noobai")

    def test_state_keys_unchanged(self):
        state = sam3_args.build_state({}, capabilities=live_capabilities())
        self.assertEqual(set(state) - {"sam3_enable", "enabled"}, set(sam3_args.SAM3_KEYS))


# (이름, 채널, Tile & Repair) — 확장 tests/test_sam3_cn_lllite.py NameTests.CASES ·
# frontend/src/utils/sam3ControlNet.test.ts 와 같은 표다.
LLLITE_NAME_CASES = (
    ("animaTileRepair_v20", 3, True),
    ("animaTileRepair_v10", 3, True),
    ("anima_tile-repair_v3", 3, True),
    ("anima_tiled_lllite_v1", 3, True),
    ("anima_lllite_lineart_v1", 3, False),
    ("anima-lllite-canny", 3, False),
    ("Anima_LLLite_Depth", 3, False),
    ("anima-lllite-inpainting-v2", 4, False),
    ("Anima-LLLite-Inpainting-V2", 4, False),
    ("new-lllite-inpaint", None, False),
    ("kohya_controllllite_xl_inpaint", None, False),
    ("kohya_controllllite_xl_canny_anime", None, False),
    ("controlnet_tile_sdxl", None, False),
    ("TileRepair_sdxl", None, False),
    ("my_lineart_cn", None, False),
    ("None", None, False),
    ("", None, False),
    (None, None, False),
)
# 표의 모든 이름에 대해 세 곳이 같은 결과를 내는지 볼 전처리기
GUARD_MODULES = ("inpaint_only", "inpaint_noobai", "inpaint_global_harmonious", "tile_resample",
                 "lineart_anime", "canny", "None")


def _extension_file(*parts):
    return os.path.join(str(EXT_ROOT), *parts) if EXT_ROOT is not None else ""


def _extension_lllite_module():
    """설치된 확장의 순수 모듈 sam3ext/sam3_cn_lllite.py (stdlib 만 쓴다)."""
    import importlib.util
    path = _extension_file("sam3ext", "sam3_cn_lllite.py")
    spec = importlib.util.spec_from_file_location("_sam_extra_sam3_cn_lllite", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _injection_calls():
    """확장 inpaint_core.inject_controlnet_unit 이 부르는 이름들(AST — Forge 없이)."""
    import ast
    with open(_extension_file("sam3ext", "inpaint_core.py"), encoding="utf-8") as f:
        tree = ast.parse(f.read())
    func = next((node for node in ast.walk(tree)
                 if isinstance(node, ast.FunctionDef) and node.name == "inject_controlnet_unit"), None)
    assert func is not None, "inject_controlnet_unit 없음"
    return {node.func.id for node in ast.walk(func)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)}


class LLLiteModuleGuardTests(unittest.TestCase):
    """Anima ControlNet-LLLite 전처리기 가드 (plan §7.2 #7·#14, TR-SAM3).

    원본은 LLLite 에 사용자가 준 제어 이미지를 그대로 준다:
      origin: kohya-ss/sd-scripts@690ea7f9:anima_minimal_inference_control_net_lllite.py:65-74
        img = Image.open(path).convert("RGB") → resize(BICUBIC) → arr / 127.5 - 1.0  (전처리 없음)
      origin: kohya-ss/ComfyUI-Anima-LLLite@b7495bd8:nodes.py:168-183
        cond_in_channels = int(meta.get("lllite.cond_in_channels", 3)); 4ch 는 MASK 필수, 3ch 는 MASK 를 버린다
      origin: kohya-ss/sd-scripts@690ea7f9:docs/anima_train_control_net_lllite.md:355 ('"3" standard, "4" inpaint'),
        :77 (conditioning images e.g. lineart, canny, depth), anima_minimal_inference_control_net_lllite.py:23,32
        (--control_image canny.png) — 3채널은 Tile & Repair 전용이 아니다.
    SAM3 CN 유닛은 인페인트 입력 이미지를 cond 로 쓴다 — Tile & Repair 는 그 그림 자체가 제어 이미지라 전처리기
    None, lineart·canny·depth LLLite 는 전처리기가 그 맵을 만드는 유일한 단계라 그대로, inpaint_* 는 원본에 없는
    마스크 굽기라 None 이다.
    """

    def _state(self, **over):
        state = sam3_args.default_settings()
        state.update({"sam3_cn_enable": True, **over})
        return state

    def test_name_table(self):
        for name, want, tile in LLLITE_NAME_CASES:
            with self.subTest(name=name):
                self.assertEqual(names.lllite_channels_from_name(name), want)
                self.assertIs(names.lllite_tile_repair_from_name(name), tile)

    def test_default_inpaint_only_with_tile_repair_is_sent_as_none(self):
        got = []
        state = self._state(sam3_cn_model="animaTileRepair_v20")
        self.assertEqual(state["sam3_cn_module"], "inpaint_only")          # SAM3_SPEC 기본값 그대로 들어온다
        names.guard_lllite_module(state, warn=got.append)
        self.assertEqual(state["sam3_cn_module"], "None")
        self.assertEqual(len(got), 1)
        self.assertIn("Tile & Repair", got[0])
        self.assertIn("'inpaint_only'", got[0])

    def test_any_preprocessor_with_tile_repair_becomes_none(self):
        for module in ("tile_resample", "canny", "inpaint_global_harmonious", "inpaint_only", "none"):
            with self.subTest(module=module):
                self.assertEqual(names.lllite_module_override(module, "anima_tiled_lllite_v1")[0], "None")
                self.assertEqual(names.lllite_module_override(module, "animaTileRepair_v20")[0], "None")
        self.assertEqual(names.lllite_module_override("None", "animaTileRepair_v20"), ("None", None))

    def test_standard_three_channel_lllite_keeps_lineart_canny_depth(self):
        """3채널 lineart·canny Anima LLLite — 전처리기가 인페인트 이미지를 제어 맵으로 바꾸는 유일한 단계다."""
        for module, model in (("lineart_anime", "anima_lllite_lineart_v1"), ("canny", "anima-lllite-canny"),
                              ("depth_anything_v2", "Anima_LLLite_Depth"), ("tile_resample", "anima-lllite-canny")):
            with self.subTest(model=model):
                self.assertEqual(names.lllite_module_override(module, model), (module, None))
        got = []
        state = names.guard_lllite_module(
            self._state(sam3_cn_module="lineart_anime", sam3_cn_model="anima_lllite_lineart_v1"), warn=got.append)
        self.assertEqual((state["sam3_cn_module"], got), ("lineart_anime", []))

    def test_standard_three_channel_lllite_drops_inpaint_preprocessors(self):
        """원본 3채널은 마스크를 버린다 — inpaint_* 는 마스크 영역을 제어 이미지에서 비우므로 None."""
        got = []
        state = names.guard_lllite_module(self._state(sam3_cn_model="anima_lllite_lineart_v1"), warn=got.append)
        self.assertEqual(state["sam3_cn_module"], "None")           # 기본값 inpaint_only
        self.assertEqual(len(got), 1)
        self.assertIn("3채널 Anima LLLite", got[0])
        self.assertIn("마스크", got[0])
        for module in ("inpaint_global_harmonious", "inpaint_only+lama", "inpaint_noobai"):
            self.assertEqual(names.lllite_module_override(module, "anima-lllite-canny")[0], "None")

    def test_sdxl_controllllite_inpaint_name_is_left_to_the_extension(self):
        """'controllllite' 에 'lllite' 가 들어 있어도 Anima 가 아니다 — 확장이 헤더로 보고 그대로 두는 것을
        앱이 먼저 바꾸면 안 된다(확장 test_non_anima_lllite_header_wins_over_a_misleading_name)."""
        for module in ("inpaint_only", "inpaint_global_harmonious", "tile_resample"):
            self.assertEqual(names.lllite_module_override(module, "kohya_controllllite_xl_inpaint"), (module, None))
        with mock.patch.object(names, "_warned", set()), \
                self.assertLogs("core.sam3_cn_names", level="WARNING") as logs:
            state = sam3_args.build_state({"sam3_cn_enable": True, "sam3_cn_module": "inpaint_only",
                                           "sam3_cn_model": "kohya_controllllite_xl_inpaint"},
                                          capabilities=live_capabilities())
        self.assertEqual(state["sam3_cn_module"], "inpaint_only")
        self.assertEqual(len(logs.output), 1)                      # 목록 밖 모델 경고 하나뿐 — 가드 경고 없음
        self.assertIn("연결 때 받은 Forge ControlNet 목록에 없습니다", logs.output[0])

    def test_four_channel_inpaint_lllite_only_drops_inpaint_preprocessors(self):
        module, message = names.lllite_module_override("inpaint_only", "anima-lllite-inpainting-v2")
        self.assertEqual(module, "None")
        self.assertIn("4채널", message)
        self.assertEqual(names.lllite_module_override("depth_leres", "anima-lllite-inpainting-v2"),
                         ("depth_leres", None))
        self.assertEqual(names.lllite_module_override("None", "anima-lllite-inpainting-v2"), ("None", None))

    def test_other_models_and_disabled_controlnet_are_untouched(self):
        got = []
        state = names.guard_lllite_module(self._state(sam3_cn_model="None"), warn=got.append)
        self.assertEqual((state["sam3_cn_module"], got), ("inpaint_only", []))
        state = names.guard_lllite_module(
            self._state(sam3_cn_enable=False, sam3_cn_model="animaTileRepair_v20"), warn=got.append)
        self.assertEqual((state["sam3_cn_module"], got), ("inpaint_only", []))

    def test_build_state_applies_the_guard_after_name_normalisation(self):
        caps = live_capabilities()
        with mock.patch.object(names, "_warned", set()), \
                self.assertLogs("core.sam3_cn_names", level="WARNING") as logs:
            state = sam3_args.build_state({"sam3_cn_enable": True, "sam3_cn_module": "INPAINT_ONLY",
                                           "sam3_cn_model": "animatilerepair_v20"}, capabilities=caps)
        self.assertEqual(state["sam3_cn_model"], "animaTileRepair_v20")    # 대소문자는 라이브 목록 표기로
        self.assertEqual(state["sam3_cn_module"], "None")
        self.assertTrue(any("Tile & Repair" in line for line in logs.output))
        # T2I 저장값('inpaint_only' + Tile & Repair)도 보낼 때 고쳐진다
        payload = {"prompt": "p"}
        with mock.patch.object(names, "_warned", set()), self.assertLogs("core.sam3_cn_names", "WARNING"):
            sam3_args.apply_to_payload(payload, {"sam3_cn_enable": True, "sam3_cn_model": "animaTileRepair_v10"},
                                       capabilities=caps)
        self.assertEqual(payload["alwayson_scripts"][sam3_args.SCRIPT_SAM3]["args"][0]["sam3_cn_module"], "None")

    def test_live_models_are_classified(self):
        self.assertEqual({m: names.lllite_channels_from_name(m) for m in LIVE_MODELS},
                         {"None": None, "anima-lllite-inpainting-v2": 4,
                          "animaTileRepair_v10": 3, "animaTileRepair_v20": 3})
        self.assertEqual([m for m in LIVE_MODELS if names.lllite_tile_repair_from_name(m)],
                         ["animaTileRepair_v10", "animaTileRepair_v20"])


@requires_extension
class ExtensionSourceTests(unittest.TestCase):
    """앱 규칙·문구의 전제를 설치된 확장 소스로 확인한다 (확장이 없으면 skip, AISTUDIO_REQUIRE_FORGE_EXT=1 이면 실패).

    앱 이름 규칙 = 확장의 이름 폴백·전처리기 가드, 확장이 SAM3 패스마다 models/sam3 를 다시 스캔한다.
    """

    def test_name_rules_match_the_installed_extension(self):
        """확장이 헤더를 못 읽을 때의 이름 폴백과 앱 규칙이 같은지."""
        ext = _extension_lllite_module()
        for name, _want, _tile in LLLITE_NAME_CASES:
            with self.subTest(name=name):
                self.assertEqual(names.lllite_channels_from_name(name), ext.lllite_channels_from_name(name))
                self.assertIs(names.lllite_tile_repair_from_name(name), ext.lllite_tile_repair_from_name(name))
                info = ext.anima_lllite(name)
                for module in GUARD_MODULES:
                    self.assertEqual(names.lllite_module_override(module, name)[0],
                                     ext.forced_cn_module(module, info, name)[0], module)

    def test_extension_injection_uses_the_guard(self):
        """앱 경고 문구의 전제 — 확장 inject_controlnet_unit 이 헤더/이름으로 전처리기를 고친다."""
        self.assertTrue({"forced_cn_module", "anima_lllite"} <= _injection_calls())

    def test_extension_rescans_models_sam3_at_injection(self):
        """목록 밖 모델 경고 문구(test_unlisted_model_warning_does_not_claim_a_certain_failure)의 전제 —
        확장 소스가 바뀌면(재스캔을 없애면) 이 테스트가 알린다."""
        self.assertIn("_scan_sam3_dir_for_cn_models", _injection_calls())


def _png_b64() -> str:
    from PIL import Image
    buf = io.BytesIO()
    Image.new("RGB", (8, 8)).save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


class WebuiBackendPathTests(unittest.TestCase):
    """I2I SAM3 정밀화(refine)·배치 SAM3(sam3) — 연결 때 받아 둔 스냅샷(peek, 네트워크 없음)으로 맞춘다."""

    def setUp(self):
        from backends.webui_backend import WebUIBackend
        self.backend = WebUIBackend.__new__(WebUIBackend)
        self.backend.api_url = "http://127.0.0.1:7860"
        self.sent = []
        self.backend._run_img2img_postprocess = lambda image, payload: self.sent.append(payload) or image

    def _state(self):
        return self.sent[-1]["alwayson_scripts"]["SAM3 Mask"]["args"][0]

    def test_snapshot_comes_from_the_probe_cache_only(self):
        caps = live_capabilities()
        with mock.patch("core.sam_extra_probe.peek_capabilities", return_value=caps) as peek:
            self.assertIs(self.backend._sam_extra_snapshot(), caps)
        peek.assert_called_once_with("http://127.0.0.1:7860")

    def test_batch_sam3_and_refine_use_live_names(self):
        caps = live_capabilities()
        # 전처리기는 LLLite 가드가 건드리지 않는 라이브 전용 이름(4채널 LLLite + inpaint_* 는 None 이 된다)
        settings = {"sam3_cn_enable": True, "sam3_cn_module": "DEPTH_LERES",
                    "sam3_cn_model": "Anima-LLLite-Inpainting-V2", "target": "face"}
        with mock.patch("core.sam_extra_probe.peek_capabilities", return_value=caps), \
                mock.patch("core.forge_modules.resolve_sam3_checkpoint", side_effect=lambda name: name):
            self.backend.sam3(_png_b64(), dict(settings))
            self.assertEqual(self._state()["sam3_cn_module"], "depth_leres")
            self.assertEqual(self._state()["sam3_cn_model"], "anima-lllite-inpainting-v2")
            self.backend.refine(_png_b64(), dict(settings))
            self.assertEqual(self._state()["sam3_cn_module"], "depth_leres")

    def test_not_probed_yet_uses_static_casing(self):
        with mock.patch("core.sam_extra_probe.peek_capabilities", return_value=None), \
                mock.patch("core.forge_modules.resolve_sam3_checkpoint", side_effect=lambda name: name), \
                mock.patch.object(names, "_warned", set()), \
                self.assertLogs("core.sam3_cn_names", "WARNING"):
            self.backend.sam3(_png_b64(), {"sam3_cn_enable": True, "sam3_cn_module": "none"})
        self.assertEqual(self._state()["sam3_cn_module"], "None")


class T2IPathTests(unittest.TestCase):
    """T2I SAM3 Mask 카드 — _apply_postprocess_chain 이 메인 창의 스냅샷을 넘긴다."""

    def test_postprocess_chain_uses_the_window_snapshot(self):
        from ui.generator_generation import GenerationMixin

        class _Proxy:
            def __init__(self, value="", checked=False):
                self.value, self.checked = value, checked

            def text(self):
                return self.value

            def toPlainText(self):
                return self.value

            def currentText(self):
                return self.value

            def isChecked(self):
                return self.checked

        class _Widgets(dict):
            def __missing__(self, key):
                self[key] = proxy = _Proxy()
                return proxy

        class _Host(GenerationMixin):
            pass

        host = _Host()
        host.adetailer_group = _Proxy(checked=False)
        host.sam3_group = _Proxy(checked=True)
        host.anima_guidance_widgets = {}
        host.sam3_widgets = _Widgets({
            "cn_enable": _Proxy(checked=True),
            "cn_module": _Proxy("NONE"),
            "cn_model": _Proxy("Anima-LLLite-Inpainting-V2"),
        })
        host.sam_extra_capabilities = live_capabilities()
        payload = {"prompt": "p", "negative_prompt": ""}
        with mock.patch("core.forge_modules.resolve_sam3_checkpoint", side_effect=lambda name: name):
            host._apply_postprocess_chain(payload)
        state = payload["alwayson_scripts"]["SAM3 Mask"]["args"][0]
        self.assertEqual(state["sam3_cn_module"], "None")
        self.assertEqual(state["sam3_cn_model"], "anima-lllite-inpainting-v2")


class SettingsLoadTests(unittest.TestCase):
    """저장된 소문자 값은 불러올 때 Forge 표기로 (설정은 연결 전에 불러오므로 정적 기준)."""

    class _Text:
        def __init__(self):
            self.value = ""

        def text(self):
            return self.value

        def setText(self, value):
            self.value = value

    class _Check:
        def __init__(self):
            self.checked = False

        def isChecked(self):
            return self.checked

        def setChecked(self, value):
            self.checked = bool(value)

    def _widgets(self):
        return {key: (self._Check() if kind == "check" else self._Text()) for key, kind in cn.CN_FIELDS}

    def test_lowercase_none_is_restored_as_forge_casing(self):
        widgets = self._widgets()
        cn.apply_settings(widgets, {"cn_module": "none", "cn_model": "NONE"})
        self.assertEqual(widgets["cn_module"].text(), "None")
        self.assertEqual(widgets["cn_model"].text(), "None")

    def test_other_saved_names_and_empty_model_are_kept(self):
        widgets = self._widgets()
        cn.apply_settings(widgets, {"cn_module": "inpaint_noobai", "cn_model": ""})
        self.assertEqual(widgets["cn_module"].text(), "inpaint_noobai")
        self.assertEqual(widgets["cn_model"].text(), "")


class ComfyParityTests(unittest.TestCase):
    def test_comfy_node_accepts_forge_casing(self):
        """Comfy SAM3 노드는 전처리기 이름을 casefold 한다 — 'None' 으로 바꿔도 Comfy 결과는 같다."""
        from comfy_custom_nodes.ai_studio_forge_parity import sam3_nodes
        self.assertEqual(sam3_nodes._control_module_name("None"), sam3_nodes._control_module_name("none"))
        self.assertEqual(sam3_nodes._control_module_name("None"), "none")


if __name__ == "__main__":
    unittest.main()
