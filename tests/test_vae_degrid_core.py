"""VAE DeGrid(sam-extra Anima VAE DeGrid (NAFNet)) — core/vae_degrid 순수 로직.

상수는 녹화된 script-info 픽스처(tests/fixtures/sam_extra_script_info.json — 확장 395854b 의 t2i·i2i)와 대조한다.
확장 상수와의 일치(설치된 소스)는 레지스트리 SEMANTIC_PINS 가 지킨다. Qt·Forge·torch 없이 돈다.
"""
import json
import math
import os
import unittest
from dataclasses import replace
from types import SimpleNamespace

from core import vae_degrid as vdg
from core.sam_extra_capabilities import SamExtraCapabilities, _freeze

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIXTURE = os.path.join(ROOT, "tests", "fixtures", "sam_extra_script_info.json")
LIVE_MODELS = ("qwenVAEDegridNafnet_v11", "NAFNet-QwenVAE-DeGrid")


def _fixture_entries():
    with open(FIXTURE, encoding="utf-8") as fh:
        scripts = json.load(fh)["scripts"]
    return [s for s in scripts if s.get("name") == vdg.SCRIPT_NAME.lower()]


def known_caps(models=LIVE_MODELS):
    choices = {} if models is None else {"degrid_models": tuple(models)}
    return SamExtraCapabilities(status="ok", installed=True, degrid=True, choices=_freeze(choices))


class FixtureTests(unittest.TestCase):
    """녹화된 script-info(확장 ui())의 인자 모양·범위·기본값이 앱 상수와 같다."""

    def test_both_tabs_have_the_five_args_with_the_app_ranges_and_defaults(self):
        entries = _fixture_entries()
        self.assertEqual(sorted(e["is_img2img"] for e in entries), [False, True])
        for entry in entries:
            with self.subTest(img2img=entry["is_img2img"]):
                enabled, model, mode, strength, tile = entry["args"]
                self.assertEqual(len(entry["args"]), len(vdg.ARG_NAMES))
                self.assertEqual(enabled["label"], vdg.SCRIPT_NAME)      # 아코디언 라벨 = 제목
                self.assertIs(enabled["value"], vdg.EXTENSION_DEFAULTS.enabled)
                self.assertEqual(tuple(model["choices"]), LIVE_MODELS)
                self.assertEqual(model["value"], model["choices"][0])   # 기본 = 첫 파일 = 자동
                self.assertEqual(tuple(mode["choices"]), vdg.MODE_CHOICES)
                self.assertEqual(vdg.normalize_mode(mode["value"]), vdg.EXTENSION_DEFAULTS.mode)
                self.assertEqual((strength["minimum"], strength["maximum"], strength["step"], strength["value"]),
                                 (vdg.STRENGTH_MIN, vdg.STRENGTH_MAX, vdg.STRENGTH_STEP, vdg.DEFAULT_STRENGTH))
                self.assertEqual((tile["minimum"], tile["maximum"], tile["step"], tile["value"]),
                                 (0, vdg.MAX_TILE, vdg.TILE_STEP, vdg.DEFAULT_TILE))

    def test_every_live_mode_label_normalizes_to_a_distinct_key_in_extension_order(self):
        labels = _fixture_entries()[0]["args"][2]["choices"]
        self.assertEqual(tuple(vdg.normalize_mode(label) for label in labels), vdg.MODES)

    def test_app_defaults_equal_the_extension_defaults(self):
        self.assertEqual(vdg.APP_DEFAULTS, vdg.DegridSettings(False, "", "full", 1.0, 512, False))
        self.assertIs(vdg.APP_DEFAULTS, vdg.EXTENSION_DEFAULTS)

    def test_widget_ids_settings_key_and_comfy_constants(self):
        self.assertEqual([vdg.widget_id(k) for k in vdg.WIDGET_KEYS],
                         ["_degrid_enabled", "_degrid_model", "_degrid_mode", "_degrid_strength", "_degrid_tile",
                          "_degrid_apply_img2img"])
        with self.assertRaises(KeyError):
            vdg.widget_id("device")
        self.assertEqual(vdg.SETTINGS_KEY, "vae_degrid_settings")
        self.assertEqual(dict(vdg.COMFY_OPTIONS), {"device": "auto", "precision": "fp32", "keep_loaded": False})
        self.assertEqual(vdg.COMFY_INPUTS[:2], ("image", "enabled"))
        self.assertTrue(vdg.COMFY_FORGE_QUANTIZE)
        self.assertEqual(tuple(vdg.MODE_LABELS), vdg.MODES)
        self.assertEqual(set(vdg.EXTENSION_OPTION_DEFAULTS.values()) - {False},
                         {vdg.DEFAULT_DEVICE, vdg.DEFAULT_PRECISION})


class NormalizeTests(unittest.TestCase):
    def test_modes_accept_keys_names_ui_labels_and_aliases_case_insensitively(self):
        cases = {
            "full": "full", "FULL": "full", " Full ": "full", "Full (전체)": "full",
            "dark": "dark", "Dark Pixels Mainly": "dark", "dark pixels mainly": "dark", "dark_pixels": "dark",
            "Dark Pixels": "dark", "Dark Pixels Mainly (어두운 점 위주)": "dark",
            "bright": "bright", "BRIGHT PIXELS MAINLY": "bright", "bright_pixels": "bright",
            "Bright Pixels Mainly (밝은 점 위주)": "bright",
        }
        for value, key in cases.items():
            with self.subTest(value=value):
                self.assertEqual(vdg.normalize_mode(value), key)
                self.assertEqual(vdg.coerce_mode(value), key)
        for value in (None, "", "  ", "medium", "(전체)", 3):
            with self.subTest(value=value):
                self.assertIsNone(vdg.normalize_mode(value))
                self.assertEqual(vdg.coerce_mode(value), "full")    # 확장 coerce_args 처럼 Full

    def test_tile_follows_the_extension_rule(self):
        for value, expected in ((0, 0), (-5, 0), (1, 128), (127, 128), (128, 128), (300, 300), (4096, 4096),
                                (5000, 4096), ("512", 512), ("256.9", 256), ("x", 512), (None, 512),
                                (float("inf"), 512), (float("nan"), 512)):
            with self.subTest(value=value):
                self.assertEqual(vdg.coerce_tile(value), expected)

    def test_strength_is_clamped_and_nan_falls_back(self):
        for value, expected in ((-1, 0.0), (0, 0.0), (0.85, 0.85), (1.5, 1.5), (2, 1.5), ("0.5", 0.5),
                                (float("nan"), 1.0), ("x", 1.0), (None, 1.0), (float("inf"), 1.5)):
            with self.subTest(value=value):
                self.assertEqual(vdg.coerce_strength(value), expected)

    def test_model_auto_spellings_and_strength_format(self):
        for value in (None, "", "  ", "None", "none", "auto", "AUTO"):
            self.assertEqual(vdg.normalize_model(value), "")
        self.assertEqual(vdg.normalize_model(" qwenVAEDegridNafnet_v11 "), "qwenVAEDegridNafnet_v11")
        self.assertEqual([vdg.format_strength(v) for v in (1.0, 0.85, 0.5, 0, 1.2345)], ["1", "0.85", "0.5", "0",
                                                                                          "1.234"])


class SettingsTests(unittest.TestCase):
    def test_widget_values_round_trip(self):
        settings = vdg.DegridSettings(True, "NAFNet-QwenVAE-DeGrid", "dark", 0.8, 0, True)
        values = vdg.widget_values(settings)
        self.assertEqual(values, {"enabled": "true", "model": "NAFNet-QwenVAE-DeGrid", "mode": "dark",
                                  "strength": "0.8", "tile": "0", "apply_img2img": "true"})
        self.assertEqual(vdg.parse_settings(values), settings)
        self.assertEqual(vdg.widget_values(vdg.APP_DEFAULTS),
                         {"enabled": "false", "model": "", "mode": "full", "strength": "1", "tile": "512",
                          "apply_img2img": "false"})

    def test_missing_or_unreadable_cells_keep_the_base(self):
        self.assertEqual(vdg.parse_settings(None), vdg.APP_DEFAULTS)
        self.assertEqual(vdg.parse_settings({"mode": "medium", "strength": "x", "tile": "", "enabled": ""}),
                         vdg.APP_DEFAULTS)
        base = vdg.DegridSettings(True, "m", "bright", 0.5, 256, True)
        self.assertEqual(vdg.parse_settings({"strength": "nan", "tile": "abc"}, base=base), base)
        self.assertEqual(vdg.parse_settings({"model": "None"}, base=base).model, "")    # 자동은 값이다
        self.assertEqual(vdg.parse_settings({"tile": "64", "strength": "9"}),
                         replace(vdg.APP_DEFAULTS, tile=128, strength=1.5))
        self.assertEqual(vdg.parse_settings({"mode": "Dark Pixels Mainly (어두운 점 위주)"}).mode, "dark")

    def test_block_is_a_single_dict_and_reads_back(self):
        settings = vdg.DegridSettings(True, "", "bright", 1.25, 1024, True)
        block = vdg.as_block(settings)
        self.assertEqual(block, {"args": [{"enabled": True, "model": "", "mode": "bright", "strength": 1.25,
                                           "tile": 1024}]})
        self.assertEqual(vdg.parse_script_block(block), replace(settings, apply_img2img=False))
        json.dumps(block)   # JSON 으로 나간다

    def test_parse_script_block_reads_positional_and_dict_like_the_extension(self):
        positional = {"args": [True, "None", "Dark Pixels Mainly (어두운 점 위주)", "0.7", 100]}
        self.assertEqual(vdg.parse_script_block(positional), vdg.DegridSettings(True, "", "dark", 0.7, 128))
        short = {"args": [True, "v11"]}                                    # 뒤 인자는 기본값
        self.assertEqual(vdg.parse_script_block(short), vdg.DegridSettings(True, "v11", "full", 1.0, 512))
        loose = {"args": [{"enabled": "true", "model": "auto", "mode": "weird", "strength": "nan", "tile": -1}]}
        self.assertEqual(vdg.parse_script_block(loose), vdg.DegridSettings(True, "", "full", 1.0, 0))
        self.assertEqual(vdg.parse_script_block({"args": []}), vdg.DegridSettings())
        for bad in (None, [], "x"):
            self.assertIsNone(vdg.parse_script_block(bad))

    def test_from_infotext_mirrors_the_extension_paste(self):
        success = {vdg.KEY_MODEL: "qwenVAEDegridNafnet_v11", vdg.KEY_MODE: "Dark Pixels Mainly",
                   vdg.KEY_STRENGTH: 0.8, vdg.KEY_TILE: 256, vdg.KEY_PRECISION: "fp16-autocast"}
        pasted = vdg.from_infotext(success)
        self.assertEqual(pasted, vdg.DegridSettings(True, "qwenVAEDegridNafnet_v11", "dark", 0.8, 256, False))
        # 정밀도는 설정이라 붙여 넣지 않는다(결과 같음) — 기본값 칸은 base 그대로
        self.assertEqual(vdg.from_infotext({vdg.KEY_MODEL: "m"}, base=replace(vdg.APP_DEFAULTS, apply_img2img=True)),
                         vdg.DegridSettings(True, "m", "full", 1.0, 512, True))
        # 실패한 이미지는 오류 키만 — 꺼짐으로 붙이고 오류는 보여 주기만 한다
        error = {vdg.KEY_ERROR: '"not a DeGrid residual model: x, y output does not look like a residual"'}
        self.assertEqual(vdg.from_infotext(error), vdg.APP_DEFAULTS)
        self.assertEqual(vdg.infotext_error(error),
                         "not a DeGrid residual model: x, y output does not look like a residual")
        self.assertEqual(vdg.infotext_error({vdg.KEY_ERROR: "model not found: auto"}), "model not found: auto")
        self.assertEqual(vdg.infotext_error({}), "")
        self.assertEqual(vdg.from_infotext({"Steps": 20}, base=replace(vdg.APP_DEFAULTS, enabled=True)).enabled, False)
        self.assertEqual(vdg.from_infotext({vdg.KEY_MODEL: "m", vdg.KEY_TILE: "9999", vdg.KEY_MODE: "?"}).tile, 4096)

    def test_describe_matches_the_card_summary_wording(self):
        self.assertEqual(vdg.describe(replace(vdg.APP_DEFAULTS, enabled=True)), "Full 1 · 512")
        self.assertEqual(vdg.describe(vdg.DegridSettings(True, "v11", "dark", 0.8, 0)), "Dark 0.8 · 타일 없음 · v11")


class ModelListTests(unittest.TestCase):
    def test_live_models_drop_the_none_placeholder_and_unknown_is_none(self):
        self.assertEqual(vdg.live_models(known_caps()), list(LIVE_MODELS))
        self.assertEqual(vdg.live_models(known_caps(("None",))), [])
        self.assertIsNone(vdg.live_models(known_caps(None)))                       # 목록을 못 읽었다
        for unknown in (None, SamExtraCapabilities(), SimpleNamespace(known=True, choices=None)):
            self.assertIsNone(vdg.live_models(unknown))

    def test_check_model_and_resolution_mirror_the_extension(self):
        cases = ((known_caps(), "", vdg.MODEL_OK), (known_caps(), "qwenvaedegridnafnet_v11", vdg.MODEL_OK),
                 (known_caps(), "qwenVAEDegridNafnet_v11.safetensors", vdg.MODEL_OK),
                 (known_caps(), "gone", vdg.MODEL_MISSING), (known_caps(("None",)), "", vdg.MODEL_NONE_INSTALLED),
                 (None, "gone", vdg.MODEL_UNKNOWN))
        for capabilities, model, expected in cases:
            with self.subTest(model=model, caps=capabilities):
                settings = replace(vdg.APP_DEFAULTS, enabled=True, model=model)
                self.assertEqual(vdg.check_model(settings, capabilities), expected)
        names = ["ESRGAN/x", "DeGrid/x", "y"]
        self.assertEqual(vdg.resolve_name("", names), "ESRGAN/x")                   # 자동 = 첫 이름
        self.assertEqual(vdg.resolve_name("degrid/X", names), "DeGrid/x")
        self.assertEqual(vdg.resolve_name("x", names), "ESRGAN/x")                  # 파일 이름으로
        self.assertIsNone(vdg.resolve_name("z", names))
        self.assertIsNone(vdg.resolve_name("", ["None"]))


def _object_info(choices, *, inputs=vdg.COMFY_INPUTS, new_combo=False):
    required = {name: ["BOOLEAN", {}] for name in inputs}
    if "model_name" in required:
        required["model_name"] = ["COMBO", {"options": list(choices)}] if new_combo else [list(choices), {}]
    return {vdg.COMFY_NODE_CLASS: {"input": {"required": required}}, "SaveImage": {}}


class ComfyTests(unittest.TestCase):
    CHOICES = ["auto", "qwenVAEDegridNafnet_v11.safetensors", "upscale_models/NAFNet-QwenVAE-DeGrid.safetensors",
               "degrid/NAFNet-QwenVAE-DeGrid.safetensors", "sub\\deep.pth"]

    def test_model_choices_map_comfy_files_to_forge_names(self):
        for new_combo in (False, True):
            with self.subTest(new_combo=new_combo):
                self.assertEqual(vdg.comfy_model_choices(_object_info(self.CHOICES, new_combo=new_combo)),
                                 ["qwenVAEDegridNafnet_v11", "ESRGAN/NAFNet-QwenVAE-DeGrid",
                                  "DeGrid/NAFNet-QwenVAE-DeGrid", "deep"])
        self.assertEqual(vdg.comfy_model_choices(_object_info(["auto"])), [])

    def test_absent_class_or_other_input_contract_is_none(self):
        self.assertIsNone(vdg.comfy_model_choices({"SaveImage": {}}))
        self.assertIsNone(vdg.comfy_model_choices(None))
        old_pack = _object_info(self.CHOICES, inputs=vdg.COMFY_INPUTS[:-1])        # forge_quantize 없는 옛 노드
        self.assertIsNone(vdg.comfy_model_choices(old_pack))
        extra = _object_info(self.CHOICES, inputs=(*vdg.COMFY_INPUTS, "gamma"))
        self.assertIsNone(vdg.comfy_model_choices(extra))

    def test_comfy_file_for_resolves_stems_casing_and_folder_prefixes(self):
        cases = {"": "auto", "auto": "auto", "qwenVAEDegridNafnet_v11": "qwenVAEDegridNafnet_v11.safetensors",
                 "QWENVAEDEGRIDNAFNET_V11": "qwenVAEDegridNafnet_v11.safetensors",
                 "ESRGAN/NAFNet-QwenVAE-DeGrid": "upscale_models/NAFNet-QwenVAE-DeGrid.safetensors",
                 "DeGrid/NAFNet-QwenVAE-DeGrid": "degrid/NAFNet-QwenVAE-DeGrid.safetensors",
                 "NAFNet-QwenVAE-DeGrid": "upscale_models/NAFNet-QwenVAE-DeGrid.safetensors",
                 "deep": "sub\\deep.pth", "sub\\deep.pth": "sub\\deep.pth"}
        for stem, expected in cases.items():
            with self.subTest(stem=stem):
                self.assertEqual(vdg.comfy_file_for(stem, self.CHOICES), expected)
        self.assertIsNone(vdg.comfy_file_for("gone", self.CHOICES))
        self.assertIsNone(vdg.comfy_file_for("x", ["auto"]))

    def test_report_params_use_the_forge_infotext_keys(self):
        ok = {"status": "ok", "model": "v11", "mode": "Dark Pixels Mainly", "strength": 0.8, "tile": 256,
              "precision": "fp32", "error": ""}
        self.assertEqual(vdg.report_params(ok), {vdg.KEY_MODEL: "v11", vdg.KEY_MODE: "Dark Pixels Mainly",
                                                 vdg.KEY_STRENGTH: "0.8", vdg.KEY_TILE: 256,
                                                 vdg.KEY_PRECISION: "fp32"})
        zero = dict(ok, status="skipped", strength=0.0, precision="-")              # 강도 0 = 기록하되 정밀도 없음
        self.assertEqual(vdg.report_params(zero), {vdg.KEY_MODEL: "v11", vdg.KEY_MODE: "Dark Pixels Mainly",
                                                   vdg.KEY_STRENGTH: "0", vdg.KEY_TILE: 256})
        skipped = dict(ok, status="skipped", error="output blew up (mean |residual| 180.0/255 > 100/255)")
        self.assertEqual(vdg.report_params(skipped), {vdg.KEY_ERROR: skipped["error"]})
        self.assertEqual(vdg.report_params({"status": "off"}), {})
        self.assertEqual(vdg.report_params(None), {})


class PlanTests(unittest.TestCase):
    ON = replace(vdg.APP_DEFAULTS, enabled=True)

    def test_matrix(self):
        for target in (vdg.TARGET_T2I, vdg.TARGET_I2I, vdg.TARGET_AUX):
            for backend in (vdg.BACKEND_WEBUI, vdg.BACKEND_COMFY, vdg.BACKEND_KREA2):
                for enabled in (False, True):
                    for apply_img2img in (False, True):
                        with self.subTest(target=target, backend=backend, enabled=enabled, i2i=apply_img2img):
                            settings = replace(self.ON, enabled=enabled, apply_img2img=apply_img2img)
                            result = vdg.plan(settings, target=target, backend=backend, capabilities=known_caps())
                            if not enabled:
                                reason = "off"
                            elif backend == vdg.BACKEND_KREA2:
                                reason = "krea2"
                            elif target == vdg.TARGET_I2I and not apply_img2img:
                                reason = "i2i_off"
                            else:
                                reason = "send"
                            self.assertEqual(result.reason, reason)
                            self.assertEqual(result.block is not None, reason == "send")
                            self.assertIsNone(result.notice)
                            self.assertEqual(result.provenance, "user")
                            if result.block is not None:
                                self.assertEqual(result.block, vdg.as_block(settings))

    def test_aux_gets_exactly_the_t2i_block_without_notice(self):
        """(T15) 봉투 = 메인 체인. i2i 토글과 무관하게 t2i 값."""
        t2i = vdg.plan(self.ON, target=vdg.TARGET_T2I, backend=vdg.BACKEND_WEBUI)
        aux = vdg.plan(self.ON, target=vdg.TARGET_AUX, backend=vdg.BACKEND_WEBUI)
        self.assertEqual(aux, t2i)

    def test_a_model_missing_from_the_startup_list_is_still_sent_and_only_logged(self):
        settings = replace(self.ON, model="new_file")
        with self.assertLogs("core.vae_degrid", "INFO") as logs:
            result = vdg.plan(settings, target=vdg.TARGET_T2I, backend=vdg.BACKEND_WEBUI, capabilities=known_caps())
        self.assertEqual(result.reason, "send")
        self.assertEqual(result.block["args"][0]["model"], "new_file")
        self.assertIn("new_file", logs.output[0])


class InvariantTests(unittest.TestCase):
    def test_constants_are_json_plain_and_ranges_are_consistent(self):
        self.assertTrue(vdg.MIN_TILE < vdg.DEFAULT_TILE < vdg.MAX_TILE)
        self.assertEqual(vdg.MAX_TILE % vdg.TILE_STEP, 0)
        self.assertTrue(math.isclose(round((vdg.STRENGTH_MAX - vdg.STRENGTH_MIN) / vdg.STRENGTH_STEP), 30))
        self.assertEqual(len(vdg.MODE_CHOICES), len(vdg.MODES))
        for name in vdg.__all__:
            self.assertTrue(hasattr(vdg, name), name)


if __name__ == "__main__":
    unittest.main()
