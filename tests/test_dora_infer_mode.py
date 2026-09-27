"""DoRA 추론 방식(P8) — core/dora_infer_mode 판정·정규화, 샘플링 블록 기여자, 저장·첫 로드 안내, 결과 알림, Vue 거울.

GPU·Forge 없이: 기능 스냅샷은 가짜(``tests.test_alwayson_propagation.caps``), Forge 요청은 가짜 requests.post.
"""
import copy
import json
import re
import unittest
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from backends import BackendType
from core import alwayson_propagation as ap
from core import dora_infer_mode as dim
from core import sam_extra_contract as reg
from core import sam_extra_notices as sn
from tests.test_alwayson_propagation import A38, DD, DORA, PAG, SKIM, caps
from tests.test_sampling_blocks import ALL_PRESENT, GUIDANCE_ON, ChainHost, _webui
from ui import boot_notices
from ui import dora_infer_mode_ui as dui
from ui import sampling_blocks as sb

REPO = Path(__file__).resolve().parent.parent
TS = REPO / "frontend" / "src" / "utils" / "doraMode.ts"
CARD = REPO / "frontend" / "src" / "components" / "params" / "DoraModeCard.vue"
LYCORIS_BLOCK = {"args": [{"enabled": True, "mode": "lycoris", "inserted": "keep", "weak_strength": 0.12,
                           "weak_scope": "attn"}]}
# 사용자가 바꾼 값(Forge fp32) — 앱 기본값이 아니므로 결과 알림 대상
USER_FP32_BLOCK = {"args": [dict(LYCORIS_BLOCK["args"][0], mode="forge_fp32")]}


class _Proxy:
    def __init__(self, value=""):
        self.value = value

    def text(self):
        return self.value

    def setText(self, value):
        self.value = value


def dora_widgets(**values):
    base = dim.widget_values(dim.APP_DEFAULTS)
    base.update({k: str(v) for k, v in values.items()})
    return {key: _Proxy(base[key]) for key in dim.WIDGET_KEYS}


class DoraHost(ChainHost):
    """실제 메인 체인 + DoRA 카드 위젯."""

    def __init__(self, capabilities=None, guidance=None, **dora):
        super().__init__(guidance, capabilities)
        self.dora_widgets = dora_widgets(**dora)


def only(present):
    return caps(present=present)


# ── 1. 상수·레지스트리 ─────────────────────────────────────────────────────────
class ConstantsTests(unittest.TestCase):
    def test_title_and_arg_names_match_the_registry(self):
        entry = reg.SCRIPTS[dim.SCRIPT_NAME]
        self.assertEqual(entry["status"], reg.MAPPED)
        self.assertEqual(dim.ARG_NAMES, entry["arg_names"])
        self.assertEqual(entry["app_title"], "core.dora_infer_mode:SCRIPT_NAME")
        self.assertIs(ap.TITLE_DORA, dim.SCRIPT_NAME)

    def test_app_defaults_follow_the_forge_ui_config_txt2img(self):
        # ui-config.json:5669(켬) :5671(LyCORIS) :5673(그대로 복제) :5681(0.12) :5686(어텐션만), img2img 꺼짐 :5675
        self.assertEqual(dim.APP_DEFAULTS, dim.DoraSettings(True, "lycoris", "keep", 0.12, "attn", False))
        self.assertEqual(dim.EXTENSION_DEFAULTS, dim.DoraSettings(False, "lycoris", "keep", 0.12, "attn", False))
        self.assertTrue(dim.EXTENSION_DEFAULTS.is_stock)
        self.assertFalse(dim.APP_DEFAULTS.is_stock)

    def test_widget_ids_and_settings_key(self):
        self.assertEqual([dim.widget_id(k) for k in dim.WIDGET_KEYS],
                         ["_dora_enabled", "_dora_mode", "_dora_inserted", "_dora_weak_strength",
                          "_dora_weak_scope", "_dora_apply_img2img"])
        with self.assertRaises(KeyError):
            dim.widget_id("bogus")
        from core.generation_presets import GENERATION_KEYS, PRESET_KEYS
        self.assertIn(dim.SETTINGS_KEY, GENERATION_KEYS)
        self.assertIn(dim.SETTINGS_KEY, PRESET_KEYS)


# ── 2. 정규화 ─────────────────────────────────────────────────────────────────
class NormalizeTests(unittest.TestCase):
    def test_labels_normalize_to_distinct_keys_in_extension_order(self):
        for options, normalize in ((dim.MODE_OPTIONS, dim.normalize_mode), (dim.INSERT_OPTIONS, dim.normalize_insert),
                                   (dim.SCOPE_OPTIONS, dim.normalize_scope)):
            self.assertEqual([normalize(label) for _k, label in options], [key for key, _l in options])

    def test_aliases_mirror_the_extension(self):
        cases = {
            dim.normalize_mode: {"stock": "forge", "off": "forge", "comfy": "forge", "Forge/Comfy": "forge",
                                 "forge fp32": "forge_fp32", "Forge fp32": "forge_fp32", "LyCORIS": "lycoris",
                                 "no magnitude": "no_magnitude", "No magnitude": "no_magnitude",
                                 "DoRA off (no magnitude)": "no_magnitude", "Forge (stock)": "forge",
                                 "LyCORIS fp32": "lycoris", "": None, "mystery": None, None: None},
            dim.normalize_insert: {"weak 0.12 attn": "weak", "additive": "additive", "skip": "skip",
                                   "그대로": "keep", "mystery": None, "": None},
            dim.normalize_scope: {"어텐션+MLP": "attn_mlp", "attn_mlp": "attn_mlp", "전체": "all", "all": "all",
                                  "어텐션": "attn", "attn": "attn", "mystery": None},
        }
        for normalize, table in cases.items():
            for value, expected in table.items():
                with self.subTest(normalize=normalize.__name__, value=value):
                    self.assertEqual(normalize(value), expected)

    def test_strength_is_clamped_to_the_ui_range(self):
        self.assertEqual(dim.clamp_strength(1.5), 1.0)
        self.assertEqual(dim.clamp_strength(-1), 0.0)
        self.assertEqual(dim.clamp_strength("abc"), 0.12)
        self.assertEqual(dim.clamp_strength("0.08"), 0.08)
        self.assertEqual(dim.clamp_strength(float("nan")), 0.12)
        self.assertEqual(dim.format_strength(1.0), "1")
        self.assertEqual(dim.format_strength(0.12), "0.12")


# ── 3·4. as_arg · plan ────────────────────────────────────────────────────────
class PlanTests(unittest.TestCase):
    def plan(self, settings=dim.APP_DEFAULTS, **kw):
        kw.setdefault("target", "t2i")
        kw.setdefault("backend", "webui")
        return dim.plan(settings, **kw)

    def test_stock_sends_nothing(self):
        for settings in (dim.EXTENSION_DEFAULTS, dim.DoraSettings(True, "forge", "keep")):
            result = self.plan(settings)
            self.assertIsNone(result.block)
            self.assertIsNone(result.notice)
            self.assertEqual(result.reason, "stock")

    def test_lycoris_block_is_a_five_key_dict_with_canonical_values(self):
        result = self.plan()
        self.assertEqual(result.block, LYCORIS_BLOCK)
        self.assertEqual(result.provenance, "app_default")
        self.assertEqual(list(result.block["args"][0]), list(dim.ARG_NAMES))

    def test_forge_with_weak_copy_is_not_stock(self):
        settings = dim.DoraSettings(True, "forge", "weak", 0.08, "attn_mlp")
        result = self.plan(settings)
        self.assertEqual(result.block["args"][0]["mode"], "forge")   # mode 는 늘 싣는다
        self.assertEqual(result.block["args"][0]["inserted"], "weak")
        self.assertEqual(result.provenance, "user")

    def test_targets_i2i_follows_the_toggle_and_aux_ignores_it(self):
        self.assertIsNone(self.plan(target="i2i").block)
        self.assertIsNone(self.plan(target="i2i").notice)
        on = dim.DoraSettings(True, "lycoris", "keep", apply_img2img=True)
        self.assertIsNotNone(self.plan(on, target="i2i").block)
        self.assertEqual(self.plan(target="aux").block, LYCORIS_BLOCK)

    def test_krea2_is_silent(self):
        result = self.plan(backend="krea2", has_loras=True)
        self.assertEqual((result.block, result.notice, result.reason), (None, None, "krea2"))

    def test_comfy_never_builds_and_notes_only_user_values_with_loras(self):
        user = dim.DoraSettings(True, "forge_fp32", "keep")
        noted = self.plan(user, backend="comfyui", has_loras=True)
        self.assertIsNone(noted.block)
        self.assertEqual((noted.notice.code, noted.notice.level), (sn.CODE_DORA_COMFY_STOCK, sn.LEVEL_INFO))
        self.assertIsNone(self.plan(user, backend="comfyui", has_loras=False).notice)
        # (A6) 앱 기본값은 Comfy 사용자에게 반복 알림하지 않는다 — 로그만
        with self.assertLogs("core.dora_infer_mode", "INFO"):
            quiet = self.plan(backend="comfyui", has_loras=True)
        self.assertIsNone(quiet.notice)

    def test_live_choices_missing_the_value_skip_with_a_warning_for_user_values(self):
        live = SimpleNamespace(choices={"dora_modes": ("Forge/Comfy (순정)",), "dora_insert_policies": (),
                                        "dora_weak_scopes": ()})
        user = dim.DoraSettings(True, "lycoris", "keep", 0.2)
        result = self.plan(user, capabilities=live)
        self.assertIsNone(result.block)
        self.assertEqual((result.notice.code, result.notice.level), (sn.CODE_DORA_CHOICE, sn.LEVEL_WARNING))
        with self.assertLogs("core.dora_infer_mode", "INFO"):
            quiet = self.plan(capabilities=live)
        self.assertEqual((quiet.block, quiet.notice), (None, None))

    def test_weak_copy_scope_missing_from_the_live_choices_is_reported(self):
        """약한 복사면 범위도 라이브 선택지로 확인한다(확장은 모르는 범위를 조용히 '어텐션만' 으로 읽는다)."""
        live = SimpleNamespace(choices={"dora_modes": dim.MODE_LABELS, "dora_insert_policies": dim.INSERT_LABELS,
                                        "dora_weak_scopes": (dim.SCOPE_LABELS[0],)})   # 어텐션만
        user = dim.DoraSettings(True, "lycoris", "weak", 0.2, "attn_mlp")
        self.assertIn("약한 복사 범위", dim.live_choice_problem(live, user))
        result = self.plan(user, capabilities=live)
        self.assertIsNone(result.block)
        self.assertEqual((result.notice.code, result.notice.level), (sn.CODE_DORA_CHOICE, sn.LEVEL_WARNING))
        self.assertIn("어텐션+MLP", result.notice.message)
        # 고른 범위가 있으면 보내고, 약한 복사가 아니면 범위는 보지 않는다
        self.assertIsNotNone(self.plan(replace(user, weak_scope="attn"), capabilities=live).block)
        self.assertIsNotNone(self.plan(replace(user, inserted="keep"), capabilities=live).block)

    def test_new_live_labels_still_send_and_are_reported_by_the_snapshot(self):
        labels = (*dim.MODE_LABELS, "Quantum (new)")
        live = SimpleNamespace(choices={"dora_modes": labels})
        self.assertEqual(self.plan(capabilities=live).block, LYCORIS_BLOCK)
        self.assertEqual(dim.unknown_live_choices({"dora_modes": labels}), {"dora_modes": ("Quantum (new)",)})
        self.assertIsNone(dim.live_choice_problem(None, dim.APP_DEFAULTS))


# ── 6. 설정 왕복·infotext ──────────────────────────────────────────────────────
class SettingsTests(unittest.TestCase):
    def test_widget_values_round_trip(self):
        for settings in (dim.APP_DEFAULTS, dim.DoraSettings(True, "no_magnitude", "weak", 0.3, "all", True)):
            self.assertEqual(dim.parse_settings(dim.widget_values(settings)), settings)

    def test_missing_or_unknown_cells_fall_back_to_app_defaults(self):
        self.assertEqual(dim.parse_settings(None), dim.APP_DEFAULTS)
        self.assertEqual(dim.parse_settings({}), dim.APP_DEFAULTS)
        parsed = dim.parse_settings({"enabled": "false", "mode": "mystery", "weak_strength": "9", "weak_scope": ""})
        self.assertEqual(parsed, dim.DoraSettings(False, "lycoris", "keep", 1.0, "attn", False))
        # 라벨·Vue 숫자 입력도 받는다
        self.assertEqual(dim.parse_settings({"mode": dim.MODE_LABELS[1], "weak_strength": 0.5}).mode, "forge_fp32")

    def test_from_infotext_mirrors_the_extension_paste(self):
        self.assertFalse(dim.from_infotext({}).enabled)
        skip = dim.from_infotext({"DoRA inserted": "skip"})
        self.assertEqual((skip.enabled, skip.mode, skip.inserted), (True, "forge", "skip"))
        weak = dim.from_infotext({"DoRA mode": "LyCORIS", "DoRA inserted": "weak 0.08 attn_mlp"})
        self.assertEqual((weak.mode, weak.inserted, weak.weak_strength, weak.weak_scope),
                         ("lycoris", "weak", 0.08, "attn_mlp"))
        self.assertEqual(dim.from_infotext({"DoRA mode": "Forge fp32"}).mode, "forge_fp32")

    def test_parse_script_block_reads_like_the_extension(self):
        self.assertIsNone(dim.parse_script_block(None))
        positional = dim.parse_script_block({"args": [True, "LyCORIS (학습과 동일 · fp32)", "약한 복사", 1.5, "전체"]})
        self.assertEqual(positional, dim.DoraSettings(True, "lycoris", "weak", 1.5, "all"))   # API 는 0-2
        bare = dim.parse_script_block({"args": [{"enabled": "on"}]})
        self.assertEqual((bare.enabled, bare.mode, bare.inserted), (True, "lycoris", "keep"))  # 빠진 mode = LyCORIS


# ── 7. 샘플링 블록 기여자 (메인 체인·보조 패스) ─────────────────────────────────
class ContributorTests(unittest.TestCase):
    def setUp(self):
        _webui(self)

    def test_t2i_known_present_sends_the_app_default_block_after_guidance(self):
        result = sb.build_sampling_blocks(DoraHost(ALL_PRESENT), ap.TARGET_T2I)
        self.assertEqual(list(result.blocks), [ap.TITLE_NEGPIP, PAG, SKIM, DD, DORA])
        self.assertEqual(result.blocks[DORA], LYCORIS_BLOCK)
        self.assertEqual(result.provenance[DORA], ap.PROVENANCE_APP_DEFAULT)
        self.assertEqual(result.notices, [])

    def test_unknown_snapshot_skips_quietly_for_app_defaults_and_defers_user_values(self):
        """(A2) 모르면 늘 보내지 않는다. (A6) 앱 기본값은 알림 없이 로그만, 사용자 값은 '확인 전' 정보 알림."""
        with self.assertLogs("core.alwayson_propagation", "INFO") as logs:
            quiet = sb.build_sampling_blocks(DoraHost(None), ap.TARGET_T2I)
        self.assertNotIn(DORA, quiet.blocks)
        self.assertEqual(quiet.notices, [])
        self.assertTrue(any("DoRA" in line for line in logs.output))
        user = sb.build_sampling_blocks(DoraHost(None, mode="forge_fp32"), ap.TARGET_T2I)
        self.assertNotIn(DORA, user.blocks)
        self.assertEqual([(n.code, n.level) for n in user.notices], [(sn.CODE_BLOCK_DEFERRED, sn.LEVEL_INFO)])

    def test_missing_script_warns_only_for_user_values(self):
        present = (PAG, SKIM, DD, A38)
        quiet = sb.build_sampling_blocks(DoraHost(only(present)), ap.TARGET_T2I)
        self.assertNotIn(DORA, quiet.blocks)
        self.assertEqual(quiet.notices, [])
        user = sb.build_sampling_blocks(DoraHost(only(present), inserted="skip"), ap.TARGET_T2I)
        self.assertEqual([(n.code, n.level) for n in user.notices], [(sn.CODE_BLOCK_NOT_SENT, sn.LEVEL_WARNING)])

    def test_i2i_follows_the_toggle_and_uses_the_img2img_list(self):
        caps_i2i = caps(present=(PAG, SKIM, DD, A38, DORA), img2img=(DORA,))
        off = sb.build_sampling_blocks(DoraHost(caps_i2i), ap.TARGET_I2I)
        self.assertNotIn(DORA, off.blocks)
        on = sb.build_sampling_blocks(DoraHost(caps_i2i, apply_img2img="true"), ap.TARGET_I2I)
        self.assertEqual(on.blocks[DORA]["args"][0]["mode"], "lycoris")
        self.assertEqual(on.provenance[DORA], ap.PROVENANCE_USER)   # 토글을 켠 것은 사용자 값
        gone = sb.build_sampling_blocks(DoraHost(caps(present=(DORA,), img2img=()), apply_img2img="true"),
                                        ap.TARGET_I2I)
        self.assertNotIn(DORA, gone.blocks)

    def test_aux_gets_the_t2i_value_regardless_of_the_i2i_toggle_and_is_gated_by_the_worker(self):
        result = sb.build_sampling_blocks(DoraHost(None), ap.TARGET_AUX)
        self.assertEqual(result.blocks[DORA], LYCORIS_BLOCK)
        self.assertEqual(result.provenance[DORA], ap.PROVENANCE_APP_DEFAULT)

    def test_comfy_and_krea2_build_no_block(self):
        with mock.patch("backends.get_backend_type", return_value=BackendType.COMFYUI):
            comfy = sb.build_sampling_blocks(DoraHost(None, mode="forge_fp32"), ap.TARGET_T2I)
        self.assertNotIn(DORA, comfy.blocks)
        self.assertEqual([n.code for n in comfy.notices], [sn.CODE_DORA_COMFY_STOCK])   # 스택에 켜진 LoRA
        krea2 = sb.build_sampling_blocks(DoraHost(ALL_PRESENT), ap.TARGET_T2I, krea2=True)
        self.assertNotIn(DORA, krea2.blocks)
        self.assertEqual(krea2.notices, [])

    def test_comfy_notes_user_values_only_when_the_request_has_loras(self):
        """Comfy 순정 안내는 켜진 스택 항목(이름 있는) 또는 프롬프트의 <lora:>/<lyco:> 태그가 있을 때만(_has_loras)."""
        from tests.test_chat_generation_snapshot import ReadOnlyWidget

        def comfy_notices(entries, prompt):
            host = DoraHost(None, mode="forge_fp32")                  # 사용자가 바꾼 값
            host._vue_lora_entries = entries
            host.total_prompt_display = ReadOnlyWidget(prompt)
            with mock.patch("backends.get_backend_type", return_value=BackendType.COMFYUI):
                result = sb.build_sampling_blocks(host, ap.TARGET_T2I)
            self.assertNotIn(DORA, result.blocks)
            return [n.code for n in result.notices]

        style = {"name": "style", "weight": 0.7, "enabled": True, "triggerWords": []}
        for label, entries, prompt in (("스택 없음", [], "1girl"),
                                       ("꺼진 항목만", [dict(style, enabled=False)], "1girl"),
                                       ("이름 없는 항목", [dict(style, name="")], "1girl")):
            with self.subTest(label):
                self.assertEqual(comfy_notices(entries, prompt), [])
        for label, entries, prompt in (("켜진 항목", [style], "1girl"),
                                       ("lora 태그", [], "1girl, <lora:style:0.7>"),
                                       ("lyco 태그", [dict(style, enabled=False)], "1girl, <LyCo:style:0.7>")):
            with self.subTest(label):
                self.assertEqual(comfy_notices(entries, prompt), [sn.CODE_DORA_COMFY_STOCK])

    def test_guidance_failure_still_sends_dora_and_dora_failure_still_generates(self):
        host = DoraHost(ALL_PRESENT)
        host._build_anima_settings = mock.Mock(side_effect=RuntimeError("guidance broke"))
        with self.assertLogs("generation", "WARNING"):
            result = sb.build_sampling_blocks(host, ap.TARGET_T2I)
        self.assertIn(DORA, result.blocks)
        with mock.patch.object(dim, "plan", side_effect=RuntimeError("dora broke")), \
                self.assertLogs("generation", "WARNING"):
            payload, error = DoraHost(ALL_PRESENT)._build_generation_payload(snapshot=True)
        self.assertIsNone(error)
        self.assertNotIn(DORA, payload["alwayson_scripts"])
        self.assertIn(PAG, payload["alwayson_scripts"])

    def test_main_chain_payload_carries_the_block(self):
        payload, error = DoraHost(ALL_PRESENT)._build_generation_payload(snapshot=True)
        self.assertIsNone(error)
        self.assertEqual(payload["alwayson_scripts"][DORA], LYCORIS_BLOCK)

    def test_chat_image_edit_uses_the_i2i_rule(self):
        """(A4) 채팅 이미지 편집 = i2i — 'I2I·인페인트에도 적용' 토글을 따른다."""
        caps_all = caps(present=(PAG, SKIM, DD, A38, DORA))
        _model, edit = DoraHost(caps_all)._chat_generation_snapshot("p", target="i2i")
        self.assertNotIn(DORA, edit["alwayson_scripts"])
        _model, edit_on = DoraHost(caps_all, apply_img2img="true")._chat_generation_snapshot("p", target="i2i")
        self.assertIn(DORA, edit_on["alwayson_scripts"])
        _model, t2i = DoraHost(caps_all)._chat_generation_snapshot("p")
        self.assertIn(DORA, t2i["alwayson_scripts"])


class ImageTabEntryTests(unittest.TestCase):
    """I2I·인페인트 탭의 실제 진입점 — ``apply_alwayson_extensions``(ui/i2i_actions·ui/inpaint_actions 가 부른다)가
    ``target='i2i'`` 로 체인을 불러 'I2I·인페인트에도 적용' 토글과 img2img 목록을 따르는지. 빌더를 직접 부르는
    테스트로는 호출부가 target 을 빠뜨려도(=t2i 규칙으로 앱 기본값이 늘 실림) 잡히지 않는다."""

    CAPS = caps(present=(PAG, SKIM, DD, A38, DORA), img2img=(DORA,))

    def setUp(self):
        _webui(self)

    def test_apply_alwayson_extensions_follows_the_i2i_toggle(self):
        off = DoraHost(self.CAPS).apply_alwayson_extensions({"prompt": "x", "init_images": []})
        self.assertNotIn(DORA, off["alwayson_scripts"])                        # 기본: I2I·인페인트는 순정
        on = DoraHost(self.CAPS, apply_img2img="true").apply_alwayson_extensions({"prompt": "x", "init_images": []})
        self.assertEqual(on["alwayson_scripts"][DORA], LYCORIS_BLOCK)
        # img2img 목록으로 판정한다(txt2img 에만 있는 Forge 면 토글을 켜도 보내지 않는다)
        t2i_only = caps(present=(PAG, SKIM, DD, A38, DORA), img2img=(PAG, SKIM, DD, A38))
        gone = DoraHost(t2i_only, apply_img2img="true").apply_alwayson_extensions({"prompt": "x", "init_images": []})
        self.assertNotIn(DORA, gone["alwayson_scripts"])

    def test_i2i_and_inpaint_tabs_send_what_the_toggle_says(self):
        from tests import test_i2i_payload as i2i_tests
        from tests import test_inpaint_payload as inpaint_tests
        from ui.i2i_actions import start_vue_i2i
        from ui.inpaint_actions import start_vue_inpaint

        image = i2i_tests._data_url(i2i_tests._png(64, 64))
        for toggle, expected in (("false", None), ("true", LYCORIS_BLOCK)):
            with self.subTest(toggle=toggle):
                host = DoraHost(self.CAPS, apply_img2img=toggle)
                host.vue_bridge = i2i_tests._Bridge()
                i2i_tests._Worker.instances.clear()
                self.assertTrue(start_vue_i2i(host, {"image": image}, worker_factory=i2i_tests._Worker))
                sent = i2i_tests._Worker.instances[-1].payload["alwayson_scripts"]
                self.assertEqual(sent.get(DORA), expected)

                host = DoraHost(self.CAPS, apply_img2img=toggle)
                host.vue_bridge = i2i_tests._Bridge()
                inpaint_tests._Worker.instances.clear()
                self.assertTrue(start_vue_inpaint(host, {"image": image, "mask": inpaint_tests.MASK},
                                                  worker_factory=inpaint_tests._Worker))
                sent = inpaint_tests._Worker.instances[-1].payload["alwayson_scripts"]
                self.assertEqual(sent.get(DORA), expected)


class DefaultBehaviourTests(unittest.TestCase):
    """기본 카드(앱 기본값)가 바꾸지 않아야 하는 요청: 스냅샷을 모를 때·ComfyUI·Krea2 는 P8 전과 바이트 단위로 같다."""

    def setUp(self):
        _webui(self)

    def _payload(self, host, backend):
        with mock.patch("backends.get_backend_type", return_value=backend), \
                mock.patch("core.comfy_workflow_controls.snapshot_comfy_payload", side_effect=lambda _b, p, _m: p), \
                mock.patch("ui.comfy_workflow_actions.quality_preset_payload", return_value={}), \
                mock.patch("core.spectrum_settings.spectrum_payload_from_prefs", return_value={}), \
                mock.patch("ui.generator_generation.GenerationMixin._is_krea2_generation", return_value=False):
            payload, error = host._build_generation_payload(snapshot=True)
        self.assertIsNone(error)
        return json.dumps(payload, sort_keys=True)

    def test_unknown_snapshot_and_comfy_payloads_are_byte_identical_to_the_pre_p8_host(self):
        for label, capabilities, backend in (("unknown", None, BackendType.WEBUI),
                                             ("comfy", None, BackendType.COMFYUI),
                                             ("known without DoRA", only((PAG, SKIM, DD, A38)), BackendType.WEBUI)):
            for guidance in (GUIDANCE_ON, {"guid_enabled": "false"}):
                with self.subTest(label, guidance=guidance):
                    pre_p8 = ChainHost(guidance, capabilities)          # DoRA 위젯이 없는 호스트 = P8 전
                    host = DoraHost(capabilities, guidance)
                    self.assertEqual(self._payload(host, backend), self._payload(pre_p8, backend))
                    self.assertEqual(host.vue_bridge.showNotification.calls, [])


# ── Forge 보조 요청(가짜 requests.post) ────────────────────────────────────────
class ForgeAuxRequestTests(unittest.TestCase):
    def setUp(self):
        _webui(self)

    def _refine_request(self, host, peek_caps):
        from backends.webui_backend import WebUIBackend
        from tests.test_webui_aux_propagation import ok, png_b64
        from ui.aux_pass_snapshot import attach_sampling_snapshot

        settings = attach_sampling_snapshot(host, {"target": "face"})
        sent = []

        def post(url, json=None, **_kw):
            sent.append(copy.deepcopy(json))
            return ok()

        with mock.patch("core.sam_extra_probe.peek_capabilities", return_value=peek_caps), \
                mock.patch("core.forge_modules.resolve_sam3_checkpoint", side_effect=lambda name: name), \
                mock.patch("backends.webui_backend.requests.post", side_effect=post):
            WebUIBackend("http://127.0.0.1:7860").refine(png_b64(), settings)
        return sent[0]["alwayson_scripts"]

    def test_refine_carries_the_t2i_dora_block_only_when_forge_has_it(self):
        host = DoraHost(None, guidance={"guid_enabled": "false"})
        scripts = self._refine_request(host, caps(present=(DORA,)))
        self.assertEqual(list(scripts), ["SAM3 Mask", DORA])
        self.assertEqual(scripts[DORA], LYCORIS_BLOCK)
        self.assertEqual(list(self._refine_request(host, None)), ["SAM3 Mask"])       # 모름 → 안 보냄(A2)
        self.assertEqual(list(self._refine_request(host, caps(present=(), img2img=()))), ["SAM3 Mask"])


# ── 8. 저장·첫 로드 안내(A5) ─────────────────────────────────────────────────
class ProxyWiringTests(unittest.TestCase):
    """앱이 시작할 때 DoRA 위젯 프록시를 만드는지. 다른 테스트는 모두 ``dora_widgets`` 를 손으로 심은 호스트라, 이
    배선이 빠지면(current_settings()=None → 블록 없음, 저장값 {}) 기능이 죽어도 아무 테스트도 실패하지 않는다 —
    카드는 readValues 가 DEFAULTS 로 채워 'LyCORIS · 그대로 복제' 를 계속 보여 준다."""

    class _Bridge:
        def __init__(self):
            self.registered, self.pushed = {}, []

        def _register_proxy(self, widget_id, proxy):
            self.registered[widget_id] = proxy

        def pushWidgetValue(self, widget_id, value):
            self.pushed.append((widget_id, value))

        def pushWidgetProperty(self, *_args):
            pass

    def test_generator_ui_setup_creates_the_dora_proxies(self):
        source = (REPO / "ui" / "generator_ui_setup.py").read_text(encoding="utf-8")
        self.assertIn("from ui.dora_infer_mode_ui import init_dora_proxies", source)
        self.assertIn("self.dora_widgets = init_dora_proxies(b)", source)

    def test_init_dora_proxies_registers_every_widget_id_with_the_app_defaults(self):
        bridge = self._Bridge()
        widgets = dui.init_dora_proxies(bridge)
        ids = {dim.widget_id(key) for key in dim.WIDGET_KEYS}
        self.assertEqual(set(bridge.registered), ids)
        self.assertEqual(set(widgets), set(dim.WIDGET_KEYS))
        for key, proxy in widgets.items():
            self.assertIs(bridge.registered[dim.widget_id(key)], proxy)
        defaults = dim.widget_values(dim.APP_DEFAULTS)
        self.assertEqual({key: widgets[key].text() for key in dim.WIDGET_KEYS}, defaults)
        self.assertEqual(dict(bridge.pushed), {dim.widget_id(key): value for key, value in defaults.items()})
        # 실제 프록시를 읽는 체인이 앱 기본값 블록을 만든다
        host = ChainHost(None, ALL_PRESENT)
        host.dora_widgets = widgets
        with mock.patch("backends.get_backend_type", return_value=BackendType.WEBUI):
            self.assertEqual(sb.build_sampling_blocks(host, ap.TARGET_T2I).blocks[DORA], LYCORIS_BLOCK)
        self.assertEqual(dui.get_settings(host), defaults)


class PersistenceTests(unittest.TestCase):
    def test_settings_dict_and_preset_restore(self):
        from ui.generator_settings import _dora_settings
        from ui.generation_settings_apply import apply_generation_settings
        from tests.test_generation_presets import _host

        host = _host([])
        host.dora_widgets = dora_widgets(mode="forge_fp32", inserted="weak", weak_strength="0.3")
        saved = _dora_settings(host)
        self.assertEqual(saved["mode"], "forge_fp32")
        self.assertEqual(saved["weak_strength"], "0.3")
        other = _host([])
        other.dora_widgets = dora_widgets()
        apply_generation_settings(other, {dim.SETTINGS_KEY: saved}, only_present=True)
        self.assertEqual(dui.get_settings(other), saved)
        # 프리셋에 키가 없으면 그대로(알림도 없다)
        apply_generation_settings(other, {"steps": "30"}, only_present=True)
        self.assertEqual(dui.get_settings(other), saved)
        self.assertEqual(boot_notices.pending_boot_notices(other), [])

    def test_build_settings_dict_saves_the_card(self):
        from ui.generator_settings import SettingsMixin

        class _Host(SettingsMixin):
            def __getattr__(self, name):
                if name.startswith("__"):
                    raise AttributeError(name)
                value = mock.MagicMock(name=name)
                setattr(self, name, value)
                return value

        host = _Host()
        host.dora_widgets = dora_widgets(enabled="false")
        host.s1_widgets = {}
        host.s2_widgets = {}
        host._get_slot_settings = lambda *_a: {}
        host._get_sam3_settings = lambda *_a: {}
        host._get_anima_guidance_settings = lambda *_a: {}
        host._persisted_combo_text = lambda *_a: ""
        host._get_existing_setting = lambda key, default="": default
        host.random_resolutions = []
        host.base_prefix_prompt = host.base_suffix_prompt = host.base_neg_prompt = ""
        with mock.patch("ui.generator_settings.extras_of") as extras:
            extras.return_value.to_settings.return_value = {}
            settings = SettingsMixin._build_settings_dict(host)
        self.assertEqual(settings[dim.SETTINGS_KEY]["enabled"], "false")

    def test_old_settings_file_starts_with_app_defaults_and_toasts_once_after_vue_is_ready(self):
        _webui(self)
        clock = [0.0]
        host = SimpleNamespace(dora_widgets=dora_widgets(enabled="false"),
                               vue_bridge=SimpleNamespace(showNotification=mock.Mock()),
                               _sam_extra_notice_throttle=sn.NoticeThrottle(clock=lambda: clock[0]))
        dui.apply_saved_settings(host, {"steps": "28"}, only_present=False)       # load_settings, 옛 파일
        self.assertEqual(dui.current_settings(host), dim.APP_DEFAULTS)
        host.vue_bridge.showNotification.emit.assert_not_called()               # __init__ 에서는 띄우지 않는다
        pending = boot_notices.pending_boot_notices(host)
        self.assertEqual([item.key for item in pending], [sn.CODE_DORA_APP_DEFAULT])
        self.assertIn("LyCORIS", pending[0].build().message)
        self.assertEqual(boot_notices.flush_boot_notices(host), 1)             # Vue 준비 신호(set_lora_stack)
        # 한 번만 — 비워서 막는다(억제 시간에 기대지 않는다: 10분 뒤의 set_lora_stack 마다 다시 뜨면 안 된다)
        self.assertEqual(boot_notices.pending_boot_notices(host), [])
        clock[0] = 10 * sn.PRE_GENERATION_NOTICE_TTL_S
        self.assertEqual(boot_notices.flush_boot_notices(host), 0)
        host.vue_bridge.showNotification.emit.assert_called_once()

    def test_first_load_notice_is_decided_by_the_backend_when_it_is_shown(self):
        """ComfyUI 는 DoRA 블록을 만들지 않으므로(순정) 첫 로드 안내를 띄우지 않는다. Forge 문구는 Forge 로 한정한다
        (시작 게이트에서 백엔드를 아직 안 골랐을 수 있다 — 쌓을 때가 아니라 띄울 때 정한다)."""
        for backend, shown in ((BackendType.COMFYUI, 0), (BackendType.WEBUI, 1)):
            with self.subTest(backend=backend):
                host = SimpleNamespace(dora_widgets=dora_widgets(enabled="false"),
                                       vue_bridge=SimpleNamespace(showNotification=mock.Mock()))
                with mock.patch("backends.get_backend_type", return_value=BackendType.WEBUI):
                    dui.apply_saved_settings(host, {"steps": "28"}, only_present=False)   # load_settings 때
                with mock.patch("backends.get_backend_type", return_value=backend):
                    self.assertEqual(boot_notices.flush_boot_notices(host), shown)      # 띄울 때
                self.assertEqual(boot_notices.pending_boot_notices(host), [])
                self.assertEqual(dui.current_settings(host), dim.APP_DEFAULTS)          # 값은 백엔드와 무관
                if shown:
                    message = host.vue_bridge.showNotification.emit.call_args.args[1]
                    self.assertIn("Forge 로 생성할 때", message)
                    self.assertIn("ComfyUI 는 늘 순정", message)
                else:
                    host.vue_bridge.showNotification.emit.assert_not_called()

    def test_saved_key_restores_quietly(self):
        host = SimpleNamespace(dora_widgets=dora_widgets())
        dui.apply_saved_settings(host, {dim.SETTINGS_KEY: {"enabled": "false"}}, only_present=False)
        self.assertFalse(dui.current_settings(host).enabled)
        self.assertEqual(boot_notices.pending_boot_notices(host), [])

    def test_vue_ready_signal_flushes_the_boot_notices(self):
        """set_lora_stack(부팅 복원 뒤 Vue 가 늘 보내는 액션) 처리 가지가 쌓인 알림을 비운다."""
        source = (REPO / "ui" / "generator_main.py").read_text(encoding="utf-8")
        branch = source[source.index("elif action == 'set_lora_stack':"):]
        branch = branch[:branch.index("elif action", 10)]
        self.assertIn("flush_boot_notices(self)", branch)
        lora = (REPO / "frontend" / "src" / "composables" / "useLoraStack.js").read_text(encoding="utf-8")
        self.assertRegex(lora, r"function restoreFromPrefs[\s\S]*?nextTick\(\(\) => \{[\s\S]*?syncLoraStack\(\)")


class BootNoticeTimingTests(unittest.TestCase):
    """(A5) 부팅 알림은 Vue 준비와 메인 창 표시가 둘 다 참일 때 띄운다. 데스크톱은 창을 숨긴 채 Vue 를 로드하고 백엔드
    연결을 최대 8초 기다린 뒤에야 창을 띄운다(new_main_ui.main) — Vue 준비 신호는 스플래시 단계에 오고 토스트 수명은
    3초라, 그때 띄우면 사용자가 보기 전에 사라지고 pending 도 비어 다시 뜨지 않는다."""

    def _host(self, *, visible):
        state = {"visible": visible}
        host = SimpleNamespace(isVisible=lambda: state["visible"],
                               vue_bridge=SimpleNamespace(showNotification=mock.Mock()))
        boot_notices.defer_boot_notice(host, sn.dora_app_default_notice(dim.describe(dim.APP_DEFAULTS)))
        return host, state

    def test_hidden_window_holds_the_notices_and_shows_them_once_after_it_appears(self):
        host, state = self._host(visible=False)
        self.assertEqual(boot_notices.flush_boot_notices(host), 0)          # Vue 준비 — 창은 아직 숨김(스플래시)
        host.vue_bridge.showNotification.emit.assert_not_called()
        self.assertEqual(len(boot_notices.pending_boot_notices(host)), 1)   # 버리지 않고 보류
        self.assertEqual(boot_notices.flush_after_window_shown(host), 0)    # 아직 숨김
        state["visible"] = True                                               # main() 의 showMaximized
        self.assertEqual(boot_notices.flush_after_window_shown(host), 1)
        self.assertEqual(boot_notices.pending_boot_notices(host), [])
        self.assertEqual(boot_notices.flush_after_window_shown(host), 0)
        self.assertEqual(boot_notices.flush_boot_notices(host), 0)
        host.vue_bridge.showNotification.emit.assert_called_once()

    def test_window_shown_before_vue_is_ready_waits_for_the_vue_signal(self):
        host, _state = self._host(visible=True)
        self.assertEqual(boot_notices.flush_after_window_shown(host), 0)    # Vue 가 아직 구독 전 — 보내면 사라진다
        host.vue_bridge.showNotification.emit.assert_not_called()
        self.assertEqual(boot_notices.flush_boot_notices(host), 1)          # 창이 보이니 신호에서 바로
        host.vue_bridge.showNotification.emit.assert_called_once()

    def test_web_mode_needs_only_the_vue_signal(self):
        host, _state = self._host(visible=False)                             # 웹 모드는 호스트 창을 띄우지 않는다
        host.web_mode = True
        self.assertEqual(boot_notices.flush_boot_notices(host), 1)

    def _scheduled_timer(self, host):
        """``schedule_flush_after_show`` 가 QTimer.singleShot 에 건 (지연, 콜백) — 콜백은 부르지 않고 돌려준다."""
        with mock.patch("PyQt6.QtCore.QTimer") as timer:
            boot_notices.schedule_flush_after_show(host)
        timer.singleShot.assert_called_once()
        return timer.singleShot.call_args.args

    def test_main_schedules_the_flush_right_after_showing_the_window(self):
        source = (REPO / "new_main_ui.py").read_text(encoding="utf-8")
        body = source[source.index("def main("):]
        shown = body.index("window.showMaximized()")
        self.assertIn("schedule_flush_after_show(window)", body[shown:body.index("run_main_loop(app)")])
        host, state = self._host(visible=False)
        boot_notices.flush_boot_notices(host)                                 # 스플래시 단계의 Vue 준비
        state["visible"] = True
        delay, callback = self._scheduled_timer(host)
        self.assertEqual(delay, boot_notices.SHOW_FLUSH_DELAY_MS)
        self.assertLess(delay, 3000)                                          # 토스트 수명(useToasts TOAST_TTL_MS)
        callback()
        host.vue_bridge.showNotification.emit.assert_called_once()

    def test_timer_after_show_does_not_stand_in_for_the_vue_signal(self):
        """시작 게이트 경로: 창이 먼저 뜨고 Vue 복원(getInitialConfig → uiPrefsLoaded → set_lora_stack)이 1초보다 늦으면
        타이머 콜백이 Vue 준비 신호보다 먼저 돈다. 콜백이 스스로 Vue 준비로 치고 띄우면(예: flush_boot_notices 로 바꿈)
        App.vue 가 showNotification 을 구독하기 전이라 토스트가 사라지고 pending 도 비어 다시 뜨지 않는다
        (다음 저장이 설정 키를 쓴다). 그래서 콜백은 보류만 하고, 띄우는 것은 그 뒤의 Vue 신호다."""
        host, _state = self._host(visible=True)
        _delay, callback = self._scheduled_timer(host)                       # Vue 신호보다 먼저 잡는다
        callback()                                                            # 창 표시 1초 뒤 — Vue 는 아직
        host.vue_bridge.showNotification.emit.assert_not_called()
        self.assertEqual(len(boot_notices.pending_boot_notices(host)), 1)   # 버리지 않고 보류
        self.assertEqual(boot_notices.flush_boot_notices(host), 1)          # Vue 준비 신호 — 창이 보이니 바로
        callback()                                                            # 늦게 한 번 더 돌아도 더 띄우지 않는다
        host.vue_bridge.showNotification.emit.assert_called_once()
        self.assertEqual(boot_notices.pending_boot_notices(host), [])


# ── 11. 결과 알림 ─────────────────────────────────────────────────────────────
def _info(*infotexts):
    return {"infotexts": list(infotexts)}


class ResultNoticeTests(unittest.TestCase):
    def payload(self, block):
        return {"prompt": "x", "alwayson_scripts": {DORA: block}}

    def test_missing_mode_record_warns(self):
        notices = sn.result_notices(_info("x\nSteps: 20, Seed: 1"), self.payload(USER_FP32_BLOCK))
        self.assertEqual([(n.code, n.level) for n in notices], [(sn.CODE_DORA_NOT_APPLIED, sn.LEVEL_WARNING)])
        self.assertEqual(sn.notice_ttl(notices[0], sn.RESULT_NOTICE_TTL_S), sn.PRE_GENERATION_NOTICE_TTL_S)  # B14

    def test_recorded_mode_is_quiet(self):
        self.assertEqual(sn.result_notices(_info("x\nSteps: 20, DoRA mode: Forge fp32"),
                                           self.payload(USER_FP32_BLOCK)), [])
        self.assertEqual(sn.result_notices(_info("x\nSteps: 20, DoRA mode: LyCORIS"), self.payload(LYCORIS_BLOCK)), [])

    def test_app_default_block_without_record_only_logs(self):
        """(A6) 사용자가 켜지 않은 앱 기본값 블록이 훅 폴백(weight_adapter 훅 없는 Forge)으로 순정이 됐으면 로그만 —
        생성마다·억제 시간(600초)마다 경고하지 않는다. 게이트가 같은 블록을 뺄 때(gate_notices)와 같은 규칙."""
        clock = [0.0]
        throttle = sn.NoticeThrottle(clock=lambda: clock[0])
        shown = []
        with self.assertLogs("core.sam_extra_notices", "INFO") as logs:
            for now in (0.0, 601.0, 1202.0):
                clock[0] = now
                for notice in sn.result_notices(_info("x\nSteps: 20, Seed: 1"), self.payload(LYCORIS_BLOCK)):
                    if throttle.allow(notice, sn.notice_ttl(notice, sn.RESULT_NOTICE_TTL_S)):
                        shown.append(notice.code)
        self.assertEqual(shown, [])
        self.assertTrue(any("DoRA mode" in line for line in logs.output))
        self.assertEqual(sn.requested_features(self.payload(LYCORIS_BLOCK)).dora, dim.APP_DEFAULTS)
        # 같은 LyCORIS 라도 사용자가 바꾼 값(약한 복사)이면 여전히 경고한다
        user = {"args": [dict(LYCORIS_BLOCK["args"][0], inserted="weak")]}
        self.assertEqual([n.detail for n in sn.result_notices(_info("x\nSteps: 20"), self.payload(user))],
                         ["mode:lycoris", "inserted:weak"])

    def test_stock_or_absent_block_is_quiet(self):
        stock = {"args": [{"enabled": False, "mode": "lycoris"}]}
        self.assertEqual(sn.result_notices(_info("x\nSteps: 20"), self.payload(stock)), [])
        self.assertEqual(sn.result_notices(_info("x\nSteps: 20"), {"alwayson_scripts": {}}), [])

    def test_inserted_policy_without_record_warns(self):
        block = {"args": [{"enabled": True, "mode": "forge", "inserted": "weak", "weak_strength": 0.12,
                           "weak_scope": "attn"}]}
        notices = sn.result_notices(_info("x\nSteps: 20"), self.payload(block))
        self.assertEqual([n.detail for n in notices], ["inserted:weak"])
        self.assertEqual(sn.result_notices(_info("x\nSteps: 20, DoRA inserted: weak 0.12 attn"),
                                           self.payload(block)), [])

    def test_requested_features_reads_the_block(self):
        self.assertEqual(sn.requested_features(self.payload(LYCORIS_BLOCK)).dora, dim.APP_DEFAULTS.__class__(
            True, "lycoris", "keep", 0.12, "attn"))
        self.assertIsNone(sn.requested_features({"alwayson_scripts": {}}).dora)

    def test_new_codes_are_throttled_like_pre_generation_warnings(self):
        for code in (sn.CODE_DORA_NOT_APPLIED, sn.CODE_DORA_COMFY_STOCK, sn.CODE_DORA_CHOICE):
            self.assertEqual(sn.NOTICE_MIN_TTL_S[code], sn.PRE_GENERATION_NOTICE_TTL_S)


# ── 기능 스냅샷 경고 ───────────────────────────────────────────────────────────
class CapabilitiesTests(unittest.TestCase):
    def test_unknown_live_labels_become_a_snapshot_warning(self):
        from core import sam_extra_capabilities as sec
        from tests.test_sam_extra_capabilities import _codes, _script, build, live_bodies

        bodies = live_bodies()
        mode = _script(bodies, sec.TITLE_DORA)["args"][1]
        mode["choices"] = [*mode["choices"], "Quantum (new)"]
        snapshot, _fake = build(bodies)
        warnings = [w for w in snapshot.warnings if w["code"] == "dora_choices_unknown"]
        self.assertEqual(len(warnings), 1)
        self.assertIn("Quantum (new)", warnings[0]["message"])
        clean, _fake = build(live_bodies())
        self.assertNotIn("dora_choices_unknown", _codes(clean))
        # 고른 값(LyCORIS)은 그대로 있으니 계속 보낸다
        self.assertIsNone(dim.live_choice_problem(snapshot, dim.APP_DEFAULTS))


# ── 9. Vue 거울 ───────────────────────────────────────────────────────────────
class VueMirrorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.ts = TS.read_text(encoding="utf-8")
        cls.card = CARD.read_text(encoding="utf-8")

    def _options(self, name):
        block = re.search(rf"export const {name}: readonly DoraOption\[\] = \[(.*?)\n\]", self.ts, re.S).group(1)
        return tuple(re.findall(r"\{ key: '([^']+)', label: '([^']+)' \}", block))

    def test_options_keys_and_labels_match_python(self):
        self.assertEqual(self._options("MODE_OPTIONS"), dim.MODE_OPTIONS)
        self.assertEqual(self._options("INSERT_OPTIONS"), dim.INSERT_OPTIONS)
        self.assertEqual(self._options("SCOPE_OPTIONS"), dim.SCOPE_OPTIONS)

    def test_widget_ids_and_defaults_match_python(self):
        ids = dict(re.findall(r"^\s+(\w+): '(_dora_\w+)',$", self.ts, re.M))
        self.assertEqual(ids, {key: dim.widget_id(key) for key in dim.WIDGET_KEYS})
        block = re.search(r"export const DEFAULTS: Readonly<DoraValues> = \{(.*?)\n\}", self.ts, re.S).group(1)
        self.assertEqual(dict(re.findall(r"(\w+): '([^']*)'", block)), dim.widget_values(dim.APP_DEFAULTS))
        short = re.search(r"export const SHORT_LABELS[^=]*= \{(.*?)\n\}", self.ts, re.S).group(1)
        self.assertEqual(dict(re.findall(r"(\w+): '([^']*)'", short)), dim.SHORT_LABELS)

    def test_card_binds_every_widget_through_the_shared_ids(self):
        self.assertIn("WIDGET_IDS", self.card)
        for key in dim.WIDGET_KEYS:
            with self.subTest(key=key):
                self.assertTrue(re.search(rf"WIDGET_IDS\.{key}\b|'{key}'", self.card), key)
        self.assertNotIn("requestAction", self.card)          # 새 브리지 액션 없음
        app = (REPO / "frontend" / "src" / "App.vue").read_text(encoding="utf-8")
        self.assertIn("<DoraModeCard />", app)
        self.assertIn("import DoraModeCard from './components/params/DoraModeCard.vue'", app)


if __name__ == "__main__":
    unittest.main()
