"""Anima 3.8B 제어(P9) — core/anima38 블록 규칙·출처, 모델 종류별 판정, 샘플링 블록 기여자, 보조 패스 모델 대조(A7),
모델 종류 미리 읽기, 저장·첫 로드 안내(A5), 알림, Vue 거울.

GPU·Forge 없이: 기능 스냅샷은 가짜(``tests.test_alwayson_propagation.caps``), Forge 요청은 가짜 requests.post·
``WebUIBackend._active_checkpoint``, 체크포인트 헤더는 임시 폴더의 가짜 safetensors.
"""
import copy
import json
import os
import re
import tempfile
import unittest
from dataclasses import asdict, replace
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from backends import BackendType
from core import alwayson_propagation as ap
from core import anima38 as a38
from core import anima_model_kind as amk
from core import sam_extra_contract as reg
from core import sam_extra_notices as sn
from tests.test_alwayson_propagation import A38, DD, DORA, PAG, SKIM, caps
from tests.test_anima_model_kind import ANIMA_HEADER, OTHER_HEADER, V2_HEADER, write_safetensors
from tests.test_chat_generation_snapshot import ReadOnlyWidget
from tests.test_sampling_blocks import ALL_PRESENT, GUIDANCE_ON, ChainHost, _webui
from ui import anima38_ui as aui
from ui import boot_notices
from ui import sampling_blocks as sb

REPO = Path(__file__).resolve().parent.parent
TS = REPO / "frontend" / "src" / "utils" / "anima38Card.ts"
CARD = REPO / "frontend" / "src" / "components" / "params" / "Anima38Card.vue"
V2_MODEL = "Anima-3.8B-v1.1.safetensors"          # 이름으로 v2(SnapshotHost 의 모델)
ANIMA_MODEL = "Anima-2.9B-preview-v1.safetensors"  # 이름으로 비 번들 Anima
PLAIN_MODEL = "sdxl_base.safetensors"             # 이름으로는 모름
APP_BLOCK = {"args": [{"enabled": False, "adapter": a38.DEFAULT_ADAPTER, "strength": 1.0, "negative": True,
                       "negative_strength": 1.0, "bypass": False}]}
OFF = a38.DEFAULT_SETTINGS
KINDS = (a38.KIND_V2, a38.KIND_ANIMA, a38.KIND_OTHER, a38.KIND_UNKNOWN)


class _Proxy:
    def __init__(self, value=""):
        self.value = value

    def text(self):
        return self.value

    def setText(self, value):
        self.value = value


def a38_widgets(**values):
    base = a38.widget_values(a38.APP_T2I_DEFAULTS, False)
    base.update({k: str(v) for k, v in values.items()})
    return {key: _Proxy(base[key]) for key in a38.WIDGET_KEYS}


class Anima38Host(ChainHost):
    """실제 메인 체인 + Anima 3.8B 카드 위젯(모델은 T2I 콤보)."""

    def __init__(self, capabilities=None, guidance=None, model=V2_MODEL, **a38_values):
        super().__init__(guidance, capabilities)
        self.model_combo = ReadOnlyWidget(model)
        self.anima38_widgets = a38_widgets(**a38_values)


class _KindCache(unittest.TestCase):
    def setUp(self):
        amk.forget_kinds()
        self.addCleanup(amk.forget_kinds)


# ── 1. 상수·앱 기본값 ─────────────────────────────────────────────────────────
class ConstantsTests(unittest.TestCase):
    def test_extension_defaults_and_app_defaults(self):
        self.assertEqual(dict(a38.ARG_DEFAULTS), asdict(a38.DEFAULT_SETTINGS))
        self.assertFalse(a38.V1_ENABLED_DEFAULT)                                   # 사용자 결정 D1=B
        diff = {k for k, v in asdict(a38.APP_T2I_DEFAULTS).items() if v != a38.ARG_DEFAULTS[k]}
        self.assertEqual(diff, {"negative"})                                       # ui-config :5452 만
        self.assertTrue(a38.APP_T2I_DEFAULTS.negative)

    def test_widget_ids_and_settings_key(self):
        self.assertEqual(a38.WIDGET_KEYS, (*a38.ARG_NAMES, "apply_img2img"))
        self.assertEqual({a38.widget_id(k) for k in a38.WIDGET_KEYS},
                         {f"_a38_{k}" for k in a38.WIDGET_KEYS})
        with self.assertRaises(KeyError):
            a38.widget_id("bogus")
        self.assertEqual(a38.SETTINGS_KEY, "anima38_settings")
        from core.generation_presets import PRESET_KEYS
        self.assertIn(a38.SETTINGS_KEY, PRESET_KEYS)


# ── 2. effective·build_block — 모델 종류 × 설정 ──────────────────────────────────
class EffectiveTests(unittest.TestCase):
    CASES = {
        # 이름: (설정, {종류: 보내나})
        "모두 끔": (OFF, {"v2": False, "anima": False, "other": False, "unknown": False}),
        "부정 커넥터": (replace(OFF, negative=True), {"v2": True, "anima": False, "other": False, "unknown": True}),
        "Bypass": (replace(OFF, bypass=True), {"v2": True, "anima": False, "other": False, "unknown": True}),
        "v1 켬": (replace(OFF, enabled=True), {"v2": False, "anima": True, "other": False, "unknown": True}),
        "v1 + Bypass": (replace(OFF, enabled=True, bypass=True),
                        {"v2": True, "anima": False, "other": False, "unknown": True}),
        "v1 + 부정": (replace(OFF, enabled=True, negative=True, strength=0.65),
                     {"v2": True, "anima": True, "other": False, "unknown": True}),
        "강도만 바꿈": (replace(OFF, strength=0.5, negative_strength=0.3),
                    {"v2": False, "anima": False, "other": False, "unknown": False}),
    }

    def test_table(self):
        for label, (settings, expected) in self.CASES.items():
            for kind in KINDS:
                with self.subTest(label, kind=kind):
                    block = a38.build_block(settings, kind=kind)
                    self.assertEqual(block is not None, expected[kind])
                    self.assertEqual(a38.effective(settings, kind) is not None, expected[kind])
                    if block is not None:
                        self.assertEqual(list(block["args"][0]), list(a38.ARG_NAMES))      # dict 한 개, 6키
                        self.assertEqual(a38.parse_script_block(block), settings)          # 왕복

    def test_app_default_block_shape(self):
        self.assertEqual(a38.build_block(a38.APP_T2I_DEFAULTS, kind=a38.KIND_V2), APP_BLOCK)


# ── 3. 설정 ↔ 위젯·infotext ─────────────────────────────────────────────────────
class SettingsTests(unittest.TestCase):
    def test_widget_values_round_trip_and_string_booleans(self):
        settings = replace(a38.APP_T2I_DEFAULTS, enabled=True, strength=0.65, bypass=True)
        values = a38.widget_values(settings, True)
        self.assertEqual(values["strength"], "0.65")
        self.assertEqual(values["negative_strength"], "1")
        self.assertEqual(a38.settings_from_widgets(values), (settings, True))
        parsed, img2img = a38.settings_from_widgets({"enabled": "1", "negative": "off", "strength": "5",
                                                     "negative_strength": "nan", "bypass": "yes"})
        self.assertEqual((parsed.enabled, parsed.negative, parsed.strength, parsed.negative_strength, parsed.bypass),
                         (True, False, 2.0, 1.0, True))
        self.assertFalse(img2img)

    def test_missing_cells_fall_back_to_app_defaults(self):
        self.assertEqual(a38.settings_from_widgets(None), (a38.APP_T2I_DEFAULTS, False))
        self.assertEqual(a38.settings_from_widgets({}), (a38.APP_T2I_DEFAULTS, False))
        parsed, img2img = a38.settings_from_widgets({"negative": "", "adapter": "", "apply_img2img": "true"})
        self.assertEqual(parsed, a38.APP_T2I_DEFAULTS)
        self.assertTrue(img2img)

    def test_settings_from_infotext_mirrors_the_extension_paste(self):
        """확장 _paste_*(scripts/anima_3_8b.py): 흔적이 있는 칸만 채운다."""
        v2 = {"Anima38": "v2 bundle", "Anima38 negative": "connector", "Anima38 negative strength": 1.0,
              "Anima38 architecture": "anima_qwen35_quality_anchored_semantic_connector_v2",
              "Anima38 adapter": "bundle_adapter.safetensors", "Anima38 strength": 1.0}
        self.assertEqual(a38.settings_from_infotext(v2), {"bypass": False, "negative": True, "negative_strength": 1.0,
                                                          "enabled": False})
        self.assertEqual(a38.settings_from_infotext({"Anima38": "bypass"}), {"bypass": True})
        native = {"Anima38": "v2 bundle", "Anima38 negative": "native",
                  "Anima38 architecture": "anima_qwen35_quality_anchored_semantic_connector_v2"}
        self.assertEqual(a38.settings_from_infotext(native), {"bypass": False, "negative": False, "enabled": False})
        v1 = {"Anima38": "v1 adapter", "Anima38 architecture": amk.V1_ADAPTER_ARCHITECTURE,
              "Anima38 adapter": "Anima-3.8B-expanded_adapter.safetensors", "Anima38 strength": "0.7",
              "Anima38 negative strength": "0.3"}
        self.assertEqual(a38.settings_from_infotext(v1), {
            "bypass": False, "negative": True, "negative_strength": 0.3, "enabled": True,
            "adapter": "Anima-3.8B-expanded_adapter.safetensors", "strength": 0.7})
        legacy = {"8B": "v1 adapter", "8B architecture": amk.V1_ADAPTER_ARCHITECTURE, "8B strength": "1.2"}
        self.assertEqual(a38.settings_from_infotext(legacy), {"bypass": False, "enabled": True, "strength": 1.2})
        self.assertEqual(a38.settings_from_infotext({"Steps": "20"}), {})
        self.assertEqual(a38.settings_from_infotext(None), {})


# ── 4. plan — 대상·백엔드·종류·출처 ───────────────────────────────────────────────
class PlanTests(unittest.TestCase):
    def plan(self, settings=a38.APP_T2I_DEFAULTS, **kw):
        kw.setdefault("target", a38.TARGET_T2I)
        kw.setdefault("backend", a38.BACKEND_WEBUI)
        kw.setdefault("kind", a38.KIND_V2)
        return a38.plan(settings, **kw)

    def test_app_default_goes_to_v2_only(self):
        sent = self.plan()
        self.assertEqual((sent.block, sent.reason, sent.provenance), (APP_BLOCK, "send", a38.PROVENANCE_APP_DEFAULT))
        self.assertEqual(self.plan(kind=a38.KIND_ANIMA).reason, "not_effective")
        self.assertEqual(self.plan(kind=a38.KIND_OTHER).reason, "other")
        unknown = self.plan(kind=a38.KIND_UNKNOWN)                 # 확인 전·원격 — 앱 기본값은 보내지 않는다
        self.assertEqual((unknown.block, unknown.reason), (None, "unknown_default"))

    def test_user_values_are_sent_to_unknown_models(self):
        user = self.plan(replace(a38.APP_T2I_DEFAULTS, bypass=True), kind=a38.KIND_UNKNOWN)
        self.assertEqual((user.reason, user.provenance), ("send", a38.PROVENANCE_USER))
        self.assertTrue(user.block["args"][0]["bypass"])
        v1 = self.plan(replace(a38.APP_T2I_DEFAULTS, enabled=True), kind=a38.KIND_ANIMA)
        self.assertTrue(v1.block["args"][0]["enabled"])

    def test_i2i_follows_the_toggle_and_is_user_provenance(self):
        self.assertEqual(self.plan(target=a38.TARGET_I2I).reason, "i2i_off")
        on = self.plan(target=a38.TARGET_I2I, apply_img2img=True)
        self.assertEqual((on.block, on.provenance), (APP_BLOCK, a38.PROVENANCE_USER))
        aux = self.plan(target=a38.TARGET_AUX)                      # 보조 패스는 토글과 무관하게 T2I 값
        self.assertEqual((aux.block, aux.provenance), (APP_BLOCK, a38.PROVENANCE_APP_DEFAULT))

    def test_krea2_is_silent(self):
        quiet = self.plan(backend=a38.BACKEND_KREA2, settings=replace(OFF, enabled=True))
        self.assertEqual((quiet.block, quiet.notice, quiet.reason), (None, None, "krea2"))

    def test_comfy_builds_the_same_block_and_notes_the_module_pair(self):
        pair = ["vae.safetensors", "text/qwen35_4b.safetensors", "text/Anima-3.8B-expanded_adapter.safetensors"]
        comfy = self.plan(backend=a38.BACKEND_COMFY, modules=pair)
        self.assertEqual((comfy.block, comfy.notice), (APP_BLOCK, None))                 # v2 는 자동 — 알릴 것 없음
        anima = self.plan(backend=a38.BACKEND_COMFY, kind=a38.KIND_ANIMA, modules=pair)
        self.assertIsNone(anima.block)
        self.assertEqual((anima.notice.code, anima.notice.level), (sn.CODE_ANIMA38_COMFY_V1, sn.LEVEL_INFO))
        for label, kw in (("v1 켬", {"settings": replace(OFF, enabled=True)}),
                          ("Bypass", {"settings": replace(OFF, bypass=True)}),
                          ("쌍 없음", {"modules": pair[:2]}),
                          ("Forge", {"backend": a38.BACKEND_WEBUI}),
                          ("보조 패스", {"target": a38.TARGET_AUX})):
            with self.subTest(label):
                args = {"backend": a38.BACKEND_COMFY, "kind": a38.KIND_ANIMA, "modules": pair, **kw}
                self.assertIsNone(self.plan(**args).notice)
        self.assertIn("'v1 어댑터 켜기'를 켤 때만", anima.notice.message)
        self.assertNotIn("I2I·인페인트에도 적용", anima.notice.message)
        # I2I 토글이 꺼져 v1 이 가지 않는 요청도 알린다(카드는 v1 켬) — 원인은 I2I 토글이다(P9 리뷰 4):
        # 이미 켠 'v1 어댑터 켜기'를 켜라고 하지 않고 'I2I·인페인트에도 적용'을 말한다
        i2i = self.plan(replace(OFF, enabled=True), target=a38.TARGET_I2I, backend=a38.BACKEND_COMFY,
                        kind=a38.KIND_ANIMA, modules=pair)
        self.assertEqual((i2i.reason, i2i.notice.code), ("i2i_off", sn.CODE_ANIMA38_COMFY_V1))
        self.assertIn("'I2I·인페인트에도 적용'", i2i.notice.message)
        self.assertNotIn("'v1 어댑터 켜기'를 켤 때만", i2i.notice.message)
        # v1 도 꺼져 있으면 두 토글을 모두 말한다
        both = self.plan(OFF, target=a38.TARGET_I2I, backend=a38.BACKEND_COMFY, kind=a38.KIND_ANIMA, modules=pair)
        self.assertEqual(both.reason, "i2i_off")
        self.assertIn("'v1 어댑터 켜기'", both.notice.message)
        self.assertIn("'I2I·인페인트에도 적용'", both.notice.message)
        # 토글을 켠 I2I 는 T2I 와 같은 문구
        on = self.plan(OFF, target=a38.TARGET_I2I, backend=a38.BACKEND_COMFY, kind=a38.KIND_ANIMA, modules=pair,
                       apply_img2img=True)
        self.assertEqual(on.notice.message, anima.notice.message)

    def test_comfy_notice_names_the_module_adapter_the_card_must_pick(self):
        """(P9 리뷰 2) 모듈 목록의 어댑터 이름이 카드 어댑터와 다르면 v1 을 켤 때 카드에서 그 어댑터를 골라야 한다 —
        컴파일러는 카드 값만 쓴다(Forge 와 같게). 기본 이름이면 고를 것이 없다."""
        alt = "text/MyAnima_v1_adapter.safetensors"
        pair = ["text/qwen35_4b.safetensors", alt]
        named = self.plan(backend=a38.BACKEND_COMFY, kind=a38.KIND_ANIMA, modules=pair)
        self.assertIn(f"'{alt}'", named.notice.message)
        self.assertIn("어댑터 칸", named.notice.message)
        default = self.plan(backend=a38.BACKEND_COMFY, kind=a38.KIND_ANIMA,
                            modules=["text/qwen35_4b.safetensors", f"text/{a38.DEFAULT_ADAPTER}"])
        self.assertNotIn("어댑터 칸", default.notice.message)
        picked = self.plan(replace(OFF, adapter=alt), backend=a38.BACKEND_COMFY, kind=a38.KIND_ANIMA, modules=pair)
        self.assertNotIn("어댑터 칸", picked.notice.message)                          # 이미 골랐다
        two = self.plan(backend=a38.BACKEND_COMFY, kind=a38.KIND_ANIMA,
                        modules=[*pair, "text/Other_anima_adapter.safetensors"])
        self.assertNotIn("어댑터 칸", two.notice.message)                             # 여럿이면 하나를 집지 않는다


# ── 5. 기여자(실제 빌더·메인 체인) ─────────────────────────────────────────────────
class ContributorTests(_KindCache):
    def setUp(self):
        super().setUp()
        _webui(self)

    def test_t2i_known_present_sends_the_app_default_block_between_guidance_and_dora(self):
        from ui.dora_infer_mode_ui import contribute as dora
        from tests.test_dora_infer_mode import dora_widgets
        host = Anima38Host(ALL_PRESENT)
        host.dora_widgets = dora_widgets()
        result = sb.build_sampling_blocks(host, ap.TARGET_T2I)
        self.assertEqual(list(result.blocks), [ap.TITLE_NEGPIP, PAG, SKIM, DD, A38, DORA])
        self.assertEqual(result.blocks[A38], APP_BLOCK)
        self.assertEqual(result.provenance[A38], ap.PROVENANCE_APP_DEFAULT)
        self.assertEqual(result.notices, [])
        self.assertLess(sb.CONTRIBUTORS.index(aui.contribute), sb.CONTRIBUTORS.index(dora))

    def test_unknown_snapshot_skips_app_defaults_quietly_and_sends_user_values(self):
        """(A2) 앱 기본값은 확장 확인 전에는 보내지 않고(로그만 — A6), 사용자가 바꾼 값은 지금처럼 보낸다."""
        with self.assertLogs("core.alwayson_propagation", "INFO") as logs:
            quiet = sb.build_sampling_blocks(Anima38Host(None), ap.TARGET_T2I)
        self.assertNotIn(A38, quiet.blocks)
        self.assertEqual(quiet.notices, [])
        self.assertTrue(any(A38 in line for line in logs.output))
        user = sb.build_sampling_blocks(Anima38Host(None, bypass="true"), ap.TARGET_T2I)
        self.assertTrue(user.blocks[A38]["args"][0]["bypass"])
        self.assertEqual(user.notices, [])

    def test_missing_script_warns_only_for_user_values(self):
        present = (PAG, SKIM, DD, DORA)
        quiet = sb.build_sampling_blocks(Anima38Host(caps(present=present)), ap.TARGET_T2I)
        self.assertNotIn(A38, quiet.blocks)
        self.assertEqual(quiet.notices, [])                                    # A6 — 반복 경고 없음
        user = sb.build_sampling_blocks(Anima38Host(caps(present=present), bypass="true"), ap.TARGET_T2I)
        self.assertEqual([(n.code, n.level) for n in user.notices], [(sn.CODE_BLOCK_NOT_SENT, sn.LEVEL_WARNING)])

    def test_model_kind_from_the_cache_decides_the_block(self):
        for label, model, remembered, expected in (
                ("이름 v2", V2_MODEL, {}, APP_BLOCK),
                ("헤더로 확인한 비 Anima", "renamed-3.8b-v2-anima.safetensors",
                 {"renamed-3.8b-v2-anima.safetensors": amk.KIND_OTHER}, None),
                ("헤더로 확인한 v2(이름 무관)", PLAIN_MODEL, {PLAIN_MODEL: amk.KIND_V2}, APP_BLOCK),
                ("비 번들 Anima", ANIMA_MODEL, {}, None),
                ("모름", PLAIN_MODEL, {}, None)):
            with self.subTest(label):
                amk.remember_kinds(remembered, replace=True)
                with mock.patch.object(amk, "header_kind", side_effect=AssertionError("GUI 스레드에서 디스크")):
                    result = sb.build_sampling_blocks(Anima38Host(ALL_PRESENT, model=model), ap.TARGET_T2I)
                self.assertEqual(result.blocks.get(A38), expected)

    def test_i2i_follows_the_toggle_and_uses_the_img2img_list(self):
        caps_i2i = caps(present=(PAG, SKIM, DD, A38, DORA), img2img=(A38,))
        self.assertNotIn(A38, sb.build_sampling_blocks(Anima38Host(caps_i2i), ap.TARGET_I2I).blocks)
        on = sb.build_sampling_blocks(Anima38Host(caps_i2i, apply_img2img="true"), ap.TARGET_I2I)
        self.assertEqual((on.blocks[A38], on.provenance[A38]), (APP_BLOCK, ap.PROVENANCE_USER))
        gone = sb.build_sampling_blocks(Anima38Host(caps(present=(A38,), img2img=()), {"guid_enabled": "false"},
                                                    apply_img2img="true"), ap.TARGET_I2I)
        self.assertNotIn(A38, gone.blocks)
        self.assertEqual([n.code for n in gone.notices], [sn.CODE_BLOCK_NOT_SENT])   # 토글을 켠 것은 사용자

    def test_i2i_and_inpaint_entry_points_follow_the_toggle(self):
        caps_all = caps(present=(PAG, SKIM, DD, A38, DORA))
        off = Anima38Host(caps_all).apply_alwayson_extensions({"prompt": "x", "init_images": []})
        self.assertNotIn(A38, off["alwayson_scripts"])                          # 기본: Forge img2img 탭 = 모두 끔
        on = Anima38Host(caps_all, apply_img2img="true").apply_alwayson_extensions({"prompt": "x", "init_images": []})
        self.assertEqual(on["alwayson_scripts"][A38], APP_BLOCK)

    def test_aux_uses_the_t2i_value_ungated(self):
        result = sb.build_sampling_blocks(Anima38Host(None), ap.TARGET_AUX)
        self.assertEqual((result.blocks[A38], result.provenance[A38]), (APP_BLOCK, ap.PROVENANCE_APP_DEFAULT))
        self.assertEqual(result.context.model, V2_MODEL)

    def test_comfy_builds_ungated_and_krea2_builds_nothing(self):
        with mock.patch("backends.get_backend_type", return_value=BackendType.COMFYUI):
            comfy = sb.build_sampling_blocks(Anima38Host(None), ap.TARGET_T2I)
        self.assertEqual(comfy.blocks[A38], APP_BLOCK)                            # 컴파일러가 같은 블록을 읽는다
        krea2 = sb.build_sampling_blocks(Anima38Host(ALL_PRESENT), ap.TARGET_T2I, krea2=True)
        self.assertNotIn(A38, krea2.blocks)
        self.assertEqual(krea2.notices, [])

    def test_comfy_module_pair_notice_reaches_the_builder(self):
        host = Anima38Host(None, model=ANIMA_MODEL)
        host.te_main_input = ReadOnlyWidget("qwen35_4b.safetensors, Anima-3.8B-expanded_adapter.safetensors")
        with mock.patch("backends.get_backend_type", return_value=BackendType.COMFYUI):
            result = sb.build_sampling_blocks(host, ap.TARGET_T2I)
        self.assertNotIn(A38, result.blocks)
        self.assertEqual([n.code for n in result.notices], [sn.CODE_ANIMA38_COMFY_V1])
        self.assertEqual(sn.NOTICE_MIN_TTL_S[sn.CODE_ANIMA38_COMFY_V1], sn.PRE_GENERATION_NOTICE_TTL_S)   # B14

    def test_contributor_failure_is_isolated(self):
        with mock.patch.object(a38, "plan", side_effect=RuntimeError("a38 broke")), \
                self.assertLogs("generation", "WARNING"):
            payload, error = Anima38Host(ALL_PRESENT)._build_generation_payload(snapshot=True)
        self.assertIsNone(error)
        self.assertNotIn(A38, payload["alwayson_scripts"])
        self.assertIn(PAG, payload["alwayson_scripts"])

    def test_main_chain_payload_carries_the_block_and_its_app_default_provenance(self):
        """(P10 검토 2) Forge 메인 요청은 앱 기본값 블록의 출처를 비공개 키로 백엔드에 넘긴다 — 422 재시도 알림 수준."""
        payload, error = Anima38Host(ALL_PRESENT)._build_generation_payload(snapshot=True)
        self.assertIsNone(error)
        self.assertEqual(payload["alwayson_scripts"][A38], APP_BLOCK)
        self.assertEqual(payload[ap.PROVENANCE_KEY], {A38: ap.PROVENANCE_APP_DEFAULT})
        self.assertIn(A38, ap.main_retry_titles(payload))

    def test_chat_image_edit_uses_the_i2i_rule(self):
        """(A4) 채팅 이미지 편집 = i2i — 'I2I·인페인트에도 적용' 토글을 따른다."""
        caps_all = caps(present=(PAG, SKIM, DD, A38, DORA))
        _model, edit = Anima38Host(caps_all)._chat_generation_snapshot("p", target="i2i")
        self.assertNotIn(A38, edit["alwayson_scripts"])
        _model, t2i = Anima38Host(caps_all)._chat_generation_snapshot("p")
        self.assertIn(A38, t2i["alwayson_scripts"])


class DefaultBehaviourTests(_KindCache):
    """앱 기본값 카드가 바꾸지 않아야 하는 요청 — 비 번들 Anima·비 Anima·모르는 모델, 스냅샷 모름, 없음, Krea2 는 P9 전과
    바이트 단위로 같다. 3.8B v2 는 스크립트가 확인된 Forge(와 Comfy)에서만 블록이 생긴다(의도한 변화)."""

    def setUp(self):
        super().setUp()
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

    def _pre_p9(self, capabilities, guidance, model):
        host = ChainHost(guidance, capabilities)          # Anima38 위젯이 없는 호스트 = P9 전
        host.model_combo = ReadOnlyWidget(model)
        return host

    def test_default_requests_are_byte_identical_except_confirmed_v2(self):
        for model in (ANIMA_MODEL, PLAIN_MODEL, V2_MODEL):
            for label, capabilities, backend in (("unknown", None, BackendType.WEBUI),
                                                 ("known all", ALL_PRESENT, BackendType.WEBUI),
                                                 ("known without Anima38", caps(present=(PAG, SKIM, DD, DORA)),
                                                  BackendType.WEBUI),
                                                 ("comfy", None, BackendType.COMFYUI)):
                for guidance in (GUIDANCE_ON, {"guid_enabled": "false"}):
                    changes = model == V2_MODEL and label in ("known all", "comfy")
                    with self.subTest(model=model, snapshot=label, guidance=guidance):
                        host = Anima38Host(capabilities, guidance, model=model)
                        before = self._payload(self._pre_p9(capabilities, guidance, model), backend)
                        after = self._payload(host, backend)
                        self.assertEqual(after != before, changes)
                        self.assertEqual(host.vue_bridge.showNotification.calls, [])
                        if changes:
                            self.assertEqual(json.loads(after)["alwayson_scripts"][A38], APP_BLOCK)

    def test_turning_negative_off_restores_the_pre_p9_v2_request(self):
        before = self._payload(self._pre_p9(ALL_PRESENT, None, V2_MODEL), BackendType.WEBUI)
        after = self._payload(Anima38Host(ALL_PRESENT, model=V2_MODEL, negative="false"), BackendType.WEBUI)
        self.assertEqual(after, before)


# ── 6. 보조 패스 — 실제 모델과 대조(A7) ────────────────────────────────────────────
class ForgeAuxRequestTests(_KindCache):
    def setUp(self):
        super().setUp()
        _webui(self)

    def _refine_request(self, host, peek_caps, checkpoint):
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
                mock.patch.object(WebUIBackend, "_active_checkpoint", return_value=checkpoint) as active, \
                mock.patch("backends.webui_backend.requests.post", side_effect=post):
            WebUIBackend("http://127.0.0.1:7860").refine(png_b64(), settings)
        return sent[0]["alwayson_scripts"], active

    def test_refine_carries_the_t2i_anima38_block_only_for_the_same_forge_model(self):
        host = Anima38Host(None, guidance={"guid_enabled": "false"})
        scripts, active = self._refine_request(host, caps(present=(A38,)), f"{V2_MODEL} [0a1b2c3d]")
        self.assertEqual(list(scripts), ["SAM3 Mask", A38])
        self.assertEqual(scripts[A38], APP_BLOCK)
        active.assert_called_once_with()
        other, _active = self._refine_request(host, caps(present=(A38,)), ANIMA_MODEL)   # Forge 에 다른 모델
        self.assertEqual(list(other), ["SAM3 Mask"])
        unknown, _active = self._refine_request(host, caps(present=(A38,)), "")          # 확인 실패
        self.assertEqual(list(unknown), ["SAM3 Mask"])
        quiet, _active = self._refine_request(host, None, V2_MODEL)                     # 스냅샷 모름 — 앱 기본값 SKIP
        self.assertEqual(list(quiet), ["SAM3 Mask"])

    def test_no_anima38_block_means_no_checkpoint_lookup(self):
        host = Anima38Host(caps(present=(A38,)), guidance={"guid_enabled": "false"}, model=ANIMA_MODEL)
        scripts, active = self._refine_request(host, caps(present=(A38,)), ANIMA_MODEL)
        self.assertEqual(list(scripts), ["SAM3 Mask"])
        active.assert_not_called()


# ── 7. 모델 종류 미리 읽기(데몬 스레드) ─────────────────────────────────────────────
class PrewarmTests(_KindCache):
    class _Bridge:
        def __init__(self):
            self.pushed = []

        def pushWidgetProperty(self, widget_id, prop, value):
            self.pushed.append((widget_id, prop, copy.deepcopy(value)))

    def setUp(self):
        super().setUp()
        amk.clear_header_cache()
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.paths = {
            "Anima-3.8B-v1.1.safetensors [abc12345]": write_safetensors(os.path.join(self.tmp.name, "v2.safetensors"),
                                                                        V2_HEADER),
            "anima_baseV10.safetensors": write_safetensors(os.path.join(self.tmp.name, "a.safetensors"), ANIMA_HEADER),
            "krea2Anime_v15.safetensors": write_safetensors(os.path.join(self.tmp.name, "k.safetensors"), OTHER_HEADER),
        }
        self.titles = [*self.paths, "remote-only.safetensors"]

    def test_reads_headers_off_the_gui_thread_and_pushes_the_kinds(self):
        host = SimpleNamespace(vue_bridge=self._Bridge())
        thread = aui.prewarm_model_kinds(host, self.titles, self.paths, start=False)
        self.assertTrue(thread.daemon)
        self.assertEqual(host.vue_bridge.pushed, [])                   # 호출한 스레드(GUI)에서는 읽지 않는다
        thread.run()
        expected = {"Anima-3.8B-v1.1.safetensors [abc12345]": "v2", "anima_baseV10.safetensors": "anima",
                    "krea2Anime_v15.safetensors": "other", "remote-only.safetensors": "unknown"}
        self.assertEqual(host.vue_bridge.pushed, [("model_combo", aui.KIND_PROPERTY, expected)])
        self.assertEqual(amk.cached_kinds(), {k: v for k, v in expected.items() if v != "unknown"})
        self.assertEqual(aui.model_kind("krea2Anime_v15.safetensors"), amk.KIND_OTHER)

    def test_stale_results_are_dropped_and_clear_empties_the_property(self):
        host = SimpleNamespace(vue_bridge=self._Bridge())
        older = aui.prewarm_model_kinds(host, self.titles, self.paths, start=False)
        newer = aui.prewarm_model_kinds(host, ["only.safetensors"], {}, start=False)
        newer.run()
        older.run()                                                     # 늦게 끝난 옛 읽기 — 버린다
        self.assertEqual(host.vue_bridge.pushed, [("model_combo", aui.KIND_PROPERTY, {"only.safetensors": "unknown"})])
        pending = aui.prewarm_model_kinds(host, self.titles, self.paths, start=False)
        aui.clear_model_kinds(host)
        pending.run()
        self.assertEqual(host.vue_bridge.pushed[-1], ("model_combo", aui.KIND_PROPERTY, {}))
        self.assertEqual(amk.cached_kinds(), {})

    def test_checkpoint_paths_uses_backend_matched_inventory_entries(self):
        inventory = mock.Mock()
        inventory.entries.return_value = [
            {"runtimeName": "a.safetensors [1]", "path": "C:/m/a.safetensors", "backendAvailable": True},
            {"runtimeName": "b.safetensors", "path": "C:/m/b.safetensors", "backendAvailable": False},
            {"runtimeName": "", "path": "C:/m/c.safetensors", "backendAvailable": True}]
        self.assertEqual(aui.checkpoint_paths(inventory, ["a.safetensors [1]"]), {"a.safetensors [1]": "C:/m/a.safetensors"})
        inventory.entries.assert_called_once_with("checkpoints", backend_items=["a.safetensors [1]"])
        self.assertEqual(aui.checkpoint_paths(None, ["x"]), {})

    def test_connection_slots_start_and_clear_the_prewarm(self):
        source = (REPO / "ui" / "generator_webui.py").read_text(encoding="utf-8")
        loaded = source[source.index("def on_webui_info_loaded"):source.index("def on_webui_info_error")]
        self.assertIn("prewarm_model_kinds(self, models, checkpoint_paths(model_inventory, models))", loaded)
        failed = source[source.index("def on_webui_info_error"):]
        self.assertIn("clear_model_kinds(self)", failed[:failed.index("\n    def ", 10)])



class ComfyAdapterChoicesTests(unittest.TestCase):
    """(P9 리뷰 2) ComfyUI 에서 카드의 v1 어댑터 선택지는 object_info ``ForgeNeoAnimaQwen35Prompt.adapter_name``
    (메타데이터로 확인한 어댑터 — 팩 prompt.adapter_candidates)이다. Forge 스냅샷 choices 가 없어 기본 이름 하나뿐이면
    다른 이름의 v1 어댑터를 고를 수 없었다(컴파일러는 카드 값만 쓴다)."""
    ALT = "text/MyAnima_v1_adapter.safetensors"

    def _object_info(self, *choices):
        return {"ForgeNeoAnimaQwen35Prompt": {"input": {"required": {
            "prompt": ["STRING", {}], "adapter_name": [list(choices)]}}}}

    def test_choices_come_from_the_prompt_node(self):
        self.assertEqual(a38.comfy_adapter_choices(self._object_info(self.ALT, "b.safetensors")),
                         [self.ALT, "b.safetensors"])
        self.assertEqual(a38.comfy_adapter_choices(self._object_info()), [])
        optional = {"ForgeNeoAnimaQwen35Prompt": {"input": {"optional": {"adapter_name": [[self.ALT]]}}}}
        self.assertEqual(a38.comfy_adapter_choices(optional), [self.ALT])
        for missing in (None, {}, {"ForgeNeoAnimaQwen35Prompt": "x"}, "nope"):
            with self.subTest(missing=missing):
                self.assertIsNone(a38.comfy_adapter_choices(missing))       # 팩 없음·모름 — 카드는 Forge 규칙

    def test_pack_placeholder_is_not_offered_as_a_choice(self):
        """(P9 리뷰 2차 1) 팩이 v1 어댑터를 못 찾으면 콤보는 자리표시자 이름 하나뿐이다(파일 아님 — 고르면 실행 때
        FileNotFoundError). 카드에는 빈 목록(= 팩이 못 찾음, 설치 여부 미확인)으로 보낸다. 같은 이름의 진짜 파일은
        text_encoders 목록(CLIPLoader.clip_name — 팩과 같은 folder_paths 목록)에 보이므로 그대로 둔다."""
        fallback = a38.COMFY_ADAPTER_FALLBACK
        placeholder = self._object_info(fallback)
        placeholder["CLIPLoader"] = {"input": {"required": {"clip_name": [["text/qwen35_4b.safetensors"], {}]}}}
        self.assertEqual(a38.comfy_adapter_choices(placeholder), [])
        self.assertTrue(a38.comfy_adapter_is_placeholder([fallback], placeholder))
        no_clip_list = self._object_info(fallback)                         # text_encoders 목록을 모르면 자리표시자로 본다
        self.assertEqual(a38.comfy_adapter_choices(no_clip_list), [])
        real = self._object_info(fallback)
        real["CLIPLoader"] = {"input": {"required": {"clip_name": [["text/qwen35_4b.safetensors", fallback], {}]}}}
        self.assertEqual(a38.comfy_adapter_choices(real), [fallback])
        self.assertFalse(a38.comfy_adapter_is_placeholder([fallback], real))
        # 팩은 찾은 것이 있으면 자리표시자를 붙이지 않는다 — 목록에 그 이름이 섞여 있으면 진짜 파일이다
        self.assertEqual(a38.comfy_adapter_choices(self._object_info(self.ALT, fallback)), [self.ALT, fallback])
        self.assertFalse(a38.comfy_adapter_is_placeholder([self.ALT, fallback], {}))
        from backends.comfyui_backend import ComfyUIBackend
        backend = ComfyUIBackend("http://127.0.0.1:1", workflow_path="")
        with mock.patch.object(backend, "get_object_info", return_value=placeholder):
            self.assertEqual(backend.get_info().anima38_adapters, [])      # 카드는 [기본 이름](+저장값), 미확인

    def test_fallback_name_matches_the_vendored_pack(self):
        prompt = (REPO / "comfy_custom_nodes" / "ai_studio_forge_parity" / "vendor" / "comfyui_anima_3_8b"
                  / "prompt.py").read_text(encoding="utf-8")
        self.assertIn(f'return sorted(candidates) or ["{a38.COMFY_ADAPTER_FALLBACK}"]', prompt)
        self.assertIn('"adapter_name": (adapter_candidates(),)', prompt)

    def test_comfy_get_info_carries_the_adapter_choices(self):
        from backends.comfyui_backend import ComfyUIBackend
        backend = ComfyUIBackend("http://127.0.0.1:1", workflow_path="")
        with mock.patch.object(backend, "get_object_info", return_value=self._object_info(self.ALT)):
            info = backend.get_info()
        self.assertEqual(info.anima38_adapters, [self.ALT])
        with mock.patch.object(backend, "get_object_info", return_value={}):
            self.assertIsNone(backend.get_info().anima38_adapters)
        from backends.base import BackendInfo
        self.assertIsNone(BackendInfo().anima38_adapters)                  # Forge 는 기능 스냅샷 choices 를 쓴다

    def test_info_worker_forwards_the_adapter_choices(self):
        from backends.base import BackendInfo
        from workers.generation_worker import WebUIInfoWorker
        worker = WebUIInfoWorker()
        seen = []
        worker.info_ready.connect(seen.append)
        backend = mock.Mock()
        backend.get_info.return_value = BackendInfo(models=["m"], anima38_adapters=[self.ALT])
        with mock.patch("workers.generation_worker.get_backend", return_value=backend):
            worker.run()
        self.assertEqual(seen[0]["anima38_adapters"], [self.ALT])

    def test_push_sets_the_adapter_widget_property(self):
        pushed = []
        host = SimpleNamespace(vue_bridge=SimpleNamespace(pushWidgetProperty=lambda *a: pushed.append(a)))
        aui.push_comfy_adapters(host, (self.ALT, "", None, "b.safetensors"))
        aui.push_comfy_adapters(host, None)
        aui.push_comfy_adapters(SimpleNamespace(), [self.ALT])             # 브리지 없음 — 조용히
        self.assertEqual(pushed, [(a38.widget_id("adapter"), aui.COMFY_ADAPTER_PROPERTY, [self.ALT, "b.safetensors"]),
                                  (a38.widget_id("adapter"), aui.COMFY_ADAPTER_PROPERTY, None)])

    def test_connection_slots_push_and_clear_the_comfy_adapters(self):
        source = (REPO / "ui" / "generator_webui.py").read_text(encoding="utf-8")
        loaded = source[source.index("def on_webui_info_loaded"):source.index("def on_webui_info_error")]
        self.assertIn("push_comfy_adapters(self, info.get('anima38_adapters'))", loaded)
        failed = source[source.index("def on_webui_info_error"):]
        self.assertIn("push_comfy_adapters(self, None)", failed[:failed.index("\n    def ", 10)])

    def test_card_reads_the_same_property_and_fallback(self):
        ts = TS.read_text(encoding="utf-8")
        self.assertEqual(re.search(r"export const COMFY_ADAPTER_PROPERTY = '([^']+)'", ts).group(1),
                         aui.COMFY_ADAPTER_PROPERTY)
        self.assertEqual(re.search(r"export const COMFY_ADAPTER_FALLBACK = '([^']+)'", ts).group(1),
                         a38.COMFY_ADAPTER_FALLBACK)
        card = CARD.read_text(encoding="utf-8")
        self.assertIn("getProperty(WIDGET_IDS.adapter, COMFY_ADAPTER_PROPERTY", card)


# ── 8. 저장·첫 로드 안내(A5) ─────────────────────────────────────────────────────
class ProxyWiringTests(unittest.TestCase):
    class _Bridge:
        def __init__(self):
            self.registered, self.pushed = {}, []

        def _register_proxy(self, widget_id, proxy):
            self.registered[widget_id] = proxy

        def pushWidgetValue(self, widget_id, value):
            self.pushed.append((widget_id, value))

        def pushWidgetProperty(self, *_args):
            pass

    def test_generator_ui_setup_creates_the_proxies(self):
        source = (REPO / "ui" / "generator_ui_setup.py").read_text(encoding="utf-8")
        self.assertIn("from ui.anima38_ui import init_anima38_proxies", source)
        self.assertIn("self.anima38_widgets = init_anima38_proxies(b)", source)

    def test_init_registers_every_widget_id_with_the_app_defaults(self):
        _webui(self)
        bridge = self._Bridge()
        widgets = aui.init_anima38_proxies(bridge)
        self.assertEqual(set(bridge.registered), {a38.widget_id(key) for key in a38.WIDGET_KEYS})
        defaults = a38.widget_values(a38.APP_T2I_DEFAULTS, False)
        self.assertEqual({key: widgets[key].text() for key in a38.WIDGET_KEYS}, defaults)
        host = ChainHost(None, ALL_PRESENT)
        host.anima38_widgets = widgets
        self.assertEqual(sb.build_sampling_blocks(host, ap.TARGET_T2I).blocks[A38], APP_BLOCK)
        self.assertEqual(aui.get_settings(host), defaults)


class PersistenceTests(unittest.TestCase):
    def test_settings_dict_and_preset_restore(self):
        from ui.generator_settings import _anima38_settings
        from ui.generation_settings_apply import apply_generation_settings
        from tests.test_generation_presets import _host

        host = _host([])
        host.anima38_widgets = a38_widgets(negative="false", enabled="true", strength="0.6", apply_img2img="true")
        saved = _anima38_settings(host)
        self.assertEqual((saved["negative"], saved["enabled"], saved["strength"], saved["apply_img2img"]),
                         ("false", "true", "0.6", "true"))
        other = _host([])
        other.anima38_widgets = a38_widgets()
        apply_generation_settings(other, {a38.SETTINGS_KEY: saved}, only_present=True)
        self.assertEqual(aui.get_settings(other), saved)
        apply_generation_settings(other, {"steps": "30"}, only_present=True)          # 프리셋에 키 없음 — 그대로
        self.assertEqual(aui.get_settings(other), saved)
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
        host.anima38_widgets = a38_widgets(bypass="true")
        host.dora_widgets = {}
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
        self.assertEqual(settings[a38.SETTINGS_KEY]["bypass"], "true")

    def test_old_settings_file_starts_with_app_defaults_and_toasts_once_after_vue_is_ready(self):
        """(A5) load_settings 는 __init__ 에서 Vue 보다 먼저 돈다 — 안내는 쌓아 두고 Vue 준비 신호에서 한 번만."""
        clock = [0.0]
        host = SimpleNamespace(anima38_widgets=a38_widgets(negative="false"),
                               vue_bridge=SimpleNamespace(showNotification=mock.Mock()),
                               _sam_extra_notice_throttle=sn.NoticeThrottle(clock=lambda: clock[0]))
        aui.apply_saved_settings(host, {"steps": "28"}, only_present=False)     # 옛 파일 — 키 없음
        self.assertEqual(aui.current_values(host), (a38.APP_T2I_DEFAULTS, False))
        host.vue_bridge.showNotification.emit.assert_not_called()
        pending = boot_notices.pending_boot_notices(host)
        self.assertEqual([item.key for item in pending], [sn.CODE_ANIMA38_APP_DEFAULT])
        notice = pending[0].build()
        self.assertEqual((notice.code, notice.level), (sn.CODE_ANIMA38_APP_DEFAULT, sn.LEVEL_INFO))
        self.assertIn("Qwen3.5 커넥터", notice.message)
        self.assertEqual(boot_notices.flush_boot_notices(host), 1)
        self.assertEqual(boot_notices.pending_boot_notices(host), [])
        clock[0] = 10 * sn.PRE_GENERATION_NOTICE_TTL_S
        self.assertEqual(boot_notices.flush_boot_notices(host), 0)
        host.vue_bridge.showNotification.emit.assert_called_once()

    def test_saved_key_restores_quietly_and_first_run_has_no_toast(self):
        host = SimpleNamespace(anima38_widgets=a38_widgets())
        aui.apply_saved_settings(host, {a38.SETTINGS_KEY: {"negative": "false"}}, only_present=False)
        self.assertFalse(aui.current_values(host)[0].negative)
        self.assertEqual(boot_notices.pending_boot_notices(host), [])
        bare = SimpleNamespace()                                                  # 위젯 없는 호스트
        aui.apply_saved_settings(bare, {"steps": "1"}, only_present=False)
        self.assertEqual(boot_notices.pending_boot_notices(bare), [])
        self.assertIsNone(aui.current_values(bare))
        self.assertEqual(aui.get_settings(bare), {})


# ── 9. 알림 ─────────────────────────────────────────────────────────────────────
def _info(*infotexts):
    return {"infotexts": list(infotexts)}


class NoticeTests(_KindCache):
    def test_off_hint_names_a_non_anima_model_first(self):
        status = "off: install failed (RuntimeError)"
        self.assertIn("Anima 가 아닙니다", sn.anima38_off_hint(status, kind=amk.KIND_OTHER))
        unknown = sn.anima38_off_hint(status, kind=amk.KIND_UNKNOWN)
        self.assertTrue(unknown.startswith("선택한 모델이 Anima 가 아니면"))
        self.assertIn("VRAM", unknown)
        for kind in (amk.KIND_ANIMA, amk.KIND_V2, None):
            with self.subTest(kind=kind):
                self.assertTrue(sn.anima38_off_hint(status, kind=kind).startswith("Qwen3.5 커넥터 설치 실패"))
        oom = sn.anima38_off_hint("off: install failed (OutOfMemoryError)", kind=amk.KIND_OTHER)
        self.assertTrue(oom.startswith("Qwen3.5 커넥터 설치 실패"))                  # RuntimeError 만 비 Anima 탓

    def test_result_notice_uses_the_result_model_kind(self):
        amk.remember_kinds({"sdxl_base.safetensors [1234abcd]": amk.KIND_OTHER})
        payload = {"alwayson_scripts": {A38: {"args": [{"enabled": True}]}}}
        other = sn.result_notices(_info("x\nSteps: 20, Model: sdxl_base, Anima38: off: install failed (RuntimeError)"),
                                  payload)
        self.assertEqual([n.code for n in other], [sn.CODE_ANIMA38_OFF])
        self.assertIn("Anima 가 아닙니다", other[0].message)
        anima = sn.result_notices(_info("x\nSteps: 20, Model: anima_baseV10, Anima38: off: install failed (RuntimeError)"),
                                  payload)
        self.assertIn("Qwen3.5 커넥터 설치 실패", anima[0].message)
        no_model = sn.result_notices(_info("x\nSteps: 20, Anima38: off: install failed (RuntimeError)"), payload)
        self.assertIn("Qwen3.5 커넥터 설치 실패", no_model[0].message)

    def test_comfy_v1_notice_wording_is_accurate(self):
        """(B8) Comfy 는 파일이 없으면 생성 전에 오류, Forge 는 순정으로 계속 — 동작은 바꾸지 않고 문구만 정확히."""
        notice = sn.anima38_comfy_v1_notice()
        self.assertEqual((notice.code, notice.level, notice.feature), (sn.CODE_ANIMA38_COMFY_V1, sn.LEVEL_INFO, "anima38"))
        self.assertIn("생성 전에 오류", notice.message)
        self.assertIn("Forge 는 순정으로 계속", notice.message)


# ── 10. Vue 거울·레지스트리 ────────────────────────────────────────────────────────
class VueMirrorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.ts = TS.read_text(encoding="utf-8")
        cls.card = CARD.read_text(encoding="utf-8")

    def _record(self, name):
        block = re.search(rf"export const {name}: [^=]+= \{{(.*?)\n\}}", self.ts, re.S).group(1)
        return dict(re.findall(r"(\w+): ('[^']*'|DEFAULT_ADAPTER)", block))

    def test_widget_ids_and_defaults_match_python(self):
        ids = {key: value.strip("'") for key, value in self._record("WIDGET_IDS").items()}
        self.assertEqual(ids, {key: a38.widget_id(key) for key in a38.WIDGET_KEYS})
        adapter = re.search(r"export const DEFAULT_ADAPTER = '([^']+)'", self.ts).group(1)
        self.assertEqual(adapter, a38.DEFAULT_ADAPTER)
        defaults = {key: (adapter if value == "DEFAULT_ADAPTER" else value.strip("'"))
                    for key, value in self._record("DEFAULTS").items()}
        self.assertEqual(defaults, a38.widget_values(a38.APP_T2I_DEFAULTS, False))
        low, high, step = a38.STRENGTH_RANGE
        self.assertRegex(self.ts, rf"STRENGTH_RANGE = \{{ min: {low:g}, max: {high:g}, step: {step:g} \}}")

    def test_kind_property_name_is_shared(self):
        self.assertEqual(re.search(r"export const KIND_PROPERTY = '([^']+)'", self.ts).group(1), aui.KIND_PROPERTY)
        self.assertIn("KIND_PROPERTY", self.card)
        self.assertEqual(re.search(r"KIND_V2, KIND_ANIMA, KIND_OTHER, KIND_UNKNOWN = (.*)", Path(amk.__file__).read_text(
            encoding="utf-8")).group(1), '"v2", "anima", "other", "unknown"')
        self.assertIn("'v2' | 'anima' | 'other' | 'unknown'", self.ts)

    def test_card_binds_every_widget_through_the_shared_ids_and_sits_above_the_guidance_panel(self):
        self.assertNotRegex(self.card, r"""['"]_a38_""")                          # 리터럴 id 대신 WIDGET_IDS
        for key in a38.WIDGET_KEYS:
            with self.subTest(key=key):
                self.assertRegex(self.card, rf"WIDGET_IDS(\.{key}\b|\[key\])")
        self.assertIn('<script setup lang="ts">', self.card)
        app = (REPO / "frontend" / "src" / "App.vue").read_text(encoding="utf-8")
        self.assertIn("import Anima38Card from './components/params/Anima38Card.vue'", app)
        self.assertLess(app.index("<Anima38Card />"), app.index("<AnimaGuidancePanel"))


class RegistryTests(unittest.TestCase):
    def test_p9_gaps_are_closed_and_the_card_is_registered(self):
        entry = reg.SCRIPTS[a38.SCRIPT_NAME]
        self.assertFalse([gap for gap in entry["gaps"] if gap.startswith("P9")])
        self.assertIn("ui/anima38_ui.py:contribute", entry["app"])
        self.assertIn("frontend/src/components/params/Anima38Card.vue", entry["app"])
        for key in ("scripts/anima_3_8b.py", "sam3ext/anima38/"):
            self.assertNotIn("P9", reg.MODULES[key]["gaps"])
        self.assertEqual(ap.PROPAGATION[A38].package, "")
        self.assertTrue(ap.PROPAGATION[A38].model_bound)
        self.assertEqual(reg.KNOWN_DIFFS[(a38.SCRIPT_NAME, "negative", "default")]["app"], a38.APP_T2I_DEFAULTS.negative)


if __name__ == "__main__":
    unittest.main()
