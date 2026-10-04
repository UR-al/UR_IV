"""Colorcraft 카드 ↔ 위젯 프록시 ↔ 샘플링 블록(ui/colorcraft_ui) — 실제 페이로드 빌더를 거친 요청 모양.

GPU·Forge 없이: 기능 스냅샷은 가짜(``tests.test_alwayson_propagation.caps``), Forge 요청은 가짜 requests.post.
결정(design_colorcraft §3·§4·§5.1, 계획 R7·R16): 메인 요청(T2I·대기열·XYZ·시드 탐색·채팅·만화 컷, I2I·인페인트·채팅 편집은
카드 'I2I·인페인트에도 적용' 토글)에만 첫 인자 하나(확장 infotext 문자열)의 블록을 싣고, 보조 작업(Refine·단독/배치 SAM3·
ADetailer·손 재구성)에는 싣지 않는다. 돌 항목이 없으면 싣지 않는다. ComfyUI 도 같은 블록이다(계획 C6 — 컴파일러가 앱
노드 팩 1.7.0 의 노드로).
"""
import copy
import json
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from backends import BackendType
from core import alwayson_propagation as ap
from core import colorcraft as cc
from core import sam_extra_notices as sn
from tests.test_alwayson_propagation import A38, DD, DEGRID, DORA, PAG, SKIM, caps
from tests.test_sampling_blocks import ChainHost, _webui
from ui import boot_notices
from ui import colorcraft_ui as cui
from ui import sampling_blocks as sb

REPO = Path(__file__).resolve().parent.parent
CC = cc.SCRIPT_NAME
PRESENT = caps(present=(PAG, SKIM, DD, A38, DORA, DEGRID, CC))
# design_colorcraft.md §4.1 의 블록 그대로 — 확장 spec.to_infotext 가 그 설정으로 만든 문자열(10-03 CPU 실행)
DESIGN_STRING = ("v1;mods=I,III;masking=1;masks=M1;combos=;I.kind=Luma;I.mask=M1;I.exposure=0.3;I.tone_compression=-0.2;"
                 "III.kind=Punch;III.pass=Hires;III.clarity=0.5;III.sharpness=0.25;M1.mask_mode=lows;M1.mask_hardness=2;"
                 "M1.blur=8")
DESIGN_BLOCK_JSON = json.dumps({"args": [DESIGN_STRING]})
# 그 설정을 카드에서 만든 값(편집기가 쓰는 경로 — 상태 문자열은 기본값과 다른 경로만)
DESIGN_CARD = {"I.kind": "Luma", "I.mask": "M1", "I.exposure": 0.3, "I.tone_compression": -0.2,
               "III.active": True, "III.kind": "Punch", "III.pass": "Hires", "III.clarity": 0.5, "III.sharpness": 0.25,
               "M1.mask_mode": "lows", "M1.mask_hardness": 2.0, "M1.blur": 8.0}
EXPOSURE = {"I.exposure": 0.3}
ON_BLOCK = {"args": ["v1;mods=I;I.exposure=0.3"]}


def card_state(paths=None) -> str:
    """편집기 경로 값 → 정규 상태 문자열(``_cc_state``)."""
    config = cc.default_config()
    for path, value in (paths or {}).items():
        tag, _, name = path.partition(".")
        node = (config.modifiers if tag in cc.MODIFIER_TAGS else config.leaves if tag in cc.MASK_TAGS
                else config.combos)
        index = (cc.MODIFIER_TAGS if tag in cc.MODIFIER_TAGS else cc.MASK_TAGS if tag in cc.MASK_TAGS
                 else cc.COMBO_TAGS).index(tag)
        node[index][name] = value
    return cc.dump_state(config)


def _known(present, absent):
    """known 스냅샷 — ``absent`` 제목은 '없음'으로 확정(``caps`` 는 목록 밖 새 제목을 모름으로 둔다)."""
    from core.sam_extra_capabilities import SamExtraCapabilities, _freeze
    scripts = {title.lower(): {"present": True, "img2img": True} for title in present}
    scripts.update({title.lower(): {"present": False, "img2img": False} for title in absent})
    return SamExtraCapabilities(status="ok", installed=True, scripts=_freeze(scripts))


WITHOUT_CC = _known((PAG, SKIM, DD, A38, DORA, DEGRID), (CC,))


class _Proxy:
    def __init__(self, value=""):
        self.value = value

    def text(self):
        return self.value

    def setText(self, value):
        self.value = value


def cc_widgets(paths=None, **values):
    base = cc.widget_values(cc.APP_DEFAULTS)
    base["state"] = card_state(paths)
    base.update({k: str(v) for k, v in values.items()})
    return {key: _Proxy(base[key]) for key in cc.WIDGET_KEYS}


class ColorcraftHost(ChainHost):
    """실제 메인 체인 + Colorcraft 카드 위젯(기본: 켬, 수정자 I 노출 0.3)."""

    def __init__(self, capabilities=None, guidance=None, paths=None, **values):
        super().__init__(guidance, capabilities)
        values.setdefault("enabled", "true")
        self.colorcraft_widgets = cc_widgets(EXPOSURE if paths is None else paths, **values)


# ── 기여자 ─────────────────────────────────────────────────────────────────────
class ContributorTests(unittest.TestCase):
    def setUp(self):
        _webui(self)

    def test_registered_after_the_guidance_and_before_anima38(self):
        """계획 R7 — 가이던스 바로 뒤(Forge 에서 Anima Optimal Scale 바로 아래), Anima38 앞."""
        from ui.anima38_ui import contribute as anima38
        self.assertEqual(sb.CONTRIBUTORS.index(cui.contribute), sb.CONTRIBUTORS.index(sb._anima_guidance) + 1)
        self.assertEqual(sb.CONTRIBUTORS.index(anima38), sb.CONTRIBUTORS.index(cui.contribute) + 1)
        self.assertEqual(sb.CONTRIBUTORS.count(cui.contribute), 1)

    def test_t2i_sends_the_infotext_string_as_a_user_value(self):
        result = sb.build_sampling_blocks(ColorcraftHost(PRESENT), ap.TARGET_T2I)
        self.assertEqual(result.blocks[CC], ON_BLOCK)
        self.assertEqual(result.provenance[CC], ap.PROVENANCE_USER)
        self.assertEqual(result.notices, [])
        self.assertEqual(list(result.blocks)[-1], CC)                         # 가이던스 뒤(Anima38 칸은 없는 호스트)
        self.assertLess(list(result.blocks).index(DD), list(result.blocks).index(CC))
        off = sb.build_sampling_blocks(ColorcraftHost(PRESENT, enabled="false"), ap.TARGET_T2I)
        self.assertNotIn(CC, off.blocks)
        bare = sb.build_sampling_blocks(ChainHost(None, PRESENT), ap.TARGET_T2I)   # 위젯 없는 호스트
        self.assertNotIn(CC, bare.blocks)

    def test_nothing_to_run_sends_nothing(self):
        for label, paths in (("defaults", {}), ("zero strength", {"I.exposure": 0.3, "I.strength": 0.0}),
                             ("inactive", {"I.active": False, "I.exposure": 0.3})):
            with self.subTest(label):
                result = sb.build_sampling_blocks(ColorcraftHost(PRESENT, paths=paths), ap.TARGET_T2I)
                self.assertNotIn(CC, result.blocks)
                self.assertEqual([n for n in result.notices if n.feature == "colorcraft"], [])

    def test_hires_only_modifiers_still_send_for_t2i(self):
        hires = {"I.pass": "Hires", "I.exposure": 0.3}
        result = sb.build_sampling_blocks(ColorcraftHost(PRESENT, paths=hires), ap.TARGET_T2I)
        self.assertEqual(result.blocks[CC], {"args": ["v1;mods=I;I.pass=Hires;I.exposure=0.3"]})
        i2i = sb.build_sampling_blocks(ColorcraftHost(caps(present=(CC,), img2img=(CC,)), paths=hires, img2img="true"),
                                       ap.TARGET_I2I)
        self.assertNotIn(CC, i2i.blocks)                                          # img2img 에는 Hires 패스가 없다

    def test_unknown_snapshot_sends_the_user_block(self):
        result = sb.build_sampling_blocks(ColorcraftHost(None), ap.TARGET_T2I)
        self.assertEqual(result.blocks[CC], ON_BLOCK)
        self.assertEqual((result.dropped, result.notices), ((), []))

    def test_missing_script_drops_the_block_with_a_warning(self):
        result = sb.build_sampling_blocks(ColorcraftHost(WITHOUT_CC), ap.TARGET_T2I)
        self.assertNotIn(CC, result.blocks)
        self.assertEqual(result.dropped, (CC,))
        self.assertEqual([(n.code, n.level, n.feature) for n in result.notices],
                         [(sn.CODE_BLOCK_NOT_SENT, sn.LEVEL_WARNING, "colorcraft")])

    def test_i2i_follows_the_card_toggle(self):
        i2i_caps = caps(present=(CC,), img2img=(CC,))
        off = sb.build_sampling_blocks(ColorcraftHost(i2i_caps), ap.TARGET_I2I)
        self.assertNotIn(CC, off.blocks)
        self.assertEqual([n for n in off.notices if n.feature == "colorcraft"], [])     # 토글 끔은 조용히
        on = sb.build_sampling_blocks(ColorcraftHost(i2i_caps, img2img="true"), ap.TARGET_I2I)
        self.assertEqual(on.blocks[CC], ON_BLOCK)
        gone = sb.build_sampling_blocks(ColorcraftHost(caps(present=(CC,), img2img=()), img2img="true"), ap.TARGET_I2I)
        self.assertNotIn(CC, gone.blocks)

    def test_comfy_carries_the_forge_block_and_krea2_nothing(self):
        with mock.patch("backends.get_backend_type", return_value=BackendType.COMFYUI):
            comfy = sb.build_sampling_blocks(ColorcraftHost(None), ap.TARGET_T2I)
            comfy_i2i = sb.build_sampling_blocks(ColorcraftHost(None), ap.TARGET_I2I)
            empty = sb.build_sampling_blocks(ColorcraftHost(None, paths={}), ap.TARGET_T2I)
        self.assertEqual(comfy.blocks[CC], ON_BLOCK)                                 # 계획 C6
        self.assertEqual(comfy.provenance[CC], ap.PROVENANCE_USER)
        self.assertNotIn(CC, comfy_i2i.blocks)                                       # 'I2I·인페인트에도 적용' 끔
        self.assertNotIn(CC, empty.blocks)                                           # 돌 항목 없음
        krea2 = sb.build_sampling_blocks(ColorcraftHost(PRESENT), ap.TARGET_T2I, krea2=True)
        self.assertNotIn(CC, krea2.blocks)

    def test_aux_envelope_carries_the_t2i_block_without_notices(self):
        result = sb.build_sampling_blocks(ColorcraftHost(PRESENT), ap.TARGET_AUX)
        self.assertEqual(result.blocks[CC], ON_BLOCK)
        self.assertEqual(result.notices, [])                                       # 봉투 = 메인 체인, 알림 없음

    def test_contributor_failure_is_isolated(self):
        with mock.patch.object(cc, "plan", side_effect=RuntimeError("colorcraft broke")), \
                self.assertLogs("generation", "WARNING"):
            payload, error = ColorcraftHost(PRESENT)._build_generation_payload(snapshot=True)
        self.assertIsNone(error)
        self.assertNotIn(CC, payload["alwayson_scripts"])
        self.assertIn(PAG, payload["alwayson_scripts"])


# ── 실제 진입점 ────────────────────────────────────────────────────────────────
class MainRequestTests(unittest.TestCase):
    def setUp(self):
        _webui(self)

    def test_t2i_payload_carries_the_design_block(self):
        """계획 GA-CC: 기여자를 거친 Forge 요청의 블록 = design_colorcraft.md §4.1 의 JSON(첫 인자 하나, 바이트 그대로)."""
        payload, error = ColorcraftHost(PRESENT, paths=DESIGN_CARD, masking="true")._build_generation_payload(
            snapshot=True)
        self.assertIsNone(error)
        self.assertEqual(json.dumps(payload["alwayson_scripts"][CC]), DESIGN_BLOCK_JSON)
        self.assertNotIn(ap.PROVENANCE_KEY, payload)                  # 사용자 값 — 출처 키를 적지 않는다
        self.assertEqual(ap.main_retry_titles(payload), ())             # 메인 422 재시도 없음(Rule 문서의 규칙)
        # 확장이 이 블록을 읽은 설정 = 카드 설정(parse_own 의 거울 — 설치된 확장 대조는 tests/test_colorcraft_spec_parity.py)
        sent = cc.parse_script_block(payload["alwayson_scripts"][CC])
        card = cc.config_of(cui.current_settings(ColorcraftHost(PRESENT, paths=DESIGN_CARD, masking="true")))
        for is_hires in (False, True):
            with self.subTest(is_hires=is_hires):
                self.assertEqual(cc.build_chain(sent, is_hires), cc.build_chain(card, is_hires))
        self.assertEqual((cc.build_chain(sent, False).tags, cc.build_chain(sent, True).tags), (["I"], ["III"]))

    def test_requests_without_the_feature_stay_byte_identical(self):
        """카드가 꺼져 있거나(앱 기본값) 돌 항목이 없으면 Colorcraft 위젯이 없던 때와 요청이 같다. ComfyUI 에서 켜면 그
        요청에 Forge 와 같은 블록 하나만 더해진다(계획 C6)."""
        for label, capabilities, backend, enabled, paths in (
                ("unknown", None, BackendType.WEBUI, "false", EXPOSURE),
                ("known", PRESENT, BackendType.WEBUI, "false", EXPOSURE),
                ("nothing to run", PRESENT, BackendType.WEBUI, "true", {}),
                ("comfy off", None, BackendType.COMFYUI, "false", EXPOSURE),
                ("comfy on", None, BackendType.COMFYUI, "true", EXPOSURE)):
            with self.subTest(label), mock.patch("backends.get_backend_type", return_value=backend), \
                    mock.patch("core.comfy_workflow_controls.snapshot_comfy_payload", side_effect=lambda _b, p, _m: p), \
                    mock.patch("ui.comfy_workflow_actions.quality_preset_payload", return_value={}), \
                    mock.patch("core.spectrum_settings.spectrum_payload_from_prefs", return_value={}):
                before, _ = ChainHost(capabilities=capabilities)._build_generation_payload(snapshot=True)
                after, _ = ColorcraftHost(capabilities, enabled=enabled, paths=paths)._build_generation_payload(
                    snapshot=True)
                if label == "comfy on":
                    before = copy.deepcopy(before)
                    before["alwayson_scripts"][CC] = ON_BLOCK
                self.assertEqual(json.dumps(after, sort_keys=True), json.dumps(before, sort_keys=True))

    def test_image_tab_entry_follows_the_toggle(self):
        i2i_caps = caps(present=(CC,), img2img=(CC,))
        off = ColorcraftHost(i2i_caps).apply_alwayson_extensions({"prompt": "x", "init_images": []})
        self.assertNotIn(CC, off["alwayson_scripts"])
        on = ColorcraftHost(i2i_caps, img2img="true").apply_alwayson_extensions({"prompt": "x", "init_images": []})
        self.assertEqual(on["alwayson_scripts"][CC], ON_BLOCK)

    def test_chat_generation_and_chat_image_edit(self):
        _model, t2i = ColorcraftHost(PRESENT)._chat_generation_snapshot("p")
        self.assertEqual(t2i["alwayson_scripts"][CC], ON_BLOCK)
        _model, edit = ColorcraftHost(PRESENT)._chat_generation_snapshot("p", target="i2i")
        self.assertNotIn(CC, edit["alwayson_scripts"])
        _model, edit_on = ColorcraftHost(PRESENT, img2img="true")._chat_generation_snapshot("p", target="i2i")
        self.assertEqual(edit_on["alwayson_scripts"][CC], ON_BLOCK)

    def test_frozen_xyz_seed_payload_is_regated_when_sent(self):
        """시드 탐색·XYZ 대기열: 클릭 때 실린 블록을 보낼 때의 스냅샷이 '없다'고 하면 빼고 경고한다(모르면 그대로)."""
        host = ColorcraftHost(PRESENT)
        payload, _ = host._build_generation_payload(snapshot=True)
        self.assertEqual(sb.freeze_sampling_payload(host, payload), [])
        frozen = copy.deepcopy(payload)
        host.sam_extra_capabilities = None
        with mock.patch("core.sam_extra_probe.peek_capabilities", return_value=None):
            self.assertEqual(sb.regate_frozen_payload(host, frozen), [])
        self.assertIn(CC, frozen["alwayson_scripts"])
        host.sam_extra_capabilities = WITHOUT_CC
        notices = sb.regate_frozen_payload(host, frozen)
        self.assertNotIn(CC, frozen["alwayson_scripts"])
        self.assertEqual([n.code for n in notices], [sn.CODE_BLOCK_NOT_SENT])


# ── 보조 작업에는 없다(메인 요청 전용) ─────────────────────────────────────────────
class AuxPassTests(unittest.TestCase):
    def setUp(self):
        _webui(self)

    def test_hand_repair_snapshot_excludes_the_block(self):
        from ui.hand_reconstruction_actions import _sampling_scripts
        snapshot = {"alwayson_scripts": {PAG: {"args": [1]}, CC: ON_BLOCK}}
        for backend in ("webui", "comfyui"):
            with self.subTest(backend=backend):
                self.assertNotIn(CC, _sampling_scripts(snapshot, backend=backend))
                self.assertNotIn(CC, _sampling_scripts(snapshot, backend=backend, capabilities=PRESENT))

    def test_forge_refine_request_never_carries_the_block(self):
        from backends.webui_backend import WebUIBackend
        from tests.test_webui_aux_propagation import ok, png_b64
        from ui.aux_pass_snapshot import attach_sampling_snapshot

        host = ColorcraftHost(None, guidance={"guid_enabled": "false"})
        settings = attach_sampling_snapshot(host, {"target": "face"})
        self.assertEqual(settings[ap.SETTINGS_KEY]["blocks"][CC], ON_BLOCK)       # 봉투에는 있다(T15)
        sent = []

        def post(url, json=None, **_kw):
            sent.append(copy.deepcopy(json))
            return ok()

        for peek in (None, PRESENT):
            with self.subTest(peek=peek), \
                    mock.patch("core.sam_extra_probe.peek_capabilities", return_value=peek), \
                    mock.patch("core.forge_modules.resolve_sam3_checkpoint", side_effect=lambda name: name), \
                    mock.patch("backends.webui_backend.requests.post", side_effect=post):
                WebUIBackend("http://127.0.0.1:7860").refine(png_b64(), settings)
                self.assertNotIn(CC, sent[-1]["alwayson_scripts"])
                self.assertIn("SAM3 Mask", sent[-1]["alwayson_scripts"])


# ── 프록시·저장 ────────────────────────────────────────────────────────────────
class _Bridge:
    def __init__(self):
        self.registered, self.pushed, self.properties = {}, [], []

    def _register_proxy(self, widget_id, proxy):
        self.registered[widget_id] = proxy

    def pushWidgetValue(self, widget_id, value):
        self.pushed.append((widget_id, value))

    def pushWidgetProperty(self, *args):
        self.properties.append(args)


class ProxyWiringTests(unittest.TestCase):
    """앱이 시작할 때 프록시를 만드는지 — 다른 테스트는 모두 ``colorcraft_widgets`` 를 손으로 심은 호스트라, 이 배선이
    빠지면(current_settings()=None → 블록 없음) 카드를 켜도 아무 테스트도 실패하지 않는다."""

    def test_generator_ui_setup_creates_the_proxies(self):
        source = (REPO / "ui" / "generator_ui_setup.py").read_text(encoding="utf-8")
        self.assertIn("from ui.colorcraft_ui import init_colorcraft_proxies", source)
        self.assertIn("self.colorcraft_widgets = init_colorcraft_proxies(b)", source)

    def test_init_registers_every_widget_id_with_the_app_defaults(self):
        bridge = _Bridge()
        widgets = cui.init_colorcraft_proxies(bridge)
        self.assertEqual(set(bridge.registered), {"_cc_enabled", "_cc_masking", "_cc_img2img", "_cc_state"})
        for key, proxy in widgets.items():
            self.assertIs(bridge.registered[cc.widget_id(key)], proxy)
        defaults = cc.widget_values(cc.APP_DEFAULTS)
        self.assertEqual(defaults, {"enabled": "false", "masking": "false", "img2img": "false", "state": ""})
        self.assertEqual({key: widgets[key].text() for key in cc.WIDGET_KEYS}, defaults)
        # LineEditProxy 는 같은 값이면 보내지 않는다 — 빈 상태 문자열('' = 모두 기본값)은 처음 값 그대로
        self.assertEqual(dict(bridge.pushed), {cc.widget_id(k): v for k, v in defaults.items() if v})
        host = ChainHost(None, PRESENT)
        host.colorcraft_widgets = widgets
        self.assertEqual(cui.current_settings(host), cc.APP_DEFAULTS)
        widgets["enabled"].setText("true")
        with mock.patch("backends.get_backend_type", return_value=BackendType.WEBUI):
            self.assertNotIn(CC, sb.build_sampling_blocks(host, ap.TARGET_T2I).blocks)   # 기본값(모두 0) — 돌 항목 없음
            widgets["state"].setText(card_state(EXPOSURE))
            self.assertEqual(sb.build_sampling_blocks(host, ap.TARGET_T2I).blocks[CC], ON_BLOCK)
        self.assertEqual(cui.get_settings(host)["enabled"], "true")


class PersistenceTests(unittest.TestCase):
    def test_settings_dict_and_preset_restore(self):
        from tests.test_generation_presets import _host
        from ui.generation_settings_apply import apply_generation_settings
        from ui.generator_settings import _colorcraft_settings

        host = _host([])
        host.colorcraft_widgets = cc_widgets(DESIGN_CARD, enabled="true", masking="true", img2img="true")
        saved = _colorcraft_settings(host)
        self.assertEqual(saved, {"enabled": "true", "masking": "true", "img2img": "true",
                                 "state": card_state(DESIGN_CARD)})
        self.assertTrue(saved["state"].startswith('{"v":1,"I.kind":"Luma"'), saved["state"])
        other = _host([])
        other.colorcraft_widgets = cc_widgets()
        apply_generation_settings(other, {cc.SETTINGS_KEY: saved}, only_present=True)
        self.assertEqual(cui.get_settings(other), saved)
        apply_generation_settings(other, {"steps": "30"}, only_present=True)    # 프리셋에 키가 없으면 그대로
        self.assertEqual(cui.get_settings(other), saved)

    def test_old_settings_file_starts_with_app_defaults_quietly(self):
        host = SimpleNamespace(colorcraft_widgets=cc_widgets(EXPOSURE, enabled="true", masking="true"))
        cui.apply_saved_settings(host, {"steps": "28"}, only_present=False)
        self.assertEqual(cui.current_settings(host), cc.APP_DEFAULTS)
        self.assertEqual(boot_notices.pending_boot_notices(host), [])          # 동작이 같아 안내 없음
        cui.apply_saved_settings(SimpleNamespace(), {cc.SETTINGS_KEY: {"enabled": "true"}}, only_present=False)

    def test_saved_values_are_normalized_on_restore(self):
        host = SimpleNamespace(colorcraft_widgets=cc_widgets())
        cui.apply_saved_settings(host, {cc.SETTINGS_KEY: {
            "enabled": "1", "masking": "yes", "state": '{"v": 1, "I.exposure": 0.30, "Z.bogus": 1, "II.kind": "punch"}'}},
            only_present=True)
        self.assertEqual(cui.get_settings(host), {"enabled": "true", "masking": "true", "img2img": "false",
                                                  "state": card_state({"I.exposure": 0.3, "II.kind": "Punch"})})
        before = cui.get_settings(host)
        with self.assertLogs("core.colorcraft", "WARNING"):                     # 읽을 수 없는 상태 — 이전 값을 둔다
            cui.apply_saved_settings(host, {cc.SETTINGS_KEY: {"state": "{not json"}}, only_present=True)
        self.assertEqual(cui.get_settings(host)["state"], cc.DEFAULT_STATE)     # 저장값 위의 칸이 기본값(APP_DEFAULTS 위)
        self.assertEqual(before["enabled"], "true")

    def test_build_settings_dict_and_preset_keys(self):
        from core.generation_presets import GENERATION_KEYS
        from ui.generator_settings import SettingsMixin

        self.assertIn(cc.SETTINGS_KEY, GENERATION_KEYS)
        source = (REPO / "ui" / "generator_settings.py").read_text(encoding="utf-8")
        self.assertIn('"colorcraft_settings": _colorcraft_settings(self)', source)

        class _Host(SettingsMixin):
            def __getattr__(self, name):
                if name.startswith("__"):
                    raise AttributeError(name)
                value = mock.MagicMock(name=name)
                setattr(self, name, value)
                return value

        host = _Host()
        host.colorcraft_widgets = cc_widgets(EXPOSURE, enabled="true")
        host.dora_widgets = None
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
        self.assertEqual((settings[cc.SETTINGS_KEY]["enabled"], settings[cc.SETTINGS_KEY]["state"]),
                         ("true", card_state(EXPOSURE)))


if __name__ == "__main__":
    unittest.main()
