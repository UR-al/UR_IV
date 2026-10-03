"""Anima SPEED 카드 ↔ 위젯 프록시 ↔ 샘플링 블록(ui/anima_speed_ui) — 실제 페이로드 빌더를 거친 요청 모양.

GPU·Forge 없이: 기능 스냅샷은 가짜(``tests.test_alwayson_propagation.caps``), Forge 요청은 가짜 requests.post.
결정(design_speed D3·D4·D5, 계획 R7·R16): 메인 요청(T2I·대기열·XYZ·시드 탐색·채팅·만화 컷, I2I·인페인트·채팅 편집은
카드 'I2I에도 적용' 토글)에만 14키 dict 블록 하나를 싣고, 보조 작업(Refine·단독/배치 SAM3·ADetailer·손 재구성)에는 싣지
않는다. ComfyUI 는 팩 1.7.0 단계 전이라 싣지 않는다.
"""
import copy
import json
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from backends import BackendType
from core import alwayson_propagation as ap
from core import anima_speed as asp
from core import sam_extra_notices as sn
from tests.test_alwayson_propagation import A38, DD, DEGRID, DORA, PAG, SKIM, caps
from tests.test_sampling_blocks import ChainHost, _webui
from ui import anima_speed_ui as aui
from ui import boot_notices
from ui import sampling_blocks as sb

REPO = Path(__file__).resolve().parent.parent
SPEED = asp.SCRIPT_NAME
PRESENT = caps(present=(PAG, SKIM, DD, A38, DORA, DEGRID, SPEED))
# design_speed.md §3.1 의 블록 그대로(키 순서 = ARG_NAMES, 불리언은 JSON 불리언) — 카드를 켜기만 한 값
DESIGN_BLOCK_JSON = ('{"args": [{"enabled": true, "mode": "transition", "threshold": "neo_shift", "preset": "anima", '
                     '"scales": "0.5,1.0", "delta": 0.01, "sigma_divisor": 1.03, "manual_sigmas": "0.7", '
                     '"adaptive": true, "transform": "dct", "spectrum_A": 203.615097, "spectrum_beta": 1.915461, '
                     '"seed": -1, "hires": false}]}')
ON_BLOCK = json.loads(DESIGN_BLOCK_JSON)


def _known(present, absent):
    """known 스냅샷 — ``absent`` 제목은 '없음'으로 확정(``caps`` 는 목록 밖 새 제목을 모름으로 둔다)."""
    from core.sam_extra_capabilities import SamExtraCapabilities, _freeze
    scripts = {title.lower(): {"present": True, "img2img": True} for title in present}
    scripts.update({title.lower(): {"present": False, "img2img": False} for title in absent})
    return SamExtraCapabilities(status="ok", installed=True, scripts=_freeze(scripts))


WITHOUT_SPEED = _known((PAG, SKIM, DD, A38, DORA, DEGRID), (SPEED,))


class _Proxy:
    def __init__(self, value=""):
        self.value = value

    def text(self):
        return self.value

    def setText(self, value):
        self.value = value


def speed_widgets(**values):
    base = asp.widget_values(asp.APP_DEFAULTS)
    base.update({k: str(v) for k, v in values.items()})
    return {key: _Proxy(base[key]) for key in asp.WIDGET_KEYS}


class SpeedHost(ChainHost):
    """실제 메인 체인 + Anima SPEED 카드 위젯(기본: 켬)."""

    def __init__(self, capabilities=None, guidance=None, **values):
        super().__init__(guidance, capabilities)
        values.setdefault("enabled", "true")
        self.anima_speed_widgets = speed_widgets(**values)


# ── 기여자 ─────────────────────────────────────────────────────────────────────
class ContributorTests(unittest.TestCase):
    def setUp(self):
        _webui(self)

    def test_registered_right_after_anima38(self):
        """계획 R7 — Anima38 바로 뒤(Forge 의 Anima 레인 순서), DoRA 앞."""
        from ui.anima38_ui import contribute as anima38
        from ui.dora_infer_mode_ui import contribute as dora
        self.assertEqual(sb.CONTRIBUTORS.index(aui.contribute), sb.CONTRIBUTORS.index(anima38) + 1)
        self.assertEqual(sb.CONTRIBUTORS.index(dora), sb.CONTRIBUTORS.index(aui.contribute) + 1)
        self.assertEqual(sb.CONTRIBUTORS.count(aui.contribute), 1)

    def test_t2i_sends_the_dict_block_as_a_user_value(self):
        result = sb.build_sampling_blocks(SpeedHost(PRESENT), ap.TARGET_T2I)
        self.assertEqual(result.blocks[SPEED], ON_BLOCK)
        self.assertEqual(result.provenance[SPEED], ap.PROVENANCE_USER)
        self.assertEqual(result.notices, [])
        self.assertEqual(list(result.blocks)[-1], SPEED)                       # 가이던스 뒤(DoRA·DeGrid 칸은 없는 호스트)
        self.assertLess(list(result.blocks).index(PAG), list(result.blocks).index(SPEED))
        off = sb.build_sampling_blocks(SpeedHost(PRESENT, enabled="false"), ap.TARGET_T2I)
        self.assertNotIn(SPEED, off.blocks)
        bare = sb.build_sampling_blocks(ChainHost(None, PRESENT), ap.TARGET_T2I)   # 위젯 없는 호스트
        self.assertNotIn(SPEED, bare.blocks)

    def test_unknown_snapshot_sends_the_user_block(self):
        result = sb.build_sampling_blocks(SpeedHost(None), ap.TARGET_T2I)
        self.assertEqual(result.blocks[SPEED], ON_BLOCK)
        self.assertEqual((result.dropped, result.notices), ((), []))

    def test_missing_script_drops_the_block_with_a_warning(self):
        result = sb.build_sampling_blocks(SpeedHost(WITHOUT_SPEED), ap.TARGET_T2I)
        self.assertNotIn(SPEED, result.blocks)
        self.assertEqual(result.dropped, (SPEED,))
        self.assertEqual([(n.code, n.level, n.feature) for n in result.notices],
                         [(sn.CODE_BLOCK_NOT_SENT, sn.LEVEL_WARNING, "speed")])

    def test_i2i_follows_the_card_toggle(self):
        i2i_caps = caps(present=(SPEED,), img2img=(SPEED,))
        off = sb.build_sampling_blocks(SpeedHost(i2i_caps), ap.TARGET_I2I)
        self.assertNotIn(SPEED, off.blocks)
        self.assertEqual([n for n in off.notices if n.feature == "speed"], [])     # 토글 끔은 조용히
        on = sb.build_sampling_blocks(SpeedHost(i2i_caps, apply_img2img="true"), ap.TARGET_I2I)
        self.assertEqual(on.blocks[SPEED], ON_BLOCK)
        gone = sb.build_sampling_blocks(SpeedHost(caps(present=(SPEED,), img2img=()), apply_img2img="true"),
                                        ap.TARGET_I2I)
        self.assertNotIn(SPEED, gone.blocks)

    def test_comfy_and_krea2_build_nothing(self):
        with mock.patch("backends.get_backend_type", return_value=BackendType.COMFYUI):
            comfy = sb.build_sampling_blocks(SpeedHost(None), ap.TARGET_T2I)
        self.assertNotIn(SPEED, comfy.blocks)                                     # 팩 1.7.0 단계 전(R16)
        krea2 = sb.build_sampling_blocks(SpeedHost(PRESENT), ap.TARGET_T2I, krea2=True)
        self.assertNotIn(SPEED, krea2.blocks)

    def test_aux_envelope_carries_the_t2i_block_without_notices(self):
        result = sb.build_sampling_blocks(SpeedHost(PRESENT, scales="1.0", seed="9"), ap.TARGET_AUX)
        self.assertEqual(result.blocks[SPEED]["args"][0]["seed"], 9)
        self.assertEqual(result.notices, [])                                       # 봉투 = 메인 체인, 알림 없음

    def test_settings_only_skip_is_one_info_notice_and_invalid_settings_are_still_sent(self):
        single = sb.build_sampling_blocks(SpeedHost(PRESENT, scales="1.0"), ap.TARGET_T2I)
        self.assertEqual(single.blocks[SPEED]["args"][0]["scales"], "1.0")         # 그래도 싣는다(Forge status 기록)
        self.assertEqual([(n.code, n.level, n.detail) for n in single.notices],
                         [(sn.CODE_SPEED_SKIPPED, sn.LEVEL_INFO, asp.SKIP_NO_COARSE_STAGE)])
        with self.assertLogs("core.anima_speed", "INFO") as logs:
            invalid = sb.build_sampling_blocks(SpeedHost(PRESENT, preset="custom", spectrum_A="-1"), ap.TARGET_T2I)
        self.assertEqual(invalid.blocks[SPEED]["args"][0]["spectrum_A"], -1.0)
        self.assertEqual(invalid.notices, [])                                      # 카드가 미리 알린다
        self.assertIn(asp.VALIDATION_MESSAGES[asp.ERR_SPECTRUM_A], "\n".join(logs.output))

    def test_contributor_failure_is_isolated(self):
        with mock.patch.object(asp, "plan", side_effect=RuntimeError("speed broke")), \
                self.assertLogs("generation", "WARNING"):
            payload, error = SpeedHost(PRESENT)._build_generation_payload(snapshot=True)
        self.assertIsNone(error)
        self.assertNotIn(SPEED, payload["alwayson_scripts"])
        self.assertIn(PAG, payload["alwayson_scripts"])


# ── 실제 진입점 ────────────────────────────────────────────────────────────────
class MainRequestTests(unittest.TestCase):
    def setUp(self):
        _webui(self)

    def test_t2i_payload_carries_the_design_block(self):
        """계획 GA-SP: 기여자를 거친 Forge 요청의 블록 = design_speed.md §3.1 의 JSON(키 순서·JSON 불리언까지)."""
        payload, error = SpeedHost(PRESENT)._build_generation_payload(snapshot=True)
        self.assertIsNone(error)
        self.assertEqual(json.dumps(payload["alwayson_scripts"][SPEED]), DESIGN_BLOCK_JSON)
        self.assertNotIn(ap.PROVENANCE_KEY, payload)                  # 사용자 값 — 출처 키를 적지 않는다
        self.assertEqual(ap.main_retry_titles(payload), ())             # 메인 422 재시도 없음(Rule 문서의 규칙)
        # 확장이 이 블록을 읽은 값 = 카드 값(coerce_settings 의 거울 — 설치된 확장 대조는 tests/test_anima_speed.py)
        self.assertEqual(asp.coerce(payload["alwayson_scripts"][SPEED]["args"]),
                         asp.script_settings(asp.SpeedSettings(enabled=True)))
        user, _ = SpeedHost(PRESENT, mode="respace", threshold="manual", manual_sigmas="0.8, 0.6",
                            scales="0.25,0.5,1", transform="fft", preset="custom", spectrum_A="100",
                            spectrum_beta="2.5", seed="42", hires="true", adaptive="false", delta="0.02",
                            sigma_divisor="1.2")._build_generation_payload(snapshot=True)
        self.assertEqual(user["alwayson_scripts"][SPEED], {"args": [{
            "enabled": True, "mode": "respace", "threshold": "manual", "preset": "custom", "scales": "0.25,0.5,1",
            "delta": 0.02, "sigma_divisor": 1.2, "manual_sigmas": "0.8, 0.6", "adaptive": False, "transform": "fft",
            "spectrum_A": 100.0, "spectrum_beta": 2.5, "seed": 42, "hires": True}]})

    def test_requests_without_the_feature_stay_byte_identical(self):
        """카드가 꺼져 있으면(앱 기본값) SPEED 위젯이 없던 때와 요청이 같다. ComfyUI 는 켜도 같다(팩 1.7.0 단계 전)."""
        for label, capabilities, backend, enabled in (("unknown", None, BackendType.WEBUI, "false"),
                                                      ("known", PRESENT, BackendType.WEBUI, "false"),
                                                      ("comfy off", None, BackendType.COMFYUI, "false"),
                                                      ("comfy on", None, BackendType.COMFYUI, "true")):
            with self.subTest(label), mock.patch("backends.get_backend_type", return_value=backend), \
                    mock.patch("core.comfy_workflow_controls.snapshot_comfy_payload", side_effect=lambda _b, p, _m: p), \
                    mock.patch("ui.comfy_workflow_actions.quality_preset_payload", return_value={}), \
                    mock.patch("core.spectrum_settings.spectrum_payload_from_prefs", return_value={}):
                before, _ = ChainHost(capabilities=capabilities)._build_generation_payload(snapshot=True)
                after, _ = SpeedHost(capabilities, enabled=enabled)._build_generation_payload(snapshot=True)
                self.assertEqual(json.dumps(after, sort_keys=True), json.dumps(before, sort_keys=True))

    def test_image_tab_entry_follows_the_toggle(self):
        i2i_caps = caps(present=(SPEED,), img2img=(SPEED,))
        off = SpeedHost(i2i_caps).apply_alwayson_extensions({"prompt": "x", "init_images": []})
        self.assertNotIn(SPEED, off["alwayson_scripts"])
        on = SpeedHost(i2i_caps, apply_img2img="true").apply_alwayson_extensions({"prompt": "x", "init_images": []})
        self.assertEqual(on["alwayson_scripts"][SPEED], ON_BLOCK)

    def test_i2i_and_inpaint_tabs_send_what_the_toggle_says(self):
        from tests import test_i2i_payload as i2i_tests
        from tests import test_inpaint_payload as inpaint_tests
        from ui.i2i_actions import start_vue_i2i
        from ui.inpaint_actions import start_vue_inpaint

        image = i2i_tests._data_url(i2i_tests._png(64, 64))
        for toggle, expected in (("false", None), ("true", ON_BLOCK)):
            with self.subTest(toggle=toggle):
                host = SpeedHost(PRESENT, apply_img2img=toggle)
                host.vue_bridge = i2i_tests._Bridge()
                i2i_tests._Worker.instances.clear()
                self.assertTrue(start_vue_i2i(host, {"image": image}, worker_factory=i2i_tests._Worker))
                self.assertEqual(i2i_tests._Worker.instances[-1].payload["alwayson_scripts"].get(SPEED), expected)

                host = SpeedHost(PRESENT, apply_img2img=toggle)
                host.vue_bridge = i2i_tests._Bridge()
                inpaint_tests._Worker.instances.clear()
                self.assertTrue(start_vue_inpaint(host, {"image": image, "mask": inpaint_tests.MASK},
                                                  worker_factory=inpaint_tests._Worker))
                # 인페인트도 토글대로 싣는다 — Forge 는 마스크라 건너뛰고 status 에 이유를 남긴다(결과 알림이 말한다)
                self.assertEqual(inpaint_tests._Worker.instances[-1].payload["alwayson_scripts"].get(SPEED), expected)

    def test_chat_generation_and_chat_image_edit(self):
        _model, t2i = SpeedHost(PRESENT)._chat_generation_snapshot("p")
        self.assertEqual(t2i["alwayson_scripts"][SPEED], ON_BLOCK)
        _model, edit = SpeedHost(PRESENT)._chat_generation_snapshot("p", target="i2i")
        self.assertNotIn(SPEED, edit["alwayson_scripts"])
        _model, edit_on = SpeedHost(PRESENT, apply_img2img="true")._chat_generation_snapshot("p", target="i2i")
        self.assertEqual(edit_on["alwayson_scripts"][SPEED], ON_BLOCK)

    def test_every_comic_cut_carries_the_block(self):
        from tests.test_comic_generation import _Document, _panel_payloads
        from ui.creator_actions import CreatorActionsMixin

        class _ComicHost(SpeedHost, CreatorActionsMixin):
            pass

        host = _ComicHost(PRESENT)
        document = _Document(3)
        host._comic_studio = lambda: SimpleNamespace(normalize=lambda _d: document, save=lambda d: d)
        with mock.patch("core.comic_studio.panel_generation_payloads", side_effect=_panel_payloads):
            block = host._comic_t2i_snapshots({"document": {}, "model": "anima.safetensors"}, "comic_generate_all")
        self.assertEqual(len(block["panels"]), 3)
        for entry in block["panels"]:
            self.assertEqual(entry["request"]["alwayson_scripts"][SPEED], ON_BLOCK)

    def test_frozen_xyz_seed_payload_is_regated_when_sent(self):
        """시드 탐색·XYZ 대기열: 클릭 때 실린 블록을 보낼 때의 스냅샷이 '없다'고 하면 빼고 경고한다(모르면 그대로)."""
        host = SpeedHost(PRESENT)
        payload, _ = host._build_generation_payload(snapshot=True)
        self.assertEqual(sb.freeze_sampling_payload(host, payload), [])
        self.assertNotIn(ap.PROVENANCE_KEY, payload)
        frozen = copy.deepcopy(payload)
        host.sam_extra_capabilities = None
        with mock.patch("core.sam_extra_probe.peek_capabilities", return_value=None):
            self.assertEqual(sb.regate_frozen_payload(host, frozen), [])
        self.assertIn(SPEED, frozen["alwayson_scripts"])
        host.sam_extra_capabilities = WITHOUT_SPEED
        notices = sb.regate_frozen_payload(host, frozen)
        self.assertNotIn(SPEED, frozen["alwayson_scripts"])
        self.assertEqual([n.code for n in notices], [sn.CODE_BLOCK_NOT_SENT])


# ── 보조 작업에는 없다(메인 요청 전용 — 계획 D4) ──────────────────────────────────
class AuxPassTests(unittest.TestCase):
    def setUp(self):
        _webui(self)

    def test_hand_repair_snapshot_excludes_the_block(self):
        from ui.hand_reconstruction_actions import _sampling_scripts
        snapshot = {"alwayson_scripts": {PAG: {"args": [1]}, SPEED: ON_BLOCK}}
        for backend in ("webui", "comfyui"):
            with self.subTest(backend=backend):
                self.assertNotIn(SPEED, _sampling_scripts(snapshot, backend=backend))
                self.assertNotIn(SPEED, _sampling_scripts(snapshot, backend=backend, capabilities=PRESENT))

    def test_forge_refine_request_never_carries_the_block(self):
        from backends.webui_backend import WebUIBackend
        from tests.test_webui_aux_propagation import ok, png_b64
        from ui.aux_pass_snapshot import attach_sampling_snapshot

        host = SpeedHost(None, guidance={"guid_enabled": "false"})
        settings = attach_sampling_snapshot(host, {"target": "face"})
        self.assertEqual(settings[ap.SETTINGS_KEY]["blocks"][SPEED], ON_BLOCK)   # 봉투에는 있다(T15)
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
                self.assertNotIn(SPEED, sent[-1]["alwayson_scripts"])
                self.assertIn("SAM3 Mask", sent[-1]["alwayson_scripts"])


# ── 프록시·저장·가져오기 ────────────────────────────────────────────────────────
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
    """앱이 시작할 때 프록시를 만드는지 — 다른 테스트는 모두 ``anima_speed_widgets`` 를 손으로 심은 호스트라, 이 배선이
    빠지면(current_settings()=None → 블록 없음) 카드를 켜도 아무 테스트도 실패하지 않는다."""

    def test_generator_ui_setup_creates_the_proxies(self):
        source = (REPO / "ui" / "generator_ui_setup.py").read_text(encoding="utf-8")
        self.assertIn("from ui.anima_speed_ui import init_speed_proxies", source)
        self.assertIn("self.anima_speed_widgets = init_speed_proxies(b)", source)

    def test_init_registers_every_widget_id_with_the_app_defaults(self):
        bridge = _Bridge()
        widgets = aui.init_speed_proxies(bridge)
        self.assertEqual(set(bridge.registered), {asp.widget_id(key) for key in asp.WIDGET_KEYS})
        for key, proxy in widgets.items():
            self.assertIs(bridge.registered[asp.widget_id(key)], proxy)
        defaults = asp.widget_values(asp.APP_DEFAULTS)
        self.assertEqual({key: widgets[key].text() for key in asp.WIDGET_KEYS}, defaults)
        self.assertEqual(dict(bridge.pushed), {asp.widget_id(k): v for k, v in defaults.items()})
        host = ChainHost(None, PRESENT)
        host.anima_speed_widgets = widgets
        self.assertEqual(aui.current_settings(host), asp.APP_DEFAULTS)
        widgets["enabled"].setText("true")
        with mock.patch("backends.get_backend_type", return_value=BackendType.WEBUI):
            self.assertEqual(sb.build_sampling_blocks(host, ap.TARGET_T2I).blocks[SPEED], ON_BLOCK)
        self.assertEqual(aui.get_settings(host)["enabled"], "true")


class PersistenceTests(unittest.TestCase):
    def test_settings_dict_and_preset_restore(self):
        from tests.test_generation_presets import _host
        from ui.generation_settings_apply import apply_generation_settings
        from ui.generator_settings import _speed_settings

        host = _host([])
        host.anima_speed_widgets = speed_widgets(enabled="true", mode="respace", threshold="manual",
                                                 manual_sigmas="0.8,0.6", scales="0.25,0.5,1", delta="0.002",
                                                 seed="12", hires="true", apply_img2img="true", transform="dwt")
        saved = _speed_settings(host)
        self.assertEqual(saved, {**asp.widget_values(asp.APP_DEFAULTS), "enabled": "true", "mode": "respace",
                                 "threshold": "manual", "manual_sigmas": "0.8,0.6", "scales": "0.25,0.5,1",
                                 "delta": "0.002", "seed": "12", "hires": "true", "apply_img2img": "true",
                                 "transform": "dwt"})
        other = _host([])
        other.anima_speed_widgets = speed_widgets()
        apply_generation_settings(other, {asp.SETTINGS_KEY: saved}, only_present=True)
        self.assertEqual(aui.get_settings(other), saved)
        apply_generation_settings(other, {"steps": "30"}, only_present=True)    # 프리셋에 키가 없으면 그대로
        self.assertEqual(aui.get_settings(other), saved)

    def test_old_settings_file_starts_with_app_defaults_quietly(self):
        host = SimpleNamespace(anima_speed_widgets=speed_widgets(enabled="true", mode="respace"))
        aui.apply_saved_settings(host, {"steps": "28"}, only_present=False)
        self.assertEqual(aui.current_settings(host), asp.APP_DEFAULTS)
        self.assertEqual(boot_notices.pending_boot_notices(host), [])          # 동작이 같아 안내 없음
        aui.apply_saved_settings(SimpleNamespace(), {asp.SETTINGS_KEY: {"enabled": "true"}}, only_present=False)

    def test_saved_values_are_normalized_on_restore(self):
        host = SimpleNamespace(anima_speed_widgets=speed_widgets())
        aui.apply_saved_settings(host, {asp.SETTINGS_KEY: {
            "enabled": "1", "mode": "RESPACE", "threshold": "bogus", "preset": "Krea-2", "delta": "abc",
            "seed": "-7", "scales": " 0.5 , 1 ", "spectrum_A": "nan", "sigma_divisor": "1.5"}}, only_present=True)
        self.assertEqual(aui.current_settings(host), asp.SpeedSettings(
            enabled=True, mode="respace", preset="krea-2", scales="0.5 , 1", seed=-1, sigma_divisor=1.5))

    def test_build_settings_dict_and_preset_keys(self):
        from core.generation_presets import GENERATION_KEYS
        from ui.generator_settings import SettingsMixin

        self.assertIn(asp.SETTINGS_KEY, GENERATION_KEYS)
        source = (REPO / "ui" / "generator_settings.py").read_text(encoding="utf-8")
        self.assertIn('"anima_speed_settings": _speed_settings(self)', source)

        class _Host(SettingsMixin):
            def __getattr__(self, name):
                if name.startswith("__"):
                    raise AttributeError(name)
                value = mock.MagicMock(name=name)
                setattr(self, name, value)
                return value

        host = _Host()
        host.anima_speed_widgets = speed_widgets(enabled="true", seed="5")
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
        self.assertEqual((settings[asp.SETTINGS_KEY]["enabled"], settings[asp.SETTINGS_KEY]["seed"]), ("true", "5"))


class ApplyInfotextTests(unittest.TestCase):
    """가져오기(계획 X3c — sam_extra_import 가 연결) — 확장 붙여 넣기 규칙: 'Anima SPEED' 키가 있으면 켜고 그 요약의 값."""

    SUMMARY = ("mode=respace; threshold=manual; preset=flux; scales=0.25,0.5,1; delta=0.02; divisor=1.1; "
               "manual=0.8,0.6; adaptive=False; transform=fft; seed=7; hires=True")

    def test_summary_values_come_back_and_the_app_only_toggle_stays(self):
        host = SimpleNamespace(anima_speed_widgets=speed_widgets(apply_img2img="true", spectrum_A="50"))
        restored = aui.apply_infotext(host, {"Steps": 28, asp.INFOTEXT_KEY: self.SUMMARY})
        self.assertEqual(restored, asp.SpeedSettings(
            enabled=True, mode="respace", threshold="manual", preset="flux", scales="0.25,0.5,1", delta=0.02,
            sigma_divisor=1.1, manual_sigmas="0.8,0.6", adaptive=False, transform="fft", spectrum_A=50.0,
            seed=7, hires=True, apply_img2img=True))
        self.assertEqual(aui.current_settings(host), restored)
        off = aui.apply_infotext(host, {"Steps": 28})                          # 키 없음 → 끈다(확장 붙여 넣기와 같다)
        self.assertFalse(off.enabled)
        self.assertEqual((off.mode, off.apply_img2img), ("respace", True))
        self.assertIsNone(aui.apply_infotext(SimpleNamespace(), {asp.INFOTEXT_KEY: self.SUMMARY}))


if __name__ == "__main__":
    unittest.main()
