"""VAE DeGrid 카드 ↔ 위젯 프록시 ↔ 샘플링 블록(ui/vae_degrid_ui) — 실제 페이로드 빌더를 거친 요청 모양.

GPU·Forge 없이: 기능 스냅샷은 가짜(``tests.test_alwayson_propagation.caps``), Forge 요청은 가짜 requests.post.
결정(D1): DeGrid 는 메인 요청(T2I·대기열·XYZ·시드 탐색·채팅·만화 컷, I2I·인페인트·채팅 편집은 카드 토글)에만 싣고
보조 작업(Refine·단독/배치 SAM3·ADetailer·손 재구성)에는 싣지 않는다.
"""
import copy
import json
import unittest
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from backends import BackendType
from core import alwayson_propagation as ap
from core import sam_extra_notices as sn
from core import vae_degrid as vdg
from tests.test_alwayson_propagation import A38, DD, DEGRID, DORA, PAG, SKIM, caps
from tests.test_sampling_blocks import ChainHost, _webui
from ui import boot_notices
from ui import sampling_blocks as sb
from ui import vae_degrid_ui as vui

REPO = Path(__file__).resolve().parent.parent
PRESENT = caps(present=(PAG, SKIM, DD, A38, DORA, DEGRID))
WITHOUT_DEGRID = caps(present=(PAG, SKIM, DD, A38, DORA))
ON_BLOCK = {"args": [{"enabled": True, "model": "", "mode": "full", "strength": 1.0, "tile": 512}]}


class _Proxy:
    def __init__(self, value=""):
        self.value = value

    def text(self):
        return self.value

    def setText(self, value):
        self.value = value


def degrid_widgets(**values):
    base = vdg.widget_values(vdg.APP_DEFAULTS)
    base.update({k: str(v) for k, v in values.items()})
    return {key: _Proxy(base[key]) for key in vdg.WIDGET_KEYS}


class DegridHost(ChainHost):
    """실제 메인 체인 + VAE DeGrid 카드 위젯(기본: 켬)."""

    def __init__(self, capabilities=None, guidance=None, **values):
        super().__init__(guidance, capabilities)
        values.setdefault("enabled", "true")
        self.degrid_widgets = degrid_widgets(**values)


# ── 기여자 ─────────────────────────────────────────────────────────────────────
class ContributorTests(unittest.TestCase):
    def setUp(self):
        _webui(self)

    def test_registered_after_dora(self):
        self.assertEqual(sb.CONTRIBUTORS[-1], vui.contribute)
        self.assertEqual(sb.CONTRIBUTORS.index(vui.contribute), len(sb.CONTRIBUTORS) - 1)

    def test_t2i_known_present_sends_the_dict_block_last_as_a_user_value(self):
        result = sb.build_sampling_blocks(DegridHost(PRESENT), ap.TARGET_T2I)
        self.assertEqual(list(result.blocks)[-1], DEGRID)
        self.assertEqual(result.blocks[DEGRID], ON_BLOCK)
        self.assertEqual(result.provenance[DEGRID], ap.PROVENANCE_USER)
        self.assertEqual(result.notices, [])

    def test_off_card_and_hosts_without_widgets_build_nothing(self):
        off = sb.build_sampling_blocks(DegridHost(PRESENT, enabled="false"), ap.TARGET_T2I)
        self.assertNotIn(DEGRID, off.blocks)
        bare = sb.build_sampling_blocks(ChainHost(capabilities=PRESENT), ap.TARGET_T2I)
        self.assertNotIn(DEGRID, bare.blocks)

    def test_missing_script_is_dropped_with_the_block_not_sent_warning(self):
        result = sb.build_sampling_blocks(DegridHost(WITHOUT_DEGRID), ap.TARGET_T2I)
        self.assertNotIn(DEGRID, result.blocks)
        self.assertEqual(result.dropped, (DEGRID,))
        self.assertEqual([(n.code, n.level, n.feature) for n in result.notices],
                         [(sn.CODE_BLOCK_NOT_SENT, sn.LEVEL_WARNING, "degrid")])
        self.assertIn("VAE DeGrid", result.notices[0].message)

    def test_unknown_snapshot_sends_the_user_block(self):
        """(D3) 사용자 값 — 모를 때는 보낸다. 없는 Forge 는 422 를 가이던스처럼 설명한다(main_retry 없음)."""
        result = sb.build_sampling_blocks(DegridHost(None), ap.TARGET_T2I)
        self.assertEqual(result.blocks[DEGRID], ON_BLOCK)
        self.assertEqual(result.notices, [])

    def test_a_model_missing_from_the_startup_list_is_still_sent(self):
        capabilities = replace(PRESENT, choices=sn_choices(("qwenVAEDegridNafnet_v11",)))
        result = sb.build_sampling_blocks(DegridHost(capabilities, model="new_file"), ap.TARGET_T2I)
        self.assertEqual(result.blocks[DEGRID]["args"][0]["model"], "new_file")
        self.assertEqual(result.notices, [])

    def test_i2i_follows_the_toggle_and_the_img2img_list(self):
        i2i_caps = caps(present=(DEGRID,), img2img=(DEGRID,))
        off = sb.build_sampling_blocks(DegridHost(i2i_caps), ap.TARGET_I2I)
        self.assertNotIn(DEGRID, off.blocks)
        self.assertEqual([n for n in off.notices if n.feature == "degrid"], [])   # 토글 끔은 조용히
        on = sb.build_sampling_blocks(DegridHost(i2i_caps, apply_img2img="true"), ap.TARGET_I2I)
        self.assertEqual(on.blocks[DEGRID], ON_BLOCK)
        gone = sb.build_sampling_blocks(DegridHost(caps(present=(DEGRID,), img2img=()), apply_img2img="true"),
                                        ap.TARGET_I2I)
        self.assertNotIn(DEGRID, gone.blocks)

    def test_comfy_builds_the_block_and_krea2_builds_nothing(self):
        with mock.patch("backends.get_backend_type", return_value=BackendType.COMFYUI):
            comfy = sb.build_sampling_blocks(DegridHost(caps(present=())), ap.TARGET_T2I)
        self.assertEqual(comfy.blocks[DEGRID], ON_BLOCK)          # Comfy 는 게이트하지 않는다(컴파일러가 노드로)
        krea2 = sb.build_sampling_blocks(DegridHost(PRESENT), ap.TARGET_T2I, krea2=True)
        self.assertNotIn(DEGRID, krea2.blocks)
        self.assertEqual(krea2.notices, [])

    def test_aux_envelope_carries_the_t2i_block_without_notice(self):
        result = sb.build_sampling_blocks(DegridHost(None, apply_img2img="false"), ap.TARGET_AUX)
        self.assertEqual(result.blocks[DEGRID], ON_BLOCK)
        self.assertEqual(result.notices, [])

    def test_contributor_failure_is_isolated(self):
        with mock.patch.object(vdg, "plan", side_effect=RuntimeError("degrid broke")), \
                self.assertLogs("generation", "WARNING"):
            payload, error = DegridHost(PRESENT)._build_generation_payload(snapshot=True)
        self.assertIsNone(error)
        self.assertNotIn(DEGRID, payload["alwayson_scripts"])
        self.assertIn(PAG, payload["alwayson_scripts"])


def sn_choices(models):
    from core.sam_extra_capabilities import _freeze
    return _freeze({"degrid_models": tuple(models)})


# ── 실제 진입점 ────────────────────────────────────────────────────────────────
class MainRequestTests(unittest.TestCase):
    """메인 요청 경로마다 실제 빌더로 — 블록이 실리는지·빠지는지."""

    def setUp(self):
        _webui(self)

    def test_t2i_main_chain_payload(self):
        payload, error = DegridHost(PRESENT, mode="dark", strength="0.8", tile="0",
                                    model="qwenVAEDegridNafnet_v11")._build_generation_payload(snapshot=True)
        self.assertIsNone(error)
        self.assertEqual(payload["alwayson_scripts"][DEGRID],
                         {"args": [{"enabled": True, "model": "qwenVAEDegridNafnet_v11", "mode": "dark",
                                    "strength": 0.8, "tile": 0}]})
        self.assertNotIn(ap.PROVENANCE_KEY, payload)                  # 사용자 값 — 출처 키를 적지 않는다
        off, _ = DegridHost(PRESENT, enabled="false")._build_generation_payload(snapshot=True)
        self.assertNotIn(DEGRID, off["alwayson_scripts"])

    def test_default_card_leaves_the_request_byte_identical(self):
        """앱 기본값(꺼짐)이면 DeGrid 위젯이 없던 때와 요청이 같다(스냅샷 모름·확인됨·Comfy)."""
        for label, capabilities, backend in (("unknown", None, BackendType.WEBUI), ("known", PRESENT, BackendType.WEBUI),
                                             ("comfy", None, BackendType.COMFYUI)):
            with self.subTest(label), mock.patch("backends.get_backend_type", return_value=backend), \
                    mock.patch("core.comfy_workflow_controls.snapshot_comfy_payload", side_effect=lambda _b, p, _m: p), \
                    mock.patch("ui.comfy_workflow_actions.quality_preset_payload", return_value={}), \
                    mock.patch("core.spectrum_settings.spectrum_payload_from_prefs", return_value={}):
                before, _ = ChainHost(capabilities=capabilities)._build_generation_payload(snapshot=True)
                host = DegridHost(capabilities, enabled="false")
                after, _ = host._build_generation_payload(snapshot=True)
                self.assertEqual(json.dumps(after, sort_keys=True), json.dumps(before, sort_keys=True))

    def test_image_tab_entry_follows_the_toggle(self):
        i2i_caps = caps(present=(DEGRID,), img2img=(DEGRID,))
        off = DegridHost(i2i_caps).apply_alwayson_extensions({"prompt": "x", "init_images": []})
        self.assertNotIn(DEGRID, off["alwayson_scripts"])
        on = DegridHost(i2i_caps, apply_img2img="true").apply_alwayson_extensions({"prompt": "x", "init_images": []})
        self.assertEqual(on["alwayson_scripts"][DEGRID], ON_BLOCK)

    def test_i2i_and_inpaint_tabs_send_what_the_toggle_says(self):
        from tests import test_i2i_payload as i2i_tests
        from tests import test_inpaint_payload as inpaint_tests
        from ui.i2i_actions import start_vue_i2i
        from ui.inpaint_actions import start_vue_inpaint

        image = i2i_tests._data_url(i2i_tests._png(64, 64))
        for toggle, expected in (("false", None), ("true", ON_BLOCK)):
            with self.subTest(toggle=toggle):
                host = DegridHost(PRESENT, apply_img2img=toggle)
                host.vue_bridge = i2i_tests._Bridge()
                i2i_tests._Worker.instances.clear()
                self.assertTrue(start_vue_i2i(host, {"image": image}, worker_factory=i2i_tests._Worker))
                self.assertEqual(i2i_tests._Worker.instances[-1].payload["alwayson_scripts"].get(DEGRID), expected)

                host = DegridHost(PRESENT, apply_img2img=toggle)
                host.vue_bridge = i2i_tests._Bridge()
                inpaint_tests._Worker.instances.clear()
                self.assertTrue(start_vue_inpaint(host, {"image": image, "mask": inpaint_tests.MASK},
                                                  worker_factory=inpaint_tests._Worker))
                sent = inpaint_tests._Worker.instances[-1].payload["alwayson_scripts"]
                self.assertEqual(sent.get(DEGRID), expected)

    def test_chat_generation_and_chat_image_edit(self):
        _model, t2i = DegridHost(PRESENT)._chat_generation_snapshot("p")
        self.assertEqual(t2i["alwayson_scripts"][DEGRID], ON_BLOCK)
        _model, edit = DegridHost(PRESENT)._chat_generation_snapshot("p", target="i2i")
        self.assertNotIn(DEGRID, edit["alwayson_scripts"])
        _model, edit_on = DegridHost(PRESENT, apply_img2img="true")._chat_generation_snapshot("p", target="i2i")
        self.assertEqual(edit_on["alwayson_scripts"][DEGRID], ON_BLOCK)

    def test_every_comic_cut_carries_the_block(self):
        from tests.test_comic_generation import _Document, _panel_payloads
        from ui.creator_actions import CreatorActionsMixin

        class _ComicHost(DegridHost, CreatorActionsMixin):
            pass

        host = _ComicHost(PRESENT)
        document = _Document(3)
        host._comic_studio = lambda: SimpleNamespace(normalize=lambda _d: document, save=lambda d: d)
        with mock.patch("core.comic_studio.panel_generation_payloads", side_effect=_panel_payloads):
            block = host._comic_t2i_snapshots({"document": {}, "model": "anima.safetensors"}, "comic_generate_all")
        self.assertEqual(len(block["panels"]), 3)
        for entry in block["panels"]:
            self.assertEqual(entry["request"]["alwayson_scripts"][DEGRID], ON_BLOCK)

    def test_frozen_xyz_seed_payload_is_regated_when_sent(self):
        """시드 탐색·XYZ 대기열: 클릭 때 실린 블록을 보낼 때의 스냅샷이 '없다'고 하면 빼고 경고한다(모르면 그대로)."""
        host = DegridHost(PRESENT)
        payload, _ = host._build_generation_payload(snapshot=True)
        self.assertEqual(sb.freeze_sampling_payload(host, payload), [])
        self.assertNotIn(ap.PROVENANCE_KEY, payload)
        frozen = copy.deepcopy(payload)
        host.sam_extra_capabilities = None
        with mock.patch("core.sam_extra_probe.peek_capabilities", return_value=None):
            self.assertEqual(sb.regate_frozen_payload(host, frozen), [])
        self.assertIn(DEGRID, frozen["alwayson_scripts"])
        host.sam_extra_capabilities = WITHOUT_DEGRID
        notices = sb.regate_frozen_payload(host, frozen)
        self.assertNotIn(DEGRID, frozen["alwayson_scripts"])
        self.assertEqual([n.code for n in notices], [sn.CODE_BLOCK_NOT_SENT])


# ── 보조 작업에는 없다 ─────────────────────────────────────────────────────────
class AuxPassTests(unittest.TestCase):
    def setUp(self):
        _webui(self)

    def test_hand_repair_snapshot_excludes_the_block(self):
        from ui.hand_reconstruction_actions import _sampling_scripts
        snapshot = {"alwayson_scripts": {PAG: {"args": [1]}, DEGRID: ON_BLOCK}}
        for backend in ("webui", "comfyui"):
            with self.subTest(backend=backend):
                self.assertNotIn(DEGRID, _sampling_scripts(snapshot, backend=backend))
                self.assertNotIn(DEGRID, _sampling_scripts(snapshot, backend=backend, capabilities=PRESENT))

    def test_forge_refine_request_never_carries_the_block(self):
        from backends.webui_backend import WebUIBackend
        from tests.test_webui_aux_propagation import ok, png_b64
        from ui.aux_pass_snapshot import attach_sampling_snapshot

        host = DegridHost(None, guidance={"guid_enabled": "false"})
        settings = attach_sampling_snapshot(host, {"target": "face"})
        self.assertIn(DEGRID, settings[ap.SETTINGS_KEY]["blocks"])           # 봉투에는 있다(T15)
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
                self.assertNotIn(DEGRID, sent[-1]["alwayson_scripts"])
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
    """앱이 시작할 때 프록시를 만드는지 — 다른 테스트는 모두 ``degrid_widgets`` 를 손으로 심은 호스트라, 이 배선이
    빠지면(current_settings()=None → 블록 없음) 카드를 켜도 아무 테스트도 실패하지 않는다."""

    def test_generator_ui_setup_creates_the_proxies(self):
        source = (REPO / "ui" / "generator_ui_setup.py").read_text(encoding="utf-8")
        self.assertIn("from ui.vae_degrid_ui import init_degrid_proxies", source)
        self.assertIn("self.degrid_widgets = init_degrid_proxies(b)", source)

    def test_init_registers_every_widget_id_with_the_app_defaults(self):
        bridge = _Bridge()
        widgets = vui.init_degrid_proxies(bridge)
        self.assertEqual(set(bridge.registered), {vdg.widget_id(key) for key in vdg.WIDGET_KEYS})
        for key, proxy in widgets.items():
            self.assertIs(bridge.registered[vdg.widget_id(key)], proxy)
        defaults = vdg.widget_values(vdg.APP_DEFAULTS)
        self.assertEqual({key: widgets[key].text() for key in vdg.WIDGET_KEYS}, defaults)
        # 빈 문자열(모델 자동)은 프록시 초기값과 같아 다시 보내지 않는다 — 나머지는 Vue 로 간다
        self.assertEqual(dict(bridge.pushed), {vdg.widget_id(k): v for k, v in defaults.items() if v})
        host = ChainHost(None, PRESENT)
        host.degrid_widgets = widgets
        self.assertEqual(vui.current_settings(host), vdg.APP_DEFAULTS)
        widgets["enabled"].setText("true")
        with mock.patch("backends.get_backend_type", return_value=BackendType.WEBUI):
            self.assertEqual(sb.build_sampling_blocks(host, ap.TARGET_T2I).blocks[DEGRID], ON_BLOCK)
        self.assertEqual(vui.get_settings(host)["enabled"], "true")

    def test_push_comfy_models_sets_and_clears_the_model_property(self):
        bridge = _Bridge()
        host = SimpleNamespace(vue_bridge=bridge)
        vui.push_comfy_models(host, ["qwenVAEDegridNafnet_v11", " ", "None", "ESRGAN/x"])
        vui.push_comfy_models(host, [])
        vui.push_comfy_models(host, None)
        self.assertEqual(bridge.properties, [
            ("_degrid_model", "comfyModels", ["qwenVAEDegridNafnet_v11", "ESRGAN/x"]),
            ("_degrid_model", "comfyModels", []),
            ("_degrid_model", "comfyModels", None)])
        vui.push_comfy_models(SimpleNamespace(), ["x"])                        # 브리지 없는 호스트 — 조용히

        class _Gone:
            def pushWidgetProperty(self, *_a):
                raise RuntimeError("wrapped C/C++ object has been deleted")
        vui.push_comfy_models(SimpleNamespace(vue_bridge=_Gone()), ["x"])     # 종료 중 — 조용히


class PersistenceTests(unittest.TestCase):
    def test_settings_dict_and_preset_restore(self):
        from tests.test_generation_presets import _host
        from ui.generation_settings_apply import apply_generation_settings
        from ui.generator_settings import _degrid_settings

        host = _host([])
        host.degrid_widgets = degrid_widgets(enabled="true", model="v11", mode="bright", strength="0.6", tile="1024",
                                             apply_img2img="true")
        saved = _degrid_settings(host)
        self.assertEqual(saved, {"enabled": "true", "model": "v11", "mode": "bright", "strength": "0.6",
                                 "tile": "1024", "apply_img2img": "true"})
        other = _host([])
        other.degrid_widgets = degrid_widgets()
        apply_generation_settings(other, {vdg.SETTINGS_KEY: saved}, only_present=True)
        self.assertEqual(vui.get_settings(other), saved)
        apply_generation_settings(other, {"steps": "30"}, only_present=True)   # 프리셋에 키가 없으면 그대로
        self.assertEqual(vui.get_settings(other), saved)

    def test_old_settings_file_starts_with_app_defaults_quietly(self):
        host = SimpleNamespace(degrid_widgets=degrid_widgets(enabled="true", mode="dark"))
        vui.apply_saved_settings(host, {"steps": "28"}, only_present=False)
        self.assertEqual(vui.current_settings(host), vdg.APP_DEFAULTS)
        self.assertEqual(boot_notices.pending_boot_notices(host), [])         # 동작이 같아 안내 없음
        vui.apply_saved_settings(SimpleNamespace(), {vdg.SETTINGS_KEY: {"enabled": "true"}}, only_present=False)

    def test_saved_values_are_normalized_on_restore(self):
        host = SimpleNamespace(degrid_widgets=degrid_widgets())
        vui.apply_saved_settings(host, {vdg.SETTINGS_KEY: {"enabled": "true", "model": "None",
                                                           "mode": "Dark Pixels Mainly (어두운 점 위주)",
                                                           "strength": "7", "tile": "64"}}, only_present=True)
        self.assertEqual(vui.current_settings(host), vdg.DegridSettings(True, "", "dark", 1.5, 128, False))

    def test_build_settings_dict_and_preset_keys(self):
        from core.generation_presets import GENERATION_KEYS
        from ui.generator_settings import SettingsMixin

        self.assertIn(vdg.SETTINGS_KEY, GENERATION_KEYS)

        class _Host(SettingsMixin):
            def __getattr__(self, name):
                if name.startswith("__"):
                    raise AttributeError(name)
                value = mock.MagicMock(name=name)
                setattr(self, name, value)
                return value

        host = _Host()
        host.degrid_widgets = degrid_widgets(enabled="true")
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
        self.assertEqual(settings[vdg.SETTINGS_KEY]["enabled"], "true")


if __name__ == "__main__":
    unittest.main()
