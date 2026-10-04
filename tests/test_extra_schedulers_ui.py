"""Extra Schedulers 값 칸 ↔ 위젯 프록시 ↔ 샘플링 블록(ui/extra_schedulers_ui) — 실제 페이로드 빌더를 거친 요청 모양.

GPU·Forge 없이: 기능 스냅샷은 가짜(``tests.test_alwayson_propagation.caps``), Forge 요청은 가짜 requests.post.
규칙(계획 X5·R7·R16): 스냅샷이 스크립트를 확인하면 늘(기본값이면 앱 기본값 출처), 확인 못 하면 laplace·custom·
flow_cosmos_rho7·flow_cosmos_dynamic 을 쓰는 패스가 있을 때만 싣는다. 메인 요청과 모든 보조 img2img 에 같은 블록(늘 키 8개 — v0.33.0 D19 의
Flow Cosmos 셋 포함). ComfyUI 는 팩 1.7.0 단계 전이라 싣지 않는다.
"""
import copy
import json
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from backends import BackendType
from core import alwayson_propagation as ap
from core import extra_schedulers as xs
from core import sam_extra_notices as sn
from tests.test_alwayson_propagation import A38, DD, DEGRID, DORA, PAG, SKIM, caps
from tests.test_chat_generation_snapshot import ReadOnlyWidget
from tests.test_sampling_blocks import ChainHost, _webui
from ui import boot_notices
from ui import extra_schedulers_ui as xui
from ui import sampling_blocks as sb

REPO = Path(__file__).resolve().parent.parent
XS = xs.SCRIPT_NAME
PRESENT = caps(present=(PAG, SKIM, DD, A38, DORA, DEGRID, XS))


def _known(present, absent):
    """known 스냅샷 — ``absent`` 제목은 '없음'으로 확정(``caps`` 는 목록 밖 새 제목을 모름으로 둔다)."""
    from core.sam_extra_capabilities import SamExtraCapabilities, _freeze
    scripts = {title.lower(): {"present": True, "img2img": True} for title in present}
    scripts.update({title.lower(): {"present": False, "img2img": False} for title in absent})
    return SamExtraCapabilities(status="ok", installed=True, scripts=_freeze(scripts))


WITHOUT_XS = _known((PAG, SKIM, DD, A38, DORA, DEGRID), (XS,))
DEFAULT_BLOCK = {"args": [{"custom_mode": "expression", "custom_expression": "M * (m / M) ** x",
                           "custom_sigmas": "[1.0, 0.6, 0.25, 0.1, 0.0]", "laplace_mu": 0.0, "laplace_beta": 0.5,
                           "flow_cosmos_rho": 7.0, "flow_cosmos_sigma_max": 80.0, "flow_cosmos_sigma_min": 0.002}]}


class _Proxy:
    def __init__(self, value=""):
        self.value = value

    def text(self):
        return self.value

    def setText(self, value):
        self.value = value


def xsched_widgets(**values):
    base = xs.widget_values(xs.APP_DEFAULTS)
    base.update({k: str(v) for k, v in values.items()})
    return {key: _Proxy(base[key]) for key in xs.WIDGET_KEYS}


def user_block(**values):
    return xs.as_block(xs.parse_settings({**xs.widget_values(xs.APP_DEFAULTS), **{k: str(v) for k, v in values.items()}}))


class XschedHost(ChainHost):
    """실제 메인 체인 + Extra Schedulers 칸. ``scheduler`` 로 메인 스케줄러(ChainHost 기본 'Beta')를 바꾼다."""

    def __init__(self, capabilities=None, guidance=None, *, scheduler=None, **values):
        super().__init__(guidance, capabilities)
        self.extra_schedulers_widgets = xsched_widgets(**values)
        if scheduler is not None:
            self.scheduler_combo = ReadOnlyWidget(scheduler)


# ── 기여자 ─────────────────────────────────────────────────────────────────────
class ContributorTests(unittest.TestCase):
    def setUp(self):
        _webui(self)

    def test_registered_right_after_degrid(self):
        from ui.vae_degrid_ui import contribute as degrid
        self.assertEqual(sb.CONTRIBUTORS.index(xui.contribute), sb.CONTRIBUTORS.index(degrid) + 1)
        self.assertEqual(sb.CONTRIBUTORS.count(xui.contribute), 1)

    def test_confirmed_snapshot_always_sends_the_dict_block(self):
        default = sb.build_sampling_blocks(XschedHost(PRESENT), ap.TARGET_T2I)
        self.assertEqual(list(default.blocks)[-1], XS)
        self.assertEqual(default.blocks[XS], DEFAULT_BLOCK)
        self.assertEqual(default.provenance[XS], ap.PROVENANCE_APP_DEFAULT)
        self.assertEqual(default.notices, [])
        user = sb.build_sampling_blocks(XschedHost(PRESENT, laplace_mu="0.5", custom_mode="sigmas"), ap.TARGET_T2I)
        self.assertEqual(user.blocks[XS], user_block(laplace_mu=0.5, custom_mode="sigmas"))
        self.assertEqual(user.provenance[XS], ap.PROVENANCE_USER)

    def test_unknown_snapshot_sends_only_for_a_value_scheduler_and_user_values(self):
        idle = sb.build_sampling_blocks(XschedHost(None, laplace_mu="1"), ap.TARGET_T2I)   # 메인 'Beta'
        self.assertNotIn(XS, idle.blocks)
        self.assertEqual((idle.dropped, idle.notices), ((), []))
        used = sb.build_sampling_blocks(XschedHost(None, scheduler="Laplace", laplace_mu="1"), ap.TARGET_T2I)
        self.assertEqual(used.blocks[XS], user_block(laplace_mu=1))
        self.assertEqual(used.notices, [])
        quiet = sb.build_sampling_blocks(XschedHost(None, scheduler="laplace"), ap.TARGET_T2I)   # 기본값 — 앱 기본값
        self.assertNotIn(XS, quiet.blocks)
        self.assertEqual(quiet.dropped, (XS,))
        self.assertEqual(quiet.notices, [])                                       # 앱 기본값 블록은 알리지 않는다

    def test_flow_cosmos_rho7_reads_the_values_too(self):
        """v0.33.0 D19: Flow Cosmos rho7 도 값 스케줄러 — 확인 못 한 스냅샷에서도 그 패스가 있으면 사용자 값을 싣는다."""
        used = sb.build_sampling_blocks(XschedHost(None, scheduler="flow_cosmos_rho7", flow_cosmos_rho="5.5",
                                                   flow_cosmos_sigma_max="240", flow_cosmos_sigma_min="0.006"),
                                        ap.TARGET_T2I)
        self.assertEqual(used.blocks[XS], user_block(flow_cosmos_rho=5.5, flow_cosmos_sigma_max=240,
                                                     flow_cosmos_sigma_min=0.006))
        self.assertEqual(used.provenance[XS], ap.PROVENANCE_USER)
        self.assertEqual(used.notices, [])
        quiet = sb.build_sampling_blocks(XschedHost(None, scheduler="Flow Cosmos rho7"), ap.TARGET_T2I)
        self.assertNotIn(XS, quiet.blocks)                                        # 기본값 — 앱 기본값, 알리지 않는다
        self.assertEqual((quiet.dropped, quiet.notices), ((XS,), []))
        idle = sb.build_sampling_blocks(XschedHost(None, flow_cosmos_rho="3"), ap.TARGET_T2I)   # 메인 'Beta'
        self.assertNotIn(XS, idle.blocks)

    def test_flow_cosmos_dynamic_reads_the_same_values(self):
        """v0.33.0 D20: Flow Cosmos Dynamic 도 같은 세 값을 읽는 값 스케줄러 — 그 패스가 있으면 사용자 값을 싣는다."""
        used = sb.build_sampling_blocks(XschedHost(None, scheduler="Flow Cosmos Dynamic", flow_cosmos_rho="5.5",
                                                   flow_cosmos_sigma_min="0.01"), ap.TARGET_T2I)
        self.assertEqual(used.blocks[XS], user_block(flow_cosmos_rho=5.5, flow_cosmos_sigma_min=0.01))
        self.assertEqual(len(used.blocks[XS]["args"][0]), 8)
        self.assertEqual((used.provenance[XS], used.notices), (ap.PROVENANCE_USER, []))
        quiet = sb.build_sampling_blocks(XschedHost(None, scheduler="flow_cosmos_dynamic"), ap.TARGET_T2I)
        self.assertEqual((XS in quiet.blocks, quiet.dropped, quiet.notices), (False, (XS,), []))

    def test_missing_script_drops_user_values_with_a_warning(self):
        result = sb.build_sampling_blocks(XschedHost(WITHOUT_XS, scheduler="custom", custom_expression="M"),
                                          ap.TARGET_T2I)
        self.assertNotIn(XS, result.blocks)
        self.assertEqual([(n.code, n.feature) for n in result.notices], [(sn.CODE_BLOCK_NOT_SENT, "extra_schedulers")])
        idle = sb.build_sampling_blocks(XschedHost(WITHOUT_XS, custom_expression="M"), ap.TARGET_T2I)
        self.assertEqual((idle.dropped, idle.notices), ((), []))                  # 쓰지 않는 값 — 싣지도 알리지도

    def test_every_pass_that_reads_the_values_counts(self):
        host = XschedHost(None, laplace_beta="2")
        host.hires_scheduler_combo = ReadOnlyWidget("custom")                     # Hires 켬(ChainHost)
        self.assertIn(XS, sb.build_sampling_blocks(host, ap.TARGET_T2I).blocks)
        host = XschedHost(None, laplace_beta="2")
        host.hires_options_group = ReadOnlyWidget(False)
        host.hires_scheduler_combo = ReadOnlyWidget("custom")
        self.assertNotIn(XS, sb.build_sampling_blocks(host, ap.TARGET_T2I).blocks)
        host.adetailer_group, host.ad_slot2_group = ReadOnlyWidget(True), ReadOnlyWidget(True)
        host.s2_widgets = {"use_sampler_check": ReadOnlyWidget(True), "scheduler_combo": ReadOnlyWidget("Laplace")}
        self.assertEqual(xui.passes_in_use(host), (xs.PassUse("adetailer_2", "laplace"),))
        self.assertIn(XS, sb.build_sampling_blocks(host, ap.TARGET_T2I).blocks)

    def test_i2i_requests_carry_no_main_scheduler(self):
        """I2I·인페인트 요청은 스케줄러를 싣지 않는다(Forge 기본) — 메인 콤보의 Laplace 는 그 요청과 무관하다."""
        main_only = XschedHost(None, scheduler="Laplace", laplace_mu="1")
        self.assertNotIn(XS, sb.build_sampling_blocks(main_only, ap.TARGET_I2I).blocks)
        main_only.sam3_group = ReadOnlyWidget(True)
        main_only.sam3_widgets = {"mode": ReadOnlyWidget("Inpaint"), "use_scheduler_check": ReadOnlyWidget(True),
                                  "scheduler": ReadOnlyWidget("custom")}
        self.assertIn(XS, sb.build_sampling_blocks(main_only, ap.TARGET_I2I).blocks)
        i2i_caps = caps(present=(XS,), img2img=(XS,))
        self.assertEqual(sb.build_sampling_blocks(XschedHost(i2i_caps), ap.TARGET_I2I).blocks[XS], DEFAULT_BLOCK)

    def test_comfy_and_krea2_build_nothing(self):
        with mock.patch("backends.get_backend_type", return_value=BackendType.COMFYUI):
            comfy = sb.build_sampling_blocks(XschedHost(None, scheduler="Laplace", laplace_mu="1"), ap.TARGET_T2I)
        self.assertNotIn(XS, comfy.blocks)                                        # 팩 1.7.0 단계 전(R16)
        krea2 = sb.build_sampling_blocks(XschedHost(PRESENT), ap.TARGET_T2I, krea2=True)
        self.assertNotIn(XS, krea2.blocks)

    def test_aux_envelope_carries_the_t2i_block(self):
        result = sb.build_sampling_blocks(XschedHost(PRESENT, laplace_mu="-1"), ap.TARGET_AUX)
        self.assertEqual(result.blocks[XS], user_block(laplace_mu=-1))
        self.assertEqual(result.notices, [])

    def test_contributor_failure_is_isolated(self):
        with mock.patch.object(xs, "plan", side_effect=RuntimeError("xsched broke")), \
                self.assertLogs("generation", "WARNING"):
            payload, error = XschedHost(PRESENT)._build_generation_payload(snapshot=True)
        self.assertIsNone(error)
        self.assertNotIn(XS, payload["alwayson_scripts"])
        self.assertIn(PAG, payload["alwayson_scripts"])


# ── 실제 진입점 ────────────────────────────────────────────────────────────────
class MainRequestTests(unittest.TestCase):
    def setUp(self):
        _webui(self)

    def test_t2i_payload_carries_the_design_block_and_its_provenance(self):
        """계획 GA-XS: 기여자를 거친 Forge 요청의 블록 = design_schedulers.md §3.1 의 JSON(키 순서까지)."""
        payload, error = XschedHost(PRESENT)._build_generation_payload(snapshot=True)
        self.assertIsNone(error)
        expected = ('{"args": [{"custom_mode": "expression", "custom_expression": "M * (m / M) ** x", '
                    '"custom_sigmas": "[1.0, 0.6, 0.25, 0.1, 0.0]", "laplace_mu": 0.0, "laplace_beta": 0.5, '
                    '"flow_cosmos_rho": 7.0, "flow_cosmos_sigma_max": 80.0, "flow_cosmos_sigma_min": 0.002}]}')
        self.assertEqual(json.dumps(payload["alwayson_scripts"][XS]), expected)
        self.assertEqual(payload[ap.PROVENANCE_KEY], {XS: ap.PROVENANCE_APP_DEFAULT})   # 422 면 정보 알림으로 재시도
        user, _ = XschedHost(PRESENT, custom_mode="sigmas", custom_sigmas="[1, 0.3, 0]",
                             laplace_beta="1.5")._build_generation_payload(snapshot=True)
        self.assertEqual(user["alwayson_scripts"][XS],
                         {"args": [{"custom_mode": "sigmas", "custom_expression": "M * (m / M) ** x",
                                    "custom_sigmas": "[1, 0.3, 0]", "laplace_mu": 0.0, "laplace_beta": 1.5,
                                    "flow_cosmos_rho": 7.0, "flow_cosmos_sigma_max": 80.0,
                                    "flow_cosmos_sigma_min": 0.002}]})
        self.assertNotIn(ap.PROVENANCE_KEY, user)
        self.assertEqual(ap.main_retry_titles(user), (XS,))

    def test_requests_without_the_feature_stay_byte_identical(self):
        """스냅샷이 확인하지 않았고 laplace·custom 패스가 없으면(지금까지의 모든 요청) 요청이 칸이 없던 때와 같다."""
        for label, capabilities, backend in (("unknown", None, BackendType.WEBUI),
                                             ("known without the script", WITHOUT_XS, BackendType.WEBUI),
                                             ("comfy", None, BackendType.COMFYUI)):
            with self.subTest(label), mock.patch("backends.get_backend_type", return_value=backend), \
                    mock.patch("core.comfy_workflow_controls.snapshot_comfy_payload", side_effect=lambda _b, p, _m: p), \
                    mock.patch("ui.comfy_workflow_actions.quality_preset_payload", return_value={}), \
                    mock.patch("core.spectrum_settings.spectrum_payload_from_prefs", return_value={}):
                before, _ = ChainHost(capabilities=capabilities)._build_generation_payload(snapshot=True)
                after, _ = XschedHost(capabilities, laplace_mu="3")._build_generation_payload(snapshot=True)
                self.assertEqual(json.dumps(after, sort_keys=True), json.dumps(before, sort_keys=True))

    def test_frozen_payload_keeps_the_block_unless_the_snapshot_says_missing(self):
        host = XschedHost(PRESENT, laplace_mu="2")
        payload, _ = host._build_generation_payload(snapshot=True)
        self.assertEqual(sb.freeze_sampling_payload(host, payload), [])
        frozen = copy.deepcopy(payload)
        host.sam_extra_capabilities = None
        with mock.patch("core.sam_extra_probe.peek_capabilities", return_value=None):
            self.assertEqual(sb.regate_frozen_payload(host, frozen), [])
        self.assertIn(XS, frozen["alwayson_scripts"])
        host.sam_extra_capabilities = WITHOUT_XS
        notices = sb.regate_frozen_payload(host, frozen)
        self.assertNotIn(XS, frozen["alwayson_scripts"])
        self.assertEqual([n.code for n in notices], [sn.CODE_BLOCK_NOT_SENT])


class AuxPassTests(unittest.TestCase):
    """확장 전역 상태는 요청마다 덮인다 — 보조 img2img 도 같은 블록을 받아야 메인과 같은 μ/β·식으로 돈다."""

    def setUp(self):
        _webui(self)

    def test_hand_repair_snapshot_keeps_the_block(self):
        from ui.hand_reconstruction_actions import _sampling_scripts
        block = user_block(laplace_mu=1)
        snapshot = {"alwayson_scripts": {PAG: {"args": [1]}, XS: block}}
        self.assertEqual(_sampling_scripts(snapshot, backend="webui", capabilities=PRESENT)[XS], block)

    def test_forge_refine_request_carries_the_envelope_block(self):
        from backends.webui_backend import WebUIBackend
        from tests.test_webui_aux_propagation import ok, png_b64
        from ui.aux_pass_snapshot import attach_sampling_snapshot

        host = XschedHost(PRESENT, guidance={"guid_enabled": "false"}, laplace_mu="1.5")
        settings = attach_sampling_snapshot(host, {"target": "face"})
        self.assertEqual(settings[ap.SETTINGS_KEY]["blocks"][XS], user_block(laplace_mu=1.5))
        sent = []

        def post(url, json=None, **_kw):
            sent.append(copy.deepcopy(json))
            return ok()

        with mock.patch("core.sam_extra_probe.peek_capabilities", return_value=PRESENT), \
                mock.patch("core.forge_modules.resolve_sam3_checkpoint", side_effect=lambda name: name), \
                mock.patch("backends.webui_backend.requests.post", side_effect=post):
            WebUIBackend("http://127.0.0.1:7860").refine(png_b64(), settings)
        self.assertEqual(sent[-1]["alwayson_scripts"][XS], user_block(laplace_mu=1.5))
        self.assertIn("SAM3 Mask", sent[-1]["alwayson_scripts"])


# ── 프록시·패스 위젯·저장 ───────────────────────────────────────────────────────
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
    def test_generator_ui_setup_creates_the_proxies(self):
        source = (REPO / "ui" / "generator_ui_setup.py").read_text(encoding="utf-8")
        self.assertIn("from ui.extra_schedulers_ui import init_extra_schedulers_proxies", source)
        self.assertIn("self.extra_schedulers_widgets = init_extra_schedulers_proxies(b)", source)

    def test_init_registers_every_widget_id_with_the_app_defaults(self):
        bridge = _Bridge()
        widgets = xui.init_extra_schedulers_proxies(bridge)
        self.assertEqual(set(bridge.registered), {xs.widget_id(key) for key in xs.WIDGET_KEYS})
        defaults = xs.widget_values(xs.APP_DEFAULTS)
        self.assertEqual({key: widgets[key].text() for key in xs.WIDGET_KEYS}, defaults)
        self.assertEqual(dict(bridge.pushed), {xs.widget_id(k): v for k, v in defaults.items()})
        host = SimpleNamespace(extra_schedulers_widgets=widgets)
        self.assertEqual(xui.current_settings(host), xs.APP_DEFAULTS)

    def test_pass_values_read_every_pass_widget_of_the_host(self):
        host = ChainHost(capabilities=None)
        host.adetailer_group, host.ad_slot1_group, host.ad_slot2_group = (ReadOnlyWidget(True), ReadOnlyWidget(True),
                                                                          ReadOnlyWidget(False))
        host.s1_widgets = {"use_sampler_check": ReadOnlyWidget(True), "scheduler_combo": ReadOnlyWidget("custom")}
        host.sam3_group = ReadOnlyWidget(True)
        host.sam3_widgets = {"mode": ReadOnlyWidget("Mask only"), "use_scheduler_check": ReadOnlyWidget(True),
                             "scheduler": ReadOnlyWidget("Laplace")}
        values = xui.pass_values(host)
        self.assertEqual(tuple(values), xs.pass_widget_ids())
        self.assertEqual(values["scheduler_combo"], "Beta")
        self.assertIs(values["hires_options_group"], True)
        self.assertIsNone(values["_ad_s2_use_sampler"])                          # 슬롯 2 위젯 dict 없음
        self.assertEqual(xui.passes_in_use(host), (xs.PassUse("adetailer_1", "custom"),))   # SAM3 는 Mask only
        self.assertEqual(xui.pass_values(SimpleNamespace()), dict.fromkeys(xs.pass_widget_ids()))


class PersistenceTests(unittest.TestCase):
    def test_settings_dict_and_preset_restore(self):
        from tests.test_generation_presets import _host
        from ui.generation_settings_apply import apply_generation_settings
        from ui.generator_settings import _extra_schedulers_settings

        host = _host([])
        host.extra_schedulers_widgets = xsched_widgets(custom_mode="sigmas", custom_sigmas="[1, 0]", laplace_mu="-2.5",
                                                       laplace_beta="3", flow_cosmos_rho="5.5",
                                                       flow_cosmos_sigma_min="0.006")
        saved = _extra_schedulers_settings(host)
        self.assertEqual(saved, {"custom_mode": "sigmas", "custom_expression": "M * (m / M) ** x",
                                 "custom_sigmas": "[1, 0]", "laplace_mu": "-2.5", "laplace_beta": "3",
                                 "flow_cosmos_rho": "5.5", "flow_cosmos_sigma_max": "80",
                                 "flow_cosmos_sigma_min": "0.006"})
        other = _host([])
        other.extra_schedulers_widgets = xsched_widgets()
        apply_generation_settings(other, {xs.SETTINGS_KEY: saved}, only_present=True)
        self.assertEqual(xui.get_settings(other), saved)
        apply_generation_settings(other, {"steps": "30"}, only_present=True)    # 프리셋에 키가 없으면 그대로
        self.assertEqual(xui.get_settings(other), saved)

    def test_values_saved_before_flow_cosmos_restore_its_defaults(self):
        """v0.33.0 전에 저장한 값(키 다섯): Flow Cosmos 셋은 앱 기본값(ρ 7 · σ̃ 0.002~80) — 확장이 키 없는 옛 요청을 읽는
        것과 같다. 다른 칸은 저장값."""
        host = SimpleNamespace(extra_schedulers_widgets=xsched_widgets(flow_cosmos_rho="3", flow_cosmos_sigma_max="9"))
        old = {"custom_mode": "sigmas", "custom_expression": "x", "custom_sigmas": "[1, 0]", "laplace_mu": "1",
               "laplace_beta": "2"}
        xui.apply_saved_settings(host, {xs.SETTINGS_KEY: old}, only_present=True)
        self.assertEqual(xui.current_settings(host),
                         xs.ExtraSchedulerSettings("sigmas", "x", "[1, 0]", 1.0, 2.0, 7.0, 80.0, 0.002))
        self.assertEqual(boot_notices.pending_boot_notices(host), [])

    def test_old_settings_file_starts_with_app_defaults_quietly(self):
        host = SimpleNamespace(extra_schedulers_widgets=xsched_widgets(laplace_mu="4", custom_mode="sigmas"))
        xui.apply_saved_settings(host, {"steps": "28"}, only_present=False)
        self.assertEqual(xui.current_settings(host), xs.APP_DEFAULTS)
        self.assertEqual(boot_notices.pending_boot_notices(host), [])
        xui.apply_saved_settings(SimpleNamespace(), {xs.SETTINGS_KEY: {"laplace_mu": "1"}}, only_present=False)

    def test_saved_values_are_normalized_on_restore(self):
        host = SimpleNamespace(extra_schedulers_widgets=xsched_widgets())
        xui.apply_saved_settings(host, {xs.SETTINGS_KEY: {"custom_mode": "Sigma list", "laplace_mu": "99",
                                                          "laplace_beta": "abc", "custom_expression": "  x  "}},
                                 only_present=True)
        self.assertEqual(xui.current_settings(host),
                         xs.ExtraSchedulerSettings("sigmas", "x", xs.DEFAULT_SIGMAS, 10.0, 0.5))

    def test_generation_keys_and_settings_dict(self):
        from core.generation_presets import GENERATION_KEYS
        self.assertIn(xs.SETTINGS_KEY, GENERATION_KEYS)
        source = (REPO / "ui" / "generator_settings.py").read_text(encoding="utf-8")
        self.assertIn('"extra_schedulers_settings": _extra_schedulers_settings(self)', source)


class ApplyInfotextTests(unittest.TestCase):
    """붙여 넣기(계획 A-R1 이 '이 설정으로 생성'에 잇는다) — 확장 paste_* 규칙: 이미지가 쓰지 않은 칸은 그대로."""

    def test_laplace_and_custom_values_come_back(self):
        host = SimpleNamespace(extra_schedulers_widgets=xsched_widgets(custom_expression="x", laplace_mu="3"))
        restored = xui.apply_infotext(host, {"Schedule type": "Laplace", "Laplace mu": 0.5, "Laplace beta": 1.25})
        self.assertEqual(restored, xs.ExtraSchedulerSettings("expression", "x", xs.DEFAULT_SIGMAS, 0.5, 1.25))
        self.assertEqual(xui.current_settings(host), restored)
        xui.apply_infotext(host, {"Schedule type": "custom", "Custom scheduler sigmas": "[1.0, 0.0]"})
        self.assertEqual(xui.current_settings(host),
                         xs.ExtraSchedulerSettings("sigmas", "x", "[1.0, 0.0]", 0.5, 1.25))
        xui.apply_infotext(host, {"Schedule type": "Laplace"})                 # 키 없이 Laplace → 기본값
        self.assertEqual((xui.current_settings(host).laplace_mu, xui.current_settings(host).laplace_beta), (0.0, 0.5))
        before = xui.get_settings(host)
        self.assertEqual(xui.apply_infotext(host, {"Schedule type": "Karras"}), xui.current_settings(host))
        self.assertEqual(xui.get_settings(host), before)
        self.assertIsNone(xui.apply_infotext(SimpleNamespace(), {"Laplace mu": 1}))

    def test_flow_cosmos_values_come_back(self):
        host = SimpleNamespace(extra_schedulers_widgets=xsched_widgets(laplace_mu="3", flow_cosmos_rho="9"))
        restored = xui.apply_infotext(host, {"Schedule type": "Flow Cosmos rho7", "Flow Cosmos rho": "5.5",
                                             "Flow Cosmos sigma max": "240.0", "Flow Cosmos sigma min": "0.006"})
        self.assertEqual(restored, xs.ExtraSchedulerSettings(laplace_mu=3.0, flow_cosmos_rho=5.5,
                                                             flow_cosmos_sigma_max=240.0, flow_cosmos_sigma_min=0.006))
        self.assertEqual(xui.get_settings(host)["flow_cosmos_sigma_max"], "240")
        xui.apply_infotext(host, {"Schedule type": "Karras", "Hires schedule type": "Flow Cosmos rho7"})   # 키 없음
        self.assertEqual(xui.current_settings(host), xs.ExtraSchedulerSettings(laplace_mu=3.0))
        # v0.33.0 D20: Flow Cosmos Dynamic 이미지도 같은 세 키로 되살린다(키 없으면 기본값)
        restored = xui.apply_infotext(host, {"Schedule type": "Flow Cosmos Dynamic", "Flow Cosmos rho": "4.5",
                                             "Flow Cosmos sigma max": "80.0", "Flow Cosmos sigma min": "0.002"})
        self.assertEqual(restored, xs.ExtraSchedulerSettings(laplace_mu=3.0, flow_cosmos_rho=4.5))
        self.assertEqual(xui.get_settings(host)["flow_cosmos_rho"], "4.5")
        xui.apply_infotext(host, {"Schedule type": "flow_cosmos_dynamic"})
        self.assertEqual(xui.current_settings(host), xs.ExtraSchedulerSettings(laplace_mu=3.0))


if __name__ == "__main__":
    unittest.main()
