"""Extra Samplers ER SDE 값 칸 ↔ 위젯 프록시 ↔ 샘플링 블록(ui/extra_samplers_ui) — 실제 페이로드 빌더를 거친 요청 모양.

GPU·Forge 없이: 기능 스냅샷은 가짜(``tests.test_alwayson_propagation.caps``), Forge 요청은 가짜 requests.post.
규칙(계획 X5·X10·R7·R16): 스냅샷이 스크립트를 확인하면 늘(기본값이면 앱 기본값 출처), 확인 못 하면 ER SDE (Reverse-time)·
(ODE)·(Tunable) 을 쓰는 패스가 있을 때만 싣는다. 블록은 늘 위치 인자 다섯이고 메인 요청과 모든 보조 img2img 에 같다.
ComfyUI 는 늘 싣는다(계획 C6 — 컴파일러가 값이 기본값과 다를 때만 앱 노드 팩 1.7.0 의 노드를 넣는다). 저장 키 ``extra_samplers_settings``(키 다섯 — 빠진 칸은 앱 기본값).
"""
import copy
import json
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from backends import BackendType
from core import alwayson_propagation as ap
from core import extra_samplers as xsa
from core import extra_schedulers as xs
from core import sam_extra_notices as sn
from tests.test_alwayson_propagation import A38, DD, DEGRID, DORA, PAG, SKIM, caps
from tests.test_chat_generation_snapshot import ReadOnlyWidget
from tests.test_sampling_blocks import ChainHost, _webui
from ui import boot_notices
from ui import extra_samplers_ui as xui
from ui import sampling_blocks as sb

REPO = Path(__file__).resolve().parent.parent
XSA = xsa.SCRIPT_NAME
PRESENT = caps(present=(PAG, SKIM, DD, A38, DORA, DEGRID, XSA))
RT, ODE, TUN = xsa.LABEL_ER_SDE_REVERSE_TIME, xsa.LABEL_ER_SDE_ODE, xsa.LABEL_ER_SDE_TUNABLE


def _known(present, absent, *, argc=None):
    """known 스냅샷 — ``absent`` 제목은 '없음'으로 확정(``caps`` 는 목록 밖 새 제목을 모름으로 둔다). ``argc`` 는 Extra
    Samplers 의 script-info 인자 수(None = 모름)."""
    from core.sam_extra_capabilities import SamExtraCapabilities, _freeze
    scripts = {title.lower(): {"present": True, "img2img": True} for title in present}
    scripts.update({title.lower(): {"present": False, "img2img": False} for title in absent})
    if argc is not None and XSA.lower() in scripts:
        scripts[XSA.lower()]["live_argc"] = argc
    return SamExtraCapabilities(status="ok", installed=True, scripts=_freeze(scripts))


WITHOUT_XSA = _known((PAG, SKIM, DD, A38, DORA, DEGRID), (XSA,))
OLD_SAM_EXTRA = _known((PAG, SKIM, DD, A38, DORA, DEGRID, XSA), (), argc=2)   # 0.31·0.32 — 인자 2개
DEFAULT_BLOCK = {"args": [3, 1.0, False, 0.2, 0.8]}


class _Proxy:
    def __init__(self, value=""):
        self.value = value

    def text(self):
        return self.value

    def setText(self, value):
        self.value = value


def xsamp_widgets(**values):
    base = xsa.widget_values(xsa.APP_DEFAULTS)
    base.update({k: str(v) for k, v in values.items()})
    return {key: _Proxy(base[key]) for key in xsa.WIDGET_KEYS}


def user_block(**values):
    return xsa.as_block(xsa.parse_settings({**xsa.widget_values(xsa.APP_DEFAULTS),
                                             **{k: str(v) for k, v in values.items()}}))


class XsampHost(ChainHost):
    """실제 메인 체인 + Extra Samplers 칸. ``sampler`` 로 메인 샘플러(ChainHost 기본 'Euler')를 바꾼다."""

    def __init__(self, capabilities=None, guidance=None, *, sampler=None, **values):
        super().__init__(guidance, capabilities)
        self.extra_samplers_widgets = xsamp_widgets(**values)
        if sampler is not None:
            self.sampler_combo = ReadOnlyWidget(sampler)


# ── 기여자 ─────────────────────────────────────────────────────────────────────
class ContributorTests(unittest.TestCase):
    def setUp(self):
        _webui(self)

    def test_registered_last_right_after_extra_schedulers(self):
        from ui.extra_schedulers_ui import contribute as extra_schedulers
        self.assertEqual(sb.CONTRIBUTORS.index(xui.contribute), sb.CONTRIBUTORS.index(extra_schedulers) + 1)
        self.assertIs(sb.CONTRIBUTORS[-1], xui.contribute)
        self.assertEqual(sb.CONTRIBUTORS.count(xui.contribute), 1)

    def test_confirmed_snapshot_always_sends_the_five_argument_block(self):
        default = sb.build_sampling_blocks(XsampHost(PRESENT), ap.TARGET_T2I)
        self.assertEqual(list(default.blocks)[-1], XSA)
        self.assertEqual(default.blocks[XSA], DEFAULT_BLOCK)
        self.assertEqual(default.provenance[XSA], ap.PROVENANCE_APP_DEFAULT)
        self.assertEqual(default.notices, [])
        user = sb.build_sampling_blocks(XsampHost(PRESENT, max_stage="2", eta="0.5", noise_window="1"), ap.TARGET_T2I)
        self.assertEqual(user.blocks[XSA], {"args": [2, 0.5, True, 0.2, 0.8]})
        self.assertEqual(user.provenance[XSA], ap.PROVENANCE_USER)

    def test_unknown_snapshot_sends_only_for_an_er_sde_pass_and_user_values(self):
        idle = sb.build_sampling_blocks(XsampHost(None, eta="0.5"), ap.TARGET_T2I)           # 메인 'Euler'
        self.assertNotIn(XSA, idle.blocks)
        self.assertEqual((idle.dropped, idle.notices), ((), []))
        for sampler in (RT, ODE, TUN, "er_sde_tunable", " er sde (reverse-time) "):
            with self.subTest(sampler=sampler):
                used = sb.build_sampling_blocks(XsampHost(None, sampler=sampler, eta="0.5"), ap.TARGET_T2I)
                self.assertEqual(used.blocks[XSA], user_block(eta=0.5))
                self.assertEqual(used.notices, [])
        quiet = sb.build_sampling_blocks(XsampHost(None, sampler=RT), ap.TARGET_T2I)          # 기본값 — 앱 기본값
        self.assertNotIn(XSA, quiet.blocks)
        self.assertEqual(quiet.dropped, (XSA,))
        self.assertEqual(quiet.notices, [])                                       # 앱 기본값 블록은 알리지 않는다
        for other in ("Euler a", "ER SDE", "Euler SMEA Dy CFG++", "DPM++ 4M SDE", "Restart (flow)"):
            with self.subTest(other=other):                                       # 값을 읽지 않는 샘플러(내장 ER SDE 포함)
                self.assertNotIn(XSA, sb.build_sampling_blocks(XsampHost(None, sampler=other, max_stage="1"),
                                                               ap.TARGET_T2I).blocks)

    def test_missing_script_drops_user_values_with_a_warning(self):
        result = sb.build_sampling_blocks(XsampHost(WITHOUT_XSA, sampler=TUN, noise_window="1"), ap.TARGET_T2I)
        self.assertNotIn(XSA, result.blocks)
        self.assertEqual([(n.code, n.feature) for n in result.notices], [(sn.CODE_BLOCK_NOT_SENT, "extra_samplers")])
        idle = sb.build_sampling_blocks(XsampHost(WITHOUT_XSA, eta="2"), ap.TARGET_T2I)
        self.assertEqual((idle.dropped, idle.notices), ((), []))                  # 쓰지 않는 값 — 싣지도 알리지도

    def test_every_pass_that_reads_the_values_counts(self):
        host = XsampHost(None, eta="2")
        host.hires_sampler_combo = ReadOnlyWidget(ODE)                            # Hires 켬(ChainHost)
        self.assertIn(XSA, sb.build_sampling_blocks(host, ap.TARGET_T2I).blocks)
        host = XsampHost(None, eta="2")
        host.hires_options_group = ReadOnlyWidget(False)
        host.hires_sampler_combo = ReadOnlyWidget(ODE)
        self.assertNotIn(XSA, sb.build_sampling_blocks(host, ap.TARGET_T2I).blocks)
        host.adetailer_group, host.ad_slot2_group = ReadOnlyWidget(True), ReadOnlyWidget(True)
        host.s2_widgets = {"use_sampler_check": ReadOnlyWidget(True), "sampler_combo": ReadOnlyWidget(RT)}
        self.assertEqual(xui.passes_in_use(host), (xsa.PassUse("adetailer_2", RT),))
        self.assertIn(XSA, sb.build_sampling_blocks(host, ap.TARGET_T2I).blocks)

    def test_i2i_requests_carry_no_main_sampler(self):
        """I2I·인페인트 요청은 샘플러를 싣지 않는다(Forge 기본) — 메인 콤보의 ER SDE 는 그 요청과 무관하다."""
        main_only = XsampHost(None, sampler=RT, eta="0.3")
        self.assertNotIn(XSA, sb.build_sampling_blocks(main_only, ap.TARGET_I2I).blocks)
        main_only.sam3_group = ReadOnlyWidget(True)
        main_only.sam3_widgets = {"mode": ReadOnlyWidget("Inpaint"), "use_sampler_check": ReadOnlyWidget(True),
                                  "sampler": ReadOnlyWidget(TUN)}
        self.assertIn(XSA, sb.build_sampling_blocks(main_only, ap.TARGET_I2I).blocks)
        main_only.sam3_widgets["mode"] = ReadOnlyWidget("Mask only")
        self.assertNotIn(XSA, sb.build_sampling_blocks(main_only, ap.TARGET_I2I).blocks)
        i2i_caps = caps(present=(XSA,), img2img=(XSA,))
        self.assertEqual(sb.build_sampling_blocks(XsampHost(i2i_caps), ap.TARGET_I2I).blocks[XSA], DEFAULT_BLOCK)

    def test_comfy_always_carries_the_block_and_krea2_nothing(self):
        with mock.patch("backends.get_backend_type", return_value=BackendType.COMFYUI):
            plain = sb.build_sampling_blocks(XsampHost(None), ap.TARGET_T2I)
            comfy = sb.build_sampling_blocks(XsampHost(PRESENT, sampler="er_sde_reverse_time", eta="0.5"),
                                             ap.TARGET_T2I)
        # 계획 C6 — ComfyUI 는 늘 싣는다(ER SDE 가 아니어도, 기본값이어도 — 노드는 컴파일러가 값이 다를 때만)
        self.assertEqual(comfy.blocks[XSA], {"args": [3, 0.5, False, 0.2, 0.8]})
        self.assertEqual(comfy.provenance[XSA], ap.PROVENANCE_USER)
        self.assertEqual(plain.blocks[XSA], {"args": [3, 1.0, False, 0.2, 0.8]})
        self.assertEqual(plain.provenance[XSA], ap.PROVENANCE_APP_DEFAULT)
        self.assertEqual(xsa.plan(xsa.ExtraSamplersSettings(eta=0.5), target="t2i", backend="comfyui").reason,
                         xsa.REASON_COMFY)
        krea2 = sb.build_sampling_blocks(XsampHost(PRESENT), ap.TARGET_T2I, krea2=True)
        self.assertNotIn(XSA, krea2.blocks)

    def test_aux_envelope_carries_the_t2i_block(self):
        result = sb.build_sampling_blocks(XsampHost(PRESENT, max_stage="1"), ap.TARGET_AUX)
        self.assertEqual(result.blocks[XSA], user_block(max_stage=1))
        self.assertEqual(result.notices, [])

    def test_two_argument_sam_extra_still_gets_five_and_the_log_says_so(self):
        """계획 X10: 늘 다섯을 보낸다(Forge 가 뒤 셋을 버린다) — 구간을 켰으면 로그에 남기고 결과 알림이 말한다."""
        with self.assertLogs("ui.extra_samplers_ui", "INFO") as logs:
            result = sb.build_sampling_blocks(XsampHost(OLD_SAM_EXTRA, sampler=RT, noise_window="1"), ap.TARGET_T2I)
        self.assertEqual(result.blocks[XSA], {"args": [3, 1.0, True, 0.2, 0.8]})
        self.assertTrue(any(xsa.WINDOW_UNSUPPORTED_TEXT in line and "2개" in line for line in logs.output), logs.output)
        with self.assertNoLogs("ui.extra_samplers_ui", "INFO"):
            sb.build_sampling_blocks(XsampHost(OLD_SAM_EXTRA, sampler=RT, eta="0.5"), ap.TARGET_T2I)

    def test_contributor_failure_is_isolated(self):
        with mock.patch.object(xsa, "plan", side_effect=RuntimeError("xsamp broke")), \
                self.assertLogs("generation", "WARNING"):
            payload, error = XsampHost(PRESENT)._build_generation_payload(snapshot=True)
        self.assertIsNone(error)
        self.assertNotIn(XSA, payload["alwayson_scripts"])
        self.assertIn(PAG, payload["alwayson_scripts"])


# ── 실제 진입점 ────────────────────────────────────────────────────────────────
class MainRequestTests(unittest.TestCase):
    def setUp(self):
        _webui(self)

    def test_t2i_payload_carries_the_design_block_and_its_provenance(self):
        """design_samplers.md §4.3 + PLAN_amend X10: 위치 인자 다섯(int, float, bool, float, float)."""
        payload, error = XsampHost(PRESENT)._build_generation_payload(snapshot=True)
        self.assertIsNone(error)
        self.assertEqual(json.dumps(payload["alwayson_scripts"][XSA]), '{"args": [3, 1.0, false, 0.2, 0.8]}')
        self.assertEqual(payload[ap.PROVENANCE_KEY], {XSA: ap.PROVENANCE_APP_DEFAULT})   # 422 면 정보 알림으로 재시도
        user, _ = XsampHost(PRESENT, sampler=RT, max_stage="2", eta="0.5", noise_window="1", noise_start="0.1",
                            noise_end="0.9")._build_generation_payload(snapshot=True)
        self.assertEqual(json.dumps(user["alwayson_scripts"][XSA]), '{"args": [2, 0.5, true, 0.1, 0.9]}')
        self.assertEqual(user["sampler_name"], RT)
        self.assertNotIn(ap.PROVENANCE_KEY, user)
        self.assertEqual(ap.main_retry_titles(user), (XSA,))
        # 계획 R7: 블록 순서의 맨 끝(Extra Schedulers 칸이 있으면 그 바로 뒤)
        host = XsampHost(caps(present=(PAG, SKIM, DD, A38, DORA, DEGRID, xs.SCRIPT_NAME, XSA)))
        from tests.test_extra_schedulers_ui import xsched_widgets
        host.extra_schedulers_widgets = xsched_widgets()
        both, _ = host._build_generation_payload(snapshot=True)
        self.assertEqual(list(both["alwayson_scripts"])[-2:], [xs.SCRIPT_NAME, XSA])

    def test_requests_without_the_feature_stay_byte_identical(self):
        """스냅샷이 확인하지 않았고 ER SDE 패스가 없으면(지금까지의 모든 요청) 요청이 칸이 없던 때와 같다. ComfyUI 는
        블록 하나만 더해진다(계획 C6 — 그래프는 값이 기본값이면 그대로: tests/test_comfy_extra_samplers_compiler.py)."""
        for label, capabilities, backend in (("unknown", None, BackendType.WEBUI),
                                             ("known without the script", WITHOUT_XSA, BackendType.WEBUI),
                                             ("comfy", None, BackendType.COMFYUI)):
            with self.subTest(label), mock.patch("backends.get_backend_type", return_value=backend), \
                    mock.patch("core.comfy_workflow_controls.snapshot_comfy_payload", side_effect=lambda _b, p, _m: p), \
                    mock.patch("ui.comfy_workflow_actions.quality_preset_payload", return_value={}), \
                    mock.patch("core.spectrum_settings.spectrum_payload_from_prefs", return_value={}):
                before, _ = ChainHost(capabilities=capabilities)._build_generation_payload(snapshot=True)
                after, _ = XsampHost(capabilities, eta="3", noise_window="1")._build_generation_payload(snapshot=True)
                if label == "comfy":
                    before = copy.deepcopy(before)
                    before["alwayson_scripts"][XSA] = {"args": [3, 3.0, True, 0.2, 0.8]}
                self.assertEqual(json.dumps(after, sort_keys=True), json.dumps(before, sort_keys=True))

    def test_frozen_payload_keeps_the_block_unless_the_snapshot_says_missing(self):
        host = XsampHost(PRESENT, eta="2")
        payload, _ = host._build_generation_payload(snapshot=True)
        self.assertEqual(sb.freeze_sampling_payload(host, payload), [])
        frozen = copy.deepcopy(payload)
        host.sam_extra_capabilities = None
        with mock.patch("core.sam_extra_probe.peek_capabilities", return_value=None):
            self.assertEqual(sb.regate_frozen_payload(host, frozen), [])
        self.assertIn(XSA, frozen["alwayson_scripts"])
        host.sam_extra_capabilities = WITHOUT_XSA
        notices = sb.regate_frozen_payload(host, frozen)
        self.assertNotIn(XSA, frozen["alwayson_scripts"])
        self.assertEqual([n.code for n in notices], [sn.CODE_BLOCK_NOT_SENT])


class AuxPassTests(unittest.TestCase):
    """블록 없는 요청은 확장 ui() 기본값으로 돈다 — 보조 img2img 도 같은 블록을 받아야 메인과 같은 ER SDE 값으로 돈다."""

    def setUp(self):
        _webui(self)

    def test_hand_repair_snapshot_keeps_the_block(self):
        from ui.hand_reconstruction_actions import _sampling_scripts
        block = user_block(max_stage=2)
        snapshot = {"alwayson_scripts": {PAG: {"args": [1]}, XSA: block}}
        self.assertEqual(_sampling_scripts(snapshot, backend="webui", capabilities=PRESENT)[XSA], block)

    def refine(self, host, refine_settings, request_caps):
        from backends.webui_backend import WebUIBackend
        from tests.test_webui_aux_propagation import ok, png_b64
        from ui.aux_pass_snapshot import attach_sampling_snapshot

        settings = attach_sampling_snapshot(host, refine_settings)
        sent = []

        def post(url, json=None, **_kw):
            sent.append(copy.deepcopy(json))
            return ok()

        with mock.patch("core.sam_extra_probe.peek_capabilities", return_value=request_caps), \
                mock.patch("core.forge_modules.resolve_sam3_checkpoint", side_effect=lambda name: name), \
                mock.patch("backends.webui_backend.requests.post", side_effect=post):
            WebUIBackend("http://127.0.0.1:7860").refine(png_b64(), settings)
        return settings[ap.SETTINGS_KEY], sent[-1]

    def test_unknown_snapshot_refine_with_its_own_er_sde_sampler_gets_the_values(self):
        """FIX-forge(codex:forge): T2I 메인 Euler·Refine 샘플러 ER SDE — 스냅샷을 모르면(연결 직후·조회 실패) 예전에는
        봉투에 블록이 없어 Refine 이 확장 기본값(η 1)으로 돌았다. 이제 봉투는 싣고 보내는 곳이 Refine 요청으로 판정한다."""
        host = XsampHost(None, guidance={"guid_enabled": "false"}, eta="0.25", max_stage="2")
        for refine_sampler in (TUN, RT):
            with self.subTest(refine_sampler=refine_sampler):
                envelope, request = self.refine(host, {"target": "face", "sampler": refine_sampler}, None)
                self.assertEqual(envelope["blocks"][XSA], user_block(eta=0.25, max_stage=2))
                self.assertEqual(request["sampler_name"], refine_sampler)
                self.assertEqual(request["alwayson_scripts"][XSA], user_block(eta=0.25, max_stage=2))
        # 스냅샷이 클릭 때는 몰랐고 보낼 때 확인했어도 같다
        _envelope, request = self.refine(host, {"target": "face", "sampler": TUN}, PRESENT)
        self.assertEqual(request["alwayson_scripts"][XSA], user_block(eta=0.25, max_stage=2))

    def test_unknown_snapshot_refine_that_reads_no_values_carries_no_block(self):
        """계획 X5 는 요청 직전에도 그대로 — Refine 샘플러·SAM3 샘플러가 ER SDE 가 아니면 모르는 Forge 에 싣지 않는다.
        앱 기본값 블록은 모를 때 늘 보내지 않는다(게이트). 스크립트가 없다고 확인되면 사용자 값을 빼고 알린다."""
        host = XsampHost(None, guidance={"guid_enabled": "false"}, eta="0.25")
        for refine_settings in ({"target": "face"}, {"target": "face", "sampler": "Euler"},
                                {"target": "face", "sampler": "Euler a", "sam3_sampler": "Use same sampler"}):
            with self.subTest(refine_settings=refine_settings):
                envelope, request = self.refine(host, refine_settings, None)
                self.assertIn(XSA, envelope["blocks"])
                self.assertNotIn(XSA, request["alwayson_scripts"])
        quiet = XsampHost(None, guidance={"guid_enabled": "false"})                     # 앱 기본값
        _envelope, request = self.refine(quiet, {"target": "face", "sampler": RT}, None)
        self.assertNotIn(XSA, request["alwayson_scripts"])
        _envelope, request = self.refine(host, {"target": "face", "sampler": RT}, WITHOUT_XSA)
        self.assertNotIn(XSA, request["alwayson_scripts"])

    def test_hand_repair_sends_the_values_only_when_its_t2i_sampler_reads_them(self):
        """손 재구성은 T2I 샘플러로 도는 img2img — 스냅샷을 모르면 그 샘플러가 ER SDE 일 때만 싣는다(계획 X5)."""
        from ui.hand_reconstruction_actions import _sampling_gate
        block = user_block(eta=0.5)
        for sampler, expected in ((RT, True), ("er_sde_ode", True), ("Euler", False)):
            with self.subTest(sampler=sampler):
                snapshot = {"sampler_name": sampler, "alwayson_scripts": {PAG: {"args": [1]}, XSA: block}}
                gated = _sampling_gate(snapshot, backend="webui", capabilities=None)
                self.assertIs(XSA in gated.kept, expected)
                self.assertNotIn(XSA, gated.dropped)                              # 쓰지 않는 값 — 알리지 않는다
        snapshot = {"sampler_name": "Euler", "alwayson_scripts": {XSA: block}}
        self.assertIn(XSA, _sampling_gate(snapshot, backend="webui", capabilities=PRESENT).kept)
        self.assertIn(XSA, _sampling_gate(snapshot, backend="comfyui", capabilities=None).kept)

    def test_chat_image_edit_counts_the_t2i_sampler_it_carries(self):
        """FIX-forge(claude:forge-block): 채팅 참조 이미지 편집(i2i)은 T2I 샘플러를 싣는다 — 스냅샷을 모를 때 그 샘플러가
        ER SDE 면 값을 싣는다(예전에는 i2i 라 메인을 세지 않아 Forge 가 η 1 로 돌았다). I2I 탭은 그대로."""
        host = XsampHost(None, sampler=RT, eta="0.5")
        _model, snapshot = host._chat_generation_snapshot("a cat", target="i2i")
        self.assertEqual(snapshot["sampler_name"], RT)
        self.assertEqual(snapshot["alwayson_scripts"][XSA], user_block(eta=0.5))
        euler = XsampHost(None, eta="0.5")                                                # 메인 'Euler'
        _model, snapshot = euler._chat_generation_snapshot("a cat", target="i2i")
        self.assertNotIn(XSA, snapshot["alwayson_scripts"])
        self.assertNotIn(XSA, sb.build_sampling_blocks(host, ap.TARGET_I2I).blocks)       # I2I 탭(요청에 샘플러 없음)
        self.assertNotIn(XSA, sb.build_sampling_blocks(host, ap.TARGET_I2I, request={"prompt": "x"}).blocks)
        self.assertIn(XSA, sb.build_sampling_blocks(host, ap.TARGET_I2I, request={"sampler_name": TUN}).blocks)

    def test_forge_refine_request_carries_the_envelope_block(self):
        from backends.webui_backend import WebUIBackend
        from tests.test_webui_aux_propagation import ok, png_b64
        from ui.aux_pass_snapshot import attach_sampling_snapshot

        host = XsampHost(PRESENT, guidance={"guid_enabled": "false"}, eta="0.25")
        settings = attach_sampling_snapshot(host, {"target": "face"})
        self.assertEqual(settings[ap.SETTINGS_KEY]["blocks"][XSA], user_block(eta=0.25))
        sent = []

        def post(url, json=None, **_kw):
            sent.append(copy.deepcopy(json))
            return ok()

        with mock.patch("core.sam_extra_probe.peek_capabilities", return_value=PRESENT), \
                mock.patch("core.forge_modules.resolve_sam3_checkpoint", side_effect=lambda name: name), \
                mock.patch("backends.webui_backend.requests.post", side_effect=post):
            WebUIBackend("http://127.0.0.1:7860").refine(png_b64(), settings)
        self.assertEqual(sent[-1]["alwayson_scripts"][XSA], user_block(eta=0.25))
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
        self.assertIn("from ui.extra_samplers_ui import init_extra_samplers_proxies", source)
        self.assertIn("self.extra_samplers_widgets = init_extra_samplers_proxies(b)", source)

    def test_init_registers_every_widget_id_with_the_app_defaults(self):
        bridge = _Bridge()
        widgets = xui.init_extra_samplers_proxies(bridge)
        self.assertEqual(set(bridge.registered), {"_xsamp_max_stage", "_xsamp_eta", "_xsamp_noise_window",
                                                  "_xsamp_noise_start", "_xsamp_noise_end"})
        self.assertEqual(set(bridge.registered), {xsa.widget_id(key) for key in xsa.WIDGET_KEYS})
        defaults = {key: widgets[key].text() for key in xsa.WIDGET_KEYS}
        self.assertEqual(defaults, {"max_stage": "3", "eta": "1", "noise_window": "0", "noise_start": "0.2",
                                    "noise_end": "0.8"})
        self.assertEqual(dict(bridge.pushed), {xsa.widget_id(k): v for k, v in defaults.items()})
        host = SimpleNamespace(extra_samplers_widgets=widgets)
        self.assertEqual(xui.current_settings(host), xsa.APP_DEFAULTS)

    def test_window_checkbox_is_a_zero_one_string(self):
        host = SimpleNamespace(extra_samplers_widgets=xsamp_widgets(noise_window="true"))
        self.assertTrue(xui.current_settings(host).noise_window)
        self.assertEqual(xui.get_settings(host)["noise_window"], "1")
        for off in ("0", "false", "", "2", "x"):
            with self.subTest(off=off):
                host = SimpleNamespace(extra_samplers_widgets=xsamp_widgets(noise_window=off))
                self.assertEqual(xui.get_settings(host)["noise_window"], "0")

    def test_pass_values_read_every_pass_widget_of_the_host(self):
        host = ChainHost(capabilities=None)
        host.adetailer_group, host.ad_slot1_group, host.ad_slot2_group = (ReadOnlyWidget(True), ReadOnlyWidget(True),
                                                                          ReadOnlyWidget(False))
        host.s1_widgets = {"use_sampler_check": ReadOnlyWidget(True), "sampler_combo": ReadOnlyWidget(TUN)}
        host.sam3_group = ReadOnlyWidget(True)
        host.sam3_widgets = {"mode": ReadOnlyWidget("Mask only"), "use_sampler_check": ReadOnlyWidget(True),
                             "sampler": ReadOnlyWidget(RT)}
        values = xui.pass_values(host)
        self.assertEqual(tuple(values), xsa.pass_widget_ids())
        self.assertEqual(values["sampler_combo"], "Euler")
        self.assertIs(values["hires_options_group"], True)
        self.assertEqual(values["hires_sampler_combo"], "Use same sampler")
        self.assertIsNone(values["_ad_s2_use_sampler"])                          # 슬롯 2 위젯 dict 없음
        self.assertEqual(xui.passes_in_use(host), (xsa.PassUse("adetailer_1", TUN),))   # SAM3 는 Mask only
        self.assertEqual(xui.pass_values(SimpleNamespace()), dict.fromkeys(xsa.pass_widget_ids()))


class PersistenceTests(unittest.TestCase):
    def test_settings_dict_and_preset_restore(self):
        from tests.test_generation_presets import _host
        from ui.generation_settings_apply import apply_generation_settings
        from ui.generator_settings import _extra_samplers_settings

        host = _host([])
        host.extra_samplers_widgets = xsamp_widgets(max_stage="2", eta="0.35", noise_window="1", noise_start="0.1",
                                                    noise_end="0.7")
        saved = _extra_samplers_settings(host)
        self.assertEqual(saved, {"max_stage": "2", "eta": "0.35", "noise_window": "1", "noise_start": "0.1",
                                 "noise_end": "0.7"})
        other = _host([])
        other.extra_samplers_widgets = xsamp_widgets()
        apply_generation_settings(other, {xsa.SETTINGS_KEY: saved}, only_present=True)
        self.assertEqual(xui.get_settings(other), saved)
        apply_generation_settings(other, {"steps": "30"}, only_present=True)    # 프리셋에 키가 없으면 그대로
        self.assertEqual(xui.get_settings(other), saved)
        apply_generation_settings(other, {"steps": "30"}, only_present=False)   # 옛 설정 파일 — 앱 기본값
        self.assertEqual(xui.current_settings(other), xsa.APP_DEFAULTS)

    def test_values_saved_without_the_window_restore_its_defaults(self):
        """잡음 구간 셋이 없는 저장값(v0.33.0 전 계획의 두 칸): 그 칸은 앱 기본값(끔·0.2·0.8) — 확장이 인자 두 개 요청을
        읽는 것과 같다. 다른 칸은 저장값."""
        host = SimpleNamespace(extra_samplers_widgets=xsamp_widgets(noise_window="1", noise_start="0.4"))
        xui.apply_saved_settings(host, {xsa.SETTINGS_KEY: {"max_stage": 2, "eta": 0.5}}, only_present=True)
        self.assertEqual(xui.current_settings(host), xsa.ExtraSamplersSettings(max_stage=2, eta=0.5))
        self.assertEqual(boot_notices.pending_boot_notices(host), [])

    def test_old_settings_file_starts_with_app_defaults_quietly(self):
        host = SimpleNamespace(extra_samplers_widgets=xsamp_widgets(eta="4", noise_window="1"))
        xui.apply_saved_settings(host, {"steps": "28"}, only_present=False)
        self.assertEqual(xui.current_settings(host), xsa.APP_DEFAULTS)
        self.assertEqual(boot_notices.pending_boot_notices(host), [])
        xui.apply_saved_settings(SimpleNamespace(), {xsa.SETTINGS_KEY: {"eta": "1"}}, only_present=False)   # 위젯 없음

    def test_saved_values_are_normalized_on_restore(self):
        host = SimpleNamespace(extra_samplers_widgets=xsamp_widgets())
        xui.apply_saved_settings(host, {xsa.SETTINGS_KEY: {"max_stage": "2.6", "eta": "99", "noise_window": "on",
                                                           "noise_start": "-1", "noise_end": "abc"}},
                                 only_present=True)
        self.assertEqual(xui.current_settings(host),
                         xsa.ExtraSamplersSettings(max_stage=3, eta=10.0, noise_window=True, noise_start=0.0,
                                                   noise_end=0.8))
        self.assertEqual(xui.get_settings(host), {"max_stage": "3", "eta": "10", "noise_window": "1",
                                                  "noise_start": "0", "noise_end": "0.8"})

    def test_generation_keys_and_settings_dict(self):
        from core.generation_presets import GENERATION_KEYS
        self.assertIn(xsa.SETTINGS_KEY, GENERATION_KEYS)
        self.assertEqual(GENERATION_KEYS.index(xsa.SETTINGS_KEY), GENERATION_KEYS.index(xs.SETTINGS_KEY) + 1)
        source = (REPO / "ui" / "generator_settings.py").read_text(encoding="utf-8")
        self.assertIn('"extra_samplers_settings": _extra_samplers_settings(self)', source)
        apply_source = (REPO / "ui" / "generation_settings_apply.py").read_text(encoding="utf-8")
        self.assertIn("_apply_extra_samplers_settings(host, settings, only_present=only_present)", apply_source)


class ApplyInfotextTests(unittest.TestCase):
    """붙여 넣기(계획 A-R1 이 '이 설정으로 생성'에 잇는다) — 확장 paste_* 다섯의 규칙: 늘 다섯 칸을 정한다."""

    def test_values_come_back_and_missing_keys_are_the_defaults(self):
        host = SimpleNamespace(extra_samplers_widgets=xsamp_widgets(max_stage="1", eta="3"))
        restored = xui.apply_infotext(host, {"Sampler": TUN, "ER SDE max stage": "2", "ER SDE eta": "0.5",
                                             "ER SDE noise window": "0.1-0.9"})
        self.assertEqual(restored, xsa.ExtraSamplersSettings(max_stage=2, eta=0.5, noise_window=True, noise_start=0.1,
                                                             noise_end=0.9))
        self.assertEqual(xui.current_settings(host), restored)
        self.assertEqual(xui.get_settings(host)["noise_window"], "1")
        xui.apply_infotext(host, {"Sampler": "Euler a"})                       # 키가 없다 — 확장처럼 기본값
        self.assertEqual(xui.current_settings(host), xsa.APP_DEFAULTS)
        self.assertIsNone(xui.apply_infotext(SimpleNamespace(), {"ER SDE eta": 1}))

    def test_forge_eta_of_a_reverse_time_infotext_is_the_eta(self):
        """aoleg/Neo_ExtraSchedulers 이미지: 'ER SDE eta' 없이 Forge 'Eta' — Reverse-time 일 때만 η 로 읽는다."""
        host = SimpleNamespace(extra_samplers_widgets=xsamp_widgets())
        xui.apply_infotext(host, {"Sampler": RT, "Eta": "0.7"})
        self.assertEqual(xui.current_settings(host).eta, 0.7)
        xui.apply_infotext(host, {"Sampler": TUN, "Eta": "0.7"})
        self.assertEqual(xui.current_settings(host).eta, 1.0)
        xui.apply_infotext(host, {"Sampler": "Euler a", "Hires sampler": RT, "Eta": "0.4", "ER SDE noise window": "off"})
        self.assertEqual(xui.current_settings(host), xsa.ExtraSamplersSettings(eta=0.4))


if __name__ == "__main__":
    unittest.main()
