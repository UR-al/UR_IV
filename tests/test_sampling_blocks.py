"""샘플링 블록 빌더(ui/sampling_blocks)와 메인 체인 — P7 이음새, 게이트, 알림은 보내는 곳에서(B15), 동결 재게이트(B16)."""
import copy
import json
import threading
import unittest
from types import SimpleNamespace
from unittest import mock

from backends import BackendType
from core import alwayson_propagation as ap
from core import anima38, anima_guidance
from core import sam_extra_notices as sn
from tests.test_alwayson_propagation import A38, DD, DORA, PAG, SKIM, caps
from tests.test_chat_generation_snapshot import ReadOnlyWidget, SnapshotHost
from ui import sampling_blocks as sb
from ui.generator_generation import GenerationMixin

GUIDANCE_ON = {"guid_enabled": "true", "skim_enabled": "true", "dd_enabled": "true"}
ALL_PRESENT = caps(present=(PAG, SKIM, DD, A38, DORA))
URL = "http://127.0.0.1:7860"


class _Signal:
    def __init__(self):
        self.calls = []

    def emit(self, *args):
        self.calls.append(args)


class ChainHost(SnapshotHost):
    """실제 위젯 읽기 + 실제 메인 체인(SnapshotHost 의 가짜 체인을 되돌린다)."""
    _apply_postprocess_chain = GenerationMixin._apply_postprocess_chain

    def __init__(self, guidance=None, capabilities=None, negpip=True):
        super().__init__()
        self.adetailer_group = ReadOnlyWidget(False)
        self.sam3_group = None
        self.negpip_group = ReadOnlyWidget(negpip)
        self.anima_guidance_widgets = {k: ReadOnlyWidget(v) for k, v in (guidance or GUIDANCE_ON).items()}
        self.sam_extra_capabilities = capabilities
        self.vue_bridge = SimpleNamespace(showNotification=_Signal())


def _webui(test, backend_type=BackendType.WEBUI):
    for patcher in (mock.patch("backends.get_backend_type", return_value=backend_type),
                    mock.patch("backends.get_backend", return_value=SimpleNamespace(api_url=URL)),
                    mock.patch("core.sam_extra_probe.peek_capabilities", return_value=None)):
        patcher.start()
        test.addCleanup(patcher.stop)


def old_chain_scripts(host):
    """P7 전 _build_generation_payload 의 alwayson 조립(NegPiP 줄 → ADetailer·SAM3 → anima_guidance.apply_to_payload)."""
    payload = {"alwayson_scripts": {}}
    if host.negpip_group.isChecked():
        payload["alwayson_scripts"]["NegPiP"] = {"args": [True]}
    anima_guidance.apply_to_payload(payload, host._build_anima_settings())
    return payload["alwayson_scripts"]


class DefaultBehaviourTests(unittest.TestCase):
    def setUp(self):
        _webui(self)

    def test_default_main_payload_is_byte_identical_to_the_pre_p7_chain(self):
        for label, capabilities, backend in (("unknown", None, BackendType.WEBUI),
                                             ("known, all present", ALL_PRESENT, BackendType.WEBUI),
                                             ("comfy", None, BackendType.COMFYUI)):
            for guidance in (GUIDANCE_ON, {"guid_enabled": "false"}):
                with self.subTest(label, guidance=guidance), \
                        mock.patch("backends.get_backend_type", return_value=backend), \
                        mock.patch("core.comfy_workflow_controls.snapshot_comfy_payload",
                                   side_effect=lambda _b, p, _m: p), \
                        mock.patch("ui.comfy_workflow_actions.quality_preset_payload", return_value={}), \
                        mock.patch("core.spectrum_settings.spectrum_payload_from_prefs", return_value={}):
                    host = ChainHost(guidance, capabilities)
                    payload, error = host._build_generation_payload(snapshot=True)
                    self.assertIsNone(error)
                    self.assertEqual(json.dumps(payload["alwayson_scripts"]),
                                     json.dumps(old_chain_scripts(host)))
                    self.assertEqual(host.vue_bridge.showNotification.calls, [])

    def test_one_argument_chain_call_is_kept_for_t2i(self):
        calls = []

        class _Host(ChainHost):
            def _apply_postprocess_chain(self, payload):   # tests/test_chat_generation_snapshot 의 1-인자 더블
                calls.append(payload)

        _Host()._build_generation_payload(snapshot=True)
        self.assertEqual(len(calls), 1)


class BuilderTests(unittest.TestCase):
    def setUp(self):
        _webui(self)

    def test_contributor_order_first_wins_and_failure_is_isolated(self):
        def first(_host, _ctx):
            return sb.Contribution().add(PAG, {"args": ["first"]})

        def boom(_host, _ctx):
            raise RuntimeError("widget gone")

        def second(_host, _ctx):
            return sb.Contribution().add(PAG, {"args": ["second"]}).add(SKIM, {"args": [1]})

        with mock.patch.object(sb, "CONTRIBUTORS", (first, boom, second)), \
                self.assertLogs("generation", "WARNING") as logs:
            result = sb.build_sampling_blocks(ChainHost(capabilities=None), ap.TARGET_T2I)
        self.assertEqual(result.blocks, {PAG: {"args": ["first"]}, SKIM: {"args": [1]}})
        self.assertTrue(any("boom" in message and "widget gone" in message for message in logs.output))

    def test_contributors_without_widgets_add_nothing_quietly(self):
        class _Bare(GenerationMixin):
            pass

        with self.assertNoLogs("generation", "WARNING"):
            result = sb.build_sampling_blocks(_Bare(), ap.TARGET_T2I)
        self.assertEqual(result.blocks, {})
        self.assertEqual(result.notices, [])

    def test_t19_builder_titles_are_all_in_propagation(self):
        result = sb.build_sampling_blocks(ChainHost(capabilities=ALL_PRESENT), ap.TARGET_AUX)
        self.assertEqual(list(result.blocks), [ap.TITLE_NEGPIP, PAG, SKIM, DD])
        for title in result.blocks:
            self.assertIn(ap.canonical_title(title), ap.PROPAGATION)

    def test_webui_main_gate_drops_known_missing_and_returns_a_warning_notice(self):
        capabilities = caps(present=(SKIM, DD), img2img=(DD,))
        t2i = sb.build_sampling_blocks(ChainHost(capabilities=capabilities), ap.TARGET_T2I)
        self.assertEqual(list(t2i.blocks), [ap.TITLE_NEGPIP, SKIM, DD])
        self.assertEqual(t2i.dropped, (PAG,))
        self.assertEqual([(n.code, n.level) for n in t2i.notices], [(sn.CODE_BLOCK_NOT_SENT, sn.LEVEL_WARNING)])
        i2i = sb.build_sampling_blocks(ChainHost(capabilities=capabilities), ap.TARGET_I2I)
        self.assertEqual(list(i2i.blocks), [ap.TITLE_NEGPIP, DD])       # img2img 목록으로
        self.assertEqual(set(i2i.dropped), {PAG, SKIM})

    def test_comfy_krea2_and_aux_are_not_gated_here(self):
        capabilities = caps(present=())
        with mock.patch("backends.get_backend_type", return_value=BackendType.COMFYUI):
            comfy = sb.build_sampling_blocks(ChainHost(capabilities=capabilities), ap.TARGET_T2I)
        krea2 = sb.build_sampling_blocks(ChainHost(capabilities=capabilities), ap.TARGET_T2I, krea2=True)
        aux = sb.build_sampling_blocks(ChainHost(capabilities=capabilities), ap.TARGET_AUX)   # 워커가 게이트
        for result in (comfy, krea2, aux):
            self.assertEqual(list(result.blocks), [ap.TITLE_NEGPIP, PAG, SKIM, DD])
            self.assertEqual(result.dropped, ())
        self.assertEqual(comfy.context.backend, ap.BACKEND_COMFY)
        self.assertEqual(krea2.context.backend, ap.BACKEND_KREA2)

    def test_provenance_app_default_is_skipped_when_unknown_and_dropped_quietly(self):
        """(A6) P8·P9 가 호출부를 고치지 않고 출처만 적으면 되는지 — 가짜 Anima38 기여자로."""
        def anima38_contributor(provenance):
            return lambda _host, _ctx: sb.Contribution().add(A38, {"args": [{"negative": True}]},
                                                             provenance=provenance)

        cases = (
            (ap.PROVENANCE_APP_DEFAULT, None, [], []),                                  # 모름 → 조용히 안 보냄
            (ap.PROVENANCE_USER, None, [A38], []),                                      # 모름 → 사용자 값은 보냄
            (ap.PROVENANCE_APP_DEFAULT, caps(present=()), [], []),                      # 없음 → 로그만
            (ap.PROVENANCE_USER, caps(present=()), [], [sn.CODE_BLOCK_NOT_SENT]),       # 없음 → 경고
            (ap.PROVENANCE_APP_DEFAULT, caps(present=(A38,)), [A38], []),
        )
        for provenance, capabilities, kept, codes in cases:
            with self.subTest(provenance=provenance, caps=capabilities), \
                    mock.patch.object(sb, "CONTRIBUTORS", (anima38_contributor(provenance),)):
                result = sb.build_sampling_blocks(ChainHost(capabilities=capabilities), ap.TARGET_T2I)
                self.assertEqual(list(result.blocks), kept)
                self.assertEqual([n.code for n in result.notices], codes)
                self.assertEqual(result.provenance, {A38: provenance})
        with self.assertRaises(ValueError):
            sb.Contribution().add(A38, {}, provenance="guess")

    def test_hires_note_and_summary_logs_are_unchanged(self):
        host = ChainHost({"dd_enabled": "true", "dd_hires": "true"},
                         capabilities=SimpleNamespace(detail_daemon_hires=False))
        with self.assertLogs("generation", "INFO") as logs:
            sb.build_sampling_blocks(host, ap.TARGET_T2I)
        self.assertIn(anima_guidance.DD_HIRES_NOTE_OLD_EXTENSION, [r.getMessage() for r in logs.records])
        self.assertTrue(any(r.getMessage().startswith("Anima Guidance 적용됨: ") for r in logs.records))


class EnvelopeParityTests(unittest.TestCase):
    def setUp(self):
        _webui(self)

    def test_t15_main_chain_sampling_part_equals_the_aux_envelope(self):
        from ui.aux_pass_snapshot import capture_sampling_envelope
        host = ChainHost(capabilities=ALL_PRESENT)
        payload, _ = host._build_generation_payload(snapshot=True)
        main = {k: v for k, v in payload["alwayson_scripts"].items() if ap.canonical_title(k) in ap.PROPAGATION}
        env = capture_sampling_envelope(host)
        self.assertEqual(json.dumps(main), json.dumps(env["blocks"]))
        self.assertEqual(env["model"], "Anima-3.8B-v1.1.safetensors")
        self.assertEqual(env["backend"], ap.BACKEND_WEBUI)


class NoticeRoutingTests(unittest.TestCase):
    """(B15) 빌더·체인은 알림을 돌려주기만 하고, 보내는 곳이 띄운다."""

    def setUp(self):
        _webui(self)

    def test_notices_are_shown_only_at_the_send_site(self):
        from ui.sam_extra_notices_ui import check_before_generation
        host = ChainHost(capabilities=caps(present=(SKIM, DD)))
        payload, _ = host._build_generation_payload(snapshot=True)
        self.assertNotIn(PAG, payload["alwayson_scripts"])
        self.assertEqual(host.vue_bridge.showNotification.calls, [])        # 만들 때는 띄우지 않는다
        other, _ = host._build_generation_payload(snapshot=True)             # 보내지 않는 빌드(XYZ·Comfy 사전 점검)
        self.assertEqual(sb.take_sampling_notices(host, payload), [])        # 옛 페이로드의 알림은 버려진다
        with self.assertLogs("ui.sam_extra_notices_ui", "WARNING"):
            self.assertGreaterEqual(check_before_generation(host, other), 1)
        level, message = host.vue_bridge.showNotification.calls[0]
        self.assertEqual(level, "warning")
        self.assertIn(PAG, message)
        self.assertEqual(sb.take_sampling_notices(host, other), [])          # 한 번만

    def test_move_follows_copies(self):
        host, first, second = SimpleNamespace(), {}, {}
        sb.remember_sampling_notices(host, first, ["n"])
        sb.move_sampling_notices(host, {}, second)
        self.assertEqual(sb.take_sampling_notices(host, second), [])
        sb.move_sampling_notices(host, first, second)
        self.assertEqual(sb.take_sampling_notices(host, first), [])
        self.assertEqual(sb.take_sampling_notices(host, second), ["n"])

    def test_chat_snapshot_moves_notices_to_its_copy(self):
        host = ChainHost(capabilities=caps(present=(SKIM, DD)))
        _model, snapshot = host._chat_generation_snapshot("portrait")
        self.assertEqual([n.code for n in sb.take_sampling_notices(host, snapshot)], [sn.CODE_BLOCK_NOT_SENT])


class ChatImageEditTargetTests(unittest.TestCase):
    """(A4) 채팅 이미지 편집(참조 이미지 → img2img)은 I2I 규칙으로 샘플링 블록을 만든다."""

    def setUp(self):
        _webui(self)

    def test_snapshot_target_i2i_gates_with_the_img2img_list(self):
        capabilities = caps(present=(PAG, SKIM, DD), img2img=(SKIM, DD))    # PAG 는 txt2img 에만
        t2i = ChainHost(capabilities=capabilities)._chat_generation_snapshot("portrait")[1]
        i2i = ChainHost(capabilities=capabilities)._chat_generation_snapshot("portrait", target="i2i")[1]
        self.assertIn(PAG, t2i["alwayson_scripts"])
        self.assertNotIn(PAG, i2i["alwayson_scripts"])
        self.assertIn(SKIM, i2i["alwayson_scripts"])

    def test_chat_media_passes_i2i_only_with_a_reference_image(self):
        from core.chat_generation import GenerationPlan
        from ui.chat_actions import ChatActionsMixin

        class _Host(ChatActionsMixin):
            def __init__(self):
                self.events = []
                self._chat_generation_snapshot = mock.Mock(return_value=("m", {"prompt": "cat"}))

            def _chat_emit_media(self, event):
                self.events.append(event)

        for image, expected in (("", mock.call("cat")),
                                ("data:image/png;base64,AAAA", mock.call("cat", target="i2i"))):
            with self.subTest(image=bool(image)), \
                    mock.patch("ui.chat_actions.threading.Thread") as thread, \
                    mock.patch("core.resource_coordinator.get_generation_coordinator") as coordinator:
                coordinator.return_value.generation_active.return_value = False
                host = _Host()
                host._chat_start_media("rid", GenerationPlan(kind="image", family="current", prompt="cat",
                                                             image=image))
                self.assertEqual(host._chat_generation_snapshot.call_args, expected)
                thread.return_value.start.assert_called_once()


class FrozenPayloadTests(unittest.TestCase):
    """(B16) 시드 탐색·XYZ 대기열의 동결 페이로드는 보낼 때 다시 게이트한다."""

    def setUp(self):
        _webui(self)

    def _frozen(self):
        return {"prompt": "p", "alwayson_scripts": {"SAM3 Mask": {"args": [{}]}, PAG: {"args": [1]},
                                                    SKIM: {"args": [2]}, ap.TITLE_NEGPIP: {"args": [True]}}}

    def test_known_missing_is_dropped_and_the_notice_waits_for_the_send_site(self):
        host = SimpleNamespace(sam_extra_capabilities=caps(present=(SKIM,)))
        payload = self._frozen()
        notices = sb.regate_frozen_payload(host, payload)
        self.assertEqual([n.code for n in notices], [sn.CODE_BLOCK_NOT_SENT])
        self.assertEqual(list(payload["alwayson_scripts"]), ["SAM3 Mask", SKIM, ap.TITLE_NEGPIP])
        self.assertEqual(sb.take_sampling_notices(host, payload), notices)

    def test_unknown_comfy_and_krea2_keep_the_frozen_blocks(self):
        for label, host, payload, backend_type in (
                ("unknown", SimpleNamespace(sam_extra_capabilities=None), self._frozen(), BackendType.WEBUI),
                ("comfy", SimpleNamespace(sam_extra_capabilities=caps(present=())), self._frozen(),
                 BackendType.COMFYUI),
                ("krea2", SimpleNamespace(sam_extra_capabilities=caps(present=())),
                 {**self._frozen(), "_generation_family": "krea2"}, BackendType.WEBUI)):
            with self.subTest(label), mock.patch("backends.get_backend_type", return_value=backend_type):
                before = copy.deepcopy(payload)
                self.assertEqual(sb.regate_frozen_payload(host, payload), [])
                self.assertEqual(payload, before)

    def test_explicit_backend_uses_its_own_snapshot(self):
        backend = SimpleNamespace(api_url="http://remote:7860", get_backend_type=lambda: "webui")
        payload = self._frozen()
        with mock.patch("core.sam_extra_probe.peek_capabilities", return_value=caps(present=(PAG,))) as peek:
            sb.regate_frozen_payload(SimpleNamespace(sam_extra_capabilities=None), payload, backend=backend)
        peek.assert_called_with("http://remote:7860")
        self.assertEqual(list(payload["alwayson_scripts"]), ["SAM3 Mask", PAG, ap.TITLE_NEGPIP])

    def test_start_generation_regates_the_override_copy(self):
        host = SimpleNamespace(gen_worker=None, sam_extra_capabilities=caps(present=(SKIM,)))
        host._abort_generation = mock.Mock()
        frozen = {**self._frozen(), "steps": 0}                 # 검증에서 멈추게(워커를 만들지 않는다)
        before = copy.deepcopy(frozen)
        with mock.patch("ui.sampling_blocks.regate_frozen_payload",
                        wraps=sb.regate_frozen_payload) as regate:
            self.assertFalse(GenerationMixin.start_generation(host, payload_override=frozen))
        self.assertEqual(frozen, before)                        # 원본(대기열 항목)은 그대로
        sent = regate.call_args.args[1]
        self.assertIsNot(sent, frozen)
        self.assertNotIn(PAG, sent["alwayson_scripts"])
        host._abort_generation.assert_called_once()

    def test_regate_keeps_app_default_drops_quiet_and_never_sends_the_private_key(self):
        """(P7-R2 b) 클릭 때 적어 둔 앱 기본값 출처 — 보낼 때 다시 빠져도 경고하지 않는다(A6). 키는 어느 경로든 뗀다."""
        host = SimpleNamespace(sam_extra_capabilities=caps(present=(SKIM,)))
        payload = {**self._frozen(), sb.FROZEN_PROVENANCE_KEY: {A38: ap.PROVENANCE_APP_DEFAULT}}
        payload["alwayson_scripts"][A38] = {"args": [{"negative": True}]}
        with self.assertLogs("core.alwayson_propagation", "INFO") as logs:
            notices = sb.regate_frozen_payload(host, payload)
        self.assertEqual([(n.code, n.detail) for n in notices], [(sn.CODE_BLOCK_NOT_SENT, f"{PAG}@txt2img")])
        self.assertTrue(any(A38 in line for line in logs.output))                 # 앱 기본값은 로그만
        self.assertNotIn(A38, payload["alwayson_scripts"])
        self.assertNotIn(sb.FROZEN_PROVENANCE_KEY, payload)
        for backend_type, extra in ((BackendType.COMFYUI, {}), (BackendType.WEBUI, {"_generation_family": "krea2"})):
            with self.subTest(backend_type=backend_type, **extra), \
                    mock.patch("backends.get_backend_type", return_value=backend_type):
                other = {**self._frozen(), **extra, sb.FROZEN_PROVENANCE_KEY: {A38: ap.PROVENANCE_APP_DEFAULT}}
                self.assertEqual(sb.regate_frozen_payload(host, other), [])
                self.assertNotIn(sb.FROZEN_PROVENANCE_KEY, other)


    def test_regate_hands_the_remaining_app_default_provenance_to_the_forge_backend(self):
        """(P10 검토 2) Forge 로 보낼 때는 남은 앱 기본값 블록의 출처를 다시 적어 백엔드(메인 422 재시도 알림)가 쓰게 한다 —
        백엔드가 요청 전에 떼므로 Forge 요청 JSON 에는 나가지 않는다."""
        host = SimpleNamespace(sam_extra_capabilities=caps(present=(PAG, SKIM, A38)))
        payload = {**self._frozen(),
                   sb.FROZEN_PROVENANCE_KEY: {A38: ap.PROVENANCE_APP_DEFAULT, DORA: ap.PROVENANCE_APP_DEFAULT}}
        payload["alwayson_scripts"][A38] = {"args": [{"negative": True}]}
        self.assertEqual(sb.regate_frozen_payload(host, payload), [])
        self.assertEqual(payload[sb.FROZEN_PROVENANCE_KEY], {A38: ap.PROVENANCE_APP_DEFAULT})   # 보낼 블록만


class RequestProvenanceTests(unittest.TestCase):
    """(P10 검토 2, A2) 메인 체인은 이번 빌드가 넣은 **앱 기본값** 블록의 출처를 페이로드의 비공개 키에 적는다 — Forge
    백엔드가 요청 전에 떼어 메인 422 재시도 알림(앱 기본값 = 정보, 그 밖 = 경고)에 쓴다. 앱 기본값 블록이 없으면 키가 없다."""

    def setUp(self):
        _webui(self)

    @staticmethod
    def _contributor(provenance):
        def add_anima38(_host, _ctx):
            return sb.Contribution().add(A38, {"args": [{"negative": True}]}, provenance=provenance)
        return add_anima38

    def test_forge_main_payload_carries_app_default_provenance_only(self):
        for provenance, expected in ((ap.PROVENANCE_APP_DEFAULT, {A38: ap.PROVENANCE_APP_DEFAULT}),
                                     (ap.PROVENANCE_USER, None)):
            with self.subTest(provenance=provenance), \
                    mock.patch.object(sb, "CONTRIBUTORS", (*sb.CONTRIBUTORS, self._contributor(provenance))):
                host = ChainHost(capabilities=ALL_PRESENT)
                payload, error = host._build_generation_payload(snapshot=True)
                self.assertIsNone(error)
                self.assertIn(A38, payload["alwayson_scripts"])
                self.assertEqual(payload.get(sb.FROZEN_PROVENANCE_KEY), expected)

    def test_only_blocks_this_build_inserted_for_a_forge_main_request_are_marked(self):
        block = {"args": [{"negative": True}]}
        built = sb.SamplingBlocks({A38: block}, [], (), {A38: ap.PROVENANCE_APP_DEFAULT},
                                  sb.SamplingContext(ap.TARGET_T2I, ap.BACKEND_WEBUI))
        mine = {"alwayson_scripts": {A38: block}}
        sb.mark_request_provenance(mine, built)
        self.assertEqual(mine[sb.FROZEN_PROVENANCE_KEY], {A38: ap.PROVENANCE_APP_DEFAULT})
        callers = {"alwayson_scripts": {A38: {"args": ["caller"]}}}          # 호출자가 먼저 넣은 블록(setdefault 가 이김)
        sb.mark_request_provenance(callers, built)
        self.assertNotIn(sb.FROZEN_PROVENANCE_KEY, callers)
        for target, backend in ((ap.TARGET_AUX, ap.BACKEND_WEBUI), (ap.TARGET_T2I, ap.BACKEND_COMFY),
                                (ap.TARGET_I2I, ap.BACKEND_KREA2)):
            with self.subTest(target=target, backend=backend):
                other = sb.SamplingBlocks({A38: block}, [], (), {A38: ap.PROVENANCE_APP_DEFAULT},
                                          sb.SamplingContext(target, backend))
                payload = {"alwayson_scripts": {A38: block}}
                sb.mark_request_provenance(payload, other)
                self.assertNotIn(sb.FROZEN_PROVENANCE_KEY, payload)            # Forge 메인 요청만


class _Queue:
    """대기열 패널·매니저 더블 — 매니저 시작 순간에 이미 띄운 토스트 수를 적는다(알림은 대기열이 돌기 전에)."""

    def __init__(self, host):
        self.items = []
        self.started_after = []
        host.queue_panel = SimpleNamespace(add_single_item=self.items.append)
        host.queue_manager = SimpleNamespace(
            is_running=False, total_count=0,
            start=lambda: self.started_after.append(len(host.vue_bridge.showNotification.calls)))


class _StartHost(ChainHost):
    """start_generation 이 생성 전 확인까지 가는 최소 호스트(워커는 테스트가 바꿔 끼운다)."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.gen_worker = None
        self.btn_generate = mock.Mock()
        self.vue_bridge.send_start = lambda: None

    def setWindowTitle(self, _title):
        pass

    def show_status(self, *_args):
        pass

    def _maybe_unload_ollama(self):
        pass

    def _cleanup_gen_worker(self):
        pass

    def on_generation_finished(self, *_args):
        pass

    def _on_generation_progress(self, *_args):
        pass


class FrozenSendSiteTests(unittest.TestCase):
    """(P7-R2) 동결 페이로드(XYZ·시드 탐색)는 클릭이 보내는 곳이다 — 빌드 게이트가 뺀 사용자 블록을 그 클릭에서 한 번
    알린다. 대기열 항목은 그 블록이 이미 빠진 사본이라 보낼 때(regate_frozen_payload)는 다시 볼 수 없다."""

    def setUp(self):
        _webui(self)

    @staticmethod
    def _pag_warnings(host):
        return [call for call in host.vue_bridge.showNotification.calls if call[0] == "warning" and PAG in call[1]]

    def test_xyz_click_announces_the_dropped_block_once_before_the_queue_starts(self):
        import backends
        from ui.sam_extra_notices_ui import check_before_generation
        from ui.xyz_actions import XYZActionsMixin

        class _Host(ChainHost, XYZActionsMixin):
            pass

        host = _Host(capabilities=caps(present=(SKIM, DD)))                    # PAG 는 연결된 Forge 에 없다
        queue = _Queue(host)
        host.gen_worker = None
        host._xyz_lock = threading.RLock()
        host._xyz_seen_requests = set()
        host._xyz_emit = mock.Mock()
        host._xyz_capabilities = {"backend": backends.get_backend(), "context": host._xyz_context(), "data": {
            "capabilityId": "cap",
            "axes": [{"id": "steps", "label": "Steps", "type": "integer", "min": 1, "max": 150}]}}
        host._xyz_start_plot({"requestId": "plot", "capabilityId": "cap",
                              "axes": [{"id": "steps", "values": [20, 30]}]})
        self.assertTrue(host._xyz_emit.call_args.args[1]["ok"], host._xyz_emit.call_args)
        self.assertEqual(len(queue.items), 2)
        for item in queue.items:
            self.assertNotIn(PAG, item["alwayson_scripts"])
            self.assertNotIn(sb.FROZEN_PROVENANCE_KEY, item)                     # 앱 기본값 블록이 없으면 그대로
        self.assertEqual(len(self._pag_warnings(host)), 1)
        self.assertEqual(queue.started_after, [1])
        payload, _model, _backend = host._xyz_prepare_queue_generation(queue.items[0])
        sb.regate_frozen_payload(host, payload)
        check_before_generation(host, payload)
        self.assertEqual(len(self._pag_warnings(host)), 1)                       # 보낼 때 다시 뜨지 않는다

    def test_seed_explore_click_announces_the_dropped_block_once(self):
        from ui.seed_explore_actions import start_seed_explore
        host = ChainHost(capabilities=caps(present=(SKIM, DD)))
        queue = _Queue(host)
        host.gen_worker = None
        host.is_automating = False
        identity = mock.Mock(side_effect=lambda text: text)
        with mock.patch("core.standard_hooks.run_pipeline_on_text", identity), \
                mock.patch("utils.wildcard.process_wildcards", identity), \
                mock.patch("utils.file_wildcard.resolve_file_wildcards", identity):
            self.assertTrue(start_seed_explore(host, {"seed": "5"}))
        self.assertEqual(len(queue.items), 9)
        self.assertFalse(any(PAG in item["alwayson_scripts"] for item in queue.items))
        self.assertEqual(len(self._pag_warnings(host)), 1)
        self.assertEqual(queue.started_after, [1])

    def test_click_records_app_default_provenance_so_dispatch_drops_stay_quiet(self):
        """(P7-R2 b) 앱 기본값 블록(P8·P9)이 클릭과 보내기 사이에 '없음'이 되면 로그만, 사용자 블록은 경고."""
        def app_default_anima38(_host, _ctx):
            return sb.Contribution().add(A38, {"args": [{"negative": True}]}, provenance=ap.PROVENANCE_APP_DEFAULT)

        host = ChainHost(capabilities=ALL_PRESENT)
        with mock.patch.object(sb, "CONTRIBUTORS", (*sb.CONTRIBUTORS, app_default_anima38)):
            base, _ = host._build_generation_payload(snapshot=True)
        self.assertEqual(sb.freeze_sampling_payload(host, base), [])
        self.assertEqual(base[sb.FROZEN_PROVENANCE_KEY], {A38: ap.PROVENANCE_APP_DEFAULT})
        sent = copy.deepcopy(base)                                               # 대기열 항목 → 보낼 사본
        host.sam_extra_capabilities = caps(present=(SKIM, DD))                   # 그 사이 PAG·Anima38 이 없어졌다
        notices = sb.regate_frozen_payload(host, sent)
        self.assertEqual([n.detail for n in notices], [f"{PAG}@txt2img"])
        self.assertFalse({PAG, A38} & set(sent["alwayson_scripts"]))
        self.assertNotIn(sb.FROZEN_PROVENANCE_KEY, sent)

    def test_start_generation_moves_the_notices_of_a_fresh_override_and_regate_appends(self):
        """(P7-R2 a) ``_comfy_queued_controls`` 처럼 보내기 직전에 새로 만든 페이로드 — start_generation 의 사본으로
        알림을 옮기고, 재게이트 알림은 그 뒤에 붙는다(덮지 않는다)."""
        host = _StartHost(capabilities=caps(present=(SKIM, DD)))                # 빌드 때 PAG 제외(알림이 묶인다)
        built, error = host._build_generation_payload(snapshot=True)             # 더블은 위젯 쓰기를 막는다
        self.assertIsNone(error)
        host.sam_extra_capabilities = caps(present=(DD,))                        # 보낼 때 Skimmed 도 없어졌다
        with mock.patch("ui.generator_generation.GenerationFlowWorker") as worker, \
                self.assertLogs("ui.sam_extra_notices_ui", "WARNING"):
            self.assertTrue(host.start_generation(payload_override=built))
        worker.return_value.start.assert_called_once()
        sent = worker.call_args.args[1]
        self.assertFalse({PAG, SKIM} & set(sent["alwayson_scripts"]))
        warnings = [message for level, message in host.vue_bridge.showNotification.calls if level == "warning"]
        self.assertEqual([title for title in (PAG, SKIM) if any(title in m for m in warnings)], [PAG, SKIM])


class ChatSendSiteTests(unittest.TestCase):
    """(P7 검토 4) 채팅 이미지 요청은 보내는 곳이다 — 실제 스냅샷에 묶인 알림을 한 번 띄운다(Krea2 는 Forge 경로가 아니다)."""

    def setUp(self):
        _webui(self)

    def _start(self, capabilities, *, family="Anima", contributors=None):
        from core.chat_generation import GenerationPlan
        from ui.chat_actions import ChatActionsMixin

        class _Host(ChainHost, ChatActionsMixin):
            def _chat_emit_media(self, _event):
                pass

            def _ensure_creator_runtime(self):
                pass

        host = _Host(capabilities=capabilities)
        host.generation_family_combo = ReadOnlyWidget(family)
        with mock.patch("ui.chat_actions.threading.Thread"), \
                mock.patch("core.resource_coordinator.get_generation_coordinator") as coordinator, \
                mock.patch.object(sb, "CONTRIBUTORS", contributors or sb.CONTRIBUTORS):
            coordinator.return_value.generation_active.return_value = False
            host._chat_start_media("rid", GenerationPlan(kind="image", family="current", prompt="cat"))
        return host.vue_bridge.showNotification.calls

    def test_current_family_shows_the_gate_notice_once(self):
        calls = self._start(caps(present=(SKIM, DD)))
        self.assertEqual([(level, PAG in message) for level, message in calls], [("warning", True)])

    def test_krea2_snapshot_shows_no_forge_notice(self):
        def noisy(_host, _ctx):
            contribution = sb.Contribution()
            contribution.notices.append(sn.block_deferred_notice(PAG))
            return contribution

        self.assertEqual(len(self._start(None, contributors=(noisy,))), 1)          # 같은 기여자 알림이 current 에선 뜬다
        self.assertEqual(self._start(None, family="Krea2", contributors=(noisy,)), [])


if __name__ == "__main__":
    unittest.main()
