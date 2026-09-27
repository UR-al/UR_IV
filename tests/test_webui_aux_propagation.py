"""Forge 보조 패스(Refine·SAM3·ADetailer) 전달·재시도 — 가짜 requests.post (P7 T8-T10, A1, A3, B10)."""
import base64
import copy
import io
import json
import threading
import unittest
from unittest import mock

from PIL import Image

from backends.webui_backend import WebUIBackend
from core import alwayson_propagation as ap
from core import anima_guidance
from core import sam_extra_notices as sn
from tests.test_alwayson_propagation import A38, DD, DORA, NEGPIP, PAG, SKIM, caps, dd_block
from tests.test_sam_extra_notices import SAM3_OK, info_of

URL = "http://127.0.0.1:7860"
MODEL = "Anima-3.8B-v1.1.safetensors [abcdef12]"   # 봉투(T2I 콤보)와 Forge 현재 체크포인트 — 같으면 Anima38 도 전달(A7)


def png_b64(size=(16, 16)):
    out = io.BytesIO()
    Image.new("RGB", size, "gray").save(out, format="PNG")
    return base64.b64encode(out.getvalue()).decode("ascii")


class _Response:
    def __init__(self, body, status_code=200):
        self._body = body
        self.status_code = status_code
        self.closed = False

    def json(self):
        return self._body

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")

    def close(self):
        self.closed = True


def ok(image=None):
    return _Response({"images": [image or png_b64()], "info": info_of(SAM3_OK)})


def rejected(title):
    return _Response({"error": "HTTPException", "detail": f"always on script {title} not found"}, 422)


def sampling_blocks(**extra):
    blocks = {NEGPIP: {"args": [True]},
              PAG: {"args": anima_guidance.build_args(PAG, {"guid_enabled": True})},
              SKIM: {"args": anima_guidance.build_args(SKIM, {"skim_enabled": True})},
              DD: dd_block(), A38: {"args": [{"negative": True}]}, DORA: {"args": [{"enabled": True}]}}
    blocks.update(extra)
    return blocks


def with_envelope(settings, blocks=None, provenance=None, model=MODEL):
    env = ap.envelope(sampling_blocks() if blocks is None else blocks, source="t2i_panel", backend="webui",
                      provenance=provenance, model=model)
    return {**settings, ap.SETTINGS_KEY: env}


class AuxPropagationTests(unittest.TestCase):
    def setUp(self):
        self.backend = WebUIBackend(URL)
        self.caps = ALL = caps(present=(PAG, SKIM, DD, A38, DORA))
        self.peek = mock.patch("core.sam_extra_probe.peek_capabilities", side_effect=lambda _url: self.caps)
        self.peek.start()
        self.addCleanup(self.peek.stop)
        self.refresh = mock.patch("core.sam_extra_probe.get_capabilities", return_value=ALL)
        self.get_caps = self.refresh.start()
        self.addCleanup(self.refresh.stop)
        self.invalidate = mock.patch("core.sam_extra_probe.invalidate_capabilities")
        self.invalidated = self.invalidate.start()
        self.addCleanup(self.invalidate.stop)
        self.resolve = mock.patch("core.forge_modules.resolve_sam3_checkpoint", side_effect=lambda name: name)
        self.resolve.start()
        self.addCleanup(self.resolve.stop)
        # (A7) Forge 현재 체크포인트 — 실제 GET 을 보내지 않는다(아래 ActiveCheckpointTests 가 GET 자체를 본다)
        self.checkpoint = mock.patch.object(WebUIBackend, "_active_checkpoint", return_value=MODEL)
        self.active_checkpoint = self.checkpoint.start()
        self.addCleanup(self.checkpoint.stop)

    def _run(self, method, settings, responses, **kwargs):
        self.sent = []

        def post(url, json=None, **_kw):
            self.sent.append(copy.deepcopy(json))
            return responses.pop(0)

        with mock.patch("backends.webui_backend.requests.post", side_effect=post):
            return getattr(self.backend, method)(png_b64(), settings, **kwargs)

    def test_t8_forge_aux_requests_carry_their_own_block_then_the_main_blocks(self):
        expected = {
            "refine": ["SAM3 Mask", PAG, SKIM, DD, A38, DORA],
            "sam3": ["SAM3 Mask", PAG, SKIM, DD, A38, DORA],
            "adetailer": ["ADetailer", PAG, SKIM, DD, A38, DORA],
        }
        for method, titles in expected.items():
            with self.subTest(method=method):
                settings = with_envelope({"target": "face", "sam3_prompt": "face"})
                before = copy.deepcopy(settings)
                result = self._run(method, settings, [ok()])
                self.assertEqual(list(self.sent[0]["alwayson_scripts"]), titles)    # NegPiP 없음(A3)
                self.assertEqual(settings, before)
                codes = {n.code for n in sn.notices_of(result)}
                self.assertFalse(codes & {sn.CODE_PROPAGATION_DROPPED, sn.CODE_PROPAGATION_RETRIED})

    def test_t8_without_an_envelope_nothing_changes(self):
        for method, titles in (("refine", ["SAM3 Mask"]), ("sam3", ["SAM3 Mask"]), ("adetailer", ["ADetailer"])):
            with self.subTest(method=method):
                self._run(method, {"target": "face"}, [ok()])
                self.assertEqual(list(self.sent[0]["alwayson_scripts"]), titles)

    def test_default_panel_state_keeps_forge_aux_requests_byte_identical(self):
        """가이던스를 모두 끈 기본 상태면 봉투가 비거나 NegPiP 뿐이다 — Forge 보조 요청은 P7 전과 바이트 단위로 같다."""
        for method in ("refine", "sam3", "adetailer"):
            for blocks in ({}, {NEGPIP: {"args": [True]}}):
                with self.subTest(method=method, blocks=list(blocks)):
                    self._run(method, {"target": "face", "seed": 7}, [ok()])
                    before = json.dumps(self.sent[0], sort_keys=False)
                    self._run(method, with_envelope({"target": "face", "seed": 7}, blocks), [ok()])
                    self.assertEqual(json.dumps(self.sent[0], sort_keys=False), before)

    def test_t10_the_envelope_key_never_reaches_forge(self):
        for method in ("refine", "sam3", "adetailer"):
            with self.subTest(method=method):
                self._run(method, with_envelope({"target": "face"}), [ok()])
                self.assertNotIn(ap.SETTINGS_KEY, json.dumps(self.sent[0], ensure_ascii=False))

    def test_dd_hires_pass_and_sam3_mask_only_are_not_propagated(self):
        self._run("refine", with_envelope({"target": "face"}, sampling_blocks(**{DD: dd_block(hires=True)})), [ok()])
        self.assertNotIn(DD, self.sent[0]["alwayson_scripts"])
        self._run("sam3", with_envelope({"sam3_mode": "Mask only"}), [ok()])
        self.assertEqual(list(self.sent[0]["alwayson_scripts"]), ["SAM3 Mask"])

    def test_the_gate_uses_the_img2img_list_and_explains_user_drops_only(self):
        self.caps = caps(present=(PAG, SKIM, DD, A38, DORA), img2img=(SKIM, DD))
        result = self._run("refine", with_envelope({"target": "face"},
                                                   provenance={A38: ap.PROVENANCE_APP_DEFAULT,
                                                               DORA: ap.PROVENANCE_APP_DEFAULT}), [ok()])
        self.assertEqual(list(self.sent[0]["alwayson_scripts"]), ["SAM3 Mask", SKIM, DD])
        notices = sn.notices_of(result)
        self.assertEqual([(n.code, n.detail) for n in notices], [(sn.CODE_PROPAGATION_DROPPED, f"{PAG}@Refine")])

    def test_unknown_snapshot_sends_user_blocks_but_not_dora(self):
        self.caps = None
        self._run("sam3", with_envelope({}), [ok()])
        self.assertEqual(list(self.sent[0]["alwayson_scripts"]), ["SAM3 Mask", PAG, SKIM, DD, A38])

    def test_result_notices_mark_propagated_features(self):
        result = self._run("refine", with_envelope({"target": "face"}), [ok()])   # PAG 를 보냈는데 infotext 에 없음
        pag = [n for n in sn.notices_of(result) if n.code == sn.CODE_PAG_DROPPED]
        self.assertEqual(len(pag), 1)
        self.assertTrue(pag[0].message.endswith(sn.PROPAGATED_SUFFIX))
        plain = self._run("refine", {"target": "face"}, [ok()])
        self.assertEqual(sn.notices_of(plain), ())

    def test_t9_422_on_a_propagated_title_retries_without_only_that_title(self):
        """(A1·A3) 캐시를 버리지 않고 워커에서 새로 받고, 거절된 제목 하나만 빼고 다시 보낸다."""
        self.caps = None                                                   # 모르니 보냈다
        result = self._run("refine", with_envelope({"target": "face"}), [rejected(DD), ok()])
        self.assertEqual(len(self.sent), 2)
        self.assertEqual(list(self.sent[1]["alwayson_scripts"]), ["SAM3 Mask", PAG, SKIM, A38])
        self.get_caps.assert_called_once_with(URL, refresh=True)
        self.invalidated.assert_not_called()
        retried = [n for n in sn.notices_of(result) if n.code == sn.CODE_PROPAGATION_RETRIED]
        self.assertEqual([(n.level, n.detail) for n in retried], [(sn.LEVEL_WARNING, f"{DD}@Refine")])

    def test_t9_each_rejected_title_is_removed_while_parts_shrink(self):
        self.caps = None
        result = self._run("sam3", with_envelope({}), [rejected(PAG), rejected("anima skimmed cfg"), ok()])
        self.assertEqual([list(p["alwayson_scripts"]) for p in self.sent],
                         [["SAM3 Mask", PAG, SKIM, DD, A38], ["SAM3 Mask", SKIM, DD, A38],
                          ["SAM3 Mask", DD, A38]])
        self.assertEqual(len([n for n in sn.notices_of(result) if n.code == sn.CODE_PROPAGATION_RETRIED]), 2)

    def test_t9_own_block_or_repeated_rejection_fails_like_before(self):
        self.caps = None
        with self.assertLogs("backends.webui_backend", "WARNING"), self.assertRaises(RuntimeError) as ctx:
            self._run("refine", with_envelope({"target": "face"}), [rejected("SAM3 Mask")])
        self.assertIn("sam-extra", str(ctx.exception))
        self.assertEqual(len(self.sent), 1)
        # 이미 뺀 제목을 또 거절하면(부분이 줄지 않음) 다시 보내지 않는다
        with self.assertLogs("backends.webui_backend", "WARNING"), self.assertRaises(RuntimeError):
            self._run("refine", with_envelope({"target": "face"}), [rejected(DD), rejected(DD)])
        self.assertEqual(len(self.sent), 2)

    def test_b10_cancel_is_checked_before_the_retry(self):
        self.caps = None
        cancelled = mock.Mock(return_value=True)
        with self.assertLogs("backends.webui_backend", "WARNING"), self.assertRaises(RuntimeError) as ctx:
            self._run("sam3", with_envelope({}), [rejected(DD), ok()], cancel_check=cancelled)
        self.assertEqual(len(self.sent), 1)
        cancelled.assert_called_once_with()
        self.assertIn(DD, str(ctx.exception))
        self.get_caps.assert_not_called()

    def test_b10_cancel_during_the_capability_refresh_sends_nothing_more(self):
        """(P10 검토 1) 재시도 전 확인을 지난 뒤 스냅샷 새로고침(블로킹 GET) 중에 취소되면 다시 보내지 않고 원래 422
        설명으로 실패한다 — 단독·배치 SAM3 는 Forge 를 끊지 않으므로 다시 보내면 중지 뒤에도 결과가 디스크에 쓰인다."""
        self.caps = None
        stop = threading.Event()

        def refresh(*_args, **_kwargs):
            stop.set()                                   # 사용자가 새로고침 도중 중지를 눌렀다
            return self.caps

        self.get_caps.side_effect = refresh
        with self.assertLogs("backends.webui_backend", "WARNING"), self.assertRaises(RuntimeError) as ctx:
            self._run("sam3", with_envelope({}), [rejected(DD), ok()], cancel_check=stop.is_set)
        self.assertEqual(len(self.sent), 1)
        self.assertIn(DD, str(ctx.exception))
        self.assertIn("HTTP 422", str(ctx.exception))
        self.get_caps.assert_called_once_with(URL, refresh=True)

    def test_a7_anima38_is_forwarded_only_when_forge_runs_the_envelope_model(self):
        """(critic A7) 보조 요청은 체크포인트를 지정하지 않아 Forge 현재 체크포인트로 돈다 — 봉투(T2I 콤보) 모델과 같을 때만
        Anima38 을 보내고, 다르거나 모르면 빼고 정보 로그만 남긴다(다른 블록·알림은 그대로)."""
        for label, actual, envelope_model in (("다른 모델", "Anima-2.9B-preview-v1.safetensors", MODEL),
                                              ("Forge 모름", "", MODEL),
                                              ("봉투 모델 모름", MODEL, "")):
            with self.subTest(label):
                self.active_checkpoint.return_value = actual
                with self.assertLogs("backends.webui_backend", "INFO") as logs:
                    result = self._run("refine", with_envelope({"target": "face"}, model=envelope_model), [ok()])
                self.assertEqual(list(self.sent[0]["alwayson_scripts"]), ["SAM3 Mask", PAG, SKIM, DD, DORA])
                self.assertTrue(any(A38 in line for line in logs.output))
                self.assertFalse({n.code for n in sn.notices_of(result)}
                                 & {sn.CODE_PROPAGATION_DROPPED, sn.CODE_BLOCK_NOT_SENT})
        self.active_checkpoint.return_value = "sub\Anima-3.8B-v1.1.safetensors"   # 폴더·해시 차이는 같은 파일
        self._run("sam3", with_envelope({}), [ok()])
        self.assertIn(A38, self.sent[0]["alwayson_scripts"])

    def test_a7_forge_checkpoint_is_asked_only_for_model_bound_blocks(self):
        self.active_checkpoint.reset_mock()
        without_a38 = {title: block for title, block in sampling_blocks().items() if title != A38}
        self._run("refine", with_envelope({"target": "face"}, without_a38), [ok()])
        self.active_checkpoint.assert_not_called()
        self._run("refine", with_envelope({"target": "face"}), [ok()])
        self.active_checkpoint.assert_called_once_with()

    def test_422_without_propagation_is_unchanged(self):
        with self.assertLogs("backends.webui_backend", "WARNING"), self.assertRaises(RuntimeError):
            self._run("refine", {"target": "face"}, [rejected(PAG)])
        self.assertEqual(len(self.sent), 1)


class ActiveCheckpointTests(unittest.TestCase):
    """(A7) Forge 현재 체크포인트 — GET /sdapi/v1/options 의 sd_model_checkpoint(가짜 requests.get), 실패하면 ''."""

    def test_reads_the_loaded_checkpoint_from_the_options(self):
        backend = WebUIBackend(URL)
        with mock.patch("backends.webui_backend.requests.get",
                        return_value=_Response({"sd_model_checkpoint": MODEL, "CLIP_stop_at_last_layers": 1})) as get:
            self.assertEqual(backend._active_checkpoint(), MODEL)
        self.assertEqual(get.call_args.kwargs["url"], f"{URL}/sdapi/v1/options")
        for label, fake in (("연결 실패", mock.Mock(side_effect=OSError("refused"))),
                            ("HTTP 500", mock.Mock(return_value=_Response({}, 500)))):
            with self.subTest(label), mock.patch("backends.webui_backend.requests.get", fake),                     self.assertLogs("backends.webui_backend", "INFO"):
                self.assertEqual(backend._active_checkpoint(), "")
        for label, body in (("모양이 틀림", ["x"]), ("값 없음", {"sd_model_checkpoint": None})):
            with self.subTest(label), mock.patch("backends.webui_backend.requests.get",
                                                 return_value=_Response(body)):
                self.assertEqual(backend._active_checkpoint(), "")


class BatchWorkerCancelTests(unittest.TestCase):
    """배치 워커는 받을 수 있는 백엔드(WebUI)에만 중지 확인을 넘긴다(core.cancellable_call)."""

    def test_batch_workers_pass_the_stop_event(self):
        import os
        import tempfile
        from workers.adetailer_worker import ADetailerBatchWorker
        from workers.sam3_worker import Sam3BatchWorker

        class _Backend:
            def __init__(self):
                self.checks = []

            def sam3(self, image_b64, settings, *, cancel_check=None):
                self.checks.append(cancel_check)
                return image_b64

            adetailer = sam3

        with tempfile.TemporaryDirectory() as root:
            path = os.path.join(root, "a.png")
            with open(path, "wb") as fh:
                fh.write(base64.b64decode(png_b64()))
            for worker_class in (Sam3BatchWorker, ADetailerBatchWorker):
                with self.subTest(worker=worker_class.__name__):
                    backend = _Backend()
                    worker = worker_class([path], {"output_folder": os.path.join(root, "out")})
                    with mock.patch("backends.get_backend", return_value=backend), \
                            mock.patch("core.model_cache.release_for_generation", return_value=0):
                        worker.run()
                    self.assertEqual(len(backend.checks), 1)
                    self.assertFalse(backend.checks[0]())
                    worker.stop()
                    self.assertTrue(backend.checks[0]())


if __name__ == "__main__":
    unittest.main()
