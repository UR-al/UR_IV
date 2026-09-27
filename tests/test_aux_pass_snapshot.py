"""보조 패스 핸들러가 메인 생성 샘플링 블록 봉투를 싣는지(P7 T14·T16), Refine LoRA 상속(D2), ADetailer 알림 중계."""
import copy
import json
import unittest
from types import SimpleNamespace
from unittest import mock

from core import alwayson_propagation as ap
from core import sam_extra_notices as sn
from tests.test_alwayson_propagation import DD, PAG, SKIM
from tests.test_chat_generation_snapshot import ReadOnlyWidget
from tests.test_sampling_blocks import ChainHost, _webui
from ui import aux_pass_snapshot as aux
from ui.generator_main import GeneratorMainUI


class _Signal:
    def __init__(self):
        self.calls, self.slots = [], []

    def emit(self, *args):
        self.calls.append(args)

    def connect(self, slot):
        self.slots.append(slot)


class _FakeWorker:
    created = []

    def __init__(self, *args):
        self.args = args
        self.finished, self.single_done, self.progress, self.all_done = _Signal(), _Signal(), _Signal(), _Signal()
        _FakeWorker.created.append(self)

    def start(self):
        pass

    def completion_notice(self):
        return "success", "done"


class _Harness(ChainHost):
    _run_adetailer_single = GeneratorMainUI._run_adetailer_single
    _run_adetailer_batch = GeneratorMainUI._run_adetailer_batch
    _run_sam3_single = GeneratorMainUI._run_sam3_single
    _run_sam3_batch = GeneratorMainUI._run_sam3_batch
    _run_refine = GeneratorMainUI._run_refine

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.vue_bridge = SimpleNamespace(**{name: _Signal() for name in (
            "showNotification", "adetailerResult", "adetailerProgress", "sam3Result", "sam3Progress",
            "refineResult")})


HANDLERS = (
    ("_run_adetailer_single", "workers.adetailer_worker.ADetailerSingleWorker", {"path": "a.png"}),
    ("_run_adetailer_batch", "workers.adetailer_worker.ADetailerBatchWorker", {"paths": ["a.png"]}),
    ("_run_sam3_single", "workers.sam3_worker.Sam3SingleWorker", {"path": "a.png"}),
    ("_run_sam3_batch", "workers.sam3_worker.Sam3BatchWorker", {"paths": ["a.png"]}),
    ("_run_refine", "workers.refine_worker.RefineWorker", {"path": "a.png"}),
)


class HandlerTests(unittest.TestCase):
    def setUp(self):
        _webui(self)
        patcher = mock.patch("core.forge_modules.resolve_sam3_checkpoint", side_effect=lambda name: "C:/sam3.pt")
        patcher.start()
        self.addCleanup(patcher.stop)
        _FakeWorker.created.clear()

    def _run(self, host, handler, worker_path, payload):
        with mock.patch(worker_path, _FakeWorker):
            getattr(host, handler)(payload)
        return _FakeWorker.created[-1]

    def test_t14_every_handler_attaches_the_envelope_to_a_copy(self):
        for handler, worker_path, extra in HANDLERS:
            with self.subTest(handler=handler):
                host = _Harness()
                original = {"output_folder": "out", "target": "face"}
                payload = {**extra, "settings": original}
                snapshot = copy.deepcopy(payload)
                worker = self._run(host, handler, worker_path, payload)
                settings = worker.args[1]
                self.assertEqual(payload, snapshot)                          # 원본 dict 는 그대로
                self.assertIsNot(settings, original)
                env = settings[ap.SETTINGS_KEY]
                self.assertTrue(ap.is_envelope(env))
                self.assertEqual(list(env["blocks"]), [ap.TITLE_NEGPIP, PAG, SKIM, DD])
                self.assertEqual(env["source"], aux.SOURCE_T2I_PANEL)
                # 캡처 뒤 위젯을 바꿔도 봉투는 그대로다
                host.anima_guidance_widgets["guid_enabled"] = ReadOnlyWidget("false")
                self.assertIn(PAG, env["blocks"])

    def test_t16_krea2_sends_an_empty_envelope(self):
        host = _Harness()
        host.generation_family_combo = ReadOnlyWidget("KREA2")
        worker = self._run(host, "_run_sam3_single", "workers.sam3_worker.Sam3SingleWorker",
                           {"path": "a.png", "settings": {}})
        env = worker.args[1][ap.SETTINGS_KEY]
        self.assertEqual((env["blocks"], env["backend"]), ({}, ap.BACKEND_KREA2))

    def test_capture_failure_falls_back_to_no_envelope(self):
        host = _Harness()
        with mock.patch.object(aux, "capture_sampling_envelope", side_effect=RuntimeError("boom")), \
                self.assertLogs("ui.aux_pass_snapshot", "WARNING"):
            out = aux.attach_sampling_snapshot(host, {"a": 1})
        self.assertEqual(out, {"a": 1})

    def test_adetailer_results_are_relayed_with_notices(self):
        for handler, worker_path, extra in HANDLERS[:2]:
            with self.subTest(handler=handler):
                host = _Harness()
                worker = self._run(host, handler, worker_path, {**extra, "settings": {}})
                signal = worker.finished if handler.endswith("single") else worker.single_done
                notice = sn.propagation_retried_notice(DD, "ADetailer")
                text = json.dumps({"after": "x.png", "notices": sn.notices_to_dicts([notice])})
                with self.assertLogs("ui.sam_extra_notices_ui", "WARNING"):
                    signal.slots[0](text)
                self.assertEqual(host.vue_bridge.adetailerResult.calls, [(text,)])
                self.assertEqual(host.vue_bridge.showNotification.calls[-1][0], "warning")


class RefineInheritanceTests(unittest.TestCase):
    """(D2) Refine 이 T2I 프롬프트를 물려받을 때 LoRA 스택도 붙인다(Krea2 제외) — Forge·Comfy 가 같은 settings 를 쓴다."""

    def setUp(self):
        _webui(self)

    def test_inherited_prompt_gets_the_lora_stack(self):
        host = _Harness()
        out = aux.inherit_refine_prompts(host, {"target": "face"})
        self.assertEqual(out["main_prompt"], "T2I remains unchanged, <lora:style:0.70>")
        self.assertEqual(out["main_negative"], "__negative__")

    def test_explicit_prompt_and_krea2_are_untouched(self):
        host = _Harness()
        explicit = aux.inherit_refine_prompts(host, {"main_prompt": "mine", "main_negative": "n"})
        self.assertEqual((explicit["main_prompt"], explicit["main_negative"]), ("mine", "n"))
        host.generation_family_combo = ReadOnlyWidget("KREA2")
        self.assertEqual(aux.inherit_refine_prompts(host, {})["main_prompt"], "T2I remains unchanged")

    def test_refine_handler_sends_the_inherited_prompt(self):
        host = _Harness()
        _FakeWorker.created.clear()
        with mock.patch("workers.refine_worker.RefineWorker", _FakeWorker), \
                mock.patch("core.forge_modules.resolve_sam3_checkpoint", side_effect=lambda name: name):
            host._run_refine({"path": "a.png", "settings": {"target": "face"}})
        settings = _FakeWorker.created[-1].args[1]
        self.assertIn("<lora:style:0.70>", settings["main_prompt"])
        self.assertIn(ap.SETTINGS_KEY, settings)


class AdetailerWorkerNoticeTests(unittest.TestCase):
    def test_single_worker_result_carries_notices(self):
        import base64
        import os
        import tempfile
        from tests.test_webui_aux_propagation import png_b64
        from workers.adetailer_worker import ADetailerSingleWorker

        class _Backend:
            def adetailer(self, image_b64, _settings):
                return sn.NoticedImage(image_b64, {}, [sn.propagation_dropped_notice(PAG, "ADetailer")])

        with tempfile.TemporaryDirectory() as root:
            path = os.path.join(root, "a.png")
            with open(path, "wb") as fh:
                fh.write(base64.b64decode(png_b64()))
            results = []
            worker = ADetailerSingleWorker(path, {"output_folder": os.path.join(root, "out")})
            worker.finished.connect(results.append)
            with mock.patch("backends.get_backend", return_value=_Backend()), \
                    mock.patch("core.model_cache.release_for_generation", return_value=0):
                worker.run()
        data = json.loads(results[0])
        self.assertEqual([n["code"] for n in data["notices"]], [sn.CODE_PROPAGATION_DROPPED])


if __name__ == "__main__":
    unittest.main()
