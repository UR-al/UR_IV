"""ComfyUI backend plumbing for VAE DeGrid (L3): compile warnings and node reports reach the user
through ``info['_sam_extra_notices']`` (the channel Forge results use), the card gets ComfyUI's model
list, and standalone passes never carry DeGrid.  No ComfyUI is contacted: ``/object_info`` and the
queue are stubbed.
"""
from __future__ import annotations

import copy
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from backends.base import BackendInfo, GenerationResult
from backends.comfyui_backend import ComfyUIBackend
from core import alwayson_propagation as ap
from core import sam_extra_notices as sn
from core import vae_degrid as vdg
from tests import test_comfy_backend_regressions as regressions
from tests.test_comfy_degrid_compiler import MODEL_FILES, caps_with_degrid, degrid_block
from tests.test_comfy_object_info_cache import _png
from tests.test_comfy_workflow_compiler import _capabilities
from ui import vae_degrid_ui

ROOT = Path(__file__).resolve().parent.parent
NODE = vdg.COMFY_NODE_CLASS


def _payload(**degrid) -> dict:
    return {"prompt": "a", "seed": 5, "save_images": True,
            "alwayson_scripts": {vdg.SCRIPT_NAME: degrid_block(**degrid)}}


def _report(error="") -> dict:
    return {"status": "skipped" if error else "ok", "model": "qwenVAEDegridNafnet_v11", "mode": "Full",
            "strength": 1.0, "tile": 512, "precision": "-" if error else "fp32", "error": error}


class _Harness:
    def backend(self, info=None, *, reports=(), extra_info=None):
        backend = ComfyUIBackend("http://127.0.0.1:1", workflow_path="", img2img_workflow_path="")
        backend._preflight_bundled_node_pack = mock.Mock()
        self.fetches = []

        def fake_fetch(_self=backend):
            data = info if info is not None else caps_with_degrid()
            self.fetches.append(1)
            _self._object_info_cache.store(_self.api_url, data)
            return data

        backend.get_object_info = fake_fetch
        outputs = {"20": {"images": [{"filename": "a.png"}]}}
        if reports:
            outputs["19"] = {vdg.COMFY_UI_KEY: list(reports)}
        result_info = {"prompt_id": "p", "node_outputs": outputs, **(extra_info or {})}
        backend._queue_and_wait = mock.Mock(side_effect=lambda *a, **k: GenerationResult(
            success=True, image_data=b"img", info=copy.deepcopy(result_info)))
        backend._upload_image = mock.Mock(return_value="input_new.png")
        return backend

    @staticmethod
    def queued(backend) -> dict:
        return backend._queue_and_wait.call_args.args[0]

    @staticmethod
    def classes(graph) -> set:
        return {node["class_type"] for node in graph.values()}


class GenerationNoticeTests(_Harness, unittest.TestCase):
    def test_node_report_errors_become_the_forge_style_notice(self):
        backend = self.backend(reports=[_report(), _report("model not found: auto")])
        result = backend.txt2img("checkpoint.safetensors", _payload())
        self.assertTrue(result.success, result.error)
        self.assertIn(NODE, self.classes(self.queued(backend)))
        notices = sn.notices_from_info(result.info)
        self.assertEqual([n.code for n in notices], [sn.CODE_DEGRID_ERROR])
        self.assertIn("1/2장이", notices[0].message)
        self.assertEqual(result.info["seed"], 5)                                    # seed report kept

    def test_success_adds_nothing(self):
        backend = self.backend(reports=[_report()])
        result = backend.txt2img("checkpoint.safetensors", _payload())
        self.assertNotIn(sn.INFO_KEY, result.info)

    def test_img2img_model_missing_is_omitted_and_told_next_to_existing_notices(self):
        earlier = sn.Notice("x", sn.LEVEL_INFO, "earlier").to_dict()
        backend = self.backend(extra_info={sn.INFO_KEY: [earlier]})
        result = backend.img2img("checkpoint.safetensors",
                                 {**_payload(model="gone"), "init_images": [_png()], "denoising_strength": 0.5})
        self.assertTrue(result.success, result.error)
        self.assertNotIn(NODE, self.classes(self.queued(backend)))
        codes = [n.code for n in sn.notices_from_info(result.info)]
        self.assertEqual(codes, ["x", sn.CODE_DEGRID_ERROR])
        self.assertIn("model not found: gone", sn.notices_from_info(result.info)[1].message)

    def test_old_pack_generation_still_runs_and_says_why(self):
        backend = self.backend(_capabilities())
        result = backend.txt2img("checkpoint.safetensors", _payload())
        self.assertTrue(result.success, result.error)
        self.assertNotIn(NODE, self.classes(self.queued(backend)))
        notices = sn.notices_from_info(result.info)
        self.assertEqual([n.code for n in notices], [sn.CODE_DEGRID_COMFY_UNAVAILABLE])
        self.assertIn(vdg.COMFY_MIN_PACK_VERSION, notices[0].message)

    def test_failed_generation_gets_no_degrid_notice(self):
        backend = self.backend(_capabilities())
        backend._queue_and_wait = mock.Mock(return_value=GenerationResult(success=False, error="boom"))
        result = backend.txt2img("checkpoint.safetensors", _payload())
        self.assertEqual((result.success, result.error, result.info), (False, "boom", {}))

    def test_a_stale_snapshot_without_the_node_is_recompiled_once_fresh(self):
        backend = self.backend()
        backend._object_info_cache.store(backend.api_url, _capabilities())         # predates pack 1.5.0
        result = backend.txt2img("checkpoint.safetensors", _payload())
        self.assertEqual(len(self.fetches), 1)
        self.assertIn(NODE, self.classes(self.queued(backend)))
        self.assertNotIn(sn.INFO_KEY, result.info)                                  # last compile's warnings only

    def test_a_fresh_snapshot_is_not_retried(self):
        backend = self.backend(caps_with_degrid())
        result = backend.txt2img("checkpoint.safetensors", _payload(model="gone"))
        self.assertEqual(len(self.fetches), 1)
        self.assertEqual([n.code for n in sn.notices_from_info(result.info)], [sn.CODE_DEGRID_ERROR])
        # the cached snapshot is the very document that already said "gone" is missing — no second download
        result = backend.txt2img("checkpoint.safetensors", _payload(model="gone"))
        self.assertEqual(len(self.fetches), 1)
        self.assertEqual(len(sn.notices_from_info(result.info)), 1)
        # a different model is a different cause: one refetch for it, then none
        backend.txt2img("checkpoint.safetensors", _payload(model="other"))
        backend.txt2img("checkpoint.safetensors", _payload(model="other"))
        self.assertEqual(len(self.fetches), 2)

    def test_an_old_pack_costs_one_refetch_not_one_per_generation(self):
        backend = self.backend(_capabilities())                                     # live ComfyUI still at 1.4.2
        backend._object_info_cache.store(backend.api_url, _capabilities())         # snapshot from connect
        for _ in range(4):
            result = backend.txt2img("checkpoint.safetensors", _payload())
            self.assertTrue(result.success, result.error)
            self.assertEqual([n.code for n in sn.notices_from_info(result.info)], [sn.CODE_DEGRID_COMFY_UNAVAILABLE])
        self.assertEqual(len(self.fetches), 1)
        # installing the bundled pack (or a managed restart) can fix it — check once more
        backend._forget_schema_causes()
        backend.txt2img("checkpoint.safetensors", _payload())
        self.assertEqual(len(self.fetches), 2)


    def test_without_degrid_no_refetch_and_no_notice(self):
        backend = self.backend()
        backend.txt2img("checkpoint.safetensors", {"prompt": "a"})
        result = backend.txt2img("checkpoint.safetensors", {"prompt": "a"})
        self.assertEqual(len(self.fetches), 1)
        self.assertNotIn(sn.INFO_KEY, result.info)


class SchemaRefetchPolicyTests(unittest.TestCase):
    """``_compile_graph``: a cached compile is redone on a live schema only for DeGrid warnings a schema can change,
    and at most once per distinct cause (per URL; forgotten when the pack is installed or the runtime restarts)."""

    def setUp(self):
        self.backend = ComfyUIBackend("http://127.0.0.1:1", workflow_path="")
        self.compilers = []
        self.cached = True

        def compiler(*, fresh=False):
            self.compilers.append(fresh)
            return SimpleNamespace(object_info_cached=not fresh and self.cached)

        self.backend._workflow_compiler = compiler

    def compile(self, *warning_sets):
        """Each compile in this call (first, then a retry) appends the next set of warnings."""
        sets = iter(warning_sets)
        warnings: list = []

        def build(_compiler):
            warnings.clear()
            warnings.extend(next(sets, ()))
            return {"compiles": len(self.compilers)}

        self.compilers.clear()
        self.backend._compile_graph(build, warnings)
        return list(self.compilers), warnings

    @staticmethod
    def warning(cause, reason="r"):
        code = sn.CODE_DEGRID_ERROR if cause == vdg.COMFY_OMIT_MODEL_MISSING else sn.CODE_DEGRID_COMFY_UNAVAILABLE
        return {"code": code, "reason": reason, "feature": "degrid", "cause": cause}

    def test_custom_workflow_placement_never_refetches(self):
        placement = [self.warning(vdg.COMFY_OMIT_PLACEMENT, "no output node")]
        for _ in range(3):
            self.assertEqual(self.compile(placement)[0], [False])

    def test_each_fixable_cause_refetches_once(self):
        for cause in (vdg.COMFY_OMIT_NODE_MISSING, vdg.COMFY_OMIT_NODE_CONTRACT, vdg.COMFY_OMIT_MODEL_MISSING):
            with self.subTest(cause=cause):
                self.backend._forget_schema_causes()
                still = [self.warning(cause)]
                calls, warnings = self.compile(still, still)
                self.assertEqual(calls, [False, True])                           # cached, then one live schema
                self.assertEqual(warnings, still)                                # the last compile's warnings
                self.assertEqual(self.compile(still)[0], [False])                # same cause: no second download

    def test_a_fixed_cause_keeps_the_fresh_graph_and_mixed_warnings_still_refetch(self):
        fixable = self.warning(vdg.COMFY_OMIT_NODE_MISSING)
        placement = self.warning(vdg.COMFY_OMIT_PLACEMENT)
        calls, warnings = self.compile([fixable, placement], [placement])
        self.assertEqual((calls, warnings), ([False, True], [placement]))

    def test_distinct_reasons_are_distinct_causes(self):
        model_a = [self.warning(vdg.COMFY_OMIT_MODEL_MISSING, "model not found: a")]
        model_b = [self.warning(vdg.COMFY_OMIT_MODEL_MISSING, "model not found: b")]
        self.assertEqual(self.compile(model_a, model_a)[0], [False, True])
        self.assertEqual(self.compile(model_b, model_b)[0], [False, True])
        self.assertEqual(self.compile(model_a + model_b)[0], [False])

    def test_a_cause_first_seen_on_a_live_schema_is_not_refetched(self):
        self.cached = False                                                       # cache miss: live document
        missing = [self.warning(vdg.COMFY_OMIT_NODE_MISSING)]
        self.assertEqual(self.compile(missing)[0], [False])
        self.cached = True
        self.assertEqual(self.compile(missing)[0], [False])

    def test_a_cause_first_seen_on_the_refetched_schema_is_remembered(self):
        # cached compile: model A missing; the live schema fixes A but reveals model B missing — B must not cost
        # another download on the next generation (it was already checked on a live schema)
        model_a = [self.warning(vdg.COMFY_OMIT_MODEL_MISSING, "model not found: a")]
        model_b = [self.warning(vdg.COMFY_OMIT_MODEL_MISSING, "model not found: b")]
        self.assertEqual(self.compile(model_a, model_b)[0], [False, True])
        self.assertEqual(self.compile(model_b)[0], [False])

    def test_memory_is_per_url_and_forgotten_on_pack_install(self):
        missing = [self.warning(vdg.COMFY_OMIT_NODE_MISSING)]
        self.compile(missing, missing)
        self.backend.api_url = "http://127.0.0.1:2"                               # managed restart on a new port
        self.assertEqual(self.compile(missing, missing)[0], [False, True])
        self.backend._forget_schema_causes()
        self.assertEqual(self.compile(missing, missing)[0], [False, True])

    def test_pack_install_paths_forget_the_causes(self):
        source = (ROOT / "backends" / "comfyui_backend.py").read_text(encoding="utf-8")
        preflight = source[source.index("def _preflight_bundled_node_pack"):source.index("def _workflow_compiler")]
        invalidations = preflight.count("self._object_info_cache.invalidate()")
        self.assertGreater(invalidations, 0)
        self.assertEqual(invalidations, preflight.count("self._forget_schema_causes()"))

    def test_without_a_warnings_list_nothing_is_retried(self):
        self.compilers.clear()
        self.backend._compile_graph(lambda _compiler: {})
        self.assertEqual(self.compilers, [False])


class StandaloneStripTests(unittest.TestCase):
    _backend = regressions.TestStandaloneComfyContext._backend

    def _compiled_payload(self, backend, settings, kind):
        captured = {}

        def compile_graph(build, *_args):
            compiler = mock.Mock()

            def compile_postprocess(model_name, payload, **kwargs):
                captured["payload"] = copy.deepcopy(payload)
                return {}

            compiler.compile_postprocess.side_effect = compile_postprocess
            return build(compiler)

        with mock.patch.object(backend, "_compile_graph", side_effect=compile_graph):
            getattr(backend, kind)(_png(), settings)
        return captured["payload"]

    def test_saved_context_degrid_never_reaches_a_standalone_pass(self):
        for title in (vdg.SCRIPT_NAME, vdg.SCRIPT_NAME.lower(), f"  {vdg.SCRIPT_NAME.upper()} "):
            # 봉투는 저장 문맥의 샘플링 블록을 교체한다 — NegPiP 은 봉투 쪽 값으로 다시 들어온다
            for envelope in (None, ap.envelope({vdg.SCRIPT_NAME: degrid_block(), "NegPiP": {"args": [True]}},
                                               source="t2i_panel", backend=ap.BACKEND_COMFY)):
                for kind in ("adetailer", "sam3", "refine"):
                    with self.subTest(title=title, envelope=envelope is not None, kind=kind):
                        backend = self._backend({title: degrid_block(), "NegPiP": {"args": [True]}})
                        settings = {"sam3_prompt": "face"}
                        if envelope is not None:
                            settings[ap.SETTINGS_KEY] = envelope
                        scripts = self._compiled_payload(backend, settings, kind)["alwayson_scripts"]
                        self.assertFalse([name for name in scripts
                                          if ap.canonical_title(name) == vdg.SCRIPT_NAME], scripts)
                        self.assertIn("NegPiP", scripts)

    def test_real_standalone_graph_has_no_degrid_node(self):
        backend = self._backend({vdg.SCRIPT_NAME: degrid_block()})
        backend._workflow_compiler = mock.Mock(return_value=__import__(
            "core.comfy_workflow_compiler", fromlist=["ComfyWorkflowCompiler"]).ComfyWorkflowCompiler(None))
        backend.sam3(_png(), {"sam3_prompt": "face", "sam3_mode": "Inpaint"})
        graph = backend._queue_and_wait.call_args.args[0]
        self.assertNotIn(NODE, {node["class_type"] for node in graph.values()})


class ModelListPlumbingTests(unittest.TestCase):
    def test_get_info_carries_the_comfy_model_names(self):
        backend = ComfyUIBackend("http://127.0.0.1:1", workflow_path="")
        with mock.patch.object(backend, "get_object_info", return_value=caps_with_degrid(*MODEL_FILES)):
            self.assertEqual(backend.get_info().degrid_models, ["qwenVAEDegridNafnet_v11", "NAFNet-QwenVAE-DeGrid"])
        with mock.patch.object(backend, "get_object_info", return_value=_capabilities()):
            self.assertIsNone(backend.get_info().degrid_models)                    # old pack — card says update
        self.assertIsNone(BackendInfo().degrid_models)                              # Forge uses the snapshot

    def test_info_worker_forwards_the_list(self):
        from workers.generation_worker import WebUIInfoWorker
        worker = WebUIInfoWorker()
        seen = []
        worker.info_ready.connect(seen.append)
        backend = mock.Mock()
        backend.get_info.return_value = BackendInfo(models=["m"], degrid_models=["a"])
        with mock.patch("workers.generation_worker.get_backend", return_value=backend):
            worker.run()
        self.assertEqual(seen[0]["degrid_models"], ["a"])

    def test_connection_slots_push_and_clear_the_card_list(self):
        source = (ROOT / "ui" / "generator_webui.py").read_text(encoding="utf-8")
        loaded = source[source.index("def on_webui_info_loaded"):source.index("def on_webui_info_error")]
        self.assertIn("push_comfy_models(self, info.get('degrid_models'))", loaded)
        failed = source[source.index("def on_webui_info_error"):]
        self.assertIn("push_comfy_models(self, None)", failed[:failed.index("\n    def ", 10)])

    def test_push_sets_the_card_property(self):
        pushed = []
        host = SimpleNamespace(vue_bridge=SimpleNamespace(pushWidgetProperty=lambda *a: pushed.append(a)))
        vae_degrid_ui.push_comfy_models(host, ["a", "None", ""])
        self.assertEqual(pushed, [(vdg.widget_id("model"), vae_degrid_ui.COMFY_MODELS_PROPERTY, ["a"])])


class CardRefreshTests(unittest.TestCase):
    """The card's refresh button on ComfyUI: the same ``sam_extra_capabilities_get {refresh: true}`` action reloads
    ``/object_info`` in a worker and pushes the model list through the existing ``comfyModels`` property."""

    def setUp(self):
        self.pushed = []
        self.host = SimpleNamespace(vue_bridge=SimpleNamespace(pushWidgetProperty=lambda *a: self.pushed.append(a)))

    def values(self):
        return [value for _wid, _prop, value in self.pushed]

    def refresh(self, backend):
        with mock.patch("backends.get_backend", return_value=backend):
            thread = vae_degrid_ui.refresh_comfy_models(self.host)
            thread.join(5)

    def test_refresh_pushes_the_fresh_list_and_renews_the_compile_snapshot(self):
        backend = ComfyUIBackend("http://127.0.0.1:1", workflow_path="")
        live = caps_with_degrid(*MODEL_FILES)
        response = SimpleNamespace(json=lambda: live, content=b"{}")
        with mock.patch("core.http_retry.get_with_retry", return_value=response) as get:
            self.refresh(backend)
        self.assertTrue(get.call_args.args[0].endswith("/object_info"))
        self.assertEqual(self.values(), [["qwenVAEDegridNafnet_v11", "NAFNet-QwenVAE-DeGrid"]])
        self.assertIs(backend._object_info_cache.peek(backend.api_url), live)     # the next compile uses it

    def test_old_pack_pushes_none_and_errors_keep_the_last_list(self):
        backend = mock.Mock(get_object_info=mock.Mock(return_value=_capabilities()))
        self.refresh(backend)
        self.assertEqual(self.values(), [None])                                   # card: pack update needed
        backend.get_object_info.side_effect = OSError("down")
        self.refresh(backend)
        self.assertEqual(self.values(), [None])                                   # nothing new pushed

    def test_a_connect_or_backend_switch_meanwhile_drops_the_result(self):
        import threading
        release = threading.Event()

        def slow():
            release.wait(5)
            return caps_with_degrid(*MODEL_FILES)

        backend = mock.Mock(get_object_info=slow)
        with mock.patch("backends.get_backend", return_value=backend):
            thread = vae_degrid_ui.refresh_comfy_models(self.host)
            vae_degrid_ui.push_comfy_models(self.host, None)                      # connection failed meanwhile
            release.set()
            thread.join(5)
        self.assertEqual(self.values(), [None])
        release.clear()
        other = mock.Mock()
        with mock.patch("backends.get_backend", side_effect=[backend, other]):   # switched before the answer
            thread = vae_degrid_ui.refresh_comfy_models(self.host)
            release.set()
            thread.join(5)
        self.assertEqual(self.values(), [None])

    def _actions_host(self):
        from ui.sam_extra_capabilities_actions import SamExtraCapabilitiesActionsMixin

        class Host(SamExtraCapabilitiesActionsMixin):
            pass

        host = Host()
        host.vue_bridge = SimpleNamespace(
            samExtraCapabilities=SimpleNamespace(emit=lambda _text: None),
            pushWidgetProperty=lambda *a: self.pushed.append(a))
        return host

    def test_manual_refresh_on_comfyui_reloads_the_degrid_list_with_the_same_action(self):
        from backends import BackendType
        from ui import sam_extra_capabilities_actions as actions
        backend = mock.Mock(get_object_info=mock.Mock(return_value=caps_with_degrid(*MODEL_FILES)))
        host = self._actions_host()
        host._sam_extra_fetch = lambda *_a: self.fail("ComfyUI 에서 Forge 스냅샷을 묻지 않는다")
        clock = iter([1000.0, 1001.0, 1100.0])
        with mock.patch("backends.get_backend_type", return_value=BackendType.COMFYUI), \
                mock.patch("backends.get_backend", return_value=backend), \
                mock.patch.object(actions, "_clock", side_effect=lambda: next(clock)):
            for payload in ({"refresh": True}, {"refresh": True}, {}, {"refresh": True}):
                self.assertTrue(host._handle_sam_extra_capabilities_action("sam_extra_capabilities_get", payload))
                worker = getattr(host, "_degrid_models_worker", None)
                if worker is not None:
                    worker.join(5)
            host._refresh_sam_extra_capabilities(force=True)                      # connect hook: not manual
        # the first refresh runs, the 30 s limit drops the second, a replay never fetches, the fourth runs
        self.assertEqual(backend.get_object_info.call_count, 2)
        self.assertEqual(self.values(), [["qwenVAEDegridNafnet_v11", "NAFNet-QwenVAE-DeGrid"]] * 2)
        self.assertEqual(host.sam_extra_capabilities.status, "not_applicable")

    def test_forge_manual_refresh_does_not_touch_the_comfy_list(self):
        from backends import BackendType
        from core.sam_extra_capabilities import unknown_capabilities
        host = self._actions_host()
        host._sam_extra_fetch = lambda *_a: unknown_capabilities("unknown")
        with mock.patch("backends.get_backend_type", return_value=BackendType.WEBUI), \
                mock.patch("backends.get_backend", return_value=mock.Mock(api_url="http://127.0.0.1:7861")), \
                mock.patch("core.sam_extra_probe.invalidate_extensions"), \
                mock.patch("ui.vae_degrid_ui.refresh_comfy_models") as refresh:
            host._refresh_sam_extra_capabilities(force=True, manual=True)
            host._sam_extra_worker.join(5)
        refresh.assert_not_called()


if __name__ == "__main__":
    unittest.main()
