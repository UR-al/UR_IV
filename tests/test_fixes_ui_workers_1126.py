"""Worker B (ui/, workers/, web_main_ui.py, core/web_action_policy.py) 11-26 수정 회귀 테스트."""
from __future__ import annotations

import base64
import json
import os
import sys
import tempfile
import threading
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from PIL import Image

try:
    from PyQt6.QtCore import QCoreApplication, QEventLoop, QThread, QTimer
    _QT_AVAILABLE = True
    _QT_ERROR = ""
except ModuleNotFoundError as exc:  # pragma: no cover
    if not (exc.name or "").startswith("PyQt6"):
        raise
    _QT_AVAILABLE = False
    _QT_ERROR = str(exc)
    QCoreApplication = QEventLoop = QThread = QTimer = None


def _ensure_qt_app():
    if not _QT_AVAILABLE:
        return None
    return QCoreApplication.instance() or QCoreApplication([])


def _png_bytes(color=(30, 60, 90), size=(8, 8)) -> bytes:
    from io import BytesIO
    buf = BytesIO()
    Image.new("RGB", size, color).save(buf, format="PNG")
    return buf.getvalue()


def _png_b64(color=(30, 60, 90)) -> str:
    return base64.b64encode(_png_bytes(color)).decode("ascii")


class _StubSignal:
    def __init__(self):
        self.emitted = []

    def emit(self, *args):
        self.emitted.append(args)


class _StubBridge:
    def __init__(self):
        self.registered = {}
        self.pushed_values = []
        self.pushed_props = []
        self.showNotification = _StubSignal()

    def _register_proxy(self, widget_id, proxy):
        self.registered[widget_id] = proxy

    def pushWidgetValue(self, widget_id, value):
        self.pushed_values.append((widget_id, value))

    def pushWidgetProperty(self, widget_id, prop, value):
        self.pushed_props.append((widget_id, prop, value))


# ── B-01: 웹 모드 QMessageBox 가드 ────────────────────────────────────────────

class B01WebModeNotificationGuardTests(unittest.TestCase):
    def test_notify_or_message_box_routes_to_toast_in_web_mode(self):
        from ui.status_line import notify_or_message_box

        box = mock.Mock()
        bridge = _StubBridge()
        web_host = SimpleNamespace(web_mode=True, vue_bridge=bridge)
        shown = notify_or_message_box(web_host, "warning", "알림", "먼저 검색하세요.", box=box)
        self.assertFalse(shown)
        box.warning.assert_not_called()
        box.information.assert_not_called()
        box.critical.assert_not_called()
        self.assertEqual(bridge.showNotification.emitted, [("warning", "먼저 검색하세요.")])

        desktop_host = SimpleNamespace(web_mode=False, vue_bridge=bridge)
        shown_desktop = notify_or_message_box(desktop_host, "warning", "알림", "먼저 검색하세요.", box=box)
        self.assertTrue(shown_desktop)
        box.warning.assert_called_once_with(desktop_host, "알림", "먼저 검색하세요.")

    def test_generator_actions_prompts_settings_webui_skip_modal_in_web_mode(self):
        from ui.generator_actions import ActionsMixin
        from ui.generator_prompts import PromptHandlingMixin
        from ui.generator_settings import SettingsMixin

        bridge = _StubBridge()
        host = SimpleNamespace(
            web_mode=True,
            vue_bridge=bridge,
            filtered_results=[],
            shuffled_prompt_deck=[],
            is_automating=False,
            show_status=lambda *_a, **_kw: None,
            queue_panel=SimpleNamespace(add_single_item=lambda _p: None),
        )
        with mock.patch("ui.generator_actions.QMessageBox") as act_box, \
                mock.patch("ui.generator_prompts.QMessageBox") as prompt_box, \
                mock.patch("ui.generator_settings.QMessageBox") as set_box:
            ActionsMixin._start_automation(host)
            ActionsMixin.receive_event_scenarios(host, [{"payload": {"prompt": "1girl"}}])
            PromptHandlingMixin.apply_random_prompt(host)
            host._build_settings_dict = mock.Mock(side_effect=RuntimeError("disk full"))
            ok = SettingsMixin.save_settings(host)

            self.assertFalse(ok)
            act_box.warning.assert_not_called()
            act_box.information.assert_not_called()
            prompt_box.warning.assert_not_called()
            set_box.critical.assert_not_called()

        self.assertGreaterEqual(len(bridge.showNotification.emitted), 4)


# ── B-03, B-07, M-04: WidgetProxy 수정 ────────────────────────────────────────

@unittest.skipUnless(_QT_AVAILABLE, f"PyQt6 unavailable: {_QT_ERROR}")
class WidgetProxyFixesTests(unittest.TestCase):
    def setUp(self):
        _ensure_qt_app()

    def test_b03_combobox_set_enabled_pushes_enabled_property(self):
        from ui.widget_proxies import ComboBoxProxy

        bridge = _StubBridge()
        combo = ComboBoxProxy(bridge, "model_combo")
        combo.setEnabled(False)
        self.assertFalse(combo.isEnabled())
        self.assertIn(("model_combo", "enabled", False), bridge.pushed_props)
        combo.setEnabled(True)
        self.assertIn(("model_combo", "enabled", True), bridge.pushed_props)

    def test_b07_combobox_on_vue_changed_prefers_exact_text_over_integer_index(self):
        from ui.widget_proxies import ComboBoxProxy

        bridge = _StubBridge()
        combo = ComboBoxProxy(bridge, "numeric_combo")
        combo.addItems(["2", "0", "1"])
        combo._on_vue_changed("0")
        self.assertEqual(combo.currentText(), "0")
        self.assertEqual(combo.currentIndex(), 1)
        # 항목에 없는 숫자 문자열은 기존대로 인덱스 폴백
        combo2 = ComboBoxProxy(bridge, "label_combo")
        combo2.addItems(["Alpha", "Beta", "Gamma"])
        combo2._on_vue_changed("2")
        self.assertEqual(combo2.currentText(), "Gamma")
        self.assertEqual(combo2.currentIndex(), 2)

    def test_m04_button_proxy_push_to_bridge_false_omits_bridge_traffic(self):
        from ui.widget_proxies import ButtonProxy

        bridge = _StubBridge()
        btn = ButtonProxy(bridge, "btn_generate", push_to_bridge=False)
        btn.setCheckable(True)
        toggled = []
        btn.toggled.connect(toggled.append)
        btn.setText("생성 중")
        btn.setEnabled(False)
        btn.setChecked(True)

        self.assertNotIn("btn_generate", bridge.registered)
        self.assertEqual(bridge.pushed_props, [])
        self.assertEqual(btn.text(), "생성 중")
        self.assertFalse(btn.isEnabled())
        self.assertTrue(btn.isChecked())
        self.assertEqual(toggled, [True])


# ── B-05: 웹 모드 native_tab_switch 및 로컬 전용 액션 정책 ─────────────────────

class B05WebActionPolicyTests(unittest.TestCase):
    def test_native_tab_switch_and_local_only_actions_blocked_in_web_mode(self):
        from core.web_action_policy import (
            LOCAL_APP_ONLY_MESSAGE,
            WEB_BLOCKED_ACTIONS,
            web_action_denial,
            web_blocked_message,
        )

        expected = (
            "native_tab_switch",
            "tile_repair_options", "tile_repair_run", "tile_repair_cancel",
            "relight_preview", "relight_export", "relight_cancel",
            "hand_reconstruction_generate", "hand_reconstruction_export", "hand_reconstruction_cancel",
        )
        for action in expected:
            self.assertIn(action, WEB_BLOCKED_ACTIONS)
            self.assertEqual(web_blocked_message(action), WEB_BLOCKED_ACTIONS[action])
            self.assertTrue(web_action_denial(action, web_mode=True, remote=False))
            self.assertTrue(web_action_denial(action, web_mode=True, remote=True))
            self.assertIsNone(web_action_denial(action, web_mode=False, remote=True))
        # 표에 없는 액션은 LOCAL_APP_ONLY_MESSAGE 폴백
        self.assertEqual(web_blocked_message("unknown_local_only_action"), LOCAL_APP_ONLY_MESSAGE)

    def test_native_tab_switch_branch_and_local_mixins_notify_in_web_mode(self):
        from core.web_action_policy import web_blocked_message
        from ui.generator_main import GeneratorMainUI
        from ui.tile_repair_actions import TileRepairActionsMixin
        from ui.relight_actions import RelightActionsMixin
        from ui.hand_reconstruction_actions import HandReconstructionActionsMixin

        bridge = _StubBridge()
        backend_tab = mock.Mock()
        host = SimpleNamespace(
            web_mode=True,
            vue_bridge=bridge,
            backend_ui_tab=backend_tab,
            _handle_model_download_action=lambda *_a: False,
            _handle_chat_action=lambda *_a: False,
            _handle_creator_action=lambda *_a: False,
        )
        GeneratorMainUI._handle_vue_action(host, "native_tab_switch", {"tab": "backend"})
        backend_tab.ensure_loaded.assert_not_called()
        self.assertEqual(bridge.showNotification.emitted, [("warning", web_blocked_message("native_tab_switch"))])

        # Tile repair, relight, hand reconstruction direct mixin calls emit policy message in web mode
        bridge2 = _StubBridge()
        bridge2.tileRepairResult = _StubSignal()
        bridge2.relightEvent = _StubSignal()
        bridge2.handReconstructionEvent = _StubSignal()
        tile_host = TileRepairActionsMixin()
        tile_host.web_mode = True
        tile_host.vue_bridge = bridge2
        relight_host = RelightActionsMixin()
        relight_host.web_mode = True
        relight_host.vue_bridge = bridge2
        hand_host = HandReconstructionActionsMixin()
        hand_host.web_mode = True
        hand_host.vue_bridge = bridge2
        tile_host._tile_repair_dispatch("tile_repair_run", {"requestId": "r1"})
        relight_host._run_relight_action("relight_preview", {"requestId": "r2"})
        hand_host._handle_hand_reconstruction_action("hand_reconstruction_generate", {"requestId": "r3"})
        self.assertEqual(bridge2.showNotification.emitted, [
            ("warning", web_blocked_message("tile_repair_run")),
            ("warning", web_blocked_message("relight_preview")),
            ("warning", web_blocked_message("hand_reconstruction_generate")),
        ])


# ── B-06: web_main_ui IPv6, backendStatus 리플레이, 종료 정리 ────────────────

@unittest.skipUnless(_QT_AVAILABLE, f"PyQt6 unavailable: {_QT_ERROR}")
class B06WebMainFixesTests(unittest.TestCase):
    def test_ipv6_url_host_formatting(self):
        import web_main_ui

        self.assertEqual(web_main_ui._format_url_host("::1"), "[::1]")
        self.assertEqual(web_main_ui._format_url_host("[::1]"), "[::1]")
        self.assertEqual(web_main_ui._format_url_host("fe80::1"), "[fe80::1]")
        self.assertEqual(web_main_ui._format_url_host("127.0.0.1"), "127.0.0.1")
        self.assertEqual(web_main_ui._format_url_host("localhost"), "localhost")

    def test_web_bridge_facade_caches_and_replays_backend_status(self):
        import web_main_ui
        from ui.vue_bridge import VueBridge

        app = _ensure_qt_app()
        bridge = VueBridge()
        facade = web_main_ui.WebBridgeFacade(bridge)
        received = []
        facade.event.connect(lambda name, args: received.append((name, json.loads(args))))

        status_json = json.dumps({"connected": True, "type": "webui", "url": "http://127.0.0.1:7860"})
        bridge.backendStatus.emit(status_json)
        self.assertEqual(received, [("backendStatus", [status_json])])

        # 늦은 클라이언트 재전송 시뮬레이션 — invoke("getInitialConfig") 시 최신 backendStatus 재발행
        received.clear()
        reply = json.loads(facade.invoke("getInitialConfig", "[]"))
        self.assertTrue(reply["ok"])
        app.processEvents()
        self.assertEqual(received, [("backendStatus", [status_json])])

    def test_flush_window_state_on_quit_saves_settings_runs_hooks_and_stops_workers(self):
        import web_main_ui
        from ui.generator_main import GeneratorMainUI

        events = []

        class _RunningWorker:
            def __init__(self, tag):
                self._tag = tag

            def isRunning(self):
                return True

            def cancel(self):
                events.append(f"{self._tag}_cancel")

            def stop(self):
                events.append(f"{self._tag}_stop")

            def quit(self):
                events.append(f"{self._tag}_quit")

            def wait(self, ms):
                events.append((f"{self._tag}_wait", ms))
                return True

        fake_window = SimpleNamespace(
            _flush_deck_state=lambda: events.append("deck"),
            _flush_queue_state=lambda: events.append("queue"),
            _save_shutdown_state=lambda: events.append("save_shutdown_state"),
            _stop_owned_backend_runtimes=lambda: events.append("stop_owned"),
            _shutdown_relight=lambda: events.append("shutdown_relight"),
            _ad_worker=_RunningWorker("ad"),
            vue_bridge=SimpleNamespace(_stale_search_workers=[_RunningWorker("stale")]),
            _history_trash_undo=SimpleNamespace(close=lambda: events.append("undo_close")),
        )
        # 실제 GeneratorMainUI._shutdown_hooks_and_workers 경로를 바인딩해 검증
        fake_window._shutdown_hooks_and_workers = (
            lambda: GeneratorMainUI._shutdown_hooks_and_workers(fake_window)
        )
        web_main_ui._flush_window_state_on_quit(fake_window)
        self.assertIn("deck", events)
        self.assertIn("queue", events)
        self.assertIn("save_shutdown_state", events)
        self.assertIn("shutdown_relight", events)
        self.assertIn("ad_cancel", events)
        self.assertIn("ad_stop", events)
        self.assertIn("ad_quit", events)
        self.assertIn(("ad_wait", 500), events)
        self.assertIn("stale_stop", events)
        self.assertIn(("stale_wait", 300), events)
        self.assertIn("stop_owned", events)
        self.assertIn("undo_close", events)


# ── B-08: GenerationFlowWorker 리스 해제 시점 + terminate 후 강제 해제 ──────────

@unittest.skipUnless(_QT_AVAILABLE, f"PyQt6 unavailable: {_QT_ERROR}")
class B08GenerationLeaseReleaseTests(unittest.TestCase):
    def setUp(self):
        _ensure_qt_app()

    def test_emit_result_fires_after_gpu_lease_is_released(self):
        from core.resource_coordinator import GenerationResourceCoordinator
        from workers.generation_worker import GenerationFlowWorker

        coord = GenerationResourceCoordinator()
        lease_states_in_callback = []

        class _Backend:
            def txt2img(self, _model, _payload, **_kw):
                return SimpleNamespace(success=False, error="mock fail", image_data=None, info={})

        worker = GenerationFlowWorker("m", {"prompt": "hi"}, backend=_Backend())
        worker.finished.connect(lambda _r, _i: lease_states_in_callback.append(coord.is_busy()))

        with mock.patch("workers.generation_worker.get_generation_coordinator", return_value=coord), \
                mock.patch("workers.generation_worker.get_backend", return_value=_Backend()), \
                mock.patch("workers.generation_worker.wait_for_pending_unload", return_value=True):
            worker.run()

        self.assertEqual(lease_states_in_callback, [False])
        self.assertFalse(coord.is_busy())

    def test_cleanup_gen_worker_force_releases_lease_after_terminate(self):
        from core.resource_coordinator import GenerationResourceCoordinator
        from ui.generator_generation import GenerationMixin
        from workers.generation_worker import GenerationFlowWorker

        coord = GenerationResourceCoordinator()
        worker = GenerationFlowWorker("m", {"prompt": "hi"})
        # 워커가 리스를 쥔 채 강제 종료(terminate) 당한 상황을 모사
        cm = coord.reserve("txt2img", unload_llm=False, timeout=0)
        cm.__enter__()
        worker._active_lease_cm = cm
        worker._active_lease_coord = coord
        self.assertTrue(coord.is_busy())

        host = SimpleNamespace(gen_worker=worker)
        with mock.patch.object(worker, "isRunning", return_value=True), \
                mock.patch.object(worker, "cancel"), \
                mock.patch.object(worker, "quit"), \
                mock.patch.object(worker, "wait", side_effect=[False, True]), \
                mock.patch.object(worker, "terminate") as term:
            GenerationMixin._cleanup_gen_worker(host)

        term.assert_called_once()
        self.assertFalse(coord.is_busy(), "terminate 뒤 GPU 리스가 반드시 해제되어야 합니다")
        # 후속 생성이 즉시 리스를 잡을 수 있어야 함
        with coord.reserve("next-gen", unload_llm=False, timeout=0):
            self.assertTrue(coord.is_busy())
        self.assertFalse(coord.is_busy())


# ── B-09, B-14, B-20(4): ADetailer / SAM3 / Refine 워커 및 Creator 경로 안전성 ──

@unittest.skipUnless(_QT_AVAILABLE, f"PyQt6 unavailable: {_QT_ERROR}")
class B09B14B20WorkersAndCreatorTests(unittest.TestCase):
    def setUp(self):
        _ensure_qt_app()

    def test_adetailer_sam3_refine_reject_forbidden_paths_and_write_png_non_destructively(self):
        from workers.adetailer_worker import ADetailerSingleWorker
        from workers.sam3_worker import Sam3SingleWorker
        from workers.refine_worker import RefineWorker

        forbidden = r"C:\Windows\System32\drivers\etc\hosts" if os.name == "nt" else "/etc/passwd"
        for cls in (ADetailerSingleWorker, Sam3SingleWorker, RefineWorker):
            results = []
            w = cls(forbidden, {})
            w.finished.connect(results.append)
            with mock.patch("backends.get_backend", return_value=SimpleNamespace(
                adetailer=lambda *a, **k: _png_b64(),
                sam3=lambda *a, **k: _png_b64(),
                refine=lambda *a, **k: _png_b64(),
            )):
                w.run()
            self.assertEqual(len(results), 1)
            self.assertIn("error", json.loads(results[0]))

        # .jpg 입력 + file:/// URL → .png 로 저장 + 재실행 시 _2.png 로 비파괴 저장
        class _FakeBackend:
            def adetailer(self, _b64, _settings):
                return _png_b64()

            def sam3(self, _b64, _settings):
                return _png_b64()

        with tempfile.TemporaryDirectory() as tmp:
            src_jpg = os.path.join(tmp, "sample.jpg")
            Image.new("RGB", (8, 8), (10, 20, 30)).save(src_jpg, format="JPEG")
            out_dir = os.path.join(tmp, "out")
            file_url = "file:///" + src_jpg.replace("\\", "/")

            for cls, expected_stem in ((ADetailerSingleWorker, "sample_ad"), (Sam3SingleWorker, "sample_sam3")):
                r1, r2 = [], []
                w1 = cls(file_url, {"output_folder": out_dir})
                w2 = cls(src_jpg, {"output_folder": out_dir})
                w1.finished.connect(r1.append)
                w2.finished.connect(r2.append)
                with mock.patch("backends.get_backend", return_value=_FakeBackend()), \
                        mock.patch("core.model_cache.release_for_generation", return_value=0):
                    w1.run()
                    w2.run()
                d1, d2 = json.loads(r1[0]), json.loads(r2[0])
                self.assertTrue(d1["output_path"].endswith(f"{expected_stem}.png"), d1)
                self.assertTrue(d2["output_path"].endswith(f"{expected_stem}_2.png"), d2)
                self.assertNotEqual(d1["output_path"], d2["output_path"])

    def test_b04_workers_and_creator_accept_tif_tiff_avif_apng_extensions(self):
        from workers.adetailer_worker import ADetailerBatchWorker, ADetailerSingleWorker
        from workers.sam3_worker import Sam3BatchWorker, Sam3SingleWorker
        from workers.refine_worker import RefineWorker
        from ui.creator_actions import CreatorActionsMixin

        class _FakeBackend:
            def adetailer(self, _b64, _settings):
                return _png_b64()

            def sam3(self, _b64, _settings):
                return _png_b64()

            def refine(self, _b64, _settings):
                return _png_b64()

        extra_exts = (".tif", ".tiff", ".avif", ".apng")
        with tempfile.TemporaryDirectory() as tmp:
            out_dir = os.path.join(tmp, "out")
            files = []
            for idx, ext in enumerate(extra_exts):
                p = Path(tmp) / f"img_{idx}{ext}"
                p.write_bytes(_png_bytes())
                files.append(str(p))
                # Creator 경로 검증도 통과해야 함
                self.assertEqual(CreatorActionsMixin._creator_validate_input_path(p), p.resolve())

            with mock.patch("backends.get_backend", return_value=_FakeBackend()), \
                    mock.patch("core.model_cache.release_for_generation", return_value=0):
                for cls in (ADetailerSingleWorker, Sam3SingleWorker, RefineWorker):
                    for fpath in files:
                        res = []
                        w = cls(fpath, {"output_folder": out_dir})
                        w.finished.connect(res.append)
                        w.run()
                        data = json.loads(res[0])
                        self.assertNotIn("error", data, (cls.__name__, fpath, data))
                        self.assertTrue(os.path.isfile(data["output_path"]))

                for batch_cls in (ADetailerBatchWorker, Sam3BatchWorker):
                    res = []
                    w = batch_cls(files, {"output_folder": out_dir})
                    w.single_done.connect(res.append)
                    w.run()
                    self.assertEqual(len(res), len(files))
                    for item in res:
                        self.assertNotIn("error", json.loads(item))
                    kind, _msg = w.completion_notice()
                    self.assertEqual(kind, "success")

    def test_b03_b14_batch_workers_abort_without_false_success_toast(self):
        from ui.generator_main import GeneratorMainUI
        from workers.adetailer_worker import ADetailerBatchWorker
        from workers.sam3_worker import Sam3BatchWorker

        forbidden_out = r"C:\Windows\System32" if os.name == "nt" else "/etc"
        scenarios = (
            ("missing_backend", None, {}),
            ("invalid_output_folder", SimpleNamespace(
                adetailer=lambda *a, **k: _png_b64(),
                sam3=lambda *a, **k: _png_b64(),
            ), {"output_folder": forbidden_out}),
        )
        for label, backend_obj, settings in scenarios:
            with self.subTest(scenario=label):
                ad_done, ad_single, ad_prog = [], [], []
                ad_w = ADetailerBatchWorker(["a.png", "b.png"], settings)
                ad_w.single_done.connect(ad_single.append)
                ad_w.progress.connect(lambda c, t: ad_prog.append((c, t)))
                ad_w.all_done.connect(lambda: ad_done.append(True))

                sam_done, sam_single, sam_prog = [], [], []
                sam_w = Sam3BatchWorker(["a.png", "b.png"], settings)
                sam_w.single_done.connect(sam_single.append)
                sam_w.progress.connect(lambda c, t: sam_prog.append((c, t)))
                sam_w.all_done.connect(lambda: sam_done.append(True))

                with mock.patch("backends.get_backend", return_value=backend_obj):
                    ad_w.run()
                    sam_w.run()

                self.assertEqual(ad_done, [True])
                self.assertEqual(sam_done, [True])
                self.assertEqual(ad_prog, [(0, 0)])
                self.assertEqual(sam_prog, [(0, 0)])
                self.assertIn("error", json.loads(ad_single[0]))
                self.assertIn("error", json.loads(sam_single[0]))
                self.assertFalse(ad_w.job.running)
                self.assertFalse(sam_w.job.running)
                for worker in (ad_w, sam_w):
                    kind, msg = worker.completion_notice()
                    self.assertEqual(kind, "error", (label, kind, msg))
                    self.assertNotIn("완료", msg)

                # 실제 _run_adetailer_batch / _run_sam3_batch 핸들러를 거쳐도 false '완료(N장)' success 토스트가 뜨지 않음
                bridge = _StubBridge()
                bridge.adetailerProgress = _StubSignal()
                bridge.adetailerResult = _StubSignal()
                bridge.sam3Progress = _StubSignal()
                bridge.sam3Result = _StubSignal()
                host = SimpleNamespace(vue_bridge=bridge)

                class _SyncAD(ADetailerBatchWorker):
                    def __init__(self, paths, st, parent=None):
                        super().__init__(paths, st, parent=None)

                    def start(self):
                        self.run()

                class _SyncSam3(Sam3BatchWorker):
                    def __init__(self, paths, st, parent=None):
                        super().__init__(paths, st, parent=None)

                    def start(self):
                        self.run()

                with mock.patch("backends.get_backend", return_value=backend_obj), \
                        mock.patch("workers.adetailer_worker.ADetailerBatchWorker", _SyncAD), \
                        mock.patch("workers.sam3_worker.Sam3BatchWorker", _SyncSam3), \
                        mock.patch("ui.aux_pass_snapshot.attach_sampling_snapshot", side_effect=lambda _h, s: dict(s)), \
                        mock.patch("ui.sam_extra_notices_ui.check_standalone_sam3", lambda *_a: None):
                    GeneratorMainUI._run_adetailer_batch(host, {"paths": ["a.png", "b.png"], "settings": settings})
                    GeneratorMainUI._run_sam3_batch(host, {"paths": ["a.png", "b.png"], "settings": settings})

                success_toasts = [m for lvl, m in bridge.showNotification.emitted if lvl == "success"]
                error_toasts = [m for lvl, m in bridge.showNotification.emitted if lvl == "error"]
                self.assertEqual(success_toasts, [], bridge.showNotification.emitted)
                self.assertEqual(len(error_toasts), 2, bridge.showNotification.emitted)
                self.assertIn((0, 0), bridge.adetailerProgress.emitted)
                self.assertIn((0, 0), bridge.sam3Progress.emitted)

    def test_b14_handlers_guard_running_workers_and_attach_cleanup(self):
        from ui.generator_main import GeneratorMainUI
        from ui.worker_helpers import attach_worker_cleanup

        bridge = _StubBridge()
        running = SimpleNamespace(isRunning=lambda: True)
        host = SimpleNamespace(
            vue_bridge=bridge,
            _ad_worker=running,
            _ad_batch_worker=running,
            _sam3_worker=running,
            _sam3_batch_worker=running,
            _refine_worker=running,
        )
        GeneratorMainUI._run_adetailer_single(host, {"path": "a.png"})
        GeneratorMainUI._run_adetailer_batch(host, {"paths": ["a.png"]})
        GeneratorMainUI._run_sam3_single(host, {"path": "a.png"})
        GeneratorMainUI._run_sam3_batch(host, {"paths": ["a.png"]})
        GeneratorMainUI._run_refine(host, {"path": "a.png"})
        self.assertEqual(len(bridge.showNotification.emitted), 5)
        self.assertTrue(all(lvl == "warning" for lvl, _ in bridge.showNotification.emitted))

        # 실제 QThread 종료 시 참조 해제 + deleteLater 호출 확인
        class _ShortWorker(QThread):
            finished = None  # 의도적 가림 시뮬레이션 대신 실제 ADetailerSingleWorker 패턴 확인

        from workers.adetailer_worker import ADetailerSingleWorker
        w = ADetailerSingleWorker("nonexistent.png", {})
        cleanup_host = SimpleNamespace(_ad_worker=w)
        with mock.patch.object(w, "deleteLater") as del_mock, \
                mock.patch("backends.get_backend", return_value=None):
            attach_worker_cleanup(cleanup_host, "_ad_worker", w)
            w.start()
            self.assertTrue(w.wait(3000))
            _ensure_qt_app().processEvents()
        self.assertIsNone(cleanup_host._ad_worker)
        del_mock.assert_called_once()

    def test_b09_creator_validates_forbidden_path_and_upload_size_limit(self):
        from ui.creator_actions import CreatorActionsMixin

        forbidden = r"C:\Windows\System32\drivers\etc\hosts" if os.name == "nt" else "/etc/passwd"
        with self.assertRaises(ValueError):
            CreatorActionsMixin._creator_validate_input_path(forbidden)

        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "clip.mp4"
            p.write_bytes(b"0123456789")
            # file:/// URL 허용
            resolved = CreatorActionsMixin._creator_validate_input_path("file:///" + str(p).replace("\\", "/"))
            self.assertEqual(resolved, p.resolve())
            # 크기 상한 초과 시 거부
            with mock.patch.object(CreatorActionsMixin, "CREATOR_MAX_UPLOAD_BYTES", 4):
                with self.assertRaisesRegex(ValueError, "너무 큽니다"):
                    CreatorActionsMixin._creator_validate_input_path(p)


# ── B-13: scandir 이미지 스캔 ────────────────────────────────────────────────

class B13FolderScanTests(unittest.TestCase):
    def test_scan_image_folder_handles_brackets_unicode_and_case_without_dupes(self):
        from ui.worker_helpers import scan_image_folder

        with tempfile.TemporaryDirectory() as tmp:
            folder = os.path.join(tmp, "[Artist] 캐릭터 [2026]")
            os.makedirs(folder)
            for fn in ("01.png", "02.JPG", "03.webp", "note.txt"):
                Path(folder, fn).write_bytes(b"x")
            found = scan_image_folder(folder, frozenset({".png", ".jpg", ".jpeg", ".webp"}))
            basenames = [os.path.basename(p) for p in found]
            self.assertEqual(basenames, ["01.png", "02.JPG", "03.webp"])


# ── B-15 / B-18: is_programmatic_change try/finally 복원 ──────────────────────

class B15ProgrammaticChangeFinallyTests(unittest.TestCase):
    def test_load_settings_restores_flag_on_exception(self):
        from ui.generator_settings import SettingsMixin

        with tempfile.TemporaryDirectory() as tmp:
            cfg = os.path.join(tmp, "prompt_settings.json")
            Path(cfg).write_text("{}", encoding="utf-8")

            # (1) 기본 상태 False → load_settings 중 True로 바뀐 뒤 예외 → finally 로 False 복원
            seen_during_error = []
            host = SimpleNamespace(
                is_programmatic_change=False,
                show_status=lambda *_a, **_kw: None,
                vue_bridge=_StubBridge(),
            )

            def _explode_on_apply(h, _settings, **_kw):
                seen_during_error.append(h.is_programmatic_change)
                raise RuntimeError("boom")

            with mock.patch("ui.generator_settings.PROMPT_SETTINGS_FILE", cfg), \
                    mock.patch("ui.generation_settings_apply.apply_prompt_settings", side_effect=_explode_on_apply):
                SettingsMixin.load_settings(host)
            self.assertEqual(seen_during_error, [True])
            self.assertFalse(host.is_programmatic_change)

            # (2) 중첩 상태 True → 예외 발생 시에도 외부 True 유지
            nested_host = SimpleNamespace(
                is_programmatic_change=True,
                show_status=lambda *_a, **_kw: None,
                vue_bridge=_StubBridge(),
            )
            with mock.patch("ui.generator_settings.PROMPT_SETTINGS_FILE", cfg), \
                    mock.patch("ui.generation_settings_apply.apply_prompt_settings", side_effect=_explode_on_apply):
                SettingsMixin.load_settings(nested_host)
            self.assertTrue(nested_host.is_programmatic_change)

    def test_apply_prompt_from_data_restores_flag_on_exception_and_preserves_nested(self):
        from ui.generator_prompts import PromptHandlingMixin

        def _make_host(initial_flag: bool, char_count_widget, cond_hook=None):
            empty_txt = SimpleNamespace(toPlainText=lambda: "", setPlainText=lambda _v: None)
            plain_line = SimpleNamespace(setText=lambda _v: None)
            return SimpleNamespace(
                is_programmatic_change=initial_flag,
                char_count_input=char_count_widget,
                character_input=plain_line,
                copyright_input=plain_line,
                artist_input=empty_txt,
                main_prompt_text=empty_txt,
                prefix_prompt_text=empty_txt,
                suffix_prompt_text=empty_txt,
                neg_prompt_text=empty_txt,
                exclude_prompt_local_input=empty_txt,
                prompt_cleaner=SimpleNamespace(clean=lambda tags: tags),
                chk_remove_artist=SimpleNamespace(isChecked=lambda: False),
                chk_remove_copyright=SimpleNamespace(isChecked=lambda: False),
                chk_remove_character=SimpleNamespace(isChecked=lambda: False),
                chk_remove_meta=SimpleNamespace(isChecked=lambda: False),
                chk_remove_censorship=SimpleNamespace(isChecked=lambda: False),
                chk_remove_text=SimpleNamespace(isChecked=lambda: False),
                btn_lock_artist=SimpleNamespace(isChecked=lambda: False),
                _apply_conditional_prompts=cond_hook or (lambda: None),
                update_total_prompt_display=lambda: None,
            )

        # (1) 비중첩(False) 상태에서 True 설정 직후 예외 → try/finally 가 없으면 True 에 고착되어 실패함
        seen_during_error = []

        class _ExplodingWidget:
            def __init__(self, host_ref):
                self._host_ref = host_ref

            def setText(self, _v):
                seen_during_error.append(self._host_ref[0].is_programmatic_change)
                raise RuntimeError("boom")

        ref: list = []
        host = _make_host(False, _ExplodingWidget(ref))
        ref.append(host)
        with self.assertRaises(RuntimeError):
            PromptHandlingMixin.apply_prompt_from_data(host, {"general": "1girl"})
        self.assertEqual(seen_during_error, [True])
        self.assertFalse(host.is_programmatic_change)

        # (2) 중첩(True) 상태에서 UI 업데이트 블록이 정상 종료되어도 외부 True 를 False 로 덮지 않아야 함
        observed_after_ui_block = []
        nested_ref: list = []

        def _capture_flag_and_stop():
            observed_after_ui_block.append(nested_ref[0].is_programmatic_change)
            raise StopIteration("stop_after_ui_update")

        nested_host = _make_host(True, SimpleNamespace(setText=lambda _v: None), cond_hook=_capture_flag_and_stop)
        nested_ref.append(nested_host)
        with self.assertRaises(StopIteration):
            PromptHandlingMixin.apply_prompt_from_data(nested_host, {"general": "1girl"})
        self.assertEqual(observed_after_ui_block, [True])
        self.assertTrue(nested_host.is_programmatic_change)

        # (3) 중첩(True) 상태에서 UI 업데이트 중 예외 발생 시에도 True 유지
        nested_err_ref: list = []
        nested_err_host = _make_host(True, _ExplodingWidget(nested_err_ref))
        nested_err_ref.append(nested_err_host)
        with self.assertRaises(RuntimeError):
            PromptHandlingMixin.apply_prompt_from_data(nested_err_host, {"general": "1girl"})
        self.assertTrue(nested_err_host.is_programmatic_change)


# ── B-20(3), B-21(1,5,6), P-07(1) ────────────────────────────────────────────

class B20B21P07TargetedFixesTests(unittest.TestCase):
    def test_b20_export_canceled_emits_terminal_event_relight_and_hand(self):
        from ui.relight_actions import RelightActionsMixin
        from ui.hand_reconstruction_actions import HandReconstructionActionsMixin

        relight_events = []
        relight_host = RelightActionsMixin()
        relight_host.web_mode = False
        relight_host.vue_bridge = SimpleNamespace(
            relightEvent=SimpleNamespace(emit=lambda s: relight_events.append(json.loads(s)))
        )
        relight_host._ensure_relight_runtime()
        relight_host._relight_preview = {"requestId": "p1", "png": _png_bytes()}

        # 스레드 시작 직전에 취소 플래그가 서도록 Thread 래핑
        class _CancelBeforeRunThread:
            def __init__(self, target, **_kw):
                self._target = target

            def start(self):
                relight_host._relight_job["cancel"].set()
                self._target()

        with tempfile.TemporaryDirectory() as tmp, \
                mock.patch.dict(sys.modules, {"config": SimpleNamespace(OUTPUT_DIR=tmp)}), \
                mock.patch("ui.relight_actions.threading.Thread", _CancelBeforeRunThread):
            relight_host._run_relight_action("relight_export", {"requestId": "e1", "previewRequestId": "p1"})
        self.assertEqual(len(relight_events), 1)
        self.assertFalse(relight_events[0]["ok"])
        self.assertTrue(relight_events[0]["canceled"])

        hand_events = []
        hand_host = HandReconstructionActionsMixin()
        hand_host.web_mode = False
        hand_host.vue_bridge = SimpleNamespace(
            handReconstructionEvent=SimpleNamespace(emit=lambda s: hand_events.append(json.loads(s)))
        )
        hand_host._ensure_hand_runtime()
        hand_host._hand_preview = {"requestId": "hp1", "candidates": [{"png": _png_bytes()}]}

        class _CancelHandThread:
            def __init__(self, target, **_kw):
                self._target = target

            def start(self):
                hand_host._hand_job["cancel"].set()
                self._target()

        with tempfile.TemporaryDirectory() as tmp, \
                mock.patch.dict(sys.modules, {"config": SimpleNamespace(OUTPUT_DIR=tmp)}), \
                mock.patch("ui.hand_reconstruction_actions.threading.Thread", _CancelHandThread):
            hand_host._handle_hand_reconstruction_action(
                "hand_reconstruction_export",
                {"requestId": "he1", "previewRequestId": "hp1", "candidateIndex": 0},
            )
        self.assertEqual(len(hand_events), 1)
        self.assertFalse(hand_events[0]["ok"])
        self.assertTrue(hand_events[0]["canceled"])
        self.assertEqual(hand_events[0]["phase"], "complete")

    @unittest.skipUnless(_QT_AVAILABLE, f"PyQt6 unavailable: {_QT_ERROR}")
    def test_b21_rename_file_case_insensitive_extension(self):
        from ui.vue_bridge import VueBridge

        _ensure_qt_app()
        bridge = VueBridge()
        with tempfile.TemporaryDirectory() as tmp:
            src = Path(tmp) / "Photo.PNG"
            src.write_bytes(_png_bytes())
            res = json.loads(bridge.renameFile(str(src), "renamed.png"))
            self.assertTrue(res.get("ok"), res)
            self.assertTrue(res["new_path"].endswith("/renamed.png"), res)
            self.assertFalse(res["new_path"].endswith(".png.PNG"), res)

    def test_b21_add_search_to_queue_handles_malformed_numbers_safely(self):
        from ui.generator_main import GeneratorMainUI

        added = []
        host = SimpleNamespace(
            is_programmatic_change=False,
            apply_prompt_from_data=lambda _b: None,
            update_total_prompt_display=lambda: None,
            total_prompt_display=SimpleNamespace(toPlainText=lambda: "1girl"),
            neg_prompt_text=SimpleNamespace(toPlainText=lambda: "bad"),
            sampler_combo=SimpleNamespace(currentText=lambda: "Euler"),
            steps_input=SimpleNamespace(text=lambda: "abc"),
            cfg_input=SimpleNamespace(text=lambda: "bad_float"),
            seed_input=SimpleNamespace(text=lambda: ""),
            width_input=SimpleNamespace(text=lambda: "1024.0"),
            height_input=SimpleNamespace(text=lambda: "not_int"),
            queue_panel=SimpleNamespace(add_single_item=added.append),
            vue_bridge=_StubBridge(),
            _handle_model_download_action=lambda *_a: False,
            _handle_chat_action=lambda *_a: False,
            _handle_creator_action=lambda *_a: False,
        )
        GeneratorMainUI._handle_vue_action(host, "add_search_to_queue", {"general": "1girl"})
        self.assertEqual(len(added), 1)
        self.assertEqual(added[0]["steps"], 20)
        self.assertEqual(added[0]["cfg_scale"], 7.0)
        self.assertEqual(added[0]["seed"], -1)
        self.assertEqual(added[0]["width"], 1024)
        self.assertEqual(added[0]["height"], 1024)

    def test_p07_build_settings_dict_reads_prompt_settings_file_at_most_once(self):
        from ui.generator_settings import SettingsMixin

        with tempfile.TemporaryDirectory() as tmp:
            cfg = os.path.join(tmp, "prompt_settings.json")
            Path(cfg).write_text(json.dumps({
                "model": "m.safetensors",
                "sampler": "Euler",
                "scheduler": "Normal",
                "vae_main": "vae.pt",
                "hires_upscaler": "Latent",
                "hires_checkpoint": "m2.safetensors",
                "hires_sampler": "Euler",
                "hires_scheduler": "Normal",
            }), encoding="utf-8")

            empty_text = SimpleNamespace(text=lambda: "", toPlainText=lambda: "")
            empty_combo = SimpleNamespace(currentText=lambda: "", preservedText=lambda: "", currentIndex=lambda: 0)
            off_chk = SimpleNamespace(isChecked=lambda: False)
            host = SettingsMixin()
            for name in (
                "char_count_input", "character_input", "copyright_input", "artist_input",
                "prefix_prompt_text", "main_prompt_text", "suffix_prompt_text", "neg_prompt_text",
                "exclude_prompt_local_input", "te_main_input", "steps_input", "cfg_input", "shift_input",
                "seed_input", "width_input", "height_input", "hires_steps_input", "hires_denoising_input",
                "hires_scale_input", "hires_cfg_input", "hires_prompt_text", "hires_neg_prompt_text",
            ):
                setattr(host, name, empty_text)
            for name in (
                "model_combo", "vae_main_combo", "sampler_combo", "scheduler_combo",
                "upscaler_combo", "hires_checkpoint_combo", "hires_sampler_combo", "hires_scheduler_combo",
                "combo_char_feature_mode",
            ):
                setattr(host, name, empty_combo)
            for name in (
                "btn_lock_artist", "random_res_check", "hires_options_group",
                "adetailer_group", "ad_slot1_group", "ad_slot2_group", "sam3_group",
                "chk_remove_artist", "chk_remove_copyright", "chk_remove_character",
                "chk_remove_meta", "chk_remove_censorship", "chk_remove_text",
                "chk_prompt_focus", "chk_auto_char_features", "chk_auto_remove_char_features",
            ):
                setattr(host, name, off_chk)
            host.random_resolutions = []
            host.s1_widgets = {}
            host.s2_widgets = {}
            host.base_prefix_prompt = ""
            host.base_suffix_prompt = ""
            host.base_neg_prompt = ""
            host.web_tab = SimpleNamespace()
            host._get_slot_settings = lambda _w, _k: {}
            host._get_backend_type_str = lambda: "webui"
            host._get_webui_url = lambda: ""
            host._get_comfyui_url = lambda: ""
            host._get_comfyui_workflow_path = lambda: ""
            host._get_comfyui_workflow_img2img_path = lambda: ""
            host._model_workflow_baseline = lambda: None

            with mock.patch("ui.generator_settings.PROMPT_SETTINGS_FILE", cfg), \
                    mock.patch.object(host, "_load_existing_settings_file", wraps=host._load_existing_settings_file) as spy:
                result = host._build_settings_dict()

            self.assertEqual(spy.call_count, 1)
            self.assertEqual(result["model"], "m.safetensors")
            self.assertEqual(result["hires_scheduler"], "Normal")


if __name__ == "__main__":
    unittest.main()
