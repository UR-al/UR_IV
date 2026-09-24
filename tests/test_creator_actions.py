import json
import tempfile
import unittest
from unittest import mock
from pathlib import Path
from types import SimpleNamespace

from core.comfy_upload_names import content_upload_name
from core.comic_studio import ComicRevisionConflict
from core.creator_workflows import build
from ui.creator_actions import CreatorActionsMixin


class _UploadBackend:
    """duck-typed 어댑터 — overwrite/cancel_check 를 받지 않는다(옛 어댑터 호환 경로)."""

    def __init__(self):
        self.uploads = []

    def upload_media(self, data, filename, mime):
        self.uploads.append((data, filename, mime))
        return f"studio/{filename}"


class _ComfyInputFolder:
    """ComfyUI server.py image_upload 의미론: overwrite=true 면 같은 이름을 덮어쓰고,
    false 면 같은 이름·같은 내용은 그 이름을 돌려주고 다른 내용이면 'name (n).ext' 로 비켜 간다."""

    def __init__(self, existing=None):
        self.files = dict(existing or {})
        self.calls = []

    def upload_media(self, data, filename, mime="application/octet-stream", overwrite=True, cancel_check=None):
        self.calls.append((filename, overwrite))
        name = filename
        if not overwrite:
            stem, dot, ext = filename.rpartition(".")
            index = 1
            while name in self.files:
                if self.files[name] == bytes(data):
                    return name
                name = f"{stem} ({index}){dot}{ext}"
                index += 1
        self.files[name] = bytes(data)
        return name


class _Signal:
    def __init__(self):
        self.values = []

    def emit(self, value):
        self.values.append(json.loads(value))


class _Bridge:
    def __init__(self):
        self.comicDocumentChanged = _Signal()


class _Document:
    def __init__(self, revision=3, title="saved"):
        self.revision = revision
        self.title = title

    def to_dict(self):
        return {
            "title": self.title,
            "revision": self.revision,
            "contentHash": "a" * 64,
            "panels": [{"id": "panel-1", "prompt": "ok"}],
        }


class _ImmediateThread:
    def __init__(self, target, **_kwargs):
        self.target = target

    def start(self):
        self.target()


class CreatorActionAdapterTests(unittest.TestCase):
    def setUp(self):
        self.actions = CreatorActionsMixin()

    def test_v2v_payload_uploads_video_and_identity_to_distinct_inputs(self):
        with tempfile.TemporaryDirectory() as tmp:
            video = Path(tmp) / "motion.mp4"
            identity = Path(tmp) / "face.png"
            video.write_bytes(b"video")
            identity.write_bytes(b"image")
            backend = _UploadBackend()

            params = self.actions._creator_prepare_params(
                backend,
                {
                    "mode": "h3_v2v",
                    "sourcePath": str(video),
                    "identityPath": str(identity),
                    "prompt": "walk forward",
                    "negative": "flicker",
                    "audioPrompt": "rain and footsteps",
                    "dialogue": "hello",
                    "seed": -1,
                    "includeAudio": True,
                },
            )

        # 업로드 이름은 사용자 파일명이 아니라 역할별 접두어 + 내용 해시.
        self.assertEqual(params["input_video"], "studio/" + content_upload_name(b"video", "creator_source", "mp4"))
        self.assertEqual(params["input_image"], "studio/" + content_upload_name(b"image", "creator_identity", "png"))
        # overwrite 를 받지 않는 duck-typed 어댑터에는 3인자로만 호출한다.
        self.assertEqual([upload[2] for upload in backend.uploads], ["video/mp4", "image/png"])
        self.assertGreaterEqual(params["seed"], 0)
        self.assertTrue(params["generate_audio"])
        self.assertTrue(params["include_reference_audio"])
        self.assertIn("<Picture 1>", params["prompt"])
        self.assertIn("<Video 1>", params["prompt"])
        self.assertIn("<Audio 1>", params["prompt"])
        self.assertIn("Overall soundscape: rain and footsteps", params["prompt"])
        self.assertIn("Dialogue: hello", params["prompt"])
        self.assertIn("Avoid: flicker", params["prompt"])

    def test_krea_payload_keeps_source_and_reference_separate(self):
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "source.jpg"
            reference = Path(tmp) / "identity.webp"
            source.write_bytes(b"source")
            reference.write_bytes(b"reference")
            backend = _UploadBackend()
            params = self.actions._creator_prepare_params(
                backend,
                {
                    "mode": "krea2",
                    "sourcePath": str(source),
                    "referencePath": str(reference),
                    "prompt": "change clothing",
                    "seed": 7,
                },
            )

        self.assertEqual(params["input_image"], "studio/" + content_upload_name(b"source", "creator_source", "jpg"))
        self.assertEqual(
            params["reference_image"], "studio/" + content_upload_name(b"reference", "creator_reference", "webp"),
        )
        self.assertEqual(params["seed"], 7)

    def _prepare_krea_same_basename(self, backend, tmp):
        source = Path(tmp) / "source" / "0001.png"
        reference = Path(tmp) / "reference" / "0001.png"
        source.parent.mkdir()
        reference.parent.mkdir()
        source.write_bytes(b"SOURCE-BYTES")
        reference.write_bytes(b"REFERENCE-BYTES")
        return self.actions._creator_prepare_params(backend, {
            "mode": "krea2", "prompt": "change coat", "seed": 7,
            "sourcePath": str(source), "referencePath": str(reference),
        })

    def test_same_basename_source_and_reference_do_not_overwrite_each_other(self):
        # 다른 폴더의 0001.png 두 장: 예전엔 둘 다 '0001.png' 로 overwrite=true 업로드돼 참조가
        # 원본을 덮었고, Krea2 편집의 LoadImage 5·16 이 모두 참조 그림을 읽었다. ComfyUI/input 에
        # 원래 있던 사용자의 0001.png 도 덮였다.
        backend = _ComfyInputFolder(existing={"0001.png": b"USER-FILE"})
        with tempfile.TemporaryDirectory() as tmp:
            params = self._prepare_krea_same_basename(backend, tmp)

        self.assertNotEqual(params["input_image"], params["reference_image"])
        self.assertEqual(backend.files[params["input_image"]], b"SOURCE-BYTES")
        self.assertEqual(backend.files[params["reference_image"]], b"REFERENCE-BYTES")
        self.assertEqual(backend.files["0001.png"], b"USER-FILE")
        self.assertEqual([overwrite for _name, overwrite in backend.calls], [False, False])
        loads = {
            node_id: node["inputs"]["image"]
            for node_id, node in build("krea2_edit", params)["workflow"].items()
            if node.get("class_type") == "LoadImage"
        }
        self.assertEqual(loads, {"5": params["input_image"], "16": params["reference_image"]})

    def test_repeated_uploads_reuse_the_same_input_names(self):
        backend = _ComfyInputFolder()
        with tempfile.TemporaryDirectory() as tmp:
            first = self._prepare_krea_same_basename(backend, tmp)
        with tempfile.TemporaryDirectory() as tmp:
            second = self._prepare_krea_same_basename(backend, tmp)
        self.assertEqual(
            (first["input_image"], first["reference_image"]),
            (second["input_image"], second["reference_image"]),
        )
        self.assertEqual(len(backend.files), 2)  # input 폴더에 사본이 쌓이지 않는다

    def test_identity_left_from_v2v_does_not_replace_the_i2v_start_frame(self):
        # Vue 는 모든 영상 모드에 identityPath 를 보냈다 — 아이덴티티는 V2V 그래프만 읽는다.
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "src.png"
            face = Path(tmp) / "face.png"
            source.write_bytes(b"start-frame")
            face.write_bytes(b"identity")
            backend = _ComfyInputFolder()
            params = self.actions._creator_prepare_params(backend, {
                "mode": "h3_i2v", "prompt": "walk", "seed": 3,
                "sourcePath": str(source), "identityPath": str(face),
            })
            self.assertEqual(backend.files[params["input_image"]], b"start-frame")
            self.assertEqual(len(backend.calls), 1)
            self.assertNotIn("identityPath", params)

            # T2V/T2I 는 미디어를 읽지 않는다 — 남은 원본도 올리지 않는다.
            for mode in ("h3_t2v", "krea2_t2i"):
                with self.subTest(mode=mode):
                    idle = _ComfyInputFolder()
                    prepared = self.actions._creator_prepare_params(idle, {
                        "mode": mode, "prompt": "walk", "seed": 3,
                        "sourcePath": str(source), "identityPath": str(face), "referencePath": str(face),
                    })
                    self.assertEqual(idle.calls, [])
                    self.assertNotIn("input_image", prepared)
                    self.assertNotIn("sourcePath", prepared)

    def test_inputs_are_validated_before_any_upload(self):
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "source.png"
            reference = Path(tmp) / "reference"  # 확장자 없음
            source.write_bytes(b"source")
            reference.write_bytes(b"reference")
            backend = _ComfyInputFolder()
            with self.assertRaisesRegex(ValueError, "확장자"):
                self.actions._creator_prepare_params(backend, {
                    "mode": "krea2", "prompt": "x", "seed": 7,
                    "sourcePath": str(source), "referencePath": str(reference),
                })
            with self.assertRaises(FileNotFoundError):
                self.actions._creator_prepare_params(backend, {
                    "mode": "krea2", "prompt": "x", "seed": 7,
                    "sourcePath": str(source), "referencePath": str(Path(tmp) / "missing.png"),
                })
        self.assertEqual(backend.calls, [])  # 원본만 ComfyUI/input 에 남기지 않는다

    def test_artifact_extension_prefers_filename_then_mime(self):
        extension = CreatorActionsMixin._creator_artifact_extension
        self.assertEqual(extension("Krea2_00001_.PNG", "image/png"), "png")
        self.assertEqual(extension("", "image/jpeg"), "jpg")
        self.assertEqual(extension(None, "image/webp"), "webp")
        self.assertEqual(extension("no-suffix", ""), "png")

    def test_krea2_hires_reupload_uses_a_content_name_without_overwrite(self):
        import threading
        from contextlib import nullcontext
        from io import BytesIO

        from PIL import Image

        from backends import BackendType

        buffer = BytesIO()
        Image.new("RGB", (64, 64), (1, 2, 3)).save(buffer, format="PNG")
        edited = buffer.getvalue()
        actions = self.actions
        actions._creator_cancel_event = threading.Event()
        actions._creator_should_unload_ollama = lambda: False
        actions._creator_reserve = lambda *_a, **_k: nullcontext()
        actions._creator_emit = lambda *_a, **_k: None
        actions._creator_object_info = lambda *_a: {}
        actions._creator_check_nodes = lambda *_a: None
        actions._creator_resolve_comfy_choices = lambda *_a: None
        actions._creator_save_artifacts = lambda artifacts, _mode: []
        runs = []

        def run(_backend, built, _progress):
            runs.append(built)
            artifacts = [SimpleNamespace(kind="image", data=edited, filename="0001.png", mime="image/png")]
            return SimpleNamespace(success=True, artifacts=artifacts if len(runs) == 1 else [], info={}, error="")

        actions._creator_run_workflow = run
        backend = _ComfyInputFolder(existing={"0001.png": b"USER-FILE"})
        backend.run_workflow = mock.Mock()
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "source.png"
            reference = Path(tmp) / "reference.png"
            source.write_bytes(b"source")
            reference.write_bytes(b"reference")
            with mock.patch("backends.get_backend_type", return_value=BackendType.COMFYUI), \
                    mock.patch("backends.get_backend", return_value=backend):
                actions._creator_generate({
                    "mode": "krea2", "prompt": "x", "seed": 7, "hires": True,
                    "sourcePath": str(source), "referencePath": str(reference),
                })

        self.assertEqual(len(runs), 2)
        hires_loads = [node["inputs"]["image"] for node in runs[1]["workflow"].values()
                       if node.get("class_type") == "LoadImage"]
        self.assertEqual(hires_loads, [content_upload_name(edited, "krea2_hires_input", "png")])
        self.assertEqual(backend.files["0001.png"], b"USER-FILE")
        self.assertTrue(all(overwrite is False for _name, overwrite in backend.calls))

    def test_living_comic_panel_upload_uses_a_content_name_without_overwrite(self):
        import threading
        from contextlib import nullcontext

        from backends import BackendType

        actions = self.actions
        actions._creator_cancel_event = threading.Event()
        actions._creator_should_unload_ollama = lambda: False
        actions._creator_reserve = lambda *_a, **_k: nullcontext()
        actions._creator_emit = lambda *_a, **_k: None
        actions._creator_object_info = lambda *_a: {}
        actions._creator_configure_h3_cache = lambda *_a: None
        actions._creator_check_nodes = lambda *_a: None
        actions._creator_resolve_comfy_choices = lambda *_a: None
        queued = []

        def run(_backend, built, _progress):
            queued.append(built)
            raise RuntimeError("stop after upload")

        actions._creator_run_workflow = run
        backend = _ComfyInputFolder(existing={"panel.png": b"USER-FILE"})
        with tempfile.TemporaryDirectory() as tmp:
            image = Path(tmp) / "panel.png"
            image.write_bytes(b"panel-bytes")
            panel = SimpleNamespace(image_path=str(image), motion_prompt="walk", text="", seed=-1, video_path="")
            document = SimpleNamespace(panels=[panel], to_dict=lambda: {})
            actions._comic_studio = lambda: SimpleNamespace(normalize=lambda _d: document, save=lambda d: d)
            with mock.patch("backends.get_backend_type", return_value=BackendType.COMFYUI), \
                    mock.patch("backends.get_backend", return_value=backend):
                with self.assertRaisesRegex(RuntimeError, "stop after upload"):
                    actions._comic_animate_all({"document": {}})

        expected = content_upload_name(b"panel-bytes", "comic_panel", "png")
        self.assertEqual(backend.calls, [(expected, False)])
        self.assertEqual(backend.files["panel.png"], b"USER-FILE")
        loads = [node["inputs"]["image"] for node in queued[0]["workflow"].values()
                 if node.get("class_type") == "LoadImage"]
        self.assertEqual(loads, [expected])

    def test_required_node_check_reports_only_missing_types(self):
        with self.assertRaisesRegex(RuntimeError, "MissingNode"):
            self.actions._creator_check_nodes(
                {"required_node_types": ["LoadImage", "MissingNode"]},
                {"LoadImage"},
            )

    def test_comfy_combo_paths_are_replaced_with_server_native_separator(self):
        built = {
            "workflow": {
                "4": {
                    "class_type": "LoraLoaderModelOnly",
                    "inputs": {"lora_name": "Krea2/identity.safetensors"},
                }
            }
        }
        object_info = {
            "LoraLoaderModelOnly": {
                "input": {
                    "required": {
                        "lora_name": [["Krea2\\identity.safetensors"], {}]
                    }
                }
            }
        }
        self.actions._creator_resolve_comfy_choices(built, object_info)
        self.assertEqual(
            built["workflow"]["4"]["inputs"]["lora_name"],
            "Krea2\\identity.safetensors",
        )

    def test_comfy_model_choice_can_match_same_stem_packaging_variant(self):
        built = {
            "workflow": {
                "7": {
                    "class_type": "UpscaleModelLoader",
                    "inputs": {"model_name": "RealESRGAN_x4plus.safetensors"},
                }
            }
        }
        object_info = {
            "UpscaleModelLoader": {
                "input": {
                    "required": {
                        "model_name": [["RealESRGAN_x4plus.pth"], {}]
                    }
                }
            }
        }
        self.actions._creator_resolve_comfy_choices(built, object_info)
        self.assertEqual(
            built["workflow"]["7"]["inputs"]["model_name"],
            "RealESRGAN_x4plus.pth",
        )

    def test_comic_save_forwards_expected_revision_and_returns_ack_metadata(self):
        bridge = _Bridge()
        self.actions.vue_bridge = bridge
        studio = mock.Mock()
        studio.save.return_value = _Document(revision=8)
        self.actions._comic_studio = lambda: studio

        self.actions._comic_save(
            {
                "document": {"title": "edit", "panels": [{}]},
                "expectedRevision": 7,
                "requestId": "save-42",
            }
        )

        studio.save.assert_called_once_with(
            {"title": "edit", "panels": [{}]}, expected_revision=7
        )
        emitted = bridge.comicDocumentChanged.values[-1]
        self.assertEqual(emitted["document"]["revision"], 8)
        self.assertEqual(emitted["persistence"]["status"], "saved")
        self.assertEqual(emitted["persistence"]["requestId"], "save-42")

    def test_comic_save_conflict_returns_authoritative_document(self):
        bridge = _Bridge()
        self.actions.vue_bridge = bridge
        studio = mock.Mock()
        studio.save.side_effect = ComicRevisionConflict(2, 3, _Document(revision=3, title="newer"))
        self.actions._comic_studio = lambda: studio

        self.actions._comic_save(
            {
                "document": {"title": "stale", "panels": [{}]},
                "expectedRevision": 2,
                "requestId": "save-stale",
            }
        )

        emitted = bridge.comicDocumentChanged.values[-1]
        self.assertEqual(emitted["document"]["title"], "newer")
        self.assertEqual(emitted["persistence"]["status"], "conflict")
        self.assertEqual(emitted["persistence"]["actualRevision"], 3)

    def test_creator_state_reconciles_frontend_recovery_before_emitting_document(self):
        bridge = _Bridge()
        bridge.creatorStateChanged = _Signal()
        self.actions.vue_bridge = bridge
        self.actions._creator_running = False
        self.actions._creator_mode = ""
        recovery = {"schema": 2, "documentJson": "{}"}
        studio = mock.Mock()
        studio.reconcile.return_value = SimpleNamespace(
            document=_Document(revision=5),
            status="recovered",
            conflict_path="",
        )
        self.actions._comic_studio = lambda: studio

        with (
            mock.patch("ui.creator_actions.threading.Thread", _ImmediateThread),
            mock.patch("backends.get_backend"),
            mock.patch("backends.get_backend_type") as get_backend_type,
        ):
            get_backend_type.return_value.value = "forge"
            self.actions._creator_request_state({"comicRecovery": recovery})

        studio.reconcile.assert_called_once_with(recovery)
        emitted = bridge.comicDocumentChanged.values[-1]
        self.assertEqual(emitted["document"]["revision"], 5)
        self.assertEqual(emitted["persistence"]["status"], "recovered")

    def _request_state_with(self, backend, kind="comfyui"):
        bridge = _Bridge()
        bridge.creatorStateChanged = _Signal()
        self.actions.vue_bridge = bridge
        self.actions._creator_running = False
        self.actions._creator_mode = ""
        studio = mock.Mock()
        studio.reconcile.return_value = SimpleNamespace(
            document=_Document(), status="saved", conflict_path=""
        )
        self.actions._comic_studio = lambda: studio
        self.actions._creator_object_info = mock.Mock(
            side_effect=AssertionError("state check must not fetch /object_info")
        )
        with (
            mock.patch("ui.creator_actions.threading.Thread", _ImmediateThread),
            mock.patch("backends.get_backend", return_value=backend),
            mock.patch("backends.get_backend_type") as get_backend_type,
        ):
            get_backend_type.return_value.value = kind
            self.actions._creator_request_state({})
        return bridge.creatorStateChanged.values[-1]

    def test_connected_comfy_state_is_ready_without_fetching_node_catalog(self):
        backend = mock.Mock()
        backend.test_connection.return_value = True
        state = self._request_state_with(backend)
        self.assertEqual(state["status"], "ready")
        self.assertTrue(state["connected"])
        self.assertNotIn("nodeTypes", state)
        self.actions._creator_object_info.assert_not_called()

    def test_state_error_reports_cause_under_error_key(self):
        backend = mock.Mock()
        backend.test_connection.side_effect = RuntimeError("socket closed")
        state = self._request_state_with(backend)
        self.assertEqual(state["status"], "error")
        self.assertEqual(state["error"], "socket closed")


class CreatorOllamaUnloadTests(unittest.TestCase):
    """Ollama 미실행을 '언로드 실패'로 오인하면 Creator 생성이 시작조차 못 한다.

    ResourceCoordinator.reserve 는 unload_llm 이 False 를 돌려주면 하드 abort 하므로,
    "언로드할 게 없음"과 "언로드 실패"를 여기서 구분해야 한다.
    ui/generator_generation.py 의 _maybe_unload_ollama 와 같은 의미론.
    """

    def setUp(self):
        self.actions = CreatorActionsMixin()
        self.actions._creator_ollama_config = lambda: ("http://localhost:11434", "qwen3:8b")
        self.installed = ["qwen3:8b"]   # None = 서버 꺼짐
        self.unload_error = None
        self.unloaded = []

    def _get(self, url, **_kwargs):
        import requests
        if self.installed is None:
            raise requests.ConnectionError("refused")
        return mock.Mock(status_code=200, raise_for_status=lambda: None,
                         json=lambda: {"models": [{"name": name} for name in self.installed]})

    def _post(self, url, json=None, **_kwargs):
        if self.unload_error is not None:
            raise self.unload_error
        self.unloaded.append((url, dict(json or {})))
        return mock.Mock(status_code=200)

    def _unload(self):
        with mock.patch("core.ollama_client.requests.get", side_effect=self._get), \
                mock.patch("core.ollama_client.requests.post", side_effect=self._post):
            return self.actions._creator_unload_ollama()

    def test_unreachable_ollama_counts_as_unloaded(self):
        self.installed = None
        self.assertTrue(self._unload())
        self.assertEqual(self.unloaded, [])

    def test_running_ollama_is_actually_unloaded(self):
        self.assertTrue(self._unload())
        self.assertEqual(self.unloaded, [
            ("http://localhost:11434/api/generate", {"model": "qwen3:8b", "keep_alive": 0}),
        ])

    def test_uninstalled_configured_tag_unloads_the_model_actually_loaded(self):
        # Settings 추천 카드는 pull 안내를 위해 설치 안 된 이름만 저장한다 — 태그 강화·Comic 은
        # resolve_model 로 같은 계열의 설치 모델을 올리므로 언로드도 그 모델이어야 VRAM 이 빈다
        self.actions._creator_ollama_config = lambda: ("http://localhost:11434", "gemma3:4b")
        self.installed = ["llava:7b", "gemma3:12b"]
        self.assertTrue(self._unload())
        self.assertEqual([body["model"] for _url, body in self.unloaded], ["gemma3:12b"])

    def test_real_unload_failure_is_still_reported(self):
        import requests
        self.unload_error = requests.ConnectionError("reset")
        self.assertFalse(self._unload())

    def test_no_model_configured_is_a_noop(self):
        self.actions._creator_ollama_config = lambda: ("http://localhost:11434", "")
        self.assertTrue(self._unload())
        self.assertEqual(self.unloaded, [])


if __name__ == "__main__":
    unittest.main()
