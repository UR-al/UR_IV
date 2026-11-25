"""Regression tests for Worker A core/, backends/, utils/ fixes (review 11-26)
plus baseline unit tests for previously untested core modules (M-05).
"""
from __future__ import annotations

import base64
import io
import json
import logging
import os
import socket
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest import mock

from PIL import Image


def _make_png_b64(width: int = 8, height: int = 6, *, orientation: int | None = None) -> str:
    img = Image.new("RGB", (width, height), (120, 80, 40))
    buf = io.BytesIO()
    if orientation is not None:
        exif = img.getexif()
        exif[274] = orientation
        img.save(buf, format="JPEG", exif=exif)
    else:
        img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


class B11FailClosedGitChecksTests(unittest.TestCase):
    """B-11: safety-critical git status / rev-parse failures must fail closed."""

    def test_prepare_release_fails_closed_when_rev_parse_fails(self):
        from core.backend_runtime import BackendRuntimeError, BackendRuntimeManager, CommandResult
        from tests.test_backend_runtime import FakeRuntimeAdapter

        class FailingRevParseAdapter(FakeRuntimeAdapter):
            def run(self, argv, *, cwd=None, env=None, timeout=600, on_line=None):
                if list(argv)[:3] == ["git", "rev-parse", "HEAD"]:
                    return CommandResult(128, "fatal: not a git repository")
                return super().run(argv, cwd=cwd, env=env, timeout=timeout, on_line=on_line)

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manager = BackendRuntimeManager(
                config_path=root / "runtime.json",
                runtime_root=root / "managed",
                adapter=FailingRevParseAdapter(),
                health_timeout=0.1,
            )
            with self.assertRaises(BackendRuntimeError) as caught:
                manager.execute("forge", "install")
            self.assertEqual(caught.exception.code, "SOURCE_VERIFY_FAILED")

    def test_update_extension_fails_closed_when_git_status_fails(self):
        from core.backend_runtime import BackendRuntimeError, BackendRuntimeManager, CommandResult
        from tests.test_backend_runtime import FakeRuntimeAdapter

        class FailingStatusAdapter(FakeRuntimeAdapter):
            def run(self, argv, *, cwd=None, env=None, timeout=600, on_line=None):
                if list(argv)[:3] == ["git", "status", "--porcelain"]:
                    return CommandResult(128, "fatal: dubious ownership")
                return super().run(argv, cwd=cwd, env=env, timeout=timeout, on_line=on_line)

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            adapter = FailingStatusAdapter()
            manager = BackendRuntimeManager(
                config_path=root / "runtime.json",
                runtime_root=root / "managed",
                adapter=adapter,
                health_timeout=0.1,
            )
            manager.execute("forge", "install")
            ext_dir = manager._extension_root("forge") / "sample-ext"
            (ext_dir / ".git").mkdir(parents=True, exist_ok=True)
            (ext_dir / ".git" / "HEAD").write_text("ref: refs/heads/main\n", encoding="ascii")

            with self.assertRaises(BackendRuntimeError) as caught:
                manager.execute("forge", "update_extension", {"id": "sample-ext"})
            self.assertEqual(caught.exception.code, "EXTENSION_STATUS_FAILED")


class B12OllamaTagFenceTests(unittest.TestCase):
    """B-12: OllamaClient.enhance() tag mode strips fence markers but preserves inner tags."""

    def _enhance_with_content(self, content: str) -> str:
        from core.ollama_client import OllamaClient

        client = OllamaClient(base_url="http://127.0.0.1:11434", model="test-model")
        fake_resp = mock.Mock()
        fake_resp.status_code = 200
        fake_resp.raise_for_status.return_value = None
        fake_resp.json.return_value = {"message": {"content": content}}
        with mock.patch("core.ollama_client.requests.post", return_value=fake_resp):
            return client.enhance("1girl", mode="expand")

    def test_tag_mode_preserves_tags_inside_markdown_code_fence_with_lang_and_newline(self):
        result = self._enhance_with_content("```text\n1girl, solo, blue_eyes, smile\n```")
        self.assertEqual(result, "1girl, solo, blue_eyes, smile")

    def test_tag_mode_preserves_first_tag_when_fence_has_no_newline(self):
        result = self._enhance_with_content("```1girl, blue_hair```")
        self.assertEqual(result, "1girl, blue_hair")


class B16ComfyWebSocketCancelPollTests(unittest.TestCase):
    """B-16: ComfyUIBackend._wait_for_result polls short timeout, resets idle deadline on messages, and cancels on idle timeout."""

    def test_wait_for_result_sets_short_timeout_and_honors_cancel_after_recv_timeout(self):
        from backends import comfyui_backend
        from backends.comfyui_backend import ComfyUIBackend

        backend = ComfyUIBackend("http://127.0.0.1:8188")
        cancel_flag = [False]
        recv_calls = [0]
        timeouts_set: list[float] = []

        class StubWS:
            def settimeout(self, value):
                timeouts_set.append(value)

            def recv(self):
                recv_calls[0] += 1
                cancel_flag[0] = True
                raise socket.timeout("timed out")

        with mock.patch.object(backend, "_cancel_prompt") as cancel_prompt:
            result = backend._wait_for_result(
                StubWS(),
                "prompt-123",
                None,
                cancel_check=lambda: cancel_flag[0],
            )
        self.assertFalse(result.success)
        self.assertEqual(timeouts_set, [comfyui_backend._WS_RECV_POLL_S])
        self.assertEqual(recv_calls[0], 1)
        cancel_prompt.assert_called_once_with("prompt-123")

    def test_wait_for_result_resets_idle_deadline_on_progress_messages(self):
        from backends import comfyui_backend
        from backends.base import GenerationResult
        from backends.comfyui_backend import ComfyUIBackend

        backend = ComfyUIBackend("http://127.0.0.1:8188")
        # 4 progress ticks spaced 250s apart (> 600s total runtime), each below 600s idle limit
        clock = [0.0]
        messages = [
            json.dumps({"type": "progress", "data": {"value": 1, "max": 4, "prompt_id": "prompt-long"}}),
            json.dumps({"type": "progress", "data": {"value": 2, "max": 4, "prompt_id": "prompt-long"}}),
            json.dumps({"type": "progress", "data": {"value": 3, "max": 4, "prompt_id": "prompt-long"}}),
            json.dumps({"type": "progress", "data": {"value": 4, "max": 4, "prompt_id": "prompt-long"}}),
            json.dumps({"type": "executing", "data": {"node": None, "prompt_id": "prompt-long"}}),
        ]

        class LongJobWS:
            def settimeout(self, _value):
                pass

            def recv(self):
                clock[0] += 250.0
                return messages.pop(0)

        expected = GenerationResult(success=True, image_data=b"ok", info={"prompt_id": "prompt-long"})
        with mock.patch.object(comfyui_backend.time, "monotonic", side_effect=lambda: clock[0]), \
                mock.patch.object(backend, "_cancel_prompt") as cancel_prompt, \
                mock.patch.object(backend, "_fetch_result_artifacts", return_value=expected) as fetch:
            result = backend._wait_for_result(LongJobWS(), "prompt-long", None)

        self.assertTrue(result.success)
        self.assertGreater(clock[0], comfyui_backend._WS_IDLE_TIMEOUT_S)
        cancel_prompt.assert_not_called()
        fetch.assert_called_once_with("prompt-long")

    def test_wait_for_result_idle_timeout_cancels_prompt_and_returns_error(self):
        from backends import comfyui_backend
        from backends.comfyui_backend import ComfyUIBackend

        backend = ComfyUIBackend("http://127.0.0.1:8188")
        clock = [0.0]

        class AlwaysTimeoutWS:
            def settimeout(self, _value):
                pass

            def recv(self):
                clock[0] += 350.0
                raise TimeoutError("poll timeout")

        with mock.patch.object(comfyui_backend.time, "monotonic", side_effect=lambda: clock[0]), \
                mock.patch.object(backend, "_cancel_prompt") as cancel_prompt:
            result = backend._wait_for_result(AlwaysTimeoutWS(), "prompt-timeout", None)

        self.assertFalse(result.success)
        self.assertIn("시간 초과", result.error or "")
        cancel_prompt.assert_called_once_with("prompt-timeout")


class B17ComfyWorkflowCompilerTests(unittest.TestCase):
    """B-17: cfg alias fallback, SD3/Flux2/Cosmos empty latent sizing, VAEDecodeTiled recognition."""

    def _compiler(self):
        from core.comfy_workflow_compiler import ComfyWorkflowCompiler
        return ComfyWorkflowCompiler({})

    def test_payload_cfg_alias_fallback(self):
        from core.comfy_workflow_compiler import _payload_cfg

        self.assertEqual(_payload_cfg({"cfg": 4.5}), 4.5)
        self.assertEqual(_payload_cfg({"cfg_scale": None, "cfg": 3.5}), 3.5)
        self.assertEqual(_payload_cfg({"cfg_scale": 6.0, "cfg": 2.0}), 6.0)
        self.assertEqual(_payload_cfg({}), 7.0)

    def test_map_sampler_inputs_uses_cfg_alias(self):
        compiler = self._compiler()
        inputs: dict = {}
        compiler._map_sampler_inputs(
            inputs, "KSampler", {"cfg": 4.25, "steps": 12, "seed": 42}, mode="txt2img"
        )
        self.assertEqual(inputs["cfg"], 4.25)

    def test_apply_custom_latent_size_supports_sd3_flux2_cosmos(self):
        compiler = self._compiler()
        for cls_name in ("EmptySD3LatentImage", "EmptyFlux2LatentImage", "EmptyCosmosLatentVideo"):
            with self.subTest(class_type=cls_name):
                nodes = {
                    "1": {"class_type": cls_name, "inputs": {"width": 512, "height": 512, "batch_size": 1}},
                    "2": {"class_type": "KSampler", "inputs": {"latent_image": ["1", 0]}},
                }
                compiler._apply_custom_latent_size(
                    nodes, nodes["2"]["inputs"], {"width": 1024, "height": 768, "batch_size": 2}, mode="txt2img"
                )
                self.assertEqual(nodes["1"]["inputs"]["width"], 1024)
                self.assertEqual(nodes["1"]["inputs"]["height"], 768)
                self.assertEqual(nodes["1"]["inputs"]["batch_size"], 2)

    def test_find_decode_after_recognizes_vae_decode_tiled(self):
        compiler = self._compiler()
        workflow = {
            "3": {"class_type": "KSampler", "inputs": {}},
            "8": {"class_type": "VAEDecodeTiled", "inputs": {"samples": ["3", 0], "vae": ["1", 2]}},
        }
        self.assertEqual(compiler._find_decode_after(workflow, "3"), "8")

    def test_find_decode_after_prefers_standard_vae_decode_when_both_exist(self):
        compiler = self._compiler()
        workflow = {
            "3": {"class_type": "KSampler", "inputs": {}},
            "8": {"class_type": "VAEDecode", "inputs": {"samples": ["3", 0], "vae": ["1", 2]}},
            "9": {"class_type": "VAEDecodeTiled", "inputs": {"samples": ["3", 0], "vae": ["1", 2]}},
        }
        self.assertEqual(compiler._find_decode_after(workflow, "3"), "8")


class A12ErrorHandlerCp949Tests(unittest.TestCase):
    """A-12: handle_error never raises UnicodeEncodeError on cp949 stdout/stderr."""

    def test_handle_error_handles_em_dash_on_cp949_streams(self):
        from core import error_handler

        raw_out = io.BytesIO()
        raw_err = io.BytesIO()
        cp949_out = io.TextIOWrapper(raw_out, encoding="cp949", errors="strict")
        cp949_err = io.TextIOWrapper(raw_err, encoding="cp949", errors="strict")

        with mock.patch("sys.stdout", cp949_out), mock.patch("sys.stderr", cp949_err):
            try:
                raise RuntimeError("broken — detail \U0001f525")
            except RuntimeError as exc:
                msg = error_handler.handle_error("E030", "설정 — 저장", exc, notify=False)
            cp949_out.flush()
            cp949_err.flush()

        self.assertIn("[E030]", msg)
        self.assertIn("—", msg)
        out_text = raw_out.getvalue().decode("cp949")
        err_text = raw_err.getvalue().decode("cp949")
        self.assertIn("[ERROR E030]", out_text)
        self.assertIn("RuntimeError", err_text)


class B18GenerationApiTests(unittest.TestCase):
    """B-18: disk cleanup on evicted jobs, locked default_target(), stripped init_images in A1111 response."""

    def test_trim_jobs_removes_evicted_directories_from_disk(self):
        from core import generation_api
        from core.generation_api import GenerationApiManager

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            storage = root / "results"
            manager = GenerationApiManager(
                config_path=root / "api.json",
                storage_root=storage,
                target_factory=lambda _t: None,
            )
            self.addCleanup(manager.shutdown)

            evicted_ids = ["a" * 32, "b" * 32, "c" * 32]
            for jid in evicted_ids:
                jdir = storage / jid
                jdir.mkdir(parents=True, exist_ok=True)
                (jdir / "job.json").write_text("{}", encoding="utf-8")
                manager._jobs[jid] = {
                    "id": jid,
                    "state": "completed",
                    "_slotReleased": True,
                }

            with mock.patch.object(generation_api, "MAX_RECENT_JOBS", 1):
                with manager._lock:
                    removed = manager._trim_jobs_locked()
                manager._remove_job_dirs(removed)

            self.assertEqual(removed, ["a" * 32, "b" * 32])
            self.assertFalse((storage / ("a" * 32)).exists())
            self.assertFalse((storage / ("b" * 32)).exists())
            self.assertTrue((storage / ("c" * 32)).exists())

    def test_default_target_helper_returns_configured_target(self):
        from core.generation_api import GenerationApiManager

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manager = GenerationApiManager(
                config_path=root / "api.json",
                storage_root=root / "results",
                target_factory=lambda _t: None,
            )
            self.addCleanup(manager.shutdown)
            self.assertEqual(manager.default_target(), "active")

    def test_a1111_img2img_response_strips_init_images_from_parameters(self):
        import urllib.request
        from backends.base import GenerationResult, MediaArtifact
        from core.generation_api import GenerationApiManager
        from core.resource_coordinator import GenerationResourceCoordinator

        png_bytes = base64.b64decode(_make_png_b64(4, 4))

        class DummyBackend:
            api_url = "http://127.0.0.1:7860"

            def img2img(self, model, payload, progress=None):
                return GenerationResult(
                    success=True,
                    artifacts=[MediaArtifact(kind="image", data=png_bytes, filename="out.png", mime="image/png")],
                    info={"seed": 1},
                )

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manager = GenerationApiManager(
                config_path=root / "api.json",
                storage_root=root / "results",
                target_factory=lambda _t: DummyBackend(),
                coordinator=GenerationResourceCoordinator(),
            )
            self.addCleanup(manager.shutdown)
            cfg = manager.snapshot(True)["config"]
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind(("127.0.0.1", 0))
                free_port = s.getsockname()[1]
            manager.save_config({**cfg, "enabled": True, "port": free_port})
            manager.start(persist_enabled=False)
            token = manager.snapshot(True)["config"]["token"]

            body = json.dumps({
                "prompt": "test",
                "init_images": [_make_png_b64(4, 4)],
                "steps": 5,
            }).encode("utf-8")
            req = urllib.request.Request(
                f"http://127.0.0.1:{free_port}/sdapi/v1/img2img",
                data=body,
                headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=5) as resp:
                data = json.loads(resp.read().decode("utf-8"))
            self.assertEqual(len(data["images"]), 1)
            self.assertNotIn("init_images", data["parameters"])
            self.assertEqual(data["parameters"]["prompt"], "test")


class B19AndM03StudioPrivacyAndPathSanitizeTests(unittest.TestCase):
    """B-19 & M-03: public runtime strips *Args* and extension paths; shared path masking handles POSIX."""

    def test_public_runtime_strips_args_and_extension_path_for_web(self):
        from core.studio_application import CallContext, StudioApplication

        app = StudioApplication()
        web_ctx = CallContext(principal_id="web-user", transport="http", capabilities=frozenset())
        raw = {
            "runtimeRoot": "C:/secret/runtime",
            "engines": {
                "forge": {
                    "extraArgs": "--gradio-auth admin:secret123",
                    "detectedArgs": ["--xformers", "--api-auth", "u:p"],
                    "detectedArgsSource": "C:/secret/webui-user.bat",
                    "detectedArgsDropped": ["--listen"],
                    "extensions": [
                        {"id": "ext1", "name": "ext1", "path": "/home/alice/forge/extensions/ext1", "status": "git"},
                    ],
                    "message": "failed reading /Users/alice/secret.safetensors",
                }
            },
        }
        public = app._public_runtime(raw, web_ctx)
        forge = public["engines"]["forge"]
        for key in ("extraArgs", "detectedArgs", "detectedArgsSource", "detectedArgsDropped"):
            self.assertNotIn(key, forge)
        self.assertTrue(forge["extraArgsConfigured"])
        self.assertNotIn("path", forge["extensions"][0])
        self.assertNotIn("/Users/alice", forge["message"])
        self.assertIn("[local path]", forge["message"])

    def test_shared_path_sanitize_masks_posix_and_preserves_urls_and_api_routes(self):
        from core.path_sanitize import mask_local_paths
        from core.studio_application import StudioApplication

        self.assertEqual(
            StudioApplication._sanitise_external_text("err /home/alice/model.ckpt and /etc/hosts"),
            "err [local path] and [local path]",
        )
        self.assertEqual(
            StudioApplication._sanitise_external_text("GET https://example.com/home/page /api/v1/health"),
            "GET https://example.com/home/page /api/v1/health",
        )
        self.assertEqual(
            mask_local_paths("failed at /opt/app/run.sh", replacement="[path]"),
            "failed at [path]",
        )


class B20BackendAndRequirementsTests(unittest.TestCase):
    """B-20(1, 5): WebUIBackend closes response on 4xx raise_for_status; check_requirements uses requirement.name."""

    def test_webui_generate_once_closes_response_on_4xx(self):
        import requests
        from backends.webui_backend import WebUIBackend

        backend = WebUIBackend("http://127.0.0.1:7860")
        resp = mock.Mock()
        resp.status_code = 403
        resp.raise_for_status.side_effect = requests.HTTPError("403 Forbidden")

        with mock.patch("backends.webui_backend.requests.post", return_value=resp):
            with self.assertRaises(requests.HTTPError):
                backend._generate_once("/sdapi/v1/txt2img", {"prompt": "hi"}, None, None)
        resp.close.assert_called_once()

    def test_check_requirements_handles_extras(self):
        from core import check_requirements

        called_names: list[str] = []

        def fake_version(pkg_name: str) -> str:
            called_names.append(pkg_name)
            return "2.5.0"

        with mock.patch.object(check_requirements, "version", side_effect=fake_version):
            ok = check_requirements._requirement_satisfied("uvicorn[standard]", "uvicorn[standard]>=2.0")
        self.assertTrue(ok)
        self.assertEqual(called_names, ["uvicorn"])


class B21SmallFixesTests(unittest.TestCase):
    """B-21(2, 3, 4, 6): node pack backup rmtree OSError, backend_runtime venv fallback & .git file,
    app_updater corrupt intervalHours, prompt_history save warning."""

    def test_comfy_node_pack_ignores_backup_rmtree_oserror_and_cleans_stale(self):
        from core import comfy_node_pack

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "src"
            source.mkdir()
            (source / "__init__.py").write_text("# v2\n", encoding="utf-8")
            custom_nodes = root / "custom_nodes"
            custom_nodes.mkdir()

            # First install
            res1 = comfy_node_pack.install_bundled_node_pack(custom_nodes, source=source)
            self.assertTrue(res1.changed)

            # Create a stale backup dir and update source so reinstall swaps backup
            stale = custom_nodes / f".{comfy_node_pack.PACK_ID}.backup-stale123"
            stale.mkdir()
            (source / "__init__.py").write_text("# v3\n", encoding="utf-8")

            real_rmtree = comfy_node_pack.shutil.rmtree

            def flaky_rmtree(path, *args, **kwargs):
                if ".backup-" in str(path) and "stale123" not in str(path) and not kwargs.get("ignore_errors"):
                    raise PermissionError("locked on Windows")
                return real_rmtree(path, *args, **kwargs)

            with mock.patch.object(comfy_node_pack.shutil, "rmtree", side_effect=flaky_rmtree):
                res2 = comfy_node_pack.install_bundled_node_pack(custom_nodes, source=source)
            self.assertTrue(res2.changed)
            self.assertFalse(stale.exists())
            self.assertIn("# v3", (res2.target / "__init__.py").read_text(encoding="utf-8"))

    def test_backend_runtime_py_fallback_and_git_file_worktree(self):
        import sys
        from core.backend_runtime import BackendRuntimeManager, CommandResult
        from tests.test_backend_runtime import FakeRuntimeAdapter

        class PyFallbackAdapter(FakeRuntimeAdapter):
            def __init__(self):
                super().__init__()
                self.uv_path = None
                self.venv_attempts: list[list[str]] = []

            def which(self, command: str) -> str | None:
                if command == "py":
                    return "C:/Windows/py.exe"
                return super().which(command)

            def run(self, argv, *, cwd=None, env=None, timeout=600, on_line=None):
                args = [str(a) for a in argv]
                if "-m" in args and "venv" in args:
                    self.venv_attempts.append(args)
                    if args[0].endswith("py.exe"):
                        return CommandResult(1, "Python 3.13 not found")
                    self._make_venv(Path(args[-1]))
                    return CommandResult(0, "ok")
                return super().run(argv, cwd=cwd, env=env, timeout=timeout, on_line=on_line)

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            adapter = PyFallbackAdapter()
            manager = BackendRuntimeManager(
                config_path=root / "runtime.json",
                runtime_root=root / "managed",
                adapter=adapter,
                health_timeout=0.1,
            )
            manager.execute("forge", "install")
            self.assertEqual(len(adapter.venv_attempts), 2)
            self.assertTrue(adapter.venv_attempts[0][0].endswith("py.exe"))
            self.assertEqual(adapter.venv_attempts[1][0], sys.executable)

            # Verify .git file (worktree/submodule) is recognized as git extension
            ext_root = manager._extension_root("forge")
            real_git = root / "worktree_git"
            real_git.mkdir()
            (real_git / "HEAD").write_text("ref: refs/heads/main\n", encoding="ascii")
            wt_ext = ext_root / "wt-ext"
            wt_ext.mkdir(parents=True, exist_ok=True)
            (wt_ext / ".git").write_text(f"gitdir: {real_git}\n", encoding="utf-8")
            ext_items = {item["id"]: item for item in manager._extension_state("forge")}
            self.assertEqual(ext_items["wt-ext"]["status"], "git")

    def test_app_updater_handles_corrupt_interval_hours(self):
        from core.app_updater import AppUpdateManager, DEFAULT_INTERVAL_HOURS

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            settings_path = root / "app_update.json"
            for bad_value in ("not-a-number", True, False, None, [], {}):
                with self.subTest(bad_value=bad_value):
                    settings_path.write_text(
                        json.dumps({"intervalHours": bad_value}), encoding="utf-8"
                    )
                    mgr = AppUpdateManager(
                        project_root=root,
                        settings_path=settings_path,
                        result_path=root / "result.json",
                    )
                    self.assertEqual(mgr._load_settings()["intervalHours"], DEFAULT_INTERVAL_HOURS)

    def test_prompt_history_logs_warning_on_save_failure(self):
        from utils import prompt_history

        with mock.patch.object(prompt_history, "atomic_write_json", side_effect=OSError("disk full")):
            with self.assertLogs("utils.prompt_history", level=logging.WARNING) as cm:
                prompt_history._save([{"prompt": "1girl", "negative": ""}])
        self.assertTrue(any("disk full" in line for line in cm.output))


class P07PerformanceFixesTests(unittest.TestCase):
    """P-07(2, 3): _forge_init_size strips data URI & reads EXIF orientation without exif_transpose;
    TagClassifier precomputes _normalized_colors."""

    def test_forge_init_size_strips_data_uri_and_swaps_exif_orientation(self):
        from backends.webui_backend import _forge_init_size

        raw_normal = _make_png_b64(12, 8, orientation=1)
        raw_rot90 = _make_png_b64(12, 8, orientation=6)
        data_uri_rot90 = f"data:image/jpeg;base64,{raw_rot90}"

        with mock.patch("backends.webui_backend.ImageOps.exif_transpose", side_effect=AssertionError("must not decode full image")):
            self.assertEqual(_forge_init_size(raw_normal), (12, 8))
            self.assertEqual(_forge_init_size(raw_rot90), (8, 12))
            self.assertEqual(_forge_init_size(data_uri_rot90), (8, 12))

    def test_tag_classifier_caches_normalized_colors(self):
        from core.tag_classifier import TagClassifier

        fake_db = mock.Mock()
        fake_db.path.return_value = Path("dummy/tag_groups.parquet")
        fake_db.read_json.return_value = {}
        fake_db.read_parquet.side_effect = FileNotFoundError()
        fake_db.read_lines.side_effect = lambda asset: ["crimson_red"] if "color" in asset.value else []
        fake_db.load_tag_groups.return_value = {}
        fake_db.load_active_implications.return_value = {}

        clf = TagClassifier(database=fake_db)
        self.assertIn("crimson red", clf._normalized_colors)
        self.assertEqual(clf.classify_tag("crimson_red"), "general")  # multi-word color check
        clf._normalized_colors.add("scarlet")
        self.assertEqual(clf.classify_tag("scarlet"), "color")


class M05UntestedCoreModulesTests(unittest.TestCase):
    """M-05: baseline unit tests for core/output_files.py, core/comfy_node_classes.py, core/app_context.py."""

    def test_output_files_same_path_and_write_new_file_never_overwrites(self):
        from core.output_files import same_file_path, write_new_file

        with tempfile.TemporaryDirectory() as tmp:
            base = os.path.join(tmp, "out.png")
            self.assertTrue(same_file_path(base, os.path.join(tmp, ".", "out.png")))
            self.assertFalse(same_file_path(base, os.path.join(tmp, "other.png")))

            p1 = write_new_file(base, b"first")
            p2 = write_new_file(base, b"second")
            self.assertNotEqual(p1, p2)
            self.assertEqual(Path(p1).read_bytes(), b"first")
            self.assertEqual(Path(p2).read_bytes(), b"second")

    def test_comfy_node_classes_vocabularies_are_consistent(self):
        from core.comfy_node_classes import (
            CHECKPOINT_LOADER_NODES,
            CUSTOM_SAMPLER_NODES,
            IMAGE_SAVE_NODES,
            MODEL_LOADER_INPUTS,
            SAMPLER_NODES,
            SEMANTIC_ENCODER_NODES,
            TEXT_ENCODER_NODES,
            UNET_LOADER_NODES,
        )

        self.assertTrue(CUSTOM_SAMPLER_NODES.issubset(SAMPLER_NODES))
        self.assertTrue(SEMANTIC_ENCODER_NODES.issubset(TEXT_ENCODER_NODES))
        self.assertTrue(CHECKPOINT_LOADER_NODES.issubset(MODEL_LOADER_INPUTS.keys()))
        self.assertTrue(UNET_LOADER_NODES.issubset(MODEL_LOADER_INPUTS.keys()))
        self.assertIn("SaveImage", IMAGE_SAVE_NODES)

    def test_app_context_subscribe_publish_unsubscribe_and_isolation(self):
        from core.app_context import AppContext, Events

        ctx = AppContext()
        received: list[str] = []

        def bad_subscriber(_data):
            raise RuntimeError("boom")

        h1 = ctx.subscribe(Events.BACKEND_CHANGED, bad_subscriber)
        h2 = ctx.subscribe(Events.BACKEND_CHANGED, received.append)

        with self.assertLogs("app_context", level=logging.ERROR):
            delivered = ctx.publish(Events.BACKEND_CHANGED, "comfyui")
        self.assertEqual(delivered, 1)
        self.assertEqual(received, ["comfyui"])

        self.assertTrue(ctx.unsubscribe(Events.BACKEND_CHANGED, h1))
        self.assertFalse(ctx.unsubscribe(Events.BACKEND_CHANGED, h1))
        self.assertTrue(ctx.unsubscribe(Events.BACKEND_CHANGED, h2))
        self.assertEqual(ctx.publish(Events.BACKEND_CHANGED, "webui"), 0)

        with self.assertRaises(TypeError):
            ctx.subscribe(Events.BACKEND_CHANGED, "not-callable")  # type: ignore[arg-type]


if __name__ == "__main__":
    unittest.main()
