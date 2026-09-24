from __future__ import annotations

import hashlib
import os
import tempfile
import threading
import unittest
from pathlib import Path

from core.model_downloads import ModelArtifact, ModelDownloadError, ModelDownloadManager, ModelPack
from core.storage_paths import StoragePaths


DATA = b"tiny verified model fixture"


class FakeResponse:
    def __init__(self, data=DATA, status=200, headers=None):
        self.status_code = status
        self.headers = headers or {"Content-Length": str(len(data))}
        self.data = data

    def iter_content(self, chunk_size):
        yield self.data

    def close(self):
        pass


class FakeHTTP:
    def __init__(self, responses=None):
        self.responses = list([FakeResponse()] if responses is None else responses)
        self.requests = []

    def get(self, url, **kwargs):
        self.requests.append((url, kwargs))
        if not self.responses:
            raise AssertionError("Unexpected network request")
        return self.responses.pop(0)


class ModelDownloadTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.artifact = ModelArtifact(
            id="test-model", label="테스트 모델", category="vae",
            filename="tiny.safetensors", size=len(DATA),
            sha256=hashlib.sha256(DATA).hexdigest(),
            url="https://huggingface.co/test/model/resolve/pinned/tiny.safetensors",
        )
        self.pack = ModelPack("test-pack", "테스트", "작은 테스트 파일", ("test-model",))

    def manager(self, http=None, **kwargs):
        kwargs.setdefault("snapshot_provider", lambda: {})
        manager = ModelDownloadManager(
            artifacts=[self.artifact], packs=[self.pack],
            storage=StoragePaths(self.root), model_root=self.root / "models",
            http=http or FakeHTTP(), **kwargs,
        )
        self.addCleanup(manager.shutdown)
        return manager

    def test_download_publishes_verified_file_and_persists_receipt(self):
        manager = self.manager()
        manager.start(["test-pack"])
        result = manager.wait(3)
        self.assertEqual(result["state"], "complete", result)
        self.assertEqual(result["files"][0]["status"], "verified")
        self.assertTrue(result["packs"][0]["ready"])
        self.assertEqual(Path(result["files"][0]["path"]).read_bytes(), DATA)
        restored = self.manager(http=FakeHTTP([]))
        self.assertEqual(restored.status()["files"][0]["status"], "verified")

    def test_shutdown_never_rescans_model_libraries_on_the_ui_thread(self):
        def forbidden_scan():
            raise AssertionError("shutdown must not scan model libraries")
        manager = self.manager(snapshot_provider=forbidden_scan)
        self.assertFalse(manager.shutdown(timeout=0)["busy"])

    def test_reuses_existing_nested_shared_model_without_a_download(self):
        primary = self.root / "forge" / "VAE"
        secondary = self.root / "comfy" / "vae"
        existing = secondary / "shared" / "tiny.safetensors"
        existing.parent.mkdir(parents=True)
        existing.write_bytes(DATA)
        snapshot = {"primaryModelEngine": "forge", "engines": {
            "forge": {"modelPaths": {"vae": [str(primary)]}},
            "comfyui": {"modelPaths": {"vae": [str(secondary)]}},
        }}
        http = FakeHTTP([])
        manager = self.manager(http=http, snapshot_provider=lambda: snapshot)
        self.assertEqual(manager.status()["files"][0]["path"], str(existing))
        manager.start(["test-pack"])
        result = manager.wait(3)
        self.assertEqual(result["state"], "complete", result)
        self.assertEqual(http.requests, [])
        self.assertFalse((primary / "tiny.safetensors").exists())

    def test_resumes_partial_only_after_matching_content_range(self):
        http = FakeHTTP([FakeResponse(DATA[5:], 206, {
            "Content-Range": f"bytes 5-{len(DATA)-1}/{len(DATA)}",
            "Content-Length": str(len(DATA)-5),
        })])
        manager = self.manager(http=http)
        destination = Path(manager.status()["files"][0]["path"])
        destination.parent.mkdir(parents=True)
        destination.with_name(destination.name + ".part").write_bytes(DATA[:5])
        manager.start(["test-pack"])
        result = manager.wait(3)
        self.assertEqual(result["state"], "complete", result)
        self.assertEqual(http.requests[0][1]["headers"]["Range"], "bytes=5-")
        self.assertEqual(destination.read_bytes(), DATA)

    def test_verify_never_downloads_missing_files(self):
        http = FakeHTTP([])
        manager = self.manager(http=http)
        manager.verify(["test-pack"])
        result = manager.wait(3)
        self.assertEqual(result["state"], "error")
        self.assertEqual(http.requests, [])

    def test_disk_space_preflight_happens_before_network(self):
        from types import SimpleNamespace
        http = FakeHTTP([])
        manager = self.manager(http=http, disk_usage=lambda _path: SimpleNamespace(free=0))
        manager.start(["test-pack"])
        result = manager.wait(3)
        self.assertEqual(result["state"], "error")
        self.assertEqual(http.requests, [])
        self.assertIn("공간", result["error"])

    def test_turbo_grid_requires_real_node_folder_and_uses_its_data_location(self):
        from dataclasses import replace
        self.artifact = replace(self.artifact, id="h3-temb-grid", category="auxiliary", filename="h3_silu_temb_grid.safetensors")
        self.pack = ModelPack("test-pack", "Turbo", "grid", (self.artifact.id,))
        node = self.root / "ComfyUI" / "custom_nodes" / "ComfyUI-MiniMax-H3-Turbo"
        snapshot = {"engines": {"comfyui": {"sourceRoot": str(node.parent.parent)}}}
        http = FakeHTTP()
        manager = self.manager(http=http, snapshot_provider=lambda: snapshot)
        self.assertFalse(manager.status()["packs"][0]["downloadable"])
        with self.assertRaisesRegex(ModelDownloadError, "노드"):
            manager.start(["test-pack"])
        self.assertEqual(http.requests, [])
        node.mkdir(parents=True)
        self.assertEqual(manager.status()["files"][0]["path"], str(node / self.artifact.filename))
        manager.start(["test-pack"])
        self.assertEqual(manager.wait(3)["state"], "complete")
        self.assertEqual((node / self.artifact.filename).read_bytes(), DATA)

    def test_bad_sha_partial_is_quarantined_so_retry_can_succeed(self):
        http = FakeHTTP([FakeResponse(b"x" * len(DATA)), FakeResponse()])
        manager = self.manager(http=http)
        manager.start(["test-pack"])
        first = manager.wait(3)
        target = Path(first["files"][0]["path"])
        self.assertEqual(first["state"], "error")
        self.assertFalse(target.exists())
        manager.start(["test-pack"])
        self.assertEqual(manager.wait(3)["state"], "complete")
        self.assertEqual(target.read_bytes(), DATA)
        self.assertEqual(len(list(target.parent.glob("*.invalid-*"))), 1)

    def test_wrong_content_range_preserves_partial_and_does_not_publish(self):
        http = FakeHTTP([FakeResponse(DATA[5:], 206, {
            "Content-Range": f"bytes 6-{len(DATA)-1}/{len(DATA)}",
            "Content-Length": str(len(DATA)-5),
        })])
        manager = self.manager(http=http)
        target = Path(manager.status()["files"][0]["path"])
        target.parent.mkdir(parents=True)
        partial = target.with_name(target.name + ".part")
        partial.write_bytes(DATA[:5])
        manager.start(["test-pack"])
        self.assertEqual(manager.wait(3)["state"], "error")
        self.assertEqual(partial.read_bytes(), DATA[:5])
        self.assertFalse(target.exists())

    def test_server_ignoring_range_restarts_temporary_file_not_appends(self):
        manager = self.manager()
        target = Path(manager.status()["files"][0]["path"])
        target.parent.mkdir(parents=True)
        target.with_name(target.name + ".part").write_bytes(DATA[:5])
        manager.start(["test-pack"])
        self.assertEqual(manager.wait(3)["state"], "complete")
        self.assertEqual(target.read_bytes(), DATA)

    def test_cancel_keeps_job_ownership_until_worker_settles_then_resumes(self):
        entered, release = threading.Event(), threading.Event()

        class BlockingResponse(FakeResponse):
            def iter_content(self, chunk_size):
                yield DATA[:5]
                entered.set()
                if not release.wait(3):
                    raise AssertionError("Test did not release stream")
                yield DATA[5:]

        http = FakeHTTP([BlockingResponse(), FakeResponse(DATA[5:], 206, {
            "Content-Range": f"bytes 5-{len(DATA)-1}/{len(DATA)}", "Content-Length": str(len(DATA)-5),
        })])
        events = []
        manager = self.manager(http=http, on_event=events.append)
        first = manager.start(["test-pack"])
        self.assertTrue(entered.wait(3))
        try:
            state = manager.cancel(first["jobId"])
            self.assertTrue(state["busy"])
            self.assertEqual(state["state"], "canceling")
            with self.assertRaises(ModelDownloadError):
                manager.start(["test-pack"])
        finally:
            release.set()
        self.assertEqual(manager.wait(3)["state"], "canceled")
        second = manager.start(["test-pack"])
        self.assertNotEqual(first["jobId"], second["jobId"])
        with self.assertRaises(ModelDownloadError):
            manager.cancel(first["jobId"])
        self.assertEqual(manager.wait(3)["state"], "complete")
        self.assertEqual([event["revision"] for event in events], sorted(event["revision"] for event in events))

    def test_existing_bad_completed_file_is_never_overwritten(self):
        http = FakeHTTP([])
        manager = self.manager(http=http)
        target = Path(manager.status()["files"][0]["path"])
        target.parent.mkdir(parents=True)
        target.write_bytes(b"x" * len(DATA))
        manager.start(["test-pack"])
        self.assertEqual(manager.wait(3)["state"], "error")
        self.assertEqual(target.read_bytes(), b"x" * len(DATA))
        self.assertEqual(http.requests, [])

    def test_external_file_created_during_download_is_preserved(self):
        target = None

        class RacingResponse(FakeResponse):
            def iter_content(self, chunk_size):
                target.write_bytes(b"external good file")
                yield DATA

        manager = self.manager(http=FakeHTTP([RacingResponse()]))
        target = Path(manager.status()["files"][0]["path"])
        manager.start(["test-pack"])
        self.assertEqual(manager.wait(3)["state"], "error")
        self.assertEqual(target.read_bytes(), b"external good file")

    def test_disallowed_redirect_never_contacts_external_host(self):
        http = FakeHTTP([FakeResponse(b"", 302, {"Location": "http://127.0.0.1/secret"})])
        manager = self.manager(http=http)
        manager.start(["test-pack"])
        self.assertEqual(manager.wait(3)["state"], "error")
        self.assertEqual(len(http.requests), 1)

    def test_catalog_rejects_path_traversal_and_untrusted_hosts(self):
        from dataclasses import replace
        for filename in ("../outside", "C:/secret", "/absolute", "a/../../secret"):
            with self.assertRaises(ValueError):
                replace(self.artifact, filename=filename)
        for url in ("https://evil.huggingface.co.evil.org/model", "http://huggingface.co/model", "https://user:pass@huggingface.co/model"):
            with self.assertRaises(ValueError):
                replace(self.artifact, url=url)

    def test_shared_dependencies_download_once_and_unknown_ids_are_rejected(self):
        http = FakeHTTP()
        manager = self.manager(http=http)
        manager.packs["second-pack"] = ModelPack("second-pack", "Second", "shared", (self.artifact.id,))
        with self.assertRaises(ModelDownloadError):
            manager.start(["https://example.com/arbitrary-model"])
        manager.start(["test-pack", "second-pack"])
        self.assertEqual(manager.wait(3)["state"], "complete")
        self.assertEqual(len(http.requests), 1)

    def test_changed_receipt_signature_is_not_trusted(self):
        manager = self.manager()
        manager.start(["test-pack"])
        result = manager.wait(3)
        target = Path(result["files"][0]["path"])
        before = target.stat()
        target.write_bytes(b"x" * len(DATA))
        # 같은 크기로 곧바로 다시 쓰면 NTFS 타임스탬프 해상도 안에서 mtime 이 그대로일 수
        # 있다(간헐 실패의 원인). 실제 교체처럼 수정 시각이 달라진 상황을 명시적으로 만든다.
        os.utime(target, ns=(before.st_atime_ns, before.st_mtime_ns + 2_000_000_000))
        self.assertEqual(manager.status()["files"][0]["status"], "present")
        manager.verify(["test-pack"])
        self.assertEqual(manager.wait(3)["state"], "error")

    def test_packaged_catalog_has_pinned_sources_complete_dependencies_and_no_separate_connector(self):
        from core.model_downloads import _catalog
        artifacts, packs = _catalog()
        self.assertEqual(len(artifacts), 18)
        self.assertEqual(len(packs), 11)
        ids = {artifact.id for artifact in artifacts}
        for artifact in artifacts:
            self.assertNotIn("/resolve/main/", artifact.url)
            self.assertNotIn("/Turbo/main/", artifact.url)
        for pack in packs:
            self.assertTrue(set(pack.artifact_ids) <= ids)
        anima = next(pack for pack in packs if pack.id == "anima-3.8b")
        self.assertIn("anima-38-semantic-text", anima.artifact_ids)
        self.assertFalse(any("connector" in artifact.id for artifact in artifacts))

    def test_forge_checkpoint_folder_reuses_diffusion_artifact(self):
        from dataclasses import replace
        self.artifact = replace(self.artifact, category="diffusion_models")
        existing = self.root / "forge" / "Stable-diffusion" / self.artifact.filename
        existing.parent.mkdir(parents=True)
        existing.write_bytes(DATA)
        snapshot = {"primaryModelEngine": "forge", "engines": {"forge": {"modelPaths": {"checkpoints": [str(existing.parent)]}}}}
        http = FakeHTTP([])
        manager = self.manager(http=http, snapshot_provider=lambda: snapshot)
        manager.start(["test-pack"])
        self.assertEqual(manager.wait(3)["state"], "complete")
        self.assertEqual(http.requests, [])
        self.assertEqual(manager.status()["files"][0]["path"], str(existing))

    def test_truncated_stream_keeps_valid_prefix_for_next_resume(self):
        http = FakeHTTP([
            FakeResponse(DATA[:5], headers={"Content-Length": str(len(DATA))}),
            FakeResponse(DATA[5:], 206, {"Content-Range": f"bytes 5-{len(DATA)-1}/{len(DATA)}", "Content-Length": str(len(DATA)-5)}),
        ])
        manager = self.manager(http=http)
        manager.start(["test-pack"])
        self.assertEqual(manager.wait(3)["state"], "error")
        manager.start(["test-pack"])
        self.assertEqual(manager.wait(3)["state"], "complete")
        self.assertEqual(http.requests[1][1]["headers"]["Range"], "bytes=5-")


if __name__ == "__main__":
    unittest.main()
