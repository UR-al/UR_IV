from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from backends.base import GenerationResult, MediaArtifact
from backends.comfyui_backend import ComfyUIBackend
from core.comfy_node_pack import (
    ComfyNodePackError,
    NodePackInstallResult,
    OWNER_ID,
    OWNER_MARKER,
    PACK_VERSION,
    install_bundled_node_pack,
    node_pack_fingerprint,
)
from core.comfy_compatibility import bundle_fingerprint
import comfy_custom_nodes.ai_studio_forge_parity as bundled_pack
from comfy_custom_nodes.ai_studio_forge_parity import NODE_CLASS_MAPPINGS


# 손으로 적은 스냅숏이다. 노드가 실수로 추가/삭제되면 여기서 걸린다 —
# NODE_CLASS_MAPPINGS 자기 자신과 비교하면 가드가 사라지므로 목록을 유지한다.
# 앱 컴파일러(core/comfy_workflow_compiler.py, core/creator_workflows.py)가
# 만드는 노드와 사용자 ComfyUI 워크플로 전용 노드를 나눠 적는다.
COMPILER_BUILT_NODES = frozenset({
    "ForgeNeoAnimaQwen35Loader",
    "ForgeNeoAnimaQwen35Prompt",
    "ForgeNeoAnima38V2Loader",
    "ForgeNeoAnima38V2Prompt",
    "ForgeNeoAnimaLoraLoader",
    "ForgeNeoModelSamplingShift",
    "ForgeNeoNegPip",
    "ForgeNeoSkimmedCFG",
    "ForgeNeoAnimaGuidanceSuite",
    "ForgeNeoAnimaDetailDaemon",
    "ForgeNeoKSamplerCNS",
    "ForgeNeoLatentInput",
    "ForgeNeoHiresFix",
    "ForgeNeoSAM3Mask",
    "ForgeNeoSAM3Detailer",
    "ForgeNeoSAM3Refine",
    "ForgeNeoADetailer",
    "ForgeNeoH3ConditioningCachePrepare",
    "ForgeNeoH3ConditioningCacheLoad",
})
CUSTOM_WORKFLOW_NODES = frozenset({
    "ForgeNeoAnimaLoraLoaderModelOnly",
    "ForgeNeoAnimaDAVE",
    "ForgeNeoAnimaModGuidance",
    "ForgeNeoAnimaSafePAG",
    "ForgeNeoDCWCWMSMC",
    "ForgeNeoMaskSelector",
    "ForgeNeoLoraBlockWeight",
    "ForgeNeoCharacterReference",
    "ForgeNeoReferencePrompt",
    "ForgeNeoReferenceOutput",
    "ForgeNeoAnimaPiD",
    "ForgeNeoAnimaVAE2x",
    "ForgeNeoSAM3TileRepair",
    "ForgeNeoSaveImage",
    "AIStudioRelight",
})
EXPECTED_BUNDLED_NODES = COMPILER_BUILT_NODES | CUSTOM_WORKFLOW_NODES


def _source(root: Path, body: str = "VALUE = 1\n") -> Path:
    source = root / "source"
    source.mkdir()
    (source / "__init__.py").write_text(body, encoding="utf-8")
    return source


class TestBundledComfyNodePack(unittest.TestCase):
    def test_install_is_owned_atomic_and_idempotent(self):
        with tempfile.TemporaryDirectory() as directory:
            tmp_path = Path(directory)
            custom_nodes = tmp_path / "custom_nodes"
            custom_nodes.mkdir()
            source = _source(tmp_path)

            first = install_bundled_node_pack(custom_nodes, source=source)
            second = install_bundled_node_pack(custom_nodes, source=source)

            self.assertTrue(first.changed)
            self.assertFalse(second.changed)
            marker = json.loads((first.target / OWNER_MARKER).read_text(encoding="utf-8"))
            self.assertEqual(marker["owner"], OWNER_ID)
            self.assertEqual(marker["fingerprint"], node_pack_fingerprint(source))

    def test_install_refreshes_only_an_app_owned_target(self):
        with tempfile.TemporaryDirectory() as directory:
            tmp_path = Path(directory)
            custom_nodes = tmp_path / "custom_nodes"
            custom_nodes.mkdir()
            source = _source(tmp_path)
            installed = install_bundled_node_pack(custom_nodes, source=source)
            (source / "__init__.py").write_text("VALUE = 2\n", encoding="utf-8")

            refreshed = install_bundled_node_pack(custom_nodes, source=source)

            self.assertTrue(refreshed.changed)
            self.assertEqual(
                (installed.target / "__init__.py").read_text(encoding="utf-8"),
                "VALUE = 2\n",
            )

    def test_install_repairs_tampered_owned_target_even_when_marker_is_current(self):
        with tempfile.TemporaryDirectory() as directory:
            tmp_path = Path(directory)
            custom_nodes = tmp_path / "custom_nodes"
            custom_nodes.mkdir()
            source = _source(tmp_path)
            installed = install_bundled_node_pack(custom_nodes, source=source)
            (installed.target / "__init__.py").write_text("VALUE = 99\n", encoding="utf-8")

            repaired = install_bundled_node_pack(custom_nodes, source=source)

            self.assertTrue(repaired.changed)
            self.assertEqual(
                (installed.target / "__init__.py").read_text(encoding="utf-8"),
                "VALUE = 1\n",
            )

    def test_install_refuses_unowned_collision(self):
        with tempfile.TemporaryDirectory() as directory:
            tmp_path = Path(directory)
            custom_nodes = tmp_path / "custom_nodes"
            target = custom_nodes / "ai_studio_forge_parity"
            target.mkdir(parents=True)
            (target / "user.py").write_text("# mine\n", encoding="utf-8")
            source = _source(tmp_path)

            with self.assertRaisesRegex(ComfyNodePackError, "덮어쓰지"):
                install_bundled_node_pack(custom_nodes, source=source)

            self.assertTrue((target / "user.py").is_file())

    def test_expected_snapshot_matches_every_exported_bundled_node(self):
        self.assertFalse(COMPILER_BUILT_NODES & CUSTOM_WORKFLOW_NODES)
        self.assertEqual(len(EXPECTED_BUNDLED_NODES), 34)
        self.assertEqual(EXPECTED_BUNDLED_NODES, frozenset(NODE_CLASS_MAPPINGS))

    def test_compiler_node_split_matches_app_sources(self):
        # README 의 '컴파일러 사용 노드' / '커스텀 워크플로용 노드' 구분이
        # 실제 앱 코드와 어긋나지 않게, 그래프를 만드는 모듈에서 따옴표로 둘러싼
        # 노드 ID 가 나오는지 정적으로 확인한다.
        root = Path(__file__).resolve().parent.parent
        sources = "\n".join(
            (root / relative).read_text(encoding="utf-8")
            for relative in (
                "core/comfy_workflow_compiler.py",
                "core/creator_workflows.py",
            )
        )
        for name in sorted(COMPILER_BUILT_NODES):
            with self.subTest(node=name):
                self.assertIn(f'"{name}"', sources)
        readme = (root / "comfy_custom_nodes/ai_studio_forge_parity/README.md").read_text(
            encoding="utf-8"
        )
        compiler_section = readme.split("### Built by the app compiler", 1)[1].split("###", 1)[0]
        custom_section = readme.split("### Custom ComfyUI workflows only", 1)[1].split("##", 1)[0]
        for name in sorted(COMPILER_BUILT_NODES):
            with self.subTest(readme_compiler=name):
                self.assertIn(f"`{name}`", compiler_section)
        for name in sorted(CUSTOM_WORKFLOW_NODES):
            with self.subTest(readme_custom=name):
                self.assertIn(f"`{name}`", custom_section)

    def test_pack_version_matches_package_version(self):
        self.assertEqual(PACK_VERSION, bundled_pack.__version__)

    def test_doc_only_refresh_copies_files_without_requiring_restart(self):
        with tempfile.TemporaryDirectory() as directory:
            tmp_path = Path(directory)
            custom_nodes = tmp_path / "custom_nodes"
            custom_nodes.mkdir()
            source = _source(tmp_path)
            (source / "README.md").write_text("v1\n", encoding="utf-8")
            first = install_bundled_node_pack(custom_nodes, source=source)
            self.assertTrue(first.changed)
            self.assertTrue(first.restart_required)

            (source / "README.md").write_text("v2 docs only\n", encoding="utf-8")
            docs = install_bundled_node_pack(custom_nodes, source=source)
            self.assertTrue(docs.changed)
            self.assertFalse(docs.restart_required)
            self.assertEqual(
                (docs.target / "README.md").read_text(encoding="utf-8"), "v2 docs only\n",
            )

            (source / "data.json").write_text("{}\n", encoding="utf-8")
            runtime = install_bundled_node_pack(custom_nodes, source=source)
            self.assertTrue(runtime.changed)
            self.assertTrue(runtime.restart_required)

            unchanged = install_bundled_node_pack(custom_nodes, source=source)
            self.assertFalse(unchanged.changed)
            self.assertFalse(unchanged.restart_required)

    def test_tampered_runtime_file_requires_restart_after_repair(self):
        with tempfile.TemporaryDirectory() as directory:
            tmp_path = Path(directory)
            custom_nodes = tmp_path / "custom_nodes"
            custom_nodes.mkdir()
            source = _source(tmp_path)
            installed = install_bundled_node_pack(custom_nodes, source=source)
            (installed.target / "__init__.py").write_text("VALUE = 99\n", encoding="utf-8")
            (source / "README.md").write_text("docs\n", encoding="utf-8")

            repaired = install_bundled_node_pack(custom_nodes, source=source)

            self.assertTrue(repaired.changed)
            self.assertTrue(repaired.restart_required)

    def test_compatibility_screen_uses_the_installer_fingerprint(self):
        with tempfile.TemporaryDirectory() as directory:
            tmp_path = Path(directory)
            custom_nodes = tmp_path / "custom_nodes"
            custom_nodes.mkdir()
            source = _source(tmp_path, '__version__ = "9.9.9"\n')
            (source / "vendor").mkdir()
            (source / "vendor" / "tokenizer.json").write_text("{}\n", encoding="utf-8")
            installed = install_bundled_node_pack(custom_nodes, source=source)

            source_report = bundle_fingerprint(source)
            self.assertEqual(source_report["fingerprint"], node_pack_fingerprint(source))
            self.assertEqual(source_report["version"], "9.9.9")
            self.assertEqual(source_report["fileCount"], 2)
            self.assertEqual(
                bundle_fingerprint(installed.target)["fingerprint"],
                source_report["fingerprint"],
            )
            # A non-.py runtime file drift is visible to both, not only the installer.
            (installed.target / "vendor" / "tokenizer.json").write_text("[]\n", encoding="utf-8")
            self.assertNotEqual(
                bundle_fingerprint(installed.target)["fingerprint"],
                source_report["fingerprint"],
            )
            self.assertEqual(
                bundle_fingerprint(tmp_path / "missing")["status"], "unknown",
            )


class _FakeRuntimeManager:
    def __init__(self, engine):
        self.engine = dict(engine)
        self.operations = []

    def snapshot(self):
        return {"engines": {"comfyui": dict(self.engine)}}

    def execute(self, engine, action, payload=None):
        self.operations.append((engine, action, payload))
        if action == "start":
            return {"apiUrl": "http://127.0.0.1:18189"}
        return {"ok": True}


class TestComfyBackendNodePackPreflight(unittest.TestCase):
    def _engine(self, **overrides):
        engine = {
            "apiUrl": "http://127.0.0.1:8188",
            "extensionDir": r"C:\ComfyUI\custom_nodes",
            "extensionWritable": True,
            "owned": False,
            "running": True,
        }
        engine.update(overrides)
        return engine

    def test_unapproved_external_extension_directory_is_never_written(self):
        manager = _FakeRuntimeManager(self._engine(extensionWritable=False))
        backend = ComfyUIBackend("http://localhost:8188")
        with mock.patch(
            "core.backend_runtime.get_backend_runtime_manager", return_value=manager,
        ), mock.patch("core.comfy_node_pack.install_bundled_node_pack") as install:
            backend._preflight_bundled_node_pack()

        install.assert_not_called()
        self.assertTrue(backend._node_pack_preflight_done)
        self.assertEqual(manager.operations, [])

    def test_changed_pack_restarts_only_manager_owned_comfy(self):
        manager = _FakeRuntimeManager(self._engine(owned=True))
        backend = ComfyUIBackend("http://127.0.0.1:8188")
        # 감사 #184: the compile snapshot predates the new node pack.
        backend._object_info_cache.store(backend.api_url, {"KSampler": {}})
        installed = NodePackInstallResult(
            target=Path(r"C:\ComfyUI\custom_nodes\ai_studio_forge_parity"),
            fingerprint="abc",
            changed=True,
        )
        with mock.patch(
            "core.backend_runtime.get_backend_runtime_manager", return_value=manager,
        ), mock.patch(
            "core.comfy_node_pack.install_bundled_node_pack", return_value=installed,
        ):
            backend._preflight_bundled_node_pack()

        self.assertEqual(
            manager.operations,
            [
                ("comfyui", "stop", None),
                ("comfyui", "start", {"installIfMissing": False}),
            ],
        )
        self.assertEqual(backend.api_url, "http://127.0.0.1:18189")
        self.assertTrue(backend._node_pack_preflight_done)
        self.assertIsNone(backend._object_info_cache.peek("http://127.0.0.1:8188"))
        self.assertIsNone(backend._object_info_cache.peek(backend.api_url))

    def test_managed_restart_drops_a_snapshot_taken_while_it_restarted(self):
        # A live fetch (connect/XYZ/inspector) racing the restart can re-seed
        # the pre-restart schema; a restart on the same URL must drop it again.
        backend = ComfyUIBackend("http://127.0.0.1:8188")
        stale = {"KSampler": {}}

        class _SameUrlRestart(_FakeRuntimeManager):
            def execute(self, engine, action, payload=None):
                self.operations.append((engine, action, payload))
                if action == "stop":
                    backend._object_info_cache.store(backend.api_url, stale)
                    return {"ok": True}
                return {"apiUrl": "http://127.0.0.1:8188"}

        manager = _SameUrlRestart(self._engine(owned=True))
        installed = NodePackInstallResult(
            target=Path(r"C:\ComfyUI\custom_nodes\ai_studio_forge_parity"),
            fingerprint="abc",
            changed=True,
        )
        with mock.patch(
            "core.backend_runtime.get_backend_runtime_manager", return_value=manager,
        ), mock.patch(
            "core.comfy_node_pack.install_bundled_node_pack", return_value=installed,
        ):
            backend._preflight_bundled_node_pack()

        self.assertEqual(backend.api_url, "http://127.0.0.1:8188")
        self.assertIsNone(backend._object_info_cache.peek(backend.api_url))

    def test_doc_only_refresh_neither_restarts_nor_blocks_generation(self):
        for owned in (True, False):
            with self.subTest(owned=owned):
                manager = _FakeRuntimeManager(self._engine(owned=owned))
                backend = ComfyUIBackend("http://127.0.0.1:8188")
                snapshot = {"KSampler": {}}
                backend._object_info_cache.store(backend.api_url, snapshot)
                installed = NodePackInstallResult(
                    target=Path(r"C:\ComfyUI\custom_nodes\ai_studio_forge_parity"),
                    fingerprint="abc",
                    changed=True,
                    restart_required=False,
                )
                with mock.patch(
                    "core.backend_runtime.get_backend_runtime_manager", return_value=manager,
                ), mock.patch(
                    "core.comfy_node_pack.install_bundled_node_pack", return_value=installed,
                ):
                    backend._preflight_bundled_node_pack()

                self.assertEqual(manager.operations, [])
                self.assertEqual(backend.api_url, "http://127.0.0.1:8188")
                self.assertTrue(backend._node_pack_preflight_done)
                # Same .py/.json on disk: the running schema is still valid.
                self.assertIs(backend._object_info_cache.peek(backend.api_url), snapshot)

    def test_changed_pack_requires_manual_restart_for_external_comfy(self):
        manager = _FakeRuntimeManager(self._engine(owned=False))
        backend = ComfyUIBackend("http://127.0.0.1:8188")
        backend._object_info_cache.store(backend.api_url, {"KSampler": {}})
        installed = NodePackInstallResult(
            target=Path(r"C:\ComfyUI\custom_nodes\ai_studio_forge_parity"),
            fingerprint="abc",
            changed=True,
        )
        with mock.patch(
            "core.backend_runtime.get_backend_runtime_manager", return_value=manager,
        ), mock.patch(
            "core.comfy_node_pack.install_bundled_node_pack", return_value=installed,
        ):
            with self.assertRaisesRegex(RuntimeError, "외부 ComfyUI를 한 번 재시작"):
                backend._preflight_bundled_node_pack()

        self.assertEqual(manager.operations, [])
        self.assertFalse(backend._node_pack_preflight_done)
        # The next compile (after the user restarts ComfyUI) must refetch.
        self.assertIsNone(backend._object_info_cache.peek(backend.api_url))

    def test_refine_can_return_the_last_independent_image_like_forge(self):
        result = GenerationResult(
            success=True,
            image_data=b"first",
            artifacts=[
                MediaArtifact(kind="image", data=b"first", filename="first.png"),
                MediaArtifact(kind="image", data=b"last", filename="last.png"),
            ],
        )

        encoded = ComfyUIBackend._result_as_base64(
            result, "refine", prefer_last=True,
        )

        import base64
        self.assertEqual(base64.b64decode(encoded), b"last")
