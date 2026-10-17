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
from tests._sam_extra_ext import EXT_ROOT, requires_extension


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
    "ForgeNeoAnimaOptimalScale",
    "ForgeNeoKSamplerCNS",
    "ForgeNeoLatentInput",
    "ForgeNeoHiresFix",
    "ForgeNeoSAM3Mask",
    "ForgeNeoSAM3Detailer",
    "ForgeNeoSAM3Refine",
    "ForgeNeoADetailer",
    "ForgeNeoH3ConditioningCachePrepare",
    "ForgeNeoH3ConditioningCacheLoad",
    "ForgeNeoAnimaVAEDeGrid",
    "ForgeNeoExtraSchedulers",
    "ForgeNeoExtraSamplerSettings",
    "ForgeNeoAnimaSpeed",
    "ForgeNeoColorcraft",
})
# 컴파일러가 따옴표 리터럴이 아니라 앱 상수로 만드는 노드 — (그 상수를 정의한 모듈, 상수 이름).
# 컴파일러 소스는 ``<모듈 이름>.<상수>`` 로 참조해야 하고, 상수 값은 노드 ID 와 같아야 한다.
COMPILER_NODE_CONSTANTS = {
    "ForgeNeoAnimaVAEDeGrid": ("core/vae_degrid.py", "COMFY_NODE_CLASS"),
    # C2a: 요청에 Extra Schedulers 값이 있고 어느 패스가 laplace·custom·flow_cosmos_rho7·flow_cosmos_dynamic 이면
    # flow shift 바로 뒤(LoRA 앞)에 넣는다(core/comfy_workflow_compiler._add_extra_schedulers).
    "ForgeNeoExtraSchedulers": ("core/extra_schedulers.py", "COMFY_NODE_CLASS"),
    # C3: ER SDE 값이 기본값과 다를 때만, 패스들이 쓰는 모델 스택의 가이던스 앞에 넣는다(기본 그래프·ADetailer/SAM3 LoRA
    # 분기·사용자 워크플로 — core/comfy_workflow_compiler._add_extra_samplers).
    "ForgeNeoExtraSamplerSettings": ("core/extra_samplers.py", "COMFY_SETTINGS_NODE"),
    # C4a: Anima SPEED 블록이 켜져 있으면 메인 패스(base/img2img, hires 는 블록의 hires 일 때)마다 Detail Daemon 뒤에
    # 넣는다(core/comfy_workflow_compiler._add_speed). 디테일러 패스에는 넣지 않는다.
    "ForgeNeoAnimaSpeed": ("core/anima_speed.py", "COMFY_NODE_CLASS"),
    # C5: Colorcraft 블록에 그 패스에서 도는 수정자가 있으면 메인 패스(base/img2img, hires)마다 SPEED 뒤 마지막에 넣는다
    # (core/comfy_workflow_compiler._add_colorcraft). 디테일러 패스에는 넣지 않는다.
    "ForgeNeoColorcraft": ("core/colorcraft.py", "COMFY_NODE_CLASS"),
}
# 팩에 먼저 들어온 컴파일러용 노드 → 사유(필수). 컴파일러가 아직 만들지 않으므로 소스 검사를 건너뛰고,
# 컴파일러 소스에 나타나면(리터럴이든 상수 참조든) 낡은 항목으로 실패한다 — 그때 여기서 지운다.
PENDING_COMPILER_NODES: dict[str, str] = {
}
CUSTOM_WORKFLOW_NODES = frozenset({
    "ForgeNeoAnimaLoraLoaderModelOnly",
    "ForgeNeoAnimaDAVE",
    "ForgeNeoAnimaModGuidance",
    "ForgeNeoAnimaSafePAG",
    "ForgeNeoDCWCWMSMC",
    "ForgeNeoCNSSamplerPatch",
    "ForgeNeoMaskSelector",
    "ForgeNeoLoraBlockWeight",
    "ForgeNeoCharacterReference",
    "ForgeNeoReferencePrompt",
    "ForgeNeoReferenceOutput",
    "ForgeNeoAnimaPiD",
    "ForgeNeoAnimaVAE2x",
    "ForgeNeoSAM3TileRepair",
    "ForgeNeoAnimaTileRepair",
    "ForgeNeoSaveImage",
    "AIStudioRelight",
    "ForgeNeoMossSpectrum",       # 1.9.0: Spectrum for MOSS-Image (MODEL -> MODEL; the compiler never emits it)
    "ForgeNeoDoraInferenceMode",  # 1.9.0: sam-extra's DoRA Inference Mode on a MODEL's DoRA patches (MODEL -> MODEL)
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
        self.assertEqual(len(EXPECTED_BUNDLED_NODES), 44)
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
        for name, reason in PENDING_COMPILER_NODES.items():
            with self.subTest(pending=name):
                self.assertIn(name, COMPILER_BUILT_NODES)
                self.assertTrue(reason.strip())
                module, constant = COMPILER_NODE_CONSTANTS.get(name, ("", ""))
                reference = f"{Path(module).stem}.{constant}" if module else None
                self.assertNotIn(f'"{name}"', sources, "낡은 PENDING 항목 — 컴파일러가 이미 만든다")
                if reference:
                    self.assertNotIn(reference, sources, "낡은 PENDING 항목 — 컴파일러가 이미 만든다")
        for name in sorted(COMPILER_BUILT_NODES - set(PENDING_COMPILER_NODES)):
            with self.subTest(node=name):
                if name not in COMPILER_NODE_CONSTANTS:
                    self.assertIn(f'"{name}"', sources)
                    continue
                module, constant = COMPILER_NODE_CONSTANTS[name]
                defined = (root / module).read_text(encoding="utf-8")
                self.assertRegex(defined, rf'(?m)^{constant}\s*=\s*"{name}"')
                self.assertIn(f"{Path(module).stem}.{constant}", sources)
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


PACK_DIR = Path(bundled_pack.__file__).resolve().parent
# Pack 1.7.0's sam-extra ports (THIRD_PARTY_NOTICES.md "sam-extra ports (pack 1.7.0)" table).
SAM_EXTRA_PORT_PREFIXES = ("extra_schedulers", "extra_samplers", "speed_", "colorcraft_")
# The author's grant for the code sam-extra (GPL-3.0-only) wrote and the pack reuses (user decision 2026-10-05).
SAM_EXTRA_GRANT = ("sam-extra (GPL-3.0-only) parts written by the same author (UR-al), licensed to this pack under "
                   "the MIT License by that author")
# What the notices and the README said before the grant — the pack does contain sam-extra code.
OLD_NOT_COPIED_CLAIMS = ("No sam-extra code is included", "Its code was not copied", "No extension code is included",
                         "without copying its code", "neither copied nor imported", "used **only as a test oracle**",
                         "test oracle only", "never copied or imported", "no extension code is included")


def _section(text: str, heading: str) -> str:
    start = text.index(heading)
    return text[start:text.index("\n## ", start + 1)]


def _flat(text: str) -> str:
    """``text`` with every run of whitespace (the line breaks of wrapped Markdown) as one space."""
    return " ".join(text.split())


class TestPackNotices(unittest.TestCase):
    def setUp(self):
        self.notices = (PACK_DIR / "THIRD_PARTY_NOTICES.md").read_text(encoding="utf-8")

    def test_every_license_file_is_named_in_the_notices_and_back(self):
        files = sorted(path.name for path in (PACK_DIR / "LICENSES").iterdir() if path.is_file())
        self.assertEqual(len(files), 18)      # 1.9.0: moss-image-Apache-2.0.txt (the MOSS-Image head in moss_spectrum)
        for name in files:
            with self.subTest(license=name):
                self.assertIn(f"LICENSES/{name}", self.notices)
        import re
        for name in sorted(set(re.findall(r"LICENSES/([A-Za-z0-9_.\-]+\.txt)", self.notices))):
            with self.subTest(named=name):
                self.assertIn(name, files)

    def test_the_sam_extra_mit_text_is_the_authors(self):
        text = (PACK_DIR / "LICENSES" / "sam-extra-MIT.txt").read_text(encoding="utf-8")
        mit = (PACK_DIR / "LICENSES" / "speed-MIT.txt").read_text(encoding="utf-8")
        self.assertEqual(text, mit.replace("Copyright (c) 2026 Howard Xiao", "Copyright (c) 2026 UR-al"))
        self.assertTrue(text.startswith("MIT License\n\nCopyright (c) 2026 UR-al\n"))

    def test_the_notices_state_the_authors_mit_grant_instead_of_not_copied(self):
        for claim in OLD_NOT_COPIED_CLAIMS:
            with self.subTest(claim=claim):
                self.assertNotIn(claim, _flat(self.notices))
        shared = _section(self.notices, "## Code shared with the sam-extra Forge extension (MIT, by its author)")
        self.assertIn(SAM_EXTRA_GRANT, _flat(shared))
        self.assertIn("`LICENSES/sam-extra-MIT.txt`", shared)
        self.assertIn("Formulas shared with GPL-3.0 hosts", shared)
        readme = (PACK_DIR / "README.md").read_text(encoding="utf-8")
        for claim in OLD_NOT_COPIED_CLAIMS:
            with self.subTest(readme_claim=claim):
                self.assertNotIn(claim, _flat(readme))
        self.assertIn(SAM_EXTRA_GRANT, _flat(readme))

    def test_every_sam_extra_port_module_is_in_the_mit_grant_table(self):
        # The 1.7.0 ports reuse sam-extra's own code under its author's MIT grant; the installed extension is also
        # the test oracle. Every port module must be in that section's table.
        section = _section(self.notices, "## sam-extra ports (pack 1.7.0)")
        self.assertIn(SAM_EXTRA_GRANT, _flat(section))
        self.assertIn("test oracle", section)
        modules = sorted(path.name for path in PACK_DIR.glob("*.py") if path.name.startswith(SAM_EXTRA_PORT_PREFIXES))
        self.assertEqual(len(modules), 19)    # 1.7.0: + extra_samplers_ipndm.py (sam-extra 0.33.1's IPNDM_V)
        for name in modules:
            with self.subTest(module=name):
                self.assertIn(f"`{name}`", section)

    def test_the_forge_crop_rule_is_named_and_carries_no_forge_code(self):
        """FIX-recheck(GD recheck): ``mask_ops.expand_crop_region`` was a near-copy (0.88) of Forge's AGPL-3.0
        ``modules/masking.py`` function since pack 1.3.0 and the notices named no Forge code. The pack's own code now
        follows only the rule (``TestPackCarriesNoForgeCode``); the notices say which rule and that no Forge code is
        included."""
        section = _flat(_section(self.notices, "## Forge inpaint crop rule (`mask_ops.py`)"))
        for text in ("`mask_ops.py`", "`expand_crop_region`", "`modules/masking.py`", "AGPL-3.0",
                     "No Forge code is included", "`tests/test_comfy_sam3_nodes.py`", "`tests/test_comfy_node_pack.py`"):
            with self.subTest(text=text):
                self.assertIn(text, section)

    def test_a_module_with_shared_sam_extra_code_is_not_called_the_packs_own_without_naming_it(self):
        """FIX-recheck(GD recheck, low): the 1.7.0 table's ``guidance_common.py`` row said 'None ... the code is this
        pack's own', and so did the module docstring, while the shared-code table lists that module's Haar helpers as
        sam-extra code under the grant. A row of the 1.7.0 table that grants nothing ('None'), and a module docstring
        that calls the code the pack's own, must name the functions the shared-code table lists for that module."""
        import ast
        import re

        def cells(line):
            return [cell.strip() for cell in line.strip().strip("|").split("|")]

        listed: dict = {}
        shared = _section(self.notices, "## Code shared with the sam-extra Forge extension (MIT, by its author)")
        for line in shared.splitlines():
            row = cells(line)
            if not line.startswith("| `") or len(row) != 3:
                continue
            modules = re.findall(r"`([a-z0-9_]+\.py)`", row[0])
            groups = row[1].split(";")
            if len(modules) > 1 and len(groups) == len(modules):
                pairs = zip(modules, groups)
            else:
                pairs = ((module, row[1]) for module in modules)
            for module, text in pairs:
                listed.setdefault(module, []).extend(re.findall(r"`([A-Za-z_][A-Za-z0-9_.]*)`", text))
        self.assertEqual(listed.get("guidance_common.py"), ["_pad_even", "_haar_dwt", "_haar_idwt", "_sigma_norm"])
        checked = 0
        for line in _section(self.notices, "## sam-extra ports (pack 1.7.0)").splitlines():
            row = cells(line)
            if not line.startswith("| `") or len(row) != 5 or not row[1].startswith("None"):
                continue
            for module in re.findall(r"`([a-z0-9_]+\.py)`", row[0]):
                for function in listed.get(module, ()):
                    checked += 1
                    with self.subTest(row=module, function=function):
                        self.assertIn(f"`{function}`", row[1])
        self.assertGreaterEqual(checked, 4)
        for module, functions in sorted(listed.items()):
            docstring = " ".join((ast.get_docstring(ast.parse((PACK_DIR / module).read_text(encoding="utf-8")))
                                  or "").split())
            if "pack's own" not in docstring:
                continue
            for function in functions:
                with self.subTest(docstring=module, function=function):
                    self.assertIn(f"``{function}``", docstring)


@requires_extension
class TestSamExtraSharedCodeIsListed(unittest.TestCase):
    """Every pack function that is a near-copy of a function of the installed sam-extra (difflib token ratio of the
    bodies >= 0.80, the review probe's measure: ``tests/_copy_similarity.py``) is named in THIRD_PARTY_NOTICES.md
    next to its module — in a table row or paragraph that names the module — so the list of code shared with
    sam-extra (the author's MIT grant, or a permissive upstream both copy) stays complete."""

    def test_every_near_copy_is_named_next_to_its_module(self):
        from tests import _copy_similarity as similarity

        notices = (PACK_DIR / "THIRD_PARTY_NOTICES.md").read_text(encoding="utf-8")
        blocks = []
        for paragraph in notices.split("\n\n"):
            lines = paragraph.splitlines()
            rows = [line for line in lines if line.startswith("|")]
            blocks.extend(rows)
            blocks.append("\n".join(line for line in lines if not line.startswith("|")))
        pack = similarity.collect(PACK_DIR)
        extension = {}
        for sub in ("sam3ext", "scripts"):
            extension.update(similarity.collect(EXT_ROOT / sub, sub + "/"))
        for path in sorted(EXT_ROOT.glob("*.py")):
            extension.update(similarity.function_tokens(path, path.name))
        copies = similarity.near_copies(pack, extension)
        self.assertGreaterEqual(len(copies), 80)                       # the probe found 86 on sam-extra v0.33.0
        missing = []
        for name, (ratio, match) in sorted(copies.items()):
            module, qualname = name.split(":", 1)
            names = {qualname} | {part for part in qualname.split(".") if not part.startswith("__")}
            if not any(f"`{module}`" in block and any(f"`{candidate}`" in block for candidate in names)
                       for block in blocks):
                missing.append(f"{name} ({ratio:.2f} ~ {match})")
        self.assertEqual(missing, [])


class TestPackCarriesNoForgeCode(unittest.TestCase):
    """The pack is copied to the user's ComfyUI under MIT, so it carries no code of Forge Neo: A1111 webui's
    ``modules/`` (AGPL-3.0) and Forge's own ``modules_forge/`` and ``backend/``. No pack function may be a near-copy of
    a function there (difflib token ratio of the bodies >= 0.80 — the measure of ``tests/_copy_similarity.py`` that
    keeps the sam-extra list complete). The third-party packages Forge ships in ``modules_forge/packages`` (k-diffusion,
    ComfyUI, gguf …) keep their own licenses and are not compared here. Where the pack follows a Forge rule, it says so
    in THIRD_PARTY_NOTICES.md ("Forge inpaint crop rule"). FIX-recheck: ``mask_ops.expand_crop_region`` was a 0.88
    near-copy of ``modules/masking.py`` ``expand_crop_region`` from pack 1.3.0 on. Forge is found the way the Extra
    Samplers oracle finds it (``tests/_forge_k_diffusion.require_forge`` — skips without Forge)."""

    def test_no_pack_function_is_a_near_copy_of_forge_code(self):
        from tests import _copy_similarity as similarity
        from tests._forge_k_diffusion import require_forge

        root = require_forge()
        forge: dict = {}
        unreadable = []
        for sub in ("modules", "modules_forge", "backend"):
            for path, relative in similarity.python_files(root / sub, excluded=similarity.EXCLUDED_DIRS | {"packages"}):
                try:
                    forge.update(similarity.function_tokens(path, f"{sub}/{relative}"))
                except (SyntaxError, UnicodeDecodeError, ValueError):
                    unreadable.append(f"{sub}/{relative}")
        self.assertIn("modules/masking.py:expand_crop_region", forge)
        self.assertGreater(len(forge), 2000)                     # 2,594 functions in the installed Forge (2026-10-05)
        self.assertLessEqual(len(unreadable), 5, unreadable)
        copies = similarity.near_copies(similarity.collect(PACK_DIR), forge)
        self.assertEqual({name: f"{ratio:.2f} ~ {match}" for name, (ratio, match) in sorted(copies.items())}, {})
