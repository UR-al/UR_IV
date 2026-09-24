"""Static parity checks for the Python Studio Interface and its TypeScript client."""

from __future__ import annotations

import re
import shutil
import subprocess
import unittest
from pathlib import Path

from core.studio_application import CallContext, StudioApplication


ROOT = Path(__file__).resolve().parents[1]
TYPES = ROOT / "frontend" / "src" / "studio" / "types.ts"
SETTINGS = ROOT / "frontend" / "src" / "views" / "SettingsView.vue"
BRIDGE = ROOT / "frontend" / "src" / "bridge.js"
CLIENT = ROOT / "frontend" / "src" / "studio" / "client.ts"
TRANSPORT_TEST = (
    ROOT / "frontend" / "src" / "studio" / "resumableTransport.test.mjs"
)


class StudioFrontendContractTests(unittest.TestCase):
    def test_resumable_transport_and_fail_closed_handshake(self):
        node = shutil.which("node")
        if node is None:
            self.skipTest("Node.js가 없어 frontend Studio transport 테스트를 건너뜁니다")
        result = subprocess.run(
            [node, "--test", str(TRANSPORT_TEST)],
            cwd=ROOT,
            text=True,
            encoding="utf-8",
            errors="replace",
            capture_output=True,
            timeout=10,
            check=False,
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_bridge_and_client_use_the_resumable_fail_closed_seam(self):
        bridge = BRIDGE.read_text(encoding="utf-8")
        client = CLIENT.read_text(encoding="utf-8")
        self.assertIn("new ResumableStudioTransport", bridge)
        self.assertIn("transport.acknowledge(event.seq)", client)
        self.assertIn("selectStudioClient(transport", client)
        self.assertNotIn("v1 handshake failed; using legacy adapter", client)
        self.assertIn("transport.resumeError.connect", client)
        self.assertIn("new CursorRecoveryController", client)
        self.assertIn("type: 'reconciled'", client)
        self.assertIn("private activeResume: Promise<void> | null = null", client)
        self.assertIn("const inheritedResume = this.activeResume", client)
        self.assertIn("await finishResumeWithRecovery", client)

    def test_typescript_operations_match_python_description(self):
        source = TYPES.read_text(encoding="utf-8")
        declaration = re.search(
            r"export type StudioOperation\s*=\s*(.*?)\n\nexport type StudioTopic",
            source,
            re.DOTALL,
        )
        self.assertIsNotNone(declaration, "StudioOperation union을 찾지 못했습니다")
        frontend = set(re.findall(r"'([a-z][a-z0-9_.]+)'", declaration.group(1)))

        context = CallContext("contract-test", "in-memory", frozenset({"native"}))
        backend = {
            item["name"] for item in StudioApplication().describe(context)["operations"]
        }
        self.assertEqual(frontend, backend)

    def test_migrated_settings_domains_do_not_call_legacy_qt_methods(self):
        source = SETTINGS.read_text(encoding="utf-8")
        legacy_methods = {
            "getBackendRuntimeState",
            "runBackendRuntimeOperation",
            "getGenerationApiState",
            "runGenerationApiOperation",
            "getForgeModelPaths",
            "saveForgeModelPaths",
            "resetForgeModelPaths",
            "refreshForgeModelPaths",
            "selectBackendInstallDirectory",
            "selectBackendExtensionDirectory",
            "selectForgeModelDirectory",
        }
        used = {name for name in legacy_methods if name in source}
        self.assertEqual(used, set())

    def test_legacy_settings_client_and_slots_are_removed_on_both_sides(self):
        """studio 는 데스크톱·웹 모두 항상 등록된다 — legacy 폴백과 그 슬롯은 도달 불가라 제거(audit #176)."""
        from ui.vue_bridge import VueBridge

        legacy_methods = (
            "getBackendRuntimeState", "runBackendRuntimeOperation",
            "getGenerationApiState", "runGenerationApiOperation",
            "getForgeModelPaths", "saveForgeModelPaths", "resetForgeModelPaths",
            "refreshForgeModelPaths", "selectBackendInstallDirectory",
            "selectBackendExtensionDirectory", "selectForgeModelDirectory",
        )
        client = CLIENT.read_text(encoding="utf-8")
        for name in legacy_methods:
            self.assertNotIn(name, client, name)
            self.assertFalse(hasattr(VueBridge, name), name)
        self.assertNotIn("LegacyStudioClient", client)
        self.assertNotIn("generationApiEvent", client)
        self.assertIn("createUnavailableClient", client)

    def test_settings_mutations_and_lifecycle_fail_closed(self):
        source = SETTINGS.read_text(encoding="utf-8")
        self.assertIn("const forgeCanMutate = ref(false)", source)
        self.assertIn("forgeBusy || !forgeCanMutate", source)
        self.assertIn("forgeEnvironmentLocked[field.key] || forgeBusy || !forgeCanMutate", source)
        self.assertNotIn("웹 모드에서는 경로를 직접 입력하세요", source)
        self.assertIn("let settingsDisposed = false", source)
        self.assertIn("settingsDisposed = true", source)
        self.assertIn("disconnectModelPathsEvent?.()", source)
        self.assertIn("error instanceof StudioClientError ? error.fields", source)


if __name__ == "__main__":
    unittest.main()
