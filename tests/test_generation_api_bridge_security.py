"""Generation API bridge privilege-boundary regression tests."""

from __future__ import annotations

import unittest

try:
    from ui.vue_bridge import VueBridge
    _IMPORT_ERROR = ""
except ModuleNotFoundError as exc:
    if not (exc.name or "").startswith("PyQt6"):
        raise
    VueBridge = None
    _IMPORT_ERROR = str(exc)


class _SnapshotManager:
    def snapshot(self, include_secret=False):
        return {
            "config": {
                "token": "top-secret-token",
                "targets": [{
                    "id": "remote-comfy",
                    "name": "Remote Comfy",
                    "engine": "comfyui",
                    "url": "http://192.0.2.55:8188",
                    "workflowPath": "C:\\private\\txt2img.json",
                    "img2imgWorkflowPath": "C:\\private\\img2img.json",
                }],
            },
            "running": False,
            "listenUrl": "http://127.0.0.1:17860",
        }


@unittest.skipIf(VueBridge is None, f"PyQt6 unavailable: {_IMPORT_ERROR}")
class GenerationApiBridgeSecurityTests(unittest.TestCase):
    def test_bridge_has_no_unredacted_generation_api_slots(self):
        """생성 API 는 Studio Interface(core/studio_application.py)만 노출한다 — 예전 VueBridge
        레거시 슬롯은 도달 불가라 제거했다(audit #176). 되살아나면 redaction 이 두 벌이 된다."""
        for name in ("getGenerationApiState", "runGenerationApiOperation",
                     "_generation_api_public_snapshot", "generationApiEvent"):
            self.assertFalse(hasattr(VueBridge, name), name)

    def test_web_studio_snapshot_redacts_token_remote_url_and_workflow_paths(self):
        from core.studio_application import CallContext, StudioApplication

        app = StudioApplication(generation_api_manager=_SnapshotManager())
        web = CallContext("browser", "websocket", frozenset())
        reply = app.invoke(web, {"version": 1, "requestId": "r1",
                                 "operation": "generation_api.snapshot", "input": {}})
        snapshot = reply["data"]
        target = snapshot["config"]["targets"][0]
        self.assertNotIn("token", snapshot["config"])
        self.assertNotIn("url", target)
        self.assertNotIn("workflowPath", target)
        self.assertNotIn("img2imgWorkflowPath", target)
        self.assertTrue(target["urlConfigured"])
        self.assertTrue(target["workflowConfigured"])
        self.assertFalse(snapshot["nativeOperations"])

    def test_web_mode_rejects_native_backend_manager_action_server_side(self):
        bridge = VueBridge()
        bridge._backend_runtime_is_web_mode = lambda: True
        calls = []
        notifications = []
        bridge.set_action_handler(lambda action, payload: calls.append((action, payload)))
        bridge.showNotification.connect(lambda level, message: notifications.append((level, message)))
        bridge.onAction("show_api_manager", "{}")
        self.assertEqual(calls, [])
        self.assertTrue(notifications)
        self.assertIn("웹 모드", notifications[0][1])


if __name__ == "__main__":
    unittest.main()
