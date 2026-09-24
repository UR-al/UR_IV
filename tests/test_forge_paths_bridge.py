"""Forge 모델 경로 — Studio model_paths.* 경로와 메인 VAE/TE 위젯 새로고침 회귀 테스트.

레거시 VueBridge 슬롯(get/select/save/reset/refreshForgeModelPaths)은 도달 불가라 제거했다
(audit #176). Settings 는 Studio Interface(core/studio_application.py)만 쓴다.
"""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from PyQt6.QtCore import QObject

from core import forge_modules
from core.main_module_choices import (
    VAE_DEFAULT, filter_te_selection, keep_vae_selection, main_module_choices,
)
from core.studio_application import CallContext, StudioApplication
from ui.vue_bridge import VueBridge
from ui.widget_proxies import ComboBoxProxy, LineEditProxy

NATIVE = CallContext("desktop-ui", "qwebchannel", frozenset({"native"}))
WEB = CallContext("web-ui", "qwebchannel-websocket", frozenset())


def _request(operation, values=None, request_id="r1"):
    return {"version": 1, "requestId": request_id, "operation": operation,
            "input": {} if values is None else values}


class _Host:
    def __init__(self):
        self.refreshes = 0
        self.picks = []

    def refresh_model_widgets(self):
        self.refreshes += 1

    def pick_directory(self, kind, selector, current):
        self.picks.append((kind, selector, current))
        return None


class _FakeInventory:
    """ModelInventory.entries 계약만 흉내 — API 이름과 맞는 VAE 는 runtimeName 이 API 원본."""

    def __init__(self, vae, te):
        self._vae = vae
        self._te = te

    def entries(self, category, *, backend_items=None):
        if category == "vae":
            out = []
            for name in self._vae:
                runtime = name
                for item in backend_items or []:
                    if str(item).split(" [")[0] == name:
                        runtime = str(item)
                out.append({"runtimeName": runtime})
            return out
        if category == "text_encoders":
            return [{"runtimeName": name} for name in self._te]
        return []


class _Parent(QObject):
    def __init__(self, api_vae=None):
        super().__init__()
        self._last_vae_api_items = api_vae


class TestForgePathsStudio(unittest.TestCase):
    def _payload(self, root: Path) -> dict[str, str]:
        payload = {}
        for key in forge_modules.FORGE_PATH_KEYS:
            path = root / key
            path.mkdir()
            payload[key] = str(path)
        (root / "checkpoint_dir" / "model.safetensors").write_bytes(b"")
        (root / "lora_dir" / "style.safetensors").write_bytes(b"")
        (root / "vae_dir" / "vae.safetensors").write_bytes(b"")
        (root / "text_encoder_dir" / "clip.safetensors").write_bytes(b"")
        return payload

    def test_save_snapshot_and_reset_return_structured_state_and_refresh_widgets(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            config = root / "forge_model_paths.json"
            payload = self._payload(root)
            host = _Host()
            app = StudioApplication(host=host)

            with patch.object(forge_modules, "FORGE_PATHS_FILE", config):
                saved = app.invoke(NATIVE, _request("model_paths.save", {"paths": payload}))
                loaded = app.invoke(NATIVE, _request("model_paths.snapshot", request_id="r2"))
                reset = app.invoke(NATIVE, _request("model_paths.reset", request_id="r3"))

            self.assertEqual(saved["status"], "ok", saved)
            self.assertEqual(saved["data"]["paths"], payload)
            self.assertEqual(
                {key: saved["data"]["entries"][key]["count"] for key in forge_modules.FORGE_PATH_KEYS},
                {key: 1 for key in forge_modules.FORGE_PATH_KEYS},
            )
            self.assertEqual(loaded["status"], "ok")
            self.assertEqual(reset["status"], "ok")
            self.assertFalse(config.exists())
            self.assertEqual(host.refreshes, 2, "save/reset 뒤 VAE/TE 위젯을 새로 만든다")

    def test_invalid_payload_reports_field_errors_without_writing(self):
        with tempfile.TemporaryDirectory() as temp:
            config = Path(temp) / "forge_model_paths.json"
            app = StudioApplication(host=_Host())
            with patch.object(forge_modules, "FORGE_PATHS_FILE", config):
                reply = app.invoke(NATIVE, _request("model_paths.save", {
                    "paths": {"checkpoint_dir": "relative/path"},
                }))
            self.assertEqual(reply["status"], "error")
            self.assertIn("checkpoint_dir", reply["error"]["details"]["fields"])
            self.assertFalse(config.exists())

    def test_invalid_directory_picker_key_is_rejected(self):
        host = _Host()
        reply = StudioApplication(host=host).invoke(NATIVE, _request(
            "native.pick_directory", {"purpose": "model_path", "key": "not-a-forge-key"}))
        self.assertEqual(reply["status"], "error")
        self.assertEqual(host.picks, [])

    def test_web_context_cannot_mutate_or_pick_model_paths(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            config = root / "forge_model_paths.json"
            (root / "original").mkdir()
            (root / "replacement").mkdir()
            original = self._payload(root / "original")
            replacement = self._payload(root / "replacement")
            forge_modules.save_forge_paths(original, config_path=config, environ={})
            original_config = config.read_bytes()
            host = _Host()
            app = StudioApplication(host=host)
            with patch.object(forge_modules, "FORGE_PATHS_FILE", config):
                replies = [
                    app.invoke(WEB, _request("model_paths.save", {"paths": replacement})),
                    app.invoke(WEB, _request("model_paths.reset")),
                    app.invoke(WEB, _request("model_paths.refresh")),
                    app.invoke(WEB, _request("native.pick_directory",
                                             {"purpose": "model_path", "key": "checkpoint_dir"})),
                ]
            for reply in replies:
                self.assertEqual(reply["status"], "error")
                self.assertEqual(reply["error"]["code"], "FORBIDDEN")
            self.assertEqual(config.read_bytes(), original_config)
            self.assertEqual((host.refreshes, host.picks), (0, []))

    def test_legacy_model_path_slots_are_gone(self):
        for name in ("getForgeModelPaths", "selectForgeModelDirectory", "saveForgeModelPaths",
                     "resetForgeModelPaths", "refreshForgeModelPaths", "_forge_model_paths_web_denial"):
            self.assertFalse(hasattr(VueBridge, name), name)


class TestMainModuleChoices(unittest.TestCase):
    def test_vae_uses_inventory_with_api_runtime_names(self):
        inv = _FakeInventory(["sdxl_vae.safetensors", "sub/anime.vae.pt"], [])
        vae, _te = main_module_choices(inv, ["Use same VAE", "sdxl_vae.safetensors [abc123]"])
        self.assertEqual(vae, [VAE_DEFAULT, "sdxl_vae.safetensors [abc123]", "sub/anime.vae.pt"])

    def test_vae_falls_back_to_api_list_when_disk_is_empty(self):
        vae, te = main_module_choices(_FakeInventory([], ["t5/t5xxl.safetensors"]), ["Use same VAE", "x.vae"])
        self.assertEqual(vae, [VAE_DEFAULT, "x.vae"])
        self.assertEqual(te, ["t5/t5xxl.safetensors"])
        self.assertEqual(main_module_choices(None, None), ([VAE_DEFAULT], []))

    def test_te_filter_and_vae_keep(self):
        self.assertIsNone(filter_te_selection("a, b/c", ["a", "b/c"]))
        self.assertEqual(filter_te_selection("a, gone", ["a"]), "a")
        self.assertEqual(keep_vae_selection("x", [VAE_DEFAULT, "x"]), "x")
        self.assertEqual(keep_vae_selection("gone", [VAE_DEFAULT, "x"]), VAE_DEFAULT)


class TestRefreshForgeModuleWidgets(unittest.TestCase):
    def _bridge(self, api_vae=None):
        self._parent = _Parent(api_vae)   # 테스트 동안 parent 수명 유지
        bridge = VueBridge(self._parent)
        vae = ComboBoxProxy(bridge, "vae_main_combo")
        te = LineEditProxy(bridge, "te_main_input")
        pushed = []
        props = []
        bridge.widgetValueChanged.connect(lambda wid, value: pushed.append((wid, value)))
        bridge.widgetPropertyChanged.connect(lambda wid, prop, value: props.append((wid, prop, value)))
        return bridge, vae, te, pushed, props

    @staticmethod
    def _items(combo):
        return [combo.itemText(i) for i in range(combo.count())]

    def test_refresh_keeps_subfolder_and_secondary_root_te_and_api_named_vae(self):
        """연결 시 목록(하위폴더 TE·보조 루트 TE·API 이름 VAE)이 Settings 새로고침 뒤에도 남는다."""
        bridge, vae, te, pushed, _props = self._bridge(["anime.safetensors [f00d]"])
        vae.addItems([VAE_DEFAULT, "anime.safetensors [f00d]"])
        vae.setCurrentText("anime.safetensors [f00d]")
        te.setText("qwen/qwen_3_06b.safetensors, comfy_only_clip.safetensors")
        inventory = _FakeInventory(
            ["anime.safetensors"],
            ["qwen/qwen_3_06b.safetensors", "comfy_only_clip.safetensors"],
        )
        with patch.object(VueBridge, "_module_choice_inventory", return_value=inventory):
            bridge._refresh_forge_module_widgets()
        self.assertEqual(vae.currentText(), "anime.safetensors [f00d]")
        self.assertEqual(te.text(), "qwen/qwen_3_06b.safetensors, comfy_only_clip.safetensors")
        self.assertIn(("vae_main_combo", "anime.safetensors [f00d]"), pushed)

    def test_refresh_replaces_removed_vae_selection_with_default(self):
        bridge, vae, te, pushed, props = self._bridge()
        vae.addItems([VAE_DEFAULT, "old.safetensors"])
        vae.setCurrentText("old.safetensors")
        te.setText("old-clip.safetensors")
        with patch.object(VueBridge, "_module_choice_inventory",
                          return_value=_FakeInventory(["new.safetensors"], [])):
            bridge._refresh_forge_module_widgets()
        self.assertEqual(vae.currentText(), VAE_DEFAULT)
        self.assertEqual(self._items(vae), [VAE_DEFAULT, "new.safetensors"])
        self.assertEqual(te.text(), "")
        self.assertIn(("vae_main_combo", VAE_DEFAULT), pushed)
        self.assertIn(("te_main_input", ""), pushed)
        self.assertTrue(any(wid == "te_main_input" and prop == "items" for wid, prop, _v in props))

    def test_refresh_survives_inventory_failure_with_api_list(self):
        bridge, vae, _te, _pushed, _props = self._bridge(["api.vae"])
        with patch.object(VueBridge, "_module_choice_inventory", side_effect=RuntimeError("boom")):
            bridge._refresh_forge_module_widgets()
        self.assertEqual(self._items(vae), [VAE_DEFAULT, "api.vae"])

    def test_refresh_while_disconnected_keeps_the_users_vae_not_the_boot_fallback(self):
        """부팅 setText(A) → 연결 → 사용자가 B 선택 → 연결 오류로 비움 → Settings 경로 새로고침: B 유지."""
        bridge, vae, _te, pushed, _props = self._bridge()
        vae.setText("a.safetensors")                                    # 부팅 load_settings — 목록 전
        vae.addItems([VAE_DEFAULT, "a.safetensors", "b.safetensors"])
        vae._on_vue_changed("b.safetensors")                            # 사용자 선택
        vae.clear()                                                     # on_webui_info_error
        with patch.object(VueBridge, "_module_choice_inventory",
                          return_value=_FakeInventory(["a.safetensors", "b.safetensors"], [])):
            bridge._refresh_forge_module_widgets()
        self.assertEqual(vae.currentText(), "b.safetensors")
        self.assertIn(("vae_main_combo", "b.safetensors"), pushed)


if __name__ == "__main__":
    unittest.main()
