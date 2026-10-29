"""Anima studio web extension plumbing: route registry, WEB_DIRECTORY, installer fingerprint."""
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import comfy_custom_nodes.ai_studio_forge_parity as pack
from comfy_custom_nodes.ai_studio_forge_parity import studio_routes
from core.comfy_node_pack import install_bundled_node_pack, node_pack_fingerprint


def _fake_server():
    registered = []

    def route(method, path):
        def save(handler):
            registered.append((method, path, handler))
            return handler
        return save

    instance = SimpleNamespace(routes=SimpleNamespace(route=route))
    modules = {"server": SimpleNamespace(PromptServer=SimpleNamespace(instance=instance))}
    return registered, modules


class StudioRoutesTests(unittest.TestCase):
    def setUp(self):
        saved = dict(studio_routes._HANDLERS)
        self.addCleanup(lambda: (studio_routes._HANDLERS.clear(), studio_routes._HANDLERS.update(saved)))
        studio_routes._HANDLERS.clear()

    def test_register_routes_without_server_is_noop(self):
        with mock.patch.dict("sys.modules"):
            import sys
            sys.modules.pop("server", None)
            studio_routes.add_handler("GET", "/ping", lambda request: None)
            self.assertFalse(studio_routes.register_routes())

    def test_register_routes_is_idempotent(self):
        registered, modules = _fake_server()
        handler = lambda request: None
        studio_routes.add_handler("get", "/ping", handler)
        with mock.patch.dict("sys.modules", modules):
            self.assertTrue(studio_routes.register_routes())
            self.assertTrue(studio_routes.register_routes())
        self.assertEqual(registered, [("GET", "/aistudio/studio/ping", handler)])

    def test_handler_added_after_server_is_up_is_mounted_once(self):
        registered, modules = _fake_server()
        with mock.patch.dict("sys.modules", modules):
            studio_routes.register_routes()
            studio_routes.add_handler("POST", "/run", lambda request: None)
            studio_routes.register_routes()
        self.assertEqual([item[:2] for item in registered], [("POST", "/aistudio/studio/run")])

    def test_route_prefix(self):
        self.assertEqual(studio_routes.ROUTE_PREFIX, "/aistudio/studio")

    def test_pack_exports_web_directory(self):
        self.assertEqual(pack.WEB_DIRECTORY, "./web")
        self.assertIn("WEB_DIRECTORY", pack.__all__)
        web = Path(pack.__file__).parent / "web" / "ai_studio" / "studio_main.js"
        self.assertIn("AIStudio.Studio", web.read_text(encoding="utf-8"))

    def test_no_test_files_under_web_directory(self):
        web = Path(pack.__file__).parent / "web"
        names = [p.name for p in web.rglob("*.js") if ".test." in p.name or p.name.startswith("test_")]
        self.assertEqual(names, [])


class StudioInstallerTests(unittest.TestCase):
    def _source(self, root):
        source = root / "src"
        (source / "web" / "ai_studio").mkdir(parents=True)
        (source / "__init__.py").write_text("VALUE = 1\n", encoding="utf-8")
        (source / "web" / "ai_studio" / "a.js").write_text("// one\n", encoding="utf-8")
        return source

    def test_js_change_changes_fingerprint(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first = self._source(root)
            before = node_pack_fingerprint(first)
            (first / "web" / "ai_studio" / "a.js").write_text("// two\n", encoding="utf-8")
            self.assertNotEqual(before, node_pack_fingerprint(first))
            (first / "web" / "ai_studio" / "b.css").write_text("a{}\n", encoding="utf-8")
            self.assertNotEqual(node_pack_fingerprint(first), node_pack_fingerprint(first, runtime_only=True))

    def test_js_change_refreshes_install_without_restart(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            custom_nodes = root / "custom_nodes"
            custom_nodes.mkdir()
            source = self._source(root)
            self.assertTrue(install_bundled_node_pack(custom_nodes, source=source).restart_required)
            self.assertEqual(
                node_pack_fingerprint(source, runtime_only=True),
                node_pack_fingerprint(source, runtime_only=True),
            )
            (source / "web" / "ai_studio" / "a.js").write_text("// two\n", encoding="utf-8")
            result = install_bundled_node_pack(custom_nodes, source=source)
            self.assertTrue(result.changed)
            self.assertFalse(result.restart_required)
            self.assertEqual(
                (result.target / "web" / "ai_studio" / "a.js").read_text(encoding="utf-8"), "// two\n",
            )


if __name__ == "__main__":
    unittest.main()
