"""Anima studio web extension plumbing: route registry, WEB_DIRECTORY, installer fingerprint."""
import json
import os
import posixpath
import re
import subprocess
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from PIL import Image

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

    def test_web_imports_of_comfy_scripts_resolve_to_site_root(self):
        # JS is served at /extensions/<pack>/<rel path>; "scripts/*.js" must resolve to /scripts/*.js.
        web = Path(pack.__file__).parent / "web"
        found = 0
        for js in web.rglob("*.js"):
            rel = js.relative_to(web).as_posix()
            url_dir = posixpath.dirname("/extensions/ai_studio_forge_parity/" + rel)
            text = js.read_text(encoding="utf-8")
            for spec in re.findall(r"""from\s+["']([^"']*scripts/[^"']+)["']""", text):
                found += 1
                self.assertEqual(posixpath.normpath(posixpath.join(url_dir, spec)).startswith("/scripts/"), True,
                                 f"{rel}: {spec} resolves to {posixpath.normpath(posixpath.join(url_dir, spec))}")
        self.assertGreaterEqual(found, 1)

    def test_no_test_files_under_web_directory(self):
        web = Path(pack.__file__).parent / "web"
        names = [p.name for p in web.rglob("*.js") if ".test." in p.name or p.name.startswith("test_")]
        self.assertEqual(names, [])


PLAN_BODY = {
    "state": {"prompt": "1boy <lora:na-e10:1>", "seed": 5},
    "axes": [{"axis": "prompt_sr", "text": "na-e10, na-e14, na-e18, na-e20"},
             {"axis": "prompt_sr", "text": ":1>, :0.9>, :1.1>"},
             {"axis": "nothing", "text": ""}],
    "options": {"draw_legend": True, "keep_minus1": False, "vary_x": False, "vary_y": False, "vary_z": False,
                "include_sub_images": False, "include_sub_grids": False, "margin": 0, "row_count": 0},
}
INFOTEXT = "1boy\nSteps: 32, Sampler: Euler"


class _Dirs:
    """A fake ``folder_paths`` with real output / temp / input folders under a temporary directory."""

    def __init__(self, testcase):
        holder = tempfile.TemporaryDirectory()
        testcase.addCleanup(holder.cleanup)
        self.root = Path(holder.name).resolve()
        self.dirs = {name: self.root / name for name in ("output", "temp", "input")}
        for path in self.dirs.values():
            path.mkdir()
        self.module = SimpleNamespace(
            get_output_directory=lambda: str(self.dirs["output"]),
            get_temp_directory=lambda: str(self.dirs["temp"]),
            get_input_directory=lambda: str(self.dirs["input"]),
            get_directory_by_type=lambda kind: str(self.dirs[kind]) if kind in self.dirs else None,
        )
        patcher = mock.patch.dict("sys.modules", {"folder_paths": self.module})
        patcher.start()
        testcase.addCleanup(patcher.stop)

    def image(self, kind, name, size=(64, 64), color=(200, 30, 30), subfolder=""):
        folder = self.dirs[kind] / subfolder
        folder.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", size, color).save(folder / name)
        return {"filename": name, "subfolder": subfolder, "type": kind}


class StudioPlanRouteTests(unittest.TestCase):
    def test_plan_handler_roundtrip(self):
        status, payload = studio_routes.handle_plan(PLAN_BODY, families={})
        self.assertEqual(status, 200)
        self.assertEqual(payload["errors"], [])
        self.assertEqual(len(payload["cells"]), 12)
        self.assertEqual(set(payload["cells"][0]), {"ix", "iy", "iz", "state"})
        by_pos = {(c["ix"], c["iy"], c["iz"]): c["state"] for c in payload["cells"]}
        self.assertEqual(by_pos[(1, 2, 0)]["prompt"], "1boy <lora:na-e14:1.1>")
        self.assertEqual(sorted(payload["order"]), list(range(12)))
        self.assertEqual(payload["labels"]["x"], ["na-e10", "na-e14", "na-e18", "na-e20"])
        self.assertIn("infotext_extra", payload)
        json.dumps(payload)

    def test_plan_errors_come_back_with_status_200_and_no_cells(self):
        body = {**PLAN_BODY, "axes": [{"axis": "prompt_sr", "text": "zzz, b"}]}
        status, payload = studio_routes.handle_plan(body, families={})
        self.assertEqual(status, 200)
        self.assertEqual(payload["cells"], [])
        self.assertTrue(payload["errors"])

    def test_plan_handler_uses_the_unet_families(self):
        body = {"state": {}, "axes": [{"axis": "checkpoint", "text": "a-3.8b.safetensors, b.safetensors"}],
                "options": PLAN_BODY["options"]}
        with mock.patch.object(studio_routes.studio_choices, "unet_families",
                               return_value={"a-3.8b.safetensors": "3.8b", "b.safetensors": "2.9b"}):
            status, payload = studio_routes.handle_plan(body)
        self.assertEqual(status, 200)
        self.assertEqual(payload["cells"], [])
        self.assertIn(studio_routes.xyz_plan.MIXED_FAMILIES, payload["errors"])

    def test_plan_handler_bad_body_400(self):
        for body in (None, [], "x", {"state": [], "axes": [], "options": {}}, {"state": {}, "axes": "x", "options": {}},
                     {"state": {}, "axes": [], "options": 3}):
            status, payload = studio_routes.handle_plan(body, families={})
            self.assertEqual(status, 400, body)
            self.assertIn("error", payload)


class StudioGridRouteTests(unittest.TestCase):
    def setUp(self):
        self.dirs = _Dirs(self)

    def _body(self, **override):
        a = self.dirs.image("temp", "a.png", (64, 64), (255, 0, 0))
        b = self.dirs.image("output", "b.png", (64, 64), (0, 0, 255), subfolder="sub")
        body = {"nx": 2, "ny": 1, "nz": 1, "labels": {"x": ["a", "b"], "y": [""], "z": [""]},
                "options": {"draw_legend": True, "margin": 0, "row_count": 0, "include_sub_grids": False},
                "cells": [{"ix": 0, "iy": 0, "iz": 0, "image": a}, {"ix": 1, "iy": 0, "iz": 0, "image": b}],
                "infotext": INFOTEXT}
        body.update(override)
        return body

    def test_grid_handler_writes_png_with_parameters(self):
        status, payload = studio_routes.handle_grid(self._body())
        self.assertEqual(status, 200, payload)
        self.assertEqual(payload["subgrids"], [])
        grid = payload["grid"]
        self.assertEqual((grid["subfolder"], grid["type"]), ("AIStudio/xyz", "output"))
        self.assertRegex(grid["filename"], r"^xyz_grid_\d{5}\.png$")
        saved = self.dirs.dirs["output"] / "AIStudio" / "xyz" / grid["filename"]
        with Image.open(saved) as image:
            self.assertEqual(image.text["parameters"], INFOTEXT)
            self.assertGreaterEqual(image.width, 128)

    def test_grid_numbers_do_not_overwrite(self):
        first = studio_routes.handle_grid(self._body())[1]["grid"]["filename"]
        second = studio_routes.handle_grid(self._body())[1]["grid"]["filename"]
        self.assertEqual((first, second), ("xyz_grid_00001.png", "xyz_grid_00002.png"))

    def test_sub_grids_saved_only_when_asked(self):
        body = self._body(nz=2, labels={"x": ["a", "b"], "y": [""], "z": ["p", "q"]})
        body["cells"] = [{**c, "iz": z} for z in (0, 1) for c in body["cells"]]
        status, payload = studio_routes.handle_grid(body)
        self.assertEqual((status, payload["subgrids"]), (200, []))
        body["options"] = {**body["options"], "include_sub_grids": True}
        status, payload = studio_routes.handle_grid(body)
        self.assertEqual(status, 200, payload)
        self.assertEqual(len(payload["subgrids"]), 2)
        for sub in payload["subgrids"]:
            self.assertEqual((sub["subfolder"], sub["type"]), ("AIStudio/xyz", "output"))
            with Image.open(self.dirs.dirs["output"] / "AIStudio" / "xyz" / sub["filename"]) as image:
                self.assertEqual(image.text["parameters"], INFOTEXT)

    def test_failed_cell_is_drawn_black(self):
        body = self._body()
        body["cells"][1]["image"] = None
        self.assertEqual(studio_routes.handle_grid(body)[0], 200)

    def test_no_cell_image_is_400(self):
        body = self._body()
        for cell in body["cells"]:
            cell["image"] = None
        status, payload = studio_routes.handle_grid(body)
        self.assertEqual(status, 400)
        self.assertIn("no cell image", payload["error"])

    def test_bad_bodies_400(self):
        good = self._body()
        for patch in ({"nx": 0}, {"nx": "2"}, {"nz": 100000}, {"cells": "x"}, {"labels": []},
                      {"cells": [{"ix": 5, "iy": 0, "iz": 0, "image": None}]}, {"infotext": 5}, {"options": []}):
            status, payload = studio_routes.handle_grid({**good, **patch})
            self.assertEqual(status, 400, patch)
            self.assertIn("error", payload)
        self.assertEqual(studio_routes.handle_grid(None)[0], 400)

    def test_image_refs_cannot_leave_the_comfy_folders(self):
        secret = self.dirs.root / "secret.png"
        Image.new("RGB", (8, 8)).save(secret)
        (self.dirs.dirs["output"] / "ok.png").write_bytes(secret.read_bytes())
        refs = [
            {"filename": "../secret.png", "subfolder": "", "type": "output"},
            {"filename": "secret.png", "subfolder": "..", "type": "output"},
            {"filename": "secret.png", "subfolder": "sub/../..", "type": "temp"},
            {"filename": str(secret), "subfolder": "", "type": "output"},
            {"filename": "secret.png", "subfolder": str(self.dirs.root), "type": "output"},
            {"filename": "..\\secret.png", "subfolder": "", "type": "input"},
            {"filename": "secret.png", "subfolder": "", "type": "models"},
            {"filename": "secret.png", "subfolder": "", "type": ""},
            {"filename": "ok.png", "subfolder": "", "type": None},
            {"filename": "", "subfolder": "", "type": "output"},
            {"filename": "ok.png\0.png", "subfolder": "", "type": "output"},
            {"filename": ["ok.png"], "subfolder": "", "type": "output"},
            {"subfolder": "", "type": "output"},
        ]
        for ref in refs:
            body = self._body()
            body["cells"][0]["image"] = ref
            status, payload = studio_routes.handle_grid(body)
            self.assertEqual(status, 400, ref)
            self.assertIn("error", payload)
        ok = self._body()
        ok["cells"][0]["image"] = {"filename": "ok.png", "subfolder": "", "type": "output"}
        self.assertEqual(studio_routes.handle_grid(ok)[0], 200)

    def test_symlink_out_of_the_folder_is_refused(self):
        outside = self.dirs.root / "outside"
        outside.mkdir()
        Image.new("RGB", (8, 8)).save(outside / "x.png")
        link = self.dirs.dirs["output"] / "link"
        try:
            os.symlink(outside, link, target_is_directory=True)
        except (OSError, NotImplementedError):
            if os.name != "nt" or subprocess.run(["cmd", "/c", "mklink", "/J", str(link), str(outside)],
                                                  capture_output=True).returncode != 0:  # a junction needs no rights
                self.skipTest("symlinks are not available here")
        body = self._body()
        body["cells"][0]["image"] = {"filename": "x.png", "subfolder": "link", "type": "output"}
        self.assertEqual(studio_routes.handle_grid(body)[0], 400)

    def test_missing_file_and_non_image_extension_are_400(self):
        (self.dirs.dirs["output"] / "note.txt").write_text("hi", encoding="utf-8")
        for ref in ({"filename": "nope.png", "subfolder": "", "type": "output"},
                    {"filename": "note.txt", "subfolder": "", "type": "output"}):
            body = self._body()
            body["cells"][0]["image"] = ref
            self.assertEqual(studio_routes.handle_grid(body)[0], 400, ref)


class StudioFileRouteTests(unittest.TestCase):
    def setUp(self):
        self.dirs = _Dirs(self)

    def test_to_input_copies_without_overwrite(self):
        ref = self.dirs.image("output", "g.png", subfolder="AIStudio")
        results = [studio_routes.handle_to_input(ref) for _ in range(3)]
        self.assertEqual([r[0] for r in results], [200, 200, 200])
        self.assertEqual([r[1]["name"] for r in results], ["g.png", "g_1.png", "g_2.png"])
        for name in ("g.png", "g_1.png", "g_2.png"):
            self.assertTrue((self.dirs.dirs["input"] / name).is_file())
        (self.dirs.dirs["input"] / "g.png").write_bytes(b"keep")
        studio_routes.handle_to_input(ref)
        self.assertEqual((self.dirs.dirs["input"] / "g.png").read_bytes(), b"keep")

    def test_to_input_copies_temp_images_too(self):
        ref = self.dirs.image("temp", "preview.png")
        self.assertEqual(studio_routes.handle_to_input(ref), (200, {"name": "preview.png"}))

    def test_to_input_refuses_paths_outside_and_non_images(self):
        (self.dirs.root / "secret.png").write_bytes(b"x")
        (self.dirs.dirs["output"] / "notes.txt").write_text("hi", encoding="utf-8")
        for ref in ({"filename": "../secret.png", "subfolder": "", "type": "output"},
                    {"filename": "secret.png", "subfolder": "..", "type": "temp"},
                    {"filename": str(self.dirs.root / "secret.png"), "subfolder": "", "type": "output"},
                    {"filename": "secret.png", "subfolder": "", "type": "models"},
                    {"filename": "notes.txt", "subfolder": "", "type": "output"},
                    {"filename": "missing.png", "subfolder": "", "type": "output"}, None, []):
            status, payload = studio_routes.handle_to_input(ref)
            self.assertEqual(status, 400, ref)
            self.assertIn("error", payload)
        self.assertEqual(os.listdir(self.dirs.dirs["input"]), [])

    def test_open_output_starts_the_output_folder_on_windows(self):
        opened = []
        with mock.patch.object(studio_routes, "_is_windows", return_value=True), \
                mock.patch.object(os, "startfile", opened.append, create=True):
            status, payload = studio_routes.handle_open_output({})
        self.assertEqual((status, payload), (200, {"path": str(self.dirs.dirs["output"])}))
        self.assertEqual(opened, [str(self.dirs.dirs["output"])])

    def test_open_output_is_refused_elsewhere(self):
        opened = []
        with mock.patch.object(studio_routes, "_is_windows", return_value=False), \
                mock.patch.object(os, "startfile", opened.append, create=True):
            status, payload = studio_routes.handle_open_output({})
        self.assertEqual(status, 400)
        self.assertIn("error", payload)
        self.assertEqual(opened, [])

    def test_open_output_reports_a_failed_start(self):
        def boom(_path):
            raise OSError("no shell")

        with mock.patch.object(studio_routes, "_is_windows", return_value=True), \
                mock.patch.object(os, "startfile", boom, create=True):
            self.assertEqual(studio_routes.handle_open_output({})[0], 400)


class StudioChoicesRouteTests(unittest.TestCase):
    def test_choices_handler(self):
        with mock.patch.object(studio_routes.studio_choices, "choices", return_value={"items": ["a"]}) as choices:
            self.assertEqual(studio_routes.handle_choices({"kind": "vaes"}), (200, {"items": ["a"]}))
        choices.assert_called_once_with("vaes")

    def test_unknown_or_missing_kind_is_400(self):
        for query in ({"kind": "nope"}, {}, {"kind": ""}):
            status, payload = studio_routes.handle_choices(query)
            self.assertEqual(status, 400)
            self.assertIn("error", payload)


class StudioHandlerRegistrationTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        saved = dict(studio_routes._HANDLERS)
        self.addCleanup(lambda: (studio_routes._HANDLERS.clear(), studio_routes._HANDLERS.update(saved)))

    def test_all_routes_are_registered_at_import(self):
        # They are added while the pack imports (the server mounts what exists); none may be added lazily later.
        expected = {("GET", "/choices"), ("POST", "/xyz/plan"), ("POST", "/xyz/grid"),
                    ("POST", "/open_output"), ("POST", "/to_input")}
        self.assertEqual(set(studio_routes._HANDLERS), expected)
        studio_routes._HANDLERS.clear()
        studio_routes.register_studio_handlers()
        self.assertEqual(set(studio_routes._HANDLERS), expected)

    async def test_wrappers_run_the_handlers_and_answer_json(self):
        web = SimpleNamespace(json_response=lambda data, status=200: (data, status))

        class Request:
            def __init__(self, body=None, bad=False, query=None):
                self._body, self._bad = body, bad
                self.rel_url = SimpleNamespace(query=query or {})

            async def json(self):
                if self._bad:
                    raise ValueError("not json")
                return self._body

        with mock.patch.dict("sys.modules", {"aiohttp": SimpleNamespace(web=web)}):
            handlers = studio_routes._HANDLERS
            data, status = await handlers[("GET", "/choices")](Request(query={"kind": "dora_modes"}))
            self.assertEqual(status, 200)
            self.assertIn("items", data)
            data, status = await handlers[("POST", "/xyz/plan")](Request(PLAN_BODY))
            self.assertEqual((status, len(data["cells"])), (200, 12))
            data, status = await handlers[("POST", "/xyz/plan")](Request(bad=True))
            self.assertEqual(status, 400)
            self.assertIn("error", data)


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
