"""The Anima studio example workflow (UI workflow JSON v0.4) and its generator ``tools/build_anima_studio_workflow.py``.

The frontend restores a node from the file by position: ``inputs[i]`` overwrites the node's i-th input slot (sockets
first, then one slot per widget, in INPUT_TYPES order) and ``widgets_values`` fill the widgets in order, with an extra
``control_after_generate`` value after a ``seed`` widget (ComfyUI frontend 1.57.4 litegraphService/useIntWidget).
"""
from __future__ import annotations

import importlib.util
import inspect
import json
import re
import tempfile
import unittest
from pathlib import Path

from comfy_custom_nodes.ai_studio_forge_parity import NODE_CLASS_MAPPINGS

ROOT = Path(__file__).resolve().parents[1]
PACK = ROOT / "comfy_custom_nodes" / "ai_studio_forge_parity"
WORKFLOW = PACK / "example_workflows" / "AI Studio Anima.json"
ROLES_JS = PACK / "web" / "ai_studio" / "studio_roles.js"
BUILDER = ROOT / "tools" / "build_anima_studio_workflow.py"

CORE_TYPES = {"UNETLoader", "CLIPLoader", "VAELoader", "LoadImage", "VAEDecode", "VAEEncode"}
UI_LAYOUT = {
    "AIStudioPanel": ([0, 0], [790, 960]),
    "AIStudioResultView": ([806, 0], [680, 960]),
    "AIStudioXYZView": ([1502, 0], [400, 960]),
}
ROW_PREFIXES = ["①", "①b", "②", "③", "④", "⑤", "⑥", "⑦", "⑧", "⑨"]
POST_ROW = [
    ("post_image", "LoadImage"),
    ("post_upscale", "AIStudioUpscaleImage"),
    ("post_encode", "VAEEncode"),
    ("post_hires", "ForgeNeoHiresFix"),
    ("post_decode", "VAEDecode"),
    ("post_sam3_mask", "ForgeNeoSAM3Mask"),
    ("post_sam3_detailer", "ForgeNeoSAM3Detailer"),
    ("post_save", "AIStudioSaveForge"),
]
# studio_logic.stageModes(DEFAULT_STATE): 2.9B family, t2i, SAM3 off, no post run (R7).
DEFAULT_BYPASSED = {"unet38", "qwen35", "pos_encode38", "neg_encode38", "i2i_image", "sam3_mask", "sam3_detailer",
                    *(role for role, _ in POST_ROW)}
NODE_TITLE_HEIGHT = 30
WIDGET_TYPES = {"INT", "FLOAT", "STRING", "BOOLEAN"}


def _load_builder():
    spec = importlib.util.spec_from_file_location("build_anima_studio_workflow", BUILDER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _roles_from_js() -> list[str]:
    source = ROLES_JS.read_text(encoding="utf-8")
    body = re.search(r"export const ROLES = Object\.freeze\(\[(.*?)\]\);", source, re.S).group(1)
    body = re.sub(r"//[^\n]*", "", body)
    return re.findall(r'"([a-z0-9_]+)"', body)


def _frontend_slots(class_type: str):
    """(input names in slot order, widget names in widgets_values order) as the 1.57.4 frontend builds them."""
    spec = NODE_CLASS_MAPPINGS[class_type].INPUT_TYPES()
    entries = [*spec.get("required", {}).items(), *spec.get("optional", {}).items()]
    is_widget = [isinstance(value[0], list) or value[0] in WIDGET_TYPES for _, value in entries]
    sockets = [name for (name, _), widget in zip(entries, is_widget) if not widget]
    widget_inputs = [name for (name, value), widget in zip(entries, is_widget)
                     if widget and not (value[1] if len(value) > 1 else {}).get("socketless")]
    widgets = []
    for (name, _), widget in zip(entries, is_widget):
        if widget:
            widgets.append(name)
            if name in ("seed", "noise_seed"):
                widgets.append("control_after_generate")
    return sockets + widget_inputs, widgets


class WorkflowFileTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.workflow = json.loads(WORKFLOW.read_text(encoding="utf-8"))
        cls.nodes = {node["id"]: node for node in cls.workflow["nodes"]}
        cls.by_role = {node["properties"]["aistudio_role"]: node for node in cls.workflow["nodes"]
                       if "aistudio_role" in node.get("properties", {})}
        cls.links = {link[0]: link for link in cls.workflow["links"]}
        cls.engine = [node for node in cls.workflow["nodes"] if node["type"] not in UI_LAYOUT]

    # ---------------------------------------------------------------------------------------------- helpers

    def _source(self, node, input_name):
        """(source node, output slot) linked into ``node``'s input, or None."""
        entry = next(item for item in node["inputs"] if item["name"] == input_name)
        if entry["link"] is None:
            return None
        _, origin, slot, _, _, _ = self.links[entry["link"]]
        return self.nodes[origin], slot

    def _upstream_roles(self, node, input_name):
        found = set()
        pending = [self._source(node, input_name)]
        while pending:
            item = pending.pop()
            if item is None:
                continue
            source, _ = item
            found.add(source["properties"].get("aistudio_role"))
            pending.extend(self._source(source, entry["name"]) for entry in source.get("inputs", []))
        return found

    def _rows(self):
        groups = {group["title"].split()[0]: group for group in self.workflow["groups"]}
        rows = {}
        for prefix in ROW_PREFIXES:
            x, y, w, h = groups[prefix]["bounding"]
            members = [node for node in self.engine
                       if x <= node["pos"][0] and node["pos"][0] + node["size"][0] <= x + w
                       and y <= node["pos"][1] - NODE_TITLE_HEIGHT and node["pos"][1] + node["size"][1] <= y + h]
            rows[prefix] = sorted(members, key=lambda node: node["pos"][0])
        return groups, rows

    # ---------------------------------------------------------------------------------------------- tests

    def test_every_role_once(self):
        roles = _roles_from_js()
        self.assertEqual(len(roles), 33)
        self.assertNotIn("family_switch_clip", roles)  # R6
        found = [node["properties"]["aistudio_role"] for node in self.engine]
        self.assertCountEqual(found, roles)
        self.assertEqual(_load_builder().read_roles(), tuple(roles))
        for node in self.workflow["nodes"]:
            if node["type"] in UI_LAYOUT:
                self.assertNotIn("aistudio_role", node["properties"])

    def test_node_types_known(self):
        known = set(NODE_CLASS_MAPPINGS) | CORE_TYPES
        for node in self.workflow["nodes"]:
            with self.subTest(node=node["id"]):
                self.assertIn(node["type"], known)

    def test_opens_zoomed_to_fit_1920x1080(self):
        # spec section 7: the workflow opens in the C4 layout; at scale 1 the three columns (790 + 680 + 400 px,
        # 960 px tall) are cut off inside ComfyUI's chrome at 1920x1080 (final review M2), 0.9 fits.
        scale = self.workflow["extra"]["ds"]["scale"]
        self.assertTrue(0.85 <= scale <= 0.9, scale)
        self.assertLessEqual((790 + 680 + 400) * scale + self.workflow["extra"]["ds"]["offset"][0], 1920 - 120)

    def test_links_consistent(self):
        workflow = self.workflow
        ids = [link[0] for link in workflow["links"]]
        self.assertEqual(len(ids), len(set(ids)))
        self.assertEqual(workflow["last_link_id"], max(ids))
        self.assertEqual(workflow["last_node_id"], max(self.nodes))
        self.assertEqual(len(self.nodes), len(workflow["nodes"]))
        self.assertEqual(workflow["version"], 0.4)
        for link_id, origin, out_slot, target, in_slot, kind in workflow["links"]:
            with self.subTest(link=link_id):
                output = self.nodes[origin]["outputs"][out_slot]
                entry = self.nodes[target]["inputs"][in_slot]
                self.assertIn(link_id, output["links"])
                self.assertEqual(entry["link"], link_id)
                self.assertTrue(output["type"] == entry["type"] or "*" in (output["type"], entry["type"]))
                self.assertEqual(kind, entry["type"] if entry["type"] != "*" else output["type"])
                self.assertNotEqual(kind, "*")
        for node in workflow["nodes"]:
            for slot, entry in enumerate(node.get("inputs", [])):
                if entry["link"] is not None:
                    self.assertEqual(self.links[entry["link"]][3:5], [node["id"], slot])
            for slot, output in enumerate(node.get("outputs", [])):
                for link_id in output["links"] or []:
                    self.assertEqual(self.links[link_id][1:3], [node["id"], slot])

    def test_inputs_follow_frontend_slot_order(self):
        for node in self.engine:
            if node["type"] in CORE_TYPES or node["type"].startswith("ForgeNeoAnima"):
                continue  # not inspectable outside ComfyUI
            with self.subTest(role=node["properties"]["aistudio_role"]):
                names, _ = _frontend_slots(node["type"])
                self.assertEqual([entry["name"] for entry in node["inputs"]], names)
                for entry in node["inputs"]:
                    spec_widget = "widget" in entry
                    self.assertEqual(spec_widget, entry["type"] in WIDGET_TYPES | {"COMBO"})
                    if spec_widget:
                        self.assertEqual(entry["widget"], {"name": entry["name"]})

    def test_anima38_inputs_match_the_node_functions(self):
        for role in ("unet38", "qwen35", "pos_encode38", "neg_encode38"):
            node = self.by_role[role]
            cls = NODE_CLASS_MAPPINGS[node["type"]]
            params = list(inspect.signature(getattr(cls, cls.FUNCTION)).parameters)[1:]
            with self.subTest(role=role):
                self.assertEqual([entry["name"] for entry in node["inputs"]], params)
                self.assertEqual([output["type"] for output in node["outputs"]], list(cls.RETURN_TYPES))

    def test_widget_values_follow_frontend_order(self):
        written = {node["properties"]["aistudio_role"]: node for node in self.engine if "widgets_values" in node}
        # Nodes whose COMBO choices depend on the ComfyUI install are left to their defaults (R20).
        for role in ("unet", "te", "vae", "unet38", "qwen35", "pos_encode38", "i2i_image", "sched", "upscale",
                     "sam3_mask", "post_upscale", "post_image"):
            self.assertNotIn(role, written)
        values = {}
        for role, node in written.items():
            _, widgets = _frontend_slots(node["type"])
            with self.subTest(role=role):
                self.assertEqual(len(node["widgets_values"]), len(widgets))
            values[role] = dict(zip(widgets, node["widgets_values"]))
        sampler = values["sampler"]
        self.assertEqual((sampler["steps"], sampler["cfg"], sampler["sampler_name"], sampler["scheduler"]),
                         (32, 3.0, "exp_heun_2_x0_sde", "flow_cosmos_dynamic"))
        self.assertEqual((sampler["denoise"], sampler["forge_noise"], sampler["control_after_generate"]),
                         (1.0, True, "fixed"))
        for role in ("hires", "post_hires"):
            hires = values[role]
            self.assertEqual((hires["enabled"], hires["scale_by"], hires["upscale_method"], hires["steps"],
                              hires["denoise"], hires["sampler_name"], hires["base_steps"]),
                             (False, 1.5, "4x-UltraSharp.pth", 16, 0.4, "exp_heun_2_x0_sde", 32))
        latent = values["latent"]
        self.assertEqual((latent["mode"], latent["width"], latent["height"], latent["batch_size"], latent["mask_blur"]),
                         ("txt2img", 832, 1216, 1, 4))
        self.assertEqual(values["shift"]["shift"], 3.0)
        self.assertEqual(values["dora"], {"enabled": True, "mode": "Forge/Comfy (stock)"})  # R2: Forge처럼 on
        self.assertEqual((values["pos_encode"]["steps"], values["pos_encode"]["forge_syntax"]), (32, True))
        detailer = values["sam3_detailer"]
        self.assertEqual((detailer["enabled"], detailer["denoise"], detailer["steps"], detailer["cfg"]),
                         (False, 0.4, 32, 3.0))
        state = json.loads(values["save"]["studio_state"])
        self.assertEqual((state["steps"], state["sampler"], state["seed"]), (32, "exp_heun_2_x0_sde", -1))
        self.assertEqual({values[f"family_switch_{x}"]["select"] for x in ("model", "pos", "neg")}, {0})

    def test_both_families_reach_sampler(self):
        sampler = self.by_role["sampler"]
        families = {
            "model": ("family_switch_model", {"unet"}, {"unet38"}),
            "positive": ("family_switch_pos", {"pos_encode"}, {"pos_encode38"}),
            "negative": ("family_switch_neg", {"neg_encode"}, {"neg_encode38"}),
        }
        for input_name, (switch_role, base_roles, roles38) in families.items():
            with self.subTest(input=input_name):
                switch = self.by_role[switch_role]
                self.assertEqual(switch["type"], "AIStudioSwitch")
                self.assertIn(switch_role, self._upstream_roles(sampler, input_name))
                self.assertTrue(base_roles <= self._upstream_roles(switch, "a"))
                self.assertTrue(roles38 <= self._upstream_roles(switch, "b"))
                for role in base_roles | roles38:  # every path from a family node goes through its switch
                    targets = {self.links[link_id][3] for link_id in self.by_role[role]["outputs"][0]["links"]}
                    self.assertEqual(targets, {switch["id"]})
        self.assertEqual(self._source(self.by_role["pos_encode38"], "model")[0], self.by_role["dora"])  # R6
        self.assertEqual(self._source(self.by_role["pos_encode38"], "native_clip")[0], self.by_role["lora_stack"])
        self.assertEqual(self._source(self.by_role["lora_stack"], "clip")[0], self.by_role["te"])  # one shared TE

    def test_engine_rows_top_to_bottom(self):
        groups, rows = self._rows()
        self.assertEqual([group["title"].split()[0] for group in self.workflow["groups"]], ROW_PREFIXES)
        bounds = [groups[prefix]["bounding"] for prefix in ROW_PREFIXES]
        self.assertGreaterEqual(bounds[0][1], 1040)
        for upper, lower in zip(bounds, bounds[1:]):
            self.assertLess(upper[1] + upper[3], lower[1])
        placed = [node["id"] for prefix in ROW_PREFIXES for node in rows[prefix]]
        self.assertCountEqual(placed, [node["id"] for node in self.engine])
        for prefix in ROW_PREFIXES:
            row = rows[prefix]
            self.assertTrue(row, prefix)
            for left, right in zip(row, row[1:]):
                self.assertLess(left["pos"][0] + left["size"][0], right["pos"][0])
        self.assertEqual([node["properties"]["aistudio_role"] for node in rows["①b"]][-1], "family_switch_model")  # R6
        self.assertEqual([node["properties"]["aistudio_role"] for node in rows["③"]][-2:],
                         ["family_switch_pos", "family_switch_neg"])

    def test_links_flow_one_way(self):
        _, rows = self._rows()
        row_of = {node["id"]: index for index, prefix in enumerate(ROW_PREFIXES) for node in rows[prefix]}
        for link_id, origin, _, target, _, _ in self.workflow["links"]:
            with self.subTest(link=link_id):
                if row_of[origin] == row_of[target]:
                    self.assertLess(self.nodes[origin]["pos"][0], self.nodes[target]["pos"][0])
                else:
                    self.assertLess(row_of[origin], row_of[target])

    def test_post_row_matches_post_routes(self):
        # R19: studio_logic.js POST_ROUTES re-link slot-0 IMAGE outputs along exactly this row.
        _, rows = self._rows()
        row = [(node["properties"]["aistudio_role"], node["type"]) for node in rows["⑨"]]
        self.assertEqual(row, POST_ROW)
        image_links = [
            ("post_upscale", "image", "post_image"),
            ("post_encode", "pixels", "post_upscale"),
            ("post_hires", "samples", "post_encode"),
            ("post_decode", "samples", "post_hires"),
            ("post_sam3_mask", "image", "post_decode"),
            ("post_sam3_detailer", "image", "post_decode"),
            ("post_save", "images", "post_sam3_detailer"),
        ]
        for role, input_name, source_role in image_links:
            with self.subTest(role=role, input=input_name):
                source, slot = self._source(self.by_role[role], input_name)
                self.assertEqual((source["properties"]["aistudio_role"], slot), (source_role, 0))
        for role in ("post_image", "post_upscale", "post_decode", "post_sam3_detailer"):
            self.assertEqual(self.by_role[role]["outputs"][0]["type"], "IMAGE")

    def test_ui_nodes_layout(self):
        for class_type, (pos, size) in UI_LAYOUT.items():
            node = next(node for node in self.workflow["nodes"] if node["type"] == class_type)
            with self.subTest(node=class_type):
                self.assertEqual((node["pos"], node["size"]), (pos, size))
                self.assertFalse(node["flags"].get("collapsed", False))
                self.assertEqual(node["inputs"], [])
                self.assertEqual(node["outputs"], [])
                self.assertEqual(len(node["widgets_values"]), 1)
                json.loads(node["widgets_values"][0])
        panel = next(node for node in self.workflow["nodes"] if node["type"] == "AIStudioPanel")
        from comfy_custom_nodes.ai_studio_forge_parity.studio_state import normalize_state

        self.assertEqual(json.loads(panel["widgets_values"][0]), normalize_state({}))

    def test_engine_room_folded_and_default_stages_bypassed(self):
        for node in self.engine:
            role = node["properties"]["aistudio_role"]
            with self.subTest(role=role):
                self.assertTrue(node["flags"]["collapsed"])
                self.assertEqual(node["mode"], 4 if role in DEFAULT_BYPASSED else 0)

    def test_builder_reproduces_file(self):
        builder = _load_builder()
        with tempfile.TemporaryDirectory() as folder:
            out = Path(folder) / "out.json"
            builder.write_workflow(out)
            text = out.read_bytes()
        self.assertFalse(text.startswith(b"\xef\xbb\xbf"))
        self.assertEqual(json.loads(text.decode("utf-8")), self.workflow)


if __name__ == "__main__":
    unittest.main()
