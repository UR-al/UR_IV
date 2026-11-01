"""The post-processing field table: the JS twin, the node inputs it names, and the state it adds."""
import unittest

from comfy_custom_nodes.ai_studio_forge_parity.studio_post_schema import POST_SCHEMA, post_defaults
from comfy_custom_nodes.ai_studio_forge_parity.studio_state import DEFAULT_STATE, normalize_state
from tools import build_studio_post_schema as builder


class TwinTests(unittest.TestCase):
    def test_builder_reproduces_the_js_file(self):
        self.assertEqual(builder.TARGET.read_text(encoding="utf-8"), builder.render(POST_SCHEMA))


class FieldTests(unittest.TestCase):
    def fields(self):
        for section, fields in POST_SCHEMA.items():
            for field in fields:
                yield section, field

    def test_keys_are_unique_in_a_section(self):
        for section, fields in POST_SCHEMA.items():
            keys = [field["key"] for field in fields]
            self.assertEqual(len(keys), len(set(keys)), section)

    def test_defaults_fit_their_field(self):
        for section, field in self.fields():
            with self.subTest(section=section, key=field["key"]):
                if field["kind"] in ("int", "float"):
                    self.assertLessEqual(field["min"], field["default"])
                    self.assertLessEqual(field["default"], field["max"])
                elif field["kind"] == "choice" and field["choices"]:
                    self.assertIn(field["default"], field["choices"])

    def test_every_sam3_input_says_which_node_takes_it(self):
        for field in POST_SCHEMA["sam3"]:
            if field.get("input"):
                self.assertIn(field.get("node"), ("mask", "detailer", "both"), field["key"])


class NodeInputTests(unittest.TestCase):
    """A field's ``input`` must be a real input: a typo would be written into the prompt and ignored."""

    @staticmethod
    def inputs(node_class) -> set:
        types = node_class.INPUT_TYPES()
        return {*types.get("required", {}), *types.get("optional", {})}

    def test_inputs_exist_on_their_nodes(self):
        from comfy_custom_nodes.ai_studio_forge_parity import NODE_CLASS_MAPPINGS as nodes

        owners = {
            "hires": [nodes["ForgeNeoHiresFix"]],
            "upscale": [nodes["AIStudioUpscaleImage"]],
            "mask": [nodes["ForgeNeoSAM3Mask"]],
            "detailer": [nodes["ForgeNeoSAM3Detailer"]],
            "both": [nodes["ForgeNeoSAM3Mask"], nodes["ForgeNeoSAM3Detailer"]],
        }
        for section, fields in POST_SCHEMA.items():
            for field in fields:
                if not field.get("input"):
                    continue
                for node_class in owners[field.get("node") or section]:
                    with self.subTest(section=section, key=field["key"], node=node_class.__name__):
                        self.assertIn(field["input"], self.inputs(node_class))


class StateTests(unittest.TestCase):
    def test_every_field_is_in_the_default_state(self):
        for section in POST_SCHEMA:
            for key, value in post_defaults(section).items():
                self.assertEqual(DEFAULT_STATE[section][key], value, (section, key))

    def test_normalize_keeps_and_clamps_the_fields(self):
        state = normalize_state({
            "sam3": {"threshold": 7, "mode": "Mask only", "fill_mode": "nonsense", "convex_hull": True, "sampler": "euler"},
            "hires": {"steps": 0, "cfg": 4.5, "scheduler": "karras", "prompt": "1girl", "size_step": 1},
            "upscale": {"mode": "Scale to", "width": 9999999, "visibility": -1, "model_2": "2x.pth"},
        })
        self.assertEqual(state["sam3"]["threshold"], 1.0)
        self.assertEqual(state["sam3"]["mode"], "Mask only")
        self.assertEqual(state["sam3"]["fill_mode"], "original")
        self.assertTrue(state["sam3"]["convex_hull"])
        self.assertEqual(state["sam3"]["sampler"], "euler")
        self.assertEqual(state["hires"]["steps"], 0)
        self.assertEqual(state["hires"]["cfg"], 4.5)
        self.assertEqual(state["hires"]["scheduler"], "karras")
        self.assertEqual(state["hires"]["prompt"], "1girl")
        self.assertEqual(state["hires"]["size_step"], 8)
        self.assertEqual(state["upscale"]["width"], 8192)
        self.assertEqual(state["upscale"]["visibility"], 0.0)
        self.assertEqual(state["upscale"]["model_2"], "2x.pth")

    def test_a_saved_state_without_the_fields_gets_the_defaults(self):
        state = normalize_state({"sam3": {"on": True, "denoise": 0.5}})
        self.assertEqual(state["sam3"]["denoise"], 0.5)
        self.assertEqual(state["sam3"]["threshold"], 0.4)
        self.assertEqual(state["sam3"]["mode"], "Inpaint")


if __name__ == "__main__":
    unittest.main()
