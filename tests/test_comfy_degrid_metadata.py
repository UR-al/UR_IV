"""VAE DeGrid in image metadata (L3-P6).

Forge writes ``Anima DeGrid model/mode/strength/tile/precision`` (success) or only
``Anima DeGrid error`` per image; PNG Info, Gallery and History group them with the
extensions.  A ComfyUI PNG carries the queued graph: an enabled
``ForgeNeoAnimaVAEDeGrid`` that reaches an output yields the same keys from its
inputs (the requested values — the node's runtime skip is not in the PNG; the
registry records that gap).  Tiny PNGs only, no model execution.
"""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from PIL import Image, PngImagePlugin

from core import vae_degrid as vdg
from core.comfy_metadata import parse_comfy_metadata
from core.image_metadata import extract_from_file, group_parameters, parse_infotext
from tests.test_image_metadata import graph_fixture

NODE = vdg.COMFY_NODE_CLASS


def degrid_node(image=("9", 0), **inputs) -> dict:
    values = {"image": list(image), "enabled": True, "model_name": "qwenVAEDegridNafnet_v11.safetensors",
              "mode": "Dark Pixels Mainly", "strength": 0.85, "tile": 384, "device": "auto", "precision": "fp32",
              "keep_loaded": False, "forge_quantize": True}
    values.update(inputs)
    return {"class_type": NODE, "inputs": values}


def graph_with(node=None, *, save_from="11") -> dict:
    graph = graph_fixture()
    if node is not None:
        graph["11"] = node
        graph["10"]["inputs"]["images"] = [save_from, 0]
    return graph


class ForgeInfotextGroupingTests(unittest.TestCase):
    def test_all_six_keys_group_under_extensions(self):
        text = ('a cat\nSteps: 20, Sampler: Euler, Seed: 7, Anima DeGrid model: qwenVAEDegridNafnet_v11, '
                'Anima DeGrid mode: Dark Pixels Mainly, Anima DeGrid strength: 0.85, Anima DeGrid tile: 384, '
                'Anima DeGrid precision: fp32, Lora hashes: "a: 1"')
        groups = group_parameters(parse_infotext(text).parameters)
        for key in (vdg.KEY_MODEL, vdg.KEY_MODE, vdg.KEY_STRENGTH, vdg.KEY_TILE, vdg.KEY_PRECISION):
            with self.subTest(key=key):
                self.assertIn(f"{key}: ", groups["extensions"])
                self.assertNotIn(key, groups["other"])
        self.assertEqual(groups["other"], 'Lora hashes: "a: 1"')

    def test_quoted_error_value_stays_one_extension_entry(self):
        error = "not a DeGrid residual model: corr=0.97, mean=0.31"
        text = f'a cat\nSteps: 20, Seed: 7, {vdg.KEY_ERROR}: {json.dumps(error)}'
        params = parse_infotext(text).parameters
        self.assertEqual(vdg.infotext_error(params), error)
        groups = group_parameters(params)
        self.assertIn(f'{vdg.KEY_ERROR}: "{error}"', groups["extensions"])
        self.assertEqual(groups["other"], "")


class ComfyGraphTests(unittest.TestCase):
    def test_enabled_node_reaching_the_output_gives_forge_keys(self):
        params = parse_comfy_metadata(graph_with(degrid_node()))["parameters"]
        self.assertEqual(params[vdg.KEY_MODEL], "qwenVAEDegridNafnet_v11")              # stem, like Forge
        self.assertEqual(params[vdg.KEY_MODE], "Dark Pixels Mainly")
        self.assertEqual(params[vdg.KEY_STRENGTH], "0.85")
        self.assertEqual(params[vdg.KEY_TILE], 384)
        self.assertNotIn(vdg.KEY_PRECISION, params)                                     # runtime value — unknown
        self.assertEqual(params["Seed"], 42)                                            # sampler params kept
        pasted = vdg.from_infotext(params)
        self.assertEqual((pasted.enabled, pasted.model, pasted.mode, pasted.strength, pasted.tile),
                         (True, "qwenVAEDegridNafnet_v11", "dark", 0.85, 384))

    def test_auto_and_folder_prefixed_names(self):
        for model, expected in (("auto", "auto"), ("upscale_models/anzhc.safetensors", "ESRGAN/anzhc"),
                                ("degrid/anzhc.pth", "DeGrid/anzhc")):
            with self.subTest(model=model):
                params = parse_comfy_metadata(graph_with(degrid_node(model_name=model)))["parameters"]
                self.assertEqual(params[vdg.KEY_MODEL], expected)
        auto = parse_comfy_metadata(graph_with(degrid_node(model_name="auto")))["parameters"]
        self.assertEqual(vdg.from_infotext(auto).model, vdg.AUTO)                       # pastes as automatic

    def test_disabled_absent_or_unreached_node_gives_nothing(self):
        cases = {
            "disabled": graph_with(degrid_node(enabled=False)),
            "absent": graph_with(None),
            "not reaching an output": graph_with(degrid_node(), save_from="9"),
            "no image link": graph_with(degrid_node(image=("x", 0))),
        }
        cases["no image link"]["11"]["inputs"]["image"] = "not a link"
        for name, graph in cases.items():
            with self.subTest(name):
                params = parse_comfy_metadata(graph)["parameters"]
                self.assertFalse([key for key in params if key.startswith("Anima DeGrid")], params)

    def test_two_nodes_with_different_settings_are_not_guessed(self):
        graph = graph_with(degrid_node())
        graph["12"] = degrid_node(image=("11", 0), mode="Full")
        graph["10"]["inputs"]["images"] = ["12", 0]
        result = parse_comfy_metadata(graph)
        self.assertFalse([key for key in result["parameters"] if key.startswith("Anima DeGrid")])
        self.assertTrue(any("DeGrid" in warning for warning in result["warnings"]))

    def test_png_from_comfy_shows_the_keys_in_the_extensions_group(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "comfy.png"
            info = PngImagePlugin.PngInfo()
            info.add_text("prompt", json.dumps(graph_with(degrid_node())))
            Image.new("RGB", (8, 8)).save(path, pnginfo=info)
            meta = extract_from_file(path)
        self.assertEqual(meta.parameters[vdg.KEY_MODE], "Dark Pixels Mainly")
        self.assertIn(f"{vdg.KEY_MODEL}: qwenVAEDegridNafnet_v11", group_parameters(meta.parameters)["extensions"])

    def test_workflow_only_png_reads_the_node_widgets(self):
        nodes = [
            {"id": 1, "type": "CLIPTextEncode", "widgets_values": ["pos"], "inputs": []},
            {"id": 2, "type": "CLIPTextEncode", "widgets_values": ["neg"], "inputs": []},
            {"id": 3, "type": "KSampler", "widgets_values": [314, "randomize", 24, 7, "euler", "normal", 1],
             "inputs": [{"name": "positive", "link": 11}, {"name": "negative", "link": 12}]},
            {"id": 4, "type": "VAEDecode", "inputs": [{"name": "samples", "link": 13}]},
            {"id": 5, "type": NODE, "inputs": [{"name": "image", "link": 14}],
             "widgets_values": [True, "auto", "Full", 1.0, 512, "auto", "fp32", False, True]},
            {"id": 6, "type": "SaveImage", "inputs": [{"name": "images", "link": 15}]},
        ]
        links = [[11, 1, 0, 3, 0, "CONDITIONING"], [12, 2, 0, 3, 1, "CONDITIONING"], [13, 3, 0, 4, 0, "LATENT"],
                 [14, 4, 0, 5, 0, "IMAGE"], [15, 5, 0, 6, 0, "IMAGE"]]
        params = parse_comfy_metadata(None, {"nodes": nodes, "links": links})["parameters"]
        self.assertEqual((params[vdg.KEY_MODEL], params[vdg.KEY_MODE], params[vdg.KEY_TILE]), ("auto", "Full", 512))


if __name__ == "__main__":
    unittest.main()
