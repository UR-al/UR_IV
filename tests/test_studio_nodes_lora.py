"""AIStudioLoraStack: strips Forge LoRA tags and applies them through ForgeNeoAnimaLoraLoader."""
import sys
import unittest
from types import SimpleNamespace
from unittest import mock

from comfy_custom_nodes.ai_studio_forge_parity import studio_nodes


class _FakeLoader:
    def __init__(self):
        self.calls = []

    def load_lora(self, model, clip, lora_name, strength_model, strength_clip):
        self.calls.append((lora_name, strength_model, strength_clip))
        return (f"{model}+{lora_name}", f"{clip}+{lora_name}")


def _folder_paths(files):
    return SimpleNamespace(get_filename_list=lambda kind: list(files) if kind == "loras" else [])


class LoraStackTests(unittest.TestCase):
    def _run(self, positive, negative="", files=("a.safetensors", "b.safetensors")):
        node = studio_nodes.AIStudioLoraStack()
        node._loader = _FakeLoader()
        with mock.patch.dict(sys.modules, {"folder_paths": _folder_paths(files)}):
            result = node.apply("M", "C", positive, negative)
        return node, result

    def test_stack_applies_last_strength_once(self):
        node, result = self._run("<lora:a:1> x <lora:a:0.8>")
        self.assertEqual(node._loader.calls, [("a.safetensors", 0.8, 0.8)])
        self.assertEqual(result, ("M+a.safetensors", "C+a.safetensors", "x", ""))

    def test_order_and_negative_tags(self):
        node, result = self._run("<lora:a:0.5> 1girl, <lora:b:0.6:0.9>", "<lora:b:0.2>, bad")
        self.assertEqual(
            node._loader.calls,
            [("a.safetensors", 0.5, 0.5), ("b.safetensors", 0.2, 0.2)],
        )
        self.assertEqual(result[2:], ("1girl", "bad"))
        self.assertEqual(result[0], "M+a.safetensors+b.safetensors")

    def test_no_tags_passes_through_without_loader(self):
        node, result = self._run("1girl, smile", "bad")
        self.assertEqual(node._loader.calls, [])
        self.assertEqual(result, ("M", "C", "1girl, smile", "bad"))

    def test_missing_lora_is_skipped_with_warning(self):
        with self.assertLogs("ai_studio_forge_parity", level="WARNING") as logs:
            node, result = self._run("<lora:ghost:0.7> <lora:a:1> x")
        self.assertEqual(node._loader.calls, [("a.safetensors", 1.0, 1.0)])
        self.assertEqual(result[2], "x")
        warnings = [line for line in logs.output if "ghost" in line]
        self.assertEqual(len(warnings), 1)

    def test_unreadable_lora_folder_is_an_error_not_every_tag_missing(self):
        def broken(kind):
            raise OSError("drive gone")

        node = studio_nodes.AIStudioLoraStack()
        node._loader = _FakeLoader()
        with mock.patch.dict(sys.modules, {"folder_paths": SimpleNamespace(get_filename_list=broken)}):
            with self.assertRaisesRegex(OSError, "drive gone"):
                node.apply("M", "C", "<lora:a:1> x", "")
            # no tag: the list is never read
            self.assertEqual(node.apply("M", "C", "x", ""), ("M", "C", "x", ""))
        self.assertEqual(node._loader.calls, [])

    def test_loader_instance_is_cached_on_the_node(self):
        node = studio_nodes.AIStudioLoraStack()
        first = node._get_loader()
        self.assertIs(node._get_loader(), first)
        self.assertEqual(type(first).__name__, "ForgeNeoAnimaLoraLoader")

    def test_node_contract(self):
        cls = studio_nodes.AIStudioLoraStack
        required = cls.INPUT_TYPES()["required"]
        self.assertEqual(list(required), ["model", "clip", "positive", "negative"])
        self.assertEqual(cls.RETURN_TYPES, ("MODEL", "CLIP", "STRING", "STRING"))
        self.assertIn("AIStudioLoraStack", studio_nodes.NODE_CLASS_MAPPINGS)
        self.assertIn("AIStudioLoraStack", studio_nodes.NODE_DISPLAY_NAME_MAPPINGS)


if __name__ == "__main__":
    unittest.main()
