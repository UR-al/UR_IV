"""AIStudioPromptEncode: plain CLIPTextEncode when forge_syntax is off, Forge prompt editing as timestep ranges when on.

ComfyUI is stood in by a fake ``nodes`` module whose CLIPTextEncode / ConditioningSetTimestepRange do what
ComfyUI 0.43.1's do (tokenize + encode_from_tokens_scheduled; copy each entry and set start/end_percent).
"""
import sys
import unittest
from types import SimpleNamespace
from unittest import mock

from comfy_custom_nodes.ai_studio_forge_parity import studio_nodes


class _FakeClip:
    def __init__(self):
        self.encoded = []

    def tokenize(self, text):
        return {"text": text}

    def encode_from_tokens_scheduled(self, tokens):
        self.encoded.append(tokens["text"])
        return [[f"cond:{tokens['text']}", {"pooled_output": None}]]


class _CLIPTextEncode:
    def encode(self, clip, text):
        return (clip.encode_from_tokens_scheduled(clip.tokenize(text)),)


class _ConditioningSetTimestepRange:
    def set_range(self, conditioning, start, end):
        return ([[t[0], {**t[1], "start_percent": start, "end_percent": end}] for t in conditioning],)


def _fake_nodes():
    return SimpleNamespace(NODE_CLASS_MAPPINGS={
        "CLIPTextEncode": _CLIPTextEncode,
        "ConditioningSetTimestepRange": _ConditioningSetTimestepRange,
    })


class PromptEncodeTests(unittest.TestCase):
    def encode(self, text, steps=20, forge_syntax=True):
        clip = _FakeClip()
        with mock.patch.dict(sys.modules, {"nodes": _fake_nodes()}):
            (conditioning,) = studio_nodes.AIStudioPromptEncode().encode(clip, text, steps, forge_syntax)
        return clip, conditioning

    def test_disabled_equals_plain_encode(self):
        clip, conditioning = self.encode("[cat:dog:0.5]", forge_syntax=False)
        self.assertEqual(clip.encoded, ["[cat:dog:0.5]"])
        self.assertEqual(conditioning, [["cond:[cat:dog:0.5]", {"pooled_output": None}]])

    def test_two_segments_set_ranges(self):
        clip, conditioning = self.encode("[cat:dog:0.5]", steps=20)
        self.assertEqual(clip.encoded, ["cat", "dog"])
        self.assertEqual(conditioning, [
            ["cond:cat", {"pooled_output": None, "start_percent": 0.0, "end_percent": 0.5}],
            ["cond:dog", {"pooled_output": None, "start_percent": 0.5, "end_percent": 1.0}],
        ])

    def test_one_segment_is_left_as_it_is(self):
        clip, conditioning = self.encode("1girl, (smile:1.2)")
        self.assertEqual(clip.encoded, ["1girl, (smile:1.2)"])
        self.assertEqual(conditioning, [["cond:1girl, (smile:1.2)", {"pooled_output": None}]])

    def test_same_segment_text_is_encoded_once(self):
        clip, conditioning = self.encode("[cat|dog]", steps=4)
        self.assertEqual(clip.encoded, ["cat", "dog"])
        self.assertEqual(
            [(entry[0], entry[1]["start_percent"], entry[1]["end_percent"]) for entry in conditioning],
            [("cond:cat", 0.0, 0.25), ("cond:dog", 0.25, 0.5), ("cond:cat", 0.5, 0.75), ("cond:dog", 0.75, 1.0)],
        )

    def test_ranges_do_not_share_one_dict(self):
        _, conditioning = self.encode("[cat|dog]", steps=4)
        self.assertIsNot(conditioning[0][1], conditioning[2][1])

    def test_node_interface(self):
        cls = studio_nodes.AIStudioPromptEncode
        required = cls.INPUT_TYPES()["required"]
        self.assertEqual(list(required), ["clip", "text", "steps", "forge_syntax"])
        self.assertEqual(required["clip"], ("CLIP",))
        self.assertEqual(required["text"][0], "STRING")
        self.assertEqual(required["steps"][0], "INT")
        self.assertEqual(required["steps"][1]["default"], 32)
        self.assertEqual(required["forge_syntax"], ("BOOLEAN", {"default": True}))
        self.assertEqual(cls.RETURN_TYPES, ("CONDITIONING",))
        self.assertIn("AIStudioPromptEncode", studio_nodes.NODE_CLASS_MAPPINGS)
        self.assertIn("AIStudioPromptEncode", studio_nodes.NODE_DISPLAY_NAME_MAPPINGS)


if __name__ == "__main__":
    unittest.main()
