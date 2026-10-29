"""Pack-side Forge LoRA tag rules, pinned to the app compiler's parse_lora_tags (golden comparison)."""
import unittest

from comfy_custom_nodes.ai_studio_forge_parity import studio_lora_tags as slt
from core import comfy_workflow_compiler as compiler

# (positive, negatives...) — the 12 cases the plan requires, plus Forge's named arguments and unusable numbers.
GOLDEN_INPUTS = [
    ("1girl, <lora:meow:0.8>, smile",),                              # basic
    ("<lora:meow:0.6:0.9> 1girl",),                                  # a:b -> clip a / model b
    ("<lora:meow> 1girl",),                                          # no value
    ("<lora :meow:0.7> 1girl",),                                     # space before the colon: literal text
    ("<3: 1girl <lora:meow:0.5>",),                                  # unclosed tag swallows the lora tag
    ("1girl <hypernet:foo:0.5>, <lora:meow:0.5>",),                  # other extra-network tag is removed
    ("1girl", "bad hands <lora:meow:0.4>, <lyco:x:1>, blurry"),      # tag in the negative
    ("<lora:에나 97:0.8>, 1girl",),                                   # Korean name with a space
    ("<lora:meow:0.75> 1girl",),                                     # decimal
    ("<lora:meow:-0.5> 1girl",),                                     # negative strength
    ("<lora:meow:1> 1girl <lora:meow:0.8>",),                        # same LoRA twice
    ("<lora:a:1>,<lora:b:0.5>,, <lora:c:0.3>, 1girl",),              # commas between tags
    ("<lora:meow:te=0.5:unet=0.9> 1girl",),                          # named arguments
    ("<lora:meow:abc:xyz> 1girl",),                                  # unusable numbers fall back
    ("<lora:meow:0.5@2:0.7> 1girl",),                                # '@' in the text encoder value
    ("<lora::0.5> 1girl",),                                          # tag without a name
    ("",),
]


def _as_tuples(specs):
    return [(s.name, s.strength_model, s.strength_clip) for s in specs]


class ParseLoraTagsTests(unittest.TestCase):
    def test_matches_app_compiler_rules(self):
        self.assertGreaterEqual(len(GOLDEN_INPUTS), 12)
        for args in GOLDEN_INPUTS:
            with self.subTest(args=args):
                expected_loras, expected_texts = compiler.parse_lora_tags(*args)
                loras, texts = slt.parse_lora_tags(*args)
                self.assertEqual(_as_tuples(loras), _as_tuples(expected_loras))
                self.assertEqual(texts, expected_texts)

    def test_clip_model_order_and_defaults(self):
        loras, texts = slt.parse_lora_tags("<lora:n:0.6:0.9> x <lora:m:0.5> <lora:k>")
        self.assertEqual(_as_tuples(loras), [("n", 0.9, 0.6), ("m", 0.5, 0.5), ("k", 1.0, 1.0)])
        self.assertEqual(texts, ["x"])


class CollapseDuplicatesTests(unittest.TestCase):
    def test_last_strength_wins_once_in_first_position(self):
        tags = [slt.LoraTag("a", 1.0, 1.0), slt.LoraTag("b", 0.5, 0.5), slt.LoraTag("a", 0.8, 0.8)]
        merged = slt.collapse_duplicates(tags)
        self.assertEqual(_as_tuples(merged), [("a", 0.8, 0.8), ("b", 0.5, 0.5)])


class ResolveLoraNameTests(unittest.TestCase):
    FILES = ["other.safetensors", "sub/Meow.safetensors", "enaa97 V2.safetensors", "dir\win.pt", "x/y/deep.ckpt"]

    def test_exact_and_extension_and_stem(self):
        r = slt.resolve_lora_name
        self.assertEqual(r("enaa97 V2.safetensors", self.FILES), "enaa97 V2.safetensors")
        self.assertEqual(r("enaa97 V2", self.FILES), "enaa97 V2.safetensors")
        self.assertEqual(r("meow", self.FILES), "sub/Meow.safetensors")       # stem in a subfolder, case-insensitive
        self.assertEqual(r("sub/meow", self.FILES), "sub/Meow.safetensors")
        self.assertEqual(r("win", self.FILES), "dir\win.pt")
        self.assertEqual(r("deep.ckpt", self.FILES), "x/y/deep.ckpt")

    def test_missing_returns_none(self):
        self.assertIsNone(slt.resolve_lora_name("nothing", self.FILES))
        self.assertIsNone(slt.resolve_lora_name("", self.FILES))
        self.assertIsNone(slt.resolve_lora_name("ee", self.FILES))  # no partial-stem matches


if __name__ == "__main__":
    unittest.main()
