"""Studio nodes over optional packs (MOSS stream, Anima and Krea2 references) and the flux-type shift they rely on."""
import json
import unittest
from pathlib import Path
from unittest import mock

from comfy_custom_nodes.ai_studio_forge_parity import generation, studio_model_nodes as nodes


def fake_torch_zeros(shape):
    return shape


class ProviderCalls:
    """Stands in for ``invoke_provider``: records each call and returns the outputs a test gave that provider."""

    def __init__(self, **outputs):
        self.outputs = outputs
        self.calls = []

    def __call__(self, name, *, method=None, feature=None, args=(), kwargs=None):
        self.calls.append({"name": name, "method": method, "feature": feature, "args": args, "kwargs": kwargs or {}})
        result = self.outputs[name]
        return result(len(self.calls)) if callable(result) else result


class MossStreamTests(unittest.TestCase):
    def test_calls_the_toolkit_node(self):
        calls = ProviderCalls(MOSSStreamFP32=("streamed", "extra"))
        with mock.patch.object(nodes, "invoke_provider", calls):
            self.assertEqual(nodes.AIStudioMossStream().patch("model"), ("streamed",))
        self.assertEqual((calls.calls[0]["name"], calls.calls[0]["method"], calls.calls[0]["args"]),
                         ("MOSSStreamFP32", "patch", ("model",)))


class AnimaReferenceTests(unittest.TestCase):
    values = {"model": "m", "clip": "c", "vae": "v", "image": "i", "file": "sub/ref.safetensors", "mode": "char",
              "strength": 0.5, "guidance_on": True, "guidance": 4.5, "correction": "apg", "ref_end": 0.6,
              "megapixels": 1.25, "caption": "1girl"}

    def _apply(self, method, **outputs):
        calls = ProviderCalls(**outputs)
        with mock.patch.object(nodes, "invoke_provider", calls):
            result = nodes.AIStudioAnimaReference().apply(method=method, **self.values)
        return result, calls.calls

    def test_snapref_arguments(self):
        result, calls = self._apply("snapref", SnapRef=("patched",))
        self.assertEqual(result, ("patched",))
        self.assertEqual(calls[0]["name"], "SnapRef")
        self.assertEqual(calls[0]["kwargs"], {
            "model": "m", "clip": "c", "vae": "v", "image": "i", "adapter": "sub/ref.safetensors", "mode": "char",
            "ref_in_uncond": False, "ref_megapixels": 1.25, "strength": 0.5, "ref_caption": "1girl",
            "ref_guidance_on": True, "ref_guidance": 4.5, "correction": "apg", "ref_end": 0.6})

    def test_ref_adapter_arguments(self):
        result, calls = self._apply("ref_adapter", AnimaRefAdapter=("patched",))
        self.assertEqual(result, ("patched",))
        self.assertEqual(calls[0]["name"], "AnimaRefAdapter")
        self.assertEqual(calls[0]["kwargs"], {
            "model": "m", "vae": "v", "image": "i", "lora_name": "sub/ref.safetensors", "ref_guidance": 4.5,
            "correction": "apg", "strength": 0.5, "ref_end": 0.6, "ref_megapixels": 1.25})

    def test_empty_file_is_refused(self):
        with self.assertRaisesRegex(ValueError, "레퍼런스 파일"):
            nodes.AIStudioAnimaReference().apply(method="snapref", **{**self.values, "file": ""})


class Krea2ReferenceTests(unittest.TestCase):
    def test_bypass_order_and_lazy_prompts(self):
        spec = nodes.AIStudioKrea2Reference.INPUT_TYPES()["required"]
        self.assertEqual(list(spec)[:3], ["model", "positive", "negative"])  # a bypassed output takes its own slot
        self.assertEqual(nodes.AIStudioKrea2Reference.RETURN_TYPES, ("MODEL", "CONDITIONING", "CONDITIONING"))
        self.assertTrue(spec["positive"][1]["lazy"] and spec["negative"][1]["lazy"])
        self.assertEqual(nodes.AIStudioKrea2Reference().check_lazy_status(positive=None, negative=None), [])

    def _apply(self, **changes):
        values = {"model": "m", "clip": "c", "vae": "v", "image": "i", "text": "prompt", "negative_text": "bad",
                  "file": "krea2/QuadView_krea2_v1.safetensors", "strength": 0.9, "grounding_px": 1024, "boost": 1.5,
                  "position": "beside (right of target)", "width": 1536, "height": 1024, **changes}
        calls = ProviderCalls(LoraLoaderModelOnly=("lora model",), Krea2EditModelPatch=("edit model",),
                              Krea2EditGroundedEncode=lambda index: (f"cond {index}",))
        with mock.patch.object(nodes, "invoke_provider", calls), \
                mock.patch.object(nodes, "require_torch", return_value=mock.Mock(zeros=fake_torch_zeros)):
            return nodes.AIStudioKrea2Reference().apply(**values), calls.calls

    def test_lora_then_patch_then_both_prompts(self):
        result, calls = self._apply()
        self.assertEqual(result, ("edit model", "cond 3", "cond 4"))
        self.assertEqual([call["name"] for call in calls],
                         ["LoraLoaderModelOnly", "Krea2EditModelPatch", "Krea2EditGroundedEncode", "Krea2EditGroundedEncode"])
        self.assertEqual(calls[0]["args"], ("m", "krea2/QuadView_krea2_v1.safetensors", 0.9))
        patch = calls[1]["kwargs"]
        self.assertEqual(patch["model"], "lora model")
        self.assertEqual(patch["source_image"], "i")
        self.assertEqual(patch["vae"], "v")
        self.assertEqual(patch["fit_mode"], "fit")
        self.assertEqual((patch["ref_boost"], patch["ref_position"]), (1.5, "beside (right of target)"))
        self.assertEqual(patch["target_latent"]["samples"], (1, 16, 128, 192))  # the output size, in latent cells
        self.assertEqual(calls[2]["kwargs"], {"clip": "c", "prompt": "prompt", "image": "i", "grounding_px": 1024})
        self.assertEqual(calls[3]["kwargs"], {"clip": "c", "prompt": "bad", "image": "i", "grounding_px": 768})

    def test_empty_file_is_refused(self):
        with self.assertRaisesRegex(ValueError, "레퍼런스 파일"):
            self._apply(file="")


class FakeFlux:
    def __init__(self, config=None):
        self.config = config
        self.shift = None

    def set_parameters(self, shift=1.15):
        self.shift = shift


FakeFlux.__name__ = "ModelSamplingFlux"


class FakePatcher:
    def __init__(self, sampling):
        self.sampling = sampling
        self.model = mock.Mock(model_config="config")
        self.patches = {}

    def get_model_object(self, name):
        return self.patches.get(name, self.sampling) if name == "model_sampling" else None

    def clone(self):
        return FakePatcher(self.sampling)

    def add_object_patch(self, name, value):
        self.patches[name] = value


class FlowShiftTests(unittest.TestCase):
    def test_flux_type_sampling_keeps_its_class(self):
        class Advanced(FakeFlux):  # ComfyUI mixes the predictor into a local class, the core model_config one too
            pass

        original = Advanced()
        model = FakePatcher(original)
        with mock.patch.object(generation, "invoke_provider") as provider:
            patched = generation._patch_flow_shift(model, 1.5)
        provider.assert_not_called()  # ModelSamplingSD3 would multiply Krea2's timesteps by 1000
        sampling = patched.get_model_object("model_sampling")
        self.assertIsInstance(sampling, Advanced)
        self.assertEqual((sampling.config, sampling.shift), ("config", 1.5))
        self.assertIsNot(sampling, original)

    def test_discrete_flow_keeps_its_multiplier(self):
        class Discrete:
            multiplier = 1.0

        patched = FakePatcher(Discrete())
        with mock.patch.object(generation, "invoke_provider", return_value=(patched,)) as provider:
            generation._patch_flow_shift(patched, 5.0)
        self.assertEqual(provider.call_args.kwargs["args"][1:], (5.0, 1.0))


class ReferenceDocsTests(unittest.TestCase):
    def test_every_provider_is_named_in_the_parity_notes(self):
        notes = (Path(generation.__file__).parent / "studio_forge_parity.md").read_text(encoding="utf-8")
        for name in ("ComfyUI-MOSS-Toolkit", "SnapRef", "ComfyUI-Anima-Ref-Adapter", "comfyui-krea2edit"):
            with self.subTest(name=name):
                self.assertIn(name, notes)
        self.assertTrue(json.dumps(notes))


if __name__ == "__main__":
    unittest.main()
