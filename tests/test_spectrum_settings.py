import unittest
from unittest import mock

from core.spectrum_settings import spectrum_payload_from_prefs, validate_spectrum_payload
from comfy_custom_nodes.ai_studio_forge_parity import generation


class SpectrumSettingsTests(unittest.TestCase):
    def test_off_is_zero_effect(self):
        for prefs in ({}, {"comfySpectrum": {"enabled": False}}, {"comfySpectrum": {"enabled": "false"}}):
            self.assertEqual(spectrum_payload_from_prefs(prefs), {})

    def test_enabled_uses_bounded_defaults(self):
        payload = spectrum_payload_from_prefs({"comfySpectrum": {"enabled": True}})
        self.assertTrue(payload["spectrum_one_sampler_only"])
        validate_spectrum_payload({**payload, "steps": 28}, {"DiTSpectrumPatch": {}})

    def test_nonfinite_fractional_boolean_rejected(self):
        for value in (float('nan'), float('inf'), True, 1.5, -1, 10000):
            with self.subTest(value=value), self.assertRaises(ValueError):
                spectrum_payload_from_prefs({"comfySpectrum": {"enabled": True, "warmup_steps": value}})

    def test_missing_provider_and_no_cache_steps_rejected(self):
        for payload, info in (({"steps": 28}, {}), ({"steps": 9}, {"DiTSpectrumPatch": {}}),
                              ({"steps": 28, "speed_enabled": True}, {"DiTSpectrumPatch": {}})):
            with self.subTest(payload=payload), self.assertRaises(ValueError):
                validate_spectrum_payload({"spectrum_enabled": True, **payload}, info)

    def test_spectrum_patches_a_fresh_clone_not_the_cached_model(self):
        # ModelPatcher.clone() 가 model_options 컨테이너를 이미 격리하므로
        # 노드는 clone() 한 결과만 공급자에 넘기고 상류 MODEL 을 건드리지 않는다.
        class Model:
            def __init__(self):
                self.model_options = {"transformer_options": {"state": []}}
                self.clones = []

            def clone(self):
                clone = Model()
                self.clones.append(clone)
                return clone

        original = Model()
        received = []

        def fake_provider(name, *, method, feature, args=(), kwargs=None):
            received.append(args[0])
            args[0].model_options["transformer_options"]["state"].append(name)
            return (args[0],)

        with (
            mock.patch.object(generation, "invoke_provider", side_effect=fake_provider),
            mock.patch.object(generation, "_common_sample", side_effect=lambda model, *a, **k: (model,)),
        ):
            for _ in range(2):
                generation.ForgeNeoKSamplerCNS().sample(
                    original, "positive", "negative", {"samples": None}, 1, 28, 5.0,
                    "euler", "normal", spectrum_enabled=True,
                )
        self.assertEqual(received, original.clones)
        self.assertEqual(len(set(map(id, received))), 2)
        self.assertEqual(original.model_options["transformer_options"]["state"], [])


if __name__ == '__main__':
    unittest.main()
