"""studio_forge_noise + the samplers' ``forge_noise`` option: Forge Neo's noise in ComfyUI.

Forge Neo (Random Number Generator "CPU", the user's setting and Forge's default) draws each image's initial noise
from its own CPU generator seeded with that image's seed (seed + i), and seeds the global RNG with
``(last seed + 100000) % 65536`` while doing so; samplers that draw their step noise with ``torch.randn_like``
(sd_forge_neo_extra_samplers' EXP Heun 2 x0 SDE) use that global RNG.  ComfyUI's ``prepare_noise`` draws one
stream for the whole batch from ``seed``; its default noise sampler uses a generator seeded with the sampler seed.
The draws of both apps are restated here with plain torch calls (CPU, 64x64 latent).
"""
import sys
import types
import unittest
from types import SimpleNamespace
from unittest import mock

from comfy_custom_nodes.ai_studio_forge_parity import generation, studio_forge_noise
from tests._optional_deps import bind_torch, requires_torch


class SeedRules(unittest.TestCase):
    def test_step_noise_seed_is_forges_global_seed(self):
        self.assertEqual(studio_forge_noise.forge_step_noise_seed(42), (42 + 100000) % 65536)
        self.assertEqual(studio_forge_noise.forge_step_noise_seed(0), 100000 - 65536)
        self.assertEqual(studio_forge_noise.forge_step_noise_seed(2**40 + 7), (2**40 + 7 + 100000) % 65536)

    def test_image_seeds_count_up_like_forges_batch(self):
        self.assertEqual(studio_forge_noise.forge_image_seeds(10, 3), [10, 11, 12])
        self.assertEqual(studio_forge_noise.forge_image_seeds(10, 1), [10])

    def test_batch_index_picks_the_seed_of_that_image(self):
        self.assertEqual(studio_forge_noise.forge_image_seeds(10, 2, [3, 5]), [13, 15])


@requires_torch
class NoiseTensors(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        bind_torch(globals())

    def comfy_prepare_noise(self, latent, seed):
        # comfy/sample.py prepare_noise: one float32 CPU stream for the whole batch from seed.
        return torch.randn(latent.size(), dtype=torch.float32, generator=torch.Generator("cpu").manual_seed(seed))

    def forge_image_noise(self, seed, shape):
        # Forge rng.ImageRNG.first with the CPU source: the image's own generator seeded with its seed.
        return torch.randn(shape, generator=torch.Generator("cpu").manual_seed(seed))

    def test_one_image_is_already_the_same_in_both_apps(self):
        latent = torch.zeros(1, 16, 1, 64, 64)  # ComfyUI's Anima latent; Forge draws (16, 64, 64)
        ours = studio_forge_noise.forge_initial_noise(latent, [1234])
        self.assertTrue(torch.equal(ours, self.comfy_prepare_noise(latent, 1234)))
        self.assertTrue(torch.equal(ours.reshape(16, 64, 64), self.forge_image_noise(1234, (16, 64, 64))))

    def test_batch_images_get_their_own_seed_like_forge(self):
        latent = torch.zeros(2, 16, 1, 64, 64)
        ours = studio_forge_noise.forge_initial_noise(latent, [1234, 1235])
        self.assertTrue(torch.equal(ours[1].reshape(16, 64, 64), self.forge_image_noise(1235, (16, 64, 64))))
        comfy = self.comfy_prepare_noise(latent, 1234)
        self.assertTrue(torch.equal(ours[0], comfy[0]))
        self.assertFalse(torch.equal(ours[1], comfy[1]))  # why the option exists for batches

    def test_keeps_the_latent_dtype(self):
        latent = torch.zeros(1, 16, 1, 8, 8, dtype=torch.float16)
        self.assertEqual(studio_forge_noise.forge_initial_noise(latent, [1]).dtype, torch.float16)

    def test_step_noise_of_the_sampler_seed_equals_forges_global_draws(self):
        x = torch.zeros(1, 16, 1, 64, 64)
        seed = 1234
        with torch.random.fork_rng(devices=[]):
            # Forge: torch.manual_seed((seed + 100000) % 65536) during the initial noise, then the
            # sampler's default_noise_sampler draws torch.randn_like(x) from the global RNG each step.
            torch.manual_seed((seed + 100000) % 65536)
            forge = [torch.randn_like(x), torch.randn_like(x)]
        # ComfyUI's default_noise_sampler on a GPU latent: its own generator seeded with the sampler seed
        # (on a CPU latent ComfyUI adds 1 to the seed; the comparison here stands in for the GPU case).
        generator = torch.Generator(device=x.device).manual_seed(studio_forge_noise.forge_step_noise_seed(seed))
        comfy = [torch.randn(x.size(), dtype=x.dtype, generator=generator) for _ in range(2)]
        self.assertTrue(all(torch.equal(a, b) for a, b in zip(forge, comfy)))


def _comfy_stubs(received):
    comfy_stub = types.ModuleType("comfy")
    sample_stub = types.ModuleType("comfy.sample")
    sample_stub.fix_empty_latent_channels = lambda model, latent, *a: latent

    def prepare_noise(latent, seed, inds):
        received["prepare_noise"] = (seed, inds)
        return "comfy-noise"

    def sample(model, noise, *args, **kwargs):
        received["noise"], received["seed"] = noise, kwargs["seed"]
        return "samples"

    sample_stub.prepare_noise, sample_stub.sample = prepare_noise, sample
    utils_stub = types.ModuleType("comfy.utils")
    utils_stub.PROGRESS_BAR_ENABLED = False
    preview_stub = types.ModuleType("latent_preview")
    preview_stub.prepare_callback = lambda model, steps: None
    comfy_stub.sample, comfy_stub.utils = sample_stub, utils_stub
    return {"comfy": comfy_stub, "comfy.sample": sample_stub, "comfy.utils": utils_stub, "latent_preview": preview_stub}


@requires_torch
class CommonSampleOption(unittest.TestCase):
    """_common_sample with stubbed ComfyUI (it imports torch first, so torch is loaded before the stubs)."""

    @classmethod
    def setUpClass(cls):
        bind_torch(globals())

    def run_sample(self, latent, forge_noise, seed=77):
        received = {}
        with mock.patch.dict(sys.modules, _comfy_stubs(received)), \
                mock.patch.object(generation, "forge_initial_noise", side_effect=lambda lat, seeds: ("forge", seeds)):
            (output,) = generation._common_sample(
                "model", seed, 8, 3.0, "exp_heun_2_x0_sde", "simple", "p", "n", latent, 1.0, forge_noise=forge_noise,
            )
        return received, output

    def test_off_keeps_comfys_noise_and_seed(self):
        received, output = self.run_sample({"samples": SimpleNamespace(shape=(2, 16, 1, 8, 8))}, False)
        self.assertEqual(received["prepare_noise"], (77, None))
        self.assertEqual((received["noise"], received["seed"]), ("comfy-noise", 77))
        self.assertEqual(output["samples"], "samples")

    def test_on_uses_forges_image_seeds_and_global_seed(self):
        received, _ = self.run_sample({"samples": SimpleNamespace(shape=(2, 16, 1, 8, 8))}, True)
        self.assertNotIn("prepare_noise", received)
        self.assertEqual(received["noise"], ("forge", [77, 78]))
        self.assertEqual(received["seed"], (78 + 100000) % 65536)  # the batch's last seed, as Forge

    def test_on_follows_batch_index(self):
        latent = {"samples": SimpleNamespace(shape=(1, 16, 1, 8, 8)), "batch_index": [4]}
        received, _ = self.run_sample(latent, True)
        self.assertEqual(received["noise"], ("forge", [81]))
        self.assertEqual(received["seed"], (81 + 100000) % 65536)


class NodesPassTheOption(unittest.TestCase):
    def test_ksampler_input_and_pass_through(self):
        optional = generation.ForgeNeoKSamplerCNS.INPUT_TYPES()["optional"]
        self.assertEqual(optional["forge_noise"][0], "BOOLEAN")
        self.assertFalse(optional["forge_noise"][1]["default"])
        model = SimpleNamespace(model_options={})
        with mock.patch.object(generation, "_common_sample", side_effect=lambda *a, **k: (k,)) as sample, \
                mock.patch.object(generation, "with_pass_tag", side_effect=lambda m, tag: m):
            generation.ForgeNeoKSamplerCNS().sample(
                model, "p", "n", {"samples": None}, 1, 8, 3.0, "euler", "simple", forge_noise=True,
            )
            self.assertTrue(sample.call_args.kwargs["forge_noise"])
            generation.ForgeNeoKSamplerCNS().sample(model, "p", "n", {"samples": None}, 1, 8, 3.0, "euler", "simple")
            self.assertFalse(sample.call_args.kwargs["forge_noise"])

    def test_hires_input_and_pass_through(self):
        optional = generation.ForgeNeoHiresFix.INPUT_TYPES()["optional"]
        self.assertEqual(optional["forge_noise"][0], "BOOLEAN")
        self.assertFalse(optional["forge_noise"][1]["default"])
        comfy_stub = types.ModuleType("comfy")
        utils_stub = types.ModuleType("comfy.utils")
        utils_stub.common_upscale = lambda tensor, w, h, method, crop: ("upscaled", w, h)
        comfy_stub.utils = utils_stub
        samples = {"samples": SimpleNamespace(shape=(1, 16, 8, 10))}
        model = SimpleNamespace(model_options={})
        with mock.patch.dict(sys.modules, {"comfy": comfy_stub, "comfy.utils": utils_stub}), \
                mock.patch.object(generation, "_common_sample", side_effect=lambda *a, **k: (k,)) as sample, \
                mock.patch.object(generation, "with_pass_tag", side_effect=lambda m, tag: m):
            generation.ForgeNeoHiresFix().run(
                model, "p", "n", samples, seed=5, enabled=True, steps=4, shift=0.0, forge_noise=True,
            )
        self.assertTrue(sample.call_args.kwargs["forge_noise"])


if __name__ == "__main__":
    unittest.main()
