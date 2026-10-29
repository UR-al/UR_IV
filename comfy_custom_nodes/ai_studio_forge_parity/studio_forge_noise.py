"""Forge Neo's noise for the sampler / hires nodes' ``forge_noise`` option (the studio's "Forge처럼 노이즈").

What Forge Neo does (Settings -> Random Number Generator "CPU", its default and the user's; Eta noise seed delta 0),
restated from its behaviour (``modules/rng.py``), no Forge code:

- Image ``i`` of a batch has seed ``seed + i``; its initial noise is drawn on the CPU from a generator seeded with
  that seed alone.  ComfyUI's ``prepare_noise`` draws the whole batch as one stream from ``seed``, so the first
  image is the same in both apps and the others are not.
- While drawing each image's noise Forge also seeds torch's global RNG (all devices) with
  ``(image seed + 100000) % 65536``; the last image's call wins.  Samplers that draw their step noise with
  ``torch.randn_like`` on that global RNG (sd_forge_neo_extra_samplers' EXP Heun 2 x0 SDE and its SEEDS / SA
  family) therefore start from that seed.  ComfyUI's default noise sampler draws from a generator seeded with the
  sampler seed, so passing this value as the sampler seed gives the same draws on a GPU latent.

See ``studio_forge_parity.md`` for the comparison and its limits.
"""
from __future__ import annotations

from typing import Any, Optional, Sequence

from .compat import require_torch

FORGE_GLOBAL_SEED_OFFSET = 100000
FORGE_GLOBAL_SEED_MODULO = 65536


def forge_image_seeds(seed: int, batch_size: int, batch_index: Optional[Sequence[int]] = None) -> list[int]:
    """The seed of each image of the batch: ``seed + i`` (``i`` from ``batch_index`` when the latent has one)."""
    positions = list(batch_index) if batch_index is not None else range(int(batch_size))
    return [int(seed) + int(position) for position in positions]


def forge_step_noise_seed(last_image_seed: int) -> int:
    """The global seed Forge leaves behind after drawing the batch's initial noise."""
    return (int(last_image_seed) + FORGE_GLOBAL_SEED_OFFSET) % FORGE_GLOBAL_SEED_MODULO


def forge_initial_noise(latent_image: Any, seeds: Sequence[int]) -> Any:
    """Initial noise shaped like ``latent_image``: image ``i`` from a CPU generator seeded with ``seeds[i]``."""
    torch = require_torch()
    shape = list(latent_image.shape[1:])
    images = [
        torch.randn(shape, dtype=torch.float32, device="cpu", generator=torch.Generator("cpu").manual_seed(int(seed)))
        for seed in seeds
    ]
    return torch.stack(images).to(dtype=latent_image.dtype)


__all__ = ["forge_image_seeds", "forge_initial_noise", "forge_step_noise_seed"]
