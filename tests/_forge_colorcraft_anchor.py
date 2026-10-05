"""Forge's side of the Colorcraft anchor check (``tests.test_comfy_colorcraft_runtime``): run by Forge's own python.

Not a test module. ``ColorcraftOnRealComfyTests`` starts it as a child process with Forge's venv python on CPU
(``CUDA_VISIBLE_DEVICES=-1``, ``--cpu``, ``-B``, the working directory a temporary folder) and compares what it
writes with the anchors the app's pack makes on ComfyUI's VAE path. Read-only for Forge: no server, no webui
module, nothing written into the install.

Forge's VAE path, rebuilt from its own modules as ``backend/loader.py`` loads an Anima VAE (``backend.loader``
itself is not imported: it pulls in every engine and the webui's ``modules.shared``): ``WanVAE.load_config`` of
``backend/huggingface/circlestone-labs/Anima/vae``, built under ``no_init_weights`` and
``using_forge_operations(cpu, dtype, 'vae')``, the safetensors state dict loaded with ``load_state_dict``, wrapped
in ``backend.patcher.vae.VAE(model, is_wan=True)`` (``backend/diffusion_engine/anima.py``). Latent format:
``huggingface_guess.model_list.Anima.latent_format()``. Then the installed extension's
``sam3ext.colorcraft.hook.encode_anchors`` with the colours its chain needs for the Colorcraft argument given.

Variants: ``fp32`` (Forge's CPU VAE attention) and ``bf16_sdpa`` (``pytorch_attention_vae``, what the GPU runs use).

usage: python -B _forge_colorcraft_anchor.py <forge root> <extension root> <vae file> <colorcraft arg> <out .pt>
Writes ``{variant: {"anchors": {colour: tensor}, "latent": tensor, "vae_dtype", "attention"}, "_latent_format":
{"mean": tensor, "std": tensor}}`` with ``torch.save`` and prints one JSON summary line.
"""
from __future__ import annotations

import json
import os
import sys

VARIANTS = (("fp32", "float32", False), ("bf16_sdpa", "bfloat16", True))


def main(argv):
    forge, extension, vae_file, colorcraft_arg, out = argv[1:6]
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    sys.dont_write_bytecode = True
    sys.argv = [argv[0], "--cpu"]                      # backend.args parses the command line at import
    for path in (extension, forge, os.path.join(forge, "modules_forge", "packages")):
        sys.path.insert(0, path)

    import torch
    from transformers.modeling_utils import no_init_weights

    import backend.nn.wan_vae as forge_wan_vae
    from backend import attention as forge_attention
    from backend import memory_management
    from backend.operations import using_forge_operations
    from backend.patcher.vae import VAE
    from backend.state_dict import load_state_dict
    from backend.utils import load_torch_file
    from huggingface_guess import model_list
    from sam3ext.colorcraft import engine, hook, spec

    if memory_management.get_torch_device().type != "cpu":
        raise SystemExit("Forge's backend did not start on CPU - refusing to touch the GPU")
    chain = spec.build_chain(spec.config_from_args([colorcraft_arg]), False)
    colors = engine.needed_colors(chain.entries)
    latent_format = model_list.Anima.latent_format()
    repo = os.path.join(forge, "backend", "huggingface", "circlestone-labs", "Anima", "vae")
    state_dict = load_torch_file(vae_file)
    cpu_attention = forge_wan_vae.attention_function_vae
    result = {"_latent_format": {"mean": latent_format.latents_mean.clone(), "std": latent_format.latents_std.clone()}}
    summary = {}
    for name, dtype_name, sdpa in VARIANTS:
        dtype = getattr(torch, dtype_name)
        forge_wan_vae.attention_function_vae = forge_attention.pytorch_attention_vae if sdpa else cpu_attention
        config = forge_wan_vae.WanVAE.load_config(repo)
        with no_init_weights():
            with using_forge_operations(device=memory_management.cpu, dtype=dtype, extra_dtype="vae"):
                model = forge_wan_vae.WanVAE.from_config(config)
        load_state_dict(model, dict(state_dict))
        vae = VAE(model=model, is_wan=True, dtype=dtype)
        latents = []
        encode = vae.encode

        def keeping(image, _encode=encode, _latents=latents):
            latent = _encode(image)
            _latents.append(latent.detach().float().clone())
            return latent

        vae.encode = keeping
        hook.clear_anchor_cache()
        anchors = hook.encode_anchors(vae, latent_format, colors, {}, device=torch.device("cpu"))
        result[name] = {"anchors": {repr(tuple(key)): value.clone() for key, value in anchors.items()},
                        "latent": latents[0], "vae_dtype": str(vae.vae_dtype),
                        "attention": forge_wan_vae.attention_function_vae.__name__}
        summary[name] = {"vae_dtype": str(vae.vae_dtype), "attention": result[name]["attention"],
                         "anchors": len(anchors), "encodes": len(latents)}
        del vae, model
    torch.save(result, out)
    print(json.dumps({"ok": True, "torch": torch.__version__, "colors": [list(c) for c in colors],
                      "variants": summary}))


if __name__ == "__main__":
    main(sys.argv)
