# Pinned upstream manifest

## Source

- Repository: `https://github.com/Extraltodeus/Skimmed_CFG.git`
- Pinned commit: `d83005832ac42783adfd6f4ae96f6ef6406d1a74`
- Commit subject: `Merge pull request #22 from Comfy-Forge-Node-Hardening/addlicense`
- Upstream package version at that commit: `1.0.0` (`pyproject.toml`)
- Source-code license: Apache-2.0 (full text: `LICENSES/Skimmed_CFG-Apache-2.0.txt`;
  upstream ships no `NOTICE` file)

The two files below were copied from that Git revision without any code change.
Only the line endings differ: the upstream blobs store CRLF, and the copies are
LF, which `vendor/.gitattributes` enforces for every `*.py` in this directory.
The first SHA-256 is computed over the copied (LF) bytes, and the second over
the upstream blob.

| Vendored path | SHA-256 (LF copy) | SHA-256 (upstream CRLF blob) |
| --- | --- | --- |
| `__init__.py` | `263b752ecca74b1ae9587931ed7a02860dbd60559cd1bb3c8c458386282303c2` | `8207a91e72292ac5ec8720387f94cd70555be1e36f0a3e2390dd6a60da924a9c` |
| `skimmed_CFG.py` | `a7471ebe04fd8925d8634d26e9488508a389d6dd4b67c88cd1794833c98aadde` | `4e4da887ffb4407efa4fb966bd0d8547dcc112d76a1b83d1611bce2f0e0429c9` |

## Local integration boundary

`UPSTREAM.md` is the only file in this directory that AI Studio wrote. AI Studio
does not edit the copied code. The integration code is one level up, in
`guidance_skim.py`: `ForgeNeoSkimmedCFG` imports `skimmed_CFG.py` when the node
runs, maps its inputs to those of the upstream `CFG_Skimming_Single_Scale_Pre_CFG`
and calls that node's `execute` unchanged. The upstream code reads
`model_sampling`, clones the model and registers its `pre_cfg_patch` with
`set_model_sampler_pre_cfg_function`, as the original ComfyUI node does. The
upstream module imports `torch` and ComfyUI's `comfy_api.latest`. It is never
imported when the pack loads, and its `comfy_entrypoint` is never registered,
so the other upstream nodes (the Timed flip, Clean Skim, replace, linear
interpolation and difference variants) are not exposed by this pack.

The following upstream repository files were not copied because they are
repository metadata, documentation, packaging metadata or a sample workflow
image, not code that is imported at runtime:

- `.github/workflows/publish.yml`
- `LICENSE` (copied as `LICENSES/Skimmed_CFG-Apache-2.0.txt`)
- `README.md`
- `everything_workflow.png`
- `pyproject.toml`
