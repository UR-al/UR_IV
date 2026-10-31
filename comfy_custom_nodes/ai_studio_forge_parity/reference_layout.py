"""Anima reference adapters on the model's block layout, the way plain LoRAs get there.

SnapRef and Anima Ref Adapter were trained on 28-block Anima base. On 2.9B and 3.8B only their first 28 blocks would load:
SnapRef then copies the reference and Ref Adapter gives noise. ``anima_lora.BLOCK_MAPPINGS`` holds the published block
lineage (each inserted block inherits the block before it) that plain LoRAs are remapped along, and an adapter file follows
it the same way. Both nodes read a file by name, so the remapped copy is saved once next to the original, with the
original's modified time: any other time on the copy means the original changed.
"""
from __future__ import annotations

import importlib
import os
import re
from pathlib import Path

from .anima_lora import BLOCK_MAPPINGS
from .compat import folder_paths_module

_BLOCK_KEY = re.compile(r"(?:^|\.)blocks\.(\d+)\.")


def remapped_name(file: str, blocks: int) -> str:
    """The copy's LoRA-list name, in the original's folder."""
    path = Path(file)
    return str(path.with_name(f"{path.stem}__{blocks}b{path.suffix}"))


def layout_blocks(keys) -> int:
    """The block layout of an adapter: the highest block + 1, since a converted file can leave inserted blocks empty."""
    return 1 + max((int(match.group(1)) for match in map(_BLOCK_KEY.search, keys) if match), default=-1)


def lineage_keys(keys, source_blocks: int, blocks: int) -> list[tuple[str, str]]:
    """``(new key, source key)`` for an adapter of ``source_blocks`` on ``blocks`` blocks; keys outside the blocks stay."""
    mapping = BLOCK_MAPPINGS[(source_blocks, blocks)]
    pairs = []
    for key in keys:
        match = _BLOCK_KEY.search(key)
        if match is None:
            pairs.append((key, key))
            continue
        pairs += [(f"{key[:match.start(1)]}{target}{key[match.end(1):]}", key)
                  for target, source in enumerate(mapping) if source == int(match.group(1))]
    return pairs


def _write_remapped(source: Path, target: Path, source_blocks: int, blocks: int) -> None:
    safetensors = importlib.import_module("safetensors")
    torch_io = importlib.import_module("safetensors.torch")
    tensors = torch_io.load_file(source)
    with safetensors.safe_open(source, "pt") as handle:
        metadata = handle.metadata() or {}
    # clones: safetensors refuses tensors sharing memory
    out = {new: tensors[old].clone() for new, old in lineage_keys(tensors, source_blocks, blocks)}
    partial = target.with_name(target.name + ".partial")
    torch_io.save_file(out, partial, metadata=metadata)
    partial.replace(target)


def _file_blocks(source: Path) -> int:
    with importlib.import_module("safetensors").safe_open(source, "pt") as handle:
        return layout_blocks(handle.keys())


def adapter_for_model(model, file: str) -> str:
    """``file``, or the name of its copy on this MODEL's block layout (written on first use)."""
    blocks = len(model.get_model_object("diffusion_model").blocks)
    source = Path(folder_paths_module().get_full_path_or_raise("loras", file))
    source_blocks = _file_blocks(source)
    if source_blocks == blocks:
        return file
    if (source_blocks, blocks) not in BLOCK_MAPPINGS:
        raise ValueError(f"레퍼런스 파일의 블록 수가 맞지 않음: {file} 는 {source_blocks}블록, 모델은 {blocks}블록")
    name = remapped_name(file, blocks)
    target = source.with_name(Path(name).name)
    stamp = source.stat()
    if not target.exists() or target.stat().st_mtime_ns != stamp.st_mtime_ns:
        _write_remapped(source, target, source_blocks, blocks)
        os.utime(target, ns=(stamp.st_atime_ns, stamp.st_mtime_ns))
    return name
