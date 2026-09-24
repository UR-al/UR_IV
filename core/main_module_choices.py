"""메인 체크포인트 VAE / Text Encoder 선택지 — 연결 시와 Settings 경로 새로고침이 같이 쓰는 규칙.

예전엔 두 경로가 따로 목록을 만들었다. 연결 시(on_webui_info_loaded)는 ModelInventory 로
여러 루트(Forge 설정 폴더, 설치 폴더 models/text_encoder·text_encoders·clip·CLIP, data_root,
앱 폴백, ComfyUI 보조 라이브러리)를 보고 하위폴더 상대경로·API 원본 이름을 유지했지만,
Settings 의 model_paths.save/reset/refresh 는 Forge 설정 폴더 하나만 보고 파일명만 남긴 목록으로
같은 위젯을 덮어써서 — 목록에 없는 TE 선택은 교집합 필터로, VAE 는 'Use checkpoint default'
로 조용히 지워졌다(audit #156). 이제 둘 다 이 모듈을 쓴다.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Any

VAE_DEFAULT = "Use checkpoint default"
_VAE_PLACEHOLDERS = frozenset({"Use same VAE", VAE_DEFAULT})


def _runtime_names(entries: Iterable[Any]) -> list[str]:
    out: list[str] = []
    for entry in entries or []:
        name = str(entry.get('runtimeName') or '') if isinstance(entry, dict) else ''
        if name and name not in out:
            out.append(name)
    return out


def main_module_choices(inventory: Any, vae_api_items: Sequence[Any] | None) -> tuple[list[str], list[str]]:
    """(메인 VAE 콤보 항목, TE 칩 선택지).

    VAE 는 'Use checkpoint default' + 인벤토리(API 이름과 맞으면 API 원본 문자열) —
    인벤토리가 비면 API 목록으로 폴백한다. TE 는 인벤토리의 runtimeName(하위폴더 상대경로 유지).
    """
    api_items = list(vae_api_items) if vae_api_items is not None else None
    disk_vae: list[str] = []
    disk_te: list[str] = []
    if inventory is not None:
        disk_vae = _runtime_names(inventory.entries('vae', backend_items=api_items))
        disk_te = _runtime_names(inventory.entries('text_encoders'))
    disk_vae = [name for name in disk_vae if name not in _VAE_PLACEHOLDERS]
    if disk_vae:
        vae_items = [VAE_DEFAULT] + disk_vae
    else:
        vae_items = [VAE_DEFAULT]
        for raw in api_items or []:
            name = str(raw or '')
            if name and name not in _VAE_PLACEHOLDERS and name not in vae_items:
                vae_items.append(name)
    return vae_items, disk_te


def split_te_selection(text: str) -> list[str]:
    return [item.strip() for item in str(text or '').split(',') if item.strip()]


def filter_te_selection(text: str, available: Iterable[str]) -> str | None:
    """선택지에 없는 TE 를 뺀 새 텍스트 — 바뀌지 않으면 None."""
    allowed = set(available)
    current = split_te_selection(text)
    valid = [item for item in current if item in allowed]
    return None if valid == current else ', '.join(valid)


def resolve_te_selection(text: str, available: Iterable[str]) -> tuple[str, list[str]]:
    """다른 백엔드·설치에서 온 TE 선택(프리셋)을 지금 선택지에 맞춘다 — (새 텍스트, 뺀 항목).

    정확히 같거나 해시·경로·대소문자만 다른 같은 파일(core.model_names.match_checkpoint)은
    선택지 표기로 바꾸고, 선택지에 없는 파일은 뺀다. 연결 시 필터(filter_te_selection)와 같이
    없는 파일이 forge_additional_modules 로 흘러가지 않게 한다.
    """
    from core.model_names import match_checkpoint

    choices = [str(choice) for choice in available or () if str(choice or '')]
    kept: list[str] = []
    dropped: list[str] = []
    for item in split_te_selection(text):
        index = match_checkpoint(item, choices)
        if index < 0:
            dropped.append(item)
        elif choices[index] not in kept:
            kept.append(choices[index])
    return ', '.join(kept), dropped


def keep_vae_selection(current: str, items: Sequence[str]) -> str:
    """현재 VAE 선택을 유지하되 목록에 없으면 기본값."""
    return current if current in items else (items[0] if items else VAE_DEFAULT)
