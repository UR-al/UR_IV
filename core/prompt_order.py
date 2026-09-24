"""
Prompt Section Order — 최종 프롬프트 조립 시 섹션 순서를 사용자가 지정.

NAIA / WebUI / Forge 등 어떤 모델/스타일에서는 권장 순서가 다를 수 있어
강제 순서 대신 사용자가 선택할 수 있게 한다.

저장 파일: ``config/prompt_order.json``
형식:
    {"order": ["char_count", "character", ...], "version": 1}

기본 순서 (기존 generator_prompts.py의 하드코딩 순서):
    char_count → character → copyright → artist → prefix → main → suffix
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from utils.app_logger import get_logger
from utils.atomic_json import atomic_write_json

_logger = get_logger("prompt_order")


# 섹션 키 ↔ 사용자에게 보일 라벨
SECTION_LABELS: dict[str, str] = {
    "char_count": "인물 수",
    "character": "캐릭터",
    "copyright": "작품 (Copyright)",
    "artist": "작가 (Artist)",
    "prefix": "선행 (Prefix)",
    "main": "메인 태그",
    "suffix": "후행 (Suffix)",
}

# 기존 동작과 동일한 기본 순서
DEFAULT_ORDER: list[str] = [
    "char_count", "character", "copyright", "artist",
    "prefix", "main", "suffix",
]


def _config_path() -> Path:
    # 읽기 경로에서 폴더를 만들지 않는다 — 폴더 생성은 저장(atomic_write_json)이 맡는다.
    project_root = Path(__file__).resolve().parent.parent
    return project_root / "config" / "prompt_order.json"


def load_order() -> list[str]:
    """저장된 순서 로드. 없거나 파손 시 기본값. 누락 섹션은 끝에 추가, 미지원 키는 제거."""
    path = _config_path()
    if not path.is_file():
        return list(DEFAULT_ORDER)
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        _logger.warning(f"prompt_order.json 파손 — 기본값 사용")
        return list(DEFAULT_ORDER)
    raw = data.get("order") if isinstance(data, dict) else None
    if not isinstance(raw, list):
        return list(DEFAULT_ORDER)
    # 정제: 미지원 키 제외 + 누락 섹션은 끝에 추가
    valid = [k for k in raw if k in SECTION_LABELS]
    for k in DEFAULT_ORDER:
        if k not in valid:
            valid.append(k)
    return valid


def save_order(order: list[str]) -> bool:
    """순서 저장. 입력은 정제 후 기록."""
    # 정제 (모든 섹션을 포함시키되 사용자 순서 우선)
    cleaned: list[str] = []
    seen: set[str] = set()
    for k in order:
        if isinstance(k, str) and k in SECTION_LABELS and k not in seen:
            cleaned.append(k)
            seen.add(k)
    # 누락 섹션은 끝에 (사용자가 모르고 빠뜨려도 안전)
    for k in DEFAULT_ORDER:
        if k not in seen:
            cleaned.append(k)
    try:
        # 예전엔 write_text 로 제자리 덮어써서 쓰는 도중 종료되면 파일이 절단됐다.
        atomic_write_json(str(_config_path()), {"order": cleaned, "version": 1}, indent=2)
        return True
    except Exception:
        _logger.exception("prompt_order 저장 실패")
        return False


def reset_to_default() -> bool:
    """기본 순서로 복원."""
    return save_order(DEFAULT_ORDER)


def order_with_labels() -> list[dict[str, str]]:
    """현재 순서 + 라벨을 Vue로 전달할 dict 리스트로 반환."""
    return [{"key": k, "label": SECTION_LABELS.get(k, k)} for k in load_order()]


def compose_prompt(section_texts: dict[str, str], order: list[str] | None = None) -> str:
    """섹션 텍스트 dict + 순서 → 쉼표로 join된 최종 프롬프트.

    :param section_texts: ``{"char_count": "1girl", ...}`` — strip된 텍스트
    :param order: 사용할 순서. None이면 load_order() 결과.
    :return: 빈 섹션은 건너뛴 ", "-join 문자열
    """
    if order is None:
        order = load_order()
    parts: list[str] = []
    for key in order:
        v = section_texts.get(key, "")
        if isinstance(v, str) and v.strip():
            parts.append(v.strip())
    return ", ".join(parts)
