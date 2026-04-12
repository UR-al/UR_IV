# utils/character_presets.py
"""캐릭터별 커스텀 프리셋 저장/로드 (메모리 캐싱)"""
import os
import json

_FILE = os.path.join(os.path.dirname(os.path.dirname(__file__)), "character_presets.json")

# 모듈 레벨 캐시
_cache: dict | None = None


def _load() -> dict:
    global _cache
    if _cache is not None:
        return _cache
    if not os.path.exists(_FILE):
        _cache = {}
        return _cache
    try:
        with open(_FILE, "r", encoding="utf-8") as f:
            _cache = json.load(f)
    except Exception:
        _cache = {}
    return _cache


def _save(data: dict):
    global _cache
    _cache = data
    try:
        with open(_FILE, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"[CharPreset] 저장 실패: {e}")
        raise


def _normalize(name: str) -> str:
    return name.strip().lower().replace("_", " ")


def save_character_preset(name: str, extra_prompt: str,
                          cond_rules_json: str = ""):
    """캐릭터 프리셋 저장 (조건부 규칙 JSON 포함)"""
    data = _load()
    entry = {
        "extra_prompt": extra_prompt,
        "display_name": name,
    }
    if cond_rules_json:
        entry["cond_rules_json"] = cond_rules_json
    data[_normalize(name)] = entry
    _save(data)


def get_character_preset(name: str) -> str | None:
    """캐릭터 프리셋에서 extra_prompt 로드. 없으면 None."""
    data = _load()
    entry = data.get(_normalize(name))
    if entry:
        return entry.get("extra_prompt", "")
    return None


def get_character_preset_full(name: str) -> dict | None:
    """캐릭터 프리셋 전체 데이터 로드. 없으면 None.
    Returns: {extra_prompt, cond_rules_json, display_name}
    (기존 포맷: cond_rules, cond_neg_rules — 마이그레이션용)
    """
    data = _load()
    return data.get(_normalize(name))


def delete_character_preset(name: str):
    """캐릭터 프리셋 삭제"""
    data = _load()
    key = _normalize(name)
    if key in data:
        del data[key]
        _save(data)


def list_character_presets() -> dict[str, str]:
    """전체 프리셋 목록. {정규화이름: extra_prompt}"""
    data = _load()
    return {k: v.get("extra_prompt", "") for k, v in data.items()}


def has_preset(name: str) -> bool:
    """프리셋 존재 여부"""
    data = _load()
    return _normalize(name) in data
