# utils/character_presets.py
"""캐릭터별 커스텀 프리셋 저장/로드 (메모리 캐싱)"""
import os

from utils.atomic_json import atomic_write_json, load_json_safe

_FILE = os.path.join(os.path.dirname(os.path.dirname(__file__)), "user_data", "character_presets.json")

# 모듈 레벨 캐시
_cache: dict | None = None


def _load() -> dict:
    global _cache
    if _cache is not None:
        return _cache
    # 손상 시 .corrupt 백업 — 빈 캐시로 다음 저장이 덮어써 영구 소실되는 것 방지
    _cache = load_json_safe(_FILE, {})
    return _cache


def _save(data: dict):
    global _cache
    _cache = data
    try:
        atomic_write_json(_FILE, data)
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


def _clean_entry(key: str, value) -> tuple[str, dict] | None:
    """공유 파일의 항목 하나를 검증·정규화. 형식이 아니면 None."""
    if not isinstance(key, str) or not key.strip() or not isinstance(value, dict):
        return None
    extra = value.get("extra_prompt", "")
    if not isinstance(extra, str):
        return None
    entry = {"extra_prompt": extra, "display_name": str(value.get("display_name") or key).strip()}
    for opt in ("cond_rules_json", "cond_rules", "cond_neg_rules"):
        if isinstance(value.get(opt), str) and value.get(opt):
            entry[opt] = value[opt]
    return _normalize(key), entry


def export_character_presets() -> dict:
    """공유용 사본 — {정규화이름: 항목}."""
    import copy
    return copy.deepcopy(_load())


def import_character_presets(data, *, replace: bool = False) -> dict:
    """공유 파일을 가져온다. replace=False 면 병합(같은 캐릭터는 가져온 것으로 교체).

    예전 숨은 설정 탭은 파일을 직접 덮어써 모듈 캐시가 옛 값을 계속 돌려줬다 — 여기선 캐시와
    파일을 함께 갱신한다. 반환: {imported, total}.
    """
    if not isinstance(data, dict):
        raise ValueError("캐릭터 프리셋 파일 형식이 아닙니다(JSON 객체가 필요합니다)")
    cleaned: dict = {}
    for key, value in data.items():
        item = _clean_entry(key, value)
        if item is not None:
            cleaned[item[0]] = item[1]
    if not cleaned:
        raise ValueError("가져올 캐릭터 프리셋이 없습니다")
    merged = dict(cleaned) if replace else {**_load(), **cleaned}
    _save(merged)
    return {"imported": len(cleaned), "total": len(merged)}
