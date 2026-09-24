# core/prompt_cleaner_prefs.py
"""프롬프트 클리너 옵션의 주인 — 키마다 한 곳 (감사 #29).

예전엔 옵션 다섯 개가 두 곳에서 왔다: Vue 설정의 토글(ui_prefs.cleanDuplicates/cleanSpaces/
cleanUnderscore)과 숨은 레거시 설정 탭이 prompt_settings.json 에 저장하는 cleaning_options.
저장 200ms 뒤와 백엔드 연결마다 도는 load_settings 가 레거시 값을 ``set_options(**...)`` 로
통째로 다시 적용해 Vue 에서 고른 값을 덮었다(중복 제거가 True → False 로 되돌아감).

이제:
- Vue 토글이 있는 세 옵션은 ui_prefs 가 주인이다. 키가 없으면 Vue 화면의 기본값(모두 켜짐)을
  쓴다 — 화면에 보이는 값과 실제 동작이 같다.
- Vue 토글이 없는 두 옵션(auto_comma, auto_escape)만 prompt_settings.cleaning_options 에 남는다.
"""
from __future__ import annotations

from typing import Any, Mapping, Optional

#: ui_prefs 키 → PromptCleaner 속성
UI_PREF_TO_CLEANER: dict[str, str] = {
    "cleanDuplicates": "remove_duplicates",
    "cleanSpaces": "auto_space",
    "cleanUnderscore": "underscore_to_space",
}

#: Vue 설정 화면(SettingsView)의 기본값 — 키가 저장된 적 없을 때 화면이 보여 주는 값
UI_PREF_DEFAULTS: dict[str, bool] = {
    "cleanDuplicates": True,
    "cleanSpaces": True,
    "cleanUnderscore": True,
}

#: Vue 토글이 없어 prompt_settings.cleaning_options 가 계속 주인인 옵션과 기본값
LEGACY_ONLY_DEFAULTS: dict[str, bool] = {
    "auto_comma": True,
    "auto_escape": False,
}


def cleaner_options_from_ui_prefs(prefs: Optional[Mapping[str, Any]]) -> dict[str, bool]:
    """ui_prefs → 클리너 속성 세 개. 키가 없거나 불리언이 아니면 Vue 기본값."""
    source = prefs if isinstance(prefs, Mapping) else {}
    options: dict[str, bool] = {}
    for pref_key, attr in UI_PREF_TO_CLEANER.items():
        value = source.get(pref_key)
        options[attr] = value if isinstance(value, bool) else UI_PREF_DEFAULTS[pref_key]
    return options


def legacy_cleaning_options(cleaning_options: Optional[Mapping[str, Any]]) -> dict[str, bool]:
    """prompt_settings.cleaning_options 에서 레거시 전용 두 옵션만 — 나머지 키는 무시한다."""
    source = cleaning_options if isinstance(cleaning_options, Mapping) else {}
    return {
        key: bool(source.get(key, default))
        for key, default in LEGACY_ONLY_DEFAULTS.items()
    }
