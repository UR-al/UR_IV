"""prompt_settings.json 에서 Vue 위젯(프록시)이 주인이 아닌 키의 보관함 (audit #178).

예전엔 화면에 한 번도 붙지 않는 레거시 ``tabs/settings_tab.SettingsTab``(위젯 약 200개)이 이 값들을
숨은 위젯에 들고 있었다 — 불러올 때 JSON → 숨은 위젯, 저장할 때 숨은 위젯 → JSON. 그 왕복을 이
순수 값 객체가 맡는다(Qt 없음). 창(GeneratorMainUI)은 ``prompt_settings_extras`` 속성으로 들고 있다.

키                    의미 / 적용하는 곳
wildcard_enabled      파일 와일드카드 ON/OFF — ``utils.file_wildcard.wildcards_enabled`` 가 읽는다.
                      Vue 설정의 CheckBoxProxy 'wildcard_enabled' 가 이 값을 바꾼다.
cleaning_options      Vue 토글이 없는 클리너 두 옵션(auto_comma·auto_escape, core.prompt_cleaner_prefs).
font_family/size      남은 PyQt 창(Web/Backend 탭·네이티브 대화상자) QSS 글꼴 — ThemeManager.set_font.

:data:`RETIRED_KEYS` 는 더는 저장·복원하지 않는 옛 키다. 읽지 않고, 다음 저장 때 파일에서 빠진다.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Optional

from core.prompt_cleaner_prefs import LEGACY_ONLY_DEFAULTS, legacy_cleaning_options

#: ThemeManager 기본 글꼴(utils/theme_manager.DEFAULT_FONT_FAMILY 의 첫 이름·DEFAULT_FONT_SIZE) —
#: 이 모듈은 Qt·테마 모듈을 import 하지 않으므로 값을 따로 두고 테스트가 둘을 맞춰 본다.
DEFAULT_FONT_FAMILY_NAME = "Pretendard"
DEFAULT_FONT_SIZE_PT = 10.5
#: 예전 설정 탭 글꼴 크기 칸(QDoubleSpinBox)의 범위
FONT_SIZE_RANGE = (8.0, 20.0)

#: 저장·복원하지 않는 옛 키 → 이유. 파일에 남아 있어도 읽지 않는다.
RETIRED_KEYS: dict[str, str] = {
    "theme": "테마는 ui_prefs(theme/themeOverrides)가 주인이다 — 옛 콤보는 항목이 '모던' 하나였고 "
             "적용 메서드(set_theme)도 창에 없었다",
    "parquet_dir": "데이터셋 경로는 core.fetch_data.DATA_DIR(= config.PARQUET_DIR) 한 곳이다 — 숨은 입력칸이 "
                   "저장 때마다 절대경로를 적어, 설치 폴더를 옮기거나 다른 설치본의 백업을 가져오면 "
                   "검색·이벤트가 옛 경로를 봤다",
    "event_parquet_dir": "parquet_dir 와 같은 이유(config.EVENT_PARQUET_DIR 은 DATA_DIR/danbooru_sorted)",
    "cond_prompt_enabled": "읽는 곳이 저장·복원뿐이었다 — 조건부 프롬프트(config/cond_rules.json)는 늘 적용된다",
    "prefix_toggle": "늘 켜짐인 더미 토글이었다",
    "suffix_toggle": "늘 켜짐인 더미 토글이었다",
    "neg_toggle": "늘 켜짐인 더미 토글이었다",
    "exclude_toggle": "늘 켜짐인 더미 토글이었다",
    "res_presets": "프리셋 버튼이 없어 복원이 한 번도 돌지 않았다(Vue 랜덤 해상도는 random_resolutions)",
    "editor_defaults": "숨은 PyQt 에디터(MosaicEditor)와 함께 은퇴 — Vue 에디터 기본값은 tab_defaults.json",
    "bg_removal_model": "숨은 PyQt 에디터와 함께 은퇴 — Vue 배경 제거는 u2net 고정",
    "shortcuts": "utils/shortcut_manager 의 13개 단축키는 숨은 PyQt 에디터(MosaicEditor·InteractiveLabel)만 "
                 "읽었고 바꾸는 UI 는 숨은 설정 탭뿐이었다 — 둘 다 은퇴했고 Vue 단축키는 프론트가 정한다",
}


def _clamp_float(value: Any, default: float, low: float, high: float, digits: int) -> float:
    if isinstance(value, bool):
        return default
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        return default
    if number != number:  # NaN
        return default
    return round(max(low, min(high, number)), digits)


def _font_family(value: Any) -> str:
    text = str(value).strip() if isinstance(value, str) else ""
    return text or DEFAULT_FONT_FAMILY_NAME


def _font_size(value: Any) -> float:
    low, high = FONT_SIZE_RANGE
    return _clamp_float(value, DEFAULT_FONT_SIZE_PT, low, high, 2)


@dataclass
class PromptSettingsExtras:
    """Vue 위젯이 없는 prompt_settings 키 — :meth:`from_settings` 로 읽고 :meth:`to_settings` 로 쓴다."""

    wildcard_enabled: bool = True
    cleaning: dict[str, bool] = field(default_factory=lambda: dict(LEGACY_ONLY_DEFAULTS))
    font_family: str = DEFAULT_FONT_FAMILY_NAME
    font_size: float = DEFAULT_FONT_SIZE_PT

    @classmethod
    def from_settings(cls, settings: Optional[Mapping[str, Any]]) -> "PromptSettingsExtras":
        """prompt_settings.json 내용 → 값. 키가 없거나 형식이 틀리면 예전 설정 탭의 기본값."""
        source = settings if isinstance(settings, Mapping) else {}
        wildcard = source.get("wildcard_enabled", True)
        return cls(
            wildcard_enabled=wildcard if isinstance(wildcard, bool) else True,
            cleaning=legacy_cleaning_options(source.get("cleaning_options")),
            font_family=_font_family(source.get("font_family")),
            font_size=_font_size(source.get("font_size", DEFAULT_FONT_SIZE_PT)),
        )

    def to_settings(self) -> dict[str, Any]:
        """prompt_settings.json 에 쓸 키 — :meth:`from_settings` 와 왕복한다."""
        return {
            "wildcard_enabled": bool(self.wildcard_enabled),
            "cleaning_options": legacy_cleaning_options(self.cleaning),
            "font_family": _font_family(self.font_family),
            "font_size": _font_size(self.font_size),
        }


def extras_of(owner: Any) -> PromptSettingsExtras:
    """``owner.prompt_settings_extras`` — 없거나 다른 형식이면(부분 호스트·테스트 스텁) 기본값 한 벌."""
    extras = getattr(owner, "prompt_settings_extras", None)
    return extras if isinstance(extras, PromptSettingsExtras) else PromptSettingsExtras()


__all__ = [
    "DEFAULT_FONT_FAMILY_NAME",
    "DEFAULT_FONT_SIZE_PT",
    "FONT_SIZE_RANGE",
    "PromptSettingsExtras",
    "RETIRED_KEYS",
    "extras_of",
]
