# utils/theme_manager.py
"""PyQt 위젯용 색 — 원본은 ``core/theme_presets`` 다.

**왜 여기서 색을 정의하지 않는가**: Vue(`:root` CSS 변수)와 PyQt(QSS·인라인 스타일)가
각자 색표를 들고 있으면, 사용자가 테마를 바꿨을 때 앱이 두 개처럼 보인다.
그래서 프리셋이 단일 출처이고 이 모듈은 **번역기**다 —
하이픈 토큰(`bg-primary`)을 기존 호출부의 밑줄 키(`bg_primary`)로 옮긴다.

**키 이름을 왜 그대로 두는가**: `get_color('bg_primary')` 호출부가 수백 군데인데,
없는 키는 예외가 아니라 회색 `#888888` 로 조용히 떨어진다. 이름을 바꾸면 화면이
회색으로 얼룩지면서도 아무도 에러를 못 본다. 그래서 이름은 유지하고 값만 바꿨다.

**상태색은 글자용(-fg)을 기본으로 준다**: `get_color('error')` 호출부는 거의 전부
`color:` 로 쓴다. 배지 *배경*이 필요하면 `error_fill` 을 쓴다 — 채움색(`#D14141`)은
흰 글자와 4.6:1 을 맞춘 값이라 어두운 바탕에 글자로 쓰면 미달한다.
"""
from __future__ import annotations

import os

from core.theme_presets import (
    DEFAULT_PRESET,
    PRESETS,
    resolve,
    shift_lightness,
)

DEFAULT_FONT_FAMILY = "'Pretendard', 'Malgun Gothic', sans-serif"
DEFAULT_FONT_SIZE = "10.5pt"

# (옛 테마 '이름' API — LEGACY_THEME_NAME·THEMES·MODERN_THEME·current_theme_name·available_themes —
#  는 숨은 레거시 설정 탭의 테마 콤보와 _get_tab_title 만 썼다. 둘 다 은퇴해 지웠다(audit #175·#178).
#  색은 프리셋(ui_prefs.theme/themeOverrides)이 정한다.)

#: 테마 선택이 저장되는 곳은 ui_prefs.json — Vue 설정과 같은 파일이라 두 쪽이 같은 값을 본다.
#: 경로는 core.ui_prefs.ui_prefs_path() 한 곳에서 (읽을 때마다) 정한다.

#: 밑줄 키 → 프리셋 토큰. 값을 그대로 옮기는 것만 여기 둔다.
_TOKEN_ALIASES: dict[str, str] = {
    # ── 배경 ──
    'bg_primary': 'bg-primary',
    'bg_secondary': 'bg-secondary',
    'bg_card': 'bg-card',
    # 'bg_tertiary' 는 원래 '한 단계 올라온 면'이었다 = 지금의 카드.
    'bg_tertiary': 'bg-card',
    'bg_input': 'bg-input',
    'bg_button': 'bg-button',
    'bg_button_hover': 'bg-button-hover',
    'bg_tab_selected': 'bg-card',
    'bg_splitter': 'bg-secondary',
    'bg_status_bar': 'bg-primary',
    'scrollbar_bg': 'bg-primary',
    'scrollbar_handle': 'bg-button-hover',
    'disabled_bg': 'bg-secondary',
    # ── 글자 ──
    'text_primary': 'text-primary',
    'text_secondary': 'text-secondary',
    'text_muted': 'text-muted',
    'text_tab': 'text-muted',
    'text_tab_selected': 'text-primary',
    # 비활성 글자는 muted 보다 더 죽어야 한다 — 경계선 강조색이 그 자리다.
    'disabled_text': 'border-strong',
    # ── 경계 ──
    'border': 'border',
    'border_strong': 'border-strong',
    'rule': 'rule',
    'edge': 'edge',
    'border_input_focus': 'accent',
    # ── 강조 ──
    'accent': 'accent',
    'accent_hover': 'accent-hover',
    'accent_fill': 'accent-fill',
    'accent_fill_hover': 'accent-fill-hover',
    'on_accent': 'on-accent',
    'primary_btn_bg_hover': 'accent-fill-hover',
    'primary_btn_text_disabled': 'text-muted',
    # ── 상태: 기본은 글자용, 채움은 *_fill ──
    'success': 'state-ok-fg',
    'error': 'state-alert-fg',
    'warning': 'state-warn-fg',
    'info': 'state-info-fg',
    'success_fill': 'state-ok',
    'error_fill': 'state-alert',
    'warning_fill': 'state-warn',
    'info_fill': 'state-info',
}

#: 색이 아니라 치수. 테마와 무관하게 고정.
_LAYOUT: dict[str, str] = {
    'bg_tab': 'transparent',
    'radius_base': '6px',
    'radius_card': '8px',
    'radius_button': '8px',
    'radius_primary_btn': '10px',
    'tab_padding': '10px 18px',
    'tab_margin': '3px 6px',
}


def _build_colors(preset_name: str, overrides: dict | None = None) -> dict:
    """프리셋 토큰 → 기존 밑줄 키 색표."""
    tokens = resolve(preset_name, overrides)
    is_dark = tokens.get('mode') != 'light'

    colors = {legacy: tokens[token] for legacy, token in _TOKEN_ALIASES.items()}
    colors.update(_LAYOUT)

    # 아래 셋은 프리셋에 토큰이 없다(웹에서는 CSS filter/opacity 로 푸는 것들).
    # QSS 는 그런 게 없어 명도만 밀어 만든다.
    colors['bg_button_pressed'] = shift_lightness(tokens['bg-button'], -0.04)
    # 상태 채움(*_fill) 위의 글자. 네 채움색 모두 흰 글자와 4.5:1 이상으로 잡힌 값이라
    # 강조색과 달리 밝기 판정이 필요 없다 — on_accent 를 여기 쓰면 안 된다.
    colors['on_state_fill'] = '#FFFFFF'
    # 비활성 주 버튼의 면 — 강조색을 알아볼 만큼은 남기고 죽인다.
    dim = shift_lightness(tokens['accent'], -0.30 if is_dark else 0.30)
    colors['accent_dim'] = dim
    colors['primary_btn_bg_disabled'] = dim
    # 그라디언트는 옛 호출부 호환용. 새 코드는 accent_fill(평면)을 쓴다.
    fill = tokens['accent-fill']
    colors['accent_gradient'] = (
        'qlineargradient(x1:0, y1:0, x2:1, y2:0, '
        f'stop:0 {fill}, stop:1 {shift_lightness(fill, -0.08)})'
    )

    # 밝기 판정이 필요한 호출부용(색 아님). QSS format 은 남는 키를 무시한다.
    colors['mode'] = tokens['mode']
    return colors


_QSS_TEMPLATE = """
    QWidget {{
        background-color: {bg_primary};
        color: {text_primary};
        font-family: {font_family};
        font-size: {font_size};
    }}

    QMainWindow {{
        background-color: {bg_primary};
    }}

    QSplitter::handle {{
        background-color: transparent;
        width: 1px;
    }}

    /* ── GroupBox: 카드 기반 레이아웃 ── */
    QGroupBox {{
        background-color: {bg_secondary};
        border: 1px solid {border};
        border-radius: {radius_card};
        margin-top: 18px;
        padding: 35px 12px 12px 12px;
    }}
    QGroupBox::title {{
        subcontrol-origin: margin;
        subcontrol-position: top left;
        left: 18px;
        top: 12px;
        color: {text_secondary};
        font-weight: 600;
        font-size: 8.5pt;
        text-transform: uppercase;
        letter-spacing: 1px;
    }}

    /* ── 입력 필드 ── */
    QLineEdit, QTextEdit, QComboBox {{
        background-color: {bg_input};
        border: 1px solid {border};
        border-radius: {radius_base};
        padding: 6px 10px;
        color: {text_primary};
        selection-background-color: {accent_fill};
        selection-color: {on_accent};
    }}
    QLineEdit:focus, QTextEdit:focus, QComboBox:focus {{
        border: 1px solid {accent};
        background-color: {bg_secondary};
    }}

    QComboBox::drop-down {{
        border: none;
        width: 24px;
    }}

    /* ── 버튼 ── */
    QPushButton {{
        background-color: {bg_button};
        border: 1px solid {border};
        border-radius: {radius_button};
        color: {text_primary};
        padding: 6px 12px;
        font-weight: 500;
        min-height: 24px;
    }}
    QPushButton:hover {{
        background-color: {bg_button_hover};
        border: 1px solid {text_secondary};
    }}
    QPushButton:pressed {{
        background-color: {bg_button_pressed};
    }}
    QPushButton:checked {{
        background-color: {accent_fill};
        color: {on_accent};
        border: 1px solid {accent_fill};
    }}
    QPushButton:disabled {{
        background-color: {disabled_bg};
        color: {disabled_text};
        border: 1px solid {border};
    }}

    /* ── 생성 버튼: 면은 accent_fill, 글자는 on_accent ──
       (사용자가 아무 색이나 고를 수 있어 accent 를 그대로 깔면 글자가 안 읽힌다) */
    QPushButton#primaryButton {{
        background: {accent_fill};
        color: {on_accent};
        border: none;
        border-radius: {radius_primary_btn};
        font-weight: 600;
        font-size: 11pt;
        padding: 10px;
        min-height: 36px;
    }}
    QPushButton#primaryButton:hover {{
        background: {primary_btn_bg_hover};
    }}
    QPushButton#primaryButton:disabled {{
        background: {primary_btn_bg_disabled};
        color: {primary_btn_text_disabled};
    }}

    /* ── 스크롤바 ── */
    QScrollArea {{ border: none; background: transparent; }}
    QScrollBar:vertical {{
        background: transparent;
        width: 4px;
        margin: 0px;
    }}
    QScrollBar::handle:vertical {{
        background: {scrollbar_handle};
        min-height: 40px;
        border-radius: 2px;
    }}
    QScrollBar::handle:vertical:hover {{
        background: {accent};
    }}
    QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical,
    QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical {{
        height: 0px; background: none;
    }}

    /* ── 탭 ── */
    QTabWidget::pane {{ border: none; background-color: {bg_primary}; }}
    QTabBar {{
        alignment: center;
    }}
    QTabBar::tab {{
        background: {bg_button};
        color: {text_tab};
        padding: {tab_padding};
        margin: {tab_margin};
        font-weight: 500;
        border-radius: {radius_button};
        border: 1px solid {border};
        font-size: 9pt;
        min-width: 50px;
    }}
    QTabBar::tab:selected {{
        background: {bg_tertiary};
        color: {text_tab_selected};
        border: 1px solid {accent};
    }}
    QTabBar::tab:hover {{
        background: {bg_button_hover};
        color: {text_primary};
    }}

    QLabel {{
        color: {text_secondary};
        background: transparent;
    }}

    /* ── 체크박스: 기본 윈도우 스타일 유지 ── */
    QCheckBox {{
        color: {text_primary};
        spacing: 8px;
    }}

    QProgressBar {{
        background-color: {bg_input};
        border: none;
        border-radius: 2px;
        text-align: center;
        color: {text_primary};
        height: 4px;
    }}
    QProgressBar::chunk {{
        background-color: {accent};
    }}

    QSlider::groove:horizontal {{
        height: 4px;
        background: {bg_input};
        border-radius: 2px;
    }}
    QSlider::handle:horizontal {{
        width: 16px; height: 16px;
        margin: -6px 0;
        background: {accent};
        border-radius: 8px;
    }}

    QListWidget {{
        background-color: {bg_secondary};
        border: 1px solid {border};
        border-radius: {radius_card};
        color: {text_primary};
        padding: 10px;
    }}
    QListWidget::item {{
        padding: 12px;
        border-radius: {radius_base};
        margin-bottom: 4px;
    }}
    QListWidget::item:selected {{
        background-color: {accent_fill};
        color: {on_accent};
    }}
    QListWidget::item:hover {{
        background-color: {bg_tertiary};
    }}
"""


class ThemeManager:
    """현재 테마 색을 들고 있는 싱글톤.

    ``ui_prefs.json`` 은 **처음 색을 물어볼 때** 읽는다(지연 로드). import 시점에
    읽으면 모듈 import 순서에 따라 파일이 아직 없거나 마이그레이션 전일 수 있다.
    """

    def __init__(self):
        self._font_family = DEFAULT_FONT_FAMILY
        self._font_size = DEFAULT_FONT_SIZE
        self._preset = DEFAULT_PRESET
        self._overrides: dict = {}
        self._colors: dict | None = None  # None = ui_prefs 를 아직 안 읽음

    @property
    def current_preset(self) -> str:
        """실제로 적용 중인 프리셋 id(default/dark/light)."""
        self._ensure_colors()
        return self._preset

    # ── 글꼴 ──

    @property
    def current_font_family(self) -> str:
        """저장용 원본 글꼴 이름"""
        return self._font_family

    @property
    def current_font_size(self) -> str:
        return self._font_size

    def set_font(self, family: str, size_pt: float):
        """글꼴 설정 변경"""
        if family:
            self._font_family = f"'{family}', 'Malgun Gothic', sans-serif"
        self._font_size = f"{size_pt}pt"

    def get_font_family_name(self) -> str:
        """콤보박스 복원용: 첫 번째 폰트 이름만 추출"""
        raw = self._font_family
        if raw.startswith("'"):
            return raw.split("'")[1]
        return raw.split(",")[0].strip()

    def get_font_size_value(self) -> float:
        """스핀박스 복원용: pt 값만 추출"""
        return float(self._font_size.replace('pt', ''))

    # ── 색 ──

    def _ensure_colors(self):
        if self._colors is None:
            self.reload_from_prefs()

    def reload_from_prefs(self) -> dict:
        """``ui_prefs.json`` 에서 theme/themeOverrides 를 다시 읽어 색을 만든다."""
        preset, overrides = DEFAULT_PRESET, {}
        try:
            from core.config_migration import load_ui_prefs
            from core.ui_prefs import ui_prefs_path

            prefs = load_ui_prefs(ui_prefs_path())
            if isinstance(prefs, dict):
                preset = prefs.get('theme') or DEFAULT_PRESET
                raw = prefs.get('themeOverrides')
                overrides = raw if isinstance(raw, dict) else {}
        except Exception:
            # 설정을 못 읽어도 앱이 색 없이 뜨면 안 된다 — 기본 프리셋으로 간다.
            pass
        return self.set_theme(preset, overrides)

    def apply_prefs(self, prefs: dict) -> dict:
        """이미 읽어 둔 prefs 덩어리에서 테마만 뽑아 적용(파일 재읽기 없음)."""
        if not isinstance(prefs, dict):
            return self.get_colors()
        raw = prefs.get('themeOverrides')
        return self.set_theme(
            prefs.get('theme') or DEFAULT_PRESET,
            raw if isinstance(raw, dict) else {},
        )

    def set_theme(self, preset: str | None, overrides: dict | None = None) -> dict:
        """프리셋/덮어쓰기를 정하고 색표를 다시 계산한다."""
        self._preset = preset if preset in PRESETS else DEFAULT_PRESET
        self._overrides = dict(overrides or {})
        self._colors = _build_colors(self._preset, self._overrides)
        return self._colors

    def get_colors(self) -> dict:
        """현재 테마 색상 딕셔너리 반환"""
        self._ensure_colors()
        return self._colors

    def get_stylesheet(self) -> str:
        """전역 QSS 문자열 반환 — 색은 프리셋에서 온다.

        (앱은 ``app.setStyleSheet("")`` 로 전역 QSS 를 비우고 화면 대부분을 Vue 가 그리므로,
        이 QSS 는 남은 PyQt 창·다이얼로그용이다.)
        """
        fmt_vars = {
            **self.get_colors(),
            'font_family': self._font_family,
            'font_size': self._font_size,
        }
        return _QSS_TEMPLATE.format(**fmt_vars)


# 싱글톤
_instance: ThemeManager | None = None


def get_theme_manager() -> ThemeManager:
    global _instance
    if _instance is None:
        _instance = ThemeManager()
    return _instance


def get_color(key: str) -> str:
    """테마 색상 단축 접근 헬퍼"""
    return get_theme_manager().get_colors().get(key, '#888888')
