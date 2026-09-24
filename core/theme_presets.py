# -*- coding: utf-8 -*-
"""테마 프리셋 — 앱 전체 색의 단일 출처.

**왜 Python 이 원본인가**: 앱은 Vue 가 뜨기 전에 PyQt 다이얼로그(백엔드 선택·스플래시)를
먼저 그린다. 그때도 사용자가 고른 테마여야 하므로, 색은 Python 이 시작 시점에 읽을 수
있어야 한다. Vue 쪽 사본은 ``frontend/src/theme/presets.ts`` 에 있고
``tests/test_theme_contract.py`` 가 두 벌이 갈라지지 않는지 정적으로 검증한다.
(``core/curves.py`` ↔ ``utils/curves.ts`` 와 같은 방식.)

**프리셋 셋**
- ``default`` — 리디자인 캔버스에서 쓴 값. 숯색 바탕에 절제된 금색(#C9A227).
- ``dark``    — 더 깊은 순수 검정. 대비를 세게 준 쪽.
- ``light``   — 회백 바탕. 다크를 뒤집은 게 아니라 다시 계산했다(아래 참조).

**라이트는 뒤집기가 아니다**: 태그 6색은 명도 L72 로 어두운 배경에 맞춰 잡은 값이라
흰 배경에서는 1.5~2:1 로 무너진다. 색상각(hue)과 최소 이격은 그대로 두고 명도를
**색상마다 따로 풀어서** 전부 5:1 이상이 되게 했다. 청록·초록은 원래 밝은 색상이라
같은 명도로는 대비가 안 나온다.

**대비는 배경이 아니라 '가장 불리한 면'에서 잰다**: 글자·경계는 페이지 위에만
얹히는 게 아니라 버튼·칩 위에도 앉는다. 배경(``bg-primary``)만 보고 맞췄더니
라이트가 페이지에서는 통과하고 버튼 hover(#DDDDD9) 위에서만 4.1:1 로 무너져
브라우저 점검에서 26곳이 잡혔다. 그래서 ``--edge`` 도 #666666 이 아니라 #747474 다.

**채움색과 글자색은 다른 색이다**: ``--state-alert`` (#D14343)는 배지 *배경*으로
흰 글자와 4.57:1 을 맞춘 값이라, 어두운 바탕에 **글자로** 쓰면 4.33:1 로 미달한다.
그래서 역할마다 채움(``state-x``)과 글자(``state-x-fg``)를 나눠 둔다.
"""

from __future__ import annotations

import colorsys

#: 사용자가 설정에서 직접 바꿀 수 있는 색. 나머지는 프리셋이 정한다.
#: 배경·글자색을 열어두지 않는 이유는 한 번의 실수로 화면을 읽을 수 없게 만들 수 있어서다.
EDITABLE_KEYS: tuple[str, ...] = ('accent', 'state-info', 'state-alert', 'state-ok')

#: 태그 분류의 색상각(도). 프리셋이 바뀌어도 고정 — 같은 뜻은 같은 색상이어야 한다.
TAG_HUES: dict[str, int] = {
    'person': 218, 'scene': 178, 'pose': 130, 'wear': 272, 'fx': 318, 'nsfw': 356,
}

_DARK_TAGS = {
    'tag-person': '#8EADE1', 'tag-scene': '#8EE1DE', 'tag-pose': '#8EE19C',
    'tag-wear': '#BA8EE1', 'tag-fx': '#E18EC8', 'tag-nsfw': '#E18E94',
    'tag-neutral': '#C6C6C6', 'tag-wild-edge': '#8A8A8A',
}

#: 라이트용 태그색 — 색상각은 같고 명도만 색상별로 풀었다.
#: 기준 면은 배경이 아니라 **가장 어두운 면**(``bg-button-hover`` #DDDDD9)이다.
#: 글자는 배경 위에만 얹히는 게 아니라 버튼·칩 위에도 앉는다 — 배경만 보고 맞추면
#: 페이지에서는 통과하고 버튼 위에서만 4.1:1 로 무너진다(실제로 그랬다).
_LIGHT_TAGS = {
    'tag-person': '#385EA1', 'tag-scene': '#266968', 'tag-pose': '#266C32',
    'tag-wear': '#793DAE', 'tag-fx': '#9E3780', 'tag-nsfw': '#A63A42',
    'tag-neutral': '#55554F', 'tag-wild-edge': '#8A8A82',
}

PRESETS: dict[str, dict[str, str]] = {
    # ── 리디자인 캔버스의 팔레트 ──
    'default': {
        'label': '기본',
        'mode': 'dark',
        'bg-primary': '#0A0A0A', 'bg-secondary': '#131313', 'bg-card': '#161616',
        'bg-input': '#131313', 'bg-button': '#1E1E1E', 'bg-button-hover': '#282828',
        'accent': '#C9A227',
        'text-primary': '#E4E4E4', 'text-secondary': '#A2A2A2', 'text-muted': '#919191',
        'edge': '#747474', 'rule': '#242424',
        'border': '#242424', 'border-strong': '#4A4A4A',
        'state-info': '#4C76B0', 'state-info-fg': '#8FB8E6',
        'state-alert': '#D14141', 'state-alert-fg': '#F87171',
        'state-ok': '#2C8549', 'state-ok-fg': '#4ADE80',
        'state-warn': '#986D1C', 'state-warn-fg': '#E0B341',
        **_DARK_TAGS,
    },
    # ── 더 깊고 대비가 센 쪽 ──
    'dark': {
        'label': '다크',
        'mode': 'dark',
        'bg-primary': '#050505', 'bg-secondary': '#0D0D0D', 'bg-card': '#121212',
        'bg-input': '#181818', 'bg-button': '#1E1E1E', 'bg-button-hover': '#2A2A2A',
        'accent': '#FACC15',
        'text-primary': '#FFFFFF', 'text-secondary': '#B8B8B8', 'text-muted': '#939393',
        'edge': '#767676', 'rule': '#2E2E2E',
        'border': '#363636', 'border-strong': '#565656',
        'state-info': '#4C76B0', 'state-info-fg': '#8FB8E6',
        'state-alert': '#D14141', 'state-alert-fg': '#F87171',
        'state-ok': '#2C8549', 'state-ok-fg': '#4ADE80',
        'state-warn': '#986D1C', 'state-warn-fg': '#E0B341',
        **_DARK_TAGS,
    },
    # ── 회백 바탕. 순백은 눈이 부시고 흰 카드를 얹을 여지가 없어진다. ──
    'light': {
        'label': '라이트',
        'mode': 'light',
        'bg-primary': '#F4F4F2', 'bg-secondary': '#FFFFFF', 'bg-card': '#FFFFFF',
        'bg-input': '#FFFFFF', 'bg-button': '#E9E9E6', 'bg-button-hover': '#DDDDD9',
        'accent': '#775C00',
        'text-primary': '#1B1B19', 'text-secondary': '#4E4E48', 'text-muted': '#5F5F59',
        'edge': '#6B6B64', 'rule': '#DCDCD7',
        'border': '#DCDCD7', 'border-strong': '#B4B4AD',
        'state-info': '#2F6099', 'state-info-fg': '#2F6099',
        'state-alert': '#B3261E', 'state-alert-fg': '#B3261E',
        'state-ok': '#1F7A38', 'state-ok-fg': '#1C6D32',
        'state-warn': '#8A5A00', 'state-warn-fg': '#845600',
        **_LIGHT_TAGS,
    },
}

DEFAULT_PRESET = 'default'


# ── 색 계산 ────────────────────────────────────────────────────────────────

def _hex_to_rgb(value: str) -> tuple[float, float, float]:
    h = value.lstrip('#')
    if len(h) == 3:
        h = ''.join(c * 2 for c in h)
    return tuple(int(h[i:i + 2], 16) / 255 for i in (0, 2, 4))


def _rgb_to_hex(r: float, g: float, b: float) -> str:
    return '#%02X%02X%02X' % tuple(round(max(0.0, min(1.0, c)) * 255) for c in (r, g, b))


def relative_luminance(value: str) -> float:
    def lin(c: float) -> float:
        return c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4
    r, g, b = (lin(c) for c in _hex_to_rgb(value))
    return 0.2126 * r + 0.7152 * g + 0.0722 * b


def contrast_ratio(a: str, b: str) -> float:
    """WCAG 대비비. 1.0(같은 색) ~ 21.0(검정↔흰색)."""
    la, lb = relative_luminance(a), relative_luminance(b)
    hi, lo = max(la, lb), min(la, lb)
    return (hi + 0.05) / (lo + 0.05)


def is_valid_hex(value: str) -> bool:
    if not isinstance(value, str):
        return False
    h = value.strip().lstrip('#')
    return len(h) in (3, 6) and all(c in '0123456789abcdefABCDEF' for c in h)


def normalize_hex(value: str) -> str:
    h = value.strip().lstrip('#')
    if len(h) == 3:
        h = ''.join(c * 2 for c in h)
    return '#' + h.upper()


def shift_lightness(value: str, delta: float) -> str:
    """명도를 delta(-1~1)만큼 옮긴다. 색상·채도는 유지."""
    r, g, b = _hex_to_rgb(value)
    h, light, s = colorsys.rgb_to_hls(r, g, b)
    return _rgb_to_hex(*colorsys.hls_to_rgb(h, max(0.0, min(1.0, light + delta)), s))


def rgba(value: str, alpha: float) -> str:
    r, g, b = (round(c * 255) for c in _hex_to_rgb(value))
    return f'rgba({r},{g},{b},{alpha})'


#: 글자를 얹는 면이 지켜야 할 최소 대비.
MIN_ON_ACCENT_CONTRAST = 4.5


def readable_fill(color: str, on: str) -> str:
    """``on`` 색 글자가 4.5:1 로 읽힐 때까지 면 색의 명도만 민다.

    **왜 필요한가**: 중간 밝기 색(예: ``#D14747``)은 흰 글자도 검은 글자도 4.5:1 이
    안 나온다 — 둘 다 4.4 대다. 강조색을 그대로 버튼 배경에 쓰면 사용자가 그런 색을
    고르는 순간 주 버튼 글자가 안 읽힌다. 그래서 **사용자의 색은 그대로 두고**
    (테두리·표시등은 그 색을 쓴다) 글자를 얹는 면만 필요한 만큼 민다.
    """
    if contrast_ratio(on, color) >= MIN_ON_ACCENT_CONTRAST:
        return color
    # 글자가 밝으면 면을 어둡게, 글자가 어두우면 면을 밝게
    step = -0.02 if relative_luminance(on) > 0.5 else 0.02
    candidate = color
    for _ in range(50):
        candidate = shift_lightness(candidate, step)
        if contrast_ratio(on, candidate) >= MIN_ON_ACCENT_CONTRAST:
            return candidate
    return candidate


def readable_text(color: str, background: str, minimum: float = MIN_ON_ACCENT_CONTRAST) -> str:
    """``background`` 위에서 읽히도록 색의 명도만 민다 — 색상·채도는 그대로.

    **왜 필요한가**: 사용자가 '선택'·'알림'·'연결' 색을 바꾸면 배지(채움)만 바뀌고
    **같은 뜻의 글자는 프리셋 값 그대로** 남는다. 파란 배지 옆에 초록 글자가 남는 식이라
    편집이 반쪽이 된다. 그래서 채움색을 바꾸면 글자용 변형(``state-x-fg``)을 여기서 만든다.
    방향은 배경 밝기가 정한다 — 어두운 배경이면 밝게, 밝은 배경이면 어둡게.
    """
    if contrast_ratio(color, background) >= minimum:
        return color
    step = 0.02 if relative_luminance(background) < 0.5 else -0.02
    candidate = color
    for _ in range(60):
        candidate = shift_lightness(candidate, step)
        if contrast_ratio(candidate, background) >= minimum:
            return candidate
    return candidate


def hover_fill(fill: str, on: str, mode: str) -> str:
    """글자를 얹는 면의 호버색 — **글자색에서 멀어지는** 쪽으로 민다.

    **왜 방향을 글자에서 정하는가**: 예전에는 다크 테마면 무조건 밝게 밀었다.
    그런데 사용자가 중간 밝기 색(#D14747 류)을 고르면 ``on-accent`` 가 흰색이
    되는데, 그 면을 더 밝히면 흰 글자가 오히려 안 읽힌다 — ``accent-fill`` 이
    4.5:1 을 맞춰 놔도 **호버에서만 2.9:1 로 떨어졌다**. 면이 지켜야 할 건
    테마의 밝기가 아니라 그 위에 얹힌 글자다. 미는 폭만 테마가 정한다.
    """
    amount = 0.10 if mode == 'dark' else 0.08
    delta = -amount if relative_luminance(on) > 0.5 else amount
    shifted = shift_lightness(fill, delta)
    # 이미 순백/순검이라 그쪽으로 더 못 밀면 반대로 민다 — 호버가 원색과 같으면
    # 버튼이 마우스에 반응하지 않는 것처럼 보인다.
    return shifted if shifted != fill else shift_lightness(fill, -delta)


def derive_accent(accent: str, mode: str) -> dict[str, str]:
    """강조색 하나에서 파생색을 만든다.

    사용자가 아무 색이나 고를 수 있으므로 ``on-accent``(주 버튼의 글자색)를
    **밝기에서 계산**한다. 고정 흰색으로 두면 노란 버튼 위 흰 글자처럼
    안 읽히는 조합이 나온다. 그래도 모자라면 ``accent-fill`` 이 면을 밀어 준다.
    """
    hover_delta = 0.10 if mode == 'dark' else -0.08
    on_white = contrast_ratio('#FFFFFF', accent)
    on_black = contrast_ratio('#0A0A0A', accent)
    on = '#FFFFFF' if on_white >= on_black else '#0A0A0A'
    fill = readable_fill(accent, on)
    return {
        'accent': accent,
        # 글자를 얹지 않는 쪽(테두리·표시등)이라 테마 밝기대로 민다.
        'accent-hover': shift_lightness(accent, hover_delta),
        'accent-dim': rgba(accent, 0.14 if mode == 'dark' else 0.12),
        # 글자를 얹는 면 — 주 버튼 배경. 보통은 accent 와 같다.
        'accent-fill': fill,
        'accent-fill-hover': hover_fill(fill, on, mode),
        'on-accent': on,
    }


def resolve(preset_name: str | None = None,
            overrides: dict[str, str] | None = None) -> dict[str, str]:
    """프리셋 + 사용자 덮어쓰기 → 최종 색 표.

    덮어쓰기는 ``EDITABLE_KEYS`` 만 받는다. 값이 hex 가 아니면 조용히 무시한다 —
    설정 파일이 손상돼도 앱이 색 없이 뜨는 일은 없어야 한다.
    """
    name = preset_name if preset_name in PRESETS else DEFAULT_PRESET
    colors = dict(PRESETS[name])
    for key, value in (overrides or {}).items():
        if key in EDITABLE_KEYS and is_valid_hex(value):
            colors[key] = normalize_hex(value)
            # 채움색을 바꿨으면 같은 뜻의 글자색도 따라와야 한다 — 안 그러면
            # 배지만 바뀌고 문구는 프리셋 색으로 남아 둘이 따로 논다.
            if key.startswith('state-'):
                colors[f'{key}-fg'] = readable_text(colors[key], colors['bg-primary'])
    colors.update(derive_accent(colors['accent'], colors['mode']))
    return colors


def css_variables(colors: dict[str, str]) -> str:
    """색 표 → CSS 커스텀 속성 선언 (``mode``/``label`` 은 색이 아니라 제외)."""
    return '\n'.join(
        f'  --{key}: {value};'
        for key, value in colors.items()
        if key not in ('mode', 'label')
    )
