"""에디터 워터마크 렌더링 — 텍스트/이미지 (Qt 비의존 순수 로직).

``VueBridge._editor_process_impl`` 의 ``text_watermark`` / ``image_watermark`` 본체.
예전 구현에는 네 가지 문제가 있었다.

1. **투명도 소실** — 입력을 ``COLOR_BGR2RGB`` 로 바꾸고 3채널로 되돌려서, 배경 제거 뒤나
   투명 PNG 에 워터마크를 넣으면 알파가 영구히 사라졌다(fast 는 검정, balanced/quality 는
   번진 전경색이 드러남). 여기서는 BGRA 가 들어오면 BGRA 로 돌려준다.
2. **글꼴·크기 무시** — 패널이 보내는 표시명('Arial', 'Times New Roman' …)을 그대로
   ``ImageFont.truetype`` 에 넘겨 전부 OSError 였고, 폴백 ``load_default()`` 는 크기 인자가
   없어 늘 10px 였다. 표시명 → 글꼴 파일 매핑으로 찾고, 폴백도 요청 크기를 지킨다.
   고른 글꼴에 없는 글자는 **글자마다 실제 글리프가 있는 대체 글꼴**로 그린다
   (``plan_font_runs``). 한 글꼴이 한·중·일을 다 담지 못해서다 — 맑은 고딕에는 '込'·'働'
   같은 일본 한자가 없고, Yu Gothic 에는 한글이 없다.
3. **이미지 워터마크 배율** — 패널은 '크기 (%)' 슬라이더 값을 보내는데 예전 백엔드는
   ``max(1.0, scale) / 100`` 으로 받아 100% 가 1% 로 줄었다. 이제 퍼센트로 통일한다.
   붙일 때 워터마크 자신을 마스크로 쓰던 ``paste(wm, pos, wm)`` 는 알파를 두 번 곱해
   불투명도가 제곱으로 떨어졌다(50% → 25%) — 마스크 없이 알파 합성한다.
4. **프리뷰 배율** — 프리뷰는 긴 변 1024 로 줄인 소스에서 돈다. 원본 픽셀 단위 값(글자
   크기·타일 간격·워터마크 크기)에 ``pixel_scale`` 을 곱해야 확정 결과와 같은 비율로 보인다.
5. **메모리** — 회전 타일은 출력 픽셀마다 돌리기 전 좌표를 거꾸로 구해 한 칸을 주기로
   읽는다(``tiled_alpha``) — 대각선만큼 넓힌 캔버스를 통째로 돌리지 않는다. 이미지 워터마크는
   이미지와 겹치는 부분만 리사이즈한다(``resize_visible``). 둘 다 메모리가 이미지 크기에 비례한다.

위치(xPct/yPct)는 워터마크 **중심**이 놓일 자리(%)다. ``clamp`` 면 이미지 밖으로 나가지
않게 안쪽으로 민다 — 돌린 텍스트는 돌린 외접 상자가 이미지 안에 들도록(``rotated_origin``).
"""
from __future__ import annotations

import math
import os
import threading
import unicodedata
from functools import lru_cache
from typing import Any, Iterable, Mapping, Optional

import numpy as np

# 표시명(소문자) → 후보 글꼴 파일. Windows 파일명이 먼저, 그다음 리눅스·맥 대체 글꼴.
FONT_FILES: dict[str, tuple[str, ...]] = {
    'arial': ('arial.ttf', 'Arial.ttf', 'LiberationSans-Regular.ttf', 'DejaVuSans.ttf'),
    'times new roman': ('times.ttf', 'Times New Roman.ttf', 'LiberationSerif-Regular.ttf', 'DejaVuSerif.ttf'),
    'courier new': ('cour.ttf', 'Courier New.ttf', 'LiberationMono-Regular.ttf', 'DejaVuSansMono.ttf'),
    'verdana': ('verdana.ttf', 'Verdana.ttf', 'DejaVuSans.ttf'),
    'georgia': ('georgia.ttf', 'Georgia.ttf', 'DejaVuSerif.ttf'),
    'malgun gothic': ('malgun.ttf', 'AppleSDGothicNeo.ttc', 'NotoSansCJK-Regular.ttc'),
    '맑은 고딕': ('malgun.ttf', 'AppleSDGothicNeo.ttc', 'NotoSansCJK-Regular.ttc'),
}
DEFAULT_FAMILY = 'arial'

# 고른 글꼴에 없는 글자를 찾아볼 대체 글꼴. 한 글꼴이 한·중·일을 다 담지 못한다 — 맑은 고딕은
# KS X 1001 한자만 있어 '込'·'働' 같은 일본 한자가 두부(□)로 나왔고, Yu Gothic·MS Gothic 에는
# 한글이 없다. 그래서 순서만 정해 두고 **글자마다 실제 글리프가 있는 첫 글꼴**로 그린다.
# 이 PC 에 없는 파일 이름은 건너뛴다.
KOREAN_FONT_FILES: tuple[str, ...] = (
    'malgun.ttf', 'NotoSansKR-VF.ttf', 'NotoSansKR-Regular.otf', 'AppleSDGothicNeo.ttc', 'gulim.ttc',
)
JAPANESE_FONT_FILES: tuple[str, ...] = (
    'YuGothM.ttc', 'YuGothR.ttc', 'meiryo.ttc', 'msgothic.ttc',
    'NotoSansJP-VF.ttf', 'NotoSansJP-Regular.otf', 'ヒラギノ角ゴシック W3.ttc',
)
CHINESE_FONT_FILES: tuple[str, ...] = (
    'msyh.ttc', 'msjh.ttc', 'simsun.ttc', 'mingliub.ttc', 'simsunb.ttf',
    'NotoSansSC-Regular.otf', 'NotoSansTC-Regular.otf',
)
PAN_CJK_FONT_FILES: tuple[str, ...] = ('NotoSansCJK-Regular.ttc', 'DroidSansFallbackFull.ttf')
# 한중일이 아닌 기호(→ ★ ✓ …)를 찾아볼 글꼴 — 기본(Arial 계열) 다음에 본다.
SYMBOL_FONT_FILES: tuple[str, ...] = ('seguisym.ttf', 'segoeui.ttf', 'DejaVuSans.ttf')
# 한국어 우선(기본) 순서. 가나가 섞인 글은 일본어 글꼴을 먼저 본다 — cjk_font_files('ja').
CJK_FONT_FILES: tuple[str, ...] = KOREAN_FONT_FILES + JAPANESE_FONT_FILES + CHINESE_FONT_FILES + PAN_CJK_FONT_FILES

MAX_FONT_PX = 2000
TILE_GAP_PX = 40          # 타일 반복 간격(원본 픽셀)
LINE_SPACING_PX = 4       # 여러 줄 텍스트의 줄 간격(Pillow multiline 기본값과 같다)
MIN_SCALE_PCT, MAX_SCALE_PCT = 1.0, 1000.0

_CJK_RANGES = (
    (0x1100, 0x11FF),   # 한글 자모
    (0x2E80, 0x2FDF),   # CJK 부수
    (0x3000, 0x303F),   # CJK 기호·구두점
    (0x3040, 0x30FF),   # 히라가나·가타카나
    (0x3130, 0x318F),   # 한글 호환 자모
    (0x31F0, 0x31FF),   # 가타카나 음성 확장
    (0x3200, 0x33FF),   # 괄호·원문자 CJK, CJK 호환
    (0x3400, 0x4DBF),   # CJK 확장 A
    (0x4E00, 0x9FFF),   # CJK 통합 한자
    (0xA960, 0xA97F),   # 한글 자모 확장 A
    (0xAC00, 0xD7FF),   # 한글 음절 + 자모 확장 B
    (0xF900, 0xFAFF),   # CJK 호환 한자
    (0xFF00, 0xFFEF),   # 반각·전각
)


class WatermarkError(ValueError):
    """사용자에게 그대로 보여 줄 수 있는 워터마크 실패."""


# ─────────────────────────────────────────────
# 글꼴
# ─────────────────────────────────────────────

_KANA_RANGES = (
    (0x3040, 0x30FF),   # 히라가나·가타카나
    (0x31F0, 0x31FF),   # 가타카나 음성 확장
    (0xFF66, 0xFF9F),   # 반각 가타카나
)
# 앞 글자의 글꼴에 붙여 그릴 글자(공백·제어·결합 문자) — 따로 떼면 결합 문자가 떨어져 나간다
_NEUTRAL_CATEGORIES = frozenset({'Zs', 'Zl', 'Zp', 'Cc', 'Cf', 'Mn', 'Mc', 'Me'})
# 글리프 유무 판정: 어느 글꼴도 매핑하지 않는 비문자. U+E000 같은 사용자 영역은 Segoe UI Symbol
# 처럼 실제로 매핑한 글꼴이 있어 기준으로 못 쓴다.
_NOTDEF_PROBE = '\U0010FFFF'
_PROBE_PX = 32
_COVERAGE_MAX_PER_FONT = 20000

_coverage_lock = threading.Lock()
_coverage: dict[str, dict[str, bool]] = {}


def _in_ranges(ch: str, ranges) -> bool:
    cp = ord(ch)
    return any(lo <= cp <= hi for lo, hi in ranges)


def needs_cjk_font(text: str) -> bool:
    """한글·가나·한자가 섞였는지 — 라틴 전용 글꼴로는 두부(□)로 그려진다."""
    return any(_in_ranges(ch, _CJK_RANGES) for ch in str(text or ''))


def script_preference(text: str) -> str:
    """대체 글꼴을 볼 언어 순서 — 가나가 있으면 'ja', 아니면 'ko'.

    한자는 한국어·일본어가 같이 쓰므로 가나로 일본어 글을 가린다. 일본어 글의 한자를
    한국어 글꼴로 그리면 한 단어 안에서 자형이 섞인다('申し込み' 의 申·込).
    """
    return 'ja' if any(_in_ranges(ch, _KANA_RANGES) for ch in str(text or '')) else 'ko'


def cjk_font_files(prefer: str) -> tuple[str, ...]:
    """한중일 글자를 찾아볼 글꼴 이름 순서(``prefer`` = 'ja' 면 일본어 글꼴 먼저)."""
    if prefer == 'ja':
        return JAPANESE_FONT_FILES + KOREAN_FONT_FILES + CHINESE_FONT_FILES + PAN_CJK_FONT_FILES
    return CJK_FONT_FILES


def _dedup(names: Iterable[str]) -> tuple[str, ...]:
    seen: set[str] = set()
    return tuple(n for n in names if not (n.lower() in seen or seen.add(n.lower())))


def font_candidates(family: Any) -> tuple[str, ...]:
    """고른 표시명의 글꼴 파일 이름 순서(+ 기본 Arial 계열). 모르는 표시명은 기본으로 본다.

    경로나 임의 파일명은 받지 않는다 — 웹 모드에서도 열리는 연산이라, 요청 문자열로
    서버의 임의 파일을 글꼴로 열게 두지 않는다.
    """
    key = str(family or '').strip().lower()
    if key not in FONT_FILES:
        key = DEFAULT_FAMILY
    return _dedup(FONT_FILES[key] + FONT_FILES[DEFAULT_FAMILY])


def fallback_candidates(prefer: str, cjk_char: bool) -> tuple[str, ...]:
    """고른 글꼴에 없는 글자를 찾아볼 대체 글꼴 이름 순서.

    한중일 글자는 한·일·중 글꼴부터, 그 밖(라틴 확장·기호)은 기본·기호 글꼴부터 본다 —
    기호 하나 때문에 라틴 글자가 한글 글꼴 자형으로 바뀌지 않게.
    """
    cjk = cjk_font_files(prefer)
    other = FONT_FILES[DEFAULT_FAMILY] + SYMBOL_FONT_FILES
    return _dedup(cjk + other if cjk_char else other + cjk)


def _font_dirs() -> list[str]:
    """Pillow 가 찾지 않는 사용자 설치 글꼴 폴더(Windows 10+ 개인 설치)."""
    dirs = []
    local = os.environ.get('LOCALAPPDATA')
    if local:
        dirs.append(os.path.join(local, 'Microsoft', 'Windows', 'Fonts'))
    return dirs


@lru_cache(maxsize=256)
def find_font_file(name: str) -> Optional[str]:
    """글꼴 파일 이름 → 열 수 있는 실제 경로. 없으면 None.

    글꼴 객체가 아니라 경로를 캐시한다 — FreeType 글꼴 객체를 여러 워커 스레드가
    동시에 쓰지 않게, 그리는 쪽이 매번 새로 연다.
    """
    from PIL import ImageFont
    # Pillow 는 이름만 주면 %WINDIR%\Fonts 와 XDG/맥 글꼴 폴더를 뒤진다.
    for candidate in [name] + [os.path.join(d, name) for d in _font_dirs()]:
        try:
            font = ImageFont.truetype(candidate, 12)
        except (OSError, ValueError):
            continue
        path = getattr(font, 'path', None)
        if isinstance(path, (str, bytes)) and path:
            return os.fsdecode(path)
        return candidate
    return None


@lru_cache(maxsize=64)
def resolve_font_path(family: str) -> Optional[str]:
    """고른 표시명의 열 수 있는 첫 글꼴 경로. 없으면 None (→ 크기를 지키는 기본 글꼴)."""
    for name in font_candidates(family):
        path = find_font_file(name)
        if path:
            return path
    return None


@lru_cache(maxsize=16)
def fallback_font_paths(prefer: str, cjk_char: bool) -> tuple[str, ...]:
    """``fallback_candidates`` 중 이 PC 에 있는 글꼴 경로(중복 없이, 순서 유지)."""
    paths: list[str] = []
    seen: set[str] = set()
    for name in fallback_candidates(prefer, cjk_char):
        path = find_font_file(name)
        if path and os.path.normcase(path) not in seen:
            seen.add(os.path.normcase(path))
            paths.append(path)
    return tuple(paths)


def open_font(path: Optional[str], size_px: int):
    """``path`` 글꼴을 ``size_px`` 로 연다. 경로가 없거나 못 열면 크기를 지키는 기본 글꼴."""
    from PIL import ImageFont
    size = max(1, min(MAX_FONT_PX, int(size_px)))
    if path:
        try:
            return ImageFont.truetype(path, size)
        except (OSError, ValueError):
            pass
    try:
        return ImageFont.load_default(size=size)   # Pillow ≥ 10.1: 크기를 지키는 기본 글꼴
    except TypeError:                              # pragma: no cover - 옛 Pillow
        return ImageFont.load_default()


def _glyph_signature(font, ch: str):
    mask = font.getmask(ch)
    return font.getlength(ch), tuple(mask.size), bytes(mask)


def _probe_coverage(path: Optional[str], chars: Iterable[str]) -> dict[str, bool]:
    from PIL import ImageFont
    chars = list(chars)
    try:
        font = ImageFont.truetype(path, _PROBE_PX) if path else ImageFont.load_default(size=_PROBE_PX)
        notdef = _glyph_signature(font, _NOTDEF_PROBE)
        return {ch: _glyph_signature(font, ch) != notdef for ch in chars}
    except (OSError, ValueError, TypeError):
        return {ch: False for ch in chars}


def covered_chars(path: Optional[str], chars: Iterable[str]) -> frozenset[str]:
    """``path`` 글꼴(None 이면 기본 글꼴)에 실제 글리프가 있는 글자들.

    Pillow 는 글리프가 있는지 알려 주지 않는다 — 없는 글자는 .notdef(대개 빈 네모)로
    그리므로, 어느 글꼴에도 없는 비문자 U+10FFFF 와 모양·폭이 같으면 '없음'으로 본다.
    결과는 글꼴·글자별로 캐시한다 — 프리뷰가 슬라이더마다 불려도 글꼴을 다시 열지 않게.
    """
    wanted = set(chars)
    key = path or ''
    result: dict[str, bool] = {}
    with _coverage_lock:
        known = _coverage.get(key, {})
        for ch in wanted:
            if ch in known:
                result[ch] = known[ch]
    missing = [ch for ch in wanted if ch not in result]
    if missing:
        found = _probe_coverage(path, missing)
        result.update(found)
        with _coverage_lock:
            known = _coverage.setdefault(key, {})
            if len(known) + len(found) > _COVERAGE_MAX_PER_FONT:
                known.clear()
            known.update(found)
    return frozenset(ch for ch, ok in result.items() if ok)


def plan_font_runs(family: Any, line: str, prefer: Optional[str] = None) -> tuple[tuple[Optional[str], str], ...]:
    """한 줄을 (글꼴 경로, 글자들) 조각으로 나눈다 — 글자마다 그 글자를 가진 첫 글꼴.

    순서: 고른 글꼴 → 대체 글꼴(``fallback_candidates``). 그래서 'Hello 가나다' 는 Arial +
    맑은 고딕, '申し込み' 는 (Arial 에 없으니) 일본어 글꼴 하나, '홍길동 働く' 는 맑은 고딕 +
    Yu Gothic 으로 그려진다 — 한 글꼴이 다 담지 못해도 두부(□)가 생기지 않는다.
    공백·결합 문자는 앞 글자의 글꼴에 붙인다(그 글꼴에 있을 때). 어느 글꼴에도 없는
    글자는 고른 글꼴로 둔다. 경로 None 은 기본 글꼴이다.
    """
    line = str(line or '')
    if not line:
        return ()
    prefer = prefer or script_preference(line)
    primary = resolve_font_path(str(family or ''))
    choice: dict[str, Optional[str]] = {}
    remaining = set(line)
    for ch in covered_chars(primary, remaining):
        choice[ch] = primary
    remaining -= set(choice)
    for cjk_char in (True, False):
        group = {ch for ch in remaining if needs_cjk_font(ch) == cjk_char}
        for path in fallback_font_paths(prefer, cjk_char):
            if not group:
                break
            hit = covered_chars(path, group)
            for ch in hit:
                choice[ch] = path
            group -= hit

    runs: list[list] = []
    for ch in line:
        current = runs[-1][0] if runs else None
        if (runs and unicodedata.category(ch) in _NEUTRAL_CATEGORIES
                and ch in covered_chars(current, (ch,))):
            path = current
        else:
            path = choice.get(ch, primary)
        if runs and runs[-1][0] == path:
            runs[-1][1].append(ch)
        else:
            runs.append([path, [ch]])
    return tuple((path, ''.join(chars)) for path, chars in runs)


def rasterize_text(text: str, family: Any, size_px: int) -> np.ndarray:
    """글자를 그린 커버리지(0~255) 배열 — 높이×폭, 잉크 영역에 딱 맞춘다. 여러 줄('\\n')도 된다.

    조각(``plan_font_runs``)마다 글꼴이 달라도 같은 기준선(anchor 'ls')에 이어 그린다.
    """
    from PIL import Image, ImageDraw
    size = max(1, min(MAX_FONT_PX, int(size_px)))
    source = str(text or '')
    lines = source.replace('\r\n', '\n').replace('\r', '\n').split('\n')
    prefer = script_preference(source)
    plans = [plan_font_runs(family, line, prefer) for line in lines]
    fonts: dict[Optional[str], Any] = {}
    for plan in plans:
        for path, _ in plan:
            if path not in fonts:
                fonts[path] = open_font(path, size)
    if not fonts:
        return np.zeros((1, 1), np.uint8)

    measure = ImageDraw.Draw(Image.new('L', (1, 1)))
    line_step = max(measure.textbbox((0, 0), 'A', font=f)[3] for f in fonts.values()) + LINE_SPACING_PX
    placed = []
    box: Optional[list[int]] = None
    for index, plan in enumerate(plans):
        pen, baseline = 0.0, index * line_step
        for path, run in plan:
            font = fonts[path]
            x = int(round(pen))
            left, top, right, bottom = measure.textbbox((x, baseline), run, font=font, anchor='ls')
            if right > left and bottom > top:
                box = ([left, top, right, bottom] if box is None else
                       [min(box[0], left), min(box[1], top), max(box[2], right), max(box[3], bottom)])
            placed.append((x, baseline, font, run))
            pen += font.getlength(run)
    if box is None:
        return np.zeros((1, 1), np.uint8)
    left, top, right, bottom = (int(math.floor(box[0])), int(math.floor(box[1])),
                                int(math.ceil(box[2])), int(math.ceil(box[3])))
    glyphs = Image.new('L', (max(1, right - left), max(1, bottom - top)), 0)
    draw = ImageDraw.Draw(glyphs)
    for x, baseline, font, run in placed:
        draw.text((x - left, baseline - top), run, fill=255, font=font, anchor='ls')
    return np.array(glyphs, dtype=np.uint8)


# ─────────────────────────────────────────────
# 값 정리
# ─────────────────────────────────────────────

def _num(value: Any, default: float, lo: float, hi: float) -> float:
    try:
        v = float(value)
    except (TypeError, ValueError):
        return default
    if not math.isfinite(v):
        return default
    return max(lo, min(hi, v))


def parse_hex_color(value: Any, default: tuple[int, int, int] = (255, 255, 255)) -> tuple[int, int, int]:
    """'#RRGGBB' / '#RGB' → (r, g, b). 못 읽으면 ``default``."""
    text = str(value or '').strip().lstrip('#')
    if len(text) == 3:
        text = ''.join(c * 2 for c in text)
    if len(text) != 6:
        return default
    try:
        return int(text[0:2], 16), int(text[2:4], 16), int(text[4:6], 16)
    except ValueError:
        return default


def _as_bool(value: Any, default: bool) -> bool:
    if value is None:
        return default
    if isinstance(value, str):
        return value.strip().lower() not in ('', '0', 'false', 'no', 'off')
    return bool(value)


def centered_position(center_pct: float, extent: int, size: int, clamp: bool) -> int:
    """중심을 ``center_pct``% 에 두는 왼쪽(위쪽) 좌표. ``clamp`` 면 [0, extent-size] 로 민다."""
    pos = int(round(extent * center_pct / 100.0 - size / 2.0))
    if clamp:
        pos = max(0, min(pos, max(0, extent - size)))
    return pos


def rotated_extent(width: float, height: float, rotation: float) -> tuple[float, float]:
    """``rotation``° 돌린 width×height 사각형의 축 정렬 외접 상자 (가로, 세로)."""
    cos_r, sin_r = _rotation_terms(rotation)
    return (abs(cos_r) * width + abs(sin_r) * height,
            abs(sin_r) * width + abs(cos_r) * height)


# 돌린 도장을 영역 안으로 밀 때의 여유(px) — 왼쪽 위 좌표의 정수 반올림(최대 0.5px)과 3차 보간 번짐.
# 여유 없이 밀면 90° 에서 홀·짝 크기 도장이 반 픽셀 밀려 8% 안팎이 잘렸다.
ROTATED_CLAMP_MARGIN = 1.0


def rotated_origin(center_pct: float, extent: int, size: float, rotated_size: float, clamp: bool,
                   margin: float = ROTATED_CLAMP_MARGIN) -> int:
    """중심을 ``center_pct``% 에 두는 (돌리기 전) 도장의 왼쪽(위쪽) 좌표 — ``placed_alpha`` 가 도는 중심.

    ``clamp`` 면 돌린 외접 상자(``rotated_size``)가 [0, extent] 안에 들도록 **중심**을 민다.
    도장이 자기 중심으로 돌기 때문에, 돌리기 전 크기로 밀면(``centered_position``) 모서리에서
    돌아간 글자 끝이 이미지 밖으로 나가 잘렸다. 상자가 이미지보다 크면 가운데에 둔다.
    """
    center = extent * center_pct / 100.0
    if clamp:
        half = rotated_size / 2.0 + margin
        if 2.0 * half >= extent:
            center = extent / 2.0
        else:
            center = min(max(center, half), extent - half)
    return int(round(center - size / 2.0))


# ─────────────────────────────────────────────
# 합성 — BGR/BGRA 배열 위에 제자리 'over'
# ─────────────────────────────────────────────

BLEND_BLOCK_PIXELS = 1 << 20   # 한 번에 섞는 픽셀 수 — 실수 임시 배열을 이 크기로 묶는다


def _prepare_base(img: np.ndarray) -> np.ndarray:
    """합성할 8비트 BGR / BGRA **사본**. 흑백·2채널은 BGR 로, 16비트·실수는 8비트로 줄인다.

    알파가 있으면(BGRA) BGRA 로 둔다 — 배경 제거 뒤나 투명 PNG 의 투명도를 지킨다.
    PIL RGBA 로 바꿨다 되돌리지 않는다 — 그 왕복이 전체 크기 사본을 여러 벌 만들었다.
    """
    import cv2
    from core.editor_preview import to_display_uint8
    arr = img if img.dtype == np.uint8 else to_display_uint8(img)
    if arr.ndim == 3 and arr.shape[2] in (1, 2):
        arr = arr[:, :, 0]
    if arr.ndim == 2:
        return cv2.cvtColor(np.ascontiguousarray(arr), cv2.COLOR_GRAY2BGR)
    if arr.ndim != 3 or arr.shape[2] not in (3, 4):
        raise WatermarkError('워터마크를 넣을 수 없는 이미지 형식입니다')
    return np.array(arr, dtype=np.uint8, order='C', copy=True)


def blend_over(dst: np.ndarray, origin: tuple[int, int], alpha: np.ndarray, color) -> None:
    """``dst``(BGR/BGRA uint8) 의 ``origin`` 자리에 ``alpha``(0~255) 모양으로 ``color`` 를 제자리 합성.

    ``color`` 는 단색 (B, G, R) 이거나 ``alpha`` 와 같은 높이×폭의 BGR 배열(이미지 워터마크).
    색은 칠해 두고 모양만 알파로 준다 — 투명 바탕에 반투명 색을 바로 그리면 안티앨리어싱
    가장자리의 RGB 까지 어두워져 밝은 글자에 검은 테두리가 생긴다.
    합성식은 Porter-Duff 'over'(PIL ``alpha_composite`` 와 같은 식)::

        a = aₛ + a_d·(1 − aₛ),   c = (cₛ·aₛ + c_d·a_d·(1 − aₛ)) / a

    불투명 바탕(BGR)이면 c = c_d + (cₛ − c_d)·aₛ 로 줄어든다 — 이 흔한 경우는 OpenCV 정수
    연산으로 (c_d·(255 − aₛ) + cₛ·aₛ) / 255 를 정확히 반올림한다(8K 전체 타일도 0.3초 안팎).
    알파가 0 인 곳은 그대로 둔다. 행 블록으로 나눠 임시 배열을 묶는다.
    """
    import cv2
    x0, y0 = (int(v) for v in origin)
    h, w = alpha.shape[:2]
    if h == 0 or w == 0:
        return
    alpha = np.ascontiguousarray(alpha, dtype=np.uint8)
    per_pixel = isinstance(color, np.ndarray) and color.ndim == 3
    if per_pixel:
        color = np.ascontiguousarray(color, dtype=np.uint8)
        const = None
    else:
        b, g, r = (int(v) for v in color)
        const = np.array([b, g, r], np.float32).reshape(1, 1, 3)
        scalar = (float(b), float(g), float(r), 0.0)
    has_alpha = dst.shape[2] == 4
    rows = max(1, BLEND_BLOCK_PIXELS // max(1, w))
    for r0 in range(0, h, rows):
        r1 = min(h, r0 + rows)
        a_src = alpha[r0:r1]
        if not cv2.countNonZero(a_src):
            continue
        region = dst[y0 + r0:y0 + r1, x0:x0 + w]
        if not has_alpha:
            a3 = cv2.merge([a_src, a_src, a_src])
            keep = cv2.multiply(np.ascontiguousarray(region), cv2.bitwise_not(a3), dtype=cv2.CV_16U)
            paint = (cv2.multiply(color[r0:r1], a3, dtype=cv2.CV_16U) if per_pixel
                     else cv2.multiply(a3, scalar, dtype=cv2.CV_16U))
            region[...] = cv2.convertScaleAbs(cv2.add(keep, paint), alpha=1.0 / 255.0)
            continue
        a = a_src.astype(np.float32)[..., None] * np.float32(1.0 / 255.0)
        src = color[r0:r1].astype(np.float32) if per_pixel else const
        bgr = region[..., :3].astype(np.float32)
        a_dst = region[..., 3:4].astype(np.float32) * np.float32(1.0 / 255.0)
        keep = a_dst * (1.0 - a)
        out_a = a + keep
        visible = out_a > 0
        mixed = (src * a + bgr * keep) / np.where(visible, out_a, 1.0)
        region[..., :3] = np.clip(np.rint(np.where(visible, mixed, bgr)), 0, 255).astype(np.uint8)
        region[..., 3] = np.clip(np.rint(out_a[..., 0] * 255.0), 0, 255).astype(np.uint8)


# ─────────────────────────────────────────────
# 회전·타일 샘플링 (메모리 O(출력))
# ─────────────────────────────────────────────

SAMPLE_BLOCK_PX = 1024    # 한 번에 보간하는 출력 블록 — 좌표 배열·소스 창이 이 크기로 묶인다
_CUBIC_MARGIN = 3         # 3차 보간이 읽는 이웃(-1..+2) + 반올림 여유


def _rotation_terms(rotation: float) -> tuple[float, float]:
    """PIL ``rotate(-rotation)`` 과 같은 방향(화면에서 시계 방향)의 cos·sin."""
    r = math.radians(float(rotation) % 360.0)
    return round(math.cos(r), 15), round(math.sin(r), 15)


def _source_window(src: np.ndarray, x0: int, y0: int, x1: int, y1: int, wrap: bool) -> np.ndarray:
    """``src`` 의 [y0, y1)×[x0, x1) 창. ``wrap`` 이면 주기 반복(타일), 아니면 밖은 0."""
    h, w = src.shape[:2]
    if wrap:
        return src[np.arange(y0, y1) % h][:, np.arange(x0, x1) % w]
    out = np.zeros((y1 - y0, x1 - x0), src.dtype)
    ix0, iy0, ix1, iy1 = max(x0, 0), max(y0, 0), min(x1, w), min(y1, h)
    if ix1 > ix0 and iy1 > iy0:
        out[iy0 - y0:iy1 - y0, ix0 - x0:ix1 - x0] = src[iy0:iy1, ix0:ix1]
    return out


def sample_affine(src: np.ndarray, roi: tuple[int, int, int, int], matrix: tuple[float, ...], *,
                  wrap: bool, block: int = SAMPLE_BLOCK_PX) -> np.ndarray:
    """``roi`` = (x0, y0, x1, y1) 의 출력 픽셀 p 마다 ``src[A·p + t]`` 를 3차 보간해 뽑는다.

    matrix = (a, b, c, d, e, f) — 소스 x = a·x + b·y + c, 소스 y = d·x + e·y + f (픽셀 인덱스).
    ``wrap`` 이면 ``src`` 를 평면 전체에 주기적으로 깐 것으로(타일), 아니면 밖은 0 으로 본다.
    출력을 블록으로 나눠 블록이 닿는 소스 창만 잘라 보간한다 — 메모리는 roi 에 비례하고,
    창이 작아 OpenCV remap 의 16비트 좌표 한계(32767)에도 걸리지 않는다.
    """
    import cv2
    rx0, ry0, rx1, ry1 = (int(v) for v in roi)
    out = np.zeros((max(0, ry1 - ry0), max(0, rx1 - rx0)), np.uint8)
    if out.size == 0 or src.size == 0:
        return out
    a, b, c, d, e, f = (float(v) for v in matrix)
    sh, sw = src.shape[:2]
    block = max(16, int(block))
    for by in range(ry0, ry1, block):
        bh = min(block, ry1 - by)
        ly = np.arange(bh, dtype=np.float32)[:, None]
        for bx in range(rx0, rx1, block):
            bw = min(block, rx1 - bx)
            ox = a * bx + b * by + c
            oy = d * bx + e * by + f
            span_x = (0.0, a * (bw - 1), b * (bh - 1), a * (bw - 1) + b * (bh - 1))
            span_y = (0.0, d * (bw - 1), e * (bh - 1), d * (bw - 1) + e * (bh - 1))
            sx0 = int(math.floor(ox + min(span_x))) - _CUBIC_MARGIN
            sx1 = int(math.ceil(ox + max(span_x))) + _CUBIC_MARGIN + 1
            sy0 = int(math.floor(oy + min(span_y))) - _CUBIC_MARGIN
            sy1 = int(math.ceil(oy + max(span_y))) + _CUBIC_MARGIN + 1
            if not wrap and (sx1 <= 0 or sy1 <= 0 or sx0 >= sw or sy0 >= sh):
                continue
            window = _source_window(src, sx0, sy0, sx1, sy1, wrap)
            lx = np.arange(bw, dtype=np.float32)[None, :]
            map_x = np.float32(ox - sx0) + np.float32(a) * lx + np.float32(b) * ly
            map_y = np.float32(oy - sy0) + np.float32(d) * lx + np.float32(e) * ly
            out[by - ry0:by - ry0 + bh, bx - rx0:bx - rx0 + bw] = cv2.remap(
                window, map_x, map_y, cv2.INTER_CUBIC,
                borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    return out


def tiled_alpha(stamp: np.ndarray, width: int, height: int, step_x: int, step_y: int,
                rotation: float) -> np.ndarray:
    """``stamp`` 을 (step_x, step_y) 간격 격자로 깔고 이미지 중심으로 ``rotation``° 돌린 알파.

    격자는 이미지 원점 기준(-tw, -th 부터) — 회전 여부와 무관하게 무늬가 같은 자리에 온다.
    예전처럼 대각선만큼 넓힌 캔버스에 깔아 통째로 돌리지 않는다 — 한 축 패딩을 두 축에 다 써서
    8K 에서 12292×8932 RGBA 캔버스(≈439MB)에 회전 사본까지 1GB 넘게 잡았다. 출력 픽셀마다
    돌리기 전 평면의 좌표를 거꾸로 구해 한 칸(도장 + 간격)을 주기로 읽으므로, 메모리는
    이미지 크기(1바이트/픽셀)뿐이고 모서리도 빈틈없이 덮인다.
    """
    th, tw = stamp.shape[:2]
    step_x, step_y = max(int(step_x), tw), max(int(step_y), th)
    cell = np.zeros((step_y, step_x), np.uint8)
    cell[:th, :tw] = stamp
    cos_r, sin_r = _rotation_terms(rotation)
    cx, cy = width / 2.0, height / 2.0
    # 칸 좌표 u = R·(p + ½ − C) + C + (tw, th) − ½   (C = 이미지 중심, 칸 원점 = 도장 왼쪽 위)
    c = cos_r * (0.5 - cx) + sin_r * (0.5 - cy) + cx + tw - 0.5
    f = -sin_r * (0.5 - cx) + cos_r * (0.5 - cy) + cy + th - 0.5
    return sample_affine(cell, (0, 0, width, height), (cos_r, sin_r, c, -sin_r, cos_r, f), wrap=True)


def placed_alpha(stamp: np.ndarray, width: int, height: int, x: int, y: int,
                 rotation: float) -> tuple[np.ndarray, tuple[int, int]]:
    """왼쪽 위 (x, y) 에 놓고 자기 중심으로 ``rotation``° 돌린 ``stamp`` 의 알파와 그 원점.

    글자 자신의 중심으로 돈다 — 이미지 중심으로 돌리면 모서리 워터마크가 화면 밖으로 날아간다.
    이미지 크기 오버레이를 통째로 돌리지 않고, 돌아간 도장이 닿는 사각형만 계산한다.
    """
    th, tw = stamp.shape[:2]
    cos_r, sin_r = _rotation_terms(rotation)
    half_w, half_h = tw / 2.0, th / 2.0
    pcx, pcy = x + half_w, y + half_h
    ext_x = abs(cos_r) * half_w + abs(sin_r) * half_h
    ext_y = abs(sin_r) * half_w + abs(cos_r) * half_h
    rx0 = max(0, int(math.floor(pcx - ext_x)) - 2)
    ry0 = max(0, int(math.floor(pcy - ext_y)) - 2)
    rx1 = min(int(width), int(math.ceil(pcx + ext_x)) + 2)
    ry1 = min(int(height), int(math.ceil(pcy + ext_y)) + 2)
    if rx1 <= rx0 or ry1 <= ry0:
        return np.zeros((0, 0), np.uint8), (0, 0)
    # 도장 좌표 s = R·(p + ½ − P) + (tw, th)/2 − ½   (P = 도장 중심)
    c = cos_r * (0.5 - pcx) + sin_r * (0.5 - pcy) + half_w - 0.5
    f = -sin_r * (0.5 - pcx) + cos_r * (0.5 - pcy) + half_h - 0.5
    alpha = sample_affine(stamp, (rx0, ry0, rx1, ry1), (cos_r, sin_r, c, -sin_r, cos_r, f), wrap=False)
    return alpha, (rx0, ry0)


# ─────────────────────────────────────────────
# 렌더링
# ─────────────────────────────────────────────

def render_text_watermark(img: np.ndarray, params: Mapping[str, Any], *, pixel_scale: float = 1.0) -> np.ndarray:
    """텍스트 워터마크를 합성한다. BGRA 입력은 BGRA 로 돌려준다(투명도 보존).

    params: text, fontFamily, fontSize(원본 px), color('#RRGGBB'), opacity(0~1),
    xPct/yPct(중심 %), rotation(도, 시계 방향), tile(bool), clamp(bool, 기본 True)
    """
    from PIL import Image
    text = str(params.get('text') or '').strip()
    if not text:
        raise WatermarkError('워터마크 텍스트를 입력하세요')
    scale = _num(pixel_scale, 1.0, 1e-4, 1.0e4)
    font_size = _num(params.get('fontSize'), 36.0, 1.0, float(MAX_FONT_PX))
    opacity = _num(params.get('opacity'), 0.5, 0.0, 1.0)
    rgb = parse_hex_color(params.get('color'))
    alpha_max = int(round(opacity * 255))
    x_pct = _num(params.get('xPct'), 50.0, 0.0, 100.0)
    y_pct = _num(params.get('yPct'), 50.0, 0.0, 100.0)
    rotation = _num(params.get('rotation'), 0.0, -3600.0, 3600.0)
    tile = _as_bool(params.get('tile'), False)
    clamp = _as_bool(params.get('clamp'), True)

    # 글자 한 벌을 한 번만 래스터화한 도장(알파) — 타일은 이걸 주기로 읽기만 한다.
    try:
        glyphs = rasterize_text(text, params.get('fontFamily'), max(1, int(round(font_size * scale))))
    except (Image.DecompressionBombError, MemoryError, ValueError) as exc:
        # Pillow 는 너무 긴 글자(MAX_STRING_LENGTH)·거대한 래스터를 ValueError/폭탄 오류로 막는다
        raise WatermarkError('워터마크 글자가 너무 큽니다 — 글자 크기나 길이를 줄이세요') from exc
    stamp = ((glyphs.astype(np.uint32) * alpha_max + 127) // 255).astype(np.uint8)
    th, tw = stamp.shape

    out = _prepare_base(img)
    height, width = out.shape[:2]
    if tile:
        gap = max(1, int(round(TILE_GAP_PX * scale)))
        alpha = tiled_alpha(stamp, width, height, tw + gap, th + gap, rotation)
        origin = (0, 0)
    elif rotation % 180.0 == 0.0:
        # 0°·180° — 외접 상자가 도장 그대로다(예전 결과와 픽셀 단위로 같다)
        x = centered_position(x_pct, width, tw, clamp)
        y = centered_position(y_pct, height, th, clamp)
        alpha, origin = placed_alpha(stamp, width, height, x, y, rotation)
    else:
        # 돌린 글자는 돌린 외접 상자가 영역 안에 들도록 민다 — '이미지 영역 내 제한'의 약속
        box_w, box_h = rotated_extent(tw, th, rotation)
        x = rotated_origin(x_pct, width, tw, box_w, clamp)
        y = rotated_origin(y_pct, height, th, box_h, clamp)
        alpha, origin = placed_alpha(stamp, width, height, x, y, rotation)
    blend_over(out, origin, alpha, (rgb[2], rgb[1], rgb[0]))
    return out


def render_image_watermark(img: np.ndarray, params: Mapping[str, Any], *, pixel_scale: float = 1.0) -> np.ndarray:
    """이미지 워터마크를 합성한다. BGRA 입력은 BGRA 로 돌려준다(투명도 보존).

    params: watermark_path, scale(원래 크기 대비 %, 기본 100), opacity(0~1),
    xPct/yPct(중심 %), clamp(bool, 기본 True — 원본보다 크면 줄이고 안쪽으로 민다)
    """
    from PIL import Image, ImageOps
    wm_path = str(params.get('watermark_path') or '')
    if not wm_path or not os.path.isfile(wm_path):
        raise WatermarkError('워터마크 이미지를 먼저 불러오세요')
    try:
        with Image.open(wm_path) as opened:
            wm = ImageOps.exif_transpose(opened).convert('RGBA')
    except (OSError, ValueError, SyntaxError, Image.DecompressionBombError) as exc:
        raise WatermarkError(f'워터마크 이미지를 열 수 없습니다: {exc}') from exc

    scale = _num(pixel_scale, 1.0, 1e-4, 1.0e4)
    scale_pct = _num(params.get('scale'), 100.0, MIN_SCALE_PCT, MAX_SCALE_PCT)
    opacity = _num(params.get('opacity'), 0.5, 0.0, 1.0)
    x_pct = _num(params.get('xPct'), 50.0, 0.0, 100.0)
    y_pct = _num(params.get('yPct'), 50.0, 0.0, 100.0)
    clamp = _as_bool(params.get('clamp'), True)

    out = _prepare_base(img)
    height, width = out.shape[:2]
    ratio = scale_pct / 100.0 * scale
    new_w = max(1, int(round(wm.width * ratio)))
    new_h = max(1, int(round(wm.height * ratio)))
    if clamp:
        fit = min(1.0, width / new_w, height / new_h)
        new_w, new_h = max(1, int(new_w * fit)), max(1, int(new_h * fit))

    x = centered_position(x_pct, width, new_w, clamp)
    y = centered_position(y_pct, height, new_h, clamp)
    # 이미지와 겹치는 부분만 만든다 — clamp 를 끄면 크기에 상한이 없어, 2000px 로고를 500% 로
    # 통째로 키우면 10000×10000 RGBA(≈400MB)를 잡은 뒤 대부분을 버렸다.
    vx0, vy0 = max(0, x), max(0, y)
    vx1, vy1 = min(width, x + new_w), min(height, y + new_h)
    if vx1 <= vx0 or vy1 <= vy0:
        return out
    part = np.asarray(resize_visible(wm, (new_w, new_h), (vx0 - x, vy0 - y, vx1 - x, vy1 - y)))
    alpha = part[..., 3]
    if opacity < 1.0:
        alpha = np.rint(alpha.astype(np.float64) * opacity).astype(np.uint8)
    # 워터마크 자신의 알파로 한 번만 합성 — paste(wm, pos, wm) 는 알파를 두 번 곱해
    # 불투명도가 제곱이 됐다
    blend_over(out, (vx0, vy0), alpha, part[..., 2::-1])
    return out


def resize_visible(wm, size: tuple[int, int], crop: tuple[int, int, int, int]):
    """``wm`` 을 ``size`` 로 키운(줄인) 결과 중 ``crop`` (x0, y0, x1, y1) 부분만 만든다.

    ``Image.resize(box=...)`` 는 필터 폭·표본 위치를 전체 크기 기준 그대로 쓰므로, 전체를
    리사이즈한 뒤 잘라 낸 것과 같은 픽셀이 나온다. 메모리는 ``crop`` 크기뿐이다.
    """
    from PIL import Image
    new_w, new_h = (int(v) for v in size)
    cx0, cy0, cx1, cy1 = (int(v) for v in crop)
    if (new_w, new_h) == wm.size:
        return wm.copy() if (cx0, cy0, cx1, cy1) == (0, 0, new_w, new_h) else wm.crop((cx0, cy0, cx1, cy1))
    sx, sy = wm.width / new_w, wm.height / new_h
    box = (max(0.0, cx0 * sx), max(0.0, cy0 * sy),
           min(float(wm.width), cx1 * sx), min(float(wm.height), cy1 * sy))
    return wm.resize((cx1 - cx0, cy1 - cy0), Image.LANCZOS, box=box)
