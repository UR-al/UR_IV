# core/editor_ops.py
"""에디터 이미지 연산 — 순수 함수(테스트 가능). Qt/브리지 의존 없음.

`ui/vue_bridge.py::_editor_process_impl`이 처리하지 못하던 오퍼레이션들의 실제 구현.
그동안 프론트가 보내던 `adv_color` / `auto_correct` / 필터 프리셋 / `flatten` /
`move_region` / `perspective`는 모든 elif를 통과해 **원본을 그대로 다시 저장**했다.
사용자 입장에선 "눌렀는데 아무 일도 안 남 + undo 스택만 늘어남"이었다.

입출력 규약
- 이미지는 OpenCV BGR 또는 BGRA (uint8) numpy 배열.
- 알파가 있으면 **보존**한다. 색 연산은 BGR 채널에만 적용한다.
  (`cv2.imread`가 기본으로 알파를 버리던 버그와 짝을 이루는 규약 — 배경 제거 후
   편집해도 투명도가 살아 있어야 한다.)
"""
import cv2
import numpy as np


# ── 알파 보존 헬퍼 ──────────────────────────────────────────────────────────

def split_alpha(img: np.ndarray):
    """(bgr, alpha) 반환. 알파가 없으면 alpha는 None."""
    if img.ndim == 3 and img.shape[2] == 4:
        return img[:, :, :3].copy(), img[:, :, 3].copy()
    return img, None


def merge_alpha(bgr: np.ndarray, alpha) -> np.ndarray:
    if alpha is None:
        return bgr
    if bgr.shape[:2] != alpha.shape[:2]:
        alpha = cv2.resize(alpha, (bgr.shape[1], bgr.shape[0]),
                           interpolation=cv2.INTER_NEAREST)
    return np.dstack([bgr, alpha])


def _preserve_alpha(fn):
    """BGR만 받는 함수를 BGRA에도 안전하게 적용."""
    def wrapper(img, *args, **kwargs):
        bgr, alpha = split_alpha(img)
        return merge_alpha(fn(bgr, *args, **kwargs), alpha)
    wrapper.__name__ = getattr(fn, '__name__', 'wrapped')
    wrapper.__doc__ = fn.__doc__
    return wrapper


def _clip8(arr) -> np.ndarray:
    return np.clip(arr, 0, 255).astype(np.uint8)


def _blend(original: np.ndarray, filtered: np.ndarray, strength: float) -> np.ndarray:
    """strength(0~1)로 원본과 결과를 선형 보간. 필터 프리셋 세기 슬라이더용."""
    s = max(0.0, min(1.0, float(strength)))
    if s >= 1.0:
        return filtered
    if s <= 0.0:
        return original
    return _clip8(original.astype(np.float32) * (1 - s) + filtered.astype(np.float32) * s)


# ── 자동 보정 ───────────────────────────────────────────────────────────────

@_preserve_alpha
def auto_correct(bgr: np.ndarray) -> np.ndarray:
    """자동 보정 — LAB 명도 CLAHE + 채널별 화이트밸런스.

    단순 히스토그램 스트레치는 색이 튀기 쉬워서, 명도만 국소 대비를 올리고
    색은 gray-world 화이트밸런스로 잡는다.
    """
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    lightness, a_ch, b_ch = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    lightness = clahe.apply(lightness)
    out = cv2.cvtColor(cv2.merge([lightness, a_ch, b_ch]), cv2.COLOR_LAB2BGR)

    # gray-world 화이트밸런스 (채널 평균을 전체 평균에 맞춤)
    out = out.astype(np.float32)
    means = out.reshape(-1, 3).mean(axis=0)
    target = float(means.mean())
    for ch in range(3):
        if means[ch] > 1e-3:
            out[:, :, ch] *= target / means[ch]
    return _clip8(out)


# ── 고급 색 보정 (AdvancedColorPanel) ───────────────────────────────────────

@_preserve_alpha
def adv_color(bgr: np.ndarray, black_point=0, white_point=255,
              gamma=1.0, temperature=0, tint=0, curves=None) -> np.ndarray:
    """커브 + 레벨(흑/백점) + 감마 + 색온도/틴트.

    black_point/white_point: 0~255 입력 레벨
    gamma: >1 밝게, <1 어둡게 (패널이 slider/10으로 보냄)
    temperature: -100~100 (+ = 따뜻/주황, - = 차갑/파랑)
    tint: -100~100 (+ = 마젠타, - = 초록)
    curves: {'rgb'|'r'|'g'|'b': [[x, y], ...]} 0~1 제어점. None 이면 건너뛴다.

    적용 순서는 PyQt 판(`tabs/editor/advanced_color_panel._build_adjusted`, 은퇴해 삭제됨)과 같다 —
    커브 → 레벨/감마 → 색온도. 순서가 바뀌면 같은 설정이 다른 그림을 낸다.
    """
    if curves:
        from core.curves import apply_curves, is_identity
        if not is_identity(curves):
            bgr = apply_curves(bgr, curves)

    black = max(0.0, min(254.0, float(black_point)))
    white = max(black + 1.0, min(255.0, float(white_point)))
    gamma = float(gamma)
    if not (gamma > 0):
        gamma = 1.0

    # 8bit LUT 한 번으로 레벨+감마 처리 (픽셀 루프 대비 수십 배 빠름)
    ramp = np.arange(256, dtype=np.float32)
    ramp = (ramp - black) * (255.0 / (white - black))
    ramp = np.clip(ramp, 0, 255)
    ramp = 255.0 * np.power(ramp / 255.0, 1.0 / gamma)
    lut = _clip8(ramp)
    out = cv2.LUT(bgr, lut)

    temp = float(temperature)
    tnt = float(tint)
    if temp or tnt:
        out = out.astype(np.float32)
        # BGR 순서: 색온도는 B↓R↑, 틴트는 G를 반대로
        out[:, :, 2] += temp * 0.5      # R
        out[:, :, 0] -= temp * 0.5      # B
        out[:, :, 1] -= tnt * 0.5       # G
        out = _clip8(out)
    return out


# ── 필터 프리셋 (ColorPanel) ────────────────────────────────────────────────

def _f_grayscale(bgr):
    return cv2.cvtColor(cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR)


def _f_sepia(bgr):
    """표준 세피아 행렬. 널리 쓰이는 계수는 RGB 기준이라 BGR 입출력에 맞게 재배열한다.

    RGB 기준:  R' = .393R + .769G + .189B  /  G' = .349R + .686G + .168B
               B' = .272R + .534G + .131B
    BGR 벡터 [B,G,R]에 그대로 곱하려면 각 행의 열 순서를 (B,G,R)로 뒤집고
    행 순서도 (B',G',R')이 되게 놓아야 한다.
    """
    kernel = np.array([
        [0.131, 0.534, 0.272],   # B'
        [0.168, 0.686, 0.349],   # G'
        [0.189, 0.769, 0.393],   # R'
    ], dtype=np.float32)
    return _clip8(cv2.transform(bgr, kernel))


def _f_sharpen(bgr):
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], dtype=np.float32)
    return _clip8(cv2.filter2D(bgr, -1, kernel))


def _f_warm(bgr):
    out = bgr.astype(np.float32)
    out[:, :, 2] *= 1.12   # R
    out[:, :, 0] *= 0.92   # B
    return _clip8(out)


def _f_cool(bgr):
    out = bgr.astype(np.float32)
    out[:, :, 0] *= 1.12   # B
    out[:, :, 2] *= 0.92   # R
    return _clip8(out)


def _f_soft(bgr):
    return cv2.GaussianBlur(bgr, (0, 0), 2.0)


def _f_invert(bgr):
    return cv2.bitwise_not(bgr)


def _f_emboss(bgr):
    # 커널 합이 1이라 filter2D 결과가 이미 uint8로 잘린다 → float로 받아 +128 후 클리핑
    kernel = np.array([[-2, -1, 0], [-1, 1, 1], [0, 1, 2]], dtype=np.float32)
    embossed = cv2.filter2D(bgr.astype(np.float32), -1, kernel)
    gray = cv2.cvtColor(_clip8(embossed + 128), cv2.COLOR_BGR2GRAY)
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)


def _f_sketch(bgr):
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    inverted = 255 - gray
    blurred = cv2.GaussianBlur(inverted, (21, 21), 0)
    # dodge blend — 0 나눗셈 방지
    sketch = cv2.divide(gray, 255 - blurred, scale=256)
    return cv2.cvtColor(sketch, cv2.COLOR_GRAY2BGR)


def _f_posterize(bgr, levels=5):
    step = max(2, int(levels))
    lut = _clip8(np.floor(np.arange(256) / (256.0 / step)) * (255.0 / (step - 1)))
    return cv2.LUT(bgr, lut)


def _f_vignette(bgr):
    h, w = bgr.shape[:2]
    kx = cv2.getGaussianKernel(w, w / 2.0)
    ky = cv2.getGaussianKernel(h, h / 2.0)
    mask = ky @ kx.T
    peak = mask.max()
    if peak > 0:
        mask = mask / peak
    return _clip8(bgr.astype(np.float32) * mask[:, :, None])


def _f_denoise(bgr):
    return cv2.fastNlMeansDenoisingColored(bgr, None, 7, 7, 7, 21)


# ColorPanel.vue의 presets[].name 과 1:1 (이름이 어긋나면 필터가 조용히 죽는다)
FILTERS = {
    'grayscale': _f_grayscale,
    'sepia': _f_sepia,
    'sharpen': _f_sharpen,
    'warm': _f_warm,
    'cool': _f_cool,
    'soft': _f_soft,
    'invert': _f_invert,
    'emboss': _f_emboss,
    'sketch': _f_sketch,
    'posterize': _f_posterize,
    'vignette': _f_vignette,
    'denoise': _f_denoise,
}

FILTER_NAMES = tuple(FILTERS)


def apply_filter(img: np.ndarray, name: str, strength: float = 1.0) -> np.ndarray:
    """필터 프리셋 적용. 모르는 이름이면 ValueError (조용히 무시하지 않는다)."""
    key = str(name or '').strip().lower()
    fn = FILTERS.get(key)
    if fn is None:
        raise ValueError(f"알 수 없는 필터: {name!r} (가능: {', '.join(FILTER_NAMES)})")
    bgr, alpha = split_alpha(img)
    return merge_alpha(_blend(bgr, fn(bgr), strength), alpha)


# ── 마스크 영역 이동 (MovePanel) ────────────────────────────────────────────

def _fill_hole(bgr: np.ndarray, mask: np.ndarray, fill_color: str) -> np.ndarray:
    """잘라낸 자리를 채운다. 'black'/'white'는 단색, 그 외는 주변 픽셀로 inpaint."""
    out = bgr.copy()
    key = str(fill_color or 'black').strip().lower()
    if key == 'white':
        out[mask > 127] = 255
    elif key == 'black':
        out[mask > 127] = 0
    else:
        out = cv2.inpaint(out, (mask > 127).astype(np.uint8), 3, cv2.INPAINT_TELEA)
    return out


@_preserve_alpha
def heal(bgr: np.ndarray, mask: np.ndarray, radius: int = 3) -> np.ndarray:
    """복원 브러시 — 마스크로 칠한 자리를 주변 픽셀로 메운다.

    `cv2.inpaint` 는 8UC1 마스크만 받는다. 프론트가 보내는 것은 PNG 라 3~4채널로
    디코딩되므로 여기서 단일 채널로 줄인다. 크기가 다르면 최근접으로 맞춘다 —
    보간하면 마스크 가장자리에 중간값이 생겨 칠하지 않은 곳까지 지워진다.
    """
    if mask is None:
        return bgr
    if mask.ndim == 3:
        mask = mask[:, :, 0] if mask.shape[2] < 4 else cv2.cvtColor(mask, cv2.COLOR_BGRA2GRAY)
    h, w = bgr.shape[:2]
    if mask.shape[:2] != (h, w):
        mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
    binary = (mask > 127).astype(np.uint8)
    if not np.any(binary):
        return bgr
    return cv2.inpaint(bgr, binary, max(1, int(radius)), cv2.INPAINT_TELEA)


def move_region(img: np.ndarray, mask: np.ndarray, dx: float = 0, dy: float = 0,
                rotation: float = 0, scale: float = 100,
                fill_color: str = 'black') -> np.ndarray:
    """마스크 영역을 잘라 이동/회전/확대해 다시 합성하고, 원래 자리는 채운다.

    rotation: 도(degree), scale: 퍼센트(100 = 등배).
    회전/확대는 마스크 영역의 무게중심을 기준으로 한다.
    """
    bgr, alpha = split_alpha(img)
    h, w = bgr.shape[:2]
    if mask is None or not np.any(mask > 127):
        return img

    binary = (mask > 127).astype(np.uint8) * 255

    # 잘라낸 자리 채우기 (원본에서 먼저)
    base = _fill_hole(bgr, binary, fill_color)

    # 이동/회전/확대 변환 — 마스크 무게중심 기준
    moments = cv2.moments(binary, binaryImage=True)
    if moments['m00'] > 0:
        cx = moments['m10'] / moments['m00']
        cy = moments['m01'] / moments['m00']
    else:
        cx, cy = w / 2.0, h / 2.0

    factor = max(0.01, float(scale) / 100.0)
    matrix = cv2.getRotationMatrix2D((cx, cy), float(rotation), factor)
    matrix[0, 2] += float(dx)
    matrix[1, 2] += float(dy)

    moved_pixels = cv2.warpAffine(bgr, matrix, (w, h), flags=cv2.INTER_LINEAR,
                                  borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))
    moved_mask = cv2.warpAffine(binary, matrix, (w, h), flags=cv2.INTER_NEAREST,
                                borderMode=cv2.BORDER_CONSTANT, borderValue=0)

    keep = moved_mask > 127
    out = base
    out[keep] = moved_pixels[keep]

    if alpha is not None:
        alpha_base = _fill_hole(cv2.cvtColor(alpha, cv2.COLOR_GRAY2BGR), binary, 'black')[:, :, 0]
        alpha_moved = cv2.warpAffine(alpha, matrix, (w, h), flags=cv2.INTER_NEAREST,
                                     borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        alpha_base[keep] = alpha_moved[keep]
        alpha = alpha_base
    return merge_alpha(out, alpha)


# ── 원근 보정 (perspective) ─────────────────────────────────────────────────

def perspective(img: np.ndarray, corners, width=None, height=None) -> np.ndarray:
    """네 꼭짓점을 직사각형으로 편다.

    corners: [[x,y] x4] — 좌상, 우상, 우하, 좌하 순서.
    """
    pts = np.array(corners, dtype=np.float32)
    if pts.shape != (4, 2):
        raise ValueError(f"perspective: 꼭짓점 4개가 필요합니다 (받음: {pts.shape})")

    if width is None or height is None:
        # 대변 길이의 최댓값으로 출력 크기 추정
        widths = [np.linalg.norm(pts[1] - pts[0]), np.linalg.norm(pts[2] - pts[3])]
        heights = [np.linalg.norm(pts[3] - pts[0]), np.linalg.norm(pts[2] - pts[1])]
        width = int(max(widths)) if width is None else int(width)
        height = int(max(heights)) if height is None else int(height)
    width = max(1, int(width))
    height = max(1, int(height))

    dst = np.array([[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]],
                   dtype=np.float32)
    matrix = cv2.getPerspectiveTransform(pts, dst)
    return cv2.warpPerspective(img, matrix, (width, height), flags=cv2.INTER_LINEAR)


# ── 레이어 병합 (DrawPanel) ─────────────────────────────────────────────────

def flatten(img: np.ndarray, overlay_bgra=None, opacity: float = 1.0) -> np.ndarray:
    """드로잉 오버레이(BGRA)를 베이스 이미지에 알파 합성.

    overlay가 없으면 원본을 그대로 돌려준다(그릴 게 없으면 병합도 없음).
    """
    if overlay_bgra is None:
        return img
    base_bgr, base_alpha = split_alpha(img)
    h, w = base_bgr.shape[:2]

    over = overlay_bgra
    if over.shape[:2] != (h, w):
        over = cv2.resize(over, (w, h), interpolation=cv2.INTER_LINEAR)
    if over.ndim == 3 and over.shape[2] == 4:
        over_bgr = over[:, :, :3].astype(np.float32)
        over_a = (over[:, :, 3].astype(np.float32) / 255.0) * max(0.0, min(1.0, float(opacity)))
    else:
        over_bgr = over[:, :, :3].astype(np.float32)
        over_a = np.full((h, w), max(0.0, min(1.0, float(opacity))), dtype=np.float32)

    over_a3 = over_a[:, :, None]
    out = _clip8(base_bgr.astype(np.float32) * (1 - over_a3) + over_bgr * over_a3)

    if base_alpha is not None:
        # 오버레이가 불투명하게 덮은 곳은 알파도 불투명해진다
        merged_a = _clip8(np.maximum(base_alpha.astype(np.float32), over_a * 255.0))
        return merge_alpha(out, merged_a)
    return out
