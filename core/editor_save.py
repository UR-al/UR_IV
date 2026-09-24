"""에디터 저장 — 편집 결과를 사용자 파일로 인코딩해 쓴다 (Qt 비의존 순수 로직).

예전 '저장'(Ctrl+S)은 파일을 쓰지 않고 성공 토스트만 띄웠다. 편집본은
``image_cache/editor_temp`` 에만 남아 있다가 ``prune_editor_temp`` 가 지웠고,
'다른 이름으로 저장'은 JPEG/WebP 를 골라도 PNG 바이트를 그대로 복사했다.

여기서 정하는 규칙:

* **비파괴 저장** — '저장'은 사용자가 연 원본을 **절대 덮어쓰지 않는다**. 첫 저장은
  원본 옆에 ``<stem>_edited[_N].<ext>`` 를 새로 만들고(원본이 클립보드·복구본 같은
  임시 파일이면 기본 출력 폴더에 ``edited_<날짜_시각>.png``), 그 문서의 다음 저장은
  이 사본만 덮어쓴다. 덮어써도 되는 사본은 에디터가 이번 실행에서 직접 쓴 파일
  (``SavedCopyRegistry``)뿐이다 — 생성 원본(검열 전 원본)이 모자이크 저장 한 번에
  사라지던 문제를 막는다. 저장된 사본을 나중에 다시 열어 저장하면 ``_edited_2`` 로 간다.
* **메타데이터 보존** — 원본의 PNG 텍스트 청크(parameters·workflow·prompt 등),
  EXIF, ICC 를 읽어 새 파일에 그대로 싣는다. 다른 포맷으로 저장하면 parameters 를
  WebUI 호환 EXIF UserComment 로 옮긴다.
* **포맷** — 확장자에 맞게 다시 인코딩한다(PNG/JPEG/WebP). JPEG 는 알파가 없으므로
  '다른 이름으로 저장'에서 사용자가 JPEG 를 고르면 흰 바탕에 합성하고, 자동으로 정한
  대상이 JPEG 인데 편집본에 투명한 곳이 있으면(배경 제거 등) PNG 로 바꿔 저장한다.
* **드로잉 레이어** — 병합하지 않은 레이어(오버레이 PNG)도 저장본에 합성한다.
* **원자적 쓰기** — 같은 폴더의 임시 파일에 쓴 뒤 ``os.replace`` 한다. 인코딩이나
  디스크 오류로 원본이 반쯤 쓰인 채 남지 않는다.
* **undo 보호** — 덮어쓸 파일이 에디터 히스토리에 올라 있으면('다른 이름으로 저장'으로
  지금 연 파일을 고른 경우 등) 덮어쓰기 전 내용을 스냅숏으로 복사해 돌려준다. 프론트가
  히스토리의 그 경로를 스냅숏으로 바꿔서, 저장 뒤 undo 가 '저장 전 그림'을 보여 준다.

경로는 한글이 섞일 수 있다 — 파일 입출력은 전부 Python ``open``/PIL 로 하고
``cv2.imread``/``imwrite`` 는 쓰지 않는다.
"""
from __future__ import annotations

import base64
import io
import os
import re
import tempfile
import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Iterable, Optional

import numpy as np

from core.path_safety import strip_file_url

# 에디터가 쓸 수 있는 포맷. 이 밖(BMP/GIF/TIFF 등)은 '저장'이 덮어쓰지 않고 위치를 묻는다.
WRITABLE_FORMATS: dict[str, str] = {
    '.png': 'PNG',
    '.jpg': 'JPEG',
    '.jpeg': 'JPEG',
    '.webp': 'WEBP',
}
_DEFAULT_EXT_BY_FORMAT = {'PNG': '.png', 'JPEG': '.jpg', 'WEBP': '.webp'}

# QFileDialog 필터 — 선택한 필터로 확장자 없는 이름의 포맷을 정한다.
SAVE_DIALOG_FILTERS = 'PNG (*.png);;JPEG (*.jpg *.jpeg);;WebP (*.webp)'

JPEG_QUALITY = 95
WEBP_QUALITY = 95

_EXIF_IFD_POINTER = 0x8769
_EXIF_USER_COMMENT = 0x9286
_EXIF_ORIENTATION = 0x0112
_UNICODE_PREFIX = b'UNICODE\x00'


class EditorSaveError(Exception):
    """사용자에게 그대로 보여 줄 수 있는 저장 실패."""


# ─────────────────────────────────────────────
# 경로 판정
# ─────────────────────────────────────────────

def _norm(path: str) -> str:
    return os.path.normcase(os.path.abspath(os.path.realpath(str(path))))


def is_inside(path: str, directories: Iterable[str]) -> bool:
    """``path`` 가 ``directories`` 중 하나의 하위(또는 그 자체)인지."""
    target = _norm(path)
    for directory in directories:
        if not directory:
            continue
        root = _norm(directory)
        try:
            if os.path.commonpath([target, root]) == root:
                return True
        except ValueError:
            # 드라이브가 다르면 commonpath 가 ValueError — 하위가 아니다
            continue
    return False


def app_owned_dirs(app_root: str) -> list[str]:
    """앱이 스스로 지우는 폴더 — 여기에 저장하면 정리 작업이 사용자 파일을 지운다."""
    return [
        os.path.join(app_root, 'image_cache'),
        os.path.join(tempfile.gettempdir(), 'AIStudioPro_editor'),
    ]


def non_user_dirs(app_root: str) -> list[str]:
    """'저장'이 원본으로 덮어쓰면 안 되는 폴더 — 앱 캐시 + 시스템 임시 폴더.

    임시 폴더의 파일(압축 파일에서 바로 연 그림, 클립보드 붙여넣기)은 OS 가 언제든
    지울 수 있어서, 거기 덮어쓰는 건 저장이 아니다. 이런 원본은 위치를 묻는다.
    """
    return app_owned_dirs(app_root) + [tempfile.gettempdir()]


def format_for_path(path: str) -> Optional[str]:
    return WRITABLE_FORMATS.get(os.path.splitext(str(path))[1].lower())


def is_user_image(path: Optional[str], excluded_dirs: Iterable[str]) -> bool:
    """원본 자리에 그대로 덮어써도 되는 사용자 이미지인지."""
    if not path or not os.path.isfile(path):
        return False
    if format_for_path(path) is None:
        return False
    return not is_inside(path, excluded_dirs)


def ensure_extension(path: str, selected_filter: str = '') -> str:
    """확장자가 없거나 쓸 수 없는 확장자면 고른 필터의 확장자를 붙인다."""
    if format_for_path(path):
        return path
    lowered = (selected_filter or '').lower()
    if 'jpeg' in lowered or '*.jpg' in lowered:
        ext = '.jpg'
    elif 'webp' in lowered:
        ext = '.webp'
    else:
        ext = '.png'
    return path + ext


def unique_path(path: str) -> str:
    """이미 있으면 ``<stem>_2<ext>``, ``_3`` … 로 비어 있는 이름을 찾는다."""
    if not os.path.exists(path):
        return path
    stem, ext = os.path.splitext(path)
    n = 2
    while os.path.exists(f'{stem}_{n}{ext}'):
        n += 1
    return f'{stem}_{n}{ext}'


# 이미 편집 사본인 이름의 꼬리 — 사본을 다시 열어 저장하면 '_edited_edited' 가 아니라 '_edited_2'
_EDITED_SUFFIX_RE = re.compile(r'_edited(?:_\d+)?$', re.IGNORECASE)


def edited_copy_path(source_path: Optional[str], default_dir: str,
                     excluded_dirs: Iterable[str], now: Optional[float] = None) -> str:
    """'저장'이 새로 만들 사본 경로(비어 있는 이름). '다른 이름으로 저장'의 추천 경로도 같다.

    원본이 사용자 폴더에 있으면 그 옆에 ``<원본이름>_edited[_N].<원본확장자>``(쓸 수 없는
    포맷이면 .png), 임시 파일(클립보드·복구본·앱 캐시)이면 기본 폴더에
    ``edited_<날짜_시각>.png``. 원본 이름이 이미 ``_edited``/``_edited_N`` 로 끝나면 그
    꼬리를 떼고 번호만 올린다. 예전에는 ``edited_<uuid>_edited.png`` 가 추천돼 원본 이름을 잃었다.
    """
    excluded = list(excluded_dirs)
    if source_path and os.path.isfile(source_path) and not is_inside(source_path, excluded):
        folder = os.path.dirname(source_path)
        stem, ext = os.path.splitext(os.path.basename(source_path))
        stem = _EDITED_SUFFIX_RE.sub('', stem) or stem
        ext = ext.lower() if format_for_path(source_path) else '.png'
        return unique_path(os.path.join(folder, f'{stem}_edited{ext}'))
    stamp = time.strftime('%Y%m%d_%H%M%S', time.localtime(now if now is not None else time.time()))
    return unique_path(os.path.join(default_dir, f'edited_{stamp}.png'))


def suggest_save_path(source_path: Optional[str], default_dir: str,
                      excluded_dirs: Iterable[str], now: Optional[float] = None) -> str:
    """'다른 이름으로 저장' 대화상자의 첫 경로 — '저장'이 만들 사본 이름과 같다."""
    return edited_copy_path(source_path, default_dir, excluded_dirs, now)


def validate_chosen_target(path: str) -> Optional[str]:
    """사용자가 저장 대화상자에서 고른 파일 경로 검사. 문제가 있으면 사용자용 메시지.

    출력 폴더 설정용 ``safe_output_dir`` 는 드라이브 루트(``D:\\``)를 거부하지만, 사용자가
    대화상자에서 USB 루트 같은 곳을 직접 고른 건 정당하다 — 시스템 폴더만 막고
    폴더가 실제로 있는지만 본다.
    """
    from core.path_safety import is_forbidden_path
    try:
        resolved = os.path.realpath(os.path.abspath(str(path)))
    except (OSError, ValueError):
        return '저장 경로를 해석할 수 없습니다'
    if is_forbidden_path(resolved):
        return '이 폴더에는 저장할 수 없습니다'
    if not os.path.isdir(os.path.dirname(resolved)):
        return '저장할 폴더가 없습니다'
    return None


class SavedCopyRegistry:
    """에디터가 이번 실행에서 직접 쓴 파일(과 '다른 이름으로 저장'에서 사용자가 고른 파일).

    '저장'이 덮어써도 되는 건 여기 오른 파일뿐이다 — 사용자가 연 원본은 프론트가
    '내 사본'이라고 주장해도 덮어쓰지 않는다. 저장 작업 스레드에서 쓰고 UI 스레드에서
    읽으므로 Lock 으로 보호한다.
    """

    def __init__(self, limit: int = 4096):
        self._lock = threading.Lock()
        self._paths: dict[str, None] = {}
        self._limit = limit

    @staticmethod
    def _key(path: str) -> str:
        # file URL 은 스킴을 떼고 퍼센트 인코딩을 푼다(원시 경로의 '%' 는 이름의 일부) —
        # 백엔드 전체와 같은 규칙(core.path_safety.strip_file_url). 예전엔 'file:///' 만 떼어
        # 'file:///C:/x/a%20b.png' 를 글자 그대로의 'a%20b.png' 로 보고 'a b.png' 사본과 어긋났다.
        return _norm(strip_file_url(str(path or '')))

    def add(self, path: str) -> None:
        if not path:
            return
        key = self._key(path)
        with self._lock:
            self._paths.pop(key, None)
            self._paths[key] = None
            while len(self._paths) > self._limit:
                self._paths.pop(next(iter(self._paths)))

    def __contains__(self, path: object) -> bool:
        if not isinstance(path, str) or not path:
            return False
        key = self._key(path)
        with self._lock:
            return key in self._paths

    def __len__(self) -> int:
        with self._lock:
            return len(self._paths)


# ─────────────────────────────────────────────
# 메타데이터
# ─────────────────────────────────────────────

@dataclass
class SourceMetadata:
    """원본에서 읽은, 저장본에 다시 실을 메타데이터."""
    text: dict[str, str] = field(default_factory=dict)   # PNG tEXt/iTXt/zTXt
    exif: Optional[bytes] = None
    icc_profile: Optional[bytes] = None
    parameters: str = ''   # WebUI parameters (PNG 텍스트 또는 EXIF UserComment)

    def is_empty(self) -> bool:
        return not (self.text or self.exif or self.icc_profile or self.parameters)


def read_source_metadata(path: Optional[str]) -> SourceMetadata:
    """원본 파일의 메타데이터. 못 읽으면 빈 값 — 저장 자체를 막지는 않는다."""
    meta = SourceMetadata()
    if not path or not os.path.isfile(path):
        return meta
    try:
        from PIL import Image
        from core.image_metadata import _read_exif_user_comment
        with Image.open(path) as img:
            img.load()   # PNG 는 IDAT 뒤의 텍스트 청크까지 읽어야 한다
            text = getattr(img, 'text', None) or {}
            meta.text = {str(k): str(v) for k, v in text.items() if isinstance(k, str) and isinstance(v, str)}
            exif = img.info.get('exif')
            meta.exif = bytes(exif) if isinstance(exif, (bytes, bytearray)) and exif else None
            icc = img.info.get('icc_profile')
            meta.icc_profile = bytes(icc) if isinstance(icc, (bytes, bytearray)) and icc else None
            meta.parameters = meta.text.get('parameters') or _read_exif_user_comment(img) or ''
    except Exception:
        return SourceMetadata()
    return meta


def _build_exif(meta: SourceMetadata, *, with_user_comment: bool) -> Optional[bytes]:
    """원본 EXIF 를 이어받고 방향을 1로 되돌린다(픽셀은 이미 똑바로 세웠다).

    ``with_user_comment`` 면 parameters 를 WebUI 형식 UserComment 로 넣는다
    (JPEG/WebP 는 PNG 텍스트 청크가 없어 여기가 유일한 자리다).
    """
    if not meta.exif and not (with_user_comment and meta.parameters):
        return None
    from PIL import Image
    exif = Image.Exif()
    if meta.exif:
        try:
            exif.load(meta.exif)
        except Exception:
            exif = Image.Exif()
    if _EXIF_ORIENTATION in exif:
        exif[_EXIF_ORIENTATION] = 1
    if with_user_comment and meta.parameters:
        exif.get_ifd(_EXIF_IFD_POINTER)[_EXIF_USER_COMMENT] = (
            _UNICODE_PREFIX + meta.parameters.encode('utf-16-be'))
    data = exif.tobytes()
    return data or None


# ─────────────────────────────────────────────
# 픽셀
# ─────────────────────────────────────────────

_PNG_SIGNATURE = b'\x89PNG\r\n\x1a\n'


def load_image_array(path: str) -> np.ndarray:
    """편집 결과를 RGB/RGBA uint8 배열로. 방향·투명도는 에디터가 보는 것과 같게 읽는다.

    PNG 는 에디터 디코드(``core.editor_preview.imread_unchanged``)를 그대로 쓴다(``_load_png_like_editor``).
    그 밖의 포맷은 Pillow 로 읽고, EXIF 방향은 에디터·브라우저처럼 진짜 EXIF 만 본다
    (``core.cv_io.image_exif_orientation`` — ``ImageOps.exif_transpose`` 는 XMP 방향까지 따른다).
    TIFF 는 Pillow 가 ``load()`` 에서 방향을 이미 적용하고 태그를 지운다 — 그땐 다시 돌리지 않는다.
    Pillow 로는 ``core.cv_io.open_pil_image`` 로 연다 — 경로로 열면 방향 5~8 무압축 TIFF(L·RGBA·I;16 등)
    가 mmap 지름길에서 뒤섞이고, XMP 에만 방향이 적힌 TIFF 가 load() 에서 돌았다(Codex S5 #1-a·#1-b).
    (TIFF 를 cv2 로 읽지 않는 건 OpenCV 가 LA(흑백+알파) TIFF 의 알파를 버리기 때문 — 배치 변환이 투명도를 잃는다.)
    """
    pixels = _load_png_like_editor(path)
    if pixels is not None:
        return pixels
    from core.cv_io import apply_exif_orientation, image_exif_orientation, open_pil_image
    try:
        with open_pil_image(path) as opened:
            # 방향은 load() 전에 읽는다 — PNG 는 IDAT 뒤 eXIf 를 열린 파일에서 훑는다
            orientation = image_exif_orientation(opened)
            opened.load()
            if opened.format == 'TIFF' and _EXIF_ORIENTATION not in getattr(opened, 'tag_v2', {}):
                # Pillow TIFF 디코더(TiffImageFile.load_end)가 방향을 적용하고 태그를 지웠다. 또 돌리면
                # 3·2·4… 는 저장된 그대로, 6·8 은 180° 뒤집힌 채 누웠다(Codex S5 #1). 태그가 남아 있으면
                # (적용하지 않는 Pillow) 아래에서 직접 돌린다.
                orientation = 1
            img = opened.copy()   # close() 가 픽셀 코어를 버린다 — with 밖에서 쓸 사본(info 포함)
    except Exception as exc:
        raise EditorSaveError(f'편집 이미지를 읽을 수 없습니다: {exc}') from exc
    if img.mode in ('I;16', 'I;16B', 'I;16L', 'I', 'F'):
        # 16비트 흑백 — 그대로 RGB 로 바꾸면 255 초과가 전부 흰색으로 잘린다
        raw = np.asarray(img, dtype=np.float64)
        peak = 65535.0 if img.mode.startswith('I;16') or raw.max(initial=0) > 255 else 255.0
        arr = np.clip(raw / peak * 255.0, 0, 255).astype(np.uint8)
        key = img.info.get('transparency')
        if isinstance(key, int) and not isinstance(key, bool):
            # tRNS 투명색(흑백 키) — 키와 같은 샘플이 투명하다
            pixels = np.dstack([arr, arr, arr, np.where(raw == key, 0, 255).astype(np.uint8)])
        else:
            pixels = np.dstack([arr, arr, arr])
    else:
        # 알파 채널뿐 아니라 투명색(팔레트·GIF 의 투명 인덱스 등)도 알파다 — PIL convert('RGBA') 가
        # 투명색을 알파 0 으로 바꾼다. (PNG tRNS 키는 위 에디터 경로가 맡는다.)
        has_alpha = img.mode in ('RGBA', 'LA', 'PA') or 'transparency' in img.info
        pixels = np.asarray(img.convert('RGBA' if has_alpha else 'RGB'), dtype=np.uint8)
    return np.ascontiguousarray(apply_exif_orientation(pixels, orientation))


def _load_png_like_editor(path: str) -> Optional[np.ndarray]:
    """PNG → 에디터와 같은 디코드의 RGB/RGBA uint8 배열. PNG 가 아니거나 cv2 가 못 읽으면 None.

    Pillow ``convert('RGBA')`` 는 일부 tRNS 투명색을 틀리게 다룬다 — 1·2·4비트 흑백은 늘리지 않은
    키를 0/85/170/255 로 늘린 픽셀과 비교해 아무것도 투명해지지 않고, 16비트 RGB 는 8비트로 줄인
    뒤 비교해 키가 아닌 어두운 색(예: (200,200,200)/65535 → 0)까지 투명해진다. 에디터는 cv2(libpng
    tRNS→알파, 흑백은 ``gray_key_to_alpha`` 가 키를 깊이에 맞춰 늘림)로 읽어 맞게 보여 주므로
    '다른 이름으로 저장'·병합 안 된 레이어 저장·배치 변환도 같은 경로로 읽는다. 방향도 같은 규칙
    (eXIf 만 — XMP 무시)이고, 16비트는 에디터 프리뷰와 같은 식(``to_display_uint8``)으로 줄인다.
    """
    try:
        with open(path, 'rb') as handle:
            if handle.read(len(_PNG_SIGNATURE)) != _PNG_SIGNATURE:
                return None
    except (OSError, TypeError, ValueError):
        return None
    from core.editor_preview import imread_unchanged, to_display_uint8
    img = imread_unchanged(path)
    if img is None:
        return None
    img = to_display_uint8(img)
    if img.ndim == 3 and img.shape[2] == 1:
        img = img[:, :, 0]
    if img.ndim == 2:
        return np.ascontiguousarray(np.dstack([img, img, img]))
    if img.ndim != 3:
        return None
    channels = img.shape[2]
    if channels == 2:                       # 흑백 + 알파
        gray = img[:, :, 0]
        return np.ascontiguousarray(np.dstack([gray, gray, gray, img[:, :, 1]]))
    if channels == 3:                       # BGR → RGB
        return np.ascontiguousarray(img[:, :, ::-1])
    if channels == 4:                       # BGRA → RGBA
        return np.ascontiguousarray(img[:, :, [2, 1, 0, 3]])
    return None


def decode_overlay_base64(data: Optional[str]) -> Optional[np.ndarray]:
    """드로잉 레이어 PNG(data URL 허용) → RGBA 배열. 비었거나 완전 투명이면 None."""
    if not data or not isinstance(data, str):
        return None
    from PIL import Image
    raw = data.split(',', 1)[1] if data.startswith('data:') and ',' in data else data
    try:
        with Image.open(io.BytesIO(base64.b64decode(raw))) as opened:
            arr = np.asarray(opened.convert('RGBA'), dtype=np.uint8).copy()
    except Exception as exc:
        raise EditorSaveError(f'드로잉 레이어를 읽지 못했습니다: {exc}') from exc
    if not arr[:, :, 3].any():
        return None
    return arr


def composite_overlay(base: np.ndarray, overlay: Optional[np.ndarray], opacity: float = 1.0) -> np.ndarray:
    """병합 안 된 드로잉 레이어를 저장본에 합성한다 — '레이어 병합'과 같은 식을 쓴다."""
    if overlay is None:
        return base
    from core.editor_ops import flatten
    # flatten 은 채널 순서를 가리지 않는다(앞 3채널=색, 4번째=알파) — RGB 끼리 넘기면 된다
    return flatten(base, overlay, opacity)


def _to_pil(arr: np.ndarray):
    from PIL import Image
    if arr.ndim == 3 and arr.shape[2] == 4:
        return Image.fromarray(arr, 'RGBA')
    if arr.ndim == 3 and arr.shape[2] == 3:
        return Image.fromarray(arr, 'RGB')
    if arr.ndim == 2:
        return Image.fromarray(arr, 'L').convert('RGB')
    raise EditorSaveError(f'지원하지 않는 이미지 형태입니다: {arr.shape}')


def has_transparency(arr: np.ndarray) -> bool:
    """알파 채널에 불투명(255)이 아닌 곳이 있는지 — JPEG 로 쓰면 잃는 정보가 있는지."""
    return bool(arr.ndim == 3 and arr.shape[2] == 4 and (arr[:, :, 3] < 255).any())


def alpha_safe_target(target_path: str, pixels: np.ndarray) -> str:
    """자동으로 정한 대상이 JPEG 인데 투명한 곳이 있으면 같은 이름의 PNG(비어 있는 이름)로.

    배경을 지운 결과를 JPEG 원본 형식으로 쓰면 투명 영역이 흰색으로 뭉개진다 — 사용자가
    화면에서 본 것과 다른 파일이 남는다.
    """
    if format_for_path(target_path) != 'JPEG' or not has_transparency(pixels):
        return target_path
    return unique_path(os.path.splitext(target_path)[0] + '.png')


def _flatten_on_white(img):
    """JPEG 는 알파가 없다 — 투명한 곳은 흰 바탕으로 채운다(검게 뭉개지지 않게)."""
    from PIL import Image
    if img.mode != 'RGBA':
        return img.convert('RGB')
    background = Image.new('RGB', img.size, (255, 255, 255))
    background.paste(img, mask=img.getchannel('A'))
    return background


def encode_image(arr: np.ndarray, fmt: str, meta: Optional[SourceMetadata] = None) -> bytes:
    """배열을 ``fmt``(PNG/JPEG/WEBP)로 인코딩하면서 원본 메타데이터를 싣는다."""
    from PIL import PngImagePlugin
    meta = meta or SourceMetadata()
    img = _to_pil(arr)
    out = io.BytesIO()
    options: dict = {}
    if meta.icc_profile:
        options['icc_profile'] = meta.icc_profile
    if fmt == 'PNG':
        info = PngImagePlugin.PngInfo()
        for key, value in meta.text.items():
            info.add_text(key, value)   # latin-1 로 안 되는 값은 PIL 이 iTXt 로 쓴다
        if meta.parameters and 'parameters' not in meta.text:
            # JPEG/WebP 원본을 PNG 로 — EXIF 에 있던 parameters 를 PNG 표준 자리로 옮긴다
            info.add_text('parameters', meta.parameters)
        options['pnginfo'] = info
        exif = _build_exif(meta, with_user_comment=False)
        if exif:
            options['exif'] = exif
        img.save(out, format='PNG', **options)
    elif fmt == 'JPEG':
        exif = _build_exif(meta, with_user_comment=True)
        if exif:
            options['exif'] = exif
        _flatten_on_white(img).save(out, format='JPEG', quality=JPEG_QUALITY, subsampling=0, **options)
    elif fmt == 'WEBP':
        exif = _build_exif(meta, with_user_comment=True)
        if exif:
            options['exif'] = exif
        img.save(out, format='WEBP', quality=WEBP_QUALITY, method=4, **options)
    else:
        raise EditorSaveError(f'지원하지 않는 저장 형식입니다: {fmt}')
    return out.getvalue()


# ─────────────────────────────────────────────
# 파일 쓰기
# ─────────────────────────────────────────────

def atomic_write_bytes(path: str, data: bytes) -> None:
    """같은 폴더 임시 파일에 다 쓴 다음 교체한다 — 실패해도 원본은 그대로다."""
    folder = os.path.dirname(os.path.abspath(path)) or '.'
    tmp = os.path.join(folder, f'.{os.path.basename(path)}.{uuid.uuid4().hex}.tmp')
    try:
        with open(tmp, 'wb') as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, path)
    except OSError as exc:
        try:
            if os.path.exists(tmp):
                os.remove(tmp)
        except OSError:
            pass
        raise EditorSaveError(f'파일을 쓸 수 없습니다 ({exc.strerror or exc})') from exc


def snapshot_file(path: str, snapshot_dir: str, prefix: str = 'saved_orig') -> str:
    """파일 내용을 에디터 임시 폴더로 복사한다(덮어쓰기 전 사본, 복구본의 작업 사본).

    ``copy2`` 가 아니라 ``copyfile`` 이다 — 원본의 오래된 mtime 을 물려받으면
    ``prune_editor_temp`` 가 이 스냅숏을 '가장 오래된 것'으로 보고 먼저 지운다.
    """
    import shutil
    os.makedirs(snapshot_dir, exist_ok=True)
    ext = os.path.splitext(path)[1].lower() or '.png'
    dst = os.path.join(snapshot_dir, f'{prefix}_{uuid.uuid4().hex}{ext}')
    try:
        shutil.copyfile(path, dst)
    except OSError as exc:
        raise EditorSaveError(f'사본을 만들지 못했습니다 ({exc.strerror or exc})') from exc
    return dst


def write_autosave_snapshot(src: str, dst: str, overlay_base64: Optional[str] = None,
                            overlay_opacity: float = 1.0) -> bool:
    """크래시 복구본 — 확정 이미지 ``src`` 에 병합 안 한 드로잉 레이어를 합성해 ``dst`` 에 쓴다.

    예전 자동저장은 확정 이미지 파일만 복사했다. 드로잉 레이어는 브라우저 캔버스에만 있어서,
    그림만 그리고 크래시가 나면 '자동저장 N분 전' 표시가 있었는데도 복구본에 그림이 없었다.
    수동 저장(:func:`save_edited_image`)과 같은 합성(:func:`composite_overlay`)을 쓴다.

    :param overlay_opacity: 0~1 (프론트 슬라이더 0~100 을 호출부가 나눠 넘긴다).
    :returns: 레이어를 합성했으면 True. 레이어가 없거나 완전 투명하면 예전처럼 그대로 복사하고 False.
    :raises EditorSaveError: 이미지·레이어를 읽거나 쓰지 못했을 때.
    """
    import shutil
    overlay = decode_overlay_base64(overlay_base64)
    if overlay is None:
        try:
            shutil.copy2(src, dst)
        except OSError as exc:
            raise EditorSaveError(f'복구본을 쓰지 못했습니다 ({exc.strerror or exc})') from exc
        return False
    try:
        opacity = max(0.0, min(1.0, float(overlay_opacity)))
    except (TypeError, ValueError):
        opacity = 1.0
    pixels = composite_overlay(load_image_array(src), overlay, opacity)
    # 복구본 이름은 늘 .png 다 — 원본이 JPEG 여도 합성본은 PNG 로(투명도·화질 보존)
    atomic_write_bytes(dst, encode_image(pixels, 'PNG', read_source_metadata(src)))
    return True


def _matching_protected(target: str, protect_paths: Iterable[str]) -> list[str]:
    """``protect_paths`` 중 ``target`` 과 같은 파일을 가리키는 원래 문자열들."""
    key = _norm(target)
    matches = []
    for raw in protect_paths or ():
        if not isinstance(raw, str) or not raw:
            continue
        try:
            # SavedCopyRegistry._key 와 같은 규칙 — file URL 만 디코드, 원시 경로는 글자 그대로
            if _norm(strip_file_url(raw)) == key and raw not in matches:
                matches.append(raw)
        except (OSError, ValueError):
            continue
    return matches


def _same_existing_file(a: Optional[str], b: Optional[str]) -> bool:
    if not a or not b or not os.path.exists(a) or not os.path.exists(b):
        return False
    try:
        return _norm(a) == _norm(b)
    except (OSError, ValueError):
        return False


def _unchanged_result(path: str) -> dict:
    size = _probe_size(path)
    return {'ok': True, 'unchanged': True, 'path': _slash(path),
            'format': format_for_path(path) or '', 'width': size[0], 'height': size[1]}


def save_edited_image(edited_path: str, target_path: str, *,
                      metadata_path: Optional[str] = None,
                      overlay_base64: Optional[str] = None,
                      overlay_opacity: float = 1.0,
                      protect_paths: Iterable[str] = (),
                      snapshot_dir: Optional[str] = None,
                      unchanged_source: Optional[str] = None,
                      alpha_to_png: bool = False) -> dict:
    """편집 결과(+드로잉 레이어)를 ``target_path`` 에 그 확장자 포맷으로 저장한다.

    :param metadata_path: 메타데이터를 가져올 원본. 덮어쓰는 경우 쓰기 **전에** 읽는다.
    :param protect_paths: 에디터 히스토리가 참조하는 경로. ``target`` 이 여기 있으면
        덮어쓰기 전 내용을 ``snapshot_dir`` 에 복사하고 결과에 알린다.
    :param unchanged_source: 편집본이 이 파일 그 자체이고 드로잉도 없으면 아무것도 쓰지
        않고 '변경 없음'으로 답한다('저장'이 원본을 열기만 하고 누른 경우 — 똑같은 사본을
        만들 이유가 없다).
    :param alpha_to_png: 대상이 JPEG 인데 투명한 곳이 있으면 PNG 로 바꿔 쓴다
        (자동으로 정한 대상에만 — 사용자가 JPEG 를 고른 '다른 이름으로 저장'은 흰 바탕 합성).
    :returns: ``{'ok': True, 'path', 'format', 'width', 'height', 'unchanged'?,
        'alpha_png'?, 'replaced_paths'?, 'snapshot_path'?}``
    :raises EditorSaveError: 사용자에게 보여 줄 실패 사유.
    """
    fmt = format_for_path(target_path)
    if fmt is None:
        raise EditorSaveError('PNG · JPEG · WebP 로만 저장할 수 있습니다')
    if not edited_path or not os.path.isfile(edited_path):
        raise EditorSaveError('저장할 편집 이미지를 찾을 수 없습니다')

    overlay = decode_overlay_base64(overlay_base64)
    if overlay is None and _same_existing_file(edited_path, unchanged_source):
        return _unchanged_result(unchanged_source)
    if overlay is None and _same_existing_file(edited_path, target_path):
        # 원본을 열고 아무것도 안 바꿨다 — 다시 인코딩하면 JPEG 는 화질만 깎인다
        return _unchanged_result(target_path)

    # 덮어쓰기 전에 읽어야 한다 — 쓰고 나면 원본 메타가 사라진다
    meta = read_source_metadata(metadata_path) if metadata_path else SourceMetadata()
    pixels = load_image_array(edited_path)
    try:
        opacity = max(0.0, min(1.0, float(overlay_opacity)))
    except (TypeError, ValueError):
        opacity = 1.0
    pixels = composite_overlay(pixels, overlay, opacity)
    alpha_png = False
    if alpha_to_png:
        safe_target = alpha_safe_target(target_path, pixels)
        if safe_target != target_path:
            target_path, fmt, alpha_png = safe_target, 'PNG', True
    data = encode_image(pixels, fmt, meta)

    result: dict = {'ok': True, 'path': _slash(target_path), 'format': fmt,
                    'width': int(pixels.shape[1]), 'height': int(pixels.shape[0])}
    if alpha_png:
        result['alpha_png'] = True
    if os.path.exists(target_path):
        replaced = _matching_protected(target_path, protect_paths)
        if replaced:
            if not snapshot_dir:
                raise EditorSaveError('덮어쓰기 전 사본을 둘 폴더가 없습니다')
            result['snapshot_path'] = _slash(snapshot_file(target_path, snapshot_dir))
            result['replaced_paths'] = replaced
    atomic_write_bytes(target_path, data)
    return result


def _probe_size(path: str) -> tuple[int, int]:
    try:
        from PIL import Image
        with Image.open(path) as img:
            return int(img.width), int(img.height)
    except Exception:
        return 0, 0


def _slash(path: str) -> str:
    return str(path).replace('\\', '/')
