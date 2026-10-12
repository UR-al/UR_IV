"""SAM3 '얼굴 안에서만 찾기'(sam-extra 0.34.0) — 위젯·저장 키·확장 인자 대응표와 얼굴 모델 목록 (Qt 비의존 순수 로직).

확장은 SAM3 Mask 아코디언과 Refine 패널에 한 줄(켜기 · 얼굴 검출 모델 · 신뢰도)을 더했다. 앱은 T2I SAM3 카드·SAM3 Refine·
배치 SAM3 에 같은 줄을 둔다(Vue ``components/Sam3FaceCropFields.vue``). 이 모듈이 세 칸의 단일 표다
(``core/sam3_controlnet`` 과 같은 모양):

  * 위젯 dict 키(``face_crop``) = 저장 키 = ``'_sam3_' + 키`` 가 widget id
  * 확장 인자 키는 ``'sam3_' + 키`` (``core.sam3_args.SAM3_SPEC`` 의 맨 뒤 세 키 — 켤 때만 보낸다: ``apply_face_crop``)
  * 기본값은 SAM3_SPEC 에서 가져온다(확장 sam3ext/face_detect.py DEFAULT_MODEL·DEFAULT_CONFIDENCE — SEMANTIC_PINS)
  * 모델 칸은 자유 입력(LineEditProxy) + 추천 목록(items) — 목록은 ADetailer 모델 목록(``/adetailer/v1/ad_model``)에서
    확장 드롭다운과 같은 규칙으로 고른다(``model_choices``). 확장은 아무것도 받지 않으므로 목록에만 있고 파일이 없는
    이름은 생성 때 'SAM3 Error: FaceDetectorUnavailable' 로 알린다(core/sam_extra_notices).
"""
from __future__ import annotations

from typing import Iterable, Mapping

from core import sam3_args

# (위젯/저장 키, 프록시 종류) — 순서는 확장 SAM3 Mask 아코디언의 줄과 같다.
FACE_CROP_FIELDS: tuple[tuple[str, str], ...] = (
    ('face_crop', 'check'),
    ('face_crop_model', 'text'),        # ADetailer 모델 목록(추천) 또는 자유 입력 (LineEditProxy)
    ('face_crop_confidence', 'text'),
)
FACE_CROP_FIELD_KEYS: tuple[str, ...] = tuple(key for key, _kind in FACE_CROP_FIELDS)

# 확장 sam3ext/face_detect.py — 상자 모델 파일 확장자와 YOLO-World 표시(클래스 이름이 따로 필요해 뺀다)
MODEL_SUFFIX = '.pt'
_WORLD_MARK = '-world'

_SPEC = {key: (kind, default, extra) for key, kind, default, extra in sam3_args.SAM3_SPEC}


def spec_key(key: str) -> str:
    """위젯/저장 키 → 확장 인자 키 (face_crop → sam3_face_crop)."""
    return f'sam3_{key}'


def widget_id(key: str) -> str:
    """위젯/저장 키 → Vue widget id (face_crop → _sam3_face_crop)."""
    return f'_sam3_{key}'


def _as_text(value) -> str:
    if isinstance(value, bool):
        return 'true' if value else 'false'
    return str(value)


def default_values() -> dict:
    """저장 키 → 기본값(체크는 bool, 나머지는 위젯이 들고 있는 문자열)."""
    out = {}
    for key, kind in FACE_CROP_FIELDS:
        default = _SPEC[spec_key(key)][1]
        out[key] = bool(default) if kind == 'check' else _as_text(default)
    return out


def is_box_model(name: str) -> bool:
    """확장 ``face_detect._is_box_model`` 의 이름 규칙 — '.pt' 이고 이름(확장자 앞)에 '-world' 가 없다."""
    text = str(name or '').strip()
    lower = text.lower()
    if not lower.endswith(MODEL_SUFFIX):
        return False
    return _WORLD_MARK not in lower[:-len(MODEL_SUFFIX)]


def model_choices(names: Iterable) -> list[str]:
    """ADetailer 모델 이름 목록 → 얼굴 검출 모델 추천 목록 (확장 ``face_detect.find_face_models`` 와 같은 규칙).

    YOLO 상자 모델(.pt, YOLO-World 제외)만, 기본 모델 → 나머지 ``face*`` → 그 밖(hand_yolov8n 등) 순서, 같은 순위는
    대소문자 무시 이름순. 하위 폴더가 붙은 이름은 파일 이름만 쓴다(확장도 파일 이름으로 찾는다). 하나도 없으면 기본
    모델 하나.
    """
    picked: list[str] = []
    for raw in names or ():
        text = str(raw or '').strip().replace('\\', '/')
        name = text.rsplit('/', 1)[-1]
        if name and is_box_model(name) and name not in picked:
            picked.append(name)
    default = sam3_args.FACE_CROP_DEFAULT_MODEL

    def rank(name: str) -> tuple[int, str]:
        lowered = name.lower()
        if lowered == default:
            return (0, lowered)
        return (1 if lowered.startswith('face') else 2, lowered)

    picked.sort(key=rank)
    return picked or [default]


def _read(proxy, kind: str):
    if kind == 'check':
        return bool(proxy.isChecked())
    return proxy.text()


def read_settings(widgets: Mapping) -> dict:
    """프록시 → 저장 dict (``_get_sam3_settings`` 에 합친다)."""
    out = {}
    for key, kind in FACE_CROP_FIELDS:
        proxy = widgets.get(key)
        if proxy is not None:
            out[key] = _read(proxy, kind)
    return out


def apply_settings(widgets: Mapping, settings: Mapping | None) -> None:
    """저장 dict → 프록시. 없는 키(0.34.0 전 저장 파일·프리셋)는 확장 기본값(꺼짐)으로 채운다."""
    saved = settings if isinstance(settings, Mapping) else {}
    defaults = default_values()
    for key, kind in FACE_CROP_FIELDS:
        proxy = widgets.get(key)
        if proxy is None:
            continue
        value = saved.get(key, defaults[key])
        if kind == 'check':
            if isinstance(value, str):
                value = value.strip().lower() in ('true', '1', 'yes', 'on')
            proxy.setChecked(bool(value))
        else:
            proxy.setText(defaults[key] if value is None else _as_text(value))


def settings_to_state(widgets: Mapping) -> dict:
    """프록시 → SAM3 설정 키(``sam3_face_crop`` …) — ``_build_sam3_settings`` 가 합친다. 변환·범위는 build_state 가 한다."""
    return {spec_key(key): value for key, value in read_settings(widgets).items()}


def push_model_items(proxy, ad_models: Iterable) -> None:
    """얼굴 모델 칸의 추천 목록(items)을 ADetailer 모델 목록으로 채운다 — Vue 는 ``getProperty(id, 'items')`` 로 읽는다."""
    if proxy is not None and hasattr(proxy, 'setSuggestions'):
        proxy.setSuggestions(model_choices(ad_models))


def init_widgets(widgets: Mapping) -> None:
    """추천 목록을 기본 모델 하나로 먼저 채운 뒤 기본값을 넣는다(ADetailer 목록이 오면 push_model_items 가 바꾼다)."""
    push_model_items(widgets.get('face_crop_model'), ())
    apply_settings(widgets, None)
