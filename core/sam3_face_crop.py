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

ComfyUI(앱 노드 팩 1.8.0): ``ForgeNeoSAM3Mask`` 의 선택 입력 ``face_crop``·``face_crop_model``·``face_crop_confidence``
(확장 키에서 'sam3_' 를 뗀 이름)가 같은 계산을 한다 — 팩 ``sam3_face_crop*.py`` 는 sam-extra 의 얼굴 crop(v0.34.0 + v0.34.1 의
crop 키우기)을 그대로 옮긴 것이다(sam-extra 자체 코드, 작성자의 MIT 허락). 얼굴 검출은 확장처럼 ultralytics YOLO 라 ComfyUI 의
Python 에 ultralytics 가 있어야 한다 — 노드의 정보용 COMBO ``face_crop_detector`` 가 그 여부를 /object_info 로 알린다
(``comfy_support``). 모델은 앱 Forge ``models/adetailer`` 의 파일을 절대 경로로 넘긴다(``resolve_model_path`` — SAM3
체크포인트처럼). 컴파일러(core/comfy_workflow_compiler._sam3_face_crop_inputs)는 켠 요청만 세 입력을 넣고, 옛 팩·검출기 없음은
이유와 함께 멈춘다. 카드는 연결 때 받은 상태(위젯 ``_sam3_face_crop`` 의 ``comfySupport`` 속성, ``push_comfy_support``)로
켜기를 막고 이유를 보인다.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable, Mapping, Optional

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


# ── ComfyUI (앱 노드 팩 1.8.0 ForgeNeoSAM3Mask — comfy_custom_nodes/ai_studio_forge_parity/sam3_face_crop*.py) ─────────
COMFY_NODE_CLASS = 'ForgeNeoSAM3Mask'
# 노드 입력 = 확장 인자 키에서 'sam3_' 를 뗀 것(FACE_CROP_FIELD_KEYS 와 같은 순서)
COMFY_INPUTS: tuple[str, ...] = FACE_CROP_FIELD_KEYS
# 정보용 COMBO(팩 sam3_face_detect.DETECTOR_INPUT·DETECTOR_READY) — 그 ComfyUI 가 ultralytics 를 불러올 수 있으면 'ultralytics'
COMFY_DETECTOR_INPUT = 'face_crop_detector'
COMFY_DETECTOR_READY = 'ultralytics'
COMFY_MIN_PACK_VERSION = '1.8.0'      # 얼굴 crop 이 처음 들어온 팩(core/comfy_node_pack.PACK_VERSION 은 이 이상)
# comfy_support 결과 — frontend/src/utils/sam3FaceCrop.ts ComfyFaceCropSupport 와 같은 문자열
COMFY_READY = 'ready'
COMFY_NO_DETECTOR = 'no_detector'
COMFY_OLD_PACK = 'old_pack'
COMFY_SUPPORT_PROPERTY = 'comfySupport'   # _sam3_face_crop 위젯 속성 — utils/sam3FaceCrop.ts COMFY_SUPPORT_PROPERTY
# 확장 sam3ext/face_detect.ADETAILER_FOLDER — Forge models 아래 얼굴 모델 폴더
ADETAILER_FOLDER = 'adetailer'


def _combo_choices(spec: Any) -> Optional[list]:
    """/object_info 의 COMBO 입력 → 선택지(옛 ``[[...], {...}]`` 모양과 새 ``["COMBO", {"options": [...]}]`` 모양)."""
    if not isinstance(spec, (list, tuple)) or not spec:
        return None
    head = spec[0]
    if isinstance(head, (list, tuple)):
        return [str(item) for item in head]
    if head == 'COMBO' and len(spec) > 1 and isinstance(spec[1], Mapping):
        options = spec[1].get('options')
        if isinstance(options, (list, tuple)):
            return [str(item) for item in options]
    return None


def comfy_support(object_info: Any) -> Optional[str]:
    """ComfyUI ``/object_info`` → 얼굴 crop 을 쓸 수 있나: ``COMFY_READY`` · ``COMFY_NO_DETECTOR``(팩은 1.8.0 이상인데
    ultralytics 를 못 불러옴) · ``COMFY_OLD_PACK``(노드가 없거나 face_crop 입력·검출기 COMBO 가 없음 — 1.8.0 전 팩, 갱신 뒤
    재시작 전). 판단 재료가 없으면(문서 없음, 클래스 이름만 알리는 가벼운 탐색) None — 그때 컴파일러는 그대로 보낸다."""
    if not isinstance(object_info, Mapping):
        return None
    node = object_info.get(COMFY_NODE_CLASS)
    if not isinstance(node, Mapping):
        return COMFY_OLD_PACK
    inputs = node.get('input')
    if not isinstance(inputs, Mapping):
        return None
    sections = [inputs.get(key) for key in ('required', 'optional')]
    fields: dict = {}
    for section in sections:
        if isinstance(section, Mapping):
            fields.update(section)
    if not fields:
        return None
    if not set(COMFY_INPUTS) <= set(fields):
        return COMFY_OLD_PACK
    choices = _combo_choices(fields.get(COMFY_DETECTOR_INPUT))
    if choices is None:
        return COMFY_OLD_PACK
    return COMFY_READY if COMFY_DETECTOR_READY in choices else COMFY_NO_DETECTOR


def resolve_model_path(name: Any, models_root: Optional[Path] = None) -> str:
    """얼굴 검출 모델 이름 → 앱 Forge ``models/adetailer`` 의 절대 경로(확장 ``face_detect.resolve_face_model`` 의 규칙 —
    폴더 바로 아래 → 하위 폴더에서 같은 파일 이름(대소문자 무시), .pt 이고 YOLO-World 가 아님). ComfyUI 노드는 절대 경로를
    그대로 쓰고, 못 찾으면 이름 그대로 넘겨 노드가 ComfyUI 의 models/ultralytics·adetailer 에서 찾는다. 빈 값은 기본 모델."""
    value = str(name or '').strip() or sam3_args.FACE_CROP_DEFAULT_MODEL
    path = Path(value)
    if path.is_absolute():
        return value
    if models_root is None:
        try:
            from core.forge_modules import get_forge_root
            models_root = get_forge_root()
        except Exception:
            return value
    folder = Path(models_root) / ADETAILER_FOLDER

    def usable(candidate: Path) -> bool:
        return candidate.is_file() and is_box_model(candidate.name)

    try:
        direct = folder / value
        if usable(direct):
            return str(direct.resolve())
        if folder.is_dir():
            for candidate in sorted(folder.rglob(f'*{MODEL_SUFFIX}'), key=lambda p: p.name.lower()):
                if usable(candidate) and candidate.name.lower() == path.name.lower():
                    return str(candidate.resolve())
    except OSError:
        pass
    return value


def push_comfy_support(host, support: Optional[str]) -> None:
    """연결된 ComfyUI 의 얼굴 crop 상태(``comfy_support``)를 카드로(``_sam3_face_crop`` 의 ``comfySupport`` 속성). Forge·모름·
    끊김은 None — 카드는 기능 스냅샷 규칙을 쓴다."""
    bridge = getattr(host, 'vue_bridge', None)
    push = getattr(bridge, 'pushWidgetProperty', None)
    if not callable(push):
        return
    try:
        push(widget_id('face_crop'), COMFY_SUPPORT_PROPERTY, support)
    except RuntimeError:
        pass   # 종료 중 QObject 가 이미 사라졌다
