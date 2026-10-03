# core/colorcraft.py
"""Colorcraft (sam-extra ``scripts/colorcraft.py``) — 샘플링 중 latent 색 보정. 앱 설정·상태 문자열·infotext·체인 판정.
순수 로직(Qt·네트워크·torch 없음).

확장의 인자 모델(sam3ext/colorcraft/spec.py): 수정자 10개(``I``…``X``), 마스크 잎 10개(``M1``…``M10``), 조합 5개
(``C1``…``C5`` — 조합 ``Ck`` 는 잎과 **앞선** 조합만 고른다). 각 칸의 이름·범위·기본값·선택지는 여기 표
(``MODIFIER_FIELDS``·``LEAF_FIELDS``·``COMBO_FIELDS``)가 같은 뜻으로 다시 적는다.

- **보내는 형식**: 첫 인자 하나에 확장 infotext 문자열(``v1;mods=I,III;masking=1;…`` — ``to_infotext``). 확장은 그 문자열이
  오면 나머지 66칸(Forge 가 ``ui()`` 기본값으로 채운다)을 읽지 않는다(``spec.is_compact_arg``). 문자열은 **모든 탭 꺼짐**에서
  시작한다(``blank_config``) — dict 형식은 수정자 I 이 켜진 패널 기본값에서 시작하므로 앱은 문자열을 쓴다.
- **앱 상태 문자열**(위젯 ``_cc_state``): 기본값과 다른 수정자·잎·조합 값만, ``arg_names()`` 경로(``I.exposure``,
  ``M1.mask_axis``…)를 그 순서로, ``"v": 1`` 을 앞에 둔 JSON. 전부 기본값이면 ``""``. 확장 패널의 ``dump_state`` 에서
  ``rev`` 를 뺀 모양이다(그 문자열도 읽는다). 숫자는 **JavaScript ``JSON.stringify`` 와 같은 글자**로 쓴다(``1.0`` → ``1``,
  ``1e-7`` → ``1e-7``) — Vue(utils/colorcraft.ts)와 파이썬이 같은 바이트를 만든다(tests/fixtures/colorcraft_state_golden.json 을
  양쪽이 왕복). ``-0`` 은 ``0`` 으로 본다(JSON 이 TS 에서 지키지 못한다).
- 값 정규화(``coerce``)는 확장 ``spec.coerce`` 규칙이다: 숫자는 유한한 float(아니면 기본값), 참/거짓 낱말, 선택지는
  대소문자 무시(종류는 노드 id ``chroma_plus`` 도). **범위로 자르지 않는다**(확장도 자르지 않는다). 하나 다른 점: 너무 큰
  정수(``float()`` 가 ``OverflowError``)는 확장에선 예외가 나가지만 여기선 기본값이다(앱이 붙여 넣기에서 죽지 않게 —
  tests/test_colorcraft_spec_parity.py 가 그 차이를 적어 둔다).

- **체인**(``build_chain``): 패스마다 도는 항목(``{"kind", "params", "schedule", "mask"}`` — 업스트림 노드가 만드는 모양).
  꺼진 탭·다른 패스의 탭·아무것도 바꾸지 못하는 항목은 빠진다. ``plan`` 이 보낼지 정하고(돌 항목이 없으면 보내지 않는다),
  ComfyUI 노드(앱 팩 1.7.0 ``ForgeNeoColorcraft``)는 phase C 에서 이 항목을 ``settings_json`` 으로 받는다. 보낸 문자열과 확장
  패널의 67칸은 카드가 만들 수 있는 값(유효숫자 10자리, -0 없음)에서 같은 체인이 된다(compact == 패널, 시험이 말뭉치로 본다).
- **메인 요청 전용**: 확장은 SAM3·ADetailer 내부 패스에서 붙지 않는다 — 블록은 메인 요청에만 싣는다(T2I 계열, I2I·인페인트는
  카드 토글 ``img2img``). 보조 패스에 주지 않는 것은 core/alwayson_propagation 의 Colorcraft 행이 한다.
- **붙여 넣기**(``from_infotext``·``from_value``): 자기 열쇠 ``SAM Extra Colorcraft``, 없으면 업스트림 Forge 스크립트의
  ``Colorcraft: MOD(...)MASK(...)COMBO(...)``·aoleg 포크의 ``Colorcraft: v1;name=value`` — 확장 ``spec.decode`` 와 같은 결과.
  디버그 값은 붙여 넣지 않는다(앱 ``Config`` 에 없다).
- **화면 표**: ``schema_ts()`` 가 frontend/src/utils/colorcraftSchema.generated.ts 를 만든다(``write_schema_ts``). 한국어 라벨·
  선택지 힌트·묶음 이름은 앱 문구이고, 이름·범위·기본값·선택지 값은 확장 계약이다.

확장 코드는 GPL-3.0 이라 옮기지 않았다 — 표와 형식 규칙(업스트림 muerrilla/ComfyUI-Colorcraft·aoleg 포크의 MIT 형식과
sam-extra 의 규칙)을 다시 구현했고, 설치된 확장은 시험 대조 기준(oracle)으로만 쓴다(tests/test_colorcraft_spec_parity.py). 제목 문자열을 다른 곳에 복사하지 말고
``SCRIPT_NAME`` 을 import 한다(확장 스크립트의 ``title()`` 과 같아야 한다 — 다르면 Forge 가 422).
"""
from __future__ import annotations

import json
import logging
import math
import re
from dataclasses import dataclass, field, replace
from types import MappingProxyType
from typing import Any, Mapping, Optional

from core.forge_override_settings import (
    INFOTEXT_COLORCRAFT_PRE_DD as PRE_DD_INFOTEXT_KEY,
    OPT_COLORCRAFT_PRE_DD as OPT_PRE_DD,
)

logger = logging.getLogger(__name__)

# ── 확장 계약 ─────────────────────────────────────────────────────────────────
SCRIPT_NAME = "Colorcraft (sam-extra)"                 # scripts/colorcraft.py TITLE
INFOTEXT_KEY = "SAM Extra Colorcraft"                  # spec.INFOTEXT_KEY — to_infotext 문자열
STATUS_KEY = "SAM Extra Colorcraft status"             # spec.STATUS_KEY — 패스별 'base: krea2 (Wan21); mods I; …'
FOREIGN_KEY = "Colorcraft"                             # spec.FOREIGN_KEY — 업스트림·aoleg 포크가 쓰는 열쇠(붙여 넣기만)
VERSION = "v1"                                         # spec.VERSION
XYZ_PREFIX = "[Colorcraft]"                            # scripts/colorcraft.py XYZ_PREFIX — XYZ 축 머리(P18 — 앱은 축을 쓰지 않는다)
INFOTEXT_KEYS = (INFOTEXT_KEY, STATUS_KEY, PRE_DD_INFOTEXT_KEY)

MODIFIER_COUNT, MASK_COUNT, COMBO_COUNT = 10, 10, 5
MODIFIER_TAGS = ("I", "II", "III", "IV", "V", "VI", "VII", "VIII", "IX", "X")
MASK_TAGS = tuple(f"M{i + 1}" for i in range(MASK_COUNT))
COMBO_TAGS = tuple(f"C{i + 1}" for i in range(COMBO_COUNT))
NODE_TAGS = MODIFIER_TAGS + MASK_TAGS + COMBO_TAGS

# 종류 = 확장이 흉내 내는 ComfyUI 노드(engine.KINDS 순서). 선택지 **값**은 영어 라벨이다(그대로 보낸다).
KINDS = ("advanced", "basic", "luma", "chroma", "chroma_plus", "punch", "shift")
KIND_LABELS = MappingProxyType({
    "advanced": "Advanced", "basic": "Basic", "luma": "Luma", "chroma": "Chroma",
    "chroma_plus": "Chroma Plus", "punch": "Punch", "shift": "Shift",
})
KIND_CHOICES = tuple(KIND_LABELS[kind] for kind in KINDS)
KIND_IDS = MappingProxyType({label: kind for kind, label in KIND_LABELS.items()})
# 기저 벡터가 필요한 종류(engine.ADVANCED_KINDS) — Shift 는 마스크를 쓸 때만(needs_basis).
ADVANCED_KINDS = frozenset({"advanced", "luma", "chroma", "chroma_plus", "punch"})

PASS_BASE, PASS_HIRES, PASS_BOTH = "Base", "Hires", "Both"
PASS_CHOICES = (PASS_BASE, PASS_HIRES, PASS_BOTH)
MASK_NONE = "none"
MASK_CHOICES = (MASK_NONE,) + MASK_TAGS + COMBO_TAGS
MASK_AXIS_OPTIONS = ("exposure", "hue", "saturation", "temperature", "tint", "temp+tint", "temp-tint",
                     "lab-a", "lab-b", "lab-a+b", "lab-a-b", "clarity", "sharpness")   # masking.MASK_AXIS_OPTIONS
MASK_MODE_OPTIONS = ("highs", "lows", "split", "range", "protect range")
COMBO_OP_OPTIONS = ("and", "or", "subtract", "xor")
COLOR_SHIFT_MODE_OPTIONS = ("default", "legacy")
CHROMA_PLANE_OPTIONS = ("temp_tint", "lab")

FLOAT, BOOL, CHOICE = "float", "bool", "choice"


@dataclass(frozen=True)
class Field:
    """칸 하나. ``label`` = 앱(한국어), ``title`` = 확장 패널의 영어 이름(툴팁). 범위·간격은 확장 슬라이더 값이다 —
    입력을 자르는 데 쓰지 않는다(확장도 자르지 않는다)."""

    name: str
    kind: str
    default: Any
    label: str
    title: str
    minimum: Optional[float] = None
    maximum: Optional[float] = None
    step: Optional[float] = None
    choices: tuple = ()
    group: str = ""


def _num(name, default, label, title, minimum, maximum, group="", step=0.01):
    return Field(name, FLOAT, default, label, title, minimum, maximum, step, (), group)


def _flag(name, default, label, title, group=""):
    return Field(name, BOOL, default, label, title, group=group)


def _pick(name, default, label, title, choices, group=""):
    return Field(name, CHOICE, default, label, title, choices=tuple(choices), group=group)


# 수정자 탭 하나의 칸(확장 MODIFIER_FIELDS 순서 = 인자 순서 = 패널 슬롯 6–49).
MODIFIER_FIELDS: tuple = (
    _flag("active", False, "켜기", "Active", "head"),
    _pick("kind", "Advanced", "종류", "Type", KIND_CHOICES, "head"),
    _pick("pass", PASS_BASE, "패스", "Pass", PASS_CHOICES, "head"),
    _pick("mask", MASK_NONE, "마스크", "Mask ➜", MASK_CHOICES, "head"),
    _num("strength", 1.0, "강도", "Strength", -1.0, 1.0, "schedule"),
    _num("start", 0.5, "시작", "Start", 0.0, 1.0, "schedule"),
    _num("end", 0.75, "끝", "End", 0.0, 1.0, "schedule"),
    _flag("advanced", False, "고급 스케줄", "Advanced Schedule", "shaping"),
    _num("exponent", 0.0, "지수", "Exponent", 0.0, 3.0, "shaping"),
    _num("bias", 0.5, "치우침", "Bias", 0.0, 1.0, "shaping"),
    _num("start_off", 0.0, "시작 오프셋", "Start Offset", -1.0, 1.0, "shaping"),
    _num("end_off", 0.0, "끝 오프셋", "End Offset", -1.0, 1.0, "shaping"),
    _flag("smooth", True, "부드럽게", "Smooth", "shaping"),
    _num("exposure", 0.0, "노출", "Exposure", -4.0, 4.0, "luma"),
    _num("tone_compression", 0.0, "톤 압축", "Tone Compression", -1.0, 1.0, "luma"),
    _num("contrast", 0.0, "대비", "Contrast", -4.0, 4.0, "contrast"),
    _num("clarity", 0.0, "명료도", "Clarity", -4.0, 4.0, "detail"),
    _num("sharpness", 0.0, "선명도", "Sharpness", -4.0, 4.0, "detail"),
    _num("temperature", 0.0, "색온도", "Temperature", -4.0, 4.0, "chroma"),
    _num("tint", 0.0, "틴트", "Tint", -4.0, 4.0, "chroma"),
    _num("vibrance", 0.0, "생동감", "Vibrance", -1.0, 1.0, "chroma"),
    _num("saturation", 0.0, "채도", "Saturation", -1.0, 1.0, "chroma"),
    _num("chroma_contrast", 0.0, "색 대비", "Chroma Contrast", -4.0, 4.0, "chroma"),
    _num("chroma_center", 0.0, "색 대비 중심", "Chroma Center", -1.0, 1.0, "chroma"),
    _flag("more_colors", True, "Chroma Plus 적용", "Apply Chroma Plus", "chroma_plus"),
    _num("temp_plus_tint", 0.0, "Temp+Tint", "Temp+Tint", -4.0, 4.0, "chroma_plus"),
    _num("temp_minus_tint", 0.0, "Temp−Tint", "Temp-Tint", -4.0, 4.0, "chroma_plus"),
    _num("lab_a", 0.0, "Lab A", "Lab A", -4.0, 4.0, "chroma_plus"),
    _num("lab_b", 0.0, "Lab B", "Lab B", -4.0, 4.0, "chroma_plus"),
    _num("lab_a_plus_b", 0.0, "Lab A+B", "Lab A+B", -4.0, 4.0, "chroma_plus"),
    _num("lab_a_minus_b", 0.0, "Lab A−B", "Lab A-B", -4.0, 4.0, "chroma_plus"),
    _flag("color_shift", True, "색 이동 적용", "Apply Color Shift", "color_shift"),
    _num("color_shift_amount", 0.0, "양", "Amount", -4.0, 4.0, "color_shift"),
    _pick("color_shift_mode", "default", "방식", "Mode", COLOR_SHIFT_MODE_OPTIONS, "color_shift"),
    _num("color_shift_red", 0.0, "빨강", "Red", -2.0, 2.0, "color_shift"),
    _num("color_shift_green", 0.0, "초록", "Green", -2.0, 2.0, "color_shift"),
    _num("color_shift_blue", 0.0, "파랑", "Blue", -2.0, 2.0, "color_shift"),
    _num("color_shift_brightness", 0.0, "밝기", "Brightness", -2.0, 2.0, "color_shift"),
    _flag("recenter_override", False, "재중심 직접 지정", "Override Recenter", "dev"),
    _num("recenter", 0.5, "재중심", "Recenter", 0.0, 1.0, "dev"),
    _flag("max_chroma_override", False, "최대 채도 직접 지정", "Override Max Chroma", "dev"),
    _num("max_chroma", 2.5, "최대 채도", "Max Chroma", 0.0, 10.0, "dev"),
    _flag("chroma_plane_override", False, "색 평면 직접 지정", "Override Chroma Plane", "dev"),
    _pick("chroma_plane", "temp_tint", "색 평면", "Chroma Plane", CHROMA_PLANE_OPTIONS, "dev"),
)

# 마스크 잎 하나(확장 LEAF_FIELDS 순서 = 패널 슬롯 50–59).
LEAF_FIELDS: tuple = (
    _pick("mask_axis", MASK_AXIS_OPTIONS[0], "기준 축", "Mask Axis", MASK_AXIS_OPTIONS),
    _pick("mask_mode", "highs", "방식", "Mask Mode", MASK_MODE_OPTIONS),
    _num("mask_strength", 1.0, "강도", "Strength", 0.0, 1.0),
    _num("mask_width", 0.0, "폭", "Width", 0.0, 2.0),
    _num("mask_center", 0.0, "중심", "Center", -1.0, 1.0),
    _num("mask_hardness", 1.0, "경계 단단함", "Hardness", 0.0, 10.0, step=0.1),
    _num("blur", 0.0, "흐림(이미지 px)", "Blur", 0.0, 64.0, step=0.1),
    _num("spread", 0.0, "넓힘", "Spread", -3.0, 3.0),
    _num("contrast", 0.0, "대비", "Contrast", -10.0, 10.0),
    _flag("normalize", False, "정규화", "Normalize"),
)


def combo_ref_choices(index: int) -> tuple:
    """조합 ``index``(0부터)의 Mask A/B 선택지 — 없음·잎 전부·**앞선** 조합만(확장 ``combo_ref_choices``)."""
    return (MASK_NONE,) + MASK_TAGS + COMBO_TAGS[:index]


# 조합 하나(확장 COMBO_FIELDS 순서 = 패널 슬롯 60–66). Mask A/B 의 표 선택지는 전부(공유 편집기) — 조합마다 실제
# 선택지는 ``combo_ref_choices(index)`` 다.
COMBO_FIELDS: tuple = (
    _pick("mask_a", MASK_NONE, "Mask A", "Mask A", combo_ref_choices(COMBO_COUNT)),
    _pick("mask_b", MASK_NONE, "Mask B", "Mask B", combo_ref_choices(COMBO_COUNT)),
    _pick("operation", "and", "연산", "Operation", COMBO_OP_OPTIONS),
    _num("blur", 0.0, "흐림(이미지 px)", "Blur", 0.0, 64.0, step=0.1),
    _num("spread", 0.0, "넓힘", "Spread", -3.0, 3.0),
    _num("contrast", 0.0, "대비", "Contrast", -10.0, 10.0),
    _flag("normalize", False, "정규화", "Normalize"),
)

# 패널 머리 두 칸(확장 GLOBAL_FIELDS). Debug 두 칸(확장 DEBUG_FIELDS)은 앱이 보내지도 붙여 넣지도 않는다.
GLOBAL_FIELDS: tuple = (
    _flag("enabled", False, "Colorcraft 사용", "Enable Colorcraft"),
    _flag("masking", False, "마스크 사용", "Enable masking"),
)

MODIFIER_BY_NAME = MappingProxyType({f.name: f for f in MODIFIER_FIELDS})
LEAF_BY_NAME = MappingProxyType({f.name: f for f in LEAF_FIELDS})
COMBO_BY_NAME = MappingProxyType({f.name: f for f in COMBO_FIELDS})
_COMBO_REF_NAMES = ("mask_a", "mask_b")

# 종류마다 체인 항목이 싣는 값 — (노드 이름, 패널 이름). 색 이동은 패널에서 color_shift_* 이고 노드에서
# mode/red/green/blue/brightness 다(확장 _KIND_PARAMS, 업스트림 노드 make()).
_SHIFT_PARAMS = (("color_shift_amount", "color_shift_amount"), ("mode", "color_shift_mode"),
                 ("red", "color_shift_red"), ("green", "color_shift_green"), ("blue", "color_shift_blue"),
                 ("brightness", "color_shift_brightness"))
CHROMA_PLUS_NAMES = ("temp_plus_tint", "temp_minus_tint", "lab_a", "lab_b", "lab_a_plus_b", "lab_a_minus_b")
_CHROMA_NAMES = ("temperature", "tint", "vibrance", "saturation", "chroma_contrast", "chroma_center")
_DEV_NAMES = ("recenter_override", "recenter", "max_chroma_override", "max_chroma", "chroma_plane_override",
              "chroma_plane")


def _same_names(names) -> tuple:
    return tuple((name, name) for name in names)


KIND_PARAMS = MappingProxyType({
    "basic": (("contrast", "contrast"),) + _SHIFT_PARAMS,
    "advanced": _same_names(("exposure", "tone_compression", "contrast", "clarity", "sharpness") + _CHROMA_NAMES
                            + ("more_colors",) + CHROMA_PLUS_NAMES + ("color_shift",))
                + _SHIFT_PARAMS + _same_names(_DEV_NAMES),
    "luma": _same_names(("exposure", "tone_compression")),
    "chroma": _same_names(_CHROMA_NAMES),
    "chroma_plus": _same_names(CHROMA_PLUS_NAMES),
    "punch": _same_names(("contrast", "clarity", "sharpness")),
    "shift": _SHIFT_PARAMS,
})

# 종류마다 편집기가 보이는 묶음(확장 KIND_GROUPS), Advanced 의 두 스위치(KIND_GATES), Basic 은 마스크 입력이 없다.
KIND_GROUPS = MappingProxyType({
    "advanced": ("luma", "contrast", "detail", "chroma", "chroma_plus", "color_shift", "dev"),
    "basic": ("contrast", "color_shift"),
    "luma": ("luma",),
    "chroma": ("chroma",),
    "chroma_plus": ("chroma_plus",),
    "punch": ("contrast", "detail"),
    "shift": ("color_shift",),
})
KIND_GATES = MappingProxyType({"advanced": ("more_colors", "color_shift")})
KIND_TAKES_MASK = MappingProxyType({kind: kind != "basic" for kind in KINDS})
GROUP_ORDER = ("luma", "contrast", "detail", "chroma", "chroma_plus", "color_shift", "dev")   # panel_state.GROUP_ORDER

SCHEDULE_NAMES = ("strength", "start", "end", "bias", "exponent", "start_off", "end_off", "smooth")
SHAPING_NAMES = ("exponent", "bias", "start_off", "end_off", "smooth")

# ── 앱 화면 문구(확장 계약 아님) ─────────────────────────────────────────────────
GROUP_LABELS = MappingProxyType({
    "head": "기본", "schedule": "스케줄", "shaping": "고급 스케줄", "luma": "밝기", "contrast": "대비",
    "detail": "디테일", "chroma": "색", "chroma_plus": "Chroma Plus", "color_shift": "색 이동",
    "dev": "개발자(색 축 보정)",
})
# 선택지 표시(값은 그대로 보낸다). 종류 이름은 ComfyUI 노드 이름이라 영어로 두고 한국어 힌트를 붙인다.
CHOICE_LABELS = MappingProxyType({
    "kind": MappingProxyType({
        "Advanced": "Advanced (전부)", "Basic": "Basic (대비·색 이동)", "Luma": "Luma (밝기)",
        "Chroma": "Chroma (색)", "Chroma Plus": "Chroma Plus (색 축 6개)", "Punch": "Punch (대비·디테일)",
        "Shift": "Shift (색 이동)"}),
    "pass": MappingProxyType({PASS_BASE: "Base (1차·I2I)", PASS_HIRES: "Hires (Hires 패스)",
                              PASS_BOTH: "Both (둘 다)"}),
    "mask": MappingProxyType({MASK_NONE: "없음"}),
    "mask_mode": MappingProxyType({"highs": "highs 높은 쪽", "lows": "lows 낮은 쪽", "split": "split 중심으로 나눔",
                                   "range": "range 범위", "protect range": "protect range 범위 보호"}),
    "operation": MappingProxyType({"and": "and 교집합", "or": "or 합집합", "subtract": "subtract 빼기",
                                   "xor": "xor 한쪽만"}),
    "color_shift_mode": MappingProxyType({"default": "기본", "legacy": "예전"}),
    "chroma_plane": MappingProxyType({"temp_tint": "색온도·틴트", "lab": "Lab"}),
})


# ── 값 정규화 (확장 spec.coerce 규칙) ───────────────────────────────────────────
def _finite(value: Any, default: Any) -> Any:
    """유한한 float, 아니면 ``default``. 확장과 달리 너무 큰 정수(OverflowError)도 ``default``(모듈 설명)."""
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        return default
    return number if math.isfinite(number) else default


_TRUE_WORDS = frozenset({"true", "1", "yes", "on"})
_FALSE_WORDS = frozenset({"false", "0", "no", "off", ""})


def _truthy(value: Any, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return default
    if isinstance(value, (int, float)):
        return bool(value)
    word = str(value).strip().lower()
    if word in _TRUE_WORDS:
        return True
    if word in _FALSE_WORDS:
        return False
    return default


def _choice(f: Field, value: Any, choices=None) -> str:
    options = tuple(choices) if choices is not None else f.choices
    if value in options:
        return value
    text = "" if value is None else str(value).strip()
    folded = {str(option).lower(): option for option in options}
    if text.lower() in folded:
        return folded[text.lower()]
    if f.name == "kind":
        node_id = text.lower().replace(" ", "_")          # 노드 id(chroma_plus)도 받는다
        if node_id in KIND_LABELS:
            return KIND_LABELS[node_id]
    return f.default


def coerce(f: Field, value: Any, choices=None) -> Any:
    """``value`` 를 칸 ``f`` 의 형으로. 쓸 수 없으면 칸 기본값(확장 ``spec.coerce``)."""
    if f.kind == FLOAT:
        return _finite(value, f.default)
    if f.kind == BOOL:
        return _truthy(value, f.default)
    return _choice(f, value, choices)


def _coerce_combo(index: int, f: Field, value: Any) -> Any:
    return coerce(f, value, combo_ref_choices(index) if f.name in _COMBO_REF_NAMES else None)


def modifier_default(index: int, name: str) -> Any:
    """탭 ``index`` 의 기본값 — 수정자 I 만 켜진 채 시작한다(확장 ``modifier_default``)."""
    if name == "active":
        return index == 0
    return MODIFIER_BY_NAME[name].default


# ── 설정 묶음 ─────────────────────────────────────────────────────────────────
@dataclass
class Config:
    """확장 ``spec.Config`` 와 같은 모양(디버그 두 칸 제외). 칸 값은 ``coerce`` 를 거친 값이다."""

    enabled: bool = False
    masking: bool = False
    modifiers: list = field(default_factory=list)   # MODIFIER_COUNT 개 dict(MODIFIER_FIELDS 이름)
    leaves: list = field(default_factory=list)      # MASK_COUNT 개 dict(LEAF_FIELDS 이름)
    combos: list = field(default_factory=list)      # COMBO_COUNT 개 dict(COMBO_FIELDS 이름)

    def copy(self) -> "Config":
        return Config(self.enabled, self.masking, [dict(m) for m in self.modifiers],
                      [dict(leaf) for leaf in self.leaves], [dict(c) for c in self.combos])


def default_config() -> Config:
    """패널 기본값 — Colorcraft·마스크 꺼짐, 수정자 I 만 켜짐, 나머지는 칸 기본값."""
    return Config(
        enabled=False,
        masking=False,
        modifiers=[{f.name: modifier_default(i, f.name) for f in MODIFIER_FIELDS} for i in range(MODIFIER_COUNT)],
        leaves=[{f.name: f.default for f in LEAF_FIELDS} for _ in range(MASK_COUNT)],
        combos=[{f.name: f.default for f in COMBO_FIELDS} for _ in range(COMBO_COUNT)],
    )


def blank_config() -> Config:
    """모든 탭 꺼짐 — infotext 가 어느 탭이 켜졌는지 말할 때의 출발점(확장 ``blank_config``)."""
    config = default_config()
    for modifier in config.modifiers:
        modifier["active"] = False
    return config


def node_fields(tag: str) -> tuple:
    if tag in MODIFIER_TAGS:
        return MODIFIER_FIELDS
    if tag in MASK_TAGS:
        return LEAF_FIELDS
    if tag in COMBO_TAGS:
        return COMBO_FIELDS
    raise KeyError(tag)


def node_values(config: Config, tag: str) -> dict:
    if tag in MODIFIER_TAGS:
        return config.modifiers[MODIFIER_TAGS.index(tag)]
    if tag in MASK_TAGS:
        return config.leaves[MASK_TAGS.index(tag)]
    if tag in COMBO_TAGS:
        return config.combos[COMBO_TAGS.index(tag)]
    raise KeyError(tag)


def node_default(tag: str, f: Field) -> Any:
    if tag in MODIFIER_TAGS:
        return modifier_default(MODIFIER_TAGS.index(tag), f.name)
    return f.default


def node_choices(tag: str, f: Field) -> tuple:
    """칸 ``f`` 의 실제 선택지 — 조합의 Mask A/B 는 조합마다 다르다."""
    if tag in COMBO_TAGS and f.name in _COMBO_REF_NAMES:
        return combo_ref_choices(COMBO_TAGS.index(tag))
    return f.choices


# 상태 경로 = 확장 ``arg_names()`` 에서 머리(enabled·masking)와 디버그를 뺀 575개, 그 순서.
STATE_PATHS = tuple(f"{tag}.{f.name}" for tag in NODE_TAGS for f in node_fields(tag))
_STATE_PATH_SET = frozenset(STATE_PATHS)


def arg_names() -> tuple:
    """확장 ``spec.arg_names()`` 에서 디버그 두 칸을 뺀 것 — ``enabled``, ``masking``, ``I.active`` … ``C5.normalize``."""
    return tuple(f.name for f in GLOBAL_FIELDS) + STATE_PATHS


def config_values(config: Config) -> list:
    """``arg_names()`` 순서의 값(확장 ``config_to_args`` 에서 디버그 두 칸을 뺀 것) — 비교·시험용."""
    values = [bool(config.enabled), bool(config.masking)]
    for tag in NODE_TAGS:
        current = node_values(config, tag)
        values += [current[f.name] for f in node_fields(tag)]
    return values


def config_from_mapping(mapping: Mapping) -> Config:
    """``{"enabled": True, "I.exposure": 0.3, …}``(``arg_names()`` 경로) → ``Config``(확장 ``config_from_mapping``).

    패널 기본값(수정자 I 켜짐)에서 시작하고, 없는 경로는 기본값, 모르는 경로는 버린다. ``None`` 값은 건너뛴다.
    ``enabled`` 가 없으면 켬(dict 를 보낸 것 = Colorcraft 를 원함)."""
    config = default_config()
    config.enabled = _truthy(mapping.get("enabled", True), True)
    config.masking = _truthy(mapping.get("masking", False), False)
    for key, value in mapping.items():
        if value is None:
            continue
        tag, _, name = str(key).partition(".")
        if tag in MODIFIER_TAGS and name in MODIFIER_BY_NAME:
            config.modifiers[MODIFIER_TAGS.index(tag)][name] = coerce(MODIFIER_BY_NAME[name], value)
        elif tag in MASK_TAGS and name in LEAF_BY_NAME:
            config.leaves[MASK_TAGS.index(tag)][name] = coerce(LEAF_BY_NAME[name], value)
        elif tag in COMBO_TAGS and name in COMBO_BY_NAME:
            index = COMBO_TAGS.index(tag)
            config.combos[index][name] = _coerce_combo(index, COMBO_BY_NAME[name], value)
    return config


# ── 앱 상태 문자열 (위젯 ``_cc_state``) ─────────────────────────────────────────
STATE_VERSION = 1
DEFAULT_STATE = ""


class StateError(ValueError):
    """상태 문자열을 읽을 수 없다(JSON 아님·객체 아님·버전 다름)."""


def js_number(value: float) -> str:
    """JavaScript ``Number.prototype.toString``(= ``JSON.stringify``)과 같은 글자. ``-0`` 은 ``0``.

    자릿수는 파이썬 ``repr`` 의 최단 왕복 자릿수(JS 와 같은 규칙)이고, 소수점·지수 표기만 JS 규칙(ECMA-262
    Number::toString: 지수 n 이 -6 < n ≤ 21 이면 소수, 아니면 ``1e-7``·``1.5e+21``)으로 바꾼다."""
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"not a finite number: {value!r}")
    if number == 0:
        return "0"
    if number < 0:
        return "-" + js_number(-number)
    mantissa, _, exponent = repr(number).partition("e")
    whole, _, fraction = mantissa.partition(".")
    digits = (whole + fraction).lstrip("0")
    stripped = digits.rstrip("0")
    k = len(stripped)
    n = len(digits) + int(exponent or 0) - len(fraction)      # number = digits × 10^(n - len(digits))
    digits = stripped
    if k <= n <= 21:
        return digits + "0" * (n - k)
    if 0 < n <= 21:
        return digits[:n] + "." + digits[n:]
    if -6 < n <= 0:
        return "0." + "0" * (-n) + digits
    e = n - 1
    sign = "+" if e >= 0 else "-"
    head = digits if k == 1 else digits[0] + "." + digits[1:]
    return f"{head}e{sign}{abs(e)}"


def _js_json(value: Any) -> str:
    """JSON 한 값 — 숫자는 ``js_number``, 문자열은 ``JSON.stringify`` 와 같은 이스케이프(비 ASCII 그대로)."""
    if isinstance(value, bool):
        return "true" if value else "false"
    if value is None:
        return "null"
    if isinstance(value, (int, float)):
        return js_number(value)
    if isinstance(value, str):
        return json.dumps(value, ensure_ascii=False)
    if isinstance(value, Mapping):
        return "{" + ",".join(f"{_js_json(str(k))}:{_js_json(v)}" for k, v in value.items()) + "}"
    if isinstance(value, (list, tuple)):
        return "[" + ",".join(_js_json(v) for v in value) + "]"
    raise TypeError(f"cannot write {type(value).__name__} as JSON")


def _same_value(f: Field, value: Any, default: Any) -> bool:
    if f.kind == FLOAT:
        return float(value) == float(default)             # -0.0 == 0.0
    return value == default


def state_paths(config: Config) -> dict:
    """기본값과 다른 상태 경로 → 값(``STATE_PATHS`` 순서). 값은 ``coerce`` 를 거친다."""
    out = {}
    for tag in NODE_TAGS:
        current = node_values(config, tag)
        for f in node_fields(tag):
            value = coerce(f, current.get(f.name), node_choices(tag, f))
            if not _same_value(f, value, node_default(tag, f)):
                out[f"{tag}.{f.name}"] = 0.0 if (f.kind == FLOAT and value == 0) else value
    return out


def dump_state(config: Config) -> str:
    """``config`` 의 상태 문자열 — 전부 기본값이면 ``""``, 아니면 ``{"v":1,"I.exposure":0.3,…}``.

    Colorcraft 켬·마스크 켬은 위젯이 따로 있어 넣지 않는다. 바이트는 utils/colorcraft.ts ``dumpState`` 와 같다."""
    paths = state_paths(config)
    if not paths:
        return DEFAULT_STATE
    body = [f'"v":{STATE_VERSION}'] + [f"{_js_json(path)}:{_js_json(value)}" for path, value in paths.items()]
    return "{" + ",".join(body) + "}"


def parse_state(raw: Any) -> dict:
    """상태 문자열(또는 dict) → ``{경로: 원래 값}``(모르는 경로·``v``·``rev`` 제외). ``""``/None = 기본값 = ``{}``.

    확장 패널의 상태(``rev`` 포함)도 읽는다. 읽을 수 없으면 ``StateError``."""
    if raw is None or (isinstance(raw, str) and raw.strip() == ""):
        return {}
    if isinstance(raw, Mapping):
        data = dict(raw)
    elif isinstance(raw, str) and raw.lstrip().startswith("{"):
        try:
            data = json.loads(raw)
        except ValueError as exc:
            raise StateError(f"Colorcraft 상태가 JSON 이 아닙니다 ({exc})") from None
        if not isinstance(data, dict):
            raise StateError("Colorcraft 상태가 JSON 객체가 아닙니다")
    else:
        raise StateError(f"Colorcraft 상태가 아닙니다 ({type(raw).__name__})")
    version = data.get("v", STATE_VERSION)
    if isinstance(version, bool) or version != STATE_VERSION:
        raise StateError(f"Colorcraft 상태 버전 {version!r} 은(는) {STATE_VERSION} 이 아닙니다")
    return {key: value for key, value in data.items() if key in _STATE_PATH_SET}


def config_from_state(enabled: Any, masking: Any, state: Any) -> Config:
    """위젯 세 칸 → ``Config``(확장 ``panel_state.config_from_state`` 와 같은 뜻 — 경로는 패널 기본값 위에 얹는다)."""
    config = config_from_mapping(parse_state(state))
    config.enabled = _truthy(enabled, False)
    config.masking = _truthy(masking, False)
    return config


def canonical_state(raw: Any) -> str:
    """상태를 읽어 다시 쓴 정규 문자열(경로 순서·숫자 글자·기본값 생략). 읽을 수 없으면 ``StateError``."""
    return dump_state(config_from_mapping(parse_state(raw)))


# ── 마스크가 닿는 곳 ───────────────────────────────────────────────────────────
def kind_of(modifier: Mapping) -> str:
    """수정자의 종류 id(``advanced``…). 모르는 라벨은 ``advanced``(확장 ``kind_of``)."""
    return KIND_IDS.get(modifier.get("kind"), "advanced")


def _masked_selections(config: Config) -> list:
    """켜진 수정자 가운데 마스크를 받는 종류가 고른 마스크(없음 제외)."""
    return [m["mask"] for m in config.modifiers
            if m["active"] and KIND_TAKES_MASK[kind_of(m)] and m["mask"] != MASK_NONE]


def active_masks(config: Config) -> tuple:
    """``(잎 태그, 조합 태그)`` — 켜진 수정자의 선택이 (조합을 거쳐) 닿는 곳. 미완성 조합도 따라간다(확장
    ``active_masks`` — infotext 가 어떤 마스크를 적을지). 마스크가 꺼져 있으면 둘 다 비어 있다."""
    if not config.masking:
        return [], []
    reached, todo = set(), _masked_selections(config)
    while todo:
        tag = todo.pop()
        if tag == MASK_NONE or tag in reached:
            continue
        reached.add(tag)
        if tag in COMBO_TAGS:
            combo = config.combos[COMBO_TAGS.index(tag)]
            todo.extend((combo["mask_a"], combo["mask_b"]))
    return [t for t in MASK_TAGS if t in reached], [t for t in COMBO_TAGS if t in reached]


# ── infotext 문자열 (확장 자기 열쇠 ``SAM Extra Colorcraft``) ────────────────────
def _format(f: Field, value: Any) -> str:
    """bool → 1/0, 숫자 → 유효숫자 10자리(``.10g`` — 칸이 만드는 값은 그대로 돌아온다), 선택지 → 그대로."""
    if f.kind == BOOL:
        return "1" if value else "0"
    if f.kind == FLOAT:
        return format(float(value), ".10g")
    return str(value)


def written_modifier_fields(modifier: Mapping) -> list:
    """수정자의 종류가 읽는 칸 — infotext 에 쓰고 붙여 넣을 칸(확장 ``_relevant_modifier_fields``).

    머리(종류·패스·마스크 — 마스크는 받는 종류만), 스케줄 전부, 그 종류 노드의 값 전부(스위치로 꺼진 묶음 포함).
    다른 종류의 칸은 화면에도 없고 결과도 바꾸지 못해 쓰지 않는다."""
    kind = kind_of(modifier)
    names = ["kind", "pass"] + (["mask"] if KIND_TAKES_MASK[kind] else [])
    names += ["strength", "start", "end", "advanced", *SHAPING_NAMES]
    for _, panel_name in KIND_PARAMS[kind]:
        if panel_name not in names:
            names.append(panel_name)
    return names


def to_infotext(config: Config) -> str:
    """``v1;mods=I,III;masking=1;masks=M1;combos=;I.exposure=0.3;…`` — 기본값과 다른 값만(확장 ``to_infotext``).

    켜진 탭만, 각 탭은 그 종류가 읽는 칸만, 마스크는 켜진 탭이 닿는 잎·조합만. 이 문자열이 Forge 로 보내는 첫 인자이자
    이미지에 남는 값이다. 켬 여부는 쓰지 않는다(이 문자열을 받으면 확장은 켠 것으로 읽는다)."""
    active = [i for i, m in enumerate(config.modifiers) if m["active"]]
    parts = [VERSION, "mods=" + ",".join(MODIFIER_TAGS[i] for i in active)]
    leaves, combos = active_masks(config)
    if config.masking:
        parts += ["masking=1", "masks=" + ",".join(leaves), "combos=" + ",".join(combos)]
    for i in active:
        modifier = config.modifiers[i]
        for name in written_modifier_fields(modifier):
            f = MODIFIER_BY_NAME[name]
            if modifier[name] != f.default:
                parts.append(f"{MODIFIER_TAGS[i]}.{name}={_format(f, modifier[name])}")
    for tags, fields, values in ((leaves, LEAF_FIELDS, config.leaves), (combos, COMBO_FIELDS, config.combos)):
        all_tags = MASK_TAGS if fields is LEAF_FIELDS else COMBO_TAGS
        for tag in tags:
            node = values[all_tags.index(tag)]
            parts += [f"{tag}.{f.name}={_format(f, node[f.name])}" for f in fields if node[f.name] != f.default]
    return ";".join(parts)


def parse_own(text: Any) -> Config:
    """확장 자기 infotext 값 → ``Config``(켬). 모든 탭 꺼짐에서 시작해 ``mods`` 가 켠다. 모르는 열쇠·읽을 수 없는 값은
    기본값(확장 ``parse_own``). ``masks``/``combos`` 는 기록일 뿐 읽지 않는다(수정자의 선택이 정한다)."""
    config = blank_config()
    config.enabled = True
    for chunk in str(text).split(";"):
        chunk = chunk.strip()
        if "=" not in chunk:
            continue
        key, _, raw = chunk.partition("=")
        key, raw = key.strip(), raw.strip()
        if key == "mods":
            for tag in (t.strip() for t in raw.split(",")):
                if tag in MODIFIER_TAGS:
                    config.modifiers[MODIFIER_TAGS.index(tag)]["active"] = True
        elif key == "masking":
            config.masking = _truthy(raw, False)
        elif key in ("masks", "combos"):
            continue
        else:
            tag, _, name = key.partition(".")
            if tag in MODIFIER_TAGS and name in MODIFIER_BY_NAME and name != "active":
                config.modifiers[MODIFIER_TAGS.index(tag)][name] = coerce(MODIFIER_BY_NAME[name], raw)
            elif tag in MASK_TAGS and name in LEAF_BY_NAME:
                config.leaves[MASK_TAGS.index(tag)][name] = coerce(LEAF_BY_NAME[name], raw)
            elif tag in COMBO_TAGS and name in COMBO_BY_NAME:
                index = COMBO_TAGS.index(tag)
                config.combos[index][name] = _coerce_combo(index, COMBO_BY_NAME[name], raw)
    return config


_COMPACT_PREFIX = re.compile(r"v1\s*;\s*mods\s*=")


def is_compact_arg(value: Any) -> bool:
    """API 의 한 값 형식인가 — dict, 또는 ``v1;mods=`` 로 시작하는 문자열(확장 ``is_compact_arg``)."""
    if isinstance(value, dict):
        return True
    return isinstance(value, str) and _COMPACT_PREFIX.match(value.strip()) is not None


def parse_compact(value: Any) -> Optional[Config]:
    """첫 인자 한 값(문자열 → ``parse_own``, dict → ``config_from_mapping``) → ``Config``. 그런 값이 아니면 None.

    둘은 출발점이 다르다: 문자열은 모든 탭 꺼짐, dict 는 패널 기본값(수정자 I 켜짐) — 확장과 같다."""
    if isinstance(value, dict):
        return config_from_mapping(value)
    if is_compact_arg(value):
        return parse_own(value)
    return None


class BlockError(ValueError):
    """보낸 alwayson 블록의 첫 인자가 한 값 형식(문자열·dict)이 아니다 — 앱이 만든 블록이면 버그."""


def as_block(config: Config) -> dict:
    """alwayson 블록 — 첫 인자 하나 = ``to_infotext``(나머지 66칸은 Forge 가 ``ui()`` 기본값으로 채우고 확장이 무시)."""
    return {"args": [to_infotext(config)]}


def parse_script_block(block: Any) -> Optional[Config]:
    """보낸 alwayson 블록 → 확장이 읽는 ``Config``. 블록이 아니면 None, 첫 인자가 한 값 형식이 아니면 ``BlockError``
    (앱은 늘 문자열을 보낸다 — 컴파일러는 이를 조용히 버리지 않고 오류로 낸다)."""
    if not isinstance(block, Mapping):
        return None
    args = block.get("args")
    first = args[0] if isinstance(args, (list, tuple)) and args else None
    config = parse_compact(first)
    if config is None:
        raise BlockError(f"Colorcraft 블록의 첫 인자가 infotext 문자열·dict 가 아닙니다: {str(first)[:80]!r}")
    return config


# ── 붙여 넣기: 업스트림 Forge 스크립트의 ``Colorcraft: MOD(...)MASK(...)COMBO(...)`` ────────────────────
# muerrilla/ComfyUI-Colorcraft(MIT) Forge 스크립트가 쓰는 형식 — 탭마다 ``태그:값,값,…`` 를 ``;`` 로 잇는다. 값 순서는
# 그 스크립트의 칸 순서다(첫 칸 active 는 쓰지 않는다 — 적힌 탭이 켜진 탭이다). 'hires' 는 Hires 패스 전용 체크.
UPSTREAM_MODIFIER_VALUE_FIELDS = (
    "mask", "strength", "start", "end", "advanced", "exponent", "bias", "start_off", "end_off", "smooth",
    "exposure", "tone_compression", "contrast", "clarity", "sharpness", "temperature", "tint", "vibrance",
    "saturation", "chroma_contrast", "chroma_center", "temp_plus_tint", "temp_minus_tint", "lab_a", "lab_b",
    "lab_a_plus_b", "lab_a_minus_b", "color_shift_amount", "color_shift_mode", "color_shift_red",
    "color_shift_green", "color_shift_blue", "color_shift_brightness", "hires",
)
UPSTREAM_LEAF_FIELDS = ("mask_axis", "mask_mode", "mask_strength", "mask_width", "mask_center", "mask_hardness",
                        "blur", "spread", "contrast", "normalize")
UPSTREAM_COMBO_FIELDS = ("mask_a", "mask_b", "operation", "blur", "spread", "contrast", "normalize")
_UPSTREAM_BOOLS = frozenset({"advanced", "smooth", "normalize", "hires"})
_UPSTREAM_TEXTS = frozenset({"color_shift_mode", "mask_axis", "mask_mode", "mask", "mask_a", "mask_b", "operation"})


def _upstream_value(name: str, raw: str) -> Any:
    """업스트림 값 하나 — 참/거짓은 글자 그대로 ``True`` 만 참, 글자 칸은 그대로, 나머지는 float(못 읽으면 None)."""
    if name in _UPSTREAM_BOOLS:
        return raw == "True"
    if name in _UPSTREAM_TEXTS:
        return raw
    try:
        return float(raw)
    except (TypeError, ValueError):
        return None


def _upstream_entries(section: str, names: tuple) -> dict:
    """``태그:값,…;태그:값,…`` → ``{태그: {칸: 값}}``. 값 개수가 다르면 앞에서부터 맞는 만큼만."""
    out: dict = {}
    for entry in section.split(";") if section else ():
        if ":" not in entry:
            continue
        tag, values = entry.split(":", 1)
        out[tag.strip()] = {name: _upstream_value(name, raw) for name, raw in zip(names, values.split(","))}
    return out


def _upstream_section(text: str, label: str) -> str:
    """``LABEL(...)`` 의 안쪽 — 첫 ``)`` 까지(업스트림 형식은 괄호를 겹치지 않는다)."""
    found = re.search(re.escape(label) + r"\((.*?)\)", text)
    return found.group(1) if found else ""


def is_upstream_text(value: Any) -> bool:
    return isinstance(value, str) and "MOD(" in value


def upstream_dict(text: str) -> dict:
    """업스트림 문자열 → 그 확장의 붙여 넣기 콜백이 만드는 dict(``modifiers``·``masks``·``combos``)."""
    modifiers = _upstream_entries(_upstream_section(text, "MOD"), UPSTREAM_MODIFIER_VALUE_FIELDS)
    for values in modifiers.values():
        values["active_flag"] = True
    return {"modifiers": modifiers,
            "masks": _upstream_entries(_upstream_section(text, "MASK"), UPSTREAM_LEAF_FIELDS),
            "combos": _upstream_entries(_upstream_section(text, "COMBO"), UPSTREAM_COMBO_FIELDS)}


def parse_upstream(data: Any) -> Config:
    """업스트림 형식(문자열, 또는 그 콜백이 만든 dict) → ``Config``(켬).

    업스트림 탭은 늘 Advanced 노드이고 Chroma Plus·색 이동이 늘 적용되며 개발자 값이 없다. 'hires' 체크는 패스
    Hires/Base. 마스크나 조합이 적혀 있으면 마스크 켬(업스트림도 그렇게 되살린다)."""
    if isinstance(data, str):
        data = upstream_dict(data)
    config = blank_config()
    config.enabled = True
    modifiers = data.get("modifiers") or {}
    masks = data.get("masks") or {}
    combos = data.get("combos") or {}
    for tag, values in modifiers.items():
        if tag not in MODIFIER_TAGS:
            continue
        m = config.modifiers[MODIFIER_TAGS.index(tag)]
        m.update({"active": True, "kind": KIND_LABELS["advanced"], "more_colors": True, "color_shift": True})
        for name, value in values.items():
            if value is None or name in ("active_flag", "active"):
                continue
            if name == "hires":
                m["pass"] = PASS_HIRES if _truthy(value) else PASS_BASE
            elif name in MODIFIER_BY_NAME:
                m[name] = coerce(MODIFIER_BY_NAME[name], value)
    for tag, values in masks.items():
        if tag in MASK_TAGS:
            leaf = config.leaves[MASK_TAGS.index(tag)]
            for name, value in values.items():
                if value is not None and name in LEAF_BY_NAME:
                    leaf[name] = coerce(LEAF_BY_NAME[name], value)
    for tag, values in combos.items():
        if tag in COMBO_TAGS:
            index = COMBO_TAGS.index(tag)
            for name, value in values.items():
                if value is not None and name in COMBO_BY_NAME:
                    config.combos[index][name] = _coerce_combo(index, COMBO_BY_NAME[name], value)
    config.masking = bool(masks) or bool(combos)
    return config


# ── 붙여 넣기: aoleg 포크의 ``Colorcraft: v1;name=value;…`` ───────────────────────────────────────
# aoleg/ComfyUI-Colorcraft(MIT, Forge Neo 포크) — 수정자 하나·마스크 둘의 평평한 값. 이름: (형, 포크 기본값).
FORK_PARAMS = MappingProxyType({
    "strength": (FLOAT, 1.0), "start": (FLOAT, 0.5), "end": (FLOAT, 0.75),
    "advanced": (BOOL, False), "bias": (FLOAT, 0.5), "smooth": (BOOL, True), "exponent": (FLOAT, 0.0),
    "start_off": (FLOAT, 0.0), "end_off": (FLOAT, 0.0),
    "exposure": (FLOAT, 0.0), "tone_compression": (FLOAT, 0.0),
    "contrast": (FLOAT, 0.0), "clarity": (FLOAT, 0.0), "sharpness": (FLOAT, 0.0),
    "temperature": (FLOAT, 0.0), "tint": (FLOAT, 0.0), "vibrance": (FLOAT, 0.0), "saturation": (FLOAT, 0.0),
    "chroma_contrast": (FLOAT, 0.0), "chroma_center": (FLOAT, 0.5),
    "more_colors": (BOOL, False), "temp_plus_tint": (FLOAT, 0.0), "temp_minus_tint": (FLOAT, 0.0),
    "lab_a": (FLOAT, 0.0), "lab_b": (FLOAT, 0.0), "lab_a_plus_b": (FLOAT, 0.0), "lab_a_minus_b": (FLOAT, 0.0),
    "color_shift": (BOOL, False), "color_shift_amount": (FLOAT, 0.0),
    "mode": (CHOICE, "default"), "red": (FLOAT, 0.0), "green": (FLOAT, 0.0), "blue": (FLOAT, 0.0),
    "brightness": (FLOAT, 0.0),
    "masking": (BOOL, False),
    "mask_mode": (CHOICE, "highs"), "mask_axis": (CHOICE, "exposure"), "mask_center": (FLOAT, 0.0),
    "mask_hardness": (FLOAT, 1.0), "mask_strength": (FLOAT, 1.0), "mask_width": (FLOAT, 0.0),
    "mask_combine": (BOOL, False), "mask_operation": (CHOICE, "and"),
    "mask_b_mode": (CHOICE, "highs"), "mask_b_axis": (CHOICE, "exposure"), "mask_b_center": (FLOAT, 0.0),
    "mask_b_hardness": (FLOAT, 1.0), "mask_b_strength": (FLOAT, 1.0), "mask_b_width": (FLOAT, 0.0),
    "mask_blur_radius": (FLOAT, 0.0), "mask_spread": (FLOAT, 0.0),
    "mask_preview": (BOOL, False), "mask_preview_color": (CHOICE, "red"),
    "dev": (BOOL, False), "recenter_override": (BOOL, False), "recenter": (FLOAT, 0.5),
    "max_chroma_override": (BOOL, False), "max_chroma": (FLOAT, 2.5),
    "chroma_plane_override": (BOOL, False), "chroma_plane": (CHOICE, "temp_tint"),
    "apply_to_hr": (BOOL, True), "debug": (BOOL, False),
})
_FORK_AXES = MASK_AXIS_OPTIONS[:11]          # 포크의 마스크 축에는 clarity·sharpness 가 없다
FORK_CHOICES = MappingProxyType({
    "mode": COLOR_SHIFT_MODE_OPTIONS, "mask_mode": MASK_MODE_OPTIONS, "mask_b_mode": MASK_MODE_OPTIONS,
    "mask_axis": _FORK_AXES, "mask_b_axis": _FORK_AXES, "mask_operation": COMBO_OP_OPTIONS,
    "mask_preview_color": ("red", "green", "blue", "white", "black"), "chroma_plane": CHROMA_PLANE_OPTIONS,
})
FORK_PREVIEW_COLORS = MappingProxyType({      # 포크 마스크 미리보기 색(빨강·초록·파랑·밝기 전 RGB 오프셋)
    "red": (0.5, -0.5, -0.5), "green": (-0.5, 0.5, -0.5), "blue": (-0.5, -0.5, 0.5),
    "white": (0.5, 0.5, 0.5), "black": (-0.5, -0.5, -0.5),
})
_FORK_TRUE = ("1", "true", "True", "yes")
# 포크 수정자 칸 가운데 앱 칸 이름이 같은 것(그대로 옮긴다).
_FORK_SAME_NAMES = ("strength", "start", "end", "smooth", "exposure", "tone_compression", "contrast", "clarity",
                    "sharpness", "temperature", "tint", "vibrance", "saturation", "chroma_contrast", "more_colors",
                    "temp_plus_tint", "temp_minus_tint", "lab_a", "lab_b", "lab_a_plus_b", "lab_a_minus_b",
                    "color_shift", "color_shift_amount", "recenter_override", "recenter", "max_chroma_override",
                    "max_chroma", "chroma_plane_override", "chroma_plane")
_FORK_SHIFT_NAMES = (("color_shift_mode", "mode"), ("color_shift_red", "red"), ("color_shift_green", "green"),
                     ("color_shift_blue", "blue"), ("color_shift_brightness", "brightness"))
_FORK_LEAF_A = (("mask_mode", "mask_mode"), ("mask_axis", "mask_axis"), ("mask_center", "mask_center"),
                ("mask_hardness", "mask_hardness"), ("mask_strength", "mask_strength"), ("mask_width", "mask_width"))
_FORK_LEAF_B = (("mask_mode", "mask_b_mode"), ("mask_axis", "mask_b_axis"), ("mask_center", "mask_b_center"),
                ("mask_hardness", "mask_b_hardness"), ("mask_strength", "mask_b_strength"),
                ("mask_width", "mask_b_width"))


def is_fork_text(value: Any) -> bool:
    return isinstance(value, str) and not is_upstream_text(value) and value.strip().startswith("v1")


def fork_values(text: Any) -> dict:
    """포크 문자열 → 모든 칸의 값(없는 칸은 포크 기본값). 참은 ``1``/``true``/``True``/``yes`` 만, 숫자는 유한한 값만,
    선택지는 글자 그대로 맞아야 한다(공백도 그대로 — 포크가 그렇게 읽는다)."""
    values = {name: default for name, (_, default) in FORK_PARAMS.items()}
    for chunk in str(text or "").split(";"):
        chunk = chunk.strip()
        if "=" not in chunk:
            continue
        name, _, raw = chunk.partition("=")
        name = name.strip()
        if name not in FORK_PARAMS:
            continue
        kind, default = FORK_PARAMS[name]
        if kind == BOOL:
            values[name] = raw.strip() in _FORK_TRUE
        elif kind == FLOAT:
            values[name] = _finite(raw, default)
        else:
            values[name] = raw if raw in FORK_CHOICES.get(name, ()) else default
    return values


def parse_fork(text: Any) -> Config:
    """포크 문자열 → ``Config``(켬): 포크의 Advanced 수정자 하나를 탭 I 에, 마스크를 M1(·M2·C1)에.

    * 고급 스케줄이 꺼진 포크도 치우침(bias)을 쓴다 — 여기서 고급 스케줄 끔은 치우침을 되돌리므로, 기본값이 아닌
      치우침은 고급 스케줄을 켜고 지수·오프셋 0 으로 옮긴다(같은 스케줄).
    * 포크의 색 대비 중심은 최대 채도의 비율(0..1)이다 — 지금 식(-1..1)으로는 ``2·c − 1``.
    * 마스크 미리보기는 포크가 만들던 Shift 항목(마지막 스텝에 미리보기 색으로 예전 방식 색 이동, 마스크로 가림)이 된다.
    * apply_to_hr 켬(포크 기본값)은 패스 Both, 끔은 Base."""
    v = fork_values(text)
    config = blank_config()
    config.enabled = True
    m = config.modifiers[0]
    m.update({"active": True, "pass": PASS_BOTH if v["apply_to_hr"] else PASS_BASE, "kind": KIND_LABELS["advanced"]})
    for name in _FORK_SAME_NAMES:
        m[name] = coerce(MODIFIER_BY_NAME[name], v[name])
    m["chroma_center"] = coerce(MODIFIER_BY_NAME["chroma_center"], 2.0 * v["chroma_center"] - 1.0)
    for ours, theirs in _FORK_SHIFT_NAMES:
        m[ours] = coerce(MODIFIER_BY_NAME[ours], v[theirs])
    if v["advanced"]:
        m["advanced"] = True
        for name in ("bias", "exponent", "start_off", "end_off"):
            m[name] = coerce(MODIFIER_BY_NAME[name], v[name])
    elif v["bias"] != 0.5:
        m["advanced"] = True
        m["bias"] = coerce(MODIFIER_BY_NAME["bias"], v["bias"])
        m["exponent"], m["start_off"], m["end_off"] = 0.0, 0.0, 0.0
    if not v["masking"]:
        return config
    config.masking = True
    leaf_a = config.leaves[0]
    for ours, theirs in _FORK_LEAF_A:
        leaf_a[ours] = coerce(LEAF_BY_NAME[ours], v[theirs])
    target, m["mask"] = leaf_a, MASK_TAGS[0]
    if v["mask_combine"]:
        leaf_b = config.leaves[1]
        for ours, theirs in _FORK_LEAF_B:
            leaf_b[ours] = coerce(LEAF_BY_NAME[ours], v[theirs])
        combo = config.combos[0]
        combo["mask_a"], combo["mask_b"] = MASK_TAGS[0], MASK_TAGS[1]
        combo["operation"] = coerce(COMBO_BY_NAME["operation"], v["mask_operation"])
        target, m["mask"] = combo, COMBO_TAGS[0]
    if v["mask_blur_radius"] > 0 or v["mask_spread"] != 0:
        target["blur"] = coerce(LEAF_BY_NAME["blur"], v["mask_blur_radius"])
        target["spread"] = coerce(LEAF_BY_NAME["spread"], v["mask_spread"])
    if v["mask_preview"]:
        red, green, blue = FORK_PREVIEW_COLORS[v["mask_preview_color"]]
        m.update({
            "kind": KIND_LABELS["shift"], "color_shift_amount": 1.0, "color_shift_mode": "legacy",
            "color_shift_red": red, "color_shift_green": green, "color_shift_blue": blue,
            "color_shift_brightness": 0.0, "strength": 1.0, "start": 1.0, "end": 1.0,
            "advanced": False, "bias": 0.5, "exponent": 0.0, "start_off": 0.0, "end_off": 0.0, "smooth": True,
        })
    return config


# ── 붙여 넣기 입구 ───────────────────────────────────────────────────────────
def from_infotext(params: Any) -> Optional[Config]:
    """infotext 매개변수(``core.image_metadata.parse_infotext`` 결과 같은 dict) → ``Config``, Colorcraft 열쇠가 없으면 None.

    자기 열쇠(``SAM Extra Colorcraft``)가 먼저다. 없으면 업스트림·포크의 ``Colorcraft`` 값(문자열, 또는 업스트림 콜백이
    만든 dict). 읽을 수 없는 값(다른 모양의 dict 등)은 None — 예외를 내지 않는다(확장 ``spec.decode``). 자기 열쇠의
    문자열은 무엇이든 자기 형식으로 읽는다(못 읽는 조각은 기본값 — 켜진 탭 없음)."""
    if not isinstance(params, dict):
        return None
    if INFOTEXT_KEY in params:
        raw, own = params[INFOTEXT_KEY], True
    elif FOREIGN_KEY in params:
        raw, own = params[FOREIGN_KEY], False
    else:
        return None
    try:
        if isinstance(raw, dict):
            return parse_upstream(raw)
        text = str(raw)
        if own:
            return parse_own(text)
        if is_upstream_text(text):
            return parse_upstream(text)
        if is_fork_text(text):
            return parse_fork(text)
    except Exception:   # noqa: BLE001 — 남의 확장이 남긴 값: 읽지 못하면 붙여 넣지 않는다(확장 decode 와 같다)
        return None
    return None


def value_format(text: Any) -> Optional[str]:
    """맨 값 하나의 형식 — ``own``(``v1;mods=``), ``upstream``(``MOD(``), ``fork``(다른 ``v1``), 아니면 None.
    가져오기 칸에 값만 붙여 넣었을 때 어느 열쇠의 값인지 정한다(확장의 구분 규칙과 같은 순서)."""
    if is_compact_arg(text) and isinstance(text, str):
        return "own"
    if is_upstream_text(text):
        return "upstream"
    if is_fork_text(text):
        return "fork"
    return None


def from_value(text: Any) -> Optional[Config]:
    """맨 값 하나(``v1;mods=…`` / ``MOD(…)`` / 포크 ``v1;…``) → ``Config``. 어느 형식도 아니면 None."""
    which = value_format(text)
    if which is None:
        return None
    return from_infotext({INFOTEXT_KEY if which == "own" else FOREIGN_KEY: text})


# ── 패스마다 도는 체인 (확장 ``spec.build_chain`` — 업스트림 노드가 만드는 항목과 같은 모양) ─────────────────
def runs_on_pass(modifier: Mapping, is_hires: bool) -> bool:
    """Base = 1차 패스(와 img2img), Hires = Hires 패스, Both = 둘 다."""
    which = modifier.get("pass", PASS_BASE)
    if which == PASS_BOTH:
        return True
    return (which == PASS_HIRES) == bool(is_hires)


def schedule_of(modifier: Mapping) -> dict:
    """노드의 스케줄 dict. 고급 스케줄 끔이면 치우침·지수·오프셋을 되돌린다(업스트림 Forge 스크립트 규칙)."""
    schedule = {name: modifier[name] for name in SCHEDULE_NAMES}
    if not modifier.get("advanced"):
        schedule.update({"bias": 0.5, "exponent": 0.0, "start_off": 0.0, "end_off": 0.0})
    return schedule


def params_of(modifier: Mapping) -> dict:
    """그 종류의 노드가 체인 항목에 싣는 값만, 노드 이름으로."""
    return {node_name: modifier[panel_name] for node_name, panel_name in KIND_PARAMS[kind_of(modifier)]}


def schedule_is_zero(schedule: Mapping) -> bool:
    """강도 0·두 오프셋 0 이면 모든 스텝이 0 이다."""
    return schedule["strength"] == 0 and schedule["start_off"] == 0 and schedule["end_off"] == 0


def params_are_noop(kind: str, params: Mapping) -> bool:
    """이 항목의 모든 편집이 0 — 노드가 입력을 그대로 돌려준다(스위치로 꺼진 묶음은 보지 않는다)."""
    if kind == "basic":
        names = ["contrast", "color_shift_amount"]
    elif kind == "advanced":
        names = ["exposure", "tone_compression", "contrast", "clarity", "sharpness", "temperature", "tint",
                 "vibrance", "saturation", "chroma_contrast"]
        if params["more_colors"]:
            names += list(CHROMA_PLUS_NAMES)
        if params["color_shift"]:
            names.append("color_shift_amount")
    elif kind == "luma":
        names = ["exposure", "tone_compression"]
    elif kind == "chroma":
        names = ["temperature", "tint", "vibrance", "saturation", "chroma_contrast"]
    elif kind == "chroma_plus":
        names = list(CHROMA_PLUS_NAMES)
    elif kind == "punch":
        names = ["contrast", "clarity", "sharpness"]
    elif kind == "shift":
        names = ["color_shift_amount"]
    else:
        return True
    return all(params[name] == 0 for name in names)


def _shaping_is_default(values: Mapping) -> bool:
    return values["blur"] == 0 and values["spread"] == 0 and values["contrast"] == 0 and not values["normalize"]


def _shaped(values: Mapping, inner: dict) -> dict:
    """흐림·넓힘·대비·정규화가 기본값이 아니면 그 노드로 감싼다."""
    if _shaping_is_default(values):
        return inner
    return {"blur": values["blur"], "spread": values["spread"], "contrast": values["contrast"],
            "normalize": values["normalize"], "a": inner}


def leaf_spec(leaf: Mapping) -> dict:
    """잎 하나의 마스크 스펙(업스트림 ``build_leaf_spec``)."""
    return _shaped(leaf, {
        "mask_axis": leaf["mask_axis"], "mask_mode": leaf["mask_mode"], "mask_center": leaf["mask_center"],
        "mask_hardness": leaf["mask_hardness"], "mask_width": leaf["mask_width"],
        "mask_strength": leaf["mask_strength"],
    })


def mask_specs(config: Config) -> tuple:
    """``(잎 스펙, 조합 스펙)`` — 태그로. 미완성 조합(A 나 B 가 비었거나 미완성)은 None.

    조합은 잎과 앞선 조합만 고르므로 순서대로 한 번 지나가면 모든 참조가 풀린다."""
    leaves = {MASK_TAGS[i]: leaf_spec(leaf) for i, leaf in enumerate(config.leaves)}
    combos: dict = {}

    def resolve(ref):
        if ref == MASK_NONE:
            return None
        return combos[ref] if ref in combos else leaves.get(ref)

    for i, combo in enumerate(config.combos):
        a, b = resolve(combo["mask_a"]), resolve(combo["mask_b"])
        combos[COMBO_TAGS[i]] = None if a is None or b is None else _shaped(
            combo, {"operation": combo["operation"], "a": a, "b": b})
    return leaves, combos


def mask_spec_for(config: Config, modifier: Mapping, specs: Optional[tuple] = None) -> Optional[dict]:
    """이 수정자를 가리는 마스크 스펙, 없으면 None(마스크 끔·고른 것 없음·Basic·미완성 조합)."""
    if not config.masking or not KIND_TAKES_MASK[kind_of(modifier)]:
        return None
    selection = modifier.get("mask", MASK_NONE)
    if selection == MASK_NONE:
        return None
    leaves, combos = specs if specs is not None else mask_specs(config)
    return leaves[selection] if selection in leaves else combos.get(selection)


def needs_basis(entries) -> bool:
    """기저 벡터가 필요한 항목이 있나 — Advanced 계열 종류, 또는 마스크를 쓰는 Shift."""
    return any(e["kind"] in ADVANCED_KINDS or (e["kind"] == "shift" and e.get("mask")) for e in entries)


@dataclass
class PassChain:
    """한 샘플링 패스가 도는 체인 항목과 상태 줄이 말하는 것."""

    entries: list               # {"kind", "params", "schedule", "mask"} — 업스트림 노드가 만드는 항목
    tags: list                  # 항목마다 수정자 태그
    unresolved_masks: list      # "II->C3" — 미완성 조합을 고른 수정자(마스크 없이 돈다)
    needs_basis: bool


def build_chain(config: Config, is_hires: bool) -> PassChain:
    """한 패스의 체인. 꺼진 탭, 다른 패스의 탭, 아무것도 바꾸지 못하는 항목(스케줄 0·값 0)은 뺀다."""
    if not config.enabled:
        return PassChain([], [], [], False)
    specs = mask_specs(config)
    entries, tags, unresolved = [], [], []
    for i, modifier in enumerate(config.modifiers):
        if not modifier["active"] or not runs_on_pass(modifier, is_hires):
            continue
        kind = kind_of(modifier)
        schedule = schedule_of(modifier)
        params = params_of(modifier)
        if schedule_is_zero(schedule) or params_are_noop(kind, params):
            continue
        mask = mask_spec_for(config, modifier, specs)
        if mask is None and config.masking and KIND_TAKES_MASK[kind] and modifier["mask"] in COMBO_TAGS:
            unresolved.append(f"{MODIFIER_TAGS[i]}->{modifier['mask']}")
        entries.append({"kind": kind, "params": params, "schedule": schedule, "mask": mask})
        tags.append(MODIFIER_TAGS[i])
    return PassChain(entries, tags, unresolved, needs_basis(entries))


# ── 한눈에 보기 (확장 편집기의 ●/○ 표시와 요약 줄 — 같은 규칙) ─────────────────────────
MARK_ON, MARK_EDITED = "●", "○"   # 켜진 수정자·쓰이는 마스크 / 바꿨지만 꺼진 것
INCOMPLETE = "(미완성)"           # 요약 줄 '➜ C1' 뒤: 조합이 풀리지 않아 수정자가 마스크 없이 돈다


def edited(config: Config, tag: str) -> bool:
    """켬(active)을 빼고 기본값과 다른 칸이 있나."""
    values = node_values(config, tag)
    return any(values[f.name] != node_default(tag, f) for f in node_fields(tag) if f.name != "active")


def used_masks(config: Config) -> tuple:
    """``(쓰이는 태그, 미완성 조합 태그)`` — 켜진 수정자가 실제로 가리는 잎·조합(미완성 조합과 그 너머는 빼고)."""
    _, combos = mask_specs(config)
    unresolved = {tag for tag, spec in combos.items() if spec is None}
    used: set = set()
    if not config.masking:
        return used, unresolved
    todo = _masked_selections(config)
    while todo:
        tag = todo.pop()
        if tag == MASK_NONE or tag in used or tag in unresolved:
            continue
        used.add(tag)
        if tag in COMBO_TAGS:
            combo = config.combos[COMBO_TAGS.index(tag)]
            todo.extend((combo["mask_a"], combo["mask_b"]))
    return used, unresolved


@dataclass(frozen=True)
class Overview:
    modifier_marks: tuple       # ((라벨, 태그), …) — 'I ●', 'III ○', 'II'
    mask_marks: tuple           # 잎 10 + 조합 5, 같은 모양
    text: str                   # '켠 수정자: I Advanced ➜ M1, III Punch (Hires) · 쓰는 마스크: M1' — 꺼졌으면 '꺼짐 — …'


def overview(config: Config) -> Overview:
    """편집기 고르개의 표시와 요약 줄(확장 ``panel_state.overview`` 의 글자 — HTML 은 빼고)."""
    used, unresolved = used_masks(config)
    modifier_marks, active = [], []
    for i, tag in enumerate(MODIFIER_TAGS):
        m = config.modifiers[i]
        if m["active"]:
            mark = MARK_ON
            text = f"{tag} {m['kind']}"
            if config.masking and KIND_TAKES_MASK[kind_of(m)] and m["mask"] != MASK_NONE:
                text += f" ➜ {m['mask']}" + (INCOMPLETE if m["mask"] in unresolved else "")
            if m["pass"] != PASS_BASE:
                text += f" ({m['pass']})"
            active.append(text)
        else:
            mark = MARK_EDITED if edited(config, tag) else ""
        modifier_marks.append((f"{tag} {mark}" if mark else tag, tag))
    mask_marks = []
    for tag in MASK_TAGS + COMBO_TAGS:
        mark = MARK_ON if tag in used else (MARK_EDITED if edited(config, tag) else "")
        mask_marks.append((f"{tag} {mark}" if mark else tag, tag))
    text = ("켠 수정자: " + ", ".join(active)) if active else "켠 수정자 없음"
    if config.masking:
        text += " · 쓰는 마스크: " + (", ".join(t for t in NODE_TAGS if t in used) if used else "없음")
    if not config.enabled:
        text = "꺼짐 — " + text
    return Overview(tuple(modifier_marks), tuple(mask_marks), text)


def describe(config: Config) -> str:
    """로그·알림·카드용 한 줄 = ``overview(config).text``."""
    return overview(config).text


# ── Vue 표 (frontend/src/utils/colorcraftSchema.generated.ts) ─────────────────────────────────
SCHEMA_TS_PATH = "frontend/src/utils/colorcraftSchema.generated.ts"


def _field_schema(f: Field) -> dict:
    return {"name": f.name, "kind": f.kind, "default": f.default, "min": f.minimum, "max": f.maximum,
            "step": f.step, "choices": list(f.choices), "group": f.group, "label": f.label, "title": f.title}


def schema() -> dict:
    """카드가 쓰는 표 전부(이름·기본값·범위·선택지·묶음·한국어 라벨) — ``schema_ts`` 가 TS 로 쓴다.

    확장 ``panel_state.schema()`` 의 열쇠는 같은 값으로 다 들어 있다(tests/test_colorcraft_spec_parity.py)."""
    return {
        "version": STATE_VERSION,
        "scriptName": SCRIPT_NAME,
        "infotextKey": INFOTEXT_KEY,
        "foreignKey": FOREIGN_KEY,
        "modifierTags": list(MODIFIER_TAGS),
        "maskTags": list(MASK_TAGS),
        "comboTags": list(COMBO_TAGS),
        "maskNone": MASK_NONE,
        "passBase": PASS_BASE,
        "passHires": PASS_HIRES,
        "passBoth": PASS_BOTH,
        "kinds": list(KINDS),
        "kindLabels": dict(KIND_LABELS),
        "kindIds": dict(KIND_IDS),
        "kindParams": {kind: [list(pair) for pair in pairs] for kind, pairs in KIND_PARAMS.items()},
        "kindGroups": {kind: list(groups) for kind, groups in KIND_GROUPS.items()},
        "kindGates": {kind: list(gates) for kind, gates in KIND_GATES.items()},
        "kindTakesMask": dict(KIND_TAKES_MASK),
        "advancedKinds": [kind for kind in KINDS if kind in ADVANCED_KINDS],
        "scheduleNames": list(SCHEDULE_NAMES),
        "shapingNames": list(SHAPING_NAMES),
        "groupOrder": list(GROUP_ORDER),
        "groupLabels": dict(GROUP_LABELS),
        "globalFields": [_field_schema(f) for f in GLOBAL_FIELDS],
        "modifierFields": [_field_schema(f) for f in MODIFIER_FIELDS],
        "leafFields": [_field_schema(f) for f in LEAF_FIELDS],
        "comboFields": [_field_schema(f) for f in COMBO_FIELDS],
        "activeDefaults": [modifier_default(i, "active") for i in range(MODIFIER_COUNT)],
        "comboRefChoices": [list(combo_ref_choices(i)) for i in range(COMBO_COUNT)],
        "choiceLabels": {name: dict(labels) for name, labels in CHOICE_LABELS.items()},
        "marks": {"on": MARK_ON, "edited": MARK_EDITED, "incomplete": INCOMPLETE},
    }


def _ts_literal(value: Any, indent: int = 0) -> str:
    """TS 객체 리터럴(2칸 들여쓰기, 열쇠는 따옴표, 숫자는 ``js_number``)."""
    pad, inner = "  " * indent, "  " * (indent + 1)
    if isinstance(value, Mapping):
        if not value:
            return "{}"
        rows = [f"{inner}{json.dumps(str(k), ensure_ascii=False)}: {_ts_literal(v, indent + 1)}" for k, v in value.items()]
        return "{\n" + ",\n".join(rows) + f",\n{pad}}}"
    if isinstance(value, (list, tuple)):
        if not value:
            return "[]"
        if all(not isinstance(v, (Mapping, list, tuple)) for v in value):
            return "[" + ", ".join(_js_json(v) for v in value) + "]"
        return "[\n" + ",\n".join(f"{inner}{_ts_literal(v, indent + 1)}" for v in value) + f",\n{pad}]"
    return _js_json(value)


_SCHEMA_TS_HEAD = """\
// Generated from core/colorcraft.py by schema_ts() - do not edit (tests/test_colorcraft_card_mirror.py compares it).
// Regenerate: venv\\Scripts\\python.exe -c "from core import colorcraft; colorcraft.write_schema_ts()"

export type ColorcraftFieldKind = 'float' | 'bool' | 'choice'

export interface ColorcraftField {
  readonly name: string
  readonly kind: ColorcraftFieldKind
  readonly default: number | boolean | string
  readonly min: number | null
  readonly max: number | null
  readonly step: number | null
  readonly choices: readonly string[]
  readonly group: string
  readonly label: string
  readonly title: string
}

export interface ColorcraftSchema {
  readonly version: number
  readonly scriptName: string
  readonly infotextKey: string
  readonly foreignKey: string
  readonly modifierTags: readonly string[]
  readonly maskTags: readonly string[]
  readonly comboTags: readonly string[]
  readonly maskNone: string
  readonly passBase: string
  readonly passHires: string
  readonly passBoth: string
  readonly kinds: readonly string[]
  readonly kindLabels: Readonly<Record<string, string>>
  readonly kindIds: Readonly<Record<string, string>>
  readonly kindParams: Readonly<Record<string, readonly (readonly [string, string])[]>>
  readonly kindGroups: Readonly<Record<string, readonly string[]>>
  readonly kindGates: Readonly<Record<string, readonly string[]>>
  readonly kindTakesMask: Readonly<Record<string, boolean>>
  readonly advancedKinds: readonly string[]
  readonly scheduleNames: readonly string[]
  readonly shapingNames: readonly string[]
  readonly groupOrder: readonly string[]
  readonly groupLabels: Readonly<Record<string, string>>
  readonly globalFields: readonly ColorcraftField[]
  readonly modifierFields: readonly ColorcraftField[]
  readonly leafFields: readonly ColorcraftField[]
  readonly comboFields: readonly ColorcraftField[]
  readonly activeDefaults: readonly boolean[]
  readonly comboRefChoices: readonly (readonly string[])[]
  readonly choiceLabels: Readonly<Record<string, Readonly<Record<string, string>>>>
  readonly marks: { readonly on: string; readonly edited: string; readonly incomplete: string }
}

"""


def schema_ts() -> str:
    """frontend/src/utils/colorcraftSchema.generated.ts 의 내용(LF, 끝 줄바꿈 하나)."""
    return _SCHEMA_TS_HEAD + f"export const COLORCRAFT_SCHEMA: ColorcraftSchema = {_ts_literal(schema())}\n"


def write_schema_ts(path: Optional[str] = None) -> str:
    """``schema_ts()`` 를 UTF-8(BOM 없음)·LF 로 쓴다. 기본 위치는 저장소의 ``SCHEMA_TS_PATH``."""
    import os

    target = path or os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), SCHEMA_TS_PATH)
    with open(target, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(schema_ts())
    return target


# ── 앱 설정 (위젯 ``_cc_*``·저장·프리셋) ─────────────────────────────────────────
WIDGET_KEYS = ("enabled", "masking", "img2img", "state")
SETTINGS_KEY = "colorcraft_settings"      # prompt_settings.json·생성 프리셋 키


def widget_id(key: str) -> str:
    """위젯 프록시 id = Vue ``storeWidgets`` 키(``_cc_enabled``·``_cc_masking``·``_cc_img2img``·``_cc_state``)."""
    if key not in WIDGET_KEYS:
        raise KeyError(key)
    return f"_cc_{key}"


@dataclass(frozen=True)
class ColorcraftSettings:
    """카드 값. ``state`` 는 정규 상태 문자열(``dump_state``). ``img2img`` 는 앱 전용(Forge 는 img2img 탭에 패널이
    따로 있고 기본값이 꺼짐이다 — 'I2I·인페인트에도 적용')."""

    enabled: bool = False
    masking: bool = False
    img2img: bool = False
    state: str = DEFAULT_STATE


APP_DEFAULTS = ColorcraftSettings()       # = 확장 기본값(꺼짐·마스크 끔·수정자 I 켜짐·나머지 기본값) + I2I 끔


def parse_settings(raw: Optional[Mapping], *, base: ColorcraftSettings = APP_DEFAULTS) -> ColorcraftSettings:
    """위젯 문자열·저장 dict → 설정. 없거나 비어 있는 칸은 ``base`` 칸, 읽을 수 없는 상태도 ``base`` 상태(로그)."""
    if not isinstance(raw, Mapping):
        return base
    values: dict = {}
    for key in ("enabled", "masking", "img2img"):
        if raw.get(key) not in (None, ""):
            values[key] = _truthy(raw.get(key), getattr(base, key))
    if "state" in raw and raw.get("state") is not None:
        try:
            values["state"] = canonical_state(raw.get("state"))
        except StateError as exc:
            logger.warning("[Colorcraft] 저장된 상태를 읽지 못해 이전 값을 둔다: %s", exc)
    return replace(base, **values)


def widget_values(settings: ColorcraftSettings) -> dict:
    """설정 → 위젯 문자열('true'/'false', 상태 문자열)."""
    return {"enabled": "true" if settings.enabled else "false",
            "masking": "true" if settings.masking else "false",
            "img2img": "true" if settings.img2img else "false",
            "state": settings.state}


def config_of(settings: ColorcraftSettings) -> Config:
    """설정 → 확장 ``Config``. 상태를 읽을 수 없으면(저장 파일을 손으로 고친 경우) 기본값 위의 설정(로그)."""
    try:
        return config_from_state(settings.enabled, settings.masking, settings.state)
    except StateError as exc:
        logger.warning("[Colorcraft] 상태를 읽지 못해 기본값으로 본다: %s", exc)
        return config_from_state(settings.enabled, settings.masking, DEFAULT_STATE)


def settings_of(config: Config, *, img2img: bool = False) -> ColorcraftSettings:
    """``Config`` → 카드 설정(가져오기·붙여 넣기 결과를 위젯에 쓸 때)."""
    return ColorcraftSettings(bool(config.enabled), bool(config.masking), bool(img2img), dump_state(config))


# ── 블록 판정 ─────────────────────────────────────────────────────────────────
TARGET_T2I, TARGET_I2I, TARGET_AUX = "t2i", "i2i", "aux"             # core/alwayson_propagation TARGETS
BACKEND_WEBUI, BACKEND_COMFY, BACKEND_KREA2 = "webui", "comfyui", "krea2"
PROVENANCE_USER = "user"                                            # core/alwayson_propagation PROVENANCE_USER
REASON_OFF, REASON_KREA2, REASON_COMFY_PENDING, REASON_I2I_OFF, REASON_NO_ENTRIES, REASON_SEND = (
    "off", "krea2", "comfy_pending", "i2i_off", "no_entries", "send")


@dataclass(frozen=True)
class ColorcraftPlan:
    block: Optional[dict]          # alwayson 블록 {"args": [to_infotext]} | None(보내지 않음)
    notice: Any                    # 늘 None — 생성 전 알림 없음(결과 infotext 'SAM Extra Colorcraft status' 가 정답)
    reason: str                    # off / krea2 / comfy_pending / i2i_off / no_entries / send
    provenance: str = PROVENANCE_USER
    base_tags: tuple = ()          # 1차 패스(와 img2img)에서 도는 수정자
    hires_tags: tuple = ()         # Hires 패스에서 도는 수정자(i2i 는 늘 비어 있다 — Hires 패스가 없다)


def plan(settings: ColorcraftSettings, *, target: str, backend: str, capabilities: Any = None) -> ColorcraftPlan:
    """이 요청에 Colorcraft 블록을 실을지. 스크립트 없음·확인 전은 여기서 보지 않는다(core/alwayson_propagation 게이트).

    1 꺼짐 → 없음  2 Krea2 → 없음(카드가 말한다)  3 ComfyUI → 없음(``comfy_pending`` — 팩 1.7.0 연결 전, R16)
    4 i2i 이고 'I2I·인페인트에도 적용' 끔 → 없음  5 돌 항목이 없음(켠 수정자가 없거나 모두 0 — 확장도 아무것도 붙이지
    않는다; i2i 는 Base 패스만 본다) → 없음(``no_entries``, 카드가 '바뀌는 값 없음')  6 그 밖 → 블록. aux 도 같은 블록이다
    (봉투가 메인 체인과 같게 — ``blocks_for`` 가 어느 보조 패스에도 주지 않는다). 출처는 늘 사용자 값."""
    if not settings.enabled:
        return ColorcraftPlan(None, None, REASON_OFF)
    if backend == BACKEND_KREA2:
        return ColorcraftPlan(None, None, REASON_KREA2)
    if backend == BACKEND_COMFY:
        return ColorcraftPlan(None, None, REASON_COMFY_PENDING)
    if target == TARGET_I2I and not settings.img2img:
        return ColorcraftPlan(None, None, REASON_I2I_OFF)
    config = config_of(settings)
    base = tuple(build_chain(config, False).tags)
    hires = () if target == TARGET_I2I else tuple(build_chain(config, True).tags)
    if not base and not hires:
        return ColorcraftPlan(None, None, REASON_NO_ENTRIES)
    return ColorcraftPlan(as_block(config), None, REASON_SEND, PROVENANCE_USER, base, hires)


# ── ComfyUI (앱 노드 팩 1.7.0 ``ForgeNeoColorcraft`` — 연결은 phase C) ─────────────────────────────
COMFY_NODE_CLASS = "ForgeNeoColorcraft"
COMFY_CONTRACT = "v1"                     # 노드 입력 'contract' 의 선택지 — 형식이 바뀌면 v2(옛 팩은 컴파일 전에 거절)
COMFY_INPUTS = ("model", "vae", "settings_json", "contract", "pre_dd_sigma")
COMFY_PRE_DD_SIGMA = True                 # P10 옵션은 Forge 전용 — ComfyUI 는 확장 기본값(켬)
COMFY_MIN_PACK_VERSION = "1.7.0"          # 노드가 처음 들어온 팩


__all__ = [
    "ADVANCED_KINDS", "APP_DEFAULTS", "BACKEND_COMFY", "BACKEND_KREA2", "BACKEND_WEBUI", "BOOL", "BlockError",
    "CHOICE", "CHOICE_LABELS", "CHROMA_PLANE_OPTIONS", "CHROMA_PLUS_NAMES", "COLOR_SHIFT_MODE_OPTIONS",
    "COMBO_BY_NAME", "COMBO_COUNT", "COMBO_FIELDS", "COMBO_OP_OPTIONS", "COMBO_TAGS", "COMFY_CONTRACT",
    "COMFY_INPUTS", "COMFY_MIN_PACK_VERSION", "COMFY_NODE_CLASS", "COMFY_PRE_DD_SIGMA", "ColorcraftPlan",
    "ColorcraftSettings", "Config", "DEFAULT_STATE", "FLOAT", "FOREIGN_KEY", "FORK_CHOICES", "FORK_PARAMS",
    "FORK_PREVIEW_COLORS", "Field", "GLOBAL_FIELDS", "GROUP_LABELS", "GROUP_ORDER", "INCOMPLETE", "INFOTEXT_KEY",
    "INFOTEXT_KEYS", "KINDS", "KIND_CHOICES", "KIND_GATES", "KIND_GROUPS", "KIND_IDS", "KIND_LABELS", "KIND_PARAMS",
    "KIND_TAKES_MASK", "LEAF_BY_NAME", "LEAF_FIELDS", "MARK_EDITED", "MARK_ON", "MASK_AXIS_OPTIONS", "MASK_CHOICES",
    "MASK_COUNT", "MASK_MODE_OPTIONS", "MASK_NONE", "MASK_TAGS", "MODIFIER_BY_NAME", "MODIFIER_COUNT",
    "MODIFIER_FIELDS", "MODIFIER_TAGS", "NODE_TAGS", "OPT_PRE_DD", "Overview", "PASS_BASE", "PASS_BOTH",
    "PASS_CHOICES", "PASS_HIRES", "PRE_DD_INFOTEXT_KEY", "PROVENANCE_USER", "PassChain", "REASON_COMFY_PENDING",
    "REASON_I2I_OFF", "REASON_KREA2", "REASON_NO_ENTRIES", "REASON_OFF", "REASON_SEND", "SCHEDULE_NAMES",
    "SCHEMA_TS_PATH", "SCRIPT_NAME", "SETTINGS_KEY", "SHAPING_NAMES", "STATE_PATHS", "STATE_VERSION", "STATUS_KEY",
    "StateError", "TARGET_AUX", "TARGET_I2I", "TARGET_T2I", "UPSTREAM_COMBO_FIELDS", "UPSTREAM_LEAF_FIELDS",
    "UPSTREAM_MODIFIER_VALUE_FIELDS", "VERSION", "WIDGET_KEYS", "XYZ_PREFIX", "active_masks", "arg_names", "as_block",
    "blank_config", "build_chain", "canonical_state", "coerce", "combo_ref_choices", "config_from_mapping",
    "config_from_state", "config_of", "config_values", "default_config", "describe", "dump_state", "edited",
    "fork_values", "from_infotext", "from_value", "is_compact_arg", "is_fork_text", "is_upstream_text", "js_number",
    "kind_of", "leaf_spec", "mask_spec_for", "mask_specs", "modifier_default", "needs_basis", "node_choices",
    "node_default", "node_fields", "node_values", "overview", "params_are_noop", "params_of", "parse_compact",
    "parse_fork", "parse_own", "parse_script_block", "parse_settings", "parse_state", "parse_upstream", "plan",
    "runs_on_pass", "schedule_is_zero", "schedule_of", "schema", "schema_ts", "settings_of", "state_paths",
    "to_infotext", "upstream_dict", "used_masks", "value_format", "widget_id", "widget_values", "write_schema_ts",
    "written_modifier_fields",
]
