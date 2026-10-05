# core/extra_samplers.py
"""Extra Samplers (sam-extra ``scripts/anima_extra_samplers.py``) — 샘플러 18종 표와 ER SDE 값 다섯.
순수 로직(Qt·네트워크·torch 없음).

확장(v0.33.0)은 Forge 샘플러 목록에 18개를 등록하고(``sam3ext/extra_samplers/registry.py`` SPECS), always-on 스크립트
``alwayson_scripts["Extra Samplers"]`` 로 그중 **ER SDE 셋(Reverse-time·ODE·Tunable)이 읽는 값 5개**를 받는다 — 위치
인자 ``[max_stage, eta, noise_window, noise_start, noise_end]``(params.ARG_NAMES, 모두 생략 가능, 넘치는 것은 무시):
``{"args": [3, 1.0, false, 0.2, 0.8]}``.

- **값은 요청 하나에 하나다.** 확장 ``process()`` 가 요청에 값을 붙이고(블록이 없는 요청은 ui() 기본값), 샘플러가 그
  요청의 모든 패스(메인·Hires)에서 읽는다. ADetailer 안쪽 img2img 는 스크립트를 거르므로 직전 요청 값(``_ACTIVE``)을
  쓴다. 그래서 블록은 메인 요청과 보조 img2img(Refine·SAM3·ADetailer·손 재구성)에 같이 간다(core/alwayson_propagation).
- **언제 싣나**(계획 X5 — Extra Schedulers 와 같은 규칙, ``plan``): Forge 기능 스냅샷이 스크립트를 확인했으면 늘(XYZ
  Sampler 축이 ER SDE 로 바꾸는 칸·사용자 ui-config 기본값과 무관하게), 확인 못 했으면 어느 패스가 ER SDE 셋 중 하나를
  쓸 때만. ComfyUI 는 늘 싣는다(계획 C6 — 컴파일러가 값이 기본값과 다를 때만 앱 노드 팩 1.7.0 의
  ``ForgeNeoExtraSamplerSettings`` 를 넣는다 — 기본값이면 그래프가 바이트 단위로 그대로). Krea2 는 보내지 않는다.
- **인자 수**(계획 X10): 앱은 늘 다섯을 보낸다. Forge API 는 설치된 스크립트가 가진 수만큼만 넘기므로 sam-extra
  0.31·0.32(인자 2개)는 뒤 셋(잡음 구간)을 조용히 버린다 — 스냅샷이 인자 2개를 보이고 구간이 켜져 있으면
  ``window_unsupported`` 가 참이다(카드·결과 알림이 ``WINDOW_UNSUPPORTED_TEXT`` 로 말한다). 0.33.0 이 옛 2개 블록을 받으면
  구간은 ui() 기본값(끔)이다 — 결과가 그대로다.
- 붙여 넣기(``from_infotext``)·infotext 기록(``infotext_items``)·상태 줄(``parse_status``)은 확장 params·substep_guard 규칙의
  거울이다.

확장 코드는 GPL-3.0 이라 옮기지 않았다 — 규칙(범위·자르기·붙여 넣기·기록·표)을 다시 구현했다.
``tests/test_extra_samplers_core.py`` 가 설치된 확장(params 의 coerce_*·settings_from_args·paste_*·parse_noise_window·
record_infotext, substep_guard 의 상태 줄, registry/params/script 의 상수·표 AST)과 같은 입력으로 비교한다.
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass, replace
from types import MappingProxyType
from typing import Any, Iterable, Mapping, NamedTuple, Optional

from core import sam3_args
from core.extra_schedulers import PASS_HIRES, PASS_MAIN, PASS_SAM3, adetailer_pass, pass_label

# ── 확장 계약 ─────────────────────────────────────────────────────────────────
SCRIPT_NAME = "Extra Samplers"   # scripts/anima_extra_samplers.py TITLE
ARG_NAMES = ("max_stage", "eta", "noise_window", "noise_start", "noise_end")   # params.ARG_NAMES

# sam3ext/extra_samplers/params.py (ComfyUI SamplerER_SDE 입력 범위와 같다)
DEFAULT_MAX_STAGE = 3
MAX_STAGE_MIN = 1
MAX_STAGE_MAX = 3
DEFAULT_ETA = 1.0
ETA_MIN = 0.0
ETA_MAX = 10.0
ETA_STEP = 0.01
DEFAULT_NOISE_WINDOW = False
DEFAULT_NOISE_START = 0.2
DEFAULT_NOISE_END = 0.8
NOISE_PERCENT_MIN = 0.0
NOISE_PERCENT_MAX = 1.0
NOISE_PERCENT_STEP = 0.01

# (키, 종류, 기본값, 범위) — core/anima_guidance 스펙 형식. 레지스트리 app_spec 이 접두 'xsamp_' 를 떼어
# 확장 ui()·script-info 와 비교한다(test_app_positional_specs_match_live_values).
ARG_SPEC = (
    ("xsamp_max_stage", "int", DEFAULT_MAX_STAGE, (MAX_STAGE_MIN, MAX_STAGE_MAX)),
    ("xsamp_eta", "float", DEFAULT_ETA, (ETA_MIN, ETA_MAX)),
    ("xsamp_noise_window", "bool", DEFAULT_NOISE_WINDOW, None),
    ("xsamp_noise_start", "float", DEFAULT_NOISE_START, (NOISE_PERCENT_MIN, NOISE_PERCENT_MAX)),
    ("xsamp_noise_end", "float", DEFAULT_NOISE_END, (NOISE_PERCENT_MIN, NOISE_PERCENT_MAX)),
)
SPEC_PREFIX = "xsamp_"
MAX_STAGE_STEP = 1   # ui() 슬라이더 step

# infotext(params.py·substep_guard.py)
KEY_MAX_STAGE = "ER SDE max stage"         # ≠ 3 일 때
KEY_ETA = "ER SDE eta"                     # ≠ 1.0, 또는 그 요청의 Forge Eta ≠ 1 일 때(Reverse-time·Tunable)
KEY_NOISE_WINDOW = "ER SDE noise window"   # 구간 켬 + Reverse-time·Tunable + η > 0 일 때 — "0.2-0.8"
NOISE_WINDOW_OFF = "off"                   # XYZ 칸 값(구간 끔)
KEY_FORGE_ETA = "Eta"                      # Forge 조상 샘플러 η — aoleg infotext 붙여 넣기에서 Reverse-time η 로 읽는다
KEY_SAMPLER, KEY_HIRES_SAMPLER = "Sampler", "Hires sampler"
STATUS_KEY = "Extra Samplers status"       # 상태 부분들을 "; " 로 이은 값
STATUS_SEPARATOR = "; "
DY_SKIPPED_PREFIX = "dy sub-steps skipped"   # Dy 부분: "dy sub-steps skipped (<이유> + <이유>)"
DY_REASON_SEPARATOR = " + "
WINDOW_SKIPPED_STATUS = "noise window skipped (no percent_to_sigma)"   # params.WINDOW_SKIPPED_STATUS
# substep_guard.REASON_* — Dy 보조 스텝을 끈 이유
DY_REASON_SPECTRUM = "Spectrum"
DY_REASON_WAN = "Wan I2V concat_latent"
DY_REASON_PID = "PiD lq_latent"
DY_REASON_EXTRA_CONCAT = "extra_concat_condition"
DY_REASONS = (DY_REASON_SPECTRUM, DY_REASON_WAN, DY_REASON_PID, DY_REASON_EXTRA_CONCAT)

# XYZ 축(scripts/anima_extra_samplers.py) — 앱 XYZ 에는 값 축이 없다(샘플러 축은 있다)
XYZ_PREFIX = "[Extra Samplers]"
XYZ_AXES = (f"{XYZ_PREFIX} {KEY_MAX_STAGE}", f"{XYZ_PREFIX} {KEY_ETA}", f"{XYZ_PREFIX} {KEY_NOISE_WINDOW}")

# ── 샘플러 표(sam3ext/extra_samplers/registry.py SPECS 순서 = Forge 목록 순서) ─────────────────────
LABEL_ER_SDE_REVERSE_TIME = "ER SDE (Reverse-time)"
LABEL_ER_SDE_ODE = "ER SDE (ODE)"
LABEL_ER_SDE_TUNABLE = "ER SDE (Tunable)"
LABEL_DPMPP_4M_SDE = "DPM++ 4M SDE"
LABEL_EULER_DY_CFG_PP = "Euler Dy CFG++"
LABEL_EULER_SMEA_DY_CFG_PP = "Euler SMEA Dy CFG++"
LABEL_EULER_DY = "Euler Dy"
LABEL_EULER_SMEA_DY = "Euler SMEA Dy"
LABEL_DPMPP_2M_SDE_HEUN = "DPM++ 2M SDE Heun"
LABEL_DPMPP_2M_FLOW_ODE = "DPM++ 2M (flow ODE)"
LABEL_DPMPP_2M_HEUN_FLOW_ODE = "DPM++ 2M Heun (flow ODE)"
LABEL_DPMPP_3M_FLOW_ODE = "DPM++ 3M (flow ODE)"
LABEL_UNIPC_BH2 = "UniPC bh2"
LABEL_CFGPP_UD10_AB = "CFG++ UD10 AB"
LABEL_IPNDM = "IPNDM"
LABEL_IPNDM_V = "IPNDM_V"
LABEL_DEIS = "DEIS"
LABEL_RESTART_FLOW = "Restart (flow)"

# Forge 표기 → 확장 별칭(첫 별칭 = ComfyUI 이름 — 앱이 ComfyUI 로 컴파일하는 이름, reForge k_* 별칭은 뒤에)
ALIASES = MappingProxyType({
    LABEL_ER_SDE_REVERSE_TIME: ("er_sde_reverse_time",),
    LABEL_ER_SDE_ODE: ("er_sde_ode",),
    LABEL_ER_SDE_TUNABLE: ("er_sde_tunable",),
    LABEL_DPMPP_4M_SDE: ("dpmpp_4m_sde",),
    LABEL_EULER_DY_CFG_PP: ("euler_dy_cfg_pp",),
    LABEL_EULER_SMEA_DY_CFG_PP: ("euler_smea_dy_cfg_pp",),
    LABEL_EULER_DY: ("euler_dy", "k_euler_dy"),
    LABEL_EULER_SMEA_DY: ("euler_smea_dy", "k_euler_smea_dy"),
    LABEL_DPMPP_2M_SDE_HEUN: ("dpmpp_2m_sde_heun", "k_dpmpp_2m_sde_heun"),
    LABEL_DPMPP_2M_FLOW_ODE: ("dpmpp_2m_flow_ode",),
    LABEL_DPMPP_2M_HEUN_FLOW_ODE: ("dpmpp_2m_heun_flow_ode",),
    LABEL_DPMPP_3M_FLOW_ODE: ("dpmpp_3m_flow_ode",),
    LABEL_UNIPC_BH2: ("uni_pc_bh2",),
    LABEL_CFGPP_UD10_AB: ("cfgpp_ud10_ab",),
    LABEL_IPNDM: ("ipndm",),
    LABEL_IPNDM_V: ("ipndm_v",),
    LABEL_DEIS: ("deis",),
    LABEL_RESTART_FLOW: ("restart_flow",),
})
LABELS = tuple(ALIASES)
COMFY_NAMES = MappingProxyType({label: aliases[0] for label, aliases in ALIASES.items()})
LABEL_OF_COMFY = MappingProxyType({name: label for label, name in COMFY_NAMES.items()})
# ComfyUI 쪽(계획 X11·X12): 앱 노드 팩이 등록하는 12개 / ComfyUI 코어에 이미 같은 계산으로 있는 6개(별칭 행만)
PACK_SAMPLERS = (
    "er_sde_reverse_time", "er_sde_ode", "er_sde_tunable", "dpmpp_4m_sde", "euler_dy_cfg_pp",
    "euler_smea_dy_cfg_pp", "euler_dy", "euler_smea_dy", "dpmpp_2m_flow_ode", "dpmpp_2m_heun_flow_ode",
    "dpmpp_3m_flow_ode", "restart_flow",
)
CORE_SAMPLERS = ("dpmpp_2m_sde_heun", "uni_pc_bh2", "cfgpp_ud10_ab", "ipndm", "ipndm_v", "deis")
CFGPP_UD10_MIN_COMFY = "0.35.0"   # cfgpp_ud10_ab 이 ComfyUI 코어에 들어온 버전(3ac5d794)

# 묶음(registry SamplerSpec 의 kind·cfg_pp 와 params 의 표)
ER_SDE_LABELS = (LABEL_ER_SDE_REVERSE_TIME, LABEL_ER_SDE_ODE, LABEL_ER_SDE_TUNABLE)   # max stage 를 읽는다
ETA_LABELS = (LABEL_ER_SDE_REVERSE_TIME, LABEL_ER_SDE_TUNABLE)                        # η 를 읽는다
WINDOW_LABELS = ETA_LABELS                                                             # 잡음 구간을 읽는다
CFG_PP_LABELS = (LABEL_EULER_DY_CFG_PP, LABEL_EULER_SMEA_DY_CFG_PP, LABEL_CFGPP_UD10_AB)   # CFG 1~2 권장
DY_KIND = MappingProxyType({
    LABEL_EULER_DY_CFG_PP: "dy", LABEL_EULER_SMEA_DY_CFG_PP: "smea", LABEL_EULER_DY: "dy", LABEL_EULER_SMEA_DY: "smea",
})
FLOW_ODE_LABELS = (LABEL_DPMPP_2M_FLOW_ODE, LABEL_DPMPP_2M_HEUN_FLOW_ODE, LABEL_DPMPP_3M_FLOW_ODE)
RESTART_LABELS = (LABEL_RESTART_FLOW,)
# 패널 종류(sampler_kind) — 카드가 상자·힌트를 고른다
KIND_ER_SDE, KIND_DY, KIND_SMEA, KIND_CFG_PP, KIND_RESTART, KIND_OTHER = "er_sde", "dy", "smea", "cfg_pp", "restart", ""

# 샘플러 상수(확장 모듈 리터럴 — 앱 힌트·문서와 ComfyUI 코어 동등의 근거, 계획 X11)
RESTART_S_MIN = 0.1        # restart_flow.py — 재시작 구간(ε-등가 σ)
RESTART_S_MAX = 2.0
RESTART_MIN_STEPS = 20     # 이보다 적은 스텝이면 재시작 없음(Heun)
RESTART_TIMES = 1
RESTART_TIMES_LONG = 2
RESTART_LONG_FROM = 36
UD10_HISTORY_WEIGHT = 0.25          # cfgpp_ud10_ab.py — ComfyUI sample_cfgpp_ud10_ab 이 넘기는 값
UD10_ZERO_WEIGHT = 1.0
UD10_UNCOND_HISTORY_WEIGHT = 0.1
IPNDM_MAX_ORDER = 4                 # ipndm_deis.py — ComfyUI KSampler 기본과 같다
DEIS_MAX_ORDER = 3
DEIS_MODE = "tab"

# ── ComfyUI 계약(노드 팩 1.7.0 — 계획 PB-XSa5·C3) ───────────────────────────────────────
COMFY_SETTINGS_NODE = "ForgeNeoExtraSamplerSettings"
COMFY_INPUTS = ("model", *ARG_NAMES)
COMFY_REPORT_INPUT = "owned_names"           # compat.owned_names_input — 팩이 가진 이름(R6)
COMFY_MIN_PACK_VERSION = "1.7.0"
COMFY_MODEL_OPTIONS_KEY = "ai_studio_extra_samplers"
COMFY_DISCARD_PENULTIMATE = ("dpmpp_4m_sde", "dpmpp_3m_flow_ode")   # 팩이 KSampler 버림 집합에 더한다
COMFY_AUTOMATIC_SCHEDULER = MappingProxyType({"dpmpp_4m_sde": "exponential"})   # Forge 'Automatic'(U4)

# ── 위젯·저장 ───────────────────────────────────────────────────────────────
WIDGET_KEYS = ARG_NAMES
WIDGET_PREFIX = "_xsamp_"
SETTINGS_KEY = "extra_samplers_settings"   # prompt_settings.json·생성 프리셋 키

# 옛 sam-extra(0.31·0.32 — 인자 2개)에서 잡음 구간이 켜져 있을 때(계획 X10)
WINDOW_MIN_VERSION = "0.33.0"
WINDOW_UNSUPPORTED_TEXT = "잡음 구간은 sam-extra 0.33.0 이상에서만 적용됩니다"


def widget_id(key: str) -> str:
    """위젯 프록시 id = Vue ``storeWidgets`` 키 — 접두 + 확장 인자 이름."""
    if key not in WIDGET_KEYS:
        raise KeyError(key)
    return f"{WIDGET_PREFIX}{key}"


# ── 정규화 (확장 params.py 규칙의 거울) ─────────────────────────────────────────────
_TRUE_TEXTS = ("true", "on", "yes", "1")


def _stage_or_raise(value: Any, default: int) -> int:
    """params.coerce_max_stage 와 같다 — float() 가 OverflowError 를 내면(아주 큰 int) 그대로 낸다."""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    if not math.isfinite(number):
        return default
    return int(min(MAX_STAGE_MAX, max(MAX_STAGE_MIN, int(round(number)))))


def _eta_or_raise(value: Any, default: float) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    if not math.isfinite(number):
        return default
    return float(min(ETA_MAX, max(ETA_MIN, number)))


def _percent_or_raise(value: Any, default: float) -> float:
    if isinstance(value, bool):
        return default
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    if not math.isfinite(number):
        return default
    return float(min(NOISE_PERCENT_MAX, max(NOISE_PERCENT_MIN, number)))


def coerce_max_stage(value: Any, default: int = DEFAULT_MAX_STAGE) -> int:
    """1..3 의 정수 차수(반올림 — 2.5 는 2). 유한한 수가 아니면 ``default``. bool 은 수다(True → 1, 확장과 같다).

    확장과 다른 것 하나: float 로 바꿀 수 없을 만큼 큰 int 는 확장에서 OverflowError 지만 여기서는 ``default``
    (저장값 하나 때문에 기여자가 실패하지 않게 — 블록 전체 해석 ``coerce_args`` 는 확장 process 처럼 전부 기본값)."""
    try:
        return _stage_or_raise(value, default)
    except OverflowError:
        return default


def coerce_eta(value: Any, default: float = DEFAULT_ETA) -> float:
    """0..10 의 η. 유한한 수가 아니면 ``default`` (큰 int 는 ``coerce_max_stage`` 와 같은 차이)."""
    try:
        return _eta_or_raise(value, default)
    except OverflowError:
        return default


def coerce_noise_window(value: Any) -> bool:
    """체크 칸 — True·수 1·글자 true/on/yes/1(대소문자 무시)만 켬, 나머지는 끔(params.coerce_noise_window)."""
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return value == 1
    if isinstance(value, str):
        return value.strip().lower() in _TRUE_TEXTS
    return False


def coerce_noise_percent(value: Any, default: float) -> float:
    """샘플링 진행률 0..1. bool(여기서는 수가 아니다)·유한한 수가 아닌 것은 ``default``."""
    try:
        return _percent_or_raise(value, default)
    except OverflowError:
        return default


def coerce_noise_start(value: Any) -> float:
    return coerce_noise_percent(value, DEFAULT_NOISE_START)


def coerce_noise_end(value: Any) -> float:
    return coerce_noise_percent(value, DEFAULT_NOISE_END)


# ── 설정 ──────────────────────────────────────────────────────────────────────
@dataclass(frozen=True)
class ExtraSamplersSettings:
    """한 생성의 값 다섯(확장 params.ErSdeSettings 와 같은 칸·기본값). 시작 ≥ 끝은 빈 구간이다(바꾸지 않는다)."""

    max_stage: int = DEFAULT_MAX_STAGE
    eta: float = DEFAULT_ETA
    noise_window: bool = DEFAULT_NOISE_WINDOW
    noise_start: float = DEFAULT_NOISE_START
    noise_end: float = DEFAULT_NOISE_END

    def as_args(self) -> list:
        """확장 위치 인자 다섯(int, float, bool, float, float)."""
        return [int(self.max_stage), float(self.eta), bool(self.noise_window),
                float(self.noise_start), float(self.noise_end)]

    @property
    def window_empty(self) -> bool:
        """구간이 켜져 있지만 시작 ≥ 끝 — 어느 스텝에도 잡음을 넣지 않는다."""
        return bool(self.noise_window) and self.noise_start >= self.noise_end


# 확장 API 기본값 = ui() 기본값. 앱 기본값도 같다 — 블록이 없던 지금까지의 Forge 결과와 같다.
EXTENSION_DEFAULTS = ExtraSamplersSettings()
APP_DEFAULTS = EXTENSION_DEFAULTS
DEFAULTS = EXTENSION_DEFAULTS


def coerce_args(args: Any) -> ExtraSamplersSettings:
    """스크립트 위치 인자 → 확장이 실제로 쓰는 설정(``process`` 의 거울): ``settings_from_args`` — 빠진 뒤 인자는 기본값,
    넘치는 것은 무시 — 이고, 그것이 예외를 내면(반복할 수 없는 값·아주 큰 int) 확장처럼 전부 기본값이다."""
    try:
        items = list(args or ())

        def arg(index: int, default: Any) -> Any:
            return items[index] if len(items) > index else default

        return ExtraSamplersSettings(
            max_stage=_stage_or_raise(arg(0, DEFAULT_MAX_STAGE), DEFAULT_MAX_STAGE),
            eta=_eta_or_raise(arg(1, DEFAULT_ETA), DEFAULT_ETA),
            noise_window=coerce_noise_window(arg(2, DEFAULT_NOISE_WINDOW)),
            noise_start=_percent_or_raise(arg(3, DEFAULT_NOISE_START), DEFAULT_NOISE_START),
            noise_end=_percent_or_raise(arg(4, DEFAULT_NOISE_END), DEFAULT_NOISE_END),
        )
    except Exception:
        return EXTENSION_DEFAULTS


def _readable_number(value: Any, *, bools: bool) -> bool:
    if value is None or (isinstance(value, bool) and not bools):
        return False
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError, OverflowError):
        return False


def parse_settings(raw: Optional[Mapping], *, base: ExtraSamplersSettings = APP_DEFAULTS) -> ExtraSamplersSettings:
    """위젯 문자열·저장 dict(키 = ``ARG_NAMES``) → 설정. 없거나(None) 읽을 수 없는 칸은 ``base`` 칸을 쓴다.

    ``base`` 가 기본값이면 칸마다 확장 coerce_* 와 같다. 체크 칸은 None 이 아니면 늘 읽는다('0'·'1'·'true'…)."""
    if not isinstance(raw, Mapping):
        return base
    values: dict = {}
    if _readable_number(raw.get("max_stage"), bools=True):
        values["max_stage"] = coerce_max_stage(raw.get("max_stage"))
    if _readable_number(raw.get("eta"), bools=True):
        values["eta"] = coerce_eta(raw.get("eta"))
    if raw.get("noise_window") is not None:
        values["noise_window"] = coerce_noise_window(raw.get("noise_window"))
    for key, coerce_number in (("noise_start", coerce_noise_start), ("noise_end", coerce_noise_end)):
        if _readable_number(raw.get(key), bools=False):
            values[key] = coerce_number(raw.get(key))
    return replace(base, **values)


def widget_number(value: Any) -> str:
    """위젯 숫자 문자열 — 0.5 → '0.5', 0.0 → '0', 2.0 → '2', 1/3 → '0.3333333333333333'(값을 잃지 않는다)."""
    number = float(value)
    if number == 0:
        return "0"
    text = repr(number)
    return text[:-2] if text.endswith(".0") else text


def widget_values(settings: ExtraSamplersSettings) -> dict[str, str]:
    """설정 → 위젯 문자열(키 5개 — 차수 '3', 수는 ``widget_number``, 체크는 '1'/'0')."""
    return {
        "max_stage": str(int(settings.max_stage)),
        "eta": widget_number(settings.eta),
        "noise_window": "1" if settings.noise_window else "0",
        "noise_start": widget_number(settings.noise_start),
        "noise_end": widget_number(settings.noise_end),
    }


def is_default(settings: ExtraSamplersSettings) -> bool:
    """확장 기본값과 같은 값인가(블록 출처를 앱 기본값으로 적는다)."""
    return settings == EXTENSION_DEFAULTS


def as_block(settings: ExtraSamplersSettings) -> dict:
    """alwayson 블록 — 위치 인자 다섯(계획 X10: 옛 확장이 뒤 셋을 버려도 늘 다섯)."""
    return {"args": settings.as_args()}


def parse_script_block(block: Any) -> Optional[ExtraSamplersSettings]:
    """보낸 alwayson 블록 → 확장이 읽는 설정(인자 0·2·5개 모두 — 옛 프리셋·기록의 2개 블록은 구간 기본값). 블록이
    아니면 None."""
    if not isinstance(block, Mapping):
        return None
    return coerce_args(block.get("args"))


def format_noise_window(start: Any, end: Any) -> str:
    """infotext·XYZ 의 구간 글자 — "0.2-0.8"(각 끝을 소수 6자리로 반올림, ``:g``; params.format_noise_window)."""
    return f"{round(float(start), 6):g}-{round(float(end), 6):g}"


# 음이 아닌 십진수(지수 허용 — format_noise_window 가 1e-06 을 쓴다)
_PERCENT_TEXT = r"(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?"
_WINDOW_RE = re.compile(rf"\s*({_PERCENT_TEXT})\s*-\s*({_PERCENT_TEXT})\s*")


def parse_noise_window(text: Any) -> Optional[tuple]:
    """"off" → (False, 0.2, 0.8); "a-b"(0 ≤ a, b ≤ 1, 숫자·대시 둘레 공백 허용, a ≥ b 도 빈 구간으로 유효) →
    (True, a, b); 그 밖은 None(params.parse_noise_window)."""
    value = str(text if text is not None else "").strip()
    if value.lower() == NOISE_WINDOW_OFF:
        return False, DEFAULT_NOISE_START, DEFAULT_NOISE_END
    match = _WINDOW_RE.fullmatch(value)
    if match is None:
        return None
    start, end = float(match.group(1)), float(match.group(2))
    for bound in (start, end):
        if not (math.isfinite(bound) and NOISE_PERCENT_MIN <= bound <= NOISE_PERCENT_MAX):
            return None
    return True, start, end


def describe(settings: ExtraSamplersSettings, samplers: Optional[Iterable[Any]] = None) -> str:
    """로그·알림용 한 줄 — "ER SDE 차수 2 · η 0.5 · 잡음 0.2–0.8". ``samplers``(표기·ComfyUI 이름)를 주면 그
    샘플러가 읽는 칸만(ODE 는 차수만), 없으면 모두. 구간이 꺼져 있으면 잡음 칸은 없다."""
    labels = None if samplers is None else {resolve_sampler(item) for item in samplers}
    if labels is not None and not labels & set(ER_SDE_LABELS):
        return ""
    parts = [f"ER SDE 차수 {int(settings.max_stage)}"]
    if labels is None or labels & set(ETA_LABELS):
        parts.append(f"η {widget_number(settings.eta)}")
        if settings.noise_window:
            window = f"잡음 {widget_number(settings.noise_start)}–{widget_number(settings.noise_end)}"
            parts.append(window + (" (빈 구간)" if settings.window_empty else ""))
    return " · ".join(parts)


# ── 샘플러 이름 ────────────────────────────────────────────────────────────────
def _fold(text: Any) -> str:
    return " ".join(str(text or "").split()).casefold()


_FOLDED_SAMPLERS = MappingProxyType({
    **{_fold(label): label for label in LABELS},
    **{_fold(alias): label for label, aliases in ALIASES.items() for alias in aliases},
})


def resolve_sampler(text: Any) -> Optional[str]:
    """샘플러 표기·ComfyUI 이름·reForge 별칭(대소문자·앞뒤·겹친 공백 무시) → 이 확장의 Forge 표기. 'Use same sampler'·
    다른 샘플러·빈 값은 None."""
    return _FOLDED_SAMPLERS.get(_fold(text))


def sampler_kind(text: Any) -> Optional[str]:
    """카드의 상자·힌트 종류 — 'er_sde'·'dy'·'smea'·'cfg_pp'(UD10 AB)·'restart'·''(그 밖의 이 확장 샘플러),
    이 확장 샘플러가 아니면 None. Dy 넷 중 CFG++ 둘은 ``is_cfg_pp`` 도 참이다."""
    label = resolve_sampler(text)
    if label is None:
        return None
    if label in ER_SDE_LABELS:
        return KIND_ER_SDE
    if label in DY_KIND:
        return KIND_SMEA if DY_KIND[label] == "smea" else KIND_DY
    if label in CFG_PP_LABELS:
        return KIND_CFG_PP
    if label in RESTART_LABELS:
        return KIND_RESTART
    return KIND_OTHER


def is_cfg_pp(text: Any) -> bool:
    return resolve_sampler(text) in CFG_PP_LABELS


def uses_values(text: Any) -> bool:
    """이 블록의 값(차수)을 읽는 샘플러인가 — ER SDE 셋."""
    return resolve_sampler(text) in ER_SDE_LABELS


def uses_eta(text: Any) -> bool:
    """η 와 잡음 구간을 읽는 샘플러인가 — Reverse-time·Tunable."""
    return resolve_sampler(text) in ETA_LABELS


uses_window = uses_eta


def comfy_name(text: Any) -> Optional[str]:
    """ComfyUI 에서 쓰는 이름(첫 별칭). 이 확장 샘플러가 아니면 None."""
    label = resolve_sampler(text)
    return COMFY_NAMES[label] if label else None


# ── 블록 판정 ─────────────────────────────────────────────────────────────────
TARGET_T2I, TARGET_I2I, TARGET_AUX = "t2i", "i2i", "aux"             # core/alwayson_propagation TARGETS
BACKEND_WEBUI, BACKEND_COMFY, BACKEND_KREA2 = "webui", "comfyui", "krea2"
PROVENANCE_USER, PROVENANCE_APP_DEFAULT = "user", "app_default"     # core/alwayson_propagation PROVENANCE_*

REASON_SEND = "send"                    # 스냅샷이 스크립트를 확인했다 — 늘 싣는다
REASON_SEND_UNCONFIRMED = "send_unconfirmed"   # 확인 못 했지만 어느 패스가 ER SDE — 게이트가 정한다
REASON_NOT_NEEDED = "not_needed"        # 확인 못 했고 값을 읽는 패스가 없다 — 싣지 않는다(알림 없음)
# 보조 패스 봉투(aux) — 확인 못 했어도 싣는다. 보조 요청(Refine·SAM3·ADetailer 의 자기 샘플러, 손 재구성)의 샘플러는 보내는
# 곳만 안다 — 요청 직전에 core/alwayson_propagation.drop_unread 가 그 요청의 패스(value_sampler_passes)로 정한다(계획 X5)
REASON_SEND_AUX = "send_aux"
REASON_COMFY = "comfy"                  # ComfyUI — 늘 싣는다(노드를 넣을지는 컴파일러가 정한다, 계획 C6)
REASON_KREA2 = "krea2"


@dataclass(frozen=True)
class ExtraSamplersPlan:
    block: Optional[dict]          # alwayson 블록 {"args": [5개]} | None(보내지 않음)
    notice: Any                    # 늘 None — 생성 전 알림은 게이트(빠진 블록)·카드(X10)·결과 알림이 한다
    reason: str                    # REASON_*
    provenance: str = PROVENANCE_USER


def _script_detail(capabilities: Any) -> Optional[Mapping]:
    if getattr(capabilities, "known", None) is not True:
        return None
    script = getattr(capabilities, "script", None)
    if not callable(script):
        return None
    try:
        detail = script(SCRIPT_NAME)
    except Exception:
        return None
    return detail if isinstance(detail, Mapping) else None


def script_confirmed(capabilities: Any, *, img2img: bool) -> Optional[bool]:
    """기능 스냅샷이 이 스크립트를 확인했나 — True(있음)·False(없음)·None(모름). 판정은 core/alwayson_propagation
    ``script_available`` 와 같다(known 스냅샷의 script(title) 의 txt2img·img2img 칸)."""
    detail = _script_detail(capabilities)
    if detail is None:
        return None
    value = detail.get("img2img" if img2img else "present")
    return value if isinstance(value, bool) else None


def live_argc(capabilities: Any) -> Optional[int]:
    """스냅샷이 본 스크립트 인자 수(script-info) — 모르면 None."""
    detail = _script_detail(capabilities)
    if detail is None or detail.get("present") is not True:
        return None
    value = detail.get("live_argc")
    return value if isinstance(value, int) and not isinstance(value, bool) else None


def window_supported(capabilities: Any) -> Optional[bool]:
    """연결된 Forge 의 sam-extra 가 잡음 구간(인자 3~5)을 받나 — 인자 다섯 이상이면 True, 적으면 False(0.31·0.32),
    모르면 None(계획 X10: Forge API 는 스크립트가 가진 수만큼만 넘긴다)."""
    argc = live_argc(capabilities)
    if argc is None:
        return None
    return argc >= len(ARG_NAMES)


def window_unsupported(settings: ExtraSamplersSettings, capabilities: Any) -> bool:
    """구간을 켰는데 연결된 sam-extra 가 그 인자를 버린다(확실할 때만 참) — 카드·결과 알림 ``WINDOW_UNSUPPORTED_TEXT``.
    η 0 이면 거짓 — 구간은 어느 판에서도 아무 일도 하지 않는다(확장 registry._er_sde_window: 구간 끔이거나 η ≤ 0 이면
    구간 없음, params.record_infotext 도 η > 0 일 때만 'ER SDE noise window' 를 적는다)."""
    return bool(settings.noise_window) and settings.eta > 0 and window_supported(capabilities) is False


def plan(settings: ExtraSamplersSettings, *, target: str, backend: str, capabilities: Any = None,
         passes: Iterable[Any] = ()) -> ExtraSamplersPlan:
    """이 요청에 Extra Samplers 블록을 실을지(계획 X5 — core/extra_schedulers.plan 과 같은 규칙).

    1 Krea2 → 없음  2 ComfyUI → 늘 블록(``REASON_COMFY`` — 스냅샷·게이트가 없다. 컴파일러가 값이 기본값과 다를 때만
      ``ForgeNeoExtraSamplerSettings`` 를 넣어 다른 그래프는 바이트 단위로 그대로 — 계획 R5·C3·C6)
    3 Forge 스냅샷이 스크립트를 확인 → 늘 블록(값은 ER SDE 패스만 읽어 결과가 같다 — XYZ Sampler 축 칸·사용자 ui-config
      기본값과 무관하게 같은 요청이 된다)
    4 보조 패스 봉투(aux) → 늘 블록(``REASON_SEND_AUX``). 보조 요청의 샘플러(Refine·SAM3·ADetailer 의 자기 샘플러, 손
      재구성의 T2I 샘플러)는 T2I 패널의 패스가 아니라 보내는 곳만 안다 — 요청 직전에 core/alwayson_propagation
      ``drop_unread`` 가 새 스냅샷과 그 요청의 패스(``value_sampler_passes``)로 3·5 를 다시 판정하고 게이트가 출처로 정한다
    5 확인 못 함(모름·없음) → ``passes``(ER SDE 셋을 쓰는 패스 — ``passes_from_widgets``)가 있을 때만 블록. 그 뒤는
      core/alwayson_propagation 게이트가 출처로 정한다
    6 그 밖 → 없음(알림 없음 — 쓰지 않는 값)

    출처: 값이 확장 기본값과 같으면 앱 기본값, 다르면 사용자 값. 보조 패스 봉투(aux)는 스냅샷이 확인하면 t2i 와 같은
    블록이다. i2i 는 스냅샷의 img2img 목록을 본다. 블록은 늘 인자 다섯이다(X10 — 옛 확장의 경고는 ``window_unsupported``)."""
    if backend == BACKEND_KREA2:
        return ExtraSamplersPlan(None, None, REASON_KREA2)
    provenance = PROVENANCE_APP_DEFAULT if is_default(settings) else PROVENANCE_USER
    if backend == BACKEND_COMFY:
        return ExtraSamplersPlan(as_block(settings), None, REASON_COMFY, provenance)
    if script_confirmed(capabilities, img2img=target == TARGET_I2I) is True:
        return ExtraSamplersPlan(as_block(settings), None, REASON_SEND, provenance)
    if target == TARGET_AUX:
        return ExtraSamplersPlan(as_block(settings), None, REASON_SEND_AUX, provenance)
    if any(True for _ in passes or ()):
        return ExtraSamplersPlan(as_block(settings), None, REASON_SEND_UNCONFIRMED, provenance)
    return ExtraSamplersPlan(None, None, REASON_NOT_NEEDED, provenance)


# ── 어느 패스가 값을 읽나 ──────────────────────────────────────────────────────
USE_SAME_SAMPLER = "Use same sampler"   # Forge Hires·ADetailer·SAM3 의 '같은 샘플러'


class PassUse(NamedTuple):
    pass_id: str     # core.extra_schedulers 패스 id — PASS_MAIN | PASS_HIRES | adetailer_pass(i) | PASS_SAM3
    sampler: str     # ER_SDE_LABELS 의 표기


def _truthy(value: Any, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    text = str(value).strip().lower()
    if not text:
        return default
    return text in {"1", "true", "yes", "on"}


def _uses_same(text: Any) -> bool:
    """비었거나 'Use same sampler'(대소문자 무시) — 메인 샘플러를 쓴다."""
    folded = _fold(text)
    return not folded or folded == USE_SAME_SAMPLER.casefold()


def _value_uses(entries: Iterable[tuple], resolve) -> tuple:
    out = []
    for pass_id, text in entries:
        label = resolve(text)
        if label in ER_SDE_LABELS:
            out.append(PassUse(pass_id, label))
    return tuple(out)


def _script_block(payload: Mapping, title: str) -> Optional[Mapping]:
    scripts = payload.get("alwayson_scripts") if isinstance(payload, Mapping) else None
    if not isinstance(scripts, Mapping):
        return None
    folded = title.casefold()
    for name, block in scripts.items():
        if str(name).casefold() == folded and isinstance(block, Mapping):
            return block
    return None


def value_sampler_passes(payload: Any, *, resolve=None) -> tuple:
    """요청 페이로드에서 ER SDE 셋을 쓰는 패스 → ``PassUse`` 튜플(패스 순서). 비면 이 블록의 값을 읽는 패스가 없다.

    각 패스가 실제로 읽는 칸(core/extra_schedulers.value_scheduler_passes 와 같은 규칙 — 컴파일러는 ``resolve=`` 로
    Comfy 이름 변환을 넘긴다): 메인 ``sampler_name``; ``enable_hr`` 면 ``hr_sampler_name``(비었거나 'Use same' 이면
    메인); ADetailer 슬롯(켜진 탭·모델 'None' 아님)은 ``ad_use_sampler`` 면 ``ad_sampler`` 아니면 메인; SAM3(켜짐·
    'Mask only' 아님)는 ``sam3_sampler``('Use same' 이면 메인). ``resolve`` 기본은 ``resolve_sampler``."""
    resolve = resolve or resolve_sampler
    if not isinstance(payload, Mapping):
        return ()
    main = payload.get("sampler_name")
    entries: list = [(PASS_MAIN, main)]
    if _truthy(payload.get("enable_hr")):
        hires = payload.get("hr_sampler_name")
        entries.append((PASS_HIRES, main if _uses_same(hires) else hires))
    adetailer = _script_block(payload, "ADetailer")
    args = adetailer.get("args") if adetailer else None
    if isinstance(args, (list, tuple)) and args and _truthy(args[0], True):
        slots = [item for item in list(args)[2:]
                 if isinstance(item, Mapping) and _truthy(item.get("ad_tab_enable"), True)
                 and str(item.get("ad_model") or "None").strip().casefold() != "none"]
        for index, slot in enumerate(slots, start=1):
            text = slot.get("ad_sampler") if _truthy(slot.get("ad_use_sampler")) else main
            entries.append((adetailer_pass(index), main if _uses_same(text) else text))
    sam3 = _script_block(payload, sam3_args.SCRIPT_SAM3)
    sam3_list = sam3.get("args") if sam3 else None
    state = sam3_list[0] if isinstance(sam3_list, (list, tuple)) and sam3_list else None
    if isinstance(state, Mapping) and _truthy(state.get("sam3_enable", state.get("enabled", True)), True) \
            and sam3_args.mode_of(dict(state)) != sam3_args.MODE_MASK_ONLY:
        text = state.get("sam3_sampler")
        entries.append((PASS_SAM3, main if _uses_same(text) else text))
    return _value_uses(entries, resolve)


# 패스를 정하는 위젯 id(Vue storeWidgets 키)
MAIN_SAMPLER_WIDGET = "sampler_combo"
HIRES_GROUP_WIDGET, HIRES_SAMPLER_WIDGET = "hires_options_group", "hires_sampler_combo"
ADETAILER_GROUP_WIDGET = "adetailer_group"
# (슬롯 번호, 슬롯 켬, '별도 샘플러' 켬, 샘플러)
ADETAILER_SLOT_WIDGETS = ((1, "ad_slot1_group", "_ad_s1_use_sampler", "_ad_s1_sampler"),
                          (2, "ad_slot2_group", "_ad_s2_use_sampler", "_ad_s2_sampler"))
SAM3_GROUP_WIDGET, SAM3_MODE_WIDGET = "sam3_group", "_sam3_mode"
SAM3_USE_SAMPLER_WIDGET, SAM3_SAMPLER_WIDGET = "_sam3_use_sampler", "_sam3_sampler"


def pass_widget_ids() -> tuple:
    """``passes_from_widgets`` 가 읽는 위젯 id 전부(순서 고정)."""
    ids = [MAIN_SAMPLER_WIDGET, HIRES_GROUP_WIDGET, HIRES_SAMPLER_WIDGET, ADETAILER_GROUP_WIDGET]
    for _index, *widget_ids in ADETAILER_SLOT_WIDGETS:
        ids.extend(widget_ids)
    ids.extend((SAM3_GROUP_WIDGET, SAM3_MODE_WIDGET, SAM3_USE_SAMPLER_WIDGET, SAM3_SAMPLER_WIDGET))
    return tuple(ids)


def passes_from_widgets(values: Any, *, img2img: bool = False, img2img_main: Any = None) -> tuple:
    """화면 값(위젯 id → 값, 체크는 bool 또는 'true'/'false')에서 ER SDE 셋을 쓰는 패스 → ``PassUse`` 튜플.

    다음 T2I 요청이 실제로 만드는 패스와 같은 규칙(ui/generator_generation 의 페이로드 조립): 메인; Hires 켬이고 샘플러가
    'Use same' 이 아니면 그 값; ADetailer 켬 + 슬롯 켬이면 '별도 샘플러' 켬일 때 슬롯 샘플러 아니면 메인; SAM3 켬 +
    'Mask only' 아님이면 '별도 샘플러' 켬이고 값이 있을 때 그 값 아니면 메인.

    ``img2img``: img2img 요청에는 Hires 가 없고 메인은 그 요청이 스스로 싣는 샘플러(``img2img_main``)다 — I2I·인페인트
    요청은 샘플러를 싣지 않아(core/i2i_payload — Forge 기본 샘플러로 돈다) None, 채팅 이미지 편집은 T2I 콤보 값을 싣는다
    (``_chat_generation_snapshot(target='i2i')`` — hr_* 만 뗀다). ADetailer·SAM3 는 그 요청 안에서도 자기 샘플러(또는 그
    메인)로 돈다."""
    values = values if isinstance(values, Mapping) else {}
    main = img2img_main if img2img else values.get(MAIN_SAMPLER_WIDGET)
    entries: list = [(PASS_MAIN, main)]
    if not img2img and _truthy(values.get(HIRES_GROUP_WIDGET)):
        hires = values.get(HIRES_SAMPLER_WIDGET)
        entries.append((PASS_HIRES, main if _uses_same(hires) else hires))
    if _truthy(values.get(ADETAILER_GROUP_WIDGET)):
        for index, enabled, use_sampler, sampler in ADETAILER_SLOT_WIDGETS:
            if not _truthy(values.get(enabled)):
                continue
            text = values.get(sampler) if _truthy(values.get(use_sampler)) else main
            entries.append((adetailer_pass(index), main if _uses_same(text) else text))
    if _truthy(values.get(SAM3_GROUP_WIDGET)) and \
            sam3_args.mode_of({"sam3_mode": values.get(SAM3_MODE_WIDGET)}) != sam3_args.MODE_MASK_ONLY:
        text = values.get(SAM3_SAMPLER_WIDGET) if _truthy(values.get(SAM3_USE_SAMPLER_WIDGET)) else main
        entries.append((PASS_SAM3, main if _uses_same(text) else text))
    return _value_uses(entries, resolve_sampler)


# ── infotext (결과 기록·붙여 넣기) ───────────────────────────────────────────────
def _names_sampler(value: Any, label: str) -> bool:
    text = str(value or "").strip()
    return text == label or text.startswith(label + " ")


def _forge_eta_in_infotext(params: Mapping, forge_eta: Any) -> bool:
    """그 요청의 infotext 에 Forge ``Eta`` 가 실릴 수 있나(그러면 붙여 넣기가 그것을 우리 η 로 읽는다)."""
    if KEY_FORGE_ETA in params:
        return True
    try:
        value = float(forge_eta)
    except (TypeError, ValueError):
        return False
    return math.isfinite(value) and value != 1.0


def _record(params: Mapping, label: str, settings: ExtraSamplersSettings, forge_eta: Any,
            window_skipped: bool) -> dict:
    """샘플러 ``label`` 한 패스가 적는 키 → 값(확장 record_infotext 의 규칙; ``params`` 는 그때까지의 infotext)."""
    written: dict = {}
    if label not in ER_SDE_LABELS:
        return written
    if settings.max_stage != DEFAULT_MAX_STAGE:
        written[KEY_MAX_STAGE] = settings.max_stage
    if label not in ETA_LABELS:
        return written
    if settings.eta != DEFAULT_ETA or _forge_eta_in_infotext(params, forge_eta):
        written[KEY_ETA] = round(settings.eta, 6)
    if settings.noise_window and settings.eta > 0 and not window_skipped:
        written[KEY_NOISE_WINDOW] = format_noise_window(settings.noise_start, settings.noise_end)
    return written


def infotext_items(settings: ExtraSamplersSettings, samplers: Iterable[Any], *, forge_eta: Any = None,
                   window_skipped: bool = False, params: Optional[Mapping] = None) -> dict:
    """생성이 쓴 샘플러(패스 순서 — 메인·Hires, 표기나 ComfyUI 이름)마다 확장 record_infotext 를 차례로 적용해 이 기능이
    적는 키 → 값. ``params`` 는 그 전의 infotext(Forge ``Eta`` 가 있으면 η 가 1.0 이어도 적는다), ``forge_eta`` 는 그 요청의
    Forge η. ComfyUI PNG 메타데이터가 Forge 와 같게 적을 때 쓴다(계획 C3)."""
    current = dict(params) if isinstance(params, Mapping) else {}
    items: dict = {}
    for item in samplers or ():
        label = resolve_sampler(item) or str(item or "")
        written = _record(current, label, settings, forge_eta, window_skipped)
        current.update(written)
        items.update(written)
    return items


def _pasted_window(params: Mapping) -> tuple:
    parsed = parse_noise_window(params.get(KEY_NOISE_WINDOW)) if KEY_NOISE_WINDOW in params else None
    if parsed is None:
        return DEFAULT_NOISE_WINDOW, DEFAULT_NOISE_START, DEFAULT_NOISE_END
    return parsed


def from_infotext(params: Any) -> ExtraSamplersSettings:
    """PNG Info 붙여 넣기·'이 설정으로 생성' — 확장 paste_* 다섯의 거울(늘 다섯 칸을 모두 정한다):
    차수는 키(없으면 3), η 는 ``ER SDE eta`` → 없으면 Reverse-time infotext 의 ``Eta``(aoleg 호환) → 1.0, 구간은
    읽을 수 있는 ``ER SDE noise window`` 면 켬과 시작/끝, 없거나 못 읽으면 끔·0.2·0.8."""
    params = params if isinstance(params, Mapping) else {}
    max_stage = coerce_max_stage(params.get(KEY_MAX_STAGE, DEFAULT_MAX_STAGE))
    if KEY_ETA in params:
        eta = coerce_eta(params.get(KEY_ETA))
    elif any(_names_sampler(params.get(key), LABEL_ER_SDE_REVERSE_TIME) for key in (KEY_SAMPLER, KEY_HIRES_SAMPLER)) \
            and KEY_FORGE_ETA in params:
        eta = coerce_eta(params.get(KEY_FORGE_ETA))
    else:
        eta = DEFAULT_ETA
    noise_window, noise_start, noise_end = _pasted_window(params)
    return ExtraSamplersSettings(max_stage=max_stage, eta=eta, noise_window=noise_window,
                                 noise_start=noise_start, noise_end=noise_end)


# '이 설정으로 생성'이 되살리는 칸 — 차수는 ER SDE 셋, η·잡음 구간은 Reverse-time·Tunable 이 읽는다(계획 X3b)
RESTORE_STAGE_FIELDS = ("max_stage",)
RESTORE_ETA_FIELDS = ("eta", "noise_window", "noise_start", "noise_end")


def infotext_er_sde_samplers(params: Any) -> tuple:
    """infotext 의 메인('Sampler')·Hires('Hires sampler') 샘플러 중 ER SDE 셋(표기, 메인 먼저·중복 없이). 표기는
    ``resolve_sampler`` 로 접어 읽는다 — 앱 콤보가 표기·ComfyUI 이름·별칭을 같은 항목으로 고르는 것과 같다
    (ui/generation_settings_apply.apply_sampler_scheduler). 'Use same sampler'·다른 샘플러는 뺀다."""
    params = params if isinstance(params, Mapping) else {}
    labels: list = []
    for key in (KEY_SAMPLER, KEY_HIRES_SAMPLER):
        label = resolve_sampler(params.get(key))
        if label in ER_SDE_LABELS and label not in labels:
            labels.append(label)
    return tuple(labels)


def restore_from_infotext(params: Any, *, base: ExtraSamplersSettings = APP_DEFAULTS
                          ) -> Optional[ExtraSamplersSettings]:
    """'이 설정으로 생성'(계획 A-R1·X3b) — 이미지의 메인·Hires 샘플러가 ER SDE 셋일 때만 **그 샘플러가 읽는** 칸을
    ``from_infotext`` 규칙(확장 paste_* — 키 없으면 기본값, Reverse-time 이미지의 Forge 'Eta' 는 η)으로 되살린다: 차수는
    셋 다, η·잡음 구간은 Reverse-time·Tunable 이 있을 때만(ODE 만이면 그대로). 샘플러 이름만 오고 값이 남으면 다른
    그림이 된다. 어느 패스도 ER SDE 가 아니면 None(칸을 건드리지 않는다).

    접어 읽은 표기('er_sde_reverse_time' 같은 ComfyUI 이름)는 확장 표기로 바꿔 ``from_infotext`` 에 넘긴다 — 그 안의
    'Eta' 대체 규칙(확장의 정확한 표기 비교)이 앱이 고른 샘플러와 같은 것을 보게."""
    used = infotext_er_sde_samplers(params)
    if not used:
        return None
    pasted = dict(params)
    for key in (KEY_SAMPLER, KEY_HIRES_SAMPLER):
        label = resolve_sampler(pasted.get(key))
        if label is not None:
            pasted[key] = label
    restored = from_infotext(pasted)
    fields = RESTORE_STAGE_FIELDS + (RESTORE_ETA_FIELDS if any(label in ETA_LABELS for label in used) else ())
    return replace(base, **{field_name: getattr(restored, field_name) for field_name in fields})


# ── 상태 줄 ────────────────────────────────────────────────────────────────────
class SamplerStatus(NamedTuple):
    dy_reasons: tuple       # Dy 보조 스텝을 끈 이유(DY_REASONS 의 글자, 적힌 순서)
    window_skipped: bool    # 모델 예측기에 percent_to_sigma 가 없어 잡음 구간을 버렸다
    other: tuple            # 이 앱이 모르는 부분(새 확장)


def parse_status(value: Any) -> SamplerStatus:
    """``Extra Samplers status`` 값 → 부분들. 값은 부분을 "; " 로 이은 것이고(먼저 나온 순서) Dy 부분은
    "dy sub-steps skipped (<이유> + <이유>)" 다 — 먼저 "; " 로 나눈 뒤 읽는다."""
    dy: list = []
    window = False
    other: list = []
    parts = [piece.strip() for piece in str(value or "").split(STATUS_SEPARATOR.strip())]
    for part in (piece for piece in parts if piece):
        if part.startswith(DY_SKIPPED_PREFIX + " (") and part.endswith(")"):
            inner = part[len(DY_SKIPPED_PREFIX) + 2:-1]
            dy.extend(reason.strip() for reason in inner.split(DY_REASON_SEPARATOR.strip()) if reason.strip())
        elif part == WINDOW_SKIPPED_STATUS:
            window = True
        else:
            other.append(part)
    return SamplerStatus(tuple(dy), window, tuple(other))


__all__ = [
    "ADETAILER_GROUP_WIDGET", "ADETAILER_SLOT_WIDGETS", "ALIASES", "APP_DEFAULTS", "ARG_NAMES", "ARG_SPEC",
    "BACKEND_COMFY", "BACKEND_KREA2", "BACKEND_WEBUI", "CFGPP_UD10_MIN_COMFY", "CFG_PP_LABELS", "COMFY_AUTOMATIC_SCHEDULER",
    "COMFY_DISCARD_PENULTIMATE", "COMFY_INPUTS", "COMFY_MIN_PACK_VERSION", "COMFY_MODEL_OPTIONS_KEY", "COMFY_NAMES",
    "COMFY_REPORT_INPUT", "COMFY_SETTINGS_NODE", "CORE_SAMPLERS", "DEFAULTS", "DEFAULT_ETA", "DEFAULT_MAX_STAGE",
    "DEFAULT_NOISE_END", "DEFAULT_NOISE_START", "DEFAULT_NOISE_WINDOW", "DEIS_MAX_ORDER", "DEIS_MODE", "DY_KIND",
    "DY_REASONS", "DY_REASON_EXTRA_CONCAT", "DY_REASON_PID", "DY_REASON_SEPARATOR", "DY_REASON_SPECTRUM", "DY_REASON_WAN",
    "DY_SKIPPED_PREFIX", "ER_SDE_LABELS", "ETA_LABELS", "ETA_MAX", "ETA_MIN", "ETA_STEP", "EXTENSION_DEFAULTS",
    "ExtraSamplersPlan", "ExtraSamplersSettings", "FLOW_ODE_LABELS", "HIRES_GROUP_WIDGET", "HIRES_SAMPLER_WIDGET",
    "IPNDM_MAX_ORDER", "KEY_ETA", "KEY_FORGE_ETA", "KEY_HIRES_SAMPLER", "KEY_MAX_STAGE", "KEY_NOISE_WINDOW", "KEY_SAMPLER",
    "KIND_CFG_PP", "KIND_DY", "KIND_ER_SDE", "KIND_OTHER", "KIND_RESTART", "KIND_SMEA", "LABELS", "LABEL_CFGPP_UD10_AB",
    "LABEL_DEIS", "LABEL_DPMPP_2M_FLOW_ODE", "LABEL_DPMPP_2M_HEUN_FLOW_ODE", "LABEL_DPMPP_2M_SDE_HEUN",
    "LABEL_DPMPP_3M_FLOW_ODE", "LABEL_DPMPP_4M_SDE", "LABEL_ER_SDE_ODE", "LABEL_ER_SDE_REVERSE_TIME", "LABEL_ER_SDE_TUNABLE",
    "LABEL_EULER_DY", "LABEL_EULER_DY_CFG_PP", "LABEL_EULER_SMEA_DY", "LABEL_EULER_SMEA_DY_CFG_PP", "LABEL_IPNDM",
    "LABEL_IPNDM_V", "LABEL_OF_COMFY", "LABEL_RESTART_FLOW", "LABEL_UNIPC_BH2", "MAIN_SAMPLER_WIDGET", "MAX_STAGE_MAX",
    "MAX_STAGE_MIN", "MAX_STAGE_STEP", "NOISE_PERCENT_MAX", "NOISE_PERCENT_MIN", "NOISE_PERCENT_STEP", "NOISE_WINDOW_OFF",
    "PACK_SAMPLERS", "PROVENANCE_APP_DEFAULT", "PROVENANCE_USER", "PassUse", "REASON_COMFY", "REASON_KREA2",
    "REASON_NOT_NEEDED", "REASON_SEND", "REASON_SEND_AUX", "REASON_SEND_UNCONFIRMED", "RESTART_LABELS", "RESTART_LONG_FROM",
    "RESTART_MIN_STEPS", "RESTART_S_MAX", "RESTART_S_MIN", "RESTART_TIMES", "RESTART_TIMES_LONG", "RESTORE_ETA_FIELDS",
    "RESTORE_STAGE_FIELDS", "SAM3_GROUP_WIDGET",
    "SAM3_MODE_WIDGET", "SAM3_SAMPLER_WIDGET", "SAM3_USE_SAMPLER_WIDGET", "SCRIPT_NAME", "SETTINGS_KEY", "SPEC_PREFIX",
    "STATUS_KEY", "STATUS_SEPARATOR", "SamplerStatus", "TARGET_AUX", "TARGET_I2I", "TARGET_T2I",
    "UD10_HISTORY_WEIGHT", "UD10_UNCOND_HISTORY_WEIGHT", "UD10_ZERO_WEIGHT", "USE_SAME_SAMPLER", "WIDGET_KEYS",
    "WIDGET_PREFIX", "WINDOW_LABELS", "WINDOW_MIN_VERSION", "WINDOW_SKIPPED_STATUS", "WINDOW_UNSUPPORTED_TEXT",
    "XYZ_AXES", "XYZ_PREFIX", "adetailer_pass", "as_block", "coerce_args", "coerce_eta", "coerce_max_stage",
    "coerce_noise_end", "coerce_noise_percent", "coerce_noise_start", "coerce_noise_window", "comfy_name", "describe",
    "format_noise_window", "from_infotext", "infotext_er_sde_samplers", "infotext_items", "is_cfg_pp", "is_default",
    "live_argc",
    "parse_noise_window", "parse_script_block", "parse_settings", "parse_status", "pass_label", "pass_widget_ids",
    "passes_from_widgets", "plan", "resolve_sampler", "restore_from_infotext", "sampler_kind", "script_confirmed",
    "uses_eta", "uses_values",
    "uses_window", "value_sampler_passes", "widget_id", "widget_number", "widget_values", "window_supported",
    "window_unsupported",
]
