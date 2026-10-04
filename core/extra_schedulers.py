# core/extra_schedulers.py
"""Extra Schedulers (sam-extra)(sam-extra ``scripts/anima_extra_schedulers.py``) — custom 식·시그마 목록·Laplace μ/β·
Flow Cosmos ρ/σ̃ 값. 순수 로직(Qt·네트워크·torch 없음).

확장은 Forge "Schedule type" 목록에 스케줄러 9개(cosine·cosine_exponential·phi·laplace·karras_dynamic·custom·
react_cosinusoidal_dynsf·flow_cosmos_rho7·flow_cosmos_dynamic — 마지막 셋은 v0.33.0)를 등록하고,
always-on 스크립트 ``alwayson_scripts["Extra Schedulers (sam-extra)"]`` 로 그중 **custom·Laplace·Flow Cosmos rho7·Flow
Cosmos Dynamic 이 읽는 값 8개**를 받는다(위치 인자 ``[custom_mode, custom_expression, custom_sigmas, laplace_mu, laplace_beta, flow_cosmos_rho,
flow_cosmos_sigma_max, flow_cosmos_sigma_min]`` 또는 **dict 한 개** — sam3ext/ui_extra_schedulers.py ARG_NAMES·coerce_args.
뒤의 셋은 v0.33.0 에서 뒤에 덧붙었다). 앱은 키 이름 dict 로 보낸다 — Forge 가 캐시한 위치 기본값·사용자
``ui-config.json`` 이 바꾼 슬라이더 기본값과 무관하게 같은 뜻이 된다:
``{"args": [{"custom_mode": "expression", "custom_expression": "M * (m / M) ** x",
"custom_sigmas": "[1.0, 0.6, 0.25, 0.1, 0.0]", "laplace_mu": 0.0, "laplace_beta": 0.5, "flow_cosmos_rho": 7.0,
"flow_cosmos_sigma_max": 80.0, "flow_cosmos_sigma_min": 0.002}]}``.

- **늘 키 8개를 보낸다**(계획 X10 과 같은 규칙 — Extra Samplers 의 인자 다섯처럼). Forge API 는 dict 한 개를 그대로 넘기고
  sam-extra 0.31–0.32 의 ``settings.coerce`` 는 아는 키 다섯만 읽어(``raw.get``) 뒤의 셋을 조용히 버린다. 그 판에는
  flow_cosmos_rho7·flow_cosmos_dynamic 자체가 없어 그 값을 읽을 스케줄러도 없으므로 알림이 필요 없다(앱 Forge 스케줄러 콤보는 라이브 이름
  목록이고, 붙여 넣은 'Flow Cosmos rho7' 은 그런 Forge 콤보에서 고르지 못한다 — core/preset_choice). v0.33.0 이 옛 블록
  (키 다섯·위치 인자 다섯 이하·블록 없음)을 받으면 빠진 셋은 기본값 ρ 7 · σ̃ 0.002~80 — v0.33.0 D18 의 고정 목록(=
  GPU A/B 의 custom 식)과 비트까지 같다.
- **생성마다 전역 상태 하나다.** 확장 ``process()`` 가 요청마다 값을 전역 상태에 쓰고(sam3ext/extra_schedulers/settings.py
  set_active) 스케줄러가 그것을 읽는다 — 블록이 없는 요청도 ui() 기본값으로 덮는다. 그래서 블록은 메인 요청과 보조
  img2img(Refine·SAM3·ADetailer·손 재구성)에 같이 간다(core/alwayson_propagation 의 행). 값은 그 패스의 스케줄러가
  custom·Laplace·Flow Cosmos rho7·Flow Cosmos Dynamic 일 때만 쓰이므로 늘 보내도 결과가 같다.
- **Flow Cosmos rho7 의 값 셋**(v0.33.0 D19 — Laplace μ/β 와 같은 방식): ρ(기본 7, 1~15 step 0.1), σ̃ max(80, 1~1000 step
  0.1), σ̃ min(0.002, 0.0001~1 step 0.0001). σ̃ 는 VE 단위 잡음 크기다. flow 모델(예측기 prediction_type 'const')에서는
  s = (σ̃max^(1/ρ) + p(σ̃min^(1/ρ) − σ̃max^(1/ρ)))^ρ, t = s/(1+s) 의 목록(p = i/(n−1) — 모델 σ 범위·Settings 값은 쓰지
  않는다), flow 가 아닌 모델(eps/v)에서는 Forge 가 넘기는 σ 범위에 Forge 의 Karras 를 이 ρ 로 — σ̃ 범위는 쓰지 않는다.
  범위 밖은 자르고 유한한 숫자가 아니면 기본값(Laplace 와 같은 ``coerce_float``). flow 모델에서 σ̃ min ≥ σ̃ max(슬라이더
  안에서는 둘 다 1 일 때만)면 생성이 'Extra Schedulers: …' 로 멈춘다(``flow_cosmos_range_ok`` — 앱은 막지 않는다).
  σ̃ 범위를 k 배 하면 flow 시간 shift k 와 같다(t' = k·t/(1+(k−1)·t) ⇔ σ̃' = k·σ̃) — Anima 의 shift 3 은 σ̃ 범위 ×3
  (0.006~240)에 해당한다(수식상의 사실이지 화질 주장이 아니다).
- **Flow Cosmos Dynamic**(v0.33.0 D20)은 같은 세 값을 읽는다(따로 값 없음): flow 모델에서는 Karras Dynamic 의 스텝마다 지수
  ρᵢ = ρ + 2cos(2πi/n) 를 σ̃ 범위에 깔아 flow 시간으로, eps/v 에서는 Forge 가 넘기는 σ 범위에 Karras Dynamic 을 이 ρ 로.
  ρ ≤ 2 면 두 갈래 모두 생성이 멈추고(``flow_cosmos_dynamic_rho_ok``), ρ 가 약 4 보다 작으면 스텝 수에 따라 시그마가 다시
  올라가 멈출 수 있다(확장이 목록을 만든 뒤 검사 — 앱은 미리 계산하지 않는다).
- **언제 싣나**(계획 X5 — Extra Samplers 와 같은 규칙, ``plan``): Forge 기능 스냅샷이 스크립트를 확인했으면 늘(XYZ
  Scheduler 축이 Laplace 로 바꾸는 칸·사용자 ui-config 기본값과 무관하게), 확인 못 했으면(모름·없음) 어느 패스가
  laplace·custom·flow_cosmos_rho7·flow_cosmos_dynamic 을 쓸 때만 — 그때는 core/alwayson_propagation 게이트가 출처(값이 기본값이면 앱
  기본값)로 보낼지 정하고 빠지면 알린다. ComfyUI 는 늘 싣는다(계획 C6 — 기능 스냅샷·게이트가 없고, 컴파일러가 값
  스케줄러를 쓰는 패스가 있을 때만 앱 노드 팩 1.7.0 의 ``ForgeNeoExtraSchedulers`` 를 넣는다 — 없으면 그래프가 바이트
  단위로 그대로). Krea2 는 보내지 않는다.
- 붙여 넣기·'이 설정으로 생성'(``from_infotext``)은 확장 paste_* 규칙의 거울이다. 식·목록의 미리 검사(``check_text`` —
  계획 C7)는 계산기를 다시 쓰지 않고 ComfyUI 팩의 순수 파이썬 ``extra_schedulers_math`` 를 지연 import 해 부른다(팩 패리티
  테스트가 설치된 확장과 같은 거절·같은 영어 문구를 지킨다). σ·스텝 수와 무관한 검사만 한다 — 그 밖은 생성할 때 Forge·
  ComfyUI 가 같은 영어 문구로 멈춘다.

확장 코드는 GPL-3.0 이라 옮기지 않았다 — 규칙(범위·별칭·자르기·붙여 넣기)을 다시 구현했다. ``tests/test_extra_schedulers_core.py``
가 설치된 확장(settings.coerce·ui_extra_schedulers.coerce_args·build_controls·paste_*·infotext_items·registry·schedulers)과
같은 입력으로 비교하고, 레지스트리 SEMANTIC_PINS(xsched_*)가 공유 상수를 지킨다.
"""
from __future__ import annotations

import importlib
import logging
import math
from dataclasses import dataclass, replace
from types import MappingProxyType
from typing import Any, Iterable, Mapping, NamedTuple, Optional

from core import sam3_args

logger = logging.getLogger(__name__)

# ── 확장 계약 ─────────────────────────────────────────────────────────────────
SCRIPT_NAME = "Extra Schedulers (sam-extra)"   # sam3ext/ui_extra_schedulers.py TITLE
# sam3ext/ui_extra_schedulers.py ARG_NAMES — 뒤에만 덧붙는다(v0.33.0: 5 → 8, Flow Cosmos 셋)
ARG_NAMES = ("custom_mode", "custom_expression", "custom_sigmas", "laplace_mu", "laplace_beta",
             "flow_cosmos_rho", "flow_cosmos_sigma_max", "flow_cosmos_sigma_min")

MODE_EXPRESSION, MODE_SIGMAS = "expression", "sigmas"            # sam3ext/extra_schedulers/settings.py
MODES = (MODE_EXPRESSION, MODE_SIGMAS)
# 확장 라디오 라벨(script-info 선택지 — 'Expression'·'Sigma list'). coerce_mode 가 둘 다 읽는다.
MODE_LABELS = MappingProxyType({MODE_EXPRESSION: "Expression", MODE_SIGMAS: "Sigma list"})
_SIGMAS_MODE_WORDS = frozenset({"sigmas", "sigma list", "sigma_list", "list", "sigma"})   # settings.coerce_mode

DEFAULT_EXPRESSION = "M * (m / M) ** x"        # settings.DEFAULT_EXPRESSION — Exponential 과 같은 곡선
DEFAULT_SIGMAS = "[1.0, 0.6, 0.25, 0.1, 0.0]"  # settings.DEFAULT_SIGMAS — 1.0…0.0 이면 σmax~σmin 사이로 늘린다
LAPLACE_MU_MIN, LAPLACE_MU_MAX, LAPLACE_MU_DEFAULT = -10.0, 10.0, 0.0       # settings.py (ComfyUI LaplaceScheduler)
LAPLACE_BETA_MIN, LAPLACE_BETA_MAX, LAPLACE_BETA_DEFAULT = 0.0, 10.0, 0.5   # settings.py
LAPLACE_MU_STEP = LAPLACE_BETA_STEP = 0.1      # ui_extra_schedulers.build_controls 슬라이더 step(script-info)
# Flow Cosmos rho7 의 값 셋(v0.33.0 D19 — settings.py FLOW_COSMOS_*: 최소·최대·기본·슬라이더 step). 범위는 확장이 정한 것,
# 기본값은 아래 schedulers.py 리터럴 FLOW_COSMOS_RHO·SIGMA_MAX·SIGMA_MIN(Cosmos-Predict2 의 ρ 7 · 0.002~80)과 같은 수다.
FLOW_COSMOS_RHO_MIN, FLOW_COSMOS_RHO_MAX, FLOW_COSMOS_RHO_DEFAULT, FLOW_COSMOS_RHO_STEP = 1.0, 15.0, 7.0, 0.1
FLOW_COSMOS_SIGMA_MAX_MIN, FLOW_COSMOS_SIGMA_MAX_MAX, FLOW_COSMOS_SIGMA_MAX_DEFAULT, FLOW_COSMOS_SIGMA_MAX_STEP = (
    1.0, 1000.0, 80.0, 0.1)
FLOW_COSMOS_SIGMA_MIN_MIN, FLOW_COSMOS_SIGMA_MIN_MAX, FLOW_COSMOS_SIGMA_MIN_DEFAULT, FLOW_COSMOS_SIGMA_MIN_STEP = (
    0.0001, 1.0, 0.002, 0.0001)

# infotext 키(sam3ext/ui_extra_schedulers.py) — 이미지가 이 확장의 custom·Laplace·Flow Cosmos(rho7·Dynamic)를 썼을 때만 남는다
KEY_CUSTOM_EXPRESSION = "Custom scheduler expression"   # 공백을 한 칸으로 줄인 식
KEY_CUSTOM_SIGMAS = "Custom scheduler sigmas"           # 공백을 한 칸으로 줄인 목록
KEY_LAPLACE_MU = "Laplace mu"                           # repr(float)
KEY_LAPLACE_BETA = "Laplace beta"                       # repr(float)
KEY_FLOW_COSMOS_RHO = "Flow Cosmos rho"                 # repr(float) — 셋 다 늘 함께(eps/v 에서 돈 때도)
KEY_FLOW_COSMOS_SIGMA_MAX = "Flow Cosmos sigma max"     # repr(float) — σ̃ max
KEY_FLOW_COSMOS_SIGMA_MIN = "Flow Cosmos sigma min"     # repr(float) — σ̃ min
INFOTEXT_KEYS = (KEY_CUSTOM_EXPRESSION, KEY_CUSTOM_SIGMAS, KEY_LAPLACE_MU, KEY_LAPLACE_BETA,
                 KEY_FLOW_COSMOS_RHO, KEY_FLOW_COSMOS_SIGMA_MAX, KEY_FLOW_COSMOS_SIGMA_MIN)
KEY_SCHEDULE_TYPE, KEY_HIRES_SCHEDULE_TYPE = "Schedule type", "Hires schedule type"   # Forge 표준 키(라벨이 적힌다)
USE_SAME_SCHEDULER = "Use same scheduler"   # Forge Hires·ADetailer·SAM3 의 '같은 스케줄러'

# 실패 문구의 머리(sam3ext/extra_schedulers/schedulers.py — Forge HTTP 500·ComfyUI 노드 오류에 그대로 나온다)
ERROR_PREFIX = "Extra Schedulers:"

# ── 스케줄러 표(sam3ext/extra_schedulers/registry.py SCHEDULER_SPECS·README_LABELS) ─────────────────
# name = Forge API·앱 Forge 콤보 값(ComfyUI 팩도 같은 이름으로 등록), label = infotext 'Schedule type',
# README 별칭 = aoleg/Neo_ExtraSchedulers README 의 표기(소문자) — 확장이 schedulers_map 에 같이 넣는다.
SCHEDULER_NAMES = ("cosine", "cosine_exponential", "phi", "laplace", "karras_dynamic", "custom",
                   "react_cosinusoidal_dynsf", "flow_cosmos_rho7", "flow_cosmos_dynamic")
SCHEDULER_LABELS = MappingProxyType({
    "cosine": "Cosine", "cosine_exponential": "CosineExponential blend", "phi": "Phi", "laplace": "Laplace",
    "karras_dynamic": "Karras Dynamic", "custom": "custom", "react_cosinusoidal_dynsf": "React Cosinusoidal DynSF",
    "flow_cosmos_rho7": "Flow Cosmos rho7", "flow_cosmos_dynamic": "Flow Cosmos Dynamic",
})
README_LABELS = MappingProxyType({   # aoleg 의 여섯만 — React DynSF·Flow Cosmos rho7·Dynamic 은 README 별칭이 없다
    "cosine": "cosine", "cosine_exponential": "cosine-exponential blend", "phi": "phi", "laplace": "laplace",
    "karras_dynamic": "karras dynamic", "custom": "custom",
})
LABEL_LAPLACE = SCHEDULER_LABELS["laplace"]   # registry.LABEL_LAPLACE
LABEL_CUSTOM = SCHEDULER_LABELS["custom"]     # registry.LABEL_CUSTOM
NAME_LAPLACE, NAME_CUSTOM = "laplace", "custom"
# React Cosinusoidal DynSF(v0.33.0) — reForge 의 표기(그 이미지의 'Schedule type' 이 그대로 붙는다). 값이 없다(계수 고정).
NAME_REACT_DYNSF = "react_cosinusoidal_dynsf"
LABEL_REACT_DYNSF = SCHEDULER_LABELS[NAME_REACT_DYNSF]   # registry.LABEL_REACT_DYNSF
REACT_DYNSF_FACTOR = 2.15   # schedulers.REACT_DYNSF_FACTOR — 다른 계수는 custom 식 M*((m+(M-m)*cos(x*pi/2))/M)**(f*x)
# Flow Cosmos rho7(v0.33.0) — 이름은 KeithZ117/Comfyui-anima-sampler(effba3c5)의 스케줄 선택지(이름·착상만, 확장은 식에서
# 새로 썼다). flow 모델(예측기 prediction_type 'const')에서는 VE 단위 σ̃ 범위에 Karras ρ 를 깔고 t = s/(1+s) 로 바꾼 목록
# (모델의 σ 범위·Settings 값은 쓰지 않는다), flow 가 아닌 모델(eps/v)에서는 Forge 가 넘기는 σ 범위에 Forge 의 Karras 를
# 이 ρ 로. ρ·σ̃ max·σ̃ min 은 이 블록의 값(D19 — 위 FLOW_COSMOS_*_DEFAULT; Settings → rho·붙여 넣은 'Schedule rho' 는
# 닿지 않는다). 기본값은 Anima 의 부모 모델 Cosmos-Predict2 의 잡음 범위(EDM 0.002–80)·ρ 7 — 80/81 ≈ 0.988 에서 시작한다.
NAME_FLOW_COSMOS_RHO7 = "flow_cosmos_rho7"
LABEL_FLOW_COSMOS_RHO7 = SCHEDULER_LABELS[NAME_FLOW_COSMOS_RHO7]   # registry.LABEL_FLOW_COSMOS_RHO7
FLOW_COSMOS_SIGMA_MAX = 80.0    # schedulers.FLOW_COSMOS_SIGMA_MAX — Cosmos-Predict2 의 EDM σ 범위(ComfyUI supported_models 의 Cosmos 와 같다)
FLOW_COSMOS_SIGMA_MIN = 0.002   # schedulers.FLOW_COSMOS_SIGMA_MIN
FLOW_COSMOS_RHO = 7.0           # schedulers.FLOW_COSMOS_RHO — Karras et al. 2022 의 ρ
# Flow Cosmos Dynamic(v0.33.0 D20) — 확장 자신의 조합: Karras Dynamic 의 스텝마다 ρᵢ = ρ + 2cos(2πi/n) 를 Flow Cosmos rho7 의
# σ̃ 범위에(flow 모델), eps/v 에서는 Karras Dynamic 그대로(Forge 가 넘기는 σ 범위). 값은 Flow Cosmos rho7 의 세 칸을 같이 쓴다
# (따로 값 없음 — 확장 registry.FLOW_COSMOS_LABELS). ρ 는 2 보다 커야 한다(_check_karras_dynamic_rho — 지수가 ρ − 2 까지
# 내려간다). 스케줄러 힌트·README 별칭이 없다.
NAME_FLOW_COSMOS_DYNAMIC = "flow_cosmos_dynamic"
LABEL_FLOW_COSMOS_DYNAMIC = SCHEDULER_LABELS[NAME_FLOW_COSMOS_DYNAMIC]   # registry.LABEL_FLOW_COSMOS_DYNAMIC
FLOW_COSMOS_DYNAMIC_RHO_ABOVE = 2.0   # schedulers._check_karras_dynamic_rho — ρ 가 이 값보다 커야 한다
# Flow Cosmos 의 세 칸(ρ·σ̃ max·σ̃ min)을 읽는 스케줄러(registry.FLOW_COSMOS_LABELS 의 이름 — 같은 순서)
FLOW_COSMOS_SCHEDULERS = (NAME_FLOW_COSMOS_RHO7, NAME_FLOW_COSMOS_DYNAMIC)
FLOW_COSMOS_LABELS = tuple(SCHEDULER_LABELS[name] for name in FLOW_COSMOS_SCHEDULERS)
# 이 블록의 값을 읽는 스케줄러(custom 식·목록, Laplace μ/β, Flow Cosmos ρ/σ̃ — v0.33.0 D19 부터 셋, D20 의 Dynamic 이 넷째)
VALUE_SCHEDULERS = (NAME_LAPLACE, NAME_CUSTOM, *FLOW_COSMOS_SCHEDULERS)
# flow 모델(Anima 등)에서 뿌옇게 나오는 SD·SDXL 용 스케줄러(확장 HELP) — 카드 힌트만, 막지 않는다. React DynSF 는
# σ 범위식이지만 이 목록이 아니다(높은 σ 쪽에 스텝이 몰린다 — 카드가 따로 알린다, 계획 A-XS5/6). Flow Cosmos rho7·Dynamic 도
# 아니다(flow 용 — eps/v 에서는 Forge 의 Karras·Karras Dynamic 과 같다)
FLOW_UNSUITABLE = ("cosine", "cosine_exponential", "phi", "karras_dynamic")

# ── ComfyUI 계약(노드 팩 1.7.0 — 계획 PB-XS4·C1·C2a) ────────────────────────────────────
# 팩(comfy_custom_nodes/ai_studio_forge_parity/extra_schedulers.py)은 SCHEDULER_NAMES 아홉을 같은 이름으로 ComfyUI 스케줄러
# 표에 넣고(남의 것은 덮지 않는다), 이 노드가 MODEL 의 model_sampling 사본에 값 여덟을 싣는다. 노드의 선택 입력
# ``owned_names`` 의 선택지 = 그 ComfyUI 에서 팩이 가진 이름 — 컴파일러가 /object_info 에서 읽어 남의 이름이면 거절한다(R6).
COMFY_NODE_CLASS = "ForgeNeoExtraSchedulers"
COMFY_INPUTS = ("model", *ARG_NAMES)          # 필수 입력(팩 1.7.0 — 여덟 값 모두 필수, 범위 밖은 ComfyUI 가 거절)
COMFY_REPORT_INPUT = "owned_names"           # compat.owned_names_input — 팩이 가진 이름(R6)
COMFY_MIN_PACK_VERSION = "1.7.0"

# ── 위젯·저장 ───────────────────────────────────────────────────────────────
WIDGET_KEYS = ARG_NAMES
WIDGET_PREFIX = "_xsched_"
SETTINGS_KEY = "extra_schedulers_settings"   # prompt_settings.json·생성 프리셋 키


def widget_id(key: str) -> str:
    """위젯 프록시 id = Vue ``storeWidgets`` 키(frontend/src/utils/extraSchedulers.ts WIDGET_IDS) — 접두 + 확장 인자 이름."""
    if key not in WIDGET_KEYS:
        raise KeyError(key)
    return f"{WIDGET_PREFIX}{key}"


# ── 정규화 (확장 settings.py 규칙의 거울) ───────────────────────────────────────────
def coerce_float(value: Any, default: float, low: float, high: float) -> float:
    """``value`` 를 [low, high] 로 자른 float. bool·숫자가 아닌 것·NaN·±inf 는 ``default`` (settings.coerce_float).

    float 로 바꿀 수 없을 만큼 큰 int(JSON 이 만들 수 있다)도 ``default`` — v0.33.0 확장과 같다(D19 검토 K1). 0.32.x
    확장은 그 값에서 OverflowError 가 나 process() 가 요청 전체를 기본값으로 돌렸다(앱은 그 칸만 기본값). 자르기는
    확장처럼 max 다음 min 이라 경계의 -0.0 은 그 경계 값이 된다."""
    if isinstance(value, bool):
        return default
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        return default
    if not math.isfinite(number):
        return default
    return min(high, max(low, number))


def coerce_laplace_mu(value: Any) -> float:
    return coerce_float(value, LAPLACE_MU_DEFAULT, LAPLACE_MU_MIN, LAPLACE_MU_MAX)


def coerce_laplace_beta(value: Any) -> float:
    return coerce_float(value, LAPLACE_BETA_DEFAULT, LAPLACE_BETA_MIN, LAPLACE_BETA_MAX)


def coerce_flow_cosmos_rho(value: Any) -> float:
    return coerce_float(value, FLOW_COSMOS_RHO_DEFAULT, FLOW_COSMOS_RHO_MIN, FLOW_COSMOS_RHO_MAX)


def coerce_flow_cosmos_sigma_max(value: Any) -> float:
    return coerce_float(value, FLOW_COSMOS_SIGMA_MAX_DEFAULT, FLOW_COSMOS_SIGMA_MAX_MIN, FLOW_COSMOS_SIGMA_MAX_MAX)


def coerce_flow_cosmos_sigma_min(value: Any) -> float:
    return coerce_float(value, FLOW_COSMOS_SIGMA_MIN_DEFAULT, FLOW_COSMOS_SIGMA_MIN_MIN, FLOW_COSMOS_SIGMA_MIN_MAX)


def coerce_mode(value: Any) -> str:
    """'sigmas' — 저장 이름이나 시그마 목록 라벨('Sigma list'·'sigma_list'·'list'·'sigma'), 그 밖은 'expression'."""
    return MODE_SIGMAS if str(value or "").strip().lower() in _SIGMAS_MODE_WORDS else MODE_EXPRESSION


def coerce_text(value: Any, default: str) -> str:
    """앞뒤 공백만 뗀 글자(줄이지 않는다 — 너무 긴 식은 생성할 때 확장이 거절한다). None 이면 ``default``."""
    return default if value is None else str(value).strip()


# ── 설정 ──────────────────────────────────────────────────────────────────────
@dataclass(frozen=True)
class ExtraSchedulerSettings:
    """한 생성의 값 여덟(확장 settings.ExtraSchedulerSettings 와 같은 칸·기본값 — 뒤 셋은 v0.33.0 Flow Cosmos)."""

    custom_mode: str = MODE_EXPRESSION
    custom_expression: str = DEFAULT_EXPRESSION
    custom_sigmas: str = DEFAULT_SIGMAS
    laplace_mu: float = LAPLACE_MU_DEFAULT
    laplace_beta: float = LAPLACE_BETA_DEFAULT
    flow_cosmos_rho: float = FLOW_COSMOS_RHO_DEFAULT
    flow_cosmos_sigma_max: float = FLOW_COSMOS_SIGMA_MAX_DEFAULT
    flow_cosmos_sigma_min: float = FLOW_COSMOS_SIGMA_MIN_DEFAULT

    def as_arg(self) -> dict:
        """확장 dict 인자(키 8개, ARG_NAMES 순서, 숫자는 float)."""
        return {"custom_mode": self.custom_mode, "custom_expression": self.custom_expression,
                "custom_sigmas": self.custom_sigmas, "laplace_mu": float(self.laplace_mu),
                "laplace_beta": float(self.laplace_beta), "flow_cosmos_rho": float(self.flow_cosmos_rho),
                "flow_cosmos_sigma_max": float(self.flow_cosmos_sigma_max),
                "flow_cosmos_sigma_min": float(self.flow_cosmos_sigma_min)}


# 확장 API 기본값 = ui() 기본값(픽스처 tests/fixtures/sam_extra_script_info.json t2i·i2i 가 같다). 앱 기본값도 같다 —
# 블록이 없던 지금까지의 Forge 결과와 같다.
EXTENSION_DEFAULTS = ExtraSchedulerSettings()
APP_DEFAULTS = EXTENSION_DEFAULTS
DEFAULTS = EXTENSION_DEFAULTS


def coerce(raw: Optional[Mapping]) -> ExtraSchedulerSettings:
    """키가 빠졌거나 모양이 틀린 dict → 설정. 예외를 내지 않는다(settings.coerce). 모르는 키는 버린다."""
    values = dict(raw) if isinstance(raw, Mapping) else {}
    return ExtraSchedulerSettings(
        custom_mode=coerce_mode(values.get("custom_mode", MODE_EXPRESSION)),
        custom_expression=coerce_text(values.get("custom_expression"), DEFAULT_EXPRESSION),
        custom_sigmas=coerce_text(values.get("custom_sigmas"), DEFAULT_SIGMAS),
        laplace_mu=coerce_laplace_mu(values.get("laplace_mu", LAPLACE_MU_DEFAULT)),
        laplace_beta=coerce_laplace_beta(values.get("laplace_beta", LAPLACE_BETA_DEFAULT)),
        flow_cosmos_rho=coerce_flow_cosmos_rho(values.get("flow_cosmos_rho", FLOW_COSMOS_RHO_DEFAULT)),
        flow_cosmos_sigma_max=coerce_flow_cosmos_sigma_max(
            values.get("flow_cosmos_sigma_max", FLOW_COSMOS_SIGMA_MAX_DEFAULT)),
        flow_cosmos_sigma_min=coerce_flow_cosmos_sigma_min(
            values.get("flow_cosmos_sigma_min", FLOW_COSMOS_SIGMA_MIN_DEFAULT)),
    )


def coerce_args(args: Any) -> ExtraSchedulerSettings:
    """스크립트 인자(ARG_NAMES 순서의 위치 인자 또는 dict 한 개) → 설정(ui_extra_schedulers.coerce_args). 옛 위치 인자
    다섯(이하)·키 다섯의 dict 는 뒤 셋(Flow Cosmos)이 기본값이다."""
    items = list(args) if isinstance(args, (list, tuple)) else []
    if items and isinstance(items[0], dict):
        return coerce(items[0])
    return coerce(dict(zip(ARG_NAMES, items)))


_NUMBER_COERCERS = {"laplace_mu": coerce_laplace_mu, "laplace_beta": coerce_laplace_beta,
                    "flow_cosmos_rho": coerce_flow_cosmos_rho, "flow_cosmos_sigma_max": coerce_flow_cosmos_sigma_max,
                    "flow_cosmos_sigma_min": coerce_flow_cosmos_sigma_min}


def _readable_number(value: Any) -> bool:
    if isinstance(value, bool) or value is None:
        return False
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError, OverflowError):
        return False


def parse_settings(raw: Optional[Mapping], *, base: ExtraSchedulerSettings = APP_DEFAULTS) -> ExtraSchedulerSettings:
    """위젯 문자열·저장 dict → 설정. 없거나(None) 읽을 수 없는 칸은 ``base`` 칸을 쓴다(키 다섯의 옛 저장값 → Flow
    Cosmos 셋은 ``base``).

    ``base`` 가 기본값이면 ``coerce`` 와 같다(확장 규칙). 빈 식·빈 목록은 값이다(그대로 보내고 생성할 때 확장이 거절)."""
    if not isinstance(raw, Mapping):
        return base
    values: dict = {}
    if raw.get("custom_mode") is not None:
        values["custom_mode"] = coerce_mode(raw.get("custom_mode"))
    for key in ("custom_expression", "custom_sigmas"):
        if raw.get(key) is not None:
            values[key] = coerce_text(raw.get(key), getattr(base, key))
    for key, coerce_number in _NUMBER_COERCERS.items():
        if _readable_number(raw.get(key)):
            values[key] = coerce_number(raw.get(key))
    return replace(base, **values)


def format_number(value: Any) -> str:
    """infotext 숫자 — 같은 float 로 되읽히는 가장 짧은 글자(ui_extra_schedulers.format_number = ``repr(float)``)."""
    return repr(float(value))


def widget_number(value: Any) -> str:
    """위젯 숫자 문자열 — 0.5 → '0.5', 0.0 → '0', 2.0 → '2', 1/3 → '0.3333333333333333'(값을 잃지 않는다)."""
    number = float(value)
    if number == 0:
        return "0"
    text = repr(number)
    return text[:-2] if text.endswith(".0") else text


def widget_values(settings: ExtraSchedulerSettings) -> dict[str, str]:
    """설정 → 위젯 문자열(키 8개 — 모드 키, 식·목록 그대로, 숫자는 ``widget_number``)."""
    return {
        "custom_mode": settings.custom_mode,
        "custom_expression": settings.custom_expression,
        "custom_sigmas": settings.custom_sigmas,
        "laplace_mu": widget_number(settings.laplace_mu),
        "laplace_beta": widget_number(settings.laplace_beta),
        "flow_cosmos_rho": widget_number(settings.flow_cosmos_rho),
        "flow_cosmos_sigma_max": widget_number(settings.flow_cosmos_sigma_max),
        "flow_cosmos_sigma_min": widget_number(settings.flow_cosmos_sigma_min),
    }


def is_default(settings: ExtraSchedulerSettings) -> bool:
    """확장 기본값과 같은 값인가(블록 출처를 앱 기본값으로 적는다)."""
    return settings == EXTENSION_DEFAULTS


def flow_cosmos_range_ok(settings: ExtraSchedulerSettings) -> bool:
    """flow 모델에서 Flow Cosmos rho7 이 이 값으로 목록을 만들 수 있나 — 확장 schedulers._check_flow_cosmos_rho·
    _check_flow_cosmos_range 와 같은 조건(ρ 유한하고 0 초과, 0 < σ̃ min < σ̃ max 둘 다 유한). 거짓이면 flow 모델 생성이
    'Extra Schedulers: Flow Cosmos rho7 needs …' 로 멈춘다(eps/v 는 σ̃ 를 쓰지 않아 ρ 만 본다 — 정규화된 ρ 는 늘 1~15).
    슬라이더 범위 안에서는 σ̃ min = σ̃ max = 1 일 때만 거짓이다. 앱은 막지 않고 보낸다 — 카드 경고 줄이 쓴다(계획 A-XS6)."""
    try:
        rho = float(settings.flow_cosmos_rho)
        low, high = float(settings.flow_cosmos_sigma_min), float(settings.flow_cosmos_sigma_max)
    except (TypeError, ValueError, OverflowError):
        return False
    return math.isfinite(rho) and rho > 0.0 and math.isfinite(low) and math.isfinite(high) and 0.0 < low < high


def flow_cosmos_dynamic_rho_ok(settings: ExtraSchedulerSettings) -> bool:
    """Flow Cosmos Dynamic 이 이 ρ 를 받나 — 확장 schedulers._check_karras_dynamic_rho 와 같은 조건(ρ > 2, NaN 은 거짓 —
    flow·eps/v 두 갈래 모두 목록을 만들기 전에 본다). 거짓이면 그 스케줄러를 쓴 생성이 'Extra Schedulers: Flow Cosmos Dynamic
    needs rho above 2 …' 로 멈춘다. 슬라이더(1~15)로는 ρ 1~2 가 그렇다. ρ 가 2 보다 커도 약 4 보다 작으면 스텝 수·σ̃ 범위에
    따라 시그마가 다시 올라가 멈출 수 있다(_check_karras_dynamic_falls — 목록을 만든 뒤라 여기서는 보지 않는다). 앱은 막지
    않고 보낸다 — 카드 경고 줄이 쓴다."""
    try:
        rho = float(settings.flow_cosmos_rho)
    except (TypeError, ValueError, OverflowError):
        return False
    return rho > FLOW_COSMOS_DYNAMIC_RHO_ABOVE


# ── custom 식·시그마 목록 미리 검사(계획 C7) ─────────────────────────────────────────
# 계산기는 ComfyUI 팩의 순수 파이썬 모듈(torch·ComfyUI 없음 — ui/relight_actions 가 팩 relight 를 부르는 것과 같은 지연
# import). 팩이 없거나 읽을 수 없으면 검사하지 않는다(None — 생성은 확장이 판정한다).
PACK_MATH_MODULE = "comfy_custom_nodes.ai_studio_forge_parity.extra_schedulers_math"
ERROR_PROPERTY = "error"     # 칸 아래 오류 줄의 위젯 속성(frontend/src/utils/extraSchedulers.ts ERROR_PROPERTY)
# 시그마 목록의 σ 와 무관한 검사만: 정규화 목록(1.0…0.0)은 어떤 양수 범위로 늘려도 모두 0 보다 크다
_CHECK_SIGMA_RANGE = (1.0, 2.0)


def _pack_math() -> Any:
    try:
        return importlib.import_module(PACK_MATH_MODULE)
    except Exception as exc:   # 팩이 없는 배포·깨진 파일 — 미리 검사만 빠진다
        logger.debug("Extra Schedulers 미리 검사 생략 — %s 를 불러오지 못했다: %s", PACK_MATH_MODULE, exc)
        return None


def _no_interp(_values: Any, _count: int) -> Any:   # 스텝 0 이라 부르지 않는다
    raise AssertionError("interpolation is not used by the text check")


def check_text(mode: Any, text: Any) -> Optional[str]:
    """custom 식(``mode`` expression)·시그마 목록(sigmas) 글자를 확장과 같은 규칙으로 검사 — 쓸 수 없으면 확장의 영어
    오류 원문(Forge 'Extra Schedulers: …'·ComfyUI 노드 오류의 그 부분과 같은 문구), 쓸 수 있으면 None.

    ``text`` 는 보낼 값과 같게 읽는다(``coerce_text`` — 앞뒤 공백만 뗀다, 빈 글자는 빈 식). σ·스텝 수에 따라 달라지는 검사
    (식의 값이 유한하고 0 보다 큰가, 평평한 스텝, 보간)는 하지 않는다 — 생성할 때 확장·팩이 같은 문구로 멈춘다."""
    module = _pack_math()
    if module is None:
        return None
    sigmas = coerce_mode(mode) == MODE_SIGMAS
    value = coerce_text(text, DEFAULT_SIGMAS if sigmas else DEFAULT_EXPRESSION)
    try:
        if sigmas:
            module.sigmas_from_list(value, 0, *_CHECK_SIGMA_RANGE, interp=_no_interp)
        else:
            module.compile_expression(value)
    except ValueError as exc:      # ExpressionError·SigmaListError
        return str(exc)
    return None


def check_settings(settings: ExtraSchedulerSettings) -> Optional[str]:
    """그 설정의 custom 방식이 쓰는 글자(식 모드면 식, 목록 모드면 시그마 목록)의 ``check_text``."""
    sigmas = settings.custom_mode == MODE_SIGMAS
    return check_text(settings.custom_mode, settings.custom_sigmas if sigmas else settings.custom_expression)


def as_block(settings: ExtraSchedulerSettings) -> dict:
    """alwayson 블록 — dict 형식 인자 하나(키 8개 — 늘 8개, 옛 sam-extra 는 뒤 셋을 버린다)."""
    return {"args": [settings.as_arg()]}


def parse_script_block(block: Any) -> Optional[ExtraSchedulerSettings]:
    """보낸 alwayson 블록 → 확장이 읽는 설정(coerce_args 규칙 — 위치 인자 8·5개 이하·dict, 빠진 뒤 인자는 기본값). 블록이
    아니면 None."""
    if not isinstance(block, Mapping):
        return None
    return coerce_args(block.get("args"))


def normalize_text(text: Any) -> str:
    """infotext 의 식·목록 — 공백을 한 칸으로(ui_extra_schedulers.normalize_text)."""
    return " ".join(str(text or "").split())


def describe(settings: ExtraSchedulerSettings, schedulers: Optional[Iterable[Any]] = None) -> str:
    """로그·알림용 한 줄 — "custom 식 'M * (m / M) ** x' · Laplace μ 0 · β 0.5 · Flow Cosmos ρ 7 · σ̃ 0.002~80".
    ``schedulers`` (이름·라벨)를 주면 그 스케줄러가 읽는 칸만, 없으면 모두(Flow Cosmos 칸은 rho7·Dynamic 이 같이 쓴다)."""
    names = None if schedulers is None else {resolve_scheduler(item) for item in schedulers}
    parts = []
    if names is None or NAME_CUSTOM in names:
        if settings.custom_mode == MODE_SIGMAS:
            parts.append(f"custom 목록 {normalize_text(settings.custom_sigmas)}")
        else:
            parts.append(f"custom 식 '{normalize_text(settings.custom_expression)}'")
    if names is None or NAME_LAPLACE in names:
        parts.append(f"Laplace μ {widget_number(settings.laplace_mu)} · β {widget_number(settings.laplace_beta)}")
    if names is None or not names.isdisjoint(FLOW_COSMOS_SCHEDULERS):
        parts.append(f"Flow Cosmos ρ {widget_number(settings.flow_cosmos_rho)} · σ̃ "
                     f"{widget_number(settings.flow_cosmos_sigma_min)}~{widget_number(settings.flow_cosmos_sigma_max)}")
    return " · ".join(parts)


# ── 스케줄러 이름 ───────────────────────────────────────────────────────────────
_FOLDED_SCHEDULERS = MappingProxyType({
    **{name.casefold(): name for name in SCHEDULER_NAMES},
    **{label.casefold(): name for name, label in SCHEDULER_LABELS.items()},
    **{alias.casefold(): name for name, alias in README_LABELS.items()},
})


def resolve_scheduler(text: Any) -> Optional[str]:
    """스케줄러 이름·라벨·README 별칭(대소문자·앞뒤 공백 무시) → 이 확장의 이름. 'Use same scheduler'·다른 스케줄러·
    빈 값은 None. Forge 콤보 값(이름)·infotext 라벨·ComfyUI 이름이 모두 같은 이름이 된다."""
    return _FOLDED_SCHEDULERS.get(" ".join(str(text or "").split()).casefold())


# ── 블록 판정 ─────────────────────────────────────────────────────────────────
TARGET_T2I, TARGET_I2I, TARGET_AUX = "t2i", "i2i", "aux"             # core/alwayson_propagation TARGETS
BACKEND_WEBUI, BACKEND_COMFY, BACKEND_KREA2 = "webui", "comfyui", "krea2"
PROVENANCE_USER, PROVENANCE_APP_DEFAULT = "user", "app_default"     # core/alwayson_propagation PROVENANCE_*

REASON_SEND = "send"                    # 스냅샷이 스크립트를 확인했다 — 늘 싣는다
REASON_SEND_UNCONFIRMED = "send_unconfirmed"   # 확인 못 했지만 어느 패스가 값 스케줄러 — 게이트가 정한다
REASON_NOT_NEEDED = "not_needed"        # 확인 못 했고 값을 읽는 패스가 없다 — 싣지 않는다(알림 없음)
REASON_COMFY = "comfy"                  # ComfyUI — 늘 싣는다(노드를 넣을지는 컴파일러가 정한다, 계획 C6)
REASON_KREA2 = "krea2"


@dataclass(frozen=True)
class ExtraSchedulersPlan:
    block: Optional[dict]          # alwayson 블록 {"args": [dict]} | None(보내지 않음)
    notice: Any                    # 늘 None — 생성 전 알림은 게이트(빠진 블록)와 결과 알림이 한다
    reason: str                    # REASON_*
    provenance: str = PROVENANCE_USER


def script_confirmed(capabilities: Any, *, img2img: bool) -> Optional[bool]:
    """기능 스냅샷이 이 스크립트를 확인했나 — True(있음)·False(없음)·None(모름). 판정은 core/alwayson_propagation
    ``script_available`` 와 같다(known 스냅샷의 script(title) 의 txt2img·img2img 칸 — 테스트 더블·옛 스냅샷은 모름)."""
    if getattr(capabilities, "known", None) is not True:
        return None
    script = getattr(capabilities, "script", None)
    if not callable(script):
        return None
    try:
        detail = script(SCRIPT_NAME)
    except Exception:
        return None
    if not isinstance(detail, Mapping):
        return None
    value = detail.get("img2img" if img2img else "present")
    return value if isinstance(value, bool) else None


def plan(settings: ExtraSchedulerSettings, *, target: str, backend: str, capabilities: Any = None,
         passes: Iterable[Any] = ()) -> ExtraSchedulersPlan:
    """이 요청에 Extra Schedulers 블록을 실을지(계획 X5).

    1 Krea2 → 없음  2 ComfyUI → 늘 블록(``REASON_COMFY`` — 스냅샷·게이트가 없다. 컴파일러가 값 스케줄러 패스가 있을 때만
      ``ForgeNeoExtraSchedulers`` 를 넣어 다른 그래프는 바이트 단위로 그대로 — 계획 C2a·C6)
    3 Forge 스냅샷이 스크립트를 확인 → 늘 블록(값은 VALUE_SCHEDULERS 패스만 읽어 결과가 같다 — XYZ
      Scheduler 축 칸·사용자 ui-config 기본값과 무관하게 같은 요청이 된다)
    4 확인 못 함(모름·없음) → ``passes``(값 스케줄러를 쓰는 패스 — ``passes_from_widgets``)가 있을 때만 블록. 그
      뒤는 core/alwayson_propagation 게이트가 출처로 정한다(모름: 사용자 값만 보냄, 없음: 빼고 사용자 값이면 알림)
    5 그 밖 → 없음(알림 없음 — 쓰지 않는 값)

    출처: 값이 확장 기본값과 같으면 앱 기본값(모를 때 보내지 않고 빠져도 알리지 않는다), 다르면 사용자 값. 보조 패스
    봉투(aux)는 t2i 와 같은 블록이다(봉투 = 메인 체인, T15). i2i 는 스냅샷의 img2img 목록을 본다. 블록은 늘 키 8개다
    (옛 sam-extra 는 Flow Cosmos 셋을 버리고, 그 판에는 그 값을 읽는 스케줄러도 없다 — 모듈 설명)."""
    if backend == BACKEND_KREA2:
        return ExtraSchedulersPlan(None, None, REASON_KREA2)
    provenance = PROVENANCE_APP_DEFAULT if is_default(settings) else PROVENANCE_USER
    if backend == BACKEND_COMFY:
        return ExtraSchedulersPlan(as_block(settings), None, REASON_COMFY, provenance)
    if script_confirmed(capabilities, img2img=target == TARGET_I2I) is True:
        return ExtraSchedulersPlan(as_block(settings), None, REASON_SEND, provenance)
    if any(True for _ in passes or ()):
        return ExtraSchedulersPlan(as_block(settings), None, REASON_SEND_UNCONFIRMED, provenance)
    return ExtraSchedulersPlan(None, None, REASON_NOT_NEEDED, provenance)


# ── 어느 패스가 값을 읽나 ──────────────────────────────────────────────────────
PASS_MAIN, PASS_HIRES, PASS_SAM3 = "main", "hires", "sam3"
_ADETAILER_PREFIX = "adetailer_"


def adetailer_pass(index: int) -> str:
    """ADetailer 슬롯 패스 id('adetailer_1'…)."""
    return f"{_ADETAILER_PREFIX}{int(index)}"


def pass_label(pass_id: str) -> str:
    """카드 부제·로그의 패스 이름 — 메인 · Hires · ADetailer 1 · SAM3 (frontend/src/utils/extraSchedulers.ts passLabel)."""
    text = str(pass_id)
    if text == PASS_MAIN:
        return "메인"
    if text == PASS_HIRES:
        return "Hires"
    if text == PASS_SAM3:
        return "SAM3"
    if text.startswith(_ADETAILER_PREFIX):
        return f"ADetailer {text[len(_ADETAILER_PREFIX):]}"
    return text


class PassUse(NamedTuple):
    pass_id: str     # PASS_MAIN | PASS_HIRES | adetailer_pass(i) | PASS_SAM3
    scheduler: str   # VALUE_SCHEDULERS 의 이름(laplace | custom | flow_cosmos_rho7 | flow_cosmos_dynamic)


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
    """비었거나 'Use same scheduler'(대소문자 무시) — 메인 스케줄러를 쓴다."""
    folded = " ".join(str(text or "").split()).casefold()
    return not folded or folded == USE_SAME_SCHEDULER.casefold()


def _value_uses(entries: Iterable[tuple], resolve) -> tuple:
    out = []
    for pass_id, text in entries:
        name = resolve(text)
        if name in VALUE_SCHEDULERS:
            out.append(PassUse(pass_id, name))
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


def value_scheduler_passes(payload: Any, *, resolve=None) -> tuple:
    """요청 페이로드에서 값 스케줄러(laplace·custom·flow_cosmos_rho7·flow_cosmos_dynamic)를 쓰는 패스 → ``PassUse`` 튜플(패스 순서). 비면 이
    블록의 값을 읽는 패스가 없다.

    각 패스가 실제로 읽는 칸을 그대로 본다(ComfyUI 컴파일러와 같은 규칙 — 계획 C2a 가 ``resolve=`` 로 Comfy 이름
    변환을 넘긴다): 메인 ``scheduler``; ``enable_hr`` 면 ``hr_scheduler``(비었거나 'Use same' 이면 메인); ADetailer
    슬롯(켜진 탭·모델 'None' 아님)은 ``ad_use_sampler`` 면 ``ad_scheduler`` 아니면 메인; SAM3(켜짐·'Mask only' 아님)는
    ``sam3_scheduler``('Use same' 이면 메인). ``resolve`` 기본은 ``resolve_scheduler``(이름·라벨·별칭). ADetailer 패스
    번호는 페이로드에 실린 켜진 슬롯의 순서다(앱은 켠 슬롯만 싣는다 — 컴파일러 노드 제목의 번호와 같다)."""
    resolve = resolve or resolve_scheduler
    if not isinstance(payload, Mapping):
        return ()
    main = payload.get("scheduler")
    entries: list = [(PASS_MAIN, main)]
    if _truthy(payload.get("enable_hr")):
        hires = payload.get("hr_scheduler")
        entries.append((PASS_HIRES, main if _uses_same(hires) else hires))
    adetailer = _script_block(payload, "ADetailer")
    args = adetailer.get("args") if adetailer else None
    if isinstance(args, (list, tuple)) and args and _truthy(args[0], True):
        slots = [item for item in list(args)[2:]
                 if isinstance(item, Mapping) and _truthy(item.get("ad_tab_enable"), True)
                 and str(item.get("ad_model") or "None").strip().casefold() != "none"]
        for index, slot in enumerate(slots, start=1):
            text = slot.get("ad_scheduler") if _truthy(slot.get("ad_use_sampler")) else main
            entries.append((adetailer_pass(index), main if _uses_same(text) else text))
    sam3 = _script_block(payload, sam3_args.SCRIPT_SAM3)
    sam3_list = sam3.get("args") if sam3 else None
    state = sam3_list[0] if isinstance(sam3_list, (list, tuple)) and sam3_list else None
    if isinstance(state, Mapping) and _truthy(state.get("sam3_enable", state.get("enabled", True)), True) \
            and sam3_args.mode_of(dict(state)) != sam3_args.MODE_MASK_ONLY:
        text = state.get("sam3_scheduler")
        entries.append((PASS_SAM3, main if _uses_same(text) else text))
    return _value_uses(entries, resolve)


# 패스를 정하는 위젯 id(Vue storeWidgets 키 — frontend/src/utils/extraSchedulers.ts PASS_WIDGETS 와 같다)
MAIN_SCHEDULER_WIDGET = "scheduler_combo"
HIRES_GROUP_WIDGET, HIRES_SCHEDULER_WIDGET = "hires_options_group", "hires_scheduler_combo"
ADETAILER_GROUP_WIDGET = "adetailer_group"
# (슬롯 번호, 슬롯 켬, '별도 샘플러' 켬, 스케줄러) — ADetailer 의 ad_use_sampler 가 샘플러·스케줄러를 같이 덮는다
ADETAILER_SLOT_WIDGETS = ((1, "ad_slot1_group", "_ad_s1_use_sampler", "_ad_s1_scheduler"),
                          (2, "ad_slot2_group", "_ad_s2_use_sampler", "_ad_s2_scheduler"))
SAM3_GROUP_WIDGET, SAM3_MODE_WIDGET = "sam3_group", "_sam3_mode"
SAM3_USE_SCHEDULER_WIDGET, SAM3_SCHEDULER_WIDGET = "_sam3_use_scheduler", "_sam3_scheduler"


def pass_widget_ids() -> tuple:
    """``passes_from_widgets`` 가 읽는 위젯 id 전부(순서 고정 — 거울 테스트·Qt 접착층이 쓴다)."""
    ids = [MAIN_SCHEDULER_WIDGET, HIRES_GROUP_WIDGET, HIRES_SCHEDULER_WIDGET, ADETAILER_GROUP_WIDGET]
    for _index, *widget_ids in ADETAILER_SLOT_WIDGETS:
        ids.extend(widget_ids)
    ids.extend((SAM3_GROUP_WIDGET, SAM3_MODE_WIDGET, SAM3_USE_SCHEDULER_WIDGET, SAM3_SCHEDULER_WIDGET))
    return tuple(ids)


def passes_from_widgets(values: Any, *, img2img: bool = False) -> tuple:
    """화면 값(위젯 id → 값, 체크는 bool 또는 'true'/'false')에서 값 스케줄러(VALUE_SCHEDULERS)를 쓰는 패스 →
    ``PassUse`` 튜플.

    다음 T2I 요청이 실제로 만드는 패스와 같은 규칙(ui/generator_generation 의 페이로드 조립): 메인; Hires 켬이고 스케줄러가
    'Use same' 이 아니면 그 값; ADetailer 켬 + 슬롯 켬이면 '별도 샘플러' 켬일 때 슬롯 스케줄러 아니면 메인; SAM3 켬 +
    'Mask only' 아님이면 '별도 스케줄러' 켬이고 값이 있을 때 그 값 아니면 메인. 카드(frontend passesUsing)와 기여자
    (스냅샷이 스크립트를 확인 못 했을 때 보낼지)가 쓴다.

    ``img2img``: I2I·인페인트·채팅 편집 요청은 스케줄러·Hires 를 싣지 않는다(core/i2i_payload — Forge 기본 스케줄러로 돈다)
    — 메인은 값을 읽지 않는 것으로 보고 Hires 는 없다. ADetailer·SAM3 는 그 요청 안에서도 자기 스케줄러로 돈다."""
    values = values if isinstance(values, Mapping) else {}
    main = None if img2img else values.get(MAIN_SCHEDULER_WIDGET)
    entries: list = [(PASS_MAIN, main)]
    if not img2img and _truthy(values.get(HIRES_GROUP_WIDGET)):
        hires = values.get(HIRES_SCHEDULER_WIDGET)
        entries.append((PASS_HIRES, main if _uses_same(hires) else hires))
    if _truthy(values.get(ADETAILER_GROUP_WIDGET)):
        for index, enabled, use_sampler, scheduler in ADETAILER_SLOT_WIDGETS:
            if not _truthy(values.get(enabled)):
                continue
            text = values.get(scheduler) if _truthy(values.get(use_sampler)) else main
            entries.append((adetailer_pass(index), main if _uses_same(text) else text))
    if _truthy(values.get(SAM3_GROUP_WIDGET)) and \
            sam3_args.mode_of({"sam3_mode": values.get(SAM3_MODE_WIDGET)}) != sam3_args.MODE_MASK_ONLY:
        text = values.get(SAM3_SCHEDULER_WIDGET) if _truthy(values.get(SAM3_USE_SCHEDULER_WIDGET)) else main
        entries.append((PASS_SAM3, main if _uses_same(text) else text))
    return _value_uses(entries, resolve_scheduler)


# ── infotext (결과 기록·붙여 넣기) ───────────────────────────────────────────────
# Forge schedulers_map 에 확장이 넣는 열쇠(정확히 — 대소문자 구분): 이름·라벨·README 별칭 → 이름
_EXACT_SCHEDULERS = MappingProxyType({
    **{name: name for name in SCHEDULER_NAMES},
    **{label: name for name, label in SCHEDULER_LABELS.items()},
    **{alias: name for name, alias in README_LABELS.items()},
})


def infotext_labels(params: Any) -> frozenset:
    """붙여 넣은 infotext 의 'Schedule type'·'Hires schedule type' → 라벨 집합(확장 _labels_in_params 의 거울).

    Forge 가 이 확장의 스케줄러로 찾는 표기(이름·라벨·README 별칭 — 대소문자 그대로)는 그 라벨로, 그 밖은 적힌 대로.
    'Use same scheduler'·빈 값은 뺀다."""
    labels = set()
    params = params if isinstance(params, Mapping) else {}
    for key in (KEY_SCHEDULE_TYPE, KEY_HIRES_SCHEDULE_TYPE):
        value = params.get(key)
        if value and value != USE_SAME_SCHEDULER:
            name = _EXACT_SCHEDULERS.get(str(value))
            labels.add(SCHEDULER_LABELS[name] if name else str(value))
    return frozenset(labels)


def from_infotext(params: Any, *, base: ExtraSchedulerSettings = APP_DEFAULTS) -> ExtraSchedulerSettings:
    """PNG Info 붙여 넣기·'이 설정으로 생성' — 확장 paste_* 규칙의 거울(sam3ext/ui_extra_schedulers.py).

    모드·식·목록은 있는 custom 키에서(시그마 목록 키가 있으면 sigmas 모드), μ/β 와 Flow Cosmos ρ·σ̃ max·σ̃ min 은 키에서
    — 키가 없어도 이미지가 Laplace(또는 Flow Cosmos rho7·Dynamic)를 썼으면 그 스케줄러의 기본값(v0.33.0 D19 전 이미지·다른 도구의
    이미지). 이미지가 쓰지 않은 스케줄러의 칸은 ``base`` 그대로. 글자는 확장이 생성할 때처럼 앞뒤 공백을 뗀다(coerce).
    'Schedule rho' 는 이 확장 값이 아니다(Flow Cosmos 의 ρ 는 'Flow Cosmos rho')."""
    params = params if isinstance(params, Mapping) else {}
    values: dict = {}
    if KEY_CUSTOM_SIGMAS in params:
        values["custom_mode"] = MODE_SIGMAS
    elif KEY_CUSTOM_EXPRESSION in params:
        values["custom_mode"] = MODE_EXPRESSION
    if params.get(KEY_CUSTOM_EXPRESSION) is not None:
        values["custom_expression"] = coerce_text(str(params.get(KEY_CUSTOM_EXPRESSION)), DEFAULT_EXPRESSION)
    if params.get(KEY_CUSTOM_SIGMAS) is not None:
        values["custom_sigmas"] = coerce_text(str(params.get(KEY_CUSTOM_SIGMAS)), DEFAULT_SIGMAS)
    labels = infotext_labels(params)
    laplace_used = LABEL_LAPLACE in labels
    cosmos_used = not labels.isdisjoint(FLOW_COSMOS_LABELS)   # 확장 _paste_number(…, FLOW_COSMOS_LABELS)
    for key, field_name, coerce_number, default, used in (
            (KEY_LAPLACE_MU, "laplace_mu", coerce_laplace_mu, LAPLACE_MU_DEFAULT, laplace_used),
            (KEY_LAPLACE_BETA, "laplace_beta", coerce_laplace_beta, LAPLACE_BETA_DEFAULT, laplace_used),
            (KEY_FLOW_COSMOS_RHO, "flow_cosmos_rho", coerce_flow_cosmos_rho, FLOW_COSMOS_RHO_DEFAULT, cosmos_used),
            (KEY_FLOW_COSMOS_SIGMA_MAX, "flow_cosmos_sigma_max", coerce_flow_cosmos_sigma_max,
             FLOW_COSMOS_SIGMA_MAX_DEFAULT, cosmos_used),
            (KEY_FLOW_COSMOS_SIGMA_MIN, "flow_cosmos_sigma_min", coerce_flow_cosmos_sigma_min,
             FLOW_COSMOS_SIGMA_MIN_DEFAULT, cosmos_used)):
        if key in params:
            values[field_name] = coerce_number(params[key])
        elif used:
            values[field_name] = default
    return replace(base, **values)


# '이 설정으로 생성'이 되살리는 칸 — 그 칸을 읽는 스케줄러(이름)마다(계획 X3b: 스케줄러와 그 값을 함께)
_FLOW_COSMOS_FIELDS = ("flow_cosmos_rho", "flow_cosmos_sigma_max", "flow_cosmos_sigma_min")
RESTORE_FIELDS = MappingProxyType({
    NAME_CUSTOM: ("custom_mode", "custom_expression", "custom_sigmas"),
    NAME_LAPLACE: ("laplace_mu", "laplace_beta"),
    NAME_FLOW_COSMOS_RHO7: _FLOW_COSMOS_FIELDS,
    NAME_FLOW_COSMOS_DYNAMIC: _FLOW_COSMOS_FIELDS,
})


def infotext_value_schedulers(params: Any) -> tuple:
    """infotext 의 메인('Schedule type')·Hires('Hires schedule type') 스케줄러 중 값을 읽는 것(VALUE_SCHEDULERS 의
    이름, 메인 먼저·중복 없이). 표기는 ``resolve_scheduler`` 로 접어 읽는다 — 앱 콤보가 이름·라벨·별칭을 같은 항목으로
    고르는 것과 같다(ui/generation_settings_apply.apply_sampler_scheduler). 'Use same scheduler'·다른 스케줄러는 뺀다."""
    params = params if isinstance(params, Mapping) else {}
    names: list = []
    for key in (KEY_SCHEDULE_TYPE, KEY_HIRES_SCHEDULE_TYPE):
        name = resolve_scheduler(params.get(key))
        if name in VALUE_SCHEDULERS and name not in names:
            names.append(name)
    return tuple(names)


def restore_from_infotext(params: Any, *, base: ExtraSchedulerSettings = APP_DEFAULTS
                          ) -> Optional[ExtraSchedulerSettings]:
    """'이 설정으로 생성'(계획 A-R1·X3b) — 이미지의 메인·Hires 스케줄러가 값을 읽는 스케줄러(custom·Laplace·Flow
    Cosmos rho7·Dynamic)일 때만 **그 스케줄러의** 칸(``RESTORE_FIELDS``)을 ``from_infotext`` 규칙(확장 paste_*)으로
    되살린다 — 스케줄러 이름만 오고 값이 남으면 다른 그림이 된다. 쓰지 않은 스케줄러의 칸은 키가 있어도 ``base`` 그대로,
    어느 패스도 값 스케줄러가 아니면 None(칸을 건드리지 않는다).

    접어 읽은 표기('LAPLACE'·'flow_cosmos_rho7'·README 별칭)는 확장 라벨로 바꿔 ``from_infotext`` 에 넘긴다 — 그 안의
    '키 없으면 기본값' 규칙(확장의 정확한 라벨 비교)이 앱이 고른 스케줄러와 같은 것을 보게."""
    used = infotext_value_schedulers(params)
    if not used:
        return None
    pasted = dict(params)
    for key in (KEY_SCHEDULE_TYPE, KEY_HIRES_SCHEDULE_TYPE):
        name = resolve_scheduler(pasted.get(key))
        if name is not None:
            pasted[key] = SCHEDULER_LABELS[name]
    restored = from_infotext(pasted, base=base)
    fields = {field_name for name in used for field_name in RESTORE_FIELDS[name]}
    return replace(base, **{field_name: getattr(restored, field_name) for field_name in sorted(fields)})


def infotext_items(settings: ExtraSchedulerSettings, schedulers: Iterable[Any]) -> dict:
    """생성이 쓴 스케줄러(메인·Hires — 이름이나 라벨)에 맞는 infotext 키(확장 infotext_items 의 거울): custom 이면
    쓴 모드의 글자(공백 한 칸으로), Laplace 면 μ·β 둘 다(repr), Flow Cosmos rho7·Dynamic 이면 ρ·σ̃ max·σ̃ min 셋 다(repr — eps/v
    모델이라 σ̃ 를 쓰지 않았어도). ComfyUI PNG 메타데이터가 Forge 와 같게 적을 때 쓴다."""
    names = {resolve_scheduler(item) for item in (schedulers or ())}
    items: dict = {}
    if NAME_CUSTOM in names:
        if settings.custom_mode == MODE_SIGMAS:
            items[KEY_CUSTOM_SIGMAS] = normalize_text(settings.custom_sigmas)
        else:
            items[KEY_CUSTOM_EXPRESSION] = normalize_text(settings.custom_expression)
    if NAME_LAPLACE in names:
        items[KEY_LAPLACE_MU] = format_number(settings.laplace_mu)
        items[KEY_LAPLACE_BETA] = format_number(settings.laplace_beta)
    if not names.isdisjoint(FLOW_COSMOS_SCHEDULERS):
        items[KEY_FLOW_COSMOS_RHO] = format_number(settings.flow_cosmos_rho)
        items[KEY_FLOW_COSMOS_SIGMA_MAX] = format_number(settings.flow_cosmos_sigma_max)
        items[KEY_FLOW_COSMOS_SIGMA_MIN] = format_number(settings.flow_cosmos_sigma_min)
    return items


__all__ = [
    "ADETAILER_GROUP_WIDGET", "ADETAILER_SLOT_WIDGETS", "APP_DEFAULTS", "ARG_NAMES", "BACKEND_COMFY", "BACKEND_KREA2", "BACKEND_WEBUI",
    "COMFY_INPUTS", "COMFY_MIN_PACK_VERSION", "COMFY_NODE_CLASS", "COMFY_REPORT_INPUT", "DEFAULTS", "DEFAULT_EXPRESSION",
    "DEFAULT_SIGMAS", "ERROR_PREFIX", "ERROR_PROPERTY", "EXTENSION_DEFAULTS", "ExtraSchedulerSettings", "ExtraSchedulersPlan",
    "FLOW_COSMOS_DYNAMIC_RHO_ABOVE", "FLOW_COSMOS_LABELS", "FLOW_COSMOS_SCHEDULERS",
    "FLOW_COSMOS_RHO", "FLOW_COSMOS_RHO_DEFAULT", "FLOW_COSMOS_RHO_MAX", "FLOW_COSMOS_RHO_MIN", "FLOW_COSMOS_RHO_STEP",
    "FLOW_COSMOS_SIGMA_MAX", "FLOW_COSMOS_SIGMA_MAX_DEFAULT", "FLOW_COSMOS_SIGMA_MAX_MAX", "FLOW_COSMOS_SIGMA_MAX_MIN",
    "FLOW_COSMOS_SIGMA_MAX_STEP", "FLOW_COSMOS_SIGMA_MIN", "FLOW_COSMOS_SIGMA_MIN_DEFAULT", "FLOW_COSMOS_SIGMA_MIN_MAX",
    "FLOW_COSMOS_SIGMA_MIN_MIN", "FLOW_COSMOS_SIGMA_MIN_STEP",
    "FLOW_UNSUITABLE", "HIRES_GROUP_WIDGET", "HIRES_SCHEDULER_WIDGET", "INFOTEXT_KEYS", "KEY_CUSTOM_EXPRESSION", "KEY_CUSTOM_SIGMAS",
    "KEY_FLOW_COSMOS_RHO", "KEY_FLOW_COSMOS_SIGMA_MAX", "KEY_FLOW_COSMOS_SIGMA_MIN", "KEY_HIRES_SCHEDULE_TYPE",
    "KEY_LAPLACE_BETA", "KEY_LAPLACE_MU", "KEY_SCHEDULE_TYPE", "LABEL_CUSTOM", "LABEL_FLOW_COSMOS_DYNAMIC", "LABEL_FLOW_COSMOS_RHO7", "LABEL_LAPLACE",
    "LABEL_REACT_DYNSF",
    "LAPLACE_BETA_DEFAULT", "LAPLACE_BETA_MAX", "LAPLACE_BETA_MIN", "LAPLACE_BETA_STEP", "LAPLACE_MU_DEFAULT",
    "LAPLACE_MU_MAX", "LAPLACE_MU_MIN", "LAPLACE_MU_STEP", "MAIN_SCHEDULER_WIDGET", "MODES", "MODE_EXPRESSION", "MODE_LABELS", "MODE_SIGMAS",
    "NAME_CUSTOM", "NAME_FLOW_COSMOS_DYNAMIC", "NAME_FLOW_COSMOS_RHO7", "NAME_LAPLACE", "NAME_REACT_DYNSF", "PACK_MATH_MODULE", "PASS_HIRES", "PASS_MAIN", "PASS_SAM3", "PROVENANCE_APP_DEFAULT", "PROVENANCE_USER", "README_LABELS",
    "REACT_DYNSF_FACTOR", "REASON_COMFY", "REASON_KREA2", "REASON_NOT_NEEDED", "REASON_SEND", "REASON_SEND_UNCONFIRMED",
    "RESTORE_FIELDS", "PassUse", "SAM3_GROUP_WIDGET", "SAM3_MODE_WIDGET", "SAM3_SCHEDULER_WIDGET", "SAM3_USE_SCHEDULER_WIDGET",
    "SCHEDULER_LABELS", "SCHEDULER_NAMES", "SCRIPT_NAME", "SETTINGS_KEY", "TARGET_AUX", "TARGET_I2I", "TARGET_T2I",
    "USE_SAME_SCHEDULER", "VALUE_SCHEDULERS", "WIDGET_KEYS", "WIDGET_PREFIX", "adetailer_pass", "as_block", "check_settings", "check_text", "coerce", "coerce_args",
    "coerce_float", "coerce_flow_cosmos_rho", "coerce_flow_cosmos_sigma_max", "coerce_flow_cosmos_sigma_min",
    "coerce_laplace_beta", "coerce_laplace_mu", "coerce_mode", "coerce_text", "describe", "flow_cosmos_dynamic_rho_ok", "flow_cosmos_range_ok",
    "format_number", "from_infotext", "infotext_items", "infotext_labels", "infotext_value_schedulers", "is_default",
    "normalize_text",
    "parse_script_block", "parse_settings", "pass_label", "pass_widget_ids", "passes_from_widgets", "plan",
    "resolve_scheduler", "restore_from_infotext", "script_confirmed", "value_scheduler_passes", "widget_id",
    "widget_number", "widget_values",
]
