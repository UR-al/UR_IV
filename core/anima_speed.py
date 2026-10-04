# core/anima_speed.py
"""Anima SPEED (sam-extra ``scripts/anima_speed.py``) — 저해상도 선행 샘플링. 앱 설정·정규화·블록 판정. 순수 로직
(Qt·네트워크·torch 없음).

확장은 ``alwayson_scripts["Anima SPEED"]`` 로 위치 인자 14개(``ARG_NAMES`` 순서) 또는 **dict 한 개**(``ARG_NAMES`` 키나
infotext 별칭 ``divisor``/``manual``/``A``/``beta``)를 받는다(sam3ext/speed/forge_host.py ``coerce_settings``). 앱은 14키
dict 하나로 보낸다 — Forge 가 캐시한 위치 인자 기본값이나 확장의 인자 순서와 무관하게 같은 뜻이 된다:
``{"args": [{"enabled": true, "mode": "transition", …, "seed": -1, "hires": false}]}``.

두 가지 설정 모양이 있다.

- ``SpeedSettings`` — **카드 값**(앱 저장·프리셋·위젯). 텍스트 칸(해상도 비율·수동 σ)은 친 그대로 두고, 숫자는 유한한 값,
  선택지는 확장 선택지 안의 값이다. ``apply_img2img`` 는 앱 전용(Forge 는 img2img 탭에 아코디언이 따로 있다 — D5).
- ``ScriptSettings`` — **확장이 읽는 값**. ``coerce`` 가 ``coerce_settings`` 를 한 줄씩 옮긴 거울이다(XYZ 덮어쓰기 없음 —
  앱은 [Anima SPEED] XYZ 축을 쓰지 않는다, P18). 필드·``error`` 문구·``summary()`` 가 확장과 같다
  (tests/test_anima_speed.py 가 설치된 확장과 표로 대조한다). ``script_settings(card)`` = 카드가 보낼 블록을 Forge 가
  읽은 결과.

규칙:

- **메인 요청에만**(D4) — 보조 패스(Refine·SAM3·ADetailer·손 재구성)는 마스크라 늘 쉬거나 σ 시작이 전환 σ 보다 낮다.
  ``plan`` 은 aux 에도 같은 블록을 만들지만(봉투가 메인 체인과 같게), core/alwayson_propagation 의 SPEED 행
  (``passes=frozenset()``)이 어느 보조 패스에도 주지 않는다. 게이트(스크립트 없음·확인 전)도 그 행이 한다.
- 잘못된 설정도 **보낸다** — Forge 는 ``Anima SPEED status: … invalid settings - …`` 를 남기고 평소처럼 샘플링한다(보내지
  않은 것과 같은 그림, 기록은 남는다). 카드는 ``validate`` 결과를 미리 보여 준다. 확장 ``build_plan`` 이 설정만 보고
  거절하는 것(custom A ≤ 0 — sam-extra 0.32.1 의 ``PlanError(kind="invalid")``, delta_optimal 의 σ 나눔값 ≤ 0, 수동 σ 개수,
  dwt 비율)도 ``validate`` 가 같은 조건으로 미리 잡는다(R18).
- 해상도 비율·수동 σ 칸의 ``a/0`` 분수는 sam-extra 0.33.0 부터 ``ValueError("division by zero: 'a/0'")`` — 다른 잘못된
  값처럼 그 칸의 설정 오류다(해상도 비율은 늘, 수동 σ 는 threshold 가 manual 일 때만 — 아니면 예전처럼 무시). API 의 아주 큰
  정수(float 로 못 바꾸는 int)는 기본값으로 읽는다. ``coerce`` 는 어떤 입력에도 예외를 올리지 않는다(0.32.1 까지는 ``a/0`` 의
  ``ZeroDivisionError`` 가 ``coerce_settings`` 밖으로 나가 Forge 스크립트 오류였다 — 그런 옛 확장의 결과는 결과 알림
  'Anima SPEED 기록 없음'(CODE_SPEED_NOT_APPLIED)이 말한다).
- ComfyUI 는 앱 노드 팩 1.7.0 의 ``ForgeNeoAnimaSpeed`` 가 같은 계산을 한다 — ``plan`` 은 Forge 와 같은 규칙으로 블록을
  싣고(계획 C6), 컴파일러가 메인 패스마다 노드를 넣거나 설정 오류·생성 전에 아는 건너뜀(인페인트·ControlNet·Spectrum)이면
  경고와 함께 뺀다(C4a). img2img·Hires 시작 latent 맞춤(P10 옵션)은 ComfyUI 에서 확장 기본값(켬)이다
  (D7, ``COMFY_IMG2IMG_RESCALE``).
- 앱 기본값 = 확장 기본값(꺼짐·transition·neo_shift·anima·0.5,1.0·δ 0.01·1.03·0.7·적응형·dct·A·β·시드 -1·Hires 끔) + I2I 끔.

확장 코드는 GPL-3.0 이라 옮기지 않았다 — 계약(이름·기본값·정규화·문구)을 다시 구현했고, 설치된 확장을 시험 대조 기준
(oracle)으로만 쓴다. 상수 일치는 tests/test_anima_speed.py(AST)와 레지스트리 SEMANTIC_PINS 가 지킨다. 제목 문자열을 다른
곳에 복사하지 말고 ``SCRIPT_NAME`` 을 import 한다(확장 스크립트의 ``title()`` 과 같아야 한다 — 다르면 Forge 가 422).
"""
from __future__ import annotations

import logging
import math
import re
from dataclasses import dataclass, replace
from types import MappingProxyType
from typing import Any, Iterable, Mapping, Optional

from core.forge_override_settings import OPT_SPEED_IMG2IMG_RESCALE as OPT_IMG2IMG_RESCALE
from core.infotext import unquote_param_value

logger = logging.getLogger(__name__)

# ── 확장 계약 ─────────────────────────────────────────────────────────────────
SCRIPT_NAME = "Anima SPEED"                       # scripts/anima_speed.py TITLE
INFOTEXT_KEY = "Anima SPEED"                      # sam3ext/speed/forge_host.py KEY — 설정 요약(summary)
STATUS_KEY = "Anima SPEED status"                 # forge_host.STATUS_KEY — 'base: applied (…) - … | hires: …'
INFOTEXT_IMG2IMG_RESCALE = "Anima SPEED img2img rescale"   # forge_host.INFOTEXT_IMG2IMG_RESCALE (init latent 패스만)
INFOTEXT_KEYS = (INFOTEXT_KEY, STATUS_KEY, INFOTEXT_IMG2IMG_RESCALE)
OPT_LOG = "sam3_speed_log"                        # forge_host.OPT_LOG — 콘솔 로그(앱은 다루지 않는다)
# OPT_IMG2IMG_RESCALE = core.forge_override_settings.OPT_SPEED_IMG2IMG_RESCALE (P10 행 — 위 import 가 단일 출처)
XYZ_PREFIX = "[Anima SPEED]"                      # scripts/anima_speed.py XYZ_PREFIX (P18 — 앱은 쓰지 않는다)

# 위치 인자(확장은 뒤에만 덧붙인다) — forge_host.ARG_NAMES
ARG_NAMES = (
    "enabled", "mode", "threshold", "preset", "scales", "delta", "sigma_divisor",
    "manual_sigmas", "adaptive", "transform", "spectrum_A", "spectrum_beta", "seed", "hires",
)
# 이름 인자 별칭 = infotext 요약의 키 — forge_host.ARG_ALIASES
ARG_ALIASES = MappingProxyType(
    {"divisor": "sigma_divisor", "manual": "manual_sigmas", "A": "spectrum_A", "beta": "spectrum_beta"})

MODES = ("transition", "respace")                         # sam3ext/speed/schedule.py MODES
THRESHOLDS = ("neo_shift", "delta_optimal", "manual")     # schedule.THRESHOLDS
TRANSFORMS = ("dct", "dwt", "fft")                        # schedule.TRANSFORMS
MODE_TRANSITION, MODE_RESPACE = MODES
THRESHOLD_NEO_SHIFT, THRESHOLD_DELTA_OPTIMAL, THRESHOLD_MANUAL = THRESHOLDS
TRANSFORM_DCT, TRANSFORM_DWT, TRANSFORM_FFT = TRANSFORMS

CUSTOM_DEFAULT_A = 203.615097       # schedule.CUSTOM_DEFAULT_A (공식 Flux 맞춤값 — custom 기본값)
CUSTOM_DEFAULT_BETA = 1.915461      # schedule.CUSTOM_DEFAULT_BETA
PRESET_CUSTOM = "custom"
# schedule.PRESETS 그대로(값 같음 — 팩 미러 대조·핀용). custom = None(카드의 A·β 를 쓴다)
PRESETS: dict = {
    "flux": {"A": 1963.48906, "beta": 1.915461, "ref_latent": 128},
    "wan21": {"A": 219.484718, "beta": 2.422687, "ref_latent": 128, "ref_shift": 8.0},
    "krea-2": {"A": 2357.960557, "beta": 2.422687, "ref_latent": 128, "ref_shift": 3.158193},
    "krea-2-raw": {"A": 887.402358, "beta": 2.422687, "ref_latent": 128, "ref_shift": 3.158193},
    "z-image": {"A": 877.001476, "beta": 1.915461, "ref_latent": 128, "ref_shift": 3.0},
    "flux2": {"A": 520.495848, "beta": 1.915461, "ref_latent": 64},
    "anima": {"A": 8664.998524, "beta": 2.422687, "ref_latent": 128, "ref_shift": 3.0},
    "custom": None,
}
# UI 순서(확장 드롭다운 순서) — schedule.PRESET_NAMES
PRESET_NAMES = ("anima", "flux", "flux2", "krea-2", "krea-2-raw", "z-image", "wan21", "custom")
REFERENCE_LATENT = 128              # schedule.REFERENCE_LATENT (1024 px 기준 latent 짧은 변)

# 확장 기본값 — forge_host.DEFAULTS (= ui() 기본값, 픽스처 t2i·i2i 같음)
DEFAULTS = MappingProxyType({
    "enabled": False,
    "mode": "transition",
    "threshold": "neo_shift",
    "preset": "anima",
    "scales": "0.5,1.0",
    "delta": 0.01,
    "sigma_divisor": 1.03,
    "manual_sigmas": "0.7",
    "adaptive": True,
    "transform": "dct",
    "spectrum_A": CUSTOM_DEFAULT_A,
    "spectrum_beta": CUSTOM_DEFAULT_BETA,
    "seed": -1,
    "hires": False,
})

# 확장 슬라이더(scripts/anima_speed.py ui() — 픽스처 script-info 와 같다). API 유효 범위는 더 넓다(coerce 규칙).
DELTA_MIN, DELTA_MAX, DELTA_STEP = 0.0001, 0.5, 0.0001
DIVISOR_MIN, DIVISOR_MAX, DIVISOR_STEP = 0.5, 2.0, 0.01

# 확장이 붙이지 않는 샘플러(forge_host.UNSUPPORTED_SAMPLERS — 함수 이름 → status 이유). 0.33.0 이 sam-extra 의 UniPC bh2
# (extra_samplers/unipc.py — 이름만 다른 Forge sample_unipc)를 더했다.
UNSUPPORTED_SAMPLERS = MappingProxyType({
    "restart_sampler": "Restart re-plans its own schedule",
    "sample_unipc": "UniPC is not segment-safe",
    "sample_unipc_bh2": "UniPC is not segment-safe",
    "sample_dpm_fast": "DPM fast has no sigma schedule",
    "sample_dpm_adaptive": "DPM adaptive has no sigma schedule",
})

# 패스 이름(forge_host.pass_label — status 의 'base: … | hires: …' 머리)
PASS_BASE, PASS_IMG2IMG, PASS_HIRES = "base", "img2img", "hires"
PASS_LABELS = (PASS_BASE, PASS_IMG2IMG, PASS_HIRES)

# ── ComfyUI (ai_studio_forge_parity 팩 1.7.0 노드 — 앱 컴파일러가 만든다, phase C) ─────────────
# 입력은 스크립트 인자 13개(hires 뺌 — 컴파일러가 hires 가 켜졌을 때만 Hires 패스에 노드를 넣고 pass_label 로 알린다)
# + pass_label·image_seed(생성 시드 — 이미지마다 +i, Forge p.seeds)·img2img_rescale. design_speed §4.2.
COMFY_NODE_CLASS = "ForgeNeoAnimaSpeed"
COMFY_INPUTS = (
    "model", "enabled", "mode", "threshold", "preset", "scales", "delta", "sigma_divisor", "manual_sigmas",
    "adaptive", "transform", "spectrum_A", "spectrum_beta", "seed", "pass_label", "image_seed", "img2img_rescale",
)
COMFY_PASS_LABELS = PASS_LABELS
COMFY_IMG2IMG_RESCALE = True        # P10 옵션 sam3_speed_img2img_rescale 의 확장 기본값(D7 — ComfyUI 로는 보내지 않는다)
COMFY_MIN_PACK_VERSION = "1.7.0"    # 노드가 처음 들어온 팩(core/comfy_node_pack.PACK_VERSION 은 이 이상)

# ── 위젯·저장 ───────────────────────────────────────────────────────────────
WIDGET_PREFIX = "_aspeed_"          # 옛 ComfyUI payload 키 speed_*(sorryhyun 공급자)와 다르게
WIDGET_KEYS = (
    "enabled", "mode", "threshold", "preset", "scales", "delta", "sigma_divisor", "manual_sigmas", "adaptive",
    "hires", "apply_img2img", "transform", "spectrum_A", "spectrum_beta", "seed",
)
SETTINGS_KEY = "anima_speed_settings"   # prompt_settings.json·생성 프리셋 키


def widget_id(key: str) -> str:
    """위젯 프록시 id = Vue ``storeWidgets`` 키(frontend/src/utils/animaSpeed.ts WIDGET_IDS)."""
    if key not in WIDGET_KEYS:
        raise KeyError(key)
    return f"{WIDGET_PREFIX}{key}"


# ── 확장 정규화의 거울 (sam3ext/speed/forge_host.py·schedule.py — 같은 조건·같은 문구) ──────────────
def _as_bool(value: Any, default: bool) -> bool:
    """forge_host._as_bool."""
    if isinstance(value, bool):
        return value
    if value is None:
        return default
    if isinstance(value, (int, float)):
        return bool(value)
    text = str(value).strip().lower()
    if text in ("true", "1", "yes", "on", "enable", "enabled"):
        return True
    if text in ("false", "0", "no", "off", "disable", "disabled", ""):
        return False
    return default


def _as_float(value: Any, default: float) -> float:
    """forge_host._as_float — 읽을 수 없거나(float 로 못 바꾸는 큰 정수의 OverflowError 도 — sam-extra 0.33.0) 유한하지
    않으면 ``default``."""
    try:
        out = float(value)
    except (TypeError, ValueError, OverflowError):
        return default
    return out if math.isfinite(out) else default


def _fmt(value: Any) -> str:
    """forge_host._fmt — infotext 요약의 숫자."""
    return format(float(value), ".10g")


def _choice(value: Any, choices: tuple, default: str, name: str, errors: list) -> str:
    """forge_host._choice — 대소문자 무시, 모르면 ``default`` 와 오류 한 줄."""
    text = str(value if value is not None else default).strip().lower()
    if text in choices:
        return text
    errors.append(f"{name} must be one of {', '.join(choices)} (got {value!r})")
    return default


def _numbers(text: Any) -> list:
    """schedule._numbers — 쉼표·세미콜론, ``a/b`` 분수, 유한한 수만. ``a/0`` 은 ``ValueError`` 'division by zero: 'a/0''
    (sam-extra 0.33.0 — 0.32.1 은 ZeroDivisionError 가 그대로 나갔다)."""
    if isinstance(text, (list, tuple)):
        items: Iterable = text
    else:
        items = str(text if text is not None else "").replace(";", ",").split(",")
    out = []
    for item in items:
        token = str(item).strip()
        if not token:
            continue
        if "/" in token:
            num, den = token.split("/", 1)
            try:
                value = float(num) / float(den)
            except ZeroDivisionError:   # ValueError — 부르는 쪽이 설정 오류로 적는다(0.33.0)
                raise ValueError(f"division by zero: {token!r}") from None
        else:
            value = float(token)
        if not math.isfinite(value):
            raise ValueError(f"not a finite number: {token!r}")
        out.append(value)
    return out


def validate_scales(scales) -> None:
    """schedule.validate_scales — (0, 1] 의 순증가, 마지막 1.0."""
    if len(scales) == 0:
        raise ValueError("list of resolution scales is empty; supply at least one value.")
    if any(s <= 0.0 or s > 1.0 for s in scales):
        raise ValueError(f"every scale must be in (0, 1]; got {list(scales)}")
    if abs(scales[-1] - 1.0) > 1e-6:
        raise ValueError(f"last scale must equal 1.0 (full resolution); got {scales[-1]}")
    for a, b in zip(scales[:-1], scales[1:]):
        if not (a < b):
            raise ValueError(f"scales must be strictly increasing; got {list(scales)}")


def parse_scales(text: Any) -> tuple:
    """schedule.parse_scales — 1.0 으로 끝나는 비율(``1/2`` 분수)."""
    try:
        values = _numbers(text)
    except ValueError as exc:
        raise ValueError(f"scales: {exc}") from None
    validate_scales(values)
    return tuple(values)


def parse_sigmas(text: Any, *, strictly_decreasing: bool = True) -> tuple:
    """schedule.parse_sigmas — 전환마다 (0, 1) 의 σ, transition 은 순감소."""
    try:
        values = _numbers(text)
    except ValueError as exc:
        raise ValueError(f"manual sigmas: {exc}") from None
    if not values:
        raise ValueError("manual sigmas: empty")
    if any(not (0.0 < v < 1.0) for v in values):
        raise ValueError(f"every manual sigma must be in (0, 1); got {values}")
    if strictly_decreasing:
        for a, b in zip(values[:-1], values[1:]):
            if not (a > b):
                raise ValueError(f"manual sigmas must be strictly decreasing; got {values}")
    return tuple(values)


def preset_params(name: Any, A: float = CUSTOM_DEFAULT_A, beta: float = CUSTOM_DEFAULT_BETA) -> tuple:
    """schedule.preset_params — ``(A, beta, ref_latent, ref_shift)``; custom·모르는 이름은 받은 A·β."""
    preset = PRESETS.get(str(name))
    if preset is None:
        return float(A), float(beta), REFERENCE_LATENT, None
    return preset["A"], preset["beta"], int(preset["ref_latent"]), preset.get("ref_shift")


@dataclass(frozen=True)
class ScriptSettings:
    """확장이 읽은 값 — forge_host.SpeedSettings 의 필드(``xyz`` 뺌)·``summary()`` 와 같다."""

    enabled: bool = False
    mode: str = "transition"
    threshold: str = "neo_shift"
    preset: str = "anima"
    scales: tuple = (0.5, 1.0)
    delta: float = 0.01
    sigma_divisor: float = 1.03
    manual_sigmas: tuple = (0.7,)
    adaptive: bool = True
    transform: str = "dct"
    spectrum_A: float = CUSTOM_DEFAULT_A
    spectrum_beta: float = CUSTOM_DEFAULT_BETA
    seed: int = -1
    hires: bool = False
    error: Optional[str] = None

    def summary(self) -> str:
        """``Anima SPEED`` infotext 값(``key=value`` 쌍 — 붙여 넣기가 되읽는다). forge_host.SpeedSettings.summary."""
        parts = [
            f"mode={self.mode}",
            f"threshold={self.threshold}",
            f"preset={self.preset}",
            "scales=" + ",".join(_fmt(v) for v in self.scales),
            f"delta={_fmt(self.delta)}",
            f"divisor={_fmt(self.sigma_divisor)}",
            "manual=" + ",".join(_fmt(v) for v in self.manual_sigmas),
            f"adaptive={self.adaptive}",
            f"transform={self.transform}",
        ]
        if self.preset == "custom":
            parts += [f"A={_fmt(self.spectrum_A)}", f"beta={_fmt(self.spectrum_beta)}"]
        parts += [f"seed={self.seed}", f"hires={self.hires}"]
        return "; ".join(parts)


# ``validate`` 오류 코드 — 카드 문구는 VALIDATION_MESSAGES(frontend/src/utils/animaSpeed.ts 가 같은 코드를 쓴다)
ERR_CHOICE = "choice"              # mode·threshold·preset·transform 이 확장 선택지 밖
ERR_SCALES = "scales"
ERR_MANUAL = "manual"
ERR_DELTA = "delta"
ERR_DIVISOR = "divisor"
ERR_MANUAL_COUNT = "manual_count"  # build_plan: 수동 σ 개수 ≠ 전환 수
ERR_SPECTRUM_A = "spectrum_A"      # build_plan(0.32.1): custom A ≤ 0 또는 유한하지 않음
ERR_SPECTRUM_BETA = "spectrum_beta"  # build_plan(0.32.1): β 가 유한하지 않음
ERR_DWT_RATIO = "dwt_ratio"        # build_plan: dwt 는 단계마다 정확히 2배
VALIDATION_MESSAGES = MappingProxyType({
    ERR_CHOICE: "방식·전환 σ·스펙트럼 프리셋·변환 중 확장에 없는 값이 있습니다",
    ERR_SCALES: "해상도 비율은 (0, 1] 사이 오름차순이고 1.0으로 끝나야 합니다",
    ERR_MANUAL: "수동 σ는 0과 1 사이, transition 방식은 내림차순이어야 합니다",
    ERR_DELTA: "δ는 0과 1 사이여야 합니다",
    ERR_DIVISOR: "σ 나눔값은 0보다 커야 합니다",
    ERR_MANUAL_COUNT: "수동 σ 개수는 전환 수(해상도 비율 개수 − 1)와 같아야 합니다 (respace 는 하나도 됩니다)",
    ERR_SPECTRUM_A: "Spectrum A 는 0보다 커야 합니다",
    ERR_SPECTRUM_BETA: "Spectrum β 는 유한한 수여야 합니다",
    ERR_DWT_RATIO: "dwt 변환은 해상도 비율이 단계마다 정확히 2배여야 합니다 (dct·fft 는 아무 비율)",
})
INVALID_NOTE = "이대로 생성하면 SPEED 는 적용되지 않습니다(설정 오류)"


def _coerce(args) -> tuple:
    """forge_host.coerce_settings(args) 의 한 줄씩 거울(XYZ 없음) → ``(ScriptSettings, 오류 코드들)``.

    코드는 ``error`` 의 문구 하나마다 하나(같은 순서) — 카드 문구(``validate``)가 영어 문구를 해석하지 않게."""
    args = list(args or ())
    named = None
    if args and isinstance(args[0], dict):
        named = {ARG_ALIASES.get(str(key), str(key)): value for key, value in args[0].items()}

    def arg(name):
        if named is not None:
            return named.get(name, DEFAULTS[name])
        index = ARG_NAMES.index(name)
        return args[index] if index < len(args) else DEFAULTS[name]

    errors: list = []
    codes: list = []

    def mark(code: str, before: int) -> None:
        codes.extend([code] * (len(errors) - before))

    enabled = _as_bool(arg("enabled"), False)
    count = len(errors)
    mode = _choice(arg("mode"), MODES, "transition", "mode", errors)
    threshold = _choice(arg("threshold"), THRESHOLDS, "neo_shift", "threshold", errors)
    preset = str(arg("preset") or DEFAULTS["preset"]).strip()
    if preset not in PRESETS:
        errors.append(f"unknown spectrum preset {preset!r}")
        preset = DEFAULTS["preset"]
    transform = _choice(arg("transform"), TRANSFORMS, "dct", "transform", errors)
    mark(ERR_CHOICE, count)
    delta = _as_float(arg("delta"), DEFAULTS["delta"])
    divisor = _as_float(arg("sigma_divisor"), DEFAULTS["sigma_divisor"])
    manual_text = arg("manual_sigmas")
    scales_text = arg("scales")
    count = len(errors)
    try:
        scales = parse_scales(scales_text)
    except ValueError as exc:
        errors.append(str(exc))
        scales = (0.5, 1.0)
    mark(ERR_SCALES, count)
    count = len(errors)
    try:
        manual = parse_sigmas(manual_text, strictly_decreasing=(mode == "transition"))
    except ValueError as exc:
        manual = (0.7,)
        if threshold == "manual":
            errors.append(str(exc))
    mark(ERR_MANUAL, count)
    if threshold != "manual":
        if not (0.0 < delta < 1.0):
            errors.append(f"delta must be in (0, 1); got {delta}")
            codes.append(ERR_DELTA)
        if threshold == "neo_shift" and not (divisor > 0.0):
            errors.append(f"sigma divisor must be positive; got {divisor}")
            codes.append(ERR_DIVISOR)
    seed_value = _as_float(arg("seed"), -1.0)
    seed = int(seed_value) if seed_value >= 0 else -1
    settings = ScriptSettings(
        enabled=enabled, mode=mode, threshold=threshold, preset=preset, scales=tuple(scales),
        delta=delta, sigma_divisor=divisor, manual_sigmas=tuple(manual),
        adaptive=_as_bool(arg("adaptive"), True), transform=transform,
        spectrum_A=_as_float(arg("spectrum_A"), DEFAULTS["spectrum_A"]),
        spectrum_beta=_as_float(arg("spectrum_beta"), DEFAULTS["spectrum_beta"]),
        seed=seed, hires=_as_bool(arg("hires"), False),
        error="; ".join(errors) or None,
    )
    return settings, tuple(codes)


def coerce(args) -> ScriptSettings:
    """스크립트 인자(위치 목록 또는 첫 칸의 dict 하나) → 확장이 읽는 값. forge_host.coerce_settings(args) 와 같다
    (필드·``error`` 문구). 예외를 올리지 않는다 — ``a/0`` 도 ``error`` 다(sam-extra 0.33.0)."""
    return _coerce(args)[0]


def parse_summary(text: Any) -> dict:
    """``Anima SPEED`` infotext 값 → ``{키: 값}`` (모르는 키도 둔다). forge_host.parse_summary."""
    out: dict = {}
    for part in str(text or "").split(";"):
        if "=" in part:
            key, value = part.split("=", 1)
            out[key.strip()] = value.strip()
    return out


# ── 붙여 넣기 (scripts/anima_speed.py infotext_fields 의 거울) ──────────────────────
def _paste_bool(text: Any) -> bool:
    """scripts/anima_speed.py _paste_bool."""
    return str(text).strip().lower() in ("true", "1", "yes", "on")


def _paste_int(text: Any) -> int:
    """scripts/anima_speed.py _paste_int."""
    return int(float(text))


# (카드 필드, 요약 키, 변환) — 확장 infotext_fields 순서(enabled 는 키가 있는지로)
PASTE_FIELDS = (
    ("mode", "mode", str),
    ("threshold", "threshold", str),
    ("preset", "preset", str),
    ("scales", "scales", str),
    ("delta", "delta", float),
    ("sigma_divisor", "divisor", float),
    ("manual_sigmas", "manual", str),
    ("adaptive", "adaptive", _paste_bool),
    ("transform", "transform", str),
    ("spectrum_A", "A", float),
    ("spectrum_beta", "beta", float),
    ("seed", "seed", _paste_int),
    ("hires", "hires", _paste_bool),
)


def paste_values(params: Any) -> dict:
    """infotext dict → Gradio 가 받을 값(확장 ``_paste`` + Forge 붙여 넣기 루프). ``None`` = 그 칸은 그대로.

    ``enabled`` = ``Anima SPEED`` 키가 있나. 나머지는 요약에 그 키가 있고 변환이 되면 그 값 — 변환이 TypeError·ValueError
    면 ``_paste`` 가, 그 밖의 예외(``seed=inf`` 의 OverflowError)면 Forge 의 붙여 넣기 루프가 ``None`` 으로 둔다.
    값이 JSON 따옴표로 묶여 있으면(쉼표가 들어 있어 Forge 가 묶는다) 푼다."""
    params = params if isinstance(params, Mapping) else {}
    present = INFOTEXT_KEY in params
    out: dict = {"enabled": present}
    values = parse_summary(unquote_param_value(str(params[INFOTEXT_KEY]))) if present else {}
    for field_name, key, convert in PASTE_FIELDS:
        value = None
        if key in values:
            try:
                value = convert(values[key])
            except Exception:   # _paste: TypeError/ValueError → None; 그 밖은 Forge 붙여 넣기 루프가 None
                value = None
        out[field_name] = value
    return out


# ── 카드 설정 ─────────────────────────────────────────────────────────────────
@dataclass(frozen=True)
class SpeedSettings:
    """카드 값. 텍스트 칸은 친 그대로, 숫자는 유한, 선택지는 확장 선택지. ``apply_img2img`` 는 앱 전용."""

    enabled: bool = False
    mode: str = "transition"
    threshold: str = "neo_shift"
    preset: str = "anima"
    scales: str = "0.5,1.0"
    delta: float = 0.01
    sigma_divisor: float = 1.03
    manual_sigmas: str = "0.7"
    adaptive: bool = True
    transform: str = "dct"
    spectrum_A: float = CUSTOM_DEFAULT_A
    spectrum_beta: float = CUSTOM_DEFAULT_BETA
    seed: int = -1
    hires: bool = False
    apply_img2img: bool = False

    def as_arg(self) -> dict:
        """확장 dict 인자 — ``ARG_NAMES`` 14키, JSON 기본형(불리언은 진짜 불리언 — 문자열 'False' 는 참이 된다)."""
        return {
            "enabled": bool(self.enabled), "mode": self.mode, "threshold": self.threshold, "preset": self.preset,
            "scales": self.scales, "delta": float(self.delta), "sigma_divisor": float(self.sigma_divisor),
            "manual_sigmas": self.manual_sigmas, "adaptive": bool(self.adaptive), "transform": self.transform,
            "spectrum_A": float(self.spectrum_A), "spectrum_beta": float(self.spectrum_beta), "seed": int(self.seed),
            "hires": bool(self.hires),
        }


EXTENSION_DEFAULTS = SpeedSettings()
APP_DEFAULTS = EXTENSION_DEFAULTS


def _text(value: Any) -> str:
    return str(value if value is not None else "").strip()


def _truthy(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    return _text(value).lower() in {"1", "true", "yes", "on"}


def _finite(value: Any) -> Optional[float]:
    if isinstance(value, bool) or value is None:
        return None
    try:
        number = float(_text(value) if isinstance(value, str) else value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _choice_or_none(value: Any, choices: tuple) -> Optional[str]:
    text = _text(value).lower()
    return text if text in choices else None


def _preset_or_none(value: Any) -> Optional[str]:
    text = _text(value)
    if text in PRESET_NAMES:
        return text
    lowered = text.lower()
    return next((name for name in PRESET_NAMES if name == lowered), None)


def _seed_or_none(value: Any) -> Optional[int]:
    number = _finite(value)
    if number is None:
        return None
    return int(number) if number >= 0 else -1


def _card_values(raw: Mapping) -> dict:
    """원값 dict → 읽을 수 있는 카드 칸만(나머지는 부르는 쪽의 base)."""
    values: dict = {}
    for key in ("enabled", "adaptive", "hires", "apply_img2img"):
        if key in raw and raw.get(key) not in (None, ""):
            values[key] = _truthy(raw.get(key))
    for key, choices in (("mode", MODES), ("threshold", THRESHOLDS), ("transform", TRANSFORMS)):
        choice = _choice_or_none(raw.get(key), choices)
        if choice is not None:
            values[key] = choice
    preset = _preset_or_none(raw.get("preset"))
    if preset is not None:
        values["preset"] = preset
    for key in ("scales", "manual_sigmas"):
        if raw.get(key) is not None:
            values[key] = _text(raw.get(key))
    for key in ("delta", "sigma_divisor", "spectrum_A", "spectrum_beta"):
        number = _finite(raw.get(key))
        if number is not None:
            values[key] = number
    seed = _seed_or_none(raw.get("seed"))
    if seed is not None:
        values["seed"] = seed
    return values


def parse_settings(raw: Optional[Mapping], *, base: SpeedSettings = APP_DEFAULTS) -> SpeedSettings:
    """위젯 문자열·저장 dict → 카드 설정. 없거나 읽을 수 없는 칸(모르는 선택지·유한하지 않은 수)은 ``base`` 칸.

    텍스트 칸(해상도 비율·수동 σ)은 틀린 값이어도 친 그대로 둔다 — 카드가 ``validate`` 로 알린다. 시드는 확장 규칙대로
    음수면 -1, 소수는 버린다."""
    if not isinstance(raw, Mapping):
        return base
    return replace(base, **_card_values(raw))


def _num_text(value: float) -> str:
    """숫자 위젯 문자열 — 되읽으면 같은 float(1.0 → '1', 0.01 → '0.01')."""
    text = repr(float(value))
    return text[:-2] if text.endswith(".0") else text


def widget_values(settings: SpeedSettings) -> dict:
    """설정 → 위젯 문자열('true'/'false', 선택지, 친 텍스트, 숫자)."""
    flag = lambda value: "true" if value else "false"   # noqa: E731
    return {
        "enabled": flag(settings.enabled),
        "mode": settings.mode,
        "threshold": settings.threshold,
        "preset": settings.preset,
        "scales": settings.scales,
        "delta": _num_text(settings.delta),
        "sigma_divisor": _num_text(settings.sigma_divisor),
        "manual_sigmas": settings.manual_sigmas,
        "adaptive": flag(settings.adaptive),
        "hires": flag(settings.hires),
        "apply_img2img": flag(settings.apply_img2img),
        "transform": settings.transform,
        "spectrum_A": _num_text(settings.spectrum_A),
        "spectrum_beta": _num_text(settings.spectrum_beta),
        "seed": str(int(settings.seed)),
    }


def as_block(settings: SpeedSettings) -> dict:
    """alwayson 블록 — dict 형식 인자 하나(14키)."""
    return {"args": [settings.as_arg()]}


def script_settings(settings: SpeedSettings) -> ScriptSettings:
    """카드가 보낼 블록을 확장이 읽은 값(``coerce(as_block(settings)["args"])``)."""
    return coerce(as_block(settings)["args"])


def summary(settings: SpeedSettings) -> Optional[str]:
    """이 카드로 생성하면 Forge 가 남길 ``Anima SPEED`` 값(꺼짐이면 None — 설정 오류여도 요약은 남는다)."""
    if not settings.enabled:
        return None
    return script_settings(settings).summary()


def _block_args(block: Any) -> Optional[list]:
    if not isinstance(block, Mapping):
        return None
    args = block.get("args")
    return list(args) if isinstance(args, (list, tuple)) else []


def parse_script_block(block: Any) -> Optional[ScriptSettings]:
    """보낸 alwayson 블록 → 확장이 읽는 값(컴파일러용 — 위치 인자·dict·별칭 키 모두). 블록이 아니면 None.
    ``a/0`` 이 든 블록도 다른 잘못된 값처럼 ``error`` 가 있는 값이다(sam-extra 0.33.0 — 예외 없음)."""
    args = _block_args(block)
    if args is None:
        return None
    return coerce(args)


def from_infotext(params: Any, *, base: SpeedSettings = APP_DEFAULTS) -> SpeedSettings:
    """PNG Info 붙여 넣기 — 확장 infotext_fields 의 거울(``paste_values``)을 카드 규칙으로 받는다(P16 이 연결한다).

    ``Anima SPEED`` 키가 있을 때만 켠다. 각 칸은 붙여 넣을 값이 있고(``None`` 아님) 카드가 읽을 수 있을 때만 바꾸고
    (모르는 선택지·유한하지 않은 수는 ``base`` 칸), ``apply_img2img`` 는 ``base`` 그대로."""
    values = paste_values(params)
    raw = {key: value for key, value in values.items() if value is not None and key != "enabled"}
    return replace(base, **_card_values(raw), enabled=bool(values["enabled"]))


# ── 결과 status ('Anima SPEED status') ────────────────────────────────────────────
# 패스 하나의 status 머리 — scripts/anima_speed.py process_before_every_sampling 과 sam3ext/speed/forge_host.py
# make_wrapper 의 record_status 문구(f-string 이라 핀 대신 tests/test_sam_extra_notices.py 가 설치된 소스에서 찾는다).
STATUS_APPLIED = "applied"                         # 'applied (transition) - 5/28 steps coarse, 52x76 -> 104x152, …'
STATUS_INVALID = "invalid settings"                # 'invalid settings - <coerce 오류 또는 build_plan PlanError(invalid)>'
STATUS_SKIPPED = "skipped"                         # 'skipped - <runtime_guard·attach·build_plan PlanError(skip) 이유>'
STATUS_NOT_APPLIED = "not applied"                 # 'not applied - Apply to Hires pass is off'
STATUS_PENDING = "pending"                         # 'pending - sampler wrapped, waiting for sampling'(래퍼가 불리지 않음)
STATUS_FALLBACK = "fell back to plain sampling"    # 'fell back to plain sampling after <예외 이름>: <메시지>'
STATUS_INTERRUPTED = "interrupted"                 # 사용자가 멈췄다
STATUS_KINDS = (STATUS_APPLIED, STATUS_INVALID, STATUS_SKIPPED, STATUS_NOT_APPLIED, STATUS_PENDING, STATUS_FALLBACK,
                STATUS_INTERRUPTED)
# forge_host._write_status 는 '<패스>: <status>' 를 ' | ' 로 잇는다 — 다음 패스 이름 앞의 ' | ' 만 경계로 본다
_STATUS_SPLIT = re.compile(r"\s\|\s(?=(?:%s):\s)" % "|".join(PASS_LABELS))


@dataclass(frozen=True)
class PassStatus:
    """'Anima SPEED status' 의 패스 하나 — ``label`` base·img2img·hires, ``kind`` ``STATUS_*`` 하나(모르는 문구면 ''),
    ``detail`` 머리 뒤 설명(applied 는 요약, fallback 은 '예외 이름: 메시지'), ``text`` 그 패스 status 원문."""

    label: str
    kind: str
    detail: str
    text: str


def _status_kind(text: str) -> tuple:
    for kind in (STATUS_INVALID, STATUS_SKIPPED, STATUS_NOT_APPLIED, STATUS_PENDING, STATUS_INTERRUPTED):
        if text == kind or text.startswith(kind + " - "):
            return kind, text[len(kind) + 3:].strip() if text != kind else ""
    if text.startswith(STATUS_FALLBACK + " after "):
        return STATUS_FALLBACK, text[len(STATUS_FALLBACK) + len(" after "):].strip()
    if text.startswith(STATUS_APPLIED + " (") or text == STATUS_APPLIED:
        _head, _sep, rest = text.partition(" - ")
        return STATUS_APPLIED, rest.strip()
    return "", text


def parse_status(text: Any) -> tuple:
    """``Anima SPEED status`` 값 → 패스마다 ``PassStatus``(forge_host._write_status 의 역 — 패스 이름은
    ``PASS_LABELS`` 만 경계로 본다, 예외 메시지 안의 ' | ' 는 자르지 않는다). 비었거나 읽을 수 없으면 빈 튜플."""
    raw = unquote_param_value(str(text or "")).strip()
    if not raw:
        return ()
    out = []
    for part in _STATUS_SPLIT.split(raw):
        label, sep, value = part.partition(":")
        label = label.strip()
        if not sep or label not in PASS_LABELS:
            continue
        value = value.strip()
        kind, detail = _status_kind(value)
        out.append(PassStatus(label, kind, detail, value))
    return tuple(out)


# ── 검사 ──────────────────────────────────────────────────────────────────────
def _plan_codes(script: ScriptSettings) -> list:
    """설정만 보고 아는 build_plan 의 ``invalid`` 거절(확장 schedule.build_plan 의 순서·조건).

    해상도 비율이 1.0 하나면 확장은 그보다 먼저 'scales has no coarse stage' 로 **건너뛴다**(오류 아님) — 여기서도 없다.
    dwt 비율은 실제로 도는 전환만 본다 — T2I(σ 1 시작)는 모든 전환이 돌고, I2I·Hires 는 앞 단계가 빠지면 더 너그럽다
    (여기는 모든 단계를 본다). 격자·스펙트럼 계산이 안 되는 것(latent 크기·스텝에 달림)은 Forge status 가 말한다."""
    scales = script.scales
    n_transitions = len(scales) - 1
    if n_transitions < 1:
        return []
    codes: list = []
    if script.threshold == "manual":
        values = [float(v) for v in script.manual_sigmas]
        if script.mode == "respace" and len(values) == 1 and n_transitions > 1:
            values = values * n_transitions
        if len(values) != n_transitions:
            codes.append(ERR_MANUAL_COUNT)
    else:
        A, beta, _ref_latent, _ref_shift = preset_params(script.preset, script.spectrum_A, script.spectrum_beta)
        if not (float(script.sigma_divisor) > 0.0):
            codes.append(ERR_DIVISOR)
        elif not (math.isfinite(float(A)) and float(A) > 0.0):
            codes.append(ERR_SPECTRUM_A)
        elif not math.isfinite(float(beta)):
            codes.append(ERR_SPECTRUM_BETA)
    if not codes and script.transform == "dwt":
        for a, b in zip(scales[:-1], scales[1:]):
            if abs(float(b) / float(a) - 2.0) > 1e-6:
                codes.append(ERR_DWT_RATIO)
                break
    return codes


def validate(settings: SpeedSettings) -> tuple:
    """카드 설정의 오류 코드(없으면 빈 튜플). 확장 coerce_settings 의 오류(같은 조건 — ``a/0`` 은 그 칸의 오류이고 수동 σ
    칸은 threshold 가 manual 일 때만, sam-extra 0.33.0) → build_plan 이 설정만 보고 거절하는 것 순."""
    script, codes = _coerce(as_block(settings)["args"])
    if codes:
        return tuple(dict.fromkeys(codes))
    return tuple(_plan_codes(script))


def validation_messages(settings: SpeedSettings) -> tuple:
    """카드 검사 줄(한국어) — ``validate`` 코드 순."""
    return tuple(VALIDATION_MESSAGES[code] for code in validate(settings))


# ── 생성 전에 아는 건너뜀 ─────────────────────────────────────────────────────────
SKIP_INPAINT = "inpaint"
SKIP_CONTROLNET = "controlnet"
SKIP_COMFY_SPECTRUM = "comfy_spectrum"
SKIP_NO_COARSE_STAGE = "no_coarse_stage"
SKIP_MESSAGES = MappingProxyType({
    SKIP_INPAINT: "인페인트·마스크는 마스크가 원래 크기 격자에 묶여 있어 SPEED 를 건너뜁니다",
    SKIP_CONTROLNET: "ControlNet 힌트가 원래 크기 격자에 묶여 있어 SPEED 를 건너뜁니다",
    SKIP_COMFY_SPECTRUM: "Spectrum 실험 기능이 켜져 있어 SPEED 는 건너뜁니다(Forge 의 Spectrum Integrated 와 같은 규칙)",
    SKIP_NO_COARSE_STAGE: "해상도 비율이 1.0 하나라 저해상도 구간이 없어 SPEED 를 건너뜁니다",
})
BACKEND_WEBUI, BACKEND_COMFY, BACKEND_KREA2 = "webui", "comfyui", "krea2"   # core/alwayson_propagation BACKENDS


def _controlnet_enabled(payload: Mapping) -> bool:
    scripts = payload.get("alwayson_scripts")
    if not isinstance(scripts, Mapping):
        return False
    for title, block in scripts.items():
        if _text(title).lower() != "controlnet" or not isinstance(block, Mapping):
            continue
        units = block.get("args")
        for unit in units if isinstance(units, (list, tuple)) else ():
            if isinstance(unit, Mapping) and _truthy(unit.get("enabled", True)):
                return True
    return False


def static_skip_reason(settings: SpeedSettings, payload: Optional[Mapping] = None, *,
                       backend: str = BACKEND_WEBUI) -> Optional[str]:
    """생성 전에 앱이 아는 '건너뜀' 이유 코드(``SKIP_*``), 없으면 None — 카드 상태와 ComfyUI 컴파일러(경고, 노드 없음)가
    쓴다. 확장 runtime_guard·build_plan 과 같은 결과다(그림은 SPEED 없이):

    1 인페인트(``mask`` 또는 ``use_image_alpha_as_mask`` — ComfyUI 백엔드가 inpaint 로 고르는 규칙)  2 ControlNet 블록의
    켜진 유닛  3 ComfyUI 의 Spectrum(``spectrum_enabled`` — 앱 ComfyUI 전용 설정, Forge 의 Spectrum Integrated 처럼 SPEED 가
    비킨다, D6)  4 해상도 비율이 1.0 하나(build_plan 'scales has no coarse stage').
    Anima 레퍼런스 latent 는 앱이 보내지 않는다(P20 미구현) — 그 밖(σ 범위·예측 종류·샘플러)은 Forge status 가 말한다.
    설정 오류(``validate``)는 여기서 보지 않는다 — 부르는 쪽이 먼저 본다."""
    data = payload if isinstance(payload, Mapping) else {}
    mask = data.get("mask")
    has_mask = bool(mask.strip()) if isinstance(mask, str) else bool(mask)
    if has_mask or _truthy(data.get("use_image_alpha_as_mask") or False):
        return SKIP_INPAINT
    if _controlnet_enabled(data):
        return SKIP_CONTROLNET
    if backend == BACKEND_COMFY and _truthy(data.get("spectrum_enabled") or False):
        return SKIP_COMFY_SPECTRUM
    try:
        scales = parse_scales(settings.scales)
    except ValueError:      # a/0 도 ValueError(0.33.0)
        return None
    return SKIP_NO_COARSE_STAGE if len(scales) == 1 else None


# ── 블록 판정 ─────────────────────────────────────────────────────────────────
TARGET_T2I, TARGET_I2I, TARGET_AUX = "t2i", "i2i", "aux"             # core/alwayson_propagation TARGETS
PROVENANCE_USER = "user"                                            # core/alwayson_propagation PROVENANCE_USER
REASON_OFF, REASON_KREA2, REASON_I2I_OFF, REASON_SEND = ("off", "krea2", "i2i_off", "send")


@dataclass(frozen=True)
class SpeedPlan:
    block: Optional[dict]          # alwayson 블록 {"args": [dict]} | None(보내지 않음)
    notice: Any                    # 늘 None — 생성 전 알림 없음(결과 infotext 'Anima SPEED status' 가 정답)
    reason: str                    # off / krea2 / i2i_off / send
    provenance: str = PROVENANCE_USER
    errors: tuple = ()             # validate 코드 — 있어도 보낸다(Forge 가 invalid settings 로 기록하고 평소처럼 샘플링)


def plan(settings: SpeedSettings, *, target: str, backend: str, capabilities: Any = None) -> SpeedPlan:
    """이 요청에 SPEED 블록을 실을지. 스크립트 없음·확인 전은 여기서 보지 않는다(core/alwayson_propagation 게이트).

    1 꺼짐 → 없음  2 Krea2 → 없음  3 i2i 이고 'I2I에도 적용' 끔 → 없음  4 그 밖 → 블록(aux 도 같은 블록 —
    ``blocks_for`` 가 어느 보조 패스에도 주지 않는다). Forge·ComfyUI 같은 규칙(ComfyUI 는 컴파일러가 노드를 넣고 설정
    오류·생성 전에 아는 건너뜀은 경고와 함께 뺀다 — 계획 C4a·C6). 설정 오류도 보낸다(``errors`` 에 코드). 출처는 늘
    사용자 값(앱 기본값이 꺼짐이라 켜진 블록은 사용자가 켠 것)."""
    if not settings.enabled:
        return SpeedPlan(None, None, REASON_OFF)
    if backend == BACKEND_KREA2:
        return SpeedPlan(None, None, REASON_KREA2)
    if target == TARGET_I2I and not settings.apply_img2img:
        return SpeedPlan(None, None, REASON_I2I_OFF)
    errors = validate(settings)
    if errors:
        logger.info("[AnimaSPEED] 설정 오류로 적용되지 않는다(그대로 보내 Forge 는 기록을 남기고, ComfyUI 컴파일러는 "
                    "경고와 함께 노드를 뺀다): %s",
                    "; ".join(VALIDATION_MESSAGES[code] for code in errors))
    return SpeedPlan(as_block(settings), None, REASON_SEND, PROVENANCE_USER, errors)


def describe(settings: SpeedSettings) -> str:
    """로그·알림용 한 줄 — 'transition · neo_shift ÷1.03 δ 0.01 적응형 · anima · 0.5,1.0 · dct', 수동이면
    'respace · manual σ 0.7 · 0.5,1.0 · dct'. Hires·I2I·고정 시드는 켰을 때만, 설정 오류면 앞에 '⚠ 설정 오류 ·'."""
    if settings.threshold == "manual":
        parts = [settings.mode, f"manual σ {settings.manual_sigmas}"]
    else:
        threshold = settings.threshold
        if threshold == "neo_shift":
            threshold += f" ÷{_fmt(settings.sigma_divisor)}"
        threshold += f" δ {_fmt(settings.delta)}" + (" 적응형" if settings.adaptive else "")
        preset = settings.preset
        if preset == PRESET_CUSTOM:
            preset += f" A {_fmt(settings.spectrum_A)} β {_fmt(settings.spectrum_beta)}"
        parts = [settings.mode, threshold, preset]
    parts += [settings.scales, settings.transform]
    if settings.seed >= 0:
        parts.append(f"시드 {settings.seed}")
    if settings.hires:
        parts.append("Hires")
    if settings.apply_img2img:
        parts.append("I2I")
    text = " · ".join(parts)
    return f"⚠ 설정 오류 · {text}" if validate(settings) else text


__all__ = [
    "APP_DEFAULTS", "ARG_ALIASES", "ARG_NAMES", "BACKEND_COMFY", "BACKEND_KREA2", "BACKEND_WEBUI",
    "COMFY_IMG2IMG_RESCALE", "COMFY_INPUTS", "COMFY_MIN_PACK_VERSION", "COMFY_NODE_CLASS", "COMFY_PASS_LABELS",
    "CUSTOM_DEFAULT_A", "CUSTOM_DEFAULT_BETA", "DEFAULTS", "DELTA_MAX", "DELTA_MIN", "DELTA_STEP", "DIVISOR_MAX",
    "DIVISOR_MIN", "DIVISOR_STEP", "ERR_CHOICE", "ERR_DELTA", "ERR_DIVISOR", "ERR_DWT_RATIO", "ERR_MANUAL",
    "ERR_MANUAL_COUNT", "ERR_SCALES", "ERR_SPECTRUM_A", "ERR_SPECTRUM_BETA", "EXTENSION_DEFAULTS",
    "INFOTEXT_IMG2IMG_RESCALE", "INFOTEXT_KEY", "INFOTEXT_KEYS", "INVALID_NOTE", "MODES", "MODE_RESPACE",
    "MODE_TRANSITION", "OPT_IMG2IMG_RESCALE", "OPT_LOG", "PASS_BASE", "PASS_HIRES", "PASS_IMG2IMG", "PASS_LABELS",
    "PASTE_FIELDS", "PRESETS", "PRESET_CUSTOM", "PRESET_NAMES", "PROVENANCE_USER", "PassStatus",
    "REASON_I2I_OFF", "REASON_KREA2", "REASON_OFF", "REASON_SEND", "REFERENCE_LATENT",
    "SCRIPT_NAME", "SETTINGS_KEY", "SKIP_COMFY_SPECTRUM", "SKIP_CONTROLNET", "SKIP_INPAINT", "SKIP_MESSAGES",
    "SKIP_NO_COARSE_STAGE", "STATUS_APPLIED", "STATUS_FALLBACK", "STATUS_INTERRUPTED", "STATUS_INVALID", "STATUS_KEY",
    "STATUS_KINDS", "STATUS_NOT_APPLIED", "STATUS_PENDING", "STATUS_SKIPPED", "TARGET_AUX", "TARGET_I2I", "TARGET_T2I",
    "THRESHOLDS",
    "THRESHOLD_DELTA_OPTIMAL", "THRESHOLD_MANUAL", "THRESHOLD_NEO_SHIFT", "TRANSFORMS", "TRANSFORM_DCT",
    "TRANSFORM_DWT", "TRANSFORM_FFT", "UNSUPPORTED_SAMPLERS", "VALIDATION_MESSAGES", "WIDGET_KEYS", "WIDGET_PREFIX",
    "XYZ_PREFIX", "ScriptSettings", "SpeedPlan", "SpeedSettings", "as_block", "coerce", "describe",
    "from_infotext", "paste_values", "parse_scales", "parse_script_block", "parse_settings", "parse_sigmas",
    "parse_status", "parse_summary", "plan", "preset_params", "script_settings", "static_skip_reason", "summary", "validate",
    "validate_scales", "validation_messages", "widget_id", "widget_values",
]
