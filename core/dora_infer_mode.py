# core/dora_infer_mode.py
"""DoRA Inference Mode(sam-extra ``scripts/dora_infer_mode.py``) — 앱 설정·정규화·블록 판정. 순수 로직
(Qt·네트워크·torch 없음).

확장은 ``alwayson_scripts["DoRA Inference Mode"]`` 로 위치 인자 5개 또는 **dict 한 개**를 받는다
(scripts/dora_infer_mode.py:6-10, coerce_args :112-121). 라벨이 한국어라 앱은 키 값 dict 로 보낸다:
``{"args": [{"enabled": true, "mode": "lycoris", "inserted": "keep", "weak_strength": 0.12, "weak_scope": "attn"}]}``.

- **블록이 없으면 순정이다.** Forge API 는 빠진 스크립트 인자를 ``ui()`` 코드 기본값(아코디언 꺼짐)으로 채우고
  (modules/api/api.py:309-327), ``process()`` 는 꺼짐이면 전역 상태를 (forge, keep) 으로 되돌린 뒤 모델의 합친 상태가
  다르면 LoRA 를 다시 합친다(:147-149, :420-432). 그래서 앱은 **HTTP 요청마다** 블록을 싣는다. 실효 상태가 순정이면
  블록을 빼도 결과가 같다(``plan`` 이 뺀다 — 422 위험이 줄어든다).
- **mode 는 늘 명시한다.** 확장은 빠지거나 모르는 mode 를 순정이 아니라 LyCORIS 로 읽는다(:119).
- 게이트(스크립트 없음·확인 전)는 여기서 하지 않는다 — core/alwayson_propagation 의 DoRA 행(모르면 늘 SKIP)이
  메인·보조 패스에 같은 규칙으로 한다. 여기 ``plan`` 은 대상(t2i·i2i·aux)·백엔드·순정·라이브 선택지만 본다.
- 앱 기본값(``APP_DEFAULTS``)은 사용자 Forge ``ui-config.json`` 의 txt2img 값이다(켬·LyCORIS·그대로 복제·0.12·
  어텐션만, :5669-5686). img2img 아코디언은 꺼짐(:5675)이라 I2I·인페인트는 "I2I·인페인트에도 적용" 토글(기본 끔)을
  따른다. 런타임에 ui-config 를 읽지 않는다(라벨이 키라 깨지기 쉽고 원격 Forge 에는 파일이 없다).
- ComfyUI 는 순정 공식(weight_decompose)과 그대로 복제만 한다 — 블록을 만들지 않는다(컴파일러는 모르는 제목을
  무시한다). 팩 노드로 맞추는 일은 HOLD(P8-C).

``tests/test_dora_infer_mode.py`` 가 상수·정규화·판정을, ``tests/test_sam_extra_contract.py`` 의 SEMANTIC_PINS 가
확장 상수(모드·정책·범위 키, 라벨, infotext 키)와의 일치를 지킨다.
"""
from __future__ import annotations

import logging
import math
import re
from dataclasses import dataclass, replace
from typing import Any, Mapping, Optional

logger = logging.getLogger(__name__)

SCRIPT_NAME = "DoRA Inference Mode"                                          # scripts/dora_infer_mode.py DORA_INFER_NAME
ARG_NAMES = ("enabled", "mode", "inserted", "weak_strength", "weak_scope")   # scripts/dora_infer_mode.py ARG_NAMES

# ── 확장 키(순서도 확장과 같게 — SEMANTIC_PINS 가 비교) ─────────────────────────────
MODE_FORGE, MODE_FORGE_FP32, MODE_LYCORIS, MODE_NO_MAGNITUDE = "forge", "forge_fp32", "lycoris", "no_magnitude"
EXTENSION_MODES = (MODE_FORGE, MODE_FORGE_FP32, MODE_LYCORIS, MODE_NO_MAGNITUDE)  # sam3ext/dora_infer_mode.py MODES
INSERT_KEEP, INSERT_ADDITIVE, INSERT_SKIP, INSERT_WEAK = "keep", "additive", "skip", "weak"
INSERT_POLICIES = (INSERT_KEEP, INSERT_ADDITIVE, INSERT_SKIP, INSERT_WEAK)   # sam3ext/anima_lora_blocks.py DUPLICATE_POLICIES
WEAK_SCOPE_ATTN, WEAK_SCOPE_ATTN_MLP, WEAK_SCOPE_ALL = "attn", "attn_mlp", "all"
WEAK_SCOPES = (WEAK_SCOPE_ATTN, WEAK_SCOPE_ATTN_MLP, WEAK_SCOPE_ALL)         # sam3ext/anima_lora_blocks.py WEAK_SCOPES
DEFAULT_WEAK_STRENGTH = 0.12                                                 # sam3ext/anima_lora_blocks.py DEFAULT_WEAK_STRENGTH
# UI 슬라이더(scripts/dora_infer_mode.py ui() 약한 복사 강도: 0-1, step 0.01). API 는 0-2 로 자른다(_clamp_strength).
WEAK_STRENGTH_RANGE = (0.0, 1.0, 0.01)
_API_STRENGTH_MAX = 2.0
STOCK = (MODE_FORGE, INSERT_KEEP)   # 확장 dim.STOCK_STATE — 블록 없음과 같은 결과

# ── 표시 라벨 = 확장 UI 라벨 그대로(상류 이름 — 번역 금지). 순서 = 확장 라디오 순서 ────────
MODE_OPTIONS = (   # scripts/dora_infer_mode.py MODE_CHOICES
    (MODE_LYCORIS, "LyCORIS (학습과 동일 · fp32)"),
    (MODE_FORGE_FP32, "Forge/Comfy 공식 · fp32"),
    (MODE_FORGE, "Forge/Comfy (순정)"),
    (MODE_NO_MAGNITUDE, "DoRA 끔 (크기 보정 없이 ΔW만 · 일반 LoKr처럼)"),
)
INSERT_OPTIONS = (   # scripts/dora_infer_mode.py DUP_CHOICES
    (INSERT_KEEP, "그대로 복제 (순정 · Forge 기본)"),
    (INSERT_ADDITIVE, "덧셈형 (끼워 넣은 블록만 DoRA 크기 보정 끔)"),
    (INSERT_SKIP, "넣지 않음 (원래 블록에만 · 모든 LoRA)"),
    (INSERT_WEAK, "약한 복사 (브리지식 · 강도·범위 조절)"),
)
SCOPE_OPTIONS = (   # scripts/dora_infer_mode.py SCOPE_CHOICES
    (WEAK_SCOPE_ATTN, "어텐션만 (브리지 기본)"),
    (WEAK_SCOPE_ATTN_MLP, "어텐션+MLP"),
    (WEAK_SCOPE_ALL, "전체 (모듈레이션·노름 포함)"),
)
MODE_LABELS = tuple(label for _key, label in MODE_OPTIONS)
INSERT_LABELS = tuple(label for _key, label in INSERT_OPTIONS)
SCOPE_LABELS = tuple(label for _key, label in SCOPE_OPTIONS)
# 짧은 이름(카드 요약·로그) — frontend/src/utils/doraMode.ts SHORT_LABELS 와 같다
SHORT_LABELS = {
    MODE_LYCORIS: "LyCORIS", MODE_FORGE_FP32: "Forge fp32", MODE_FORGE: "순정", MODE_NO_MAGNITUDE: "DoRA 끔",
    INSERT_KEEP: "그대로 복제", INSERT_ADDITIVE: "덧셈형", INSERT_SKIP: "넣지 않음", INSERT_WEAK: "약한 복사",
    WEAK_SCOPE_ATTN: "어텐션만", WEAK_SCOPE_ATTN_MLP: "어텐션+MLP", WEAK_SCOPE_ALL: "전체",
}

# ── infotext (순정 칸은 두 키를 지운다 — scripts/dora_infer_mode.py _record_state) ─────────
INFOTEXT_MODE_KEY = "DoRA mode"            # sam3ext/dora_infer_mode.py INFOTEXT_KEY
INFOTEXT_INSERT_KEY = "DoRA inserted"      # scripts/dora_infer_mode.py INFOTEXT_DUP_KEY (값: additive / skip / weak 0.12 attn)
INFOTEXT_VALUES = {MODE_FORGE_FP32: "Forge fp32", MODE_LYCORIS: "LyCORIS", MODE_NO_MAGNITUDE: "No magnitude"}

# ── 위젯·저장 ───────────────────────────────────────────────────────────────
WIDGET_KEYS = ("enabled", "mode", "inserted", "weak_strength", "weak_scope", "apply_img2img")
SETTINGS_KEY = "dora_infer_settings"      # prompt_settings.json·생성 프리셋 키


def widget_id(key: str) -> str:
    """위젯 프록시 id = Vue ``storeWidgets`` 키(frontend/src/utils/doraMode.ts WIDGET_IDS)."""
    if key not in WIDGET_KEYS:
        raise KeyError(key)
    return f"_dora_{key}"


# ── 정규화 (확장 규칙의 거울 — 저장값·붙여 넣은 infotext·라이브 라벨 검증용) ──────────────
_MODE_ALIASES = {   # sam3ext/dora_infer_mode.py _ALIASES
    "forge": MODE_FORGE, "comfy": MODE_FORGE, "forge/comfy": MODE_FORGE, "stock": MODE_FORGE, "off": MODE_FORGE,
    "forge_fp32": MODE_FORGE_FP32, "forge fp32": MODE_FORGE_FP32,
    "lycoris": MODE_LYCORIS, "no_magnitude": MODE_NO_MAGNITUDE, "no magnitude": MODE_NO_MAGNITUDE,
}


def _text(value: Any) -> str:
    return str(value if value is not None else "").strip().lower()


def normalize_mode(value: Any) -> Optional[str]:
    """모드 키·UI 라벨·XYZ 값·infotext 값 → 모드 키, 모르면 None (sam3ext/dora_infer_mode.py normalize_mode)."""
    text = _text(value)
    if not text:
        return None
    if text in _MODE_ALIASES:
        return _MODE_ALIASES[text]
    if text.startswith("lycoris"):
        return MODE_LYCORIS
    if text.startswith(("dora off", "dora 끔", "no magnitude", "no_magnitude")):
        return MODE_NO_MAGNITUDE
    if text.startswith(("forge", "comfy")):
        return MODE_FORGE_FP32 if "fp32" in text else MODE_FORGE
    return None


def normalize_insert(value: Any) -> Optional[str]:
    """정책 키·UI 라벨·XYZ 값·infotext 값('weak 0.12 attn') → 정책 키, 모르면 None
    (sam3ext/anima_lora_blocks.py normalize_duplicate_policy — 키 부분 문자열을 keep·additive·skip·weak 순서로)."""
    text = _text(value)
    if not text:
        return None
    for policy in INSERT_POLICIES:
        if policy in text:
            return policy
    if "약한" in text:
        return INSERT_WEAK
    if "덧셈" in text:
        return INSERT_ADDITIVE
    if "넣지 않" in text:
        return INSERT_SKIP
    if "그대로" in text or "순정" in text:
        return INSERT_KEEP
    return None


def normalize_scope(value: Any) -> Optional[str]:
    """범위 키·UI 라벨 → attn / attn_mlp / all, 모르면 None (sam3ext/anima_lora_blocks.py normalize_weak_scope)."""
    text = _text(value)
    if not text:
        return None
    if "mlp" in text:
        return WEAK_SCOPE_ATTN_MLP
    if text.startswith("all") or "전체" in text:
        return WEAK_SCOPE_ALL
    if "attn" in text or "어텐션" in text:
        return WEAK_SCOPE_ATTN
    return None


def _strength(value: Any, high: float) -> Optional[float]:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(number):
        return None
    return round(min(high, max(0.0, number)), 3)


def clamp_strength(value: Any) -> float:
    """약한 복사 강도 — UI 범위 0-1(소수 셋째 자리), 숫자가 아니면 0.12."""
    number = _strength(value, WEAK_STRENGTH_RANGE[1])
    return DEFAULT_WEAK_STRENGTH if number is None else number


def format_strength(value: float) -> str:
    """위젯 문자열 — 0.12 → '0.12', 1.0 → '1'."""
    return f"{float(value):g}"


def _truthy(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    return _text(value) in {"1", "true", "yes", "on"}


def has_lora_tags(text: Any) -> bool:
    """프롬프트에 ``<lora:`` / ``<lyco:`` 태그가 있나 (Comfy 순정 안내 알림 판정)."""
    return bool(re.search(r"<\s*(lora|lyco)\s*:", str(text or ""), re.IGNORECASE))


# ── 설정 ──────────────────────────────────────────────────────────────────────
@dataclass(frozen=True)
class DoraSettings:
    """카드 값. ``apply_img2img`` 는 앱 전용(확장 인자 아님 — Forge 는 img2img 탭에 아코디언이 따로 있다)."""

    enabled: bool = False
    mode: str = MODE_LYCORIS
    inserted: str = INSERT_KEEP
    weak_strength: float = DEFAULT_WEAK_STRENGTH
    weak_scope: str = WEAK_SCOPE_ATTN
    apply_img2img: bool = False

    @property
    def effective(self) -> tuple[str, str]:
        """실제로 합칠 (방식, 정책). 꺼져 있으면 순정 — 확장 resolve_state(:147-149)와 같다."""
        return (self.mode, self.inserted) if self.enabled else STOCK

    @property
    def is_stock(self) -> bool:
        return self.effective == STOCK

    def as_arg(self) -> dict:
        """확장 dict 인자(5키, 정규 키 값). mode 는 늘 싣는다(빠지면 확장이 LyCORIS 로 읽는다)."""
        return {"enabled": bool(self.enabled), "mode": self.mode, "inserted": self.inserted,
                "weak_strength": float(self.weak_strength), "weak_scope": self.weak_scope}


# 확장 API 기본값 = 코드 기본값(ui() — 픽스처 tests/fixtures/sam_extra_script_info.json 의 t2i·i2i 가 같다)
EXTENSION_DEFAULTS = DoraSettings(False, MODE_LYCORIS, INSERT_KEEP, DEFAULT_WEAK_STRENGTH, WEAK_SCOPE_ATTN, False)
# 앱 기본값 = 사용자 Forge ui-config.json txt2img(켬 :5669, LyCORIS :5671, 그대로 복제 :5673, 0.12 :5681, 어텐션만 :5686).
# img2img 아코디언은 꺼짐(:5675) → I2I·인페인트 토글 끔. KNOWN_DIFFS ("DoRA Inference Mode", "enabled", "default").
APP_DEFAULTS = DoraSettings(True, MODE_LYCORIS, INSERT_KEEP, DEFAULT_WEAK_STRENGTH, WEAK_SCOPE_ATTN, False)


def parse_settings(raw: Optional[Mapping], *, base: DoraSettings = APP_DEFAULTS) -> DoraSettings:
    """위젯 문자열·저장 dict → 설정. 없거나 모르는 칸은 ``base``(앱 기본값) 칸을 쓴다."""
    if not isinstance(raw, Mapping):
        return base
    values = {}
    if "enabled" in raw and raw.get("enabled") not in (None, ""):
        values["enabled"] = _truthy(raw.get("enabled"))
    for key, normalize in (("mode", normalize_mode), ("inserted", normalize_insert), ("weak_scope", normalize_scope)):
        normalized = normalize(raw.get(key))
        if normalized is not None:
            values[key] = normalized
    strength = _strength(raw.get("weak_strength"), WEAK_STRENGTH_RANGE[1])
    if strength is not None:
        values["weak_strength"] = strength
    if "apply_img2img" in raw and raw.get("apply_img2img") not in (None, ""):
        values["apply_img2img"] = _truthy(raw.get("apply_img2img"))
    return replace(base, **values)


def widget_values(settings: DoraSettings) -> dict[str, str]:
    """설정 → 위젯 문자열('true'/'false', 키, '0.12')."""
    return {
        "enabled": "true" if settings.enabled else "false",
        "mode": settings.mode,
        "inserted": settings.inserted,
        "weak_strength": format_strength(settings.weak_strength),
        "weak_scope": settings.weak_scope,
        "apply_img2img": "true" if settings.apply_img2img else "false",
    }


def parse_script_block(block: Any) -> Optional[DoraSettings]:
    """보낸 alwayson 블록 → 확장이 읽는 설정(coerce_args·coerce_weak 규칙 — 강도는 API 범위 0-2). 블록이 아니면 None."""
    if not isinstance(block, Mapping):
        return None
    args = block.get("args")
    args = list(args) if isinstance(args, (list, tuple)) else []
    raw = args[0] if args and isinstance(args[0], Mapping) else dict(zip(ARG_NAMES, args))
    strength = _strength(raw.get("weak_strength"), _API_STRENGTH_MAX)
    return DoraSettings(
        enabled=_truthy(raw.get("enabled", False)),
        mode=normalize_mode(raw.get("mode")) or MODE_LYCORIS,
        inserted=normalize_insert(raw.get("inserted")) or INSERT_KEEP,
        weak_strength=DEFAULT_WEAK_STRENGTH if strength is None else strength,
        weak_scope=normalize_scope(raw.get("weak_scope")) or WEAK_SCOPE_ATTN,
    )


def _parse_weak_key(value: Any) -> tuple[Optional[float], Optional[str]]:
    """'weak 0.12 attn' → (0.12, 'attn') — sam3ext/anima_lora_blocks.py parse_weak_key."""
    text = str(value if value is not None else "")
    match = re.search(r"[-+]?\d*\.?\d+", text)
    strength = float(match.group()) if match else None
    rest = text[match.end():] if match else text
    return strength, normalize_scope(rest)


def from_infotext(params: Mapping, *, base: DoraSettings = APP_DEFAULTS) -> DoraSettings:
    """PNG Info 붙여 넣기 — 확장 _paste_*(scripts/dora_infer_mode.py:165-197)의 거울(P16 이 연결한다).

    두 키가 모두 없으면 꺼짐(순정으로 만든 이미지). ``DoRA inserted`` 만 있으면 mode 는 순정이다.
    """
    params = params if isinstance(params, Mapping) else {}
    has_mode, has_insert = INFOTEXT_MODE_KEY in params, INFOTEXT_INSERT_KEY in params
    if not (has_mode or has_insert):
        return replace(base, enabled=False)
    mode = normalize_mode(params.get(INFOTEXT_MODE_KEY)) or (MODE_FORGE if has_insert else base.mode)
    inserted = normalize_insert(params.get(INFOTEXT_INSERT_KEY)) or INSERT_KEEP
    out = replace(base, enabled=True, mode=mode, inserted=inserted)
    if inserted == INSERT_WEAK:
        strength, scope = _parse_weak_key(params.get(INFOTEXT_INSERT_KEY))
        if strength is not None:
            out = replace(out, weak_strength=clamp_strength(strength))
        if scope:
            out = replace(out, weak_scope=scope)
    return out


def describe(settings: DoraSettings) -> str:
    """로그·알림용 한 줄 — 'LyCORIS · 그대로 복제', '순정 · 약한 복사 0.12 어텐션만'."""
    mode, inserted = settings.effective
    text = f"{SHORT_LABELS[mode]} · {SHORT_LABELS[inserted]}"
    if inserted == INSERT_WEAK:
        text += f" {format_strength(settings.weak_strength)} {SHORT_LABELS[settings.weak_scope]}"
    return text


# ── 라이브 선택지 (기능 스냅샷 choices['dora_*'] — core/sam_extra_capabilities._CHOICE_SOURCES) ──
_CHOICE_FIELDS = (("dora_modes", normalize_mode), ("dora_insert_policies", normalize_insert),
                  ("dora_weak_scopes", normalize_scope))


def _live_labels(capabilities: Any, name: str) -> tuple:
    choices = getattr(capabilities, "choices", None)
    if not isinstance(choices, Mapping):
        return ()
    labels = choices.get(name)
    return tuple(str(label) for label in labels) if isinstance(labels, (list, tuple)) else ()


def unknown_live_choices(choices: Any) -> dict[str, tuple]:
    """라이브 선택지 중 앱 정규화가 모르는 라벨(확장이 새 방식을 더했다) — {선택지 이름: 라벨들}."""
    out: dict[str, tuple] = {}
    if not isinstance(choices, Mapping):
        return out
    for name, normalize in _CHOICE_FIELDS:
        labels = choices.get(name)
        if not isinstance(labels, (list, tuple)):
            continue
        unknown = tuple(str(label) for label in labels if normalize(label) is None)
        if unknown:
            out[name] = unknown
    return out


def live_choice_problem(capabilities: Any, settings: DoraSettings) -> Optional[str]:
    """고른 값이 연결된 확장의 선택지에 없으면 사유(없으면 None). 선택지를 모르면 None.

    확장은 모르는 값을 조용히 LyCORIS·그대로 복제·어텐션만으로 읽으므로(coerce_args·coerce_weak) 보내지 않는다.
    """
    mode, inserted = settings.effective
    wanted = [("dora_modes", mode, "계산 방식"), ("dora_insert_policies", inserted, "끼워 넣은 블록")]
    if inserted == INSERT_WEAK:
        wanted.append(("dora_weak_scopes", settings.weak_scope, "약한 복사 범위"))
    normalizers = dict(_CHOICE_FIELDS)
    for name, value, label in wanted:
        labels = _live_labels(capabilities, name)
        if not labels:
            continue
        keys = {normalizers[name](item) for item in labels} - {None}
        if value not in keys:
            return f"{label} '{SHORT_LABELS.get(value, value)}' 을(를) 연결된 Forge 의 DoRA 추론 방식이 모릅니다"
    return None


# ── 블록 판정 ─────────────────────────────────────────────────────────────────
TARGET_T2I, TARGET_I2I, TARGET_AUX = "t2i", "i2i", "aux"             # core/alwayson_propagation TARGETS
BACKEND_WEBUI, BACKEND_COMFY, BACKEND_KREA2 = "webui", "comfyui", "krea2"
PROVENANCE_USER, PROVENANCE_APP_DEFAULT = "user", "app_default"    # core/alwayson_propagation PROVENANCES


def provenance_of(settings: DoraSettings) -> str:
    """카드 값이 앱 기본값 그대로면 'app_default'(앱이 스스로 켠 블록 — 모를 때·없을 때 조용히), 아니면 'user'."""
    return PROVENANCE_APP_DEFAULT if settings == APP_DEFAULTS else PROVENANCE_USER


@dataclass(frozen=True)
class DoraPlan:
    block: Optional[dict]          # alwayson 블록 {"args": [dict]} | None(보내지 않음)
    notice: Any                    # core.sam_extra_notices.Notice | None — 보내는 곳이 띄운다
    reason: str                    # krea2 / i2i_off / stock / comfy / live_choice / send
    provenance: str = PROVENANCE_USER


def plan(settings: DoraSettings, *, target: str, backend: str, capabilities: Any = None,
         has_loras: bool = False) -> DoraPlan:
    """이 요청에 DoRA 블록을 실을지. 스크립트 없음·확인 전은 여기서 보지 않는다(core/alwayson_propagation 게이트).

    1 krea2 → 없음(조용히)  2 i2i 이고 'I2I·인페인트에도 적용' 끔 → 없음(조용히, Forge img2img 아코디언 꺼짐과 같다)
    3 실효 순정 → 없음(조용히, 블록 없음과 결과가 같다)  4 ComfyUI → 없음, 사용자가 바꾼 값이고 LoRA 가 있으면 정보 알림
    5 라이브 선택지에 없는 값 → 없음 + 경고(앱 기본값이면 로그만)  6 그 밖 → 블록
    """
    provenance = provenance_of(settings)
    if backend == BACKEND_KREA2:
        return DoraPlan(None, None, "krea2", provenance)
    if target == TARGET_I2I and not settings.apply_img2img:
        return DoraPlan(None, None, "i2i_off", provenance)
    if settings.is_stock:
        return DoraPlan(None, None, "stock", provenance)
    from core import sam_extra_notices as sn   # sam_extra_notices 가 이 모듈을 import 한다 — 순환을 피해 늦게

    if backend == BACKEND_COMFY:
        notice = None
        if has_loras:
            if provenance == PROVENANCE_USER:
                notice = sn.dora_comfy_stock_notice(describe(settings))
            else:
                logger.info("[DoRA] ComfyUI 는 순정으로 합칩니다(앱 기본값 %s — 알림 없음)", describe(settings))
        return DoraPlan(None, notice, "comfy", provenance)
    problem = live_choice_problem(capabilities, settings)
    if problem:
        if provenance == PROVENANCE_USER:
            return DoraPlan(None, sn.dora_choice_notice(problem), "live_choice", provenance)
        logger.info("[DoRA] 앱 기본값 블록을 보내지 않음: %s", problem)
        return DoraPlan(None, None, "live_choice", provenance)
    return DoraPlan({"args": [settings.as_arg()]}, None, "send", provenance)


__all__ = [
    "APP_DEFAULTS", "ARG_NAMES", "DEFAULT_WEAK_STRENGTH", "DoraPlan", "DoraSettings", "EXTENSION_DEFAULTS",
    "EXTENSION_MODES", "INFOTEXT_INSERT_KEY", "INFOTEXT_MODE_KEY", "INFOTEXT_VALUES", "INSERT_LABELS",
    "INSERT_OPTIONS", "INSERT_POLICIES", "MODE_LABELS", "MODE_OPTIONS", "SCOPE_LABELS", "SCOPE_OPTIONS",
    "SCRIPT_NAME", "SETTINGS_KEY", "SHORT_LABELS", "STOCK", "WEAK_SCOPES", "WEAK_STRENGTH_RANGE", "WIDGET_KEYS",
    "clamp_strength", "describe", "format_strength", "from_infotext", "has_lora_tags", "live_choice_problem",
    "normalize_insert", "normalize_mode", "normalize_scope", "parse_script_block", "parse_settings", "plan",
    "provenance_of", "unknown_live_choices", "widget_id", "widget_values",
]
