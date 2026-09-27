"""Pure parsing and payload rules for the Forge Anima 3.8B Semantic Connector script.

Forge exposes this feature as one always-on script with six positional
arguments.  Keeping coercion here lets both backend adapters share the exact
contract without importing Qt, ComfyUI, or the Forge extension.

P9 — Anima 3.8B 제어 (앱 설정 → 블록):

- ``DEFAULT_SETTINGS``/``ARG_DEFAULTS`` 는 확장 API 기본값 그대로다(블록이 없으면 Forge 는 ``ui()`` 코드 기본값으로
  돈다 — modules/api/api.py init_default_script_args). SEMANTIC_PINS ``anima38_arg_defaults`` 가 지킨다.
- 앱 기본값 ``APP_T2I_DEFAULTS`` 는 사용자 Forge ``ui-config.json`` txt2img 값 중 부정 커넥터 켬(:5452)만 따른다.
  v1 ``enabled``(아코디언, :5443)는 사용자 결정 D1=B 로 끔(``V1_ENABLED_DEFAULT``) — 비 번들 Anima(2.9B·base 1.0·
  UR_ANIMA) 결과는 P9 전과 같다. KNOWN_DIFFS ``("Anima 3.8B (Qwen3.5 / v2)", "negative", "default")``.
  I2I·인페인트는 카드 토글("I2I·인페인트에도 적용", 기본 끔 = Forge img2img 탭 모두 끔 :5460-5475).
- ``effective(settings, kind)``: 모델 종류마다 **블록이 없을 때와 Forge 결과가 달라지는 경우에만** 보낸다(확장
  scripts/anima_3_8b.py process_batch 에서 유도). 영향이 없는 요청은 지금과 바이트 단위로 같다.

  ====== ============================= ===========================================================
  kind   보낼 때                        이유
  ====== ============================= ===========================================================
  other  보내지 않음                    모든 값이 순정 픽셀과 같다. enabled 는 'off: install failed' 흔적만 남긴다
  v2     negative 또는 bypass           enabled·adapter·strength 는 안 쓰이고 negative_strength 는 1.0 고정
  anima  enabled 이고 bypass 아님       v1 은 enabled 일 때만 켜진다(bypass 면 아무것도 하지 않는다)
  unkn.  enabled·negative·bypass 중 하나 사용자가 바꾼 값만 — 앱 기본값은 ``plan`` 이 뺀다(확인 전 비 Anima 요청 보존)
  ====== ============================= ===========================================================

- ``plan(...)`` 은 대상(t2i·i2i·aux)·백엔드·종류로 블록과 출처를 정한다. 스크립트 없음·확인 전 게이트는
  core/alwayson_propagation 의 Anima38 행(앱 기본값은 모르면 SKIP, 사용자 값은 SEND — critic A2)이 한다.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import logging
import math
from types import MappingProxyType
from typing import Any, Iterable, Mapping, Optional, Sequence

from core import anima_model_kind as amk

logger = logging.getLogger(__name__)

SCRIPT_NAME = "Anima 3.8B (Qwen3.5 / v2)"
DEFAULT_ADAPTER = "Anima-3.8B-expanded_adapter.safetensors"
ARG_NAMES = (
    "enabled",
    "adapter",
    "strength",
    "negative",
    "negative_strength",
    "bypass",
)


@dataclass(frozen=True)
class Anima38Settings:
    enabled: bool = False
    adapter: str = DEFAULT_ADAPTER
    strength: float = 1.0
    negative: bool = False
    negative_strength: float = 1.0
    bypass: bool = False

    def as_args(self) -> list[Any]:
        """Return the Forge positional representation in its declared order."""

        return [getattr(self, name) for name in ARG_NAMES]


DEFAULT_SETTINGS = Anima38Settings()
# 확장 ARG_DEFAULTS(scripts/anima_3_8b.py) 와 같다 — SEMANTIC_PINS anima38_arg_defaults 의 "app"
ARG_DEFAULTS = MappingProxyType(asdict(DEFAULT_SETTINGS))

# ── 앱 기본값 (사용자 결정 D1=B) ────────────────────────────────────────────────
# True 로 바꾸면 Forge UI txt2img(아코디언 켬, ui-config.json:5443)와 같아진다 — 비 번들 Anima 에 v1 어댑터·부정
# 어댑터가 걸려 결과가 바뀐다(그 어댑터는 Anima-2.9B-preview-v1-pro52 용). 그때는 KNOWN_DIFFS 에 enabled 행을 더한다.
V1_ENABLED_DEFAULT = False
# txt2img: 부정 커넥터 켬(ui-config.json:5452), Bypass 끔(:5458), 어댑터·강도는 코드 기본값(:5445,:5447,:5453)
APP_T2I_DEFAULTS = Anima38Settings(enabled=V1_ENABLED_DEFAULT, negative=True)

STRENGTH_RANGE = (0.0, 2.0, 0.05)   # 확장 슬라이더(ui() Adapter strength·Negative adapter strength)
# ComfyUI: 카드의 v1 어댑터 선택지는 팩 노드의 콤보(메타데이터로 확인한 어댑터 — vendor/comfyui_anima_3_8b/prompt.py
# adapter_candidates). 아무것도 없으면 팩은 이 이름 하나로 폴백한다 — 파일이 아닌 자리표시자라(고르면 실행 때
# FileNotFoundError) comfy_adapter_choices 가 뺀다(P9 리뷰 2차 1).
COMFY_PROMPT_NODE, COMFY_ADAPTER_INPUT = "ForgeNeoAnimaQwen35Prompt", "adapter_name"
COMFY_ADAPTER_FALLBACK = "qwen35_expanded_adapter.safetensors"
# 팩이 훑는 text_encoders 폴더 목록 — ComfyUI 코어 CLIPLoader.clip_name 은 같은 folder_paths.get_filename_list("text_encoders")
COMFY_TEXT_ENCODER_NODE, COMFY_TEXT_ENCODER_INPUT = "CLIPLoader", "clip_name"
WIDGET_KEYS = (*ARG_NAMES, "apply_img2img")
SETTINGS_KEY = "anima38_settings"   # prompt_settings.json·생성 프리셋 키

KIND_V2, KIND_ANIMA, KIND_OTHER, KIND_UNKNOWN = amk.KIND_V2, amk.KIND_ANIMA, amk.KIND_OTHER, amk.KIND_UNKNOWN
TARGET_T2I, TARGET_I2I, TARGET_AUX = "t2i", "i2i", "aux"             # core/alwayson_propagation TARGETS
BACKEND_WEBUI, BACKEND_COMFY, BACKEND_KREA2 = "webui", "comfyui", "krea2"
PROVENANCE_USER, PROVENANCE_APP_DEFAULT = "user", "app_default"    # core/alwayson_propagation PROVENANCES


def widget_id(key: str) -> str:
    """위젯 프록시 id = Vue ``storeWidgets`` 키(frontend/src/utils/anima38Card.ts WIDGET_IDS)."""
    if key not in WIDGET_KEYS:
        raise KeyError(key)
    return f"_a38_{key}"


def _bool(value: Any, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    return str(value).strip().casefold() in {"1", "true", "yes", "on"}


def _strength(value: Any, default: float = 1.0) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return default
    if not math.isfinite(parsed):
        return default
    return max(0.0, min(2.0, parsed))


def parse_args(raw: Any = None) -> Anima38Settings:
    """Parse Forge positional args or its supported one-dict API form.

    ``raw`` may be the args list itself, a mapping, or ``None``.  Unknown keys
    and trailing positional values are ignored; missing values use the Forge
    defaults.
    """

    if isinstance(raw, Mapping):
        values = dict(raw)
    elif isinstance(raw, Sequence) and not isinstance(raw, (str, bytes, bytearray)):
        items = list(raw)
        if len(items) == 1 and isinstance(items[0], Mapping):
            values = dict(items[0])
        else:
            values = dict(zip(ARG_NAMES, items))
    else:
        values = {}

    adapter = str(values.get("adapter") or DEFAULT_ADAPTER).strip()
    return Anima38Settings(
        enabled=_bool(values.get("enabled"), False),
        adapter=adapter or DEFAULT_ADAPTER,
        strength=_strength(values.get("strength"), 1.0),
        negative=_bool(values.get("negative"), False),
        negative_strength=_strength(values.get("negative_strength"), 1.0),
        bypass=_bool(values.get("bypass"), False),
    )


def parse_script_block(block: Any = None) -> Anima38Settings:
    """Parse an ``alwayson_scripts[SCRIPT_NAME]`` block."""

    if not isinstance(block, Mapping):
        return DEFAULT_SETTINGS
    return parse_args(block.get("args", ()))


# ── 앱 설정 ↔ 위젯 문자열 ──────────────────────────────────────────────────────
def format_strength(value: float) -> str:
    """위젯 문자열 — 1.0 → '1', 0.65 → '0.65'."""
    return f"{float(value):g}"


def settings_from_widgets(values: Optional[Mapping], *,
                          base: Anima38Settings = APP_T2I_DEFAULTS) -> tuple[Anima38Settings, bool]:
    """위젯 문자열·저장 dict → (설정, I2I·인페인트에도 적용). 없거나 빈 칸은 ``base``(앱 기본값), 토글은 끔."""
    if not isinstance(values, Mapping):
        return base, False
    present = {key: values.get(key) for key in ARG_NAMES if values.get(key) not in (None, "")}
    merged = {**asdict(base), **present}
    settings = parse_args(merged)
    apply_img2img = _bool(values.get("apply_img2img"), False) if values.get("apply_img2img") not in (None, "") \
        else False
    return settings, apply_img2img


def widget_values(settings: Anima38Settings, apply_img2img: bool = False) -> dict[str, str]:
    """설정 → 위젯 문자열('true'/'false', 어댑터 이름, '1')."""
    return {
        "enabled": "true" if settings.enabled else "false",
        "adapter": settings.adapter,
        "strength": format_strength(settings.strength),
        "negative": "true" if settings.negative else "false",
        "negative_strength": format_strength(settings.negative_strength),
        "bypass": "true" if settings.bypass else "false",
        "apply_img2img": "true" if apply_img2img else "false",
    }


def as_dict(settings: Anima38Settings) -> dict:
    """확장 dict 인자(6키, ARG_NAMES 순서) — dict 로 보내면 순서가 어긋날 위험이 없다(coerce_args 가 dict 만 읽는다)."""
    return {name: getattr(settings, name) for name in ARG_NAMES}


def describe(settings: Anima38Settings) -> str:
    """로그·알림용 한 줄 — '부정 커넥터 켬 · v1 끔', 'Bypass'."""
    if settings.bypass:
        return "Bypass(순정 Anima)"
    parts = ["부정 커넥터 켬" if settings.negative else "부정 커넥터 끔"]
    parts.append(f"v1 {settings.adapter} {format_strength(settings.strength)}" if settings.enabled else "v1 끔")
    return " · ".join(parts)


# ── 블록 판정 ─────────────────────────────────────────────────────────────────
def effective(settings: Anima38Settings, kind: str) -> Optional[Anima38Settings]:
    """블록이 없을 때와 Forge 결과가 달라지는 경우만 설정을 돌려준다(같으면 None = 보내지 않는다). 모듈 설명의 표."""
    if kind == KIND_OTHER:
        return None
    if kind == KIND_V2:
        return settings if (settings.bypass or settings.negative) else None
    if kind == KIND_ANIMA:
        return settings if (settings.enabled and not settings.bypass) else None
    return settings if (settings.enabled or settings.negative or settings.bypass) else None


def build_block(settings: Anima38Settings, *, kind: str) -> Optional[dict]:
    """``{"args": [6키 dict]}`` 또는 None(``effective`` 가 None)."""
    chosen = effective(settings, kind)
    return None if chosen is None else {"args": [as_dict(chosen)]}


def provenance_of(settings: Anima38Settings, *, target: str = TARGET_T2I) -> str:
    """카드 값이 앱 기본값 그대로면 'app_default'(모를 때 SKIP·없을 때 로그만 — A2·A6), 아니면 'user'.

    I2I·인페인트 블록은 사용자가 'I2I·인페인트에도 적용' 을 켠 것이므로 늘 사용자 값이다."""
    if target == TARGET_I2I:
        return PROVENANCE_USER
    return PROVENANCE_APP_DEFAULT if settings == APP_T2I_DEFAULTS else PROVENANCE_USER


def _comfy_combo(object_info: Any, node_name: str, input_name: str) -> Optional[list[str]]:
    """ComfyUI /object_info 의 콤보 선택지 — 노드가 없거나 모르면 None, 노드에 그 콤보가 없으면 []."""
    if not isinstance(object_info, Mapping):
        return None
    node = object_info.get(node_name)
    if not isinstance(node, Mapping):
        return None
    inputs = node.get("input")
    inputs = inputs if isinstance(inputs, Mapping) else {}
    for section in ("required", "optional"):
        group = inputs.get(section)
        spec = group.get(input_name) if isinstance(group, Mapping) else None
        if isinstance(spec, (list, tuple)) and spec and isinstance(spec[0], (list, tuple)):
            return [str(item) for item in spec[0] if str(item or "").strip()]
    return []


def _comfy_name(value: Any) -> str:
    return str(value or "").strip().replace("\\", "/").casefold()


def comfy_adapter_is_placeholder(choices: Optional[Iterable[Any]], object_info: Any) -> bool:
    """팩 어댑터 콤보가 '팩이 v1 어댑터를 못 찾았다'는 뜻인가 — 자리표시자 이름 하나뿐이고 그 파일이 없다.

    팩(prompt.adapter_candidates)은 찾은 것이 없을 때만 ``COMFY_ADAPTER_FALLBACK`` 하나로 폴백하는데 그 이름은 파일이
    아니다(고르면 큐에 들어간 뒤 실행 때 _checkpoint 가 FileNotFoundError). 그 이름의 진짜 파일이 있으면(업스트림 팩의
    기본 이름) text_encoders 목록(``CLIPLoader.clip_name`` — 팩과 같은 folder_paths 목록)에 보이므로 자리표시자가
    아니다. 그 목록을 모르면 자리표시자로 본다(팩의 폴백은 대개 '없음'이다). 목록에 다른 이름이 섞여 있으면 팩이 찾은
    것이 있다는 뜻이라 늘 False(P9 리뷰 2차 1)."""
    names = [_comfy_name(item) for item in (choices or ()) if _comfy_name(item)]
    fallback = _comfy_name(COMFY_ADAPTER_FALLBACK)
    if names != [fallback]:
        return False
    files = _comfy_combo(object_info, COMFY_TEXT_ENCODER_NODE, COMFY_TEXT_ENCODER_INPUT) or []
    return fallback not in {_comfy_name(item) for item in files}


def comfy_adapter_choices(object_info: Any) -> Optional[list[str]]:
    """ComfyUI /object_info → 카드의 v1 어댑터 선택지(``ForgeNeoAnimaQwen35Prompt.adapter_name`` 콤보), 팩 노드가
    없거나 모르면 None(카드는 Forge 기능 스냅샷 ``choices.anima38_adapters`` 규칙으로). Forge 에는 스냅샷이 있지만
    ComfyUI 에는 없어, 이것이 없으면 카드가 확장 기본 이름밖에 고를 수 없다(컴파일러는 카드 값만 쓴다 — P9 리뷰 2).
    팩이 어댑터를 못 찾아 자리표시자 이름뿐이면 [](카드는 기본 이름 + 저장값, '설치 여부 미확인') — 없는 파일을
    선택지로 내놓지 않는다(``comfy_adapter_is_placeholder`` — P9 리뷰 2차 1)."""
    found = _comfy_combo(object_info, COMFY_PROMPT_NODE, COMFY_ADAPTER_INPUT)
    if found is None:
        return None
    return [] if comfy_adapter_is_placeholder(found, object_info) else found


def _file_name(value: Any) -> str:
    return str(value or "").replace("\\", "/").rsplit("/", 1)[-1].strip().casefold()


def _module_adapter_to_pick(settings: Anima38Settings, modules: Iterable[Any]) -> str:
    """모듈 목록의 v1 어댑터가 하나이고 카드 어댑터와 파일 이름이 다르면 그 이름(v1 을 켤 때 카드에서 골라야 한다)."""
    found = amk.v1_adapter_modules(modules)
    if len(found) != 1:
        return ""
    return found[0] if _file_name(found[0]) != _file_name(settings.adapter) else ""


def _v1_sent(block: Optional[Mapping]) -> bool:
    sent = parse_script_block(block) if block is not None else None
    return bool(sent is not None and sent.enabled and not sent.bypass)


@dataclass(frozen=True)
class Anima38Plan:
    block: Optional[dict]      # alwayson 블록 {"args": [dict]} | None(보내지 않음)
    notice: Any                # core.sam_extra_notices.Notice | None — 보내는 곳이 띄운다
    reason: str                # krea2 / i2i_off / other / not_effective / unknown_default / send
    provenance: str = PROVENANCE_USER
    kind: str = KIND_UNKNOWN


def plan(settings: Anima38Settings, *, target: str, backend: str, kind: str, apply_img2img: bool = False,
         modules: Iterable[Any] = ()) -> Anima38Plan:
    """이 요청에 Anima38 블록을 실을지. 스크립트 없음·확인 전은 여기서 보지 않는다(core/alwayson_propagation 게이트).

    1 krea2 → 없음(조용히)  2 i2i 이고 토글 끔 → 없음(조용히, Forge img2img 탭 기본값)  3 other → 없음
    4 ``effective`` 가 None → 없음  5 종류를 모르는데 앱 기본값 → 없음(확인 전·원격 Forge 의 비 Anima 요청을 그대로
    두려고 — 사용자가 바꾼 값만 보낸다)  6 그 밖 → 블록.
    Comfy: 같은 블록을 컴파일러가 읽는다. 모듈 목록에 Qwen3.5·어댑터 쌍이 있는데 v1 이 가지 않는 요청이면 정보 알림 —
    예전 컴파일러는 그 쌍만으로 v1 을 켰지만 이제 Forge 와 같게 카드의 v1 켜기를 따른다. 원인이 I2I 토글이면 그 토글을
    말하고(P9 리뷰 4), 모듈의 어댑터 이름이 카드 어댑터와 다르면 카드에서 고를 이름을 말한다(P9 리뷰 2).
    """
    provenance = provenance_of(settings, target=target)
    if backend == BACKEND_KREA2:
        return Anima38Plan(None, None, "krea2", provenance, kind)
    notice = None
    if target == TARGET_I2I and not apply_img2img:
        result = Anima38Plan(None, None, "i2i_off", provenance, kind)
    elif kind == KIND_OTHER:
        result = Anima38Plan(None, None, "other", provenance, kind)
    else:
        block = build_block(settings, kind=kind)
        if block is None:
            result = Anima38Plan(None, None, "not_effective", provenance, kind)
        elif kind == KIND_UNKNOWN and provenance == PROVENANCE_APP_DEFAULT:
            logger.debug("[Anima38] 모델 종류를 몰라 앱 기본값 블록을 보내지 않음")
            result = Anima38Plan(None, None, "unknown_default", provenance, kind)
        else:
            result = Anima38Plan(block, None, "send", provenance, kind)
    # Bypass 는 사용자가 순정을 고른 것 — 알릴 것이 없다
    if (backend == BACKEND_COMFY and target != TARGET_AUX and kind in (KIND_ANIMA, KIND_UNKNOWN)
            and not settings.bypass and not _v1_sent(result.block) and amk.has_v1_module_pair(modules)):
        from core import sam_extra_notices as sn   # sam_extra_notices 가 이 모듈을 import 한다 — 순환을 피해 늦게
        notice = sn.anima38_comfy_v1_notice(i2i_off=result.reason == "i2i_off", v1_enabled=settings.enabled,
                                            pick_adapter=_module_adapter_to_pick(settings, modules))
    return replace(result, notice=notice) if notice is not None else result


# ── 붙여 넣기(PNG Info) — 확장 _paste_* 의 거울(P16 이 연결한다) ────────────────────
_STATUS_KEY = "Anima38"          # scripts/anima_3_8b.py STATUS_KEY
_LEGACY_PREFIX = "8B"            # 옛 키 'Anima 3.8B …' 가 Forge 파서에서 잘린 이름


def _param(params: Mapping, suffix: str = "") -> Any:
    for prefix in (_STATUS_KEY, _LEGACY_PREFIX):
        key = f"{prefix} {suffix}".strip()
        if key in params:
            return params[key]
    return None


def _number(value: Any, default: float) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return default if out != out else out


def settings_from_infotext(params: Mapping) -> dict:
    """infotext → 채울 칸만 담은 dict(흔적이 없으면 {}). 확장 _paste_*(scripts/anima_3_8b.py)와 같은 규칙:
    bypass 는 상태가 있을 때, negative 는 'Anima38 negative'(없으면 negative strength 가 있으면 켬), v1 enabled·
    adapter·strength 는 architecture 가 v1 일 때만."""
    params = params if isinstance(params, Mapping) else {}
    out: dict = {}
    status = _param(params)
    if status is not None:
        out["bypass"] = status == "bypass"
    negative = _param(params, "negative")
    if negative is not None:
        out["negative"] = negative == "connector"
    elif _param(params, "negative strength") is not None:
        out["negative"] = True
    negative_strength = _param(params, "negative strength")
    if negative_strength is not None:
        out["negative_strength"] = _number(negative_strength, 1.0)
    architecture = _param(params, "architecture")
    if architecture is not None:
        out["enabled"] = architecture == amk.V1_ADAPTER_ARCHITECTURE
        if architecture == amk.V1_ADAPTER_ARCHITECTURE:
            adapter = _param(params, "adapter")
            if adapter is not None:
                out["adapter"] = adapter
            out["strength"] = _number(_param(params, "strength"), 1.0)
    return out


__all__ = [
    "APP_T2I_DEFAULTS",
    "ARG_DEFAULTS",
    "ARG_NAMES",
    "Anima38Plan",
    "COMFY_ADAPTER_FALLBACK",
    "COMFY_ADAPTER_INPUT",
    "COMFY_PROMPT_NODE",
    "DEFAULT_ADAPTER",
    "DEFAULT_SETTINGS",
    "KIND_ANIMA",
    "KIND_OTHER",
    "KIND_UNKNOWN",
    "KIND_V2",
    "SCRIPT_NAME",
    "SETTINGS_KEY",
    "STRENGTH_RANGE",
    "V1_ENABLED_DEFAULT",
    "WIDGET_KEYS",
    "Anima38Settings",
    "as_dict",
    "build_block",
    "comfy_adapter_choices",
    "comfy_adapter_is_placeholder",
    "describe",
    "effective",
    "format_strength",
    "parse_args",
    "parse_script_block",
    "plan",
    "provenance_of",
    "settings_from_infotext",
    "settings_from_widgets",
    "widget_id",
    "widget_values",
]
