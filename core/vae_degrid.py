# core/vae_degrid.py
"""Anima VAE DeGrid (NAFNet)(sam-extra ``scripts/anima_vae_degrid.py``) — 앱 설정·정규화·블록 판정. 순수 로직
(Qt·네트워크·torch 없음).

확장은 ``alwayson_scripts["Anima VAE DeGrid (NAFNet)"]`` 로 위치 인자 ``[enabled, model, mode, strength, tile]`` 또는
**dict 한 개**를 받는다(sam3ext/ui_vae_degrid.py ARG_NAMES·coerce_args). 앱은 키 이름 dict 로 보낸다 — Forge 가 캐시한
위치 인자 기본값과 무관하게 같은 뜻이 된다:
``{"args": [{"enabled": true, "model": "", "mode": "full", "strength": 1.0, "tile": 512}]}``.

- **최종 이미지 후처리다.** 확장은 ``postprocess_image_after_composite``(모든 후처리·인페인트 합성 뒤, 저장 직전)에서
  이미지마다 한 번 돌고, SAM3·ADetailer 내부 패스(``_sam3_inner``·``_ad_inner``)에서는 돌지 않는다. 그래서 블록은
  **메인 요청에만** 싣는다(T2I·대기열·XYZ·시드 탐색·채팅·만화 컷, I2I·인페인트·채팅 편집은 카드 토글). 보조 작업
  (Refine·단독/배치 SAM3·ADetailer·손 재구성)은 각자 새 요청이라 넣으면 이미 DeGrid 된 이미지에 한 번 더 걸린다 —
  core/alwayson_propagation 의 DeGrid 행이 ``passes=frozenset()`` 이라 ``blocks_for`` 가 어느 보조 패스에도 주지 않는다.
- 게이트(스크립트 없음·확인 전)는 여기서 하지 않는다 — 같은 행(사용자 값이라 모를 때는 보냄, 없다고 확인되면 빼고
  ``block_not_sent`` 경고)이 한다. 여기 ``plan`` 은 켬·대상·백엔드만 본다.
- 모델 ``''`` = 자동(확장이 찾은 첫 파일 — ``modelspec.version`` 이 높은 것 먼저). 기능 스냅샷의 ``degrid_models`` 는
  **Forge 가 시작할 때** 만든 목록이다(생성할 때 확장은 새 파일도 찾는다) — 목록에 없다고 미리 막지 않는다. 결과
  infotext(``Anima DeGrid model`` / ``Anima DeGrid error``)가 정답이다(core/sam_extra_notices 가 장마다 읽는다).
- 장치·GPU 정밀도·VRAM 상주는 Forge 설정(``sam3_degrid_*``)이다 — ComfyUI 노드는 확장 기본값(auto·fp32·끔)을 쓴다
  (``COMFY_OPTIONS``).
- 앱 기본값 = 확장 기본값(꺼짐·자동·Full·1·512). 사용자 Forge ui-config 도 같아 첫 로드 안내가 없다.

확장 코드는 GPL-3.0 이라 옮기지 않았다 — 규칙(범위·별칭·자르기)을 다시 구현했다. ``tests/test_vae_degrid_core.py`` 가
상수·정규화·판정을, 레지스트리 SEMANTIC_PINS 가 확장 상수와의 일치를 지킨다.
"""
from __future__ import annotations

import json
import logging
import math
from dataclasses import dataclass, replace
from types import MappingProxyType
from typing import Any, Iterable, Mapping, Optional

logger = logging.getLogger(__name__)

# ── 확장 계약 ─────────────────────────────────────────────────────────────────
SCRIPT_NAME = "Anima VAE DeGrid (NAFNet)"                        # sam3ext/ui_vae_degrid.py TITLE
EXTRAS_TITLE = "Anima VAE DeGrid (NAFNet, Extras)"               # sam3ext/ui_vae_degrid.py EXTRAS_TITLE (API 없음 — 앱은 무시)
ARG_NAMES = ("enabled", "model", "mode", "strength", "tile")     # sam3ext/ui_vae_degrid.py ARG_NAMES

MODE_FULL, MODE_DARK, MODE_BRIGHT = "full", "dark", "bright"
MODES = (MODE_FULL, MODE_DARK, MODE_BRIGHT)                      # sam3ext/vae_degrid.py MODES
DEFAULT_MODE = MODE_FULL
# 확장 UI 라벨 그대로(상류 이름 — 번역 금지). 순서 = 확장 라디오 순서(sam3ext/ui_vae_degrid.py MODE_CHOICES)
MODE_CHOICES = (
    "Full (전체)",
    "Dark Pixels Mainly (어두운 점 위주)",
    "Bright Pixels Mainly (밝은 점 위주)",
)
# 노드 팩 이름 = infotext 값(sam3ext/vae_degrid.py MODE_LABELS) — ComfyUI 노드의 mode 선택지도 이것이다
MODE_LABELS = MappingProxyType({MODE_FULL: "Full", MODE_DARK: "Dark Pixels Mainly", MODE_BRIGHT: "Bright Pixels Mainly"})
# 짧은 이름(로그·카드 요약) — frontend/src/utils/vaeDegrid.ts 와 같다
SHORT_MODE = MappingProxyType({MODE_FULL: "Full", MODE_DARK: "Dark", MODE_BRIGHT: "Bright"})

STRENGTH_MIN, STRENGTH_MAX, STRENGTH_STEP, DEFAULT_STRENGTH = 0.0, 1.5, 0.05, 1.0   # sam3ext/vae_degrid.py, UI step
MIN_TILE, MAX_TILE, DEFAULT_TILE, TILE_OVERLAP = 128, 4096, 512, 32                 # sam3ext/vae_degrid.py
TILE_STEP = MIN_TILE        # sam3ext/ui_vae_degrid.py TILE_SLIDER_STEP (0 과 128 단위)

NONE_NAME = "None"          # sam3ext/vae_degrid_models.py NONE_NAME — 모델이 없을 때 확장 목록의 자리 표시
AUTO = ""                   # 앱의 '자동'(확장: 비우거나 'None'/'auto' = 첫 파일)
MODEL_FOLDERS = ("ESRGAN", "DeGrid")   # sam3ext/vae_degrid_models.py MODEL_FOLDERS — 이름이 겹치면 '폴더/이름'
_MODEL_EXTENSIONS = (".safetensors", ".pth", ".pt", ".ckpt", ".bin")

# infotext 키(sam3ext/ui_vae_degrid.py) — 성공이면 model·mode·strength·tile(+precision), 실패면 error 하나만
KEY_MODEL = "Anima DeGrid model"
KEY_MODE = "Anima DeGrid mode"
KEY_STRENGTH = "Anima DeGrid strength"
KEY_TILE = "Anima DeGrid tile"            # 실제로 쓴 타일(OOM 으로 줄였으면 줄인 값)
KEY_PRECISION = "Anima DeGrid precision"  # fp32 / fp16-autocast — 설정이라 붙여 넣지 않는다(기록만)
KEY_ERROR = "Anima DeGrid error"
RESULT_KEYS = (KEY_MODEL, KEY_MODE, KEY_STRENGTH, KEY_TILE, KEY_PRECISION)

# 실패 이유의 앞부분(sam3ext/vae_degrid_runtime.py·scripts/anima_vae_degrid.py) — 종류만 가른다(문구는 앱이 다시 쓴다)
ERROR_MODEL_NOT_FOUND = "model not found"
ERROR_NOT_RESIDUAL = "not a DeGrid residual model"
ERROR_BLEW_UP = "output blew up"

# Forge 설정(sam3ext/vae_degrid_runtime.py) — 앱은 P10 덮어쓰기(Forge 전용)로만 다룬다
OPT_DEVICE = "sam3_degrid_device"
OPT_GPU_PRECISION = "sam3_degrid_gpu_precision"
OPT_KEEP_LOADED = "sam3_degrid_keep_loaded"
DEVICE_AUTO, DEVICE_CPU = "auto", "cpu"
PRECISION_FP32, PRECISION_FP16 = "fp32", "fp16"
DEVICE_CHOICES = (DEVICE_AUTO, DEVICE_CPU)
PRECISION_CHOICES = (PRECISION_FP32, PRECISION_FP16)
DEFAULT_DEVICE, DEFAULT_PRECISION, DEFAULT_KEEP_LOADED = DEVICE_AUTO, PRECISION_FP32, False
EXTENSION_OPTION_DEFAULTS = MappingProxyType(
    {"device": DEFAULT_DEVICE, "precision": DEFAULT_PRECISION, "keep_loaded": DEFAULT_KEEP_LOADED})

# ── ComfyUI (ai_studio_forge_parity 팩 노드 — 앱 컴파일러가 만든다) ────────────────
COMFY_NODE_CLASS = "ForgeNeoAnimaVAEDeGrid"
COMFY_INPUTS = ("image", "enabled", "model_name", "mode", "strength", "tile", "device", "precision",
                "keep_loaded", "forge_quantize")
COMFY_UI_KEY = "ai_studio_degrid"   # 노드 실행 결과 ui 키 — 이미지마다 리포트 dict
COMFY_AUTO = "auto"                 # 노드의 model_name '자동'
COMFY_OPTIONS = EXTENSION_OPTION_DEFAULTS   # Forge 설정은 ComfyUI 에 해당 없음 — 확장 기본값과 같은 계산(D6)
COMFY_FORGE_QUANTIZE = True         # 입력을 Forge 처럼 8비트로 내려 넣고 출력도 8비트 격자에 맞춘다(D9)
COMFY_MIN_PACK_VERSION = "1.5.0"    # 노드가 처음 들어온 팩(core/comfy_node_pack.PACK_VERSION 은 이 이상)
# ComfyUI 컴파일러가 DeGrid 노드를 빼고 생성한 원인(컴파일 경고의 ``cause`` — core/comfy_workflow_compiler).
# 앞의 셋은 /object_info 를 새로 받으면 풀릴 수 있다(팩 갱신 뒤 재시작·새 모델 파일). 넣을 자리가 없는 사용자 워크플로
# (VAEDecode·출력 노드 없음, 서로 다른 출력 여럿)는 스키마와 무관하다 — 새로 받아도 같다.
COMFY_OMIT_NODE_MISSING = "node_missing"     # 노드 클래스 없음(팩 1.5.0 이전이거나 갱신 뒤 재시작 전)
COMFY_OMIT_NODE_CONTRACT = "node_contract"   # 노드 입력이 COMFY_INPUTS 와 다름
COMFY_OMIT_MODEL_MISSING = "model_missing"   # 고른 모델(자동이면 첫 파일)이 노드 목록에 없음
COMFY_OMIT_PLACEMENT = "placement"           # 사용자 워크플로에 넣을 자리가 없음
COMFY_SCHEMA_FIXABLE = frozenset({COMFY_OMIT_NODE_MISSING, COMFY_OMIT_NODE_CONTRACT, COMFY_OMIT_MODEL_MISSING})
# 노드 팩의 카테고리 접두 → Forge 폴더 이름(겹칠 때 확장이 붙이는 'ESRGAN/…'·'DeGrid/…')
_COMFY_FOLDER_PREFIX = MappingProxyType({"upscale_models": "ESRGAN", "degrid": "DeGrid"})

# ── 위젯·저장 ───────────────────────────────────────────────────────────────
WIDGET_KEYS = ("enabled", "model", "mode", "strength", "tile", "apply_img2img")
SETTINGS_KEY = "vae_degrid_settings"      # prompt_settings.json·생성 프리셋 키


def widget_id(key: str) -> str:
    """위젯 프록시 id = Vue ``storeWidgets`` 키(frontend/src/utils/vaeDegrid.ts WIDGET_IDS)."""
    if key not in WIDGET_KEYS:
        raise KeyError(key)
    return f"_degrid_{key}"


# ── 정규화 (확장 규칙의 거울 — 저장값·붙여 넣은 infotext·보낸 블록 해석용) ─────────────
_MODE_ALIASES = {   # sam3ext/vae_degrid.py _MODE_ALIASES (노드 팩 patch 설정값·짧은 별칭)
    "full": MODE_FULL,
    "dark": MODE_DARK, "dark_pixels": MODE_DARK, "dark pixels": MODE_DARK,
    "bright": MODE_BRIGHT, "bright_pixels": MODE_BRIGHT, "bright pixels": MODE_BRIGHT,
}


def _text(value: Any) -> str:
    return str(value if value is not None else "").strip()


def normalize_mode(value: Any) -> Optional[str]:
    """모드 키·노드 팩 이름·UI 라벨('Full (전체)' 처럼 괄호 설명이 붙은 것) → 모드 키, 모르면 None
    (sam3ext/vae_degrid.py normalize_mode 와 같은 규칙 — 대소문자 무시)."""
    text = _text(value).lower()
    if not text:
        return None
    if "(" in text:
        text = text.split("(", 1)[0].strip()
    if text in _MODE_ALIASES:
        return _MODE_ALIASES[text]
    for key, label in MODE_LABELS.items():
        if text == label.lower():
            return key
    return None


def coerce_mode(value: Any) -> str:
    """모드 → 키. 모르면 Full(확장 coerce_args 와 같다 — 모르는 값을 Full 로 읽는다)."""
    return normalize_mode(value) or DEFAULT_MODE


def _finite(value: Any) -> Optional[float]:
    if isinstance(value, bool):
        return float(value)
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return None if math.isnan(number) else number


def coerce_strength(value: Any, default: float = DEFAULT_STRENGTH) -> float:
    """강도 → [0, 1.5]. 읽을 수 없거나 NaN 이면 ``default`` (sam3ext/vae_degrid.py coerce_strength)."""
    number = _finite(value)
    if number is None:
        return float(default)
    return min(max(number, STRENGTH_MIN), STRENGTH_MAX)


def coerce_tile(value: Any, default: int = DEFAULT_TILE) -> int:
    """타일 → 0(나누지 않음) 또는 [128, 4096](1~127 은 128). 읽을 수 없으면 ``default`` (coerce_tile).
    무한대는 확장에서 OverflowError 로 죽지만 앱은 ``default`` 로 둔다(앱이 그런 값을 보내지 않게)."""
    try:
        tile = int(float(value))
    except (TypeError, ValueError, OverflowError):
        return int(default)
    if tile <= 0:
        return 0
    return min(max(tile, MIN_TILE), MAX_TILE)


def normalize_model(value: Any) -> str:
    """모델 이름 → 저장·전송 값. None·''·'None'·'auto'(대소문자 무시)는 자동(''). 그 밖은 앞뒤 공백만 뗀다."""
    text = _text(value)
    return AUTO if text.lower() in ("", NONE_NAME.lower(), COMFY_AUTO) else text


def format_strength(value: Any) -> str:
    """강도 문자열 — 1.0 → '1', 0.85 → '0.85' (sam3ext/ui_vae_degrid.py format_strength·위젯 값)."""
    return f"{round(float(value), 3):g}"


def _truthy(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    return _text(value).lower() in {"1", "true", "yes", "on"}


# ── 설정 ──────────────────────────────────────────────────────────────────────
@dataclass(frozen=True)
class DegridSettings:
    """카드 값. ``apply_img2img`` 는 앱 전용(확장 인자 아님 — Forge 는 img2img 탭에 아코디언이 따로 있다)."""

    enabled: bool = False
    model: str = AUTO
    mode: str = DEFAULT_MODE
    strength: float = DEFAULT_STRENGTH
    tile: int = DEFAULT_TILE
    apply_img2img: bool = False

    def as_arg(self) -> dict:
        """확장 dict 인자(5키, 정규 값)."""
        return {"enabled": bool(self.enabled), "model": self.model, "mode": self.mode,
                "strength": float(self.strength), "tile": int(self.tile)}


# 확장 API 기본값 = ui() 기본값(픽스처 tests/fixtures/sam_extra_script_info.json 의 t2i·i2i 가 같다 — 모델 기본값은
# 목록의 첫 파일 = 자동). 사용자 Forge ui-config 도 같아(꺼짐·v11·Full·1·512) 앱 기본값이 같다 — KNOWN_DIFFS 없음.
EXTENSION_DEFAULTS = DegridSettings(False, AUTO, DEFAULT_MODE, DEFAULT_STRENGTH, DEFAULT_TILE, False)
APP_DEFAULTS = EXTENSION_DEFAULTS


def parse_settings(raw: Optional[Mapping], *, base: DegridSettings = APP_DEFAULTS) -> DegridSettings:
    """위젯 문자열·저장 dict → 설정. 없거나 읽을 수 없는 칸은 ``base`` 칸을 쓴다(모델 ''= 자동은 값이다)."""
    if not isinstance(raw, Mapping):
        return base
    values: dict = {}
    for key in ("enabled", "apply_img2img"):
        if key in raw and raw.get(key) not in (None, ""):
            values[key] = _truthy(raw.get(key))
    if "model" in raw and raw.get("model") is not None:
        values["model"] = normalize_model(raw.get("model"))
    mode = normalize_mode(raw.get("mode"))
    if mode is not None:
        values["mode"] = mode
    if _finite(raw.get("strength")) is not None:
        values["strength"] = coerce_strength(raw.get("strength"))
    if raw.get("tile") not in (None, "") and coerce_tile(raw.get("tile"), default=-1) != -1:
        values["tile"] = coerce_tile(raw.get("tile"))
    return replace(base, **values)


def widget_values(settings: DegridSettings) -> dict[str, str]:
    """설정 → 위젯 문자열('true'/'false', 모델 이름, 모드 키, '1', '512')."""
    return {
        "enabled": "true" if settings.enabled else "false",
        "model": settings.model,
        "mode": settings.mode,
        "strength": format_strength(settings.strength),
        "tile": str(int(settings.tile)),
        "apply_img2img": "true" if settings.apply_img2img else "false",
    }


def as_block(settings: DegridSettings) -> dict:
    """alwayson 블록 — dict 형식 인자 하나."""
    return {"args": [settings.as_arg()]}


def parse_script_block(block: Any) -> Optional[DegridSettings]:
    """보낸 alwayson 블록 → 확장이 읽는 설정(coerce_args 규칙). 블록이 아니면 None.

    위치 인자·dict 모두 읽는다. 뒤 인자가 빠지면 확장 기본값이다. ``apply_img2img`` 는 앱 전용이라 늘 False."""
    if not isinstance(block, Mapping):
        return None
    args = block.get("args")
    args = list(args) if isinstance(args, (list, tuple)) else []
    raw = dict(args[0]) if args and isinstance(args[0], Mapping) else dict(zip(ARG_NAMES, args))
    return DegridSettings(
        enabled=_truthy(raw.get("enabled", False)),
        model=normalize_model(raw.get("model")),
        mode=coerce_mode(raw.get("mode")),
        strength=coerce_strength(raw.get("strength", DEFAULT_STRENGTH)),
        tile=coerce_tile(raw.get("tile", DEFAULT_TILE)),
    )


def _unquote(value: Any) -> str:
    """infotext 값 — 쉼표가 든 값은 Forge 가 JSON 따옴표로 감싼다(파서가 못 벗겼으면 여기서)."""
    text = _text(value)
    if len(text) >= 2 and text[0] == text[-1] == '"':
        try:
            decoded = json.loads(text)
        except ValueError:
            return text[1:-1]
        return str(decoded)
    return text


def infotext_error(params: Any) -> str:
    """결과 infotext 의 ``Anima DeGrid error`` 원문(따옴표를 벗긴 것), 없으면 ''."""
    if not isinstance(params, Mapping) or params.get(KEY_ERROR) in (None, ""):
        return ""
    return " ".join(_unquote(params.get(KEY_ERROR)).split())


def from_infotext(params: Any, *, base: DegridSettings = APP_DEFAULTS) -> DegridSettings:
    """PNG Info 붙여 넣기 — 확장 paste_*(sam3ext/ui_vae_degrid.py)의 거울(P16 이 연결한다).

    ``Anima DeGrid model`` 이 있을 때만 켠다(실패한 이미지는 오류 키만 남으므로 꺼짐 — 오류는 ``infotext_error`` 로
    보여 줄 뿐 붙여 넣지 않는다). 모드는 이름→키, 강도·타일은 확장 규칙으로 자른다. 정밀도는 설정이라 무시한다.
    ``apply_img2img`` 는 ``base`` 값 그대로."""
    params = params if isinstance(params, Mapping) else {}
    if KEY_MODEL not in params:
        return replace(base, enabled=False)
    values: dict = {"enabled": True, "model": normalize_model(_unquote(params.get(KEY_MODEL)))}
    mode = normalize_mode(_unquote(params.get(KEY_MODE)))
    if mode is not None:
        values["mode"] = mode
    if KEY_STRENGTH in params:
        values["strength"] = coerce_strength(_unquote(params.get(KEY_STRENGTH)))
    if KEY_TILE in params:
        values["tile"] = coerce_tile(_unquote(params.get(KEY_TILE)))
    return replace(base, **values)


def tile_text(tile: int) -> str:
    return "타일 없음" if int(tile) <= 0 else str(int(tile))


def describe(settings: DegridSettings) -> str:
    """로그·알림용 한 줄 — 'Full 1 · 512', 'Dark 0.8 · 타일 없음 · qwenVAEDegridNafnet_v11'(모델은 자동이 아닐 때만).
    카드 요약(frontend/src/utils/vaeDegrid.ts summary)과 같은 문구다."""
    text = f"{SHORT_MODE.get(settings.mode, settings.mode)} {format_strength(settings.strength)} · {tile_text(settings.tile)}"
    return f"{text} · {settings.model}" if settings.model else text


# ── 모델 목록 ─────────────────────────────────────────────────────────────────
def _strip_model_extension(name: str) -> str:
    lower = name.lower()
    for extension in _MODEL_EXTENSIONS:
        if lower.endswith(extension):
            return name[: -len(extension)]
    return name


def _last_segment(name: str) -> str:
    return name.replace("\\", "/").rsplit("/", 1)[-1]


def resolve_name(name: Any, names: Iterable[Any]) -> Optional[str]:
    """이름 → 목록의 이름(sam3ext/vae_degrid_models.py resolve 의 거울 — 경로 없이 이름만). 자동이면 첫 이름.

    정확히 → 대소문자 무시 → 확장자·폴더 접두를 뗀 파일 이름 순. 목록이 비었거나 없으면 None."""
    listed = [str(item) for item in (names or ()) if str(item or "").strip() and str(item) != NONE_NAME]
    if not listed:
        return None
    wanted = normalize_model(name)
    if not wanted:
        return listed[0]
    for item in listed:
        if item == wanted:
            return item
    lowered = wanted.lower()
    for item in listed:
        if item.lower() == lowered:
            return item
    stem = _strip_model_extension(_last_segment(wanted)).lower()
    for item in listed:
        if _strip_model_extension(_last_segment(item)).lower() == stem:
            return item
    return None


def live_models(capabilities: Any) -> Optional[list]:
    """Forge 기능 스냅샷의 DeGrid 모델 목록(``choices['degrid_models']``, 'None' 자리 표시는 뺀다).

    스냅샷을 모르거나 목록을 못 읽었으면 None. Forge 시작 때 만든 목록이다(생성은 새 파일도 찾는다)."""
    if getattr(capabilities, "known", None) is not True:
        return None
    choices = getattr(capabilities, "choices", None)
    names = choices.get("degrid_models") if isinstance(choices, Mapping) else None
    if not isinstance(names, (list, tuple)):
        return None
    return [str(name) for name in names if isinstance(name, str) and name.strip() and name != NONE_NAME]


MODEL_OK, MODEL_MISSING, MODEL_UNKNOWN, MODEL_NONE_INSTALLED = "ok", "missing", "unknown", "none_installed"


def check_model(settings: DegridSettings, capabilities: Any) -> str:
    """고른 모델이 Forge 목록에 있나 — 'ok' | 'missing' | 'unknown'(목록 모름) | 'none_installed'(파일 없음)."""
    models = live_models(capabilities)
    if models is None:
        return MODEL_UNKNOWN
    if not models:
        return MODEL_NONE_INSTALLED
    return MODEL_OK if resolve_name(settings.model, models) is not None else MODEL_MISSING


def _combo_choices(spec: Any) -> Optional[list]:
    """object_info 입력 스펙 → 콤보 선택지. 옛 모양 ``[[…], {…}]`` 과 새 모양 ``["COMBO", {"options": […]}]``."""
    if not isinstance(spec, (list, tuple)) or not spec:
        return None
    head = spec[0]
    if isinstance(head, (list, tuple)):
        return [str(item) for item in head]
    if head == "COMBO" and len(spec) > 1 and isinstance(spec[1], Mapping):
        options = spec[1].get("options")
        if isinstance(options, (list, tuple)):
            return [str(item) for item in options]
    return None


def comfy_node_choices(object_info: Any) -> Optional[list]:
    """노드의 model_name 선택지(원문 — 'auto' 포함, 컴파일러가 그대로 쓴다). 노드가 없거나 입력 계약이
    ``COMFY_INPUTS`` 와 다르면(옛 팩) None."""
    if not isinstance(object_info, Mapping):
        return None
    node = object_info.get(COMFY_NODE_CLASS)
    inputs = node.get("input") if isinstance(node, Mapping) else None
    required = inputs.get("required") if isinstance(inputs, Mapping) else None
    if not isinstance(required, Mapping) or set(required) != set(COMFY_INPUTS):
        return None
    return _combo_choices(required.get("model_name"))


def comfy_stem(choice: Any) -> str:
    """ComfyUI 파일 이름(카테고리 기준 상대 경로) → Forge 식 이름. 팩이 겹침 때 붙이는 ``upscale_models/``·``degrid/``
    는 확장처럼 ``ESRGAN/``·``DeGrid/`` 로, 그 밖의 하위 폴더는 떼고 확장자를 뗀다."""
    text = _text(choice).replace("\\", "/")
    head, _sep, rest = text.partition("/")
    if rest and head in _COMFY_FOLDER_PREFIX:
        return f"{_COMFY_FOLDER_PREFIX[head]}/{_strip_model_extension(_last_segment(rest))}"
    return _strip_model_extension(_last_segment(text))


def comfy_model_choices(object_info: Any) -> Optional[list]:
    """ComfyUI object_info → 카드에 보일 모델 이름(Forge 식, 'auto' 뺌, 순서 유지). 노드 클래스가 없거나(옛 팩)
    입력 계약이 ``COMFY_INPUTS`` 와 다르면 None — 카드는 '팩 업데이트 필요' 를, 컴파일러는 노드를 뺀다."""
    choices = comfy_node_choices(object_info)
    if choices is None:
        return None
    out: list = []
    for choice in choices:
        if choice.strip().lower() in (COMFY_AUTO, "", NONE_NAME.lower()):
            continue
        stem = comfy_stem(choice)
        if stem and stem not in out:
            out.append(stem)
    return out


def comfy_file_for(stem: Any, choices: Iterable[Any]) -> Optional[str]:
    """앱이 저장한 이름(Forge 식) → 노드 model_name 값(원문 선택지). 자동('')이면 ``'auto'``, 못 찾으면 None.

    원문 그대로 → Forge 식 이름 정확히 → 대소문자 무시 → 폴더 접두·확장자를 뗀 파일 이름 순."""
    wanted = normalize_model(stem)
    if not wanted:
        return COMFY_AUTO
    raw = [str(item) for item in (choices or ()) if str(item or "").strip().lower() not in (COMFY_AUTO, "")]
    for choice in raw:
        if choice == wanted:
            return choice
    stems = [(choice, comfy_stem(choice)) for choice in raw]
    for choice, name in stems:
        if name == wanted:
            return choice
    lowered = wanted.lower()
    for choice, name in stems:
        if name.lower() == lowered:
            return choice
    bare = _strip_model_extension(_last_segment(wanted)).lower()
    for choice, name in stems:
        if _last_segment(name).lower() == bare:
            return choice
    return None


def report_params(report: Any) -> dict:
    """ComfyUI 노드 리포트 한 장 → Forge 와 같은 infotext 키 dict(결과 알림을 같은 규칙으로 — A9).

    오류가 있으면 ``Anima DeGrid error`` 하나만(Forge record_failure 와 같다). 꺼짐이면 빈 dict. 그 밖(ok·강도 0 건너뜀)
    은 모델·모드·강도·타일, 정밀도는 '-' 가 아닐 때만."""
    if not isinstance(report, Mapping):
        return {}
    error = " ".join(_text(report.get("error")).split())
    if error:
        return {KEY_ERROR: error}
    if _text(report.get("status")).lower() == "off":
        return {}
    mode = coerce_mode(report.get("mode"))
    params = {
        KEY_MODEL: _text(report.get("model")),
        KEY_MODE: MODE_LABELS[mode],
        KEY_STRENGTH: format_strength(coerce_strength(report.get("strength"))),
        KEY_TILE: coerce_tile(report.get("tile")),
    }
    precision = _text(report.get("precision"))
    if precision and precision != "-":
        params[KEY_PRECISION] = precision
    return params


# ── 블록 판정 ─────────────────────────────────────────────────────────────────
TARGET_T2I, TARGET_I2I, TARGET_AUX = "t2i", "i2i", "aux"             # core/alwayson_propagation TARGETS
BACKEND_WEBUI, BACKEND_COMFY, BACKEND_KREA2 = "webui", "comfyui", "krea2"
PROVENANCE_USER = "user"                                            # core/alwayson_propagation PROVENANCE_USER


@dataclass(frozen=True)
class DegridPlan:
    block: Optional[dict]          # alwayson 블록 {"args": [dict]} | None(보내지 않음)
    notice: Any                    # 늘 None — 생성 전 알림 없음(결과 infotext 가 정답)
    reason: str                    # off / krea2 / i2i_off / send
    provenance: str = PROVENANCE_USER


def plan(settings: DegridSettings, *, target: str, backend: str, capabilities: Any = None) -> DegridPlan:
    """이 요청에 DeGrid 블록을 실을지. 스크립트 없음·확인 전은 여기서 보지 않는다(core/alwayson_propagation 게이트).

    1 꺼짐 → 없음  2 Krea2 → 없음(카드가 상태로 말한다)  3 i2i 이고 'I2I·인페인트에도 적용' 끔 → 없음
    4 그 밖 → 블록(aux 도 t2i 와 같은 블록 — 봉투가 메인 체인과 같게(T15). ``blocks_for`` 가 어느 보조 패스에도 주지
    않는다). 출처는 늘 사용자 값(앱 기본값이 꺼짐이라 켜진 블록은 사용자가 켠 것).

    Forge 기능 스냅샷의 모델 목록에 없다고 막지 않는다 — 그 목록은 Forge 시작 때 것이고 확장은 생성할 때 새 파일도
    찾는다. 로그만 남기고, 실제 결과는 결과 알림(``Anima DeGrid error``)과 카드 상태가 말한다."""
    if not settings.enabled:
        return DegridPlan(None, None, "off")
    if backend == BACKEND_KREA2:
        return DegridPlan(None, None, "krea2")
    if target == TARGET_I2I and not settings.apply_img2img:
        return DegridPlan(None, None, "i2i_off")
    if backend == BACKEND_WEBUI and target != TARGET_AUX and check_model(settings, capabilities) == MODEL_MISSING:
        logger.info("[DeGrid] '%s' 이(가) Forge 시작 때 목록에 없음 — 그대로 보낸다(생성할 때 확장이 다시 찾는다)",
                    settings.model)
    return DegridPlan(as_block(settings), None, "send")


__all__ = [
    "APP_DEFAULTS", "ARG_NAMES", "AUTO", "COMFY_AUTO", "COMFY_FORGE_QUANTIZE", "COMFY_INPUTS", "COMFY_NODE_CLASS",
    "COMFY_MIN_PACK_VERSION", "COMFY_OMIT_MODEL_MISSING", "COMFY_OMIT_NODE_CONTRACT", "COMFY_OMIT_NODE_MISSING",
    "COMFY_OMIT_PLACEMENT", "COMFY_OPTIONS", "COMFY_SCHEMA_FIXABLE", "COMFY_UI_KEY", "DEFAULT_DEVICE", "DEFAULT_KEEP_LOADED", "DEFAULT_MODE", "DEFAULT_PRECISION",
    "DEFAULT_STRENGTH", "DEFAULT_TILE", "DEVICE_AUTO", "DEVICE_CHOICES", "DEVICE_CPU", "DegridPlan", "DegridSettings",
    "ERROR_BLEW_UP", "ERROR_MODEL_NOT_FOUND", "ERROR_NOT_RESIDUAL", "EXTENSION_DEFAULTS", "EXTENSION_OPTION_DEFAULTS",
    "EXTRAS_TITLE", "KEY_ERROR", "KEY_MODE", "KEY_MODEL", "KEY_PRECISION", "KEY_STRENGTH", "KEY_TILE", "MAX_TILE",
    "MIN_TILE", "MODEL_FOLDERS", "MODEL_MISSING", "MODEL_NONE_INSTALLED", "MODEL_OK", "MODEL_UNKNOWN", "MODES",
    "MODE_BRIGHT", "MODE_CHOICES", "MODE_DARK", "MODE_FULL", "MODE_LABELS", "NONE_NAME", "OPT_DEVICE",
    "OPT_GPU_PRECISION", "OPT_KEEP_LOADED", "PRECISION_CHOICES", "PRECISION_FP16", "PRECISION_FP32", "RESULT_KEYS",
    "SCRIPT_NAME", "SETTINGS_KEY", "SHORT_MODE", "STRENGTH_MAX", "STRENGTH_MIN", "STRENGTH_STEP", "TILE_OVERLAP",
    "TILE_STEP", "WIDGET_KEYS", "as_block", "check_model", "coerce_mode", "coerce_strength", "coerce_tile",
    "comfy_file_for", "comfy_model_choices", "comfy_node_choices", "comfy_stem", "describe", "format_strength", "from_infotext",
    "infotext_error", "live_models", "normalize_mode", "normalize_model", "parse_script_block", "parse_settings",
    "plan", "report_params", "resolve_name", "tile_text", "widget_id", "widget_values",
]
