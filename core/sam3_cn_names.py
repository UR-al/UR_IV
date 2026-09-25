# core/sam3_cn_names.py
"""SAM3 ControlNet 전처리기·모델 이름 — 라이브 목록 고르기와 대소문자 맞추기 (순수 로직, Qt·네트워크 없음).

Forge 의 ControlNet 은 이름을 **대소문자까지 그대로** 찾는다.
  - 전처리기: ``lib_controlnet/global_state.get_preprocessor`` = ``supported_preprocessors[name]``
  - 모델:     ``get_controlnet_filename``                      = ``controlnet_filename_dict[name]``
그래서 예전 정적 목록의 소문자 ``'none'`` 은 KeyError 로 SAM3 인페인트 패스를 통째로 실패시켰다
(이미지는 SAM3 없이 저장되고 infotext 에 'SAM3 Error' 만 남는다 — gap matrix S6 / P3).

목록의 출처:
  1. sam-extra 기능 스냅샷(``SamExtraCapabilities.choices``)의 ``controlnet_modules`` / ``controlnet_models``
     — 연결 때 Forge ``/controlnet/module_list`` · ``/controlnet/model_list`` 를 GET 으로 받아 둔 것이다.
     모델 목록에는 확장 ``sam3ext/ui.py`` 가 CN 목록에 등록하는 ``models/sam3`` 의 LLLite
     (예: ``anima-lllite-inpainting-v2``)도 들어 있다. 단 모델 목록은 **연결 때** 것이고 모든 모델을 담지
     않는다: 확장은 SAM3 패스마다 ``models/sam3`` 를 다시 스캔해 ``controlnet_filename_dict`` 에만 더한다
     (``sam3ext/inpaint_core.inject_controlnet_unit``). Forge 시작 뒤 넣은 models/sam3 파일, Forge 의
     ControlNet 새로고침(``update_controlnet_filenames`` 가 목록을 다시 만든다) 뒤의 models/sam3 파일은
     목록에 없어도 생성 때 찾는다. 그래서 목록 밖 모델 이름은 막거나 바꾸지 않고 경고만 한다.
     전처리기(``supported_preprocessors``)는 Forge 시작 때 정해지므로 목록 밖 = KeyError 다.
  2. 스냅샷을 모르면(확인 전·실패·ComfyUI) 전처리기는 ``core.sam3_args.CN_MODULES`` 정적 목록을 쓴다 —
     지금 설치된 확장(v0.30.0, Forge classic)의 표기다. 모델은 목록 없이 받은 이름을 그대로 쓰고
     ``'none'`` 만 ``'None'`` 으로 바꾼다. 이 경우 한 번 경고를 남긴다.

페이로드 빌더(``core.sam3_args.build_state``)는 이 모듈로 이름만 맞춘다 — 새 HTTP 요청은 하지 않는다.
Vue 쪽 거울은 ``frontend/src/utils/sam3ControlNet.ts`` (``sam3CnLiveLists`` · ``canonicalCnName``).

Anima ControlNet-LLLite 전처리기 가드(plan §7.2 #7·#14, TR-SAM3) — 이름을 바꾸는 유일한 곳:
  원본(kohya sd-scripts ``anima_minimal_inference_control_net_lllite.py``, kohya ComfyUI-Anima-LLLite)은
  LLLite 에 사용자가 준 제어 이미지(canny·lineart·depth 맵, Tile & Repair 면 고칠 그림)를 그대로 준다.
  3채널이 표준(모든 제어 종류)이고 마스크를 버린다. 4채널은 인페인트(RGB + 마스크)다.
  SAM3 CN 유닛은 인페인트 입력 이미지를 cond 로 쓰므로 lineart·canny·depth LLLite 에는 전처리기가 그 맵을
  만드는 유일한 단계다 — 그래서 필요한 경우에만 바꾼다:
    * Tile & Repair (3채널, 이름에 'tile') → 언제나 ``None`` (기본 ``inpaint_only`` 는 고칠 영역을 −1 로 비운다)
    * 그 밖의 Anima LLLite → ``inpaint_*`` 만 ``None`` — 3채널은 원본이 마스크를 쓰지 않는데 ``inpaint_*`` 가
      마스크 영역을 비우고, 4채널은 ``inpaint_*`` 가 마스크를 버려 LLLite forward 가 실패한다.
      lineart_anime·canny 같은 다른 전처리기는 고른 그대로 보낸다.
    * Anima LLLite 가 아니면(SDXL ``kohya_controllllite_*`` 포함) 건드리지 않는다.
  확장(``sam3ext/sam3_cn_lllite.py``, ``inpaint_core.inject_controlnet_unit``)은 모델 파일 헤더(채널·
  ``modelspec.title``)로 판별해 같은 규칙을 강제한다. 앱은 파일을 못 보므로 **이름**으로만 추정해
  (``lllite_channels_from_name`` · ``lllite_tile_repair_from_name`` — 확장의 이름 폴백과 같은 표) 보내기 전에
  맞추고 경고한다(``guard_lllite_module``). 이름으로 못 알아본 모델(이름을 바꾼 파일)도 확장이 헤더로 다시 잡는다.
  이름 규칙은 'anima' 가 있어야 Anima LLLite 로 본다 — 헤더가 Anima 가 아니면 확장이 전처리기를 그대로 두므로
  앱이 이름만 보고 먼저 바꾸면 안 된다.
"""
from __future__ import annotations

import logging
import re
import threading
from dataclasses import dataclass
from typing import Any, Callable, Iterable, Mapping, MutableMapping, Optional, Sequence

_log = logging.getLogger(__name__)

NONE = "None"                    # Forge 의 '없음' 표기 (전처리기·모델 모두 대문자 N)
DEFAULT_MODULE = "inpaint_only"  # SAM3_SPEC 기본값 = 확장 _default_cn_module 의 첫 선호
CHOICE_MODULES = "controlnet_modules"   # SamExtraCapabilities.choices 키 (/controlnet/module_list)
CHOICE_MODELS = "controlnet_models"     # SamExtraCapabilities.choices 키 (/controlnet/model_list)
STATUS_NONE = "unknown"                 # 스냅샷 자체가 없을 때(확인 전)


@dataclass(frozen=True)
class CnNameLists:
    """SAM3 CN 이름을 맞출 기준.

    modules       드롭다운·정규화 기준 전처리기 목록 (라이브, 없으면 정적 폴백)
    models        라이브 모델 목록. None = 모름 → 자유 입력(받은 이름 그대로)
    live_modules  modules 가 라이브 목록인가
    known         스냅샷이 판단 재료를 가졌나(status ok)
    status        스냅샷 status (스냅샷이 없으면 'unknown')
    """

    modules: tuple
    models: Optional[tuple]
    live_modules: bool
    known: bool
    status: str


def _names(values: Any) -> Optional[tuple]:
    """문자열 목록 → 중복·빈 이름을 뺀 튜플. 목록이 아니거나 비었으면 None."""
    if not isinstance(values, (list, tuple)):
        return None
    names = tuple(dict.fromkeys(v for v in values if isinstance(v, str) and v.strip()))
    return names or None


def _live(capabilities: Any, key: str) -> Optional[tuple]:
    if capabilities is None or not getattr(capabilities, "known", False):
        return None
    choices = getattr(capabilities, "choices", None)
    return _names(choices.get(key)) if isinstance(choices, Mapping) else None


def live_lists(capabilities: Any = None) -> CnNameLists:
    """기능 스냅샷(``SamExtraCapabilities`` | None) → 이름 기준. 모르면 정적 전처리기 목록."""
    from core.sam3_args import CN_MODULES

    modules = _live(capabilities, CHOICE_MODULES)
    return CnNameLists(
        modules=modules or tuple(CN_MODULES),
        models=_live(capabilities, CHOICE_MODELS),
        live_modules=modules is not None,
        known=bool(capabilities is not None and getattr(capabilities, "known", False)),
        status=str(getattr(capabilities, "status", None) or STATUS_NONE),
    )


def canonical(value: Any, names: Iterable[str]) -> Optional[str]:
    """목록에서 같은 이름(정확히 같으면 그것, 아니면 대소문자만 다른 것). 없으면 None."""
    text = "" if value is None else str(value).strip()
    if not text:
        return None
    names = tuple(names)
    if text in names:
        return text
    folded = text.casefold()
    return next((name for name in names if name.casefold() == folded), None)


def normalize_module(value: Any, modules: Sequence[str] = ()) -> str:
    """전처리기 이름을 Forge 표기로. 빈 값 = 기본(inpaint_only), 'none' = 'None', 목록 밖 이름은 그대로."""
    text = "" if value is None else str(value).strip()
    if not text:
        return DEFAULT_MODULE
    match = canonical(text, modules)
    if match is not None:
        return match
    return NONE if text.casefold() == NONE.casefold() else text


def normalize_model(value: Any, models: Optional[Sequence[str]] = None) -> str:
    """모델 이름을 Forge 표기로. 빈 값·'none' = 'None', 라이브 목록이 있으면 대소문자를 목록에 맞춘다."""
    text = "" if value is None else str(value).strip()
    if not text or text.casefold() == NONE.casefold():
        return NONE
    return canonical(text, models or ()) or text


def normalize_saved(key: str, value: Any, lists: Optional[CnNameLists] = None) -> Any:
    """저장된 위젯 값(``cn_module`` / ``cn_model``)의 대소문자만 맞춘다 — 설정 불러오기용.

    빈 값은 그대로 둔다(모델 칸을 비운 상태를 보존한다). 나머지 키는 건드리지 않는다.
    """
    if key not in ("cn_module", "cn_model") or value is None or not str(value).strip():
        return value
    lists = lists or live_lists(None)
    if key == "cn_module":
        return normalize_module(value, lists.modules)
    return normalize_model(value, lists.models)


def name_warnings(state: Mapping[str, Any], lists: CnNameLists) -> list:
    """CN 을 켠 state 의 이름 문제(정규화 뒤에 부른다). CN 이 꺼져 있으면 빈 목록."""
    if not state.get("sam3_cn_enable"):
        return []
    module, model = state.get("sam3_cn_module"), state.get("sam3_cn_model")
    if not lists.known:
        # 문구에 이름을 넣지 않는다 — 상태마다 한 번만 남게(_warn_once 는 문구로 거른다).
        return [f"SAM3 ControlNet: Forge 기능 스냅샷이 없습니다(status={lists.status}) — 지금 설치된 확장 기준 "
                "정적 전처리기 목록으로 이름을 맞춥니다. 모델 이름은 Forge 목록으로 확인하지 못했습니다."]
    out = []
    if not lists.live_modules:
        out.append("SAM3 ControlNet: 연결된 Forge 가 ControlNet 전처리기 목록을 주지 않았습니다"
                   "(sd_forge_controlnet 꺼짐?) — 확장이 CN 주입을 건너뛸 수 있습니다.")
    elif module not in lists.modules:
        out.append(f"SAM3 ControlNet: 전처리기 {module!r} 가 연결된 Forge 에 없습니다 — SAM3 패스가 "
                   "KeyError 로 실패합니다. 목록에서 다시 고르세요.")
    if lists.models is not None and model not in lists.models:
        # 모델 목록은 연결 때 받은 controlnet_names 다. 확장은 SAM3 패스마다 models/sam3 를 다시 스캔해
        # controlnet_filename_dict 에 더하므로(sam3ext/inpaint_core.inject_controlnet_unit) 목록 밖이라도
        # 생성 때 찾을 수 있다 — 실패라고 단정하지 않는다. (전처리기는 Forge 시작 때 정해져 위 문구가 맞다.)
        out.append(f"SAM3 ControlNet: 모델 {model!r} 이 연결 때 받은 Forge ControlNet 목록에 없습니다 — "
                   "models/sam3 의 파일(LLLite 등)이면 확장이 생성 때 다시 찾지만, 그 밖의 이름이면 SAM3 패스가 "
                   "KeyError 로 실패합니다.")
    return out


_warned: set = set()
_warned_lock = threading.Lock()


def _warn_once(message: str) -> None:
    """같은 문구는 프로세스에서 한 번만 — 배치 SAM3 가 이미지마다 같은 경고를 쏟지 않게."""
    with _warned_lock:
        if message in _warned:
            return
        _warned.add(message)
    _log.warning(message)


def lllite_channels_from_name(model: Any) -> Optional[int]:
    """모델 이름 → Anima ControlNet-LLLite cond 채널 추정. LLLite 로 보이지 않으면 None.

    확장 ``sam3ext/sam3_cn_lllite.lllite_channels_from_name`` (헤더를 못 읽을 때의 폴백)과 같은 표다 —
    ``tests/test_sam3_cn_names.py`` 가 설치된 확장 모듈과 대조한다. 구분자·대소문자는 무시한다.
      * 'anima' + 'lllite' + 'inpaint'         → 4  (anima-lllite-inpainting-v2)
      * 'anima' + ('lllite' 또는 'tilerepair') → 3  (animaTileRepair_v20, anima_lllite_lineart_v1)
    'anima' 가 없으면 None — SDXL ``kohya_controllllite_xl_*`` 는 'controllllite' 에 'lllite' 가 있어도 Anima
    LLLite 가 아니다(확장은 헤더에 ``lllite_dit`` 키가 없으면 전처리기를 그대로 둔다).
    """
    flat = _flat(model)
    if not flat or flat == NONE.casefold() or "anima" not in flat:
        return None
    if "lllite" in flat and "inpaint" in flat:
        return 4
    if "lllite" in flat or "tilerepair" in flat:
        return 3
    return None


def _flat(value: Any) -> str:
    """대소문자·구분자를 무시한 비교용 글자(``anima_tile-repair`` = ``animaTileRepair``)."""
    text = "" if value is None else str(value).strip().casefold()
    return re.sub(r"[^0-9a-z]", "", text)


def lllite_tile_repair_from_name(model: Any) -> bool:
    """이름으로 본 3채널 Anima LLLite 가 Tile & Repair 인가 — 이름에 'tile' (animaTileRepair_*, anima_tiled_lllite_*).

    확장은 헤더 ``modelspec.title`` (v1.0 'anima_tiled_lllite_v1', v2.0 'anima_tile_multitask_v1')로 먼저 본다.
    """
    return lllite_channels_from_name(model) == 3 and "tile" in _flat(model)


def lllite_module_override(module: Any, model: Any) -> tuple:
    """(보낼 전처리기, 바꿨다면 경고 문구 | None) — 확장 ``forced_cn_module`` 과 같은 규칙."""
    current = "" if module is None else str(module)
    channels = lllite_channels_from_name(model)
    if channels is None or current == NONE:
        return current, None
    if lllite_tile_repair_from_name(model):
        return NONE, (f"SAM3 ControlNet: {model!r} 은 Anima Tile & Repair LLLite 입니다 — 원본(kohya "
                      f"sd-scripts·ComfyUI-Anima-LLLite)처럼 고칠 그림을 그대로 받으므로 전처리기 "
                      f"{current!r} 대신 'None' 으로 보냅니다(inpaint_* 는 고칠 영역을 비웁니다).")
    if not current.startswith("inpaint"):
        return current, None
    if channels == 4:
        return NONE, (f"SAM3 ControlNet: {model!r} 은 4채널 LLLite 인페인트 모델입니다 — 전처리기 "
                      f"{current!r} 는 마스크를 버려 LLLite 가 실패하므로 'None' 으로 보냅니다.")
    return NONE, (f"SAM3 ControlNet: {model!r} 은 {channels}채널 Anima LLLite 입니다 — 원본은 마스크를 쓰지 "
                  f"않는데 전처리기 {current!r} 는 마스크 영역을 제어 이미지에서 비우므로 'None' 으로 "
                  f"보냅니다(lineart·canny 같은 전처리기는 그대로 씁니다).")


def guard_lllite_module(state: MutableMapping[str, Any], *,
                        warn: Optional[Callable[[str], None]] = None) -> MutableMapping[str, Any]:
    """CN 을 켠 state 의 ``sam3_cn_module`` 을 LLLite 모델에 맞춘다(제자리, 이름 정규화 뒤에 부른다).

    저장된 값·기본값 ``inpaint_only`` 가 Tile & Repair LLLite 와 함께 나가던 것(확장이 영역을 −1 로 비웠다)을
    막는다. 확장도 같은 규칙으로 바꾸므로 결과는 같고, 앱은 보내는 값이 실제 실행과 같도록 맞추고 알린다.
    CN 이 꺼져 있으면 건드리지 않는다(확장도 주입하지 않는다).
    """
    if not state.get("sam3_cn_enable"):
        return state
    module, message = lllite_module_override(state.get("sam3_cn_module"), state.get("sam3_cn_model"))
    if message:
        state["sam3_cn_module"] = module
        (warn or _warn_once)(message)
    return state


def normalize_state(state: MutableMapping[str, Any], capabilities: Any = None, *,
                    warn: Optional[Callable[[str], None]] = None) -> MutableMapping[str, Any]:
    """SAM3 state 의 ``sam3_cn_module`` / ``sam3_cn_model`` 을 Forge 표기로 맞춘다(제자리).

    문제(스냅샷 없음, 목록에 없는 이름)는 ``warn`` (기본: 문구마다 한 번 로그 경고)으로 알린다.
    이름을 다른 것으로 바꾸거나 CN 을 끄지는 않는다 — 대소문자만 맞춘다.
    """
    lists = live_lists(capabilities)
    state["sam3_cn_module"] = normalize_module(state.get("sam3_cn_module"), lists.modules)
    state["sam3_cn_model"] = normalize_model(state.get("sam3_cn_model"), lists.models)
    for message in name_warnings(state, lists):
        (warn or _warn_once)(message)
    return state


__all__ = [
    "CHOICE_MODELS", "CHOICE_MODULES", "CnNameLists", "DEFAULT_MODULE", "NONE", "canonical",
    "guard_lllite_module", "live_lists", "lllite_channels_from_name", "lllite_module_override",
    "name_warnings", "normalize_model", "normalize_module", "normalize_saved", "normalize_state",
]
