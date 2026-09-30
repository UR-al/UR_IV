# core/forge_override_settings.py
"""sam-extra Forge 옵션을 요청마다 덮어쓰기(``override_settings``) — 순수 로직(Qt·네트워크 없음, P10).

Forge 는 요청의 ``override_settings`` 를 ``process_images`` 시작 때 콜백 없이·저장 없이 적용하고 끝나면 되돌린다
(modules/processing.py:800-813, :843-846). config.json 은 바뀌지 않는다 — 앱은 ``POST /sdapi/v1/options`` 를 쓰지 않는다.

- **기본 = "Forge 설정 따름"(키를 보내지 않음)**: Forge 옵션은 API 요청에도 사용자 config.json 값이 그대로 적용되므로
  아무것도 보내지 않는 것이 곧 Forge UI 와 같은 동작이다(사용자 결정 D3). 기본 상태에서 요청은 P10 전과 바이트 단위로 같다.
- 설정은 ``ui_prefs.forgeOptionOverrides`` = {옵션 키: bool}. 없는 키 = 따름. **진짜 bool 만** 받는다 — Forge 는 타입을 보지
  않아 문자열 ``"False"`` 가 확장의 ``bool(getattr(...))`` 에서 True 로 뒤집힌다(scripts/anima_safe_pag.py _read_bool_option).
- **모르면 보내지 않는다**(``may_use`` 의 '모르면 보냄'과 반대): 등록되지 않은 키는 값이 다르면 KeyError 로 요청 전체가
  HTTP 500 이다(modules/options.py:165, modules/api/api.py handle_exception). 키마다 기능 스냅샷의 ``has_option``(Gradio
  ``/config`` 의 ``setting_sam3_*`` — 등록 여부와 같다)이 True 일 때만 보낸다. 모르면 ``unverified``, 없으면 ``missing``.
- 호출자가 보낸 ``override_settings`` 키가 이긴다. ``override_settings_restore_afterwards`` 는 쓰지 않고, 호출자가 True 가
  아닌 값을 보냈으면 앱 키를 넣지 않는다(앱 설정이 Forge 메모리에 남지 않게).
- 생성 워커는 GUI 가 밀어 넣은 값만 읽는다(``update_forge_option_overrides_from_prefs`` — core/forge_output_policy 와 같은
  이유: 워커가 ui_prefs.json 을 열면 GUI 의 os.replace 저장이 Windows 에서 실패한다). 파일 읽기는 한 번도 받지 못했을 때의
  폴백이다.

그래도 Forge 가 거절하면(낡은 스냅샷 — 500 KeyError, 설정 잠금 AssertionError) ``core/forge_optional_parts.plan_retry`` 가
앱 키를 빼고 한 번 더 보낸다(``WebUIBackend._generate``·``_run_img2img_postprocess``). 업스케일(extra-single-image)·
Tile & Repair 라우트·ComfyUI 는 해당 없다.

설정 잠금(``--freeze-settings``·``--freeze-settings-in-sections``·``--freeze-specific-settings``)은 Forge 시작 인자라 기능
스냅샷(script-info·/config)에 드러나지 않는다 — 새로 받아도 옵션은 '있다'로 남아 요청마다 거절·재시도를 되풀이한다.
그래서 잠금으로 거절된 키는 주소별로 기억해(``remember_frozen_options``) 다음 요청부터 보내지 않는다(``plan_overrides`` 의
``frozen`` — P10 검토 3). 기억하는 것은 Forge 가 이름을 댄 키뿐이다 — 섹션·키 잠금은 일부만 잠그므로 재시도에서 함께 뺀
나머지는 다음 요청부터 다시 보낸다(전역 잠금만 앱 키 전부, ``RetryPlan.frozen`` — P10 검토 4). 백엔드 변경·재연결과
사용자의 수동 새로고침이 잊게 한다(``forget_frozen_options``).
"""
from __future__ import annotations

import json
import os
import threading
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Iterable, Mapping, Optional

PREF_KEY = "forgeOptionOverrides"

# 묶음 — 카드의 세 칸(frontend/src/utils/forgeOptionOverrides.ts GROUPS 와 같은 순서)
MEMORY, RESULT_MINOR, RESULT = "memory", "result_minor", "result"
GROUPS = (MEMORY, RESULT_MINOR, RESULT)

# 확장 옵션 키 — 요청 중에 이 이름으로 읽는 확장 상수와 같아야 한다(core/sam_extra_contract SEMANTIC_PINS 가 지킨다)
OPT_KEEP_RESIDENT = "sam3_anima38_keep_resident"            # sam3ext/anima38/runtime.py
OPT_CONNECTOR_FP32 = "sam3_anima38_connector_fp32"          # sam3ext/anima38/connector_fp32.py
OPT_CONNECTOR_RUN_CACHE = "sam3_anima38_connector_run_cache"   # sam3ext/anima38/connector_cache.py
OPT_UNLOAD_KEEP_IN_RAM = "sam3_unload_keep_in_ram"          # sam3ext/core.py
OPT_PREFIX_DEDUP = "sam3_guidance_pag_prefix_dedup"         # scripts/anima_safe_pag.py
OPT_SEG_SEPARABLE = "sam3_guidance_seg_separable_blur"      # scripts/anima_safe_pag.py
OPT_SPARSE_FORGE_GUESS = "sam3_anima_sparse_lora_forge_guess"  # sam3ext/anima_lora_blocks.py
OPT_DAVE_PRE_DD = "sam3_guidance_dave_pre_dd_sigma"         # scripts/anima_safe_pag.py (8878b9e)

# 결과 infotext 이름(Forge 붙여넣기에서 같은 옵션으로 되돌아가는 이름)
INFOTEXT_PREFIX_DEDUP = "Anima PAG prefix dedup"
INFOTEXT_SEG_SEPARABLE = "Anima SEG separable blur"
INFOTEXT_SPARSE_GUESS = "Anima sparse LoRA"
INFOTEXT_DAVE_PRE_DD = "Anima DAVE pre-DD sigma"


@dataclass(frozen=True)
class ForgeOptionSpec:
    """앱이 요청마다 덮어쓸 수 있는 확장 옵션 한 줄.

    default    확장 기본값(Forge config 에 값이 없을 때) — 표시·계약 테스트용. 앱은 이 값을 보내지 않는다
    effect     MEMORY(결과 같음) / RESULT_MINOR(아주 미세하게 다름) / RESULT(결과가 달라짐)
    infotext   결과 infotext 이름(재현·확인용). 없으면 ''
    feature    관련 기능(표시용): anima38 / sam3 / anima_guidance / lora
    onchange   확장이 onchange 콜백을 둔 옵션 — 요청 적용 때는 돌지 않고 복원 때 돈다(modules/processing.py:813, :846)
    """

    key: str
    label: str
    default: bool
    effect: str
    infotext: str = ""
    feature: str = ""
    onchange: bool = False


# 순서 = 카드 순서. 모두 bool — 레퍼런스 IP-Adapter 옵션 둘은 자기 잡이라 요청 override 가 닿지 않는다(P20), 나머지
# 다섯은 Forge 화면 전용이다. VAE DeGrid 옵션 셋은 기능이 보류(HOLD)라 앱에 없다(core/sam_extra_contract OPTIONS).
SPECS: tuple[ForgeOptionSpec, ...] = (
    ForgeOptionSpec(OPT_KEEP_RESIDENT, "Anima 3.8B 모델 VRAM 상주", True, MEMORY, feature="anima38"),
    ForgeOptionSpec(OPT_CONNECTOR_FP32, "Anima 3.8B 커넥터 fp32 상주", True, MEMORY, feature="anima38"),
    ForgeOptionSpec(OPT_CONNECTOR_RUN_CACHE, "Anima 3.8B 커넥터 실행 캐시", True, MEMORY, feature="anima38"),
    ForgeOptionSpec(OPT_UNLOAD_KEEP_IN_RAM, "SAM3 모델 RAM 보관", True, MEMORY, feature="sam3", onchange=True),
    ForgeOptionSpec(OPT_PREFIX_DEDUP, "PAG 앞 블록 중복 계산 건너뛰기", True, RESULT_MINOR,
                    infotext=INFOTEXT_PREFIX_DEDUP, feature="anima_guidance"),
    ForgeOptionSpec(OPT_SEG_SEPARABLE, "SEG 블러 1D 분리 계산", True, RESULT_MINOR,
                    infotext=INFOTEXT_SEG_SEPARABLE, feature="anima_guidance"),
    ForgeOptionSpec(OPT_SPARSE_FORGE_GUESS, "부분 LoRA 순정 추측 변환", False, RESULT,
                    infotext=INFOTEXT_SPARSE_GUESS, feature="lora"),
    # 끄면 원본 ComfyUI 노드 조합처럼 DAVE 가 Detail Daemon 이 줄인 σ 를 '0번 스텝'으로 읽어 모든 스텝에 걸린다(이미지가
    # 무너진다). Detail Daemon 을 끈 생성은 켜고 끔이 같다. 앱 Comfy 팩(guid_dave_pre_dd, 1.4.1)도 같은 기본값이다.
    ForgeOptionSpec(OPT_DAVE_PRE_DD, "DAVE 판정을 Detail Daemon 전 σ 로 (우회)", True, RESULT,
                    infotext=INFOTEXT_DAVE_PRE_DD, feature="anima_guidance"),
)
SPEC_BY_KEY: Mapping[str, ForgeOptionSpec] = MappingProxyType({spec.key: spec for spec in SPECS})
OPTION_KEYS: tuple[str, ...] = tuple(spec.key for spec in SPECS)

# 확인하지 못한 이유(알림 문구)
UNVERIFIED_UNKNOWN = "unknown"          # 기능 스냅샷이 없다(연결 직후·확인 전·수집 실패)
UNVERIFIED_NO_CONFIG = "no_config"      # 스냅샷은 있는데 Gradio /config 를 읽지 못했다(--gradio-auth·--nowebui)


def label_of(key: str) -> str:
    spec = SPEC_BY_KEY.get(key)
    return spec.label if spec else str(key)


def normalize_overrides(raw: Any) -> dict:
    """ui_prefs 값 → {스펙 키: bool}(스펙 순서). 값이 진짜 bool 이 아니거나(1·'true'·None) 스펙 밖 키면 버린다(= 따름).
    입력은 바꾸지 않는다."""
    if not isinstance(raw, Mapping):
        return {}
    return {key: raw[key] for key in OPTION_KEYS if key in raw and isinstance(raw[key], bool)}


@dataclass(frozen=True)
class OverridePlan:
    send: Mapping[str, bool] = field(default_factory=dict)   # 보낼 것 — 키가 있다고 확인된 것만
    missing: tuple = ()          # 사용자가 정했는데 이 Forge 의 sam-extra 에 없는 키(보내지 않음)
    unverified: tuple = ()       # 스냅샷·/config 가 없어 확인하지 못한 키(보내지 않음)
    reason: str = ""             # unverified 이유(UNVERIFIED_*)
    frozen: tuple = ()           # 이 Forge 가 설정 잠금으로 거절했던 키(보내지 않음 — ``remember_frozen_options``)

    def __bool__(self) -> bool:
        return bool(self.send or self.missing or self.unverified or self.frozen)


def plan_overrides(overrides: Any, capabilities: Any, *, frozen: Iterable[str] = ()) -> OverridePlan:
    """사용자 설정 × 기능 스냅샷 → 무엇을 보낼지. 스냅샷을 모르면 아무것도 보내지 않는다(모르는 키는 500).

    ``frozen``: 이 Forge 가 설정 잠금으로 거절했던 키(``frozen_options(api_url)``) — 스냅샷과 무관하게 보내지 않는다."""
    wanted = normalize_overrides(overrides)
    if not wanted:
        return OverridePlan()
    locked = set(frozen or ())
    held = tuple(key for key in wanted if key in locked)
    wanted = {key: value for key, value in wanted.items() if key not in locked}
    if not wanted:
        return OverridePlan(frozen=held)
    if getattr(capabilities, "known", None) is not True:
        return OverridePlan(unverified=tuple(wanted), reason=UNVERIFIED_UNKNOWN, frozen=held)
    has_option = getattr(capabilities, "has_option", None)
    if not callable(has_option):
        return OverridePlan(unverified=tuple(wanted), reason=UNVERIFIED_UNKNOWN, frozen=held)
    send: dict = {}
    missing, unverified = [], []
    for key, value in wanted.items():
        try:
            present = has_option(key)
        except Exception:
            present = None
        if present is True:
            send[key] = value
        elif present is False:
            missing.append(key)
        else:
            unverified.append(key)
    return OverridePlan(MappingProxyType(send), tuple(missing), tuple(unverified),
                        UNVERIFIED_NO_CONFIG if unverified else "", held)


def merge_into_payload(payload: Mapping, plan: OverridePlan) -> tuple[dict, tuple]:
    """(새 payload, 넣은 앱 키). 입력은 바꾸지 않는다. 넣을 것이 없으면 ``override_settings`` 키를 만들지 않는다.

    호출자의 ``override_settings`` 키가 이긴다. 호출자가 ``override_settings_restore_afterwards`` 를 True 가 아닌 값으로
    보냈으면(False·null — Forge 가 되돌리지 않는다) 앱 키를 넣지 않는다. 그 플래그는 쓰지 않는다. 호출자의
    ``override_settings`` 가 dict 도 None 도 아니면 건드리지 않는다.
    """
    out = dict(payload or {})
    send = dict(getattr(plan, "send", None) or {})
    if not send:
        return out, ()
    if "override_settings_restore_afterwards" in out and out["override_settings_restore_afterwards"] is not True:
        return out, ()
    existing = out.get("override_settings")
    if existing is None:
        merged: dict = {}
    elif isinstance(existing, Mapping):
        merged = dict(existing)
    else:
        return out, ()
    sent = []
    for key, value in send.items():
        if key not in SPEC_BY_KEY or not isinstance(value, bool) or key in merged:
            continue
        merged[key] = value
        sent.append(key)
    if sent:
        out["override_settings"] = merged
    return out, tuple(sent)


def without_app_keys(payload: Mapping, keys: Iterable[str]) -> dict:
    """앱이 넣은 키를 뺀 새 payload — 남은 것이 없으면 ``override_settings`` 키도 뺀다(앱이 넣기 전과 같게).

    호출자의 키는 ``merge_into_payload`` 가 앱 키와 겹치지 않게 했으므로 그대로 남는다. 입력은 바꾸지 않는다."""
    out = dict(payload or {})
    drop = set(keys or ())
    existing = out.get("override_settings")
    if not drop or not isinstance(existing, Mapping):
        return out
    kept = {key: value for key, value in existing.items() if key not in drop}
    if kept:
        out["override_settings"] = kept
    else:
        out.pop("override_settings", None)
    return out


def plan_notices(plan: OverridePlan) -> list:
    """보내지 못한 설정 알림(결과 info 에 싣는다). 사용자가 정한 값만 여기 오므로(앱 기본값 없음 — D3) 출처 구분이 없다."""
    if not plan or not (plan.missing or plan.unverified or plan.frozen):
        return []
    from core import sam_extra_notices as sn

    out = []
    if plan.frozen:
        out.append(sn.forge_option_frozen_notice([label_of(key) for key in plan.frozen]))
    if plan.missing:
        out.append(sn.forge_option_missing_notice([label_of(key) for key in plan.missing]))
    if plan.unverified:
        out.append(sn.forge_option_unverified_notice([label_of(key) for key in plan.unverified],
                                                     no_config=plan.reason == UNVERIFIED_NO_CONFIG))
    return out


# ── GUI 가 밀어 넣는 설정값 ─────────────────────────────────────────────────────
class _PrefsOverridesCache:
    """ui_prefs.json 의 ``forgeOptionOverrides`` 를 mtime 기준으로 캐시 — 밀어 넣기 전 폴백 전용."""

    def __init__(self, path_factory=None):
        self._path_factory = path_factory
        self._lock = threading.Lock()
        self._stamp: Optional[tuple] = None
        self._value: dict = {}

    def _path(self) -> str:
        if self._path_factory is not None:
            return str(self._path_factory())
        from core.ui_prefs import ui_prefs_path
        return ui_prefs_path()

    def get(self) -> dict:
        try:
            path = self._path()
            stat = os.stat(path)
        except (OSError, ValueError):
            return {}
        stamp = (path, stat.st_mtime, stat.st_size)
        with self._lock:
            if stamp == self._stamp:
                return dict(self._value)
        try:
            with open(path, "r", encoding="utf-8") as handle:
                prefs = json.load(handle)
        except (OSError, ValueError):
            return {}
        value = normalize_overrides(prefs.get(PREF_KEY)) if isinstance(prefs, dict) else {}
        with self._lock:
            self._stamp, self._value = stamp, value
        return dict(value)


_cache = _PrefsOverridesCache()


def forge_option_overrides_from_prefs_file() -> dict:
    """현재 ui_prefs.json 의 ``forgeOptionOverrides``(정규화, 없거나 못 읽으면 {}). 생성 경로는
    :func:`forge_option_overrides_setting` 을 쓴다 — 여기는 GUI 가 아직 값을 밀어 넣지 않았을 때의 폴백이다."""
    return _cache.get()


class _PushedOverrides:
    """GUI 스레드가 밀어 넣는 값 — 생성 워커 스레드는 메모리 값만 읽는다(core/forge_output_policy._PushedFlag 와 같은 이유)."""

    def __init__(self, fallback):
        self._fallback = fallback
        self._lock = threading.Lock()
        self._value: Optional[dict] = None

    def set(self, overrides: Any) -> None:
        value = normalize_overrides(overrides)
        with self._lock:
            self._value = value

    def reset(self) -> None:
        with self._lock:
            self._value = None

    def is_set(self) -> bool:
        with self._lock:
            return self._value is not None

    def get(self) -> dict:
        with self._lock:
            value = self._value
        if value is not None:
            return dict(value)
        return normalize_overrides(self._fallback())


# 폴백은 호출 시점에 이름으로 찾는다(테스트가 모듈 함수를 바꿔 끼울 수 있게).
_setting = _PushedOverrides(lambda: forge_option_overrides_from_prefs_file())


def set_forge_option_overrides(overrides: Any) -> None:
    """설정값을 메모리에 반영(GUI 스레드: save_ui_prefs 핸들러·부팅 복원, 테스트)."""
    _setting.set(overrides)


def update_forge_option_overrides_from_prefs(prefs: Any) -> None:
    """병합된 ui_prefs dict 로 설정값 갱신 — 키가 없으면 전부 따름."""
    _setting.set(prefs.get(PREF_KEY) if isinstance(prefs, Mapping) else None)


def forge_option_overrides_setting() -> dict:
    """생성 요청이 쓸 덮어쓰기 설정 — 밀어 넣은 값, 없으면 파일 폴백. 늘 새 dict."""
    return _setting.get()


def reset_forge_option_overrides() -> None:
    """밀어 넣은 값을 지운다(다음 읽기는 파일 폴백) — 테스트 격리용."""
    _setting.reset()


# ── Forge 설정 잠금 기억(P10 검토 3) ─────────────────────────────────────────────
def _address(api_url: Any) -> str:
    return str(api_url or "").strip().rstrip("/")


class _FrozenVerdicts:
    """Forge 가 설정 잠금으로 거절한 앱 옵션 키 — 주소별. 생성 워커가 기억하고(거절을 본 스레드) 다음 요청이 읽는다.

    잠금은 Forge 시작 인자라 스냅샷을 새로 받아도 드러나지 않는다 — 기억이 없으면 요청마다 거절(샘플링 전 500)·재시도를
    되풀이한다. GUI 가 백엔드 변경·재연결(``_invalidate_sam_extra_capabilities``)과 수동 새로고침에서 잊게 한다."""

    def __init__(self):
        self._lock = threading.Lock()
        self._keys: dict = {}

    def remember(self, api_url: Any, keys: Iterable[str]) -> None:
        address = _address(api_url)
        found = frozenset(key for key in (keys or ()) if key in SPEC_BY_KEY)
        if not address or not found:
            return
        with self._lock:
            self._keys[address] = self._keys.get(address, frozenset()) | found

    def get(self, api_url: Any) -> frozenset:
        with self._lock:
            return self._keys.get(_address(api_url), frozenset())

    def forget(self, api_url: Any = None) -> None:
        with self._lock:
            if api_url is None:
                self._keys.clear()
            else:
                self._keys.pop(_address(api_url), None)


_frozen = _FrozenVerdicts()


def remember_frozen_options(api_url: Any, keys: Iterable[str]) -> None:
    """이 Forge 가 설정 잠금으로 거절한 키를 기억한다(스펙 키만). 다음 요청부터 ``plan_overrides(frozen=)`` 가 뺀다."""
    _frozen.remember(api_url, keys)


def frozen_options(api_url: Any) -> frozenset:
    """이 주소에서 설정 잠금으로 거절됐던 키(없으면 빈 집합)."""
    return _frozen.get(api_url)


def forget_frozen_options(api_url: Any = None) -> None:
    """잠금 기억을 잊는다 — ``api_url`` 이 없으면 모두(백엔드 변경·재연결), 있으면 그 주소만(수동 새로고침)."""
    _frozen.forget(api_url)


__all__ = [
    "GROUPS", "INFOTEXT_DAVE_PRE_DD", "INFOTEXT_PREFIX_DEDUP", "INFOTEXT_SEG_SEPARABLE", "INFOTEXT_SPARSE_GUESS",
    "MEMORY", "OPTION_KEYS", "OPT_CONNECTOR_FP32", "OPT_CONNECTOR_RUN_CACHE", "OPT_DAVE_PRE_DD", "OPT_KEEP_RESIDENT",
    "OPT_PREFIX_DEDUP", "OPT_SEG_SEPARABLE", "OPT_SPARSE_FORGE_GUESS", "OPT_UNLOAD_KEEP_IN_RAM", "PREF_KEY",
    "RESULT", "RESULT_MINOR", "SPECS", "SPEC_BY_KEY", "UNVERIFIED_NO_CONFIG", "UNVERIFIED_UNKNOWN",
    "ForgeOptionSpec", "OverridePlan", "forge_option_overrides_from_prefs_file", "forge_option_overrides_setting",
    "forget_frozen_options", "frozen_options", "label_of", "merge_into_payload", "normalize_overrides",
    "plan_notices", "plan_overrides", "remember_frozen_options", "reset_forge_option_overrides",
    "set_forge_option_overrides", "update_forge_option_overrides_from_prefs", "without_app_keys",
]
