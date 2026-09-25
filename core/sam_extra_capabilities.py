# core/sam_extra_capabilities.py
"""sam-extra(Forge 확장) 런타임 기능 스냅샷 — 순수 로직. Qt·requests·확장 import 없음.

입력은 GET 응답 묶음(경로 → ``HttpResult``)이고, 출력은 얼린 ``SamExtraCapabilities`` 다.
네트워크 수집과 URL 별 TTL 캐시는 ``core/sam_extra_probe.py`` 가 맡는다(GUI 스레드에서 부르지
않는다). 모두 GET 이지만 ``/sdapi/v1/extensions`` 만은 Forge 쪽 부작용이 있다(확장 목록을 비우고
다시 스캔) — 그래서 수집기가 따로 오래 캐시한다. 이 응답은 버전 힌트와 '꺼짐' 경고에만 쓴다.

**왜 필요한가** (gap matrix 5-(b), 라-8): 확장이 없거나 스크립트 제목이 바뀌면 Forge 가
422 로 생성 전체를 거절하고, 구버전이면 뒤쪽 위치 인자가 잘려 조용히 무시된다. 이 스냅샷은
그것을 **요청 전에** 알 수 있게 한다. 페이로드를 실제로 거르는 일은 뒤 패키지(P2/P3/P7-P9)가
``may_use()`` 로 한다 — 이 모듈은 판단 재료만 만든다.

보수적 기본값: 수집에 실패하면 ``status`` 가 ok 가 아니고 ``may_use()`` 는 True 를 돌려준다
(모르면 지금처럼 보낸다). 알 수 있을 때만 막는다.

규칙 (gap matrix 라-8):
- SAM3 는 script-info ``args[1].value`` 의 키에서 ``sam3_enable`` 을 빼고 앱 SAM3_KEYS 와 비교한다.
- PAG arg45(CLIP-L) 는 None 과 '' 를 같게 본다 — 선택지만 읽는다.
- PAG ``live_argc == 57`` (v0.21.2 계열) 은 프리셋 'Auto' 가 SMC 를 켠다는 경고를 낸다.
- Detail Daemon 은 arg2(amount) 의 maximum 으로 의미를 판정한다(>1 = 새 의미 ×0.1, ≤1 = 옛 의미).
- 옵션 존재는 ``/sdapi/v1/options`` 가 아니라 Gradio ``/config`` 의 ``setting_sam3_*`` 로 본다.
- 라우트(메모·TIPO·레퍼런스·Tile & Repair·LoRA Manager 설정)는 404 같은 확정 응답만 '없음'이다. 연결 실패·
  타임아웃·5xx·408·429 는 스크립트로 설치가 확인됐으면 있다고 본다(``route_unverified`` 경고, 사유는 errors).
  401·403(로그인·같은 출처 헤더 거절)은 라우트는 있어도 앱이 못 쓰는 확정 응답이라 False 로 두고, 사유만
  errors 에 남긴다 — '없음'·'벤더 미설치' 경고는 내지 않는다.
"""
from __future__ import annotations

import datetime as _dt
import re
from dataclasses import dataclass, field, fields
from types import MappingProxyType
from typing import Any, Iterable, Mapping, Optional
from urllib.parse import urlsplit, urlunsplit

from core import anima38, anima_guidance, sam3_args

# ── GET 경로 ────────────────────────────────────────────────────────────────
EP_SCRIPTS = "/sdapi/v1/scripts"
EP_SCRIPT_INFO = "/sdapi/v1/script-info"
# 부작용 있는 GET: Forge classic 은 이 요청마다 list_extensions() 로 확장 레지스트리를 비우고 다시
# 스캔한다(요구 사항 확인·콘솔 출력·git 정보 읽기). 수집기(core/sam_extra_probe.py)가 오래 캐시한다.
EP_EXTENSIONS = "/sdapi/v1/extensions"
EP_CN_MODELS = "/controlnet/model_list"
EP_CN_MODULES = "/controlnet/module_list"
EP_GRADIO_CONFIG = "/config"
EP_LORA_CONFIG = "/sam3-lora/config"          # 부작용 없음 (/sam3-lora/spawn 은 프로세스를 띄워서 안 씀)
EP_MEMOS = "/sam3-notebook/memos"             # 공유 메모 계약: 200 = 사용 가능, 404 = 옛 확장
EP_TIPO = "/sam3-tipo/status"                 # P19 제안 경로 (아직 확장에 없음)
EP_REFERENCE = "/sam-extra/reference"         # P20 제안 경로, POST 전용 → GET 은 405 면 존재
EP_CONTRACT = "/sam-extra/contract"           # 5-(b) 제안: {version, scripts, routes, options}
EP_TILE_REPAIR = "/sam-extra/tile-repair"     # T11 Anima Tile & Repair, POST 전용 → GET 은 405 면 존재(부작용 없음)

# Notebook 계열 라우트는 같은 출처 헤더가 없으면 403 이다(notebook_store.register_notebook_routes).
# 메모·Tile & Repair·LoRA Manager(/sam3-lora/config·spawn) 라우트도 같은 헤더로 막는다.
NOTEBOOK_HEADERS = MappingProxyType({"X-SAM3-Notebook": "1"})

STATUS_OK = "ok"
STATUS_UNKNOWN = "unknown"            # 아직 확인 전 / 백엔드가 바뀌어 무효화됨
STATUS_UNREACHABLE = "unreachable"    # 연결 자체가 안 됨
STATUS_ERROR = "error"                # 응답은 왔는데 스크립트 목록을 읽을 수 없음
STATUS_NOT_APPLICABLE = "not_applicable"  # ComfyUI 백엔드 — Forge 확장과 무관

# ── 스크립트 제목 (Forge API 는 소문자로 돌려준다) ─────────────────────────
TITLE_SAM3 = sam3_args.SCRIPT_SAM3.lower()
TITLE_PAG = anima_guidance.SCRIPT_PERTURBATION.lower()
TITLE_SKIMMED = anima_guidance.SCRIPT_SKIMMED_CFG.lower()
TITLE_DETAIL_DAEMON = anima_guidance.SCRIPT_DETAIL_DAEMON.lower()
TITLE_ANIMA38 = anima38.SCRIPT_NAME.lower()
TITLE_DORA = "dora inference mode"                  # scripts/dora_infer_mode.py (DORA_INFER_NAME)
TITLE_VAE2X = "anima vae 2x (spacepxl decoder)"     # scripts/anima_vae_2x.py
TITLE_LORA_BRIDGE = "sam3 lora manager bridge"      # scripts/lora_manager.py (인자 0개)
TITLE_SPARSE_LORA = "sam extra anima sparse lora"   # scripts/anima_lora_blocks.py (인자 0개)
TITLE_REFERENCE_POC = "anima reference poc (shape logger)"  # 디버그

SAM_EXTRA_TITLES = (
    TITLE_SAM3, TITLE_PAG, TITLE_SKIMMED, TITLE_DETAIL_DAEMON, TITLE_ANIMA38, TITLE_DORA,
    TITLE_VAE2X, TITLE_LORA_BRIDGE, TITLE_SPARSE_LORA, TITLE_REFERENCE_POC,
)
# 확장 폴더 이름 — 앱 설치기는 sam-extra, 수동 클론은 forge_sam3_extension (core/sam3_assets.py 와 같다)
EXTENSION_FOLDERS = ("forge_sam3_extension", "sam-extra")

# 앱이 위치 인자로 보내는 스크립트 → 앱 스펙 키 순서. 인자 수 비교와 '잘려서 무시될 키' 계산에 쓴다.
# SAM3 는 dict 한 개로 보내므로 인자 수 대신 키 집합을 비교한다. DoRA·VAE 2x 는 아직 앱이 보내지 않는다.
POSITIONAL_SPECS: Mapping[str, tuple[str, ...]] = MappingProxyType({
    TITLE_PAG: tuple(key for key, *_ in anima_guidance.PERTURBATION_SPEC),
    TITLE_SKIMMED: tuple(key for key, *_ in anima_guidance.SKIMMED_SPEC),
    TITLE_DETAIL_DAEMON: tuple(key for key, *_ in anima_guidance.DETAIL_DAEMON_SPEC),
    TITLE_ANIMA38: tuple(anima38.ARG_NAMES),
})

# v0.21.2 계열 PAG 빌드의 인자 수. 이 빌드는 SMC 프리셋이 'Off' 가 아니면 SMC 를 켠다(나-5).
PAG_OLD_BUILD_ARGC = 57
# Detail Daemon Hires Pass 인자 위치(맨 뒤 append) — 앱 스펙의 dd_hires 자리.
DD_HIRES_INDEX = POSITIONAL_SPECS[TITLE_DETAIL_DAEMON].index("dd_hires")
# 앱 기능 이름 → 스냅샷 플래그. Vue 도 같은 이름을 쓴다(frontend/src/utils/samExtraCapabilities.ts).
FEATURE_FLAGS = (
    "sam3", "anima_guidance", "skimmed_cfg", "detail_daemon", "anima38", "dora", "vae2x",
    "lora_manager", "memo_routes", "tipo_route", "reference_route", "contract_route",
    "tile_repair_route",
)
_SCRIPT_FEATURES = (
    ("sam3", TITLE_SAM3), ("anima_guidance", TITLE_PAG), ("skimmed_cfg", TITLE_SKIMMED),
    ("detail_daemon", TITLE_DETAIL_DAEMON), ("anima38", TITLE_ANIMA38), ("dora", TITLE_DORA),
    ("vae2x", TITLE_VAE2X),
)
_FEATURE_LABELS = {
    "sam3": "SAM3 Mask", "anima_guidance": "Anima 가이던스(PAG)", "skimmed_cfg": "Skimmed CFG",
    "detail_daemon": "Detail Daemon", "anima38": "Anima 3.8B", "dora": "DoRA 추론 방식",
    "vae2x": "VAE 2x", "lora_manager": "LoRA Manager", "memo_routes": "메모 동기화",
    "tipo_route": "TIPO", "reference_route": "레퍼런스", "tile_repair_route": "Tile & Repair",
}
# 있고 없음만 보는 라우트: 기능 이름 → (경로, 있음으로 보는 상태). POST 전용 라우트는 GET 405 도 있음이다.
_ROUTE_FLAGS = (
    ("memo_routes", EP_MEMOS, (200,)), ("tipo_route", EP_TIPO, (200,)),
    ("reference_route", EP_REFERENCE, (200, 405)), ("tile_repair_route", EP_TILE_REPAIR, (200, 405)),
)
# 라우트가 없다는 뜻이 아닌 응답(요청 시간 초과·과다 요청). 연결 실패·타임아웃(status None)과 5xx 도 같다.
_INCONCLUSIVE_STATUSES = frozenset({408, 429})
# Gradio 이름 엔드포인트 (REST 라우트가 없는 기능의 대체 경로 — 다-1). 이름만 기록한다.
GRADIO_API_NAMES = (
    "expand", "handle_apply", "handle_refine_click", "handle_anima_click",
    "handle_anima_reference_click", "preview_reference_layout", "load_selected_reference",
    "sam3_quick", "_apply_regional_preset",
)
# 위치 인자에서 라이브 선택지를 읽는 곳: 이름 → (제목, 인덱스, 라벨에 있어야 할 글자 | None)
_CHOICE_SOURCES = (
    ("clip_l", TITLE_PAG, 45, "clip"),
    ("smc_presets", TITLE_PAG, 56, "smc"),
    ("anima38_adapters", TITLE_ANIMA38, 1, "adapter"),
    ("vae2x_decoders", TITLE_VAE2X, 1, "vae"),
    ("dora_modes", TITLE_DORA, 1, None),
    ("dora_insert_policies", TITLE_DORA, 2, None),
    ("dora_weak_scopes", TITLE_DORA, 4, None),
)

_EMPTY: Mapping[str, Any] = MappingProxyType({})


@dataclass(frozen=True)
class HttpResult:
    """GET 한 번의 결과. status None = 연결 실패/타임아웃(error 에 예외 이름)."""

    status: Optional[int]
    body: Any = None
    error: Optional[str] = None

    @property
    def ok(self) -> bool:
        return self.status == 200 and self.error is None


def _freeze(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType({str(k): _freeze(v) for k, v in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(_freeze(v) for v in value)
    return value


def _thaw(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {k: _thaw(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_thaw(v) for v in value]
    return value


@dataclass(frozen=True)
class SamExtraCapabilities:
    """한 WebUI 백엔드에서 본 sam-extra 기능 스냅샷 (읽기 전용)."""

    status: str = STATUS_UNKNOWN
    checked_at: Optional[str] = None
    installed: bool = False
    sam3: bool = False
    anima_guidance: bool = False
    anima_guidance_argc: Optional[int] = None
    skimmed_cfg: bool = False
    detail_daemon: bool = False
    detail_daemon_hires: Optional[bool] = None  # Detail Daemon 인자 13(Hires Pass)이 있나. None = 모름·스크립트 없음
    anima38: bool = False
    dora: bool = False
    vae2x: bool = False
    lora_manager: bool = False
    memo_routes: bool = False
    tipo_route: bool = False
    reference_route: bool = False
    contract_route: bool = False
    tile_repair_route: bool = False
    version: Mapping[str, Any] = field(default_factory=lambda: _EMPTY)
    scripts: Mapping[str, Mapping[str, Any]] = field(default_factory=lambda: _EMPTY)
    sam3_keys: Mapping[str, tuple] = field(default_factory=lambda: _EMPTY)
    options: Mapping[str, Any] = field(default_factory=lambda: _EMPTY)
    options_known: bool = False
    gradio_api: tuple = ()
    choices: Mapping[str, tuple] = field(default_factory=lambda: _EMPTY)
    warnings: tuple = ()
    errors: Mapping[str, str] = field(default_factory=lambda: _EMPTY)

    @property
    def known(self) -> bool:
        return self.status == STATUS_OK

    def may_use(self, feature: str) -> bool:
        """페이로드에 그 기능 블록을 넣어도 되나. 모르면 True(지금처럼 보냄)."""
        if feature not in FEATURE_FLAGS:
            raise KeyError(feature)
        if not self.known:
            return True
        return bool(getattr(self, feature))

    def script(self, title: str) -> Mapping[str, Any]:
        return self.scripts.get(title.lower(), _EMPTY)

    def has_option(self, key: str) -> Optional[bool]:
        """Forge 설정 키(sam3_*)가 있나. /config 를 못 읽었으면 None."""
        return (key in self.options) if self.options_known else None

    def to_dict(self) -> dict:
        """JSON 으로 보낼 모양 (Vue 이벤트 samExtraCapabilities). URL 은 넣지 않는다."""
        values = {f.name: _thaw(getattr(self, f.name)) for f in fields(self)}
        return {
            "status": values["status"],
            "known": self.known,
            "checked_at": values["checked_at"],
            "installed": values["installed"],
            "features": {name: bool(values[name]) for name in FEATURE_FLAGS},
            "anima_guidance_argc": values["anima_guidance_argc"],
            "detail_daemon_hires": values["detail_daemon_hires"],
            "version": values["version"],
            "scripts": values["scripts"],
            "sam3_keys": values["sam3_keys"],
            "options": values["options"],
            "options_known": values["options_known"],
            "gradio_api": values["gradio_api"],
            "choices": values["choices"],
            "warnings": values["warnings"],
            "errors": values["errors"],
        }


def now_iso() -> str:
    return _dt.datetime.now(_dt.timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def unknown_capabilities(status: str = STATUS_UNKNOWN, *, error: Optional[str] = None,
                         checked_at: Optional[str] = None) -> SamExtraCapabilities:
    """확인 전·무효화·ComfyUI 처럼 판단 재료가 없는 스냅샷."""
    errors = {"probe": error} if error else {}
    return SamExtraCapabilities(status=status, checked_at=checked_at, errors=_freeze(errors))


# ── 응답 해석 ───────────────────────────────────────────────────────────────

def _titles(body: Any) -> tuple[set, set]:
    """/sdapi/v1/scripts → (txt2img 제목, img2img 제목) 소문자 집합."""
    if not isinstance(body, Mapping):
        return set(), set()
    def names(key):
        items = body.get(key)
        return {str(x).strip().lower() for x in items if isinstance(x, str)} if isinstance(items, list) else set()
    return names("txt2img"), names("img2img")


def _script_info(body: Any) -> tuple[dict, dict]:
    """/sdapi/v1/script-info → (txt2img 제목→항목, img2img 제목→항목). 첫 항목만 쓴다."""
    t2i, i2i = {}, {}
    if not isinstance(body, list):
        return t2i, i2i
    for item in body:
        if not isinstance(item, Mapping) or not isinstance(item.get("name"), str):
            continue
        bucket = i2i if item.get("is_img2img") else t2i
        bucket.setdefault(item["name"].strip().lower(), item)
    return t2i, i2i


def has_sam_extra_scripts(scripts_body: Any, script_info_body: Any = None) -> bool:
    """두 목록 중 하나라도 sam-extra 제목을 담고 있으면 True (수집기가 /config 를 더 받을지 정할 때)."""
    t2i, i2i = _titles(scripts_body)
    info_t2i, info_i2i = _script_info(script_info_body)
    present = t2i | i2i | set(info_t2i) | set(info_i2i)
    return any(title in present for title in SAM_EXTRA_TITLES)


def _args(item: Any) -> list:
    args = item.get("args") if isinstance(item, Mapping) else None
    return args if isinstance(args, list) else []


def _arg(item: Any, index: int) -> Mapping[str, Any]:
    args = _args(item)
    value = args[index] if 0 <= index < len(args) else None
    return value if isinstance(value, Mapping) else _EMPTY


def _string_list(values: Any) -> tuple[str, ...]:
    if not isinstance(values, list):
        return ()
    return tuple(dict.fromkeys(x for x in values if isinstance(x, str)))


def _warning(code: str, feature: str, message: str) -> dict:
    return {"code": code, "feature": feature, "message": message}


def _feature_groups(keys: Iterable[str]) -> list[str]:
    """잘릴 스펙 키 → 사람이 읽을 기능 묶음 (guid_rdc_tau → RDC)."""
    groups = []
    for key in keys:
        parts = key.split("_")
        label = parts[1].upper() if parts[0] in ("guid", "skim", "dd") and len(parts) > 1 else key
        if key == "guid_smc_master_enabled":
            label = "SMC 마스터"
        if label not in groups:
            groups.append(label)
    return groups


def _script_detail(title: str, present: bool, img2img: bool, item: Any, warnings: list) -> dict:
    live_argc = len(_args(item)) if isinstance(item, Mapping) else None
    spec = POSITIONAL_SPECS.get(title)
    detail = {"present": present, "img2img": img2img, "live_argc": live_argc,
              "spec_argc": len(spec) if spec is not None else None,
              "trailing_unmapped": [], "extra_live_args": 0}
    if not present or spec is None or live_argc is None:
        return detail
    feature = next((name for name, t in _SCRIPT_FEATURES if t == title), title)
    if live_argc < len(spec):
        dropped = list(spec[live_argc:])
        detail["trailing_unmapped"] = dropped
        warnings.append(_warning(
            "args_fewer", feature,
            f"{_FEATURE_LABELS.get(feature, title)}: 확장 인자가 {live_argc}개로 앱({len(spec)}개)보다 적습니다 "
            f"— 확장 업데이트 필요. 무시될 기능: {', '.join(_feature_groups(dropped))}"))
    elif live_argc > len(spec):
        detail["extra_live_args"] = live_argc - len(spec)
        warnings.append(_warning(
            "args_more", feature,
            f"{_FEATURE_LABELS.get(feature, title)}: 앱보다 새 확장입니다(인자 {live_argc}개, 앱 {len(spec)}개) "
            "— 새 옵션은 기본값으로 동작합니다."))
    return detail


def _is_sam_extra_entry(entry: Mapping[str, Any]) -> bool:
    name = str(entry.get("name") or "").strip().lower()
    remote = str(entry.get("remote") or "").strip().lower().rstrip("/")
    return name in EXTENSION_FOLDERS or remote.endswith(("/sam-extra", "/sam-extra.git",
                                                         "/forge_sam3_extension", "/forge_sam3_extension.git"))


def _extension_entry(body: Any) -> Mapping[str, Any]:
    """sam-extra 확장 항목. 켜진 항목을 먼저 고른다 — 수동 클론(forge_sam3_extension)과 앱이 건
    sam-extra 연결이 같이 있으면 Forge 는 폴더 이름순으로 내놓아서, 꺼진 쪽이 앞에 올 수 있다."""
    if not isinstance(body, list):
        return _EMPTY
    matches = [entry for entry in body if isinstance(entry, Mapping) and _is_sam_extra_entry(entry)]
    return next((entry for entry in matches if entry.get("enabled") is True),
                matches[0] if matches else _EMPTY)


# scp 모양 git 주소 (git@host:owner/repo.git). 호스트는 두 글자 이상 — 'C:\...' 같은 로컬 경로를 거른다.
_SCP_REMOTE = re.compile(r"^(?:[^@/\\\s]+@)?(?P<host>[A-Za-z0-9][A-Za-z0-9.-]+):(?P<path>[^\\\s]+)$")
_REMOTE_SCHEMES = frozenset({"http", "https", "ssh", "git", "git+ssh", "ssh+git"})


def _public_remote(value: Any) -> Optional[str]:
    """git remote 에서 자격 증명(user:token@)·쿼리·프래그먼트를 뺀 주소. 스냅샷은 웹 모드의 모든
    클라이언트에도 가므로 그대로 싣지 않는다. 네트워크 주소가 아닌 모양(로컬 경로·file://)은 None."""
    if not isinstance(value, str) or not value.strip():
        return None
    text = value.strip()
    if "://" not in text:
        match = _SCP_REMOTE.match(text)
        return f"{match.group('host')}:{match.group('path')}" if match else None
    try:
        parts = urlsplit(text)
        parts.port  # noqa: B018 — 잘못된 포트는 ValueError
    except ValueError:
        return None
    host = parts.netloc.rpartition("@")[2]
    if parts.scheme.lower() not in _REMOTE_SCHEMES or not host:
        return None
    return urlunsplit((parts.scheme, host, parts.path, "", ""))


def _audited_version() -> Optional[str]:
    """앱이 맞춘 확장 버전 (P1 계약 레지스트리가 있으면). 없으면 None."""
    try:
        from core import sam_extra_contract  # type: ignore[attr-defined]
    except Exception:
        return None
    value = getattr(sam_extra_contract, "EXT_VERSION_AUDITED", None)
    return str(value) if value else None


def _version_hints(ext: Mapping[str, Any], contract: HttpResult) -> dict:
    version = {"extension_version": None, "audited_version": _audited_version(), "folder": None,
               "branch": None, "commit": None, "commit_date": None, "enabled": None, "remote": None}
    if contract.ok and isinstance(contract.body, Mapping) and isinstance(contract.body.get("version"), str):
        version["extension_version"] = contract.body["version"]
    if ext:
        commit = ext.get("version") or str(ext.get("commit_hash") or "")[:8] or None
        version.update({
            "folder": ext.get("name"), "branch": ext.get("branch"), "commit": commit,
            "commit_date": ext.get("commit_date") if isinstance(ext.get("commit_date"), (int, float)) else None,
            "enabled": ext.get("enabled") if isinstance(ext.get("enabled"), bool) else None,
            "remote": _public_remote(ext.get("remote")),
        })
    return version


def _options(config: HttpResult) -> tuple[dict, bool, tuple]:
    """Gradio /config → (sam3_* 옵션 시작값, 읽었나, 있는 Gradio api 이름)."""
    if not config.ok or not isinstance(config.body, Mapping):
        return {}, False, ()
    options = {}
    for component in config.body.get("components") or ():
        props = component.get("props") if isinstance(component, Mapping) else None
        elem_id = props.get("elem_id") if isinstance(props, Mapping) else None
        if isinstance(elem_id, str) and elem_id.startswith("setting_sam3_"):
            options[elem_id[len("setting_"):]] = props.get("value")
    names = {dep.get("api_name") for dep in config.body.get("dependencies") or ()
             if isinstance(dep, Mapping)}
    return options, True, tuple(name for name in GRADIO_API_NAMES if name in names)


def _choices(t2i_info: Mapping[str, Any], cn_models: HttpResult, cn_modules: HttpResult) -> dict:
    choices = {}
    if cn_models.ok and isinstance(cn_models.body, Mapping):
        choices["controlnet_models"] = _string_list(cn_models.body.get("model_list"))
    if cn_modules.ok and isinstance(cn_modules.body, Mapping):
        choices["controlnet_modules"] = _string_list(cn_modules.body.get("module_list"))
    for name, title, index, hint in _CHOICE_SOURCES:
        arg = _arg(t2i_info.get(title), index)
        label = str(arg.get("label") or "").lower()
        if not arg or (hint and hint not in label) or not isinstance(arg.get("choices"), list):
            continue
        choices[name] = _string_list(arg.get("choices"))
    return choices


def _endpoint_error(path: str, result: HttpResult, *, required: bool) -> Optional[str]:
    if result.error:
        return result.error
    if result.status == 200:
        return (None if result.body is not None or path in (EP_MEMOS, EP_REFERENCE, EP_TIPO, EP_TILE_REPAIR)
                else "빈 응답")
    if not required and result.status in (404, 405):
        return None
    return f"HTTP {result.status}"


def _route_inconclusive(result: Optional[HttpResult]) -> bool:
    """라우트 확인이 실패했을 뿐 '없음'이 확정되지 않았나 (연결 실패·타임아웃·5xx·408·429).
    수집기가 묻지 않은 경로(None)는 실패가 아니다. 404 등은 확정된 응답이다."""
    if result is None:
        return False
    if result.status is None:
        return True
    return result.status >= 500 or result.status in _INCONCLUSIVE_STATUSES


def _assume_unverified_route(name: str, path: str, result: Optional[HttpResult], installed: bool,
                             errors: Mapping[str, str], warnings: list) -> bool:
    """스크립트로 설치가 확인됐는데 라우트 확인만 실패했으면(``_route_inconclusive``) 있다고 보고
    ``route_unverified`` 경고를 남긴다(모르면 지금처럼 보냄) — 실제로 없으면 실행할 때 라우트가 오류를 돌려준다."""
    if not installed or not _route_inconclusive(result):
        return False
    reason = errors.get(path) or f"HTTP {result.status}"
    warnings.append(_warning("route_unverified", name,
                             f"{_FEATURE_LABELS.get(name, name)} 라우트를 확인하지 못했습니다({reason}) "
                             "— 있다고 보고 사용합니다."))
    return True


def _route_flags(responses: Mapping[str, HttpResult], installed: bool, errors: Mapping[str, str],
                 warnings: list) -> dict:
    """라우트 기능 플래그. 확인만 실패한 라우트는 ``_assume_unverified_route`` 가 정한다."""
    flags = {}
    for name, path, present in _ROUTE_FLAGS:
        result = responses.get(path)
        flags[name] = ((result is not None and result.status in present)
                       or _assume_unverified_route(name, path, result, installed, errors, warnings))
    return flags


def build_capabilities(responses: Mapping[str, HttpResult], *,
                       checked_at: Optional[str] = None) -> SamExtraCapabilities:
    """GET 응답 묶음 → 스냅샷. 빠진 경로는 연결 실패로 본다."""
    missing = HttpResult(None, None, "not requested")
    get = lambda path: responses.get(path) or missing  # noqa: E731
    checked_at = checked_at or now_iso()
    scripts, info = get(EP_SCRIPTS), get(EP_SCRIPT_INFO)
    errors = {}
    for path, required in ((EP_SCRIPTS, True), (EP_SCRIPT_INFO, True), (EP_EXTENSIONS, False),
                           (EP_CN_MODELS, False), (EP_CN_MODULES, False), (EP_GRADIO_CONFIG, False),
                           (EP_LORA_CONFIG, False), (EP_MEMOS, False), (EP_TIPO, False),
                           (EP_REFERENCE, False), (EP_CONTRACT, False), (EP_TILE_REPAIR, False)):
        result = responses.get(path)
        if result is None:
            continue  # 수집기가 일부러 건너뛴 경로(/config 는 확장이 없으면 받지 않는다)
        message = _endpoint_error(path, result, required=required)
        if message:
            errors[path] = message

    scripts_ok = scripts.ok and isinstance(scripts.body, Mapping)
    info_ok = info.ok and isinstance(info.body, list)
    if not scripts_ok and not info_ok:
        unreachable = scripts.status is None and info.status is None
        return SamExtraCapabilities(status=STATUS_UNREACHABLE if unreachable else STATUS_ERROR,
                                    checked_at=checked_at, errors=_freeze(errors))

    t2i_titles, i2i_titles = _titles(scripts.body) if scripts_ok else (set(), set())
    t2i_info, i2i_info = _script_info(info.body) if info_ok else ({}, {})
    if not scripts_ok:  # scripts 가 실패하면 script-info 의 이름으로 대신한다
        t2i_titles, i2i_titles = set(t2i_info), set(i2i_info)

    warnings: list = []
    details = {}
    for title in SAM_EXTRA_TITLES:
        present = title in t2i_titles
        details[title] = _script_detail(title, present, title in i2i_titles,
                                        t2i_info.get(title), warnings)
    flags = {name: details[title]["present"] for name, title in _SCRIPT_FEATURES}

    ext = _extension_entry(get(EP_EXTENSIONS).body if get(EP_EXTENSIONS).ok else None)
    installed = any(d["present"] or d["img2img"] for d in details.values())
    if not installed and ext and ext.get("enabled") is False:
        # 스크립트가 없고, 켜진 sam-extra 항목도 없다(_extension_entry 가 켜진 쪽을 먼저 고른다).
        # 스크립트가 보이면 이 항목은 따지지 않는다 — 확장 목록은 오래 캐시해서 늦을 수 있다.
        warnings.append(_warning("extension_disabled", "sam_extra",
                                 "sam-extra 확장이 Forge 에서 꺼져 있습니다 (Extensions 탭에서 켜야 합니다)."))
    elif not installed:
        warnings.append(_warning("extension_missing", "sam_extra",
                                 "연결된 WebUI 에 sam-extra 확장이 없습니다 — SAM3·Anima 가이던스 등을 켜면 "
                                 "Forge 가 요청을 거절(422)합니다."))
    else:
        for name, _title in _SCRIPT_FEATURES:
            if not flags[name]:
                warnings.append(_warning("script_missing", name,
                                         f"{_FEATURE_LABELS[name]} 스크립트가 확장에 없습니다 (확장 버전 확인)."))

    # SAM3 dict 키 비교 (sam3_enable 은 활성화 플래그라 뺀다 — 나-3)
    sam3_keys = {"missing_in_extension": (), "unknown_to_app": ()}
    sam3_state = _arg(t2i_info.get(TITLE_SAM3), 1).get("value")
    if isinstance(sam3_state, Mapping):
        live_keys = {str(k) for k in sam3_state} - {"sam3_enable", "enabled"}
        app_keys = set(sam3_args.SAM3_KEYS)
        sam3_keys = {"missing_in_extension": tuple(sorted(app_keys - live_keys)),
                     "unknown_to_app": tuple(sorted(live_keys - app_keys))}
        if sam3_keys["missing_in_extension"]:
            warnings.append(_warning("sam3_keys_unknown_to_extension", "sam3",
                                     "확장이 모르는 SAM3 설정은 조용히 버려집니다: "
                                     + ", ".join(sam3_keys["missing_in_extension"])))
        if sam3_keys["unknown_to_app"]:
            warnings.append(_warning("sam3_keys_unknown_to_app", "sam3",
                                     "앱이 아직 모르는 SAM3 설정이 있습니다(확장 기본값으로 동작): "
                                     + ", ".join(sam3_keys["unknown_to_app"])))

    # PAG 인자 수와 구버전 SMC 동작 (나-5)
    pag_argc = details[TITLE_PAG]["live_argc"] if flags["anima_guidance"] else None
    smc_choices = _string_list(_arg(t2i_info.get(TITLE_PAG), 56).get("choices"))
    if flags["anima_guidance"] and (pag_argc == PAG_OLD_BUILD_ARGC or "Off" in smc_choices):
        warnings.append(_warning("pag_smc_auto_old_build", "anima_guidance",
                                 "구버전 Anima 가이던스(v0.21.2 계열): 앱이 SMC 프리셋 'Auto' 를 보내면 "
                                 "가이던스를 켤 때 SMC 도 함께 켜집니다. 확장을 업데이트하세요."))

    # Detail Daemon Hires Pass: 인자 13 이 있는 빌드만 base/hires 패스를 가른다. 없으면 args_fewer 경고가
    # 이미 붙고(무시될 기능: HIRES), 앱은 dd_hires 를 켠 생성에 경고를 남긴다(anima_guidance.detail_daemon_hires_note).
    # 값의 단위는 판정하지 않는다 — 앱은 원본 노드 단위 그대로 보낸다(v0.21.2 옛 의미는 지원하지 않는다).
    dd_hires = None
    dd_argc = details[TITLE_DETAIL_DAEMON]["live_argc"] if flags["detail_daemon"] else None
    if dd_argc is not None:
        dd_hires = dd_argc > DD_HIRES_INDEX

    choices = _choices(t2i_info, get(EP_CN_MODELS), get(EP_CN_MODULES))
    live_modules = choices.get("controlnet_modules")
    if flags["sam3"] and live_modules:
        unknown = [m for m in sam3_args.CN_MODULES if m not in live_modules]
        if unknown:
            warnings.append(_warning("sam3_cn_module_not_live", "sam3",
                                     "Forge 에 없는 SAM3 ControlNet 전처리기(대소문자 구분): " + ", ".join(unknown)))

    # LoRA Manager 설정 라우트: '벤더 미설치' 는 200 본문(available: false)이 말할 때만이다. 그 밖의 응답은
    # 라우트 플래그와 같은 규칙 — 확인 실패(타임아웃·5xx·408·429)는 있다고 보고, 401·403(로그인·같은 출처
    # 헤더 거절)과 404 는 확정 응답이라 False 로 둔다(사유는 errors).
    lora = get(EP_LORA_CONFIG)
    lora_manager = bool(lora.ok and isinstance(lora.body, Mapping) and lora.body.get("available", True))
    if lora.ok and not lora_manager:
        warnings.append(_warning("lora_manager_unavailable", "lora_manager",
                                 "sam-extra 의 LoRA Manager 가 준비되지 않았습니다(벤더 미설치)."))
    elif not lora_manager:
        lora_manager = _assume_unverified_route("lora_manager", EP_LORA_CONFIG, responses.get(EP_LORA_CONFIG),
                                                installed, errors, warnings)
    contract = get(EP_CONTRACT)
    options, options_known, gradio_api = _options(get(EP_GRADIO_CONFIG))
    routes = _route_flags(responses, installed, errors, warnings)
    return SamExtraCapabilities(
        status=STATUS_OK, checked_at=checked_at, installed=installed,
        anima_guidance_argc=pag_argc, detail_daemon_hires=dd_hires,
        lora_manager=lora_manager,
        contract_route=bool(contract.ok and isinstance(contract.body, Mapping)),
        version=_freeze(_version_hints(ext, contract)),
        scripts=_freeze(details), sam3_keys=_freeze(sam3_keys),
        options=_freeze(options), options_known=options_known, gradio_api=tuple(gradio_api),
        choices=_freeze(choices), warnings=_freeze(warnings), errors=_freeze(errors),
        **flags, **routes,
    )


__all__ = [
    "EP_CN_MODELS", "EP_CN_MODULES", "EP_CONTRACT", "EP_EXTENSIONS", "EP_GRADIO_CONFIG",
    "EP_LORA_CONFIG", "EP_MEMOS", "EP_REFERENCE", "EP_SCRIPTS", "EP_SCRIPT_INFO", "EP_TILE_REPAIR", "EP_TIPO",
    "FEATURE_FLAGS", "HttpResult", "NOTEBOOK_HEADERS", "SAM_EXTRA_TITLES", "STATUS_ERROR",
    "STATUS_NOT_APPLICABLE", "STATUS_OK", "STATUS_UNKNOWN", "STATUS_UNREACHABLE",
    "SamExtraCapabilities", "build_capabilities", "has_sam_extra_scripts", "now_iso",
    "unknown_capabilities",
]
