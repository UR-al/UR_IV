# ui/sampling_blocks.py
"""샘플링 블록(NegPiP·Anima 가이던스·Anima38·DoRA·VAE DeGrid)을 만드는 유일한 곳 — 얇은 Qt 접착층.

- 메인 체인(``GenerationMixin._apply_postprocess_chain``, target t2i·i2i)과 보조 패스 봉투(``ui/aux_pass_snapshot``,
  target aux)가 같은 ``build_sampling_blocks`` 를 쓴다 — 둘이 갈라지지 않게(tests/test_sampling_blocks.py T15).
- 기여자(``CONTRIBUTORS``)는 ``(host, SamplingContext) -> Contribution`` 이다. 블록마다 출처(사용자 값 / 앱 기본값)를
  ``Contribution.add(..., provenance=)`` 로 적는다. 모를 때 보낼지·빠질 때 알릴지는 core/alwayson_propagation 의
  제목별 규칙(``Rule.when_unknown``)과 그 출처로 정해진다 — P8(DoRA)·P9(Anima38)·VAE DeGrid 는 이 튜플에 한 줄씩만
  더했다(호출부는 그대로). DeGrid 는 최종 이미지 블록이라(전달 행 ``passes=()``) 봉투에는 실려도 보조 패스에는 가지 않는다.
  기여자는 위젯만 읽는다(GUI 스레드). 하나가 실패해도 나머지와 생성은 진행한다.
- 게이트: Forge(webui) **메인** 요청(t2i·i2i)만 여기서 한다(스냅샷이 '없다'고 한 블록은 422 대신 빼고 경고).
  보조 패스(aux)는 워커가 요청 직전에 새 스냅샷으로 한다(``WebUIBackend._propagate``, 손 재구성은 ``_hand_snapshot``)
  — 클릭과 전송 사이에 스냅샷이 바뀔 수 있고, 같은 제목을 두 번 알리지 않게. Comfy·Krea2 는 게이트하지 않는다.
- 알림은 여기서 띄우지 않는다(critic B15): 결과로 돌려주고 ``remember_sampling_notices`` 로 그 페이로드에 (블록 출처와
  함께) 묶어 둔다. 실제로 보내는 곳이 ``take_sampling_notices``/``take_sampling_state`` 로 꺼내 띄운다 — 생성 전 확인
  (``ui/sam_extra_notices_ui.check_before_generation``)·채팅·손 재구성, 그리고 대기열에 넣거나 워커로 바로 보내는
  클릭(XYZ·시드 탐색 ``freeze_sampling_payload``, 만화 컷은 작업당 한 번). Comfy 사전 점검처럼 보내지 않는 빌드는
  꺼내지 않으므로 뜨지 않는다. 동결 페이로드(시드 탐색·XYZ 대기열)는 보낼 때 ``regate_frozen_payload`` 가 다시
  게이트한다 — 그 사본은 빌드 때 뺀 블록을 다시 볼 수 없으므로 그 알림은 클릭에서 띄운다(P7 검토 R2).
- 블록 출처: Forge 메인 요청이면 이번 빌드가 넣은 **앱 기본값** 블록의 출처를 페이로드의 비공개 키에 적는다
  (``mark_request_provenance``). ``WebUIBackend._generate`` 가 요청 전에 떼어, Forge 가 그 블록을 422 로 거절했을 때
  앱 기본값이면 정보·그 밖이면 경고로 알린다(P10 검토 2, A2·A6). 앱 기본값 블록이 없으면 페이로드는 그대로다.
- 로거는 'generation' — 메인 체인의 기존 로그 문구·수준을 그대로 옮겼다(tests/test_detail_daemon_origin.py 가 비교한다).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Mapping, Optional

from core import alwayson_propagation as ap
from ui.anima38_ui import contribute as _anima38
from ui.dora_infer_mode_ui import contribute as _dora_infer_mode
from ui.vae_degrid_ui import contribute as _vae_degrid
from utils.app_logger import get_logger

_logger = get_logger('generation')

_NOTICE_SLOT = "_sampling_block_notices"
# 페이로드에 앱 기본값 블록의 출처를 적어 두는 비공개 키(core/alwayson_propagation.PROVENANCE_KEY) — 메인 체인이 Forge 메인
# 요청에 적고(mark_request_provenance), 동결(대기열) 페이로드는 클릭 때 적은 것을 보낼 때 regate_frozen_payload 가 다시
# 게이트해 남은 블록만 다시 적는다(Forge 가 아니면 뗀다). WebUIBackend 가 요청 전에 떼므로 Forge 요청 JSON 으로 나가지
# 않는다. 앱 기본값 블록이 없으면 쓰지 않는다(기본 페이로드·대기열 항목은 그대로).
FROZEN_PROVENANCE_KEY = ap.PROVENANCE_KEY


@dataclass(frozen=True)
class SamplingContext:
    target: str                 # TARGET_T2I | TARGET_I2I | TARGET_AUX
    backend: str                # BACKEND_WEBUI | BACKEND_COMFY | BACKEND_KREA2 (이 요청이 실제로 가는 경로)
    capabilities: Any = None    # webui 일 때만: host.sam_extra_capabilities, 없으면 peek 캐시(네트워크 없음)
    model: str = ""             # T2I 콤보의 모델(보조 패스가 실제로 쓰는 모델과 다를 수 있다 — 백엔드가 대조한다, critic A7)
    host_backend: str = ap.BACKEND_WEBUI   # 연결된 백엔드 종류(Krea2 여부와 무관)


@dataclass
class Contribution:
    """한 기여자가 내놓는 블록·알림. 출처가 없는 블록은 사용자 값으로 본다."""

    blocks: dict = field(default_factory=dict)
    notices: list = field(default_factory=list)
    provenance: dict = field(default_factory=dict)

    def add(self, title: str, block: dict, *, provenance: str = ap.PROVENANCE_USER) -> "Contribution":
        if provenance not in ap.PROVENANCES:
            raise ValueError(f"모르는 블록 출처: {provenance!r}")
        self.blocks[title] = block
        self.provenance[title] = provenance
        return self


@dataclass
class SamplingBlocks:
    blocks: dict                # 순서: NegPiP, PAG, Skimmed, DD, Anima38, DoRA, DeGrid
    notices: list               # 보내는 곳에서 띄울 알림(게이트 제외 포함)
    dropped: tuple = ()
    provenance: dict = field(default_factory=dict)
    context: Optional[SamplingContext] = None


def _host_backend() -> str:
    try:
        from backends import BackendType, get_backend_type
        return ap.BACKEND_COMFY if get_backend_type() == BackendType.COMFYUI else ap.BACKEND_WEBUI
    except Exception:
        return ap.BACKEND_WEBUI


def current_capabilities(host) -> Any:
    """host 의 기능 스냅샷, 없으면 지금 WebUI 주소의 캐시(peek — HTTP 없음). ``_webui_context`` 와 같은 규칙."""
    capabilities = getattr(host, 'sam_extra_capabilities', None)
    if capabilities is not None:
        return capabilities
    try:
        from backends import get_backend
        api_url = str(getattr(get_backend(), 'api_url', '') or '')
        if api_url:
            from core.sam_extra_probe import peek_capabilities
            return peek_capabilities(api_url)
    except Exception:
        _logger.debug("sam-extra 스냅샷 조회 실패(무시)", exc_info=True)
    return None


def _model_name(host) -> str:
    combo = getattr(host, 'model_combo', None)
    try:
        return str(combo.currentText() or '') if combo is not None and hasattr(combo, 'currentText') else ''
    except Exception:
        return ''


def sampling_context(host, target: str, *, krea2: Optional[bool] = None) -> SamplingContext:
    if target not in ap.TARGETS:
        raise ValueError(f"모르는 샘플링 대상: {target!r}")
    if krea2 is None:
        probe = getattr(host, '_is_krea2_generation', None)
        try:
            krea2 = bool(probe()) if callable(probe) else False
        except Exception:
            krea2 = False
    host_backend = _host_backend()
    backend = ap.BACKEND_KREA2 if krea2 else host_backend
    capabilities = current_capabilities(host) if backend == ap.BACKEND_WEBUI else None
    return SamplingContext(target, backend, capabilities, _model_name(host), host_backend)


# ── 기여자 ──────────────────────────────────────────────────────────────────────
def _negpip(host, ctx: SamplingContext) -> Contribution:
    """NegPiP — 메인 체인은 _build_generation_payload·apply_alwayson_extensions 가 먼저 넣으므로(setdefault) 여기서는
    봉투(Comfy 보조 패스·손 재구성)용이다."""
    group = getattr(host, 'negpip_group', None)
    checked = getattr(group, 'isChecked', None)
    if not callable(checked) or not checked():
        return Contribution()
    return Contribution().add(ap.TITLE_NEGPIP, {"args": [True]})


def _anima_guidance(host, ctx: SamplingContext) -> Contribution:
    """Anima Guidance Suite: PAG/SEG/SLG · APG/CWM/SMC · Skimmed CFG · DCW/RDC/DAVE/CNS · Modulation · Detail Daemon.
    전부 꺼져 있으면 아무것도 넣지 않는다 (확장을 건드리지 않아야 결과가 동일)."""
    build = getattr(host, '_build_anima_settings', None)
    if not callable(build):
        return Contribution()
    try:
        from core import anima_guidance
        anima_settings = build()
        # Detail Daemon 값은 원본 노드 단위 그대로 간다(변환 없음). Hires Pass 를 켰는데 연결된 Forge 확장이
        # 인자 13 을 모르면 경고만 남긴다 — 판정은 연결 때 받아 둔 기능 스냅샷(여기서 HTTP 를 부르지 않는다).
        # ComfyUI 는 컴파일러가 dd_hires 로 패스를 고르므로 경고가 없다(남은 Forge 스냅샷도 보지 않는다).
        hires_note = anima_guidance.detail_daemon_hires_note(
            anima_settings, getattr(host, 'sam_extra_capabilities', None),
            comfyui=ctx.host_backend == ap.BACKEND_COMFY)
        if hires_note:
            _logger.warning(hires_note)
        blocks = anima_guidance.build_alwayson(anima_settings)
        summary = anima_guidance.describe_active(anima_settings)
        if summary:
            _logger.info("Anima Guidance 적용됨: %s", summary)
        contribution = Contribution()
        for title, block in blocks.items():
            contribution.add(title, block)   # 가이던스는 앱 기본값이 모두 끔 — 켜진 블록은 늘 사용자 값
        return contribution
    except Exception as e:
        # guidance는 부가 기능 — 실패해도 생성 자체는 진행되어야 한다
        _logger.warning("Anima Guidance 적용 실패 (무시하고 생성 진행): %s", e)
        return Contribution()


# 순서 = 블록 순서(NegPiP → 가이던스 → Anima38 → DoRA → DeGrid).
CONTRIBUTORS: tuple[Callable[[Any, SamplingContext], Contribution], ...] = (
    _negpip, _anima_guidance, _anima38, _dora_infer_mode, _vae_degrid)


def build_sampling_blocks(host, target: str, *, krea2: Optional[bool] = None) -> SamplingBlocks:
    """기여자 블록을 모으고(먼저 넣은 쪽이 이긴다) Forge 메인 요청이면 게이트한다. 알림은 띄우지 않는다."""
    ctx = sampling_context(host, target, krea2=krea2)
    blocks: dict = {}
    provenance: dict = {}
    notices: list = []
    for contributor in CONTRIBUTORS:
        try:
            contribution = contributor(host, ctx)
        except Exception as exc:
            _logger.warning("샘플링 블록 기여자 %s 실패 (무시하고 생성 진행): %s",
                            getattr(contributor, '__name__', contributor), exc)
            continue
        if contribution is None:
            continue
        for title, block in (contribution.blocks or {}).items():
            if title in blocks:
                continue
            blocks[title] = block
            provenance[title] = (contribution.provenance or {}).get(title, ap.PROVENANCE_USER)
        notices.extend(n for n in (contribution.notices or ()) if n is not None)
    dropped: tuple = ()
    if ctx.backend == ap.BACKEND_WEBUI and target != ap.TARGET_AUX:
        img2img = target != ap.TARGET_T2I
        result = ap.gate(blocks, ctx.capabilities, img2img=img2img, provenance=provenance)
        blocks, dropped = result.kept, result.dropped
        if dropped:
            _logger.info("샘플링 블록 제외(연결된 Forge 기능 스냅샷): %s", ", ".join(dropped))
        notices.extend(ap.gate_notices(result, provenance, img2img=img2img))
    return SamplingBlocks(blocks, notices, dropped, provenance, ctx)


def merge_sampling_blocks(payload: dict, blocks: dict) -> None:
    """payload['alwayson_scripts'] 에 setdefault — 호출자가 명시한 값(메인 체인의 NegPiP 등)을 덮지 않는다."""
    scripts = payload.setdefault("alwayson_scripts", {})
    if not isinstance(scripts, dict):
        return
    for title, block in blocks.items():
        scripts.setdefault(title, block)


def mark_request_provenance(payload, result: Optional[SamplingBlocks]) -> None:
    """Forge 메인 요청(t2i·i2i)이면 이번 빌드가 넣은 앱 기본값 블록의 출처를 페이로드의 비공개 키에 적는다(P10 검토 2).

    ``merge_sampling_blocks`` 뒤에 부른다. 호출자가 먼저 넣은 같은 제목(setdefault 가 이긴 블록)은 이번 빌드의 출처가
    아니므로 적지 않는다. 앱 기본값 블록이 없으면 페이로드를 바꾸지 않는다. 보조 봉투(aux — 손 재구성은 보내는 곳이
    게이트 뒤에 적는다)·Comfy·Krea2 는 적지 않는다."""
    ctx = getattr(result, "context", None)
    if not isinstance(payload, dict) or ctx is None or ctx.backend != ap.BACKEND_WEBUI or ctx.target == ap.TARGET_AUX:
        return
    scripts = payload.get("alwayson_scripts")
    if not isinstance(scripts, dict):
        return
    marks = {title: ap.PROVENANCE_APP_DEFAULT for title, block in (result.blocks or {}).items()
             if (result.provenance or {}).get(title) == ap.PROVENANCE_APP_DEFAULT and scripts.get(title) is block}
    if marks:
        existing = payload.get(FROZEN_PROVENANCE_KEY)
        payload[FROZEN_PROVENANCE_KEY] = {**(existing if isinstance(existing, Mapping) else {}), **marks}


# ── 알림: 만든 곳이 아니라 보내는 곳에서 (B15) ────────────────────────────────────
@dataclass
class _Parked:
    """host 의 한 칸 — 마지막 빌드(또는 그 사본)의 알림과 블록 출처. 페이로드 객체 동일성으로 묶는다."""

    payload: Any
    notices: list
    provenance: dict


def remember_sampling_notices(host, payload, notices: Iterable, *, provenance: Optional[Mapping] = None,
                              append: bool = False) -> None:
    """이 페이로드의 알림(과 블록 출처)을 host 에 한 칸 묶어 둔다(마지막 빌드만). 보내는 곳이 같은 객체로 꺼낸다.
    ``append`` 면 같은 페이로드에 이미 묶인 것 뒤에 붙인다(보낼 때 재게이트 — 옮겨 온 알림을 지우지 않게)."""
    items = [n for n in (notices or ()) if n is not None]
    origin = dict(provenance or {})
    parked = getattr(host, _NOTICE_SLOT, None)
    if append and isinstance(parked, _Parked) and parked.payload is payload:
        items = [*parked.notices, *items]
        origin = {**parked.provenance, **origin}
    try:
        setattr(host, _NOTICE_SLOT, _Parked(payload, items, origin))
    except Exception:
        pass


def _take_parked(host, payload) -> Optional[_Parked]:
    parked = getattr(host, _NOTICE_SLOT, None)
    if not isinstance(parked, _Parked) or parked.payload is not payload:
        return None
    try:
        setattr(host, _NOTICE_SLOT, None)
    except Exception:
        pass
    return parked


def take_sampling_notices(host, payload) -> list:
    """그 페이로드에 묶인 알림을 꺼낸다(한 번만). 다른 페이로드면 빈 목록 — 보내지 않은 빌드의 알림이 새지 않게."""
    parked = _take_parked(host, payload)
    return list(parked.notices) if parked else []


def take_sampling_state(host, payload) -> tuple[list, dict]:
    """(알림, 블록 출처)를 꺼낸다(한 번만). 스냅샷을 다시 게이트하는 보내는 곳(손 재구성)·동결 클릭이 출처를 쓴다."""
    parked = _take_parked(host, payload)
    return (list(parked.notices), dict(parked.provenance)) if parked else ([], {})


def move_sampling_notices(host, source, target) -> None:
    """페이로드를 복사·교체했으면(채팅 스냅샷 deepcopy, Comfy 컨트롤 스냅숏, start_generation 의 사본) 알림도 옮긴다."""
    parked = getattr(host, _NOTICE_SLOT, None)
    if isinstance(parked, _Parked) and parked.payload is source and source is not target:
        remember_sampling_notices(host, target, parked.notices, provenance=parked.provenance)


def show_sampling_notice_list(host, notices: Iterable) -> int:
    """보내는 곳에서: 샘플링 블록 알림을 토스트로(생성 전 경고와 같은 억제 시간 — 같은 알림은 한 번)."""
    items = [n for n in (notices or ()) if n is not None]
    if not items:
        return 0
    try:
        from core.sam_extra_notices import PRE_GENERATION_NOTICE_TTL_S
        from ui.sam_extra_notices_ui import show_notices
        return show_notices(host, items, ttl=PRE_GENERATION_NOTICE_TTL_S)
    except Exception:
        _logger.debug("샘플링 블록 알림 표시 실패(무시)", exc_info=True)
        return 0


def show_sampling_notices(host, payload) -> int:
    """보내는 곳에서: 그 페이로드에 묶인 샘플링 블록 알림을 토스트로."""
    return show_sampling_notice_list(host, take_sampling_notices(host, payload))


def freeze_sampling_payload(host, payload) -> list:
    """대기열에 넣을 동결 페이로드(XYZ·시드 탐색)를 만든 클릭에서 부른다 — 그 클릭이 '보내기'다(P7 검토 R2).

    빌드 때 묶어 둔 알림(게이트가 뺀 사용자 블록 경고·기여자 안내)을 꺼내 돌려준다. 대기열 항목은 그 블록이 이미 빠진
    사본이라 보낼 때(``regate_frozen_payload``)는 다시 볼 수 없으므로, 호출자가 항목을 넣은 뒤(대기열이 돌기 전)
    ``show_sampling_notice_list`` 로 띄운다. 앱 기본값 블록의 출처는 ``FROZEN_PROVENANCE_KEY`` 에 적어 두어 보낼 때
    다시 빠져도 경고하지 않게 한다(A6). 앱 기본값 블록이 없으면 페이로드를 바꾸지 않는다.
    """
    notices, provenance = take_sampling_state(host, payload)
    scripts = payload.get("alwayson_scripts") if isinstance(payload, dict) else None
    if isinstance(scripts, dict):
        app_default = {title: ap.PROVENANCE_APP_DEFAULT for title in scripts
                       if provenance.get(title) == ap.PROVENANCE_APP_DEFAULT}
        if app_default:
            payload[FROZEN_PROVENANCE_KEY] = app_default
    return notices


def _backend_capabilities(host, backend) -> tuple[str, Any]:
    """(보낼 백엔드 종류, 그 백엔드의 스냅샷|None). ``backend`` 가 없으면 지금 백엔드."""
    if backend is None:
        kind = _host_backend()
        return kind, (current_capabilities(host) if kind == ap.BACKEND_WEBUI else None)
    try:
        kind = str(backend.get_backend_type() or '')
    except Exception:
        return '', None
    if kind != ap.BACKEND_WEBUI:
        return kind, None
    capabilities = None
    try:
        from core.sam_extra_probe import peek_capabilities
        capabilities = peek_capabilities(str(getattr(backend, 'api_url', '') or ''))
    except Exception:
        capabilities = None
    if capabilities is None:
        try:
            from backends import get_backend
            if get_backend() is backend:
                capabilities = getattr(host, 'sam_extra_capabilities', None)
        except Exception:
            pass
    return kind, capabilities


def regate_frozen_payload(host, payload, *, backend=None) -> list:
    """동결 페이로드(시드 탐색·XYZ 대기열 — 만들 때 게이트했다)를 보내기 직전에 다시 게이트한다(critic B16).

    지금 스냅샷이 '없다'고 한 샘플링 블록만 빼고(모르면 만들 때 판단 그대로) 알림을 이 페이로드에 **덧붙여** 묶는다 —
    생성 전 확인(``check_before_generation``)이 띄운다. 앱 기본값 블록(클릭 때 ``FROZEN_PROVENANCE_KEY`` 에 적은 것)은
    로그만 남긴다(A6). 그 비공개 키는 Forge 가 아니면(Comfy·Krea2) 떼고, Forge 면 남은 앱 기본값 블록만 다시 적는다 —
    ``WebUIBackend._generate`` 가 요청 전에 떼어 메인 422 재시도 알림에 쓴다(P10 검토 2). SAM3·ADetailer(이미지 패스)는
    게이트하지 않는다.
    """
    try:
        if not isinstance(payload, dict):
            return []
        raw = payload.pop(FROZEN_PROVENANCE_KEY, None)
        provenance = dict(raw) if isinstance(raw, Mapping) else {}
        if str(payload.get('_generation_family') or '').lower() == 'krea2':
            return []
        kind, capabilities = _backend_capabilities(host, backend)
        if kind != ap.BACKEND_WEBUI:
            return []
        img2img = "init_images" in payload
        dropped = ap.drop_missing(payload, capabilities, img2img=img2img, provenance=provenance)
        if dropped:
            _logger.info("동결 페이로드 샘플링 블록 제외(연결된 Forge 기능 스냅샷): %s", ", ".join(dropped))
        notices = ap.gate_notices(ap.GateResult({}, dropped), provenance, img2img=img2img)
        remember_sampling_notices(host, payload, notices, append=True)
        ap.mark_provenance(payload, provenance)   # 남은 앱 기본값 블록 출처 — Forge 백엔드가 떼어 쓴다
        return notices
    except Exception:
        _logger.debug("동결 페이로드 재게이트 실패(무시하고 그대로 보냄)", exc_info=True)
        return []


__all__ = [
    "CONTRIBUTORS", "Contribution", "FROZEN_PROVENANCE_KEY", "SamplingBlocks", "SamplingContext",
    "build_sampling_blocks", "current_capabilities", "freeze_sampling_payload", "mark_request_provenance",
    "merge_sampling_blocks", "move_sampling_notices", "regate_frozen_payload", "remember_sampling_notices",
    "sampling_context", "show_sampling_notice_list", "show_sampling_notices", "take_sampling_notices",
    "take_sampling_state",
]
