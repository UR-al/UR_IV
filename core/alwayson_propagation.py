# core/alwayson_propagation.py
"""샘플링 블록(NegPiP·가이던스·Anima38·DoRA·VAE DeGrid)의 제목별 전달 규칙 — 순수 로직(Qt·네트워크·torch 없음).

메인 생성은 T2I 패널의 샘플링 블록을 ``alwayson_scripts`` 로 보낸다. 보조 패스(Refine·단독/배치 SAM3·단독 ADetailer·
손 재구성)도 같은 블록을 받아야 Forge 생성 안의 SAM3 패스·🎯 퀵 버튼과 결과가 같다(P7). 이 모듈은

- ``PROPAGATION``/``NEVER``   제목마다 어느 보조 패스·백엔드로 보내는지와 이유(허용 목록 — 모르는 제목은 보내지 않는다)
- ``gate``                    기능 스냅샷(``SamExtraCapabilities``)으로 '없다고 확인된' 스크립트를 뺀다. 모를 때는
                              제목별 규칙 × 블록 출처(사용자가 바꾼 값 / 앱 기본값)로 보낼지 정한다
- ``envelope``/``take_envelope``/``blocks_for``/``apply``  워커 settings 에 싣는 봉투와 보조 요청에 넣는 일

을 맡는다. 블록을 만드는 곳은 ``ui/sampling_blocks.py`` 하나다(메인 체인과 봉투가 같은 빌더를 쓴다).

**출처(provenance)**: 기여자(``ui/sampling_blocks.Contribution``)가 블록마다 ``PROVENANCE_USER``(사용자가 켠 값) 또는
``PROVENANCE_APP_DEFAULT``(앱이 스스로 넣은 기본값 — P8 DoRA·P9 Anima38 의 Forge ui-config 맞춤)를 적는다. 스냅샷을
모를 때(연결 직후·수집 실패) 보낼지는 ``Rule.when_unknown[출처]`` 로 정하고, 앱 기본값 블록이 빠질 때는 알림 없이
로그만 남긴다(사용자가 켜지 않은 기능으로 경고를 반복하지 않는다). 출처를 모르는 블록(봉투 밖·옛 스냅샷)은 사용자
값으로 본다.

Forge **메인** 요청은 앱 기본값 블록의 출처를 비공개 키 ``PROVENANCE_KEY`` 에 실어 백엔드까지 보낸다(``mark_provenance`` —
메인 체인·동결 재게이트·손 재구성). ``WebUIBackend._generate`` 가 요청 전에 떼어(``take_provenance``) Forge 가 DoRA·
Anima38 블록을 422 로 거절했을 때 알림 수준을 정한다: 앱 기본값 = 정보, 사용자 값·출처 모름 = 경고(P10 검토 2, A2·A6).

**모델에 묶인 블록**(``Rule.model_bound`` — Anima38): 블록 내용을 T2I 콤보 모델의 종류로 정했으므로(core/anima38.plan)
보조 패스는 실제로 쓰는 모델이 봉투의 모델과 같을 때만 보낸다(``drop_model_bound``, critic A7). Forge 보조 요청은
체크포인트를 지정하지 않아 Forge 에 지금 걸린 체크포인트로 돌고, Comfy 단독 후처리는 저장된 문맥의 모델로 돈다.
실제 모델을 모르거나 다르면 뺀다(정보 로그).

**최종 이미지 블록**(``Rule.passes`` 가 빈 행 → 메인 요청만 — VAE DeGrid): 확장이 이미지마다 저장 직전에 한 번 도는
후처리(``postprocess_image_after_composite``)라 메인 요청에만 싣는다. 보조 패스는 각자 새 요청이라 넣으면 이미 처리된
이미지에 한 번 더 걸린다 — ``blocks_for`` 는 어느 보조 패스에도 주지 않는다. 그래도 PROPAGATION 행이라 메인 게이트·동결
재게이트·봉투(메인 체인과 같은 모양, T15)는 다른 샘플링 블록과 같은 규칙을 탄다. ``final_image_titles`` 가 이 제목들을
돌려준다(Comfy 단독 후처리가 저장된 문맥에서 뗄 때 — 봉투 없는 API 호출).

NegPiP 은 Forge 에 보조 전달하지 않는다: sam-extra 내장 NegPiP(2026-09-30 sd-forge-negpip 편입,
extensions/forge_sam3_extension/scripts/negpip.py — 예전 extensions/sd-forge-negpip/scripts/negpip.py 와 파일 이름·제목·
계약이 같아 ADetailer 가 파일 이름 'negpip' 으로 자기 패스에 넣는 것도 그대로)은 ``ui()`` 가 None 이라 인자가 0개이고 always-on 이라 요청과 무관하게 돈다(modules/api/api.py:350-354). 보내도
효과는 없고 NegPiP 가 없는 Forge 에서 422 만 생긴다. Comfy 는 ``_add_negpip`` 로 적용하므로 보낸다. 손 재구성은 메인 경로
(``WebUIBackend._generate``)에 메인 T2I 스냅샷을 그대로 쓰므로 메인 생성처럼 보낸다.
"""
from __future__ import annotations

import copy
import logging
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Callable, Mapping, Optional

from core import anima38, anima_guidance, anima_model_kind, dora_infer_mode, sam3_args, vae_degrid

logger = logging.getLogger(__name__)

TARGET_T2I, TARGET_I2I, TARGET_AUX = "t2i", "i2i", "aux"
TARGETS = (TARGET_T2I, TARGET_I2I, TARGET_AUX)
AUX_REFINE, AUX_SAM3, AUX_ADETAILER, AUX_HAND = "refine", "sam3", "adetailer", "hand_repair"
AUX_PASSES = (AUX_REFINE, AUX_SAM3, AUX_ADETAILER, AUX_HAND)
BACKEND_WEBUI, BACKEND_COMFY, BACKEND_KREA2 = "webui", "comfyui", "krea2"
SEND, SKIP = "send", "skip"
PROVENANCE_USER, PROVENANCE_APP_DEFAULT = "user", "app_default"
PROVENANCES = (PROVENANCE_USER, PROVENANCE_APP_DEFAULT)
SETTINGS_KEY = "_alwayson_propagation"   # 워커 settings 안의 봉투. Forge/Comfy 페이로드로는 절대 나가지 않는다
# 요청 페이로드의 비공개 키 — {제목: PROVENANCE_APP_DEFAULT}(앱 기본값 블록만). Forge 메인 요청에만 적고 WebUIBackend 가
# 요청 전에 뗀다(Forge 요청 JSON 으로 나가지 않는다. Comfy 컴파일러는 이 키를 읽지 않는다). ui/sampling_blocks 의
# FROZEN_PROVENANCE_KEY 와 같은 키다.
PROVENANCE_KEY = "_sampling_provenance"
ENVELOPE_VERSION = 1

TITLE_NEGPIP = "NegPiP"
TITLE_DORA = dora_infer_mode.SCRIPT_NAME   # 확장 scripts/dora_infer_mode.py DORA_INFER_NAME
TITLE_DEGRID = vae_degrid.SCRIPT_NAME      # 확장 sam3ext/ui_vae_degrid.py TITLE (앱 노출)

_AUX_LABELS = {AUX_REFINE: "Refine", AUX_SAM3: "SAM3", AUX_ADETAILER: "ADetailer", AUX_HAND: "손 재구성"}


def aux_label(aux: str) -> str:
    return _AUX_LABELS.get(aux, "보조 작업")


def _truthy(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    return str(value or "").strip().lower() in {"1", "true", "yes", "on"}


def _dd_not_hires(block: Mapping, _aux: str, _aux_settings: Mapping) -> bool:
    """Detail Daemon 은 Hires Pass(인자 13)가 꺼져 있을 때만 보조 패스에 보낸다.

    img2img 는 hr 패스가 없어 ``on = (is_hr_pass == dd_hires)`` 가 dd_hires 끔일 때만 참이다(확장
    scripts/anima_detail_daemon.py:611-616). 13인자 옛 빌드는 인자 13 을 잘라 모든 패스에 거므로 켠 사용자가 원한
    'hires 만' 과 다르다 — 빼는 편이 맞다. 인자가 14개보다 짧으면 끔으로 본다. 기본 샘플러(DPM adaptive·HeunPP2)
    로는 미리 빼지 않는다 — Forge DD 는 그 패스의 샘플러로 다시 판정하고(:572, :602-605), Comfy 컴파일러도
    SAM3 패스를 자기 샘플러로 판정한다(core/comfy_workflow_compiler.py _add_detail_daemon).
    """
    args = block.get("args") if isinstance(block, Mapping) else None
    if not isinstance(args, (list, tuple)) or len(args) <= anima_guidance.DD_HIRES_INDEX:
        return True
    return not _truthy(args[anima_guidance.DD_HIRES_INDEX])


def _unknown(user: str = SEND, app_default: str = SEND) -> Mapping[str, str]:
    return MappingProxyType({PROVENANCE_USER: user, PROVENANCE_APP_DEFAULT: app_default})


_ALL_BACKENDS = frozenset({BACKEND_WEBUI, BACKEND_COMFY})


@dataclass(frozen=True)
class Rule:
    """한 제목의 전달 규칙.

    passes         이 블록을 받는 보조 패스. 비면 최종 이미지 블록 — 메인 요청에만 싣는다(VAE DeGrid)
    backends       보조 패스에 보내는 백엔드(메인 체인은 이 표로 거르지 않는다)
    feature        기능 스냅샷 이름. None = 스냅샷 기능이 아님(NegPiP — sam-extra 내장이지만 스냅샷에 없다) — 게이트하지 않는다
    when_unknown   스냅샷을 모를 때 출처별 SEND/SKIP
    reason         이 행의 근거(비면 테스트 실패)
    package        블록 생산자가 아직 없으면 그 패키지 id (생산자가 머지되면 지운다 — 지금은 없다)
    condition      (block, aux, aux_settings) → 보낼까 (DD Hires Pass)
    pass_backends  패스별 백엔드 예외(NegPiP 손 재구성)
    main_retry     메인 생성에서 Forge 가 이 제목을 422 로 거절하면 빼고 한 번 더 보낸다(P10, critic A2) — 앱이 Forge
                   설정에 맞춰 스스로 넣는 블록(DoRA·Anima38)만. 사용자가 패널에서 켠 가이던스는 지금처럼 실패·설명한다.
                   알림 수준은 요청의 블록 출처(``PROVENANCE_KEY``)로 정한다 — 앱 기본값 = 정보, 그 밖 = 경고(P10 검토 2)
    model_bound    블록 내용이 T2I 모델 종류로 정해진다(Anima38) — 보조 패스는 실제 모델이 봉투 모델과 같을 때만(A7)
    """

    passes: frozenset
    backends: frozenset
    feature: Optional[str]
    when_unknown: Mapping[str, str]
    reason: str
    package: str = ""
    condition: Optional[Callable[[Mapping, str, Mapping], bool]] = None
    pass_backends: Mapping[str, frozenset] = field(default_factory=lambda: MappingProxyType({}))
    main_retry: bool = False
    model_bound: bool = False

    def allows(self, aux: str, backend: str) -> bool:
        if aux not in self.passes:
            return False
        return backend in self.pass_backends.get(aux, self.backends)

    def unknown_action(self, provenance: str) -> str:
        return self.when_unknown.get(provenance, self.when_unknown.get(PROVENANCE_USER, SEND))


PROPAGATION: Mapping[str, Rule] = MappingProxyType({
    TITLE_NEGPIP: Rule(
        passes=frozenset(AUX_PASSES), backends=frozenset({BACKEND_COMFY}), feature=None,
        when_unknown=_unknown(),
        pass_backends=MappingProxyType({AUX_HAND: _ALL_BACKENDS}),
        reason="기능 스냅샷 밖(sam-extra 내장 NegPiP, 예전 sd-forge-negpip). Forge 판은 인자 0개·always-on 이라 보조 "
               "요청에 넣어도 효과가 없고 NegPiP 가 없는 Forge 에서 422 만 난다 — Comfy(_add_negpip)에만 보낸다. 손 재구성은 "
               "메인 경로라 메인 T2I 처럼"),
    anima_guidance.SCRIPT_PERTURBATION: Rule(
        passes=frozenset(AUX_PASSES), backends=_ALL_BACKENDS, feature="anima_guidance", when_unknown=_unknown(),
        reason="SAM3 p2 는 SAM3 만 빼고 바깥 alwayson 을 그대로 돌린다(sam3ext/inpaint_core.py:142-158) — 생성 안 "
               "SAM3·🎯 퀵 버튼과 같게. 모를 때 보내는 것은 지금 동작(가이던스 블록은 늘 사용자가 켠 값)"),
    anima_guidance.SCRIPT_SKIMMED_CFG: Rule(
        passes=frozenset(AUX_PASSES), backends=_ALL_BACKENDS, feature="skimmed_cfg", when_unknown=_unknown(),
        reason="SAM3 p2 에서 돈다. ADetailer 내부 패스에는 unet 패치가 닿지 않지만 '생성 안과 같은 상태' 규칙 하나로 보낸다"),
    anima_guidance.SCRIPT_DETAIL_DAEMON: Rule(
        passes=frozenset(AUX_PASSES), backends=_ALL_BACKENDS, feature="detail_daemon", when_unknown=_unknown(),
        condition=_dd_not_hires,
        reason="img2img 는 hr 패스가 없어 dd_hires 끔일 때만 켜진다(anima_detail_daemon.py:611-616). ADetailer 내부 "
               "패스는 _DD['on'] 이 남아 닿는다(:656-659)"),
    anima38.SCRIPT_NAME: Rule(
        passes=frozenset(AUX_PASSES), backends=_ALL_BACKENDS, feature="anima38",
        when_unknown=_unknown(user=SEND, app_default=SKIP), main_retry=True, model_bound=True,
        reason="블록이 없으면 부정 커넥터·Bypass 가 끔으로 돌아간다(scripts/anima_3_8b.py ARG_DEFAULTS). SAM3 p2·"
               "ADetailer 는 바깥 설치를 물려받으므로(process_batch _sam3_outer) 부모 요청에 넣으면 된다. 앱 기본값 "
               "블록(부정 커넥터 켬 — core/anima38.APP_T2I_DEFAULTS)은 확장 확인 전에는 보내지 않는다(모르는 Forge 에 "
               "앱이 켠 기능으로 422 를 내지 않게, critic A2). 블록은 T2I 모델 종류로 만들었으므로 보조 패스는 실제 "
               "모델이 같을 때만(model_bound, A7)"),
    TITLE_DORA: Rule(
        passes=frozenset(AUX_PASSES), backends=frozenset({BACKEND_WEBUI}), feature="dora",
        when_unknown=_unknown(user=SKIP, app_default=SKIP), main_retry=True,
        reason="전역 상태 — 블록이 없는 요청은 순정으로 되돌리고 LoRA 를 다시 합친다(scripts/dora_infer_mode.py:145-155). "
               "한 커밋(4d0028e)에만 있는 최신 스크립트이고 앱 기본이 켬이라 모르면 늘 보내지 않는다(422 위험, critic "
               "A2 — 사용자가 바꾼 값도: 모를 때 보내지 않는 것은 결과가 순정일 뿐 요청은 성공하고, 보냈다가 422 면 생성 "
               "전체가 실패한다 — 보조 패스는 전달 블록 422 를, 메인 생성은 main_retry 로 이 블록 422 를 빼고 한 번 더 보낸다). "
               "Comfy 컴파일러는 모르는 제목을 무시하므로 만들지 않는다(core/dora_infer_mode.plan)"),
    TITLE_DEGRID: Rule(
        passes=frozenset(), backends=_ALL_BACKENDS, feature="degrid",
        when_unknown=_unknown(user=SEND, app_default=SKIP),   # main_retry 기본 False — Rule 문서의 규칙(사용자가 켠 블록)
        reason="최종 이미지 후처리(postprocess_image_after_composite) — 메인 요청 결과에만 이미지마다 한 번. 보조 패스"
               "(Refine·단독/배치 SAM3·ADetailer·손 재구성)는 각자 새 요청이라 넣으면 이미 DeGrid 된 이미지에 한 번 더 "
               "걸린다(확장도 _sam3_inner·_ad_inner 에서는 돌지 않는다). 사용자가 켠 값이라 모를 때는 보내고, 스크립트가 "
               "없으면 가이던스처럼 422 를 설명한다(main_retry 없음 — Rule 문서의 규칙)"),
})

NEVER: Mapping[str, str] = MappingProxyType({
    sam3_args.SCRIPT_SAM3: "보조 경로 자신의 이미지 패스 — 복사하면 T2I 의 감지 대상·인페인트 프롬프트로 한 번 더 "
                           "인페인트한다(Forge 도 파생 패스에서 스스로 뺀다: inpaint_core.py:56, !sam3.py:394-397)",
    "ADetailer": "이미지 패스 — 복사하면 부모 패스 뒤에 T2I 얼굴 보정이 한 번 더 돈다(_ad_disabled 는 SAM3 p2 에만)",
    "Anima VAE 2x (spacepxl decoder)": "HOLD(M9) — 앱이 만들지 않는다. 2x 디코더가 인페인트 패스 출력 크기를 바꿀 수 있다",
    "Anima Reference PoC (shape logger)": "디버그용(N9) — 앱이 만들지 않는다",
    "SAM Extra Anima sparse LoRA": "인자 0개 자동 훅(N7) — 페이로드에 나오지 않는다",
    "SAM3 LoRA Manager bridge": "인자 0개 숨은 Gradio 브리지(N8) — 페이로드에 나오지 않는다",
})

_CANONICAL = {title.casefold(): title for title in (*PROPAGATION, *NEVER)}


def canonical_title(title: Any) -> Optional[str]:
    """제목 → PROPAGATION/NEVER 의 정식 제목(Forge 처럼 대소문자·앞뒤 공백 무시). 표에 없으면 None."""
    return _CANONICAL.get(str(title or "").strip().casefold())


def _provenance_of(provenance: Optional[Mapping], title: str) -> str:
    if not isinstance(provenance, Mapping):
        return PROVENANCE_USER
    value = provenance.get(title)
    if value is None:
        canonical = canonical_title(title)
        value = provenance.get(canonical) if canonical else None
    return value if value in PROVENANCES else PROVENANCE_USER


def provenance_of(provenance: Optional[Mapping], title: Any) -> str:
    """블록 출처(PROVENANCE_*) — 적힌 것이 없거나 모르는 값이면 사용자 값. 제목은 대소문자·앞뒤 공백을 무시한다."""
    return _provenance_of(provenance, str(title or ""))


def take_provenance(payload: Any) -> tuple:
    """(비공개 출처 키를 뗀 payload, 출처 dict) — 백엔드가 요청 전에 부른다(P10 검토 2).

    키가 없으면 **같은 객체**와 {} 를 돌려준다(요청이 바이트 단위로 같다). 입력은 바꾸지 않는다(얕은 사본에서 뗀다).
    모양이 틀린 값은 버린다(키는 늘 뗀다)."""
    if not isinstance(payload, Mapping) or PROVENANCE_KEY not in payload:
        return payload, {}
    out = dict(payload)
    raw = out.pop(PROVENANCE_KEY, None)
    return out, (dict(raw) if isinstance(raw, Mapping) else {})


def mark_provenance(payload: Any, provenance: Optional[Mapping]) -> None:
    """``payload`` 에 지금 실린 샘플링 블록 중 앱 기본값인 것의 출처를 ``PROVENANCE_KEY`` 에 적는다(제자리). 없으면 키를
    뺀다(기본 페이로드는 그대로). Forge 메인 요청으로 보낼 페이로드에만 부른다 — ``WebUIBackend._generate`` 가 떼어 쓴다."""
    if not isinstance(payload, dict):
        return
    scripts = payload.get("alwayson_scripts")
    marks = {title: PROVENANCE_APP_DEFAULT for title in (scripts if isinstance(scripts, Mapping) else ())
             if _provenance_of(provenance, title) == PROVENANCE_APP_DEFAULT}
    if marks:
        payload[PROVENANCE_KEY] = marks
    else:
        payload.pop(PROVENANCE_KEY, None)


def script_available(capabilities: Any, title: str, *, img2img: bool) -> Optional[bool]:
    """스냅샷이 known 이면 그 제목이 (img2img 면 img2img 목록, 아니면 txt2img 목록에) 있는지, 모르면 None.

    None·unknown·error·unreachable·not_applicable 과 가짜 객체(``known`` 이 True 가 아니거나 ``script`` 가 없는
    테스트 더블)는 모름이다.
    """
    if getattr(capabilities, "known", None) is not True:
        return None
    script = getattr(capabilities, "script", None)
    if not callable(script):
        return None
    try:
        detail = script(title)
    except Exception:
        return None
    if not isinstance(detail, Mapping):
        return None
    value = detail.get("img2img" if img2img else "present")
    return value if isinstance(value, bool) else None


@dataclass(frozen=True)
class GateResult:
    kept: dict
    dropped_missing: tuple = ()     # 스냅샷이 '없다'고 한 제목
    dropped_unknown: tuple = ()     # 모르는데 when_unknown=SKIP 이라 뺀 제목

    @property
    def dropped(self) -> tuple:
        return (*self.dropped_missing, *self.dropped_unknown)


def gate(blocks: Mapping, capabilities: Any, *, img2img: bool,
         provenance: Optional[Mapping] = None) -> GateResult:
    """기능 스냅샷으로 샘플링 블록을 거른다. 표에 없는 제목·feature None(NegPiP)은 그대로 둔다(순서 유지)."""
    kept: dict = {}
    missing, unknown = [], []
    for title, block in (blocks or {}).items():
        canonical = canonical_title(title)
        rule = PROPAGATION.get(canonical) if canonical else None
        if rule is None or rule.feature is None:
            kept[title] = block
            continue
        available = script_available(capabilities, canonical, img2img=img2img)
        if available is True:
            kept[title] = block
        elif available is False:
            missing.append(title)
        elif rule.unknown_action(_provenance_of(provenance, title)) == SEND:
            kept[title] = block
        else:
            unknown.append(title)
    return GateResult(kept, tuple(missing), tuple(unknown))


def drop_missing(payload: dict, capabilities: Any, *, img2img: bool,
                 provenance: Optional[Mapping] = None) -> tuple:
    """동결 페이로드(시드 탐색·XYZ 대기열)를 보낼 때: 스냅샷이 '없다'고 확인한 샘플링 블록만 뺀다(제자리).

    모를 때는 만들 때의 판단을 그대로 둔다(출처와 무관하게 빼지 않는다). SAM3·ADetailer 같은 이미지 패스는 게이트하지
    않는다(지금처럼 보내고 422 면 오류 알림). 뺀 제목을 돌려준다 — 알릴지는 호출자가 같은 ``provenance`` 로
    ``gate_notices`` 에 묻는다(출처가 없는 블록은 사용자 값).
    """
    scripts = payload.get("alwayson_scripts") if isinstance(payload, dict) else None
    if not isinstance(scripts, dict):
        return ()
    sampling = {name: block for name, block in scripts.items() if canonical_title(name) in PROPAGATION}
    result = gate(sampling, capabilities, img2img=img2img, provenance=provenance)
    for name in result.dropped_missing:
        scripts.pop(name, None)
    return result.dropped_missing


def envelope(blocks: Mapping, *, source: str, model: str = "", backend: str = "",
             provenance: Optional[Mapping] = None) -> dict:
    """샘플링 블록 → 워커 settings 에 실을 봉투. PROPAGATION 제목만 정식 제목으로, 깊은 복사한다."""
    out_blocks: dict = {}
    out_provenance: dict = {}
    for title, block in (blocks or {}).items():
        canonical = canonical_title(title)
        if canonical not in PROPAGATION or canonical in out_blocks:
            continue
        out_blocks[canonical] = copy.deepcopy(block)
        out_provenance[canonical] = _provenance_of(provenance, title)
    return {"version": ENVELOPE_VERSION, "source": str(source or ""), "model": str(model or ""),
            "backend": str(backend or ""), "blocks": out_blocks, "provenance": out_provenance}


def is_envelope(value: Any) -> bool:
    return (isinstance(value, Mapping) and value.get("version") == ENVELOPE_VERSION
            and isinstance(value.get("blocks"), Mapping))


def take_envelope(settings: Any) -> tuple[dict, Optional[dict]]:
    """(봉투를 뺀 settings 사본, 봉투|None). 입력은 바꾸지 않는다. 모양이 틀린 봉투는 버린다(키는 늘 뺀다)."""
    if not isinstance(settings, Mapping):
        return {}, None
    clean = dict(settings)
    raw = clean.pop(SETTINGS_KEY, None)
    return clean, (dict(raw) if is_envelope(raw) else None)


def envelope_provenance(value: Any) -> dict:
    raw = value.get("provenance") if isinstance(value, Mapping) else None
    return dict(raw) if isinstance(raw, Mapping) else {}


def blocks_for(source: Any, aux: str, *, backend: str, aux_settings: Optional[Mapping] = None) -> dict:
    """봉투 또는 alwayson dict → 이 보조 패스·백엔드에 보낼 블록(정식 제목, 깊은 복사, 입력 순서).

    SAM3 'Mask only' 는 인페인트 패스가 없어 아무것도 보내지 않는다(sam3_args.mode_of 정규화 — 확장에 가는 값과 같은 판정).
    """
    if aux not in AUX_PASSES:
        raise ValueError(f"모르는 보조 패스: {aux!r}")
    settings = aux_settings if isinstance(aux_settings, Mapping) else {}
    if aux == AUX_SAM3 and sam3_args.mode_of(dict(settings)) == sam3_args.MODE_MASK_ONLY:
        return {}
    scripts = source.get("blocks") if is_envelope(source) else source
    if not isinstance(scripts, Mapping):
        return {}
    out: dict = {}
    for title, block in scripts.items():
        canonical = canonical_title(title)
        rule = PROPAGATION.get(canonical) if canonical else None
        if rule is None or canonical in out or not rule.allows(aux, backend):
            continue
        if rule.condition is not None and not rule.condition(block, aux, settings):
            continue
        out[canonical] = copy.deepcopy(block)
    return out


def drop_model_bound(blocks: Mapping, *, envelope_model: Any, actual_model: Any) -> tuple:
    """보조 패스(critic A7): 모델에 묶인 블록(``Rule.model_bound`` — Anima38)은 실제로 쓰는 모델이 봉투의 모델(클릭한
    순간의 T2I 콤보)과 같을 때만 남긴다 → (남긴 블록, 뺀 제목). 한쪽이라도 모르면 뺀다(블록 종류를 믿을 수 없다).
    이름 비교는 폴더·해시·확장자·대소문자를 무시한다(core/anima_model_kind.same_checkpoint — Forge short_title·별칭).
    입력은 바꾸지 않는다."""
    kept: dict = {}
    dropped = []
    same = anima_model_kind.same_checkpoint(envelope_model, actual_model)
    for title, block in (blocks or {}).items():
        canonical = canonical_title(title)
        rule = PROPAGATION.get(canonical) if canonical else None
        if rule is not None and rule.model_bound and not same:
            dropped.append(title)
            continue
        kept[title] = block
    return kept, tuple(dropped)


def has_model_bound(blocks: Mapping) -> bool:
    """모델에 묶인 블록이 있나 — 있을 때만 백엔드가 실제 모델을 알아본다(Forge 는 GET 한 번)."""
    for title in (blocks or {}):
        canonical = canonical_title(title)
        rule = PROPAGATION.get(canonical) if canonical else None
        if rule is not None and rule.model_bound:
            return True
    return False


def final_image_titles() -> tuple:
    """최종 이미지 블록의 정식 제목(``Rule.passes`` 가 빈 행 — 메인 요청에만 싣는다). 보조 경로가 저장된 문맥·호출자
    페이로드에서 뗄 때 쓴다(PROPAGATION 순서)."""
    return tuple(title for title, rule in PROPAGATION.items() if not rule.passes)


def main_retry_titles(payload: Any) -> tuple:
    """메인 생성 요청에서 Forge 가 422 로 거절하면 빼고 다시 보낼 제목(``Rule.main_retry``) — payload 에 적힌 표기 그대로.

    어느 제목을 다시 보낼 수 있는지는 이 제목표로 정하고, 알림 수준(앱 기본값 = 정보, 사용자 값·모름 = 경고)은 요청의
    비공개 출처(``take_provenance`` — 백엔드가 요청 전에 뗀다)로 정한다(critic A2·A6, P10 검토 2). 외부 생성 API 는
    ``alwayson_scripts`` 를 받지 않으므로(core/generation_api.py 금지 키) 이 제목이 호출자에게서 올 수 없다."""
    scripts = payload.get("alwayson_scripts") if isinstance(payload, Mapping) else None
    if not isinstance(scripts, Mapping):
        return ()
    out = []
    for name in scripts:
        canonical = canonical_title(name)
        rule = PROPAGATION.get(canonical) if canonical else None
        if rule is not None and rule.main_retry:
            out.append(name)
    return tuple(out)


def apply(payload: dict, blocks: Mapping) -> tuple:
    """``alwayson_scripts`` 에 setdefault(대소문자 무시) — 보조 경로가 만든 SAM3·ADetailer 를 덮지 않는다.

    실제로 넣은 제목을 돌려준다(Forge 가 거절하면 그것만 빼고 다시 보낼 목록 — core/forge_optional_parts).
    """
    scripts = payload.setdefault("alwayson_scripts", {})
    if not isinstance(scripts, dict):
        return ()
    present = {str(name).strip().casefold() for name in scripts}
    inserted = []
    for title, block in (blocks or {}).items():
        folded = str(title).strip().casefold()
        if folded in present:
            continue
        scripts[title] = copy.deepcopy(block)
        present.add(folded)
        inserted.append(title)
    return tuple(inserted)


def gate_notices(result: GateResult, provenance: Optional[Mapping] = None, *, img2img: bool,
                 aux: str = "") -> list:
    """게이트가 뺀 제목 → 알림. 앱 기본값 블록은 로그만(반복 경고 금지), 사용자 값만 알린다.

    ``aux`` 가 있으면 보조 패스 문구(``CODE_PROPAGATION_DROPPED``), 없으면 메인 생성 문구(``CODE_BLOCK_NOT_SENT``).
    """
    from core import sam_extra_notices as sn

    out = []
    for title in result.dropped_missing:
        if _provenance_of(provenance, title) == PROVENANCE_APP_DEFAULT:
            logger.info("[sam-extra] 앱 기본값 블록 '%s' — 연결된 Forge 에 없어 보내지 않음", title)
            continue
        out.append(sn.propagation_dropped_notice(title, aux_label(aux)) if aux
                   else sn.block_not_sent_notice(title, img2img=img2img))
    for title in result.dropped_unknown:
        if _provenance_of(provenance, title) == PROVENANCE_APP_DEFAULT:
            logger.info("[sam-extra] 앱 기본값 블록 '%s' — 확장 확인 전이라 보내지 않음", title)
            continue
        out.append(sn.block_deferred_notice(title))
    return out


__all__ = [
    "AUX_ADETAILER", "AUX_HAND", "AUX_PASSES", "AUX_REFINE", "AUX_SAM3", "BACKEND_COMFY", "BACKEND_KREA2",
    "BACKEND_WEBUI", "ENVELOPE_VERSION", "GateResult", "NEVER", "PROPAGATION", "PROVENANCES",
    "PROVENANCE_APP_DEFAULT", "PROVENANCE_KEY", "PROVENANCE_USER", "Rule", "SEND", "SETTINGS_KEY", "SKIP", "TARGETS",
    "TARGET_AUX", "TARGET_I2I", "TARGET_T2I", "TITLE_DEGRID", "TITLE_DORA", "TITLE_NEGPIP", "apply", "aux_label",
    "blocks_for", "canonical_title", "drop_missing", "drop_model_bound", "envelope", "envelope_provenance",
    "final_image_titles", "gate",
    "gate_notices", "has_model_bound", "is_envelope", "main_retry_titles", "mark_provenance", "provenance_of", "script_available",
    "take_envelope", "take_provenance",
]
