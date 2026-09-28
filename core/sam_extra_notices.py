# core/sam_extra_notices.py
"""sam-extra(Forge 확장) 실패·알림을 사용자에게 — 순수 로직 (Qt·네트워크 없음).

확장은 실패해도 생성을 멈추지 않고 infotext 에 흔적만 남긴다. 앱은 그 흔적을 읽지 않아서 사용자는
'켰는데 아무 일도 없다'를 겪었다 (gap matrix S7·M1·C2, 다-2, 다-5, 라-7). 이 모듈은 보낸 요청
(payload)과 Forge 응답 info(JSON)를 맞대 보고 한국어 알림(``Notice``)을 만든다. 토스트로 띄우는 일은
``ui/sam_extra_notices_ui.py`` 가 한다.

- ``result_notices(info, payload)``        생성 뒤: 'SAM3 Error', SAM3 적용 흔적 없음, 원본 기준 SAM3 요청(단독 SAM3·
                                            Refine)의 'SAM3 Source' 없음(확장 미지원, 정보)·'output (<이유>)'(경고),
                                            'Anima38: off: …',
                                            PAG/SEG/SLG 를 켰는데 'Anima Perturbation Guidance' 가 없음,
                                            보낸 DoRA 방식의 'DoRA mode'·'DoRA inserted' 가 없음(훅 폴백, P8),
                                            부분 LoRA 추측 변환('Anima sparse LoRA', 정보)
- ``pre_generation_notices(payload, …)``   생성 전: CFG 1 에서 켠 SMC/APG/CWM(Forge 는 CFG 1 에 네거티브를
                                            인코딩하지 않아 v0.30 은 어느 빌드든 건너뜀, 가드 없는 v0.21.2 계열은
                                            망가짐), SAM3 체크포인트 'sam3.pt' 가 없어 HF 3.4 GB 자동 다운로드,
                                            스크립트 없음(→ 422)
- ``explain_rejected_request(status, body, payload)``  Forge HTTP 422 본문 → 어느 확장·기능인지
  (``missing_script_title`` 은 같은 규칙으로 제목만 — 보조 패스 재시도 판정 core/forge_optional_parts)
- ``block_*_notice``/``propagation_*_notice``  샘플링 블록 게이트·보조 패스 전달(P7, core/alwayson_propagation)
- ``dora_*_notice``                         DoRA 추론 방식 — Comfy 순정 안내·라이브 선택지 불일치·첫 로드 앱 기본값(P8)
- ``anima38_*_notice``                      Anima 3.8B — 첫 로드 앱 기본값(부정 커넥터 켬)·Comfy v1 자동 켜짐 폐지 안내(P9).
                                            'off: install failed (RuntimeError)' 힌트는 결과 모델 종류(비 Anima)를 먼저 본다
- ``app_block_retried_notice``/``block_retried_notice``  메인 생성 재시도(DoRA·Anima38 422 — 앱 기본값이면 정보, 사용자
                                            값·출처 모름이면 경고, P10 검토 2)
- ``forge_option_*_notice``                 Forge 옵션 덮어쓰기(P10,
                                            설정 잠금으로 거절됐던 옵션은 기억해 보내지 않음 — ``forge_option_frozen_notice``)
- ``NoticedImage``                          단독 SAM3/Refine/ADetailer 결과(base64 str)에 info·알림을 싣는 str
- ``NoticeThrottle``                        같은 알림을 잠시 한 번만 (자동화·배치에서 토스트 폭주 방지)

infotext 키는 확장 v0.30.0 소스 기준이다: 'SAM3 Error'·'SAM3 Enable'·'SAM3 Version'·'SAM3 Source'(scripts/!sam3.py),
'Anima38'(scripts/anima_3_8b.py STATUS_KEY — 옛 이름 'Anima 3.8B', Forge 파서가 자른 '8B' 도 읽는다),
'Anima Perturbation Guidance'(scripts/anima_safe_pag.py), 'Anima sparse LoRA'(sam3ext/anima_lora_blocks.py
INFOTEXT_SPARSE_GUESS_KEY). ``tests/test_sam_extra_notices.py`` 가 설치된 확장
소스에 이 키들이 그대로 있는지 AST 로 본다.

버전에 따라 뜻이 다른 판단(CFG 1 가드, PAG OOM 폴백은 v0.30 부터 — v0.21.2 PAG 빌드는 인자 57개)은 기능
스냅샷(``SamExtraCapabilities.anima_guidance_argc``)으로 고른다. 스냅샷이 없으면(확인 전·ComfyUI) 지금 설치된
확장(v0.30) 기준으로 판단하고 경고 로그를 한 번 남긴다. 페이로드 빌더에서 새 HTTP 요청은 하지 않는다.
"""
from __future__ import annotations

import json
import logging
import math
import re
import time
from dataclasses import dataclass, replace
from pathlib import PurePath, PurePosixPath, PureWindowsPath
from typing import Any, Callable, Iterable, Mapping, Optional, Sequence
from urllib.parse import urlsplit

from core import anima38, anima_guidance, anima_model_kind, dora_infer_mode, sam3_args

logger = logging.getLogger(__name__)

# ── 확장이 쓰는 infotext 키 ──────────────────────────────────────────────────
KEY_SAM3_ERROR = "SAM3 Error"
KEY_SAM3_ENABLE = "SAM3 Enable"
KEY_SAM3_VERSION = "SAM3 Version"
# 원본 기준 요청(state 'sam3_source_image': 'init' — core/sam3_args.with_init_source)을 받았을 때만 쓴다: 'init image' =
# init 이미지로 돌았다, 'output (<이유>)' = 조건이 안 맞아 예전처럼 부모 패스 출력으로(scripts/!sam3.py INFOTEXT_SOURCE·
# SOURCE_NOTE_*, SEMANTIC_PINS 가 지킨다). 요청하지 않은 생성에는 키가 없다.
KEY_SAM3_SOURCE = "SAM3 Source"
SAM3_SOURCE_INIT_NOTE = "init image"
SAM3_SOURCE_OUTPUT_NOTE = "output"
KEY_ANIMA38_STATUS = "Anima38"                          # v2 bundle / v1 adapter / bypass / off: <이유>
ANIMA38_LEGACY_STATUS_KEYS = ("Anima 3.8B", "8B")       # 옛 키, Forge 파서가 '.' 에서 자른 이름
KEY_PAG = "Anima Perturbation Guidance"                 # PAG/SEG/SLG 가 실제로 붙었을 때만 쓴다
# 부분(sparse) Anima LoRA 를 순정 추측 변환으로 로드했을 때만 — 'Forge guess (<파일> 28->40, …)'
# (sam3ext/anima_lora_blocks.py INFOTEXT_SPARSE_GUESS_KEY, 기록은 scripts/anima_lora_blocks.py). 건너뛴 부분 LoRA
# 는 gr.Warning 만 남겨 API 응답에 흔적이 없다 — 앱은 알 수 없다(gap matrix M4).
KEY_SPARSE_LORA_GUESS = "Anima sparse LoRA"

# generation info 에 알림을 싣는 키 (list[dict]) — 백엔드가 넣고 UI 가 읽는다. '_' 로 시작해 메타데이터로 새지 않는다.
INFO_KEY = "_sam_extra_notices"

LEVEL_INFO, LEVEL_WARNING, LEVEL_ERROR = "info", "warning", "error"
LEVELS = (LEVEL_INFO, LEVEL_WARNING, LEVEL_ERROR)

CODE_SAM3_ERROR = "sam3_error"
CODE_SAM3_NOT_APPLIED = "sam3_not_applied"
CODE_ANIMA38_OFF = "anima38_off"
CODE_ANIMA38_NOT_APPLIED = "anima38_not_applied"
CODE_PAG_DROPPED = "pag_dropped"
CODE_LORA_SPARSE_GUESS = "lora_sparse_guess"
CODE_CFG1_CFG_BASE = "cfg1_cfg_base"
CODE_SAM3_HF_DOWNLOAD = "sam3_hf_download"
CODE_SCRIPT_MISSING = "script_missing"
CODE_EXTENSION_MISSING = "extension_missing"
CODE_REQUEST_REJECTED = "request_rejected"
# 샘플링 블록 게이트·보조 패스 전달(P7, core/alwayson_propagation)
CODE_BLOCK_NOT_SENT = "block_not_sent"                # 메인 생성: 스냅샷이 '없다'고 한 스크립트 블록을 빼고 보냄
CODE_BLOCK_DEFERRED = "block_deferred"                # 확장 확인 전이라 이번 요청에는 보내지 않음
CODE_PROPAGATION_DROPPED = "propagation_dropped"      # 보조 패스: 연결된 Forge 에 없어 전달하지 않음
CODE_PROPAGATION_RETRIED = "propagation_retried"      # 보조 패스: Forge 가 전달 블록을 422 로 거절 → 빼고 다시 보냄
# DoRA 추론 방식(P8, core/dora_infer_mode)
CODE_DORA_NOT_APPLIED = "dora_not_applied"            # 결과: 보낸 방식의 infotext 기록('DoRA mode'·'DoRA inserted')이 없음
CODE_DORA_COMFY_STOCK = "dora_comfy_stock"            # 생성 전: ComfyUI 는 순정으로만 합친다(사용자가 바꾼 값일 때만)
CODE_DORA_CHOICE = "dora_choice"                      # 생성 전: 고른 값이 연결된 확장 선택지에 없어 보내지 않음
CODE_DORA_APP_DEFAULT = "dora_app_default"            # 첫 로드: 옛 설정에 키가 없어 앱 기본값(LyCORIS)으로 시작
# Anima 3.8B(P9, core/anima38)
CODE_ANIMA38_APP_DEFAULT = "anima38_app_default"      # 첫 로드: 옛 설정에 키가 없어 앱 기본값(부정 커넥터 켬)으로 시작
CODE_ANIMA38_COMFY_V1 = "anima38_comfy_v1"            # 생성 전(Comfy): Qwen3.5·어댑터 모듈 쌍이 있어도 v1 은 카드에서 켤 때만
# 메인 생성 재시도·Forge 옵션 덮어쓰기(P10, core/forge_override_settings·core/forge_optional_parts)
CODE_APP_BLOCK_RETRIED = "app_block_retried"          # 메인 생성: 앱이 넣은 블록(DoRA·Anima38)을 Forge 가 422 로 거절 → 빼고 다시
CODE_BLOCK_RETRIED = "block_retried"                  # 메인 생성: 사용자 값(또는 출처 모름) 블록을 422 로 거절 → 빼고 다시(경고)
CODE_FORGE_OPTION_MISSING = "forge_option_missing"    # 설정한 Forge 옵션이 연결된 sam-extra 에 없어 보내지 않음
CODE_FORGE_OPTION_UNVERIFIED = "forge_option_unverified"   # 옵션 목록을 확인하지 못해 보내지 않음
CODE_FORGE_OPTION_REJECTED = "forge_option_rejected"  # Forge 가 앱이 넣은 옵션을 500(KeyError·설정 잠금)으로 거절 → 빼고 다시
CODE_FORGE_OPTION_FROZEN = "forge_option_frozen"      # 설정 잠금으로 거절됐던 옵션 — 기억해 두고 보내지 않음(P10 검토 3)
# 단독 SAM3·Refine 의 원본 기준 요청(core/sam3_args.with_init_source) — denoise 0 부모 패스의 VAE 왕복 드리프트
CODE_SAM3_SOURCE_UNSUPPORTED = "sam3_source_unsupported"   # 결과: 'SAM3 Source' 없음 — 이 sam-extra 는 요청을 모른다(정보)
CODE_SAM3_SOURCE_FALLBACK = "sam3_source_fallback"         # 결과: 'output (<이유>)' — 요청했지만 부모 출력으로 돌았다(경고)

# SAM3 가 적용되지 않았다는 뜻인 알림 — 단독 SAM3/Refine 은 그 결과(원본 또는 VAE 왕복본)를 저장하지 않는다
SAM3_FAILURE_CODES = (CODE_SAM3_ERROR, CODE_SAM3_NOT_APPLIED)

HF_CHECKPOINT_NAME = "sam3.pt"          # sam3ext/core.py HF_CHECKPOINT_NAME
HF_CHECKPOINT_REPO = "facebook/sam3"    # gated, 약 3.4 GB
HF_CHECKPOINT_KEYWORDS = ("auto", "huggingface")

# v0.21.2 계열 PAG 빌드(인자 57개)에는 CFG 1 가드도, 확장 배치 OOM 폴백도 없다.
PAG_OLD_BUILD_ARGC = 57

# 요청에 hr_cfg 가 없을 때 Forge 하이레스 패스의 CFG — processing.py:1226 ``hr_cfg: float = 1.0`` 이고 API 모델은
# __init__ 기본값을 그대로 쓴다(modules/api/models.py:63-69). 앱은 하이레스 CFG 가 0(끔)이면 hr_cfg 를 안 보낸다.
FORGE_DEFAULT_HR_CFG = 1.0

RESULT_NOTICE_TTL_S = 30.0          # 같은 결과 알림(배치·자동화의 같은 실패)은 이 동안 한 번만
PRE_GENERATION_NOTICE_TTL_S = 600.0  # 같은 생성 전 경고는 10분에 한 번 (자동화가 장마다 띄우지 않게)
# 알림 종류별 최소 억제 시간 — 추측 변환 기록은 같은 LoRA 묶음이면 생성마다 남으므로(확장 gr.Info 는 로드 때 한 번)
# 생성 전 경고처럼 드물게 띄운다. 게이트·전달 알림도 설정이 그대로면 요청마다 같으므로(배치 100장이 30초마다
# 다시 띄우지 않게) 생성 전 경고와 같게 둔다.
NOTICE_MIN_TTL_S = {
    CODE_LORA_SPARSE_GUESS: PRE_GENERATION_NOTICE_TTL_S,
    CODE_BLOCK_NOT_SENT: PRE_GENERATION_NOTICE_TTL_S,
    CODE_BLOCK_DEFERRED: PRE_GENERATION_NOTICE_TTL_S,
    CODE_PROPAGATION_DROPPED: PRE_GENERATION_NOTICE_TTL_S,
    CODE_PROPAGATION_RETRIED: PRE_GENERATION_NOTICE_TTL_S,
    # DoRA 알림도 설정이 그대로면 요청마다 같다(배치 100장이 30초마다 다시 띄우지 않게 — critic B14)
    CODE_DORA_NOT_APPLIED: PRE_GENERATION_NOTICE_TTL_S,
    CODE_DORA_COMFY_STOCK: PRE_GENERATION_NOTICE_TTL_S,
    CODE_DORA_CHOICE: PRE_GENERATION_NOTICE_TTL_S,
    # Anima 3.8B Comfy 안내도 모듈 목록이 그대로면 요청마다 같다(critic B14)
    CODE_ANIMA38_COMFY_V1: PRE_GENERATION_NOTICE_TTL_S,
    # 재시도·옵션 알림도 스냅샷·설정이 그대로면 요청마다 같다(배치 100장이 30초마다 다시 띄우지 않게 — critic B14)
    CODE_APP_BLOCK_RETRIED: PRE_GENERATION_NOTICE_TTL_S,
    CODE_BLOCK_RETRIED: PRE_GENERATION_NOTICE_TTL_S,
    CODE_FORGE_OPTION_MISSING: PRE_GENERATION_NOTICE_TTL_S,
    CODE_FORGE_OPTION_UNVERIFIED: PRE_GENERATION_NOTICE_TTL_S,
    CODE_FORGE_OPTION_REJECTED: PRE_GENERATION_NOTICE_TTL_S,
    CODE_FORGE_OPTION_FROZEN: PRE_GENERATION_NOTICE_TTL_S,
    # 원본 기준 SAM3 알림도 확장·설정이 그대로면 장마다 같다(배치 SAM3 100장이 30초마다 다시 띄우지 않게)
    CODE_SAM3_SOURCE_UNSUPPORTED: PRE_GENERATION_NOTICE_TTL_S,
    CODE_SAM3_SOURCE_FALLBACK: PRE_GENERATION_NOTICE_TTL_S,
}
# 띄울 때 GUI 가 기능 스냅샷을 다시 받아야 하는 알림(연결 때 받은 스냅샷이 틀렸다는 뜻) —
# ui/sam_extra_notices_ui.show_notices 가 _refresh_sam_extra_capabilities(force=True) 를 부른다.
REFRESH_CAPABILITIES_CODES = frozenset({CODE_PROPAGATION_RETRIED, CODE_APP_BLOCK_RETRIED, CODE_BLOCK_RETRIED,
                                        CODE_FORGE_OPTION_REJECTED})

# ── alwayson 제목 → 사람이 읽는 이름 ─────────────────────────────────────────
_TITLE_DORA = dora_infer_mode.SCRIPT_NAME
_TITLE_VAE2X = "Anima VAE 2x (spacepxl decoder)"
SAM_EXTRA_FEATURES = {
    sam3_args.SCRIPT_SAM3.lower(): ("sam3", "SAM3 Mask"),
    anima_guidance.SCRIPT_PERTURBATION.lower(): ("anima_guidance", "Anima 가이던스(PAG·SEG·SLG·APG·SMC…)"),
    anima_guidance.SCRIPT_SKIMMED_CFG.lower(): ("skimmed_cfg", "Skimmed CFG"),
    anima_guidance.SCRIPT_DETAIL_DAEMON.lower(): ("detail_daemon", "Detail Daemon"),
    anima38.SCRIPT_NAME.lower(): ("anima38", "Anima 3.8B"),
    _TITLE_DORA.lower(): ("dora", "DoRA 추론 방식"),
    _TITLE_VAE2X.lower(): ("vae2x", "VAE 2x"),
}
# sam-extra 밖 확장: 소문자 제목 → (기능, 확장 이름)
OTHER_EXTENSIONS = {
    "adetailer": ("adetailer", "ADetailer 확장(Bing-su/adetailer)"),
    "negpip": ("negpip", "NegPiP 확장(sd-webui-negpip)"),
    "controlnet": ("controlnet", "ControlNet(Forge 내장 sd_forge_controlnet)"),
}

# SAM3 설정 키 → 앱 화면 라벨 (Sam3MaskCard.vue · Sam3ControlNetPanel.vue)
SAM3_SETTING_LABELS = {
    "sam3_mode": "SAM3 Mode",
    "sam3_mask_mode": "마스크 처리",
    "sam3_prompt": "SAM3 Detect Prompt",
    "sam3_exclude_prompt": "SAM3 Exclude Prompt",
    "sam3_threshold": "SAM3 Threshold",
    "sam3_checkpoint": "SAM3 Checkpoint",
    "sam3_device": "SAM3 Device",
    "sam3_inpainting_fill": "Masked content",
    "sam3_inpaint_width": "Inpaint Width",
    "sam3_inpaint_height": "Inpaint Height",
    "sam3_steps": "단계",
    "sam3_cfg_scale": "CFG 스케일",
    "sam3_sampler": "샘플러",
    "sam3_scheduler": "Scheduler",
    "sam3_seed": "Seed",
    "sam3_cn_model": "ControlNet Model",
    "sam3_cn_module": "ControlNet Module(전처리기)",
    "sam3_cn_control_mode": "ControlNet Control mode",
    "sam3_cn_resize_mode": "ControlNet Resize mode",
}
# KeyError 값과 맞대 볼 SAM3 설정 (먼저 맞는 것)
_KEYED_SETTINGS = ("sam3_cn_module", "sam3_cn_model", "sam3_sampler", "sam3_scheduler", "sam3_checkpoint")


@dataclass(frozen=True)
class Notice:
    """사용자에게 보일 알림 한 건.

    ``detail`` 은 확장이 남긴 원문(로그·중복 판정용 — 장마다 숫자만 다른 원문은 ``key`` 가 알림을 억제하지 못하므로
    종류로 줄여 둔다: CODE_SAM3_SOURCE_FALLBACK), ``hint`` 는 확인할 설정 한 문장(단독 경로가 자기 문구를
    만들 때 쓴다 — ``standalone_failure_text``).
    """

    code: str
    level: str
    message: str
    feature: str = ""
    detail: str = ""
    hint: str = ""

    @property
    def key(self) -> str:
        return f"{self.code}|{self.feature}|{self.detail or self.message}"

    def to_dict(self) -> dict:
        return {"code": self.code, "level": self.level, "message": self.message,
                "feature": self.feature, "detail": self.detail, "hint": self.hint}

    @classmethod
    def from_dict(cls, data: Any) -> Optional["Notice"]:
        if not isinstance(data, Mapping):
            return None
        message = str(data.get("message") or "").strip()
        if not message:
            return None
        level = str(data.get("level") or LEVEL_WARNING)
        return cls(code=str(data.get("code") or ""), level=level if level in LEVELS else LEVEL_WARNING,
                   message=message, feature=str(data.get("feature") or ""),
                   detail=str(data.get("detail") or ""), hint=str(data.get("hint") or ""))


def notices_from_info(info: Any) -> list[Notice]:
    """generation info 의 ``INFO_KEY`` → 알림 목록 (없거나 모양이 틀리면 빈 목록)."""
    raw = info.get(INFO_KEY) if isinstance(info, Mapping) else None
    if not isinstance(raw, (list, tuple)):
        return []
    return [n for n in (Notice.from_dict(item) for item in raw) if n is not None]


def notices_to_dicts(notices: Iterable[Notice]) -> list[dict]:
    return [n.to_dict() for n in notices]


# ── 단독 경로 결과 (base64 str + 알림) ────────────────────────────────────────
class NoticedImage(str):
    """base64 이미지 문자열에 Forge info 와 알림을 붙인 str.

    ``WebUIBackend._run_img2img_postprocess`` 가 돌려준다. str 이라 ``base64.b64decode(result)`` 같은 기존
    호출자(단독 SAM3·Refine·ADetailer 워커)는 그대로 동작하고, 알림을 보려면 ``notices_of(result)`` 를 부른다.
    ComfyUI 백엔드는 평범한 str 을 돌려주므로 알림이 없다.
    """

    def __new__(cls, value: str, info: Optional[Mapping] = None, notices: Iterable[Notice] = ()):
        obj = super().__new__(cls, value)
        obj.info = dict(info or {})
        obj.notices = tuple(notices)
        return obj


def notices_of(value: Any) -> tuple[Notice, ...]:
    notices = getattr(value, "notices", ())
    return tuple(n for n in notices if isinstance(n, Notice)) if isinstance(notices, (list, tuple)) else ()


def standalone_sam3_failure(value: Any) -> Optional[Notice]:
    """단독 SAM3/Refine 결과가 'SAM3 가 돌지 않은 원본'인가 — 그렇다면 그 알림 (결과는 저장하지 않는다).

    단독 경로는 부모 img2img 를 denoise 0 으로 통과시키므로 SAM3 가 실패하면 결과는 SAM3 가 손대지 않은 이미지다 —
    확장이 원본 기준 요청('sam3_source_image', ``KEY_SAM3_SOURCE``)을 따랐으면 원본 그대로, 모르는 예전 확장이면
    VAE 왕복으로 조금 바뀐 원본. 어느 쪽이든 그 파일을 '_sam3'·'_refine' 이름으로 저장하면 적용된 것처럼 보이므로
    워커가 실패로 알린다.
    """
    return next((n for n in notices_of(value) if n.code in SAM3_FAILURE_CODES), None)


def standalone_failure_text(notice: Notice, *, saved_without_sam3: bool = False) -> str:
    """단독 SAM3·배치·Refine 의 실패 문구.

    기본은 '결과가 원본과 같아 저장하지 않았다'. ``saved_without_sam3`` 는 앞 단계(업스케일·ADetailer)가 이미
    적용돼 그 결과를 SAM3 없이 저장했을 때다(``BatchUpscaleWorker`` 'both').
    """
    outcome = ("SAM3 없이 앞 단계 결과만 저장했습니다" if saved_without_sam3
               else "결과가 원본과 같아 저장하지 않았습니다")
    if notice.code == CODE_SAM3_ERROR:
        why = notice.hint or "Forge 콘솔의 '[-] SAM3: failed' 줄을 확인하세요"
        return f"SAM3 실패 — {why}. {outcome}. (원인: {_short(notice.detail, 120)})"
    return (f"SAM3 가 적용되지 않았습니다(결과에 'SAM3 Enable' 기록 없음) — {outcome}. "
            "Forge 콘솔의 '[-] SAM3' 줄을 확인하세요.")


def sam3_failure_repeats(notice: Optional[Notice]) -> bool:
    """같은 설정의 다음 이미지도 같은 이유로 실패할 SAM3 실패인가 — 배치 SAM3 가 남은 이미지를 건너뛸 때 쓴다.

    설정 값·설치 문제(검증 실패, 없는 CN 모듈·샘플러 이름, 체크포인트·패키지·HF 권한, 없는 장치)와 'SAM3 Enable'
    기록이 아예 없는 경우(구버전 확장·설정 미전달)만 True. VRAM 부족이나 원인을 모르는 오류는 이미지마다 다를 수
    있어 False 다.
    """
    if notice is None:
        return False
    if notice.code == CODE_SAM3_NOT_APPLIED:
        return True
    return notice.code == CODE_SAM3_ERROR and _sam3_error_kind(notice.detail)[0] in _REPEATING_SAM3_ERRORS


# ── 값 도우미 ──────────────────────────────────────────────────────────────────
def _truthy(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    return str(value or "").strip().lower() in {"1", "true", "yes", "on"}


def _number(value: Any) -> Optional[float]:
    if isinstance(value, bool):
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def _is_one(value: Optional[float]) -> bool:
    return value is not None and math.isclose(value, 1.0, rel_tol=0.0, abs_tol=1e-6)


def _short(text: Any, limit: int = 160) -> str:
    flat = " ".join(str(text or "").split())
    return flat if len(flat) <= limit else flat[: limit - 1] + "…"


def _script_block(payload: Any, title: str) -> Optional[Mapping]:
    """alwayson_scripts 에서 제목으로 블록을 찾는다 — Forge 처럼 대소문자를 가리지 않는다."""
    scripts = payload.get("alwayson_scripts") if isinstance(payload, Mapping) else None
    if not isinstance(scripts, Mapping):
        return None
    wanted = title.strip().lower()
    for name, block in scripts.items():
        if str(name).strip().lower() == wanted:
            return block if isinstance(block, Mapping) else {}
    return None


def _block_args(block: Optional[Mapping]) -> list:
    args = block.get("args") if isinstance(block, Mapping) else None
    return list(args) if isinstance(args, (list, tuple)) else []


# ── 요청에서 켠 기능 ───────────────────────────────────────────────────────────
_PAG_INDEX = {key: index for index, (key, *_rest) in enumerate(anima_guidance.PERTURBATION_SPEC)}
_CFG_MODE_MAP = {"preserve incoming": "preserve", "preserve": "preserve", "apg": "apg", "cwm": "cwm",
                 "smc": "smc", "smc + cwm": "smc+cwm", "smc+cwm": "smc+cwm"}


@dataclass(frozen=True)
class RequestedFeatures:
    """요청 payload 에서 읽은 sam-extra 기능 (확장이 인자를 해석하는 규칙을 그대로 따른다)."""

    scripts: tuple = ()                       # alwayson 제목(보낸 그대로)
    sam3: bool = False
    sam3_state: Mapping = None                # type: ignore[assignment]
    pag: bool = False                         # PAG/SEG/SLG perturbation 을 켰나
    pag_parts: tuple = ()                     # ('PAG',) ('SEG', 'SLG') …
    cfg_bases: tuple = ()                     # ('SMC', 'APG', 'CWM') 중 실제로 거는 것(CWM 은 alpha ≠ 0)
    anima38: Optional[anima38.Anima38Settings] = None   # 블록을 보냈을 때만
    dora: Optional[dora_infer_mode.DoraSettings] = None   # 블록을 보냈을 때만(확장이 읽는 규칙 — parse_script_block)
    cfg_scale: Optional[float] = None
    hires: bool = False
    hires_cfg: Optional[float] = None         # 요청에 hr_cfg 가 있을 때만
    img2img: bool = False

    @property
    def effective_hires_cfg(self) -> Optional[float]:
        """하이레스 패스가 실제로 쓰는 CFG — hr_cfg 를 안 보내면 Forge 기본값(``FORGE_DEFAULT_HR_CFG``)."""
        if not self.hires:
            return None
        return self.hires_cfg if self.hires_cfg is not None else FORGE_DEFAULT_HR_CFG

    @property
    def sam3_checkpoint(self) -> str:
        state = self.sam3_state or {}
        return str(state.get("sam3_checkpoint") or "").strip()

    @property
    def anima38_expected(self) -> bool:
        return bool(self.anima38 is not None and self.anima38.enabled and not self.anima38.bypass)

    @property
    def sam3_init_source(self) -> bool:
        """원본 기준 SAM3 요청(단독 SAM3·Refine) — 확장 process() 와 같은 규칙(공백·대소문자 무시)."""
        value = (self.sam3_state or {}).get(sam3_args.SOURCE_STATE_KEY)
        return str(value or "").strip().lower() == sam3_args.SOURCE_INIT


def _sam3_request(block: Optional[Mapping]) -> tuple[bool, dict]:
    """``!sam3.py`` process() 와 같은 규칙: (켜짐, state)."""
    args = _block_args(block)
    enabled, state = False, {}
    if args:
        first = args[0]
        if isinstance(first, bool):
            enabled = first
            if len(args) > 1 and isinstance(args[1], Mapping):
                state = dict(args[1])
        elif isinstance(first, Mapping):
            state = dict(first)
            enabled = bool(state.get("sam3_enable", state.get("enabled", False)))
    if not state:
        state = next((dict(arg) for arg in args if isinstance(arg, Mapping)), {})
        enabled = enabled or bool(state.get("sam3_enable", state.get("enabled", False)))
    return enabled, state


def _pag_request(args: Sequence, live_argc: Optional[int]) -> tuple[bool, tuple, tuple]:
    """``anima_safe_pag.py`` process_before_every_sampling 과 같은 규칙: (perturbation, 부분, CFG base)."""
    if not args or isinstance(args[0], Mapping):
        return False, (), ()
    if live_argc is not None and live_argc > 0:
        args = list(args)[:live_argc]     # Forge 는 확장 인자 수만큼만 넘긴다 (뒤는 조용히 잘림)

    def arg(key: str, default: Any = None) -> Any:
        index = _PAG_INDEX[key]
        return args[index] if index < len(args) else default

    parts = []
    if bool(arg("guid_enabled", False)):
        method = str(arg("guid_attn_method", "PAG") or "").strip().upper()
        if method in ("PAG", "SEG") and (_number(arg("guid_scale", 0)) or 0) > 0:
            parts.append(method)
        if _truthy(arg("guid_slg_on", False)) and (_number(arg("guid_slg_scale", 0)) or 0) > 0:
            parts.append("SLG")

    apg_enabled = bool(arg("guid_apg_enabled", False))
    cfg_mode = _CFG_MODE_MAP.get(str(arg("guid_cfg_mode", "Preserve incoming") or "").strip().lower(), "preserve")
    stack = _truthy(arg("guid_experimental_stack", False))
    if cfg_mode == "preserve" and apg_enabled:
        cfg_mode = "apg"
    legacy_smc = _truthy(arg("guid_smc_enabled", False))
    cwm_enabled = _truthy(arg("guid_cwm_enabled", False))
    smc_preset = str(arg("guid_smc_preset", "Off") or "Off").strip()
    master_index = _PAG_INDEX["guid_smc_master_enabled"]
    if len(args) > master_index:      # 확장: len(args) > 57 이면 마스터 토글이 명시적
        smc = legacy_smc or stack or cfg_mode in ("smc", "smc+cwm") or _truthy(arg("guid_smc_master_enabled"))
    else:
        smc = legacy_smc or stack or cfg_mode in ("smc", "smc+cwm") or smc_preset.lower() != "off"
    bases = []
    if smc:
        bases.append("SMC")
    if apg_enabled or cfg_mode == "apg" or stack:
        bases.append("APG")
    # CWM 은 alpha 가 하나라도 0 이 아닐 때만 건다 — 원본 ``cwm_alpha_active`` (origin: namemechan/ComfyUI-DCW@66aaf9dd:
    # dcw_node.py:817-821 — 뜻만), 확장 DCW-F ``_apply_cfg_base`` 도 같다. alpha 0 의 CWM 은 대역 가중이 모두 w 라
    # 어느 빌드에서든 평범한 CFG 와 같다(건너뛰어도 달라질 것이 없다).
    cwm_alpha = (_number(arg("guid_cwm_alpha_low", 0.0)) or 0.0, _number(arg("guid_cwm_alpha_high", 0.0)) or 0.0)
    if (cwm_enabled or stack or cfg_mode in ("cwm", "smc+cwm")) and any(cwm_alpha):
        bases.append("CWM")
    return bool(parts), tuple(parts), tuple(bases)


def requested_features(payload: Any, *, live_pag_argc: Optional[int] = None) -> RequestedFeatures:
    """요청 payload → ``RequestedFeatures``. ``live_pag_argc`` 는 스냅샷의 PAG 인자 수(없으면 None)."""
    if not isinstance(payload, Mapping):
        return RequestedFeatures(sam3_state={})
    scripts = payload.get("alwayson_scripts")
    titles = tuple(str(name) for name in scripts) if isinstance(scripts, Mapping) else ()
    sam3_on, sam3_state = _sam3_request(_script_block(payload, sam3_args.SCRIPT_SAM3))
    pag, parts, bases = _pag_request(
        _block_args(_script_block(payload, anima_guidance.SCRIPT_PERTURBATION)), live_pag_argc)
    block38 = _script_block(payload, anima38.SCRIPT_NAME)
    hires = _truthy(payload.get("enable_hr"))
    return RequestedFeatures(
        scripts=titles, sam3=sam3_on, sam3_state=sam3_state, pag=pag, pag_parts=parts, cfg_bases=bases,
        anima38=anima38.parse_script_block(block38) if block38 is not None else None,
        dora=dora_infer_mode.parse_script_block(_script_block(payload, dora_infer_mode.SCRIPT_NAME)),
        cfg_scale=_number(payload.get("cfg_scale")),
        hires=hires, hires_cfg=_number(payload.get("hr_cfg")) if hires and "hr_cfg" in payload else None,
        # 키가 있으면 img2img 요청이다 — 재인코딩하는 I2I 는 워커가 채울 때까지 init_images=[] 로 둔다
        # (core/i2i_payload.py). T2I 페이로드에는 이 키가 없다.
        img2img="init_images" in payload,
    )


# ── 기능 스냅샷 (버전 판단) ────────────────────────────────────────────────────
_ASSUMED_LOGGED = False


def _capabilities_known(capabilities: Any) -> bool:
    return bool(getattr(capabilities, "known", False))


def _pag_argc(capabilities: Any) -> Optional[int]:
    if not _capabilities_known(capabilities):
        return None
    value = getattr(capabilities, "anima_guidance_argc", None)
    return value if isinstance(value, int) and value > 0 else None


def _pag_old_build(capabilities: Any, *, reason: str) -> bool:
    """PAG 가 v0.21.2 계열(57개)인가. 스냅샷이 없으면 지금 설치된 확장(v0.30) 기준 — 경고 로그 한 번."""
    global _ASSUMED_LOGGED
    argc = _pag_argc(capabilities)
    if argc is None:
        if not _ASSUMED_LOGGED:
            _ASSUMED_LOGGED = True
            logger.warning("[sam-extra] 기능 스냅샷이 없어 %s 판단을 설치된 확장(v0.30) 기준으로 합니다 "
                           "— WebUI 연결 뒤 확인이 끝나면 실제 버전을 따릅니다", reason)
        return False
    return argc <= PAG_OLD_BUILD_ARGC


# ── infotext → 장마다 파라미터 ─────────────────────────────────────────────────
def _parse_infotext(text: str) -> dict:
    from core.infotext import parse_parameters_text, split_infotext

    _positive, _negative, tail = split_infotext(text)
    return parse_parameters_text(tail) if tail else {}


def image_parameters(info: Any) -> list[dict]:
    """Forge info(dict 또는 JSON 문자열) → 결과 이미지마다 파라미터 dict.

    ``infotexts`` 가 있으면 장마다(격자는 ``index_of_first_image`` 로 뺀다), 없으면 ``extra_generation_params``
    한 벌. 읽을 것이 없으면(ComfyUI·raw_info) 빈 목록 — 호출자는 판단하지 않는다.
    """
    if isinstance(info, str):
        try:
            info = json.loads(info)
        except ValueError:
            return []
    if not isinstance(info, Mapping):
        return []
    texts = info.get("infotexts")
    if isinstance(texts, list) and texts:
        first = info.get("index_of_first_image")
        if isinstance(first, int) and not isinstance(first, bool) and 0 < first < len(texts):
            texts = texts[first:]
        parsed = [_parse_infotext(text) for text in texts if isinstance(text, str) and text.strip()]
        if parsed:
            return parsed
    extra = info.get("extra_generation_params")
    if isinstance(extra, Mapping):
        return [{str(k): v for k, v in extra.items()}]
    return []


def _anima38_status(params: Mapping) -> Optional[str]:
    for key in (KEY_ANIMA38_STATUS, *ANIMA38_LEGACY_STATUS_KEYS):
        if key in params and params[key] is not None:
            return str(params[key]).strip()
    return None


KEY_MODEL = "Model"   # Forge infotext 모델 이름(확장자·해시 없음)


def _result_model_kind(images: Sequence[Mapping]) -> Optional[str]:
    """결과 infotext 'Model' → 모델 종류(core/anima_model_kind — 연결 때 헤더로 확인한 목록, 없으면 이름). 없으면 None."""
    for params in images:
        name = params.get(KEY_MODEL)
        if name not in (None, ""):
            return anima_model_kind.kind_for_model_name(name)
    return None


# ── 원인 → 확인할 설정 ─────────────────────────────────────────────────────────
_SAM3_FIELD_RE = re.compile(r"\bsam3_[a-z0-9_]+\b")
_KEYERROR_RE = re.compile(r"KeyError:\s*['\"]?([^'\"]+?)['\"]?\s*$")


def _setting_label(key: str) -> str:
    label = SAM3_SETTING_LABELS.get(key)
    return f"{label}({key})" if label else key


# 'SAM3 Error' 원문의 종류 — 힌트 문구와 '다음 이미지도 같은 이유로 실패하는가'(배치 중단)를 함께 정한다.
_ERR_VALIDATION, _ERR_OOM, _ERR_HF, _ERR_CHECKPOINT, _ERR_PACKAGE = "validation", "oom", "hf", "checkpoint", "package"
_ERR_KEY, _ERR_DEVICE, _ERR_CONTROLNET, _ERR_OTHER = "key", "device", "controlnet", "other"
# 설정 값·설치 문제 — 같은 설정의 배치에서는 모든 이미지가 같게 실패한다
_REPEATING_SAM3_ERRORS = frozenset({_ERR_VALIDATION, _ERR_HF, _ERR_CHECKPOINT, _ERR_PACKAGE, _ERR_KEY, _ERR_DEVICE})

# HF 다운로드 실패만 — 숫자('401'·'403')나 'huggingface' 를 맨몸으로 찾으면 텐서 모양 [1, 4032, 256] 이나
# HF 캐시 경로(.cache/huggingface/hub)가 든 체크포인트 로드 오류까지 '승인·로그인 필요'로 오해한다.
_HF_ERROR_WORDS = ("gatedrepo", "gated repo", "repositorynotfound", "localentrynotfound", "entrynotfound",
                   "hfhubhttperror", "huggingface.co", "huggingface_hub", "hf_hub", "hugging face")
_HTTP_AUTH_RE = re.compile(r"\b40[13]\s+(?:client error|forbidden|unauthorized)\b")
# 장치 설정 문제만 — 'Expected all tensors to be on the same device … cuda:0 and cpu' 같은 내부 오류는 아니다.
# 확장 검증기는 모르는 장치 문자열을 조용히 auto 로 바꾸므로(sam3ext/args.py _normalise_device) 남는 것은 torch 오류다.
_DEVICE_ERROR_RE = re.compile(r"invalid device|device ordinal|not compiled with cuda|no cuda gpus|"
                              r"no nvidia driver|cuda driver version is insufficient")


def _sam3_error_kind(reason: Any) -> tuple[str, str]:
    """'SAM3 Error' 원문 → (종류, KeyError 값). KeyError 가 아니면 값은 ''."""
    text = str(reason or "")
    lower = text.lower()
    if "validation error" in lower:
        return _ERR_VALIDATION, ""
    if "out of memory" in lower or "outofmemory" in lower:
        return _ERR_OOM, ""
    if "checkpoint not found" not in lower and (
            any(word in lower for word in _HF_ERROR_WORDS) or _HTTP_AUTH_RE.search(lower)):
        return _ERR_HF, ""
    if "checkpoint not found" in lower or ("filenotfounderror" in lower and "sam3" in lower):
        return _ERR_CHECKPOINT, ""
    if "package is not installed" in lower or "no module named" in lower:
        return _ERR_PACKAGE, ""
    match = _KEYERROR_RE.search(text)
    if match:
        return _ERR_KEY, match.group(1).strip()
    if _DEVICE_ERROR_RE.search(lower):
        return _ERR_DEVICE, ""
    if "controlnet" in lower:
        return _ERR_CONTROLNET, ""
    return _ERR_OTHER, ""


def sam3_error_hint(reason: str, state: Optional[Mapping] = None) -> str:
    """'SAM3 Error' 원문 → 사용자가 확인할 설정 (한 문장). 모르는 원인은 콘솔을 보라고만 한다(추측하지 않는다)."""
    text = str(reason or "")
    state = state or {}
    kind, value = _sam3_error_kind(text)
    if kind == _ERR_VALIDATION:
        fields = [f for f in dict.fromkeys(_SAM3_FIELD_RE.findall(text))]
        labels = ", ".join(_setting_label(f) for f in fields) or "SAM3 카드의 값"
        return f"SAM3 설정 값이 확장 검증을 통과하지 못했습니다 — 확인할 설정: {labels}"
    if kind == _ERR_OOM:
        return ("VRAM 부족 — 'Unload SAM3 from VRAM after detection' 을 켜고 인페인트 해상도를 낮추거나 "
                "SAM3 Device 를 cpu 로 바꾸세요")
    if kind == _ERR_HF:
        return (f"SAM3 체크포인트 자동 다운로드 실패 — Hugging Face {HF_CHECKPOINT_REPO} 는 승인·로그인(과 네트워크)이 "
                "필요합니다. sam3.pt 를 Forge models/sam3 에 직접 두고 SAM3 Checkpoint 를 다시 고르세요")
    if kind == _ERR_CHECKPOINT:
        return "SAM3 Checkpoint — 파일이 Forge 에 없습니다. models/sam3 에 두거나 SAM3 카드에서 다시 고르세요"
    if kind == _ERR_PACKAGE:
        return "Forge 에 SAM3 패키지가 없습니다 — 확장의 install.py 가 돌도록 Forge 를 다시 시작하세요"
    if kind == _ERR_KEY:
        for key in _KEYED_SETTINGS:
            if str(state.get(key, "")).strip().lower() == value.lower() and value:
                extra = " — Forge 목록에 없는 이름입니다(대소문자 구분, 예: 'None')" if key.startswith("sam3_cn_") else ""
                return f"{_setting_label(key)} 값 '{value}'{extra}"
        return f"값 '{value}' 을(를) 쓰는 SAM3 설정(ControlNet Module·Model, 샘플러, Scheduler)"
    if kind == _ERR_DEVICE:
        return "SAM3 Device — 이 PC 에서 쓸 수 없는 장치입니다 (auto·cuda·cpu 중에서 고르세요)"
    if kind == _ERR_CONTROLNET:
        return "SAM3 ControlNet 설정(Model·Module) — Forge 의 ControlNet 목록에 있는 이름인지 확인하세요"
    return "Forge 콘솔의 '[-] SAM3: failed' 줄에서 원인을 확인하세요"


def anima38_off_hint(status: str, *, kind: Optional[str] = None) -> str:
    """'Anima38: off: …' → 확인할 것 (한 문장).

    확장의 FileNotFoundError 문구 세 가지(sam3ext/anima38/files.py tokenizer_dir, runtime.py _qwen35_path·
    _load_adapter)를 가른다. 토크나이저 문구에도 'qwen35'('assets/qwen35_tokenizer')가 들어 있고 어댑터 이름에도
    들어갈 수 있으므로 토크나이저 → 어댑터 → 텍스트 인코더 순서로 본다.

    ``kind``(결과 모델 종류, core/anima_model_kind): 'install failed (RuntimeError)' 는 비 Anima 모델에 v1 을 켰을 때
    확장이 남기는 흔적이다(runtime._require_anima — 'requires a loaded Anima checkpoint'). 종류가 other·unknown 이면
    그 원인을 먼저 말한다(P9). None(모름·옛 호출)이면 예전 문구.
    """
    lower = str(status or "").lower()
    if "install failed" in lower:
        cause = re.search(r"\(([^)]+)\)", status or "")
        why = f"({cause.group(1)}) " if cause else ""
        if "(runtimeerror)" in lower and kind == anima_model_kind.KIND_OTHER:
            return (f"선택한 모델이 Anima 가 아닙니다 {why}— 'Anima 3.8B' 카드의 'v1 어댑터 켜기'를 끄세요 "
                    "(확장은 Anima 체크포인트에만 Qwen3.5 커넥터를 설치합니다)")
        if "(runtimeerror)" in lower and kind == anima_model_kind.KIND_UNKNOWN:
            return (f"선택한 모델이 Anima 가 아니면 'Anima 3.8B' 카드의 'v1 어댑터 켜기'를 끄세요 {why}— Anima 라면 "
                    "VRAM 이 부족했거나 파일이 손상됐을 수 있습니다. Forge 콘솔의 '[Anima38]' 줄을 확인하세요")
        return (f"Qwen3.5 커넥터 설치 실패 {why}— VRAM 이 부족했거나 파일이 손상됐을 수 있습니다. "
                "Forge 콘솔의 '[Anima38]' 줄을 확인하세요")
    if "tokenizer" in lower:
        return "확장의 assets/qwen35_tokenizer 가 없습니다 — sam-extra 확장을 다시 설치하세요"
    if "adapter" in lower:
        return "v1 어댑터 파일을 찾지 못했습니다 — Forge 목록을 새로고침한 뒤 어댑터를 다시 고르세요"
    if "qwen35" in lower or "text_encoder" in lower:
        return ("models/text_encoder 에 qwen35_4b.safetensors 가 없습니다 (앱이 관리하는 Forge 는 "
                "--text-encoder-dirs 로 넘긴 폴더를 봅니다)")
    return "Forge 콘솔의 '[Anima38]' 줄에서 원인을 확인하세요"


# ── 생성 뒤 알림 ───────────────────────────────────────────────────────────────
PROPAGATED_SUFFIX = " (메인 생성 설정에서 전달됨)"

_SOURCE_OUTPUT_RE = re.compile(rf"^{re.escape(SAM3_SOURCE_OUTPUT_NOTE)}\s*\((.*)\)\s*$", re.IGNORECASE | re.DOTALL)
# 'output (<이유>)' 의 이유 → 확인할 것 (사실만 — 그 밖의 이유는 원문만 보인다)
_SOURCE_FALLBACK_HINTS = (
    ("size ", "부모 패스 결과의 크기가 원본과 달랐습니다"),
    ("face restoration", "Forge 의 얼굴 복원(Restore faces) 설정이 켜져 있습니다"),
    ("init image unreadable", "Forge 가 init 이미지를 읽지 못했습니다 — Forge 콘솔의 '[-] SAM3' 줄을 확인하세요"),
)


def _source_reason_kind(reason: str) -> str:
    """폴백 이유의 종류(중복 판정용) — 숫자만 다른 이유('size 1199x1601 != 1192x1600')는 한 종류('size #x# != #x#')다.

    ``Notice.key`` 가 detail 을 쓰므로 원문을 detail 로 두면 크기가 다른 이미지마다 억제 키가 새로 생겨,
    크롭·스크린샷 폴더의 단독 SAM3 배치가 NOTICE_MIN_TTL_S 를 무시하고 장마다 경고를 띄운다. 원문은 문구에 남는다.
    """
    return re.sub(r"\d+", "#", " ".join(reason.lower().split()))


def _sam3_source_notices(images: Sequence[Mapping]) -> list[Notice]:
    """원본 기준 SAM3 요청(단독 SAM3·Refine)의 결과 — 확장이 'SAM3 Source' 로 어느 이미지를 썼는지 알린다.

    denoise 0 이어도 부모 img2img 는 VAE 왕복으로 픽셀을 조금씩 바꾼다. 확장이 init 이미지를 쓰면('init image')
    마스크 밖이 원본 그대로다. 키가 아예 없으면 이 요청을 모르는 예전 sam-extra 다(정보 — 결과는 예전과 같다).
    'output (<이유>)' 면 요청했지만 조건(img2img·마스크 없음·denoise 0·얼굴 복원 없음·같은 크기 등)이 안 맞아 출력으로
    돌았다(경고). 경고는 이유의 종류마다 하나다(``_source_reason_kind``). 호출자는 SAM3 가 실제로 적용된 결과('SAM3 Error' 없음, 'SAM3 Enable' 있음)에서만 부른다 —
    실패는 CODE_SAM3_ERROR·CODE_SAM3_NOT_APPLIED 가 알린다.
    """
    values = [str(p.get(KEY_SAM3_SOURCE)).strip() for p in images if p.get(KEY_SAM3_SOURCE) not in (None, "")]
    if not values:
        return [Notice(
            CODE_SAM3_SOURCE_UNSUPPORTED, LEVEL_INFO,
            "연결된 sam-extra 는 원본 기준 SAM3 를 지원하지 않아, SAM3 가 부모 img2img 패스의 출력(VAE 왕복)을 "
            "썼습니다 — 마스크 밖도 조금 바뀔 수 있습니다. sam-extra 를 업데이트한 뒤 Forge Settings → Reload UI "
            "(또는 Forge 재시작)를 하세요.",
            feature="sam3", detail=KEY_SAM3_SOURCE)]
    out: list[Notice] = []
    kinds: set[str] = set()
    for value in dict.fromkeys(values):
        if value.lower() == SAM3_SOURCE_INIT_NOTE:
            continue
        match = _SOURCE_OUTPUT_RE.match(value)
        reason = match.group(1).strip() if match else value
        kind = _source_reason_kind(reason)
        if kind in kinds:
            continue            # 한 결과에 크기만 다른 폴백이 여러 장 — 첫 원문 하나로
        kinds.add(kind)
        hint = next((text for prefix, text in _SOURCE_FALLBACK_HINTS if reason.lower().startswith(prefix)), "")
        out.append(Notice(
            CODE_SAM3_SOURCE_FALLBACK, LEVEL_WARNING,
            "원본 기준 SAM3 를 요청했지만 확장이 부모 img2img 패스의 출력(VAE 왕복)으로 돌렸습니다 — 마스크 밖도 조금 "
            f"바뀔 수 있습니다{' — ' + hint if hint else ''}. (이유: {_short(reason, 120)})",
            feature="sam3", detail=f"{SAM3_SOURCE_OUTPUT_NOTE} ({kind})", hint=hint))
    return out


def _dora_result_notices(request: RequestedFeatures, images: Sequence[Mapping]) -> list[Notice]:
    """보낸 DoRA 방식이 결과 infotext 에 없으면 경고 — 확장은 순정이 아닌 방식을 켠 요청마다(LoRA 가 없어도)
    'DoRA mode' 를, 그대로 복제가 아닌 정책이면 'DoRA inserted' 를 남긴다(scripts/dora_infer_mode.py _record_state).
    빠졌다면 훅이 없어 순정으로 떨어진 것이다(process() 의 훅 폴백 :414-419).

    앱 기본값 블록(카드를 건드리지 않은 사용자에게 앱이 스스로 넣은 LyCORIS·그대로 복제 — ``APP_DEFAULTS``)이면
    로그만 남긴다(critic A6 — 사용자가 켜지 않은 기능으로 생성마다 경고하지 않는다. 게이트가 같은 블록을 뺄 때
    ``core/alwayson_propagation.gate_notices`` 와 같은 규칙). 훅이 없는 Forge 는 Forge UI 에서도 순정으로 합치므로 결과는
    Forge 와 같다. 보낸 블록만으로 출처를 가른다 — 블록 내용이 앱 기본값과 같으면 앱 기본값으로 본다."""
    sent = request.dora
    if sent is None or sent.is_stock:
        return []
    if sent == dora_infer_mode.APP_DEFAULTS:
        if not any(dora_infer_mode.INFOTEXT_MODE_KEY in p for p in images):
            logger.info("[DoRA] 앱 기본값 %s 을(를) 보냈지만 결과에 '%s' 기록이 없음 — Forge 의 weight_adapter 훅이 없어 "
                        "순정으로 합쳤다(알림 없음)", dora_infer_mode.describe(sent), dora_infer_mode.INFOTEXT_MODE_KEY)
        return []
    mode, inserted = sent.effective
    out = []
    if mode != dora_infer_mode.MODE_FORGE and not any(dora_infer_mode.INFOTEXT_MODE_KEY in p for p in images):
        out.append(Notice(
            CODE_DORA_NOT_APPLIED, LEVEL_WARNING,
            f"DoRA 추론 방식({dora_infer_mode.SHORT_LABELS[mode]})을 보냈지만 결과에 '{dora_infer_mode.INFOTEXT_MODE_KEY}' "
            "기록이 없습니다 — Forge 의 weight_adapter 훅이 없어 순정으로 합쳤습니다(Forge 버전 차이). Forge 콘솔의 "
            "'[DoRA Inference]' 줄을 확인하세요.",
            feature="dora", detail=f"mode:{mode}"))
    if inserted != dora_infer_mode.INSERT_KEEP and not any(dora_infer_mode.INFOTEXT_INSERT_KEY in p for p in images):
        out.append(Notice(
            CODE_DORA_NOT_APPLIED, LEVEL_WARNING,
            f"끼워 넣은 블록 정책({dora_infer_mode.SHORT_LABELS[inserted]})을 보냈지만 결과에 "
            f"'{dora_infer_mode.INFOTEXT_INSERT_KEY}' 기록이 없습니다 — ANIMA LoRA 블록 변환 훅이 없어 Forge 기본대로 "
            "복제했습니다. Forge 콘솔의 '[DoRA Inference]' 줄을 확인하세요.",
            feature="dora", detail=f"inserted:{inserted}"))
    return out


def result_notices(info: Any, payload: Any = None, *, capabilities: Any = None,
                   propagated_titles: Iterable[str] = ()) -> list[Notice]:
    """Forge 응답 info + 보낸 payload → 결과 알림. 판단할 재료가 없으면 빈 목록.

    ``propagated_titles``: 보조 패스가 메인 생성 설정에서 전달한 제목(P7). 그 기능의 알림 문구는 사용자가 이
    패널에서 켠 것이 아니므로 '(메인 생성 설정에서 전달됨)' 을 붙인다.
    """
    out = _result_notices(info, payload, capabilities=capabilities)
    features = {SAM_EXTRA_FEATURES[key][0] for key in (str(t).strip().lower() for t in propagated_titles or ())
                if key in SAM_EXTRA_FEATURES}
    if not features:
        return out
    return [replace(n, message=n.message + PROPAGATED_SUFFIX) if n.feature in features else n for n in out]


def _result_notices(info: Any, payload: Any = None, *, capabilities: Any = None) -> list[Notice]:
    images = image_parameters(info)
    if not images:
        return []
    request = requested_features(payload, live_pag_argc=_pag_argc(capabilities))
    total = len(images)
    out: list[Notice] = []

    # SAM3 — 실패하면 확장이 'SAM3 Enable' 을 걷고 'SAM3 Error' 를 남긴 채 이미지를 저장한다.
    errors = [str(p[KEY_SAM3_ERROR]) for p in images if p.get(KEY_SAM3_ERROR) not in (None, "")]
    if errors:
        counts: dict[str, int] = {}
        for reason in errors:
            counts[reason] = counts.get(reason, 0) + 1
        for reason, count in counts.items():
            scope = f"{count}/{total}장에서 " if total > 1 else ""
            hint = sam3_error_hint(reason, request.sam3_state)
            out.append(Notice(
                CODE_SAM3_ERROR, LEVEL_WARNING,
                f"SAM3 가 {scope}실패해 SAM3 없이 저장됐습니다 — {hint}. (원인: {_short(reason, 140)})",
                feature="sam3", detail=reason, hint=hint))
    elif request.sam3 and not any(_truthy(p.get(KEY_SAM3_ENABLE)) for p in images):
        out.append(Notice(
            CODE_SAM3_NOT_APPLIED, LEVEL_WARNING,
            "SAM3 를 켰지만 결과에 SAM3 적용 기록('SAM3 Enable')이 없습니다 — 확장이 설정을 받지 못했을 수 "
            "있습니다(구버전 확장은 설정 검증 실패를 기록 없이 넘깁니다). Forge 콘솔의 '[-] SAM3' 줄을 확인하세요.",
            feature="sam3"))
    elif request.sam3 and request.sam3_init_source:
        # 단독 SAM3·Refine 의 원본 기준 요청 — SAM3 가 적용된 결과에서만(실패는 위 두 알림이 말한다)
        out.extend(_sam3_source_notices(images))

    # Anima 3.8B — 'off: …' 는 3.8B 가 켜져야 했을 때(v2 번들 또는 명시적으로 켬)만 남는다.
    statuses = [_anima38_status(p) for p in images]
    offs = list(dict.fromkeys(s for s in statuses if s and s.lower().startswith("off")))
    model_kind = _result_model_kind(images) if offs else None
    for status in offs:
        out.append(Notice(
            CODE_ANIMA38_OFF, LEVEL_WARNING,
            f"Anima 3.8B(Qwen3.5 커넥터)가 꺼진 채 순정 Anima 로 생성됐습니다 — "
            f"{anima38_off_hint(status, kind=model_kind)}. "
            f"(기록: {_short(status, 120)})",
            feature="anima38", detail=status))
    if not offs and request.anima38_expected and not any(statuses):
        out.append(Notice(
            CODE_ANIMA38_NOT_APPLIED, LEVEL_WARNING,
            "Anima 3.8B 를 켰지만 결과에 적용 기록('Anima38')이 없습니다 — 확장이 3.8B 런타임을 불러오지 "
            "못했을 수 있습니다. Forge 콘솔의 '[Anima38]' 줄을 확인하세요.",
            feature="anima38"))

    # PAG/SEG/SLG — 붙었을 때만 'Anima Perturbation Guidance' 를 쓴다. 패스마다 지우고 다시 쓰므로
    # OOM 폴백(v0.30) 뒤의 하이레스 패스·다음 배치 이미지에는 이 키가 없다.
    if request.pag:
        missing = sum(1 for p in images if KEY_PAG not in p)
        if missing:
            parts = "/".join(request.pag_parts) or "PAG"
            scope = f"{missing}/{total}장" if total > 1 else "결과"
            partial = missing < total
            if _pag_old_build(capabilities, reason="PAG OOM 폴백"):
                why = ("모델이 Anima 가 아니거나 블록 번호가 모델에 없어 perturbation 이 건너뛰어졌습니다 "
                       "(구버전 Anima 가이던스)")
            elif partial:
                # 앞 장에는 붙었다 — 모델·블록 문제라면 모든 장에서 빠진다. 남은 설명은 OOM 폴백뿐이다.
                why = ("Safe PAG 가 VRAM 부족(OOM)으로 이번 생성의 PAG/SEG/SLG 를 끈 것으로 보입니다 — "
                       "해상도·배치 크기·블록 수를 줄여 보세요")
            else:
                # 모든 장에서 빠졌다 — Anima 가 아닌 모델(앱은 모델과 무관하게 블록을 보낸다)·잘못된 블록 번호·OOM
                # 가 infotext 로는 똑같이 보인다(하이레스를 켜도 마찬가지: 패스마다 키를 지우고 다시 쓴다).
                hires = " (OOM 이면 첫 패스에서 끈 채 하이레스 패스까지 갑니다)" if request.hires else ""
                why = ("모델이 Anima 가 아니거나 블록 번호가 모델에 없거나, VRAM 부족(OOM)으로 Safe PAG 가 "
                       f"PAG/SEG/SLG 를 껐습니다{hires} — Forge 콘솔의 '[Anima PAG]' 줄을 확인하세요")
            out.append(Notice(
                CODE_PAG_DROPPED, LEVEL_WARNING,
                f"{parts} 를 켰지만 {scope}의 infotext 에 '{KEY_PAG}' 가 없습니다 — {why}.",
                feature="anima_guidance", detail=f"{missing}/{total}"))

    out.extend(_dora_result_notices(request, images))

    # 부분 Anima LoRA 추측 변환 — Forge 설정을 켠 사용자에게 블록 대응이 틀릴 수 있음을 알린다(확장의 gr.Info 대응).
    guesses = list(dict.fromkeys(str(p[KEY_SPARSE_LORA_GUESS]).strip() for p in images
                                 if p.get(KEY_SPARSE_LORA_GUESS) not in (None, "")))
    for guess in guesses:
        out.append(Notice(
            CODE_LORA_SPARSE_GUESS, LEVEL_INFO,
            "일부 블록만 담은 Anima LoRA 를 순정 Forge 규칙으로 추측 변환해 적용했습니다 — 블록 대응이 틀릴 수 "
            f"있습니다(Forge 설정 'SAM Extra LoRA → 부분 LoRA 순정 추측 변환'). ({_short(guess, 160)})",
            feature="lora", detail=guess))
    return out


def notice_ttl(notice: Notice, ttl: float) -> float:
    """이 알림을 다시 띄우지 않을 시간 — 호출자 기본값과 종류별 최소값(``NOTICE_MIN_TTL_S``) 중 긴 쪽."""
    return max(float(ttl), NOTICE_MIN_TTL_S.get(notice.code, 0.0))


# ── 생성 전 알림 ───────────────────────────────────────────────────────────────
def is_loopback_url(url: Any) -> bool:
    """WebUI 주소가 이 PC(루프백)인가. 모양이 이상하면 True(원격이라고 단정하지 않는다)."""
    try:
        host = (urlsplit(str(url or "")).hostname or "").strip().lower()
    except ValueError:
        return True
    if not host:
        return True
    return host in ("localhost", "::1", "0.0.0.0") or host.startswith("127.")


def _bare_default_checkpoint(value: str) -> bool:
    """앱이 로컬에서 찾지 못해 이름만 남은 기본 체크포인트 'sam3.pt' 인가."""
    text = str(value or "").strip()
    if not text:
        return True     # 확장이 빈 값을 HF 기본값으로 본다
    for pure in (PureWindowsPath(text), PurePosixPath(text)):
        if len(pure.parts) != 1:
            return False
    return PurePath(text).name.lower() == HF_CHECKPOINT_NAME


def sam3_checkpoint_notice(checkpoint: Any, *, remote: bool = False) -> Optional[Notice]:
    """SAM3 체크포인트가 이름만 남은 'sam3.pt' 면 HF 자동 다운로드(약 3.4 GB) 경고 (다-5)."""
    value = str(checkpoint or "").strip()
    if value.lower() in HF_CHECKPOINT_KEYWORDS or not _bare_default_checkpoint(value):
        return None
    where = ("연결된 원격 Forge 에 sam3.pt 가 없으면" if remote
             else "이 PC 의 Forge 모델 폴더(models/sam3)에서 sam3.pt 를 찾지 못했습니다. Forge 쪽에도 없으면")
    return Notice(
        CODE_SAM3_HF_DOWNLOAD, LEVEL_WARNING,
        f"{where} sam-extra 가 생성 도중 Hugging Face {HF_CHECKPOINT_REPO}(승인 필요, 약 3.4 GB)에서 "
        "체크포인트를 내려받습니다 — 원하지 않으면 sam3.pt 를 models/sam3 에 두거나 SAM3 Checkpoint 를 다시 고르세요.",
        feature="sam3", detail=value or HF_CHECKPOINT_NAME)


def _cfg1_notice(request: RequestedFeatures, capabilities: Any) -> Optional[Notice]:
    """CFG 1 에서 켠 CFG base(SMC/APG/CWM) 알림.

    Forge 는 CFG 가 1 이면 네거티브를 아예 인코딩하지 않는다(modules/processing.py:481-483 ``self.uc = None``,
    하이레스 :1606-1608). 그래서 v0.30 확장은 어느 빌드든 CFG 1 패스에서 켠 base 를 전부 건너뛴다:

    - DCW-F 전 빌드(커밋 3522928 까지): cond_scale≈1 이면 SMC·APG·CWM 전부 건너뜀(``_cfg_base_skip_reason``).
    - DCW-F 빌드: 원본(namemechan/ComfyUI-DCW@66aaf9dd:dcw_node.py:877-878)처럼 SMC/CWM 에
      ``disable_cfg1_optimization`` 을 걸지만, 풀 uncond 가 없으니 SMC/CWM 도 비켜선다(``_uncond_ran_at_cfg1`` —
      확장 tests/test_dcw_origin.py ``test_forge_cfg_exactly_one_has_no_uncond_so_smc_cwm_step_aside``). APG 는 원본에
      없는 기능이라 uncond 가 있어도 CFG≈1 이면 건너뛴다.

    두 빌드는 버전 문자열·인자 수로 가를 수 없지만(둘 다 0.30.0·62개) CFG 1 에서의 답이 같아 가를 필요가 없다.
    원본 ComfyUI 노드는 CFG 1 에도 네거티브를 인코딩해 SMC/CWM 을 적용한다 — Forge 와의 호스트 차이라 문구에 적는다.
    가드가 아예 없는 v0.21.2 계열은 켠 base 전부가 빈 uncond 에 적용돼 망가지므로 따로 알린다.

    첫 패스와 하이레스 패스는 따로 본다 — 하이레스 패스의 cond_scale 은 ``p.hr_cfg`` 다
    (sd_samplers_cfg_denoiser.py:136-137, 확장 ``_pass_cfg_near_one``). hr_cfg 를 안 보내면 Forge 기본값 1."""
    hires_cfg = request.effective_hires_cfg
    first_one = _is_one(request.cfg_scale)
    if not request.cfg_bases or not (first_one or _is_one(hires_cfg)):
        return None
    bases = "/".join(request.cfg_bases)
    hires_only = not first_one           # 첫 패스는 적용되고 하이레스 패스만 CFG 1
    first_only = first_one and hires_cfg is not None and not _is_one(hires_cfg)
    # 억제 키(Notice.key)를 CFG 1 인 패스로 가른다 — 하이레스 기본값(hr_cfg 없음 → 1)이면 하이레스 INFO 가 생성마다
    # 뜨는데, 키가 같으면 뒤이은 첫 패스 CFG 1 WARNING 이 PRE_GENERATION_NOTICE_TTL_S 동안 가려진다.
    detail = f"{bases}@hires" if hires_only else (f"{bases}@first" if first_only else bases)
    where = "CFG 1"
    if hires_only:
        where = "하이레스 패스 CFG 1" + (
            f"(하이레스 CFG 를 보내지 않아 Forge 기본값 {FORGE_DEFAULT_HR_CFG:g})" if request.hires_cfg is None else "")
    raise_cfg = "하이레스 CFG" if hires_only else "CFG"
    if _pag_old_build(capabilities, reason="CFG 1 가드"):
        return Notice(
            CODE_CFG1_CFG_BASE, LEVEL_WARNING,
            f"{where} 에서 {bases} 를 켰습니다 — 구버전 Anima 가이던스(v0.21.2 계열)에는 CFG 1 가드가 없어 "
            f"빈 uncond 에 적용돼 결과가 크게 망가질 수 있습니다. {raise_cfg} 를 1 보다 크게 하거나 끄세요.",
            feature="anima_guidance", detail=detail)
    why = "Forge 는 CFG 1 에서 네거티브를 인코딩하지 않아(uncond 없음) 다듬을 CFG 가이던스가 없습니다"
    host = ("" if not {"SMC", "CWM"} & set(request.cfg_bases)
            else " (원본 ComfyUI DCW(+a) 노드는 CFG 1 에서도 네거티브로 SMC/CWM 을 적용합니다 — Forge 와의 차이)")
    if hires_only:
        first = f"(CFG {request.cfg_scale:g})" if request.cfg_scale is not None else ""
        return Notice(
            CODE_CFG1_CFG_BASE, LEVEL_INFO,
            f"{where} 에서는 {bases} 가 효과가 없습니다 — {why}. 첫 패스{first}에서는 적용되고 하이레스 패스에서는 "
            f"건너뜁니다{host}. 하이레스 패스에도 적용하려면 하이레스 CFG 를 1 보다 크게 하세요.",
            feature="anima_guidance", detail=detail)
    if first_only:
        return Notice(
            CODE_CFG1_CFG_BASE, LEVEL_INFO,
            f"CFG 1 에서는 {bases} 가 효과가 없습니다 — {why}. 첫 패스에서는 건너뛰고 하이레스 패스"
            f"(CFG {hires_cfg:g})에서만 적용됩니다{host}.",
            feature="anima_guidance", detail=detail)
    return Notice(
        CODE_CFG1_CFG_BASE, LEVEL_WARNING,
        f"CFG 1 에서는 {bases} 가 효과가 없습니다 — {why}. 확장이 건너뜁니다{host}. "
        "CFG 를 1 보다 크게 하거나 이 옵션을 끄세요.",
        feature="anima_guidance", detail=detail)


def _feature_of(title: str) -> tuple[str, str, str]:
    """제목 → (기능 id, 기능 이름, 확장 이름)."""
    key = title.strip().lower()
    if key in SAM_EXTRA_FEATURES:
        feature, label = SAM_EXTRA_FEATURES[key]
        return feature, label, "sam-extra 확장(forge_sam3_extension)"
    if key in OTHER_EXTENSIONS:
        feature, extension = OTHER_EXTENSIONS[key]
        return feature, title, extension
    return "", title, f"'{title}' 을(를) 제공하는 확장"


def _missing_script_notices(request: RequestedFeatures, capabilities: Any) -> list[Notice]:
    """스냅샷이 확실히 '없다'고 할 때만 — 모르면 아무것도 말하지 않는다(보수적)."""
    if not _capabilities_known(capabilities):
        return []
    sam_extra = [t for t in request.scripts if t.strip().lower() in SAM_EXTRA_FEATURES]
    if not sam_extra:
        return []
    if not getattr(capabilities, "installed", True):
        labels = ", ".join(_feature_of(t)[1] for t in sam_extra)
        return [Notice(
            CODE_EXTENSION_MISSING, LEVEL_ERROR,
            f"연결된 Forge 에 sam-extra 확장이 없어 요청이 거절됩니다(HTTP 422) — 켠 기능: {labels}. "
            "확장을 설치·활성화한 뒤 Forge 를 다시 시작하거나 이 기능들을 끄세요.",
            feature="sam_extra", detail=",".join(sam_extra))]
    out = []
    for title in sam_extra:
        detail = capabilities.script(title) if hasattr(capabilities, "script") else {}
        present = detail.get("img2img" if request.img2img else "present") if isinstance(detail, Mapping) else None
        if present is False:
            feature, label, _ext = _feature_of(title)
            tab = "img2img" if request.img2img else "txt2img"
            out.append(Notice(
                CODE_SCRIPT_MISSING, LEVEL_ERROR,
                f"연결된 Forge 의 sam-extra 에 '{title}' 스크립트가 없어({tab}) 요청이 거절됩니다(HTTP 422) — "
                f"{label} 를 끄거나 확장을 업데이트하세요.",
                feature=feature, detail=title))
    return out


def pre_generation_notices(payload: Any, *, capabilities: Any = None, remote: bool = False) -> list[Notice]:
    """보내기 직전 payload → 경고 (생성은 막지 않는다). WebUI(Forge) 백엔드에서만 부른다."""
    request = requested_features(payload, live_pag_argc=_pag_argc(capabilities))
    out = _missing_script_notices(request, capabilities)
    cfg1 = _cfg1_notice(request, capabilities)
    if cfg1 is not None:
        out.append(cfg1)
    if request.sam3:
        checkpoint = sam3_checkpoint_notice(request.sam3_checkpoint, remote=remote)
        if checkpoint is not None:
            out.append(checkpoint)
    return out


# ── HTTP 422 설명 ──────────────────────────────────────────────────────────────
_MISSING_SCRIPT_RES = (
    re.compile(r"^Script '(?P<title>.+)' not found$", re.IGNORECASE),
    re.compile(r"^always on script (?P<title>.+) not found$", re.IGNORECASE),
)


def _detail_of(body: Any) -> Any:
    if isinstance(body, (bytes, bytearray)):
        body = body.decode("utf-8", "replace")
    if isinstance(body, str):
        try:
            body = json.loads(body)
        except ValueError:
            return body.strip()
    if isinstance(body, Mapping):
        detail = body.get("detail")
        if detail in (None, "", []):
            detail = body.get("errors") or body.get("error")
        return detail
    return body


def missing_script_title(body: Any) -> Optional[str]:
    """Forge 422 본문이 '없는 스크립트' 거절이면 그 제목(보낸 표기가 아니라 Forge 가 적은 표기), 아니면 None.

    ``always on script X not found`` (modules/api/api.py:344-345), ``Script 'X' not found``. 재시도 판정
    (core/forge_optional_parts.plan_retry)과 설명(``explain_rejected_request``)이 같은 규칙을 쓴다.
    """
    detail = _detail_of(body)
    if not isinstance(detail, str):
        return None
    text = detail.strip()
    for pattern in _MISSING_SCRIPT_RES:
        match = pattern.match(text)
        if match:
            return match.group("title").strip() or None
    return None


# ── 샘플링 블록 게이트·보조 패스 전달 알림 (P7) ─────────────────────────────────
def block_not_sent_notice(title: str, *, img2img: bool) -> Notice:
    """메인 생성: 기능 스냅샷이 '없다'고 한 스크립트 블록을 422 대신 빼고 보냈다(경고)."""
    feature, label, _ext = _feature_of(title)
    tab = "img2img" if img2img else "txt2img"
    return Notice(
        CODE_BLOCK_NOT_SENT, LEVEL_WARNING,
        f"연결된 Forge 의 sam-extra 에 '{title}' 스크립트가 없어({tab}) 이 설정을 빼고 생성합니다 — {label} 를 "
        "끄거나 확장을 업데이트하세요.",
        feature=feature, detail=f"{title}@{tab}")


def block_deferred_notice(title: str) -> Notice:
    """확장 확인 전(기능 스냅샷을 모름)이라 이번 요청에는 보내지 않았다(정보)."""
    feature, _label, _ext = _feature_of(title)
    return Notice(
        CODE_BLOCK_DEFERRED, LEVEL_INFO,
        f"연결된 Forge 의 sam-extra 확인이 아직 끝나지 않아 이번 요청에는 '{title}' 을(를) 보내지 않았습니다 — "
        "확인되면 다음 요청부터 적용됩니다.",
        feature=feature, detail=title)


def propagation_dropped_notice(title: str, where: str) -> Notice:
    """보조 패스: 연결된 Forge 에 없어 메인 생성 설정을 전달하지 않았다(정보)."""
    feature, _label, _ext = _feature_of(title)
    return Notice(
        CODE_PROPAGATION_DROPPED, LEVEL_INFO,
        f"연결된 Forge 의 sam-extra 에 '{title}' 스크립트가 없어(img2img) 이 {where} 작업에는 메인 생성 설정의 "
        "그 블록을 전달하지 않았습니다.",
        feature=feature, detail=f"{title}@{where}")


def propagation_retried_notice(title: str, where: str) -> Notice:
    """보조 패스: 전달한 블록을 Forge 가 422 로 거절해 그 블록만 빼고 다시 보냈다(경고)."""
    feature, _label, _ext = _feature_of(title)
    return Notice(
        CODE_PROPAGATION_RETRIED, LEVEL_WARNING,
        f"Forge 가 메인 생성 설정에서 전달한 '{title}' 을(를) 거절해(HTTP 422, 스크립트 없음) 그 블록을 빼고 "
        f"{where} 을(를) 다시 실행했습니다 — 확장 목록을 다시 확인합니다.",
        feature=feature, detail=f"{title}@{where}")


# ── DoRA 추론 방식 알림 (P8, core/dora_infer_mode) ─────────────────────────────
def dora_comfy_stock_notice(summary: str) -> Notice:
    """ComfyUI 는 DoRA 추론 방식을 고를 수 없어 순정으로 합친다(정보, 사용자가 바꾼 값 + LoRA 가 있을 때만)."""
    return Notice(
        CODE_DORA_COMFY_STOCK, LEVEL_INFO,
        f"ComfyUI 는 DoRA 추론 방식({summary})을 지원하지 않아 순정으로 합칩니다 — DoRA LoRA 는 Forge 결과와 다를 수 "
        "있습니다. 이 설정은 Forge 에만 적용됩니다.",
        feature="dora", detail=summary)


def dora_choice_notice(problem: str) -> Notice:
    """고른 값을 연결된 확장이 몰라 이번 요청에는 DoRA 블록을 보내지 않았다(경고)."""
    return Notice(
        CODE_DORA_CHOICE, LEVEL_WARNING,
        f"{problem} — 이번 요청은 DoRA 추론 방식 없이(순정) 보냅니다. 확장을 업데이트하거나 DoRA 카드의 값을 바꾸세요.",
        feature="dora", detail=problem)


def dora_app_default_notice(summary: str) -> Notice:
    """옛 설정 파일에 DoRA 키가 없어 앱 기본값(Forge ui-config 의 txt2img 값)으로 시작한다(정보, 첫 로드 한 번).

    Forge 에만 해당한다 — ComfyUI 백엔드면 띄우지 않는다(ui/dora_infer_mode_ui.app_default_boot_notice). 시작 게이트에서
    백엔드를 아직 고르지 않았을 때도 틀리지 않게 문구를 Forge 로 한정한다."""
    return Notice(
        CODE_DORA_APP_DEFAULT, LEVEL_INFO,
        f"Forge 로 생성할 때 DoRA LoRA 를 Forge 설정과 같게 {summary} 방식으로 합칩니다(T2I·보조 작업, I2I·인페인트는 "
        "순정 · ComfyUI 는 늘 순정) — 바꾸거나 끄려면 파라미터의 'DoRA 추론 방식' 카드를 여세요.",
        feature="dora", detail=summary)


# ── Anima 3.8B 알림 (P9) ────────────────────────────────────────────────────────
def anima38_app_default_notice() -> Notice:
    """옛 설정 파일에 Anima38 키가 없어 앱 기본값(사용자 Forge ui-config txt2img — 부정 커넥터 켬)으로 시작한다(정보, 첫
    로드 한 번). v1 어댑터는 끔(사용자 결정 D1=B)이라 비 번들 Anima 결과는 그대로다. ComfyUI 컴파일러도 같은 블록을
    읽으므로 백엔드와 무관한 문구다."""
    return Notice(
        CODE_ANIMA38_APP_DEFAULT, LEVEL_INFO,
        "Anima 3.8B v2 번들 모델로 생성할 때 Forge 설정과 같게 부정 프롬프트도 Qwen3.5 커넥터로 인코딩합니다(T2I·보조 "
        "작업, I2I·인페인트는 끔) — 끄려면 파라미터의 'Anima 3.8B' 카드를 여세요.",
        feature="anima38", detail=anima38.describe(anima38.APP_T2I_DEFAULTS))


def anima38_comfy_v1_notice(*, i2i_off: bool = False, v1_enabled: bool = False, pick_adapter: str = "") -> Notice:
    """ComfyUI: 모듈 목록의 Qwen3.5·v1 어댑터 쌍만으로는 v1 을 켜지 않는다(정보, 생성 전). 예전 컴파일러는 그 쌍으로 v1 을
    자동으로 켰다 — 이제 Forge 처럼 카드의 v1 켜기를 따른다. 켰는데 파일이 없을 때의 동작은 백엔드마다 다르다(critic B8):
    ComfyUI 는 큐에 넣기 전에 컴파일 오류로 멈추고, Forge 는 'off: …' 를 남기고 순정으로 계속한다.

    ``i2i_off`` — I2I·인페인트 요청인데 카드의 'I2I·인페인트에도 적용'이 꺼져 블록이 없다. v1 을 이미 켰으면
    원인(그 토글)만 말하고, 꺼져 있으면 두 토글을 모두 말한다(P9 리뷰 4). ``pick_adapter`` — 모듈 목록의 어댑터
    이름이 카드 어댑터와 달라 v1 을 켤 때 카드 어댑터 칸에서 골라야 하는 이름(컴파일러는 카드 값만 쓴다 — P9 리뷰 2)."""
    if i2i_off and v1_enabled:
        message = ("'Anima 3.8B' 카드의 'I2I·인페인트에도 적용'이 꺼져 있어 이번 I2I·인페인트 생성에는 v1 어댑터를 "
                   "쓰지 않습니다(Forge img2img 탭 기본값과 같음) — 모듈 목록의 Qwen3.5·어댑터도 쓰지 않고 순정 Anima "
                   "조건으로 인코딩합니다.")
    else:
        when = ("'v1 어댑터 켜기'와 'I2I·인페인트에도 적용'을 모두 켤 때만" if i2i_off
                else "'v1 어댑터 켜기'를 켤 때만")
        message = ("모듈 목록에 Qwen3.5 인코더와 Anima v1 어댑터가 있지만, v1 어댑터는 Forge 와 같게 'Anima 3.8B' "
                   f"카드의 {when} 씁니다 — 이번 생성은 순정 Anima 조건으로 인코딩합니다. (켰는데 ComfyUI 에 "
                   "Qwen3.5·어댑터 파일이 없으면 생성 전에 오류로 멈춥니다. Forge 는 순정으로 계속합니다.)")
    if pick_adapter:
        message += (f" 켤 때는 카드의 어댑터 칸에서 '{pick_adapter}'를 고르세요 — 모듈 목록의 어댑터가 아니라 "
                    "카드에서 고른 어댑터를 씁니다.")
    return Notice(CODE_ANIMA38_COMFY_V1, LEVEL_INFO, message, feature="anima38")


# ── 메인 생성 재시도·Forge 옵션 덮어쓰기 알림 (P10) ───────────────────────────────
def app_block_retried_notice(title: str) -> Notice:
    """메인 생성: 앱이 스스로 넣은 블록(DoRA·Anima38 — Forge 설정 맞춤)을 Forge 가 422 로 거절해 그 블록만 빼고 다시
    생성했다(정보 — 사용자가 켜지 않은 앱 기본값이라 경고로 띄우지 않는다, critic A6). 요청의 블록 출처가 앱 기본값일
    때만 쓴다(그 밖은 ``block_retried_notice`` — P10 검토 2). 연결 때 받은 스냅샷이 틀렸다는 뜻이라 GUI 가 스냅샷을 다시
    받는다(``REFRESH_CAPABILITIES_CODES``) — 다음 요청부터는 게이트가 뺀다."""
    feature, _label, _ext = _feature_of(title)
    return Notice(
        CODE_APP_BLOCK_RETRIED, LEVEL_INFO,
        f"Forge 가 앱이 넣은 '{title}' 블록을 거절해(HTTP 422, 스크립트 없음) 그 블록 없이 다시 생성했습니다 — "
        "확장 목록을 다시 확인합니다.",
        feature=feature or "sam_extra", detail=title)


def block_retried_notice(title: str) -> Notice:
    """메인 생성: 사용자가 켰거나 출처를 모르는 블록(DoRA·Anima38)을 Forge 가 422 로 거절해 그 블록만 빼고 다시 생성했다
    (경고 — 이번 결과에 그 설정이 빠졌다, critic A6). 누가 넣었는지는 말하지 않는다(P10 검토 2). 스냅샷이 틀렸거나 확인
    전이라는 뜻이라 GUI 가 스냅샷을 다시 받는다 — 다음 요청부터는 게이트가 빼고 ``block_not_sent`` 로 알린다."""
    feature, label, _ext = _feature_of(title)
    return Notice(
        CODE_BLOCK_RETRIED, LEVEL_WARNING,
        f"연결된 Forge 가 '{title}' 블록을 거절해(HTTP 422, 스크립트 없음) 그 블록 없이 다시 생성했습니다 — 이번 결과에는 "
        f"{label} 설정이 적용되지 않았습니다. 확장을 업데이트하거나 그 설정을 끄세요(확장 목록을 다시 확인합니다).",
        feature=feature or "sam_extra", detail=title)


def _option_labels(labels: Iterable[str]) -> str:
    return ", ".join(f"'{label}'" for label in labels)


def forge_option_missing_notice(labels: Sequence[str]) -> Notice:
    """설정 › Forge 에서 정한 옵션이 연결된 sam-extra 에 없어 보내지 않았다(경고 — 그 요청은 Forge 설정대로)."""
    names = _option_labels(labels)
    return Notice(
        CODE_FORGE_OPTION_MISSING, LEVEL_WARNING,
        f"설정한 Forge 옵션 {names} 이(가) 연결된 Forge 의 sam-extra 에 없어 보내지 않았습니다 — 이 요청은 Forge "
        "설정대로 생성합니다. 확장을 업데이트하거나 설정 › Forge 에서 'Forge 설정 따름'으로 되돌리세요.",
        feature="forge_options", detail=names)


def forge_option_unverified_notice(labels: Sequence[str], *, no_config: bool) -> Notice:
    """옵션 목록을 확인하지 못해 앱의 sam-extra 설정을 보내지 않았다(정보 — 모르는 키는 요청 전체를 500 으로 실패시킨다)."""
    names = _option_labels(labels)
    why = ("Forge 설정 목록(/config)을 읽지 못해(로그인 설정·--nowebui)" if no_config
           else "연결된 Forge 의 sam-extra 확인이 아직 끝나지 않아")
    return Notice(
        CODE_FORGE_OPTION_UNVERIFIED, LEVEL_INFO,
        f"{why} 앱의 sam-extra 설정 {names} 을(를) 이번 요청에는 보내지 않았습니다 — Forge 설정대로 생성합니다.",
        feature="forge_options", detail=f"{'no_config' if no_config else 'unknown'}:{names}")


def forge_option_rejected_notice(labels: Sequence[str], *, frozen: bool, where: str = "",
                                 also: Sequence[str] = ()) -> Notice:
    """Forge 가 앱이 넣은 옵션을 거절해(HTTP 500 KeyError·설정 잠금) 앱 옵션 없이 다시 보냈다(경고). ``labels`` 는 거절된
    (잠금이면 기억해 보내지 않을) 설정, ``also`` 는 다시 보낼 때 함께 뺐지만 다음 요청부터 다시 보내는 설정(섹션·키
    잠금은 일부만 잠근다 — P10 검토 4). ``where`` 는 보조 작업 이름(비면 메인 생성)."""
    names = _option_labels(labels)
    # 잠금 인자는 셋(--freeze-settings·-in-sections·--freeze-specific-settings) — 어느 것인지 단정하지 않는다
    why = "설정 잠금 --freeze-settings…" if frozen else "KeyError — 이 Forge 에 없는 설정"
    what = f"{where} 을(를) 다시 실행했습니다" if where else "다시 생성했습니다"
    # 잠금은 Forge 시작 인자라 기능 확인에 드러나지 않는다 — 기억해 두고 보내지 않는다(P10 검토 3)
    then = "백엔드를 다시 연결할 때까지 이 설정은 보내지 않습니다" if frozen else "기능 확인을 새로 고칩니다"
    extra = f" 이번에는 {_option_labels(also)} 도 함께 뺐습니다 — 다음 요청부터 다시 보냅니다." if also else ""
    return Notice(
        CODE_FORGE_OPTION_REJECTED, LEVEL_WARNING,
        f"Forge 가 앱의 sam-extra 설정 {names} 을(를) 거절해(HTTP 500, {why}) 그 설정 없이 {what} — {then}.{extra}",
        feature="forge_options", detail=f"{'frozen' if frozen else 'key'}:{names}@{where}")


def forge_option_frozen_notice(labels: Sequence[str]) -> Notice:
    """이 Forge 가 설정 잠금으로 거절했던 옵션을 기억해 보내지 않았다(경고 — 사용자가 정한 값이 적용되지 않는다).
    ``labels`` 는 Forge 가 잠겼다고 이름을 댄 설정뿐이다(P10 검토 4). 기능 확인으로는 풀리지 않으므로 GUI 스냅샷을 다시
    받지 않는다(``REFRESH_CAPABILITIES_CODES`` 밖)."""
    names = _option_labels(labels)
    return Notice(
        CODE_FORGE_OPTION_FROZEN, LEVEL_WARNING,
        f"연결된 Forge 가 설정 잠금(--freeze-settings…)으로 거절했던 앱의 sam-extra 설정 {names} 을(를) 보내지 않았습니다 "
        "— 이 요청은 Forge 설정대로 생성합니다. 잠금 없이 Forge 를 다시 켰다면 백엔드를 다시 연결하세요(또는 설정 › Forge "
        "에서 'Forge 설정 따름'으로 되돌리세요).",
        feature="forge_options", detail=names)


def explain_rejected_request(status: Any, body: Any, payload: Any = None) -> Optional[str]:
    """Forge 가 요청을 거절한 응답(HTTP 422) → 어느 확장·기능 때문인지 한국어 설명. 422 가 아니면 None."""
    try:
        code = int(status)
    except (TypeError, ValueError):
        return None
    if code != 422:
        return None
    detail = _detail_of(body)
    if isinstance(detail, str):
        text = detail.strip()
        for pattern in _MISSING_SCRIPT_RES:
            match = pattern.match(text)
            if match:
                title = match.group("title").strip()
                feature, label, extension = _feature_of(title)
                sent = any(str(t).strip().lower() == title.lower()
                           for t in requested_features(payload).scripts)
                what = f"켠 기능: {label}" if sent else (f"기능: {label}" if feature else "요청의 스크립트")
                return (f"Forge 가 요청을 거절했습니다(HTTP 422): '{title}' 스크립트가 없습니다 — {extension} 이(가) "
                        f"설치·활성화돼 있지 않거나 버전이 달라 제목이 바뀌었습니다({what}). 그 기능을 끄거나 "
                        "확장을 설치·업데이트한 뒤 Forge 를 다시 시작하세요.")
        if "selectable script" in text.lower():
            return ("Forge 가 요청을 거절했습니다(HTTP 422): 선택형 스크립트(XYZ plot 등)를 alwayson_scripts 에 "
                    "넣었습니다 — 그 스크립트는 script_name 으로 보내야 합니다.")
        if text:
            return f"Forge 가 요청을 거절했습니다(HTTP 422): {_short(text, 200)}"
    if isinstance(detail, list):
        problems = []
        for item in detail[:3]:
            if isinstance(item, Mapping):
                loc = ".".join(str(part) for part in item.get("loc") or () if part != "body")
                problems.append(f"{loc}: {item.get('msg')}" if loc else str(item.get("msg")))
        if problems:
            return f"Forge 가 요청 형식을 거절했습니다(HTTP 422): {_short('; '.join(problems), 200)}"
    return "Forge 가 요청을 거절했습니다(HTTP 422) — 확장이 없거나 요청 값이 맞지 않습니다. Forge 콘솔을 확인하세요."


# ── 같은 알림 억제 ─────────────────────────────────────────────────────────────
class NoticeThrottle:
    """같은 ``Notice.key`` 를 ``ttl`` 초 안에 다시 띄우지 않는다 (배치·자동화의 토스트 폭주 방지)."""

    def __init__(self, clock: Callable[[], float] = time.monotonic, *, max_entries: int = 256):
        self._clock = clock
        self._seen: dict[str, float] = {}
        self._max = max_entries

    def allow(self, notice: Notice, ttl: float) -> bool:
        now = self._clock()
        last = self._seen.get(notice.key)
        if last is not None and now - last < ttl:
            return False
        self._seen[notice.key] = now
        if len(self._seen) > self._max:
            for key in sorted(self._seen, key=self._seen.get)[: len(self._seen) - self._max]:
                self._seen.pop(key, None)
        return True


__all__ = [
    "CODE_ANIMA38_APP_DEFAULT", "CODE_ANIMA38_COMFY_V1", "CODE_ANIMA38_NOT_APPLIED", "CODE_ANIMA38_OFF",
    "CODE_APP_BLOCK_RETRIED", "CODE_BLOCK_DEFERRED",
    "CODE_BLOCK_NOT_SENT", "CODE_BLOCK_RETRIED", "CODE_CFG1_CFG_BASE", "CODE_DORA_APP_DEFAULT", "CODE_DORA_CHOICE",
    "CODE_DORA_COMFY_STOCK", "CODE_DORA_NOT_APPLIED", "CODE_EXTENSION_MISSING", "CODE_FORGE_OPTION_FROZEN",
    "CODE_FORGE_OPTION_MISSING", "CODE_FORGE_OPTION_REJECTED", "CODE_FORGE_OPTION_UNVERIFIED",
    "CODE_LORA_SPARSE_GUESS", "CODE_PAG_DROPPED", "CODE_PROPAGATION_DROPPED", "CODE_PROPAGATION_RETRIED",
    "CODE_REQUEST_REJECTED", "CODE_SAM3_ERROR", "CODE_SAM3_HF_DOWNLOAD", "CODE_SAM3_NOT_APPLIED",
    "CODE_SAM3_SOURCE_FALLBACK", "CODE_SAM3_SOURCE_UNSUPPORTED",
    "CODE_SCRIPT_MISSING", "INFO_KEY", "KEY_ANIMA38_STATUS", "KEY_PAG", "KEY_SAM3_ENABLE", "KEY_SAM3_ERROR",
    "KEY_SAM3_SOURCE", "KEY_SAM3_VERSION", "KEY_SPARSE_LORA_GUESS", "LEVEL_ERROR", "LEVEL_INFO", "LEVEL_WARNING", "NOTICE_MIN_TTL_S",
    "Notice", "NoticeThrottle", "NoticedImage", "PRE_GENERATION_NOTICE_TTL_S", "PROPAGATED_SUFFIX",
    "REFRESH_CAPABILITIES_CODES", "RESULT_NOTICE_TTL_S", "RequestedFeatures", "anima38_app_default_notice",
    "anima38_comfy_v1_notice", "anima38_off_hint",
    "app_block_retried_notice", "block_deferred_notice", "block_not_sent_notice", "block_retried_notice",
    "dora_app_default_notice", "dora_choice_notice", "dora_comfy_stock_notice", "explain_rejected_request",
    "forge_option_frozen_notice", "forge_option_missing_notice", "forge_option_rejected_notice",
    "forge_option_unverified_notice", "image_parameters", "is_loopback_url", "missing_script_title", "notice_ttl",
    "notices_from_info", "notices_of", "notices_to_dicts", "pre_generation_notices", "propagation_dropped_notice",
    "propagation_retried_notice", "requested_features", "result_notices", "sam3_checkpoint_notice",
    "sam3_error_hint", "sam3_failure_repeats", "standalone_failure_text", "standalone_sam3_failure",
]
