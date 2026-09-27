# core/forge_optional_parts.py
"""앱이 스스로 덧붙인 요청 부분을 Forge 가 거절하면 그것만 빼고 다시 보낼지 — 순수 판정(Qt·네트워크 없음).

'앱이 덧붙인 부분'은 사용자가 그 작업 패널에서 켠 것이 아니라 앱이 맞춰 준 것이다:
- P7: 메인 생성 설정에서 보조 패스(Refine·SAM3·ADetailer)로 전달한 alwayson 블록 제목(``OptionalParts.titles``)
- P10: 앱이 넣은 ``override_settings`` 키(``OptionalParts.option_keys`` — P10 이 500 KeyError 가지를 더한다)

규칙 (critic A3, B10):
- 422 본문이 가리킨 **제목 하나만** 뺀다(``always on script X not found`` / ``Script 'X' not found``). 그 제목이 앱이
  덧붙인 것이 아니면(보조 경로 자신의 SAM3 Mask·ADetailer, 외부 호출자의 블록) None — 지금처럼 실패한다.
- 500(P10): ``{"error": "KeyError", "errors": "'<key>'"}`` 이고 그 키가 앱이 넣은 옵션이면, 또는 설정 잠금
  (``AssertionError`` — 전역 ``changing settings is disabled``, 또는 ``not possible to set '<key>' … frozen …`` 이 앱이
  넣은 키를 가리킬 때, modules/options.py:112,129,134)이면 앱이 넣은 옵션 키를 **모두** 뺀다(한 번에 — 옵션은 시도 한
  번으로 센다). 잠금 문구가 앱이 넣지 않은 키만 가리키거나 키도 전역 문구도 없으면 None.
  Forge 는 이 오류들을 샘플링 전(``process_images`` 의 override 적용, processing.py:813)에 낸다 — 다시 보내도 안전하다.
- 잠겼다고 적는 키(``RetryPlan.frozen`` — 백엔드가 주소별로 기억해 다음 요청부터 보내지 않는다)는 Forge 가 이름을 댄
  앱 키뿐이다. 섹션·키 잠금(``--freeze-settings-in-sections``·``--freeze-specific-settings``)은 확장 옵션 일부만 잠그므로
  (확장 옵션은 섹션이 여럿) 함께 뺀 나머지는 이번 한 번만 빠진다. 이름 없는 전역 잠금만 앱 키 전부다(P10 검토 4).
- 부분이 줄어드는 동안만 반복한다. 시도 수 상한 = 1 + 전달 제목 수 + 옵션 종류 수(옵션은 한 번에 모두 뺀다),
  그리고 ``MAX_ATTEMPTS_CAP``.
- 재시도 전 취소 확인은 송신 쪽(``WebUIBackend._generate``·``_run_img2img_postprocess``)이 한다.

메인 생성(``WebUIBackend._generate``)에서 다시 보낼 수 있는 제목은 앱이 Forge 설정에 맞춰 넣는 블록(DoRA·Anima38 —
``core/alwayson_propagation.main_retry_titles``, 제목표)이다. 그 블록이 앱 기본값인지 사용자 값인지는 요청의 비공개 출처
키(``PROVENANCE_KEY`` — 백엔드가 요청 전에 뗀다)로 알고 알림 수준만 가른다(critic A2·A6, P10 검토 2).
"""
from __future__ import annotations

import copy
import json
import re
from dataclasses import dataclass, replace
from typing import Any, Mapping, Optional

MAX_ATTEMPTS_CAP = 8


@dataclass(frozen=True)
class OptionalParts:
    titles: tuple = ()        # 앱이 전달한 alwayson 제목(P7)
    option_keys: tuple = ()   # 앱이 넣은 override_settings 키(P10)

    def __bool__(self) -> bool:
        return bool(self.titles or self.option_keys)


@dataclass(frozen=True)
class RetryPlan:
    payload: dict             # 거절된 부분을 뺀 깊은 복사
    remaining: OptionalParts
    code: str                 # CODE_PROPAGATION_RETRIED(422 제목) / CODE_FORGE_OPTION_REJECTED(500 옵션)
    removed: tuple
    reason: str = ""          # 500 옵션: REASON_KEY_ERROR / REASON_FROZEN
    frozen: tuple = ()        # REASON_FROZEN: Forge 가 잠겼다고 이름을 댄 앱 키(전역 잠금은 앱 키 전부) — ``removed`` 의 부분


REASON_KEY_ERROR, REASON_FROZEN = "key_error", "frozen"
_GLOBAL_FROZEN = "changing settings is disabled"    # --freeze-settings(키 이름 없음, modules/options.py:112)
_QUOTED_KEY_RE = re.compile(r"'([^']+)'")


def max_attempts(parts: Optional[OptionalParts]) -> int:
    """이 요청을 최대 몇 번 보낼 수 있나 — 1 + 전달 제목 수 + 옵션 종류 수(있으면 1), 상한 ``MAX_ATTEMPTS_CAP``."""
    if not parts:
        return 1
    return min(MAX_ATTEMPTS_CAP, 1 + len(parts.titles) + (1 if parts.option_keys else 0))


def _without_title(payload: Mapping, folded: str) -> dict:
    out = copy.deepcopy(dict(payload))
    scripts = out.get("alwayson_scripts")
    if isinstance(scripts, Mapping):
        out["alwayson_scripts"] = {name: block for name, block in scripts.items()
                                   if str(name).strip().casefold() != folded}
    return out


def _error_body(body: Any) -> Optional[Mapping]:
    """Forge 오류 본문(dict·JSON 문자열·bytes) → dict. 읽을 수 없으면 None."""
    if isinstance(body, (bytes, bytearray)):
        body = body.decode("utf-8", "replace")
    if isinstance(body, str):
        try:
            body = json.loads(body)
        except ValueError:
            return None
    return body if isinstance(body, Mapping) else None


def _classify_option_rejection(body: Any, option_keys) -> Optional[tuple]:
    """Forge 500 본문 → ``(REASON_*, 잠긴 앱 키)``, 앱이 넣은 옵션 탓이 아니면 None. 잠긴 앱 키는 REASON_FROZEN 일 때만
    채운다 — 전역 잠금은 앱 키 전부, 섹션·키 잠금은 문구가 이름을 댄 앱 키(``option_keys`` 순서)."""
    data = _error_body(body)
    ordered = tuple(dict.fromkeys(str(key) for key in (option_keys or ())))
    if data is None or not ordered:
        return None
    error = str(data.get("error") or "")
    text = str(data.get("errors") or data.get("detail") or "").strip()
    if error == "KeyError":
        key = text[1:-1] if len(text) >= 2 and text[0] == text[-1] and text[0] in "'\"" else text
        return (REASON_KEY_ERROR, ()) if key in ordered else None
    if error != "AssertionError":
        return None
    lowered = text.lower()
    if _GLOBAL_FROZEN in lowered:
        return REASON_FROZEN, ordered      # --freeze-settings: 키를 대지 않는다 — 모든 설정이 잠겼다
    if "frozen" not in lowered:
        return None
    # --freeze-settings-in-sections·--freeze-specific-settings: "not possible to set '<key>' because … frozen …" — 섹션
    # 이름도 따옴표 안이지만 앱 키와 겹치지 않는다. 앱 키를 대지 않으면(호출자 키가 잠겼거나 Forge 잠금 문구가 아니면)
    # 앱 키를 빼도 소용없다 — 지금처럼 실패한다.
    named = set(_QUOTED_KEY_RE.findall(text))
    locked = tuple(key for key in ordered if key in named)
    return (REASON_FROZEN, locked) if locked else None


def rejected_option_reason(body: Any, option_keys) -> Optional[str]:
    """Forge 500 본문 → 앱이 넣은 옵션 때문이면 이유(REASON_*), 아니면 None.

    ``handle_exception``(modules/api/api.py) 모양 ``{"error": 타입 이름, "errors": str(e)}``. KeyError 는 ``str(e)`` 가
    ``"'<key>'"`` 다(modules/options.py:165 ``self.data_labels[key]``)."""
    verdict = _classify_option_rejection(body, option_keys)
    return verdict[0] if verdict else None


def plan_retry(status: Any, body: Any, payload: Mapping, parts: Optional[OptionalParts]) -> Optional[RetryPlan]:
    """거절 응답 → 다시 보낼 계획, 아니면 None(지금처럼 실패). 입력은 바꾸지 않는다."""
    if not parts:
        return None
    try:
        code = int(status)
    except (TypeError, ValueError):
        return None
    if code == 422 and parts.titles:
        from core.sam_extra_notices import CODE_PROPAGATION_RETRIED, missing_script_title

        title = missing_script_title(body)
        folded = str(title or "").strip().casefold()
        match = next((t for t in parts.titles if str(t).strip().casefold() == folded), None) if folded else None
        if match is None:
            return None
        remaining = replace(parts, titles=tuple(t for t in parts.titles if str(t).strip().casefold() != folded))
        return RetryPlan(_without_title(payload, folded), remaining, CODE_PROPAGATION_RETRIED, (match,))
    if code == 500 and parts.option_keys:
        verdict = _classify_option_rejection(body, parts.option_keys)
        if verdict is None:
            return None
        reason, locked = verdict
        from core.forge_override_settings import without_app_keys
        from core.sam_extra_notices import CODE_FORGE_OPTION_REJECTED

        # 다시 보낼 때는 앱 키를 모두 뺀다(옵션은 시도 한 번) — 잠겼다고 적는 것은 Forge 가 이름을 댄 키뿐(P10 검토 4)
        stripped = without_app_keys(copy.deepcopy(dict(payload)), parts.option_keys)
        return RetryPlan(stripped, replace(parts, option_keys=()), CODE_FORGE_OPTION_REJECTED,
                         tuple(parts.option_keys), reason, locked)
    return None


__all__ = ["MAX_ATTEMPTS_CAP", "OptionalParts", "REASON_FROZEN", "REASON_KEY_ERROR", "RetryPlan", "max_attempts",
           "plan_retry", "rejected_option_reason"]
