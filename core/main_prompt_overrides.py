# core/main_prompt_overrides.py
"""T2I '메인 프롬프트 대체' 텍스트 — 다른 프롬프트로 T2I 설정을 재사용할 때 떼는 순수 로직 (Qt 비의존).

T2I 탭의 세 입력은 모두 "비우면 메인 프롬프트 사용" 의미로, 그 T2I 메인 프롬프트를 위해
쓰인 **내용** 텍스트다(설정값이 아니다).

- Hires Prompt / Negative   → ``hr_prompt`` / ``hr_negative_prompt``
  (Forge hr_prompt, Comfy ForgeNeoHiresFix positive_text/negative_text — 두 번째 패스 전체를 다시 칠한다)
- ADetailer 슬롯 prompt     → ``alwayson_scripts.ADetailer.args[*].ad_prompt`` / ``ad_negative_prompt``
- SAM3 인페인트 prompt      → ``alwayson_scripts["SAM3 Mask"].args[*].sam3_inpaint_prompt`` /
  ``sam3_negative_prompt`` (빌더가 비었으면 빌드 시점 메인 프롬프트로 채워 둔다 — core/sam3_args)

T2I 빌더 스냅샷을 **다른** 프롬프트(Comic 컷 등)에 입히면 이 텍스트가 그대로 따라가, 컷의
Hires/얼굴 패스가 컷 프롬프트 대신 T2I 텍스트로 다시 칠해진다(오류 없이 틀린 그림).
``clear_main_prompt_overrides`` 로 떼면 각 패스는 백엔드 기본 규칙대로 그 요청의 최종
prompt/negative 로 돌아간다(Forge·Comfy 노드 모두 빈 값 = 메인 사용). SAM3 는 T2I 와 똑같이
최종 프롬프트가 정해진 뒤(와일드카드·훅 이후) ``fill_sam3_prompt_fallback`` 으로 채운다.

검출 대상(sam3_prompt/sam3_exclude_prompt)·샘플러·체크포인트 같은 **설정**은 건드리지 않는다.
"""
from __future__ import annotations

from typing import Any, Iterator, MutableMapping

HIRES_PROMPT_KEYS: tuple[str, ...] = ("hr_prompt", "hr_negative_prompt")
ADETAILER_PROMPT_KEYS: tuple[str, ...] = ("ad_prompt", "ad_negative_prompt")
SAM3_PROMPT_KEYS: tuple[str, ...] = ("sam3_inpaint_prompt", "sam3_negative_prompt")

ADETAILER_SCRIPT = "ADetailer"
SAM3_SCRIPT = "SAM3 Mask"  # core.sam3_args.SCRIPT_SAM3 와 같은 값 (tests 가 고정)


def _script_dicts(payload: MutableMapping[str, Any], script: str) -> Iterator[MutableMapping[str, Any]]:
    """alwayson_scripts[script].args 안의 dict(슬롯/state) 들."""
    scripts = payload.get("alwayson_scripts")
    if not isinstance(scripts, MutableMapping):
        return
    block = scripts.get(script)
    args = block.get("args") if isinstance(block, MutableMapping) else None
    if not isinstance(args, list):
        return
    for item in args:
        if isinstance(item, MutableMapping):
            yield item


def clear_main_prompt_overrides(payload: MutableMapping[str, Any]) -> MutableMapping[str, Any]:
    """payload(제자리)에서 T2I 메인 프롬프트용 대체 텍스트를 뗀다. 같은 payload 를 돌려준다.

    Hires 키는 빼고(T2I 도 비었으면 안 보낸다), ADetailer/SAM3 는 빈 문자열로 둔다(슬롯 스키마 유지).
    """
    for key in HIRES_PROMPT_KEYS:
        payload.pop(key, None)
    for slot in _script_dicts(payload, ADETAILER_SCRIPT):
        for key in ADETAILER_PROMPT_KEYS:
            if key in slot:
                slot[key] = ""
    for state in _script_dicts(payload, SAM3_SCRIPT):
        for key in SAM3_PROMPT_KEYS:
            if key in state:
                state[key] = ""
    return payload


def fill_sam3_prompt_fallback(payload: MutableMapping[str, Any]) -> MutableMapping[str, Any]:
    """SAM3 인페인트 prompt/negative 가 비었으면 payload 의 최종 prompt/negative 로 채운다(제자리).

    core.sam3_args.build_state 의 폴백 규칙과 같다 — 다만 지연 프롬프트(와일드카드·훅)가 풀린
    뒤에 불러야 최종 텍스트가 들어간다.
    """
    prompt = str(payload.get("prompt") or "")
    negative = str(payload.get("negative_prompt") or "")
    for state in _script_dicts(payload, SAM3_SCRIPT):
        if not str(state.get("sam3_inpaint_prompt") or "").strip() and prompt:
            state["sam3_inpaint_prompt"] = prompt
        if not str(state.get("sam3_negative_prompt") or "").strip() and negative:
            state["sam3_negative_prompt"] = negative
    return payload
