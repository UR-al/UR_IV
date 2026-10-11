# core/refine_prompt.py
"""Refine 패널 프롬프트 수술 — 순수 함수(테스트 가능). Qt 의존 없음.

sam-extra의 Refine 워크플로 핵심은 "Target 토큰이 들어간 **콤마 세그먼트를 통째로**
지우고, 첫 매치 자리에 Replacement를 **한 번만** 끼워 넣는" 것이다.

단순 substring replace를 쓰면 안 되는 이유:
    "white shirt, black necktie" 에서 'shirt'를 지우면 → "white , black necktie"
    처럼 고아 조각('white')이 남아 프롬프트가 오염된다.
세그먼트 단위로 지우면 → "nude, black necktie" 로 깔끔하다.

동작은 확장 `sam3ext/ui_refine.py`의 `_strip_patterns_with_replacement` /
`_apply_prompt_sr` / `_parse_detect_tokens`와 1:1로 맞췄다. 앱과 Forge UI에서
같은 입력에 같은 프롬프트가 나와야 하기 때문이다(CLAUDE.md 'Forge Neo 동기화 원칙').
"""
import re

# SAM3가 detect prompt를 쪼개는 방식과 동일 — ',' '/' ';' 개행
_TOKEN_SPLIT = re.compile(r"[,/;\n]")


def normalize_prompt(text: str) -> str:
    """공백/콤마 정리 — 빈 세그먼트와 연속 콤마 제거."""
    text = re.sub(r"\s+", " ", text or "")
    text = re.sub(r"\s*,\s*", ", ", text)
    text = re.sub(r"(,\s*){2,}", ", ", text)
    return text.strip(" ,")


def parse_detect_tokens(detect_prompt: str) -> list:
    """detect prompt를 SAM3와 같은 규칙으로 토큰 분리."""
    if not detect_prompt:
        return []
    return [t.strip() for t in _TOKEN_SPLIT.split(detect_prompt) if t.strip()]


def strip_patterns_with_replacement(text: str, patterns, replacement: str) -> str:
    """patterns 중 하나라도 포함한 콤마 세그먼트를 제거하고,
    첫 제거 위치에 replacement를 **한 번만** 삽입한다.

    매치가 하나도 없으면 원문을 그대로 돌려준다(정규화도 하지 않음).
    replacement가 빈 문자열이면 삽입 없이 삭제만 한다.
    """
    patterns = [p for p in (patterns or []) if p]
    if not patterns or not text:
        return text

    out = []
    inserted = False
    matched_any = False
    for raw in text.split(","):
        seg = raw.strip()
        if not seg:
            continue
        if any(pat in seg for pat in patterns):
            matched_any = True
            if not inserted and replacement:
                out.append(replacement)
                inserted = True
            continue
        out.append(seg)

    if not matched_any:
        return text
    return ", ".join(out)


def apply_prompt_sr(text: str, rules_field: str) -> str:
    """S/R 규칙(줄당 하나)을 적용.

    문법::
        pat                    = replacement
        pat1, pat2, ..., patN  = replacement

    여러 패턴을 쓰면 각각을 포함한 세그먼트가 모두 지워지고 replacement는
    첫 매치 자리에 한 번만 들어간다 ('nude, nude'가 되지 않게).
    """
    if not text or not rules_field:
        return text
    out = text
    matched_any = False
    for raw_line in rules_field.splitlines():
        line = raw_line.strip()
        if not line or "=" not in line:
            continue
        pattern_part, _, replacement = line.partition("=")
        patterns = [p.strip() for p in pattern_part.split(",") if p.strip()]
        if not patterns:
            continue
        new_out = strip_patterns_with_replacement(out, patterns, replacement.strip())
        if new_out != out:
            matched_any = True
            out = new_out
    if not matched_any:
        return text
    return normalize_prompt(out)


def build_refine_prompts(*, main_prompt: str = '', main_negative: str = '',
                         target: str = '', replacement: str = '',
                         negative: str = '',
                         inherit_main: bool = True,
                         inherit_negative: bool = True) -> dict:
    """Refine 패널 입력 → 실제 인페인트에 쓸 (prompt, negative_prompt).

    확장 `handle_refine_click`의 규칙 그대로:
      · Target 있음 + inherit ON  → 메인 프롬프트에서 Target 세그먼트를 지우고
                                     그 자리에 Replacement를 넣은 것이 최종 프롬프트
      · Target 없음 + inherit ON  → 메인 프롬프트 뒤에 Replacement를 덧붙임
      · inherit OFF               → Replacement만 사용
    네거티브는 Replacement를 넣지 않는다(새 주제를 anti-prompt에 흘리면 안 됨).

    반환: {'prompt', 'negative_prompt', 'detect_tokens', 'changed'}

    다섯 입력 모두 프롬프트 주석(``//`` · ``/* */``, core/prompt_comments)을 먼저 뗀다 — 아래 정규화가
    줄바꿈을 공백으로 바꾸면 줄 주석이 다음 태그를 삼킨다. ``changed`` 는 주석을 뗀 뒤 기준이다.
    """
    from core.prompt_comments import strip_prompt_comments
    main_prompt = strip_prompt_comments(main_prompt or '')
    main_negative = strip_prompt_comments(main_negative or '')
    target = strip_prompt_comments(target or '')
    replacement = strip_prompt_comments(replacement or '').strip()
    negative = strip_prompt_comments(negative or '').strip()

    detect_tokens = parse_detect_tokens(target)

    sr_positive = f"{', '.join(detect_tokens)} = {replacement}" if detect_tokens else ''
    sr_negative = f"{', '.join(detect_tokens)} = " if detect_tokens else ''

    cleaned_main = apply_prompt_sr(main_prompt, sr_positive) if main_prompt else main_prompt
    cleaned_neg = apply_prompt_sr(main_negative, sr_negative) if main_negative else main_negative

    # ── positive ──
    if inherit_main and cleaned_main:
        if detect_tokens:
            prompt = cleaned_main
        else:
            prompt = f"{cleaned_main}, {replacement}".rstrip(', ') if replacement else cleaned_main
    else:
        prompt = replacement

    # ── negative ──
    if inherit_negative and cleaned_neg:
        neg = f"{cleaned_neg}, {negative}".rstrip(', ') if negative else cleaned_neg
    else:
        neg = negative

    return {
        'prompt': normalize_prompt(prompt),
        'negative_prompt': normalize_prompt(neg),
        'detect_tokens': detect_tokens,
        'changed': cleaned_main != main_prompt or cleaned_neg != main_negative,
    }
