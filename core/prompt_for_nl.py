# core/prompt_for_nl.py
"""nl_caption(태그→자연어) LLM 입력 정리 — 순수 함수(Qt 의존 없음, 테스트 가능).

LLM에는 '묘사할 내용'만 보내야 정확하다. 다음은 묘사 대상이 아니거나(오히려
네거티브를 *있는 것*으로 묘사하게 만들어) 작은 모델을 혼란시키므로 제거한다:
  - 네거티브 가중치 그룹: (tag, tag:-1.0)  ← 빼고 싶은 것 (음수 weight)
  - LoRA: <lora:name:0.8>, 그리고 @트리거(화풍/로라 태그) @enast 등
  - 선행/후행 고정 품질·메타: masterpiece, best quality, score_9, highres,
    absurdres, newest/oldest, year2025, rating: 등

유지: 인물수(1boy/2girls), 캐릭터(시리즈 괄호 'alex (stardew valley)' 포함),
작품(copyright), 일반(메인) 태그. 양수 가중치 괄호 (tag:1.2)는 내용만 남기고 weight 제거.
"""
import re

from core.prompt_comments import strip_prompt_comments

# 묘사 대상이 아닌 품질/메타 태그 (정확히 일치 시 제거, 소문자 비교)
_META_EXACT = {
    'masterpiece', 'best quality', 'high quality', 'normal quality', 'low quality',
    'worst quality', 'bad quality', 'highres', 'absurdres', 'lowres', 'high resolution',
    'very aesthetic', 'aesthetic', 'official art', 'newest', 'recent', 'mid', 'early',
    'oldest', 'ai-generated', 'ai generated',
}
# 접두 일치 시 제거 (score_9, year2025, rating:safe, source_anime 등)
_META_PREFIX = ('score_', 'year ', 'year2', 'year1', 'rating:', 'source_')

# 네거티브 가중치 그룹: (...:-N) — 내부 콤마 포함 통째로 제거
_RE_NEG_GROUP = re.compile(r'\([^()]*:\s*-\d+(?:\.\d+)?\s*\)')
# 양수 가중치 그룹: (content:N) → content (weight 제거). 'char (series)'는 콜론 없어 미해당
_RE_POS_GROUP = re.compile(r'\(([^()]*?):\s*\d+(?:\.\d+)?\s*\)')
# <lora:...>
_RE_LORA = re.compile(r'<lora:[^>]*>', re.IGNORECASE)


def clean_tags_for_nl(prompt: str) -> str:
    """nl_caption LLM 입력용으로 태그 문자열 정리. 인물수/캐릭터/작품/메인만 남긴다."""
    if not prompt:
        return ''
    # 주석(// 줄 끝까지, /* 블록 */)은 생성에 보내지 않는다 — LLM 에도 보내지 않아야 꺼 둔 태그가
    # 자연어 캡션으로 되살아나지 않는다. 결과를 덧붙일 칸 글(base)은 주석째 그대로 둔다(호출자 몫).
    s = strip_prompt_comments(prompt)
    s = _RE_LORA.sub('', s)
    # 네거티브 그룹 제거 — 중첩( ((tag:-1.2)) )에서 안쪽이 먼저 지워지고 바깥
    # 빈 괄호가 남으므로, 변화가 없을 때까지 반복.
    for _ in range(5):
        s2 = _RE_NEG_GROUP.sub('', s)
        if s2 == s:
            break
        s = s2
    s = _RE_POS_GROUP.sub(r'\1', s)     # 양수 가중치 → 내용만
    # 남은 빈/외톨이 괄호 정리 — '1girl, (), blue hair' → '1girl, , blue hair'
    s = re.sub(r'\(\s*\)', '', s)

    out = []
    for raw in s.split(','):
        t = raw.strip()
        # 영숫자/한글이 전혀 없는 토큰(외톨이 '(' ')' '()' 등)은 버림.
        # 'alex (stardew valley)'처럼 괄호가 정상 포함된 토큰은 보존(영숫자 있음).
        if not t or not re.search(r'[A-Za-z0-9가-힣]', t):
            continue
        low = t.lower()
        if t.startswith('@'):           # 화풍/LoRA 트리거 태그
            continue
        if low in _META_EXACT:
            continue
        if any(low.startswith(p) for p in _META_PREFIX):
            continue
        out.append(t)
    return ', '.join(out)
