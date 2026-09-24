# core/infotext.py
"""A1111 infotext 문법 — 파서·직렬화기의 유일본 (Qt·PIL 비의존).

형식 (A1111 modules/infotext_utils.create_infotext)::

    <positive prompt — 여러 줄 가능, 비어 있으면 줄 자체가 없음>
    Negative prompt: <negative — 비어 있으면 줄 자체가 없음>
    Steps: 20, Sampler: Euler, CFG scale: 7, Seed: 12345, Size: 512x768, ...
    Template: …            ← Dynamic Prompts 같은 확장이 덧붙이는 꼬리 줄(값이 여러 줄일 수 있음)
    Negative Template: …

- A1111 은 전체를 ``.strip()`` 하므로 빈 positive 면 'Negative prompt:' 가, 프롬프트가 둘 다
  없으면 'Steps:' 가 첫 줄이 된다.
- 값에 쉼표·콜론·줄바꿈이 있으면 JSON 문자열로 따옴표친다(quote). dict/list 값은 따옴표 없이
  JSON 그대로 쓴다 — ``Hashes: {"vae": …}``, ``Civitai resources: [{…}]``, ``Tiled Diffusion: {…}``.
- 키에 '.' 이 들어가는 확장도 있다 — ``Anima 3.8B adapter: …``.

파라미터 줄은 이렇게 알아본다.
1. ``Steps: <정수>`` 로 시작하는 마지막 줄(네거티브 줄보다 뒤) — 뒤 세그먼트가 무엇이든, 뒤에
   어떤 줄이 오든(여러 줄 Template 값, 빈 'Negative Template:') 그 줄부터가 파라미터 꼬리다.
2. Steps 줄이 없을 때만: 모든 세그먼트가 'Key: value' 인 줄이면서 쌍이 3개 이상(A1111
   parse_generation_parameters 기준)이거나 알려진 파라미터 키로 시작하는 2쌍 이상 — 그리고 그
   뒤 줄이 모두 'Key: value' 꼬리 줄일 때. 'Size: huge, masterpiece' 나 '(masterpiece:1.2), …'
   같은 프롬프트 줄은 여기서 걸러진다.

모든 판정은 줄 길이에 선형이다. 예전에는 줄 전체를 중첩 정규식 하나로 fullmatch 해서, 실패하는
줄이면 LoRA 해시 하나마다 백트래킹이 두 배로 늘었다(12개에 수십 초, GIL 을 쥔 채 GUI 정지).
이제 세그먼트는 손으로 나누고, 정규식은 모호하지 않은 조각에만 쓴다.
"""
from __future__ import annotations

import json
import re
from typing import Any

# 키 — A1111 re_param 의 ``\w[\w \-/]+`` 에 '.', '(', ')' 를 더했다(Anima 3.8B adapter 등).
# 문자 집합에 ':'·','·'"' 가 없어 매치 비용은 세그먼트 길이를 넘지 않는다.
_KEY_CHARS = r"[\w .\-/()]"
_PAIR_KEY_RE = re.compile(rf"(\w{_KEY_CHARS}+):\s*")
# 따옴표 값의 본문과 닫는 따옴표 — 두 갈래의 첫 글자가 겹치지 않아 선형이다.
_QUOTED_BODY_RE = re.compile(r'(?:[^"\\]|\\.)*"')
# 값 뒤 구분자(쉼표 또는 줄 끝)
_SEPARATOR_RE = re.compile(r"\s*(?:,|$)")
# 괄호 값 스캔에서 의미 있는 글자
_BRACKET_TOKEN_RE = re.compile(r'[\\"{}\[\]]')
_WHITESPACE_RE = re.compile(r"\s*")

_NEGATIVE_PREFIX_RE = re.compile(r"negative prompt:[ \t]*", re.IGNORECASE)
# A1111·Forge 파라미터 줄의 첫 세그먼트
_STEPS_LINE_RE = re.compile(r"Steps:\s*\d+\s*(?:,|$)")
# 파라미터 줄 뒤에 붙는 'Key: value' 한 줄 항목 (Template: … / 빈 'Negative Template:')
_LINE_ITEM_RE = re.compile(rf"(\w{_KEY_CHARS}*):(?:\s|$)")
# 줄 전체가 값인 항목 — 값에 'a: b, c: d' 가 있어도 쌍으로 쪼개지 않는다.
_WHOLE_LINE_KEYS = frozenset({"template", "negative template"})
# 'Steps' 없이 시작하는 파라미터 줄을 알아보는 첫 키 (Forge·구버전·타 도구)
_KNOWN_LEADING_KEYS = frozenset({
    "sampler", "cfg scale", "seed", "size", "model", "model hash", "denoising strength",
    "schedule type", "version", "clip skip",
})
_INT_RE = re.compile(r"^-?\d+$")
_FLOAT_RE = re.compile(r"^-?\d*\.\d+$")


def normalize_infotext(text: str) -> str:
    return str(text or "").replace("\r\n", "\n").replace("\r", "\n").strip()


# ─────────────────────────────────────────────
# 한 줄 → (key, raw value) 쌍
# ─────────────────────────────────────────────

def _bracketed_end(line: str, start: int) -> int:
    """``line[start]`` 의 '{'/'[' 와 짝이 맞는 닫는 괄호 다음 위치(짝이 없으면 -1).

    JSON 문자열("…", 역슬래시 이스케이프) 안의 괄호·쉼표는 세지 않는다.
    """
    depth = 0
    in_string = False
    pos = start
    while True:
        match = _BRACKET_TOKEN_RE.search(line, pos)
        if match is None:
            return -1
        char = match.group()
        pos = match.end()
        if in_string:
            if char == "\\":
                pos += 1
            elif char == '"':
                in_string = False
        elif char == '"':
            in_string = True
        elif char in "{[":
            depth += 1
        elif char in "}]":
            depth -= 1
            if depth <= 0:
                return pos if depth == 0 else -1


def _after_separator(line: str, pos: int) -> int:
    """pos 뒤가 (공백 +) 쉼표/줄 끝이면 구분자 다음 위치, 아니면 -1."""
    if pos < 0:
        return -1
    match = _SEPARATOR_RE.match(line, pos)
    return match.end() if match else -1


def scan_param_pairs(line: str) -> tuple[list[tuple[str, str]], bool]:
    """한 줄 → ([(key, 원문 값)], 모든 세그먼트가 'Key: value' 였는가).

    A1111 re_param 과 같은 규칙으로 값을 읽는다 — JSON 따옴표 값은 따옴표째, 그 밖은 다음
    쉼표까지. 여기에 따옴표 없는 JSON dict/list 값(괄호 짝이 맞고 뒤가 구분자)을 한 값으로
    읽는 규칙을 더했다. 'Key:' 로 시작하지 않는 세그먼트는 건너뛰고 clean=False 로 알린다.

    선형 보장: 닫는 따옴표가 더 없으면(이후 따옴표 시작도 모두 실패한다) 따옴표 값을, 괄호 값이
    한 번 실패하면 괄호 값을 더 찾지 않는다. 성공한 스캔은 서로 겹치지 않는다.
    """
    pairs: list[tuple[str, str]] = []
    clean = True
    quotes_open = True
    brackets_ok = True
    size = len(line)
    pos = 0
    while True:
        pos = _WHITESPACE_RE.match(line, pos).end()
        if pos >= size:
            break
        key_match = _PAIR_KEY_RE.match(line, pos)
        if key_match is None:
            clean = False
            comma = line.find(",", pos)
            if comma < 0:
                break
            pos = comma + 1
            continue
        start = key_match.end()
        value_end = next_pos = -1
        opener = line[start] if start < size else ""
        if opener == '"' and quotes_open:
            body = _QUOTED_BODY_RE.match(line, start + 1)
            if body is None:
                quotes_open = False
            else:
                next_pos = _after_separator(line, body.end())
                if next_pos >= 0:
                    value_end = body.end()
        elif opener in ("{", "[") and brackets_ok:
            close = _bracketed_end(line, start)
            next_pos = _after_separator(line, close)
            if next_pos >= 0:
                value_end = close
            else:
                brackets_ok = False
        if value_end < 0:
            comma = line.find(",", start)
            value_end = size if comma < 0 else comma
            next_pos = size if comma < 0 else comma + 1
        pairs.append((key_match.group(1).strip(), line[start:value_end]))
        pos = next_pos
    return pairs, clean


# ─────────────────────────────────────────────
# 줄 분류
# ─────────────────────────────────────────────

def _bare_parameter_pairs(line: str) -> list[tuple[str, str]]:
    """Steps 없이 쓰는 파라미터 줄이면 그 쌍, 아니면 [].

    모든 세그먼트가 'Key: value' 이고, 쌍이 3개 이상(A1111 기준)이거나 알려진 파라미터 키로
    시작하는 2쌍 이상. 'Template: …' 같은 줄 전체 항목은 값에 쌍이 있어도 파라미터 줄이 아니다.
    """
    text = line.strip()
    if not text or _NEGATIVE_PREFIX_RE.match(text):
        return []
    pairs, clean = scan_param_pairs(text)
    if not clean or not pairs or pairs[0][0].lower() in _WHOLE_LINE_KEYS:
        return []
    if len(pairs) >= 3 or (len(pairs) >= 2 and pairs[0][0].lower() in _KNOWN_LEADING_KEYS):
        return pairs
    return []


def _line_item_key(line: str) -> str:
    """'Key: value' 한 줄 항목이면 키, 아니면 ''."""
    text = line.strip()
    if _NEGATIVE_PREFIX_RE.match(text):
        return ""
    match = _LINE_ITEM_RE.match(text)
    return match.group(1).strip() if match else ""


def _is_tail_extra_line(line: str) -> bool:
    """Steps 없는 파라미터 줄 뒤에 와도 되는 줄 — 빈 줄 또는 'Template: …' 같은 한 줄 항목."""
    return not line.strip() or bool(_line_item_key(line))


def _starts_with_known_key(pairs: list[tuple[str, str]]) -> bool:
    return bool(pairs) and pairs[0][0].lower() in _KNOWN_LEADING_KEYS


def find_parameter_tail(lines: list[str]) -> int:
    """파라미터 꼬리가 시작하는 줄 번호(없으면 len(lines)).

    1. 네거티브 줄보다 뒤에 있는 마지막 ``Steps: N`` 줄. 프롬프트에 붙여 넣은 파라미터 줄보다
       실제 파라미터 줄(항상 마지막)이 이기고, 그 뒤 줄은 무엇이든 꼬리다(A1111·예전 파서와 같다).
    2. Steps 줄이 없으면 엄격한 파라미터 줄 중 마지막 것 — 그 뒤가 모두 한 줄 항목일 때만.
       바로 위 줄들도 알려진 키(Sampler·Size …)로 시작하는 파라미터 줄이면 함께 꼬리로 묶는다
       ('hair: blue, eyes: red, …' 같은 프롬프트 줄은 알려진 키가 아니라 프롬프트로 남는다).
    """
    for index in range(len(lines) - 1, -1, -1):
        text = lines[index].strip()
        if _NEGATIVE_PREFIX_RE.match(text):
            break
        if _STEPS_LINE_RE.match(text):
            return index
    for index in range(len(lines) - 1, -1, -1):
        if _bare_parameter_pairs(lines[index]):
            while index > 0 and _starts_with_known_key(_bare_parameter_pairs(lines[index - 1])):
                index -= 1
            return index
        if not _is_tail_extra_line(lines[index]):
            break
    return len(lines)


def split_infotext(text: str) -> tuple[str, str, str]:
    """A1111 infotext → (positive, negative, 파라미터 꼬리 원문).

    - 네거티브가 비어 줄이 없는 이미지도 파라미터 줄이 프롬프트로 새지 않는다.
    - 빈 positive('Negative prompt:' 로 시작)와 파라미터만 있는 텍스트('Steps:' 로 시작)도 읽는다.
    - 꼬리는 재조립하지 않은 원문이라 저장·복사에서 따옴표와 Template 줄이 보존된다.
    """
    body = normalize_infotext(text)
    if not body:
        return "", "", ""
    lines = body.split("\n")
    tail_at = find_parameter_tail(lines)
    positive: list[str] = []
    negative: list[str] = []
    in_negative = False
    for line in lines[:tail_at]:
        if not in_negative:
            match = _NEGATIVE_PREFIX_RE.match(line.lstrip())
            if match:
                in_negative = True
                negative.append(line.lstrip()[match.end():])
                continue
            positive.append(line)
        else:
            negative.append(line)
    return "\n".join(positive).strip(), "\n".join(negative).strip(), "\n".join(lines[tail_at:]).strip()


# ─────────────────────────────────────────────
# 값 따옴표 / 파라미터 꼬리 → dict / dict → 줄
# ─────────────────────────────────────────────

def unquote_param_value(value: str) -> str:
    """A1111 infotext_utils.unquote — JSON 문자열로 따옴표친 값만 푼다(\\" 이스케이프 포함)."""
    text = str(value)
    if len(text) < 2 or text[0] != '"' or text[-1] != '"':
        return text
    try:
        decoded = json.loads(text)
    except (ValueError, TypeError):
        return text
    return decoded if isinstance(decoded, str) else text


def _is_bracketed_value(text: str) -> bool:
    """줄바꿈 없이 괄호 짝이 맞는 한 덩어리 — 따옴표 없이 써도 다시 한 값으로 읽힌다."""
    return (text[:1] in ("{", "[") and "\n" not in text
            and _bracketed_end(text, 0) == len(text))


def quote_param_value(value: Any) -> str:
    """A1111 infotext_utils.quote — 쉼표·콜론·줄바꿈이 든 값만 JSON 따옴표.

    dict/list 는 A1111 처럼 따옴표 없는 JSON 으로, 파싱에서 받은 JSON 원문 문자열
    (``{"vae": …}``)은 그대로 둔다 — 다시 파싱하면 같은 값이 된다.
    """
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False)
    text = str(value)
    if "," not in text and "\n" not in text and ":" not in text:
        return text
    if _is_bracketed_value(text):
        return text
    return json.dumps(text, ensure_ascii=False)


def _typed_param_value(raw_value: str) -> Any:
    value = raw_value.strip()
    if value.startswith('"'):
        return unquote_param_value(value)
    if _INT_RE.match(value):
        return int(value)
    if _FLOAT_RE.match(value):
        return float(value)
    return value


def parse_parameters_text(text: str) -> dict[str, Any]:
    """파라미터 꼬리(``Steps: …`` 줄 + 꼬리 줄) → dict.

    - 파라미터 줄은 따옴표·JSON 괄호를 아는 규칙으로 쌍을 나눈다(따옴표 없는 정수/실수는 숫자로).
    - 'Template: …' 같은 한 줄 항목은 줄 전체를 값으로 둔다(값 안의 쉼표에서 자르지 않는다).
    - 'Key:' 로 시작하지 않는 줄은 바로 앞 한 줄 항목의 이어지는 줄이다(여러 줄 Template 값).
    """
    result: dict[str, Any] = {}
    line_key = ""
    for line in normalize_infotext(text).split("\n"):
        stripped = line.strip()
        if not stripped:
            continue
        item_key = _line_item_key(stripped)
        whole_line = item_key.lower() in _WHOLE_LINE_KEYS
        if _STEPS_LINE_RE.match(stripped) or (not whole_line and _bare_parameter_pairs(stripped)):
            for key, value in scan_param_pairs(stripped)[0]:
                result[key] = _typed_param_value(value)
            line_key = ""
        elif item_key:
            line_key = item_key
            result[line_key] = unquote_param_value(stripped.partition(":")[2].strip())
        elif line_key:
            result[line_key] = f"{result[line_key]}\n{stripped}"
        else:
            for key, value in scan_param_pairs(stripped)[0]:
                result[key] = _typed_param_value(value)
    return result


def format_parameters_line(params: dict) -> str:
    """dict → 'Key: value, …' (A1111 create_infotext 규칙). None 값은 뺀다."""
    return ", ".join(
        f"{key}: {quote_param_value(value)}" for key, value in (params or {}).items() if value is not None
    )


def format_infotext(prompt: str, negative: str, parameters_text: str) -> str:
    """(positive, negative, 파라미터 꼬리) → A1111 infotext. 빈 부분은 줄째 생략한다."""
    negative_text = f"\nNegative prompt: {negative}" if negative else ""
    return f"{prompt or ''}{negative_text}\n{parameters_text or ''}".strip()
