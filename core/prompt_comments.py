# core/prompt_comments.py
"""프롬프트 주석 — ``// 줄 끝까지`` 와 ``/* 블록 */``. 순수 함수(Qt·torch 없음).

주석은 편집 칸(섹션 칸·프리셋·프로필·템플릿·세션 백업)에만 남고, 백엔드로 가는 요청에는 절대
들어가지 않는다 — Forge 와 ComfyUI 가 같은 결과를 받도록 갈라지기 전(``_build_generation_payload``)에
떼고, 두 백엔드 입구(``strip_request_comments``)에서 한 번 더 뗀다. 그래서 Forge infotext·Comfy 그래프·
프롬프트 기록에는 실제로 보낸(주석을 뗀) 글자가 남는다.

언제 주석이 시작되나 — ``//`` 또는 ``/*`` 가 다음 자리에 있을 때만:
  * 텍스트 맨 앞, 공백 문자(``_WS``) 뒤, ASCII 쉼표 뒤, 또는 블록 주석이 끝난 바로 그 자리.
  그래서 ``http://``·``file:///``·``a//b``·``>//<``·``__style/*__``·``~/x/~``·``<lora:dir/name:1>`` 은 글자 그대로다.
  이스케이프 문법은 없다.
  * ``//`` = 줄 주석(``\\n``/``\\r`` 앞까지, 줄바꿈은 남는다).
  * ``/*`` = 블록 주석(처음 만나는 ``*/`` 까지, 포함). 닫히지 않으면(``open``) **그 줄 끝까지만** —
    떨어진 ``/*`` 하나가 프롬프트 나머지를 통째로 지우지 않게. 중첩 없음, 홀로 선 ``*/`` 는 평문.

떼어 낸 뒤 정리(:func:`strip_prompt_comments`) — 주석만 있던 줄은 줄째로 없애고, 내용 뒤의 줄바꿈은
남겨 BREAK 같은 줄 단위 문법이 그대로다. 주석이 없는 줄은 손대지 않는다. 결과는 멱등이다.
줄 주석은 태그를 끝낸다 — 블록 칸·태그 도구(``split_tags``)가 ``a // 메모\\nb`` 를 두 태그로 보므로, 보내는
글자도 같게 괄호 밖 줄 주석 왼쪽 내용 뒤에 쉼표를 하나 둔다(``a,\\nb``). 왼쪽이 이미 쉼표로 끝나거나, 다음 줄에
내용이 없거나 쉼표로 시작하거나, 왼쪽 끝 낱말이 ``BREAK``·``AND`` 이면 두지 않는다.

프론트(``frontend/src/utils/promptComments.ts``)가 같은 문법을 블록 칸 흐리게 표시·자동완성 끄기에
쓴다 — 두 구현은 ``frontend/src/utils/promptComments.cases.json`` 골든 사례로 묶여 있다.
"""
from __future__ import annotations

import re
from collections.abc import Container, Iterable, Mapping
from typing import Any

#: 주석 시작 앞에 올 수 있는 공백 문자(명시 집합 — ``str.isspace`` 와 JS ``\s`` 의 차이를 피한다)
_WS = frozenset(" \t\n\r\f\v\u00a0\u3000")
#: 줄 끝 문자 — 줄 주석·닫히지 않은 블록은 여기 앞에서 끝난다
_EOL_CHARS = "\r\n"
_EOL = re.compile(r"[\r\n]")
#: 줄 끝을 뺀 공백(가로 공백) — 떼어 낸 자리 양옆을 정리할 때 쓴다
_HWS = "".join(sorted(_WS - set(_EOL_CHARS)))
#: 줄 주석이 태그 경계인지 볼 때 세는 괄호(블록 칸 splitPromptBlocks 와 같다). ``\\`` 뒤 글자는 세지 않는다
_OPEN_BRACKETS = "([{"
_CLOSE_BRACKETS = ")]}"
#: 줄 주석 왼쪽이 이 낱말로 끝나면 쉼표를 두지 않는다 — 줄 단위 문법(Forge BREAK·AND)
_LINE_KEYWORD_TAIL = re.compile("(?:^|[" + re.escape("".join(sorted(_WS))) + r",])(?:BREAK|AND)\Z")
#: 밑줄 뒤 ``//``·``/*`` — 밑줄을 공백으로 바꾸면 그 자리가 주석 시작이 된다
_UNDERSCORE_OPENS_COMMENT = re.compile(r"_(?=/[/*])")

LINE, BLOCK, OPEN = "line", "block", "open"

#: 요청 최상위에서 주석을 떼는 키
PROMPT_KEYS = ("prompt", "negative_prompt", "hr_prompt", "hr_negative_prompt")
#: ``alwayson_scripts['ADetailer'].args`` 의 슬롯(dict)마다
ADETAILER_PROMPT_KEYS = ("ad_prompt", "ad_negative_prompt")
#: ``alwayson_scripts['SAM3 Mask'].args[0]`` state
SAM3_PROMPT_KEYS = ("sam3_prompt", "sam3_exclude_prompt", "sam3_inpaint_prompt", "sam3_negative_prompt")
GUARDED_KEYS = frozenset(PROMPT_KEYS + ADETAILER_PROMPT_KEYS + SAM3_PROMPT_KEYS)

_ADETAILER_TITLE = "adetailer"
_SAM3_TITLE = "sam3 mask"


def comment_spans(text: str) -> list[tuple[int, int, str]]:
    """주석 구간 ``[(시작, 끝, 종류)]`` — 종류는 ``line``·``block``·``open``. 끝은 포함하지 않는다."""
    if not text or "/" not in text:
        return []
    spans: list[tuple[int, int, str]] = []
    n = len(text)
    block_end = -1
    i = text.find("/")
    while 0 <= i < n - 1:
        nxt = text[i + 1]
        if nxt in "/*" and (i == 0 or i == block_end or text[i - 1] in _WS or text[i - 1] == ","):
            if nxt == "*":
                close = text.find("*/", i + 2)
                if close >= 0:
                    end = close + 2
                    spans.append((i, end, BLOCK))
                    block_end = end
                    i = text.find("/", end)
                    continue
                kind = OPEN
            else:
                kind = LINE
            m = _EOL.search(text, i + 2)
            end = m.start() if m else n
            spans.append((i, end, kind))
            i = text.find("/", end)
            continue
        i = text.find("/", i + 1)
    return spans


def comment_spans_through_eol(text: str) -> list[tuple[int, int, str]]:
    """:func:`comment_spans` 와 같되 줄 주석·닫히지 않은 블록은 뒤따르는 줄 끝(``\\r\\n`` 은 하나)까지 품는다.

    주석을 가렸다 되돌리는 쪽(실시간 정리)이 쓴다 — 줄바꿈이 공백으로 바뀌면 다음 줄이 주석에 먹히므로
    줄 끝도 주석과 함께 가린다.
    """
    return [(start, end if kind == BLOCK else end + _eol_len(text, end), kind)
            for start, end, kind in comment_spans(text)]


def has_prompt_comment(text: str) -> bool:
    """주석이 하나라도 있는가."""
    return bool(text) and "/" in text and bool(comment_spans(text))


def _eol_len(text: str, at: int) -> int:
    """``at`` 자리 줄 끝 길이(``\\r\\n`` = 2, ``\\n``·``\\r`` = 1, 줄 끝이 아니면 0)."""
    if text.startswith("\r\n", at):
        return 2
    return 1 if at < len(text) and text[at] in _EOL_CHARS else 0


def _last_char(chunks: list[str]) -> str:
    for chunk in reversed(chunks):
        if chunk:
            return chunk[-1]
    return ""


def _tail(chunks: list[str], size: int) -> str:
    """조각 목록 끝 ``size`` 글자."""
    tail = ""
    for chunk in reversed(chunks):
        tail = chunk + tail
        if len(tail) >= size:
            break
    return tail[-size:]


def _bracket_depth(text: str, start: int, end: int, depth: int) -> int:
    """``text[start:end]`` 을 지난 뒤 괄호 깊이(``\\`` 뒤 글자는 건너뛴다, 0 아래로 내려가지 않는다)."""
    i = start
    while i < end:
        ch = text[i]
        if ch == "\\":
            i += 2
            continue
        if ch in _OPEN_BRACKETS:
            depth += 1
        elif ch in _CLOSE_BRACKETS:
            depth = max(0, depth - 1)
        i += 1
    return depth


def _content_follows(text: str, spans: list[tuple[int, int, str]], index: int, pos: int) -> bool:
    """``pos`` 뒤에 (공백·줄 끝·이어지는 주석을 건너뛰고) 쉼표가 아닌 내용이 오는가."""
    n = len(text)
    k = index + 1
    while pos < n:
        if text[pos] in _WS:
            pos += 1
            continue
        if k < len(spans) and spans[k][0] == pos:
            pos = spans[k][1]
            k += 1
            continue
        return text[pos] != ","
    return False


def _rstrip_chunks(chunks: list[str]) -> None:
    """조각 목록 끝의 가로 공백을 지운다(제자리)."""
    while chunks:
        tail = chunks[-1].rstrip(_HWS)
        if tail:
            chunks[-1] = tail
            return
        chunks.pop()


def _strip_spans(text: str, spans: list[tuple[int, int, str]], tag_breaks: bool = True) -> str:
    # 지금까지의 출력(L)은 조각 목록으로 들고 끝자리만 본다 — 주석이 많은 긴 글도 한 번 훑기.
    out: list[str] = []
    pos = 0
    n = len(text)
    depth = 0
    for index, (start, end, kind) in enumerate(spans):
        if start < pos:          # 앞 정리에서 이미 지난 자리(겹침 없음 — 방어)
            continue
        if start > pos:
            out.append(text[pos:start])
            depth = _bracket_depth(text, pos, start, depth)
        pos = end
        # (b) 왼쪽이 가로 공백·줄 처음이면 오른쪽 가로 공백은 버린다 — 'a /* x */ b' → 'a b',
        #     '/* x */ b' → 'b'
        last = _last_char(out)
        if (not last or last in _HWS or last in _EOL_CHARS) and pos < n and text[pos] in _HWS:
            while pos < n and text[pos] in _HWS:
                pos += 1
        # (b2) 오른쪽이 쉼표면 왼쪽 가로 공백을 지운다 — 'a /* x */, b' → 'a, b'
        if pos < n and text[pos] == ",":
            _rstrip_chunks(out)
        # (c) 주석이 줄 끝(또는 글 끝)에 닿았으면 왼쪽 가로 공백을 지우고, 주석만 있던 줄은 줄째로 없앤다
        eol = _eol_len(text, pos)
        if eol or pos >= n:
            _rstrip_chunks(out)
            last = _last_char(out)
            if not last or last in _EOL_CHARS:
                if eol:
                    pos += eol
                elif out:
                    tail = out[-1]
                    out[-1] = tail[:-2] if tail.endswith("\r\n") else tail[:-1]
            # (d) 괄호 밖 줄 주석이 내용 뒤에서 줄을 끝내면 태그 경계 — 다음 줄 내용과 붙지 않게 쉼표 하나
            elif (tag_breaks and eol and kind != BLOCK and depth == 0 and last != ","
                  and not _LINE_KEYWORD_TAIL.search(_tail(out, 6))
                  and _content_follows(text, spans, index, pos)):
                out.append(",")
    out.append(text[pos:])
    return "".join(out)


def strip_prompt_comments(text: str, *, tag_breaks: bool = True) -> str:
    """주석을 뗀 텍스트. 주석이 없으면 **같은 객체**를 돌려준다.

    ``tag_breaks=False`` — 줄 하나가 항목인 글(와일드카드 파일)은 줄 주석 뒤에 쉼표를 두지 않는다.
    """
    if not isinstance(text, str):
        return text
    spans = comment_spans(text)
    if not spans:
        return text
    return _strip_spans(text, spans, tag_breaks)


def ends_in_line_comment(text: str) -> bool:
    """텍스트가 줄 주석(또는 닫히지 않은 ``/*``)으로 끝나는가 — 뒤에 붙인 글자가 주석에 먹힌다."""
    spans = comment_spans(text)
    return bool(spans) and spans[-1][2] in (LINE, OPEN) and spans[-1][1] == len(text)


def append_prompt_text(base: str, add: str, sep: str = ", ") -> str:
    """``base`` 뒤에 ``add`` 를 잇는다. ``base`` 가 줄 주석으로 끝나면 ``sep`` 대신 줄바꿈으로."""
    base = base or ""
    add = add or ""
    if not base:
        return add
    if not add:
        return base
    if ends_in_line_comment(base):
        return f"{base}\n{add}"
    return f"{base}{sep}{add}"


def underscores_to_spaces(tag: str) -> str:
    """태그 밑줄 → 공백(블록 칸·자동완성이 넣는 모양). 밑줄 뒤에 ``//``·``/*`` 가 오는 태그(Danbooru
    ``z.a.t.o._//_i_love_the_world_and_everything_in_it``)는 바꾸면 그 자리부터 주석이 되어 뒤가 잘리므로
    그대로 둔다."""
    if not tag or "_" not in tag or _UNDERSCORE_OPENS_COMMENT.search(tag):
        return tag
    return tag.replace("_", " ")


def comment_code(token: str) -> str:
    """토큰의 실제 내용(주석을 떼고 앞뒤 공백 정리). 주석만 있는 토큰이면 ``''``."""
    return strip_prompt_comments(token or "").strip()


def filter_tokens_keep_comments(tokens: Iterable[str], keep: Container[str]) -> list[str]:
    """``keep`` 에 든 토큰만 남긴다 — 비교는 주석을 뗀 내용으로, 주석만 있는 토큰은 늘 남긴다."""
    out: list[str] = []
    for token in tokens:
        code = comment_code(token)
        if not code or code in keep:
            out.append(token)
    return out


def _strip_keys(mapping: Mapping, keys: tuple[str, ...]):
    """``mapping`` 의 ``keys`` 문자열 값에서 주석을 뗀 사본. 바뀐 게 없으면 같은 객체."""
    out = None
    for key in keys:
        value = mapping.get(key)
        if isinstance(value, str):
            stripped = strip_prompt_comments(value)
            if stripped is not value:
                if out is None:
                    out = dict(mapping)
                out[key] = stripped
    return mapping if out is None else out


def _strip_block(block: Any, keys: tuple[str, ...], *, first_only: bool):
    if not isinstance(block, Mapping):
        return block
    args = block.get("args")
    if not isinstance(args, (list, tuple)):
        return block
    new_args = None
    for index, arg in enumerate(args):
        if first_only and index > 0:
            break
        if not isinstance(arg, Mapping):
            continue
        stripped = _strip_keys(arg, keys)
        if stripped is not arg:
            if new_args is None:
                new_args = list(args)
            new_args[index] = stripped
    if new_args is None:
        return block
    out = dict(block)
    out["args"] = new_args if isinstance(args, list) else tuple(new_args)
    return out


def strip_request_comments(payload):
    """요청 payload(Forge 모양) 사본에서 프롬프트 주석을 뗀다 — 두 백엔드 입구의 마지막 방어선.

    최상위 :data:`PROMPT_KEYS`, ``ADetailer`` 블록의 모든 슬롯(dict) :data:`ADETAILER_PROMPT_KEYS`,
    ``SAM3 Mask`` 블록 ``args[0]`` 의 :data:`SAM3_PROMPT_KEYS`. 제목은 대소문자·앞뒤 공백 무시.
    입력은 절대 고치지 않는다(copy-on-write). 뗄 게 없으면 **같은 객체**를 돌려준다.
    """
    if not isinstance(payload, Mapping):
        return payload
    out = _strip_keys(payload, PROMPT_KEYS)
    scripts = payload.get("alwayson_scripts")
    if isinstance(scripts, Mapping):
        new_scripts = None
        for title, block in scripts.items():
            folded = str(title).strip().casefold()
            if folded == _ADETAILER_TITLE:
                new_block = _strip_block(block, ADETAILER_PROMPT_KEYS, first_only=False)
            elif folded == _SAM3_TITLE:
                new_block = _strip_block(block, SAM3_PROMPT_KEYS, first_only=True)
            else:
                continue
            if new_block is not block:
                if new_scripts is None:
                    new_scripts = dict(scripts)
                new_scripts[title] = new_block
        if new_scripts is not None:
            if out is payload:
                out = dict(payload)
            out["alwayson_scripts"] = new_scripts
    return out
