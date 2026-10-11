# utils/prompt_cleaner.py
"""
프롬프트 자동 정리
- 쉼표/공백 정리
- 괄호 이스케이프
- 중복 제거
- 밑줄 처리
"""
import re
from typing import Tuple, List, Optional

from core.instant_wildcards import INSTANT_WILDCARD_PATTERN
from core.prompt_comments import comment_spans_through_eol
from utils.file_wildcard import DUNDER_WILDCARD_PATTERN, FILE_WILDCARD_PATTERN

# 와일드카드 토큰(__이름__ · ~/이름/~ · 즉석 $$이름$$)은 정리 규칙을 건너뛴다 — 밑줄→공백이
# '__hairstyle__' 을 'hairstyle' 로, '~/hair_style/~' 을 '~/hair style/~' 로, '$$hair_style$$' 을
# '$$hair style$$' 로 바꿔 해석기가 못 찾게 만들었다 (Vue '사용' 버튼이 넣은 토큰이 실시간 정리
# 0.5초 뒤 평문이 되어 그대로 전송됐다). 해석기(utils.file_wildcard · core.instant_wildcards)와
# 같은 정규식으로 찾아 자리표시자로 가렸다가 되돌린다.
_WC_OPEN = ''
_WC_CLOSE = ''
_WC_SLOT = re.compile(_WC_OPEN + r'(\d+)' + _WC_CLOSE)
# 주석 자리표시자는 와일드카드와 다른 글자다 — 태그 단위 규칙(밑줄 보존 목록·중복 제거·빈 괄호·이스케이프)이
# 앞뒤 태그에 붙은 주석 자리표시자만 골라 빼고 비교할 수 있게. 원문 목록(saved)은 모두 함께 쓴다.
# 블록 주석과 줄 주석(닫히지 않은 /* 포함)은 글자가 다르다 — 줄 주석은 태그를 끝낸다(core/prompt_comments:
# 보내는 글자에서 줄 주석 왼쪽 내용 뒤에 쉼표가 붙는다). 그래서 태그 단위 규칙도 줄 주석 자리에서 태그를 나눈다.
_CM_OPEN = '\ue002'
_CM_CLOSE = '\ue003'
_LINE_OPEN = '\ue004'
_LINE_CLOSE = '\ue005'
_CM_SLOT = re.compile('[' + _CM_OPEN + _LINE_OPEN + r']\d+[' + _CM_CLOSE + _LINE_CLOSE + ']')
_LINE_SLOT = re.compile(_LINE_OPEN + r'\d+' + _LINE_CLOSE)
_ANY_SLOT = re.compile('[' + _WC_OPEN + _CM_OPEN + _LINE_OPEN + r'](\d+)['
                       + _WC_CLOSE + _CM_CLOSE + _LINE_CLOSE + ']')
_MARKS = (_WC_OPEN, _WC_CLOSE, _CM_OPEN, _CM_CLOSE, _LINE_OPEN, _LINE_CLOSE)
_CM_LEAD = re.compile(r'(?:\s*' + _CM_SLOT.pattern + r')+\s*')
_CM_TRAIL = re.compile(r'\s*(?:' + _CM_SLOT.pattern + r'\s*)+\Z')
# 중복 제거가 나누는 자리 — 쉼표와 줄 주석 자리표시자(나눈 자리도 결과에 남긴다)
_DEDUPE_SPLIT = re.compile('(,|' + _LINE_SLOT.pattern + ')')
# 밑줄 뒤에 // · /* 가 오면 밑줄→공백이 새 주석 시작을 만든다(Danbooru 'z.a.t.o._//_i_love_…')
_UNDERSCORE_OPENS_COMMENT = re.compile(r'(?<!\\)_(?=/[/*])')
# 빈 괄호 — 안에 주석 자리표시자만 있어도 빈 괄호다(주석은 남긴다). 바로 뒤에 // · /* 가 오면 두는데,
# 지우면 그 자리가 새 주석 시작이 될 수 있어서다('()// x' → '// x').
_EMPTY_PARENS = tuple(
    re.compile(re.escape(o) + r'(?:\s|' + _CM_SLOT.pattern + r')*,?(?:\s|' + _CM_SLOT.pattern + r')*'
               + re.escape(c) + r'(?!/[/*])')
    for o, c in ('()', '[]', '{}'))


def _comment_free(tag: str) -> str:
    """태그에서 주석 자리표시자를 뺀 비교용 글자(앞뒤 공백 정리). 주석이 없으면 ``strip()`` 그대로."""
    if _CM_OPEN not in tag and _LINE_OPEN not in tag:
        return tag.strip()
    return ' '.join(_CM_SLOT.sub(' ', tag).split())


def _split_comment_edges(tag: str) -> Tuple[str, str, str]:
    """``(앞 주석, 가운데, 뒤 주석)`` — 앞뒤 주석 자리표시자(와 둘레 공백)를 떼어 낸다. 없으면 ``''``."""
    lead = _CM_LEAD.match(tag)
    head = lead.group(0) if lead else ''
    rest = tag[len(head):]
    trail = _CM_TRAIL.search(rest) if rest else None
    tail = trail.group(0) if trail else ''
    return head, rest[:len(rest) - len(tail)], tail


def _split_at_line_comments(tag: str) -> List[str]:
    """괄호 밖 줄 주석 자리표시자에서 태그를 나눈다 → ``[글, 자리표시자, 글, …]`` (짝수 칸이 글).

    괄호 깊이는 :meth:`PromptCleaner._split_by_top_level_commas` 와 같게 센다(``\\(`` 는 세지 않는다).
    """
    if _LINE_OPEN not in tag:
        return [tag]
    out: List[str] = []
    depth = 0
    start = i = 0
    n = len(tag)
    while i < n:
        ch = tag[i]
        if ch == '\\' and i + 1 < n and tag[i + 1] in '()':
            i += 2
            continue
        if ch == '(':
            depth += 1
        elif ch == ')':
            depth -= 1
        elif ch == _LINE_OPEN and depth <= 0:
            m = _LINE_SLOT.match(tag, i)
            if m:
                out.append(tag[start:i])
                out.append(m.group(0))
                start = i = m.end()
                continue
        i += 1
    out.append(tag[start:])
    return out


def _map_tag_pieces(tag: str, fn) -> str:
    """줄 주석 자리에서 나눈 글 조각마다 ``fn`` 을 쓰고 자리표시자는 그대로 둔 채 다시 잇는다."""
    pieces = _split_at_line_comments(tag)
    if len(pieces) == 1:
        return fn(tag)

    def keep_trailing_space(piece: str) -> str:
        # 조각 끝 공백은 그대로 — 줄 주석 앞 공백이 사라지면 '//' 가 더는 주석이 아니다
        core = piece.rstrip()
        return fn(core) + piece[len(core):] if core else piece

    return ''.join(piece if k % 2 else keep_trailing_space(piece) for k, piece in enumerate(pieces))


def _mask_comment_spans(text: str, saved: List[str]) -> str:
    """프롬프트 주석(// · /* */, core/prompt_comments)을 자리표시자로 가린다 — 원문은 ``saved`` 에 쌓는다.

    정리 규칙(줄바꿈→공백·밑줄→공백·이스케이프·중복 제거)이 주석에 닿으면 안 된다: 줄바꿈이 공백이 되면
    다음 줄이 줄 주석 안으로 들어가 조용히 사라진다. 그래서 줄 주석·닫히지 않은 블록의 자리표시자는 뒤따르는
    줄 끝까지 품고, 줄 처음에서 시작하면 앞 줄 끝도 품는다(주석 줄이 앞 줄에 붙지 않게). 주석은 **나올 때마다
    다른** 자리표시자를 쓴다 — 중복 제거가 같은 글자의 두 번째 주석을 지우지 않게. 같은 줄의 내용 뒤에서 줄을
    끝내는 줄 주석(core/prompt_comments 가 쉼표를 두는 자리)만 줄 주석 자리표시자(태그 경계)이고, 주석만 있는
    줄·블록 주석은 블록 자리표시자다.
    """
    spans = comment_spans_through_eol(text)
    if not spans:
        return text
    out: List[str] = []
    pos = 0
    line_has_content = False   # 지금 줄에 (주석이 아닌) 내용이 있었나
    for start, end, kind in spans:
        gap = text[pos:start]
        cut = max(gap.rfind('\n'), gap.rfind('\r'))
        if cut >= 0:
            line_has_content = bool(gap[cut + 1:].strip())
        elif gap.strip():
            line_has_content = True
        if kind != 'block' and start > pos and text[start - 1] in '\r\n':
            start -= 2 if start - 2 >= pos and text.startswith('\r\n', start - 2) else 1
        out.append(text[pos:start])
        saved.append(text[start:end])
        if kind != 'block' and line_has_content:
            out.append(f'{_LINE_OPEN}{len(saved) - 1}{_LINE_CLOSE}')
        else:
            out.append(f'{_CM_OPEN}{len(saved) - 1}{_CM_CLOSE}')
        if kind != 'block':
            line_has_content = False   # 줄 주석 자리표시자는 줄 끝까지 품는다
        pos = end
    out.append(text[pos:])
    return ''.join(out)


def _mask_wildcard_tokens(text: str) -> Tuple[str, List[str]]:
    """프롬프트 주석과 와일드카드 토큰을 자리표시자로 바꾼다 → (가린 텍스트, 원문 목록).

    주석을 먼저 가린다(:func:`_mask_comment_spans` — 주석 안의 ``__x__`` 는 주석의 일부다).
    같은 와일드카드 토큰은 같은 자리표시자를 쓴다(중복 제거가 같은 글자를 같은 태그로 보던 규칙 유지).
    자리표시자 문자가 이미 들어 있는 드문 입력은 가리지 않는다(되돌릴 때 섞이지 않게).
    """
    if not text or any(mark in text for mark in _MARKS):
        return text, []
    saved: List[str] = []
    text = _mask_comment_spans(text, saved)

    def _keep(m):
        token = m.group(0)
        if token not in saved:
            saved.append(token)
        return f'{_WC_OPEN}{saved.index(token)}{_WC_CLOSE}'

    # 즉석 $$이름$$ 을 먼저 통째로 가린다 — '$$a__b__c$$' 안의 '__b__' 를 파일 문법이 반쪽만
    # 가리지 않게. 그다음 해석 순서(~/이름/~ 먼저)와 같게 — 안쪽에 먼저 가린 자리표시자가 있으면
    # 되돌릴 때 풀린다.
    text = INSTANT_WILDCARD_PATTERN.sub(_keep, text)
    text = FILE_WILDCARD_PATTERN.sub(_keep, text)
    text = DUNDER_WILDCARD_PATTERN.sub(_keep, text)
    return text, saved


def _unmask_wildcard_tokens(text: str, saved: List[str]) -> str:
    """_mask_wildcard_tokens 의 반대 — 중첩된 자리표시자까지 원문으로 되돌린다."""
    if not saved:
        return text
    for _ in range(len(saved) + 1):
        restored = _ANY_SLOT.sub(lambda m: saved[int(m.group(1))], text)
        if restored == text:
            break
        text = restored
    return text


class PromptCleaner:
    """프롬프트 자동 정리기"""
    
    def __init__(self):
        # 설정 (기본값)
        self.auto_comma = True           # 쉼표 정리
        self.auto_space = True           # 공백 정리
        self.auto_escape = False         # 괄호 이스케이프
        self.remove_duplicates = False   # 중복 제거
        self.underscore_to_space = True  # 밑줄 → 공백
        self.remove_empty_parens = True  # 빈 괄호 제거
        self.trim_trailing_comma = True  # 마지막 쉼표 제거
        # 밑줄을 보존할 태그/패턴 (정확 일치 또는 패턴 매치)
        # 예: score_9, score_8 등이 score 9, score 8로 안 바뀌게
        self._underscore_preserve_load()
    
    def clean(self, text: str) -> str:
        """전체 정리 실행"""
        if not text:
            return ""

        # 프롬프트 주석(// · /* */)과 와일드카드 토큰(__이름__ · ~/이름/~ · $$이름$$)은 글자 그대로 둔다 — 끝에서 되돌린다
        result, wildcard_tokens = _mask_wildcard_tokens(text)

        # 1. 공백 정리
        if self.auto_space:
            result = self._clean_spaces(result)
        
        # 2. 쉼표 정리
        if self.auto_comma:
            result = self._clean_commas(result)
        
        # 3. 빈 괄호 제거
        if self.remove_empty_parens:
            result = self._remove_empty_parentheses(result)
        
        # 4. 밑줄 → 공백
        if self.underscore_to_space:
            result = self._convert_underscores(result)
        
        # 5. 중복 제거
        if self.remove_duplicates:
            result = self._remove_duplicate_tags(result)
        
        # 6. 괄호 이스케이프
        if self.auto_escape:
            result = self._escape_parentheses(result)
        
        # 7. 마지막 쉼표 제거
        if self.trim_trailing_comma:
            result = result.rstrip().rstrip(',').strip()

        return _unmask_wildcard_tokens(result, wildcard_tokens)
    
    def _clean_spaces(self, text: str) -> str:
        """공백 정리"""
        # 여러 공백 → 단일 공백
        text = re.sub(r' +', ' ', text)
        # 탭, 줄바꿈 → 공백
        text = re.sub(r'[\t\n\r]+', ' ', text)
        return text.strip()
    
    def _clean_commas(self, text: str) -> str:
        """쉼표 정리"""
        # 쉼표 앞 공백 제거
        text = re.sub(r'\s+,', ',', text)
        # 쉼표 뒤 공백 통일
        text = re.sub(r',\s*', ', ', text)
        # 연속 쉼표 제거
        text = re.sub(r',(\s*,)+', ',', text)
        return text
    
    def _remove_empty_parentheses(self, text: str) -> str:
        """빈 괄호 제거"""
        # (), ( ), (,), ( , ) 등 제거 — 안에 주석만 있으면 괄호만 지우고 주석은 남긴다
        def _keep_comments(m):
            # 앞에 공백을 둔다 — 주석은 공백·쉼표 뒤에서만 시작하므로 앞 글자에 붙으면 평문이 된다
            return ''.join(' ' + slot.group(0) for slot in _CM_SLOT.finditer(m.group(0)))
        for pattern in _EMPTY_PARENS:
            text = pattern.sub(_keep_comments, text)
        # 정리 후 연속 공백/쉼표 처리
        text = re.sub(r',\s*,', ',', text)
        text = re.sub(r'\s+', ' ', text)
        return text
    
    def _convert_underscores(self, text: str) -> str:
        """밑줄을 공백으로 변환 (태그 내부만).

        예외 리스트 (``underscore_preserve_set``)에 정확히 일치하는 태그는
        밑줄을 유지한다. 예: ``score_9``, ``score_8`` → 그대로 보존.

        이스케이프된 밑줄(``\\_``)은 항상 ``_``로 환원 (기존 동작 유지).
        """
        # 태그 단위로 split → 각 태그가 예외 리스트인지 확인 → 변환 결정. 줄 주석도 태그를 끝낸다.
        preserve = self._get_underscore_preserve_set()
        tags = self._split_by_top_level_commas(text)
        return ','.join(_map_tag_pieces(raw_tag, lambda piece: self._convert_tag_underscores(piece, preserve))
                        for raw_tag in tags)

    @staticmethod
    def _convert_tag_underscores(raw_tag: str, preserve) -> str:
        """태그 하나의 밑줄 → 공백(예외 목록·주석을 여는 태그는 그대로, ``\\_`` 는 ``_`` 로)."""
        # 정확 일치 (대소문자 무시) — 앞뒤에 붙은 주석 자리표시자는 빼고 비교
        if _comment_free(raw_tag).lower() in preserve:
            return raw_tag  # 원본 그대로 (밑줄 유지)
        # '_//' · '_/*' — 밑줄이 공백이 되면 뒤가 주석이 되어 태그가 잘린다 → 그 태그는 그대로
        if _UNDERSCORE_OPENS_COMMENT.search(raw_tag):
            return raw_tag
        # 일반 변환 (이스케이프 처리 포함)
        converted = []
        i = 0
        while i < len(raw_tag):
            if raw_tag[i] == '\\' and i + 1 < len(raw_tag) and raw_tag[i + 1] == '_':
                converted.append('_')
                i += 2
            elif raw_tag[i] == '_':
                converted.append(' ')
                i += 1
            else:
                converted.append(raw_tag[i])
                i += 1
        return ''.join(converted)

    def _underscore_preserve_load(self) -> None:
        """기본값 + ``config/underscore_preserve.txt`` 로딩 (한 줄당 한 태그).

        파일이 없거나 비어있으면 기본값만 사용. ``#``로 시작하는 줄은 주석.
        """
        # 기본 — Pony/SDXL score 태그 + 일부 자주 쓰이는 underscore 보존 태그
        self.underscore_preserve_set: set[str] = {
            "score_9", "score_8", "score_7", "score_6", "score_5", "score_4",
            "score_3", "score_2", "score_1", "score_0",
            "score_9_up", "score_8_up", "score_7_up", "score_6_up", "score_5_up",
            "rating_safe", "rating_questionable", "rating_explicit", "rating_general",
            "source_anime", "source_furry", "source_cartoon", "source_pony",
        }
        # config 파일 머지
        try:
            import os
            from pathlib import Path
            project_root = Path(__file__).resolve().parent.parent
            cfg = project_root / "config" / "underscore_preserve.txt"
            if cfg.is_file():
                for line in cfg.read_text(encoding="utf-8").splitlines():
                    s = line.strip()
                    if not s or s.startswith("#"):
                        continue
                    self.underscore_preserve_set.add(s.lower())
        except Exception:
            pass  # 보수적 — 파일 파싱 실패해도 기본값으로 동작

    def _get_underscore_preserve_set(self) -> set[str]:
        """예외 리스트 조회 — 외부 변경에 안전한 사본 반환."""
        if not hasattr(self, "underscore_preserve_set"):
            self._underscore_preserve_load()
        return self.underscore_preserve_set
    
    def _remove_duplicate_tags(self, text: str) -> str:
        """중복 태그 제거.

        주석이 있으면 쉼표와 줄 주석 자리에서 태그를 나누고(줄 주석은 태그 경계), 주석 자리표시자를 뺀 글자로
        비교한다(``1girl, // 메모\\n1girl`` 의 두 번째 1girl 도 중복). 겹쳐 지운 태그의 주석은 버리지 않는다.
        """
        if _CM_OPEN not in text and _LINE_OPEN not in text:
            tags = [t.strip() for t in text.split(',')]
            seen = set()
            unique_tags = []

            for tag in tags:
                tag_lower = tag.lower()
                if tag_lower and tag_lower not in seen:
                    seen.add(tag_lower)
                    unique_tags.append(tag)

            return ', '.join(unique_tags)

        parts = _DEDUPE_SPLIT.split(text)
        seen = set()
        out = ''
        carry = ''   # 겹쳐 지운 태그에 붙어 있던 주석 자리표시자 — 다음 태그 앞에 붙인다
        for k in range(0, len(parts), 2):
            tag = parts[k]
            sep = parts[k + 1] if k + 1 < len(parts) else None
            key = _comment_free(tag).lower()
            kept = tag.strip()
            if key and key in seen:
                slots = ' '.join(slot.group(0) for slot in _CM_SLOT.finditer(tag))
                if sep in (',', None):
                    carry += slots + ' ' if slots else ''
                    continue
                kept = slots
            elif key:
                seen.add(key)
            if not kept and sep in (',', None):
                continue   # 빈 태그(와 그 쉼표)는 버린다 — 주석 없는 글과 같은 규칙
            if carry:
                kept = carry + kept if kept else carry.rstrip()
                carry = ''
            out += kept
            if sep == ',':
                out += ', '
            elif sep:   # 줄 주석 앞 공백은 원문에 있던 만큼만(없던 공백을 넣지 않는다)
                out += (' ' if kept and tag[-1:].isspace() else '') + sep
        out = out[:-2] if out.endswith(', ') else out
        if carry:
            out = f'{out} {carry.rstrip()}' if out else carry.rstrip()
        return out

    def _escape_parentheses(self, text: str) -> str:
        """
        괄호 이스케이프 (쉼표 구분 태그 단위로 판단):
        - 태그 전체가 (...)로 감싸진 경우 → SD 가중치/강조 → 바깥 유지, 안쪽 중첩만 이스케이프
          예) (tag:1.2) → (tag:1.2)
              (a (b)) → (a \(b\))
        - 태그 일부에 (...)가 있는 경우 → 수식어 괄호 → 전부 이스케이프
          예) ike (fire emblem) → ike \(fire emblem\)
              kafka (honkai: star rail) (cosplay) → kafka \(honkai: star rail\) \(cosplay\)
        """
        tags = self._split_by_top_level_commas(text)
        escaped = [_map_tag_pieces(t, self._escape_tag) for t in tags]   # 줄 주석도 태그를 끝낸다
        return ','.join(escaped)

    def _escape_tag(self, tag: str) -> str:
        """개별 태그의 괄호 이스케이프"""
        stripped = tag.strip()
        if not stripped:
            return tag
        head, core, tail = _split_comment_edges(tag)
        if head or tail:   # 앞뒤에 붙은 주석 자리표시자는 떼고 가운데만 판단
            return head + self._escape_tag(core) + tail

        # 태그 전체가 (...)로 감싸진 경우 → SD 구문, 바깥 유지
        if stripped.startswith('(') and stripped.endswith(')'):
            close_idx = self._find_matching_close_paren(stripped, 0)
            if close_idx == len(stripped) - 1:
                inner = stripped[1:-1]
                escaped_inner = self._escape_inner_parens(inner)
                lead = tag[:len(tag) - len(tag.lstrip())]
                return f'{lead}({escaped_inner})'

        # 태그 일부에 괄호 → 수식어, 전부 이스케이프
        return self._escape_inner_parens(tag)

    def _split_by_top_level_commas(self, text: str) -> list:
        """괄호 깊이를 고려하여 최상위 쉼표로 분리"""
        parts = []
        current = []
        depth = 0
        i = 0
        while i < len(text):
            if text[i] == '\\' and i + 1 < len(text) and text[i + 1] in '()':
                current.append(text[i:i + 2])
                i += 2
                continue
            if text[i] == '(':
                depth += 1
            elif text[i] == ')':
                depth -= 1
            if text[i] == ',' and depth <= 0:
                parts.append(''.join(current))
                current = []
                depth = 0
            else:
                current.append(text[i])
            i += 1
        parts.append(''.join(current))
        return parts

    def _escape_inner_parens(self, text: str) -> str:
        """안쪽 내용의 모든 괄호를 이스케이프 (이미 이스케이프된 건 건너뛰기)"""
        result = []
        i = 0
        while i < len(text):
            if text[i] == '\\' and i + 1 < len(text) and text[i + 1] in '()':
                result.append(text[i:i + 2])
                i += 2
            elif text[i] == '(':
                result.append(r'\(')
                i += 1
            elif text[i] == ')':
                result.append(r'\)')
                i += 1
            else:
                result.append(text[i])
                i += 1
        return ''.join(result)

    @staticmethod
    def _find_matching_close_paren(text: str, open_idx: int) -> int:
        """깊이 기반으로 대응하는 닫는 괄호 위치를 찾는다"""
        depth = 0
        i = open_idx
        while i < len(text):
            if text[i] == '\\' and i + 1 < len(text) and text[i + 1] in '()':
                i += 2  # 이스케이프된 괄호 건너뛰기
                continue
            if text[i] == '(':
                depth += 1
            elif text[i] == ')':
                depth -= 1
                if depth == 0:
                    return i
            i += 1
        return -1

    def set_options(self, **kwargs):
        """옵션 설정"""
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)


# 싱글톤
_cleaner_instance = None

def get_prompt_cleaner() -> PromptCleaner:
    global _cleaner_instance
    if _cleaner_instance is None:
        _cleaner_instance = PromptCleaner()
    return _cleaner_instance


def escape_parentheses(text: str) -> str:
    """간편 함수: 괄호 이스케이프 (싱글턴 재사용 — 호출마다 PromptCleaner 생성/디스크 IO 제거)"""
    return get_prompt_cleaner()._escape_parentheses(text)
