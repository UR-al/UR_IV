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
from utils.file_wildcard import DUNDER_WILDCARD_PATTERN, FILE_WILDCARD_PATTERN

# 와일드카드 토큰(__이름__ · ~/이름/~ · 즉석 $$이름$$)은 정리 규칙을 건너뛴다 — 밑줄→공백이
# '__hairstyle__' 을 'hairstyle' 로, '~/hair_style/~' 을 '~/hair style/~' 로, '$$hair_style$$' 을
# '$$hair style$$' 로 바꿔 해석기가 못 찾게 만들었다 (Vue '사용' 버튼이 넣은 토큰이 실시간 정리
# 0.5초 뒤 평문이 되어 그대로 전송됐다). 해석기(utils.file_wildcard · core.instant_wildcards)와
# 같은 정규식으로 찾아 자리표시자로 가렸다가 되돌린다.
_WC_OPEN = ''
_WC_CLOSE = ''
_WC_SLOT = re.compile(_WC_OPEN + r'(\d+)' + _WC_CLOSE)


def _mask_wildcard_tokens(text: str) -> Tuple[str, List[str]]:
    """와일드카드 토큰을 자리표시자로 바꾼다 → (가린 텍스트, 원문 토큰 목록).

    같은 토큰은 같은 자리표시자를 쓴다(중복 제거가 같은 글자를 같은 태그로 보던 규칙 유지).
    자리표시자 문자가 이미 들어 있는 드문 입력은 가리지 않는다(되돌릴 때 섞이지 않게).
    """
    if not text or _WC_OPEN in text or _WC_CLOSE in text:
        return text, []
    saved: List[str] = []

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
        restored = _WC_SLOT.sub(lambda m: saved[int(m.group(1))], text)
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

        # 와일드카드 토큰(__이름__ · ~/이름/~ · $$이름$$)은 글자 그대로 둔다 — 끝에서 되돌린다
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
        # (), ( ), (,), ( , ) 등 제거
        text = re.sub(r'\(\s*,?\s*\)', '', text)
        text = re.sub(r'\[\s*,?\s*\]', '', text)
        text = re.sub(r'\{\s*,?\s*\}', '', text)
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
        # 태그 단위로 split → 각 태그가 예외 리스트인지 확인 → 변환 결정
        preserve = self._get_underscore_preserve_set()
        tags = self._split_by_top_level_commas(text) if hasattr(self, '_split_by_top_level_commas') else text.split(',')
        out_tags: list[str] = []
        for raw_tag in tags:
            stripped = raw_tag.strip()
            # 정확 일치 (대소문자 무시)
            if stripped.lower() in preserve:
                out_tags.append(raw_tag)  # 원본 그대로 (밑줄 유지)
                continue
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
            out_tags.append(''.join(converted))
        return ','.join(out_tags)

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
        """중복 태그 제거"""
        tags = [t.strip() for t in text.split(',')]
        seen = set()
        unique_tags = []
        
        for tag in tags:
            tag_lower = tag.lower()
            if tag_lower and tag_lower not in seen:
                seen.add(tag_lower)
                unique_tags.append(tag)
        
        return ', '.join(unique_tags)
    
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
        escaped = [self._escape_tag(t) for t in tags]
        return ','.join(escaped)

    def _escape_tag(self, tag: str) -> str:
        """개별 태그의 괄호 이스케이프"""
        stripped = tag.strip()
        if not stripped:
            return tag

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
