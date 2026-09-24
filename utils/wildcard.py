# utils/wildcard.py
"""
와일드카드 시스템
- 기본: {A|B|C} → 랜덤 선택
- 가중치: {A:3|B:1|C:1} → A가 60% 확률
- 중첩: {{red|blue} hair|ponytail}
- 범위: {1-10} → 1~10 중 랜덤 숫자
"""
import re
import random


class WildcardProcessor:
    """와일드카드 처리기 — 외부 진입점은 :func:`process_wildcards` 하나다."""

    # 와일드카드 패턴: {내용}
    PATTERN = re.compile(r'\{([^{}]+)\}')

    # 범위 패턴: {1-10} 또는 {1-10:2} (step 포함)
    RANGE_PATTERN = re.compile(r'^(\d+)-(\d+)(?::(\d+))?$')

    def process(self, text: str) -> str:
        """
        와일드카드 처리 (랜덤 선택)
        중첩된 와일드카드도 처리
        """
        if not text:
            return text

        # 중첩 처리를 위해 반복
        prev_text = None
        while prev_text != text:
            prev_text = text
            text = self.PATTERN.sub(self._replace_wildcard, text)

        return text

    def _replace_wildcard(self, match) -> str:
        """단일 와일드카드 교체"""
        content = match.group(1)

        # 범위 체크 {1-10}
        range_match = self.RANGE_PATTERN.match(content.strip())
        if range_match:
            return self._process_range(range_match)

        # 일반 옵션 {A|B|C} 또는 {A:3|B:1}
        return self._process_options(content)

    def _process_range(self, match) -> str:
        """범위 처리: {1-10} → 랜덤 숫자. 스텝 0 은 1 로 본다(range 가 ValueError 를 낸다)."""
        start = int(match.group(1))
        end = int(match.group(2))
        step = int(match.group(3)) if match.group(3) else 1
        if step <= 0:
            step = 1

        if start > end:
            start, end = end, start

        return str(random.choice(range(start, end + 1, step)))

    def _process_options(self, content: str) -> str:
        """옵션 처리: {A|B|C} 또는 {A:3|B:1}

        가중치가 음수면 0 으로 자른다. 가중치 합이 0 이하(예: {a:0|b:0})이면 random.choices 가
        ValueError 로 생성 준비를 통째로 멈추므로, 그때는 후보를 균등하게 고른다.
        """
        parts = content.split('|')

        options = []
        weights = []

        for part in parts:
            part = part.strip()
            if not part:
                continue

            # 가중치 확인 {A:3|B:1}
            if ':' in part:
                # 마지막 : 이후가 숫자인지 확인 ('artist:foo' 는 가중치가 아니다)
                last_colon = part.rfind(':')
                weight_str = part[last_colon + 1:]

                try:
                    weight = int(weight_str)
                    option = part[:last_colon].strip()
                    options.append(option)
                    weights.append(max(0, weight))
                except ValueError:
                    # 숫자가 아니면 그냥 옵션으로
                    options.append(part)
                    weights.append(1)
            else:
                options.append(part)
                weights.append(1)

        if not options:
            return ''

        if sum(weights) <= 0:
            return random.choice(options)
        return random.choices(options, weights=weights, k=1)[0]


# 싱글톤 인스턴스
_processor_instance = None

def get_wildcard_processor() -> WildcardProcessor:
    """싱글톤 인스턴스 반환"""
    global _processor_instance
    if _processor_instance is None:
        _processor_instance = WildcardProcessor()
    return _processor_instance


def process_wildcards(text: str) -> str:
    """간편 함수: 와일드카드 처리"""
    return get_wildcard_processor().process(text)
