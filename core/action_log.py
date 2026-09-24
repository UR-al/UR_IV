# core/action_log.py
"""Vue 액션 로그용 페이로드 요약 — 페이로드 전체를 직렬화하지 않는다.

update_prompt_deck(수만 행)·export_search_results·chat_save(base64 이미지) 같은 큰 페이로드를
로그 앞 100자 때문에 json.dumps 하면 GUI 스레드에서 수십~수백 ms 와 같은 크기의 임시
메모리를 쓴다. 여기서는 키 이름과 컨테이너 크기만 본다(O(키 수)).

로그 줄은 항상 ASCII 다(비ASCII 는 ``\\xe9``/``\\uc800`` 식 escape). 출력이 cp949 파이프·파일로
리다이렉트되면(로그 캡처·CDP 검증·스크립트 실행) 'é'·이모지 한 글자로 print 가
UnicodeEncodeError 를 내고, 그 예외가 onAction 의 except 에 먹혀 액션이 조용히 사라졌다.
예전 ``json.dumps`` 로그는 ensure_ascii 라 이 문제가 없었다.

순수 로직 — Qt 의존 없음.
"""
from __future__ import annotations

from typing import Any, TextIO

_MAX_KEYS = 8
_MAX_SCALAR = 40


def _to_ascii(text: str) -> str:
    return text.encode('ascii', 'backslashreplace').decode('ascii')


def _describe_value(value: Any) -> str:
    if isinstance(value, (list, tuple)):
        return f'[{len(value)}]'
    if isinstance(value, dict):
        return f'{{{len(value)}}}'
    if isinstance(value, str):
        if len(value) > _MAX_SCALAR:
            return f'str({len(value)})'
        return ascii(value)
    if value is None or isinstance(value, (bool, int, float)):
        return ascii(value)
    return _to_ascii(type(value).__name__)


def summarize_action_payload(payload: Any, limit: int = 160) -> str:
    """로그 한 줄용 요약(ASCII). dict 가 아닌 payload(None 포함)에서도 예외를 내지 않는다."""
    if isinstance(payload, dict):
        keys = list(payload)
        parts = [f'{key}={_describe_value(payload[key])}' for key in keys[:_MAX_KEYS]]
        if len(keys) > _MAX_KEYS:
            parts.append(f'+{len(keys) - _MAX_KEYS} keys')
        text = '{' + ', '.join(parts) + '}'
    else:
        text = _describe_value(payload)
    text = _to_ascii(text)   # 키 이름 등 남은 비ASCII 도 escape
    if len(text) > limit:
        text = text[: max(0, limit - 3)] + '...'
    return text


def format_action_log(action: Any, payload: Any) -> str:
    """``_handle_vue_action`` 첫 줄 로그(ASCII)."""
    name = _to_ascii(action) if isinstance(action, str) else ascii(action)
    return f'[Bridge] Action Received: {name} | Payload: {summarize_action_payload(payload)}'


def log_action(action: Any, payload: Any, *, stream: TextIO | None = None) -> None:
    """액션 로그를 찍는다. 로그 실패(닫힌 stdout·깨진 파이프·인코딩)가 액션을 막지 않는다.

    ``stream`` 이 None 이면 ``print`` 기본값(sys.stdout — pythonw 에선 None 이라 무출력).
    """
    try:
        print(format_action_log(action, payload), file=stream)
    except (OSError, ValueError):   # UnicodeEncodeError 는 ValueError 의 하위 클래스
        pass


__all__ = ['format_action_log', 'log_action', 'summarize_action_payload']
