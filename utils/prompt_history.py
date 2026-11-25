# utils/prompt_history.py
"""최근 프롬프트 히스토리 관리"""
import logging
import os

from utils.atomic_json import atomic_write_json, load_json_safe

_logger = logging.getLogger(__name__)

_HISTORY_FILE = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'user_data', 'prompt_history.json')
_MAX_HISTORY = 50


def _load() -> list[dict]:
    """히스토리 파일 로드"""
    return load_json_safe(_HISTORY_FILE, [])


def _save(history: list[dict]):
    """히스토리 파일 저장"""
    try:
        atomic_write_json(_HISTORY_FILE, history[:_MAX_HISTORY])
    except Exception as exc:
        _logger.warning("프롬프트 히스토리 저장 실패: %s", exc)


def add_entry(prompt: str, negative: str):
    """새 프롬프트 기록 추가 (중복 시 맨 앞으로 이동)"""
    prompt = prompt.strip()
    if not prompt:
        return
    history = _load()
    entry = {"prompt": prompt, "negative": negative.strip()}
    # 동일 프롬프트 제거
    history = [h for h in history if h.get("prompt") != prompt]
    history.insert(0, entry)
    _save(history)


def get_history() -> list[dict]:
    """히스토리 목록 반환 (최신순)"""
    return _load()


def clear_history():
    """히스토리 전체 삭제"""
    _save([])
