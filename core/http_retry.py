# core/http_retry.py
"""트랜지언트 HTTP 실패에 대한 exponential backoff 재시도.

- 5xx / 연결오류 / 타임아웃만 재시도 (4xx는 즉시 반환 — 호출자가 status 를 본다)
- 지수 백오프(backoff_factor ** attempt) + 최대 ``jitter``초(기본 0.2초)의 무작위 지연
- requests 세션은 호출자가 관리 (여기선 stateless 호출만 지원)
"""
from __future__ import annotations

import logging
import random
import time
from typing import Callable

import requests

logger = logging.getLogger(__name__)

# 재시도 대상: 네트워크 단절 / 타임아웃
_RETRY_EXCEPTIONS = (
    requests.exceptions.ConnectionError,
    requests.exceptions.Timeout,
    requests.exceptions.ChunkedEncodingError,
)


def request_with_retry(
    method: str,
    url: str,
    *,
    retries: int = 3,
    backoff_factor: float = 2.0,
    retry_on_status: tuple[int, ...] = (500, 502, 503, 504),
    jitter: float = 0.2,
    **kwargs,
) -> requests.Response:
    """requests.request 래퍼.

    retries: 총 재시도 횟수 (성공 포함 최대 retries+1회 시도)
    backoff_factor: delay = backoff_factor ** attempt + random<jitter
    retry_on_status: 이 상태코드만 재시도 (5xx 기본)
    """
    last_err: Exception | None = None
    for attempt in range(retries + 1):
        try:
            resp = requests.request(method, url, **kwargs)
            if resp.status_code in retry_on_status and attempt < retries:
                wait = backoff_factor ** attempt + random.random() * jitter
                logger.warning(
                    "HTTP %s %s → %d, retrying in %.1fs (attempt %d/%d)",
                    method, url, resp.status_code, wait, attempt + 1, retries,
                )
                time.sleep(wait)
                continue
            return resp
        except _RETRY_EXCEPTIONS as e:
            last_err = e
            if attempt >= retries:
                break
            wait = backoff_factor ** attempt + random.random() * jitter
            logger.warning(
                "HTTP %s %s failed (%s), retrying in %.1fs (attempt %d/%d)",
                method, url, type(e).__name__, wait, attempt + 1, retries,
            )
            time.sleep(wait)
    # 재시도 소진
    assert last_err is not None
    raise last_err


def get_with_retry(url: str, **kwargs) -> requests.Response:
    return request_with_retry("GET", url, **kwargs)
