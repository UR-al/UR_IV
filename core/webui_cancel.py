# core/webui_cancel.py
"""Forge/A1111 생성 취소 — '우리 작업이 실제로 도는 동안에만' 전역 interrupt (순수 로직).

A1111/Forge 는 ``POST /sdapi/v1/interrupt`` 하나뿐인 **전역** 중단만 있다. 예전에는
cancel_check 가 켜지면 HTTP 가 살아 있는 동안 0.25초마다 전역 interrupt 를 반복했다 —
우리 요청이 외부 Forge 클라이언트 작업 뒤에 줄 서 있으면 **남의 작업**을 끊었다
(ui/hand_reconstruction_actions.py 의 '외부 작업은 절대 취소하지 않는다' 정책과 충돌).
반대로 한 번만 보내면 Forge 가 ``state.begin()`` 에서 플래그를 지워 취소가 사라졌다.

그래서 요청마다 ``force_task_id`` 를 붙이고, 취소가 요청되면 ``POST /internal/progress``
(``{"id_task": ...}`` → active/queued/completed)로 **우리 작업의 상태**를 본다:

- active: ``state.begin()`` 뒤이므로 지금 보낸 interrupt 는 지워지지 않는다 → 보낸다(반복)
- queued: 남의 작업이 도는 중 → 보내지 않고 기다린다(우리 차례가 되면 active 로 바뀐다)
- completed: 이미 끝났다 → 보내지 않는다
- unseen: 아직 큐 등록 전(본문 업로드·요청 파싱 중) → 기다린다. **처음 unseen 을 본 뒤**
  유예 시간(본문 크기에 비례해 늘어남)이 지나도 한 번도 안 보이면 서버가 force_task_id 를
  모르는 옛 버전이라 보고 예전 방식(전역 interrupt)으로 되돌린다
- unknown: ``/internal/progress`` 가 없음(404/405, ``--nowebui``)·모르는 응답 모양 → 예전 방식
- transient: 조회 일시 실패(타임아웃·연결 끊김·5xx). 한 번의 실패로 전역 interrupt 를 보내면
  외부 작업을 끊는다 — 마지막으로 확인한 상태를 따르고(active 면 계속 보냄, queued/unseen/
  completed 면 기다림), 한 번도 응답을 못 받았을 때만 여러 번 연속 실패 뒤 예전 방식으로 간다
"""
from __future__ import annotations

import uuid
from typing import Any, Mapping, Optional

TASK_ACTIVE = "active"
TASK_QUEUED = "queued"
TASK_COMPLETED = "completed"
TASK_UNSEEN = "unseen"
TASK_UNKNOWN = "unknown"
TASK_TRANSIENT = "transient"

# 요청 수신 → add_task_to_queue 사이(스크립트 인자 준비·init 이미지 디코드)는 보통 1초 미만.
UNSEEN_GRACE_SECONDS = 3.0
# 큰 img2img 본문(init 이미지·마스크 base64)은 업로드와 파싱이 add_task_to_queue 전에 끝나야
# 한다 — 느린 원격 링크까지 보수적으로 2MB/s 로 잡아 유예를 늘린다(옛 서버 취소만 늦어질 뿐이다).
UNSEEN_GRACE_BYTES_PER_SECOND = 2 * 1024 * 1024
UNSEEN_GRACE_MAX_SECONDS = 30.0
# 한 번도 응답을 못 받은 채 연속으로 이만큼 실패하면 엔드포인트가 없는 것으로 보고 예전 방식.
TRANSIENT_FAILURE_LIMIT = 5


def approx_payload_bytes(payload: Any) -> int:
    """JSON 본문 크기 근사 — 문자열 길이 합(base64 이미지가 대부분). 직렬화하지 않는다."""
    total = 0
    stack = [payload]
    while stack:
        item = stack.pop()
        if isinstance(item, str):
            total += len(item)
        elif isinstance(item, (bytes, bytearray)):
            total += len(item)
        elif isinstance(item, Mapping):
            for key, value in item.items():
                total += len(str(key))
                stack.append(value)
        elif isinstance(item, (list, tuple)):
            stack.extend(item)
        else:
            total += 8
    return total


def unseen_grace_seconds(payload_bytes: int) -> float:
    """본문 크기에 맞춘 unseen 유예 — 기본 3초 + 업로드·파싱 여유, 최대 30초."""
    size = max(0, int(payload_bytes or 0))
    return min(UNSEEN_GRACE_MAX_SECONDS, UNSEEN_GRACE_SECONDS + size / UNSEEN_GRACE_BYTES_PER_SECOND)


def new_task_id() -> str:
    return f"task(aistudio-{uuid.uuid4().hex})"


def with_task_id(payload: Mapping[str, Any]) -> tuple[dict, str]:
    """``force_task_id`` 가 붙은 새 payload 와 그 id. 호출자가 준 id 가 있으면 그대로 쓴다."""
    out = dict(payload or {})
    existing = out.get("force_task_id")
    if isinstance(existing, str) and existing.strip():
        return out, existing
    task_id = new_task_id()
    out["force_task_id"] = task_id
    return out, task_id


def parse_task_state(data: Any) -> str:
    """``/internal/progress`` 응답 JSON → 작업 상태."""
    if not isinstance(data, Mapping):
        return TASK_UNKNOWN
    if data.get("active") is True:
        return TASK_ACTIVE
    if data.get("queued") is True:
        return TASK_QUEUED
    if data.get("completed") is True:
        return TASK_COMPLETED
    if all(key in data for key in ("active", "queued", "completed")):
        return TASK_UNSEEN
    return TASK_UNKNOWN


class TaskInterruptPolicy:
    """취소 요청 뒤 매 폴링마다 '지금 전역 interrupt 를 보낼지' 결정한다.

    ``now`` 는 단조 시계(time.monotonic). unseen 유예는 요청 발송 시각이 아니라 **처음 unseen 을
    본 시각**부터 잰다 — 발송 시각은 본문 업로드 전이라, 큰 요청이 파싱되는 동안 유예가 끝나
    외부 작업에 전역 interrupt 가 나갔다.
    """

    def __init__(self, grace_seconds: float = UNSEEN_GRACE_SECONDS,
                 transient_limit: int = TRANSIENT_FAILURE_LIMIT):
        self.grace_seconds = float(grace_seconds)
        self.transient_limit = max(1, int(transient_limit))
        self.seen = False
        self.last_known: Optional[str] = None
        self.unseen_since: Optional[float] = None
        self.transient_failures = 0

    def should_interrupt(self, state: str, now: float) -> bool:
        if state == TASK_TRANSIENT:
            self.transient_failures += 1
            if self.last_known == TASK_ACTIVE:
                return True    # 직전까지 우리 작업이 돌고 있었고 HTTP 도 살아 있다
            if self.last_known is not None:
                return False   # queued/unseen/completed 를 확인했다 — 한 번의 실패로 남을 끊지 않는다
            # 한 번도 응답을 못 받았다 — 여러 번 연속 실패일 때만 엔드포인트 없음으로 보고 폴백
            return self.transient_failures >= self.transient_limit
        self.transient_failures = 0
        self.last_known = state
        if state in (TASK_ACTIVE, TASK_QUEUED, TASK_COMPLETED):
            self.seen = True
        if state == TASK_ACTIVE:
            return True
        if state == TASK_UNKNOWN:
            return True   # 옛 서버/엔드포인트 없음 — 예전과 같은 전역 interrupt 로 폴백
        if state == TASK_UNSEEN:
            # 한 번 보였다가 사라졌으면(예외로 끝남) 보낼 대상이 없다
            if self.seen:
                return False
            if self.unseen_since is None:
                self.unseen_since = float(now)
            return float(now) - self.unseen_since >= self.grace_seconds
        return False       # queued / completed
