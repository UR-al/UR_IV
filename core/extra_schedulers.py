# core/extra_schedulers.py
"""Extra Schedulers (sam-extra) (sam-extra ``scripts/anima_extra_schedulers.py``) — Laplace μ/β·custom 스케줄 값. 순수 로직(Qt·네트워크·torch 없음).

지금은 제목 상수만 있는 자리다(계획 F1): 기능 스냅샷(core/sam_extra_capabilities ``TITLE_*``·``FEATURE_FLAGS``)이 이
상수 하나에서 제목을 읽는다. 설정 정규화·블록·infotext 는 A-XS1/A-XS2 작업이 이 파일을 채운다. 제목 문자열을 다른 곳에
복사하지 말고 ``SCRIPT_NAME`` 을 import 한다(확장 스크립트의 ``title()`` 과 같아야 한다 — 다르면 Forge 가 422).
"""
from __future__ import annotations

SCRIPT_NAME = "Extra Schedulers (sam-extra)"

__all__ = ["SCRIPT_NAME"]
