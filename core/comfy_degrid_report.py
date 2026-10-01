# core/comfy_degrid_report.py
"""ComfyUI 의 VAE DeGrid 결과 → 사용자 알림 — 순수 로직(Qt·네트워크 없음).

Forge 는 이미지마다 infotext 에 ``Anima DeGrid model…``(성공) 또는 ``Anima DeGrid error``(실패)를 남기고, 앱은 그
흔적으로 알린다(core/sam_extra_notices). ComfyUI 에서는 두 곳에서 같은 사실을 안다:

- **컴파일 경고** — 컴파일러가 노드를 넣지 않고 생성했다(core/comfy_workflow_compiler._add_degrid): 노드가 없거나
  입력 계약이 다름(팩 1.5.0 이전)·사용자 워크플로에 넣을 자리가 없음 → ``CODE_DEGRID_COMFY_UNAVAILABLE``,
  고른 모델이 노드 목록에 없음(``model not found: <이름|auto>`` — Forge 문구) → ``CODE_DEGRID_ERROR``.
  경고마다 원인(``cause`` — ``vae_degrid.COMFY_OMIT_*``)이 붙는다. 백엔드는 새 /object_info 로 풀릴 수 있는 원인
  (``COMFY_SCHEMA_FIXABLE`` — 노드 없음·입력 다름·모델 없음)만 보고 스키마를 다시 받는다(``schema_fixable_causes``).
- **노드 리포트** — 노드가 돌았다: ``/history`` 출력의 ``ui.ai_studio_degrid`` 에 이미지마다 리포트 dict 가 있다
  (comfy_custom_nodes/ai_studio_forge_parity/degrid_nodes.py). ``core/vae_degrid.report_params`` 로 Forge 와 같은
  infotext 키로 바꿔 Forge 결과와 **같은 규칙**(장마다 n/m, 이유 종류마다 한 번)으로 알린다(A9).

리포트가 이미지 메타데이터(PNG)를 바꾸지는 않는다 — ComfyUI PNG 는 앞으로도 그래프에 적힌 요청 값을 보여 준다
(core/comfy_metadata). 그 차이는 레지스트리 gap 으로 적어 둔다.
"""
from __future__ import annotations

from typing import Any, Iterable, Mapping, Optional

from core import sam_extra_notices as notices
from core import vae_degrid


def _node_order(node_id: Any) -> tuple:
    text = str(node_id)
    return (0, int(text), text) if text.isdigit() else (1, 0, text)


def reports_from_outputs(node_outputs: Any) -> list[dict]:
    """``/history`` 의 ``outputs``(노드 id → ui 출력) → DeGrid 리포트 목록(노드 id 순, 노드 안에서는 이미지 순).

    리포트가 없거나 모양이 틀리면 빈 목록 — 판단하지 않는다(알림도 없다)."""
    if not isinstance(node_outputs, Mapping):
        return []
    out: list[dict] = []
    for node_id in sorted(node_outputs, key=_node_order):
        output = node_outputs.get(node_id)
        items = output.get(vae_degrid.COMFY_UI_KEY) if isinstance(output, Mapping) else None
        if not isinstance(items, (list, tuple)):
            continue
        out.extend(dict(item) for item in items if isinstance(item, Mapping))
    return out


def params_from_outputs(node_outputs: Any) -> list[dict]:
    """노드 리포트 → 이미지마다 Forge 식 infotext 파라미터(꺼진 리포트는 빈 dict 라 뺀다)."""
    return [params for params in (vae_degrid.report_params(report) for report in reports_from_outputs(node_outputs))
            if params]


def schema_fixable_causes(warnings: Optional[Iterable[Any]]) -> frozenset:
    """컴파일 경고 중 새 /object_info 가 바꿀 수 있는 것 → ``{(cause, reason)}`` (원인이 같아도 이유가 다르면 —
    다른 모델 이름 — 다른 항목). 사용자 워크플로에 넣을 자리가 없는 경고(``COMFY_OMIT_PLACEMENT``)·원인이 없는 옛
    모양·다른 기능의 경고는 뺀다 — 스키마를 다시 받아도 같다."""
    out = set()
    for item in warnings or ():
        if not isinstance(item, Mapping) or item.get("feature") != "degrid":
            continue
        cause = item.get("cause")
        if cause in vae_degrid.COMFY_SCHEMA_FIXABLE:
            out.add((str(cause), " ".join(str(item.get("reason") or "").split())))
    return frozenset(out)


def _warning_notices(payload: Any, warnings: Iterable[Any]) -> list[notices.Notice]:
    out: list[notices.Notice] = []
    for item in warnings or ():
        if not isinstance(item, Mapping) or item.get("feature") != "degrid":
            continue
        reason = " ".join(str(item.get("reason") or "").split())
        if item.get("code") == notices.CODE_DEGRID_ERROR:
            # Forge 의 'model not found' 결과와 같은 알림(그 이미지는 DeGrid 없이 저장된다)
            out.extend(notices.degrid_result_notices(payload, [{vae_degrid.KEY_ERROR: reason}], comfy=True))
        elif item.get("code") == notices.CODE_DEGRID_COMFY_UNAVAILABLE:
            out.append(notices.degrid_comfy_unavailable_notice(reason))
    return out


def result_notices(payload: Any, node_outputs: Any, compile_warnings: Optional[Iterable[Any]] = None
                   ) -> list[notices.Notice]:
    """보낸 페이로드 + ``/history`` 출력 + 컴파일 경고 → VAE DeGrid 알림.

    컴파일 경고가 있으면 노드가 그래프에 없었으므로 그 경고만 알린다. 없으면 노드 리포트를 Forge 결과와 같은
    규칙으로 읽는다(켜진 DeGrid 블록을 보내지 않았으면 — 사용자 워크플로에 원래 있던 노드 — 아무것도 말하지 않는다)."""
    warned = _warning_notices(payload, compile_warnings or ())
    if warned:
        return _dedupe(warned)
    return notices.degrid_result_notices(payload, params_from_outputs(node_outputs), comfy=True)


def _dedupe(items: list[notices.Notice]) -> list[notices.Notice]:
    seen: set = set()
    out = []
    for item in items:
        if item.key not in seen:
            seen.add(item.key)
            out.append(item)
    return out


def merge_into_info(info: Any, new: Iterable[notices.Notice]) -> dict:
    """generation info 의 ``INFO_KEY`` 에 알림을 덧붙인 새 dict(이미 있던 알림은 앞에 그대로)."""
    merged = dict(info) if isinstance(info, Mapping) else {}
    items = notices.notices_to_dicts(new)
    if items:
        existing = merged.get(notices.INFO_KEY)
        merged[notices.INFO_KEY] = [*(existing if isinstance(existing, list) else []), *items]
    return merged


__all__ = ["merge_into_info", "params_from_outputs", "reports_from_outputs", "result_notices",
           "schema_fixable_causes"]
