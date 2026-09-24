# core/comic_generation.py
"""Comic 컷 T2I — 메인 T2I payload 빌더 스냅샷을 컷마다 재사용하는 순수 로직 (Qt 비의존).

예전 Comic 컷 생성은 {prompt, negative_prompt, seed, width, height} 만 담은 dict 로
``backend.txt2img(model_combo 모델)`` 을 직접 불렀다. 그래서 sampler·scheduler·steps·cfg,
VAE/TE(forge_additional_modules), Hires, LoRA 스택, NegPiP, ADetailer/SAM3(alwayson),
Comfy 품질·스펙트럼 프리셋이 모두 빠지고 서버 기본값(Forge Euler/50/cfg7 · Comfy euler/20/cfg7)
으로 생성됐으며, T2I 에서 KREA2 를 골라도 일반 체크포인트로 조용히 생성됐다.

이제 UI 스레드가 컷마다 ``GenerationMixin._build_generation_payload(prompt_override=컷 프롬프트,
snapshot=True)`` 로 T2I 스냅샷을 뜨고(ui/creator_actions), 이 모듈은
- ``panel_request``: 스냅샷에 컷 seed·네거티브·고정 크기를 입히고, T2I 메인 프롬프트용
  대체 텍스트(Hires/ADetailer/SAM3 prompt — core/main_prompt_overrides)를 떼어 모든 패스가
  컷 프롬프트·합친 네거티브를 따르게 하고,
- ``panel_entry``/``find_panel_request``: 스냅샷을 컷 **인덱스 + id** 로 묶어 워커가 둘 다
  맞는 것만 꺼낸다(중복 id·정규화 때 새로 붙는 id 로 다른 컷 설정을 쓰지 않게),
- ``unique_panel_ids``: 생성 요청 문서의 겹친 컷 id 를 결정적으로 풀고,
- ``resolve_panel_index``: 단일 컷 요청의 id 를 요청 원문 문서 기준으로 인덱스로 푼다,
- ``run_panel``: 워커 스레드에서 지연 프롬프트 훅을 돌린 뒤 비공개 키를 떼고
  family 에 따라 Krea2 러너 또는 ``backend.txt2img`` 로 보낸다(GenerationFlowWorker 와 같은 규칙).
"""
from __future__ import annotations

import copy
from typing import Any, Callable, Mapping, Optional

# Forge POST 로 새면 안 되는 앱 내부 키. (_chat_deferred_prompt 는 prepare_prompt_payload 가,
# _comfy_workflow_snapshot/_comfy_detail_passes 는 ComfyUI 백엔드가 소비한다.)
PRIVATE_KEYS: tuple[str, ...] = ("_generation_family", "_comfy_model_snapshot", "_xyz_info")

DEFAULT_SIZE = 1024


def merge_negative(base: Any, extra: Any) -> str:
    """T2I 네거티브(글자 그대로 보존) 뒤에 컷 네거티브 중 아직 없는 태그(대소문자 무시)만 덧붙인다."""
    base_text = str(base or "").strip().rstrip(",").rstrip()
    existing = {tag.strip().casefold() for tag in base_text.split(",") if tag.strip()}
    added: list[str] = []
    for tag in str(extra or "").split(","):
        cleaned = tag.strip()
        key = cleaned.casefold()
        if not cleaned or key in existing:
            continue
        existing.add(key)
        added.append(cleaned)
    if not added:
        return base_text
    return f"{base_text}, {', '.join(added)}" if base_text else ", ".join(added)


def explicit_size(payload: Mapping[str, Any]) -> Optional[tuple[int, int]]:
    """요청이 width/height 를 명시했으면 (w, h), 아니면 None (T2I 설정 크기를 쓴다)."""
    try:
        width = int(payload.get("width") or 0)
        height = int(payload.get("height") or 0)
    except (TypeError, ValueError):
        return None
    if width > 0 and height > 0:
        return width, height
    return None


def fallback_snapshot(prompt: str) -> dict[str, Any]:
    """T2I 빌더가 없는 호스트(테스트 fake 등)용 최소 스냅샷 — 예전 동작과 같다."""
    return {"prompt": str(prompt or ""), "negative_prompt": "", "width": DEFAULT_SIZE, "height": DEFAULT_SIZE}


def is_krea2_snapshot(snapshot: Mapping[str, Any]) -> bool:
    return str(snapshot.get("_generation_family") or "").strip().lower() == "krea2"


def panel_request(snapshot: Mapping[str, Any], panel_payload: Mapping[str, Any],
                  *, size: Optional[tuple[int, int]] = None) -> dict[str, Any]:
    """컷 프롬프트로 뜬 T2I 스냅샷 + 컷 seed/네거티브 + 고정 크기 → 컷 요청 payload.

    - prompt: 스냅샷 것(빌더가 컷 프롬프트에 LoRA 스택을 붙인 결과). 없으면 컷 프롬프트.
    - negative_prompt: T2I 네거티브에 컷 네거티브를 덧붙인다.
    - seed: 컷 seed(-1 이면 백엔드가 랜덤) — 컷마다 따로 고정할 수 있어야 한다.
    - size: 랜덤 해상도 때문에 컷 크기가 흩어지지 않게 호출자가 한 번 정한 값.
    - Hires/ADetailer/SAM3 의 '비우면 메인 사용' 텍스트는 T2I 메인 프롬프트용이라 뗀다 —
      남기면 컷의 Hires 두 번째 패스·얼굴 패스가 컷 대신 T2I 텍스트로 다시 칠해지고, 설정된
      hr_negative_prompt 는 방금 합친 컷 네거티브도 버린다. 비운 SAM3 는 run_panel 이 최종
      프롬프트로 채운다(T2I 빌더와 같은 폴백).
    """
    from core.main_prompt_overrides import clear_main_prompt_overrides

    request = clear_main_prompt_overrides(copy.deepcopy(dict(snapshot)))
    if not str(request.get("prompt") or "").strip():
        request["prompt"] = str(panel_payload.get("prompt") or "")
    request["negative_prompt"] = merge_negative(
        request.get("negative_prompt", ""), panel_payload.get("negative_prompt", ""),
    )
    try:
        request["seed"] = int(panel_payload.get("seed", -1))
    except (TypeError, ValueError):
        request["seed"] = -1
    if size is not None:
        request["width"], request["height"] = int(size[0]), int(size[1])
    return request


def panel_entry(index: int, panel_id: str, request: Mapping[str, Any]) -> dict[str, Any]:
    """UI 스레드 스냅샷 한 개 — 컷 인덱스와 id 를 같이 싣는다(dict 키 하나로 뭉치지 않게)."""
    return {"index": int(index), "panelId": str(panel_id), "request": dict(request)}


def find_panel_request(entries: Any, index: int, panel_id: str) -> Optional[dict[str, Any]]:
    """워커: 인덱스와 컷 id 가 **둘 다** 맞는 스냅샷의 사본. 없거나 어긋나면 None.

    id 만으로 찾으면 id 가 겹친 두 컷이 한 스냅샷을 나눠 쓰고, 인덱스만으로 찾으면 컷 순서가
    바뀐 문서에 다른 컷 설정을 입힌다 — 둘 다 조용히 틀린 그림이므로 호출자가 오류로 멈춘다.
    """
    if not isinstance(entries, list):
        return None
    for entry in entries:
        if not isinstance(entry, Mapping):
            continue
        entry_index = entry.get("index")
        if isinstance(entry_index, bool) or not isinstance(entry_index, int) or entry_index != index:
            continue
        request = entry.get("request")
        if str(entry.get("panelId", "")) == str(panel_id) and isinstance(request, Mapping):
            return copy.deepcopy(dict(request))
    return None


PANEL_ID_MAX = 80  # core.comic_studio._id 와 같은 길이 제한


def unique_panel_ids(panel_ids: list[str]) -> list[str]:
    """겹치는 컷 id 에 결정적 접미사(-2, -3 …)를 붙인 목록. 이미 유일하면 그대로 돌려준다.

    ComicStudio.normalize 는 컷 id 중복을 막지 않는다(웹 모드·원격 payload·손으로 고친
    comic_studio.json). 생성 경로에서 id 가 겹치면 결과 panelId 로 프론트가 엉뚱한 컷에 이미지를
    붙이므로 생성 요청 문서에서 한 번 풀어 준다. 먼저 나온 id 와 원래 유일한 id 는 바꾸지 않고,
    결과를 다시 넣어도 같은 목록이 나온다(멱등 — 워커 재정규화와 어긋나지 않게).
    normalize 자체에서 바꾸면 저장된 중복 문서의 content hash 검증이 깨져 문서를 못 읽는다.
    """
    taken = set(panel_ids)
    seen: set[str] = set()
    result: list[str] = []
    for panel_id in panel_ids:
        candidate = panel_id
        number = 2
        while candidate in seen:
            suffix = f"-{number}"
            candidate = f"{panel_id[:PANEL_ID_MAX - len(suffix)]}{suffix}"
            if candidate in taken:
                candidate = panel_id  # 다른 컷의 원래 id — 건너뛰고 다음 번호
            number += 1
        seen.add(candidate)
        result.append(candidate)
    return result


def resolve_panel_index(raw_document: Any, panel_ids: list[str], panel_id: Any) -> Optional[int]:
    """단일 컷 요청의 id → 컷 인덱스. 못 찾으면 None.

    ``raw_document`` 는 프론트가 보낸 원문(정규화 전), ``panel_ids`` 는 정규화된 문서의 컷 id 순서.
    정규화는 컷 순서를 그대로 두지만 id 는 바꿀 수 있다('컷 1'→'1', '컷'→무작위 uuid). 그래서
    요청 id 는 같은 원문의 컷 id 와 먼저 맞춰 보고(정확 일치 — 같은 id 가 여럿이면 첫 컷),
    원문에 없을 때만 정규화된 id 로 찾는다.
    """
    wanted = str(panel_id or "")
    if not wanted:
        return None
    raw_panels = raw_document.get("panels") if isinstance(raw_document, Mapping) else None
    if isinstance(raw_panels, list) and len(raw_panels) == len(panel_ids):
        for index, item in enumerate(raw_panels):
            if isinstance(item, Mapping) and str(item.get("id", "")) == wanted:
                return index
    for index, normalized_id in enumerate(panel_ids):
        if normalized_id == wanted:
            return index
    return None


def run_panel(backend: Any, model: str, payload: Mapping[str, Any],
              progress_callback: Optional[Callable[..., None]] = None,
              cancel_check: Optional[Callable[[], bool]] = None):
    """워커 스레드에서 컷 하나를 생성한다 (GenerationFlowWorker.run 과 같은 라우팅).

    1) 스냅샷의 지연 프롬프트(와일드카드·프롬프트 훅)를 여기서 푼다.
       비어 있는 SAM3 인페인트 prompt/negative 는 그 최종 텍스트로 채운다(T2I 빌더와 같은 폴백).
    2) family 를 떼고 비공개 키를 버린다(Forge POST 로 새지 않게).
    3) krea2 면 Krea2 러너, 아니면 backend.txt2img — 받을 수 있으면 cancel_check 도 넘긴다.
    """
    from core.cancellable_call import call_with_optional_cancel
    from core.chat_generation import prepare_prompt_payload
    from core.main_prompt_overrides import fill_sam3_prompt_fallback

    params = fill_sam3_prompt_fallback(prepare_prompt_payload(dict(payload)))
    family = str(params.pop("_generation_family", "standard") or "standard").strip().lower()
    for key in PRIVATE_KEYS:
        params.pop(key, None)
    if family == "krea2":
        from core.krea2_generation import run_krea2_generation

        return call_with_optional_cancel(
            run_krea2_generation, backend, "t2i", params,
            progress_callback=progress_callback, cancel_check=cancel_check,
        )
    return call_with_optional_cancel(
        backend.txt2img, model, params, progress_callback, cancel_check=cancel_check,
    )
