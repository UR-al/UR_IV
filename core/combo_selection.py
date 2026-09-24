# core/combo_selection.py
"""백엔드 연결 때 다시 채운 콤보의 선택 복원 — 순수 로직.

연결(on_webui_info_loaded)은 모델·샘플러·VAE 콤보를 비우고 새 목록으로 채운다. 예전엔 그 뒤
``load_settings()`` 전체를 다시 돌려 디스크 값으로 선택을 되살렸는데, 그 과정에서 저장하지
않은 프롬프트 편집이 디스크 값으로 덮이고 백엔드 어댑터가 새로 만들어졌다(감사 #29). 이제는
비우기 직전의 선택을 기억했다가 같은 항목을 다시 고르고, 기억한 값이 없거나 새 목록에 없을
때만 디스크 값을 쓴다.

비교는 콤보마다 load_settings 와 **같은 matcher** 를 쓴다(ui/combo_restore.BACKEND_COMBOS) —
체크포인트·VAE 는 해시·경로 무시, 샘플러·스케줄러는 Forge 표기 ↔ ComfyUI 이름 별칭
('DPM++ 2M' ↔ 'dpmpp_2m', 'Karras' ↔ 'karras'). 정확히 같은 문자열만 보면 Forge 에서 고른
샘플러가 ComfyUI 로 연결·전환할 때 첫 항목('euler')으로 조용히 바뀌었다.
"""
from __future__ import annotations

from typing import Callable, Iterable, Optional, Sequence

#: ``(값, 항목들) -> index(없으면 -1)`` — ui/generation_settings_apply 의 match_*_index 와 같은 모양.
Matcher = Callable[[str, Sequence[str]], int]


def exact_match_index(value: str, items: Sequence[str]) -> int:
    """정확히 같은 문자열만."""
    names = list(items or [])
    return names.index(value) if value in names else -1


def pick_combo_index(
    items: Sequence[str],
    candidates: Iterable[Optional[str]],
    *,
    match: Optional[Matcher] = None,
) -> int:
    """``candidates`` 를 순서대로 보고 ``items`` 안에서 처음 맞는 항목의 index. 없으면 -1.

    ``match`` 가 없으면 정확히 같은 문자열만 본다. 빈 후보는 건너뛴다. 앞 후보가 별칭으로라도
    맞으면 뒤 후보(디스크 값)는 보지 않는다 — 지금 선택이 저장값보다 우선이다.
    """
    names = [str(item) for item in (items or [])]
    if not names:
        return -1
    matcher = match or exact_match_index
    for candidate in candidates:
        text = str(candidate or "").strip()
        if not text:
            continue
        index = matcher(text, names)
        if 0 <= index < len(names):
            return index
    return -1


# ── ComfyUI 워크플로 모델 자동 선택 (ui/generator_webui._auto_select_workflow_model) ──

def _file_name(name: str) -> str:
    return str(name or "").replace("\\", "/").rsplit("/", 1)[-1].lower()


def workflow_checkpoint_index(items: Sequence[str], checkpoint: Optional[str]) -> int:
    """워크플로 로더가 싣는 ``checkpoint`` 의 콤보 index — 정확히 같은 이름, 없으면 파일명만
    (폴더·대소문자 무시) 같은 첫 항목. 없으면 -1."""
    text = str(checkpoint or "").strip()
    names = [str(item) for item in (items or [])]
    if not text or not names:
        return -1
    if text in names:
        return names.index(text)
    wanted = _file_name(text)
    for index, name in enumerate(names):
        if _file_name(name) == wanted:
            return index
    return -1


def _same_workflow(a: str, b: str) -> bool:
    def norm(path: str) -> str:
        return str(path or "").strip().replace("\\", "/").rstrip("/").lower()
    return norm(a) == norm(b)


def workflow_model_wins(*, kept_selection: bool, workflow_path: str,
                        previous_workflow_path: Optional[str]) -> bool:
    """연결 때 모델 콤보를 워크플로 로더의 체크포인트로 바꿀지.

    - 복원한 선택이 없으면(첫 연결·저장한 모델이 새 목록에 없음) 워크플로 모델을 고른다.
    - 복원한 선택(연결 전 선택·저장값)이 있으면 그대로 둔다 — 예전엔 연결·재시도·게이트·PRIMARY
      전환마다 워크플로 기본 모델로 덮어, 컴파일러가 그 이름을 로더에 써 넣고 다음 생성이 사용자가
      고르지 않은 모델로 돌았다.
    - 단 모델을 고른 뒤(``previous_workflow_path`` — 설정을 불러온 때나 지난 연결 때의 경로) 사용자가
      다른 워크플로를 골랐으면 새 워크플로의 모델을 따른다. 모르면(None) 바뀌지 않은 것으로 본다.
    """
    if not kept_selection:
        return True
    if previous_workflow_path is None:
        return False
    return not _same_workflow(workflow_path, previous_workflow_path)
