"""프리셋·저장값을 지금 백엔드 콤보 항목에 맞추기 — 순수 로직(Qt 없음).

ComboBoxProxy.setText 는 목록에 없는 값을 fallback 으로 기억하고 Vue 에 그대로 보낸다. 목록이
아직 비어 있는 시작 시점(load_settings)에는 그게 맞지만, 목록이 이미 찬 뒤 프리셋을 불러올 때
Forge 이름('Euler a', 'Karras', Forge 전용 VAE)이 ComfyUI 목록에 없으면 Vue 는 프리셋 값을
보여 주고 Python 은 이전 선택(currentText)으로 생성했다(audit #154 후속). 이제 적용 쪽
(ui/generation_settings_apply.py)이 여기서 index 를 찾아 setCurrentIndex 로 고르고, 못 찾으면
현재 선택을 그대로 두고 경고한다.

맞추는 순서: 정확히 같음 → 대소문자·공백/밑줄 무시 → 샘플러/스케줄러 별칭(Forge 표기 ↔ ComfyUI
KSampler 이름). 별칭 표는 ComfyUI 생성이 Forge 표기를 바꿀 때 쓰는 core.comfy_workflow_compiler
의 표를 그대로 쓴다 — 두 곳이 갈라지지 않게(tests/test_comfy_sampler_aliases.py 가 그 표를 고정).
"""

from __future__ import annotations

import re
from collections.abc import Callable, Iterable, Mapping
from typing import Any

#: 'Hires/ADetailer 는 메인 값 사용' 같은 자리표시 — 별칭 비교에서 실제 샘플러로 취급하지 않는다.
PLACEHOLDER_CHOICES = frozenset({
    'use same sampler', 'use same scheduler', 'use same checkpoint', 'use same vae',
})

_SEPARATORS = re.compile(r'[\s_]+')


def fold_choice(value: Any) -> str:
    """비교용 — 대소문자·연속 공백·밑줄 차이를 없앤다('SGM Uniform' == 'sgm_uniform')."""
    return _SEPARATORS.sub(' ', str(value or '').strip()).casefold()


def _alias_tables() -> tuple[Mapping[str, str], Mapping[str, str]]:
    try:
        from core.comfy_workflow_compiler import _FORGE_SAMPLER_ALIASES, _FORGE_SCHEDULER_ALIASES
    except Exception:   # 컴파일러를 못 읽어도 1·2단계 비교는 된다
        return {}, {}
    return _FORGE_SAMPLER_ALIASES, _FORGE_SCHEDULER_ALIASES


def _alias_key(value: Any, table: Mapping[str, str]) -> str:
    folded = re.sub(r'\s+', ' ', str(value or '').strip()).casefold()
    if not folded or fold_choice(folded) in PLACEHOLDER_CHOICES:
        return ''
    if folded in table:
        return str(table[folded]).casefold()
    return folded.replace(' ', '_')


def sampler_key(value: Any) -> str:
    """Forge 표기('Euler a', 'DPM++ 2M')와 ComfyUI 이름('euler_ancestral', 'dpmpp_2m')이 같은 키.

    'DPM++ 2M Karras' 같은 옛 결합 표기는 스케줄러 정보를 잃지 않게 별칭으로 풀지 않는다.
    """
    return _alias_key(value, _alias_tables()[0])


def scheduler_key(value: Any) -> str:
    """'Karras' ↔ 'karras', 'SGM Uniform' ↔ 'sgm_uniform'. 'Automatic' 은 따로 둔다(뜻이 다르다)."""
    return _alias_key(value, _alias_tables()[1])


def match_choice(value: Any, items: Iterable[Any], *,
                 key: Callable[[Any], str] | None = None) -> int:
    """``items`` 에서 ``value`` 와 같은 항목의 index. 없으면 -1.

    정확히 같음 → :func:`fold_choice` 같음 → ``key`` 가 있으면 별칭 키 같음(자리표시 항목 제외).
    """
    text = str(value or '').strip()
    names = [str(item) for item in items or []]
    if not text or not names:
        return -1
    if text in names:
        return names.index(text)
    folded = fold_choice(text)
    for index, name in enumerate(names):
        if fold_choice(name) == folded:
            return index
    if key is None:
        return -1
    wanted = key(text)
    if not wanted:
        return -1
    for index, name in enumerate(names):
        if fold_choice(name) in PLACEHOLDER_CHOICES:
            continue
        if key(name) == wanted:
            return index
    return -1


def match_sampler(value: Any, items: Iterable[Any]) -> int:
    return match_choice(value, items, key=sampler_key)


def match_scheduler(value: Any, items: Iterable[Any]) -> int:
    return match_choice(value, items, key=scheduler_key)


__all__ = [
    'PLACEHOLDER_CHOICES',
    'fold_choice',
    'match_choice',
    'match_sampler',
    'match_scheduler',
    'sampler_key',
    'scheduler_key',
]
