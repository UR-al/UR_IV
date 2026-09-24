"""Settings '기본값' 패널의 저장소(config/tab_defaults.json) — Qt 없는 순수 로직.

누가 무엇을 읽나(모두 실제로 쓰인다 — audit #140):
- T2I steps/cfg/width/height/seed/sampler/scheduler: 시작 시 빈 위젯(첫 실행이면 전부)에 적용
  (generator_main._apply_tab_defaults_to_empty_widgets).
- denoising: I2I 화면의 Denoising 초기값(I2IView).
- brushSize/effectStrength/yoloConf/snapRadius: 에디터 마스크 브러시·효과 세기·YOLO 신뢰도·
  자석 올가미 스냅 반경 초기값(EditorView/EffectPanel/EditorCanvas).
- hires_enabled/ad_enabled/sam3_enabled(+레거시 ad_s1/ad_s2): 첫 실행(prompt_settings.json 없음)에만.

저장은 **부분 병합**이다. Settings(keep-alive)가 들고 있던 옛 값 전체를 1.5초 뒤 통째로 써서
'전역 저장'이 방금 T2I 값으로 갱신한 steps/cfg/… 를 되덮던 양방향 덮어쓰기를 막는다 — 화면은
바뀐 키만 보내고, 여기서 기존 파일 위에 병합한다.

의미가 없어진 키(defaultRating: 검색 등급 필터는 ui_prefs.ratingFilter 가 주인 /
negpip_enabled: NegPiP 는 상시 적용)는 다음 저장 때 파일에서 지운다.
"""

from __future__ import annotations

import math
import os
from collections.abc import Mapping
from typing import Any, Callable

from utils.atomic_json import atomic_write_json, load_json_safe


def _int(lo: int, hi: int) -> Callable[[Any], int]:
    def conv(value: Any) -> int:
        number = float(value)
        if not math.isfinite(number):
            raise ValueError('not finite')
        return max(lo, min(hi, int(round(number))))
    return conv


def _float(lo: float, hi: float) -> Callable[[Any], float]:
    def conv(value: Any) -> float:
        number = float(value)
        if not math.isfinite(number):
            raise ValueError('not finite')
        return max(lo, min(hi, round(number, 4)))
    return conv


def _text(value: Any) -> str:
    return str(value if value is not None else '').strip()


def _seed(value: Any) -> str:
    text = _text(value)
    return text or '-1'


def _bool(value: Any) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in {'1', 'true', 'yes', 'on'}
    return bool(value)


#: 키 → 정규화 함수. 프론트 frontend/src/utils/tabDefaults.ts 의 FACTORY_DEFAULTS 와 같은 키.
TAB_DEFAULT_FIELDS: dict[str, Callable[[Any], Any]] = {
    'steps': _int(1, 500),
    'cfg': _float(0.0, 100.0),
    'width': _int(64, 8192),
    'height': _int(64, 8192),
    'seed': _seed,
    'sampler': _text,
    'scheduler': _text,
    'denoising': _float(0.0, 1.0),
    'brushSize': _int(1, 500),
    'effectStrength': _int(1, 100),
    'yoloConf': _float(0.01, 1.0),
    'snapRadius': _int(1, 100),
    'hires_enabled': _bool,
    'ad_enabled': _bool,
    'sam3_enabled': _bool,
    # 화면에는 없지만 첫 실행 적용이 읽는 레거시 키 — 보존한다.
    'ad_s1_enabled': _bool,
    'ad_s2_enabled': _bool,
}

#: 더는 의미가 없어 저장 때 지우는 키.
REMOVED_TAB_DEFAULT_KEYS = frozenset({'defaultRating', 'negpip_enabled'})

#: '전역 저장'이 현재 T2I 값으로 갱신하는 키(f90af4319 'Default 연동').
T2I_LINKED_KEYS = ('steps', 'cfg', 'width', 'height', 'seed')


def default_tab_defaults_path() -> str:
    return os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'config', 'tab_defaults.json')


def normalize_tab_defaults(raw: Any, *, keep_unknown: bool = False) -> dict[str, Any]:
    """알려진 키만 정규화해 돌려준다. 값이 깨졌으면 그 키를 뺀다(기본값은 호출자가 정한다)."""
    if not isinstance(raw, Mapping):
        return {}
    out: dict[str, Any] = {}
    for key, value in raw.items():
        name = str(key)
        if name in REMOVED_TAB_DEFAULT_KEYS:
            continue
        conv = TAB_DEFAULT_FIELDS.get(name)
        if conv is None:
            if keep_unknown:
                out[name] = value
            continue
        try:
            out[name] = conv(value)
        except (TypeError, ValueError, OverflowError):
            continue
    return out


def load_tab_defaults(path: str | None = None) -> dict[str, Any]:
    data = load_json_safe(path or default_tab_defaults_path(), {}, backup_corrupt=False)
    return normalize_tab_defaults(data)


def merge_tab_defaults(existing: Any, patch: Any) -> dict[str, Any]:
    """기존 파일 위에 patch 의 알려진 키만 덮는다. 모르는 기존 키는 보존, 폐기 키는 지운다."""
    merged = normalize_tab_defaults(existing, keep_unknown=True)
    merged.update(normalize_tab_defaults(patch))
    return merged


def save_tab_defaults_patch(patch: Any, path: str | None = None) -> dict[str, Any]:
    """patch 를 병합 저장하고 저장된 (정규화된) 값을 돌려준다."""
    target = path or default_tab_defaults_path()
    existing = load_json_safe(target, {}, backup_corrupt=False)
    merged = merge_tab_defaults(existing, patch)
    atomic_write_json(target, merged)
    return normalize_tab_defaults(merged)


def sync_t2i_defaults(values: Mapping[str, Any], path: str | None = None) -> dict[str, Any] | None:
    """'전역 저장' 연동 — 현재 T2I steps/cfg/width/height/seed 를 기본값으로 반영한다.

    파일이 없으면(기본값을 한 번도 저장하지 않음) 만들지 않는다 — 예전 동작과 같다.
    """
    target = path or default_tab_defaults_path()
    if not os.path.exists(target):
        return None
    patch = {key: values[key] for key in T2I_LINKED_KEYS if key in values and values[key] not in (None, '')}
    return save_tab_defaults_patch(patch, target)
