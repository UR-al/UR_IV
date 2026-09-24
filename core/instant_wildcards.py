"""
Instant Wildcards — 파일 기반 와일드카드의 보조 인라인 시스템.

NAIA 2.0 ``modules/instant_wildcard_module.py`` 패턴 참고.

목적
----
``wildcards/*.txt`` 파일 와일드카드(utils/file_wildcard — ``~/name/~`` · ``__name__``)와
별도로, **JSON 한 곳에 모아 관리**하는 가벼운 와일드카드.
자주 쓰는 짧은 후보군(의상, 표정 등)을 매번 파일로 만들지 않고 정의한다.

파일 형식 (``user_data/instant_wildcards.json``):
```json
{
  "version": 1,
  "wildcards": {
    "mood":      ["happy", "sad", "{2}:neutral"],
    "outfit":    ["dress", "swimsuit", "uniform"]
  }
}
```

줄 가중치: ``{N}:tag`` (N 은 양수, 없으면 1 — :func:`parse_weighted_line`).

치환은 PromptPipeline 훅(POST_PROCESSING, priority 80)으로 한다. 훅은 부팅 때
``core.standard_hooks.register_standard_hooks`` 가 :func:`ensure_hook_registered` 로 한 번
등록한다 — '즉석 WC' 창을 열지 않아도 모든 생성 경로(run_pipeline_on_text)가 ``$$name$$`` 를 푼다.
"""
from __future__ import annotations

import json
import random
import re
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

from utils.app_logger import get_logger

_logger = get_logger("instant_wildcards")


# 패턴: $$NAME$$ (파일 와일드카드의 ~/NAME/~ · __NAME__ 과 충돌 안 함)
# 공개 이름은 실시간 프롬프트 정리(utils.prompt_cleaner)가 같은 정규식으로 토큰을 가릴 때 쓴다 —
# 이름에 공백이 허용되지 않으므로 밑줄→공백이 끼어들면 토큰이 풀리지 않는다.
INSTANT_WILDCARD_PATTERN = re.compile(r"\$\$(?P<name>[\w\-/.]+)\$\$")
_PATTERN_RE = INSTANT_WILDCARD_PATTERN

# 줄 가중치 문법: {weight}:line — weight 는 양수(정수/실수)
_WEIGHT_RE = re.compile(r"^\{(?P<w>[0-9]+(?:\.[0-9]+)?)\}:(?P<line>.*)$")

# PromptPipeline 에 등록할 때 쓰는 이름 — 이 이름으로 중복 등록을 막는다.
HOOK_NAME = "instant_wildcards"
HOOK_PRIORITY = 80


def parse_weighted_line(line: str) -> tuple[float, str]:
    """``{100}:tag`` 형태에서 ``(가중치, 텍스트)`` 추출.

    가중치가 없거나 0 이하이면 기본 가중치 1 — ``{0}:x`` 는 ``(1.0, 'x')``.
    """
    line = line.strip()
    m = _WEIGHT_RE.match(line)
    if not m:
        return 1.0, line
    try:
        w = float(m.group("w"))
        if w <= 0:
            return 1.0, m.group("line").strip()
        return w, m.group("line").strip()
    except ValueError:
        return 1.0, line


@dataclass
class InstantWildcardSet:
    """저장 단위 — 이름 → 후보 라인 리스트."""
    version: int = 1
    wildcards: dict[str, list[str]] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {"version": self.version, "wildcards": dict(self.wildcards)}

    @classmethod
    def from_dict(cls, d: dict) -> "InstantWildcardSet":
        wc = d.get("wildcards", {})
        return cls(
            version=int(d.get("version", 1)),
            wildcards={k: list(v) for k, v in wc.items() if isinstance(v, list)},
        )


class InstantWildcards:
    """JSON 와일드카드 매니저.

    스레드 안전. RNG는 인스턴스마다 별도 가능 (테스트 시 시드 고정).
    """

    def __init__(self,
                 store_path: Optional[Path] = None,
                 rng: Optional[random.Random] = None):
        self.store_path = Path(store_path) if store_path else None
        self._rng = rng or random.Random()
        self._set = InstantWildcardSet()
        self._lock = threading.RLock()

        if self.store_path and self.store_path.is_file():
            self.load()

    # ────────────────────────────────────────
    # 영속화
    # ────────────────────────────────────────

    def load(self) -> bool:
        if not self.store_path or not self.store_path.is_file():
            return False
        try:
            data = json.loads(self.store_path.read_text(encoding="utf-8"))
            with self._lock:
                self._set = InstantWildcardSet.from_dict(data)
            return True
        except Exception:
            _logger.exception(f"instant wildcards load failed: {self.store_path}")
            return False

    def save(self) -> bool:
        if not self.store_path:
            return False
        try:
            self.store_path.parent.mkdir(parents=True, exist_ok=True)
            with self._lock:
                data = self._set.to_dict()
            from utils.atomic_json import atomic_write_json
            atomic_write_json(str(self.store_path), data, indent=2)
            return True
        except Exception:
            _logger.exception(f"instant wildcards save failed: {self.store_path}")
            return False

    # ────────────────────────────────────────
    # CRUD
    # ────────────────────────────────────────

    def names(self) -> list[str]:
        with self._lock:
            return sorted(self._set.wildcards.keys())

    def get(self, name: str) -> list[str]:
        with self._lock:
            return list(self._set.wildcards.get(name, []))

    def set(self, name: str, lines: list[str]) -> None:
        with self._lock:
            self._set.wildcards[name] = list(lines)

    def delete(self, name: str) -> bool:
        with self._lock:
            return self._set.wildcards.pop(name, None) is not None

    # ────────────────────────────────────────
    # 선택 (가중치 랜덤)
    # ────────────────────────────────────────

    def pick(self, name: str) -> str:
        with self._lock:
            lines = self._set.wildcards.get(name, [])
            if not lines:
                return f"$${name}$$"
            weights: list[float] = []
            values: list[str] = []
            for ln in lines:
                w, v = parse_weighted_line(ln)
                weights.append(w)
                values.append(v)
            idx = self._rng.choices(range(len(values)), weights=weights, k=1)[0]
            return values[idx]

    def resolve(self, text: str, max_depth: int = 8) -> str:
        """``$$name$$`` 패턴을 치환. 중첩도 처리."""
        if not text or max_depth <= 0:
            return text

        def _repl(m: re.Match) -> str:
            try:
                return self.pick(m.group("name"))
            except Exception:
                _logger.exception(f"instant resolve failed: {m.group(0)}")
                return m.group(0)

        result = _PATTERN_RE.sub(_repl, text)
        if _PATTERN_RE.search(result) and result != text:
            return self.resolve(result, max_depth - 1)
        return result

    # ────────────────────────────────────────
    # Pipeline 훅 팩토리
    # ────────────────────────────────────────

    def make_hook(self) -> Callable:
        """PromptPipeline 훅 — main_tags를 join → resolve → split.

        ``$$`` 가 없는 프롬프트는 손대지 않는다(태그 목록을 다시 쪼개지 않는다).
        """
        def hook(ctx) -> None:
            combined = ", ".join(ctx.main_tags)
            if "$$" not in combined:
                return
            resolved = self.resolve(combined)
            if resolved != combined:
                ctx.main_tags = [t.strip() for t in resolved.split(",") if t.strip()]
        return hook


# ─────────────────────────────────────────────
# 프로세스 싱글톤 + 멱등 훅 등록
# ─────────────────────────────────────────────

_instance_lock = threading.Lock()
_instance: Optional[InstantWildcards] = None


def default_store_path() -> Path:
    """``user_data/instant_wildcards.json`` (옛 ``save/`` 위치에서 1회 이관)."""
    from core.storage_paths import user_data_file
    return user_data_file(
        "instant_wildcards.json",
        legacy_paths="save/instant_wildcards.json",
    )


def get_instant_wildcards() -> InstantWildcards:
    """프로세스 전역 InstantWildcards — 관리 액션과 생성 훅이 같은 객체를 쓴다."""
    global _instance
    if _instance is None:
        with _instance_lock:
            if _instance is None:
                _instance = InstantWildcards(store_path=default_store_path())
    return _instance


def ensure_hook_registered(pipeline=None, instance: Optional[InstantWildcards] = None) -> bool:
    """``$$name$$`` 치환 훅을 pipeline 에 한 번만 등록한다(이름 기준 멱등).

    :return: 이번 호출로 새로 등록했으면 True, 이미 있으면 False.
    PromptPipeline.register 는 중복을 거르지 않으므로 반드시 이 함수로만 등록한다.
    """
    from core.prompt_pipeline import HookPoint, get_pipeline
    pl = pipeline if pipeline is not None else get_pipeline()
    if pl.has_hook(HookPoint.POST_PROCESSING, HOOK_NAME):
        return False
    iw = instance if instance is not None else get_instant_wildcards()
    pl.register(
        HookPoint.POST_PROCESSING,
        iw.make_hook(),
        priority=HOOK_PRIORITY,
        name=HOOK_NAME,
    )
    return True
