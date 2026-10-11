# core/related_tags.py
"""연관 태그 — 프롬프트 태그와 Danbooru 게시물에서 자주 함께 쓰인 태그 추천(순수 로직, 표준 라이브러리만).

아이디어(동시 출현 표로 '함께 쓰인 태그'를 고른다)는 cstria0106/anima-studio(MIT)에서 왔고, 코드는
가져오지 않았다. 표는 앱의 로컬 Danbooru 데이터(danbooru_optimized/*.parquet)로 만든다
(core/related_tags_build.py · core/related_tags_store.py · core/related_tags_service.py).

순위(원본의 '횟수 합' 순위는 이 데이터에서 프롬프트와 상관없이 늘 같은 일반 태그를 냈다 — 실측):
  1. 기준 태그(context) 중 몇 개와 '함께 쓰였나'(coverage) — 큰 것 먼저.
     '함께 쓰였다' = 합친 쌍 횟수 ≥ min_pair 이고 더한 값 P(g|c) − P(g) > 0.
  2. 더한 값의 합 — 큰 것 먼저.
  3. 전체 게시물 수 — 큰 것 먼저. 4. 이름 — 사전 순.
기준 태그: 프롬프트의 어휘 안 태그 중 게시물 비율 GENERIC_SHARE 이상(1girl·solo·long_hair …)은 더 구체적인
태그가 있으면 뺀다. MAX_CONTEXT 개가 넘으면 가장 드문 것만 남긴다.
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import AbstractSet, Callable, Iterable, Mapping, Sequence

from core.prompt_comments import strip_prompt_comments

FORMAT_VERSION = 1
N_CANDIDATES = 4096       # 추천 후보 = 일반 태그 상위 4096(약 5.7k 게시물 이상)
N_GENERAL = 8192          # 기준 태그로 쓰는 일반 태그(후보 포함)
N_CHARACTER = 3072
N_COPYRIGHT = 1024
TOP_K = 128               # 표에 남기는 태그마다의 이웃 수
MIN_PAIR_BUILD = 20       # 샤드 표에 남기는 쌍 횟수 하한
MIN_PAIR_QUERY = 30       # 합친 표에서 '함께 쓰였다'로 치는 하한
MAX_CONTEXT = 16
GENERIC_SHARE = 0.20
MAX_TAGS_PER_POST = 96    # 어휘 안 태그가 이보다 많은 게시물(약 0.4%)은 건너뛴다
BATCH_ROWS = 16384
DEFAULT_LIMIT = 16
MAX_LIMIT = 32
SHARDS = ("g", "s", "q", "e")
DEFAULT_SHARDS = ("g", "s")
SEC_PER_MILLION_POSTS = 15
MAX_FIELD_CHARS = 100_000

# 요청 페이로드의 칸 이름(프론트 utils/relatedTags.ts RELATED_TAG_FIELDS 와 같은 키).
QUERY_FIELDS = ("count", "character", "copyright", "prefix", "main", "suffix")

_REQUEST_ID = re.compile(r"^[A-Za-z0-9_-]{1,120}$")
_ANGLE = re.compile(r"<[^<>]*>")
_WILDCARD = re.compile(r"__[^,\n]+?__")
_ALTERNATIVES = re.compile(r"\{[^{}]*\|[^{}]*\}")
_KEYWORDS = re.compile(r"\b(?:BREAK|AND)\b")
_ESCAPED = re.compile(r"\\(.)", re.S)
_BRACKETS = re.compile(r"[()\[\]{}]")
_SPLIT = re.compile(r"[,\n|]")
_WEIGHT = re.compile(r"^(?P<body>.*?\S)\s*:\s*(?P<w>[+-]?(?:\d+(?:\.\d*)?|\.\d+))$", re.S)
_SPACES = re.compile(r"\s+")
_PERSON_COUNT = re.compile(r"^\d+\+?(?:girl|boy|other)s?$")
_PERSON_WORDS = frozenset({
    "multiple_girls", "multiple_boys", "multiple_others",
    "solo", "solo_focus", "male_focus", "no_humans",
})
# 이스케이프한 글자는 괄호 제거·가중치 해석을 피하려고 잠시 사용자 영역 문자로 바꿔 둔다.
_PLACEHOLDER_BASE = 0xE000


def normalize_tag(text: str) -> str:
    """앞뒤 공백 제거 → 소문자 → 공백 묶음을 밑줄로(Danbooru 표기)."""
    return _SPACES.sub("_", str(text or "").strip().lower())


def extract_prompt_tags(text: object) -> list[tuple[str, float]]:
    """프롬프트 → [(정규화한 태그, 가중치)] (나온 순서, 중복 포함).

    <…>(LoRA·임베딩), __와일드카드__, {a|b} 선택지, BREAK·AND 는 버린다. ``\\(`` ``\\)`` 는 글자 괄호로 남기고
    이스케이프 안 된 ()[]{} 는 지운다. 끝의 ``:숫자`` 는 태그 본문 뒤일 때만 가중치로 읽는다(``:3`` · ``:d`` 는 태그).
    프롬프트 주석(// · /* */)은 생성에 보내지 않으므로 먼저 뗀다(core/prompt_comments) — 주석 속 태그는 기준
    태그도, '이미 있는 태그'도 아니다.
    """
    if not isinstance(text, str) or not text.strip():
        return []
    source = strip_prompt_comments(text[:MAX_FIELD_CHARS])
    literals: list[str] = []

    def hide(match: re.Match) -> str:
        literals.append(match.group(1))
        return chr(_PLACEHOLDER_BASE + len(literals) - 1)

    if len(re.findall(r"\\", source)) > 4096:      # 비정상 입력 — 자리표시 문자가 모자라지 않게
        source = source.replace("\\", "")
    source = _ESCAPED.sub(hide, source)
    source = _ANGLE.sub(",", source)
    source = _WILDCARD.sub(",", source)
    previous = None
    while previous != source:
        previous = source
        source = _ALTERNATIVES.sub(",", source)
    source = _KEYWORDS.sub(",", source)
    source = _BRACKETS.sub("", source)

    def reveal(value: str) -> str:
        return "".join(
            literals[ord(ch) - _PLACEHOLDER_BASE]
            if _PLACEHOLDER_BASE <= ord(ch) < _PLACEHOLDER_BASE + len(literals) else ch
            for ch in value)

    out: list[tuple[str, float]] = []
    for piece in _SPLIT.split(source):
        piece = piece.strip()
        if not piece:
            continue
        weight = 1.0
        match = _WEIGHT.match(piece)
        if match:
            try:
                value = float(match.group("w"))
            except ValueError:
                value = math.nan
            if math.isfinite(value):
                weight = value
                piece = match.group("body")
        name = normalize_tag(reveal(piece))
        if name:
            out.append((name, weight))
    return out


def is_person_count_tag(name: str) -> bool:
    """인원·인물 표지 태그(1girl · 2boys · multiple_girls · solo · no_humans …) — 추천하지 않는다."""
    key = normalize_tag(name)
    return bool(_PERSON_COUNT.match(key)) or key in _PERSON_WORDS


def resolve(name: str, index: Mapping[str, int], aliases: Mapping[str, int]) -> int | None:
    """정규화한 태그 → 어휘 id(별칭이면 원래 태그의 id). 어휘 밖이면 None."""
    key = normalize_tag(name)
    found = index.get(key)
    if found is None:
        found = aliases.get(key)
    return found


def select_context(ids_in_order: Iterable[int], occ: Sequence[int], total_posts: float, *,
                   limit: int = MAX_CONTEXT, generic_share: float = GENERIC_SHARE) -> list[int]:
    """기준 태그 고르기 — 중복 제거, 게시물이 없는 태그 제외, 흔한 태그는 다른 게 있으면 제외,
    limit 개가 넘으면 가장 드문 것(같으면 먼저 나온 것)만. 순서는 프롬프트 순서 그대로."""
    seen: set[int] = set()
    ids: list[int] = []
    for tag_id in ids_in_order:
        if tag_id in seen or not (0 <= tag_id < len(occ)) or occ[tag_id] <= 0:
            continue
        seen.add(tag_id)
        ids.append(tag_id)
    if not ids or total_posts <= 0:
        return []
    specific = [i for i in ids if occ[i] / total_posts < generic_share]
    chosen = specific or ids
    if len(chosen) > limit:
        position = {tag_id: n for n, tag_id in enumerate(chosen)}
        keep = set(sorted(chosen, key=lambda i: (occ[i], position[i]))[:limit])
        chosen = [i for i in chosen if i in keep]
    return chosen


def implied_closure(ids: Iterable[int], implications: Mapping[int, Sequence[int]]) -> set[int]:
    """ids 가 (거듭) 함의하는 태그 전부 — ids 자신은 함의 사슬에 다시 나올 때만 들어간다."""
    out: set[int] = set()
    stack = [i for i in ids]
    while stack:
        current = stack.pop()
        for implied in implications.get(current, ()):
            if implied not in out:
                out.add(implied)
                stack.append(implied)
    return out


@dataclass(frozen=True)
class Ranked:
    id: int
    coverage: int
    score: float
    via: tuple[int, ...]


def rank_related(context: Sequence[int], neighbors: Callable[[int], Mapping[int, int]],
                 occ: Sequence[int], total_posts: float, names: Sequence[str], *,
                 blocked: AbstractSet[int] = frozenset(), allowed: Callable[[int], bool] = lambda _id: True,
                 min_pair: int = MIN_PAIR_QUERY, limit: int = DEFAULT_LIMIT) -> list[Ranked]:
    """기준 태그와 함께 쓰인 후보 태그의 순위(모듈 설명 참조). via = 그 후보와 함께 쓰인 기준 태그(기준 순서)."""
    if total_posts <= 0 or limit <= 0:
        return []
    coverage: dict[int, int] = {}
    score: dict[int, float] = {}
    via: dict[int, list[int]] = {}
    context_set = set(context)
    for c in context:
        oc = occ[c]
        if oc <= 0:
            continue
        for g, count in neighbors(c).items():
            if count < min_pair or g in blocked or g in context_set:
                continue
            added = count / oc - occ[g] / total_posts
            if added <= 0:
                continue
            if g not in coverage:
                if not allowed(g):
                    blocked = blocked | {g}   # 같은 후보를 다시 묻지 않는다
                    continue
                coverage[g] = 0
                score[g] = 0.0
                via[g] = []
            coverage[g] += 1
            score[g] += added
            via[g].append(c)
    order = sorted(coverage, key=lambda g: (-coverage[g], -score[g], -occ[g], names[g]))
    return [Ranked(g, coverage[g], score[g], tuple(via[g])) for g in order[:limit]]


# ── 브리지 페이로드 ────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class QueryRequest:
    request_id: str
    fields: tuple[tuple[str, str], ...]   # QUERY_FIELDS 순서의 (칸, 텍스트)
    negative: str
    exclude: str
    limit: int


@dataclass(frozen=True)
class BuildRequest:
    request_id: str
    shards: tuple[str, ...]


def _request_id(payload: object) -> str:
    request_id = payload.get("requestId") if isinstance(payload, dict) else None
    if not isinstance(request_id, str) or not _REQUEST_ID.fullmatch(request_id):
        raise ValueError("요청 식별자가 올바르지 않습니다. 패널을 다시 열어 주세요.")
    return request_id


def _text(value: object, label: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"연관 태그 요청의 {label} 값이 문자열이 아닙니다.")
    return value[:MAX_FIELD_CHARS]


def parse_query_payload(payload: object) -> QueryRequest:
    """related_tags_query 페이로드 검사. 틀리면 ValueError(사용자 문구)."""
    request_id = _request_id(payload)
    fields = payload.get("fields")
    if not isinstance(fields, dict):
        raise ValueError("연관 태그 요청에 프롬프트 칸이 없습니다.")
    texts = tuple((key, _text(fields.get(key), key)) for key in QUERY_FIELDS)
    limit = payload.get("limit", DEFAULT_LIMIT)
    if isinstance(limit, bool) or not isinstance(limit, (int, float)) or not math.isfinite(limit):
        limit = DEFAULT_LIMIT
    return QueryRequest(
        request_id=request_id, fields=texts,
        negative=_text(payload.get("negative", ""), "negative"),
        exclude=_text(payload.get("exclude", ""), "exclude"),
        limit=max(1, min(MAX_LIMIT, int(limit))),
    )


def parse_shards(value: object) -> tuple[str, ...]:
    """등급 목록 → SHARDS 순서의 튜플. 모르는 등급·빈 목록은 ValueError."""
    if not isinstance(value, (list, tuple)) or not value:
        raise ValueError("만들 등급이 없습니다.")
    wanted = set()
    for item in value:
        if not isinstance(item, str) or item.lower() not in SHARDS:
            raise ValueError("알 수 없는 등급입니다.")
        wanted.add(item.lower())
    return tuple(s for s in SHARDS if s in wanted)


def parse_build_payload(payload: object) -> BuildRequest:
    request_id = _request_id(payload)
    return BuildRequest(request_id, parse_shards(payload.get("shards")))


def parse_cancel_payload(payload: object) -> str:
    return _request_id(payload)


def rating_shards(ratings: object) -> tuple[str, ...]:
    """등급 필터(set·list) → SHARDS 순서. 비었거나 모르면 기본 G·S."""
    if isinstance(ratings, (set, frozenset, list, tuple)):
        chosen = tuple(s for s in SHARDS if s in {str(r).lower() for r in ratings})
        if chosen:
            return chosen
    return DEFAULT_SHARDS


__all__ = [
    "BATCH_ROWS", "DEFAULT_LIMIT", "DEFAULT_SHARDS", "FORMAT_VERSION", "GENERIC_SHARE", "MAX_CONTEXT",
    "MAX_LIMIT", "MAX_TAGS_PER_POST", "MIN_PAIR_BUILD", "MIN_PAIR_QUERY", "N_CANDIDATES", "N_CHARACTER",
    "N_COPYRIGHT", "N_GENERAL", "QUERY_FIELDS", "SEC_PER_MILLION_POSTS", "SHARDS", "TOP_K",
    "BuildRequest", "QueryRequest", "Ranked",
    "extract_prompt_tags", "implied_closure", "is_person_count_tag", "normalize_tag",
    "parse_build_payload", "parse_cancel_payload", "parse_query_payload", "parse_shards",
    "rank_related", "rating_shards", "resolve", "select_context",
]
