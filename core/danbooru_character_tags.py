# core/danbooru_character_tags.py
"""danbooru 게시물 표본에서 캐릭터의 공통 general 태그를 집계한다(Qt 비의존).

CharacterPresetModal 의 'danbooru' 버튼이 쓴다. 예전엔 동기 슬롯(fetchCharacterTagsOnline)이
GUI 스레드에서 HTTPS 를 최대 2번(각 timeout 12초) 보내 앱 전체(생성 진행률·미리보기 포함)가
멈췄다 — 지금은 VueBridge.requestCharacterTagsOnline 이 워커 스레드에서 이 함수를 부르고
characterTagsOnlineReady 로 결과를 보낸다.

- 질의는 ``"<tag> solo"`` 를 먼저, 비었거나 실패하면 ``"<tag>"`` 로 한 번 더(순차).
- 타임아웃은 (연결 5초, 읽기 10초). 결과는 절대 예외로 새지 않고 ``{"error": ...}`` 로 돌아온다
  — 신호가 안 오면 프론트의 '불러오는 중'이 영영 켜진 채 남는다.
"""
from __future__ import annotations

from collections import Counter
from typing import Any, Callable, Optional

DANBOORU_POSTS_URL = "https://danbooru.donmai.us/posts.json"
USER_AGENT = "UR_IV/1.0 (character tag lookup)"
REQUEST_TIMEOUT = (5, 10)
SAMPLE_LIMIT = 100
RANK_LIMIT = 60
RESULT_LIMIT = 40


def normalize_character_tag(name: str) -> str:
    """'Hatsune Miku' → 'hatsune_miku' (danbooru 태그 표기)."""
    return "_".join(str(name or "").strip().lower().split())


def rank_general_tags(posts: Any) -> tuple[list[str], int]:
    """posts.json 표본 → (빈도순 태그 상위 RESULT_LIMIT 개(공백 표기), 태그가 있던 게시물 수)."""
    counts: Counter = Counter()
    sampled = 0
    for post in posts if isinstance(posts, list) else []:
        general = post.get("tag_string_general") if isinstance(post, dict) else ""
        if not general or not isinstance(general, str):
            continue
        sampled += 1
        counts.update(general.split())
    ranked = [tag.replace("_", " ") for tag, _count in counts.most_common(RANK_LIMIT)]
    return ranked[:RESULT_LIMIT], sampled


def fetch_character_tags_online(name: str, *, get: Optional[Callable[..., Any]] = None) -> dict:
    """캐릭터 이름 → ``{"tags": [...], "sampled": n}`` 또는 ``{"error": "..."}``. 예외를 내지 않는다."""
    tag = normalize_character_tag(name)
    if not tag:
        return {"error": "캐릭터 이름 없음"}
    if get is None:
        import requests

        get = requests.get
    posts: Any = []
    last_error = ""
    for query in (f"{tag} solo", tag):
        try:
            response = get(
                DANBOORU_POSTS_URL,
                params={"tags": query, "limit": SAMPLE_LIMIT, "only": "tag_string_general"},
                timeout=REQUEST_TIMEOUT,
                headers={"User-Agent": USER_AGENT},
            )
            response.raise_for_status()
            posts = response.json()
            if isinstance(posts, list) and posts:
                break
        except Exception as exc:   # 네트워크·HTTP·JSON 오류 — 다음 질의로
            last_error = str(exc)
            posts = []
            continue
    if not isinstance(posts, list) or not posts:
        detail = f" — {last_error[:120]}" if last_error else ""
        return {"error": f"danbooru 게시물 없음 (이름/철자 확인){detail}"}
    tags, sampled = rank_general_tags(posts)
    if not sampled:
        return {"error": "태그 없음"}
    return {"tags": tags, "sampled": sampled}


__all__ = [
    "REQUEST_TIMEOUT",
    "fetch_character_tags_online",
    "normalize_character_tag",
    "rank_general_tags",
]
