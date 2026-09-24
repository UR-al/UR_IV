# core/tag_intelligence.py
"""Tag intelligence for classification, noise filtering, and NSFW signals.

The module intentionally knows stable :class:`TagAsset` identifiers rather
than filenames.  Data remains lazy-loaded and all lookups use the same
lowercase, underscore-to-space comparison form.
"""

from __future__ import annotations

import threading

from core.tag_database import TagAsset, TagDatabase, get_tag_database


def _norm(t: str) -> str:
    return (t or "").strip().lower().replace("_", " ").replace(r"\(", "(").replace(r"\)", ")")


# danbooru 실태그는 아니지만 프롬프트에서 흔히 쓰는 품질/메타 태그 (노이즈 필터에서 보존)
_PROMPT_ALLOW = {_norm(tag) for tag in {
    "masterpiece", "best quality", "high quality", "normal quality", "low quality",
    "worst quality", "amazing quality", "great quality", "good quality", "high resolution",
    "very aesthetic", "aesthetic", "very awa", "newest", "recent", "oldest", "early", "mid",
    "ultra-detailed", "ultra detailed", "highly detailed", "detailed", "best aesthetic",
    "score_9", "score_8_up", "score_7_up", "source anime", "nsfw", "sfw",
}}

# 의류 부위(region) → 한국어 라벨 (UI 그룹 표시용)
REGION_LABELS = {
    "HEAD_NECK_FACE": "머리·목·얼굴",
    "UPPER_BODY": "상의",
    "WAIST_HIP": "허리·하의",
    "ARMS_HANDS": "팔·손",
    "LEGS_FEET": "다리·발",
    "FULL_BODY": "전신",
    "STYLE": "스타일",
    "UNASSIGNED": "기타",
}


class TagIntelligence:
    def __init__(self, database: TagDatabase | None = None):
        self._database = database or get_tag_database()
        self._loaded = False
        self._cat: dict[str, str] = {}     # norm → category 문자열
        self._count: dict[str, int] = {}   # norm → 빈도
        self._rating: dict[str, tuple] = {}  # norm → (g, s, q, e)
        self._clothes: set[str] = set()
        self._charac: set[str] = set()
        self._colors: set[str] = set()
        self._clothes_curated: set[str] = set()
        self._clothes_extended: set[str] = set()
        self._charac_curated: set[str] = set()
        self._charac_extended: set[str] = set()
        self._colors_curated: set[str] = set()
        self._colors_extended: set[str] = set()
        self._totals = None
        # 신규: region / copyright / 카테고리 사전
        self._regions: dict[str, str] = {}      # norm 의류태그 → REGION 키
        self._copyright: dict[str, str] = {}    # norm 캐릭터/별칭 → 시리즈(copyright)
        self._copyright_vals: set[str] = set()   # copyright 값 집합(적재 끝에 함께 생성)
        self._expression: set[str] = set()
        self._location: set[str] = set()
        self._pose: set[str] = set()
        self._object: set[str] = set()
        self._meta: set[str] = set()
        self._group_tags: set[str] = set()
        # 적재 직렬화 — 여러 스레드(예열·GUI 슬롯·워커)가 첫 조회를 동시에 해도 한 번만
        # 적재하고, 적재 중인 반쪽 사전을 누구도 조회하지 않게 한다.
        self._load_lock = threading.RLock()

    def _ensure(self):
        if self._loaded:          # 빠른 경로 — 적재가 **끝난 뒤에만** 참
            return
        with self._load_lock:
            if self._loaded:
                return
            try:
                self._load_all()
            finally:
                # copyright 값 집합은 여기서 한 번에 만든다 — 예전처럼 is_known 이 적재
                # 도중에 지연 생성하면 부분 집합이 세션 내내 캐시될 수 있었다.
                self._copyright_vals = set(self._copyright.values())
                # 실패해도 표시 — 자산이 없을 때 조회마다 재적재하지 않게.
                self._loaded = True

    def _load_lines(self, asset: TagAsset) -> set[str]:
        try:
            return {n for value in self._database.read_lines(asset) if (n := _norm(value))}
        except Exception as exc:
            print(f"[TagIntel] {asset.value} load failed: {exc}")
            return set()

    def _load_all(self):
        """모든 자산 적재 — _ensure 가 락 안에서 한 번만 부른다."""
        # 1) Korean category/count catalog.
        try:
            df = self._database.read_parquet(TagAsset.KOREAN_TAG_CATALOG)
            cats = df["category"] if "category" in df.columns else [None] * len(df)
            cnts = df["count"] if "count" in df.columns else [0] * len(df)
            for tag, cat, cnt in zip(df["tag"], cats, cnts):
                n = _norm(str(tag))
                if not n:
                    continue
                self._cat[n] = str(cat) if cat is not None and cat == cat else ""
                try:
                    self._count[n] = int(cnt)
                except (ValueError, TypeError):
                    self._count[n] = 0
        except Exception as exc:
            print(f"[TagIntel] Korean tag catalog load failed: {exc}")

        # 2) Rating distribution.
        try:
            data = self._database.read_json(TagAsset.RATING_COUNTS)
            if isinstance(data, dict):
                self._totals = (data.get("_meta") or {}).get("total_posts")
                for key, values in data.items():
                    if key != "_meta" and isinstance(values, list) and len(values) == 4:
                        self._rating[_norm(key)] = tuple(int(value) for value in values)
        except Exception as exc:
            print(f"[TagIntel] rating counts load failed: {exc}")

        # 3) Keep curated and extended lexicons distinct, then expose their union.
        self._clothes_curated = self._load_lines(TagAsset.CLOTHING_TAGS_CURATED)
        self._clothes_extended = self._load_lines(TagAsset.CLOTHING_TAGS_EXTENDED)
        self._charac_curated = self._load_lines(TagAsset.APPEARANCE_TAGS_CURATED)
        self._charac_extended = self._load_lines(TagAsset.APPEARANCE_TAGS_EXTENDED)
        self._colors_curated = self._load_lines(TagAsset.COLOR_TERMS_CURATED)
        self._colors_extended = self._load_lines(TagAsset.COLOR_TERMS_EXTENDED)
        self._clothes = self._clothes_curated | self._clothes_extended
        self._charac = self._charac_curated | self._charac_extended
        self._colors = self._colors_curated | self._colors_extended

        # 4) Clothing region mapping also acts as a high-confidence whitelist.
        try:
            region_data = self._database.read_json(TagAsset.CLOTHING_REGIONS)
            if isinstance(region_data, dict):
                for region, tags in (region_data.get("regions") or {}).items():
                    for tag in tags:
                        n = _norm(tag)
                        if n:
                            self._regions[n] = region
                            self._clothes.add(n)
        except Exception as exc:
            print(f"[TagIntel] clothing regions load failed: {exc}")

        # 5) Character -> series, in confidence order.  Later sources only fill gaps.
        self._load_character_series()

        # 6) Category dictionaries used by split toggles.
        self._expression = self._load_tagset(TagAsset.EXPRESSION_TAGS)
        self._location = self._load_tagset(TagAsset.LOCATION_TAGS)
        self._pose = self._load_tagset(TagAsset.POSE_ACTION_TAGS)
        self._object = self._load_tagset(TagAsset.OBJECT_TAGS)
        self._meta = self._load_tagset(TagAsset.META_TAGS)

        # 7) Official Wiki group members are also known tags, even when a
        # separate catalog snapshot does not contain them.
        try:
            self._group_tags = {
                normalized
                for tag in self._database.all_group_tags()
                if (normalized := _norm(tag))
            }
        except Exception as exc:
            print(f"[TagIntel] tag group load failed: {exc}")

        # (활성 implication 표는 올리지 않는다 — 쓰던 remove_redundant_subtags 가 호출자 없던
        #  refineToSpecificTags 슬롯과 함께 은퇴했다(감사 #135). 상위 태그 추론은 TagClassifier 가
        #  TagDatabase.load_active_implications 로 따로 읽는다.)

        print(f"[TagIntel] KR태그 {len(self._cat):,} · 레이팅 {len(self._rating):,} · "
              f"의류 {len(self._clothes):,} · 특징 {len(self._charac):,} · 색상 {len(self._colors):,} · "
              f"region {len(self._regions):,} · copyright {len(self._copyright):,} · "
              f"표정 {len(self._expression):,} · 장소 {len(self._location):,} · "
              f"포즈 {len(self._pose):,} · 사물 {len(self._object):,} · 메타 {len(self._meta):,}")

    def _load_character_series(self) -> None:
        """Build character-series lookup using profile, curated, then extended data."""
        try:
            profiles = self._database.read_json(TagAsset.CHARACTER_PROFILES)
            if isinstance(profiles, list):
                for entry in profiles:
                    if not isinstance(entry, dict):
                        continue
                    tag = _norm(entry.get("tag") or "")
                    copyright_tag = _norm(entry.get("copyright") or "")
                    if tag and copyright_tag and tag not in self._copyright:
                        self._copyright[tag] = copyright_tag
        except Exception as exc:
            print(f"[TagIntel] character profiles load failed: {exc}")

        try:
            curated = self._database.read_json(TagAsset.CURATED_CHARACTER_SERIES)
            candidates: dict[str, set[str]] = {}
            if isinstance(curated, dict):
                for series, groups in curated.items():
                    if str(series).startswith("_") or not isinstance(groups, dict):
                        continue
                    normalized_series = _norm(series)
                    for gender in ("girl", "boy", "other"):
                        for character in groups.get(gender) or []:
                            if not isinstance(character, dict):
                                continue
                            names = [character.get("name", ""), *(character.get("aliases") or [])]
                            for name in names:
                                key = _norm(name)
                                if key and normalized_series:
                                    candidates.setdefault(key, set()).add(normalized_series)
            for key, series_values in candidates.items():
                if key not in self._copyright and len(series_values) == 1:
                    self._copyright[key] = next(iter(series_values))
        except Exception as exc:
            print(f"[TagIntel] curated character series load failed: {exc}")

        try:
            extended = self._database.read_parquet(TagAsset.EXTENDED_CHARACTER_SERIES)
            required = {"character", "copyright"}
            missing = required - set(extended.columns)
            if missing:
                raise ValueError(f"missing columns: {', '.join(sorted(missing))}")
            for character, copyright_tag in extended[["character", "copyright"]].itertuples(
                index=False,
                name=None,
            ):
                key = _norm(character)
                value = _norm(copyright_tag)
                if key and value and key not in self._copyright:
                    self._copyright[key] = value
        except Exception as exc:
            print(f"[TagIntel] extended character series load failed: {exc}")

    def _load_tagset(self, asset: TagAsset) -> set:
        """다양한 형태(tags / modifiers / groups / categories)의 JSON에서 태그 집합 추출."""
        out: set[str] = set()
        try:
            data = self._database.read_json(asset)
        except Exception as exc:
            print(f"[TagIntel] {asset.value} load failed: {exc}")
            return out

        def _collect(v):
            if isinstance(v, str):
                n = _norm(v)
                if n:
                    out.add(n)
            elif isinstance(v, list):
                for x in v:
                    _collect(x)
            elif isinstance(v, dict):
                for k, x in v.items():
                    if k in ("version", "description"):
                        continue
                    _collect(x)

        # Collect every data branch so new top-level buckets such as
        # ``uncategorized`` are not silently discarded.  Metadata keys are
        # excluded by _collect above.
        _collect(data)
        return out

    # ── 분류 ──
    def category_of(self, tag: str) -> str:
        self._ensure()
        return self._cat.get(_norm(tag), "")

    def top_category(self, tag: str) -> str:
        c = self.category_of(tag)
        return c.split(">")[0].strip() if c else ""

    def is_clothing(self, tag: str) -> bool:
        self._ensure()
        n = _norm(tag)
        if n in self._clothes:
            return True
        c = self._cat.get(n, "")
        return c.startswith("패션") and "헤어" not in c or "의류" in c or "의상" in c

    def is_appearance(self, tag: str) -> bool:
        """외견(신체/머리 등) 특징 태그인지."""
        self._ensure()
        n = _norm(tag)
        if n in self._charac:
            return True
        c = self._cat.get(n, "")
        return c.startswith("신체") or c.startswith("패션 > 헤어")

    def is_color(self, tag: str) -> bool:
        self._ensure()
        n = _norm(tag)
        if n in self._colors:
            return True
        return bool(set(n.split()) & self._colors)

    # ── 노이즈 / 검증 ──
    def is_known(self, tag: str) -> bool:
        self._ensure()
        n = _norm(tag)
        return (n in _PROMPT_ALLOW or n in self._cat or n in self._rating or
                n in self._clothes or n in self._charac or n in self._colors or
                n in self._expression or n in self._location or n in self._pose or
                n in self._object or n in self._meta or n in self._group_tags or
                n in self._copyright or n in self._copyright_vals)

    def tag_freq(self, tag: str) -> int:
        self._ensure()
        n = _norm(tag)
        if n in self._count:
            return self._count[n]
        r = self._rating.get(n)
        return sum(r) if r else 0

    def filter_noise(self, tags, min_count: int = 0, drop_unknown: bool = True):
        """리스트에서 가짜(미등록)/저빈도 태그 제거. Returns (kept, dropped)."""
        self._ensure()
        kept, dropped = [], []
        for t in tags:
            n = _norm(t)
            if not n:
                continue
            if drop_unknown and not self.is_known(t):
                dropped.append(t)
                continue
            if min_count and self.tag_freq(t) < min_count:
                dropped.append(t)
                continue
            kept.append(t)
        return kept, dropped

    # ── NSFW / 레이팅 ──
    def nsfw_ratio(self, tag: str):
        """questionable+explicit 비중 (0~1). 데이터 없으면 None."""
        self._ensure()
        r = self._rating.get(_norm(tag))
        if not r:
            return None
        tot = sum(r)
        if not tot:
            return None
        g, s, q, e = r
        return (q + e) / tot

    def is_nsfw(self, tag: str, threshold: float = 0.6) -> bool:
        ratio = self.nsfw_ratio(tag)
        return ratio is not None and ratio >= threshold

    # ── ④ 의류 region (부위) ──
    def region_of(self, tag: str) -> str:
        """의류 태그의 부위(region) 키. 매핑 없으면 ''."""
        self._ensure()
        return self._regions.get(_norm(tag), "")

    def region_label(self, region: str) -> str:
        return REGION_LABELS.get(region, region)

    # ── ③ copyright (캐릭터 → 시리즈) ──
    def copyright_of(self, character: str) -> str:
        """캐릭터명/별칭 → 대표 copyright(시리즈) 태그. 없으면 ''."""
        self._ensure()
        return self._copyright.get(_norm(character), "")

    def is_character(self, tag: str) -> bool:
        """알려진 캐릭터 프로필/작품 매핑에 등록된 캐릭터 태그인지."""
        self._ensure()
        return _norm(tag) in self._copyright

    def is_copyright(self, tag: str) -> bool:
        """알려진 copyright(시리즈) 태그인지."""
        self._ensure()
        return _norm(tag) in self._copyright_vals

    # ── ⑤ 카테고리 판별 ──
    def is_expression(self, tag: str) -> bool:
        self._ensure()
        return _norm(tag) in self._expression

    def is_location(self, tag: str) -> bool:
        self._ensure()
        return _norm(tag) in self._location

    def is_pose(self, tag: str) -> bool:
        self._ensure()
        return _norm(tag) in self._pose

    def is_object(self, tag: str) -> bool:
        self._ensure()
        return _norm(tag) in self._object

    def is_meta(self, tag: str) -> bool:
        self._ensure()
        return _norm(tag) in self._meta

    def _cat_check(self, cat: str):
        return {
            "expression": self.is_expression, "location": self.is_location,
            "pose": self.is_pose, "object": self.is_object, "meta": self.is_meta,
            "clothing": self.is_clothing, "appearance": self.is_appearance,
            "color": self.is_color,
        }.get(cat, lambda t: False)

    def tag_group(self, tag: str) -> str:
        """단일 카테고리 분류(우선순위). UI 색상/그룹용."""
        self._ensure()
        n = _norm(tag)
        if n in _PROMPT_ALLOW:
            return "quality"
        # 표정/포즈/장소/사물이 외견 사전의 오분류보다 우선 (예: 'sitting'이 특징DB에 섞여있음)
        for cat in ("meta", "clothing", "expression", "pose",
                    "location", "object", "appearance", "color"):
            if self._cat_check(cat)(n):
                return cat
        return "other"

    def split_by_categories(self, tags, categories):
        """태그를 지정 카테고리로 분리. 각 태그는 categories 순서상 첫 매칭으로 들어감.
        Returns {'rest': [...], 'groups': {cat: [...]}}."""
        self._ensure()
        groups = {c: [] for c in categories}
        checks = {c: self._cat_check(c) for c in categories}
        rest = []
        for t in tags:
            for c in categories:
                if checks[c](t):
                    groups[c].append(t)
                    break
            else:
                rest.append(t)
        return {"rest": rest, "groups": groups}


_instance = None
_instance_lock = threading.Lock()


def get_tag_intelligence() -> TagIntelligence:
    """프로세스 싱글턴 — 여러 스레드가 동시에 불러도 한 벌만 만든다(이중 확인)."""
    global _instance
    instance = _instance
    if instance is not None:
        return instance
    with _instance_lock:
        if _instance is None:
            _instance = TagIntelligence()
        return _instance
