# utils/character_features.py
"""캐릭터 특징 조회 유틸리티.

메인 프로필과 보충 특징은 ``TagDatabase``가 관리하는 정식 자산에서 읽는다.
"""
import re
import threading

from core.tag_database import TagAsset, get_tag_database

# ── 의상/액세서리 키워드 (word-level 매칭) ──
# 태그를 단어로 분리한 뒤, 이 집합과 교집합이 있으면 의상으로 분류
_COSTUME_WORDS: set[str] = {
    # 상의
    "shirt", "blouse", "sweater", "hoodie", "cardigan", "vest", "jacket",
    "coat", "blazer", "tunic", "camisole", "bustier", "corset", "crop",
    # 하의
    "skirt", "pants", "shorts", "jeans", "trousers", "leggings",
    # 원피스/전신
    "dress", "gown", "robe", "kimono", "yukata", "uniform", "suit",
    "leotard", "bodysuit", "jumpsuit", "overalls", "toga", "bikini",
    "swimsuit", "nightgown", "pajamas", "costume", "outfit", "clothes",
    "clothing", "garment", "attire", "hanfu", "cheongsam", "qipao",
    # 갑옷/군복
    "armor", "armour", "cape", "cloak", "tabard", "pauldrons", "greaves",
    "breastplate", "vambraces", "cuirass", "chainmail",
    # 속옷
    "bra", "panties", "underwear", "lingerie", "thong",
    # 다리/발
    "stockings", "thighhighs", "kneehighs", "socks", "tights", "pantyhose",
    "legwear", "garter", "boots", "shoes", "sandals", "heels", "slippers",
    "sneakers", "loafers", "footwear", "pumps", "tabi",
    # 손/팔
    "gloves", "gauntlets", "mittens", "cuffs", "warmers", "wraps",
    # 머리 장식(액세서리)
    "hat", "cap", "crown", "tiara", "helmet", "beret", "headwear",
    "hood", "veil", "headpiece", "headband", "hairband", "hairpin",
    "hairclip", "headphones", "headdress",
    # 목 장식
    "choker", "collar", "necklace", "necktie", "scarf", "bowtie",
    "neckerchief", "cravat", "ascot",
    # 귀/보석
    "earrings", "bracelet", "anklet", "pendant", "brooch", "amulet",
    # 허리
    "belt", "sash", "obi",
    # 눈 액세서리
    "glasses", "sunglasses", "goggles", "eyepatch", "monocle", "eyewear",
    # 가방/소지품
    "bag", "purse", "backpack", "handbag",
    # 기타 액세서리
    "mask", "apron", "ribbon", "bow", "sleeves", "ornament",
    "frills", "lace", "epaulettes", "armband", "wristband",
    # 무기/장비
    "sword", "gun", "shield", "wand", "staff", "weapon", "spear",
    "axe", "dagger", "lance", "halberd", "pistol", "rifle", "scythe",
    # 일본 전통
    "hakama", "haori", "fundoshi", "sarashi",
}


def _is_costume_tag(tag: str) -> bool:
    """태그가 의상/액세서리/장비인지 판별.
    1순위: NAIA 의류 사전(11k) + KR 카테고리(정확), 폴백: word-level 휴리스틱."""
    try:
        from core.tag_intelligence import get_tag_intelligence
        ti = get_tag_intelligence()
        if ti.is_clothing(tag):
            return True
        if ti.is_appearance(tag):   # 눈/머리/신체 → 의상 아님(확정)
            return False
    except Exception:
        pass
    words = set(tag.strip().lower().replace("_", " ").split())
    return bool(words & _COSTUME_WORDS)


# ── 눈/머리 충돌 판정 (자동화 특징 자동추가 시 'auto remove' / override 용) ──
def _norm_tag(tag: str) -> str:
    return (tag or "").strip().lower().replace("_", " ").replace(r"\(", "(").replace(r"\)", ")")


# 눈 색을 가리는 상태 태그 — 이게 프롬프트에 있으면 캐릭터 눈색 특징을 추가하지 않음
# (정책: 완전히 감은 'closed eyes'만 차단; 'one eye closed'/'half-closed eyes'는 색이 보일 수 있어 허용)
_EYE_COLOR_HIDING: set[str] = {
    "closed eyes",
}
# 눈 색 단어 (색 + eyes 조합이면 눈색 특징으로 간주)
_EYE_COLOR_WORDS: set[str] = {
    "aqua", "black", "blue", "brown", "green", "grey", "gray", "orange",
    "pink", "purple", "red", "white", "yellow", "violet", "amber",
    "gold", "golden", "silver", "light", "dark", "pale",
}
# 색 자체가 의미인 눈 관련 특징 (eyes 단어 없이도)
_EYE_COLOR_SPECIAL: set[str] = {
    "heterochromia", "multicolored eyes",
}
# 머리 길이 카테고리 (서로 배타적 — 하나만 가능)
_HAIR_LENGTH_TAGS: set[str] = {
    "very short hair", "short hair", "medium hair", "long hair",
    "very long hair", "absurdly long hair", "bald",
}


def is_eye_color_tag(tag: str) -> bool:
    """'blue eyes' 같은 눈 색 지정 특징인지. 'closed eyes' 등 상태 태그는 False."""
    n = _norm_tag(tag)
    if not n or n in _EYE_COLOR_HIDING:
        return False
    if n in _EYE_COLOR_SPECIAL:
        return True
    if not n.endswith(" eyes") and n != "eyes":
        return False
    return bool(set(n.split()) & _EYE_COLOR_WORDS)


def is_eye_color_hider(tag: str) -> bool:
    """이 태그가 눈 색을 가리는 상태 태그인지 (closed eyes 등)."""
    return _norm_tag(tag) in _EYE_COLOR_HIDING


def is_hair_length_tag(tag: str) -> bool:
    """머리 길이 카테고리 태그인지 (short/long hair 등)."""
    return _norm_tag(tag) in _HAIR_LENGTH_TAGS


# ── 핵심 외형 분류 (색 + 고정 신체특징) — 나머지는 '기타' ──
_HAIR_COLOR_WORDS: set[str] = {
    "aqua", "black", "blonde", "blue", "brown", "green", "grey", "gray",
    "orange", "pink", "purple", "red", "white", "silver", "light", "dark",
    "multicolored", "two-tone", "gradient", "streaked", "rainbow", "colored",
}
_HAIR_COLOR_SPECIAL: set[str] = {
    "multicolored hair", "two-tone hair", "gradient hair", "streaked hair",
    "rainbow hair", "colored inner hair", "split-color hair",
}
# 종족/특수 신체 (고정 외형 — 헤어스타일·일반 체형 제외)
_SPECIES_WORDS: set[str] = {
    "elf", "demon", "monster", "dragon", "robot", "android", "cyborg",
    "mermaid", "merfolk", "angel", "vampire", "oni", "kitsune", "succubus",
    "lamia", "harpy", "slime", "ghost", "fairy", "centaur", "orc", "goblin",
    "dullahan", "yokai", "youkai",
}
# 색·종족 외에 핵심으로 남길 신체 특징 (정확 일치)
_PHYS_EXACT: set[str] = {
    "pointy ears", "third eye", "extra eyes", "glowing eyes", "empty eyes",
    "no pupils", "slit pupils", "constricted pupils", "ringed eyes",
    "sharp teeth", "fangs", "fang", "halo", "claws", "tusks", "freckles",
    "tan", "tanlines", "dark-skinned female", "dark-skinned male",
}


def is_hair_color_tag(tag: str) -> bool:
    """'blue hair' 같은 머리 색 태그인지 (길이/스타일은 False)."""
    n = _norm_tag(tag)
    if not n:
        return False
    if n in _HAIR_COLOR_SPECIAL:
        return True
    if is_hair_length_tag(tag) or not n.endswith(" hair"):
        return False
    return bool(set(n.split()) & _HAIR_COLOR_WORDS)


def is_core_appearance_tag(tag: str) -> bool:
    """핵심 외형 = 머리색 + 눈색 + 고정 신체특징(귀/뿔/꼬리/날개/종족/특수동공/점·흉터/피부 등).
    헤어스타일(long/twintails), 일반 체형(breasts/navel), 포즈(hand up),
    주관표현(bishounen)은 False → '기타'로 분류된다."""
    if is_hair_color_tag(tag) or is_eye_color_tag(tag):
        return True
    n = _norm_tag(tag)
    if not n or is_hair_length_tag(tag):
        return False
    if n in _PHYS_EXACT:
        return True
    words = set(n.split())
    if "ears" in words and "earrings" not in words:   # 귀(귀걸이 제외)
        return True
    if "horns" in words or "horn" in words:           # 뿔
        return True
    if "tail" in words or "tails" in words:            # 꼬리
        return True
    if "wings" in words or "wing" in words:            # 날개
        return True
    if "pupils" in words:                              # 특수 동공
        return True
    if "mole" in words or "scar" in n or "skin" in words:   # 점/흉터/피부
        return True
    if words & _SPECIES_WORDS:                         # 종족
        return True
    return False


# ── 보조 특징 (수동 선별 외형 태그) ──
_AUX_SET = None


def _load_aux_set() -> set:
    """수동 선별 외형 태그를 normalize해서 로드 (lazy, 1회)."""
    global _AUX_SET
    if _AUX_SET is not None:
        return _AUX_SET
    _AUX_SET = set()
    try:
        for line in get_tag_database().read_lines(TagAsset.APPEARANCE_TAGS_CURATED):
            t = _norm_tag(line)
            if t:
                _AUX_SET.add(t)
    except Exception:
        pass
    return _AUX_SET


def is_aux_feature_tag(tag: str) -> bool:
    """보조 특징 — 수동 선별 목록에 등재된 외형 특성(ahoge, 헤어스타일, 체형, 피부 등).
    (색/핵심 신체특징은 lookup_core가 먼저 가져가므로 lookup_aux에선 core 제외)"""
    return _norm_tag(tag) in _load_aux_set()


class CharacterFeatureLookup:
    """캐릭터 이름 → 핵심/의상 특징 분리 조회 (lazy loading, singleton)"""

    def __init__(self):
        self.database = get_tag_database()

        # 핵심 특징 (character_profiles)
        # (프로필의 copyright·gender 는 싣지 않는다 — 읽는 곳이 없어 3만여 항목 사전만 메모리에
        #  남았다. 작품명은 core.tag_intelligence.copyright_of 가 맡는다.)
        self._core_dict: dict[str, list[str]] | None = None   # name → core_tags
        self._post_count: dict[str, int] = {}                   # name → post_count

        # 전체 특징 (character_features parquet)
        self._full_dict: dict[str, str] | None = None           # name → features_str
        self._full_count: dict[str, int] | None = None

        # 인덱스
        self._norm_index: dict[str, str] = {}   # normalized → original key
        self._short_index: dict[str, str] = {}  # 괄호 제거 → original key

        # 적재 직렬화 — 여러 스레드가 첫 조회를 동시에 해도 한 번만 적재하고, 적재 중인
        # 반쪽 사전을 누구도 조회하지 않게 한다(_loaded 는 적재가 끝난 뒤에만 참).
        self._loaded = False
        self._load_lock = threading.RLock()

    def _ensure_loaded(self):
        """첫 호출 시 데이터 로드"""
        if self._loaded:
            return
        with self._load_lock:
            if self._loaded:
                return
            try:
                self._load_all()
            finally:
                # 실패해도 표시 — 자산이 없을 때 조회마다 재적재하지 않게.
                self._loaded = True

    def _load_all(self):
        """모든 자산 적재 — _ensure_loaded 가 락 안에서 한 번만 부른다."""
        self._core_dict = {}
        self._full_dict = {}
        self._full_count = {}

        # 1. 캐릭터 프로필 로드 (메인 — 핵심 특징)
        try:
            data = self.database.read_json(TagAsset.CHARACTER_PROFILES)
            for entry in data:
                tag = entry.get("tag", "")
                name = tag.replace("_", " ").strip().lower()
                if not name:
                    continue
                core_tags = [t.replace("_", " ") for t in entry.get("core_tags", [])]
                self._core_dict[name] = core_tags
                self._post_count[name] = entry.get("post_count", 0)
            print(f"[CharacterFeatures] profiles loaded: {len(self._core_dict):,}")
        except Exception as e:
            print(f"[CharacterFeatures] profile load failed: {e}")

        # 2. 전체 특징 Parquet 로드 (보충 — 의상 포함 전체)
        self._full_norm_to_key: dict[str, str] = {}  # normalized → original key
        try:
            frame = self.database.read_parquet(
                TagAsset.CHARACTER_FEATURES,
                columns=["character", "features", "post_count"],
            )
            for character, features, post_count in frame.itertuples(index=False, name=None):
                key = str(character).strip()
                if not key:
                    continue
                self._full_dict[key] = str(features or "")
                try:
                    self._full_count[key] = int(post_count or 0)
                except (TypeError, ValueError):
                    self._full_count[key] = 0
                self._full_norm_to_key[key.lower().replace("_", " ")] = key
            print(f"[CharacterFeatures] full features loaded: {len(self._full_dict):,}")
        except Exception as e:
            print(f"[CharacterFeatures] full feature load failed: {e}")

        # 3. 통합 카운트 인덱스 (정규화 키 → count)
        self._count_index: dict[str, int] = {}
        for key, count in self._post_count.items():
            self._count_index[key] = count
        if self._full_count:
            for k, v in self._full_count.items():
                norm_k = k.strip().lower().replace("_", " ")
                if norm_k not in self._count_index:
                    self._count_index[norm_k] = v

        # 4. 이름 인덱스 빌드 (양쪽 소스 병합)
        all_keys = set()
        if self._core_dict:
            all_keys.update(self._core_dict.keys())
        if self._full_dict:
            all_keys.update(k.strip().lower().replace("_", " ") for k in self._full_dict.keys())

        paren_re = re.compile(r'\s*\([^)]*\)\s*$')
        for key in all_keys:
            norm = key.strip().lower().replace("_", " ")
            if norm not in self._norm_index:
                self._norm_index[norm] = key
            short = paren_re.sub("", norm).strip()
            if short and short != norm and short not in self._short_index:
                self._short_index[short] = key

    @staticmethod
    def _normalize(name: str) -> str:
        return name.strip().lower().replace("_", " ")

    def _resolve_key(self, name: str) -> str | None:
        """이름 → 정규화 키 해석"""
        norm = self._normalize(name)
        if norm in self._norm_index:
            return self._norm_index[norm]
        if norm in self._short_index:
            return self._short_index[norm]
        return None

    def _get_full_features_for(self, key: str) -> str:
        """전체 특징 Parquet에서 특징 문자열 가져오기 (O(1))."""
        if not self._full_dict:
            return ""
        # 직접 키 매칭
        if key in self._full_dict:
            return self._full_dict[key]
        # 정규화 인덱스로 O(1) 조회
        orig = self._full_norm_to_key.get(key, "")
        return self._full_dict.get(orig, "") if orig else ""

    def _all_feature_tags(self, key: str) -> list[str]:
        """프로필 핵심 태그와 전체 특징을 합쳐 중복을 제거한다."""
        tags: list[str] = []
        seen: set[str] = set()
        for t in self._core_dict.get(key, []):
            ts = t.strip()
            tn = ts.lower()
            if ts and tn not in seen and tn != key:
                tags.append(ts)
                seen.add(tn)
        full_str = self._get_full_features_for(key)
        if full_str:
            for t in full_str.split(","):
                ts = t.strip()
                tn = ts.lower()
                if ts and tn not in seen and tn != key:
                    tags.append(ts)
                    seen.add(tn)
        return tags

    def _count_for(self, key: str) -> int:
        return self._post_count.get(key, 0) or self._count_index.get(key, 0)

    def lookup_core(self, name: str) -> tuple[str, int] | None:
        """핵심(비의상) 특징 조회 — 전체 특징에서 _is_costume_tag 인 것을 제외.
        (의상/머리장식 등은 lookup_costume으로 분리되어 모달의 '의상' 섹션에 표시됨)"""
        self._ensure_loaded()
        key = self._resolve_key(name)
        if key is None:
            return None
        core_tags = [t for t in self._all_feature_tags(key)
                     if not _is_costume_tag(t) and is_core_appearance_tag(t)]
        if not core_tags:
            return None
        return (", ".join(core_tags), self._count_for(key))

    def lookup_aux(self, name: str) -> tuple[str, int] | None:
        """보조 특징 — 수동 선별 목록에 등재됐고 비의상·비핵심인 외형 특성.
        헤어스타일(long/twintails/ahoge), 체형(breasts), 피부, 헤어 디테일 등."""
        self._ensure_loaded()
        key = self._resolve_key(name)
        if key is None:
            return None
        aux_tags = [t for t in self._all_feature_tags(key)
                    if not _is_costume_tag(t) and not is_core_appearance_tag(t)
                    and is_aux_feature_tag(t)]
        if not aux_tags:
            return None
        return (", ".join(aux_tags), self._count_for(key))

    def lookup_etc(self, name: str) -> tuple[str, int] | None:
        """기타 특징 — 비의상·비핵심·비보조(목록 밖)인 것.
        포즈(hand up/standing), 표정(smile), 동작(holding), 주관표현(bishounen) 등."""
        self._ensure_loaded()
        key = self._resolve_key(name)
        if key is None:
            return None
        etc_tags = [t for t in self._all_feature_tags(key)
                    if not _is_costume_tag(t) and not is_core_appearance_tag(t)
                    and not is_aux_feature_tag(t)]
        if not etc_tags:
            return None
        return (", ".join(etc_tags), self._count_for(key))

    def lookup_costume(self, name: str) -> tuple[str, int] | None:
        """의상/액세서리 특징만 조회 — 전체 특징에서 _is_costume_tag 인 것만."""
        self._ensure_loaded()
        key = self._resolve_key(name)
        if key is None:
            return None
        costume_tags = [t for t in self._all_feature_tags(key) if _is_costume_tag(t)]
        if not costume_tags:
            return None
        return (", ".join(costume_tags), self._count_for(key))

    def lookup(self, name: str) -> tuple[str, int] | None:
        """전체 특징 조회 (기존 호환). full 우선, 없으면 core."""
        self._ensure_loaded()
        key = self._resolve_key(name)
        if key is None:
            return None

        # full에서 가져오기
        full_str = self._get_full_features_for(key)
        if full_str:
            count = self._count_index.get(key, 0)
            return (full_str, count)

        # core만 있는 경우
        core_tags = self._core_dict.get(key, [])
        if core_tags:
            count = self._post_count.get(key, 0)
            return (", ".join(core_tags), count)

        return None

    def search(self, query: str, limit: int = 50) -> list[tuple[str, str, int]]:
        """캐릭터 이름 검색 (2단계 최적화).
        Phase 1: 키 매칭 + 우선순위/카운트만 (O(1) 카운트 조회)
        Phase 2: 상위 limit개만 feature lookup (무거운 연산 최소화)
        """
        self._ensure_loaded()
        if not query.strip():
            return []

        q = self._normalize(query)

        # Phase 1: 빠른 키 매칭 (feature lookup 없이)
        candidates: list[tuple[str, str, int, int]] = []  # (orig_key, norm_key, count, priority)
        for norm_key, orig_key in self._norm_index.items():
            if q not in norm_key:
                continue
            count = self._count_index.get(norm_key, 0)
            if norm_key == q:
                priority = 0
            elif norm_key.startswith(q):
                priority = 1
            else:
                priority = 2
            candidates.append((orig_key, norm_key, count, priority))

        # Phase 2: 정렬 후 상위 limit개만 feature lookup
        candidates.sort(key=lambda x: (x[3], -x[2]))
        results: list[tuple[str, str, int]] = []
        for orig_key, _, count, _ in candidates[:limit]:
            result = self.lookup(orig_key)
            features = result[0] if result else ""
            if result and result[1]:
                count = result[1]
            results.append((orig_key, features, count))

        return results

    def all_keys(self) -> list[str]:
        """모든 캐릭터 키 반환"""
        self._ensure_loaded()
        return list(self._norm_index.values())


_instance: CharacterFeatureLookup | None = None
_instance_lock = threading.Lock()


def get_character_features() -> CharacterFeatureLookup:
    """싱글턴 인스턴스 반환 — 여러 스레드가 동시에 불러도 한 벌만 만든다(이중 확인)."""
    global _instance
    instance = _instance
    if instance is not None:
        return instance
    with _instance_lock:
        if _instance is None:
            _instance = CharacterFeatureLookup()
        return _instance
