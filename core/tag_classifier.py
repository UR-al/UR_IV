"""Tag classification and filtering backed by the consolidated tag database."""

from __future__ import annotations

import threading
from collections import deque
from typing import Iterable

from core.tag_database import TagAsset, TagDatabase, get_tag_database


_CATEGORY_MAPPING = {
    "body_parts": {
        "body_parts", "ass", "breasts_tags", "hair", "hair_color",
        "hair_styles", "eyes_tags", "ears_tags", "hands", "legs", "feet",
        "shoulders", "neck_and_neckwear", "skin_color", "skin_folds", "bra",
    },
    "clothing": {
        "clothes_list", "dress", "attire", "shirt", "pants", "legwear",
        "sleeves", "headwear", "eyewear", "handwear", "covering", "mask",
        "fashion_style", "patterns", "embellishment", "panties", "sexual_attire",
        "piercings",
    },
    "pose": {
        "posture", "gestures", "sexual_positions", "dances",
        "verbs_and_gerunds",
    },
    "expression": {"face_tags"},
    "composition": {"focus_tags", "image_composition", "scan"},
    "background": {
        "backgrounds", "locations", "real_world_locations",
        "holidays_and_celebrations", "history",
    },
    "effect": {
        "lighting", "censorship", "metatags", "visual_novel_games", "year_tags", "water",
        "fire", "flowers", "symbols",
    },
    "objects": {
        "audio_tags", "food_tags", "weapons", "technology", "video_game",
        "board_games", "fighting_games", "platform_games", "role-playing_games",
        "shooter_games", "text", "prints", "tail", "wings", "cards", "sports",
    },
    "character_trait": {
        "characteristic_list", "family_relationships", "groups", "jobs", "subjective",
        "legendary_creatures", "people", "companies_and_brand_names",
    },
    "animals": {"birds", "cats", "dogs"},
    "art_style": {
        "fine_art_parody", "drawing_software", "japanese_dialects",
        "artistic_license", "phrases", "pixiv_projects",
    },
    "sexual": {
        "sex_acts", "sex_objects", "nudity", "pussy", "sexual_attire",
        "sexual_positions", "simulated_sex_acts",
    },
    "color": {"colors", "hair_color", "skin_color"},
}

_MIXED_GROUPS = {
    "ass": ("body_parts", "pose", "composition"),
    "breasts_tags": ("body_parts", "pose"),
    "pussy": ("body_parts", "sexual"),
    "metatags": ("effect", "composition"),
}

_CATEGORY_PRIORITY = (
    "sexual", "body_parts", "clothing", "pose", "expression",
    "character_trait", "composition", "background", "effect", "objects",
    "animals", "art_style", "color",
)

_DEFAULT_CENSORSHIP = {
    "censored", "mosaic censoring", "bar censoring", "blur censor",
    "light censoring", "novelty censoring", "heart censor", "steam censor",
    "convenient censoring", "censored nipples", "censored pussy",
    "censored penis", "mosaic_censoring", "bar_censoring", "light_censoring",
}


def _tag_key(tag: object) -> str:
    """Return the comparison form used by groups and implications."""
    return (
        str(tag or "")
        .strip()
        .lower()
        .replace(r"\(", "(")
        .replace(r"\)", ")")
        .replace("_", " ")
    )


def _group_key(group: object) -> str:
    value = str(group or "").strip().lower()
    if value.startswith("tag_group:"):
        value = value.split(":", 1)[1]
    if value.endswith(".parquet"):
        value = value[:-8]
    return value.replace(" ", "_")


def _variants(tag: object) -> set[str]:
    raw = str(tag or "").strip().lower()
    unescaped = raw.replace(r"\(", "(").replace(r"\)", ")")
    spaced = unescaped.replace("_", " ")
    underscored = unescaped.replace(" ", "_")
    escaped = unescaped.replace("(", r"\(").replace(")", r"\)")
    return {value for value in (raw, unescaped, spaced, underscored, escaped) if value}


def _variant_set(tags: Iterable[object]) -> set[str]:
    result: set[str] = set()
    for tag in tags:
        result.update(_variants(tag))
    return result


def _lower_set(tags: Iterable[object]) -> set[str]:
    return {str(tag).strip().lower() for tag in tags if str(tag).strip()}


class TagClassifier:
    """Classify prompt tags while hiding tag-asset layout from callers."""

    def __init__(self, database: TagDatabase | None = None):
        self._use_shared_tag_data = database is None
        self._database = database or get_tag_database()

        # Public sets retained for existing removal/filter callers.
        self.characters: set[str] = set()
        self.copyrights: set[str] = set()
        self.artists: set[str] = set()
        self.meta_tags: set[str] = set()
        self.clothes: set[str] = set()
        self.characteristics: set[str] = set()
        self.colors: set[str] = set()

        self.wiki_groups: dict[str, set[str]] = {}
        self.tag_to_category: dict[str, list[dict[str, str]]] = {}
        self.censorship_tags: set[str] = set()
        self.text_tags: set[str] = set()

        # Kept for compatibility with diagnostics that display the former folder.
        try:
            self.tags_db_dir = str(self._database.path(TagAsset.TAG_GROUPS).parent)
        except Exception:
            self.tags_db_dir = ""

        self._implications: dict[str, set[str]] = {}
        self._load_from_tag_data()
        self._load_manifest_meta_tags()
        self._load_text_files()
        self._load_wiki_groups()
        self._load_special_tags()

    def _load_from_tag_data(self):
        """Load the optimized Danbooru catalog, with curated database fallbacks."""
        if self._use_shared_tag_data:
            try:
                from utils.tag_data import get_tag_data

                # TagData logs through `logging` only, so no stdout redirection is
                # needed (redirect_stdout swaps a process-global and raced with the
                # completer warm-up thread).
                tag_data = get_tag_data()
                if tag_data.is_loaded:
                    # TagData already stores these as stripped, lower-cased, non-empty
                    # names (the exact `_lower_set` form), so share them read-only
                    # instead of copying ~740k strings per classifier instance.
                    self.characters = tag_data.character_set
                    self.copyrights = tag_data.copyright_set
                    self.artists = tag_data.artist_set
                    # meta_tags is extended in place by _load_manifest_meta_tags —
                    # copy it so the shared TagData set is never mutated.
                    self.meta_tags = set(tag_data.meta_set)
                    print(
                        f"[TagClassifier] TagData: characters={len(self.characters):,}, "
                        f"copyrights={len(self.copyrights):,}, artists={len(self.artists):,}, "
                        f"meta={len(self.meta_tags):,}"
                    )
                    return
            except Exception as exc:
                print(f"[TagClassifier] TagData load failed: {exc}")

        self._load_curated_name_fallbacks()

    def _load_curated_name_fallbacks(self) -> None:
        """Recover high-confidence names without treating broad catalogs as groups."""
        characters: list[object] = []
        copyrights: list[object] = []
        try:
            profiles = self._database.read_json(TagAsset.CHARACTER_PROFILES)
            if isinstance(profiles, list):
                for entry in profiles:
                    if not isinstance(entry, dict):
                        continue
                    characters.append(entry.get("tag", ""))
                    copyrights.append(entry.get("copyright", ""))
        except Exception as exc:
            print(f"[TagClassifier] character profile fallback failed: {exc}")

        try:
            curated = self._database.read_json(TagAsset.CURATED_CHARACTER_SERIES)
            if isinstance(curated, dict):
                for series, groups in curated.items():
                    if str(series).startswith("_") or not isinstance(groups, dict):
                        continue
                    copyrights.append(series)
                    for gender in ("girl", "boy", "other"):
                        for character in groups.get(gender) or []:
                            if isinstance(character, dict):
                                characters.append(character.get("name", ""))
                                characters.extend(character.get("aliases") or [])
        except Exception as exc:
            print(f"[TagClassifier] curated series fallback failed: {exc}")

        try:
            extended = self._database.read_parquet(
                TagAsset.EXTENDED_CHARACTER_SERIES,
                columns=["character", "copyright"],
            )
            characters.extend(extended["character"].dropna().tolist())
            copyrights.extend(extended["copyright"].dropna().tolist())
        except Exception as exc:
            print(f"[TagClassifier] extended series fallback failed: {exc}")

        self.characters = _lower_set(characters)
        self.copyrights = _lower_set(copyrights)
        print(
            f"[TagClassifier] Curated name fallback: characters={len(self.characters):,}, "
            f"copyrights={len(self.copyrights):,}"
        )

    def _load_manifest_meta_tags(self) -> None:
        """Augment TagData with the curated UI meta-tag catalog."""
        try:
            data = self._database.read_json(TagAsset.META_TAGS)
            if isinstance(data, dict):
                self.meta_tags.update(_lower_set(data.get("tags") or []))
        except Exception as exc:
            print(f"[TagClassifier] meta tag catalog load failed: {exc}")

    def _load_text_files(self):
        """Load the small curated lists used as high-confidence fallbacks."""
        self.clothes = self._read_lines(TagAsset.CLOTHING_TAGS_CURATED)
        self.characteristics = self._read_lines(TagAsset.APPEARANCE_TAGS_CURATED)
        self.colors = self._read_lines(TagAsset.COLOR_TERMS_CURATED)

    def _read_lines(self, asset: TagAsset) -> set[str]:
        try:
            return _lower_set(self._database.read_lines(asset))
        except Exception as exc:
            print(f"[TagClassifier] {asset.value} load failed: {exc}")
            return set()

    def _load_wiki_groups(self):
        """Load only the consolidated group asset, never unrelated parquet catalogs."""
        try:
            groups = self._database.load_tag_groups()
        except Exception as exc:
            print(f"[TagClassifier] tag groups load failed: {exc}")
            groups = {}

        for raw_group, raw_tags in groups.items():
            group = _group_key(raw_group)
            if not group:
                continue
            tags = _lower_set(raw_tags)
            if not tags:
                continue
            self.wiki_groups.setdefault(group, set()).update(tags)
            categories = _MIXED_GROUPS.get(group, (self._find_category(group),))
            for tag in tags:
                entries = self.tag_to_category.setdefault(tag, [])
                for category in categories:
                    entry = {"group": group, "category": category}
                    if entry not in entries:
                        entries.append(entry)

        try:
            raw_implications = self._database.load_active_implications()
        except Exception as exc:
            print(f"[TagClassifier] tag implications load failed: {exc}")
            raw_implications = {}
        normalized_implications: dict[str, set[str]] = {}
        for raw_antecedent, raw_consequences in raw_implications.items():
            antecedent = _tag_key(raw_antecedent)
            consequences = {_tag_key(value) for value in raw_consequences if _tag_key(value)}
            if antecedent and consequences:
                normalized_implications.setdefault(antecedent, set()).update(consequences)
        self._implications = normalized_implications
        print(
            f"[TagClassifier] {len(self.wiki_groups)} tag groups, "
            f"{len(self.tag_to_category)} indexed forms loaded"
        )

    @staticmethod
    def _find_category(group_name: str) -> str:
        for category, groups in _CATEGORY_MAPPING.items():
            if group_name in groups:
                return category
        return "general"

    def _load_special_tags(self):
        """Derive censorship/text filters from the consolidated group asset."""
        self.censorship_tags = set(self.wiki_groups.get("censorship", set()))
        self.censorship_tags.update(_variant_set(_DEFAULT_CENSORSHIP))
        self.text_tags = set(self.wiki_groups.get("text", set()))
        print(f"[TagClassifier] censorship tags: {len(self.censorship_tags)}")
        print(f"[TagClassifier] text tags: {len(self.text_tags)}")

    def filter_tags(self, tags_list, remove_censorship=False, remove_text=False):
        """Filter tags while preserving their original spelling and order."""
        result = []
        for tag in tags_list:
            if remove_censorship and self.is_censorship_tag(tag):
                continue
            if remove_text and self.is_text_tag(tag):
                continue
            result.append(tag)
        return result

    def is_censorship_tag(self, tag):
        return any(value in self.censorship_tags for value in _variants(tag))

    def is_text_tag(self, tag):
        return any(value in self.text_tags for value in _variants(tag))

    def is_meta_tag(self, tag: str) -> bool:
        if any(value in self.meta_tags for value in _variants(tag)):
            return True
        return "art_style" in self._categories_for_tag(tag)

    def _direct_categories(self, tag: object) -> list[str]:
        categories: list[str] = []
        for variant in _variants(tag):
            for info in self.tag_to_category.get(variant, []):
                category = info.get("category", "general")
                if category not in categories:
                    categories.append(category)
        return categories

    def _categories_for_tag(self, tag: object) -> list[str]:
        """Find direct or nearest implied categories without crossing cycles."""
        direct = [category for category in self._direct_categories(tag) if category != "general"]
        if direct:
            return direct

        start = _tag_key(tag)
        if not start:
            return []
        seen = {start}
        frontier = deque(self._implications.get(start, set()))
        while frontier:
            level_size = len(frontier)
            inherited: list[str] = []
            next_level: set[str] = set()
            for _ in range(level_size):
                parent = frontier.popleft()
                if parent in seen:
                    continue
                seen.add(parent)
                for category in self._direct_categories(parent):
                    if category == "general":
                        continue
                    if category not in inherited:
                        inherited.append(category)
                next_level.update(self._implications.get(parent, set()) - seen)
            if inherited:
                return inherited
            frontier.extend(sorted(next_level))
        return []

    def classify_tag(self, tag):
        """Classify a tag, accepting both Danbooru underscores and display spaces."""
        variants = _variants(tag)
        if any(value in self.characters for value in variants):
            return "character"
        if any(value in self.copyrights for value in variants):
            return "copyright"
        if any(value in self.artists for value in variants):
            return "artist"

        categories = self._categories_for_tag(tag)
        for category in _CATEGORY_PRIORITY:
            if category in categories:
                return category
        if categories:
            return categories[0]

        if variants & self.clothes:
            return "clothing"
        if variants & self.characteristics:
            return "character_trait"

        words = set(_tag_key(tag).split())
        normalized_colors = {_tag_key(color) for color in self.colors}
        if words & normalized_colors:
            return "color"
        return "general"

    def classify_tags_for_event(self, tags_list):
        """Classify tags into the event-generation buckets used by the UI."""
        classified = {
            "count": [], "character": [], "copyright": [], "costume": [],
            "appearance": [], "expression": [], "action": [], "background": [],
            "composition": [], "effect": [], "objects": [], "general": [],
        }
        count_tags = {
            "1boy", "2boys", "3boys", "4boys", "5boys", "6+boys",
            "1girl", "2girls", "3girls", "4girls", "5girls", "6+girls",
            "1other", "2others", "3others", "4others", "5others", "6+others",
        }
        mapping = {
            "character": "character", "copyright": "copyright", "clothing": "costume",
            "body_parts": "appearance", "expression": "expression", "pose": "action",
            "background": "background", "composition": "composition", "effect": "effect",
            "objects": "objects",
        }
        for tag in tags_list:
            if _tag_key(tag).replace(" ", "_") in count_tags:
                classified["count"].append(tag)
                continue
            classified[mapping.get(self.classify_tag(tag), "general")].append(tag)
        return classified


# Process-wide shared classifier.  The main window (GeneratorBase.tag_classifier)
# and VueBridge used to build one each — every build re-reads the tag-group and
# implication parquets and indexes ~2.6k groups (~0.5 s on the GUI thread).
_shared_classifier: TagClassifier | None = None
_shared_classifier_lock = threading.Lock()


def get_tag_classifier() -> TagClassifier:
    """Return the shared default-database classifier, building it once."""
    global _shared_classifier
    instance = _shared_classifier
    if instance is not None:
        return instance
    with _shared_classifier_lock:
        if _shared_classifier is None:
            _shared_classifier = TagClassifier()
        return _shared_classifier


def reset_tag_classifier() -> None:
    """Drop the shared classifier (after the tag database has been refreshed)."""
    global _shared_classifier
    with _shared_classifier_lock:
        _shared_classifier = None
