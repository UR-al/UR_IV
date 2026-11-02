"""Wildcards of the Anima studio: the app's file syntax plus Dynamic Prompts' ``{a|b}``.

Pure Python (no ComfyUI / torch).  ``__name__`` and ``~/name/~`` read ``name.txt``: each line is a group, each group gives
one of its comma-separated options, ``__name:n__`` takes n groups at random, a chosen option may hold ``[a|b]`` and
other wildcards.  Same rules as the app's ``utils/file_wildcard.py`` (the pack cannot import it), except that ``[a|b]``
of the prompt itself stays: there it is Forge's prompt editing.  ``{a|b}`` picks one, ``{2::a|b}`` weighs it.

Everything is drawn from the image's seed, each prompt by itself, so a seed gives its prompts back whatever the others
hold.  Only prompts that run are drawn, and a ``<lora:...>`` tag is left alone.  Files come from ``<ComfyUI>/wildcards`` and
the ``wildcards:`` entries of ``extra_model_paths.yaml``; a name is never a path.
"""
from __future__ import annotations

import copy
import importlib
import random
import re
from pathlib import Path

from .studio_state import PROMPT_BOXES, assembled_prompt

SEED_RANGE = 4294967294  # xyz_plan.SEED_RANGE
MAX_DEPTH = 10
_NAME = r"[\w .:\-]+?"  # narrower than the app's: "o__o ^__^" must not read as a name
_FILE_TILDE = re.compile(rf"~/({_NAME})/~")
_FILE_DUNDER = re.compile(rf"__(?![\s_])({_NAME})(?<![\s_])__")
_FILE_PATTERNS = (_FILE_TILDE, _FILE_DUNDER)
_TAG = re.compile(r"(<[^<>\n]*>)")
_BRACES = re.compile(r"(?<!\\)\{([^{}]*\|[^{}]*?)(?<!\\)\}")
_OR = re.compile(r"\[([^\[\]]+?\|[^\[\]]+?)\]")
_WEIGHT = re.compile(r"^\s*(\d+(?:\.\d+)?)::(.*)$", re.S)
_BAD_NAME = re.compile(r'[\\/:*?"<>|\x00-\x1f]|^\.+$')
# As core/prompt_comments.py: a comment opens at the start, after white space, after a comma or right after a block comment.
_COMMENT = re.compile(r"(?:(?<![^\s,])|(?<=\*/))(?:/\*.*?\*/|/\*[^\r\n]*|//[^\r\n]*)", re.S)
_NEWLINE = re.compile(r"\r\n|\r|\n")  # not str.splitlines: it also cuts at \f and U+2028, as the app learned
# The prompts a run encodes besides the main two: studio_logic.js passPrompts.
_PASS_PROMPTS = (
    ("hires", "prompt"), ("hires", "negative"), ("sam3", "inpaint_prompt"), ("sam3", "negative_prompt"),
    ("ext", "sam3_refine", "inpaint_prompt"), ("ext", "sam3_refine", "negative_prompt"),
    ("ext", "sam3_tile", "inpaint_prompt"), ("ext", "sam3_tile", "negative_prompt"),
)


def _outside_tags(text: str, replace) -> str:
    parts = _TAG.split(text)
    parts[::2] = map(replace, parts[::2])
    return "".join(parts)


def has_wildcard(text: str) -> bool:
    if not text:
        return False
    plain = _TAG.sub("", text)
    return any(pattern.search(plain) for pattern in _FILE_PATTERNS) or bool(_BRACES.search(text))


def _split_name(raw: str) -> tuple[str, int]:
    """``name:n`` -> ``(name, n)``; no number, or one that is not a number, means every group (the app's rule)."""
    name, colon, count = raw.rpartition(":")
    if not colon:
        return raw.strip(), 0
    try:
        return name.strip(), int(count.strip())
    except ValueError:
        return name.strip(), 0


def _resolve_or(text: str, rng: random.Random) -> str:
    def pick(match):
        options = [option.strip() for option in match.group(1).split("|") if option.strip()]
        return rng.choice(options) if options else ""

    previous = None
    while previous != text:
        previous = text
        text = _OR.sub(pick, text)
    return text


def _file_replacer(rng: random.Random, lookup, missing: list):
    def replace(match):
        name, count = _split_name(match.group(1))
        groups = lookup(name)
        if not groups:
            if name not in missing:
                missing.append(name)
            return match.group(0)
        chosen = rng.sample(groups, count) if 0 < count < len(groups) else groups
        return ", ".join(_resolve_or(rng.choice(group), rng).strip() for group in chosen)

    return replace


def _brace_options(body: str) -> list:
    """``a|b``, with the bars inside ``[a|b]``, ``(...)`` and ``<...>`` left where they are."""
    options, depth, start = [], 0, 0
    for index, char in enumerate(body):
        if char in "[(<":
            depth += 1
        elif char in "])>":
            depth = max(0, depth - 1)
        elif char == "|" and depth == 0:
            options.append(body[start:index])
            start = index + 1
    return [*options, body[start:]]


def _pick_braced(match, rng: random.Random) -> str:
    weights, options = [], []
    for option in _brace_options(match.group(1)):
        weighted = _WEIGHT.match(option)
        weights.append(float(weighted.group(1)) if weighted else 1.0)
        options.append(weighted.group(2) if weighted else option)
    if not any(weights):
        return ""
    return rng.choices(options, weights=weights, k=1)[0].strip()


def expand_text(text: str, rng: random.Random, lookup, missing: list) -> str:
    """``text`` with its wildcards replaced; a name no file has stays as written and goes into ``missing``."""
    if not text:
        return text
    replace = _file_replacer(rng, lookup, missing)
    for _ in range(MAX_DEPTH):
        before = text
        for pattern in _FILE_PATTERNS:
            text = _outside_tags(text, lambda part: pattern.sub(replace, part))
        text = _BRACES.sub(lambda match: _pick_braced(match, rng), text)
        if text == before:
            break
    return text


def _running_pass(state: dict, path: tuple):
    """The dict holding the prompt at ``path``, None when its pass is off: a leftover wildcard there changes nothing."""
    node = state
    for key in path[:-1]:
        node = node.get(key) if isinstance(node, dict) else None
    return node if isinstance(node, dict) and node.get("on") else None


def uses_wildcards(state: dict) -> bool:
    texts = [assembled_prompt(state), state.get("negative") or ""]
    texts += [node.get(path[-1]) or "" for path in _PASS_PROMPTS if (node := _running_pass(state, path))]
    return any(map(has_wildcard, texts))


def expand_state(state: dict, seed: int, lookup) -> tuple[dict, list]:
    """A copy of ``state`` with this seed's prompts, and the names no file has.  The templates go to the saved info;
    the prompt boxes are joined first, so a box can use what another one wrote."""
    out = copy.deepcopy(state)
    missing: list = []

    def drawn(text: str, where: str) -> str:
        return expand_text(text, random.Random(f"{int(seed)}/{where}"), lookup, missing)

    prompt = assembled_prompt(out)
    expanded = drawn(prompt, "prompt")
    if expanded != prompt:
        out.update(dict.fromkeys(PROMPT_BOXES, ""), prompt=expanded, prompt_template=prompt)
    negative = out.get("negative") or ""
    expanded = drawn(negative, "negative")
    if expanded != negative:
        out.update(negative=expanded, negative_template=negative)
    for path in _PASS_PROMPTS:
        node = _running_pass(out, path)
        if node is not None and node.get(path[-1]):
            node[path[-1]] = drawn(node[path[-1]], "/".join(path))
    return out, missing


def expand_run(state: dict, lookup, rng: random.Random | None = None) -> dict:
    """``{"states", "missing"}``: one state per image, each with its own seed and prompts, as Forge's Dynamic Prompts
    gives every image of a batch its own.  Nothing to draw = no states: the run stays one batch, untouched."""
    if not uses_wildcards(state):
        return {"states": [], "missing": []}
    seed = int(state.get("seed", -1))
    if seed < 0:
        seed = (rng or random).randrange(SEED_RANGE)
    states, missing, changed = [], [], False
    for index in range(max(1, int(state.get("batch") or 1))):
        one, names = expand_state(state, seed + index, lookup)
        changed = changed or one != state
        one.update(seed=seed + index, batch=1)
        states.append(one)
        missing += [name for name in names if name not in missing]
    return {"states": states if changed else [], "missing": missing}


# ------------------------------------------------------------------------------------------------- files

def _read_groups(path: Path) -> list:
    groups = []
    for line in _NEWLINE.split(_COMMENT.sub("", path.read_text(encoding="utf-8-sig"))):
        if line.strip().startswith("#"):
            continue
        options = [option.strip() for option in line.split(",") if option.strip()]
        if options:
            groups.append(options)
    return groups


class WildcardFiles:
    """The ``.txt`` files of ``directories()`` (first folder wins), read again when a file changes."""

    def __init__(self, directories) -> None:
        self.directories = directories
        self._cache: dict = {}  # path -> ((size, time), groups)

    def groups(self, name: str) -> list:
        if name.lower().endswith(".txt"):
            name = name[:-4]
        if not name or _BAD_NAME.search(name):
            return []
        for folder in self.directories():
            path = Path(folder) / f"{name}.txt"
            try:
                info = path.stat()
                stamp = (info.st_size, info.st_mtime_ns)
                cached = self._cache.get(path)
                if cached is None or cached[0] != stamp:
                    cached = (stamp, _read_groups(path))
                    self._cache[path] = cached
                return cached[1]
            except (OSError, UnicodeDecodeError):
                continue
        return []

    def names(self) -> list:
        found = set()
        for folder in self.directories():
            try:
                found.update(path.stem for path in Path(folder).glob("*.txt") if path.is_file())
            except OSError:
                continue
        return sorted(found)


def wildcard_directories() -> list:
    try:
        folder_paths = importlib.import_module("folder_paths")
    except Exception:
        return []
    folders = [Path(folder_paths.base_path) / "wildcards"] if getattr(folder_paths, "base_path", None) else []
    try:
        folders += [Path(folder) for folder in folder_paths.get_folder_paths("wildcards")]
    except Exception:  # no such entry in extra_model_paths.yaml
        pass
    return folders


FILES = WildcardFiles(wildcard_directories)
