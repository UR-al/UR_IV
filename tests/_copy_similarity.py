"""Token similarity of the bundled node pack's functions against another code base (the sam-extra extension).

The same measure as the review probe that found the pack's near-copies of sam-extra code: every function body
with its docstring and comments removed, unparsed, tokenised, strings replaced by ``STR``; ``difflib``'s ratio
(autojunk off) between a pack function of at least ``MIN_TOKENS`` tokens and each candidate function. Used by
``tests/test_comfy_node_pack.py`` to keep THIRD_PARTY_NOTICES.md's list of code shared with sam-extra complete.
"""
from __future__ import annotations

import ast
import bisect
import difflib
import io
import os
import tokenize
from pathlib import Path

MIN_TOKENS = 60
THRESHOLD = 0.80
# The extension's runtime code only: tests, vendored third-party trees and non-Python assets are not compared.
EXCLUDED_DIRS = frozenset({"tests", "anima_vendor", "lora_manager_vendor", "node_modules", "__pycache__", "vendor",
                           "outputs", "pip", "mcp_server", "javascript", "assets", "docs"})


def function_tokens(path: Path, label: str) -> dict[str, list[str]]:
    """``{"<label>:<qualname>": tokens}`` for every function in ``path`` (nested ones too)."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    found: dict[str, list[str]] = {}

    def visit(node, prefix):
        for child in ast.iter_child_nodes(node):
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                name = f"{prefix}{child.name}"
                body = child.body
                if (body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant)
                        and isinstance(body[0].value.value, str)):
                    body = body[1:]
                if body:
                    source = "\n".join(ast.unparse(statement) for statement in body)
                    tokens = []
                    for token in tokenize.generate_tokens(io.StringIO(source).readline):
                        if token.type in (tokenize.COMMENT, tokenize.NL, tokenize.NEWLINE, tokenize.INDENT,
                                          tokenize.DEDENT, tokenize.ENDMARKER):
                            continue
                        tokens.append("STR" if token.type == tokenize.STRING else token.string)
                    found[f"{label}:{name}"] = tokens
                visit(child, f"{name}.")
            elif isinstance(child, ast.ClassDef):
                visit(child, f"{prefix}{child.name}.")
            else:
                visit(child, prefix)

    visit(tree, "")
    return found


def python_files(root: Path, excluded=EXCLUDED_DIRS):
    for directory, names, files in os.walk(root):
        names[:] = sorted(name for name in names if name not in excluded and not name.startswith("."))
        for name in sorted(files):
            if name.endswith(".py"):
                path = Path(directory) / name
                yield path, path.relative_to(root).as_posix()


def collect(root: Path, prefix: str = "") -> dict[str, list[str]]:
    functions: dict[str, list[str]] = {}
    for path, relative in python_files(root):
        functions.update(function_tokens(path, prefix + relative))
    return functions


def near_copies(pack: dict[str, list[str]], other: dict[str, list[str]], threshold: float = THRESHOLD,
                min_tokens: int = MIN_TOKENS) -> dict[str, tuple[float, str]]:
    """``{pack function: (ratio, best match in other)}`` for the pack functions at or above ``threshold``."""
    candidates = sorted((len(tokens), name) for name, tokens in other.items())
    lengths = [length for length, _ in candidates]
    best: dict[str, tuple[float, str]] = {}
    # one matcher per candidate: its seq2 index and token counts are built once and reused for every pack function
    matchers: dict[str, difflib.SequenceMatcher] = {}
    for name, tokens in sorted(pack.items()):
        size = len(tokens)
        if size < min_tokens:
            continue
        # ratio = 2M/(a+b) >= t needs min(a, b) >= t(a+b)/2: b in [a·t/(2−t), a·(2−t)/t]
        low = bisect.bisect_left(lengths, int(size * threshold / (2 - threshold)))
        high = bisect.bisect_right(lengths, int(size * (2 - threshold) / threshold) + 1)
        for _length, other_name in candidates[low:high]:
            matcher = matchers.get(other_name)
            if matcher is None:
                matcher = matchers[other_name] = difflib.SequenceMatcher(None, autojunk=False)
                matcher.set_seq2(other[other_name])
            matcher.set_seq1(tokens)
            if matcher.real_quick_ratio() < threshold or matcher.quick_ratio() < threshold:
                continue
            ratio = matcher.ratio()
            if ratio >= threshold and ratio > best.get(name, (0.0, ""))[0]:
                best[name] = (ratio, other_name)
    return best
