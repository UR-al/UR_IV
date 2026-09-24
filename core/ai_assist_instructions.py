"""Settings-backed instructions exclusively for T2I AI assistance.

Chat and image captioning do not use this module. Empty configuration keeps
the existing system prompt byte-for-byte; this module never edits its defaults.
"""
from __future__ import annotations

import json
import threading
from pathlib import Path

from core.config_migration import save_ui_prefs
from core.storage_paths import config_file
# 경로 이름만 공유한다 — 이 모듈의 reader 는 마이그레이션(.bak) 부작용을 일부러 피한다(load_instructions).
from core.ui_prefs import UI_PREFS_NAME

FEATURES = ("expand", "suggest", "nl2tags", "nl_caption", "nl_scene",
            "translate", "creative", "negative", "auto_nl")
MAX_INSTRUCTION_LENGTH = 8000
PREFS_KEY = "aiAssistInstructions"
_SAVE_LOCK = threading.RLock()

_OUTPUT_CONTRACTS = {
    "expand": "Output ONLY comma-separated tags. No explanations, headings, numbering or markdown.",
    "suggest": "Output ONLY comma-separated tags. No explanations, headings, numbering or markdown.",
    "nl2tags": "Output ONLY comma-separated tags. No explanations, headings, numbering or markdown.",
    "negative": "Output ONLY comma-separated NEGATIVE tags. No explanations, headings, numbering or markdown.",
    "nl_caption": "Output only the final English caption on one line. Use sentences without commas, starting with a capital letter and ending with a period. No explanations, labels or markdown.",
    "nl_scene": "Output only the final English scene description. Use sentences without commas, starting with a capital letter and ending with a period. No explanations, labels or markdown.",
    "translate": "Output ONLY the translated text in the target language. No explanations, notes or markdown.",
    "creative": "Keep the existing three-part format: one line of comma-separated Danbooru tags, a blank line, the scene description, then a final line exactly 'Resolution: WIDTHxHEIGHT'. The description uses sentences without commas. No additional explanations, headings or markdown.",
}


def normalize_instructions(value, *, strict=False):
    """Return a detached schema; tolerant reads ignore bad fields and cap text.

    Strict saves reject malformed input instead of silently dropping user text.
    Missing known fields are empty; unknown top-level/feature keys are rejected.
    """
    result = {"common": "", "features": {name: "" for name in FEATURES}}
    if not isinstance(value, dict):
        if strict:
            raise ValueError("AI 지침은 common과 features를 가진 객체여야 합니다.")
        return result
    if strict:
        if set(value) - {"common", "features"}:
            raise ValueError("알 수 없는 AI 지침 항목이 있습니다.")
        if not isinstance(value.get("common", ""), str):
            raise ValueError("공통 지침은 문자열이어야 합니다.")
        if len(value.get("common", "")) > MAX_INSTRUCTION_LENGTH:
            raise ValueError("공통 지침은 8,000자 이하여야 합니다.")
        if not isinstance(value.get("features", {}), dict):
            raise ValueError("기능별 지침은 객체여야 합니다.")
        for name, content in value.get("features", {}).items():
            if name not in FEATURES:
                raise ValueError(f"지원하지 않는 AI 기능입니다: {name}")
            if not isinstance(content, str):
                raise ValueError(f"{name} 지침은 문자열이어야 합니다.")
            if len(content) > MAX_INSTRUCTION_LENGTH:
                raise ValueError(f"{name} 지침은 8,000자 이하여야 합니다.")
    if isinstance(value.get("common"), str):
        result["common"] = value["common"][:MAX_INSTRUCTION_LENGTH]
    features = value.get("features", {})
    if isinstance(features, dict):
        for name in FEATURES:
            if isinstance(features.get(name), str):
                result["features"][name] = features[name][:MAX_INSTRUCTION_LENGTH]
    return result


def _read_prefs(path, *, strict=False):
    try:
        with Path(path).open("r", encoding="utf-8") as stream:
            prefs = json.load(stream)
    except FileNotFoundError:
        return {}
    except (OSError, ValueError, UnicodeError, RecursionError):
        if strict:
            raise RuntimeError("기존 UI 설정을 읽지 못해 AI 지침을 저장하지 않았습니다. 설정 파일을 확인하세요.") from None
        return {}
    if not isinstance(prefs, dict):
        if strict:
            raise RuntimeError("기존 UI 설정 형식이 잘못되어 AI 지침을 저장하지 않았습니다.")
        return {}
    return prefs


def load_instructions(path=None):
    """Read a settings snapshot without migration, backups or config rewrites."""
    try:
        prefs_path = path if path is not None else config_file(UI_PREFS_NAME)
        with _SAVE_LOCK:
            prefs = _read_prefs(prefs_path)
        return normalize_instructions(prefs.get(PREFS_KEY))
    except (OSError, ValueError, TypeError):
        return normalize_instructions(None)


def save_instructions(value, path=None):
    """Validate and merge one preference key, preserving unrelated settings.

    Invalid existing files are never replaced by an empty preference object.
    The established UI preference writer owns schema/version normalization.
    """
    normalized = normalize_instructions(value, strict=True)
    prefs_path = Path(path if path is not None else config_file(UI_PREFS_NAME)).absolute()
    with _SAVE_LOCK:
        prefs = _read_prefs(prefs_path, strict=True)
        prefs[PREFS_KEY] = normalized
        save_ui_prefs(str(prefs_path), prefs)
    return normalized


def compose_system_prompt(base_prompt, mode, instructions, *, feature=None):
    """Compose only instructions that belong to the requested AI assist feature."""
    if mode not in _OUTPUT_CONTRACTS:
        return base_prompt
    if feature not in (None, mode) and not (mode == "nl_caption" and feature == "auto_nl"):
        return base_prompt
    settings = normalize_instructions(instructions)
    layers = [("Common content instructions", settings["common"]),
              (f"Feature content instructions ({mode})", settings["features"][mode])]
    if feature == "auto_nl":
        layers.append(("Automatic caption content instructions (auto_nl)", settings["features"]["auto_nl"]))
    layers = [(label, content) for label, content in layers if content.strip()]
    if not layers:
        return base_prompt
    blocks = [base_prompt,
              "The following user-configured content instructions take precedence over conflicting default content rules above. "
              "Feature instructions refine common instructions; automatic-caption instructions refine caption instructions. "
              "When content rules conflict, the more specific feature instruction takes precedence. "
              "These instructions may change content, but must preserve the application's output format below."]
    blocks.extend(f"{label}:\n{content}" for label, content in layers)
    blocks.append("Required application output contract (unchanged):\n" + _OUTPUT_CONTRACTS[mode])
    return "\n\n".join(blocks)
