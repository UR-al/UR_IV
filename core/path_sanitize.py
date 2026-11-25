"""Shared local-path and URL masking utilities for UI and external API boundaries.

Consolidates the path-redaction patterns previously duplicated across
``core/error_handler.py``, ``core/studio_application.py``, and
``core/generation_api.py``.
"""
from __future__ import annotations

import re

URL_RE = re.compile(r"https?://[^\s\]\[()<>{}\"']+", re.IGNORECASE)

# Compact variants stop at whitespace (suitable for human-facing error strings
# like "D:/models/x.ckpt 을 열 수 없음" where trailing prose follows a space).
WINDOWS_PATH_COMPACT_RE = re.compile(r"(?<![A-Za-z0-9])[A-Za-z]:[\\/][^\s'\"<>|]+")
UNC_PATH_COMPACT_RE = re.compile(r"(?<![\\/\w])\\\\[^\s'\"<>\\/]+[\\/][^\s'\"<>|]+")

# Wide variants extend across spaces until quote/newline/delimiter (suitable for
# Windows paths containing spaces such as "C:\Program Files\App\file").
WINDOWS_PATH_WIDE_RE = re.compile(r"(?<![A-Za-z0-9])[A-Za-z]:[\\/][^\r\n\"'<>|]+")
UNC_PATH_WIDE_RE = re.compile(r"(?<![\\/\w])\\\\[^\s\\/]+[\\/][^\r\n\"'<>|]+")

# POSIX absolute paths under common system/user roots, avoiding URL paths
# such as "http://127.0.0.1:7860/home/x" or API routes like "/api/v1/...".
POSIX_SENSITIVE_PATH_RE = re.compile(
    r"(?<![\w.\-\]])/(?:Users|home|var|etc|opt|tmp|mnt|srv|root|usr)/[^\s'\"<>|]*"
)

# Broader multi-segment POSIX path used by the generation API after URLs have
# already been replaced with "[backend endpoint]".
POSIX_DEEP_PATH_RE = re.compile(r"(?<![:\w])/(?:[^/\s]+/)+[^/\s]+")


def mask_local_paths(
    text: str,
    *,
    replacement: str = "[local path]",
    match_spaces: bool = False,
    mask_urls: bool = False,
    url_replacement: str = "[backend endpoint]",
    mask_all_posix: bool = False,
) -> str:
    """Replace local filesystem paths (and optionally URLs) in *text*.

    Parameters
    ----------
    text:
        Input string to sanitize.
    replacement:
        Placeholder substituted for masked filesystem paths.
    match_spaces:
        When ``True``, Windows/UNC path matches extend across spaces until
        quotes, angle brackets, pipes, or newlines.
    mask_urls:
        When ``True``, ``http://`` and ``https://`` URLs are replaced first
        with *url_replacement*.
    url_replacement:
        Placeholder substituted for masked URLs when *mask_urls* is ``True``.
    mask_all_posix:
        When ``True``, mask any multi-segment POSIX absolute path (``/a/b``)
        instead of only sensitive root prefixes.
    """
    if not text:
        return ""
    result = str(text)
    if mask_urls:
        result = URL_RE.sub(url_replacement, result)
    if match_spaces:
        result = UNC_PATH_WIDE_RE.sub(replacement, result)
        result = WINDOWS_PATH_WIDE_RE.sub(replacement, result)
    else:
        result = UNC_PATH_COMPACT_RE.sub(replacement, result)
        result = WINDOWS_PATH_COMPACT_RE.sub(replacement, result)
    posix_re = POSIX_DEEP_PATH_RE if mask_all_posix else POSIX_SENSITIVE_PATH_RE
    result = posix_re.sub(replacement, result)
    return result
