# core/path_safety.py
"""경로 검증 유틸리티 — path traversal / 시스템 디렉터리 쓰기 방어.

입력(읽기) / 출력(쓰기) 양쪽에서 재사용할 수 있는 얇은 래퍼.
"""
from __future__ import annotations

import functools
import logging
import ntpath
import os
import re
from collections.abc import Mapping
from pathlib import Path, PurePosixPath, PureWindowsPath
from urllib.parse import unquote

logger = logging.getLogger(__name__)

# Windows 주요 시스템 루트 — 쓰기 금지, 읽기도 명시 허용 외엔 금지.
# 실제 위치는 환경변수(SystemRoot/ProgramFiles/...)로 다시 계산해 더한다 —
# 시스템 드라이브가 C: 가 아닌 PC 에서도 막혀야 하기 때문. 이 목록은 그 바닥값이다.
_WINDOWS_DEFAULT_ROOTS: tuple[str, ...] = (
    "C:\\Windows",
    "C:\\Program Files",
    "C:\\Program Files (x86)",
    "C:\\ProgramData",
)
# 환경변수 이름 → 시스템 루트. 값이 없거나 이상하면 건너뛴다(바닥값이 남는다).
_WINDOWS_ROOT_ENV_VARS: tuple[str, ...] = (
    "SystemRoot", "windir", "ProgramFiles", "ProgramFiles(x86)", "ProgramW6432",
    "ProgramData", "ALLUSERSPROFILE",
)
# SystemDrive 기준 상대 루트 — 환경변수가 비어 있어도 시스템 드라이브에서 막는다.
_WINDOWS_SYSTEM_DRIVE_DIRS: tuple[str, ...] = (
    "Windows", "Program Files", "Program Files (x86)", "ProgramData",
)
# 드라이브 공유(C$)가 아니면서 시스템 폴더로 곧장 통하는 관리 공유.
# ADMIN$ = %SystemRoot%, PRINT$ = system32\spool\drivers, IPC$ = 명명 파이프.
_WINDOWS_SYSTEM_SHARES: frozenset[str] = frozenset({"admin$", "print$", "ipc$"})
_DRIVE_SHARE_RE = re.compile(r"^[A-Za-z]\$$")
_DRIVE_RE = re.compile(r"^[A-Za-z]:(?:\\|$)")

_POSIX_FORBIDDEN_ROOTS: tuple[str, ...] = (
    "/etc",
    "/var",
    "/usr",
    "/proc",
    "/sys",
    "/boot",
)

_DEFAULT_IMAGE_EXTS: frozenset[str] = frozenset({
    ".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tiff", ".gif",
})


def strip_file_url(path: str) -> str:
    """``file://`` URL 이면 스킴을 떼고 퍼센트 인코딩을 푼다. 원시 경로는 글자 그대로 둔다.

    퍼센트 인코딩은 file URL 에만 있다. 원시 Windows 경로의 ``%`` 는 이름의 일부다(웹에서 받은
    ``image%20(1).png`` 등) — 예전에는 모든 입력을 unquote 해서 그런 파일이 다른 이름으로 바뀌어
    썸네일·메타데이터·즐겨찾기·편집 저장·삭제가 실패했고, 디코드된 이름의 형제 파일(``image (1).png``)이
    있으면 그 파일을 대상으로 삼았다(휴지통 이동 포함). 프론트 utils/fileUrl.ts(stripFileUrl —
    dropPaths·imageDeleteResult·에디터가 씀) · utils/media.js 와 같은 규칙이고,
    ui/generator_main._clean_path · core/editor_save(사본 레지스트리·protect_paths)도 이 함수를 쓴다.
    """
    text = str(path)
    head = text[:8].lower()
    for prefix in ("file:///", "file://"):
        if head.startswith(prefix):
            return unquote(text[len(prefix):])
    return text


def _strip_file_scheme(path: str) -> str:
    """file:// URL 은 스킴과 퍼센트 인코딩을 해제, 원시 경로는 그대로(:func:`strip_file_url`)."""
    return strip_file_url(path)


def _windows_forbidden_roots(environ: Mapping[str, str] | None = None) -> tuple[PureWindowsPath, ...]:
    """환경변수 기준 시스템 루트 + C: 바닥값. 드라이브가 붙은 절대 경로만 받는다.

    갤러리 썸네일마다 불리므로 관련 환경변수 값 묶음으로 캐시한다(값이 바뀌면 다시 계산).
    """
    env = os.environ if environ is None else environ
    key = tuple(str(env.get(name) or "").strip() for name in (*_WINDOWS_ROOT_ENV_VARS, "SystemDrive"))
    return _windows_forbidden_roots_cached(key)


@functools.lru_cache(maxsize=8)
def _windows_forbidden_roots_cached(key: tuple[str, ...]) -> tuple[PureWindowsPath, ...]:
    *env_values, system_drive = key
    raw: list[str] = list(_WINDOWS_DEFAULT_ROOTS)
    raw.extend(value for value in env_values if value)
    if re.fullmatch(r"[A-Za-z]:", system_drive):
        raw.extend(f"{system_drive}\\{name}" for name in _WINDOWS_SYSTEM_DRIVE_DIRS)
    roots: list[PureWindowsPath] = []
    for value in raw:
        candidate = PureWindowsPath(value.replace("/", "\\"))
        # 드라이브 루트(C:\) 자체나 상대 경로는 '시스템 폴더' 가 아니다 — 모든 것을 막게 된다.
        if not _DRIVE_RE.match(str(candidate)) or len(candidate.parts) < 2:
            continue
        if candidate not in roots:
            roots.append(candidate)
    return tuple(roots)


def _normalize_windows_for_check(path_text: str) -> str | None:
    r"""검사용 Windows 경로 표기를 하나로 모은다. None = 판정 불가(막아야 함).

    - ``\\?\UNC\srv\share`` / ``\\.\UNC\srv\share`` → ``\\srv\share``
    - ``\\?\C:\...`` / ``\\.\C:\...`` → ``C:\...``
    - 그 밖의 장치 이름공간(``\\?\GLOBALROOT``, ``\\?\Volume{..}``, ``\\.\pipe`` …) → None
    - 관리 공유 ``\\아무호스트\C$\...`` → ``C:\...`` — 원격 호스트여도 같은 규칙.
      그 PC 의 시스템 폴더를 막을 뿐, ``\\pc\D$\images`` 같은 갤러리는 그대로 통과한다.
    - ``ADMIN$``/``PRINT$``/``IPC$`` 공유 → None
    UNC 전체를 막지는 않는다 — 매핑 드라이브(Z:)는 realpath 에서 ``\\server\share`` 로
    바뀌므로, 막으면 NAS 갤러리의 읽기·삭제·이름변경이 모두 깨진다.
    """
    text = str(path_text).replace("/", "\\")
    for prefix in ("\\\\?\\", "\\\\.\\"):
        if text.startswith(prefix):
            rest = text[len(prefix):]
            if rest[:4].upper() == "UNC\\":
                text = "\\\\" + rest[4:]
                break
            if _DRIVE_RE.match(rest):
                text = rest
                break
            return None
    if text.startswith("\\\\"):
        parts = text[2:].split("\\")
        if len(parts) < 2 or not parts[0] or not parts[1]:
            return None
        share = parts[1]
        if share.lower() in _WINDOWS_SYSTEM_SHARES:
            return None
        if _DRIVE_SHARE_RE.match(share):
            tail = "\\".join(p for p in parts[2:] if p)
            return f"{share[0].upper()}:\\{tail}"
    return text


def _is_forbidden_windows(path_text: str, environ: Mapping[str, str] | None = None) -> bool:
    normalized = _normalize_windows_for_check(path_text)
    if normalized is None:
        return True
    # resolve 를 거친 값이면 이미 정규화돼 있지만, 관리 공유를 드라이브로 바꾼 꼬리에
    # 남은 '..' 같은 것도 접어서 비교한다.
    candidate = PureWindowsPath(ntpath.normpath(normalized))
    # PureWindowsPath 비교는 대소문자를 무시하고 구성요소 단위다 —
    # 'C:\WindowsImages' 가 'C:\Windows' 로 오인되지 않는다.
    return any(candidate.is_relative_to(root) for root in _windows_forbidden_roots(environ))


def _is_forbidden_posix(path_text: str) -> bool:
    candidate = PurePosixPath(str(path_text))
    return any(candidate.is_relative_to(PurePosixPath(root)) for root in _POSIX_FORBIDDEN_ROOTS)


def is_forbidden_path(
    path_text: str,
    *,
    windows: bool | None = None,
    environ: Mapping[str, str] | None = None,
) -> bool:
    """resolve 된 경로 문자열이 시스템 폴더(또는 판정 불가 장치 경로)인지."""
    if windows is None:
        windows = os.name == "nt"
    if windows:
        return _is_forbidden_windows(path_text, environ)
    return _is_forbidden_posix(path_text)


def _is_forbidden(resolved: Path) -> bool:
    return is_forbidden_path(str(resolved))


class UnsafePathError(ValueError):
    """검증에 실패한 경로가 사용되었음을 나타낸다."""


def safe_input_path(
    raw: str,
    *,
    allowed_exts: frozenset[str] | None = _DEFAULT_IMAGE_EXTS,
) -> str | None:
    """외부(Vue/설정/EXIF)에서 온 읽기용 경로를 검증한다.

    - 존재 + 일반 파일
    - 시스템 디렉터리 차단
    - 확장자 화이트리스트 (None이면 생략)
    성공 시 정규화된 절대 경로, 실패 시 None.
    """
    if not raw:
        return None
    try:
        resolved = Path(_strip_file_scheme(raw)).resolve(strict=False)
    except (OSError, ValueError) as e:
        logger.warning("input path resolve failed: %r (%s)", raw, e)
        return None
    if _is_forbidden(resolved):
        logger.warning("input path forbidden root: %s", resolved)
        return None
    if allowed_exts is not None and resolved.suffix.lower() not in allowed_exts:
        logger.warning("input path blocked ext: %s", resolved)
        return None
    if not resolved.is_file():
        return None
    return str(resolved)


def missing_input_path(
    raw: str,
    *,
    allowed_exts: frozenset[str] | None = _DEFAULT_IMAGE_EXTS,
) -> bool:
    """safe_input_path 가 None 인 이유가 '허용된 경로인데 파일이 없음' 인지.

    삭제 요청처럼 '이미 없으면 목록에서 빼면 되는' 호출부가 거부(시스템 폴더·막힌
    확장자·해석 실패)와 구분하려고 쓴다. 거부 사유면 False, 폴더 등 파일이 아닌 대상이
    그 자리에 있어도 False 다.
    """
    return resolve_missing_input_path(raw, allowed_exts=allowed_exts) is not None


def resolve_missing_input_path(
    raw: str,
    *,
    allowed_exts: frozenset[str] | None = _DEFAULT_IMAGE_EXTS,
) -> str | None:
    """:func:`missing_input_path` 와 같은 판정 — 참이면 **판정에 쓴** 정규화 절대 경로, 아니면 None.

    '없는 파일' 로 받은 뒤 그 경로로 무언가를 열거나 캐시 키를 만드는 호출부는 원문이 아니라
    이 값을 써야 한다. 원문을 OS 에 다시 넘기면 file:// 제거·URL 디코딩('%5C'·'%2F')·'..' 접기가
    검사 때와 달라, 검사는 허용 폴더의 없는 파일로 통과하고 OS 는 시스템 폴더의 파일을 여는
    원문을 만들 수 있다(core/thumb_prefetch.py).
    """
    if not raw:
        return None
    try:
        resolved = Path(_strip_file_scheme(raw)).resolve(strict=False)
    except (OSError, ValueError):
        return None
    if _is_forbidden(resolved):
        return None
    if allowed_exts is not None and resolved.suffix.lower() not in allowed_exts:
        return None
    try:
        if resolved.exists() or resolved.is_symlink():
            return None
    except OSError:
        return None
    return str(resolved)


def safe_output_dir(raw: str, *, create: bool = True) -> str:
    """설정에서 받은 출력 폴더 경로를 검증.

    - 시스템 디렉터리로 쓰기 금지
    - 드라이브/루트 자체(C:\\, /)로의 직접 쓰기 금지
    - 필요 시 mkdir(parents=True)
    실패 시 UnsafePathError.
    """
    if not raw:
        raise UnsafePathError("output folder is empty")
    try:
        resolved = Path(_strip_file_scheme(raw)).resolve(strict=False)
    except (OSError, ValueError) as e:
        raise UnsafePathError(f"resolve failed: {e}") from e
    if _is_forbidden(resolved):
        raise UnsafePathError(f"forbidden system dir: {resolved}")
    if resolved.parent == resolved:
        raise UnsafePathError(f"refuse to write to filesystem root: {resolved}")
    if create:
        try:
            resolved.mkdir(parents=True, exist_ok=True)
        except OSError as e:
            raise UnsafePathError(f"mkdir failed: {e}") from e
    if not resolved.is_dir():
        raise UnsafePathError(f"not a directory: {resolved}")
    return str(resolved)


def safe_config_file(raw: str, *, must_exist: bool = True) -> str:
    """설정 파일 경로(워크플로우 JSON 등) 검증.

    - 시스템 디렉터리 차단
    - 필요 시 존재 강제
    실패 시 UnsafePathError.
    """
    if not raw:
        raise UnsafePathError("path is empty")
    try:
        resolved = Path(_strip_file_scheme(raw)).resolve(strict=False)
    except (OSError, ValueError) as e:
        raise UnsafePathError(f"resolve failed: {e}") from e
    if _is_forbidden(resolved):
        raise UnsafePathError(f"forbidden system dir: {resolved}")
    if must_exist and not resolved.is_file():
        raise UnsafePathError(f"file not found: {resolved}")
    return str(resolved)
