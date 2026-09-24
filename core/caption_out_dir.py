# core/caption_out_dir.py
"""캡션 .txt 저장 폴더(outDir) 검증 — 캡션 불러오기·수동 저장·일괄 처리가 같은 규칙을 쓴다.

outDir 은 폴더 대화상자(caption_pick_outdir)로만 고르고, 화면에는 읽기 전용으로 보인다.
그래서 **이미 있는 폴더만** 받는다(만들지 않는다).
  - 웹 클라이언트가 이 값으로 임의 위치에 폴더 트리를 만들지 못한다.
  - 저장해 둔 폴더가 지워지거나 이름이 바뀌면 조용히 새로 만들어 엉뚱한 곳에 쌓지 않고,
    다시 고르라고 알린다. 예전엔 수동 저장은 영어 오류('not a directory: [path]')로 실패하고
    일괄 처리는 폴더를 새로 만들어 두 경로가 같은 설정을 다르게 다뤘다.
빈 값은 '이미지 옆(.txt 사이드카)' 이다. Qt 를 모르는 순수 모듈이다.

**서버가 승인한 폴더만** (``approved``) — outDir 는 클라이언트가 보내는 값이라, '있는 폴더'
검사만으로는 웹 클라이언트가 아무 폴더나 가리킬 수 있었다. renameFile 로 이미지 이름을
``requirements.png`` 로 바꾸고 outDir 에 앱 폴더를 주면 loadCaption/saveCaption 이 앱의
``requirements.txt`` 를 읽고 덮어썼다(다음 실행 때 check_requirements 가 그 목록을 pip 설치 →
호스트 코드 실행). 이제 호스트의 폴더 대화상자가 돌려준 폴더만 승인 목록에 오르고
(ui/generator_main.caption_pick_outdir → VueBridge.approve_caption_out_dir), 목록은 ui_prefs 의
서버 전용 키(:data:`APPROVED_PREFS_KEY`)에 남는다 — 클라이언트의 save_ui_prefs 는 이 키를 쓰지 못한다.

웹 모드에서는 여기에 더해 **앱 설치 폴더 안의 .txt** (생성 이미지 폴더 제외)를 캡션 대상으로 읽거나
쓰지 않는다(:func:`is_protected_caption_target`) — 이미지 옆 사이드카 경로로도 requirements.txt·
config/*.txt 에 닿지 못하게 하는 두 번째 벽이다.

세 번째 벽(웹 모드, 위치 무관): **설치·빌드 목록 이름**(requirements*·*constraints*·CMakeLists)의
.txt 는 캡션 대상이 아니다(:func:`is_manifest_caption_target`). 두 번째 벽은 이 앱 폴더만 지켜서,
빈 outDir(이미지 옆)과 renameFile(stem 선택)로 **다른** 파이썬 앱(로컬 Forge·ComfyUI 등) 폴더의
requirements.txt 를 덮어쓸 수 있었다 — 그 폴더에 이미지가 하나라도 있으면.

남는 한계(의도된 기능): 웹 모드에서도 앱 폴더 밖·시스템 폴더 밖의 **이미지 옆** ``<stem>.txt`` 는
읽고 쓸 수 있다 — 그게 캡션 사이드카다. 위 이름이 아닌 다른 앱의 .txt 설정 파일이 이미지와 같은
폴더에 같은 stem 으로 있으면 막지 못한다. 이미지 옆이 아닌 곳(outDir)은 호스트가 승인한 폴더뿐이다.
"""
from __future__ import annotations

import os
import re
from collections.abc import Iterable
from pathlib import Path

from core.path_safety import (
    UnsafePathError,
    _strip_file_scheme,
    is_forbidden_path,
    safe_output_dir,
)

MISSING_MESSAGE = '캡션 저장 폴더가 없습니다. 폴더를 다시 선택하거나 기본값(이미지 옆)으로 되돌리세요.'
NOT_A_FOLDER_MESSAGE = '캡션 저장 위치가 폴더가 아닙니다. 폴더를 다시 선택하거나 기본값(이미지 옆)으로 되돌리세요.'
REFUSED_MESSAGE = '시스템 폴더나 드라이브 루트에는 캡션을 저장할 수 없습니다. 다른 폴더를 선택하세요.'
NOT_APPROVED_MESSAGE = (
    '캡션 저장 폴더는 호스트 PC 의 폴더 선택 창에서 고른 폴더만 쓸 수 있습니다. '
    '폴더를 다시 선택하거나 기본값(이미지 옆)으로 되돌리세요.'
)
PROTECTED_TARGET_MESSAGE = (
    '웹 모드에서는 앱 설치 폴더 안(생성 이미지 폴더 제외)의 캡션 파일을 읽거나 저장할 수 없습니다. '
    '다른 폴더의 이미지를 쓰거나 호스트 PC 의 앱에서 작업하세요.'
)
MANIFEST_TARGET_MESSAGE = (
    '웹 모드에서는 requirements·constraints 같은 설치 목록 이름의 캡션 파일을 읽거나 저장할 수 없습니다. '
    '이미지 이름을 바꾸거나 호스트 PC 의 앱에서 작업하세요.'
)

# 웹 모드에서 캡션 대상으로 받지 않는 .txt 이름(stem, 대소문자 무시). 다른 파이썬 앱(Forge 의
# requirements_versions.txt, ComfyUI·확장의 requirements.txt …)이 실행 때 pip 로 설치하거나 빌드가
# 읽는 목록이다 — 덮어쓰면 그 앱을 다음에 켤 때 호스트 코드 실행으로 이어진다.
_MANIFEST_STEM_RE = re.compile(r'requirements|constraints|^cmakelists$', re.IGNORECASE)

#: ui_prefs 안의 **서버 전용** 키 — 호스트 대화상자로 고른 캡션 저장 폴더 목록(최근 것 먼저).
#: 클라이언트 save_ui_prefs 페이로드에서는 빠지고(ui/generator_main), 클라이언트에게도 보내지 않는다.
APPROVED_PREFS_KEY = 'captionOutDirApproved'
#: 프론트(BatchView)가 저장하는 현재 캡션 저장 폴더 키.
OUT_DIR_PREFS_KEY = 'captionOutDir'
#: 승인 목록 상한 — 오래된 것부터 밀려난다(지금 고른 폴더는 늘 맨 앞이라 빠지지 않는다).
MAX_APPROVED = 32


class CaptionOutDirError(ValueError):
    """캡션 저장 폴더를 쓸 수 없다. 메시지는 그대로 사용자에게 보여도 된다(경로 없음)."""


def _reason(raw: str) -> str:
    try:
        resolved = Path(raw).resolve(strict=False)
    except (OSError, ValueError):
        return REFUSED_MESSAGE
    if is_forbidden_path(str(resolved)) or resolved.parent == resolved:
        return REFUSED_MESSAGE
    try:
        if resolved.exists():
            return NOT_A_FOLDER_MESSAGE
    except OSError:
        pass
    return MISSING_MESSAGE


def caption_dir_key(path) -> str:
    """폴더 비교 키 — file:// 제거·절대화·심볼릭 링크/정션 해석·대소문자 정규화. 실패하면 ''."""
    text = str(path or '').strip()
    if not text:
        return ''
    try:
        resolved = Path(_strip_file_scheme(text)).resolve(strict=False)
    except (OSError, ValueError, RuntimeError):
        return ''
    return os.path.normcase(str(resolved))


def normalize_approved(values) -> list[str]:
    """ui_prefs 에서 읽은 승인 목록 → 중복 없는 문자열 목록(형식이 틀리면 [])."""
    if not isinstance(values, (list, tuple)):
        return []
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        if not isinstance(value, str):
            continue
        key = caption_dir_key(value)
        if not key or key in seen:
            continue
        seen.add(key)
        result.append(value.strip())
        if len(result) >= MAX_APPROVED:
            break
    return result


def remember_approved(approved: Iterable[str], folder: str, *, limit: int = MAX_APPROVED) -> list[str]:
    """folder 를 맨 앞에 두고 같은 폴더 중복을 뺀 새 목록(최대 limit 개)."""
    key = caption_dir_key(folder)
    if not key:
        return normalize_approved(list(approved or ()))[:max(1, int(limit))]
    rest = [p for p in normalize_approved(list(approved or ())) if caption_dir_key(p) != key]
    return ([str(folder)] + rest)[:max(1, int(limit))]


def is_approved(folder, approved: Iterable[str]) -> bool:
    key = caption_dir_key(folder)
    if not key:
        return False
    return any(caption_dir_key(item) == key for item in (approved or ()))


def filter_client_caption_prefs(payload, approved: Iterable[str]) -> list[str]:
    """클라이언트 save_ui_prefs 페이로드를 제자리에서 거른다. → 뺀 키 이름 목록.

    - 서버 전용 승인 목록(:data:`APPROVED_PREFS_KEY`)은 늘 뺀다 — 클라이언트가 승인을 만들지 못한다.
    - captionOutDir 은 ''(이미지 옆)이거나 승인된 폴더일 때만 받는다. 아니면 빼서 저장값을 지킨다.
      그래서 ui_prefs.captionOutDir 은 늘 호스트가 고른 값이고, 데스크톱의 1회 이식
      (VueBridge.seed_caption_out_dir_approval_from_prefs)이 웹 클라이언트가 심은 값을 승인하지 않는다.
    """
    dropped: list[str] = []
    if not isinstance(payload, dict):
        return dropped
    if APPROVED_PREFS_KEY in payload:
        payload.pop(APPROVED_PREFS_KEY, None)
        dropped.append(APPROVED_PREFS_KEY)
    if OUT_DIR_PREFS_KEY in payload:
        value = payload.get(OUT_DIR_PREFS_KEY)
        text = value.strip() if isinstance(value, str) else None
        if text is None or (text and not is_approved(text, approved)):
            payload.pop(OUT_DIR_PREFS_KEY, None)
            dropped.append(OUT_DIR_PREFS_KEY)
    return dropped


def resolve_caption_out_dir(raw, *, approved: Iterable[str] | None = None) -> str:
    """outDir 원문 → 검증된 절대 경로. 빈 값이면 ''(이미지 옆).

    폴더가 없거나 폴더가 아니거나 시스템 폴더면 CaptionOutDirError. 폴더를 만들지 않는다.
    ``approved`` 를 주면 그 목록(호스트 대화상자가 돌려준 폴더)에 있는 폴더만 받는다 —
    없으면 :data:`NOT_APPROVED_MESSAGE`. None 은 승인 검사를 하지 않는다(순수 규칙 검사용).

    승인 검사가 **먼저**다. 승인 안 된 경로는 없든·파일이든·있는 폴더든 늘 같은 답
    (NOT_APPROVED)이라, 웹 클라이언트가 loadCaption/saveCaption 오류 문구로 호스트의 경로 존재·종류를
    알아내지 못한다(Codex R3 재검토). 존재·종류 문구(MISSING·NOT_A_FOLDER)는 승인한 폴더가 그 뒤
    지워지거나 바뀐 경우에만 나온다.
    """
    text = str(raw or '').strip()
    if not text:
        return ''
    # is_approved 는 safe_output_dir 과 같은 해석(file:// 제거·resolve)으로 비교한다 — 답이 존재 여부로 갈리지 않는다.
    if approved is not None and not is_approved(text, approved):
        raise CaptionOutDirError(NOT_APPROVED_MESSAGE)
    try:
        resolved = safe_output_dir(text, create=False)
    except UnsafePathError as exc:
        # 판정은 safe_output_dir 이 한다 — 여기서는 사용자에게 보일 이유만 고른다.
        raise CaptionOutDirError(_reason(_strip_file_scheme(text))) from exc
    return resolved


def _within(child_key: str, parent_key: str) -> bool:
    if not child_key or not parent_key:
        return False
    try:
        return os.path.commonpath([child_key, parent_key]) == parent_key
    except ValueError:  # 다른 드라이브·절대/상대 혼합
        return False


def is_protected_caption_target(txt_path, *, app_root, allowed_roots: Iterable = ()) -> bool:
    """캡션 .txt 대상이 앱 설치 폴더(app_root) 안이고 허용 폴더(allowed_roots) 밖인지.

    경로는 심볼릭 링크·정션까지 해석해 비교한다(정션으로 앱 폴더를 가리켜 우회하지 못한다).
    대상 경로를 해석하지 못하면 보호 대상으로 본다(fail closed).
    """
    target = caption_dir_key(txt_path)
    if not target:
        return True
    root = caption_dir_key(app_root)
    if not _within(target, root):
        return False
    return not any(_within(target, caption_dir_key(allowed)) for allowed in (allowed_roots or ()))


def is_manifest_caption_target(txt_path) -> bool:
    """캡션 .txt 대상 이름이 설치·빌드 목록(requirements*·*constraints*·CMakeLists)인지.

    앱 설치 폴더 밖이라도 이미지 옆 사이드카(``<stem>.txt``)는 어디든 쓸 수 있고 renameFile 로 stem 을
    고를 수 있다 — 다른 파이썬 앱 폴더에 이미지가 하나라도 있으면 그 앱의 requirements.txt 를
    덮어쓸 수 있었다. 원문 이름과 해석한 이름(8.3 짧은 이름 ``REQUIR~1.TXT`` → 긴 이름) 둘 다 본다.
    대상 경로를 해석하지 못하면 막는다(fail closed).
    """
    text = str(txt_path or '').strip()
    if not text:
        return True
    resolved = caption_dir_key(text)
    if not resolved:
        return True
    stems = {Path(_strip_file_scheme(text)).stem, Path(resolved).stem}
    return any(_MANIFEST_STEM_RE.search(stem.strip(' .')) for stem in stems)
