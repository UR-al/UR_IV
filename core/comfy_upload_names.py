# core/comfy_upload_names.py
"""ComfyUI input 업로드 파일명 — 내용 해시 기반 (순수 함수, Qt 비의존).

업로드마다 ``uuid4`` 로 새 이름을 만들면 같은 원본을 N번 처리할 때 ComfyUI/input 에
N개의 사본이 영구히 쌓이고, LoadImage 목록과 /object_info 도 따라 커진다(감사 #149).
이름을 내용의 sha256 앞 32자(128비트)로 정하면:

- 같은 바이트 → 같은 이름: 다시 올려도 사본이 늘지 않는다. ``overwrite=False`` 로 보내면
  ComfyUI 는 같은 이름·같은 내용의 파일을 다시 쓰지 않고 그 이름을 돌려준다.
- 다른 바이트 → 다른 이름: 원본/마스크, 대기열의 다른 작업이 서로를 덮어쓰지 않는다.

``ComfyUIBackend._upload_image``, Krea2 원본/참조 업로드(메인 I2I·Creator), Creator 의
H3 원본/아이덴티티 업로드, Krea2 hires 재업로드, Living Comic 컷 업로드가 이 규칙을 같이 쓴다
(``upload_content``). 사용자 파일명(basename)을 그대로 ``overwrite=true`` 로 올리면 다른 폴더의
동명 파일(0001.png 등)이 서로를, 그리고 ComfyUI/input 에 원래 있던 같은 이름의 파일을 덮어썼다.
"""
from __future__ import annotations

import hashlib
import re
from typing import Any, Callable, Optional

from core.cancellable_call import accepts_keyword, call_with_optional_cancel

_DIGEST_CHARS = 32
_SAFE_PART = re.compile(r"^[A-Za-z0-9_-]+$")
_SAFE_EXTENSION = re.compile(r"^[A-Za-z0-9]+$")


def content_upload_name(data: bytes | bytearray | memoryview, prefix: str, extension: str) -> str:
    """``{prefix}_{sha256(data)[:32]}.{extension}`` — 경로 구분자 없는 안전한 파일명.

    ``prefix`` 는 영숫자·``_``·``-`` 만, ``extension`` 은 영숫자만 허용한다(앞의 점은 떼어 준다).
    """
    if not isinstance(data, (bytes, bytearray, memoryview)):
        raise TypeError("data는 bytes 계열이어야 합니다.")
    clean_prefix = str(prefix or "").strip()
    if not _SAFE_PART.match(clean_prefix):
        raise ValueError(f"업로드 이름 접두어가 안전하지 않습니다: {prefix!r}")
    clean_extension = str(extension or "").strip().lstrip(".").lower()
    if not _SAFE_EXTENSION.match(clean_extension):
        raise ValueError(f"업로드 확장자가 안전하지 않습니다: {extension!r}")
    digest = hashlib.sha256(bytes(data)).hexdigest()[:_DIGEST_CHARS]
    return f"{clean_prefix}_{digest}.{clean_extension}"


def upload_content(
    backend: Any,
    data: bytes | bytearray | memoryview,
    prefix: str,
    extension: str,
    mime: str,
    *,
    cancel_check: Optional[Callable[[], bool]] = None,
) -> str:
    """``backend.upload_media`` 로 내용 해시 이름(``content_upload_name``)을 붙여 올린다.

    ``overwrite=False`` 를 받는 어댑터(``ComfyUIBackend.upload_media``)에는 넘겨서 ComfyUI 가
    같은 이름·같은 내용이면 다시 쓰지 않고, 같은 이름·다른 내용(해시 충돌은 사실상 없음)이면
    ``name (1).ext`` 로 비켜 가게 한다. 받지 못하는 duck-typed 어댑터·테스트 fake 에는
    ``overwrite``·``cancel_check`` 를 넘기지 않는다. 돌려주는 값은 서버가 저장한 입력 이름.
    """
    name = content_upload_name(data, prefix, extension)
    options = {"overwrite": False} if accepts_keyword(backend.upload_media, "overwrite") else {}
    return call_with_optional_cancel(
        backend.upload_media, data, name, mime, cancel_check=cancel_check, **options,
    )
