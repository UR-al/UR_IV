# core/url_safety.py
"""외부로 여는 URL 검사 — 데스크톱 내비게이션 정책과 open_url 액션이 함께 쓴다.

Windows 에서 ``webbrowser.open`` 은 ``os.startfile(url)`` 이다. 그래서 스킴 검사 없이
넘기면 ``file:``·드라이브 경로·UNC·``search-ms:`` 같은 값이 호스트에서 로컬 프로그램을
실행하거나 탐색기를 연다. 여기서는 호스트가 있는 http/https(선택적으로 mailto)만
허용한다. Qt 를 모르는 순수 모듈이다.
"""
from __future__ import annotations

from urllib.parse import urlsplit

#: 너무 긴 URL 은 브라우저·셸 양쪽에서 잘리거나 거부된다 — 미리 막는다.
MAX_EXTERNAL_URL_LENGTH = 8192
_WEB_SCHEMES = frozenset({"http", "https"})


def is_safe_external_url(raw: object, *, allow_mailto: bool = False) -> bool:
    """호스트 PC 의 기본 브라우저/메일 앱으로 넘겨도 되는 URL 인지."""
    if not isinstance(raw, str):
        return False
    url = raw.strip()
    if not url or len(url) > MAX_EXTERNAL_URL_LENGTH:
        return False
    # 제어문자·공백·역슬래시는 셸/브라우저가 서로 다르게 해석하는 틈이 된다.
    if any(ch.isspace() or ord(ch) < 0x20 or ord(ch) == 0x7F or ch == "\\" for ch in url):
        return False
    try:
        parts = urlsplit(url)
        scheme = parts.scheme.lower()
        if scheme in _WEB_SCHEMES:
            # urlsplit 은 'http:evil' 도 받는다 — '//' 권한부와 호스트가 반드시 있어야 한다.
            if not url[len(parts.scheme) + 1:].startswith("//"):
                return False
            host = parts.hostname
            _ = parts.port  # 잘못된 포트는 ValueError
            return bool(host)
        if scheme == "mailto" and allow_mailto:
            address = parts.path
            return bool(address) and "@" in address and not address.startswith("/")
    except ValueError:
        return False
    return False


UNSAFE_URL_MESSAGE = "열 수 없는 링크입니다 — http/https 주소만 브라우저로 열 수 있습니다."


def open_external_url(raw: object, *, web_mode: bool, opener=None) -> tuple[bool, str]:
    """호스트 기본 브라우저로 URL 을 연다. (열었는지, 안 열었을 때 사용자 메시지).

    웹 모드에선 열지 않는다 — 호스트 PC 에서 열려 원격 사용자에겐 보이지 않고, 권한상
    호스트 셸 실행이 된다. 브라우저가 스스로 연다(frontend utils/externalUrl.ts).
    빈 값은 조용히 무시한다(예전 동작).
    """
    url = raw.strip() if isinstance(raw, str) else ""
    if not url:
        return False, ""
    if web_mode:
        from core.web_action_policy import WEB_BLOCKED_ACTIONS
        return False, WEB_BLOCKED_ACTIONS["open_url"]
    if not is_safe_external_url(url):
        return False, UNSAFE_URL_MESSAGE
    if opener is None:
        import webbrowser
        opener = webbrowser.open
    opener(url)
    return True, ""
