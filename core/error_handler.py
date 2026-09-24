# core/error_handler.py
"""전역 에러 핸들러 — 에러 코드 + CMD 출력 + Vue Toast 연동"""
import re
import traceback
import sys

# 절대 경로를 감지하는 정규식 — 로그용/UI 분리에 활용.
# Windows: C:\foo\bar, C:/foo/bar, UNC \\server\share / POSIX: /home/foo 등
# 드라이브 문자 앞에는 영숫자가 올 수 없다 — 예전엔 'http://127.0.0.1:7860/…' 의 'p:/' 를 드라이브로
# 오인해 백엔드 오류의 URL(가장 중요한 진단 정보)을 'htt[path]' 로 망가뜨렸다. POSIX 분기도
# URL 경로('http://host/home/…')를 가리지 않도록 앞 글자가 단어·점·하이픈이면 건너뛴다.
_ABS_PATH_RE = re.compile(
    r"(?:"
    r"(?<![A-Za-z0-9])[A-Za-z]:[\\/][^\s'\"<>]+"                         # C:\foo, C:/foo
    r"|(?<![\\/\w])\\\\[^\s'\"<>\\/]+[\\/][^\s'\"<>]+"                  # \\server\share\…
    r"|(?<![\w.\-])/(?:home|Users|root|etc|var|usr|tmp)/[^\s'\"<>]*"    # /home/…
    r")"
)


def sanitize_for_ui(message: str, max_len: int = 160) -> str:
    """UI로 보내기 전 민감 정보(절대 경로)를 제거하고 길이를 제한.

    로그에는 원본을, 프론트엔드에는 이 값만 전달하여 내부 파일 구조 노출을 막는다.
    """
    if not message:
        return ""
    text = _ABS_PATH_RE.sub("[path]", str(message))
    if len(text) > max_len:
        text = text[: max_len - 1] + "…"
    return text

# 에러 코드 정의
ERROR_CODES = {
    'E001': 'Boot Error — 앱 초기화 실패',
    'E010': 'Action Error — Vue 액션 처리 실패',
    'E020': 'Generation Error — 이미지 생성 실패',
    'E030': 'Settings Error — 설정 저장/로드 실패',
    'E040': 'Editor Error — 에디터 처리 실패',
    'E050': 'Search Error — 검색 실패',
    'E060': 'Gallery Error — 갤러리 처리 실패',
    'E070': 'File Error — 파일 처리 실패',
    'E080': 'API Error — 백엔드 API 통신 실패',
    'E090': 'Ollama Error — AI 처리 실패',
    'E100': 'YOLO/SAM Error — 모델 처리 실패',
    'E110': 'Preset Error — 프리셋 처리 실패',
    'E120': 'Wildcard Error — 와일드카드 처리 실패',
    'E130': 'Queue Error — 대기열 처리 실패',
    'E999': 'Unknown Error — 알 수 없는 오류',
}

_vue_bridge = None


def set_bridge(bridge):
    """VueBridge 인스턴스 설정 (앱 시작 시 호출)"""
    global _vue_bridge
    _vue_bridge = bridge


def handle_error(code: str, context: str, exception: Exception, notify: bool = True):
    """
    전역 에러 처리
    - CMD에 에러 코드 + traceback 출력
    - Vue Toast로 사용자에게 알림
    """
    desc = ERROR_CODES.get(code, 'Unknown Error')
    msg = f"[{code}] {desc} | {context}: {str(exception)}"

    # CMD 출력
    print(f"\n{'='*60}")
    print(f"[ERROR {code}] {desc}")
    print(f"  Context: {context}")
    print(f"  Detail: {exception}")
    print(f"{'='*60}")
    traceback.print_exc()
    print()

    # Vue Toast — 콘솔(위)은 원문, 토스트는 다른 UI 경로와 같은 sanitize_for_ui 정책(절대 경로 가림·길이 제한).
    if notify and _vue_bridge and hasattr(_vue_bridge, 'showNotification'):
        toast_msg = sanitize_for_ui(f"[{code}] {context}: {exception}", 120)
        try:
            _vue_bridge.showNotification.emit('error', toast_msg)
        except Exception:
            pass

    return msg


