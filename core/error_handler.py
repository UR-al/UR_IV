# core/error_handler.py
"""전역 에러 핸들러 — 에러 코드 + CMD 출력 + Vue Toast 연동"""
import traceback
import sys

from core.path_sanitize import mask_local_paths


def sanitize_for_ui(message: str, max_len: int = 160) -> str:
    """UI로 보내기 전 민감 정보(절대 경로)를 제거하고 길이를 제한.

    로그에는 원본을, 프론트엔드에는 이 값만 전달하여 내부 파일 구조 노출을 막는다.
    """
    if not message:
        return ""
    text = mask_local_paths(str(message), replacement="[path]", match_spaces=False)
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


def _safe_write(text: str, stream=None) -> None:
    """콘솔 인코딩(cp949 등)에서 표현 불가 문자가 있어도 UnicodeEncodeError 없이 출력."""
    target = sys.stdout if stream is None else stream
    if target is None:
        return
    try:
        target.write(text)
    except UnicodeEncodeError:
        encoding = getattr(target, "encoding", None) or "utf-8"
        safe_text = text.encode(encoding, errors="replace").decode(encoding, errors="replace")
        try:
            target.write(safe_text)
        except Exception:
            pass
    except Exception:
        pass


def _safe_print(text: str = "", stream=None) -> None:
    _safe_write(f"{text}\n", stream=stream)


def handle_error(code: str, context: str, exception: Exception, notify: bool = True):
    """
    전역 에러 처리
    - CMD에 에러 코드 + traceback 출력
    - Vue Toast로 사용자에게 알림
    """
    desc = ERROR_CODES.get(code, 'Unknown Error')
    msg = f"[{code}] {desc} | {context}: {str(exception)}"

    # CMD 출력
    _safe_print(f"\n{'='*60}")
    _safe_print(f"[ERROR {code}] {desc}")
    _safe_print(f"  Context: {context}")
    _safe_print(f"  Detail: {exception}")
    _safe_print(f"{'='*60}")
    tb_text = traceback.format_exc()
    if tb_text:
        _safe_write(tb_text, stream=sys.stderr)
    _safe_print()

    # Vue Toast — 콘솔(위)은 원문, 토스트는 다른 UI 경로와 같은 sanitize_for_ui 정책(절대 경로 가림·길이 제한).
    if notify and _vue_bridge and hasattr(_vue_bridge, 'showNotification'):
        toast_msg = sanitize_for_ui(f"[{code}] {context}: {exception}", 120)
        try:
            _vue_bridge.showNotification.emit('error', toast_msg)
        except Exception:
            pass

    return msg


