# core/safe_print.py
"""예외를 내지 않는 콘솔 출력 — 진단 로그가 제어 흐름을 바꾸지 못하게 한다.

한국어 Windows에서는 stdout이 콘솔이 아니면(파이프·파일: 테스트 러너, /verify, 로그
리다이렉트) 인코딩이 cp949가 되고, print()는 em dash(—)나 '✓' 같은 문자에서
UnicodeEncodeError를 낸다. 그런 print 한 줄이 except 블록 안에서 터지면 정리 코드
(사용자 알림·VRAM 반납)를 건너뛰고 엉뚱한 오류가 사용자에게 간다.

safe_print는 print와 같은 모양으로 쓰되
- 스트림 인코딩으로 쓸 수 없는 문자는 \\uXXXX 로 바꿔 쓰고
- 그 밖의 쓰기 실패(stdout 없음·닫힌 스트림 등)는 조용히 버린다.
"""
import sys


def safe_print(*args, sep=' ', end='\n', file=None, flush=False) -> None:
    """print()와 같지만 절대 예외를 내지 않는다. 한 줄은 한 번의 write로 쓴다."""
    try:
        stream = sys.stdout if file is None else file
        if stream is None:                   # pythonw 등 콘솔 없는 실행
            return
        text = (' ' if sep is None else str(sep)).join(str(arg) for arg in args)
        text += '\n' if end is None else str(end)
    except Exception:
        return
    try:
        stream.write(text)
    except UnicodeEncodeError:
        try:
            stream.write(_encodable(text, getattr(stream, 'encoding', None)))
        except Exception:
            return
    except Exception:
        return
    if flush:
        try:
            stream.flush()
        except Exception:
            pass


def _encodable(text: str, encoding) -> str:
    """encoding으로 쓸 수 없는 문자를 \\uXXXX 이스케이프로 바꾼 문자열."""
    encoding = encoding or 'ascii'
    return text.encode(encoding, errors='backslashreplace').decode(encoding, errors='replace')
