# utils/atomic_json.py
"""원자적 JSON 파일 쓰기 공용 유틸.

직접 open(path,'w') + json.dump 는 쓰는 도중 강제 종료/디스크 오류 시
파일이 절단되어 JSON 전체가 손상된다. tmp 파일에 완전히 쓴 뒤
os.replace(동일 볼륨에서 원자적)로 교체하면 '이전 버전 그대로' 또는
'새 버전 완성본' 둘 중 하나만 존재한다.

읽기 측: load_json_safe 는 손상 파일을 만나면 .corrupt 백업으로 옮겨
사용자가 복구할 수 있게 한다 — 조용히 빈 컨테이너를 돌려준 뒤 다음
저장이 빈 데이터로 덮어써 영구 소실되는 패턴을 방지.

쓰기 측 계약 (앱 전체의 tmp→os.replace 사본을 이 한 벌로 통일):
- durable=True(기본): replace 전에 fsync — 전원 차단 뒤에 '이름만 새 파일, 내용은 0바이트'가
  되는 것을 막는다. 실측 비용은 1회 1ms 안팎이라 기본으로 켠다.
- 직렬화·쓰기·교체 중 어느 단계가 실패해도 tmp 를 지운다(반쪽 tmp 잔존 방지).
- tmp 이름은 path + tmp_suffix(기본 '.tmp') 고정 — 호출처가 락/메인 스레드로 직렬화한다.
"""
import os
import json


def _discard(tmp: str) -> None:
    try:
        if os.path.exists(tmp):
            os.remove(tmp)
    except OSError:
        pass


def _prepare(path, tmp_suffix: str) -> tuple[str, str]:
    path = os.fspath(path)
    d = os.path.dirname(path)
    if d:
        os.makedirs(d, exist_ok=True)
    return path, path + tmp_suffix


def _finish(handle, durable: bool) -> None:
    if durable:
        handle.flush()
        os.fsync(handle.fileno())


def atomic_write_bytes(path, data: bytes, *, durable: bool = True, tmp_suffix: str = '.tmp') -> None:
    """bytes 를 path 에 원자적으로 기록. 실패 시 tmp 를 지우고 예외 전파."""
    path, tmp = _prepare(path, tmp_suffix)
    try:
        with open(tmp, 'wb') as f:
            f.write(data)
            _finish(f, durable)
        os.replace(tmp, path)
    finally:
        _discard(tmp)


def atomic_write_text(path, text: str, *, encoding: str = 'utf-8', newline=None,
                      durable: bool = True, tmp_suffix: str = '.tmp') -> None:
    """text 를 path 에 원자적으로 기록. newline 은 open() 과 같은 의미(None=플랫폼 개행)."""
    path, tmp = _prepare(path, tmp_suffix)
    try:
        with open(tmp, 'w', encoding=encoding, newline=newline) as f:
            f.write(text)
            _finish(f, durable)
        os.replace(tmp, path)
    finally:
        _discard(tmp)


def atomic_write_json(path, data, *, indent=2, ensure_ascii=False, separators=None,
                      durable: bool = True, tmp_suffix: str = '.tmp'):
    """data를 path에 원자적으로 기록. 실패 시 tmp 를 지우고 예외 전파(호출자가 처리).

    텍스트 모드로 스트리밍 기록한다(큰 검색 결과를 문자열로 한 번 더 복제하지 않도록).
    """
    path, tmp = _prepare(path, tmp_suffix)
    try:
        with open(tmp, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=ensure_ascii, indent=indent, separators=separators)
            _finish(f, durable)
        os.replace(tmp, path)
    finally:
        _discard(tmp)


def load_json_safe(path: str, default, *, backup_corrupt: bool = True):
    """JSON 로드. 없으면 default, 손상이면 .corrupt로 백업 후 default.

    default 타입(list/dict)과 다른 루트 타입이 로드되면 default 반환.
    """
    if not os.path.exists(path):
        return default
    try:
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        if default is not None and not isinstance(data, type(default)):
            return default
        return data
    except (json.JSONDecodeError, UnicodeDecodeError):
        if backup_corrupt:
            try:
                corrupt = path + '.corrupt'
                if os.path.exists(corrupt):
                    os.remove(corrupt)
                os.replace(path, corrupt)
            except OSError:
                pass
        return default
    except OSError:
        return default
