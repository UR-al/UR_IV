# core/cache_cleanup.py
"""이미지 캐시 정리 — 순수 파일시스템 유틸(테스트 가능). Qt 의존 없음.

`image_cache/`에 정리 루틴이 아예 없어서 무한히 쌓였다. 실측(2026-07):
    image_cache 전체     812 MB
    ├─ thumbs        96,641 files / 389 MB   ← NTFS 단일 디렉터리 10만 파일
    ├─ editor_temp      115 files / 256 MB   ← 편집 1회당 풀해상도 PNG 1장
    └─ compare            1 file  /  15 MB

thumbs는 개수가 문제라 sha1 앞 2자리로 샤딩한다. 디렉터리당 ~380개로 떨어져
`os.path.exists` 조회와 탐색기 접근이 모두 빨라진다. (지금 캐시는 처음부터 샤딩된
image_cache/thumbs_v2 — 옛 평면 폴더 정리는 core.legacy_thumb_cache 가 한 번 한다.)
"""
import os
import time

# 썸네일 샤딩: sha1 앞 2자리 → 256개 하위 디렉터리
SHARD_PREFIX_LEN = 2


def shard_path(base_dir: str, digest: str, ext: str = '.jpg') -> str:
    """sha1 해시로 샤딩된 캐시 경로. 디렉터리는 만들지 않는다(호출자 책임)."""
    digest = str(digest)
    prefix = digest[:SHARD_PREFIX_LEN] or '00'
    return os.path.join(base_dir, prefix, f"{digest}{ext}")


def ensure_shard_dir(path: str) -> None:
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
    except OSError:
        pass


def source_signature(path: str):
    """원본 서명 ``(st_mtime_ns, st_size)``. 못 읽으면 None."""
    try:
        stat = os.stat(path)
    except OSError:
        return None
    return stat.st_mtime_ns, stat.st_size


# 썸네일 JPEG 의 COM(주석) 마커에 '어느 원본으로 만들었는지'를 적는다.
THUMB_SIGNATURE_PREFIX = b'aistudio-src:'
_SIGNATURE_READ_BYTES = 4096   # COM 은 SOI·APP0 바로 뒤에 온다 — 머리만 읽는다


def thumb_signature_comment(signature) -> bytes:
    """``source_signature`` → 썸네일 JPEG 주석 바이트."""
    mtime_ns, size = signature
    return THUMB_SIGNATURE_PREFIX + f'{int(mtime_ns)}:{int(size)}'.encode('ascii')


def read_thumb_signature(thumb: str):
    """썸네일 JPEG 주석에 적힌 원본 서명 ``(mtime_ns, size)``. 없거나(옛 썸네일) 못 읽으면 None."""
    try:
        with open(thumb, 'rb') as handle:
            head = handle.read(_SIGNATURE_READ_BYTES)
    except OSError:
        return None
    if not head.startswith(b'\xff\xd8'):
        return None
    pos = 2
    while pos + 4 <= len(head):
        if head[pos] != 0xFF:
            return None
        marker = head[pos + 1]
        if marker == 0xFF:            # 채움 바이트
            pos += 1
            continue
        if marker in (0xD9, 0xDA):    # EOI · SOS — 여기부터는 이미지 데이터
            return None
        if marker == 0x01 or 0xD0 <= marker <= 0xD7:
            pos += 2
            continue
        length = int.from_bytes(head[pos + 2:pos + 4], 'big')
        if length < 2:
            return None
        payload = head[pos + 4:pos + 2 + length]
        if marker == 0xFE and payload.startswith(THUMB_SIGNATURE_PREFIX):
            try:
                mtime_ns, size = payload[len(THUMB_SIGNATURE_PREFIX):].decode('ascii').split(':')
                return int(mtime_ns), int(size)
            except ValueError:
                return None
        pos += 2 + length
    return None


def thumb_is_stale(source: str, thumb: str) -> bool:
    """썸네일을 (다시) 만들어야 하는지 — 없거나, 만든 뒤 원본이 바뀌었으면.

    캐시 키는 경로@폭이라, 예전에는 파일이 있기만 하면 옛 그림을 계속 보여 줬다
    (에디터 '저장'이 사본을 갱신하거나, '다른 이름으로 저장'이 기존 파일을 덮어쓴 경우).
    판정은 썸네일 주석에 적힌 원본 서명(mtime_ns, 크기)과 지금 원본 서명의 **일치**다.
    예전 '원본 mtime > 썸네일 mtime' 순서 비교는 렌더 중에 덮어쓴 원본(에디터 원자 저장은 임시
    파일의 옛 mtime 을 가진다)이나 더 옛 mtime 을 가진 파일로 바꾼 원본(copy2·탐색기 덮어쓰기)을
    놓쳐 옛 그림을 새것처럼 남겼다. 서명이 없는 옛 썸네일은 한 번 다시 만든다.
    원본을 못 읽으면(지워짐 등) 있는 썸네일을 그대로 쓴다. 웹 모드 /thumbnail 과 같은 규칙.
    """
    if not os.path.isfile(thumb):
        return True
    current = source_signature(source)
    if current is None:
        return False
    return read_thumb_signature(thumb) != current


def _entries_by_mtime(directory: str, exts=None):
    """(mtime, size, path) 목록을 오래된 순으로. scandir의 stat 캐시 사용."""
    out = []
    try:
        with os.scandir(directory) as scan:
            for entry in scan:
                if not entry.is_file():
                    continue
                if exts and not entry.name.lower().endswith(exts):
                    continue
                try:
                    stat = entry.stat()
                    out.append((stat.st_mtime, stat.st_size, entry.path))
                except OSError:
                    continue
    except (OSError, FileNotFoundError):
        return []
    out.sort(key=lambda item: item[0])
    return out


def _unlink(path: str) -> bool:
    try:
        os.remove(path)
        return True
    except OSError:
        return False


def prune_editor_temp(directory: str, keep: int = 40, max_age_hours: float = 24.0) -> int:
    """편집 임시본 정리 — 최근 `keep`개는 남기고 나머지 중 오래된 것을 지운다.

    최근 것을 남기는 이유: undo 스택이 이 파일 경로를 그대로 참조하므로,
    작업 중인 히스토리를 지우면 undo가 깨진다. keep은 EditorView의 MAX_UNDO(30)보다
    넉넉해야 한다.

    반환: 지운 파일 수.
    """
    if keep < 0:
        keep = 0
    entries = _entries_by_mtime(directory, exts=('.png', '.jpg', '.jpeg', '.webp'))
    if len(entries) <= keep:
        return 0

    cutoff = time.time() - max(0.0, float(max_age_hours)) * 3600.0
    # 최근 keep개는 무조건 보존, 그 앞쪽(오래된 것)만 후보
    candidates = entries[:-keep] if keep else entries
    removed = 0
    for mtime, _size, path in candidates:
        if mtime <= cutoff:
            removed += _unlink(path)
    return removed


def prune_by_total_size(directory: str, max_bytes: int, *, recursive: bool = False) -> int:
    """디렉터리 총 용량이 상한을 넘으면 오래된 것부터 지운다 (LRU 근사).

    썸네일 캐시처럼 '언제든 다시 만들 수 있는' 캐시에만 쓴다.
    반환: 지운 파일 수.
    """
    if recursive:
        entries = []
        for root, _dirs, files in os.walk(directory):
            for name in files:
                path = os.path.join(root, name)
                try:
                    stat = os.stat(path)
                    entries.append((stat.st_mtime, stat.st_size, path))
                except OSError:
                    continue
        entries.sort(key=lambda item: item[0])
    else:
        entries = _entries_by_mtime(directory)

    total = sum(size for _m, size, _p in entries)
    if total <= max_bytes:
        return 0

    removed = 0
    for _mtime, size, path in entries:
        if total <= max_bytes:
            break
        if _unlink(path):
            total -= size
            removed += 1
    return removed


def prune_thumbs(directory: str, max_bytes: int = 300 * 1024 * 1024) -> int:
    """썸네일 캐시 상한(기본 300MB). 샤딩된 하위 디렉터리까지 훑는다."""
    return prune_by_total_size(directory, max_bytes, recursive=True)
