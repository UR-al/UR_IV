"""PNG 청크 단위 읽기·쓰기 — 픽셀을 다시 인코딩하지 않는다 (Qt·Pillow 비의존).

Gallery 'EXIF 저장'은 예전에 Pillow 로 PNG 를 열어 다시 저장했다. 그러면
  * APNG 는 첫 프레임만 남고 acTL/fcTL/fdAT(프레임·시간·반복)가 사라졌고,
  * 16비트 PNG 는 8비트로 줄었고,
  * gAMA·sRGB·cHRM·sBIT·bKGD·tIME 처럼 손으로 옮기지 않은 청크는 조용히 빠졌다.
여기서는 ``parameters`` 텍스트 청크만 바꾸고 다른 청크는 **바이트 그대로** 옮긴다
(``replace_text_chunk``).

또 Pillow 는 IDAT 뒤에 붙은 텍스트 청크를 픽셀을 전부 디코드해야(``img.text``) 준다.
메타 읽기는 ``read_text_chunks`` 로 청크 머리만 훑어(IDAT 는 seek 로 건너뜀) 뒤쪽 텍스트를 얻는다.

텍스트 청크 해석·기록 규칙은 Pillow(PngImagePlugin)와 같다 — tEXt/zTXt 는 latin-1,
iTXt 는 UTF-8, 기록은 ``PngInfo.add_text(zip=False)`` 처럼 latin-1 로 되면 tEXt, 아니면
압축 없는 iTXt.
"""
from __future__ import annotations

import io
import struct
import zlib
from typing import BinaryIO, Iterator, Optional

PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"
TEXT_CHUNK_TYPES = frozenset({b"tEXt", b"zTXt", b"iTXt"})
# 새 텍스트 청크를 이 청크들 가운데 첫 것 바로 앞에 끼운다(헤더 영역 — Pillow 가 열 때 읽는다).
_INSERT_BEFORE = frozenset({b"acTL", b"fcTL", b"IDAT"})
_IMAGE_DATA_TYPES = frozenset({b"IDAT", b"fdAT"})

# PNG 규격상 청크 길이 상한(2^31 - 1)
MAX_CHUNK_LENGTH = 0x7FFFFFFF
# 읽기 상한 — 텍스트 청크 하나(압축 해제 뒤 포함)와 파일당 합계. ComfyUI workflow JSON 도 넉넉히 든다.
# 합계는 **압축 해제한 바이트**로 센다(``read_text_and_exif`` — Pillow MAX_TEXT_MEMORY 와 같은 뜻).
# 예전엔 디스크의 압축 길이만 더해, 16KB 짜리 zTXt 가 16MiB 로 풀리는 청크를 키만 바꿔 수천 개
# 붙인 PNG 한 장이 파일 크기의 ~1000배(최대 ~64GiB)를 메모리에 올렸다(Codex S5 #0).
MAX_TEXT_CHUNK = 16 * 1024 * 1024
MAX_TEXT_TOTAL = 64 * 1024 * 1024


class PngChunkError(ValueError):
    """PNG 가 아니거나 청크 구조가 깨졌다(잘림·길이 초과·IHDR/IEND 없음)."""


def _is_chunk_type(ctype: bytes) -> bool:
    return len(ctype) == 4 and all(65 <= b <= 90 or 97 <= b <= 122 for b in ctype)


def make_chunk(ctype: bytes, payload: bytes) -> bytes:
    """길이·종류·데이터·CRC 를 갖춘 청크 바이트."""
    return (struct.pack(">I", len(payload)) + ctype + payload
            + struct.pack(">I", zlib.crc32(ctype + payload) & 0xFFFFFFFF))


# ─────────────────────────────────────────────
# 텍스트 청크 해석·기록 (Pillow 와 같은 규칙)
# ─────────────────────────────────────────────

# 해제는 압축 입력을 이만큼씩 먹인다(memoryview 조각 — 복사 없음). 깨진 스트림(zlib.error)은 실패한
# 호출이 만든 바이트를 돌려주지 않으므로, 그 조각이 만들 수 있었던 최대치(deflate 최대 팽창비 1032:1,
# zlib 비트 버퍼에 남은 앞 조각 몇 바이트 여유 포함)를 청구한다. 조각이 작을수록 청구가 실제에 가깝다.
_INFLATE_STEP = 1024
_DEFLATE_MAX_RATIO = 1032
_INFLATE_CARRY = 8


def _inflate(data: bytes, limit: int) -> tuple[Optional[bytes], int]:
    """zlib 해제 → (결과, 비용). ``limit`` 을 넘거나 깨졌으면 결과는 None(압축 폭탄 방지).

    비용은 해제로 만든 바이트 수다 — 한도를 넘는 청크는 ``limit + 1``(한도 + 1 바이트에서 멈춘다),
    깨진 스트림은 그때까지 만든 바이트 + 실패한 조각이 만들 수 있었던 최대치(한도 이내). 그래서 한도에서
    터지거나 한참 풀다 깨지는 청크를 수천 개 붙여도 파일당 합계(``read_text_and_exif``)가 CPU 일까지
    막고, 첫 바이트부터 깨진 작은 청크는 수십 KB 만 청구된다(Pillow 는 깨진 압축 텍스트를 빈 값으로 치고
    청구하지 않는다). 예전엔 깨지기만 하면 ``limit + 1``(최대 16MiB)을 청구해, 몇 바이트짜리 손상
    zTXt/iTXt 4개가 IDAT 뒤 예산(64MiB)을 다 써서 뒤의 정상 parameters/prompt/workflow 를 버렸다
    (Codex S5 #0-a). 스트림 끝 뒤의 바이트는 무시하고, 잘린 스트림은 거기까지 푼 값이다(Pillow 와 같게).
    """
    inflater = zlib.decompressobj()
    view = memoryview(data)
    parts: list[bytes] = []
    produced = 0
    for start in range(0, len(view), _INFLATE_STEP):
        piece = view[start:start + _INFLATE_STEP]
        room = limit + 1 - produced               # 늘 1 이상 — 0 은 zlib 에서 '무제한'이다
        try:
            out = inflater.decompress(piece, room)
        except zlib.error:
            return None, produced + min(room, (len(piece) + _INFLATE_CARRY) * _DEFLATE_MAX_RATIO)
        produced += len(out)
        parts.append(out)
        if produced > limit or inflater.unconsumed_tail:
            return None, limit + 1
        if inflater.eof:
            break
    return b"".join(parts), produced


def text_chunk_keyword(payload: bytes) -> bytes:
    """텍스트 청크의 키워드(첫 NUL 앞)."""
    return payload.split(b"\0", 1)[0]


def decode_text_chunk(ctype: bytes, payload: bytes, *, limit: int = MAX_TEXT_CHUNK) -> Optional[tuple[str, str]]:
    """tEXt/zTXt/iTXt 데이터 → (키, 값). 해석할 수 없으면 None. ``limit`` = 압축 해제 상한."""
    return _decode_text_chunk_cost(ctype, payload, limit)[0]


def _decode_text_chunk_cost(ctype: bytes, payload: bytes, limit: int) -> tuple[Optional[tuple[str, str]], int]:
    """``decode_text_chunk`` 와 같은 해석 + 그 비용(읽은 데이터 길이와 압축 해제로 만든 바이트 중 큰 쪽).

    파일당 합계(``MAX_TEXT_TOTAL``)를 이 비용으로 센다. 해석에 실패한 청크도 한 일만큼 청구된다.
    """
    cost = len(payload)
    key_raw, sep, rest = payload.partition(b"\0")
    if not key_raw:
        return None, cost
    try:
        key = key_raw.decode("latin-1", "strict")
    except UnicodeError:
        return None, cost
    if ctype == b"tEXt":
        return (key, (rest if sep else b"").decode("latin-1", "replace")), cost
    if ctype == b"zTXt":
        if not rest:
            return (key, ""), cost
        if rest[0] != 0:          # 알 수 없는 압축 방식
            return None, cost
        value, inflated = _inflate(rest[1:], limit)
        cost = max(cost, inflated)
        return (None if value is None else (key, value.decode("latin-1", "replace"))), cost
    if ctype == b"iTXt":
        if not sep or len(rest) < 2:
            return None, cost
        flag, method, rest = rest[0], rest[1], rest[2:]
        parts = rest.split(b"\0", 2)
        if len(parts) != 3:
            return None, cost
        value = parts[2]
        if flag:
            if method != 0:
                return None, cost
            value, inflated = _inflate(value, limit)
            cost = max(cost, inflated)
            if value is None:
                return None, cost
        try:
            return (key, value.decode("utf-8", "strict")), cost
        except UnicodeError:
            return None, cost
    return None, cost


def encode_text_chunk(key: str, text: str) -> bytes:
    """``PngInfo.add_text(key, text)`` 과 같은 청크: latin-1 로 되면 tEXt, 아니면 압축 없는 iTXt."""
    keyword = str(key).encode("latin-1")
    if not keyword or len(keyword) > 79 or b"\0" in keyword:
        raise ValueError(f"PNG 텍스트 키가 올바르지 않습니다: {key!r}")
    value = str(text)
    try:
        return make_chunk(b"tEXt", keyword + b"\0" + value.encode("latin-1", "strict"))
    except UnicodeEncodeError:
        # 키\0 + 압축 플래그 0 + 압축 방식 0 + 언어 태그 '' \0 + 번역 키 '' \0 + UTF-8 본문
        return make_chunk(b"iTXt", keyword + b"\0" + b"\0\0" + b"\0" + b"\0" + value.encode("utf-8"))


# ─────────────────────────────────────────────
# 청크 목록
# ─────────────────────────────────────────────

def split_chunks(data: bytes) -> tuple[list[tuple[bytes, int, int]], int]:
    """``data`` → ([(종류, 청크 시작, 청크 끝)], IEND 끝 위치). 구조가 깨졌으면 PngChunkError.

    청크 시작~끝은 길이 4바이트부터 CRC 4바이트까지 전체다. IEND 뒤에 붙은 바이트는 호출자가
    ``data[IEND 끝:]`` 으로 그대로 옮긴다.
    """
    if not data.startswith(PNG_SIGNATURE):
        raise PngChunkError("PNG 파일만 메타데이터 수정 가능")
    chunks: list[tuple[bytes, int, int]] = []
    pos = len(PNG_SIGNATURE)
    size = len(data)
    while True:
        if pos + 8 > size:
            raise PngChunkError("PNG 파일이 잘려 있어 메타데이터를 수정할 수 없습니다")
        length, ctype = struct.unpack_from(">I4s", data, pos)
        end = pos + 12 + length
        if length > MAX_CHUNK_LENGTH or not _is_chunk_type(ctype) or end > size:
            raise PngChunkError("PNG 청크 구조가 손상되어 메타데이터를 수정할 수 없습니다")
        if not chunks and ctype != b"IHDR":
            raise PngChunkError("PNG 청크 구조가 손상되어 메타데이터를 수정할 수 없습니다")
        chunks.append((ctype, pos, end))
        pos = end
        if ctype == b"IEND":
            return chunks, pos


def replace_text_chunk(data: bytes, key: str, text: str) -> bytes:
    """``key`` 텍스트 청크(tEXt/zTXt/iTXt, 여러 개면 전부)를 새 청크 하나로 바꾼 PNG 바이트.

    - 새 청크 자리: 헤더 영역(IDAT 앞)에 같은 키 청크가 있었으면 그 첫 자리, 없으면 첫
      acTL/fcTL/IDAT 바로 앞 — Pillow 가 열자마자 ``img.info`` 로 읽는 자리다.
      IDAT 뒤에만 있던 청크도 이 자리로 옮긴다.
    - 나머지 청크(IDAT·APNG 프레임·gAMA·iCCP·eXIf·pHYs·tRNS·다른 텍스트 …)와 IEND 뒤
      바이트는 바이트 그대로 둔다. 픽셀은 다시 인코딩하지 않는다.
    """
    chunks, iend_end = split_chunks(data)
    if not any(ctype == b"IDAT" for ctype, _s, _e in chunks):
        raise PngChunkError("PNG 이미지 데이터(IDAT)가 없어 메타데이터를 수정할 수 없습니다")
    keyword = str(key).encode("latin-1")
    new_chunk = encode_text_chunk(key, text)

    def is_target(ctype: bytes, start: int, end: int) -> bool:
        return ctype in TEXT_CHUNK_TYPES and text_chunk_keyword(data[start + 8:end - 4]) == keyword

    anchor = None
    for index, (ctype, start, end) in enumerate(chunks):
        if ctype in _INSERT_BEFORE:
            anchor = index if anchor is None else anchor
            break
        if is_target(ctype, start, end):
            anchor = index
            break
    parts = [data[:len(PNG_SIGNATURE)]]
    for index, (ctype, start, end) in enumerate(chunks):
        if index == anchor:
            parts.append(new_chunk)
        if is_target(ctype, start, end):
            continue
        parts.append(data[start:end])
    parts.append(data[iend_end:])
    return b"".join(parts)


def _walk(fp: BinaryIO) -> Iterator[tuple[bytes, int, int]]:
    """``fp``(시그니처 바로 뒤)에서 (종류, 데이터 길이, 데이터 위치). 데이터는 읽지 않고 건너뛴다."""
    while True:
        head = fp.read(8)
        if len(head) < 8:
            return
        length, ctype = struct.unpack(">I4s", head)
        if length > MAX_CHUNK_LENGTH or not _is_chunk_type(ctype):
            return
        offset = fp.tell()
        yield ctype, length, offset
        if ctype == b"IEND":
            return
        fp.seek(offset + length + 4)


def read_text_chunks(fp: BinaryIO, *, after_idat: bool = False) -> dict[str, str]:
    """PNG 스트림의 텍스트 청크 {키: 값}. 같은 키는 뒤의 것이 이긴다(Pillow 와 같게).

    ``after_idat`` 면 첫 IDAT 뒤에 있는 것만 — Pillow 가 열 때 ``img.info`` 로 이미 준 헤더
    청크는 건너뛴다. 이미지 데이터는 seek 로 건너뛰어 픽셀을 디코드하지 않는다.
    잘렸거나 깨진 청크를 만나면 거기까지 읽은 것을 돌려준다. PNG 가 아니면 빈 dict.
    ``fp`` 의 위치는 바뀐다(호출자가 필요하면 되돌린다).
    """
    return read_text_and_exif(fp, after_idat=after_idat)[0]


def read_text_and_exif(fp: BinaryIO, *, after_idat: bool = False) -> tuple[dict[str, str], Optional[bytes]]:
    """``read_text_chunks`` 의 텍스트와 첫 eXIf 청크 데이터(IDAT 앞뒤 어디든, 없으면 None)를 한 번에.

    메타 없는 PNG 에서 텍스트와 EXIF(UserComment)를 따로 찾으면 청크 머리를 두 번 훑는다 —
    큰 PNG 는 IDAT 가 수천 개다. Pillow ``getexif()`` 는 헤더에 eXIf 가 없으면 픽셀을 전부
    디코드하므로(``load()``) 그 대신 쓴다. ``fp`` 의 위치는 바뀐다.

    텍스트는 청크 하나당 ``MAX_TEXT_CHUNK``, 파일당 ``MAX_TEXT_TOTAL`` 까지다 — 합계는 **압축을 푼
    바이트**(실패한 해제는 한도만큼)로 세고, 다 쓰면 거기까지 읽은 것을 돌려준다.
    """
    fp.seek(0)
    if fp.read(len(PNG_SIGNATURE)) != PNG_SIGNATURE:
        return {}, None
    out: dict[str, str] = {}
    exif: Optional[bytes] = None
    seen_image_data = False
    # 텍스트 예산 — 청크마다 읽은 길이와 압축 해제로 만든 바이트 중 큰 쪽을 더한다(실패한 해제 포함).
    # 상한은 부를 때 모듈 값을 읽는다(기본 인자로 묶지 않는다 — 테스트가 작게 바꿔 검증한다).
    spent = 0
    for ctype, length, offset in _walk(fp):
        if ctype in _IMAGE_DATA_TYPES:
            seen_image_data = True
            continue
        if ctype == b"eXIf":
            if exif is None and length <= MAX_TEXT_CHUNK:
                fp.seek(offset)
                payload = fp.read(length)
                if len(payload) < length:
                    break
                exif = payload
            continue
        if ctype not in TEXT_CHUNK_TYPES or (after_idat and not seen_image_data):
            continue
        if length > MAX_TEXT_CHUNK:
            continue
        remaining = MAX_TEXT_TOTAL - spent
        if remaining <= 0 or length > remaining:
            break
        fp.seek(offset)
        payload = fp.read(length)
        if len(payload) < length:
            break
        item, cost = _decode_text_chunk_cost(ctype, payload, min(MAX_TEXT_CHUNK, remaining))
        spent += cost
        if item is not None:
            out[item[0]] = item[1]
    return out, exif


def find_chunk(data, ctype: bytes, *, before_idat: bool = False) -> Optional[bytes]:
    """``data``(PNG 바이트 — bytes·uint8 배열, 또는 seek 되는 스트림)에서 ``ctype`` 첫 청크의 데이터.

    없거나 PNG 가 아니면 None. ``before_idat`` 면 첫 IDAT 앞(헤더 영역)만 본다.
    스트림을 넘기면 복사하지 않고 그 위치를 바꾼다.
    """
    fp = data if hasattr(data, "read") else io.BytesIO(data)
    fp.seek(0)
    if fp.read(len(PNG_SIGNATURE)) != PNG_SIGNATURE:
        return None
    for found, length, offset in _walk(fp):
        if before_idat and found == b"IDAT":
            return None
        if found == ctype:
            fp.seek(offset)
            payload = fp.read(length)
            return payload if len(payload) == length else None
    return None
