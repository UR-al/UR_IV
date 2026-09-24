"""PNG 청크 단위 읽기·쓰기(core.png_chunks) — 픽셀을 다시 인코딩하지 않는 메타 수정의 바탕."""
import io
import struct
import unittest
import zlib

from PIL import Image, PngImagePlugin

from core.png_chunks import (
    PNG_SIGNATURE,
    PngChunkError,
    decode_text_chunk,
    encode_text_chunk,
    find_chunk,
    make_chunk,
    read_text_chunks,
    replace_text_chunk,
    split_chunks,
)


def png_bytes(image=None, *, text=None, extra_chunks=(), **save_kwargs) -> bytes:
    info = PngImagePlugin.PngInfo()
    for key, value in (text or {}).items():
        info.add_text(key, value)
    for ctype, data in extra_chunks:
        info.add(ctype, data)
    buf = io.BytesIO()
    (image or Image.new("RGB", (4, 3), "red")).save(buf, format="PNG", pnginfo=info, **save_kwargs)
    return buf.getvalue()


def insert_before_iend(data: bytes, chunk: bytes) -> bytes:
    iend = data.rindex(b"IEND") - 4
    return data[:iend] + chunk + data[iend:]


def chunk_list(data: bytes) -> list[tuple[bytes, bytes]]:
    chunks, _end = split_chunks(data)
    return [(ctype, data[start:end]) for ctype, start, end in chunks]


def keyword(chunk: bytes) -> bytes:
    return chunk[8:-4].split(b"\0", 1)[0]


class TextChunkCodecTests(unittest.TestCase):
    def test_latin1_text_is_text_and_other_text_is_uncompressed_itxt_like_pillow(self):
        for value, ctype in (("a cat, 1girl", b"tEXt"), ("한글 고양이", b"iTXt"), ("café", b"tEXt")):
            with self.subTest(value=value):
                chunk = encode_text_chunk("parameters", value)
                self.assertEqual(chunk[4:8], ctype)
                self.assertEqual(decode_text_chunk(ctype, chunk[8:-4]), ("parameters", value))
                # Pillow 가 같은 청크를 같은 값으로 읽는다
                data = insert_before_iend(png_bytes(), chunk)
                with Image.open(io.BytesIO(data)) as img:
                    self.assertEqual(img.text["parameters"], value)

    def test_decodes_pillow_ztxt_and_compressed_itxt(self):
        info = PngImagePlugin.PngInfo()
        info.add_text("z", "zipped latin", zip=True)
        info.add_itxt("i", "압축 iTXt", zip=True)
        buf = io.BytesIO()
        Image.new("RGB", (2, 2)).save(buf, format="PNG", pnginfo=info)
        found = read_text_chunks(io.BytesIO(buf.getvalue()))
        self.assertEqual(found, {"z": "zipped latin", "i": "압축 iTXt"})

    def test_bad_or_oversized_chunks_are_skipped_not_raised(self):
        self.assertIsNone(decode_text_chunk(b"tEXt", b"\0no key"))
        self.assertIsNone(decode_text_chunk(b"zTXt", b"k\0\x01junk"))            # 모르는 압축 방식
        self.assertIsNone(decode_text_chunk(b"iTXt", b"k\0\0\0en\0\0\xff\xfe"))  # UTF-8 아님
        bomb = b"k\0\0" + zlib.compress(b"x" * 4096)
        self.assertIsNone(decode_text_chunk(b"zTXt", bomb, limit=1024))
        self.assertEqual(decode_text_chunk(b"zTXt", bomb, limit=8192), ("k", "x" * 4096))

    def test_keyword_must_be_a_valid_png_keyword(self):
        for bad in ("", "x" * 80, "a\0b"):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                encode_text_chunk(bad, "v")


class ReplaceTextChunkTests(unittest.TestCase):
    def test_header_chunk_is_replaced_in_place_and_everything_else_is_byte_identical(self):
        data = png_bytes(text={"parameters": "old", "workflow": "{}"},
                         extra_chunks=[(b"gAMA", struct.pack(">I", 45455)), (b"tIME", b"\x07\xea\x09\x18\x0c\x00\x00")])
        data = insert_before_iend(data, make_chunk(b"tEXt", b"Comment\0after idat"))
        out = replace_text_chunk(data, "parameters", "new")
        before, after = chunk_list(data), chunk_list(out)
        index = [keyword(c) for _t, c in before].index(b"parameters")
        self.assertEqual(after[index], (b"tEXt", encode_text_chunk("parameters", "new")))
        self.assertEqual(after[:index] + after[index + 1:], before[:index] + before[index + 1:])

    def test_duplicates_and_trailing_copies_collapse_into_one_header_chunk(self):
        info = PngImagePlugin.PngInfo()
        info.add_text("parameters", "zipped old", zip=True)
        buf = io.BytesIO()
        Image.new("RGB", (3, 3), "blue").save(buf, format="PNG", pnginfo=info)
        data = insert_before_iend(buf.getvalue(), make_chunk(b"tEXt", b"parameters\0trailing old"))
        out = replace_text_chunk(data, "parameters", "only one")
        kinds = [(t, keyword(c)) for t, c in chunk_list(out)]
        self.assertEqual([k for k in kinds if k[1] == b"parameters"], [(b"tEXt", b"parameters")])
        first_idat = [t for t, _k in kinds].index(b"IDAT")
        self.assertLess(kinds.index((b"tEXt", b"parameters")), first_idat)
        with Image.open(io.BytesIO(out)) as img:
            self.assertEqual(img.info["parameters"], "only one")   # 열자마자 헤더에서 읽힌다

    def test_missing_chunk_is_inserted_before_the_animation_control_or_image_data(self):
        frames = [Image.new("RGB", (4, 4), c) for c in ("red", "green")]
        buf = io.BytesIO()
        frames[0].save(buf, format="PNG", save_all=True, append_images=frames[1:], duration=50, loop=0)
        out = replace_text_chunk(buf.getvalue(), "parameters", "x")
        types = [t for t, _c in chunk_list(out)]
        self.assertEqual(types[types.index(b"tEXt") + 1], b"acTL")
        still = replace_text_chunk(png_bytes(), "parameters", "x")
        types = [t for t, _c in chunk_list(still)]
        self.assertEqual(types[types.index(b"tEXt") + 1], b"IDAT")

    def test_bytes_after_iend_are_kept(self):
        data = png_bytes() + b"appended-trailer"
        self.assertTrue(replace_text_chunk(data, "parameters", "x").endswith(b"appended-trailer"))

    def test_non_png_and_broken_structures_are_refused(self):
        good = png_bytes()
        idat = good.index(b"IDAT") - 4
        cases = {
            "not png": b"\xff\xd8\xff\xe0 jpeg",
            "truncated": good[:-20],
            "length past end": good[:idat] + struct.pack(">I", 1 << 30) + good[idat + 4:],
            "no ihdr": PNG_SIGNATURE + good[good.index(b"IDAT") - 4:],
            "no idat": PNG_SIGNATURE + good[8:idat] + make_chunk(b"IEND", b""),
            "bad type": good[:idat + 4] + b"ID@T" + good[idat + 8:],
        }
        for name, data in cases.items():
            with self.subTest(name=name), self.assertRaises(PngChunkError):
                replace_text_chunk(data, "parameters", "x")
        self.assertTrue(issubclass(PngChunkError, ValueError))


class ReadTextChunksTests(unittest.TestCase):
    def test_after_idat_returns_only_trailing_chunks_and_last_one_wins(self):
        data = png_bytes(text={"parameters": "header"})
        data = insert_before_iend(data, make_chunk(b"tEXt", b"prompt\0first"))
        data = insert_before_iend(data, make_chunk(b"iTXt", b"prompt\0\0\0\0\0" + "둘째".encode("utf-8")))
        self.assertEqual(read_text_chunks(io.BytesIO(data), after_idat=True), {"prompt": "둘째"})
        self.assertEqual(read_text_chunks(io.BytesIO(data)), {"parameters": "header", "prompt": "둘째"})

    def test_truncated_or_non_png_streams_return_what_was_read(self):
        data = insert_before_iend(png_bytes(text={"a": "1"}), make_chunk(b"tEXt", b"b\x002"))
        self.assertEqual(read_text_chunks(io.BytesIO(data[:-30])), {"a": "1"})
        self.assertEqual(read_text_chunks(io.BytesIO(b"GIF89a....")), {})

    def test_find_chunk(self):
        data = png_bytes(extra_chunks=[(b"gAMA", struct.pack(">I", 45455))])
        self.assertEqual(find_chunk(data, b"gAMA"), struct.pack(">I", 45455))
        self.assertIsNone(find_chunk(data, b"eXIf"))
        trailing = insert_before_iend(data, make_chunk(b"eXIf", b"II*\0"))
        self.assertEqual(find_chunk(trailing, b"eXIf"), b"II*\0")
        self.assertIsNone(find_chunk(trailing, b"eXIf", before_idat=True))
        self.assertIsNone(find_chunk(b"not a png", b"IHDR"))

    def test_text_and_exif_in_one_walk(self):
        from core.png_chunks import read_text_and_exif
        data = png_bytes(text={"parameters": "header"})
        self.assertEqual(read_text_and_exif(io.BytesIO(data), after_idat=True), ({}, None))
        # IDAT 뒤 eXIf·텍스트 — 한 번 훑기로 둘 다(첫 eXIf 가 이긴다)
        trailing = insert_before_iend(data, make_chunk(b"eXIf", b"II*\0first"))
        trailing = insert_before_iend(trailing, make_chunk(b"tEXt", b"prompt\0tail"))
        trailing = insert_before_iend(trailing, make_chunk(b"eXIf", b"II*\0second"))
        self.assertEqual(read_text_and_exif(io.BytesIO(trailing), after_idat=True),
                         ({"prompt": "tail"}, b"II*\0first"))
        # 헤더 eXIf 도 찾는다(after_idat 는 텍스트에만 적용)
        plain = png_bytes()
        idat = plain.index(b"IDAT") - 4   # Pillow PngInfo 는 eXIf 를 싣지 않는다 — 손으로 IDAT 앞에 끼운다
        header = plain[:idat] + make_chunk(b"eXIf", b"MM\0*head") + plain[idat:]
        self.assertEqual(read_text_and_exif(io.BytesIO(header), after_idat=True), ({}, b"MM\0*head"))
        self.assertEqual(read_text_and_exif(io.BytesIO(b"GIF89a....")), ({}, None))
        # read_text_chunks 는 같은 텍스트를 준다
        self.assertEqual(read_text_chunks(io.BytesIO(trailing), after_idat=True), {"prompt": "tail"})


class TextBudgetTests(unittest.TestCase):
    """파일당 텍스트 합계는 **압축을 푼 바이트**로 센다(Codex S5 #0).

    예전엔 디스크의 압축 길이만 더해, 16KB 짜리 zTXt 가 16MiB 로 풀리는 청크를 키만 바꿔 여러 개
    IDAT 뒤에 붙인 PNG(Pillow 의 텍스트 한도는 IDAT 앞에만 걸린다)가 파일 크기의 ~1000배를
    메모리에 올렸고, 한도에서 터지는 청크는 버려지면서도 매번 끝까지 풀려 CPU 를 태웠다.
    상한을 작게 바꿔(부를 때 모듈 값을 읽는다) 작은 파일로 검증한다.
    """

    CHUNK = 4096
    TOTAL = 8192

    def _limits(self):
        from unittest import mock
        from core import png_chunks
        for name, value in (("MAX_TEXT_CHUNK", self.CHUNK), ("MAX_TEXT_TOTAL", self.TOTAL)):
            patcher = mock.patch.object(png_chunks, name, value)
            patcher.start()
            self.addCleanup(patcher.stop)
        return png_chunks

    @staticmethod
    def ztxt(key: bytes, raw: bytes) -> bytes:
        return make_chunk(b"zTXt", key + b"\0\0" + zlib.compress(raw))

    def test_distinct_keys_stop_at_the_decompressed_total(self):
        png_chunks = self._limits()
        data = png_bytes()
        for i in range(4):
            data = insert_before_iend(data, self.ztxt(b"k%d" % i, b"x" * self.CHUNK))
        self.assertLess(len(data), self.TOTAL, "전제: 압축 길이 합계로는 한도에 닿지 않는다")
        found, _exif = png_chunks.read_text_and_exif(io.BytesIO(data), after_idat=True)
        self.assertEqual(sorted(found), ["k0", "k1"])
        self.assertLessEqual(sum(len(v) for v in found.values()), self.TOTAL)

    def test_chunks_that_blow_the_per_chunk_limit_are_charged_and_stop_the_walk(self):
        # 같은 키로 한도+1 바이트로 풀리는 청크 10개 뒤에 정상 parameters — 실패한 해제도 청구돼
        # 두 번째에서 예산이 바닥나고, 나머지는 풀지 않는다(예전: 10개 전부 한도까지 풀었다).
        from unittest import mock
        png_chunks = self._limits()
        data = png_bytes()
        for _ in range(10):
            data = insert_before_iend(data, self.ztxt(b"bomb", b"x" * (self.CHUNK + 1)))
        data = insert_before_iend(data, make_chunk(b"tEXt", b"parameters\0after bombs"))
        with mock.patch.object(png_chunks, "_inflate", wraps=png_chunks._inflate) as inflate:
            found, _exif = png_chunks.read_text_and_exif(io.BytesIO(data), after_idat=True)
        self.assertEqual(found, {})
        self.assertEqual(inflate.call_count, 2)

    def test_uncompressed_text_counts_too_and_a_chunk_past_the_rest_of_the_budget_stops_the_walk(self):
        png_chunks = self._limits()
        data = png_bytes()
        data = insert_before_iend(data, make_chunk(b"tEXt", b"a\0" + b"x" * 3000))
        data = insert_before_iend(data, make_chunk(b"tEXt", b"b\0" + b"y" * 3000))
        data = insert_before_iend(data, make_chunk(b"tEXt", b"c\0" + b"z" * 3000))   # 남은 예산(~2188) 초과
        data = insert_before_iend(data, make_chunk(b"tEXt", b"d\0small"))
        found = png_chunks.read_text_chunks(io.BytesIO(data), after_idat=True)
        self.assertEqual(sorted(found), ["a", "b"])

    def test_mixed_text_under_the_budget_is_read_in_full(self):
        png_chunks = self._limits()
        data = png_bytes(text={"header": "h"})
        data = insert_before_iend(data, make_chunk(b"tEXt", b"parameters\0" + b"p" * 1000))
        data = insert_before_iend(data, self.ztxt(b"prompt", b"{" + b"q" * 2000 + b"}"))
        itxt = b"workflow\0\x01\0\0\0" + zlib.compress("워크플로".encode("utf-8") * 100)
        data = insert_before_iend(data, make_chunk(b"iTXt", itxt))
        found = png_chunks.read_text_chunks(io.BytesIO(data), after_idat=True)
        self.assertEqual(found, {"parameters": "p" * 1000, "prompt": "{" + "q" * 2000 + "}",
                                 "workflow": "워크플로" * 100})

    def test_default_limits_bound_a_real_ztxt_bomb(self):
        # 실제 상한(16MiB/64MiB) — 서로 다른 키 6개 × 16MiB 가 ~100KB 파일에 든다. 합계는 64MiB 이하.
        from core.png_chunks import MAX_TEXT_CHUNK, MAX_TEXT_TOTAL, read_text_and_exif
        payload = zlib.compress(b"\0" * MAX_TEXT_CHUNK, 9)
        data = png_bytes()
        for i in range(6):
            data = insert_before_iend(data, make_chunk(b"zTXt", b"k%d\0\0" % i + payload))
        found, _exif = read_text_and_exif(io.BytesIO(data), after_idat=True)
        self.assertLessEqual(sum(len(v) for v in found.values()), MAX_TEXT_TOTAL)
        self.assertEqual(len(found), MAX_TEXT_TOTAL // MAX_TEXT_CHUNK)

    # 첫 바이트부터 깨진 deflate(알 수 없는 블록 종류) — zlib.error
    _CORRUPT = b"x\x9c\xff\xff\xff\xff\xff"

    def test_small_corrupt_compressed_text_does_not_starve_later_metadata(self):
        """(Codex S5 #0-a) 예전엔 깨진 해제마다 한도(16MiB)+1 을 청구해, 몇 바이트짜리 손상 zTXt/iTXt
        4개가 IDAT 뒤 예산(64MiB)을 다 써서 뒤의 정상 parameters/prompt/workflow 가 사라졌다.
        실제 상한으로 — 손상 청크 32개(한 일 없음) 뒤의 메타가 전부 읽혀야 한다."""
        from core.png_chunks import read_text_and_exif
        data = png_bytes()
        for i in range(16):
            data = insert_before_iend(data, make_chunk(b"zTXt", b"bad%d\0\0" % i + self._CORRUPT))
            data = insert_before_iend(data, make_chunk(b"iTXt", b"badi%d\0\x01\0\0\0" % i + self._CORRUPT))
        data = insert_before_iend(data, make_chunk(b"tEXt", b"parameters\0a cat, masterpiece"))
        data = insert_before_iend(data, self.ztxt(b"prompt", b'{"3": {"class_type": "KSampler"}}'))
        workflow = '{"nodes": ["워크플로"]}'
        data = insert_before_iend(data, make_chunk(b"iTXt", b"workflow\0\x01\0\0\0"
                                                   + zlib.compress(workflow.encode("utf-8"))))
        found, _exif = read_text_and_exif(io.BytesIO(data), after_idat=True)
        self.assertEqual(found, {"parameters": "a cat, masterpiece",
                                 "prompt": '{"3": {"class_type": "KSampler"}}', "workflow": workflow})

    def test_inflate_charges_the_work_done_not_the_whole_limit(self):
        from core.png_chunks import _inflate
        limit = 16 * 1024 * 1024
        # 첫 바이트부터 깨진 작은 스트림 — 수십 KB 이하(예전: 16MiB+1)
        value, cost = _inflate(self._CORRUPT, limit)
        self.assertIsNone(value)
        self.assertLess(cost, 64 * 1024)

        def broken_checksum(raw: bytes, level: int) -> bytes:
            packed = zlib.compress(raw, level)
            return packed[:-4] + bytes(b ^ 0xFF for b in packed[-4:])   # 끝의 adler32 만 틀림

        # 끝까지 풀고서야 깨지는 스트림 — 푼 만큼은 반드시 청구된다(CPU 상한이 그대로 선다)
        for name, raw, level in (("zeros", b"\0" * (3 * 1024 * 1024), 9),
                                 ("stored", bytes(range(256)) * 1200, 0),
                                 ("text", b'{"class_type": "KSampler"}, ' * 20000, 6)):
            with self.subTest(name=name):
                value, cost = _inflate(broken_checksum(raw, level), limit)
                self.assertIsNone(value)
                self.assertGreaterEqual(cost, len(raw))
                self.assertLessEqual(cost, limit + 1)
                # 멀쩡한 같은 스트림은 조각으로 먹여도 그대로 풀리고, 비용은 정확히 푼 바이트다
                self.assertEqual(_inflate(zlib.compress(raw, level), limit), (raw, len(raw)))
        # 폭탄은 그대로 한도 + 1(한도 + 1 바이트에서 멈춘다)
        self.assertEqual(_inflate(zlib.compress(b"x" * 5000), 4096), (None, 4097))
        self.assertEqual(_inflate(zlib.compress(b"x" * 4096), 4096), (b"x" * 4096, 4096))
        # 스트림 끝 뒤 바이트는 무시, 잘린 스트림은 거기까지(Pillow 와 같게), 빈 입력은 빈 값
        self.assertEqual(_inflate(zlib.compress(b"abc") + b"junk", limit), (b"abc", 3))
        long_text = bytes(range(256)) * 64
        partial, cost = _inflate(zlib.compress(long_text, 0)[:5000], limit)
        self.assertTrue(partial and long_text.startswith(partial))
        self.assertEqual(cost, len(partial))
        self.assertEqual(_inflate(b"", limit), (b"", 0))


if __name__ == "__main__":
    unittest.main()
