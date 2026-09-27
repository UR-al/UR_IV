"""체크포인트 종류 판정(core/anima_model_kind, P9) — safetensors 헤더·이름·캐시·락.

가짜 safetensors(8바이트 길이 + JSON 헤더)를 임시 폴더에 만든다. torch·GPU 없음.
"""
import json
import os
import struct
import tempfile
import threading
import time
import unittest
from unittest import mock

from core import anima_model_kind as amk


def write_safetensors(path, header, body=b"\0" * 16):
    raw = json.dumps(header).encode("utf-8")
    with open(path, "wb") as handle:
        handle.write(struct.pack("<Q", len(raw)))
        handle.write(raw)
        handle.write(body)
    return path


V2_HEADER = {"__metadata__": {"architecture": amk.BUNDLE_ARCHITECTURE, amk.BUNDLE_FORMAT_KEY: "1"},
             "net.llm_adapter.blocks.0.weight": {"dtype": "BF16", "shape": [1], "data_offsets": [0, 2]}}
ANIMA_HEADER = {"__metadata__": {"modelspec.architecture": "anima-preview"},
                "net.llm_adapter.blocks.0.weight": {"dtype": "BF16", "shape": [1], "data_offsets": [0, 2]},
                "net.blocks.0.weight": {"dtype": "BF16", "shape": [1], "data_offsets": [2, 4]}}
OTHER_HEADER = {"model.diffusion_model.input_blocks.0.weight": {"dtype": "F16", "shape": [1], "data_offsets": [0, 2]}}


def _old_compiler_kind(value):
    """P9 전 core/comfy_workflow_compiler 의 _looks_like_anima38_v2_bundle / _looks_like_anima_model(그대로 옮김)."""
    name = os.path.basename(str(value or "").replace("\\", "/")).strip().casefold()
    family = "anima" in name and any(marker in name for marker in ("3.8b", "3-8b", "3_8b"))
    release = any(marker in name for marker in ("-v2", "_v2", ".v2", "-v1.1", "_v1.1", ".v1.1"))
    if family and release:
        return "v2"
    return "anima" if "anima" in name else "unknown"


class _CacheReset(unittest.TestCase):
    def setUp(self):
        amk.forget_kinds()
        amk.clear_header_cache()
        self.addCleanup(amk.forget_kinds)
        self.addCleanup(amk.clear_header_cache)
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)

    def path(self, *parts):
        target = os.path.join(self.tmp.name, *parts)
        os.makedirs(os.path.dirname(target), exist_ok=True)
        return target


class HeaderKindTests(_CacheReset):
    def test_bundle_anima_and_other_like_the_extension(self):
        self.assertEqual(amk.header_kind(write_safetensors(self.path("v2.safetensors"), V2_HEADER)), amk.KIND_V2)
        self.assertEqual(amk.header_kind(write_safetensors(self.path("a.safetensors"), ANIMA_HEADER)), amk.KIND_ANIMA)
        self.assertEqual(amk.header_kind(write_safetensors(self.path("o.safetensors"), OTHER_HEADER)), amk.KIND_OTHER)

    def test_anima_key_with_the_comfy_prefix_and_wrong_bundle_format(self):
        comfy_style = {"model.diffusion_model.llm_adapter.x": {"dtype": "F16", "shape": [1], "data_offsets": [0, 2]}}
        self.assertEqual(amk.header_kind(write_safetensors(self.path("c.safetensors"), comfy_style)), amk.KIND_ANIMA)
        wrong_format = dict(V2_HEADER, __metadata__={"architecture": amk.BUNDLE_ARCHITECTURE,
                                                     amk.BUNDLE_FORMAT_KEY: "2"})
        # 확장 bundle_metadata 는 형식이 다르면 번들이 아니다 — Anima 키가 있으니 비 번들 Anima
        self.assertEqual(amk.header_kind(write_safetensors(self.path("w.safetensors"), wrong_format)), amk.KIND_ANIMA)
        lookalike = {"net.not_llm_adapter.x": {"dtype": "F16", "shape": [1], "data_offsets": [0, 2]}}
        self.assertEqual(amk.header_kind(write_safetensors(self.path("l.safetensors"), lookalike)), amk.KIND_OTHER)

    def test_unreadable_files_are_none(self):
        broken = self.path("broken.safetensors")
        with open(broken, "wb") as handle:
            handle.write(struct.pack("<Q", 5) + b"{nope")
        huge = self.path("huge.safetensors")
        with open(huge, "wb") as handle:
            handle.write(struct.pack("<Q", amk.MAX_HEADER_BYTES + 1) + b"{}")
        short = self.path("short.safetensors")
        with open(short, "wb") as handle:
            handle.write(b"\x01\x02")
        zip_like = self.path("model.ckpt")
        with open(zip_like, "wb") as handle:
            handle.write(b"PK\x03\x04" + b"\0" * 64)
        for label, path in (("깨진 JSON", broken), ("64 MB 초과", huge), ("8바이트 미만", short),
                            ("다른 형식", zip_like), ("없는 경로", self.path("missing.safetensors")),
                            ("빈 값", ""), ("None", None)):
            with self.subTest(label):
                self.assertIsNone(amk.header_kind(path))

    def test_header_over_the_limit_is_not_read(self):
        path = write_safetensors(self.path("big.safetensors"), ANIMA_HEADER)
        with mock.patch.object(amk, "MAX_HEADER_BYTES", 16):
            self.assertIsNone(amk.header_kind(path))

    def test_korean_path(self):
        path = write_safetensors(self.path("한글 폴더", "애니마 3.8B.safetensors"), V2_HEADER)
        self.assertEqual(amk.header_kind(path), amk.KIND_V2)

    def test_cache_is_invalidated_by_size_or_mtime(self):
        path = write_safetensors(self.path("m.safetensors"), OTHER_HEADER)
        self.assertEqual(amk.header_kind(path), amk.KIND_OTHER)
        with mock.patch.object(amk, "_read_header_kind", side_effect=AssertionError("다시 읽음")):
            self.assertEqual(amk.header_kind(path), amk.KIND_OTHER)          # 같은 파일 — 캐시
        write_safetensors(path, ANIMA_HEADER)
        stamp = time.time() + 5
        os.utime(path, (stamp, stamp))
        self.assertEqual(amk.header_kind(path), amk.KIND_ANIMA)              # 바뀐 파일 — 다시 읽는다


class NameAndClassifyTests(_CacheReset):
    NAMES = ("Anima-3.8B-v1.1.safetensors", "Anima-3.8B-v1.1.safetensors [e8b2c9a1]", "sub\\anima_3_8b_v2.safetensors",
             "Anima-3-8B.v2.safetensors", "Anima-3.8B.safetensors", "anima_baseV10.safetensors",
             "Anima-2.9B-preview-v1.safetensors", "UR_ANIMA_V1.0-000008.safetensors", "krea2Anime_v15.safetensors",
             "sdxl_base.safetensors", "", None, "diffusion_models/anima-v2.safetensors")

    def test_name_kind_matches_the_old_compiler_heuristic(self):
        for name in self.NAMES:
            with self.subTest(name=name):
                self.assertEqual(amk.name_kind(name), _old_compiler_kind(name))
        self.assertNotIn(amk.KIND_OTHER, {amk.name_kind(name) for name in self.NAMES})   # 이름으로 other 판정 안 함

    def test_classify_prefers_the_title_cache_then_header_then_name(self):
        path = write_safetensors(self.path("renamed.safetensors"), V2_HEADER)
        self.assertEqual(amk.classify("renamed.safetensors", cached_only=True), amk.KIND_UNKNOWN)   # 디스크 안 봄
        self.assertEqual(amk.classify("renamed.safetensors", path), amk.KIND_V2)                    # 헤더
        with mock.patch.object(amk, "header_kind", side_effect=AssertionError("디스크")):
            self.assertEqual(amk.classify("renamed.safetensors", cached_only=True), amk.KIND_V2)    # 제목 캐시
            self.assertEqual(amk.classify("Anima-2.9B.safetensors", "/x", cached_only=True), amk.KIND_ANIMA)
        amk.remember_kinds({"sdxl.safetensors": amk.KIND_OTHER}, replace=True)
        self.assertEqual(amk.classify("sdxl.safetensors", cached_only=True), amk.KIND_OTHER)
        self.assertEqual(amk.classify("renamed.safetensors", cached_only=True), amk.KIND_UNKNOWN)   # replace

    def test_remember_ignores_unknown_and_copies_are_detached(self):
        amk.remember_kinds({"a": amk.KIND_V2, "b": amk.KIND_UNKNOWN, "": amk.KIND_V2, "c": "mystery"})
        kinds = amk.cached_kinds()
        self.assertEqual(kinds, {"a": amk.KIND_V2})
        kinds["z"] = amk.KIND_OTHER
        self.assertNotIn("z", amk.cached_kinds())

    def test_kind_for_infotext_model_name_matches_the_title_stem(self):
        amk.remember_kinds({"folder/sdxl_base.safetensors [0123abcd]": amk.KIND_OTHER,
                            "renamed-bundle.safetensors": amk.KIND_V2})
        self.assertEqual(amk.kind_for_model_name("sdxl_base"), amk.KIND_OTHER)
        self.assertEqual(amk.kind_for_model_name("renamed-bundle"), amk.KIND_V2)
        self.assertEqual(amk.kind_for_model_name("anima_baseV10"), amk.KIND_ANIMA)       # 이름
        self.assertEqual(amk.kind_for_model_name("mystery"), amk.KIND_UNKNOWN)

    def test_same_checkpoint_ignores_folder_hash_and_case(self):
        self.assertTrue(amk.same_checkpoint("sub\\Anima-3.8B-v1.1.safetensors [abcdef12]", "anima-3.8b-v1.1.safetensors"))
        self.assertFalse(amk.same_checkpoint("Anima-3.8B-v1.1.safetensors", "Anima-2.9B.safetensors"))
        self.assertFalse(amk.same_checkpoint("", ""))
        self.assertFalse(amk.same_checkpoint(None, "x.safetensors"))

    def test_same_checkpoint_accepts_forge_short_title_and_alias(self):
        """(P9 리뷰 1) Forge 'Show filenames without folder'(sd_checkpoint_dropdown_use_short) 를 켜면
        sd_model_checkpoint 는 short_title = 확장자 없는 이름 + 해시(modules/sd_models.py short_title)이고,
        name_for_extra 별칭(확장자 없는 이름)도 Forge 가 받는 값이다 — 같은 파일이다."""
        title = "sub/Anima-3.8B-v1.1.safetensors [5f3c1a2b]"          # 앱 콤보 = /sdapi/v1/sd-models title
        for forge_value in ("Anima-3.8B-v1.1 [5f3c1a2b]", "Anima-3.8B-v1.1", "anima-3.8b-v1.1.SAFETENSORS"):
            with self.subTest(forge_value=forge_value):
                self.assertTrue(amk.same_checkpoint(title, forge_value))
                self.assertTrue(amk.same_checkpoint(forge_value, title))
        self.assertFalse(amk.same_checkpoint(title, "Anima-3.8B-v1 [5f3c1a2b]"))       # 다른 이름
        self.assertFalse(amk.same_checkpoint(title, "Anima-2.9B"))
        self.assertFalse(amk.same_checkpoint(".safetensors", ".safetensors"))          # 이름이 없으면 모름

    def test_v1_module_pair_uses_the_compiler_markers(self):
        self.assertTrue(amk.has_v1_module_pair(["vae.safetensors", "text/qwen35_4b.safetensors",
                                                "text/Anima-3.8B-expanded_adapter.safetensors"]))
        self.assertFalse(amk.has_v1_module_pair(["text/qwen35_4b.safetensors"]))
        self.assertFalse(amk.has_v1_module_pair(["text/Anima-3.8B-expanded_adapter.safetensors"]))
        self.assertFalse(amk.has_v1_module_pair(["text/anima_qwen35_4b_connector.safetensors"]))   # Qwen 으로 분류
        self.assertFalse(amk.has_v1_module_pair([]))


class LockTests(_CacheReset):
    def test_concurrent_writer_and_readers_see_whole_maps(self):
        """(critic B16) 데몬 스레드가 쓰고 GUI 스레드가 읽는다 — 읽는 쪽은 늘 한 번의 replace 결과 전체를 본다."""
        maps = [{f"m{j}-{i}": amk.KIND_V2 for j in range(50)} for i in range(40)]
        seen, errors = [], []

        def writer():
            for mapping in maps:
                amk.remember_kinds(mapping, replace=True)

        def reader():
            try:
                for _ in range(200):
                    kinds = amk.cached_kinds()
                    suffixes = {name.rsplit("-", 1)[1] for name in kinds}
                    seen.append(len(suffixes) <= 1)
                    amk.classify("m0-0", cached_only=True)
            except Exception as exc:   # pragma: no cover - 실패 보고용
                errors.append(exc)

        threads = [threading.Thread(target=writer), *(threading.Thread(target=reader) for _ in range(3))]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        self.assertEqual(errors, [])
        self.assertTrue(all(seen))
        self.assertIsInstance(amk._lock, type(threading.Lock()))


if __name__ == "__main__":
    unittest.main()
