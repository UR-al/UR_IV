"""SAM3/ADetailer 배치 'EXIF 프롬프트 사용'이 core.image_metadata 한 곳에서 읽는지.

워커마다 있던 split('\\nNegative prompt: ') 파서는 네거티브 없는 A1111 이미지의 파라미터
줄을 프롬프트로 넣고, ComfyUI 기본 제목 그래프의 네거티브를 positive 에 합치고,
JPEG/WebP 는 빈 프롬프트로 조용히 돌렸다. 모델 실행 없이 작은 파일로만 검증한다.
"""
import json
import tempfile
import unittest
from pathlib import Path

from PIL import Image, PngImagePlugin

from core.image_metadata import (
    EXIF_WARNING_AMBIGUOUS,
    EXIF_WARNING_MISSING,
    EXIF_WARNING_NO_POSITIVE,
    read_applicable_prompts,
)
from tests.test_image_metadata import _insert_png_chunk_before_iend, graph_fixture


def _titled_graph():
    """ComfyUI 기본 저장 형태 — 두 인코더 모두 제목이 'CLIP Text Encode (Prompt)'."""
    graph = graph_fixture()
    graph.pop("99")
    for node_id in ("7", "8"):
        graph[node_id]["_meta"] = {"title": "CLIP Text Encode (Prompt)"}
    return graph


class ReadApplicablePromptsTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.dir = Path(temp.name)

    def png(self, name="image.png", **chunks):
        path = self.dir / name
        info = PngImagePlugin.PngInfo()
        for key, value in chunks.items():
            info.add_text(key, value if isinstance(value, str) else json.dumps(value))
        Image.new("RGB", (8, 8)).save(path, pnginfo=info)
        return path

    def test_negative_free_parameters_do_not_leak_the_steps_line(self):
        path = self.png(parameters="1girl, solo\nSteps: 20, Sampler: Euler a, CFG scale: 7, Seed: 1, Size: 512x512")
        self.assertEqual(read_applicable_prompts(path), ("1girl, solo", "", ""))

    def test_comfy_default_titles_follow_sampler_roles(self):
        path = self.png(prompt=_titled_graph())
        self.assertEqual(read_applicable_prompts(path), ("a blue bird", "bad anatomy", ""))

    def test_ambiguous_comfy_graph_fails_closed_with_warning(self):
        graph = graph_fixture()
        graph["7"] = {"class_type": "UnknownPromptEncoder", "inputs": {"text": "do not guess"}}
        path = self.png(prompt=graph)
        self.assertEqual(read_applicable_prompts(path), ("", "", EXIF_WARNING_AMBIGUOUS))

    def test_jpeg_and_webp_usercomment_are_read(self):
        raw = "jpeg positive\nNegative prompt: jpeg negative\nSteps: 30, Seed: 111"
        for suffix in (".jpg", ".webp"):
            path = self.dir / f"photo{suffix}"
            exif = Image.Exif()
            exif[0x8769] = {0x9286: b"UNICODE\x00" + raw.encode("utf-16-be")}
            Image.new("RGB", (8, 8)).save(path, exif=exif)
            with self.subTest(suffix=suffix):
                self.assertEqual(read_applicable_prompts(path), ("jpeg positive", "jpeg negative", ""))

    def test_parameters_written_after_idat_are_read(self):
        path = self.png()
        text = "trail pos\nNegative prompt: trail neg\nSteps: 20, Seed: 1"
        _insert_png_chunk_before_iend(path, b"tEXt", b"parameters\x00" + text.encode("latin-1"))
        self.assertEqual(read_applicable_prompts(path), ("trail pos", "trail neg", ""))

    def test_description_chunk_with_a1111_text_is_still_accepted(self):
        path = self.png(Description="desc pos\nNegative prompt: desc neg\nSteps: 20, Seed: 1")
        self.assertEqual(read_applicable_prompts(path), ("desc pos", "desc neg", ""))

    def test_missing_metadata_and_negative_only_are_reported(self):
        self.assertEqual(read_applicable_prompts(self.png("plain.png")), ("", "", EXIF_WARNING_MISSING))
        self.assertEqual(
            read_applicable_prompts(self.png("neg.png", parameters="Negative prompt: bad\nSteps: 20, Seed: 1")),
            ("", "bad", EXIF_WARNING_NO_POSITIVE))
        self.assertEqual(read_applicable_prompts(self.dir / "missing.png")[2] != "", True)


class WorkerPrepareSettingsTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.path = Path(temp.name) / "image.png"
        info = PngImagePlugin.PngInfo()
        info.add_text("parameters", "exif pos\nNegative prompt: exif neg\nSteps: 20, Seed: 1")
        Image.new("RGB", (8, 8)).save(self.path, pnginfo=info)

    def test_sam3_keeps_user_prompts_and_fills_from_exif(self):
        from workers.sam3_worker import _prepare_settings
        settings, warning = _prepare_settings({"use_exif_prompt": True, "sam3_inpaint_prompt": ""}, str(self.path))
        self.assertEqual((settings["sam3_inpaint_prompt"], settings["sam3_negative_prompt"], warning),
                         ("exif pos", "exif neg", ""))
        settings, _warning = _prepare_settings(
            {"use_exif_prompt": True, "sam3_inpaint_prompt": "mine", "sam3_negative_prompt": "my neg"}, str(self.path))
        self.assertEqual((settings["sam3_inpaint_prompt"], settings["sam3_negative_prompt"]), ("mine", "my neg"))

    def test_adetailer_uses_exif_and_reports_warning(self):
        from workers.adetailer_worker import _prepare_settings
        settings, warning = _prepare_settings({"use_exif_prompt": True}, str(self.path))
        self.assertEqual((settings["ad_prompt"], settings["ad_negative"], warning), ("exif pos", "exif neg", ""))
        plain = self.path.with_name("plain.png")
        Image.new("RGB", (8, 8)).save(plain)
        settings, warning = _prepare_settings({"use_exif_prompt": True}, str(plain))
        self.assertEqual((settings["ad_prompt"], settings["ad_negative"]), ("", ""))
        self.assertEqual(warning, EXIF_WARNING_MISSING)

    def test_exif_mode_off_never_reads_the_file(self):
        from workers.adetailer_worker import _prepare_settings as ad_prepare
        from workers.sam3_worker import _prepare_settings as sam3_prepare
        missing = str(self.path.with_name("missing.png"))
        self.assertEqual(ad_prepare({"ad_prompt": "x"}, missing), ({"ad_prompt": "x"}, ""))
        self.assertEqual(sam3_prepare({}, missing)[1], "")


if __name__ == "__main__":
    unittest.main()
