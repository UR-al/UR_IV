"""Task 12 - SAM3.1 연결: 파일 이름이 fp16 이어도 요청한 precision 이 유지된다.

torch 없이 가짜 로더·가짜 torch 네임스페이스만 쓴다(모델 파일은 열지 않는다).
"""
from __future__ import annotations

import json
import struct
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from comfy_custom_nodes.ai_studio_forge_parity import sam3_files, sam3_nodes

SAM31_NAME = "sam3.1_multiplex_fp16.safetensors"
_FAKE_TORCH = SimpleNamespace(
    bfloat16="BF16", float16="FP16", float32="FP32",
    cuda=SimpleNamespace(is_available=lambda: False),
)


class TransformersLayoutTests(unittest.TestCase):
    """Easy SAM3 loads with strict=False: such a file would run on random weights and find nothing, without an error."""

    def _file(self, directory, *keys):
        path = Path(directory) / "sam3.safetensors"
        header = json.dumps({key: {"dtype": "F32", "shape": [1], "data_offsets": [0, 4]} for key in keys}).encode("utf-8")
        path.write_bytes(struct.pack("<Q", len(header)) + header + b"\0\0\0\0")
        return path

    def test_the_transformers_layout_stops_before_loading(self):
        with tempfile.TemporaryDirectory() as directory,                 mock.patch.object(sam3_nodes, "_invoke_sam3_loader") as loader, mock.patch.object(sam3_nodes, "_torch", return_value=_FAKE_TORCH):
            path = self._file(directory, "detector_model.vision_encoder.a")
            with self.assertRaisesRegex(ValueError, "Hugging Face"):
                sam3_nodes._acquire_sam3(str(path), "cpu", "fp32", use_cache=False)
            loader.assert_not_called()

    def test_the_native_layout_loads(self):
        with tempfile.TemporaryDirectory() as directory,                 mock.patch.object(sam3_nodes, "_invoke_sam3_loader", return_value={"model": None}) as loader,                 mock.patch.object(sam3_nodes, "_torch", return_value=_FAKE_TORCH):
            path = self._file(directory, "detector.backbone.a")
            sam3_nodes._acquire_sam3(str(path), "cpu", "fp32", use_cache=False)
            loader.assert_called_once()

    def test_a_pt_file_or_an_unreadable_header_is_left_to_the_loader(self):
        with tempfile.TemporaryDirectory() as directory:
            self.assertFalse(sam3_files.is_transformers_layout(Path(directory) / "sam3.pt"))
            broken = Path(directory) / "broken.safetensors"
            broken.write_bytes(b"abc")
            self.assertFalse(sam3_files.is_transformers_layout(broken))


class Sam31LoaderTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.path = Path(self.tmp.name) / SAM31_NAME
        self.path.write_bytes(b"x")
        self.calls = []
        test = self

        class Loader:
            @classmethod
            def execute(cls, model, segmentor, device, precision):
                test.calls.append({"model": model, "precision": precision})
                # easy-sam3 처럼: 이름에 fp16 이 있으면 요청과 상관없이 fp16
                effective = "fp16" if "fp16" in model.lower() else precision
                dtype = {"bf16": "BF16", "fp16": "FP16", "fp32": "FP32"}[effective]
                return ({"model": object(), "device": device, "segmentor": segmentor, "dtype": dtype},)

        folder_paths = SimpleNamespace(folder_names_and_paths={"sam3": ([], {".pt"})})
        patches = [
            mock.patch.dict(sys.modules, {"folder_paths": folder_paths}),
            mock.patch.object(sam3_nodes, "_resolve_easy_node", return_value=Loader),
            mock.patch.object(sam3_nodes, "_torch", return_value=_FAKE_TORCH),
        ]
        for patch in patches:
            patch.start()
            self.addCleanup(patch.stop)

    def test_precision_restored_after_loader(self):
        bundle = sam3_nodes._invoke_sam3_loader(self.path, SAM31_NAME, "cuda", "bf16")
        self.assertEqual(bundle["dtype"], "BF16")

    def test_requested_fp16_and_fp32_are_kept_too(self):
        for precision, expected in (("fp16", "FP16"), ("fp32", "FP32")):
            with self.subTest(precision=precision):
                bundle = sam3_nodes._invoke_sam3_loader(self.path, SAM31_NAME, "cuda", precision)
                self.assertEqual(bundle["dtype"], expected)

    def test_sam31_path_accepted(self):
        device, precision, path, name = sam3_nodes._resolve_sam3_request(str(self.path), "cuda", "bf16")
        self.assertEqual(path, self.path)
        self.assertEqual(name, SAM31_NAME)
        sam3_nodes._invoke_sam3_loader(path, name, device, precision)
        self.assertEqual(self.calls, [{"model": SAM31_NAME, "precision": "bf16"}])


if __name__ == "__main__":
    unittest.main()
