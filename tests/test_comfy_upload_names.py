"""ComfyUI input 업로드 이름 — 내용 해시라 같은 원본을 반복해도 input 폴더에 쌓이지 않는다(감사 #149)."""
import base64
import hashlib
import json
import tempfile
import unittest
from io import BytesIO
from pathlib import Path

from PIL import Image

from backends.base import GenerationResult
from core.comfy_upload_names import content_upload_name
from core.krea2_generation import run_krea2_generation
from tests.test_krea2_generation import _FakeComfy, _png_b64


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()[:32]


class ContentUploadNameTests(unittest.TestCase):
    def test_same_bytes_same_name_and_different_bytes_different_name(self):
        first = content_upload_name(b"source-bytes", "input", "png")
        self.assertEqual(first, f"input_{_sha(b'source-bytes')}.png")
        self.assertEqual(content_upload_name(bytearray(b"source-bytes"), "input", "png"), first)
        self.assertEqual(content_upload_name(memoryview(b"source-bytes"), "input", "png"), first)
        self.assertNotEqual(content_upload_name(b"mask-bytes", "input", "png"), first)
        # 접두어가 다르면(원본/참조) 같은 바이트라도 서로의 이름을 쓰지 않는다.
        self.assertNotEqual(content_upload_name(b"source-bytes", "krea2_reference", "png"), first)

    def test_extension_is_normalised_and_unsafe_parts_are_rejected(self):
        self.assertTrue(content_upload_name(b"x", "input", ".PNG").endswith(".png"))
        for prefix in ("", "../input", "a/b", "a\\b", "in put"):
            with self.subTest(prefix=prefix):
                with self.assertRaises(ValueError):
                    content_upload_name(b"x", prefix, "png")
        for extension in ("", "p/ng", "png.exe/..", "p ng"):
            with self.subTest(extension=extension):
                with self.assertRaises(ValueError):
                    content_upload_name(b"x", "input", extension)
        with self.assertRaises(TypeError):
            content_upload_name("text", "input", "png")


class UploadContentHelperTests(unittest.TestCase):
    """Krea2 러너·Creator 가 같이 쓰는 업로드 헬퍼 — overwrite/cancel_check 는 받는 어댑터에만."""

    def test_adapter_with_overwrite_and_cancel_gets_both(self):
        from core.comfy_upload_names import upload_content

        calls = []

        class Adapter:
            def upload_media(self, data, filename, mime, overwrite=True, cancel_check=None):
                calls.append((bytes(data), filename, mime, overwrite, cancel_check))
                return f"in/{filename}"

        check = lambda: False  # noqa: E731
        name = upload_content(Adapter(), b"abc", "creator_source", "PNG", "image/png", cancel_check=check)
        expected = f"creator_source_{_sha(b'abc')}.png"
        self.assertEqual(name, f"in/{expected}")
        self.assertEqual(calls, [(b"abc", expected, "image/png", False, check)])

    def test_duck_typed_adapter_is_called_with_three_arguments(self):
        from core.comfy_upload_names import upload_content

        calls = []
        adapter = type("Adapter", (), {"upload_media": lambda self, *args: calls.append(args) or "ok"})()
        self.assertEqual(upload_content(adapter, b"abc", "comic_panel", "png", "image/png",
                                        cancel_check=lambda: False), "ok")
        self.assertEqual(calls, [(b"abc", f"comic_panel_{_sha(b'abc')}.png", "image/png")])

    def test_unsafe_extension_fails_before_uploading(self):
        from core.comfy_upload_names import upload_content

        calls = []
        adapter = type("Adapter", (), {"upload_media": lambda self, *args: calls.append(args)})()
        with self.assertRaises(ValueError):
            upload_content(adapter, b"abc", "creator_source", "", "image/png")
        self.assertEqual(calls, [])


class _OverwriteAwareComfy(_FakeComfy):
    """ComfyUIBackend.upload_media 처럼 overwrite 를 받는 어댑터."""

    def upload_media(self, data, filename, mime, overwrite=True):
        self.uploads.append((bytes(data), filename, mime, overwrite))
        return f"uploads/{filename}"


def _i2i_payload(source_color, reference_color=None):
    payload = {
        "prompt": "change the coat to blue",
        "init_images": [_png_b64(source_color)],
        "width": 1000, "height": 760, "steps": 15, "cfg_scale": 1, "seed": 7,
    }
    if reference_color is not None:
        payload["krea2_reference_image"] = "data:image/png;base64," + _png_b64(reference_color)
    return payload


class Krea2UploadNameTests(unittest.TestCase):
    def test_repeated_i2i_reuses_the_same_input_names(self):
        backend = _OverwriteAwareComfy()
        for _ in range(3):
            result = run_krea2_generation(backend, "i2i", _i2i_payload((255, 0, 0), (0, 0, 255)))
            self.assertTrue(result.success, result.error)
        names = [upload[1] for upload in backend.uploads]
        self.assertEqual(len(names), 6)
        # 원본 3회는 한 이름, 참조 3회도 한 이름 — input 폴더에 두 파일만 남는다.
        self.assertEqual(len(set(names)), 2)
        source = base64.b64decode(_png_b64((255, 0, 0)))
        reference = base64.b64decode(_png_b64((0, 0, 255)))
        self.assertEqual(names[0], f"krea2_source_{_sha(source)}.png")
        self.assertEqual(names[1], f"krea2_reference_{_sha(reference)}.png")
        # 같은 내용이면 ComfyUI 가 다시 쓰지 않도록 overwrite=False 로 보낸다.
        self.assertTrue(all(upload[3] is False for upload in backend.uploads))

    def test_different_sources_get_different_names(self):
        backend = _OverwriteAwareComfy()
        run_krea2_generation(backend, "i2i", _i2i_payload((255, 0, 0)))
        run_krea2_generation(backend, "i2i", _i2i_payload((0, 255, 0)))
        self.assertNotEqual(backend.uploads[0][1], backend.uploads[1][1])

    def test_duck_typed_adapter_without_overwrite_still_uploads(self):
        backend = _FakeComfy()  # upload_media(data, filename, mime) 만 받는다
        result = run_krea2_generation(backend, "i2i", _i2i_payload((255, 0, 0)))
        self.assertTrue(result.success, result.error)
        source = base64.b64decode(_png_b64((255, 0, 0)))
        self.assertEqual(backend.uploads[0][1], f"krea2_source_{_sha(source)}.png")


class GenerationApiNamedComfyUploadTests(unittest.TestCase):
    """Named ComfyUI profiles always go through ComfyUIBackend.generate_workflow,
    whose img2img upload uses the content-hash name (감사 #149); the old
    adapter fallback that uploaded by itself was unreachable and is gone (#138)."""

    def _encoded(self):
        buffer = BytesIO()
        Image.new("RGB", (2, 2), (9, 9, 9)).save(buffer, format="PNG")
        return buffer.getvalue(), base64.b64encode(buffer.getvalue()).decode("ascii")

    def _run(self, backend, encoded):
        from core.generation_api import GenerationApiManager

        manager = GenerationApiManager.__new__(GenerationApiManager)
        with tempfile.TemporaryDirectory() as directory:
            workflow_path = Path(directory) / "i2i.json"
            workflow_path.write_text(json.dumps({
                "1": {"class_type": "LoadImage", "inputs": {"image": "old.png"}},
                "2": {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": "model"}},
                "3": {"class_type": "CLIPTextEncode", "inputs": {"clip": ["2", 1], "text": ""}},
                "4": {"class_type": "CLIPTextEncode", "inputs": {"clip": ["2", 1], "text": ""}},
                "5": {"class_type": "VAEEncode", "inputs": {"pixels": ["1", 0], "vae": ["2", 2]}},
                "6": {"class_type": "KSampler", "inputs": {
                    "model": ["2", 0], "positive": ["3", 0], "negative": ["4", 0],
                    "latent_image": ["5", 0], "seed": 1, "steps": 20, "cfg": 7,
                    "sampler_name": "euler", "scheduler": "normal", "denoise": 1.0,
                }},
            }), encoding="utf-8")
            profile = {"img2imgWorkflowPath": str(workflow_path)}
            return manager._run_named_comfy(
                backend, profile, "img2img", "model", {"init_images": [encoded]},
                lambda *args: None, lambda: False,
            )

    def test_repeated_api_img2img_reuses_one_content_named_input(self):
        from unittest import mock
        from backends.comfyui_backend import ComfyUIBackend

        raw, encoded = self._encoded()
        from core.comfy_object_info_cache import ObjectInfoCacheRegistry

        backend = ComfyUIBackend("http://127.0.0.1:1")
        backend.get_object_info_bounded = mock.Mock(return_value={
            "CheckpointLoaderSimple": {"input": {"required": {"ckpt_name": [["model"], {}]}}},
        })
        backend._external_object_info_caches = ObjectInfoCacheRegistry()
        uploads = []
        backend.upload_media = lambda data, filename, mime, **kwargs: uploads.append(
            (filename, kwargs.get("overwrite"))) or filename
        queued = []
        backend.run_workflow = lambda workflow, *args, **kwargs: queued.append(workflow) or GenerationResult(
            success=True, image_data=b"ok")
        for _ in range(2):
            result = self._run(backend, encoded)
            self.assertTrue(result.success, result.error)
        expected = f"input_{_sha(raw)}.png"
        self.assertEqual(uploads, [(expected, False), (expected, False)])
        self.assertEqual(queued[0]["1"]["inputs"]["image"], expected)

    def test_adapter_without_generate_workflow_is_an_explicit_error(self):
        class NoWorkflowAdapter:
            def __init__(self):
                self.uploads = []

            def upload_media(self, data, filename, mime):
                self.uploads.append(filename)
                return filename

        backend = NoWorkflowAdapter()
        result = self._run(backend, self._encoded()[1])
        self.assertFalse(result.success)
        self.assertIn("generate_workflow", result.error)
        self.assertEqual(backend.uploads, [])


if __name__ == "__main__":
    unittest.main()
