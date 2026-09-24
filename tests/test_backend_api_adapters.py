"""Focused contracts used by the bidirectional generation API adapters."""
from __future__ import annotations

import base64
import hashlib
import io
import json
import unittest
from unittest import mock

from PIL import Image

from backends.base import GenerationResult
from backends.comfyui_backend import ComfyUIBackend
from backends.webui_backend import WebUIBackend


class _JsonResponse:
    def __init__(self, payload):
        self._payload = payload

    def json(self):
        return self._payload

    def raise_for_status(self):
        return None


def _image_bytes(image_format: str, color: str) -> bytes:
    output = io.BytesIO()
    Image.new('RGB', (2, 2), color).save(output, format=image_format)
    return output.getvalue()


class TestWebUIResultArtifacts(unittest.TestCase):
    def test_generate_preserves_every_image_and_detects_actual_mime(self):
        png = _image_bytes('PNG', 'red')
        jpeg = _image_bytes('JPEG', 'blue')
        response = _JsonResponse({
            'images': [
                # The declared type is intentionally wrong: decoded content wins.
                'data:image/jpeg;base64,' + base64.b64encode(png).decode('ascii'),
                base64.b64encode(jpeg).decode('ascii'),
            ],
            'info': json.dumps({'seed': 42}),
        })
        backend = WebUIBackend('http://127.0.0.1:7860')

        with mock.patch.object(backend, '_switch_model_if_needed'), mock.patch(
            'backends.webui_backend.requests.post', return_value=response
        ):
            result = backend.txt2img('model.safetensors', {'prompt': 'test'})

        self.assertTrue(result.success)
        self.assertEqual(result.image_data, png)
        self.assertEqual([item.data for item in result.artifacts], [png, jpeg])
        self.assertEqual(
            [item.mime for item in result.artifacts],
            ['image/png', 'image/jpeg'],
        )
        self.assertEqual(
            [item.filename for item in result.artifacts],
            ['image_001.png', 'image_002.jpg'],
        )
        self.assertEqual(result.info['seed'], 42)
        self.assertEqual(result.info['artifact_count'], 2)

    def test_generate_rejects_non_image_data_uri_mime(self):
        response = _JsonResponse({
            'images': [
                'data:text/html;base64,'
                + base64.b64encode(b'<html></html>').decode('ascii')
            ],
            'info': '{}',
        })
        backend = WebUIBackend('http://127.0.0.1:7860')

        with mock.patch.object(backend, '_switch_model_if_needed'), mock.patch(
            'backends.webui_backend.requests.post', return_value=response
        ):
            result = backend.txt2img('', {})

        self.assertFalse(result.success)
        self.assertIn('MIME', result.error)

    def test_generate_rejects_active_svg_even_when_declared_as_image(self):
        svg = b'<svg xmlns="http://www.w3.org/2000/svg"><script>alert(1)</script></svg>'
        response = _JsonResponse({
            'images': ['data:image/svg+xml;base64,' + base64.b64encode(svg).decode('ascii')],
            'info': '{}',
        })
        backend = WebUIBackend('http://127.0.0.1:7860')
        with mock.patch.object(backend, '_switch_model_if_needed'), mock.patch(
            'backends.webui_backend.requests.post', return_value=response
        ):
            result = backend.txt2img('', {})
        self.assertFalse(result.success)

    def test_cancel_check_before_dispatch_skips_generation_post(self):
        backend = WebUIBackend('http://127.0.0.1:7860')
        cancel_check = mock.Mock(side_effect=[False, False, True])

        with mock.patch('backends.webui_backend.requests.post') as post:
            result = backend.txt2img(
                '',
                {'prompt': 'must not run'},
                cancel_check=cancel_check,
            )

        self.assertFalse(result.success)
        self.assertIn('취소', result.error)
        post.assert_not_called()

    def test_generate_enforces_artifact_count_and_total_size_limits(self):
        png = _image_bytes('PNG', 'red')
        encoded = base64.b64encode(png).decode('ascii')
        scenarios = (
            (
                'count',
                '_MAX_RESULT_ARTIFACTS',
                1,
                _JsonResponse({'images': [encoded, encoded], 'info': '{}'}),
                '최대 1개',
            ),
            (
                'bytes',
                '_MAX_RESULT_BYTES',
                len(png) - 1,
                _JsonResponse({'images': [encoded], 'info': '{}'}),
                '총 용량',
            ),
        )

        for name, constant, limit, response, expected_error in scenarios:
            with self.subTest(limit=name):
                backend = WebUIBackend('http://127.0.0.1:7860')
                with mock.patch.object(backend, '_switch_model_if_needed'), mock.patch(
                    'backends.webui_backend.requests.post', return_value=response
                ), mock.patch(f'backends.webui_backend.{constant}', limit):
                    result = backend.txt2img('', {})

                self.assertFalse(result.success)
                self.assertIn(expected_error, result.error)


def _api_workflow(*, include_load_image: bool = True) -> dict:
    workflow = {
        '1': {'class_type': 'CheckpointLoaderSimple', 'inputs': {'ckpt_name': 'old'}},
        '2': {'class_type': 'CLIPTextEncode', 'inputs': {'clip': ['1', 1], 'text': 'old positive'}},
        '3': {'class_type': 'CLIPTextEncode', 'inputs': {'clip': ['1', 1], 'text': 'old negative'}},
        '4': {
            'class_type': 'KSampler',
            'inputs': {
                'model': ['1', 0],
                'positive': ['2', 0],
                'negative': ['3', 0],
                'latent_image': ['5', 0],
                'seed': 1, 'steps': 20, 'cfg': 7, 'sampler_name': 'euler',
                'scheduler': 'normal', 'denoise': 1.0,
            },
        },
        '5': {
            'class_type': 'EmptyLatentImage',
            'inputs': {'width': 512, 'height': 512, 'batch_size': 1},
        },
        '8': {'class_type': 'VAEDecode', 'inputs': {'samples': ['4', 0], 'vae': ['1', 2]}},
        '7': {'class_type': 'SaveImage', 'inputs': {'images': ['8', 0]}},
    }
    if include_load_image:
        workflow['6'] = {'class_type': 'LoadImage', 'inputs': {'image': 'old.png'}}
    return workflow


def _profile_object_info() -> dict:
    """Stock ComfyUI schema of a Generation API profile target (no node pack)."""
    def choice(*values):
        return [list(values), {}]

    return {
        'CheckpointLoaderSimple': {'input': {'required': {
            'ckpt_name': choice('old', 'new-model.safetensors', 'sdxl/base.safetensors'),
        }}},
        'LoraLoader': {'input': {'required': {
            'lora_name': choice('styles/ink.safetensors'),
        }}},
        'KSampler': {'input': {'required': {
            'sampler_name': choice('euler', 'euler_ancestral', 'dpmpp_2m'),
            'scheduler': choice('normal', 'karras'),
        }}},
    }


class TestComfyProfileWorkflowGeneration(unittest.TestCase):
    def setUp(self):
        self.backend = ComfyUIBackend(
            'http://127.0.0.1:8188',
            workflow_path='profile-txt.json',
            img2img_workflow_path='profile-img.json',
        )
        # Never reach a real ComfyUI: the profile target's (bounded) schema read
        # is stubbed, and the per-endpoint snapshot is private to this test.
        from core.comfy_object_info_cache import ObjectInfoCacheRegistry

        self.backend.get_object_info_bounded = mock.Mock(return_value=_profile_object_info())
        self.backend._external_object_info_caches = ObjectInfoCacheRegistry()

    def test_instance_paths_override_process_config(self):
        self.assertEqual(
            self.backend._configured_workflow_path('txt2img'),
            'profile-txt.json',
        )
        self.assertEqual(
            self.backend._configured_workflow_path('img2img'),
            'profile-img.json',
        )

    def test_cancel_check_before_dispatch_skips_prompt_post(self):
        workflow = {'1': {'class_type': 'SaveImage', 'inputs': {}}}
        websocket = mock.Mock()
        cancel_check = mock.Mock(side_effect=[False, True])

        with mock.patch(
            'backends.comfyui_backend.websocket.create_connection',
            return_value=websocket,
        ), mock.patch('backends.comfyui_backend.requests.post') as post:
            result = self.backend.run_workflow(
                workflow,
                cancel_check=cancel_check,
            )

        self.assertFalse(result.success)
        self.assertIn('취소', result.error)
        post.assert_not_called()
        websocket.close.assert_called_once_with()

    def test_txt2img_deep_copies_and_maps_supplied_workflow(self):
        workflow = _api_workflow()
        original = json.loads(json.dumps(workflow))
        payload = {
            'prompt': 'new positive',
            'negative_prompt': 'new negative',
            'seed': 123,
            'steps': 18,
            'cfg_scale': 5.5,
            'width': 768,
            'height': 1024,
            'batch_size': 2,
            'n_iter': 2,
        }
        expected = GenerationResult(success=True, image_data=b'generated')
        callback = mock.Mock()

        with mock.patch.object(
            self.backend, 'run_workflow', return_value=expected
        ) as run:
            result = self.backend.generate_workflow(
                't2i', workflow, 'new-model.safetensors', payload, callback
            )

        self.assertIs(result, expected)
        self.assertEqual(workflow, original)
        prepared, passed_callback = run.call_args.args
        self.assertIsNot(prepared, workflow)
        self.assertIs(passed_callback, callback)
        self.assertEqual(prepared['1']['inputs']['ckpt_name'], 'new-model.safetensors')
        self.assertEqual(prepared['2']['inputs']['text'], 'new positive')
        self.assertEqual(prepared['3']['inputs']['text'], 'new negative')
        self.assertEqual(prepared['4']['inputs']['seed'], 123)
        self.assertEqual(prepared['5']['inputs']['width'], 768)
        self.assertEqual(prepared['5']['inputs']['height'], 1024)
        self.assertEqual(prepared['5']['inputs']['batch_size'], 4)

    def test_img2img_uploads_input_and_uses_default_denoise(self):
        workflow = _api_workflow()
        original = json.loads(json.dumps(workflow))
        expected = GenerationResult(success=True)

        with mock.patch.object(
            self.backend, '_upload_image', return_value='api/source.png'
        ) as upload, mock.patch.object(
            self.backend, 'run_workflow', return_value=expected
        ) as run:
            result = self.backend.generate_workflow(
                'i2i', workflow, '', {'init_images': ['aW1hZ2U=']}
            )

        self.assertIs(result, expected)
        self.assertEqual(workflow, original)
        upload.assert_called_once_with('aW1hZ2U=')
        prepared = run.call_args.args[0]
        self.assertEqual(prepared['6']['inputs']['image'], 'api/source.png')
        self.assertEqual(prepared['4']['inputs']['denoise'], 0.75)

    def test_upload_image_detects_actual_format_and_uses_content_hash_filename(self):
        expected_formats = (
            ('JPEG', 'jpg', 'image/jpeg'),
            ('WEBP', 'webp', 'image/webp'),
            ('BMP', 'bmp', 'image/bmp'),
            ('TIFF', 'tiff', 'image/tiff'),
        )
        encoded_images = [
            base64.b64encode(_image_bytes(image_format, 'green')).decode('ascii')
            for image_format, _extension, _mime in expected_formats
        ]

        with mock.patch.object(
            self.backend,
            'upload_media',
            side_effect=lambda _data, filename, _mime, **_kwargs: filename,
        ) as upload:
            uploaded_names = [
                self.backend._upload_image(encoded) for encoded in encoded_images
            ]

        def content_name(encoded, extension):
            digest = hashlib.sha256(base64.b64decode(encoded)).hexdigest()[:32]
            return f'input_{digest}.{extension}'

        self.assertEqual(
            uploaded_names,
            [
                content_name(encoded, extension)
                for encoded, (_format, extension, _mime) in zip(encoded_images, expected_formats)
            ],
        )
        self.assertEqual(len(set(uploaded_names)), len(uploaded_names))
        for call, encoded, (_format, extension, mime) in zip(
            upload.call_args_list, encoded_images, expected_formats
        ):
            data, filename, passed_mime = call.args
            self.assertEqual(data, base64.b64decode(encoded))
            self.assertEqual(filename, content_name(encoded, extension))
            self.assertEqual(passed_mime, mime)
            # 같은 이름·같은 내용이면 ComfyUI 가 다시 쓰지 않고 그 이름을 돌려준다.
            self.assertIs(call.kwargs['overwrite'], False)

    def test_repeated_upload_of_the_same_image_reuses_one_input_name(self):
        encoded = base64.b64encode(_image_bytes('PNG', 'green')).decode('ascii')
        other = base64.b64encode(_image_bytes('PNG', 'red')).decode('ascii')
        with mock.patch.object(
            self.backend,
            'upload_media',
            side_effect=lambda _data, filename, _mime, **_kwargs: filename,
        ):
            first = self.backend._upload_image(encoded)
            second = self.backend._upload_image('data:image/png;base64,' + encoded)
            different = self.backend._upload_image(other)
        self.assertEqual(first, second)
        # 원본과 마스크처럼 내용이 다르면 이름도 달라 서로 덮어쓰지 않는다.
        self.assertNotEqual(first, different)

    def test_upload_image_rejects_non_image_bytes_before_upload(self):
        encoded = base64.b64encode(b'not an image').decode('ascii')

        with mock.patch.object(self.backend, 'upload_media') as upload:
            with self.assertRaisesRegex(ValueError, '이미지'):
                self.backend._upload_image(encoded)

        upload.assert_not_called()

    def test_img2img_rejects_workflow_without_load_image_before_upload(self):
        workflow = _api_workflow(include_load_image=False)

        with mock.patch.object(self.backend, '_upload_image') as upload, mock.patch.object(
            self.backend, 'run_workflow'
        ) as run:
            result = self.backend.generate_workflow(
                'img2img', workflow, '', {'init_images': ['aW1hZ2U=']}
            )

        self.assertFalse(result.success)
        self.assertIn('LoadImage', result.error)
        upload.assert_not_called()
        run.assert_not_called()


if __name__ == '__main__':
    unittest.main()
