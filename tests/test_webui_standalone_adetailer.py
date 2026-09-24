"""단독·배치 ADetailer(Forge) — skip_img2img 로 부모 재확산 없이 (#24).

예전엔 확장 인자가 [True, False, slot] 이라 부모 img2img 가 denoise 0.1·빈 프롬프트·API 기본
steps 로 원본 전체를 한 번 더 재확산했다(얼굴이 없어도 드리프트, 그 재확산본이 저장됨).
"""
from __future__ import annotations

import base64
import io
import unittest
from unittest import mock

from PIL import Image

from backends.webui_backend import WebUIBackend


def _png(width: int, height: int) -> str:
    out = io.BytesIO()
    Image.new('RGB', (width, height), 'gray').save(out, format='PNG')
    return base64.b64encode(out.getvalue()).decode('ascii')


class _Response:
    def __init__(self, image_b64: str):
        self._image = image_b64

    def raise_for_status(self):
        return None

    def json(self):
        return {'images': [self._image]}


class StandaloneADetailerTests(unittest.TestCase):
    def _post(self, settings, size=(832, 1216)):
        backend = WebUIBackend('http://127.0.0.1:7860')
        image = _png(*size)
        with mock.patch('backends.webui_backend.requests.post', return_value=_Response('OUT')) as post:
            result = backend.adetailer(image, settings)
        self.assertEqual(result, 'OUT')
        url = post.call_args.args[0]
        self.assertTrue(url.endswith('/sdapi/v1/img2img'))
        return post.call_args.kwargs['json'], image

    def test_skip_img2img_is_on_and_parent_uses_the_original_size(self):
        payload, image = self._post({'ad_model': 'face_yolov8n.pt', 'ad_confidence': 0.3,
                                     'ad_denoise': 0.4, 'ad_prompt': ''})
        args = payload['alwayson_scripts']['ADetailer']['args']
        self.assertIs(args[0], True)
        self.assertIs(args[1], True)          # skip_img2img
        self.assertEqual(args[2]['ad_model'], 'face_yolov8n.pt')
        self.assertEqual(args[2]['ad_denoising_strength'], 0.4)
        # 확장이 _ad_orig 로 보관해 인페인트 패스에 쓰는 값 — 원본 크기와 명시 샘플링
        self.assertEqual((payload['width'], payload['height']), (832, 1216))
        self.assertEqual(payload['init_images'], [image])
        self.assertEqual((payload['steps'], payload['cfg_scale'], payload['seed']), (28, 7.0, -1))
        self.assertEqual(payload['resize_mode'], 0)
        self.assertFalse(payload['save_images'])
        self.assertNotIn('sampler_name', payload)
        self.assertNotIn('scheduler', payload)
        self.assertEqual(set(payload['alwayson_scripts']), {'ADetailer'})

    def test_explicit_sampling_settings_are_sent(self):
        payload, _ = self._post({'ad_model': 'hand_yolov8n.pt', 'steps': '32', 'cfg_scale': 5.5,
                                 'seed': 123, 'sampler': 'DPM++ 2M', 'scheduler': 'Karras'})
        self.assertEqual((payload['steps'], payload['cfg_scale'], payload['seed']), (32, 5.5, 123))
        self.assertEqual(payload['sampler_name'], 'DPM++ 2M')
        self.assertEqual(payload['scheduler'], 'Karras')

    def test_placeholder_sampler_and_bad_numbers_fall_back(self):
        payload, _ = self._post({'sampler': 'Use same sampler', 'scheduler': 'Use same scheduler',
                                 'steps': '', 'cfg_scale': 'x', 'seed': None})
        self.assertNotIn('sampler_name', payload)
        self.assertNotIn('scheduler', payload)
        self.assertEqual((payload['steps'], payload['cfg_scale'], payload['seed']), (28, 7.0, -1))

    def test_prompts_come_from_the_ad_settings(self):
        payload, _ = self._post({'ad_prompt': 'smile', 'ad_negative': 'blurry'})
        self.assertEqual((payload['prompt'], payload['negative_prompt']), ('smile', 'blurry'))
        slot = payload['alwayson_scripts']['ADetailer']['args'][2]
        self.assertEqual((slot['ad_prompt'], slot['ad_negative_prompt']), ('smile', 'blurry'))

    def test_unused_postprocess_base_payload_is_gone(self):
        payload, _ = self._post({'_postprocess_base_payload': {'enable_hr': True, 'hr_scale': 2,
                                                               'alwayson_scripts': {'X': {}}}})
        self.assertNotIn('enable_hr', payload)
        self.assertNotIn('X', payload['alwayson_scripts'])
        import inspect
        self.assertNotIn('_postprocess_base_payload',
                         inspect.getsource(WebUIBackend._build_postprocess_payload).split('"""')[-1])


if __name__ == '__main__':
    unittest.main()
