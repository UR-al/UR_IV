"""Chat generation's public planning and execution contracts; no GPU required."""
import unittest
import tempfile
import base64
from io import BytesIO
from pathlib import Path
from types import SimpleNamespace

from core.chat_generation import plan_chat_generation, MediaGenerationJob
from core.resource_coordinator import GenerationResourceCoordinator
from core.chat_store import ChatStore


class CurrentBackendTests(unittest.TestCase):
    def test_current_generation_honors_optional_llm_unload_inside_shared_lease(self):
        plan = plan_chat_generation({'generation': {'mode': 'image'}, 'messages': [{'role': 'user', 'content': 'cat'}]})
        for enabled in (False, True):
            calls = []
            class Backend:
                def txt2img(self, *args, **kwargs):
                    calls.append('generate')
                    return SimpleNamespace(success=True, image_data=b'result', artifacts=[], info={})
            coordinator = GenerationResourceCoordinator(unload_llm=lambda: calls.append('unload') or True)
            with self.subTest(enabled=enabled), tempfile.TemporaryDirectory() as directory:
                result = MediaGenerationJob('unload-policy', plan).run_current(
                    Backend(), 'model', {}, directory, coordinator=coordinator, unload_llm=enabled)
                self.assertTrue(result['ok'], result.get('error'))
                self.assertEqual(calls, ['unload', 'generate'] if enabled else ['generate'])
                self.assertEqual(coordinator.state.phase, 'idle')

    def test_repeated_stop_is_idempotent(self):
        plan = plan_chat_generation({'generation': {'mode': 'image'}, 'messages': [{'role': 'user', 'content': 'cat'}]})
        job = MediaGenerationJob('stop-twice', plan)
        self.assertTrue(job.cancel())
        self.assertFalse(job.cancel())

    def test_failure_and_busy_lease_never_publish_success(self):
        class Backend:
            def txt2img(self, *args, **kwargs):
                return SimpleNamespace(success=False, error='model missing')
        plan = plan_chat_generation({'generation': {'mode': 'image'}, 'messages': [{'role': 'user', 'content': 'cat'}]})
        coordinator = GenerationResourceCoordinator()
        with tempfile.TemporaryDirectory() as directory:
            failed = MediaGenerationJob('failed', plan).run_current(Backend(), 'model', {}, directory, coordinator=coordinator)
            self.assertFalse(failed['ok'])
            self.assertEqual(failed['error'], 'model missing')
            with coordinator.reserve('manual', unload_llm=False):
                busy = MediaGenerationJob('busy', plan).run_current(Backend(), 'model', {}, directory, coordinator=coordinator)
                self.assertFalse(busy['ok'])
                self.assertEqual(coordinator.state.owner, 'manual')
            self.assertFalse(list(Path(directory).iterdir()))

    def test_cancel_at_saving_boundary_discards_result(self):
        class Backend:
            def txt2img(self, *args, **kwargs):
                return SimpleNamespace(success=True, image_data=b'partial', artifacts=[], info={})
        plan = plan_chat_generation({'generation': {'mode': 'image'}, 'messages': [{'role': 'user', 'content': 'cat'}]})
        job = MediaGenerationJob('cancel-saving', plan, lambda e: job.cancel() if e['phase'] == 'saving' else None)
        with tempfile.TemporaryDirectory() as directory:
            result = job.run_current(Backend(), 'model', {}, directory, coordinator=GenerationResourceCoordinator())
            self.assertTrue(result['stopped'])
            self.assertEqual(result['artifacts'], [])
            self.assertFalse(list(Path(directory).iterdir()))

    def test_cancel_before_preparation_never_uses_backend(self):
        plan = plan_chat_generation({'generation': {'mode': 'image'}, 'messages': [{'role': 'user', 'content': 'cat'}]})
        job = MediaGenerationJob('cancel-first', plan)
        job.cancel()
        with tempfile.TemporaryDirectory() as directory:
            result = job.run_current(object(), 'model', {}, directory, coordinator=GenerationResourceCoordinator())
            self.assertTrue(result['stopped'])
            self.assertFalse(list(Path(directory).iterdir()))

    def test_media_and_retry_request_survive_chat_store_round_trip(self):
        with tempfile.TemporaryDirectory() as directory:
            store = ChatStore(Path(directory) / 'chat.json')
            store.save([{'id': 'chat', 'messages': [{'id': 'answer', 'role': 'assistant', 'content': '완료',
                'artifacts': [{'kind': 'video', 'path': 'C:/output/movie.mp4', 'mime': 'video/mp4', 'filename': 'movie.mp4'}],
                'generationRequest': {'mode': 'video', 'family': 'current', 'duration': 3, 'denoise': .4, 'hadImage': True},
                'generation': {'kind': 'video', 'phase': 'complete', 'progress': 100}}]}])
            message = store.load()[0]['messages'][0]
            self.assertEqual(message['artifacts'][0]['path'], 'C:/output/movie.mp4')
            self.assertTrue(message['generationRequest']['hadImage'])
            self.assertEqual(message['generation']['phase'], 'complete')

    def test_creator_preparation_correlates_request_and_owns_only_its_temp_input(self):
        from PIL import Image
        image = BytesIO()
        Image.new('RGB', (8, 8)).save(image, 'PNG')
        plan = plan_chat_generation({'generation': {'mode': 'video', 'duration': 3}, 'messages': [
            {'role': 'user', 'content': 'Waves', 'images': ['data:image/png;base64,' + base64.b64encode(image.getvalue()).decode()]}]})
        job = MediaGenerationJob('video-request', plan)
        prepared = job.prepare_creator({})
        self.assertEqual(prepared['requestId'], 'video-request')
        self.assertEqual(prepared['mode'], 'h3_i2v')
        self.assertEqual(prepared['duration'], 3)
        self.assertEqual(prepared['quality'], 'quality')
        self.assertEqual(prepared['steps'], 20)
        path = Path(prepared['sourcePath'])
        self.assertTrue(path.is_file())
        job.close()
        self.assertFalse(path.exists())

    def test_reference_image_uses_img2img_and_keeps_selected_settings(self):
        from PIL import Image
        image = BytesIO()
        Image.new('RGB', (8, 8), 'red').save(image, 'PNG')
        encoded = 'data:image/png;base64,' + base64.b64encode(image.getvalue()).decode()
        class Backend:
            def img2img(self, model, payload, progress_callback=None):
                self.received = (model, payload)
                return SimpleNamespace(success=True, image_data=image.getvalue(), artifacts=[], info={})
        backend = Backend()
        plan = plan_chat_generation({'generation': {'mode': 'image', 'denoise': .4},
                                     'messages': [{'role': 'user', 'content': 'cat', 'images': [encoded]}]})
        with tempfile.TemporaryDirectory() as directory:
            result = MediaGenerationJob('i2i', plan).run_current(
                backend, 'anima', {'prompt': 'cat', 'steps': 12, 'width': 512, 'enable_hr': True},
                directory, coordinator=GenerationResourceCoordinator())
            self.assertTrue(result['ok'], result.get('error'))
            self.assertEqual(backend.received[1]['init_images'], [encoded.split(',', 1)[1]])
            self.assertEqual(backend.received[1]['denoising_strength'], .4)
            self.assertEqual(backend.received[1]['width'], 512)
            self.assertNotIn('enable_hr', backend.received[1])

    def test_current_model_generation_returns_a_persisted_image(self):
        class Backend:
            def txt2img(self, model, payload, progress_callback=None):
                self.received = (model, payload)
                progress_callback(2, 4, None)
                return SimpleNamespace(success=True, image_data=b'generated-image', artifacts=[], info={})
        backend, events = Backend(), []
        plan = plan_chat_generation({'generation': {'mode': 'image'}, 'messages': [{'role': 'user', 'content': 'cat'}]})
        with tempfile.TemporaryDirectory() as directory:
            result = MediaGenerationJob('request-1', plan, events.append).run_current(
                backend, 'anima-3.8.safetensors', {'prompt': 'cat', 'steps': 20},
                directory, coordinator=GenerationResourceCoordinator())
            self.assertTrue(result['ok'])
            self.assertEqual(Path(result['artifacts'][0]['path']).read_bytes(), b'generated-image')
            self.assertEqual(backend.received[0], 'anima-3.8.safetensors')
            self.assertEqual(backend.received[1]['steps'], 20)
            self.assertTrue(all(event['id'] == 'request-1' for event in events))
            self.assertTrue(any(event.get('progress') == 50 for event in events))

    def test_backend_receives_live_cancel_check_and_cancel_is_reported_as_stopped(self):
        # 감사 #31: 채팅 생성도 cancel_check 를 넘겨 모델 전환 중·발송 직후의 중지를 백엔드가 확인
        plan = plan_chat_generation({'generation': {'mode': 'image'}, 'messages': [{'role': 'user', 'content': 'cat'}]})
        job = MediaGenerationJob('cancel-check', plan)

        class Backend:
            def txt2img(self, model, payload, progress_callback=None, cancel_check=None):
                self.cancel_check = cancel_check
                self.before = cancel_check()
                job.cancelled.set()          # 체크포인트 전환 중에 사용자가 중지
                self.after = cancel_check()
                return SimpleNamespace(success=False, image_data=None, artifacts=[], info={},
                                       error='사용자가 작업을 취소했습니다')

        backend = Backend()
        with tempfile.TemporaryDirectory() as directory:
            result = job.run_current(backend, 'model', {'prompt': 'cat'}, directory,
                                     coordinator=GenerationResourceCoordinator())
        self.assertEqual((backend.before, backend.after), (False, True))
        self.assertTrue(result['stopped'])
        self.assertFalse(result['ok'])


class GenerationIntentTests(unittest.TestCase):
    def test_explicit_media_mode_uses_scene_text_even_when_it_mentions_prompts(self):
        for mode, text in [('image', '프롬프트가 적힌 종이 사진'), ('video', 'A camera pans across a sign explaining a prompt')]:
            plan = plan_chat_generation({'generation': {'mode': mode}, 'messages': [{'role': 'user', 'content': text}]})
            self.assertIsNotNone(plan)
            self.assertEqual(plan.prompt, text)

    def test_retry_with_dropped_reference_requires_reattachment(self):
        with self.assertRaisesRegex(ValueError, '다시 첨부'):
            plan_chat_generation({'generation': {'mode': 'image', 'hadImage': True},
                                  'messages': [{'role': 'user', 'content': 'cat'}]})

    def test_automatic_routing_distinguishes_actions_from_discussion(self):
        examples = {
            '고양이 이미지 만들어줘': 'image', '눈밭에 있는 고양이를 그려 줘': 'image',
            '해변의 짧은 영상 생성해줘': 'video', 'Please generate an image of a cat': 'image',
            'Create a video of ocean waves': 'video', 'Draw a sleeping cat': 'image',
            '이미지를 생성하는 방법을 알려줘': None, '이미지 생성하지 마': None,
            '고양이 이미지용 프롬프트를 작성해줘': None, 'Do not generate an image': None,
            'How do I generate a video?': None, 'Write a prompt to generate an image': None,
            'Explain the image generation settings': None,
            '다음 코드를 설명해줘\n```\n이미지 만들어줘\n```': None,
        }
        for text, expected in examples.items():
            with self.subTest(text=text):
                plan = plan_chat_generation({'messages': [{'role': 'user', 'content': text}]})
                self.assertEqual(plan.kind if plan else None, expected)

    def test_explicit_image_request_does_not_require_a_chat_model(self):
        plan = plan_chat_generation({
            'generation': {'mode': 'image', 'family': 'krea2'},
            'messages': [{'role': 'user', 'content': 'A small cat in the snow'}],
        })
        self.assertEqual((plan.kind, plan.family, plan.prompt),
                         ('image', 'krea2', 'A small cat in the snow'))


if __name__ == '__main__':
    unittest.main()
