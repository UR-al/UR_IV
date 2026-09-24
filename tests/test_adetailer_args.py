"""core/adetailer_args — ADetailer REST 슬롯 단일 소스 골든 테스트.

예전 두 사본(ui/generator_generation._build_adetailer_slot, workers/upscale_worker
._build_adetailer_slot)이 만들던 페이로드와 **JSON 직렬화 바이트까지** 같아야 한다
(키 순서 포함 — Forge 쪽 로그/재현 비교가 깨지지 않게).
"""
import json
import unittest

from core import adetailer_args
from core.adetailer_args import (
    DEFAULT_SLOT,
    build_simple_slot,
    build_slot,
    slot_from_settings,
)


def _legacy_literal(**v):
    """리팩터 전 리터럴(키 순서 그대로). v 로 바뀌던 자리만 채운다."""
    return {
        "ad_model": v.get('model', 'face_yolov8n.pt'),
        "ad_model_classes": "",
        "ad_tab_enable": True,
        "ad_prompt": v.get('prompt', ''),
        "ad_negative_prompt": v.get('neg', ''),
        "ad_confidence": v.get('confidence', 0.3),
        "ad_mask_filter_method": "Area",
        "ad_mask_k": 0,
        "ad_mask_min_ratio": 0.0,
        "ad_mask_max_ratio": 1.0,
        "ad_dilate_erode": v.get('dilate', 4),
        "ad_x_offset": 0,
        "ad_y_offset": 0,
        "ad_mask_merge_invert": v.get('merge', "None"),
        "ad_mask_blur": v.get('blur', 4),
        "ad_denoising_strength": v.get('denoise', 0.4),
        "ad_inpaint_only_masked": True,
        "ad_inpaint_only_masked_padding": v.get('padding', 32),
        "ad_use_inpaint_width_height": v.get('use_size', False),
        "ad_inpaint_width": v.get('iw', 512),
        "ad_inpaint_height": v.get('ih', 512),
        "ad_use_steps": v.get('use_steps', False),
        "ad_steps": v.get('steps', 28),
        "ad_use_cfg_scale": v.get('use_cfg', False),
        "ad_cfg_scale": v.get('cfg', 7.0),
        "ad_use_checkpoint": v.get('use_ckpt', False),
        "ad_checkpoint": v.get('ckpt', None),
        "ad_use_vae": v.get('use_vae', False),
        "ad_vae": v.get('vae', None),
        "ad_use_sampler": v.get('use_sampler', False),
        "ad_sampler": v.get('sampler', "DPM++ 2M Karras"),
        "ad_scheduler": v.get('scheduler', "Use same scheduler"),
        "ad_use_noise_multiplier": False,
        "ad_noise_multiplier": 1.0,
        "ad_use_clip_skip": False,
        "ad_clip_skip": 1,
        "ad_restore_face": False,
        "ad_controlnet_model": "None",
        "ad_controlnet_module": "None",
        "ad_controlnet_weight": 1.0,
        "ad_controlnet_guidance_start": 0.0,
        "ad_controlnet_guidance_end": 1.0,
    }


def _dumps(d):
    return json.dumps(d, ensure_ascii=False)


class _W:
    """proxy/LineEdit/ComboBox/PlainTextEdit/CheckBox 흉내."""

    def __init__(self, text='', checked=False):
        self._text = text
        self._checked = checked

    def text(self):
        return self._text

    def toPlainText(self):
        return self._text

    def currentText(self):
        return self._text

    def isChecked(self):
        return self._checked


def _widgets(**over):
    base = {
        'model': _W('face_yolov8n.pt'), 'confidence': _W('0.3'), 'denoise': _W('0.4'),
        'mask_blur': _W('4'), 'padding': _W('32'), 'prompt': _W(''), 'neg_prompt': _W(''),
        'use_inpaint_size_check': _W(checked=False), 'inpaint_width': _W('512'),
        'inpaint_height': _W('512'), 'use_steps_check': _W(checked=False), 'steps': _W('28'),
        'use_cfg_check': _W(checked=False), 'cfg': _W('7.0'),
        'use_checkpoint_check': _W(checked=False), 'checkpoint_combo': _W(''),
        'use_vae_check': _W(checked=False), 'vae_combo': _W(''),
        'use_sampler_check': _W(checked=False), 'sampler_combo': _W(''),
        'scheduler_combo': _W(''),
    }
    base.update(over)
    return base


class DefaultSlotTests(unittest.TestCase):
    def test_default_slot_matches_legacy_literal_bytes(self):
        self.assertEqual(len(DEFAULT_SLOT), 42)
        self.assertEqual(_dumps(build_slot()), _dumps(_legacy_literal()))

    def test_default_slot_is_read_only_and_build_returns_fresh_copies(self):
        with self.assertRaises(TypeError):
            DEFAULT_SLOT['ad_model'] = 'x'  # type: ignore[index]
        a, b = build_slot(), build_slot()
        a['ad_prompt'] = 'changed'
        self.assertEqual(b['ad_prompt'], '')
        self.assertEqual(DEFAULT_SLOT['ad_prompt'], '')

    def test_unknown_override_is_rejected(self):
        with self.assertRaises(KeyError):
            build_slot(ad_modle='typo.pt')

    def test_unified_fallback_defaults(self):
        # 's'/0.25(webui 폴백)와 'n'/0.4(t2i)로 갈라졌던 기본값을 'n'/0.4 로 통일
        self.assertEqual(adetailer_args.DEFAULT_MODEL, 'face_yolov8n.pt')
        self.assertEqual(adetailer_args.DEFAULT_DENOISE, 0.4)
        self.assertEqual(adetailer_args.DEFAULT_CONFIDENCE, 0.3)


class SimpleSlotTests(unittest.TestCase):
    def test_simple_slot_matches_legacy_upscale_worker_output(self):
        got = build_simple_slot('hand_yolov8n.pt', confidence=0.5, denoise=0.35, prompt='detailed hands')
        want = _legacy_literal(model='hand_yolov8n.pt', confidence=0.5, denoise=0.35,
                               prompt='detailed hands')
        self.assertEqual(_dumps(got), _dumps(want))

    def test_upscale_worker_keeps_no_dead_slot_builders(self):
        """옛 사본(_build_adetailer_slot)·빈 슬롯(_build_empty_adetailer_slot)은 운영 호출자가 없다 —
        WebUIBackend 폴백은 core.adetailer_args.slot_from_settings 를 직접 부른다. 별칭도 남기지 않는다."""
        from workers import upscale_worker
        self.assertFalse(hasattr(upscale_worker, '_build_adetailer_slot'))
        self.assertFalse(hasattr(upscale_worker, '_build_empty_adetailer_slot'))

    def test_slot_from_settings_uses_explicit_values_and_negative(self):
        slot = slot_from_settings({
            'ad_model': 'face_yolov8s.pt', 'ad_confidence': 0.45, 'ad_denoise': 0.3,
            'ad_prompt': 'smile', 'ad_negative': 'blurry',
        })
        want = _legacy_literal(model='face_yolov8s.pt', confidence=0.45, denoise=0.3,
                               prompt='smile', neg='blurry')
        self.assertEqual(_dumps(slot), _dumps(want))

    def test_slot_from_settings_fills_missing_or_blank_values_with_defaults(self):
        self.assertEqual(_dumps(slot_from_settings({})), _dumps(_legacy_literal()))
        blank = slot_from_settings({'ad_model': '', 'ad_confidence': None, 'ad_prompt': None})
        self.assertEqual(_dumps(blank), _dumps(_legacy_literal()))
        self.assertEqual(_dumps(slot_from_settings(None)), _dumps(_legacy_literal()))


class WebUIBackendFallbackTests(unittest.TestCase):
    def test_adetailer_fallback_builds_slot_from_core(self):
        from backends.webui_backend import WebUIBackend
        backend = WebUIBackend.__new__(WebUIBackend)
        captured = {}
        backend._build_postprocess_payload = lambda image, settings, **kw: {'alwayson_scripts': {}}

        def _run(image, payload):
            captured.update(payload)
            return 'out'

        backend._run_img2img_postprocess = _run
        result = backend.adetailer('img', {'ad_model': 'face_yolov8n.pt', 'ad_confidence': 0.3,
                                           'ad_denoise': 0.4, 'ad_prompt': '', 'ad_negative': 'bad'})
        self.assertEqual(result, 'out')
        args = captured['alwayson_scripts']['ADetailer']['args']
        # [enable, skip_img2img] — 단독 ADetailer 는 부모 img2img 재확산을 건너뛴다
        # (tests/test_webui_standalone_adetailer.py 참고).
        self.assertEqual(args[:2], [True, True])
        self.assertEqual(_dumps(args[2]), _dumps(_legacy_literal(neg='bad')))

    def test_adetailer_passes_prebuilt_args_through(self):
        from backends.webui_backend import WebUIBackend
        backend = WebUIBackend.__new__(WebUIBackend)
        captured = {}
        backend._build_postprocess_payload = lambda image, settings, **kw: {'alwayson_scripts': {}}
        backend._run_img2img_postprocess = lambda image, payload: captured.update(payload) or 'ok'
        prebuilt = [True, False, {'ad_model': 'x'}]
        backend.adetailer('img', {'adetailer_args': prebuilt})
        self.assertIs(captured['alwayson_scripts']['ADetailer']['args'], prebuilt)


class T2ISlotTests(unittest.TestCase):
    """GenerationMixin._build_adetailer_slot — 위젯 값만 override, 나머지는 core 기본값."""

    def _build(self, widgets):
        from ui.generator_generation import GenerationMixin
        return GenerationMixin._build_adetailer_slot(object(), widgets)

    def test_default_widgets_match_legacy_bytes(self):
        self.assertEqual(_dumps(self._build(_widgets())), _dumps(_legacy_literal()))

    def test_widget_values_and_use_overrides_match_legacy_bytes(self):
        widgets = _widgets(
            model=_W('None'),  # 'None'/빈 값은 기본 모델로
            confidence=_W('0.55'), denoise=_W('0.33'), mask_blur=_W('6'), padding=_W('48'),
            prompt=_W('smile'), neg_prompt=_W('lowres'),
            dilate_erode=_W('8'), mask_merge_invert=_W('Merge'),
            use_inpaint_size_check=_W(checked=True), inpaint_width=_W('768'),
            inpaint_height=_W('640'), use_steps_check=_W(checked=True), steps=_W('20'),
            use_cfg_check=_W(checked=True), cfg=_W('5.5'),
            use_checkpoint_check=_W(checked=True), checkpoint_combo=_W('ckpt.safetensors'),
            use_vae_check=_W(checked=True), vae_combo=_W(''),  # 빈 VAE 는 None 유지
            use_sampler_check=_W(checked=True), sampler_combo=_W('Euler a'),
            scheduler_combo=_W(''),
        )
        want = _legacy_literal(
            confidence=0.55, denoise=0.33, blur=6, padding=48, prompt='smile', neg='lowres',
            dilate=8, merge='Merge', use_size=True, iw=768, ih=640, use_steps=True, steps=20,
            use_cfg=True, cfg=5.5, use_ckpt=True, ckpt='ckpt.safetensors', use_vae=True,
            use_sampler=True, sampler='Euler a', scheduler='Use same scheduler',
        )
        self.assertEqual(_dumps(self._build(widgets)), _dumps(want))

    def test_unchecked_use_flags_keep_combo_values_out(self):
        widgets = _widgets(checkpoint_combo=_W('ckpt'), vae_combo=_W('vae'),
                           sampler_combo=_W('Euler'), scheduler_combo=_W('Karras'))
        slot = self._build(widgets)
        self.assertIsNone(slot['ad_checkpoint'])
        self.assertIsNone(slot['ad_vae'])
        self.assertEqual(slot['ad_sampler'], 'DPM++ 2M Karras')
        self.assertEqual(slot['ad_scheduler'], 'Use same scheduler')


if __name__ == '__main__':
    unittest.main()
