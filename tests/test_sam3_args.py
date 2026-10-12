"""SAM3 인자 계약 회귀 테스트.

확장의 `Sam3Args`는 `extra=Extra.forbid` — 모르는 키가 하나라도 섞이면 검증이 실패하고
SAM3가 **조용히 꺼진다**(에러도 안 뜸). 그래서 필드 집합이 정확히 맞는지 못 박는다.
확장이 설치돼 있으면 실제 `sam3ext/args.py`를 파싱해 교차검증한다.

API 경로의 `process()` 는 state 에서 정해진 키만 골라 Sam3Args 에 넘기므로(gap matrix 나-4) 반대쪽 위험 —
확장이 키 이름을 바꿔 앱 값이 조용히 버려지는 것 — 은 `tests/test_sam_extra_contract.py` 가 process() 의
키 집합으로 본다. 확장 위치 찾기와 skip/강제 규칙은 `tests/_sam_extra_ext.py` 공용 헬퍼.
"""
import ast
import os
import unittest

from core.sam3_args import (
    FACE_CROP_DEFAULT_CONFIDENCE,
    FACE_CROP_DEFAULT_MODEL,
    FACE_CROP_KEYS,
    SAM3_KEYS,
    SAM3_SPEC,
    SCRIPT_SAM3,
    apply_to_payload,
    build_alwayson,
    build_state,
    default_settings,
)
from tests._sam_extra_ext import EXT_ROOT, requires_extension

_ARGS_PY = os.path.join(str(EXT_ROOT), 'sam3ext', 'args.py') if EXT_ROOT is not None else ''

# Sam3Args 스키마 밖의 활성화 플래그 — state에는 있지만 스펙에는 없어야 하는 키
_ACTIVATION_ONLY = {'sam3_enable', 'enabled'}


class TestSpec(unittest.TestCase):
    def test_field_count(self):
        # 49 + 얼굴 안에서만 찾기 셋(sam-extra 0.34.0, 맨 뒤)
        self.assertEqual(len(SAM3_SPEC), 52)
        self.assertEqual(SAM3_KEYS[-3:], FACE_CROP_KEYS)

    def test_keys_unique(self):
        self.assertEqual(len(SAM3_KEYS), len(set(SAM3_KEYS)))

    def test_controlnet_block_present(self):
        cn = [k for k in SAM3_KEYS if k.startswith('sam3_cn_')]
        self.assertEqual(len(cn), 13)

    def test_defaults_match_forge_ui(self):
        d = default_settings()
        # README/메모리에 기록된 Forge 통일값
        self.assertEqual(d['sam3_denoising_strength'], 0.4)
        self.assertEqual(d['sam3_mask_blur'], 4)
        self.assertEqual(d['sam3_inpainting_fill'], 'original')
        # API 경로 기본값이 False라 반드시 True로 명시 전송해야 하는 항목
        self.assertIs(d['sam3_unload_after'], True)


class TestBuildState(unittest.TestCase):
    def test_state_has_activation_flags(self):
        state = build_state({})
        self.assertIs(state['sam3_enable'], True)
        self.assertIs(state['enabled'], True)

    def test_state_keys_are_spec_plus_activation(self):
        # 얼굴 안에서만 찾기 세 키는 켤 때만 — 끈 요청은 0.34.0 전과 같다
        self.assertEqual(set(build_state({})), (set(SAM3_KEYS) - set(FACE_CROP_KEYS)) | _ACTIVATION_ONLY)
        self.assertEqual(set(build_state({'sam3_face_crop': True})), set(SAM3_KEYS) | _ACTIVATION_ONLY)

    def test_face_crop_off_payload_is_unchanged(self):
        """끈 state 는 0.34.0 전 앱이 보내던 키·순서 그대로(얼굴 키 없음) — 꺼짐을 명시로 보내도, 다른 값을 보내도."""
        legacy = build_state({})
        self.assertEqual(list(legacy)[-3:], ['sam3_cn_threshold_b', 'sam3_enable', 'enabled'])
        for settings in ({'sam3_face_crop': False}, {'sam3_face_crop': 'false', 'sam3_face_crop_model': 'x.pt',
                                                      'sam3_face_crop_confidence': 0.9}):
            with self.subTest(settings=settings):
                self.assertEqual(build_state(settings), legacy)

    def test_face_crop_on_sends_three_keys_with_extension_rules(self):
        state = build_state({'sam3_face_crop': 'true', 'sam3_face_crop_model': '  face_yolov8s.pt ',
                             'sam3_face_crop_confidence': '0.45'})
        self.assertEqual([state[key] for key in FACE_CROP_KEYS], [True, 'face_yolov8s.pt', 0.45])
        # 키 순서: 확장 ALL_ARGS 처럼 SAM3 키의 맨 뒤(활성화 플래그 앞)
        self.assertEqual(list(state)[-5:], [*FACE_CROP_KEYS, 'sam3_enable', 'enabled'])
        # 빈 모델 → 기본 모델(확장 _normalise_face_crop_model), 신뢰도 0–1 로 잘림(_NUMERIC_BOUNDS)
        state = build_state({'sam3_face_crop': True, 'sam3_face_crop_model': ' ', 'sam3_face_crop_confidence': 3})
        self.assertEqual(state['sam3_face_crop_model'], FACE_CROP_DEFAULT_MODEL)
        self.assertEqual(state['sam3_face_crop_confidence'], 1.0)
        state = build_state({'sam3_face_crop': True, 'sam3_face_crop_confidence': 'nan'})
        self.assertEqual(state['sam3_face_crop_confidence'], FACE_CROP_DEFAULT_CONFIDENCE)

    def test_face_crop_defaults(self):
        d = default_settings()
        self.assertIs(d['sam3_face_crop'], False)
        self.assertEqual(d['sam3_face_crop_model'], 'face_yolov8n.pt')
        self.assertEqual(d['sam3_face_crop_confidence'], 0.3)

    def test_unknown_keys_are_dropped(self):
        # extra=forbid 대응 — 우리가 모르는 키가 밖으로 새면 안 된다
        state = build_state({'totally_unknown': 1, 'sam3_bogus': 'x'})
        self.assertNotIn('totally_unknown', state)
        self.assertNotIn('sam3_bogus', state)

    def test_empty_detect_prompt_falls_back_to_face(self):
        self.assertEqual(build_state({'sam3_prompt': '   '})['sam3_prompt'], 'face')

    def test_prompt_fallback_from_parent(self):
        state = build_state({}, prompt='1girl, solo', negative_prompt='bad hands')
        self.assertEqual(state['sam3_inpaint_prompt'], '1girl, solo')
        self.assertEqual(state['sam3_negative_prompt'], 'bad hands')

    def test_explicit_prompt_wins_over_parent(self):
        state = build_state({'sam3_inpaint_prompt': 'face closeup'}, prompt='1girl')
        self.assertEqual(state['sam3_inpaint_prompt'], 'face closeup')

    def test_string_coercion(self):
        state = build_state({
            'sam3_mask_hull': 'true',
            'sam3_threshold': '0.55',
            'sam3_mask_dilation': '6',
        })
        self.assertIs(state['sam3_mask_hull'], True)
        self.assertEqual(state['sam3_threshold'], 0.55)
        self.assertEqual(state['sam3_mask_dilation'], 6)

    def test_bad_choice_falls_back(self):
        self.assertEqual(build_state({'sam3_mode': 'nonsense'})['sam3_mode'], 'Inpaint')
        self.assertEqual(
            build_state({'sam3_cn_control_mode': 'x'})['sam3_cn_control_mode'], 'Balanced')

    def test_choice_is_canonicalised(self):
        self.assertEqual(build_state({'sam3_mode': 'mask only'})['sam3_mode'], 'Mask only')

    def test_ranges_clamped(self):
        self.assertEqual(build_state({'sam3_threshold': 5})['sam3_threshold'], 1.0)
        self.assertEqual(build_state({'sam3_seed': -99})['sam3_seed'], -1)
        self.assertEqual(build_state({'sam3_cn_weight': 9})['sam3_cn_weight'], 2.0)


class TestPayload(unittest.TestCase):
    def test_alwayson_shape(self):
        block = build_alwayson({})
        self.assertEqual(list(block), [SCRIPT_SAM3])
        self.assertEqual(set(block[SCRIPT_SAM3]), {'args'})
        args = block[SCRIPT_SAM3]['args']
        self.assertEqual(len(args), 1)
        self.assertIsInstance(args[0], dict)

    def test_apply_uses_payload_prompt_as_fallback(self):
        payload = {'prompt': 'a cat', 'negative_prompt': 'blurry'}
        apply_to_payload(payload, {})
        state = payload['alwayson_scripts'][SCRIPT_SAM3]['args'][0]
        self.assertEqual(state['sam3_inpaint_prompt'], 'a cat')
        self.assertEqual(state['sam3_negative_prompt'], 'blurry')

    def test_apply_does_not_clobber(self):
        sentinel = {'args': ['keep']}
        payload = {'alwayson_scripts': {SCRIPT_SAM3: sentinel}}
        apply_to_payload(payload, {})
        self.assertEqual(payload['alwayson_scripts'][SCRIPT_SAM3], sentinel)


def _parse_sam3args_fields(path: str) -> list:
    """sam3ext/args.py의 `class Sam3Args` 어노테이션 필드 이름 순서."""
    with open(path, 'r', encoding='utf-8') as fh:
        tree = ast.parse(fh.read())
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == 'Sam3Args':
            return [stmt.target.id for stmt in node.body
                    if isinstance(stmt, ast.AnnAssign) and isinstance(stmt.target, ast.Name)]
    return []


@requires_extension
class TestAgainstInstalledExtension(unittest.TestCase):
    def test_field_names_match_exactly(self):
        live = _parse_sam3args_fields(_ARGS_PY)
        self.assertTrue(live, "Sam3Args 파싱 실패")
        missing = sorted(set(live) - set(SAM3_KEYS))
        extra = sorted(set(SAM3_KEYS) - set(live))
        self.assertEqual(
            (missing, extra), ([], []),
            f"\nSam3Args와 core/sam3_args.py 필드 불일치."
            f"\n확장에만 있음(누락): {missing}"
            f"\n앱에만 있음(extra=forbid 위반 위험): {extra}")

    def test_no_extra_keys_would_break_validation(self):
        """build_state 결과에서 활성화 플래그를 뺀 나머지는 전부 Sam3Args가 아는 키여야 한다."""
        live = set(_parse_sam3args_fields(_ARGS_PY))
        sent = set(build_state({})) - _ACTIVATION_ONLY
        self.assertEqual(sent - live, set())


if __name__ == '__main__':
    unittest.main()
