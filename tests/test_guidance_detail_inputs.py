"""Anima 가이던스 디테일 묶음(Perturbation 인자 62-90) 칸 = 확장 슬라이더·라디오 그대로.

앱 스펙(core/anima_guidance.py)의 범위·선택지는 tests/test_sam_extra_contract.py 가 픽스처와 대조한다. 여기서는
Vue 섹션(PagSection·CwmSmcSection·DetailStagesSection)의 숫자 칸 min/max/step 과 선택지 목록을 같은 픽스처
(라이브 Forge script-info, tests/fixtures/sam_extra_script_info.json)와 대조하고, 62-90 의 모든 키가 어느 섹션엔가
바인딩돼 있는지 본다 — 빠진 칸은 앱에서 바꿀 수 없는 설정이 된다.
"""
import json
import re
import unittest
from pathlib import Path

from core import anima_guidance

ROOT = Path(__file__).resolve().parent.parent
FIXTURE = ROOT / 'tests' / 'fixtures' / 'sam_extra_script_info.json'
GUIDANCE = ROOT / 'frontend' / 'src' / 'components' / 'guidance'
SECTION_FILES = ('PagSection.vue', 'CwmSmcSection.vue', 'DetailStagesSection.vue')
DETAIL_SPEC = anima_guidance.PERTURBATION_SPEC[anima_guidance.PAG_DETAIL_SUITE_FROM:]
INDEX = {key: i for i, (key, *_rest) in enumerate(anima_guidance.PERTURBATION_SPEC)}


def _live_args() -> list:
    data = json.loads(FIXTURE.read_text(encoding='utf-8'))
    for script in data['scripts']:
        if script.get('name') == anima_guidance.SCRIPT_PERTURBATION.casefold() and not script.get('is_img2img'):
            return script['args']
    raise AssertionError('픽스처에 txt2img Perturbation 항목이 없다')


def _sources() -> dict:
    return {name: (GUIDANCE / name).read_text(encoding='utf-8') for name in SECTION_FILES}


def _bound_inputs(source: str) -> dict:
    """v-model="w._key" 로 묶인 <input> 태그(키 → 태그 목록)."""
    out = {}
    for tag in re.findall(r'<input\b[^>]*>', source):
        match = re.search(r'v-model="w\._(\w+)"', tag)
        if match:
            out.setdefault(match.group(1), []).append(tag)
    return out


def _attr(tag: str, name: str):
    match = re.search(rf'\s{name}="([^"]*)"', tag)
    return match.group(1) if match else None


def _string_list(source: str, name: str) -> list:
    match = re.search(rf'const {name} = \[([^\]]*)\]', source)
    if match is None:
        raise AssertionError(f'{name} 배열이 없다')
    return re.findall(r"'([^']*)'", match.group(1))


class DetailSuiteInputTests(unittest.TestCase):
    def setUp(self):
        self.args = _live_args()
        self.sources = _sources()
        self.inputs = {}
        for source in self.sources.values():
            for key, tags in _bound_inputs(source).items():
                self.inputs.setdefault(key, []).extend(tags)

    def test_number_inputs_use_the_extension_slider_ranges(self):
        floats = [key for key, kind, *_rest in DETAIL_SPEC if kind == 'float']
        self.assertGreater(len(floats), 0)
        for key in floats:
            with self.subTest(key=key):
                tags = self.inputs.get(key, [])
                self.assertEqual(len(tags), 1, '숫자 칸은 한 번만 묶인다')
                tag = tags[0]
                self.assertEqual(_attr(tag, 'type'), 'number')
                live = self.args[INDEX[key]]
                self.assertEqual(
                    [float(_attr(tag, name)) for name in ('min', 'max', 'step')],
                    [float(live['minimum']), float(live['maximum']), float(live['step'])])

    def test_every_detail_key_is_editable_in_a_section(self):
        text = '\n'.join(self.sources.values())
        for key, kind, *_rest in DETAIL_SPEC:
            with self.subTest(key=key):
                if kind == 'bool':
                    self.assertIn(f"setB('{key}'", text)
                elif kind == 'choice':
                    self.assertIn(f'v-model="w._{key}"', text)
                else:
                    self.assertIn(key, self.inputs)

    def test_text_field_is_a_text_input(self):
        self.assertEqual(_attr(self.inputs['guid_s2_blocks'][0], 'type'), 'text')

    def test_select_options_are_the_extension_radio_choices(self):
        cases = (
            ('guid_slg_mode', 'PagSection.vue', 'slgModes', anima_guidance.SLG_MODE_S2),
            ('guid_smc_mode', 'CwmSmcSection.vue', 'smcModes', anima_guidance.SMC_MODE_ADAPTIVE),
        )
        for key, file, name, switch_value in cases:
            with self.subTest(key=key):
                source = self.sources[file]
                live = self.args[INDEX[key]]['choices']
                spec_choices = next(extra for k, _kind, _d, extra in DETAIL_SPEC if k == key)
                self.assertEqual(_string_list(source, name), live)
                self.assertEqual(list(spec_choices), live)
                # 칸을 바꿔 보여 주는 비교도 같은 글자다
                self.assertIn(f"=== '{switch_value}'", source)

    def test_ts_mirror_of_the_detail_suite_start(self):
        source = (ROOT / 'frontend' / 'src' / 'utils' / 'guidanceDetailSuite.ts').read_text(encoding='utf-8')
        match = re.search(r'export const PAG_DETAIL_SUITE_FROM = (\d+)', source)
        self.assertIsNotNone(match)
        self.assertEqual(int(match.group(1)), anima_guidance.PAG_DETAIL_SUITE_FROM)

    def test_panel_badge_uses_the_same_mode_literals(self):
        panel = (ROOT / 'frontend' / 'src' / 'components' / 'AnimaGuidancePanel.vue').read_text(encoding='utf-8')
        self.assertIn(f"=== '{anima_guidance.SLG_MODE_S2}'", panel)
        self.assertIn(f"=== '{anima_guidance.SMC_MODE_ADAPTIVE}'", panel)


class OptimalScaleInputTests(unittest.TestCase):
    """Anima Optimal Scale 섹션(인자 4개) = 확장 슬라이더 그대로, 네 칸 모두 바인딩."""

    def test_inputs_use_the_extension_slider_ranges(self):
        data = json.loads(FIXTURE.read_text(encoding='utf-8'))
        live = next(s['args'] for s in data['scripts']
                    if s.get('name') == anima_guidance.SCRIPT_OPTIMAL_SCALE.casefold() and not s.get('is_img2img'))
        source = (GUIDANCE / 'OptimalScaleSection.vue').read_text(encoding='utf-8')
        inputs = _bound_inputs(source)
        self.assertIn("setB('ocfg_enabled'", source)
        for index, (key, kind, _default, _extra) in enumerate(anima_guidance.OPTIMAL_SCALE_SPEC):
            if kind != 'float':
                continue
            with self.subTest(key=key):
                self.assertEqual(len(inputs.get(key, [])), 1)
                tag = inputs[key][0]
                self.assertEqual(
                    [float(_attr(tag, name)) for name in ('min', 'max', 'step')],
                    [float(live[index]['minimum']), float(live[index]['maximum']), float(live[index]['step'])])


if __name__ == '__main__':
    unittest.main()
