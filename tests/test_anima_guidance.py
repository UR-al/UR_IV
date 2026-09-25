"""Anima Guidance Suite 인자 계약 회귀 테스트.

확장(sam-extra)의 세 스크립트는 args를 **위치(index)로만** 읽는다. 한 칸만 밀려도
조용히 엉뚱한 파라미터가 적용되고, 로그에도 안 남는다. 그래서 두 겹으로 막는다:

1) 리터럴 고정 — 아래 EXPECTED_* 가 core/anima_guidance 스펙과 일치하는지.
   (우리 쪽 스펙을 실수로 건드리면 여기서 깨짐)
2) 설치된 확장과 교차검증 — Forge 경로에 확장이 있으면 실제 `ui()`의 `return [...]`을
   파싱해 순서를 대조. 확장이 없으면 skip (다른 PC/CI에서도 테스트가 돌아야 하므로).
"""
import ast
import os
import re
import unittest

from core.anima_guidance import (
    DETAIL_DAEMON_SPEC,
    PERTURBATION_SPEC,
    SCRIPT_DETAIL_DAEMON,
    SCRIPT_PERTURBATION,
    SCRIPT_SKIMMED_CFG,
    SKIMMED_SPEC,
    build_alwayson,
    build_args,
    default_settings,
    describe_active,
    is_script_active,
    parse_forge_script_info,
)
from tests._sam_extra_ext import EXT_ROOT, requires_extension

# 확장 위치 찾기·skip/강제 규칙은 공용 헬퍼(tests/_sam_extra_ext.py). 값 범위·기본값·API 가 실제로 읽는
# 인덱스 같은 더 넓은 계약은 tests/test_sam_extra_contract.py 가 본다.
_EXT_DIR = os.path.join(str(EXT_ROOT), 'scripts') if EXT_ROOT is not None else ''

# ── 1) 리터럴 고정 ───────────────────────────────────────────────────────────
# 확장 varname 순서 그대로. 우리 키는 접두사만 다름(guid_/skim_/dd_).
EXPECTED_PERTURBATION = [
    'enabled', 'attn_method', 'scale', 'legacy_strength', 'block_indices',
    'slg_on', 'slg_scale', 'slg_blocks',
    'start_percent', 'end_percent', 'rescale', 'auto_decay',
    'apg_enabled', 'apg_eta', 'apg_norm', 'apg_momentum', 'apg_autooff',
    'adg_enabled', 'adg_start', 'adg_interval',
    'legacy_attn', 'seg_sigma',
    'cfg_mode', 'experimental_stack',
    'cwm_alpha_low', 'cwm_alpha_high', 'smc_lambda', 'smc_k',
    'dcw_enabled', 'dcw_lambda_low', 'dcw_lambda_high',
    'dave_enabled', 'dave_strength', 'dave_tau', 'dave_blocks',
    'cns_enabled', 'cns_strength', 'cns_gamma_power', 'cns_gamma_scale',
    'official_strength', 'head_indices', 'rescale_mode',
    'smc_enabled', 'cwm_enabled',
    'mod_enabled', 'mod_clip_model', 'mod_weight',
    'mod_start_layer', 'mod_end_layer',
    'mod_base_source', 'mod_base_prompt', 'mod_positive_prompt',
    'mod_negative_source', 'mod_negative_prompt',
    'mod_adapter_mode', 'mod_adapter_path',
    'smc_preset', 'smc_master_enabled',
    'rdc_enabled', 'rdc_tau', 'rdc_alpha_ll', 'rdc_alpha_hh',
]
EXPECTED_SKIMMED = [
    'enabled', 'skimming_cfg', 'full_skim_negative',
    'disable_flipping_filter', 'start_percent', 'end_percent', 'flip_at',
]
EXPECTED_DETAIL_DAEMON = [
    'enabled', 'preset', 'amount', 'start', 'end', 'bias', 'exponent',
    'start_offset', 'end_offset', 'fade', 'multiplier', 'smooth', 'cfg_couple',
    'hires',
]

_PREFIXES = {
    SCRIPT_PERTURBATION: ('guid_', PERTURBATION_SPEC, EXPECTED_PERTURBATION,
                          'anima_safe_pag.py'),
    SCRIPT_SKIMMED_CFG: ('skim_', SKIMMED_SPEC, EXPECTED_SKIMMED,
                         'anima_skimmed_cfg.py'),
    SCRIPT_DETAIL_DAEMON: ('dd_', DETAIL_DAEMON_SPEC, EXPECTED_DETAIL_DAEMON,
                           'anima_detail_daemon.py'),
}


# ── 원본 노드 입력 상수표 ────────────────────────────────────────────────────
# 원본 INPUT_TYPES 에서 옮긴 (default, min, max, step). GPL-3.0 원본(DCW·CNS)은 숫자만 옮겼다 — 코드는 복사하지 않는다.
# step 은 앱 스펙에 없다(범위만 자른다) — Vue 칸의 step 은 frontend guidanceOriginInputs.test.ts 가 본다.
_DCW = 'namemechan/ComfyUI-DCW@66aaf9dd:dcw_node.py'                          # GPL-3.0, 숫자만
_CNS = 'namemechan/comfyui-cns_sampler_patch@42278b13:cns_sampler_patch.py'   # GPL-3.0, 숫자만
_DAVE = 'sorryhyun/ComfyUI-Anima-DAVE@83143e8d:nodes.py'                       # MIT
_SKIM = 'Extraltodeus/Skimmed_CFG@d8300583:skimmed_CFG.py'                     # Apache-2.0
_PAG = 'iljung1106/comfyui-anima-safe-pag@905b0107:__init__.py'                # MIT
ORIGIN_INPUTS = {
    # 앱 키: (default, min, max, step, 'origin: 저장소@커밋:파일:줄')
    'guid_dcw_lambda_low':    (0.05, -0.5, 0.5, 0.005, f'{_DCW}:637-652'),
    'guid_dcw_lambda_high':   (0.01, -0.3, 0.3, 0.001, f'{_DCW}:653-668'),
    'guid_cwm_alpha_low':     (0.0, -1.0, 2.0, 0.01, f'{_DCW}:675-690'),
    'guid_cwm_alpha_high':    (0.0, -1.0, 2.0, 0.01, f'{_DCW}:691-706'),
    'guid_smc_lambda':        (6.0, 0.5, 30.0, 0.1, f'{_DCW}:732-743'),
    'guid_smc_k':             (0.1, 0.0, 5.0, 0.01, f'{_DCW}:744-755'),
    'guid_rdc_tau':           (0.0, 0.0, 0.5, 0.01, f'{_DCW}:761-783'),
    'guid_rdc_alpha_ll':      (0.03, 0.0, 0.3, 0.005, f'{_DCW}:784-799'),
    'guid_rdc_alpha_hh':      (0.0, 0.0, 0.1, 0.001, f'{_DCW}:800-815'),
    'guid_cns_strength':      (1.0, 0.0, 1.0, 0.05, f'{_CNS}:396-408'),
    'guid_cns_gamma_power':   (0.5, 0.1, 2.0, 0.05, f'{_CNS}:409-422'),
    'guid_cns_gamma_scale':   (2.0, 0.1, 25.0, 0.1, f'{_CNS}:423-437'),
    'guid_dave_strength':     (0.30, 0.0, 1.0, 0.01, f'{_DAVE}:121-133'),
    'guid_dave_tau':          (0.10, 0.0, 1.0, 0.01, f'{_DAVE}:134-146'),
    'guid_scale':             (4.0, 0.0, 100.0, 0.1, f'{_PAG}:201'),
    'guid_official_strength': (0.75, 0.0, 1.0, 0.01, f'{_PAG}:203'),
    'guid_start_percent':     (0.0, 0.0, 1.0, 0.001, f'{_PAG}:205'),
    'guid_end_percent':       (0.7, 0.0, 1.0, 0.001, f'{_PAG}:206'),
    'guid_rescale':           (0.2, 0.0, 1.0, 0.01, f'{_PAG}:207'),
    'skim_skimming_cfg':      (7.0, 0.0, 10.0, 0.5, f'{_SKIM}:5-6, :93-100'),
    'skim_start_percent':     (0.0, 0.0, 1.0, 0.01, f'{_SKIM}:111-118'),
    'skim_end_percent':       (1.0, 0.0, 1.0, 0.01, f'{_SKIM}:119-126'),
    'skim_flip_at':           (0.0, 0.0, 1.0, 0.01, f'{_SKIM}:127-134'),
}
# 원본과 일부러 다른 칸 — (키, 필드): (앱 값, 사유). 여기 없는 차이는 실패, 더는 다르지 않으면 역시 실패.
ORIGIN_HOST_DIFFS = {
    ('skim_skimming_cfg', 'min'): (-1.0, '원본 래퍼 노드(Clean Skim·Timed flip, skimmed_CFG.py:204-281)가 넘기는 −1'
                                         '(= 현재 CFG)을 스크립트 하나로 대신한다'),
}
# 원본 문자열·선택 입력 (값, origin)
ORIGIN_TEXT_INPUTS = {
    'guid_block_indices':  ('18', f'{_PAG}:202'),
    'guid_head_indices':   ('', f'{_PAG}:204'),
    'guid_rescale_mode':   ('full', f'{_PAG}:208'),
    # dave_alpha.npz = 평평한 블록 8~18 가중치 1 (nodes.py:117-120 의 기본 마스크) — 앱은 블록 칸으로 대신한다
    'guid_dave_blocks':    ('8-18', f'{_DAVE}:117-120'),
    'skim_full_skim_negative':      (False, f'{_SKIM}:101-105'),
    'skim_disable_flipping_filter': (False, f'{_SKIM}:106-110'),
}


def _spec_entry(key):
    for spec in (PERTURBATION_SPEC, SKIMMED_SPEC, DETAIL_DAEMON_SPEC):
        for entry in spec:
            if entry[0] == key:
                return entry
    raise KeyError(key)


class TestOriginInputs(unittest.TestCase):
    """새 기본값·범위 = 원본 노드 INPUT_TYPES (저장된 사용자 값은 건드리지 않는다 — 아래 TestSavedValues)."""

    def test_numeric_defaults_and_ranges_follow_origin(self):
        seen = set()
        for key, (default, lo, hi, _step, origin) in ORIGIN_INPUTS.items():
            _k, kind, app_default, (app_lo, app_hi) = _spec_entry(key)
            with self.subTest(key=key, origin=origin):
                self.assertEqual(kind, 'float')
                for field, app, want in (('default', app_default, default), ('min', app_lo, lo),
                                         ('max', app_hi, hi)):
                    diff = ORIGIN_HOST_DIFFS.get((key, field))
                    if diff is not None:
                        seen.add((key, field))
                        self.assertEqual(app, diff[0], diff[1])
                        self.assertNotEqual(app, want, f'{key}.{field} 는 이제 원본과 같다 — ORIGIN_HOST_DIFFS 에서 지워라')
                    else:
                        self.assertEqual(app, want, f'{key}.{field}')
        self.assertEqual(seen, set(ORIGIN_HOST_DIFFS))

    def test_text_and_choice_defaults_follow_origin(self):
        for key, (default, origin) in ORIGIN_TEXT_INPUTS.items():
            with self.subTest(key=key, origin=origin):
                self.assertEqual(_spec_entry(key)[2], default)

    def test_sent_values_are_clamped_to_origin_ranges(self):
        # 원본 노드는 범위 밖 값을 받지 않는다(ComfyUI 입력 검사) — 앱은 끝값으로 잘라 보낸다
        for key, (_default, lo, hi, _step, origin) in ORIGIN_INPUTS.items():
            if key.startswith('guid_rdc_'):
                continue   # RDC 칸은 켜짐 규칙(아래 TestRdcGate)을 따로 본다
            title = SCRIPT_SKIMMED_CFG if key.startswith('skim_') else SCRIPT_PERTURBATION
            index = [entry[0] for entry in SPECS_BY_TITLE[title]].index(key)
            lo = ORIGIN_HOST_DIFFS.get((key, 'min'), (lo,))[0]
            with self.subTest(key=key, origin=origin):
                self.assertEqual(build_args(title, {key: hi + 1})[index], hi)
                self.assertEqual(build_args(title, {key: lo - 1})[index], lo)


SPECS_BY_TITLE = {SCRIPT_PERTURBATION: PERTURBATION_SPEC, SCRIPT_SKIMMED_CFG: SKIMMED_SPEC}
_ARG = {key: index for index, (key, *_rest) in enumerate(PERTURBATION_SPEC)}


class TestRdcGate(unittest.TestCase):
    """원본 RDC: 스위치 없이 rdc_tau > 0 이면 켜지고, DCW 보정 안에서만 돈다(dcw_enabled 가 꺼지면 같이 꺼짐).

    origin: namemechan/ComfyUI-DCW@66aaf9dd:dcw_node.py:757-760(스위치 없음, tau 0 = 끔), :855-856
    (``rdc_on = rdc_tau > 0``; ``dcw_active = dcw_enabled and (λ_l ≠ 0 or λ_h ≠ 0 or rdc_on)``) — 뜻만 옮김.
    앱의 Enable RDC 스위치는 저장된 사용자 값이라 남고(RdcSection.vue), 세 가지가 다 켜져야 RDC 가 간다.
    보내는 칸: 58 = 원본 켜짐(옛 확장 빌드는 이것만 본다), 59 = 켜짐일 때만 tau(원본 동등성 빌드 DCW-F 는
    dcw_enabled and tau > 0 으로 켜고 58 은 끄는 쪽으로만 읽는다 — 꺼짐이면 tau 0 도 보내 어느 판단이든 같게)."""

    CASES = (
        # (dcw, 스위치, tau) → (arg58, arg59)
        (False, False, 0.0, False, 0.0),
        (True, False, 0.0, False, 0.0),
        (True, True, 0.0, False, 0.0),        # tau 0 = 원본에서 끔
        (False, True, 0.2, False, 0.0),       # DCW 가 꺼지면 RDC 도 꺼짐(옛 확장은 여기서 켰다)
        (True, False, 0.15, False, 0.0),      # 스위치를 끈 채 예전 기본 0.15 를 저장한 사용자 — DCW 를 켜도 RDC 는 끔
        (True, True, 0.15, True, 0.15),
        (True, True, 0.5, True, 0.5),
    )

    def test_sent_rdc_slots_follow_origin_gate(self):
        for dcw, switch, tau, on, sent_tau in self.CASES:
            settings = {'guid_dcw_enabled': dcw, 'guid_rdc_enabled': switch, 'guid_rdc_tau': tau}
            with self.subTest(**settings):
                args = build_args(SCRIPT_PERTURBATION, settings)
                self.assertIs(args[_ARG['guid_rdc_enabled']], on)
                self.assertEqual(args[_ARG['guid_rdc_tau']], sent_tau)
                # 원본 동등성 빌드의 판단(58 을 읽지 않음)과 옛 빌드의 판단(58 만 봄)이 같은 답을 낸다
                self.assertEqual(bool(args[_ARG['guid_dcw_enabled']]) and args[_ARG['guid_rdc_tau']] > 0, on)
                self.assertEqual(('RDC' in describe_active(settings)), on)

    def test_alphas_and_dcw_slots_are_sent_untouched(self):
        args = build_args(SCRIPT_PERTURBATION, {'guid_dcw_enabled': True, 'guid_rdc_enabled': False,
                                                'guid_rdc_tau': 0.3, 'guid_rdc_alpha_ll': 0.05,
                                                'guid_dcw_lambda_low': 0.07})
        self.assertEqual(args[_ARG['guid_rdc_alpha_ll']], 0.05)
        self.assertEqual(args[_ARG['guid_dcw_lambda_low']], 0.07)
        self.assertIs(args[_ARG['guid_dcw_enabled']], True)


class TestSummaryFollowsOriginActivity(unittest.TestCase):
    """요약(describe_active)은 원본이 실제로 훅을 거는 조건으로 적는다.

    origin: namemechan/ComfyUI-DCW@66aaf9dd:dcw_node.py:855-858 (dcw_active·cwm_alpha_active — 뜻만),
    sorryhyun/ComfyUI-Anima-DAVE@83143e8d:nodes.py:163-166 (atten > 1e-3 인 블록만)."""

    def test_cwm_needs_a_non_zero_alpha(self):
        self.assertEqual(describe_active({'guid_cwm_enabled': True}), '')          # 새 기본 alpha 0/0 = 보통 CFG
        self.assertEqual(describe_active({'guid_cwm_enabled': True, 'guid_cwm_alpha_high': 0.2}), 'CWM')
        self.assertEqual(describe_active({'guid_cwm_enabled': True, 'guid_cwm_alpha_low': -0.1}), 'CWM')

    def test_dcw_needs_a_lambda_or_rdc(self):
        self.assertEqual(describe_active({'guid_dcw_enabled': True}), 'DCW')       # 기본 0.05/0.01
        zero = {'guid_dcw_enabled': True, 'guid_dcw_lambda_low': 0, 'guid_dcw_lambda_high': 0}
        self.assertEqual(describe_active(zero), '')
        self.assertEqual(describe_active({**zero, 'guid_rdc_enabled': True, 'guid_rdc_tau': 0.1}), 'DCW + RDC')

    def test_dave_needs_strength_above_1e_3(self):
        self.assertEqual(describe_active({'guid_dave_enabled': True}), 'DAVE')
        self.assertEqual(describe_active({'guid_dave_enabled': True, 'guid_dave_strength': 0.001}), '')
        self.assertEqual(describe_active({'guid_dave_enabled': True, 'guid_dave_strength': 0.002}), 'DAVE')


class TestVueInputsFollowSpec(unittest.TestCase):
    """숫자 칸(frontend/src/components/guidance/*Section.vue)의 min/max/step = 앱 스펙 범위 = 원본 노드 입력.

    칸의 step 만 원본 상수표에서 보고, min/max 는 앱 스펙(= 원본, 호스트 차이는 ORIGIN_HOST_DIFFS)과 같아야 한다 —
    화면이 허용하는 값과 앱이 잘라 보내는 값이 갈라지지 않게. vitest guidanceOriginInputs.test.ts 와 같은 대조를
    파이썬 스펙 쪽에서 한 번 더 한다."""

    _INPUT = re.compile(r'<input\b[^>]*\bv-model="w\._(?P<key>\w+)"[^>]*>')

    def _inputs(self):
        folder = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                              'frontend', 'src', 'components', 'guidance')
        found = {}
        for name in sorted(os.listdir(folder)):
            if name.endswith('.vue'):
                with open(os.path.join(folder, name), encoding='utf-8') as fh:
                    for match in self._INPUT.finditer(fh.read()):
                        attrs = dict(re.findall(r'\b(min|max|step)="([^"]+)"', match.group(0)))
                        found[match.group('key')] = (name, attrs)
        return found

    def test_numeric_inputs_match_spec_ranges_and_origin_steps(self):
        found = self._inputs()
        owned = [key for key in ORIGIN_INPUTS if not key.startswith('guid_') or key.split('_')[1] in
                 ('dcw', 'cwm', 'smc', 'rdc', 'cns', 'dave')]
        missing = [key for key in owned if key not in found]
        self.assertEqual(missing, [], '원본 대조 칸이 Vue 섹션에서 사라졌다')
        for key in owned:
            name, attrs = found[key]
            _k, _kind, _default, (lo, hi) = _spec_entry(key)
            with self.subTest(key=key, file=name):
                self.assertEqual(float(attrs['min']), lo)
                self.assertEqual(float(attrs['max']), hi)
                self.assertEqual(float(attrs['step']), ORIGIN_INPUTS[key][3])


class TestDaveBlankBlocks(unittest.TestCase):
    """빈 DAVE 블록 칸 = 원본 기본 마스크(블록 8~18) — 옛 확장은 {18}, 팩은 8-18 로 읽어 백엔드마다 달랐다."""

    def test_blank_blocks_are_sent_as_8_18(self):
        index = _ARG['guid_dave_blocks']
        for blank in ('', '   ', None):
            with self.subTest(blocks=blank):
                args = build_args(SCRIPT_PERTURBATION, {'guid_dave_enabled': True, 'guid_dave_blocks': blank})
                self.assertEqual(args[index], '8-18')
        self.assertEqual(build_args(SCRIPT_PERTURBATION, {'guid_dave_blocks': '10-12'})[index], '10-12')
        # 다른 텍스트 칸은 빈 값을 그대로 보낸다(PAG head_indices '' = 모든 헤드)
        self.assertEqual(build_args(SCRIPT_PERTURBATION, {'guid_head_indices': ''})[_ARG['guid_head_indices']], '')

    def test_blank_blocks_import_as_8_18(self):
        entry = {'name': SCRIPT_PERTURBATION.lower(), 'is_img2img': False,
                 'args': [{'value': ('' if key == 'guid_dave_blocks' else default)}
                          for key, _kind, default, _extra in PERTURBATION_SPEC]}
        settings, _meta = parse_forge_script_info([entry])
        self.assertEqual(settings['guid_dave_blocks'], '8-18')


class TestSavedValues(unittest.TestCase):
    """새 기본값은 비어 있는 칸에만 — 저장된 값(예전 기본값 포함)은 그대로 읽어 원본 범위 안이면 그대로 보낸다."""

    OLD_DEFAULTS = {'guid_dcw_lambda_low': 0.10, 'guid_dcw_lambda_high': 0.02, 'guid_cwm_alpha_low': 0.30,
                    'guid_cwm_alpha_high': 0.15, 'guid_cns_gamma_scale': 3.0, 'guid_cns_gamma_power': 0.5}

    def test_old_default_values_are_sent_as_saved(self):
        args = build_args(SCRIPT_PERTURBATION, {key: str(value) for key, value in self.OLD_DEFAULTS.items()})
        for key, value in self.OLD_DEFAULTS.items():
            with self.subTest(key=key):
                self.assertEqual(args[_ARG[key]], value)

    def test_saved_rdc_tau_is_sent_when_rdc_really_runs(self):
        args = build_args(SCRIPT_PERTURBATION, {'guid_dcw_enabled': 'true', 'guid_rdc_enabled': 'true',
                                                'guid_rdc_tau': '0.15'})
        self.assertEqual(args[_ARG['guid_rdc_tau']:_ARG['guid_rdc_tau'] + 1], [0.15])


class TestSpecOrder(unittest.TestCase):
    def test_spec_matches_expected_order(self):
        for title, (prefix, spec, expected, _f) in _PREFIXES.items():
            with self.subTest(script=title):
                got = [key[len(prefix):] for key, _k, _d, _e in spec]
                self.assertEqual(got, expected)

    def test_arg_counts(self):
        self.assertEqual(len(PERTURBATION_SPEC), 62)
        self.assertEqual(len(SKIMMED_SPEC), 7)
        self.assertEqual(len(DETAIL_DAEMON_SPEC), 14)

    def test_keys_are_globally_unique(self):
        keys = [k for spec in (PERTURBATION_SPEC, SKIMMED_SPEC, DETAIL_DAEMON_SPEC)
                for k, _kind, _d, _e in spec]
        self.assertEqual(len(keys), len(set(keys)))


class TestBuildArgs(unittest.TestCase):
    def test_defaults_roundtrip(self):
        # 빈 설정 → 전부 기본값, 길이는 스펙과 동일. Detail Daemon 도 변환 없이 노드 기본값 그대로
        # (원본 대조는 tests/test_detail_daemon_origin.py).
        for title, (_p, spec, _e, _f) in _PREFIXES.items():
            with self.subTest(script=title):
                defaults = [d for _k, _kind, d, _e2 in spec]
                args = build_args(title, {})
                self.assertEqual(len(args), len(spec))
                self.assertEqual(args, defaults)

    def test_default_settings_produce_same_args(self):
        for title in _PREFIXES:
            with self.subTest(script=title):
                self.assertEqual(build_args(title, default_settings()),
                                 build_args(title, {}))

    def test_string_values_are_coerced(self):
        # Vue 위젯 프록시는 전부 문자열로 넘어온다
        args = build_args(SCRIPT_PERTURBATION, {
            'guid_enabled': 'true',
            'guid_scale': '6.5',
            'guid_adg_interval': '3',
            'guid_slg_on': 'false',
        })
        self.assertIs(args[0], True)
        self.assertEqual(args[2], 6.5)
        self.assertEqual(args[19], 3)
        self.assertIs(args[5], False)

    def test_out_of_range_is_clamped_not_dropped(self):
        # PAG scale 범위는 원본 노드 0~100 (iljung1106/comfyui-anima-safe-pag@905b0107:__init__.py:201)
        args = build_args(SCRIPT_PERTURBATION, {'guid_scale': 999})
        self.assertEqual(args[2], 100.0)
        args = build_args(SCRIPT_PERTURBATION, {'guid_scale': 40})
        self.assertEqual(args[2], 40.0)
        args = build_args(SCRIPT_PERTURBATION, {'guid_apg_eta': -50})
        self.assertEqual(args[13], -10.0)

    def test_garbage_falls_back_to_default(self):
        self.assertEqual(build_args(SCRIPT_DETAIL_DAEMON, {'dd_amount': 'abc'})[2], 0.10)
        # 옛 dd_preset 값은 무엇이든 읽지 않는다 — 자리(arg1)에는 늘 중립값
        args = build_args(SCRIPT_DETAIL_DAEMON, {'dd_preset': 'nonsense'})
        self.assertEqual(args[1:3], ['Custom', 0.10])

    def test_choice_is_case_insensitive_but_canonical(self):
        args = build_args(SCRIPT_PERTURBATION, {'guid_cfg_mode': 'smc + cwm'})
        self.assertEqual(args[22], 'SMC + CWM')
        args = build_args(SCRIPT_PERTURBATION, {'guid_attn_method': 'seg'})
        self.assertEqual(args[1], 'SEG')
        args = build_args(SCRIPT_PERTURBATION, {'guid_smc_preset': 'cosmos / wan'})
        self.assertEqual(args[56], 'Cosmos / Wan')

    def test_smc_and_rdc_defaults_follow_origin(self):
        # SMC k 0.1, RDC tau 0 (= 끔) · alpha_ll 0.03 · alpha_hh 0 — ORIGIN_INPUTS 아래 표(원본 노드 숫자)
        args = build_args(SCRIPT_PERTURBATION, {})
        self.assertEqual(args[27], ORIGIN_INPUTS['guid_smc_k'][0])
        self.assertEqual(args[56:62], ['Auto', False, False, ORIGIN_INPUTS['guid_rdc_tau'][0],
                                       ORIGIN_INPUTS['guid_rdc_alpha_ll'][0], ORIGIN_INPUTS['guid_rdc_alpha_hh'][0]])

    def test_unknown_script_raises(self):
        with self.assertRaises(KeyError):
            build_args('Nope', {})


class TestActivation(unittest.TestCase):
    def test_all_off_yields_empty_payload(self):
        self.assertEqual(build_alwayson({}), {})
        self.assertEqual(build_alwayson(default_settings()), {})

    def test_only_active_scripts_included(self):
        block = build_alwayson({'skim_enabled': True})
        self.assertEqual(list(block), [SCRIPT_SKIMMED_CFG])
        self.assertEqual(len(block[SCRIPT_SKIMMED_CFG]['args']), 7)

    def test_dave_alone_activates_perturbation_script(self):
        # DAVE는 PAG 본체 토글과 별개지만 같은 스크립트에 산다
        self.assertTrue(is_script_active(SCRIPT_PERTURBATION, {'guid_dave_enabled': True}))
        block = build_alwayson({'guid_dave_enabled': 'true'})
        self.assertIn(SCRIPT_PERTURBATION, block)
        self.assertIs(block[SCRIPT_PERTURBATION]['args'][31], True)

    def test_smc_master_activates_but_rdc_switch_alone_does_not(self):
        smc = build_alwayson({'guid_smc_master_enabled': 'true'})
        self.assertIs(smc[SCRIPT_PERTURBATION]['args'][57], True)
        # 원본 RDC 는 DCW 보정 안에서만 돈다(origin: namemechan/ComfyUI-DCW@66aaf9dd:dcw_node.py:855-856 — 뜻만):
        # RDC 스위치만 켜면 켜지는 것이 없으니 스크립트를 붙이지 않는다('전부 끄면 Forge 결과 그대로').
        self.assertFalse(is_script_active(SCRIPT_PERTURBATION, {'guid_rdc_enabled': 'true', 'guid_rdc_tau': 0.2}))
        self.assertEqual(build_alwayson({'guid_rdc_enabled': 'true', 'guid_rdc_tau': 0.2}), {})
        rdc = build_alwayson({'guid_rdc_enabled': 'true', 'guid_rdc_tau': 0.2, 'guid_dcw_enabled': 'true'})
        self.assertEqual(rdc[SCRIPT_PERTURBATION]['args'][58:60], [True, 0.2])

    def test_forge_import_of_hidden_rdc_true_does_not_attach_the_script(self):
        """원본 동등성 확장(DCW-F)은 58 칸을 숨은 ``gr.Checkbox(value=True, visible=False)`` 로 내놓는다 —
        'Forge에서 가져오기' 뒤 guid_rdc_enabled 는 늘 True 다. 그것만으로 모든 생성에 중립 스크립트가 붙으면 안 된다
        (sam-extra 가 없는 Forge 에서는 켜지 않은 기능 때문에 422)."""
        entry = TestForgeScriptInfoImport._entry(SCRIPT_PERTURBATION, PERTURBATION_SPEC,
                                                 overrides={'guid_rdc_enabled': True})
        settings, _meta = parse_forge_script_info([entry])
        self.assertIs(settings['guid_rdc_enabled'], True)
        self.assertEqual(build_alwayson(settings), {})

    def test_payload_shape_is_args_list(self):
        block = build_alwayson({'dd_enabled': True})
        self.assertEqual(set(block[SCRIPT_DETAIL_DAEMON]), {'args'})
        self.assertIsInstance(block[SCRIPT_DETAIL_DAEMON]['args'], list)

    def test_describe_active(self):
        text = describe_active({'guid_enabled': True, 'guid_attn_method': 'SEG',
                                'guid_scale': 5, 'skim_enabled': True})
        self.assertIn('SEG(5)', text)
        self.assertIn('Skimmed CFG', text)
        self.assertEqual(describe_active({}), '')
        # 요약도 전송과 같은 0~100 범위로 자른다
        self.assertEqual(describe_active({'guid_enabled': True, 'guid_scale': 50}), 'PAG(50)')
        self.assertEqual(describe_active({'guid_enabled': True, 'guid_scale': 500}), 'PAG(100)')

    def test_describe_current_smc_and_rdc(self):
        text = describe_active({
            'guid_smc_master_enabled': True,
            'guid_rdc_enabled': True, 'guid_rdc_tau': 0.1, 'guid_dcw_enabled': True,
        })
        self.assertIn('SMC', text)
        self.assertIn('RDC', text)
        # 요약도 보내는 값과 같은 규칙 — DCW 가 꺼졌거나 tau 0 이면 RDC 는 돌지 않는다
        self.assertEqual(describe_active({'guid_rdc_enabled': True, 'guid_rdc_tau': 0.1}), '')
        self.assertEqual(describe_active({'guid_rdc_enabled': True, 'guid_dcw_enabled': True}), 'DCW')


class TestApplyToPayload(unittest.TestCase):
    def test_merges_without_clobbering(self):
        from core.anima_guidance import apply_to_payload
        payload = {'alwayson_scripts': {'SAM3 Mask': {'args': [{}]}}}
        apply_to_payload(payload, {'dd_enabled': True})
        self.assertIn('SAM3 Mask', payload['alwayson_scripts'])
        self.assertIn(SCRIPT_DETAIL_DAEMON, payload['alwayson_scripts'])

    def test_existing_key_wins(self):
        from core.anima_guidance import apply_to_payload
        sentinel = {'args': ['keep-me']}
        payload = {'alwayson_scripts': {SCRIPT_DETAIL_DAEMON: sentinel}}
        apply_to_payload(payload, {'dd_enabled': True})
        self.assertEqual(payload['alwayson_scripts'][SCRIPT_DETAIL_DAEMON], sentinel)

    def test_creates_alwayson_key_when_missing(self):
        from core.anima_guidance import apply_to_payload
        payload = {}
        apply_to_payload(payload, {'skim_enabled': True})
        self.assertIn(SCRIPT_SKIMMED_CFG, payload['alwayson_scripts'])

    def test_noop_when_all_off(self):
        from core.anima_guidance import apply_to_payload
        payload = {}
        apply_to_payload(payload, {})
        self.assertEqual(payload, {})


class TestForgeScriptInfoImport(unittest.TestCase):
    @staticmethod
    def _entry(title, spec, *, img2img=False, overrides=None, trailing=0):
        overrides = overrides or {}
        values = [overrides.get(key, default) for key, _kind, default, _extra in spec]
        values.extend(f'future-{i}' for i in range(trailing))
        return {
            'name': title.lower(),
            'is_alwayson': True,
            'is_img2img': img2img,
            'args': [{'value': value} for value in values],
        }

    def test_imports_known_values_by_positional_contract(self):
        payload = [
            self._entry(SCRIPT_PERTURBATION, PERTURBATION_SPEC,
                        overrides={'guid_enabled': True, 'guid_scale': 6.5}),
            self._entry(SCRIPT_SKIMMED_CFG, SKIMMED_SPEC,
                        overrides={'skim_enabled': True}),
            self._entry(SCRIPT_DETAIL_DAEMON, DETAIL_DAEMON_SPEC,
                        overrides={'dd_amount': 2.5, 'dd_hires': True}),
        ]
        settings, meta = parse_forge_script_info(payload)
        self.assertIs(settings['guid_enabled'], True)
        self.assertEqual(settings['guid_scale'], 6.5)
        self.assertIs(settings['skim_enabled'], True)
        # Detail Daemon 은 노드 단위 그대로, 고정 칸(preset 등)은 가져오지 않는다
        self.assertEqual(settings['dd_amount'], 2.5)
        self.assertIs(settings['dd_hires'], True)
        self.assertNotIn('dd_preset', settings)
        self.assertEqual(meta['missing_scripts'], [])

    def test_prefers_txt2img_when_forge_returns_both_modes(self):
        payload = [
            self._entry(SCRIPT_PERTURBATION, PERTURBATION_SPEC, img2img=True,
                        overrides={'guid_scale': 9.0}),
            self._entry(SCRIPT_PERTURBATION, PERTURBATION_SPEC, img2img=False,
                        overrides={'guid_scale': 5.0}),
        ]
        settings, _meta = parse_forge_script_info(payload)
        self.assertEqual(settings['guid_scale'], 5.0)

    def test_future_trailing_forge_args_are_ignored_and_reported(self):
        payload = [self._entry(
            SCRIPT_PERTURBATION, PERTURBATION_SPEC, trailing=2,
        )]
        settings, meta = parse_forge_script_info(payload)
        self.assertEqual(len(settings), len(PERTURBATION_SPEC))
        self.assertEqual(meta['ignored_trailing_args'], 2)

    def test_short_positional_payload_is_rejected(self):
        payload = [self._entry(SCRIPT_PERTURBATION, PERTURBATION_SPEC)]
        payload[0]['args'].pop()
        with self.assertRaisesRegex(ValueError, '인자 수'):
            parse_forge_script_info(payload)

    def test_requires_at_least_one_supported_script(self):
        with self.assertRaisesRegex(ValueError, 'Anima 스크립트'):
            parse_forge_script_info([{'name': 'unrelated', 'args': []}])


# ── 2) 설치된 확장과 교차검증 ────────────────────────────────────────────────
def _parse_ui_return(source_path: str) -> list:
    """스크립트의 `def ui(self, is_img2img)` 안 마지막 `return [...]`에서 이름 순서 추출."""
    with open(source_path, 'r', encoding='utf-8') as fh:
        tree = ast.parse(fh.read())
    for node in ast.walk(tree):
        if not (isinstance(node, ast.FunctionDef) and node.name == 'ui'):
            continue
        for stmt in ast.walk(node):
            if isinstance(stmt, ast.Return) and isinstance(stmt.value, (ast.List, ast.Tuple)):
                names = [el.id for el in stmt.value.elts if isinstance(el, ast.Name)]
                if names:
                    return names
    return []


@requires_extension
class TestAgainstInstalledExtension(unittest.TestCase):
    """확장을 업데이트해 인자 순서가 바뀌면 여기서 먼저 깨진다."""

    def test_live_order_matches_spec(self):
        for title, (prefix, spec, _exp, filename) in _PREFIXES.items():
            path = os.path.join(_EXT_DIR, filename)
            if not os.path.exists(path):
                self.skipTest(f"{filename} 없음")
            with self.subTest(script=title):
                live = _parse_ui_return(path)
                ours = [key[len(prefix):] for key, _k, _d, _e in spec]
                self.assertEqual(
                    live, ours,
                    f"\n{filename}의 ui() 인자 순서가 core/anima_guidance.py 스펙과 다릅니다."
                    f"\n확장: {live}\n앱  : {ours}")

    def test_titles_match(self):
        import re
        expected_titles = {
            'anima_safe_pag.py': SCRIPT_PERTURBATION,
            'anima_skimmed_cfg.py': SCRIPT_SKIMMED_CFG,
            'anima_detail_daemon.py': SCRIPT_DETAIL_DAEMON,
        }
        for filename, title in expected_titles.items():
            path = os.path.join(_EXT_DIR, filename)
            if not os.path.exists(path):
                self.skipTest(f"{filename} 없음")
            with self.subTest(script=filename):
                with open(path, 'r', encoding='utf-8') as fh:
                    src = fh.read()
                match = re.search(r'def title\(self\):\s*\n\s*return\s+"([^"]+)"', src)
                self.assertIsNotNone(match, f"{filename}에서 title() 파싱 실패")
                self.assertEqual(match.group(1), title)


if __name__ == '__main__':
    unittest.main()
