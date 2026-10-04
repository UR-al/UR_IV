"""프리셋 값을 지금 백엔드 콤보 항목에 맞추기(core.preset_choice)와 TE 선택 맞추기(audit #154 후속)."""
import ast
import unittest

from core import extra_samplers as xsa
from core import extra_schedulers as xs
from core.comfy_workflow_compiler import _FORGE_SAMPLER_ALIASES, _FORGE_SCHEDULER_ALIASES
from core.main_module_choices import resolve_te_selection
from core.preset_choice import (
    FORGE_SCHEDULER_LABELS,
    fold_choice,
    match_choice,
    match_sampler,
    match_scheduler,
    sampler_key,
    scheduler_key,
)
from tests._sam_extra_ext import EXT_ROOT, REQUIRED, requires_extension

COMFY_SAMPLERS = ['euler', 'euler_ancestral', 'dpmpp_2m', 'dpmpp_2m_sde', 'uni_pc', 'er_sde']
FORGE_SAMPLERS = ['Euler', 'Euler a', 'DPM++ 2M', 'DPM++ 2M SDE', 'UniPC', 'ER SDE']


class MatchChoiceTests(unittest.TestCase):
    def test_exact_then_case_and_separator_insensitive(self):
        self.assertEqual(match_choice('Euler', ['euler', 'Euler']), 1, '정확히 같은 항목이 먼저')
        self.assertEqual(match_choice('SGM Uniform', ['normal', 'sgm_uniform']), 1)
        self.assertEqual(fold_choice('  DPM++_2M   Karras '), 'dpm++ 2m karras')

    def test_empty_value_or_list_never_matches(self):
        self.assertEqual(match_choice('', ['a']), -1)
        self.assertEqual(match_choice('a', []), -1)
        self.assertEqual(match_choice(None, None), -1)

    def test_no_alias_without_key(self):
        self.assertEqual(match_choice('Euler a', COMFY_SAMPLERS), -1)


class SamplerAliasTests(unittest.TestCase):
    def test_forge_labels_pick_comfy_names_and_back(self):
        for forge, comfy in zip(FORGE_SAMPLERS, COMFY_SAMPLERS):
            with self.subTest(forge=forge):
                self.assertEqual(COMFY_SAMPLERS[match_sampler(forge, COMFY_SAMPLERS)], comfy)
                self.assertEqual(FORGE_SAMPLERS[match_sampler(comfy, FORGE_SAMPLERS)], forge)

    def test_combined_label_is_not_split_so_the_scheduler_is_not_lost(self):
        self.assertEqual(match_sampler('DPM++ 2M Karras', COMFY_SAMPLERS), -1)

    def test_unknown_sampler_is_a_miss(self):
        self.assertEqual(match_sampler('Restart', COMFY_SAMPLERS), -1)

    def test_placeholders_are_only_matched_exactly(self):
        items = ['Use same sampler', 'euler']
        self.assertEqual(match_sampler('Use same sampler', items), 0)
        self.assertEqual(sampler_key('Use same sampler'), '')
        self.assertEqual(match_sampler('Restart', ['Use same sampler']), -1)

    def test_keys_agree_with_the_comfy_compiler_tables(self):
        """ComfyUI 생성이 Forge 표기를 바꾸는 표와 같은 결과 — 두 곳이 갈라지지 않는다."""
        for forge, comfy in _FORGE_SAMPLER_ALIASES.items():
            self.assertEqual(sampler_key(forge), sampler_key(comfy), forge)
        for forge, comfy in _FORGE_SCHEDULER_ALIASES.items():
            self.assertEqual(scheduler_key(forge), scheduler_key(comfy), forge)


class SchedulerAliasTests(unittest.TestCase):
    def test_forge_schedulers_pick_comfy_names(self):
        comfy = ['normal', 'karras', 'exponential', 'sgm_uniform', 'beta57']
        self.assertEqual(match_scheduler('Karras', comfy), 1)
        self.assertEqual(match_scheduler('SGM Uniform', comfy), 3)
        self.assertEqual(match_scheduler('Beta57 (RES4LYF)', comfy), 4)
        self.assertEqual(match_scheduler('karras', ['Automatic', 'Karras']), 1)

    def test_automatic_is_not_silently_turned_into_normal(self):
        self.assertEqual(match_scheduler('Automatic', ['normal', 'karras']), -1)

    def test_use_same_scheduler_is_not_an_alias_target(self):
        self.assertEqual(match_scheduler('Automatic', ['Use same scheduler', 'normal']), -1)
        self.assertEqual(match_scheduler('Use same scheduler', ['Use same scheduler', 'normal']), 0)


# Forge 스케줄러 콤보는 이름(API name), infotext 'Schedule type' 은 표기(label) — sam-extra Extra Schedulers 포함
FORGE_SCHEDULER_NAMES = ['automatic', 'karras', 'exponential', 'sgm_uniform', 'simple', 'beta', 'flow_match',
                         'cosine', 'cosine_exponential', 'phi', 'laplace', 'karras_dynamic', 'custom']


class ForgeScheduleTypeLabelTests(unittest.TestCase):
    """'이 설정으로 생성'(X3a): Forge infotext 의 표기가 이름 목록에서 골라진다."""

    def test_labels_pick_the_name_valued_items(self):
        for label, name in (('Karras', 'karras'), ('SGM Uniform', 'sgm_uniform'), ('SGMUniform', 'sgm_uniform'),
                            ('Laplace', 'laplace'), ('CosineExponential blend', 'cosine_exponential'),
                            ('cosine-exponential blend', 'cosine_exponential'), ('Karras Dynamic', 'karras_dynamic'),
                            ('FlowMatchEulerDiscrete', 'flow_match'), ('Phi', 'phi'), ('custom', 'custom')):
            with self.subTest(label=label):
                self.assertEqual(FORGE_SCHEDULER_NAMES[match_scheduler(label, FORGE_SCHEDULER_NAMES)], name)

    def test_comfy_names_on_forge_lists(self):
        self.assertEqual(FORGE_SAMPLERS[match_sampler('euler_ancestral', FORGE_SAMPLERS)], 'Euler a')
        self.assertEqual(FORGE_SAMPLERS[match_sampler('uni_pc', FORGE_SAMPLERS)], 'UniPC')
        for name in ('karras', 'sgm_uniform', 'laplace', 'cosine_exponential'):
            self.assertEqual(FORGE_SCHEDULER_NAMES[match_scheduler(name, FORGE_SCHEDULER_NAMES)], name)

    def test_unknown_values_miss(self):
        self.assertEqual(match_scheduler('Align Your Steps 32', FORGE_SCHEDULER_NAMES), -1)
        self.assertEqual(match_scheduler('CosineExponential', FORGE_SCHEDULER_NAMES), -1)
        self.assertEqual(match_sampler('Restart', FORGE_SAMPLERS), -1)

    def test_sam_extra_v033_names_and_labels_pick_each_other(self):
        """sam-extra v0.33.0(계획 C1): 새 샘플러 표기 ↔ ComfyUI 이름, 새 스케줄러 표기 ↔ 이름 — 프리셋·'이 설정으로 생성'."""
        comfy_samplers = [*COMFY_SAMPLERS, *xsa.COMFY_NAMES.values()]
        forge_samplers = [*FORGE_SAMPLERS, *xsa.LABELS]
        for label, name in (('ER SDE (Tunable)', 'er_sde_tunable'), ('UniPC bh2', 'uni_pc_bh2'),
                            ('IPNDM_V', 'ipndm_v'), ('IPNDM', 'ipndm'), ('DPM++ 3M (flow ODE)', 'dpmpp_3m_flow_ode'),
                            *xsa.COMFY_NAMES.items()):
            with self.subTest(sampler=label):
                self.assertEqual(comfy_samplers[match_sampler(label, comfy_samplers)], name)
                self.assertEqual(forge_samplers[match_sampler(name, forge_samplers)], label)
        names = [*FORGE_SCHEDULER_NAMES, 'react_cosinusoidal_dynsf', 'flow_cosmos_rho7', 'flow_cosmos_dynamic']
        for name in xs.SCHEDULER_NAMES:
            for text in (name, xs.SCHEDULER_LABELS[name], *((xs.README_LABELS[name],) if name in xs.README_LABELS else ())):
                with self.subTest(scheduler=text):
                    self.assertEqual(names[match_scheduler(text, names)], name)
        self.assertEqual(names[match_scheduler('Flow Cosmos rho7', names)], 'flow_cosmos_rho7')
        self.assertEqual(match_scheduler('Flow Cosmos', names), -1)

    def test_the_label_table_only_holds_labels_that_do_not_fold_to_their_name(self):
        for label, name in FORGE_SCHEDULER_LABELS.items():
            with self.subTest(label=label):
                self.assertEqual(label, label.strip().casefold())
                self.assertNotEqual(fold_choice(label), fold_choice(name))
                self.assertEqual(scheduler_key(label), scheduler_key(name))


def _call_rows(path, func_name):
    """``func_name(name, label, …, aliases=…)`` 호출에서 (이름, 표기, 별칭들) — 상수 dict 첨자와 모듈 문자열 상수
    (``SchedulerSpec("flow_cosmos_rho7", LABEL_FLOW_COSMOS_RHO7, …)``)도 푼다."""
    tree = ast.parse(path.read_text(encoding='utf-8'))
    tables, constants = {}, {}
    for node in tree.body:
        if isinstance(node, (ast.Assign, ast.AnnAssign)):
            target = node.targets[0] if isinstance(node, ast.Assign) else node.target
            if isinstance(target, ast.Name) and isinstance(node.value, ast.Constant)                     and isinstance(node.value.value, str):
                constants[target.id] = node.value.value
    for node in ast.walk(tree):
        if isinstance(node, (ast.Assign, ast.AnnAssign)) and isinstance(node.value, ast.Dict):
            target = node.targets[0] if isinstance(node, ast.Assign) else node.target
            if isinstance(target, ast.Name):
                try:
                    tables[target.id] = ast.literal_eval(node.value)
                except ValueError:
                    pass

    def value(expr):
        if isinstance(expr, ast.Subscript) and isinstance(expr.value, ast.Name):
            return tables[expr.value.id][ast.literal_eval(expr.slice)]
        if isinstance(expr, ast.Name) and expr.id in constants:
            return constants[expr.id]
        return ast.literal_eval(expr)

    def readable(expr):
        return isinstance(expr, ast.Constant) or (isinstance(expr, ast.Name) and expr.id in constants)

    rows = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == func_name:
            if len(node.args) < 2 or not all(readable(a) for a in node.args[:2]):
                continue
            aliases = []
            for keyword in node.keywords:
                if keyword.arg == 'aliases':
                    aliases = [value(item) for item in keyword.value.elts]
            rows.append((value(node.args[0]), value(node.args[1]), tuple(aliases)))
    return rows


@requires_extension
class InstalledForgeScheduleLabelTests(unittest.TestCase):
    """설치된 Forge(modules/sd_schedulers.py)와 sam-extra(Extra Schedulers registry)의 표기·별칭이 모두 그 이름으로
    골라진다 — 새 표기가 접어도 이름과 다르면 FORGE_SCHEDULER_LABELS 에 더하라고 실패한다."""

    def test_every_installed_label_and_alias_resolves_to_its_name(self):
        forge = EXT_ROOT.parents[1] / 'modules' / 'sd_schedulers.py'
        if not forge.is_file():
            if REQUIRED:
                self.fail(f'Forge 스케줄러 소스가 없다: {forge}')
            self.skipTest(f'Forge 스케줄러 소스가 없다: {forge}')
        registry = EXT_ROOT / 'sam3ext' / 'extra_schedulers' / 'registry.py'
        rows = _call_rows(forge, 'Scheduler') + _call_rows(registry, 'SchedulerSpec')
        names = [name for name, _label, _aliases in rows]
        self.assertIn('karras', names)
        self.assertIn('cosine_exponential', names)
        # 표기가 상수 이름인 행(v0.33.0 의 셋)도 읽힌다 — 확장 registry 와 앱 표가 같은 아홉
        self.assertEqual([row for row in rows if row[0] in xs.SCHEDULER_NAMES and row[1] in xs.SCHEDULER_LABELS.values()],
                         [row for row in rows if row[0] in xs.SCHEDULER_NAMES])
        self.assertEqual({name: label for name, label, _aliases in rows if name in xs.SCHEDULER_NAMES},
                         dict(xs.SCHEDULER_LABELS))
        problems = []
        for name, label, aliases in rows:
            for text in (label, *aliases):
                index = match_scheduler(text, names)
                if index < 0 or names[index] != name:
                    problems.append(f"{text!r} → {names[index] if index >= 0 else None!r} (이름 {name!r})")
        self.assertEqual(problems, [], '\nFORGE_SCHEDULER_LABELS 에 더할 표기:\n' + '\n'.join(problems))


class ResolveTeSelectionTests(unittest.TestCase):
    def test_same_file_with_other_folder_or_hash_uses_the_choice_spelling(self):
        text, dropped = resolve_te_selection(
            'qwen_3_06b.safetensors [abc123], CLIP_L.safetensors, gone.safetensors',
            ['text_encoders/qwen_3_06b.safetensors', 'clip_l.safetensors'])
        self.assertEqual(text, 'text_encoders/qwen_3_06b.safetensors, clip_l.safetensors')
        self.assertEqual(dropped, ['gone.safetensors'])

    def test_duplicates_collapse_and_empty_choices_drop_everything(self):
        self.assertEqual(resolve_te_selection('a.safetensors, a.safetensors', ['a.safetensors']),
                         ('a.safetensors', []))
        self.assertEqual(resolve_te_selection('a.safetensors', []), ('', ['a.safetensors']))
        self.assertEqual(resolve_te_selection('', ['a.safetensors']), ('', []))


if __name__ == '__main__':
    unittest.main()
