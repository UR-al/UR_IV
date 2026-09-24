"""생성 프리셋 — 저장·미리보기·불러오기·공유가 같은 PRESET_KEYS 를 쓰고, 모델은 match_checkpoint 로
맞추며 못 맞추면 경고한다(audit #154)."""
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from core.generation_presets import (
    BUNDLE_FORMAT,
    PRESET_KEYS,
    PresetError,
    delete_preset,
    export_bundle,
    import_bundle,
    list_presets,
    parse_bundle,
    preset_from_settings,
    read_preset,
    write_preset,
)
from ui.generation_settings_apply import apply_generation_settings, apply_prompt_settings


FULL_SETTINGS = {
    # 프리셋 키
    'character': 'hatsune miku', 'main_prompt': '1girl, solo', 'negative_prompt': 'lowres',
    'model': 'animagine.safetensors [abc123]', 'vae_main': 'sdxl_vae.safetensors', 'te_main': '',
    'sampler': 'Euler a', 'scheduler': 'Karras', 'steps': '28', 'cfg': '5.5', 'shift': '0',
    'seed': '-1', 'width': '832', 'height': '1216',
    'hires_enabled': True, 'hires_upscaler': 'R-ESRGAN 4x+', 'hires_checkpoint': 'Use same checkpoint',
    'adetailer_enabled': True, 'adetailer_slot1': {'model': 'face_yolov8n.pt'},
    'sam3_enabled': False, 'anima_guidance_settings': {'cfg': '4'},
    # 프리셋이 아닌 키(백엔드·단축키·테마 등)
    'webui_url': 'http://127.0.0.1:7860', 'shortcuts': {'generate': 'Ctrl+Enter'}, 'theme': '다크',
    'font_size': 10.5, 'backend_type': 'webui', 'negpip_enabled': False,
}


class PresetStoreTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.dir = Path(self._tmp.name)

    def test_save_keeps_only_generation_keys(self):
        name = write_preset('My Preset', FULL_SETTINGS, self.dir)
        saved = json.loads((self.dir / f'{name}.json').read_text(encoding='utf-8'))
        self.assertTrue(set(saved) <= set(PRESET_KEYS))
        for leaked in ('webui_url', 'shortcuts', 'theme', 'font_size', 'backend_type', 'negpip_enabled'):
            self.assertNotIn(leaked, saved)
        self.assertEqual(saved['model'], 'animagine.safetensors [abc123]')

    def test_old_full_snapshot_presets_read_as_preset_keys(self):
        (self.dir / 'legacy.json').write_text(json.dumps(FULL_SETTINGS), encoding='utf-8')
        preset = read_preset('legacy', self.dir)
        self.assertEqual(preset, preset_from_settings(FULL_SETTINGS))
        self.assertNotIn('webui_url', preset)

    def test_names_are_sanitized_for_every_operation(self):
        name = write_preset('../evil:name', {'steps': '20'}, self.dir)
        self.assertEqual(name, '..evilname')
        self.assertEqual(list_presets(self.dir), ['..evilname'])
        self.assertIsNotNone(read_preset('../evil:name', self.dir))
        self.assertTrue(delete_preset('../evil:name', self.dir))
        self.assertFalse(delete_preset('../evil:name', self.dir))
        with self.assertRaises(PresetError):
            write_preset('  ', {'steps': '20'}, self.dir)
        with self.assertRaises(PresetError):
            write_preset('empty', {'theme': 'x'}, self.dir)

    def test_bundle_round_trip_and_merge_modes(self):
        write_preset('a', {'steps': '20'}, self.dir)
        write_preset('b', {'steps': '30'}, self.dir)
        bundle = export_bundle(self.dir)
        self.assertEqual(bundle['format'], BUNDLE_FORMAT)
        self.assertEqual(set(bundle['presets']), {'a', 'b'})

        other = Path(self._tmp.name) / 'other'
        other.mkdir()
        write_preset('a', {'steps': '99'}, other)
        kept = import_bundle(bundle, overwrite=False, directory=other)
        self.assertEqual((kept['added'], kept['skipped']), (['b'], ['a']))
        self.assertEqual(read_preset('a', other)['steps'], '99')
        replaced = import_bundle(bundle, overwrite=True, directory=other)
        self.assertEqual(sorted(replaced['replaced']), ['a', 'b'])
        self.assertEqual(read_preset('a', other)['steps'], '20')

    def _fresh_dir(self, name):
        path = self.dir / name
        path.mkdir()
        return path

    def _fs_same_name(self, a, b):
        """이 파일 시스템이 두 프리셋 이름을 같은 파일로 보는지 — 기대값을 이름 규칙이 아니라 실제
        동작에서 얻는다(NTFS 의 대문자 표는 파이썬 upper() 와 다르다)."""
        folder = self._fresh_dir(f'probe_{len(list(self.dir.iterdir()))}')
        (folder / f'{a}.json').write_text('{}', encoding='utf-8')
        return (folder / f'{b}.json').is_file()

    def _require_case_insensitive_fs(self):
        if not self._fs_same_name('Alpha', 'alpha'):
            self.skipTest('대소문자를 구분하는 파일 시스템 — Windows(NTFS) 전용 기대값')

    def test_import_bundle_case_collisions_inside_one_file_keep_the_first_entry(self):
        """파일 안에서 대소문자만 다른 이름 — 예전엔 둘 다 '추가'로 세고 Windows 에선 뒤 항목이 앞 파일을
        덮었다(overwrite=False 여도). 이제 앞 항목만 쓰고 뒤 항목은 duplicate 로 알린다."""
        self._require_case_insensitive_fs()
        presets = {'Alpha': {'steps': '20'}, 'alpha': {'steps': '30'}}
        for label, data in (('bundle', {'format': BUNDLE_FORMAT, 'version': 1, 'presets': presets}),
                            ('legacy', presets)):
            for overwrite in (False, True):
                with self.subTest(form=label, overwrite=overwrite):
                    folder = self._fresh_dir(f'{label}_{overwrite}')
                    result = import_bundle(data, overwrite=overwrite, directory=folder)
                    self.assertEqual(result['added'], ['Alpha'])
                    self.assertEqual(result['duplicate'], ['alpha'])
                    self.assertEqual((result['replaced'], result['skipped']), ([], []))
                    self.assertEqual(list_presets(folder), ['Alpha'])
                    self.assertEqual(read_preset('Alpha', folder)['steps'], '20')

    def test_names_that_collide_after_sanitizing_are_reported_too(self):
        self._require_case_insensitive_fs()
        folder = self._fresh_dir('sanitized')
        result = import_bundle({'Beta.': {'steps': '1'}, 'BETA:': {'steps': '2'}, 'Beta:': {'steps': '3'}},
                               overwrite=False, directory=folder)
        self.assertEqual(result['added'], ['Beta'])
        self.assertEqual(result['duplicate'], ['BETA', 'Beta'])
        self.assertEqual(read_preset('Beta', folder)['steps'], '1')
        self.assertEqual(parse_bundle({'Beta.': {'steps': '1'}, 'Beta:': {'steps': '3'}}),
                         {'Beta': {'steps': '1'}}, '정리한 이름이 같으면 앞 항목')

    def test_existing_preset_with_bundle_case_variants_is_replaced_once(self):
        self._require_case_insensitive_fs()
        folder = self._fresh_dir('mixed')
        write_preset('Alpha', {'steps': '99'}, folder)
        result = import_bundle({'alpha': {'steps': '1'}, 'ALPHA': {'steps': '2'}}, overwrite=True, directory=folder)
        self.assertEqual(result['replaced'], ['alpha'])
        self.assertEqual(result['duplicate'], ['ALPHA'])
        self.assertEqual(read_preset('alpha', folder)['steps'], '1')
        kept = import_bundle({'ALPHA': {'steps': '5'}}, overwrite=False, directory=folder)
        self.assertEqual((kept['skipped'], kept['added']), (['ALPHA'], []))

    #: 파이썬 upper() 는 같게 만들지만 NTFS 대문자 표와는 다를 수 있는 짝 — ß→SS·ŉ→ʼN(글자 수가
    #: 바뀜), ς/σ·ı/i(되돌아오지 않는 짝), 조지아 Mkhedruli→Mtavruli(Unicode 11), 보조 평면(Deseret),
    #: 그리고 실제로 겹치는 é/É. 이 PC 의 NTFS 실측: é/É 만 같은 파일.
    UNICODE_PAIRS = (('Straße', 'STRASSE'), ('ŉote', 'ʼNOTE'), ('σigma', 'ςigma'), ('fil', 'fıl'),
                     ('აx', 'Აx'), ('\U00010428x', '\U00010400x'), ('émile', 'Émile'))

    def test_bundle_collisions_follow_the_file_system_not_python_upper(self):
        """파일 안의 두 이름은 파일 시스템이 같은 파일로 볼 때만 duplicate — 아니면 둘 다 가져온다."""
        for index, (first, second) in enumerate(self.UNICODE_PAIRS):
            with self.subTest(first=ascii(first), second=ascii(second)):
                same = self._fs_same_name(first, second)
                for overwrite in (False, True):
                    folder = self._fresh_dir(f'pair_{index}_{overwrite}')
                    result = import_bundle({first: {'steps': '1'}, second: {'steps': '2'}},
                                           overwrite=overwrite, directory=folder)
                    if same:
                        self.assertEqual((result['added'], result['duplicate']), ([first], [second]))
                        self.assertEqual(read_preset(second, folder)['steps'], '1', '앞 항목을 덮지 않는다')
                    else:
                        self.assertEqual((result['added'], result['duplicate']), ([first, second], []))
                        self.assertEqual(read_preset(first, folder)['steps'], '1')
                        self.assertEqual(read_preset(second, folder)['steps'], '2')
                    self.assertEqual((result['replaced'], result['skipped']), ([], []))

    def test_existing_file_report_matches_what_was_written(self):
        """Codex R6#3 — 'STRASSE' 가 있을 때 'Straße': upper() 키로는 overwrite=True 에서 '교체'라 보고하고
        실제론 새 파일을 만들었고(STRASSE 는 그대로), overwrite=False 에선 겹치지 않는 프리셋을 버렸다.
        이제 보고는 디스크에서 실제로 일어난 일과 같다."""
        for index, (existing, incoming) in enumerate(self.UNICODE_PAIRS):
            with self.subTest(existing=ascii(existing), incoming=ascii(incoming)):
                same = self._fs_same_name(existing, incoming)
                for overwrite in (False, True):
                    folder = self._fresh_dir(f'existing_{index}_{overwrite}')
                    write_preset(existing, {'steps': '99'}, folder)
                    result = import_bundle({incoming: {'steps': '1'}}, overwrite=overwrite, directory=folder)
                    if not same:
                        self.assertEqual(result['added'], [incoming])
                        self.assertEqual(read_preset(incoming, folder)['steps'], '1')
                        self.assertEqual(read_preset(existing, folder)['steps'], '99', '기존 파일은 그대로')
                        self.assertEqual(len(list_presets(folder)), 2)
                    elif overwrite:
                        self.assertEqual(result['replaced'], [incoming])
                        self.assertEqual(read_preset(existing, folder)['steps'], '1', '보고대로 교체됐다')
                        self.assertEqual(len(list_presets(folder)), 1)
                    else:
                        self.assertEqual(result['skipped'], [incoming])
                        self.assertEqual(read_preset(existing, folder)['steps'], '99')

    def test_skipped_existing_name_still_marks_later_case_variants_as_duplicates(self):
        self._require_case_insensitive_fs()
        folder = self._fresh_dir('skip_then_dup')
        write_preset('Alpha', {'steps': '99'}, folder)
        result = import_bundle({'alpha': {'steps': '1'}, 'ALPHA': {'steps': '2'}}, overwrite=False, directory=folder)
        self.assertEqual((result['skipped'], result['duplicate']), (['alpha'], ['ALPHA']))
        self.assertEqual(read_preset('Alpha', folder)['steps'], '99')

    def test_parse_bundle_accepts_plain_mapping_and_rejects_garbage(self):
        self.assertEqual(parse_bundle({'x': {'steps': '5', 'theme': 't'}}), {'x': {'steps': '5'}})
        for bad in ([], {'x': 'not a dict'}, {'format': BUNDLE_FORMAT, 'version': 99, 'presets': {}},
                    {'format': BUNDLE_FORMAT, 'version': 1, 'presets': []}):
            with self.assertRaises(PresetError):
                parse_bundle(bad)


class _Line:
    def __init__(self, text=''):
        self._text = text

    def setText(self, v):
        self._text = v

    def text(self):
        return self._text

    def setPlainText(self, v):
        self._text = v

    def toPlainText(self):
        return self._text


class _Combo:
    def __init__(self, items):
        self._items = list(items)
        self.index = -1
        self.fallback = None

    def findText(self, text):
        return self._items.index(text) if text in self._items else -1

    def setCurrentIndex(self, idx):
        self.index = idx

    def currentText(self):
        return self._items[self.index] if self.index >= 0 else ''

    def setText(self, text):   # ComboBoxProxy: 없으면 fallback 으로 기억
        if text in self._items:
            self.index = self._items.index(text)
        else:
            self.fallback = text


class _Check:
    def __init__(self):
        self.checked = None

    def setChecked(self, v):
        self.checked = bool(v)

    def isChecked(self):
        return bool(self.checked)


def _host(models):
    host = SimpleNamespace(
        model_combo=_Combo(models), vae_main_combo=_Combo(['Use checkpoint default']),
        te_main_input=_Line(), sampler_combo=_Combo(['Euler a']), scheduler_combo=_Combo(['Karras']),
        steps_input=_Line('25'), cfg_input=_Line('7'), shift_input=_Line('0'), seed_input=_Line('-1'),
        width_input=_Line('1024'), height_input=_Line('1024'), random_res_check=_Check(),
        hires_options_group=_Check(), upscaler_combo=_Combo(['Latent']),
        hires_steps_input=_Line(), hires_denoising_input=_Line(), hires_scale_input=_Line(), hires_cfg_input=_Line(),
        hires_checkpoint_combo=_Combo(['Use same checkpoint']), hires_sampler_combo=_Combo([]),
        hires_scheduler_combo=_Combo([]), hires_prompt_text=_Line(), hires_neg_prompt_text=_Line(),
        adetailer_group=_Check(), ad_slot1_group=_Check(), ad_slot2_group=_Check(),
        s1_widgets={}, s2_widgets={}, slot_calls=[],
        char_count_input=_Line(), character_input=_Line(), copyright_input=_Line(), artist_input=_Line(),
        prefix_prompt_text=_Line(), main_prompt_text=_Line('keep me'), suffix_prompt_text=_Line(),
        neg_prompt_text=_Line(), exclude_prompt_local_input=_Line(),
    )
    host._sync_slider = lambda widget: None
    host._set_slot_settings = lambda widgets, data: host.slot_calls.append(data)
    return host


class ApplyGenerationSettingsTests(unittest.TestCase):
    def test_model_saved_on_forge_matches_comfy_name_without_hash(self):
        host = _host(['other.safetensors', 'animagine.safetensors'])
        warnings = apply_generation_settings(host, {'model': 'animagine.safetensors [abc123]'}, only_present=True)
        self.assertEqual(host.model_combo.currentText(), 'animagine.safetensors')
        self.assertEqual(warnings, [])

    def test_unknown_model_and_upscaler_are_reported_not_silently_ignored(self):
        host = _host(['other.safetensors'])
        warnings = apply_generation_settings(
            host, {'model': 'missing.safetensors', 'hires_upscaler': 'R-ESRGAN 4x+'}, only_present=True)
        self.assertEqual(host.model_combo.index, -1)
        self.assertEqual(len(warnings), 2)
        self.assertIn('missing.safetensors', warnings[0])

    def test_preset_path_only_touches_present_keys(self):
        host = _host([])
        apply_generation_settings(host, {'steps': '30', 'hires_enabled': True,
                                         'adetailer_slot1': {'model': 'face.pt'}}, only_present=True)
        apply_prompt_settings(host, {'character': 'miku'}, only_present=True)
        self.assertEqual(host.steps_input.text(), '30')
        self.assertEqual(host.cfg_input.text(), '7', '프리셋에 없는 키는 그대로')
        self.assertTrue(host.hires_options_group.checked)
        self.assertIsNone(host.adetailer_group.checked)
        self.assertEqual(host.slot_calls, [{'model': 'face.pt'}])
        self.assertEqual(host.character_input.text(), 'miku')
        self.assertEqual(host.main_prompt_text.toPlainText(), 'keep me')

    def test_load_settings_path_fills_defaults_for_missing_keys(self):
        host = _host([])
        apply_generation_settings(host, {}, only_present=False)
        apply_prompt_settings(host, {}, only_present=False)
        self.assertEqual(host.steps_input.text(), '25')
        self.assertEqual(host.hires_denoising_input.text(), '0.4')
        self.assertFalse(host.hires_options_group.checked)
        self.assertEqual(host.main_prompt_text.toPlainText(), '')

    def test_vae_before_the_list_arrives_is_remembered_as_fallback(self):
        """시작 시 load_settings — VAE 목록이 아직 비어 있으면 fallback 으로 기억(addItems 때 복원)."""
        host = _host([])
        host.vae_main_combo = _Combo([])
        warnings = apply_generation_settings(host, {'vae_main': 'later.vae'}, only_present=False)
        self.assertEqual(host.vae_main_combo.fallback, 'later.vae')
        self.assertFalse([w for w in warnings if 'VAE' in w])

    def test_vae_missing_from_a_populated_list_keeps_selection_and_warns(self):
        host = _host([])
        warnings = apply_generation_settings(host, {'vae_main': 'forge_only.vae'}, only_present=True)
        self.assertIsNone(host.vae_main_combo.fallback, '찬 목록에 setText fallback 을 쓰지 않는다')
        self.assertEqual(len(warnings), 1)
        self.assertIn('forge_only.vae', warnings[0])

    def test_combos_with_no_list_are_reported_once(self):
        """연결 전 프리셋 — 목록이 빈 모델·Hires 콤보는 항목마다가 아니라 한 줄로 알린다."""
        host = _host([])
        host.upscaler_combo = _Combo([])
        warnings = apply_generation_settings(
            host, {'model': 'm.safetensors', 'hires_upscaler': 'Latent', 'hires_sampler': 'Euler'},
            only_present=True)
        self.assertEqual(len(warnings), 1)
        for label in ('모델', '업스케일러', 'Hires 샘플러'):
            self.assertIn(label, warnings[0])
        self.assertEqual(host.model_combo.index, -1)


class RealProxyPresetTests(unittest.TestCase):
    """실제 VueBridge + ComboBoxProxy — Vue 가 보는 값(push)과 생성이 쓰는 값(currentText)이 같아야 한다."""

    def setUp(self):
        from ui.vue_bridge import VueBridge
        from ui.widget_proxies import ComboBoxProxy, LineEditProxy

        self.bridge = VueBridge()
        self.pushed = []
        self.bridge.widgetValueChanged.connect(lambda wid, value: self.pushed.append((wid, value)))
        host = _host([])
        host.vue_bridge = self.bridge
        host._backend_connected = True
        host.sampler_combo = ComboBoxProxy(self.bridge, 'sampler_combo')
        host.scheduler_combo = ComboBoxProxy(self.bridge, 'scheduler_combo')
        host.vae_main_combo = ComboBoxProxy(self.bridge, 'vae_main_combo')
        host.hires_sampler_combo = ComboBoxProxy(self.bridge, 'hires_sampler_combo')
        host.hires_scheduler_combo = ComboBoxProxy(self.bridge, 'hires_scheduler_combo')
        host.te_main_input = LineEditProxy(self.bridge, 'te_main_input')
        # ComfyUI 에 연결된 상태 — 목록이 이미 찼다.
        host.sampler_combo.addItems(['dpmpp_2m', 'euler_ancestral', 'euler'])
        host.scheduler_combo.addItems(['normal', 'karras', 'sgm_uniform'])
        host.vae_main_combo.addItems(['Use checkpoint default', 'sdxl_vae.safetensors'])
        host.hires_sampler_combo.addItems(['Use same sampler', 'dpmpp_2m', 'euler_ancestral'])
        host.hires_scheduler_combo.addItems(['Use same scheduler', 'normal', 'karras'])
        self.host = host
        self.pushed.clear()

    def _vue(self, widget_id):
        values = [value for wid, value in self.pushed if wid == widget_id]
        return values[-1] if values else None

    def test_forge_sampler_and_scheduler_names_pick_the_comfy_equivalents(self):
        warnings = apply_generation_settings(
            self.host, {'sampler': 'Euler a', 'scheduler': 'Karras',
                        'hires_sampler': 'DPM++ 2M', 'hires_scheduler': 'Karras'}, only_present=True)
        self.assertEqual(warnings, [])
        self.assertEqual(self.host.sampler_combo.currentText(), 'euler_ancestral')
        self.assertEqual(self._vue('sampler_combo'), 'euler_ancestral')
        self.assertEqual(self.host.scheduler_combo.currentText(), 'karras')
        self.assertEqual(self._vue('scheduler_combo'), 'karras')
        self.assertEqual(self.host.hires_sampler_combo.currentText(), 'dpmpp_2m')
        self.assertEqual(self.host.hires_scheduler_combo.currentText(), 'karras')

    def test_unknown_sampler_keeps_the_selection_and_is_not_pushed_to_vue(self):
        warnings = apply_generation_settings(self.host, {'sampler': 'Restart', 'scheduler': 'Automatic'},
                                             only_present=True)
        self.assertEqual(self.host.sampler_combo.currentText(), 'dpmpp_2m', '생성은 이전 선택 그대로')
        self.assertIsNone(self._vue('sampler_combo'), 'Vue 도 이전 선택 그대로 — 없는 값을 보내지 않는다')
        self.assertIsNone(self._vue('scheduler_combo'))
        self.assertEqual(len(warnings), 2)
        self.assertIn("'Restart'", warnings[0])
        self.assertIn("'dpmpp_2m' 유지", warnings[0])

    def test_forge_only_vae_keeps_the_default_and_warns(self):
        warnings = apply_generation_settings(self.host, {'vae_main': 'forge_only.vae'}, only_present=True)
        self.assertEqual(self.host.vae_main_combo.currentText(), 'Use checkpoint default')
        self.assertIsNone(self._vue('vae_main_combo'))
        self.assertEqual(len(warnings), 1)
        self.assertIn('forge_only.vae', warnings[0])

    def test_vae_with_hash_or_folder_picks_the_same_file(self):
        warnings = apply_generation_settings(self.host, {'vae_main': 'VAE/sdxl_vae.safetensors [0f1e2d3c4b]'},
                                             only_present=True)
        self.assertEqual(warnings, [])
        self.assertEqual(self.host.vae_main_combo.currentText(), 'sdxl_vae.safetensors')
        self.assertEqual(self._vue('vae_main_combo'), 'sdxl_vae.safetensors')

    def test_hires_sampler_miss_is_reported(self):
        warnings = apply_generation_settings(self.host, {'hires_sampler': 'Restart'}, only_present=True)
        self.assertEqual(self.host.hires_sampler_combo.currentText(), 'Use same sampler')
        self.assertEqual(len(warnings), 1)
        self.assertIn('Hires 샘플러', warnings[0])

    def test_empty_list_at_startup_still_uses_the_fallback(self):
        """시작 시 load_settings — 목록 전이면 fallback 으로 기억하고 Vue·currentText 가 같은 값을 본다."""
        from ui.widget_proxies import ComboBoxProxy

        self.host.sampler_combo = ComboBoxProxy(self.bridge, 'sampler_combo')
        apply_generation_settings(self.host, {'sampler': 'ER SDE'}, only_present=True)
        self.assertEqual(self.host.sampler_combo.currentText(), 'ER SDE')
        self.assertEqual(self._vue('sampler_combo'), 'ER SDE')
        self.host.sampler_combo.addItems(['Euler', 'ER SDE'])
        self.assertEqual(self.host.sampler_combo.currentIndex(), 1)

    def test_preset_te_is_filtered_by_the_current_choices(self):
        self.bridge.pushWidgetProperty('te_main_input', 'items',
                                       ['qwen/qwen_3_06b.safetensors', 'clip_l.safetensors'])
        warnings = apply_generation_settings(
            self.host, {'te_main': 'qwen_3_06b.safetensors, forge_only_te.safetensors, clip_l.safetensors'},
            only_present=True)
        self.assertEqual(self.host.te_main_input.text(), 'qwen/qwen_3_06b.safetensors, clip_l.safetensors')
        self.assertEqual(len(warnings), 1)
        self.assertIn('forge_only_te.safetensors', warnings[0])

    def test_te_is_kept_when_the_choices_are_not_known_yet(self):
        """TE 선택지를 아직 모르거나(연결 전) 연결이 끊겨 비었으면 그대로 둔다 — 연결 때 다시 거른다."""
        warnings = apply_generation_settings(self.host, {'te_main': 'a.safetensors'}, only_present=True)
        self.assertEqual(self.host.te_main_input.text(), 'a.safetensors')
        self.bridge.pushWidgetProperty('te_main_input', 'items', [])
        self.host._backend_connected = False
        warnings += apply_generation_settings(self.host, {'te_main': 'b.safetensors'}, only_present=True)
        self.assertEqual(self.host.te_main_input.text(), 'b.safetensors')
        self.assertEqual(warnings, [])

    def test_connected_backend_without_te_files_drops_the_preset_te(self):
        self.bridge.pushWidgetProperty('te_main_input', 'items', [])
        warnings = apply_generation_settings(self.host, {'te_main': 'a.safetensors'}, only_present=True)
        self.assertEqual(self.host.te_main_input.text(), '')
        self.assertEqual(len(warnings), 1)

    def test_last_widget_property_returns_the_last_push_or_default(self):
        self.assertIsNone(self.bridge.last_widget_property('te_main_input', 'items'))
        self.assertEqual(self.bridge.last_widget_property('te_main_input', 'items', []), [])
        self.bridge.pushWidgetProperty('te_main_input', 'items', ['x'])
        self.bridge.pushWidgetProperty('te_main_input', 'items', ['y', 'z'])
        self.assertEqual(self.bridge.last_widget_property('te_main_input', 'items'), ['y', 'z'])


class SlotAndSam3ComboTests(unittest.TestCase):
    """ADetailer 슬롯·SAM3 체크포인트 — 같은 콤보 규칙(찬 목록엔 setText 금지, 못 고르면 경고)."""

    def setUp(self):
        from ui.generator_settings import SettingsMixin
        from ui.vue_bridge import VueBridge
        from ui.widget_proxies import ComboBoxProxy

        self.mixin = SettingsMixin()
        self.bridge = VueBridge()
        self.pushed = []
        self.bridge.widgetValueChanged.connect(lambda wid, value: self.pushed.append((wid, value)))
        self.ComboBoxProxy = ComboBoxProxy

    def _slot_widgets(self, populated=True):
        from collections import defaultdict

        combos = {name: self.ComboBoxProxy(self.bridge, f'_ad_s1_{name}')
                  for name in ('model', 'checkpoint_combo', 'vae_combo', 'sampler_combo', 'scheduler_combo')}
        if populated:
            combos['model'].addItems(['face_yolov8n.pt', 'hand_yolov8n.pt'])
            combos['checkpoint_combo'].addItems(['Use same checkpoint', 'anima.safetensors'])
            combos['vae_combo'].addItems(['Use same VAE', 'sdxl_vae.safetensors'])
            combos['sampler_combo'].addItems(['Use same sampler', 'dpmpp_2m', 'euler_ancestral'])
            combos['scheduler_combo'].addItems(['normal', 'karras'])
        widgets = defaultdict(_Line)
        for key in ('use_inpaint_size_check', 'use_steps_check', 'use_cfg_check', 'use_checkpoint_check',
                    'use_vae_check', 'use_sampler_check'):
            widgets[key] = _Check()
        widgets.update(combos)
        self.pushed.clear()
        return widgets

    def test_slot_combos_match_across_backends_and_report_misses(self):
        widgets = self._slot_widgets()
        misses = self.mixin._set_slot_settings(widgets, {
            'model': 'missing_detector.pt', 'checkpoint': 'anima.safetensors [4a458d26b2]',
            'vae': 'Use same VAE', 'sampler': 'Euler a', 'scheduler': 'Beta57'})
        self.assertEqual(widgets['checkpoint_combo'].currentText(), 'anima.safetensors')
        self.assertEqual(widgets['sampler_combo'].currentText(), 'euler_ancestral')
        self.assertEqual(widgets['model'].currentText(), 'face_yolov8n.pt')
        self.assertNotIn(('_ad_s1_model', 'missing_detector.pt'), self.pushed)
        self.assertEqual(len(misses), 2)
        self.assertIn('missing_detector.pt', misses[0])
        self.assertIn('Beta57', misses[1])

    def test_slot_combos_without_a_list_are_deferred_not_dropped(self):
        """목록 전(시작 시 load_settings·연결 전 프리셋) — 다섯 콤보 모두 fallback 으로 미룬다(경고 없음)."""
        widgets = self._slot_widgets(populated=False)
        misses = self.mixin._set_slot_settings(widgets, {
            'checkpoint': 'a.safetensors', 'vae': 'v.safetensors', 'sampler': 'Euler', 'scheduler': 'Karras'})
        self.assertEqual(misses, [])
        self.assertEqual(widgets['model'].currentText(), 'face_yolov8n.pt', '검출 모델은 fallback 으로 미룬다')
        for key, value in (('checkpoint_combo', 'a.safetensors'), ('vae_combo', 'v.safetensors'),
                           ('sampler_combo', 'Euler'), ('scheduler_combo', 'Karras')):
            self.assertEqual(widgets[key].currentText(), value, key)
            self.assertIn((f'_ad_s1_{key}', value), self.pushed, 'Vue 도 같은 값을 본다')

    def test_preset_before_connect_does_not_report_slot_combos_as_unavailable(self):
        host = _host([])
        host.s1_widgets = self._slot_widgets(populated=False)
        host._set_slot_settings = self.mixin._set_slot_settings
        warnings = apply_generation_settings(
            host, {'adetailer_slot1': {'checkpoint': 'a.safetensors', 'sampler': 'Euler'}}, only_present=True)
        self.assertEqual(warnings, [])
        self.assertEqual(host.s1_widgets['checkpoint_combo'].currentText(), 'a.safetensors')

    def test_startup_save_before_connect_keeps_slot_values_and_connect_restores_them(self):
        """시작 load_settings → 게이트의 연결 전 save_settings → 연결(비우고 채움 → restore) 순서.

        예전엔 슬롯 콤보를 미루지 않아 연결 전 저장이 ''를 썼고, 연결 뒤 'Use same checkpoint'·
        'Use same VAE'·'Use same sampler'·첫 스케줄러로 돌아간 채 종료 저장으로 굳었다."""
        from ui.combo_restore import restore_backend_combos, snapshot_backend_combos

        saved = {'use_checkpoint': True, 'checkpoint': 'animagine.safetensors [abc123]',
                 'use_vae': True, 'vae': 'sdxl_vae.safetensors',
                 'use_sampler': True, 'sampler': 'DPM++ 2M', 'scheduler': 'Karras'}
        widgets = self._slot_widgets(populated=False)
        self.assertEqual(self.mixin._set_slot_settings(widgets, saved), [])

        # 게이트: 연결 전에 저장한다 — 디스크를 읽지 않고도 부팅 값을 그대로 쓴다
        self.mixin._get_existing_setting = mock.Mock(return_value={})
        written = self.mixin._get_slot_settings(widgets, 'adetailer_slot1')
        for key in ('checkpoint', 'vae', 'sampler', 'scheduler'):
            self.assertEqual(written[key], saved[key], key)
        self.mixin._get_existing_setting.assert_not_called()

        # 연결: on_webui_info_loaded 와 같은 순서(스냅숏 → 비우고 채움 → 복원), ComfyUI 이름 목록
        host = SimpleNamespace(is_programmatic_change=False, s1_widgets=widgets)
        preserved = snapshot_backend_combos(host)
        for key, items in (('checkpoint_combo', ['Use same checkpoint', 'animagine.safetensors']),
                           ('vae_combo', ['Use same VAE', 'sdxl_vae.safetensors']),
                           ('sampler_combo', ['Use same sampler', 'euler', 'dpmpp_2m']),
                           ('scheduler_combo', ['normal', 'karras'])):
            widgets[key].clear()
            widgets[key].addItems(items)
        restored = restore_backend_combos(host, preserved, load_saved=lambda: {'adetailer_slot1': written})

        self.assertEqual(widgets['checkpoint_combo'].currentText(), 'animagine.safetensors')
        self.assertEqual(widgets['vae_combo'].currentText(), 'sdxl_vae.safetensors')
        self.assertEqual(widgets['sampler_combo'].currentText(), 'dpmpp_2m')
        self.assertEqual(widgets['scheduler_combo'].currentText(), 'karras')
        self.assertTrue({'s1.checkpoint_combo', 's1.vae_combo', 's1.sampler_combo', 's1.scheduler_combo'}
                        <= set(restored))

    def test_slot_save_after_a_connection_error_keeps_the_selection_or_the_disk_value(self):
        """on_webui_info_error 가 슬롯 체크포인트·VAE 를 비운 뒤 저장해도 ''를 쓰지 않는다."""
        widgets = self._slot_widgets()
        widgets['checkpoint_combo'].setCurrentIndex(1)            # 'anima.safetensors'
        widgets['vae_combo'].setCurrentIndex(1)                   # 'sdxl_vae.safetensors'
        widgets['checkpoint_combo'].clear()
        widgets['vae_combo'].clear()
        self.assertEqual(widgets['checkpoint_combo'].currentText(), '', '생성 요청 쪽은 여전히 빈 값')
        widgets['sampler_combo'] = self.ComboBoxProxy(self.bridge, '_ad_s1_sampler_fresh')   # 고른 적 없음
        disk = {'checkpoint': 'disk.safetensors', 'vae': 'disk_vae.safetensors', 'sampler': 'DPM++ 2M'}
        self.mixin._get_existing_setting = mock.Mock(return_value=disk)

        written = self.mixin._get_slot_settings(widgets, 'adetailer_slot1')

        self.assertEqual(written['checkpoint'], 'anima.safetensors')   # 비우기 직전 선택
        self.assertEqual(written['vae'], 'sdxl_vae.safetensors')
        self.assertEqual(written['sampler'], 'DPM++ 2M')              # 남은 선택이 없으면 디스크 값
        self.assertEqual(written['scheduler'], 'normal')
        self.mixin._get_existing_setting.assert_called_once_with('adetailer_slot1', {})

    def test_build_settings_dict_keeps_vae_and_slot_values_after_a_connection_error(self):
        """저장 경로 전체(_build_settings_dict) — 비운 메인 VAE·슬롯 콤보가 ''로 저장되지 않는다."""
        from ui.generator_settings import SettingsMixin

        class _Host(SettingsMixin):
            def __getattr__(self, name):   # 이 테스트가 보지 않는 위젯은 MagicMock
                if name.startswith('__'):
                    raise AttributeError(name)
                value = mock.MagicMock(name=name)
                setattr(self, name, value)
                return value

        host = _Host()
        host.vae_main_combo = self.ComboBoxProxy(self.bridge, 'vae_main_combo')
        host.vae_main_combo.addItems(['Automatic', 'sdxl_vae.safetensors'])
        host.vae_main_combo.setCurrentIndex(1)
        host.vae_main_combo.clear()                               # on_webui_info_error
        host.s1_widgets = self._slot_widgets()
        host.s1_widgets['checkpoint_combo'].setCurrentIndex(1)
        host.s1_widgets['checkpoint_combo'].clear()
        host.s2_widgets = self._slot_widgets(populated=False)     # 부팅 뒤 한 번도 못 채운 슬롯
        host._get_existing_setting = mock.Mock(side_effect=lambda key, default='': {
            'adetailer_slot2': {'checkpoint': 'slot2.safetensors'}}.get(key, default))

        settings = SettingsMixin._build_settings_dict(host)

        self.assertEqual(settings['vae_main'], 'sdxl_vae.safetensors')
        self.assertEqual(settings['adetailer_slot1']['checkpoint'], 'anima.safetensors')
        self.assertEqual(settings['adetailer_slot2']['checkpoint'], 'slot2.safetensors')

    def _settings_host(self):
        from ui.generator_settings import SettingsMixin

        class _Host(SettingsMixin):
            def __getattr__(self, name):   # 이 테스트가 보지 않는 위젯은 MagicMock
                if name.startswith('__'):
                    raise AttributeError(name)
                value = mock.MagicMock(name=name)
                setattr(self, name, value)
                return value

        host = _Host()
        host.s1_widgets = self._slot_widgets()
        host.s2_widgets = self._slot_widgets()
        return host

    def test_build_settings_dict_keeps_model_and_hires_checkpoint_after_a_connection_error(self):
        """Codex R6#0 — on_webui_info_error 는 모델·Hires 체크포인트 콤보도 비운다. 끊긴 채 종료 저장해도
        디스크의 옛 값이 아니라 비우기 직전 선택을 쓰고, 한 번도 고르지 못했으면 디스크 값."""
        from ui.generator_settings import SettingsMixin

        host = self._settings_host()
        host.model_combo = self.ComboBoxProxy(self.bridge, 'model_combo')
        host.model_combo.addItems(['old.safetensors', 'picked.safetensors'])
        host.model_combo.setCurrentIndex(1)
        host.hires_checkpoint_combo = self.ComboBoxProxy(self.bridge, 'hires_checkpoint_combo')
        host.hires_checkpoint_combo.addItems(['Use same checkpoint', 'hires.safetensors'])
        host.hires_checkpoint_combo.setCurrentIndex(1)
        host.model_combo.clear()                                  # on_webui_info_error
        host.hires_checkpoint_combo.clear()
        disk = {'model': 'old.safetensors', 'hires_checkpoint': 'Use same checkpoint'}
        host._get_existing_setting = mock.Mock(side_effect=lambda key, default='': disk.get(key, default))

        settings = SettingsMixin._build_settings_dict(host)
        self.assertEqual(host.model_combo.currentText(), '', '생성 요청 쪽은 여전히 빈 값')
        self.assertEqual(settings['model'], 'picked.safetensors')
        self.assertEqual(settings['hires_checkpoint'], 'hires.safetensors')

        host.model_combo = self.ComboBoxProxy(self.bridge, 'model_combo_fresh')          # 연결 전
        host.hires_checkpoint_combo = self.ComboBoxProxy(self.bridge, 'hires_ckpt_fresh')
        settings = SettingsMixin._build_settings_dict(host)
        self.assertEqual(settings['model'], 'old.safetensors')
        self.assertEqual(settings['hires_checkpoint'], 'Use same checkpoint')

    def _slot_extras(self, widgets, prefix='_ad_s1'):
        """generator_ui_setup._ad_slot 의 네거티브·Dilate/Erode·마스크 병합 프록시."""
        from ui.widget_proxies import SliderProxy, TextEditProxy

        widgets['neg_prompt'] = TextEditProxy(self.bridge, f'{prefix}_neg')
        dilate = SliderProxy(self.bridge, f'{prefix}_dilate_erode')
        dilate.setText('4')
        widgets['dilate_erode'] = dilate
        widgets['mask_merge_invert'] = self.ComboBoxProxy(self.bridge, f'{prefix}_mask_merge')
        return widgets

    def test_slot_negative_dilate_and_mask_merge_survive_save_and_restore(self):
        """생성 페이로드(_build_adetailer_slot)가 읽는 ad_negative_prompt·ad_dilate_erode·
        ad_mask_merge_invert 를 예전엔 저장·복원하지 않아 재시작·프리셋마다 초기화됐다."""
        from ui.generator_generation import GenerationMixin

        widgets = self._slot_extras(self._slot_widgets())
        widgets['neg_prompt']._on_vue_changed('bad hands, blurry')      # 사용자가 Vue 에서 입력
        widgets['dilate_erode']._on_vue_changed('12')
        widgets['mask_merge_invert']._on_vue_changed('Merge and Invert')
        written = self.mixin._get_slot_settings(widgets, 'adetailer_slot1')
        self.assertEqual((written['neg_prompt'], written['dilate_erode'], written['mask_merge_invert']),
                         ('bad hands, blurry', '12', 'Merge and Invert'))

        restored = self._slot_extras(self._slot_widgets(), prefix='_ad_s1_next')   # 재시작 뒤 새 위젯
        self.assertEqual(self.mixin._set_slot_settings(restored, json.loads(json.dumps(written))), [])
        self.assertIn(('_ad_s1_next_neg', 'bad hands, blurry'), self.pushed, 'Vue 도 같은 값을 본다')
        self.assertIn(('_ad_s1_next_dilate_erode', '12'), self.pushed)
        self.assertIn(('_ad_s1_next_mask_merge', 'Merge and Invert'), self.pushed)
        slot = GenerationMixin._build_adetailer_slot(None, restored)
        self.assertEqual(slot['ad_negative_prompt'], 'bad hands, blurry')
        self.assertEqual(slot['ad_dilate_erode'], 12)
        self.assertEqual(slot['ad_mask_merge_invert'], 'Merge and Invert')

    def test_old_slot_settings_without_the_new_keys_leave_the_current_values(self):
        """키가 없는 옛 설정·프리셋은 지금 입력해 둔 네거티브·Dilate/Erode·마스크 병합을 지우지 않는다."""
        widgets = self._slot_extras(self._slot_widgets())
        widgets['neg_prompt']._on_vue_changed('keep me')
        widgets['dilate_erode']._on_vue_changed('7')
        widgets['mask_merge_invert']._on_vue_changed('Merge')
        self.mixin._set_slot_settings(widgets, {'model': 'face_yolov8n.pt', 'prompt': 'detailed face'})
        self.assertEqual(widgets['neg_prompt'].toPlainText(), 'keep me')
        self.assertEqual(widgets['dilate_erode'].text(), '7')
        self.assertEqual(widgets['mask_merge_invert'].text(), 'Merge')
        self.mixin._set_slot_settings(widgets, {'dilate_erode': 0, 'neg_prompt': None})   # 0 은 값이다
        self.assertEqual(widgets['dilate_erode'].text(), '0')
        self.assertEqual(widgets['neg_prompt'].toPlainText(), '')

    def test_slot_extras_are_part_of_the_preset_slot_keys(self):
        """PRESET_KEYS 의 adetailer_slot1/2 는 슬롯 dict 통째 — 새 키도 프리셋 저장·불러오기를 탄다."""
        widgets = self._slot_extras(self._slot_widgets())
        widgets['neg_prompt']._on_vue_changed('lowres')
        with tempfile.TemporaryDirectory() as folder:
            name = write_preset('slot', {'adetailer_slot1': self.mixin._get_slot_settings(widgets)}, folder)
            loaded = read_preset(name, folder)
        self.assertEqual(loaded['adetailer_slot1']['neg_prompt'], 'lowres')
        host = _host([])
        host.s1_widgets = self._slot_extras(self._slot_widgets(), prefix='_ad_s1_preset')
        host._set_slot_settings = self.mixin._set_slot_settings
        apply_generation_settings(host, loaded, only_present=True)
        self.assertEqual(host.s1_widgets['neg_prompt'].toPlainText(), 'lowres')

    def test_preset_apply_prefixes_slot_warnings(self):
        host = _host([])
        host.s1_widgets = self._slot_widgets()
        host._set_slot_settings = self.mixin._set_slot_settings
        warnings = apply_generation_settings(host, {'adetailer_slot1': {'sampler': 'Restart'}}, only_present=True)
        self.assertEqual(len(warnings), 1)
        self.assertTrue(warnings[0].startswith('ADetailer 1: 샘플러'))

    def test_sam3_checkpoint_on_a_populated_list_is_not_faked(self):
        from collections import defaultdict

        checkpoint = self.ComboBoxProxy(self.bridge, '_sam3_checkpoint')
        checkpoint.addItems(['sam3.pt'])
        widgets = defaultdict(_Line)
        for key in ('mask_hull', 'inpaint_only_masked', 'preview_overlay', 'save_artifacts', 'unload_after',
                    'use_inpaint_size_check', 'use_steps_check', 'use_cfg_check', 'use_sampler_check',
                    'use_scheduler_check', 'use_seed_check', 'use_noise_multiplier_check', 'restore_face'):
            widgets[key] = _Check()
        widgets['checkpoint'] = checkpoint
        self.pushed.clear()
        with mock.patch('core.sam3_controlnet.apply_settings'):
            misses = self.mixin._set_sam3_settings(widgets, {'checkpoint': 'models/sam3/other_sam3.safetensors'})
        self.assertEqual(checkpoint.currentText(), 'sam3.pt')
        self.assertNotIn(('_sam3_checkpoint', 'models/sam3/other_sam3.safetensors'), self.pushed)
        self.assertEqual(len(misses), 1)


class ApplyGenerationPresetBatchTests(unittest.TestCase):
    def test_batch_mode_is_released_even_when_apply_fails(self):
        from ui.generator_main import GeneratorMainUI

        calls = []
        bridge = SimpleNamespace(beginBatchUpdate=lambda: calls.append('begin'),
                                 endBatchUpdate=lambda: calls.append('end'))
        class _Broken:
            def setText(self, _value):
                raise RuntimeError('widget gone')

        host = SimpleNamespace(vue_bridge=bridge, is_programmatic_change=False,
                               character_input=_Broken(), update_total_prompt_display=lambda: None)
        with self.assertRaises(RuntimeError):
            GeneratorMainUI._apply_generation_preset(host, {'character': 'x'})
        self.assertEqual(calls, ['begin', 'end'])
        self.assertFalse(host.is_programmatic_change)

    def test_preset_apply_returns_model_warnings(self):
        from ui.generator_main import GeneratorMainUI

        host = _host(['a.safetensors'])
        host.vue_bridge = SimpleNamespace(beginBatchUpdate=lambda: None, endBatchUpdate=lambda: None)
        host.is_programmatic_change = False
        host.update_total_prompt_display = lambda: None
        warnings = GeneratorMainUI._apply_generation_preset(host, {'model': 'b.safetensors', 'steps': '12'})
        self.assertEqual(host.steps_input.text(), '12')
        self.assertEqual(len(warnings), 1)

    def test_preset_prompt_fields_become_base_templates(self):
        """프리셋의 선행/네거티브는 새 base_* 템플릿 — 프리셋에 없는 후행 템플릿은 그대로."""
        from ui.generator_main import GeneratorMainUI

        host = _host([])
        host.vue_bridge = SimpleNamespace(beginBatchUpdate=lambda: None, endBatchUpdate=lambda: None)
        host.is_programmatic_change = False
        host.update_total_prompt_display = lambda: None
        host.base_prefix_prompt, host.base_suffix_prompt, host.base_neg_prompt = 'OLD_P', 'OLD_S', 'OLD_N'
        host.suffix_prompt_text.setPlainText('OLD_S, cond_tag')   # 지난 사이클의 조건부 태그
        GeneratorMainUI._apply_generation_preset(host, {'prefix_prompt': 'NEW_P', 'negative_prompt': 'NEW_N'})
        self.assertEqual(host.base_prefix_prompt, 'NEW_P')
        self.assertEqual(host.base_neg_prompt, 'NEW_N')
        self.assertEqual(host.base_suffix_prompt, 'OLD_S', '프리셋에 없는 칸의 글을 템플릿으로 굳히지 않는다')
        self.assertFalse(host.is_programmatic_change)

    def test_next_random_cycle_keeps_the_preset_prompts(self):
        """실제 TextEditProxy + on_base_prompts_changed 배선(generator_actions) 그대로 — 프리셋 뒤
        Vue 에코와 apply_prompt_from_data 의 base 복원(8단계)을 거쳐도 프리셋 값이 남는다."""
        from ui.generator_main import GeneratorMainUI
        from ui.generator_prompts import PromptHandlingMixin
        from ui.vue_bridge import VueBridge
        from ui.widget_proxies import TextEditProxy

        bridge = VueBridge()
        host = _host([])
        host.vue_bridge = bridge
        host.is_programmatic_change = False
        host.update_total_prompt_display = lambda: None
        host.base_prefix_prompt = host.base_suffix_prompt = host.base_neg_prompt = ''
        for attr in ('prefix_prompt_text', 'suffix_prompt_text', 'neg_prompt_text'):
            proxy = TextEditProxy(bridge, attr)
            proxy.textChanged.connect(lambda: PromptHandlingMixin.on_base_prompts_changed(host))
            setattr(host, attr, proxy)
        # 사용자가 Vue 에서 입력 → 템플릿이 따라간다
        host.prefix_prompt_text._on_vue_changed('masterpiece, OLD_PREFIX')
        host.suffix_prompt_text._on_vue_changed('OLD_SUFFIX')
        host.neg_prompt_text._on_vue_changed('OLD_NEG, lowres')
        self.assertEqual(host.base_neg_prompt, 'OLD_NEG, lowres')

        GeneratorMainUI._apply_generation_preset(
            host, {'prefix_prompt': 'NEW_PREFIX', 'suffix_prompt': 'NEW_SUFFIX', 'negative_prompt': 'NEW_NEG'})
        host.prefix_prompt_text._on_vue_changed('NEW_PREFIX')     # Vue 에코 — 같은 값이라 아무 일 없음

        # 다음 랜덤 프롬프트·자동화 사이클(apply_prompt_from_data 8단계)의 base 복원
        host.is_programmatic_change = True
        host.prefix_prompt_text.setPlainText(host.base_prefix_prompt)
        host.suffix_prompt_text.setPlainText(host.base_suffix_prompt)
        host.neg_prompt_text.setPlainText(host.base_neg_prompt)
        host.is_programmatic_change = False

        self.assertEqual(host.prefix_prompt_text.toPlainText(), 'NEW_PREFIX')
        self.assertEqual(host.suffix_prompt_text.toPlainText(), 'NEW_SUFFIX')
        self.assertEqual(host.neg_prompt_text.toPlainText(), 'NEW_NEG')

    def test_preset_save_writes_the_templates_not_the_resolved_cycle_text(self):
        """Codex R6#1 — 사이클 뒤 칸엔 해석된 와일드카드·조건부 태그가 있다. 예전 저장은 그 글을 썼고,
        불러오기의 sync_base_prompts 가 그걸 템플릿으로 굳혀 와일드카드가 더 이상 새로 뽑히지 않고
        조건부 태그는 조건이 맞지 않아도 남았다. 이제 저장은 base_* 템플릿을 쓴다."""
        from ui.generator_main import GeneratorMainUI
        from ui.generator_prompts import PromptHandlingMixin
        from ui.generator_settings import SettingsMixin
        from ui.vue_bridge import VueBridge
        from ui.widget_proxies import TextEditProxy

        bridge = VueBridge()
        host = _host([])
        host.vue_bridge = bridge
        host.is_programmatic_change = False
        host.update_total_prompt_display = lambda: None
        host.base_prefix_prompt = host.base_suffix_prompt = host.base_neg_prompt = ''
        for attr in ('prefix_prompt_text', 'suffix_prompt_text', 'neg_prompt_text'):
            proxy = TextEditProxy(bridge, attr)
            proxy.textChanged.connect(lambda: PromptHandlingMixin.on_base_prompts_changed(host))
            setattr(host, attr, proxy)
        host.prefix_prompt_text._on_vue_changed('masterpiece, __hair_color__')   # 사용자가 입력한 템플릿
        host.neg_prompt_text._on_vue_changed('lowres')
        # 랜덤·자동화 사이클(apply_prompt_from_data): base 복원 → 조건부 태그 → 와일드카드 해석
        host.is_programmatic_change = True
        host.prefix_prompt_text.setPlainText('masterpiece, blonde hair')
        host.suffix_prompt_text.setPlainText('cond_tag')
        host.neg_prompt_text.setPlainText('lowres, cond_neg')
        host.is_programmatic_change = False

        host._build_settings_dict = lambda: {
            'prefix_prompt': host.prefix_prompt_text.toPlainText(), 'main_prompt': '1girl',
            'suffix_prompt': host.suffix_prompt_text.toPlainText(),
            'negative_prompt': host.neg_prompt_text.toPlainText(), 'steps': '28'}
        with tempfile.TemporaryDirectory() as folder:
            name = write_preset('cycle', SettingsMixin._build_preset_settings(host), folder)
            preset = read_preset(name, folder)
        self.assertEqual((preset['prefix_prompt'], preset['suffix_prompt'], preset['negative_prompt']),
                         ('masterpiece, __hair_color__', '', 'lowres'))
        self.assertEqual((preset['main_prompt'], preset['steps']), ('1girl', '28'), '다른 키는 그대로')
        self.assertEqual(host.prefix_prompt_text.toPlainText(), 'masterpiece, blonde hair', '칸은 건드리지 않는다')

        # 불러오면 템플릿이 그대로 — 다음 사이클이 와일드카드를 새로 뽑고 조건부 태그를 다시 판단한다
        GeneratorMainUI._apply_generation_preset(host, preset)
        self.assertEqual((host.base_prefix_prompt, host.base_suffix_prompt, host.base_neg_prompt),
                         ('masterpiece, __hair_color__', '', 'lowres'))

    def test_preset_prompt_templates_only_replace_present_keys_of_hosts_with_templates(self):
        from ui.generation_settings_apply import preset_prompt_templates

        settings = {'prefix_prompt': 'widget', 'steps': '20'}
        self.assertEqual(preset_prompt_templates(SimpleNamespace(), settings), settings)
        host = SimpleNamespace(base_prefix_prompt='template', base_suffix_prompt='S', base_neg_prompt=None)
        out = preset_prompt_templates(host, {**settings, 'negative_prompt': 'widget neg'})
        self.assertEqual(out, {'prefix_prompt': 'template', 'steps': '20', 'negative_prompt': ''})
        self.assertNotIn('suffix_prompt', out, '없는 키를 만들지 않는다')
        self.assertEqual(settings['prefix_prompt'], 'widget', '입력 dict 는 바꾸지 않는다')

    def test_legacy_settings_without_base_keys_use_the_field_text_as_templates(self):
        """base_* 키가 없는 옛 prompt_settings — 예전엔 템플릿이 ''라 다음 사이클이 칸을 비웠고,
        프리셋 저장(템플릿 기준)도 빈 선행/네거티브를 썼다."""
        from ui.generator_settings import SettingsMixin

        class _Host(SettingsMixin):
            def __getattr__(self, name):   # 이 테스트가 보지 않는 위젯은 MagicMock
                if name.startswith('__'):
                    raise AttributeError(name)
                value = mock.MagicMock(name=name)
                setattr(self, name, value)
                return value

        cases = (
            ({'prefix_prompt': 'legacy P', 'suffix_prompt': 'legacy S', 'negative_prompt': 'legacy N'},
             ('legacy P', 'legacy S', 'legacy N')),
            ({'prefix_prompt': 'resolved', 'base_prefix_prompt': '__tpl__', 'base_suffix_prompt': '',
              'suffix_prompt': 'cond', 'negative_prompt': 'n', 'base_neg_prompt': 'n'},
             ('__tpl__', '', 'n')),
        )
        for saved, expected in cases:
            with self.subTest(saved=saved), tempfile.TemporaryDirectory() as folder:
                path = Path(folder) / 'prompt_settings.json'
                path.write_text(json.dumps(saved), encoding='utf-8')
                host = _Host()
                with mock.patch('ui.generator_settings.PROMPT_SETTINGS_FILE', str(path)), \
                        mock.patch('ui.generator_settings.migrate_legacy_gallery_folder', return_value=''), \
                        mock.patch('ui.generation_settings_apply.apply_generation_settings', return_value=[]):
                    SettingsMixin.load_settings(host)
                self.assertEqual((host.base_prefix_prompt, host.base_suffix_prompt, host.base_neg_prompt),
                                 expected)


if __name__ == '__main__':
    unittest.main()
