"""파일 와일드카드 — Vue 관리자가 넣는 __name__ 이 ~/name/~ 와 같게 풀리는지, 주석 보존·이름 규칙 회귀."""
import json
import os
import random
import stat
import tempfile
import unittest
from types import SimpleNamespace
from unittest import mock

import utils.file_wildcard as fw
from utils.file_wildcard import FileWildcardManager, wildcards_enabled


class _Tmp(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.dir = os.path.join(self._tmp.name, 'wildcards')
        self.mgr = FileWildcardManager(self.dir)
        random.seed(7)

    def tearDown(self):
        self._tmp.cleanup()

    def write(self, name, text):
        with open(os.path.join(self.dir, name + '.txt'), 'w', encoding='utf-8') as f:
            f.write(text)


class SyntaxTests(_Tmp):
    def test_dunder_and_tilde_syntax_resolve_the_same_way(self):
        self.write('hair', '# 머리색\nred hair, blue hair\n\n  # 길이\nlong hair\n')
        for text in ('a, __hair__, b', 'a, ~/hair/~, b'):
            out = self.mgr.resolve(text)
            self.assertRegex(out, r'^a, (red|blue) hair, long hair, b$')
            self.assertNotIn('#', out)

    def test_vue_inserted_syntax_is_resolved_by_the_generation_entry_point(self):
        # App.vue useWcSyntax 가 넣는 '__이름__, ' 이 생성 경로(resolve_file_wildcards)에서 풀린다
        self.write('mood', 'smile')
        with mock.patch.object(fw, '_instance', self.mgr):
            self.assertEqual(fw.resolve_file_wildcards('1girl, __mood__, '), '1girl, smile, ')

    def test_unknown_names_and_plain_underscores_stay_literal(self):
        self.write('mood', 'smile')
        for text in ('__missing__', '^__^', 'a__b', '__ spaced__', '__a, b__', '___mood___'):
            self.assertEqual(self.mgr.resolve(text), text.replace('___mood___', '_smile_'))

    def test_names_with_inner_underscores_and_n_pick(self):
        self.write('hair_style', 'a1, a2\nb1\nc1')
        self.assertEqual(len(self.mgr.resolve('__hair_style:2__').split(', ')), 2)
        self.assertEqual(self.mgr.resolve('~/hair_style/~').split(', ')[1:], ['b1', 'c1'])

    def test_nested_references(self):
        self.write('outer', '__inner__ outfit')
        self.write('inner', 'red')
        self.assertEqual(self.mgr.resolve('__outer__'), 'red outfit')

    def test_names_dunder_cannot_express_resolve_with_the_tilde_form(self):
        # frontend/src/utils/wildcardFile.ts wildcardSyntax 는 이런 이름에 ~/이름/~ 을 넣는다
        # (같은 벡터를 wildcardFile.test.ts 가 고정한다). __이름__ 으로는 못 풀거나 남의 파일을 푼다.
        self.write('base', 'WRONG')
        for name in ('_base', 'base_', 'a,b', 'a__b'):
            self.write(name, 'picked')
        for name in ('_base', 'base_', 'a,b', 'a__b'):
            self.assertEqual(self.mgr.resolve(f'~/{name}/~'), 'picked', name)
            self.assertNotEqual(self.mgr.resolve(f'__{name}__'), 'picked', name)
        self.assertEqual(self.mgr.resolve('___base__'), '_WRONG')     # '_base' 가 아니라 base.txt 를 푼다

    def test_names_cannot_escape_the_wildcards_folder(self):
        outside = os.path.join(self._tmp.name, 'secret.txt')
        with open(outside, 'w', encoding='utf-8') as f:
            f.write('leaked')
        for text in ('~/..\\secret/~', '__..\\secret__'):
            self.assertEqual(self.mgr.resolve(text), text)


class ManagementTests(_Tmp):
    def test_tree_keeps_comments_in_lines_and_counts_only_effective_tags(self):
        self.write('hair', '# 색\nred hair, blue hair\n\n   # 길이\nlong hair\n\n')
        [entry] = self.mgr.get_wildcard_tree()
        self.assertEqual(entry['name'], 'hair')
        self.assertEqual(entry['file'], 'hair.txt')
        self.assertEqual(entry['lines'], ['# 색', 'red hair, blue hair', '', '   # 길이', 'long hair'])
        self.assertEqual(entry['tags'], ['red hair, blue hair', 'long hair'])

    def test_save_round_trips_comments_and_invalidates_cache(self):
        self.assertEqual(self.mgr.save_wildcard('mood.txt', '# note\nsmile'), 'mood')
        self.assertEqual(self.mgr.resolve('__mood__'), 'smile')
        self.mgr.save_wildcard('mood', '# note\ncry')
        self.assertEqual(self.mgr.resolve('__mood__'), 'cry')
        self.assertEqual(self.mgr.get_wildcard_content('mood'), '# note\ncry')

    def test_save_sanitizes_names_and_rejects_unusable_ones(self):
        self.assertEqual(self.mgr.save_wildcard('a/b:c?', 'x'), 'abc')
        self.assertTrue(os.path.isfile(os.path.join(self.dir, 'abc.txt')))
        with self.assertRaises(ValueError):
            self.mgr.save_wildcard('///', 'x')

    def test_rename_refuses_to_overwrite_and_returns_the_saved_name(self):
        self.write('a', 'one')
        self.write('b', 'two')
        with self.assertRaises(FileExistsError):
            self.mgr.rename_wildcard('a', 'b')
        self.assertEqual(self.mgr.get_wildcard_content('b'), 'two')
        self.assertEqual(self.mgr.rename_wildcard('a', 'c?'), 'c')
        self.assertEqual(self.mgr.get_wildcard_names(), ['b', 'c'])

    def test_delete(self):
        self.write('a', 'one')
        self.mgr.delete_wildcard('a.txt')
        self.assertEqual(self.mgr.get_wildcard_names(), [])

    def test_delete_removes_the_listed_file_even_when_the_name_rule_would_change_it(self):
        # 밖에서 만든 ' lead.txt' · 'dot..txt' 는 목록에 ' lead' · 'dot.' 으로 나오지만 이름 규칙(앞뒤 공백·
        # 끝의 점 제거)으론 lead.txt · dot.txt 를 가리킨다. 예전 삭제는 그 없는 경로만 보고 조용히 성공해
        # 브리지가 {ok:true} 를, Vue 가 '와일드카드 삭제됨'을 띄웠고 파일은 다음 목록에 되살아났다.
        self.write(' lead', 'odd')
        self.write('dot.', 'dotted')
        self.write('keep', 'k')
        self.assertEqual(self.mgr.get_wildcard_names(), [' lead', 'dot.', 'keep'])
        from ui.vue_bridge import VueBridge
        with mock.patch.object(fw, '_instance', self.mgr):
            for listed in (' lead', 'dot.'):
                self.assertEqual(json.loads(VueBridge().deleteWildcard(listed)), {'ok': True}, listed)
        self.assertEqual(self.mgr.get_wildcard_names(), ['keep'])
        self.assertEqual(sorted(os.listdir(self.dir)), ['keep.txt'])

    def test_delete_never_removes_a_different_file_the_name_rule_maps_to(self):
        # ' lead' 를 지우는데 규칙상 같은 lead.txt 를 지우면 안 된다(예전엔 그 파일을 지웠다).
        self.write(' lead', 'odd')
        self.write('lead', 'real')
        self.mgr.delete_wildcard(' lead')
        self.assertEqual(self.mgr.get_wildcard_names(), ['lead'])
        # 가리키는 파일이 이미 없으면(목록이 낡았다) 목표 상태라 성공 — 규칙상 같아지는 파일은 그대로
        self.mgr.delete_wildcard(' lead')
        self.mgr.delete_wildcard('lead.')
        self.assertEqual(self.mgr.get_wildcard_content('lead'), 'real')

    def test_rename_moves_the_listed_file_and_fails_when_it_is_missing(self):
        self.write(' lead', 'odd')
        self.write('lead', 'real')
        self.assertEqual(self.mgr.rename_wildcard(' lead', 'fixed'), 'fixed')
        self.assertEqual(self.mgr.get_wildcard_names(), ['fixed', 'lead'])
        self.assertEqual(self.mgr.get_wildcard_content('fixed'), 'odd')
        self.assertEqual(self.mgr.get_wildcard_content('lead'), 'real')
        # 옛 파일이 없으면 아무것도 바꾸지 않고 실패 — 예전엔 조용히 새 이름을 돌려줘 Vue 가 목록 이름을
        # 바꿨다(applyWildcardRename). 규칙상 같은 lead.txt 를 대신 옮기지도 않는다.
        with self.assertRaises(FileNotFoundError):
            self.mgr.rename_wildcard(' lead', 'other')
        from ui.vue_bridge import VueBridge
        with mock.patch.object(fw, '_instance', self.mgr):
            reply = json.loads(VueBridge().renameWildcard('dot.', 'other'))
        self.assertIn('error', reply)
        self.assertNotIn('ok', reply)
        self.assertEqual(self.mgr.get_wildcard_names(), ['fixed', 'lead'])

    @unittest.skipUnless(os.path.normcase('A') == os.path.normcase('a'), '대소문자를 가리지 않는 파일 시스템')
    def test_case_only_different_names_still_reach_the_file(self):
        self.write('mood', 'smile')
        self.write('hair', 'red')
        self.assertEqual(self.mgr.rename_wildcard('HAIR', 'Hair'), 'Hair')
        self.assertEqual(sorted(os.listdir(self.dir)), ['Hair.txt', 'mood.txt'])
        self.mgr.delete_wildcard('MOOD')
        self.assertEqual(self.mgr.get_wildcard_names(), ['Hair'])

    @unittest.skipUnless(os.name == 'nt', '읽기 전용 파일의 삭제 거부는 Windows 동작')
    def test_failed_delete_raises_and_the_bridge_answers_with_an_error(self):
        # Vue(useWildcardManager.deleteWildcard)는 {error} 응답이면 목록·선택·편집 중인 줄을 그대로 두고
        # 실패를 알린다 — 그러려면 삭제 실패(읽기 전용 WinError 5 · 잠김 WinError 32)가 삼켜지지 않고
        # 올라가 브리지가 {error} 로 바꿔야 한다. 예전 Vue 는 응답을 안 읽어 남은 파일을 지운 것처럼 보였다.
        self.write('locked', 'keep')
        path = os.path.join(self.dir, 'locked.txt')
        os.chmod(path, stat.S_IREAD)
        try:
            with self.assertRaises(PermissionError):
                self.mgr.delete_wildcard('locked')
            from ui.vue_bridge import VueBridge
            with mock.patch.object(fw, '_instance', self.mgr):
                reply = json.loads(VueBridge().deleteWildcard('locked'))
            self.assertIn('error', reply)
            self.assertNotIn('ok', reply)
            self.assertTrue(os.path.isfile(path))
            self.assertEqual(self.mgr.get_wildcard_names(), ['locked'])
            self.assertEqual(self.mgr.resolve('__locked__'), 'keep')     # 생성은 계속 이 파일을 푼다
        finally:
            os.chmod(path, stat.S_IWRITE | stat.S_IREAD)

    def test_create_makes_a_new_empty_file(self):
        self.assertEqual(self.mgr.create_wildcard('mood'), {'name': 'mood', 'created': True})
        self.assertEqual(self.mgr.get_wildcard_content('mood'), '')
        self.assertEqual(self.mgr.create_wildcard('a/b?'), {'name': 'ab', 'created': True})
        with self.assertRaises(ValueError):
            self.mgr.create_wildcard('///')

    def test_create_never_truncates_an_existing_file(self):
        # '+ NEW' 에 기존 파일과 규칙상 같은 이름을 쳐도 주석까지 그대로 남는다
        self.write('hairstyle', '# my comment\nponytail, twintails')
        names = ['hairstyle', 'hairstyle.', 'hair:style', 'hairstyle.txt', 'hair?style']
        if os.path.normcase('A') == os.path.normcase('a'):   # 대소문자를 가리지 않는 파일 시스템(Windows)
            names += ['Hairstyle', 'HAIRSTYLE']
        for typed in names:
            self.assertEqual(self.mgr.create_wildcard(typed), {'name': 'hairstyle', 'created': False}, typed)
            self.assertEqual(self.mgr.get_wildcard_content('hairstyle'), '# my comment\nponytail, twintails')
        self.assertEqual(self.mgr.get_wildcard_names(), ['hairstyle'])

    def test_names_longer_than_64_characters_round_trip(self):
        name = 'summer_festival_night_outfits_for_characters_with_accessories_v2_extra'
        self.assertGreater(len(name), 64)
        self.write(name, '# 긴 이름\nyukata, kimono')
        self.assertIn(self.mgr.resolve(f'~/{name}/~'), ('yukata', 'kimono'))
        self.assertIn(self.mgr.resolve(f'__{name}__'), ('yukata', 'kimono'))
        [entry] = self.mgr.get_wildcard_tree()
        self.assertEqual(entry['name'], name)
        self.assertEqual(entry['lines'], ['# 긴 이름', 'yukata, kimono'])
        self.assertEqual(self.mgr.save_wildcard(name, 'geta'), name)
        self.assertEqual(self.mgr.get_wildcard_names(), [name])      # 잘린 이름으로 새 파일을 만들지 않는다
        self.mgr.delete_wildcard(name)
        self.assertEqual(self.mgr.get_wildcard_names(), [])

    def test_names_are_capped_at_the_file_name_limit(self):
        self.assertEqual(len(fw.wildcard_file_name('x' * 400)), fw.WILDCARD_NAME_MAX)


class WildcardsEnabledTests(unittest.TestCase):
    def test_reads_the_prompt_settings_extras_store(self):
        from core.prompt_settings_extras import PromptSettingsExtras

        on = SimpleNamespace(prompt_settings_extras=PromptSettingsExtras(wildcard_enabled=True))
        off = SimpleNamespace(prompt_settings_extras=PromptSettingsExtras(wildcard_enabled=False))
        self.assertTrue(wildcards_enabled(on))
        self.assertFalse(wildcards_enabled(off))
        self.assertFalse(wildcards_enabled(SimpleNamespace()), '보관함 없는 호스트는 꺼짐')
        # 예전 숨은 SettingsTab 체크박스는 더 이상 보지 않는다(audit #178)
        legacy = SimpleNamespace(settings_tab=SimpleNamespace(
            chk_wildcard_enabled=SimpleNamespace(isChecked=lambda: True)))
        self.assertFalse(wildcards_enabled(legacy))

    def test_vue_toggle_updates_the_store_and_load_syncs_the_toggle(self):
        """Vue 토글(CheckBoxProxy 'wildcard_enabled') → 보관함, 설정 불러오기 → 토글."""
        from core.prompt_settings_extras import PromptSettingsExtras
        from ui.generator_settings import SettingsMixin
        from ui.generator_ui_setup import UISetupMixin

        class _Bridge:
            def __init__(self):
                self.pushed = []

            def _register_proxy(self, widget_id, proxy):
                pass

            def pushWidgetValue(self, widget_id, value):
                self.pushed.append((widget_id, value))

        class Host(UISetupMixin, SettingsMixin):
            pass

        host = Host.__new__(Host)
        host.vue_bridge = _Bridge()
        host.prompt_settings_extras = PromptSettingsExtras()
        host.prompt_cleaner = SimpleNamespace(set_options=lambda **_kw: None)
        UISetupMixin._bind_wildcard_enabled_proxy(host)
        self.assertTrue(host.wildcard_enabled_check.isChecked())

        host.wildcard_enabled_check._on_vue_changed('false')        # 사용자가 Vue 에서 끔
        self.assertFalse(host.prompt_settings_extras.wildcard_enabled)
        self.assertFalse(wildcards_enabled(host))

        SettingsMixin._apply_prompt_settings_extras(host, PromptSettingsExtras(wildcard_enabled=True))
        self.assertTrue(host.wildcard_enabled_check.isChecked())
        self.assertTrue(wildcards_enabled(host))
        self.assertIn(('wildcard_enabled', 'true'), host.vue_bridge.pushed, 'Vue 토글도 맞춘다')


if __name__ == '__main__':
    unittest.main()
