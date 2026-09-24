"""ui_prefs.json 의 경로·읽기는 core.ui_prefs 한 곳 (감사 #180).

예전엔 약 15곳이 파일을 열었고 10곳은 ``os.path.join(..., 'config', 'ui_prefs.json')`` 을 손으로
조립했으며 4곳은 ``json.load`` 로 마이그레이션·정규화를 건너뛰었다. 저장 경로 정책이나 키 마이그레이션을
바꾸면 그쪽만 조용히 어긋난다. Ollama 언로드 판단도 두 벌이었다.
"""
from __future__ import annotations

import ast
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from core import ui_prefs
from core.storage_paths import config_file

ROOT = Path(__file__).resolve().parents[1]
PRODUCTION_DIRS = ('backends', 'core', 'tabs', 'ui', 'utils', 'widgets', 'workers')
# 리터럴이 있어도 되는 곳: 경로의 주인, 설정 백업 파일 목록(경로가 아니라 백업할 이름표)
ALLOWED = {Path('core/ui_prefs.py'), Path('core/settings_backup.py')}


def _production_files():
    for folder in PRODUCTION_DIRS:
        yield from sorted((ROOT / folder).rglob('*.py'))
    yield from sorted(ROOT.glob('*.py'))


class UiPrefsLiteralGuardTests(unittest.TestCase):
    def test_no_one_else_spells_the_ui_prefs_file_name(self):
        offenders = []
        for path in _production_files():
            relative = path.relative_to(ROOT)
            if relative in ALLOWED or '__pycache__' in path.parts:
                continue
            try:
                tree = ast.parse(path.read_text(encoding='utf-8'))
            except (SyntaxError, UnicodeDecodeError):
                continue
            for node in ast.walk(tree):
                if isinstance(node, ast.Constant) and isinstance(node.value, str):
                    value = node.value.replace('\\', '/')
                    if value == ui_prefs.UI_PREFS_NAME or value.endswith('/' + ui_prefs.UI_PREFS_NAME):
                        offenders.append(f'{relative}:{node.lineno}')
        self.assertEqual(
            offenders, [],
            "ui_prefs.json 경로는 core.ui_prefs.ui_prefs_path()/read_ui_prefs() (또는 UI_PREFS_NAME) 로만",
        )


class ReadUiPrefsTests(unittest.TestCase):
    def test_path_follows_the_storage_policy(self):
        self.assertEqual(ui_prefs.ui_prefs_path(), str(config_file('ui_prefs.json')))

    def test_reads_through_the_migrating_loader_and_returns_fresh_dicts(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'ui_prefs.json'
            path.write_text(json.dumps({'schema_version': 1, 'ollamaModel': 'x'}), encoding='utf-8')
            first = ui_prefs.read_ui_prefs(str(path))
            first['ollamaModel'] = 'mutated'
            second = ui_prefs.read_ui_prefs(str(path))
            self.assertEqual(second['ollamaModel'], 'x')
            # 로더가 정규화한 키(iconAnimationStyle)가 들어온다 — raw json.load 가 아니다
            self.assertEqual(second['iconAnimationStyle'], 'none')

    def test_missing_or_broken_file_yields_only_loader_defaults(self):
        with tempfile.TemporaryDirectory() as tmp:
            # 로더가 정규화 기본값(iconAnimationStyle)만 채운다 — 사용자 키는 없다
            self.assertEqual(ui_prefs.read_ui_prefs(str(Path(tmp) / 'none.json')), {'iconAnimationStyle': 'none'})
            broken = Path(tmp) / 'broken.json'
            broken.write_text('{nope', encoding='utf-8')
            self.assertEqual(ui_prefs.read_ui_prefs(str(broken)), {'iconAnimationStyle': 'none'})

    def test_loader_failure_is_empty(self):
        with mock.patch('core.config_migration.load_ui_prefs', side_effect=OSError('locked')):
            self.assertEqual(ui_prefs.read_ui_prefs('x.json'), {})

    def test_loader_is_looked_up_at_call_time(self):
        with mock.patch('core.config_migration.load_ui_prefs', return_value={'a': 1}) as loader:
            self.assertEqual(ui_prefs.read_ui_prefs('whatever.json'), {'a': 1})
        loader.assert_called_once_with('whatever.json')


class OllamaUnloadDecisionTests(unittest.TestCase):
    def test_single_rule(self):
        from core.ollama_client import DEFAULT_OLLAMA_URL
        target = ui_prefs.ollama_unload_target
        self.assertIsNone(target({}))
        self.assertIsNone(target({'ollamaUnloadOnGen': True}))
        self.assertIsNone(target({'ollamaUnloadOnGen': True, 'ollamaModel': '   '}))
        self.assertIsNone(target({'ollamaUnloadOnGen': False, 'ollamaModel': 'm'}))
        self.assertEqual(target({'ollamaUnloadOnGen': True, 'ollamaModel': ' m '}), (DEFAULT_OLLAMA_URL.rstrip('/'), 'm'))
        self.assertEqual(target({'ollamaUnloadOnGen': True, 'ollamaModel': 'm', 'ollamaUrl': 'http://h:1/'}), ('http://h:1', 'm'))

    def test_creator_and_manual_generation_agree(self):
        from ui.creator_actions import CreatorActionsMixin
        cases = [
            {'ollamaUnloadOnGen': True, 'ollamaModel': 'gemma'},
            {'ollamaUnloadOnGen': True, 'ollamaModel': ''},
            {'ollamaUnloadOnGen': False, 'ollamaModel': 'gemma'},
            {},
        ]
        for prefs in cases:
            with self.subTest(prefs=prefs), mock.patch('core.ui_prefs.read_ui_prefs', return_value=prefs):
                self.assertEqual(
                    CreatorActionsMixin()._creator_should_unload_ollama(),
                    ui_prefs.ollama_unload_target(prefs) is not None,
                )

    def test_generation_unload_skips_when_disabled(self):
        from ui.generator_generation import GenerationMixin
        with mock.patch('core.ui_prefs.read_ui_prefs', return_value={'ollamaUnloadOnGen': False, 'ollamaModel': 'm'}), \
                mock.patch('threading.Thread') as thread:
            GenerationMixin._maybe_unload_ollama(object())
        thread.assert_not_called()


if __name__ == '__main__':
    unittest.main()
