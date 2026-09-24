"""조건식 저장: 제어 플래그는 파일에 남지 않고, 수동 저장만 토스트 (감사 #108)."""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from core.cond_rules_store import (
    COND_RULES_NAME,
    cond_rules_path,
    now_ms,
    save_cond_rules_payload,
    split_control_flags,
    stamp_cond_rules_file,
    stamped_cond_rules,
)
from core.storage_paths import config_file


class CondRulesStoreTests(unittest.TestCase):
    def test_control_flag_is_split_off_without_touching_the_input(self):
        payload = {'enabled': True, 'positive': [], 'negative': [], 'updatedAt': 5, '_manual': True}
        data, manual = split_control_flags(payload)
        self.assertTrue(manual)
        self.assertNotIn('_manual', data)
        self.assertEqual(data['updatedAt'], 5)          # updatedAt 은 데이터 — 부팅 최신 판단 근거
        self.assertIn('_manual', payload)
        self.assertEqual(split_control_flags(None), ({}, False))

    def test_autosave_is_silent_and_manual_save_is_reported(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = str(Path(tmp) / 'cond_rules.json')
            self.assertFalse(save_cond_rules_payload({'enabled': True, 'positive': [], 'negative': []}, path))
            self.assertTrue(save_cond_rules_payload({'enabled': False, 'positive': [], 'negative': [], '_manual': True}, path))
            with open(path, encoding='utf-8') as f:
                saved = json.load(f)
            self.assertEqual(saved, {'enabled': False, 'positive': [], 'negative': []})

    def test_path_is_the_config_boundary(self):
        self.assertEqual(cond_rules_path(), str(config_file(COND_RULES_NAME)))


class CondRulesStampTests(unittest.TestCase):
    """설정 백업으로 가져온 조건식은 가져온 시각을 찍어, 그 전에 편집한 브라우저 캐시보다 최신이 된다."""

    def test_stamped_copy_replaces_updated_at_and_drops_control_flags(self):
        source = {'enabled': True, 'positive': [{'condition': 'a', 'target': 'b'}], 'negative': [],
                  'updatedAt': 5, '_manual': True}
        stamped = stamped_cond_rules(source, 1_700_000_000_000)
        self.assertEqual(stamped['updatedAt'], 1_700_000_000_000)
        self.assertNotIn('_manual', stamped)
        self.assertEqual(stamped['positive'], source['positive'])
        self.assertEqual(source['updatedAt'], 5)                          # 입력은 그대로
        self.assertEqual(stamped_cond_rules({'positive': []}, 7)['updatedAt'], 7)   # 옛 백업 — 시각 없음
        self.assertIsNone(stamped_cond_rules([1, 2], 7))

    def test_file_stamp_uses_now_and_keeps_the_rules(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'cond_rules.json'
            # 옛 백업: BOM 이 붙고 updatedAt 이 없다
            path.write_bytes(('﻿' + json.dumps({'enabled': False, 'positive': [], 'negative': []})).encode('utf-8'))
            before = now_ms()
            self.assertTrue(stamp_cond_rules_file(str(path)))
            saved = json.loads(path.read_text(encoding='utf-8'))
            self.assertGreaterEqual(saved['updatedAt'], before)
            self.assertIs(saved['enabled'], False)
            self.assertTrue(stamp_cond_rules_file(str(path), stamp_ms=42))
            self.assertEqual(json.loads(path.read_text(encoding='utf-8'))['updatedAt'], 42)

    def test_unreadable_or_foreign_files_are_left_alone(self):
        with tempfile.TemporaryDirectory() as tmp:
            broken = Path(tmp) / 'broken.json'
            broken.write_text('{not json', encoding='utf-8')
            listing = Path(tmp) / 'list.json'
            listing.write_text('[1, 2]', encoding='utf-8')
            self.assertFalse(stamp_cond_rules_file(str(broken)))
            self.assertFalse(stamp_cond_rules_file(str(listing)))
            self.assertFalse(stamp_cond_rules_file(str(Path(tmp) / 'missing.json')))
            self.assertEqual(broken.read_text(encoding='utf-8'), '{not json')
            self.assertEqual(listing.read_text(encoding='utf-8'), '[1, 2]')


if __name__ == '__main__':
    unittest.main()
