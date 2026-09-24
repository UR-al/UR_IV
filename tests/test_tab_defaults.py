"""Settings '기본값' 저장소(core/tab_defaults.py) — 부분 병합·정규화·전역 저장 연동 (audit #140)."""
import json
import os
import tempfile
import unittest

from core import tab_defaults
from core.tab_defaults import (
    REMOVED_TAB_DEFAULT_KEYS,
    load_tab_defaults,
    merge_tab_defaults,
    normalize_tab_defaults,
    save_tab_defaults_patch,
    sync_t2i_defaults,
)


class NormalizeTests(unittest.TestCase):
    # 프론트 frontend/src/utils/tabDefaults.test.ts 와 같은 골든 케이스 — 두 구현이 갈라지지 않게.
    def test_golden_cases_match_the_frontend(self):
        d = normalize_tab_defaults({'steps': 9999, 'width': 1000.6, 'yoloConf': 7, 'denoising': -1,
                                    'seed': '', 'cfg': 'abc'})
        self.assertEqual(d['steps'], 500)
        self.assertEqual(d['width'], 1001)
        self.assertEqual(d['yoloConf'], 1.0)
        self.assertEqual(d['denoising'], 0.0)
        self.assertEqual(d['seed'], '-1')
        self.assertNotIn('cfg', d)   # 깨진 값은 뺀다(기본값은 호출자가)
        self.assertEqual(normalize_tab_defaults({'steps': '28'})['steps'], 28)

    def test_retired_keys_are_dropped(self):
        self.assertEqual(REMOVED_TAB_DEFAULT_KEYS, {'defaultRating', 'negpip_enabled'})
        d = normalize_tab_defaults({'defaultRating': 'e', 'negpip_enabled': True, 'hires_enabled': 'true'})
        self.assertEqual(d, {'hires_enabled': True})

    def test_non_mapping_is_empty(self):
        self.assertEqual(normalize_tab_defaults([1, 2]), {})
        self.assertEqual(normalize_tab_defaults(None), {})


class PersistenceTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.path = os.path.join(self._tmp.name, 'tab_defaults.json')

    def _write(self, data):
        with open(self.path, 'w', encoding='utf-8') as f:
            json.dump(data, f)

    def _read(self):
        with open(self.path, 'r', encoding='utf-8') as f:
            return json.load(f)

    def test_patch_merges_instead_of_overwriting_fresh_t2i_values(self):
        """전역 저장이 steps 를 44 로 갱신한 뒤 Settings 가 denoising 만 바꿔도 steps 는 남는다."""
        self._write({'steps': 30, 'cfg': 7, 'ad_s1_enabled': True, 'defaultRating': 'g'})
        sync_t2i_defaults({'steps': '44', 'cfg': '6.5', 'width': '832', 'height': '1216', 'seed': ''}, self.path)
        save_tab_defaults_patch({'denoising': 0.5}, self.path)
        data = self._read()
        self.assertEqual(data['steps'], 44)
        self.assertEqual(data['cfg'], 6.5)
        self.assertEqual(data['width'], 832)
        self.assertEqual(data['denoising'], 0.5)
        self.assertTrue(data['ad_s1_enabled'], '화면에 없는 레거시 첫 실행 키도 보존')
        self.assertNotIn('defaultRating', data)
        self.assertNotIn('seed', data, "빈 seed 는 연동에서 건너뛴다")

    def test_sync_does_not_create_the_file(self):
        self.assertIsNone(sync_t2i_defaults({'steps': '20'}, self.path))
        self.assertFalse(os.path.exists(self.path))

    def test_unknown_patch_keys_are_ignored_and_unknown_file_keys_kept(self):
        self._write({'future_key': 1})
        result = save_tab_defaults_patch({'steps': 25, '_silent': True, 'evil': 'x'}, self.path)
        self.assertEqual(result, {'steps': 25})
        self.assertEqual(self._read(), {'future_key': 1, 'steps': 25})

    def test_load_normalizes_and_tolerates_corruption(self):
        self._write({'steps': '18', 'negpip_enabled': True})
        self.assertEqual(load_tab_defaults(self.path), {'steps': 18})
        with open(self.path, 'w', encoding='utf-8') as f:
            f.write('{broken')
        self.assertEqual(load_tab_defaults(self.path), {})

    def test_merge_is_pure(self):
        existing = {'steps': 20}
        merged = merge_tab_defaults(existing, {'steps': 30})
        self.assertEqual(existing, {'steps': 20})
        self.assertEqual(merged, {'steps': 30})

    def test_default_path_points_at_config(self):
        self.assertTrue(tab_defaults.default_tab_defaults_path().replace('\\', '/').endswith('config/tab_defaults.json'))


if __name__ == '__main__':
    unittest.main()
