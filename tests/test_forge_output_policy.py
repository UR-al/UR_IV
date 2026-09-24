"""Forge save_images 정책 — 숨은 출력 폴더 중복 저장 방지 + 사용자 옵트인 (감사 #122)."""
from __future__ import annotations

import json
import os
import re
import tempfile
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import core.forge_output_policy as forge_output_policy
from core.forge_output_policy import (
    PREF_KEY, _PrefsFlagCache, _PushedFlag, apply_save_policy, effective_save_images,
    forge_save_outputs_enabled, forge_save_outputs_setting, set_forge_save_outputs,
    update_forge_save_outputs_from_prefs,
)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _read(*parts: str) -> str:
    with open(os.path.join(ROOT, *parts), encoding="utf-8") as f:
        return f.read()


class PushedSettingTests(unittest.TestCase):
    """생성 워커 스레드는 ui_prefs.json 을 열지 않는다 — GUI 가 밀어 넣은 메모리 값만 읽는다.

    Windows 에선 읽기로 열린 파일을 os.replace 로 교체하지 못해(WinError 5), 워커의 읽기와
    GUI 의 save_ui_prefs 가 겹치면 설정 저장이 실패했다.
    """

    def setUp(self):
        forge_output_policy._setting.reset()
        self.addCleanup(forge_output_policy._setting.reset)

    def test_flag_uses_file_fallback_only_until_a_value_is_pushed(self):
        calls = []
        flag = _PushedFlag(lambda: calls.append(1) or True)
        self.assertFalse(flag.is_set())
        self.assertTrue(flag.get())
        self.assertEqual(len(calls), 1)
        flag.set(False)
        self.assertTrue(flag.is_set())
        self.assertFalse(flag.get())                  # 파일이 True 여도 밀어 넣은 값이 이긴다
        flag.set(True)
        self.assertTrue(flag.get())
        flag.set("true")                              # 불리언 True 만 켠다
        self.assertFalse(flag.get())
        self.assertEqual(len(calls), 1)               # 값을 받은 뒤로는 파일을 다시 안 연다
        flag.reset()
        self.assertTrue(flag.get())
        self.assertEqual(len(calls), 2)

    def test_pushed_setting_never_touches_the_prefs_file(self):
        set_forge_save_outputs(True)
        with mock.patch.object(forge_output_policy, "forge_save_outputs_from_prefs_file",
                               side_effect=AssertionError("worker must not open ui_prefs.json")), \
             mock.patch("builtins.open", side_effect=AssertionError("worker must not open files")):
            self.assertTrue(forge_save_outputs_setting())
            update_forge_save_outputs_from_prefs({PREF_KEY: False, "other": 1})
            self.assertFalse(forge_save_outputs_setting())
            update_forge_save_outputs_from_prefs(None)
            self.assertFalse(forge_save_outputs_setting())

    def test_setting_falls_back_to_the_file_before_the_first_push(self):
        with mock.patch.object(forge_output_policy, "forge_save_outputs_from_prefs_file",
                               return_value=True) as fallback:
            self.assertTrue(forge_save_outputs_setting())
            fallback.assert_called_once_with()

    def test_boot_restore_pushes_the_setting(self):
        from ui.generator_main import GeneratorMainUI
        host = SimpleNamespace(_apply_anima_guard_prefs=lambda _prefs: None)
        GeneratorMainUI._restore_runtime_prefs(host, {PREF_KEY: True})
        self.assertTrue(forge_output_policy._setting.is_set())
        self.assertTrue(forge_save_outputs_setting())
        GeneratorMainUI._restore_runtime_prefs(host, {"theme": "dark"})
        self.assertFalse(forge_save_outputs_setting())

    def test_save_ui_prefs_handler_pushes_the_merged_prefs_after_saving(self):
        source = _read("ui", "generator_main.py")
        block = source.split("elif action == 'save_ui_prefs':", 1)
        self.assertEqual(len(block), 2)
        handler = block[1].split("\n            elif action ==", 1)[0]
        saved_at = handler.find("save_ui_prefs(prefs_path, prefs)")
        pushed_at = handler.find("update_forge_save_outputs_from_prefs(prefs)")
        self.assertGreater(saved_at, 0)
        self.assertGreater(pushed_at, saved_at)

    def test_webui_generate_reads_only_the_pushed_setting(self):
        backend_source = _read("backends", "webui_backend.py")
        self.assertIn("forge_save_outputs_setting()", backend_source)
        self.assertIsNone(re.search(r"forge_save_outputs_from_prefs_file\(\)", backend_source))


class ForgeOutputPolicyTests(unittest.TestCase):
    def test_effective_save_images_requires_request_and_opt_in(self):
        self.assertFalse(effective_save_images(True, False))
        self.assertTrue(effective_save_images(True, True))
        self.assertFalse(effective_save_images(False, True))
        self.assertFalse(effective_save_images("true", True))   # 문자열은 요청으로 보지 않는다

    def test_apply_save_policy_returns_new_payload(self):
        payload = {"prompt": "p", "save_images": True}
        out = apply_save_policy(payload, False)
        self.assertIs(out["save_images"], False)
        self.assertIs(payload["save_images"], True)
        self.assertIs(apply_save_policy({}, True)["save_images"], False)

    def test_pref_reader_accepts_only_boolean_true(self):
        self.assertTrue(forge_save_outputs_enabled({PREF_KEY: True}))
        self.assertFalse(forge_save_outputs_enabled({PREF_KEY: "yes"}))
        self.assertFalse(forge_save_outputs_enabled({}))
        self.assertFalse(forge_save_outputs_enabled(None))

    def test_prefs_file_cache_follows_file_changes(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "ui_prefs.json"
            cache = _PrefsFlagCache(PREF_KEY, path_factory=lambda: path)
            self.assertFalse(cache.get())                        # 파일 없음
            path.write_text(json.dumps({PREF_KEY: True}), encoding="utf-8")
            self.assertTrue(cache.get())
            time.sleep(0.02)
            path.write_text(json.dumps({PREF_KEY: False, "other": 1}), encoding="utf-8")
            os.utime(path, (time.time() + 5, time.time() + 5))   # mtime 해상도와 무관하게 변경 보장
            self.assertFalse(cache.get())
            path.write_text("{broken", encoding="utf-8")
            os.utime(path, (time.time() + 10, time.time() + 10))
            self.assertFalse(cache.get())

    def test_settings_toggle_is_wired_to_the_same_pref_key(self):
        with open(os.path.join(ROOT, "frontend", "src", "views", "SettingsView.vue"), encoding="utf-8") as f:
            settings = f.read()
        self.assertIn(PREF_KEY, settings)
        # ui_prefs 쓰기 경로는 composables/uiPrefs.persistUiPrefs 한 곳(캐시 미러 + save_ui_prefs, 감사 #41)
        self.assertIn(f"persistUiPrefs({{ {PREF_KEY}:", settings)
        with open(os.path.join(ROOT, "frontend", "src", "composables", "uiPrefs.ts"), encoding="utf-8") as f:
            self.assertIn("requestAction('save_ui_prefs', payload)", f.read())


if __name__ == "__main__":
    unittest.main()
