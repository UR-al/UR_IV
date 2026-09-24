from __future__ import annotations

import json
import os
import tempfile
import time
import unittest
import zipfile
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from core.settings_backup import (
    SettingsBackupError,
    export_settings_archive,
    import_settings_archive,
    resolve_import_target,
)
from ui.generator_main import GeneratorMainUI


class SettingsBackupTests(unittest.TestCase):
    def test_round_trip_includes_config_and_nested_creator_document(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "project"
            (root / "config").mkdir(parents=True)
            (root / "user_data/creator").mkdir(parents=True)
            prompt = root / "config/prompt_settings.json"
            comic = root / "user_data/creator/comic_studio.json"
            prompt.write_text('{"steps": 24}', encoding="utf-8")
            comic.write_text('{"title": "test"}', encoding="utf-8")
            archive = Path(temporary) / "settings.zip"

            self.assertEqual(export_settings_archive(archive, project_root=root), 2)
            prompt.unlink()
            comic.unlink()

            self.assertEqual(import_settings_archive(archive, project_root=root), 2)
            self.assertEqual(prompt.read_text(encoding="utf-8"), '{"steps": 24}')
            self.assertEqual(comic.read_text(encoding="utf-8"), '{"title": "test"}')

    def test_imported_cond_rules_get_the_import_time_so_an_older_browser_cache_cannot_revert_them(self) -> None:
        """백업 안의 updatedAt 은 내보낸 때(옛 백업은 없음) — 그대로 두면 그 뒤에 편집한 localStorage
        캐시가 재시작 부팅(pickCondRulesSource)에서 이겨 가져온 규칙을 조용히 되덮었다."""
        rules = {"enabled": True, "positive": [{"condition": "a", "target": "b"}], "negative": []}
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "project"
            (root / "config").mkdir(parents=True)
            current = root / "config/cond_rules.json"
            current.write_text(json.dumps({**rules, "positive": [], "updatedAt": 9_000}), encoding="utf-8")
            for name, payload in (
                ("exported_earlier.zip", {**rules, "updatedAt": 1_000}),
                ("legacy_without_stamp.zip", rules),
            ):
                archive = Path(temporary) / name
                with zipfile.ZipFile(archive, "w") as handle:
                    handle.writestr("config/cond_rules.json", json.dumps(payload))
                    handle.writestr("config/ui_prefs.json", '{"theme": "dark"}')
                before = int(time.time() * 1000)
                self.assertEqual(import_settings_archive(archive, project_root=root), 2)
                imported = json.loads(current.read_text(encoding="utf-8"))
                self.assertGreaterEqual(imported["updatedAt"], before, name)
                self.assertEqual(imported["positive"], rules["positive"], name)
                # 다른 파일은 바이트 그대로
                self.assertEqual((root / "config/ui_prefs.json").read_text(encoding="utf-8"), '{"theme": "dark"}')
            leftovers = [p.name for p in (root / "config").iterdir() if ".import-" in p.name or p.suffix == ".tmp"]
            self.assertEqual(leftovers, [])

    def test_traversal_and_windows_alternate_stream_names_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            with self.assertRaises(SettingsBackupError):
                resolve_import_target(root, "wildcards/../outside.txt")
            with self.assertRaises(SettingsBackupError):
                resolve_import_target(root, "wildcards/file:stream.txt")

    def test_import_rejects_entries_over_the_size_limit_before_writing(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "project"
            root.mkdir()
            archive = Path(temporary) / "oversized.zip"
            with zipfile.ZipFile(archive, "w") as handle:
                handle.writestr("config/ui_prefs.json", b"1234")

            with mock.patch("core.settings_backup.MAX_BACKUP_FILE_BYTES", 3):
                with self.assertRaises(SettingsBackupError):
                    import_settings_archive(archive, project_root=root)
            self.assertFalse((root / "config/ui_prefs.json").exists())

    def test_import_rejects_a_symlink_escape_when_supported(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            root = base / "project"
            external = base / "external"
            (root / "wildcards").mkdir(parents=True)
            external.mkdir()
            link = root / "wildcards/link"
            try:
                os.symlink(external, link, target_is_directory=True)
            except OSError as exc:
                self.skipTest(f"directory symlink unavailable: {exc}")
            archive = base / "escape.zip"
            with zipfile.ZipFile(archive, "w") as handle:
                handle.writestr("wildcards/link/victim.txt", "unsafe")

            with self.assertRaises(SettingsBackupError):
                import_settings_archive(archive, project_root=root)
            self.assertFalse((external / "victim.txt").exists())

    def test_imported_settings_skip_the_next_shutdown_autosave(self) -> None:
        subject = SimpleNamespace(
            _preserve_imported_settings_on_quit=True,
            save_settings=mock.Mock(),
            ui_state=SimpleNamespace(save_all=mock.Mock()),
        )

        # 세션 백업 정상 종료 표시는 가짜로 — 실제 cache/session/session_backup.json 을 건드리지 않는다
        with mock.patch("core.session_backup.mark_session_clean") as mark:
            self.assertFalse(GeneratorMainUI._save_shutdown_state(subject))
        subject.save_settings.assert_not_called()
        subject.ui_state.save_all.assert_not_called()
        mark.assert_called_once_with()   # 가져오기는 의도한 교체 — 크래시 복구 제안이 뜨지 않게

    def test_restart_after_import_goes_through_the_preserving_quit_path(self) -> None:
        """Vue '앱 재시작'·복원 후 재시작은 _quit_app(→ _save_shutdown_state) 경로라 가져온 파일을 지킨다."""
        from ui.settings_data_actions import restart_app

        notices = []
        subject = SimpleNamespace(
            _preserve_imported_settings_on_quit=True,
            _quit_app=mock.Mock(),
            vue_bridge=SimpleNamespace(showNotification=SimpleNamespace(emit=lambda *a: notices.append(a))),
        )
        scheduled = []
        spawn = mock.Mock()
        self.assertTrue(restart_app(subject, spawn=spawn, schedule=lambda ms, fn: scheduled.append((ms, fn))))
        spawn.assert_called_once()
        self.assertEqual(scheduled[0][1], subject._quit_app)
        self.assertIn("가져온 설정을 유지한 채", notices[0][1])

    def test_export_includes_user_authored_config_and_chat_only_when_asked(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "project"
            (root / "config").mkdir(parents=True)
            (root / "user_data").mkdir(parents=True)
            for name in ("instruction_presets.json", "comfy_workflow_controls.json",
                         "comfy_quality_preset.json", "chat_threads.json"):
                (root / "config" / name).write_text("{}", encoding="utf-8")
            (root / "user_data" / "prompt_presets.json").write_text("{}", encoding="utf-8")
            plain = Path(temporary) / "plain.zip"
            with_chat = Path(temporary) / "chat.zip"
            export_settings_archive(plain, project_root=root)
            export_settings_archive(with_chat, project_root=root, include_chat=True)
            with zipfile.ZipFile(plain) as handle:
                names = set(handle.namelist())
            self.assertIn("config/instruction_presets.json", names)
            self.assertIn("config/comfy_workflow_controls.json", names)
            self.assertIn("config/comfy_quality_preset.json", names)
            self.assertNotIn("config/chat_threads.json", names)
            self.assertNotIn("user_data/prompt_presets.json", names, "어디서도 읽지 않는 옛 파일")
            with zipfile.ZipFile(with_chat) as handle:
                self.assertIn("config/chat_threads.json", handle.namelist())

            (root / "config" / "chat_threads.json").unlink()
            self.assertGreaterEqual(import_settings_archive(with_chat, project_root=root), 1)
            self.assertTrue((root / "config" / "chat_threads.json").exists(), "대화 기록은 가져오기 허용 목록에 있다")

    def test_settings_save_is_locked_until_restart_after_import(self) -> None:
        from ui.generator_settings import SettingsMixin
        from ui.settings_data_actions import should_skip_persist_action

        subject = SimpleNamespace(_preserve_imported_settings_on_quit=True, _build_settings_dict=mock.Mock())
        self.assertFalse(SettingsMixin.save_settings(subject))
        subject._build_settings_dict.assert_not_called()
        self.assertTrue(should_skip_persist_action(subject, "save_ui_prefs"))
        self.assertTrue(should_skip_persist_action(subject, "save_tab_defaults"))
        self.assertFalse(should_skip_persist_action(subject, "generate"))
        self.assertFalse(should_skip_persist_action(SimpleNamespace(), "save_ui_prefs"))


if __name__ == "__main__":
    unittest.main()
