from __future__ import annotations

import tempfile
import unittest
from unittest.mock import patch

from core.app_instance import (
    _registry_dir,
    live_app_instance_pids,
    register_app_instance,
    unregister_app_instance,
)
from core.app_update_lock import UpdateLockBusy


class AppInstanceRegistryTests(unittest.TestCase):
    def test_register_list_and_unregister_are_project_scoped(self):
        with tempfile.TemporaryDirectory() as temp:
            marker = register_app_instance(temp, pid=43210)
            self.assertTrue(marker.is_file())
            with patch("core.app_instance.process_exists", return_value=True):
                self.assertEqual(live_app_instance_pids(temp), [43210])
                self.assertEqual(live_app_instance_pids(temp, exclude_pid=43210), [])
            unregister_app_instance(temp, pid=43210)
            self.assertFalse(marker.exists())

    def test_dead_instance_marker_is_cleaned(self):
        with tempfile.TemporaryDirectory() as temp:
            marker = register_app_instance(temp, pid=43211)
            with patch("core.app_instance.process_exists", return_value=False):
                self.assertEqual(live_app_instance_pids(temp), [])
            self.assertFalse(marker.exists())

    def test_reused_pid_does_not_keep_a_stale_instance_marker_live(self):
        with tempfile.TemporaryDirectory() as temp:
            with patch("core.app_instance.process_start_identity", return_value="old"):
                marker = register_app_instance(temp, pid=43212)
            with (
                patch("core.app_instance.process_exists", return_value=True),
                patch("core.app_instance.process_start_identity", return_value="new"),
            ):
                self.assertEqual(live_app_instance_pids(temp), [])
            self.assertFalse(marker.exists())

    def test_live_update_lock_prevents_a_new_instance(self):
        with tempfile.TemporaryDirectory() as temp:
            with (
                patch(
                    "core.app_update_lock.acquire_update_lock",
                    side_effect=UpdateLockBusy("busy"),
                ) as acquire,
                self.assertRaises(UpdateLockBusy),
            ):
                register_app_instance(temp, pid=43213)
            acquire.assert_called_once_with(temp, timeout=120.0)
            self.assertFalse((_registry_dir(temp) / "43213.json").exists())

    def test_update_guarded_entrypoint_does_not_reacquire_the_lock(self):
        with tempfile.TemporaryDirectory() as temp:
            with patch("core.app_update_lock.acquire_update_lock") as acquire:
                marker = register_app_instance(temp, pid=43214, update_guarded=True)
            acquire.assert_not_called()
            self.assertTrue(marker.is_file())


if __name__ == "__main__":
    unittest.main()
