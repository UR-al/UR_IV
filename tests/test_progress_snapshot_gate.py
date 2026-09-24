"""core.progress_snapshot_gate — progress 이벤트에 스냅샷은 단계가 바뀔 때·일정 간격으로만."""
from __future__ import annotations

import unittest

from core.progress_snapshot_gate import ProgressSnapshotGate


class ProgressSnapshotGateTests(unittest.TestCase):
    def test_first_update_and_each_phase_change_are_due(self):
        now = [0.0]
        gate = ProgressSnapshotGate(interval=2.0, clock=lambda: now[0])
        self.assertTrue(gate.due({"phase": "download"}))
        self.assertFalse(gate.due({"phase": "download"}))
        now[0] += 0.5
        self.assertFalse(gate.due({"phase": "download", "message": "more output"}))
        self.assertTrue(gate.due({"phase": "venv"}))
        self.assertTrue(gate.due({"stage": "checking"}), "app update 는 stage 키를 쓴다")

    def test_same_phase_is_refreshed_after_the_interval(self):
        now = [10.0]
        gate = ProgressSnapshotGate(interval=2.0, clock=lambda: now[0])
        self.assertTrue(gate.due({"phase": "health"}))
        now[0] += 1.9
        self.assertFalse(gate.due({"phase": "health"}))
        now[0] += 0.2
        self.assertTrue(gate.due({"phase": "health"}))

    def test_updates_without_a_phase_are_throttled_by_time(self):
        now = [0.0]
        gate = ProgressSnapshotGate(interval=2.0, clock=lambda: now[0])
        self.assertTrue(gate.due({"message": "a"}))
        self.assertFalse(gate.due({"message": "b"}))
        self.assertFalse(gate.due("not a mapping"))
        now[0] += 3
        self.assertTrue(gate.due({"message": "c"}))


if __name__ == "__main__":
    unittest.main()
