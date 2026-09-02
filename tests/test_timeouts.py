from __future__ import annotations

import unittest
from pathlib import Path

from qorl.fixture import TaskSet
from qorl.timeouts import CalibratedTimeouts


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = Path("experiments/004-rl-run-v2/timeouts.json")


class CalibratedTimeoutTest(unittest.TestCase):
    def test_checked_manifest_covers_the_selected_400_tasks(self) -> None:
        timeouts = CalibratedTimeouts.load(
            ROOT, MANIFEST, TaskSet.load(ROOT, "ceb-v1")
        )

        self.assertEqual(len(timeouts.by_task_id), 400)
        self.assertEqual(len(timeouts.manifest_sha256), 64)
        self.assertEqual(
            timeouts.identity()["id"], "qorl-rl-run-v2-timeouts-v1"
        )
        replacement = timeouts.task("ceb-7a-7a14")
        self.assertEqual(replacement.timeout_ms, 5_000)
        self.assertEqual(replacement.calibrated_default_ms, 1362.7295)


if __name__ == "__main__":
    unittest.main()
