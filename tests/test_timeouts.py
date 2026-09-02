from __future__ import annotations

import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

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

    def test_legacy_manifest_runtime_image_can_be_enforced(self) -> None:
        task_set = TaskSet.load(ROOT, "ceb-v1")
        with self.assertRaisesRegex(RuntimeError, "different runtime"):
            CalibratedTimeouts.load(
                ROOT,
                MANIFEST,
                task_set,
                {
                    "postgres_image_id": "sha256:different-runtime",
                    "benchmark_config_id": "benchmark-v1",
                },
            )

    def test_current_manifest_enforces_runtime_separately_from_data(self) -> None:
        task_set = TaskSet.load(ROOT, "ceb-v1")
        document = json.loads((ROOT / MANIFEST).read_text(encoding="utf-8"))
        document["data_identity"] = task_set.data_identity
        document["runtime_identity"] = {
            "postgres_image_id": "sha256:current-runtime",
            "benchmark_config_id": "benchmark-v2",
        }
        document.pop("database")
        with TemporaryDirectory() as temporary:
            path = Path(temporary) / "timeouts.json"
            path.write_text(json.dumps(document), encoding="utf-8")
            loaded = CalibratedTimeouts.load(
                ROOT, path, task_set, document["runtime_identity"]
            )
            self.assertEqual(len(loaded.by_task_id), 400)
            with self.assertRaisesRegex(RuntimeError, "different runtime"):
                CalibratedTimeouts.load(
                    ROOT,
                    path,
                    task_set,
                    {
                        "postgres_image_id": "sha256:different-runtime",
                        "benchmark_config_id": "benchmark-v2",
                    },
                )


if __name__ == "__main__":
    unittest.main()
