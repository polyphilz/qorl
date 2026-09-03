from __future__ import annotations

import json
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest

from qorl.evaluation.benchmark import (
    load_run_config,
    run_task_on_worker,
    summarize,
)


class TestBenchmark:
    def test_task_keeps_one_claimed_worker_for_its_rollout(self) -> None:
        resources = Mock()
        resources.manifest.return_value = {"slot": 2}
        slot = SimpleNamespace(resources=resources, worker=object())
        pool = Mock()
        pool.claim_worker.return_value = nullcontext(slot)
        task_set = object()
        task = {"task_id": "job-01a"}
        policy = {"type": "qo_agent"}
        agent = object()

        with patch(
            "qorl.evaluation.benchmark.run_task",
            return_value={"status": "completed"},
        ) as run_task:
            claimed, result = run_task_on_worker(pool, task_set, task, policy, agent)

        assert claimed is slot
        assert result["worker"] == {"slot": 2}
        run_task.assert_called_once_with(slot.worker, task_set, task, policy, agent)

    def test_run_config_loads_its_shared_policy(self, tmp_path: Path) -> None:
        policy_path = tmp_path / "configs/policy/test.json"
        policy_path.parent.mkdir(parents=True)
        policy_path.write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "policy": {
                        "type": "random_structured_action",
                        "seed": 7,
                    },
                }
            )
        )
        run_path = tmp_path / "experiments/000-test/run.json"
        run_path.parent.mkdir(parents=True)
        run_path.write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "run_id_prefix": "test",
                    "policy_config": "configs/policy/test.json",
                }
            )
        )

        loaded_path, config = load_run_config(tmp_path, str(run_path))

        assert loaded_path == run_path
        assert config["run_id_prefix"] == "test"
        assert config["policy"]["seed"] == 7
        assert config["_policy_config_path"] == policy_path

        policy = json.loads(policy_path.read_text())
        del policy["policy"]["seed"]
        policy_path.write_text(json.dumps(policy))
        with pytest.raises(RuntimeError, match="integer seed"):
            load_run_config(tmp_path, str(run_path))

    def test_summary_reports_primary_metrics(self) -> None:
        results = [
            {
                "candidates": [{"constraints_satisfied": True, "duplicate_of": None}],
                "final": {
                    "status": "completed",
                    "score": 2.0,
                    "candidate_median_execution_time_ms": 5.0,
                    "default_median_execution_time_ms": 10.0,
                },
            },
            {
                "candidates": [{"constraints_satisfied": False, "duplicate_of": None}],
                "final": {"status": "no_valid_candidate"},
            },
        ]
        summary = summarize(results)
        assert summary["scored_task_count"] == 1
        assert summary["failure_rate"] == 0.5
        assert summary["geometric_mean_speedup"] == 2.0
        assert summary["invalid_attempt_count"] == 1
