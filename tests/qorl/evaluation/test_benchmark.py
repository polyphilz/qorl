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
from qorl.measure.schemas import NoValidCandidateOutcome, RolloutRecord
from qorl.rl.schemas import RlRolloutRecord


class TestBenchmark:
    def test_task_keeps_one_claimed_worker_for_its_rollout(
        self, rl_rollout_record: RlRolloutRecord
    ) -> None:
        resources = Mock()
        resources.manifest.return_value = rl_rollout_record.database_worker
        slot = SimpleNamespace(resources=resources, client=object())
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
            claimed, result = run_task_on_worker(
                pool, task_set, task, policy, agent, rl_rollout_record.measurement
            )

        assert claimed is slot
        assert result["worker"] == rl_rollout_record.database_worker.model_dump()
        run_task.assert_called_once_with(
            slot.client, task_set, task, policy, agent, rl_rollout_record.measurement
        )

    def test_run_config_loads_policy_without_schema_version(
        self, tmp_path: Path
    ) -> None:
        policy_path = tmp_path / "experiments/000-test/random-policy.json"
        policy_path.parent.mkdir(parents=True)
        policy_path.write_text(
            json.dumps(
                {
                    "policy": {
                        "type": "random_structured_action",
                        "seed": 7,
                    },
                }
            )
        )
        run_path = tmp_path / "experiments/000-test/run.json"
        run_path.write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "run_id_prefix": "test",
                    "policy_config": "experiments/000-test/random-policy.json",
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

    def test_default_run_config_resolves_existing_model_config(
        self, repository_root: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("QORL_RUN_CONFIG", raising=False)
        path, config = load_run_config(repository_root)

        assert path == repository_root / "experiments/000-vanilla-baseline/run.json"
        assert config["_policy_config_path"] == (
            repository_root / "model/configs/000-modelconf/modelconf.json"
        )
        assert config["policy"]["context_length"] == 262_144
        assert config["policy"]["sampling"]["presence_penalty"] == 2.0

    def test_random_run_config_resolves_its_local_policy(
        self, repository_root: Path
    ) -> None:
        _, config = load_run_config(
            repository_root, "experiments/000-vanilla-baseline/random.json"
        )
        assert config["policy"] == {
            "type": "random_structured_action",
            "seed": 20260827,
        }
        assert config["_policy_config_path"] == (
            repository_root / "experiments/000-vanilla-baseline/random-policy.json"
        )

    def test_summary_reports_primary_metrics(
        self, rollout_record: RolloutRecord
    ) -> None:
        invalid = RolloutRecord.model_validate(
            {
                **rollout_record.model_dump(),
                "candidates": [],
                "final": NoValidCandidateOutcome(),
            }
        )
        results = [
            {"rollout": rollout_record.to_wire()},
            {"rollout": invalid.to_wire()},
            {"status": "failed"},
        ]
        summary = summarize(results)
        assert summary["scored_task_count"] == 1
        assert summary["failure_count"] == 1
        assert summary["no_valid_candidate_count"] == 1
        assert summary["geometric_mean_speedup"] == 2.0
        assert summary["candidate_workload_time_ms"] == 5.0
        assert summary["default_workload_time_ms"] == 10.0
