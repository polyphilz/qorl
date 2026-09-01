from __future__ import annotations

import asyncio
import json
import random
from dataclasses import replace
from pathlib import Path

import verifiers.v1 as vf

from qorl.agent import QoAgentConfig, QoAgentPolicy
from qorl.agent.client import OpenAIModelClient
from qorl.agent.tools import candidate_feedback
from qorl.rollout import (
    TrainingRolloutEvaluatorV1,
    TrainingRolloutEvaluatorV2,
)
import qorl_training.runtime as runtime


class QorlHarnessConfig(vf.HarnessConfig):
    id: str = "qorl-training"
    run_config: Path = Path("configs/evaluation/run-v1.json")
    context_length: int = 20_480


class QorlHarness(vf.Harness[QorlHarnessConfig]):
    """Run qo-agent in-process while Prime-RL intercepts every model turn."""

    EXECUTES_CODE = False
    NEEDS_CONTAINER = False

    async def launch(
        self,
        ctx: vf.ModelContext,
        trace: vf.Trace,
        sandbox: vf.Runtime,
        endpoint: str,
        secret: str,
        mcp_urls: dict[str, str],
        data: vf.TaskData,
    ) -> vf.ProgramResult:
        del sandbox, mcp_urls
        await asyncio.to_thread(self._run, ctx, trace, endpoint, secret, data)
        return vf.ProgramResult(exit_code=0, stdout="", stderr="")

    def _run(
        self,
        ctx: vf.ModelContext,
        trace: vf.Trace,
        endpoint: str,
        secret: str,
        data: vf.TaskData,
    ) -> None:
        active = runtime.current()
        task = next(
            task
            for task in active.task_set.inventory["tasks"]
            if task["task_id"] == data.task_id
        )
        config_path = self.config.run_config
        if not config_path.is_absolute():
            config_path = active.task_set.repository / config_path
        policy_data = json.loads(config_path.read_text(encoding="utf-8"))["policy"]
        policy_config = replace(
            QoAgentConfig.from_dict(policy_data),
            model=ctx.model,
            base_url=endpoint.rstrip("/"),
            context_length=self.config.context_length,
            seed=None,
        )
        client = OpenAIModelClient(
            policy_config.base_url,
            policy_config.request_timeout_seconds,
            api_key=secret,
        )

        with active.claim_worker() as slot:
            if active.calibrated_timeouts is None:
                evaluator = TrainingRolloutEvaluatorV1(
                    slot.worker, active.task_set, task
                )
            else:
                evaluator = TrainingRolloutEvaluatorV2(
                    slot.worker,
                    active.task_set,
                    task,
                    active.calibrated_timeouts.task(task["task_id"]),
                    active.calibrated_timeouts.manifest["manifest_id"],
                )
            baseline = evaluator.start()
            policy_trace = QoAgentPolicy(policy_config, client).search(evaluator)
            final = evaluator.finish(
                random.Random(f"qorl-rl:{task['task_id']}:{trace.id}:pairs")
            )

        trace.info["qorl"] = {
            "task_id": task["task_id"],
            "template_id": task["template_id"],
            "database_pool": active.pool_manifest(),
            "database_worker": slot.resources.manifest(),
            "candidate_timeout_manifest": (
                active.calibrated_timeouts.identity()
                if active.calibrated_timeouts is not None
                else None
            ),
            "measurement_protocol": evaluator.measurement_protocol.manifest(),
            "default": {
                "plan_sha256": baseline["plan_sha256"],
                "median_execution_time_ms": baseline[
                    "median_execution_time_ms"
                ],
                "candidate_timeout": baseline["candidate_timeout"],
            },
            "candidates": [
                {**candidate_feedback(candidate), "action": candidate["action"]}
                for candidate in evaluator.candidates
            ],
            "final": final,
            "policy": {
                "stop_reason": policy_trace["stop_reason"],
                "tools_sha256": policy_trace["tools_sha256"],
                "usage": policy_trace["usage"],
                "context_estimate_tokens": policy_trace[
                    "context_estimate_tokens"
                ],
            },
        }
