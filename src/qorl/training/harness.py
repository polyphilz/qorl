from __future__ import annotations

import asyncio
import json
import random
from contextlib import suppress
from dataclasses import replace
from pathlib import Path
from threading import Event

import verifiers.v1 as vf

from qorl.agent import QoAgentConfig, QoAgentPolicy
from qorl.agent.client import OpenAIModelClient
from qorl.agent.schemas import AgentSettings
from qorl.measure.rollout import RolloutEvaluator
from qorl.measure.schemas import RolloutMeasurementSettings, RolloutRecord
from qorl.paths import REPOSITORY_ROOT
from qorl.rl.reward import scalar_reward
from qorl.rl.schemas import AnchoredGrpoSettings, RlRolloutRecord, RlSettings
from qorl.training import runtime as shared_runtime
from qorl.training.taskset import QorlTaskData


class QorlHarnessConfig(vf.HarnessConfig):
    """Allow Verifiers' fallback construction; experiments supply resolved settings."""

    id: str = "qorl"
    run_config: Path = Path("model/configs/000-modelconf/modelconf.json")
    context_length: int = 20_480
    agent: AgentSettings = AgentSettings(
        candidate_attempts=1,
        maximum_model_turns=64,
        inspection_turns_per_alias=3,
    )
    measurement: RolloutMeasurementSettings = RolloutMeasurementSettings(
        default_warmups=1,
        default_measurements=1,
        paired_warmups=1,
        paired_measurements=3,
        default_timeout_seconds=300.0,
        candidate_timeout_floor_seconds=5.0,
        candidate_timeout_multiplier=3.0,
    )
    rl: RlSettings = RlSettings(
        algorithm=AnchoredGrpoSettings(
            type="qorl_anchored_grpo",
            tau=0.05,
            c=0.10,
            d=0.02,
            t=0.10,
            min_peers=2,
        )
    )


class QorlHarness(vf.Harness[QorlHarnessConfig]):
    """Run qo-agent in-process while Prime-RL intercepts every model turn."""

    EXECUTES_CODE = False
    NEEDS_CONTAINER = False

    async def launch(
        self,
        ctx: vf.ModelContext,
        trace: vf.Trace[vf.TaskData],
        runtime: vf.Runtime,
        endpoint: str,
        secret: str,
        mcp_urls: dict[str, str],
        data: vf.TaskData,
    ) -> vf.ProgramResult:
        del runtime, mcp_urls
        cancel = Event()
        work = asyncio.create_task(
            asyncio.to_thread(self._run, ctx, trace, endpoint, secret, data, cancel)
        )
        try:
            await asyncio.shield(work)
        except asyncio.CancelledError:
            cancel.set()
            # Repeated cancellation must not release the slot while its thread is active.
            with suppress(Exception, asyncio.CancelledError):
                while not work.done():
                    with suppress(asyncio.CancelledError):
                        await asyncio.shield(work)
                work.result()
            raise
        return vf.ProgramResult(exit_code=0, stdout="", stderr="")

    def _run(
        self,
        ctx: vf.ModelContext,
        trace: vf.Trace[vf.TaskData],
        endpoint: str,
        secret: str,
        data: vf.TaskData,
        cancel: Event,
    ) -> None:
        active = shared_runtime.current()
        task_data = QorlTaskData.model_validate(data.model_dump())
        task = next(
            task for task in active.task_set.tasks if task.task_id == task_data.task_id
        )
        config_path = self.config.run_config
        if not config_path.is_absolute():
            config_path = REPOSITORY_ROOT / config_path
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
            evaluator = RolloutEvaluator(
                slot.client,
                active.task_set,
                task,
                measurement=self.config.measurement,
                max_candidates=self.config.agent.candidate_attempts,
                cancel=cancel,
            )

            def store_record(record: RolloutRecord) -> None:
                reward = (
                    scalar_reward(record, self.config.rl.reward)
                    if record.final is not None and self.config.rl.reward is not None
                    else None
                )
                result = RlRolloutRecord(
                    **record.model_dump(),
                    database_pool=active.pool_manifest(),
                    database_worker=slot.resources.manifest(),
                    scalar_reward=reward,
                )
                trace.info["qorl"] = result.to_wire()

            try:
                evaluator.start()
                policy_trace = QoAgentPolicy(policy_config, client).search(
                    evaluator, settings=self.config.agent
                )
                trace.info["qorl_policy"] = policy_trace
                evaluator.finish(
                    random.Random(f"qorl-rl:{task.task_id}:{trace.id}:pairs")
                )
                store_record(evaluator.record())
            except BaseException as error:
                store_record(evaluator.record(error))
                raise
