"""The native Verifiers environment owns one PostgreSQL pool for all rollouts."""

import asyncio
from dataclasses import replace
from typing import Protocol

import verifiers.v1 as vf

from qorl.postgres.config import PostgresConfig
from qorl.rl import runtime
from qorl.rl.schemas import (
    QorlEnvironmentConfig,
    QorlHarnessConfig,
    QorlTaskData,
    QorlTasksetConfig,
)
from qorl.rl.tasks import QorlTask
from qorl.taskset.schemas import TaskSelection
from qorl.taskset.taskset import TaskSet
from qorl.util.seeds import derive_seed
from qorl.worker_pool.config import load_pool_config


class RolloutAgent(Protocol):
    """The native bound agent surface used for a single QORL conversation."""

    ctx: vf.ModelContext
    config: vf.AgentConfig

    async def run(self, task: QorlTask) -> object: ...


class QorlEnvironment(vf.Env[QorlEnvironmentConfig]):
    """One agent per episode, with shared resources drained before teardown."""

    def __init__(self, config: QorlEnvironmentConfig) -> None:
        super().__init__(config)
        self._rollouts: dict[str, int] = {}

    async def run(self, task: vf.Task[vf.TaskData], agents: vf.Agents) -> None:
        data = QorlTaskData.model_validate(task.data.model_dump())
        index = self._rollouts.get(data.task_id, 0)
        self._rollouts[data.task_id] = index + 1
        config = self.config.agent.harness
        if not isinstance(config, QorlHarnessConfig):
            raise ValueError("QORL requires its configured harness")
        seed = derive_seed(config.seed, "rl-model", data.task_id, str(index))
        await run_seeded_agent(
            agents.agent,
            QorlTask(data.model_copy(update={"rollout_index": index}), task.config),
            seed,
        )

    async def start(self) -> None:
        selected = self.config.taskset
        if not isinstance(selected, QorlTasksetConfig):
            raise ValueError("QORL requires its selected task adapter")
        config = self.config
        selection_path = selected.selection
        if (
            config.postgres_config is None
            or config.pool_config is None
            or selection_path is None
        ):
            raise ValueError(
                "RL execution requires selected tasks and PostgreSQL/pool configuration"
            )
        repository = selected.repository.resolve()
        selection = TaskSelection.model_validate_json(
            (repository / selection_path).read_bytes()
        )
        task_set = TaskSet.load(repository, selection.benchmark_id.value)
        task_set.resolve(selection)
        harness = self.config.agent.harness
        if not isinstance(harness, QorlHarnessConfig) or harness.model is None:
            raise ValueError("RL execution requires resolved model settings")
        work = asyncio.create_task(
            asyncio.to_thread(
                runtime.start,
                repository,
                task_set,
                PostgresConfig.load(repository / config.postgres_config),
                load_pool_config(repository / config.pool_config),
                harness.model.max_concurrent_requests,
            )
        )
        try:
            await asyncio.shield(work)
        except asyncio.CancelledError:
            while not work.done():
                try:
                    await asyncio.shield(work)
                except asyncio.CancelledError:
                    continue
            try:
                work.result()
            finally:
                await self.stop()
            raise

    async def stop(self) -> None:
        await runtime.drain()
        await asyncio.to_thread(runtime.stop)


async def run_seeded_agent(agent: RolloutAgent, task: QorlTask, seed: int) -> None:
    """Set sampling before the native training proxy acquires its session."""
    sampling = vf.SamplingConfig.model_validate(
        {**agent.ctx.sampling.model_dump(), "seed": seed}
    )
    agent.ctx = replace(agent.ctx, sampling=sampling)
    agent.config = agent.config.model_copy(update={"sampling": sampling})
    await agent.run(task)
