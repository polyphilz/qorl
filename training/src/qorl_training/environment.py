from __future__ import annotations

import asyncio
from pathlib import Path
from typing import cast

from verifiers.v1.envs.single_agent import SingleAgentEnv

import qorl_training.runtime as runtime
from qorl_training.taskset import QorlTaskset


class QorlEnvironment(SingleAgentEnv):
    """Own the persistent PostgreSQL worker pool used by QORL episodes."""

    async def start(self) -> None:
        repository = Path(cast(QorlTaskset, self.taskset).config.repository).resolve()
        await asyncio.to_thread(runtime.start, repository)

    async def stop(self) -> None:
        await asyncio.to_thread(runtime.stop)
