from __future__ import annotations

import asyncio
import random
from contextlib import suppress
from threading import Event

import verifiers.v1 as vf

from qorl.agent.agent import QoAgentPolicy
from qorl.measure.rollout import RolloutEvaluator
from qorl.measure.schemas import RolloutRecord
from qorl.model.client import HttpTransport, LocalModelClient
from qorl.rl import runtime as shared_runtime
from qorl.rl.evidence import write_native
from qorl.rl.reward import scalar_reward
from qorl.rl.schemas import (
    QorlHarnessConfig,
    QorlTaskData,
    RlRolloutRecord,
)
from qorl.util.hashing import sha256_json
from qorl.util.seeds import derive_seed


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
        task_data = QorlTaskData.model_validate(data.model_dump())
        index = task_data.rollout_index
        trace.info["qorl_seeds"] = {
            "rollout_index": index,
            "model": derive_seed(
                self.config.seed, "rl-model", task_data.task_id, str(index)
            ),
            "pairs": derive_seed(
                self.config.seed, "rl-pairs", task_data.task_id, str(index)
            ),
        }
        cancel = Event()
        work = asyncio.create_task(
            asyncio.to_thread(self._run, ctx, trace, endpoint, secret, data, cancel)
        )
        active = shared_runtime.current()
        active.work[work] = cancel
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
        finally:
            active.work.pop(work, None)
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
        model, inference = self.config.model, self.config.inference
        if model is None or inference is None:
            raise ValueError(
                "RL execution requires resolved model and inference settings"
            )
        active = shared_runtime.current()
        task_data = QorlTaskData.model_validate(data.model_dump())
        task = next(
            task for task in active.task_set.tasks if task.task_id == task_data.task_id
        )
        client = LocalModelClient(
            model,
            inference,
            served_model_name=ctx.model,
            transport=HttpTransport(
                model.model_copy(update={"base_url": endpoint.rstrip("/")}),
                api_key=secret,
                semaphore=active.requests,
            ),
        )
        policy = QoAgentPolicy(
            client,
            self.config.agent,
            context_length=model.context_length,
            max_tokens=inference.max_tokens,
            seed=None,  # The environment supplies the recorded native per-rollout seed.
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
                if self.config.evidence_directory is not None:
                    write_native(
                        self.config.evidence_directory
                        / "rollouts"
                        / f"{sha256_json(trace.id)}.msgpack",
                        trace,
                    )

            try:
                evaluator.start()
                policy_trace = policy.search(evaluator)
                trace.info["qorl_policy"] = policy_trace.model_dump(mode="json")
                evaluator.finish(
                    random.Random(trace.info["qorl_seeds"]["pairs"]),
                    selected_candidate_id=policy_trace.selection.selected_candidate_id,
                )
                store_record(evaluator.record())
            except BaseException as error:
                if policy.trace is not None:
                    trace.info["qorl_policy"] = policy.trace.model_dump(mode="json")
                store_record(evaluator.record(error))
                raise
