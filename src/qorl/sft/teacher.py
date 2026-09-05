from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from pathlib import Path

from qorl.agent import QoAgentConfig, QoAgentPolicy
from qorl.agent.client import (
    ModelClient,
    ModelError,
    ModelRequestError,
    OpenAIModelClient,
)
from qorl.agent.types import ToolName
from qorl.db.fixture import DatabaseFixture
from qorl.db.pool import WorkerPool, WorkerSlot
from qorl.measure.run import TaskRun
from qorl.measure.schemas import Candidate, RunStatus
from qorl.plans.verify import contains_node
from qorl.sft.assemble import action_families, canonical_sha256
from qorl.sft.filter import rejection_reason
from qorl.sft.sample import PlanValidationEvaluator, selected_tasks
from qorl.sft.schemas import (
    JSON_OBJECT_ADAPTER,
    ActionFamily,
    DatasetConfig,
    DatasetSelection,
    FileIdentity,
    FilterRecord,
    JsonObject,
    SampleRecord,
    TeacherAttempt,
    TeacherAttemptStatus,
    TeacherConfig,
    TeacherGenerationRecord,
    TeacherIdentity,
    TeacherManifest,
    TeacherPrefix,
    TeacherRecordIdentity,
    TeacherSummary,
    load_json_lines,
    load_record,
    require_list,
    require_object,
    require_string,
)
from qorl.util.hashing import sha256_file
from qorl.util.io import utc_now, write_json
from qorl.workload.taskset import TaskSet
from qorl.workload.timeouts import CalibratedTimeouts

GENERATION_ID = "qorl-protocol-sft-v2-teacher-v1"
TEACHER_SAMPLE_NUMBER = 5
INITIAL_MESSAGE_COUNT = 2
DEFAULT_TEACHER_OUTPUT = Path("experiments/005-protocol-sft-v2/teacher")
TARGET_FAMILY_ORDER = (
    ActionFamily.PARALLEL,
    ActionFamily.LEADING,
    ActionFamily.JOIN,
)
DECISION_TOOLS = {
    ToolName.EVALUATE_CANDIDATE.value,
    ToolName.KEEP_DEFAULT.value,
    ToolName.FINISH.value,
}
FAMILY_INSTRUCTIONS = {
    ActionFamily.LEADING: (
        "Propose a physically novel join order. The leading tree must include every "
        "query alias exactly once, and each internal join must combine two subtrees "
        "connected by a query join predicate."
    ),
    ActionFamily.JOIN: (
        "Propose a physically novel join-method constraint. A joins[].relations "
        "target binds only to a plan node whose complete leaf-alias set exactly "
        "equals that target. Without Leading, choose a set that is an internal node "
        "in the default plan; with Leading, create that exact node in the tree."
    ),
    ActionFamily.PARALLEL: (
        "Propose a physically novel structured parallel action for a relation beneath "
        "the default Gather node. Request one or two workers so the action differs "
        "from PostgreSQL's default physical plan."
    ),
}


@dataclass(frozen=True)
class PrefixCandidate:
    path: Path
    sample: SampleRecord
    messages: list[JsonObject]
    assistant_turns: int


@dataclass(frozen=True)
class TeacherTask:
    task: JsonObject
    prefix: PrefixCandidate


@dataclass(frozen=True)
class TeacherDecision:
    action: JsonObject
    assistant_message: JsonObject


@dataclass(frozen=True)
class AttemptResult:
    attempts: list[TeacherAttempt]
    accepted: SampleRecord | None


class ScriptedModelClient:
    def __init__(self, messages: list[JsonObject]) -> None:
        self.messages = messages
        self.index = 0

    def models(self) -> JsonObject:
        raise RuntimeError("scripted policy has no model catalog")

    def version(self) -> JsonObject:
        raise RuntimeError("scripted policy has no server version")

    def chat(self, body: JsonObject) -> JsonObject:
        del body
        if self.index >= len(self.messages):
            raise RuntimeError("scripted policy exhausted its responses")
        message = self.messages[self.index]
        self.index += 1
        return JSON_OBJECT_ADAPTER.validate_python(
            {
                "choices": [{"message": message}],
                "usage": {
                    "prompt_tokens": 0,
                    "completion_tokens": 0,
                    "total_tokens": 0,
                },
            }
        )


def assistant_tool(call_id: str, name: ToolName, arguments: JsonObject) -> JsonObject:
    return JSON_OBJECT_ADAPTER.validate_python(
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": call_id,
                    "type": "function",
                    "function": {
                        "name": name.value,
                        "arguments": json.dumps(arguments, sort_keys=True),
                    },
                }
            ],
        }
    )


def assistant_tool_name(message: JsonObject) -> str | None:
    if message.get("role") != "assistant":
        return None
    calls = message.get("tool_calls")
    if not isinstance(calls, list) or len(calls) != 1:
        return None
    call = require_object(calls[0], "assistant.tool_calls[0]")
    function = require_object(call.get("function"), "assistant.tool_call.function")
    name = function.get("name")
    return name if isinstance(name, str) else None


def inspection_prefix(sample: SampleRecord) -> list[JsonObject] | None:
    trace = sample.policy_trace
    if trace is None:
        return None
    messages = [
        require_object(value, "policy_trace.transcript[]")
        for value in require_list(trace.get("transcript"), "policy_trace.transcript")
    ]
    if len(messages) < INITIAL_MESSAGE_COUNT:
        return None
    for index, message in enumerate(messages[INITIAL_MESSAGE_COUNT:], start=2):
        if message.get("role") != "assistant":
            continue
        name = assistant_tool_name(message)
        if name is None:
            return None
        if name in DECISION_TOOLS:
            prefix = messages[:index]
            assistant_turns = sum(item.get("role") == "assistant" for item in prefix)
            if len(prefix) != INITIAL_MESSAGE_COUNT + assistant_turns * 2:
                return None
            return prefix if assistant_turns in {1, 2} else None
    return None


def choose_prefix(paths: list[Path]) -> PrefixCandidate | None:
    candidates: list[PrefixCandidate] = []
    for path in paths:
        sample = load_record(path, SampleRecord)
        messages = inspection_prefix(sample)
        if messages is None:
            continue
        candidates.append(
            PrefixCandidate(
                path=path,
                sample=sample,
                messages=messages,
                assistant_turns=sum(
                    message.get("role") == "assistant" for message in messages
                ),
            )
        )
    if not candidates:
        return None
    return min(
        candidates,
        key=lambda item: item.sample.sample,
    )


def teacher_identity(
    repository: Path, config_path: Path, config: TeacherConfig
) -> TeacherIdentity:
    return TeacherIdentity(
        teacher_id=config.teacher_id,
        model=config.model,
        base_url=config.base_url,
        decoding=config.decoding,
        config=FileIdentity(
            path=config_path.relative_to(repository).as_posix(),
            sha256=sha256_file(config_path),
        ),
    )


def task_rank(config: TeacherConfig, family: ActionFamily, task: TeacherTask) -> str:
    task_id = require_string(task.task.get("task_id"), "task.task_id")
    return hashlib.sha256(
        f"{config.teacher_id}:{family.value}:{task_id}".encode()
    ).hexdigest()


def eligible_for_family(family: ActionFamily, task: TeacherTask) -> bool:
    default = task.prefix.sample.default
    if default is None:
        return False
    if family == ActionFamily.PARALLEL:
        return any(
            contains_node(default.compact_plan, node_type)
            for node_type in ("Gather", "Gather Merge")
        )
    return True


def ordered_tasks(
    config: TeacherConfig,
    family: ActionFamily,
    tasks: list[TeacherTask],
    used_task_ids: set[str],
) -> list[TeacherTask]:
    priority = {
        template_id: index
        for index, template_id in enumerate(config.priority_templates)
    }
    return sorted(
        (
            task
            for task in tasks
            if require_string(task.task.get("task_id"), "task.task_id")
            not in used_task_ids
            and eligible_for_family(family, task)
        ),
        key=lambda task: (
            priority.get(
                require_string(task.task.get("template_id"), "task.template_id"),
                len(priority),
            ),
            task_rank(config, family, task),
        ),
    )


def prefix_tools(prefix: PrefixCandidate) -> list[JsonObject]:
    trace = prefix.sample.policy_trace
    if trace is None:
        raise RuntimeError("teacher prefix has no policy trace")
    tools = [
        require_object(value, "policy_trace.tools[]")
        for value in require_list(trace.get("tools"), "policy_trace.tools")
    ]
    for tool in tools:
        function = require_object(tool.get("function"), "tool.function")
        if function.get("name") == ToolName.EVALUATE_CANDIDATE.value:
            return tools
    raise RuntimeError("teacher tools do not include evaluate_candidate")


def generation_prompt(family: ActionFamily, feedback: str | None) -> str:
    parts = [
        "Call evaluate_candidate exactly once now; do not emit prose or call another tool.",
        "Pass its action argument as a JSON object, never as text or an array.",
        FAMILY_INSTRUCTIONS[family],
        "Ground the proposal in the observed plan and statistics. For a new join "
        "order, start from the most selective filtered relation. Keep PostgreSQL's "
        "default join methods unless row estimates justify changing one, and never "
        "force a nested loop over a large unfiltered relation.",
        "Use only relations, indexes, join edges, and planner settings present in the task observation.",
    ]
    if feedback is not None:
        parts.append(f"The previous action was rejected: {feedback}")
    return "\n".join(parts)


def teacher_request(
    config: TeacherConfig,
    family: ActionFamily,
    prefix: PrefixCandidate,
    feedback: str | None,
) -> JsonObject:
    messages = [
        *prefix.messages,
        {"role": "user", "content": generation_prompt(family, feedback)},
    ]
    return JSON_OBJECT_ADAPTER.validate_python(
        {
            "model": config.model,
            "messages": messages,
            "tools": prefix_tools(prefix),
            "tool_choice": "auto",
            "max_tokens": config.max_tokens,
        }
    )


def response_decision(response: JsonObject) -> TeacherDecision:
    choices = require_list(response.get("choices"), "response.choices")
    if len(choices) != 1:
        raise ValueError("teacher response must contain exactly one choice")
    choice = require_object(choices[0], "response.choices[0]")
    message = require_object(choice.get("message"), "response.message")
    if message.get("role") != "assistant":
        raise ValueError("teacher response message must have the assistant role")
    raw_calls = message.get("tool_calls")
    if not isinstance(raw_calls, list) or len(raw_calls) != 1:
        raise ValueError("teacher response must contain exactly one tool call")
    call = require_object(raw_calls[0], "response.tool_calls[0]")
    function = require_object(call.get("function"), "response.tool_call.function")
    if function.get("name") != ToolName.EVALUATE_CANDIDATE.value:
        raise ValueError("teacher must call evaluate_candidate")
    arguments = function.get("arguments")
    if not isinstance(arguments, str):
        raise ValueError("teacher tool arguments must be a JSON string")
    try:
        parsed = json.loads(arguments)
    except json.JSONDecodeError as error:
        raise ValueError("teacher tool arguments are not valid JSON") from error
    values = JSON_OBJECT_ADAPTER.validate_python(parsed)
    return TeacherDecision(
        action=require_object(values.get("action"), "evaluate_candidate.action"),
        assistant_message=JSON_OBJECT_ADAPTER.validate_python(
            {"role": "assistant", "content": "", "tool_calls": [call]}
        ),
    )


def scripted_messages(
    prefix: PrefixCandidate, decision: TeacherDecision
) -> list[JsonObject]:
    inspections = [
        message
        for message in prefix.messages[INITIAL_MESSAGE_COUNT:]
        if message.get("role") == "assistant"
    ]
    return [
        *inspections,
        decision.assistant_message,
        assistant_tool("teacher-finish", ToolName.FINISH, {}),
    ]


def replay_action(
    slot: WorkerSlot,
    task_set: TaskSet,
    task: JsonObject,
    prefix: PrefixCandidate,
    decision: TeacherDecision,
    policy_config: QoAgentConfig,
    timeouts: CalibratedTimeouts,
) -> SampleRecord:
    task_id = require_string(task.get("task_id"), "task.task_id")
    evaluator = PlanValidationEvaluator(
        slot.worker,
        task_set,
        task,
        timeouts.task(task_id),
    )
    baseline = evaluator.start()
    client = ScriptedModelClient(scripted_messages(prefix, decision))
    trace = JSON_OBJECT_ADAPTER.validate_python(
        QoAgentPolicy(
            replace(policy_config, seed=prefix.sample.seed), client=client
        ).search(evaluator)
    )
    transcript = [
        require_object(value, "policy_trace.transcript[]")
        for value in require_list(trace.get("transcript"), "policy_trace.transcript")
    ]
    if transcript[: len(prefix.messages)] != prefix.messages:
        raise RuntimeError("scripted replay changed the student inspection prefix")
    return SampleRecord(
        status=RunStatus.COMPLETED,
        completed_at_utc=utc_now(),
        task_id=task_id,
        template_id=require_string(task.get("template_id"), "task.template_id"),
        sample=TEACHER_SAMPLE_NUMBER,
        seed=prefix.sample.seed,
        sampling_mode=prefix.sample.sampling_mode,
        steered=False,
        guidance=None,
        worker=JSON_OBJECT_ADAPTER.validate_python(slot.resources.manifest()),
        data_identity=JSON_OBJECT_ADAPTER.validate_python(
            slot.worker.fixture.data_identity
        ),
        runtime_identity=JSON_OBJECT_ADAPTER.validate_python(
            slot.worker.fixture.runtime_identity
        ),
        sampler=prefix.sample.sampler,
        default=baseline,
        candidates=evaluator.candidates,
        policy_trace=trace,
        training_transcript=transcript,
        error=None,
    )


def candidate_feedback(candidate: Candidate, reason: str) -> str:
    if candidate.errors_or_diagnostics:
        return "; ".join(candidate.errors_or_diagnostics)
    if candidate.duplicate_of is not None:
        return "the action reproduced PostgreSQL's default physical plan"
    return reason.replace("_", " ")


def generate_attempts(
    family: ActionFamily,
    config: TeacherConfig,
    prefix: PrefixCandidate,
    client: ModelClient,
    replay: Callable[[TeacherDecision], SampleRecord],
    context_length: int,
    maximum_attempts: int,
    pause: Callable[[float], None] = time.sleep,
) -> AttemptResult:
    attempts: list[TeacherAttempt] = []
    feedback: str | None = None
    tools_sha256 = canonical_sha256(prefix_tools(prefix))
    for attempt_number in range(1, maximum_attempts + 1):
        prompt = generation_prompt(family, feedback)
        request = teacher_request(config, family, prefix, feedback)
        try:
            response = JSON_OBJECT_ADAPTER.validate_python(client.chat(request))
        except ModelRequestError:
            raise
        except ModelError as error:
            attempts.append(
                TeacherAttempt(
                    attempt=attempt_number,
                    completed_at_utc=utc_now(),
                    status=TeacherAttemptStatus.PROVIDER_ERROR,
                    prompt=prompt,
                    tool_schema_sha256=tools_sha256,
                    response=None,
                    action=None,
                    candidate=None,
                    rejection_reason=str(error),
                )
            )
            if attempt_number < maximum_attempts:
                pause(config.provider_retry_delay_seconds)
            continue
        try:
            decision = response_decision(response)
        except (RuntimeError, ValueError, TypeError) as error:
            feedback = str(error)
            attempts.append(
                TeacherAttempt(
                    attempt=attempt_number,
                    completed_at_utc=utc_now(),
                    status=TeacherAttemptStatus.RESPONSE_ERROR,
                    prompt=prompt,
                    tool_schema_sha256=tools_sha256,
                    response=response,
                    action=None,
                    candidate=None,
                    rejection_reason=str(error),
                )
            )
            continue
        try:
            sample = replay(decision)
        except (RuntimeError, ValueError, TypeError) as error:
            attempts.append(
                TeacherAttempt(
                    attempt=attempt_number,
                    completed_at_utc=utc_now(),
                    status=TeacherAttemptStatus.REPLAY_ERROR,
                    prompt=prompt,
                    tool_schema_sha256=tools_sha256,
                    response=response,
                    action=decision.action,
                    candidate=None,
                    rejection_reason=str(error),
                )
            )
            break
        candidate = sample.candidates[0] if len(sample.candidates) == 1 else None
        reason = rejection_reason(sample, context_length)
        if reason is None and candidate is not None:
            families = {
                ActionFamily(value)
                for value in action_families(
                    JSON_OBJECT_ADAPTER.validate_python(candidate.action)
                )
            }
            if family not in families:
                reason = "requested_family_missing"
        status = (
            TeacherAttemptStatus.ACCEPTED
            if reason is None
            else TeacherAttemptStatus.VALIDATION_ERROR
        )
        attempts.append(
            TeacherAttempt(
                attempt=attempt_number,
                completed_at_utc=utc_now(),
                status=status,
                prompt=prompt,
                tool_schema_sha256=tools_sha256,
                response=response,
                action=decision.action,
                candidate=candidate,
                rejection_reason=reason,
            )
        )
        if reason is None:
            return AttemptResult(attempts, sample)
        if candidate is not None:
            feedback = candidate_feedback(candidate, reason)
    return AttemptResult(attempts, None)


def generate_task(
    pool: WorkerPool,
    task_set: TaskSet,
    task: TeacherTask,
    family: ActionFamily,
    config: TeacherConfig,
    identity: TeacherIdentity,
    policy_config: QoAgentConfig,
    timeouts: CalibratedTimeouts,
    client: ModelClient,
    output_root: Path,
    maximum_attempts: int,
) -> TeacherGenerationRecord:
    with pool.claim_worker() as slot:

        def replay(decision: TeacherDecision) -> SampleRecord:
            return replay_action(
                slot,
                task_set,
                task.task,
                task.prefix,
                decision,
                policy_config,
                timeouts,
            )

        result = generate_attempts(
            family,
            config,
            task.prefix,
            client,
            replay,
            policy_config.context_length,
            maximum_attempts,
        )
    return TeacherGenerationRecord(
        task_id=require_string(task.task.get("task_id"), "task.task_id"),
        template_id=require_string(task.task.get("template_id"), "task.template_id"),
        requested_family=family,
        teacher=identity,
        prefix=TeacherPrefix(
            sample_path=task.prefix.path.relative_to(output_root).as_posix(),
            sample_sha256=sha256_file(task.prefix.path),
            sample=task.prefix.sample.sample,
            assistant_turns=task.prefix.assistant_turns,
        ),
        attempts=result.attempts,
        accepted_sample=result.accepted,
    )


def record_path(output: Path, record: TeacherGenerationRecord) -> Path:
    return output / "records" / record.requested_family.value / f"{record.task_id}.json"


def load_generation_records(output: Path) -> list[tuple[Path, TeacherGenerationRecord]]:
    return [
        (path, load_record(path, TeacherGenerationRecord))
        for path in sorted((output / "records").glob("*/*.json"))
    ]


def summarize(records: list[TeacherGenerationRecord]) -> TeacherSummary:
    accepted = [record for record in records if record.accepted_sample is not None]
    reasons = Counter(
        attempt.rejection_reason or attempt.status.value
        for record in records
        for attempt in record.attempts
        if attempt.status != TeacherAttemptStatus.ACCEPTED
    )
    return TeacherSummary(
        tasks_attempted=len(records),
        api_attempts=sum(len(record.attempts) for record in records),
        accepted=len(accepted),
        accepted_by_family=dict(
            sorted(Counter(record.requested_family for record in accepted).items())
        ),
        rejected_by_reason=dict(sorted(reasons.items())),
    )


def generation_targets(
    config: TeacherConfig,
    *,
    smoke: bool,
    requested: tuple[ActionFamily, ...] | None = None,
) -> dict[ActionFamily, int]:
    if smoke:
        if requested is not None:
            raise ValueError("--families cannot be combined with --smoke")
        return dict.fromkeys(TARGET_FAMILY_ORDER, config.smoke_accepted_per_family)
    families = requested or TARGET_FAMILY_ORDER
    if len(families) != len(set(families)):
        raise ValueError("--families contains duplicates")
    configured = config.accepted_targets.as_families()
    return {family: configured[family] for family in families}


def record_identities(
    repository: Path, records: list[tuple[Path, TeacherGenerationRecord]]
) -> list[TeacherRecordIdentity]:
    return [
        TeacherRecordIdentity(
            task_id=record.task_id,
            template_id=record.template_id,
            requested_family=record.requested_family,
            path=path.relative_to(repository).as_posix(),
            sha256=sha256_file(path),
            accepted=record.accepted_sample is not None,
        )
        for path, record in records
    ]


def verify_records(repository: Path, output: Path) -> TeacherManifest:
    manifest = load_record(output / "manifest.json", TeacherManifest)
    paths = {
        path.relative_to(repository).as_posix(): path
        for path, _ in load_generation_records(output)
    }
    if set(paths) != {record.path for record in manifest.records}:
        raise RuntimeError("teacher manifest record inventory differs")
    for record in manifest.records:
        if sha256_file(paths[record.path]) != record.sha256:
            raise RuntimeError(f"teacher record checksum differs: {record.path}")
    actual = summarize([record for _, record in load_generation_records(output)])
    if actual != manifest.summary:
        raise RuntimeError("teacher manifest summary differs")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate validated Fable continuations for protocol SFT v2."
    )
    parser.add_argument("--repository", type=Path, default=Path.cwd())
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("experiments/005-protocol-sft-v2/dataset.json"),
    )
    parser.add_argument(
        "--teacher-config",
        type=Path,
        default=Path("experiments/005-protocol-sft-v2/teacher.json"),
    )
    parser.add_argument(
        "--timeouts",
        type=Path,
        default=Path("experiments/005-protocol-sft-v2/timeouts.json"),
    )
    parser.add_argument(
        "--dataset", type=Path, default=Path("outputs/sft/protocol-sft-v2")
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_TEACHER_OUTPUT,
    )
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument(
        "--families",
        nargs="+",
        choices=tuple(family.value for family in TARGET_FAMILY_ORDER),
        help="Override the teacher families.",
    )
    parser.add_argument("--check", action="store_true")
    arguments = parser.parse_args()

    repository = arguments.repository.resolve()
    output = (repository / arguments.output).resolve()
    requested = (
        tuple(ActionFamily(value) for value in arguments.families)
        if arguments.families is not None
        else None
    )
    if arguments.check:
        manifest = verify_records(repository, output)
        print(json.dumps(manifest.summary.to_wire(), indent=2, sort_keys=True))
        return

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise RuntimeError("ANTHROPIC_API_KEY is required")
    dataset = (repository / arguments.dataset).resolve()
    config_path = (repository / arguments.config).resolve()
    teacher_config_path = (repository / arguments.teacher_config).resolve()
    timeout_path = (repository / arguments.timeouts).resolve()
    config = load_record(config_path, DatasetConfig)
    teacher_config = load_record(teacher_config_path, TeacherConfig)
    identity = teacher_identity(repository, teacher_config_path, teacher_config)
    selection = load_record(repository / config.selection, DatasetSelection)
    policy_document = JSON_OBJECT_ADAPTER.validate_json(
        (repository / config.policy_config).read_text(encoding="utf-8")
    )
    policy_config = QoAgentConfig.from_dict(
        require_object(policy_document.get("policy"), "policy")
    )
    source_filter_path = dataset / "filter/sampling/records.jsonl"
    source_filter = FileIdentity(
        path=source_filter_path.relative_to(repository).as_posix(),
        sha256=sha256_file(source_filter_path),
    )
    student_records = load_json_lines(source_filter_path, FilterRecord)
    accepted_task_ids = {
        record.task_id for record in student_records if record.accepted
    }

    fixture = DatabaseFixture.load(repository)
    task_set = TaskSet.load(repository, "ceb")
    tasks = selected_tasks(task_set, selection, "sampling")
    paths_by_task: dict[str, list[Path]] = {}
    for path in sorted((dataset / "samples/sampling").glob("*/sample-*.json")):
        paths_by_task.setdefault(path.parent.name, []).append(path)
    teacher_tasks: list[TeacherTask] = []
    for task in tasks:
        task_id = require_string(task.get("task_id"), "task.task_id")
        if task_id in accepted_task_ids:
            continue
        prefix = choose_prefix(paths_by_task.get(task_id, []))
        if prefix is not None:
            teacher_tasks.append(TeacherTask(task=task, prefix=prefix))

    output.mkdir(parents=True, exist_ok=True)
    existing_pairs = load_generation_records(output)
    existing = [record for _, record in existing_pairs]
    if any(record.teacher != identity for record in existing):
        raise RuntimeError("teacher records belong to a different teacher config")
    targets = generation_targets(
        teacher_config,
        smoke=arguments.smoke,
        requested=requested,
    )
    started_at = datetime.now(UTC).isoformat()
    previous_manifest_path = output / "manifest.json"
    if previous_manifest_path.is_file():
        previous = load_record(previous_manifest_path, TeacherManifest)
        if previous.teacher != identity or previous.source_filter != source_filter:
            raise RuntimeError("teacher output belongs to different experiment inputs")
        started_at = previous.started_at_utc
    initial_manifest = TeacherManifest(
        generation_id=GENERATION_ID,
        status=RunStatus.RUNNING,
        started_at_utc=started_at,
        completed_at_utc=None,
        teacher=identity,
        source_filter=source_filter,
        targets=targets,
        database_pool=None,
        summary=summarize(existing),
        records=record_identities(repository, existing_pairs),
    )
    manifest_wire = initial_manifest.to_wire()
    write_json(previous_manifest_path, manifest_wire)
    timeouts = CalibratedTimeouts.load(
        repository, timeout_path, task_set, fixture.runtime_identity
    )
    client = OpenAIModelClient(
        teacher_config.base_url,
        teacher_config.request_timeout_seconds,
        api_key=api_key,
    )
    run = TaskRun(
        fixture,
        f"qorl-sft-v2-teacher-{os.getpid()}",
        dataset / "teacher-runtime",
        previous_manifest_path,
        manifest_wire,
        pool_field="database_pool",
        environment_dir=dataset / "teacher-runtime/environment",
    )
    used_task_ids = {record.task_id for record in existing}
    accepted_counts = Counter(
        record.requested_family
        for record in existing
        if record.accepted_sample is not None
    )
    attempt_counts = Counter(
        record.requested_family for record in existing for _ in record.attempts
    )
    with run:
        if run.pool is None:
            raise RuntimeError("teacher database pool did not start")
        for family in targets:
            budget = targets[family] * teacher_config.attempt_budget_multiplier
            for task in ordered_tasks(
                teacher_config, family, teacher_tasks, used_task_ids
            ):
                if accepted_counts[family] >= targets[family]:
                    break
                if attempt_counts[family] >= budget:
                    break
                result = generate_task(
                    run.pool,
                    task_set,
                    task,
                    family,
                    teacher_config,
                    identity,
                    policy_config,
                    timeouts,
                    client,
                    dataset,
                    min(
                        teacher_config.maximum_attempts_per_task,
                        budget - attempt_counts[family],
                    ),
                )
                used_task_ids.add(result.task_id)
                attempt_counts[family] += len(result.attempts)
                if result.accepted_sample is not None:
                    accepted_counts[family] += 1
                path = record_path(output, result)
                write_json(path, result.to_wire())
                pairs = load_generation_records(output)
                records = [record for _, record in pairs]
                manifest_wire["summary"] = summarize(records).to_wire()
                manifest_wire["records"] = [
                    record.to_wire() for record in record_identities(repository, pairs)
                ]
                run.write()
                print(
                    f"[{family.value}] {result.task_id}: "
                    f"{'accepted' if result.accepted_sample is not None else 'rejected'}",
                    flush=True,
                )

    pairs = load_generation_records(output)
    records = [record for _, record in pairs]
    summary = summarize(records)
    complete = all(
        summary.accepted_by_family.get(family, 0) >= count
        for family, count in targets.items()
    )
    final = initial_manifest.model_copy(
        update={
            "status": RunStatus.COMPLETED
            if complete
            else RunStatus.COMPLETED_WITH_FAILURES,
            "completed_at_utc": utc_now(),
            "database_pool": require_object(
                manifest_wire.get("database_pool"), "database pool"
            ),
            "summary": summary,
            "records": record_identities(repository, pairs),
        }
    )
    write_json(previous_manifest_path, final.to_wire())
    print(json.dumps(final.summary.to_wire(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
