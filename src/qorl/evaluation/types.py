from enum import StrEnum


class RunStatus(StrEnum):
    PENDING = "pending"
    STARTING = "starting"
    RUNNING = "running"
    COMPLETED = "completed"
    COMPLETED_WITH_FAILURES = "completed_with_failures"
    INTERRUPTED = "interrupted"
    FAILED = "failed"
    PASSED = "passed"


class PolicyType(StrEnum):
    RANDOM_STRUCTURED_ACTION = "random_structured_action"
    QO_AGENT = "qo_agent"
