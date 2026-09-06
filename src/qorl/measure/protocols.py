from typing import Protocol

from qorl.postgres.schemas import ExplainResult, PostgresIndexes
from qorl.taskset.schemas import Task


class QueryExecutor(Protocol):
    def explain(
        self,
        sql: str,
        timeout_ms: int,
        *,
        analyze: bool = False,
        hint: str = "",
    ) -> ExplainResult: ...

    @property
    def indexes(self) -> PostgresIndexes: ...


class SqlSource(Protocol):
    def load_sql(self, task: Task) -> str: ...
