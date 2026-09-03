from typing import Any, Protocol

from qorl.db.worker import ExplainResult


class QueryExecutor(Protocol):
    def explain(
        self,
        sql: str,
        timeout_ms: int,
        *,
        analyze: bool = False,
        hint: str = "",
    ) -> ExplainResult: ...

    def task_indexes(self, task: dict[str, Any]) -> dict[str, set[str]]: ...


class SqlSource(Protocol):
    def load_sql(self, task: dict[str, Any]) -> str: ...
