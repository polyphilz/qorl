class WorkerError(RuntimeError):
    pass


class QueryTimeout(WorkerError):  # noqa: N818
    def __init__(self, timeout_ms: int) -> None:
        super().__init__(f"query exceeded statement_timeout={timeout_ms} ms")
        self.timeout_ms = timeout_ms
