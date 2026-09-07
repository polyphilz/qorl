import asyncio
import tomllib
from concurrent.futures import CancelledError
from pathlib import Path
from threading import Event
from unittest.mock import Mock

import pytest

from qorl.training.harness import QorlHarness, QorlHarnessConfig

WAIT_SECONDS = 2.0


def test_fallback_config_matches_rl_defaults(repository_root: Path) -> None:
    defaults = tomllib.loads(
        (repository_root / "configs/defaults/000-rl.toml").read_text()
    )
    expected = QorlHarnessConfig.model_validate(
        {key: defaults[key] for key in ("agent", "measurement", "rl")}
    )

    fallback = QorlHarnessConfig(id="qorl")

    assert fallback.agent == expected.agent
    assert fallback.measurement == expected.measurement
    assert fallback.rl == expected.rl


@pytest.mark.parametrize("cancellations", [1, 3])
@pytest.mark.parametrize("worker_fails", [False, True])
def test_cancellation_waits_for_worker_cleanup(
    repository_root: Path,
    monkeypatch: pytest.MonkeyPatch,
    cancellations: int,
    worker_fails: bool,
) -> None:
    defaults = tomllib.loads(
        (repository_root / "configs/defaults/000-rl.toml").read_text()
    )
    harness = QorlHarness(
        QorlHarnessConfig.model_validate(
            {key: defaults[key] for key in ("agent", "measurement", "rl")}
        )
    )
    started, cancellation_seen, release_worker, cleaned_up = (
        Event(),
        Event(),
        Event(),
        Event(),
    )

    def run(
        ctx: Mock, trace: Mock, endpoint: str, secret: str, data: Mock, cancel: Event
    ) -> None:
        started.set()
        assert cancel.wait(WAIT_SECONDS)
        cancellation_seen.set()
        assert release_worker.wait(WAIT_SECONDS)
        cleaned_up.set()
        if worker_fails:
            raise CancelledError("rollout cancelled")

    monkeypatch.setattr(harness, "_run", run)

    async def check() -> None:
        work = asyncio.create_task(
            harness.launch(Mock(), Mock(), Mock(), "unused", "unused", {}, Mock())
        )
        try:
            assert await asyncio.to_thread(started.wait, WAIT_SECONDS)
            for _ in range(cancellations):
                work.cancel()
                assert await asyncio.to_thread(cancellation_seen.wait, WAIT_SECONDS)
                # Let the cancellation handler run while the worker still owns its slot.
                await asyncio.sleep(0)
                assert not work.done()
                assert not cleaned_up.is_set()
        finally:
            release_worker.set()
            with pytest.raises(asyncio.CancelledError):
                await work
        assert cleaned_up.is_set()

    asyncio.run(check())
