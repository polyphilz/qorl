from __future__ import annotations

import json
from pathlib import Path

from qorl.plans.schemas import (
    BOOLEAN_SETTINGS,
    INTEGER_SETTINGS,
    MAX_PARALLEL_WORKERS,
    NUMERIC_SETTINGS,
)


class TestDockerContract:
    def test_fixture_entrypoints_expose_src_and_repository_to_python(
        self, repository_root: Path
    ) -> None:
        expected = (
            'export PYTHONPATH="$repository_root/src:$repository_root'
            '${PYTHONPATH:+:$PYTHONPATH}"'
        )
        for name in (
            "build-job-v1.sh",
            "load-job-v1.sh",
            "restore-verify-job-v1.sh",
        ):
            script = (repository_root / "scripts/job" / name).read_text(
                encoding="utf-8"
            )
            assert expected in script

    def test_contract_defines_every_prompt_visible_planner_setting(
        self, repository_root: Path
    ) -> None:
        contract = json.loads(
            (
                repository_root / "docker/postgres/contract/benchmark.expected.json"
            ).read_text(encoding="utf-8")
        )
        names = set(BOOLEAN_SETTINGS) | set(INTEGER_SETTINGS) | set(NUMERIC_SETTINGS)
        values = {name: contract["settings"][name] for name in sorted(names)}

        assert set(values) == names
        assert all(isinstance(value, str) for value in values.values())

    def test_benchmark_v2_disables_geqo(self, repository_root: Path) -> None:
        contract = json.loads(
            (
                repository_root / "docker/postgres/contract/benchmark.expected.json"
            ).read_text(encoding="utf-8")
        )
        versions = json.loads(
            (repository_root / "docker/postgres/versions.json").read_text(
                encoding="utf-8"
            )
        )

        assert contract["benchmark_config_id"] == "benchmark-v2"
        assert versions["benchmark"]["config_id"] == "benchmark-v2"
        assert contract["settings"]["geqo"] == "off"
        assert (
            int(contract["settings"]["max_parallel_workers_per_gather"])
            == MAX_PARALLEL_WORKERS
        )
